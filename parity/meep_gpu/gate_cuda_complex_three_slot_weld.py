"""The COMPLEX no-absorber THREE-SLOT weld's device gate: byte identity per complete
driver step, the launch structure, the seam order, and the refusals that license it.

=============================================================================
WHAT IS BEING CERTIFIED, AND WHY NEITHER HALF'S VERDICT COVERS IT
=============================================================================

``cuda_three_slot_complex_no_pml_dispersive_weld`` owns ``step_D``, ``update_E`` and
``update_P`` and performs all three in ONE launch per E component (three launches per
step, host rotation between), carrying this component's flux density from the curl
to the constitutive in a register and the constitutive product from ``update_E`` to
every ADE recurrence in a register. Its two halves are separately RELEASED --
``cuda_complex_no_pml_2026-08-27`` (the singles) and
``cuda_complex_no_pml_fused_polarization_pair_2026-09-02b`` (the E->P weld) -- and
its D->E half (``no_pml_complex_fused_electric_pair``) is lifted but NOT certified.
None of those is a verdict about a launch that welds a curl block ahead of an ADE
chain, and none measures the one ordering claim this product makes: that with no
deposit inside the span, advancing P in the same launch as the constitutive is the
array path's arithmetic word for word.

=============================================================================
THE COMPARISONS -- FOUR ENGINES FROM ONE SEED
=============================================================================

Every product case compares per COMPLETE DRIVER STEP as uint32 words over every
stored volume AND every ``P`` / ``P_prev`` / ``_scratch`` buffer BY SLOT, against:

  1. the ARRAY PATH (``stepping``'s own passes);
  2. the CERTIFIED SINGLES -- ``step_D_no_pml_complex[_conductive]``,
     ``update_E_no_pml_complex_stored``, ``update_P_no_pml_complex[_uniform]`` per
     (state, component) with the certified rotation;
  3. the certified ``step_D`` + the E->P WELD -- what the composer installed on
     these rows until this product landed;
  4. the D->E WELD + the certified ADE singles -- the other two-slot arrangement.

All four must agree with the weld word for word. What differs is the LAUNCH COUNT,
measured by two instruments on every launch_structure fixture: the compile-memo
proxy (per kernel) and CUDA graph capture of the whole step (every kernel, CuPy's
included), borrowed from the real three-slot gate rather than re-spelled.

=============================================================================
THE SEAM ORDER, MEASURED FROM BOTH SIDES
=============================================================================

``p_at_update_P_consult`` steps the weld with the curl+constitutive half at the
``step_D`` consult and the polarization advance MOVED to the ``update_P`` consult
(the certified ADE singles). On an admitted row NOTHING sits between the two, so this
must be BYTE-IDENTICAL -- it is the measurement that the single launch is licensed.
``electric_deposit_inside_the_span`` is the converse and the refusal's consequence:
the predicate refuses an electric in-seam source by name, and the leg injects one
anyway on the array path and on the weld's composition; it MUST DIVERGE, because a
launch that already advanced E and P cannot see a deposit made after it.

=============================================================================
THE MUTATIONS
=============================================================================

Device text, through the launcher's ``kernel`` door: the two seam NULLS (D reloaded
from global instead of the register; E reloaded from global instead of the register
-- a float32 word pair stored and reloaded is the identity on the bits, so both MUST
be inert), the wrong drive (D fed to the recurrence instead of E: MUST diverge), the
store landing on P^n, a bank lane swapped, the conductive tail reassociated, the
mask dropped. Host: an unconjugated Bloch table, the plain tail on a conductive run,
and the three rotation defects of the certified launcher's loop beside a local copy
with no defect (required inert).

Progress is one flushed line per case and the artifact is rewritten after every leg
(the progress-reporting rule). Correctness only; nothing here is timed.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_API = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_API not in sys.path:
    sys.path.insert(0, _REPO_API)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    import cupy as cp
except ImportError:  # laptop: only the source legs can run
    cp = None

import gate_provenance  # noqa: E402
import probe_fused_kernel_bit_identity as probe  # noqa: E402
import gate_cuda_complex_polarization_pair as ep_gate  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.cuda_kernels import complex_no_pml_fused_polarization_pair as ep  # noqa: E402
from meep_gpu.cuda_kernels import complex_no_pml_kernels as certified  # noqa: E402
from meep_gpu.cuda_kernels import complex_no_pml_three_slot_dispersive_weld as weld  # noqa: E402
from meep_gpu.cuda_kernels import no_pml_complex_fused_electric_pair as de  # noqa: E402
from meep_gpu.cuda_kernels import fused_pairs  # noqa: E402

log = ep_gate.log
to_host = probe.to_host
operand_census = probe.operand_census
words_differ = ep_gate.words_differ
compared_hosts = ep_gate.compared_hosts
compare_all = ep_gate.compare_all
COMPONENTS = ep_gate.COMPONENTS
VALUE_CLASSES = ep_gate.VALUE_CLASSES

SEED = 20260904

#: Complete DRIVER STEPS every product leg runs -- the hand-CUDA record's budget.
STEPS = 60

#: The driver's pass order (driver.py FdtdDriver.step), the shape every engine walks.
DRIVER_ORDER: Tuple[str, ...] = (
    "step_B", "fill_symmetry_bc_B", "zero_metal_B", "fill_folded_far_ghosts_B",
    "update_H",
    "step_D", "fill_symmetry_bc_D", "zero_metal_D", "fill_folded_far_ghosts_D",
    "update_E", "update_P")

_TAKES_PML = {"step_B", "update_H", "step_D", "update_E", "update_P"}


def run_pass(name: str, fields, pml) -> None:
    function = getattr(stepping, name)
    if name in _TAKES_PML:
        function(fields, pml)
    else:
        function(fields)


def array_step(fields) -> None:
    for name in DRIVER_ORDER:
        run_pass(name, fields, None)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

#: THE CORPUS CELL FIRST: TestLoadDump.*_3d lifts to grid (35, 32, 41), complex64,
#: Bloch k = (0.4, -1.3, 0.7), periodic on all three axes, a D AND a B conductivity,
#: one Lorentzian driving every component, a magnetic source only. Then the axes the
#: corpus does not vary: the pole count and the sigma form (the mixed volume/uniform
#: signature is a compile-time axis), an unphased and a lossless row, a component
#: with NO driving state (its launch is the curl and the constitutive alone), and a
#: 2-D reduction. NO WALLED ROW: the product refuses a wall by name, and the refusal
#: leg exercises that rather than the sweep.
SPECS: Tuple[Dict[str, Any], ...] = (
    {"label": "corpus_load_dump_3d", "cell": (35.0, 32.0, 41.0), "dimensions": 3,
     "boundaries": ("periodic",) * 3, "k_point": (0.4, -1.3, 0.7),
     "conductivity": "volume", "kinds": ("volume",)},
    {"label": "3d_phased_conductive_two_mixed", "cell": (5.0, 6.0, 7.0),
     "dimensions": 3, "boundaries": ("periodic",) * 3, "k_point": (0.2, -0.35, 0.1),
     "conductivity": "volume", "kinds": ("volume", "uniform")},
    {"label": "3d_unphased_lossless_two_volume", "cell": (5.0, 6.0, 7.0),
     "dimensions": 3, "boundaries": ("periodic",) * 3, "k_point": (0.0, 0.0, 0.0),
     "conductivity": "none", "kinds": ("volume", "volume")},
    {"label": "3d_phased_lossless_one_uniform", "cell": (6.0, 5.0, 7.0),
     "dimensions": 3, "boundaries": ("periodic",) * 3, "k_point": (0.25, 0.0, -0.15),
     "conductivity": "none", "kinds": ("uniform",)},
    {"label": "3d_conductive_ey_undriven", "cell": (5.0, 6.0, 7.0),
     "dimensions": 3, "boundaries": ("periodic",) * 3, "k_point": (0.2, -0.35, 0.1),
     "conductivity": "volume", "kinds": ("volume",), "undriven": ("Ey",)},
    {"label": "2d_phased_conductive_one_volume", "cell": (9.0, 11.0, 0.0),
     "dimensions": 2, "boundaries": ("periodic",) * 3, "k_point": (0.2, 0.4, 0.0),
     "conductivity": "volume", "kinds": ("volume",)},
)

REDUCED_LABELS: Tuple[str, ...] = ("corpus_load_dump_3d",
                                  "3d_phased_conductive_two_mixed")
LAUNCH_STRUCTURE_LABELS: Tuple[str, ...] = ("corpus_load_dump_3d",
                                           "3d_phased_conductive_two_mixed",
                                           "3d_conductive_ey_undriven")
MUTATION_LABELS: Tuple[str, ...] = ("corpus_load_dump_3d",
                                   "3d_phased_conductive_two_mixed",
                                   "3d_unphased_lossless_two_volume")


def case_seed(label: str) -> int:
    return SEED + int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], "big")


def build(spec: Dict[str, Any], value_class: str, seed: int):
    """The E->P gate's fixture builder, plus this gate's one axis: an UNDRIVEN
    component (a sigma of exactly zero there, which ``PolarizationState`` reads as
    ``trivial_sigma`` and allocates no P for)."""
    from meep_gpu.dispersion import PolarizationState, Susceptibility  # noqa: PLC0415

    undriven = tuple(spec.get("undriven", ()))
    if not undriven:
        return ep_gate.build(spec, value_class, seed)
    base = dict(spec)
    base["kinds"] = ()
    fields, grid, seeded = ep_gate.build(base, value_class, seed)
    rng = np.random.default_rng(seed + 7)
    shape = tuple(grid.shape)
    for index, kind in enumerate(spec["kinds"]):
        sigma: Dict[str, Any] = {}
        for component in COMPONENTS:
            if component in undriven:
                sigma[component] = 0.0
            elif kind == "volume":
                sigma[component] = cp.asarray(np.ascontiguousarray(
                    rng.uniform(0.1, 0.9, size=shape).astype(np.float32)))
            else:
                sigma[component] = float(0.2 + 0.13 * index)
        fields.polarizations.append(PolarizationState(
            Susceptibility(frequency=1.05 + 0.4 * index, gamma=0.05 + 0.02 * index),
            sigma, grid, fields._field_dtype()))  # noqa: SLF001
    for index, state in enumerate(fields.polarizations):
        for component in state.driven():
            for slot, name in ((state.P, "P"), (state.P_prev, "P_prev")):
                host = ep_gate.sibling._complex_host(shape, value_class, rng)  # noqa: SLF001
                seeded[f"{name}[{index}][{component}]"] = host.view(np.float32).ravel().copy()
                slot[component][...] = cp.asarray(host)
        host = ep_gate.sibling._complex_host(shape, value_class, rng)  # noqa: SLF001
        seeded[f"_scratch[{index}]"] = host.view(np.float32).ravel().copy()
        state._scratch[...] = cp.asarray(host)  # noqa: SLF001
    return fields, grid, seeded


# ---------------------------------------------------------------------------
# The five compositions
# ---------------------------------------------------------------------------

class _CountingKernels:
    """Count launches at the weld's ``_get_kernel`` seam, wrapping the real binary."""

    def __init__(self, module) -> None:
        self.module = module
        self.count = 0
        self._original = module._get_kernel  # noqa: SLF001

    def __enter__(self):
        def wrapped(*args, **kwargs):
            kernel = self._original(*args, **kwargs)

            def launch(*launch_args, **launch_kwargs):
                self.count += 1
                return kernel(*launch_args, **launch_kwargs)
            return launch
        self.module._get_kernel = wrapped  # noqa: SLF001
        return self

    def __exit__(self, *exc):
        self.module._get_kernel = self._original  # noqa: SLF001
        return False


def without_the_recurrences(_arm, _conductive, _component, _count, _kinds,
                            code: str) -> str:
    """The emitted kernel with its ADE lines CUT and NOTHING ELSE TOUCHED.

    THE POLE BANK AND ``np`` SURVIVE THE CUT, which is the whole point: the
    specialization, the signature and the bank binding are the shipped ones, so the
    constitutive still subtracts every pole through ``minus_poles_reg``. What moves
    is only WHERE the recurrence runs. Cutting instead at the ``component_specs``
    seam -- telling the launcher a component has no driving state -- would not move
    the recurrence at all: it would drop ``np`` to 0 and bind the bank to spares,
    which changes the CONSTITUTIVE'S ARITHMETIC and measures nothing about the seam
    order. A component with no driving state carries no pole marker and is returned
    unchanged.
    """
    marker = "\n    // pole 0:"
    if marker not in code:
        return code
    return code[:code.index(marker)] + "}\n"


def curl_and_constitutive_only(fields, grid, arm) -> Dict[str, Any]:
    """The weld's three launches with the recurrences cut out of the device text.

    THE HOST ROTATION IS NOT PERFORMED EITHER, and driving the per-component launches
    here rather than through :func:`run_three_slot_no_pml_complex` is what leaves it
    out -- the certified ``update_P`` at the trailing consult performs its own. Every
    other argument is the shipped launcher's, taken from the shipped defaults.
    """
    from meep_gpu.cuda_kernels.complex_pml_kernels import (  # noqa: PLC0415
        bloch_phase_arguments, complex_boundary_codes)

    curl_arm = de.no_pml_complex_curl_arm(fields, "step_D")
    codes = complex_boundary_codes(grid)
    flags, values = bloch_phase_arguments(grid, True)
    dtdx = float(grid.dt / grid.dx)
    previous = weld.SOURCE_TRANSFORM
    weld.SOURCE_TRANSFORM = without_the_recurrences
    launches = 0
    try:
        for target, _displacement, _stem, _count in weld._COMPONENTS:  # noqa: SLF001
            states = weld.component_specs(fields)[target]["states"]
            weld.launch_three_slot_component(
                fields, arm, curl_arm, target, states, codes, flags, values, dtdx)
            launches += 1
    finally:
        weld.SOURCE_TRANSFORM = previous
    return {"launched": True, "launches": launches, "curl_arm": curl_arm,
            "recurrences_in_the_launch": 0}


def weld_step(fields, grid, arm, kernel=None, curl_arm=None,
              polarization: Optional[Callable[..., Any]] = None,
              phase_flags=None, phase_values=None) -> Dict[str, Any]:
    """ONE complete driver step with the weld at the ``step_D`` consult.

    Every pass the product declares in ``REPLACES`` is skipped because the launches
    perform it. ``polarization`` is the gate's door into the seam order: when given,
    the weld runs WITHOUT its recurrences -- cut from the DEVICE TEXT by
    :func:`without_the_recurrences`, with the pole bank and ``np`` left bound -- and
    the E->P slot is answered by the callable at the ``update_P`` consult instead,
    which is the ``p_at_update_P_consult`` leg.
    """
    report: Dict[str, Any] = {}
    for name in DRIVER_ORDER:
        if name == "step_D":
            report = (curl_and_constitutive_only(fields, grid, arm)
                      if polarization is not None else
                      weld.run_three_slot_no_pml_complex(
                          fields, grid, arm, curl_arm=curl_arm, kernel=kernel,
                          phase_flags=phase_flags, phase_values=phase_values))
        elif name == "update_P" and polarization is not None:
            polarization(fields)
        elif name not in weld.REPLACES:
            run_pass(name, fields, None)
    return report


def certified_singles_step(fields, grid, arm) -> Dict[str, Any]:
    """The composition with no weld: the three certified singles."""
    launches = 0
    for name in DRIVER_ORDER:
        if name == "step_D":
            certified.step_complex_no_pml_curl(fields, "step_D", arm, grid=grid)
            launches += 1
        elif name == "update_E":
            certified.update_E_complex_no_pml_stored(fields, arm)
            launches += 1
        elif name == "update_P":
            launches += int(certified.update_P_complex_no_pml(fields, arm))
        else:
            run_pass(name, fields, None)
    return {"launches": launches}


def step_d_plus_e_to_p_weld(fields, grid, arm) -> Dict[str, Any]:
    """What the composer installed on these rows until this product landed."""
    launches = 0
    for name in DRIVER_ORDER:
        if name == "step_D":
            certified.step_complex_no_pml_curl(fields, "step_D", arm, grid=grid)
            launches += 1
        elif name == "update_E":
            launches += int(ep.run_complex_no_pml_fused_polarization_pair(
                fields, None, arm)["launches"])
        elif name == "update_P":
            continue
        else:
            run_pass(name, fields, None)
    return {"launches": launches}


def d_to_e_weld_plus_ade(fields, grid, arm) -> Dict[str, Any]:
    """The other two-slot arrangement: the D->E weld, then the certified ADE."""
    launches = 0
    for name in DRIVER_ORDER:
        if name == "step_D":
            report = de.run_no_pml_complex_fused_electric_pair(
                fields, grid, None, float(grid.dt / grid.dx), arm, sources=(),
                license=LICENCE["verdict"], subnormal_policy=POLICY["name"])
            if not report.get("launched"):
                raise RuntimeError(f"the D->E weld refused: {report.get('reason')}")
            launches += 1
        elif name == "update_E":
            continue
        elif name == "update_P":
            launches += int(certified.update_P_complex_no_pml(fields, arm))
        else:
            run_pass(name, fields, None)
    return {"launches": launches}


# ---------------------------------------------------------------------------
# Legs
# ---------------------------------------------------------------------------

def leg_driver_order() -> Dict[str, Any]:
    """``REPLACES`` is an ordered subsequence of the driver's pass list starting at
    ``step_D`` and ending at ``update_P``, and every pass inside the run that is NOT
    replaced is inert on an admitted row (the fills need a fold, the wall clear a
    wall; the predicate refuses both)."""
    replaces = list(weld.REPLACES)
    positions = [DRIVER_ORDER.index(name) for name in replaces]
    span = DRIVER_ORDER[positions[0]:positions[-1] + 1]
    inert = [name for name in span if name not in replaces]
    return {"passed": bool(positions == sorted(positions)
                           and replaces[0] == "step_D" and replaces[-1] == "update_P"
                           and set(inert) <= {"fill_symmetry_bc_D", "zero_metal_D",
                                              "fill_folded_far_ghosts_D"}),
            "replaces": replaces, "driver_order": list(DRIVER_ORDER),
            "passes_inside_the_run_left_on_the_array_path": inert,
            "why_those_are_inert": (
                "the predicate refuses a mirror plane and a metallic wall by name, "
                "so both fills and the wall clear return at their first line")}


def leg_lift(arm) -> Dict[str, Any]:
    """The emitted text against the certified bodies, re-asked on the device host."""
    sources = weld.device_sources()
    checks: List[Dict[str, Any]] = []
    for conductive in (False, True):
        for component in range(3):
            header, block = weld.certified_curl_block(arm, conductive, component)
            body = de.certified_curl_body(arm, conductive)
            source = weld.three_slot_no_pml_complex_source(arm, conductive, component,
                                                           2, (True, False))
            twin = ep.fused_polarization_pair_no_pml_complex_source(arm, component, 2,
                                                                   (True, False))
            checks.append({
                "conductive": conductive, "component": component,
                "block_is_in_the_certified_body": block in body,
                "header_is_in_the_certified_body": header in body,
                "one_block_per_launch": source.count("\n    // Target ") == 1,
                "constitutive_value_named": f"    cf_store(h{component}, idx, ev);\n"
                                            in source,
                "poles_are_the_e_to_p_welds": (
                    source.split("// pole 0")[1] == twin.split("// pole 0")[1]),
                "drive_is_the_register": "cf_load(drive, idx)" not in source,
            })
    # THE IDENTITY KEYS ARE NOT CHECKS. ``conductive`` is False on the plain arm and
    # ``component`` is 0 on the first one; scoring every bool in the row made the
    # plain arm's three rows fail whatever they measured, which is a leg that can
    # never pass. The checks are named, so they are read by name.
    named = ("block_is_in_the_certified_body", "header_is_in_the_certified_body",
             "one_block_per_launch", "constitutive_value_named",
             "poles_are_the_e_to_p_welds", "drive_is_the_register")
    passed = bool(sources) and all(all(c[k] for k in named) for c in checks)
    return {"passed": passed, "specializations": len(sources), "checks": checks,
            "checks_scored": list(named),
            "lift_edits": [row["line"] for row in weld.LIFT_EDITS]}


def leg_refusal(arm) -> Dict[str, Any]:
    """What the weld refuses BY NAME, the admitted control, and who takes the slots."""
    from meep_gpu.cuda_kernels import arms  # noqa: PLC0415
    from meep_gpu.cuda_kernels.registry import (LICENSE_COMPLEX,  # noqa: PLC0415
                                                LICENSE_COMPLEX_NO_PML)
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415

    checks: List[Dict[str, Any]] = []

    def record(name, expectation, fields, pml, grid, sources=()):
        covered, reason = weld.covers_three_slot_complex_no_pml_dispersive_weld(
            fields, pml, grid, sources, LICENCE["verdict"], POLICY["name"])
        checks.append({"case": name, "expected": expectation,
                       "covered": bool(covered), "reason": str(reason),
                       "passed": (not covered) if expectation == "refused"
                       else bool(covered)})

    live, live_grid, _ = build(SPECS[0], "uniform", case_seed("refusal"))
    record("the_corpus_fixture", "covered", live, None, live_grid, ())
    record("undeclared_source_set", "refused", live, None, live_grid, None)

    class _Electric:
        field_type = "D"

    class _Magnetic:
        field_type = "B"
    record("electric_in_seam_source", "refused", live, None, live_grid, (_Electric(),))
    record("magnetic_source_one_seam_earlier", "covered", live, None, live_grid,
           (_Magnetic(),))

    none = dict(SPECS[0], kinds=())
    bare, bare_grid, _ = build(none, "uniform", case_seed("refusal|none"))
    record("no_polarization_to_advance", "refused", bare, None, bare_grid, ())

    real_grid = Grid(resolution=1.0, cell_size=(5.0, 6.0, 7.0),
                     boundaries=("periodic",) * 3, k_point=(0.0, 0.0, 0.0), xp=cp,
                     courant=0.5)
    real = Fields(grid=real_grid)
    real.enable_field_storage()
    record("real_storage", "refused", real, None, real_grid, ())

    walled = dict(SPECS[1], boundaries=("metallic", "periodic", "periodic"),
                  k_point=(0.0, 0.0, 0.0))
    wall, wall_grid, _ = build(walled, "uniform", case_seed("refusal|wall"))
    record("metallic_wall_inside_the_span", "refused", wall, None, wall_grid, ())

    layered_grid = Grid(resolution=1.0, cell_size=(9.0, 10.0, 11.0),
                        boundaries=("periodic",) * 3, xp=cp, courant=0.5)
    layered = Fields(grid=layered_grid, force_complex_fields=True)
    layered.enable_field_storage()
    layered.enable_pml_storage()
    record("active_absorber", "refused", layered,
           PML(grid=layered_grid, thickness=2), layered_grid, ())

    # THE ARBITRATION, THROUGH THE SHIPPED COMPOSER.
    plan = arms.plan_step(live, None, live_grid,
                          licenses={LICENSE_COMPLEX_NO_PML: LICENCE["verdict"],
                                    LICENSE_COMPLEX: LICENCE["verdict"]},
                          subnormal_policy=POLICY["name"], sources=(), fuse=True)
    arbitration = {
        "selected": {slot: plan.selected.get(slot)
                     for slot in ("step_D", "update_E", "update_P")},
        "d_to_e_refusal": list(plan.reasons.get(
            "fused_pair_cuda_no_pml_complex_fused_electric_pair", ())),
        "e_to_p_refusal": list(plan.reasons.get(
            "fused_pair_cuda_complex_no_pml_fused_polarization_pair", ())),
        "installed": [type(plan.plans.get(slot)).__name__
                      for slot in ("step_D", "update_E", "update_P")],
    }
    arbitration["ok"] = (
        len(set(arbitration["selected"].values())) == 1
        and None not in arbitration["selected"].values()
        and any("strictly contains" in r for r in arbitration["d_to_e_refusal"])
        and bool(arbitration["e_to_p_refusal"])
        and arbitration["installed"] == ["_TripleHalfPlan", "NoopPlan",
                                         "_TripleHalfPlan"])
    _ = fused_pairs
    passed = all(c["passed"] for c in checks) and arbitration["ok"]
    return {"passed": passed, "checks": checks, "arbitration": arbitration,
            "admitted_control_present": any(
                c["expected"] == "covered" and c["passed"] for c in checks)}


def run_case(spec: Dict[str, Any], value_class: str, steps: int, arm,
             kernel=None, curl_arm=None, polarization=None, phase_override=None,
             label_suffix: str = "", references: Sequence[str] = ("array",)
             ) -> Dict[str, Any]:
    """One fixture, ``steps`` COMPLETE driver steps, the weld beside its references."""
    started = time.time()
    label = f"{spec['label']}/{value_class}{label_suffix}"
    seed = case_seed(label)
    reference, ref_grid, seeded = build(spec, value_class, seed)
    actual, grid, _ = build(spec, value_class, seed)
    others: Dict[str, Tuple[Any, Any]] = {}
    for name in references:
        if name != "array":
            others[name] = build(spec, value_class, seed)[:2]
    out: Dict[str, Any] = {"label": label, "value_class": value_class,
                           "steps_requested": steps, "spec": spec["label"],
                           "shape": [int(n) for n in grid.shape],
                           "operand_census": operand_census(seeded),
                           "references": list(references)}
    covered, reason = weld.covers_three_slot_complex_no_pml_dispersive_weld(
        actual, None, grid, (), LICENCE["verdict"], POLICY["name"])
    out["predicate"] = {"covered": bool(covered), "reason": str(reason)}
    if not covered:
        out.update({"passed": False,
                    "why": f"the predicate refused the sweep fixture: {reason}"})
        return out
    out["driven_components"] = [c for c in COMPONENTS if any(
        s.drives(c) for s in actual.polarizations)]
    phase_flags = phase_values = None
    if phase_override is not None:
        phase_flags, phase_values = phase_override(grid)

    initial = compared_hosts(reference)
    weld_launches: List[int] = []
    counted: List[int] = []
    diffs: Dict[str, List[int]] = {name: [] for name in references}
    for _step in range(1, steps + 1):
        array_step(reference)
        for name, (engine, engine_grid) in others.items():
            if name == "certified_singles":
                certified_singles_step(engine, engine_grid, arm)
            elif name == "step_d_plus_e_to_p_weld":
                step_d_plus_e_to_p_weld(engine, engine_grid, arm)
            elif name == "d_to_e_weld_plus_ade":
                d_to_e_weld_plus_ade(engine, engine_grid, arm)
        with _CountingKernels(weld) as counter:
            report = weld_step(actual, grid, arm, kernel=kernel, curl_arm=curl_arm,
                               polarization=polarization, phase_flags=phase_flags,
                               phase_values=phase_values)
        weld_launches.append(int(report.get("launches", -1)))
        counted.append(counter.count)
        cp.cuda.runtime.deviceSynchronize()
        diffs["array"].append(sum(compare_all(reference, actual).values()))
        for name, (engine, _g) in others.items():
            diffs[name].append(sum(compare_all(engine, actual).values()))
        if any(rows[-1] for rows in diffs.values()):
            break

    final = compared_hosts(actual)
    moved = {name: words_differ(initial[name], final[name]) for name in initial}
    must_move = list(COMPONENTS) + ["Dx", "Dy", "Dz"] + [
        name for name in initial if name.startswith("P[")]
    if value_class != "uniform":
        # The E->P gate's own floor for the band class: under the flush policy a
        # subnormal D flushed to zero may round back to the same word, and E is the
        # volume update_E rewrites unconditionally.
        must_move = list(COMPONENTS)
    frozen = [name for name in must_move if moved.get(name, 0) == 0]
    steps_run = len(diffs["array"])
    expected_memo = 3 * steps_run if kernel is None else 0
    out.update({
        "steps_run": steps_run,
        "bit_identical": {name: not any(rows) for name, rows in diffs.items()},
        "first_divergence": {name: next((i + 1 for i, n in enumerate(rows) if n),
                                        None) for name, rows in diffs.items()},
        "differing_words": {name: (rows[-1] if rows else -1)
                            for name, rows in diffs.items()},
        "weld_launches_per_step_reported": sorted(set(weld_launches)),
        "weld_launches_per_step_counted": sorted(set(counted)),
        "moved_words": int(sum(moved.values())),
        "arrays_compared": len(initial),
        "pole_arrays_compared": sum(1 for n in initial if n.startswith(("P[", "P_prev[", "_scratch["))),
        "movement_floor_frozen": frozen,
        "launch_counts_agree": (sorted(set(weld_launches)) == [3]
                                and sum(counted) == expected_memo),
        "seconds": time.time() - started,
    })
    out["passed"] = bool(steps_run == steps and all(out["bit_identical"].values())
                         and not frozen and out["launch_counts_agree"])
    return out


def leg_launch_structure(spec: Dict[str, Any], steps: int, arm) -> Dict[str, Any]:
    """FIVE compositions from one seed, all identical, counted by two instruments."""
    from gate_cuda_three_slot_weld import GraphNodeCounter  # noqa: PLC0415

    seed = case_seed(f"launch|{spec['label']}")
    engines = {name: build(spec, "uniform", seed)[:2] for name in (
        "array_path", "certified_singles", "step_d_plus_e_to_p_weld",
        "d_to_e_weld_plus_ade", "three_slot_weld")}
    steppers = {
        "array_path": lambda f, g: array_step(f),
        "certified_singles": lambda f, g: certified_singles_step(f, g, arm),
        "step_d_plus_e_to_p_weld": lambda f, g: step_d_plus_e_to_p_weld(f, g, arm),
        "d_to_e_weld_plus_ade": lambda f, g: d_to_e_weld_plus_ade(f, g, arm),
        "three_slot_weld": lambda f, g: weld_step(f, g, arm),
    }
    # WARM every composition on a throwaway engine first: capture forbids allocation.
    for name, stepper in steppers.items():
        warm = build(spec, "uniform", seed)[:2]
        stepper(*warm)
    cp.cuda.runtime.deviceSynchronize()
    graph = GraphNodeCounter()
    counts = {name: graph.count(lambda s=stepper, e=engines[name]: s(*e))
              for name, stepper in steppers.items()}
    per_step: List[Dict[str, Any]] = []
    # TWO INSTRUMENTS, EACH READ FOR WHAT IT CAN SEE. The graph counter spans the
    # WHOLE driver step and counts every kernel, CuPy's included, so it is the only
    # one that can say the weld reduces launches against the array path -- and it can
    # never say the weld launched three, because a whole step is far more than the
    # span. The memo counter wraps the weld's own ``_get_kernel`` and is blind to
    # everything else, which is exactly what "this product launches three times per
    # step" is a claim about. Asserting the weld's own count off the graph number was
    # a leg that could not pass whatever the product did.
    with _CountingKernels(weld) as counter:
        for step in range(1, steps + 1):
            for name, stepper in steppers.items():
                stepper(*engines[name])
            cp.cuda.runtime.deviceSynchronize()
            row = {"step": step}
            for name in steppers:
                if name != "three_slot_weld":
                    row[name] = sum(compare_all(engines[name][0],
                                                engines["three_slot_weld"][0]).values())
            per_step.append(row)
    identical = all(v == 0 for row in per_step for k, v in row.items() if k != "step")
    captured = all(c.get("captured") for c in counts.values())
    launches = {name: c.get("kernel_launches") for name, c in counts.items()}
    reduces = [name for name in launches
               if launches[name] is not None and launches["three_slot_weld"] is not None
               and launches[name] > launches["three_slot_weld"]]
    memo_per_step = counter.count / float(steps)
    return {"label": spec["label"], "steps": steps,
            "graph_counter": counts, "cuda_graph_kernel_launches_whole_step": launches,
            "weld_launches_per_step_memo": memo_per_step,
            "per_step": per_step, "all_compositions_identical": identical,
            "weld_reduces_launches_against": reduces,
            "passed": bool(identical and captured and memo_per_step == 3.0
                           and "array_path" in reduces
                           and "step_d_plus_e_to_p_weld" in reduces
                           and "certified_singles" in reduces),
            "reading": (f"whole-step kernel launches: {launches}; the weld itself "
                        f"launches {memo_per_step} per step at its own memo seam and "
                        f"reduces the whole step against {reduces}")}


def leg_seam_order(spec: Dict[str, Any], steps: int, arm) -> Dict[str, Any]:
    """The seam order measured from BOTH sides (module docstring)."""
    rows: List[Dict[str, Any]] = []

    # (a) the polarization advance MOVED to the update_P consult: MUST be identical.
    #     The recurrence lines are cut from the DEVICE TEXT (the bank and ``np`` stay
    #     bound, so the constitutive still subtracts every pole) and the host rotation
    #     is not performed; the certified ADE singles then advance P and rotate at the
    #     update_P consult. Cutting at the ``component_specs`` seam instead would drop
    #     the poles out of the constitutive and measure nothing about the seam order.
    case = run_case(spec, "uniform", steps, arm,
                    polarization=lambda f: certified.update_P_complex_no_pml(f, arm),
                    label_suffix="|p_at_update_P_consult")
    rows.append({"mode": "p_at_update_P_consult", "must_diverge": False,
                 "bit_identical": case["bit_identical"]["array"],
                 "first_divergence": case["first_divergence"]["array"],
                 "passed": bool(case["bit_identical"]["array"]
                                and case["steps_run"] == steps)})

    # (b) an electric deposit INSIDE the span: the predicate refuses it, and the
    #     consequence of ignoring the refusal MUST be a divergence.
    seed = case_seed(f"deposit|{spec['label']}")
    reference, ref_grid, _ = build(spec, "uniform", seed)
    actual, grid, _ = build(spec, "uniform", seed)
    index = tuple(int(n) // 2 for n in grid.shape)
    amount = np.complex64(0.25 - 0.125j)

    def inject(fields):
        fields.Dz[index] = fields.Dz[index] + amount

    class _Electric:
        field_type = "D"
    refused = not weld.covers_three_slot_complex_no_pml_dispersive_weld(
        actual, None, grid, (_Electric(),), LICENCE["verdict"], POLICY["name"])[0]
    diverged_at = None
    for step in range(1, min(steps, 12) + 1):
        for name in DRIVER_ORDER:
            run_pass(name, reference, None)
            if name == "step_D":
                inject(reference)
        for name in DRIVER_ORDER:
            if name == "step_D":
                weld.run_three_slot_no_pml_complex(actual, grid, arm)
                inject(actual)  # the driver's injection point, now AFTER the launch
            elif name not in weld.REPLACES:
                run_pass(name, actual, None)
        cp.cuda.runtime.deviceSynchronize()
        if sum(compare_all(reference, actual).values()):
            diverged_at = step
            break
    rows.append({"mode": "electric_deposit_inside_the_span", "must_diverge": True,
                 "refused_by_the_predicate": refused,
                 "bit_identical": diverged_at is None,
                 "first_divergence": diverged_at,
                 "passed": bool(refused and diverged_at is not None)})
    return {"label": spec["label"], "rows": rows,
            "passed": all(r["passed"] for r in rows),
            "reading": ("the polarization advance may sit in the launch or at the "
                        "update_P consult with identical bytes, and a deposit inside "
                        "the span diverges -- which is why it is refused")}


#: Device mutations. ``{c}`` is the component index and ``{s}`` its bank stem.
DEVICE_MUTATIONS: Tuple[Dict[str, Any], ...] = (
    {"name": "constitutive_reloads_d_from_global",
     "old": "minus_poles_reg(d{c}, ", "new": "minus_poles_reg(cf_load(f{c}, idx), ",
     "expect": "inert",
     "why": "THE FIRST SEAM NULL: D read back from global instead of the register "
            "the curl stored. A float32 word pair stored and reloaded is the "
            "identity on the bits."},
    {"name": "recurrence_reloads_e_from_global",
     "old": "cf_load(p_prev_0, idx), ev,", "new": "cf_load(p_prev_0, idx), cf_load(h{c}, idx),",
     "expect": "inert",
     "why": "THE SECOND SEAM NULL: pole 0's drive becomes a reload of the E word "
            "the constitutive just stored."},
    {"name": "wrong_drive_displacement",
     "old": "cf_load(p_prev_0, idx), ev,", "new": "cf_load(p_prev_0, idx), d{c},",
     "expect": "caught",
     "why": "pole 0 driven by the DISPLACEMENT register instead of E: the certified "
            "ADE gate's wrong-drive control against the fused text."},
    {"name": "store_lands_on_p_now",
     "old": "cf_store(p_out_0, idx, ade_step(",
     "new": "cf_store((float*){s}0, idx, ade_step(",
     "expect": "caught",
     "why": "pole 0's store lands on P^n instead of the scratch."},
    {"name": "bank_lane_swapped",
     "old": "cf_load({s}1, idx), cf_load(p_prev_1, idx)",
     "new": "cf_load({s}0, idx), cf_load(p_prev_1, idx)",
     "expect": "caught", "min_poles": 2,
     "why": "pole 1's recurrence reads pole 0's P^n."},
    {"name": "conductive_tail_reassociated",
     "old": "    cf t = mul_field_left(cf_load(f, idx), condfac);\n    t = cf_sub(t, curl);\n",
     "new": "    cf t = cf_sub(cf_load(f, idx), curl);\n    t = mul_field_left(t, condfac);\n",
     "expect": "caught", "conductive_only": True,
     "why": "((f - curl) * condfac) * condinv instead of the certified in-place "
            "order: the same three operations, a different rounding."},
    {"name": "ownership_mask_dropped",
     "old": {0: "if (bc_y == BC_METALLIC && j == 0) curl = cf_zero();",
             1: "if (bc_x == BC_METALLIC && i == 0) curl = cf_zero();",
             2: "if (bc_x == BC_METALLIC && i == 0) curl = cf_zero();"},
     "new": "",
     "expect": "inert",
     "why": "the mask is dead on every admitted row (no wall is admitted), so "
            "dropping it must move nothing -- the refusal's other half."},
)


def leg_mutations(spec: Dict[str, Any], steps: int, arm) -> List[Dict[str, Any]]:
    legs: List[Dict[str, Any]] = []
    probe_fields, probe_grid, _ = build(spec, "uniform", case_seed("mut|" + spec["label"]))
    curl_arm = de.no_pml_complex_curl_arm(probe_fields, "step_D")
    conductive = curl_arm == "conductive"
    kinds = tuple(kind == "volume" for kind in spec["kinds"])
    for mutation in DEVICE_MUTATIONS:
        if len(kinds) < int(mutation.get("min_poles", 0)):
            continue
        if mutation.get("conductive_only") and not conductive:
            continue
        compiled: Dict[Any, Any] = {}
        missed = False
        for component in range(3):
            states = [s for s in probe_fields.polarizations if s.drives(COMPONENTS[component])]
            count = len(states)
            kinds_c = tuple(bool(weld.ade_sigma_is_volume(s, COMPONENTS[component]))
                            for s in states)
            stem = weld._COMPONENTS[component][2]  # noqa: SLF001
            needle = mutation["old"]
            if isinstance(needle, dict):
                needle = needle[component]
            old = needle.format(c=component, s=stem)
            new = mutation["new"].format(c=component, s=stem)
            source = weld.three_slot_no_pml_complex_source(arm, conductive, component,
                                                           count, kinds_c)
            hits = source.count(old)
            if hits == 0 and not (mutation["name"] in ("bank_lane_swapped",
                                                       "store_lands_on_p_now",
                                                       "recurrence_reloads_e_from_global",
                                                       "wrong_drive_displacement")
                                  and count == 0):
                missed = True
                break
            if hits == 0:
                compiled[(component, count, kinds_c)] = None
                continue
            compiled[(component, count, kinds_c)] = cp.RawKernel(
                source.replace(old, new), weld.kernel_name(curl_arm),
                options=("--fmad=false",))
        if missed:
            legs.append({"leg": mutation["name"], "label": spec["label"],
                         "scored": False, "passed": False,
                         "why_not_scored": f"the needle {mutation['old']!r} matched nothing"})
            log(f"    mutation {mutation['name']} [{spec['label']}]: NEEDLE MISSED")
            continue
        original = weld._get_kernel  # noqa: SLF001

        def patched(arm_value, conductive_value, component_index, count, kinds_value,
                    _table=compiled):
            kernel = _table.get((component_index, count, tuple(kinds_value)))
            return kernel if kernel is not None else original(
                arm_value, conductive_value, component_index, count, kinds_value)

        weld._get_kernel = patched  # noqa: SLF001
        try:
            case = run_case(spec, "uniform", steps, arm,
                            label_suffix=f"|{mutation['name']}")
        finally:
            weld._get_kernel = original  # noqa: SLF001
        diverged = not case.get("bit_identical", {}).get("array", False)
        passed = diverged if mutation["expect"] == "caught" else (
            not diverged and case.get("steps_run") == steps)
        legs.append({"leg": mutation["name"], "label": spec["label"], "scored": True,
                     "expect": mutation["expect"], "diverged": bool(diverged),
                     "passed": bool(passed),
                     "first_divergence": case.get("first_divergence", {}).get("array"),
                     "why": mutation["why"]})
        log(f"    mutation {mutation['name']} [{spec['label']}]: "
            f"{'PASS' if passed else 'FAIL'} (diverged={diverged})")
    return legs


def _phase_forward(grid):
    from meep_gpu.cuda_kernels.complex_pml_kernels import bloch_phase_arguments  # noqa: PLC0415

    return bloch_phase_arguments(grid, False)


def leg_host_mutations(spec: Dict[str, Any], steps: int, arm) -> List[Dict[str, Any]]:
    legs: List[Dict[str, Any]] = []
    fields, _grid, _ = build(spec, "uniform", case_seed("host|" + spec["label"]))
    conductive = de.no_pml_complex_curl_arm(fields, "step_D") == "conductive"
    phased = any(float(k) != 0.0 for k in spec["k_point"])
    rows = [("unconjugated_phase_table", dict(phase_override=_phase_forward),
             "caught" if phased else "inert")]
    if conductive:
        rows.append(("plain_tail_on_a_conductive_run", dict(curl_arm="plain"), "caught"))
    for name, kwargs, expect in rows:
        case = run_case(spec, "uniform", steps, arm, label_suffix=f"|{name}", **kwargs)
        diverged = not case.get("bit_identical", {}).get("array", False)
        passed = diverged if expect == "caught" else (
            not diverged and case.get("steps_run") == steps)
        legs.append({"leg": name, "label": spec["label"], "kind": "host",
                     "expect": expect, "diverged": bool(diverged), "passed": bool(passed),
                     "first_divergence": case.get("first_divergence", {}).get("array")})
        log(f"    host mutation {name} [{spec['label']}]: {'PASS' if passed else 'FAIL'}")
    return legs


def _loop_copy(defect: Optional[str]):
    """A LOCAL copy of the shipped component loop with one named rotation defect."""

    def run(fields, grid, arm):
        from meep_gpu.cuda_kernels.complex_pml_kernels import (  # noqa: PLC0415
            bloch_phase_arguments, complex_boundary_codes)

        curl_arm = de.no_pml_complex_curl_arm(fields, "step_D")
        codes = complex_boundary_codes(grid)
        flags, values = bloch_phase_arguments(grid, True)
        dtdx = float(grid.dt / grid.dx)
        launches = 0
        pending: List[Tuple[str, Any]] = []
        for order, (target, _d, _s, _c) in enumerate(weld._COMPONENTS):  # noqa: SLF001
            states = weld.component_specs(fields)[target]["states"]
            if defect == "rotate_before_launch":
                for state in states:
                    p, p_prev, scratch = state.P[target], state.P_prev[target], state._scratch  # noqa: SLF001
                    state.P[target], state.P_prev[target], state._scratch = scratch, p, p_prev  # noqa: SLF001
            weld.launch_three_slot_component(fields, arm, curl_arm, target, states,
                                             codes, flags, values, dtdx)
            launches += 1
            if defect == "rotate_before_launch":
                continue
            if defect == "rotate_after_all_launches":
                pending.extend((target, state) for state in states)
                continue
            for index, state in enumerate(states):
                if defect == "freeze_one_states_rotation" and order == 0 and index == 0:
                    continue
                p, p_prev, scratch = state.P[target], state.P_prev[target], state._scratch  # noqa: SLF001
                state.P[target], state.P_prev[target], state._scratch = scratch, p, p_prev  # noqa: SLF001
        for target, state in pending:
            p, p_prev, scratch = state.P[target], state.P_prev[target], state._scratch  # noqa: SLF001
            state.P[target], state.P_prev[target], state._scratch = scratch, p, p_prev  # noqa: SLF001
        return {"launched": True, "launches": launches}

    return run


ROTATION_LEGS: Tuple[Tuple[str, Optional[str], str], ...] = (
    ("launcher_copy_NULL", None, "inert"),
    ("rotate_before_launch", "rotate_before_launch", "caught"),
    ("rotate_after_all_launches", "rotate_after_all_launches", "caught"),
    ("freeze_one_states_rotation", "freeze_one_states_rotation", "caught"),
)


def leg_rotation(spec: Dict[str, Any], steps: int, arm) -> List[Dict[str, Any]]:
    legs: List[Dict[str, Any]] = []
    original = weld.run_three_slot_no_pml_complex
    for name, defect, expect in ROTATION_LEGS:
        copy = _loop_copy(defect)

        def substitute(fields, grid, expansion, _copy=copy, **_kwargs):
            return _copy(fields, grid, expansion)
        weld.run_three_slot_no_pml_complex = substitute
        try:
            case = run_case(spec, "uniform", steps, arm, label_suffix=f"|{name}")
        finally:
            weld.run_three_slot_no_pml_complex = original
        diverged = not case.get("bit_identical", {}).get("array", False)
        passed = diverged if expect == "caught" else (
            not diverged and case.get("steps_run") == steps)
        legs.append({"leg": name, "label": spec["label"], "expect": expect,
                     "diverged": bool(diverged), "passed": bool(passed),
                     "first_divergence": case.get("first_divergence", {}).get("array")})
        log(f"    rotation {name}: {'PASS' if passed else 'FAIL'} (diverged={diverged})")
    return legs


# ---------------------------------------------------------------------------
# Driving
# ---------------------------------------------------------------------------

LICENCE: Dict[str, Any] = {"verdict": None}
POLICY: Dict[str, Any] = {"name": None}

#: The files this gate BINDS a verdict to.
SUBJECT_FILES: Tuple[str, ...] = (
    "meep_gpu/cuda_kernels/complex_no_pml_three_slot_dispersive_weld.py",
    "meep_gpu/cuda_kernels/no_pml_complex_fused_electric_pair.py",
    "meep_gpu/cuda_kernels/complex_no_pml_fused_polarization_pair.py",
    "meep_gpu/cuda_kernels/complex_no_pml_kernels.py",
    "meep_gpu/cuda_kernels/fused_pairs.py",
)


def _subject_digests() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for relative in SUBJECT_FILES:
        with open(os.path.join(_REPO_API, *relative.split("/")), "rb") as handle:
            out[relative] = hashlib.sha256(handle.read()).hexdigest()
    return out


def save(results: Dict[str, Any], out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    gate_provenance.stamp(results)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True, default=str)
    os.replace(tmp, out_path)


def summarize(results: Dict[str, Any], steps: int) -> Dict[str, Any]:
    cases = results.get("cases", [])
    mutations = results.get("mutations", []) + results.get("host_mutations", [])
    rotations = results.get("rotation", [])
    structure = results.get("launch_structure", [])
    order = results.get("seam_order", [])
    scored = [m for m in mutations if m.get("scored", True)]
    verdict = {
        "steps_per_case": steps,
        "cases": len(cases),
        "cases_passed": sum(1 for c in cases if c.get("passed")),
        "cases_bit_identical_to_the_array_path": sum(
            1 for c in cases if c.get("bit_identical", {}).get("array")),
        "cases_bit_identical_to_the_certified_singles": sum(
            1 for c in cases if c.get("bit_identical", {}).get("certified_singles")),
        "cases_bit_identical_to_the_e_to_p_arrangement": sum(
            1 for c in cases if c.get("bit_identical", {}).get("step_d_plus_e_to_p_weld")),
        "cases_bit_identical_to_the_d_to_e_arrangement": sum(
            1 for c in cases if c.get("bit_identical", {}).get("d_to_e_weld_plus_ade")),
        "launch_structure_passed": bool(structure) and all(s["passed"] for s in structure),
        "seam_order_passed": bool(order) and all(s["passed"] for s in order),
        "mutations_scored": len(scored),
        "mutation_legs_that_failed": [f"{m['leg']}|{m.get('label')}" for m in mutations
                                      if not m.get("passed")],
        "rotation_legs_that_failed": [r["leg"] for r in rotations if not r.get("passed")],
        "refusal_leg_passed": bool(results.get("refusal", {}).get("passed")),
        "lift_leg_passed": bool(results.get("lift", {}).get("passed")),
        "driver_order_passed": bool(results.get("driver_order", {}).get("passed")),
        "band_contains_subnormals": all(
            c["operand_census"]["subnormals"] > 0 for c in cases
            if c.get("value_class") == "subnormal_band" and "operand_census" in c),
        "uniform_contains_none": all(
            c["operand_census"]["subnormals"] == 0 for c in cases
            if c.get("value_class") == "uniform" and "operand_census" in c),
    }
    verdict["released"] = bool(
        cases and verdict["cases_passed"] == len(cases)
        and verdict["cases_bit_identical_to_the_array_path"] == len(cases)
        and verdict["cases_bit_identical_to_the_certified_singles"] == len(cases)
        and verdict["cases_bit_identical_to_the_e_to_p_arrangement"] == len(cases)
        and verdict["cases_bit_identical_to_the_d_to_e_arrangement"] == len(cases)
        and verdict["launch_structure_passed"] and verdict["seam_order_passed"]
        and scored and not verdict["mutation_legs_that_failed"]
        and rotations and not verdict["rotation_legs_that_failed"]
        and verdict["refusal_leg_passed"] and verdict["lift_leg_passed"]
        and verdict["driver_order_passed"]
        and verdict["band_contains_subnormals"] and verdict["uniform_contains_none"])
    return verdict


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--product", choices=("full", "reduced"), default="full")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--subnormal-policy", choices=("keep", "flush"), required=True)
    parser.add_argument("--import-meep-for-host-policy", action="store_true")
    parser.add_argument("--expansion-probe", default=None)
    parser.add_argument("--skip-mutations", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    started = time.time()
    results: Dict[str, Any] = {
        "gate": "cuda_complex_three_slot_weld",
        "families": [weld.FAMILY],
        "kernel": weld.KERNEL_NAME,
        "kernel_symbols": list(weld.KERNEL_KEYS.values()),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "host": __import__("socket").gethostname(),
        "argv": list(argv or sys.argv[1:]),
        "steps": args.steps,
        "subnormal_policy": args.subnormal_policy,
        "subject_sha256": _subject_digests(),
        "question": (
            "does ONE launch per E component spanning step_D -> update_E -> update_P "
            "-- the flux density and the constitutive product each handed on in a "
            "register -- leave every stored volume and every P / P_prev / scratch "
            "buffer byte-identical to the array path, to the certified singles and to "
            "both two-slot arrangements per COMPLETE driver step, on rows with a "
            "magnetic source only and no absorber?"),
        "module": {
            "family": weld.FAMILY, "kernel": weld.KERNEL_NAME,
            "replaces": list(weld.REPLACES),
            "carries_deposit_repair": bool(weld.CARRIES_DEPOSIT_REPAIR),
            "certified_kernels": list(weld.CERTIFIED_KERNELS),
            "uncertified_kernels": dict(weld.UNCERTIFIED_KERNELS),
        },
    }
    gate_provenance.stamp(results)
    POLICY["name"] = args.subnormal_policy

    if args.import_meep_for_host_policy:
        results["meep_import_for_host_policy"] = probe.import_meep_for_host_policy()
    if cp is not None:
        results["nvrtc_observer"] = probe.install_nvrtc_binary_observer()
        results["subnormal_policy_install"] = probe.install_subnormal_policy_for_run(
            args.subnormal_policy, _REPO_API)
        results["environment"] = probe.device_info()
    results["subnormal_policy_stamp"] = probe.subnormal_policy_stamp(_REPO_API)

    licence = ep_gate.sibling.load_licence(args.subnormal_policy, args.expansion_probe)
    results["licence"] = {key: licence.get(key) for key in
                          ("arm", "record", "record_sha256", "policy_reasons",
                           "refusals", "basis")}
    LICENCE["verdict"] = licence
    usable = bool(licence.get("arm") and not licence.get("refusals")
                  and not licence.get("policy_reasons"))
    arm = licence.get("arm") or "NAIVE"

    results["driver_order"] = leg_driver_order()
    log(f"driver order: {'PASS' if results['driver_order']['passed'] else 'FAIL'}")
    results["lift"] = leg_lift(arm)
    log(f"lift: {'PASS' if results['lift']['passed'] else 'FAIL'} "
        f"specializations={results['lift']['specializations']}")
    save(results, args.out)
    if cp is None:
        results["verdict"] = {"released": False,
                              "why": "CuPy is absent; only the source legs ran"}
        save(results, args.out)
        return 1
    if not usable:
        results["verdict"] = {"released": False,
                              "why": f"the expansion licence is unusable: "
                                     f"{licence.get('policy_reasons') or licence.get('refusals')}"}
        save(results, args.out)
        return 1

    results["refusal"] = leg_refusal(arm)
    log(f"refusal: {'PASS' if results['refusal']['passed'] else 'FAIL'} "
        f"arbitration={results['refusal']['arbitration']['selected']}")
    save(results, args.out)

    specs = SPECS if args.product == "full" else tuple(
        s for s in SPECS if s["label"] in REDUCED_LABELS)
    references = ("array", "certified_singles", "step_d_plus_e_to_p_weld",
                  "d_to_e_weld_plus_ade")
    cases: List[Dict[str, Any]] = []
    results["cases"] = cases
    for spec in specs:
        for value_class in VALUE_CLASSES:
            case = run_case(spec, value_class, args.steps, arm, references=references)
            cases.append(case)
            log(f"  case {case['label']}: {'PASS' if case.get('passed') else 'FAIL'} "
                f"steps={case.get('steps_run')} identical={case.get('bit_identical')} "
                f"launches={case.get('weld_launches_per_step_counted')} "
                f"({case.get('seconds', 0.0):.1f} s)")
            save(results, args.out)

    structure: List[Dict[str, Any]] = []
    results["launch_structure"] = structure
    for spec in [s for s in specs if s["label"] in LAUNCH_STRUCTURE_LABELS]:
        leg = leg_launch_structure(spec, min(args.steps, 20), arm)
        structure.append(leg)
        log(f"  launches {spec['label']}: {'PASS' if leg['passed'] else 'FAIL'} "
            f"{leg['cuda_graph_kernel_launches_whole_step']}")
        save(results, args.out)

    order: List[Dict[str, Any]] = []
    results["seam_order"] = order
    for spec in [s for s in specs if s["label"] in REDUCED_LABELS]:
        leg = leg_seam_order(spec, min(args.steps, 20), arm)
        order.append(leg)
        log(f"  seam order {spec['label']}: {'PASS' if leg['passed'] else 'FAIL'} "
            f"{[(r['mode'], r['passed']) for r in leg['rows']]}")
        save(results, args.out)

    if not args.skip_mutations:
        mutations: List[Dict[str, Any]] = []
        host: List[Dict[str, Any]] = []
        for spec in [s for s in specs if s["label"] in MUTATION_LABELS]:
            mutations.extend(leg_mutations(spec, min(args.steps, 12), arm))
            host.extend(leg_host_mutations(spec, min(args.steps, 12), arm))
            results["mutations"] = mutations
            results["host_mutations"] = host
            save(results, args.out)
        results["rotation"] = leg_rotation(SPECS[1], min(args.steps, 12), arm)
        save(results, args.out)
    else:
        results["mutations"] = []
        results["host_mutations"] = []
        results["rotation"] = []

    results["verdict"] = summarize(results, args.steps)
    results["canonical_verdict"] = {"released": results["verdict"]["released"],
                                    "reasons": [] if results["verdict"]["released"]
                                    else ["see verdict"]}
    results["passed"] = bool(results["verdict"]["released"])
    results["seconds"] = time.time() - started
    save(results, args.out)
    v = results["verdict"]
    log(f"VERDICT released={v['released']} cases={v['cases_passed']}/{v['cases']} "
        f"array={v['cases_bit_identical_to_the_array_path']} "
        f"singles={v['cases_bit_identical_to_the_certified_singles']} "
        f"e_to_p={v['cases_bit_identical_to_the_e_to_p_arrangement']} "
        f"d_to_e={v['cases_bit_identical_to_the_d_to_e_arrangement']} "
        f"mutations_failed={v['mutation_legs_that_failed']} "
        f"rotation_failed={v['rotation_legs_that_failed']}")
    return 0 if v["released"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
