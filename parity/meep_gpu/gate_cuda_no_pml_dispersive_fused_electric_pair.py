"""Byte-identity gate for the NO-ABSORBER DISPERSIVE-STORE hand-CUDA electric pair.

WHAT IS UNDER TEST. ``meep_gpu/cuda_kernels/no_pml_dispersive_fused_electric_pair.py``:
ONE kernel performing three driver passes in one launch, on a run with no absorber, a
registered susceptibility and stored E::

    step_D (driver.py:3306, with a PER-COMPONENT conductive tail)
    -> zero_metal_D (:3310)
    -> plain stored-E update_E (:3313, stepping.py:1019-1022)

THE TWO CELLS ARE THE LAST POINTWISE-BUILDABLE ``D_to_E`` CELLS THIS BOARD HAD, and
ONE product occupies both::

    (cuda_conductive/conductive,   cuda_no_pml_dispersive)  2 rows
    (cuda_no_pml_curl/no-PML curl, cuda_no_pml_dispersive)  1 row

All three rows carry an ELECTRIC DEPOSIT inside the seam, so the deposit legs are not
a side observation: without the bracket this product serves ZERO.

=============================================================================
THE TWO THINGS THIS GATE MEASURES THAT NO OTHER CUDA GATE DOES
=============================================================================

1. **THE PLAIN DEPOSIT REPAIR.** Every shipped CUDA electric weld inverts the
   SPLIT-FIELD recurrence ``field + kps*fresh - kms*fw_prev``. This cell's layer is
   INERT, so ``update_E`` is a PURE OVERWRITE with no ``f_w`` at all, and the module
   declares ``REPAIR_PATHS = (deposit_repair.PLAIN_PATH,)``. :func:`leg_deposit`
   measures the bracket end to end AND scores two controls: the unbracketed launch,
   which must DIVERGE, and the same product declaring the SPLIT-FIELD path, which must
   be REFUSED BY NAME rather than mis-repair.

2. **ONE EMITTER SERVING BOTH CURL ARMS.** ``conductive_kernels`` bakes the
   per-component conductivity into three ``COND`` defines and takes the certified
   lossless tail behind ``#else``, so the ``(False, False, False)`` build IS
   ``no_pml_curl``'s ``step_D_no_pml_real``. Both builds are swept, and
   :data:`SOURCE_MUTATIONS` arms BOTH branches of the certified ``#if`` -- a mutation
   table that armed only the conductive one would leave the lossless corpus row's
   arithmetic unmeasured.

``parity/meep_gpu/probe_cuda_no_pml_dispersive_electric_pair.py`` answered the same
questions OFF DEVICE first: 7 of 7 configurations byte-identical over 3 complete steps
against the driver's own pass order, six armed mutations, every null reasoned, and the
three deposit legs green. This gate answers them about the COMPILED kernel.

=============================================================================
WHY THE COMPARISON IS PER COMPLETE DRIVER STEP, WITH THE POLE VOLUMES IN IT
=============================================================================

Two engines are built from one seed and stepped side by side; the two are compared as
raw uint32 WORDS after EVERY step, over every stored field volume AND every
``P``/``P_prev`` buffer. The pole volumes are in the comparison because ``update_P``
closes the step from the stored E -- on the plain path ``Fields.drive_field`` returns
the STORED E rather than ``f_w`` (fields.py:1160-1162) -- so a wrong constitutive
product reaches ``P`` on the same step and ``D`` on the next.

THE POLE VOLUMES ARE SEEDED NON-ZERO and a case whose poles are all zero FAILS: zero
is a fixed point of ``s = s - P``, so a zero-seeded fixture turns every pole mutation
into a null and would report a green sweep that measured nothing.

THERE IS NO ``f_w_E`` AND NO ``fu_D`` ON THIS PATH, and their absence is asserted per
case rather than assumed. A fixture that found either allocated would be measuring the
split-field recurrence, which is a different sub-step and a different product.

=============================================================================
WHAT THIS GATE REFUSES TO LET PASS SILENTLY
=============================================================================

1. A step this gate invented -- the driver-order leg reads ``FdtdDriver.step``'s
   source and requires ``REPLACES`` to be a subset of the pass list, in that order.
2. A launch that did not happen -- two INDEPENDENT counters (the launcher's own report
   and a proxy over the compile memo) must agree with the step budget.
3. A mutation that was never armed -- an unarmed row fails the leg, and a leg whose
   case never reached a comparison is recorded UNMEASURED and fails rather than
   defaulting to "identical".
4. A case that measured nothing -- the movement floor requires every volume the class
   should move to have moved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_API = _HERE.parents[1]
for _path in (str(_HERE), str(_REPO_API)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import probe_fused_kernel_bit_identity as probe  # noqa: E402

# THE REAL TWIN'S GATE, IMPORTED FOR ITS SHARED MACHINERY. A second transcription of
# the driver's pass list is a second place for this gate to walk an order the driver
# does not.
import gate_cuda_fused_electric_pair as base  # noqa: E402

import cupy as cp  # noqa: E402

from meep_gpu import deposit_repair, stepping  # noqa: E402
from meep_gpu.dispersion import PolarizationState, Susceptibility  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.sources import GaussianPulsedSource  # noqa: E402

log = probe.log
to_host = probe.to_host
operand_census = probe.operand_census

run_pass = base.run_pass
array_step = base.array_step
needle = base.needle
words = base.words
MemoLaunchCounter = base.MemoLaunchCounter
DRIVER_ORDER = base.DRIVER_ORDER
GUARD_OPTIONS = base.GUARD_OPTIONS
VALUE_CLASSES = base.VALUE_CLASSES
_verdict = base._verdict

SEED = 20260902

#: Complete DRIVER STEPS every product leg runs.
STEPS = 60

#: The three passes ONE launch performs -- read from the module, never transcribed.
FUSED_PASSES: Tuple[str, ...] = ("step_D", "zero_metal_D", "update_E")

_TARGETS: Tuple[str, ...] = ("Dx", "Dy", "Dz")
_ELECTRIC: Tuple[str, ...] = ("Ex", "Ey", "Ez")

#: Every stored FIELD volume a complete step on THIS path can touch. Shorter than the
#: real twin's by six: there is no ``fu_D`` (PML storage) and no ``f_w_E`` (the plain
#: branch allocates none), and their ABSENCE is checked per case.
FIELD_NAMES: Tuple[str, ...] = _TARGETS + _ELECTRIC + ("Bx", "By", "Bz")

#: Volumes this configuration must NOT have allocated.
MUST_BE_ABSENT: Tuple[str, ...] = ("fu_Dx", "fu_Dy", "fu_Dz",
                                   "f_w_Ex", "f_w_Ey", "f_w_Ez")

MOVEMENT_FLOOR_NOTE = (
    "A case whose volumes never moved measured nothing. Every field volume in the "
    "comparison is required to move on the uniform class, and every pole volume with "
    "it: update_P writes each driven P from the stored E on every step (on the plain "
    "path drive_field returns the stored E, fields.py:1160-1162), so a P that never "
    "moved is a chain that never ran.")

CELL: Tuple[float, float, float] = (9.0, 10.0, 11.0)

#: The sweep. THE FIRST TWO ROWS ARE THE CORPUS SHAPES the two cells own, read off the
#: stamped census: absorber-1d.py and TestAbsorber.test_absorber are conductive on
#: every D component with five poles and METALLIC on z, and material-dispersion.py is
#: LOSSLESS with two poles and periodic on all three. The rest exist to arm mutations
#: those two cannot -- every axis walled somewhere, one row walled nowhere, mixed
#: conductive masks (a run may be lossy on one target and not another), half the rows
#: diagonally ANISOTROPIC (an isotropic run binds one inverse-epsilon allocation three
#: times and would hide a per-component binding defect), and arities spanning 0 to the
#: corpus ceiling.
SPECS: Tuple[Dict[str, Any], ...] = (
    {"label": "corpus_absorber_wall_z", "boundaries": ("periodic", "periodic",
                                                       "metallic"),
     "cell": CELL, "epsilon": "anisotropic", "arity": (5, 5, 5),
     "conductivity": 0.4, "cond_mask": (True, True, True)},
    {"label": "corpus_material_dispersion", "boundaries": ("periodic",) * 3,
     "cell": CELL, "epsilon": "anisotropic", "arity": (2, 2, 2),
     "conductivity": None, "cond_mask": (False, False, False)},
    {"label": "wall_xyz_conductive", "boundaries": ("metallic",) * 3,
     "cell": CELL, "epsilon": "anisotropic", "arity": (2, 2, 2),
     "conductivity": 0.35, "cond_mask": (True, True, True)},
    {"label": "wall_xyz_lossless", "boundaries": ("metallic",) * 3,
     "cell": CELL, "epsilon": "isotropic", "arity": (3, 3, 3),
     "conductivity": None, "cond_mask": (False, False, False)},
    {"label": "mixed_arity_wall_y", "boundaries": ("periodic", "metallic",
                                                   "periodic"),
     "cell": CELL, "epsilon": "anisotropic", "arity": (2, 0, 3),
     "conductivity": 0.5, "cond_mask": (True, True, True)},
    {"label": "single_pole_periodic", "boundaries": ("periodic",) * 3,
     "cell": CELL, "epsilon": "anisotropic", "arity": (1, 1, 1),
     "conductivity": 0.25, "cond_mask": (True, True, True)},
    {"label": "wall_x_lossless_high_arity", "boundaries": ("metallic", "periodic",
                                                           "periodic"),
     "cell": CELL, "epsilon": "anisotropic", "arity": (6, 6, 6),
     "conductivity": None, "cond_mask": (False, False, False)},
)

MUTATION_SPEC_LABELS: Tuple[str, ...] = (
    "corpus_absorber_wall_z", "corpus_material_dispersion", "wall_xyz_conductive",
    "wall_xyz_lossless", "mixed_arity_wall_y", "single_pole_periodic")
SEPARATE_CONTROL_LABELS: Tuple[str, ...] = ("corpus_absorber_wall_z",
                                            "wall_xyz_lossless")
DEPOSIT_SPEC_LABELS: Tuple[str, ...] = ("corpus_absorber_wall_z",
                                        "corpus_material_dispersion",
                                        "wall_xyz_conductive")


# ---------------------------------------------------------------------------
# The fixture
# ---------------------------------------------------------------------------

def case_rng(label: str) -> np.random.Generator:
    return np.random.default_rng(
        SEED + int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], "big"))


def seeded_hosts(names: Sequence[str], value_class: str, shape, rng
                 ) -> Dict[str, np.ndarray]:
    """One host array per name, in the requested value class.

    THE BASE GATE'S OWN TWO CLASSES, over THIS family's name set (which carries the
    pole buffers and carries no ``fu``/``f_w``). The subnormal band is
    ``probe.subnormal_band_hosts``' rather than a second construction: the point of
    that class is the float32 flush-to-zero policy, and two generators for it would be
    two policies.
    """
    if value_class == "uniform":
        return {name: rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
                for name in names}
    if value_class == "subnormal_band":
        return probe.subnormal_band_hosts(tuple(names), shape, rng)
    raise ValueError(f"value class {value_class!r} is not one of {VALUE_CLASSES}")


def leg_driver_order(replaces: Sequence[str]) -> Dict[str, Any]:
    """``REPLACES`` must be a CONTIGUOUS run of ``FdtdDriver.step``'s own sequence.

    Read out of ``driver.py``'s TEXT rather than by importing the driver: the question
    is what the source says, and a gate that imported the module to ask would be
    asking a different object than the one a reader checks.

    CONTIGUITY IS WHAT THE VERDICT REQUIRES. A gap would be a driver pass the launch
    skipped while the array path performed it -- and on this family the run is
    ``step_D -> zero_metal_D -> update_E``, which is contiguous only because the two
    symmetry fills between them are INERT on every row this product admits (it refuses
    a mirror plane by name). That is why the fills are excluded from the sequence this
    leg walks rather than tolerated as a gap: their absence is a refusal, not an
    oversight, and a gate that let an arbitrary gap pass could not tell the two apart.
    """
    import re  # noqa: PLC0415

    path = os.path.join(_REPO_API, "meep_gpu", "driver.py")
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    match = re.search(r"\n    def step\(self.*?(?=\n    def )", text, re.S)
    if match is None:
        return {"passed": False,
                "why": "FdtdDriver.step could not be located in driver.py"}
    body = match.group(0)
    names = "|".join(sorted(set(DRIVER_ORDER), key=len, reverse=True))
    found = tuple(re.findall(rf"\b({names})\(self\.fields", body))
    declared = tuple(replaces)
    positions = [found.index(name) for name in declared if name in found]
    subsequence = (len(positions) == len(declared)
                   and positions == sorted(positions))
    # THE FILLS ARE THE ONLY PASSES ALLOWED INSIDE THE RUN, and only because this
    # family refuses a fold: without a mirror both return at their first line
    # (stepping.py:1481-1482, :1565-1566), so the launch skips nothing the array path
    # would have done.
    inert = {"fill_symmetry_bc_D", "fill_folded_far_ghosts_D"}
    span = found[positions[0]:positions[-1] + 1] if positions else ()
    contiguous = bool(positions) and all(
        name in declared or name in inert for name in span)
    return {
        "passed": bool(found == DRIVER_ORDER and subsequence and contiguous
                       and declared == FUSED_PASSES),
        "driver_file_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "driver_call_sequence": list(found),
        "gate_declared_order": list(DRIVER_ORDER),
        "fused_replaces": list(declared),
        "replaces_is_an_ordered_subsequence": subsequence,
        "run_is_contiguous_modulo_the_inert_fills": contiguous,
        "passes_inside_the_run_left_on_the_array_path": [
            name for name in span if name not in declared],
        "why_those_are_inert": (
            "this family refuses a mirror plane by name, and without one both "
            "symmetry fills return at their first line"),
        "passes_left_on_the_array_path": [name for name in DRIVER_ORDER
                                          if name not in declared],
    }


def competing_step_D_products(fields, pml, grid, sources) -> List[str]:
    """Every OTHER ``step_D`` product whose predicate admits this fixture.

    TWO ADMITTERS ON ONE SEAM leave it UNFUSED naming both, so an overlap does not add
    coverage -- it COSTS whatever the other product was serving. Asked over the whole
    table rather than against a hand-picked neighbour, so a product landing later
    cannot quietly overlap this one.
    """
    from meep_gpu.cuda_kernels import fused_pairs  # noqa: PLC0415

    mine = "cuda_no_pml_dispersive_fused_electric_pair"

    class _Context:
        def __init__(self):
            self.fields, self.pml, self.grid, self.sources = fields, pml, grid, sources

    context = _Context()
    admitting: List[str] = []
    for family_name, product in fused_pairs.FUSED_PRODUCTS.items():
        if family_name == mine or product["curl_slot"] != "step_D":
            continue
        try:
            verdict = product["coverage"](context)
        except Exception:  # a raising predicate is a refusal, as in the composer
            continue
        if verdict.covered:
            admitting.append(family_name)
    return admitting


def pole_names(arity: Sequence[int]) -> Tuple[str, ...]:
    return tuple(f"P_{component}_{index}"
                 for component, count in zip(_ELECTRIC, arity)
                 for index in range(int(count)))


def state_names(arity: Sequence[int]) -> Tuple[str, ...]:
    return FIELD_NAMES + pole_names(arity)


def build(spec: Dict[str, Any], value_class: str, rng):
    """One CuPy engine on the PLAIN branch: ``(fields, grid, pml, host_snapshot)``.

    ``PML(thickness=0)`` absorbs on no face, so ``stepping._pml_is_active`` is False
    and ``update_E`` takes the pure overwrite at :993. ``enable_field_storage``
    allocates E WITHOUT ``f_w``, which is the storage that branch writes into.
    """
    grid = Grid(resolution=1.0, cell_size=tuple(spec["cell"]),
                boundaries=tuple(spec["boundaries"]), xp=cp, courant=0.5)
    fields = Fields(grid=grid)
    shape = tuple(int(n) for n in grid.shape)
    epsilon, inverse = {}, {}
    for offset, component in enumerate(_ELECTRIC):
        low = 1.2 + (0.4 * offset if spec["epsilon"] == "anisotropic" else 0.0)
        values = rng.uniform(low, low + 2.2, size=shape).astype(np.float32)
        if spec["epsilon"] == "isotropic" and offset:
            values = np.asarray(to_host(epsilon[_ELECTRIC[0]]))
        epsilon[component] = cp.asarray(values)
        inverse[component] = cp.asarray(
            (np.float32(1.0) / values).astype(np.float32))
    fields.set_epsilon_volumes(epsilon, inverse)

    if spec["conductivity"] is not None:
        volume = np.full(shape, float(spec["conductivity"]), dtype=np.float32)
        volume *= np.linspace(0.5, 1.5, shape[0], dtype=np.float32)[:, None, None]
        fields.set_d_conductivity(cp.asarray(volume))

    kinds = ("lorentzian", "drude")
    for index in range(max(int(v) for v in spec["arity"])):
        term = Susceptibility(frequency=0.20 + 0.03 * index, gamma=0.008,
                              kind=kinds[index % 2])
        sigma = {name: (0.35 + 0.04 * index if index < int(spec["arity"][axis]) else 0.0)
                 for axis, name in enumerate(_ELECTRIC)}
        fields.polarizations.append(
            PolarizationState(term, sigma, grid, fields._field_dtype()))
    fields.enable_field_storage()
    pml = PML(grid=grid, thickness=0)

    # ONE SEEDING CALL over every volume the comparison covers, so the value class is
    # applied uniformly. THE POLE VOLUMES ARE IN IT and are seeded NON-ZERO: zero is a
    # fixed point of ``s = s - P``, so a zero-seeded chain turns every pole mutation
    # into a null.
    pole_slots = tuple(f"{attribute}_{component}_{index}"
                       for index, _state in enumerate(fields.polarizations)
                       for attribute in ("P", "P_prev")
                       for component in _ELECTRIC)
    host = seeded_hosts(FIELD_NAMES + pole_slots, value_class, shape, rng)
    for name in FIELD_NAMES:
        getattr(fields, name)[...] = cp.asarray(np.ascontiguousarray(host[name]))
    for index, state in enumerate(fields.polarizations):
        for attribute in ("P", "P_prev"):
            store = getattr(state, attribute, None) or {}
            for component in _ELECTRIC:
                array = store.get(component)
                if array is None:
                    continue
                values = host[f"{attribute}_{component}_{index}"]
                array[...] = cp.asarray(np.ascontiguousarray(values))
    return fields, grid, pml, host


def _pole_arrays(fields) -> List[Any]:
    return [state.P[component] for component in _ELECTRIC
            for state in fields.polarizations if state.drives(component)]


def state_of(fields, arity) -> Dict[str, Any]:
    out = {name: getattr(fields, name) for name in FIELD_NAMES}
    for name, array in zip(pole_names(arity), _pole_arrays(fields)):
        out[name] = array
    return out


def compare(left, right, arity) -> Dict[str, int]:
    a, b = state_of(left, arity), state_of(right, arity)
    return {name: count for name in state_names(arity)
            if (count := int(np.count_nonzero(words(a[name]) != words(b[name]))))}


def frozen(fields, arity) -> Dict[str, np.ndarray]:
    return {name: words(array).copy()
            for name, array in state_of(fields, arity).items()}


def live_pole_words(host: Dict[str, np.ndarray], arity=None) -> int:
    _ = arity
    return int(sum(int(np.count_nonzero(values != 0))
                   for name, values in host.items()
                   if name.startswith("P_") or name.startswith("P_prev_")))


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------

def run_case(spec: Dict[str, Any], value_class: str, steps: int,
             kernel: Optional[Any] = None,
             codes_patch: Optional[Callable[[Sequence[Any]], Sequence[Any]]] = None,
             walls_patch: Optional[Callable[[Sequence[int], Any], Sequence[int]]] = None,
             plan_patch: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
             label_suffix: str = "") -> Dict[str, Any]:
    """Step two engines side by side and compare per COMPLETE DRIVER STEP."""
    from meep_gpu.cuda_kernels import dispersive_kernels  # noqa: PLC0415
    from meep_gpu.cuda_kernels import in_seam_coverage, step_curl_kernels  # noqa: PLC0415
    from meep_gpu.cuda_kernels import (  # noqa: PLC0415
        no_pml_dispersive_fused_electric_pair as family,
    )

    started = time.time()
    label = f"{spec['label']}|{value_class}{label_suffix}"
    arity = tuple(int(v) for v in spec["arity"])
    case: Dict[str, Any] = {"label": spec["label"], "value_class": value_class,
                            "case_seed_label": label, "steps_requested": steps,
                            "boundaries": list(spec["boundaries"]),
                            "epsilon": spec["epsilon"], "arity": list(arity),
                            "conductivity": spec["conductivity"]}

    reference, ref_grid, ref_pml, host = build(spec, value_class, case_rng(label))
    actual, grid, pml, _ = build(spec, value_class, case_rng(label))
    case["shape"] = [int(n) for n in grid.shape]
    case["dtdx"] = float(grid.dt / grid.dx)
    case["live_pole_words"] = live_pole_words(host, arity)
    case["distinct_inverse_epsilon_allocations"] = len({
        int(actual.inverse_epsilon_for(c).data.ptr) for c in _ELECTRIC})

    # THE SUB-STEP THIS GATE CLAIMS TO MEASURE, CHECKED RATHER THAN ASSUMED.
    allocated = [name for name in MUST_BE_ABSENT
                 if getattr(actual, name, None) is not None]
    case["unexpectedly_allocated"] = allocated
    case["pml_is_active"] = bool(stepping._pml_is_active(pml))
    if allocated or case["pml_is_active"] or not actual.stores_E:
        case["passed"] = False
        case["why"] = (f"this fixture is not on the plain stored-E branch: "
                       f"pml_is_active={case['pml_is_active']}, "
                       f"stores_E={bool(actual.stores_E)}, allocated={allocated}")
        return case
    if sum(arity) and not case["live_pole_words"]:
        case["passed"] = False
        case["why"] = ("the pole volumes are all zero, so every pole mutation is a "
                       "null and this case measures nothing about the chain")
        return case

    drift = compare(reference, actual, arity)
    if drift:
        case["passed"] = False
        case["why"] = f"the two builds are not identical: {drift}"
        return case

    covered, reason = family.covers_no_pml_dispersive_fused_electric_pair(
        actual, pml, grid, ())
    case["predicate"] = {"covered": bool(covered), "reason": reason}
    if not covered:
        case["passed"] = False
        case["why"] = f"the predicate refused this fixture: {reason}"
        return case

    # NO OTHER step_D PRODUCT MAY ADMIT THIS FIXTURE. Two admitters leave the seam
    # UNFUSED naming both, so an overlap costs coverage rather than adding it.
    overlaps = competing_step_D_products(actual, pml, grid, ())
    case["competing_products"] = overlaps
    if overlaps:
        case["passed"] = False
        case["why"] = (f"another step_D product admits this fixture too "
                       f"({overlaps}); install_fused_pairs would leave the seam "
                       f"unfused naming both")
        return case

    conductivity, cond = family.conductive_bindings(actual)
    case["cond_mask"] = [bool(flag) for flag in cond]
    if tuple(cond) != tuple(bool(v) for v in spec["cond_mask"]):
        case["passed"] = False
        case["why"] = (f"the resolved conductive mask {tuple(cond)} is not the spec's "
                       f"{tuple(spec['cond_mask'])}; the compiled body and the "
                       f"bindings would describe different runs")
        return case

    plan = dispersive_kernels.resolve_pole_plan(actual)
    if plan_patch is not None:
        plan = plan_patch(plan)
    poles = tuple(array for component in _ELECTRIC for array in plan[component])
    case["distinct_allocations_checked"] = family.assert_disjoint_bindings(
        actual, poles, conductivity)
    case["compiled_arity"] = list(dispersive_kernels.pole_counts_of(plan))

    codes = step_curl_kernels.real_curl_boundary_codes(grid)
    walls = in_seam_coverage.zero_metal_axes(grid)
    case["boundary_codes"] = [int(c) for c in codes]
    case["zero_metal_axes"] = [bool(w) for w in walls]
    if codes_patch is not None:
        codes = codes_patch(codes)
    if walls_patch is not None:
        walls = walls_patch(walls, grid)
    case["boundary_codes_used"] = [int(c) for c in codes]
    case["zero_metal_axes_used"] = [bool(w) for w in walls]

    before = frozen(actual, arity)
    launched = 0
    counter = MemoLaunchCounter()
    per_step: List[Dict[str, Any]] = []
    with counter:
        for step in range(1, steps + 1):
            array_step(reference, ref_pml)
            # THE POLE PLAN IS RE-RESOLVED EVERY STEP: PolarizationState.update
            # rotates the buffers (dispersion.py:689-691) and a cached plan names
            # last step's arrays -- stale in a way that still computes.
            live = dispersive_kernels.resolve_pole_plan(actual)
            if plan_patch is not None:
                live = plan_patch(live)
            step_poles = tuple(array for component in _ELECTRIC
                               for array in live[component])
            live_conductivity, live_cond = family.conductive_bindings(actual)
            for name in DRIVER_ORDER:
                if name == "step_D":
                    report = family.launch_no_pml_dispersive_fused_electric_pair(
                        actual, codes, walls, case["dtdx"], step_poles,
                        dispersive_kernels.pole_counts_of(live), live_conductivity,
                        live_cond, kernel)
                    launched += int(bool(report.get("launched")))
                    cp.cuda.runtime.deviceSynchronize()
                elif name not in FUSED_PASSES:
                    run_pass(name, actual, pml)
            cp.cuda.runtime.deviceSynchronize()
            difference = compare(reference, actual, arity)
            census = operand_census({n: to_host(v)
                                     for n, v in state_of(reference, arity).items()})
            per_step.append({"step": step,
                             "differing_words": sum(difference.values()),
                             "differing_arrays": dict(sorted(difference.items())),
                             "reference_subnormals": census["subnormals"],
                             "reference_negative_zeros": census["negative_zeros"]})
            if difference:
                break

    after = state_of(actual, arity)
    moved = {name: int(np.count_nonzero(before[name] != words(after[name])))
             for name in state_names(arity)}
    still = sorted(name for name, count in moved.items() if count == 0)
    required = (state_names(arity) if value_class == "uniform"
                else _TARGETS + _ELECTRIC + pole_names(arity))
    unmoved_required = sorted(name for name in required if moved.get(name, 0) == 0)
    identical = (len(per_step) == steps
                 and all(row["differing_words"] == 0 for row in per_step))
    launch_ok = (launched == len(per_step)
                 and counter.named().get(family.KERNEL_NAME, 0)
                 == (len(per_step) if kernel is None else 0))
    case.update({
        "passed": bool(identical and not unmoved_required and launch_ok),
        "bit_identical": identical,
        "steps_run": len(per_step),
        "first_divergence": next((row["step"] for row in per_step
                                  if row["differing_words"]), None),
        "per_step": per_step,
        "differing_words": per_step[-1]["differing_words"] if per_step else -1,
        "differing_arrays": per_step[-1]["differing_arrays"] if per_step else {},
        "arrays_compared": len(state_names(arity)),
        "pole_arrays_compared": len(pole_names(arity)),
        "arrays_that_never_moved": still,
        "movement_floor": {"class": value_class, "required": list(required),
                           "unmet": unmoved_required, "note": MOVEMENT_FLOOR_NOTE},
        "launch_counts": {
            "launcher_reports": launched,
            "memo_proxy": counter.named(),
            "memo_proxy_total": counter.total,
            "kernel_supplied_by_gate": kernel is not None,
            "agree": bool(launch_ok),
            "expected_per_step": 1,
        },
        "seconds": time.time() - started,
    })
    return case


# ---------------------------------------------------------------------------
# The separate composition: what the fusion removes, measured
# ---------------------------------------------------------------------------

def leg_separate_control(spec: Dict[str, Any], steps: int) -> Dict[str, Any]:
    """THREE engines from one seed: array path, separate certified products, fused.

    The separate composition launches the certified conductive (or lossless) curl, the
    certified ``zero_metal_D`` where a wall is live, and the certified
    ``update_E_no_pml_real_dispersive``. All three must agree word for word; what
    differs is the LAUNCH COUNT, which is the only place the fusion is visible.
    """
    from meep_gpu.cuda_kernels import conductive_kernels, dispersive_kernels  # noqa: PLC0415
    from meep_gpu.cuda_kernels import no_pml_curl  # noqa: PLC0415
    from meep_gpu.cuda_kernels import in_seam_coverage, in_seam_passes  # noqa: PLC0415
    from meep_gpu.cuda_kernels import step_curl_kernels  # noqa: PLC0415
    from meep_gpu.cuda_kernels import (  # noqa: PLC0415
        no_pml_dispersive_fused_electric_pair as family,
    )

    started = time.time()
    label = f"{spec['label']}|separate"
    arity = tuple(int(v) for v in spec["arity"])
    reference, ref_grid, ref_pml, _ = build(spec, "uniform", case_rng(label))
    separate, sep_grid, sep_pml, _ = build(spec, "uniform", case_rng(label))
    fused, grid, pml, _ = build(spec, "uniform", case_rng(label))

    codes = step_curl_kernels.real_curl_boundary_codes(grid)
    walls = in_seam_coverage.zero_metal_axes(grid)
    dtdx = float(grid.dt / grid.dx)
    row: Dict[str, Any] = {"label": spec["label"], "arity": list(arity),
                           "zero_metal_axes": [bool(w) for w in walls]}

    separate_counter = MemoLaunchCounter()
    fused_counter = MemoLaunchCounter()
    for _step in range(steps):
        array_step(reference, ref_pml)
        # EACH COMPOSITION RUNS AT step_D'S POSITION IN THE DRIVER'S ORDER, not before
        # the whole walk. The first cut of this leg ran the D/E work first and then
        # every other pass, which puts step_B and update_H AFTER the displacement they
        # feed -- and BOTH compositions then diverged from the array path in exactly
        # the same words, which is what showed the fault was the harness's rather than
        # either composition's.
        for name in DRIVER_ORDER:
            if name == "step_D":
                with separate_counter:
                    # THE CERTIFIED CURL FOR THIS SPEC'S ARM, and the branch is the
                    # point of this control rather than a convenience.
                    # `covers_conductive_curl` REFUSES a run where no target carries a
                    # sigma, so launching the conductive family on the lossless spec
                    # would be running a certified kernel outside the domain its own
                    # predicate admits -- and the control would be comparing the weld
                    # against a composition the array path would never build.
                    if any(spec["cond_mask"]):
                        conductive_kernels.step_conductive_curl(
                            separate, sep_pml, "step_D", codes, dtdx)
                    else:
                        no_pml_curl.step_no_pml_curl(separate, "step_D", codes, dtdx)
                    if any(walls):
                        # THE CERTIFIED IN-SEAM PASS, through its own dispatcher.
                        in_seam_passes.run_pass("zero_metal", separate, "D",
                                                grid=sep_grid)
                    dispersive_kernels.update_E_no_pml_real_dispersive(
                        separate, sep_pml)
                cp.cuda.runtime.deviceSynchronize()
            elif name not in FUSED_PASSES:
                run_pass(name, separate, sep_pml)
        for name in DRIVER_ORDER:
            if name == "step_D":
                with fused_counter:
                    poles, counts = family.pole_bindings(fused)
                    conductivity, cond = family.conductive_bindings(fused)
                    family.launch_no_pml_dispersive_fused_electric_pair(
                        fused, codes, walls, dtdx, poles, counts, conductivity, cond)
                cp.cuda.runtime.deviceSynchronize()
            elif name not in FUSED_PASSES:
                run_pass(name, fused, pml)
    cp.cuda.runtime.deviceSynchronize()

    row["separate_vs_array"] = compare(reference, separate, arity)
    row["fused_vs_array"] = compare(reference, fused, arity)
    row["separate_launches"] = separate_counter.total
    row["fused_launches"] = fused_counter.total
    row["launches_removed_per_step"] = (
        (separate_counter.total - fused_counter.total) / max(steps, 1))
    row["passed"] = (not row["separate_vs_array"] and not row["fused_vs_array"]
                     and fused_counter.total < separate_counter.total)
    row["seconds"] = time.time() - started
    return row


# ---------------------------------------------------------------------------
# The deposit legs -- what PLAIN_PATH is for
# ---------------------------------------------------------------------------

def leg_deposit(spec: Dict[str, Any], steps: int) -> Dict[str, Any]:
    """The bracket end to end, with TWO controls that must not pass.

    ``bracketed``            save -> launch on an UNINJECTED D -> inject -> clear
                             -> repair, byte-identical to the driver's own order.
    ``unbracketed``          the same without the repair. MUST DIVERGE -- the fused
                             launch consumed a pre-injection displacement.
    ``split_field_declared`` the same product declaring the WRONG repair. MUST BE
                             REFUSED BY NAME -- the split-field inverse would write an
                             answer the step never computed.
    """
    from meep_gpu.cuda_kernels import in_seam_coverage, step_curl_kernels  # noqa: PLC0415
    from meep_gpu.cuda_kernels import (  # noqa: PLC0415
        no_pml_dispersive_fused_electric_pair as family,
    )

    started = time.time()
    arity = tuple(int(v) for v in spec["arity"])
    row: Dict[str, Any] = {"label": spec["label"], "arity": list(arity)}

    def engine(tag: str):
        fields, grid, pml, _ = build(spec, "uniform",
                                     case_rng(f"{spec['label']}|deposit"))
        source = GaussianPulsedSource(grid=grid, component="Ez",
                                      center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                                      frequency=1.0, fwidth=0.2, amplitude=1.0)
        _ = tag
        return fields, grid, pml, [source]

    reference, ref_grid, ref_pml, ref_sources = engine("reference")
    dt = float(ref_grid.dt)

    # DOES THE FIXTURE ACTUALLY DEPOSIT? A source whose deposit never moves a word
    # makes every leg below a null, so the question is asked before the legs are
    # scored -- and it is asked AT THE TIMES THE LEGS THEMSELVES USE.
    #
    # THAT LAST CLAUSE IS NOT PEDANTRY. The first cut probed a single hardcoded
    # t = 0.5, and this source's carrier has period 1/frequency = 1.0, so every
    # half-integer time is an exact ZERO of its amplitude: the probe reported
    # `deposited_words = 0` and declared three perfectly good legs vacuous. Sampling
    # the legs' own step times cannot land on a systematic null the legs never see.
    probe_fields, _pg, _pp, probe_sources = engine("probe")
    deposited = 0
    for step in range(steps):
        before = words(probe_fields.Dz).copy()
        probe_sources[0].inject(probe_fields, (step + 0.5) * dt)
        cp.cuda.runtime.deviceSynchronize()
        deposited += int(np.count_nonzero(before != words(probe_fields.Dz)))
    row["deposited_words"] = deposited

    legs: Dict[str, Any] = {}
    for tag in ("bracketed", "unbracketed", "split_field_declared"):
        fields, grid, pml, sources = engine(tag)
        codes = step_curl_kernels.real_curl_boundary_codes(grid)
        walls = in_seam_coverage.zero_metal_axes(grid)
        dtdx = float(grid.dt / grid.dx)
        paths = ((deposit_repair.SPLIT_FIELD_PATH,) if tag == "split_field_declared"
                 else family.REPAIR_PATHS)
        refusal = None
        for step in range(steps):
            when = (step + 0.5) * dt
            # THE FUSED ROUTE RUNS AT step_D'S POSITION, with every other driver pass
            # in its own place around it -- the same walk `run_case` uses, and the one
            # the reference below performs.
            for name in DRIVER_ORDER:
                if name == "step_D":
                    try:
                        saved = deposit_repair.save(fields, tuple(sources), "D", pml,
                                                    paths=paths)
                    except Exception as error:  # a refusal is the measurement
                        refusal = f"{type(error).__name__}: {error}"
                        break
                    poles, counts = family.pole_bindings(fields)
                    conductivity, cond = family.conductive_bindings(fields)
                    family.launch_no_pml_dispersive_fused_electric_pair(
                        fields, codes, walls, dtdx, poles, counts, conductivity, cond)
                    cp.cuda.runtime.deviceSynchronize()
                    # THE DRIVER'S OWN SEAM, between the two consults: inject, then
                    # the wall clear that sits behind no consult at all.
                    for source in sources:
                        source.inject(fields, when)
                    stepping.zero_metal_D(fields)
                    if tag != "unbracketed":
                        deposit_repair.apply(fields, pml, tuple(sources), "D", saved)
                elif name not in FUSED_PASSES:
                    run_pass(name, fields, pml)
            if refusal is not None:
                break
        # THE ENGINE IS KEPT AND COMPARED AFTER THE REFERENCE HAS RUN. The first cut
        # compared here, inside this loop -- against a reference still sitting at its
        # initial state, because its own run is below -- so all three legs reported
        # every word of every array differing and the leg still passed, because the
        # verdict below never asked whether `bracketed` was identical. Both halves of
        # that are fixed: the comparison moved out, and the floor was added.
        legs[tag] = {"refused": refusal, "fields": None if refusal else fields}

    # The reference's own run, in the driver's order, with the array path's three
    # passes at step_D's position.
    for step in range(steps):
        when = (step + 0.5) * dt
        for name in DRIVER_ORDER:
            if name == "step_D":
                stepping.step_D(reference, ref_pml)
                for source in ref_sources:
                    source.inject(reference, when)
                stepping.zero_metal_D(reference)
                stepping.update_E(reference, ref_pml)
            elif name not in FUSED_PASSES:
                run_pass(name, reference, ref_pml)
    cp.cuda.runtime.deviceSynchronize()

    scored = {tag: {"refused": entry["refused"],
                    "differing_words": (None if entry["fields"] is None else
                                        sum(compare(reference, entry["fields"],
                                                    arity).values()))}
              for tag, entry in legs.items()}
    row["legs"] = scored
    # THE FLOOR, AND ITS FIRST CLAUSE IS THE ONE THAT WAS MISSING. A leg that ran the
    # bracket and diverged is a FAILURE, not a leg that "ran"; without this clause the
    # first cut reported OK while the bracketed route differed in every word of every
    # array. The other three clauses are the controls: the repair must not refuse the
    # configuration it exists for, the UNBRACKETED route must diverge (otherwise the
    # fixture deposits nothing the launch could have missed), and declaring the WRONG
    # recurrence must be refused BY NAME rather than mis-repaired.
    row["passed"] = bool(
        row["deposited_words"] > 0
        and scored["bracketed"]["refused"] is None
        and scored["bracketed"]["differing_words"] == 0
        and scored["unbracketed"]["refused"] is None
        and (scored["unbracketed"]["differing_words"] or 0) > 0
        and scored["split_field_declared"]["refused"] is not None)
    row["seconds"] = time.time() - started
    return row


# ---------------------------------------------------------------------------
# The mutation tables
# ---------------------------------------------------------------------------

def _arity(spec) -> Tuple[int, int, int]:
    return tuple(int(v) for v in spec["arity"])


def _mask(spec) -> Tuple[bool, bool, bool]:
    return tuple(bool(v) for v in spec["cond_mask"])


def _driven(component_index: int, minimum: int = 1):
    return lambda spec: _arity(spec)[component_index] >= minimum


def _wall(axis: str):
    return lambda spec: spec["boundaries"]["xyz".index(axis)] == "metallic"


def _conductive(spec) -> bool:
    return any(_mask(spec))


def _lossless(spec) -> bool:
    return not any(_mask(spec))


def _anisotropic(spec) -> bool:
    return spec["epsilon"] == "anisotropic"


def _always(spec) -> bool:
    return True


#: DEVICE-TEXT needles. Each is (label, (old, new, count), must_be_caught, applies).
SOURCE_MUTATIONS: Tuple[Tuple[str, Tuple[str, str, int], Optional[bool],
                              Callable[[Dict[str, Any]], bool]], ...] = (
    # --- THE SEAM, and this is what the fusion is ----------------------------
    # THE CHAIN OPENS ON THE REGISTER, NOT ON A RELOAD OF THE VOLUME -- and this leg
    # is a CONFIRMED NULL rather than a caught defect, deliberately. By the time the
    # constitutive body runs, the store above has already written the CLEARED register
    # into D, so `Dx[idx]` and `d_x` hold the same word and the substitution is the
    # identity. The first cut of this table scored it `must_be_caught=True` and it
    # came back UNCAUGHT; the honest reading is that the reload is exact HERE, and the
    # ordering claim is carried by `wall_clear_misses_the_register` below, which is
    # the same question asked where it IS observable.
    ("chain_reloads_the_stored_displacement",
     ("    float s_x = d_x;", "    float s_x = Dx[idx];", 1), False,
     lambda spec: _driven(0)(spec) and (_wall("y")(spec) or _wall("z")(spec))),
    ("seam_takes_the_wrong_component",
     ("    float s_x = d_x;", "    float s_x = d_y;", 1), True, _driven(0)),
    # --- BOTH BRANCHES OF THE CERTIFIED #if ----------------------------------
    # THE CONDUCTIVE TAIL. Dropping the condinv factor is the whole conductivity.
    ("conductive_tail_drops_condinv",
     ("    return ((f[idx] * cf[idx]) - curl) * ci[idx];",
      "    return (f[idx] * cf[idx]) - curl;", 1), True, _conductive),
    ("conductive_tail_drops_condfac",
     ("    return ((f[idx] * cf[idx]) - curl) * ci[idx];",
      "    return (f[idx] - curl) * ci[idx];", 1), True, _conductive),
    # THE LOSSLESS TAIL, armed on the rows whose mask takes it. A table that armed
    # only the conductive branch would leave material-dispersion.py's arithmetic
    # unmeasured.
    ("plain_tail_adds_instead_of_subtracting",
     ("        d_x = Dx[idx] - curl;", "        d_x = Dx[idx] + curl;", 1), True,
     _lossless),
    # --- THE CHAIN'S ORDER AND SHAPE -----------------------------------------
    # SEQUENTIAL, LEFT TO RIGHT: ((D - P0) - P1). float32 addition is not
    # associative, so summing first is a different number -- and only at two poles or
    # more, which is why the applicability predicate asks for two.
    ("chain_summed_then_subtracted",
     ("    s_x = s_x - P_Ex_0[idx];\n    s_x = s_x - P_Ex_1[idx];",
      "    s_x = s_x - (P_Ex_0[idx] + P_Ex_1[idx]);", 1), True, _driven(0, 2)),
    ("chain_order_reversed",
     ("    s_x = s_x - P_Ex_0[idx];\n    s_x = s_x - P_Ex_1[idx];",
      "    s_x = s_x - P_Ex_1[idx];\n    s_x = s_x - P_Ex_0[idx];", 1), True,
     _driven(0, 2)),
    ("chain_drops_its_last_pole",
     ("    s_z = s_z - P_Ez_1[idx];", "", 1), True, _driven(2, 2)),
    ("chain_takes_another_components_pole",
     ("    s_x = s_x - P_Ex_0[idx];", "    s_x = s_x - P_Ez_0[idx];", 1), True,
     lambda spec: _arity(spec)[0] >= 1 and _arity(spec)[2] >= 1),
    # --- THE INVERSE-EPSILON MULTIPLY ----------------------------------------
    # D on the LEFT is transcription discipline (IEEE multiply commutes), so this is a
    # NULL that must come back UNCAUGHT -- paired with the component swap below, which
    # must be caught. Without the pair, "the operand order is inert" would be a claim
    # about a leg nobody showed could fail.
    ("inverse_epsilon_operand_order_swapped",
     ("    float src_x = s_x * inv_eps_Ex[idx];",
      "    float src_x = inv_eps_Ex[idx] * s_x;", 1), False, _driven(0)),
    ("inverse_epsilon_component_swapped",
     ("    float src_x = s_x * inv_eps_Ex[idx];",
      "    float src_x = s_x * inv_eps_Ez[idx];", 1), True,
     lambda spec: _anisotropic(spec) and _driven(0)(spec)),
    # --- THE WALL CARRY ------------------------------------------------------
    ("wall_clear_uses_the_b_diagonal",
     ("if (wall_x && i == 0) { d_y = 0.0f; d_z = 0.0f; }",
      "if (wall_x && i == 0) { d_x = 0.0f; }", 1), True, _wall("x")),
    # THE CLEAR MUST REACH THE REGISTER, and this is where that ordering IS
    # observable. Clearing the VOLUME instead leaves the register uncleared, the store
    # below then writes the uncleared value back over the zero, and BOTH consumers see
    # a wall cell the array path holds at zero. (The first cut tried to arm the same
    # claim by moving the store above the clear -- unobservable, because the real
    # store then overwrites it. It scored UNCAUGHT and was replaced by this.)
    ("wall_clear_misses_the_register",
     ("    if (wall_z && k == 0) { d_x = 0.0f; d_y = 0.0f; }",
      "    if (wall_z && k == 0) { Dx[idx] = 0.0f; Dy[idx] = 0.0f; }", 1), True,
     _wall("z")),
    # --- THE CURL ------------------------------------------------------------
    ("curl_shift_direction_flipped",
     ("float sf = shift_dn(Hz, idx, j, ny, sy, bc_y);",
      "float sf = shift_up(Hz, idx, j, ny, sy, bc_y);", 1), True, _always),
)


def _reversed_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    return {component: list(reversed(arrays)) for component, arrays in plan.items()}


def _rotated_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    return {"Ex": list(plan["Ez"]), "Ey": list(plan["Ex"]), "Ez": list(plan["Ey"])}


def _own_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    return {component: list(arrays) for component, arrays in plan.items()}


HOST_MUTATIONS: Tuple[Tuple[str, Dict[str, Any], Optional[bool],
                            Callable[[Dict[str, Any]], bool]], ...] = (
    ("pole_plan_reversed", {"plan_patch": _reversed_plan}, True,
     lambda spec: max(_arity(spec)) >= 2),
    ("pole_plan_rotated_across_components", {"plan_patch": _rotated_plan}, True,
     lambda spec: len(set(_arity(spec))) == 1 and min(_arity(spec)) >= 1),
    # A NULL: the plan the launcher would have resolved, handed in explicitly, must
    # change nothing. Without it "caught" above could mean the harness diverges on any
    # plan object at all.
    ("pole_plan_resolved_explicitly", {"plan_patch": _own_plan}, False, _always),
    ("walls_dropped", {"walls_patch": base._walls_dropped}, True,
     lambda spec: any(_wall(axis)(spec) for axis in "xyz")),
)


def compile_source(source: str, options: Sequence[str]):
    from meep_gpu.cuda_kernels import (  # noqa: PLC0415
        no_pml_dispersive_fused_electric_pair as family,
    )
    return cp.RawKernel(source, family.KERNEL_NAME, options=tuple(options))


# ---------------------------------------------------------------------------
# The runners
# ---------------------------------------------------------------------------

def save(results, path):
    """Serialise the payload, WELDED TO THE BYTES THIS PROCESS IMPORTED.

    ``gate_provenance.stamp`` records a sha256 for every repo module the running
    interpreter actually loaded, under its own key. That is what
    ``rebind_cuda_welds`` binds a ledger entry against: without it the campaign is
    released and UNBINDABLE -- the artifact looks green and nothing in
    ``fingerprints.json`` ever points at it. (Measured on this family's first cut:
    both campaigns released and the rebind skipped them with "the legs record no
    agreed imported_source_sha256".)

    It is a DISTINCT key from any curated ``source_sha256`` a gate builds, and
    re-stamping is allowed as long as no already-recorded digest CHANGES -- a file
    rewritten mid-run means the measurements on either side describe different
    programs.
    """
    from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415

    _stamp_provenance(results)
    Path(path).write_text(json.dumps(results, indent=1, default=str))


def warm_memo(specs_by_label: Dict[str, Any]) -> Dict[str, Any]:
    """Compile every (mask, arity) this run will launch, OUTSIDE any counted region.

    ``MemoLaunchCounter`` proxies the shipped compile memo, so a kernel first compiled
    inside a counted region is launched and NOT counted, and the case's launch total
    comes back one short of its step budget -- a failure that looks like a fallback
    and is not one.
    """
    from meep_gpu.cuda_kernels import (  # noqa: PLC0415
        no_pml_dispersive_fused_electric_pair as family,
    )

    started = time.time()
    builds = sorted({(_mask(spec), _arity(spec))
                     for spec in specs_by_label.values()})
    for mask, arity in builds:
        family._get_kernel(mask, arity)
    return {"builds_compiled": [[list(m), list(a)] for m, a in builds],
            "seconds": round(time.time() - started, 2)}


def run_product(results, out_path, specs, classes, steps):
    cases: List[Dict[str, Any]] = []
    total = len(specs) * len(classes)
    for number, spec in enumerate(specs):
        for index, value_class in enumerate(classes):
            case = run_case(spec, value_class, steps)
            cases.append(case)
            results["product"] = cases
            save(results, out_path)
            log(f"  case {number * len(classes) + index + 1}/{total} "
                f"{spec['label']}|{value_class} arity={case.get('arity')} "
                f"cond={case.get('cond_mask')}: "
                f"{'OK' if case.get('passed') else 'FAIL'} "
                f"words={case.get('differing_words')} "
                f"poles={case.get('live_pole_words')} "
                f"({case.get('seconds', 0):.1f} s)")
    return cases


def _score(results, out_path, steps, specs_by_label, table, key, runner):
    rows: List[Dict[str, Any]] = []
    for entry in table:
        label, payload, must, applies = entry
        scored, caught, legs, unmeasured = 0, 0, [], []
        for spec_label in MUTATION_SPEC_LABELS:
            spec = specs_by_label[spec_label]
            if not applies(spec):
                legs.append({"spec": spec_label, "applicable": False})
                continue
            case = runner(spec, payload, steps, label)
            if case is None:
                legs.append({"spec": spec_label, "applicable": True, "armed": False})
                continue
            # A CASE THAT NEVER REACHED A COMPARISON IS NOT A VERDICT. It is recorded
            # unmeasured and FAILS the leg, because a mutation table with an
            # unmeasurable row is a table whose denominator is wrong.
            if "bit_identical" not in case:
                unmeasured.append(spec_label)
                legs.append({"spec": spec_label, "applicable": True, "armed": True,
                             "measured": False, "why": case.get("why")})
                continue
            diverged = not case["bit_identical"]
            scored += 1
            caught += int(diverged)
            legs.append({"spec": spec_label, "applicable": True, "armed": True,
                         "measured": True, "diverged": diverged,
                         "first_divergence": case.get("first_divergence"),
                         "differing_words": case.get("differing_words")})
        verdict = ("UNMEASURABLE LEG" if unmeasured
                   else _verdict(caught, scored, must))
        rows.append({"mutation": label, "must_be_caught": must, "scored": scored,
                     "caught": caught, "verdict": verdict,
                     "unmeasured_specs": unmeasured, "legs": legs})
        results[key] = rows
        save(results, out_path)
        log(f"  {key[:-1]} {label}: {verdict} ({caught}/{scored})")
    return rows


def run_mutations(results, out_path, steps, specs_by_label):
    from meep_gpu.cuda_kernels import (  # noqa: PLC0415
        no_pml_dispersive_fused_electric_pair as family,
    )

    def runner(spec, payload, step_count, label):
        old, new, count = payload
        source = family.no_pml_dispersive_fused_electric_pair_source(
            _mask(spec), _arity(spec))
        mutated, hits = needle(source, old, new, count)
        if hits != count:
            return None
        kernel = compile_source(mutated, GUARD_OPTIONS)
        return run_case(spec, "uniform", step_count, kernel=kernel,
                        label_suffix=f"|{label}")

    return _score(results, out_path, steps, specs_by_label, SOURCE_MUTATIONS,
                  "source_mutations", runner)


def run_host_mutations(results, out_path, steps, specs_by_label):
    def runner(spec, hooks, step_count, label):
        return run_case(spec, "uniform", step_count, label_suffix=f"|{label}", **hooks)

    return _score(results, out_path, steps, specs_by_label, HOST_MUTATIONS,
                  "host_mutations", runner)


#: Every leg a RELEASED run must carry. A ``--only`` run skips some, and each skipped
#: leg leaves an empty list whose ``all()`` is True -- so the release verdict has to ask
#: which legs actually RAN rather than which of the ones that ran passed.
REQUIRED_LEGS: Tuple[str, ...] = ("driver_order", "product", "separate_control", "deposit",
                                  "source_mutations", "host_mutations")


def summarize(results: Dict[str, Any], steps: int) -> Dict[str, Any]:
    product = results.get("product", [])
    source_rows = results.get("source_mutations", [])
    host_rows = results.get("host_mutations", [])
    unarmed = [row["mutation"] for row in source_rows + host_rows
               if row["scored"] == 0]
    failed = [row["mutation"] for row in source_rows + host_rows
              if row["verdict"] in ("UNCAUGHT", "PARTIAL", "NULL VIOLATED",
                                    "NO LEGS", "UNMEASURABLE LEG")]
    legs = {
        "driver_order": results.get("driver_order", {}).get("passed"),
        "product": all(case.get("passed") for case in product) and bool(product),
        "separate_control": all(row.get("passed")
                                for row in results.get("separate_control", [])),
        "deposit": all(row.get("passed") for row in results.get("deposit", [])),
        "source_mutations": not failed and not unarmed,
    }
    masks = sorted({tuple(case["cond_mask"]) for case in product
                    if "cond_mask" in case})
    arities = sorted({tuple(case["arity"]) for case in product if "arity" in case})
    # A PARTIAL RUN MAY NOT REPORT ITSELF RELEASED. ``--only`` exists for re-running one
    # leg during development, and every skipped leg leaves an EMPTY list whose ``all()``
    # is True -- so without this a two-leg smoke run would carry
    # ``canonical_verdict.released`` and the record writer would transcribe it as a
    # campaign. The legs are named rather than counted so the artifact says WHICH is
    # missing.
    missing = [name for name, value in legs.items() if value is None]
    absent = [name for name in REQUIRED_LEGS if name not in results]
    return {
        "legs": legs,
        "legs_not_run": absent,
        "legs_with_no_verdict": missing,
        "released": (all(bool(value) for value in legs.values())
                     and not absent and not missing),
        "cases": len(product),
        "cases_passed": sum(1 for case in product if case.get("passed")),
        "steps_per_case": steps,
        "conductive_masks_swept": [list(m) for m in masks],
        "arities_swept": [list(a) for a in arities],
        "mutations_scored": sum(row["scored"] for row in source_rows + host_rows),
        "mutations_caught": sum(row["caught"] for row in source_rows + host_rows),
        "mutations_never_armed": unarmed,
        "mutations_that_failed": failed,
        "denominators": {
            "board_cells": [
                "D_to_E (cuda_conductive/conductive, "
                "cuda_no_pml_dispersive/no-PML dispersive store)",
                "D_to_E (cuda_no_pml_curl/no-PML curl, "
                "cuda_no_pml_dispersive/no-PML dispersive store)",
            ],
            "corpus_rows_on_the_cells": 3,
            "corpus_rows_with_an_electric_deposit": 3,
            "corpus_rows_folded": 0,
        },
        "what_this_does_not_claim": [
            "no throughput or dispatch claim of any kind",
            "no verdict about a FOLDED run -- a mirror plane is refused BY NAME and "
            "none is swept",
            "no verdict about a complex, cylindrical, BFAST, special-kz, "
            "off-diagonal, nonlinear or ACTIVE-absorber run -- each is refused BY "
            "NAME and none swept",
            "no verdict about the split-field electric welds, which have their own "
            "gates; the only claim made about that repair here is that declaring it "
            "on THIS cell is REFUSED",
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="FILE to write gate.json to")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--subnormal-policy", choices=("keep", "flush"),
                        required=True)
    parser.add_argument("--import-meep-for-host-policy", action="store_true")
    parser.add_argument("--only", default=None,
                        help="comma-separated leg names, for a partial re-run")
    arguments = parser.parse_args(argv)

    out_path = Path(arguments.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    only = set(arguments.only.split(",")) if arguments.only else None
    specs_by_label = {spec["label"]: spec for spec in SPECS}
    results: Dict[str, Any] = {
        "gate": "cuda_no_pml_dispersive_fused_electric_pair",
        "subject": "meep_gpu/cuda_kernels/no_pml_dispersive_fused_electric_pair.py",
        "kernel": "no_pml_dispersive_fused_electric_pair_real",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename,
        "argv": list(argv if argv is not None else sys.argv[1:]),
        "question": ("does ONE launch of no_pml_dispersive_fused_electric_pair_real leave every stored volume -- the pole buffers included -- byte-identical to stepping's step_D -> zero_metal_D -> plain stored-E update_E inside a complete driver step, on BOTH curl arms, and with an electric source deposited in the seam carried by the PLAIN repair?"),
        "subnormal_policy": arguments.subnormal_policy,
        "steps": arguments.steps,
        "specs": [spec["label"] for spec in SPECS],
        "spec_detail": [dict(spec) for spec in SPECS],
        "movement_floor_note": MOVEMENT_FLOOR_NOTE,
    }
    if arguments.import_meep_for_host_policy:
        results["meep_host_import"] = probe.import_meep_for_host_policy()
    # THE OBSERVER GOES IN BEFORE THE POLICY: under 'keep' the policy's strip wraps
    # it, so it records the option tuple NVRTC was really given.
    results["nvrtc_observer"] = probe.install_nvrtc_binary_observer()
    results["subnormal_policy_install"] = probe.install_subnormal_policy_for_run(
        arguments.subnormal_policy, _REPO_API)
    results["environment"] = probe.device_info()
    results["subnormal_policy_stamp"] = probe.subnormal_policy_stamp(_REPO_API)
    policy = arguments.subnormal_policy
    save(results, out_path)

    def wanted(name: str) -> bool:
        return only is None or name in only

    if wanted("driver_order"):
        from meep_gpu.cuda_kernels import (  # noqa: PLC0415
            no_pml_dispersive_fused_electric_pair as family,
        )
        results["driver_order"] = leg_driver_order(family.REPLACES)
        log(f"driver order: {results['driver_order'].get('passed')}")
        save(results, out_path)

    results["warm_memo"] = warm_memo(specs_by_label)
    save(results, out_path)

    if wanted("product"):
        log(f"product: {len(SPECS)} specs x {len(VALUE_CLASSES)} value classes")
        run_product(results, out_path, SPECS, VALUE_CLASSES, arguments.steps)

    if wanted("separate_control"):
        rows = []
        for label in SEPARATE_CONTROL_LABELS:
            row = leg_separate_control(specs_by_label[label], min(arguments.steps, 8))
            rows.append(row)
            results["separate_control"] = rows
            save(results, out_path)
            log(f"  separate control {label}: "
                f"{'OK' if row.get('passed') else 'FAIL'} "
                f"removed={row.get('launches_removed_per_step')}/step")

    if wanted("deposit"):
        rows = []
        for label in DEPOSIT_SPEC_LABELS:
            row = leg_deposit(specs_by_label[label], min(arguments.steps, 8))
            rows.append(row)
            results["deposit"] = rows
            save(results, out_path)
            log(f"  deposit {label}: {'OK' if row.get('passed') else 'FAIL'} "
                f"deposited={row.get('deposited_words')} "
                f"legs={ {k: v.get('differing_words') for k, v in row['legs'].items()} }")

    if wanted("source_mutations"):
        log(f"source mutations: {len(SOURCE_MUTATIONS)}")
        run_mutations(results, out_path, arguments.steps, specs_by_label)
        log(f"host mutations: {len(HOST_MUTATIONS)}")
        run_host_mutations(results, out_path, arguments.steps, specs_by_label)

    results["summary"] = summarize(results, arguments.steps)
    # THE LEDGER'S OWN KEY, spelled here rather than left to a reader of ``summary``:
    # ``rebind_cuda_welds`` binds a weld only where every canonical policy leg carries
    # ``canonical_verdict.released`` true, and it treats a MISSING key as "no verdict"
    # rather than as a failure. A gate reporting its release under some other name
    # would be silently unbindable -- the artifact would look released and nothing in
    # ``fingerprints.json`` would ever point at it.
    results["canonical_verdict"] = {
        "read_from": "summary.released",
        "released": bool(results["summary"]["released"]),
        "reasons": [f"leg {name} did not pass"
                    for name, value in results["summary"]["legs"].items()
                    if not value],
    }
    results["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    save(results, out_path)
    log(json.dumps(results["summary"]["legs"]))
    log(f"released={results['summary']['released']} "
        f"cases={results['summary']['cases_passed']}/{results['summary']['cases']} "
        f"mutations={results['summary']['mutations_caught']}/"
        f"{results['summary']['mutations_scored']}")
    return 0 if results["summary"]["released"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
