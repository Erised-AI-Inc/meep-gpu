"""The device gate for the four Metal residue welds — the board's last two
CANNOT-BIND cells and the slot-UNSELECTED special_kz pair.

WHAT THIS CERTIFIES, AND ON WHAT. Four products landed in one round:

    bfast_fused_electric_pair        BFAST x BFAST                D -> E  (1 corpus row)
    conductive_fused_electric_pair   conductive PML x ordinary    D -> E  (1)
    beta_complex_fused_electric_pair special_kz complex beta      D -> E  (1)
    beta_complex_fused_magnetic_pair special_kz complex beta      B -> H  (1)

The first two are the cells the 2026-09-01 board scored ``UNFUSABLE ON METAL`` (33
and 39 unpacked pointers against a 30-pointer ceiling); each is rebound under the
ceiling by :mod:`meep_gpu.metal_kernels.coefficient_pack` — the curl half's six
vectors for the BFAST pair, BOTH halves' twelve for the conductive one — and this
gate COMPILES the refused unpacked signatures, COMPILES the shipped packed ones,
and BISECTS the pointer ceiling on this host exactly as the cylindrical
precedent's gate did. The other two serve the one corpus row the tranche-6 census
left slot-UNSELECTED (the complex-beta constitutive arm landed 2026-08-19; the
reclose artifact records the corrected selections), welding the certified
``special_kz.beta_bloch_curl`` body to the certified complex constitutive body.

THE COMPARISON IS PER COMPLETE DRIVER STEP, AS uint32 WORDS, over every stored
volume a step can touch — never ``allclose`` and never a single sub-step.

THE LEGS ARE ARMED (a negative leg that does not diverge is VACUOUS and fails):

    A  identity   the full mechanism per case: 0 differing words at every step,
                  the declared launch count, every compared array MOVED, and the
                  reference walk subnormal-free (the flush-policy precondition);
    B  mutation   one arithmetic/index/offset line of the shipped source replaced,
                  compiled, and handed through the plan builder's ``functions``
                  seam. Each MUST diverge; each declared equivalence must NOT.
                  The two packed families each arm a PACK-OFFSET swap, because
                  the offsets are the pack's whole cost and a wrong one is a
                  smooth wrong absorber rather than a crash;
    C  deposit    the in-seam electric source, bracketed by the SHIPPED composer
                  (``plan_step(..., fuse=True)``) and unbracketed. The bracketed
                  walk must be bit-identical with a nonzero repaired-point count;
                  the unbracketed one MUST diverge. The conductive family's walk
                  injects THROUGH THE DRIVER'S OWN CONDUCTIVE BRANCH — the sparse
                  per-deposit-cell condinv replay — imported from the no-PML
                  pairs' gate so the transcription has one home. The magnetic
                  beta-complex pair has NO deposit leg: its seam is empty on its
                  cell and its flag is False, the measured position its host
                  suite pins;
    D  ceiling    the shipped signatures COMPILE; every refuted signature is
                  REFUSED; and the pointer ceiling is BISECTED — a synthetic
                  30-pointer-plus-``Params`` kernel COMPILES and the same kernel
                  with one more pointer is REFUSED with the platform's own error.

Progress reporting: one flushed line per case. Runs locally on MPS, routed through
``metal_gate_runner``. Nothing here is dispatch: ``meep_gpu.fastpath
.plan_fast_path`` still returns ``None`` on every branch.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")
HERE = Path(__file__).resolve().parent
API_ROOT = next(parent for parent in HERE.parents
                if (parent / "meep_gpu" / "metal_kernels").is_dir())
for _path in (str(API_ROOT), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402

import metal_composition_matrix as matrix  # noqa: E402

# THE DRIVER'S CONDUCTIVE INJECTION, transcribed once. The no-PML pairs' gate owns
# the live transcription (sparse per-deposit-cell replay + dense fallback) and its
# lifted_refusal leg holds it against the driver; this gate imports it rather than
# writing a second copy that could drift.
from gate_metal_no_pml_fused_electric_pairs import (  # noqa: E402
    _inject_like_the_driver,
)

from meep_gpu import deposit_repair, stepping  # noqa: E402
from meep_gpu.metal_kernels import (  # noqa: E402
    beta_complex_fused_electric_pair,
    beta_complex_fused_magnetic_pair,
    bfast_fused_electric_pair,
    conductive_fused_electric_pair,
    launch as metal_launch,
    shaders,
    subnormal,
)
from meep_gpu.metal_kernels.device import (  # noqa: E402
    MAX_BUFFER_BINDINGS, Residency, compile_source, metal_frontend_version,
)

#: THIRTEEN, NOT TWELVE, AND THE ODDNESS IS LOAD-BEARING. The BFAST Tustin state
#: on a zero-coefficient target advances by exactly ``-2 * state`` per step —
#: the marginally stable ``(-1)^n`` mode stepping.py:868-874 records — so after
#: any EVEN number of steps it is BYTE-EQUAL to its seed and the every-array-moved
#: floor reads it as untouched (measured on this gate's first smoke:
#: ``f_bfast_Bx``/``f_bfast_Dx`` at k=(0.2,0,0), 0 moved words at 12 steps, moved
#: at every odd step). An odd budget defeats every period-2 mode without
#: weakening the floor.
STEPS = 13
SEED = 20260901

ARRAY_PATH: Dict[str, Callable[[Any, Any], None]] = {
    "step_B": lambda f, p: stepping.step_B(f, p),
    "update_H": lambda f, p: stepping.update_H(f, p),
    "step_D": lambda f, p: stepping.step_D(f, p),
    "update_E": lambda f, p: stepping.update_E(f, p),
    "update_P": lambda f, p: stepping.update_P(f, p),
    "fill_B": lambda f, p: stepping.fill_symmetry_bc_B(f),
    "fill_D": lambda f, p: stepping.fill_symmetry_bc_D(f),
    "zero_metal_B": lambda f, p: stepping.zero_metal_B(f),
    "zero_metal_D": lambda f, p: stepping.zero_metal_D(f),
    "fill_folded_far_ghosts_B": lambda f, p: stepping.fill_folded_far_ghosts_B(f),
    "fill_folded_far_ghosts_D": lambda f, p: stepping.fill_folded_far_ghosts_D(f),
}

#: Every stored volume a complete step can touch, INCLUDING the BFAST Tustin
#: states and the conductive histories: a corrupted state volume reaches the
#: fields one step later, and a comparison blind to them reports on half the
#: mechanism the two new curl halves carry.
STATE_NAMES: Tuple[str, ...] = tuple(
    [f"{stem}{axis}" for stem in ("B", "D", "E", "H") for axis in "xyz"]
    + [f"fu_{stem}{axis}" for stem in ("B", "D") for axis in "xyz"]
    + [f"f_w_{stem}{axis}" for stem in ("E", "H") for axis in "xyz"]
    + [f"f_bfast_{stem}{axis}" for stem in ("B", "D") for axis in "xyz"]
    + [f"f_cond_{stem}{axis}" for stem in ("B", "D") for axis in "xyz"])


def log(message: str) -> None:
    print(message, flush=True)


def words(array: Any) -> np.ndarray:
    return np.frombuffer(np.ascontiguousarray(array).tobytes(), dtype=np.uint32)


def differing(left: Any, right: Any) -> int:
    a, b = words(left), words(right)
    if a.shape != b.shape:
        return max(a.size, b.size)
    return int(np.count_nonzero(a != b))


def state_of(fields: Any) -> Dict[str, Any]:
    return {name: getattr(fields, name) for name in STATE_NAMES
            if getattr(fields, name, None) is not None}


def frozen(fields: Any) -> Dict[str, np.ndarray]:
    return {name: np.array(value, copy=True)
            for name, value in state_of(fields).items()}


def compare(left: Any, right: Any) -> Dict[str, int]:
    a, b = state_of(left), state_of(right)
    assert set(a) == set(b), sorted(set(a) ^ set(b))
    return {name: n for name in sorted(a) if (n := differing(a[name], b[name]))}


def seed_state(fields: Any, seed: int, scale: float = 1.0) -> Any:
    """Physical-band values in every allocated state volume, plus a NON-UNIFORM
    inverse epsilon — `metal_composition_matrix._epsilon` fills one value per
    component, and on that fixture a wrong-cell coefficient read is invisible."""
    rng = np.random.default_rng(seed)
    for name in STATE_NAMES:
        array = getattr(fields, name, None)
        if array is None:
            continue
        real = rng.normal(0.0, 0.37, size=array.shape).astype(np.float32) * scale
        if np.iscomplexobj(array):
            imag = rng.normal(0.0, 0.37, size=array.shape).astype(np.float32) * scale
            array[...] = (real + 1j * imag).astype(array.dtype)
        else:
            array[...] = real.astype(array.dtype)
    rng_eps = np.random.default_rng(seed ^ 0x5EED)
    for component in ("Ex", "Ey", "Ez"):
        reader = getattr(fields, "inverse_epsilon_for", None)
        if not callable(reader):
            continue
        volume = reader(component)
        if volume is None or not getattr(volume, "flags", None):
            continue
        if not volume.flags.writeable:
            continue
        volume[...] = (0.2 + 0.6 * rng_eps.random(volume.shape)).astype(volume.dtype)
    return fields


def live_passes(fields: Any, pml: Any) -> Tuple[str, ...]:
    live = metal_launch.live_sub_steps(fields, pml, ())
    assert live is not None, (
        "the live pass set is unreadable for this configuration; the walk would "
        "silently step a subset and the comparison would certify it")
    return tuple(live)


def needle(source: str, old: str, new: str, count: int = 1) -> str:
    hits = source.count(old)
    if hits != count:
        raise AssertionError(
            f"the mutation needle {old!r} matches {hits} times, expected {count}; "
            f"an unarmed mutation reports its defect as uncaught")
    return source.replace(old, new)


# ---------------------------------------------------------------------------
# The families
# ---------------------------------------------------------------------------

def _conductive_mixed() -> Tuple[Any, Any]:
    """A ONE-SIDED sigma, so the lossless tails compile and run beside the
    conductive one in the same dispatch."""
    fields, pml = matrix.cart()
    shape = tuple(fields.grid.shape)
    fields.set_d_conductivity({"Dx": np.full(shape, np.float32(0.3))})
    return fields, pml


FAMILIES: Tuple[Dict[str, Any], ...] = (
    {
        "name": "bfast_fused_electric_pair",
        "label": "BFAST fused electric D/E pair",
        "module": bfast_fused_electric_pair,
        "plan": bfast_fused_electric_pair.plan_metal_bfast_fused_electric_pair,
        "coverage":
            bfast_fused_electric_pair.metal_bfast_fused_electric_pair_coverage,
        "takes_probe": False,
        "cases": (
            ("bfast_kx_periodic_3d", lambda: matrix.cart(
                bfast_scaled_k=(0.2, 0.0, 0.0))),
            ("bfast_kxz_periodic_3d", lambda: matrix.cart(
                bfast_scaled_k=(0.2, 0.0, 0.3))),
            ("bfast_kx_wall_xyz_3d", lambda: matrix.cart(
                bfast_scaled_k=(0.2, 0.0, 0.0), boundaries="metallic")),
        ),
        "mutation_case": "bfast_kx_wall_xyz_3d",
        "deposit_case": "bfast_kx_periodic_3d",
        "deposit_component": "Ez",
        "source": lambda plan, mode: (
            bfast_fused_electric_pair.bfast_fused_electric_pair_source(
                plan.codes, plan.zero_metal, mode)),
        "entry": "bfast_fused_electric_pair_step",
        "mutations": (
            ("seam_reads_the_split_field_auxiliary",
             "float src0 = v0 * ie0[ii];", "float src0 = u0[ii] * ie0[ii];"),
            ("bfast_state_store_dropped",
             "    s0[ii] = st0 + adv0;\n", ""),
            ("bfast_advance_left_in_the_curl",
             "    curl1 = curl1 - adv1;\n", ""),
            ("pack_offset_x_reads_the_y_vector",
             "device const float* kmx = cpml + prm.off_kmx;",
             "device const float* kmx = cpml + prm.off_kmy;"),
            ("inverse_epsilon_on_the_wrong_component",
             "float src1 = v1 * ie1[ii];", "float src1 = v1 * ie0[ii];"),
        ),
        "equivalences": (
            ("seam_reloads_the_flux_it_just_stored",
             "float src0 = v0 * ie0[ii];", "float src0 = f0[ii] * ie0[ii];"),
        ),
        "refuted": (
            ("unpacked_pointers",
             bfast_fused_electric_pair.refuted_unpacked_pointer_source),
            ("separate_scalars",
             bfast_fused_electric_pair.refuted_separate_scalar_source),
        ),
    },
    {
        "name": "conductive_fused_electric_pair",
        "label": "conductive fused electric D/E pair",
        "module": conductive_fused_electric_pair,
        "plan": (conductive_fused_electric_pair
                 .plan_metal_conductive_fused_electric_pair),
        "coverage": (conductive_fused_electric_pair
                     .metal_conductive_fused_electric_pair_coverage),
        "takes_probe": False,
        "cases": (
            ("sigma_all_periodic_3d", lambda: matrix.conductive(matrix.cart())),
            ("sigma_all_wall_z_3d", lambda: matrix.conductive(
                matrix.cart(boundaries={"z": "metallic"}))),
            ("sigma_dx_only_3d", _conductive_mixed),
        ),
        "mutation_case": "sigma_all_wall_z_3d",
        "deposit_case": "sigma_all_periodic_3d",
        "deposit_component": "Ez",
        "conductive_injection": True,
        "source": lambda plan, mode: (
            conductive_fused_electric_pair.conductive_fused_electric_pair_source(
                plan.codes, plan.conductive, plan.zero_metal, mode)),
        "entry": "conductive_fused_electric_pair_step",
        "mutations": (
            ("seam_reads_the_split_field_auxiliary",
             "float src0 = v0 * ie0[ii];", "float src0 = u0[ii] * ie0[ii];"),
            ("wall_clear_dropped_on_the_reload",
             "    v0 = at_z ? 0.0f : v0;\n", ""),
            ("pack_offset_curl_reads_the_half_integer_lattice",
             "device const float* kmx = cpack + prm.off_kmx;",
             "device const float* kmx = cpack + prm.off_kp0;"),
            ("conductive_history_drops_condfac", 2,
             "float c0_new = ((c0_previous * cf0[ii]) - curl0) * ci0[ii];",
             "float c0_new = ((c0_previous) - curl0) * ci0[ii];"),
            ("inverse_epsilon_on_the_wrong_component",
             "float src1 = v1 * ie1[ii];", "float src1 = v1 * ie0[ii];"),
        ),
        "equivalences": (
            ("seam_reloads_the_stored_flux",
             "float src0 = v0 * ie0[ii];", "float src0 = f0[ii] * ie0[ii];"),
        ),
        "refuted": (
            ("unpacked_pointers",
             conductive_fused_electric_pair.refuted_unpacked_pointer_source),
            ("separate_scalars",
             conductive_fused_electric_pair.refuted_separate_scalar_source),
        ),
    },
    {
        "name": "beta_complex_fused_electric_pair",
        "label": "complex-beta fused electric D/E pair",
        "module": beta_complex_fused_electric_pair,
        "plan": (beta_complex_fused_electric_pair
                 .plan_metal_beta_complex_fused_electric_pair),
        "coverage": (beta_complex_fused_electric_pair
                     .metal_beta_complex_fused_electric_pair_coverage),
        "takes_probe": True,
        "cases": (
            ("beta_complex_2d", lambda: matrix.flat(beta=0.33,
                                                    complex_storage=True)),
            ("beta_complex_negative_2d", lambda: matrix.flat(
                beta=-0.685, complex_storage=True)),
            ("beta_complex_bloch_2d", lambda: matrix.flat(
                beta=0.33, complex_storage=True, k_point=(0.3, 0.2, 0.0))),
            ("beta_complex_wall_x_2d", lambda: matrix.flat(
                beta=0.33, complex_storage=True,
                boundaries={"x": "metallic"})),
        ),
        "mutation_case": "beta_complex_bloch_2d",
        "deposit_case": "beta_complex_2d",
        "deposit_component": "Ez",
        "source": lambda plan, mode: (
            beta_complex_fused_electric_pair
            .beta_complex_fused_electric_pair_source(
                plan.codes, plan.phased, plan.zero_metal, plan.expansion, mode)),
        "entry": "beta_complex_fused_electric_pair_step",
        "mutations": (
            ("seam_reads_the_split_field_auxiliary",
             "float2 src0 = c_mul_field_left(v0, e0[ii]);",
             "float2 src0 = c_mul_field_left(u0[ii], e0[ii]);"),
            ("beta_insert_dropped",
             "    curl0 = curl0 - c_mul(float2(bpr, bpi), b);\n", ""),
            ("beta_partners_swapped",
             "curl1 = curl1 - c_mul(float2(bmr, bmi), a);",
             "curl1 = curl1 - c_mul(float2(bmr, bmi), b);"),
            ("bloch_phase_dropped_on_one_operand",
             "b_x = wx ? c_mul(b_x, float2(pxr, pxi)) : b_x;",
             "b_x = b_x;"),
        ),
        "equivalences": (
            ("seam_reloads_the_flux_it_just_stored",
             "float2 src0 = c_mul_field_left(v0, e0[ii]);",
             "float2 src0 = c_mul_field_left(f0[ii], e0[ii]);"),
        ),
        "refuted": (
            ("separate_scalars",
             beta_complex_fused_electric_pair.refuted_separate_scalar_source),
        ),
    },
    {
        "name": "beta_complex_fused_magnetic_pair",
        "label": "complex-beta fused magnetic B/H pair",
        "module": beta_complex_fused_magnetic_pair,
        "plan": (beta_complex_fused_magnetic_pair
                 .plan_metal_beta_complex_fused_magnetic_pair),
        "coverage": (beta_complex_fused_magnetic_pair
                     .metal_beta_complex_fused_magnetic_pair_coverage),
        "takes_probe": True,
        "cases": (
            ("beta_complex_2d", lambda: matrix.flat(beta=0.33,
                                                    complex_storage=True)),
            ("beta_complex_negative_2d", lambda: matrix.flat(
                beta=-0.685, complex_storage=True)),
            ("beta_complex_bloch_2d", lambda: matrix.flat(
                beta=0.33, complex_storage=True, k_point=(0.3, 0.2, 0.0))),
            ("beta_complex_wall_x_2d", lambda: matrix.flat(
                beta=0.33, complex_storage=True,
                boundaries={"x": "metallic"})),
        ),
        "mutation_case": "beta_complex_bloch_2d",
        # NO DEPOSIT LEG: this cell's magnetic seam is EMPTY (its one corpus row
        # declares source_field_types == ['D']) and the flag is False — the
        # measured position its host suite pins. A magnetic deposit walk here
        # would certify a bracket the module deliberately does not claim.
        "deposit_case": None,
        "deposit_component": None,
        "source": lambda plan, mode: (
            beta_complex_fused_magnetic_pair
            .beta_complex_fused_magnetic_pair_source(
                plan.codes, plan.phased, plan.zero_metal, plan.expansion, mode)),
        "entry": "beta_complex_fused_magnetic_pair_step",
        "mutations": (
            ("seam_takes_the_curl_auxiliary",
             "float2 src0 = v0;", "float2 src0 = u0[ii];"),
            ("beta_insert_dropped",
             "    curl0 = curl0 - c_mul(float2(bpr, bpi), b);\n", ""),
            ("beta_partners_swapped",
             "curl1 = curl1 - c_mul(float2(bmr, bmi), a);",
             "curl1 = curl1 - c_mul(float2(bmr, bmi), b);"),
        ),
        "equivalences": (
            ("seam_reloads_the_flux_it_just_stored",
             "float2 src0 = v0;", "float2 src0 = f0[ii];"),
        ),
        "refuted": (
            ("separate_scalars",
             beta_complex_fused_magnetic_pair.refuted_separate_scalar_source),
        ),
    },
)


def build_plan(spec: Mapping[str, Any], fields: Any, pml: Any,
               residency: Residency, probes: Mapping[str, Any],
               sources: Sequence[Any] = (),
               functions: Optional[Mapping[Any, Any]] = None) -> Any:
    kwargs: Dict[str, Any] = {}
    if spec["takes_probe"]:
        kwargs["probe"] = probes["beta_probe"]
    if functions is not None:
        kwargs["functions"] = functions
    return spec["plan"](fields, pml, sources=tuple(sources),
                        residency=residency, **kwargs)


def _coverage_kwargs(spec: Mapping[str, Any],
                     probes: Mapping[str, Any]) -> Dict[str, Any]:
    return {"probe": probes["beta_probe"]} if spec["takes_probe"] else {}


# ---------------------------------------------------------------------------
# Leg A / B: the walk
# ---------------------------------------------------------------------------

def metal_step(fields: Any, pml: Any, plan: Any, owned: Sequence[str],
               residency: Residency, live: Sequence[str]) -> None:
    """One complete step, on the device where the plan owns the pass; a pass no
    plan owns runs on the array path bracketed by sync_out / sync_in."""
    skip = set(owned)
    for name in live:
        if name in skip:
            if name == owned[0]:
                plan.run()
            continue
        residency.sync_out()
        ARRAY_PATH[name](fields, pml)
        residency.sync_in()


def run_case(spec: Mapping[str, Any], build: Callable[[], Tuple[Any, Any]],
             probes: Mapping[str, Any], steps: Optional[int] = None,
             functions: Optional[Mapping[Any, Any]] = None) -> Dict[str, Any]:
    """Step the two engines side by side and compare per COMPLETE step."""
    steps = STEPS if steps is None else int(steps)
    reference, reference_pml = build()
    actual, actual_pml = build()
    seed_state(reference, SEED)
    seed_state(actual, SEED)

    drift = compare(reference, actual)
    if drift:
        return {"passed": False, "reason": "the two builds are not identical",
                "drift": drift}

    residency = Residency()
    plan = build_plan(spec, actual, actual_pml, residency, probes,
                      functions=functions)
    if plan is None:
        return {"passed": False, "reason": "the pair was refused",
                "refusals": list(spec["coverage"](
                    actual, actual_pml, (), residency,
                    **_coverage_kwargs(spec, probes)).reasons)[:8]}

    live = live_passes(actual, actual_pml)
    assert live == live_passes(reference, reference_pml)
    before = frozen(actual)
    residency.sync_in()

    per_step: List[Dict[str, Any]] = []
    for step in range(1, steps + 1):
        for name in live:
            ARRAY_PATH[name](reference, reference_pml)
        metal_step(actual, actual_pml, plan, plan.replaces_sub_steps, residency,
                   live)
        residency.sync_out()
        difference = compare(reference, actual)
        census = sum(subnormal.census(np.real(value))
                     + subnormal.census(np.imag(value))
                     if np.iscomplexobj(value) else subnormal.census(value)
                     for value in state_of(reference).values())
        per_step.append({"step": step,
                         "differing_words": sum(difference.values()),
                         "differing_arrays": dict(sorted(difference.items())),
                         "reference_subnormals": int(census)})
        if difference or census:
            break

    after = state_of(actual)
    moved = {name: differing(before[name], after[name]) for name in before}
    still = sorted(name for name, count in moved.items() if count == 0)
    identical = (len(per_step) == steps
                 and all(row["differing_words"] == 0 for row in per_step))
    clean = all(row["reference_subnormals"] == 0 for row in per_step)
    launches_ok = (plan.runs == len(per_step)
                   and plan.launches == plan.launches_per_run * len(per_step))
    return {
        "passed": bool(identical and clean and launches_ok and not still),
        "bit_identical": identical,
        "subnormal_free": clean,
        "first_divergence": next((row["step"] for row in per_step
                                  if row["differing_words"]), None),
        "steps_compared": len(per_step),
        "differing_words": per_step[-1]["differing_words"],
        "differing_arrays": per_step[-1]["differing_arrays"],
        "arrays_compared": len(before),
        "arrays_that_never_moved": still,
        "moved_words": int(sum(moved.values())),
        "runs": plan.runs, "launches": plan.launches,
        "launches_per_run": plan.launches_per_run,
        "live_passes": list(live),
        "replaces": list(plan.replaces_sub_steps),
        "shape": list(plan.shape),
        "mirrors": len(residency.names),
    }


def leg_identity(spec: Mapping[str, Any],
                 probes: Mapping[str, Any]) -> Dict[str, Any]:
    """A: the full mechanism. Every case must be bit-identical at every step."""
    rows: Dict[str, Any] = {}
    for label, build in spec["cases"]:
        started = time.time()
        rows[label] = run_case(spec, build, probes)
        log(f"    identity  {spec['name']:36s} {label:26s} "
            f"passed={rows[label]['passed']} "
            f"words={rows[label].get('differing_words')} "
            f"launches={rows[label].get('launches')} "
            f"({time.time() - started:5.1f} s)")
    return {"passed": all(row["passed"] for row in rows.values()), "cases": rows}


def _mutated_functions(spec: Mapping[str, Any], plan: Any, old: str, new: str,
                       count: int = 1) -> Dict[Any, Any]:
    mode = shaders.CONTRACT_OFF
    source = needle(spec["source"](plan, mode), old, new, count)
    function = getattr(compile_source(source), spec["entry"])
    return {mode: function}


def leg_mutation(spec: Mapping[str, Any],
                 probes: Mapping[str, Any]) -> Dict[str, Any]:
    """B: the DEGENERATE mechanism. Every armed mutation MUST diverge; every
    declared equivalence must NOT."""
    build = dict(spec["cases"])[spec["mutation_case"]]
    rows: Dict[str, Any] = {}
    for mutation in spec["mutations"]:
        if len(mutation) == 4:
            label, count, old, new = mutation
        else:
            label, old, new = mutation
            count = 1
        reference, reference_pml = build()
        residency = Residency()
        seed_state(reference, SEED)
        plan = build_plan(spec, reference, reference_pml, residency, probes)
        if plan is None:
            rows[label] = {"passed": False, "reason": "the pair was refused"}
            continue
        functions = _mutated_functions(spec, plan, old, new, count)
        result = run_case(spec, build, probes, functions=functions)
        diverged = not result.get("bit_identical", True)
        rows[label] = {
            "passed": bool(diverged and result.get("launches")),
            "caught": diverged,
            "first_divergence": result.get("first_divergence"),
            "differing_words": result.get("differing_words"),
            "launches": result.get("launches"),
        }
        log(f"    mutation  {spec['name']:36s} {label:44s} "
            f"caught={diverged} words={result.get('differing_words')}")

    equivalences: Dict[str, Any] = {}
    for label, old, new in spec.get("equivalences", ()):
        reference, reference_pml = build()
        residency = Residency()
        seed_state(reference, SEED)
        plan = build_plan(spec, reference, reference_pml, residency, probes)
        if plan is None:
            equivalences[label] = {"passed": False,
                                   "reason": "the pair was refused"}
            continue
        functions = _mutated_functions(spec, plan, old, new)
        result = run_case(spec, build, probes, functions=functions)
        identical = bool(result.get("bit_identical"))
        equivalences[label] = {
            "passed": identical, "bit_identical": identical,
            "differing_words": result.get("differing_words"),
            "launches": result.get("launches"),
        }
        log(f"    equivalent{spec['name']:36s} {label:44s} "
            f"identical={identical} words={result.get('differing_words')}")
    return {"passed": (all(row["passed"] for row in rows.values())
                       and all(row["passed"] for row in equivalences.values())),
            "mutations": rows, "declared_equivalences": equivalences}


# ---------------------------------------------------------------------------
# Leg C: the deposit, with and WITHOUT the repair bracket
# ---------------------------------------------------------------------------

def _volume_source(fields: Any, component: str) -> Any:
    from meep_gpu.sources import ContinuousEnvelope, VolumeSource  # noqa: PLC0415

    return VolumeSource(grid=fields.grid, component=component,
                        center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                        envelope=ContinuousEnvelope(frequency=1.0,
                                                    is_integrated=False))


def _withdraw(sources: Sequence[Any], fields: Any) -> None:
    for source in sources:
        hook = getattr(source, "withdraw", None)
        if callable(hook):
            hook(fields)


def _inject_plain(fields: Any, sources: Sequence[Any], when: float) -> None:
    for source in sources:
        if source.field_type != "B":
            source.inject(fields, when)


def run_deposit_case(spec: Mapping[str, Any], probes: Mapping[str, Any],
                     repair: bool = True,
                     steps: Optional[int] = None) -> Dict[str, Any]:
    """Complete driver steps with an in-seam electric source, byte compared.

    THE WALK IS THE DRIVER'S: the injection lands between the fused launch and
    the trailing repair — through the driver's OWN conductive branch on the
    conductive family (the sparse condinv replay) and through the plain loop
    everywhere else — and every in-seam host pass the driver runs
    unconditionally runs on the host after it. ``repair=False`` builds the
    SHIPPED plan for an EMPTY source list — the bare pair plus the ``NoopPlan``
    — and injects anyway: the no-mechanism control, which MUST diverge.
    """
    steps = STEPS if steps is None else int(steps)
    build = dict(spec["cases"])[spec["deposit_case"]]
    inject = (_inject_like_the_driver if spec.get("conductive_injection")
              else _inject_plain)
    reference, reference_pml = build()
    actual, actual_pml = build()
    seed_state(reference, SEED)
    seed_state(actual, SEED)
    drift = compare(reference, actual)
    if drift:
        return {"passed": False, "reason": "the two builds are not identical"}

    reference_sources = (_volume_source(reference, spec["deposit_component"]),)
    actual_sources = (_volume_source(actual, spec["deposit_component"]),)
    in_seam = deposit_repair.in_seam_sources(actual_sources, "D")
    if len(in_seam) != 1:
        return {"passed": False,
                "reason": "the source is not in the D seam; the walk would "
                          "inject nothing and compare a no-op with a no-op"}

    residency = Residency()
    composed = metal_launch.plan_step(
        actual, actual_pml, residency=residency,
        sources=(actual_sources if repair else ()), fuse=True,
        beta_probe=probes["beta_probe"],
        complex_probe=probes["complex_probe"],
        folded_complex_probe=probes["folded_complex_probe"],
        cylindrical_complex_probe=probes["cylindrical_complex_probe"])
    leading = composed.plans.get("step_D")
    trailing = composed.plans.get("update_E")
    if leading is None or trailing is None:
        return {"passed": False, "reason": "the composer did not fuse the seam",
                "selected": dict(composed.selected),
                "refusals": [r for key, value in composed.reasons.items()
                             if key.startswith("fused_pair") for r in value][:6]}
    installed = (type(leading).__name__, type(trailing).__name__)
    if repair:
        installed_ok = installed == ("LeadingRepairPlan", "TrailingRepairPlan")
        installed_ok = installed_ok and (
            tuple(leading.repair_paths) == (deposit_repair.SPLIT_FIELD_PATH,))
    else:
        installed_ok = (installed[0] != "LeadingRepairPlan"
                        and installed[1] == "NoopPlan")
    inner = getattr(leading, "absorbed_by", leading)
    if composed.selected.get("step_D") != spec["label"]:
        return {"passed": False,
                "reason": f"another product won step_D: "
                          f"{composed.selected.get('step_D')!r}"}

    live = live_passes(actual, actual_pml)
    before = frozen(actual)
    dt = float(actual.grid.dt)

    per_step: List[Dict[str, Any]] = []
    for step in range(1, steps + 1):
        when = (step - 1) * dt
        _withdraw(reference_sources, reference)
        for name in live:
            ARRAY_PATH[name](reference, reference_pml)
            if name == "step_D":
                inject(reference, reference_sources, when)

        _withdraw(actual_sources, actual)
        for name in live:
            if name == "step_D":
                residency.sync_in()
                leading.run()
                residency.sync_out()
                inject(actual, actual_sources, when)
            elif name == "update_E":
                trailing.run()
            elif name == "zero_metal_D":
                # UNCONDITIONAL in the driver, even though the kernel carries
                # the clear inline; idempotent, and what lets the repair read a
                # final field.
                ARRAY_PATH[name](actual, actual_pml)
            elif name in set(inner.replaces_sub_steps):
                continue
            else:
                ARRAY_PATH[name](actual, actual_pml)

        difference = compare(reference, actual)
        per_step.append({"step": step, "differing_words": sum(difference.values()),
                         "differing_arrays": dict(sorted(difference.items()))})
        if difference:
            break

    after = state_of(actual)
    moved = {name: differing(before[name], after[name]) for name in before}
    still = sorted(name for name, count in moved.items() if count == 0)
    identical = (len(per_step) == steps
                 and all(row["differing_words"] == 0 for row in per_step))
    repairs = int(getattr(leading, "repairs", 0))
    launches_ok = (inner.runs == len(per_step)
                   and inner.launches == inner.launches_per_run * len(per_step))
    return {
        "passed": bool(identical and launches_ok and installed_ok
                       and (repairs > 0 if repair else repairs == 0)
                       and not still),
        "repair_wired": repair,
        "installed_plans": list(installed),
        "installed_as_expected": installed_ok,
        "repair_paths": list(getattr(leading, "repair_paths", ())),
        "deposit_points_repaired": repairs,
        "bit_identical": identical,
        "first_divergence": next((row["step"] for row in per_step
                                  if row["differing_words"]), None),
        "differing_words": per_step[-1]["differing_words"],
        "differing_arrays": per_step[-1]["differing_arrays"],
        "arrays_that_never_moved": still,
        "runs": inner.runs, "launches": inner.launches,
        "launches_per_run": inner.launches_per_run,
        "selected": {slot: composed.selected.get(slot)
                     for slot in ("step_D", "update_E")},
    }


def leg_deposit(spec: Mapping[str, Any],
                probes: Mapping[str, Any]) -> Dict[str, Any]:
    """C: the in-seam deposit, bracketed and unbracketed."""
    carried = run_deposit_case(spec, probes, repair=True)
    control = run_deposit_case(spec, probes, repair=False)
    diverged = not control.get("bit_identical", True)
    log(f"    deposit   {spec['name']:36s} carried={carried['passed']} "
        f"words={carried.get('differing_words')} "
        f"repaired={carried.get('deposit_points_repaired')} "
        f"| control diverged={diverged}")
    return {"passed": bool(carried.get("passed") and diverged),
            "vacuous": not diverged,
            "carried": carried, "null_control": control}


# ---------------------------------------------------------------------------
# Leg D: the binding ceiling, compiled and BISECTED
# ---------------------------------------------------------------------------

def _synthetic_pointer_signature(pointers: int) -> str:
    """``pointers`` device buffers plus one packed ``Params&`` — the bisect probe.

    The body reads every input and writes the first, so dead-code elimination
    cannot decide the answer; what is measured is the SIGNATURE.
    """
    lines = [f"    device float* b{index} [[buffer({index})]],"
             for index in range(pointers)]
    lines.append(f"    constant Params& prm [[buffer({pointers})]],")
    touch = " + ".join(f"b{index}[0]" for index in range(1, pointers))
    return "\n".join((
        "#include <metal_stdlib>",
        "using namespace metal;",
        "struct Params { uint n_elem; float dtdx; };",
        f"kernel void bisect_{pointers}(",
        *lines,
        "    uint idx [[thread_position_in_grid]])",
        "{",
        "    if (idx >= prm.n_elem) { return; }",
        f"    b0[idx] = b0[idx] * prm.dtdx + {touch};",
        "}",
        "",
    ))


def leg_binding_ceiling() -> Dict[str, Any]:
    """D: shipped signatures COMPILE, refuted ones are REFUSED, and the pointer
    ceiling is BISECTED on this host rather than cited."""
    rows: Dict[str, Any] = {}
    shipped = {
        "bfast_fused_electric_pair":
            bfast_fused_electric_pair.bfast_fused_electric_pair_source(
                (0, 0, 1), (False, False, True)),
        "conductive_fused_electric_pair":
            conductive_fused_electric_pair.conductive_fused_electric_pair_source(
                (0, 0, 1), (True, True, True), (False, False, True)),
        "beta_complex_fused_electric_pair":
            beta_complex_fused_electric_pair
            .beta_complex_fused_electric_pair_source(
                (0, 0, 0), (1, 0, 0), (False, False, False), "FMA_V1"),
        "beta_complex_fused_magnetic_pair":
            beta_complex_fused_magnetic_pair
            .beta_complex_fused_magnetic_pair_source(
                (0, 0, 0), (1, 0, 0), (False, False, False), "FMA_V1"),
    }
    declared = {
        "bfast_fused_electric_pair": bfast_fused_electric_pair.PACKED_BINDINGS,
        "conductive_fused_electric_pair":
            conductive_fused_electric_pair.PACKED_BINDINGS,
        "beta_complex_fused_electric_pair":
            beta_complex_fused_electric_pair.PACKED_BINDINGS,
        "beta_complex_fused_magnetic_pair":
            beta_complex_fused_magnetic_pair.PACKED_BINDINGS,
    }
    import re as _re
    binding = _re.compile(r"\[\[buffer\((\d+)\)\]\]")
    for name, source in shipped.items():
        slots = sorted({int(number) for number in binding.findall(source)})
        try:
            compile_source(source)
            compiled, error = True, ""
        except Exception as exc:  # noqa: BLE001
            compiled, error = False, str(exc).strip().splitlines()[-1][:160]
        rows[f"{name}/shipped"] = {
            "passed": bool(compiled and len(slots) == declared[name]
                           and slots == list(range(len(slots)))
                           and len(slots) <= MAX_BUFFER_BINDINGS),
            "compiled": compiled, "error": error,
            "bindings_parsed": len(slots), "bindings_declared": declared[name],
        }
        log(f"    ceiling   {name}/shipped bindings={len(slots)} "
            f"compiled={compiled}")
    for spec in FAMILIES:
        for label, builder in spec.get("refuted", ()):
            key = f"{spec['name']}/{label}"
            source = builder()
            slots = sorted({int(number) for number in binding.findall(source)})
            try:
                compile_source(source)
                rows[key] = {"passed": False, "compiled": True,
                             "bindings_parsed": len(slots)}
            except Exception as error:  # noqa: BLE001 - the failure IS the result
                rows[key] = {"passed": True, "compiled": False,
                             "bindings_parsed": len(slots),
                             "error": str(error).strip().splitlines()[-1][:160]}
            log(f"    ceiling   {key:64s} refused={not rows[key]['compiled']}")
    # THE BISECT: MAX_POINTERS with Params compiles; one more pointer is refused
    # with the platform's own error. This is the measurement that licenses BOTH
    # packs, re-taken on this host rather than cited from the cylindrical round.
    bisect: Dict[str, Any] = {}
    for pointers, must_compile in ((MAX_BUFFER_BINDINGS - 1, True),
                                   (MAX_BUFFER_BINDINGS, False)):
        source = _synthetic_pointer_signature(pointers)
        try:
            compile_source(source)
            compiled, error = True, ""
        except Exception as exc:  # noqa: BLE001
            compiled, error = False, str(exc).strip().splitlines()[-1][:160]
        bisect[f"{pointers}_pointers_plus_params"] = {
            "passed": compiled is must_compile,
            "compiled": compiled, "must_compile": must_compile, "error": error,
        }
        log(f"    bisect    {pointers} pointers + Params  compiled={compiled} "
            f"(must_compile={must_compile})")
    rows["ceiling_bisected_on_this_host"] = {
        "passed": all(row["passed"] for row in bisect.values()),
        "probes": bisect,
        "max_buffer_bindings": MAX_BUFFER_BINDINGS,
    }
    return {"passed": all(row["passed"] for row in rows.values()),
            "signatures": rows}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    global STEPS

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument("--only", default=None, help="run one family by name")
    parser.add_argument("--legs", default="identity,mutation,deposit,ceiling")
    parser.add_argument("--steps", type=int, default=STEPS)
    arguments = parser.parse_args()
    STEPS = int(arguments.steps)

    environment = matrix.prepare_environment()
    import torch  # noqa: PLC0415
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available; this gate must run on an Apple GPU")

    # THE PROBES ARE LOADED RECORDS, NOT PATHS — handing a family's
    # `expansion_from_probe` a path returns None and the family refuses BY NAME,
    # which would make this gate measure the artifact's absence.
    from meep_gpu.metal_kernels import complex_fields as _complex  # noqa: PLC0415
    from meep_gpu.metal_kernels import folded_complex as _folded  # noqa: PLC0415
    from meep_gpu.metal_kernels import special_kz as _beta  # noqa: PLC0415
    from meep_gpu.metal_kernels import (  # noqa: PLC0415
        cylindrical_complex as _cylindrical)

    paths = {
        "complex_probe":
            environment.get("MEEP_GPU_METAL_COMPLEX_EXPANSION_PROBE"),
        "beta_probe": environment.get("MEEP_GPU_METAL_EXPANSION_PROBE"),
        "folded_complex_probe":
            environment.get("MEEP_GPU_METAL_FOLDED_COMPLEX_EXPANSION_PROBE"),
        "cylindrical_complex_probe":
            environment.get("MEEP_GPU_METAL_CYLINDRICAL_COMPLEX_EXPANSION_PROBE"),
    }
    if paths["beta_probe"] is None:
        raise SystemExit(
            "no beta expansion artifact; the two special_kz families would "
            "refuse BY NAME and this gate would measure the artifact's absence "
            "rather than the kernels' arithmetic")
    probes = {
        "complex_probe": _complex.load_expansion_probe(),
        "beta_probe": _beta.load_expansion_probe(),
        "folded_complex_probe": _folded.load_expansion_probe(),
        "cylindrical_complex_probe": _cylindrical.load_expansion_probe(),
    }
    if probes["beta_probe"] is None:
        raise SystemExit(
            f"the beta expansion artifact at {paths['beta_probe']!r} did not load")

    legs = tuple(part.strip() for part in arguments.legs.split(",") if part.strip())
    started = time.time()
    record: Dict[str, Any] = {
        "gate": "metal_residue_fused_pairs",
        "host": {"platform": sys.platform, "torch": torch.__version__,
                 "mps_available": bool(torch.backends.mps.is_available())},
        "steps": arguments.steps,
        "seed": SEED,
        "probes": paths,
        "families": {},
        "corpus_rows": {
            "census": "parity/meep_gpu/results/metal_coverage_tranche6_2026-08-19"
                      " (+ the special_kz reclose substitution)",
            "bfast_fused_electric_pair": {
                "cell": "D_to_E (BFAST, BFAST)", "seam_instances": 1,
                "rows": ["tests:TestReflectanceAngular"
                         ".test_reflectance_angular_2_35_7"]},
            "conductive_fused_electric_pair": {
                "cell": "D_to_E (conductive PML curl, ordinary)",
                "seam_instances": 1,
                "rows": ["tests:TestAdjointSolver.test_damping"]},
            "beta_complex_fused_electric_pair": {
                "cell": "D_to_E (special_kz complex beta, special_kz complex "
                        "beta)", "seam_instances": 1,
                "rows": ["tests:TestSpecialKz.test_special_kz"]},
            "beta_complex_fused_magnetic_pair": {
                "cell": "B_to_H (special_kz complex beta, special_kz complex "
                        "beta)", "seam_instances": 1,
                "rows": ["tests:TestSpecialKz.test_special_kz"]},
        },
    }

    for spec in FAMILIES:
        if arguments.only and spec["name"] != arguments.only:
            continue
        log(f"  {spec['name']}")
        entry: Dict[str, Any] = {}
        if "identity" in legs:
            entry["identity"] = leg_identity(spec, probes)
        if "mutation" in legs:
            entry["mutation"] = leg_mutation(spec, probes)
        if "deposit" in legs and spec.get("deposit_case"):
            entry["deposit"] = leg_deposit(spec, probes)
        entry["passed"] = all(value.get("passed") for value in entry.values()
                              if isinstance(value, dict))
        record["families"][spec["name"]] = entry

    if "ceiling" in legs:
        record["binding_ceiling"] = leg_binding_ceiling()

    record["torch_version"] = torch.__version__
    record["numpy_version"] = np.__version__
    record["metal_frontend"] = metal_frontend_version()
    record["subnormal_policy"] = os.environ.get("MEEP_GPU_SUBNORMAL_POLICY")
    record["elapsed_s"] = round(time.time() - started, 2)
    record["verdict"] = "PASS" if all(
        value.get("passed") for value in record["families"].values()) and (
        record.get("binding_ceiling", {}).get("passed", True)) else "FAIL"
    from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
    _stamp_provenance(record)  # bytes THIS process imported; see gate_provenance
    text = json.dumps(record, indent=1, sort_keys=True)
    if arguments.out:
        path = Path(arguments.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")
        log(f"  wrote {path}")
    else:
        print(text)
    log(f"  VERDICT {record['verdict']}  ({record['elapsed_s']} s)")
    return 0 if record["verdict"] == "PASS" else 1


if __name__ == "__main__":
    # ROUTED THROUGH THE RELEASE RUNNER, as every `gate_metal_*.py` is: the runner
    # re-derives the imported source set, welds it against `source_sha256.txt`,
    # and only then records `release.released`.
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))
