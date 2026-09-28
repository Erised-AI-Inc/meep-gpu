"""Byte-identity gate for the CONDUCTIVE hand-CUDA fused pair on the electric seam.

WHAT IS UNDER TEST. ``meep_gpu/cuda_kernels/conductive_fused_electric_pair.py``: ONE
kernel performing three driver passes in one launch, where the curl is the
THREE-HISTORY conductive-PML recurrence::

    step_D (driver.py:3306) -> zero_metal_D (:3310) -> update_E (:3313)

THE CELL IS ``D_to_E (cuda_conductive/conductive, cuda_constitutive/ordinary)`` and it
owns exactly one corpus row -- ``tests:TestAdjointSolver.test_damping``, which declares
BOTH a D and a B source, is METALLIC on x and y, carries an ACTIVE absorber and a
conductivity, and has no susceptibility.

=============================================================================
THIS IS THE SIGNED-ZERO ROW, AND THAT IS WHAT MAKES THIS GATE DIFFERENT
=============================================================================

``FdtdDriver._inject_electric_through_conductivity`` used to apply its ``condinv``
scaling as three WHOLE-VOLUME passes, which are the identity at every finite value
EXCEPT that they canonicalise ``-0.0`` to ``+0.0`` at every cell of the component --
and on a device that flushes subnormals also ZERO every non-deposit subnormal word.
Both rewrites land where no deposit closure can name them, so the fused electric pairs
on three backends refused conductive rows BY NAME and this cell stood empty on every
backend.

THE DRIVER NOW REPLAYS THAT RESCALE SPARSELY at the published deposit indices.
:func:`leg_rescale` is where this gate re-measures that ON A DEVICE rather than
inheriting it:

* the SHIPPED injection must leave every non-deposit word UNTOUCHED -- measured on a
  fixture seeded with all four word classes and an exactly-zero curl, so a ``-0.0``
  survives ``step_D`` to the injection point;
* the RETIRED whole-volume composition, replayed as a control, MUST diverge, and every
  differing word must classify as SIGNED-ZERO or SUBNORMAL. The classifier admits BOTH
  classes deliberately: the published CuPy figure is 68 subnormal beside 54
  signed-zero words of 588, and a classifier that knew only about signed zeros would
  report an unexplained divergence on a device that flushes.

UNLIKE THE OFF-DEVICE PROBE, BOTH CLASSES CAN ARM HERE. NumPy does not flush
subnormals, so the probe recorded that half ``predicted_null`` with its reason; under
the ``flush`` policy this gate can reach it, and the leg records which classes actually
armed under which policy rather than assuming either.

=============================================================================
WHY THE COMPARISON IS PER COMPLETE DRIVER STEP
=============================================================================

Two engines are built from one seed and stepped side by side, compared as raw uint32
WORDS after EVERY step over every stored volume the seam touches -- ``D``, the
split-field auxiliary ``fu_D``, the conductive third history ``f_cond_D``, ``E`` and
``f_w_E``. THE TWO CURL HISTORIES ARE IN THE COMPARISON and that is not incidental:
the weld turns both certified helpers into VALUES, and a rewrite that swallowed one of
their stores would leave the recurrence a step behind at every cell while every
seam-order check still passed.

=============================================================================
WHAT THIS GATE REFUSES TO LET PASS SILENTLY
=============================================================================

1. A step this gate invented -- the driver-order leg reads ``FdtdDriver.step``'s source.
2. A launch that did not happen -- two INDEPENDENT counters must agree with the budget.
3. An overlap -- every OTHER ``step_D`` product is asked about each fixture, and one
   that admits it fails the case: two admitters leave the seam UNFUSED naming both,
   which costs the 79 rows the real twin serves.
4. A mutation that was never armed, or a leg whose case never reached a comparison.
5. A case that measured nothing -- the movement floor requires every volume the class
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
import gate_cuda_fused_electric_pair as base  # noqa: E402

import cupy as cp  # noqa: E402

from meep_gpu import deposit_repair, stepping  # noqa: E402
from meep_gpu.driver import FdtdDriver  # noqa: E402
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
STEPS = 60

#: The three passes ONE launch performs.
FUSED_PASSES: Tuple[str, ...] = ("step_D", "zero_metal_D", "update_E")

_TARGETS: Tuple[str, ...] = ("Dx", "Dy", "Dz")
_ELECTRIC: Tuple[str, ...] = ("Ex", "Ey", "Ez")

#: Every stored volume a complete step on this path touches. BOTH CURL HISTORIES ARE
#: IN IT: a value rewrite that swallowed either store would be invisible otherwise.
STATE_NAMES: Tuple[str, ...] = (
    _TARGETS + tuple("fu_" + name for name in _TARGETS)
    + tuple("f_cond_" + name for name in _TARGETS)
    + _ELECTRIC + tuple("f_w_" + name for name in _ELECTRIC)
    + ("Bx", "By", "Bz", "Hx", "Hy", "Hz"))

#: The families the SUBNORMAL BAND class is held to.
BAND_MUST_MOVE: Tuple[str, ...] = _TARGETS + _ELECTRIC

MOVEMENT_FLOOR_NOTE = (
    "A case whose volumes never moved measured nothing. On the uniform class every "
    "volume in the comparison is required to move, INCLUDING f_cond: it is the "
    "conductive recurrence's third history and a run where it never moved is a run "
    "with no conductivity in it, which is the whole subject of this family.")

CELL: Tuple[float, float, float] = (9.0, 10.0, 11.0)

#: The sweep. THE FIRST ROW IS THE CORPUS SHAPE, read off the stamped census:
#: TestAdjointSolver.test_damping is METALLIC on x and y, periodic on z, with an
#: ACTIVE absorber and a conductivity on every D component. The rest arm what it
#: cannot: every axis walled somewhere, one row walled nowhere, MIXED conductive masks
#: (``conductive_targets`` is per COMPONENT and a run may be lossy on one target and
#: not another), an ASYMMETRIC absorber (so the integer and half-integer sigma
#: sub-lattices differ on every axis and the rename is a defect that is PRESENT rather
#: than one the profile hides), and half the rows diagonally ANISOTROPIC.
SPECS: Tuple[Dict[str, Any], ...] = (
    {"label": "corpus_damping_wall_xy",
     "boundaries": ("metallic", "metallic", "periodic"), "pml": 2, "cell": CELL,
     "epsilon": "anisotropic", "conductivity": 0.4, "cond_mask": (True, True, True)},
    {"label": "wall_xyz_asymmetric_pml", "boundaries": ("metallic",) * 3,
     "pml": ((3, 3), (2, 2), (1, 1)), "cell": CELL, "epsilon": "anisotropic",
     "conductivity": 0.35, "cond_mask": (True, True, True)},
    {"label": "periodic_no_wall", "boundaries": ("periodic",) * 3, "pml": 2,
     "cell": CELL, "epsilon": "isotropic", "conductivity": 0.5,
     "cond_mask": (True, True, True)},
    {"label": "wall_z_deep_pml", "boundaries": ("periodic", "periodic", "metallic"),
     "pml": ((2, 2), (2, 2), (4, 4)), "cell": CELL, "epsilon": "anisotropic",
     "conductivity": 0.25, "cond_mask": (True, True, True)},
)

MUTATION_SPEC_LABELS: Tuple[str, ...] = (
    "corpus_damping_wall_xy", "wall_xyz_asymmetric_pml", "periodic_no_wall",
    "wall_z_deep_pml")
SEPARATE_CONTROL_LABELS: Tuple[str, ...] = ("corpus_damping_wall_xy",
                                            "periodic_no_wall")
DEPOSIT_SPEC_LABELS: Tuple[str, ...] = ("corpus_damping_wall_xy",
                                        "wall_xyz_asymmetric_pml")
RESCALE_SPEC_LABELS: Tuple[str, ...] = ("corpus_damping_wall_xy", "periodic_no_wall")


# ---------------------------------------------------------------------------
# The fixture
# ---------------------------------------------------------------------------

def case_rng(label: str) -> np.random.Generator:
    return np.random.default_rng(
        SEED + int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], "big"))


def seeded_hosts(names: Sequence[str], value_class: str, shape, rng
                 ) -> Dict[str, np.ndarray]:
    """One host array per name, in the requested value class."""
    if value_class == "uniform":
        return {name: rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
                for name in names}
    if value_class == "subnormal_band":
        return probe.subnormal_band_hosts(tuple(names), shape, rng)
    raise ValueError(f"value class {value_class!r} is not one of {VALUE_CLASSES}")


def build(spec: Dict[str, Any], value_class: str, rng,
          zero_curl: bool = False, word_classes: bool = False):
    """One CuPy engine on the CONDUCTIVE-PML branch: ``(fields, grid, pml, host)``.

    ``zero_curl`` seeds H and B to EXACTLY zero, which is what lets a ``-0.0`` word
    survive ``step_D`` to the injection point: the conductive tail is
    ``((f * cf) - curl) * ci`` and with ``curl`` exactly ``+0.0`` it maps ``-0.0`` to
    ``-0.0``. Used only by :func:`leg_rescale`, whose subject IS the sign of zero.

    ``word_classes`` seeds D with a deliberate rotation of ``-0.0``, ``+0.0``, the
    smallest subnormal and a normal value, so the classifier has all four to sort at
    cells the deposit does NOT name.
    """
    grid = Grid(resolution=1.0, cell_size=tuple(spec["cell"]),
                boundaries=tuple(spec["boundaries"]), xp=cp, courant=0.5)
    fields = Fields(grid=grid)
    shape = tuple(int(n) for n in grid.shape)

    epsilon, inverse = {}, {}
    first = None
    for offset, component in enumerate(_ELECTRIC):
        low = 1.2 + (0.4 * offset if spec["epsilon"] == "anisotropic" else 0.0)
        values = rng.uniform(low, low + 2.2, size=shape).astype(np.float32)
        if spec["epsilon"] == "isotropic":
            # ONE ALLOCATION BOUND THREE TIMES, which is what an isotropic run does
            # (fields.py:1321-1326) and what the non-restrict inv_eps parameters exist
            # for. Building three equal COPIES instead would leave that binding
            # untested.
            first = values if first is None else first
            values = first
        epsilon[component] = cp.asarray(values)
        inverse[component] = cp.asarray(
            (np.float32(1.0) / values).astype(np.float32))
    fields.set_epsilon_volumes(epsilon, inverse)

    volume = np.full(shape, float(spec["conductivity"]), dtype=np.float32)
    volume *= np.linspace(0.5, 1.5, shape[0], dtype=np.float32)[:, None, None]
    fields.set_d_conductivity(cp.asarray(volume))
    fields.enable_pml_storage()
    thickness = (spec["pml"] if isinstance(spec["pml"], int)
                 else tuple(tuple(int(v) for v in pair) for pair in spec["pml"]))
    pml = PML(grid=grid, thickness=thickness)

    host = seeded_hosts(STATE_NAMES, value_class, shape, rng)
    magnetic = ("Bx", "By", "Bz", "Hx", "Hy", "Hz")
    if zero_curl:
        for name in magnetic:
            host[name] = np.zeros(shape, dtype=np.float32)
    if word_classes:
        classes = np.array([-0.0, 0.0, np.float32(1e-45), 0.37], dtype=np.float32)
        for name in _TARGETS:
            flat = np.resize(classes, int(np.prod(shape)))
            host[name] = flat.reshape(shape).astype(np.float32)
    for name in STATE_NAMES:
        array = getattr(fields, name, None)
        if array is None:
            continue
        array[...] = cp.asarray(np.ascontiguousarray(host[name]))
    return fields, grid, pml, host


def state_of(fields) -> Dict[str, Any]:
    return {name: getattr(fields, name) for name in STATE_NAMES}


def compare(left, right) -> Dict[str, int]:
    a, b = state_of(left), state_of(right)
    return {name: count for name in STATE_NAMES
            if (count := int(np.count_nonzero(words(a[name]) != words(b[name]))))}


def frozen(fields) -> Dict[str, np.ndarray]:
    return {name: words(array).copy() for name, array in state_of(fields).items()}


def classify(before: np.ndarray, after: np.ndarray) -> Dict[str, int]:
    """Sort every differing word into the classes this campaign has measured.

    ``signed_zero``  a zero whose SIGN BIT changed, either direction.
    ``subnormal``    a subnormal word that became zero, or a zero that came from one.
    ``other``        a divergence this classifier cannot explain, which is what makes
                     a leg FAIL rather than pass.
    """
    a = np.ascontiguousarray(before, dtype=np.float32).ravel()
    b = np.ascontiguousarray(after, dtype=np.float32).ravel()
    wa, wb = a.view(np.uint32), b.view(np.uint32)
    differing = wa != wb
    counts = {"signed_zero": 0, "subnormal": 0, "other": 0}
    if not np.any(differing):
        return counts
    ia, ib = wa[differing], wb[differing]
    zero_a, zero_b = (ia & 0x7FFFFFFF) == 0, (ib & 0x7FFFFFFF) == 0
    sub_a = ((ia & 0x7F800000) == 0) & ~zero_a
    sub_b = ((ib & 0x7F800000) == 0) & ~zero_b
    counts["signed_zero"] = int(np.count_nonzero(zero_a & zero_b))
    counts["subnormal"] = int(np.count_nonzero((sub_a & zero_b) | (zero_a & sub_b)))
    counts["other"] = (int(np.count_nonzero(differing))
                       - counts["signed_zero"] - counts["subnormal"])
    return counts


def leg_driver_order(replaces: Sequence[str]) -> Dict[str, Any]:
    """``REPLACES`` must be a contiguous run of ``FdtdDriver.step``'s own sequence.

    Read out of ``driver.py``'s TEXT rather than by importing the driver. The two
    symmetry fills sit inside the run and are INERT here because this family refuses a
    mirror plane by name -- their absence from ``REPLACES`` is a refusal, not an
    oversight, which is why they are named as the only tolerated gap rather than an
    arbitrary one.
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
    inert = {"fill_symmetry_bc_D", "fill_folded_far_ghosts_D"}
    span = found[positions[0]:positions[-1] + 1] if positions else ()
    contiguous = bool(positions) and all(
        name in declared or name in inert for name in span)
    return {
        "passed": bool(found == DRIVER_ORDER and subsequence and contiguous
                       and declared == FUSED_PASSES),
        "driver_file_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "driver_call_sequence": list(found),
        "fused_replaces": list(declared),
        "replaces_is_an_ordered_subsequence": subsequence,
        "run_is_contiguous_modulo_the_inert_fills": contiguous,
        "passes_inside_the_run_left_on_the_array_path": [
            name for name in span if name not in declared],
        "passes_left_on_the_array_path": [name for name in DRIVER_ORDER
                                          if name not in declared],
    }


def competing_step_D_products(fields, pml, grid, sources) -> List[str]:
    """Every OTHER ``step_D`` product whose predicate admits this fixture.

    Asked over the whole table rather than against a hand-picked neighbour, so a
    product landing later cannot quietly overlap this one.
    """
    from meep_gpu.cuda_kernels import fused_pairs  # noqa: PLC0415

    mine = "cuda_conductive_fused_electric_pair"

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


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------

def run_case(spec: Dict[str, Any], value_class: str, steps: int,
             kernel: Optional[Any] = None,
             tables_patch: Optional[Callable[[Dict[str, Any], Any],
                                             Dict[str, Any]]] = None,
             codes_patch: Optional[Callable[[Sequence[Any]], Sequence[Any]]] = None,
             walls_patch: Optional[Callable[[Sequence[int], Any],
                                            Sequence[int]]] = None,
             label_suffix: str = "") -> Dict[str, Any]:
    """Step two engines side by side and compare per COMPLETE DRIVER STEP."""
    from meep_gpu.cuda_kernels import (  # noqa: PLC0415
        conductive_fused_electric_pair as family,
    )
    from meep_gpu.cuda_kernels import in_seam_coverage, step_curl_kernels  # noqa: PLC0415
    from meep_gpu.cuda_kernels import fused_electric_pair as real  # noqa: PLC0415

    started = time.time()
    label = f"{spec['label']}|{value_class}{label_suffix}"
    case: Dict[str, Any] = {"label": spec["label"], "value_class": value_class,
                            "case_seed_label": label, "steps_requested": steps,
                            "boundaries": list(spec["boundaries"]),
                            "epsilon": spec["epsilon"], "pml_cells": spec["pml"],
                            "conductivity": spec["conductivity"]}

    reference, ref_grid, ref_pml, host = build(spec, value_class, case_rng(label))
    actual, grid, pml, _ = build(spec, value_class, case_rng(label))
    case["shape"] = [int(n) for n in grid.shape]
    case["dtdx"] = float(grid.dt / grid.dx)
    case["distinct_inverse_epsilon_allocations"] = len({
        int(actual.inverse_epsilon_for(c).data.ptr) for c in _ELECTRIC})
    case["f_cond_allocated"] = bool(getattr(actual, "f_cond_Dx", None) is not None)
    case["pml_is_active"] = bool(stepping._pml_is_active(pml))
    if not case["f_cond_allocated"] or not case["pml_is_active"]:
        case["passed"] = False
        case["why"] = (f"this fixture is not the conductive x ordinary cell: "
                       f"f_cond={case['f_cond_allocated']}, "
                       f"pml_is_active={case['pml_is_active']}")
        return case

    drift = compare(reference, actual)
    if drift:
        case["passed"] = False
        case["why"] = f"the two builds are not identical: {drift}"
        return case

    covered, reason = family.covers_conductive_fused_electric_pair(
        actual, pml, grid, ())
    case["predicate"] = {"covered": bool(covered), "reason": reason}
    if not covered:
        case["passed"] = False
        case["why"] = f"the predicate refused this fixture: {reason}"
        return case

    overlaps = competing_step_D_products(actual, pml, grid, ())
    case["competing_products"] = overlaps
    if overlaps:
        case["passed"] = False
        case["why"] = (f"another step_D product admits this fixture too "
                       f"({overlaps}); install_fused_pairs would leave the seam "
                       f"unfused naming both")
        return case

    tables = family.conductive_fused_electric_pair_tables(pml)
    conductivity, histories, cond = family.conductive_bindings(actual)
    case["cond_mask"] = [bool(flag) for flag in cond]
    if tuple(cond) != tuple(bool(v) for v in spec["cond_mask"]):
        case["passed"] = False
        case["why"] = (f"the resolved conductive mask {tuple(cond)} is not the spec's "
                       f"{tuple(spec['cond_mask'])}")
        return case
    case["distinct_allocations_checked"] = family.assert_disjoint_bindings(
        actual, tables, conductivity, histories)
    if tables_patch is not None:
        tables = tables_patch(tables, pml)

    codes = step_curl_kernels.real_curl_boundary_codes(grid)
    walls = in_seam_coverage.zero_metal_axes(grid)
    case["boundary_codes"] = [int(c) for c in codes]
    case["zero_metal_axes"] = [bool(w) for w in walls]
    if codes_patch is not None:
        codes = codes_patch(codes)
    if walls_patch is not None:
        walls = walls_patch(walls, grid)
    case["zero_metal_axes_used"] = [bool(w) for w in walls]

    before = frozen(actual)
    launched = 0
    counter = MemoLaunchCounter()
    per_step: List[Dict[str, Any]] = []
    with counter:
        for step in range(1, steps + 1):
            array_step(reference, ref_pml)
            live_conductivity, live_histories, live_cond = family.conductive_bindings(
                actual)
            for name in DRIVER_ORDER:
                if name == "step_D":
                    report = family.launch_conductive_fused_electric_pair(
                        actual, tables, codes, walls, case["dtdx"],
                        live_conductivity, live_histories, live_cond, kernel)
                    launched += int(bool(report.get("launched")))
                    cp.cuda.runtime.deviceSynchronize()
                elif name not in FUSED_PASSES:
                    run_pass(name, actual, pml)
            cp.cuda.runtime.deviceSynchronize()
            difference = compare(reference, actual)
            census = operand_census({n: to_host(v)
                                     for n, v in state_of(reference).items()})
            per_step.append({"step": step,
                             "differing_words": sum(difference.values()),
                             "differing_arrays": dict(sorted(difference.items())),
                             "reference_subnormals": census["subnormals"],
                             "reference_negative_zeros": census["negative_zeros"]})
            if difference:
                break

    after = state_of(actual)
    moved = {name: int(np.count_nonzero(before[name] != words(after[name])))
             for name in STATE_NAMES}
    still = sorted(name for name, count in moved.items() if count == 0)
    required = STATE_NAMES if value_class == "uniform" else BAND_MUST_MOVE
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
        "arrays_compared": len(STATE_NAMES),
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
    _ = (host, real)
    return case


# ---------------------------------------------------------------------------
# The separate composition
# ---------------------------------------------------------------------------

def leg_separate_control(spec: Dict[str, Any], steps: int) -> Dict[str, Any]:
    """THREE engines from one seed: array path, separate certified products, fused."""
    from meep_gpu.cuda_kernels import conductive_kernels  # noqa: PLC0415
    from meep_gpu.cuda_kernels import constitutive_kernels  # noqa: PLC0415
    from meep_gpu.cuda_kernels import in_seam_coverage, in_seam_passes  # noqa: PLC0415
    from meep_gpu.cuda_kernels import step_curl_kernels  # noqa: PLC0415
    from meep_gpu.cuda_kernels import (  # noqa: PLC0415
        conductive_fused_electric_pair as family,
    )

    started = time.time()
    label = f"{spec['label']}|separate"
    reference, ref_grid, ref_pml, _ = build(spec, "uniform", case_rng(label))
    separate, sep_grid, sep_pml, _ = build(spec, "uniform", case_rng(label))
    fused, grid, pml, _ = build(spec, "uniform", case_rng(label))

    codes = step_curl_kernels.real_curl_boundary_codes(grid)
    walls = in_seam_coverage.zero_metal_axes(grid)
    dtdx = float(grid.dt / grid.dx)
    tables = family.conductive_fused_electric_pair_tables(pml)
    # THE CURL'S OWN TABLE IS THE INTEGER SUB-LATTICE (half_integer False for step_D);
    # the certified launcher takes it directly rather than through this family's pair.
    curl_tables = step_curl_kernels.real_pml_curl_tables(sep_pml, False)
    row: Dict[str, Any] = {"label": spec["label"],
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
                    conductive_kernels.step_conductive_curl(
                        separate, sep_pml, "step_D", codes, dtdx,
                        tables=curl_tables)
                    if any(walls):
                        # THE CERTIFIED IN-SEAM PASS, through its own dispatcher.
                        in_seam_passes.run_pass("zero_metal", separate, "D",
                                                grid=sep_grid)
                    constitutive_kernels.update_fused_pml_real("E", separate, sep_pml)
                cp.cuda.runtime.deviceSynchronize()
            elif name not in FUSED_PASSES:
                run_pass(name, separate, sep_pml)
        for name in DRIVER_ORDER:
            if name == "step_D":
                with fused_counter:
                    conductivity, histories, cond = family.conductive_bindings(fused)
                    family.launch_conductive_fused_electric_pair(
                        fused, tables, codes, walls, dtdx, conductivity, histories,
                        cond)
                cp.cuda.runtime.deviceSynchronize()
            elif name not in FUSED_PASSES:
                run_pass(name, fused, pml)
    cp.cuda.runtime.deviceSynchronize()

    row["separate_vs_array"] = compare(reference, separate)
    row["fused_vs_array"] = compare(reference, fused)
    row["separate_launches"] = separate_counter.total
    row["fused_launches"] = fused_counter.total
    row["launches_removed_per_step"] = (
        (separate_counter.total - fused_counter.total) / max(steps, 1))
    row["passed"] = (not row["separate_vs_array"] and not row["fused_vs_array"]
                     and fused_counter.total < separate_counter.total)
    row["seconds"] = time.time() - started
    return row


# ---------------------------------------------------------------------------
# The deposit leg
# ---------------------------------------------------------------------------

class _Injector:
    """The SHIPPED ``_inject_electric_through_conductivity``, bound to bare fields.

    The method reads ``self.fields`` and nothing else, so binding it to a two-line
    shim drives the TREE'S OWN code rather than a transcription of it -- which is the
    whole point of the rescale leg.
    """

    __slots__ = ("fields",)

    def __init__(self, fields: Any) -> None:
        self.fields = fields

    inject = FdtdDriver._inject_electric_through_conductivity


def _dense_rescale(fields: Any, sources: Sequence[Any], when: float) -> None:
    """THE RETIRED COMPOSITION, kept as a control and never as a code path.

    Three WHOLE-VOLUME passes, transcribed from the docstring of the method that
    replaced them.
    """
    by_target: Dict[str, List[Any]] = {}
    for source in sources:
        if getattr(source, "is_integrated", False):
            continue
        by_target.setdefault("D" + source.component[1], []).append(source)
    before = {}
    for name in sorted(by_target):
        if fields.condinv_for(name) is None:
            continue
        before[name] = cp.array(getattr(fields, name), copy=True)
    for source in sources:
        source.inject(fields, when)
    for name, snapshot in before.items():
        array = getattr(fields, name)
        array -= snapshot
        array *= fields.condinv_for(name)
        array += snapshot


def _source_for(grid):
    return GaussianPulsedSource(grid=grid, component="Ez", center=(0.0, 0.0, 0.0),
                                size=(0.0, 0.0, 0.0), frequency=1.0, fwidth=0.2,
                                amplitude=1.0)


def leg_deposit(spec: Dict[str, Any], steps: int) -> Dict[str, Any]:
    """The bracket end to end, with the unbracketed launch as a control."""
    from meep_gpu.cuda_kernels import in_seam_coverage, step_curl_kernels  # noqa: PLC0415
    from meep_gpu.cuda_kernels import (  # noqa: PLC0415
        conductive_fused_electric_pair as family,
    )

    started = time.time()
    row: Dict[str, Any] = {"label": spec["label"]}
    label = f"{spec['label']}|deposit"

    reference, ref_grid, ref_pml, _ = build(spec, "uniform", case_rng(label))
    ref_sources = [_source_for(ref_grid)]
    dt = float(ref_grid.dt)

    # DOES THE FIXTURE ACTUALLY DEPOSIT? Asked AT THE TIMES THE LEGS THEMSELVES USE,
    # because this source's carrier has period 1/frequency = 1.0 and every
    # half-integer time is an exact ZERO of its amplitude: a single hardcoded t = 0.5
    # reported zero and declared perfectly good legs vacuous on the first cut.
    probe_fields, probe_grid, _pp, _ph = build(spec, "uniform", case_rng(label))
    probe_source = _source_for(probe_grid)
    deposited = 0
    for step in range(steps):
        before = words(probe_fields.Dz).copy()
        _Injector(probe_fields).inject([probe_source], (step + 0.5) * dt)
        cp.cuda.runtime.deviceSynchronize()
        deposited += int(np.count_nonzero(before != words(probe_fields.Dz)))
    row["deposited_words"] = deposited

    legs: Dict[str, Any] = {}
    for tag in ("bracketed", "unbracketed"):
        fields, grid, pml, _ = build(spec, "uniform", case_rng(label))
        sources = [_source_for(grid)]
        codes = step_curl_kernels.real_curl_boundary_codes(grid)
        walls = in_seam_coverage.zero_metal_axes(grid)
        tables = family.conductive_fused_electric_pair_tables(pml)
        dtdx = float(grid.dt / grid.dx)
        for step in range(steps):
            when = (step + 0.5) * dt
            # THE FUSED ROUTE RUNS AT step_D'S POSITION, with every other driver pass
            # in its own place around it -- the same walk `run_case` uses.
            for name in DRIVER_ORDER:
                if name == "step_D":
                    saved = deposit_repair.save(fields, tuple(sources), "D", pml,
                                                paths=family.REPAIR_PATHS)
                    conductivity, histories, cond = family.conductive_bindings(fields)
                    family.launch_conductive_fused_electric_pair(
                        fields, tables, codes, walls, dtdx, conductivity, histories,
                        cond)
                    cp.cuda.runtime.deviceSynchronize()
                    # THE SHIPPED CONDUCTIVE INJECTION, not a plain one: on this cell
                    # the driver routes every electric source through the condinv
                    # rescale, and a leg that injected plainly would be bracketing a
                    # seam the driver does not run.
                    _Injector(fields).inject(list(sources), when)
                    stepping.zero_metal_D(fields)
                    if tag != "unbracketed":
                        deposit_repair.apply(fields, pml, tuple(sources), "D", saved)
                elif name not in FUSED_PASSES:
                    run_pass(name, fields, pml)
        cp.cuda.runtime.deviceSynchronize()
        legs[tag] = {"fields": fields}

    for step in range(steps):
        when = (step + 0.5) * dt
        for name in DRIVER_ORDER:
            if name == "step_D":
                stepping.step_D(reference, ref_pml)
                _Injector(reference).inject(list(ref_sources), when)
                stepping.zero_metal_D(reference)
                stepping.update_E(reference, ref_pml)
            elif name not in FUSED_PASSES:
                run_pass(name, reference, ref_pml)
    cp.cuda.runtime.deviceSynchronize()

    row["legs"] = {tag: {"differing_words": sum(
        compare(reference, entry["fields"]).values())}
        for tag, entry in legs.items()}
    # THE FLOOR: the bracketed route must be byte-IDENTICAL, the UNBRACKETED control
    # must diverge (otherwise the fixture deposits nothing the launch could have
    # missed), and the fixture must actually deposit.
    row["passed"] = bool(row["deposited_words"] > 0
                         and row["legs"]["bracketed"]["differing_words"] == 0
                         and row["legs"]["unbracketed"]["differing_words"] > 0)
    row["seconds"] = time.time() - started
    return row


# ---------------------------------------------------------------------------
# The rescale leg -- the charter's own question, on a device
# ---------------------------------------------------------------------------

def leg_rescale(spec: Dict[str, Any], steps: int, policy: str) -> Dict[str, Any]:
    """Does the SHIPPED injection touch a non-deposit word, and what does the retired
    composition destroy?

    Two measurements on a fixture seeded with all four word classes and an
    exactly-zero curl, so the classes survive ``step_D`` to the injection point.
    """
    from meep_gpu.cuda_kernels import in_seam_coverage, step_curl_kernels  # noqa: PLC0415
    from meep_gpu.cuda_kernels import (  # noqa: PLC0415
        conductive_fused_electric_pair as family,
    )

    started = time.time()
    label = f"{spec['label']}|rescale"
    row: Dict[str, Any] = {"label": spec["label"], "subnormal_policy": policy}

    # --- (1) does the shipped injection touch a non-deposit word? -----------------
    fields, grid, _pml, _host = build(spec, "uniform", case_rng(label),
                                      zero_curl=True, word_classes=True)
    source = _source_for(grid)
    index = deposit_repair._deposit_index(source)
    row["source_publishes_index"] = index is not None
    snapshots = {name: words(getattr(fields, name)).copy() for name in _TARGETS}
    row["seeded_signed_zeros"] = int(sum(
        int(np.count_nonzero(snapshots[name] == 0x80000000)) for name in _TARGETS))
    row["seeded_subnormals"] = int(sum(
        int(np.count_nonzero(((snapshots[name] & 0x7F800000) == 0)
                             & ((snapshots[name] & 0x7FFFFFFF) != 0)))
        for name in _TARGETS))
    _Injector(fields).inject([source], 0.5)
    cp.cuda.runtime.deviceSynchronize()
    touched: Dict[str, int] = {}
    host_index = tuple(to_host(part).ravel() for part in index) if index else None
    for name in _TARGETS:
        after = words(getattr(fields, name))
        mask = np.ones(after.shape, dtype=bool)
        if host_index is not None and name == "D" + source.component[1]:
            flat = np.ravel_multi_index(host_index,
                                        tuple(int(n) for n in grid.shape))
            mask[flat] = False
        touched[name] = int(np.count_nonzero(snapshots[name][mask] != after[mask]))
    row["non_deposit_words_touched"] = touched
    row["sparse_untouched"] = all(count == 0 for count in touched.values())

    # --- (2) the retired dense composition, as a control that must diverge --------
    reference, ref_grid, ref_pml, _ = build(spec, "uniform", case_rng(label),
                                            zero_curl=True, word_classes=True)
    engine, grid2, pml2, _ = build(spec, "uniform", case_rng(label),
                                   zero_curl=True, word_classes=True)
    ref_sources = [_source_for(ref_grid)]
    eng_sources = [_source_for(grid2)]
    codes = step_curl_kernels.real_curl_boundary_codes(grid2)
    walls = in_seam_coverage.zero_metal_axes(grid2)
    tables = family.conductive_fused_electric_pair_tables(pml2)
    dtdx = float(grid2.dt / grid2.dx)
    dt = float(ref_grid.dt)
    for step in range(steps):
        when = (step + 0.5) * dt
        # THE REFERENCE TAKES THE SHIPPED SPARSE INJECTION -- it is the array path as
        # the tree ships it today. The engine replays the RETIRED dense passes on the
        # fused route, which is the composition whose word rewrites this leg is about.
        stepping.step_D(reference, ref_pml)
        _Injector(reference).inject(list(ref_sources), when)
        stepping.zero_metal_D(reference)
        stepping.update_E(reference, ref_pml)

        saved = deposit_repair.save(engine, tuple(eng_sources), "D", pml2,
                                    paths=family.REPAIR_PATHS)
        conductivity, histories, cond = family.conductive_bindings(engine)
        family.launch_conductive_fused_electric_pair(
            engine, tables, codes, walls, dtdx, conductivity, histories, cond)
        cp.cuda.runtime.deviceSynchronize()
        _dense_rescale(engine, eng_sources, when)
        stepping.zero_metal_D(engine)
        deposit_repair.apply(engine, pml2, tuple(eng_sources), "D", saved)
    cp.cuda.runtime.deviceSynchronize()

    counts = {"signed_zero": 0, "subnormal": 0, "other": 0}
    total = 0
    for name in STATE_NAMES:
        part = classify(to_host(getattr(reference, name)),
                        to_host(getattr(engine, name)))
        for key in counts:
            counts[key] += part[key]
        total += sum(part.values())
    row["dense_control_differing_words"] = total
    row["dense_control_classes"] = counts
    row["dense_diverges"] = total > 0
    row["dense_is_fully_classified"] = counts["other"] == 0 and total > 0
    # WHICH CLASSES ACTUALLY ARMED, recorded rather than assumed. NumPy cannot arm the
    # subnormal half at all (it does not flush); under the ``flush`` policy this leg
    # can, and under ``keep`` it may not. Either is a real answer and is recorded with
    # the policy beside it.
    row["classes_armed"] = sorted(k for k, v in counts.items()
                                  if v and k != "other")
    row["passed"] = bool(row["sparse_untouched"] and row["dense_is_fully_classified"])
    row["seconds"] = time.time() - started
    return row


# ---------------------------------------------------------------------------
# The mutation tables
# ---------------------------------------------------------------------------

def _wall(axis: str):
    return lambda spec: spec["boundaries"]["xyz".index(axis)] == "metallic"


def _anisotropic(spec) -> bool:
    return spec["epsilon"] == "anisotropic"


def _graded_pml(spec) -> bool:
    """An absorber whose two sigma sub-lattices differ on some axis this run resolves."""
    return not isinstance(spec["pml"], int) or int(spec["pml"]) > 0


def _always(spec) -> bool:
    return True


SOURCE_MUTATIONS: Tuple[Tuple[str, Tuple[str, str, int], Optional[bool],
                              Callable[[Dict[str, Any]], bool]], ...] = (
    # --- THE SEAM ------------------------------------------------------------
    # A CONFIRMED NULL, and declared as one rather than left at "either answer is
    # fine". By the time the constitutive body runs, the store above has written the
    # CLEARED register into D, so the reload is the identity. The ordering claim it
    # looks like it is making is carried by `wall_clear_misses_the_register` below,
    # which is the same question asked where it IS observable.
    ("seam_reads_the_stored_volume_instead_of_the_register",
     ("    float src_x = d_x * inv_eps_Ex[idx];",
      "    float src_x = Dx[idx] * inv_eps_Ex[idx];", 1), False, _wall("x")),
    ("seam_takes_the_wrong_component",
     ("    float src_x = d_x * inv_eps_Ex[idx];",
      "    float src_x = d_y * inv_eps_Ex[idx];", 1), True, _always),
    # --- THE SUB-LATTICE SHADOW, and this defect exists only because the fusion put
    # both halves in one scope: the curl's kms is INTEGER and the constitutive's
    # HALF-INTEGER, and letting the certified name shadow is a half-cell error in the
    # absorber profile -- converged, smooth and wrong.
    ("constitutive_sublattice_shadow",
     ("    constitutive_apply(Ex, f_w_Ex, idx, src_x, kps_x[i], kms_half_x[i]);",
      "    constitutive_apply(Ex, f_w_Ex, idx, src_x, kps_x[i], kms_x[i]);", 1),
     True, _graded_pml),
    # --- THE CONDUCTIVE RECURRENCE ------------------------------------------
    ("conductive_case_predicate_uses_a_tolerance",
     ("    bool dsig = (km1 != 1.0f) || (si1 != 1.0f);",
      "    bool dsig = (fabsf(km1 - 1.0f) > 1e-6f) || (fabsf(si1 - 1.0f) > 1e-6f);",
      1), None, _always),
    ("conductive_selection_prefers_the_wrong_case",
     ("    float value = dsigu ? f_split : (dsig ? f_first : f_direct);",
      "    float value = dsigu ? f_split : f_direct;", 1), True, _always),
    ("conductive_drops_condinv",
     ("    float c_new    = ((cv * cfv) - curl) * civ;            "
      "// cases A and C",
      "    float c_new    = (cv * cfv) - curl;            // cases A and C", 1),
     True, _always),
    # THE THIRD HISTORY MUST STILL BE WRITTEN. A weld that swallowed this store would
    # leave the conductive recurrence a step behind at every cell.
    ("conductive_history_store_dropped",
     ("    if (dsig) c[idx] = c_new;\n    return value;",
      "    return value;", 1), True, _always),
    ("split_field_auxiliary_store_dropped",
     ("    if (dsigu) u[idx] = u_new;", "", 1), True, _always),
    # --- THE WALL CARRY ------------------------------------------------------
    ("wall_clear_uses_the_b_diagonal",
     ("if (wall_x && i == 0) { d_y = 0.0f; d_z = 0.0f; }",
      "if (wall_x && i == 0) { d_x = 0.0f; }", 1), True, _wall("x")),
    # ZERO_METAL WRITES D ALONE (stepping.py:2206-2247). A carry that cleared the
    # split-field auxiliary too would damp the absorber differently at every wall cell.
    ("wall_clear_also_zeroes_the_histories",
     ("    Dx[idx] = d_x;",
      "    Dx[idx] = d_x; if (wall_y && j == 0) fu_Dx[idx] = 0.0f;", 1), True,
     _wall("y")),
    # THE CLEAR MUST REACH THE REGISTER: clearing the VOLUME instead leaves the
    # register uncleared and the store below writes it back over the zero, so both
    # consumers see a wall cell the array path holds at zero.
    ("wall_clear_misses_the_register",
     ("    if (wall_x && i == 0) { d_y = 0.0f; d_z = 0.0f; }",
      "    if (wall_x && i == 0) { Dy[idx] = 0.0f; Dz[idx] = 0.0f; }", 1), True,
     _wall("x")),
    # --- THE INVERSE-EPSILON MULTIPLY ---------------------------------------
    # D on the LEFT is transcription discipline (IEEE multiply commutes), so this is a
    # NULL that must come back UNCAUGHT -- paired with the component swap below, which
    # must be caught.
    ("inverse_epsilon_operand_order_swapped",
     ("    float src_x = d_x * inv_eps_Ex[idx];",
      "    float src_x = inv_eps_Ex[idx] * d_x;", 1), False, _always),
    ("inverse_epsilon_component_swapped",
     ("    float src_x = d_x * inv_eps_Ex[idx];",
      "    float src_x = d_x * inv_eps_Ez[idx];", 1), True, _anisotropic),
    # --- THE CURL ------------------------------------------------------------
    ("curl_shift_direction_flipped",
     ("float sf = shift_dn(Hz, idx, j, ny, sy, bc_y);",
      "float sf = shift_up(Hz, idx, j, ny, sy, bc_y);", 1), True, _always),
)


HOST_MUTATIONS: Tuple[Tuple[str, Dict[str, Any], Optional[bool],
                            Callable[[Dict[str, Any]], bool]], ...] = (
    ("curl_takes_the_half_integer_lattice",
     {"tables_patch": base._half_integer_curl_tables}, True, _graded_pml),
    ("constitutive_takes_the_integer_lattice",
     {"tables_patch": base._integer_constitutive_tables}, True, _graded_pml),
    ("curl_takes_its_own_integer_lattice",
     {"tables_patch": base._integer_curl_tables}, False, _always),
    ("walls_dropped", {"walls_patch": base._walls_dropped}, True,
     lambda spec: any(_wall(axis)(spec) for axis in "xyz")),
)


def compile_source(source: str, options: Sequence[str]):
    from meep_gpu.cuda_kernels import (  # noqa: PLC0415
        conductive_fused_electric_pair as family,
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
    """Compile every mask this run will launch, OUTSIDE any counted region."""
    from meep_gpu.cuda_kernels import (  # noqa: PLC0415
        conductive_fused_electric_pair as family,
    )

    started = time.time()
    masks = sorted({tuple(bool(v) for v in spec["cond_mask"])
                    for spec in specs_by_label.values()})
    for mask in masks:
        family._get_kernel(mask)
    return {"masks_compiled": [list(m) for m in masks],
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
                f"{spec['label']}|{value_class}: "
                f"{'OK' if case.get('passed') else 'FAIL'} "
                f"words={case.get('differing_words')} "
                f"({case.get('seconds', 0):.1f} s)")
    return cases


def _score(results, out_path, steps, specs_by_label, table, key, runner):
    rows: List[Dict[str, Any]] = []
    for label, payload, must, applies in table:
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
        conductive_fused_electric_pair as family,
    )

    def runner(spec, payload, step_count, label):
        old, new, count = payload
        source = family.conductive_fused_electric_pair_source(spec["cond_mask"])
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
REQUIRED_LEGS: Tuple[str, ...] = ("driver_order", "product", "separate_control", "deposit", "rescale",
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
        "rescale": all(row.get("passed") for row in results.get("rescale", [])),
        "source_mutations": not failed and not unarmed,
    }
    masks = sorted({tuple(case["cond_mask"]) for case in product
                    if "cond_mask" in case})
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
        "mutations_scored": sum(row["scored"] for row in source_rows + host_rows),
        "mutations_caught": sum(row["caught"] for row in source_rows + host_rows),
        "mutations_never_armed": unarmed,
        "mutations_that_failed": failed,
        "rescale_classes_armed": sorted({
            name for row in results.get("rescale", [])
            for name in row.get("classes_armed", [])}),
        "denominators": {
            "board_cell": ("D_to_E (cuda_conductive/conductive, "
                           "cuda_constitutive/ordinary)"),
            "corpus_rows_on_the_cell": 1,
            "corpus_rows_with_an_electric_deposit": 1,
            "corpus_rows_folded": 0,
        },
        "what_this_does_not_claim": [
            "no throughput or dispatch claim of any kind",
            "no verdict about a FOLDED run -- a mirror plane is refused BY NAME",
            "no verdict about a dispersive, complex, cylindrical, BFAST, special-kz, "
            "off-diagonal, nonlinear or INERT-absorber run -- each is refused BY NAME "
            "and none swept",
            "no verdict about the driver's dense condinv fallback beyond the control "
            "measurement: a scaled source publishing no deposit table is REFUSED by "
            "the predicate and is not served",
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="FILE to write gate.json to")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--subnormal-policy", choices=("keep", "flush"),
                        required=True)
    parser.add_argument("--import-meep-for-host-policy", action="store_true")
    parser.add_argument("--only", default=None)
    arguments = parser.parse_args(argv)

    out_path = Path(arguments.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    only = set(arguments.only.split(",")) if arguments.only else None
    specs_by_label = {spec["label"]: spec for spec in SPECS}
    results: Dict[str, Any] = {
        "gate": "cuda_conductive_fused_electric_pair",
        "subject": "meep_gpu/cuda_kernels/conductive_fused_electric_pair.py",
        "kernel": "conductive_fused_electric_pair_pml_real",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename,
        "argv": list(argv if argv is not None else sys.argv[1:]),
        "question": ("does ONE launch of conductive_fused_electric_pair_pml_real leave every stored volume -- both curl histories included -- byte-identical to stepping's step_D -> zero_metal_D -> update_E inside a complete driver step, with an electric source deposited in the seam, and does the driver's condinv injection still leave every non-deposit word untouched?"),
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
            conductive_fused_electric_pair as family,
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
                f"legs={ {k: v['differing_words'] for k, v in row['legs'].items()} }")

    if wanted("rescale"):
        rows = []
        for label in RESCALE_SPEC_LABELS:
            row = leg_rescale(specs_by_label[label], min(arguments.steps, 4), policy)
            rows.append(row)
            results["rescale"] = rows
            save(results, out_path)
            log(f"  rescale {label}: {'OK' if row.get('passed') else 'FAIL'} "
                f"sparse_untouched={row.get('sparse_untouched')} "
                f"classes={row.get('dense_control_classes')}")

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
    # would be silently unbindable.
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
