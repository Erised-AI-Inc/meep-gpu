"""Byte-identity gate for the COMPLEX E->P fused pair: ``update_E`` -> ``update_P`` welded under complex64 storage with no absorber.

WHAT IS UNDER TEST, and it is a claim about the WELD, which no verdict on either
half establishes:

* ``complex_no_pml_fused_polarization_pair.run_complex_no_pml_fused_polarization_pair``
  -- THREE per-component launches with the host rotation between them --
  reproduces the driver's own seam, ``stepping.update_E`` immediately followed
  by ``stepping.update_P`` (driver.py:3313/:3315), byte for byte over EVERY
  array the seam reads or writes, THE POLE VOLUMES INCLUDED and compared BY
  SLOT (the rotation moves array objects between ``P``, ``P_prev`` and
  ``_scratch``, so only a slot-named comparison can see a wrong rotation).
* the same weld reproduces the SEPARATELY CERTIFIED kernels it replaces --
  ``complex_no_pml_kernels.update_E_complex_no_pml_stored`` followed by
  ``update_P_complex_no_pml`` -- word for word, from the same frozen state.
  Both references are scored per case; a weld identical to the engine but not
  to the certified kernels (or the reverse) is a defect in somebody's
  transcription and must be SEEN, not averaged away.
* the LAUNCH ACCOUNTING: 3 fused launches per seam step, counted TWICE and
  independently -- once by the launcher's own result and once by a counting
  wrapper installed at the module's ``_get_kernel`` seam -- against
  ``1 + sum over components of len(driving states)`` on the certified route
  (the stored-E single is one launch for all three components; the certified
  ``update_P`` is one launch per (state, component)).

THE SEAM NULL AND ITS CONTROL. The weld's one arithmetic claim is that the
register hand-off ``ev`` IS the certified drive load (``fields.drive_field(c)``
is the stored E, and a complex64 word pair stored and reloaded is the identity
on the bits):

* ``reload_drive_from_global`` rewrites pole 0's drive operand back into a load
  of the array the E half just stored (``f0``). It MUST come back UNCAUGHT.
* ``wrong_drive_register`` rewrites it to the DISPLACEMENT (``g0``) -- the
  certified ADE gate's own wrong-drive control, restated against the fused
  text. It MUST DIVERGE.

THE ROTATION LEGS. ``update_P``'s three buffers rotate per (state, component)
on the host, and the weld transcribes that rotation between its own launches.
The defect that "still computes" is armed THREE ways (``rotate_before_launch``,
``rotate_after_all_launches``, ``freeze_one_states_rotation``), each against a
LOCAL copy of the shipped loop, with ``launcher_copy_NULL`` -- the same copy
WITHOUT the defect -- required UNCAUGHT so the copy is shown equivalent before
the defect is blamed.

FIXTURE FLOORS carried from the sibling gates: the pole volumes are IN the byte
comparison and are seeded NON-ZERO (a zero pole bank is a vacuous recurrence);
the scratch is seeded non-zero (it is the output and a lazy kernel must be
seen); the three inverse-epsilon volumes are DISTINCT and inhomogeneous; at
least one swept case carries TWO states of MIXED sigma kinds (volume beside
uniform), because the kind tuple is a compile-time axis of the emitted kernel;
and the conductive axis is swept because every corpus row on this product's
board cell (the four ``TestLoadDump`` 3-D rows) declares one.

Both float32 subnormal policies are gated (two runs of this script); the keep
run needs a ``CUPY_CACHE_DIR`` containing ``ftz_stripped`` and the flush run
``--import-meep-for-host-policy``. One flushed line per case (the development notes rule
7); the artifact is rewritten atomically after every case.

Usage (from ``the repository root``)::

    CUDA_VISIBLE_DEVICES=$GPU CUPY_CACHE_DIR=$SCRATCH/ftz_stripped \\
        python -u parity/meep_gpu/gate_cuda_complex_polarization_pair.py \\
        --subnormal-policy keep --out $OUT/keep/gate.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_API = os.path.abspath(os.path.join(_HERE, "..", ".."))
for path in (_HERE, _REPO_API):
    if path not in sys.path:
        sys.path.insert(0, path)

import probe_fused_kernel_bit_identity as probe  # noqa: E402
import gate_provenance  # noqa: E402
import gate_cuda_complex_no_pml as sibling  # noqa: E402 - fixtures + licence

from meep_gpu import stepping  # noqa: E402
from meep_gpu.dispersion import PolarizationState, Susceptibility  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.cuda_kernels import complex_no_pml_kernels as certified  # noqa: E402
from meep_gpu.cuda_kernels import (  # noqa: E402
    complex_no_pml_fused_polarization_pair as weld,
)
from meep_gpu.cuda_kernels.coverage import ade_sigma_is_volume  # noqa: E402

try:
    import cupy as cp
except ImportError:  # pragma: no cover - source legs still run
    cp = None

to_host = probe.to_host
words_differ = sibling.words_differ

STEPS = 24
VALUE_CLASSES: Tuple[str, ...] = ("uniform", "subnormal_band")
COMPONENTS: Tuple[str, ...] = ("Ex", "Ey", "Ez")

#: The driver's passes BEFORE this seam, in driver order (driver.py:3291-3311).
#: Run on the array path on every engine of a case, so the seam is measured at
#: the state the driver hands it.
BEFORE_SEAM: Tuple[str, ...] = (
    "step_B", "fill_symmetry_bc_B", "zero_metal_B", "fill_folded_far_ghosts_B",
    "update_H",
    "step_D", "fill_symmetry_bc_D", "zero_metal_D", "fill_folded_far_ghosts_D",
)

_TAKES_PML = {"step_B", "update_H", "step_D", "update_E", "update_P"}


def _run_pass(name: str, fields, pml) -> None:
    function = getattr(stepping, name)
    if name in _TAKES_PML:
        function(fields, pml)
    else:
        function(fields)


def log(message: str) -> None:
    print(message, flush=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

#: The swept shapes. 3-D dominates because the product's whole board cell is
#: the four TestLoadDump 3-D rows; the 2-D row is the reduction control.
SPECS: Tuple[Dict[str, Any], ...] = (
    {"label": "3d_phased_conductive", "cell": (5.0, 6.0, 7.0), "dimensions": 3,
     "boundaries": ("periodic",) * 3, "k_point": (0.2, -0.35, 0.1),
     "conductivity": "volume", "kinds": ("volume", "uniform")},
    {"label": "3d_metallic_two_volume", "cell": (5.0, 6.0, 7.0), "dimensions": 3,
     "boundaries": ("metallic",) * 3, "k_point": (0.0, 0.0, 0.0),
     "conductivity": "none", "kinds": ("volume", "volume")},
    {"label": "3d_conductive_one_uniform", "cell": (6.0, 5.0, 7.0),
     "dimensions": 3, "boundaries": ("periodic", "metallic", "periodic"),
     "k_point": (0.25, 0.0, -0.15), "conductivity": "volume",
     "kinds": ("uniform",)},
    {"label": "2d_phased_one_volume", "cell": (9.0, 11.0, 0.0), "dimensions": 2,
     "boundaries": ("periodic",) * 3, "k_point": (0.2, 0.4, 0.0),
     "conductivity": "none", "kinds": ("volume",)},
)

REDUCED_LABELS: Tuple[str, ...] = ("3d_phased_conductive",
                                  "2d_phased_one_volume")


def build(spec: Dict[str, Any], value_class: str, seed: int):
    """One complete ``(fields, grid)`` fixture, built from ``seed`` alone.

    The material and seeding rules are the sibling gate's (distinct
    inhomogeneous inverse-epsilon, graded per-component conductivity, non-zero
    pole banks INCLUDING the scratch); what this adds is the per-pole KINDS
    tuple, because the mixed volume-beside-uniform signature is a compile-time
    axis of the fused kernel that a single-form sweep would leave ungated.

    RETURNS THE SEEDED HOST WORDS BESIDE THE FIXTURE, and that third value is
    what :func:`operand_census` is asked about -- the site every sibling CUDA
    gate measures (``gate_cuda_fused_complex_pairs._seed_complex_state``
    returns exactly this dict and the case censuses it). The band class's
    precondition is a statement about the DRAW, so it is asked of the draw;
    reading it back off the device would have measured the upload as well,
    which is a second question wearing the first one's name.

    (The 2026-09-02 flush failure this pairs with was NOT about the site: the
    gate's own ``np.abs(x) < FLT_MIN`` counter could not see subnormal words at
    all under a DAZ policy, wherever it read them. That is fixed at
    :data:`operand_census` by using the shipped bit-level classifier.)
    """
    rng = np.random.default_rng(seed)
    grid = Grid(resolution=1.0, cell_size=tuple(spec["cell"]),
                boundaries=tuple(spec["boundaries"]),
                dimensions=spec.get("dimensions", 3), xp=cp, courant=0.5,
                k_point=tuple(spec["k_point"]))
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_field_storage()
    shape = tuple(grid.shape)

    inverse, epsilon = {}, {}
    for component in COMPONENTS:
        values = rng.uniform(0.2, 0.9, size=shape).astype(np.float32)
        inverse[component] = cp.asarray(np.ascontiguousarray(values))
        epsilon[component] = cp.asarray(
            np.ascontiguousarray((1.0 / values).astype(np.float32)))
    fields.set_epsilon_volumes(epsilon, inverse)

    if spec["conductivity"] == "volume":
        for setter, names in ((fields.set_d_conductivity, ("Dx", "Dy", "Dz")),
                              (fields.set_b_conductivity, ("Bx", "By", "Bz"))):
            setter({name: cp.asarray(np.ascontiguousarray(
                rng.uniform(0.05, 0.4, size=shape).astype(np.float32)))
                for name in names})

    for index, kind in enumerate(spec["kinds"]):
        if kind == "volume":
            sigma: Any = {name: cp.asarray(np.ascontiguousarray(
                rng.uniform(0.1, 0.9, size=shape).astype(np.float32)))
                for name in COMPONENTS}
        else:
            sigma = float(0.2 + 0.13 * index)
        fields.polarizations.append(PolarizationState(
            Susceptibility(frequency=1.05 + 0.4 * index,
                           gamma=0.05 + 0.02 * index),
            sigma, grid, fields._field_dtype()))  # noqa: SLF001

    seeded: Dict[str, np.ndarray] = {}

    def _seed(target, key: str) -> None:
        host = sibling._complex_host(shape, value_class, rng)  # noqa: SLF001
        seeded[key] = host.view(np.float32).ravel().copy()
        target[...] = cp.asarray(host)

    for name in sibling.STATE:
        _seed(getattr(fields, name), name)
    for index, state in enumerate(fields.polarizations):
        for component in state.driven():
            _seed(state.P[component], f"P[{index}][{component}]")
            _seed(state.P_prev[component], f"P_prev[{index}][{component}]")
        _seed(state._scratch, f"_scratch[{index}]")  # noqa: SLF001
    return fields, grid, seeded


def compared_hosts(fields) -> Dict[str, np.ndarray]:
    """Every array the comparison covers: the nine stored volumes plus the
    pole slots, BY SLOT NAME."""
    out = {name: np.ascontiguousarray(to_host(getattr(fields, name))).copy()
           for name in sibling.STATE}
    for index, state in enumerate(fields.polarizations):
        for component in state.driven():
            out[f"P[{index}][{component}]"] = np.ascontiguousarray(
                to_host(state.P[component])).copy()
            out[f"P_prev[{index}][{component}]"] = np.ascontiguousarray(
                to_host(state.P_prev[component])).copy()
        out[f"_scratch[{index}]"] = np.ascontiguousarray(
            to_host(state._scratch)).copy()  # noqa: SLF001
    return out


def compare_all(reference, actual) -> Dict[str, int]:
    left = compared_hosts(reference)
    right = compared_hosts(actual)
    assert sorted(left) == sorted(right), (sorted(left), sorted(right))
    return {name: words_differ(left[name], right[name]) for name in left}


#: THE SHIPPED CLASSIFIER, not a second spelling of it. This gate carried its
#: own ``np.abs(x) < FLT_MIN`` counter until 2026-09-02 and it read ZERO
#: subnormals under the FLUSH policy on a seed that held 4629 of them -- because
#: `set_zero_subnormals(True)` turns on DAZ as well as FTZ, so every subnormal
#: OPERAND reads as zero and the comparison cannot see the words that are
#: actually in memory. The precondition then failed the whole flush leg on a
#: fixture that was correct. ``probe.operand_census`` classifies OFF THE BITS
#: (exponent field zero, mantissa nonzero), which is DAZ-immune, is what every
#: sibling CUDA gate uses, and whose own docstring says in as many words why
#: ``x < FLT_MIN`` is the wrong test. It also counts the signed zeros and zeros
#: this gate was not recording at all.
operand_census = probe.operand_census


# ---------------------------------------------------------------------------
# The weld route, the certified route, and the launch counting
# ---------------------------------------------------------------------------

class _CountingKernels:
    """A counting wrapper at the module's ``_get_kernel`` seam.

    The launcher's own result already reports its launches; this second counter
    exists so the two can disagree. It wraps the REAL compiled kernel object,
    so what runs is byte-for-byte the shipped binary.
    """

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


def weld_seam(fields, pml, arm) -> Dict[str, Any]:
    """The subject: three fused launches, host rotation between."""
    return weld.run_complex_no_pml_fused_polarization_pair(fields, pml, arm)


def certified_seam(fields, arm) -> Dict[str, Any]:
    """The certified singles: the stored-E launch, then the per-(state,
    component) ADE launches with the module's own rotation."""
    certified.update_E_complex_no_pml_stored(fields, arm)
    launches = certified.update_P_complex_no_pml(fields, arm)
    return {"update_P_launches": int(launches)}


def array_seam(fields, pml) -> None:
    stepping.update_E(fields, pml)
    stepping.update_P(fields, pml)


# ---------------------------------------------------------------------------
# Legs
# ---------------------------------------------------------------------------

def leg_driver_order() -> Dict[str, Any]:
    expected = ("update_E", "update_P")
    passed = tuple(weld.REPLACES) == expected
    return {"passed": bool(passed), "replaces": list(weld.REPLACES),
            "expected": list(expected)}


def leg_lift(arm) -> Dict[str, Any]:
    """The lift, measured two ways per emitted variant.

    FIRST the emit itself: the module derives its text from the CERTIFIED
    emissions through ``_certified_store_line``/``_certified_recurrence_line``
    plus anchored single replacements, and RAISES when any certified anchor is
    absent -- so an emit that returns at all has verified its own splice.
    SECOND this leg's independent read: the RENAMED anchors (the per-slot
    spellings ``LIFT_EDITS`` records -- the ``ev`` register in the store and in
    every recurrence's drive operand, the per-pole out/prev/sigma names) each
    appear EXACTLY ONCE in the emitted text, so a template edit that silently
    dropped a rename fails here rather than compiling."""
    checks: List[Dict[str, Any]] = []
    for spec in SPECS:
        kinds = tuple(kind == "volume" for kind in spec["kinds"])
        for component_index in range(3):
            stem = weld._COMPONENTS[component_index][2]  # noqa: SLF001
            count_name = weld._COMPONENTS[component_index][3]  # noqa: SLF001
            try:
                source = weld.fused_polarization_pair_no_pml_complex_source(
                    arm, component_index, len(kinds), kinds)
                emitted = True
            except Exception as exc:  # noqa: BLE001
                checks.append({"spec": spec["label"],
                               "component_index": component_index,
                               "line": "emit", "present": False,
                               "why": f"{type(exc).__name__}: {exc}"})
                continue
            banks = ", ".join(f"{stem}{slot}"
                              for slot in range(int(weld.MAX_POLES)))
            wanted = [
                ("store_register", (
                    f"    cf ev = mul_field_left(\n"
                    f"        minus_poles(g{component_index}, {banks}, idx, "
                    f"{count_name}),\n"
                    f"        inv_eps_{component_index}[idx]);\n"
                    f"    cf_store(f{component_index}, idx, ev);\n")),
            ]
            for index, sigma_is_volume in enumerate(kinds):
                sigma_read = (f"sigma_{index}[idx]" if sigma_is_volume
                              else f"sigma_{index}")
                wanted.append((f"recurrence_pole_{index}", (
                    f"    cf_store(p_out_{index}, idx, ade_step(\n"
                    f"        cf_load({stem}{index}, idx), "
                    f"cf_load(p_prev_{index}, idx), ev,\n"
                    f"        {sigma_read}, c_now_{index}, c_prev_{index}, "
                    f"c_drive_{index}));\n")))
            for label, line in wanted:
                checks.append({
                    "spec": spec["label"], "component_index": component_index,
                    "line": label,
                    "present": bool(emitted) and source.count(line) == 1,
                })
    passed = all(check["present"] for check in checks)
    return {"passed": bool(passed), "checks": checks}


def leg_refusal(arm, licence) -> Dict[str, Any]:
    """Every configuration the predicate refuses BY NAME, built and refused,
    plus the admitted control built by this gate's own builder."""
    checks: List[Dict[str, Any]] = []

    def record(name: str, expectation: str, fields, pml, grid,
               license_value="use") -> None:
        value = licence if license_value == "use" else license_value
        covered, reason = weld.covers_complex_no_pml_fused_polarization_pair(
            fields, pml, grid, (), value, POLICY["name"])
        checks.append({"case": name, "expected": expectation,
                       "covered": bool(covered), "reason": str(reason),
                       "passed": (not covered) if expectation == "refused"
                                 else bool(covered)})

    fields, grid, _seeded = build(SPECS[0], "uniform", 11)
    record("the_fixture_this_gate_sweeps", "covered", fields, None, grid)
    record("no_licence", "refused", fields, None, grid, license_value=None)

    real_grid = Grid(resolution=1.0, cell_size=(5.0, 6.0, 7.0),
                     boundaries=("periodic",) * 3, dimensions=3, xp=cp,
                     courant=0.5)
    real_fields = Fields(grid=real_grid)
    real_fields.enable_field_storage()
    real_fields.polarizations.append(PolarizationState(
        Susceptibility(frequency=1.1, gamma=0.05), 0.3, real_grid,
        real_fields._field_dtype()))  # noqa: SLF001
    record("real_storage_belongs_to_the_real_pairs", "refused", real_fields,
           None, real_grid)

    from meep_gpu.pml import PML  # noqa: PLC0415
    active_grid = Grid(resolution=1.0, cell_size=(5.0, 6.0, 7.0),
                       boundaries=("periodic", "periodic", "metallic"),
                       dimensions=3, xp=cp, courant=0.5)
    active = Fields(grid=active_grid, force_complex_fields=True)
    active.enable_field_storage()
    active.enable_pml_storage()
    active.polarizations.append(PolarizationState(
        Susceptibility(frequency=1.1, gamma=0.05), 0.3, active_grid,
        active._field_dtype()))  # noqa: SLF001
    record("an_active_absorber", "refused", active,
           PML(grid=active_grid, thickness=((0, 0), (0, 0), (2, 2))),
           active_grid)

    over = Fields(grid=grid, force_complex_fields=True)
    over.enable_field_storage()
    for index in range(int(weld.MAX_POLES) + 1):
        over.polarizations.append(PolarizationState(
            Susceptibility(frequency=1.05 + 0.1 * index, gamma=0.05),
            0.2 + 0.01 * index, grid, over._field_dtype()))  # noqa: SLF001
    record("past_the_certified_bank_width", "refused", over, None, grid)

    passed = all(check["passed"] for check in checks)
    return {"passed": passed, "checks": checks,
            "admitted_control_present": any(
                c["expected"] == "covered" and c["passed"] for c in checks)}


def run_case(spec: Dict[str, Any], value_class: str, steps: int, arm,
             seam: Optional[Callable] = None,
             label_suffix: str = "") -> Dict[str, Any]:
    """One fixture swept ``steps`` COMPLETE driver steps, three engines abreast:
    the array-path oracle, the certified-singles control, and the weld."""
    started = time.time()
    label = f"{spec['label']}/{value_class}{label_suffix}"
    seed = abs(hash(("complex_polarization", label))) % (2 ** 31)
    reference, ref_grid, seeded = build(spec, value_class, seed)
    control, control_grid, _control_seed = build(spec, value_class, seed)
    actual, grid, _actual_seed = build(spec, value_class, seed)
    out: Dict[str, Any] = {"label": label, "value_class": value_class,
                           "steps_requested": steps,
                           "operand_census": operand_census(seeded)}

    covered, reason = weld.covers_complex_no_pml_fused_polarization_pair(
        actual, None, grid, (), LICENCE["verdict"], POLICY["name"])
    if not covered:
        out.update({"passed": False,
                    "why": f"the predicate refused the sweep fixture: {reason}"})
        return out

    initial = compared_hosts(reference)
    weld_launches: List[int] = []
    counted_launches: List[int] = []
    certified_p_launches: List[int] = []
    diffs_vs_array: List[int] = []
    diffs_vs_certified: List[int] = []
    run_seam = seam if seam is not None else weld_seam
    for _step in range(1, steps + 1):
        for name in BEFORE_SEAM:
            _run_pass(name, reference, None)
            _run_pass(name, control, None)
            _run_pass(name, actual, None)
        array_seam(reference, None)
        certified_p_launches.append(
            certified_seam(control, arm)["update_P_launches"])
        with _CountingKernels(weld) as counter:
            result = run_seam(actual, None, arm)
        weld_launches.append(int(result.get("launches", -1))
                             if isinstance(result, dict) else -1)
        counted_launches.append(counter.count)
        cp.cuda.runtime.deviceSynchronize()
        diffs_vs_array.append(sum(compare_all(reference, actual).values()))
        diffs_vs_certified.append(sum(compare_all(control, actual).values()))
        if diffs_vs_array[-1] or diffs_vs_certified[-1]:
            break

    final = compared_hosts(actual)
    moved = {name: words_differ(initial[name], final[name]) for name in initial}
    must_move = list(COMPONENTS) + [name for name in initial
                                    if name.startswith("P[")]
    if value_class != "uniform":
        must_move = list(COMPONENTS)
    frozen = [name for name in must_move if moved.get(name, 0) == 0]
    expected_p = 1 + sum(
        len(weld._driving_states(actual, target))  # noqa: SLF001
        for target in COMPONENTS)
    out.update({
        "steps_run": len(diffs_vs_array),
        "bit_identical_to_the_array_path": not any(diffs_vs_array),
        "weld_bit_identical_to_the_certified_singles":
            not any(diffs_vs_certified),
        "first_divergence_vs_array": next(
            (i + 1 for i, n in enumerate(diffs_vs_array) if n), None),
        "first_divergence_vs_certified": next(
            (i + 1 for i, n in enumerate(diffs_vs_certified) if n), None),
        "weld_launches_per_step_reported": sorted(set(weld_launches)),
        "weld_launches_per_step_counted": sorted(set(counted_launches)),
        "certified_route_launches_per_step": sorted(
            {1 + n for n in certified_p_launches}),
        "certified_route_expected": expected_p,
        "moved_words": int(sum(moved.values())),
        "arrays_that_never_moved": sorted(
            name for name, count in moved.items() if count == 0),
        "movement_floor_frozen": frozen,
        "launch_counts_agree": (
            sorted(set(weld_launches)) == [3]
            and sorted(set(counted_launches)) == [3]),
        "seconds": time.time() - started,
    })
    out["passed"] = bool(
        len(diffs_vs_array) == steps and not any(diffs_vs_array)
        and not any(diffs_vs_certified) and not frozen
        and out["launch_counts_agree"])
    return out


# ---------------------------------------------------------------------------
# Device mutations, through the shipped launcher's own compile seam
# ---------------------------------------------------------------------------

#: ``{i}`` is the component index and ``{s}`` the component's bank stem
#: (``a``/``b``/``c``), substituted per emitted variant -- the bank names are
#: per component, so a single spelling would needle only ``Ex``'s kernel.
DEVICE_MUTATIONS: Tuple[Dict[str, Any], ...] = (
    {"name": "reload_drive_from_global",
     "old": "cf_load(p_prev_0, idx), ev,",
     "new": "cf_load(p_prev_0, idx), cf_load(f{i}, idx),",
     "expect": "inert",
     "why": "THE SEAM NULL: pole 0's drive becomes a reload of the word the E "
            "half just stored. A complex64 word pair stored to global and "
            "reloaded is the identity on the bits, so this MUST be uncaught -- "
            "it is what shows the register IS the certified load."},
    {"name": "wrong_drive_register",
     "old": "cf_load(p_prev_0, idx), ev,",
     "new": "cf_load(p_prev_0, idx), cf_load(g{i}, idx),",
     "expect": "caught",
     "why": "pole 0 driven by the DISPLACEMENT instead of the stored E -- the "
            "certified ADE gate's own wrong-drive control, restated against "
            "the fused text. MUST diverge, or the null above proves nothing."},
    {"name": "store_lands_on_p_now",
     "old": "cf_store(p_out_0, idx, ade_step(",
     "new": "cf_store((float*){s}0, idx, ade_step(",
     "expect": "caught",
     "why": "pole 0's store lands on P^n instead of the scratch: the "
            "second-order recurrence silently reduced to first order, plus a "
            "scratch the rotation then promotes unwritten."},
    {"name": "bank_lane_swapped",
     "old": "cf_load({s}1, idx), cf_load(p_prev_1, idx)",
     "new": "cf_load({s}0, idx), cf_load(p_prev_1, idx)",
     "expect": "caught", "min_poles": 2,
     "why": "pole 1's recurrence reads pole 0's P^n: the bank binding "
            "rotated by one lane. Scored only on two-state fixtures, where a "
            "second lane exists to swap."},
)


def leg_mutations(arm, steps: int) -> List[Dict[str, Any]]:
    legs: List[Dict[str, Any]] = []
    for mutation in DEVICE_MUTATIONS:
        for spec in SPECS:
            if len(spec["kinds"]) < int(mutation.get("min_poles", 0)):
                continue
            kinds = tuple(kind == "volume" for kind in spec["kinds"])
            compiled: Dict[Tuple[int, int, Tuple[bool, ...]], Any] = {}
            sites_by_variant: Dict[Tuple[int, int, Tuple[bool, ...]], int] = {}
            missed = False
            for component_index in range(3):
                key = (component_index, len(kinds), kinds)
                stem = weld._COMPONENTS[component_index][2]  # noqa: SLF001
                old = mutation["old"].format(i=component_index, s=stem)
                new = mutation["new"].format(i=component_index, s=stem)
                source = weld.fused_polarization_pair_no_pml_complex_source(
                    arm, component_index, len(kinds), kinds)
                hits = source.count(old)
                if hits == 0:
                    missed = True
                    break
                mutated = source.replace(old, new)
                compiled[key] = cp.RawKernel(mutated, weld.KERNEL_NAME,
                                             options=("--fmad=false",))
                sites_by_variant[key] = hits
            if missed:
                legs.append({"leg": mutation["name"], "label": spec["label"],
                             "scored": False, "passed": False,
                             "why_not_scored": (
                                 f"the needle {mutation['old']!r} matched "
                                 f"nothing in an emitted variant; a mutation "
                                 f"that changed no byte scores nothing")})
                log(f"    mutation {mutation['name']} [{spec['label']}]: "
                    f"NEEDLE MISSED")
                continue

            original = weld._get_kernel  # noqa: SLF001

            def patched(arm_value, component_index, count, kinds_value,
                        _table=compiled):
                return _table[(component_index, count, tuple(kinds_value))]

            weld._get_kernel = patched  # noqa: SLF001
            try:
                case = run_case(spec, "uniform", steps, arm,
                                label_suffix=f"|{mutation['name']}")
            finally:
                weld._get_kernel = original  # noqa: SLF001
            diverged = not (case.get("bit_identical_to_the_array_path", False)
                            and case.get(
                                "weld_bit_identical_to_the_certified_singles",
                                False))
            passed = diverged if mutation["expect"] == "caught" else (
                not diverged and case.get("passed", False))
            legs.append({"leg": mutation["name"], "label": spec["label"],
                         "scored": True, "expect": mutation["expect"],
                         "diverged": bool(diverged), "passed": bool(passed),
                         "needle_sites": dict(
                             (str(k), v) for k, v in sites_by_variant.items()),
                         "first_divergence": case.get(
                             "first_divergence_vs_array")})
            log(f"    mutation {mutation['name']} [{spec['label']}]: "
                f"{'PASS' if passed else 'FAIL'} (diverged={diverged})")
    return legs


# ---------------------------------------------------------------------------
# The rotation legs: local copies of the shipped loop, one defect each
# ---------------------------------------------------------------------------

def _loop_copy(defect: Optional[str]):
    """A LOCAL copy of the shipped component loop, with one named defect.

    ``None`` is the NULL: the copy with no defect, required identical so the
    copy itself is shown equivalent before any defect is blamed on it.
    """

    def run(fields, pml, arm, kernel=None):  # noqa: ARG001
        launches = 0
        components = [target for target, _d, _s, _c in weld._COMPONENTS]  # noqa: SLF001
        for order, target in enumerate(components):
            specs = weld.component_specs(fields)[target]
            states = specs["states"]
            if defect == "rotate_before_launch":
                for state in states:
                    p = state.P[target]
                    p_prev = state.P_prev[target]
                    scratch = state._scratch  # noqa: SLF001
                    state.P[target] = scratch
                    state.P_prev[target] = p
                    state._scratch = p_prev  # noqa: SLF001
            weld.launch_complex_no_pml_fused_polarization_component(
                fields, arm, target, states)
            launches += 1
            if defect == "rotate_before_launch":
                continue
            for index, state in enumerate(states):
                if (defect == "freeze_one_states_rotation"
                        and order == 0 and index == 0):
                    continue
                p = state.P[target]
                p_prev = state.P_prev[target]
                scratch = state._scratch  # noqa: SLF001
                state.P[target] = scratch
                state.P_prev[target] = p
                state._scratch = p_prev  # noqa: SLF001
        if defect == "rotate_after_all_launches":
            raise AssertionError("handled by its own runner")
        return {"launched": True, "launches": launches}

    def run_rotate_after(fields, pml, arm, kernel=None):  # noqa: ARG001
        launches = 0
        pending: List[Tuple[str, Any]] = []
        for target, _d, _s, _c in weld._COMPONENTS:  # noqa: SLF001
            states = weld.component_specs(fields)[target]["states"]
            weld.launch_complex_no_pml_fused_polarization_component(
                fields, arm, target, states)
            launches += 1
            pending.extend((target, state) for state in states)
        for target, state in pending:
            p = state.P[target]
            p_prev = state.P_prev[target]
            scratch = state._scratch  # noqa: SLF001
            state.P[target] = scratch
            state.P_prev[target] = p
            state._scratch = p_prev  # noqa: SLF001
        return {"launched": True, "launches": launches}

    return run_rotate_after if defect == "rotate_after_all_launches" else run


ROTATION_LEGS: Tuple[Tuple[str, Optional[str], str], ...] = (
    ("launcher_copy_NULL", None, "inert"),
    ("rotate_before_launch", "rotate_before_launch", "caught"),
    ("rotate_after_all_launches", "rotate_after_all_launches", "caught"),
    ("freeze_one_states_rotation", "freeze_one_states_rotation", "caught"),
)


def leg_rotation(arm, steps: int) -> List[Dict[str, Any]]:
    legs: List[Dict[str, Any]] = []
    spec = SPECS[0]  # two states, mixed kinds: every rotation defect can bite
    for name, defect, expect in ROTATION_LEGS:
        case = run_case(spec, "uniform", steps, arm, seam=_loop_copy(defect),
                        label_suffix=f"|{name}")
        diverged = not (case.get("bit_identical_to_the_array_path", False)
                        and case.get(
                            "weld_bit_identical_to_the_certified_singles",
                            False))
        passed = diverged if expect == "caught" else (
            not diverged and case.get("passed", False))
        legs.append({"leg": name, "label": spec["label"], "expect": expect,
                     "diverged": bool(diverged), "passed": bool(passed),
                     "first_divergence": case.get("first_divergence_vs_array")})
        log(f"    rotation {name}: {'PASS' if passed else 'FAIL'} "
            f"(diverged={diverged})")
    return legs


# ---------------------------------------------------------------------------
# Driving
# ---------------------------------------------------------------------------

LICENCE: Dict[str, Any] = {"verdict": None}
POLICY: Dict[str, Any] = {"name": None}


def save(results: Dict[str, Any], out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    gate_provenance.stamp(results)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True, default=str)
    os.replace(tmp, out_path)


def summarize(results: Dict[str, Any], steps: int) -> Dict[str, Any]:
    cases = results.get("cases", [])
    mutations = results.get("mutations", [])
    rotations = results.get("rotation", [])
    scored = [m for m in mutations if m.get("scored")]
    verdict = {
        "steps_per_case": steps,
        "cases": len(cases),
        "cases_passed": sum(1 for c in cases if c.get("passed")),
        "cases_bit_identical_to_the_array_path": sum(
            1 for c in cases if c.get("bit_identical_to_the_array_path")),
        "cases_weld_bit_identical_to_the_certified_singles": sum(
            1 for c in cases
            if c.get("weld_bit_identical_to_the_certified_singles")),
        "mutations_scored": len(scored),
        "mutation_legs_that_failed": [
            f"{m['leg']}|{m.get('label')}" for m in mutations
            if not m.get("passed")],
        "rotation_legs_that_failed": [
            r["leg"] for r in rotations if not r.get("passed")],
        "refusal_leg_passed": bool(results.get("refusal", {}).get("passed")),
        "lift_leg_passed": bool(results.get("lift", {}).get("passed")),
        "driver_order_passed": bool(
            results.get("driver_order", {}).get("passed")),
        "band_contains_subnormals": all(
            c["operand_census"]["subnormals"] > 0 for c in cases
            if c.get("value_class") == "subnormal_band"
            and "operand_census" in c),
        "uniform_contains_none": all(
            c["operand_census"]["subnormals"] == 0 for c in cases
            if c.get("value_class") == "uniform" and "operand_census" in c),
    }
    verdict["released"] = bool(
        cases and verdict["cases_passed"] == len(cases)
        and verdict["cases_bit_identical_to_the_array_path"] == len(cases)
        and verdict["cases_weld_bit_identical_to_the_certified_singles"]
            == len(cases)
        and scored and not verdict["mutation_legs_that_failed"]
        and rotations and not verdict["rotation_legs_that_failed"]
        and verdict["refusal_leg_passed"] and verdict["lift_leg_passed"]
        and verdict["driver_order_passed"]
        and verdict["band_contains_subnormals"]
        and verdict["uniform_contains_none"])
    return verdict


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--product", choices=("full", "reduced"),
                        default="full")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--subnormal-policy", choices=("keep", "flush"),
                        required=True)
    parser.add_argument("--import-meep-for-host-policy", action="store_true")
    parser.add_argument("--expansion-probe", default=None)
    parser.add_argument("--skip-mutations", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    started = time.time()
    results: Dict[str, Any] = {
        "gate": "cuda_complex_polarization_pair",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                     time.gmtime(started)),
        "host": __import__("socket").gethostname(),
        "argv": list(argv or sys.argv[1:]),
        "steps": args.steps,
        "subnormal_policy": args.subnormal_policy,
        "module": {
            "family": weld.FAMILY,
            "kernel": weld.KERNEL_NAME,
            "replaces": list(weld.REPLACES),
            "carries_deposit_repair": bool(weld.CARRIES_DEPOSIT_REPAIR),
            "certified_kernels": list(weld.CERTIFIED_KERNELS),
            "uncertified_kernels": dict(weld.UNCERTIFIED_KERNELS),
        },
    }
    gate_provenance.stamp(results)
    POLICY["name"] = args.subnormal_policy

    if args.import_meep_for_host_policy:
        results["meep_import_for_host_policy"] = (
            probe.import_meep_for_host_policy())
    if cp is not None:
        results["subnormal_policy_install"] = (
            probe.install_subnormal_policy_for_run(args.subnormal_policy,
                                                   _REPO_API))
        # THE DEVICE THIS RAN ON, in the one spelling the recorders read. Every
        # other CUDA gate stamps this block (``gate_cuda_complex_no_pml.py`` sets
        # it in the same position, between the policy install and the policy
        # stamp); this gate did not, and the omission is why
        # ``record_cuda_regate.py`` could not describe a campaign covering it:
        # ``_stamps`` collects ``payload["environment"]`` across the tree and
        # ``build_block`` FAILS CLOSED unless exactly one environment is found,
        # so zero stamps refuses with an empty disagreement list. Derived from the
        # run by ``probe.device_info()`` rather than typed, so the block is the
        # device's own answer and not a claim about it.
        results["environment"] = probe.device_info()
    results["subnormal_policy_stamp"] = probe.subnormal_policy_stamp(_REPO_API)

    licence = sibling.load_licence(args.subnormal_policy, args.expansion_probe)
    results["licence"] = {key: licence.get(key) for key in
                          ("arm", "record", "record_sha256", "policy_reasons",
                           "refusals", "basis")}
    LICENCE["verdict"] = licence
    usable = bool(licence.get("arm") and not licence.get("refusals")
                  and not licence.get("policy_reasons"))

    results["driver_order"] = leg_driver_order()
    log(f"driver order: "
        f"{'PASS' if results['driver_order']['passed'] else 'FAIL'}")
    arm = licence.get("arm") or "NAIVE"
    results["lift"] = leg_lift(arm)
    log(f"lift: {'PASS' if results['lift']['passed'] else 'FAIL'}")
    save(results, args.out)

    if cp is None:
        results["verdict"] = {"released": False,
                              "why": "CuPy is absent; only the source legs ran"}
        save(results, args.out)
        return 1
    if not usable:
        results["verdict"] = {
            "released": False,
            "why": f"the expansion licence is unusable: "
                   f"{licence.get('policy_reasons') or licence.get('refusals')}"}
        save(results, args.out)
        return 1

    results["refusal"] = leg_refusal(arm, licence)
    log(f"refusal: {'PASS' if results['refusal']['passed'] else 'FAIL'}")
    save(results, args.out)

    specs = SPECS
    if args.product == "reduced":
        specs = tuple(s for s in SPECS if s["label"] in REDUCED_LABELS)
    cases: List[Dict[str, Any]] = []
    results["cases"] = cases
    for spec in specs:
        for value_class in VALUE_CLASSES:
            case = run_case(spec, value_class, args.steps, arm)
            cases.append(case)
            log(f"  case {case['label']}: "
                f"{'PASS' if case.get('passed') else 'FAIL'} "
                f"steps={case.get('steps_run')} "
                f"launches={case.get('weld_launches_per_step_counted')} "
                f"certified={case.get('certified_route_launches_per_step')} "
                f"({case.get('seconds', 0.0):.1f} s)")
            save(results, args.out)

    results["rotation"] = leg_rotation(arm, min(12, args.steps))
    save(results, args.out)

    if args.skip_mutations:
        results["mutations"] = []
    else:
        results["mutations"] = leg_mutations(arm, min(12, args.steps))
    save(results, args.out)

    results["verdict"] = summarize(results, args.steps)
    results["elapsed_s"] = time.time() - started
    save(results, args.out)
    log(f"verdict released={results['verdict']['released']} "
        f"({results['elapsed_s']:.1f} s)")
    return 0 if results["verdict"]["released"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
