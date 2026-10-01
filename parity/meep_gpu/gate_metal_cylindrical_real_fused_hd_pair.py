"""Byte gate for the Metal Dcyl m = 0 fused H->D pair: ``update_H`` + the radial prefix
in one launch, the certified ``step_D`` curl in the other.

THE CLAIM THIS GATE IS ALLOWED TO SUPPORT: for every configuration
:func:`~meep_gpu.metal_kernels.cylindrical_real_fused_hd_pair.metal_cylindrical_real_fused_hd_pair_coverage`
admits, the plan's two launches -- ``cyl_real_hd_constitutive_prefix`` (the pointwise
``update_H`` into scratch plus the column-leader radial scan over the RECOMPUTED ``Hy``)
then the certified ``cyl_pml_curl_step`` for ``step_D`` over the rotated ``H`` -- leave the
engine in a state that is BIT-IDENTICAL, PER COMPLETE DRIVER STEP, to

  * the NumPy array path under ``FdtdDriver.step``'s own loop,
  * the two SEPARATELY CERTIFIED Metal products it replaces -- the ``cylindrical m=0``
    constitutive arm on ``update_H`` and the ``cylindrical m=0`` curl plan (scan, then
    curl) on ``step_D``,
  * the composition ``plan_step(fuse=True)`` installs today, and the same slots
    dispatched unfused,

over every stored volume, compared as uint32 words. ``allclose`` appears nowhere.

THE QUESTION THIS SEAM TURNS ON, and the leg that answers it: the radial prefix's source
is what the seam's first half WRITES. Launch 1 never reads its own output -- the column
leader RECOMPUTES ``Hy`` row by row from the pre-launch state -- and leg
``recompute_identity`` measures that the prefix so formed is word-identical to the
SHIPPED device scan run over the ``H`` launch 1 stored, and to
``stepping.cylindrical_rderiv_prefix`` itself over the synced host ``Hy``, with the same
scan over the PRE-launch ``Hy`` as the control that must differ.

THE LEGS

  1  binding_ceiling      launch 1's 25 bindings COMPILE, +6 (31) compiles, +7 (32) is
                          REFUSED with the platform's own error, the ceiling is bisected
                          on this host, and launch 2 (the certified curl, 22) compiles
  2  transcription        launch 1 is the certified ``update_H`` cell function plus the
                          certified scan text with EXACTLY the two declared edits, reads
                          nothing it writes; launch 2 IS the certified curl, byte for byte
  3  scan_order_host      ``numpy.cumsum`` is a strictly-serial accumulation over the
                          increments the SHIPPED prefix forms, at ir0 = 0.5, on the
                          fixture and corpus radial extents; a blocked scan must differ
  4  recompute_identity   the prefix from the recomputed ``Hy`` equals the shipped scan
                          over the stored ``H`` (device) and the oracle (host); the scan
                          over the pre-launch ``Hy`` must differ
  5  driver_order         REPLACES is the driver's two adjacent consults with only the
                          electric withdraw between them, read off ``driver.py`` by ast
  6  refusal              undeclared sources, a standing integrated withdraw, |m| >= 1,
                          Cartesian -- each refused by name; a non-integrated electric
                          source admitted
  7  arbitration          the shipped composer refuses this product today (no absorb
                          row); with the row applied IN-PROCESS and INSTALLABLE held
                          True it still refuses it, naming the released D->E incumbent,
                          and the installed composition does not move
  8  product              every case, six arrangements, bit-identical per complete
                          step, the prefix against the oracle every step, both launch
                          counters advancing, two launch witnesses agreeing
  9  value_class          the same on the +-0 lattice, plus the subnormal ladder
 10  launch_structure     launches per step at the seam (singles 3, this 2) and over
                          the whole step (composition today 5, unfused 6, this 5)
 11  byte_neutral         the leader's row-0 recompute replaced by the register must
                          NOT diverge
 12  mutation             armed defects, each scored: caught, a DECLARED null (confirmed
                          with a live sibling), or -- for one race -- schedule-dependent
                          and recorded on two grid sizes, must-catch on the large one
 13  disarm               the same harness, shipped bytes, must not diverge
 14  lift                 the cell's three corpus rows, re-lifted from MEEP in the
                          census's own child and driven through the same six
                          arrangements

WHAT THE GATE REFUSES TO INFER

* **A no-op agreeing with a no-op is trivially identical.** Every compared volume must
  MOVE during a synthetic case; a lifted row must move the seam's own outputs.
* **A DECLARED NULL MUST BE CONFIRMED**, and each has a live sibling: the commuted weight
  multiply and accumulator (float32 multiply and add are commutative to the bit) beside
  the reciprocal-multiply divide, which is not; ``dtdx`` distributed over Dz's OUTER sum
  (its phi pair is an exact +0.0 on the one-cell invariant axis) beside ``dtdx``
  distributed INTO the prefix difference, which is the ``bz_flat_grouping`` class and is
  caught.
* **A race is not a null and not a catch.** The leader reading ``H_out`` instead of
  recomputing read the intended value on a 400-thread grid and the wrong one on the
  corpus-sized grids (measured before this gate was written). It is armed on both: the
  large grid must catch it, the small grid's outcome is recorded and does not decide the
  verdict, and the property it would violate -- zero reads of anything the launch writes
  -- is pinned as SOURCE TEXT by leg 2.
* **The composition this product would install into is out of reach until wiring.** The
  product is not in the composer's tables; leg 7 measures what the composer does with it
  today and with the row applied in-process, and says which is which.

Progress reporting: one flushed line per case, every row appended and fsynced as it lands; the lift
leg writes one JSON per corpus row and a progress log the child appends to.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")
HERE = Path(__file__).resolve().parent
API_ROOT = HERE.parents[1]
for _path in (str(API_ROOT), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402

import h_to_d_seam  # noqa: E402
import metal_value_classes as values  # noqa: E402

from meep_gpu import stepping, withdraw_hoist  # noqa: E402
from meep_gpu.fastpath import SYNC_PASS_OWNERS, SYNC_PATH_SLOTS  # noqa: E402
from meep_gpu.metal_kernels import (  # noqa: E402
    cylindrical_real as cyl,
    cylindrical_real_fused_electric_pair as electric_twin,
    cylindrical_real_fused_hd_pair as family,
    cylindrical_real_fused_magnetic_pair as magnetic_twin,
    fused_hd_pair as cartesian,
    launch as metal_launch,
    shaders,
    subnormal,
)
from meep_gpu.metal_kernels.device import (  # noqa: E402
    MAX_BUFFER_BINDINGS, Residency, compile_source, metal_frontend_version,
)
from meep_gpu.triton_kernels.launch import STEP_ORDER  # noqa: E402

# ---------------------------------------------------------------------------
# The battery contract, so the census driver can lift a corpus row into this file
# ---------------------------------------------------------------------------

#: ``measure_predicate_coverage`` digests this package into every lifted row's
#: ``subject_manifest_sha256``. Declared, as every battery declares it.
SUBJECT_PACKAGE = "metal_kernels"

#: The budget every synthetic case runs -- the cylindrical siblings' twelve. The
#: comparison is per COMPLETE DRIVER STEP and the state this weld carries between
#: steps (the rotated ``H``/``f_w_H`` pair, the in-place ``fu_D``, the prefix) compounds.
STEPS = 12

#: How many clean complete steps a LIFTED corpus row must reach to count. A lifted
#: row starts at the engine's zeros and is driven by its own source, so its
#: wavefront's leading cells enter the float32 denormal band on their own schedule;
#: past that step MPS flushes where NumPy does not and no byte claim can be made. A
#: floor, not a target: the record carries each row's own count.
LIFT_CLEAN_STEP_FLOOR = 8

#: The census this gate lifts its corpus rows from, and the seam record whose
#: ``withdraw_in_seam`` flag names the rows the predicate must refuse.
CENSUS = "metal_coverage_2026-09-04_m0complex"
SEAM_RECORD = "h_to_d_seam_2026-09-04"
CELL_ARMS: Tuple[str, str] = ("cylindrical m=0", "cylindrical m=0")

EXIT_INCOMPLETE = 75

#: name -> the grid this case builds. The rows are the two released cylindrical
#: pairs' gates', deliberately, plus one at the corpus's own scale:
#:
#:   THE z DECLARATION   a metallic z puts ``zero_metal_B``/``zero_metal_D`` in the
#:                       live set (they stay on the array path here; neither is inside
#:                       this seam); a periodic z drops them. Both are walked.
#:   THE COURANT NUMBER  three of the five are NOT powers of two, which is the family
#:                       of non-representable multiplicands the association defects
#:                       need in order to be visible at all.
#:   THE SHAPE           square, tall-and-thin, short-and-wide (the scan's column count
#:                       and row count trade off), the composition matrix's own
#:                       (16, 1, 20) row, and a 120 x 120 grid at the corpus rows'
#:                       scale -- the one on which the scratch-read race is caught.
CASES: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("square_metallic", dict(shape=(20, 1, 20), courant=0.5, z_kind="metallic")),
    ("square_metallic_odd_courant",
     dict(shape=(20, 1, 20), courant=0.3141592653589793, z_kind="metallic")),
    ("tall_metallic_odd_courant",
     dict(shape=(32, 1, 13), courant=0.4142135623730951, z_kind="metallic")),
    ("wide_metallic_odd_courant",
     dict(shape=(13, 1, 32), courant=0.37, z_kind="metallic")),
    ("square_periodic", dict(shape=(20, 1, 20), courant=0.5, z_kind="periodic")),
    ("composition_matrix_row",
     dict(shape=(16, 1, 20), courant=0.37, z_kind="metallic", pml=(4, 4))),
    ("corpus_scale_metallic_odd_courant",
     dict(shape=(120, 1, 120), courant=0.3141592653589793, z_kind="metallic")),
)

#: The case the mutations are armed on: SQUARE (so the coefficient-index defect reads a
#: length-nr vector at the z index IN BOUNDS and measures a wrong value) at a
#: non-power-of-two Courant (so the grouping defects are visible at all).
MUTATION_CASE = "square_metallic_odd_courant"

#: The case the scratch-read RACE must be caught on: the corpus rows' own scale.
RACE_CASE = "corpus_scale_metallic_odd_courant"

REFUSAL_CASE = "square_metallic"

#: The two epsilon tables the composition matrix's own ``_epsilon`` uses.
EPSILON: Dict[str, float] = {"Ex": 2.0, "Ey": 2.5, "Ez": 3.0}

LEG_GROUPS: Dict[str, Tuple[str, ...]] = {
    "host": ("transcription", "scan_order_host", "driver_order", "refusal",
             "arbitration"),
    "compile": ("binding_ceiling", "mutants_compile"),
    "device": ("recompute_identity", "product", "value_class", "launch_structure",
               "byte_neutral", "mutation", "disarm", "lift"),
}
ALL_LEGS: Tuple[str, ...] = tuple(leg for group in LEG_GROUPS.values() for leg in group)


def log(message: str) -> None:
    print(message, flush=True)


def words(array: Any) -> np.ndarray:
    """One array as uint32 WORDS. Byte compares, never allclose."""
    return np.frombuffer(np.ascontiguousarray(array).tobytes(), dtype=np.uint32)


def differing(left: Any, right: Any) -> int:
    a, b = words(left), words(right)
    if a.shape != b.shape:
        return max(a.size, b.size)
    return int(np.count_nonzero(a != b))


# ---------------------------------------------------------------------------
# The state: every stored volume, captured and restored IN PLACE
# ---------------------------------------------------------------------------

_RESET_VOLUMES: Optional[Tuple[str, ...]] = None


def reset_declared_volumes() -> Tuple[str, ...]:
    """Every array ``Fields.reset`` zeroes, read out of ``fields.py``'s own text.

    DERIVED, NOT TRANSCRIBED: a volume added to the engine joins the comparison
    without an edit here. Private scratch is dropped by the leading-underscore rule.
    """
    global _RESET_VOLUMES  # noqa: PLW0603 - read once from disk, then held
    if _RESET_VOLUMES is not None:
        return _RESET_VOLUMES
    text = (API_ROOT / "meep_gpu" / "fields.py").read_text(encoding="utf-8")
    match = re.search(r"\n    def reset\(self\).*?(?=\n    def )", text, re.S)
    if match is None:
        raise SystemExit("cannot find Fields.reset in fields.py; refusing to guess "
                         "the list of stored volumes")
    names = set(re.findall(r"self\.([A-Za-z_][A-Za-z0-9_]*)", match.group(0)))
    _RESET_VOLUMES = tuple(sorted(name for name in names - {"polarizations"}
                                  if not name.startswith("_")))
    return _RESET_VOLUMES


def stored_volumes(fields: Any) -> Dict[str, Any]:
    """Every stored volume this engine allocates, LIVE (not copied), by name."""
    out: Dict[str, Any] = {}
    for name in reset_declared_volumes():
        array = getattr(fields, name, None)
        if array is not None:
            out[name] = array
    for index, state in enumerate(getattr(fields, "polarizations", ()) or ()):
        for component, array in getattr(state, "P", {}).items():
            out[f"P[{index}].{component}"] = array
        for component, array in getattr(state, "P_prev", {}).items():
            out[f"P_prev[{index}].{component}"] = array
    return out


def capture(driver: Any) -> Dict[str, Any]:
    return {
        "volumes": {name: np.array(array, copy=True)
                    for name, array in stored_volumes(driver.fields).items()},
        "dipoles": [getattr(source, "_applied_dipole", None)
                    for source in getattr(driver, "_sources", ())],
        "step_count": int(driver.step_count),
    }


def restore(driver: Any, snapshot: Mapping[str, Any]) -> List[str]:
    lazily: List[str] = []
    for name, array in stored_volumes(driver.fields).items():
        saved = snapshot["volumes"].get(name)
        if saved is None:
            array.fill(0)
            lazily.append(name)
        else:
            array[...] = saved
    for source, dipole in zip(getattr(driver, "_sources", ()), snapshot["dipoles"]):
        if dipole is not None:
            source._applied_dipole = dipole  # noqa: SLF001 - the offset IS the state
    driver.step_count = snapshot["step_count"]
    return lazily


def compare_snapshots(reference: Mapping[str, Any],
                      other: Mapping[str, Any]) -> Dict[str, int]:
    a, b = reference["volumes"], other["volumes"]
    assert set(a) == set(b), sorted(set(a) ^ set(b))
    return {name: n for name in sorted(a) if (n := differing(a[name], b[name]))}


def _classify(word: int) -> str:
    if not word & 0x7FFFFFFF:
        return "zero"
    if not word & 0x7F800000:
        return "subnormal"
    return "normal"


def autopsy(reference: Mapping[str, Any], other: Mapping[str, Any],
            difference: Mapping[str, int], limit: int = 8) -> Dict[str, Any]:
    """The BIT PATTERNS behind a divergence, so a row can be attributed."""
    out: Dict[str, Any] = {}
    for name in sorted(difference):
        left, right = words(reference["volumes"][name]), words(other["volumes"][name])
        if left.shape != right.shape:
            out[name] = {"shape_differs": [int(left.size), int(right.size)]}
            continue
        where = np.flatnonzero(left != right)
        transcript = []
        for index in where[:limit].tolist():
            a_word, b_word = int(left[index]), int(right[index])
            transcript.append({
                "index": int(index),
                "reference_word": f"0x{a_word:08x}", "other_word": f"0x{b_word:08x}",
                "reference_class": _classify(a_word), "other_class": _classify(b_word),
                "word_distance": abs(a_word - b_word)})
        out[name] = {"differing_words": int(where.size), "words_in_volume": int(left.size),
                     "transcript": transcript}
    return out


# ---------------------------------------------------------------------------
# The synthetic fixture: a REAL driver on a Dcyl m = 0 grid
# ---------------------------------------------------------------------------

def build_driver(label: str, seed: int, value_class: str = values.UNIFORM,
                 scale: float = 1.0) -> Any:
    """One seeded Dcyl m = 0 ``FdtdDriver`` with an active r/z split-field PML.

    A DRIVER, not a bare ``Fields``: every leg is a claim about a COMPLETE driver step
    -- the consult order, the withdraw loop, the wall clears -- so the walk is
    ``FdtdDriver.step``'s own and never a second model of what a step is.

    THE SEED COVERS ``fu_*`` AND ``f_w_*`` AS WELL AS THE TWELVE FIELD VOLUMES. Zero
    init is a fixed point of the constitutive sub-step and a near-fixed-point of the
    split-field recurrence, so the auxiliaries carry physical-band values from step
    zero and the vacuity floor has something to measure.
    """
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415

    options = dict(CASES)[label]
    shape = tuple(int(n) for n in options["shape"])
    driver = FdtdDriver(cell_size=(float(shape[0]), 0.0, float(shape[2])),
                        resolution=1.0, courant=float(options["courant"]),
                        force_complex_fields=False, boundaries={"z": options["z_kind"]},
                        cylindrical=True, m=0)
    pml = options.get("pml", (max(2, shape[0] // 4), max(2, shape[2] // 4)))
    driver.setup_pml({"x": (0, int(pml[0])), "z": int(pml[1])})
    assert tuple(driver.grid.shape) == shape, (tuple(driver.grid.shape), shape)
    driver.fields.set_epsilon_volumes(
        {name: np.full(shape, np.float32(value), np.float32)
         for name, value in EPSILON.items()},
        {name: np.full(shape, np.float32(1.0 / value), np.float32)
         for name, value in EPSILON.items()})
    if value_class == values.PM_ZERO_LATTICE:
        lattice = values.pm_zero_lattice(shape)
        for array in stored_volumes(driver.fields).values():
            array[...] = lattice.astype(array.dtype)
    elif value_class == values.UNIFORM:
        rng = np.random.default_rng(seed)
        for array in stored_volumes(driver.fields).values():
            array[...] = (rng.uniform(-1.0, 1.0, size=array.shape)
                          * scale).astype(array.dtype)
    else:
        raise ValueError(f"unknown value class {value_class!r}")
    driver.invalidate_fast_path()
    pin_array_path(driver)
    return driver


def pin_array_path(driver: Any) -> None:
    """Freeze this driver's fast path as the pure array path (or as a shim)."""
    driver._fast_path = None  # noqa: SLF001 - the engine's own freeze slot
    driver._fast_path_stale = False  # noqa: SLF001


def install(driver: Any, shim: Optional["Shim"]) -> None:
    driver._fast_path = shim  # noqa: SLF001
    driver._fast_path_stale = False  # noqa: SLF001


# ---------------------------------------------------------------------------
# The dispatch shim -- the engine's own seam, answered by this gate's plans
# ---------------------------------------------------------------------------

class ABSORBED:
    """The marker for a slot a fused launch already performed: answer True, run nothing."""


class SyncedPlan:
    """A device plan bracketed by the residency's host->device / device->host copies."""

    __slots__ = ("inner", "residency")

    def __init__(self, inner: Any, residency: Residency) -> None:
        self.inner = inner
        self.residency = residency

    def run(self, *args: Any, **kwargs: Any) -> None:
        self.residency.sync_in()
        self.inner.run(*args, **kwargs)
        self.residency.sync_out()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def synced(plan: Any, residency: Residency) -> Any:
    """Wrap the DEVICE half of whatever sits in a slot; leave host-only work alone."""
    if plan is None or plan is ABSORBED:
        return plan
    inner = getattr(plan, "inner", None)
    if inner is not None and getattr(plan, "absorbed_by", None) is inner:
        plan.inner = SyncedPlan(inner, residency)
        return plan
    if getattr(plan, "absorbed_by", None) is not None:
        return plan
    if not getattr(plan, "performs_device_work", True):
        return plan
    return SyncedPlan(plan, residency)


def declaring(plan: Any) -> Any:
    """The plan whose counters describe the device work in a slot."""
    if isinstance(plan, SyncedPlan):
        plan = plan.inner
    inner = getattr(plan, "absorbed_by", None)
    if inner is not None:
        return declaring(inner)
    return plan


class Shim:
    """A ``FastPathPlan``-shaped object that runs this gate's plans inside the driver.

    THE SEAM IS THE ENGINE'S OWN: ``driver.step`` reads ``self._fast_path`` once per
    step and consults ``fast.dispatch(slot, fields)``. The by-name sync channel is
    refused for any plan whose span leaves the magnetic half-step's slots. EVERY
    ANSWER IS COUNTED: a shim that quietly declined everything would still produce a
    green byte comparison, because the oracle IS the array path.
    """

    __slots__ = ("fields", "plans", "dispatched", "declined", "absorbed",
                 "sync_refusals")

    def __init__(self, fields: Any, plans: Mapping[str, Any]) -> None:
        self.fields = fields
        self.plans = {name: plan for name, plan in plans.items() if plan is not None}
        self.dispatched: Dict[str, int] = {}
        self.declined: Dict[str, int] = {}
        self.absorbed: Dict[str, int] = {}
        self.sync_refusals = 0

    def span_of(self, slot: str) -> Tuple[str, ...]:
        plan = self.plans.get(slot)
        if plan is None or plan is ABSORBED:
            return (slot,)
        return tuple(getattr(declaring(plan), "replaces_sub_steps", None) or (slot,))

    def dispatch(self, slot: str, fields: Any) -> bool:
        owner = SYNC_PASS_OWNERS.get(slot)
        if owner is not None:
            if owner not in self.plans:
                self.declined[slot] = self.declined.get(slot, 0) + 1
                return False
            if any(name not in SYNC_PATH_SLOTS for name in self.span_of(owner)):
                self.sync_refusals += 1
                return False
            slot = owner
        plan = self.plans.get(slot)
        if plan is None or fields is not self.fields:
            self.declined[slot] = self.declined.get(slot, 0) + 1
            return False
        if plan is ABSORBED:
            self.absorbed[slot] = self.absorbed.get(slot, 0) + 1
            return True
        plan.run()
        self.dispatched[slot] = self.dispatched.get(slot, 0) + 1
        return True

    @property
    def launches(self) -> int:
        """Launches the PLANS counted, a different witness from the function wrappers."""
        seen: List[int] = []
        total = 0
        for plan in self.plans.values():
            if plan is ABSORBED:
                continue
            owner = declaring(plan)
            if id(owner) in seen:
                continue
            seen.append(id(owner))
            total += int(getattr(owner, "launches", 0))
        return total


class CountingFunction:
    """An independent launch counter: wraps a compiled function and counts calls."""

    __slots__ = ("function", "calls")

    def __init__(self, function: Any) -> None:
        self.function = function
        self.calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return self.function(*args, **kwargs)


#: Every function-table attribute a plan in this composition can hold. The two-launch
#: plans carry two tables; the single-launch ones carry ``_functions``.
FUNCTION_TABLES: Tuple[str, ...] = ("_functions", "_prefix_functions", "_curl_functions",
                                    "_fused_functions", "_constitutive_functions")


def count_functions(plan: Any) -> List[CountingFunction]:
    """Replace every compiled function the plan holds by a counting wrapper."""
    owner = declaring(plan)
    out: List[CountingFunction] = []
    for attribute in FUNCTION_TABLES:
        table = getattr(owner, attribute, None)
        if not isinstance(table, dict):
            continue
        wrapped = {mode: CountingFunction(function) for mode, function in table.items()}
        setattr(owner, attribute, wrapped)
        out.extend(wrapped.values())
    return out


def settle_rotation(plan: Any, fields: Any, originals: Mapping[str, Any]) -> int:
    """After a step that rotated, put the ENGINE's references back on the seed arrays.

    This gate drives six arrangements on ONE driver, and the other five bound their
    device mirrors to the seed arrays by identity; a rotation left standing at the end
    of a step would leave them writing an array the engine no longer names. The
    advanced values are copied into the seed array, the reference is put back, and the
    twin resumes its role as scratch. The rotation INSIDE the step is untouched.
    """
    settled = 0
    for name in getattr(plan, "rotated_names", ()):
        current = getattr(fields, name)
        original = originals[name]
        if current is not original:
            original[...] = current
            setattr(fields, name, original)
            plan.rotated[name] = current
            settled += 1
    return settled


def rotating_owners(shim: Optional[Shim]) -> Tuple[Any, ...]:
    if shim is None:
        return ()
    owners: List[Any] = []
    seen: List[int] = []
    for plan in shim.plans.values():
        if plan is ABSORBED:
            continue
        owner = declaring(plan)
        if id(owner) in seen or not getattr(owner, "rotated_names", ()):
            continue
        seen.append(id(owner))
        owners.append(owner)
    return tuple(owners)


def orphaned_mirrors(arrangements: Mapping[str, "Arrangement"], fields: Any) -> List[str]:
    """Every MIRRORED volume the engine has stopped naming, across all arrangements."""
    seen: List[str] = []
    for arrangement in arrangements.values():
        residency = arrangement.residency
        if residency is None:
            continue
        for name in residency.names:
            bound = residency.host(name)
            current = getattr(fields, name, None)
            if bound is None or current is None:
                continue
            if current is not bound and name not in seen:
                seen.append(name)
    return sorted(seen)


class Arrangement:
    """One engine: a shim (or the array path), its residency, and its bookkeeping."""

    __slots__ = ("name", "shim", "residency", "counters", "rotating", "originals",
                 "selected", "reasons", "subject")

    def __init__(self, name: str, shim: Optional[Shim], residency: Optional[Residency],
                 selected: Optional[Mapping[str, str]] = None,
                 reasons: Optional[Mapping[str, Any]] = None,
                 subject: Any = None) -> None:
        self.name = name
        self.shim = shim
        self.residency = residency
        self.counters: List[CountingFunction] = []
        if shim is not None:
            wrapped: List[int] = []
            for plan in shim.plans.values():
                if plan is ABSORBED:
                    continue
                owner = declaring(plan)
                if id(owner) in wrapped:
                    continue
                wrapped.append(id(owner))
                self.counters.extend(count_functions(plan))
        self.rotating = rotating_owners(shim)
        self.originals = ({} if shim is None
                          else {name: getattr(shim.fields, name)
                                for owner in self.rotating for name in owner.rotated_names})
        self.selected = dict(selected or {})
        self.reasons = dict(reasons or {})
        #: The fused plan under test, where this arrangement carries it.
        self.subject = subject

    def settle(self, fields: Any) -> int:
        return sum(settle_rotation(plan, fields, self.originals) for plan in self.rotating)

    def launches(self) -> Dict[str, int]:
        return {"plans": 0 if self.shim is None else self.shim.launches,
                "functions": sum(counter.calls for counter in self.counters)}


def arrangement_singles(driver: Any) -> Arrangement:
    """Reference 2: the two certified singles dispatched at the seam's two slots."""
    residency = Residency()
    fields, pml = driver.fields, driver.pml
    constitutive = cyl.plan_cylindrical_real_constitutive(fields, pml, "H", residency)
    curl = cyl.plan_cylindrical_real_curl(fields, pml, "step_D", residency)
    if constitutive is None or curl is None:
        raise RuntimeError("a certified single was refused on a fixture this gate "
                           "expects it to admit")
    shim = Shim(fields, {"update_H": synced(constitutive, residency),
                         "step_D": synced(curl, residency)})
    return Arrangement("singles", shim, residency,
                       selected={"update_H": "cylindrical m=0", "step_D": "cylindrical m=0"})


def _composed(driver: Any, fuse: bool,
              residency: Optional[Residency] = None) -> Tuple[Any, Residency]:
    """``plan_step`` on this driver, on ``residency`` or on a fresh one.

    ONE ARRANGEMENT IS ONE RESIDENCY: every plan touching a volume must bind the SAME
    device tensor, or two registries launch against two buffers for one host array.
    """
    residency = Residency() if residency is None else residency
    plan = metal_launch.plan_step(driver.fields, driver.pml, residency=residency,
                                  sources=tuple(getattr(driver, "_sources", ())),
                                  fuse=fuse)
    return plan, residency


def arrangement_composition(driver: Any, fuse: bool) -> Arrangement:
    """References 3 (``fuse=True``, what the composer installs today) and 4 (unfused)."""
    plan, residency = _composed(driver, fuse)
    plans = {slot: synced(plan.plans[slot], residency)
             for slot in STEP_ORDER if slot in plan.plans}
    return Arrangement("composition_today" if fuse else "unfused",
                       Shim(driver.fields, plans), residency,
                       selected=plan.selected,
                       reasons={key: list(value) for key, value in plan.reasons.items()
                                if key.startswith("fused_pair")})


def build_subject(driver: Any, residency: Residency,
                  constitutive_function: Optional[Mapping[str, Any]] = None,
                  curl_function: Optional[Mapping[str, Any]] = None,
                  scan_vectors: Optional[Tuple[Any, Any]] = None) -> Any:
    plan = family.plan_metal_cylindrical_real_fused_hd_pair(
        driver.fields, driver.pml, sources=tuple(getattr(driver, "_sources", ())),
        residency=residency, constitutive_function=constitutive_function,
        curl_function=curl_function, scan_vectors=scan_vectors)
    if plan is None:
        reasons = family.metal_cylindrical_real_fused_hd_pair_coverage(
            driver.fields, driver.pml, tuple(getattr(driver, "_sources", ())),
            residency).reasons
        raise RuntimeError("the cylindrical fused H/D pair was refused: "
                           + "; ".join(reasons))
    return plan


def arrangement_weld(driver: Any,
                     constitutive_function: Optional[Mapping[str, Any]] = None,
                     curl_function: Optional[Mapping[str, Any]] = None,
                     scan_vectors: Optional[Tuple[Any, Any]] = None,
                     launcher: Optional[Callable[[Any, Residency, Any], Any]] = None,
                     rest_unfused: bool = True, name: str = "weld") -> Arrangement:
    """The SUBJECT: the pair force-installed at ``update_H``, ``step_D`` absorbed.

    FORCE-INSTALLED, because the composer holds no row for this product and this gate
    must measure it anyway. With ``rest_unfused`` every other slot carries the unfused
    single ``plan_step(fuse=False)`` selects; without it the rest stays on the array
    path, which isolates the seam.

    ``launcher`` is the HOST mutation seam: it receives the built plan and returns the
    object the slot runs. ``constitutive_function`` / ``curl_function`` /
    ``scan_vectors`` are handed to the family's own builder.
    """
    residency = Residency()
    plan = build_subject(driver, residency, constitutive_function, curl_function,
                         scan_vectors)
    occupant: Any = plan if launcher is None else launcher(plan, residency, driver.pml)
    occupant = SyncedPlan(occupant, residency)
    plans: Dict[str, Any] = {}
    if rest_unfused:
        base, _base_residency = _composed(driver, fuse=False, residency=residency)
        for slot in STEP_ORDER:
            if slot in base.plans and slot not in family.REPLACES:
                plans[slot] = synced(base.plans[slot], residency)
    plans["update_H"] = occupant
    plans["step_D"] = ABSORBED
    return Arrangement(name, Shim(driver.fields, plans), residency,
                       selected={"update_H": f"{family.FAMILY} (forced)",
                                 "step_D": f"{family.FAMILY} (forced)"},
                       subject=plan)


def all_arrangements(driver: Any, **weld_kwargs: Any) -> Dict[str, Arrangement]:
    return {
        "array": Arrangement("array", None, None),
        "singles": arrangement_singles(driver),
        "composition_today": arrangement_composition(driver, fuse=True),
        "unfused": arrangement_composition(driver, fuse=False),
        "weld": arrangement_weld(driver, **weld_kwargs),
        "weld_seam_only": arrangement_weld(driver, rest_unfused=False,
                                           name="weld_seam_only", **weld_kwargs),
    }


# ---------------------------------------------------------------------------
# The walk: six engines on ONE driver, in lockstep from one seed
# ---------------------------------------------------------------------------

def prefix_against_the_oracle(arrangement: Arrangement, fields: Any) -> Optional[int]:
    """The subject's device prefix against ``stepping.cylindrical_rderiv_prefix``.

    Taken AFTER the complete step, when ``fields.Hy`` is exactly what launch 1 stored
    (nothing after ``update_H`` writes H in a step) and the residency's ``sync_out``
    has refreshed the plan's prefix host array.
    """
    plan = arrangement.subject
    if plan is None:
        return None
    oracle = stepping.cylindrical_rderiv_prefix(np, np.asarray(fields.Hy), family.IR0)
    return differing(oracle, plan.prefix_host)


def drive(driver: Any, arrangements: Mapping[str, Arrangement], steps: int,
          reference: str = "array", secondary: Optional[str] = "singles",
          progress: Optional[Callable[[str], None]] = None,
          stop_on_divergence: bool = True) -> Dict[str, Any]:
    """Step every arrangement in lockstep from one seed and compare per complete step.

    ONE driver, restored to each arrangement's own state before its step, captured
    after it, with the subject's rotation settled so every other arrangement's mirrors
    stay bound to the arrays the engine names.

    THE PRECONDITION IS TAKEN PER STEP, BEFORE THE COMPARISON. Byte identity on this
    backend is claimed subject to the oracle's state staying out of the float32
    denormal band -- MPS flushes natively and NumPy does not -- so a banded step is
    counted, named and NOT compared, and nothing past it is measurable. Two halves:
    the stored-state census, and the reference's own IEEE underflow flag over the
    step's ufuncs (which a tiny EXACT result does not raise; that gap is recorded).
    """
    seed = capture(driver)
    states = {name: seed for name in arrangements}
    per_step: Dict[str, List[Dict[str, Any]]] = {name: [] for name in arrangements
                                                  if name != reference}
    autopsies: Dict[str, Dict[str, Any]] = {}
    census_per_step: List[Dict[str, Any]] = []
    prefix_rows: Dict[str, List[Optional[int]]] = {name: [] for name in arrangements}
    orphaned: Dict[str, List[str]] = {name: [] for name in arrangements}
    lazily: List[str] = []
    steps_done = 0
    steps_stepped = 0
    error: Optional[str] = None
    started = time.time()
    try:
        for step in range(1, steps + 1):
            for name, arrangement in arrangements.items():
                lazily.extend(restore(driver, states[name]))
                install(driver, arrangement.shim)
                if name == reference:
                    underflowed = False

                    def _flag(kind: str, _bits: int) -> None:  # noqa: ANN001
                        nonlocal underflowed
                        if kind == "underflow":
                            underflowed = True

                    with np.errstate(under="call", over="ignore", divide="ignore",
                                     invalid="ignore", call=_flag):
                        driver.step()
                    reference_underflowed = underflowed
                else:
                    driver.step()
                arrangement.settle(driver.fields)
                prefix_rows[name].append(prefix_against_the_oracle(arrangement,
                                                                   driver.fields))
                orphans = orphaned_mirrors(arrangements, driver.fields)
                if orphans:
                    orphaned[name] = sorted(set(orphaned[name]) | set(orphans))
                states[name] = capture(driver)
            steps_stepped = step
            census = int(sum(subnormal.census(value)
                             for value in states[reference]["volumes"].values()))
            census_per_step.append({"step": step, "reference_subnormals": census,
                                    "reference_underflow_flag": bool(reference_underflowed)})
            if census or reference_underflowed:
                if progress is not None:
                    progress(f"step {step}/{steps} NOT COMPARED: the oracle entered the "
                             f"denormal band (stored {census} words, underflow flag "
                             f"{reference_underflowed}); the walk stops "
                             f"({time.time() - started:.1f} s)")
                break
            for name in per_step:
                difference = compare_snapshots(states[reference], states[name])
                row: Dict[str, Any] = {"step": step,
                                       "differing_words": int(sum(difference.values())),
                                       "differing_volumes": difference,
                                       "prefix_vs_oracle": prefix_rows[name][-1]}
                if secondary is not None and secondary in states and name != secondary:
                    against = compare_snapshots(states[secondary], states[name])
                    row[f"differing_words_vs_{secondary}"] = int(sum(against.values()))
                if difference and name not in autopsies:
                    autopsies[name] = {"step": step, "against": reference,
                                       "volumes": autopsy(states[reference], states[name],
                                                          difference)}
                per_step[name].append(row)
            steps_done = step
            if progress is not None:
                progress(f"step {step}/{steps} "
                         + " ".join(f"{name}={rows[-1]['differing_words']}"
                                    for name, rows in per_step.items())
                         + f" prefix_vs_oracle={prefix_rows['weld'][-1]} subnormals=0 "
                         f"({time.time() - started:.1f} s)")
            if stop_on_divergence and any(rows[-1]["differing_words"]
                                          for rows in per_step.values()):
                break
    except Exception as exc:  # noqa: BLE001 - a row that cannot be stepped is named
        error = f"{type(exc).__name__}: {exc}"[:600]
    finally:
        pin_array_path(driver)
    reference_state = states[reference]
    moved = {name: differing(seed["volumes"][name], reference_state["volumes"][name])
             for name in seed["volumes"]}
    banded = next((row["step"] for row in census_per_step
                   if row["reference_subnormals"] or row["reference_underflow_flag"]), None)
    legs: Dict[str, Any] = {}
    for name, rows in per_step.items():
        first = next((row["step"] for row in rows if row["differing_words"]), None)
        subject = arrangements[name].subject
        legs[name] = {
            "identical": bool(rows and all(not row["differing_words"] for row in rows)),
            "identical_vs_secondary": (
                None if secondary is None or name == secondary
                else bool(rows and all(not row.get(f"differing_words_vs_{secondary}")
                                       for row in rows))),
            "steps_compared": len(rows),
            "first_divergence": first,
            "differing_words_final": rows[-1]["differing_words"] if rows else None,
            "differing_volumes_final": rows[-1]["differing_volumes"] if rows else {},
            "prefix_vs_oracle_per_step": [row["prefix_vs_oracle"] for row in rows],
            "prefix_identical_to_oracle_every_step": (
                None if subject is None
                else bool(rows and all(row["prefix_vs_oracle"] == 0 for row in rows))),
            "launches": arrangements[name].launches(),
            "constitutive_launches": (None if subject is None
                                      else subject.constitutive_launches),
            "curl_launches": None if subject is None else subject.curl_launches,
            "runs": None if subject is None else subject.runs,
            "dispatched": dict(arrangements[name].shim.dispatched)
            if arrangements[name].shim else {},
            "absorbed": dict(arrangements[name].shim.absorbed)
            if arrangements[name].shim else {},
            "sync_refusals": (arrangements[name].shim.sync_refusals
                              if arrangements[name].shim else 0),
            "selected": arrangements[name].selected,
            "first_divergence_autopsy": autopsies.get(name),
            "per_step": rows,
        }
    return {
        "steps_requested": steps,
        "steps_compared": steps_done,
        "steps_stepped": steps_stepped,
        "step_error": error,
        "volumes_compared": sorted(seed["volumes"]),
        "words_per_step": int(sum(words(value).size for value in seed["volumes"].values())),
        "words_compared": int(sum(words(value).size for value in seed["volumes"].values())
                              * steps_done * len(per_step)),
        "lazily_allocated_volumes": sorted(set(lazily)),
        "moved_from_seed": {name: n for name, n in sorted(moved.items()) if n},
        "arrays_that_never_moved": sorted(name for name, n in moved.items() if not n),
        "reference_subnormals_per_step": census_per_step,
        "first_banded_step": banded,
        "precondition_clean": banded is None,
        "precondition_instruments": ["stored-state census", "IEEE underflow flag on the "
                                     "reference's ufuncs (does not see an exact tiny "
                                     "intermediate)"],
        "mirrored_volumes_left_orphaned": {name: v for name, v in orphaned.items() if v},
        "every_arrangement_gave_the_engine_its_volumes_back": not any(orphaned.values()),
        "seconds": round(time.time() - started, 2),
        "arrangements": legs,
    }


def run_product(driver: Any, steps: int, progress: Optional[Callable[[str], None]] = None,
                movement_floor: str = "all", stop_on_divergence: bool = True,
                require_full_budget: bool = True, clean_floor: int = 0,
                value_class: str = values.UNIFORM, **weld_kwargs: Any) -> Dict[str, Any]:
    """The four-reference identity on one driver, per complete step."""
    arrangements = all_arrangements(driver, **weld_kwargs)
    result = drive(driver, arrangements, steps, progress=progress,
                   stop_on_divergence=stop_on_divergence)
    legs = result["arrangements"]
    identical = all(legs[name]["identical"] for name in legs)
    compared = int(result["steps_compared"])
    stepped = int(result["steps_stepped"])
    budget_ok = (compared == steps if require_full_budget
                 else compared >= max(clean_floor, 1))
    launched = all(legs[name]["launches"]["plans"] > 0
                   and legs[name]["launches"]["plans"] == legs[name]["launches"]["functions"]
                   for name in legs)
    weld_ok = all(
        legs[name]["dispatched"].get("update_H", 0) == stepped
        and legs[name]["absorbed"].get("step_D", 0) == stepped
        and legs[name]["constitutive_launches"] == stepped
        and legs[name]["curl_launches"] == stepped
        and legs[name]["runs"] == stepped
        and legs[name]["launches"]["plans"] >= 2 * stepped
        for name in ("weld", "weld_seam_only"))
    prefix_ok = all(legs[name]["prefix_identical_to_oracle_every_step"]
                    for name in ("weld", "weld_seam_only"))
    seam_outputs = tuple(name for name in
                         [f"{stem}{axis}" for stem in ("D", "fu_D", "H", "f_w_H")
                          for axis in "xyz"]
                         if name in result["volumes_compared"])
    seam_moved = {name: int(result["moved_from_seed"].get(name, 0)) for name in seam_outputs}
    if value_class == values.PM_ZERO_LATTICE:
        signs = values.zero_sign_census(
            capture(driver)["volumes"].values())
        floor = bool(signs["positive_zero_words"] and signs["negative_zero_words"])
        result["reference_zero_sign_census"] = signs
    elif movement_floor == "all":
        floor = not result["arrays_that_never_moved"]
    else:
        floor = bool(seam_outputs) and sum(seam_moved.values()) > 0
    settled = bool(result["every_arrangement_gave_the_engine_its_volumes_back"])
    result.update({
        "passed": bool(identical and launched and weld_ok and prefix_ok and floor
                       and budget_ok and settled and result["step_error"] is None
                       and (result["precondition_clean"] or not require_full_budget)),
        "bit_identical": identical,
        "budget_met": budget_ok,
        "require_full_budget": require_full_budget,
        "clean_floor": clean_floor,
        "every_arrangement_launched_and_the_two_counters_agree": launched,
        "weld_launched_twice_per_step_both_counters_and_absorbed_step_D": weld_ok,
        "prefix_identical_to_the_oracle_every_step": prefix_ok,
        "movement_floor": movement_floor,
        "movement_floor_met": floor,
        "seam_output_words_moved": seam_moved,
        "weld_agrees_with_the_certified_singles": legs["weld"]["identical_vs_secondary"],
        "the_seam_alone_agrees_with_the_array_path": legs["weld_seam_only"]["identical"],
        "references_that_disagree_with_the_array_path": sorted(
            name for name in ("singles", "composition_today", "unfused")
            if not legs[name]["identical"]),
        "composition_today_selected": legs["composition_today"]["selected"],
        "composition_today_refuses_this_product":
            arrangements["composition_today"].reasons.get(f"fused_pair_{family.FAMILY}"),
        "value_class": value_class,
    })
    return result


# ---------------------------------------------------------------------------
# Host legs
# ---------------------------------------------------------------------------

def _compiles(source: str) -> Tuple[bool, str]:
    try:
        compile_source(source)
        return True, ""
    except Exception as exc:  # noqa: BLE001 - the failure IS the measurement
        message = str(exc)
        return False, (message.splitlines()[0] if message else "")


def _sweep_source(pointers: int) -> str:
    lines = [f"    device float* p{n} [[buffer({n})]]," for n in range(pointers)]
    touch = " + ".join(f"p{n}[0]" for n in range(pointers))
    return "\n".join((
        "#include <metal_stdlib>", "using namespace metal;", "",
        "struct Params { uint n_elem; float dtdx; };", "",
        "kernel void ceiling_sweep(", *lines,
        f"    constant Params& prm [[buffer({pointers})]],",
        "    uint idx [[thread_position_in_grid]])", "{",
        "    if (idx >= prm.n_elem) { return; }",
        f"    float touch = {touch};",
        *(f"    p{n}[idx] = p{n}[idx] + touch * prm.dtdx;" for n in range(pointers)),
        "}", ""))


def leg_binding_ceiling() -> Dict[str, Any]:
    """Launch 1's 25 compile, 31 compiles, 32 is refused; the ceiling bisected; launch 2 compiles."""
    row: Dict[str, Any] = {
        "declared_max_buffer_bindings": MAX_BUFFER_BINDINGS,
        "constitutive_bindings_declared": family.CONSTITUTIVE_BINDINGS,
        "constitutive_bindings_counted_off_the_text": family.shipped_signature_bindings(),
        "constitutive_pointers": family.CONSTITUTIVE_POINTERS,
        "headroom_declared": family.HEADROOM,
        "curl_bindings_declared": family.CURL_BINDINGS,
    }
    sweep: Dict[str, bool] = {}
    for pointers in range(MAX_BUFFER_BINDINGS - 4, MAX_BUFFER_BINDINGS + 2):
        ok, _error = _compiles(_sweep_source(pointers))
        sweep[f"{pointers}_pointers_{pointers + 1}_bindings"] = ok
    row["ceiling_sweep"] = sweep
    largest = max((int(k.split("_")[0]) for k, ok in sweep.items() if ok), default=-1)
    smallest = min((int(k.split("_")[0]) for k, ok in sweep.items() if not ok), default=-1)
    row["largest_pointer_count_that_compiles_with_one_struct"] = largest
    row["smallest_pointer_count_refused_with_one_struct"] = smallest
    ceiling_ok = largest + 1 == MAX_BUFFER_BINDINGS and smallest == largest + 1
    row["ceiling_measured_equals_declared"] = ceiling_ok

    shipped = {}
    for mode in shaders.CONTRACT_MODES:
        ok, error = _compiles(family.cylindrical_real_fused_hd_pair_constitutive_source(mode))
        shipped[mode] = {"compiled": ok, "error": error}
    row["constitutive_shipped_compiled"] = shipped
    fitting_ok, fitting_error = _compiles(family.largest_fitting_source())
    refused_ok, refused_error = _compiles(family.refuted_over_the_ceiling_source())
    row["plus_headroom_compiled"] = fitting_ok
    row["plus_headroom_error"] = fitting_error
    row["plus_headroom_plus_one_compiled"] = refused_ok
    row["plus_headroom_plus_one_error"] = refused_error
    row["refused_for_the_right_reason"] = (not refused_ok and "out of bounds" in refused_error
                                           and "buffer" in refused_error)
    row["measured_headroom"] = largest - family.CONSTITUTIVE_POINTERS
    curl = {}
    for codes in ((1, 0, 1), (1, 0, 0)):
        source = family.cylindrical_real_fused_hd_pair_curl_source(codes)
        ok, error = _compiles(source)
        signature = source.split("kernel void cyl_pml_curl_step(", 1)[1]
        signature = signature.split("uint idx [[thread_position_in_grid]])", 1)[0]
        curl[str(codes)] = {"compiled": ok, "error": error,
                            "bindings": signature.count("[[buffer(")}
    row["curl_compiled"] = curl
    row["passed"] = bool(
        ceiling_ok and all(r["compiled"] for r in shipped.values()) and fitting_ok
        and row["refused_for_the_right_reason"]
        and row["constitutive_bindings_counted_off_the_text"] == family.CONSTITUTIVE_BINDINGS
        and row["measured_headroom"] == family.HEADROOM
        and all(r["compiled"] and r["bindings"] == family.CURL_BINDINGS
                for r in curl.values()))
    return row


def leg_transcription() -> Dict[str, Any]:
    """Launch 1 is lifted certified text with exactly the declared edits; launch 2 IS the curl."""
    source = family.cylindrical_real_fused_hd_pair_constitutive_source()
    h_cell = cartesian.h_cell_function()
    h_cell_verbatim = h_cell in source
    certified_scan = cyl.cylindrical_prefix_source("step_D")
    scan_tail = certified_scan.split("    int base = j * nzi + k;\n", 1)[1][: -len("}\n")]
    lifted = family.certified_scan_tail()
    certified_lines = scan_tail.splitlines()
    lifted_lines = lifted.splitlines()
    changed = [(a, b) for a, b in zip(certified_lines, lifted_lines) if a != b]
    expected = [(edit["line"], edit["became"]) for edit in family.SCAN_LIFT_EDITS]
    lengths_match = len(certified_lines) == len(lifted_lines)
    lifted_in_source = lifted in source
    reads_of_written = [line for line in source.splitlines()
                        if re.search(r"\b(ho|wo)[012]\[", line)
                        and not re.match(r"^\s*(ho|wo)[012]\[ii\] = .*;\s*$",
                                         line.split(";")[0] + ";")]
    # every ho*/wo* occurrence must be a store statement (possibly several per line)
    store_lines = [line for line in source.splitlines() if re.search(r"\b(ho|wo)[012]\[", line)]
    only_stores = all(
        all(re.match(r"^(ho|wo)[012]\[ii\] = own\.[a-z0-9]+$", piece.strip())
            for piece in line.split(";") if piece.strip())
        for line in store_lines)
    no_src = "src[" not in source
    curl_identity = {}
    for codes in ((1, 0, 1), (1, 0, 0)):
        curl_identity[str(codes)] = (family.cylindrical_real_fused_hd_pair_curl_source(codes)
                                     == cyl.cylindrical_curl_source("step_D", codes))
    return {
        "passed": bool(h_cell_verbatim and lengths_match and changed == expected
                       and lifted_in_source and only_stores and not reads_of_written
                       and no_src and all(curl_identity.values())),
        "h_cell_function_verbatim": h_cell_verbatim,
        "scan_lines_certified": len(certified_lines),
        "scan_lines_lifted": len(lifted_lines),
        "scan_lines_changed": [list(pair) for pair in changed],
        "scan_lines_expected": [list(pair) for pair in expected],
        "lifted_scan_in_source": lifted_in_source,
        "scratch_pointer_lines": store_lines,
        "scratch_pointers_appear_only_as_stores": only_stores,
        "no_src_pointer_read_survives": no_src,
        "launch_2_is_the_certified_curl_text": curl_identity,
    }


def _rebuild_increment(f_p: np.ndarray, ir0: float) -> np.ndarray:
    """The increment EXACTLY as stepping.py:1313-1331 spells it (float64 ladder,
    rounded once; field left of the weight; difference over the row divisor)."""
    counts = np.arange(f_p.shape[0], dtype=np.float64) + ir0
    weights = counts.reshape(-1, 1, 1).astype(np.float32)
    divisor = (counts[1:] - 0.5).reshape(-1, 1, 1).astype(np.float32)
    weighted = f_p * weights
    increment = np.zeros_like(f_p)
    increment[1:] = (weighted[1:] - weighted[:-1]) / divisor
    return increment


def _serial_scan(inc: np.ndarray) -> np.ndarray:
    out = np.empty_like(inc)
    acc = np.zeros(inc.shape[1:], dtype=inc.dtype)
    for i in range(inc.shape[0]):
        acc = (acc + inc[i]).astype(inc.dtype)
        out[i] = acc
    return out


def _blocked_scan(inc: np.ndarray, tile: int = 4) -> np.ndarray:
    """NULL CONTROL: tile-local scans then a carry add -- a legitimate other order.

    FOUR-ROW TILES, so the control has more than one tile on EVERY extent this leg
    walks: at 32-row tiles the blocked scan IS ``numpy.cumsum`` on a column of 32 rows
    or fewer and cannot bite there (measured on this gate's first run: 0 differing
    words on nr = 13, 16, 20 and 32 while 120 and 150 differed by more than half).
    """
    out = np.empty_like(inc)
    carry = np.zeros(inc.shape[1:], dtype=inc.dtype)
    for start in range(0, inc.shape[0], tile):
        end = min(start + tile, inc.shape[0])
        local = np.cumsum(inc[start:end], axis=0).astype(inc.dtype)
        out[start:end] = (local + carry).astype(inc.dtype)
        carry = out[end - 1]
    return out


def leg_scan_order_host() -> Dict[str, Any]:
    """``numpy.cumsum`` IS strictly serial over the shipped increments, at ir0 = 0.5.

    Re-measured on THIS host over the fixture and corpus radial extents rather than
    cited from the adjudication: it is the premise the device scan reproduces by
    construction, and the blocked scan is the control that must differ.
    """
    rows = []
    total = serial = blocked = 0
    for nr, nz in ((13, 32), (16, 20), (20, 20), (32, 13), (120, 120), (150, 175)):
        for index, kind in enumerate(("uniform", "wide_dynamic", "cancelling")):
            rng = np.random.default_rng(3000 + 7 * nr + index)
            if kind == "uniform":
                a = rng.standard_normal((nr, 1, nz))
            elif kind == "wide_dynamic":
                a = rng.standard_normal((nr, 1, nz)) * (
                    10.0 ** rng.integers(-12, 12, (nr, 1, nz)))
            else:
                a = (rng.standard_normal((nr, 1, nz))
                     * np.where(np.arange(nr).reshape(-1, 1, 1) % 2, 1.0, -1.0) * 1e6)
            f_p = a.astype(np.float32)
            inc = _rebuild_increment(f_p, family.IR0)
            shipped = stepping.cylindrical_rderiv_prefix(np, f_p, family.IR0)
            rebuild_ok = differing(shipped, np.cumsum(inc, axis=0)) == 0
            d_serial = differing(shipped, _serial_scan(inc))
            d_blocked = differing(shipped, _blocked_scan(inc))
            n = int(words(shipped).size)
            total += n
            serial += d_serial
            blocked += d_blocked
            rows.append({"nr": nr, "nz": nz, "class": kind, "words": n,
                         "rebuild_matches_shipped": rebuild_ok,
                         "serial_differing": d_serial, "blocked_differing": d_blocked})
            log(f"    scan_order nr={nr} nz={nz} {kind}: serial={d_serial}/{n} "
                f"blocked={d_blocked}/{n} rebuild_ok={rebuild_ok}")
    return {"passed": bool(serial == 0 and all(r["rebuild_matches_shipped"] for r in rows)
                           and all(r["blocked_differing"] > 0 for r in rows)),
            "ir0": family.IR0, "cases": len(rows), "words": total,
            "serial_differing": serial, "blocked_differing": blocked, "rows": rows}


def leg_driver_order() -> Dict[str, Any]:
    """REPLACES is the driver's two adjacent consults, with only the withdraw between."""
    fact = h_to_d_seam.driver_seam_fact()
    seam_row = metal_launch.FUSED_PAIR_SEAMS.get(family.SLOT)
    return {
        "passed": bool(tuple(family.REPLACES) == tuple(h_to_d_seam.HALVES)
                       and family.SLOT == h_to_d_seam.HALVES[0]
                       and family.SEAM == withdraw_hoist.SEAM == h_to_d_seam.SEAM
                       and seam_row == ("step_D", withdraw_hoist.SEAM)
                       and family.CARRIES_DEPOSIT_REPAIR is False
                       and family.HOISTS_THE_WITHDRAW is False),
        "driver_fact": fact,
        "replaces": list(family.REPLACES),
        "seam_row": list(seam_row) if seam_row else None,
    }


def _volume_source(fields: Any, component: str, integrated: bool) -> Any:
    from meep_gpu.sources import GaussianEnvelope, VolumeSource  # noqa: PLC0415

    return VolumeSource(grid=fields.grid, component=component,
                        center=(0.5, 0.0, 0.5), size=(0.0, 0.0, 0.0),
                        envelope=GaussianEnvelope(frequency=1.0, fwidth=0.5,
                                                  is_integrated=integrated))


def leg_refusal() -> Dict[str, Any]:
    driver = build_driver(REFUSAL_CASE, 91000)
    fields, pml = driver.fields, driver.pml
    residency = Residency()
    undeclared = family.metal_cylindrical_real_fused_hd_pair_coverage(fields, pml, None,
                                                                     residency)
    plain = _volume_source(fields, "Ez", integrated=False)
    admitted = family.metal_cylindrical_real_fused_hd_pair_coverage(fields, pml, (plain,),
                                                                   residency)
    integrated = _volume_source(fields, "Ez", integrated=True)
    standing = withdraw_hoist.standing_withdraws((integrated,))
    refused = family.metal_cylindrical_real_fused_hd_pair_coverage(fields, pml,
                                                                  (integrated,), residency)
    refused_named = [r for r in refused.reasons if "standing integrated" in r]
    refused_plan = family.plan_metal_cylindrical_real_fused_hd_pair(
        fields, pml, (integrated,), Residency())
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415
    complex_driver = FdtdDriver(cell_size=(16.0, 0.0, 20.0), resolution=1.0, courant=0.37,
                                force_complex_fields=True, boundaries={"z": "metallic"},
                                cylindrical=True, m=1)
    complex_driver.setup_pml({"x": (0, 4), "z": 4})
    m_verdict = family.metal_cylindrical_real_fused_hd_pair_coverage(
        complex_driver.fields, complex_driver.pml, (), Residency())
    m_named = [r for r in m_verdict.reasons if "grid.m = 1" in r]
    cart_driver = FdtdDriver(cell_size=(2.0, 2.1, 1.2), resolution=10.0, courant=0.35,
                             force_complex_fields=False, dimensions=3)
    cart_driver.setup_pml(2)
    cart_verdict = family.metal_cylindrical_real_fused_hd_pair_coverage(
        cart_driver.fields, cart_driver.pml, (), Residency())
    cart_named = [r for r in cart_verdict.reasons if "cylindrical" in r]
    twins = {
        "magnetic_twin_admits": magnetic_twin.metal_cylindrical_real_fused_magnetic_pair_coverage(
            fields, pml, (plain,), Residency()).covered,
        "electric_twin_admits": electric_twin.metal_cylindrical_real_fused_electric_pair_coverage(
            fields, pml, (plain,), Residency()).covered,
    }
    registered = [spec for spec in _registered_rows()]
    return {
        "passed": bool(not undeclared.covered
                       and any("was not declared" in r for r in undeclared.reasons)
                       and admitted.covered
                       and len(standing) == 1
                       and not refused.covered and refused_named and refused_plan is None
                       and not m_verdict.covered and m_named
                       and not cart_verdict.covered and cart_named
                       and len(registered) == 1 and registered[0].wired is False
                       and registered[0].slot == family.SLOT
                       and tuple(registered[0].replaces) == family.REPLACES
                       # WIRED 2026-09-07: the absorb row is declared and the weld is
                       # minted, so the two declarations this leg pins are the wired ones.
                       and metal_launch.FUSED_PAIR_ARMS.get(family.FAMILY)
                       == ("cylindrical m=0", "cylindrical m=0")
                       and family.INSTALLABLE is False and not family.WELD_OWED),
        "undeclared_refused": not undeclared.covered,
        "plain_electric_source_admitted": admitted.covered,
        "plain_source_reasons": list(admitted.reasons),
        "integrated_source_has_a_standing_withdraw": len(standing) == 1,
        "integrated_source_refused": not refused.covered,
        "integrated_named": refused_named[:1],
        "integrated_plan_is_none": refused_plan is None,
        "m_is_one_refused": not m_verdict.covered, "m_named": m_named[:1],
        "cartesian_refused": not cart_verdict.covered, "cartesian_named": cart_named[:2],
        "released_twins_on_the_same_fixture": twins,
        "registered_rows": [f"{s.family}/{s.slot} wired={s.wired} replaces={s.replaces}"
                            for s in registered],
        "absorb_row_today": metal_launch.FUSED_PAIR_ARMS.get(family.FAMILY),
        "installable": family.INSTALLABLE,
        "weld_owed_declared": bool(family.WELD_OWED),
    }


def _registered_rows() -> List[Any]:
    from meep_gpu.metal_kernels import arms  # noqa: PLC0415
    return [spec for spec in arms.registered() if spec.family == family.FAMILY]


@contextlib.contextmanager
def _absorb_row_in_process(installable: bool):
    """Hold the absorb row present and ``INSTALLABLE`` at ``installable``, then put both back.

    IN-PROCESS AND RESTORED: ``launch.py`` is not edited. This is the only way to
    measure what the composer WOULD do with the row, which is the arbitration verdict
    :data:`family.INSTALLABLE_REASON` carries.
    """
    had_row = family.FAMILY in metal_launch.FUSED_PAIR_ARMS
    previous_row = metal_launch.FUSED_PAIR_ARMS.get(family.FAMILY)
    previous_flag = family.INSTALLABLE
    metal_launch.FUSED_PAIR_ARMS[family.FAMILY] = CELL_ARMS
    family.INSTALLABLE = installable
    try:
        yield
    finally:
        family.INSTALLABLE = previous_flag
        if had_row:
            metal_launch.FUSED_PAIR_ARMS[family.FAMILY] = previous_row
        else:
            metal_launch.FUSED_PAIR_ARMS.pop(family.FAMILY, None)


def _composer_outcome(driver: Any) -> Dict[str, Any]:
    plan, _residency = _composed(driver, fuse=True)
    owners: Dict[str, Any] = {}
    for slot in ("step_B", "update_H", "step_D", "update_E"):
        occupant = plan.plans.get(slot)
        owners[slot] = None if occupant is None else declaring(occupant)
    distinct: List[Any] = []
    for owner in owners.values():
        if owner is not None and not any(owner is seen for seen in distinct):
            distinct.append(owner)
    return {"selected": dict(plan.selected),
            "launches": sum(int(getattr(owner, "launches_per_run", 1)) for owner in distinct),
            "owners": {slot: type(owner).__name__ for slot, owner in owners.items()},
            "refusal": list(plan.reasons.get(f"fused_pair_{family.FAMILY}", ()))}


def leg_arbitration() -> Dict[str, Any]:
    """The composer, asked three ways: today, with the row, with the row and the flag off."""
    driver = build_driver(REFUSAL_CASE, 93000)
    today = _composer_outcome(driver)
    with _absorb_row_in_process(installable=False):
        with_row = _composer_outcome(driver)
    with _absorb_row_in_process(installable=True):
        unflagged = _composer_outcome(driver)
    incumbent = today["selected"].get("step_D")
    return {
        # WIRED 2026-09-07: with the absorb row declared, "today" IS the with-row state,
        # so the composer's first refusal is the INSTALLABLE = False one on both asks.
        "passed": bool(today["refusal"]
                       and today["refusal"][0].startswith(
                           f"{family.FAMILY} declares INSTALLABLE = False: ")
                       and with_row["refusal"]
                       and with_row["refusal"][0].startswith(
                           f"{family.FAMILY} declares INSTALLABLE = False: ")
                       and unflagged["refusal"]
                       # EITHER SEAM'S INCUMBENT, and 2026-09-17 is when that stopped
                       # being a pedantic allowance. This clause read
                       # `step_D was selected by the {incumbent!r} arm` and nothing
                       # else, which was exact while the B->H seam of a Dcyl m = 0 run
                       # carried SEPARATE `cylindrical m=0` arms: the only released
                       # PAIR this product collided with was the D->E one, so step_D
                       # was the only slot that could refuse it.
                       #
                       # The all-paths batch gave `cylindrical_real_fused_magnetic_pair`
                       # a launch.FUSED_PAIR_ARMS row, so update_H now carries a fused
                       # pair too, and the composer's FIRST refusal moved to it:
                       #     "update_H was selected by the 'cylindrical m=0 fused
                       #      magnetic B/H pair' arm; this fused product implements the
                       #      'cylindrical m=0' one and may not substitute it"
                       # The property this leg exists to check is unchanged and in fact
                       # stronger -- with INSTALLABLE held out of the way the product is
                       # STILL refused BY NAME because a released neighbour holds a slot
                       # it needs -- so the clause asks for that, on whichever seam
                       # answers first, rather than for one spelling of it.
                       and any(
                           unflagged["refusal"][0].startswith(
                               f"{slot} was selected by the "
                               f"{unflagged['selected'].get(slot)!r} arm")
                           for slot in ("step_D", "update_H"))
                       and unflagged["selected"] == today["selected"] == with_row["selected"]
                       and family.FAMILY not in " ".join(unflagged["selected"].values())
                       and today["launches"] == unflagged["launches"]),
        "today": today,
        "with_the_absorb_row_in_process": with_row,
        "with_the_row_and_INSTALLABLE_held_True": unflagged,
        "incumbent_on_step_D": incumbent,
        "what_this_says": (
            f"the composer installs {today['launches']} launches over step_B..update_E "
            f"today and, the absorb row being wired, refuses this product on "
            f"INSTALLABLE = False (the in-process row changes nothing); with the flag "
            f"held out of the way it refuses it BY NAME, because a released neighbour "
            f"installed first and holds a slot this product needs. THE REFUSING SLOT "
            f"MOVED ON 2026-09-17: it was step_D, held by the released D->E pair "
            f"({incumbent!r}), while the B->H cylindrical twin held no absorb row and "
            f"could not install; that twin has one now, so update_H carries "
            f"{unflagged['selected'].get('update_H')!r} and answers first. The "
            f"arbitration moved with it -- this product takes one slot from EACH "
            f"neighbouring seam, and where both neighbours fuse that is a LOSS (two "
            f"pairs become one) rather than the TIE it was while only one did. "
            f"Measured here: {unflagged['refusal'][0] if unflagged['refusal'] else '-'}"),
    }


# ---------------------------------------------------------------------------
# Device legs
# ---------------------------------------------------------------------------

def leg_recompute_identity(label: str, seed: int) -> Dict[str, Any]:
    """The prefix from the RECOMPUTED Hy equals the shipped scan over the STORED H.

    Three comparisons after one ``run`` of the plan:

    * device: the shipped ``cyl_rderiv_prefix`` over the rotated ``H_out`` tensor into a
      second scratch, against the plan's prefix tensor -- 0 words or the recompute is
      not the pointwise value;
    * host: ``stepping.cylindrical_rderiv_prefix`` over the synced ``Hy`` -- the oracle;
    * NULL: the shipped scan over the PRE-launch ``Hy`` (the plan's twin after the
      rotation) must DIFFER, or the leg could not tell the two H's apart.
    """
    import torch  # noqa: PLC0415

    driver = build_driver(label, seed)
    residency = Residency()
    plan = build_subject(driver, residency)
    residency.sync_in()
    plan.run()
    residency.sync_out()
    fields = driver.fields
    oracle = stepping.cylindrical_rderiv_prefix(np, np.asarray(fields.Hy), family.IR0)
    host_diff = differing(oracle, plan.prefix_host)
    shipped_scan = cyl.compile_cylindrical_prefix("step_D")
    nx, ny, nz = plan.shape
    new_hy = residency.tensor_for_host(fields.Hy)
    old_hy = residency.tensor_for_host(plan.rotated["Hy"])
    weights = residency.tensor(f"cyl_weights:{family.CURL_SUB_STEP}")
    divisor = residency.tensor(f"cyl_divisor:{family.CURL_SUB_STEP}")
    prefix = residency.tensor(f"cyl_pfx:{family.CURL_SUB_STEP}")
    second = torch.zeros_like(prefix)
    shipped_scan(second, new_hy, weights, divisor, nx, ny, nz, ny * nz)
    torch.mps.synchronize()
    device_diff = differing(second.cpu().numpy(), prefix.cpu().numpy())
    control = torch.zeros_like(prefix)
    shipped_scan(control, old_hy, weights, divisor, nx, ny, nz, ny * nz)
    torch.mps.synchronize()
    null_diff = differing(control.cpu().numpy(), prefix.cpu().numpy())
    h_moved = differing(np.asarray(fields.Hy), plan.rotated["Hy"])
    return {
        "passed": bool(host_diff == 0 and device_diff == 0 and null_diff > 0 and h_moved > 0
                       and plan.constitutive_launches == plan.curl_launches == 1),
        "words": int(oracle.size),
        "prefix_vs_oracle_host_differing": host_diff,
        "prefix_vs_shipped_scan_over_H_out_differing": device_diff,
        "null_shipped_scan_over_pre_launch_Hy_differing": null_diff,
        "hy_words_moved_by_update_H": h_moved,
        "constitutive_launches": plan.constitutive_launches,
        "curl_launches": plan.curl_launches,
        "shape": list(plan.shape),
    }


def leg_launch_structure(driver: Any, steps: int) -> Dict[str, Any]:
    arrangements = all_arrangements(driver)
    result = drive(driver, arrangements, steps, stop_on_divergence=False)
    rows = {}
    stepped = max(result["steps_stepped"], 1)
    for name, arrangement in arrangements.items():
        shim = arrangement.shim
        counts = arrangement.launches()
        seam = 0
        if shim is not None:
            for slot in family.REPLACES:
                plan = shim.plans.get(slot)
                if plan is not None and plan is not ABSORBED:
                    seam += int(getattr(declaring(plan), "launches_per_run", 1))
        rows[name] = {
            "launches_per_step_whole_step_plans": counts["plans"] / stepped,
            "launches_per_step_whole_step_functions": counts["functions"] / stepped,
            "launches_per_step_at_the_seam": seam,
            "dispatched": {} if shim is None else dict(shim.dispatched),
            "absorbed": {} if shim is None else dict(shim.absorbed),
            "selected": arrangement.selected,
        }
    weld, singles = rows["weld"], rows["singles"]
    today, unfused = rows["composition_today"], rows["unfused"]
    return {
        "passed": bool(weld["launches_per_step_at_the_seam"] == 2
                       and singles["launches_per_step_at_the_seam"] == 3
                       and all(row["launches_per_step_whole_step_plans"]
                               == row["launches_per_step_whole_step_functions"]
                               for row in rows.values())
                       and result["step_error"] is None
                       and unfused["launches_per_step_whole_step_plans"]
                       - weld["launches_per_step_whole_step_plans"] == 1),
        "arrangements": rows,
        "seam_reduction_against_the_singles":
            singles["launches_per_step_at_the_seam"] - weld["launches_per_step_at_the_seam"],
        "whole_step_reduction_against_the_unfused_composition":
            unfused["launches_per_step_whole_step_plans"]
            - weld["launches_per_step_whole_step_plans"],
        "whole_step_reduction_against_the_composition_installed_today":
            today["launches_per_step_whole_step_plans"]
            - weld["launches_per_step_whole_step_plans"],
        "what_this_says": (
            "at the seam the pair is 2 launches where the certified singles are 3 (the "
            "scan, the curl, the constitutive); over the whole step it saves one against "
            "the unfused composition and ties the composition the composer installs "
            "today, which already fuses the D->E seam of these rows. Counts, not time"),
    }


# ---------------------------------------------------------------------------
# The armed defects
# ---------------------------------------------------------------------------

def needle(source: str, old: str, new: str, count: int = 1) -> str:
    hits = source.count(old)
    if hits < count or hits == 0:
        raise AssertionError(f"mutation needle matches {hits} times, expected >= {count}: "
                             f"{old!r}")
    mutated = source.replace(old, new, count)
    if mutated == source:
        raise AssertionError(f"mutation needle changed nothing: {old!r}")
    return mutated


_TAIL = cartesian.H_CELL_TAIL_ARGS
_OWN_CALL = f"h_cell_result own = h_cell(i, j, k, {_TAIL});"
_SCAN_READ = f"float w = h_cell(i, j, k, {_TAIL}).a1 * weights[i];"
_SCAN_ROW0 = f"float prev = h_cell(0, j, k, {_TAIL}).a1 * weights[0];"

#: Mutation id -> what it is DECLARED to do. Anything not named is scored on CATCHING.
#: ``null`` must be CONFIRMED (a null that is caught means its reasoning is wrong);
#: ``schedule_dependent`` is recorded and does not decide the verdict -- its must-catch
#: twin is armed on RACE_CASE below.
MUTATION_EXPECTATION: Dict[str, str] = {
    "scan_weight_multiply_commuted": "null",
    "scan_accumulator_commuted": "null",
    "dz_curl_distributes_dtdx_over_the_outer_sum": "null",
    "scan_reads_the_scratch_hy_small_grid": "schedule_dependent",
}


def constitutive_mutations() -> Dict[str, str]:
    """Armed defects in launch 1's source, each a plausible transcription slip."""
    base = family.cylindrical_real_fused_hd_pair_constitutive_source()
    return {
        # --- THE SCAN --------------------------------------------------------
        "scan_divide_is_a_reciprocal_multiply":
            needle(base, "float inc = (w - prev) / divisor[i - 1];",
                   "float inc = (w - prev) * (1.0f / divisor[i - 1]);"),
        "scan_weights_shifted_down":
            needle(base, _SCAN_READ, _SCAN_READ.replace("weights[i]", "weights[i - 1]")),
        "scan_row0_weight_shifted_up":
            needle(base, _SCAN_ROW0, _SCAN_ROW0.replace("weights[0]", "weights[1]")),
        "scan_skips_the_last_row":
            needle(base, "for (int i = 1; i < nxi; ++i)", "for (int i = 1; i < nxi - 1; ++i)"),
        "scan_recomputes_hx_not_hy":
            needle(base, _SCAN_READ, _SCAN_READ.replace(").a1", ").a0")),
        "scan_reads_the_pre_launch_hy":
            needle(base, _SCAN_READ, "float w = hi1[i * nyz + base] * weights[i];"),
        "scan_reads_the_scratch_hy_small_grid":
            needle(base, _SCAN_READ, "float w = ho1[i * nyz + base] * weights[i];"),
        "scan_weight_multiply_commuted":
            needle(base, _SCAN_READ,
                   f"float w = weights[i] * h_cell(i, j, k, {_TAIL}).a1;"),
        "scan_accumulator_commuted":
            needle(base, "acc = acc + inc;", "acc = inc + acc;"),
        # --- THE CONSTITUTIVE -------------------------------------------------
        "own_h_from_the_scratch":
            needle(base, _OWN_CALL, _OWN_CALL.replace("hi1", "ho1")),
        "own_fw_prev_from_the_scratch":
            needle(base, _OWN_CALL, _OWN_CALL.replace("wi1", "wo1")),
        "constitutive_accumulations_reversed":
            needle(base, "        a1 = a1 + kp_1 * src1;\n        a1 = a1 - km_1 * prev1;",
                   "        a1 = a1 - km_1 * src1;\n        a1 = a1 + kp_1 * prev1;"),
        "constitutive_coefficient_index_moved":
            needle(base, "float kp_0 = kp0[i], km_0 = kmx[i];",
                   "float kp_0 = kp0[k], km_0 = kmx[k];"),
        "fw_store_dropped": needle(base, "wo0[ii] = own.src0; ", ""),
        "h_store_dropped": needle(base, " ho2[ii] = own.a2;", ""),
    }


def race_mutation() -> str:
    """The scratch-read race, for the corpus-scale case where it must be caught."""
    base = family.cylindrical_real_fused_hd_pair_constitutive_source()
    return needle(base, _SCAN_READ, "float w = ho1[i * nyz + base] * weights[i];")


def curl_mutations(codes: Sequence[int]) -> Dict[str, str]:
    """Armed defects in launch 2 -- the certified curl's text, mutated through the seam."""
    base = family.cylindrical_real_fused_hd_pair_curl_source(codes)
    dz = "float curl2 = dtdx * ((pb_x - pb) + (a - a_y));"
    return {
        # THE bz_flat_grouping CLASS: dtdx distributed INTO the prefix difference.
        "dz_curl_distributes_dtdx_into_the_prefix_difference":
            needle(base, dz, "float curl2 = ((dtdx * pb_x) - (dtdx * pb)) + dtdx * (a - a_y);"),
        "dz_curl_parens_flattened":
            needle(base, dz, "float curl2 = dtdx * (pb_x - pb + a - a_y);"),
        # DECLARED NULL: the phi pair is an exact +0.0 on the one-cell invariant axis.
        "dz_curl_distributes_dtdx_over_the_outer_sum":
            needle(base, dz, "float curl2 = (dtdx * (pb_x - pb)) + (dtdx * (a - a_y));"),
        "dy_curl_parens_flattened_on_a_live_pair":
            needle(base, "dtdx * ((a_z - a) + (c - c_x))", "dtdx * (a_z - a + c - c_x)"),
        "dz_prefix_difference_is_forward":
            needle(base, dz, "float curl2 = dtdx * ((pb - pb_x) + (a - a_y));"),
        "dx_reads_the_prefixed_hy":
            needle(base, "float b   = g1[ii];", "float b   = pfx[ii];"),
        "axis_increment_dropped":
            needle(base, "    v2 = at_x ? (v2 + (axis_coef * g1[ii])) : v2;\n", ""),
    }


def byte_neutral_source() -> str:
    """The leader's row-0 recompute replaced by the REGISTER the thread already holds.

    Not a defect: at the leader ``i == 0``, ``own`` IS ``h_cell(0, j, k, ...)`` -- the
    same function at the same cell over the same const inputs -- so this edit is the
    recompute's central claim written as a program, and must NOT diverge.
    """
    base = family.cylindrical_real_fused_hd_pair_constitutive_source()
    return needle(base, _SCAN_ROW0, "float prev = own.a1 * weights[0];")


def _compile_constitutive(source: str) -> Dict[str, Any]:
    return {shaders.CONTRACT_OFF: compile_source(source).cyl_real_hd_constitutive_prefix}


def _compile_curl(source: str) -> Dict[str, Any]:
    return {shaders.CONTRACT_OFF: compile_source(source).cyl_pml_curl_step}


# --- host defects: the launcher seam ----------------------------------------------

class Launcher:
    """A ``run`` replacement over the plan's own primitives; counters stay the plan's."""

    __slots__ = ("inner", "kind")

    def __init__(self, plan: Any, kind: str) -> None:
        self.inner = plan
        self.kind = kind

    @property
    def absorbed_by(self) -> Any:
        return self.inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def run(self) -> None:
        plan = self.inner
        plan.runs += 1
        writes, reads = plan.resolve()
        if self.kind == "in_place_H":
            for index, name in enumerate(plan.rotated_names):
                if name in ("Hx", "Hy", "Hz"):
                    writes[index] = reads[index]
            plan.launch_constitutive(writes, reads)
            for name in plan.rotated_names:
                if name in ("Hx", "Hy", "Hz"):
                    continue
                current = getattr(plan.fields, name)
                setattr(plan.fields, name, plan.rotated[name])
                plan.rotated[name] = current
            plan.launch_curl(writes[:3])
        elif self.kind == "in_place_fw":
            for index, name in enumerate(plan.rotated_names):
                if name.startswith("f_w_"):
                    writes[index] = reads[index]
            plan.launch_constitutive(writes, reads)
            for name in plan.rotated_names:
                if name.startswith("f_w_"):
                    continue
                current = getattr(plan.fields, name)
                setattr(plan.fields, name, plan.rotated[name])
                plan.rotated[name] = current
            plan.launch_curl(writes[:3])
        elif self.kind == "curl_reads_the_pre_launch_H":
            plan.launch_constitutive(writes, reads)
            plan.rotate()
            plan.launch_curl(reads[:3])
        elif self.kind == "rotation_skipped":
            plan.launch_constitutive(writes, reads)
            plan.launch_curl(writes[:3])
        elif self.kind == "curl_launched_first":
            plan.launch_curl(reads[:3])
            plan.launch_constitutive(writes, reads)
            plan.rotate()
        else:  # pragma: no cover - a misspelled kind is a harness defect
            raise ValueError(self.kind)


def _launcher(kind: str) -> Callable[[Any, Residency, Any], Any]:
    return lambda plan, _residency, _pml: Launcher(plan, kind)


def _rebind_constitutive(plan: Any, slots: Sequence[int], tensors: Sequence[Any]) -> None:
    static = list(plan._constitutive_static)  # noqa: SLF001 - the family's own seam
    for slot, tensor in zip(slots, tensors):
        static[slot] = tensor
    plan._constitutive_static = tuple(static)  # noqa: SLF001


def _rebind_curl(plan: Any, slots: Sequence[int], tensors: Sequence[Any]) -> None:
    static = list(plan._curl_static)  # noqa: SLF001
    for slot, tensor in zip(slots, tensors):
        static[slot] = tensor
    plan._curl_static = tuple(static)  # noqa: SLF001


def constitutive_takes_the_half_integer_lattice(plan: Any, residency: Residency,
                                                pml: Any) -> Any:
    """HOST DEFECT: kps/kms from the ``_h`` sub-lattice for the H constitutive.

    ``update_H`` reads INTEGER positions; the kernel takes six pointers and never asks
    which lattice they came from, so no shader mutation can reach this.
    """
    _rebind_constitutive(
        plan, range(3, 9),
        [residency.mirror(f"mutation:kps_{axis}_h", getattr(pml, f"kps_{axis}_h"),
                          constant=True) for axis in "xyz"]
        + [residency.mirror(f"mutation:kms_{axis}_h", getattr(pml, f"kms_{axis}_h"),
                            constant=True) for axis in "xyz"])
    return plan


def curl_takes_the_half_integer_lattice(plan: Any, residency: Residency, pml: Any) -> Any:
    """HOST DEFECT: the curl's six kms/sinv vectors from the ``_h`` sub-lattice."""
    _rebind_curl(
        plan, range(7, 13),
        [residency.mirror(f"mutation:{stem}_{axis}_h", getattr(pml, f"{stem}_{axis}_h"),
                          constant=True)
         for axis in "xyz" for stem in ("kms", "sinv")])
    return plan


HOST_MUTATIONS: Dict[str, Tuple[Callable[[Any, Residency, Any], Any], str]] = {
    "in_place_H": (_launcher("in_place_H"),
                   "THE IN-PLACE NULL CONTROL: H written where it is read. The leader's "
                   "recompute of a row already stored reads the NEW H where it needs "
                   "the old one"),
    "in_place_fw": (_launcher("in_place_fw"),
                    "THE SECOND IN-PLACE NULL CONTROL: f_w_H written where it is read. A "
                    "recompute of a stored row reads B where it needs B_prev"),
    "curl_reads_the_pre_launch_H": (_launcher("curl_reads_the_pre_launch_H"),
                                    "launch 2 bound to the pre-launch H: the curl over the "
                                    "previous half-step's magnetic field"),
    "rotation_skipped": (_launcher("rotation_skipped"),
                         "the engine keeps naming the pre-launch buffers; H is stale "
                         "under its own name"),
    "curl_launched_first": (_launcher("curl_launched_first"),
                            "the order reversed: the curl reads the stale prefix and the "
                            "pre-launch H"),
    "constitutive_takes_the_half_integer_lattice":
        (constitutive_takes_the_half_integer_lattice,
         "a half-cell-wrong absorber profile in update_H"),
    "curl_takes_the_half_integer_lattice":
        (curl_takes_the_half_integer_lattice,
         "a half-cell-wrong absorber profile in step_D"),
}


def scan_ladder_at_ir0_zero(shape: Sequence[int]) -> Tuple[Any, Any]:
    """HOST DEFECT: the ``ir0 = 0.0`` ladder -- the B side's -- bound to the D side's scan."""
    weights, divisor = stepping._cylindrical_rderiv_weights(  # noqa: SLF001
        np, int(shape[0]), 0.0, np.float32)
    return (np.ascontiguousarray(weights).reshape(-1),
            np.ascontiguousarray(divisor).reshape(-1))


def leg_mutants_compile(codes: Sequence[int]) -> Dict[str, Any]:
    rows = {}
    for name, source in constitutive_mutations().items():
        ok, error = _compiles(source)
        rows[name] = {"compiled": ok, "error": error, "launch": 1}
    for name, source in curl_mutations(codes).items():
        ok, error = _compiles(source)
        rows[name] = {"compiled": ok, "error": error, "launch": 2}
    ok, error = _compiles(race_mutation())
    rows["scan_reads_the_scratch_hy_corpus_scale"] = {"compiled": ok, "error": error,
                                                      "launch": 1}
    ok, error = _compiles(byte_neutral_source())
    rows["byte_neutral_control"] = {"compiled": ok, "error": error, "launch": 1}
    return {"passed": all(row["compiled"] for row in rows.values()),
            "armed": len(rows) - 1, "mutants": rows}


# ---------------------------------------------------------------------------
# The corpus lift -- the census driver's battery hook, and the parent that spawns it
# ---------------------------------------------------------------------------

class _HostGrid:
    xp = np


def runtime_reasons() -> List[str]:
    """Why THIS process cannot evaluate this battery at all, or ``[]``."""
    from meep_gpu.metal_kernels import coverage  # noqa: PLC0415
    return list(coverage._metal_backend_reasons(_HostGrid()))  # noqa: SLF001


def _child_progress(message: str) -> None:
    path = os.environ.get("MEEP_GPU_CYLHD_GATE_PROGRESS")
    label = os.environ.get("MEEP_GPU_CYLHD_GATE_LABEL", "?")
    line = f"[{time.strftime('%H:%M:%S')}] {label} {message}"
    print(line, flush=True)
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()


def evaluate(driver: Any, probe: Any) -> Dict[str, Any]:  # noqa: ARG001 - the census probe is unused
    """The census driver's battery hook: the four-reference identity on ONE lifted row."""
    steps = int(os.environ.get("MEEP_GPU_CYLHD_GATE_STEPS", STEPS))
    sources = tuple(getattr(driver, "_sources", ()) or ())
    block: Dict[str, Any] = {
        "family": family.FAMILY, "steps_requested": steps, "n_sources": len(sources),
        "sources": [{"type": type(s).__name__,
                     "field_type": str(getattr(s, "field_type", "")),
                     "is_integrated": bool(getattr(s, "is_integrated", False)),
                     "withdraw_does_work": bool(withdraw_hoist._withdraw_does_work(s))}  # noqa: SLF001
                    for s in sources],
        "grid_shape": [int(n) for n in driver.grid.shape],
        "grid_cells": int(np.prod(driver.grid.shape)),
        "courant": float(driver.grid.dt / driver.grid.dx),
    }
    pin_array_path(driver)
    verdict = family.metal_cylindrical_real_fused_hd_pair_coverage(
        driver.fields, driver.pml, sources, Residency())
    block["predicate_admits"] = bool(verdict.covered)
    block["predicate_reasons"] = list(verdict.reasons)
    composed, _residency = _composed(driver, fuse=True)
    block["composer_selected"] = dict(composed.selected)
    block["composer_refuses_this_product"] = list(
        composed.reasons.get(f"fused_pair_{family.FAMILY}", ()))
    if not verdict.covered:
        block["driven"] = False
        block["why_not_driven"] = "the predicate refused this row"
        return {"metal_cylindrical_real_fused_hd_pair_gate": block}
    block["driven"] = True
    block.update(run_product(driver, steps, progress=_child_progress,
                             movement_floor="seam", require_full_budget=False,
                             clean_floor=1))
    _child_progress(f"passed={block['passed']} steps={block['steps_compared']} "
                    f"words={block['words_compared']} ({block['seconds']} s)")
    return {"metal_cylindrical_real_fused_hd_pair_gate": block}


def _load_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def lift_basis(results: Path) -> Tuple[List[dict], Dict[str, Any]]:
    """The rows the standing census puts in this product's cell, and the seam facts.

    DERIVED from the census's own ``plan_step.selected`` -- the arms the composer chose
    for each row -- never from a list here. The parameterised leg's rows are read from
    ``tests_param_matched.jsonl`` and carry BOTH names: the census child enumerates the
    shim's ``__idxN`` spelling (``case``), the boards file the matched name (``row``).
    """
    census = results / CENSUS
    seam = results / SEAM_RECORD
    if not census.is_dir() or not seam.is_dir():
        raise SystemExit(f"the lift leg needs {census} and {seam}")
    seam_rows = {row["label"]: row for row in _load_jsonl(seam / "h_to_d_seam.jsonl")}
    rows: List[dict] = []
    for leg, filename in (("examples", "examples.jsonl"), ("tests", "tests.jsonl"),
                          ("tests", "tests_param_matched.jsonl")):
        path = census / filename
        if not path.is_file():
            continue
        for record in _load_jsonl(path):
            if not record.get("measured"):
                continue
            selected = ((record.get("plan_step") or {}).get("selected") or {})
            if (selected.get("update_H"), selected.get("step_D")) != CELL_ARMS:
                continue
            label = f"{record.get('leg', leg)}:{record['row']}"
            seam_row = seam_rows.get(label, {})
            rows.append({
                "label": label, "leg": record.get("leg", leg), "row": record["row"],
                "case": record.get("case") or record["row"],
                "module": record.get("module") or seam_row.get("module"),
                "interpreter": (record.get("interpreter") or record.get("python")
                                or seam_row.get("interpreter") or sys.executable),
                "grid_cells": record.get("grid_cells"),
                "grid_shape": record.get("grid_shape"),
                "withdraw_in_seam": bool((seam_row.get("h_to_d_seam") or {})
                                         .get("withdraw_in_seam")),
                "census_selected": selected,
            })
    unique = {row["label"]: row for row in rows}
    facts = {"census": CENSUS, "seam_record": SEAM_RECORD, "cell_arms": list(CELL_ARMS),
             "rows_in_cell": len(unique),
             "rows_with_a_standing_withdraw": sorted(
                 label for label, row in unique.items() if row["withdraw_in_seam"])}
    return list(unique.values()), facts


def leg_lift(out_dir: Path, steps: int, timeout: float, resume: bool) -> Dict[str, Any]:
    """Every corpus row in the cell, re-lifted in its own interpreter and driven."""
    import gate_provenance  # noqa: PLC0415
    import measure_predicate_coverage as census  # noqa: PLC0415

    out_dir = Path(out_dir).resolve()
    rows, facts = lift_basis(HERE / "results")
    lift_dir = out_dir / "lift"
    per_row = lift_dir / "per_row"
    per_row.mkdir(parents=True, exist_ok=True)
    progress_log = lift_dir / "steps.progress.log"
    examples_dir = Path(census.EXAMPLES_DIR)
    tests_dir = Path(census.TESTS_DIR)
    if not tests_dir.is_dir():
        return {"passed": False, "facts": facts,
                "reason": (f"the MEEP corpus is not at {tests_dir}; set "
                           f"MEEP_GPU_CORPUS_ROOT to the checkout the census was cut "
                           f"over. Refused rather than measured on nothing")}
    probe_path = API_ROOT / census.PROBE
    environment = dict(os.environ)
    environment.update({"KMP_DUPLICATE_LIB_OK": "TRUE", "MPLBACKEND": "Agg",
                        "MEEP_GPU_SUBNORMAL_POLICY": "flush",
                        "MEEP_GPU_CYLHD_GATE_STEPS": str(steps),
                        "MEEP_GPU_CYLHD_GATE_PROGRESS": str(progress_log)})
    work = lift_dir / "workdir"
    work.mkdir(exist_ok=True)
    if examples_dir.is_dir():
        for entry in examples_dir.iterdir():
            if entry.suffix in (".py", ".ipynb"):
                continue
            link = work / entry.name
            if not link.exists():
                with contextlib.suppress(OSError):
                    link.symlink_to(entry)
    # THE CENSUS'S OWN SHIM, for the modules that decorate with `parameterized`.
    shim_path = Path(census.__file__).resolve().parent / "shim"
    needs_shim: set = set()
    for row in rows:
        module_name = row.get("module")
        if row["leg"] == "examples" or not module_name:
            continue
        module_file = tests_dir / module_name
        if module_file.is_file() and "import parameterized" in module_file.read_text(
                encoding="utf-8", errors="replace"):
            needs_shim.add(module_name)
    measured: List[dict] = []
    for index, row in enumerate(rows, start=1):
        record_path = per_row / (row["label"].replace(":", "__").replace(".", "_") + ".json")
        environment["MEEP_GPU_CYLHD_GATE_LABEL"] = row["label"]
        environment["PYTHONPATH"] = (f"{shim_path}{os.pathsep}{API_ROOT}"
                                     if row.get("module") in needs_shim else str(API_ROOT))
        started = time.time()
        if not (resume and record_path.exists()):
            if row["leg"] == "examples":
                command = [row["interpreter"], "-u", str(Path(census.__file__).resolve()),
                           "--leg", "examples", "--child-script",
                           str(examples_dir / row["row"]),
                           "--out-json", str(record_path), "--probe", str(probe_path),
                           "--progress-log", str(progress_log),
                           "--battery", Path(__file__).stem]
            else:
                if not row["module"]:
                    log(f"lift {index}/{len(rows)} {row['label']}: REFUSED, no module "
                        f"recorded for this tests row")
                    measured.append({**row, "measured": False, "note": "no module recorded"})
                    continue
                command = [row["interpreter"], "-u", str(Path(census.__file__).resolve()),
                           "--leg", "tests", "--child-module", str(tests_dir / row["module"]),
                           "--child-cases", json.dumps([row["case"]]),
                           "--out-json", str(record_path), "--probe", str(probe_path),
                           "--progress-log", str(progress_log),
                           "--battery", Path(__file__).stem]
            log(f"lift {index}/{len(rows)} {row['label']} start (cells {row.get('grid_cells')}, "
                f"case {row['case']})")
            stderr_text = ""
            note = "child died"
            try:
                completed = subprocess.run(command, cwd=str(work), env=environment,
                                           timeout=timeout, stdout=subprocess.DEVNULL,
                                           stderr=subprocess.PIPE, check=False)
                stderr_text = (completed.stderr or b"").decode("utf-8", "replace")
            except subprocess.TimeoutExpired as expired:
                note = "child timeout"
                stderr_text = ((expired.stderr or b"").decode("utf-8", "replace")
                               if expired.stderr else "")
            if not record_path.exists():
                died = {"row": row["row"], "case": row["case"], "measured": False,
                        "note": note, "stderr_tail": stderr_text[-1500:]}
                gate_provenance.stamp(died)
                record_path.write_text(json.dumps(died, default=str), encoding="utf-8")
        payload = json.loads(record_path.read_text(encoding="utf-8"))
        candidates = payload if isinstance(payload, list) else [payload]
        record = next((r for r in candidates
                       if r.get("row") in (row["case"], row["row"])),
                      candidates[0] if candidates else {})
        record.update({key: value for key, value in row.items() if key != "label"})
        record["label"] = row["label"]
        measured.append(record)
        block = record.get("metal_cylindrical_real_fused_hd_pair_gate") or {}
        log(f"lift {index}/{len(rows)} {row['label']}: measured={record.get('measured')} "
            f"admits={block.get('predicate_admits')} driven={block.get('driven')} "
            f"passed={block.get('passed')} steps={block.get('steps_compared')} "
            f"banded_at={block.get('first_banded_step')} "
            f"words={block.get('words_compared')} ({time.time() - started:.1f} s)")
    with (lift_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for record in measured:
            handle.write(json.dumps(record, default=str) + "\n")

    blocks = {r["label"]: (r.get("metal_cylindrical_real_fused_hd_pair_gate") or {})
              for r in measured}
    admitted = sorted(label for label, b in blocks.items() if b.get("predicate_admits"))
    unmeasured = sorted(label for label, b in blocks.items() if "predicate_admits" not in b)
    driven = sorted(label for label in admitted if blocks[label].get("driven"))
    identical = sorted(label for label in driven if blocks[label].get("bit_identical"))
    cleared = sorted(label for label in identical
                     if int(blocks[label].get("steps_compared") or 0) >= LIFT_CLEAN_STEP_FLOOR)
    below_floor = {label: {"steps_compared": blocks[label].get("steps_compared"),
                           "first_banded_step": blocks[label].get("first_banded_step")}
                   for label in identical if label not in cleared}
    diverged = sorted(label for label in driven if not blocks[label].get("bit_identical"))
    composer_refuses = all(blocks[label].get("composer_refuses_this_product")
                           for label in admitted)
    return {
        "passed": bool(rows and not unmeasured and len(admitted) == len(rows)
                       and len(driven) == len(admitted) and not diverged
                       and len(cleared) == len(driven) and composer_refuses),
        "facts": facts,
        "rows_in_cell": len(rows),
        "rows": [r["label"] for r in rows],
        "unmeasured": unmeasured,
        "admitted": admitted,
        "driven": driven,
        "bit_identical": identical,
        "cleared_the_floor": cleared,
        "below_the_floor": below_floor,
        "diverged": diverged,
        "clean_step_floor": LIFT_CLEAN_STEP_FLOOR,
        "complete_driver_steps_compared": int(sum(
            int(blocks[label].get("steps_compared") or 0) for label in driven)),
        "words_compared": int(sum(int(blocks[label].get("words_compared") or 0)
                                  for label in driven)),
        "per_row": {label: {
            "grid_shape": blocks[label].get("grid_shape"),
            "steps_compared": blocks[label].get("steps_compared"),
            "first_banded_step": blocks[label].get("first_banded_step"),
            "words_compared": blocks[label].get("words_compared"),
            "bit_identical": blocks[label].get("bit_identical"),
            "prefix_identical_to_the_oracle_every_step":
                blocks[label].get("prefix_identical_to_the_oracle_every_step"),
            "weld_agrees_with_the_certified_singles":
                blocks[label].get("weld_agrees_with_the_certified_singles"),
            "composer_selected": blocks[label].get("composer_selected"),
            "composer_refuses_this_product": blocks[label].get("composer_refuses_this_product"),
            "seam_output_words_moved": blocks[label].get("seam_output_words_moved"),
            "sources": blocks[label].get("sources"),
        } for label in blocks},
        "the_composer_refuses_this_product_on_every_admitted_row": composer_refuses,
        "corpus_root": str(tests_dir.parent),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def emit(handle: Any, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
    log(f"{row['index']}/{row['total']} [{row['leg']}] {row['label']}: "
        f"passed={row.get('passed')} diff={row.get('differing_words', '-')} "
        f"launches={row.get('launches', '-')}")


def parse_legs(value: str) -> Tuple[str, ...]:
    if value.strip() == "all":
        return ALL_LEGS
    out: List[str] = []
    for token in value.split(","):
        token = token.strip()
        if token in LEG_GROUPS:
            out.extend(LEG_GROUPS[token])
        elif token in ALL_LEGS:
            out.append(token)
        elif token:
            raise SystemExit(f"unknown leg {token!r}; legs are {ALL_LEGS}")
    return tuple(dict.fromkeys(out))


def _mutation_row(name: str, result: Mapping[str, Any], expectation: str,
                  extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    weld = result["arrangements"]["weld"]
    caught = not weld["identical"]
    if expectation == "caught":
        passed = caught and weld["launches"]["plans"] > 0
    elif expectation == "null":
        passed = (not caught) and bool(result["passed"])
    else:  # schedule_dependent: recorded, never decides
        passed = weld["launches"]["plans"] > 0
    return {"caught": caught, "expectation": expectation, "passed": bool(passed),
            "first_divergence": weld["first_divergence"],
            "differing_words": weld["differing_words_final"],
            "differing_volumes": weld["differing_volumes_final"],
            "launches": weld["launches"], "steps_stepped": result["steps_stepped"],
            **(extra or {})}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--legs", default="all",
                        help="comma list of legs or groups (host, compile, device); "
                             "anything short of all exits 75 (cannot certify)")
    parser.add_argument("--lift-timeout", type=float, default=3600.0)
    parser.add_argument("--lift-resume", action="store_true")
    parser.add_argument("--progress-log", type=Path, default=None)
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.steps < 1:
        raise SystemExit("--steps must be positive")
    legs = parse_legs(args.legs)
    progress_path = args.progress_log.resolve() if args.progress_log else None

    def progress(message: str) -> None:
        log(message)
        if progress_path:
            with open(progress_path, "a", encoding="utf-8") as handle:
                handle.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")

    report = subnormal.mps_policy_report()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    jsonl = args.out.with_suffix(".jsonl")
    started = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    needs_device = any(leg in LEG_GROUPS["device"] or leg in LEG_GROUPS["compile"]
                       for leg in legs)
    device_ok = True
    if needs_device:
        try:
            import torch  # noqa: PLC0415
            device_ok = bool(torch.backends.mps.is_available())
        except Exception:  # noqa: BLE001
            device_ok = False
    from meep_gpu.stepping import _boundary_kinds  # noqa: PLC0415
    probe = build_driver(MUTATION_CASE, 1)
    codes = cyl.boundary_codes(_boundary_kinds(probe.grid, probe.pml))
    assert probe.grid.shape[0] == probe.grid.shape[2], (
        f"{MUTATION_CASE} is not square; the coefficient-index mutation would read a "
        f"length-nr vector past its end")
    assert float(probe.grid.dt / probe.grid.dx) not in (0.5, 0.25), (
        f"{MUTATION_CASE} has a power-of-two Courant; the grouping defects would be exact")
    constitutive = constitutive_mutations()
    curl = curl_mutations(codes)
    total = (len([leg for leg in legs if leg not in ("product", "value_class", "mutation")])
             + (len(CASES) if "product" in legs else 0)
             + (len(CASES) + 1 if "value_class" in legs else 0)
             + (len(constitutive) + len(curl) + len(HOST_MUTATIONS) + 2
                if "mutation" in legs else 0))
    index = 0

    def add(leg: str, label: str, payload: Mapping[str, Any]) -> None:
        nonlocal index
        index += 1
        row = {"index": index, "total": total, "leg": leg, "label": label, **payload}
        rows.append(row)
        emit(handle, row)

    with jsonl.open("w", encoding="utf-8") as handle:
        if not report["admitted"]:
            add("policy", "resolved_subnormal_policy_refused",
                {"passed": False, "refused": True, "report": report})
            legs = tuple(leg for leg in legs if leg in LEG_GROUPS["host"])
        if needs_device and not device_ok:
            add("device", "mps_unavailable",
                {"passed": False, "refused": True,
                 "reason": "MPS is not available; the compile and device legs cannot run"})
            legs = tuple(leg for leg in legs if leg in LEG_GROUPS["host"])

        if "binding_ceiling" in legs:
            add("binding_ceiling", "twenty_five_bindings_six_of_headroom_bisected",
                leg_binding_ceiling())
        if "transcription" in legs:
            add("transcription", "lifted_certified_text_with_the_declared_edits",
                leg_transcription())
        if "scan_order_host" in legs:
            add("scan_order_host", "numpy_cumsum_is_serial_over_the_shipped_increments",
                leg_scan_order_host())
        if "driver_order" in legs:
            add("driver_order", "replaces_is_the_drivers_two_adjacent_consults",
                leg_driver_order())
        if "refusal" in legs:
            add("refusal", "the_source_seam_the_geometry_and_the_declarations",
                leg_refusal())
        if "arbitration" in legs:
            add("arbitration", "the_composer_today_and_with_the_row_in_process",
                leg_arbitration())
        if "mutants_compile" in legs:
            add("mutants_compile", "every_armed_defect_compiles", leg_mutants_compile(codes))
        if "recompute_identity" in legs:
            add("recompute_identity", "recomputed_hy_prefix_equals_the_shipped_scan",
                leg_recompute_identity(MUTATION_CASE, 51000))
        if "product" in legs:
            for offset, (name, _keywords) in enumerate(CASES):
                driver = build_driver(name, 61000 + offset)
                result = run_product(driver, args.steps, progress=progress)
                add("product", name, {"steps": args.steps, **result,
                                      "differing_words":
                                          result["arrangements"]["weld"]["differing_words_final"],
                                      "launches": result["arrangements"]["weld"]["launches"]})
        if "value_class" in legs:
            for offset, (name, _keywords) in enumerate(CASES):
                driver = build_driver(name, 63000 + offset, values.PM_ZERO_LATTICE)
                result = run_product(driver, args.steps, progress=progress,
                                     value_class=values.PM_ZERO_LATTICE)
                add("value_class", f"{name}:{values.PM_ZERO_LATTICE}",
                    {"steps": args.steps, **result,
                     "differing_words": result["arrangements"]["weld"]["differing_words_final"],
                     "launches": result["arrangements"]["weld"]["launches"]})
            ladder_rows = []
            scale = 1.0
            for rung in range(4):
                scale *= 1e-9
                driver = build_driver(MUTATION_CASE, 65000 + rung, scale=scale)
                driver.step()
                census = sum(subnormal.census(v) for v in stored_volumes(driver.fields).values())
                ladder_rows.append({"rung": rung, "scale": scale, "reference_subnormals": census})
                log(f"    ladder rung {rung}: scale={scale:.3e} subnormals={census}")
                if census:
                    break
            add("value_class", "subnormal_band_is_refused_and_the_census_fires",
                {"passed": bool(ladder_rows and ladder_rows[-1]["reference_subnormals"] > 0),
                 "rows": ladder_rows, "note": values.band_refusal_note(),
                 "policy": os.environ.get("MEEP_GPU_SUBNORMAL_POLICY")})
        if "launch_structure" in legs:
            add("launch_structure", "launches_per_step_two_witnesses",
                leg_launch_structure(build_driver(MUTATION_CASE, 67000), args.steps))
        if "byte_neutral" in legs:
            driver = build_driver(MUTATION_CASE, 71000)
            result = run_product(driver, args.steps,
                                 constitutive_function=_compile_constitutive(byte_neutral_source()))
            add("byte_neutral", "row0_recompute_replaced_by_the_register",
                {"diverged": not result["arrangements"]["weld"]["identical"],
                 "passed": bool(result["passed"]),
                 "differing_words": result["arrangements"]["weld"]["differing_words_final"],
                 "launches": result["arrangements"]["weld"]["launches"]})
        if "mutation" in legs:
            for name, source in constitutive.items():
                driver = build_driver(MUTATION_CASE, 72000 + index)
                result = run_product(driver, args.steps,
                                     constitutive_function=_compile_constitutive(source))
                add("mutation", name, _mutation_row(
                    name, result, MUTATION_EXPECTATION.get(name, "caught"),
                    {"launch": 1, "case": MUTATION_CASE}))
            driver = build_driver(RACE_CASE, 72000 + index)
            result = run_product(driver, min(args.steps, 3),
                                 constitutive_function=_compile_constitutive(race_mutation()))
            add("mutation", "scan_reads_the_scratch_hy_corpus_scale", _mutation_row(
                "scan_reads_the_scratch_hy_corpus_scale", result, "caught",
                {"launch": 1, "case": RACE_CASE,
                 "why": "a leader reading H_out races the threads storing it; on this "
                        "grid the read lands before the store and is caught. Its "
                        "small-grid twin reads the intended value by schedule and is "
                        "recorded, not scored"}))
            for name, source in curl.items():
                driver = build_driver(MUTATION_CASE, 72000 + index)
                result = run_product(driver, args.steps, curl_function=_compile_curl(source))
                add("mutation", name, _mutation_row(
                    name, result, MUTATION_EXPECTATION.get(name, "caught"),
                    {"launch": 2, "case": MUTATION_CASE}))
            driver = build_driver(MUTATION_CASE, 72000 + index)
            result = run_product(driver, args.steps,
                                 scan_vectors=scan_ladder_at_ir0_zero(driver.grid.shape))
            add("mutation", "scan_ladder_at_ir0_zero", _mutation_row(
                "scan_ladder_at_ir0_zero", result, "caught",
                {"host_defect": True, "case": MUTATION_CASE}))
            for name, (launcher, why) in HOST_MUTATIONS.items():
                driver = build_driver(MUTATION_CASE, 72000 + index)
                result = run_product(driver, args.steps, launcher=launcher)
                add("mutation", name, _mutation_row(
                    name, result, "caught", {"host_defect": True, "case": MUTATION_CASE,
                                             "why": why}))
        if "disarm" in legs:
            driver = build_driver(MUTATION_CASE, 97000)
            result = run_product(driver, args.steps)
            add("disarm", "shipped_bytes_on_the_mutation_case",
                {"passed": bool(result["passed"]),
                 "differing_words": result["arrangements"]["weld"]["differing_words_final"],
                 "launches": result["arrangements"]["weld"]["launches"],
                 "arrays_that_never_moved": result["arrays_that_never_moved"]})
        if "lift" in legs:
            add("lift", "every_corpus_row_in_the_cell",
                leg_lift(args.out.parent, args.steps, args.lift_timeout, args.lift_resume))

    ran = tuple(dict.fromkeys(row["leg"] for row in rows))
    complete = all(leg in ran for leg in ALL_LEGS)
    all_passed = bool(rows) and all(row["passed"] for row in rows)
    if complete:
        verdict = "PASS" if all_passed else "FAIL"
    else:
        verdict = ("INCOMPLETE: not every leg ran -- "
                   f"{'all requested legs passed' if all_passed else 'a requested leg FAILED'}; "
                   f"missing {sorted(set(ALL_LEGS) - set(ran))}")
    import torch  # noqa: PLC0415

    import gate_provenance  # noqa: PLC0415

    mutation_rows = [r for r in rows if r["leg"] == "mutation"]
    result = {
        "verdict": verdict,
        "complete": complete,
        "legs_requested": list(legs),
        "legs_run": list(ran),
        "legs_missing": sorted(set(ALL_LEGS) - set(ran)),
        "elapsed_seconds": time.perf_counter() - started,
        "steps": args.steps,
        "rows": rows,
        "counts": {
            "product_cases": len(CASES) if "product" in ran else 0,
            "value_class_cases": len(CASES) if "value_class" in ran else 0,
            "mutations_armed": len(mutation_rows),
            "mutations_scored_on_catching": sum(
                1 for r in mutation_rows if r["expectation"] == "caught"),
            "mutations_caught": sum(1 for r in mutation_rows
                                    if r["expectation"] == "caught" and r["caught"]),
            "declared_nulls": sum(1 for r in mutation_rows if r["expectation"] == "null"),
            "declared_nulls_confirmed": sum(
                1 for r in mutation_rows if r["expectation"] == "null" and not r["caught"]),
            "schedule_dependent_recorded": sum(
                1 for r in mutation_rows if r["expectation"] == "schedule_dependent"),
            "schedule_dependent_caught": sum(
                1 for r in mutation_rows
                if r["expectation"] == "schedule_dependent" and r["caught"]),
            "shader_mutations": len(constitutive) + len(curl) + 1 if "mutation" in ran else 0,
            "host_mutations": len(HOST_MUTATIONS) + 1 if "mutation" in ran else 0,
            "byte_neutral_controls": 1 if "byte_neutral" in ran else 0,
            "lift_rows_in_cell": next((r.get("rows_in_cell") for r in rows
                                       if r["leg"] == "lift"), None),
            "lift_rows_admitted": next((len(r.get("admitted") or []) for r in rows
                                        if r["leg"] == "lift"), None),
            "lift_rows_driven": next((len(r.get("driven") or []) for r in rows
                                      if r["leg"] == "lift"), None),
            "lift_rows_bit_identical": next((len(r.get("bit_identical") or []) for r in rows
                                             if r["leg"] == "lift"), None),
            "lift_complete_driver_steps": next(
                (r.get("complete_driver_steps_compared") for r in rows
                 if r["leg"] == "lift"), None),
            "lift_words_compared": next((r.get("words_compared") for r in rows
                                         if r["leg"] == "lift"), None),
            "product_words_compared": int(sum(r.get("words_compared", 0) for r in rows
                                              if r["leg"] in ("product", "value_class")
                                              and "words_compared" in r)),
        },
        "mutation_case": MUTATION_CASE,
        "race_case": RACE_CASE,
        "mutation_specialisation": {"codes": list(codes)},
        "references": {
            "1": "the array path under the driver's own loop",
            "2": "the certified singles: cylindrical m=0 constitutive at update_H, "
                 "cylindrical m=0 curl (scan then curl) at step_D",
            "3": "plan_step(fuse=True): the composition the composer installs today",
            "4": "plan_step(fuse=False): the same slots dispatched unfused",
        },
        "signature": {"constitutive_pointers": family.CONSTITUTIVE_POINTERS,
                      "constitutive_bindings": family.CONSTITUTIVE_BINDINGS,
                      "curl_bindings": family.CURL_BINDINGS,
                      "ceiling": MAX_BUFFER_BINDINGS, "headroom": family.HEADROOM},
        "product": {"family": family.FAMILY, "slot": family.SLOT,
                    "replaces": list(family.REPLACES), "seam": family.SEAM,
                    "installable": family.INSTALLABLE,
                    "hoists_the_withdraw": family.HOISTS_THE_WITHDRAW,
                    "carries_deposit_repair": family.CARRIES_DEPOSIT_REPAIR,
                    "absorb_row_today": metal_launch.FUSED_PAIR_ARMS.get(family.FAMILY),
                    "weld_owed": family.WELD_OWED,
                    "corpus_digest": family.corpus_digest()},
        "corpus": {"census": CENSUS, "seam_record": SEAM_RECORD, "cell_arms": list(CELL_ARMS)},
        "subnormal_policy": report,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "metal_frontend": metal_frontend_version(),
        "jsonl": str(jsonl),
        "what_this_does_not_license": (
            "installing the product (the composer holds no row for it, and with the row "
            "applied in-process refuses it by arbitration -- leg arbitration), flipping "
            "HOISTS_THE_WITHDRAW, any throughput claim (launch counts are not time), the "
            "composition the composer would install AFTER wiring (out of reach until the "
            "rows land), complex64 cylindrical storage or any |m| >= 1 row, the E->B "
            "cylindrical seam, a keep-resolved run, or any step past a row's own first "
            "banded step -- the byte claim is over the steps the flush precondition held "
            "for, and the intermediate-arithmetic census the Cartesian H->D gate carries is "
            "NOT run here (the stored census and the IEEE underflow flag are)"),
        "source_sha256": {
            "family": hashlib.sha256(Path(family.__file__).read_bytes()).hexdigest(),
            "gate": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    gate_provenance.stamp(result)  # bytes THIS process imported; see gate_provenance
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
                        encoding="utf-8")
    log(f"VERDICT {verdict} in {result['elapsed_seconds']:.2f}s; artifact {args.out}")
    if not all_passed:
        return 1
    return 0 if complete else EXIT_INCOMPLETE


if __name__ == "__main__":
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))
