"""Shared machinery for the two CUDA cylindrical H->D two-launch product gates.

THE TWO GATES THAT IMPORT THIS are ``gate_cuda_cylindrical_real_fused_hd_pair.py``
(real float32 storage, m = 0) and ``gate_cuda_cylindrical_fused_hd_pair.py``
(complex64 storage, any m). Both certify the SAME SHAPE -- launch 1 in the ``update_H``
slot computes the certified magnetic constitutive at the thread's own cell into
launch-local scratch and RECOMPUTES it at the one-cell backward radial neighbour to
form the pre-``cumsum`` increment of ``stepping.cylindrical_rderiv_prefix``;
``xp.cumsum`` runs untouched on the array path (it IS the oracle on this backend); the
bindings rotate; launch 2 is the certified cylindrical ``step_D`` curl with the prefix
handed in -- over two storage classes whose arithmetic differs in exactly the places
``lanes/cyl_round/cupy_probe/FINDINGS.md`` measured. Everything that does not depend on
the storage class lives here ONCE: the fixture builder, the five arrangements, the
lock-step driver walk, the launch counters, the increment/cumsum legs, the floors, the
halo ledger, the driver-level sync and withdraw legs, the compiler leg, the corpus lift
(parent and child), the arbitration leg and the runner. A second transcription of a
launch counter is a second place for a counter to stop counting.

WHAT A FAMILY SUPPLIES is an :class:`Adapter`: the module, the fixtures, how to seed a
volume of its storage, the certified single-slot launchers and the released neighbouring
pairs it is measured against, the armed mutations with their null controls, and the
two host legs whose text is the family's own (transcription, refusal).

THE EVIDENCE STANDARD is the brief's: bit identity as uint32 WORDS per COMPLETE driver
step against the array path (never ``allclose``; ``-0.0 == 0.0`` lies), every zero
beside a control that moves words, every count as N of D with D named, nothing
unmeasured credited.

NOTHING PASSES BY NOT EXECUTING. Every weld step advances THREE counters -- the family's
own tally (3 per run), the shipped compile memo's launch count (the two ``RawKernel``
launches) and a wrapper around ``cupy.cumsum`` (the one scan) -- and the product leg
requires all three to have moved on every case.

Rule 7: one flushed line per case; the artifact is rewritten after every leg.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import inspect
import json
import os
import platform
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# BY NAME, never by parents[N]: a moved harness resolving a wrong root measures nothing.
_REPO_API = str(next(parent for parent in Path(_HERE).parents
                     if (parent / "meep_gpu" / "cuda_kernels").is_dir()))
for _path in (_REPO_API, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    import cupy as cp
except ImportError:  # laptop: the host legs and the tests only
    cp = None

import gate_provenance  # noqa: E402
import probe_fused_kernel_bit_identity as probe  # noqa: E402

# THE SIBLING GATES, imported for the machinery every scratch-output weld gate shares.
import gate_cuda_offdiag_stencil_welds as scratch_gate  # noqa: E402
import gate_cuda_fused_hd_pair as hd_gate  # noqa: E402

from meep_gpu import stepping, withdraw_hoist  # noqa: E402
from meep_gpu.fastpath import (SYNC_PASS_OWNERS, SYNC_PATH_SLOTS,  # noqa: E402
                               SYNC_UPDATE_H_PASS)

log = probe.log
to_host = scratch_gate.to_host
words = scratch_gate.words
differing = scratch_gate.differing
MemoLaunchCounter = scratch_gate.MemoLaunchCounter
needle = scratch_gate.needle
Shim = hd_gate.Shim
STATE_NAMES: Tuple[str, ...] = scratch_gate.STATE_NAMES
IMMUTABLE_PML: Tuple[str, ...] = scratch_gate.IMMUTABLE_PML
STEP_PASSES: Tuple[str, ...] = scratch_gate.STEP_PASSES

#: Complete DRIVER STEPS a fixture runs: the budget every hand-CUDA record is cut at.
STEPS = 60
#: The budget the mutation and disarm legs run: long enough for a defect in the
#: recurrence to reach every volume, short enough to arm thirty of them.
MUTATION_STEPS = 12
#: The block sizes the schedule sweep runs.
BLOCK_SIZES: Tuple[int, ...] = (32, 64, 128, 256, 512, 1024)
#: The steps at which the sync leg synchronizes the magnetic fields.
SYNC_STEPS: Tuple[int, ...] = (3, 7)
#: The power-of-two seed scale the driver-level legs use (the Cartesian gate's own
#: argument: the solver is linear, so the scale moves exponents and no rounding).
SEED_SCALE_BITS = 80
#: How many clean complete steps a lifted corpus row must reach to count.
LIFT_CLEAN_STEP_FLOOR = 8

#: The board this round's two cells are read from, and the census it was cut over.
BOARD = "fusion_matrix_cuda_2026-09-06_hd"
CENSUS = "cuda_predicate_coverage_2026-09-06_hd"

#: The expansion probe artifact the complex family's licence is read from, per policy.
EXPANSION_PROBES: Dict[str, str] = {
    "keep": "parity/meep_gpu/results/unified_expansion_2026-08-27/keep/gate.json",
    "flush": "parity/meep_gpu/results/unified_expansion_2026-08-27/flush/gate.json",
}

#: The arrangements every product case drives in lockstep. ``weld`` is the subject.
MODES: Tuple[str, ...] = ("array", "singles", "composition_today", "unfused", "weld")

#: The volumes that must MOVE for a band-class case to be non-vacuous. Under ``flush``
#: a state seeded in the band drives the constitutive to ``(f + 0) - 0`` so H need
#: not move; D, fu_D and f_w_H are written unconditionally and must.
BAND_MUST_MOVE: Tuple[str, ...] = tuple(
    [f"D{axis}" for axis in "xyz"] + [f"fu_D{axis}" for axis in "xyz"]
    + [f"f_w_H{axis}" for axis in "xyz"])

#: The statements ``stepping.cylindrical_rderiv_prefix`` runs on the D side today
#: (stepping.py:1322-1333), COUNTED BY STATEMENT: the four elementwise passes and the
#: scan. Launch 1 absorbs the first four; the fifth stays. Recorded as a statement
#: count because CuPy's own elementwise kernels do not pass through the memo counter.
ARRAY_PREFIX_PASSES_BY_STATEMENT = 5


def case_rng(seed: int, label: str) -> "np.random.Generator":
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return np.random.default_rng(seed + int.from_bytes(digest[:4], "big"))


# ---------------------------------------------------------------------------
# The adapter: what a family tells this module about itself
# ---------------------------------------------------------------------------

class Adapter:
    """One family's answers to the questions the shared legs ask.

    Subclassed by each gate. Everything here is a HOOK with a documented meaning; the
    shared code never reaches into a family module except through these.
    """

    #: The family module (``meep_gpu.cuda_kernels.<...>``), imported lazily by the gate.
    family: Any = None
    #: ``True`` for complex64 storage: word views on every launch argument.
    complex_storage: bool = False
    #: The board cell, in the census's ``(update_H arm, step_D arm)`` spelling.
    cell_arms: Tuple[str, str] = ("", "")
    #: The synthetic fixtures, the ones the mutations score on, the reduced subset.
    specs: Tuple[Dict[str, Any], ...] = ()
    mutation_spec_labels: Tuple[str, ...] = ()
    reduced_labels: Tuple[str, ...] = ()
    value_classes: Tuple[str, ...] = ("uniform", "subnormal_band")
    #: Per-case RNG seed base.
    seed: int = 20260907
    #: The float32 subnormal policy this run installs (a name) and the expansion
    #: licence (complex only; ``None`` for the real family).
    policy: Optional[str] = None
    license: Optional[Dict[str, Any]] = None
    #: The arm launch 1 and 2 compile under (complex only; ``None`` for real).
    arm: Any = None
    #: ``{name: {"old", "new", "why", "live_on", ["also"], ["predicted_null"],
    #: ["options"]}}`` -- device-source rewrites. ``options`` overrides the compile
    #: options (the contraction-guard control).
    device_mutations: Dict[str, Dict[str, Any]] = {}
    #: The host-side defects (launcher, tables, order), applied through
    #: :meth:`weld_state_mutation` and the arrangement flags.
    host_mutations: Tuple[str, ...] = ()
    #: Host mutations that are predicted NULL, with the reason.
    host_predicted_null: Dict[str, str] = {}
    #: ``{name: (policies,)}`` -- a mutation (device or host) that is NULL BY
    #: CONSTRUCTION under the named policies and must be CAUGHT under the others; the
    #: reason travels in ``null_reasons[name]``. Policy-conditional defects are the
    #: flushed-underflow class, and a battery that averaged the two policies would
    #: report them as flaky rather than as what they are.
    null_under: Dict[str, Tuple[str, ...]] = {}
    null_reasons: Dict[str, str] = {}
    #: ``{name: (classes,)}`` -- the value classes a HOST mutation is scored on.
    host_mutation_classes: Dict[str, Tuple[str, ...]] = {}
    #: The one armed edit required NOT to diverge.
    byte_neutral: Dict[str, str] = {}
    #: The rows the wiring change adds to ``fused_pairs`` -- used by the arbitration
    #: leg to measure the composer WITH the product in the tables, in-process only.
    wiring_arms_row: Tuple[str, str] = ("", "")

    # -- fixtures ------------------------------------------------------------
    def build(self, spec: Mapping[str, Any], value_class: str, rng):
        """``(fields, grid, pml, dtdx)`` on the device, seeded for this class."""
        from meep_gpu.fields import Fields  # noqa: PLC0415
        from meep_gpu.grid import Grid  # noqa: PLC0415
        from meep_gpu.pml import PML  # noqa: PLC0415

        shape = tuple(int(n) for n in spec["shape"])
        grid = Grid(resolution=1.0,
                    cell_size=(float(shape[0]), 0.0, float(shape[2])),
                    cylindrical=True, m=int(spec.get("m", 0)),
                    boundaries={"z": spec["z_kind"]},
                    accurate_fields_near_cylorigin=bool(spec.get("accurate", False)),
                    courant=float(spec.get("courant", 0.35)), xp=cp)
        if tuple(grid.shape) != shape:
            raise SystemExit(f"grid built {tuple(grid.shape)}, case asked for {shape}")
        fields = Fields(grid=grid, force_complex_fields=self.complex_storage)
        fields.enable_field_storage()
        fields.enable_pml_storage()
        thickness = {"x": (0, max(2, shape[0] // 4)), "z": max(2, shape[2] // 4)}
        if spec.get("thin_absorber"):
            thickness = {"x": (0, 2), "z": 2}
        if shape[2] < 4:
            # A ONE-COLUMN ROW absorbs on r only: no layer fits a one-cell z axis, and
            # the one-column corpus rows carry exactly that.
            thickness = {"x": thickness["x"]}
        pml = PML(grid=grid, thickness=thickness)
        self.seed_state(fields, grid, value_class, rng)
        return fields, grid, pml, float(grid.dt / grid.dx)

    def seed_state(self, fields, grid, value_class: str, rng) -> None:
        """Seed every stored volume. Complex planes are ASSIGNED, never ``re + 1j*im``
        (``1j * im`` carries ``0.0 * im`` and destroys every negative zero)."""
        shape = tuple(int(n) for n in grid.shape)
        for name in STATE_NAMES:
            array = getattr(fields, name, None)
            if array is None:
                continue
            if not self.complex_storage:
                host = self.real_plane(name, shape, value_class, rng)
            else:
                host = np.empty(shape, dtype=np.complex64)
                host.real = self.real_plane(name, shape, value_class, rng)
                host.imag = self.real_plane(name + "/imag", shape, value_class, rng)
            array[...] = cp.asarray(np.ascontiguousarray(host))
        self.plant(fields, grid, value_class)

    @staticmethod
    def real_plane(name: str, shape: Tuple[int, int, int], value_class: str,
                   rng) -> np.ndarray:
        if value_class in ("uniform", "planted"):
            return rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
        if value_class == "subnormal_band":
            return probe.subnormal_band_hosts((name,), shape, rng)[name]
        if value_class == "signed_zero":
            # HALF THE CELLS OF EVERY REAL PLANE ARE EXACT ZEROS OF RANDOM SIGN (the
            # imaginary planes stay uniform): the class on which the zero cross
            # terms, the divide's zero-valued terms and the sum's sign are decided.
            if name.endswith("/imag"):
                return rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
            values = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
            zeroed = rng.integers(0, 2, size=shape) == 0
            signs = np.where(rng.integers(0, 2, size=shape) == 0, -0.0, 0.0).astype(np.float32)
            return np.where(zeroed, signs, values).astype(np.float32)
        raise ValueError(f"value class {value_class!r} is not one this gate seeds")

    def plant(self, fields, grid, value_class: str) -> None:
        """A family-specific plant on top of the seed (the complex row-0 class)."""

    # -- the product -----------------------------------------------------------
    def predicate(self, fields, pml, grid, sources) -> Tuple[bool, str]:
        raise NotImplementedError

    def resolve(self, fields, grid, pml, **overrides) -> Dict[str, Any]:
        raise NotImplementedError

    def launch1(self, fields, state, kernel=None, threads: Optional[int] = None):
        raise NotImplementedError

    def scan_rotate(self, fields, state, rotate: bool = True):
        raise NotImplementedError

    def launch2(self, fields, state):
        raise NotImplementedError

    def weld_state_mutation(self, name: str, fields, grid, pml, state) -> Dict[str, Any]:
        """Apply one HOST mutation to a resolved state; return what it changed."""
        raise NotImplementedError

    def kernel_source(self) -> str:
        raise NotImplementedError

    def kernel_name(self) -> str:
        return self.family.KERNEL_NAME

    def compile_options(self) -> Tuple[str, ...]:
        return tuple(self.family._COMPILE_OPTIONS)  # noqa: SLF001

    def compile(self, source: str, options: Optional[Tuple[str, ...]] = None):
        kernel = cp.RawKernel(source, self.kernel_name(),
                              options=self.compile_options() if options is None
                              else tuple(options))
        kernel.compile()
        return kernel

    def source_digest(self) -> str:
        return self.family.source_digest()

    def prefix_source_volume(self, fields):
        return getattr(fields, self.family.PREFIX_SOURCE)

    def word_view(self, array):
        if not self.complex_storage:
            return array
        from meep_gpu.cuda_kernels import cylindrical_complex_kernels  # noqa: PLC0415
        return cylindrical_complex_kernels.word_view(array)

    # -- the references ----------------------------------------------------------
    def singles(self, fields, grid, pml, dtdx: float) -> Dict[str, Callable[[], None]]:
        """The certified single-slot launchers, one per slot of the slot path."""
        raise NotImplementedError

    def released_pairs(self, fields, grid, pml, dtdx: float) -> Dict[str, Callable[[], Dict]]:
        """The composition the composer installs on these rows today: the released
        cylindrical B->H pair (at ``step_B``) and D->E pair (at ``step_D``)."""
        raise NotImplementedError

    def released_pair_replaces(self) -> Dict[str, Tuple[str, ...]]:
        raise NotImplementedError

    # -- the driver-level legs -----------------------------------------------------
    def driver_kwargs(self) -> Dict[str, Any]:
        """Extra ``FdtdDriver`` keywords: the storage and the azimuthal order."""
        return {"force_complex_fields": self.complex_storage, "m": 0}

    def driver_source(self, integrated: bool) -> Dict[str, Any]:
        return {"component": "Ez", "center": (0.0, 0.0, 0.0), "size": (0.0, 0.0, 0.0),
                "frequency": 1.0,
                "source_type": "continuous" if integrated else "gaussian",
                **({"is_integrated": True} if integrated else {"fwidth": 0.2})}

    # -- the family's own host legs -----------------------------------------------
    def leg_transcription(self) -> Dict[str, Any]:
        raise NotImplementedError

    def leg_refusal(self) -> Dict[str, Any]:
        raise NotImplementedError

    def leg_spelling(self) -> Dict[str, Any]:
        """The divide (and multiply) spelling legs on standalone kernels."""
        raise NotImplementedError

    def wiring_product_row(self) -> Dict[str, Any]:
        """The ``FUSED_PRODUCTS`` row the wiring change adds, for the in-process
        arbitration measurement."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def state_of(fields) -> Dict[str, Any]:
    return {name: getattr(fields, name) for name in STATE_NAMES
            if getattr(fields, name, None) is not None}


def frozen(fields) -> Dict[str, np.ndarray]:
    return {name: to_host(value).copy() for name, value in state_of(fields).items()}


def compare(left: Mapping[str, np.ndarray],
            right: Mapping[str, np.ndarray]) -> Dict[str, int]:
    if set(left) != set(right):
        return {"__volume_set__": len(set(left) ^ set(right))}
    return {name: n for name in sorted(left)
            if (n := differing(left[name], right[name]))}


def word_total(snapshot: Mapping[str, np.ndarray]) -> int:
    return int(sum(words(value).size for value in snapshot.values()))


def read_only_snapshot(pml) -> Dict[str, np.ndarray]:
    return {f"pml.{name}": to_host(array).copy()
            for name in IMMUTABLE_PML
            if (array := getattr(pml, name, None)) is not None}


def read_only_drift(before: Mapping[str, np.ndarray], pml) -> List[str]:
    after = read_only_snapshot(pml)
    return sorted(name for name in before
                  if name in after and differing(before[name], after[name]))


def float_words(host: np.ndarray) -> np.ndarray:
    """The float32 words of a real or complex host array, for the census."""
    host = np.ascontiguousarray(host)
    if host.dtype == np.complex64:
        host = host.view(np.float32)
    return host.ravel()


def operand_census(snapshot: Mapping[str, np.ndarray]) -> Dict[str, int]:
    return probe.operand_census({name: float_words(value) for name, value in snapshot.items()})


def subnormal_words(host: np.ndarray) -> int:
    word = np.frombuffer(np.ascontiguousarray(float_words(host), dtype=np.float32).tobytes(),
                         dtype=np.uint32)
    return int(np.count_nonzero(((word & 0x7F800000) == 0) & ((word & 0x7FFFFF) != 0)))


class CumsumCounter:
    """Count every ``cupy.cumsum`` call: the ONE scan launch the memo cannot see.

    ``stepping`` and both launchers call ``xp.cumsum`` with ``xp`` the module, so a
    module attribute is the seam. Restored on exit.
    """

    def __init__(self) -> None:
        self.calls = 0
        self._saved = None

    def __enter__(self) -> "CumsumCounter":
        self._saved = cp.cumsum
        counter = self

        def counted(*args: Any, **kwargs: Any) -> Any:
            counter.calls += 1
            return counter._saved(*args, **kwargs)

        cp.cumsum = counted
        return self

    def __exit__(self, *exc: Any) -> None:
        cp.cumsum = self._saved


def run_pass(name: str, fields, pml) -> None:
    scratch_gate.run_pass(name, fields, pml)


# ---------------------------------------------------------------------------
# The arrangements
# ---------------------------------------------------------------------------

class Arrangement:
    __slots__ = ("name", "fields", "grid", "pml", "dtdx", "step", "launches",
                 "state", "kernel_launches", "cumsum_calls")

    def __init__(self, name: str, fields, grid, pml, dtdx: float) -> None:
        self.name, self.fields, self.grid, self.pml, self.dtdx = name, fields, grid, pml, dtdx
        self.launches = 0
        self.kernel_launches = 0
        self.cumsum_calls = 0
        self.state: Optional[Dict[str, Any]] = None
        self.step: Callable[[], None] = lambda: None


def make_arrangement(adapter: Adapter, mode: str, fields, grid, pml, dtdx: float, *,
                     kernel: Optional[Any] = None, threads: Optional[int] = None,
                     rotate: bool = True, order: str = "shipped",
                     host_mutation: Optional[str] = None,
                     state_overrides: Optional[Dict[str, Any]] = None) -> Arrangement:
    """One of :data:`MODES` (plus ``weld_composed``), against an already-seeded engine.

    ``weld`` isolates the seam: every other pass runs on the array path, so a
    divergence there is this product's. ``weld_composed`` is what a composer that
    installed this product would build -- the weld plus the two released neighbours'
    OTHER halves as certified singles -- and exists so the launch-structure leg
    compares like with like.
    """
    arrangement = Arrangement(mode, fields, grid, pml, dtdx)
    family = adapter.family

    if mode == "array":
        def step() -> None:
            for name in STEP_PASSES:
                run_pass(name, fields, pml)
    elif mode in ("singles", "unfused"):
        singles = adapter.singles(fields, grid, pml, dtdx)
        dispatched = (("update_H", "step_D") if mode == "singles"
                      else ("step_B", "update_H", "step_D", "update_E"))

        def step() -> None:
            for name in STEP_PASSES:
                if name in dispatched:
                    singles[name]()
                    arrangement.launches += 1
                    arrangement.kernel_launches += 1
                else:
                    run_pass(name, fields, pml)
    elif mode == "composition_today":
        pairs = adapter.released_pairs(fields, grid, pml, dtdx)
        absorbed = {slot: set(passes) for slot, passes in
                    adapter.released_pair_replaces().items()}
        skipped = set().union(*absorbed.values()) - set(absorbed)

        def step() -> None:
            for name in STEP_PASSES:
                if name in pairs:
                    report = pairs[name]()
                    if not report.get("launched"):
                        raise SystemExit(
                            f"the released pair at {name} refused a fixture this "
                            f"product admits: {report.get('reason')}")
                    arrangement.launches += 1
                    arrangement.kernel_launches += 1
                elif name in skipped:
                    continue          # absorbed by the released pair's launch
                else:
                    run_pass(name, fields, pml)
    elif mode in ("weld", "weld_composed"):
        overrides = dict(state_overrides or {})
        state = adapter.resolve(fields, grid, pml, **overrides)
        if host_mutation is not None:
            state["host_mutation"] = adapter.weld_state_mutation(
                host_mutation, fields, grid, pml, state)
        arrangement.state = state
        neighbours = (adapter.singles(fields, grid, pml, dtdx)
                      if mode == "weld_composed" else None)

        def weld_run() -> None:
            family.assert_bindings_are_disjoint(fields, state)
            if order == "curl_first":
                # THE SUB-STEP ORDER INVERTED: the certified curl runs on the
                # PRE-update H and a prefix of the pre-update Hy, then the
                # constitutive. An armed host mutation, never a shipped path.
                from meep_gpu.cuda_kernels.cylindrical_prefix import (  # noqa: PLC0415
                    cylindrical_prefix)
                state["prefix"][...] = cylindrical_prefix(fields, "step_D")
                adapter.launch2(fields, state)
                adapter.launch1(fields, state, kernel, threads)
                adapter.scan_rotate(fields, state, rotate)
                return
            adapter.launch1(fields, state, kernel, threads)
            if host_mutation == "stale_prefix":
                # The scan replaced by a prefix of the STORED pre-launch Hy: the
                # curl differences a prefix one sub-step behind.
                from meep_gpu.cuda_kernels.cylindrical_prefix import (  # noqa: PLC0415
                    cylindrical_prefix)
                stale = cylindrical_prefix(fields, "step_D")
                adapter.scan_rotate(fields, state, rotate)
                state["prefix"][...] = stale
            else:
                adapter.scan_rotate(fields, state, rotate)
            adapter.launch2(fields, state)

        def step() -> None:
            for name in STEP_PASSES:
                if name == "step_D":
                    continue           # absorbed by the launches at update_H
                if name != "update_H":
                    if neighbours is not None and name in ("step_B", "update_E"):
                        neighbours[name]()
                        arrangement.launches += 1
                        arrangement.kernel_launches += 1
                    else:
                        run_pass(name, fields, pml)
                    continue
                before = state["launches"]
                weld_run()
                arrangement.launches += state["launches"] - before
                arrangement.kernel_launches += 2
                arrangement.cumsum_calls += 1
    else:
        raise ValueError(f"unknown arrangement {mode!r}")

    arrangement.step = step  # type: ignore[assignment]
    return arrangement


def drive(adapter: Adapter, spec: Mapping[str, Any], value_class: str, steps: int, *,
          modes: Sequence[str] = MODES, kernel: Optional[Any] = None,
          rotate: bool = True, threads: Optional[int] = None, order: str = "shipped",
          host_mutation: Optional[str] = None,
          state_overrides: Optional[Dict[str, Any]] = None,
          progress: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """Step every arrangement side by side from ONE seed, compared per COMPLETE step.

    Against the ARRAY PATH for every other arrangement; the record carries the first
    divergence per arrangement rather than stopping the others. The weld's three
    counters are recorded and the case is non-vacuous only if all three advanced.
    """
    label = f"{spec['label']}/{value_class}"
    engines: Dict[str, Arrangement] = {}
    for mode in modes:
        fields, grid, pml, dtdx = adapter.build(spec, value_class,
                                                case_rng(adapter.seed, label))
        weld = mode == "weld"
        engines[mode] = make_arrangement(
            adapter, mode, fields, grid, pml, dtdx,
            kernel=kernel if weld else None, rotate=rotate if weld else True,
            threads=threads if weld else None, order=order if weld else "shipped",
            host_mutation=host_mutation if weld else None,
            state_overrides=state_overrides if weld else None)
    reference = engines[modes[0]]
    seeded = frozen(reference.fields)
    for mode in modes[1:]:
        mismatch = compare(seeded, frozen(engines[mode].fields))
        if mismatch:
            raise SystemExit(f"{label}: {mode} was not seeded identically: {mismatch}")
    read_only = read_only_snapshot(reference.pml)
    census = operand_census(seeded)
    first: Dict[str, Optional[Dict[str, Any]]] = {mode: None for mode in modes[1:]}
    compared = 0
    refused: Optional[str] = None
    started = time.perf_counter()
    with MemoLaunchCounter() as counter, CumsumCounter() as scans:
        for step in range(steps):
            try:
                for mode in modes:
                    engines[mode].step()
            except ValueError as error:
                # A REFUSAL RAISED BY THE LAUNCHER (the aliasing assert) is a
                # measurement, recorded as such; the case cannot continue.
                refused = repr(error)[:400]
                break
            cp.cuda.runtime.deviceSynchronize()
            state = {mode: frozen(engines[mode].fields) for mode in modes}
            compared = step + 1
            for mode in modes[1:]:
                if first[mode] is not None:
                    continue
                moved = compare(state[modes[0]], state[mode])
                if moved:
                    first[mode] = {"step": step + 1, "volumes": moved}
            if all(first[mode] is not None for mode in modes[1:]):
                break
            if progress is not None and (step + 1) % 10 == 0:
                progress(f"{label} step {step + 1}/{steps}")
        memo_named, memo_total = counter.named(), counter.total
        cumsum_total = scans.calls
    final = frozen(engines[modes[-1]].fields)
    array_final = frozen(engines[modes[0]].fields)
    must_move = STATE_NAMES if value_class != "subnormal_band" else BAND_MUST_MOVE
    # A VOLUME THE ARRAY PATH ITSELF LEAVES UNTOUCHED IS NOT A VACUITY. On a one-column
    # row (nz = 1, phi = 1) Dx's two curl terms are self-differences on one-cell axes
    # and fu_Dx/fu_Bx never move on ANY arrangement; the floor is that the product's
    # OWN outputs moved. Both sets are recorded.
    array_unmoved = sorted(name for name in must_move
                           if name in seeded and not differing(seeded[name], array_final[name]))
    unmoved = sorted(name for name in must_move
                     if name in seeded and name not in array_unmoved
                     and not differing(seeded[name], final[name]))
    product_outputs = ("Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz", "Dz", "fu_Dz")
    product_outputs_unmoved = sorted(name for name in product_outputs
                                     if name in seeded and not differing(seeded[name], final[name]))
    weld_engine = engines.get("weld")
    weld_counters = None
    if weld_engine is not None and weld_engine.state is not None:
        weld_counters = {
            "family_tally": int(weld_engine.state["launches"]),
            "family_launches_per_run": adapter.family.LAUNCHES_PER_RUN,
            "raw_kernel_launches_expected": 2 * compared,
            "cumsum_calls_expected": compared,
            "family_cumsum_calls": int(weld_engine.state["cumsum_calls"]),
            "family_curl_launches": int(weld_engine.state["curl_launches"]),
            "all_three_advanced": bool(
                compared > 0 and weld_engine.state["launches"]
                == adapter.family.LAUNCHES_PER_RUN * compared
                and weld_engine.state["cumsum_calls"] == compared
                and weld_engine.state["curl_launches"] == compared),
        }
    identical = {mode: first[mode] is None for mode in modes[1:]}
    return {
        "label": label, "spec": spec["label"], "value_class": value_class,
        "m": int(spec.get("m", 0)), "shape": [int(n) for n in reference.grid.shape],
        "steps_requested": steps, "steps_compared": compared,
        "threads": int(threads or adapter.family._FUSED_THREADS),  # noqa: SLF001
        "rotated": bool(rotate), "order": order, "host_mutation": host_mutation,
        "refused": refused,
        "identical": identical, "first_divergence": {m: first[m] for m in modes[1:]},
        "words_compared_per_step": word_total(seeded),
        "words_compared": word_total(seeded) * compared * max(len(modes) - 1, 1),
        "launches": {mode: engines[mode].launches for mode in modes},
        "kernel_launches_by_arrangement": {mode: engines[mode].kernel_launches
                                           for mode in modes},
        "memo_launches": {"named": memo_named, "total": memo_total},
        "cumsum_calls_total_all_arrangements": cumsum_total,
        "weld_counters": weld_counters,
        "read_only_drift": read_only_drift(read_only, reference.pml),
        "operand_census": census,
        "unmoved_volumes": unmoved, "array_path_unmoved_volumes": array_unmoved,
        "product_outputs_unmoved": product_outputs_unmoved,
        "non_vacuous": not unmoved and not product_outputs_unmoved,
        "seconds": round(time.perf_counter() - started, 2),
        "passed": (refused is None and not unmoved and not product_outputs_unmoved
                   and not read_only_drift(read_only, reference.pml)
                   and all(identical.values()) and compared == steps
                   and (weld_counters is None or weld_counters["all_three_advanced"])),
    }


# ---------------------------------------------------------------------------
# Host leg: the driver's order, read from the driver
# ---------------------------------------------------------------------------

def leg_driver_order(adapter: Adapter) -> Dict[str, Any]:
    """``REPLACES`` is the driver's two ADJACENT consults; the seam holds one pass;
    ``step_D`` is outside the sync window. All read off the tree with ``ast``."""
    from meep_gpu import driver as driver_module  # noqa: PLC0415

    family = adapter.family
    source = textwrap.dedent(inspect.getsource(driver_module.FdtdDriver.step))
    tree = ast.parse(source)
    called: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", None)
            if name in STEP_PASSES and name not in called:
                called.append(name)
    order = [name for name in STEP_PASSES if name in called]
    replaces = list(family.REPLACES)
    start, end = order.index(replaces[0]), order.index(replaces[-1])
    span = order[start:end + 1]
    body = ast.parse(source).body[0]
    statements = list(ast.walk(body))
    consults = [node for node in statements
                if isinstance(node, ast.Call)
                and getattr(node.func, "attr", None) == "dispatch"]
    consulted = [node.args[0].value for node in consults
                 if node.args and isinstance(node.args[0], ast.Constant)]
    between = [name for name in consulted
               if consulted.index("update_H") < consulted.index(name) < consulted.index("step_D")]
    withdraw_calls = sum(
        1 for node in statements
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Call)
        and getattr(node.func.func, "id", None) == "getattr")
    # THE ORACLE'S OWN TEXT: the prefix runs over post-update_H Hy (stepping.py:447)
    # and feeds Dz only (:461-462) -- read off stepping rather than trusted.
    step_d_source = inspect.getsource(stepping.step_D)
    record = {
        "driver_calls": called, "walked_order": order,
        "replaces": replaces, "driver_span": span,
        "replaces_is_the_whole_span": span == replaces,
        "adjacent": order.index("step_D") - order.index("update_H") == 1,
        "consulted_slots_in_order": consulted,
        "slots_consulted_between_the_two": between,
        "the_seam_holds_no_other_consult": between == [],
        "withdraw_getattr_call_sites_in_step": withdraw_calls,
        "seam_module": family.SEAM,
        "seam_module_is_the_withdraw_hoist": family.SEAM == withdraw_hoist.SEAM,
        "carries_deposit_repair": bool(family.CARRIES_DEPOSIT_REPAIR),
        "hoists_the_withdraw": bool(family.HOISTS_THE_WITHDRAW),
        "launches_per_run_declared": int(family.LAUNCHES_PER_RUN),
        "slot": family.SLOT,
        "sync_path_slots": list(SYNC_PATH_SLOTS),
        "sync_pass_owner": dict(SYNC_PASS_OWNERS),
        "step_D_is_outside_the_sync_window": "step_D" not in SYNC_PATH_SLOTS,
        "the_sync_consult_owner_is_this_products_leading_slot":
            SYNC_PASS_OWNERS.get(SYNC_UPDATE_H_PASS) == family.SLOT,
        "prefix_source_is_Hy_at_half_shift":
            (family.PREFIX_SOURCE, float(family.PREFIX_IR0_VALUE)) == ("Hy", 0.5),
        "stepping_step_D_prefixes_Hy_at_0_5":
            'cylindrical_rderiv_prefix(grid.xp, magnetic["Hy"], 0.5' in step_d_source,
        "stepping_step_D_feeds_the_prefix_to_Dz_only":
            'if cylindrical and term.target == "Dz":' in step_d_source
            and "term_sources = prefixed" in step_d_source,
    }
    record["passed"] = bool(
        record["replaces_is_the_whole_span"] and record["adjacent"]
        and record["the_seam_holds_no_other_consult"]
        and record["seam_module_is_the_withdraw_hoist"]
        and not record["carries_deposit_repair"] and not record["hoists_the_withdraw"]
        and record["launches_per_run_declared"] == 3 and family.SLOT == "update_H"
        and record["step_D_is_outside_the_sync_window"]
        and record["the_sync_consult_owner_is_this_products_leading_slot"]
        and record["prefix_source_is_Hy_at_half_shift"]
        and record["stepping_step_D_prefixes_Hy_at_0_5"]
        and record["stepping_step_D_feeds_the_prefix_to_Dz_only"]
        and withdraw_calls >= 2)
    return record


def encodes_ascii(source: str) -> bool:
    try:
        source.encode("ascii")
    except UnicodeEncodeError:
        return False
    return True


def mutation_needles_resolve(adapter: Adapter, source: str) -> Dict[str, Any]:
    """Every armed device mutation's anchor appears exactly once in the shipped text."""
    counts: Dict[str, int] = {}
    for name, spec in adapter.device_mutations.items():
        if spec.get("old") is None:
            counts[name] = 1     # a compile-option control, no text anchor
            continue
        counts[name] = source.count(spec["old"])
        for also_old, _new in spec.get("also", ()):
            counts[name] = min(counts[name], source.count(also_old))
    return {"counts": counts, "passed": all(count == 1 for count in counts.values())}


# ---------------------------------------------------------------------------
# Device leg: the increment stage and the scan order, on the product's own launch 1
# ---------------------------------------------------------------------------

_SERIAL_SCAN_REAL = r'''
extern "C" __global__ void column_serial_scan(
    float* __restrict__ out, const float* __restrict__ inc,
    int nx, int ny, int nz, int n_cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cols) return;
    int nyz = ny * nz;
    int k = idx % nz;
    int j = idx / nz;
    if (j >= ny) return;
    int base = j * nz + k;
    float acc = inc[base];
    out[base] = acc;
    for (int i = 1; i < nx; ++i) {
        int o = i * nyz + base;
        acc = acc + inc[o];
        out[o] = acc;
    }
}
'''

_SERIAL_SCAN_COMPLEX = r'''
extern "C" __global__ void column_serial_scan_complex(
    float* __restrict__ out, const float* __restrict__ inc,
    int nx, int ny, int nz, int n_cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cols) return;
    int nyz = ny * nz;
    int k = idx % nz;
    int j = idx / nz;
    if (j >= ny) return;
    int base = j * nz + k;
    float acc_re = inc[2 * base];
    float acc_im = inc[2 * base + 1];
    out[2 * base] = acc_re;
    out[2 * base + 1] = acc_im;
    for (int i = 1; i < nx; ++i) {
        int o = i * nyz + base;
        acc_re = acc_re + inc[2 * o];
        acc_im = acc_im + inc[2 * o + 1];
        out[2 * o] = acc_re;
        out[2 * o + 1] = acc_im;
    }
}
'''


def array_path_increment(source_new, ir0: float):
    """``stepping.cylindrical_rderiv_prefix``'s pooled branch, statement for statement,
    stopped before the scan. The row vectors are the SHIPPED builder's."""
    from meep_gpu.fields import StepScratch  # noqa: PLC0415

    xp = cp
    scratch = StepScratch(xp)
    real_dtype = source_new.real.dtype
    key = ("cyl_rderiv", int(source_new.shape[0]), float(ir0), real_dtype)
    weights, divisor = scratch.constant(
        key, lambda: stepping._cylindrical_rderiv_weights(  # noqa: SLF001
            xp, source_new.shape[0], ir0, real_dtype))
    weighted = scratch.take("cyl_weighted", source_new.shape, source_new.dtype)
    xp.multiply(source_new, weights, out=weighted)
    increment = scratch.take("cyl_increment", source_new.shape, source_new.dtype)
    increment[stepping._face(0, 0)] = 0  # noqa: SLF001
    xp.subtract(weighted[1:], weighted[:-1], out=increment[1:])
    increment[1:] /= divisor
    shipped = stepping.cylindrical_rderiv_prefix(xp, source_new, ir0,
                                                 scratch=StepScratch(xp))
    return increment.copy(), shipped.copy()


def serial_scan(inc_dev, complex_storage: bool):
    nx, ny, nz = inc_dev.shape
    kernel = cp.RawKernel(_SERIAL_SCAN_COMPLEX if complex_storage else _SERIAL_SCAN_REAL,
                          "column_serial_scan_complex" if complex_storage
                          else "column_serial_scan", options=("--fmad=false",))
    out = cp.empty_like(inc_dev)
    cols = ny * nz
    blocks = (cols + 127) // 128
    view = (lambda a: a.view(cp.float32)) if complex_storage else (lambda a: a)
    kernel((blocks,), (128,), (view(out), view(inc_dev), np.int32(nx), np.int32(ny),
                                np.int32(nz), np.int32(cols)))
    cp.cuda.runtime.deviceSynchronize()
    return out


def leg_increment_stage(adapter: Adapter, specs: Sequence[Mapping[str, Any]],
                        classes: Sequence[str]) -> Dict[str, Any]:
    """Launch 1's increment against the array path's own pre-``cumsum`` stage, and
    ``xp.cumsum`` of it against the SHIPPED prefix, on the product's own launch.

    The array path's stage is built from ``Hy_new`` -- the scratch launch 1 just
    wrote -- because that is the volume the shipped prefix runs over after
    ``update_H``. Floors: the increment is non-constant and finite, and its row 0 is
    an exact +0.0 on every word. The SCAN-ORDER floor is restated per case: a
    column-serial CUDA scan of the same increment must DIFFER from ``cupy.cumsum``
    and equal ``numpy.cumsum``, and ``cupy.cumsum`` twice must agree -- which is why
    the scan stays on the array path.
    """
    family = adapter.family
    cases: List[Dict[str, Any]] = []
    for spec in specs:
        for value_class in tuple(spec.get("classes", classes)):
            label = f"{spec['label']}/{value_class}"
            try:
                fields, grid, pml, _dtdx = adapter.build(spec, value_class,
                                                         case_rng(adapter.seed, label))
            except Exception as error:  # noqa: BLE001
                cases.append({"label": label, "error": repr(error)[:400], "words": 0,
                              "increment_vs_array_path_differing": 0,
                              "cumsum_of_increment_vs_shipped_prefix_differing": 0,
                              "scan_order": {"serial_cuda_vs_cupy_cumsum_differing": 0,
                                             "serial_cuda_vs_numpy_cumsum_differing": 0,
                                             "cupy_cumsum_twice_differing": 0},
                              "passed": False})
                continue
            state = adapter.resolve(fields, grid, pml)
            family.assert_bindings_are_disjoint(fields, state)
            adapter.launch1(fields, state)
            cp.cuda.runtime.deviceSynchronize()
            h_new = state["scratch"]["Hy"]
            increment_ap, shipped = array_path_increment(h_new, family.PREFIX_IR0_VALUE)
            fused = state["increment"]
            inc_words = differing(fused, increment_ap)
            xp_cumsum = cp.cumsum(fused, axis=0)
            cum_words = differing(xp_cumsum, shipped)
            fused_host = to_host(fused)
            fw = float_words(fused_host)
            row0 = float_words(fused_host[0])
            serial = serial_scan(fused, adapter.complex_storage)
            numpy_cumsum = np.cumsum(to_host(fused), axis=0, dtype=to_host(fused).dtype)
            entry = {
                "label": label, "shape": [int(n) for n in grid.shape],
                "words": int(fw.size),
                "increment_vs_array_path_differing": inc_words,
                "cumsum_of_increment_vs_shipped_prefix_differing": cum_words,
                "increment_nonzero_words": int(np.count_nonzero(
                    fw.view(np.uint32) & np.uint32(0x7FFFFFFF))),
                "increment_all_finite": bool(np.all(np.isfinite(fw))),
                "row0_all_exact_positive_zero": bool(np.all(row0.view(np.uint32) == 0)),
                "scan_order": {
                    "serial_cuda_vs_cupy_cumsum_differing": differing(serial, xp_cumsum),
                    "serial_cuda_vs_numpy_cumsum_differing": differing(serial, numpy_cumsum),
                    "cupy_cumsum_twice_differing": differing(cp.cumsum(fused, axis=0),
                                                             xp_cumsum),
                    "cupy_vs_numpy_cumsum_differing": differing(xp_cumsum, numpy_cumsum),
                },
            }
            entry["passed"] = bool(
                inc_words == 0 and cum_words == 0 and entry["increment_nonzero_words"] > 0
                and entry["increment_all_finite"] and entry["row0_all_exact_positive_zero"]
                and entry["scan_order"]["serial_cuda_vs_cupy_cumsum_differing"] > 0
                and entry["scan_order"]["serial_cuda_vs_numpy_cumsum_differing"] == 0
                and entry["scan_order"]["cupy_cumsum_twice_differing"] == 0)
            cases.append(entry)
            log(f"[increment_stage] {label:34s} inc={inc_words}/{fw.size} "
                f"cumsum={cum_words} serial_vs_cupy="
                f"{entry['scan_order']['serial_cuda_vs_cupy_cumsum_differing']} "
                f"serial_vs_numpy={entry['scan_order']['serial_cuda_vs_numpy_cumsum_differing']}")
    return {
        "cases": cases, "denominator": len(cases),
        "words": sum(c["words"] for c in cases),
        "increment_differing_total": sum(c["increment_vs_array_path_differing"] for c in cases),
        "cumsum_differing_total": sum(
            c["cumsum_of_increment_vs_shipped_prefix_differing"] for c in cases),
        "serial_scan_differs_from_cupy_on_every_case": all(
            c["scan_order"]["serial_cuda_vs_cupy_cumsum_differing"] > 0 for c in cases),
        "reading": ("a column-serial CUDA scan is NOT bit-equal to cupy.cumsum on the "
                    "product's own increments and IS bit-equal to numpy.cumsum: the "
                    "oracle's summation order is CuPy's own, so the scan stays on the "
                    "array path -- MEASURED HERE, per case, not inherited"),
        "passed": bool(cases) and all(c["passed"] for c in cases),
    }


# ---------------------------------------------------------------------------
# Device leg: the floors and the halo ledger, on the array path
# ---------------------------------------------------------------------------

def leg_floors(adapter: Adapter, specs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """The cylindrical gate's floors on the array path: the prefix is live, the axis
    rows move, the absorber deviates from the identity on both axes."""
    from meep_gpu.cuda_kernels.cylindrical_prefix import cylindrical_prefix  # noqa: PLC0415

    rows: List[Dict[str, Any]] = []
    for spec in specs:
        try:
            fields, grid, pml, _ = adapter.build(spec, "uniform",
                                                 case_rng(adapter.seed, f"floors/{spec['label']}"))
        except Exception as error:  # noqa: BLE001 - a fixture the engine refuses is a failed row
            rows.append({"spec": spec["label"], "error": repr(error)[:400], "meets_floors": False})
            continue
        before = frozen(fields)
        run_pass("update_H", fields, pml)
        prefix = cylindrical_prefix(fields, "step_D")
        host = float_words(to_host(prefix)).reshape(-1)
        rows_r = int(prefix.shape[0])
        host_rows = np.ascontiguousarray(to_host(prefix))
        diffs = float_words(host_rows[1:]) - float_words(host_rows[:-1])
        run_pass("step_D", fields, pml)
        after = frozen(fields)
        axis_before = float_words(before["Dy"][0]).view(np.uint32)
        axis_after = float_words(after["Dy"][0]).view(np.uint32)
        dz_before = float_words(before["Dz"][0]).view(np.uint32)
        dz_after = float_words(after["Dz"][0]).view(np.uint32)
        deviation = {}
        # The z floor applies only where the fixture absorbs on z (a one-column row
        # absorbs on r alone); the r floor always.
        absorbing_axes = ("x", "z") if int(grid.shape[2]) >= 4 else ("x",)
        for axis in absorbing_axes:
            dev = 0.0
            for stem in ("kms", "sinv"):
                values = to_host(getattr(pml, f"{stem}_{axis}")).astype(np.float64)
                dev = max(dev, float(np.max(np.abs(values - 1.0))))
            deviation[axis] = dev
        row = {
            "spec": spec["label"], "shape": [int(n) for n in grid.shape],
            "prefix_rows": rows_r,
            "prefix_distinct_radial_differences": int(np.count_nonzero(diffs)),
            "prefix_row0_all_exact_positive_zero": bool(
                np.all(float_words(host_rows[0]).view(np.uint32) == 0)),
            "axis_row_Dy_moved_words": int(np.count_nonzero(axis_before != axis_after)),
            "axis_row_Dz_moved_words": int(np.count_nonzero(dz_before != dz_after)),
            "absorber_deviation_from_identity": deviation,
        }
        row["meets_floors"] = bool(
            row["prefix_distinct_radial_differences"] > 0
            and row["prefix_row0_all_exact_positive_zero"]
            and (row["axis_row_Dy_moved_words"] + row["axis_row_Dz_moved_words"]) > 0
            and min(deviation.values()) > 0.0)
        rows.append(row)
        del host
    return {"rows": rows, "denominator": len(rows),
            "passed": bool(rows) and all(r["meets_floors"] for r in rows)}


def leg_halo_ledger(adapter: Adapter, specs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Of the increment's backward radial taps, how many land on a cell ``update_H``
    MOVED -- the count of taps whose RECOMPUTED value differs from the stored word a
    stale-read weld would have used. Near zero would mean the identity licenses
    nothing about the recompute. Measured on the ARRAY PATH."""
    rows: List[Dict[str, Any]] = []
    for spec in specs:
        try:
            fields, grid, pml, _ = adapter.build(spec, "uniform",
                                                 case_rng(adapter.seed, f"halo/{spec['label']}"))
        except Exception as error:  # noqa: BLE001
            rows.append({"spec": spec["label"], "error": repr(error)[:400],
                         "taps_that_read_a_neighbour": 0, "taps_whose_recomputed_value_differs": 0})
            continue
        before = to_host(fields.Hy).copy()
        run_pass("update_H", fields, pml)
        after = to_host(fields.Hy).copy()
        moved = float_words(before[:-1]).view(np.uint32) != float_words(after[:-1]).view(np.uint32)
        taps = int(moved.size)
        rows.append({"spec": spec["label"], "shape": [int(n) for n in grid.shape],
                     "taps_that_read_a_neighbour": taps,
                     "taps_whose_recomputed_value_differs": int(moved.sum()),
                     "fraction": round(float(moved.sum()) / taps, 6) if taps else None,
                     "row0_taps_are_the_zero_row": True})
    numerator = sum(r["taps_whose_recomputed_value_differs"] for r in rows)
    denominator = sum(r["taps_that_read_a_neighbour"] for r in rows)
    return {"rows": rows, "taps_that_read_a_neighbour": denominator,
            "taps_whose_recomputed_value_differs": numerator,
            "errored": [r["spec"] for r in rows if r.get("error")],
            "fraction": round(numerator / denominator, 6) if denominator else None,
            "what_this_licenses": (
                "the halo recompute is LOAD-BEARING wherever this fraction is high: every "
                "such tap is a word an in-place weld would have read at a schedule-decided "
                "time, so the identity legs are evidence about THIS design"),
            "passed": bool(denominator > 0 and numerator == denominator
                           and not any(r.get("error") for r in rows))}


# ---------------------------------------------------------------------------
# Device leg: launch structure
# ---------------------------------------------------------------------------

def leg_launch_structure(adapter: Adapter, spec: Mapping[str, Any], steps: int) -> Dict[str, Any]:
    """Launches per step at the seam and over the whole step, all arrangements, three
    counters (arrangement tally, memo counter, cumsum counter).

    THE HONEST NUMBER IS REPORTED WHOLE. The D-side seam is ``update_H`` + the prefix's
    five statements + ``step_D`` = 7 today, of which the memo can see 2 (the two
    kernels) and the cumsum counter 1; the weld is 3 (two kernels + one scan). The
    array path's four elementwise prefix passes are CuPy's own kernels and are
    counted BY STATEMENT, never claimed as measured.
    """
    out: Dict[str, Any] = {"spec": spec["label"], "steps": steps, "modes": {}}
    # THE B-SIDE SCAN IS NOT THIS SEAM'S. ``step_B`` prefixes Ey at ir0 = 0.0 on every
    # arrangement (stepping.py:361), so a whole-step cumsum count is two on every one of
    # them; the B side's count is MEASURED here (the array pass alone under the counter)
    # and subtracted to give the D-side seam's own.
    fields, grid, pml, dtdx = adapter.build(spec, "uniform", case_rng(adapter.seed, "launch"))
    run_pass("step_B", fields, pml)
    with CumsumCounter() as b_scans:
        for _ in range(steps):
            run_pass("step_B", fields, pml)
        cp.cuda.runtime.deviceSynchronize()
    b_side_cumsums = b_scans.calls / steps
    out["b_side_cumsums_per_step_measured"] = b_side_cumsums
    for mode in MODES + ("weld_composed",):
        fields, grid, pml, dtdx = adapter.build(spec, "uniform", case_rng(adapter.seed, "launch"))
        arrangement = make_arrangement(adapter, mode, fields, grid, pml, dtdx)
        arrangement.step()      # warm the memo: an uncounted first compile
        cp.cuda.runtime.deviceSynchronize()
        arrangement.launches = 0
        arrangement.kernel_launches = 0
        arrangement.cumsum_calls = 0
        tally_before = (0 if arrangement.state is None
                        else int(arrangement.state["cumsum_calls"]))
        with MemoLaunchCounter() as counter, CumsumCounter() as scans:
            for _ in range(steps):
                arrangement.step()
            cp.cuda.runtime.deviceSynchronize()
            named, total, cumsums = counter.named(), counter.total, scans.calls
        out["modes"][mode] = {
            "arrangement_launches_per_step": arrangement.launches / steps,
            "arrangement_kernel_launches_per_step": arrangement.kernel_launches / steps,
            "memo_kernel_launches_per_step": total / steps,
            "memo_named": named,
            "cumsum_calls_per_step_whole_step": cumsums / steps,
            "cumsum_calls_per_step": cumsums / steps - b_side_cumsums,
            "family_cumsum_tally_per_step": (
                None if arrangement.state is None
                else (arrangement.state["cumsum_calls"] - tally_before) / steps),
            "kernel_counters_agree": arrangement.kernel_launches == total,
        }
    weld = out["modes"]["weld"]
    singles = out["modes"]["singles"]
    out["seam_launches"] = {
        "weld_kernels_measured": weld["memo_kernel_launches_per_step"],
        "weld_cumsum_measured": weld["cumsum_calls_per_step"],
        "weld_total": weld["memo_kernel_launches_per_step"] + weld["cumsum_calls_per_step"],
        "certified_singles_kernels_measured": singles["memo_kernel_launches_per_step"],
        "certified_singles_cumsum_measured": singles["cumsum_calls_per_step"],
        "certified_singles_array_prefix_passes_by_statement": ARRAY_PREFIX_PASSES_BY_STATEMENT,
        "certified_singles_total_by_statement":
            singles["memo_kernel_launches_per_step"] + ARRAY_PREFIX_PASSES_BY_STATEMENT,
        "note": ("the H_to_D seam ALONE. The weld is two kernels and one cupy.cumsum, "
                 "all three MEASURED; the singles are two kernels (measured) plus the "
                 "array path's prefix (four elementwise CuPy passes, by statement, plus "
                 "the same cumsum, measured). 7 -> 3 is a count, not a timing"),
    }
    today = out["modes"]["composition_today"]
    composed = out["modes"]["weld_composed"]
    out["whole_step"] = {
        "composition_today_kernels": today["memo_kernel_launches_per_step"],
        "composition_today_cumsums": today["cumsum_calls_per_step"],
        "weld_composed_kernels": composed["memo_kernel_launches_per_step"],
        "weld_composed_cumsums": composed["cumsum_calls_per_step"],
        "unfused_kernels": out["modes"]["unfused"]["memo_kernel_launches_per_step"],
        "unfused_cumsums": out["modes"]["unfused"]["cumsum_calls_per_step"],
        "what_this_says": (
            "over step_B .. update_E the composition that SHIPS on a cylindrical row is "
            "the two released cylindrical pairs (two kernels, each computing its own "
            "prefix on the array path), and the composition with this product installed "
            "is the step_B single, the two launches of this product, and the update_E "
            "single. The saving the product makes is the four elementwise prefix passes "
            "at the D-side seam; the arbitration algebra that decides who holds "
            "update_H is the INSTALLABLE_REASON's and is not flipped by this count"),
    }
    out["passed"] = bool(
        all(entry["kernel_counters_agree"] for entry in out["modes"].values())
        and b_side_cumsums == 1.0
        and weld["memo_kernel_launches_per_step"] == 2.0
        and weld["cumsum_calls_per_step"] == 1.0
        and weld["family_cumsum_tally_per_step"] == 1.0
        and singles["memo_kernel_launches_per_step"] == 2.0
        and singles["cumsum_calls_per_step"] == 1.0
        and out["modes"]["array"]["memo_kernel_launches_per_step"] == 0.0
        and out["modes"]["array"]["cumsum_calls_per_step"] == 1.0)
    return out


# ---------------------------------------------------------------------------
# Driver-level legs: the sync channel and the withdraw
# ---------------------------------------------------------------------------

def build_driver(adapter: Adapter, spec: Mapping[str, Any], *, integrated: bool = False,
                 seed: Optional[int] = None, scale_bits: int = SEED_SCALE_BITS):
    """One seeded cylindrical ``FdtdDriver`` on the device with one electric source."""
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415

    shape = tuple(int(n) for n in spec["shape"])
    driver = FdtdDriver(cell_size=(float(shape[0]), 0.0, float(shape[2])),
                        resolution=1.0, courant=float(spec.get("courant", 0.35)),
                        boundaries={"z": spec["z_kind"]}, cylindrical=True,
                        prefer_gpu=True, gpu_id=0, **adapter.driver_kwargs())
    driver.setup_pml({"x": (0, max(2, shape[0] // 4)), "z": max(2, shape[2] // 4)}
                     if shape[2] >= 4 else {"x": (0, max(2, shape[0] // 4))})
    driver.add_source(adapter.driver_source(integrated))
    rng = np.random.default_rng(adapter.seed if seed is None else seed)
    scale = np.float32(2.0) ** int(scale_bits)
    for name in STATE_NAMES:
        array = getattr(driver.fields, name, None)
        if array is None:
            continue
        if adapter.complex_storage:
            host = np.empty(array.shape, dtype=np.complex64)
            host.real = (0.37 * rng.standard_normal(array.shape)).astype(np.float32) * scale
            host.imag = (0.37 * rng.standard_normal(array.shape)).astype(np.float32) * scale
        else:
            host = (0.37 * rng.standard_normal(array.shape)).astype(np.float32) * scale
        array[...] = cp.asarray(np.ascontiguousarray(host))
    driver.invalidate_fast_path()
    return driver


def weld_plan_for(adapter: Adapter, driver, *, hoisted: bool = False,
                  placement: str = withdraw_hoist.BEFORE_UPDATE_H):
    """``(plans, state)`` -- the product installed at ``update_H`` with ``step_D`` absorbed."""
    fields, pml, grid = driver.fields, driver.pml, driver.grid
    state = adapter.resolve(fields, grid, pml)
    counters = {"withdrawn": 0, "runs": 0}

    def run() -> None:
        if hoisted:
            counters["withdrawn"] += withdraw_hoist.hoist(
                fields, driver._sources, span=adapter.family.REPLACES,  # noqa: SLF001
                placement=placement)
        adapter.family.assert_bindings_are_disjoint(fields, state)
        adapter.launch1(fields, state)
        adapter.scan_rotate(fields, state, True)
        adapter.launch2(fields, state)
        counters["runs"] += 1

    return {"update_H": run, "step_D": lambda: None}, state, counters


def driver_state(driver) -> Dict[str, np.ndarray]:
    return {name: to_host(array).copy()
            for name in STATE_NAMES
            if (array := getattr(driver.fields, name, None)) is not None}


def leg_sync(adapter: Adapter, spec: Mapping[str, Any], steps: int) -> Dict[str, Any]:
    """The magnetic half-step's channel with the product FORCE-INSTALLED.

    ``synchronize_magnetic_fields`` repeats the magnetic half of ``step`` and UNDOES
    it from a backup of magnetic names only. A product spanning ``update_H`` and
    ``step_D`` that answered the ``update_H_synchronize`` consult would advance D and
    fu_D inside a window nothing restores. The hazard arm MUST diverge in D against a
    run that never synchronizes; the guarded arm (declining by the containment rule)
    MUST NOT. ``field_energy_in_box`` raises on a Dcyl run (its reducers are
    Cartesian), so the channel is driven through ``synchronize_magnetic_fields`` and
    ``restore_magnetic_fields`` directly -- the consult sits in the first.
    """
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415

    facts: Dict[str, Any] = {"spec": spec["label"], "steps": steps,
                             "sync_steps": list(SYNC_STEPS),
                             "sync_fields": list(FdtdDriver._SYNC_FIELDS),  # noqa: SLF001
                             "sync_auxiliary": list(FdtdDriver._SYNC_AUXILIARY)}  # noqa: SLF001
    facts["D_is_in_neither_backup_list"] = not any(
        name.startswith(("D", "fu_D", "f_cond_D"))
        for name in facts["sync_fields"] + facts["sync_auxiliary"])
    arms_out: Dict[str, Any] = {}
    baselines: Dict[str, Dict[str, np.ndarray]] = {}
    for arm, answers, synchronizes in (("no_sync", False, False),
                                       ("guarded", False, True),
                                       ("hazard", True, True)):
        driver = build_driver(adapter, spec)
        plans, _state, counters = weld_plan_for(adapter, driver)
        driver._fast_path = Shim(plans, answers_the_sync_consult=answers)  # noqa: SLF001
        driver._fast_path_stale = False  # noqa: SLF001
        for step in range(steps):
            driver.step()
            if synchronizes and (step + 1) in SYNC_STEPS:
                driver.synchronize_magnetic_fields()
                driver.restore_magnetic_fields()
        cp.cuda.runtime.deviceSynchronize()
        baselines[arm] = driver_state(driver)
        arms_out[arm] = {"runs": counters["runs"],
                         "consults": dict(driver._fast_path.calls)}  # noqa: SLF001
        driver.close()
    electric = tuple(f"{stem}{axis}" for stem in ("D", "fu_D") for axis in "xyz")
    for arm in ("guarded", "hazard"):
        moved = compare(baselines["no_sync"], baselines[arm])
        arms_out[arm]["differing_volumes"] = moved
        arms_out[arm]["differing_electric_volumes"] = {n: c for n, c in moved.items()
                                                       if n in electric}
    facts["arms"] = arms_out
    facts["the_guarded_arm_is_identical"] = not arms_out["guarded"]["differing_volumes"]
    facts["the_hazard_arm_diverges_in_D"] = bool(arms_out["hazard"]["differing_electric_volumes"])
    facts["the_hazard_arm_answered_the_consult"] = arms_out["hazard"]["consults"].get("update_H", 0) \
        > arms_out["guarded"]["consults"].get("update_H", 0)
    facts["passed"] = bool(facts["D_is_in_neither_backup_list"]
                           and facts["the_guarded_arm_is_identical"]
                           and facts["the_hazard_arm_diverges_in_D"]
                           and facts["the_hazard_arm_answered_the_consult"])
    return facts


def leg_withdraw(adapter: Adapter, spec: Mapping[str, Any], steps: int) -> Dict[str, Any]:
    """The seam's one pass on a device: a standing integrated electric withdraw (the
    configuration the predicate REFUSES today), driven with the hoist wired. The hoisted
    arrangement must be identical to the array path stepped beside it; the un-hoisted
    launch and the ``after_step_D`` placement must both diverge. Unscaled seed, for the
    Cartesian gate's own reason (a 2^80 state swallows the dipole's subtraction)."""
    probe_driver = build_driver(adapter, spec, integrated=True, scale_bits=0)
    sources = tuple(probe_driver._sources)  # noqa: SLF001
    standing = withdraw_hoist.standing_withdraws(sources)
    covered, reason = adapter.predicate(probe_driver.fields, probe_driver.pml,
                                        probe_driver.grid, sources)
    hoistable, hoist_reasons = withdraw_hoist.hoistable(
        probe_driver.fields, sources, span=adapter.family.REPLACES)
    probe_driver.close()
    arms_out: Dict[str, Any] = {}
    for arm in ("hoisted", "not_hoisted", "after_step_D"):
        reference = build_driver(adapter, spec, integrated=True, scale_bits=0)
        driver = build_driver(adapter, spec, integrated=True, scale_bits=0)
        if arm == "after_step_D":
            plans, _state, counters = weld_plan_for(adapter, driver)
            electric = tuple(driver._sources)  # noqa: SLF001
            inner = plans["update_H"]

            def _after(_inner=inner, _fields=driver.fields, _electric=electric):
                _inner()
                for source in _electric:
                    getattr(source, "withdraw", lambda *_a: None)(_fields)

            plans = {"update_H": _after, "step_D": lambda: None}
        else:
            plans, _state, counters = weld_plan_for(adapter, driver, hoisted=(arm == "hoisted"))
        driver._fast_path = Shim(plans)  # noqa: SLF001
        driver._fast_path_stale = False  # noqa: SLF001
        first: Optional[Dict[str, Any]] = None
        compared = 0
        for step in range(steps):
            reference.step()
            driver.step()
            cp.cuda.runtime.deviceSynchronize()
            moved = compare(driver_state(reference), driver_state(driver))
            compared = step + 1
            if moved:
                first = {"step": step + 1, "volumes": moved}
                break
        arms_out[arm] = {"first_divergence": first, "identical": first is None,
                         "steps_compared": compared,
                         "withdrawn_total": counters["withdrawn"], "runs": counters["runs"]}
        reference.close()
        driver.close()
    record = {
        "spec": spec["label"], "steps": steps,
        "predicate_refuses_this_row": not covered, "predicate_reason": reason,
        "predicate_names_the_withdraw": "standing integrated" in reason,
        "standing_withdraws": len(standing),
        "hoistable": bool(hoistable), "hoistable_reasons": list(hoist_reasons),
        "arms": arms_out,
        "campaign": dict(withdraw_hoist.MEASUREMENT),
        "what_this_licenses": (
            "the LeadingWithdrawPlan placement on this device against the array path, "
            "with both null controls diverging. It does NOT flip HOISTS_THE_WITHDRAW: the "
            "two withdraw rows stay refused by name, and the flag moves only with "
            "INSTALLABLE and the wiring"),
    }
    record["passed"] = bool(
        record["predicate_refuses_this_row"] and record["predicate_names_the_withdraw"]
        and record["hoistable"] and standing
        and arms_out["hoisted"]["identical"]
        and not arms_out["not_hoisted"]["identical"]
        and not arms_out["after_step_D"]["identical"])
    return record


# ---------------------------------------------------------------------------
# Device legs: byte-neutral, disarm, mutation, compiler
# ---------------------------------------------------------------------------

def leg_byte_neutral(adapter: Adapter, spec: Mapping[str, Any], steps: int) -> Dict[str, Any]:
    source = needle(adapter.kernel_source(), adapter.byte_neutral["old"],
                    adapter.byte_neutral["new"])
    kernel = adapter.compile(source)
    record = drive(adapter, spec, "uniform", steps, modes=("array", "weld"), kernel=kernel)
    return {"spec": spec["label"], "identical": record["identical"]["weld"],
            "first_divergence": record["first_divergence"]["weld"],
            "edit": dict(adapter.byte_neutral),
            "why": ("the own-cell register feeding the increment is replaced by a RELOAD "
                    "of the scratch word this thread has just stored -- the same value by "
                    "a different route. A float32 stored and loaded back is the identity "
                    "on the bits, so an edit that diverged here would mean the harness "
                    "rather than the product decides the answer"),
            "passed": bool(record["identical"]["weld"] and record["non_vacuous"])}


def leg_disarm(adapter: Adapter, spec: Mapping[str, Any], steps: int) -> Dict[str, Any]:
    record = drive(adapter, spec, "uniform", steps, modes=("array", "weld"))
    return {"spec": spec["label"], "identical": record["identical"]["weld"],
            "first_divergence": record["first_divergence"]["weld"],
            "passed": bool(record["identical"]["weld"] and record["non_vacuous"])}


def mutated_source(adapter: Adapter, name: str) -> str:
    spec = adapter.device_mutations[name]
    source = adapter.kernel_source()
    if spec.get("old") is None:
        return source
    source = needle(source, spec["old"], spec["new"])
    for old, new in spec.get("also", ()):
        source = needle(source, old, new)
    return source


def leg_mutation(adapter: Adapter, specs: Sequence[Mapping[str, Any]], steps: int) -> Dict[str, Any]:
    """Every armed defect, each required to DIVERGE on at least one scored fixture,
    unless it is declared null ahead of the run WITH its reason -- in which case the
    prediction is recorded either way."""
    results: List[Dict[str, Any]] = []
    for name, spec in adapter.device_mutations.items():
        try:
            kernel = adapter.compile(mutated_source(adapter, name), spec.get("options"))
            compiled, compile_error = True, None
        except Exception as error:  # noqa: BLE001
            kernel, compiled, compile_error = None, False, repr(error)[:400]
        caught_on: List[str] = []
        caught_at_increment: List[Dict[str, Any]] = []
        classes = tuple(spec.get("classes", ("uniform",)))
        for fixture in specs:
            if not compiled:
                break
            if spec.get("live_on_labels") and fixture["label"] not in spec["live_on_labels"]:
                continue
            for value_class in classes:
                record = drive(adapter, fixture, value_class, steps,
                               modes=("array", "weld"), kernel=kernel)
                if not record["identical"]["weld"]:
                    caught_on.append(f"{fixture['label']}/{value_class}")
                if spec.get("score_increment"):
                    # THE INCREMENT ITSELF, before the scan. A zero-SIGN difference in
                    # one increment word is laundered by cupy.cumsum the moment the
                    # running sum is nonzero, so a whole-step identity cannot see the
                    # zero-sign spellings; the probe measured them on the increment,
                    # and so does this: launch 1 alone, its increment against the
                    # array path's own four passes over the SAME Hy_new.
                    label = f"{fixture['label']}/{value_class}"
                    fields, grid, pml, _ = adapter.build(fixture, value_class,
                                                         case_rng(adapter.seed, label))
                    state = adapter.resolve(fields, grid, pml)
                    adapter.launch1(fields, state, kernel)
                    cp.cuda.runtime.deviceSynchronize()
                    expected, _shipped = array_path_increment(
                        state["scratch"]["Hy"], adapter.family.PREFIX_IR0_VALUE)
                    moved = differing(state["increment"], expected)
                    caught_at_increment.append({"case": label, "differing_words": moved,
                                                "words": int(float_words(to_host(expected)).size)})
        entry = {"name": name, "kind": "device", "compiled": compiled,
                 "compile_error": compile_error, "why": spec["why"],
                 "live_on": spec.get("live_on", "every fixture"),
                 "caught_on": caught_on, "caught": bool(caught_on)}
        if spec.get("score_increment"):
            entry["increment_stage"] = caught_at_increment
            entry["caught_at_the_increment"] = any(c["differing_words"] > 0
                                                   for c in caught_at_increment)
            entry["caught_in_the_state"] = bool(caught_on)
            entry["caught"] = entry["caught_at_the_increment"] or bool(caught_on)
            entry["_scoring_note"] = (
                "scored on the INCREMENT as well as on the state: a zero-sign difference "
                "in an increment word is laundered by cupy.cumsum once the running sum is "
                "nonzero, so the whole-step identity is blind to the zero-sign spellings "
                "the probe measured on the increment")
        entry["passed"] = bool(compiled and entry["caught"])
        predicted = spec.get("predicted_null")
        if adapter.policy in spec.get("null_under", ()):
            predicted = (f"null under the {adapter.policy!r} policy by construction: "
                         + spec.get("null_reason", ""))
            entry["policy_conditional"] = {"null_under": list(spec["null_under"]),
                                           "policy": adapter.policy}
        if predicted:
            entry["predicted_null"] = predicted
            entry["prediction_held"] = not caught_on
            entry["passed"] = bool(compiled)
        results.append(entry)
        log(f"[mutation] {name:44s} compiled={compiled} caught={entry['caught']} "
            f"{'(predicted null: held=' + str(entry.get('prediction_held')) + ')' if spec.get('predicted_null') else ''}")

    for name in adapter.host_mutations:
        classes = tuple(adapter.host_mutation_classes.get(name, ("uniform",)))
        caught_on: List[str] = []
        details: List[Dict[str, Any]] = []
        entry: Dict[str, Any] = {"name": name, "kind": "host", "compiled": True}
        try:
            for fixture in specs[:1] if len(classes) == 1 else specs:
                for value_class in classes:
                    if name == "rotation_skipped":
                        record = drive(adapter, fixture, value_class, steps,
                                       modes=("array", "weld"), rotate=False)
                    elif name == "curl_launched_first":
                        record = drive(adapter, fixture, value_class, steps,
                                       modes=("array", "weld"), order="curl_first")
                    else:
                        record = drive(adapter, fixture, value_class, steps,
                                       modes=("array", "weld"), host_mutation=name)
                    caught = bool(record["refused"]) or not record["identical"]["weld"]
                    details.append({"case": record["label"],
                                    "first_divergence": record["first_divergence"]["weld"],
                                    "refused": record["refused"], "caught": caught,
                                    "state_mutation": (record.get("host_mutation") or name)})
                    if caught:
                        caught_on.append(record["label"])
            entry.update({"caught": bool(caught_on), "caught_on": caught_on,
                          "cases": details, "passed": bool(caught_on)})
        except Exception as error:  # noqa: BLE001
            entry.update({"compiled": False, "caught": False, "passed": False,
                          "error": repr(error)[:400], "cases": details})
        predicted = adapter.host_predicted_null.get(name)
        if adapter.policy in adapter.null_under.get(name, ()):
            predicted = (f"null under the {adapter.policy!r} policy by construction: "
                         + adapter.null_reasons.get(name, ""))
            entry["policy_conditional"] = {"null_under": list(adapter.null_under[name]),
                                           "policy": adapter.policy}
        if predicted:
            entry["predicted_null"] = predicted
            entry["prediction_held"] = not entry["caught"]
            entry["passed"] = entry["compiled"]
        results.append(entry)
        log(f"[mutation] {name:44s} host caught={entry['caught']}")
    return {"mutations": results, "denominator": len(results),
            "armed": len(results),
            "caught": sum(1 for e in results if e["caught"]),
            "predicted_null": [e["name"] for e in results if e.get("predicted_null")],
            "predictions_held": [e["name"] for e in results
                                 if e.get("predicted_null") and e.get("prediction_held")],
            "predictions_refuted": [e["name"] for e in results
                                    if e.get("predicted_null") and not e.get("prediction_held")],
            "passed": all(e["passed"] for e in results)}


def cupy_cache_snapshot() -> Dict[str, Any]:
    return hd_gate._cupy_cache_snapshot()  # noqa: SLF001


def leg_compiler(adapter: Adapter, policy: Optional[str],
                 at_install: Mapping[str, Any]) -> Dict[str, Any]:
    """Did THIS process compile the kernels it ran, under THIS policy, from an EMPTY
    cache? The Cartesian gate's four clauses, with this family's digest in clause 4."""
    report = probe.nvrtc_binary_report()
    outer = probe.subnormal_policy_stamp(_REPO_API)
    stamp = dict(outer.get("stamp") or {})
    stamp.setdefault("policy", outer.get("policy"))
    now = cupy_cache_snapshot()
    try:
        family_source = adapter.source_digest()
    except Exception as exc:  # noqa: BLE001
        family_source = f"unavailable: {exc!r}"
    observed = int(report.get("nvrtc_calls_observed") or 0)
    counters = {"nvrtc_calls": stamp.get("nvrtc_calls"), "ftz_removed": stamp.get("ftz_removed")}
    checks: Dict[str, bool] = {
        "the_compiler_was_exercised": observed > 0,
        "every_binary_this_process_ran_was_compiled_by_it":
            at_install.get("entries") == 0 and at_install.get("error") is None,
    }
    if policy == "keep":
        checks["the_strip_reached_every_compile"] = bool(
            (counters["ftz_removed"] or 0) > 0 and counters["nvrtc_calls"] == observed
            and not report.get("any_ftz_true_reached_nvrtc"))
    elif policy == "flush":
        checks["ftz_true_reached_every_compile"] = bool(
            report.get("all_ftz_true_reached_nvrtc") and (counters["ftz_removed"] or 0) == 0)
    else:
        checks["a_policy_was_requested"] = False
    try:
        emitted = hashlib.sha256(adapter.kernel_source().encode("utf-8")).hexdigest()
    except Exception as exc:  # noqa: BLE001
        emitted = f"unavailable: {exc!r}"
    return {
        "policy_requested": policy, "policy_stamp": stamp.get("policy"),
        "counters_after_every_in_process_leg": counters,
        "observed": {key: report.get(key) for key in
                     ("nvrtc_calls_observed", "distinct_binaries", "distinct_sources",
                      "any_ftz_true_reached_nvrtc", "all_ftz_true_reached_nvrtc")},
        "cache": {"dir": at_install.get("dir"), "entries_at_install": at_install.get("entries"),
                  "preexisting_at_install": at_install.get("preexisting"),
                  "entries_after_every_in_process_leg": now["entries"]},
        "family_source_sha256": family_source,
        "launch1_emitted_source_sha256": emitted,
        "launch1_source_observed_at_nvrtc": any(
            entry.get("source_sha256") == emitted for entry in (report.get("observations") or [])),
        "checks": checks,
        "passed": all(checks.values()),
    }


# ---------------------------------------------------------------------------
# Host leg: the arbitration, as shipped and with the wiring rows patched in
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def wired_in_process(adapter: Adapter):
    """The two ``fused_pairs`` rows the wiring change adds, installed for the duration
    of one measurement and REMOVED afterwards. Nothing on disk moves."""
    from meep_gpu.cuda_kernels import fused_pairs  # noqa: PLC0415

    name = adapter.family.FAMILY
    had_arms = name in fused_pairs.FUSED_PAIR_ARMS
    had_product = name in fused_pairs.FUSED_PRODUCTS
    saved = (fused_pairs.FUSED_PAIR_ARMS.get(name), fused_pairs.FUSED_PRODUCTS.get(name))
    fused_pairs.FUSED_PAIR_ARMS[name] = tuple(adapter.wiring_arms_row)
    fused_pairs.FUSED_PRODUCTS[name] = adapter.wiring_product_row()
    try:
        yield fused_pairs
    finally:
        if had_arms:
            fused_pairs.FUSED_PAIR_ARMS[name] = saved[0]
        else:
            fused_pairs.FUSED_PAIR_ARMS.pop(name, None)
        if had_product:
            fused_pairs.FUSED_PRODUCTS[name] = saved[1]
        else:
            fused_pairs.FUSED_PRODUCTS.pop(name, None)


def leg_arbitration(adapter: Adapter, spec: Mapping[str, Any]) -> Dict[str, Any]:
    """The COMPOSER, asked twice on a cylindrical fixture: AS SHIPPED, and WITH THE
    WIRING ROWS PATCHED IN-PROCESS. The product is refused BY NAME through its own
    INSTALLABLE declaration, holds no slot, and the two released cylindrical pairs keep
    theirs; the seam row already routes to the withdraw hoist.

    THE SHIPPED STATE MOVED ON 2026-09-07, and the clause reads which one it is
    measuring rather than assuming. Before the wiring landed the family was ABSENT from
    ``fused_pairs``' tables (the build campaigns, ``results/cuda_cylindrical_*_2026-09-07``,
    record ``as_shipped.product_in_tables`` False and the refusal only under the
    in-process rows); since the wiring the family's ``FUSED_PAIR_ARMS`` row and
    ``FUSED_PRODUCTS`` entry SHIP, so the refusal by name is measured as shipped and the
    in-process patch is the idempotence control: the rows it installs are the rows that
    ship (``shipped_rows_are_the_wiring_rows``) and the selection does not move. A pass
    requires the refusal wherever the rows are present, and it requires the shipped
    rows to BE the wiring rows once they are -- a composer carrying a different row
    under this family's name would be a different product."""
    from meep_gpu.cuda_kernels import arms, fused_pairs  # noqa: PLC0415

    family = adapter.family
    fields, grid, pml, _dtdx = adapter.build(spec, "uniform", case_rng(adapter.seed, "arbitration"))
    licenses = {"complex": adapter.license} if adapter.license is not None else None

    def ask() -> Tuple[Dict[str, str], Dict[str, List[str]]]:
        plan = arms.plan_step(fields=fields, pml=pml, grid=grid, sources=(), fuse=True,
                              licenses=licenses, subnormal_policy=adapter.policy)
        return (dict(getattr(plan, "selected", {}) or {}),
                {k: list(v) for k, v in (getattr(plan, "reasons", {}) or {}).items()})

    selected_shipped, reasons_shipped = ask()
    shipped_arms_row = fused_pairs.FUSED_PAIR_ARMS.get(family.FAMILY)
    shipped_product_row = fused_pairs.FUSED_PRODUCTS.get(family.FAMILY)
    mine_shipped = [text for key, texts in reasons_shipped.items() if family.FAMILY in key
                    for text in texts]
    with wired_in_process(adapter) as patched:
        selected_wired, reasons_wired = ask()
        mine = [text for key, texts in reasons_wired.items() if family.FAMILY in key
                for text in texts]
        declared = patched._declared_uninstallable(  # noqa: SLF001
            family.FAMILY, patched.FUSED_PRODUCTS[family.FAMILY])
        span = patched.span_of(family.FAMILY, patched.FUSED_PRODUCTS[family.FAMILY])
    seam_row = fused_pairs.FUSED_PAIR_SEAMS.get("update_H", (None, None))
    record = {
        "spec": spec["label"],
        "as_shipped": {
            "selected": selected_shipped,
            "update_H_holder": selected_shipped.get("update_H"),
            "step_D_holder": selected_shipped.get("step_D"),
            "product_in_tables": shipped_product_row is not None,
            "arms_row": list(shipped_arms_row) if shipped_arms_row is not None else None,
            "shipped_rows_are_the_wiring_rows": bool(
                shipped_product_row is not None
                and tuple(shipped_arms_row or ()) == tuple(adapter.wiring_arms_row)
                and shipped_product_row.get("curl_slot") == adapter.wiring_product_row().get("curl_slot")
                and shipped_product_row.get("module") == adapter.wiring_product_row().get("module")),
            "this_product_is_refused": bool(mine_shipped),
            "refusal_reasons": mine_shipped,
            "refusal_names_installable": any("INSTALLABLE = False" in t for t in mine_shipped),
            "this_product_holds_no_slot": not any(family.FAMILY in str(v)
                                                  for v in selected_shipped.values()),
        },
        "wired_in_process": {
            "arms_row": list(adapter.wiring_arms_row),
            "selected": selected_wired,
            "this_product_is_refused": bool(mine),
            "refusal_reasons": mine,
            "refusal_names_installable": any("INSTALLABLE = False" in t for t in mine),
            "declared_uninstallable": declared,
            "span_of": list(span),
            "this_product_holds_no_slot": not any(family.FAMILY in str(v)
                                                  for v in selected_wired.values()),
            "selection_unchanged_by_the_wiring": selected_wired == selected_shipped,
        },
        "seam_row": list(seam_row),
        "seam_row_routes_to_the_withdraw_hoist": seam_row[1] == withdraw_hoist.SEAM,
        "installable": bool(family.INSTALLABLE),
        "installable_reason": family.INSTALLABLE_REASON,
        "what_a_release_does_not_license": family.WHAT_A_RELEASE_DOES_NOT_LICENSE,
    }
    holder_h = selected_shipped.get("update_H")
    holder_d = selected_shipped.get("step_D")
    record["the_released_b_to_h_pair_holds_update_H_today"] = bool(
        holder_h is not None and selected_shipped.get("step_B") == holder_h
        and family.FAMILY not in str(holder_h))
    record["the_released_d_to_e_pair_holds_step_D_today"] = bool(
        holder_d is not None and selected_shipped.get("update_E") == holder_d
        and family.FAMILY not in str(holder_d))
    shipped = record["as_shipped"]
    # ABSENT (pre-wiring) or PRESENT AND REFUSED BY NAME on the wiring's own rows
    # (post-wiring); never present under some other row, and never holding a slot.
    shipped["consistent"] = bool(
        (not shipped["product_in_tables"])
        or (shipped["shipped_rows_are_the_wiring_rows"]
            and shipped["this_product_is_refused"]
            and shipped["refusal_names_installable"]))
    record["passed"] = bool(
        shipped["consistent"]
        and shipped["this_product_holds_no_slot"]
        and record["wired_in_process"]["this_product_is_refused"]
        and record["wired_in_process"]["refusal_names_installable"]
        and record["wired_in_process"]["declared_uninstallable"]
        and tuple(span) == tuple(family.REPLACES)
        and record["wired_in_process"]["this_product_holds_no_slot"]
        and record["wired_in_process"]["selection_unchanged_by_the_wiring"]
        and record["seam_row_routes_to_the_withdraw_hoist"]
        and not record["installable"]
        and record["the_released_b_to_h_pair_holds_update_H_today"]
        and record["the_released_d_to_e_pair_holds_step_D_today"])
    return record


# ---------------------------------------------------------------------------
# The corpus lift: basis, parent, child
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def lift_basis(adapter: Adapter, results: Path) -> Tuple[List[dict], Dict[str, Any]]:
    """The rows the standing BOARD puts in this product's cell, joined to the census
    for the module each tests row replays. Derived, never listed."""
    board = results / BOARD / "fusion_matrix_cuda.json"
    if not board.is_file():
        raise SystemExit(f"the lift leg needs {board}")
    payload = json.loads(board.read_text(encoding="utf-8"))
    census = results / CENSUS
    modules: Dict[str, str] = {}
    replay_case: Dict[str, str] = {}
    for leg in ("tests", "tests_param_matched"):
        path = census / f"{leg}.jsonl"
        if not path.is_file():
            continue
        for record in _load_jsonl(path):
            case = record.get("case") or record.get("row")
            label = f"tests:{record.get('row') or case}"
            if case and record.get("module"):
                modules[label] = record["module"]
            if case and record.get("row") and case != record["row"]:
                replay_case[label] = case
    rows: List[dict] = []
    for instance in payload["h_to_d_seam"]["instances"]:
        cell = (instance.get("update_H"), instance.get("step_D"))
        if cell != tuple(adapter.cell_arms):
            continue
        label = instance["row"]
        leg, row = label.split(":", 1)
        rows.append({"label": label, "leg": leg, "row": row,
                     "module": modules.get(label) if leg != "examples" else None,
                     "replay_case": replay_case.get(label, row),
                     "cell": list(cell), "bucket": instance.get("bucket"),
                     "withdraw_in_seam": bool(instance.get("withdraw_in_seam")),
                     "integrated_electric_sources": instance.get("integrated_electric_sources")})
    facts = {
        "board": BOARD, "census": CENSUS,
        "board_census_stamp": payload.get("census"),
        "cell": list(adapter.cell_arms), "rows_total": len(rows),
        "rows_by_bucket": {b: sum(1 for r in rows if r["bucket"] == b)
                           for b in sorted({r["bucket"] for r in rows})},
        "rows_with_a_standing_withdraw": sorted(r["label"] for r in rows if r["withdraw_in_seam"]),
        "tests_rows_without_a_module": sorted(r["label"] for r in rows
                                             if r["leg"] != "examples" and not r["module"]),
        "parametrised_cases_replayed_under_the_shim_name": {
            r["label"]: r["replay_case"] for r in rows if r["replay_case"] != r["row"]},
    }
    return rows, facts


def child_progress(message: str) -> None:
    path = os.environ.get("MEEP_GPU_CYL_HD_GATE_PROGRESS")
    label = os.environ.get("MEEP_GPU_CYL_HD_GATE_LABEL", "?")
    line = f"[{time.strftime('%H:%M:%S')}] {label} {message}"
    print(line, flush=True)
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()


def drive_lifted_row(adapter: Adapter, driver: Any, steps: int) -> Dict[str, Any]:
    """One lifted corpus row through the driver's own consult order: per complete step
    the state is captured, the weld runs one step and is rolled back, the certified
    singles run one step and are rolled back, the array path runs the same step and
    the run continues from it. The weld's claim is against the certified singles; a
    step where both dispatched arrangements agree and leave the array path is a finding
    about the backend, recorded with its subnormal census. The replay control gates
    every row first."""
    plans, state, counters = weld_plan_for(adapter, driver)
    weld_shim = Shim(plans)
    singles_shim: Optional[Shim] = None
    singles_error: Optional[str] = None
    try:
        singles = adapter.singles(driver.fields, driver.grid, driver.pml,
                                  float(driver.grid.dt / driver.grid.dx))
        singles_shim = Shim({"update_H": singles["update_H"], "step_D": singles["step_D"]})
    except Exception as error:  # noqa: BLE001
        singles_error = repr(error)[:400]
    detached = {"dft": list(getattr(driver, "_dft_monitors", ()) or ()),
                "flux": list(getattr(driver, "_flux_monitors", ()) or ())}
    driver._dft_monitors = []  # noqa: SLF001
    driver._flux_monitors = []  # noqa: SLF001
    capture, restore = hd_gate._capture, hd_gate._restore  # noqa: SLF001
    first: Optional[Dict[str, Any]] = None
    backend_steps: List[Dict[str, Any]] = []
    clean = scored = 0
    per_step_words = 0
    banded_at: Optional[int] = None
    error: Optional[str] = None
    not_replayable: Optional[Dict[str, Any]] = None
    started = time.perf_counter()
    with CumsumCounter() as scans:
        for step in range(steps):
            before = capture(driver)
            per_step_words = sum(words(v).size for v in before["volumes"].values())
            if step == 0:
                try:
                    driver._fast_path = None  # noqa: SLF001
                    driver._fast_path_stale = False  # noqa: SLF001
                    driver.step()
                    cp.cuda.runtime.deviceSynchronize()
                    replay_a = capture(driver)
                    restore(driver, before)
                    driver.step()
                    cp.cuda.runtime.deviceSynchronize()
                    replay_b = capture(driver)
                    restore(driver, before)
                except Exception as exc:  # noqa: BLE001
                    error = repr(exc)[:400]
                    break
                moved = compare(replay_a["volumes"], replay_b["volumes"])
                if moved:
                    not_replayable = {"volumes": moved,
                                      "reason": ("one complete driver step run twice from "
                                                 "ONE captured state on the array path did "
                                                 "not reproduce itself; byte identity across "
                                                 "arrangements is undefined on this row")}
                    break
            try:
                driver._fast_path_stale = False  # noqa: SLF001
                driver._fast_path = weld_shim  # noqa: SLF001
                driver.step()
                cp.cuda.runtime.deviceSynchronize()
                weld_state = capture(driver)
                restore(driver, before)
                singles_state = None
                if singles_shim is not None:
                    driver._fast_path = singles_shim  # noqa: SLF001
                    driver.step()
                    cp.cuda.runtime.deviceSynchronize()
                    singles_state = capture(driver)
                    restore(driver, before)
                driver._fast_path = None  # noqa: SLF001
                driver.step()
                cp.cuda.runtime.deviceSynchronize()
                array_state = capture(driver)
            except Exception as exc:  # noqa: BLE001
                error = repr(exc)[:400]
                break
            census = {
                "array_path": sum(subnormal_words(v) for v in array_state["volumes"].values()),
                "weld": sum(subnormal_words(v) for v in weld_state["volumes"].values()),
                "certified_singles": (None if singles_state is None else
                                      sum(subnormal_words(v) for v in singles_state["volumes"].values())),
            }
            if any(value for value in census.values() if value is not None):
                banded_at = step + 1
                break
            scored = step + 1
            against_singles = ({} if singles_state is None else
                               compare(singles_state["volumes"], weld_state["volumes"]))
            if against_singles:
                first = {"step": step + 1, "against": "the certified singles",
                         "volumes": against_singles, "subnormal_census": census}
                break
            against_array = compare(array_state["volumes"], weld_state["volumes"])
            if against_array:
                if singles_state is None:
                    first = {"step": step + 1, "against": "the array path",
                             "volumes": against_array, "subnormal_census": census,
                             "note": "no certified-singles arrangement was available"}
                    break
                backend_steps.append({"step": step + 1, "volumes": against_array,
                                      "subnormal_census": census})
                continue
            clean = step + 1
        scans_total = scans.calls
    driver._dft_monitors = detached["dft"]  # noqa: SLF001
    driver._flux_monitors = detached["flux"]  # noqa: SLF001
    return {
        "steps_compared": clean, "steps_scored": scored,
        "first_divergence": first, "step_error": error,
        "not_replayable": not_replayable,
        "certified_singles_error": singles_error,
        "backend_disagreement_steps": backend_steps,
        "monitors_detached": {key: len(value) for key, value in detached.items()},
        "entered_the_denormal_band_at_step": banded_at,
        "words_compared_per_step": per_step_words,
        "words_compared": per_step_words * scored * (2 if singles_shim else 1),
        "weld_runs": counters["runs"], "weld_family_tally": int(state["launches"]),
        "weld_cumsum_calls": int(state["cumsum_calls"]),
        "weld_curl_launches": int(state["curl_launches"]),
        "cumsum_calls_all_arrangements": scans_total,
        "every_weld_run_advanced_three_counters": bool(
            counters["runs"] > 0 and state["launches"] == 3 * counters["runs"]
            and state["cumsum_calls"] == counters["runs"]
            and state["curl_launches"] == counters["runs"]),
        "seconds": round(time.perf_counter() - started, 2),
        "clean_step_floor": LIFT_CLEAN_STEP_FLOOR,
        "below_the_floor": scored < LIFT_CLEAN_STEP_FLOOR and first is None and not_replayable is None,
        "passed": bool(first is None and error is None and not_replayable is None
                       and scored >= LIFT_CLEAN_STEP_FLOOR and counters["runs"] > 0
                       and state["launches"] == 3 * counters["runs"]),
    }


def evaluate_row(adapter: Adapter, driver: Any, steps: int, max_cells: Optional[str],
                 block_key: str) -> Dict[str, Any]:
    from meep_gpu.cuda_kernels import arms  # noqa: PLC0415

    family = adapter.family
    sources = tuple(getattr(driver, "_sources", ()) or ())
    block: Dict[str, Any] = {
        "family": family.FAMILY, "steps_requested": steps, "n_sources": len(sources),
        "sources": [{"type": type(s).__name__, "field_type": str(getattr(s, "field_type", "")),
                     "is_integrated": bool(getattr(s, "is_integrated", False)),
                     "withdraw_does_work": bool(withdraw_hoist._withdraw_does_work(s))}  # noqa: SLF001
                    for s in sources],
        "grid_shape": [int(n) for n in driver.grid.shape], "m": int(driver.grid.m),
        "storage": str(driver.fields.Hy.dtype),
        "policy_stamp_before_drive": probe.subnormal_policy_stamp(_REPO_API).get("policy"),
    }
    covered, reason = adapter.predicate(driver.fields, driver.pml, driver.grid, sources)
    block["predicate_admits"] = bool(covered)
    block["predicate_reason"] = reason
    try:
        licenses = {"complex": adapter.license} if adapter.license is not None else None
        plan = arms.plan_step(fields=driver.fields, pml=driver.pml, grid=driver.grid,
                              sources=sources, fuse=True, licenses=licenses,
                              subnormal_policy=adapter.policy)
        block["composer_selected_as_shipped"] = dict(getattr(plan, "selected", {}) or {})
    except Exception as error:  # noqa: BLE001
        block["composer_error"] = repr(error)[:400]
    cells = int(np.prod(driver.grid.shape))
    block["grid_cells"] = cells
    if not covered:
        block.update({"driven": False, "why_not_driven": "the predicate refused this row"})
        return {block_key: block}
    if max_cells and cells > int(max_cells):
        block.update({"driven": False,
                      "why_not_driven": f"{cells} cells exceeds the cap {max_cells}; refused"})
        return {block_key: block}
    block["driven"] = True
    block.update(drive_lifted_row(adapter, driver, steps))
    block["policy_stamp_after_drive"] = probe.subnormal_policy_stamp(_REPO_API).get("policy")
    child_progress(f"passed={block.get('passed')} words={block.get('words_compared')} "
                   f"steps={block.get('steps_compared')}")
    return {block_key: block}


def lift_child(adapter: Adapter, leg: str, target: str, cases: Sequence[str],
               out_json: str, steps: int, progress: str, block_key: str) -> int:
    """Lift ONE corpus row onto the device and drive it. Writes ``out_json`` always."""
    import meep_gpu  # noqa: PLC0415

    def mark(message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {Path(target).name:<38} {message}"
        print(line, flush=True)
        if progress:
            with open(progress, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()

    records: List[Dict[str, Any]] = []

    def finish() -> int:
        for record in records:
            gate_provenance.stamp(record)
        Path(out_json).write_text(json.dumps(records, default=str), encoding="utf-8")
        return 0

    if leg == "examples":
        from parity.meep_gpu.sweep_corpus_lift_parity import capture_simulation  # noqa: PLC0415

        record, sim, restore = capture_simulation(target)
        record["row"] = Path(target).name
        records.append(record)
        if sim is None:
            record.update({"measured": False, "note": "no mp.Simulation to lift"})
            return finish()
        restore()
        pairs = [(record, sim)]
    else:
        from parity.meep_gpu import survey_meep_tests as harness  # noqa: PLC0415

        namespace = harness.build_child_namespace()
        module = namespace["_import_module"](target)
        wanted = set(cases)
        pairs = []
        for class_name, method_name in namespace["_enumerate_cases"](module):
            case_id = f"{class_name}.{method_name}"
            record, sim, restore, _case = namespace["_run_case"](
                module, target, class_name, method_name, 900.0)
            restore()
            if case_id not in wanted:
                continue
            record["row"] = case_id
            records.append(record)
            if sim is None:
                record.update({"measured": False, "note": "no mp.Simulation to lift on this replay"})
                continue
            pairs.append((record, sim))
    max_cells = os.environ.get("MEEP_GPU_CYL_HD_GATE_MAX_CELLS")
    for record, sim in pairs:
        started = time.time()
        try:
            driver = meep_gpu.lift_simulation(sim, prefer_gpu=True, gpu_id=0)
        except BaseException as exc:  # noqa: BLE001
            record.update({"measured": False, "lift_error": f"{type(exc).__name__}: {exc}"[:600]})
            mark(f"LIFT FAILED {type(exc).__name__}")
            continue
        record["lift_s"] = round(time.time() - started, 2)
        record["grid_shape"] = [int(v) for v in driver.shape]
        record["grid_cells"] = int(np.prod([int(v) for v in driver.shape]))
        try:
            record.update(evaluate_row(adapter, driver, steps, max_cells, block_key))
            record["measured"] = True
        except BaseException as exc:  # noqa: BLE001
            record.update({"measured": False, "battery_error": f"{type(exc).__name__}: {exc}"[:600]})
        finally:
            with contextlib.suppress(BaseException):
                driver.close()
        block = record.get(block_key) or {}
        mark(f"measured={record.get('measured')} admits={block.get('predicate_admits')} "
             f"passed={block.get('passed')} steps={block.get('steps_compared')}")
    return finish()


def leg_lift(adapter: Adapter, gate_file: str, block_key: str, out_dir: Path, steps: int,
             max_cells: Optional[int], timeout: float, resume: bool,
             only: Optional[Sequence[str]], policy: Optional[str],
             import_meep: bool) -> Dict[str, Any]:
    """Every corpus row in the cell, re-lifted in its own interpreter and driven. The
    child installs the SAME policy this parent installed, before its first compile."""
    import measure_predicate_coverage as census  # noqa: PLC0415

    out_dir = Path(out_dir).resolve()
    rows, facts = lift_basis(adapter, Path(_HERE) / "results")
    if only:
        wanted = set(only)
        rows = [row for row in rows if row["label"] in wanted]
    lift_dir = out_dir / "lift"
    per_row = lift_dir / "per_row"
    per_row.mkdir(parents=True, exist_ok=True)
    progress_log = lift_dir / "steps.progress.log"
    examples_dir = Path(census.EXAMPLES_DIR)
    tests_dir = Path(census.TESTS_DIR)
    if not examples_dir.is_dir() or not tests_dir.is_dir():
        return {"passed": False, "facts": facts,
                "reason": (f"the MEEP corpus is not at {examples_dir} / {tests_dir}; set "
                           f"MEEP_GPU_CORPUS_ROOT. Refused rather than measured on nothing")}
    environment = dict(os.environ)
    environment.update({"KMP_DUPLICATE_LIB_OK": "TRUE", "MPLBACKEND": "Agg",
                        "PYTHONPATH": _REPO_API,
                        "MEEP_GPU_CYL_HD_GATE_PROGRESS": str(progress_log)})
    if max_cells is not None:
        environment["MEEP_GPU_CYL_HD_GATE_MAX_CELLS"] = str(max_cells)
    work = Path(tempfile.mkdtemp(prefix="cuda_cyl_hd_lift_"))
    for entry in examples_dir.iterdir():
        if entry.suffix in (".py", ".ipynb"):
            continue
        link = work / entry.name
        if not link.exists():
            with contextlib.suppress(OSError):
                link.symlink_to(entry)
    shim_path = Path(census.__file__).resolve().parent / "shim"
    needs_shim = set()
    for row in rows:
        module_name = row.get("module")
        if row["leg"] == "examples" or not module_name:
            continue
        module_file = tests_dir / module_name
        if module_file.is_file() and "import parameterized" in module_file.read_text(
                encoding="utf-8", errors="replace"):
            needs_shim.add(module_name)
    policy_flags = ([] if not policy else ["--subnormal-policy", policy]) + (
        ["--import-meep-for-host-policy"] if import_meep else [])
    measured: List[dict] = []
    for index, row in enumerate(rows, start=1):
        record_path = per_row / (row["label"].replace(":", "__").replace(".", "_") + ".json")
        environment["MEEP_GPU_CYL_HD_GATE_LABEL"] = row["label"]
        environment["PYTHONPATH"] = (f"{shim_path}{os.pathsep}{_REPO_API}"
                                     if row.get("module") in needs_shim else _REPO_API)
        started = time.time()
        if not (resume and record_path.exists()):
            if row["leg"] == "examples":
                command = [sys.executable, "-u", gate_file, "--lift-child", "examples",
                           "--lift-child-target", str(examples_dir / row["row"]),
                           "--lift-child-out", str(record_path),
                           "--lift-child-progress", str(progress_log),
                           "--lift-steps", str(steps), *policy_flags]
            elif not row["module"]:
                log(f"lift {index}/{len(rows)} {row['label']}: REFUSED, no module recorded")
                measured.append({**row, "measured": False, "note": "no module recorded"})
                continue
            else:
                command = [sys.executable, "-u", gate_file, "--lift-child", "tests",
                           "--lift-child-target", str(tests_dir / row["module"]),
                           "--lift-child-cases", json.dumps([row.get("replay_case") or row["row"]]),
                           "--lift-child-out", str(record_path),
                           "--lift-child-progress", str(progress_log),
                           "--lift-steps", str(steps), *policy_flags]
            log(f"lift {index}/{len(rows)} {row['label']} start")
            stderr_text, note = "", "child died"
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
                died = {"row": row["row"], "measured": False, "note": note,
                        "stderr_tail": stderr_text[-1500:]}
                gate_provenance.stamp(died)
                record_path.write_text(json.dumps(died, default=str), encoding="utf-8")
        payload = json.loads(record_path.read_text(encoding="utf-8"))
        candidates = payload if isinstance(payload, list) else [payload]
        wanted_case = row.get("replay_case") or row["row"]
        record = next((r for r in candidates if r.get("row") in (row["row"], wanted_case)),
                      candidates[0] if candidates else {})
        record.update({key: value for key, value in row.items() if key != "label"})
        record["label"] = row["label"]
        measured.append(record)
        block = record.get(block_key) or {}
        log(f"lift {index}/{len(rows)} {row['label']}: measured={record.get('measured')} "
            f"admits={block.get('predicate_admits')} driven={block.get('driven')} "
            f"passed={block.get('passed')} steps={block.get('steps_compared')} "
            f"words={block.get('words_compared')} ({time.time() - started:.1f} s)")
    (lift_dir / "rows.json").write_text(
        json.dumps(measured, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    blocks = {r["label"]: (r.get(block_key) or {}) for r in measured}
    admitted = sorted(label for label, b in blocks.items() if b.get("predicate_admits"))
    refused = sorted(label for label, b in blocks.items() if b.get("predicate_admits") is False)
    refused_by_the_withdraw = sorted(
        label for label in refused
        if "standing integrated" in str(blocks[label].get("predicate_reason", "")))
    absent_dependency = ("ModuleNotFoundError", "ImportError", "must be configured/compiled with")

    def unlifted_reason(record: Mapping[str, Any]) -> Optional[str]:
        if record.get("has_simulation"):
            return None
        for key in ("error", "lift_error", "note", "battery_error"):
            text = str(record.get(key) or "")
            if any(marker in text for marker in absent_dependency):
                return text[:300]
        return None

    by_label = {r["label"]: r for r in measured}
    unlifted = {label: reason for label, record in by_label.items()
                if "predicate_admits" not in (blocks.get(label) or {})
                and (reason := unlifted_reason(record))}
    unmeasured = sorted(label for label, b in blocks.items()
                        if "predicate_admits" not in b and label not in unlifted)
    driven = sorted(label for label in admitted if blocks[label].get("driven"))
    diverged = sorted(label for label in driven if blocks[label].get("first_divergence"))
    backend_rows = {label: len(blocks[label].get("backend_disagreement_steps") or ())
                    for label in driven if blocks[label].get("backend_disagreement_steps")}
    below_floor = sorted(label for label in driven if blocks[label].get("below_the_floor"))
    errored = sorted(label for label in driven if blocks[label].get("step_error"))
    unreplayable = {label: blocks[label]["not_replayable"]["volumes"]
                    for label in driven if blocks[label].get("not_replayable")}
    passed_rows = sorted(label for label in driven if blocks[label].get("passed"))
    expected_refused = sorted(set(facts["rows_with_a_standing_withdraw"]) - set(unlifted))
    if only:
        expected_refused = sorted(set(expected_refused) & set(blocks))
    return {
        "facts": facts, "rows_measured": len(measured),
        "admitted": len(admitted), "refused": len(refused),
        "refused_by_the_withdraw_clause": refused_by_the_withdraw,
        "expected_refused": expected_refused,
        "the_refusals_are_exactly_the_withdraw_rows":
            refused_by_the_withdraw == expected_refused and refused == expected_refused,
        "unmeasured": unmeasured, "unlifted_on_this_host": unlifted,
        "rows_the_engine_was_asked_about": len(blocks) - len(unlifted),
        "driven": len(driven), "passed_rows": len(passed_rows), "diverged": diverged,
        "backend_disagreement_rows": backend_rows,
        "steps_scored": sum(int(blocks[l].get("steps_scored") or 0) for l in driven),
        "below_the_floor": below_floor, "errored": errored,
        "not_replayable_rows": unreplayable,
        "complete_driver_steps": sum(int(blocks[l].get("steps_compared") or 0) for l in driven),
        "words_compared": sum(int(blocks[l].get("words_compared") or 0) for l in driven),
        "weld_runs": sum(int(blocks[l].get("weld_runs") or 0) for l in driven),
        "every_driven_row_advanced_three_counters": all(
            blocks[l].get("every_weld_run_advanced_three_counters") for l in driven),
        "passed": bool(measured and driven and not diverged and not errored and not unmeasured
                       and not below_floor and refused == expected_refused
                       and all(blocks[l].get("every_weld_run_advanced_three_counters")
                               for l in driven)),
    }


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------

LEG_GROUPS: Dict[str, Tuple[str, ...]] = {
    "host": ("driver_order", "transcription", "refusal", "arbitration"),
    "device": ("spelling", "increment_stage", "floors", "halo_ledger", "product",
               "block_sizes", "launch_structure", "sync", "withdraw", "byte_neutral",
               "mutation", "disarm", "compiler"),
    "corpus": ("lift",),
}
ALL_LEGS: Tuple[str, ...] = tuple(leg for group in LEG_GROUPS.values() for leg in group)


def save(results: Dict[str, Any], out: str) -> None:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    gate_provenance.stamp(results)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")


def load_expansion_licence(policy: Optional[str]) -> Dict[str, Any]:
    """The complex family's licence for THIS policy, through the one arbiter."""
    from meep_gpu.triton_kernels import complex_fields  # noqa: PLC0415

    out: Dict[str, Any] = {"policy_required": policy, "path": None, "verdict": None,
                           "policy_reasons": [], "usable": False}
    if policy not in EXPANSION_PROBES:
        out["policy_reasons"] = [f"no expansion probe is recorded for policy {policy!r}"]
        return out
    path = Path(_REPO_API) / EXPANSION_PROBES[policy]
    out["path"] = str(path)
    if not path.is_file():
        out["policy_reasons"] = [f"the expansion probe artifact {path} is not on this host"]
        return out
    raw = path.read_bytes()
    out["record_sha256"] = hashlib.sha256(raw).hexdigest()
    record = json.loads(raw.decode("utf-8"))
    verdict = complex_fields.expansion_license(record)
    out["verdict"] = verdict
    out["policy_reasons"] = list(complex_fields.expansion_policy_reasons(record, policy))
    out["certification_reasons_triton_package"] = list(
        complex_fields.expansion_certification_reasons(policy))
    out["arm"] = verdict.get("arm")
    out["usable"] = bool(verdict.get("arm") and not verdict.get("refusals")
                         and not out["policy_reasons"])
    return out


def make_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--out", default=None, help="artifact FILE path")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--product", choices=("full", "reduced"), default="full")
    parser.add_argument("--subnormal-policy", choices=("keep", "flush"), default=None)
    parser.add_argument("--import-meep-for-host-policy", action="store_true")
    parser.add_argument("--legs", default=None, help="comma-separated subset of " + ",".join(ALL_LEGS))
    parser.add_argument("--lift-steps", type=int, default=STEPS)
    parser.add_argument("--lift-max-cells", type=int, default=None)
    parser.add_argument("--lift-timeout", type=float, default=2400.0)
    parser.add_argument("--lift-only", default=None, help="comma-separated row labels")
    parser.add_argument("--lift-resume", action="store_true")
    parser.add_argument("--no-device", action="store_true")
    parser.add_argument("--lift-child", choices=("examples", "tests"), default=None)
    parser.add_argument("--lift-child-target", default=None)
    parser.add_argument("--lift-child-cases", default="[]")
    parser.add_argument("--lift-child-out", default=None)
    parser.add_argument("--lift-child-progress", default="")
    return parser


def install_policy(results: Dict[str, Any], args) -> Dict[str, Any]:
    """Observer first (innermost), MEEP import if the host half needs it, then the
    policy -- BEFORE the first compile. Returns the cache snapshot at install."""
    if args.import_meep_for_host_policy:
        results["meep_host_import"] = probe.import_meep_for_host_policy()
    results["nvrtc_observer"] = probe.install_nvrtc_binary_observer()
    if args.subnormal_policy:
        results["subnormal_policy_install"] = probe.install_subnormal_policy_for_run(
            args.subnormal_policy, _REPO_API)
    results["cupy_cache"] = cupy_cache_snapshot()
    results["host"] = platform.node()
    results["environment"] = probe.device_info()
    results["subnormal_policy_stamp"] = probe.subnormal_policy_stamp(_REPO_API)
    return results["cupy_cache"]


def run_gate(make_adapter: Callable[[Optional[str], Optional[Dict[str, Any]]], Adapter],
             gate_file: str, family_name: str, block_key: str, description: str,
             argv: Optional[Sequence[str]] = None) -> int:
    """The shared ``main``. ``make_adapter(policy, licence_verdict)`` builds the family's
    adapter once the policy is installed and the licence (complex) is read."""
    parser = make_parser(description)
    args = parser.parse_args(argv)

    if args.lift_child:
        if cp is None:
            print("no CuPy in the lift child", file=sys.stderr, flush=True)
            return 2
        scratch: Dict[str, Any] = {}
        install_policy(scratch, args)
        licence_report = (load_expansion_licence(args.subnormal_policy)
                          if args.subnormal_policy else {"verdict": None})
        adapter = make_adapter(args.subnormal_policy, licence_report.get("verdict"))
        return lift_child(adapter, args.lift_child, args.lift_child_target,
                          json.loads(args.lift_child_cases), args.lift_child_out,
                          args.lift_steps, args.lift_child_progress, block_key)

    if not args.out:
        parser.error("--out is required unless --lift-child is given")
    started = time.perf_counter()
    legs = tuple(args.legs.split(",")) if args.legs else ALL_LEGS
    results: Dict[str, Any] = {
        "gate": Path(gate_file).name, "family": family_name, "seam": withdraw_hoist.SEAM,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "argv": list(argv if argv is not None else sys.argv[1:]),
        "legs_requested": list(legs), "steps": args.steps,
        "standalone_note": (
            "this family is NOT in fused_pairs' tables (the wiring change is a patch, "
            "not applied): every arrangement here is planned from arrays by the gate "
            "itself and the array path is driven by the gate. The composition-installed-"
            "today reference IS driven (the two released cylindrical pairs, launched "
            "through their own run_* entry points); what awaits wiring is the composer "
            "building a composition that includes THIS product, which the arbitration "
            "leg measures in-process with the wiring rows patched in and then removed"),
    }
    if cp is None or args.no_device:
        results["device_mode"] = False
        results["status"] = "refused: no CuPy" if cp is None else "structural only"
        save(results, args.out)
        log(results["status"])
        return 1
    results["device_mode"] = True
    at_install = install_policy(results, args)
    licence_report = load_expansion_licence(args.subnormal_policy) if args.subnormal_policy else None
    adapter = make_adapter(args.subnormal_policy,
                           licence_report.get("verdict") if licence_report else None)
    if adapter.complex_storage:
        results["expansion_licence"] = licence_report
        if not (licence_report and licence_report.get("usable")):
            results["status"] = "refused: no usable expansion licence for this policy"
            save(results, args.out)
            log(results["status"])
            return 1
    family = adapter.family
    results["what_a_release_does_not_license"] = family.WHAT_A_RELEASE_DOES_NOT_LICENSE
    results["installable"] = bool(family.INSTALLABLE)
    results["installable_reason"] = family.INSTALLABLE_REASON
    results["launches_per_run"] = int(family.LAUNCHES_PER_RUN)
    results["source_sha256"] = adapter.source_digest()
    results["launch1_source_sha256"] = hashlib.sha256(
        adapter.kernel_source().encode("utf-8")).hexdigest()
    results["arm"] = adapter.arm
    save(results, args.out)

    specs = list(adapter.specs) if args.product == "full" else [
        spec for spec in adapter.specs if spec["label"] in adapter.reduced_labels]
    by_label = {spec["label"]: spec for spec in adapter.specs}
    mutation_specs = [by_label[label] for label in adapter.mutation_spec_labels]
    primary = specs[0]

    def run_leg(name: str, thunk: Callable[[], Dict[str, Any]]) -> None:
        if name not in legs:
            return
        started_leg = time.perf_counter()
        try:
            results[name] = thunk()
        except Exception as error:  # noqa: BLE001
            results[name] = {"passed": False, "error": repr(error)[:2000]}
        results[name]["seconds"] = round(time.perf_counter() - started_leg, 1)
        log(f"[{name}] passed={results[name].get('passed')} ({results[name]['seconds']} s)")
        save(results, args.out)

    run_leg("driver_order", lambda: leg_driver_order(adapter))
    run_leg("transcription", adapter.leg_transcription)
    run_leg("refusal", adapter.leg_refusal)
    run_leg("arbitration", lambda: leg_arbitration(adapter, primary))
    run_leg("spelling", adapter.leg_spelling)
    run_leg("increment_stage", lambda: leg_increment_stage(adapter, specs, adapter.value_classes))
    run_leg("floors", lambda: leg_floors(adapter, specs))
    run_leg("halo_ledger", lambda: leg_halo_ledger(adapter, specs))

    if "product" in legs:
        cases: List[Dict[str, Any]] = []
        for spec in specs:
            for value_class in tuple(spec.get("classes", adapter.value_classes)):
                try:
                    record = drive(adapter, spec, value_class, args.steps,
                                   progress=lambda m: log(f"  {m}"))
                except Exception as error:  # noqa: BLE001 - one broken fixture is a
                    # failed case, not a dead campaign; the other legs still measure.
                    record = {"label": f"{spec['label']}/{value_class}", "spec": spec["label"],
                              "value_class": value_class, "error": repr(error)[:600],
                              "steps_compared": 0, "words_compared": 0, "passed": False,
                              "identical": {}, "weld_counters": None,
                              "operand_census": {"subnormals": None}, "read_only_drift": [],
                              "non_vacuous": False}
                cases.append(record)
                log(f"[product] {record['label']:40s} "
                    f"{'IDENTICAL' if record['passed'] else 'DIVERGED'} "
                    f"steps={record['steps_compared']} words={record['words_compared']} "
                    f"subnormals={record['operand_census']['subnormals']} "
                    f"tally={record['weld_counters']['family_tally'] if record['weld_counters'] else None}")
                results["product"] = {"cases": cases}
                save(results, args.out)
        band = [r for r in cases if r["value_class"] == "subnormal_band"]
        uniform = [r for r in cases if r["value_class"] == "uniform"]
        results["product"] = {
            "cases": cases, "denominator": len(cases),
            "complete_driver_steps": sum(r["steps_compared"] for r in cases),
            "words_compared": sum(r["words_compared"] for r in cases),
            "arrangements": list(MODES),
            "the_band_class_really_contains_subnormals":
                bool(band) and all((r["operand_census"]["subnormals"] or 0) > 0 for r in band),
            "the_uniform_class_contains_none":
                bool(uniform) and all(r["operand_census"]["subnormals"] == 0 for r in uniform),
            "errored_cases": [r["label"] for r in cases if r.get("error")],
            "every_case_advanced_all_three_weld_counters": all(
                r["weld_counters"] and r["weld_counters"]["all_three_advanced"] for r in cases),
            "passed": bool(cases) and all(r["passed"] for r in cases),
        }
        save(results, args.out)

    if "block_sizes" in legs:
        sweep: List[Dict[str, Any]] = []
        for spec in specs:
            for threads in BLOCK_SIZES:
                try:
                    record = drive(adapter, spec, "uniform", min(args.steps, MUTATION_STEPS),
                                   modes=("array", "weld"), threads=threads)
                except Exception as error:  # noqa: BLE001
                    record = {"label": f"{spec['label']}/uniform", "spec": spec["label"],
                              "error": repr(error)[:600], "identical": {"weld": False},
                              "passed": False}
                record["threads"] = threads
                sweep.append(record)
                log(f"[block_sizes] {spec['label']:30s} b{threads:<5d} "
                    f"{'IDENTICAL' if record['identical']['weld'] else 'DIVERGED'}")
            results["block_sizes"] = {"cases": sweep}
            save(results, args.out)
        results["block_sizes"] = {"cases": sweep, "denominator": len(sweep),
                                  "block_sizes": list(BLOCK_SIZES),
                                  "passed": bool(sweep) and all(r["passed"] for r in sweep)}
        save(results, args.out)

    run_leg("launch_structure",
            lambda: leg_launch_structure(adapter, primary, min(args.steps, MUTATION_STEPS)))
    run_leg("sync", lambda: leg_sync(adapter, primary, min(args.steps, MUTATION_STEPS)))
    run_leg("withdraw", lambda: leg_withdraw(adapter, primary, min(args.steps, MUTATION_STEPS)))
    run_leg("byte_neutral", lambda: leg_byte_neutral(adapter, primary, args.steps))
    run_leg("mutation", lambda: leg_mutation(adapter, mutation_specs, min(args.steps, MUTATION_STEPS)))
    run_leg("disarm", lambda: leg_disarm(adapter, primary, args.steps))
    run_leg("compiler", lambda: leg_compiler(adapter, args.subnormal_policy, at_install))
    run_leg("lift", lambda: leg_lift(
        adapter, gate_file, block_key, Path(args.out).parent, args.lift_steps,
        args.lift_max_cells, args.lift_timeout, args.lift_resume,
        args.lift_only.split(",") if args.lift_only else None,
        args.subnormal_policy, args.import_meep_for_host_policy))

    clauses = {name: bool(results.get(name, {}).get("passed")) for name in legs if name in results}
    results["verdict"] = {"clauses": clauses, "passed": bool(clauses) and all(clauses.values()),
                          "legs_run": sorted(clauses), "legs_requested": list(legs),
                          "legs_missing": sorted(set(legs) - set(clauses))}
    results["canonical_verdict"] = {
        "released": bool(results["verdict"]["passed"] and not results["verdict"]["legs_missing"]),
        "reasons": [name for name, value in clauses.items() if not value]
                   + [f"leg did not run: {name}" for name in results["verdict"]["legs_missing"]],
        "what_it_licenses": (
            "a two-launch product (the certified constitutive recompute plus the fused "
            "pre-cumsum increment; xp.cumsum untouched; the certified cylindrical curl) "
            "measured byte-identical, per COMPLETE DRIVER STEP and as uint32 words over "
            "every stored volume, to the array path, the certified singles, the "
            "composition the composer installs today and the unfused slots, at every "
            "block size, on the fixtures and corpus rows this record names, with every "
            "weld step advancing three independent launch counters. It licenses NO "
            "throughput claim, NO dispatch claim and NO composition claim: INSTALLABLE = "
            "False, the family is not in the composer's tables until the wiring lands, "
            "and a credited seam-instance is PREDICATE ADMISSION"),
    }
    results["elapsed_s"] = round(time.perf_counter() - started, 1)
    save(results, args.out)
    log("=" * 78)
    for clause, value in clauses.items():
        log(f"  {'PASS' if value else 'FAIL'}  {clause}")
    log(f"VERDICT: {'PASS' if results['verdict']['passed'] else 'FAIL'}  ({results['elapsed_s']} s)")
    return 0 if results["verdict"]["passed"] else 1
