"""Shared machinery for the two CUDA ``grid.beta`` H->D one-launch product gates.

THE TWO GATES THAT IMPORT THIS are ``gate_special_kz_fused_hd_pair.py`` (REAL float32
storage) and ``gate_complex_beta_fused_hd_pair.py`` (complex64 storage). Both certify
the SAME SHAPE -- one launch in the ``update_H`` slot that computes the certified
magnetic constitutive at the thread's own cell AND at every backward neighbour the
curl taps, into launch-local WRITE-ONLY scratch, then runs the certified beta ``step_D``
body with its twelve magnetic reads redirected, with ``D``/``fu_D`` stepped in place,
and rotates the ``H``/``f_w_H`` bindings against the scratch afterwards -- over two
storage classes whose arithmetic differs in exactly the places
``lanes/cyl_round/cupy_probe/FINDINGS.md`` measured.

Everything that does not depend on the storage class lives here ONCE: the five
arrangements, the lock-step driver walk, the launch counters, the purity/race ledger,
the ghost-observability derivation, the block-size and seed-scale sweeps, the
driver-level sync and withdraw legs, the arbitration leg (as shipped AND with the
wiring rows patched in-process), the compiler leg, the corpus lift (parent and child)
and the runner. A second transcription of a launch counter is a second place for a
counter to stop counting.

WHAT A FAMILY SUPPLIES is an :class:`Adapter`: the module, the fixtures, how to seed a
volume of its storage, the certified single-slot launchers and the released
neighbouring pairs it is measured against, the armed mutations with their null
controls, and the three host legs whose text is the family's own (transcription,
refusal, spelling).

WHAT BOTH FAMILIES ADD THAT NO OTHER H->D SEAM HAS is the BETA PARTNER. The beta term
consumes the SAME-CELL magnetic snapshot (stepping.py:437), which the array path takes
AFTER ``update_H``; welded, that snapshot must be this launch's own register. It is a
site no sibling H->D gate has, and ``beta_partner_reads_stale_h`` is the mutation that
arms it -- one per family, in that family's own spelling.

THE EVIDENCE STANDARD is the brief's: bit identity as uint32 WORDS per COMPLETE driver
step against the array path (never ``allclose``; ``-0.0 == 0.0`` lies), every zero
beside a control that moves words, every count as N of D with D named, nothing
unmeasured credited.

POSITIONAL CLAUSES ARE RECORDED, NOT ASSERTED. Three released gates failed on the day
the composer tables were wired because they asserted where a product sits in
``fused_pairs`` rather than what it DOES there. The arbitration leg here asserts the
substantive claims -- refused BY NAME, the refusal names ``INSTALLABLE = False``, the
selection unchanged, both released neighbours keeping their slots -- and records the
positional ones (present/absent in the tables, rows removed afterwards) beside them.

Rule 7: one flushed line per case; the artifact is rewritten after every leg.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
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

# THE SIBLING GATES, imported for the machinery every scratch-output weld gate on this
# track shares. ONE copy of a launch counter, of a word comparator, of a policy
# installer, of a driver capture.
import cylindrical_hd_gate_common as cyl  # noqa: E402
import gate_cuda_fused_hd_pair as hd_gate  # noqa: E402
import gate_cuda_offdiag_stencil_welds as scratch_gate  # noqa: E402

from meep_gpu import stepping, withdraw_hoist  # noqa: E402
from meep_gpu.fastpath import SYNC_PASS_OWNERS, SYNC_UPDATE_H_PASS  # noqa: E402

log = probe.log
to_host = scratch_gate.to_host
differing = scratch_gate.differing
needle = scratch_gate.needle
MemoLaunchCounter = scratch_gate.MemoLaunchCounter
STATE_NAMES: Tuple[str, ...] = scratch_gate.STATE_NAMES
IMMUTABLE_PML: Tuple[str, ...] = scratch_gate.IMMUTABLE_PML
STEP_PASSES: Tuple[str, ...] = scratch_gate.STEP_PASSES
Shim = hd_gate.Shim

#: Complete DRIVER STEPS a fixture runs: the budget every hand-CUDA record is cut at.
STEPS = 60
#: The block sizes the schedule sweep runs.
BLOCK_SIZES: Tuple[int, ...] = cyl.BLOCK_SIZES
#: The steps at which the sync leg synchronizes the magnetic fields.
SYNC_STEPS: Tuple[int, ...] = cyl.SYNC_STEPS
#: The power-of-two seed scale the driver-level legs use. The solver is linear, so the
#: scale moves exponents and no rounding -- which :func:`leg_seed_scale` DRIVES rather
#: than asserts.
SEED_SCALE_BITS: int = cyl.SEED_SCALE_BITS
#: How many clean complete steps a lifted corpus row must reach to count.
LIFT_CLEAN_STEP_FLOOR: int = cyl.LIFT_CLEAN_STEP_FLOOR
#: The volumes that must MOVE for a band-class case to be non-vacuous.
BAND_MUST_MOVE: Tuple[str, ...] = cyl.BAND_MUST_MOVE
#: The expansion probe artifact the complex family's licence is read from, per policy.
EXPANSION_PROBES: Dict[str, str] = cyl.EXPANSION_PROBES

#: The arrangements every product case drives in lockstep. ``weld`` is the subject.
MODES: Tuple[str, ...] = ("array", "singles", "composition_today", "unfused", "weld")

#: The board the two cells are read from, and the census whose ``tests*.jsonl`` names
#: the MODULE each tests row lives in. Both are carried into the staging tree; a gate
#: that named an absent census would fail a leg in a way that reads like a measurement.
BOARD = "fusion_matrix_cuda_2026-09-07_wired"
CENSUS = "cuda_predicate_coverage_2026-09-07_cyl2"

LEG_GROUPS: Dict[str, Tuple[str, ...]] = {
    "host": ("driver_order", "transcription", "refusal", "ghost_observability",
             "spelling"),
    # ``compiler`` is LAST among the in-process device legs on purpose: it reads the
    # NVRTC observer and the policy's strip counters AFTER every kernel this process
    # will launch has been compiled.
    "device": ("product", "purity_ledger", "block_sizes", "seed_scale",
               "launch_structure", "arbitration", "sync", "withdraw", "byte_neutral",
               "mutation", "disarm", "compiler"),
    "corpus": ("lift",),
}
ALL_LEGS: Tuple[str, ...] = tuple(leg for group in LEG_GROUPS.values() for leg in group)


def case_rng(seed: int, label: str) -> "np.random.Generator":
    return cyl.case_rng(seed, label)


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------

class Adapter:
    """One family's answers to the questions the shared legs ask.

    Subclassed by each gate. Everything here is a HOOK with a documented meaning; the
    shared code never reaches into a family module except through these.
    """

    #: The family module (``meep_gpu.cuda_kernels.<...>``), set by the gate.
    family: Any = None
    #: ``True`` for complex64 storage: word views on every launch argument, an
    #: expansion licence and an arm, and two uint32 words per cell in every count.
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
    #: The float32 subnormal policy this run installs, and the expansion licence
    #: (complex only; ``None`` for the real family).
    policy: Optional[str] = None
    license: Optional[Dict[str, Any]] = None
    #: The arm the kernel compiles under (complex only; ``None`` for real).
    arm: Any = None
    #: ``{name: {"old", "new", "why", "live_on", ["also"], ["options"], ["null_under"],
    #: ["null_reason"]}}`` -- device-source rewrites.
    device_mutations: Dict[str, Dict[str, Any]] = {}
    #: The host-side defects (launcher, tables, order), applied by
    #: :meth:`host_mutation`.
    host_mutations: Tuple[str, ...] = ()
    #: The one armed edit required NOT to diverge: ``{"old", "new", "why"}``.
    byte_neutral: Dict[str, str] = {}
    #: The rows the wiring change adds to ``fused_pairs``, used by the arbitration leg.
    wiring_arms_row: Tuple[str, str] = ("", "")
    wiring_module: str = ""
    #: The released neighbouring products' labels, for the arbitration leg's report.
    released_neighbours: Tuple[str, str] = ("", "")
    #: The environment-variable stems the lift children read.
    env_stem: str = "MEEP_GPU_BETA_HD"

    # -- fixtures -------------------------------------------------------------
    def build(self, spec: Mapping[str, Any], value_class: str, rng):
        """``(fields, grid, pml, dtdx, plant)`` on the device, seeded for this class.

        A BETA GRID IS 2-D BY CONSTRUCTION AND THE BUILDER CHECKS IT. ``grid.beta`` is
        MEEP's out-of-plane 2-D wavevector: ``Grid._resolve_beta`` refuses it outside
        an effective-2-D Cartesian grid, so every fixture has ``nz == 1`` and a
        nonzero ``beta``, and a fixture whose beta silently failed to materialise
        would measure a beta-free kernel and credit this cell.
        """
        from meep_gpu.fields import Fields  # noqa: PLC0415
        from meep_gpu.grid import Grid, Mirror  # noqa: PLC0415
        from meep_gpu.pml import PML  # noqa: PLC0415

        grid = Grid(resolution=10.0, cell_size=spec["cell"],
                    courant=float(spec.get("courant", 0.35)), xp=cp,
                    dimensions=2, beta=float(spec["beta"]),
                    boundaries=spec["boundaries"],
                    k_point=tuple(spec["k_point"]),
                    symmetry=tuple(Mirror(axis, phase)
                                   for axis, phase in spec["symmetry"]))
        if float(getattr(grid, "beta", 0.0)) == 0.0:
            raise SystemExit(
                f"fixture {spec['label']!r} asked for beta={spec['beta']} and the grid "
                f"it built carries beta=0; a beta-free fixture measures a kernel this "
                f"cell does not contain")
        shape = tuple(int(n) for n in grid.shape)
        if shape[2] != 1:
            raise SystemExit(
                f"fixture {spec['label']!r} built shape {shape}; a beta grid is "
                f"effective-2-D and every corpus row of this cell has nz == 1")
        thickness = tuple((0, 0) if shape[axis] < 6
                          else (0, 2) if grid.is_mirrored(axis)
                          else (2, 2) for axis in range(3))
        pml = PML(grid=grid, thickness=thickness)
        fields = Fields(grid=grid, force_complex_fields=self.complex_storage)
        fields.enable_field_storage()
        fields.enable_pml_storage()
        # THREE DISTINCT VOLUMES, never one -- binding a single volume for all three
        # components is a defect a fixture that handed one array to all three could
        # not see. ``update_E`` consumes them; the weld binds none, and that is
        # exactly why they must be right: an update_E that stepped a different
        # equation would put the divergence on the weld.
        epsilon = {name: cp.full(shape, np.float32(value), cp.float32)
                   for name, value in (("Ex", 2.0), ("Ey", 2.5), ("Ez", 3.0))}
        inverse = {name: cp.full(shape, np.float32(1.0 / value), cp.float32)
                   for name, value in (("Ex", 2.0), ("Ey", 2.5), ("Ez", 3.0))}
        fields.set_epsilon_volumes(epsilon, inverse)
        self.seed_state(fields, grid, value_class, rng)
        plant = self.plant(fields, grid, value_class)
        return fields, grid, pml, float(grid.dt / grid.dx), plant

    def seed_state(self, fields, grid, value_class: str, rng) -> None:
        """Seed every stored volume.

        Complex planes are ASSIGNED, never ``re + 1j*im``: ``1j * im`` carries
        ``0.0 * im`` and destroys every negative zero the seed contains.
        """
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

    @staticmethod
    def real_plane(name: str, shape: Tuple[int, int, int], value_class: str,
                   rng) -> np.ndarray:
        if value_class in ("uniform", "planted_row0"):
            return rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
        if value_class == "subnormal_band":
            return probe.subnormal_band_hosts((name,), shape, rng)[name]
        raise ValueError(f"value class {value_class!r} is not one this gate seeds")

    def plant(self, fields, grid, value_class: str) -> Dict[str, Any]:
        """The family's own planted class on top of the seed. Reports what landed."""
        return {"planted": False}

    # -- the product ----------------------------------------------------------
    def predicate(self, fields, pml, grid, sources) -> Tuple[bool, str]:
        raise NotImplementedError

    def resolve(self, fields, grid, pml, dtdx: float, **overrides) -> Dict[str, Any]:
        """Everything one launch needs, with the gate's overrides applied."""
        raise NotImplementedError

    def assert_disjoint(self, fields, state) -> int:
        raise NotImplementedError

    def launch(self, fields, state, kernel=None, threads: Optional[int] = None):
        raise NotImplementedError

    def rotate(self, fields, state) -> None:
        raise NotImplementedError

    def kernel_source(self, source: Optional[str] = None) -> str:
        """The shipped device text for this run's arm."""
        raise NotImplementedError

    def compile(self, source: Optional[str] = None,
                options: Optional[Tuple[str, ...]] = None):
        raise NotImplementedError

    def kernel_name(self) -> str:
        return self.family.KERNEL_NAME

    def compile_options(self) -> Tuple[str, ...]:
        return tuple(self.family._COMPILE_OPTIONS)  # noqa: SLF001

    def source_digest(self) -> str:
        return self.family.source_digest()

    def clear_kernel_cache(self) -> int:
        return self.family._clear_kernel_cache()  # noqa: SLF001

    def boundary_codes(self, grid) -> Tuple[Any, Any]:
        """``(codes, refusal)`` -- the per-axis triple this launch binds.

        The CURL's own resolution, exactly as the launcher asks for it, so the purity
        ledger walks the same ``shift_dn`` branch the kernel takes.
        """
        raise NotImplementedError

    def attribution_source(self) -> Optional[str]:
        """The device text ``--fmad=false`` is attributed on.

        ``None`` means the shipped text. The COMPLEX family overrides this with the
        UNCONTRACTED arm, because its shipped arm spells every fusion as an explicit
        ``__fmaf_rn`` and leaves the flag nothing to remove -- so an attribution run on
        the shipped text would measure the harness rather than the spelling.
        """
        return None

    def attribution_arm_name(self) -> str:
        return str(self.arm)

    def host_mutation(self, name: str, spec: Mapping[str, Any],
                      steps: int) -> Dict[str, Any]:
        """Apply one HOST mutation and return ``{"caught": bool, ...}``."""
        raise NotImplementedError

    # -- the references -------------------------------------------------------
    def singles(self, fields, grid, pml, dtdx: float) -> Dict[str, Callable[[], None]]:
        """The certified single-slot launchers, one per slot of the slot path."""
        raise NotImplementedError

    def released_pairs(self, fields, grid, pml, dtdx: float):
        """``(magnetic module, electric module, run_magnetic, run_electric)``."""
        raise NotImplementedError

    # -- the driver-level legs ------------------------------------------------
    def driver_kwargs(self, spec: Mapping[str, Any]) -> Dict[str, Any]:
        return {"force_complex_fields": self.complex_storage,
                "dimensions": 2, "beta": float(spec["beta"])}

    def driver_source(self, integrated: bool) -> Dict[str, Any]:
        return {"component": "Ez", "center": (0.0, 0.0, 0.0), "size": (0.0, 0.0, 0.0),
                "frequency": 1.0,
                "source_type": "continuous" if integrated else "gaussian",
                **({"is_integrated": True} if integrated else {"fwidth": 0.2})}

    # -- the family's own host legs -------------------------------------------
    def leg_transcription(self) -> Dict[str, Any]:
        raise NotImplementedError

    def leg_refusal(self) -> Dict[str, Any]:
        raise NotImplementedError

    def leg_spelling(self) -> Dict[str, Any]:
        raise NotImplementedError

    def wiring_product_row(self) -> Dict[str, Any]:
        """The ``FUSED_PRODUCTS`` row the wiring change adds, for the arbitration leg."""
        raise NotImplementedError

    def composer_licenses(self) -> Optional[Dict[str, Any]]:
        """What ``arms.plan_step`` needs to reach this family's predicates."""
        return None

    def child_env(self) -> Dict[str, str]:
        """Extra environment the lift CHILDREN need, resolved AFTER the policy install.

        Called by :func:`run_campaign` at the moment the lift leg runs rather than at
        parser time, because on the complex family the value is the expansion licence
        the run installs -- and a child handed a licence resolved before the install
        would be handed ``null``.
        """
        return {}


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def state_of(fields) -> Dict[str, Any]:
    return {name: getattr(fields, name) for name in STATE_NAMES
            if getattr(fields, name, None) is not None}


def frozen(fields) -> Dict[str, np.ndarray]:
    return {name: to_host(array).copy() for name, array in state_of(fields).items()}


def compare(left: Mapping[str, np.ndarray],
            right: Mapping[str, np.ndarray]) -> Dict[str, int]:
    """Differing uint32 WORDS per volume. Never ``allclose``: -0.0 == 0.0 lies."""
    out: Dict[str, int] = {}
    for name in sorted(set(left) | set(right)):
        n = differing(left[name], right[name])
        if n:
            out[name] = int(n)
    return out


def word_total(snapshot: Mapping[str, np.ndarray]) -> int:
    """uint32 words in one snapshot. complex64 is TWO words per cell."""
    return int(sum(int(np.asarray(array).size)
                   * (2 if np.iscomplexobj(array) else 1)
                   for array in snapshot.values()))


def read_only_snapshot(pml) -> Dict[str, np.ndarray]:
    return {name: to_host(array).copy() for name in IMMUTABLE_PML
            if (array := getattr(pml, name, None)) is not None}


def read_only_drift(before: Mapping[str, np.ndarray], pml) -> List[str]:
    return sorted(name for name, saved in before.items()
                  if differing(saved, to_host(getattr(pml, name))))


def run_pass(name: str, fields, pml) -> None:
    scratch_gate.run_pass(name, fields, pml)


def operand_census(snapshot: Mapping[str, np.ndarray]) -> Dict[str, int]:
    return cyl.operand_census(snapshot)


# ---------------------------------------------------------------------------
# The arrangements
# ---------------------------------------------------------------------------

class Arrangement:
    """One way of performing a complete driver step, and its own launch counter."""

    __slots__ = ("name", "fields", "grid", "pml", "dtdx", "step", "launches", "state")

    def __init__(self, name: str, fields, grid, pml, dtdx: float) -> None:
        self.name = name
        self.fields, self.grid, self.pml, self.dtdx = fields, grid, pml, dtdx
        self.launches = 0
        self.state: Any = None

    def run_step(self) -> None:  # pragma: no cover - replaced per arrangement
        raise NotImplementedError


def make_arrangement(adapter: Adapter, mode: str, fields, grid, pml, dtdx: float, *,
                     kernel: Optional[Any] = None, rotate: bool = True,
                     threads: Optional[int] = None,
                     state_overrides: Optional[Dict[str, Any]] = None) -> Arrangement:
    """One of :data:`MODES` (or ``weld_composed``), against a seeded engine."""
    arrangement = Arrangement(mode, fields, grid, pml, dtdx)

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
                else:
                    run_pass(name, fields, pml)
    elif mode == "composition_today":
        magnetic, electric, run_magnetic, run_electric = adapter.released_pairs(
            fields, grid, pml, dtdx)
        # EACH PAIR LAUNCHES AT THE POSITION OF ITS OWN FIRST REPLACED PASS and every
        # pass it names is skipped; everything else runs on the array path in the
        # driver's own order. Written as a walk rather than as two slices because a
        # released pair's REPLACES is not necessarily contiguous.
        absorbed = set(magnetic.REPLACES) | set(electric.REPLACES)
        first_of = {magnetic.REPLACES[0]: ("magnetic", run_magnetic),
                    electric.REPLACES[0]: ("electric", run_electric)}
        arrangement.state = {"uncarried_passes": [name for name in STEP_PASSES
                                                  if name not in absorbed]}

        def step() -> None:
            for name in STEP_PASSES:
                entry = first_of.get(name)
                if entry is not None:
                    label, run = entry
                    report = run()
                    if not report.get("launched"):
                        raise SystemExit(
                            f"the released {label} pair refused a fixture the H->D "
                            f"weld admits: {report.get('reason')}")
                    arrangement.launches += 1
                if name in absorbed:
                    continue
                run_pass(name, fields, pml)
    elif mode in ("weld", "weld_composed"):
        state = adapter.resolve(fields, grid, pml, dtdx, **(state_overrides or {}))
        adapter.assert_disjoint(fields, state)
        arrangement.state = state
        block = (adapter.family._FUSED_THREADS if threads is None  # noqa: SLF001
                 else int(threads))
        neighbours = (adapter.singles(fields, grid, pml, dtdx)
                      if mode == "weld_composed" else None)

        def step() -> None:
            for name in STEP_PASSES:
                if name == "step_D":
                    continue           # absorbed by the launch at update_H
                if name != "update_H":
                    if neighbours is not None and name in ("step_B", "update_E"):
                        neighbours[name]()
                        arrangement.launches += 1
                    else:
                        run_pass(name, fields, pml)
                    continue
                adapter.launch(fields, state, kernel, block)
                arrangement.launches += 1
                if rotate:
                    adapter.rotate(fields, state)
    else:
        raise ValueError(f"unknown arrangement {mode!r}")

    arrangement.step = step  # type: ignore[assignment]
    return arrangement


def drive(adapter: Adapter, spec: Mapping[str, Any], value_class: str, steps: int, *,
          modes: Sequence[str] = MODES, kernel: Optional[Any] = None,
          rotate: bool = True, threads: Optional[int] = None,
          scale_bits: int = 0,
          state_overrides: Optional[Dict[str, Any]] = None,
          progress: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """Step every arrangement side by side from ONE seed, compared per COMPLETE step."""
    label = f"{spec['label']}/{value_class}"
    engines: Dict[str, Arrangement] = {}
    plant: Dict[str, Any] = {}
    for mode in modes:
        fields, grid, pml, dtdx, plant = adapter.build(
            spec, value_class, case_rng(adapter.seed, label))
        if scale_bits:
            _scale_state(fields, scale_bits)
        engines[mode] = make_arrangement(
            adapter, mode, fields, grid, pml, dtdx,
            kernel=kernel if mode == "weld" else None,
            rotate=rotate if mode == "weld" else True,
            threads=threads if mode == "weld" else None,
            state_overrides=state_overrides if mode.startswith("weld") else None)
    reference = engines[modes[0]]
    seeded = frozen(reference.fields)
    for mode in modes[1:]:
        mismatch = compare(seeded, frozen(engines[mode].fields))
        if mismatch:
            raise SystemExit(f"{label}: {mode} was not seeded identically: {mismatch}")

    read_only = read_only_snapshot(reference.pml)
    census = operand_census({name: to_host(value)
                             for name, value in state_of(reference.fields).items()})
    first: Dict[str, Optional[Dict[str, Any]]] = {mode: None for mode in modes[1:]}
    compared = 0
    started = time.perf_counter()
    with MemoLaunchCounter() as counter:
        for step in range(steps):
            for mode in modes:
                engines[mode].step()
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

    final = frozen(engines[modes[-1]].fields)
    must_move = STATE_NAMES if value_class != "subnormal_band" else BAND_MUST_MOVE
    unmoved = sorted(name for name in must_move
                     if name in seeded and not differing(seeded[name], final[name]))
    identical = {mode: first[mode] is None for mode in modes[1:]}
    drift = read_only_drift(read_only, reference.pml)
    return {
        "label": label, "spec": spec["label"], "value_class": value_class,
        "beta": float(spec["beta"]), "arm": str(adapter.arm),
        "modes": list(modes), "steps_requested": steps, "steps_compared": compared,
        "words_compared": word_total(seeded) * compared * max(1, len(modes) - 1),
        "identical": identical, "first_divergence": first,
        "operand_census": census, "plant": plant,
        "read_only_drift": drift,
        "volumes_that_did_not_move": unmoved,
        "memo_launches": memo_total, "memo_named": memo_named,
        "weld_launches": engines["weld"].launches if "weld" in engines else None,
        "seconds": round(time.perf_counter() - started, 1),
        "passed": bool(all(identical.values()) and not unmoved and not drift
                       and ("weld" not in engines
                            or engines["weld"].launches > 0)),
    }


def _scale_state(fields, scale_bits: int) -> None:
    """Multiply every stored volume by ``2**scale_bits`` -- exponents only."""
    scale = np.float32(2.0) ** int(scale_bits)
    for name in STATE_NAMES:
        array = getattr(fields, name, None)
        if array is None:
            continue
        array *= scale


# ---------------------------------------------------------------------------
# Host leg: the driver's own seam
# ---------------------------------------------------------------------------

def leg_driver_order(adapter: Adapter) -> Dict[str, Any]:
    """The seam this product spans, READ from the driver rather than asserted here."""
    family = adapter.family
    source = (Path(_REPO_API) / "meep_gpu" / "driver.py").read_text(encoding="utf-8")
    order = list(STEP_PASSES)
    span = list(family.REPLACES)
    try:
        start = order.index(span[0])
    except ValueError:
        start = -1
    adjacent = start >= 0 and order[start:start + len(span)] == span
    withdraw_line = "getattr(source, \"withdraw\""
    record = {
        "driver_step_passes": order, "span": span,
        "the_span_is_adjacent_in_the_driver_step": adjacent,
        "seam": family.SEAM,
        "seam_is_the_withdraw_hoist_seam": family.SEAM == withdraw_hoist.SEAM,
        "the_driver_runs_an_electric_withdraw_in_this_seam": withdraw_line in source,
        "carries_deposit_repair": bool(family.CARRIES_DEPOSIT_REPAIR),
        "hoists_the_withdraw": bool(family.HOISTS_THE_WITHDRAW),
        "why_no_deposit_repair": (
            "nothing is INJECTED between the update_H and step_D consults; the "
            "electric injection is one seam later and the magnetic one is one seam "
            "earlier, so there is no deposit in this seam to bracket"),
    }
    record["passed"] = bool(
        adjacent and record["seam_is_the_withdraw_hoist_seam"]
        and record["the_driver_runs_an_electric_withdraw_in_this_seam"]
        and not record["carries_deposit_repair"]
        and not record["hoists_the_withdraw"])
    return record


# ---------------------------------------------------------------------------
# Host leg: is the metallic ghost observable at all?
# ---------------------------------------------------------------------------

_TAP_CALL = re.compile(
    r"shift_dn_recompute\((\d+),\s*idx,\s*([ijk]),\s*n[xyz],\s*s[xyz],\s*bc_([xyz])")
#: The ownership mask's NEAR-PLANE clauses, per boundary code. ``cshift_dn``/``shift_dn``
#: returns its constant ghost on the ``ia == 0`` plane for EVERY non-periodic code, so
#: both codes have to be derived: a check that walked only ``BC_METALLIC`` would leave
#: the MIRROR arm's ghost unaccounted for and would make this leg's own null claim
#: narrower than the mutation it licenses.
_NEAR_PLANE_MASK = re.compile(
    r"if \(bc_([xyz]) == (BC_METALLIC|BC_MIRROR_PERIODIC) && ([ijk]) == 0\)")

#: The boundary codes whose ``ia == 0`` branch returns a CONSTANT rather than reading
#: memory. ``BC_PERIODIC`` is the complement and is the observable one, which is why
#: the periodic-wrap mutation is armed and these two are derived nulls.
_CONSTANT_GHOST_CODES: Tuple[str, ...] = ("BC_METALLIC", "BC_MIRROR_PERIODIC")


def leg_ghost_observability(adapter: Adapter) -> Dict[str, Any]:
    """Does any GHOST BOUNDARY VALUE reach ``step_D``'s output?

    THE ANSWER IS NO, AND IT IS DERIVED FROM THE EMITTED TEXT rather than tabulated.
    Each redirected tap shifts along one axis, and on the ``ia == 0`` plane of that
    axis the certified helper returns a CONSTANT -- an exact zero it never reaches the
    recompute for -- under every non-periodic boundary code. The ownership mask's own
    ``if (bc_A == CODE && coord == 0) curl = ZERO;`` lines say which (block, plane)
    pairs are zeroed. On the D side the two sets coincide exactly, FOR BOTH CODES,
    which is why an edit to what a ghost SERVES moves no byte -- a fact this gate
    derives once rather than rediscovering per mutation, and the reason the ghost
    mutation on this seam is a recorded NULL while its PERIODIC sibling is armed.

    BOTH CODES ARE DERIVED AND NOT JUST THE METALLIC ONE. A leg that covered only
    ``BC_METALLIC`` would license a null for half of what the ghost mutation edits,
    and the mirror half would be an unexplained uncaught defect. Measured on this
    kit's first mutation smoke (2026-09-07), where exactly that gap showed up as a
    complex-family mutation that could not be classified.
    """
    source = adapter.kernel_source()
    body = source.split(f"void {adapter.kernel_name()}(", 1)[-1]
    blocks = body.split("\n    }\n")
    rows: List[Dict[str, Any]] = []
    for index, block in enumerate(blocks):
        taps = _TAP_CALL.findall(block)
        if not taps:
            continue
        masked = {(axis, code, coordinate)
                  for axis, code, coordinate in _NEAR_PLANE_MASK.findall(block)}
        for component, coordinate, axis in taps:
            rows.append({
                "block": index, "tap_component": int(component),
                "shifts_along": axis, "coordinate": coordinate,
                "ghost_fires_on_plane": f"{coordinate} == 0",
                "masked_there": {
                    code: (axis, code, coordinate) in masked
                    for code in _CONSTANT_GHOST_CODES},
            })
        rows[-1]["mask_clauses_in_this_block"] = sorted(masked)
    findings: List[str] = []
    if len(rows) != adapter.family.HALO_TAPS:
        findings.append(
            f"the emitted text carries {len(rows)} redirected taps and the family "
            f"declares {adapter.family.HALO_TAPS}; the reader has probably stopped "
            f"matching and every clause below would pass vacuously")
    per_code = {code: all(row["masked_there"][code] for row in rows)
                for code in _CONSTANT_GHOST_CODES}
    for code, held in per_code.items():
        if not held:
            findings.append(
                f"at least one shifted tap feeds a curl the ownership mask does NOT "
                f"zero at the plane its {code} ghost fires on, so a {code} boundary "
                f"value IS observable in step_D's output -- and the ghost mutation is "
                f"then an ARMED defect that must be caught rather than a derived null")
    return {
        "taps": rows, "denominator": len(rows), "findings": findings,
        "codes_derived": list(_CONSTANT_GHOST_CODES),
        "every_tap_is_masked_where_its_ghost_fires": per_code,
        "what_this_licenses": (
            "that no edit to what a CONSTANT ghost serves -- metallic or mirror -- can "
            "move a byte of step_D's output, which is why this gate's ghost mutation "
            "is a DERIVED null and its PERIODIC sibling is the armed one. It licenses "
            "NOTHING about the periodic wrap, which is observable and IS armed"),
        "passed": bool(rows and not findings),
    }


# ---------------------------------------------------------------------------
# Device leg: the purity / race ledger
# ---------------------------------------------------------------------------

def leg_purity_ledger(adapter: Adapter,
                      specs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Of the curl's valid foreign taps, how many land on a cell ``update_H`` MOVED?

    THIS IS THE MEASUREMENT THAT MAKES THE IDENTITY RESULT MEAN ANYTHING. If the
    recomputed value equalled the stored one almost everywhere, the identity legs
    would be consistent with an IN-PLACE weld too and would license nothing about the
    design. Measured on the ARRAY PATH: ``update_H`` is run on a copy of the seeded
    engine and the stepped ``H`` compared to the pre-launch ``H`` at exactly the cells
    the certified ``shift_dn`` resolves for each of the six taps.

    The ghost arm (an exact zero) is excluded from the denominator BY NAME: it reads
    no memory in either arrangement and can never race.
    """
    #: (target, tap component, axis) for the six shifted reads, read off the certified
    #: step_D template: Dx reads Hz on y and Hy on z, Dy reads Hx on z and Hz on x,
    #: Dz reads Hy on x and Hx on y.
    taps: Tuple[Tuple[int, int, int], ...] = (
        (0, 2, 1), (0, 1, 2), (1, 0, 2), (1, 2, 0), (2, 1, 0), (2, 0, 1))
    rows: List[Dict[str, Any]] = []
    for spec in specs:
        fields, grid, pml, _dtdx, _plant = adapter.build(
            spec, "uniform", case_rng(adapter.seed, f"purity/{spec['label']}"))
        before = {name: to_host(getattr(fields, name)).copy()
                  for name in ("Hx", "Hy", "Hz")}
        stepping.update_H(fields, pml)
        after = {name: to_host(getattr(fields, name)).copy()
                 for name in ("Hx", "Hy", "Hz")}
        codes, refusal = adapter.boundary_codes(grid)
        if refusal is not None:
            rows.append({"spec": spec["label"], "refused": refusal})
            continue
        codes = tuple(int(code) for code in codes)
        shape = tuple(int(n) for n in grid.shape)
        total = ghost = moved_taps = 0
        for _target, component, axis in taps:
            name = ("Hx", "Hy", "Hz")[component]
            n = shape[axis]
            index = np.arange(n)
            # ``shift_dn``'s own branch, per coordinate on this axis: 0 = PERIODIC
            # wraps, everything else is the exact-zero ghost.
            source = np.where(index > 0, index - 1, n - 1 if codes[axis] == 0 else -1)
            valid = source >= 0
            plane_cells = int(np.prod(shape) // n)
            total += int(valid.sum()) * plane_cells
            ghost += int((~valid).sum()) * plane_cells
            moved = np.take(after[name] != before[name], source[valid], axis=axis)
            moved_taps += int(moved.sum())
        rows.append({
            "spec": spec["label"], "shape": list(shape), "boundary_codes": list(codes),
            "taps_that_read_memory": total,
            "taps_that_read_the_exact_zero_ghost": ghost,
            "taps_whose_recomputed_value_differs": moved_taps,
            "fraction": round(moved_taps / total, 6) if total else None,
        })
    measured = [row for row in rows if "taps_that_read_memory" in row]
    denominator = sum(row["taps_that_read_memory"] for row in measured)
    numerator = sum(row["taps_whose_recomputed_value_differs"] for row in measured)
    return {
        "rows": rows, "fixtures": len(measured),
        "taps_that_read_memory": denominator,
        "taps_whose_recomputed_value_differs": numerator,
        "fraction": round(numerator / denominator, 6) if denominator else None,
        "what_this_licenses": (
            "the recompute is LOAD-BEARING wherever this fraction is high: every such "
            "tap is a cell an in-place weld would have read at a schedule-decided "
            "time, so the identity legs are evidence about THIS design and not a "
            "result an in-place weld would have shared"),
        "passed": bool(denominator > 0 and numerator == denominator),
    }


# ---------------------------------------------------------------------------
# Device leg: launch structure
# ---------------------------------------------------------------------------

def leg_launch_structure(adapter: Adapter, spec: Mapping[str, Any],
                         steps: int) -> Dict[str, Any]:
    """Launches per step at the seam and over the whole step, every arrangement.

    TWO INDEPENDENT COUNTERS: each arrangement's own tally and a wrapper around the
    shipped compile memo, which sees every launch through every shipped route.

    THE HONEST NUMBER IS REPORTED WHETHER OR NOT IT FLATTERS THE WELD.
    """
    out: Dict[str, Any] = {"spec": spec["label"], "steps": steps, "modes": {}}
    for mode in MODES + ("weld_composed",):
        fields, grid, pml, dtdx, _plant = adapter.build(
            spec, "uniform", case_rng(adapter.seed, "launch"))
        arrangement = make_arrangement(adapter, mode, fields, grid, pml, dtdx)
        arrangement.step()      # warm the memo: an uncounted first compile
        cp.cuda.runtime.deviceSynchronize()
        arrangement.launches = 0
        with MemoLaunchCounter() as counter:
            for _ in range(steps):
                arrangement.step()
            cp.cuda.runtime.deviceSynchronize()
            named, total = counter.named(), counter.total
        out["modes"][mode] = {
            "arrangement_launches_per_step": arrangement.launches / steps,
            "memo_launches_per_step": total / steps,
            "memo_named": named,
            "counters_agree": arrangement.launches == total,
        }
    singles = out["modes"]["singles"]["memo_launches_per_step"]
    composed = out["modes"]["weld_composed"]["memo_launches_per_step"]
    today = out["modes"]["composition_today"]["memo_launches_per_step"]
    unfused = out["modes"]["unfused"]["memo_launches_per_step"]
    out["seam_launches"] = {
        "weld": 1.0, "certified_singles": singles,
        "note": ("the H->D seam ALONE, with every other pass on the array path: one "
                 "launch welded against the two the certified singles take. This is "
                 "the only number the weld improves, and it is a seam number rather "
                 "than a step number")}
    out["whole_step"] = {
        "weld_isolated": out["modes"]["weld"]["memo_launches_per_step"],
        "weld_composed": composed, "composition_today": today, "unfused": unfused,
        "weld_composed_minus_composition_today": round(composed - today, 6),
        "weld_composed_minus_unfused": round(composed - unfused, 6),
        "the_weld_costs_a_launch_against_the_composition_that_ships": composed > today,
        "the_weld_saves_a_launch_against_the_unfused_slots": composed < unfused,
        "what_this_says": (
            "over step_B .. update_E the composition that SHIPS is two launches (the "
            "two released beta pairs) and the composition with this product installed "
            "is three (step_B single, the weld, update_E single). The weld saves one "
            "launch against the four unfused slots and COSTS one against the "
            "composition that ships, which is INSTALLABLE_REASON reproduced as a "
            "MEASUREMENT rather than re-litigated. No timing is claimed either way"),
    }
    out["passed"] = bool(
        all(entry["counters_agree"] for entry in out["modes"].values())
        and out["seam_launches"]["certified_singles"] == 2.0
        and out["whole_step"]["the_weld_saves_a_launch_against_the_unfused_slots"]
        and out["whole_step"][
            "the_weld_costs_a_launch_against_the_composition_that_ships"])
    return out


# ---------------------------------------------------------------------------
# Device leg: the seed scale
# ---------------------------------------------------------------------------

def leg_seed_scale(adapter: Adapter, spec: Mapping[str, Any],
                   steps: int) -> Dict[str, Any]:
    """The seed scale is a change of EXPONENT and nothing else -- DRIVEN, not asserted.

    Every case runs at three scales and the verdict must be the same at each; the
    record carries the denormal census at each so a reader can see why the shipped
    scale is the one it is.
    """
    findings: List[str] = []
    rows: Dict[str, Any] = {}
    for bits in (0, 40, SEED_SCALE_BITS):
        record = drive(adapter, spec, "uniform", min(steps, 12),
                       modes=("array", "weld"), scale_bits=bits)
        rows[f"2^{bits}"] = {
            "bit_identical": record["identical"]["weld"],
            "first_divergence": record["first_divergence"]["weld"],
            "steps_compared": record["steps_compared"],
            "subnormal_words": record["operand_census"].get("subnormals"),
        }
    for label, row in rows.items():
        if not row["bit_identical"]:
            findings.append(f"the weld diverges at seed scale {label}")
    if rows[f"2^{SEED_SCALE_BITS}"]["subnormal_words"]:
        findings.append(
            f"the shipped scale 2^{SEED_SCALE_BITS} still leaves stored words in the "
            f"denormal band: {rows[f'2^{SEED_SCALE_BITS}']['subnormal_words']}")
    return {"passed": not findings, "findings": findings, "scales": rows,
            "spec": spec["label"],
            "what_this_licenses": (
                "that the identity is not an artefact of one amplitude. The solver is "
                "linear, so a power-of-two scale moves exponents and rounds nothing; "
                "this leg DRIVES that rather than resting on it")}


# ---------------------------------------------------------------------------
# Device legs that need a driver: sync and withdraw
# ---------------------------------------------------------------------------

def build_driver(adapter: Adapter, spec: Mapping[str, Any], *,
                 integrated: bool = False, seed: int = 20260907,
                 scale_bits: int = 0):
    """One seeded ``FdtdDriver`` on the device, with one source."""
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415

    boundaries = ({axis: kind for axis, kind in zip("xyz", spec["boundaries"])}
                  if spec["boundaries"] else None)
    driver = FdtdDriver(cell_size=spec["cell"], resolution=10.0,
                        courant=float(spec.get("courant", 0.35)),
                        boundaries=boundaries, k_point=tuple(spec["k_point"]),
                        symmetry=[{"direction": axis, "phase": phase}
                                  for axis, phase in spec["symmetry"]],
                        prefer_gpu=True, gpu_id=0, **adapter.driver_kwargs(spec))
    # THE ABSORBER GOES ON X AND Y ONLY, and that is forced rather than tidy: a beta
    # run is ``dimensions=2``, whose z axis is TRANSLATIONALLY INVARIANT, and
    # ``PML.__init__`` REFUSES a layer there by name -- an invariant axis has no face
    # to absorb at, and the one cell that stands for it would be entirely inside the
    # absorber. A scalar ``setup_pml(2)`` asks for all six faces and dies before the
    # first step. Measured on this kit's first driver smoke (2026-09-07): the sync and
    # withdraw legs both failed on that ValueError with nothing measured.
    driver.setup_pml({"x": 2, "y": 2})
    shape = tuple(int(n) for n in driver.grid.shape)
    driver.fields.set_epsilon_volumes(
        {name: cp.full(shape, np.float32(value), cp.float32)
         for name, value in (("Ex", 2.0), ("Ey", 2.5), ("Ez", 3.0))},
        {name: cp.full(shape, np.float32(1.0 / value), cp.float32)
         for name, value in (("Ex", 2.0), ("Ey", 2.5), ("Ez", 3.0))})
    driver.add_source(adapter.driver_source(integrated))
    rng = np.random.default_rng(seed)
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
            host = ((0.37 * rng.standard_normal(array.shape)).astype(np.float32)
                    * scale)
        array[...] = cp.asarray(np.ascontiguousarray(host))
    driver.invalidate_fast_path()
    return driver


def weld_plan_for(adapter: Adapter, driver, *, hoisted: bool = False,
                  placement: str = withdraw_hoist.BEFORE_UPDATE_H):
    """``(plans, book)`` -- the weld installed at ``update_H``, ``step_D`` absorbed."""
    family = adapter.family
    fields, pml, grid = driver.fields, driver.pml, driver.grid
    state = adapter.resolve(fields, grid, pml, float(grid.dt / grid.dx))
    book = {"launches": 0, "withdrawn": 0}

    def run() -> None:
        if hoisted:
            book["withdrawn"] += withdraw_hoist.hoist(
                fields, driver._sources, span=family.REPLACES,  # noqa: SLF001
                placement=placement)
        adapter.launch(fields, state)
        book["launches"] += 1
        adapter.rotate(fields, state)

    return {"update_H": run, "step_D": lambda: None}, book


def driver_state(driver) -> Dict[str, np.ndarray]:
    return {name: to_host(array).copy()
            for name in STATE_NAMES
            if (array := getattr(driver.fields, name, None)) is not None}


def leg_sync(adapter: Adapter, spec: Mapping[str, Any], steps: int) -> Dict[str, Any]:
    """The magnetic half-step's channel, with the product FORCE-INSTALLED.

    ``synchronize_magnetic_fields`` repeats the magnetic half of ``step`` and then
    UNDOES it, but its backup is magnetic names only. ``D``, ``fu_D`` and ``f_cond_D``
    are in NEITHER list, so a product spanning ``update_H`` and ``step_D`` that
    answered the ``update_H_synchronize`` consult would advance the electric state
    inside a window nothing can undo.

    THIS LEG IS THE CITATION FOR THE COMPOSER REFUSAL and carries its own null control.
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
    for label, answers, synchronizes in (("no_sync", False, False),
                                         ("guarded", False, True),
                                         ("hazard", True, True)):
        driver = build_driver(adapter, spec)
        plans, book = weld_plan_for(adapter, driver)
        driver._fast_path = Shim(plans, answers_the_sync_consult=answers)  # noqa: SLF001
        driver._fast_path_stale = False  # noqa: SLF001
        for step in range(steps):
            driver.step()
            if synchronizes and (step + 1) in SYNC_STEPS:
                driver.synchronize_magnetic_fields()
                driver.restore_magnetic_fields()
        cp.cuda.runtime.deviceSynchronize()
        baselines[label] = driver_state(driver)
        arms_out[label] = {"launches": book["launches"],
                           "consults": dict(driver._fast_path.calls)}  # noqa: SLF001
    electric = tuple(f"{stem}{axis}" for stem in ("D", "fu_D") for axis in "xyz")
    for label in ("guarded", "hazard"):
        moved = compare(baselines["no_sync"], baselines[label])
        arms_out[label]["differing_volumes"] = moved
        arms_out[label]["differing_electric_volumes"] = {
            name: n for name, n in moved.items() if name in electric}
    facts["arms"] = arms_out
    facts["the_guarded_arm_is_identical"] = not arms_out["guarded"]["differing_volumes"]
    facts["the_hazard_arm_diverges_in_D"] = bool(
        arms_out["hazard"]["differing_electric_volumes"])
    facts["what_this_cites"] = (
        "a product spanning update_H and step_D advances D and fu_D inside "
        "synchronize_magnetic_fields, whose backup list holds neither, so the restore "
        "cannot reach them. The containment rule in FastPathPlan.dispatch is what "
        "stops it, and this leg is the measurement behind that rule for THIS product")
    facts["passed"] = bool(facts["D_is_in_neither_backup_list"]
                           and facts["the_guarded_arm_is_identical"]
                           and facts["the_hazard_arm_diverges_in_D"])
    return facts


def leg_withdraw(adapter: Adapter, spec: Mapping[str, Any],
                 steps: int) -> Dict[str, Any]:
    """The seam's one pass, on a device, with both of the campaign's null controls."""
    family = adapter.family
    # THE SEED IS NOT SCALED HERE: this leg compares a subtraction of the SOURCE's
    # standing dipole, whose magnitude is the source amplitude. Against a scaled state
    # that subtraction is below the float32 resolution of every word it touches and is
    # swallowed whole -- which silently stops the two null controls controlling
    # anything.
    probe_driver = build_driver(adapter, spec, integrated=True, scale_bits=0)
    sources = tuple(probe_driver._sources)  # noqa: SLF001
    standing = withdraw_hoist.standing_withdraws(sources)
    covered, reason = adapter.predicate(
        probe_driver.fields, probe_driver.pml, probe_driver.grid, sources)
    hoistable, hoist_reasons = withdraw_hoist.hoistable(
        probe_driver.fields, sources, span=family.REPLACES)

    arms_out: Dict[str, Any] = {}
    for label in ("hoisted", "not_hoisted", "after_step_D"):
        reference = build_driver(adapter, spec, integrated=True, scale_bits=0)
        driver = build_driver(adapter, spec, integrated=True, scale_bits=0)
        if label == "after_step_D":
            plans, book = weld_plan_for(adapter, driver)
            electric = tuple(driver._sources)  # noqa: SLF001
            inner = plans["update_H"]

            def _after(_inner=inner, _fields=driver.fields, _electric=electric):
                _inner()
                for source in _electric:
                    getattr(source, "withdraw", lambda *_a: None)(_fields)

            plans = {"update_H": _after, "step_D": lambda: None}
        else:
            plans, book = weld_plan_for(adapter, driver,
                                        hoisted=(label == "hoisted"))
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
        arms_out[label] = {"first_divergence": first, "identical": first is None,
                           "steps_compared": compared,
                           "withdrawn_on_the_last_step": book.get("withdrawn"),
                           "launches": book.get("launches")}
    record = {
        "spec": spec["label"], "steps": steps,
        "predicate_refuses_this_row": not covered, "predicate_reason": reason,
        "predicate_names_the_withdraw": "standing integrated" in str(reason),
        "standing_withdraws": len(standing),
        "hoistable": bool(hoistable), "hoistable_reasons": list(hoist_reasons),
        "arms": arms_out, "campaign": dict(withdraw_hoist.MEASUREMENT),
        "what_this_licenses": (
            "the LeadingWithdrawPlan placement on this device against the array path, "
            "with both of the array-path campaign's null controls diverging. It does "
            "NOT flip HOISTS_THE_WITHDRAW: the product still refuses these rows by "
            "name, and the flag moves only together with INSTALLABLE and the wiring"),
    }
    record["passed"] = bool(
        record["predicate_refuses_this_row"] and record["predicate_names_the_withdraw"]
        and record["hoistable"] and standing
        and arms_out["hoisted"]["identical"]
        and not arms_out["not_hoisted"]["identical"]
        and not arms_out["after_step_D"]["identical"])
    return record


# ---------------------------------------------------------------------------
# Device leg: arbitration, as shipped AND with the wiring rows patched in-process
# ---------------------------------------------------------------------------

def wired_in_process(adapter: Adapter):
    """Context manager: the product's rows in ``fused_pairs``, then removed.

    NOTHING ON DISK CHANGES. The rows ``wiring.patch`` would add are installed in the
    live dicts, the composer is asked, and the rows are removed -- so the composer can
    be asked what it does WITH this product in its tables on a tree where it is not.
    IDEMPOTENT ON A WIRED TREE: a row already present is left alone and left in place,
    which is what makes this leg survive the wiring round rather than invert with it.
    """
    from meep_gpu.cuda_kernels import fused_pairs  # noqa: PLC0415

    family = adapter.family

    @contextlib.contextmanager
    def scope():
        added_product = family.FAMILY not in fused_pairs.FUSED_PRODUCTS
        added_arm = family.FAMILY not in fused_pairs.FUSED_PAIR_ARMS
        if added_product:
            fused_pairs.FUSED_PRODUCTS[family.FAMILY] = adapter.wiring_product_row()
        if added_arm:
            fused_pairs.FUSED_PAIR_ARMS[family.FAMILY] = adapter.wiring_arms_row
        try:
            yield {"product_row_added_in_process": added_product,
                   "arm_row_added_in_process": added_arm}
        finally:
            if added_product:
                fused_pairs.FUSED_PRODUCTS.pop(family.FAMILY, None)
            if added_arm:
                fused_pairs.FUSED_PAIR_ARMS.pop(family.FAMILY, None)

    return scope()


def ask_the_composer(adapter: Adapter, fields, grid, pml, sources=()) -> Dict[str, Any]:
    from meep_gpu.cuda_kernels import arms  # noqa: PLC0415

    family = adapter.family
    plan = arms.plan_step(fields=fields, pml=pml, grid=grid, sources=sources,
                          fuse=True, licenses=adapter.composer_licenses(),
                          subnormal_policy=adapter.policy)
    selected = dict(getattr(plan, "selected", {}) or {})
    reasons = {name: list(value) for name, value
               in (getattr(plan, "reasons", {}) or {}).items()}
    mine = [text for key, texts in reasons.items() if family.FAMILY in key
            for text in texts]
    return {"selected": selected, "this_product_is_refused": bool(mine),
            "refusal_reasons": mine,
            "this_product_holds_no_slot": not any(
                family.FAMILY in str(value) for value in selected.values())}


def leg_arbitration(adapter: Adapter, spec: Mapping[str, Any]) -> Dict[str, Any]:
    """The COMPOSER, asked twice: AS SHIPPED and WITH the wiring rows patched in.

    THE POSITIONAL CLAUSES ARE RECORDED, NOT ASSERTED. Whether this product is present
    in ``FUSED_PRODUCTS`` on disk, and whether the in-process rows had to be added and
    were removed, are facts about a table that a wiring round INVERTS -- three released
    gates failed on exactly that. What is ASSERTED is what the composer DOES: the
    product is refused BY NAME, the refusal names ``INSTALLABLE = False``, the
    selection is unchanged by the wiring, this product holds no slot, and both
    released neighbours keep theirs.
    """
    from meep_gpu.cuda_kernels import fused_pairs  # noqa: PLC0415

    family = adapter.family
    fields, grid, pml, _dtdx, _plant = adapter.build(
        spec, "uniform", case_rng(adapter.seed, "arbitration"))
    shipped = ask_the_composer(adapter, fields, grid, pml)
    with wired_in_process(adapter) as added:
        wired = ask_the_composer(adapter, fields, grid, pml)
        wired["declared_uninstallable"] = fused_pairs._declared_uninstallable(  # noqa: SLF001
            family.FAMILY, fused_pairs.FUSED_PRODUCTS[family.FAMILY])
    holder = wired["selected"].get("update_H")
    record = {
        "spec": spec["label"],
        "as_shipped": shipped,
        "with_the_wiring_rows_patched_in_process": wired,
        # --- RECORDED, NOT ASSERTED: positional facts a wiring round inverts.
        "positional": {
            "product_row_had_to_be_added_in_process": added["product_row_added_in_process"],
            "arm_row_had_to_be_added_in_process": added["arm_row_added_in_process"],
            "the_rows_were_removed_afterwards": (
                added["product_row_added_in_process"]
                and family.FAMILY not in fused_pairs.FUSED_PRODUCTS),
            "the_product_is_absent_as_shipped": (
                not shipped["this_product_is_refused"]
                and shipped["this_product_holds_no_slot"]),
            "why_recorded_rather_than_asserted": (
                "these describe WHERE the product sits in the composer's tables, "
                "which the wiring round changes by design. Three gates failed on "
                "clauses of exactly this shape. The substantive claims are below"),
        },
        # --- ASSERTED: what the composer DOES.
        "the_wired_product_is_refused_by_name": wired["this_product_is_refused"],
        "the_refusal_names_installable": any(
            "INSTALLABLE = False" in text for text in wired["refusal_reasons"]),
        "the_selection_is_unchanged_by_the_wiring":
            shipped["selected"] == wired["selected"],
        "the_wired_product_holds_no_slot": wired["this_product_holds_no_slot"],
        "the_released_b_to_h_pair_keeps_update_H": bool(
            holder is not None and wired["selected"].get("step_B") == holder
            and family.FAMILY not in str(holder)),
        "the_released_d_to_e_pair_keeps_step_D": bool(
            wired["selected"].get("step_D") is not None
            and wired["selected"].get("step_D") == wired["selected"].get("update_E")
            and family.FAMILY not in str(wired["selected"].get("step_D"))),
        "released_neighbours": list(adapter.released_neighbours),
        "installable": bool(family.INSTALLABLE),
        "installable_reason": family.INSTALLABLE_REASON,
        "what_a_release_does_not_license": family.WHAT_A_RELEASE_DOES_NOT_LICENSE,
    }
    record["passed"] = bool(
        record["the_wired_product_is_refused_by_name"]
        and record["the_refusal_names_installable"]
        and record["the_selection_is_unchanged_by_the_wiring"]
        and record["the_wired_product_holds_no_slot"]
        and not record["installable"]
        and record["the_released_b_to_h_pair_keeps_update_H"]
        and record["the_released_d_to_e_pair_keeps_step_D"])
    return record


# ---------------------------------------------------------------------------
# The mutations
# ---------------------------------------------------------------------------

def mutation_needles_resolve(adapter: Adapter) -> Dict[str, Any]:
    """Does every armed needle -- and the byte-neutral one -- match EXACTLY ONCE?

    A HOST CHECK RUN BEFORE ANY DEVICE LEG, because a needle that matches zero times
    is a mutation that silently compiles the SHIPPED source and reports the defect as
    uncaught, and a needle that matches twice edits a site nobody chose. Both cost a
    dead campaign leg; both are cheap to catch here. Reported per needle with its site
    count, so a reader sees which one moved rather than a boolean.
    """
    source = adapter.kernel_source()
    rows: List[Dict[str, Any]] = []
    for name, spec in adapter.device_mutations.items():
        if spec.get("old") is None:
            rows.append({"needle": name, "kind": "compile_options", "sites": None,
                         "resolves": True})
            continue
        sites = source.count(spec["old"])
        row = {"needle": name, "kind": "device", "sites": sites,
               "resolves": sites == 1}
        extra = []
        for old, _new in spec.get("also", ()):
            extra.append(source.count(old))
        if extra:
            row["also_sites"] = extra
            row["resolves"] = row["resolves"] and all(n == 1 for n in extra)
        rows.append(row)
    sites = source.count(adapter.byte_neutral["old"])
    rows.append({"needle": "byte_neutral", "kind": "byte_neutral", "sites": sites,
                 "resolves": sites == 1})
    unresolved = [row["needle"] for row in rows if not row["resolves"]]
    return {"needles": rows, "denominator": len(rows), "unresolved": unresolved,
            "passed": not unresolved}


def mutated_source(adapter: Adapter, name: str) -> Optional[str]:
    spec = adapter.device_mutations[name]
    if spec["old"] is None:
        return None
    source = adapter.kernel_source()
    source = needle(source, spec["old"], spec["new"])
    for old, new in spec.get("also", ()):
        source = needle(source, old, new)
    return source


def leg_mutation(adapter: Adapter, specs: Sequence[Mapping[str, Any]],
                 steps: int) -> Dict[str, Any]:
    """Every armed defect, each required to DIVERGE on at least one scored fixture."""
    results: List[Dict[str, Any]] = []
    for name, spec in adapter.device_mutations.items():
        if spec.get("arm_only") and spec["arm_only"] != str(adapter.arm):
            results.append({"name": name, "kind": "device", "compiled": None,
                            "caught": False, "passed": True,
                            "skipped": f"armed only under {spec['arm_only']}"})
            continue
        caught_on: List[str] = []
        compiled, compile_error = True, None
        for case in specs:
            try:
                source = mutated_source(adapter, name)
                kernel = adapter.compile(source, spec.get("options"))
            except Exception as error:  # noqa: BLE001
                compiled, compile_error = False, repr(error)[:400]
                break
            for value_class in adapter.value_classes:
                record = drive(adapter, case, value_class, min(steps, 12),
                               modes=("array", "weld"), kernel=kernel)
                if not record["identical"]["weld"]:
                    caught_on.append(f"{case['label']}/{value_class}")
        entry = {"name": name, "kind": "device", "compiled": compiled,
                 "compile_error": compile_error, "why": spec["why"],
                 "live_on": spec["live_on"], "caught_on": caught_on,
                 "caught": bool(caught_on)}
        entry["passed"] = bool(compiled and caught_on)
        if adapter.policy in (spec.get("null_under") or ()):
            entry["predicted_null_under_this_policy"] = spec["null_reason"]
            entry["prediction_held"] = not caught_on
            entry["passed"] = bool(compiled)
        elif spec.get("predicted_null"):
            entry["predicted_null"] = spec["predicted_null"]
            entry["prediction_held"] = not caught_on
            entry["passed"] = bool(compiled)
        results.append(entry)

    for name in adapter.host_mutations:
        try:
            entry = adapter.host_mutation(name, specs[0], min(steps, 12))
            entry.setdefault("passed", entry.get("caught", False))
            results.append({"name": name, "kind": "host", "compiled": True, **entry})
        except Exception as error:  # noqa: BLE001
            results.append({"name": name, "kind": "host", "compiled": False,
                            "caught": False, "passed": False,
                            "error": repr(error)[:400]})
    attribution = _fmad_attribution(adapter, specs[0], min(steps, 8))
    adapter.clear_kernel_cache()
    armed = [entry for entry in results if not entry.get("skipped")]
    return {"mutations": results, "denominator": len(results), "armed": len(armed),
            "caught": sum(1 for entry in results if entry["caught"]),
            "predicted_null": [entry["name"] for entry in results
                               if entry.get("predicted_null")
                               or entry.get("predicted_null_under_this_policy")],
            "held": [entry["name"] for entry in results
                     if entry.get("prediction_held")],
            # A PREDICTED NULL THAT BIT IS NOT A DEFECT IN THE PRODUCT and is not
            # reported as one: it is the STRONGER outcome (a race twin that fired, a
            # policy-conditional defect that reached further than the prior finding
            # said). Reported separately from a prediction that was supposed to bite
            # and did not, which is the one a reader must act on.
            "predicted_null_that_bit": [
                entry["name"] for entry in results
                if entry.get("prediction_held") is False and entry.get("caught")],
            "predicted_to_bite_and_did_not": [
                entry["name"] for entry in results
                if entry.get("prediction_held") is False and not entry.get("caught")],
            "uncaught": [entry["name"] for entry in results if not entry["caught"]
                         and not entry.get("skipped")],
            "fmad_attribution": attribution,
            "passed": (all(entry["passed"] for entry in results)
                       and attribution["passed"])}


def _fmad_attribution(adapter: Adapter, spec: Mapping[str, Any],
                      steps: int) -> Dict[str, Any]:
    """Does ``--fmad=false`` reach NVRTC and change the emitted arithmetic?

    REPORTED, and required to bite ONLY where the family says it must. On the real
    family every multiply in the text is a plain ``*`` and the flag is load-bearing;
    on the complex family the shipped arm spells its fusions as explicit ``__fmaf_rn``
    and the flag is null there, so the attribution runs on the UNCONTRACTED arm and
    that is where the difference must appear. Either way, a null whose real cause was
    a flag that never reached NVRTC would fail here rather than be argued away.
    """
    states: Dict[str, Dict[str, np.ndarray]] = {}
    for label, options in (("guarded", adapter.compile_options()), ("default", ())):
        kernel = adapter.compile(adapter.attribution_source(), options)
        fields, grid, pml, dtdx, _plant = adapter.build(
            spec, "uniform", case_rng(adapter.seed, "fmad"))
        arrangement = make_arrangement(adapter, "weld", fields, grid, pml, dtdx,
                                       kernel=kernel)
        for _ in range(steps):
            arrangement.step()
        cp.cuda.runtime.deviceSynchronize()
        states[label] = frozen(fields)
    moved = compare(states["guarded"], states["default"])
    return {
        "spec": spec["label"], "steps": steps,
        "arm_under_attribution": adapter.attribution_arm_name(),
        "differing_volumes": moved,
        "the_flag_changes_the_answer_on_this_arm": bool(moved),
        "what_this_attributes": (
            "--fmad=false DOES reach NVRTC and DOES change the emitted arithmetic on "
            "a source that carries contractible multiply-adds -- so a null on the "
            "shipped arm is attributable to that arm's explicit __fmaf_rn spellings "
            "and not to a build option that never arrived"),
        "passed": bool(moved),
    }


def leg_byte_neutral(adapter: Adapter, spec: Mapping[str, Any],
                     steps: int) -> Dict[str, Any]:
    """The one armed edit required NOT to diverge."""
    source = needle(adapter.kernel_source(), adapter.byte_neutral["old"],
                    adapter.byte_neutral["new"])
    kernel = adapter.compile(source)
    record = drive(adapter, spec, "uniform", min(steps, 12),
                   modes=("array", "weld"), kernel=kernel)
    adapter.clear_kernel_cache()
    return {"spec": spec["label"], "identical": record["identical"]["weld"],
            "first_divergence": record["first_divergence"]["weld"],
            "why": adapter.byte_neutral["why"],
            "passed": bool(record["identical"]["weld"])}


def leg_disarm(adapter: Adapter, spec: Mapping[str, Any],
               steps: int) -> Dict[str, Any]:
    """The identical harness on the SHIPPED bytes, required not to diverge."""
    record = drive(adapter, spec, "uniform", min(steps, 12), modes=("array", "weld"))
    return {"spec": spec["label"], "identical": record["identical"]["weld"],
            "first_divergence": record["first_divergence"]["weld"],
            "passed": bool(record["identical"]["weld"])}


# ---------------------------------------------------------------------------
# Device leg: the compiler
# ---------------------------------------------------------------------------

def leg_compiler(adapter: Adapter, policy: Optional[str],
                 at_install: Mapping[str, Any]) -> Dict[str, Any]:
    """Did THIS process compile the kernels it ran, and did the policy reach NVRTC?"""
    report = probe.nvrtc_binary_report()
    outer = probe.subnormal_policy_stamp(_REPO_API)
    stamp = dict(outer.get("stamp") or {})
    stamp.setdefault("policy", outer.get("policy"))
    now = cyl.cupy_cache_snapshot()
    try:
        family_source = adapter.source_digest()
    except Exception as exc:  # noqa: BLE001
        family_source = f"unavailable: {exc!r}"
    observed = int(report.get("nvrtc_calls_observed") or 0)
    counters = {"nvrtc_calls": stamp.get("nvrtc_calls"),
                "ftz_removed": stamp.get("ftz_removed")}
    checks: Dict[str, bool] = {
        "the_compiler_was_exercised": observed > 0,
        "every_binary_this_process_ran_was_compiled_by_it":
            at_install.get("entries") == 0 and at_install.get("error") is None,
    }
    if policy == "keep":
        checks["the_strip_reached_every_compile"] = bool(
            (counters["ftz_removed"] or 0) > 0
            and counters["nvrtc_calls"] == observed
            and not report.get("any_ftz_true_reached_nvrtc"))
    elif policy == "flush":
        checks["ftz_true_reached_every_compile"] = bool(
            report.get("all_ftz_true_reached_nvrtc")
            and (counters["ftz_removed"] or 0) == 0)
    else:
        checks["a_policy_was_requested"] = False
    observed_sources = {entry.get("source_sha256")
                        for entry in (report.get("observations") or [])}
    shipped = hashlib.sha256(adapter.kernel_source().encode("utf-8")).hexdigest()
    return {
        "policy_requested": policy, "policy_stamp": stamp.get("policy"),
        "counters_after_every_in_process_leg": counters,
        "observed": {key: report.get(key) for key in
                     ("nvrtc_calls_observed", "distinct_binaries", "distinct_sources",
                      "any_ftz_true_reached_nvrtc", "all_ftz_true_reached_nvrtc")},
        "cache": {"dir": at_install.get("dir"),
                  "entries_at_install": at_install.get("entries"),
                  "preexisting_at_install": at_install.get("preexisting"),
                  "entries_after_every_in_process_leg": now["entries"]},
        "family_source_sha256": family_source,
        "shipped_source_sha256": shipped,
        "the_shipped_source_was_observed_at_nvrtc": shipped in observed_sources,
        "checks": checks,
        "what_this_licenses": (
            "that the binaries this process executed were compiled by this process, "
            "under the installed policy, with the policy's option treatment observed "
            "at the NVRTC seam. The lift children install the policy themselves "
            "before their first compile and stamp it before and after the drive"),
        "passed": all(checks.values()),
    }


# ---------------------------------------------------------------------------
# The corpus lift
# ---------------------------------------------------------------------------

def runtime_reasons() -> List[str]:
    """Why THIS process cannot evaluate this battery at all, or ``[]``."""
    if cp is None:
        return ["cupy is not importable in this interpreter"]
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception as error:  # noqa: BLE001
        return [f"no CUDA device reachable from this interpreter: {error!r}"]
    return []


def _load_jsonl(path: Path) -> List[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def lift_basis(adapter: Adapter, results: Path) -> Tuple[List[dict], Dict[str, Any]]:
    """The rows the standing BOARD puts in this product's cell.

    DERIVED from the board's own ``h_to_d_seam.instances`` -- the arms the census chose
    for each row -- never from a list here. The MODULE a tests row lives in is the
    CENSUS's fact and is joined from its ``tests*.jsonl``; a tests row with no module
    in either record is REFUSED BY NAME rather than guessed at.
    """
    board = results / BOARD / "fusion_matrix_cuda.json"
    if not board.is_file():
        raise SystemExit(f"the lift leg needs {board}")
    census = results / CENSUS
    modules: Dict[str, str] = {}
    replay_case: Dict[str, str] = {}
    for leg in ("tests", "tests_param_matched", "tests_param"):
        for record in _load_jsonl(census / f"{leg}.jsonl"):
            case = record.get("case") or record.get("row")
            label = f"tests:{record.get('row') or case}"
            if case and record.get("module"):
                modules[label] = record["module"]
            if case and record.get("row") and case != record["row"]:
                replay_case[label] = case
    for record in _load_jsonl(census / "examples.jsonl"):
        label = f"examples:{record.get('row')}"
        if record.get("module"):
            modules[label] = record["module"]
    board_json = json.loads(board.read_text(encoding="utf-8"))
    instances = (board_json.get("h_to_d_seam") or {}).get("instances") or []
    rows: List[dict] = []
    for record in instances:
        if (record.get("update_H"), record.get("step_D")) != tuple(adapter.cell_arms):
            continue
        label = record["row"]
        leg, _, row = label.partition(":")
        rows.append({
            "label": label, "leg": leg, "row": row,
            "cell": list(adapter.cell_arms),
            "module": modules.get(label),
            "replay_case": replay_case.get(label, row),
            "interpreter": sys.executable,
            "bucket": record.get("bucket"),
            "withdraw_in_seam": bool(record.get("withdraw_in_seam")),
            "integrated_electric_sources":
                int(record.get("integrated_electric_sources") or 0),
        })
    facts = {
        "board": BOARD, "census": CENSUS, "cell": list(adapter.cell_arms),
        "rows_total": len(rows),
        "rows_with_a_standing_withdraw": sorted(row["label"] for row in rows
                                                if row["withdraw_in_seam"]),
        "module_join": {
            "source": f"{CENSUS}/(tests*|examples).jsonl",
            "cases_resolved": len(modules),
            "rows_without_a_module": sorted(row["label"] for row in rows
                                            if not row["module"])},
    }
    return rows, facts


def child_progress(adapter: Adapter, message: str) -> None:
    path = os.environ.get(f"{adapter.env_stem}_PROGRESS")
    label = os.environ.get(f"{adapter.env_stem}_LABEL", "?")
    line = f"[{time.strftime('%H:%M:%S')}] {label} {message}"
    print(line, flush=True)
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()


def drive_lifted_row(adapter: Adapter, driver: Any, steps: int) -> Dict[str, Any]:
    """One lifted corpus row, driven through the driver's own consult order.

    ONE DRIVER, CAPTURED AND RESTORED, rather than two engines: a lifted row carries
    its own sources, monitors and material. Per complete step the state is captured,
    each dispatched arrangement runs one ``FdtdDriver.step`` from that state and is
    captured and rolled back, the ARRAY path then runs the same step, and the run
    continues from the ARRAY result -- so the reference is never a subject's own
    output.

    THREE ARRANGEMENTS, NOT TWO, and the third is what makes a divergence
    ATTRIBUTABLE: if the weld and the CERTIFIED SINGLES agree with each other and both
    differ from the array path, the finding is about the certified halves and not
    about the weld.
    """
    fields, pml, grid = driver.fields, driver.pml, driver.grid
    dtdx = float(grid.dt / grid.dx)
    singles = adapter.singles(fields, grid, pml, dtdx)
    state = adapter.resolve(fields, grid, pml, dtdx)
    adapter.assert_disjoint(fields, state)
    book = {"launches": 0}

    def weld() -> None:
        adapter.launch(fields, state)
        book["launches"] += 1
        adapter.rotate(fields, state)

    arrangements = {
        "weld": {"update_H": weld, "step_D": lambda: None},
        "singles": {"update_H": singles["update_H"], "step_D": singles["step_D"]},
    }
    first: Dict[str, Optional[Dict[str, Any]]] = {"weld": None, "singles": None}
    agree = 0
    words = compared = subnormals = 0
    for step in range(steps):
        base = hd_gate._capture(driver)  # noqa: SLF001
        results: Dict[str, Dict[str, np.ndarray]] = {}
        for name, plans in arrangements.items():
            hd_gate._restore(driver, base)  # noqa: SLF001
            driver._fast_path = Shim(plans)  # noqa: SLF001
            driver._fast_path_stale = False  # noqa: SLF001
            driver.step()
            cp.cuda.runtime.deviceSynchronize()
            results[name] = {key: to_host(value).copy() for key, value
                             in hd_gate._stored_volumes(fields).items()}  # noqa: SLF001
        hd_gate._restore(driver, base)  # noqa: SLF001
        driver._fast_path = None  # noqa: SLF001
        driver._fast_path_stale = False  # noqa: SLF001
        driver.step()
        cp.cuda.runtime.deviceSynchronize()
        array = {key: to_host(value).copy() for key, value
                 in hd_gate._stored_volumes(fields).items()}  # noqa: SLF001
        compared = step + 1
        words += int(sum(np.asarray(value).nbytes // 4 for value in array.values()))
        subnormals += int(sum(cyl.subnormal_words(
            np.asarray(value).view(np.float32) if np.iscomplexobj(value) else value)
            for value in array.values()))
        for name in arrangements:
            moved = {key: int(differing(array[key], results[name][key]))
                     for key in array
                     if differing(array[key], results[name][key])}
            if moved and first[name] is None:
                first[name] = {"step": step + 1, "volumes": moved}
        if first["weld"] and first["singles"] and (
                first["weld"]["step"] == first["singles"]["step"]):
            if all(differing(results["weld"][key], results["singles"][key]) == 0
                   for key in array):
                agree += 1
        if first["weld"] is not None:
            break
    return {
        "steps_compared": compared, "words_compared": words,
        "subnormal_words_seen": subnormals, "weld_launches": book["launches"],
        "first_divergence": first,
        "the_weld_and_the_certified_singles_agree_where_both_differ": bool(agree),
        "identical_to_the_array_path": first["weld"] is None,
        "certified_singles_identical_to_the_array_path": first["singles"] is None,
        "passed": bool(first["weld"] is None and book["launches"] > 0),
    }


def evaluate_row(adapter: Adapter, driver: Any, steps: int,
                 max_cells: Optional[str], block_key: str) -> Dict[str, Any]:
    """The identity on ONE lifted row, plus the predicate and composer facts."""
    from meep_gpu.cuda_kernels import arms  # noqa: PLC0415

    family = adapter.family
    sources = tuple(getattr(driver, "_sources", ()) or ())
    block: Dict[str, Any] = {
        "family": family.FAMILY, "steps_requested": steps,
        "n_sources": len(sources), "arm": str(adapter.arm),
        "grid_shape": [int(n) for n in driver.grid.shape],
        "beta": float(getattr(driver.grid, "beta", 0.0)),
        "storage": str(driver.fields.Hx.dtype),
        "sources": [{"type": type(s).__name__,
                     "field_type": str(getattr(s, "field_type", "")),
                     "is_integrated": bool(getattr(s, "is_integrated", False)),
                     "withdraw_does_work": bool(
                         withdraw_hoist._withdraw_does_work(s))}  # noqa: SLF001
                    for s in sources],
        "policy_stamp_before_drive":
            probe.subnormal_policy_stamp(_REPO_API).get("policy"),
    }
    covered, reason = adapter.predicate(driver.fields, driver.pml, driver.grid, sources)
    block["predicate_admits"] = bool(covered)
    block["predicate_reason"] = reason
    try:
        plan = arms.plan_step(fields=driver.fields, pml=driver.pml, grid=driver.grid,
                              sources=sources, fuse=True,
                              licenses=adapter.composer_licenses(),
                              subnormal_policy=adapter.policy)
        block["composer_selected_as_shipped"] = dict(getattr(plan, "selected", {}) or {})
    except Exception as error:  # noqa: BLE001
        block["composer_error"] = repr(error)[:400]
    cells = int(np.prod(driver.grid.shape))
    block["grid_cells"] = cells
    if not covered:
        block.update({"driven": False,
                      "why_not_driven": "the predicate refused this row"})
        child_progress(adapter, f"REFUSED {str(reason)[:120]}")
        return {block_key: block}
    if max_cells and cells > int(max_cells):
        block.update({"driven": False,
                      "why_not_driven": (f"{cells} cells exceeds the cap {max_cells}; "
                                         f"refused rather than run partially")})
        return {block_key: block}
    block["driven"] = True
    block.update(drive_lifted_row(adapter, driver, steps))
    block["policy_stamp_after_drive"] = (
        probe.subnormal_policy_stamp(_REPO_API).get("policy"))
    child_progress(adapter,
                   f"passed={block.get('passed')} words={block.get('words_compared')}")
    return {block_key: block}


def lift_child(adapter: Adapter, leg: str, target: str, cases: Sequence[str],
               out_json: str, steps: int, progress: str, policy: Optional[str],
               import_meep: bool, block_key: str) -> int:
    """One corpus row, lifted in its OWN interpreter and driven. ALWAYS writes.

    THE POLICY IS INSTALLED HERE, before the child's first compile, and stamped before
    and after the drive: a child that inherited only a cache directory would compile
    under whatever policy the parent left installed in ITS process, which is not this
    process.
    """
    import meep_gpu  # noqa: PLC0415

    os.environ[f"{adapter.env_stem}_PROGRESS"] = progress or ""
    os.environ[f"{adapter.env_stem}_LABEL"] = f"{leg}:{Path(str(target)).name}"
    if import_meep:
        probe.import_meep_for_host_policy()
    probe.install_nvrtc_binary_observer()
    if policy:
        probe.install_subnormal_policy_for_run(policy, _REPO_API)
    adapter.policy = policy
    child_progress(adapter, f"child start policy={policy} steps={steps} "
                            f"arm={adapter.arm}")

    records: List[Dict[str, Any]] = []

    def finish() -> int:
        for record in records:
            gate_provenance.stamp(record)
        Path(out_json).write_text(json.dumps(records, indent=2, default=str),
                                  encoding="utf-8")
        return 0

    if leg == "examples":
        from parity.meep_gpu.sweep_corpus_lift_parity import (  # noqa: PLC0415
            capture_simulation)

        record, sim, restore = capture_simulation(target)
        record["row"] = Path(str(target)).name
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
                record.update({"measured": False,
                               "note": "no mp.Simulation to lift on this replay"})
                continue
            pairs.append((record, sim))
    max_cells = os.environ.get(f"{adapter.env_stem}_MAX_CELLS")
    for record, sim in pairs:
        started = time.time()
        try:
            driver = meep_gpu.lift_simulation(sim, prefer_gpu=True, gpu_id=0)
        except BaseException as exc:  # noqa: BLE001
            record.update({"measured": False,
                           "lift_error": f"{type(exc).__name__}: {exc}"[:600]})
            child_progress(adapter, f"LIFT FAILED {type(exc).__name__}")
            continue
        record["lift_s"] = round(time.time() - started, 2)
        record["grid_cells"] = int(np.prod([int(v) for v in driver.shape]))
        try:
            record.update(evaluate_row(adapter, driver, steps, max_cells, block_key))
            record["measured"] = True
        except BaseException as exc:  # noqa: BLE001
            record.update({"measured": False,
                           "battery_error": f"{type(exc).__name__}: {exc}"[:600]})
        finally:
            with contextlib.suppress(BaseException):
                driver.close()
    return finish()


def leg_lift(adapter: Adapter, gate_file: str, out_dir: Path, steps: int,
             max_cells: Optional[int], timeout: float, resume: bool,
             only: Optional[Sequence[str]], policy: Optional[str],
             import_meep: bool, block_key: str,
             child_env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Every corpus row in the cell, re-lifted in its own interpreter and driven."""
    import measure_predicate_coverage as census  # noqa: PLC0415

    out_dir = Path(out_dir).resolve()
    rows, facts = lift_basis(adapter, Path(_HERE) / "results")
    if only:
        wanted = set(only)
        rows = [row for row in rows if row["label"] in wanted]
    # THE CORPUS PATH IS THE CENSUS'S, and it is REFUSED rather than guessed at. A
    # child handed a bare script NAME resolves it against its own cwd and dies with a
    # FileNotFoundError that the parent records as a right-shaped row measuring
    # nothing. ``census.EXAMPLES_DIR``/``TESTS_DIR`` read ``MEEP_GPU_CORPUS_ROOT``.
    examples_dir = Path(census.EXAMPLES_DIR)
    tests_dir = Path(census.TESTS_DIR)
    if not examples_dir.is_dir() or not tests_dir.is_dir():
        return {"passed": False, "basis": facts,
                "reason": (f"the MEEP corpus is not at {examples_dir} / {tests_dir}; "
                           f"set MEEP_GPU_CORPUS_ROOT to the checkout the census was "
                           f"cut over. Refused rather than measured on nothing")}
    lift_dir = out_dir / "lift"
    lift_dir.mkdir(parents=True, exist_ok=True)
    progress_path = lift_dir / "steps.progress.log"
    # THE CHILDREN'S WORKING DIRECTORY IS NOT EVIDENCE AND STAYS OUTSIDE THE ARTIFACT:
    # a corpus script writes whatever it likes there, and a file inside the artifact
    # that the manifest rule cannot hash is a file the record cannot pin. The corpus's
    # DATA files are symlinked in, because several scripts read one beside themselves.
    workdir = Path(tempfile.mkdtemp(prefix="cuda_beta_hd_lift_"))
    for entry in examples_dir.iterdir():
        if entry.suffix in (".py", ".ipynb"):
            continue
        with contextlib.suppress(OSError):
            (workdir / entry.name).symlink_to(entry)
    # THE `parameterized` SHIM, on the PYTHONPATH of the modules that import it: the
    # package is not installed on the validation host and the census ships a shim whose
    # expansion names cases `<method>__idx<N>`, which is the `replay_case` the basis
    # already joined.
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
    driven: List[Dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        target = (examples_dir / row["row"] if row["leg"] == "examples"
                  else (tests_dir / row["module"]) if row["module"] else None)
        out_json = lift_dir / (row["label"].replace("/", "_").replace(":", "__")
                               + ".json")
        line = (f"[{time.strftime('%H:%M:%S')}] lift {index}/{len(rows)} "
                f"{row['label']}")
        print(line, flush=True)
        with open(progress_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        if resume and out_json.is_file():
            driven.append({**row,
                           "payload": json.loads(out_json.read_text(encoding="utf-8")),
                           "resumed": True})
            continue
        if row["withdraw_in_seam"]:
            driven.append({**row, "refused_by_the_predicate": True,
                           "why": "standing integrated electric withdraw in the seam"})
            continue
        if target is None:
            driven.append({**row, "error": "no module for this row in the census"})
            continue
        command = [sys.executable, "-u", str(Path(_HERE) / gate_file),
                   "--lift-child", row["leg"], "--lift-child-target", str(target),
                   "--lift-child-cases", json.dumps([row["replay_case"]]),
                   "--lift-child-out", str(out_json),
                   "--lift-steps", str(steps),
                   "--lift-child-progress", str(progress_path)]
        if policy:
            command += ["--subnormal-policy", policy]
        if import_meep:
            command += ["--import-meep-for-host-policy"]
        if max_cells:
            command += ["--lift-max-cells", str(max_cells)]
        environment = dict(os.environ)
        environment.update({"KMP_DUPLICATE_LIB_OK": "TRUE", "MPLBACKEND": "Agg"})
        environment["PYTHONPATH"] = (
            f"{shim_path}{os.pathsep}{_REPO_API}" if row.get("module") in needs_shim
            else _REPO_API)
        if max_cells:
            environment[f"{adapter.env_stem}_MAX_CELLS"] = str(max_cells)
        environment.update(child_env or {})
        started = time.perf_counter()
        try:
            # ENCODING AND ERRORS ARE EXPLICIT, and both are load-bearing. ``text=True``
            # alone decodes with the interpreter's LOCALE encoding -- ASCII under
            # C/POSIX -- and a corpus test that prints one non-ASCII character raises
            # UnicodeDecodeError IN THE PARENT, killing the whole leg after the child
            # has already measured correctly.
            completed = subprocess.run(command, cwd=str(workdir), env=environment,
                                       capture_output=True, encoding="utf-8",
                                       errors="replace", timeout=timeout, check=False)
            record = {"returncode": completed.returncode,
                      "stdout_tail": completed.stdout[-3000:],
                      "stderr_tail": completed.stderr[-3000:]}
        except subprocess.TimeoutExpired:
            record = {"returncode": None, "timeout": timeout}
        record["seconds"] = round(time.perf_counter() - started, 1)
        if out_json.is_file():
            record["payload"] = json.loads(out_json.read_text(encoding="utf-8"))
        driven.append({**row, **record})

    def block(entry) -> Dict[str, Any]:
        payload = entry.get("payload")
        if isinstance(payload, dict):
            payload = [payload]
        for record in (payload or []):
            found = record.get(block_key)
            if found:
                return found
        return {}

    blocks = {entry["label"]: block(entry) for entry in driven}
    admitted = [label for label, value in blocks.items() if value.get("predicate_admits")]
    ran = [label for label, value in blocks.items() if value.get("driven")]
    identical = [label for label, value in blocks.items()
                 if value.get("identical_to_the_array_path")]
    below = [label for label, value in blocks.items()
             if value.get("driven") and 0 < int(value.get("steps_compared") or 0)
             < LIFT_CLEAN_STEP_FLOOR]
    diverged = [label for label, value in blocks.items()
                if value.get("driven") and not value.get("identical_to_the_array_path")]
    errored = [entry["label"] for entry in driven
               if entry.get("error") or (entry.get("returncode") not in (0, None)
                                         and not blocks[entry["label"]])]
    refused = sorted(entry["label"] for entry in driven
                     if entry.get("refused_by_the_predicate"))
    record = {
        "basis": facts, "rows": driven, "blocks": blocks,
        "rows_on_the_board_cell": len(rows),
        "rows_admitted": len(admitted), "rows_driven": len(ran),
        "rows_identical": len(identical),
        "rows_refused_for_the_withdraw": refused,
        "rows_below_the_step_floor": below,
        "rows_diverged": diverged, "rows_errored": errored,
        "complete_driver_steps": sum(int(value.get("steps_compared") or 0)
                                     for value in blocks.values()),
        "words_compared": sum(int(value.get("words_compared") or 0)
                              for value in blocks.values()),
        "weld_runs": sum(int(value.get("weld_launches") or 0)
                         for value in blocks.values()),
        "step_floor": LIFT_CLEAN_STEP_FLOOR,
    }
    record["passed"] = bool(
        ran and not diverged and not errored and not below
        and len(identical) == len(ran)
        and sorted(refused) == sorted(facts["rows_with_a_standing_withdraw"]))
    return record


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------

def make_parser(description: str):
    return cyl.make_parser(description)


def install_policy(results: Dict[str, Any], args) -> Dict[str, Any]:
    return cyl.install_policy(results, args)


def load_expansion_licence(policy: Optional[str]) -> Dict[str, Any]:
    return cyl.load_expansion_licence(policy)


def save(results: Dict[str, Any], out: str) -> None:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    gate_provenance.stamp(results)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")


def run_campaign(adapter: Adapter, args, gate_file: str, family_name: str,
                 block_key: str, licences_what: str) -> int:
    """Every leg, in order, with the artifact rewritten after each one.

    THE AUTHORITATIVE FIELD IS ``canonical_verdict.released``. A leg that did not run
    is a reason to withhold, not a clause to omit: the released flag requires every
    requested leg to have produced a verdict.
    """
    started = time.perf_counter()
    legs = tuple(args.legs.split(",")) if args.legs else ALL_LEGS
    family = adapter.family
    results: Dict[str, Any] = {
        "gate": gate_file, "family": family_name, "seam": withdraw_hoist.SEAM,
        "cell": list(adapter.cell_arms), "board": BOARD, "census": CENSUS,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "argv": list(sys.argv[1:]), "legs_requested": list(legs), "steps": args.steps,
        "runtime_reasons": runtime_reasons(),
    }
    if cp is None or args.no_device:
        results["device_mode"] = False
        results["status"] = "refused: no CuPy" if cp is None else "structural only"
        save(results, args.out)
        log(results["status"])
        return 1

    results["device_mode"] = True
    at_install = install_policy(results, args)
    adapter.policy = args.subnormal_policy
    if adapter.complex_storage:
        licence = load_expansion_licence(args.subnormal_policy)
        results["expansion_licence"] = licence
        if not licence.get("usable"):
            results["status"] = "refused: no usable expansion licence for this policy"
            results["verdict"] = {"passed": False, "clauses": {}}
            results["canonical_verdict"] = {"released": False,
                                            "reasons": ["no usable expansion licence"]}
            save(results, args.out)
            log(results["status"])
            return 1
        adapter.arm = licence["arm"]
        adapter.license = licence["verdict"]
        results["arm"] = adapter.arm
    results["what_a_release_does_not_license"] = family.WHAT_A_RELEASE_DOES_NOT_LICENSE
    results["installable"] = bool(family.INSTALLABLE)
    results["source_digest"] = adapter.source_digest()
    results["kernel"] = adapter.kernel_name()
    save(results, args.out)

    specs = (list(adapter.specs) if args.product == "full"
             else [spec for spec in adapter.specs
                   if spec["label"] in adapter.reduced_labels])
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
        log(f"[{name}] passed={results[name].get('passed')} "
            f"({results[name]['seconds']} s)")
        save(results, args.out)

    run_leg("driver_order", lambda: leg_driver_order(adapter))
    run_leg("transcription", adapter.leg_transcription)
    run_leg("refusal", adapter.leg_refusal)
    run_leg("ghost_observability", lambda: leg_ghost_observability(adapter))
    run_leg("spelling", adapter.leg_spelling)

    if "product" in legs:
        cases: List[Dict[str, Any]] = []
        for spec in specs:
            for value_class in adapter.value_classes:
                record = drive(adapter, spec, value_class, args.steps,
                               progress=lambda m: log(f"  {m}"))
                cases.append(record)
                log(f"[product] {record['label']:38s} "
                    f"{'IDENTICAL' if record['passed'] else 'DIVERGED'} "
                    f"steps={record['steps_compared']} "
                    f"words={record['words_compared']} "
                    f"subnormals={record['operand_census']['subnormals']}")
                results["product"] = {"cases": cases}
                save(results, args.out)
        band = [r for r in cases if r["value_class"] == "subnormal_band"]
        uniform = [r for r in cases if r["value_class"] == "uniform"]
        results["product"] = {
            "cases": cases, "denominator": len(cases),
            "arrangements_per_case": len(MODES),
            "complete_driver_steps": sum(r["steps_compared"] for r in cases) * len(MODES),
            "words_compared": sum(r["words_compared"] for r in cases),
            "the_band_class_really_contains_subnormals":
                bool(band) and all(r["operand_census"]["subnormals"] > 0 for r in band),
            "the_uniform_class_contains_none":
                bool(uniform) and all(r["operand_census"]["subnormals"] == 0
                                      for r in uniform),
            "every_fixture_carries_a_nonzero_beta": all(r["beta"] for r in cases),
            "failing_cases": [r["label"] for r in cases if not r["passed"]],
            "passed": bool(cases) and all(r["passed"] for r in cases)
                      and all(r["beta"] for r in cases),
        }
        save(results, args.out)

    run_leg("purity_ledger", lambda: leg_purity_ledger(adapter, specs))

    if "block_sizes" in legs:
        sweep: List[Dict[str, Any]] = []
        for spec in specs:
            for threads in BLOCK_SIZES:
                record = drive(adapter, spec, "uniform", min(args.steps, 12),
                               modes=("array", "weld"), threads=threads)
                record["threads"] = threads
                sweep.append(record)
                log(f"[block_sizes] {spec['label']:28s} b{threads:<5d} "
                    f"{'IDENTICAL' if record['identical']['weld'] else 'DIVERGED'}")
            results["block_sizes"] = {"cases": sweep}
            save(results, args.out)
        results["block_sizes"] = {
            "cases": sweep, "denominator": len(sweep),
            "block_sizes": list(BLOCK_SIZES),
            "what_this_licenses": (
                "the answer does not depend on which block ran first, driven over six "
                "block sizes rather than argued from the scratch design"),
            "passed": bool(sweep) and all(r["identical"]["weld"] for r in sweep)}
        save(results, args.out)

    run_leg("seed_scale", lambda: leg_seed_scale(adapter, primary, args.steps))
    run_leg("launch_structure",
            lambda: leg_launch_structure(adapter, primary, min(args.steps, 12)))
    run_leg("arbitration", lambda: leg_arbitration(adapter, primary))
    run_leg("sync", lambda: leg_sync(adapter, primary, min(args.steps, 12)))
    run_leg("withdraw", lambda: leg_withdraw(adapter, primary, min(args.steps, 12)))
    run_leg("byte_neutral",
            lambda: leg_byte_neutral(adapter, specs[-1], args.steps))
    run_leg("mutation", lambda: leg_mutation(adapter, mutation_specs, args.steps))
    run_leg("disarm", lambda: leg_disarm(adapter, specs[-1], args.steps))
    run_leg("compiler",
            lambda: leg_compiler(adapter, args.subnormal_policy, at_install))
    run_leg("lift", lambda: leg_lift(
        adapter, gate_file, Path(args.out).parent, args.lift_steps,
        args.lift_max_cells, args.lift_timeout, args.lift_resume,
        args.lift_only.split(",") if args.lift_only else None,
        args.subnormal_policy, bool(args.import_meep_for_host_policy), block_key,
        adapter.child_env()))

    clauses = {name: bool(results.get(name, {}).get("passed"))
               for name in legs if name in results}
    results["verdict"] = {
        "clauses": clauses, "passed": bool(clauses) and all(clauses.values()),
        "legs_run": sorted(clauses), "legs_requested": list(legs),
        "legs_missing": sorted(set(legs) - set(clauses)),
    }
    results["canonical_verdict"] = {
        "released": bool(results["verdict"]["passed"]
                         and not results["verdict"]["legs_missing"]),
        "reasons": [name for name, value in clauses.items() if not value]
                   + [f"leg did not run: {name}"
                      for name in results["verdict"]["legs_missing"]],
        "what_it_licenses": licences_what,
    }
    results["elapsed_s"] = round(time.perf_counter() - started, 1)
    save(results, args.out)
    log("=" * 78)
    for clause, value in clauses.items():
        log(f"  {'PASS' if value else 'FAIL'}  {clause}")
    log(f"VERDICT: {'PASS' if results['verdict']['passed'] else 'FAIL'}  "
        f"released={results['canonical_verdict']['released']}  "
        f"({results['elapsed_s']} s)")
    return 0 if results["canonical_verdict"]["released"] else 1
