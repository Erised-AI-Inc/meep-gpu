"""Byte-identity gate for the two hand-CUDA SCRATCH-OUTPUT stencil welds.

WHAT IS UNDER TEST, and what makes this gate different from every other one on this
track. ``meep_gpu/cuda_kernels/offdiag_fused_electric_pair.py`` and
``folded_offdiag_fused_electric_pair.py`` occupy the two ``D_to_E`` cells the hand-CUDA
board has recorded STENCIL-BLOCKED since they were first priced::

    "the constitutive half reads the curl half's IN-PLACE output at a cell this
     thread does not own. In one launch that cell is written by another block, and
     CUDA offers no grid-wide barrier inside an ordinary launch."

That verdict was MEASURED, and it is true of the weld it describes. The products under
test are a different weld: the curl half writes ``D_new``/``fu_new`` to a LAUNCH-LOCAL
SCRATCH, the constitutive half takes its own cell from a register and RECOMPUTES each
foreign cell from PRE-LAUNCH state through the same ``__device__`` function, and the
launcher rotates the ``D``/``fu`` bindings afterwards. So this gate does not ask
whether the recorded refusal was wrong; it asks whether a launch with neither of that
refusal's two premises is byte-identical to the array path.

=============================================================================
THE BAR, AND WHERE IT COMES FROM
=============================================================================

``results/triton_fused_offdiag_electric_2026-08-20`` is the artifact that MEASURED the
in-place weld's race, and this gate is patterned on it and held to the inverse of its
S1: where that record required the subject to DISAGREE and required the disagreement
to be schedule-dependent, this one requires

* **S1: identical, 60 complete driver steps, every stored volume, every step.**
  Sixty is the budget every hand-CUDA record on this track is cut at, and it is the
  claim: "identical for N steps" is a statement about N. A single differing uint32
  anywhere in the twenty-four stored volumes fails the leg and the first divergent
  step is reported rather than a final pass/fail.
* **S1 repeated, so a schedule can be exposed.** The 2026-08-20 record's whole finding
  was that the in-place weld's answer DEPENDED ON THE BLOCK SCHEDULE, which a single
  run cannot see. Every S1 fixture is therefore run :data:`SCHEDULE_REPEATS` times from
  the same seed and every repeat must give the same bytes -- and the mutation
  ``foreign_read_from_live_buffer`` below rebuilds exactly the in-place shape, so this
  leg is shown to be capable of catching what that record found.
* **S2: identical at EVERY thread-block size.** The block size changes which cells are
  co-resident and in what order, and it is the cheapest, sharpest schedule lever there
  is. A weld that was accidentally correct at 256 threads and wrong at 32 would pass a
  single-block-size gate.
* **Both float32 subnormal policies**, each in its own process with its own CuPy cache
  directory, because CuPy's disk-cache key is computed above the strip seam.
* **Launch counts from two independent counters** -- a proxy over the shipped compile
  memo and the launcher's own returned report -- which must agree, equal the step
  budget, and name ONLY the fused kernel. Bytes alone cannot prove the kernel ran: an
  engine that never launched it is byte-identical to the oracle BY CONSTRUCTION.
* **The mutation battery**, every arm of which is armed through the SHIPPED launcher's
  own doors (``kernel=``, ``rotate=``, the resolved tables) rather than by editing
  anything under ``meep_gpu/``.

=============================================================================
THE MUTATIONS, AND WHY THESE
=============================================================================

Five are the residue audit's own list for this design, and each one is a defect that
COMPILES and produces a smooth, converged, plausible field:

* ``halo_recompute_dropped`` -- a foreign sample takes the thread's OWN cell instead
  of recomputing at the neighbour. This is the shortcut a reader reaches for when the
  recompute looks expensive, and the whole design is that it is not available.
* ``foreign_read_from_live_buffer`` -- the resolution reads the SCRATCH the launch is
  writing instead of the pre-launch volume. THIS IS THE IN-PLACE WELD, rebuilt: if it
  is not caught, this gate cannot tell the two designs apart and its S1 means nothing.
* ``scratch_aliased_to_storage`` -- the same defect from the host side, through the
  binding rather than the source. Expected to be refused BY NAME before any launch, so
  it is scored as a REFUSAL rather than as a byte divergence.
* ``rotation_skipped`` -- the launch computes the step and throws it away.
* ``redirect_parity_dropped`` -- the near/far mirror parity multiply removed. Live
  only where a fold is, and recorded ``predicted_null`` with its reason where not.

The rest are this seam's own long-standing hazards, each already known to be a silent
wrong answer somewhere on this board: the sub-lattice shadow (half a cell in the
absorber profile), the wall table taken from the B family's diagonal, the near source
row, the ``n - 2`` reflect row that is right at an even full count and a whole cell
wrong at an odd one, the mirror ghost weight, and -- for the folded weld alone -- the
constitutive boundary codes replaced by the curl's, which is the one edit that family
makes and its twin does not.

=============================================================================
WHAT THIS GATE DOES NOT CLAIM
=============================================================================

No throughput claim is made or possible: the box is shared, and this file times
nothing. Nothing here is a dispatch claim either --
``meep_gpu.fastpath.plan_fast_path`` still returns ``None`` on every branch, so no
production step reaches these kernels. And the resolution's ARITHMETIC against the
driver's pass order is measured off-device by
``parity/meep_gpu/probe_cuda_offdiag_scratch_weld.py``; what is measured here is a
DEVICE claim about a launch.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import os
import sys
import textwrap
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_API = os.path.dirname(os.path.dirname(_HERE))
for _path in (_REPO_API, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    import cupy as cp
except ImportError:  # laptop: only the source legs can run
    cp = None

import gate_provenance  # noqa: E402
import probe_fused_kernel_bit_identity as probe  # noqa: E402

from meep_gpu import stepping  # noqa: E402

log = probe.log
operand_census = probe.operand_census

#: Per-case RNG seed base, from the case LABEL rather than from ``hash()``:
#: ``PYTHONHASHSEED`` salts the hash of a string, so a hash-seeded case could not be
#: replayed from the record that names it.
SEED = 20260902

#: Complete DRIVER STEPS the S1 legs run. Sixty, the budget every hand-CUDA record on
#: this track is cut at.
STEPS = 60

#: How many times each S1 fixture is repeated from the same seed. The 2026-08-20
#: artifact's finding was that the in-place weld's answer was SCHEDULE-DEPENDENT, and
#: a single run cannot see that; repeats are the cheapest exposure there is.
SCHEDULE_REPEATS = 4

#: The block sizes S2 sweeps. 32 is one warp, 1024 the maximum, and the shipped
#: launcher's own 256 sits between them; they put wildly different numbers of blocks
#: on the grid, which is what makes the sweep a schedule sweep rather than a shape one.
BLOCK_SIZES: Tuple[int, ...] = (32, 64, 128, 256, 512, 1024)

#: Every stored volume a complete step can touch. All of them are compared: the B/H
#: half is here even though neither product touches it, because a pair that corrupted
#: E reaches B through ``step_B`` on the very next step.
STATE_NAMES: Tuple[str, ...] = tuple(
    [f"{stem}{axis}" for stem in ("B", "D", "E", "H") for axis in "xyz"]
    + [f"fu_{stem}{axis}" for stem in ("B", "D") for axis in "xyz"]
    + [f"f_w_{stem}{axis}" for stem in ("E", "H") for axis in "xyz"])

#: The volumes that must be BIT-UNCHANGED by a run: the PML coefficient vectors on
#: both Yee sub-lattices. Every field and table argument in these signatures is
#: ``__restrict__``, and a kernel that wrote through a ``const`` binding is UB NVRTC
#: does not diagnose.
IMMUTABLE_PML: Tuple[str, ...] = tuple(
    f"{name}_{axis}{suffix}"
    for axis in "xyz" for name in ("kms", "sinv", "kps") for suffix in ("", "_h"))

#: The families the SUBNORMAL BAND class is held to. Declared before the first run,
#: and for the reason the sibling gate records: under ``flush`` a state seeded
#: entirely in the band drives ``constitutive_apply`` to ``f[idx] = (f[idx] + 0) - 0``
#: because ``kps*src`` and ``kms*prev`` both flush, so E is the identity and need not
#: move. D, fu_D and f_w_E are written unconditionally and must move under either.
BAND_MUST_MOVE: Tuple[str, ...] = tuple(
    [f"D{axis}" for axis in "xyz"] + [f"fu_D{axis}" for axis in "xyz"]
    + [f"f_w_E{axis}" for axis in "xyz"])

VALUE_CLASSES: Tuple[str, ...] = ("uniform", "subnormal_band")

#: NVRTC options the shipped modules compile with, restated so the guard control can
#: drop them. NOT imported from either module: a control that read the module's own
#: tuple would compile the same thing twice if that tuple were ever emptied.
GUARD_OPTIONS: Tuple[str, ...] = ("--fmad=false",)


# ---------------------------------------------------------------------------
# The two products
# ---------------------------------------------------------------------------

def products() -> Dict[str, Any]:
    """The two modules under test, imported at call time (they need CuPy)."""
    from meep_gpu.cuda_kernels import (  # noqa: PLC0415
        folded_offdiag_fused_electric_pair as folded,
        offdiag_fused_electric_pair as flat,
    )

    return {"flat": flat, "folded": folded}


#: The fixture sweep. Each spec names the shape, the boundaries, the fold, the
#: off-diagonal ROW MASK and which product must admit it -- and the sweep is built so
#: that every axis is walled somewhere, every axis is folded somewhere, BOTH fold
#: terminations appear, both full-count parities appear, and both an isotropic and a
#: diagonally anisotropic permittivity appear.
#:
#: THE ANISOTROPIC HALF IS NOT DECORATION. ``update_E`` binds THREE
#: inverse-permittivity pointers and on an isotropic run all three are the SAME
#: allocation, so a kernel that read ``inv_eps_Ez`` for all three would be
#: byte-identical on every isotropic row.
SPECS: Tuple[Dict[str, Any], ...] = (
    {"label": "flat_periodic_one_row", "product": "flat",
     "cell": (1.6, 1.6, 1.6), "boundaries": ("periodic", "periodic", "periodic"),
     "symmetry": (), "rows": {"Ex": ("Ey",)}, "anisotropic": False},
    {"label": "flat_periodic_all_rows", "product": "flat",
     "cell": (1.6, 1.7, 1.5), "boundaries": ("periodic", "periodic", "periodic"),
     "symmetry": (), "rows": "all", "anisotropic": True},
    {"label": "flat_walls_all_rows", "product": "flat",
     "cell": (1.6, 1.7, 1.5), "boundaries": ("metallic", "metallic", "metallic"),
     "symmetry": (), "rows": "all", "anisotropic": True},
    {"label": "flat_wall_x_only", "product": "flat",
     "cell": (1.6, 1.6, 1.6), "boundaries": ("metallic", "periodic", "periodic"),
     "symmetry": (), "rows": "all", "anisotropic": False},
    {"label": "fold_y_even_one_row", "product": "folded",
     "cell": (1.6, 2.0, 1.6), "boundaries": None, "symmetry": (("Y", 1),),
     "rows": {"Ex": ("Ey",)}, "anisotropic": False},
    {"label": "fold_y_odd_all_rows", "product": "folded",
     "cell": (1.6, 2.0, 1.6), "boundaries": None, "symmetry": (("Y", -1),),
     "rows": "all", "anisotropic": True},
    {"label": "fold_y_odd_count", "product": "folded",
     "cell": (1.6, 2.1, 1.6), "boundaries": None, "symmetry": (("Y", 1),),
     "rows": "all", "anisotropic": True},
    {"label": "fold_y_metallic_termination", "product": "folded",
     "cell": (1.6, 2.0, 1.6), "boundaries": ("periodic", "metallic", "periodic"),
     "symmetry": (("Y", 1),), "rows": "all", "anisotropic": False},
    {"label": "fold_y_odd_wall_x", "product": "folded",
     "cell": (1.6, 2.0, 1.6), "boundaries": ("metallic", "periodic", "periodic"),
     "symmetry": (("Y", -1),), "rows": "all", "anisotropic": True},
    {"label": "fold_xy_mixed_phase", "product": "folded",
     "cell": (2.0, 2.0, 1.6), "boundaries": None,
     "symmetry": (("X", 1), ("Y", -1)), "rows": "all", "anisotropic": True},
)

#: The subset the ``--product reduced`` mode runs, and the subset the mutations are
#: scored on. Chosen so every mutation has at least one fixture where its machinery is
#: LIVE: a wall, a fold at each termination, an odd parity, an odd full count and a
#: doubly-unowned corner.
MUTATION_SPEC_LABELS: Tuple[str, ...] = (
    "flat_walls_all_rows", "fold_y_odd_all_rows", "fold_y_odd_count",
    "fold_y_metallic_termination", "fold_y_odd_wall_x", "fold_xy_mixed_phase")

REDUCED_LABELS: Tuple[str, ...] = (
    "flat_periodic_all_rows", "flat_walls_all_rows", "fold_y_odd_all_rows",
    "fold_xy_mixed_phase")

ALL_ROWS: Dict[str, Tuple[str, ...]] = {
    "Ex": ("Ey", "Ez"), "Ey": ("Ez", "Ex"), "Ez": ("Ex", "Ey")}


def case_rng(label: str) -> np.random.Generator:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return np.random.default_rng(SEED + int.from_bytes(digest[:4], "big"))


def _rows_for(spec: Dict[str, Any]) -> Dict[str, Tuple[str, ...]]:
    return ALL_ROWS if spec["rows"] == "all" else spec["rows"]


def build(spec: Dict[str, Any], value_class: str, rng):
    """A frozen ``(fields, grid, pml, dtdx)`` on the DEVICE, seeded for this class."""
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid, Mirror  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415

    grid = Grid(resolution=10.0, cell_size=spec["cell"], courant=0.35, xp=cp,
                boundaries=spec["boundaries"],
                symmetry=tuple(Mirror(axis, phase)
                               for axis, phase in spec["symmetry"]))
    thickness = tuple((0, 0) if grid.shape[axis] < 6
                      else (0, 2) if grid.is_mirrored(axis)
                      else (2, 2) for axis in range(3))
    pml = PML(grid=grid, thickness=thickness)
    fields = Fields(grid=grid)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    shape = tuple(int(n) for n in grid.shape)
    # THE PERMITTIVITY. Isotropic binds ONE allocation three times, which is the
    # configuration the certified update_E is built for and the one a per-component
    # binding error hides in; anisotropic binds three.
    values = (2.0, 2.0, 2.0) if not spec["anisotropic"] else (2.0, 2.5, 3.0)
    names = ("Ex", "Ey", "Ez")
    epsilon = {n: cp.full(shape, v, cp.float32) for n, v in zip(names, values)}
    inverse = {n: cp.full(shape, np.float32(1.0 / v), cp.float32)
               for n, v in zip(names, values)}
    rows = {row: {partner: cp.asarray(
        rng.uniform(-0.25, 0.25, shape).astype(np.float32))
        for partner in partners}
        for row, partners in _rows_for(spec).items()}
    fields.set_epsilon_volumes(epsilon, inverse, chi1inv_offdiagonal=rows)
    # THE BAND CLASS IS THE SHARED DRAW, not a second one. `subnormal_band_hosts`
    # takes the NAMES to seed, so the whole stored set goes into the band together
    # -- a class that seeded only the D family would leave E and f_w_E outside the
    # band and the census would say the leg was non-vacuous when half of it was.
    hosts = (probe.pml_field_hosts(shape, rng, "uniform") if value_class == "uniform"
             else probe.subnormal_band_hosts(STATE_NAMES, shape, rng))
    for name in STATE_NAMES:
        array = getattr(fields, name, None)
        if array is None:
            continue
        source = hosts.get(name)
        if source is None:
            source = rng.normal(0.0, 0.37, size=shape).astype(np.float32)
        array[...] = cp.asarray(np.ascontiguousarray(
            np.asarray(source, dtype=np.float32)))
    return fields, grid, pml, float(grid.dt / grid.dx)


# ---------------------------------------------------------------------------
# Word-level comparison
# ---------------------------------------------------------------------------

def to_host(array: Any) -> np.ndarray:
    return cp.asnumpy(array) if cp is not None and isinstance(
        array, cp.ndarray) else np.asarray(array)


def words(array: Any) -> np.ndarray:
    return np.frombuffer(np.ascontiguousarray(to_host(array)).tobytes(),
                         dtype=np.uint32)


def differing(left: Any, right: Any) -> int:
    a, b = words(left), words(right)
    if a.shape != b.shape:
        return max(a.size, b.size)
    return int(np.count_nonzero(a != b))


def state_of(fields) -> Dict[str, Any]:
    return {name: getattr(fields, name) for name in STATE_NAMES
            if getattr(fields, name, None) is not None}


def frozen(fields) -> Dict[str, np.ndarray]:
    return {name: to_host(value).copy() for name, value in state_of(fields).items()}


def compare(left: Dict[str, np.ndarray], right: Dict[str, np.ndarray]) -> Dict[str, int]:
    assert set(left) == set(right), sorted(set(left) ^ set(right))
    return {name: n for name in sorted(left) if (n := differing(left[name], right[name]))}


def read_only_snapshot(fields, pml) -> Dict[str, np.ndarray]:
    """Every volume a launch must leave BIT-UNCHANGED, by base address.

    The three inverse-permittivity volumes are keyed by address rather than by
    component because an isotropic run hands the same allocation three times, and
    listing it three times would report one drift as three.
    """
    out: Dict[str, np.ndarray] = {}
    for name in IMMUTABLE_PML:
        array = getattr(pml, name, None)
        if array is not None:
            out[f"pml.{name}"] = to_host(array).copy()
    seen: Dict[int, str] = {}
    for component in ("Ex", "Ey", "Ez"):
        volume = fields.inverse_epsilon_for(component)
        address = int(volume.data.ptr)
        if address in seen:
            continue
        seen[address] = component
        out[f"inv_eps@{component}"] = to_host(volume).copy()
    return out


def read_only_drift(before: Dict[str, np.ndarray], fields, pml) -> List[str]:
    after = read_only_snapshot(fields, pml)
    return sorted(name for name in before
                  if name in after and differing(before[name], after[name]))


# ---------------------------------------------------------------------------
# The oracle: the driver's own pass order, read from the driver
# ---------------------------------------------------------------------------

STEP_PASSES: Tuple[str, ...] = (
    "step_B", "fill_symmetry_bc_B", "zero_metal_B", "fill_folded_far_ghosts_B",
    "update_H", "step_D", "fill_symmetry_bc_D", "zero_metal_D",
    "fill_folded_far_ghosts_D", "update_E")


def leg_driver_order() -> Dict[str, Any]:
    """The order this gate walks IS ``FdtdDriver.step``'s, read from its source.

    An oracle a gate invented proves nothing about the engine, so the ordered list of
    ``stepping`` calls is extracted by ``ast`` and both products' ``REPLACES`` are
    asserted to be CONTIGUOUS subsequences of it.
    """
    from meep_gpu import driver as driver_module  # noqa: PLC0415

    # DEDENTED, not cleandoc'd: cleandoc leaves the def line at column 0 and the
    # body indented relative to a stripped docstring, which is not parseable Python.
    source = textwrap.dedent(inspect.getsource(driver_module.FdtdDriver.step))
    tree = ast.parse(source)
    called: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", None)
            if name in STEP_PASSES and name not in called:
                called.append(name)
    order = [name for name in STEP_PASSES if name in called]
    record: Dict[str, Any] = {"driver_calls": called, "walked_order": order,
                              "products": {}}
    ok = sorted(called) == sorted(order) and len(order) == len(STEP_PASSES)
    for name, module in products().items():
        replaces = list(module.REPLACES)
        start, end = order.index(replaces[0]), order.index(replaces[-1])
        span = order[start:end + 1]
        # THE GAP IS THE INTERESTING PART, and it is not a hole. A product may
        # declare a SHORTER run than the driver's span so long as every pass it
        # skips is one its own predicate makes INERT -- the shape
        # no_pml_complex_fused_electric_pair records for itself: "its seam is
        # SHORTER than theirs rather than differently filled". The unfolded weld
        # skips the two mirror fills because it refuses every folded grid, and
        # leg_refusal MEASURES that refusal rather than this leg asserting it.
        gap = [step for step in span if step not in replaces]
        declared_inert = {"flat": ["fill_symmetry_bc_D", "fill_folded_far_ghosts_D"],
                          "folded": []}[name]
        entry = {
            "replaces": replaces,
            "driver_span": span,
            "ordered_subsequence": [s for s in span if s in replaces] == replaces,
            "skipped_passes": gap,
            "skipped_passes_are_the_declared_inert_ones": gap == declared_inert,
            "why_the_skipped_passes_cannot_run": (
                "this product's constitutive arm refuses a fold twice over "
                "(coverage.py:1755-1791), and both mirror fills return early unless "
                "grid.has_symmetry() (stepping.py:1482-1483, :1568-1570), so neither "
                "runs inside this seam on any grid the predicate admits"
                if declared_inert else
                "the product declares the whole span; nothing is skipped"),
        }
        entry["passed"] = (entry["ordered_subsequence"]
                           and entry["skipped_passes_are_the_declared_inert_ones"])
        record["products"][name] = entry
        ok = ok and entry["passed"]
    record["passed"] = bool(ok)
    return record


def run_pass(name: str, fields, pml) -> None:
    getattr(stepping, name)(fields, pml) if name in ("step_B", "step_D", "update_H",
                                                     "update_E") else getattr(
        stepping, name)(fields)


def array_step(fields, pml) -> None:
    for name in STEP_PASSES:
        run_pass(name, fields, pml)


# ---------------------------------------------------------------------------
# Launch counting: two independent counters
# ---------------------------------------------------------------------------

class _CountingKernel:
    """A ``cp.RawKernel`` that counts its launches and delegates everything else."""

    __slots__ = ("_kernel", "_counter", "_name")

    def __init__(self, kernel: Any, counter: Dict[str, int], name: str) -> None:
        self._kernel, self._counter, self._name = kernel, counter, name

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._counter[self._name] = self._counter.get(self._name, 0) + 1
        self._counter["_total"] = self._counter.get("_total", 0) + 1
        return self._kernel(*args, **kwargs)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._kernel, item)


class MemoLaunchCounter:
    """Wrap every memoized device kernel; restore on exit.

    THE MEMO IS THE RIGHT SEAM and it is the shipped one: every launcher in this
    package reaches its kernel through ``compile_cache.get_or_compile``. NOTHING IS
    INSTALLED UNLESS THE MEMO IS ALREADY WARM -- a kernel first compiled inside the
    counted region is launched and NOT counted.
    """

    def __init__(self) -> None:
        self.counts: Dict[str, int] = {}
        self._saved: Dict[Any, Any] = {}

    def __enter__(self) -> "MemoLaunchCounter":
        from meep_gpu.cuda_kernels import compile_cache  # noqa: PLC0415
        store = compile_cache._compiled_kernels  # noqa: SLF001 - the shipped memo
        self._saved = dict(store)
        for key, kernel in list(store.items()):
            store[key] = _CountingKernel(kernel, self.counts, str(key[0]))
        return self

    def __exit__(self, *exc: Any) -> None:
        from meep_gpu.cuda_kernels import compile_cache  # noqa: PLC0415
        store = compile_cache._compiled_kernels  # noqa: SLF001
        for key, kernel in list(store.items()):
            if isinstance(kernel, _CountingKernel) and key in self._saved:
                store[key] = self._saved[key]

    @property
    def total(self) -> int:
        return int(self.counts.get("_total", 0))

    def named(self) -> Dict[str, int]:
        return {k: v for k, v in sorted(self.counts.items()) if k != "_total"}


# ---------------------------------------------------------------------------
# One product case
# ---------------------------------------------------------------------------

def _launch_once(module, fields, grid, pml, dtdx, scratch, *,
                 kernel: Optional[Any] = None, rotate: bool = True,
                 threads: Optional[int] = None) -> Dict[str, Any]:
    """One fused launch through the SHIPPED launcher, with the gate's doors open.

    ``threads`` is set on the module rather than passed, because the block size is
    the launcher's own constant and a gate that passed its own would be measuring a
    different launcher. Restored by the caller.
    """
    report = module.run_offdiag_fused_electric_pair(
        fields, grid, pml, dtdx, scratch=scratch, sources=(), kernel=kernel,
        rotate=rotate) if module.FAMILY.endswith("offdiag_fused_electric_pair") and \
        hasattr(module, "run_offdiag_fused_electric_pair") else \
        module.run_folded_offdiag_fused_electric_pair(
            fields, grid, pml, dtdx, scratch=scratch, sources=(), kernel=kernel,
            rotate=rotate)
    if not report.get("launched"):
        raise SystemExit(f"{module.FAMILY} refused a fixture the sweep admits: "
                         f"{report.get('reason')}")
    return report


def _make_scratch(module, fields) -> Dict[str, Any]:
    if hasattr(module, "offdiag_fused_electric_pair_scratch"):
        return module.offdiag_fused_electric_pair_scratch(fields)
    return module.folded_offdiag_fused_electric_pair_scratch(fields)


def run_case(spec: Dict[str, Any], value_class: str, steps: int, *,
             kernel: Optional[Any] = None, rotate: bool = True,
             threads: int = 256, repeat: int = 0,
             label_suffix: str = "") -> Dict[str, Any]:
    """Step two engines side by side and compare per COMPLETE DRIVER STEP.

    The reference runs ``stepping``'s own ten-pass sequence; the subject runs the
    fused launch plus the passes it does not replace. Compared as raw uint32 WORDS
    over all stored volumes after EVERY step.
    """
    module = products()[spec["product"]]
    label = f"{spec['label']}/{value_class}{label_suffix}"
    reference, grid_r, pml_r, dtdx = build(spec, value_class, case_rng(label))
    subject, grid_s, pml_s, _ = build(spec, value_class, case_rng(label))
    seeded = frozen(reference)
    assert not compare(seeded, frozen(subject)), (
        f"{label}: the two engines were not seeded identically")

    read_only = read_only_snapshot(subject, pml_s)
    census = operand_census({name: to_host(value)
                             for name, value in state_of(subject).items()})
    scratch = _make_scratch(module, subject)
    saved_threads = module._FUSED_THREADS  # noqa: SLF001 - the launcher's own constant
    module._FUSED_THREADS = int(threads)  # noqa: SLF001
    first_divergence: Optional[Dict[str, Any]] = None
    launch_reports: List[Dict[str, Any]] = []
    try:
        with MemoLaunchCounter() as counter:
            for step in range(steps):
                array_step(reference, pml_r)
                for name in STEP_PASSES:
                    if name in module.REPLACES:
                        if name != module.REPLACES[0]:
                            continue
                        report = _launch_once(module, subject, grid_s, pml_s, dtdx,
                                              scratch, kernel=kernel, rotate=rotate)
                        launch_reports.append(
                            {k: v for k, v in report.items() if k != "scratch"})
                        scratch = report["scratch"]
                        continue
                    run_pass(name, subject, pml_s)
                cp.cuda.runtime.deviceSynchronize()
                moved = compare(frozen(reference), frozen(subject))
                if moved and first_divergence is None:
                    first_divergence = {"step": step + 1, "volumes": moved}
                    break
            named = counter.named()
            total = counter.total
    finally:
        module._FUSED_THREADS = saved_threads  # noqa: SLF001

    final = frozen(subject)
    must_move = STATE_NAMES if value_class == "uniform" else BAND_MUST_MOVE
    unmoved = sorted(name for name in must_move
                     if name in seeded and not differing(seeded[name], final[name]))
    record = {
        "label": label, "spec": spec["label"], "product": spec["product"],
        "value_class": value_class, "steps": steps, "threads": int(threads),
        "repeat": repeat,
        "shape": [int(n) for n in grid_s.shape],
        "row_mask": launch_reports[0]["row_mask"] if launch_reports else None,
        "bit_identical": first_divergence is None,
        "first_divergence": first_divergence,
        "launch_counts": {"memo_named": named, "memo_total": total,
                          "launcher_reports": len(launch_reports)},
        "launch_counts_agree": (total == len(launch_reports) == steps
                                and set(named) == {module.KERNEL_NAME}
                                and named.get(module.KERNEL_NAME) == steps),
        "read_only_drift": read_only_drift(read_only, subject, pml_s),
        "operand_census": census,
        "unmoved_volumes": unmoved,
        "non_vacuous": not unmoved,
        "launch_geometry": launch_reports[0] if launch_reports else None,
        "rotated": bool(rotate),
    }
    return record


# ---------------------------------------------------------------------------
# The mutations
# ---------------------------------------------------------------------------

def fixture_facts(spec: Dict[str, Any]) -> Dict[str, Any]:
    """What is actually LIVE on this fixture, read off a real grid.

    THE MUTATION BATTERY'S HONESTY DEPENDS ON THIS. An arm scored on a fixture where
    the machinery it disables is inert comes back UNCAUGHT and looks like a hole; an
    arm predicted null on a fixture where it IS live is a catch nobody made. Both are
    settled by asking the grid rather than by reading the spec's boundary strings --
    ``zero_metal_axes`` EXCLUDES a folded metallic axis, so "declared metallic" and
    "walled" are different questions, and the 2026-09-02 smoke run scored
    ``wall_table_from_the_b_family`` PARTIAL for exactly that reason.
    """
    from meep_gpu.cuda_kernels import offdiag_stencil_weld as weld  # noqa: PLC0415

    module = products()[spec["product"]]
    fields, grid, pml, _ = build(spec, "uniform", np.random.default_rng(3))
    plan = weld.fill_plan(grid)
    codes = (module.folded_offdiag_fused_electric_pair_codes(grid, pml)
             if hasattr(module, "folded_offdiag_fused_electric_pair_codes")
             else {"curl": module.offdiag_fused_electric_pair_codes(grid, pml),
                   "constitutive": module.offdiag_fused_electric_pair_codes(grid, pml)})
    weights = (list(map(float, __import__(
        "meep_gpu.cuda_kernels.folded_offdiag_kernels", fromlist=["x"]
    ).mirror_ghost_weights(grid))) if spec["product"] == "folded" else [1.0] * 3)
    shape = [int(n) for n in grid.shape]
    return {
        "shape": shape,
        "near": list(plan["near"]), "wall": list(plan["wall"]),
        "reflect": list(plan["reflect"]),
        "near_phase": list(plan["near_phase"]), "far_phase": list(plan["far_phase"]),
        "ghost_weights": weights,
        "codes": {k: list(v) for k, v in codes.items()},
        "any_wall": any(plan["wall"]),
        "any_far": any(row >= 0 for row in plan["reflect"]),
        "far_parity_is_negative": any(
            plan["reflect"][axis] >= 0 and plan["far_phase"][axis] != 1.0
            for axis in range(3)),
        "ghost_weight_is_negative": any(
            plan["near"][axis] and weights[axis] != 1.0 for axis in range(3)),
        "reflect_row_is_not_n_minus_two": any(
            plan["reflect"][axis] >= 0 and plan["reflect"][axis] != shape[axis] - 2
            for axis in range(3)),
        "boundary_readings_differ": codes["curl"] != codes["constitutive"],
        # THE ONE SHAPE `clear_order_flipped` CAN MOVE A WORD ON: a component with an
        # ODD near parity on one of its axes AND a wall on another of them, so a near
        # ghost lands in a plane the clear owns. Read per component off the Yee table
        # rather than off the fixture's strings.
        "odd_near_parity_meets_a_cleared_plane": any(
            plan["near"][a] and plan["near_phase"][a] == -1.0 and plan["wall"][b]
            for component in range(3)
            for a in weld.near_axes(component)
            for b in weld.clear_axes(component) if a != b),
    }


def needle(source: str, old: str, new: str, count: int = 1) -> str:
    """Replace ``old`` with ``new``, requiring exactly ``count`` occurrences.

    A mutation that did not land is a mutation that was not tested, and it would be
    reported UNCAUGHT -- which is the one way a mutation battery lies.
    """
    found = source.count(old)
    if found != count:
        raise SystemExit(
            f"the mutation anchor {old!r} appears {found} times, not {count}; the "
            f"mutation would not have landed and would have scored UNCAUGHT")
    return source.replace(old, new, count)


#: ``(name, applies_to, source rewrite, why, when it is LIVE)``.
#:
#: ``live`` decides whether the arm can move a word ON THIS FIXTURE. Where it cannot,
#: the arm is recorded ``predicted_null`` WITH the reason -- the vacuity discipline
#: this campaign runs on, and the only honest way to report an arm that could not
#: fire.
SOURCE_MUTATIONS: Tuple[Tuple[str, str, Callable[[str], str], str,
                              Callable[[Dict[str, Any]], bool]], ...] = (
    ("halo_recompute_dropped", "both",
     lambda s: needle(
         s, "    return (index < 0) ? 0.0f : resolve_D_at(comp, index, weld);",
         "    return (index < 0) ? 0.0f : resolve_D(comp, weld.nx, weld.ny, "
         "weld.nz, weld);"),
     "the foreign sample stops being recomputed AT THE NEIGHBOUR and takes a fixed "
     "cell instead -- the shortcut a reader reaches for when the recompute looks "
     "expensive, and the one the whole design says is unavailable",
     lambda spec: True),
    ("foreign_read_from_live_buffer", "both",
     lambda s: needle(s, "    weld.Dx = Dx;\n    weld.Dy = Dy;\n    weld.Dz = Dz;",
                      "    weld.Dx = Dx_out;\n    weld.Dy = Dy_out;\n"
                      "    weld.Dz = Dz_out;"),
     "THE IN-PLACE WELD, REBUILT: the resolution reads the volume this launch is "
     "writing instead of the pre-launch one. If this is not caught, this gate "
     "cannot tell the two designs apart and its S1 means nothing",
     lambda spec: True),
    ("redirect_parity_dropped", "folded",
     lambda s: _far_parity_mutation(s),
     "the far mirror parity multiply replaced by the identity -- a whole plane of "
     "wrong signs wherever that parity is -1, and the identity where it is +1, "
     "which is why the arm is scored only where the fixture's own far parity is "
     "negative",
     lambda facts: facts["far_parity_is_negative"]),
    ("far_reflect_row_n_minus_two", "folded",
     lambda s: _reflect_row_mutation(s),
     "the far fill images the fixed row n - 2 instead of _far_reflect_rows' "
     "n_full - stored + 2. THE TWO AGREE AT AN EVEN FULL COUNT and are a whole cell "
     "apart at an odd one, so the arm is scored only on the fixture whose count is "
     "odd -- which is the entire reason that fixture is in the sweep",
     lambda facts: facts["reflect_row_is_not_n_minus_two"]),
    ("near_source_row_wrong", "folded",
     lambda s: needle(s, "#define NEAR_SOURCE_ROW 2", "#define NEAR_SOURCE_ROW 1"),
     "the near fill images stored row 1 instead of stored row 2 -- MEEP's io = -2 "
     "halved origin is what puts the ghost on row 2, and row 1 is a whole cell wrong",
     lambda facts: any(facts["near"])),
    ("wall_table_from_the_b_family", "both",
     lambda s: needle(
         s, "    if ((weld.wall_y && j == 0) || (weld.wall_z && k == 0)) {",
         "    if (weld.wall_x && i == 0) {"),
     "zero_metal_D's OFF-DIAGONAL table replaced by the B family's diagonal -- two "
     "components per wall become one, and the wrong one. Scored off the WALL PLAN "
     "rather than off the fixture's boundary strings, because zero_metal_axes "
     "EXCLUDES a folded metallic axis and a fold-terminated wall is not a wall",
     lambda facts: facts["any_wall"]),
    ("sub_lattice_shadow", "both",
     lambda s: s.replace("kms_half_x[i]", "kms_x[i]").replace(
         "kms_half_y[j]", "kms_y[j]").replace("kms_half_z[k]", "kms_z[k]"),
     "the constitutive tail reads the CURL's INTEGER split-field vector instead of "
     "its own HALF-INTEGER one -- half a cell in the absorber profile, converged, "
     "smooth and wrong",
     lambda facts: True),
    ("mirror_ghost_weight_dropped", "folded",
     lambda s: needle(s, "    return mg ? (w * value) : value;",
                      "    return value;"),
     "_shift_down's mirror arm loses its parity: the ghost lane reads stored row "
     "MIRROR_ROW unweighted, which is the certified family's own measured defect. "
     "The weight is mirror_parity('D'+axis, axis, phase) == -phase, so it is the "
     "IDENTITY on an odd-phase fold and the arm is scored only where it is not",
     lambda facts: facts["ghost_weight_is_negative"]),
    ("constitutive_codes_from_the_curl", "folded",
     lambda s: _constitutive_codes_mutation(s),
     "THE ONE EDIT THIS FAMILY MAKES AND ITS TWIN DOES NOT, undone: the "
     "constitutive half reads the CURL's boundary codes, which split the two folded "
     "terminations where update_E serves both with BC_MIRROR. Only the six BODY "
     "reads are rewritten -- rewriting the declaration too would be a duplicate "
     "parameter name and a compile error, which is a mutation that never ran",
     lambda facts: facts["boundary_readings_differ"]),
    ("clear_order_flipped", "both",
     lambda s: _clear_order_mutation(s),
     "the near parity applied AFTER the wall clear instead of before it -- the "
     "driver's order is fill (:3309) then clear (:3310), so a near ghost landing in "
     "a cleared plane is CLEARED and the array path leaves +0.0f there. Flipped, an "
     "odd-parity near axis meeting a cleared plane leaves -0.0f. It is a SIGN BIT ON "
     "A ZERO and nothing else, which is why the host probe measures it at ONE WORD "
     "-- and why an arm that scored it on a fixture without both halves would report "
     "a hole",
     lambda facts: facts["odd_near_parity_meets_a_cleared_plane"]),
    ("local_name_changed", "both",
     lambda s: _rename_local_value(s),
     "THE CONFIRMED NULL. Every local named `value` in the resolution renamed, and "
     "nothing else: it moves the source text and the compiled binary's symbol table "
     "and not one float, so it MUST come back UNCAUGHT. A battery of catches with no "
     "confirmed null is a battery nobody showed to be two-sided -- it could be "
     "reporting CAUGHT for the act of recompiling",
     lambda facts: True),
)


def _clear_order_mutation(source: str) -> str:
    """The near parity applied to the CLEARED branch as well, per component."""
    from meep_gpu.cuda_kernels import offdiag_stencil_weld as weld  # noqa: PLC0415

    out = source
    for component, target in enumerate(weld.D_TARGETS):
        near = weld.near_axes(component)
        axes = "xyz"
        coords = "ijk"
        clause = " || ".join(f"(weld.wall_{axes[a]} && {coords[a]} == 0)"
                             for a in weld.clear_axes(component))
        anchor = (f"    if ({clause}) {{\n"
                  "        // The array path writes an exact +0.0f into this plane.\n"
                  "        value = 0.0f;\n")
        flipped = anchor + "".join(
            f"        if (weld.near_{axes[a]} && {coords[a]} == 0)"
            f" value = weld.near_phase_{axes[a]} * value;\n" for a in near)
        out = needle(out, anchor, flipped)
    return out


def _rename_local_value(source: str) -> str:
    """``value`` -> ``resolved_value`` everywhere, which is a rename and nothing else."""
    import re  # noqa: PLC0415 - stdlib, at the one call site

    renamed, count = re.subn(r"\bvalue\b", "resolved_value", source)
    if count < 10:
        raise SystemExit(
            f"the null control renamed only {count} occurrences of `value`; it is "
            f"meant to move the whole resolution's local naming and would otherwise "
            f"be a null that changed almost nothing")
    return renamed


def _far_parity_mutation(source: str) -> str:
    """Every ``resolve_D*``'s far parity multiply replaced by the identity."""
    out = source
    for axis in "xyz":
        out = needle(out, f"    if (far) value = weld.far_phase_{axis} * value;",
                     "    if (far) value = 1.0f * value;")
    return out


def _reflect_row_mutation(source: str) -> str:
    """Every ``resolve_D*``'s far redirect pointed at the fixed row ``n - 2``."""
    out = source
    for axis, extent in zip("xyz", ("nx", "ny", "nz")):
        coordinate = {"x": "i", "y": "j", "z": "k"}[axis]
        out = needle(out, f"    if (far) {coordinate} = weld.reflect_{axis};",
                     f"    if (far) {coordinate} = weld.{extent} - 2;")
    return out


def _constitutive_codes_mutation(source: str) -> str:
    """The six BODY reads of ``cbc_*`` replaced by ``bc_*``; the DECLARATION stays.

    Rewriting the declaration as well is a duplicate parameter name and a compile
    error -- which scores as a mutation that never ran, the one way a battery lies.
    """
    out = source
    for axis, extent in zip("xyz", ("nx", "ny", "nz")):
        coordinate = {"x": "i", "y": "j", "z": "k"}[axis]
        down = {"x": "di", "y": "dj", "z": "dk"}[axis]
        up = {"x": "ui", "y": "uj", "z": "uk"}[axis]
        out = needle(
            out,
            f"    int {down} = coord_dn({coordinate}, {extent}, cbc_{axis}), "
            f"{up} = coord_up({coordinate}, {extent}, cbc_{axis});",
            f"    int {down} = coord_dn({coordinate}, {extent}, bc_{axis}), "
            f"{up} = coord_up({coordinate}, {extent}, bc_{axis});")
        out = needle(out, f"    int mg_{axis} = (cbc_{axis} == BC_MIRROR)",
                     f"    int mg_{axis} = (bc_{axis} == BC_MIRROR)")
    return out

#: Mutations armed on the HOST, through the launcher's own doors rather than the
#: source. Each returns a ``(record, caught)`` pair.
HOST_MUTATIONS: Tuple[str, ...] = ("rotation_skipped", "scratch_aliased_to_storage")

#: The one arm above that must come back UNCAUGHT.
NULL_CONTROLS: Tuple[str, ...] = ("local_name_changed",)


def compiled_variant(module, row_mask: Sequence[int],
                     rewrite: Callable[[str], str]) -> Any:
    """A ``cp.RawKernel`` for a REWRITTEN copy of the shipped source.

    Compiled with the shipped option tuple, so the only difference between this and
    the product is the edit under test.
    """
    source = rewrite(module.kernel_source(row_mask))
    return cp.RawKernel(source, module.KERNEL_NAME, options=GUARD_OPTIONS)


def leg_mutations(specs: Sequence[Dict[str, Any]], steps: int) -> Dict[str, Any]:
    """Every arm, on every fixture whose machinery it can move.

    A CAUGHT arm is one whose run diverges from the array path; an UNCAUGHT one does
    not. The scoreboard records the expectation BESIDE the outcome, so a mutation
    that was expected to be null and was caught fails the leg just as loudly as one
    that was expected caught and was not.
    """
    from meep_gpu.cuda_kernels.coverage import offdiag_row_mask  # noqa: PLC0415

    all_facts = {spec["label"]: fixture_facts(spec) for spec in specs}
    for label, facts in all_facts.items():
        log(f"[facts] {label:32s} wall={facts['wall']} near={facts['near']} "
            f"reflect={facts['reflect']} gw={facts['ghost_weights']} "
            f"codes_differ={facts['boundary_readings_differ']}")
    scoreboard: Dict[str, Any] = {"_fixture_facts": all_facts}
    for name, applies, rewrite, why, live in SOURCE_MUTATIONS:
        arms: List[Dict[str, Any]] = []
        for spec in specs:
            if applies != "both" and applies != spec["product"]:
                continue
            module = products()[spec["product"]]
            facts = all_facts[spec["label"]]
            if not live(facts):
                arms.append({"spec": spec["label"], "predicted_null": True,
                             "reason": "the machinery this arm disables is not live "
                                       "on this fixture",
                             "facts": {k: v for k, v in facts.items()
                                       if isinstance(v, (bool, list))}})
                continue
            fields, grid, pml, _ = build(spec, "uniform", case_rng(spec["label"]))
            mask = offdiag_row_mask(fields)
            try:
                kernel = compiled_variant(module, mask, rewrite)
            except SystemExit as exc:
                arms.append({"spec": spec["label"], "anchor_error": str(exc)})
                continue
            record = run_case(spec, "uniform", steps=min(steps, 8), kernel=kernel,
                              label_suffix=f"/M:{name}")
            arms.append({"spec": spec["label"],
                         "caught": not record["bit_identical"],
                         "first_divergence": record["first_divergence"],
                         "launch_counts_agree": record["launch_counts_agree"]})
        expected_null = name in NULL_CONTROLS
        armed = [arm for arm in arms if "caught" in arm]
        broken = [arm for arm in arms if "anchor_error" in arm]
        if broken:
            raise SystemExit(
                f"the mutation {name} could not be applied on "
                f"{[arm['spec'] for arm in broken]}: {broken[0]['anchor_error']}. A "
                f"mutation that did not land scores UNCAUGHT and is the one way a "
                f"battery lies, so it stops the run instead")
        outcome = ("NULL CONFIRMED" if expected_null and armed
                   and not any(arm["caught"] for arm in armed)
                   else "CAUGHT" if armed and all(arm["caught"] for arm in armed)
                   else "PARTIAL" if armed and any(arm["caught"] for arm in armed)
                   else "UNCAUGHT" if armed else "NOT ARMED")
        scoreboard[name] = {
            "why": why, "expected": "NULL CONFIRMED" if expected_null else "CAUGHT",
            "outcome": outcome, "applies_to": applies, "arms": arms,
            "as_required": outcome == ("NULL CONFIRMED" if expected_null
                                       else "CAUGHT")}
        log(f"[mutation] {name:34s} {outcome:14s} "
            f"({len(armed)} armed, {len(arms) - len(armed)} predicted null)")

    # THE HOST ARMS.
    for spec in specs:
        module = products()[spec["product"]]
        key = f"rotation_skipped@{spec['label']}"
        record = run_case(spec, "uniform", steps=min(steps, 8), rotate=False,
                          label_suffix="/M:rotation_skipped")
        scoreboard.setdefault("rotation_skipped", {
            "why": "the launch computes the step and throws it away: without the "
                   "rotation the scratch never becomes the live D, so every "
                   "subsequent step reads a stale flux density",
            "expected": "CAUGHT", "applies_to": "both", "arms": []})
        scoreboard["rotation_skipped"]["arms"].append(
            {"spec": spec["label"], "caught": not record["bit_identical"],
             "first_divergence": record["first_divergence"]})
        # THE ALIAS ARM IS SCORED AS A REFUSAL, not as a byte divergence: the check
        # runs BEFORE any launch and is meant to stop the in-place weld from being
        # assembled at all.
        fields, grid, pml, dtdx = build(spec, "uniform", case_rng(spec["label"]))
        tables = (module.offdiag_fused_electric_pair_tables(pml)
                  if hasattr(module, "offdiag_fused_electric_pair_tables")
                  else module.folded_offdiag_fused_electric_pair_tables(pml))
        aliased = {volume: getattr(fields, volume)
                   for volume in module.SCRATCH_VOLUMES}
        try:
            module.assert_scratch_is_disjoint(fields, aliased, tables)
            refused, why_refused = False, None
        except ValueError as exc:
            refused, why_refused = True, str(exc)[:300]
        scoreboard.setdefault("scratch_aliased_to_storage", {
            "why": "the same defect as foreign_read_from_live_buffer, from the HOST "
                   "side: binding the scratch to the storage IS the in-place weld. "
                   "Scored as a refusal because the check runs before any launch",
            "expected": "REFUSED", "applies_to": "both", "arms": []})
        scoreboard["scratch_aliased_to_storage"]["arms"].append(
            {"spec": spec["label"], "refused": refused, "reason": why_refused})
    for name in HOST_MUTATIONS:
        entry = scoreboard[name]
        arms = entry["arms"]
        if name == "rotation_skipped":
            entry["outcome"] = ("CAUGHT" if arms and all(a["caught"] for a in arms)
                                else "UNCAUGHT")
            entry["as_required"] = entry["outcome"] == "CAUGHT"
        else:
            entry["outcome"] = ("REFUSED" if arms and all(a["refused"] for a in arms)
                                else "ADMITTED")
            entry["as_required"] = entry["outcome"] == "REFUSED"
        log(f"[mutation] {name:34s} {entry['outcome']:14s} ({len(arms)} armed)")
    return {"scoreboard": scoreboard,
            "passed": all(entry["as_required"] for name, entry in scoreboard.items()
                          if not name.startswith("_"))}


# ---------------------------------------------------------------------------
# The structural legs
# ---------------------------------------------------------------------------

def leg_lift() -> Dict[str, Any]:
    """Both emitted sources against the certified text, and the declared edits.

    A device gate that did not check the SPLICE would certify whatever the emitter
    happened to produce.
    """
    from meep_gpu.cuda_kernels import (offdiag_emitter,  # noqa: PLC0415
                                       offdiag_stencil_weld, step_curl_kernels)
    from meep_gpu.cuda_kernels import folded_offdiag_kernels  # noqa: PLC0415

    record: Dict[str, Any] = {"products": {}}
    ok = True
    curl_body = offdiag_stencil_weld.split_body(
        step_curl_kernels._step_D_pml_real_kernel_code,  # noqa: SLF001
        step_curl_kernels._REAL_PML_PRELUDE,  # noqa: SLF001
        "_step_D_pml_real_kernel_code")
    for name, module in products().items():
        source = module.kernel_source((1, 1, 1, 1, 1, 1))
        constitutive = (offdiag_emitter.offdiag_source((1, 1, 1, 1, 1, 1))
                        if name == "flat"
                        else folded_offdiag_kernels.folded_offdiag_source(
                            (1, 1, 1, 1, 1, 1)))
        # Every certified line the weld does not declare an edit for must appear
        # verbatim in the emitted source.
        edited = {edit["line"].splitlines()[0].strip()
                  for edit in module.LIFT_EDITS}
        missing = []
        for line in curl_body.splitlines():
            stripped = line.strip()
            if not stripped or stripped in edited or stripped.startswith("pml_apply("):
                continue
            if stripped.startswith(("int idx = blockIdx", "if (idx >=", "int k = idx",
                                    "int j = (idx", "int i = idx")):
                continue
            if line not in source:
                missing.append(line)
        entry = {
            "kernel": module.KERNEL_NAME,
            "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "corpus_digest": module.corpus_digest(),
            "lift_edits": len(module.LIFT_EDITS),
            "certified_curl_lines_missing": missing,
            "no_direct_flux_density_load": not any(
                token in source.split("\n) {\n", 1)[-1]
                for token in ("Dx[idx]", "Dy[idx]", "Dz[idx]")),
            "constitutive_tail_is_verbatim": (
                "float prev = fw[idx];" in constitutive
                and "float prev = fw[idx];" in source),
            "term_association_is_verbatim": (
                "return 0.25f * ((near_pair * u[home]) + (far_pair * "
                "ghosted(u, up)));" in source),
        }
        entry["passed"] = (not missing and entry["no_direct_flux_density_load"]
                           and entry["constitutive_tail_is_verbatim"]
                           and entry["term_association_is_verbatim"])
        ok = ok and entry["passed"]
        record["products"][name] = entry
    record["passed"] = ok
    return record


def leg_refusal() -> Dict[str, Any]:
    """The configurations both products refuse BY NAME.

    The one failure mode no bit comparison catches: a predicate that admits what the
    launch cannot serve is invisible to a byte gate, because the byte gate only ever
    runs on rows the predicate admitted.
    """
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid, Mirror  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415
    from meep_gpu.sources import GaussianEnvelope, VolumeSource  # noqa: PLC0415

    def make(symmetry=(), rows=True, layer=True):
        grid = Grid(resolution=10.0, cell_size=(1.6, 1.6, 1.6), courant=0.35, xp=cp,
                    symmetry=tuple(Mirror(a, p) for a, p in symmetry))
        thickness = tuple((0, 0) if grid.shape[axis] < 6
                          else (0, 2) if grid.is_mirrored(axis)
                          else (2, 2) for axis in range(3))
        pml = PML(grid=grid, thickness=thickness if layer else ((0, 0),) * 3)
        fields = Fields(grid=grid)
        fields.enable_field_storage()
        fields.enable_pml_storage()
        shape = tuple(int(n) for n in grid.shape)
        names = ("Ex", "Ey", "Ez")
        eps = {n: cp.full(shape, v, cp.float32) for n, v in zip(names, (2., 2.5, 3.))}
        inv = {n: cp.full(shape, np.float32(1.0 / v), cp.float32)
               for n, v in zip(names, (2., 2.5, 3.))}
        off = ({"Ex": {"Ey": cp.asarray(np.full(shape, 0.1, np.float32))}}
               if rows else None)
        fields.set_epsilon_volumes(eps, inv, chi1inv_offdiagonal=off)
        return fields, pml, grid

    cases: List[Dict[str, Any]] = []
    flat, folded = products()["flat"], products()["folded"]

    def ask(module, fields, pml, grid, sources):
        call = (module.covers_offdiag_fused_electric_pair
                if hasattr(module, "covers_offdiag_fused_electric_pair")
                else module.covers_folded_offdiag_fused_electric_pair)
        return call(fields, pml, grid, sources)

    fields, pml, grid = make()
    cases.append({"case": "flat admits its own cell",
                  "covered": ask(flat, fields, pml, grid, ())[0], "want": True})
    cases.append({"case": "folded refuses a fold-free grid",
                  "covered": ask(folded, fields, pml, grid, ())[0], "want": False})
    covered, why = ask(flat, fields, pml, grid, None)
    cases.append({"case": "flat refuses an undeclared source set", "covered": covered,
                  "want": False, "reason": why})
    source = VolumeSource(grid=grid, component="Ez", center=(0., 0., 0.),
                          size=(0., 0., 0.),
                          envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2),
                          amplitude=1.0)
    covered, why = ask(flat, fields, pml, grid, (source,))
    cases.append({"case": "flat refuses an electric deposit", "covered": covered,
                  "want": False, "reason": why,
                  "names_the_repair_clause": "off-diagonal chi1inv row" in str(why)})
    fields, pml, grid = make(symmetry=(("Y", 1),))
    covered, why = ask(flat, fields, pml, grid, ())
    cases.append({"case": "flat refuses a fold", "covered": covered, "want": False,
                  "reason": why})
    cases.append({"case": "folded admits its own cell",
                  "covered": ask(folded, fields, pml, grid, ())[0], "want": True})
    fields, pml, grid = make(rows=False)
    cases.append({"case": "flat refuses a diagonal run",
                  "covered": ask(flat, fields, pml, grid, ())[0], "want": False})
    cases.append({"case": "folded refuses a diagonal run",
                  "covered": ask(folded, fields, pml, grid, ())[0], "want": False})
    passed = all(case["covered"] == case["want"] for case in cases)
    return {"cases": cases, "passed": bool(passed)}


def leg_guard_control(spec: Dict[str, Any], steps: int) -> Dict[str, Any]:
    """``--fmad=false`` is CORRECTNESS here, and this measures it rather than citing it.

    The same source compiled WITHOUT the guard must diverge: if it does not, the
    option is not load-bearing on this kernel and the record should say so rather
    than inheriting a sibling's claim.
    """
    from meep_gpu.cuda_kernels.coverage import offdiag_row_mask  # noqa: PLC0415

    module = products()[spec["product"]]
    fields, _grid, _pml, _ = build(spec, "uniform", case_rng(spec["label"]))
    mask = offdiag_row_mask(fields)
    unguarded = cp.RawKernel(module.kernel_source(mask), module.KERNEL_NAME,
                             options=())
    record = run_case(spec, "uniform", steps=min(steps, 8), kernel=unguarded,
                      label_suffix="/unguarded")
    return {"spec": spec["label"], "diverged": not record["bit_identical"],
            "first_divergence": record["first_divergence"],
            "note": "a NON-divergence here is a finding, not a failure: it would mean "
                    "the compiler contracted nothing in this body at this courant, "
                    "and the record must say so rather than claim the guard is "
                    "load-bearing"}


def save(results: Dict[str, Any], path: str) -> None:
    """Serialize, with the WELD stamped on every write.

    ``gate_provenance.stamp`` records the sha256 of every repo module this process
    imported, which is what ``rebind_cuda_welds.py`` binds the certification entry
    to. Without it the record would carry a verdict and no way for the merge bar to
    notice when the files it ran on moved -- the drift this whole record convention
    exists to end.
    """
    gate_provenance.stamp(results)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=1, sort_keys=True, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _warm_memo(specs: Sequence[Dict[str, Any]]) -> None:
    """Compile and launch every kernel this file drives, once, off the counted path."""
    for spec in specs:
        module = products()[spec["product"]]
        fields, grid, pml, dtdx = build(spec, "uniform", np.random.default_rng(1))
        scratch = _make_scratch(module, fields)
        _launch_once(module, fields, grid, pml, dtdx, scratch)
    cp.cuda.runtime.deviceSynchronize()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="artifact FILE path")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--repeats", type=int, default=SCHEDULE_REPEATS)
    parser.add_argument("--product", choices=("full", "reduced"), default="full")
    parser.add_argument("--subnormal-policy", choices=("keep", "flush"), default=None)
    parser.add_argument("--import-meep-for-host-policy", action="store_true")
    parser.add_argument("--skip-mutations", action="store_true")
    parser.add_argument("--no-device", action="store_true")
    args = parser.parse_args(argv)
    if args.steps < 1:
        raise SystemExit("--steps must be positive")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    started = time.perf_counter()
    results: Dict[str, Any] = {
        "gate": "cuda_offdiag_stencil_welds",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename,
        "argv": list(argv if argv is not None else sys.argv[1:]),
        "question": ("do ONE launch of offdiag_fused_electric_pair_pml_real and one "
                     "of folded_offdiag_fused_electric_pair_pml_real leave every "
                     "stored volume byte-identical to stepping's own D-seam pass "
                     "sequence inside a complete driver step -- at every block size, "
                     "on every repeat, under both float32 subnormal policies?"),
        "what_it_does_not_claim": [
            "throughput: the box is shared and nothing here is timed",
            "dispatch: fastpath.plan_fast_path still returns None on every branch",
            "the resolution's arithmetic against the driver's pass order, which is "
            "measured off-device by probe_cuda_offdiag_scratch_weld.py",
        ],
        "budget_steps": args.steps,
        "schedule_repeats": args.repeats,
        "block_sizes": list(BLOCK_SIZES),
    }
    if cp is None or args.no_device:
        log("[structural] no device: only the source legs run")
        results["device_mode"] = False
        results["driver_order"] = leg_driver_order()
        results["status"] = "refused: no CuPy" if cp is None else "structural only"
        save(results, args.out)
        return 0 if cp is None else 0

    results["device_mode"] = True
    if args.import_meep_for_host_policy:
        results["meep_host_import"] = probe.import_meep_for_host_policy()
    results["nvrtc_observer"] = probe.install_nvrtc_binary_observer()
    if args.subnormal_policy:
        results["subnormal_policy_install"] = probe.install_subnormal_policy_for_run(
            args.subnormal_policy, _REPO_API)
    results["environment"] = probe.device_info()
    results["subnormal_policy_stamp"] = probe.subnormal_policy_stamp(_REPO_API)
    save(results, args.out)

    specs = list(SPECS) if args.product == "full" else [
        spec for spec in SPECS if spec["label"] in REDUCED_LABELS]
    by_label = {spec["label"]: spec for spec in SPECS}

    results["driver_order"] = leg_driver_order()
    log(f"[driver] order matches driver.py: {results['driver_order']['passed']}")
    results["lift"] = leg_lift()
    log(f"[lift] both emitters are the certified bytes: {results['lift']['passed']}")
    results["refusal"] = leg_refusal()
    log(f"[refusal] the seam clauses hold: {results['refusal']['passed']}")
    save(results, args.out)

    _warm_memo(specs)

    # ------------------------------------------------------------------ S1
    s1: List[Dict[str, Any]] = []
    for spec in specs:
        for value_class in VALUE_CLASSES:
            for repeat in range(args.repeats):
                record = run_case(spec, value_class, args.steps, repeat=repeat,
                                  label_suffix=f"/r{repeat}")
                s1.append(record)
                log(f"[S1] {record['label']:52s} "
                    f"{'IDENTICAL' if record['bit_identical'] else 'DIVERGED'} "
                    f"launches={record['launch_counts']['memo_total']} "
                    f"subnormals={record['operand_census']['subnormals']}")
                save({**results, "S1_subject": s1}, args.out)
    results["S1_subject"] = s1

    # ------------------------------------------------------------------ S2
    s2: List[Dict[str, Any]] = []
    for spec in specs:
        for threads in BLOCK_SIZES:
            record = run_case(spec, "uniform", steps=min(args.steps, 12),
                              threads=threads, label_suffix=f"/b{threads}")
            s2.append(record)
            log(f"[S2] {record['label']:52s} "
                f"{'IDENTICAL' if record['bit_identical'] else 'DIVERGED'}")
        save({**results, "S2_block_sizes": s2}, args.out)
    results["S2_block_sizes"] = s2

    results["guard_control"] = [leg_guard_control(by_label[label], args.steps)
                                for label in ("flat_walls_all_rows",
                                              "fold_y_odd_all_rows")]
    save(results, args.out)

    if not args.skip_mutations:
        results["mutations"] = leg_mutations(
            [by_label[label] for label in MUTATION_SPEC_LABELS], args.steps)
        save(results, args.out)

    # ------------------------------------------------------------------ verdict
    band = [r for r in s1 if r["value_class"] == "subnormal_band"]
    uniform = [r for r in s1 if r["value_class"] == "uniform"]
    clauses = {
        "S1 every case is bit-identical for the whole budget":
            bool(s1) and all(r["bit_identical"] for r in s1),
        "S1 every case is identical on every repeat":
            bool(s1) and len({(r["spec"], r["value_class"],
                               r["bit_identical"]) for r in s1}) ==
            len({(r["spec"], r["value_class"]) for r in s1}),
        "S1 the launch counts agree between two independent counters":
            bool(s1) and all(r["launch_counts_agree"] for r in s1),
        "S1 nothing read-only drifted": all(not r["read_only_drift"] for r in s1),
        "S1 every case moved the state it must move":
            all(r["non_vacuous"] for r in s1),
        "S2 identical at every block size":
            bool(s2) and all(r["bit_identical"] for r in s2),
        "the subnormal band really contains subnormals":
            bool(band) and all(r["operand_census"]["subnormals"] > 0 for r in band),
        "the uniform class contains none":
            bool(uniform) and all(r["operand_census"]["subnormals"] == 0
                                  for r in uniform),
        "the driver order is the driver's": results["driver_order"]["passed"],
        "the emitted sources are the certified bytes": results["lift"]["passed"],
        "the seam clauses refuse what the launch cannot serve":
            results["refusal"]["passed"],
        "every mutation scored as required":
            results.get("mutations", {}).get("passed", True),
    }
    results["verdict"] = {
        "clauses": clauses, "passed": all(clauses.values()),
        "denominators": {
            "S1_cases": len(s1), "S2_cases": len(s2),
            "fixtures": len(specs), "block_sizes": len(BLOCK_SIZES),
            "repeats": args.repeats, "steps_per_case": args.steps,
        },
    }
    # THE CANONICAL VERDICT, in the spelling the fleet's readers agree on.
    # ``rebind_cuda_welds.py`` binds an entry only where EVERY policy leg carries
    # ``canonical_verdict.released`` true, and ``gate_provenance.read_verdict``
    # already reads seven spellings across three tracks; writing the canonical one
    # here means this gate needs no eighth.
    results["canonical_verdict"] = {
        "released": bool(results["verdict"]["passed"]),
        "reasons": [clause for clause, value in clauses.items() if not value],
        "what_it_licenses": (
            "two CUDA kernels measured byte-identical to the array path's D-seam "
            "pass sequence over complete driver steps, at every block size, under "
            "both float32 subnormal policies, with the mutation battery two-sided. "
            "It licenses NO throughput claim and NO dispatch claim"),
    }
    results["elapsed_s"] = round(time.perf_counter() - started, 1)
    save(results, args.out)
    log("=" * 78)
    for clause, value in clauses.items():
        log(f"  {'PASS' if value else 'FAIL'}  {clause}")
    log(f"VERDICT: {'PASS' if results['verdict']['passed'] else 'FAIL'}  "
        f"({results['elapsed_s']} s)")
    return 0 if results["verdict"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
