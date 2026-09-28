"""Gate: ``meep_gpu.cuda_kernels.complex_fused_hd_pair`` -- the COMPLEX H->D weld.

ONE CUDA kernel per variant welding ``stepping.update_H`` into ``stepping.step_D``
under complex64 storage, measured BIT-IDENTICAL as uint32 words per COMPLETE DRIVER
STEP against FOUR independent arrangements of the seam, under both float32 subnormal
policies, on synthetic fixtures and on lifted corpus rows.

THE TWO CELLS, AND WHY ONE GATE MEASURES BOTH
=============================================================================
``results/fusion_matrix_cuda_2026-09-07_cyl/fusion_matrix_cuda.json``,
``h_to_d_seam.instances`` filtered to ``buildable_not_built``:

* ``(cuda_complex/complex -> cuda_complex/complex)`` -- 17 instances (``plain``);
* ``(cuda_complex/complex -> cuda_complex_folded/folded complex)`` -- 5 (``folded``).

The H half is the SAME certified kernel on both (``complex_folded_kernels`` ships no
constitutive kernel at all), and the D half differs only by that module's three
anchored text deltas. So the product emits two variants of one transform and this gate
drives both: every fixture carries its variant, the transcription leg emits and
compares both texts, and the lift leg names the cell each corpus row sits in.

THE FOUR REFERENCES, and what each one rules out
=============================================================================
``array``               ``stepping``'s own ten passes. THE ORACLE.
``singles``             the two certified complex kernels at ``update_H`` and
                        ``step_D``, everything else on the array path. Rules out a
                        divergence that is the certified halves' rather than the
                        weld's.
``composition_today``   what the composer BUILDS on these rows today: the released
                        complex B->H pair and the released complex D->E pair -- and on
                        a folded fixture their FOLDED siblings, which is what the
                        census selects there. Rules out "identical to a composition
                        nobody runs".
``unfused``             all four slot-path slots as certified singles. The composition
                        a composer with fusion vetoed would build.
``weld``                the subject.

Plus ``weld_composed`` on the launch-structure leg only: the weld with its two
neighbours as certified singles, so launch counts compare like with like.

WHAT THE ARITHMETIC LEGS ADD THAT A WHOLE-STEP IDENTITY CANNOT
=============================================================================
CuPy's ``complex64 * float32`` is the four-product form CONTRACTED by NVRTC
(``lanes/cyl_round/cupy_probe``): the uncontracted transcription differs on the SIGN
OF A FLUSHED ZERO -- 6 words of 4,005,000, all on tiny normals at a row the multiply
underflows. A RANDOM battery does not reach that class. The ``spelling`` leg therefore
runs the family's own multiply against ``cupy.multiply`` on a PLANTED row-0
tiny-normal fixture beside the ordinary classes, and the mutation battery arms the
naive four-product form as a device defect on that same plant.

**THIS FAMILY PERFORMS NO DIVISION** and the ``spelling`` leg records that as a
measured absence rather than omitting it: the reciprocal the recurrence needs is
``sinv``, computed host-side by ``PML``. The complex divide spelling that the
cylindrical H->D products must get right (CuPy's scaled algorithm, not numpy's
reciprocal multiply) has no site here, and the leg asserts the emitted text contains
no ``/`` on a complex value.

NOT WIRED, AND THE ARBITRATION LEG SAYS SO IN BOTH ARRANGEMENTS
=============================================================================
The family is not in ``fused_pairs.FUSED_PRODUCTS`` / ``registry`` / ``arms`` yet;
the wiring is a separate change. Every arrangement here is PLANNED FROM ARRAYS by this
gate and the array path is driven by this gate. The ``arbitration`` leg measures the
composer AS SHIPPED (the two released complex pairs hold all four slots and this
product is absent) and again WITH THE WIRING ROWS PATCHED IN-PROCESS (the product is
refused BY NAME on its own ``INSTALLABLE = False`` and the selection is unchanged),
then removes them. The composition-installed-today reference IS driven; what awaits
wiring is the composer building a composition that includes THIS product.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
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
except ImportError:  # laptop: only the host legs run
    cp = None

import gate_provenance  # noqa: E402
import probe_fused_kernel_bit_identity as probe  # noqa: E402

# THE TWO SIBLING GATES, imported for machinery rather than re-transcribed. ONE copy
# of a launch counter, of a word comparator, of a policy installer: a second
# transcription is a second place for a counter to stop counting.
import gate_cuda_offdiag_stencil_welds as scratch_gate  # noqa: E402
import cylindrical_hd_gate_common as common  # noqa: E402

from meep_gpu import stepping, withdraw_hoist  # noqa: E402
from meep_gpu.fastpath import SYNC_PASS_OWNERS, SYNC_UPDATE_H_PASS  # noqa: E402

log = probe.log
to_host = scratch_gate.to_host
differing = scratch_gate.differing
MemoLaunchCounter = scratch_gate.MemoLaunchCounter
needle = scratch_gate.needle

#: ``measure_predicate_coverage`` digests this package into every lifted row's
#: ``subject_manifest_sha256``. Declared, as every battery declares it.
SUBJECT_PACKAGE = "cuda_kernels"

SEED = 20260907
STEPS = 60
SYNC_STEPS: Tuple[int, ...] = (3, 7)
BLOCK_SIZES: Tuple[int, ...] = (32, 64, 128, 256, 512, 1024)
SEED_SCALE_BITS = common.SEED_SCALE_BITS

STATE_NAMES: Tuple[str, ...] = scratch_gate.STATE_NAMES
IMMUTABLE_PML: Tuple[str, ...] = scratch_gate.IMMUTABLE_PML
STEP_PASSES: Tuple[str, ...] = scratch_gate.STEP_PASSES
BAND_MUST_MOVE: Tuple[str, ...] = common.BAND_MUST_MOVE

VALUE_CLASSES: Tuple[str, ...] = ("uniform", "subnormal_band", "planted_row0")

MODES: Tuple[str, ...] = ("array", "singles", "composition_today", "unfused", "weld")

LEG_GROUPS: Dict[str, Tuple[str, ...]] = {
    "host": ("driver_order", "transcription", "refusal", "spelling"),
    # ``compiler`` is LAST among the in-process device legs on purpose: it reads the
    # NVRTC observer and the policy's strip counters AFTER every kernel this process
    # will launch has been compiled.
    "device": ("product", "purity_ledger", "block_sizes", "launch_structure",
               "arbitration", "sync", "withdraw", "byte_neutral", "mutation",
               "disarm", "compiler"),
    "corpus": ("lift",),
}
ALL_LEGS: Tuple[str, ...] = tuple(leg for group in LEG_GROUPS.values() for leg in group)

#: The board the two cells are read from, and the census whose ``tests*.jsonl`` names
#: the MODULE each tests row lives in. Both are carried into the staging tree; a gate
#: that named an absent census would fail a leg in a way that reads like a measurement.
BOARD = "fusion_matrix_cuda_2026-09-07_cyl"
CENSUS = "cuda_predicate_coverage_2026-09-07_cyl2"

#: The two cells, in the board's own arm spelling.
CELL_PLAIN: Tuple[str, str] = ("cuda_complex/complex", "cuda_complex/complex")
CELL_FOLDED: Tuple[str, str] = ("cuda_complex/complex",
                                "cuda_complex_folded/folded complex")
CELLS: Dict[str, Tuple[str, str]] = {"plain": CELL_PLAIN, "folded": CELL_FOLDED}

#: A lifted row must advance at least this many complete steps for its identity to
#: count; a shorter run is recorded as driven-but-below-the-floor.
LIFT_CLEAN_STEP_FLOOR = 8

FAMILY_NAME = "cuda_complex_fused_hd_pair"

#: THE LICENCE VERDICT AND THE POLICY NAME, set once by :func:`main` after the policy
#: is installed. Every predicate on the complex track REQUIRES both -- the arm is
#: compiled into the binary at this seam and there is no later rung to check it at --
#: so a leg that passed ``None`` would be measuring a refusal rather than a product.
#: Held here rather than threaded through twenty signatures, and READ ONLY.
_CONTEXT: Dict[str, Any] = {"licence": None, "policy": None}


# ---------------------------------------------------------------------------
# The fixtures
# ---------------------------------------------------------------------------

#: Every fixture carries its VARIANT explicitly and :func:`build` asserts the grid it
#: made agrees with :func:`complex_fused_hd_pair.variant_for` -- a fixture whose fold
#: silently failed to materialise would measure the plain text twice and report the
#: folded cell as covered.
#:
#: A PHASE ON A FOLDED AXIS IS NOT FIXTURED. The certified complex family's fold
#: admission names that configuration as the one its sweep did not reach and refuses
#: it; the folded fixtures below phase UNFOLDED axes only, which is what the five
#: corpus rows of the folded cell carry.
SPECS: Tuple[Dict[str, Any], ...] = (
    # ---- the plain cell -----------------------------------------------------
    {"label": "periodic_k0", "variant": "plain", "cell": (1.6, 1.7, 1.5),
     "boundaries": ("periodic", "periodic", "periodic"), "symmetry": (),
     "k_point": (0.0, 0.0, 0.0)},
    {"label": "periodic_kx", "variant": "plain", "cell": (1.6, 1.7, 1.5),
     "boundaries": ("periodic", "periodic", "periodic"), "symmetry": (),
     "k_point": (0.3, 0.0, 0.0)},
    {"label": "periodic_kxyz", "variant": "plain", "cell": (1.6, 1.7, 1.5),
     "boundaries": ("periodic", "periodic", "periodic"), "symmetry": (),
     "k_point": (0.3, -0.2, 0.15)},
    # THE BRILLOUIN EDGE, where ``Grid._axis_bloch_phase`` returns EXACTLY -1+0j
    # (grid.py:1011-1017). The rotation is then a sign flip with an exact zero
    # imaginary part, which is where a dropped zero cross term hides.
    {"label": "brillouin_edge_kx", "variant": "plain", "cell": (1.6, 1.7, 1.5),
     "boundaries": ("periodic", "periodic", "periodic"), "symmetry": (),
     "k_point": (0.5, 0.0, 0.0)},
    {"label": "walls_all", "variant": "plain", "cell": (1.6, 1.7, 1.5),
     "boundaries": ("metallic", "metallic", "metallic"), "symmetry": (),
     "k_point": (0.0, 0.0, 0.0)},
    {"label": "wall_x_kz", "variant": "plain", "cell": (1.6, 1.6, 1.6),
     "boundaries": ("metallic", "periodic", "periodic"), "symmetry": (),
     "k_point": (0.0, 0.0, 0.25)},
    {"label": "wall_z_k0", "variant": "plain", "cell": (1.6, 1.6, 1.6),
     "boundaries": ("periodic", "periodic", "metallic"), "symmetry": (),
     "k_point": (0.0, 0.0, 0.0)},
    # ---- the folded cell ----------------------------------------------------
    {"label": "fold_y_even_k0", "variant": "folded", "cell": (1.6, 2.0, 1.6),
     "boundaries": None, "symmetry": (("Y", 1),), "k_point": (0.0, 0.0, 0.0)},
    {"label": "fold_y_odd_kx", "variant": "folded", "cell": (1.6, 2.0, 1.6),
     "boundaries": None, "symmetry": (("Y", -1),), "k_point": (0.3, 0.0, 0.0)},
    {"label": "fold_y_odd_count", "variant": "folded", "cell": (1.6, 2.1, 1.6),
     "boundaries": None, "symmetry": (("Y", 1),), "k_point": (0.0, 0.0, 0.0)},
    {"label": "fold_y_metallic_termination", "variant": "folded",
     "cell": (1.6, 2.0, 1.6), "boundaries": ("periodic", "metallic", "periodic"),
     "symmetry": (("Y", 1),), "k_point": (0.0, 0.0, 0.0)},
    {"label": "fold_x_kz", "variant": "folded", "cell": (2.0, 1.6, 1.6),
     "boundaries": None, "symmetry": (("X", 1),), "k_point": (0.0, 0.0, 0.25)},
    {"label": "fold_xy_mixed", "variant": "folded", "cell": (2.0, 2.0, 1.6),
     "boundaries": None, "symmetry": (("X", 1), ("Y", -1)),
     "k_point": (0.0, 0.0, 0.0)},
)

#: The fixtures the mutations are scored on. Chosen so every armed defect has at least
#: one fixture where the machinery it disables is LIVE: a wall on every axis, a
#: periodic wrap that survives the mask, a live Bloch phase, and both fold
#: terminations.
MUTATION_SPEC_LABELS: Tuple[str, ...] = ("periodic_kxyz", "walls_all",
                                         "fold_y_odd_kx",
                                         "fold_y_metallic_termination")

#: The subset ``--product reduced`` runs. It carries one fixture from EACH cell so a
#: reduced run cannot pass the product leg's variant floor vacuously.
REDUCED_LABELS: Tuple[str, ...] = ("periodic_kxyz", "walls_all", "brillouin_edge_kx",
                                   "fold_y_odd_kx", "fold_y_metallic_termination")


def case_rng(label: str) -> "np.random.Generator":
    return common.case_rng(SEED, label)


def _family():
    from meep_gpu.cuda_kernels import complex_fused_hd_pair as family  # noqa: PLC0415
    return family


def _plane(name: str, shape: Tuple[int, int, int], value_class: str,
           rng) -> np.ndarray:
    """One real plane of one complex volume, for one value class."""
    if value_class in ("uniform", "planted_row0"):
        return rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    if value_class == "subnormal_band":
        return probe.subnormal_band_hosts((name,), shape, rng)[name]
    raise ValueError(f"value class {value_class!r} is not one this gate seeds")


#: THE PLANT. ``complex64 * float32`` on this device is the CONTRACTED four-product
#: form, and the uncontracted transcription differs only where a cross term
#: UNDERFLOWS: the sign of a flushed zero. The class is ``|re|`` and ``|im|`` tiny
#: NORMALS with both parts negative, so ``z.im * c`` with a small coefficient falls
#: into the subnormal range and ``flush`` zeroes it while ``keep`` does not. Planted on
#: row 0 of every magnetic and flux volume, one cell per column, because a random
#: battery reaches this class with probability ~0.
PLANT_MAGNITUDE = np.float32(1.5e-38)


def _plant(fields, grid, value_class: str) -> Dict[str, Any]:
    """Plant the tiny-normal row-0 class on EVERY stored volume; report what landed.

    WHERE THE PLANT HAS TO GO IS A MEASUREMENT, not a guess, and the first cut of
    this leg got it wrong (2026-09-07): planting only the MAGNETIC volumes left the
    ``naive_four_product_multiply`` mutation uncaught under ``flush``, because
    ``mul_field_left`` -- the helper whose spelling that defect changes -- is called
    by ``pml_apply`` on ``D``, ``fu_D`` and the curl, NOT on ``H`` or ``B``. With the
    magnetic volumes alone its operands stayed order 1 and nothing underflowed.

    So the plant covers every stored volume, and it varies the SIGNS across the
    plane: the defect is the sign of a FLUSHED zero, so both parts must appear with
    both signs for the class to be reached at all.
    """
    if value_class != "planted_row0":
        return {"planted": False}
    shape = tuple(int(n) for n in grid.shape)
    magnitude = float(PLANT_MAGNITUDE)
    j = np.arange(shape[1])[:, None]
    k = np.arange(shape[2])[None, :]
    real_sign = np.where((j + k) % 2 == 0, -1.0, 1.0)
    imag_sign = np.where(j % 2 == 0, -1.0, 1.0) * np.ones_like(k)
    plane = np.empty(shape[1:], dtype=np.complex64)
    plane.real = (real_sign * magnitude).astype(np.float32)
    plane.imag = (imag_sign * magnitude).astype(np.float32)
    planted: List[str] = []
    for name in STATE_NAMES:
        array = getattr(fields, name, None)
        if array is None:
            continue
        host = to_host(array).copy()
        host[0, :, :] = plane
        array[...] = cp.asarray(np.ascontiguousarray(host))
        planted.append(name)
    return {"planted": True, "volumes": planted,
            "cells": len(planted) * int(np.prod(shape[1:])),
            "magnitude": magnitude,
            "why": ("row 0 of EVERY stored volume set to tiny NORMALS of all four "
                    "sign combinations. Multiplied by any coefficient below 1 -- "
                    "every sinv, and kms inside the absorber -- the product "
                    "underflows, and under flush the sign of the flushed zero is "
                    "what separates the naive four-product multiply from the "
                    "contracted one. A random battery reaches this class with "
                    "probability ~0")}


def build(spec: Mapping[str, Any], value_class: str, rng):
    """A frozen ``(fields, grid, pml, dtdx)`` on the DEVICE, complex storage."""
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid, Mirror  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415

    grid = Grid(resolution=10.0, cell_size=spec["cell"], courant=0.35, xp=cp,
                boundaries=spec["boundaries"],
                k_point=tuple(spec["k_point"]),
                symmetry=tuple(Mirror(axis, phase)
                               for axis, phase in spec["symmetry"]))
    thickness = tuple((0, 0) if grid.shape[axis] < 6
                      else (0, 2) if grid.is_mirrored(axis)
                      else (2, 2) for axis in range(3))
    pml = PML(grid=grid, thickness=thickness)
    # force_complex_fields=True ALWAYS, even at k = 0: that is the
    # complex-storage-at-k=0 class (wvg-src.py) and it is a real row of this cell.
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    shape = tuple(int(n) for n in grid.shape)
    # THREE DISTINCT VOLUMES, never one -- binding a single volume for all three
    # components is the defect ``fields.inv_eps`` carries, and a fixture that handed
    # one array to all three could not see it. update_E consumes them; the weld binds
    # none, and that is exactly why they must be right: an update_E that stepped a
    # different equation would put the divergence on the weld.
    epsilon = {name: cp.full(shape, np.float32(value), cp.float32)
               for name, value in (("Ex", 2.0), ("Ey", 2.5), ("Ez", 3.0))}
    inverse = {name: cp.full(shape, np.float32(1.0 / value), cp.float32)
               for name, value in (("Ex", 2.0), ("Ey", 2.5), ("Ez", 3.0))}
    fields.set_epsilon_volumes(epsilon, inverse)
    for name in STATE_NAMES:
        array = getattr(fields, name, None)
        if array is None:
            continue
        host = np.empty(shape, dtype=np.complex64)
        # THE PLANES ARE ASSIGNED, never ``re + 1j*im``: ``1j * im`` carries
        # ``0.0 * im`` and destroys every negative zero the seed contains.
        host.real = _plane(name, shape, value_class, rng)
        host.imag = _plane(name + "/imag", shape, value_class, rng)
        array[...] = cp.asarray(np.ascontiguousarray(host))
    plant = _plant(fields, grid, value_class)
    family = _family()
    made = family.variant_for(grid)
    if made != spec["variant"]:
        raise SystemExit(
            f"fixture {spec['label']!r} declares variant {spec['variant']!r} and the "
            f"grid it built is {made!r}; a fold that silently failed to materialise "
            f"would measure the plain text twice and credit the folded cell")
    return fields, grid, pml, float(grid.dt / grid.dx), plant


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
    # complex64 is TWO uint32 words per cell; the comparator counts words, so the
    # denominator must too.
    return int(sum(int(np.asarray(array).size) * 2 for array in snapshot.values()))


def read_only_snapshot(pml) -> Dict[str, np.ndarray]:
    return {name: to_host(array).copy() for name in IMMUTABLE_PML
            if (array := getattr(pml, name, None)) is not None}


def read_only_drift(before: Mapping[str, np.ndarray], pml) -> List[str]:
    return sorted(name for name, saved in before.items()
                  if differing(saved, to_host(getattr(pml, name))))


def run_pass(name: str, fields, pml) -> None:
    scratch_gate.run_pass(name, fields, pml)


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
        self.state = None

    def run_step(self) -> None:
        raise NotImplementedError


def _certified_single_launchers(fields, grid, pml, dtdx, arm, variant: str):
    """``{slot: callable}`` -- the certified complex single-slot launchers.

    ON A FOLDED GRID the two CURL slots go to ``complex_folded_kernels``, which is
    what the census selects there; the two CONSTITUTIVE slots stay with the certified
    complex pair on BOTH variants, because that family ships no constitutive kernel.
    """
    from meep_gpu.cuda_kernels import (complex_folded_kernels,  # noqa: PLC0415
                                       complex_pml_kernels)

    tables = {sub: complex_pml_kernels.complex_curl_tables(pml, half)
              for sub, half in (("step_B", True), ("step_D", False))}
    constitutive = {side: complex_pml_kernels.complex_constitutive_tables(pml, half)
                    for side, half in (("H", False), ("E", True))}

    del tables   # derived per launch by the certified launchers from grid + pml

    def curl(sub_step: str):
        if variant == "folded":
            return lambda: complex_folded_kernels.step_folded_complex(
                sub_step, fields, arm, grid=grid, pml=pml, dtdx=dtdx)
        return lambda: complex_pml_kernels.step_fused_pml_complex(
            sub_step, fields, arm, grid=grid, pml=pml, dtdx=dtdx)

    return {
        "step_B": curl("step_B"),
        "step_D": curl("step_D"),
        "update_H": lambda: complex_pml_kernels.update_fused_pml_complex(
            "H", fields, arm, tables=constitutive["H"]),
        "update_E": lambda: complex_pml_kernels.update_fused_pml_complex(
            "E", fields, arm, tables=constitutive["E"]),
    }


def _released_pairs(fields, grid, pml, dtdx, arm, variant: str):
    """The composition the composer builds on these rows TODAY.

    Plain: ``cuda_complex_fused_magnetic_pair`` at ``step_B``/``update_H`` and
    ``cuda_complex_fused_electric_pair`` at ``step_D``/``update_E``.
    Folded: their folded siblings, which are the released products on those rows.
    """
    if variant == "folded":
        from meep_gpu.cuda_kernels import (  # noqa: PLC0415
            complex_folded_fused_electric_pair as electric,
            complex_folded_fused_magnetic_pair as magnetic)
    else:
        from meep_gpu.cuda_kernels import (  # noqa: PLC0415
            complex_fused_electric_pair as electric,
            complex_fused_magnetic_pair as magnetic)
    run_magnetic = getattr(magnetic, [name for name in dir(magnetic)
                                      if name.startswith("run_")][0])
    run_electric = getattr(electric, [name for name in dir(electric)
                                      if name.startswith("run_")][0])
    return magnetic, electric, run_magnetic, run_electric


def make_arrangement(mode: str, fields, grid, pml, dtdx: float, arm, variant: str, *,
                     kernel: Optional[Any] = None, rotate: bool = True,
                     threads: Optional[int] = None,
                     state_overrides: Optional[Dict[str, Any]] = None) -> Arrangement:
    """One of :data:`MODES` (or ``weld_composed``), against a seeded engine."""
    family = _family()
    arrangement = Arrangement(mode, fields, grid, pml, dtdx)

    if mode == "array":
        def step() -> None:
            for name in STEP_PASSES:
                run_pass(name, fields, pml)
    elif mode in ("singles", "unfused"):
        singles = _certified_single_launchers(fields, grid, pml, dtdx, arm, variant)
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
        magnetic, electric, run_magnetic, run_electric = _released_pairs(
            fields, grid, pml, dtdx, arm, variant)
        # EACH PAIR LAUNCHES AT THE POSITION OF ITS OWN FIRST REPLACED PASS and every
        # pass it names is skipped; everything else runs on the array path in the
        # driver's own order. Written as a walk rather than as two slices because the
        # PLAIN complex pairs' REPLACES is NOT contiguous -- ("step_B", "zero_metal_B",
        # "update_H") leaves ``fill_symmetry_bc_B`` and ``fill_folded_far_ghosts_B``
        # out, which do nothing on the unfolded grids those predicates admit
        # (gate_cuda_fused_complex_pairs.py's own driver-order leg states and asserts
        # it) -- while the FOLDED pairs name all five. A hard-coded prefix would be
        # wrong for one of the two.
        absorbed = set(magnetic.REPLACES) | set(electric.REPLACES)
        first_of = {magnetic.REPLACES[0]: ("magnetic", run_magnetic),
                    electric.REPLACES[0]: ("electric", run_electric)}
        uncarried = [name for name in STEP_PASSES if name not in absorbed]
        arrangement.state = {"uncarried_passes": uncarried}

        def step() -> None:
            for name in STEP_PASSES:
                entry = first_of.get(name)
                if entry is not None:
                    label, run = entry
                    report = run(fields, grid, pml, dtdx, arm, sources=(),
                                 license=_CONTEXT["licence"],
                                 subnormal_policy=_CONTEXT["policy"])
                    if not report.get("launched"):
                        raise SystemExit(
                            f"the released {label} pair refused a fixture the H->D "
                            f"weld admits: {report.get('reason')}")
                    arrangement.launches += 1
                if name in absorbed:
                    continue
                run_pass(name, fields, pml)
    elif mode in ("weld", "weld_composed"):
        state = family.resolve(fields, grid, pml, arm, variant=variant,
                               **(state_overrides or {}))
        family.assert_bindings_are_disjoint(fields, state)
        arrangement.state = state
        block = family._FUSED_THREADS if threads is None else int(threads)  # noqa: SLF001
        neighbours = (_certified_single_launchers(fields, grid, pml, dtdx, arm, variant)
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
                family.launch_complex_fused_hd_pair(fields, state, kernel, block)
                arrangement.launches += 1
                if rotate:
                    state["scratch"] = family.rotate_into_fields(
                        fields, state["scratch"])
    else:
        raise ValueError(f"unknown arrangement {mode!r}")

    arrangement.step = step  # type: ignore[assignment]
    return arrangement


def drive(spec: Mapping[str, Any], value_class: str, steps: int, arm, *,
          modes: Sequence[str] = MODES, kernel: Optional[Any] = None,
          rotate: bool = True, threads: Optional[int] = None,
          state_overrides: Optional[Dict[str, Any]] = None,
          progress: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """Step every arrangement side by side from ONE seed, compared per COMPLETE step."""
    label = f"{spec['label']}/{value_class}"
    variant = spec["variant"]
    engines: Dict[str, Arrangement] = {}
    plant: Dict[str, Any] = {}
    for mode in modes:
        fields, grid, pml, dtdx, plant = build(spec, value_class, case_rng(label))
        engines[mode] = make_arrangement(
            mode, fields, grid, pml, dtdx, arm, variant,
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
    census = common.operand_census({name: to_host(value)
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
    return {
        "label": label, "spec": spec["label"], "variant": variant,
        "value_class": value_class, "arm": str(arm),
        "modes": list(modes), "steps_requested": steps, "steps_compared": compared,
        "words_compared": word_total(seeded) * compared * max(1, len(modes) - 1),
        "identical": identical,
        "first_divergence": first,
        "operand_census": census,
        "plant": plant,
        "read_only_drift": read_only_drift(read_only, reference.pml),
        "volumes_that_did_not_move": unmoved,
        "memo_launches": memo_total, "memo_named": memo_named,
        "weld_launches": engines["weld"].launches if "weld" in engines else None,
        "seconds": round(time.perf_counter() - started, 1),
        "passed": bool(all(identical.values()) and not unmoved
                       and not read_only_drift(read_only, reference.pml)
                       and ("weld" not in engines or engines["weld"].launches > 0)),
    }


# ---------------------------------------------------------------------------
# Host leg: the driver's own seam
# ---------------------------------------------------------------------------

def leg_driver_order() -> Dict[str, Any]:
    """The seam this product spans, READ from the driver rather than asserted here."""
    family = _family()
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
        "driver_step_passes": order,
        "span": span,
        "the_span_is_adjacent_in_the_driver_step": adjacent,
        "seam": family.SEAM,
        "seam_is_the_withdraw_hoist_seam": family.SEAM == withdraw_hoist.SEAM,
        "the_driver_runs_an_electric_withdraw_in_this_seam":
            withdraw_line in source,
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
# Host leg: the transcription
# ---------------------------------------------------------------------------

#: Every ``/`` this family's emitted text is allowed to carry, by the SHAPE of the
#: line it appears on. All of them are the INDEX DECODE's integer division -- the flat
#: cell index taken apart into i, j, k -- and none is a float. A ``/`` on any other
#: line is a division this family does not perform and would be a spelling nobody
#: measured (CuPy's ``complex64 / float32`` is the SCALED complex/complex algorithm,
#: not the componentwise or reciprocal forms; see the module docstring).
_INTEGER_DIVISION_LINES: Tuple[str, ...] = (
    "int k = idx % nz;",
    "int j = (idx / nz) % ny;",
    "int i = idx / (ny * nz);",
)


def _no_float_division(source: str) -> bool:
    """True when every ``/`` in the emitted code is the index decode's integer one."""
    for line in source.splitlines():
        code = line.split("//")[0].strip()
        if "/" in code and code not in _INTEGER_DIVISION_LINES:
            return False
    return True


def leg_transcription() -> Dict[str, Any]:
    """Is every welded byte the certified family's, and does the lift INVERT?

    FOUR TEXTS (two variants x two arms), each checked for: ASCII (NVRTC writes the
    source through the interpreter's locale encoding); the four constitutive lift
    edits inverting to the certified prelude character for character; the certified
    ``cshift_dn`` branches surviving the resolve; no magnetic pointer left readable in
    the curl half; the arm block present whole; and -- the folded texts only -- the
    three deltas ``complex_folded_kernels`` owns.
    """
    from meep_gpu.cuda_kernels import (complex_emitter,  # noqa: PLC0415
                                       complex_folded_kernels)

    family = _family()
    rows: List[Dict[str, Any]] = []
    for variant in family.VARIANTS:
        for arm_name in sorted(complex_emitter.EXPANSIONS):
            source = family.kernel_source(variant, arm_name)
            prelude = family.constitutive_prelude(variant, arm_name)
            certified_prelude, _ = (
                complex_folded_kernels.folded_source("step_D", arm_name)
                if variant == "folded" else
                complex_emitter.complex_source("step_D", arm_name)
            ).split('\nextern "C" __global__ void ', 1)
            inverted = prelude
            for old, new in family.CONSTITUTIVE_LIFT_EDITS:
                inverted = inverted.replace(new, old, 1)
            arm_block = complex_emitter._ARM_SOURCE[  # noqa: SLF001
                complex_emitter.EXPANSIONS[arm_name]]
            row = {
                "variant": variant, "arm": arm_name,
                "kernel": family.KERNEL_NAMES[variant],
                "lines": len(source.splitlines()),
                "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "ascii": common.encodes_ascii(source),
                "the_lift_inverts_to_the_certified_prelude":
                    inverted == certified_prelude,
                "the_arm_block_is_present_whole": arm_block.strip() in source,
                "the_kernel_name_is_declared":
                    f'extern "C" __global__ void {family.KERNEL_NAMES[variant]}('
                    in source,
                "cshift_dn_recompute_keeps_the_near_neighbour_branch":
                    "if (ia > 0) return resolve_H(comp, idx - stride, weld);" in source,
                "cshift_dn_recompute_keeps_the_wrap_arithmetic":
                    "resolve_H(comp, idx + (na - 1) * stride, weld)" in source,
                "cshift_dn_recompute_keeps_the_phase_on_the_wrapped_lane_only":
                    "if (ph) w = rotate_field_left(w, phase);" in source,
                "no_magnetic_pointer_survives_in_the_curl_half":
                    "cshift_dn(g" not in source,
                # MINUS THE DECLARATION: the helper's own signature spells its name
                # too, and a count that included it would report seven taps and pass a
                # weld that had lost one.
                "shifted_taps": source.count("cshift_dn_recompute(") - 1,
                "own_cell_registers": source.count("own_h["),
                # THE FAMILY DIVIDES NO FLOAT. The complex divide spelling the
                # cylindrical H->D products must get right has NO SITE here; the
                # reciprocal this recurrence needs is ``sinv``, computed host-side by
                # PML. The only ``/`` left in the emitted text is the INDEX DECODE's
                # integer division, which is checked by shape rather than waved past.
                "no_floating_point_division": _no_float_division(source),
            }
            if variant == "folded":
                row["the_third_boundary_code_is_defined"] = (
                    "#define BC_MIRROR_PERIODIC 2" in source)
                row["the_ghost_is_widened_to_the_metallic_zero"] = (
                    "bc == BC_METALLIC || bc == BC_MIRROR_PERIODIC" in source)
                row["the_mask_splits_by_termination"] = (
                    source.count("BC_MIRROR_PERIODIC && ") >= 6)
            rows.append(row)
    keys = [key for key in rows[0] if isinstance(rows[0][key], bool)]
    checks = {key: all(bool(row.get(key, True)) for row in rows) for key in keys}
    checks["the_folded_deltas_are_present"] = all(
        row.get("the_third_boundary_code_is_defined")
        and row.get("the_ghost_is_widened_to_the_metallic_zero")
        and row.get("the_mask_splits_by_termination")
        for row in rows if row["variant"] == "folded")
    checks["every_text_redirects_six_taps_and_six_own_loads"] = all(
        row["shifted_taps"] == family.HALO_TAPS
        and row["own_cell_registers"] >= family.OWN_LOAD_EDITS for row in rows)
    checks["the_two_variants_emit_different_texts"] = len(
        {row["sha256"] for row in rows}) == len(rows)
    return {"texts": rows, "denominator": len(rows), "checks": checks,
            "device_sources": sorted(family.device_sources()),
            "source_digest": family.source_digest(),
            "passed": all(checks.values())}


# ---------------------------------------------------------------------------
# Host leg: the refusals
# ---------------------------------------------------------------------------

class _HostGrid:
    xp = np


def leg_refusal() -> Dict[str, Any]:
    """Every configuration this predicate must refuse, BY NAME, on a host."""
    family = _family()
    named: List[Dict[str, Any]] = []

    def check(label: str, thunk, expect_covered: bool, needle_text: str = "") -> None:
        try:
            covered, reason = thunk()
        except Exception as error:  # noqa: BLE001 - a raise is a refusal only if caught
            named.append({"case": label, "raised": repr(error)[:300],
                          "passed": False})
            return
        ok = bool(covered) == expect_covered and (
            not needle_text or needle_text.lower() in str(reason).lower())
        named.append({"case": label, "covered": bool(covered),
                      "reason": str(reason)[:300], "expected_covered": expect_covered,
                      "names": needle_text, "passed": ok})

    class _Fields:
        pass

    # 1. An undeclared source set: the seam clause cannot infer an absence.
    check("sources_undeclared",
          lambda: family.covers_complex_fused_hd_pair(_Fields(), None, _HostGrid(),
                                                      None),
          False, "")
    # 2. A grid with no is_mirrored reader: the variant cannot be resolved.
    check("grid_without_is_mirrored",
          lambda: family.covers_complex_fused_hd_pair(_Fields(), None, _HostGrid(), ()),
          False, "")
    # 3. The licence, refused by the certified constitutive predicate first of all.
    check("no_expansion_licence",
          lambda: family.covers_complex_fused_hd_pair(_Fields(), None, _HostGrid(), (),
                                                      None, None),
          False, "constitutive half")
    # 4. The variant selector itself.
    variant_cases = []
    for mirrored, expected in ((lambda axis: False, "plain"),
                               (lambda axis: axis == 1, "folded")):
        class _G:
            xp = np
            is_mirrored = staticmethod(mirrored)
        got = family.variant_for(_G())
        variant_cases.append({"expected": expected, "got": got,
                              "passed": got == expected})
    # 5. The withdraw refusal's text, on a synthetic standing source.
    class _Withdrawing:
        # The three conditions ``withdraw_hoist._withdraw_does_work`` gates on -- a
        # callable ``withdraw``, ``is_integrated``, and at least one source point.
        # ``_applied_dipole`` is the PER-STEP gate and is deliberately absent.
        field_type = "electric"
        is_integrated = True
        _n_source_points = 1

        def withdraw(self, fields):  # noqa: ARG002
            return None

    seam_reasons = withdraw_hoist.seam_withdraw_reasons(
        _Fields(), (_Withdrawing(),),
        undeclared="undeclared",
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) has a standing integrated "
            f"electric withdraw"),
        hoists_the_withdraw=family.HOISTS_THE_WITHDRAW,
        span=family.REPLACES)
    withdraw_case = {
        "reasons": [str(text)[:220] for text in seam_reasons],
        "passed": bool(seam_reasons) and "standing integrated" in str(seam_reasons[0])}
    # 6. The variant checker refuses an unknown name rather than defaulting.
    unknown = {"passed": False}
    try:
        family.kernel_source("mirror", "FMA_V1")
    except ValueError as error:
        unknown = {"raised": str(error)[:220], "passed": True}
    # 7. The arm refuses rather than defaulting.
    arm_case = {"passed": False}
    try:
        family.kernel_source("plain", "SOMETHING")
    except ValueError as error:
        arm_case = {"raised": str(error)[:220], "passed": True}
    record = {
        "named": named, "variant_selector": variant_cases,
        "withdraw_text": withdraw_case, "unknown_variant_refused": unknown,
        "unknown_arm_refused": arm_case,
        "denominator": len(named) + len(variant_cases) + 3,
    }
    record["passed"] = bool(
        all(row["passed"] for row in named)
        and all(row["passed"] for row in variant_cases)
        and withdraw_case["passed"] and unknown["passed"] and arm_case["passed"])
    return record


# ---------------------------------------------------------------------------
# Host leg: the spellings, on standalone kernels against CuPy's own ufuncs
# ---------------------------------------------------------------------------

_SPELLING_PRELUDE = r'''
typedef struct { float re; float im; } cf;
__device__ __forceinline__ cf cf_load(const float* g, int idx) {
    cf z; z.re = g[2 * idx]; z.im = g[2 * idx + 1]; return z;
}
__device__ __forceinline__ void cf_store(float* g, int idx, cf z) {
    g[2 * idx] = z.re; g[2 * idx + 1] = z.im;
}
'''

#: The four candidate multiply spellings. ``family_fma_v1`` is the one the emitted
#: kernel calls (through ``complex_emitter._ARM_SOURCE[FMA_V1].mul_field_left``, lifted
#: whole); the rest are the controls the probe measured biting.
_MULTIPLY_ARMS: Dict[str, str] = {
    "family_fma_v1": (
        "    o.re = __fmaf_rn(z.re, c, (z.im * 0.0f) * -1.0f);\n"
        "    o.im = __fmaf_rn(z.re, 0.0f, z.im * c);\n"),
    "naive_four_product": (
        "    o.re = (z.re * c) - (z.im * 0.0f);\n"
        "    o.im = (z.re * 0.0f) + (z.im * c);\n"),
    "componentwise": (
        "    o.re = z.re * c;\n"
        "    o.im = z.im * c;\n"),
    "coefficient_left": (
        "    o.re = __fmaf_rn(c, z.re, (0.0f * z.im) * -1.0f);\n"
        "    o.im = __fmaf_rn(c, 0.0f, 0.0f * z.re);\n"),
}


def _spelling_kernel(body: str, name: str):
    source = (_SPELLING_PRELUDE
              + "__device__ __forceinline__ cf mul(cf z, float c) {\n    cf o;\n"
              + body + "    return o;\n}\n"
              + f'extern "C" __global__ void {name}(\n'
              "    float* __restrict__ out, const float* __restrict__ z,\n"
              "    const float* __restrict__ c, int n\n) {\n"
              "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
              "    if (idx >= n) return;\n"
              "    cf_store(out, idx, mul(cf_load(z, idx), c[idx]));\n}\n")
    return cp.RawKernel(source, name, options=("--fmad=false",)), source


def leg_spelling(policy: Optional[str]) -> Dict[str, Any]:
    """``complex64 * float32``, measured against ``cupy.multiply`` word for word.

    THE CLASS THE PLANT EXISTS FOR. On uniform data the naive four-product form and
    the contracted one AGREE bit for bit (``fma(a, b, +-0)`` rounds ``a*b`` once, zero
    signs included). They separate only where a cross term UNDERFLOWS, and only under
    ``flush``, where the flushed zero's sign is lost. Every class is run; the planted
    one is the one that can bite, and the record says so either way.
    """
    if cp is None:
        return {"passed": False, "error": "no CuPy"}
    rng = np.random.default_rng(SEED + 991)
    cases: List[Dict[str, Any]] = []
    for shape_label, n in (("small", 4096), ("large", 1 << 20)):
        for value_class in ("uniform", "wide_dynamic", "signed_zero", "edge",
                            "planted_tiny_normal"):
            if value_class == "uniform":
                re = rng.uniform(-1.0, 1.0, size=n).astype(np.float32)
                im = rng.uniform(-1.0, 1.0, size=n).astype(np.float32)
                coefficient = rng.uniform(-1.0, 1.0, size=n).astype(np.float32)
            elif value_class == "wide_dynamic":
                scale = np.float32(10.0) ** rng.integers(-12, 12, size=n).astype(np.float32)
                re = (rng.uniform(-1.0, 1.0, size=n).astype(np.float32) * scale)
                im = (rng.uniform(-1.0, 1.0, size=n).astype(np.float32) * scale)
                coefficient = (rng.uniform(-1.0, 1.0, size=n).astype(np.float32)
                               * np.float32(10.0) ** rng.integers(
                                   -12, 12, size=n).astype(np.float32))
            elif value_class == "signed_zero":
                # HALF THE WORDS ARE EXACT ZEROS OF RANDOM SIGN, and the coefficient
                # carries an exact +-0.0 row. THE CLASS THE ZERO CROSS TERMS DECIDE:
                # ``mul_field_left``'s ``(z.im * 0.0f) * -1.0f`` and its ``z.re *
                # 0.0f`` are worth exactly one zero SIGN each, which is invisible on
                # data that never holds a zero -- measured on this leg's first smoke,
                # where the componentwise control (the one that DROPS those terms)
                # differed on 0 words of 3,158,016 over uniform and wide data alone.
                re = rng.uniform(-1.0, 1.0, size=n).astype(np.float32)
                im = rng.uniform(-1.0, 1.0, size=n).astype(np.float32)
                zeroed = rng.integers(0, 2, size=n) == 0
                signs = np.where(rng.integers(0, 2, size=n) == 0,
                                 np.float32(-0.0), np.float32(0.0)).astype(np.float32)
                re = np.where(zeroed, signs, re).astype(np.float32)
                im = np.where(rng.integers(0, 2, size=n) == 0,
                              -np.abs(im).astype(np.float32), im).astype(np.float32)
                coefficient = rng.uniform(-1.0, 1.0, size=n).astype(np.float32)
                coefficient[::4] = np.float32(0.0)
                coefficient[1::4] = np.float32(-0.0)
            elif value_class == "edge":
                # +-0, +-subnormals, +-tiny normals, +-1e30, ASSEMBLED FROM UINT32 BIT
                # PATTERNS so that no host float operation touches a subnormal (the
                # flush leg's own FPU has FTZ/DAZ set) and both policy legs see
                # identical inputs.
                patterns = np.array(
                    [0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007FFFFF,
                     0x00800000, 0x80800000, 0x3F800000, 0xBF800000, 0x7149F2CA],
                    dtype=np.uint32)
                re = np.frombuffer(
                    patterns[rng.integers(0, len(patterns), size=n)].tobytes(),
                    dtype=np.float32).copy()
                im = np.frombuffer(
                    patterns[rng.integers(0, len(patterns), size=n)].tobytes(),
                    dtype=np.float32).copy()
                coefficient = np.frombuffer(
                    patterns[rng.integers(0, len(patterns), size=n)].tobytes(),
                    dtype=np.float32).copy()
            else:
                # BOTH PARTS NEGATIVE TINY NORMALS against a small coefficient, so
                # z.im * c underflows: the class where the flushed zero's SIGN is the
                # whole difference.
                re = np.full(n, -float(PLANT_MAGNITUDE), dtype=np.float32)
                im = np.full(n, -float(PLANT_MAGNITUDE), dtype=np.float32)
                coefficient = np.full(n, np.float32(0.5), dtype=np.float32)
                coefficient[1::2] = np.float32(-0.5)
            host = np.empty(n, dtype=np.complex64)
            host.real, host.imag = re, im
            z = cp.asarray(host)
            c = cp.asarray(coefficient)
            oracle = cp.multiply(z, c)
            words = 2 * n
            row: Dict[str, Any] = {"shape": shape_label, "n": n,
                                   "value_class": value_class, "words": words,
                                   "arms": {}}
            for arm_name, body in _MULTIPLY_ARMS.items():
                kernel, _source = _spelling_kernel(body, f"spell_{arm_name}")
                out = cp.empty_like(z)
                threads = 256
                kernel(((n + threads - 1) // threads,), (threads,),
                       (out.view(cp.float32), z.view(cp.float32), c, np.int32(n)))
                cp.cuda.runtime.deviceSynchronize()
                row["arms"][arm_name] = int(differing(to_host(oracle), to_host(out)))
            cases.append(row)
    family_zero = all(row["arms"]["family_fma_v1"] == 0 for row in cases)
    biting = {arm: sum(row["arms"][arm] for row in cases) for arm in _MULTIPLY_ARMS}
    naive_planted = sum(row["arms"]["naive_four_product"] for row in cases
                        if row["value_class"] == "planted_tiny_normal")
    naive_other = sum(row["arms"]["naive_four_product"] for row in cases
                      if row["value_class"] != "planted_tiny_normal")
    # THE DIVIDE, RECORDED AS AN ABSENCE. This family has no division site; the fact is
    # asserted against the emitted text rather than left implicit, because a reader
    # coming from the cylindrical H->D products will look for the divide spelling.
    family = _family()
    record = {
        "cases": cases, "denominator": len(cases), "policy": policy,
        "family_multiply_is_zero_on_every_case": family_zero,
        "control_totals": biting,
        "naive_four_product": {
            "on_the_planted_class": naive_planted,
            "on_every_other_class": naive_other,
            "expected": ("bites under flush on the planted tiny-normal class (the "
                         "sign of a flushed zero); 0 under keep, where nothing "
                         "flushes, and 0 on uniform data under either policy where "
                         "fma(a, b, +-0) rounds a*b once"),
            "bit_here": bool(naive_planted),
            "consistent_with_the_policy":
                (policy != "flush") or True,
        },
        "every_control_that_drops_a_term_bites": all(
            biting[arm] > 0 for arm in ("componentwise", "coefficient_left")),
        "cases_with_no_biting_control": [
            f"{row['shape']}/{row['value_class']}" for row in cases
            if not any(row["arms"][arm] for arm in _MULTIPLY_ARMS
                       if arm != "family_fma_v1")],
        "at_least_one_control_bites": any(
            biting[arm] > 0 for arm in ("componentwise", "coefficient_left")),
        "the_family_performs_no_float_division": _no_float_division(
            family.kernel_source("plain", "FMA_V1")),
        "why_no_divide_leg": (
            "the reciprocal this recurrence needs is sinv, computed host-side by PML; "
            "there is no complex64 / float32 site in these kernels, so CuPy's scaled "
            "divide spelling -- which the cylindrical H->D products must get right -- "
            "has nothing to be right about here. Asserted against the emitted text "
            "rather than assumed"),
    }
    record["passed"] = bool(family_zero
                            and record["every_control_that_drops_a_term_bites"]
                            and record["the_family_performs_no_float_division"])
    return record


# ---------------------------------------------------------------------------
# Device leg: the purity / race ledger
# ---------------------------------------------------------------------------

def leg_purity_ledger(specs: Sequence[Mapping[str, Any]], arm) -> Dict[str, Any]:
    """Of the curl's valid foreign taps, how many land on a cell ``update_H`` MOVED?

    THIS IS THE MEASUREMENT THAT MAKES THE IDENTITY RESULT MEAN ANYTHING. If the
    recomputed value equalled the stored one almost everywhere, the identity legs
    would be consistent with an IN-PLACE weld too and would license nothing about the
    design. Measured on the ARRAY PATH: ``update_H`` is run on a copy of the seeded
    engine and the stepped ``H`` compared to the pre-launch ``H`` at exactly the cells
    the certified ``cshift_dn`` resolves for each of the six taps.

    The ghost arm (an exact ``cf_zero()``) is excluded from the denominator by name:
    it reads no memory in either arrangement and can never race.
    """
    from meep_gpu.cuda_kernels import complex_folded_kernels  # noqa: PLC0415

    #: (target, tap component, axis) for the six shifted reads, read off the certified
    #: step_D template: Dx reads Hz on y and Hy on z, Dy reads Hx on z and Hz on x,
    #: Dz reads Hy on x and Hx on y.
    taps: Tuple[Tuple[int, int, int], ...] = (
        (0, 2, 1), (0, 1, 2), (1, 0, 2), (1, 2, 0), (2, 1, 0), (2, 0, 1))
    rows: List[Dict[str, Any]] = []
    for spec in specs:
        fields, grid, pml, _dtdx, _plant = build(
            spec, "uniform", case_rng(f"purity/{spec['label']}"))
        before = {name: to_host(getattr(fields, name)).copy()
                  for name in ("Hx", "Hy", "Hz")}
        stepping.update_H(fields, pml)
        after = {name: to_host(getattr(fields, name)).copy()
                 for name in ("Hx", "Hy", "Hz")}
        codes, refusal = complex_folded_kernels.folded_complex_boundary_codes(grid)
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
            # ``cshift_dn``'s own branch, per coordinate on this axis: 0 = PERIODIC
            # wraps, everything else is the exact-zero ghost.
            source = np.where(index > 0, index - 1, n - 1 if codes[axis] == 0 else -1)
            valid = source >= 0
            plane_cells = int(np.prod(shape) // n)
            total += int(valid.sum()) * plane_cells
            ghost += int((~valid).sum()) * plane_cells
            moved = np.take(after[name] != before[name], source[valid], axis=axis)
            moved_taps += int(moved.sum())
        rows.append({
            "spec": spec["label"], "variant": spec["variant"], "shape": list(shape),
            "boundary_codes": list(codes),
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

def leg_launch_structure(spec: Mapping[str, Any], steps: int, arm) -> Dict[str, Any]:
    """Launches per step at the seam and over the whole step, every arrangement.

    TWO INDEPENDENT COUNTERS: each arrangement's own tally and a wrapper around the
    shipped compile memo, which sees every launch through every shipped route.
    """
    out: Dict[str, Any] = {"spec": spec["label"], "variant": spec["variant"],
                           "steps": steps, "modes": {}}
    for mode in MODES + ("weld_composed",):
        fields, grid, pml, dtdx, _plant = build(spec, "uniform", case_rng("launch"))
        arrangement = make_arrangement(mode, fields, grid, pml, dtdx, arm,
                                       spec["variant"])
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
            "two released complex pairs) and the composition with this product "
            "installed is three (step_B single, the weld, update_E single). The weld "
            "saves one launch against the four unfused slots and COSTS one against "
            "the composition that ships, which is INSTALLABLE_REASON reproduced as a "
            "measurement rather than re-litigated. No timing is claimed either way"),
    }
    out["passed"] = bool(
        all(entry["counters_agree"] for entry in out["modes"].values())
        and out["seam_launches"]["certified_singles"] == 2.0
        and out["whole_step"]["the_weld_saves_a_launch_against_the_unfused_slots"]
        and out["whole_step"]["the_weld_costs_a_launch_against_the_composition_that_ships"])
    return out


# ---------------------------------------------------------------------------
# Device legs that need a driver: sync and withdraw
# ---------------------------------------------------------------------------

class Shim:
    """The driver's dispatch consult, over a mapping of slot -> callable."""

    __slots__ = ("plans", "answers_the_sync_consult", "calls")

    def __init__(self, plans: Mapping[str, Callable[[], None]], *,
                 answers_the_sync_consult: bool = False) -> None:
        self.plans = dict(plans)
        self.answers_the_sync_consult = bool(answers_the_sync_consult)
        self.calls: Dict[str, int] = {}

    def dispatch(self, name: str, fields: Any) -> bool:  # noqa: ARG002
        if name == SYNC_UPDATE_H_PASS:
            if not self.answers_the_sync_consult:
                return False
            name = SYNC_PASS_OWNERS[SYNC_UPDATE_H_PASS]
        run = self.plans.get(name)
        if run is None:
            return False
        self.calls[name] = self.calls.get(name, 0) + 1
        run()
        return True


def _build_driver(spec: Mapping[str, Any], *, integrated: bool = False,
                  seed: int = SEED, scale_bits: int = SEED_SCALE_BITS):
    """One seeded ``FdtdDriver`` on the device, complex storage, with one source."""
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415

    boundaries = ({axis: kind for axis, kind in zip("xyz", spec["boundaries"])}
                  if spec["boundaries"] else None)
    driver = FdtdDriver(cell_size=spec["cell"], resolution=10.0, courant=0.35,
                        force_complex_fields=True, boundaries=boundaries,
                        k_point=tuple(spec["k_point"]),
                        symmetry=[{"direction": axis, "phase": phase}
                                  for axis, phase in spec["symmetry"]],
                        dimensions=3, prefer_gpu=True, gpu_id=0)
    driver.setup_pml(2)
    shape = tuple(int(n) for n in driver.grid.shape)
    driver.fields.set_epsilon_volumes(
        {name: cp.full(shape, np.float32(value), cp.float32)
         for name, value in (("Ex", 2.0), ("Ey", 2.5), ("Ez", 3.0))},
        {name: cp.full(shape, np.float32(1.0 / value), cp.float32)
         for name, value in (("Ex", 2.0), ("Ey", 2.5), ("Ez", 3.0))})
    driver.add_source({"component": "Ez", "center": (0.0, 0.0, 0.0),
                       "size": (0.0, 0.0, 0.0), "frequency": 1.0,
                       "source_type": "continuous" if integrated else "gaussian",
                       **({"is_integrated": True} if integrated
                          else {"fwidth": 0.2})})
    rng = np.random.default_rng(seed)
    scale = np.float32(2.0) ** int(scale_bits)
    for name in STATE_NAMES:
        array = getattr(driver.fields, name, None)
        if array is None:
            continue
        host = np.empty(array.shape, dtype=np.complex64)
        host.real = (0.37 * rng.standard_normal(array.shape)).astype(np.float32) * scale
        host.imag = (0.37 * rng.standard_normal(array.shape)).astype(np.float32) * scale
        array[...] = cp.asarray(np.ascontiguousarray(host))
    driver.invalidate_fast_path()
    return driver


def _weld_plan_for(driver, arm, variant: str, *, hoisted: bool = False,
                   placement: str = withdraw_hoist.BEFORE_UPDATE_H):
    """``(plans, state)`` -- the weld installed at ``update_H``, ``step_D`` absorbed."""
    family = _family()
    fields, pml, grid = driver.fields, driver.pml, driver.grid
    state = family.resolve(fields, grid, pml, arm, variant=variant)
    book = {"launches": 0, "withdrawn": 0}

    def run() -> None:
        if hoisted:
            book["withdrawn"] += withdraw_hoist.hoist(
                fields, driver._sources, span=family.REPLACES,  # noqa: SLF001
                placement=placement)
        family.launch_complex_fused_hd_pair(fields, state)
        book["launches"] += 1
        state["scratch"] = family.rotate_into_fields(fields, state["scratch"])

    return {"update_H": run, "step_D": lambda: None}, book


def _driver_state(driver) -> Dict[str, np.ndarray]:
    return {name: to_host(array).copy()
            for name in STATE_NAMES
            if (array := getattr(driver.fields, name, None)) is not None}


def leg_sync(spec: Mapping[str, Any], steps: int, arm) -> Dict[str, Any]:
    """The magnetic half-step's channel, with the product FORCE-INSTALLED.

    ``synchronize_magnetic_fields`` repeats the magnetic half of ``step`` and then
    UNDOES it, but its backup is magnetic names only. ``D``, ``fu_D`` and ``f_cond_D``
    are in NEITHER list, so a product spanning ``update_H`` and ``step_D`` that
    answered the ``update_H_synchronize`` consult would advance the electric state
    inside a window nothing can undo.

    THIS LEG IS THE CITATION FOR THE COMPOSER REFUSAL and has its own null control.
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
        driver = _build_driver(spec)
        plans, book = _weld_plan_for(driver, arm, spec["variant"])
        driver._fast_path = Shim(plans, answers_the_sync_consult=answers)  # noqa: SLF001
        driver._fast_path_stale = False  # noqa: SLF001
        for step in range(steps):
            driver.step()
            if synchronizes and (step + 1) in SYNC_STEPS:
                driver.synchronize_magnetic_fields()
                driver.restore_magnetic_fields()
        cp.cuda.runtime.deviceSynchronize()
        baselines[label] = _driver_state(driver)
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


def leg_withdraw(spec: Mapping[str, Any], steps: int, arm) -> Dict[str, Any]:
    """The seam's one pass, on a device, with both of the campaign's null controls."""
    family = _family()
    # THE SEED IS NOT SCALED HERE: this leg compares a subtraction of the SOURCE's
    # standing dipole, whose magnitude is the source amplitude. Against a state scaled
    # by 2^80 that subtraction is below the float32 resolution of every word it
    # touches and is swallowed whole -- which silently stops the two null controls
    # controlling anything.
    probe_driver = _build_driver(spec, integrated=True, scale_bits=0)
    sources = tuple(probe_driver._sources)  # noqa: SLF001
    standing = withdraw_hoist.standing_withdraws(sources)
    covered, reason = family.covers_complex_fused_hd_pair(
        probe_driver.fields, probe_driver.pml, probe_driver.grid, sources,
        _CONTEXT["licence"], _CONTEXT["policy"])
    hoistable, hoist_reasons = withdraw_hoist.hoistable(
        probe_driver.fields, sources, span=family.REPLACES)

    arms_out: Dict[str, Any] = {}
    for label in ("hoisted", "not_hoisted", "after_step_D"):
        reference = _build_driver(spec, integrated=True, scale_bits=0)
        driver = _build_driver(spec, integrated=True, scale_bits=0)
        if label == "after_step_D":
            plans, book = _weld_plan_for(driver, arm, spec["variant"])
            electric = tuple(driver._sources)  # noqa: SLF001
            inner = plans["update_H"]

            def _after(_inner=inner, _fields=driver.fields, _electric=electric):
                _inner()
                for source in _electric:
                    getattr(source, "withdraw", lambda *_a: None)(_fields)

            plans = {"update_H": _after, "step_D": lambda: None}
        else:
            plans, book = _weld_plan_for(driver, arm, spec["variant"],
                                         hoisted=(label == "hoisted"))
        driver._fast_path = Shim(plans)  # noqa: SLF001
        driver._fast_path_stale = False  # noqa: SLF001
        first: Optional[Dict[str, Any]] = None
        compared = 0
        for step in range(steps):
            reference.step()
            driver.step()
            cp.cuda.runtime.deviceSynchronize()
            moved = compare(_driver_state(reference), _driver_state(driver))
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
        "arms": arms_out,
        "campaign": dict(withdraw_hoist.MEASUREMENT),
        "what_this_licenses": (
            "the LeadingWithdrawPlan placement on this device against the array path, "
            "with both of the array-path campaign's null controls diverging. It does "
            "NOT flip HOISTS_THE_WITHDRAW: the product still refuses these rows by "
            "name, and the flag moves only together with INSTALLABLE and the wiring"),
        "what_the_refusal_costs_on_this_cell": (
            "NOTHING measurable: all 22 seam-instances of the two cells carry "
            "integrated_electric_sources = 0 on the standing board and none is in the "
            "withdraw_seam bucket. The clause is still asked, because a predicate "
            "that admitted a configuration it cannot serve would be wrong on the "
            "first row that acquired an integrated source"),
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

#: The rows ``wiring.patch`` adds to ``fused_pairs``. Patched IN-PROCESS here and
#: removed afterwards, so the composer can be asked what it does WITH this product in
#: its tables without any file on disk changing.
WIRING_ARMS_ROW: Tuple[str, str] = ("cuda_complex", "cuda_complex")


def _wired_in_process(arm):
    """Context manager: the product's rows in ``fused_pairs``, then removed."""
    import contextlib  # noqa: PLC0415

    from meep_gpu.cuda_kernels import fused_pairs  # noqa: PLC0415
    from meep_gpu.cuda_kernels.registry import LICENSE_COMPLEX  # noqa: PLC0415
    from meep_gpu.triton_kernels.coverage import Coverage  # noqa: PLC0415

    family = _family()

    def coverage(context: Any) -> Coverage:
        covered, reason = family.covers_complex_fused_hd_pair(
            context.fields, context.pml, context.grid, context.sources,
            context.license_for(LICENSE_COMPLEX), context.subnormal_policy)
        return Coverage(bool(covered), (str(reason),))

    def plan(context: Any):
        held: Dict[str, Any] = {}

        def resolve(ctx: Any) -> Dict[str, Any]:
            if "state" not in held:
                held["state"] = family.resolve(
                    ctx.fields, ctx.grid, ctx.pml,
                    ctx.license_for(LICENSE_COMPLEX)["arm"])
            return {"state": held["state"]}

        def launch(fields: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
            return family.run_complex_fused_hd_pair(
                fields, fields.grid, context.pml, arguments["state"]["arm"],
                sources=context.sources, state=arguments["state"], check=False)

        return fused_pairs.CudaFusedPairPlan(
            family=family.FAMILY, label="complex fused H/D pair",
            kernel_label=family.KERNEL_NAME, replaces=tuple(family.REPLACES),
            slots=("update_H", "step_D"), context=context,
            resolve_launch_args=resolve, launch=launch)

    @contextlib.contextmanager
    def scope():
        added_product = family.FAMILY not in fused_pairs.FUSED_PRODUCTS
        added_arm = family.FAMILY not in fused_pairs.FUSED_PAIR_ARMS
        if added_product:
            fused_pairs.FUSED_PRODUCTS[family.FAMILY] = {
                "curl_slot": "update_H", "coverage": coverage, "plan": plan,
                "module": "complex_fused_hd_pair"}
        if added_arm:
            fused_pairs.FUSED_PAIR_ARMS[family.FAMILY] = WIRING_ARMS_ROW
        try:
            yield
        finally:
            if added_product:
                fused_pairs.FUSED_PRODUCTS.pop(family.FAMILY, None)
            if added_arm:
                fused_pairs.FUSED_PAIR_ARMS.pop(family.FAMILY, None)

    return scope()


def _ask_the_composer(fields, grid, pml, sources=()) -> Dict[str, Any]:
    from meep_gpu.cuda_kernels import arms  # noqa: PLC0415

    family = _family()
    licenses = ({"complex": _CONTEXT["licence"]} if _CONTEXT["licence"] is not None
                else None)
    plan = arms.plan_step(fields=fields, pml=pml, grid=grid, sources=sources,
                          fuse=True, licenses=licenses,
                          subnormal_policy=_CONTEXT["policy"])
    selected = dict(getattr(plan, "selected", {}) or {})
    reasons = {name: list(value) for name, value
               in (getattr(plan, "reasons", {}) or {}).items()}
    mine = [text for key, texts in reasons.items() if family.FAMILY in key
            for text in texts]
    return {"selected": selected, "this_product_is_refused": bool(mine),
            "refusal_reasons": mine,
            "this_product_holds_no_slot": not any(
                family.FAMILY in str(value) for value in selected.values())}


def leg_arbitration(spec: Mapping[str, Any], arm) -> Dict[str, Any]:
    """The COMPOSER, asked twice: AS SHIPPED and WITH the wiring rows patched in."""
    from meep_gpu.cuda_kernels import fused_pairs  # noqa: PLC0415

    family = _family()
    fields, grid, pml, _dtdx, _plant = build(spec, "uniform", case_rng("arbitration"))
    shipped = _ask_the_composer(fields, grid, pml)
    with _wired_in_process(arm):
        wired = _ask_the_composer(fields, grid, pml)
        wired["product_row_present"] = family.FAMILY in fused_pairs.FUSED_PRODUCTS
        wired["declared_uninstallable"] = fused_pairs._declared_uninstallable(  # noqa: SLF001
            family.FAMILY, fused_pairs.FUSED_PRODUCTS[family.FAMILY])
    after = family.FAMILY in fused_pairs.FUSED_PRODUCTS
    holder = wired["selected"].get("update_H")
    record = {
        "spec": spec["label"], "variant": spec["variant"],
        "as_shipped": shipped,
        "with_the_wiring_rows_patched_in_process": wired,
        "the_rows_were_removed_afterwards": not after,
        "the_selection_is_unchanged_by_the_wiring":
            shipped["selected"] == wired["selected"],
        "the_product_is_absent_as_shipped": (
            not shipped["this_product_is_refused"]
            and shipped["this_product_holds_no_slot"]),
        "the_wired_product_is_refused_by_name": wired["this_product_is_refused"],
        "the_refusal_names_installable": any(
            "INSTALLABLE = False" in text for text in wired["refusal_reasons"]),
        "the_released_b_to_h_pair_keeps_update_H": bool(
            holder is not None and wired["selected"].get("step_B") == holder
            and family.FAMILY not in str(holder)),
        "the_released_d_to_e_pair_keeps_step_D": bool(
            wired["selected"].get("step_D") is not None
            and wired["selected"].get("step_D") == wired["selected"].get("update_E")
            and family.FAMILY not in str(wired["selected"].get("step_D"))),
        "installable": bool(family.INSTALLABLE),
        "installable_reason": family.INSTALLABLE_REASON,
        "what_a_release_does_not_license": family.WHAT_A_RELEASE_DOES_NOT_LICENSE,
        "standalone_note": (
            "WIRED 2026-09-07: this family IS in fused_pairs/registry/arms on disk now. "
            "Before that round it was not, and the sentence below describes that state.  Every "
            "arrangement in this gate is planned from arrays by the gate and the "
            "array path is driven by the gate; the composition-installed-today "
            "reference IS driven (the two released complex pairs). What awaits the "
            "wiring round is the composer building a composition that INCLUDES this "
            "product, which is measured here with the rows patched in-process"),
    }
    # REPINNED TO THE WIRED STATE, 2026-09-07. Two of these clauses asserted the
    # PRE-WIRING tree -- that the product is absent from `FUSED_PRODUCTS` as shipped, and
    # that the in-process rows were removed afterwards -- and the wiring round is exactly
    # what inverts them: the rows are permanent now, so `after` is True and the product is
    # present as shipped. Neither clause was ever the substantive claim. The arbitration
    # finding is the other five, and every one still holds on the wired tree: the product
    # is refused BY NAME, the refusal names INSTALLABLE = False, the selection is unchanged
    # by the wiring, and both released complex pairs keep their own slots. Those are
    # asserted below; the two positional ones are recorded, not asserted.
    record["passed"] = bool(
        record["the_wired_product_is_refused_by_name"]
        and record["the_refusal_names_installable"]
        and record["the_selection_is_unchanged_by_the_wiring"]
        and not record["installable"]
        and wired["this_product_holds_no_slot"]
        and record["the_released_b_to_h_pair_keeps_update_H"]
        and record["the_released_d_to_e_pair_keeps_step_D"])
    return record


# ---------------------------------------------------------------------------
# The mutations
# ---------------------------------------------------------------------------

DEVICE_MUTATIONS: Dict[str, Dict[str, Any]] = {
    "foreign_tap_reads_stale_H": {
        "old": "    if (ia > 0) return resolve_H(comp, idx - stride, weld);",
        "new": ("    if (ia > 0) return cf_load(comp == 0 ? weld.Hx\n"
                "                        : comp == 1 ? weld.Hy : weld.Hz,\n"
                "                        idx - stride);"),
        "why": ("the foreign tap loads the PRE-LAUNCH magnetic field instead of "
                "recomputing update_H there -- which is what an unfused step_D would "
                "read and is exactly one sub-step behind"),
        "live_on": "every fixture",
    },
    "foreign_tap_reads_B": {
        "old": "    if (ia > 0) return resolve_H(comp, idx - stride, weld);",
        "new": ("    if (ia > 0) return cf_load(comp == 0 ? weld.Bx\n"
                "                        : comp == 1 ? weld.By : weld.Bz,\n"
                "                        idx - stride);"),
        "why": ("the tap reads the constitutive SOURCE rather than its result. Under "
                "mu = 1 outside the absorber H and B are close, so this is the "
                "plausible wrong answer rather than an obvious one"),
        "live_on": "every fixture",
    },
    "own_cell_reads_stale_H": {
        "old": "        cf f_1 = own_h[2];",
        "new": "        cf f_1 = cf_load(g2, idx);",
        "why": ("the thread's OWN cell load reverts to the pre-launch volume, so Dx's "
                "curl differences an H one sub-step behind on one of its two terms"),
        "live_on": "every fixture",
    },
    "f_w_H_written_in_place": {
        "old": "    *fw_out = src;",
        "new": "    cf_store(const_cast<float*>(fw), idx, src);",
        "why": ("the split-field history is stored IN PLACE, so a foreign recompute "
                "reading fw at a cell another block already wrote gets B where its "
                "recurrence needs B_prev. RACY BY CONSTRUCTION: whether it fires is a "
                "schedule, which is why the deterministic twin below is armed beside "
                "it"),
        "live_on": "every fixture (schedule-dependent)",
    },
    "foreign_history_reads_the_post_store_value": {
        "old": "    cf prev = cf_load(fw, idx);",
        "new": "    cf prev = src;",
        "why": ("THE DETERMINISTIC TWIN of the in-place store: the recurrence reads "
                "the value the in-place arrangement would have handed it, with no "
                "race to depend on"),
        "live_on": "every fixture with an active absorber (kms != 0)",
    },
    "the_wrapped_lane_is_not_phased": {
        # ANCHORED ON THE RECOMPUTE'S OWN WRAP, not on the bare phase line: the
        # certified ``cshift_up`` and ``cshift_dn`` ride along in the lifted prelude
        # and spell the same line, so a one-line needle finds three sites and lands
        # nowhere. Measured on this gate's first mutation smoke (2026-09-07).
        "old": ("    cf w = resolve_H(comp, idx + (na - 1) * stride, weld);\n"
                "    if (ph) w = rotate_field_left(w, phase);"),
        "new": ("    cf w = resolve_H(comp, idx + (na - 1) * stride, weld);\n"
                "    if (ph) w = w;"),
        "why": ("the Bloch factor is dropped from the wrapped lane, which is the "
                "band-structure error: every magnitude stays plausible and only the "
                "phase moves"),
        "live_on": "periodic_kx, periodic_kxyz, brillouin_edge_kx, fold_y_odd_kx",
    },
    "the_phase_is_applied_to_the_near_neighbour_too": {
        "old": "    if (ia > 0) return resolve_H(comp, idx - stride, weld);",
        "new": ("    if (ia > 0) {\n"
                "        cf n = resolve_H(comp, idx - stride, weld);\n"
                "        if (ph) n = rotate_field_left(n, phase);\n"
                "        return n;\n    }"),
        "why": ("the phase is applied to EVERY lane rather than to the wrapped one "
                "only -- ``_apply_bloch_phase`` multiplies the single plane "
                "_face(axis, -1) of the rolled buffer and nothing else"),
        "live_on": "periodic_kx, periodic_kxyz, brillouin_edge_kx",
    },
    "accumulations_regrouped": {
        "old": ("    a = cf_add(a, mul_coefficient_left(kps, src));\n"
                "    a = cf_sub(a, mul_coefficient_left(kms, prev));"),
        "new": ("    a = cf_add(a, cf_sub(mul_coefficient_left(kps, src),\n"
                "                         mul_coefficient_left(kms, prev)));"),
        "why": ("the two SEPARATE accumulations become one expression "
                "f + (kps*src - kms*prev), which is a different float32 number "
                "because addition is not associative. MEEP's step_update_EDHB "
                "accumulates left to right and the certified body transcribes that; "
                "flattening it is the defect the twelve uncertified complex kernels "
                "in step_curl_kernels carry, and it reads as a slightly weaker "
                "absorber rather than as a bug"),
        "live_on": "every fixture with an active absorber",
    },
    "coefficient_index_is_the_threads_own": {
        "old": ("    int k = idx % nz;\n    int j = (idx / nz) % ny;\n"
                "    int i = idx / (ny * nz);\n\n    // NO ghost rule"),
        "new": ("    int k = 0;\n    int j = 0;\n    int i = 0;\n\n    // NO ghost rule"),
        "why": ("the recomputed cell's absorber coefficients are taken from the origin "
                "instead of from the recomputed cell -- the half-cell class of error, "
                "converged and smooth"),
        "live_on": "every fixture with an active absorber",
    },
    "naive_four_product_multiply": {
        "old": ("__device__ __forceinline__ cf mul_field_left(cf z, float c) {\n"
                "    cf o;\n"
                "    o.re = __fmaf_rn(z.re, c, (z.im * 0.0f) * -1.0f);\n"
                "    o.im = __fmaf_rn(z.re, 0.0f, z.im * c);\n"),
        "new": ("__device__ __forceinline__ cf mul_field_left(cf z, float c) {\n"
                "    cf o;\n"
                "    o.re = (z.re * c) - (z.im * 0.0f);\n"
                "    o.im = (z.re * 0.0f) + (z.im * c);\n"),
        "why": ("CuPy's complex64 * float32 is the four-product form CONTRACTED by "
                "NVRTC; the uncontracted transcription differs on the SIGN OF A "
                "FLUSHED ZERO. Measured 6 words of 4,005,000 on (flush, edge) by "
                "lanes/cyl_round/cupy_probe -- a class a random battery does not "
                "reach, which is why the planted_row0 value class exists"),
        "live_on": "the planted_row0 class under flush",
        "arm_only": "FMA_V1",
        "null_under": ("keep",),
        "null_reason": (
            "under keep nothing flushes, so the two forms agree bit for bit: "
            "fma(a, b, +-0) rounds a*b once, zero signs included. The defect is "
            "POLICY-CONDITIONAL and a battery that averaged the two policies would "
            "report it as flaky rather than as what it is"),
    },
    "fmad_default_build": {
        "old": None,   # not a text edit: the compile options
        "new": None,
        "options": (),
        "why": ("the same source built WITHOUT --fmad=false, so NVRTC may contract "
                "the accidental multiply-adds the transcription does not spell"),
        "live_on": "every fixture",
        "null_under": ("keep", "flush"),
        "null_reason": (
            "PREDICTED NULL ON THE COMPLEX PRIMARY, and the prediction is a prior "
            "MEASUREMENT rather than this run's excuse: lanes/cyl_round/cupy_probe "
            "measured the same source built with and without --fmad=false at 0 of "
            "14,387,504 words under both policies on the complex spelling "
            "(FINDINGS.md section 3, the row 'same source built WITHOUT --fmad=false "
            "... does NOT bite on the complex primary'), and the released cylindrical "
            "complex H->D campaign carries the same held null. THE CAUSE is that with "
            "the arm's multiplies spelled as explicit __fmaf_rn there is no "
            "contractible multiply-add left for the flag to remove. That cause is "
            "MEASURED here rather than argued: the fmad_attribution sub-leg builds "
            "the UNCONTRACTED arm (NAIVE) both ways and requires those two to differ, "
            "so a null whose real cause was a flag that never reached NVRTC would "
            "fail. The control is kept armed because it must still COMPILE and RUN"),
    },
}

#: The one armed edit required NOT to diverge: the own-cell register replaced by a
#: RELOAD of the scratch word this thread has just stored -- the same value by a
#: different route. A byte-neutral control that DID diverge would mean the harness,
#: not the weld, decides the answer.
BYTE_NEUTRAL: Dict[str, str] = {
    "old": "        cf f_1 = own_h[2];",
    "new": "        cf f_1 = cf_load(Hz_out, idx);",
}

HOST_MUTATIONS: Tuple[str, ...] = ("rotation_skipped", "scratch_aliased_to_storage",
                                   "swap_constitutive_sublattice",
                                   "unconjugated_backward_phase",
                                   "the_wrong_variant_text")


def _mutated_source(name: str, variant: str, arm) -> Optional[str]:
    family = _family()
    spec = DEVICE_MUTATIONS[name]
    if spec["old"] is None:
        return None
    source = family.kernel_source(variant, arm)
    source = needle(source, spec["old"], spec["new"])
    for old, new in spec.get("also", ()):
        source = needle(source, old, new)
    return source


def _fmad_attribution(spec: Mapping[str, Any], steps: int) -> Dict[str, Any]:
    """Why ``fmad_default_build`` is null, MEASURED rather than argued.

    The SAME two builds are compared on the arm whose multiplies are NOT explicit
    fmas -- ``NAIVE``, the uncontracted transcription. If the flag were never
    reaching NVRTC, those two would agree too and the null would be about the harness
    rather than about the spelling. That they DIFFER attributes the null to the
    explicit ``__fmaf_rn`` in ``FMA_V1``, which is the claim the record makes.
    """
    family = _family()
    states: Dict[str, Dict[str, np.ndarray]] = {}
    for label, options in (("guarded", family._COMPILE_OPTIONS), ("default", ())):  # noqa: SLF001
        kernel = family._get_kernel(spec["variant"], "NAIVE", options=options)  # noqa: SLF001
        fields, grid, pml, dtdx, _plant = build(spec, "uniform", case_rng("fmad"))
        arrangement = make_arrangement("weld", fields, grid, pml, dtdx, "NAIVE",
                                       spec["variant"], kernel=kernel)
        for _ in range(steps):
            arrangement.step()
        cp.cuda.runtime.deviceSynchronize()
        states[label] = frozen(fields)
    moved = compare(states["guarded"], states["default"])
    return {
        "spec": spec["label"], "steps": steps, "arm": "NAIVE",
        "differing_volumes": moved,
        "the_flag_changes_the_answer_on_the_uncontracted_arm": bool(moved),
        "what_this_attributes": (
            "--fmad=false DOES reach NVRTC and DOES change the emitted arithmetic on "
            "a source that carries contractible multiply-adds. What makes the flag "
            "null on the SHIPPED arm is that FMA_V1 spells its fusions as explicit "
            "__fmaf_rn, leaving nothing for the flag to remove -- not a build option "
            "that never arrived"),
        "passed": bool(moved),
    }


def leg_mutation(specs: Sequence[Mapping[str, Any]], steps: int, arm,
                 policy: Optional[str]) -> Dict[str, Any]:
    """Every armed defect, each required to DIVERGE on at least one scored fixture."""
    family = _family()
    arm_name = str(arm)
    results: List[Dict[str, Any]] = []
    for name, spec in DEVICE_MUTATIONS.items():
        if spec.get("arm_only") and spec["arm_only"] != arm_name:
            results.append({"name": name, "kind": "device", "compiled": None,
                            "caught": False, "passed": True,
                            "skipped": f"armed only under {spec['arm_only']}"})
            continue
        caught_on: List[str] = []
        compiled, compile_error = True, None
        for case in specs:
            variant = case["variant"]
            try:
                source = _mutated_source(name, variant, arm)
                options = spec.get("options")
                kernel = family._get_kernel(  # noqa: SLF001
                    variant, arm, source=source,
                    options=options if options is not None else None)
                kernel.compile()
            except Exception as error:  # noqa: BLE001
                compiled, compile_error = False, repr(error)[:400]
                break
            for value_class in ("uniform", "planted_row0"):
                record = drive(case, value_class, min(steps, 12), arm,
                               modes=("array", "weld"), kernel=kernel)
                if not record["identical"]["weld"]:
                    caught_on.append(f"{case['label']}/{value_class}")
        entry = {"name": name, "kind": "device", "compiled": compiled,
                 "compile_error": compile_error, "why": spec["why"],
                 "live_on": spec["live_on"], "caught_on": caught_on,
                 "caught": bool(caught_on)}
        entry["passed"] = bool(compiled and caught_on)
        if policy in (spec.get("null_under") or ()):
            entry["predicted_null_under_this_policy"] = spec["null_reason"]
            entry["prediction_held"] = not caught_on
            entry["passed"] = bool(compiled)
        if not entry["passed"] and name == "f_w_H_written_in_place":
            entry["predicted_null"] = True
            entry["reason"] = (
                "schedule-dependent by construction; the deterministic twin "
                "foreign_history_reads_the_post_store_value carries the same "
                "arithmetic consequence and must be CAUGHT")
            entry["passed"] = compiled
        results.append(entry)

    case = specs[0]
    for name in HOST_MUTATIONS:
        try:
            if name == "rotation_skipped":
                record = drive(case, "uniform", min(steps, 12), arm,
                               modes=("array", "weld"), rotate=False)
                caught = not record["identical"]["weld"]
                detail = {"first_divergence": record["first_divergence"]["weld"]}
            elif name == "scratch_aliased_to_storage":
                fields, grid, pml, _d, _p = build(case, "uniform", case_rng("alias"))
                state = family.resolve(fields, grid, pml, arm,
                                       variant=case["variant"])
                state["scratch"] = {vol: getattr(fields, vol)
                                    for vol in family.SCRATCH_VOLUMES}
                try:
                    family.assert_bindings_are_disjoint(fields, state)
                    caught, detail = False, {"refused": None}
                except ValueError as error:
                    caught, detail = True, {"refused": str(error)[:400]}
            elif name == "swap_constitutive_sublattice":
                from meep_gpu.cuda_kernels import (  # noqa: PLC0415
                    complex_emitter, complex_pml_kernels)
                fields, grid, pml, _d, _p = build(case, "uniform", case_rng("swap"))
                half = complex_pml_kernels.complex_constitutive_tables(pml, True)
                record = drive(case, "uniform", min(steps, 8), arm,
                               modes=("array", "weld"),
                               state_overrides={"constitutive": half,
                                                "tables": complex_pml_kernels
                                                .complex_curl_tables(
                                                    pml, complex_emitter
                                                    .HALF_INTEGER["step_D"])})
                caught = not record["identical"]["weld"]
                detail = {"first_divergence": record["first_divergence"]["weld"],
                          "note": ("update_H's kps/kms taken from the HALF-INTEGER "
                                   "sub-lattice -- a half-cell error in the absorber "
                                   "profile, converged and smooth")}
            elif name == "unconjugated_backward_phase":
                from meep_gpu.cuda_kernels import complex_pml_kernels  # noqa: PLC0415
                phased = next((c for c in specs if any(c["k_point"])), None) or case
                fields, grid, pml, _d, _p = build(phased, "uniform",
                                                  case_rng("phase"))
                # THE FORWARD factor where the backward one belongs: the classic sign
                # error, in which every magnitude stays plausible and only the phase
                # moves.
                flags, values = complex_pml_kernels.bloch_phase_arguments(grid, False)
                record = drive(phased, "uniform", min(steps, 8), arm,
                               modes=("array", "weld"),
                               state_overrides={"phase_flags": flags,
                                                "phase_values": values})
                caught = not record["identical"]["weld"]
                detail = {"first_divergence": record["first_divergence"]["weld"],
                          "fixture": phased["label"]}
            else:   # the_wrong_variant_text
                folded = next((c for c in specs if c["variant"] == "folded"), None)
                if folded is None:
                    caught, detail = True, {"skipped": "no folded fixture in scope"}
                else:
                    kernel = family._get_kernel(  # noqa: SLF001
                        "folded", arm,
                        source=family.kernel_source("plain", arm).replace(
                            family.KERNEL_NAMES["plain"],
                            family.KERNEL_NAMES["folded"]))
                    record = drive(folded, "uniform", min(steps, 12), arm,
                                   modes=("array", "weld"), kernel=kernel)
                    caught = not record["identical"]["weld"]
                    detail = {"first_divergence": record["first_divergence"]["weld"],
                              "fixture": folded["label"],
                              "note": ("the PLAIN text run on a FOLDED grid -- the "
                                       "defect complex_folded_kernels' site counts "
                                       "exist to prevent: it compiles, it runs, and "
                                       "it is wrong on one plane of one component")}
            results.append({"name": name, "kind": "host", "compiled": True,
                            "caught": caught, "passed": caught, **detail})
        except Exception as error:  # noqa: BLE001
            results.append({"name": name, "kind": "host", "compiled": False,
                            "caught": False, "passed": False,
                            "error": repr(error)[:400]})
    attribution = _fmad_attribution(specs[0], min(steps, 8))
    family._clear_kernel_cache()  # noqa: SLF001
    armed = [entry for entry in results if not entry.get("skipped")]
    return {"mutations": results, "denominator": len(results), "armed": len(armed),
            "caught": sum(1 for entry in results if entry["caught"]),
            "predicted_null": [entry["name"] for entry in results
                               if entry.get("predicted_null")
                               or entry.get("predicted_null_under_this_policy")],
            "held": [entry["name"] for entry in results
                     if entry.get("prediction_held")],
            "refuted_predictions": [entry["name"] for entry in results
                                    if entry.get("prediction_held") is False],
            "uncaught": [entry["name"] for entry in results if not entry["caught"]
                         and not entry.get("skipped")],
            "fmad_attribution": attribution,
            "passed": (all(entry["passed"] for entry in results)
                       and attribution["passed"])}


def leg_byte_neutral(spec: Mapping[str, Any], steps: int, arm) -> Dict[str, Any]:
    """The one armed edit required NOT to diverge."""
    family = _family()
    source = needle(family.kernel_source(spec["variant"], arm),
                    BYTE_NEUTRAL["old"], BYTE_NEUTRAL["new"])
    kernel = family._get_kernel(spec["variant"], arm, source=source)  # noqa: SLF001
    record = drive(spec, "uniform", min(steps, 12), arm,
                   modes=("array", "weld"), kernel=kernel)
    family._clear_kernel_cache()  # noqa: SLF001
    return {"spec": spec["label"], "identical": record["identical"]["weld"],
            "first_divergence": record["first_divergence"]["weld"],
            "why": ("the own-cell register is replaced by a RELOAD of the scratch "
                    "word this thread has just stored -- the same value by a "
                    "different route. A float32 stored to global memory and loaded "
                    "back is the identity on the bits, so an edit that diverged here "
                    "would mean the harness rather than the weld decides the answer"),
            "passed": bool(record["identical"]["weld"])}


def leg_disarm(spec: Mapping[str, Any], steps: int, arm) -> Dict[str, Any]:
    """The identical harness on the SHIPPED bytes, required not to diverge."""
    record = drive(spec, "uniform", min(steps, 12), arm, modes=("array", "weld"))
    return {"spec": spec["label"], "identical": record["identical"]["weld"],
            "first_divergence": record["first_divergence"]["weld"],
            "passed": bool(record["identical"]["weld"])}


# ---------------------------------------------------------------------------
# Device leg: the compiler
# ---------------------------------------------------------------------------

def leg_compiler(policy: Optional[str], at_install: Mapping[str, Any]) -> Dict[str, Any]:
    """Did THIS process compile the kernels it ran, and did the policy reach NVRTC?"""
    report = probe.nvrtc_binary_report()
    outer = probe.subnormal_policy_stamp(_REPO_API)
    stamp = dict(outer.get("stamp") or {})
    stamp.setdefault("policy", outer.get("policy"))
    now = common.cupy_cache_snapshot()
    family = _family()
    try:
        family_source = family.source_digest()
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
    emitted = {variant: hashlib.sha256(
        family.kernel_source(variant, "FMA_V1").encode("utf-8")).hexdigest()
        for variant in family.VARIANTS}
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
        "per_variant_source_sha256": emitted,
        "variant_sources_observed_at_nvrtc": {
            variant: digest in observed_sources for variant, digest in emitted.items()},
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


def lift_basis(results: Path) -> Tuple[List[dict], Dict[str, Any]]:
    """The rows the standing BOARD puts in this product's two cells.

    DERIVED from the board's own ``h_to_d_seam.instances`` -- the arms the census
    chose for each row -- never from a list here. The MODULE a tests row lives in is
    the CENSUS's fact and is joined from its ``tests*.jsonl``; a tests row with no
    module in either record is REFUSED BY NAME rather than guessed at.
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
    by_cell = {cell: variant for variant, cell in CELLS.items()}
    rows: List[dict] = []
    for record in instances:
        cell = (record.get("update_H"), record.get("step_D"))
        variant = by_cell.get(cell)
        if variant is None:
            continue
        label = record["row"]
        leg, _, row = label.partition(":")
        rows.append({
            "label": label, "leg": leg, "row": row, "variant": variant,
            "cell": list(cell),
            "module": modules.get(label),
            "replay_case": replay_case.get(label, row),
            "interpreter": sys.executable,
            "bucket": record.get("bucket"),
            "withdraw_in_seam": bool(record.get("withdraw_in_seam")),
            "integrated_electric_sources":
                int(record.get("integrated_electric_sources") or 0),
        })
    facts = {
        "board": BOARD, "census": CENSUS,
        "cells": {variant: list(cell) for variant, cell in CELLS.items()},
        "rows_per_cell": {variant: sum(1 for row in rows if row["variant"] == variant)
                          for variant in CELLS},
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


def _child_progress(message: str) -> None:
    path = os.environ.get("MEEP_GPU_CXHD_GATE_PROGRESS")
    label = os.environ.get("MEEP_GPU_CXHD_GATE_LABEL", "?")
    line = f"[{time.strftime('%H:%M:%S')}] {label} {message}"
    print(line, flush=True)
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()


BLOCK_KEY = "cuda_complex_fused_hd_pair_gate"


def evaluate_row(driver: Any, steps: int, max_cells: Optional[str]) -> Dict[str, Any]:
    """The identity on ONE lifted row, plus the predicate and composer facts."""
    from meep_gpu.cuda_kernels import arms  # noqa: PLC0415

    family = _family()
    licence = _CONTEXT["licence"]
    policy = _CONTEXT["policy"]
    arm = (licence or {}).get("arm")
    sources = tuple(getattr(driver, "_sources", ()) or ())
    block: Dict[str, Any] = {
        "family": family.FAMILY, "steps_requested": steps,
        "n_sources": len(sources), "arm": arm,
        "grid_shape": [int(n) for n in driver.grid.shape],
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
    try:
        block["variant"] = family.variant_for(driver.grid)
    except Exception as error:  # noqa: BLE001
        block["variant"] = None
        block["variant_error"] = repr(error)[:200]
    covered, reason = family.covers_complex_fused_hd_pair(
        driver.fields, driver.pml, driver.grid, sources, licence, policy)
    block["predicate_admits"] = bool(covered)
    block["predicate_reason"] = reason
    try:
        plan = arms.plan_step(
            fields=driver.fields, pml=driver.pml, grid=driver.grid, sources=sources,
            fuse=True,
            licenses={"complex": licence} if licence is not None else None,
            subnormal_policy=policy)
        block["composer_selected_as_shipped"] = dict(getattr(plan, "selected", {}) or {})
    except Exception as error:  # noqa: BLE001
        block["composer_error"] = repr(error)[:400]
    cells = int(np.prod(driver.grid.shape))
    block["grid_cells"] = cells
    if not covered:
        block.update({"driven": False,
                      "why_not_driven": "the predicate refused this row"})
        _child_progress(f"REFUSED {str(reason)[:120]}")
        return {BLOCK_KEY: block}
    if max_cells and cells > int(max_cells):
        block.update({"driven": False,
                      "why_not_driven": (f"{cells} cells exceeds the cap "
                                         f"{max_cells}; refused rather than run "
                                         f"partially")})
        return {BLOCK_KEY: block}
    block["driven"] = True
    block.update(_drive_lifted_row(driver, steps, arm, block["variant"]))
    block["policy_stamp_after_drive"] = (
        probe.subnormal_policy_stamp(_REPO_API).get("policy"))
    _child_progress(f"passed={block.get('passed')} words={block.get('words_compared')}")
    return {BLOCK_KEY: block}


def _capture(driver: Any) -> Dict[str, Any]:
    from gate_cuda_fused_hd_pair import _capture as capture  # noqa: PLC0415
    return capture(driver)


def _restore(driver: Any, snapshot: Mapping[str, Any]) -> None:
    from gate_cuda_fused_hd_pair import _restore as restore  # noqa: PLC0415
    restore(driver, snapshot)


def _stored_volumes(fields: Any) -> Dict[str, Any]:
    from gate_cuda_fused_hd_pair import _stored_volumes as stored  # noqa: PLC0415
    return stored(fields)


def _drive_lifted_row(driver: Any, steps: int, arm, variant: str) -> Dict[str, Any]:
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
    family = _family()
    fields, pml, grid = driver.fields, driver.pml, driver.grid
    dtdx = float(grid.dt / grid.dx)
    singles = _certified_single_launchers(fields, grid, pml, dtdx, arm, variant)
    state = family.resolve(fields, grid, pml, arm, variant=variant)
    family.assert_bindings_are_disjoint(fields, state)
    book = {"launches": 0}

    def weld() -> None:
        family.launch_complex_fused_hd_pair(fields, state)
        book["launches"] += 1
        state["scratch"] = family.rotate_into_fields(fields, state["scratch"])

    arrangements = {
        "weld": {"update_H": weld, "step_D": lambda: None},
        "singles": {"update_H": singles["update_H"], "step_D": singles["step_D"]},
    }
    first: Dict[str, Optional[Dict[str, Any]]] = {"weld": None, "singles": None}
    agree = 0
    words = 0
    compared = 0
    subnormals = 0
    for step in range(steps):
        base = _capture(driver)
        results: Dict[str, Dict[str, np.ndarray]] = {}
        for name, plans in arrangements.items():
            _restore(driver, base)
            driver._fast_path = Shim(plans)  # noqa: SLF001
            driver._fast_path_stale = False  # noqa: SLF001
            driver.step()
            cp.cuda.runtime.deviceSynchronize()
            results[name] = {key: to_host(value).copy()
                             for key, value in _stored_volumes(fields).items()}
        _restore(driver, base)
        driver._fast_path = None  # noqa: SLF001
        driver._fast_path_stale = False  # noqa: SLF001
        driver.step()
        cp.cuda.runtime.deviceSynchronize()
        array = {key: to_host(value).copy()
                 for key, value in _stored_volumes(fields).items()}
        compared = step + 1
        words += int(sum(np.asarray(value).nbytes // 4 for value in array.values()))
        subnormals += int(sum(common.subnormal_words(np.asarray(value).view(
            np.float32) if np.iscomplexobj(value) else value)
            for value in array.values()))
        for name in arrangements:
            moved = {key: int(differing(array[key], results[name][key]))
                     for key in array
                     if differing(array[key], results[name][key])}
            if moved and first[name] is None:
                first[name] = {"step": step + 1, "volumes": moved}
        if first["weld"] and first["singles"] and (
                first["weld"]["step"] == first["singles"]["step"]):
            same = all(differing(results["weld"][key], results["singles"][key]) == 0
                       for key in array)
            if same:
                agree += 1
        if first["weld"] is not None:
            break
    return {
        "steps_compared": compared, "words_compared": words,
        "subnormal_words_seen": subnormals,
        "weld_launches": book["launches"],
        "first_divergence": first,
        "the_weld_and_the_certified_singles_agree_where_both_differ": bool(agree),
        "identical_to_the_array_path": first["weld"] is None,
        "certified_singles_identical_to_the_array_path": first["singles"] is None,
        "passed": bool(first["weld"] is None and book["launches"] > 0),
    }


def _lift_child(leg: str, target: str, cases: Sequence[str], out_json: str,
                steps: int, progress: str, policy: Optional[str],
                import_meep: bool, licence: Optional[dict]) -> int:
    """One corpus row, lifted in its OWN interpreter and driven. ALWAYS writes.

    THE POLICY IS INSTALLED HERE, before the child's first compile, and stamped before
    and after the drive: a child that inherited only a cache directory would compile
    under whatever policy the parent left installed in ITS process, which is not this
    process. The two lift entry points are the ones the corpus harness already owns --
    ``sweep_corpus_lift_parity.capture_simulation`` for an example script and
    ``survey_meep_tests``' child namespace for a test module -- so this file replays a
    corpus row the same way every other lift on this track does.
    """
    import contextlib  # noqa: PLC0415

    import meep_gpu  # noqa: PLC0415

    os.environ["MEEP_GPU_CXHD_GATE_PROGRESS"] = progress or ""
    os.environ["MEEP_GPU_CXHD_GATE_LABEL"] = f"{leg}:{Path(str(target)).name}"
    if import_meep:
        probe.import_meep_for_host_policy()
    probe.install_nvrtc_binary_observer()
    if policy:
        probe.install_subnormal_policy_for_run(policy, _REPO_API)
    _CONTEXT["licence"] = licence
    _CONTEXT["policy"] = policy
    _child_progress(f"child start policy={policy} steps={steps} arm="
                    f"{(licence or {}).get('arm')}")

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
    max_cells = os.environ.get("MEEP_GPU_CXHD_GATE_MAX_CELLS")
    for record, sim in pairs:
        started = time.time()
        try:
            driver = meep_gpu.lift_simulation(sim, prefer_gpu=True, gpu_id=0)
        except BaseException as exc:  # noqa: BLE001
            record.update({"measured": False,
                           "lift_error": f"{type(exc).__name__}: {exc}"[:600]})
            _child_progress(f"LIFT FAILED {type(exc).__name__}")
            continue
        record["lift_s"] = round(time.time() - started, 2)
        record["grid_cells"] = int(np.prod([int(v) for v in driver.shape]))
        try:
            record.update(evaluate_row(driver, steps, max_cells))
            record["measured"] = True
        except BaseException as exc:  # noqa: BLE001
            record.update({"measured": False,
                           "battery_error": f"{type(exc).__name__}: {exc}"[:600]})
        finally:
            with contextlib.suppress(BaseException):
                driver.close()
    return finish()


def leg_lift(out_dir: Path, steps: int, max_cells: Optional[int], timeout: float,
             resume: bool, only: Optional[Sequence[str]], policy: Optional[str],
             import_meep: bool, licence: Optional[dict]) -> Dict[str, Any]:
    """Every corpus row in the two cells, re-lifted in its own interpreter and driven."""
    import contextlib  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    import measure_predicate_coverage as census  # noqa: PLC0415

    out_dir = Path(out_dir).resolve()
    rows, facts = lift_basis(Path(_HERE) / "results")
    if only:
        wanted = set(only)
        rows = [row for row in rows if row["label"] in wanted]
    # THE CORPUS PATH IS THE CENSUS'S, and it is REFUSED rather than guessed at. A
    # child handed a bare script NAME resolves it against its own cwd and dies with a
    # FileNotFoundError that the parent records as a right-shaped row measuring
    # nothing -- which is exactly what happened on this gate's first full run
    # (2026-09-07, seven examples rows). ``census.EXAMPLES_DIR``/``TESTS_DIR`` read
    # ``MEEP_GPU_CORPUS_ROOT``.
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
    workdir = Path(tempfile.mkdtemp(prefix="cuda_cxhd_lift_"))
    for entry in examples_dir.iterdir():
        if entry.suffix in (".py", ".ipynb"):
            continue
        with contextlib.suppress(OSError):
            (workdir / entry.name).symlink_to(entry)
    # THE `parameterized` SHIM, on the PYTHONPATH of the modules that import it: the
    # package is not installed on the validation host and the census ships a shim
    # whose expansion names cases `<method>__idx<N>`, which is the `replay_case` the
    # basis already joined.
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
                f"{row['label']} cell={row['variant']}")
        print(line, flush=True)
        with open(progress_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        if resume and out_json.is_file():
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            driven.append({**row, "payload": payload, "resumed": True})
            continue
        if row["withdraw_in_seam"]:
            driven.append({**row, "refused_by_the_predicate": True,
                           "why": "standing integrated electric withdraw in the seam"})
            continue
        if target is None:
            driven.append({**row, "error": "no module for this row in the census"})
            continue
        command = [sys.executable, "-u", str(Path(_HERE) / Path(__file__).name),
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
            environment["MEEP_GPU_CXHD_GATE_MAX_CELLS"] = str(max_cells)
        environment["MEEP_GPU_CXHD_GATE_LICENCE"] = json.dumps(licence)
        started = time.perf_counter()
        try:
            # ENCODING AND ERRORS ARE EXPLICIT, and both are load-bearing. ``text=True``
            # alone decodes the child's output with the interpreter's LOCALE encoding
            # -- ASCII under C/POSIX, which is what a non-interactive shell on the
            # validation host gets -- and a corpus test that prints one non-ASCII
            # character raises UnicodeDecodeError IN THE PARENT, killing the whole leg
            # after the child has already measured correctly. Measured 2026-09-07:
            # ``tests:TestAdjointSolver.test_complex_fields`` prints "ANISOTROPIC eps"
            # with a Greek epsilon and took the lift leg down at row 9 of 22 under
            # both policies, with fifteen other legs already green.
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
        """The gate block out of one child's payload -- a LIST of replay records."""
        payload = entry.get("payload")
        if isinstance(payload, dict):
            payload = [payload]
        for record in (payload or []):
            found = record.get(BLOCK_KEY)
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
        "rows_on_the_board_cells": len(rows),
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
        "per_cell": {variant: {
            "rows": sum(1 for row in rows if row["variant"] == variant),
            "driven": sum(1 for row in rows if row["variant"] == variant
                          and blocks[row["label"]].get("driven")),
            "identical": sum(1 for row in rows if row["variant"] == variant
                             and blocks[row["label"]].get(
                                 "identical_to_the_array_path")),
        } for variant in CELLS},
        "step_floor": LIFT_CLEAN_STEP_FLOOR,
    }
    record["passed"] = bool(
        ran and not diverged and not errored and not below
        and len(identical) == len(ran)
        and sorted(refused) == sorted(facts["rows_with_a_standing_withdraw"]))
    return record


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def save(results: Dict[str, Any], out: str) -> None:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    gate_provenance.stamp(results)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = common.make_parser(__doc__.splitlines()[0])
    args = parser.parse_args(argv)

    if args.lift_child:
        return _lift_child(
            args.lift_child, args.lift_child_target,
            json.loads(args.lift_child_cases), args.lift_child_out,
            args.lift_steps, args.lift_child_progress, args.subnormal_policy,
            bool(args.import_meep_for_host_policy),
            json.loads(os.environ.get("MEEP_GPU_CXHD_GATE_LICENCE", "null")))

    if not args.out:
        parser.error("--out is required unless --lift-child is given")
    started = time.perf_counter()
    legs = tuple(args.legs.split(",")) if args.legs else ALL_LEGS
    results: Dict[str, Any] = {
        "gate": Path(__file__).name,
        "family": FAMILY_NAME,
        "seam": withdraw_hoist.SEAM,
        "cells": {variant: list(cell) for variant, cell in CELLS.items()},
        "board": BOARD, "census": CENSUS,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "argv": list(argv if argv is not None else sys.argv[1:]),
        "legs_requested": list(legs), "steps": args.steps,
    }
    if cp is None or args.no_device:
        results["device_mode"] = False
        results["status"] = "refused: no CuPy" if cp is None else "structural only"
        save(results, args.out)
        log(results["status"])
        return 1

    results["device_mode"] = True
    at_install = common.install_policy(results, args)
    licence = common.load_expansion_licence(args.subnormal_policy)
    results["expansion_licence"] = licence
    if not licence.get("usable"):
        results["status"] = "refused: no usable expansion licence for this policy"
        results["verdict"] = {"passed": False, "clauses": {}}
        save(results, args.out)
        log(results["status"])
        return 1
    arm = licence["arm"]
    results["arm"] = arm
    # THE FULL VERDICT, not the arm alone: every complex predicate checks the verdict's
    # shape, its refusal list, its basis and the policy it was cut under, and a caller
    # that handed it a two-key summary would be refused at the last rung before the
    # arm is compiled into a binary.
    _CONTEXT["licence"] = licence["verdict"]
    _CONTEXT["policy"] = args.subnormal_policy
    os.environ["MEEP_GPU_CXHD_GATE_LICENCE"] = json.dumps(licence["verdict"],
                                                          default=str)

    family = _family()
    results["what_a_release_does_not_license"] = family.WHAT_A_RELEASE_DOES_NOT_LICENSE
    results["installable"] = bool(family.INSTALLABLE)
    results["source_digest"] = family.source_digest()
    results["kernels"] = dict(family.KERNEL_NAMES)
    save(results, args.out)

    specs = list(SPECS) if args.product == "full" else [
        spec for spec in SPECS if spec["label"] in REDUCED_LABELS]
    by_label = {spec["label"]: spec for spec in SPECS}
    mutation_specs = [by_label[label] for label in MUTATION_SPEC_LABELS]

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

    run_leg("driver_order", leg_driver_order)
    run_leg("transcription", leg_transcription)
    run_leg("refusal", leg_refusal)
    run_leg("spelling", lambda: leg_spelling(args.subnormal_policy))

    if "product" in legs:
        cases: List[Dict[str, Any]] = []
        for spec in specs:
            for value_class in VALUE_CLASSES:
                record = drive(spec, value_class, args.steps, arm,
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
        per_variant = {variant: sum(1 for r in cases if r["variant"] == variant)
                       for variant in family.VARIANTS}
        results["product"] = {
            "cases": cases, "denominator": len(cases),
            "arrangements_per_case": len(MODES),
            "complete_driver_steps": sum(r["steps_compared"] for r in cases) * len(MODES),
            "words_compared": sum(r["words_compared"] for r in cases),
            "cases_per_variant": per_variant,
            "the_band_class_really_contains_subnormals":
                bool(band) and all(r["operand_census"]["subnormals"] > 0 for r in band),
            "the_uniform_class_contains_none":
                bool(uniform) and all(r["operand_census"]["subnormals"] == 0
                                      for r in uniform),
            "the_planted_class_was_planted": all(
                r["plant"].get("planted") for r in cases
                if r["value_class"] == "planted_row0"),
            "failing_cases": [r["label"] for r in cases if not r["passed"]],
            # BOTH CELLS ARE A FLOOR, NOT A BONUS: a product leg that scored no folded
            # case cannot release, so the 5 folded seam-instances can never rest on the
            # plain variant's evidence.
            "passed": (bool(cases) and all(r["passed"] for r in cases)
                       and all(per_variant[v] > 0 for v in family.VARIANTS)),
        }
        save(results, args.out)

    run_leg("purity_ledger", lambda: leg_purity_ledger(specs, arm))

    if "block_sizes" in legs:
        sweep: List[Dict[str, Any]] = []
        for spec in specs:
            for threads in BLOCK_SIZES:
                record = drive(spec, "uniform", min(args.steps, 12), arm,
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
            "passed": bool(sweep) and all(r["identical"]["weld"] for r in sweep)}
        save(results, args.out)

    run_leg("launch_structure",
            lambda: leg_launch_structure(by_label["periodic_kxyz"],
                                         min(args.steps, 12), arm))
    run_leg("arbitration", lambda: leg_arbitration(by_label["periodic_kxyz"], arm))
    run_leg("sync", lambda: leg_sync(by_label["periodic_k0"], min(args.steps, 12), arm))
    run_leg("withdraw",
            lambda: leg_withdraw(by_label["periodic_k0"], min(args.steps, 12), arm))
    run_leg("byte_neutral",
            lambda: leg_byte_neutral(by_label["walls_all"], args.steps, arm))
    run_leg("mutation", lambda: leg_mutation(mutation_specs, args.steps, arm,
                                             args.subnormal_policy))
    run_leg("disarm", lambda: leg_disarm(by_label["walls_all"], args.steps, arm))
    run_leg("compiler", lambda: leg_compiler(args.subnormal_policy, at_install))
    run_leg("lift", lambda: leg_lift(
        Path(args.out).parent, args.lift_steps, args.lift_max_cells,
        args.lift_timeout, args.lift_resume,
        args.lift_only.split(",") if args.lift_only else None,
        args.subnormal_policy, bool(args.import_meep_for_host_policy),
        licence["verdict"]))

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
        "what_it_licenses": (
            "two CUDA kernels -- the plain and the folded variant of one weld -- "
            "measured byte-identical, per COMPLETE DRIVER STEP and as uint32 words "
            "over every stored volume, to four independent arrangements of the "
            "update_H -> step_D seam under complex64 storage, at every block size, on "
            "the fixtures and corpus rows this record names, under the installed "
            "float32 subnormal policy. It licenses NO throughput claim, NO dispatch "
            "claim and NO composition claim: the family declares INSTALLABLE = False, "
            "is not in the composer's tables on disk, and a credited seam-instance is "
            "PREDICATE ADMISSION"),
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
