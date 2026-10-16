"""Byte-identity gate for the COMPLEX x OFF-DIAGONAL ``update_E`` CUDA family.

THE QUESTION
============
``meep_gpu/cuda_kernels/complex_offdiag_update_e.py`` carries two kernels for
MEEP's tensor ROW PRODUCT under complex64 storage -- one with the split-field
tail, one with the direct store. Neither has ever run on a device. This gate asks,
per tail, whether each is byte-identical to ``stepping.update_E`` from ONE frozen
state -- at one launch and at 60 -- across mirror folds on every axis at both
declared terminations and both plane parities, Bloch phases on unfolded axes,
metallic walls, 3-D / 2-D / 1-D shapes, four row masks, three value classes, both
courants and BOTH float32 subnormal policies.

WHAT THE CORPUS ASKED FOR
=========================
The 2026-08-20 final census leaves 6 of 759 slots unserved, all at ``update_E``,
and FIVE are this one shape: three under an ACTIVE layer with a mirror fold
(``examples/solve-cw.py``, ``TestArrayMetadata.test_array_metadata``,
``TestHoleyWvgBands.test_fields_at_kx``) and two under an INERT one, unfolded,
with a nonzero ``k_point`` (``TestMaterialGrid.test_matgrid_3d``,
``TestMaterialGrid.test_subpixel_smoothing``). Every family that was asked refused
on KERNEL SHAPE: "complex64 storage" from the two real off-diagonal families,
"off-diagonal chi1inv" from the three complex ones.

THE TWO ARMS, AND WHY THEY ARE TWO KERNELS
==========================================
``update_E`` accumulates onto ``E`` through ``_apply_constitutive_pml`` under a
layer (stepping.py:1015) and OVERWRITES ``E`` without one (:1022). Binding
``kps = 1``, ``kms = 0`` does not collapse the first into the second -- it computes
``E_old + constitutive`` -- and ``update_E``'s input ``E`` is the field the previous
step wrote. ``meep_gpu/cuda_kernels/test_complex_offdiag_update_e.py`` measures
that on the array path (1024 of 1024 float32 words differ on an 8x8x8 fixture);
this gate carries it as a DEFECT LEG as well, so the split is measured on a device
and not only on a laptop.

WHAT WOULD MAKE THIS GATE VACUOUS, AND THE FLOORS THAT REFUSE IT
================================================================
* ``oracle_moved`` -- the array path must change at least one output word from the
  frozen input. A zero-initialised constitutive step leaves every word ``+0.0``
  forever, so a reference with deliberately swapped coefficients would still report
  IDENTICAL. Half of one earlier gate on this track could not fail for exactly
  that reason. Cases that moved nothing are SKIPPED, not passed.
* ``coupling_is_live`` -- the row product must differ from the bare diagonal
  somewhere, or no defect in this family can bite.
* ``coefficient_profile`` -- the three inverse-epsilon volumes and every live row
  volume must be NON-CONSTANT and pairwise distinct. Against a constant volume a
  coefficient-INDEX error is invisible; against one shared volume a
  component-aliasing defect is.
* ``mirror_ghost_is_live`` -- on a folded axis, stored row ``MIRROR_ROW`` of the
  partner volumes must not be identically zero, or the mirror ghost EQUALS the
  metallic zero it replaces and an exact result is a fact about the fixture.
* ``folded_axis_absorbs`` -- the folded axis's half-integer profile must beat the
  identity, or the case measures every axis except the one the fold moved.
* ``phase_is_live`` -- on a phased spec at least one axis must carry a factor that
  is not ``1+0j``, or every connection mutation is a null for a reason about the
  fixture.
* The mutation battery INCLUDING legs that must come back UNCAUGHT. A battery of
  only-must-be-caught legs scores identically whether the comparator works or has
  degenerated into failing everything. The three DEAD-PRELUDE nulls
  (``cshift_up``, ``cshift_dn``, ``pml_apply`` -- compiled into every kernel here
  and called by none) are the measurement that this family really does not route
  through the certified curl's ghost helpers or the split-field recurrence.
* ``verdict_flips_against_planted_defect`` -- the release verdict is recomputed
  against a record with a defect planted in it and the run FAILS if it does not
  flip. A gate whose verdict cannot go red is not a gate.

THERE IS NO NumPy BACKEND, DELIBERATELY
=======================================
``gate_cuda_complex_no_pml``'s decision and its reason: a NumPy backend costs a
SECOND TRANSCRIPTION of the kernel on the leg that is supposed to be checking the
first. What a laptop CAN settle is in
``meep_gpu/cuda_kernels/test_complex_offdiag_update_e.py``: the emitted text
against the certified templates it is built from, the emitted text EXECUTED in
float32 against ``stepping.update_E`` word for word, the chunk forms and presence
words on layered divisions, and the predicates' whole verdict table on real
engine objects.

RUNNING IT
==========
Device (the GPU host, ONE verified-empty GPU; the cache dir MUST carry the policy
token because CuPy's disk-cache key is computed above the strip seam)::

    CUDA_VISIBLE_DEVICES=$GPU CUPY_CACHE_DIR=$OUT/cupy_cache/keep \\
        python -u gate_cuda_complex_offdiag_update_e.py \\
        --subnormal-policy keep --out $OUT/keep/gate.json

One flushed line per case (the progress-reporting rule); the artifact is rewritten atomically
after every case, so an interrupted run keeps everything up to the failure.
Correctness only -- no throughput claim is made or possible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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
except ImportError:  # the laptop can still import this file and read its tables
    cp = None

import probe_fused_kernel_bit_identity as probe  # noqa: E402
import gate_provenance  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid, Mirror  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.cuda_kernels import coverage  # noqa: E402
from meep_gpu.cuda_kernels import folded_offdiag_kernels as folded  # noqa: E402
from meep_gpu.cuda_kernels import complex_offdiag_update_e as family  # noqa: E402

log = probe.log
bit_compare = probe.bit_compare
combine = probe.combine
to_host = probe.to_host
operand_census = probe.operand_census

SEED = 20260820

#: Consecutive launches in the multi-step leg, the budget every certified
#: hand-CUDA record is cut at. "Identical for N steps" is a claim about N.
MULTI_STEP_BUDGET = 60

#: The six arrays the split-field arm writes; the store arm writes the first three.
OUTPUTS_PML: Tuple[str, ...] = ("Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez")
OUTPUTS_STORE: Tuple[str, ...] = ("Ex", "Ey", "Ez")
COMPONENTS: Tuple[str, ...] = ("Ex", "Ey", "Ez")

#: 0.5 is exactly representable in float32 and 0.35 is not. ONLY THE SECOND can
#: distinguish a contracted expression from an uncontracted one, so the unguarded
#: control is scored at the inexact one. The courant also moves dt, which moves
#: every PML coefficient, so it is a real second draw of the tables.
COURANTS: Tuple[float, ...] = (0.5, 0.35)
INEXACT_COURANT = 0.35

#: ``signed_zero`` is here because this family's zero cross terms and its ghost
#: planes are the whole question, and ``uniform(-1, 1)`` provably draws no signed
#: zero.
VALUE_CLASSES: Tuple[str, ...] = ("uniform", "subnormal_band", "signed_zero")

GUARD_SETS: Tuple[Tuple[str, Tuple[str, ...], bool], ...] = (
    ("fmad_false", ("--fmad=false",), True),
    ("default_no_options", (), False),
)

#: The row masks, CHOSEN so a folded axis lands in each of its three roles. A
#: sweep over corpus masks alone would answer one third of the question and report
#: it as the whole.
ROW_MASK_SPECS: Tuple[Dict[str, Any], ...] = (
    {"label": "full_tensor", "mask": (1, 1, 1, 1, 1, 1),
     "why": "every axis is some live slot's partner: the fold and both phase "
            "directions are always reachable"},
    {"label": "corpus_pair", "mask": (1, 0, 0, 1, 0, 0),
     "why": "the mask two corpus rows drive -- slots (Ex,Ey) and (Ey,Ex), so "
            "partner axes X and Y are live and Z is in NEITHER role"},
    {"label": "row_Ex_only", "mask": (1, 1, 0, 0, 0, 0),
     "why": "one row component: X is its OWN axis and never its partner, so a "
            "folded X exercises the own-axis up shift alone"},
    {"label": "single_Ey_Ez", "mask": (0, 0, 1, 0, 0, 0),
     "why": "one slot: own axis Y, partner axis Z -- a folded X is in neither "
            "role, which is the fold discriminators' NULL arm"},
)

#: The grids. EVERY AXIS IS FOLDED SOMEWHERE, both declared terminations appear on
#: a fold, both plane parities appear, an odd stored count appears, a Bloch phase
#: appears BESIDE a fold and on its own, metallic walls appear beside a fold (which
#: no folded spec gets on its own, because a folded axis always reports ``wm = 0``),
#: one / two / three simultaneous planes appear, and 3-D / 2-D / 1-D shapes all do.
#: The two UNFOLDED PML specs carry the reduction claim; the store specs are the
#: corpus's own two shapes plus a metallic and a 1-D control.
SPECS: Tuple[Dict[str, Any], ...] = (
    # --- the split-field arm -------------------------------------------------
    {"label": "pml_unfolded_periodic", "arm": "pml", "cell": (8.0, 10.0, 12.0),
     "boundaries": ("periodic",) * 3, "symmetry": (), "k_point": (0.0, 0.0, 0.0)},
    {"label": "pml_unfolded_phased", "arm": "pml", "cell": (8.0, 10.0, 12.0),
     "boundaries": ("periodic",) * 3, "symmetry": (),
     "k_point": (0.2, -0.35, 0.1)},
    {"label": "pml_unfolded_walls", "arm": "pml", "cell": (9.0, 10.0, 11.0),
     "boundaries": ("metallic", "periodic", "metallic"), "symmetry": (),
     "k_point": (0.0, 0.2, 0.0)},
    {"label": "pml_fold_X_metallic", "arm": "pml", "cell": (16.0, 10.0, 12.0),
     "boundaries": ("metallic", "periodic", "periodic"),
     "symmetry": (("X", 1),), "k_point": (0.0, 0.0, 0.0)},
    {"label": "pml_fold_X_periodic", "arm": "pml", "cell": (16.0, 10.0, 12.0),
     "boundaries": ("periodic",) * 3, "symmetry": (("X", 1),),
     "k_point": (0.0, 0.0, 0.0)},
    {"label": "pml_fold_X_periodic_odd_plane", "arm": "pml",
     "cell": (16.0, 10.0, 12.0), "boundaries": ("periodic",) * 3,
     "symmetry": (("X", -1),), "k_point": (0.0, 0.0, 0.0)},
    {"label": "pml_fold_X_odd_count", "arm": "pml", "cell": (17.0, 10.0, 12.0),
     "boundaries": ("periodic",) * 3, "symmetry": (("X", 1),),
     "k_point": (0.0, 0.0, 0.0)},
    {"label": "pml_fold_Y_phased_X", "arm": "pml", "cell": (10.0, 16.0, 12.0),
     "boundaries": ("periodic",) * 3, "symmetry": (("Y", 1),),
     "k_point": (0.31, 0.0, 0.0)},
    {"label": "pml_fold_Y_metallic_odd_plane", "arm": "pml",
     "cell": (10.0, 16.0, 12.0),
     "boundaries": ("periodic", "metallic", "periodic"),
     "symmetry": (("Y", -1),), "k_point": (0.0, 0.0, 0.0)},
    {"label": "pml_fold_Z_metallic", "arm": "pml", "cell": (8.0, 12.0, 16.0),
     "boundaries": ("periodic", "periodic", "metallic"),
     "symmetry": (("Z", 1),), "k_point": (0.0, 0.0, 0.0)},
    {"label": "pml_fold_Z_periodic_odd_plane", "arm": "pml",
     "cell": (8.0, 12.0, 16.0), "boundaries": ("periodic",) * 3,
     "symmetry": (("Z", -1),), "k_point": (0.0, 0.0, 0.0)},
    # A FOLD BESIDE A LIVE METALLIC WALL, which no other folded spec carries: a
    # folded axis always reports wm = 0 (``is_metallic and not is_mirrored``), so a
    # sweep of folded specs alone leaves every wall flag zero and the wall mask
    # untested on a fold.
    {"label": "pml_fold_X_wall_Y", "arm": "pml", "cell": (16.0, 10.0, 12.0),
     "boundaries": ("periodic", "metallic", "periodic"),
     "symmetry": (("X", 1),), "k_point": (0.0, 0.0, 0.0)},
    {"label": "pml_fold_Z_wall_X", "arm": "pml", "cell": (8.0, 12.0, 16.0),
     "boundaries": ("metallic", "periodic", "periodic"),
     "symmetry": (("Z", 1),), "k_point": (0.0, 0.0, 0.0)},
    {"label": "pml_fold_XY_mixed", "arm": "pml", "cell": (16.0, 16.0, 10.0),
     "boundaries": ("metallic", "periodic", "periodic"),
     "symmetry": (("X", 1), ("Y", -1)), "k_point": (0.0, 0.0, 0.0)},
    {"label": "pml_fold_XZ_phased_Y", "arm": "pml", "cell": (16.0, 10.0, 16.0),
     "boundaries": ("periodic",) * 3, "symmetry": (("X", -1), ("Z", 1)),
     "k_point": (0.0, 0.27, 0.0)},
    # THE PRESENCE CASE (0.10). MEEP deletes a row's chi1inv array in every chunk where
    # it is identically zero over the samples the chunk allocates and drops its term
    # there, which differs from adding a zero only on a -0 row. ``row_block`` keeps the
    # rows on an interior block, so every chunk that carries a layer deletes them; the
    # signed-zero class plants the -0 words the dropped term leaves alone.
    {"label": "pml_presence_block", "arm": "pml", "cell": (14.0, 14.0, 14.0),
     "boundaries": ("metallic",) * 3, "symmetry": (), "k_point": (0.0, 0.0, 0.0),
     "row_block": True},
    {"label": "pml_fold_XYZ", "arm": "pml", "cell": (16.0, 16.0, 16.0),
     "boundaries": ("periodic",) * 3,
     "symmetry": (("X", 1), ("Y", 1), ("Z", 1)), "k_point": (0.0, 0.0, 0.0)},
    {"label": "pml_fold_2d_phased", "arm": "pml", "dimensions": 2,
     "cell": (16.0, 14.0, 0.0),
     "boundaries": ("metallic", "periodic", "periodic"),
     "symmetry": (("X", 1),), "k_point": (0.0, 0.27, 0.0)},
    # --- the direct-store arm ------------------------------------------------
    {"label": "store_3d_phased", "arm": "no_pml", "cell": (8.0, 10.0, 12.0),
     "boundaries": ("periodic",) * 3, "symmetry": (),
     "k_point": (0.23, -0.17, 0.35)},
    {"label": "store_3d_unphased", "arm": "no_pml", "cell": (8.0, 10.0, 12.0),
     "boundaries": ("periodic",) * 3, "symmetry": (), "k_point": (0.0, 0.0, 0.0)},
    {"label": "store_metallic_walls", "arm": "no_pml", "cell": (9.0, 10.0, 11.0),
     "boundaries": ("metallic", "metallic", "periodic"), "symmetry": (),
     "k_point": (0.0, 0.0, 0.4)},
    {"label": "store_2d_phased", "arm": "no_pml", "dimensions": 2,
     "cell": (12.0, 14.0, 0.0), "boundaries": ("periodic",) * 3, "symmetry": (),
     "k_point": (0.3892, 0.1597, 0.0)},
    {"label": "store_1d_phased", "arm": "no_pml", "dimensions": 1,
     "cell": (0.0, 0.0, 24.0), "boundaries": ("periodic",) * 3, "symmetry": (),
     "k_point": (0.0, 0.0, 0.35)},
    # THE INERT LAYER, swept because "inert is the same as absent" is otherwise an
    # inference about pml.py: ``_pml_is_active`` is False for a zero-thickness
    # layer, so stepping takes the store branch and the predicate admits it.
    {"label": "store_inert_layer_phased", "arm": "no_pml",
     "cell": (9.0, 10.0, 11.0), "boundaries": ("periodic",) * 3, "symmetry": (),
     "k_point": (0.3, 0.1, 0.0), "inert_layer": True},
)

#: The reduced product's specs, NAMED rather than sliced. A head slice of the list
#: is all X folds and no store arm at all.
REDUCED_SPEC_LABELS: Tuple[str, ...] = (
    "pml_unfolded_phased", "pml_fold_X_metallic", "pml_fold_X_periodic_odd_plane",
    "pml_fold_Y_phased_X", "pml_fold_Z_wall_X", "pml_fold_XYZ",
    "pml_presence_block", "store_3d_phased", "store_metallic_walls")


# ---------------------------------------------------------------------------
# The fixture
# ---------------------------------------------------------------------------

def _complex_host(shape, value_class: str, rng) -> np.ndarray:
    """One complex64 volume in a value class.

    THE TWO PLANES ARE ASSIGNED, NEVER COMBINED AS ``re + 1j*im``. Measured by the
    sibling tranche before it was reasoned about: ``1j * im`` carries a real part
    of ``0.0 * im``, so ``re + 1j*im`` computes ``(-0.0) + (+0.0) = +0.0`` and
    DESTROYS every negative zero in the plane the zero cross terms act on.
    """
    if value_class == "uniform":
        real = rng.uniform(-1.0, 1.0, size=shape)
        imag = rng.uniform(-1.0, 1.0, size=shape)
    elif value_class == "subnormal_band":
        real = probe.subnormal_band_hosts(("re",), tuple(shape), rng)["re"]
        imag = probe.subnormal_band_hosts(("im",), tuple(shape), rng)["im"]
    elif value_class == "signed_zero":
        values = rng.uniform(-1.0, 1.0, size=shape)
        zeroed = rng.integers(0, 2, size=shape) == 0
        signs = np.where(rng.integers(0, 2, size=shape) == 0, -0.0, 0.0)
        real = np.where(zeroed, signs, values)
        imag = rng.uniform(-1.0, 1.0, size=shape)
    else:
        raise ValueError(f"value class {value_class!r} is not one of {VALUE_CLASSES}")
    host = np.empty(tuple(shape), dtype=np.complex64)
    host.real = np.asarray(real, dtype=np.float32)
    host.imag = np.asarray(imag, dtype=np.float32)
    return np.ascontiguousarray(host)


#: Every complex volume a fixture owns on the split-field arm. The store arm owns
#: the first six.
STATE_PML: Tuple[str, ...] = ("Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
                              "f_w_Ex", "f_w_Ey", "f_w_Ez")
STATE_STORE: Tuple[str, ...] = ("Dx", "Dy", "Dz", "Ex", "Ey", "Ez")


def build(xp, spec: Dict[str, Any], mask: Sequence[int], courant: float,
          value_class: str, seed: int):
    """One complete ``(fields, layer, grid)`` fixture, built from ``seed`` alone.

    Called TWICE per case with the same seed, which is what makes the oracle
    fixture and the kernel fixture the same PROBLEM rather than two draws. The
    two are asserted byte-identical before either is stepped.

    THE THICKNESS RULE IS THE SIBLING FOLD GATES': skip an axis too thin to hold a
    layer, and on a MIRRORED axis ask for the HIGH face only.
    ``PML._resolve_mirror_faces`` refuses a named low face on a folded axis
    outright -- cell 0 is the mirror plane, a boundary condition rather than a wall
    -- and ``stepping._require_consistent_pml`` raises if the two disagree.
    """
    rng = np.random.default_rng(seed)
    active = spec["arm"] == "pml"
    grid = Grid(resolution=1.0, cell_size=tuple(spec["cell"]),
                boundaries=tuple(spec["boundaries"]),
                symmetry=tuple(Mirror(name, phase)
                               for name, phase in spec["symmetry"]),
                xp=xp, courant=courant, k_point=tuple(spec["k_point"]),
                dimensions=spec.get("dimensions", 3))
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_field_storage()
    if active:
        fields.enable_pml_storage()
    shape = tuple(grid.shape)

    # THREE DISTINCT, INHOMOGENEOUS inverse-epsilon volumes drawn away from 1.0.
    # Against ones the multiply is invisible; against one shared volume the
    # component-aliasing defect is invisible; against a constant volume a
    # coefficient-index error is invisible. ``coefficient_profile`` is the floor
    # that refuses a case where any of those could not bite.
    epsilon, inverse = {}, {}
    for component in COMPONENTS:
        values = rng.uniform(1.2, 3.4, size=shape).astype(np.float32)
        epsilon[component] = xp.asarray(np.ascontiguousarray(values))
        inverse[component] = xp.asarray(
            np.ascontiguousarray((np.float32(1.0) / values).astype(np.float32)))
    rows: Dict[str, Dict[str, Any]] = {}
    for slot, (row, partner) in enumerate(coverage.OFFDIAG_ROW_SLOTS):
        if not mask[slot]:
            continue
        # SPATIALLY VARYING AND STRADDLING ZERO. A uniform coefficient is what
        # makes the REGISTRATION rather than merely the rounding invisible, and the
        # inverse of a real tensor carries NEGATIVE off-diagonals -- a sign error
        # that only shows on one side of zero is exactly what a positive-only draw
        # hides.
        values = rng.uniform(-0.45, 0.45, size=shape).astype(np.float32)
        if spec.get("row_block"):
            # Zero outside the interior block [5, n - 5) of every live axis: the
            # chunks a layer owns (and the one sample past each) hold only zeros.
            block = np.zeros(shape, dtype=bool)
            block[tuple(slice(5, n - 5) if n > 10 else slice(None) for n in shape)] = True
            values = np.where(block, values, np.float32(0.0)).astype(np.float32)
        rows.setdefault(row, {})[partner] = xp.asarray(
            np.ascontiguousarray(values))
    fields.set_epsilon_volumes(epsilon, inverse, chi1inv_offdiagonal=rows)

    for name in (STATE_PML if active else STATE_STORE):
        getattr(fields, name)[...] = xp.asarray(
            _complex_host(shape, value_class, rng))

    layer = None
    if active:
        thickness = tuple(
            (0, 0) if grid.shape[axis] < 6
            else (0, 2) if grid.is_mirrored(axis)
            else (2, 2)
            for axis in range(3))
        layer = PML(grid=grid, thickness=thickness)
    elif spec.get("inert_layer"):
        layer = PML(grid=grid, thickness=((0, 0), (0, 0), (0, 0)))
    return fields, layer, grid


def outputs_for(arm: str) -> Tuple[str, ...]:
    return OUTPUTS_PML if arm == "pml" else OUTPUTS_STORE


def slot_hosts(fields, arm: str) -> Dict[str, np.ndarray]:
    return {name: to_host(getattr(fields, name)).copy()
            for name in outputs_for(arm)}


def fixtures_agree(left, right, arm: str) -> Dict[str, Any]:
    """The two fixtures of one case start byte-identical, or the case compares two
    different problems and a pass means nothing."""
    names = (STATE_PML if arm == "pml" else STATE_STORE)
    differing = 0
    for name in names:
        differing += int(np.count_nonzero(
            to_host(getattr(left, name)).view(np.float32).view(np.uint32)
            != to_host(getattr(right, name)).view(np.float32).view(np.uint32)))
    for component in COMPONENTS:
        differing += int(np.count_nonzero(
            to_host(left.inverse_epsilon_for(component)).view(np.uint32)
            != to_host(right.inverse_epsilon_for(component)).view(np.uint32)))
    for index, volume in enumerate(coverage.offdiag_row_volumes(left)):
        other = coverage.offdiag_row_volumes(right)[index]
        if volume is None:
            differing += int(other is not None)
            continue
        differing += int(np.count_nonzero(
            to_host(volume).view(np.uint32) != to_host(other).view(np.uint32)))
    return {"differing_words": differing, "meets_floor": differing == 0}


def advance_sources(fields, arm: str) -> None:
    """Keep the operands moving between launches of the multi-step leg.

    Held fixed, the sub-step reads the same numbers every launch and a 60-launch
    leg becomes a slow single-launch leg. The same exact float32 scale is applied
    on both fixtures, so the two stay the same problem.
    """
    scale = np.float32(0.97)
    for name in ("Dx", "Dy", "Dz"):
        getattr(fields, name)[...] = getattr(fields, name) * scale


# ---------------------------------------------------------------------------
# The non-vacuity floors
# ---------------------------------------------------------------------------

def _profile(volume) -> Dict[str, Any]:
    host = to_host(volume).astype(np.float64)
    return {"min": float(np.min(host)), "max": float(np.max(host)),
            "constant": bool(np.min(host) == np.max(host))}


def coefficient_profile(fields) -> Dict[str, Any]:
    """Every per-cell coefficient volume the case binds, and whether it can bite."""
    out: Dict[str, Any] = {"volumes": {}, "identity": [], "constant": [],
                           "shared": []}
    seen: Dict[int, str] = {}
    volumes: List[Tuple[str, Any]] = [
        (f"inv_eps_{c}", fields.inverse_epsilon_for(c)) for c in COMPONENTS]
    for index, volume in enumerate(coverage.offdiag_row_volumes(fields)):
        if volume is None:
            continue
        row, partner = coverage.OFFDIAG_ROW_SLOTS[index]
        volumes.append((f"chi1inv[{row}][{partner}]", volume))
    for label, volume in volumes:
        detail = _profile(volume)
        out["volumes"][label] = detail
        if detail["constant"]:
            out["constant"].append(label)
        if detail["min"] == 1.0 and detail["max"] == 1.0:
            out["identity"].append(label)
        address = coverage._base_address(volume)
        if address is not None:
            if address in seen:
                out["shared"].append(f"{label} is {seen[address]}")
            seen[address] = label
    out["meets_floor"] = not (out["constant"] or out["identity"] or out["shared"])
    return out


def folded_axis_absorbs(layer, grid) -> Dict[str, Any]:
    """Does the FOLDED axis's HALF-INTEGER profile beat the identity?

    ``kps = kms = 1`` is the interior pass-through. An axis whose whole vector is
    that identity cannot distinguish a coefficient-index error on it -- which is
    precisely the axis the fold moved.
    """
    out: Dict[str, Any] = {"folded_axes": [], "max_deviation": 0.0}
    if layer is None or not getattr(layer, "is_active", False):
        out["meets_floor"] = True
        return out
    tables = family.complex_offdiag_tables(layer)
    for axis, name in enumerate("xyz"):
        if not grid.is_mirrored(axis):
            continue
        deviation = 0.0
        for stem in ("kps", "kms"):
            values = to_host(tables[f"{stem}_{name}"]).astype(np.float64)
            deviation = max(deviation, float(np.max(np.abs(values - 1.0))))
        out["folded_axes"].append({"axis": axis, "name": name,
                                   "max_deviation_from_identity": deviation})
        out["max_deviation"] = max(out["max_deviation"], deviation)
    out["meets_floor"] = (not out["folded_axes"]) or out["max_deviation"] > 0.0
    return out


def mirror_ghost_is_live(fields, grid) -> Dict[str, Any]:
    """Is the mirror ghost distinguishable from the metallic zero it replaces?

    The ghost is ``parity * g[MIRROR_ROW]`` and ``MIRROR_SOURCE_INDEX`` is asked of
    ``stepping`` rather than spelled: a second spelling of the row the ghost
    reflects from is a second place for it to drift, and this floor's whole job is
    to be about that row.
    """
    row = int(stepping.MIRROR_SOURCE_INDEX)
    out: Dict[str, Any] = {"source_row": row, "axes": [], "min_max_abs": None}
    for axis in range(3):
        if not grid.is_mirrored(axis):
            continue
        for name in ("Dx", "Dy", "Dz"):
            volume = to_host(getattr(fields, name))
            key: List[Any] = [slice(None)] * 3
            key[axis] = row
            plane = volume[tuple(key)]
            value = float(np.max(np.abs(plane.astype(np.complex128))))
            out["axes"].append({"axis": axis, "volume": name, "max_abs": value})
            out["min_max_abs"] = (value if out["min_max_abs"] is None
                                  else min(out["min_max_abs"], value))
    out["meets_floor"] = (not out["axes"]) or (out["min_max_abs"] or 0.0) > 0.0
    return out


def phase_is_live(grid, flags) -> Dict[str, Any]:
    """At least one phased axis must carry a factor that is not ``1+0j``."""
    live = []
    for axis in range(3):
        if not flags[axis]:
            continue
        value = complex(np.complex64(grid.bloch_phase(axis)))
        live.append({"axis": axis, "real": value.real, "imag": value.imag,
                     "is_unity": value == complex(1.0, 0.0)})
    return {"axes": live, "any_phase": bool(live),
            "meets_floor": (not live) or any(not entry["is_unity"]
                                             for entry in live)}


def presence_drops_a_term(fields, record: Sequence[int], codes: Sequence[int],
                          kept: Sequence[int]) -> bool:
    """Whether some live row's cells read a chunk code its presence word clears."""
    from meep_gpu.cuda_kernels import offdiag_presence  # noqa: PLC0415
    from meep_gpu.fields import IYEE_SHIFTS  # noqa: PLC0415

    shape = tuple(int(n) for n in fields.grid.shape)
    for (row, _partner), word in zip(coverage.OFFDIAG_ROW_SLOTS, kept):
        if not word:
            continue
        read = np.unique(offdiag_presence.cell_chunk_codes(
            record, IYEE_SHIFTS[row], shape, codes))
        if any(not (int(word) >> int(code)) & 1 for code in read):
            return True
    return False


def wall_mask_bites(mask: Sequence[int], walls: Sequence[int]) -> bool:
    """Does the metallic wall mask change ANY word on this (mask, grid) pair?

    A LIVE WALL FLAG IS NOT ENOUGH. ``_mask_metallic_wall_coupling`` zeroes face 0
    of the axes on which THE COMPONENT'S Yee shift is 0, so ``wm_y`` reaches Ex and
    Ez and never reaches Ey. A grid that declares a wall on an axis no live row
    component masks against is one where dropping the mask is a genuine NULL.
    """
    for slot, (row, _partner) in enumerate(coverage.OFFDIAG_ROW_SLOTS):
        if not mask[slot]:
            continue
        component = COMPONENTS.index(row)
        if any(walls[axis] for axis in coverage.OFFDIAG_WALL_MASK_AXES[component]):
            return True
    return False


def fold_roles(grid, mask: Sequence[int]) -> Dict[str, Any]:
    """Which role each folded axis plays in this row mask -- the SHIPPED rule."""
    mirrored = tuple(bool(grid.is_mirrored(axis)) for axis in range(3))
    roles = dict(coverage.offdiag_fold_roles(tuple(mask), mirrored))
    for key, value in list(roles.items()):
        if isinstance(value, tuple):
            roles[key] = list(value)
    return roles


# ---------------------------------------------------------------------------
# THE DEFECTS
# ---------------------------------------------------------------------------
#
# Every one is a SILENT wrong answer -- a smooth, converged, plausible field with
# the wrong tensor in it -- and none would be caught by a magnitude comparison at
# any tolerance a physicist would accept.

def _sub(pattern: str, replacement, text: str) -> Tuple[str, int]:
    new, count = re.subn(pattern, replacement, text)
    return new, count


def _replace(needle: str, replacement: str, text: str) -> Tuple[str, int]:
    return text.replace(needle, replacement), text.count(needle)


# --- the mirror arm, stated wrongly in each of the ways the reading invites ----

def m_mirror_ghost_as_the_metallic_zero(text):
    """The certified flat kernel's only answer for a folded axis, planted."""
    return _replace("    if (bc == BC_MIRROR) return MIRROR_ROW;",
                    "    if (bc == BC_MIRROR) return -1;", text)


def m_mirror_ghost_as_the_periodic_wrap(text):
    return _replace("    if (bc == BC_MIRROR) return MIRROR_ROW;",
                    "    if (bc == BC_MIRROR) return n - 1;", text)


def m_mirror_row_off_by_one(text):
    return _replace("#define MIRROR_ROW 2", "#define MIRROR_ROW 1", text)


_PARITY_LINE = "    if (gl && bc == BC_MIRROR) return mul_coefficient_left(w, z);"


def m_flip_the_parity_sign(text):
    return _replace(_PARITY_LINE,
                    "    if (gl && bc == BC_MIRROR) return mul_coefficient_left("
                    "0.0f - w, z);", text)


def m_drop_the_parity_weight(text):
    return _replace(_PARITY_LINE, "    if (gl && bc == BC_MIRROR) return z;", text)


def m_parity_full_complex_product(text):
    """INVERTED (KERNELS.md section 7.1): the previous family's full complex product.

    The parity times the stored plane is one product per part
    (``part_products.scale``); the previous family spelled it as the complex product
    against ``(w + 0j)``, whose zero cross terms carry the other part's sign into an
    addend that is exactly zero -- byte-wrong on signed zeros at the multiply.
    """
    return _replace(
        _PARITY_LINE,
        "    if (gl && bc == BC_MIRROR) { cf o; o.re = (w * z.re) - (0.0f * z.im); "
        "o.im = (w * z.im) + (0.0f * z.re); return o; }", text)


def m_weight_every_lane(text):
    """The blanket weight the module refuses: an operation the oracle does not have."""
    return _replace(_PARITY_LINE,
                    "    if (bc == BC_MIRROR) return mul_coefficient_left(w, z);", text)

def m_folded_own_axis_wraps(text):
    """A folded axis wrapping in ``coord_up`` instead of serving the exact zero."""
    return _replace("    return (bc == BC_PERIODIC) ? 0 : -1;",
                    "    return (bc == BC_METALLIC) ? -1 : 0;", text)


# --- the connections, stated wrongly -------------------------------------------

_NEAR_TERM = ("    cf near_term = cconnect(mul_field_left(near_pair, u[home]), oa, oph, "
              "ophase);")
_FAR_TERM = "    cf far_term = mul_field_left(far_pair, ghosted(u, up));"


def m_conjugate_the_pair_connection(text):
    """The classic sign error: the pair's connection conjugated.

    The same-index operand takes MEEP's not-owned word ABOVE the window (eikna);
    conjugating it reads the word below. Every magnitude stays plausible and only the
    phase moves, which is what a band structure is made of. Planted on ALL THREE axes
    at the point where the kernel unpacks the table, so the leg bites whichever axis
    the scored case phases.
    """
    return _sub(r"(cf pp([xyz]); pp\2\.re = p\2r; pp\2\.im = )p\2i;",
                r"\1(0.0f - p\2i);", text)


def m_connect_the_field_not_the_product(text):
    """Connect the pre-coefficient pair instead of the formed near product.

    ``stepping._offdiagonal_terms`` connects the PRODUCT on the own axis
    (``_bloch_ghost_operand`` applied after the coefficient); connecting the pair and
    then scaling is a different float32 number.
    """
    return _replace(
        _NEAR_TERM,
        "    cf near_term = mul_field_left(cconnect(near_pair, oa, oph, ophase), u[home]);",
        text)


def m_drop_the_pair_connection(text):
    """The pair's same-index operand read as the stored word on plane 0 of a phased
    partner axis: MEEP's not-owned word served as the owned one."""
    return _replace("    return cwindow(g, index, pa, pph, pphase);",
                    "    return cf_load(g, index);", text)


def m_connect_every_plane(text):
    """The near product connected on every plane of the own axis, not plane 0 only.

    ``_bloch_ghost_operand`` replaces exactly plane 0 of a phased axis; a kernel that
    connected the volume would be smooth, converged and carrying a wrong dispersion
    relation.
    """
    return _replace(
        _NEAR_TERM,
        "    cf near_term = cconnect(mul_field_left(near_pair, u[home]), 0, oph, ophase);",
        text)


def m_old_package_ghost_phase(text):
    """INVERTED (KERNELS.md section 7.1): the previous package's ghost rule.

    The phase on the FAR product's wrapped lane (the top plane of the own axis, where
    the up index wraps below the home index) and none on the near product, where
    MEEP's stored window connects the near product on plane 0 and wraps the far one
    plainly.
    """
    return _replace(
        _NEAR_TERM + "\n" + _FAR_TERM,
        "    cf near_term = mul_field_left(near_pair, u[home]);\n" + _FAR_TERM + "\n"
        "    if (up >= 0 && up < home) far_term = cconnect(far_term, 0, oph, ophase);",
        text)


# --- the row product, stated wrongly ------------------------------------------

def m_hoist_the_coefficient(text):
    """The four-point average times ``u[i]`` -- the same ALGEBRA only for a uniform
    coefficient, and a different float32 number even then."""
    return _replace(_FAR_TERM,
                    "    cf far_term = mul_field_left(far_pair, u[home]);", text)


def m_ungosted_up_coefficient(text):
    """Read a coefficient where ``_shift_up`` writes an exact zero -- bit-identity 4."""
    return _replace("mul_field_left(far_pair, ghosted(u, up))",
                    "mul_field_left(far_pair, u[home])", text)


def m_same_direction_shifts(text):
    """Both shifts UP: a half-cell registration error, smooth and wrong."""
    return _sub(r"down_sample\(g, down,", "down_sample(g, up,", text)


def m_down_ghost_real_plane_only(text):
    """The METALLIC near ghost cleared on the real plane only -- a MEASURED NULL.

    complex64 is float2 and a clear that touches one plane is a known defect class
    in this tree, so this leg was written expecting a catch. The device says
    otherwise, and the reason is structural rather than a fixture accident: the
    partner axis of a component is ALWAYS one of that component's two wall-mask
    axes (``OFFDIAG_TRANSVERSE_PARTNERS[c]`` is the two axes that are not ``c``,
    and so is ``OFFDIAG_WALL_MASK_AXES[c]``), and a metallic axis is wall-masked
    unless it is mirrored (``is_metallic and not is_mirrored``,
    stepping.py:1282). So every cell at which ``coord_dn`` serves the metallic
    ghost is a cell at which ``_mask_metallic_wall_coupling`` then zeroes the whole
    coupling: the ghost's VALUE cannot reach a stored word.

    WHAT IS STILL LOAD-BEARING, and the legs that measure it: the MASK itself
    (``drop_the_wall_mask``, CAUGHT -- without it the face-0 coupling is
    ``g[home] * u``, and ``g[home]`` is alive beside the wall, which is the
    2.6e-02 the array path's own note records), and the ghosted UP coefficient
    (``ungosted_up_coefficient``, CAUGHT). The two ghost VALUES are belt to those
    braces, and this leg is the measurement that says so.
    """
    return _replace(
        "    if (index < 0) return cf_zero();\n    cf z = cf_load(g, index);",
        "    if (index < 0) { cf o = cf_load(g, 0); o.re = 0.0f; return o; }\n"
        "    cf z = cf_load(g, index);", text)


def m_up_ghost_real_plane_only(text):
    """The OWN-AXIS ghost cleared on the real plane only -- and it must be a NULL.

    NOT a slack leg: it is the MEASUREMENT of a structural fact this family relies
    on twice. Where ``coord_up`` returns -1 the coefficient read is ghosted to
    ``0.0f`` as well (bit-identity fact 4), so whatever ``up_sample`` returns is
    annihilated by ``mul_field_left(far_pair, ghosted(u, up))`` before it can reach the
    output. The two ghosts are therefore belt AND braces on this leg, and
    ``ungosted_up_coefficient`` -- which removes the braces -- is the leg that
    proves the belt alone is not enough (CAUGHT, thousands of words).
    """
    return _replace(
        "    if (index < 0) return cf_zero();\n"
        "    return cwindow(g, index, pa, pph, pphase);",
        "    if (index < 0) { cf o = cf_load(g, 0); o.re = 0.0f; return o; }\n"
        "    return cwindow(g, index, pa, pph, pphase);", text)


def m_couple_the_wrong_components(text):
    """Cycle the partner volumes: Dy where Dz belongs."""
    return _sub(r"        (D[xyz]), (chi1inv_\w+), idx,",
                lambda m: "        "
                          + {"Dx": "Dy", "Dy": "Dz", "Dz": "Dx"}[m.group(1)]
                          + f", {m.group(2)}, idx,", text)


def m_transpose_the_coefficient(text):
    """Read the coefficient one cell up its own axis -- the registration MEEP fixes."""
    return _replace("mul_field_left(near_pair, u[home])",
                    "mul_field_left(near_pair, ghosted(u, up))", text)


_ROW_ADD = (r"    if \(chunk_keeps\(kept_\w+, chunk_(E[xyz])\)\) \{\n"
            r"        row_\1_re = row_\1_re \+ \(0\.25 \* \(double\)\(term_\1_([01])\.re\)\);\n"
            r"        row_\1_im = row_\1_im \+ \(0\.25 \* \(double\)\(term_\1_\2\.im\)\);\n"
            r"    \}\n")


def m_drop_a_row_from_the_volume_sum(text):
    return _sub(_ROW_ADD, "", text)


def m_presence_ignored(text):
    """INVERTED (KERNELS.md section 7.1): every term added in every chunk.

    The previous family added both terms everywhere; MEEP drops a term in a chunk that
    deleted the array (``update_eh.cpp:191-196``), and the presence word says which.
    """
    return _sub(r"if \(chunk_keeps\(kept_\w+, chunk_E[xyz]\)\) \{", "if (1) {", text)


_WALL_TERM = r"\((wm_[xyz]) && (at_[xyz])\) \? cf_zero\(\) : (term_E[xyz]_[01]);"


def m_invert_the_wall_mask(text):
    return _sub(_WALL_TERM, r"(\1 && \2) ? \3 : cf_zero();", text)


def m_drop_the_wall_mask(text):
    return _sub(_WALL_TERM, r"\3;", text)


def m_wall_mask_real_plane_only(text):
    return _sub(
        r"    (term_E[xyz]_[01]) = \((wm_[xyz]) && (at_[xyz])\) \? cf_zero\(\) : \1;",
        r"    if (\2 && \3) \1.re = 0.0f;", text)


# --- the tails ----------------------------------------------------------------

def m_wrong_own_axis_table(text):
    """The dsig/dsigu cycle the CURL uses, where MEEP's dsigw belongs. A smooth,
    converged, entirely wrong absorber."""
    return _sub(r"kps_x\[i\], kms_x\[i\]", "kps_y[j], kms_y[j]", text)


_PREV_LINE = "    cf prev = dsigw ? cf_load(fw, idx) : cf_zero();"
_FW_STORE = "    if (dsigw) cf_store(fw, idx, src);"


def m_store_fw_before_reading_prev(text):
    """Wrong only where kms != 0, i.e. INSIDE the absorber -- which reads as a
    slightly weaker layer rather than as a bug."""
    return _replace(_PREV_LINE + "\n" + _FW_STORE, _FW_STORE + "\n" + _PREV_LINE, text)


def m_old_package_constitutive_grouping(text):
    """INVERTED (KERNELS.md section 7.1): two accumulations where MEEP's
    ``step_update_EDHB`` forms the right-hand side first, ``f + ((kps*src) -
    (kms*prev))`` -- a different float32 number on the cells whose chunk carries the
    dsigw layer."""
    return _replace(
        "cf_add(cf_load(f, idx), cf_sub(mul_coefficient_left(kps, src), "
        "mul_coefficient_left(kms, prev)))",
        "cf_sub(cf_add(cf_load(f, idx), mul_coefficient_left(kps, src)), "
        "mul_coefficient_left(kms, prev))", text)


# --- KERNELS.md section 7.2: the B battery on the constitutive chunk forms ------

def b_select_removed(text):
    """The whole-grid recurrence: every cell takes the dsigw form whatever its chunk
    carries -- the form the family stepped before the chunk forms."""
    return _replace(_PREV_LINE, "    dsigw = 1;\n" + _PREV_LINE, text)


def b_predicate_on_coefficient_identity(text):
    """The form chosen by the coefficient's value rather than by the chunk's
    membership, the predicate the family carried until 0.10."""
    return _sub(r"(constitutive_apply\(E[xyz], f_w_E[xyz], idx, src_E[xyz], "
                r"kps_([xyz])\[([ijk])\], kms_\2\[\3\]), in_h_\2\);",
                r"\1, ((kps_\2[\3] != 1.0f) || (kms_\2[\3] != 1.0f)));", text)


def b_auxiliary_stored_on_a_plain_cell(text):
    """The auxiliary written on every cell: the auxiliary contract leaves it untouched
    on a plain cell, and every auxiliary is compared."""
    return _replace(_FW_STORE, "    cf_store(fw, idx, src);", text)


def b_auxiliary_load_unguarded(text):
    """A DECLARED NULL: the auxiliary load issued on every cell, the store still
    guarded. The loaded word feeds only the dsigw form, so no stored word changes."""
    return _replace(_PREV_LINE, "    cf prev = cf_load(fw, idx);", text)


def m_store_arm_accumulates(text):
    """The store arm made into a recurrence -- the two-kernel decision, planted."""
    return _sub(r"    cf_store\((E[xyz]), idx, (src_E[xyz])\);",
                r"    cf_store(\1, idx, cf_add(cf_load(\1, idx), \2));", text)


# --- the must-be-UNCAUGHT nulls ----------------------------------------------

def n_commute_the_row_sum(text):
    """Addition is bitwise COMMUTATIVE, so swapping the two addends of the row
    accumulation must move NO word. A leg that caught it would be reporting a
    comparator that fails everything."""
    return _sub(r"row_(E[xyz])_(re|im) = row_\1_\2 \+ \(0\.25 \* \(double\)"
                r"\(term_\1_([01])\.\2\)\);",
                r"row_\1_\2 = (0.25 * (double)(term_\1_\3.\2)) + row_\1_\2;", text)


def n_commute_the_pair_add(text):
    """Same fact on the two-point partner pair."""
    return _replace(
        "    cf near_pair = cf_add(pair_window(g, home, pa, pph, pphase),\n"
        "                          down_sample(g, down, dgl, dbc, w));",
        "    cf near_pair = cf_add(down_sample(g, down, dgl, dbc, w),\n"
        "                          pair_window(g, home, pa, pph, pphase));", text)


def n_dead_cshift_up(text):
    """``cshift_up`` is compiled into every kernel here and called by none.

    Its silence is the MEASUREMENT that this family does not route through the
    certified curl's ghost helper, which is why the dead prelude is carried whole
    rather than forked.
    """
    return _replace("    if (ia + 1 < na) return cf_load(g, idx + stride);",
                    "    if (ia + 1 < na) return cf_zero();", text)


def n_dead_cshift_dn(text):
    return _replace("    if (ia > 0) return cf_load(g, idx - stride);",
                    "    if (ia > 0) return cf_zero();", text)


def n_dead_pml_apply(text):
    """``pml_apply`` is the SPLIT-FIELD CURL recurrence, dead in both arms here."""
    return _replace(
        "    cf fu_new = mul_field_left(cf_sub(mul_field_left(fprev, kms), curl), sinv);",
        "    cf fu_new = mul_field_left(cf_sub(curl, mul_field_left(fprev, kms)), sinv);",
        text)


SOURCE_MUTATIONS: Dict[str, Callable[[str], Tuple[str, int]]] = {
    "mirror_ghost_as_the_metallic_zero": m_mirror_ghost_as_the_metallic_zero,
    "mirror_ghost_as_the_periodic_wrap": m_mirror_ghost_as_the_periodic_wrap,
    "mirror_row_off_by_one": m_mirror_row_off_by_one,
    "flip_the_parity_sign": m_flip_the_parity_sign,
    "drop_the_parity_weight": m_drop_the_parity_weight,
    "parity_full_complex_product": m_parity_full_complex_product,
    "weight_every_lane": m_weight_every_lane,
    "folded_own_axis_wraps": m_folded_own_axis_wraps,
    "conjugate_the_pair_connection": m_conjugate_the_pair_connection,
    "connect_the_field_not_the_product": m_connect_the_field_not_the_product,
    "drop_the_pair_connection": m_drop_the_pair_connection,
    "connect_every_plane": m_connect_every_plane,
    "old_package_ghost_phase": m_old_package_ghost_phase,
    "hoist_the_coefficient": m_hoist_the_coefficient,
    "ungosted_up_coefficient": m_ungosted_up_coefficient,
    "same_direction_shifts": m_same_direction_shifts,
    "down_ghost_real_plane_only": m_down_ghost_real_plane_only,
    "up_ghost_real_plane_only": m_up_ghost_real_plane_only,
    "couple_the_wrong_components": m_couple_the_wrong_components,
    "transpose_the_coefficient": m_transpose_the_coefficient,
    "drop_a_row_from_the_volume_sum": m_drop_a_row_from_the_volume_sum,
    "presence_ignored": m_presence_ignored,
    "invert_the_wall_mask": m_invert_the_wall_mask,
    "drop_the_wall_mask": m_drop_the_wall_mask,
    "wall_mask_real_plane_only": m_wall_mask_real_plane_only,
    "wrong_own_axis_table": m_wrong_own_axis_table,
    "store_fw_before_reading_prev": m_store_fw_before_reading_prev,
    "old_package_constitutive_grouping": m_old_package_constitutive_grouping,
    "store_arm_accumulates": m_store_arm_accumulates,
    "select_removed": b_select_removed,
    "predicate_on_coefficient_identity": b_predicate_on_coefficient_identity,
    "auxiliary_stored_on_a_plain_cell": b_auxiliary_stored_on_a_plain_cell,
    "auxiliary_load_unguarded": b_auxiliary_load_unguarded,
    "commute_the_row_sum": n_commute_the_row_sum,
    "commute_the_pair_add": n_commute_the_pair_add,
    "dead_cshift_up": n_dead_cshift_up,
    "dead_cshift_dn": n_dead_cshift_dn,
    "dead_pml_apply": n_dead_pml_apply,
}

#: MUST come back UNCAUGHT. Two are bitwise-commutative rewrites of a real line;
#: three are rewrites of prelude helpers this family compiles and never calls; two
#: are MEASURED FINDINGS this gate's first device run established, each recorded
#: here with the number behind it rather than quietly dropped from the battery.
#:
#: ``parity_full_complex_product`` LEFT THIS LIST (bit_parity_0_10 lane KC6): the
#: previous family's full complex spelling of the mirror parity multiply, armed as
#: the mutant, was a null on the previous row, whose additions with a normal operand
#: absorbed the sign of a zero in the ghost. On MEEP's row -- float32 pair sums,
#: scaled and summed in double, a term absent where its chunk deleted the array --
#: that sign reaches a stored word: the first device run of the re-derived family
#: CAUGHT it on the signed-zero class (1 differing word, reduced product, keep
#: policy). It is must-catch, as every inverted spelling is (KERNELS.md section 7.1).
#:
#: ``auxiliary_load_unguarded`` -- KERNELS.md section 7.2's declared null: the load
#: feeds only the dsigw form, so no stored word changes.
#:
#: ``up_ghost_real_plane_only`` and ``down_ghost_real_plane_only`` -- see each
#: mutation's own docstring. Both were written expecting a catch and both are NULL
#: for a STRUCTURAL reason the device established: the own-axis ghost is annihilated
#: by the ghosted coefficient, and the partner-axis metallic ghost sits exactly
#: under the wall mask. The legs that remove those two structures --
#: ``ungosted_up_coefficient`` and ``drop_the_wall_mask`` -- are CAUGHT, which is
#: what turns a pair of nulls into a measurement instead of two blind legs.
NULL_MUTATIONS: Tuple[str, ...] = (
    "commute_the_row_sum", "commute_the_pair_add",
    "dead_cshift_up", "dead_cshift_dn", "dead_pml_apply",
    "up_ghost_real_plane_only",
    "down_ghost_real_plane_only", "auxiliary_load_unguarded")

#: Legs that can only bite where a fold reaches a LIVE PARTNER axis -- the only arm
#: on which the mirror ghost is read at all. Scored on a case whose ``roles`` say
#: so, and carried as a NULL on a case whose roles say it cannot.
FOLD_PARTNER_MUTATIONS: Tuple[str, ...] = (
    "mirror_ghost_as_the_metallic_zero", "mirror_ghost_as_the_periodic_wrap",
    "mirror_row_off_by_one", "flip_the_parity_sign", "drop_the_parity_weight",
    "parity_full_complex_product", "weight_every_lane")

#: Legs that need a fold on a live OWN axis.
FOLD_OWN_MUTATIONS: Tuple[str, ...] = ("folded_own_axis_wraps",)

#: Legs whose defect sits on the PAIR's connection (the PARTNER axis's window), so the
#: case needs a live partner axis that is PHASED. A phased axis that no live row slot
#: takes as its partner is a genuine NULL and scoring it a miss would be a statement
#: about the fixture rather than about the kernel -- the DEAD-BRANCH TRAP, per leg.
DOWN_PHASE_MUTATIONS: Tuple[str, ...] = (
    "conjugate_the_pair_connection", "drop_the_pair_connection")

#: Legs whose defect sits on the NEAR PRODUCT's connection (the OWN axis's window), so
#: the case needs a live own axis that is PHASED.
UP_PHASE_MUTATIONS: Tuple[str, ...] = (
    "connect_the_field_not_the_product", "connect_every_plane",
    "old_package_ghost_phase")

#: Legs that need the wall mask to BITE on this (mask, grid) pair.
WALL_MUTATIONS: Tuple[str, ...] = (
    "invert_the_wall_mask", "drop_the_wall_mask", "wall_mask_real_plane_only")

#: Legs whose defect lives in the SIGN OF A ZERO, so the signed-zero value class is
#: the one that can show it. Every other leg prefers the uniform class, where a
#: fixture is all normal words and an ordinary defect has the most room.
SIGNED_ZERO_MUTATIONS: Tuple[str, ...] = ("parity_full_complex_product",
                                          "presence_ignored")

#: Legs whose defect reaches a stored word on SOME armable cases and not others: the
#: sign of a zero in the mirror ghost survives MEEP's row only where the rest of the
#: row is itself a zero of the other sign. Scored over the armable cases in order,
#: and CAUGHT on the first that shows it; a leg no armable case catches has escaped.
LIVE_ON_SOME_CASE_MUTATIONS: Tuple[str, ...] = ("parity_full_complex_product",)

#: Legs that exist only on one arm.
PML_ONLY_MUTATIONS: Tuple[str, ...] = (
    "wrong_own_axis_table", "store_fw_before_reading_prev",
    "old_package_constitutive_grouping", "select_removed",
    "predicate_on_coefficient_identity", "auxiliary_stored_on_a_plain_cell",
    "auxiliary_load_unguarded")
STORE_ONLY_MUTATIONS: Tuple[str, ...] = ("store_arm_accumulates",)

#: Legs that need TWO live slots on one component to have anything to drop.
TWO_SLOT_MUTATIONS: Tuple[str, ...] = ("drop_a_row_from_the_volume_sum",
                                       "commute_the_row_sum")

#: HOST mutations: the launch arguments, corrupted. Each is a deliberately wrong
#: answer a launcher could be handed by a caller that skipped the predicate.
HOST_MUTATIONS: Tuple[str, ...] = (
    "swap_sub_lattice", "hand_the_fold_the_metallic_code", "flip_the_parity",
    "drop_the_wall_flags", "swap_the_phase_directions", "clear_the_phase_flags",
    "transpose_the_boundary_codes", "clear_the_presence_words",
    "record_bound_off_by_one", "record_lattices_swapped")

#: The host mutations that corrupt the chunk record the launch binds (KERNELS.md
#: section 7.2); the oracle steps the run's own division.
RECORD_HOST_MUTATIONS: Tuple[str, ...] = ("record_bound_off_by_one",
                                          "record_lattices_swapped")


def mutated_record(name: str, record: Sequence[int], grid) -> np.ndarray:
    """``record`` corrupted as KERNELS.md section 7.2 names.

    ``record_bound_off_by_one`` moves each layered axis's low run end one row up on
    both lattices (the boundary plane changes form); ``record_lattices_swapped`` hands
    each lattice the other's runs (the boundary plane moves by one cell).
    """
    from meep_gpu.chunk_map import record_index  # noqa: PLC0415

    source = np.asarray(record, dtype=np.int32)
    out = source.copy()
    for axis in range(3):
        if int(grid.shape[axis]) <= 1:
            continue
        for half in (False, True):
            a0 = int(source[record_index(half, axis, 0)])
            a1 = int(source[record_index(half, axis, 1)])
            if name == "record_bound_off_by_one" and a1 > a0:
                out[record_index(half, axis, 1)] = a1 + 1
        if name == "record_lattices_swapped":
            for word in range(4):
                low = record_index(False, axis, word)
                high = record_index(True, axis, word)
                out[low], out[high] = source[high], source[low]
    return out

#: HOST mutations that MUST come back UNCAUGHT on a case they cannot reach.
NULL_HOST_MUTATIONS: Tuple[str, ...] = ("rebind_identical_tables",)


def apply_host_mutation(name: Optional[str], layer, grid, arm, tables, codes,
                        walls, weights, connections, record, kept):
    """Return the launch arguments this leg feeds, mutated or not.

    ``connections`` is ``bloch_connection_table``'s four-tuple (pair kinds, pair
    parts, own kinds, own parts), ``record`` the host chunk record and ``kept`` the
    six presence words; the record is uploaded here.
    """
    if name is None or name == "rebind_identical_tables":
        if name == "rebind_identical_tables" and tables is not None:
            tables = {key: value for key, value in tables.items()}
    elif name == "swap_sub_lattice":
        # The INTEGER pair -- the H side's sub-lattice, wrong for E by half a cell.
        from meep_gpu.cuda_kernels.constitutive_kernels import (  # noqa: PLC0415
            real_constitutive_tables)
        tables = real_constitutive_tables(layer, False)
    elif name == "hand_the_fold_the_metallic_code":
        codes = tuple(coverage.BC_CODES["metallic"]
                      if code == family.BC_MIRROR_CODE else code
                      for code in codes)
    elif name == "flip_the_parity":
        weights = tuple(-value for value in weights)
    elif name == "drop_the_wall_flags":
        walls = (0, 0, 0)
    elif name == "swap_the_phase_directions":
        # Each operand handed the OTHER window's connection: the pair the word below,
        # the near product the word above.
        pair_kinds, pair_parts, own_kinds, own_parts = connections
        connections = (own_kinds, own_parts, pair_kinds, pair_parts)
    elif name == "clear_the_phase_flags":
        pair_kinds, pair_parts, own_kinds, own_parts = connections
        connections = ((0, 0, 0), pair_parts, (0, 0, 0), own_parts)
    elif name == "transpose_the_boundary_codes":
        codes = (codes[1], codes[2], codes[0])
    elif name == "clear_the_presence_words":
        kept = (0,) * len(kept)
    elif name in RECORD_HOST_MUTATIONS:
        record = mutated_record(name, record, grid)
    else:  # pragma: no cover - the table above is the whole set
        raise ValueError(f"no host mutation {name!r}")
    chunk_bounds = cp.asarray(np.asarray(record, dtype=np.int32), dtype=cp.int32)
    return tables, codes, walls, weights, connections, chunk_bounds, tuple(kept)

# ---------------------------------------------------------------------------
# The kernel backend
# ---------------------------------------------------------------------------

class KernelBackend:
    """The shipped CuPy launcher, driven through its own gate door.

    THE GUARD AND THE MUTATION BOTH REACH NVRTC THROUGH THE SOURCE-KEYED MEMO:
    ``_get_kernel`` keys on ``(name, options, policy, source)``, so overriding
    ``_COMPILE_OPTIONS`` or installing a source override is a MISS and the compiler
    sees the bytes this leg intends. The memo is dropped at both ends anyway,
    because a leg that measured a guard it never applied is a failure family this
    track has three recorded instances of.
    """

    name = "cupy"
    certifies = True

    def __init__(self):
        self.source_transform: Optional[Callable[[str], Tuple[str, int]]] = None
        self.sources_used: Dict[str, str] = {}
        self._overridden: List[Tuple[str, Tuple[int, ...], Any]] = []

    def set_guard(self, guard: Sequence[str]) -> None:
        """Install the compile options, and DROP THE MEMO ONLY WHEN THEY MOVE.

        The memo key already carries the option tuple, so a stale entry cannot be
        served to a different guard. Clearing on every case instead would recompile
        1104 times per policy and measure nothing extra -- and the cost is not
        free: a gate that spends its budget on NVRTC runs fewer cases.
        """
        wanted = tuple(guard)
        if family._COMPILE_OPTIONS == wanted:
            return
        family._COMPILE_OPTIONS = wanted
        family.clear_kernel_cache()

    def set_source_mutation(self, transform) -> None:
        self.clear()
        self.source_transform = transform

    def pristine_source(self, arm, mask) -> str:
        """The UNMUTATED source. Site counting and digesting must never go through
        an installed override: that would apply the transform twice and the leg
        would hash a body it never compiled."""
        return family.complex_offdiag_source(arm, mask)

    def source_for(self, arm, mask) -> Tuple[str, int]:
        text = self.pristine_source(arm, mask)
        if self.source_transform is None:
            return text, 0
        return self.source_transform(text)

    def launch(self, fields, arm, mask, tables, codes, walls, weights,
               connections, chunk_bounds, kept) -> None:
        text, _sites = self.source_for(arm, mask)
        key = (arm, tuple(mask))
        if self.source_transform is not None:
            family.set_kernel_source(arm, mask, text)
            if key not in self._overridden:
                self._overridden.append(key)
        self.sources_used[str(key)] = hashlib.sha256(
            text.encode("utf-8")).hexdigest()
        if arm == "pml":
            family.update_E_complex_offdiag_fused_pml(
                fields, tables=tables, codes=codes, walls=walls,
                weights=weights, connections=connections,
                chunk_bounds=chunk_bounds, kept=kept)
        else:
            family.update_E_complex_no_pml_offdiag(
                fields, codes=codes, walls=walls, weights=weights,
                connections=connections, chunk_bounds=chunk_bounds, kept=kept)
        cp.cuda.runtime.deviceSynchronize()

    def compile_log(self) -> List[Dict[str, Any]]:
        from meep_gpu.cuda_kernels import compile_cache  # noqa: PLC0415

        return list(compile_cache.compile_log())

    def clear(self) -> int:
        for arm, mask in self._overridden:
            family.set_kernel_source(arm, mask, None)
        self._overridden = []
        self.source_transform = None
        return family.clear_kernel_cache()


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------

def case_key(spec, mask_spec, courant, value_class, guard, steps) -> str:
    return "|".join([spec["label"], mask_spec["label"], f"C{courant}",
                     value_class, guard, f"n{steps}"])


def case_seed(key: str) -> int:
    """A STABLE PER-CASE SEED, from a DIGEST rather than from ``hash()``.

    Python salts ``hash()`` of a str with PYTHONHASHSEED -- measured, one key gave
    four distinct values in four interpreters -- so a hash-seeded gate draws a
    different fixture every process and a failing case cannot be replayed.
    """
    return (SEED + int.from_bytes(
        hashlib.sha256(key.encode("ascii")).digest()[:4], "big")) % (2 ** 32)


def one_case(backend, xp, spec: Dict[str, Any], mask_spec: Dict[str, Any],
             courant: float, value_class: str, guard_label: str,
             guard: Sequence[str], steps: int,
             source_mutation: Optional[str] = None,
             host_mutation: Optional[str] = None) -> Dict[str, Any]:
    """One frozen state, run twice: the array path, then the kernel.

    TWO FIXTURES, NOT ONE FIXTURE RESTORED. Both are built from the same seed and
    asserted byte-identical; the oracle runs on one and the kernel on the other.
    That is the ADE gate's shape and it removes the whole class of failure where a
    restore reproduced values but not the object graph.
    """
    started = time.time()
    key = case_key(spec, mask_spec, courant, value_class, guard_label, steps)
    seed = case_seed(key)
    arm = spec["arm"]
    mask = tuple(mask_spec["mask"])

    oracle_fields, oracle_layer, oracle_grid = build(
        xp, spec, mask, courant, value_class, seed)
    kernel_fields, kernel_layer, grid = build(
        xp, spec, mask, courant, value_class, seed)

    codes = family.complex_offdiag_boundary_codes(grid)
    walls = coverage.offdiag_wall_mask_flags(grid)
    weights = family.mirror_ghost_weights(grid)
    connections = family.bloch_connection_table(grid)
    flags = connections[0]
    # THE RUN'S CHUNK RECORD AND PRESENCE WORDS, from the real family's builder
    # (offdiag_presence.launch_chunk_arguments through chunk_arguments): the division
    # the array path's update_E reads on this layer.
    record, kept = family.chunk_arguments(kernel_fields, kernel_layer, codes)
    tables = (family.complex_offdiag_tables(kernel_layer) if arm == "pml"
              else None)

    case: Dict[str, Any] = {
        "key": key, "seed": seed, "label": spec["label"], "arm": arm,
        "row_mask_label": mask_spec["label"], "row_mask": list(mask),
        "installed_row_mask": list(coverage.offdiag_row_mask(kernel_fields)),
        "courant": courant, "value_class": value_class, "guard": guard_label,
        "steps": steps,
        "source_mutation": source_mutation, "host_mutation": host_mutation,
        "shape": [int(n) for n in grid.shape],
        "boundaries": list(spec["boundaries"]),
        "symmetry": [list(plane) for plane in spec["symmetry"]],
        "k_point": list(spec["k_point"]),
        "mirrored": [bool(grid.is_mirrored(a)) for a in range(3)],
        "metallic": [bool(grid.is_metallic(a)) for a in range(3)],
        "stored_past_owned": [int(grid.stored_cells(a)) - int(grid.owned_cells(a))
                              for a in range(3)],
        "boundary_kinds": list(coverage.real_pml_boundary_kinds(grid)),
        "codes": list(codes), "walls": list(walls),
        "ghost_weights": [float(value) for value in weights],
        "phase_flags": list(flags),
        "own_kinds": list(connections[2]),
        "pair_parts": [float(value) for value in connections[1]],
        "own_parts": [float(value) for value in connections[3]],
        "chunk_record": [int(word) for word in record],
        "presence_words": [int(word) for word in kept],
        "layered": any(int(word) for word in list(record)[:24]),
        "presence_drops_a_term": presence_drops_a_term(kernel_fields, record, codes,
                                                       kept),
        "roles": fold_roles(grid, mask),
        "wall_mask_bites": wall_mask_bites(mask, walls),
        "row_block": bool(spec.get("row_block")),
    }
    case["fold_reaches_the_partner_arm"] = bool(
        case["roles"]["fold_is_a_live_partner_axis"])
    case["fold_reaches_the_own_arm"] = bool(
        case["roles"]["fold_is_a_live_own_axis"])

    # THE CONNECTION TABLES, CROSS-CHECKED against the certified complex curl
    # launcher's own spelling: the pair takes the D curl's (eikna), the near product
    # the B curl's (conj(eikna)). Two spellings of one fact are equal only until
    # someone edits one, and the merge bar cannot check this pair because that module
    # imports cupy at scope.
    from meep_gpu.cuda_kernels import complex_pml_kernels  # noqa: PLC0415

    their_pair = complex_pml_kernels.bloch_phase_arguments(grid, True)
    their_own = complex_pml_kernels.bloch_phase_arguments(grid, False)
    case["phase_table_agrees_with_complex_pml_kernels"] = bool(
        tuple(int(f) for f in their_pair[0]) == tuple(connections[0])
        and tuple(float(v) for v in their_pair[1]) == tuple(connections[1])
        and tuple(int(f) for f in their_own[0]) == tuple(connections[2])
        and tuple(float(v) for v in their_own[1]) == tuple(connections[3]))

    # THE PREDICATES' ANSWERS, recorded rather than acted on.
    mine_pml = family.covers_complex_offdiag_pml_update_e(
        kernel_fields, kernel_layer, grid)
    mine_store = family.covers_complex_no_pml_offdiag_update_e(
        kernel_fields, kernel_layer, grid)
    theirs = coverage.covers_real_pml_complex_constitutive(
        kernel_fields, kernel_layer, grid, "E")
    folded_verdict = folded.covers_folded_offdiag_composition(
        kernel_fields, kernel_layer, grid)
    case["predicates"] = {
        "pml_arm": {"covered": bool(mine_pml[0]), "reason": mine_pml[1]},
        "store_arm": {"covered": bool(mine_store[0]), "reason": mine_store[1]},
        "complex_constitutive_E": {"covered": bool(theirs[0]),
                                   "reason": theirs[1]},
        "folded_offdiag": {"covered": bool(folded_verdict[0]),
                           "reason": folded_verdict[1]},
        "arms_disjoint": not (bool(mine_pml[0]) and bool(mine_store[0])),
        "expected_arm_admits": bool(
            mine_pml[0] if arm == "pml" else mine_store[0]),
        "disjoint_from_siblings": not (bool(theirs[0]) or bool(folded_verdict[0])),
    }

    agree = fixtures_agree(oracle_fields, kernel_fields, arm)
    case["fixtures_agree"] = agree
    case["coefficient_profile"] = coefficient_profile(kernel_fields)
    case["folded_axis_absorbs"] = folded_axis_absorbs(kernel_layer, grid)
    case["mirror_ghost_is_live"] = mirror_ghost_is_live(kernel_fields, grid)
    case["phase_is_live"] = phase_is_live(grid, flags)
    case["operand_census"] = operand_census(
        {name: to_host(getattr(kernel_fields, name)).view(np.float32)
         for name in ("Dx", "Dy", "Dz")})

    for floor in ("fixtures_agree", "coefficient_profile", "folded_axis_absorbs",
                  "mirror_ghost_is_live", "phase_is_live"):
        if not case[floor]["meets_floor"]:
            case["skipped"] = f"{floor} is below its floor: {case[floor]}"
            case["seconds"] = round(time.time() - started, 3)
            return case

    frozen = slot_hosts(oracle_fields, arm)

    # --- leg 1: the oracle -------------------------------------------------
    for _ in range(steps):
        stepping.update_E(oracle_fields, oracle_layer)
        advance_sources(oracle_fields, arm)
    oracle = slot_hosts(oracle_fields, arm)

    moved = sum(int(np.count_nonzero(
        oracle[name].view(np.float32).view(np.uint32)
        != frozen[name].view(np.float32).view(np.uint32)))
        for name in outputs_for(arm))
    case["oracle_moved_words"] = moved
    case["oracle_total_words"] = sum(
        int(oracle[name].view(np.float32).size) for name in outputs_for(arm))
    case["oracle_moved"] = moved > 0

    # THE COUPLING MUST BE LIVE: the row product has to differ from the bare
    # diagonal somewhere, or no defect in THIS family can bite.
    constitutive = (oracle["f_w_Ex"] if arm == "pml" else oracle["Ex"])
    diagonal = (to_host(oracle_fields.Dx).astype(np.complex128)
                * to_host(oracle_fields.inverse_epsilon_for("Ex")).astype(
                    np.float64))
    case["coupling_max_abs"] = float(np.max(np.abs(
        constitutive.astype(np.complex128) - diagonal)))
    case["coupling_is_live"] = case["coupling_max_abs"] > 0.0

    # --- leg 2: the kernel, on the SECOND fixture ---------------------------
    backend.set_guard(guard)
    launch = apply_host_mutation(host_mutation, kernel_layer, grid, arm, tables,
                                 codes, walls, weights, connections, record, kept)
    launch_error = None
    try:
        for _ in range(steps):
            backend.launch(kernel_fields, arm, mask, *launch)
            advance_sources(kernel_fields, arm)
    except Exception as exc:  # noqa: BLE001 - a refusal is a result, recorded
        launch_error = f"{type(exc).__name__}: {exc}"[:600]
    case["launch_error"] = launch_error

    if launch_error is None:
        parts = {name: bit_compare(oracle[name], getattr(kernel_fields, name))
                 for name in outputs_for(arm)}
        verdict = combine(parts)
        case["differing_words"] = int(verdict["differing_floats"])
        case["per_component"] = {
            name: int(part["differing_floats"]) for name, part in parts.items()}
    else:
        verdict = {"bit_identical": False, "differing_floats": 0,
                   "total_floats": 0}
        case["differing_words"] = 0
        case["per_component"] = {}
    case["bit_identical"] = bool(verdict["bit_identical"]) and launch_error is None
    case["total_floats"] = verdict["total_floats"]
    case["max_ulp"] = verdict.get("max_ulp")
    case["seconds"] = round(time.time() - started, 3)
    return case


def case_is_valid(case: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """A case that could not distinguish anything is REFUSED, not passed."""
    if case.get("skipped"):
        return False, case["skipped"]
    if not case.get("oracle_moved"):
        return False, ("the array path changed no output word: a zero-initialised "
                       "constitutive step is a fixed point and a case that moved "
                       "nothing certifies nothing")
    if not case.get("coupling_is_live"):
        return False, ("the off-diagonal coupling is identically zero: this case "
                       "cannot distinguish any defect in this family")
    return True, None


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

def case_product(product: str, steps: int) -> List[Dict[str, Any]]:
    specs = (SPECS if product == "full"
             else [s for s in SPECS if s["label"] in REDUCED_SPEC_LABELS])
    courants = COURANTS if product == "full" else (INEXACT_COURANT,)
    classes = (VALUE_CLASSES if product == "full"
               else ("uniform", "signed_zero"))
    masks = (ROW_MASK_SPECS if product == "full" else ROW_MASK_SPECS[:2])
    plan: List[Dict[str, Any]] = []
    for spec in specs:
        for mask_spec in masks:
            for courant in courants:
                for value_class in classes:
                    plan.append({"spec": spec, "mask_spec": mask_spec,
                                 "courant": courant, "value_class": value_class,
                                 "steps": steps})
    return plan


def summarize(cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    scored = [case for case in cases if case.get("case_is_valid")]
    per_guard: Dict[str, Dict[str, int]] = {}
    per_arm: Dict[str, Dict[str, int]] = {}
    #: The unguarded control at the inexact courant, PER ARM, with every case it
    #: agreed on named: the guard clause asks the flag to be load-bearing on each
    #: emitted tail, and an agreement is recorded rather than averaged away.
    control_by_arm: Dict[str, Dict[str, Any]] = {}
    for case in scored:
        if case["guard"] == "default_no_options" and case["courant"] == INEXACT_COURANT:
            control = control_by_arm.setdefault(
                case["arm"], {"ran": 0, "diverged": 0, "agreed": []})
            control["ran"] += 1
            if case["bit_identical"]:
                control["agreed"].append(case["key"])
            else:
                control["diverged"] += 1
        bucket = per_guard.setdefault(case["guard"], {
            "ran": 0, "identical": 0, "inexact_ran": 0, "inexact_identical": 0})
        bucket["ran"] += 1
        bucket["identical"] += int(case["bit_identical"])
        if case["courant"] == INEXACT_COURANT:
            bucket["inexact_ran"] += 1
            bucket["inexact_identical"] += int(case["bit_identical"])
        if case["guard"] != "fmad_false":
            continue
        arm = per_arm.setdefault(case["arm"], {"ran": 0, "identical": 0})
        arm["ran"] += 1
        arm["identical"] += int(case["bit_identical"])
    guarded = [case for case in scored if case["guard"] == "fmad_false"]
    folded_cases = [case for case in guarded if any(case["mirrored"])]
    phased = [case for case in guarded if any(case["phase_flags"])]
    return {
        "cases": len(cases),
        "scored": len(scored),
        "refused_as_vacuous": len(cases) - len(scored),
        "per_guard": per_guard,
        "per_arm": per_arm,
        "unguarded_inexact_by_arm": control_by_arm,
        "guarded_identical": sum(int(c["bit_identical"]) for c in guarded),
        "guarded_ran": len(guarded),
        "folded_identical": sum(int(c["bit_identical"]) for c in folded_cases),
        "folded_ran": len(folded_cases),
        "phased_identical": sum(int(c["bit_identical"]) for c in phased),
        "phased_ran": len(phased),
        "fold_is_a_live_partner_axis": sum(
            int(c["fold_reaches_the_partner_arm"]) for c in guarded),
        "fold_is_a_live_own_axis": sum(
            int(c["fold_reaches_the_own_arm"]) for c in guarded),
        "wall_mask_bites": sum(int(c["wall_mask_bites"]) for c in guarded),
        "subnormal_band_scored": sum(
            1 for c in guarded if c["value_class"] == "subnormal_band"),
        "signed_zero_scored": sum(
            1 for c in guarded if c["value_class"] == "signed_zero"),
        "phase_table_agrees_with_complex_pml_kernels": all(
            c["phase_table_agrees_with_complex_pml_kernels"] for c in cases),
        "predicates_disjoint": all(
            c["predicates"]["arms_disjoint"]
            and c["predicates"]["disjoint_from_siblings"]
            and c["predicates"]["expected_arm_admits"]
            for c in cases if "predicates" in c),
        # BOTH declared fold terminations, counted separately: a total alone
        # cannot show the prediction that ONE code serves both was exercised.
        "folded_periodic_identical": _fraction(
            [c for c in folded_cases
             if any(m and not t for m, t in zip(c["mirrored"], c["metallic"]))]),
        "folded_metallic_identical": _fraction(
            [c for c in folded_cases
             if any(m and t for m, t in zip(c["mirrored"], c["metallic"]))]),
        "simultaneous_planes_identical": {
            str(n): _fraction([c for c in folded_cases
                               if sum(c["mirrored"]) == n])
            for n in (1, 2, 3)},
    }


def _fraction(cases: Sequence[Dict[str, Any]]) -> str:
    return f"{sum(int(c['bit_identical']) for c in cases)}/{len(cases)}"


def run_sweep(backend, xp, results, out_path, product, steps,
              leg_name) -> None:
    plan = case_product(product, steps)
    total = len(plan) * len(GUARD_SETS)
    cases: List[Dict[str, Any]] = []
    index = 0
    for entry in plan:
        for guard_label, guard, _scored in GUARD_SETS:
            index += 1
            case = one_case(backend, xp, entry["spec"], entry["mask_spec"],
                            entry["courant"], entry["value_class"], guard_label,
                            guard, entry["steps"])
            valid, why = case_is_valid(case)
            case["case_is_valid"] = valid
            case["invalid_reason"] = why
            cases.append(case)
            log(f"{leg_name} {index}/{total} {case['key']} "
                f"identical={case['bit_identical']} "
                f"diff={case.get('differing_words')} "
                f"valid={valid} ({case['seconds']} s)")
            results[leg_name] = {"cases": cases, "summary": summarize(cases)}
            save(results, out_path)
    backend.clear()


# ---------------------------------------------------------------------------
# The mutation battery
# ---------------------------------------------------------------------------

def _armable(name: str, case: Dict[str, Any]) -> bool:
    """Whether a leg's defect can REACH the grid this case scores on.

    THE DEAD-BRANCH TRAP THIS EXISTS FOR: a mutation can rewrite REAL lines the
    scored grid never enters, because they sit under a guard that case does not
    take. It then reports UNCAUGHT while measuring nothing. Every leg is checked
    against the CASE, not just against the rewrite count.
    """
    if name == "parity_full_complex_product":
        # THE SPELLING DIFFERS ONLY ON SIGNED ZEROS, so the leg is armed on the
        # signed-zero class -- the class the difference lives in -- and on a fold
        # the mirror ghost actually reaches. It is must-catch there; arming it
        # anywhere weaker would make a miss a statement about the fixture.
        return (case["fold_reaches_the_partner_arm"]
                and case["value_class"] == "signed_zero")
    if name in ("drop_the_parity_weight", "weight_every_lane"):
        # A PARITY OF +1 MAKES BOTH REWRITES NEAR-IDENTITIES. ``return z`` against
        # ``mul_coefficient_left(1.0f, z)`` differs only in the sign of a zero, and
        # a blanket weight of +1.0 is the identity on every normal word. The defect
        # these legs exist for is a DROPPED or MISPLACED parity, and only an ODD
        # plane (-1) can show it -- the dead-branch trap in its arithmetic form.
        return any(case["ghost_weights"][axis] == -1.0
                   for axis in case["roles"]["folded_partner_axes"])
    if name == "down_ghost_real_plane_only":
        # The METALLIC near ghost, which is reached only where ``coord_dn`` returns
        # -1: a live PARTNER axis whose code is METALLIC. On a periodic or mirrored
        # partner the rewritten branch is never entered.
        return any(case["codes"][axis] == coverage.BC_CODES["metallic"]
                   for axis in case["roles"]["live_partner_axes"])
    if name == "up_ghost_real_plane_only":
        # Reached where ``coord_up`` returns -1: a live OWN axis that is METALLIC
        # or MIRRORED. The leg is a declared NULL and the arming is what makes the
        # NULL a measurement rather than an unentered branch.
        return any(case["codes"][axis] != coverage.BC_CODES["periodic"]
                   for axis in case["roles"]["live_own_axes"])
    if name in FOLD_PARTNER_MUTATIONS:
        return case["fold_reaches_the_partner_arm"]
    if name in FOLD_OWN_MUTATIONS:
        return case["fold_reaches_the_own_arm"]
    if name in DOWN_PHASE_MUTATIONS:
        return any(case["phase_flags"][axis]
                   for axis in case["roles"]["live_partner_axes"])
    if name in UP_PHASE_MUTATIONS:
        return any(case["phase_flags"][axis]
                   for axis in case["roles"]["live_own_axes"])
    if name in WALL_MUTATIONS:
        # The presence case keeps its rows on an interior block, so the coupling the
        # wall mask clears is already zero there and the mask cannot move a word.
        return case["wall_mask_bites"] and not case["row_block"]
    if name in ("select_removed", "predicate_on_coefficient_identity",
                "auxiliary_stored_on_a_plain_cell"):
        # THE CHUNK FORMS NEED A LAYERED DIVISION: with no chunk carrying a layer
        # every cell takes the plain form and the select has nothing to choose.
        return case["arm"] == "pml" and case["layered"]
    if name in PML_ONLY_MUTATIONS:
        return case["arm"] == "pml"
    if name == "presence_ignored":
        # A chunk must DROP an array somewhere for the presence word to decide a
        # term: where every chunk code a live row's cells read is kept, ignoring the
        # word changes nothing.
        return bool(case["presence_drops_a_term"])
    if name in STORE_ONLY_MUTATIONS:
        return case["arm"] == "no_pml"
    if name in TWO_SLOT_MUTATIONS:
        mask = case["row_mask"]
        return any(mask[2 * c] and mask[2 * c + 1] for c in range(3))
    if name == "dead_pml_apply":
        return True  # dead on BOTH arms, which is the point
    return True


def mutation_plan(scored: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One armable case per leg, chosen from the SCORED sweep cases.

    A leg with no armable case is recorded as ``UNARMED`` rather than silently
    dropped: a battery that quietly loses a leg reports a smaller denominator and
    the same numerator.
    """
    plan: List[Dict[str, Any]] = []
    for name in SOURCE_MUTATIONS:
        candidates = [case for case in scored if _armable(name, case)]
        # DETERMINISTIC, and biased toward the case with the most room for THIS
        # leg's defect: the signed-zero class first where the defect lives in the
        # sign of a zero, the UNIFORM class first everywhere else (a fixture whose
        # real planes are half zeros has fewer normal words for an ordinary defect
        # to move), then the largest grid, then the key.
        #
        # "The first armable case" is deterministic too, and it is what the first
        # device run used: it picked whatever the spec order happened to put first,
        # which for three legs was a case the defect could not reach at all.
        wants_zeros = name in SIGNED_ZERO_MUTATIONS
        candidates.sort(key=lambda c: (
            (c["value_class"] != "signed_zero") if wants_zeros
            else (c["value_class"] != "uniform"),
            -int(np.prod(c["shape"])), c["key"]))
        plan.append({"mutation": name, "kind": "source",
                     "must_be_caught": name not in NULL_MUTATIONS,
                     "case": candidates[0]["key"] if candidates else None,
                     "armable_cases": len(candidates)})
        if name in LIVE_ON_SOME_CASE_MUTATIONS:
            plan[-1]["cases"] = [c["key"] for c in candidates]
    for name in HOST_MUTATIONS + NULL_HOST_MUTATIONS:
        candidates = [case for case in scored
                      if (name != "hand_the_fold_the_metallic_code"
                          or case["fold_reaches_the_partner_arm"])
                      and (name != "swap_sub_lattice" or case["arm"] == "pml")
                      and (name != "drop_the_wall_flags" or case["wall_mask_bites"])
                      and (name not in ("swap_the_phase_directions",
                                        "clear_the_phase_flags")
                           or any(case["phase_flags"][axis]
                                  for axis in (case["roles"]["live_partner_axes"]
                                               + case["roles"]["live_own_axes"])))
                      and (name != "flip_the_parity"
                           or case["fold_reaches_the_partner_arm"])
                      # A TRANSPOSE OF THREE EQUAL CODES IS THE IDENTITY. On an
                      # all-periodic grid the rewrite changes no argument and the
                      # NULL would be about the fixture, not the kernel.
                      and (name != "transpose_the_boundary_codes"
                           or len(set(case["codes"])) > 1)
                      # THE RECORD DECIDES THE CONSTITUTIVE FORM ONLY UNDER A LAYER.
                      and (name not in RECORD_HOST_MUTATIONS
                           or (case["arm"] == "pml" and case["layered"]))]
        plan.append({"mutation": name, "kind": "host",
                     "must_be_caught": name not in NULL_HOST_MUTATIONS,
                     "case": candidates[0]["key"] if candidates else None,
                     "armable_cases": len(candidates)})
    return plan


def run_mutations(backend, xp, results, out_path, scored) -> None:
    by_key = {case["key"]: case for case in scored}
    plan = mutation_plan(scored)
    legs: List[Dict[str, Any]] = []
    for index, entry in enumerate(plan, 1):
        leg = dict(entry)
        if entry["case"] is None:
            leg["outcome"] = "UNARMED"
            leg["as_required"] = False
            legs.append(leg)
            log(f"mutation {index}/{len(plan)} {entry['mutation']} UNARMED")
            results["mutations"] = {"legs": legs,
                                    "summary": mutation_summary(legs)}
            save(results, out_path)
            continue
        case = by_key[entry["case"]]
        spec = next(s for s in SPECS if s["label"] == case["label"])
        mask_spec = next(m for m in ROW_MASK_SPECS
                         if m["label"] == case["row_mask_label"])
        sites = 0
        if entry["kind"] == "source":
            transform = SOURCE_MUTATIONS[entry["mutation"]]
            pristine = family.complex_offdiag_source(
                spec["arm"], tuple(mask_spec["mask"]))
            mutated, sites = transform(pristine)
            leg["rewritten_sites"] = sites
            leg["source_changed"] = mutated != pristine
            if not leg["source_changed"]:
                leg["outcome"] = "NOT_APPLIED"
                leg["as_required"] = False
                legs.append(leg)
                log(f"mutation {index}/{len(plan)} {entry['mutation']} NOT_APPLIED")
                results["mutations"] = {"legs": legs,
                                        "summary": mutation_summary(legs)}
                save(results, out_path)
                continue
            backend.set_source_mutation(transform)
            mutated_case = one_case(
                backend, xp, spec, mask_spec, case["courant"],
                case["value_class"], "fmad_false", ("--fmad=false",),
                case["steps"], source_mutation=entry["mutation"])
            tried = [entry["case"]]
            for key in entry.get("cases", [])[1:]:
                if (not mutated_case["bit_identical"]
                        or mutated_case["launch_error"] is not None):
                    break
                other = by_key[key]
                other_spec = next(s for s in SPECS if s["label"] == other["label"])
                other_mask = next(m for m in ROW_MASK_SPECS
                                  if m["label"] == other["row_mask_label"])
                mutated_case = one_case(
                    backend, xp, other_spec, other_mask, other["courant"],
                    other["value_class"], "fmad_false", ("--fmad=false",),
                    other["steps"], source_mutation=entry["mutation"])
                tried.append(key)
            if "cases" in entry:
                leg["cases_tried"] = len(tried)
                leg["case"] = tried[-1]
            backend.clear()
        else:
            mutated_case = one_case(
                backend, xp, spec, mask_spec, case["courant"],
                case["value_class"], "fmad_false", ("--fmad=false",),
                case["steps"], host_mutation=entry["mutation"])
        caught = (not mutated_case["bit_identical"]) or (
            mutated_case["launch_error"] is not None)
        leg["caught"] = bool(caught)
        leg["differing_words"] = mutated_case.get("differing_words")
        leg["launch_error"] = mutated_case.get("launch_error")
        leg["outcome"] = ("CAUGHT" if caught else "NULL_CONFIRMED")
        leg["as_required"] = bool(caught) == bool(entry["must_be_caught"])
        legs.append(leg)
        log(f"mutation {index}/{len(plan)} {entry['mutation']} "
            f"{leg['outcome']} required_caught={entry['must_be_caught']} "
            f"as_required={leg['as_required']} diff={leg['differing_words']}")
        results["mutations"] = {"legs": legs, "summary": mutation_summary(legs)}
        save(results, out_path)
    backend.clear()


def mutation_summary(legs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "legs": len(legs),
        "as_required": sum(int(bool(leg.get("as_required"))) for leg in legs),
        "caught": sum(1 for leg in legs if leg.get("outcome") == "CAUGHT"),
        "null_confirmed": sum(1 for leg in legs
                              if leg.get("outcome") == "NULL_CONFIRMED"),
        "unarmed": sum(1 for leg in legs if leg.get("outcome") == "UNARMED"),
        "not_applied": sum(1 for leg in legs
                           if leg.get("outcome") == "NOT_APPLIED"),
        "escaped": [leg["mutation"] for leg in legs
                    if leg.get("must_be_caught")
                    and leg.get("outcome") == "NULL_CONFIRMED"],
        "false_positives": [leg["mutation"] for leg in legs
                            if not leg.get("must_be_caught")
                            and leg.get("outcome") == "CAUGHT"],
    }


# ---------------------------------------------------------------------------
# THE GUARD, MEASURED RATHER THAN ASSUMED
# ---------------------------------------------------------------------------

def guard_binary_probe() -> Dict[str, Any]:
    """Do the two option sets compile DIFFERENT PTX for this family's sources?

    WHY THIS LEG EXISTS. Every certified sibling on this track carries an
    ``--fmad=false`` control that DIVERGES: the real off-diagonal family measured
    its unguarded control 0/384 identical, which is what makes the flag
    CORRECTNESS there rather than prudence. On the FIRST device run of this family
    the unguarded control agreed on every case instead, and an agreement has two
    very different explanations:

    a. the flag changed the binary and the change happened to move no word --
       which would be luck, and luck is not a licence;
    b. the flag changed NOTHING, because the source has no free multiply-then-add
       pair left to contract. That was the 0.9.2 kernel's case (every fusion
       spelled ``__fmaf_rn``, a single ``fma.rn.f32`` whatever the flag says). The
       part-wise body is full of such pairs (the tail's
       ``kps * src`` minus ``kms * prev``, the row's per-part products), so (b) is
       not expected and a run that reports it names the source that refutes it.

    Those are distinguishable AT THE COMPILER, and this leg distinguishes them:
    it compiles the same source under both option tuples through the same NVRTC
    entry point CuPy uses and hashes the generated PTX. IDENTICAL PTX settles (b)
    -- the case agreement is then a property of the arithmetic and not a
    coincidence -- and DIFFERENT PTX with agreeing cases would be (a) and is
    recorded as such rather than passed over.

    THE FLAG IS KEPT EITHER WAY: MEEP's contraction-off build rounds every product
    and every sum on its own, and a family that compiled the same bytes under
    different options than its siblings would not be comparable to them.
    """
    from cupy.cuda import compiler  # noqa: PLC0415
    from cupy.cuda import device as cuda_device  # noqa: PLC0415

    arch = None
    try:
        properties = cuda_device.Device().attributes
        arch = f"{properties['ComputeCapabilityMajor']}{properties['ComputeCapabilityMinor']}"
    except Exception:  # noqa: BLE001 - the fallback is the compiler's own default
        arch = None
    out: Dict[str, Any] = {"arch": arch}
    entry: Dict[str, Any] = {}
    for tail in family.ARMS:
        source = family.complex_offdiag_source(
            tail, (1, 1, 1, 1, 1, 1))
        digests = {}
        for label, options, _scored in GUARD_SETS:
            try:
                ptx = compiler.compile_using_nvrtc(
                    source, options=tuple(options),
                    arch=arch, name_expressions=None)
            except TypeError:
                ptx = compiler.compile_using_nvrtc(
                    source, options=tuple(options), arch=arch)
            payload = ptx[0] if isinstance(ptx, tuple) else ptx
            if isinstance(payload, str):
                payload = payload.encode("utf-8")
            digests[label] = hashlib.sha256(payload).hexdigest()
        entry[tail] = {
            "digests": digests,
            "distinct": len(set(digests.values())) == len(digests),
        }
    out["per_tail"] = entry
    out["ptx_distinct"] = any(value["distinct"] for value in entry.values())
    return out


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------

def build_verdict(results: Dict[str, Any]) -> Dict[str, Any]:
    """The release verdict, and every clause it rests on, spelled out."""
    sweep = results.get("sweep", {}).get("summary", {})
    multi = results.get("multistep", {}).get("summary", {})
    mutations = results.get("mutations", {}).get("summary", {})
    guarded = sweep.get("per_guard", {}).get("fmad_false", {})
    unguarded = sweep.get("per_guard", {}).get("default_no_options", {})
    control_by_arm = sweep.get("unguarded_inexact_by_arm", {})
    # LOAD-BEARING ON EVERY TAIL: the unguarded control diverged on at least one
    # inexact case of each arm. The part-wise body hands the flag multiply-then-add
    # pairs on both tails; a case on which the two compilations happen to round
    # alike is named in ``unguarded_inexact_by_arm`` rather than excused.
    load_bearing_per_arm = bool(
        set(control_by_arm) == set(family.ARMS)
        and all(entry.get("diverged", 0) > 0 for entry in control_by_arm.values()))
    clauses = {
        "backend_certifies": results.get("backend") == "cupy",
        "single_launch_all_identical": bool(
            guarded.get("ran", 0) > 0
            and guarded.get("identical") == guarded.get("ran")),
        "multi_step_all_identical": bool(
            multi.get("guarded_ran", 0) > 0
            and multi.get("guarded_identical") == multi.get("guarded_ran")),
        "folded_cases_scored": bool(sweep.get("folded_ran", 0) > 0),
        "phased_cases_scored": bool(sweep.get("phased_ran", 0) > 0),
        "both_arms_scored": bool(
            sweep.get("per_arm", {}).get("pml", {}).get("ran", 0) > 0
            and sweep.get("per_arm", {}).get("no_pml", {}).get("ran", 0) > 0),
        "fold_reaches_both_roles": bool(
            sweep.get("fold_is_a_live_partner_axis", 0) > 0
            and sweep.get("fold_is_a_live_own_axis", 0) > 0),
        "wall_mask_exercised": bool(sweep.get("wall_mask_bites", 0) > 0),
        "subnormal_band_scored": bool(sweep.get("subnormal_band_scored", 0) > 0),
        "signed_zero_scored": bool(sweep.get("signed_zero_scored", 0) > 0),
        "phase_table_agrees_with_complex_pml_kernels": bool(
            sweep.get("phase_table_agrees_with_complex_pml_kernels")),
        "predicates_disjoint": bool(sweep.get("predicates_disjoint")),
        # THE GUARD AXIS MUST BE SWEPT, and the two explanations of its result must
        # be distinguished. The sibling clause -- "the unguarded control must
        # DIVERGE" -- does not transfer, and pretending it did would be a claim
        # this family's first device run contradicted. What is required here is:
        # the control ran at the inexact courant, AND either it diverged (the
        # flag is load-bearing) or the compiler produced IDENTICAL PTX under both
        # option sets (the flag has nothing to contract, so the agreement is a
        # property of the arithmetic rather than luck). ``guard_finding`` below
        # records which of the two the run measured.
        "guard_axis_swept": bool(unguarded.get("inexact_ran", 0) > 0),
        "guard_result_explained": bool(
            unguarded.get("inexact_ran", 0) > 0
            and (unguarded.get("inexact_identical") == 0
                 or load_bearing_per_arm
                 or results.get("guard_binary_probe", {}).get(
                     "ptx_distinct") is False)),
        "mutations_all_as_required": bool(
            mutations.get("legs", 0) > 0
            and mutations.get("as_required") == mutations.get("legs")),
        "no_mutation_escaped": not mutations.get("escaped"),
        "no_false_positive": not mutations.get("false_positives"),
        "no_leg_unarmed": mutations.get("unarmed", 1) == 0,
    }
    probe = results.get("guard_binary_probe", {})
    finding = None
    if unguarded.get("inexact_ran", 0) > 0:
        if unguarded.get("inexact_identical") == 0:
            finding = ("--fmad=false is LOAD-BEARING: the unguarded control "
                       f"diverged on {unguarded['inexact_ran']}/"
                       f"{unguarded['inexact_ran']} cases at the inexact courant")
        elif load_bearing_per_arm:
            agreed = sorted({key.split("|")[0] for entry in control_by_arm.values()
                             for key in entry.get("agreed", [])})
            finding = (
                "--fmad=false is LOAD-BEARING on every tail: the unguarded control "
                "diverged on "
                f"{unguarded['inexact_ran'] - unguarded['inexact_identical']}/"
                f"{unguarded['inexact_ran']} cases at the inexact courant, "
                + ", ".join(f"{arm} {entry.get('diverged', 0)}/{entry.get('ran', 0)}"
                            for arm, entry in sorted(control_by_arm.items()))
                + f"; it agreed on {unguarded['inexact_identical']}, every one on "
                f"{agreed} (named per case in unguarded_inexact_by_arm)")
        elif probe.get("ptx_distinct") is False:
            finding = (
                "--fmad=false is NOT load-bearing on this source and the "
                "compiler says why: NVRTC emitted IDENTICAL PTX under both option "
                "tuples for both tails, so no bare multiply fed a bare add in the "
                "emitted source. The unguarded control accordingly agreed on "
                f"{unguarded['inexact_identical']}/{unguarded['inexact_ran']} cases. "
                "The part-wise body is expected to carry such pairs, so this "
                "refutes that expectation on this source; the flag is KEPT, as "
                "MEEP's contraction-off build requires.")
        else:
            finding = (
                "UNEXPLAINED: the unguarded control agreed on "
                f"{unguarded['inexact_identical']}/{unguarded['inexact_ran']} "
                "cases while NVRTC emitted DIFFERENT PTX under the two option "
                "tuples. That is agreement by luck rather than by construction "
                "and it does not license the guard.")
    return {"clauses": clauses, "guard_finding": finding,
            "release": all(clauses.values())}


def verdict_flips_against_planted_defect(results: Dict[str, Any]) -> Dict[str, Any]:
    """The release verdict, recomputed against a record with a defect planted.

    A gate whose verdict cannot go red is not a gate. Three plants, each in a
    different clause family.
    """
    import copy  # noqa: PLC0415

    out: Dict[str, Any] = {"baseline": build_verdict(results)["release"],
                           "plants": {}}
    plants = {
        "one_sweep_case_diverges": lambda r: r["sweep"]["summary"]["per_guard"][
            "fmad_false"].__setitem__("identical", max(
                0, r["sweep"]["summary"]["per_guard"]["fmad_false"]["identical"] - 1)),
        "one_mutation_escapes": lambda r: r["mutations"]["summary"].__setitem__(
            "escaped", ["planted"]),
        "guard_result_unexplained": lambda r: (
            r["sweep"]["summary"]["per_guard"]["default_no_options"].__setitem__(
                "inexact_identical", 1),
            r["sweep"]["summary"].__setitem__("unguarded_inexact_by_arm", {
                arm: dict(entry, diverged=0) for arm, entry in r["sweep"]["summary"].get(
                    "unguarded_inexact_by_arm", {"pml": {}}).items()}),
            r.setdefault("guard_binary_probe", {}).__setitem__(
                "ptx_distinct", True)),
    }
    for name, plant in plants.items():
        planted = copy.deepcopy(results)
        try:
            plant(planted)
        except Exception as exc:  # noqa: BLE001
            out["plants"][name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        out["plants"][name] = {"release": build_verdict(planted)["release"]}
    out["all_flip"] = all(
        entry.get("release") is False for entry in out["plants"].values())
    return out


def save(results: Dict[str, Any], out_path: str) -> None:
    """Atomic rewrite, STAMPED IMMEDIATELY BEFORE THE WRITE.

    ``meep_gpu/test_gate_provenance.py`` reads a three-line window above every
    write site rather than the first stamp in the file, because the first
    mechanical wiring on this track anchored on the first ``json.dump`` and that
    occurrence sat inside a refused early-exit branch: a run that actually released
    wrote its artifact from a different branch and recorded NO provenance at all.
    So the stamp goes in the window, not merely in the function.
    """
    directory = os.path.dirname(os.path.abspath(out_path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = out_path + ".tmp"
    gate_provenance.stamp(results)
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=False)
    os.replace(tmp, out_path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    global _POLICY_NAME

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--product", choices=("full", "reduced"), default="full")
    parser.add_argument("--subnormal-policy", choices=("keep", "flush"),
                        default="keep")
    parser.add_argument("--legs", default="sweep,multistep,mutations,falsify")
    parser.add_argument("--multi-step-budget", type=int,
                        default=MULTI_STEP_BUDGET)
    args = parser.parse_args(argv)
    _POLICY_NAME = args.subnormal_policy

    if cp is None:
        raise SystemExit("this gate needs CuPy; run it on the device host")

    # THE FLUSH LEG NEEDS MEEP IMPORTED FIRST, and that is a MEASURED REFUSAL
    # rather than a precaution: ``install_subnormal_policy('flush')`` raises in a
    # process that has not imported MEEP and says why in its own words --
    # ``mp.set_zero_subnormals`` is the only exposure of this process's FTZ/DAZ
    # bits the package may use, and ``subnormal_policy`` will not import MEEP
    # itself because that would break ``meep_gpu``'s no-MEEP-import boundary and
    # initialize MPI as a side effect. This gate's first flush attempt was refused
    # exactly that way. It is NOT unconditional: importing MEEP moves the host FPU
    # and initializes MPI, so a ``keep`` leg that imported it would differ from a
    # ``keep`` leg that did not for reasons unrelated to the kernel.
    if args.subnormal_policy == "flush":
        results_meep_import = probe.import_meep_for_host_policy()
    else:
        results_meep_import = {"requested": False}

    from meep_gpu import subnormal_policy as policy_module  # noqa: PLC0415

    policy = policy_module.install_subnormal_policy(args.subnormal_policy,
                                                    strict=True)

    results: Dict[str, Any] = {
        "gate": os.path.basename(__file__),
        "backend": "cupy",
        "subnormal_policy": policy,
        "requested_policy": args.subnormal_policy,
        "meep_import_for_host_policy": results_meep_import,
        "product": args.product,
        "multi_step_budget": args.multi_step_budget,
        "emitter_corpus_digest": family.corpus_digest(),
        "kernels": dict(family.KERNEL_NAMES),
        "row_mask_specs": [dict(spec) for spec in ROW_MASK_SPECS],
        "specs": [spec["label"] for spec in SPECS],
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    save(results, args.out)

    backend = KernelBackend()
    xp = cp
    legs = tuple(name.strip() for name in args.legs.split(",") if name.strip())

    if "sweep" in legs:
        run_sweep(backend, xp, results, args.out, args.product, 1,
                  "sweep")
    if "multistep" in legs:
        run_sweep(backend, xp, results, args.out, "reduced",
                  args.multi_step_budget, "multistep")
    if "mutations" in legs:
        scored = [case for case in results.get("sweep", {}).get("cases", [])
                  if case.get("case_is_valid") and case["guard"] == "fmad_false"]
        run_mutations(backend, xp, results, args.out, scored)
    if "sweep" in legs or "guard" in legs:
        results["guard_binary_probe"] = guard_binary_probe()
        log(f"guard PTX probe: distinct="
            f"{results['guard_binary_probe']['ptx_distinct']}")
        save(results, args.out)
    results["verdict"] = build_verdict(results)
    if "falsify" in legs:
        results["falsification"] = verdict_flips_against_planted_defect(results)
    results["source_digests"] = dict(backend.sources_used)
    results["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save(results, args.out)

    verdict = results["verdict"]
    log(f"RELEASE={verdict['release']}")
    for name, value in verdict["clauses"].items():
        log(f"  {name}: {value}")
    if verdict.get("guard_finding"):
        log(f"  GUARD FINDING: {verdict['guard_finding']}")
    falsification = results.get("falsification")
    if falsification is not None and not falsification.get("all_flip"):
        log("FALSIFICATION FAILED: the release verdict did not flip against a "
            "planted defect")
        return 2
    return 0 if verdict["release"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
