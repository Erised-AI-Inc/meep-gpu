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

THE PARITY LICENCE
==================
The mirror ghost is ``_symmetry_phase(...) * _mirror_source(...)``
(stepping.py:1873-1875): a python int on the LEFT of a complex64 plane, which is
``folded_complex.PARITY_PROBE_PATTERN`` and none of the base four orientations. So
this gate binds ``folded_complex.parity_expansion_license`` over the FIVE-pattern
set, not ``complex_fields.expansion_license`` over four. The shipped
``expansion_probe_2026-08-17`` record carries the fifth pattern under BOTH
policies, which is why this family can span both where
``gate_cuda_complex_no_pml`` spans one.

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
  is not ``1+0j``, or every rotation mutation is a null for a reason about the
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
first, and this family's bodies are the complex ones, where the zero cross terms
and the FMA arm are the whole question and NumPy expresses neither. What a laptop
CAN settle is in ``meep_gpu/cuda_kernels/test_complex_offdiag_update_e.py``: the
emitted text against the certified templates it is built from, a TEXT-ANCHORED
complex128 evaluator measured against ``stepping.update_E`` over 112 structural
configurations, three float32 hazards, and the predicates' whole verdict table on
real engine objects.

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
from meep_gpu.cuda_kernels import complex_emitter  # noqa: E402
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

#: The FIVE-pattern probe record. The parity orientation is the fifth, and the
#: base-four licence cannot bind the mirror arm.
PROBE_RECORD = ("parity/meep_gpu/results/expansion_probe_2026-08-17/"
                "expansion_probe_{policy}.json")

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
    "store_3d_phased", "store_metallic_walls")


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


def m_flip_the_parity_sign(text):
    return _replace("    if (bc == BC_MIRROR) return mul_coefficient_left(w, z);",
                    "    if (bc == BC_MIRROR) return mul_coefficient_left("
                    "0.0f - w, z);", text)


def m_drop_the_parity_weight(text):
    return _replace("    if (bc == BC_MIRROR) return mul_coefficient_left(w, z);",
                    "    if (bc == BC_MIRROR) return z;", text)


def m_parity_plane_wise(text):
    """The plane-wise spelling of the parity multiply -- byte-wrong on signed zeros."""
    return _replace(
        "    if (bc == BC_MIRROR) return mul_coefficient_left(w, z);",
        "    if (bc == BC_MIRROR) { cf o; o.re = w * z.re; o.im = w * z.im; "
        "return o; }", text)


def m_weight_every_lane(text):
    """The blanket weight the module refuses: an operation the oracle does not have."""
    return _replace("    if (!gl) return z;",
                    "    if (!gl) return (bc == BC_MIRROR) ? "
                    "mul_coefficient_left(w, z) : z;", text)


def m_folded_own_axis_wraps(text):
    """A folded axis wrapping in ``coord_up`` instead of serving the exact zero."""
    return _replace("    return (bc == BC_PERIODIC) ? 0 : -1;",
                    "    return (bc == BC_METALLIC) ? -1 : 0;", text)


# --- the phase, stated wrongly ------------------------------------------------

def m_unconjugated_down_phase(text):
    """The classic sign error: the same factor in both directions.

    Every magnitude stays plausible and only the phase moves, which is what a band
    structure is made of. Planted on ALL THREE axes at the point where the kernel
    unpacks the table, so the leg bites whichever axis the scored case phases.
    """
    return _sub(r"(cf dp([xyz]); dp\2\.re = dp\2r; dp\2\.im = )dp\2i;",
                r"\1(0.0f - dp\2i);", text)


def m_phase_the_field_not_the_product(text):
    """Rotate the pre-coefficient field instead of the formed product."""
    old = ("    cf far_term = mul_field_left(far_pair, (up < 0) ? 0.0f : u[up]);\n"
           "    if (uw && uph) far_term = rotate_field_left(far_term, uphase);")
    new = ("    if (uw && uph) far_pair = rotate_field_left(far_pair, uphase);\n"
           "    cf far_term = mul_field_left(far_pair, (up < 0) ? 0.0f : u[up]);")
    return _replace(old, new, text)


def m_drop_the_down_phase(text):
    """The wrapped near face left unrotated: ``f(x - L)`` served as ``f(x)``.

    Inverting the ``ph`` test drops the rotation on a phased axis (and applies a
    1+0j rotation on an unphased one, which is why the leg is armed on a case whose
    live PARTNER axis is phased rather than on any phased case).
    """
    return _replace("    return ph ? rotate_field_left(z, phase) : z;",
                    "    return ph ? z : rotate_field_left(z, phase);", text)


def m_phase_the_whole_volume(text):
    """The Bloch factor applied to every cell instead of the single wrapped plane.

    ``_apply_bloch_phase`` multiplies exactly ``_face(axis, -1)`` of the rolled
    buffer (stepping.py:1893-1909); a kernel that phased the volume would be
    smooth, converged and carrying a wrong dispersion relation.
    """
    return _replace(
        "    if (uw && uph) far_term = rotate_field_left(far_term, uphase);",
        "    if (uph) far_term = rotate_field_left(far_term, uphase);", text)


# --- the row product, stated wrongly ------------------------------------------

def m_hoist_the_coefficient(text):
    """The four-point average times ``u[i]`` -- the same ALGEBRA only for a uniform
    coefficient, and a different float32 number even then."""
    old = ("    cf near_term = mul_field_left(near_pair, u[home]);\n"
           "    cf far_term = mul_field_left(far_pair, (up < 0) ? 0.0f : u[up]);")
    new = ("    cf near_term = mul_field_left(near_pair, u[home]);\n"
           "    cf far_term = mul_field_left(far_pair, u[home]);")
    return _replace(old, new, text)


def m_ungosted_up_coefficient(text):
    """Read ``u[up]`` where ``_shift_up`` writes an exact zero -- bit-identity 4."""
    return _replace("mul_field_left(far_pair, (up < 0) ? 0.0f : u[up])",
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
    annihilated by ``mul_field_left(far_pair, 0.0f)`` before it can reach the
    output. The two ghosts are therefore belt AND braces on this leg, and
    ``ungosted_up_coefficient`` -- which removes the braces -- is the leg that
    proves the belt alone is not enough (CAUGHT, thousands of words).
    """
    return _replace(
        "    return (index < 0) ? cf_zero() : cf_load(g, index);",
        "    if (index < 0) { cf o = cf_load(g, 0); o.re = 0.0f; return o; }\n"
        "    return cf_load(g, index);", text)


def m_couple_the_wrong_components(text):
    """Cycle the partner volumes: Dy where Dz belongs."""
    return _sub(r"        (D[xyz]), (chi1inv_\w+), idx,",
                lambda m: "        "
                          + {"Dx": "Dy", "Dy": "Dz", "Dz": "Dx"}[m.group(1)]
                          + f", {m.group(2)}, idx,", text)


def m_transpose_the_coefficient(text):
    """Read the coefficient one cell up its own axis -- the registration MEEP fixes."""
    return _replace("mul_field_left(near_pair, u[home])",
                    "mul_field_left(near_pair, (up < 0) ? 0.0f : u[up])", text)


def m_drop_a_row_from_the_volume_sum(text):
    return _sub(r"    total_(E[xyz]) = cf_add\(total_\1, term_\1_([01])\);", "",
                text)


def m_invert_the_wall_mask(text):
    return _sub(r"\? cf_zero\(\) : (total_E[xyz]);", r": \1 ? cf_zero();", text) \
        if False else _sub(
            r"\((wm_[xyz]) && (at_[xyz])\) \? cf_zero\(\) : (total_E[xyz]);",
            r"(\1 && \2) ? \3 : cf_zero();", text)


def m_drop_the_wall_mask(text):
    return _sub(r"\((wm_[xyz]) && (at_[xyz])\) \? cf_zero\(\) : (total_E[xyz]);",
                r"\3;", text)


def m_wall_mask_real_plane_only(text):
    return _sub(
        r"    (total_E[xyz]) = \((wm_[xyz]) && (at_[xyz])\) \? cf_zero\(\) : \1;",
        r"    if (\2 && \3) \1.re = 0.0f;", text)


# --- the tails ----------------------------------------------------------------

def m_wrong_own_axis_table(text):
    """The dsig/dsigu cycle the CURL uses, where MEEP's dsigw belongs. A smooth,
    converged, entirely wrong absorber."""
    return _sub(r"kps_x\[i\], kms_x\[i\]", "kps_y[j], kms_y[j]", text)


def m_store_fw_before_reading_prev(text):
    """Wrong only where kms != 0, i.e. INSIDE the absorber -- which reads as a
    slightly weaker layer rather than as a bug."""
    old = ("    cf prev = cf_load(fw, idx);\n"
           "    cf_store(fw, idx, src);")
    new = ("    cf_store(fw, idx, src);\n"
           "    cf prev = cf_load(fw, idx);")
    return _replace(old, new, text)


def m_flatten_the_tail(text):
    """``f + (kps*src - kms*prev)`` -- a different float32 number, and what the
    twelve uncertified complex kernels in step_curl_kernels.py write."""
    old = ("    cf a = cf_load(f, idx);\n"
           "    a = cf_add(a, mul_coefficient_left(kps, src));\n"
           "    a = cf_sub(a, mul_coefficient_left(kms, prev));")
    new = ("    cf a = cf_load(f, idx);\n"
           "    a = cf_add(a, cf_sub(mul_coefficient_left(kps, src),\n"
           "                         mul_coefficient_left(kms, prev)));")
    return _replace(old, new, text)


def m_store_arm_accumulates(text):
    """The store arm made into a recurrence -- the two-kernel decision, planted."""
    return _sub(r"    cf_store\((E[xyz]), idx, (src_E[xyz])\);",
                r"    cf_store(\1, idx, cf_add(cf_load(\1, idx), \2));", text)


# --- the must-be-UNCAUGHT nulls ----------------------------------------------

def n_commute_the_row_sum(text):
    """float32 addition is bitwise COMMUTATIVE, so swapping the two addends of the
    row accumulation must move NO word. A leg that caught it would be reporting a
    comparator that fails everything."""
    return _sub(r"    total_(E[xyz]) = cf_add\(total_\1, term_\1_([01])\);",
                r"    total_\1 = cf_add(term_\1_\2, total_\1);", text)


def n_commute_the_pair_add(text):
    """Same fact on the two-point partner pair."""
    return _replace(
        "    cf near_pair = cf_add(cf_load(g, home),\n"
        "                          down_sample(g, down, dgl, dbc, dph, dphase, w));",
        "    cf near_pair = cf_add(down_sample(g, down, dgl, dbc, dph, dphase, w),\n"
        "                          cf_load(g, home));", text)


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
    "parity_plane_wise": m_parity_plane_wise,
    "weight_every_lane": m_weight_every_lane,
    "folded_own_axis_wraps": m_folded_own_axis_wraps,
    "unconjugated_down_phase": m_unconjugated_down_phase,
    "phase_the_field_not_the_product": m_phase_the_field_not_the_product,
    "drop_the_down_phase": m_drop_the_down_phase,
    "phase_the_whole_volume": m_phase_the_whole_volume,
    "hoist_the_coefficient": m_hoist_the_coefficient,
    "ungosted_up_coefficient": m_ungosted_up_coefficient,
    "same_direction_shifts": m_same_direction_shifts,
    "down_ghost_real_plane_only": m_down_ghost_real_plane_only,
    "up_ghost_real_plane_only": m_up_ghost_real_plane_only,
    "couple_the_wrong_components": m_couple_the_wrong_components,
    "transpose_the_coefficient": m_transpose_the_coefficient,
    "drop_a_row_from_the_volume_sum": m_drop_a_row_from_the_volume_sum,
    "invert_the_wall_mask": m_invert_the_wall_mask,
    "drop_the_wall_mask": m_drop_the_wall_mask,
    "wall_mask_real_plane_only": m_wall_mask_real_plane_only,
    "wrong_own_axis_table": m_wrong_own_axis_table,
    "store_fw_before_reading_prev": m_store_fw_before_reading_prev,
    "flatten_the_tail": m_flatten_the_tail,
    "store_arm_accumulates": m_store_arm_accumulates,
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
#: ``parity_plane_wise`` -- the plane-wise spelling of the mirror parity multiply.
#: The two spellings DO differ at the multiply: measured on the host over the four
#: signed-zero patterns, 4 of 16 float32 words
#: (``test_complex_offdiag_update_e.test_the_parity_multiply_must_be_the_full_
#: complex_product``). They do NOT differ at this kernel's OUTPUT, because with the
#: parity exactly +/-1 the only difference is the SIGN OF A ZERO in the ghost, and
#: every path from the ghost to a stored word passes through an addition with a
#: normal operand (``near_pair``, then ``0.25*(near+far)``, then ``diag + total``)
#: which absorbs it. So ``mul_coefficient_left`` is kept on TRANSCRIPTION FIDELITY
#: -- it is what the array path dispatches -- and NOT on a measured byte difference
#: at the output. That is a weaker claim than the module's other five, and it is
#: written down as the weaker claim it is.
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
    "parity_plane_wise", "up_ghost_real_plane_only",
    "down_ghost_real_plane_only")

#: Legs that can only bite where a fold reaches a LIVE PARTNER axis -- the only arm
#: on which the mirror ghost is read at all. Scored on a case whose ``roles`` say
#: so, and carried as a NULL on a case whose roles say it cannot.
FOLD_PARTNER_MUTATIONS: Tuple[str, ...] = (
    "mirror_ghost_as_the_metallic_zero", "mirror_ghost_as_the_periodic_wrap",
    "mirror_row_off_by_one", "flip_the_parity_sign", "drop_the_parity_weight",
    "parity_plane_wise", "weight_every_lane")

#: Legs that need a fold on a live OWN axis.
FOLD_OWN_MUTATIONS: Tuple[str, ...] = ("folded_own_axis_wraps",)

#: Legs whose defect sits on the PARTNER-axis DOWN shift, so the case needs a live
#: partner axis that is PHASED. A phased axis that no live row slot takes as its
#: partner is a genuine NULL and scoring it a miss would be a statement about the
#: fixture rather than about the kernel -- the DEAD-BRANCH TRAP, per leg.
DOWN_PHASE_MUTATIONS: Tuple[str, ...] = (
    "unconjugated_down_phase", "drop_the_down_phase")

#: Legs whose defect sits on the OWN-axis UP shift, so the case needs a live own
#: axis that is PHASED.
UP_PHASE_MUTATIONS: Tuple[str, ...] = (
    "phase_the_field_not_the_product", "phase_the_whole_volume")

#: Legs that need the wall mask to BITE on this (mask, grid) pair.
WALL_MUTATIONS: Tuple[str, ...] = (
    "invert_the_wall_mask", "drop_the_wall_mask", "wall_mask_real_plane_only")

#: Legs whose defect lives in the SIGN OF A ZERO, so the signed-zero value class is
#: the one that can show it. Every other leg prefers the uniform class, where a
#: fixture is all normal words and an ordinary defect has the most room.
SIGNED_ZERO_MUTATIONS: Tuple[str, ...] = ("parity_plane_wise",)

#: Legs that exist only on one arm.
PML_ONLY_MUTATIONS: Tuple[str, ...] = (
    "wrong_own_axis_table", "store_fw_before_reading_prev", "flatten_the_tail")
STORE_ONLY_MUTATIONS: Tuple[str, ...] = ("store_arm_accumulates",)

#: Legs that need TWO live slots on one component to have anything to drop.
TWO_SLOT_MUTATIONS: Tuple[str, ...] = ("drop_a_row_from_the_volume_sum",
                                       "commute_the_row_sum")

#: HOST mutations: the launch arguments, corrupted. Each is a deliberately wrong
#: answer a launcher could be handed by a caller that skipped the predicate.
HOST_MUTATIONS: Tuple[str, ...] = (
    "swap_sub_lattice", "hand_the_fold_the_metallic_code", "flip_the_parity",
    "drop_the_wall_flags", "swap_the_phase_directions", "clear_the_phase_flags",
    "transpose_the_boundary_codes")

#: HOST mutations that MUST come back UNCAUGHT on a case they cannot reach.
NULL_HOST_MUTATIONS: Tuple[str, ...] = ("rebind_identical_tables",)


def apply_host_mutation(name: Optional[str], layer, grid, arm, tables, codes,
                        walls, flags, weights, down, up):
    """Return the launch arguments this leg feeds, mutated or not."""
    if name is None or name == "rebind_identical_tables":
        if name == "rebind_identical_tables" and tables is not None:
            tables = {key: value for key, value in tables.items()}
        return tables, codes, walls, flags, weights, down, up
    if name == "swap_sub_lattice":
        # The INTEGER pair -- the H side's sub-lattice, wrong for E by half a cell.
        from meep_gpu.cuda_kernels.constitutive_kernels import (  # noqa: PLC0415
            real_constitutive_tables)
        return (real_constitutive_tables(layer, False), codes, walls, flags,
                weights, down, up)
    if name == "hand_the_fold_the_metallic_code":
        codes = tuple(coverage.BC_CODES["metallic"]
                      if code == family.BC_MIRROR_CODE else code
                      for code in codes)
    elif name == "flip_the_parity":
        weights = tuple(-value for value in weights)
    elif name == "drop_the_wall_flags":
        walls = (0, 0, 0)
    elif name == "swap_the_phase_directions":
        down, up = up, down
    elif name == "clear_the_phase_flags":
        flags = (0, 0, 0)
    elif name == "transpose_the_boundary_codes":
        codes = (codes[1], codes[2], codes[0])
    else:  # pragma: no cover - the table above is the whole set
        raise ValueError(f"no host mutation {name!r}")
    return tables, codes, walls, flags, weights, down, up


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

    def pristine_source(self, arm, mask, expansion) -> str:
        """The UNMUTATED source. Site counting and digesting must never go through
        an installed override: that would apply the transform twice and the leg
        would hash a body it never compiled."""
        return family.complex_offdiag_source(arm, mask, expansion)

    def source_for(self, arm, mask, expansion) -> Tuple[str, int]:
        text = self.pristine_source(arm, mask, expansion)
        if self.source_transform is None:
            return text, 0
        return self.source_transform(text)

    def launch(self, fields, arm, mask, expansion, tables, codes, walls, flags,
               weights, down, up) -> None:
        text, _sites = self.source_for(arm, mask, expansion)
        key = (arm, tuple(mask), expansion)
        if self.source_transform is not None:
            family.set_kernel_source(arm, mask, expansion, text)
            if key not in self._overridden:
                self._overridden.append(key)
        self.sources_used[str(key)] = hashlib.sha256(
            text.encode("utf-8")).hexdigest()
        if arm == "pml":
            family.update_E_complex_offdiag_fused_pml(
                fields, expansion, tables=tables, codes=codes, walls=walls,
                flags=flags, weights=weights, phases=(down, up))
        else:
            family.update_E_complex_no_pml_offdiag(
                fields, expansion, codes=codes, walls=walls, flags=flags,
                weights=weights, phases=(down, up))
        cp.cuda.runtime.deviceSynchronize()

    def compile_log(self) -> List[Dict[str, Any]]:
        from meep_gpu.cuda_kernels import compile_cache  # noqa: PLC0415

        return list(compile_cache.compile_log())

    def clear(self) -> int:
        for arm, mask, expansion in self._overridden:
            family.set_kernel_source(arm, mask, expansion, None)
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
             guard: Sequence[str], steps: int, expansion,
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
    flags, down, up = family.bloch_phase_table(grid)
    tables = (family.complex_offdiag_tables(kernel_layer) if arm == "pml"
              else None)

    case: Dict[str, Any] = {
        "key": key, "seed": seed, "label": spec["label"], "arm": arm,
        "row_mask_label": mask_spec["label"], "row_mask": list(mask),
        "installed_row_mask": list(coverage.offdiag_row_mask(kernel_fields)),
        "courant": courant, "value_class": value_class, "guard": guard_label,
        "steps": steps, "expansion": expansion,
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
        "down_phases": [float(value) for value in down],
        "up_phases": [float(value) for value in up],
        "roles": fold_roles(grid, mask),
        "wall_mask_bites": wall_mask_bites(mask, walls),
    }
    case["fold_reaches_the_partner_arm"] = bool(
        case["roles"]["fold_is_a_live_partner_axis"])
    case["fold_reaches_the_own_arm"] = bool(
        case["roles"]["fold_is_a_live_own_axis"])

    # THE PHASE TABLE, CROSS-CHECKED against the certified complex launcher's own
    # spelling. Two spellings of one fact are equal only until someone edits one,
    # and the merge bar cannot check this pair because that module imports cupy at
    # scope.
    from meep_gpu.cuda_kernels import complex_pml_kernels  # noqa: PLC0415

    their_flags_down, their_down = complex_pml_kernels.bloch_phase_arguments(
        grid, backward=True)
    their_flags_up, their_up = complex_pml_kernels.bloch_phase_arguments(
        grid, backward=False)
    case["phase_table_agrees_with_complex_pml_kernels"] = bool(
        tuple(int(f) for f in their_flags_down) == tuple(flags)
        and tuple(int(f) for f in their_flags_up) == tuple(flags)
        and tuple(float(v) for v in their_down) == tuple(float(v) for v in down)
        and tuple(float(v) for v in their_up) == tuple(float(v) for v in up))

    # THE PREDICATES' ANSWERS, recorded rather than acted on.
    licence = _licence()
    mine_pml = family.covers_complex_offdiag_pml_update_e(
        kernel_fields, kernel_layer, grid, license=licence,
        subnormal_policy=_POLICY_NAME)
    mine_store = family.covers_complex_no_pml_offdiag_update_e(
        kernel_fields, kernel_layer, grid, license=licence,
        subnormal_policy=_POLICY_NAME)
    theirs = coverage.covers_real_pml_complex_constitutive(
        kernel_fields, kernel_layer, grid, "E", license=licence,
        subnormal_policy=_POLICY_NAME)
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
                                 codes, walls, flags, weights, down, up)
    launch_error = None
    try:
        for _ in range(steps):
            backend.launch(kernel_fields, arm, mask, expansion, *launch)
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
    for case in scored:
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


def run_sweep(backend, xp, results, out_path, product, steps, expansion,
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
                            guard, entry["steps"], expansion)
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
    if name == "parity_plane_wise":
        # THE SPELLING DIFFERS ONLY ON SIGNED ZEROS, so the leg is armed on the
        # signed-zero class -- the class the difference lives in -- and on a fold
        # the mirror ghost actually reaches. It is a declared NULL; arming it
        # anywhere weaker would make the NULL a statement about the fixture.
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
        return case["wall_mask_bites"]
    if name in PML_ONLY_MUTATIONS:
        return case["arm"] == "pml"
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
                           or len(set(case["codes"])) > 1)]
        plan.append({"mutation": name, "kind": "host",
                     "must_be_caught": name not in NULL_HOST_MUTATIONS,
                     "case": candidates[0]["key"] if candidates else None,
                     "armable_cases": len(candidates)})
    return plan


def run_mutations(backend, xp, results, out_path, scored, expansion) -> None:
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
                spec["arm"], tuple(mask_spec["mask"]), expansion)
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
                case["steps"], expansion, source_mutation=entry["mutation"])
            backend.clear()
        else:
            mutated_case = one_case(
                backend, xp, spec, mask_spec, case["courant"],
                case["value_class"], "fmad_false", ("--fmad=false",),
                case["steps"], expansion, host_mutation=entry["mutation"])
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

def guard_binary_probe(expansion) -> Dict[str, Any]:
    """Do the two option sets compile DIFFERENT PTX for this family's sources?

    WHY THIS LEG EXISTS. Every certified sibling on this track carries an
    ``--fmad=false`` control that DIVERGES: the real off-diagonal family measured
    its unguarded control 0/384 identical, which is what makes the flag
    CORRECTNESS there rather than prudence. On the FIRST device run of this family
    the unguarded control agreed on every case instead, and an agreement has two
    very different explanations:

    a. the flag changed the binary and the change happened to move no word --
       which would be luck, and luck is not a licence;
    b. the flag changed NOTHING, because under the FMA_V1 arm this family has no
       free multiply-then-add pair left to contract: every fusion is spelled
       ``__fmaf_rn`` (a single ``fma.rn.f32`` whatever the flag says), every other
       product feeds a multiply or an fma ADDEND rather than a bare add, and
       ``cf_add``/``cf_sub`` add values the compiler cannot see a multiply behind.

    Those are distinguishable AT THE COMPILER, and this leg distinguishes them:
    it compiles the same source under both option tuples through the same NVRTC
    entry point CuPy uses and hashes the generated PTX. IDENTICAL PTX settles (b)
    -- the case agreement is then a property of the arithmetic and not a
    coincidence -- and DIFFERENT PTX with agreeing cases would be (a) and is
    recorded as such rather than passed over.

    THE FLAG IS KEPT EITHER WAY. A family that compiled the same bytes under
    different options than its siblings would not be comparable to them, and the
    NAIVE arm -- which this licence does not bind but the emitter can still emit --
    DOES carry a bare ``(z.re * c) - (z.im * 0.0f)`` multiply-then-subtract pair.
    So the probe sweeps BOTH arms and reports each.
    """
    from cupy.cuda import compiler  # noqa: PLC0415
    from cupy.cuda import device as cuda_device  # noqa: PLC0415

    arch = None
    try:
        properties = cuda_device.Device().attributes
        arch = f"{properties['ComputeCapabilityMajor']}{properties['ComputeCapabilityMinor']}"
    except Exception:  # noqa: BLE001 - the fallback is the compiler's own default
        arch = None
    out: Dict[str, Any] = {"arch": arch, "per_arm": {}}
    for arm_name in sorted(complex_emitter.EXPANSIONS):
        entry: Dict[str, Any] = {}
        for tail in family.ARMS:
            source = family.complex_offdiag_source(
                tail, (1, 1, 1, 1, 1, 1), arm_name)
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
        out["per_arm"][arm_name] = entry
    licensed = complex_emitter.EXPANSION_NAMES[
        complex_emitter.normalized_expansion(expansion)]
    out["licensed_arm"] = licensed
    out["licensed_arm_ptx_distinct"] = any(
        entry["distinct"] for entry in out["per_arm"][licensed].values())
    out["naive_arm_ptx_distinct"] = any(
        entry["distinct"] for entry in out["per_arm"]["NAIVE"].values())
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
                 or results.get("guard_binary_probe", {}).get(
                     "licensed_arm_ptx_distinct") is False)),
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
        elif probe.get("licensed_arm_ptx_distinct") is False:
            finding = (
                "--fmad=false is NOT load-bearing on the licensed arm and the "
                "compiler says why: NVRTC emitted IDENTICAL PTX under both option "
                "tuples for both tails, because every fusion in this family is "
                "spelled __fmaf_rn and no bare multiply feeds a bare add. The "
                f"unguarded control accordingly agreed on {unguarded['inexact_identical']}"
                f"/{unguarded['inexact_ran']} cases. The flag is KEPT for "
                "comparability with the certified siblings and because the NAIVE "
                "arm this licence does not bind DOES carry a contractable pair "
                f"(ptx_distinct={probe.get('naive_arm_ptx_distinct')}).")
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
            r.setdefault("guard_binary_probe", {}).__setitem__(
                "licensed_arm_ptx_distinct", True)),
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


# ---------------------------------------------------------------------------
# The licence
# ---------------------------------------------------------------------------

_LICENCE: Optional[Dict[str, Any]] = None
_POLICY_NAME = "keep"


def _licence() -> Dict[str, Any]:
    """The PARITY-pattern verdict, computed once and refused rather than defaulted."""
    global _LICENCE
    if _LICENCE is None:
        from meep_gpu.triton_kernels import complex_fields  # noqa: PLC0415
        from meep_gpu.triton_kernels import folded_complex  # noqa: PLC0415

        path = os.path.join(_REPO_API,
                            PROBE_RECORD.format(policy=_POLICY_NAME))
        with open(path, "r", encoding="utf-8") as handle:
            record = json.load(handle)
        verdict = folded_complex.parity_expansion_license(record)
        reasons = list(complex_fields.expansion_policy_reasons(
            record, _POLICY_NAME))
        if not verdict.get("arm") or verdict.get("refusals") or reasons:
            raise SystemExit(
                f"{path} licenses no PARITY arm under {_POLICY_NAME!r}: "
                f"arm={verdict.get('arm')!r} refusals={verdict.get('refusals')} "
                f"policy_reasons={reasons[:1]}")
        verdict["record_path"] = PROBE_RECORD.format(policy=_POLICY_NAME)
        _LICENCE = verdict
    return _LICENCE


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
    licence = _licence()
    expansion = licence["arm"]

    results: Dict[str, Any] = {
        "gate": os.path.basename(__file__),
        "backend": "cupy",
        "subnormal_policy": policy,
        "requested_policy": args.subnormal_policy,
        "meep_import_for_host_policy": results_meep_import,
        "expansion_licence": {
            "arm": licence.get("arm"), "basis": licence.get("basis"),
            "probe_patterns": list(licence.get("probe_patterns") or ()),
            "record": licence.get("record_path"),
            "parity_pattern": family.PARITY_PROBE_PATTERN,
        },
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
        run_sweep(backend, xp, results, args.out, args.product, 1, expansion,
                  "sweep")
    if "multistep" in legs:
        run_sweep(backend, xp, results, args.out, "reduced",
                  args.multi_step_budget, expansion, "multistep")
    if "mutations" in legs:
        scored = [case for case in results.get("sweep", {}).get("cases", [])
                  if case.get("case_is_valid") and case["guard"] == "fmad_false"]
        run_mutations(backend, xp, results, args.out, scored, expansion)
    if "sweep" in legs or "guard" in legs:
        results["guard_binary_probe"] = guard_binary_probe(expansion)
        log(f"guard PTX probe: licensed arm distinct="
            f"{results['guard_binary_probe']['licensed_arm_ptx_distinct']} "
            f"naive arm distinct="
            f"{results['guard_binary_probe']['naive_arm_ptx_distinct']}")
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
