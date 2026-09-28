"""Sub-step byte-identity gate for the SHIPPED CUDA off-diagonal ``update_E`` ON A FOLD.

THE CLAIM UNDER TEST, and it is a claim about a PREDICATE, not about a kernel.
``meep_gpu/cuda_kernels/offdiag_constitutive_kernels.py`` ships one certified
kernel, ``update_E_pml_real_offdiag``, emitted per row mask by
``offdiag_emitter.py``. This gate does not change one byte of either. It asks
whether the refusal ``covers_real_pml_offdiag_constitutive`` states at
``coverage.py:930-932`` --

    "mirror symmetry: the fold changes the stored extent, and the extent is what
     turns a cell index into a coefficient index"

-- is a refusal the arithmetic requires, and if so WHERE, plane by plane.

WHY NEITHER SIBLING VERDICT TRANSFERS, and the gate exists because of it. On
2026-08-19 the same sentence was retired from ``covers_real_pml_constitutive``
because that sub-step READS NO NEIGHBOUR. On 2026-08-20 the curl pair was
measured folded and SPLIT: exact on a folded METALLIC axis, 0/64 on a folded
PERIODIC one, every differing word on the ``_stored_past_owned`` top plane. THIS
FAMILY IS NEITHER. Its row product reads NEIGHBOURING cells of the OTHER
components' volumes (``stepping._offdiagonal_terms``, stepping.py:1243-1249), so
the element-wise argument does not apply; and it runs inside ``update_E``, which
never calls ``_mask_non_owned_cells`` (three call sites, stepping.py:397, :479,
:902 -- ``step_B``, ``step_D``, ``_bfast_term``), so the curl's ownership-mask
verdict does not apply either. A third measurement was owed and this is it.

WHAT THE READING SAYS BEFORE ANY DEVICE RUNS (four facts off ``stepping.py``):

1. THE FOLD ENTERS THROUGH EXACTLY ONE STENCIL LEG -- the PARTNER-axis
   ``_shift_down`` (stepping.py:1243-1245). On a MIRROR boundary that helper
   writes ``_symmetry_phase(component, axis, mirror_phase) * field[2]`` into
   stored cell 0 (stepping.py:1873-1875, ``MIRROR_SOURCE_INDEX = 2``). The
   kernel's ``coord_dn`` has two branches, PERIODIC (wrap to ``n-1``) and
   METALLIC (the ``-1`` zero ghost), and the mirror ghost is NEITHER.
2. THE OWN-AXIS LEG IS AN EXACT ZERO ON A FOLD, AND METALLIC ALREADY SPELLS IT.
   ``_offdiagonal_terms`` calls ``_shift_up(xp, product, own_axis, boundary,
   phase)`` with NO ``component`` and NO ``reflect_row``, so the MIRROR reflect
   branch (stepping.py:1826-1833) is not taken and the call falls through to
   ``shifted[_face(axis, -1)] = 0`` (:1834-1836) -- the same value
   ``coord_up``'s METALLIC arm produces and NOT the value its PERIODIC arm does.
3. THE WALL MASK ALREADY ABSTAINS ON A FOLD, ON BOTH PATHS.
   ``_mask_metallic_wall_coupling`` asks ``is_metallic(axis) and not
   is_mirrored(axis)`` (stepping.py:1282) and ``coverage.offdiag_wall_mask_flags``
   asks the same question of the same grid. Nothing has to change there, and the
   docstring at stepping.py:1273-1278 already carries the array path's own
   fold-equivalence measurement (8.3e-13 .. 4.7e-12) for why it abstains.
4. THE FOLD KIND DOES NOT SPLIT THIS FAMILY THE WAY IT SPLITS THE CURL.
   ``stepping._boundary_kinds`` resolves a folded axis to MIRROR whatever its
   declared termination is, and neither leg above reads ``_stored_past_owned``.
   So a folded METALLIC and a folded PERIODIC axis should behave IDENTICALLY
   here -- which is the opposite of the curl's verdict and is therefore measured
   rather than asserted.

So the reading predicts a THREE-WAY verdict decided by the ROW MASK, not by the
fold kind, and the gate is built to resolve it per case:

* a folded axis that is the PARTNER axis of a surviving row slot -> DIVERGE, on
  plane ``(folded axis, 0)`` of that row's component and nowhere else;
* a folded axis that is only some component's OWN axis -> exact under
  ``mirror_as_metallic``, diverging on plane ``(folded axis, -1)`` under
  ``mirror_as_periodic``;
* a folded axis in NEITHER role -> exact under both.

=============================================================================
THE SUBSTITUTION IS AN EXPERIMENTAL VARIABLE, NOT A DEFAULT
=============================================================================

A folded axis resolves to ``mirror``, and ``offdiag_boundary_codes`` refuses that
spelling (``BC_CODES`` has no entry, and the launcher deliberately lets the
KeyError escape). So a gate that runs the shipped kernel on a fold has to CHOOSE
a code for the folded axis, and the choice is part of what is measured. Both are
swept -- ``mirror_as_metallic`` and ``mirror_as_periodic`` -- because reporting
one alone would report a divergence caused by the harness's own choice as though
it were a property of the kernel.

=============================================================================
THE VERDICT IS TWO-SIDED
=============================================================================

Divergence alone licenses nothing: a kernel that diverged everywhere, including
where the fold cannot reach the arithmetic, would evidence a broken harness. So
every case carries a PREDICTION computed from its row mask and its folded axes
(:func:`predicted_diverges`), and the leg fails if measurement and prediction
disagree ANYWHERE. The sweep is built so both sides are populated: masks that put
a folded axis in the partner role, in the own-axis role only, and in neither.

=============================================================================
LOCALIZATION: WHAT MAKES A DIVERGENCE A FINDING RATHER THAN A FAILURE
=============================================================================

Every diverging case is decomposed by PLANE. For each output array the gate
computes the GHOST-DELTA planes -- the planes where the array path's resolved
ghost rule and the substituted kernel's disagree -- then counts the differing
words on those planes against the differing words that are not. "Diverged" is a
failure; "diverged, and every one of the N differing words lies on the
ghost-delta planes, none elsewhere" is a specification for the kernel change that
would fix it.

=============================================================================
WHAT WOULD MAKE THIS GATE VACUOUS, AND THE FLOORS THAT REFUSE IT
=============================================================================

* ``oracle_moved`` -- the fraction of output words the array path changed from
  the frozen input. Zero-init is a fixed point of this recurrence; a case that
  moved nothing is refused, not passed.
* ``coupling_is_live`` -- ``f_w_Ex - Dx*inv_eps_Ex``, i.e. what the off-diagonal
  rows actually contributed. A case whose coupling is identically zero cannot
  distinguish ANY defect in this family; it is testing the certified PLAIN
  constitutive kernel with extra steps.
* ``folded_axis_absorbs`` -- ``max|kps-1|``, ``max|kms-1|`` ON THE FOLDED AXIS.
  A folded axis whose coefficient vector is the identity everywhere cannot
  distinguish a coefficient-index error on the axis the fold moved.
* ``mirror_ghost_is_live`` -- the array path's mirror ghost is
  ``parity * g[2]``. Where stored row 2 of the partner volume is zero, the mirror
  ghost EQUALS the metallic zero ghost and an exact result would be a fact about
  the fixture. Measured per case, on the folded axes.
* ``ghost_delta_planes_are_live`` -- on a predicted-diverging case the oracle
  must have moved words on the planes the delta names.
* The mutation battery, including legs that MUST BE UNCAUGHT. A battery of
  only-must-be-caught legs scores identically whether the comparator works or has
  degenerated into failing everything. Every mutation leg is scoped to (spec,
  mask, substitution) triples whose UNMUTATED baseline is bit-identical -- the
  folded-curl gate's own lesson, where legs planted on an already-diverging
  baseline scored CAUGHT for free and both nulls read as violated.

=============================================================================
RUNNING IT
=============================================================================

Device (the GPU host, ONE verified-empty GPU; the cache dir MUST carry the policy
token because CuPy's disk-cache key is computed above the strip seam)::

    CUDA_VISIBLE_DEVICES=7 CUPY_CACHE_DIR=$OUT/cupy_cache/ftz_stripped \\
        python -u gate_cuda_folded_offdiag.py --backend cupy --product full \\
        --subnormal-policy keep --out $OUT/keep/gate.json

Laptop (no CUDA). The NumPy backend EXECUTES the emitted device source through
the slice test module's own evaluator -- one grammar, one copy, imported rather
than duplicated. IT COMPILES NOTHING AND CERTIFIES NOTHING; what it settles is
whether the harness can localize a divergence and whether it can fail::

    python -u gate_cuda_folded_offdiag.py --backend numpy --product reduced \\
        --out /tmp/folded_offdiag_local.json

One flushed line per case (the progress-reporting rule); the artifact is rewritten
atomically after every case, so an interrupted run keeps everything up to the
failure. Correctness only -- no throughput claim is made or possible.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_API = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_API not in sys.path:
    sys.path.insert(0, _REPO_API)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    import cupy as cp
except ImportError:  # laptop: the evaluator backend still runs
    cp = None

import probe_fused_kernel_bit_identity as probe  # noqa: E402
import gate_provenance  # noqa: E402
import gate_cuda_offdiag as flat  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid, Mirror  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.cuda_kernels import coverage, offdiag_emitter  # noqa: E402

log = probe.log
save = probe.save
bit_compare = probe.bit_compare
combine = probe.combine
to_host = probe.to_host
operand_census = probe.operand_census

SEED = 20260820

#: The six arrays this sub-step writes, and everything it touches. Imported from
#: the flat gate rather than respelled: two spellings of one output list is how a
#: gate quietly stops comparing an auxiliary.
OUTPUTS: Tuple[str, ...] = flat.OUTPUTS
STATE: Tuple[str, ...] = flat.STATE
COMPONENTS: Tuple[str, ...] = flat.COMPONENTS

#: Consecutive launches in the multi-step leg. 60 is the budget every certified
#: hand-CUDA record is cut at. "Identical for N steps" is a claim about N.
MULTI_STEP_BUDGET = 60

#: The two codes a folded axis can be handed, since ``mirror`` has none. BOTH are
#: swept: the choice is an experimental variable, and a divergence reported under
#: one alone could be a property of the choice rather than of the kernel.
SUBSTITUTIONS: Tuple[str, ...] = ("mirror_as_metallic", "mirror_as_periodic")

#: EVERY AXIS IS FOLDED SOMEWHERE IN THIS LIST, and both terminations appear on
#: each. The two terminations are carried because the reading says they should
#: NOT differ here -- the opposite of the curl's verdict -- and a prediction that
#: is not swept is not measured. Both plane parities, an odd full count, two
#: simultaneous planes and THREE simultaneous planes are all present: the
#: constitutive fold admission is capped at two planes because its gate stopped
#: there, and a cap costs corpus slots (2, on TestLDOS.test_ldos_3D).
FOLD_SPECS: Tuple[Dict[str, Any], ...] = (
    # The unfolded controls. Already certified; carried so the harness is shown
    # to reproduce the standing result under ITS OWN fixture rather than only
    # under the fixture that produced the record.
    {"label": "unfolded_periodic", "axes": "", "phase": 1,
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (8.0, 10.0, 12.0)},
    {"label": "unfolded_metallic", "axes": "", "phase": 1,
     "boundaries": ("metallic", "metallic", "metallic"), "cell": (8.0, 10.0, 12.0)},

    # Folded X: the SLOWEST-stride coefficient index. Both terminations, both
    # plane parities, an odd full count.
    {"label": "fold_X_periodic", "axes": "X", "phase": 1,
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (16.0, 8.0, 12.0)},
    {"label": "fold_X_metallic", "axes": "X", "phase": 1,
     "boundaries": ("metallic", "periodic", "periodic"), "cell": (16.0, 8.0, 12.0)},
    {"label": "fold_X_periodic_odd_plane", "axes": "X", "phase": -1,
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (16.0, 8.0, 12.0)},
    {"label": "fold_X_periodic_odd_count", "axes": "X", "phase": 1,
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (17.0, 8.0, 12.0)},

    # Folded Y, both terminations.
    {"label": "fold_Y_periodic", "axes": "Y", "phase": 1,
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (8.0, 16.0, 12.0)},
    {"label": "fold_Y_metallic", "axes": "Y", "phase": -1,
     "boundaries": ("periodic", "metallic", "periodic"), "cell": (8.0, 16.0, 12.0)},

    # Folded Z: the FASTEST-stride coefficient index. A decomposition defect that
    # confuses it with X is only visible if both are folded somewhere in the sweep.
    {"label": "fold_Z_periodic", "axes": "Z", "phase": -1,
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (8.0, 12.0, 16.0)},
    {"label": "fold_Z_metallic", "axes": "Z", "phase": 1,
     "boundaries": ("periodic", "periodic", "metallic"), "cell": (8.0, 12.0, 16.0)},

    # A FOLD BESIDE A LIVE METALLIC WALL, which no other spec here carries: a
    # folded axis always reports ``wm = 0`` (``is_metallic and not is_mirrored``),
    # so a sweep of folded specs alone leaves every wall flag zero and the wall
    # mask untested on a fold. That is exactly the seam
    # ``offdiag_wall_mask_flags`` was kept separate from ``bc_*`` for: "the day a
    # fold is admitted, conflating the two zeroes a plane MEEP steps".
    {"label": "fold_X_wall_Y", "axes": "X", "phase": 1,
     "boundaries": ("periodic", "metallic", "periodic"), "cell": (16.0, 8.0, 12.0)},
    {"label": "fold_Z_wall_X", "axes": "Z", "phase": 1,
     "boundaries": ("metallic", "periodic", "periodic"), "cell": (8.0, 12.0, 16.0)},

    # TWO PLANES AT ONCE, one termination each way and both the same way.
    {"label": "fold_XY_mixed", "axes": "XY", "phase": 1,
     "boundaries": ("periodic", "metallic", "periodic"), "cell": (16.0, 16.0, 10.0)},
    {"label": "fold_XZ_periodic", "axes": "XZ", "phase": -1,
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (16.0, 10.0, 16.0)},

    # THREE PLANES AT ONCE, both terminations. The constitutive family's
    # admission is capped at two because its gate stopped there; this one does
    # not have to be.
    {"label": "fold_XYZ_periodic", "axes": "XYZ", "phase": 1,
     "boundaries": ("periodic", "periodic", "periodic"), "cell": (16.0, 16.0, 16.0)},
    {"label": "fold_XYZ_metallic", "axes": "XYZ", "phase": -1,
     "boundaries": ("metallic", "metallic", "metallic"), "cell": (17.0, 16.0, 16.0)},
)

#: The row masks, CHOSEN so that a folded axis lands in each of its three roles.
#: A sweep over corpus masks alone would answer one third of the question and
#: report it as the whole: on a folded Z the corpus mask puts Z in NEITHER role
#: and the kernel is exact, which read alone would license a fold outright.
ROW_MASK_SPECS: Tuple[Dict[str, Any], ...] = (
    {"label": "full_tensor", "mask": (1, 1, 1, 1, 1, 1),
     "why": "every axis is some live slot's partner: the fold is always reachable"},
    {"label": "corpus_15row", "mask": (1, 0, 0, 1, 0, 0),
     "why": "the mask the 186-row corpus drives -- slots (Ex,Ey) and (Ey,Ex), so "
            "partner axes X and Y are live and Z is in NEITHER role"},
    {"label": "row_Ex_only", "mask": (1, 1, 0, 0, 0, 0),
     "why": "one row component: X is its OWN axis and never its partner, so a "
            "folded X discriminates the own-axis up shift alone"},
    {"label": "single_Ey_Ez", "mask": (0, 0, 1, 0, 0, 0),
     "why": "one slot: own axis Y, partner axis Z -- a folded X is in neither "
            "role and the kernel must be exact under BOTH substitutions"},
)

#: The reduced product's specs, NAMED. It has to carry an unfolded control, both
#: fold terminations, two folded AXES (slowest and fastest coefficient stride)
#: and a declared metallic wall beside a fold -- otherwise the wall
#: discriminators have one arm and the axis question is answered on one axis.
REDUCED_SPEC_LABELS: Tuple[str, ...] = (
    "unfolded_metallic", "fold_X_periodic", "fold_X_metallic",
    "fold_Z_periodic", "fold_X_wall_Y", "fold_Z_wall_X")

#: 0.5 is exactly representable in float32 and 0.35 is not. Only the second can
#: distinguish a contracted expression from an uncontracted one, so the guard
#: control is scored at the inexact one. The courant also moves dt, which moves
#: every PML coefficient, so it is a real second draw of the tables.
COURANTS: Tuple[float, ...] = (0.5, 0.35)
INEXACT_COURANT = 0.35

VALUE_CLASSES: Tuple[str, ...] = ("uniform", "subnormal_band")

#: Spatially VARYING row coefficients: a uniform coefficient is what makes the
#: registration rather than merely the rounding invisible.
ROW_FORM = "varying"

#: ``--fmad=false`` is CORRECTNESS on this sub-step, not tuning. The control is
#: EXPECTED TO DIVERGE at the inexact courant; if it does not, the guard is
#: decorative here and the record must say so rather than implying evidence it
#: does not have.
GUARD_SETS: Tuple[Tuple[str, Tuple[str, ...], bool], ...] = flat.GUARD_SETS


# ---------------------------------------------------------------------------
# The fold's three roles -- the prediction, computed rather than tabulated
# ---------------------------------------------------------------------------

def _axis_of(component: str) -> int:
    """0/1/2 for Ex/Ey/Ez -- the component's own axis."""
    return "xyz".index(component[1])


def fold_roles(grid, mask: Sequence[int]) -> Dict[str, Any]:
    """Which role, if any, each folded axis plays in this row mask.

    THE MIRROR RULE ENTERS ONLY THROUGH THE PARTNER-AXIS DOWN SHIFT
    (stepping.py:1243-1245), and a component's own axis is never its own
    partner (``OFFDIAG_TRANSVERSE_PARTNERS`` is own+1 and own+2), so the two
    roles are disjoint per slot and both are counted.
    """
    folded = tuple(axis for axis in range(3) if grid.is_mirrored(axis))
    partner_axes, own_axes = set(), set()
    for slot, (row, partner) in enumerate(coverage.OFFDIAG_ROW_SLOTS):
        if not mask[slot]:
            continue
        own_axes.add(_axis_of(row))
        partner_axes.add(_axis_of(partner))
    return {
        "folded_axes": list(folded),
        "live_partner_axes": sorted(partner_axes),
        "live_own_axes": sorted(own_axes),
        "fold_is_a_live_partner_axis": any(a in partner_axes for a in folded),
        "fold_is_a_live_own_axis": any(a in own_axes for a in folded),
    }


def predicted_diverges(roles: Dict[str, Any], substitution: str) -> bool:
    """The two-sided prediction, per case.

    ``mirror_as_metallic`` reproduces the array path's own-axis leg exactly (both
    write an exact zero into the top plane), so only the partner-axis mirror
    ghost can differ. ``mirror_as_periodic`` wraps that plane instead, so a folded
    axis in EITHER role diverges.
    """
    if not roles["folded_axes"]:
        return False
    if substitution == "mirror_as_metallic":
        return bool(roles["fold_is_a_live_partner_axis"])
    return bool(roles["fold_is_a_live_partner_axis"]
                or roles["fold_is_a_live_own_axis"])


def ghost_delta_planes(component: str, grid, mask: Sequence[int],
                       substitution: str) -> List[List[int]]:
    """The planes where the array path's ghost rule and the kernel's disagree.

    ``(axis, index)`` with index 0 for the near plane and -1 for the last stored
    slot -- the same convention the folded-curl gate's mask delta uses, so the
    two families' localization tables read alike.

    * partner-axis leg: the array path writes ``parity * g[2]`` at cell 0 of a
      folded axis (stepping.py:1873-1875); ``coord_dn`` writes the metallic zero
      or the periodic wrap. Either way it differs, so the plane is named under
      BOTH substitutions.
    * own-axis leg: the array path writes an exact 0 at the last stored slot
      (stepping.py:1834-1836, reached because ``_offdiagonal_terms`` passes no
      ``component`` and no ``reflect_row``). ``coord_up``'s METALLIC arm writes
      the same zero and its PERIODIC arm wraps, so the plane is named only under
      ``mirror_as_periodic``.
    """
    planes: List[Tuple[int, int]] = []
    live = [(row, partner) for slot, (row, partner)
            in enumerate(coverage.OFFDIAG_ROW_SLOTS)
            if mask[slot] and row == component]
    if not live:
        return []
    for _row, partner in live:
        axis = _axis_of(partner)
        if grid.is_mirrored(axis):
            planes.append((axis, 0))
    own = _axis_of(component)
    if grid.is_mirrored(own) and substitution == "mirror_as_periodic":
        planes.append((own, -1))
    return [list(plane) for plane in sorted(set(planes))]


def wall_mask_bites(mask: Sequence[int], walls: Sequence[int]) -> bool:
    """Does the metallic wall mask change ANY word on this (mask, grid) pair?

    A LIVE WALL FLAG IS NOT ENOUGH, and assuming it was is what made the first
    full run of this gate report both wall discriminators DISCRIMINATOR VIOLATED
    at 3/24. ``_mask_metallic_wall_coupling`` zeroes face 0 of the axes on which
    THE COMPONENT'S Yee shift is 0 (:data:`coverage.OFFDIAG_WALL_MASK_AXES`), so
    ``wm_y`` reaches Ex and Ez and never reaches Ey. A grid that declares a wall
    on an axis no live row component masks against is one where dropping the mask
    is a genuine NULL, and scoring it a miss would be a statement about the
    fixture rather than about the kernel.
    """
    for slot, (row, _partner) in enumerate(coverage.OFFDIAG_ROW_SLOTS):
        if not mask[slot]:
            continue
        component = COMPONENTS.index(row)
        if any(walls[axis] for axis in coverage.OFFDIAG_WALL_MASK_AXES[component]):
            return True
    return False


def _plane_index(axis: int, index: int, shape) -> Tuple[Any, ...]:
    key: List[Any] = [slice(None)] * 3
    key[axis] = index if index >= 0 else shape[axis] - 1
    return tuple(key)


def plane_mask_array(planes: Sequence[Sequence[int]], shape) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for axis, index in planes:
        out[_plane_index(int(axis), int(index), shape)] = True
    return out


def localize(reference: np.ndarray, produced: Any, component: str, grid,
             mask: Sequence[int], substitution: str) -> Dict[str, Any]:
    """WHERE the two answers differ, decomposed against the ghost delta.

    This is what turns a divergence into a specification. A kernel that is wrong
    everywhere and a kernel that is wrong on exactly the planes whose ghost rule
    it does not implement are the same verdict at the ``bit_identical`` level and
    completely different findings.
    """
    a = np.ascontiguousarray(reference, dtype=np.float32).view(np.uint32)
    b = np.ascontiguousarray(to_host(produced), dtype=np.float32).view(np.uint32)
    differing = a != b
    planes = ghost_delta_planes(component, grid, mask, substitution)
    named = plane_mask_array(planes, a.shape)
    return {
        "differing_words": int(np.count_nonzero(differing)),
        "differing_on_ghost_delta_planes": int(np.count_nonzero(differing & named)),
        "differing_elsewhere": int(np.count_nonzero(differing & ~named)),
        "ghost_delta_planes": planes,
        "words_on_ghost_delta_planes": int(np.count_nonzero(named)),
    }


# ---------------------------------------------------------------------------
# The fixture
# ---------------------------------------------------------------------------

def build(xp, spec: Dict[str, Any], courant: float):
    """A frozen ``(fields, layer, grid)`` triple for one fold spec.

    THE THICKNESS RULE IS THE SIBLING FOLD GATES', not a new one: skip an axis
    too thin to hold a layer, and on a MIRRORED axis ask for the HIGH face only.
    ``PML._resolve_mirror_faces`` (pml.py:387-415) refuses a named low face on a
    folded axis outright -- cell 0 is the mirror plane, a boundary condition
    rather than a wall -- and ``stepping._require_consistent_pml``
    (stepping.py:2305) raises if the two disagree. Asking for the high face is
    also what keeps ``folded_axis_absorbs`` above its floor.
    """
    planes = tuple(Mirror(name, spec["phase"]) for name in spec["axes"])
    grid = Grid(resolution=1.0, cell_size=tuple(spec["cell"]),
                boundaries=tuple(spec["boundaries"]), symmetry=planes,
                xp=xp, courant=courant)
    thickness = tuple(
        (0, 0) if grid.shape[axis] < 6
        else (0, 2) if grid.is_mirrored(axis)
        else (2, 2)
        for axis in range(3))
    layer = PML(grid=grid, thickness=thickness)
    fields = Fields(grid=grid)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    return fields, layer, grid


def codes_for(grid, substitution: str) -> Tuple[Tuple[int, int, int], Tuple[str, ...]]:
    """The three ``bc_*`` arguments, with the folded axes SUBSTITUTED.

    ``offdiag_boundary_codes`` would raise here -- ``BC_CODES`` has no ``mirror``
    entry and the launcher deliberately lets the KeyError escape rather than
    defaulting, because a default would be a silent wrong ghost rule. Choosing
    one is exactly what this gate is measuring, so the choice is named per case
    and both are swept.
    """
    kinds = coverage.real_pml_boundary_kinds(grid)
    stand_in = "metallic" if substitution == "mirror_as_metallic" else "periodic"
    codes = tuple(coverage.BC_CODES[stand_in if kind == "mirror" else kind]
                  for kind in kinds)
    return codes, tuple(kinds)


def folded_axis_absorbs(layer, grid) -> Dict[str, Any]:
    """Does the FOLDED axis's HALF-INTEGER coefficient profile beat the identity?

    ``kps = kms = 1`` is the interior pass-through. An axis whose whole vector is
    that identity cannot distinguish a coefficient-index error on it, so a folded
    case at this floor is measuring every axis except the one the fold touched.
    """
    tables = flat.half_integer_tables(layer)
    out: Dict[str, Any] = {"folded_axes": [], "max_deviation": 0.0}
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
    """Is the array path's mirror ghost distinguishable from the metallic zero?

    The mirror ghost is ``parity * g[2]`` (stepping.py:1873-1875,
    ``MIRROR_SOURCE_INDEX = 2``). Where stored row 2 of the partner volume is
    identically zero the mirror ghost EQUALS the metallic zero ghost, and an
    exact result would be a fact about the fixture rather than about the kernel.
    Measured on the D primaries, which are the volumes the coupling reads with no
    poles admitted (fields.py:1107-1138).
    """
    # ``MIRROR_SOURCE_INDEX`` is asked of ``stepping`` rather than spelled: a
    # second spelling of the row the mirror ghost reflects from is a second place
    # for it to drift, and this floor's whole job is to be about that row.
    source_row = int(stepping.MIRROR_SOURCE_INDEX)
    out: Dict[str, Any] = {"source_row": source_row, "axes": [],
                           "min_max_abs": None}
    for axis in range(3):
        if not grid.is_mirrored(axis):
            continue
        for name in ("Dx", "Dy", "Dz"):
            volume = to_host(getattr(fields, name))
            plane = volume[_plane_index(axis, source_row, volume.shape)]
            value = float(np.max(np.abs(plane.astype(np.float64))))
            out["axes"].append({"axis": axis, "volume": name, "max_abs": value})
            out["min_max_abs"] = (value if out["min_max_abs"] is None
                                  else min(out["min_max_abs"], value))
    out["meets_floor"] = (not out["axes"]) or (out["min_max_abs"] or 0.0) > 0.0
    return out


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------

def case_key(spec, mask_spec, substitution, courant, value_class, guard,
             steps) -> str:
    return "|".join([spec["label"], mask_spec["label"], substitution,
                     f"C{courant}", value_class, guard, f"n{steps}"])


def one_case(backend, xp, spec: Dict[str, Any], mask_spec: Dict[str, Any],
             substitution: str, courant: float, value_class: str,
             guard_label: str, guard: Sequence[str], steps: int,
             source_mutation: Optional[str] = None,
             host_mutation: Optional[str] = None) -> Dict[str, Any]:
    """One frozen state, run twice: the array path, then the kernel.

    THE ORACLE IS ``stepping.update_E`` ITSELF on real ``Grid``/``Fields``/``PML``
    objects with the rows installed through the PUBLIC installer, so the
    install-time validation, the zero-row drop and the stored-E switch are all in
    force. There is no second transcription on the oracle leg to drift.
    """
    started = time.time()
    key = case_key(spec, mask_spec, substitution, courant, value_class,
                   guard_label, steps)
    # A STABLE per-case seed. ``hash()`` of a str is salted per process, so a run
    # seeded from it is not re-runnable to the same bytes.
    rng = np.random.default_rng(
        (SEED + int(hashlib.sha256(key.encode("ascii")).hexdigest()[:8], 16))
        % (2 ** 32))
    mask = tuple(mask_spec["mask"])

    fields, layer, grid = build(xp, spec, courant)
    rows = flat.install_rows(fields, grid, mask, ROW_FORM, rng)
    host = flat.seed_state(fields, grid, value_class, rng)

    roles = fold_roles(grid, mask)
    codes, kinds = codes_for(grid, substitution)
    walls = coverage.offdiag_wall_mask_flags(grid)

    case: Dict[str, Any] = {
        "key": key,
        "label": spec["label"], "fold_axes": spec["axes"],
        "mirror_phase": spec["phase"], "boundaries": list(spec["boundaries"]),
        "row_mask_label": mask_spec["label"], "row_mask": list(mask),
        "installed_row_mask": list(coverage.offdiag_row_mask(fields)),
        "substitution": substitution, "courant": courant,
        "value_class": value_class, "guard": guard_label, "steps": steps,
        "backend": backend.name,
        "source_mutation": source_mutation, "host_mutation": host_mutation,
        "shape": [int(n) for n in grid.shape],
        "stored_past_owned": [int(grid.stored_cells(a)) - int(grid.owned_cells(a))
                              for a in range(3)],
        "mirrored": [bool(grid.is_mirrored(a)) for a in range(3)],
        "metallic": [bool(grid.is_metallic(a)) for a in range(3)],
        "boundary_kinds": list(kinds), "codes": list(codes), "walls": list(walls),
        "roles": roles,
        "operand_census": operand_census(host),
    }
    case["predicted_diverges"] = predicted_diverges(roles, substitution)
    case["wall_mask_bites"] = wall_mask_bites(mask, walls)

    # THE PREDICATE'S CURRENT ANSWER, recorded rather than acted on. This gate
    # exists to decide whether that answer should change, so it must not be the
    # thing that gates the measurement -- but a record that did not carry it
    # could not show WHICH clause the run was taken against.
    covered, reason = coverage.covers_real_pml_offdiag_constitutive(
        fields, layer, grid)
    case["predicate_today"] = {"covered": bool(covered), "reason": reason}

    absorbs = folded_axis_absorbs(layer, grid)
    case["folded_axis_absorbs"] = absorbs
    ghost = mirror_ghost_is_live(fields, grid)
    case["mirror_ghost_is_live"] = ghost
    if not absorbs["meets_floor"]:
        case["skipped"] = ("the folded axis's half-integer coefficient profile is "
                           "the identity everywhere; this case cannot distinguish "
                           "a coefficient index error on the axis the fold moved")
        case["seconds"] = round(time.time() - started, 3)
        return case
    if not ghost["meets_floor"]:
        case["skipped"] = ("stored row 2 of a partner volume is identically zero "
                           "on a folded axis, so the mirror ghost equals the "
                           "metallic zero ghost and an exact result would be a "
                           "fact about the fixture")
        case["seconds"] = round(time.time() - started, 3)
        return case

    frozen = flat.snapshot(fields)

    # --- leg 1: the oracle -------------------------------------------------
    for _ in range(steps):
        stepping.update_E(fields, layer)
        flat.advance_sources(fields)
    oracle = {name: to_host(getattr(fields, name)).copy() for name in OUTPUTS}

    moved = sum(int(np.count_nonzero(
        oracle[name].ravel().view(np.uint32)
        != np.ascontiguousarray(frozen[name], dtype=np.float32).ravel().view(np.uint32)))
        for name in OUTPUTS)
    total_words = sum(int(oracle[name].size) for name in OUTPUTS)
    diagonal = to_host(fields.Dx) * to_host(fields.inverse_epsilon_for("Ex"))
    coupling = to_host(fields.f_w_Ex) - diagonal
    case["oracle_moved_words"] = moved
    case["oracle_total_words"] = total_words
    case["oracle_moved"] = moved > 0
    case["coupling_max_abs"] = float(np.max(np.abs(coupling.astype(np.float64))))
    case["coupling_is_live"] = case["coupling_max_abs"] > 0.0

    # --- leg 2: the kernel, from the SAME frozen state ---------------------
    flat.restore(fields, frozen)
    backend.set_guard(guard)
    tables = flat.half_integer_tables(layer)
    tables, launch_codes, launch_walls = flat.apply_host_mutation(
        host_mutation if host_mutation in flat.HOST_MUTATIONS else None,
        layer, grid, tables, codes, walls)
    if host_mutation == "reverse_folded_axis_coefficients":
        tables = reversed_folded_axis_tables(layer, grid)
    launch_error = None
    try:
        for _ in range(steps):
            backend.launch(fields, layer, rows, mask, tables, launch_codes,
                           launch_walls)
            flat.advance_sources(fields)
    except Exception as exc:  # noqa: BLE001 - a refusal is a result, recorded
        launch_error = f"{type(exc).__name__}: {exc}"[:600]
    case["launch_error"] = launch_error

    if launch_error is None:
        parts = {name: bit_compare(oracle[name], getattr(fields, name))
                 for name in OUTPUTS}
        verdict = combine(parts)
        localization = {}
        for name in OUTPUTS:
            component = name[-2:]
            localization[name] = localize(oracle[name], getattr(fields, name),
                                          component, grid, mask, substitution)
        case["localization"] = localization
        case["differing_words"] = sum(v["differing_words"]
                                      for v in localization.values())
        case["differing_on_ghost_delta_planes"] = sum(
            v["differing_on_ghost_delta_planes"] for v in localization.values())
        case["differing_elsewhere"] = sum(v["differing_elsewhere"]
                                          for v in localization.values())
        case["ghost_delta_plane_words"] = sum(
            v["words_on_ghost_delta_planes"] for v in localization.values())
    else:
        verdict = {"bit_identical": False, "differing_floats": 0,
                   "total_floats": 0, "per_component": {}}
        case["differing_words"] = 0
        case["differing_on_ghost_delta_planes"] = 0
        case["differing_elsewhere"] = 0
        case["ghost_delta_plane_words"] = 0
    case["bit_identical"] = bool(verdict["bit_identical"]) and launch_error is None
    case["total_floats"] = verdict["total_floats"]
    case["max_ulp"] = verdict.get("max_ulp")

    # A predicted-diverging case is only measured if the oracle MOVED words on
    # the planes the delta names; a delta plane the oracle never touched cannot
    # distinguish the ghost rule from anything.
    delta_moved = 0
    for name in OUTPUTS:
        planes = ghost_delta_planes(name[-2:], grid, mask, substitution)
        if not planes:
            continue
        named = plane_mask_array(planes, tuple(grid.shape))
        before = np.ascontiguousarray(frozen[name], dtype=np.float32).view(np.uint32)
        after = np.ascontiguousarray(oracle[name], dtype=np.float32).view(np.uint32)
        delta_moved += int(np.count_nonzero((before != after) & named))
    case["oracle_moved_on_ghost_delta_planes"] = delta_moved
    case["ghost_delta_planes_are_live"] = bool(
        case["ghost_delta_plane_words"] == 0 or delta_moved > 0)

    case["agrees_with_prediction"] = bool(
        (not case["bit_identical"]) == case["predicted_diverges"])
    case["seconds"] = round(time.time() - started, 3)
    return case


def reversed_folded_axis_tables(layer, grid) -> Dict[str, Any]:
    """THE REFUSAL REASON, ARMED. The folded axis's half-integer profile reversed.

    "The fold changes the stored extent, and the extent is what turns a cell
    index into a coefficient index" is the predicate's stated reason for refusing
    a fold here. If that mattered to this sub-step, the coefficient a cell reads
    on the folded axis would be wrong. This makes it wrong -- IN BOUNDS, by
    reversing the vector rather than substituting a longer or shorter one, so the
    leg measures a WRONG ANSWER and not a memory fault.

    It is also what makes ``folded_axis_absorbs`` load-bearing: a folded axis
    whose profile is the identity everywhere reverses to itself, and the leg
    would come back UNCAUGHT for a reason about the fixture.
    """
    tables = dict(flat.half_integer_tables(layer))
    xp = grid.xp
    for axis, name in enumerate("xyz"):
        if not grid.is_mirrored(axis):
            continue
        for stem in ("kps", "kms"):
            key = f"{stem}_{name}"
            tables[key] = xp.ascontiguousarray(tables[key][::-1])
    return tables


def case_is_valid(case: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """A case that could not distinguish anything is REFUSED, not passed."""
    if case.get("skipped"):
        return False, case["skipped"]
    if not case.get("oracle_moved"):
        return False, ("the array path changed no output word: zero-init is a "
                       "fixed point of this recurrence and a case that moved "
                       "nothing certifies nothing")
    if not case.get("coupling_is_live"):
        return False, ("the off-diagonal coupling is identically zero: this case "
                       "cannot distinguish any defect in this family")
    if not case.get("ghost_delta_planes_are_live"):
        return False, ("the oracle moved no word on the planes the ghost delta "
                       "names; the rule under test is not distinguishable here")
    return True, None


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

def case_product(product: str, steps: int) -> List[Dict[str, Any]]:
    """The swept cases, before the guard axis multiplies them.

    An UNFOLDED spec is swept under ONE substitution: with no folded axis the two
    resolve to the same codes, and running both would double the control's weight
    in every summary for no second question.

    THE REDUCED SUBSET IS NAMED, NOT SLICED. A head slice of this list is all X
    folds and no declared wall, which leaves the wall discriminators with one arm
    and every fold-axis question answered on one axis.
    """
    specs = (FOLD_SPECS if product == "full"
             else [s for s in FOLD_SPECS if s["label"] in REDUCED_SPEC_LABELS])
    masks = ROW_MASK_SPECS
    courants = COURANTS if product == "full" else (INEXACT_COURANT,)
    classes = VALUE_CLASSES if product == "full" else ("uniform",)
    plan: List[Dict[str, Any]] = []
    for spec in specs:
        subs = SUBSTITUTIONS if spec["axes"] else SUBSTITUTIONS[:1]
        for mask_spec in masks:
            for substitution in subs:
                for courant in courants:
                    for value_class in classes:
                        plan.append({"spec": spec, "mask_spec": mask_spec,
                                     "substitution": substitution,
                                     "courant": courant,
                                     "value_class": value_class, "steps": steps})
    return plan


def summarize(cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    scored = [c for c in cases if c.get("case_is_valid")]
    per_guard: Dict[str, Dict[str, int]] = {}
    for case in scored:
        bucket = per_guard.setdefault(case["guard"], {
            "ran": 0, "identical": 0, "inexact_ran": 0, "inexact_identical": 0})
        bucket["ran"] += 1
        bucket["identical"] += int(case["bit_identical"])
        if case["courant"] == INEXACT_COURANT:
            bucket["inexact_ran"] += 1
            bucket["inexact_identical"] += int(case["bit_identical"])

    primary = [c for c in scored if c["guard"] == "fmad_false"]
    licensed = [c for c in primary if not c["predicted_diverges"]]
    diverging = [c for c in primary if c["predicted_diverges"]]
    folded_licensed = [c for c in licensed if c["fold_axes"]]

    by_fold_kind: Dict[str, Dict[str, int]] = {}
    for case in primary:
        if not case["fold_axes"]:
            continue
        kind = "folded_metallic" if any(
            case["metallic"][a] for a in case["roles"]["folded_axes"]
        ) else "folded_periodic"
        bucket = by_fold_kind.setdefault(kind, {"ran": 0, "identical": 0,
                                                "predicted_diverges": 0})
        bucket["ran"] += 1
        bucket["identical"] += int(case["bit_identical"])
        bucket["predicted_diverges"] += int(case["predicted_diverges"])

    census = {"values": 0, "subnormals": 0, "negative_zeros": 0, "zeros": 0}
    for case in scored:
        if case["value_class"] != "subnormal_band":
            continue
        for key in census:
            census[key] += case["operand_census"][key]

    # THE PREDICTION IS SCORED ON THE GUARDED LEG ALONE. The unguarded control's
    # whole job is to DIVERGE where the guarded leg is identical, so scoring it
    # against a prediction about the ghost rule would read the contraction guard
    # doing its work as a failed prediction about the fold.
    disagreements = [c["key"] for c in primary if not c["agrees_with_prediction"]]
    smeared = [c["key"] for c in diverging if c["differing_elsewhere"] > 0]
    return {
        "per_guard": per_guard,
        "by_fold_kind": by_fold_kind,
        "scored": len(scored),
        "refused": len(cases) - len(scored),
        "predicted_exact": len(licensed),
        "predicted_exact_identical": sum(1 for c in licensed if c["bit_identical"]),
        "predicted_exact_folded": len(folded_licensed),
        "predicted_exact_folded_identical": sum(
            1 for c in folded_licensed if c["bit_identical"]),
        "predicted_diverging": len(diverging),
        "predicted_diverging_that_diverged": sum(
            1 for c in diverging if not c["bit_identical"]),
        "prediction_disagreements": disagreements,
        "every_case_agrees_with_prediction": not disagreements,
        "differing_words_total": sum(c["differing_words"] for c in diverging),
        "differing_on_ghost_delta_planes": sum(
            c["differing_on_ghost_delta_planes"] for c in diverging),
        "differing_elsewhere": sum(c["differing_elsewhere"] for c in diverging),
        "divergence_is_localized": (not smeared) and bool(diverging),
        "cases_with_smeared_divergence": smeared,
        "subnormal_band_operands": census,
        "subnormal_band_is_non_vacuous": census["subnormals"] > 0,
        "every_case_moved_the_oracle": all(c["oracle_moved"] for c in scored),
        "every_case_had_live_coupling": all(c["coupling_is_live"] for c in scored),
        "min_coupling_max_abs": (min(c["coupling_max_abs"] for c in scored)
                                 if scored else None),
        "min_folded_axis_deviation": min(
            [c["folded_axis_absorbs"]["max_deviation"] for c in scored
             if c["folded_axis_absorbs"]["folded_axes"]] or [None]),
    }


def run_sweep(backend, xp, results: Dict[str, Any], out_path: str, product: str,
              steps: int, leg_name: str,
              guards: Sequence[Tuple[str, Tuple[str, ...], bool]] = GUARD_SETS
              ) -> Dict[str, Any]:
    cases: List[Dict[str, Any]] = []
    plan = case_product(product, steps)
    swept = [g for g in guards if backend.certifies or g[0] == "fmad_false"]
    total = len(plan) * len(swept)
    index = 0
    started = time.time()
    for guard_label, guard, _primary in swept:
        for spec in plan:
            index += 1
            case = one_case(backend, xp, spec["spec"], spec["mask_spec"],
                            spec["substitution"], spec["courant"],
                            spec["value_class"], guard_label, guard,
                            spec["steps"])
            valid, why = case_is_valid(case)
            case["case_is_valid"] = valid
            case["refused_because"] = why
            cases.append(case)
            log(f"[{leg_name}] {index}/{total} {case['key']} "
                f"shape={case['shape']} "
                f"pred={'DIVERGE' if case['predicted_diverges'] else 'EXACT'} "
                f"got={'IDENTICAL' if case.get('bit_identical') else 'DIVERGED'} "
                f"diff={case.get('differing_words')} "
                f"on_delta={case.get('differing_on_ghost_delta_planes')} "
                f"elsewhere={case.get('differing_elsewhere')}"
                + ("" if valid else f" REFUSED: {why[:60]}")
                + f" ({case['seconds']} s)")
            results[leg_name] = {"cases": cases, "summary": summarize(cases),
                                 "seconds": round(time.time() - started, 1)}
            save(results, out_path)
    return results[leg_name]


# ---------------------------------------------------------------------------
# The mutation battery
# ---------------------------------------------------------------------------
#
# SCOPED TO THE LICENSED ARM. A leg planted on a baseline that already diverges
# scores CAUGHT for free and every null reads as violated -- measured on the
# folded-curl gate's first full run, which reported both nulls NULL VIOLATED at
# 1/5, the 1 being its single already-diverging leg. The plan below is built from
# (spec, mask, substitution) triples whose UNMUTATED baseline is bit-identical
# AND which carry a fold, because a battery run only on the unfolded control
# would say nothing about the fold.

#: Every source defect this family owns, imported from the flat gate rather than
#: respelled, so the two records cannot disagree about what a defect IS.
SOURCE_MUTATIONS = flat.SOURCE_MUTATIONS
NULL_SOURCE_MUTATIONS: Tuple[str, ...] = tuple(flat.NULL_MUTATIONS)

#: Host defects. ``reverse_folded_axis_coefficients`` is THIS gate's own and is
#: the refusal reason armed as a defect; the other two are the flat gate's.
HOST_MUTATIONS: Tuple[str, ...] = ("reverse_folded_axis_coefficients",
                                   "swap_sub_lattice", "drop_the_wall_flags")

#: THE TWO WALL DISCRIMINATORS. Each must be a CATCH on a case whose grid
#: declares a metallic wall on an UNFOLDED axis and a NULL on one that declares
#: none -- which is what makes "the wall mask is caught" a statement about the
#: mask rather than about something else the metallic case changes. A folded axis
#: always reports ``wm = 0``, so without ``fold_X_wall_Y`` / ``fold_Z_wall_X`` in
#: the sweep the catch arm would be empty and both legs would score NULL for a
#: reason about the fixture.
WALL_DISCRIMINATORS: Tuple[str, ...] = ("drop_the_wall_mask",
                                        "drop_the_wall_flags")

#: Masks the transpose mutation may be armed on: it renames coefficient
#: parameters, so on a mask whose image under (row, partner) -> (partner, row) is
#: not itself it would name a parameter the emitted signature never declared and
#: fail to COMPILE -- a different event from a defect being caught.
TRANSPOSE_CLOSED = set(flat.TRANSPOSE_CLOSED_MASKS)

#: Legs that need a component with BOTH slots live.
BOTH_SLOTS_MASKS = {(1, 1, 1, 1, 1, 1), (1, 1, 0, 0, 0, 0)}


def mutation_plan(sweep_cases: Sequence[Dict[str, Any]], product: str
                  ) -> List[Dict[str, Any]]:
    """The FOLDED triples whose unmutated baseline was bit-identical.

    Read off the sweep that just ran rather than tabulated, so the battery cannot
    be armed on an arm the device did not actually certify in this same process.

    THE REDUCED SELECTION IS GREEDY, NOT A HEAD SLICE. Slicing the front of a
    case product landed the folded-curl gate's whole battery on its two UNFOLDED
    controls, so its one fold-specific mutation was skipped on every leg and
    scored 0/0 UNCAUGHT -- a harness defect that reads exactly like a kernel
    defect. This selection covers, in order: both wall arms, every distinct row
    mask, and every distinct folded axis.
    """
    seen, plan = set(), []
    for case in sweep_cases:
        if not case.get("case_is_valid") or case["guard"] != "fmad_false":
            continue
        if not case["bit_identical"] or not case["fold_axes"]:
            continue
        if case["courant"] != INEXACT_COURANT or case["value_class"] != "uniform":
            continue
        key = (case["label"], case["row_mask_label"], case["substitution"])
        if key in seen:
            continue
        seen.add(key)
        spec = next(s for s in FOLD_SPECS if s["label"] == case["label"])
        mask_spec = next(m for m in ROW_MASK_SPECS
                         if m["label"] == case["row_mask_label"])
        plan.append({"spec": spec, "mask_spec": mask_spec,
                     "substitution": case["substitution"],
                     "has_wall": bool(case["wall_mask_bites"]),
                     "folded_axes": tuple(case["roles"]["folded_axes"])})
    if product == "full":
        return plan
    chosen: List[Dict[str, Any]] = []
    covered = {"wall": set(), "mask": set(), "axes": set()}
    for entry in plan:
        wanted = (entry["has_wall"] not in covered["wall"]
                  or entry["mask_spec"]["label"] not in covered["mask"]
                  or entry["folded_axes"] not in covered["axes"])
        if not wanted:
            continue
        covered["wall"].add(entry["has_wall"])
        covered["mask"].add(entry["mask_spec"]["label"])
        covered["axes"].add(entry["folded_axes"])
        chosen.append(entry)
    return chosen


def leg_armable(name: str, entry: Dict[str, Any]) -> bool:
    mask = tuple(entry["mask_spec"]["mask"])
    if name == "transpose_the_chi1inv_term_table":
        return mask in TRANSPOSE_CLOSED
    if name == "drop_a_row_from_the_volume_sum":
        return mask in BOTH_SLOTS_MASKS
    return True


def expected_caught(name: str, case: Dict[str, Any]) -> bool:
    """Must THIS case diverge under THIS defect?

    Decided per case rather than per leg, because two of the defects are only
    defects where the grid declares a metallic wall on an unfolded axis. Scoring
    them "caught everywhere" would be false of the physics rather than of the
    kernel, and scoring them "uncaught" on a grid that has no wall would be a
    null nothing could have violated.
    """
    if name in WALL_DISCRIMINATORS:
        return bool(case["wall_mask_bites"])
    return name not in NULL_SOURCE_MUTATIONS


def run_mutations(backend, xp, results: Dict[str, Any], out_path: str,
                  product: str, steps: int,
                  sweep_cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    plan = mutation_plan(sweep_cases, product)
    legs: List[Dict[str, Any]] = []
    started = time.time()
    names = [(name, "host") for name in HOST_MUTATIONS] + \
            [(name, "source") for name in sorted(SOURCE_MUTATIONS)]
    for name, kind in names:
        leg_started = time.time()
        entries = [e for e in plan if kind == "host" or leg_armable(name, e)]
        transform = SOURCE_MUTATIONS[name] if kind == "source" else None

        if not entries:
            # NOT ARMABLE is not UNCAUGHT and not a null. The mutation was never
            # asked, so it gets no verdict -- and the count is surfaced, because
            # a defect with no licensed arm to plant it on is a HOLE in the
            # battery rather than a clean result.
            legs.append({"mutation": name, "kind": kind, "ran": 0, "caught": 0,
                         "verdict": "NOT ARMABLE ON ANY LICENSED CASE",
                         "as_required": None, "cases": [], "seconds": 0.0})
            log(f"[mutations] {kind}:{name}: NOT ARMABLE on any licensed case")
            continue

        sites = None
        classification: Optional[Dict[str, Any]] = None
        if transform is not None:
            sites = 0
            for entry in entries:
                pristine = backend.pristine_source(tuple(entry["mask_spec"]["mask"]))
                mutated_text, count = transform(pristine)
                sites += count
                if classification is None or classification["evaluator_sees_it"]:
                    classification = flat.classify_for_evaluator(pristine,
                                                                 mutated_text)
            backend.set_source_mutation(transform)

        # THE EVALUATOR BACKEND CANNOT SEE EVERY DEFECT, and a leg it cannot see
        # must say so rather than score a number. The classification is the flat
        # gate's, imported: an edit landing in the pinned helpers or the welded
        # tail is TRANSCRIBED rather than executed off device, and an edit that
        # rewrites a statement out of the anchored grammar is not applied at all.
        # On device the compiler executes every line and the question does not
        # arise, which is why the two backends' catch tables differ.
        if (not backend.certifies and classification is not None
                and not classification["evaluator_sees_it"]):
            backend.set_source_mutation(None)
            legs.append({"mutation": name, "kind": kind, "ran": 0, "caught": 0,
                         "verdict": f"NOT MEASURABLE ON THIS BACKEND "
                                    f"({classification['why']})",
                         "as_required": None, "mutation_sites": sites,
                         "evaluator_classification": classification,
                         "cases": [], "seconds": 0.0})
            log(f"[mutations] {kind}:{name}: NOT MEASURABLE on backend="
                f"{backend.name} ({classification['why']}); sites={sites}")
            continue
        before_compiles = len(backend.compile_log())

        cases: List[Dict[str, Any]] = []
        for entry in entries:
            case = one_case(backend, xp, entry["spec"], entry["mask_spec"],
                            entry["substitution"], INEXACT_COURANT, "uniform",
                            "fmad_false", ("--fmad=false",), steps,
                            source_mutation=name if kind == "source" else None,
                            host_mutation=name if kind == "host" else None)
            valid, why = case_is_valid(case)
            case["case_is_valid"] = valid
            case["refused_because"] = why
            cases.append(case)

        compiles = backend.compile_log()[before_compiles:]
        mutated_digests = set()
        if transform is not None:
            for entry in entries:
                text, _n = transform(backend.pristine_source(
                    tuple(entry["mask_spec"]["mask"])))
                mutated_digests.add(hashlib.sha256(text.encode("utf-8")).hexdigest())
        from_mutated = sum(1 for e in compiles
                           if e.get("source_sha256") in mutated_digests)
        if transform is not None:
            backend.set_source_mutation(None)
        backend.clear()

        scored = [c for c in cases if c["case_is_valid"]]
        for case in scored:
            case["expected_caught"] = expected_caught(name, case)
            case["was_caught"] = not case["bit_identical"]
            case["leg_as_required"] = case["expected_caught"] == case["was_caught"]
        caught = sum(1 for c in scored if c["was_caught"])
        catch_arm = [c for c in scored if c["expected_caught"]]
        null_arm = [c for c in scored if not c["expected_caught"]]
        if not scored:
            # NO LEGS is its own verdict, never UNCAUGHT. A mutation that was
            # never run has not been shown inert; it has not been asked.
            verdict = "NO LEGS"
            as_required = False
        elif name in WALL_DISCRIMINATORS and not (catch_arm and null_arm):
            # A DISCRIMINATOR WITH ONE ARM IS NOT A DISCRIMINATOR. Either half
            # alone is consistent with the mask being inert.
            verdict = "NO DISCRIMINATOR"
            as_required = False
        else:
            agreeing = sum(1 for c in scored if c["leg_as_required"])
            as_required = agreeing == len(scored)
            if not catch_arm:
                verdict = "NULL CONFIRMED" if as_required else "NULL VIOLATED"
            elif not null_arm:
                verdict = ("CAUGHT" if as_required
                           else "PARTIAL" if caught else "UNCAUGHT")
            else:
                verdict = ("CAUGHT WHERE A WALL IS DECLARED, NULL WHERE NONE IS"
                           if as_required else "DISCRIMINATOR VIOLATED")
        if transform is not None and sites == 0:
            as_required = False
            verdict = "REFUSED: the mutation matched nothing in any emitted source"
        if backend.certifies and transform is not None and from_mutated == 0:
            as_required = False
            verdict = ("REFUSED: the mutated bytes never reached the compiler; a "
                       "verdict for a mutation that never compiled is "
                       "indistinguishable from one that found nothing")

        record = {"mutation": name, "kind": kind,
                  "catch_arm": len(catch_arm), "null_arm": len(null_arm),
                  "ran": len(scored), "caught": caught,
                  "uncaught": len(scored) - caught, "verdict": verdict,
                  "as_required": bool(as_required), "mutation_sites": sites,
                  "compiles_on_this_leg": len(compiles),
                  "compiles_from_mutated_source": from_mutated,
                  "evaluator_classification": classification,
                  "cases": cases, "seconds": round(time.time() - leg_started, 1)}
        legs.append(record)
        log(f"[mutations] {kind}:{name}: {verdict} ({caught}/{len(scored)}) "
            f"as_required={as_required} sites={sites} "
            f"compiles_from_mutated={from_mutated} ({record['seconds']} s)")
        results["mutations"] = {
            "plan": [{"label": e["spec"]["label"],
                      "row_mask": e["mask_spec"]["label"],
                      "substitution": e["substitution"]} for e in plan],
            "legs": legs,
            "legs_as_required":
                f"{sum(1 for l in legs if l['as_required'])}/"
                f"{sum(1 for l in legs if l['as_required'] is not None)}",
            "legs_not_scored": sum(1 for l in legs if l["as_required"] is None),
            "legs_not_armable": sum(
                1 for l in legs if l["verdict"].startswith("NOT ARMABLE")),
            "legs_not_measurable_on_this_backend": sum(
                1 for l in legs if l["verdict"].startswith("NOT MEASURABLE")),
            "all_legs_as_required": all(
                l["as_required"] for l in legs if l["as_required"] is not None),
            "seconds": round(time.time() - started, 1)}
        save(results, out_path)
    return results["mutations"]


# ---------------------------------------------------------------------------
# The source manifest: a gate refuses to release against a tree it disagrees with
# ---------------------------------------------------------------------------

def manifest_reasons(out_path: str, imported: Dict[str, str]) -> List[str]:
    """Write or check ``source_sha256.txt`` beside the artifact.

    An append-only-in-scope campaign manifest. A later run in the same directory
    that imported DIFFERENT bytes for a path already recorded is refused: a
    single artifact directory mixing two trees is an artifact describing a run
    that never happened as one program.
    """
    path = os.path.join(os.path.dirname(out_path) or ".", "source_sha256.txt")
    existing: Dict[str, str] = {}
    reasons: List[str] = []
    if os.path.exists(path):
        with open(path, "r", encoding="ascii") as handle:
            for number, line in enumerate(handle.read().splitlines(), 1):
                if not line.strip():
                    continue
                parts = line.split(None, 1)
                if len(parts) != 2 or len(parts[0]) != 64:
                    reasons.append(f"source manifest line {number} is malformed")
                    continue
                existing[parts[1]] = parts[0]
    for name, digest in sorted(imported.items()):
        if name in existing and existing[name] != digest:
            reasons.append(f"campaign source manifest disagrees for {name}")
    merged = dict(existing)
    merged.update(imported)
    body = "".join(f"{digest} {name}\n" for name, digest in sorted(merged.items()))
    with open(path + ".tmp", "w", encoding="ascii") as handle:
        handle.write(body)
    os.replace(path + ".tmp", path)
    return reasons


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

LEG_NAMES = ("sweep", "multistep", "mutations")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--backend", default="auto",
                        choices=("auto", "cupy", "numpy"))
    parser.add_argument("--legs", default=",".join(LEG_NAMES))
    parser.add_argument("--product", default="auto",
                        choices=("auto", "full", "reduced"))
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--multi-step-budget", type=int, default=MULTI_STEP_BUDGET)
    parser.add_argument("--subnormal-policy", default=None,
                        choices=("keep", "flush", "match_meep", "ieee"))
    parser.add_argument("--import-meep-for-host-policy", action="store_true")
    args = parser.parse_args(argv)

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    legs = tuple(name for name in args.legs.split(",") if name)
    backend_name = args.backend
    if backend_name == "auto":
        backend_name = "cupy" if cp is not None else "numpy"
    product = args.product
    if product == "auto":
        product = "full" if backend_name == "cupy" else "reduced"

    results: Dict[str, Any] = {
        "gate": "cuda_folded_offdiag_constitutive",
        "kernel": offdiag_emitter.KERNEL_NAME,
        "question": ("does the SHIPPED off-diagonal update_E kernel, with the "
                     "fold refusal removed from covers_real_pml_offdiag_"
                     "constitutive and nothing else changed, serve a mirror "
                     "fold -- and if not, exactly which planes carry the "
                     "divergence?"),
        "substitutions": list(SUBSTITUTIONS),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "backend": backend_name,
        "certifies": backend_name == "cupy",
        "product": product,
        "legs_requested": list(legs),
        "seed": SEED,
        "multi_step_budget": args.multi_step_budget,
        "emitter_corpus_digest": offdiag_emitter.corpus_digest(),
        "gate_sha256": probe.source_digest(os.path.abspath(__file__)),
    }

    # THE ORDER IS THE WHOLE POINT: the observer wraps NVRTC BELOW the strip so it
    # records the options NVRTC really received, and the policy installs before
    # the first compile.
    if backend_name == "cupy":
        results["nvrtc_binary_observer"] = probe.install_nvrtc_binary_observer()
    if args.import_meep_for_host_policy:
        results["meep_import_for_host_policy"] = probe.import_meep_for_host_policy()
    if args.subnormal_policy:
        results["subnormal_policy_install"] = probe.install_subnormal_policy_for_run(
            args.subnormal_policy, _REPO_API)
    results["subnormal_policy_stamp"] = probe.subnormal_policy_stamp(_REPO_API)

    backend = flat.build_backend(backend_name)
    xp = cp if backend_name == "cupy" else flat._NumpyWearingCupysName()
    if backend_name == "cupy":
        results["environment"] = probe.device_info()
    save(results, out_path)

    if "sweep" in legs:
        run_sweep(backend, xp, results, out_path, product, args.steps, "sweep")
    if "multistep" in legs:
        # THE PRIMARY GUARD ONLY. The control's job is to show the guard is doing
        # work, and one launch per case already shows it; paying 60x for the same
        # demonstration buys nothing.
        run_sweep(backend, xp, results, out_path,
                  "reduced" if product == "full" else product,
                  args.multi_step_budget, "multistep", guards=GUARD_SETS[:1])
    if "mutations" in legs:
        run_mutations(backend, xp, results, out_path, product, args.steps,
                      results.get("sweep", {}).get("cases", []))

    if backend_name == "cupy":
        results["nvrtc_binaries"] = probe.nvrtc_binary_report()

    single = results.get("sweep", {}).get("summary", {})
    multi = results.get("multistep", {}).get("summary", {})
    mutations = results.get("mutations", {})
    control = single.get("per_guard", {}).get("default_no_options", {})
    verdict = {
        "licensed_arm_single_launch":
            f"{single.get('predicted_exact_folded_identical', 0)}"
            f"/{single.get('predicted_exact_folded', 0)}",
        "licensed_arm_multi_step":
            f"{multi.get('predicted_exact_folded_identical', 0)}"
            f"/{multi.get('predicted_exact_folded', 0)}",
        "predicted_diverging_that_diverged":
            f"{single.get('predicted_diverging_that_diverged', 0)}"
            f"/{single.get('predicted_diverging', 0)}",
        "every_case_agrees_with_prediction":
            single.get("every_case_agrees_with_prediction"),
        "prediction_disagreements": single.get("prediction_disagreements"),
        "differing_words_total": single.get("differing_words_total"),
        "differing_on_ghost_delta_planes":
            single.get("differing_on_ghost_delta_planes"),
        "differing_elsewhere": single.get("differing_elsewhere"),
        "divergence_is_localized": single.get("divergence_is_localized"),
        "guard_control_identical_at_inexact_courant":
            f"{control.get('inexact_identical', 0)}/{control.get('inexact_ran', 0)}",
        "guard_control_diverged": (control.get("inexact_ran", 0) > 0
                                   and control.get("inexact_identical", 0)
                                   < control.get("inexact_ran", 0)),
        "subnormal_band_is_non_vacuous": single.get("subnormal_band_is_non_vacuous"),
        "every_case_had_live_coupling": single.get("every_case_had_live_coupling"),
        "every_case_moved_the_oracle": single.get("every_case_moved_the_oracle"),
        "mutation_legs_as_required": mutations.get("legs_as_required"),
        "all_mutation_legs_as_required": mutations.get("all_legs_as_required"),
        # ON A CERTIFYING BACKEND BOTH MUST BE ZERO. The compiler executes every
        # line, so no defect is invisible for a reason of the harness; and a
        # defect with no licensed arm is a hole rather than a clean result.
        "mutation_legs_not_measurable":
            mutations.get("legs_not_measurable_on_this_backend"),
        "mutation_legs_not_armable": mutations.get("legs_not_armable"),
    }
    verdict["passed"] = bool(
        results["certifies"]
        and single.get("predicted_exact_folded", 0) > 0
        and single["predicted_exact_folded_identical"] == single["predicted_exact_folded"]
        and multi.get("predicted_exact_folded", 0) > 0
        and multi["predicted_exact_folded_identical"] == multi["predicted_exact_folded"]
        and single.get("every_case_agrees_with_prediction")
        and single.get("divergence_is_localized")
        and single.get("subnormal_band_is_non_vacuous")
        and single.get("every_case_had_live_coupling")
        and single.get("every_case_moved_the_oracle")
        and mutations.get("all_legs_as_required", False)
        and mutations.get("legs_not_measurable_on_this_backend", 1) == 0
        and mutations.get("legs_not_armable", 1) == 0)
    if not results["certifies"]:
        verdict["why_not_certified"] = (
            "the numpy backend executes the emitted source and compiles nothing; "
            "it exercises the harness, it does not measure NVRTC's output")
    results["verdict"] = verdict

    gate_provenance.stamp(results)
    reasons = manifest_reasons(out_path, results.get(
        gate_provenance.IMPORTED_KEY, {}))
    results["release"] = {
        "released": bool(verdict["passed"] and not reasons),
        "reasons": reasons or ([] if verdict["passed"] else ["gate did not pass"]),
    }
    save(results, out_path)

    log(f"[done] backend={backend_name} "
        f"licensed={verdict['licensed_arm_single_launch']} "
        f"multi={verdict['licensed_arm_multi_step']} "
        f"diverged={verdict['predicted_diverging_that_diverged']} "
        f"localized={verdict['divergence_is_localized']} "
        f"on_delta={verdict['differing_on_ghost_delta_planes']} "
        f"elsewhere={verdict['differing_elsewhere']} "
        f"mutations={verdict['mutation_legs_as_required']} "
        f"released={results['release']['released']}")
    if not results["certifies"]:
        return 0 if (single.get("every_case_agrees_with_prediction")
                     and mutations.get("all_legs_as_required", True)) else 1
    return 0 if results["release"]["released"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
