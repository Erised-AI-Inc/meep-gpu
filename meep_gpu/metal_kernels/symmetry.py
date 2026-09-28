"""Mirror-symmetry (folded-grid) Metal family — real float32 curl, fill, constitutive.

The Metal port of ``triton_kernels/symmetry.py``. Source of truth for the
ARITHMETIC is ``stepping.py``; source of truth for the STRUCTURE — the predicate
clauses, the plan shape, the refusals, the composition boundary — is the Triton
module, which is certified (``parity/meep_gpu/results/triton_symmetry_2026-08-10/``:
512/512 synthetic sub-steps, 7/7 real folded grids, 14/14 ghost fills, 14 mutations
caught, 1 measured null).

TEMPLATE FOLLOWED: :mod:`.bfast_curl`, read end to end before this file was
written. It is the closest shape in the package — a curl body that is the certified
``shaders._CURL_TEMPLATE`` plus one specialised block, registered on both curl
slots, with a CONSTITUTIVE COMPANION that adds no kernel and only re-admits the
certified body on a configuration the shipped predicate refuses for a clause that
does not apply to that sub-step. Everything that family does structurally — the
restated-and-pinned clause list with ONE inverted clause, the source-substitution
specialisation, ``_needle``-able template text, the ``register_arms()`` /
``ARMS = register_arms()`` tail — this file does the same way.

WHAT A FOLD IS, in one paragraph (transcription, with citations kept). An
``mp.Mirror`` plane HALVES an axis: the engine stores the half above the plane and
the discarded half is reconstructed by parity. ``stepping._boundary_kinds``
(stepping.py:2191-2196) resolves such an axis to MIRROR, and the fold OUTRANKS the
outer declaration, so a folded axis never reports metallic or periodic to the
stencils. The outer declaration survives in exactly one place —
``_stored_past_owned`` (stepping.py:1503-1518), TRUE on a folded PERIODIC axis and
FALSE on a folded METALLIC one — which is what splits MIRROR into the two codes
this file specialises on.

WHY THE CURL NEEDS ALMOST NOTHING, and it is a MEASURED claim on the Triton track
rather than an argument. A fold changes the ghost rule on the folded axis: the near
face becomes ``parity * field[2]`` (``_shift_down``, stepping.py:1865-1873, source
index ``MIRROR_SOURCE_INDEX = 2``) and the far face becomes
``parity * field[reflect_row]`` on a folded PERIODIC axis (``_shift_up``,
stepping.py:1815-1827) or an exact zero on a folded METALLIC one (:1828-1830). BOTH
GHOST VALUES ARE DEAD IN THE CURL: their only consumer is a plane one of the two
ownership masks zeroes. ``_shift_down`` is used by ``step_D`` only and its ghost
lands at stored cell 0, whose target has Yee shift 0 there — the cell the widened
cell-0 mask drops. ``_shift_up`` is used by ``step_B`` only and its ghost lands at
the LAST stored slot, whose target has Yee shift 1 there — the cell the fold's OWN
mask drops, but ONLY on a folded PERIODIC axis; on a folded METALLIC axis that top
plane IS stepped and the ghost there must be exactly ``+0.0``, which is what the
inherited metallic ternary already delivers (stepping.py:1788-1791).

So the whole device-code delta over the certified curl is ONE BLOCK, and this file
makes that structural rather than asserted: :func:`folded_curl_source` feeds
REDUCED codes (every mirror code mapped to ``METALLIC``) to the CERTIFIED emitters
``templates.ghost`` and ``templates.ownership_mask``, which is exactly the two
documented deltas — "both mirror codes take the metallic ghost branch" and "the
cell-0 mask widens from ``== METALLIC`` to ``!= PERIODIC``" — expressed as a call
rather than as a copy. The only text this file emits that the certified emitters do
not is :func:`folded_top_plane_mask`. ``test_metal_symmetry`` pins that by
CHARACTER, in both directions.

THE MASKS ARE INVISIBLE AT WHOLE-STEP GRANULARITY (symmetry.py:42-49 on the Triton
side): the driver's fill passes overwrite exactly the planes the two masks protect,
so a kernel carrying NEITHER mask is bytewise-identical after a complete step. A
whole-step check therefore CERTIFIES A MASK-LESS KERNEL AS CORRECT. Both legs are
needed and each catches what the other cannot — the sub-step leg for the masks, the
whole-step leg for the seams and the accumulating ``fu_*`` auxiliaries.

===========================================================================
WHAT IS METAL-SPECIFIC, AND IT IS NOT WHAT THE TRITON TRACK WOULD PREDICT
===========================================================================

THE PARITY WEIGHT IS A COMPILE-TIME SPECIALISATION HERE, NOT A RUNTIME SCALAR, and
that is the OPPOSITE of the Triton folded off-diagonal family's central platform
rule (``triton_kernels/folded_offdiag_update_e.py:93-107`` holds the ghost weight
RUNTIME because Triton lowers ``-x`` as ``0.0 - x`` and canonicalizes signed
zeros). MEASURED ON THIS HOST, 2026-08-16, torch 2.10.0 MPS / numpy 2.4.3 /
macOS 26.2 arm64, under the contraction guard (``shaders.contraction_pragma``
in its ``off`` mode — spelled there and nowhere else), against NumPy float32
``(+/-1) * x`` over a 30-word set carrying both signed zeros, twelve values across
the whole subnormal band, normals and +/-FLT_MAX:

    -x                              0/30    EXACT   (sign-bit op)
    x * -1.0f                       0/30    EXACT   (folded to a sign-bit op)
    0.0f - x                       13/30    WRONG   (12 subnormals + one -0.0)
    x * 1.0f                        0/30    EXACT   (folded away)
    RUNTIME w * x,  w = -1.0f       12/30    WRONG   (every subnormal flushes)
    RUNTIME w * x,  w = +1.0f       12/30    WRONG   (a runtime *1.0 is NOT identity)
    (at ? w : 1.0f) * x             12/30    WRONG   (flushes the WHOLE volume)
    at ? -x : x   (select VALUE)    0/30    EXACT
    plain copy                      0/30    EXACT

So on Metal the ODD spelling is ``-x`` and the EVEN spelling is A PLAIN COPY.
Neither ``0.0f - x`` nor any runtime weight is admissible, and a select that picks
the WEIGHT is as wrong as the weight itself while a select that picks the VALUE is
exact. That makes ``PHASE`` a source-specialisation axis in
:func:`mirror_ghost_fill_source`, where Triton can leave it a constexpr multiply.
The refuted spellings are gate mutations, not comments.

THE ``x * -1.0f`` DISAGREEMENT IN THE TREE IS SETTLED, AND IT WAS A NUMPY FACT.
``shaders.py:47-52`` records ``x * -1.0f`` at 13 mismatches on an 8045-value probe
while ``templates.py:51-56`` records it EXACT at 0/512; the fold's parity multiply
is precisely that expression, so it was re-measured here by VALUE CLASS rather than
inherited from either record. On finite (0/12), subnormal (0/10) and infinite (0/2)
operands ``-x`` and ``x * -1.0f`` are BYTE-IDENTICAL TO EACH OTHER and to NumPy. On
NaN they are byte-identical to each other and BOTH differ from NumPy 6/6 — NumPy's
``float32(-1.0) * nan`` leaves the sign bit alone, Metal's negation flips it. That
is the whole of the 13, it is a property of the NumPy reference and not of the two
Metal spellings, and it is unreachable here: a NaN in a folded field is a dead run,
not a parity question. ``0.0f - x`` additionally CANONICALIZES NaN to
``0x7FC00000`` (5/6), which is the Triton behaviour reproducing on this platform
for that one spelling — and is a second reason it is refused.

THE SUBNORMAL PRECONDITION BOUNDS THE ARITHMETIC HALF OF THIS FAMILY AND NOT THE
FILL, and that is measured rather than assumed for the family as a whole. On MPS
the float32 subnormal flush is native and has no lever, so every ARITHMETIC claim
rides on a checked subnormal-free precondition — but the fill performs no
arithmetic at all: its parity reduces to a sign-bit operation or to a plain copy,
and a subnormal CAN live in a Metal buffer (it is the arithmetic that flushes, not
the storage). Measured on this host 2026-08-16, on a folded grid scaled into the
band: THE FILL is identical to the array path with 576 subnormal operand words in
play, at 1e-38 AND at 1e-40, at BOTH parities (0 differing words each); THE CURL on
the same grid at 1e-38 carries 1,152 subnormal operand words in and DIVERGES in
1,104. That is the honest split — a family-wide precondition would understate what
the fill delivers, and stating it per sub-step is what makes it a measurement.
A folded run is also where the band is most likely to be reached: the fold puts a
deep-PML plane on the far face, which is exactly where tiny magnitudes live, and
the fill reads and writes exactly those planes.

NOTHING ELSE IN THIS FAMILY DIVIDES, TAKES A SQUARE ROOT, OR TAKES A MIN OR A MAX,
so the signed-zero ``min``/``max`` hazard (shaders.py:41-46) and ``fast::divide``
do not arise. Both ownership masks are selects writing the literal ``0.0f``,
matching the array path's ``curl[face] = 0`` (stepping.py:1943, :1949) — a Python
int zero into a float32 array, which is ``+0.0``. ``reflect_row`` is an integer
runtime scalar and the index arithmetic is exact.

THERE IS NO GENERATED-CODE AUDIT ON THIS BACKEND. ``torch.mps.compile_shader``
exposes no disassembly, so unlike the Triton track this family CANNOT refuse a
compile whose emitted code violates the policy, and it cannot establish that
the contraction guard was obeyed. The byte gate and the mutation legs
are the only arbiters here and they are BEHAVIOURAL: they catch a wrong answer, not
a wrong instruction. That is this certification's one weakness against the Triton
one and it is stated rather than buried.

===========================================================================
THE INVERTED CLAUSES — how this family stays disjoint from the other seven
===========================================================================

``registry.py``'s contract is that every family's grid-reason list carries the same
numbered questions and answers exactly one of them THE OTHER WAY. A new family
states its inversion rather than inheriting disjointness. Fold is the hardest case
because it composes with complex storage and with off-diagonal epsilon, so the
folded arms must invert against the plain arms AND against each other.

CLAUSE 5 IS THE INVERSION, IN BOTH DIRECTIONS AND ON EVERY SLOT:

* ``coverage._grid_reasons`` clause 5 (metal, coverage.py:168-174) REFUSES a fold
  — "a mirror plane is active (symmetry folding is not carried)" plus one
  "axis N is folded by a mirror plane" per folded axis. That covers ``pml_curl``
  and ``constitutive``, the two shipped families;
* ``bfast_curl._bfast_grid_reasons`` clause 5 refuses it again, by name;
* ``complex_fields`` and ``special_kz`` refuse it again, by name;
* HERE IT IS REQUIRED. :func:`folded_composition_curl_coverage` and
  :func:`folded_constitutive_coverage` append "no mirror plane is active: ..." when
  the grid carries no fold, so the two verdicts can never both admit.

CLAUSE 4 IS REPLACED, NOT WIDENED. The plain families refuse a fold a SECOND time
through the boundary whitelist (``kind not in COVERED_BOUNDARIES``, where
``COVERED_BOUNDARIES == ("periodic", "metallic")``). This family does not widen
that shared tuple — widening it would silently admit folds into every plain arm at
once — it replaces the clause entirely with :func:`folded_axis_kinds`.

THE STANDALONE PREDICATE IS NOT THE ROUTING PREDICATE.
:func:`folded_pml_curl_coverage` deliberately ADMITS an unfolded grid so the gate
can prove the reduction to the certified kernel. Registering THAT on a slot would
make every unfolded row ambiguous. The narrower
:func:`folded_composition_curl_coverage` is what :func:`register_arms` registers,
and the reason is the Triton module's: selecting between two valid products by
branch order would make the numerical method depend on composer order.

THE CURL'S CONTRACT AND THE CONSTITUTIVE'S ARE SEPARATE.
:func:`_folded_curl_grid_reasons` refuses conductivity because conductivity changes
the CURL recurrence; :func:`_folded_constitutive_grid_reasons` deliberately does
NOT, because conductivity does not change ``update_H``/``update_E``. Inheriting the
curl-only refusal would silently narrow an independent sub-step.

FOLD x CYLINDRICAL IS A THREE-WAY HAZARD, NOT A TWO-WAY ONE, and it is refused
TWICE on purpose. ``_boundary_kinds`` puts ``is_axis`` AHEAD of ``is_mirrored``
(stepping.py:2191-2196), so a folded r axis would report CYL_AXIS and the fold
would vanish silently; and ``_mirror_phases`` (stepping.py:2346-2366) puts
``(-1)**grid.m`` into the SAME SLOT the mirror phase occupies, so a predicate that
reads the phase without first refusing ``is_axis`` reads a cylindrical m-factor as
a mirror parity. The grid flag alone is not the inversion; the per-axis clause is
there too.

WHAT THIS FAMILY DOES NOT CLAIM, each refused by name rather than by omission:
complex64 storage (the folded-complex family's), a nonzero ``beta`` (the folded
special_kz family's), an off-diagonal chi1inv row on ``update_E`` (the folded
off-diagonal family's), a registered susceptibility on ``update_E`` (the folded
dispersive family's), BFAST, a nonzero ``k_point`` on any axis, and cylindrical
coordinates folded or not.

===========================================================================
THE RESIDENCY MODEL WAS SHORT TWO SLOTS AND THAT WAS BLOCKING
===========================================================================

``coverage.RESIDENCY_ORDER`` had no entry for ``fill_folded_far_ghosts_B/D`` — the
driver's FIFTH pass per half (driver.py:3287 and :3302), which WRITES B and D on
every folded PERIODIC run. A device mirror held across a step would have been stale
in exactly the plane the fold owns and the residency predicate would not have said
so. The two entries are added in this round's ``coverage.py`` change, and
``live_sub_steps`` now derives the fold's own seam passes from the grid rather than
from the source list: ``fill_symmetry_bc_*`` runs whenever ``grid.has_symmetry()``
whether or not any source exists, which the source-driven derivation could not see.

THE FILL IS TWO PASSES, NOT ONE, AND THEY MAY NOT BE FUSED — ``zero_metal_*`` SITS
BETWEEN THEM. The driver's order is ``step_B`` -> inject -> ``fill_symmetry_bc_B``
-> ``zero_metal_B`` -> ``fill_folded_far_ghosts_B`` -> ``update_H``
(driver.py:3282-3287; the D half at :3293-3302). The far pass READS a whole plane
at ``reflect_row``, which on a run with a live wall includes cells ``zero_metal``
has just cleared, so running the far pass before the wall clear is a different
answer. :class:`MirrorGhostFillPlan` therefore holds the two passes SEPARATELY
(:meth:`~MirrorGhostFillPlan.run_near` / :meth:`~MirrorGhostFillPlan.run_far`) and
:func:`mirror_ghost_fill_coverage` REFUSES outright the one configuration where the
fused ``run()`` would differ — a live far pass on a grid that also has a live
``zero_metal`` axis. See :func:`_wall_seam_reasons`; the refusal costs nothing on
the measured corpus, where no MIRROR_PERIODIC boundary triple carries a METALLIC
axis at all.

WIRED. Six arms: the folded curl on ``step_B``/``step_D``, the CERTIFIED
constitutive body re-admitted on ``update_H``/``update_E``, and the mirror fill on
``fill_B``/``fill_D``. ``fastpath.plan_fast_path`` still returns ``None`` on every
branch, which is the separate decision this port does not make.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

# ONE DEFINITION, IMPORTED — the same doctrine `metal_kernels.coverage` uses for
# every leaf helper. Each of these is pure Python over duck-typed engine objects
# and carries no Triton clause: the four boundary codes, the halved-origin source
# index, the Yee-shift table, the fill families, and the two engine readers
# (`folded_axis_kinds` routes through `stepping._boundary_kinds` and
# `stepping._stored_past_owned`; `_far_reflect_rows` through
# `stepping._far_reflect_rows`). Re-deriving ANY of them here would be a second
# place to get the fold's single point of failure wrong — the MIRROR_METALLIC /
# MIRROR_PERIODIC split, and the reflect row that is `stored - 2` at an even full
# count and `stored - 3` at an odd one.
from ..triton_kernels.symmetry import (  # noqa: F401 - re-exported deliberately
    CODE_METALLIC,
    CODE_MIRROR_METALLIC,
    CODE_MIRROR_PERIODIC,
    CODE_PERIODIC,
    GHOST_FILL_FAMILIES,
    MIRROR_SOURCE_INDEX,
    TARGET_IYEE,
    _far_reflect_rows,
    _has_real_fold,
    folded_axis_kinds,
)
from ..triton_kernels.coverage import (
    CONSTITUTIVE_SIDES,
    CURL_SUB_STEPS,
    CURL_TARGETS,
    Coverage,
    _call,
    _coefficient_reasons,
    _inverse_epsilon_reasons,
    _layout_reasons,
    _susceptibility_reasons,
    zero_metal_axes,
)
from ..triton_kernels.launch import SUB_STEPS
from . import shaders, templates
from .coverage import _metal_backend_reasons, _residency_declaration_reasons
from .device import Residency, compile_source
from .plans import KernelPlan

#: The family name the arm table carries. One spelling, so a refusal message, a
#: registry row and an artifact column cannot drift apart.
FAMILY = "folded"

#: The two mirror codes, as one tuple, so "is this axis folded" is asked one way.
MIRROR_CODES: Tuple[int, int] = (CODE_MIRROR_METALLIC, CODE_MIRROR_PERIODIC)

#: The two fill passes. ``near`` writes stored cell 0 of every shift-0 component on
#: a folded axis; ``far`` writes the last stored slot of every shift-1 component on
#: a folded PERIODIC axis. They are separate because ``zero_metal_*`` runs between
#: them (driver.py:3286 / :3301) — see the module docstring.
FILL_PASSES: Tuple[str, str] = ("near", "far")

__all__ = [
    "ARMS",
    "FAMILY",
    "FILL_PASSES",
    "FoldedPmlCurlPlan",
    "MIRROR_CODES",
    "MirrorGhostFillPlan",
    "compile_folded_curl",
    "compile_mirror_ghost_fill",
    "enumerate_folded_sources",
    "folded_axis_kinds",
    "folded_cell_zero_mask",
    "folded_composition_curl_coverage",
    "folded_constitutive_coverage",
    "folded_curl_source",
    "folded_ghost",
    "folded_pml_curl_coverage",
    "folded_top_plane_mask",
    "ghost_fill_axis_entries",
    "mirror_ghost_fill_coverage",
    "mirror_ghost_fill_source",
    "plan_folded_constitutive",
    "plan_folded_pml_curl",
    "plan_folded_pml_curl_from_arrays",
    "plan_mirror_ghost_fill",
    "plan_mirror_ghost_fill_from_arrays",
    "register_arms",
]


# ---------------------------------------------------------------------------
# The folded curl shader
# ---------------------------------------------------------------------------

#: The certified curl body with ONE addition and no other change: the
#: ``__TOP_MASK__`` slot between the cell-0 mask and the split-field recurrence.
#: Every other line — the guard, the index decode, the ghost gather, the curl
#: grouping, the cell-0 mask, the dsig/dsigu cycle and the six stores — is
#: character-for-character ``shaders._CURL_TEMPLATE``'s, and the two blocks that
#: LOOK fold-aware are emitted by ``shaders``' OWN emitters through reduced codes
#: (see :func:`folded_curl_source`).
#:
#: 20 BINDINGS, identical to the certified curl's: 9 volume pointers, 6 coefficient
#: vectors, 4 uints and dtdx. The fold adds no argument at all — the stored extent
#: is what carries it, and the PML coefficient vectors are built at that extent
#: already (measured on the Triton track: ``kms_y.shape == (1, 22, 1)`` on a grid
#: storing 22), which is why the coefficient index needs no fold-aware change.
_FOLDED_CURL_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

kernel void folded_pml_curl_step(
    device float*       f0      [[buffer(0)]],
    device float*       f1      [[buffer(1)]],
    device float*       f2      [[buffer(2)]],
    device float*       u0      [[buffer(3)]],
    device float*       u1      [[buffer(4)]],
    device float*       u2      [[buffer(5)]],
    device const float* g0      [[buffer(6)]],
    device const float* g1      [[buffer(7)]],
    device const float* g2      [[buffer(8)]],
    device const float* kmx     [[buffer(9)]],
    device const float* sinvx   [[buffer(10)]],
    device const float* kmy     [[buffer(11)]],
    device const float* sinvy   [[buffer(12)]],
    device const float* kmz     [[buffer(13)]],
    device const float* sinvz   [[buffer(14)]],
    constant uint&      nx      [[buffer(15)]],
    constant uint&      ny      [[buffer(16)]],
    constant uint&      nz      [[buffer(17)]],
    constant uint&      n_elem  [[buffer(18)]],
    constant float&     dtdx    [[buffer(19)]],
    uint idx [[thread_position_in_grid]])
{
    // The dispatch is sized from the first tensor argument's element count, so a
    // volume wider than n_elem would step cells the array path does not own. On a
    // folded axis that extent is the STORED one, which is what carries the fold.
    if (idx >= n_elem) { return; }

    int nxi = int(nx), nyi = int(ny), nzi = int(nz);
    int nyz = nyi * nzi;
    int ii  = int(idx);
    int k   = ii % nzi;
    int plane = ii / nzi;
    int j   = plane % nyi;
    int i   = plane / nyi;

    // --- the ghost rule, per axis (stepping._shift_up:1723 / _shift_down:1787) --
    // BOTH MIRROR CODES TAKE THE METALLIC BRANCH, serving an exact 0.0 past the
    // face. That is NOT the array path's ghost VALUE on a fold (`_shift_down`
    // serves `parity * field[2]`, `_shift_up` serves `parity * field[reflect_row]`
    // on a folded periodic axis) and it does not have to be: the only cell that
    // reads either ghost is a cell one of the two masks below zeroes. On a folded
    // METALLIC axis the far ghost IS exactly 0.0 and this branch is the array
    // path's own value (stepping.py:1781-1783, :1741-1744).
    int si = i __SHIFT__, sj = j __SHIFT__, sk = k __SHIFT__;
    bool vx = true, vy = true, vz = true;
__GHOST_X__
__GHOST_Y__
__GHOST_Z__

    int ox = si * nyz + j * nzi + k;
    int oy = i * nyz + sj * nzi + k;
    int oz = i * nyz + j * nzi + sk;

    // METALLIC serves an exact 0.0 past the wall -- Triton's `other=0.0`.
    float a   = g0[ii];
    float b   = g1[ii];
    float c   = g2[ii];
    float a_y = vy ? g0[oy] : 0.0f;
    float a_z = vz ? g0[oz] : 0.0f;
    float b_x = vx ? g1[ox] : 0.0f;
    float b_z = vz ? g1[oz] : 0.0f;
    float c_x = vx ? g2[ox] : 0.0f;
    float c_y = vy ? g2[oy] : 0.0f;

    // --- the curl (stepping._curl_from_operands:1601): DO NOT flatten these parens
    float curl0 = dtdx * ((c_y - c) + (b - b_z));
    float curl1 = dtdx * ((a_z - a) + (c - c_x));
    float curl2 = dtdx * ((b_x - b) + (a - a_y));

    // --- ownership mask, cell 0 (stepping._mask_non_owned_cells:1865) -----------
    // Every target whose Yee shift is 0 on a NON-PERIODIC axis. `_mask_non_owned_
    // cells` asks `is_mirrored or is_metallic or is_axis` (stepping.py:1898-1902),
    // and `_boundary_kinds` resolves exactly those three to a non-periodic kind
    // (the cylindrical one is refused by the predicate), so the certified emitter
    // is called with every mirror code reduced to METALLIC and the widening is a
    // call rather than a copy.
    bool at_x = (i == 0), at_y = (j == 0), at_z = (k == 0);
__MASK__

    // --- ownership mask, the TOP plane of a folded PERIODIC axis ----------------
    // The COMPLEMENT of the block above: every target whose Yee shift is 1 there.
    // `_mask_non_owned_cells`' `if iyee[axis] != 0` arm (stepping.py:1887-1897)
    // does this because that slot sits half a cell past MEEP's `big_corner`,
    // outside `owns` (vec.cpp:445-462), so the FILL PASS writes it and the curl
    // must not. This is THE ONLY TEXT IN THIS KERNEL THE CERTIFIED EMITTERS DO NOT
    // PRODUCE. On a folded METALLIC axis the top plane IS stepped and no line is
    // emitted here, which is why a sweep on one termination proves nothing about
    // the other.
    bool last_x = (i == nxi - 1), last_y = (j == nyi - 1), last_z = (k == nzi - 1);
__TOP_MASK__

    // --- split-field recurrence (stepping._apply_pml_update:1905) ---------------
    // dsig/dsigu follow vec.hpp's cycle_direction and are the same triple on both
    // sides: target 0 takes (y, z), target 1 (z, x), target 2 (x, y).
    float km_x = kmx[i], si_x = sinvx[i];
    float km_y = kmy[j], si_y = sinvy[j];
    float km_z = kmz[k], si_z = sinvz[k];

    float p0 = u0[ii];
    float n0 = ((p0 * km_y) - curl0) * si_y;
    float v0 = (((f0[ii] * km_z) + n0) - p0) * si_z;

    float p1 = u1[ii];
    float n1 = ((p1 * km_z) - curl1) * si_z;
    float v1 = (((f1[ii] * km_x) + n1) - p1) * si_x;

    float p2 = u2[ii];
    float n2 = ((p2 * km_x) - curl2) * si_x;
    float v2 = (((f2[ii] * km_y) + n2) - p2) * si_y;

    u0[ii] = n0; u1[ii] = n1; u2[ii] = n2;
    f0[ii] = v0; f1[ii] = v1; f2[ii] = v2;
}
"""  # stepping.py live lines for the frozen device-text citation(s) in this string: 1781-1783->1828-1830, 1741-1744->1788-1791, 1898-1902->1945-1949, 1887-1897->1934-1944


def _reduced_codes(codes: Sequence[int]) -> Tuple[int, int, int]:
    """The four folded codes as the certified emitters' two.

    THIS FUNCTION IS THE FIRST OF THE TWO DOCUMENTED FOLD DELTAS, expressed as a
    mapping instead of as a copied emitter: "both mirror codes take the METALLIC
    ghost branch" and "the cell-0 mask widens from ``== METALLIC`` to
    ``!= PERIODIC``" are the SAME statement once every non-periodic code is spelled
    ``METALLIC`` on the way in. Feeding this to ``templates.ghost`` and
    ``templates.ownership_mask`` is what makes the folded curl's ghost gather and
    cell-0 mask CHARACTER-IDENTICAL to the certified kernel's rather than
    similar to them.
    """
    out = []
    for code in codes:
        code = int(code)
        if code == CODE_PERIODIC:
            out.append(shaders.PERIODIC)
        elif code in (CODE_METALLIC,) + MIRROR_CODES:
            out.append(shaders.METALLIC)
        else:
            raise ValueError(
                f"boundary code {code!r} is none of PERIODIC/METALLIC/"
                f"MIRROR_METALLIC/MIRROR_PERIODIC; folded_axis_kinds is the only "
                f"supported source of these codes")
    return (out[0], out[1], out[2])


def folded_ghost(axis: str, code: int, backward: bool) -> str:
    """One axis's ghost rule on a folded grid, through the CERTIFIED emitter.

    A mirror code takes the metallic branch — mask the out-of-range neighbour and
    serve ``other = 0.0`` — so the emitted text is ``templates.ghost``'s own, and
    an edit to the certified ghost rule reaches this family without a second edit.
    """
    reduced = _reduced_codes((code, code, code))[0]
    return templates.ghost(axis, reduced, backward)


def folded_cell_zero_mask(codes: Sequence[int], backward: bool) -> str:
    """``stepping._mask_non_owned_cells``' cell-0 arm, widened to ``!= PERIODIC``.

    Emitted by ``templates.ownership_mask`` over :func:`_reduced_codes`, so it is
    the certified emitter's own output. A folded axis and a metallic axis mask
    exactly the same cells here, which is what ``_mask_non_owned_cells``' single
    ``is_mirrored or is_metallic or is_axis`` test says (stepping.py:1945-1949).
    """
    return templates.ownership_mask(_reduced_codes(codes), backward)


#: Per direction, the (target, axis, flag) triples whose Yee shift on that axis is
#: ONE — the exact COMPLEMENT of ``shaders.ownership_mask``'s zero-shift pairs. The
#: B family's shifts are 1 on the two axes that are NOT its own (Bx:(0,1,1),
#: By:(1,0,1), Bz:(1,1,0)); the D family's are 1 on its OWN axis only (Dx:(1,0,0),
#: Dy:(0,1,0), Dz:(0,0,1)). That complementarity is why a sweep on one sub-step
#: proves nothing about the other: the near ghost lands on a cell-0-masked plane
#: and the far ghost on a top-plane-masked one, and the two sub-steps mask
#: DIFFERENT planes.
#:
#: DERIVED FROM ``TARGET_IYEE``, NOT SPELLED, and asserted against
#: ``shaders.ownership_mask``'s complement by ``test_metal_symmetry`` — a literal
#: here would be a fourth transcription of the Yee table.
_TOP_PLANE_FLAGS: Tuple[str, str, str] = ("last_x", "last_y", "last_z")


def _top_plane_pairs(backward: bool) -> Tuple[Tuple[int, int, str], ...]:
    """(target index, axis, predicate) for every Yee shift of ONE, in target order."""
    names = SUB_STEPS["step_D" if backward else "step_B"]["targets"]
    return tuple((target, axis, _TOP_PLANE_FLAGS[axis])
                 for target, name in enumerate(names)
                 for axis in range(3) if TARGET_IYEE[name][axis] == 1)


def folded_top_plane_mask(codes: Sequence[int], backward: bool,
                          zero: str = "0.0f") -> str:
    """``_mask_non_owned_cells``' top-plane arm — a folded PERIODIC axis ONLY.

    THE ONE BLOCK THIS FAMILY EMITS THAT THE CERTIFIED EMITTERS DO NOT. It fires
    only where ``stepping._stored_past_owned`` is True, which is exactly a folded
    axis whose outer declaration is periodic, at either full-count parity; on a
    folded METALLIC axis the stored array stops at ``big_corner`` and the top plane
    is owned and stepped, so masking it there would delete a real cell.

    ``zero`` is the literal the mask writes, defaulting to the real family's
    ``0.0f`` for the same reason ``shaders.ownership_mask`` takes the parameter: a
    complex folded family passes ``float2(0.0f, 0.0f)``, which is the array path
    assigning a complex zero to BOTH planes (stepping.py:1943, :1949).
    """
    codes = tuple(int(code) for code in codes)
    if len(codes) != 3:
        raise ValueError(f"codes must be a per-axis triple, got {codes!r}")
    lines = [f"    curl{target} = {flag} ? {zero} : curl{target};"
             for target, axis, flag in _top_plane_pairs(bool(backward))
             if codes[axis] == CODE_MIRROR_PERIODIC]
    return "\n".join(lines) or ("    // no folded PERIODIC axis: the top plane is "
                                "owned and stepped")


def folded_curl_source(codes: Sequence[int], backward: bool,
                       contract: str = shaders.CONTRACT_OFF) -> str:
    """The specialised ``folded_pml_curl_step`` source for one configuration.

    ``codes`` is the per-axis PERIODIC / METALLIC / MIRROR_METALLIC /
    MIRROR_PERIODIC quadruple :func:`folded_axis_kinds` resolves — NEVER a
    hand-built triple, because the MIRROR_METALLIC / MIRROR_PERIODIC split is this
    family's single point of failure and backwards on one axis is a plane of wrong
    values, not a crash. ``backward`` selects ``step_D``'s negated strides over
    ``step_B``'s forward ones. Both are baked into the string, as Triton bakes its
    constexprs, so no branch survives around a float expression.

    THE CORPUS DRIVES TWELVE OF THESE, not the static 128: six boundary triples
    appear across the 86 folded rows and each is specialised in both directions.
    """
    codes = tuple(int(code) for code in codes)
    if len(codes) != 3:
        raise ValueError(f"codes must be a per-axis triple, got {codes!r}")
    reduced = _reduced_codes(codes)
    return templates.substitute(_FOLDED_CURL_TEMPLATE, {
        "__CONTRACT__": templates.contraction_pragma(contract),
        "__SHIFT__": "- 1" if backward else "+ 1",
        "__GHOST_X__": templates.ghost("x", reduced[0], backward),
        "__GHOST_Y__": templates.ghost("y", reduced[1], backward),
        "__GHOST_Z__": templates.ghost("z", reduced[2], backward),
        "__MASK__": templates.ownership_mask(reduced, backward),
        "__TOP_MASK__": folded_top_plane_mask(codes, backward),
    })


def compile_folded_curl(codes: Sequence[int], backward: bool,
                        contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised folded curl entry point for one configuration."""
    return compile_source(
        folded_curl_source(codes, backward, contract)).folded_pml_curl_step


# ---------------------------------------------------------------------------
# The mirror ghost fill shader
# ---------------------------------------------------------------------------

#: ONE family's ghost plane on ONE axis for ONE pass, in one launch instead of up
#: to three strided whole-plane assignments.
#:
#: 8 BINDINGS: three volume pointers, four uints and the runtime reflect row. The
#: reflect row STAYS RUNTIME (it is per-axis and a single 3-D grid can carry both
#: terminations with different rows — Mirror(x)+Mirror(y) at n_full (20, 21) gives
#: stored (12, 13) with reflect (10, 10), which is stored-2 on x and stored-3 on
#: y), while the PARITY is COMPILE-TIME (measured: a runtime weight flushes every
#: subnormal on this backend at BOTH signs; see the module docstring).
_MIRROR_FILL_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

kernel void mirror_ghost_fill(
    device float*       f0          [[buffer(0)]],
    device float*       f1          [[buffer(1)]],
    device float*       f2          [[buffer(2)]],
    constant uint&      nx          [[buffer(3)]],
    constant uint&      ny          [[buffer(4)]],
    constant uint&      nz          [[buffer(5)]],
    constant uint&      n_plane     [[buffer(6)]],
    constant int&       reflect_row [[buffer(7)]],
    uint idx [[thread_position_in_grid]])
{
    // One thread per IN-PLANE position, and ONE PASS per launch. Every read and
    // every write a thread performs shares that thread's own `base`, so there is
    // no cross-thread hazard in this kernel at all: the near pass reads row 2 and
    // writes row 0 of its own column, the far pass reads `reflect_row` and writes
    // the last row of its own column. What the PREDICATE checks, rather than
    // assumes, is that those rows are DISTINCT and that the far pass's read row is
    // not a row the near launch already wrote.
    if (idx >= n_plane) { return; }

    int nxi = int(nx), nyi = int(ny), nzi = int(nz);
    int ii  = int(idx);
__PLANE__

__BODY__
}
"""

#: The flat-index decode of ONE plane of a C-contiguous (nx, ny, nz) volume,
#: specialised per axis. Integer and exact: no float rounds on this path, so a
#: spelling difference here cannot change a bit. ``(void)`` swallows the two
#: extents this axis does not use — a compile warning is not a correctness event,
#: but a source that emits one every launch buries the ones that are.
_PLANE_INDEX: Dict[int, str] = {
    0: ("    int stride = nyi * nzi;\n"
        "    int base   = ii;\n"
        "    int last   = nxi - 1;"),
    1: ("    int stride = nzi;\n"
        "    int base   = (ii / nzi) * (nyi * nzi) + (ii % nzi);\n"
        "    int last   = nyi - 1;\n"
        "    (void)nxi;"),
    2: ("    int stride = 1;\n"
        "    int base   = (ii / nyi) * (nyi * nzi) + (ii % nyi) * nzi;\n"
        "    int last   = nzi - 1;\n"
        "    (void)nxi;"),
}


def _parity_spelling(weight: int, operand: str) -> str:
    """The ONE place a parity is turned into Metal text, and it is MEASURED.

    ``weight`` is +1 or -1 — ``fields.mirror_parity(component, axis, phase)``,
    which collapses to ``+phase`` for a shift-0 component and ``-phase`` for a
    shift-1 one (fields.py:180-182, and the Yee-table collapse pinned at
    symmetry.py:365-371 on the Triton side).

    ON THIS BACKEND +1 IS A PLAIN COPY AND -1 IS ``-x``, AND NOTHING ELSE IS
    ADMISSIBLE. Measured 2026-08-16 against NumPy float32 ``(+/-1) * x`` over
    signed zeros, the whole subnormal band, normals and +/-FLT_MAX: ``-x`` 0/30 and
    a plain copy 0/30, against ``0.0f - x`` 13/30 (twelve subnormals plus one
    -0.0), a RUNTIME ``w * x`` 12/30 AT BOTH SIGNS — a runtime multiply by 1.0 is
    not the identity here — and ``(at ? w : 1.0f) * x`` 12/30, which flushes the
    whole volume rather than the ghost lane. ``x * -1.0f`` is also 0/30 and is
    byte-identical to ``-x`` on every value class including NaN; ``-x`` is used
    because it is the literal operation and needs no constant fold to be one.

    That is why ``PHASE`` is a SOURCE-SPECIALISATION axis in this family where the
    Triton twin can leave it a constexpr multiply, and it is the opposite of what
    porting ``triton_kernels/folded_offdiag_update_e.py``'s runtime-weight idiom on
    faith would have produced.
    """
    if int(weight) == 1:
        return operand
    if int(weight) == -1:
        return f"-{operand}"
    raise ValueError(f"a mirror parity is +1 or -1, got {weight!r}")


def mirror_ghost_fill_source(axis: int, phase: int, pass_name: str,
                             shifts: Sequence[int],
                             contract: str = shaders.CONTRACT_OFF) -> str:
    """One (axis, phase, pass, Yee-shift triple) specialisation of the fill.

    ``shifts`` is the three targets' Yee shift ON THIS AXIS, in kernel argument
    order — ``TARGET_IYEE[name][axis]`` for the family's three components. It is
    not free: ``(family, axis)`` determines it, and :func:`ghost_fill_axis_entries`
    is what reads it off the table.

    THE NEAR PASS writes stored cell 0 from stored cell ``MIRROR_SOURCE_INDEX``
    (= 2) with weight ``+phase``, for every component whose shift here is ZERO —
    MEEP's ``little_owned_corner0(c) = little_corner + 2 - iyee_shift(c)``, which
    on a halved grid (``io = -2``) excludes cell 0 for exactly the shift-0
    components. THE FAR PASS writes the last stored slot from ``reflect_row`` with
    weight ``-phase``, for every component whose shift here is ONE, and runs only
    on a folded PERIODIC axis.

    An empty body is REFUSED rather than emitted: a fill launch that writes nothing
    is a no-op that a gate comparing "before" to "after" would report as a pass.
    """
    if pass_name not in FILL_PASSES:
        raise ValueError(f"pass_name must be one of {FILL_PASSES}, got {pass_name!r}")
    if int(axis) not in _PLANE_INDEX:
        raise ValueError(f"axis must be 0, 1 or 2, got {axis!r}")
    if int(phase) not in (1, -1):
        raise ValueError(f"a mirror plane's phase is +1 or -1, got {phase!r}")
    shifts = tuple(int(s) for s in shifts)
    if len(shifts) != 3 or any(s not in (0, 1) for s in shifts):
        raise ValueError(f"shifts must be three Yee shifts of 0 or 1, got {shifts!r}")

    lines: List[str] = []
    if pass_name == "near":
        lines.append("    // near face (stepping._write_mirror_ghost:1450): "
                     "cell 0 = +phase * cell 2")
        weight = int(phase)
        for slot, shift in enumerate(shifts):
            if shift != 0:
                continue
            operand = f"f{slot}[base + {MIRROR_SOURCE_INDEX} * stride]"
            lines.append(f"    f{slot}[base] = {_parity_spelling(weight, operand)};")
    else:
        lines.append("    // far face (stepping._fill_folded_far_ghosts:1529): "
                     "last = -phase * cell reflect_row, folded PERIODIC only")
        weight = -int(phase)
        for slot, shift in enumerate(shifts):
            if shift != 1:
                continue
            operand = f"f{slot}[base + reflect_row * stride]"
            lines.append(f"    f{slot}[base + last * stride] = "
                         f"{_parity_spelling(weight, operand)};")
    if len(lines) == 1:
        raise ValueError(
            f"the {pass_name!r} pass on axis {axis} would write nothing for shifts "
            f"{shifts!r}: an empty fill launch is a no-op that a before/after "
            f"comparison reports as a pass, so it is refused rather than emitted")
    if pass_name == "near":
        lines.append("    (void)last; (void)reflect_row;")

    return templates.substitute(_MIRROR_FILL_TEMPLATE, {
        "__CONTRACT__": templates.contraction_pragma(contract),
        "__PLANE__": _PLANE_INDEX[int(axis)],
        "__BODY__": "\n".join(lines),
    })


def compile_mirror_ghost_fill(axis: int, phase: int, pass_name: str,
                              shifts: Sequence[int],
                              contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised fill entry point for one configuration."""
    return compile_source(mirror_ghost_fill_source(
        axis, phase, pass_name, shifts, contract)).mirror_ghost_fill


def enumerate_folded_sources(contract: str = shaders.CONTRACT_OFF) -> Dict[str, str]:
    """Every shipped folded specialisation, keyed by a stable label.

    The shape ``shaders.enumerate_sources`` returns, so this family's gate records
    one sha256 per emitted source in its own provenance block. The curl's static
    product is 2 directions x 4^3 boundary quadruples = 128 and the fill's is
    2 families x 3 axes x 2 phases x 2 passes = 24; the CORPUS drives 12 and at
    most 20 of those respectively, which is the honest size of this port.
    """
    out: Dict[str, str] = {}
    codes = (CODE_PERIODIC, CODE_METALLIC, CODE_MIRROR_METALLIC, CODE_MIRROR_PERIODIC)
    for backward, name in ((False, "step_B"), (True, "step_D")):
        for cx in codes:
            for cy in codes:
                for cz in codes:
                    label = f"folded_pml_curl_step/{name}/{cx}{cy}{cz}"
                    out[label] = folded_curl_source((cx, cy, cz), backward, contract)
    for family, spec in GHOST_FILL_FAMILIES.items():
        targets = tuple(spec["targets"])
        for axis in range(3):
            shifts = tuple(TARGET_IYEE[n][axis] for n in targets)
            for phase in (1, -1):
                for pass_name in FILL_PASSES:
                    label = (f"mirror_ghost_fill/{family}/axis{axis}/"
                             f"phase{phase:+d}/{pass_name}")
                    out[label] = mirror_ghost_fill_source(
                        axis, phase, pass_name, shifts, contract)
    return out


# ---------------------------------------------------------------------------
# Coverage — positive refusal enumeration, with clause 5 INVERTED
# ---------------------------------------------------------------------------
#
# The clause numbering mirrors `coverage._grid_reasons` so the two can be diffed.
# Clause 5 is INVERTED: this product REQUIRES a mirror plane where every shipped
# kernel requires none, which is what makes an admitted overlap impossible on any
# slot. Clause 4 is REPLACED rather than widened — the shared `COVERED_BOUNDARIES`
# tuple stays ("periodic", "metallic") so no plain arm silently acquires the fold.
# Everything else is KEPT and RESTATED, because the shipped reason list is built
# inside a function whose clause 5 cannot be subtracted from outside.
# `test_metal_symmetry` runs this list against the Triton symmetry module's over a
# shared configuration matrix and asserts they agree once each side's own backend
# clause is removed, so the restatement cannot drift.


def _folded_axis_reasons(grid: Any, pml: Any) -> Tuple[Optional[Tuple[int, int, int]],
                                                       List[str]]:
    """Clauses 4 and 5 together, through the engine's own classifier.

    ``folded_axis_kinds`` is IMPORTED, not re-derived: it routes the ghost rule
    through ``stepping._boundary_kinds`` (where a fold outranks the declaration),
    splits MIRROR through ``stepping._stored_past_owned``, and CROSS-CHECKS that
    split against ``grid.is_metallic`` — two independent routes to the same fact,
    with a disagreement a refusal rather than a coin toss.
    """
    codes, reasons = folded_axis_kinds(
        grid, pml if (pml is not None and getattr(pml, "is_active", False)) else None)
    return codes, list(reasons)


def _requires_a_fold(grid: Any, codes: Optional[Sequence[int]],
                     what: str) -> List[str]:
    """CLAUSE 5, INVERTED — the whole of this family's disjointness.

    Asked TWICE and deliberately: once of the grid (``has_symmetry`` and some
    ``is_mirrored`` axis) and once of the RESOLVED CODES (some axis landed on a
    mirror code). The two can disagree — a grid can declare a mirror the boundary
    resolver reports as something else, which is precisely the cylindrical hazard
    (``_boundary_kinds`` puts ``is_axis`` ahead of ``is_mirrored``) — and a family
    that asked only the first would admit a run whose kernel carries no fold at all.
    """
    reasons: List[str] = []
    if not _has_real_fold(grid):
        reasons.append(f"no mirror plane is active: {what}")
    elif codes is not None and not any(int(code) in MIRROR_CODES for code in codes):
        reasons.append("no axis resolves to a mirror boundary: the grid declares a "
                       "fold that stepping._boundary_kinds does not report as one")
    return reasons


def _folded_curl_grid_reasons(fields: Any, pml: Any, grid: Any,
                              ) -> Tuple[Optional[Tuple[int, int, int]], List[str]]:
    """The clauses every Metal folded CURL predicate shares."""
    reasons: List[str] = []

    # 1. The Metal backend, and the host array module the mirrors copy from.
    #    IMPORTED from this package's own coverage module: one backend clause for
    #    the whole package, so a family cannot quietly admit a run on a host whose
    #    subnormal policy the MPS executor refuses.
    reasons.extend(_metal_backend_reasons(grid))

    # 2. Real storage. The folded recurrence is the same shape in complex64, the
    #    STORAGE is not, and under complex the FILL's parity multiply stops being a
    #    sign flip and becomes a full complex multiply by (+/-1, +0) with its zero
    #    cross terms — measured on the Triton track to diverge in 8 of 128
    #    engineered words at BOTH parities. That is the folded-complex family's.
    if getattr(fields, "force_complex_fields", False):
        reasons.append("force_complex_fields=True: under complex storage the fold's "
                       "parity multiply is a full complex multiply with zero cross "
                       "terms, not a sign flip; that is the folded complex family")

    # 3. An absorber that actually absorbs. Without one the plain path is the
    #    bit-identical one (stepping._pml_is_active).
    if pml is None or not getattr(pml, "is_active", False):
        reasons.append("no active PML layer (this product implements the "
                       "split-field path only)")

    # 4/5. The ghost rules, INCLUDING the two folded ones. `COVERED_BOUNDARIES` is
    #      NOT consulted and NOT widened: widening that shared tuple would admit a
    #      fold into every plain arm at once. `folded_axis_kinds` replaces it.
    codes, fold_reasons = _folded_axis_reasons(grid, pml)
    reasons.extend(fold_reasons)

    # 6. Cartesian only, refused TWICE. `_boundary_kinds` puts `is_axis` AHEAD of
    #    `is_mirrored` (stepping.py:2191-2196), so a folded r axis reports CYL_AXIS
    #    and the fold vanishes silently; and `_mirror_phases` (stepping.py:2346-2366)
    #    puts (-1)**grid.m into the SAME SLOT the mirror phase occupies, so reading
    #    a phase without first refusing `is_axis` reads an m-factor as a parity.
    #    The grid flag alone is not the inversion.
    if getattr(grid, "cylindrical", False):
        reasons.append("cylindrical (Dcyl) coordinates are not carried")
    for axis in range(3):
        if _call(grid, "is_axis", axis, default=False):
            reasons.append(f"axis {axis} is the cylindrical r = 0 axis")

    # 7. k = 0 on EVERY axis. On a folded axis the engine refuses a nonzero k
    #    outright (`stepping._bloch_phases` raises with "a mirror plane reflects
    #    rather than repeating"; `driver._require_bloch_is_representable` refuses
    #    the zone edge too), so a kernel cannot lift what the array path will not
    #    run. A Bloch phase on ANOTHER axis is supported by the engine and refused
    #    here anyway, because it needs complex storage.
    if getattr(grid, "has_bloch", False):
        reasons.append(f"nonzero k_point {getattr(grid, 'k_point', None)!r}")
    k_point = getattr(grid, "k_point", (0.0, 0.0, 0.0))
    if any(float(component) != 0.0 for component in k_point):
        reasons.append(f"k_point {tuple(k_point)!r} is not exactly zero")

    # 10. No instantaneous nonlinearity. Beyond the Pade factor replacing the
    #     constitutive product, a nonlinear folded run with a LIVE far face is
    #     refused by the driver itself (`_require_folded_far_face_is_quiet`,
    #     driver.py:2986-3038): the transverse sums shift component PRODUCTS, which
    #     carry no single parity. Refused here regardless of the far face.
    if getattr(fields, "has_nonlinearity", False):
        reasons.append("chi2/chi3 is installed (the Pade constitutive factor is "
                       "not carried, and a nonlinear fold has no single parity)")

    # 11/12. BFAST adds a second additive curl term; beta adds out-of-plane
    #        couplings. BOTH LAND INSIDE THE FOLD'S MASK REACH — the increments are
    #        added at stepping.py:371-396 / :459-478 and masked at :397 / :479, and
    #        a fold WIDENS that mask from metallic-at-cell-0 to non-periodic-at-
    #        cell-0 PLUS folded-periodic-at-the-top-plane. So the pairings are not
    #        additive, they are a coordinated follow-up, and both are refused here.
    if getattr(grid, "bfast_active", False):
        reasons.append("BFAST is active (the second additive curl term is not "
                       "carried, and a fold widens the mask that term lands inside)")
    if float(getattr(grid, "beta", 0.0)) != 0.0:
        reasons.append(f"special_kz beta={getattr(grid, 'beta', 0.0)!r} is nonzero "
                       f"(the beta increment lands inside the fold's widened mask; "
                       f"that pairing is folded_beta's real arm, which CARRIES it)")

    # 9c. STORED E — the load-bearing invariant behind admitting dispersion on the
    #     curl. The curl differences the STORED E and H arrays; a run that
    #     recomputes E from D has nothing for it to difference.
    if not getattr(fields, "stores_E", False):
        reasons.append("E is recomputed from D rather than stored")

    return codes, reasons


def _curl_conductivity_reasons(fields: Any, sub_step: Optional[str]) -> List[str]:
    """Conductivity refused on the NAMED sub-step's curl targets.

    A conductivity routes to the three-history conductive-PML recurrence, which is
    a DIFFERENT recurrence in the same sub-step, so it is a curl clause and not a
    grid one. This package ships no conductive product at all, so the refusal is
    unconditional rather than a hand-off.

    A missing or non-callable reader is refused OUTRIGHT on the magnetic side:
    inferring "no conductivity" from the absence of ``condfac_for`` is admission by
    attribute absence.
    """
    reasons: List[str] = []
    targets = CURL_SUB_STEPS[sub_step] if sub_step is not None else CURL_TARGETS
    reader = getattr(fields, "condfac_for", None)
    if callable(reader):
        for target in targets:
            try:
                conductive = reader(target) is not None
            except Exception as exc:  # noqa: BLE001 - unreadable means not covered
                reasons.append(f"condfac_for({target!r}) raised {exc!r}")
                continue
            if conductive:
                reasons.append(
                    f"a conductivity is installed on {target}; this curl "
                    f"transcribes the plain split-field recurrence only")
    if (sub_step in (None, "step_B")
            and getattr(fields, "has_magnetic_conductivity", False)
            and not callable(reader)):
        reasons.append(
            "a magnetic (B) conductivity is installed but condfac_for is unavailable")
    return reasons


def folded_pml_curl_coverage(fields: Any, pml: Any, sub_step: Optional[str] = None,
                             residency: Any = None) -> Coverage:
    """May the Metal folded curl step this (fields, pml, sub_step)? THE WIDE VERDICT.

    ZERO FOLDED AXES IS ADMITTED HERE, and that is not an oversight: with every
    axis resolving to PERIODIC/METALLIC, :func:`folded_curl_source` reduces to the
    certified kernel's ghost gather and cell-0 mask CHARACTER FOR CHARACTER and its
    top-plane block is empty, which is a property the gate MEASURES rather than a
    claim. THIS VERDICT MUST NOT BE REGISTERED ON A SLOT — it would make every
    unfolded row ambiguous. :func:`folded_composition_curl_coverage` is the routing
    verdict; this one is the gate's.

    WHAT IT ADMITS, in one sentence: a real-field Cartesian run under an active
    split-field PML on a host whose subnormal policy this executor can deliver, at
    k = 0 on every axis, with no conductivity, no nonlinearity, no BFAST, no
    special_kz, no complex storage, whose susceptibilities if any are electric
    lorentzian/drude poles, and with up to three axes folded by an even or odd
    mirror plane over a periodic or metallic outer declaration.
    """
    if sub_step is not None and sub_step not in CURL_SUB_STEPS:
        raise ValueError(
            f"sub_step must be one of {tuple(CURL_SUB_STEPS)}, got {sub_step!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    codes, reasons = _folded_curl_grid_reasons(fields, pml, grid)
    reasons.extend(_residency_declaration_reasons(residency))
    reasons.extend(_curl_conductivity_reasons(fields, sub_step))

    # 9a/9b. A registered susceptibility must be one this package understands.
    #        Dispersion is ADMITTED for the curl: it changes the VALUES update_E
    #        writes into the array the curl differences, not an operation the curl
    #        performs.
    reasons.extend(_susceptibility_reasons(fields))

    # 13. The six PML auxiliaries and the six curl sources.
    for name in (tuple("fu_" + target for target in CURL_TARGETS)
                 + tuple(SUB_STEPS["step_B"]["sources"])
                 + tuple(SUB_STEPS["step_D"]["sources"])):
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")

    # 14. Layout, at the STORED extent. This clause is where "the coefficient
    #     vectors are built at the folded stored extent" stops being an assumption:
    #     `_coefficient_reasons` compares each vector's length against
    #     `grid.shape[axis]`, which on a folded axis IS the stored count.
    shape = tuple(getattr(grid, "shape", ()))
    volumes = (tuple(CURL_TARGETS) + tuple("fu_" + t for t in CURL_TARGETS)
               + tuple(SUB_STEPS["step_B"]["sources"])
               + tuple(SUB_STEPS["step_D"]["sources"]))
    reasons.extend(_layout_reasons(fields, shape, volumes))
    if pml is not None and getattr(pml, "is_active", False) and len(shape) == 3:
        reasons.extend(_coefficient_reasons(pml, shape, ("kms", "sinv"), ("", "_h")))

    # A folded axis must be able to hold its own two ghost planes, and the near
    # ghost images stored cell 2. `folded_axis_kinds` already refuses fewer than
    # three stored cells; this is the shape-side companion for the top-plane mask,
    # which needs `last` to be distinct from cell 0.
    if codes is not None and len(shape) == 3:
        for axis, code in enumerate(codes):
            if int(code) in MIRROR_CODES and int(shape[axis]) <= MIRROR_SOURCE_INDEX:
                reasons.append(
                    f"axis {axis} is folded with {int(shape[axis])} stored cells; "
                    f"the near ghost images stored cell {MIRROR_SOURCE_INDEX}")

    if codes is None and not reasons:  # pragma: no cover - belt and braces
        reasons.append("the per-axis boundary codes could not be resolved")
    return Coverage(not reasons, tuple(reasons))


def folded_composition_curl_coverage(fields: Any, pml: Any, sub_step: str,
                                     residency: Any = None) -> Coverage:
    """THE ROUTING VERDICT — :func:`folded_pml_curl_coverage` with a fold MANDATORY.

    The one clause that separates this family from ``pml_curl``, ``bfast_curl``,
    ``complex_fields`` and ``special_kz`` on the two curl slots, spelled as its own
    function so a reader can see the inversion in one place. Registering the wide
    verdict instead would make every unfolded row ambiguous and would leave the
    numerical method depending on composer order.
    """
    if sub_step not in CURL_SUB_STEPS:
        raise ValueError(
            f"sub_step must be one of {tuple(CURL_SUB_STEPS)}, got {sub_step!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))
    base = folded_pml_curl_coverage(fields, pml, sub_step, residency)
    codes, _ = _folded_axis_reasons(grid, pml)
    reasons = list(base.reasons)
    reasons.extend(_requires_a_fold(
        grid, codes,
        "the folded curl is an equivalence product here, not a composition "
        "candidate; an unfolded grid belongs to the certified plain kernel"))
    return Coverage(not reasons, tuple(reasons))


def _folded_constitutive_grid_reasons(fields: Any, pml: Any, grid: Any) -> List[str]:
    """Grid clauses for the ELEMENT-WISE constitutive product on a real fold.

    DELIBERATELY NOT :func:`_folded_curl_grid_reasons`. That helper is the folded
    CURL's contract and refuses conductivity because conductivity changes the curl
    recurrence; conductivity does not change ``update_H``/``update_E``, which read
    no neighbour and consult no ``condfac_for``. Re-stating the element-wise
    contract prevents a curl-only refusal from silently narrowing an independent
    sub-step — the same split the Triton twin makes, for the same reason.

    THE FOLD REACHES THIS SUB-STEP THROUGH EXACTLY ONE THING: THE STORED EXTENT.
    ``update_H`` (stepping.py:907-924) and the diagonal ``update_E``
    (stepping.py:1012-1020) are three ``_apply_constitutive_pml`` calls over
    integer-position coefficients, with no shift helper, no ghost and no ownership
    mask. The PML coefficient vectors are already built at the folded stored extent,
    so this family adds NO DEVICE CODE on these two slots — it re-admits the
    certified body, and the layout clauses below are where "already built at the
    stored extent" stops being an assumption.
    """
    reasons: List[str] = []
    reasons.extend(_metal_backend_reasons(grid))
    if getattr(fields, "force_complex_fields", False):
        reasons.append("force_complex_fields=True (complex64 storage is not carried)")
    if pml is None or not getattr(pml, "is_active", False):
        reasons.append("no active PML layer (this constitutive product implements "
                       "the dsigw path only)")

    codes, fold_reasons = _folded_axis_reasons(grid, pml)
    reasons.extend(fold_reasons)
    reasons.extend(_requires_a_fold(
        grid, codes,
        "the folded constitutive product has no array-path work to specialize"))

    if getattr(grid, "cylindrical", False):
        reasons.append("cylindrical (Dcyl) coordinates are not carried")
    for axis in range(3):
        if _call(grid, "is_axis", axis, default=False):
            reasons.append(f"axis {axis} is the cylindrical r = 0 axis")

    if getattr(grid, "has_bloch", False):
        reasons.append(f"nonzero k_point {getattr(grid, 'k_point', None)!r}")
    k_point = getattr(grid, "k_point", (0.0, 0.0, 0.0))
    if any(float(component) != 0.0 for component in k_point):
        reasons.append(f"k_point {tuple(k_point)!r} is not exactly zero")

    if getattr(fields, "has_nonlinearity", False):
        reasons.append("chi2/chi3 is installed (the Pade constitutive factor is "
                       "not carried, and a nonlinear fold has no single parity)")
    if getattr(grid, "bfast_active", False):
        reasons.append("BFAST is active (the second additive curl term is not carried)")
    if float(getattr(grid, "beta", 0.0)) != 0.0:
        reasons.append(f"special_kz beta={getattr(grid, 'beta', 0.0)!r} is nonzero "
                       f"(folded_beta's real arm re-admits this same certified "
                       f"constitutive body on a folded beta run)")
    return reasons


def folded_constitutive_coverage(fields: Any, pml: Any, side: str,
                                 residency: Any = None) -> Coverage:
    """May the CERTIFIED Metal constitutive kernel step a FOLDED run?

    No new kernel and no new arithmetic: only the ADMISSION is this family's. The
    shipped ``constitutive_coverage`` correctly refuses a folded grid, because the
    ordinary curl/constitutive product family cannot step one; this verdict proves
    the folded extent and the coefficient lengths explicitly instead of weakening
    that family globally.

    THE E SIDE REFUSES EVERYTHING THAT CHANGES WHAT ``source`` IS, and each refusal
    names the family that owns it: a registered susceptibility (the source is
    ``D - sum P``, not ``D`` — the folded dispersive family, which likewise needs no
    kernel), an off-diagonal chi1inv row (the row product READS NEIGHBOURS through
    ``_shift_down``/``_shift_up``, and on a fold the partner-axis down shift is a
    LIVE interior plane — the folded off-diagonal family), and ``stores_E`` false.
    """
    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, "
                         f"got {side!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    spec = CONSTITUTIVE_SIDES[side]
    reasons = _folded_constitutive_grid_reasons(fields, pml, grid)
    reasons.extend(_residency_declaration_reasons(residency))
    reasons.extend(_susceptibility_reasons(fields))

    if side == "E":
        if getattr(fields, "has_polarizations", False) or (
                getattr(fields, "polarizations", ()) or ()):
            reasons.append(
                "a susceptibility is registered: folded dispersive update_E is not "
                "composed here; its source is (D - sum P), not D")
        if getattr(fields, "has_offdiagonal_epsilon", False):
            reasons.append(
                "an off-diagonal chi1inv row is installed (the row product reads "
                "neighbours; this sub-step is element-wise)")
        if not getattr(fields, "stores_E", False):
            reasons.append("E is recomputed from D rather than stored")

    names = tuple(spec["targets"]) + tuple(spec["aux"]) + tuple(spec["sources"])
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")

    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(_layout_reasons(fields, shape, names))
    if side == "E" and len(shape) == 3:
        reasons.extend(_inverse_epsilon_reasons(fields, shape))
    if pml is not None and getattr(pml, "is_active", False) and len(shape) == 3:
        suffix = ("_h",) if spec["half_integer"] else ("",)
        reasons.extend(_coefficient_reasons(pml, shape, ("kps", "kms"), suffix))

    return Coverage(not reasons, tuple(reasons))


def _wall_seam_reasons(grid: Any, codes: Optional[Sequence[int]]) -> List[str]:
    """THE ONE CONFIGURATION THIS FAMILY REFUSES RATHER THAN FUSES.

    ``zero_metal_B`` / ``zero_metal_D`` run BETWEEN the two fill passes
    (driver.py:3285-3287 and :3300-3302). The far pass READS a whole plane at
    ``reflect_row``, which on a run with a live wall includes the cells
    ``_zero_metal`` has just cleared, so a plan that ran the far pass before the
    wall clear would read pre-clear values — a plane of wrong numbers, not a crash.

    :class:`MirrorGhostFillPlan` exposes the two passes separately so a driver
    adapter CAN place them correctly, but nothing in this package is a driver
    adapter yet and ``run()`` walks them back to back. So the pairing is refused BY
    NAME rather than fused, and the refusal is free on the measured corpus: across
    the 86 folded rows, no boundary triple carrying a MIRROR_PERIODIC axis carries
    a METALLIC one — 10 rows at (PERIODIC, MIRROR_PERIODIC, PERIODIC) and 2 at
    (MIRROR_PERIODIC, MIRROR_PERIODIC, PERIODIC), and nothing else.

    Note ``zero_metal_axes`` asks the grid's own DECLARATION rather than
    ``_boundary_kinds``, which is the question ``_zero_metal`` asks
    (stepping.py:2282-2284) — and it already excludes a folded metallic axis, whose
    cell 0 holds the parity ghost and must not be cleared (measured on the array
    path: 1.28e+00 complex relative L2 against CPU MEEP with the ghost zeroed,
    against 2.0e-07 with the skip).
    """
    if codes is None or not any(int(code) == CODE_MIRROR_PERIODIC for code in codes):
        return []
    walled = zero_metal_axes(grid)
    if not any(walled):
        return []
    return [f"a folded PERIODIC axis needs the far ghost pass, and axes "
            f"{tuple(i for i, w in enumerate(walled) if w)} are live zero_metal "
            f"walls: the driver clears stored cell 0 BETWEEN the two fill passes "
            f"(driver.py:3286 / :3301) and the far pass reads a whole plane that "
            f"includes those cells, so running the two back to back is a different "
            f"answer. The passes are separately launchable (run_near / run_far); "
            f"until a driver adapter places them, the pairing is refused"]


def mirror_ghost_fill_coverage(fields: Any, family: str,
                               residency: Any = None) -> Coverage:
    """May the Metal mirror fill write this family's ghost planes?

    NARROWER THAN THE CURL'S PREDICATE AND INDEPENDENT OF IT: this pass reads no
    coefficient, no source and no PML, so it has no Courant number, no sub-lattice
    and no absorber clause — ``fill_symmetry_bc_*`` runs whether or not an absorber
    is installed. A PARTIAL FOLD COVERAGE IS THEREFORE LEGAL AND HAPPENS: on a
    folded no-PML run these two slots are covered while every curl and constitutive
    slot is on the array path. That is a substitution with no throughput benefit and
    a driver-adapter seam, and it is named here rather than discovered later.

    What it DOES need: a real fold, a readable phase, a stored extent that can hold
    both planes, and — on a folded PERIODIC axis — a reflect row from
    ``stepping._far_reflect_rows``, checked to be distinct from BOTH planes this
    kernel touches.
    """
    if family not in GHOST_FILL_FAMILIES:
        raise ValueError(f"family must be one of {tuple(GHOST_FILL_FAMILIES)}, "
                         f"got {family!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    reasons: List[str] = []
    reasons.extend(_metal_backend_reasons(grid))
    reasons.extend(_residency_declaration_reasons(residency))
    if getattr(fields, "force_complex_fields", False):
        reasons.append("force_complex_fields=True: the real fill's parity is an "
                       "exact sign flip; under complex64 it is a full complex "
                       "multiply by (+/-1, +0) and is NOT replaced here")
    if getattr(grid, "cylindrical", False):
        reasons.append("cylindrical (Dcyl) coordinates are not carried")
    for axis in range(3):
        if _call(grid, "is_axis", axis, default=False):
            reasons.append(f"axis {axis} is the cylindrical r = 0 axis")

    codes, fold_reasons = _folded_axis_reasons(grid, None)
    reasons.extend(fold_reasons)
    reasons.extend(_requires_a_fold(
        grid, codes,
        "the array path's fill passes return immediately (stepping.py:1482-1483) "
        "and there is nothing to replace"))
    reasons.extend(_wall_seam_reasons(grid, codes))

    rows = _far_reflect_rows(grid)
    shape = tuple(getattr(grid, "shape", ()))
    if codes is not None and len(shape) == 3:
        for axis, code in enumerate(codes):
            extent = int(shape[axis])
            if int(code) in MIRROR_CODES and extent <= MIRROR_SOURCE_INDEX:
                reasons.append(
                    f"axis {axis} is folded with {extent} stored cells; the near "
                    f"ghost images stored cell {MIRROR_SOURCE_INDEX}")
            if int(code) != CODE_MIRROR_PERIODIC:
                continue
            row = rows[axis] if rows is not None else None
            if row is None:
                reasons.append(f"axis {axis} is a folded PERIODIC axis with no "
                               f"reflect row from stepping._far_reflect_rows")
                continue
            row = int(row)
            # THREE CHECKS, NOT ONE, AND THE SECOND TWO ARE THIS PORT'S.
            #
            # 1. The Triton predicate's: the row must be inside the allocation and
            #    below the plane the far pass writes, or the far ghost images the
            #    plane it writes, or reads past the end.
            # 2. The row must not be stored cell 0, which the NEAR launch has
            #    already written by the time the far launch runs. Imaging a ghost
            #    instead of an owned cell is a plane of wrong values.
            # 3. An extent this small makes the far pass's WRITE plane the same
            #    plane the near pass READS. That is actually SAFE as the passes are
            #    launched — near reads it before far overwrites it, which is the
            #    array path's own order (stepping.py:1431 then :1521) — but it is
            #    NOT safe for the fused :meth:`MirrorGhostFillPlan.run`, which
            #    exists. Refused, because a coincidence that holds only under one
            #    of two launch orders this plan offers is the fail-closed case.
            if not (0 <= row < extent - 1):
                reasons.append(
                    f"axis {axis} reflect row {row} is outside [0, {extent - 1}); "
                    f"the far ghost would image the plane it writes, or read "
                    f"outside the allocation")
            if row == 0:
                reasons.append(
                    f"axis {axis} reflect row is 0, the plane the near pass writes: "
                    f"the far ghost would image a ghost rather than an owned cell")
            if extent - 1 == MIRROR_SOURCE_INDEX:
                reasons.append(
                    f"axis {axis} stores {extent} cells, so the far pass's write "
                    f"plane IS the near pass's read plane (cell "
                    f"{MIRROR_SOURCE_INDEX}); that is safe only in the near-then-far "
                    f"order and this plan also offers a fused run(), so it is "
                    f"refused rather than made order-dependent")

    names = tuple(GHOST_FILL_FAMILIES[family]["targets"])
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")
    reasons.extend(_layout_reasons(fields, shape, names))
    return Coverage(not reasons, tuple(reasons))


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------

class FoldedPmlCurlPlan(KernelPlan):
    """A launchable, allocation-free folded curl sub-step.

    ``launch.PmlCurlPlan``'s bindings EXACTLY — the fold adds no argument, because
    the stored extent is what carries it — over a different kernel. It subclasses
    :class:`.plans.KernelPlan` rather than duplicating the launch path, so the
    contraction-variant contract has one implementation for the whole package.

    Built two ways and launched ONE way: :func:`plan_folded_pml_curl` from the
    engine's own objects, :func:`plan_folded_pml_curl_from_arrays` from bare host
    arrays for the gate and the benchmark. The bytes the gate certifies are the
    bytes the engine would launch.
    """

    __slots__ = ("sub_step", "shape", "n_elem", "dtdx", "backward", "bc",
                 "residency", "volumes")

    family = "real-field folded PML curl"

    REPR_FIELDS = ("sub_step", "shape", "bc")

    def __init__(self, sub_step: str, shape, dtdx: float, bc, residency: Residency,
                 targets, auxiliaries, sources, coefficients,
                 functions: Dict[str, Any], volumes: Sequence[str]) -> None:
        if sub_step not in SUB_STEPS:
            raise ValueError(
                f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
        self.sub_step = sub_step
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.dtdx = float(dtdx)
        self.backward = bool(SUB_STEPS[sub_step]["backward"])
        self.bc = tuple(int(code) for code in bc)
        self.residency = residency
        self.volumes = tuple(volumes)
        super().__init__(
            functions,
            tuple(targets) + tuple(auxiliaries) + tuple(sources)
            + tuple(coefficients)
            + (self.shape[0], self.shape[1], self.shape[2],
               self.n_elem, self.dtdx))


def _folded_curl_functions(codes, backward: bool,
                           contract_variants: Sequence[str]) -> Dict[str, Any]:
    return {mode: compile_folded_curl(codes, backward, mode)
            for mode in contract_variants}


def plan_folded_pml_curl(fields: Any, pml: Any, sub_step: str,
                         residency: Optional[Residency] = None,
                         contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
                         ) -> Optional[FoldedPmlCurlPlan]:
    """Build a folded curl plan from the engine's own objects, or None when refused.

    None is the only refusal: a configuration this kernel does not carry must fall
    back to the array path, never raise into a caller that would otherwise have
    stepped correctly.

    The codes come from :func:`folded_axis_kinds` and from nowhere else — building
    them from ``_boundary_kinds`` alone would lose the MIRROR_METALLIC /
    MIRROR_PERIODIC split, which decides the top-plane mask.
    """
    if sub_step not in SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
    if not folded_composition_curl_coverage(fields, pml, sub_step, residency).covered:
        return None
    grid = fields.grid
    codes, _ = folded_axis_kinds(grid, pml)
    if codes is None:  # pragma: no cover - the predicate already refused
        return None

    spec = SUB_STEPS[sub_step]
    targets = [residency.mirror(n, getattr(fields, n)) for n in spec["targets"]]
    auxiliaries = [residency.mirror("fu_" + n, getattr(fields, "fu_" + n))
                   for n in spec["targets"]]
    sources = [residency.mirror(n, getattr(fields, n)) for n in spec["sources"]]
    coefficients = [
        residency.mirror(f"pml:{stem}_{axis}{spec['suffix']}",
                         getattr(pml, f"{stem}_{axis}{spec['suffix']}"),
                         constant=True)
        for axis in "xyz" for stem in ("kms", "sinv")]
    volumes = (tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
               + tuple(spec["sources"]))
    return FoldedPmlCurlPlan(
        sub_step, grid.shape, grid.dt / grid.dx, codes, residency,
        targets, auxiliaries, sources, coefficients,
        _folded_curl_functions(codes, spec["backward"], contract_variants), volumes)


def plan_folded_pml_curl_from_arrays(sub_step: str, arrays: Dict[str, Any],
                                     flat: Dict[str, Any], codes, dtdx: float,
                                     residency: Residency,
                                     functions: Optional[Dict[str, Any]] = None,
                                     contract_variants: Sequence[str] = (
                                         shaders.CONTRACT_OFF,),
                                     ) -> FoldedPmlCurlPlan:
    """Build a folded curl plan from bare host arrays — the gate's route.

    No predicate runs: the caller is a harness that constructed the configuration
    deliberately, including the deliberately wrong ones. ``functions`` is the
    mutation seam — dropping it is not a silent slowdown but a silent DISARMING,
    since every mutation leg would then launch the shipped kernel and report the
    defect as uncaught.
    """
    spec = SUB_STEPS[sub_step]
    shape = tuple(int(n) for n in arrays[spec["targets"][0]].shape)
    targets = [residency.mirror(n, arrays[n]) for n in spec["targets"]]
    auxiliaries = [residency.mirror("fu_" + n, arrays["fu_" + n])
                   for n in spec["targets"]]
    sources = [residency.mirror(n, arrays[n]) for n in spec["sources"]]
    coefficients = [residency.mirror(f"pml:{stem}_{axis}:{sub_step}",
                                     flat[f"{stem}_{axis}"], constant=True)
                    for axis in "xyz" for stem in ("kms", "sinv")]
    volumes = (tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
               + tuple(spec["sources"]))
    return FoldedPmlCurlPlan(
        sub_step, shape, dtdx, codes, residency, targets, auxiliaries, sources,
        coefficients,
        functions if functions is not None
        else _folded_curl_functions(codes, spec["backward"], contract_variants),
        volumes)


class MirrorGhostFillPlan:
    """One family's mirror ghost planes, as one launch per (pass, folded axis).

    DELIBERATELY A SIBLING OF :class:`.plans.KernelPlan` RATHER THAN A SUBCLASS.
    That base's whole contract is that ``run`` unpacks ONE argument tuple and calls
    ONE function, and one implementation of that is what makes the launch path
    uniform across the family tree. This plan cannot honour it: the array path
    applies the axes in X, Y, Z ORDER — so a corner unowned on two planes carries
    the PRODUCT of both parities, and a corner unowned on all three the product of
    three (stepping.py:1476-1480) — and ordering inside one dispatch is not
    something a grid of threads can promise. So the plan holds a LIST of launches
    and walks it in that order, and the divergence from the base is stated here
    rather than hidden behind an overridden ``run``.

    THE TWO PASSES ARE SEPARATE ATTRIBUTES, NOT ONE LIST, because ``zero_metal_*``
    runs between them in the driver (driver.py:3286 / :3301) and the far pass reads
    a plane the wall clear can touch. :meth:`run` walks near-then-far for a caller
    that has established the wall is not live — which
    :func:`mirror_ghost_fill_coverage` refuses to admit otherwise — and
    :meth:`run_near` / :meth:`run_far` are what a driver adapter places.

    ``replaces_sub_steps`` IS DECLARED, NOT INFERRED. The composer reads it to
    learn that a plan filling ``fill_B`` also performs
    ``fill_folded_far_ghosts_B`` on the device, which is a residency fact no slot
    name carries. A plan that does not declare it replaces only its own slot, which
    is the answer that DEMANDS a mirror rather than waiving one.
    """

    __slots__ = ("family", "slot", "shape", "residency", "volumes", "near", "far",
                 "replaces_sub_steps", "launches", "_functions")

    #: This plan launches kernels; the composer's planned/null split reads it.
    performs_device_work = True

    def __init__(self, family: str, slot: str, shape, residency: Residency,
                 near: Sequence[Dict[str, Any]], far: Sequence[Dict[str, Any]],
                 functions: Dict[str, Dict[str, Any]],
                 volumes: Sequence[str]) -> None:
        if family not in GHOST_FILL_FAMILIES:
            raise ValueError(f"family must be one of {tuple(GHOST_FILL_FAMILIES)}, "
                             f"got {family!r}")
        self.family = family
        self.slot = slot
        self.shape = tuple(int(n) for n in shape)
        self.residency = residency
        self.volumes = tuple(volumes)
        self.near = tuple(dict(entry) for entry in near)
        self.far = tuple(dict(entry) for entry in far)
        self._functions = {mode: dict(table) for mode, table in functions.items()}
        self.launches = 0
        far_slot = ("fill_folded_far_ghosts_B" if family == "B"
                    else "fill_folded_far_ghosts_D")
        self.replaces_sub_steps = ((slot, far_slot) if self.far else (slot,))

    @property
    def variants(self) -> Tuple[str, ...]:
        return tuple(sorted(self._functions))

    def _walk(self, entries: Sequence[Dict[str, Any]], mode: str) -> None:
        table = self._functions.get(mode)
        if table is None:
            raise KeyError(
                f"this plan holds no {mode!r} variant (it was built with "
                f"{self.variants}); build it with contract_variants={(mode,)} "
                f"rather than launching the pinned one")
        nx, ny, nz = self.shape
        for entry in entries:
            axis = int(entry["axis"])
            n_plane = (ny * nz, nx * nz, nx * ny)[axis]
            self.launches += 1
            table[entry["key"]](*entry["targets"], nx, ny, nz, n_plane,
                                int(entry["reflect_row"]))

    def run_near(self, contract: Optional[str] = None) -> None:
        """``fill_symmetry_bc_*`` for every folded axis, in X, Y, Z order. In place."""
        self._walk(self.near, shaders.CONTRACT_OFF if contract is None else contract)

    def run_far(self, contract: Optional[str] = None) -> None:
        """``fill_folded_far_ghosts_*`` for every folded PERIODIC axis. In place."""
        self._walk(self.far, shaders.CONTRACT_OFF if contract is None else contract)

    def run(self, contract: Optional[str] = None) -> None:
        """Both passes, near then far — the driver's order with the wall clear out.

        Only correct where ``_zero_metal`` performs no work, which is what
        :func:`mirror_ghost_fill_coverage` refuses to admit otherwise.
        """
        self.run_near(contract)
        self.run_far(contract)

    def describe(self) -> str:
        return (f"MirrorGhostFillPlan({self.family}, shape={self.shape}, "
                f"near={[e['axis'] for e in self.near]}, "
                f"far={[e['axis'] for e in self.far]}, variants={self.variants})")

    def __repr__(self) -> str:
        return self.describe()


def ghost_fill_axis_entries(grid: Any, family: str,
                            pass_name: str) -> Tuple[Dict[str, Any], ...]:
    """One entry per folded axis for one pass, in X, Y, Z order.

    PUBLIC AND SEPARATE FROM THE PLAN BUILDER so it can be checked on a host with
    no GPU, which is where the two things that silently go wrong here live: the
    X, Y, Z ORDER (a doubly-unowned corner must end up carrying the product of both
    parities) and the REFLECT ROW (``stored - 2`` at an even full count,
    ``stored - 3`` at an odd one — baking ``n - 2`` is a whole cell wrong on every
    odd-count run).
    """
    if pass_name not in FILL_PASSES:
        raise ValueError(f"pass_name must be one of {FILL_PASSES}, got {pass_name!r}")
    codes, reasons = folded_axis_kinds(grid, None)
    if codes is None:
        raise ValueError("this grid's folded axes cannot be classified: "
                         + "; ".join(reasons))
    rows = _far_reflect_rows(grid) or (None, None, None)
    targets = tuple(GHOST_FILL_FAMILIES[family]["targets"])
    entries: List[Dict[str, Any]] = []
    for axis, code in enumerate(codes):  # X, Y, Z — the order the fills apply in.
        code = int(code)
        if code not in MIRROR_CODES:
            continue
        far = code == CODE_MIRROR_PERIODIC
        if pass_name == "far" and not far:
            continue
        entries.append({
            "axis": axis,
            "phase": int(grid.mirror_phase(axis)),
            "pass": pass_name,
            # -1 is never read by a near launch; it is passed rather than left
            # unset so the kernel signature is ONE shape.
            "reflect_row": int(rows[axis]) if far and rows[axis] is not None else -1,
            "shifts": tuple(TARGET_IYEE[name][axis] for name in targets),
        })
    return tuple(entries)


def _fill_functions(entries: Sequence[Dict[str, Any]],
                    contract_variants: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Compile one entry point per distinct (axis, phase, pass, shifts) key."""
    table: Dict[str, Dict[str, Any]] = {mode: {} for mode in contract_variants}
    for entry in entries:
        key = entry["key"]
        for mode in contract_variants:
            if key not in table[mode]:
                table[mode][key] = compile_mirror_ghost_fill(
                    entry["axis"], entry["phase"], entry["pass"], entry["shifts"],
                    mode)
    return table


def _keyed(entries: Sequence[Dict[str, Any]], targets: Sequence[Any],
           ) -> Tuple[Dict[str, Any], ...]:
    """Attach the compile key and the bound target tensors to each entry."""
    out: List[Dict[str, Any]] = []
    for entry in entries:
        item = dict(entry)
        item["key"] = (f"axis{entry['axis']}/phase{int(entry['phase']):+d}/"
                       f"{entry['pass']}/{''.join(str(s) for s in entry['shifts'])}")
        item["targets"] = tuple(targets)
        out.append(item)
    return tuple(out)


def plan_mirror_ghost_fill(fields: Any, family: str, slot: str,
                           residency: Optional[Residency] = None,
                           contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
                           ) -> Optional[MirrorGhostFillPlan]:
    """Build one family's ghost-fill plan from the engine's own objects, or None.

    The per-axis entries carry the phase, the reflect row and the three targets'
    Yee shifts on that axis — everything the kernel specialises on — read from the
    grid and from ``stepping._far_reflect_rows``, never derived here.
    """
    if not mirror_ghost_fill_coverage(fields, family, residency).covered:
        return None
    grid = fields.grid
    names = tuple(GHOST_FILL_FAMILIES[family]["targets"])
    targets = [residency.mirror(n, getattr(fields, n)) for n in names]
    near = _keyed(ghost_fill_axis_entries(grid, family, "near"), targets)
    far = _keyed(ghost_fill_axis_entries(grid, family, "far"), targets)
    if not near:  # pragma: no cover - the predicate already refused
        return None
    return MirrorGhostFillPlan(
        family, slot, grid.shape, residency, near, far,
        _fill_functions(tuple(near) + tuple(far), contract_variants), names)


def plan_mirror_ghost_fill_from_arrays(family: str, slot: str,
                                       arrays: Dict[str, Any],
                                       near: Sequence[Dict[str, Any]],
                                       far: Sequence[Dict[str, Any]],
                                       residency: Residency,
                                       functions: Optional[
                                           Dict[str, Dict[str, Any]]] = None,
                                       contract_variants: Sequence[str] = (
                                           shaders.CONTRACT_OFF,),
                                       ) -> MirrorGhostFillPlan:
    """Build a ghost-fill plan from bare host arrays — the gate's route.

    ``near`` and ``far`` are the entry lists :func:`ghost_fill_axis_entries` builds;
    the gate supplies them directly so it can hand over a deliberately wrong reflect
    row, a deliberately wrong phase or a reversed axis order and watch each be
    caught. ``functions`` is the mutation seam.
    """
    names = tuple(GHOST_FILL_FAMILIES[family]["targets"])
    shape = tuple(int(n) for n in arrays[names[0]].shape)
    targets = [residency.mirror(n, arrays[n]) for n in names]
    near = _keyed(near, targets)
    far = _keyed(far, targets)
    return MirrorGhostFillPlan(
        family, slot, shape, residency, near, far,
        functions if functions is not None
        else _fill_functions(tuple(near) + tuple(far), contract_variants), names)


def plan_folded_constitutive(fields: Any, pml: Any, side: str,
                             residency: Any = None,
                             contract_variants: Sequence[str] = (
                                 shaders.CONTRACT_OFF,)) -> Any:
    """A CERTIFIED constitutive plan for a folded run, or None.

    The arithmetic, the source and the plan class are the certified ones; only the
    ADMISSION is this family's. Building the plan through the certified builder —
    rather than re-deriving one here — is what makes "same kernel" a fact instead of
    a claim: a divergence would have to come from the predicate, which is the only
    thing this family contributes on these two slots. The same move
    ``bfast_curl.plan_bfast_run_constitutive`` and
    ``special_kz.beta_run_constitutive`` make, for the same reason.
    """
    from .launch import (  # noqa: PLC0415 - avoids a circular import at module load
        ConstitutivePlan, _constitutive_functions,
    )

    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, "
                         f"got {side!r}")
    if not folded_constitutive_coverage(fields, pml, side, residency).covered:
        return None
    spec = CONSTITUTIVE_SIDES[side]
    suffix = "_h" if spec["half_integer"] else ""
    targets = [residency.mirror(n, getattr(fields, n)) for n in spec["targets"]]
    auxiliaries = [residency.mirror(n, getattr(fields, n)) for n in spec["aux"]]
    sources = [residency.mirror(n, getattr(fields, n)) for n in spec["sources"]]
    inverse_epsilon = (
        [residency.mirror("inv_eps_" + n, fields.inverse_epsilon_for(n),
                          constant=True) for n in spec["targets"]]
        if side == "E" else None)
    coefficients = [
        residency.mirror(f"pml:{stem}_{axis}{suffix}",
                         getattr(pml, f"{stem}_{axis}{suffix}"), constant=True)
        for axis in "xyz" for stem in ("kps", "kms")]
    volumes = tuple(spec["targets"]) + tuple(spec["aux"]) + tuple(spec["sources"])
    return ConstitutivePlan(
        side, fields.grid.shape, residency, targets, auxiliaries, sources,
        inverse_epsilon, coefficients,
        _constitutive_functions(side, contract_variants), volumes)


# ---------------------------------------------------------------------------
# WIRING — six arms over six slots
# ---------------------------------------------------------------------------
#
# WHAT SEPARATES THEM, per slot, and every one of these is a clause that names the
# fold rather than an omission:
#
#   step_B / step_D    `coverage._grid_reasons` clause 5 refuses a fold BY NAME
#                      (two strings: the grid-level one and one per folded axis);
#                      `bfast_curl` clause 5 refuses it by name; `complex_fields`
#                      and `special_kz` refuse it by name. Clause 5 HERE is
#                      inverted through `_requires_a_fold`.
#   update_H/update_E  the same clause 5 covers `constitutive`; `no_pml_constitutive`
#                      inverts on the ABSORBER instead (it requires an INACTIVE one
#                      and this family requires an active one), which is a second,
#                      independent separation on those two slots; `offdiag_update_e`
#                      requires a LIVE off-diagonal row and this family refuses one.
#   fill_B / fill_D    no other Metal family registers here at all. Those two slots
#                      previously carried a `NO_ARM_REASONS` entry saying no product
#                      carried the source seam; they now carry this family's arm,
#                      and the source-injection question is unchanged — the driver
#                      injects BEFORE the fill (stepping.py:396-400, :1452-1465),
#                      which is why the fill is placed after it and not fused over
#                      it.
#
# REGISTERED-BUT-CANNOT-WIN IS NOT USED HERE, and the absence is worth stating.
# `registry.py` keeps `registered` and `wins` separate so a missing capability gets
# NAMED at composition time instead of falling silently to the array path, and this
# family would have used it if either constitutive half were unported. Both halves
# ARE ported — they are the certified body over a folded extent — so all six arms
# can win, and the gap this family DOES have is named in a predicate instead: the
# folded far pass on a walled grid is a refusal by name (`_wall_seam_reasons`),
# not a silent fall-through.


def _curl_arm_coverage(context: Any, slot: str) -> Coverage:
    return folded_composition_curl_coverage(context.fields, context.pml, slot,
                                            context.residency)


def _curl_arm_plan(context: Any, slot: str) -> Optional[FoldedPmlCurlPlan]:
    return plan_folded_pml_curl(context.fields, context.pml, slot,
                                context.residency, context.contract_variants)


#: Which constitutive side each slot names, so one pair of callables serves both.
_CONSTITUTIVE_SLOT_SIDES: Dict[str, str] = {"update_H": "H", "update_E": "E"}

#: Which fill family each seam slot names, likewise.
_FILL_SLOT_FAMILIES: Dict[str, str] = {"fill_B": "B", "fill_D": "D"}


def _constitutive_arm_coverage(context: Any, slot: str) -> Coverage:
    return folded_constitutive_coverage(
        context.fields, context.pml, _CONSTITUTIVE_SLOT_SIDES[slot],
        context.residency)


def _constitutive_arm_plan(context: Any, slot: str) -> Any:
    return plan_folded_constitutive(
        context.fields, context.pml, _CONSTITUTIVE_SLOT_SIDES[slot],
        context.residency, context.contract_variants)


def _fill_arm_coverage(context: Any, slot: str) -> Coverage:
    return mirror_ghost_fill_coverage(context.fields, _FILL_SLOT_FAMILIES[slot],
                                      context.residency)


def _fill_arm_plan(context: Any, slot: str) -> Optional[MirrorGhostFillPlan]:
    return plan_mirror_ghost_fill(context.fields, _FILL_SLOT_FAMILIES[slot], slot,
                                  context.residency, context.contract_variants)


def _has_fold(context: Any) -> bool:
    """The cheap gate that decides whether this family is CONSULTED at all.

    A gated-out arm contributes NO reason; a consulted-and-refusing arm contributes
    all of its reasons by name. So the gate is deliberately the cheapest possible
    read of the grid and never the predicate itself — on an unfolded run this
    family should be silent on the four slots where OTHER arms will speak.
    """
    grid = getattr(getattr(context, "fields", None), "grid", None)
    return grid is not None and _has_real_fold(grid)


def register_arms() -> Tuple[Any, ...]:
    """This family's six arms: two curls, the certified constitutive pair, two fills.

    THE FILL ARMS ARE DELIBERATELY UNGATED and the other four are not. The gate
    exists so a gated-out arm contributes NO reason, which is right where other
    arms will speak — ``step_B``/``step_D`` carry four other families and
    ``update_H``/``update_E`` carry four more, so a silent folded arm on an
    unfolded run costs a reader nothing. On ``fill_B``/``fill_D`` this family is
    the ONLY arm, and a gated-out sole arm leaves the composer reporting "no
    consulted product admitted fill_B, and none gave a reason" — which is exactly
    the "nobody asked" reading ``NO_ARM_REASONS`` exists to prevent. Ungated, an
    unfolded run gets the fill's own named refusal instead.
    """
    from . import arms  # noqa: PLC0415 - deferred: `arms` imports nothing of ours

    registered = [
        arms.register(family=FAMILY, slot=slot, label="folded",
                      coverage=_curl_arm_coverage, plan=_curl_arm_plan,
                      prefix="folded: ", noun="folded PML curl",
                      gate=_has_fold, wired=True)
        for slot in ("step_B", "step_D")]
    registered.extend(
        arms.register(family=FAMILY, slot=slot, label="folded",
                      coverage=_constitutive_arm_coverage,
                      plan=_constitutive_arm_plan,
                      prefix="folded: ", noun="folded constitutive",
                      gate=_has_fold, wired=True)
        for slot in _CONSTITUTIVE_SLOT_SIDES)
    registered.extend(
        arms.register(family=FAMILY, slot=slot, label="mirror fill",
                      coverage=_fill_arm_coverage, plan=_fill_arm_plan,
                      prefix="mirror fill: ", noun="mirror ghost fill",
                      gate=None, wired=True)
        for slot in _FILL_SLOT_FAMILIES)
    return tuple(registered)


#: Registered ON IMPORT, once — the registry refuses a duplicate by design.
ARMS: Tuple[Any, ...] = register_arms()
