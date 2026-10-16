"""The mirror FOLD under COMPLEX64 storage — a composition of two certified families.

Both halves are already byte-certified on this backend alone: :mod:`.complex_fields`
(the complex/Bloch curl and constitutive) and :mod:`.symmetry` (the real fold's curl,
fill and constitutive admission). This module is what happens when they meet, and the
whole point of writing it as a family rather than as a flag is that MEETING IS NOT
FREE: one of the two halves stops being a transcription and becomes new arithmetic.

WHAT COMPOSES, AND IT IS MOST OF THE FAMILY:

* the CURL. ``symmetry.py``'s argument for why a fold changes nothing arithmetic in
  the curl is an OWNERSHIP argument, not a storage argument, and it transfers
  unchanged: the near ghost ``parity * field[2]`` lands on a cell the widened cell-0
  mask zeroes, and the far ghost ``parity * field[reflect_row]`` lands on a cell the
  top-plane mask zeroes — on a folded PERIODIC axis. On a folded METALLIC axis the
  top plane IS stepped and the ghost there must be exactly zero, which under complex
  storage is the ``(+0.0, +0.0)`` word pair the inherited metallic ternary already
  delivers. So :func:`folded_bloch_curl_source` is ``complex_fields``' OWN certified
  template with ONE block inserted, and the insertion is performed by string surgery
  against a checked anchor rather than by copying the template — an edit to the
  certified complex curl reaches this family without a second edit, and a template
  whose anchor moved is a build-time failure rather than a silent divergence;
* the CONSTITUTIVE PAIR. ``update_H`` and the diagonal ``update_E`` read no
  neighbour, consult no ghost and run no ownership mask, so the fold reaches them
  through the STORED EXTENT alone and the PML coefficient vectors already carry it.
  This family adds NO device code on those two slots: it re-admits
  ``complex_fields.bloch_constitutive_step`` through
  ``complex_fields.ComplexConstitutivePlan``, and only the ADMISSION is new;
* the ownership masks, the ghost rule, the Bloch phase block, the plane-index decode
  and the fold's own axis-entry table, all IMPORTED from the two parents.

WHAT CHANGES UNDER COMPLEX STORAGE, and it is exactly one thing: the ghost fill is
ARITHMETIC. The array path writes a mirror ghost as ``part_products.scale(parity,
plane)`` (``stepping._write_mirror_ghost`` / ``_fill_folded_far_ghosts`` through
``host_writes.copy_face``): the parity ``+/-1`` rounded to the part dtype, times each
part, one float32 product per word. So the fill gets its own body here,
``c_mul_coefficient_left(c, plane)`` with the parity passed as one host-rounded float
(:func:`mirror_parity_coefficients`). A product by exactly ``+/-1.0f`` is the copy or
the sign flip on every normal word and zero, and on a subnormal word it flushes as
any arithmetic does on this backend; the array path's NumPy product flushes the same
words under the flush host policy, which is the policy these kernels run under.

THE FILL'S SUBNORMAL REACH IS THE CURL'S. The real fold's fill copies words and is
band-safe where its curl is not; this one multiplies, so it rides the same
subnormal-free precondition (or flush policy) as the curl. The fold is also the
configuration most likely to reach the band -- a folded run puts a deep-PML plane on
the far face, and the two fill passes read and write exactly those planes -- so any
census this family takes reports the WINDOW (first step, last step), not a count.

THE X, Y, Z FILL ORDER IS KEPT. With the parity a product by +/-1 per part the
composition at a doubly-unowned corner commutes exactly, so the order is the array
path's by transcription rather than by necessity; the plan applies the axes in X, Y,
Z order and holds the two passes separately, because the driver runs
``fill_symmetry_bc_*`` over every folded axis, then ``zero_metal_*``, then
``fill_folded_far_ghosts_*``, and the far pass reads a plane the wall clear can
touch. :class:`FoldedComplexFillPlan` inherits :class:`.symmetry.MirrorGhostFillPlan`'s
refusal of the one configuration where the fused ``run()`` would differ (a live far
pass on a grid with a live ``zero_metal`` axis).

THE CURL AND THE CONSTITUTIVE ARE THE COMPLEX FAMILY'S. :func:`folded_bloch_curl_source`
is ``complex_fields``' template with the top-plane mask inserted, so it carries the
chunk record, the per-chunk PML forms, the presence-keyed stencil and the plane-0
Bloch connection of the unfolded kernel unchanged; a folded axis carries no phase.

THERE IS NO GENERATED-CODE AUDIT ON THIS BACKEND. ``torch.mps.compile_shader``
exposes no disassembly, so the byte legs and the mutation legs are the only arbiters
and they are BEHAVIOURAL: they catch a wrong answer, not a wrong instruction.

===========================================================================
THE INVERTED CLAUSES — four families to stay disjoint from, not one
===========================================================================

``registry.py``'s contract is that every family's grid-reason list carries the same
numbered questions and answers exactly one of them THE OTHER WAY, and that a new
family STATES its inversion rather than inheriting disjointness. This family is the
hardest case in the package so far, because it is a product of two levers and each
lever has a family on each side of it. Every clause below is spelled in
:func:`_folded_complex_grid_reasons` with the number it carries there.

CLAUSE 2 (storage) INVERTS AGAINST THE REAL FAMILIES, INCLUDING THE REAL FOLD:

* ``coverage._grid_reasons`` clause 2 refuses complex storage, which covers
  ``pml_curl`` and ``constitutive``; ``bfast_curl``, ``offdiag_update_e`` and
  ``special_kz``'s real arm restate it;
* ``symmetry._folded_curl_grid_reasons`` clause 2 refuses ``force_complex_fields``
  BY NAME and names this family in the message; its constitutive twin and its FILL
  predicate do the same;
* HERE complex64 is REQUIRED — ``force_complex_fields`` OR a nonzero ``k_point``.

  THE TWO SPELLINGS OF CLAUSE 2 ARE NOT THE SAME PREDICATE and the gap was checked
  rather than assumed. The real fold asks ``force_complex_fields`` alone; this family
  also admits ``has_bloch``. A run with ``has_bloch`` and ``force_complex_fields``
  False is therefore not separated by clause 2 in both directions — it is separated
  by the real fold's CLAUSE 7 (``nonzero k_point``, on the curl and the constitutive)
  and, on the FILL slots where the real family carries no k clause, by the DTYPE:
  the real fill runs ``_layout_reasons``, which pins float32, while this family runs
  ``_complex_layout_reasons``, which pins complex64. One of the two must refuse any
  concrete run, because a volume has one dtype. The composition sweep MEASURES that
  rather than trusting this paragraph.

CLAUSE 5 (fold) INVERTS AGAINST THE UNFOLDED COMPLEX FAMILY:

* ``complex_fields._complex_grid_reasons`` refuses a fold by name — "a mirror plane
  is active (symmetry folding is not carried)" plus one line per folded axis;
* HERE a fold is REQUIRED, on the composition verdict and on every constitutive and
  fill verdict. :func:`folded_complex_pml_curl_coverage` deliberately ADMITS an
  unfolded grid so a gate can prove the reduction to the certified complex kernel;
  that verdict is NOT registered on a slot, for ``symmetry.py``'s reason — selecting
  between two valid products by branch order would make the numerical method depend
  on composer order. :func:`folded_complex_composition_curl_coverage` is the routing
  verdict.

CLAUSE 8 (beta) INVERTS AGAINST :mod:`.folded_beta`, WHICH NOW CARRIES THOSE SLOTS:

* ``special_kz``'s two arms refuse a fold by name (special_kz.py:838-845 real,
  :1055-1060 complex), so a folded beta run is refused there;
* HERE ``grid.beta`` must be ZERO, refused by name, and the message says whose it is.
  THIS PARAGRAPH WAS A PRE-REGISTERED INVERSION and :mod:`.folded_beta` honoured it
  rather than re-spelling the question the other way: that module's complex arm
  requires a fold AND a nonzero beta, so the two are separated by BOTH clauses in
  both directions and the sweep measures no admitted overlap.

CLAUSE 6 (row product) INVERTS AGAINST THE FOLDED COMPLEX OFF-DIAGONAL FAMILY, WHICH
IS ALSO NOT BUILT. The E side refuses an installed off-diagonal chi1inv row by name;
the CURL admits it, because the row product's whole effect is inside ``update_E``
(stepping.py:1001-1008) — the same per-sub-step split ``complex_fields`` makes.
``offdiag_update_e`` cannot take those rows either (it refuses complex storage), so
``update_E`` on a folded complex off-diagonal run is uncovered WITH A REASON FROM
BOTH SIDES.

CLAUSE 4 (cylindrical) is refused TWICE, for ``symmetry.py``'s three-way reason:
``_boundary_kinds`` puts ``is_axis`` AHEAD of ``is_mirrored`` (stepping.py:2191-2196)
so a folded r axis would report CYL_AXIS and the fold would vanish silently, and
``_mirror_phases`` (stepping.py:2346-2366) puts ``(-1)**grid.m`` into the SAME SLOT
the mirror phase occupies. The grid flag alone is not the inversion.

CLAUSE 9 (Bloch on a folded axis) IS THIS FAMILY'S OWN AND HAS NO TWIN. A folded axis
may carry NO phase and NO k component, zone edge included:
``stepping._bloch_phases`` raises ("a mirror plane reflects rather than repeating")
and ``driver._require_bloch_is_representable`` (driver.py:1047-1075) refuses the
configuration outright, so a kernel cannot lift what the array path will not run. A
Bloch phase on ANOTHER axis composes multiplicatively with no cross term and is
admitted. Neither parent predicate asks this question — ``complex_fields`` never sees
a fold and ``symmetry`` never sees a phase — so it is the one clause the composition
had to add rather than restate.

===========================================================================
WHAT THIS FAMILY DOES NOT CLAIM, each by name rather than by omission
===========================================================================

A nonzero ``grid.beta`` (:mod:`.folded_beta`'s complex arm, which carries it);
an off-diagonal chi1inv row on ``update_E`` (the folded complex off-diagonal product,
unbuilt); a registered susceptibility (complex-storage ADE is unbuilt on either
folded or unfolded complex); BFAST; a conductivity; an instantaneous nonlinearity;
cylindrical coordinates; and real float32 storage, which is the real fold's.

WIRED. Six arms — the folded complex curl on ``step_B``/``step_D``, the CERTIFIED
complex constitutive re-admitted on ``update_H``/``update_E``, and the folded complex
fill on ``fill_B``/``fill_D``. ``fastpath.plan_fast_path`` still returns ``None`` on
every branch, which is the separate decision this port does not make.

GATED BY ``parity/meep_gpu/gate_metal_folded_complex.py``. This module's own sources
carry no row in ``fingerprints.json``, as ``complex_fields``' do not; the gate hashes
the ones it launches into its own provenance block. This module's BYTES are hashed, under
``host_sha256``, because ``launch._host_modules()`` enumerates the package.

Import contract: importable WITHOUT torch. The predicates and the plan builders (to
``None``) must answer on a host with no GPU, which is the merge bar.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..triton_kernels.coverage import (
    CONSTITUTIVE_SIDES,
    CURL_SUB_STEPS,
    CURL_TARGETS,
    Coverage,
    _boundary_kinds,
    _call,
    _coefficient_reasons,
    _inverse_epsilon_reasons,
    _susceptibility_reasons,
)
from ..triton_kernels.launch import SUB_STEPS
from . import chunk_bounds as _chunk_bounds
from . import complex_fields, shaders, templates
from .coverage import _metal_backend_reasons, _residency_declaration_reasons
from .device import Residency, compile_source
from .plans import KernelPlan

# ONE DEFINITION, IMPORTED FROM THE TWO PARENTS. Every name below is a fact this
# family must not re-derive: the four boundary codes and their MIRROR split, the
# halved-origin source index, the Yee-shift table, the fill families and passes, the
# reflect-row reader, the fold classifier, the reduced-code mapping the certified
# emitters take, the top-plane mask block, the per-axis plane decode, the fold's
# axis-entry table and the wall-seam refusal. Re-deriving ANY of them here would be a
# second place to get the fold's single point of failure wrong.
from .symmetry import (  # noqa: F401 - re-exported deliberately
    CODE_METALLIC,
    CODE_MIRROR_METALLIC,
    CODE_MIRROR_PERIODIC,
    CODE_PERIODIC,
    FILL_PASSES,
    GHOST_FILL_FAMILIES,
    MIRROR_CODES,
    MIRROR_SOURCE_INDEX,
    TARGET_IYEE,
    _far_reflect_rows,
    _has_real_fold,
    _PLANE_INDEX,
    _reduced_codes,
    _wall_seam_reasons,
    folded_axis_kinds,
    folded_top_plane_mask,
    ghost_fill_axis_entries,
)

#: The family name the arm table carries. One spelling, so a refusal message, a
#: registry row and an artifact column cannot drift apart.
FAMILY = "folded_complex"


#: The two fill slots this family claims, and which ghost-fill family each names.
_FILL_SLOT_FAMILIES: Dict[str, str] = {"fill_B": "B", "fill_D": "D"}

#: Which constitutive side each slot names, so one pair of callables serves both.
_CONSTITUTIVE_SLOT_SIDES: Dict[str, str] = {"update_H": "H", "update_E": "E"}

__all__ = [
    "ARMS",
    "FAMILY",
    "FoldedComplexFillPlan",
    "FoldedComplexPmlCurlPlan",
    "compile_folded_bloch_curl",
    "compile_folded_mirror_fill_complex",
    "enumerate_folded_complex_sources",
    "fill_axis_entries",
    "folded_bloch_curl_source",
    "folded_complex_composition_curl_coverage",
    "folded_complex_constitutive_coverage",
    "folded_complex_fill_coverage",
    "folded_complex_pml_curl_coverage",
    "folded_curl_template",
    "folded_mirror_fill_complex_source",
    "mirror_parity_coefficients",
    "plan_folded_complex_constitutive",
    "plan_folded_complex_fill",
    "plan_folded_complex_fill_from_arrays",
    "plan_folded_complex_pml_curl",
    "plan_folded_complex_pml_curl_from_arrays",
    "register_arms",
]


# ---------------------------------------------------------------------------
# K1 — the folded complex curl, DERIVED from the certified complex template
# ---------------------------------------------------------------------------

#: The anchor the top-plane mask is inserted after: the certified complex curl's
#: cell-0 ownership mask slot. Checked to appear EXACTLY ONCE, so a template edit
#: that moves or duplicates it is a build failure rather than a kernel that masks the
#: wrong plane.
_MASK_ANCHOR = "__MASK__\n"

#: The block the fold adds, and the ONLY text in this kernel that neither parent
#: emits. It is the exact analogue of ``symmetry._FOLDED_CURL_TEMPLATE``'s, with the
#: complex zero: ``_mask_non_owned_cells``' ``if iyee[axis] != 0`` arm
#: (stepping.py:1934-1944) zeroes the LAST stored slot of every target whose Yee
#: shift is 1 on a folded PERIODIC axis, because that slot sits half a cell past
#: MEEP's ``big_corner``, outside ``owns`` (vec.cpp:445-462), so the FILL PASS writes
#: it and the curl must not. On a folded METALLIC axis the top plane IS stepped and
#: no line is emitted, which is why a sweep on one termination proves nothing about
#: the other.
_TOP_MASK_BLOCK = """
    // --- ownership mask, the TOP plane of a folded PERIODIC axis ----------------
    // The COMPLEMENT of the block above: every target whose Yee shift is 1 there.
    // Writes a complex zero to BOTH planes, as the array path does (S:1896, :1902).
    bool last_x = (i == nxi - 1), last_y = (j == nyi - 1), last_z = (k == nzi - 1);
__TOP_MASK__
"""


def folded_curl_template() -> str:
    """``complex_fields._CURL_TEMPLATE`` with ONE slot inserted, and nothing else.

    DERIVED RATHER THAN COPIED, and that is the composition made structural. A copy
    would be a second home for twenty-three bindings, the ghost gather, the Bloch
    rotation, the curl grouping and the split-field recurrence — every one of which
    is certified in the parent and none of which the fold changes. Deriving means an
    edit to the certified complex curl reaches this family in the same commit, and
    means the only text this module can be blamed for is :data:`_TOP_MASK_BLOCK`.

    The anchor is checked to occur exactly once. A parent edit that moved,
    duplicated or removed the cell-0 mask slot would otherwise silently produce a
    kernel with the top-plane mask in the wrong place, which is a plane of wrong
    values on every folded PERIODIC run and not a crash.
    """
    template = complex_fields._CURL_TEMPLATE
    occurrences = template.count(_MASK_ANCHOR)
    if occurrences != 1:
        raise RuntimeError(
            f"complex_fields._CURL_TEMPLATE carries the ownership-mask anchor "
            f"{_MASK_ANCHOR!r} {occurrences} times, not once; the folded complex "
            f"curl inserts its top-plane mask after that anchor and cannot place it "
            f"unambiguously. Re-read the parent template before editing this family")
    return template.replace(_MASK_ANCHOR, _MASK_ANCHOR + _TOP_MASK_BLOCK, 1)


def folded_bloch_curl_source(codes: Sequence[int], backward: bool,
                             phased: Sequence[int],
                             contract: str = shaders.CONTRACT_OFF) -> str:
    """The specialised folded ``bloch_pml_curl_step`` source for one configuration.

    ``codes`` is the per-axis PERIODIC / METALLIC / MIRROR_METALLIC /
    MIRROR_PERIODIC quadruple :func:`.symmetry.folded_axis_kinds` resolves — NEVER a
    hand-built triple, because the MIRROR_METALLIC / MIRROR_PERIODIC split is the
    fold's single point of failure and backwards on one axis is a plane of wrong
    values. ``phased`` is the per-axis Bloch flag.

    THE THREE FOLD DELTAS OVER THE CERTIFIED COMPLEX CURL, and all three are
    expressed as calls rather than as copied text: (1) both mirror codes take the
    METALLIC ghost branch and (2) the cell-0 mask widens from ``== METALLIC`` to
    ``!= PERIODIC`` — which are the SAME statement once every non-periodic code is
    spelled ``METALLIC`` on the way into ``templates.ghost`` and
    ``templates.ownership_mask``, which is what :func:`.symmetry._reduced_codes`
    does; and (3) a folded PERIODIC axis masks its LAST plane for every target whose
    Yee shift is 1 there, which is :func:`.symmetry.folded_top_plane_mask` with the
    complex zero. Nothing arithmetic changes.

    A PHASED AXIS MUST BE PLAIN PERIODIC. A mirror code carrying a Bloch flag is
    refused HERE as well as by the predicate, because a mis-baked flag is a plane of
    wrong values rather than a crash and this is the last place it can be seen.
    ``phased`` is the per-axis connection kind (``complex_fields.CONNECT_KINDS``).
    """
    codes = tuple(int(code) for code in codes)
    phased = tuple(int(flag) for flag in phased)
    if len(codes) != 3 or len(phased) != 3:
        raise ValueError(f"codes and phased must be per-axis triples, got "
                         f"{codes!r} / {phased!r}")
    for axis, (code, flag) in enumerate(zip(codes, phased)):
        if flag not in complex_fields.CONNECT_KINDS:
            raise ValueError(f"axis {axis} connection kind {flag!r} is not one of "
                             f"{complex_fields.CONNECT_KINDS}")
        if not flag:
            continue
        if code in MIRROR_CODES:
            raise ValueError(
                f"axis {axis} is folded and carries a Bloch phase; a mirror plane "
                f"reflects rather than repeating, so stepping._bloch_phases raises "
                f"(S:2338-2341) and driver._require_bloch_is_representable refuses "
                f"the configuration outright (driver.py:1047-1075)")
        if code != CODE_PERIODIC:
            raise ValueError(
                f"axis {axis} carries a Bloch phase but its ghost rule is METALLIC; "
                f"only a periodic wrap can carry a phase (S:2346-2360)")
    reduced = _reduced_codes(codes)
    flags = shaders.curl_membership_flags(backward)
    return templates.substitute(folded_curl_template(), {
        "__CONTRACT__": shaders.contraction_pragma(contract),
        "__HELPERS__": templates.complex_helpers(),
        "__RECORD_STRUCT__": _chunk_bounds.struct_source(),
        "__RECORD_BINDING__": _chunk_bounds.binding(complex_fields.CURL_RECORD_BUFFER),
        "__GUARD__": templates.GUARD,
        "__DECODE__": templates.DECODE_IJK,
        "__BOUNDS__": shaders.curl_bounds_block(backward),
        "__SHIFT__": "- 1" if backward else "+ 1",
        "__GHOST_X__": templates.ghost("x", reduced[0], backward),
        "__GHOST_Y__": templates.ghost("y", reduced[1], backward),
        "__GHOST_Z__": templates.ghost("z", reduced[2], backward),
        "__PHASE_X__": complex_fields._phase_block("x", phased[0]),
        "__PHASE_Y__": complex_fields._phase_block("y", phased[1]),
        "__PHASE_Z__": complex_fields._phase_block("z", phased[2]),
        "__MASK__": templates.ownership_mask(reduced, backward,
                                             zero=templates.COMPLEX_ZERO),
        "__TOP_MASK__": folded_top_plane_mask(codes, backward,
                                              zero=templates.COMPLEX_ZERO),
        "__DU0__": flags[0],
        "__DU1__": flags[1],
        "__DU2__": flags[2],
    })


def compile_folded_bloch_curl(codes: Sequence[int], backward: bool,
                              phased: Sequence[int],
                              contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised folded complex curl entry point for one configuration."""
    return compile_source(folded_bloch_curl_source(
        codes, backward, phased, contract)).bloch_pml_curl_step


# ---------------------------------------------------------------------------
# K2 — the folded complex ghost fill, THE ONE ARITHMETIC DELTA
# ---------------------------------------------------------------------------

#: ONE family's ghost plane on ONE axis for ONE pass, in one launch.
#:
#: NINE BINDINGS: three ``float2`` volume pointers, four uints, the runtime reflect
#: row, and ONE ``float`` parity. The reflect row STAYS RUNTIME (it is per-axis and a
#: single 3-D grid can carry both terminations with different rows), and so does the
#: PARITY: it is a host-rounded word the array path multiplies each part by
#: (``part_products.scale``), so it is passed rather than synthesised in-kernel.
_FILL_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

__HELPERS__

kernel void folded_mirror_fill_complex(
    device float2*      f0          [[buffer(0)]],
    device float2*      f1          [[buffer(1)]],
    device float2*      f2          [[buffer(2)]],
    constant uint&      nx          [[buffer(3)]],
    constant uint&      ny          [[buffer(4)]],
    constant uint&      nz          [[buffer(5)]],
    constant uint&      n_plane     [[buffer(6)]],
    constant int&       reflect_row [[buffer(7)]],
    constant float&     c           [[buffer(8)]],
    uint idx [[thread_position_in_grid]])
{
    // One thread per IN-PLANE position, and ONE PASS per launch. Every read and
    // every write a thread performs shares that thread's own `base`, so there is no
    // cross-thread hazard in this kernel at all: the near pass reads row 2 and
    // writes row 0 of its own column, the far pass reads `reflect_row` and writes
    // the last row of its own column. What the PREDICATE checks, rather than
    // assumes, is that those rows are DISTINCT and that the far pass's read row is
    // not a row the near launch already wrote.
    //
    // THE CELL ARITHMETIC IS THE REAL FILL'S, UNCHANGED. Binding `float2` rather
    // than a word view is what makes that true: the Triton twin steps complex64 as
    // float32 word pairs and has to address `2*off` and `2*off + 1`, which halves
    // its int32 element bound; here the index is a CELL index and the bound is the
    // cell count.
    if (idx >= n_plane) { return; }

    int nxi = int(nx), nyi = int(ny), nzi = int(nz);
    int ii  = int(idx);
__PLANE__

__BODY__
}
"""


def mirror_parity_coefficients(phase: int) -> Tuple[float, float]:
    """The ``(near, far)`` parity words for one folded axis.

    ``fields.mirror_parity(component, axis, phase) == phase * (1 - 2*iyee[c][axis])``,
    so the near fill's shift-0 components take ``+phase`` and the far fill's shift-1
    components take ``-phase`` -- two words per folded axis and nothing else is
    parity input. Each is the array path's ``part_products.scale`` coefficient: the
    Python int rounded to the part dtype, here ``numpy.float32``, exactly once and
    PASSED.
    """
    import numpy  # noqa: PLC0415

    if int(phase) not in (1, -1):
        raise ValueError(f"a mirror plane's declared phase is +1 or -1, got {phase!r}")
    near = float(numpy.float32(int(phase)))
    far = float(numpy.float32(-int(phase)))
    return near, far


def folded_mirror_fill_complex_source(axis: int, pass_name: str,
                                      shifts: Sequence[int],
                                      contract: str = shaders.CONTRACT_OFF) -> str:
    """One (axis, pass, Yee-shift triple) specialisation of the fill.

    ``shifts`` is the three targets' Yee shift ON THIS AXIS, in kernel argument order
    — ``TARGET_IYEE[name][axis]`` for the family's three components. It is not free:
    ``(family, axis)`` determines it, and :func:`fill_axis_entries` reads it off the
    table.

    THE NEAR PASS writes stored cell 0 from stored cell ``MIRROR_SOURCE_INDEX`` (= 2)
    for every component whose shift here is ZERO — MEEP's
    ``little_owned_corner0(c) = little_corner + 2 - iyee_shift(c)``, which on a halved
    grid (``io = -2``) excludes cell 0 for exactly the shift-0 components. THE FAR
    PASS writes the last stored slot from ``reflect_row`` for every component whose
    shift here is ONE, and runs only on a folded PERIODIC axis.

    THE MULTIPLY IS ``c_mul_coefficient_left(c, plane)``: the array path's
    ``part_products.scale(parity, plane)``, one float32 product per part. The operand
    order is the array path's spelling and moves no bit (IEEE multiplication is
    commutative).

    An empty body is REFUSED rather than emitted: a fill launch that writes nothing
    is a no-op that a before/after comparison reports as a pass.
    """
    if pass_name not in FILL_PASSES:
        raise ValueError(f"pass_name must be one of {FILL_PASSES}, got {pass_name!r}")
    if int(axis) not in _PLANE_INDEX:
        raise ValueError(f"axis must be 0, 1 or 2, got {axis!r}")
    shifts = tuple(int(s) for s in shifts)
    if len(shifts) != 3 or any(s not in (0, 1) for s in shifts):
        raise ValueError(f"shifts must be three Yee shifts of 0 or 1, got {shifts!r}")

    lines: List[str] = []
    if pass_name == "near":
        lines.append("    // near face (stepping._write_mirror_ghost): "
                     "cell 0 = (+phase) * cell 2, per part")
        want = 0
    else:
        lines.append("    // far face (stepping._fill_folded_far_ghosts): "
                     "last = (-phase) * cell reflect_row, per part, folded PERIODIC only")
        want = 1
    for slot, shift in enumerate(shifts):
        if shift != want:
            continue
        if want == 0:
            operand = f"f{slot}[base + {MIRROR_SOURCE_INDEX} * stride]"
            lines.append(f"    f{slot}[base] = c_mul_coefficient_left(c, {operand});")
        else:
            operand = f"f{slot}[base + reflect_row * stride]"
            lines.append(f"    f{slot}[base + last * stride] = "
                         f"c_mul_coefficient_left(c, {operand});")
    if len(lines) == 1:
        raise ValueError(
            f"the {pass_name!r} pass on axis {axis} would write nothing for shifts "
            f"{shifts!r}: an empty fill launch is a no-op that a before/after "
            f"comparison reports as a pass, so it is refused rather than emitted")
    if pass_name == "near":
        lines.append("    (void)last; (void)reflect_row;")

    return templates.substitute(_FILL_TEMPLATE, {
        "__CONTRACT__": shaders.contraction_pragma(contract),
        "__HELPERS__": templates.complex_helpers(),
        "__PLANE__": _PLANE_INDEX[int(axis)],
        "__BODY__": "\n".join(lines),
    })


def compile_folded_mirror_fill_complex(axis: int, pass_name: str,
                                       shifts: Sequence[int],
                                       contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised folded complex fill entry point for one configuration."""
    return compile_source(folded_mirror_fill_complex_source(
        axis, pass_name, shifts, contract)).folded_mirror_fill_complex


def enumerate_folded_complex_sources(contract: str = shaders.CONTRACT_OFF,
                                     ) -> Dict[str, str]:
    """Every shipped folded complex specialisation, keyed by a stable label.

    The shape ``shaders.enumerate_sources`` returns, so a gate records one sha256 per
    emitted source in its own provenance block. The curl's phase triple is enumerated
    only over the axes a given code quadruple can LEGALLY phase — a metallic or
    folded axis cannot carry one — so the count is the REACHABLE set and a label that
    cannot be built is not fingerprinted as if it could.
    """
    out: Dict[str, str] = {}
    codes = (CODE_PERIODIC, CODE_METALLIC, CODE_MIRROR_METALLIC, CODE_MIRROR_PERIODIC)
    for backward, name in ((False, "step_B"), (True, "step_D")):
        for cx in codes:
            for cy in codes:
                for cz in codes:
                    quad = (cx, cy, cz)
                    for phased in _reachable_phase_flags(quad):
                        label = (f"folded_bloch_pml_curl_step/{name}/{cx}{cy}{cz}/"
                                 f"ph{phased[0]}{phased[1]}{phased[2]}")
                        out[label] = folded_bloch_curl_source(
                            quad, backward, phased, contract)
    for family, spec in GHOST_FILL_FAMILIES.items():
        targets = tuple(spec["targets"])
        for axis in range(3):
            shifts = tuple(TARGET_IYEE[n][axis] for n in targets)
            for pass_name in FILL_PASSES:
                label = (f"folded_mirror_fill_complex/{family}/axis{axis}/"
                         f"{pass_name}")
                out[label] = folded_mirror_fill_complex_source(
                    axis, pass_name, shifts, contract)
    return out


def _reachable_phase_flags(codes: Sequence[int]) -> Tuple[Tuple[int, int, int], ...]:
    """Which per-axis Bloch flags a code quadruple can legally carry.

    A PLAIN PERIODIC axis may be phased; a METALLIC one has no lattice vector and a
    FOLDED one reflects rather than repeating, so neither may. That is narrower than
    ``complex_fields._reachable_phase_flags``, which never sees a mirror code, and
    the narrowing is this family's clause 9 expressed as an enumeration.
    """
    options = [complex_fields.CONNECT_KINDS if int(code) == CODE_PERIODIC
               else (complex_fields.CONNECT_NONE,) for code in codes]
    return tuple((x, y, z) for x in options[0] for y in options[1] for z in options[2])


# ---------------------------------------------------------------------------
# Coverage — positive refusal enumeration, with FOUR inverted clauses
# ---------------------------------------------------------------------------
#
# The clause numbering mirrors `complex_fields._complex_grid_reasons` so the two can
# be diffed. Clause 2 is that module's own INVERSION (complex storage required) and
# is KEPT; clause 5 is INVERTED AGAINST IT (a fold is required where it refuses one);
# clause 8's beta half is kept and clause 9 is NEW. Clause 4's boundary whitelist is
# REPLACED by `folded_axis_kinds` rather than widened — widening the shared
# `COVERED_BOUNDARIES` tuple would silently admit folds into every plain arm at once.
#
# Everything else is RESTATED rather than subtracted. The shipped reason lists are
# built inside functions whose clauses cannot be removed from outside, and reaching a
# predicate by subtracting another's output is how a clause silently goes missing.


def _folded_complex_axis_reasons(grid: Any, pml: Any) -> Tuple[
        Optional[Tuple[int, int, int]], List[str]]:
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
    """CLAUSE 5, INVERTED against ``complex_fields`` — asked TWICE, deliberately.

    Once of the GRID (``has_symmetry`` and some ``is_mirrored`` axis) and once of the
    RESOLVED CODES (some axis landed on a mirror code). The two can disagree — a grid
    can declare a mirror the boundary resolver reports as something else, which is
    precisely the cylindrical hazard — and a family that asked only the first would
    admit a run whose kernel carries no fold at all.
    """
    reasons: List[str] = []
    if not _has_real_fold(grid):
        reasons.append(f"no mirror plane is active: {what}")
    elif codes is not None and not any(int(code) in MIRROR_CODES for code in codes):
        reasons.append("no axis resolves to a mirror boundary: the grid declares a "
                       "fold that stepping._boundary_kinds does not report as one")
    return reasons


def _folded_phase_reasons(grid: Any, kinds: Optional[Sequence[str]],
                          codes: Optional[Sequence[int]]) -> List[str]:
    """CLAUSE 9 — per-axis Bloch legality INCLUDING the fold clause. This family's own.

    Three questions, and the third has no twin in either parent. A phased axis must
    resolve PERIODIC (``stepping._bloch_phases`` raises otherwise, S:2346-2360); a
    metallic axis must carry a k component of exactly 0; and NO FOLDED AXIS MAY CARRY
    ANY PHASE OR ANY NONZERO k COMPONENT, zone edge included, because
    ``driver._require_bloch_is_representable`` (driver.py:1047-1075) refuses the
    configuration outright and a kernel cannot lift what the array path will not run.

    The phase is read DEFENSIVELY rather than through ``_call``: an unreadable phase
    is not an unphased one, and admitting one would let ``bloch_phase_table`` raise
    inside a plan builder whose refusal contract is None-only.
    """
    reasons: List[str] = []
    if kinds is None:
        # UNRESOLVED IS A REFUSAL, NOT A SKIP. Returning an empty list here would
        # make an unreadable boundary table look like a phase-free one, which is the
        # admission-by-absence this whole clause list refuses everywhere else.
        return ["boundary kinds could not be resolved for this grid, so per-axis "
                "Bloch legality could not be asked"]
    k_point = tuple(getattr(grid, "k_point", (0.0, 0.0, 0.0)))
    phase_reader = getattr(grid, "bloch_phase", None)
    if not callable(phase_reader):
        reasons.append("grid.bloch_phase is missing or not callable; an unreadable "
                       "phase table is not an unphased one")
    for axis, kind in enumerate(kinds):
        folded = codes is not None and int(codes[axis]) in MIRROR_CODES
        if callable(phase_reader):
            try:
                phase = phase_reader(axis)
            except Exception as exc:  # noqa: BLE001 - unreadable is refused
                reasons.append(f"grid.bloch_phase({axis}) raised {exc!r}; an "
                               f"unreadable phase is not an unphased one")
            else:
                if phase is not None and kind != "periodic":
                    reasons.append(
                        f"axis {axis} carries Bloch phase {phase!r} but resolved to "
                        f"{kind!r}; only a periodic wrap can carry a phase")
                if phase is not None and folded:
                    reasons.append(
                        f"axis {axis} is folded and carries Bloch phase {phase!r}; "
                        f"driver._require_bloch_is_representable refuses a nonzero k "
                        f"on a mirror plane's own axis, zone EDGE included")
        if kind == "metallic" and float(k_point[axis]) != 0.0:
            reasons.append(
                f"axis {axis} is metallic with k component {k_point[axis]!r}; a PEC "
                f"wall gives the axis no lattice vector for the phase")
        if folded and float(k_point[axis]) != 0.0:
            reasons.append(
                f"axis {axis} is folded with k component {k_point[axis]!r}; k must "
                f"be EXACTLY 0 on every folded axis (zone edge included)")
    return reasons


def _folded_complex_grid_reasons(fields: Any, pml: Any, grid: Any, residency: Any,
                                 ) -> Tuple[List[str],
                                            Optional[Tuple[int, int, int]]]:
    """The clauses every predicate in this family shares, plus the resolved codes."""
    reasons: List[str] = []

    # 1. The Metal backend and the host array module the mirrors copy from.
    reasons.extend(_metal_backend_reasons(grid))
    reasons.extend(_residency_declaration_reasons(residency))

    # 2 (INVERTED against every REAL family, the real fold included). Complex64
    #    storage REQUIRED. Real storage bound as float2 is the wrong-stride wrong
    #    answer in the other direction, and under real storage the fold's parity IS
    #    an exact sign flip, which is `symmetry.py`'s family and not this one.
    if not (getattr(fields, "force_complex_fields", False)
            or getattr(grid, "has_bloch", False)):
        reasons.append(
            "storage is real float32 (neither force_complex_fields nor a nonzero "
            "k_point): under real storage the fold's parity is an exact sign flip "
            "and that is the folded real family's product")

    # 3. An absorber that actually absorbs. Without one the plain path is the
    #    bit-identical one (stepping._pml_is_active).
    if pml is None or not getattr(pml, "is_active", False):
        reasons.append("no active PML layer (this product implements the complex "
                       "split-field path only)")

    # 4/5. The ghost rules, INCLUDING the two folded ones. `COVERED_BOUNDARIES` is
    #      NOT consulted and NOT widened; `folded_axis_kinds` replaces that clause.
    codes, fold_reasons = _folded_complex_axis_reasons(grid, pml)
    reasons.extend(fold_reasons)

    # 4b. Cartesian only, refused TWICE. `_boundary_kinds` puts `is_axis` AHEAD of
    #     `is_mirrored` (stepping.py:2191-2196), so a folded r axis reports CYL_AXIS
    #     and the fold vanishes silently; and `_mirror_phases` (stepping.py:2346-2366)
    #     puts (-1)**grid.m into the SAME SLOT the mirror phase occupies, so reading
    #     a phase without first refusing `is_axis` reads an m-factor as a parity.
    if getattr(grid, "cylindrical", False):
        reasons.append("cylindrical (Dcyl) coordinates are not carried")
    for axis in range(3):
        if _call(grid, "is_axis", axis, default=False):
            reasons.append(f"axis {axis} is the cylindrical r = 0 axis")

    # 9 (NEW — neither parent asks it). Per-axis Bloch legality with the fold clause.
    active_pml = pml if (pml is not None
                         and getattr(pml, "is_active", False)) else None
    reasons.extend(_folded_phase_reasons(grid, _boundary_kinds(grid, active_pml),
                                         codes))

    # 6. No conductivity on any curl target, refused OUTRIGHT on an unreadable
    #    reader: inferring "no conductivity" from the ABSENCE of `condfac_for` is
    #    admission by attribute absence, and an electric conductivity has no
    #    fallback flag at all.
    reader = getattr(fields, "condfac_for", None)
    if not callable(reader):
        reasons.append("fields does not expose condfac_for; an unreadable "
                       "conductivity table is not an absent one")
    else:
        for target in CURL_TARGETS:
            try:
                conductive = reader(target) is not None
            except Exception as exc:  # noqa: BLE001 - unreadable means not covered
                reasons.append(f"condfac_for({target!r}) raised {exc!r}")
                continue
            if conductive:
                reasons.append(f"a conductivity is installed on {target}; conductive "
                               f"folded complex stepping is a separate, unbuilt "
                               f"product")

    # 7. No dispersion. Complex-storage ADE is unbuilt folded or not; the shape
    #    clauses still run so an unreadable susceptibility is NAMED.
    if getattr(fields, "has_polarizations", False) or (
            getattr(fields, "polarizations", ()) or ()):
        reasons.append("a susceptibility is registered: complex-storage ADE is a "
                       "later tranche, folded or not")
    reasons.extend(_susceptibility_reasons(fields))

    # 8. No nonlinearity, no BFAST, and — INVERTED against the folded beta family,
    #    which nothing on this backend carries — no nonzero grid.beta. Both the
    #    BFAST term and the beta increment land INSIDE the fold's widened mask
    #    (added at stepping.py:371-396 / :459-478, masked at :397 / :479), so the
    #    pairings are a coordinated follow-up rather than an additive one.
    if getattr(fields, "has_nonlinearity", False):
        reasons.append("chi2/chi3 is installed (the Pade constitutive factor is not "
                       "carried, and a nonlinear fold has no single parity)")
    if getattr(grid, "bfast_active", False):
        reasons.append("BFAST is active (the second additive curl term is not "
                       "carried, and a fold widens the mask that term lands inside)")
    if float(getattr(grid, "beta", 0.0)) != 0.0:
        reasons.append(f"special_kz beta={getattr(grid, 'beta', 0.0)!r} is nonzero "
                       f"(that is grid.beta, not a Bloch phase; the beta increment "
                       f"lands inside the fold's widened mask, and that pairing is "
                       f"folded_beta's complex arm, which CARRIES it)")

    # 9c. Stored E — the load-bearing invariant behind admitting anything on the
    #     curl. The curl differences the STORED E and H arrays.
    if not getattr(fields, "stores_E", False):
        reasons.append("E is recomputed from D rather than stored")

    return reasons, codes


def _stored_extent_reasons(codes: Optional[Sequence[int]],
                           shape: Sequence[int]) -> List[str]:
    """A folded axis must be able to hold its own ghost planes.

    ``folded_axis_kinds`` already refuses fewer than three stored cells; this is the
    shape-side companion for the top-plane mask, which needs the last stored slot to
    be distinct from cell 0.
    """
    if codes is None or len(shape) != 3:
        return []
    out: List[str] = []
    for axis, code in enumerate(codes):
        if int(code) in MIRROR_CODES and int(shape[axis]) <= MIRROR_SOURCE_INDEX:
            out.append(f"axis {axis} is folded with {int(shape[axis])} stored cells; "
                       f"the near ghost images stored cell {MIRROR_SOURCE_INDEX}")
    return out


def folded_complex_pml_curl_coverage(fields: Any, pml: Any, sub_step: str,
                                     residency: Any = None) -> Coverage:
    """May the folded complex curl step this (fields, pml, sub_step)? THE WIDE VERDICT.

    ZERO FOLDED AXES IS ADMITTED HERE, and that is not an oversight: with every axis
    resolving to PERIODIC/METALLIC, :func:`folded_bloch_curl_source` reduces to
    ``complex_fields.bloch_curl_source``'s output CHARACTER FOR CHARACTER except for
    an empty top-plane block, which is a property a gate MEASURES rather than a claim.
    THIS VERDICT MUST NOT BE REGISTERED ON A SLOT — it would make every unfolded
    complex row ambiguous against the certified family.
    :func:`folded_complex_composition_curl_coverage` is the routing verdict.

    OFF-DIAGONAL EPSILON IS ADMITTED HERE and refused by the E-side constitutive
    predicate: the row product's whole effect is inside ``update_E``
    (stepping.py:1001-1008). The same per-sub-step split ``complex_fields`` makes.
    """
    if sub_step not in CURL_SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(CURL_SUB_STEPS)}, "
                         f"got {sub_step!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    reasons, codes = _folded_complex_grid_reasons(fields, pml, grid, residency)

    spec = SUB_STEPS[sub_step]
    names = (tuple(spec["targets"])
             + tuple("fu_" + name for name in spec["targets"])
             + tuple(spec["sources"]))
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")

    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(complex_fields._complex_layout_reasons(fields, shape, names))
    if pml is not None and getattr(pml, "is_active", False) and len(shape) == 3:
        # BOTH sub-lattices, as the shipped predicates check: the suffix the plan
        # binds is the sub-step's own, and a swap is the half-cell mutation a gate
        # carries. On a folded axis `grid.shape[axis]` IS the stored count, which is
        # where "the coefficient vectors are built at the folded stored extent"
        # stops being an assumption.
        reasons.extend(_coefficient_reasons(pml, shape, ("kms", "sinv"), ("", "_h")))
    reasons.extend(_stored_extent_reasons(codes, shape))
    return Coverage(not reasons, tuple(reasons))


def folded_complex_composition_curl_coverage(fields: Any, pml: Any, sub_step: str,
                                             residency: Any = None) -> Coverage:
    """THE ROUTING VERDICT — the wide one with a fold MANDATORY.

    The one clause that separates this family from ``complex_fields`` on the two curl
    slots, spelled as its own function so a reader can see the inversion in one
    place. Registering the wide verdict instead would make every unfolded complex row
    ambiguous and would leave the numerical method depending on composer order.
    """
    if sub_step not in CURL_SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(CURL_SUB_STEPS)}, "
                         f"got {sub_step!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))
    base = folded_complex_pml_curl_coverage(fields, pml, sub_step, residency)
    codes, _ = _folded_complex_axis_reasons(grid, pml)
    reasons = list(base.reasons)
    reasons.extend(_requires_a_fold(
        grid, codes,
        "the folded complex curl is an equivalence product here, not a composition "
        "candidate; an unfolded complex grid belongs to the certified complex kernel"))
    return Coverage(not reasons, tuple(reasons))


def folded_complex_constitutive_coverage(fields: Any, pml: Any, side: str,
                                         residency: Any = None) -> Coverage:
    """May the CERTIFIED complex constitutive kernel step a FOLDED complex run?

    No new kernel and no new arithmetic: only the ADMISSION is this family's.
    ``complex_fields.complex_constitutive_coverage`` correctly refuses a folded grid,
    because the unfolded product family cannot step one; this verdict proves the
    folded extent and the coefficient lengths explicitly instead of weakening that
    family globally.

    THE FOLD REACHES THIS SUB-STEP THROUGH EXACTLY ONE THING: THE STORED EXTENT.
    ``update_H`` (stepping.py:907-924) and the diagonal ``update_E`` (:1012-1020) are
    three ``_apply_constitutive_pml`` calls over integer-position coefficients, with
    no shift helper, no ghost and no ownership mask. The PML coefficient vectors are
    already built at the folded stored extent, and the layout clauses below are where
    that stops being an assumption.

    THE E SIDE REFUSES WHAT CHANGES WHAT ``source`` IS, each naming its owner: an
    off-diagonal chi1inv row (the row product READS NEIGHBOURS through
    ``_shift_down``/``_shift_up``, and on a fold the partner-axis down shift is a
    LIVE interior plane — the folded complex off-diagonal product, unbuilt here), and
    ``stores_E`` false. A registered susceptibility is already refused module-wide.
    """
    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, "
                         f"got {side!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    spec = CONSTITUTIVE_SIDES[side]
    reasons, codes = _folded_complex_grid_reasons(fields, pml, grid, residency)
    reasons.extend(_requires_a_fold(
        grid, codes,
        "the folded complex constitutive product has no array-path work to "
        "specialize; an unfolded complex grid belongs to the certified complex pair"))

    if side == "E" and getattr(fields, "has_offdiagonal_epsilon", False):
        reasons.append(
            "an off-diagonal chi1inv row is installed (the row product reads "
            "neighbours; this sub-step is element-wise). On a fold the partner-axis "
            "down shift is a live interior plane, which is the folded complex "
            "off-diagonal product and is not carried on this backend")

    names = tuple(spec["targets"]) + tuple(spec["aux"]) + tuple(spec["sources"])
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")

    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(complex_fields._complex_layout_reasons(fields, shape, names))
    if side == "E" and len(shape) == 3:
        # inv_eps stays float32 under complex storage (stepping.py:41-50,
        # fields.py:1203-1204), so the shipped float32 pin is exactly right.
        reasons.extend(_inverse_epsilon_reasons(fields, shape))
    if pml is not None and getattr(pml, "is_active", False) and len(shape) == 3:
        suffix = ("_h",) if spec["half_integer"] else ("",)
        reasons.extend(_coefficient_reasons(pml, shape, ("kps", "kms"), suffix))
    reasons.extend(_stored_extent_reasons(codes, shape))
    return Coverage(not reasons, tuple(reasons))


def folded_complex_fill_coverage(fields: Any, family: str, residency: Any = None) -> Coverage:
    """May the folded complex fill write this family's ghost planes?

    NARROWER THAN THE CURL'S PREDICATE AND INDEPENDENT OF IT: this pass reads no
    coefficient, no source and no PML, so it has no Courant number, no sub-lattice
    and no absorber clause — ``fill_symmetry_bc_*`` runs whether or not an absorber
    is installed. A PARTIAL FOLD COVERAGE IS THEREFORE LEGAL AND HAPPENS, exactly as
    it does for the real fold: on a folded complex no-PML run these two slots are
    covered while every curl and constitutive slot is on the array path.

    What it DOES need: complex64 storage, a real fold, a readable phase, no phase or
    k on the folded axis, a stored extent that can hold both planes, a reflect row on
    every folded PERIODIC axis.

    THE WALL SEAM IS REFUSED BY NAME, through the REAL family's own helper. The
    reason is identical and the helper is imported rather than restated:
    ``zero_metal_*`` runs BETWEEN the two fill passes (driver.py:3286 / :3301) and
    the far pass reads a
    whole plane that includes the cells the wall clear has just touched, so running
    the two back to back is a different answer. The passes are separately launchable;
    until a driver adapter places them, the pairing is refused.
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

    # 2 (INVERTED against symmetry.mirror_ghost_fill_coverage, which refuses complex
    #    storage by name and says why: under REAL storage the parity IS an exact sign
    #    flip on this backend and that family is not replaced here).
    if not (getattr(fields, "force_complex_fields", False)
            or getattr(grid, "has_bloch", False)):
        reasons.append(
            "storage is real float32: the real fold's fill is "
            "symmetry.mirror_ghost_fill, whose parity is a sign-bit operation and "
            "EXACT there (measured on this host over signed zeros, the whole "
            "subnormal band and the normal range); it is not replaced here")

    if getattr(grid, "cylindrical", False):
        reasons.append("cylindrical (Dcyl) coordinates are not carried")
    for axis in range(3):
        if _call(grid, "is_axis", axis, default=False):
            reasons.append(f"axis {axis} is the cylindrical r = 0 axis")

    codes, fold_reasons = _folded_complex_axis_reasons(grid, None)
    reasons.extend(fold_reasons)
    reasons.extend(_requires_a_fold(
        grid, codes,
        "the array path's fill passes return immediately (stepping.py:1482-1483) "
        "and there is nothing to replace"))
    # A folded axis carries no wrap factor at all (stepping._far_reflect_rows:1677),
    # so the same phase legality the curl asks is asked here.
    reasons.extend(_folded_phase_reasons(grid, _boundary_kinds(grid, None), codes))
    reasons.extend(_wall_seam_reasons(grid, codes))

    rows = _far_reflect_rows(grid)
    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(_stored_extent_reasons(codes, shape))
    if codes is not None and len(shape) == 3:
        for axis, code in enumerate(codes):
            if int(code) != CODE_MIRROR_PERIODIC:
                continue
            extent = int(shape[axis])
            row = rows[axis] if rows is not None else None
            if row is None:
                reasons.append(f"axis {axis} is a folded PERIODIC axis with no "
                               f"reflect row from stepping._far_reflect_rows")
                continue
            row = int(row)
            # THE SAME THREE CHECKS THE REAL FILL MAKES, and for the same reasons:
            # the row must be inside the allocation and below the plane the far pass
            # writes; it must not be stored cell 0, which the NEAR launch has already
            # written by the time the far launch runs; and an extent this small makes
            # the far pass's WRITE plane the near pass's READ plane, which is safe
            # only in the near-then-far order while this plan also offers a fused
            # run().
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
    reasons.extend(complex_fields._complex_layout_reasons(fields, shape, names))
    return Coverage(not reasons, tuple(reasons))


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------

class FoldedComplexPmlCurlPlan(KernelPlan):
    """A launchable, allocation-free FOLDED COMPLEX split-field PML curl sub-step.

    ``complex_fields.ComplexPmlCurlPlan``'s bindings EXACTLY — the fold adds no
    argument, because the stored extent is what carries it — over a different kernel.
    DELIBERATELY A SIBLING RATHER THAN A SUBCLASS: the two hold the same argument
    tuple and launch different compiled functions, and a subclass would inherit
    nothing but the risk that a future edit to the parent's ``__init__`` silently
    rebinds this one.

    Built two ways and launched ONE way — :func:`plan_folded_complex_pml_curl` from
    the engine's own objects, :func:`plan_folded_complex_pml_curl_from_arrays` from
    bare host arrays for a gate — so the bytes a gate certifies are the bytes the
    engine would launch.
    """

    __slots__ = ("sub_step", "shape", "n_elem", "dtdx", "backward", "bc", "phased",
                 "phase_values", "residency", "volumes", "chunk_bounds")

    family = "folded complex PML curl"

    REPR_FIELDS = ("sub_step", "shape", "bc", "phased")

    def __init__(self, sub_step: str, shape, dtdx: float, bc, phased, phase_values, residency: Any,
                 targets, auxiliaries, sources,
                 coefficients, functions: Dict[str, Any],
                 volumes: Sequence[str], *, chunk_bounds: Any) -> None:
        if sub_step not in SUB_STEPS:
            raise ValueError(
                f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
        self.sub_step = sub_step
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.dtdx = float(dtdx)
        self.backward = bool(SUB_STEPS[sub_step]["backward"])
        self.bc = tuple(int(code) for code in bc)
        self.phased = tuple(int(flag) for flag in phased)
        self.phase_values = tuple((float(re), float(im)) for re, im in phase_values)
        self.residency = residency
        self.volumes = tuple(volumes)
        self.chunk_bounds = _chunk_bounds.plan_words(
            chunk_bounds, self.shape, f"FoldedComplexPmlCurlPlan({sub_step})")
        super().__init__(functions, (
            tuple(targets) + tuple(auxiliaries) + tuple(sources) + tuple(coefficients)
            + (self.shape[0], self.shape[1], self.shape[2], self.n_elem, self.dtdx)
            + self.phase_values
            + (_chunk_bounds.tensor(self.chunk_bounds,
                                    str(getattr(residency, "device", "mps"))),)))


def _curl_functions(codes, backward: bool, phased,
                    contract_variants: Sequence[str]) -> Dict[str, Any]:
    return {mode: compile_folded_bloch_curl(codes, backward, phased, mode)
            for mode in contract_variants}


def plan_folded_complex_pml_curl(fields: Any, pml: Any, sub_step: str,
                                 residency: Optional[Residency] = None,
                                 contract_variants: Sequence[str] = (
                                     shaders.CONTRACT_OFF,), *, chunk_bounds: Any,
                                 ) -> Optional[FoldedComplexPmlCurlPlan]:
    """Build a folded complex curl plan from the engine's objects, or None.

    None is the only refusal: a configuration this kernel does not carry must fall
    back to the array path, never raise into a caller that would otherwise have
    stepped correctly.

    The codes come from :func:`.symmetry.folded_axis_kinds` and from nowhere else —
    building them from ``_boundary_kinds`` alone would lose the MIRROR_METALLIC /
    MIRROR_PERIODIC split, which decides the top-plane mask.
    """
    if sub_step not in SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, "
                         f"got {sub_step!r}")
    if not folded_complex_composition_curl_coverage(
            fields, pml, sub_step, residency).covered:
        return None
    grid = fields.grid
    codes, _ = folded_axis_kinds(grid, pml)
    if codes is None:  # pragma: no cover - the predicate already refused
        return None

    spec = SUB_STEPS[sub_step]
    kinds = _boundary_kinds(grid, pml)
    flags, values = complex_fields.phase_arguments(
        complex_fields.bloch_phase_table(grid, kinds),
        backward=bool(spec["backward"]))
    mirror = complex_fields._complex_mirror
    targets = [mirror(residency, n, getattr(fields, n)) for n in spec["targets"]]
    auxiliaries = [mirror(residency, "fu_" + n, getattr(fields, "fu_" + n))
                   for n in spec["targets"]]
    sources = [mirror(residency, n, getattr(fields, n)) for n in spec["sources"]]
    coefficients = [
        residency.mirror(f"pml:{stem}_{axis}{spec['suffix']}",
                         getattr(pml, f"{stem}_{axis}{spec['suffix']}"),
                         constant=True)
        for axis in "xyz" for stem in ("kms", "sinv")]
    volumes = (tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
               + tuple(spec["sources"]))
    return FoldedComplexPmlCurlPlan(
        sub_step, grid.shape, grid.dt / grid.dx, codes, flags, values,
        residency, targets, auxiliaries, sources, coefficients,
        _curl_functions(codes, spec["backward"], flags, contract_variants),
        volumes, chunk_bounds=chunk_bounds)


def plan_folded_complex_pml_curl_from_arrays(
        sub_step: str, arrays: Dict[str, Any], flat: Dict[str, Any], codes,
        phases: Sequence[Optional[complex]], dtdx: float,
        residency: Residency, functions: Optional[Dict[str, Any]] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        *, chunk_bounds: Any) -> FoldedComplexPmlCurlPlan:
    """Build a folded complex curl plan from bare host arrays — a gate's route.

    ``codes`` is the FOUR-valued per-axis quadruple and ``phases`` the per-axis
    ``Optional[complex]`` table of ``eikna``; the side's connection is chosen HERE,
    per sub-step, exactly as the engine route chooses it. ``chunk_bounds`` is the
    record of the division the harness's reference steps with. No predicate runs: the caller
    is a harness that constructed the configuration deliberately, including the
    deliberately wrong ones. ``functions`` is the MUTATION SEAM — dropping it is not
    a silent slowdown but a silent DISARMING, since every mutation leg would then
    launch the shipped kernel and report the defect as uncaught.
    """
    spec = SUB_STEPS[sub_step]
    shape = tuple(int(n) for n in arrays[spec["targets"][0]].shape)
    flags, values = complex_fields.phase_arguments(
        tuple(phases), backward=bool(spec["backward"]))
    mirror = complex_fields._complex_mirror
    targets = [mirror(residency, n, arrays[n]) for n in spec["targets"]]
    auxiliaries = [mirror(residency, "fu_" + n, arrays["fu_" + n])
                   for n in spec["targets"]]
    sources = [mirror(residency, n, arrays[n]) for n in spec["sources"]]
    coefficients = [residency.mirror(f"pml:{stem}_{axis}:{sub_step}",
                                     flat[f"{stem}_{axis}"], constant=True)
                    for axis in "xyz" for stem in ("kms", "sinv")]
    volumes = (tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
               + tuple(spec["sources"]))
    return FoldedComplexPmlCurlPlan(
        sub_step, shape, dtdx, codes, flags, values, residency,
        targets, auxiliaries, sources, coefficients,
        functions if functions is not None
        else _curl_functions(codes, spec["backward"], flags,
                             contract_variants),
        volumes, chunk_bounds=chunk_bounds)


class FoldedComplexFillPlan:
    """One family's COMPLEX mirror ghost planes, as one launch per (pass, folded axis).

    DELIBERATELY A SIBLING OF :class:`.plans.KernelPlan` RATHER THAN A SUBCLASS, for
    :class:`.symmetry.MirrorGhostFillPlan`'s reason: that base's whole contract is
    that ``run`` unpacks ONE argument tuple and calls ONE function, and this plan
    cannot honour it — the array path applies the axes in X, Y, Z ORDER and ordering
    inside one dispatch is not something a grid of threads can promise.

    THE ORDER IS THE ARRAY PATH'S. Every fill is a product by exactly +/-1 per part,
    so a corner unowned on two axes carries the product of both parities in either
    order; the plan keeps the X, Y, Z order by transcription.

    THE TWO PASSES ARE SEPARATE ATTRIBUTES, NOT ONE LIST, for two reasons that stack:
    ``zero_metal_*`` runs between them in the driver (driver.py:3286 / :3301), and a
    component unowned on two axes with DIFFERENT Yee shifts sees near-then-far on the
    array path where a fused per-axis launch would give it far-then-near.
    :meth:`run` walks near-then-far for a caller that has established the wall is not
    live — which :func:`folded_complex_fill_coverage` refuses to admit otherwise —
    and :meth:`run_near` / :meth:`run_far` are what a driver adapter places.

    ``replaces_sub_steps`` IS DECLARED, NOT INFERRED. The composer reads it to learn
    that a plan filling ``fill_B`` also performs ``fill_folded_far_ghosts_B`` on the
    device, which is a residency fact no slot name carries.
    """

    __slots__ = ("family", "slot", "shape", "residency", "volumes",
                 "near", "far", "replaces_sub_steps", "launches", "_functions")

    #: This plan launches kernels; the composer's planned/null split reads it.
    performs_device_work = True

    def __init__(self, family: str, slot: str, shape,
                 residency: Residency, near: Sequence[Dict[str, Any]],
                 far: Sequence[Dict[str, Any]],
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
                                int(entry["reflect_row"]), entry["coefficient"])

    def run_near(self, contract: Optional[str] = None) -> None:
        """``fill_symmetry_bc_*`` for every folded axis, in X, Y, Z order. In place."""
        self._walk(self.near, shaders.CONTRACT_OFF if contract is None else contract)

    def run_far(self, contract: Optional[str] = None) -> None:
        """``fill_folded_far_ghosts_*`` for every folded PERIODIC axis. In place."""
        self._walk(self.far, shaders.CONTRACT_OFF if contract is None else contract)

    def run(self, contract: Optional[str] = None) -> None:
        """Both passes, near then far — the driver's order with the wall clear out.

        Only correct where ``_zero_metal`` performs no work, which is what
        :func:`folded_complex_fill_coverage` refuses to admit otherwise.
        """
        self.run_near(contract)
        self.run_far(contract)

    def describe(self) -> str:
        return (f"FoldedComplexFillPlan({self.family}, shape={self.shape}, "
                f"near={[e['axis'] for e in self.near]}, "
                f"far={[e['axis'] for e in self.far]}, "
                f"variants={self.variants})")

    def __repr__(self) -> str:
        return self.describe()


def fill_axis_entries(grid: Any, family: str,
                      pass_name: str) -> Tuple[Dict[str, Any], ...]:
    """The real fold's axis entries, plus this family's parity COEFFICIENT WORDS.

    ``symmetry.ghost_fill_axis_entries`` already resolves everything the fold decides
    — the folded axes in X, Y, Z order, the declared phase, the reflect row from
    ``stepping._far_reflect_rows`` and the three targets' Yee shifts — so it is
    called rather than restated. What this family adds is the one thing the real fill
    does not carry: the parity word the per-part multiply consumes, host-rounded once
    through :func:`mirror_parity_coefficients`.
    """
    entries = ghost_fill_axis_entries(grid, family, pass_name)
    out: List[Dict[str, Any]] = []
    for entry in entries:
        near_word, far_word = mirror_parity_coefficients(int(entry["phase"]))
        item = dict(entry)
        item["coefficient"] = near_word if pass_name == "near" else far_word
        out.append(item)
    return tuple(out)


def _fill_functions(entries: Sequence[Dict[str, Any]],
                    contract_variants: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Compile one entry point per distinct (axis, pass, shifts) key.

    NOT KEYED ON THE PHASE, and that is the visible consequence of the parity being a
    passed word rather than a compiled constant: two folded axes with opposite
    declared phases share one compiled body here, where the REAL fold compiles two.
    """
    table: Dict[str, Dict[str, Any]] = {mode: {} for mode in contract_variants}
    for entry in entries:
        key = entry["key"]
        for mode in contract_variants:
            if key not in table[mode]:
                table[mode][key] = compile_folded_mirror_fill_complex(
                    entry["axis"], entry["pass"], entry["shifts"], mode)
    return table


def _keyed(entries: Sequence[Dict[str, Any]],
           targets: Sequence[Any]) -> Tuple[Dict[str, Any], ...]:
    """Attach the compile key and the bound target tensors to each entry."""
    out: List[Dict[str, Any]] = []
    for entry in entries:
        item = dict(entry)
        item["key"] = (f"axis{entry['axis']}/{entry['pass']}/"
                       f"{''.join(str(s) for s in entry['shifts'])}")
        item["targets"] = tuple(targets)
        out.append(item)
    return tuple(out)


def plan_folded_complex_fill(fields: Any, family: str, slot: str,
                             residency: Optional[Residency] = None,
                             contract_variants: Sequence[str] = (
                                 shaders.CONTRACT_OFF,)) -> Optional[FoldedComplexFillPlan]:
    """Build one family's folded complex ghost-fill plan, or None when refused."""
    if not folded_complex_fill_coverage(fields, family, residency).covered:
        return None
    grid = fields.grid
    names = tuple(GHOST_FILL_FAMILIES[family]["targets"])
    targets = [complex_fields._complex_mirror(residency, n, getattr(fields, n))
               for n in names]
    near = _keyed(fill_axis_entries(grid, family, "near"), targets)
    far = _keyed(fill_axis_entries(grid, family, "far"), targets)
    if not near:  # pragma: no cover - the predicate already refused
        return None
    return FoldedComplexFillPlan(
        family, slot, grid.shape, residency, near, far,
        _fill_functions(tuple(near) + tuple(far), contract_variants),
        names)


def plan_folded_complex_fill_from_arrays(
        family: str, slot: str, arrays: Dict[str, Any],
        near: Sequence[Dict[str, Any]], far: Sequence[Dict[str, Any]], residency: Residency,
        functions: Optional[Dict[str, Dict[str, Any]]] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        ) -> FoldedComplexFillPlan:
    """Build a folded complex fill plan from bare host arrays — a gate's route.

    ``near`` and ``far`` are the entry lists :func:`fill_axis_entries` builds; a gate
    supplies them directly so it can hand over a deliberately wrong reflect row, a
    deliberately wrong coefficient or a reversed axis order and watch each be caught.
    ``functions`` is the mutation seam.
    """
    names = tuple(GHOST_FILL_FAMILIES[family]["targets"])
    shape = tuple(int(n) for n in arrays[names[0]].shape)
    targets = [complex_fields._complex_mirror(residency, n, arrays[n]) for n in names]
    near = _keyed(near, targets)
    far = _keyed(far, targets)
    return FoldedComplexFillPlan(
        family, slot, shape, residency, near, far,
        functions if functions is not None
        else _fill_functions(tuple(near) + tuple(far), contract_variants),
        names)


def plan_folded_complex_constitutive(fields: Any, pml: Any, side: str,
                                     residency: Any = None,
                                     contract_variants: Sequence[str] = (
                                         shaders.CONTRACT_OFF,), *, chunk_bounds: Any) -> Any:
    """A CERTIFIED complex constitutive plan for a folded run, or None.

    The arithmetic, the source and the plan class are ``complex_fields``' own; only
    the ADMISSION is this family's. Building the plan through the certified pieces —
    rather than re-deriving one here — is what makes "same kernel" a fact instead of
    a claim: a divergence would have to come from the predicate, which is the only
    thing this family contributes on these two slots. The same move
    ``symmetry.plan_folded_constitutive`` makes for the real pair.
    """
    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, "
                         f"got {side!r}")
    if not folded_complex_constitutive_coverage(
            fields, pml, side, residency).covered:
        return None

    spec = CONSTITUTIVE_SIDES[side]
    suffix = "_h" if spec["half_integer"] else ""
    mirror = complex_fields._complex_mirror
    targets = [mirror(residency, n, getattr(fields, n)) for n in spec["targets"]]
    auxiliaries = [mirror(residency, n, getattr(fields, n)) for n in spec["aux"]]
    sources = [mirror(residency, n, getattr(fields, n)) for n in spec["sources"]]
    inverse_epsilon = (
        [residency.mirror("inv_eps_" + n, fields.inverse_epsilon_for(n),
                          constant=True) for n in spec["targets"]]
        if side == "E" else None)
    coefficients = [
        residency.mirror(f"pml:{stem}_{axis}{suffix}",
                         getattr(pml, f"{stem}_{axis}{suffix}"), constant=True)
        for axis in "xyz" for stem in ("kps", "kms")]
    volumes = tuple(spec["targets"]) + tuple(spec["aux"]) + tuple(spec["sources"])
    return complex_fields.ComplexConstitutivePlan(
        side, fields.grid.shape, residency, targets, auxiliaries, sources,
        inverse_epsilon, coefficients,
        complex_fields._constitutive_functions(side, contract_variants),
        volumes, chunk_bounds=chunk_bounds)


# ---------------------------------------------------------------------------
# WIRING — six arms over six slots
# ---------------------------------------------------------------------------
#
# WHAT SEPARATES THEM, per slot, and every one of these is a clause that names the
# other family rather than an omission:
#
#   step_B / step_D    `coverage._grid_reasons` clause 2 refuses complex storage
#                      (pml_curl); `bfast_curl` and `special_kz`'s real arm restate
#                      it; `symmetry` (the real fold) refuses complex storage BY NAME
#                      and names this family; `complex_fields` refuses the FOLD by
#                      name; `special_kz`'s complex arm refuses the fold by name.
#                      Clause 5 HERE is inverted through `_requires_a_fold` and
#                      clause 8 refuses beta by name.
#   update_H/update_E  the same separations, plus `no_pml_constitutive` inverting on
#                      the ABSORBER (it requires an INACTIVE one and this family an
#                      active one) and `offdiag_update_e` requiring a LIVE
#                      off-diagonal row under REAL storage, which this family refuses
#                      on the E side and cannot supply on either.
#   fill_B / fill_D    `symmetry`'s mirror fill is the only other arm on these slots
#                      and inverts on STORAGE, in both directions and by name. Both
#                      arms are deliberately UNGATED so an unfolded or wrong-storage
#                      run gets a named refusal rather than the composer's "nobody
#                      asked".
#
# REGISTERED-BUT-CANNOT-WIN IS NOT USED HERE, and the absence is worth stating.
# `registry.py` keeps `registered` and `wins` separate so a missing capability gets
# NAMED at composition time instead of falling silently to the array path, and this
# family would have used it if a slot's product were unported. Every slot it claims
# IS ported. The two capabilities it LACKS — a folded complex BETA curl and a folded
# complex OFF-DIAGONAL update_E — are refused by name inside predicates that would
# otherwise admit, which is the same naming through a cheaper mechanism: a
# registered-but-losing arm would report the same sentence from one slot further out.


def _has_complex_fold(context: Any) -> bool:
    """The cheap gate that decides whether this family is CONSULTED at all.

    A gated-out arm contributes NO reason; a consulted-and-refusing arm contributes
    all of its reasons by name. So the gate is deliberately the cheapest possible
    read of the grid and never the predicate itself — on an unfolded run this family
    should be silent on the four slots where OTHER arms will speak.

    THE FILL SLOTS ARE NOT GATED (see :func:`register_arms`), so this asks only about
    the fold and not about storage: an arm that is consulted must be free to say
    "storage is real float32", which is the sentence that makes the inversion visible.
    """
    grid = getattr(getattr(context, "fields", None), "grid", None)
    return grid is not None and _has_real_fold(grid)


def _curl_arm_coverage(context: Any, slot: str) -> Coverage:
    return folded_complex_composition_curl_coverage(
        context.fields, context.pml, slot, context.residency)


def _curl_arm_plan(context: Any, slot: str) -> Any:
    return plan_folded_complex_pml_curl(
        context.fields, context.pml, slot, context.residency,
        context.contract_variants,
        chunk_bounds=context.chunk_bounds)


def _constitutive_arm_coverage(context: Any, slot: str) -> Coverage:
    return folded_complex_constitutive_coverage(
        context.fields, context.pml, _CONSTITUTIVE_SLOT_SIDES[slot],
        context.residency)


def _constitutive_arm_plan(context: Any, slot: str) -> Any:
    return plan_folded_complex_constitutive(
        context.fields, context.pml, _CONSTITUTIVE_SLOT_SIDES[slot],
        context.residency, context.contract_variants, chunk_bounds=context.chunk_bounds)


def _fill_arm_coverage(context: Any, slot: str) -> Coverage:
    return folded_complex_fill_coverage(
        context.fields, _FILL_SLOT_FAMILIES[slot], context.residency)


def _fill_arm_plan(context: Any, slot: str) -> Optional[FoldedComplexFillPlan]:
    return plan_folded_complex_fill(
        context.fields, _FILL_SLOT_FAMILIES[slot], slot, context.residency,
        context.contract_variants)


def register_arms() -> Tuple[Any, ...]:
    """This family's six arms: two curls, the certified complex pair, two fills.

    THE FILL ARMS ARE UNGATED AND THE OTHER FOUR ARE NOT, for the reason
    :mod:`.symmetry` states and this family inherits: on ``step_B``/``step_D`` and
    ``update_H``/``update_E`` five other families speak, so a silent folded-complex
    arm on an unfolded run costs a reader nothing. On ``fill_B``/``fill_D`` there are
    exactly two arms — this one and the real fold's — and they invert on STORAGE, so
    both must be free to say which storage they need. A gated-out pair would leave a
    real-storage folded run reading "no consulted product admitted fill_B" with only
    one of the two reasons that actually apply.
    """
    from . import arms  # noqa: PLC0415 - deferred: `arms` imports nothing of ours

    registered = [
        arms.register(family=FAMILY, slot=slot, label="folded complex",
                      coverage=_curl_arm_coverage, plan=_curl_arm_plan,
                      prefix="folded complex: ", noun="folded complex PML curl",
                      gate=_has_complex_fold, wired=True)
        for slot in ("step_B", "step_D")]
    registered.extend(
        arms.register(family=FAMILY, slot=slot, label="folded complex",
                      coverage=_constitutive_arm_coverage,
                      plan=_constitutive_arm_plan,
                      prefix="folded complex: ",
                      noun="folded complex constitutive",
                      gate=_has_complex_fold, wired=True)
        for slot in _CONSTITUTIVE_SLOT_SIDES)
    registered.extend(
        arms.register(family=FAMILY, slot=slot, label="folded complex fill",
                      coverage=_fill_arm_coverage, plan=_fill_arm_plan,
                      prefix="folded complex fill: ",
                      noun="folded complex ghost fill",
                      gate=None, wired=True)
        for slot in _FILL_SLOT_FAMILIES)
    return tuple(registered)


#: Registered ON IMPORT, once — the registry refuses a duplicate by design.
ARMS: Tuple[Any, ...] = register_arms()
