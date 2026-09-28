"""Merge-bar tests for the Metal FOLDED COMPLEX family — a composition of two.

THE QUESTION THIS FILE ANSWERS IS NOT "IS EACH HALF RIGHT". Both halves are already
byte-certified alone: ``complex_fields`` (the complex/Bloch curl and constitutive)
and ``symmetry`` (the real fold). The question is whether they remain right
TOGETHER, and the answer is "in three of four places yes, and in the fourth NO" —
which is exactly why this is a family and not a flag.

**1. THE STRUCTURAL COMPOSITION.** The folded complex curl is
``complex_fields._CURL_TEMPLATE`` with ONE block inserted at a checked anchor, and
its ghost gather and cell-0 mask are the CERTIFIED emitters' own output reached by
mapping every mirror code to ``METALLIC``. Both halves of that are checked by
CHARACTER, in both directions, because ``torch.mps.compile_shader`` exposes no
disassembly and source text is the only place the claim can be held on this backend.
The strongest form is here: on an UNFOLDED grid the emitted source must be
character-identical to ``complex_fields.bloch_curl_source``'s except for an empty
top-plane block.

**2. THE ONE PLACE THE COMPOSITION CHANGES ARITHMETIC.** Under real storage the
fold's parity is a sign-bit operation and ``symmetry.mirror_ghost_fill`` spells it
``-x`` or a plain copy. Under complex64 the array path's ``phase * plane`` is a FULL
complex multiply by ``(+/-1.0, +0.0)`` with its zero cross terms, and that is NOT the
identity even at the EVEN mirror. Measured here, not asserted: the real fold's
spelling is applied to a live folded complex state and the differing words are
counted, at both parities. If that count were ever zero this whole family should be
deleted and ``symmetry.mirror_ghost_fill`` admitted on complex storage instead, so
the leg asserts it is NOT.

**3. THE SUBNORMAL REACH GOES BACKWARDS**, and that is the composition's doing
rather than either port's. ``symmetry.py`` records its FILL as band-safe because it
performs no arithmetic; making the parity a complex product makes it arithmetic, and
arithmetic flushes on this backend. Measured here per value class.

**4. THE FOLD'S OWN HAZARDS**, unchanged from the real family and re-run because a
leg carrying one storage proves nothing about the other: the two ownership masks are
INVISIBLE at whole-step granularity (the fill passes overwrite exactly the planes
they protect), so the mask leg is a SUB-STEP leg; the mirror plane is written by one
pass and READ by the next, so the whole-step leg compares per COMPLETE STEP and
reports the FIRST DIVERGENT STEP; and the two terminations are different kernels.

**5. THE DISJOINTNESS**, stated as inversions rather than inherited. This family has
FOUR neighbours to stay clear of — the plain real families, the folded REAL family,
the unfolded COMPLEX family and the beta family — and every one is pinned here.

Device-touching tests are skipped without MPS. EVERY behavioural leg asserts a
VACUITY FLOOR — words actually moved — because zero-init is a fixed point of half of
this family's work and a no-op agreeing with a no-op is trivially identical.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_PARITY = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                       "parity", "meep_gpu")
_PARITY = os.path.abspath(_PARITY)
if _PARITY not in sys.path:
    sys.path.insert(0, _PARITY)

import metal_composition_matrix as matrix  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.metal_kernels import (  # noqa: E402
    complex_fields, device, folded_complex, launch, shaders, symmetry, templates,
)
from meep_gpu.triton_kernels import symmetry as triton_symmetry  # noqa: E402

MODULE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "metal_kernels", "folded_complex.py")

ENVIRONMENT = matrix.prepare_environment()

#: The expansion arm every source leg builds with. Read from the SAME artifact the
#: predicates read, so a test cannot certify a body the family would not launch.
PROBE = folded_complex.load_expansion_probe()
EXPANSION = folded_complex.expansion_from_probe(PROBE) if PROBE else None

requires_probe = pytest.mark.skipif(
    EXPANSION is None,
    reason="no folded-complex expansion probe artifact on this host")

#: Both terminations of a fold on axis Y over a periodic x. ``(0, 3, 0)`` is
#: MIRROR_PERIODIC — the top plane is NOT owned, so the mask and the far fill both
#: fire; ``(0, 2, 0)`` is MIRROR_METALLIC, where neither does.
PERIODIC_FOLD = (folded_complex.CODE_PERIODIC, folded_complex.CODE_MIRROR_PERIODIC,
                 folded_complex.CODE_PERIODIC)
METALLIC_FOLD = (folded_complex.CODE_PERIODIC, folded_complex.CODE_MIRROR_METALLIC,
                 folded_complex.CODE_PERIODIC)
UNFOLDED = (folded_complex.CODE_PERIODIC, folded_complex.CODE_PERIODIC,
            folded_complex.CODE_PERIODIC)

ALL_CODES = (folded_complex.CODE_PERIODIC, folded_complex.CODE_METALLIC,
             folded_complex.CODE_MIRROR_METALLIC, folded_complex.CODE_MIRROR_PERIODIC)

NO_PHASE = (0, 0, 0)


def _mps_available() -> bool:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False
    return bool(getattr(getattr(torch.backends, "mps", None), "is_available",
                        lambda: False)())


requires_mps = pytest.mark.skipif(not _mps_available(),
                                  reason="no MPS device on this host")


def _words(array) -> np.ndarray:
    """uint32 WORD view of a complex64 (or float32) volume — never allclose."""
    flat = np.ascontiguousarray(array).reshape(-1)
    if flat.dtype == np.complex64:
        flat = flat.view(np.float32)
    return np.ascontiguousarray(flat, dtype=np.float32).view(np.uint32)


def _differing(a, b) -> int:
    return int(np.count_nonzero(_words(a) != _words(b)))


def _moved(before, after) -> int:
    return _differing(before, after)


def _subnormal_words(array) -> int:
    w = _words(array)
    return int(np.count_nonzero(((w >> 23) & 0xFF == 0) & ((w & 0x7FFFFFFF) != 0)))


def _source(codes=PERIODIC_FOLD, backward: bool = False, phased=NO_PHASE) -> str:
    return folded_complex.folded_bloch_curl_source(codes, backward, phased, EXPANSION)


# ---------------------------------------------------------------------------
# 1. The structural composition — the ONLY new text is the top-plane mask
# ---------------------------------------------------------------------------

def test_the_curl_template_is_the_certified_one_plus_one_block():
    """Derived, not copied, and the anchor is checked to be unique.

    The whole composition argument is that this family owns exactly one block of
    device text. Removing that block from the derived template must give back
    ``complex_fields._CURL_TEMPLATE`` CHARACTER FOR CHARACTER — which is a stronger
    statement than "it contains the parent's fragments", because it also forbids
    silently dropping one.
    """
    derived = folded_complex.folded_curl_template()
    assert derived != complex_fields._CURL_TEMPLATE
    restored = derived.replace(folded_complex._TOP_MASK_BLOCK, "", 1)
    assert restored == complex_fields._CURL_TEMPLATE
    assert complex_fields._CURL_TEMPLATE.count(folded_complex._MASK_ANCHOR) == 1


def test_a_parent_template_without_the_anchor_is_refused_loudly(monkeypatch):
    """A build failure, never a kernel with the mask in the wrong place.

    If the certified complex curl ever moves or duplicates its ownership-mask slot,
    inserting after "the anchor" would put the top-plane mask somewhere else — a
    plane of wrong values on every folded PERIODIC run, not a crash. So the count is
    asserted rather than assumed, and this leg proves the assertion can fire.
    """
    monkeypatch.setattr(complex_fields, "_CURL_TEMPLATE",
                        complex_fields._CURL_TEMPLATE.replace("__MASK__\n", "", 1))
    with pytest.raises(RuntimeError, match="ownership-mask anchor"):
        folded_complex.folded_curl_template()


@requires_probe
@pytest.mark.parametrize("backward", (False, True))
@pytest.mark.parametrize("codes", [(cx, cy, cz) for cx in ALL_CODES
                                   for cy in ALL_CODES for cz in ALL_CODES][::7])
def test_the_ghost_and_cell_zero_mask_are_the_certified_emitters_output(codes,
                                                                       backward):
    """Both fold deltas are a CALL, not a copy, and that is checkable by character.

    ``templates.ghost`` and ``templates.ownership_mask`` are the shipped emitters.
    Feeding them the reduced codes is the whole of deltas (1) and (2), so their
    output must appear verbatim in the folded source.
    """
    reduced = symmetry._reduced_codes(codes)
    source = _source(codes, backward)
    for axis, code in zip("xyz", reduced):
        assert templates.ghost(axis, code, backward) in source, (codes, axis)
    mask = templates.ownership_mask(reduced, backward, zero=templates.COMPLEX_ZERO)
    assert mask in source, codes


@requires_probe
@pytest.mark.parametrize("backward", (False, True))
@pytest.mark.parametrize("mirror", (folded_complex.CODE_MIRROR_METALLIC,
                                    folded_complex.CODE_MIRROR_PERIODIC))
def test_a_mirror_axis_gathers_exactly_as_a_metallic_one(mirror, backward):
    """Delta (1) stated as an equality the compiler cannot disagree with.

    The two MIRROR codes and METALLIC must emit the SAME ghost text; the only place
    the two mirror codes may differ from each other is the top-plane mask.
    """
    metallic = (folded_complex.CODE_PERIODIC, folded_complex.CODE_METALLIC,
                folded_complex.CODE_PERIODIC)
    folded = (folded_complex.CODE_PERIODIC, mirror, folded_complex.CODE_PERIODIC)
    a = _source(metallic, backward)
    b = _source(folded, backward)
    top = folded_complex.folded_top_plane_mask(folded, backward,
                                               zero=templates.COMPLEX_ZERO)
    metallic_top = folded_complex.folded_top_plane_mask(
        metallic, backward, zero=templates.COMPLEX_ZERO)
    assert b.replace(top, metallic_top) == a


@requires_probe
@pytest.mark.parametrize("backward", (False, True))
@pytest.mark.parametrize("phased", ((0, 0, 0), (1, 0, 0), (0, 0, 1)))
def test_an_unfolded_grid_emits_the_certified_complex_source(backward, phased):
    """THE REDUCTION, by character. The strongest form of the composition claim.

    With every axis PERIODIC/METALLIC the folded emitter must produce
    ``complex_fields.bloch_curl_source``'s output plus an EMPTY top-plane block —
    so removing that block gives back the certified source exactly. On the Triton
    track the analogous claim needs a PTX read; here it is text, and text is all
    this backend offers.
    """
    codes = (folded_complex.CODE_PERIODIC, folded_complex.CODE_METALLIC,
             folded_complex.CODE_PERIODIC)
    phased = tuple(f if codes[i] == folded_complex.CODE_PERIODIC else 0
                   for i, f in enumerate(phased))
    folded_source = _source(codes, backward, phased)
    certified = complex_fields.bloch_curl_source(
        symmetry._reduced_codes(codes), backward, phased, EXPANSION)
    empty = folded_complex._TOP_MASK_BLOCK.replace(
        "__TOP_MASK__",
        folded_complex.folded_top_plane_mask(codes, backward,
                                             zero=templates.COMPLEX_ZERO))
    assert folded_source.replace(empty, "", 1) == certified


@requires_probe
@pytest.mark.parametrize("backward", (False, True))
def test_the_top_plane_mask_fires_only_on_a_folded_periodic_axis(backward):
    """MIRROR_METALLIC's top plane is OWNED and STEPPED — masking it deletes a cell."""
    periodic = folded_complex.folded_top_plane_mask(PERIODIC_FOLD, backward,
                                                    zero=templates.COMPLEX_ZERO)
    metallic = folded_complex.folded_top_plane_mask(METALLIC_FOLD, backward,
                                                    zero=templates.COMPLEX_ZERO)
    unfolded = folded_complex.folded_top_plane_mask(UNFOLDED, backward,
                                                    zero=templates.COMPLEX_ZERO)
    assert "curl" in periodic
    assert "curl" not in metallic
    assert "curl" not in unfolded
    assert templates.COMPLEX_ZERO in periodic


@requires_probe
def test_the_folded_curl_binds_exactly_what_the_certified_complex_curl_binds():
    """The fold adds no argument: the STORED EXTENT is what carries it."""
    folded_source = _source()
    certified = complex_fields.bloch_curl_source(
        (templates.PERIODIC,) * 3, False, NO_PHASE, EXPANSION)
    assert folded_source.count("[[buffer(") == certified.count("[[buffer(")
    assert folded_source.count("[[buffer(") == complex_fields.CURL_BINDINGS


@requires_probe
@pytest.mark.parametrize("phased", ((0, 1, 0), (1, 1, 1)))
def test_a_phased_folded_axis_is_refused_at_emission(phased):
    """Clause 9, held at the LAST place it can be seen.

    A mis-baked phase flag on a folded axis is a plane of wrong values rather than a
    crash, and the predicate is not the only thing that can build a source — a gate
    hands codes and flags in directly. So the emitter refuses too.
    """
    with pytest.raises(ValueError, match="folded and carries a Bloch phase"):
        _source(PERIODIC_FOLD, False, phased)


# ---------------------------------------------------------------------------
# 2. The parity — the one place the composition changes arithmetic
# ---------------------------------------------------------------------------

def test_the_parity_coefficients_imaginary_word_is_bitwise_zero():
    """The property everything else in this family rests on.

    ``complex64(+/-1)`` must have an imaginary word of BITWISE ``0x00000000`` — not
    merely equal to zero — and a real word of exactly ``+/-1.0``, at both phases and
    for the near and far coefficient alike. Every claim about the two expansion arms
    agreeing on the parity product follows from it, and the probe's own
    ``AMBIGUOUS_BOTH`` verdict on the parity pattern is only meaningful while it
    holds.
    """
    for phase in (1, -1):
        near, far = folded_complex.mirror_parity_coefficients(phase)
        assert near == (float(phase), 0.0)
        assert far == (float(-phase), 0.0)
        for words in (near, far):
            assert _words(np.float32([words[1]]))[0] == 0x00000000
            assert abs(words[0]) == 1.0


def test_a_parity_outside_plus_or_minus_one_is_refused():
    with pytest.raises(ValueError, match=r"\+1 or -1"):
        folded_complex.mirror_parity_coefficients(0)


@requires_probe
@pytest.mark.parametrize("pass_name", folded_complex.FILL_PASSES)
@pytest.mark.parametrize("axis", (0, 1, 2))
@pytest.mark.parametrize("family", ("B", "D"))
def test_the_fill_multiplies_and_never_flips_a_sign(family, axis, pass_name):
    """THE REFUTED SPELLING IS ABSENT AND THE FULL PRODUCT IS PRESENT.

    Under real storage ``symmetry.mirror_ghost_fill`` writes ``-x`` or a plain copy,
    which is EXACT there. Here that spelling is WRONG — measured below on a live
    state — so no emitted folded complex fill may contain it, and every write must go
    through the full complex product with the coefficient on the LEFT.

    The shift triple is read from ``TARGET_IYEE`` rather than spelled, so this leg
    would fail if the emitter and the Yee table disagreed about one component.
    """
    targets = tuple(triton_symmetry.GHOST_FILL_FAMILIES[family]["targets"])
    shifts = tuple(triton_symmetry.TARGET_IYEE[n][axis] for n in targets)
    want = 0 if pass_name == "near" else 1
    if want not in shifts:
        pytest.skip("no component of this family has that shift on this axis")
    source = folded_complex.folded_mirror_fill_complex_source(
        axis, pass_name, shifts, EXPANSION)
    # THE STORES, and only the stores. The helper block this source carries QUOTES
    # the refuted spellings in its own comments — that is `templates.complex_helpers`
    # recording why they are refuted — so a substring search over the whole source
    # would pass on the comment and never look at the body.
    stores = [line.strip() for line in source.splitlines()
              if "] = " in line and line.strip().startswith(("f0[", "f1[", "f2["))]
    assert len(stores) == sum(1 for s in shifts if s == want), (stores, shifts)
    for slot, shift in enumerate(shifts):
        if shift != want:
            assert not any(statement.startswith(f"f{slot}[") for statement in stores)
            continue
        assert any(statement.startswith(f"f{slot}[")
                   and f"c_mul(c, f{slot}[" in statement for statement in stores), \
            (family, axis, pass_name, slot, stores)
    for statement in stores:
        assert "-f" not in statement, statement
        assert "0.0f - " not in statement, statement
        # The coefficient is BOUND, never synthesised: no literal +/-1 in a store.
        assert "1.0f" not in statement, statement


@requires_probe
def test_the_fill_binds_the_reflect_row_at_runtime_and_bakes_no_row_literal():
    """``stored - 2`` at an even full count and ``stored - 3`` at an odd one.

    Baking ``n - 2`` reflects about the window top instead of about the second
    mirror — a whole cell wrong on every odd-count run. A single 3-D grid can carry
    both terminations with different rows, so the row cannot even be a per-run
    scalar.
    """
    source = folded_complex.folded_mirror_fill_complex_source(
        1, "far", (1, 0, 1), EXPANSION)
    assert "constant int&       reflect_row" in source
    assert "reflect_row * stride" in source
    assert "last - 2" not in source and "nyi - 2" not in source


@requires_probe
def test_an_empty_fill_body_is_refused_rather_than_emitted():
    """A launch writing nothing is a no-op a before/after comparison calls a pass."""
    with pytest.raises(ValueError, match="would write nothing"):
        folded_complex.folded_mirror_fill_complex_source(
            0, "near", (1, 1, 1), EXPANSION)
    with pytest.raises(ValueError, match="would write nothing"):
        folded_complex.folded_mirror_fill_complex_source(
            0, "far", (0, 0, 0), EXPANSION)


def test_the_fill_compiles_one_body_per_axis_and_pass_not_per_phase():
    """The visible consequence of the parity having become arithmetic.

    The REAL fold compiles a separate body per PHASE, because on this backend a
    runtime weight flushes subnormals and the parity had to be a source constant.
    Here the parity is inside an arithmetic expression that flushes anyway, so it
    rides as a passed word and two opposite phases SHARE one compiled body. Pinned
    because it is the shape of the port that most invites a mistaken symmetry with
    its sibling.
    """
    entries = [{"axis": 1, "pass": "near", "shifts": (0, 1, 0), "phase": p}
               for p in (1, -1)]
    keys = {folded_complex._keyed([entry], ())[0]["key"] for entry in entries}
    assert len(keys) == 1, keys


# ---------------------------------------------------------------------------
# 3. Disjointness — FOUR neighbours, four inversions
# ---------------------------------------------------------------------------

def test_the_real_fold_refuses_complex_storage_and_this_family_requires_it():
    """CLAUSE 2, in both directions, on every slot both families claim."""
    fields, pml = matrix.folded(complex_storage=True)
    residency = device.Residency()
    for sub_step in ("step_B", "step_D"):
        real = symmetry.folded_composition_curl_coverage(fields, pml, sub_step,
                                                         residency)
        assert not real.covered
        assert any("force_complex_fields" in reason for reason in real.reasons)
        mine = folded_complex.folded_complex_composition_curl_coverage(
            fields, pml, sub_step, residency, PROBE)
        assert mine.covered, mine.reasons
    for side, slot in (("H", "update_H"), ("E", "update_E")):
        assert not symmetry.folded_constitutive_coverage(
            fields, pml, side, residency).covered
        assert folded_complex.folded_complex_constitutive_coverage(
            fields, pml, side, residency, PROBE).covered
    for family in ("B", "D"):
        real = symmetry.mirror_ghost_fill_coverage(fields, family, residency)
        assert not real.covered
        mine = folded_complex.folded_complex_fill_coverage(fields, family, residency,
                                                           PROBE)
        assert mine.covered, mine.reasons


def test_this_family_refuses_real_storage_and_names_the_real_fold():
    """The other direction of clause 2, with the message naming the owner."""
    fields, pml = matrix.folded(complex_storage=False)
    residency = device.Residency()
    verdict = folded_complex.folded_complex_composition_curl_coverage(
        fields, pml, "step_B", residency, PROBE)
    assert not verdict.covered
    assert any("folded real family" in reason for reason in verdict.reasons)
    fill = folded_complex.folded_complex_fill_coverage(fields, "B", residency, PROBE)
    assert not fill.covered
    assert any("symmetry.mirror_ghost_fill" in reason for reason in fill.reasons)
    assert symmetry.mirror_ghost_fill_coverage(fields, "B", residency).covered


def test_the_two_fill_arms_can_never_both_admit_because_a_volume_has_one_dtype():
    """THE GAP CLAUSE 2 DOES NOT CLOSE, CHECKED RATHER THAN ARGUED.

    The real fold's fill asks ``force_complex_fields`` and carries NO k clause; this
    family also admits ``has_bloch``. So on a ``has_bloch`` run with
    ``force_complex_fields`` False the two clause-2s do not separate the arms — the
    DTYPE does, because the real fill pins float32 and this one pins complex64, and a
    volume has one dtype. Asserted on both a complex-storage and a real-storage
    folded grid rather than left to the module docstring.
    """
    residency = device.Residency()
    for complex_storage in (True, False):
        fields, _pml = matrix.folded(complex_storage=complex_storage)
        for family in ("B", "D"):
            real = symmetry.mirror_ghost_fill_coverage(fields, family, residency)
            mine = folded_complex.folded_complex_fill_coverage(
                fields, family, residency, PROBE)
            assert not (real.covered and mine.covered), (complex_storage, family)
            assert real.covered or mine.covered, (complex_storage, family)


def test_the_unfolded_complex_family_refuses_the_fold_and_this_one_requires_it():
    """CLAUSE 5, in both directions."""
    residency = device.Residency()
    folded_pair = matrix.folded(complex_storage=True)
    unfolded_pair = matrix.flat(complex_storage=True)
    for sub_step in ("step_B", "step_D"):
        theirs = complex_fields.complex_pml_curl_coverage(
            *folded_pair, sub_step, residency, PROBE)
        assert not theirs.covered
        assert any("mirror plane is active" in reason for reason in theirs.reasons)
        mine = folded_complex.folded_complex_composition_curl_coverage(
            *unfolded_pair, sub_step, residency, PROBE)
        assert not mine.covered
        assert any("no mirror plane is active" in reason for reason in mine.reasons)


def test_the_wide_verdict_admits_an_unfolded_grid_and_the_routing_one_does_not():
    """The gate's verdict is NOT the composer's, for the composer-order reason.

    ``folded_complex_pml_curl_coverage`` admits an unfolded grid so a gate can prove
    the reduction to the certified complex kernel; registering THAT would make every
    unfolded complex row ambiguous.
    """
    fields, pml = matrix.flat(complex_storage=True)
    residency = device.Residency()
    wide = folded_complex.folded_complex_pml_curl_coverage(
        fields, pml, "step_B", residency, PROBE)
    routing = folded_complex.folded_complex_composition_curl_coverage(
        fields, pml, "step_B", residency, PROBE)
    assert wide.covered, wide.reasons
    assert not routing.covered
    assert set(routing.reasons) - set(wide.reasons)


def test_beta_is_refused_by_name_on_the_arithmetic_slots_and_not_on_the_fill():
    """CLAUSE 8, and the SCOPE of the refusal, which is the interesting half.

    A folded complex beta run belongs to ANOTHER family — :mod:`.folded_beta`, which
    now carries it — so every arithmetic slot must be refused BY NAME AND THE NAME
    MUST BE THE OWNER'S. The two FILL slots must NOT be refused:
    ``fill_symmetry_bc_*`` copies ``phase * field[2]`` and touches no beta term at
    all, so a beta clause there would refuse a pass beta cannot reach — which is
    also why ``folded_beta`` registers no fill arm and these two slots stay this
    family's.

    THE PIN NAMES THE OWNER RATHER THAN THE ABSENCE. This clause was written as a
    PRE-REGISTERED inversion while nothing carried those slots; when ``folded_beta``
    honoured it, "the folded beta product, which this backend does not carry"
    stopped being true while the refusal itself stayed exactly right. A reader
    chasing this message must land on the module that owns the slots.
    """
    fields, pml = matrix.folded(complex_storage=True, beta=0.3)
    residency = device.Residency()
    for sub_step in ("step_B", "step_D"):
        verdict = folded_complex.folded_complex_composition_curl_coverage(
            fields, pml, sub_step, residency, PROBE)
        assert not verdict.covered
        assert any("beta" in reason and "folded_beta" in reason
                   for reason in verdict.reasons), verdict.reasons
    for side in ("H", "E"):
        assert not folded_complex.folded_complex_constitutive_coverage(
            fields, pml, side, residency, PROBE).covered
    for family in ("B", "D"):
        assert folded_complex.folded_complex_fill_coverage(
            fields, family, residency, PROBE).covered


def test_the_offdiagonal_row_is_refused_on_the_e_side_only():
    """Constitutive-only: the row product's whole effect is inside ``update_E``."""
    fields, pml = matrix.folded(complex_storage=True, rows={"Ey": ["Ez"]})
    residency = device.Residency()
    assert folded_complex.folded_complex_composition_curl_coverage(
        fields, pml, "step_B", residency, PROBE).covered
    assert folded_complex.folded_complex_constitutive_coverage(
        fields, pml, "H", residency, PROBE).covered
    verdict = folded_complex.folded_complex_constitutive_coverage(
        fields, pml, "E", residency, PROBE)
    assert not verdict.covered
    assert any("off-diagonal chi1inv row" in reason for reason in verdict.reasons)


def test_a_bloch_phase_on_the_folded_axis_is_refused_by_name():
    """CLAUSE 9 — this family's own, with no twin in either parent.

    MEASURED WHILE WRITING THIS LEG, and it changes what the clause is FOR: the
    ENGINE ITSELF refuses to construct the configuration. ``Grid`` raises "mirror
    symmetry on the Y axis requires k_point component y = 0" before ``Fields``
    exists, so clause 9 is UNREACHABLE through the engine's own classes, exactly as
    ``driver._require_bloch_is_representable`` is a second refusal one level up.

    That does not make the clause dead weight and it does not make it a comment. A
    predicate's contract here is over DUCK-TYPED objects — a gate, a probe or a
    harness hands one in — and admitting a run whose kernel would bake a phase flag
    onto a mirror axis is a plane of wrong values rather than a crash. So the leg
    asserts BOTH: the engine refuses to build it, and the predicate refuses one built
    anyway. A Bloch phase on ANOTHER axis composes and is admitted, and that half
    goes through the engine.
    """
    residency = device.Residency()
    with pytest.raises(ValueError, match="requires k_point component y = 0"):
        matrix.folded(axis="Y", complex_storage=True, k_point=(0.0, 0.31, 0.0))

    off_axis, pml = matrix.folded(axis="Y", complex_storage=True,
                                  k_point=(0.27, 0.0, 0.0))
    assert folded_complex.folded_complex_composition_curl_coverage(
        off_axis, pml, "step_B", residency, PROBE).covered

    class _PhasedFoldGrid:
        """The engine's grid with a phase forced onto the folded axis."""

        def __init__(self, grid):
            self._grid = grid

        def __getattr__(self, name):
            return getattr(self._grid, name)

        @property
        def k_point(self):
            return (0.27, 0.31, 0.0)

        def bloch_phase(self, axis):
            return complex(0.4, 0.9) if axis in (0, 1) else None

    class _PhasedFoldFields:
        def __init__(self, fields, grid):
            self._fields = fields
            self.grid = grid

        def __getattr__(self, name):
            return getattr(self._fields, name)

    planted = _PhasedFoldFields(off_axis, _PhasedFoldGrid(off_axis.grid))
    verdict = folded_complex.folded_complex_composition_curl_coverage(
        planted, pml, "step_B", residency, PROBE)
    assert not verdict.covered
    assert any("folded and carries Bloch phase" in reason
               for reason in verdict.reasons), verdict.reasons
    assert any("folded with k component" in reason
               for reason in verdict.reasons), verdict.reasons


def test_a_missing_probe_is_a_named_refusal_and_never_a_default_arm():
    """Which arm the reference takes is a MEASURED platform fact."""
    fields, pml = matrix.folded(complex_storage=True)
    residency = device.Residency()
    empty = {"backend": "numpy", "patterns": {}}
    verdict = folded_complex.folded_complex_composition_curl_coverage(
        fields, pml, "step_B", residency, empty)
    assert not verdict.covered
    assert any(folded_complex.PARITY_PROBE_PATTERN in reason
               for reason in verdict.reasons)
    assert folded_complex.expansion_from_probe(empty) is None
    assert folded_complex.plan_folded_complex_pml_curl(
        fields, pml, "step_B", residency, probe=empty) is None


def test_a_probe_missing_only_the_parity_pattern_licenses_nothing():
    """An artifact cut before this tranche classified four orientations, not five."""
    inherited = {"backend": "numpy",
                 "patterns": {name: "FMA_V1"
                              for name in complex_fields.PROBE_PATTERNS}}
    assert complex_fields.expansion_from_probe(inherited) == "FMA_V1"
    assert folded_complex.expansion_from_probe(inherited) is None


def test_exactly_one_arm_admits_each_slot_on_a_folded_complex_grid():
    """The composer's own safety property, on this family's own configuration."""
    from meep_gpu.metal_kernels import arms

    fields, pml = matrix.folded(complex_storage=True)
    context = arms.StepContext(fields, pml, device.Residency(),
                               (shaders.CONTRACT_OFF,), sources=(),
                               extra={"complex_probe": None, "beta_probe": None,
                                      "folded_complex_probe": PROBE})
    for slot in ("step_B", "step_D", "update_H", "update_E", "fill_B", "fill_D"):
        admitted = []
        for spec in arms.registered(slot):
            if not spec.wired:
                continue
            if spec.gate is not None and not spec.gate(context):
                continue
            try:
                verdict = spec.coverage(context, slot)
            except Exception:  # noqa: BLE001 - a raising predicate is a refusal
                continue
            if verdict.covered:
                admitted.append(spec.family)
            else:
                assert verdict.reasons, (slot, spec.family)
        assert admitted == [folded_complex.FAMILY], (slot, admitted)


def test_the_family_is_in_the_registry_module_list():
    """A family absent from ``FAMILY_MODULES`` is INVISIBLE to ``plan_step``."""
    from meep_gpu.metal_kernels import registry

    assert "folded_complex" in registry.FAMILY_MODULES


def test_no_module_level_torch():
    """The predicates must answer on a host with no GPU — that is the merge bar."""
    with open(MODULE, "r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            stripped = line.strip()
            if stripped.startswith("import torch") or stripped.startswith(
                    "from torch"):
                assert line.startswith(" "), (number, stripped)


# ---------------------------------------------------------------------------
# 4. Behavioural — the composition's own hazards, on the device
# ---------------------------------------------------------------------------

SNAPSHOT = ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
            "Hx", "Hy", "Hz", "fu_Bx", "fu_By", "fu_Bz",
            "fu_Dx", "fu_Dy", "fu_Dz", "f_w_Hx", "f_w_Hy", "f_w_Hz",
            "f_w_Ex", "f_w_Ey", "f_w_Ez")


def _snapshot(fields):
    return {name: np.array(getattr(fields, name), copy=True)
            for name in SNAPSHOT if getattr(fields, name, None) is not None}


def _restore(fields, state):
    for name, array in state.items():
        getattr(fields, name)[...] = array


def _complex_folded(**kwargs):
    kwargs.setdefault("complex_storage", True)
    return matrix.folded(**kwargs)


#: THE AXES A BEHAVIOURAL LEG MUST CARRY. The real fold's eight, re-run rather than
#: inherited — a leg carrying one storage proves nothing about the other — PLUS the
#: two this composition adds and neither parent has.
#:
#: THE BLOCH ROWS ARE THE COMPOSITION'S OWN CASE and without them the byte legs would
#: never launch a folded curl that also ROTATES. Three of the five non-beta folded
#: complex corpus rows carry a Bloch k on an UNFOLDED axis (``holey-wvg-bands``,
#: ``triangular_lattice_oblique``, ``binary_grating_special_kz_2_21_2``), and the
#: phase block is emitted per axis right beside the fold's ghost gather, so a
#: matrix without them would certify the fold and the rotation only separately.
#: The phase is on x while the fold is on y, which is the only pairing the engine
#: will build: ``Grid`` refuses a k component on a mirror plane's own axis outright.
FOLD_CASES = (
    ("periodic_even", dict()),
    ("periodic_odd", dict(phase=-1)),
    ("metallic_even", dict(boundaries={"y": "metallic"})),
    ("metallic_odd", dict(phase=-1, boundaries={"y": "metallic"})),
    ("x_fold", dict(axis="X")),
    ("odd_full_count", dict(extent=2.1)),
    ("two_axis_mixed_phase", dict(axis="XY", phase=(1, -1))),
    ("two_axis_mixed_phase_odd", dict(axis="XY", phase=(-1, 1), extent=2.1)),
    ("bloch_off_axis", dict(k_point=(0.3, 0.0, 0.0))),
    ("bloch_off_axis_metallic_fold",
     dict(k_point=(0.3, 0.0, 0.0), boundaries={"y": "metallic"})),
)


@requires_mps
@requires_probe
def test_every_shipped_specialisation_compiles_in_both_contraction_modes():
    """A specialisation that fails to COMPILE is a crash at plan time.

    The curl's static product is 2 directions x 4^3 code quadruples, restricted to
    the phase flags each quadruple can LEGALLY carry — a metallic or folded axis
    cannot phase — plus the fill's 2 families x 3 axes x 2 passes. The corpus drives
    a fraction of that; the whole reachable product is compiled here.
    """
    from meep_gpu.metal_kernels.device import compile_source

    counts = []
    for mode in shaders.CONTRACT_MODES:
        sources = folded_complex.enumerate_folded_complex_sources(EXPANSION, mode)
        counts.append(len(sources))
        for source in sources.values():
            compile_source(source)
    assert counts[0] == counts[1] > 0
    # 2 directions x (the reachable phase flags per code quadruple) + 2 families x
    # 3 axes x 2 passes. MEASURED 2026-08-16 at 262 per contraction mode, 524 total.
    assert counts[0] == 262, counts


@requires_mps
@requires_probe
@pytest.mark.parametrize("sub_step", ("step_B", "step_D"))
@pytest.mark.parametrize("label,kwargs", FOLD_CASES, ids=[c[0] for c in FOLD_CASES])
def test_the_folded_complex_curl_reproduces_stepping_bit_for_bit(label, kwargs,
                                                                 sub_step):
    """SUB-STEP granularity, which is where the two ownership masks live.

    THE MASKS ARE INVISIBLE AT WHOLE-STEP GRANULARITY — the driver's fill passes
    overwrite exactly the planes they protect — so a whole-step check certifies a
    mask-less kernel as correct. This leg is the one that can fail on a mask, and it
    runs BOTH terminations because the top-plane mask exists only on MIRROR_PERIODIC.
    """
    fields, pml = _complex_folded(**kwargs)
    spec = launch.SUB_STEPS[sub_step]
    before = _snapshot(fields)
    getattr(stepping, sub_step)(fields, pml)
    after = _snapshot(fields)

    _restore(fields, before)
    residency = device.Residency()
    plan = folded_complex.plan_folded_complex_pml_curl(
        fields, pml, sub_step, residency, probe=PROBE)
    assert plan is not None, folded_complex.folded_complex_composition_curl_coverage(
        fields, pml, sub_step, residency, PROBE).reasons
    residency.sync_in()
    plan.run()
    residency.sync_out()
    assert plan.launches == 1

    compared = tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
    moved = sum(_moved(before[n], after[n]) for n in compared)
    assert moved > 100, (label, sub_step, moved,
                         "VACUOUS: the reference barely moved, so identity here "
                         "would be a no-op agreeing with a no-op")
    for name in compared:
        assert _differing(getattr(fields, name), after[name]) == 0, (label, name)


@requires_mps
@requires_probe
@pytest.mark.parametrize("label,kwargs", FOLD_CASES, ids=[c[0] for c in FOLD_CASES])
def test_the_folded_complex_fill_reproduces_both_array_path_passes(label, kwargs):
    """THE MIRROR PLANE ITSELF — near and far, in the driver's own X, Y, Z order."""
    fields, pml = _complex_folded(**kwargs)
    for family, near_fill, far_fill in (
            ("B", stepping.fill_symmetry_bc_B, stepping.fill_folded_far_ghosts_B),
            ("D", stepping.fill_symmetry_bc_D, stepping.fill_folded_far_ghosts_D)):
        names = tuple(triton_symmetry.GHOST_FILL_FAMILIES[family]["targets"])
        before = {n: np.array(getattr(fields, n), copy=True) for n in names}
        near_fill(fields)
        far_fill(fields)
        after = {n: np.array(getattr(fields, n), copy=True) for n in names}

        for name in names:
            getattr(fields, name)[...] = before[name]
        residency = device.Residency()
        plan = folded_complex.plan_folded_complex_fill(
            fields, family, "fill_" + family, residency, probe=PROBE)
        assert plan is not None, folded_complex.folded_complex_fill_coverage(
            fields, family, residency, PROBE).reasons
        residency.sync_in()
        plan.run_near()
        plan.run_far()
        residency.sync_out()
        assert plan.launches == len(plan.near) + len(plan.far) > 0

        moved = sum(_moved(before[n], after[n]) for n in names)
        assert moved > 0, (label, family, "VACUOUS: the fill wrote nothing")
        for name in names:
            assert _differing(getattr(fields, name), after[name]) == 0, \
                (label, family, name)


@requires_mps
@requires_probe
@pytest.mark.parametrize("label,kwargs", FOLD_CASES, ids=[c[0] for c in FOLD_CASES])
def test_a_complete_step_agrees_and_the_first_divergent_step_is_reported(label,
                                                                        kwargs):
    """THE WHOLE-STEP LEG — the only one that can see the fold's fourth risk.

    Three failure classes live ONLY in a complete step: a STALE MIRROR, a SEAM, and
    an ACCUMULATING AUXILIARY (``fu_*`` and ``f_w_*`` are STATE, and a kernel right
    for one launch and wrong forever after is identical in a single-launch leg). THE
    FOLD ADDS A FOURTH: the folded axis's ghost plane is written by one pass and READ
    by the next.

    Every slot's launch counter is asserted, because a slot that passes by NOT
    EXECUTING is the hollow pass this discipline exists to make impossible.
    """
    budget = 6
    fields, pml = _complex_folded(**kwargs)
    reference_fields, reference_pml = _complex_folded(**kwargs)

    residency = device.Residency()
    plans = {
        "step_B": folded_complex.plan_folded_complex_pml_curl(
            fields, pml, "step_B", residency, probe=PROBE),
        "fill_B": folded_complex.plan_folded_complex_fill(
            fields, "B", "fill_B", residency, probe=PROBE),
        "update_H": folded_complex.plan_folded_complex_constitutive(
            fields, pml, "H", residency, probe=PROBE),
        "step_D": folded_complex.plan_folded_complex_pml_curl(
            fields, pml, "step_D", residency, probe=PROBE),
        "fill_D": folded_complex.plan_folded_complex_fill(
            fields, "D", "fill_D", residency, probe=PROBE),
        "update_E": folded_complex.plan_folded_complex_constitutive(
            fields, pml, "E", residency, probe=PROBE),
    }
    assert all(plan is not None for plan in plans.values()), {
        name: plan for name, plan in plans.items() if plan is None}
    residency.sync_in()

    first_divergent = None
    per_step = []
    for step in range(budget):
        # The reference: the driver's own order, five passes per half.
        stepping.step_B(reference_fields, reference_pml)
        stepping.fill_symmetry_bc_B(reference_fields)
        stepping.fill_folded_far_ghosts_B(reference_fields)
        stepping.update_H(reference_fields, reference_pml)
        stepping.step_D(reference_fields, reference_pml)
        stepping.fill_symmetry_bc_D(reference_fields)
        stepping.fill_folded_far_ghosts_D(reference_fields)
        stepping.update_E(reference_fields, reference_pml)

        plans["step_B"].run()
        plans["fill_B"].run_near()
        plans["fill_B"].run_far()
        plans["update_H"].run()
        plans["step_D"].run()
        plans["fill_D"].run_near()
        plans["fill_D"].run_far()
        plans["update_E"].run()
        residency.sync_out()

        differing = {name: _differing(getattr(fields, name),
                                      getattr(reference_fields, name))
                     for name in SNAPSHOT
                     if getattr(fields, name, None) is not None}
        total = sum(differing.values())
        per_step.append(total)
        if total and first_divergent is None:
            first_divergent = (step, {k: v for k, v in differing.items() if v})

    assert first_divergent is None, (label, "FIRST DIVERGENT STEP", first_divergent,
                                     "per-step differing words", per_step)
    moved = sum(_moved(np.zeros_like(getattr(reference_fields, name)),
                       getattr(reference_fields, name))
                for name in ("Bx", "By", "Bz", "Ex", "Ey", "Ez"))
    assert moved > 100, (label, moved, "VACUOUS whole-step leg")
    # EVERY SLOT, not a sample. A slot that passes by NOT EXECUTING is the hollow
    # pass this discipline exists to make impossible, and checking three of six
    # leaves three able to do exactly that.
    for slot in ("step_B", "step_D", "update_H", "update_E"):
        assert plans[slot].launches == budget, (label, slot, plans[slot].launches)
    for slot in ("fill_B", "fill_D"):
        plan = plans[slot]
        expected = budget * (len(plan.near) + len(plan.far))
        assert plan.launches == expected, (label, slot, plan.launches, expected)
        assert len(plan.near) > 0, (label, slot, "no near pass: the fill was a no-op")


# ---------------------------------------------------------------------------
# 5. Mutations — a leg that cannot fail certifies nothing
# ---------------------------------------------------------------------------

@requires_mps
@requires_probe
@pytest.mark.parametrize("phase", (1, -1))
def test_the_real_folds_sign_flip_spelling_is_measurably_wrong_here(phase):
    """THE WHOLE REASON THIS FAMILY EXISTS, measured on a LIVE folded complex state.

    ``symmetry.mirror_ghost_fill``'s spelling — ``-x`` for the odd plane, a plain
    copy for the even one — is EXACT under real storage. Applied to the same fill
    under complex64 it must DIFFER from the array path, at BOTH parities including
    the EVEN mirror, where a plain copy is the identity and ``phase * plane`` is not.

    IF THIS EVER MEASURED ZERO, this family should be deleted and
    ``symmetry.mirror_ghost_fill`` admitted on complex storage instead. So it asserts
    the count is NOT zero rather than merely reporting it.
    """
    fields, _pml = _complex_folded(phase=phase)
    names = tuple(triton_symmetry.GHOST_FILL_FAMILIES["B"]["targets"])
    # PLANT ALL FOUR SIGNED-ZERO SIGN COMBINATIONS, and that is not thoroughness for
    # its own sake — it is what makes the leg reach both parities. Measured while
    # writing it: `(-0, -0)` and `(-0, 1.5)` discriminate at phase +1 and NOT at
    # phase -1, because which zero the cross term contributes depends on the
    # coefficient's sign. A plant chosen for one parity would have reported the odd
    # mirror as a place where the sign-bit spelling is correct, which it is not.
    plants = np.array([complex(0.0, 0.0), complex(-0.0, -0.0),
                       complex(0.0, -0.0), complex(-0.0, 0.0),
                       complex(1.5, -0.0), complex(-0.0, 1.5)],
                      dtype=np.complex64)
    for name in names:
        volume = getattr(fields, name)
        pattern = np.resize(plants, volume.shape[0] * volume.shape[2]).reshape(
            volume.shape[0], volume.shape[2])
        for row in range(volume.shape[1]):
            volume[:, row, :] = np.roll(pattern, row, axis=0)
    before = {n: np.array(getattr(fields, n), copy=True) for n in names}

    stepping.fill_symmetry_bc_B(fields)
    stepping.fill_folded_far_ghosts_B(fields)
    reference = {n: np.array(getattr(fields, n), copy=True) for n in names}

    # The REAL fold's spelling, applied by word surgery to the same planes.
    candidate = {n: np.array(before[n], copy=True) for n in names}
    codes, _ = folded_complex.folded_axis_kinds(fields.grid, None)
    rows = folded_complex._far_reflect_rows(fields.grid) or (None, None, None)
    for axis, code in enumerate(codes):
        if int(code) not in folded_complex.MIRROR_CODES:
            continue
        declared = int(fields.grid.mirror_phase(axis))
        for slot, name in enumerate(names):
            shift = triton_symmetry.TARGET_IYEE[name][axis]
            volume = candidate[name]
            index = [slice(None)] * 3
            if shift == 0:
                source = list(index)
                source[axis] = folded_complex.MIRROR_SOURCE_INDEX
                index[axis] = 0
                volume[tuple(index)] = _sign_bit(volume[tuple(source)], declared)
            elif int(code) == folded_complex.CODE_MIRROR_PERIODIC:
                source = list(index)
                source[axis] = int(rows[axis])
                index[axis] = -1
                volume[tuple(index)] = _sign_bit(volume[tuple(source)], -declared)

    differing = sum(_differing(candidate[n], reference[n]) for n in names)
    moved = sum(_moved(before[n], reference[n]) for n in names)
    assert moved > 0, "VACUOUS: the array path's fill wrote nothing"
    assert differing > 0, (
        phase, "the real fold's sign-bit spelling reproduced the array path on every "
        "word: this family's whole reason to exist is unmeasured")


def _sign_bit(plane, parity: int):
    """``-x`` (parity -1) or a plain copy (+1), by WORD, preserving every bit."""
    words = np.ascontiguousarray(plane, dtype=np.complex64).view(np.uint32).copy()
    if int(parity) == -1:
        words = words ^ np.uint32(0x80000000)
    return words.view(np.float32).view(np.complex64).reshape(plane.shape)


@requires_mps
@requires_probe
def test_the_parity_multiply_is_not_band_safe_and_the_real_folds_was():
    """THE REACH REGRESSION THE COMPOSITION CAUSES, per value class.

    ``symmetry.py`` records its fill as band-safe BECAUSE it performs no arithmetic.
    Making the parity a complex product makes it arithmetic, and arithmetic flushes
    on this backend. Both halves are measured here so the family's precondition
    sentence is a measurement rather than an inheritance.
    """
    import torch

    from meep_gpu.metal_kernels.device import compile_source

    source = folded_complex.folded_mirror_fill_complex_source(
        1, "near", (0, 1, 0), EXPANSION)
    module = compile_source(source)
    rng = np.random.default_rng(20260816)
    outcome = {}
    for scale in (1e-30, 1e-38, 1e-40):
        shape = (4, 6, 5)
        plane = ((rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
                 * scale).astype(np.complex64)
        volumes = [np.ascontiguousarray(plane.copy()) for _ in range(3)]
        reference = [v.copy() for v in volumes]
        for index, target in enumerate((0, 2)):  # shifts (0, 1, 0): slots 0 and 2
            del index
            reference[target][:, 0, :] = np.complex64(1) * reference[target][:, 2, :]
        tensors = [torch.from_numpy(v.view(np.float32).reshape(-1, 2).copy()).to("mps")
                   for v in volumes]
        module.folded_mirror_fill_complex(
            *tensors, shape[0], shape[1], shape[2], shape[0] * shape[2], -1,
            (1.0, 0.0))
        torch.mps.synchronize()
        got = [t.cpu().numpy().reshape(-1).view(np.complex64).reshape(shape)
               for t in tensors]
        band = sum(_subnormal_words(v[:, 2, :]) for v in volumes)
        differing = sum(_differing(got[i][:, 0, :], reference[i][:, 0, :])
                        for i in (0, 2))
        outcome[scale] = (band, differing)
    assert outcome[1e-30][1] == 0, outcome
    assert outcome[1e-38][0] > 0 and outcome[1e-38][1] > 0, outcome
    assert outcome[1e-40][0] > 0 and outcome[1e-40][1] > 0, outcome


@requires_mps
@requires_probe
@pytest.mark.parametrize("kwargs,caught", ((dict(), True),
                                           (dict(boundaries={"y": "metallic"}),
                                            False)))
def test_the_top_plane_mask_is_load_bearing_on_mirror_periodic_only(kwargs, caught):
    """A GATE THAT CANNOT FAIL CERTIFIES NOTHING, and the conditionality is the point.

    Dropping the top-plane mask must change bytes on a MIRROR_PERIODIC run and must
    NOT on a MIRROR_METALLIC one, where the emitter produces no mask lines at all.
    Both halves are asserted so the leg discriminates rather than merely fires.
    """
    from meep_gpu.metal_kernels.device import compile_source

    fields, pml = _complex_folded(**kwargs)
    codes, _ = folded_complex.folded_axis_kinds(fields.grid, pml)
    mask = folded_complex.folded_top_plane_mask(codes, backward=True,
                                                zero=templates.COMPLEX_ZERO)
    assert ("curl" in mask) is caught, (kwargs, mask)

    before = _snapshot(fields)
    stepping.step_D(fields, pml)
    after = _snapshot(fields)
    _restore(fields, before)

    source = folded_complex.folded_bloch_curl_source(codes, True, NO_PHASE, EXPANSION)
    mutant_source = source.replace(mask, "    // MUTANT: top-plane mask dropped")
    # On MIRROR_METALLIC the "mask" is already a comment, so replacing it changes the
    # SOURCE and cannot change a BYTE. The behavioural half below is the claim; this
    # only records that a mutant was actually built in both arms.
    assert mutant_source != source

    residency = device.Residency()
    plan = folded_complex.plan_folded_complex_pml_curl(
        fields, pml, "step_D", residency, probe=PROBE)
    assert plan is not None
    mutant = {shaders.CONTRACT_OFF:
              compile_source(mutant_source).bloch_pml_curl_step}
    hurt = folded_complex.plan_folded_complex_pml_curl_from_arrays(
        "step_D", {n: getattr(fields, n) for n in SNAPSHOT
                   if getattr(fields, n, None) is not None},
        {f"{stem}_{axis}": getattr(pml, f"{stem}_{axis}")
         for axis in "xyz" for stem in ("kms", "sinv")},
        codes, (None, None, None), fields.grid.dt / fields.grid.dx, EXPANSION,
        residency, functions=mutant)
    residency.sync_in()
    hurt.run()
    residency.sync_out()
    names = tuple(launch.SUB_STEPS["step_D"]["targets"])
    differing = sum(_differing(getattr(fields, n), after[n]) for n in names)
    assert (differing > 0) is caught, (kwargs, differing)


@requires_mps
@requires_probe
@pytest.mark.parametrize("extent,caught", ((2.0, False), (2.1, True)))
def test_the_reflect_row_mutation_is_caught_at_an_odd_full_count_only(extent, caught):
    """``stored - 2`` at an even full count and ``stored - 3`` at an odd one.

    At the default extent the wrong ``n - 2`` formula HAPPENS TO BE RIGHT, so a
    matrix carrying only that extent measures nothing about the reflect row. Both
    halves are asserted.
    """
    fields, _pml = _complex_folded(extent=extent)
    names = tuple(triton_symmetry.GHOST_FILL_FAMILIES["B"]["targets"])
    before = {n: np.array(getattr(fields, n), copy=True) for n in names}
    stepping.fill_symmetry_bc_B(fields)
    stepping.fill_folded_far_ghosts_B(fields)
    after = {n: np.array(getattr(fields, n), copy=True) for n in names}
    for name in names:
        getattr(fields, name)[...] = before[name]

    near = list(folded_complex.fill_axis_entries(fields.grid, "B", "near"))
    far = [dict(entry) for entry in
           folded_complex.fill_axis_entries(fields.grid, "B", "far")]
    assert far, "this configuration has no far pass at all"
    for entry in far:
        entry["reflect_row"] = int(fields.grid.shape[entry["axis"]]) - 2

    residency = device.Residency()
    plan = folded_complex.plan_folded_complex_fill_from_arrays(
        "B", "fill_B", {n: getattr(fields, n) for n in names}, near, far,
        EXPANSION, residency)
    residency.sync_in()
    plan.run_near()
    plan.run_far()
    residency.sync_out()
    differing = sum(_differing(getattr(fields, n), after[n]) for n in names)
    assert (differing > 0) is caught, (extent, differing)


def _fill_both_ways(fields, family, near, far, walk):
    """Run one family's fill through ``walk`` and return the resulting volumes."""
    names = tuple(triton_symmetry.GHOST_FILL_FAMILIES[family]["targets"])
    residency = device.Residency()
    plan = folded_complex.plan_folded_complex_fill_from_arrays(
        family, "fill_" + family, {n: getattr(fields, n) for n in names},
        near, far, EXPANSION, residency)
    residency.sync_in()
    walk(plan)
    residency.sync_out()
    return {n: np.array(getattr(fields, n), copy=True) for n in names}


@requires_mps
@requires_probe
def test_the_fill_axis_order_is_load_bearing_at_mixed_phase():
    """THE MEASURED NULL THAT DOES NOT TRANSFER, re-measured in both directions.

    ``symmetry.py`` records ``reverse_axis_order`` as a null under REAL storage on
    the argument that every fill there is a multiply by exactly +/-1. Complex
    multiplication is not associative, so a corner unowned on two folded axes sees
    ``c_y (x) (c_x (x) z)`` against ``c_x (x) (c_y (x) z)`` and the argument does not
    transfer.

    MEASURED HERE 2026-08-16, on the matrix's own two-axis folded complex grids,
    BOTH families because the two write different corners:

        MIXED phase (+1, -1)   B  0 words     D  1 word (Dz)
        MATCHED phase (+1, +1) B  0 words     D  0 words

    ``Dz`` has Yee shift 0 on BOTH folded axes, so the NEAR pass writes its
    doubly-unowned corner twice and the two orders compose the two coefficients the
    other way round. With ``c_x == c_y`` the composition commutes bit-exactly, which
    is why a matched-phase grid makes the question vanish rather than answer it.

    Both directions are asserted: matched must be 0 (or the leg is not measuring
    order at all) and mixed must NOT be, or the X, Y, Z ordering this plan preserves
    is decorative.
    """
    outcome = {}
    for label, kwargs in (("mixed", dict(axis="XY", phase=(1, -1))),
                          ("matched", dict(axis="XY", phase=(1, 1)))):
        total = 0
        for family in ("B", "D"):
            fields, _pml = _complex_folded(**kwargs)
            names = tuple(triton_symmetry.GHOST_FILL_FAMILIES[family]["targets"])
            before = {n: np.array(getattr(fields, n), copy=True) for n in names}
            near = list(folded_complex.fill_axis_entries(fields.grid, family, "near"))
            far = list(folded_complex.fill_axis_entries(fields.grid, family, "far"))
            assert len(near) == 2, (label, family,
                                    "REACHABILITY: two folded axes are required")

            def walk(plan):
                plan.run_near()
                plan.run_far()

            forward = _fill_both_ways(fields, family, near, far, walk)
            for name in names:
                getattr(fields, name)[...] = before[name]
            reversed_ = _fill_both_ways(fields, family, list(reversed(near)),
                                        list(reversed(far)), walk)
            assert sum(_moved(before[n], forward[n]) for n in names) > 0, \
                (label, family, "VACUOUS: the fill wrote nothing")
            total += sum(_differing(forward[n], reversed_[n]) for n in names)
        outcome[label] = total
    assert outcome["matched"] == 0, outcome
    assert outcome["mixed"] > 0, (
        outcome, "reversing the axis order moved no byte at MIXED phase either: the "
        "X, Y, Z order this plan preserves would then be decorative here, and the "
        "real fold's fused plan could be reused")


@requires_mps
@requires_probe
def test_fusing_the_two_fill_passes_per_axis_is_a_measured_null_here():
    """The near/far split, re-asked under complex storage — and it is a NULL.

    A component unowned on two axes with DIFFERENT Yee shifts sees near-then-far on
    the array path and would see far-then-near under a fused per-axis launch. The
    Triton twin measures 5 words on its own engineered states; MEASURED HERE
    2026-08-16 on the matrix's two-axis folded complex grids, at mixed and matched
    phase, for both families, with random-uniform volumes AND with all four
    signed-zero sign combinations planted on the rows the fill reads: 0 words in
    every cell.

    THAT DOES NOT LICENSE FUSING THE PASSES, and the reason the split is kept is
    INDEPENDENT of this measurement: ``zero_metal_*`` runs BETWEEN them in the driver
    (driver.py:3286 / :3301) and the far pass reads a whole plane the wall clear can
    touch. The ordering question is a second reason that happens to be a null on the
    states reachable here; the driver-seam reason is not a measurement at all, it is
    the call order.

    The leg is kept because it is the one that would go non-null if a future grid
    shape or a plant reached the case, and because a null asserted is a null
    measured while a null assumed is a comment.
    """
    outcome = {}
    for label, kwargs in (("mixed", dict(axis="XY", phase=(1, -1))),
                          ("matched", dict(axis="XY", phase=(1, 1)))):
        for family in ("B", "D"):
            fields, _pml = _complex_folded(**kwargs)
            names = tuple(triton_symmetry.GHOST_FILL_FAMILIES[family]["targets"])
            codes, _ = folded_complex.folded_axis_kinds(fields.grid, None)
            reachable = [
                n for n in names
                if any(triton_symmetry.TARGET_IYEE[n][a] == 0
                       for a, c in enumerate(codes)
                       if int(c) in folded_complex.MIRROR_CODES)
                and any(triton_symmetry.TARGET_IYEE[n][a] == 1
                        for a, c in enumerate(codes)
                        if int(c) == folded_complex.CODE_MIRROR_PERIODIC)]
            assert reachable, (
                label, family,
                "REACHABILITY: no target is near on one folded axis and far on "
                "another, so the fusion question does not exist here")

            before = {n: np.array(getattr(fields, n), copy=True) for n in names}
            near = list(folded_complex.fill_axis_entries(fields.grid, family, "near"))
            far = list(folded_complex.fill_axis_entries(fields.grid, family, "far"))

            def split_walk(plan):
                plan.run_near()
                plan.run_far()

            def fused_walk(plan):
                by_axis = {}
                for entry in plan.near:
                    by_axis.setdefault(entry["axis"], [[], []])[0].append(entry)
                for entry in plan.far:
                    by_axis.setdefault(entry["axis"], [[], []])[1].append(entry)
                for axis in sorted(by_axis):
                    plan._walk(by_axis[axis][0], shaders.CONTRACT_OFF)
                    plan._walk(by_axis[axis][1], shaders.CONTRACT_OFF)

            split = _fill_both_ways(fields, family, near, far, split_walk)
            for name in names:
                getattr(fields, name)[...] = before[name]
            fused = _fill_both_ways(fields, family, near, far, fused_walk)
            assert sum(_moved(before[n], split[n]) for n in names) > 0, (label, family)
            outcome[(label, family)] = sum(_differing(split[n], fused[n])
                                           for n in names)
    assert set(outcome.values()) == {0}, outcome


@requires_mps
@requires_probe
@pytest.mark.parametrize("label,kwargs", FOLD_CASES, ids=[c[0] for c in FOLD_CASES])
def test_a_flipped_parity_is_caught_on_every_fold_case(label, kwargs):
    """The coefficient is PASSED, which is what makes this mutation reachable."""
    fields, _pml = _complex_folded(**kwargs)
    names = tuple(triton_symmetry.GHOST_FILL_FAMILIES["D"]["targets"])
    before = {n: np.array(getattr(fields, n), copy=True) for n in names}
    stepping.fill_symmetry_bc_D(fields)
    stepping.fill_folded_far_ghosts_D(fields)
    after = {n: np.array(getattr(fields, n), copy=True) for n in names}
    for name in names:
        getattr(fields, name)[...] = before[name]

    near = [dict(entry) for entry in
            folded_complex.fill_axis_entries(fields.grid, "D", "near")]
    far = [dict(entry) for entry in
           folded_complex.fill_axis_entries(fields.grid, "D", "far")]
    for entry in near + far:
        re, im = entry["coefficient"]
        entry["coefficient"] = (-re, im)

    residency = device.Residency()
    plan = folded_complex.plan_folded_complex_fill_from_arrays(
        "D", "fill_D", {n: getattr(fields, n) for n in names}, near, far,
        EXPANSION, residency)
    residency.sync_in()
    plan.run_near()
    plan.run_far()
    residency.sync_out()
    moved = sum(_moved(before[n], after[n]) for n in names)
    assert moved > 0, (label, "VACUOUS: the array path's fill wrote nothing")
    differing = sum(_differing(getattr(fields, n), after[n]) for n in names)
    assert differing > 0, (label, "a flipped parity was not caught")


@requires_mps
@requires_probe
def test_the_coefficient_orientation_is_a_measured_equivalence():
    """NAMED AS AN EQUIVALENCE RATHER THAN LEFT AS AN UNTESTED BELIEF.

    The array path spells ``phase * plane`` — coefficient on the LEFT — and the
    transcription keeps that. For a GENERAL complex coefficient the orientation is
    load-bearing (``complex_fields`` measures the Bloch rotation's swap as caught).
    For THIS coefficient it is not, and the reason is arithmetic: with ``c_im`` an
    exact ``+0.0`` the two fma addends are the same exact zero and float addition is
    sign-commutative. Measured rather than argued.
    """
    import torch

    from meep_gpu.metal_kernels.device import compile_source

    shipped = folded_complex.folded_mirror_fill_complex_source(
        1, "near", (0, 1, 0), EXPANSION)
    swapped = shipped.replace("c_mul(c, f0[", "c_mul_swapped(c, f0[").replace(
        "c_mul(c, f2[", "c_mul_swapped(c, f2[")
    swapped = swapped.replace(
        "kernel void folded_mirror_fill_complex(",
        "static inline float2 c_mul_swapped(float2 c, float2 z) "
        "{ return c_mul(z, c); }\n\nkernel void folded_mirror_fill_complex(")
    assert swapped != shipped

    shape = (4, 6, 5)
    rng = np.random.default_rng(7)
    plane = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
             ).astype(np.complex64)
    plane[:, 2, :] = np.complex64(complex(-0.0, -0.0))
    results = []
    for source in (shipped, swapped):
        module = compile_source(source)
        volumes = [np.ascontiguousarray(plane.copy()) for _ in range(3)]
        tensors = [torch.from_numpy(v.view(np.float32).reshape(-1, 2).copy()).to("mps")
                   for v in volumes]
        module.folded_mirror_fill_complex(
            *tensors, shape[0], shape[1], shape[2], shape[0] * shape[2], -1,
            (-1.0, 0.0))
        torch.mps.synchronize()
        results.append([t.cpu().numpy().reshape(-1).view(np.complex64).reshape(shape)
                        for t in tensors])
    moved = sum(_moved(plane, results[0][i]) for i in (0, 2))
    assert moved > 0, "VACUOUS: the shipped fill wrote nothing"
    differing = sum(_differing(results[0][i], results[1][i]) for i in (0, 2))
    assert differing == 0, (differing,
                            "the orientation swap moved bytes: it is NOT the "
                            "equivalence this family records it as")
