"""Merge-bar tests for the Metal FOLDED BETA family — a composition of two.

THE QUESTION THIS FILE ANSWERS IS NOT "IS EACH HALF RIGHT". Both halves are already
byte-certified alone: ``special_kz`` (the real and Bloch beta curls and the real
arm's constitutive companion) and ``symmetry`` / ``folded_complex`` (the fold). The
question is whether they remain right TOGETHER, and the composition moves exactly
one thing: THE FOLD WIDENS THE MASK THE BETA TERM LANDS INSIDE. Unfolded, the beta
increment meets one ownership mask (cell 0 on a non-periodic axis); folded, it meets
two, because a folded PERIODIC axis also masks its LAST plane for every target whose
Yee shift is 1 there. Every leg below exists because of that sentence.

**1. THE STRUCTURAL COMPOSITION.** Each folded beta curl is
``special_kz``'s OWN certified template with ONE block inserted at a CHECKED anchor,
and its ghost gather and cell-0 mask are the certified emitters' output reached by
mapping every mirror code to ``METALLIC``. Both halves are checked BY CHARACTER, in
both directions, because ``torch.mps.compile_shader`` exposes no disassembly and
source text is the only place the claim can be held on this backend. The strongest
form is here: on an UNFOLDED grid the emitted source must be character-identical to
the parent's except for an empty top-plane block.

**2. THE COMPOSITION'S OWN HAZARD, WHICH NEITHER PARENT CAN TEST.** The beta term
must sit AFTER the curl and BEFORE **both** masks — the array path adds it at
stepping.py:384-391 / :467-474 and masks at :397 / :479, whose
``_mask_non_owned_cells`` carries the cell-0 arm at :1945-1949 and the top-plane arm
at :1887-1897. A kernel that put it after either mask would write a beta
contribution into a cell the array path zeroes. Checked in the SOURCE by ordering,
and refuted on the DEVICE by the ordering mutations.

**3. THE TWO MASKS REACH DIFFERENT TARGETS ON DIFFERENT SIDES, and both reach a
beta-modified curl.** On a y fold, ``step_B``'s beta targets are ``curl0`` (Bx, Yee
shift 1 on y — TOP-plane masked) and ``curl1`` (By, shift 0 — CELL-0 masked);
``step_D``'s are ``curl0`` (Dx, shift 0 — cell-0 masked) and ``curl1`` (Dy, shift 1
— top-plane masked). So a leg on one sub-step proves nothing about the other, and
between them the two sub-steps put the beta term under each mask exactly once.

**4. THE TWO TERMINATIONS ARE DIFFERENT KERNELS.** On MIRROR_METALLIC the top plane
is OWNED and stepped and ``folded_top_plane_mask`` emits nothing; on MIRROR_PERIODIC
it is masked. A sweep on one termination proves nothing about the other — and on
this family that mask sits directly downstream of the beta term.

**5. THE DISJOINTNESS**, stated as inversions rather than inherited. This family has
four neighbours to stay clear of — ``special_kz``'s two arms and the two fold
families — and it inverts TWO clauses against them rather than one. Both directions
are pinned here.

**6. THE FILL SLOTS ARE NOT CLAIMED, DELIBERATELY**, and that is pinned as an
invariant rather than left as an absence: the parents' fill predicates carry no beta
clause and already admit a folded beta run, so a third arm would make both seam
slots AMBIGUOUS and drop them to the array path.

Device-touching tests are skipped without MPS. EVERY behavioural leg asserts a
VACUITY FLOOR — words actually moved — because zero-init is a fixed point of the
constitutive sub-step and a no-op agreeing with a no-op is trivially identical.

THERE IS NO GENERATED-CODE AUDIT ON THIS BACKEND. Every device leg here is
BEHAVIOURAL: it catches a wrong answer, never a wrong instruction.
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
    arms, device, folded_beta, folded_complex, shaders, special_kz, symmetry,
)

ENVIRONMENT = matrix.prepare_environment()


def _mps() -> bool:
    try:
        import torch
    except Exception:  # noqa: BLE001 - no torch is a skip, not a failure
        return False
    return bool(torch.backends.mps.is_available())


requires_mps = pytest.mark.skipif(not _mps(), reason="no MPS device on this host")

#: The per-axis code triples the corpus's four folded-beta rows drive, plus the
#: METALLIC termination the corpus does not drive but the emitter must get right.
#: Every one is a y fold, which is what all four corpus rows are.
CODES_PERIODIC = (symmetry.CODE_PERIODIC, symmetry.CODE_MIRROR_PERIODIC,
                  symmetry.CODE_PERIODIC)
CODES_METALLIC = (symmetry.CODE_PERIODIC, symmetry.CODE_MIRROR_METALLIC,
                  symmetry.CODE_PERIODIC)
CODES_UNFOLDED = (symmetry.CODE_PERIODIC,) * 3

#: The volumes a whole-step leg compares. Both storages carry the same names.
SNAPSHOT = ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
            "Hx", "Hy", "Hz", "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy",
            "fu_Dz")


def _words(array) -> np.ndarray:
    flat = np.ascontiguousarray(array).reshape(-1)
    return flat.view(np.uint32)


def _differing(a, b) -> int:
    return int(np.count_nonzero(_words(a) != _words(b)))


def _moved(before, after) -> int:
    return int(np.count_nonzero(_words(before) != _words(after)))


# ---------------------------------------------------------------------------
# 1. The structural composition — checked BY CHARACTER, in both directions
# ---------------------------------------------------------------------------

def _pass(source: str) -> str:
    """The beta pass's text, between its two marker lines."""
    from meep_gpu.metal_kernels import folded_pass

    start = source.index(folded_pass.FOLDED_PASS_BEGIN)
    return source[start:source.index(folded_pass.FOLDED_PASS_END, start)]


def _without_pass(source: str, kernel: str, parent_kernel: str, first: int) -> str:
    """``source`` with the pass, the two beta bindings and the rename taken back out."""
    from meep_gpu.metal_kernels import folded_pass

    start = source.index(folded_pass.FOLDED_PASS_BEGIN)
    end = source.index(folded_pass.FOLDED_PASS_END)
    out = source[:start] + source[end:].split("\n", 2)[2]
    out = out.replace(f"kernel void {kernel}(", f"kernel void {parent_kernel}(")
    pad = "     " if kernel == "beta_pml_curl_step" else "      "
    out = out.replace(special_kz.beta_parameters(first, pad), "")
    return out.replace(f"[[buffer({first + 2})]]", f"[[buffer({first})]]")


@pytest.mark.parametrize("backward", (False, True), ids=("step_B", "step_D"))
def test_the_real_source_is_the_certified_folded_curl_plus_exactly_the_pass(backward):
    """DERIVED RATHER THAN COPIED, and that is the claim this pins.

    The certified folded curl (:func:`.symmetry.folded_curl_source`) carries the
    fold's ghost rule, both ownership masks, the chunk record and the per-chunk
    forms. This family adds the beta pass, two bindings and an entry-point name, and
    NOTHING else: removing those three gives the parent back byte for byte.
    """
    for codes in (CODES_PERIODIC, CODES_METALLIC, CODES_UNFOLDED,
                  (symmetry.CODE_MIRROR_PERIODIC, symmetry.CODE_METALLIC,
                   symmetry.CODE_PERIODIC)):
        ours = folded_beta.folded_beta_curl_source(codes, backward)
        parent = symmetry.folded_curl_source(codes, backward)
        assert _without_pass(ours, "beta_pml_curl_step", "folded_pml_curl_step",
                             shaders.CURL_RECORD_BUFFER) == parent, codes


@pytest.mark.parametrize("backward", (False, True), ids=("step_B", "step_D"))
def test_the_complex_source_is_the_certified_folded_complex_curl_plus_the_pass(
        backward):
    """The complex twin: :func:`.folded_complex.folded_bloch_curl_source` plus the
    complex pass, and nothing else."""
    from meep_gpu.metal_kernels import complex_fields

    for codes, phased in ((CODES_PERIODIC, (0, 0, 0)), (CODES_PERIODIC, (2, 0, 0)),
                          (CODES_METALLIC, (1, 0, 0))):
        ours = folded_beta.folded_beta_bloch_curl_source(codes, backward, phased)
        parent = folded_complex.folded_bloch_curl_source(codes, backward, phased)
        assert _without_pass(ours, "beta_bloch_pml_curl_step", "bloch_pml_curl_step",
                             complex_fields.CURL_RECORD_BUFFER) == parent, codes


def test_the_pass_is_special_kz_s_own_text_under_the_fold_s_mask_rows():
    """One spelling of the beta pass: :func:`.special_kz.beta_pass` on the rows the
    folded source carries, so the pass's increment is masked by exactly the cell-0
    and top-plane rows that mask the curl."""
    from meep_gpu.metal_kernels import folded_pass

    for backward in (False, True):
        source = folded_beta.folded_beta_curl_source(CODES_PERIODIC, backward)
        rows = folded_pass.ownership_rows(symmetry.folded_curl_source(CODES_PERIODIC,
                                                                      backward))
        assert special_kz.beta_pass(rows, False) in source


def test_the_ghost_and_cell_zero_mask_are_the_certified_emitters_output():
    """Both mirror codes take the METALLIC branch — a CALL, never copied text."""
    for backward in (False, True):
        source = folded_beta.folded_beta_curl_source(CODES_PERIODIC, backward)
        reduced = symmetry._reduced_codes(CODES_PERIODIC)
        assert shaders.ownership_mask(reduced, backward) in source
        for index, axis in enumerate("xyz"):
            assert shaders.ghost(axis, reduced[index], backward) in source


# ---------------------------------------------------------------------------
# 2. THE COMPOSITION'S OWN HAZARD — where the beta term sits
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backward", (False, True), ids=("step_B", "step_D"))
def test_the_pass_follows_both_masks_and_the_forms_in_the_real_source(backward):
    """MEEP's beta pass runs after every component's curl (step_db.cpp:161-175).

    The curl, both of its ownership masks and the per-chunk forms come first; the
    pass then masks its OWN increment by the same rows and adds it through the
    curl's ladder, in front of the stores.
    """
    from meep_gpu.metal_kernels import folded_pass

    source = folded_beta.folded_beta_curl_source(CODES_PERIODIC, backward)
    curl = source.index("float curl0 =")
    cell_zero = source.index("bool at_x")
    top = source.index("bool last_x")
    forms = source.index("float v2 = ")
    beta = source.index("-(beta_plus * b)")
    store = source.index(folded_pass.FOLDED_PASS_ANCHOR)
    assert curl < cell_zero < top < forms < beta < store, (curl, cell_zero, top,
                                                           forms, beta, store)


@pytest.mark.parametrize("backward", (False, True), ids=("step_B", "step_D"))
def test_the_pass_follows_both_masks_and_the_forms_in_the_complex_source(backward):
    from meep_gpu.metal_kernels import folded_pass

    source = folded_beta.folded_beta_bloch_curl_source(
        CODES_PERIODIC, backward, (2, 0, 0))
    curl = source.index("float2 curl0 =")
    cell_zero = source.index("bool at_x")
    top = source.index("bool last_x")
    forms = source.index("float2 v2 = ")
    beta = source.index("float2(-(beta_plus * b.y), beta_plus * b.x)")
    store = source.index(folded_pass.FOLDED_PASS_ANCHOR)
    assert curl < cell_zero < top < forms < beta < store


@pytest.mark.parametrize("backward,beta_targets,cell_zero,top", (
    # On a Y fold. B side: Bx (0,1,1) shift 1 on y -> TOP masked; By (1,0,1) shift 0
    # -> CELL-0 masked; Bz (1,1,0) shift 1 -> TOP masked.
    (False, ("curl0", "curl1"), ("curl1",), ("curl0", "curl2")),
    # D side: Dx (1,0,0) shift 0 -> CELL-0; Dy (0,1,0) shift 1 -> TOP; Dz (0,0,1)
    # shift 0 -> CELL-0.
    (True, ("curl0", "curl1"), ("curl0", "curl2"), ("curl1",)),
), ids=("step_B", "step_D"))
def test_each_mask_reaches_a_beta_modified_curl_on_exactly_one_side(
        backward, beta_targets, cell_zero, top):
    """WHY A LEG ON ONE SUB-STEP PROVES NOTHING ABOUT THE OTHER.

    The beta pass always reaches targets 0 and 1; which of those two the
    cell-0 mask takes and which the top-plane mask takes SWAPS between the sides,
    because the B family's Yee shifts are 1 on the two axes that are NOT its own and
    the D family's are 1 on its own axis only. Between them the two sub-steps put
    the beta term under each mask exactly once, and a family gated on one would
    leave the other's interaction unmeasured.
    """
    source = folded_beta.folded_beta_curl_source(CODES_PERIODIC, backward)

    def masked(flag: str, name: str):
        return {line.split(" =")[0].strip()
                for line in source.splitlines()
                if f"{flag} ?" in line and line.strip().startswith(name)}

    masked_at_zero = masked("at_y", "curl")
    masked_at_top = masked("last_y", "curl")
    assert masked_at_zero == set(cell_zero), (backward, masked_at_zero)
    assert masked_at_top == set(top), (backward, masked_at_top)
    # The composition's content: each mask deletes exactly one of the two targets
    # the pass reaches, and the pass's increment is masked on exactly those rows.
    assert len(masked_at_zero & set(beta_targets)) == 1
    assert len(masked_at_top & set(beta_targets)) == 1
    as_increment = {name.replace("curl", "bi") for name in beta_targets}
    assert masked("at_y", "bi") == {n.replace("curl", "bi")
                                    for n in masked_at_zero} & as_increment
    assert masked("last_y", "bi") == {n.replace("curl", "bi")
                                      for n in masked_at_top} & as_increment


def test_a_metallic_fold_emits_no_top_plane_mask_at_all():
    """MIRROR_METALLIC stores up to ``big_corner``: the top plane is OWNED.

    Masking it there would delete a real cell — and on this family it would delete a
    real cell that CARRIES the beta term. A sweep on one termination proves nothing
    about the other.
    """
    for backward in (False, True):
        metallic = folded_beta.folded_beta_curl_source(CODES_METALLIC, backward)
        periodic = folded_beta.folded_beta_curl_source(CODES_PERIODIC, backward)
        assert "last_y ?" not in metallic
        assert "no folded PERIODIC axis" in metallic
        assert "last_y ?" in periodic
        assert metallic != periodic


def test_the_beta_partner_is_a_center_load_and_is_never_bloch_connected():
    """The array path's beta partner is the stored same-cell snapshot.

    MEEP's Bloch connection applies to the unshifted operand AS READ ACROSS A WRAP
    (the curl's ``c_dx``/``b_dx``); the pass reads the stored centres ``a``/``b``.
    Pinned by character.
    """
    source = folded_beta.folded_beta_bloch_curl_source(
        CODES_PERIODIC, False, (2, 0, 0))
    block = _pass(source)
    assert "b_dx" in source and "b_dx" not in block
    assert "float2(-(beta_plus * b.y), beta_plus * b.x)" in block
    assert "float2(-(beta_minus * a.y), beta_minus * a.x)" in block


def test_a_phased_folded_axis_is_refused_by_the_emitter_as_well_as_the_predicate():
    """A mis-baked flag is a plane of wrong values, not a crash.

    ``driver._require_bloch_is_representable`` refuses the configuration outright,
    so a kernel cannot lift what the array path will not run — and this is the LAST
    place it can be seen.
    """
    with pytest.raises(ValueError, match="folded and carries a Bloch phase"):
        folded_beta.folded_beta_bloch_curl_source(
            CODES_PERIODIC, False, (0, 1, 0))
    with pytest.raises(ValueError, match="ghost rule is METALLIC"):
        folded_beta.folded_beta_bloch_curl_source(
            (symmetry.CODE_METALLIC,) * 3, False, (1, 0, 0))


# ---------------------------------------------------------------------------
# 3. The predicate — TWO inverted clauses, both directions
# ---------------------------------------------------------------------------

def _reason_text(coverage) -> str:
    return " | ".join(coverage.reasons)


def test_an_unfolded_beta_run_is_refused_and_special_kz_admits_it():
    """CLAUSE 5, both directions: this family requires the fold, ``special_kz`` refuses it."""
    fields, pml = matrix.flat(beta=0.33)
    residency = device.Residency()
    for slot in ("step_B", "step_D"):
        ours = folded_beta.folded_beta_composition_curl_coverage(
            fields, pml, slot, residency)
        assert not ours.covered
        assert "no mirror plane is active" in _reason_text(ours)
        theirs = special_kz.beta_pml_curl_coverage(fields, pml, slot, residency)
        assert theirs.covered, theirs.reasons


def test_a_folded_run_at_beta_zero_is_refused_and_the_fold_families_admit_it():
    """CLAUSE 12, both directions: this family requires beta, the folds refuse it."""
    fields, pml = matrix.folded()
    residency = device.Residency()
    for slot in ("step_B", "step_D"):
        ours = folded_beta.folded_beta_composition_curl_coverage(
            fields, pml, slot, residency)
        assert not ours.covered
        assert "grid.beta is zero" in _reason_text(ours)
        theirs = symmetry.folded_composition_curl_coverage(
            fields, pml, slot, residency)
        assert theirs.covered, theirs.reasons


def test_the_two_fold_families_refuse_a_folded_beta_run_and_name_this_one():
    """The PRE-REGISTERED inversion, honoured rather than re-spelled.

    ``symmetry`` clause 12 and ``folded_complex`` clause 8 each refuse ``grid.beta``
    and NAME the owner. If either stopped naming it, a reader chasing the refusal
    would find no owner — which is the silent-gap failure the whole clause
    discipline exists to prevent.
    """
    residency = device.Residency()
    real_fields, real_pml = matrix.folded(beta=0.3)
    complex_fields_, complex_pml = matrix.folded(complex_storage=True, beta=0.3)
    for slot in ("step_B", "step_D"):
        real = symmetry.folded_composition_curl_coverage(
            real_fields, real_pml, slot, residency)
        assert not real.covered
        assert "folded_beta" in _reason_text(real), real.reasons
        cplx = folded_complex.folded_complex_composition_curl_coverage(
            complex_fields_, complex_pml, slot, residency)
        assert not cplx.covered
        assert "folded_beta" in _reason_text(cplx), cplx.reasons


def test_the_two_arms_split_on_storage_in_both_directions():
    """CLAUSE 2, and both arms must be able to SAY which storage they need."""
    residency = device.Residency()
    real_fields, real_pml = matrix.folded(beta=0.3)
    cplx_fields, cplx_pml = matrix.folded(complex_storage=True, beta=0.3)
    for slot in ("step_B", "step_D"):
        assert folded_beta.folded_beta_composition_curl_coverage(
            real_fields, real_pml, slot, residency).covered
        refused = folded_beta.folded_beta_composition_bloch_curl_coverage(
            real_fields, real_pml, slot, residency)
        assert not refused.covered
        assert "storage is real float32" in _reason_text(refused)

        assert folded_beta.folded_beta_composition_bloch_curl_coverage(
            cplx_fields, cplx_pml, slot, residency).covered
        refused = folded_beta.folded_beta_composition_curl_coverage(
            cplx_fields, cplx_pml, slot, residency)
        assert not refused.covered
        assert "force_complex_fields=True" in _reason_text(refused)


def test_the_real_arm_refuses_an_off_diagonal_row_outright_because_the_engine_raises():
    """stepping.py:800-810 RAISES and MEEP fields.cpp:548-549 ABORTS on that trio.

    The implicit-i trick cancels only while TE and TM stay uncoupled, so the refusal
    is for the WHOLE real arm rather than only ``update_E``: the run raises inside
    the first beta curl.
    """
    fields, pml = matrix.folded(beta=0.3, rows={"Ex": ("Ey",)})
    residency = device.Residency()
    for slot in ("step_B", "step_D"):
        verdict = folded_beta.folded_beta_composition_curl_coverage(
            fields, pml, slot, residency)
        assert not verdict.covered
        assert "implicit-i trick no longer cancels" in _reason_text(verdict)


def test_a_folded_axis_may_carry_no_bloch_phase_and_an_unfolded_one_may():
    """CLAUSE 9 — imported from ``folded_complex``, which established it.

    Two of the four corpus rows carry ``kx != 0`` on the UNFOLDED axis with the fold
    on y, so the admitting half of this pin is a real corpus shape rather than a
    hypothetical.
    """
    residency = device.Residency()
    fields, pml = matrix.folded(complex_storage=True, beta=0.3,
                               k_point=(0.3, 0.0, 0.0))
    for slot in ("step_B", "step_D"):
        assert folded_beta.folded_beta_composition_bloch_curl_coverage(
            fields, pml, slot, residency).covered


def test_the_constitutive_pair_is_admitted_and_the_e_side_scopes_dispersion():
    """The fold and beta both reach ``update_H``/diagonal ``update_E`` NOT AT ALL.

    So the pair is the certified body under a restated predicate — and dispersion,
    which DOES change what ``source`` is, is refused on the E side alone.
    """
    residency = device.Residency()
    fields, pml = matrix.folded(beta=0.3)
    for side in ("H", "E"):
        assert folded_beta.folded_beta_constitutive_coverage(
            fields, pml, side, residency).covered

    fields, pml = matrix.dispersive(matrix.folded(beta=0.3))
    assert folded_beta.folded_beta_constitutive_coverage(
        fields, pml, "H", residency).covered
    refused = folded_beta.folded_beta_constitutive_coverage(
        fields, pml, "E", residency)
    assert not refused.covered
    assert "D - sum P" in _reason_text(refused)


def test_the_two_complex_beta_constitutive_products_are_disjoint_on_the_fold():
    """THE GAP THIS FAMILY MADE VISIBLE IS CLOSED, and the seam is the fold.

    This used to assert that the UNFOLDED complex beta constitutive was still
    nobody's — two slots, one corpus row — by pinning ``special_kz``'s refusal
    message. ``special_kz`` restated the predicate one level out on 2026-08-19, so
    both cases now have a product and the fact worth pinning changed with them: they
    must not BOTH admit. This family requires a mirror plane; ``special_kz``'s
    requires there be none, and that single clause is the whole separation.
    """
    unfolded, unfolded_pml = matrix.flat(beta=0.33, complex_storage=True)
    residency = device.Residency()
    for side in ("H", "E"):
        mine = folded_beta.folded_beta_complex_constitutive_coverage(
            unfolded, unfolded_pml, side, residency)
        theirs = special_kz.beta_run_complex_constitutive_coverage(
            unfolded, unfolded_pml, side, residency)
        assert theirs.covered, (side, theirs.reasons)
        assert not mine.covered
        assert "no mirror plane is active" in _reason_text(mine)


# ---------------------------------------------------------------------------
# 4. The wiring — what this family claims, and what it deliberately does not
# ---------------------------------------------------------------------------

def test_the_family_is_named_in_the_registry_module_list():
    """A family in the tree but not in ``FAMILY_MODULES`` is INVISIBLE to plan_step.

    That is a silent COVERAGE LOSS rather than an error: the composer would leave
    every slot on the array path and never complain. Asserted against the PACKAGE
    contents, not against a literal, so a module added and forgotten is caught.
    """
    from meep_gpu.metal_kernels import registry

    assert "folded_beta" in registry.FAMILY_MODULES
    contributed = {spec.coverage.__module__.rsplit(".", 1)[-1]
                   for spec in arms.registered()}
    assert "folded_beta" in contributed
    assert contributed <= set(registry.FAMILY_MODULES) | {"launch", "coverage"}


def test_the_family_registers_eight_arms_over_the_four_arithmetic_slots():
    labels = {}
    for slot in ("step_B", "step_D", "update_H", "update_E"):
        here = [spec for spec in arms.registered(slot)
                if spec.family in (folded_beta.FAMILY_REAL,
                                   folded_beta.FAMILY_COMPLEX)]
        assert len(here) == 2, (slot, [s.family for s in here])
        labels[slot] = {spec.label for spec in here}
    assert all(names == {folded_beta.LABEL_REAL, folded_beta.LABEL_COMPLEX}
               for names in labels.values()), labels
    # BY FAMILY IDENTITY, NOT BY NAME PREFIX. This asserts that FOLDED_BETA'S OWN arms
    # are wired, and a prefix match is the wrong instrument for that: since 2026-08-27
    # the tree also carries `folded_beta_complex_fused_magnetic_pair`, a WELD over this
    # family's curl, which registers wired=False deliberately -- unwired keeps it
    # enumerable for the disjointness sweep while arms_for skips it, so plan_step cannot
    # select it and no existing arm's selection changes. Matching on the prefix swept
    # that weld into a clause about a different family and would sweep in any future one.
    assert all(spec.wired for spec in arms.registered()
               if spec.family in (folded_beta.FAMILY_REAL, folded_beta.FAMILY_COMPLEX))


def test_no_fill_arm_is_registered_and_the_parents_still_own_those_slots():
    """A THIRD ARM ON THE SEAM SLOTS WOULD BE A COVERAGE LOSS DRESSED AS COMPLETENESS.

    ``fill_symmetry_bc_*`` copies ``parity * field[plane]`` and reads no beta term,
    so both parents' fill predicates carry no beta clause and already admit a folded
    beta run. Registering here would make both slots AMBIGUOUS and drop them to the
    array path.
    """
    for slot in ("fill_B", "fill_D"):
        families = {spec.family for spec in arms.registered(slot)}
        assert not any(name.startswith("folded_beta") for name in families), (
            slot, families)
    residency = device.Residency()
    fields, _ = matrix.folded(beta=0.3)
    assert symmetry.mirror_ghost_fill_coverage(fields, "B", residency).covered
    cplx, _ = matrix.folded(complex_storage=True, beta=0.3)
    assert folded_complex.folded_complex_fill_coverage(
        cplx, "B", residency).covered


def test_update_p_belongs_only_to_the_independent_ade_family():
    """This folded-beta family does not absorb the geometry-free ADE recurrence."""
    owners = {spec.family for spec in arms.registered("update_P")}
    assert owners == {"ade_update_p"}
    assert matrix.UNREGISTERED_SLOTS == ()


@pytest.mark.parametrize("row", ("fold_real_2d_beta", "fold_complex_2d_beta"))
def test_the_two_rows_left_uncarried_in_this_change(row):
    """THE NON-VACUITY FLOOR, MOVED DELIBERATELY.

    ``UNCARRIED`` pins rows that must select NOTHING on any arithmetic slot. These
    two were on that list until this family made them real, and moving them is only
    legitimate in the SAME change — never to make a red floor go green. The matrix
    row must now pin all four arithmetic slots.
    """
    assert row not in matrix.UNCARRIED
    pinned = dict((name, expected) for name, _, expected, _ in matrix.MATRIX)[row]
    assert set(matrix.ARITHMETIC_SLOTS) <= set(pinned), pinned


# ---------------------------------------------------------------------------
# 5. The device legs — a whole step, against stepping.py, with launch counters
# ---------------------------------------------------------------------------

WHOLE_STEP_CASES = (
    ("real_y_periodic", dict(beta=0.3)),
    ("real_y_metallic", dict(beta=0.3, boundaries={"y": "metallic"})),
    ("real_y_odd_parity", dict(beta=-0.41, phase=-1)),
    ("real_x_fold", dict(beta=0.3, axis="X")),
)


@requires_mps
@pytest.mark.parametrize("label,kwargs", WHOLE_STEP_CASES,
                         ids=[c[0] for c in WHOLE_STEP_CASES])
def test_a_complete_real_step_agrees_and_the_first_divergent_step_is_reported(
        label, kwargs):
    """THE REAL ARBITER. Six green sub-step legs say nothing about the object the
    engine would run.

    Failure classes that live ONLY in a complete step: a STALE MIRROR, a SEAM, and
    an ACCUMULATING AUXILIARY (``fu_*`` is STATE, and a kernel right for one launch
    and wrong forever after is identical in a single-launch leg). THE FOLD ADDS A
    FOURTH: the folded axis's ghost plane is written by one pass and READ by the
    next, and the beta term rides the plane the second mask deletes.

    The fill passes are the PARENTS' — this family registers none — so this leg also
    measures that the parents' fills compose with this family's curls, which is the
    claim the "no fill arm" decision rests on.
    """
    budget = 6
    fields, pml = matrix.folded(**kwargs)
    reference_fields, reference_pml = matrix.folded(**kwargs)

    residency = device.Residency()
    plans = {
        "step_B": folded_beta.plan_folded_beta_pml_curl(
            fields, pml, "step_B", residency,
            chunk_bounds=_one_rank_record(fields, pml)),
        "fill_B": symmetry.plan_mirror_ghost_fill(fields, "B", "fill_B", residency),
        "update_H": folded_beta.plan_folded_beta_constitutive(
            fields, pml, "H", residency, chunk_bounds=_one_rank_record(fields, pml)),
        "step_D": folded_beta.plan_folded_beta_pml_curl(
            fields, pml, "step_D", residency,
            chunk_bounds=_one_rank_record(fields, pml)),
        "fill_D": symmetry.plan_mirror_ghost_fill(fields, "D", "fill_D", residency),
        "update_E": folded_beta.plan_folded_beta_constitutive(
            fields, pml, "E", residency, chunk_bounds=_one_rank_record(fields, pml)),
    }
    assert all(plan is not None for plan in plans.values()), {
        name: plan for name, plan in plans.items() if plan is None}
    residency.sync_in()

    first_divergent = None
    per_step = []
    for step in range(budget):
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
    for slot in ("step_B", "step_D", "update_H", "update_E"):
        assert plans[slot].launches == budget, (label, slot, plans[slot].launches)


COMPLEX_CASES = (
    ("complex_y_periodic", dict(complex_storage=True, beta=0.3)),
    ("complex_y_bloch_x", dict(complex_storage=True, beta=0.3,
                               k_point=(0.3, 0.0, 0.0))),
    ("complex_y_metallic", dict(complex_storage=True, beta=-0.41,
                                boundaries={"y": "metallic"})),
)


@requires_mps
@pytest.mark.parametrize("label,kwargs", COMPLEX_CASES,
                         ids=[c[0] for c in COMPLEX_CASES])
def test_a_complete_complex_step_agrees_and_reports_the_first_divergent_step(
        label, kwargs):
    """The complex arm's whole-step leg, including the Bloch-on-an-unfolded-axis row
    two of the four corpus rows actually are."""
    budget = 6
    fields, pml = matrix.folded(**kwargs)
    reference_fields, reference_pml = matrix.folded(**kwargs)

    residency = device.Residency()
    plans = {
        "step_B": folded_beta.plan_folded_beta_bloch_pml_curl(
            fields, pml, "step_B", residency,
            chunk_bounds=_one_rank_record(fields, pml)),
        "fill_B": folded_complex.plan_folded_complex_fill(
            fields, "B", "fill_B", residency),
        "update_H": folded_beta.plan_folded_beta_complex_constitutive(
            fields, pml, "H", residency,
            chunk_bounds=_one_rank_record(fields, pml)),
        "step_D": folded_beta.plan_folded_beta_bloch_pml_curl(
            fields, pml, "step_D", residency,
            chunk_bounds=_one_rank_record(fields, pml)),
        "fill_D": folded_complex.plan_folded_complex_fill(
            fields, "D", "fill_D", residency),
        "update_E": folded_beta.plan_folded_beta_complex_constitutive(
            fields, pml, "E", residency,
            chunk_bounds=_one_rank_record(fields, pml)),
    }
    assert all(plan is not None for plan in plans.values()), {
        name: plan for name, plan in plans.items() if plan is None}
    residency.sync_in()

    first_divergent = None
    per_step = []
    for step in range(budget):
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
    for slot in ("step_B", "step_D", "update_H", "update_E"):
        assert plans[slot].launches == budget, (label, slot, plans[slot].launches)


# ---------------------------------------------------------------------------
# 6. Mutations — a leg that cannot fail certifies nothing
# ---------------------------------------------------------------------------

#: Each mutation is a SOURCE EDIT applied to the shipped folded beta curl, launched
#: through the plan's ``functions`` seam so the mutant actually RUNS. ``must_catch``
#: records whether the shipped answer is required to differ; a declared EQUIVALENCE
#: is measured too, and asserting it is null is what stops it becoming folklore.
MUTATIONS = (
    ("beta_folded_into_the_curl",
     lambda src: _fold_into_curl(src), "must_catch"),
    ("increment_top_plane_row_dropped",
     lambda src: "\n".join(line for line in src.splitlines()
                           if not (line.strip().startswith("bi") and "last_y ?" in line)),
     "must_catch"),
    ("top_plane_mask_dropped",
     lambda src: "\n".join(line for line in src.splitlines()
                           if "last_y ?" not in line), "must_catch"),
    ("beta_signs_swapped",
     lambda src: src.replace("-(beta_plus * b)", "-(__TMP__ * b)")
                    .replace("-(beta_minus * a)", "-(beta_plus * a)")
                    .replace("-(__TMP__ * b)", "-(beta_minus * b)"), "must_catch"),
    # A MEASURED EQUIVALENCE, not a belief: float multiplication is commutative, so
    # the shipped operand order is transcription fidelity rather than a bit
    # requirement. Asserted null so it cannot quietly become folklore.
    ("multiply_operands_swapped",
     lambda src: src.replace("(beta_plus * b)", "(b * beta_plus)")
                    .replace("(beta_minus * a)", "(a * beta_minus)"),
     "equivalence"),
    # `0.0f - x` differs from `-x` only on x = +0, and the ladder's subtraction then
    # differs only where the field word is -0 in the same cell; see the test below
    # for why this family's physical-band state reaches neither.
    ("negation_spelled_zero_minus",
     lambda src: src.replace("float bi0 = -(beta_plus * b);",
                             "float bi0 = 0.0f - (beta_plus * b);"),
     "unobservable_here"),
)


def _fold_into_curl(source: str) -> str:
    """The PREVIOUS package form: the term inside the curl, before both masks and
    the per-chunk forms, with no pass of its own."""
    from meep_gpu.metal_kernels import folded_pass

    start = source.index(folded_pass.FOLDED_PASS_BEGIN)
    end = source.index(folded_pass.FOLDED_PASS_END) + len(folded_pass.FOLDED_PASS_END) + 1
    source = source[:start] + source[end:]
    anchor = "    bool at_x = (i == 0)"
    assert source.count(anchor) == 1
    return source.replace(anchor, "    curl0 = curl0 - (beta_plus * b);\n"
                                  "    curl1 = curl1 - (beta_minus * a);\n" + anchor)


def _pml_vectors(pml, sub_step: str):
    """The coefficient vectors for ONE sub-step, at ITS OWN sub-lattice.

    ``step_B`` writes the HALF-INTEGER lattice and takes the ``_h`` suffix;
    ``step_D`` takes the integer one. Binding the wrong pair is a whole-volume
    divergence that looks like a broken kernel and is a broken HARNESS — measured
    here at 241 differing words on a leg that was otherwise correct, which is why
    the suffix is derived from ``SUB_STEPS`` rather than spelled per call site.
    """
    from meep_gpu.metal_kernels.launch import SUB_STEPS

    suffix = SUB_STEPS[sub_step]["suffix"]
    return {f"{stem}_{axis}": getattr(pml, f"{stem}_{axis}{suffix}")
            for axis in "xyz" for stem in ("kms", "sinv")}


def _from_arrays_plan(fields, pml, sub_step, codes, residency, *, beta_plus=0.0,
                      beta_minus=0.0, functions=None, has_beta=True):
    """The gate route: bare arrays, explicit codes, and the MUTATION SEAM open."""
    arrays = {name: getattr(fields, name) for name in SNAPSHOT
              if getattr(fields, name, None) is not None}
    return folded_beta.plan_folded_beta_pml_curl_from_arrays(
        sub_step, arrays, _pml_vectors(pml, sub_step), codes,
        fields.grid.dt / fields.grid.dx, beta_plus, beta_minus, residency,
        functions=functions, has_beta=has_beta,
        chunk_bounds=_one_rank_record(fields, pml))


@requires_mps
@pytest.mark.parametrize("name,mutate,verdict", MUTATIONS,
                         ids=[m[0] for m in MUTATIONS])
def test_the_armed_mutations_are_caught_on_a_folded_beta_step(name, mutate,
                                                              verdict):
    """THE LEG THAT MAKES THE OTHER LEGS MEAN SOMETHING.

    Every mutation is compiled and LAUNCHED through the plan's ``functions`` SEAM.
    Dropping that argument is not a slowdown, it is a silent DISARMING: every leg
    would launch the shipped kernel and report every defect as uncaught.
    """
    fields, pml = matrix.folded(beta=0.3)
    reference_fields, reference_pml = matrix.folded(beta=0.3)
    grid = fields.grid
    codes, _ = symmetry.folded_axis_kinds(grid, pml)
    residency = device.Residency()

    plus, minus = special_kz.beta_curl_coefficients(
        grid.beta, grid.dt, magnetic=True, complex_storage=False)
    source = folded_beta.folded_beta_curl_source(codes, backward=False)
    mutant_source = mutate(source)
    assert mutant_source != source, (name, "the mutation edited nothing")
    entry = device.compile_source(mutant_source).beta_pml_curl_step

    plan = _from_arrays_plan(fields, pml, "step_B", codes, residency,
                             beta_plus=plus, beta_minus=minus,
                             functions={shaders.CONTRACT_OFF: entry})
    residency.sync_in()
    stepping.step_B(reference_fields, reference_pml)
    plan.run()
    residency.sync_out()
    assert plan.launches == 1, (name, "the mutant did not run")

    differing = sum(_differing(getattr(fields, n), getattr(reference_fields, n))
                    for n in ("Bx", "By", "Bz", "fu_Bx", "fu_By", "fu_Bz"))
    moved = sum(_moved(np.zeros_like(getattr(reference_fields, n)),
                       getattr(reference_fields, n))
                for n in ("Bx", "By", "Bz"))
    assert moved > 0, (name, "VACUOUS: the reference wrote nothing")
    if verdict == "must_catch":
        assert differing > 0, (name, "MUTATION NOT CAUGHT")
    else:
        assert differing == 0, (name, verdict, "expected null, measured", differing)


@requires_mps
def test_the_refuted_negation_spelling_is_unobservable_here_for_a_stated_reason():
    """WHY ONE REFUTED SPELLING IS A NULL ON THIS FAMILY, PROVED IN BOTH HALVES.

    The pass's ladder subtracts the increment, and ``v + (0.0f - x)``-style
    respellings of a negation differ from ``v - x`` in EXACTLY ONE input pattern:
    ``v`` a NEGATIVE zero and ``x`` a positive zero. (``-0.0 - +0.0`` is
    ``-0.0``; ``0.0f - +0.0`` is ``+0.0`` and ``-0.0 + +0.0`` is ``+0.0``. Every
    other zero pairing agrees, and no normal input separates them at all.)

    HALF ONE — THE SPELLING IS GENUINELY DIFFERENT ON THIS BACKEND. Measured here on
    that exact pattern: the two kernels disagree. So the mutation is not a no-op and
    Triton's ``* -1.0`` workaround is not needed here for the reason Triton needs it.

    HALF TWO — THIS FAMILY CANNOT REACH THE PATTERN. ``grid.beta`` is legal ONLY on
    an effective-2-D grid (``Grid._resolve_beta``; MEEP fields.cpp:546-547), whose
    invariant axis stores ONE cell, so the z-shifted operand IS the centre operand
    and ``b - b_z`` is ``+0.0`` for every finite ``b`` — negative zeros included,
    since ``-0.0 - -0.0`` is ``+0.0``. A float sum with a ``+0.0`` addend is never
    ``-0.0``, so ``curl0`` and ``curl1`` cannot BE a negative zero on any grid this
    family is admitted on, and the discriminating pattern is unreachable.

    Recording it this way — rather than as a "declared equivalence" — is the
    difference between "the spellings are the same" (false) and "this family cannot
    tell them apart, and here is why" (measured, and true).
    """
    import torch

    body = """
#include <metal_stdlib>
using namespace metal;

#pragma clang fp contract(off)

kernel void tail(device float* out [[buffer(0)]],
                 device const float* curl [[buffer(1)]],
                 device const float* term [[buffer(2)]],
                 constant uint& n [[buffer(3)]],
                 uint idx [[thread_position_in_grid]])
{
    if (idx >= n) { return; }
    __BODY__
}
"""
    shipped = body.replace("__BODY__", "out[idx] = curl[idx] - term[idx];")
    refuted = body.replace("__BODY__",
                           "out[idx] = curl[idx] + (0.0f - term[idx]);")
    # The one discriminating pattern, and three controls that must agree.
    curl = np.array([-0.0, +0.0, -0.0, +0.0], dtype=np.float32)
    term = np.array([+0.0, -0.0, -0.0, +0.0], dtype=np.float32)
    results = []
    for source in (shipped, refuted):
        library = device.compile_source(source)
        sentinel = np.full(curl.size, np.float32(-7.5), dtype=np.float32)
        out = torch.from_numpy(sentinel.copy()).to("mps")
        library.tail(out, torch.from_numpy(curl.copy()).to("mps"),
                     torch.from_numpy(term.copy()).to("mps"), int(curl.size))
        torch.mps.synchronize()
        results.append(out.cpu().numpy())
        assert _moved(sentinel, results[-1]) == curl.size, "VACUOUS: nothing written"
    assert _differing(results[0], results[1]) == 1, (
        "the refuted spelling must differ on EXACTLY the (-0.0, +0.0) lane; if it "
        "does not, this backend canonicalizes zeros and the whole negation "
        "discussion changes", results)

    # HALF TWO: on this family's grids the curl operand cannot be a negative zero,
    # because the invariant axis stores one cell and `b - b_z` is therefore +0.0.
    fields, _ = matrix.folded(beta=0.3)
    assert fields.grid.shape[2] == 1, fields.grid.shape
    assert int(getattr(fields.grid, "dimensions", 0)) == 2


@requires_mps
def test_the_shipped_source_through_the_same_route_is_the_mutation_legs_control():
    """THE CONTROL FOR THE MUTATION LEG, without which a null proves nothing.

    Every mutant is launched through ``plan_folded_beta_pml_curl_from_arrays``. If
    that route itself diverged from the array path, a ``must_catch=False``
    mutation's failure would be the harness and a ``must_catch=True`` one's success
    would be the harness too. This runs the UNMUTATED source through the identical
    route and requires 0.
    """
    fields, pml = matrix.folded(beta=0.3)
    reference_fields, reference_pml = matrix.folded(beta=0.3)
    grid = fields.grid
    codes, _ = symmetry.folded_axis_kinds(grid, pml)
    residency = device.Residency()
    plus, minus = special_kz.beta_curl_coefficients(
        grid.beta, grid.dt, magnetic=True, complex_storage=False)

    plan = _from_arrays_plan(fields, pml, "step_B", codes, residency,
                             beta_plus=plus, beta_minus=minus)
    residency.sync_in()
    stepping.step_B(reference_fields, reference_pml)
    plan.run()
    residency.sync_out()

    differing = sum(_differing(getattr(fields, n), getattr(reference_fields, n))
                    for n in ("Bx", "By", "Bz", "fu_Bx", "fu_By", "fu_Bz"))
    moved = sum(_moved(np.zeros_like(getattr(reference_fields, n)),
                       getattr(reference_fields, n))
                for n in ("Bx", "By", "Bz"))
    assert moved > 0, "VACUOUS: the reference wrote nothing"
    assert differing == 0, differing


@requires_mps
@pytest.mark.parametrize("sub_step", ("step_B", "step_D"))
def test_the_beta_free_arm_reduces_to_the_certified_folded_curl_on_the_device(
        sub_step):
    """THE IDENTITY LEG: ``has_beta=False`` must be the FOLD, exactly.

    It is what says the fold half of the composition is unchanged — the ONE claim a
    source diff against ``special_kz`` cannot make, because that parent carries no
    fold at all. The reference is a beta = 0 folded run, which the array path steps
    through the plain folded path.
    """
    fields, pml = matrix.folded()
    reference_fields, reference_pml = matrix.folded()
    codes, _ = symmetry.folded_axis_kinds(fields.grid, pml)
    residency = device.Residency()

    plan = _from_arrays_plan(fields, pml, sub_step, codes, residency,
                             has_beta=False)
    residency.sync_in()
    (stepping.step_B if sub_step == "step_B" else stepping.step_D)(
        reference_fields, reference_pml)
    plan.run()
    residency.sync_out()

    targets = (("Bx", "By", "Bz") if sub_step == "step_B"
               else ("Dx", "Dy", "Dz"))
    names = targets + tuple("fu_" + t for t in targets)
    differing = sum(_differing(getattr(fields, n), getattr(reference_fields, n))
                    for n in names)
    moved = sum(_moved(np.zeros_like(getattr(reference_fields, n)),
                       getattr(reference_fields, n)) for n in targets)
    assert moved > 0, (sub_step, "VACUOUS: the reference wrote nothing")
    assert differing == 0, (sub_step, differing)


def _one_rank_record(fields, pml):
    """The record of the one-rank division ``stepping`` steps these ``Fields`` with."""
    from meep_gpu.chunk_map import ChunkLayout, chunk_bounds_record

    return chunk_bounds_record(ChunkLayout.one_rank(fields.grid, pml))
