"""The three below-the-cut fused magnetic welds, as claims a laptop can check.

WHAT THIS SUITE OWNS AND WHAT IT DELIBERATELY DOES NOT. The BYTES are the device
gate's — ``parity/meep_gpu/gate_metal_below_the_cut_fused_pairs.py`` steps two
engines side by side for twelve complete steps, per product, and compares uint32
words. No assertion here duplicates that. What lives here is everything true about
these three families WITHOUT a device:

* that each weld's emitted source is the SHIPPED weld's construction with exactly
  one emitter swapped — and, for the nonlinear one, with nothing swapped at all;
* that neither the beta nor the BFAST weld can silently degenerate into the ordinary
  product, which is the one defect every byte leg would pass;
* that all three arms are registered UNWIRED, and WHY that is not bookkeeping here:
  each would contend on ``step_B`` with a WIRED sub-step arm that admits exactly the
  same configurations, so wiring one would leave the slot unselected and take the
  certified curl off the device too;
* that the three predicates are mutually disjoint and disjoint from the shipped
  weld's, in both directions;
* that the binding counts each module spells are the counts its emitter produces —
  and for BFAST, that the count is EXACTLY the platform ceiling with zero margin.

THE CORPUS DEMAND, so the suite states what these are worth rather than leaving it
to a docstring: 2 + 1 + 1 = 4 reachable B->H seam-instances of 387, measured at
``parity/meep_gpu/results/below_the_cut_census_2026-08-20/census.json``. Every one of
the three cells' D->E partners is worth ZERO — each row injects electrically inside
that seam — and none is built.

AND THE CELL THAT WAS NOT BUILT, with its reason as a check rather than a note. The
tranche's fourth candidate — ``B_to_H (folded beta complex, folded beta complex)``,
worth 2 — was first written down as "declined on cost". It is declined on something
sharper and testable: both reachable rows are MIRROR_PERIODIC folds, so
``fill_folded_far_ghosts_B`` (driver.py:3287) runs INSIDE the seam and ``update_H``
reads what it writes, and no fused product on either backend carries that pass. The
consequence is measured word for word in
``parity/meep_gpu/results/metal_folded_beta_complex_seam_2026-08-20/``; the premise
and the census's own new carry column are pinned here.
"""

from __future__ import annotations

import os
import pathlib

import pytest

# THE POLICY THIS MODULE ASKS ITS PREDICATE QUESTIONS UNDER, declared at IMPORT and
# not per test. Measured 2026-08-20: without it, seven tests here failed when the
# module ran ALONE and passed in a full ``pytest meep_gpu/`` run — because collection
# imports about twenty Metal parity modules that ``setdefault`` this variable, and
# ``conftest._no_leaked_subnormal_policy`` then re-establishes whatever collection
# left behind (conftest.py:157-160, :211-280). A module whose green depends on a
# SIBLING's import order is green by accident. The MPS executor cannot honour
# ``keep`` — Metal flushes denormals natively and exposes no lever — so every Metal
# predicate asked under it refuses and the suite measures nothing; ``flush`` is the
# only policy under which these welds' coverage questions have an answer.
# ``setdefault``, so an explicit export from a campaign runner still wins.
os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")

PACKAGE_DIR = pathlib.Path(__file__).resolve().parent
REPO = PACKAGE_DIR.parent
GATE = REPO / "parity" / "meep_gpu" / "gate_metal_below_the_cut_fused_pairs.py"
CENSUS = (REPO / "parity" / "meep_gpu" / "results"
          / "below_the_cut_census_2026-08-20" / "census.json")

from meep_gpu.metal_kernels import (  # noqa: E402
    arms, beta_fused_magnetic_pair as beta, bfast_curl,
    bfast_fused_magnetic_pair as bfast, fused_magnetic_pair as shipped,
    nonlinear_fused_magnetic_pair as nonlinear, nonlinear_update_e, registry,
    shaders, special_kz,
)
from meep_gpu.metal_kernels.device import MAX_BUFFER_BINDINGS  # noqa: E402

#: The three welds, and for each: its module, the emitter its curl half lifts, and
#: the reachable B->H seam-instances the corpus census measured for its cell.
WELDS = (
    ("nonlinear", nonlinear, None, 2),
    ("beta", beta, lambda codes: special_kz.beta_curl_source(codes, False, True), 1),
    ("bfast", bfast,
     lambda codes: bfast_curl.bfast_curl_source(codes, False,
                                                shaders.CONTRACT_OFF, True), 1),
)

CODES = (1, 1, 1)
WALLS = (True, True, True)


def source_of(module, codes=CODES, walls=WALLS, contract=shaders.CONTRACT_OFF):
    """One weld's emitted source, by the naming convention all three follow."""
    prefix = module.__name__.rsplit(".", 1)[-1].replace("_fused_magnetic_pair", "")
    return getattr(module, f"{prefix}_fused_magnetic_pair_source")(
        codes, walls, contract)


# ---------------------------------------------------------------------------
# 1. Each weld is the shipped construction with ONE emitter swapped
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,module,emitter,_reach", WELDS)
@pytest.mark.parametrize("codes", ((0, 0, 0), (1, 0, 1), (1, 1, 1)))
def test_the_curl_half_is_its_own_emitters_bytes_on_both_sides_of_the_splice(
        name, module, emitter, _reach, codes):
    """The lifted curl body appears VERBATIM, on both sides of the wall clear.

    None of these families retypes a half: each lifts the curl from its own
    certified emitter and the constitutive from ``shaders.constitutive_source``.
    That makes the transcription rule a substring check rather than a promise.
    """
    walls = (True, False, True)
    source = source_of(module, codes, walls)
    body = (shipped.certified_curl_body(codes) if name == "nonlinear"
            else getattr(module, f"certified_{name}_curl_body")(codes))
    head, tail = body.split(shipped._CURL_STORE, 1)
    assert head in source
    assert (shipped._CURL_STORE + tail) in source


#: The two welds that lift a curl of their OWN. The nonlinear weld is deliberately
#: absent: it lifts nothing, because its emitted bytes ARE the shipped weld's, and
#: ``test_the_nonlinear_weld_emits_the_shipped_welds_bytes`` is the stronger claim
#: for it. Spelled as a separate table rather than skipped inside the shared one —
#: a skip would read as a coverage gap where this is a different question.
SWAPPED = tuple(row for row in WELDS if row[2] is not None)


@pytest.mark.parametrize("name,module,emitter,_reach", SWAPPED)
def test_the_lifted_body_is_its_emitters_own_output(name, module, emitter, _reach):
    """The lift is a SLICE of the emitter's string, never a retyping.

    Asserted as containment rather than equality because the lift drops the
    ``#include``/signature preamble and the closing brace — and nothing else, which
    is what the containment measures.
    """
    body = getattr(module, f"certified_{name}_curl_body")(CODES)
    assert body in emitter(CODES)
    assert body.strip()


@pytest.mark.parametrize("name,module,emitter,_reach", WELDS)
def test_the_constitutive_half_differs_only_in_the_three_seam_lines(
        name, module, emitter, _reach):
    """The lift edits the SOURCE READ and nothing else, on all three welds.

    The certified H body opens each component with ``float srcN = gN[ii];`` — a
    reload of the flux density the curl just stored. Fused, that is the register.
    Any OTHER differing line would mean a rename reached the arithmetic.
    """
    certified = shipped.certified_constitutive_body().splitlines()
    source = source_of(module)
    marker = "    // --- update_H (stepping.update_H"
    spliced = [line for line in source.split(marker, 1)[1].splitlines()
               if "// THE SEAM:" not in line]
    spliced = spliced[1:]
    while spliced and spliced[-1].strip() in ("", "}"):
        spliced.pop()
    assert len(spliced) == len(certified)
    changed = [(a, b) for a, b in zip(certified, spliced) if a != b]
    assert changed == [(f"    float src{t} = g{t}[ii];", f"    float src{t} = v{t};")
                       for t in range(3)]


@pytest.mark.parametrize("name,module,emitter,_reach", WELDS)
def test_the_curl_half_is_the_forward_direction(name, module, emitter, _reach):
    """``step_B`` differences UP. ``step_D``'s negated strides are another product."""
    source = source_of(module)
    assert "int si = i + 1, sj = j + 1, sk = k + 1;" in source
    assert "int si = i - 1, sj = j - 1, sk = k - 1;" not in source


@pytest.mark.parametrize("name,module,emitter,_reach", WELDS)
def test_the_h_side_carries_no_inverse_epsilon(name, module, emitter, _reach):
    """mu = 1 is baked into the array path too (``update_H`` passes ``fields.Bx``).

    The certified constitutive kernel keeps three inverse-epsilon pointers on BOTH
    sides so its buffer layout is one layout; every one of these welds serves the H
    side only and DROPS them.
    """
    source = source_of(module)
    assert "device const float* e0" not in source


@pytest.mark.parametrize("name,module,emitter,_reach", WELDS)
def test_a_boundary_triple_that_is_not_a_triple_is_refused(
        name, module, emitter, _reach):
    prefix = name
    emit = getattr(module, f"{prefix}_fused_magnetic_pair_source")
    with pytest.raises(ValueError):
        emit((1, 1), WALLS)
    with pytest.raises(ValueError):
        emit(CODES, (True, True))


# ---------------------------------------------------------------------------
# 2. THE DEGENERACY TRAP — the one defect every byte leg would pass
# ---------------------------------------------------------------------------

def test_the_nonlinear_weld_emits_the_shipped_welds_bytes():
    """This family's whole claim: on a nonlinear run the B/H seam IS the ordinary one.

    Byte equality rather than "close enough". If the two strings ever differed, the
    nonlinear weld would be a second, uncertified kernel wearing an admission port's
    name — and its own module says it forwards, so the forward is checked.
    """
    for codes in ((0, 0, 0), (1, 0, 1), (1, 1, 1)):
        for walls in ((False, False, False), (True, False, True), WALLS):
            assert (nonlinear.nonlinear_fused_magnetic_pair_source(codes, walls)
                    == shipped.fused_magnetic_pair_source(codes, walls))


@pytest.mark.parametrize("name,module", (("beta", beta), ("bfast", bfast)))
def test_the_swapped_weld_can_never_degenerate_into_the_ordinary_one(name, module):
    """THE TRAP THIS TEST EXISTS FOR, and it is not hypothetical.

    Both swapped emitters carry a ``has_beta`` / ``has_bfast`` arm whose body IS the
    certified ordinary curl — that arm exists so a zero-beta or zero-k run is named
    rather than silently taken. A lift that reached for it would produce a weld that
    is byte-identical to the array path on EVERY case a gate can build, because
    ``beta = 0`` and ``k = 0`` ARE the ordinary arithmetic. So the emitted source
    must DIFFER from the shipped ordinary weld's, and must carry the inserted term.
    """
    source = source_of(module)
    assert source != shipped.fused_magnetic_pair_source(CODES, WALLS)
    inserted = {"beta": ("curl0 = curl0 - (beta_plus * b);",
                         "curl1 = curl1 - (beta_minus * a);"),
                "bfast": ("float st0 = s0[ii];", "s0[ii] = st0 + adv0;",
                          "curl0 = curl0 - adv0;")}[name]
    for statement in inserted:
        assert statement in source


@pytest.mark.parametrize("name,module", (("beta", beta), ("bfast", bfast)))
def test_the_emitter_refuses_a_lift_that_lost_its_inserted_term(name, module,
                                                                monkeypatch):
    """The emit-time assertion is armed, measured by DISARMING the emitter.

    Each module asserts its own inserted statements are present before splicing. A
    guard nothing tests is a comment, so this replaces the certified body with the
    ORDINARY one — exactly what a ``has_beta=False`` lift would return — and requires
    the emitter to raise rather than ship the ordinary kernel under this name.
    """
    monkeypatch.setattr(module, f"certified_{name}_curl_body",
                        lambda codes, contract=shaders.CONTRACT_OFF:
                        shipped.certified_curl_body(codes, contract))
    with pytest.raises(AssertionError, match="ordinary product under another name"):
        source_of(module)


# ---------------------------------------------------------------------------
# 3. The wall clear is carried, and it is the shipped weld's own function
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,module,emitter,_reach", WELDS)
def test_the_wall_clear_is_the_shipped_welds_diagonal_table(
        name, module, emitter, _reach):
    """One wall clears exactly ONE B component, the one sharing the wall's axis.

    ``IYEE_SHIFTS`` (fields.py:216-217) gives Bx (0,1,1), By (1,0,1), Bz (1,1,0).
    All three welds call ``fused_magnetic_pair.zero_metal_mask`` rather than
    spelling a table, so this asserts the IMPORT is load-bearing: the emitted line
    for a single wall must be the diagonal one.
    """
    for axis, (flag, target) in enumerate((("at_x", "v0"), ("at_y", "v1"),
                                           ("at_z", "v2"))):
        walls = tuple(index == axis for index in range(3))
        source = source_of(module, CODES, walls)
        lines = [line.strip() for line in source.splitlines()
                 if "? 0.0f : v" in line and "curl" not in line]
        assert lines == [f"{target} = {flag} ? 0.0f : {target};"], (name, axis, lines)


@pytest.mark.parametrize("name,module,emitter,_reach", WELDS)
def test_a_run_with_no_wall_emits_no_clear_at_all(name, module, emitter, _reach):
    """``_zero_metal`` returns before touching a plane when nothing is metallic
    (stepping.py:2280-2286), so a select that is always false would be correct
    arithmetic and a DIFFERENT kernel."""
    source = source_of(module, (0, 0, 0), (False, False, False))
    assert "? 0.0f : v0" not in source
    assert "no walled axis clears a B component" in source


@pytest.mark.parametrize("name,module,emitter,_reach", WELDS)
def test_the_auxiliary_is_never_masked(name, module, emitter, _reach):
    """``zero_metal_B`` passes ``B_COMPONENTS`` (stepping.py:2250); ``fu_B`` is not
    in it, so masking the split-field auxiliary would be an over-carry."""
    source = source_of(module)
    assert "    u0[ii] = n0; u1[ii] = n1; u2[ii] = n2;" in source
    assert "u0[ii] = at_x" not in source


# ---------------------------------------------------------------------------
# 4. The binding counts — and BFAST's zero margin
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,module,emitter,_reach", WELDS)
def test_the_packed_signature_is_the_count_the_module_spells(
        name, module, emitter, _reach):
    """The emitter's highest buffer attribute must equal ``PACKED_BINDINGS``.

    Whether it COMPILES is the gate's ``binding_ceiling`` leg; what belongs here is
    that the number this module argues from is the number its own emitter produces.
    """
    source = source_of(module)
    highest = max(int(chunk.split(")")[0])
                  for chunk in source.split("[[buffer(")[1:])
    assert highest + 1 == module.PACKED_BINDINGS
    assert module.PACKED_BINDINGS <= MAX_BUFFER_BINDINGS


def test_the_bfast_weld_sits_exactly_on_the_ceiling_with_no_margin():
    """THE NUMBER THAT DECIDES WHETHER THIS CELL IS BUILDABLE AT ALL.

    30 pointers plus one packed ``constant Params&`` is 31 bindings and
    ``MAX_BUFFER_BINDINGS`` is 31 — attribute indices run 0..30 and a 32nd is a
    compile error. One more pointer and this cell would be UNFUSABLE ON METAL rather
    than merely tight, which is the verdict the fusion matrix hands the D/E
    off-diagonal cells. Pinned so a later edit that adds a buffer fails HERE with the
    reason, rather than on a device with a compiler diagnostic.
    """
    assert bfast.PACKED_BINDINGS == MAX_BUFFER_BINDINGS == 31
    source = source_of(bfast)
    pointers = source.count("[[buffer(") - source.count("constant Params&")
    assert pointers == 30
    assert shipped.PACKED_BINDINGS == beta.PACKED_BINDINGS == 28


@pytest.mark.parametrize("name,module,emitter,_reach", WELDS)
def test_the_refuted_separate_scalar_signature_really_does_break_the_ceiling(
        name, module, emitter, _reach):
    """The separate-scalar count is spelled once and the emitter agrees with it.

    The nonlinear family owns no emitter of its own — its kernel IS the shipped
    weld's — so its declared width must be the shipped weld's width, and that
    equality is what makes the gate's fallback to the shipped refuted signature the
    same measurement rather than a different one.
    """
    emit = getattr(module, "refuted_separate_scalar_source", None)
    if emit is None:
        assert module.SEPARATE_SCALAR_BINDINGS == shipped.SEPARATE_SCALAR_BINDINGS
        emit = shipped.refuted_separate_scalar_source
    source = emit()
    highest = max(int(chunk.split(")")[0])
                  for chunk in source.split("[[buffer(")[1:])
    assert highest + 1 == module.SEPARATE_SCALAR_BINDINGS
    assert module.SEPARATE_SCALAR_BINDINGS > MAX_BUFFER_BINDINGS


@pytest.mark.parametrize("name,module,emitter,_reach", WELDS)
def test_the_contraction_guard_is_emitted_and_selects_a_different_kernel(
        name, module, emitter, _reach):
    """The guard is a property of the SOURCE on this backend, so each mode is a
    different compiled kernel."""
    off = source_of(module, contract=shaders.CONTRACT_OFF)
    fast = source_of(module, contract=shaders.CONTRACT_FAST)
    assert "contract(off)" in off
    assert off != fast


# ---------------------------------------------------------------------------
# 5. The arm table — unwired, and why that is load-bearing here
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,module,emitter,_reach", WELDS)
def test_the_arm_is_registered_unwired_on_step_B(name, module, emitter, _reach):
    """Registered so the disjointness sweep can SEE it; unwired so ``plan_step``
    cannot select it."""
    arms.ensure_registered()
    rows = [spec for spec in arms.registered() if spec.family == module.FAMILY]
    assert len(rows) == 1, rows
    assert rows[0].slot == "step_B"
    assert rows[0].wired is False


@pytest.mark.parametrize("name,module,emitter,_reach", WELDS)
def test_the_unwired_arm_is_invisible_to_the_composer(name, module, emitter, _reach):
    """WHY UNWIRED IS NOT BOOKKEEPING FOR THIS TRANCHE.

    Each of these welds inherits a half that is WIRED — the nonlinear spine curl, the
    ``special_kz`` real-beta curl, the BFAST curl — and admits exactly the
    configurations that arm admits. Wired, ``_select_slot`` would see two admitters
    on ``step_B`` and leave the slot UNSELECTED, which does not merely decline the
    fusion: it takes the CERTIFIED CURL off the device as well. So the flag is what
    keeps this tranche additive, and ``arms_for`` must not return it.
    """
    import numpy as np

    from meep_gpu.metal_kernels.device import Residency

    arms.ensure_registered()
    pair = _fixtures()[name]
    context = arms.StepContext(pair[0], pair[1], Residency(),
                               (shaders.CONTRACT_OFF,), ())
    spec = next(s for s in arms.registered() if s.family == module.FAMILY)
    # THE LABEL MUST EXIST IN THE TABLE, or the membership test below is filtering a
    # name nothing carries and would pass for the wrong reason.
    assert spec.label in {s.label for s in arms.registered()}
    bound = arms.arms_for("step_B", context)
    assert spec.label not in {arm.label for arm in bound}
    # AND THE CONTENDER IT WOULD DISPLACE IS REALLY THERE: exactly one WIRED arm
    # admits this configuration on step_B today, which is the certified curl this
    # weld would take off the device if it were wired. ``_arm_verdict`` is the
    # composer's OWN verdict function — a predicate that raises is a refusal there,
    # and asking it any other way would give this test a second opinion about what
    # admission means.
    from meep_gpu.triton_kernels.launch import _arm_verdict

    admitting = [arm.label for arm in bound
                 if arm.gate and _arm_verdict(arm).covered]
    assert len(admitting) == 1, admitting
    assert np is not None


@pytest.mark.parametrize("name,module,emitter,_reach", WELDS)
def test_the_family_is_named_in_the_registry_module(name, module, emitter, _reach):
    """A family in the tree but not in ``FAMILY_MODULES`` is invisible to
    ``plan_step``, which is a silent coverage loss rather than an error."""
    stem = module.__name__.rsplit(".", 1)[-1]
    assert stem in registry.FAMILY_MODULES


# ---------------------------------------------------------------------------
# 6. Disjointness — asked in both directions, on real configurations
# ---------------------------------------------------------------------------

def _fixtures():
    """One configuration per run kind, plus a plain one. Built from the engine's own
    objects so the predicates answer about a real grid rather than a stub."""
    import numpy as np
    from meep_gpu.fields import Fields
    from meep_gpu.grid import Grid
    from meep_gpu.pml import PML

    def cart(**kwargs):
        grid = Grid(resolution=10.0, cell_size=(2.0, 2.1, 1.2), courant=0.35,
                    xp=np, **kwargs)
        fields = Fields(grid=grid)
        fields.enable_pml_storage()
        return fields, PML(grid=grid, thickness=2)

    def flat(**kwargs):
        grid = Grid(resolution=10.0, cell_size=(2.0, 2.1, 0.0), dimensions=2,
                    courant=0.35, xp=np, **kwargs)
        fields = Fields(grid=grid)
        fields.enable_pml_storage()
        return fields, PML(grid=grid, thickness=((2, 2), (2, 2), (0, 0)))

    plain = cart()
    nonlinear_pair = cart()
    shape = tuple(nonlinear_pair[0].grid.shape)
    nonlinear_pair[0].set_nonlinear_volumes(
        {"Ez": np.full(shape, np.float32(0.1))},
        {"Ez": np.full(shape, np.float32(0.2))})
    return {
        "plain": plain,
        "nonlinear": nonlinear_pair,
        "beta": flat(beta=0.33),
        "bfast": cart(bfast_scaled_k=(0.2, 0.0, 0.0)),
    }


def _coverage(module, pair):
    from meep_gpu.metal_kernels.device import Residency

    prefix = module.__name__.rsplit(".", 1)[-1].replace("_fused_magnetic_pair", "")
    call = getattr(module, f"metal_{prefix}_fused_magnetic_pair_coverage")
    return call(pair[0], pair[1], (), Residency())


def test_the_four_welds_are_mutually_disjoint_on_the_four_run_kinds():
    """A 4x4 admission table with exactly the diagonal set.

    Three welds on ONE slot that co-admitted anything would be a composition hazard
    rather than a coverage gain, and "they are disjoint" is the kind of claim that is
    true when written and false two tranches later. So it is a table, and the whole
    table is asserted rather than the cells anyone happened to think of.
    """
    fixtures = _fixtures()
    welds = {"plain": shipped, "nonlinear": nonlinear, "beta": beta, "bfast": bfast}
    table = {}
    for weld_name, module in welds.items():
        for run_kind, pair in fixtures.items():
            module_ = shipped if weld_name == "plain" else module
            if weld_name == "plain":
                from meep_gpu.metal_kernels.device import Residency
                verdict = shipped.metal_fused_magnetic_pair_coverage(
                    pair[0], pair[1], (), Residency()).covered
            else:
                verdict = _coverage(module_, pair).covered
            table[(weld_name, run_kind)] = bool(verdict)
    admitted = {key for key, value in table.items() if value}
    assert admitted == {("plain", "plain"), ("nonlinear", "nonlinear"),
                        ("beta", "beta"), ("bfast", "bfast")}, sorted(admitted)


@pytest.mark.parametrize("name,module,emitter,_reach", WELDS)
def test_the_weld_is_never_wider_than_the_two_halves_it_inherits(
        name, module, emitter, _reach):
    """A weld is ``half AND half AND seam clauses``; it may only be NARROWER.

    The device gate measures this on the cases it steps. Here it is measured on the
    case that matters most and that the gate cannot reach — the run kind the weld is
    FOR — by calling the two shipped sub-step predicates directly. A weld wider than
    a half it claims to inherit would be serving rows the certified sub-step refuses,
    and no byte comparison would show it, because the harness only ever builds
    configurations the weld admits.
    """
    from meep_gpu.metal_kernels.device import Residency

    pair = _fixtures()[name]
    residency = Residency()
    halves = {
        "nonlinear": (
            lambda: nonlinear_update_e.nonlinear_run_pml_curl_coverage(
                pair[0], pair[1], "step_B", residency),
            lambda: nonlinear_update_e.nonlinear_run_constitutive_coverage(
                pair[0], pair[1], "H", residency)),
        "beta": (
            lambda: special_kz.beta_pml_curl_coverage(
                pair[0], pair[1], "step_B", residency),
            lambda: special_kz.beta_run_constitutive_coverage(
                pair[0], pair[1], "H", residency)),
        "bfast": (
            lambda: bfast_curl.bfast_pml_curl_coverage(
                pair[0], pair[1], "step_B", residency),
            lambda: bfast_curl.bfast_run_constitutive_coverage(
                pair[0], pair[1], "H", residency)),
    }[name]
    weld = _coverage(module, pair)
    curl, constitutive = (half().covered for half in halves)
    assert weld.covered, weld.reasons
    assert curl and constitutive


@pytest.mark.parametrize("name,module,emitter,_reach", WELDS)
def test_an_undeclared_source_list_is_refused(name, module, emitter, _reach):
    """IGNORANCE IS NEVER AN EMPTY SET. ``Fields`` does not hold the source list, so
    a predicate that inferred "no sources" from not being told would be exactly the
    over-covering refusal the magnetic-seam clause exists to prevent."""
    from meep_gpu.metal_kernels.device import Residency

    pair = _fixtures()[name]
    prefix = name
    call = getattr(module, f"metal_{prefix}_fused_magnetic_pair_coverage")
    verdict = call(pair[0], pair[1], None, Residency())
    assert not verdict.covered
    assert any("was not declared" in reason for reason in verdict.reasons)


# ---------------------------------------------------------------------------
# 7. The census this tranche was built against, and the gate that certified it
# ---------------------------------------------------------------------------

def test_the_corpus_census_is_present_and_says_what_these_cells_are_worth():
    """The reachable demand is READ from the census artifact, never transcribed.

    Four seam-instances of the 387 priced on the 2026-08-20 census is a small number
    and this suite says so out loud: if the census ever measures something else, this
    fails rather than letting a stale figure survive in three docstrings.
    """
    import json

    assert CENSUS.is_file(), (
        f"{CENSUS} is missing; the demand these three welds were built against is "
        f"not on disk and every count in their docstrings is unbacked")
    record = json.loads(CENSUS.read_text())
    cells = {(c["seam"], c["curl_arm"], c["constitutive_arm"]): c
             for c in record["cells"]}
    expected = {
        ("B_to_H", "nonlinear PML curl", "nonlinear PML magnetic"): 2,
        ("B_to_H", "special_kz real beta", "special_kz real beta"): 1,
        ("B_to_H", "BFAST", "BFAST"): 1,
    }
    for key, reach in expected.items():
        assert cells[key]["reachable"] == reach, (key, cells[key])
    # AND THE PARTNERS ARE WORTH ZERO. Building a D->E weld for any of these three
    # would buy nothing: every row injects electrically inside that seam.
    for curl, const in (("nonlinear PML curl", "nonlinear"),
                        ("special_kz real beta", "special_kz real beta"),
                        ("BFAST", "BFAST")):
        assert cells[("D_to_E", curl, const)]["reachable"] == 0


def test_the_declined_cells_are_declined_on_a_measured_zero_or_a_measured_cost():
    """The two cells this tranche did NOT build, with the measurement behind each.

    ``folded beta real`` is worth ZERO — its single row declares two magnetic sources
    — which is the census's answer and not a judgement call, and this pins it so a
    later round does not spend a kernel on it. ``folded beta complex`` is worth 2;
    the demand is pinned here and the REASON it is declined is pinned in the two
    tests below, which replaced the original "declined on cost" note with something
    a later round can check.
    """
    import json

    record = json.loads(CENSUS.read_text())
    cells = {(c["seam"], c["curl_arm"], c["constitutive_arm"]): c
             for c in record["cells"]}
    folded_real = cells[("B_to_H", "folded beta real", "folded beta real")]
    assert folded_real["rows"] == 1 and folded_real["reachable"] == 0
    assert any("B" in row["source_field_types"]
               for row in folded_real["row_detail"])
    folded_complex = cells[("B_to_H", "folded beta complex", "folded beta complex")]
    assert folded_complex["rows"] == 3 and folded_complex["reachable"] == 2


def test_the_folded_beta_complex_cell_runs_a_pass_no_product_carries():
    """WHY that cell is declined, from the census block rather than from prose.

    Both reachable rows are MIRROR_PERIODIC folds — mirrored on an axis that is NOT
    declared metallic — and ``stepping._fill_folded_far_ghosts`` selects exactly
    those axes (stepping.py:1567, :1503-1518, through ``Grid.stored_cells``' extra
    slot). So ``fill_folded_far_ghosts_B`` (driver.py:3287) runs INSIDE the B->H
    seam on both of them, and ``update_H`` reads what it writes.

    That is the difference between this cell and the three this tranche built: each
    of those is one certified emitter swapped into a shipped weld, while this one
    would have to DERIVE a second ownership restructure for a pass no kernel on
    either backend carries. The device probe measures the consequence word for word
    (``parity/meep_gpu/results/metal_folded_beta_complex_seam_2026-08-20/``); this
    checks the premise on a laptop.
    """
    import json

    record = json.loads(CENSUS.read_text())
    cells = {(c["seam"], c["curl_arm"], c["constitutive_arm"]): c
             for c in record["cells"]}
    cell = cells[("B_to_H", "folded beta complex", "folded beta complex")]
    reachable = [row for row in cell["row_detail"]
                 if row["clears_the_source_seam"]]
    assert len(reachable) == 2, reachable
    for row in reachable:
        folded = [axis for axis in range(3) if row["mirrored"][axis]]
        assert folded, row
        # MIRROR_PERIODIC on every folded axis: mirrored and NOT metallic is exactly
        # the split `symmetry.folded_axis_kinds` resolves and `_stored_past_owned`
        # measures (triton_kernels/symmetry.py:465-473).
        assert all(not row["metallic"][axis] for axis in folded), row
    # THE CLAUSE MUST DISCRIMINATE, or it is decoration. "mirrored and not metallic"
    # is narrower than "mirrored" ONLY if the corpus holds folds of both kinds, and
    # this file's census block covers twelve selected cells rather than the corpus —
    # so the check is made where the whole corpus is counted, in the carry cut's
    # per-row liveness column: the near fill (any mirrored axis) is live on strictly
    # more rows than the far fill (mirrored and not metallic). If those two counts
    # were equal, every folded row would be MIRROR_PERIODIC and the assertion above
    # would be measuring nothing.
    carry = (REPO / "parity" / "meep_gpu" / "results"
             / "fusion_matrix_metal_2026-08-20_carry" / "fusion_matrix.json")
    if carry.is_file():
        live = json.loads(carry.read_text())[
            "in_seam_passes_live_per_corpus_row"]["B_to_H"]
        assert live["fill_B"] > live["fill_folded_far_ghosts_B"] > 0, live


def test_the_carry_gap_is_asked_by_the_census_and_not_only_by_a_docstring():
    """The fusion matrix now ASKS whether a cell's rows run an uncarried pass.

    A census that does not ask cannot report, and until the 2026-08-20 carry cut the
    ranking's only buildability signal was the binding bracket — so this cell was
    printed ``FITS ... NOT BUILT``, which reads as an invitation. This pins that the
    question is wired in, that its answer for the far fill is NOT CARRIED, and that
    the capability is sized: 16 reachable unserved seam-instances, 11 of them inside
    cells a SHIPPED product already covers and refuses.

    NAME-DRIFT FLOOR: the keys asserted below are required to be PRESENT, so an
    artifact that stopped carrying the column fails here instead of silently
    answering "no gap".
    """
    import json

    carry = (REPO / "parity" / "meep_gpu" / "results"
             / "fusion_matrix_metal_2026-08-20_carry" / "fusion_matrix.json")
    if not carry.is_file():
        pytest.skip(f"{carry} not present in this tree")
    record = json.loads(carry.read_text())

    carried = record["in_seam_passes_carried_by_some_product"]
    assert "fill_folded_far_ghosts_B" not in carried["B_to_H"], carried
    assert "fill_B" in carried["B_to_H"] and "zero_metal_B" in carried["B_to_H"], (
        "the near fill and the wall clear ARE carried; if they read as uncarried "
        "the liveness rule is broken and the far-fill answer means nothing")

    live = record["in_seam_passes_live_per_corpus_row"]["B_to_H"]
    assert live["fill_folded_far_ghosts_B"] > 0, (
        "the far fill is live on no corpus row, so 'no product carries it' would be "
        "a fact about the corpus rather than about the products")

    totals = record["carry_gap_totals"]
    assert totals["fill_folded_far_ghosts_B"] + totals["fill_folded_far_ghosts_D"] \
        == record["headline"]["reachable_unserved_running_an_uncarried_in_seam_pass"]
    buckets = record["headline"][
        "reachable_unserved_running_an_uncarried_in_seam_pass_by_bucket"]
    assert sum(buckets.values()) == sum(totals.values()), (buckets, totals)
    # THE ACTIONABLE HALF, asserted rather than described: most of the gap is inside
    # products that already exist, which is where the carry is cheapest to build.
    assert buckets["a product covers the pair and REFUSES"] > buckets.get("FITS", 0)

    # This cell is one of them, and the matrix says so on the cell's own row.
    gap = next(g for g in record["gaps_ranked_by_demand"]
               if (g["seam"], g["curl_arm"], g["constitutive_arm"])
               == ("B_to_H", "folded beta complex", "folded beta complex"))
    assert gap["in_seam_passes_no_product_carries"] == [
        "fill_folded_far_ghosts_B"], gap
    assert "NEEDS AN UNCARRIED IN-SEAM PASS" in gap["verdict"], gap["verdict"]

    # THE OTHER BOARD AGREES, through a ladder nobody wrote twice. The Triton cut
    # runs its own arm attribution over its own census and reaches the SAME B-side
    # count. Two derivations landing on one number is the only reason to believe
    # either; a disagreement means one of the two liveness walks is wrong.
    triton = (REPO / "parity" / "meep_gpu" / "results"
              / "fusion_matrix_triton_2026-08-20_carry" / "fusion_matrix.json")
    if triton.is_file():
        other = json.loads(triton.read_text())["aggregate"]
        assert other["carry_gap_totals"]["fill_folded_far_ghosts_B"] == totals[
            "fill_folded_far_ghosts_B"], (other["carry_gap_totals"], totals)
        # And it reports the D seam UNDETERMINED rather than "not carried": that cut
        # serves 0 instances there, so a two-state answer would have been an
        # inference from ignorance and would have inflated its own gap.
        assert other["seam_has_evidence_for_the_carry_question"] == {
            "B->H": True, "D->E": False}, other[
                "seam_has_evidence_for_the_carry_question"]
        assert "fill_folded_far_ghosts_D" not in other["carry_gap_totals"], other[
            "carry_gap_totals"]


def test_the_device_gate_is_present_and_routes_through_the_runner():
    assert GATE.is_file()
    text = GATE.read_text(encoding="utf-8")
    assert "run_current_measurement" in text, (
        "a Metal gate must run through metal_gate_runner so its artifact records "
        "the sources the PROCESS imported")
    assert "gate_provenance" in text


def test_the_gate_arms_a_defect_for_every_weld_and_scores_the_neutral_edit():
    """The gate's own shape, asserted rather than assumed.

    Two properties this suite can check without a GPU and that a hollow gate would
    fail: that the byte-neutral control exists (the one armed edit required to be
    UNCAUGHT, which is what makes "the fusion changes no arithmetic" a measurement),
    and that each swapped weld contributes mutations of its OWN — a shared-only
    mutation set would leave the beta term and the Tustin tail unmeasured.
    """
    text = GATE.read_text(encoding="utf-8")
    assert "byte_neutral_source" in text and "byte_neutral_control" in text
    for marker in ("beta_term_dropped_on_target_0", "beta_partners_swapped",
                   "bfast_curl_correction_dropped", "bfast_k_pair_swapped_on_target_0"):
        assert marker in text, marker
    # NAME-DRIFT GUARD: the gate's own product keys must be the ones this suite
    # names, or the markers above would be filtering a set that no longer exists.
    for name, _module, _emitter, _reach in WELDS:
        assert f'key="{name}"' in text, name
