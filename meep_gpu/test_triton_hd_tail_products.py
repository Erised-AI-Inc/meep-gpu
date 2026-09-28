"""Laptop contracts for the five 2026-09-07 Triton H->D TAIL products.

THE MERGE BAR RUNS WITHOUT TRITON, which is exactly why these contracts exist: every
one of them is a claim about HOST TEXT or a HOST predicate, and every one of them
would otherwise be checked for the first time on a device an hour away.

FIVE FAMILIES, ONE MODULE, and the shared table is what keeps a sixth from being added
with a silently weaker contract:

    conductive_bfast_fused_hd_pair   2 instances  2 cells  2 kernels
    beta_real_fused_hd_pair          2 instances  2 cells  2 kernels
    folded_complex_fused_hd_pair     5 instances  2 cells  1 kernel
    beta_complex_fused_hd_pair       4 instances  2 cells  2 kernels
    nonlinear_fused_hd_pair          2 instances  1 cell   0 kernels (admission only)

WHAT IS PINNED, and why each one is a defect that is otherwise SILENT:

* **the lift equality**, per variant. The fused kernel's curl body must be the
  CERTIFIED body with exactly the declared redirects. A drift here is a kernel that
  compiles, launches and converges on the wrong arithmetic;
* **the parsed tap table**. Which cell each foreign tap lands on and which mask guards
  it are PARSED out of the certified body's own lines. A tap redirected to the wrong
  cell is a stencil that collapses quietly;
* **no ``gN +`` below the seam**. In the fused signature those pointers do not exist,
  and a survivor would be a magnetic read this weld did not redirect;
* **the decode anchor's INDENT**, read off the shipped body rather than remembered.
  ``conductivity`` nests its kernels one level deeper than the others, and cutting at
  the wrong indent raises -- but only when someone runs the lift;
* **the four ghost codes**, equal to :mod:`.folded_complex`' own integers. A
  divergence there is a plane of wrong values, not a crash;
* **the arm tables**, agreeing with each other and with the board's own cell labels;
* **``INSTALLABLE`` False**, and the reason naming the arbitration rather than only
  the label.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from meep_gpu.triton_kernels import beta_complex_fused_hd_pair as beta_complex
from meep_gpu.triton_kernels import beta_real_fused_hd_pair as beta_real
from meep_gpu.triton_kernels import complex_fused_hd_pair as complex_hd
from meep_gpu.triton_kernels import conductive_bfast_fused_hd_pair as cond_bfast
from meep_gpu.triton_kernels import coverage as tcoverage
from meep_gpu.triton_kernels import folded_complex as folded_complex_module
from meep_gpu.triton_kernels import folded_complex_fused_hd_pair as folded_complex
from meep_gpu.triton_kernels import fused_hd_pair as plain
from meep_gpu.triton_kernels import nonlinear_fused_hd_pair as nonlinear

KERNELS = Path(plain.__file__).parent

#: The five families, with the cells the 2026-09-07 board puts in each and the
#: instance count it prices them at. The counts are the fusion matrix's own
#: (``aggregate.h_to_d_seam.instances``, ``one_per_row`` true, denominator 597) and
#: are pinned HERE so a product that quietly widened its arm table fails the merge bar
#: rather than the board.
FAMILIES: Tuple[Tuple[Any, int, int], ...] = (
    (cond_bfast, 2, 2),
    (beta_real, 2, 2),
    (folded_complex, 5, 2),
    (beta_complex, 4, 2),
    (nonlinear, 2, 1),
)

#: The board's own labels for the nine cells these five families claim, in the order
#: the board lists them. Transcribed ONCE, here, and compared against every family's
#: own arm declarations -- which is what makes "this product claims that cell" a
#: checkable statement rather than a docstring.
BOARD_CELLS: Tuple[Tuple[str, str], ...] = (
    ("ordinary", "conductive PML"),
    ("BFAST run", "BFAST PML"),
    ("real beta run", "real beta PML"),
    ("folded beta run", "folded real beta PML"),
    ("folded complex", "folded complex PML"),
    ("folded complex off-diagonal", "folded complex off-diagonal PML"),
    ("folded complex", "folded complex beta PML"),
    ("complex beta run", "complex beta PML"),
    ("nonlinear run", "nonlinear run PML"),
)

#: ``family -> {variant: (curl module file, curl function)}``, so the indent check and
#: the lift check walk the same table the family does.
MULTI_VARIANT = (cond_bfast, beta_real, beta_complex)


def _decode_indent(path: Path, function: str) -> int:
    """The indent of the decode line in a shipped body, READ rather than remembered."""
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function:
            return int(node.col_offset) + 4
    raise AssertionError(f"{path.name} declares no {function}")


# ---------------------------------------------------------------------------
# The lift
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family", MULTI_VARIANT, ids=lambda m: m.FAMILY)
def test_every_variants_curl_body_is_the_certified_body_plus_the_declared_redirects(
        family: Any) -> None:
    for variant in family.VARIANTS:
        certified = family.certified_curl_tail(variant)
        lifted = family.lifted_curl_tail(variant)
        assert certified == lifted, (
            f"{family.FAMILY}/{variant}: the fused kernel's curl body is NOT the "
            f"certified body with exactly the declared redirects")


def test_the_folded_complex_curl_body_is_the_certified_body() -> None:
    assert (folded_complex.certified_curl_tail()
            == folded_complex.lifted_curl_tail())


@pytest.mark.parametrize("family", MULTI_VARIANT, ids=lambda m: m.FAMILY)
def test_no_magnetic_pointer_survives_below_the_seam(family: Any) -> None:
    """In the fused signature ``gN`` does not exist; a survivor is an un-redirected read."""
    for variant in family.VARIANTS:
        lifted = family.lifted_curl_tail(variant)
        for target in range(3):
            assert f"g{target} +" not in lifted, (
                f"{family.FAMILY}/{variant} still reads g{target}")


def test_no_magnetic_pointer_survives_in_the_folded_complex_body() -> None:
    lifted = folded_complex.lifted_curl_tail()
    for target in range(3):
        assert f"g{target} +" not in lifted


@pytest.mark.parametrize("family", MULTI_VARIANT, ids=lambda m: m.FAMILY)
def test_the_decode_anchor_indent_is_the_shipped_bodys_own(family: Any) -> None:
    """READ off the shipped body, never remembered.

    ``conductivity`` declares its kernels INSIDE its Triton guard (``col_offset`` 4)
    and every other module here at module scope, so the anchors are eight spaces and
    four respectively. Cutting at the wrong one finds nothing -- and raises only when
    someone runs the lift.
    """
    for variant in family.VARIANTS:
        anchor = family.DECODE_END[variant]
        measured = _decode_indent(family.CURL_PATHS[variant],
                                  family.CURL_FUNCTIONS[variant])
        assert len(anchor) - len(anchor.lstrip()) == measured, (
            f"{family.FAMILY}/{variant}: the declared anchor indent does not match "
            f"{family.CURL_FUNCTIONS[variant]}'s own")


def test_the_folded_complex_decode_anchor_indent_is_the_shipped_bodys_own() -> None:
    anchor = folded_complex.DECODE_END
    measured = _decode_indent(folded_complex.CURL_PATH, folded_complex.CURL_FUNCTION)
    assert len(anchor) - len(anchor.lstrip()) == measured


@pytest.mark.parametrize("family", MULTI_VARIANT, ids=lambda m: m.FAMILY)
def test_every_certified_body_parses_to_the_declared_six_taps(family: Any) -> None:
    """The tap table is PARSED, and the parse must produce exactly the declared six.

    This is what makes "the foreign-tap set is the plain product's" a MEASUREMENT
    rather than a reading of the certified body's history.
    """
    for variant in family.VARIANTS:
        raw = family.raw_curl_tail(variant)
        parser = complex_hd if family is beta_complex else plain
        taps = parser.halo_taps(raw)
        offsets = parser.offset_coordinates(raw)
        assert set(taps) == {name for name, _ in parser.HALO_TAPS}
        assert set(offsets) == {"ox", "oy", "oz"}


def test_the_folded_complex_body_parses_to_the_declared_six_word_pairs() -> None:
    raw = folded_complex.raw_curl_tail()
    taps = complex_hd.halo_taps(raw)
    assert set(taps) == {name for name, _ in complex_hd.HALO_TAPS}
    for stem, planes in taps.items():
        assert set(planes) == {"re", "im"}, stem
        assert planes["re"] == planes["im"], (
            f"{stem}: the two word planes disagree on component, cell or guard, so "
            f"the pair is not one complex operand")


@pytest.mark.parametrize("family", MULTI_VARIANT, ids=lambda m: m.FAMILY)
def test_the_lift_edits_are_generated_not_transcribed(family: Any) -> None:
    """Every edit's replacement must name a recompute helper, never a raw load."""
    for variant in family.VARIANTS:
        edits = family.curl_lift_edits(variant)
        assert edits, f"{family.FAMILY}/{variant} declares no edits"
        for _old, new in edits:
            assert ("_h_tap" in new or "own" in new), new


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family,instances,cells", FAMILIES,
                         ids=lambda value: getattr(value, "FAMILY", str(value)))
def test_the_arm_tables_agree_with_each_other(family: Any, instances: int,
                                              cells: int) -> None:
    primary = tuple(family.ARMS)
    extra = tuple(tuple(pair) for pair in getattr(family, "EXTRA_ARMS", ()))
    claimed = (primary,) + extra
    assert len(claimed) == cells, (
        f"{family.FAMILY} claims {len(claimed)} cells, not the board's {cells}")
    table = getattr(family, "VARIANT_ARMS", None) or getattr(
        family, "ADMISSION_ARMS", None)
    if table is not None:
        assert tuple(tuple(pair) for pair in table.values()) == claimed, (
            f"{family.FAMILY}'s per-variant arm table disagrees with ARMS/EXTRA_ARMS")
    for pair in claimed:
        assert pair in BOARD_CELLS, (
            f"{family.FAMILY} claims {pair}, which is not one of the nine cells the "
            f"2026-09-07 board prices for this round")


def test_the_five_families_claim_the_nine_board_cells_exactly_once_each() -> None:
    """No cell is claimed twice and none is left unclaimed.

    Two products admitting one cell is a dispatcher picking by ordering; a cell with
    no product is a board row this round did not close.
    """
    claimed: List[Tuple[str, str]] = []
    for family, _instances, _cells in FAMILIES:
        claimed.append(tuple(family.ARMS))
        claimed.extend(tuple(pair) for pair in getattr(family, "EXTRA_ARMS", ()))
    assert sorted(claimed) == sorted(BOARD_CELLS)
    assert len(set(claimed)) == len(claimed), "a cell is claimed twice"


@pytest.mark.parametrize("family,instances,cells", FAMILIES,
                         ids=lambda value: getattr(value, "FAMILY", str(value)))
def test_the_seam_declarations_are_the_shipped_tables(family: Any, instances: int,
                                                      cells: int) -> None:
    assert tuple(family.REPLACES) == ("update_H", "step_D")
    assert family.CONSTITUTIVE_SIDE == "H"
    assert family.CURL_SUB_STEP == "step_D"
    assert family.CARRIES_DEPOSIT_REPAIR is False
    assert family.REPAIR_PATHS == ()
    assert family.HOISTS_THE_WITHDRAW is False
    assert tuple(family.ROTATED) == tuple(plain.ROTATED)
    assert tuple(family.IN_PLACE) == tuple(plain.IN_PLACE)
    assert tuple(family.CONSTITUTIVE_SOURCES) == tuple(plain.CONSTITUTIVE_SOURCES)


@pytest.mark.parametrize("family,instances,cells", FAMILIES,
                         ids=lambda value: getattr(value, "FAMILY", str(value)))
def test_installable_is_false_and_the_reason_names_the_arbitration(
        family: Any, instances: int, cells: int) -> None:
    assert family.INSTALLABLE is False
    reason = family.INSTALLABLE_REASON
    assert "arbitration" in reason.lower(), (
        f"{family.FAMILY}'s reason gives only the label half; the arbitration is the "
        f"half a reader needs to know this product is not free")
    assert "4 - (installed pairs)" in reason or "installs nothing of its own" in reason


def test_the_nonlinear_admission_restates_the_welded_familys_flag() -> None:
    """It builds ``FusedHdPairPlan``, so THAT family's flag is what the composer reads."""
    assert nonlinear.INSTALLABLE == plain.INSTALLABLE
    assert nonlinear.WELDED_FAMILY == plain.FAMILY
    assert nonlinear.CERTIFIES_NO_NEW_KERNEL is True


def test_the_nonlinear_admission_ships_no_kernel_of_its_own() -> None:
    """A kernel here would make "no new kernel" false without anything else changing."""
    text = Path(nonlinear.__file__).read_text(encoding="utf-8")
    assert "@triton.jit" not in text
    assert "tl.store" not in text
    assert "import triton" not in text


#: The families whose kernels take the FOUR folded ghost codes. Declared rather than
#: discovered by a skip: a family that stopped declaring them would otherwise leave a
#: silent hole where a contract used to be.
FOLDED_CODE_FAMILIES = (beta_real, folded_complex, beta_complex)


@pytest.mark.parametrize("family", FOLDED_CODE_FAMILIES, ids=lambda m: m.FAMILY)
def test_the_ghost_codes_are_the_certified_modules_own(family: Any) -> None:
    """Read off the family's source, because the constants live behind a Triton guard.

    A plan built here and a plan built in :mod:`.folded_complex` index the same table,
    so a divergence in these four integers is a plane of wrong values rather than a
    crash -- and the module-scope constants cannot be imported on a host with no
    Triton, which is exactly the host this contract has to hold on.
    """
    text = Path(family.__file__).read_text(encoding="utf-8")
    assert "MIRROR_PERIODIC = tl.constexpr" in text, (
        f"{family.FAMILY} no longer declares the four folded ghost codes")
    for name, value in (("PERIODIC", folded_complex_module.CODE_PERIODIC),
                        ("METALLIC", folded_complex_module.CODE_METALLIC),
                        ("MIRROR_METALLIC",
                         folded_complex_module.CODE_MIRROR_METALLIC),
                        ("MIRROR_PERIODIC",
                         folded_complex_module.CODE_MIRROR_PERIODIC)):
        assert f"{name} = tl.constexpr(_folded.CODE_{name})" in text, (
            f"{family.FAMILY} spells {name} as a literal rather than reading "
            f"folded_complex's own integer")
        assert value == {"PERIODIC": 0, "METALLIC": 1, "MIRROR_METALLIC": 2,
                         "MIRROR_PERIODIC": 3}[name]


@pytest.mark.parametrize("family,instances,cells", FAMILIES,
                         ids=lambda value: getattr(value, "FAMILY", str(value)))
def test_default_block_and_backward_are_pinned_to_the_shipped_tables(
        family: Any, instances: int, cells: int) -> None:
    if hasattr(family, "DEFAULT_BLOCK"):
        assert family.DEFAULT_BLOCK == 256
    if hasattr(family, "BACKWARD"):
        assert family.BACKWARD == plain.BACKWARD == 1


def test_the_constitutive_side_is_not_half_integer() -> None:
    """The whole premise of the shared ``kms`` group, checked where it is cheap.

    Both halves sit on the INTEGER sub-lattice, which is what lets one coefficient
    group serve both. Binding the half-integer set instead compiles, launches and
    converges -- a half-cell error in the absorber profile.
    """
    assert tcoverage.CONSTITUTIVE_SIDES["H"]["half_integer"] is False


# ---------------------------------------------------------------------------
# Predicates, on a host with no device
# ---------------------------------------------------------------------------

class _Bare:
    """An object with no ``grid`` -- the first thing every predicate must refuse."""


@pytest.mark.parametrize("family,instances,cells", FAMILIES,
                         ids=lambda value: getattr(value, "FAMILY", str(value)))
def test_a_fields_without_a_grid_is_refused(family: Any, instances: int,
                                            cells: int) -> None:
    coverage = getattr(family, f"{family.FAMILY}_coverage")
    verdict = coverage(_Bare(), None, ())
    assert not verdict.covered
    assert verdict.reasons, f"{family.FAMILY} refused without a reason"
    if family is nonlinear:
        # THE INVERTED CLAUSE RUNS FIRST on this admission, deliberately: a LINEAR run
        # is the ordinary weld's row and saying so is more useful than "no grid".
        assert any("chi2/chi3" in reason for reason in verdict.reasons)
    else:
        assert any("grid" in reason for reason in verdict.reasons)


@pytest.mark.parametrize("family,instances,cells", FAMILIES,
                         ids=lambda value: getattr(value, "FAMILY", str(value)))
def test_the_plan_builders_refuse_rather_than_raise(family: Any, instances: int,
                                                    cells: int) -> None:
    """``None`` is the only refusal: a configuration this kernel does not carry must
    fall back to the array path, never raise into a caller that would otherwise have
    stepped correctly."""
    builder = getattr(family, f"plan_{family.FAMILY}")
    assert builder(_Bare(), None, ()) is None


def test_the_two_variant_resolvers_refuse_an_ambiguous_run_by_name() -> None:
    """Both predicates admitting is refused BY NAME rather than resolved by order.

    Two predicates admitting one configuration is a dispatcher picking by branch
    order, which is the failure the arm table exists to prevent.
    """
    for family in (cond_bfast, beta_real, beta_complex):
        variant, reasons = family.resolve_variant(_Bare(), None)
        assert variant is None
        assert reasons


def test_the_folded_complex_admission_resolver_refuses_a_bare_object() -> None:
    admission, reasons = folded_complex.resolve_admission(_Bare(), None)
    assert admission is None
    assert reasons


def test_every_family_is_importable_without_triton() -> None:
    """The merge bar. Every module above imported at collection time; this states it."""
    for family, _instances, _cells in FAMILIES:
        assert family.FAMILY
        assert callable(getattr(family, f"plan_{family.FAMILY}"))


def test_the_nonlinear_equivalence_identity_is_reported_on_a_bare_object() -> None:
    """The identity is DATA, not a docstring: a caller can assert it."""
    identity = nonlinear.nonlinear_fused_hd_pair_equivalence(_Bare(), None, ())
    assert identity["weld_never_wider_than_the_halves"] is True
    assert identity["weld_covered"] is False


def test_the_conductive_variant_reads_its_state_names_from_the_certified_table(
) -> None:
    """A rename in ``bfast_curl`` must RAISE here rather than bind the wrong volume."""
    from meep_gpu.triton_kernels import bfast_curl  # noqa: PLC0415

    assert tuple(cond_bfast.BFAST_STATES) == tuple(
        bfast_curl.BFAST_STATE_NAMES["step_D"])
    assert cond_bfast.CONDUCTIVE_HISTORY == ("f_cond_Dx", "f_cond_Dy", "f_cond_Dz")


def test_the_families_declare_the_multiply_helpers_they_may_reach() -> None:
    """A helper outside the declared set is an orientation neither half launches."""
    import re

    for family in (folded_complex, beta_complex):
        text = Path(family.__file__).read_text(encoding="utf-8")
        called = set(re.findall(r"\b(_(?:mul|rotate|div)_[A-Za-z0-9_]*)\s*\(", text))
        assert called <= set(family.MULTIPLY_HELPERS), (
            f"{family.FAMILY} reaches {sorted(called - set(family.MULTIPLY_HELPERS))}")


def test_the_complex_families_restate_the_certified_subnormal_policy() -> None:
    """The expansion licence is the certified family's, not this product's."""
    from meep_gpu.triton_kernels import complex_fields  # noqa: PLC0415

    for family in (folded_complex, beta_complex):
        assert (family.CERTIFIED_UNDER_SUBNORMAL_POLICY
                == complex_fields.CERTIFIED_UNDER_SUBNORMAL_POLICY)


# ---------------------------------------------------------------------------
# THE LIFT JOIN. Six of this lane's fifteen corpus rows were silently excluded on the
# first full campaign because a name match found nothing and the miss was reported as
# a fact about the host's MEEP build. These pin the repair on the host, where it is
# cheap, rather than on a device an hour away.
# ---------------------------------------------------------------------------

class _ShimMethod:
    """One method as the census's `parameterized` stand-in generates it.

    The three attributes are exactly what `parity/meep_gpu/shim/parameterized.py`
    puts on each generated method, and the ARGS are what make the join a check.
    """

    def __init__(self, source: str, index: int, *args: object) -> None:
        self.__parameterized_source__ = source
        self.__parameterized_index__ = index
        self.__parameterized_args__ = args


class _PlainMethod:
    """A method the module wrote by hand: no parameterized attributes at all."""


class _ShimModule:
    """A module whose class carries shim-generated methods under shim names."""

    # The parameter tuples are TRANSCRIBED from the standing census's own
    # `parameterized` blocks for these rows
    # (results/predicate_coverage_triton_2026-09-04_cylm0/per_row_tests_param/
    #  test_binary_grating.json), so this fixture is the corpus, not an invention.
    class TestEigCoeffs:
        test_binary_grating_special_kz__idx0 = _ShimMethod(
            "test_binary_grating_special_kz", 0, 13.2, "real/imag")
        test_binary_grating_special_kz__idx1 = _ShimMethod(
            "test_binary_grating_special_kz", 1, 17.7, "complex")
        test_binary_grating_special_kz__idx2 = _ShimMethod(
            "test_binary_grating_special_kz", 2, 21.2, "3d")
        test_plain = _PlainMethod()


_SHIM_NAMES = (("TestEigCoeffs", "test_binary_grating_special_kz__idx0"),
               ("TestEigCoeffs", "test_binary_grating_special_kz__idx1"),
               ("TestEigCoeffs", "test_binary_grating_special_kz__idx2"),
               ("TestEigCoeffs", "test_plain"))


def _kit():
    """The gate kit, imported the way a gate script reaches it."""
    import sys  # noqa: PLC0415

    root = Path(__file__).resolve().parents[1]
    for entry in (str(root), str(root / "parity" / "meep_gpu")):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    import importlib  # noqa: PLC0415

    return importlib.import_module("parity.meep_gpu.triton_hd_tail_gate_kit")


def test_the_lift_join_resolves_an_upstream_parameterized_name_to_the_shim_method(
) -> None:
    """The defect: upstream names the case, the shim names the method, and they differ.

    ``parameterized.expand`` builds ``{func}_{index}_{safe_first_param}`` and the
    census's stand-in deliberately builds ``{func}__idx{index}``. The census joins by
    facts; this gate joins by (source, index) and CONFIRMS with the recorded grid.
    """
    kit = _kit()
    for index, recorded in enumerate(("test_binary_grating_special_kz_0_13_2",
                                      "test_binary_grating_special_kz_1_17_7",
                                      "test_binary_grating_special_kz_2_21_2")):
        target, why = kit.resolve_case(_ShimModule, _SHIM_NAMES,
                                       f"TestEigCoeffs.{recorded}")
        assert target == ("TestEigCoeffs",
                          f"test_binary_grating_special_kz__idx{index}"), why
        assert "parameterized join" in why


def test_the_lift_join_prefers_an_exact_name_and_says_so() -> None:
    kit = _kit()
    target, why = kit.resolve_case(_ShimModule, _SHIM_NAMES,
                                   "TestEigCoeffs.test_plain")
    assert target == ("TestEigCoeffs", "test_plain")
    assert why == "exact name"


def test_the_lift_join_REFUSES_rather_than_picks_when_two_methods_claim_the_name(
) -> None:
    """THE CONTROL THAT BITES. Ambiguity must not be broken by iteration order.

    Two generated methods whose (source, index) both prefix the recorded name is a
    join that cannot decide, and deciding anyway is how a measurement is attributed to
    the wrong corpus row without anything going red.
    """
    kit = _kit()

    class _Ambiguous:
        class TestEigCoeffs:
            a = _ShimMethod("test_x", 1, 13)
            b = _ShimMethod("test_x", 1, 13)

    target, why = kit.resolve_case(_Ambiguous, (("TestEigCoeffs", "a"),
                                                ("TestEigCoeffs", "b")),
                                   "TestEigCoeffs.test_x_1_13")
    assert target is None
    assert "REFUSED rather" in why


def test_the_lift_join_refuses_a_name_no_enumerated_case_carries() -> None:
    """And it does NOT call the miss a fact about this host's MEEP build."""
    kit = _kit()
    target, why = kit.resolve_case(_ShimModule, _SHIM_NAMES,
                                   "TestEigCoeffs.test_absent_9_9")
    assert target is None
    assert "no enumerated case matches" in why
    assert "host" not in why.lower()


def test_an_index_that_does_not_exist_is_not_rounded_to_a_neighbour() -> None:
    """Index 7 is not index 0. A prefix join that matched loosely would take one."""
    kit = _kit()
    target, _why = kit.resolve_case(
        _ShimModule, _SHIM_NAMES, "TestEigCoeffs.test_binary_grating_special_kz_7_1")
    assert target is None


def test_the_index_is_not_confused_with_a_longer_index_sharing_its_digits() -> None:
    """``_1_...`` must not match the method whose index is 12."""
    kit = _kit()

    class _Wide:
        class T:
            m = _ShimMethod("test_x", 12, 2)

    target, _why = kit.resolve_case(_Wide, (("T", "m"),), "T.test_x_1_2")
    assert target is None


def test_the_gate_kit_names_the_policy_certification_needle_it_scores_on() -> None:
    """The unlicensed-policy leg scores a refusal BY CLAUSE, so the needle is data."""
    kit = _kit()
    assert kit.CERTIFICATION_NEEDLE == "CERTIFIED under the"


def test_only_the_complex_products_ask_the_policy_for_a_licence() -> None:
    """A real-storage product has no expansion licence to lose, and must not fake one."""
    kit = _kit()
    for family, _instances, _cells in FAMILIES:
        if family in (folded_complex, beta_complex):
            continue

        class _RealProduct(kit.Product):
            complex_storage = False

        licensed, reasons = kit.policy_licenses_the_arms(_RealProduct())
        assert licensed is True
        assert reasons == []


def test_upstream_safe_name_reproduces_the_three_shapes_the_corpus_uses() -> None:
    """Transcribed from the census's own recorded parameter tuples for these rows.

    A float, a slash-bearing string and a plain word are the three shapes the six
    rows of this lane carry, and each one's recorded name fixes the answer.
    """
    kit = _kit()
    assert kit.upstream_safe_name(13.2) == "13_2"
    assert kit.upstream_safe_name("real/imag") == "real_imag"
    assert kit.upstream_safe_name("complex") == "complex"
    assert kit.upstream_safe_name(35.7) == "35_7"


def test_every_parameterized_row_this_lane_lifts_reconstructs_its_recorded_name(
) -> None:
    """THE WHOLE DEFECT, as a table: six recorded names, six generated methods.

    Every tuple here is transcribed from the standing census's `parameterized` block
    for that row; the recorded names are the board's, off the seam record. A join that
    landed on the wrong index or dropped the suffix check fails this outright.
    """
    kit = _kit()
    corpus = (
        ("TestEigCoeffs", "test_binary_grating_special_kz", 0, (13.2, "real/imag"),
         "test_binary_grating_special_kz_0_13_2"),
        ("TestEigCoeffs", "test_binary_grating_special_kz", 1, (17.7, "complex"),
         "test_binary_grating_special_kz_1_17_7"),
        ("TestEigCoeffs", "test_binary_grating_special_kz", 2, (21.2, "3d"),
         "test_binary_grating_special_kz_2_21_2"),
        ("TestSpecialKz", "test_eigsrc_kz", 0, ("complex",),
         "test_eigsrc_kz_0_complex"),
        ("TestSpecialKz", "test_eigsrc_kz", 1, ("real/imag",),
         "test_eigsrc_kz_1_real_imag"),
        ("TestReflectanceAngular", "test_reflectance_angular", 2, (35.7, True),
         "test_reflectance_angular_2_35_7"),
    )
    for class_name, source, index, args, recorded in corpus:
        method_name = f"{source}__idx{index}"
        cls = type(class_name, (), {method_name: _ShimMethod(source, index, *args)})
        module = type("_M", (), {class_name: cls})
        target, why = kit.resolve_case(module, ((class_name, method_name),),
                                       f"{class_name}.{recorded}")
        assert target == (class_name, method_name), why
        assert "RECONSTRUCTED exactly" in why


def test_the_join_refuses_when_the_index_matches_but_the_parameter_does_not() -> None:
    """THE CONTROL THAT BITES HARDEST. The index alone must not carry a row.

    An index join with no suffix check would take this method and attribute a
    measurement to the wrong corpus row in silence -- the failure mode that has no
    red anywhere.
    """
    kit = _kit()

    class _M:
        class T:
            test_x__idx1 = _ShimMethod("test_x", 1, 99.9)

    target, why = kit.resolve_case(_M, (("T", "test_x__idx1"),), "T.test_x_1_13_2")
    assert target is None
    assert "no method's parameter tuple RECONSTRUCTS that name" in why
