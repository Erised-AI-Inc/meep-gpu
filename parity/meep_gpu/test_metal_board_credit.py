"""The Metal board's credit rules: the composer-absorb credit, the credit withheld by name,
and the per-product three-seam disjointness floor (``metal_board_credit``).

What each test pins, and what would break it:

- The floor REFUSES a genuine predicate overlap: a row one product's ladder admits while
  the composer installs another product's pair on it is named, product and row, and the
  cut stops. Also refused: a credit the ladder does not admit, a row both credited and
  withheld, and an absorb credit naming a pair other than the product it credits.
- An absorb credit is accepted, counted apart from the selected-pair credits, and names
  its pair.
- A row withheld by name is accounted on the product's ladder side and listed; without
  its entry the same row refuses the cut.
- ``absorb_credit`` credits exactly the composer's rule: a start-clause pop the absorbing
  pair's bracket carries, one admitter at the slot, one pair that may absorb it, the
  pair's ladder admitting the row, and a pair plan that does not declare
  ``starts_h_from_b = False``. Every other row is ``None``; a pop with two admitters is a
  census/composer contradiction and raises.
- ``withheld_disagreement`` stands only while the census still shows the disagreement,
  and the shipped table is empty (the one 1.0 entry was settled at the composer).
- Where the composer is importable, the start-clause wording is derived from
  ``launch.h_start_reasons`` itself and the shipped absorb table credits the ordinary
  B/H pair through the composer's own ``_pair_may_absorb``.

No census, no board and no device: the rules are exercised on synthetic rows shaped like
the census's, and the composer-backed tests skip where ``meep_gpu.metal_kernels.launch``
does not import.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import metal_board_credit as credit  # noqa: E402

LEAD = ("the run's first update_H starts H and f_w from B on the dsigw cells "
        "(stepping.separate_h_from_b) and the base kernel does not; ")
VOLUMES = ("Bx", "fu_Bx", "Ex")


def _h_start_reasons(fields, sources, pml=None):
    """A stand-in with the composer's three clauses and its wording's shape."""
    if getattr(fields, "h_separated", False):
        return ()
    reasons = []
    if sources is None:
        reasons.append(LEAD + "the source set was not declared, so no magnetic deposit "
                       "into B before that update can be ruled out")
    else:
        magnetic = [i for i, s in enumerate(sources)
                    if str(getattr(s, "field_type", "")) == "B"]
        if magnetic:
            reasons.append(LEAD + f"magnetic source(s) {magnetic} deposit into B before "
                           f"that update")
    for name in VOLUMES:
        values = getattr(fields, name, None)
        if values is not None and values.any():
            reasons.append(LEAD + f"{name} holds a word other than +0 now, so B need not "
                           f"be +0 at that update")
    return tuple(reasons)


def _may_absorb(selected, curl, update, arms):
    """The composer's ``_pair_may_absorb`` rule, restated for the synthetic tests."""
    for name, arm in ((curl, arms[0]), (update, arms[1])):
        won = selected.get(name)
        if won is None:
            return f"{name} is not selected"
        if won != arm:
            return f"{name} was selected by {won!r}"
    return None


START = credit.derive_start_clause(_h_start_reasons, VOLUMES)
SEAM_ABSORBS = {"step_B": "update_H", "step_D": "update_E"}
CANDIDATES = {"pair_ordinary": (("PML", "ordinary"),),
              "pair_complex": (("complex/Bloch", "complex/Bloch"),)}
LABEL = "examples:row.py"


def _row(reasons=None, update_h=None, covered=("constitutive",), extra_arms=()):
    arms = [{"slot": "update_H", "family": "constitutive", "label": "ordinary",
             "wired": True},
            {"slot": "update_H", "family": "complex_fields", "label": "complex/Bloch",
             "wired": True},
            {"slot": "update_H", "family": "pair_hd", "label": "fused H/D pair",
             "wired": False}] + list(extra_arms)
    predicates = {
        f"{family}@update_H": {"covered": family in covered, "family": family,
                               "sub_step": "update_H"}
        for family in ("constitutive", "complex_fields", "pair_hd")}
    selected = {"step_B": "PML", "step_D": "PML"}
    if update_h is not None:
        selected["update_H"] = update_h
    return {"leg": "examples", "row": "row.py", "arm_table": arms,
            "predicates": predicates,
            "plan_step": {"selected": selected,
                          "reasons": {"update_H": list(reasons or ())}}}


MAGNETIC = LEAD + "magnetic source(s) [1] deposit into B before that update (Hy's own layer)"


def _absorb(row, admitted=None, starts=None):
    return credit.absorb_credit(
        row, LABEL, "B_to_H", "step_B", "update_H", START, SEAM_ABSORBS, CANDIDATES,
        _may_absorb, admitted if admitted is not None else {"pair_ordinary": [LABEL]},
        starts or (lambda product: None))


# --- the start clause -------------------------------------------------------------


def test_the_start_clause_wording_is_derived_not_typed():
    assert START.lead == LEAD
    assert len(START.pair_refusing) == len(VOLUMES) + 1
    assert START.pair_carries(MAGNETIC)
    assert not START.pair_carries(LEAD + "Bx holds a word other than +0 now, so B need "
                                  "not be +0 at that update")
    assert not START.is_start_reason("ordinary: force_complex_fields=True")


def test_a_composer_whose_clauses_do_not_separate_is_refused():
    with pytest.raises(SystemExit, match="returned 0 reasons"):
        credit.derive_start_clause(lambda fields, sources: (), VOLUMES)


# --- the absorb credit ------------------------------------------------------------


def test_a_start_clause_pop_is_credited_to_the_pair_that_absorbs_and_names_it():
    got = _absorb(_row([MAGNETIC]))
    assert got is not None
    assert got["pair"] == "pair_ordinary"
    assert (got["slot"], got["arm"], got["curl_arm"]) == ("update_H", "ordinary", "PML")
    assert got["base_composer_reasons"] == [MAGNETIC]


def test_a_reason_the_pair_refuses_too_is_not_credited():
    volume = LEAD + "Bx holds a word other than +0 now, so B need not be +0 at that update"
    assert _absorb(_row([MAGNETIC, volume])) is None
    undeclared = LEAD + ("the source set was not declared, so no magnetic deposit into B "
                         "before that update can be ruled out")
    assert _absorb(_row([undeclared])) is None


def test_a_slot_refused_for_another_reason_or_selected_is_not_an_absorb():
    assert _absorb(_row(["ordinary: a mirror plane is active"])) is None
    assert _absorb(_row([MAGNETIC], update_h="ordinary")) is None
    assert _absorb(_row([])) is None


def test_no_credit_where_the_pair_ladder_refuses_or_its_plan_does_not_start_h():
    assert _absorb(_row([MAGNETIC]), admitted={"pair_ordinary": []}) is None
    assert _absorb(_row([MAGNETIC]), starts=lambda product: False) is None
    assert _absorb(_row([MAGNETIC]), starts=lambda product: True) is not None


def test_no_credit_where_no_pair_or_two_pairs_may_absorb():
    complex_row = _row([MAGNETIC], covered=("complex_fields",))
    complex_row["plan_step"]["selected"]["step_B"] = "BFAST"
    assert _absorb(complex_row) is None
    twice = dict(CANDIDATES, pair_twin=(("PML", "ordinary"),))
    assert credit.absorb_credit(
        _row([MAGNETIC]), LABEL, "B_to_H", "step_B", "update_H", START, SEAM_ABSORBS,
        twice, _may_absorb, {"pair_ordinary": [LABEL], "pair_twin": [LABEL]},
        lambda product: None) is None


def test_a_pop_on_a_slot_with_two_admitters_contradicts_the_composer():
    with pytest.raises(SystemExit, match="2 wired arms cover the slot"):
        _absorb(_row([MAGNETIC], covered=("constitutive", "complex_fields")))


def test_an_unwired_weld_is_not_an_admitter():
    row = _row([MAGNETIC], covered=("constitutive", "pair_hd"))
    assert credit.slot_admitters(row, "update_H") == ["ordinary"]
    assert _absorb(row)["pair"] == "pair_ordinary"


# --- the floor ---------------------------------------------------------------------


def _instance(row, product, served, absorbed=None, withheld=None):
    return {"row": row, "product": product, "served": served,
            "absorbed": absorbed, "credit_withheld": withheld}


def test_the_floor_refuses_a_planted_overlap_naming_product_and_row():
    ladders = {"pair_ordinary": ["r1", "r2"], "pair_complex": ["r2"]}
    instances = [_instance("r1", "pair_ordinary", True),
                 _instance("r2", "pair_ordinary", True)]
    with pytest.raises(SystemExit) as caught:
        credit.three_seam_floor(ladders, instances)
    text = str(caught.value)
    assert "admit 3 rows but only 2 seam-instances credit a product" in text
    assert "{'pair_complex': ['r2']}" in text


def test_the_floor_accepts_an_absorb_credit_and_counts_it_apart():
    ladders = {"pair_ordinary": ["r1", "r2"]}
    instances = [_instance("r1", "pair_ordinary", True),
                 _instance("r2", "pair_ordinary", True,
                           absorbed={"pair": "pair_ordinary", "slot": "update_H"})]
    block = credit.three_seam_floor(ladders, instances)
    assert block["agree"] is True
    assert (block["credited"], block["credited_by_the_selected_pair"],
            block["credited_by_an_absorbing_pair"]) == (2, 1, 1)
    assert block["absorb_credits_by_pair"] == {"pair_ordinary": 1}


def test_an_absorb_credit_naming_another_pair_is_refused():
    instances = [_instance("r1", "pair_ordinary", True,
                           absorbed={"pair": "pair_complex"})]
    with pytest.raises(SystemExit, match="names 'pair_complex'"):
        credit.three_seam_floor({"pair_ordinary": ["r1"]}, instances)


def test_a_row_withheld_by_name_is_accounted_and_listed_and_refuses_without_its_entry():
    ladders = {"pair_offdiag": ["r1", "r2"]}
    held = {"product": "pair_offdiag", "reason": "the composer refuses the arm"}
    block = credit.three_seam_floor(ladders, [
        _instance("r1", "pair_offdiag", True),
        _instance("r2", None, False, withheld=held)])
    assert block["credit_withheld_by_name"] == 1
    assert block["withheld_by_product"] == {"pair_offdiag": ["r2"]}
    with pytest.raises(SystemExit, match=r"\{'pair_offdiag': \['r2'\]\}"):
        credit.three_seam_floor(ladders, [_instance("r1", "pair_offdiag", True),
                                          _instance("r2", None, False)])


def test_a_credit_without_admission_or_a_row_credited_and_withheld_is_refused():
    with pytest.raises(SystemExit, match="credited or withheld without admission"):
        credit.three_seam_floor({"pair_ordinary": []},
                                [_instance("r1", "pair_ordinary", True)])
    held = {"product": "pair_ordinary", "reason": "x"}
    with pytest.raises(SystemExit, match=r"both credited and withheld: \{'pair_ordinary'"):
        credit.three_seam_floor({"pair_ordinary": ["r1"]},
                                [_instance("r1", "pair_ordinary", True, withheld=held)])


# --- the credit withheld by name ---------------------------------------------------


ENTRY = {"product": "pair_offdiag", "slot": "update_E", "arm": "offdiag E",
         "predicate_key": "offdiag@update_E", "curl_arm": "folded complex",
         "reason": "the composer refuses the arm", "evidence": "a measurement"}


def _withheld_row(update_e=None, covered=True):
    selected = {"step_D": "folded complex"}
    if update_e is not None:
        selected["update_E"] = update_e
    return {"plan_step": {"selected": selected,
                          "reasons": {"update_E": ["ordinary: complex storage"]}},
            "predicates": {"offdiag@update_E": {"covered": covered,
                                                "sub_step": "update_E"}}}


def _check(row, pair=("folded complex", "offdiag E"), admitted=(LABEL,)):
    return credit.withheld_disagreement(row, LABEL, "D_to_E", "step_D", ENTRY, pair,
                                        admitted)


def test_a_withheld_entry_stands_while_the_census_shows_the_disagreement():
    got = _check(_withheld_row())
    assert got["product"] == "pair_offdiag" and got["reason"] == ENTRY["reason"]
    assert got["composer_reasons_on_the_census"] == ["ordinary: complex storage"]


@pytest.mark.parametrize("row, pair, admitted, needle", [
    (_withheld_row(update_e="offdiag E"), ("folded complex", "offdiag E"), (LABEL,),
     "no longer refuses the arm"),
    (_withheld_row(covered=False), ("folded complex", "offdiag E"), (LABEL,),
     "does not cover update_E"),
    (_withheld_row(), ("folded complex", "offdiag E"), (),
     "no longer admits the row"),
    (_withheld_row(), ("folded", "offdiag E"), (LABEL,), "the product's pair is"),
])
def test_a_stale_withheld_entry_refuses_the_cut(row, pair, admitted, needle):
    with pytest.raises(SystemExit, match=needle):
        _check(row, pair, admitted)


def test_the_shipped_table_is_empty():
    """The one 1.0 entry was settled at the composer: a composer handed no division
    reads the run's own, so the census selects the arm the product implements there."""
    assert credit.CREDIT_WITHHELD == {}


def _launch():
    try:
        from meep_gpu.metal_kernels import launch  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001 - no composer on this host
        pytest.skip(f"meep_gpu.metal_kernels.launch does not import here: {exc!r}")
    return launch


def test_the_start_clause_is_derived_from_the_composer_s_own_function():
    launch = _launch()
    start = credit.derive_start_clause(launch.h_start_reasons, launch.H_START_VOLUMES)
    assert start.lead.startswith("the run's first update_H starts H and f_w from B")
    assert len(start.pair_refusing) == len(launch.H_START_VOLUMES) + 1
    assert start.pair_carries(start.magnetic_example)


def test_the_composer_s_absorb_table_credits_the_ordinary_b_h_pair():
    launch = _launch()
    start = credit.derive_start_clause(launch.h_start_reasons, launch.H_START_VOLUMES)
    seam_absorbs = {curl: update for curl, (update, _name)
                    in launch.FUSED_PAIR_SEAMS.items()}
    assert seam_absorbs["step_B"] == "update_H"
    candidates = {"fused_magnetic_pair": (tuple(launch.FUSED_PAIR_ARMS[
        "fused_magnetic_pair"]),)}
    magnetic = start.lead + "magnetic source(s) [1] deposit into B before that update"
    got = credit.absorb_credit(
        _row([magnetic]), LABEL, "B_to_H", "step_B", "update_H", start, seam_absorbs,
        candidates, launch._pair_may_absorb,  # noqa: SLF001
        {"fused_magnetic_pair": [LABEL]}, lambda product: None)
    assert got is not None and got["pair"] == "fused_magnetic_pair"
    assert got["arm"] == "ordinary"
