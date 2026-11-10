"""The size-aware cost model of the dispatch preference (timing records stamped ``/3``).

A ``/3`` record carries, per product class, the fused and the baseline ms/step at each
size it measured, and the preference reads them at the run's own cell count
(``dispatch_preference.consult_cost``). This file holds the format and the consult to:

* KNOWN VALUE -- the points are the medians and largest spreads of each size's rows,
  with the least margin of those rows; a run between two measured sizes is priced on
  the straight line between them, the margin included, and is vetoed exactly when that
  margin is positive;
* DEGENERATE -- a ``/2`` record carries no cost model and is read as before; a class
  with no measured key, a class whose keys disagree on what they displace and a size
  outside the measured ones have no price; a record whose curve its rows do not give,
  a cost rule this reader does not know and a ``/2`` record carrying a curve refuse by
  name; the consult's barriers answer exactly as the band consult's;
* SCALING -- the prediction at, between and beyond every point, and the crossover;
* SHAPE -- a ``/3`` record round-trips and is idempotent under ``add_costs``;
* WIRED -- at the composer rung the cost model answers first on a ``/3`` record (clean
  and bracketed products alike), the bands and the prior where it has no price, the
  confirmation re-asks it with the singles that took the slots, and a ``/2`` record's
  candidates carry nothing new;
* REALISTIC -- the shipped records, re-stamped ``/3`` by ``add_costs``, validate and
  price the sizes their rows measured.

Nothing here launches a kernel.
"""

from __future__ import annotations

import json
import os
import statistics

import pytest

import meep_gpu.fastpath as fastpath
from meep_gpu import dispatch_preference as dp
from meep_gpu.test_dispatch_preference import (CAPABILITY, LIVE_EMITTERS, LIVE_ROUTE,
                                               _measurement, _record)
from meep_gpu.test_dispatch_preference_wiring import (Composer, _arms, _candidate,
                                                      _clean_process, _freeze, _write)

HERE = os.path.dirname(os.path.abspath(__file__))
SHAPE = {"dimensions": 2, "grid_shape": [200, 120, 1], "complex_storage": False,
         "susceptibilities": 0}

assert _clean_process  # the autouse fixture, imported so it runs for this module too


# ---------------------------------------------------------------------------
# Building records whose rows span sizes
# ---------------------------------------------------------------------------

def _rows(cells, fused, baseline, count=2):
    return [_measurement(f"example_{cells}_{index}", cells, fused, baseline)
            for index in range(count)]


def _key(product, rows, *, bracketed=True, displaces=("PML", "ordinary"),
         slots=("step_D", "update_E"), status="measured", reason=None, storage="real",
         dimensions=2):
    """One key over ``rows`` with the numbers the cutter states (medians, spreads)."""
    ratios = sorted(row["ratio"] for row in rows)
    margin = dp.least_margin(rows)
    entry = {
        "status": status, "product": product, "storage": storage,
        "dimensions": dimensions, "susceptibilities": 0, "bracketed": bracketed,
        "band": {"lo_cells": min(r["cells"] for r in rows),
                 "hi_cells": max(r["cells"] for r in rows)},
        "slots": list(slots), "displaces": list(displaces), "rows": len(rows),
        "fused_ms_per_step": statistics.median(r["fused_ms_per_step"] for r in rows),
        "baseline_ms_per_step": statistics.median(r["baseline_ms_per_step"] for r in rows),
        "ratio": statistics.median(ratios), "ratio_min": ratios[0], "ratio_max": ratios[-1],
        "fused_spread": 0.02, "baseline_spread": 0.01, "veto_margin": margin,
        "slower_beyond_spread": margin > 0.0, "measurements": rows,
    }
    if status != "measured":
        entry["reason"] = reason or "1 row, minimum 2"
    return entry


def _d_curve(product="fused pair D", bracketed=True):
    """Slower at 24,000 and 2,400,000 cells (0.75x, 0.765x), faster at 5,400,000 (1.05x)."""
    return [_key(product, _rows(24000, 0.80, 0.60) + _rows(2400000, 0.85, 0.65),
                 bracketed=bracketed),
            _key(product, _rows(5400000, 1.20, 1.26), bracketed=bracketed)]


def _costed(entries, **kwargs):
    return dp.add_costs(_record(entries, **kwargs))


def _ask(record, cells, *, displaces=None, candidate="fused pair D", bracketed=True,
         **overrides):
    question = dict(table="triton", capability=CAPABILITY, candidate=candidate,
                    displaces=displaces, run_shape=SHAPE, bracketed=bracketed,
                    mode="measured", repair_route=LIVE_ROUTE,
                    subject_sha256=LIVE_EMITTERS, cells=cells)
    question.update(overrides)
    return dp.consult_cost(record, **question)


# ---------------------------------------------------------------------------
# Known value
# ---------------------------------------------------------------------------

def test_a_point_is_the_median_and_largest_spread_of_its_sizes_rows():
    rows = _rows(1000, 1.0, 0.5, count=1) + _rows(1000, 3.0, 0.7, count=1)
    rows += _rows(1000, 2.0, 0.9, count=1) + _rows(2000, 9.0, 9.0, count=1)
    rows[1]["fused_spread"] = 0.07
    points = dp.cost_points(rows, min_rows=2)
    assert points == [{"cells": 1000, "rows": 3, "fused_ms_per_step": 2.0,
                       "baseline_ms_per_step": 0.7, "fused_spread": 0.07,
                       "baseline_spread": 0.01, "veto_margin": 0.47,
                       "slower_beyond_spread": True}]
    assert [p["cells"] for p in dp.cost_points(rows, min_rows=1)] == [1000, 2000]


def test_between_two_sizes_the_cost_is_the_straight_line_between_them():
    record = _costed(_d_curve())
    entry = record["costs"]["fused pair D|real|2|0|bracketed"]
    assert [p["cells"] for p in entry["points"]] == [24000, 2400000, 5400000]
    low, high = entry["points"][1], entry["points"][2]
    at = dp.predict_cost(entry, 4000000)
    weight = (4000000 - 2400000) / 3000000
    assert at["between"] == [2400000, 5400000]
    assert at["fused_ms_per_step"] == round(0.85 + weight * 0.35, 6)
    assert at["baseline_ms_per_step"] == round(0.65 + weight * 0.61, 6)
    assert at["ratio"] == round((0.65 + weight * 0.61) / (0.85 + weight * 0.35), 6)
    # the verdict is the margin's own line, between the two points' least row margins
    assert (low["veto_margin"], high["veto_margin"]) == (
        dp.veto_margin(0.65 / 0.85, 0.02, 0.01), dp.veto_margin(1.26 / 1.20, 0.02, 0.01))
    assert at["veto_margin"] == round(
        low["veto_margin"] + weight * (high["veto_margin"] - low["veto_margin"]), 6)


def test_between_two_sizes_the_margin_is_the_points_own_not_the_medians():
    """Each size: one row 0.90x (slower beyond its spreads) and one 0.98x (not). Every
    point keeps, so every size between them keeps, although the medians (0.94x against
    spreads of 0.03) would read slower; and the mirror: two points that veto by their
    rows veto between them, although their medians against their largest spreads keep."""
    keep = (_rows(24000, 1.0, 0.90, count=1) + _rows(24000, 1.0, 0.98, count=1)
            + _rows(2400000, 1.0, 0.90, count=1) + _rows(2400000, 1.0, 0.98, count=1))
    veto = []
    for cells in (24000, 2400000):
        loose, tight = _rows(cells, 1.0, 0.70, count=1) + _rows(cells, 1.0, 0.90, count=1)
        loose["fused_spread"], tight["baseline_spread"], tight["fused_spread"] = (
            0.20, 0.04, 0.01)
        veto += [loose, tight]
    for rows, outcome, margin in ((keep, "not_vetoed", -0.01), (veto, "vetoed", 0.05)):
        record = _costed([_key("fused pair D", rows)])
        for cells in (24000, 24001, 1212000, 2399999, 2400000):
            answer = _ask(record, cells)
            assert (answer.outcome, answer.evidence["predicted"]["veto_margin"]) == (
                outcome, margin), cells


def test_at_a_measured_size_the_curve_and_the_key_give_one_verdict():
    """Two rows at one size, one slower beyond its spreads and one not: the key keeps
    (every row must be slower), and so does the curve, although the medians alone
    (ratio 0.94 against spreads 0.03) would read slower."""
    rows = _rows(24000, 1.0, 0.90, count=1) + _rows(24000, 1.0, 0.98, count=1)
    record = _costed([_key("fused pair D", rows)])
    band = dp.consult(record, table="triton", capability=CAPABILITY,
                      candidate="fused pair D", displaces=None, run_shape=SHAPE,
                      bracketed=True, mode="measured", repair_route=LIVE_ROUTE,
                      subject_sha256=LIVE_EMITTERS, cells=24000)
    cost = _ask(record, 24000)
    assert band.outcome == cost.outcome == "not_vetoed"
    assert cost.evidence["predicted"]["veto_margin"] == dp.least_margin(rows) < 0
    for cells in (24000, 2400000, 5400000):
        assert (_ask(_costed(_d_curve()), cells).vetoed is dp.consult(
            _costed(_d_curve()), table="triton", capability=CAPABILITY,
            candidate="fused pair D", displaces=None, run_shape=SHAPE, bracketed=True,
            mode="measured", repair_route=LIVE_ROUTE, subject_sha256=LIVE_EMITTERS,
            cells=cells).vetoed)


def test_the_veto_reads_the_prediction_at_the_runs_own_size():
    record = _costed(_d_curve())
    slower = _ask(record, 4000000)
    assert slower.vetoed and slower.reason == "predicted_slower_beyond_spread"
    assert slower.key == "fused pair D|real|2|0|bracketed"
    assert slower.evidence["predicted"]["between"] == [2400000, 5400000]
    assert slower.evidence["displaces_checked"] is False
    kept = _ask(record, 5000000)
    assert kept.outcome == "not_vetoed"
    assert kept.reason == "predicted_not_slower_beyond_spread"


def test_the_crossover_lies_where_the_line_meets_the_spreads():
    record = _costed(_d_curve())
    verdicts = [_ask(record, cells).vetoed for cells in range(2400000, 5400001, 100000)]
    # monotone: slower up to one size, not slower from the next on
    flip = verdicts.index(False)
    assert all(verdicts[:flip]) and not any(verdicts[flip:])
    assert 4000000 < 2400000 + flip * 100000 <= 5000000
    # where the margin's line crosses zero, 2.4M + 3M * m_low / (m_low - m_high), to
    # within the cells its six-decimal rounding spans (about ten here)
    entry = record["costs"]["fused pair D|real|2|0|bracketed"]
    m_low, m_high = (point["veto_margin"] for point in entry["points"][1:])
    zero = int(2400000 + 3000000 * m_low / (m_low - m_high))
    assert _ask(record, zero - 50).vetoed and not _ask(record, zero + 50).vetoed


# ---------------------------------------------------------------------------
# Degenerate
# ---------------------------------------------------------------------------

def test_a_v2_record_carries_no_cost_model_and_reads_as_before():
    record = _record(_d_curve())
    assert record["schema"] == dp.SCHEMA and not dp.carries_costs(record)
    answer = _ask(record, 24000)
    assert answer.outcome == "no_record"
    assert answer.reason == "no_cost_curve_for_the_class"
    assert answer.reason in dp.COST_SILENCE


def test_a_v2_record_is_never_priced_even_unvalidated_with_a_curve_in_it():
    """The stamp decides, not the presence of a block: a caller that skips validation
    (``validated=True``) on a /2 record carrying a curve still gets no price."""
    smuggled = dict(_record(_d_curve()), costs=_costed(_d_curve())["costs"])
    answer = _ask(smuggled, 4000000, validated=True)
    assert answer.reason == "no_cost_curve_for_the_class"
    assert dp.SCHEMA in answer.evidence["why"]


def test_both_stamps_are_read_and_a_third_is_refused():
    dp.validate(_record(_d_curve()))
    dp.validate(_costed(_d_curve()))
    other = dict(_costed(_d_curve()), schema="meep_gpu.dispatch_preference.timing/4")
    with pytest.raises(dp.RecordRefused) as refused:
        dp.validate(dp.seal(other))
    assert refused.value.code == "unsupported_schema"


def test_a_v2_record_carrying_a_curve_refuses():
    costed = _costed(_d_curve())
    smuggled = dict(_record(_d_curve()), costs=costed["costs"])
    with pytest.raises(dp.RecordRefused) as refused:
        dp.validate(dp.seal(smuggled))
    assert refused.value.code == "malformed_record"


def test_a_curve_its_rows_do_not_give_refuses():
    costed = json.loads(json.dumps(_costed(_d_curve())))
    costed["costs"]["fused pair D|real|2|0|bracketed"]["points"][0][
        "fused_ms_per_step"] = 0.5
    with pytest.raises(dp.RecordRefused) as refused:
        dp.validate(dp.seal(costed))
    assert refused.value.code == "malformed_record"
    assert "do not give" in str(refused.value)


@pytest.mark.parametrize("damage", [
    lambda r: r["keys"][sorted(r["keys"])[0]]["measurements"][0].pop("cells"),
    lambda r: r["keys"][sorted(r["keys"])[0]]["measurements"][0].__setitem__(
        "fused_ms_per_step", "slow"),
    lambda r: r["costs"].__setitem__("fused pair D|real|2|0|bracketed", ["not", "a", "curve"]),
    lambda r: r["costs"]["fused pair D|real|2|0|bracketed"].__setitem__("points", []),
])
def test_a_v3_record_its_curve_cannot_be_read_from_refuses_by_name(damage):
    """Never an exception the dispatch path would read as an internal error."""
    costed = json.loads(json.dumps(_costed(_d_curve())))
    damage(costed)
    with pytest.raises(dp.RecordRefused) as refused:
        dp.validate(dp.seal(costed))
    assert refused.value.code == "malformed_record"


def test_a_cost_rule_this_reader_does_not_know_refuses():
    costed = dict(_costed(_d_curve()))
    costed["cost_model"] = dict(costed["cost_model"], rule="cubic_spline")
    with pytest.raises(dp.RecordRefused) as refused:
        dp.validate(dp.seal(costed))
    assert refused.value.code == "unsupported_cost_model"


def test_a_cost_model_without_a_row_minimum_refuses():
    costed = dict(_costed(_d_curve()))
    costed["rules"] = {}
    with pytest.raises(dp.RecordRefused) as refused:
        dp.validate(dp.seal(costed))
    assert refused.value.code == "malformed_record"


def test_a_class_with_no_measured_key_has_no_curve():
    lonely = _key("fused pair D", _rows(24000, 0.8, 0.6, count=1), status="unmeasured")
    record = _costed([lonely])
    assert record["costs"] == {}
    assert _ask(record, 24000).reason == "no_cost_curve_for_the_class"


def test_a_class_the_cutter_left_unmeasured_has_no_curve_whatever_its_rows():
    """Rows that disagree on what the product displaces are cut as ONE unmeasured key
    over every size, whose ``displaces`` names only the first set: its sizes may carry
    two rows each, and still no curve is priced from them."""
    rows = _rows(24000, 0.80, 0.60) + _rows(48000, 0.82, 0.62)
    mixed = _key("fused pair D", rows, status="unmeasured",
                 reason="the rows disagree on what the product displaces")
    record = _costed([mixed])
    assert record["costs"] == {}
    assert _ask(record, 24000).reason == "no_cost_curve_for_the_class"


def test_a_class_whose_keys_disagree_on_what_they_displace_has_no_curve():
    one, two = _d_curve()
    two["displaces"] = ["conductive PML", "ordinary"]
    assert _costed([one, two])["costs"] == {}


def test_a_size_with_too_few_rows_sets_no_point():
    entries = _d_curve() + [_key("fused pair D", _rows(9000000, 2.0, 2.2, count=1),
                                 status="unmeasured")]
    record = _costed(entries)
    sizes = [p["cells"] for p in record["costs"]["fused pair D|real|2|0|bracketed"]["points"]]
    assert sizes == [24000, 2400000, 5400000]
    assert _ask(record, 9000000).reason == "outside_the_cost_points"


def test_outside_the_measured_sizes_there_is_no_price():
    record = _costed(_d_curve())
    for cells in (23999, 5400001):
        answer = _ask(record, cells)
        assert answer.outcome == "no_record" and answer.reason == "outside_the_cost_points"
    assert _ask(record, 24000).vetoed and not _ask(record, 5400000).vetoed


def test_the_barriers_answer_exactly_as_the_band_consult_answers():
    record = _costed(_d_curve())
    cases = [dict(table="cuda"), dict(capability="8.9"), dict(capability=None),
             dict(repair_route=dict(LIVE_ROUTE, repairs_per_bracketed_step=1.0)),
             dict(subject_sha256={"fused_pairs": "e5" * 32}), dict(subject_sha256={}),
             dict(mode="span")]
    for overrides in cases:
        band = dp.consult(record, table=overrides.get("table", "triton"),
                          capability=overrides.get("capability", CAPABILITY),
                          candidate="fused pair D", displaces=None, run_shape=SHAPE,
                          bracketed=True, mode=overrides.get("mode", "measured"),
                          repair_route=overrides.get("repair_route", LIVE_ROUTE),
                          subject_sha256=overrides.get("subject_sha256", LIVE_EMITTERS),
                          cells=4000000)
        cost = _ask(record, 4000000, **overrides)
        assert band.reason in dp.BARRIER_REASONS, overrides
        assert cost.as_dict() == band.as_dict(), overrides


def test_a_baseline_that_is_not_what_the_curve_priced_has_no_price():
    record = _costed(_d_curve())
    answer = _ask(record, 4000000, displaces=["conductive PML", "ordinary"])
    assert answer.reason == "baseline_is_not_what_the_candidate_displaces"
    assert _ask(record, 4000000, displaces=["ordinary", "PML"]).vetoed


def test_an_unknown_mode_is_refused_by_name():
    with pytest.raises(dp.ConsultRefused):
        _ask(_costed(_d_curve()), 24000, mode="fastest")


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------

def test_add_costs_is_idempotent_and_round_trips():
    once = _costed(_d_curve())
    assert dp.add_costs(once) == once
    assert json.loads(dp.dumps(once)) == once
    assert dp.record_digest(once) == once[dp.DIGEST_FIELD]
    assert once["cost_model"] == {"rule": dp.COST_RULE, "min_rows_per_point": 2}


def test_the_cost_verdict_serializes():
    answer = _ask(_costed(_d_curve()), 4000000)
    assert json.loads(json.dumps(answer.as_dict()))["evidence"]["cost_model"] == dp.COST_RULE


# ---------------------------------------------------------------------------
# Wired at the composer rung
# ---------------------------------------------------------------------------

def _write_costed(tmp_path, entries, **kwargs):
    path = _write(tmp_path, entries, **kwargs)
    with open(path) as handle:
        record = json.load(handle)
    path.write_text(dp.dumps(dp.add_costs(record)), encoding="utf-8")
    return path


GRID_4M = (2000, 2000, 1)
GRID_5M = (2500, 2000, 1)


def test_the_cost_model_vetoes_where_the_bands_are_silent(monkeypatch, tmp_path, capsys):
    """4,000,000 cells lies between a veto band and a keep band: a /2 record is silent
    there and the prior, above its ceiling, keeps the product; the curve says slower."""
    _write(tmp_path, _d_curve())
    _plan, banded = _freeze(monkeypatch, Composer(), shape=GRID_4M)
    assert _arms(banded)["step_D"] == "fused pair D"
    assert _candidate(banded, "fused pair D")["why_not_applied"] == (
        "prior_above_the_recorded_crossover")
    _write_costed(tmp_path, _d_curve())
    _plan, costed = _freeze(monkeypatch, Composer(), shape=GRID_4M)
    assert _arms(costed)["step_D"] == "PML" and _arms(costed)["update_E"] == "ordinary"
    item = _candidate(costed, "fused pair D")
    assert item["applied"] is True and item["rule"] == "cost_model"
    assert item["verdict"]["reason"] == "predicted_slower_beyond_spread"
    assert item["verdict"]["evidence"]["displaces_checked"] is True
    assert item["band_verdict"]["reason"] == "outside_the_measured_bands"
    assert costed["preference"]["records"]["triton"]["cost_model"] == dp.COST_RULE
    note = costed["preference"]["note"]
    assert "predicts it at" in note and "between its 2,400,000- and 5,400,000-cell" in note
    assert note in capsys.readouterr().err


def test_the_cost_model_keeps_where_it_predicts_no_loss(monkeypatch, tmp_path):
    _write_costed(tmp_path, _d_curve())
    _plan, record = _freeze(monkeypatch, Composer(), shape=GRID_5M)
    assert _arms(record)["step_D"] == "fused pair D"
    item = _candidate(record, "fused pair D")
    assert item["rule"] == "cost_model" and item["applied"] is False
    assert item["why_not_applied"] == "predicted_not_slower_beyond_spread"
    assert record["preference"]["compositions"] == 1


def test_a_clean_product_is_priced_by_size_too(monkeypatch, tmp_path):
    """Every product the record measured, not only the bracketed ones: a clean pair
    measured slower at both of its sizes gives way between them, where a /2 record's
    single-size keys and the prior (which never answers for a clean seam) keep it."""
    clean = [_key("fused pair D", _rows(24000, 0.70, 0.60), bracketed=False),
             _key("fused pair D", _rows(5400000, 1.40, 1.20), bracketed=False)]
    _write(tmp_path, clean)
    _plan, banded = _freeze(monkeypatch, Composer(bracketed=()), shape=GRID_4M)
    assert _arms(banded)["step_D"] == "fused pair D"
    _write_costed(tmp_path, clean)
    _plan, costed = _freeze(monkeypatch, Composer(bracketed=()), shape=GRID_4M)
    assert _arms(costed)["step_D"] == "PML"
    item = _candidate(costed, "fused pair D")
    assert item["bracketed"] is False and item["rule"] == "cost_model"


def test_where_the_curve_is_silent_the_bands_and_the_prior_answer(monkeypatch, tmp_path):
    _write_costed(tmp_path, _d_curve())
    _plan, record = _freeze(monkeypatch, Composer(), shape=(40, 30, 1))
    item = _candidate(record, "fused pair D")
    assert item["cost_silence"] == "outside_the_cost_points"
    assert item["rule"] == "portable_prior" and item["applied"] is True
    assert item["record_silence"] == "outside_the_measured_bands"
    assert "band_verdict" not in item


def test_the_confirmation_re_asks_the_curve_with_the_singles_that_took_the_slots(
        monkeypatch, tmp_path):
    _write_costed(tmp_path, _d_curve())
    _plan, record = _freeze(monkeypatch, Composer(singles=("conductive PML", "ordinary")),
                            shape=GRID_4M)
    assert _arms(record)["step_D"] == "fused pair D"
    assert _candidate(record, "fused pair D")["why_not_applied"] == (
        "the record does not price these singles: "
        "baseline_is_not_what_the_candidate_displaces")


def test_a_cost_veto_never_hands_a_slot_to_the_array_path(monkeypatch, tmp_path):
    _write_costed(tmp_path, _d_curve())
    _plan, record = _freeze(monkeypatch, Composer(singles=None, pairs=("D",)),
                            shape=GRID_4M)
    assert _arms(record)["step_D"] == "fused pair D"
    assert _candidate(record, "fused pair D")["why_not_applied"].startswith(
        "withholding it left a slot on the array path")


def test_certified_scope_reads_the_cost_model_as_the_default_does(monkeypatch, tmp_path):
    _write_costed(tmp_path, _d_curve())
    monkeypatch.setenv(fastpath.HARDWARE_SWITCH, "certified")
    plan, record = _freeze(monkeypatch, Composer(), shape=GRID_4M)
    assert plan is not None, record["refused_because"]
    assert _arms(record)["step_D"] == "PML" and record["certified"] is True


def test_an_uncertified_device_with_a_record_of_its_own_is_priced_and_never_refused(
        monkeypatch, tmp_path):
    """The fast path is not gated on certification (fast everywhere): a device no
    certification run reached reads its own record's cost model like any other."""
    _write_costed(tmp_path, _d_curve(), capability="8.9")
    plan, record = _freeze(monkeypatch, Composer(), capability=(8, 9), shape=GRID_4M)
    assert plan is not None, record["refused_because"]
    assert record["decision"] == "dispatched"
    assert _arms(record)["step_D"] == "PML"
    assert _candidate(record, "fused pair D")["rule"] == "cost_model"


def test_span_reads_no_cost_model(monkeypatch, tmp_path):
    _write_costed(tmp_path, _d_curve())
    monkeypatch.setenv(fastpath.PREFERENCE_SWITCH, "span")
    _plan, record = _freeze(monkeypatch, Composer(), shape=GRID_4M)
    assert _arms(record)["step_D"] == "fused pair D"
    assert record["preference"]["records"] == {}


def test_a_curve_another_architecture_measured_never_prices_this_one(monkeypatch, tmp_path):
    """Cost is the architecture's: an unrecorded device reads no curve, only the prior."""
    _write_costed(tmp_path, _d_curve())
    _plan, record = _freeze(monkeypatch, Composer(), capability=(8, 9), shape=GRID_4M)
    item = _candidate(record, "fused pair D")
    assert item["rule"] == "portable_prior" and "band_verdict" not in item
    assert _arms(record)["step_D"] == "fused pair D"  # above the prior's ceiling


def test_a_v3_record_that_cannot_be_applied_is_staleness_not_a_price(monkeypatch,
                                                                     tmp_path):
    """Another repair route: the barrier answers, no cost is read, and no prior."""
    _write_costed(tmp_path, _d_curve())
    _plan, record = _freeze(monkeypatch, Composer(), shape=GRID_4M,
                            route={"repairs_per_bracketed_step": 3.0, "restricted": False})
    item = _candidate(record, "fused pair D")
    assert item["why_not_applied"] == "record_is_for_another_repair_route"
    assert not {"cost_silence", "band_verdict", "record_silence"} & set(item)
    assert item["rule"] == "record"


def test_a_v2_records_candidates_carry_nothing_new(monkeypatch, tmp_path):
    _write(tmp_path, _d_curve())
    _plan, record = _freeze(monkeypatch, Composer(), shape=GRID_4M)
    for item in record["preference"]["candidates"]:
        assert not {"cost_silence", "band_verdict"} & set(item)
        assert item["rule"] in ("record", "portable_prior")
    assert "cost_model" not in record["preference"]["records"]["triton"]


def test_a_refused_v3_record_is_loud_and_its_table_takes_the_prior(monkeypatch, tmp_path):
    path = _write_costed(tmp_path, _d_curve())
    record = json.loads(path.read_text())
    record["costs"]["fused pair D|real|2|0|bracketed"]["points"].pop()
    path.write_text(dp.dumps(dp.seal(record)), encoding="utf-8")
    plan, frozen = _freeze(monkeypatch, Composer())
    assert plan is not None
    assert frozen["preference"]["records"]["triton"]["status"] == "refused"
    assert frozen["preference"]["rule_by_table"] == {"triton": "portable_prior"}


# ---------------------------------------------------------------------------
# Realistic: the shipped records, re-stamped /3
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("table", ["triton", "cuda"])
def test_the_shipped_records_restamped_price_the_sizes_their_rows_measured(table):
    path = dp.record_path(table, "8.6")
    shipped = dp.load_record(path)
    costed = dp.add_costs(shipped)
    dp.validate(costed, expect_table=table)
    assert costed["keys"] == shipped["keys"]
    assert costed["costs"], "every shipped record measures at least one class"
    for name, entry in costed["costs"].items():
        sizes = [point["cells"] for point in entry["points"]]
        assert sizes == sorted(set(sizes))
        measured = {row["cells"] for key, value in shipped["keys"].items()
                    if key.rsplit("|", 1)[0] == name
                    for row in value.get("measurements") or ()}
        assert set(sizes) <= measured
        for point in entry["points"]:
            assert point["rows"] >= shipped["rules"]["min_rows_per_key"]
