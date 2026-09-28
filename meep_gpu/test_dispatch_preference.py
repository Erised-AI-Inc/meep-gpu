"""Host tests for the dispatch-preference consult and its timing-record format.

Written before ``dispatch_preference.py`` (the repository's test-first rule for a new
decision engine). Five groups, in the order that rule names them:

* KNOWN VALUE -- a hand-built record vetoes exactly the candidate it measured slower
  beyond its spreads, and nothing else;
* DEGENERATE -- no record, an empty one, a missing key, a key marked unmeasured, a
  ratio inside the spread, a record for another table, a record cut on another repair
  route (the WHOLE route identity, not its digest alone), a record cut on another
  fused-pair emitter, a live route or emitter the caller cannot state, a baseline that
  is not what the candidate would displace, and every malformed record refusing BY NAME;
* SCALING -- the band lookup at, below and above each edge, and monotone in cells;
* SHAPE / SERIALIZATION -- round trip, deterministic bytes, a verdict that serializes;
* REALISTIC -- the records shipped beside the kernel tables assert the three facts the
  fused-product benchmark document states.

The consult is UNWIRED: the last group of structural tests pins that no dispatch
module imports it and that it reads no environment variable, which is the
environment-variable contract in this package's ``the development notes``.
"""

from __future__ import annotations

import ast
import copy
import json
import os

import pytest

from meep_gpu import dispatch_preference as dp

HERE = os.path.dirname(os.path.abspath(__file__))

ROUTE = "a" * 64
OTHER_ROUTE = "b" * 64
PAIRS_SHA = "c3" * 32
OTHER_PAIRS_SHA = "d4" * 32

#: The live route facts a caller states: the digest AND what the bracket did per step.
#: One digest carries two routes (the corpus has both on ``e4f88c1e``), so the consult
#: compares the whole identity and not the digest alone.
LIVE_ROUTE = {"deposit_repair_sha256": ROUTE, "index": "linear",
              "repairs_per_bracketed_step": 3.0, "restricted": False,
              "fell_back": False}
LIVE_EMITTERS = {"fused_pairs": PAIRS_SHA}


# ---------------------------------------------------------------------------
# A hand-built record
# ---------------------------------------------------------------------------

def _measurement(case, cells, fused, baseline, co_fused=()):
    return {
        "artifact": f"results/example/{case}/rows.jsonl",
        "line": 1,
        "baseline_ms_per_step": baseline,
        "baseline_spread": 0.01,
        "case": case,
        "cells": cells,
        "co_fused": [dict(item) for item in co_fused],
        "fused_ms_per_step": fused,
        "fused_spread": 0.02,
        "grid_shape": [cells, 1, 1],
        "ratio": round(baseline / fused, 6),
        "source": "bench",
        "utc": "2026-09-19T13:02:20Z",
    }


def _entry(product, *, bracketed, lo, hi, fused, baseline, displaces,
           storage="real", dimensions=2, susceptibilities=0, status="measured",
           reason=None, rows=2, more_rows=()):
    """One key. ``more_rows`` adds ``(fused, baseline)`` measurements beside the first."""
    measurements = [_measurement(f"example_{index}", lo, row_fused, row_baseline)
                    for index, (row_fused, row_baseline)
                    in enumerate(((fused, baseline),) + tuple(more_rows))]
    ratios = sorted(row["ratio"] for row in measurements)
    margin = dp.least_margin(measurements)
    entry = {
        "status": status,
        "product": product,
        "storage": storage,
        "dimensions": dimensions,
        "susceptibilities": susceptibilities,
        "bracketed": bracketed,
        "band": {"lo_cells": lo, "hi_cells": hi},
        "slots": ["step_D", "update_E"],
        "displaces": list(displaces),
        "rows": rows,
        "fused_ms_per_step": fused,
        "baseline_ms_per_step": baseline,
        "ratio": ratios[0],
        "ratio_min": ratios[0],
        "ratio_max": ratios[-1],
        "fused_spread": 0.02,
        "baseline_spread": 0.01,
        "veto_margin": margin,
        "slower_beyond_spread": margin > 0.0,
        "measurements": measurements,
    }
    if status != "measured":
        entry["reason"] = reason or "1 row, minimum 2"
    return entry


def _record(entries, table="triton", route=ROUTE, barrier=None, **route_facts):
    keys = {}
    for entry in entries:
        keys[dp.key_string(entry["product"], entry["storage"], entry["dimensions"],
                           entry["susceptibilities"], entry["bracketed"],
                           entry["band"]["lo_cells"], entry["band"]["hi_cells"])] = entry
    block = dict(LIVE_ROUTE, deposit_repair_sha256=route, **route_facts)
    body = {
        "schema": dp.SCHEMA,
        "table": table,
        "recorded_utc": "2026-09-19T13:02:20Z",
        "route": dict(block, id=dp.route_id(block)),
        "subject_barrier": barrier if barrier is not None else {
            "sources": ["fused_pairs"], "digests": {"fused_pairs": [PAIRS_SHA]},
            "rows_not_recording": {"fused_pairs": 0}, "allowed_unpinned": []},
        "rules": {"min_rows_per_key": 2},
        "keys": keys,
    }
    return dp.seal(body)


SLOW_PAIR = dict(product="fused pair D", bracketed=True, lo=1000, hi=100000,
                 fused=1.40, baseline=0.50, displaces=("PML", "ordinary"))
FAST_PAIR = dict(product="fused pair B", bracketed=False, lo=1000, hi=100000,
                 fused=0.45, baseline=0.49, displaces=("PML", "ordinary"))


def _shape(cells=24000, dimensions=2, complex_storage=False, susceptibilities=0):
    return {"dimensions": dimensions, "grid_shape": [cells, 1, 1],
            "complex_storage": complex_storage, "susceptibilities": susceptibilities}


AS_TIMED = object()          # "what the record was cut on", so None can mean None


def _ask(record, candidate, *, bracketed, displaces=("PML", "ordinary"), table="triton",
         mode="measured", route=ROUTE, live_route=AS_TIMED, emitters=AS_TIMED, **shape):
    if live_route is AS_TIMED:
        live_route = dict(LIVE_ROUTE, deposit_repair_sha256=route)
    return dp.consult(record, table=table, candidate=candidate, displaces=displaces,
                      run_shape=_shape(**shape), bracketed=bracketed, mode=mode,
                      repair_route=live_route,
                      subject_sha256=dict(LIVE_EMITTERS) if emitters is AS_TIMED
                      else emitters)


# ---------------------------------------------------------------------------
# Known value
# ---------------------------------------------------------------------------

def test_a_hand_built_record_vetoes_exactly_the_slower_candidate():
    record = _record([_entry(**SLOW_PAIR), _entry(**FAST_PAIR)])
    slow = _ask(record, "fused pair D", bracketed=True)
    fast = _ask(record, "fused pair B", bracketed=False)
    assert slow.outcome == "vetoed" and slow.vetoed
    assert fast.outcome == "not_vetoed" and not fast.vetoed
    # the SAME product on the other side of the bracket axis has no record at all
    assert _ask(record, "fused pair D", bracketed=False).outcome == "no_record"
    assert _ask(record, "fused pair B", bracketed=True).outcome == "no_record"


def test_the_verdict_carries_the_key_and_the_numbers_it_was_decided_on():
    record = _record([_entry(**SLOW_PAIR)])
    verdict = _ask(record, "fused pair D", bracketed=True)
    assert verdict.key == "fused pair D|real|2|0|bracketed|1000-100000"
    evidence = verdict.evidence
    assert evidence["fused_ms_per_step"] == 1.40
    assert evidence["baseline_ms_per_step"] == 0.50
    assert evidence["ratio_max"] == pytest.approx(0.357143)
    assert evidence["veto_margin"] == pytest.approx(1.0 - 0.03 - 0.357143)
    assert evidence["cells"] == 24000
    assert evidence["band"] == {"lo_cells": 1000, "hi_cells": 100000}
    assert evidence["artifacts"] == ["results/example/example_0/rows.jsonl"]
    assert evidence["cases"] == ["example_0"]
    assert evidence["record_sha256"] == record["record_sha256"]
    line = verdict.describe()
    assert "vetoed" in line and "fused pair D" in line and "0.357" in line


def test_a_row_is_slower_only_beyond_both_of_its_own_spreads():
    assert dp.slower_beyond_spread(0.96, 0.02, 0.01)          # 0.96 < 0.97
    assert not dp.slower_beyond_spread(0.97, 0.02, 0.01)      # on the threshold
    assert not dp.slower_beyond_spread(0.98, 0.02, 0.01)      # inside the spread
    assert not dp.slower_beyond_spread(1.20, 0.02, 0.01)      # faster
    assert dp.veto_margin(0.96, 0.02, 0.01) == pytest.approx(0.01)
    assert dp.veto_margin(1.20, 0.02, 0.01) == pytest.approx(-0.23)


def test_a_key_vetoes_only_when_every_row_in_it_is_slower():
    """Row by row, never on pooled extremes.

    A band pooling a 3x loss with a near-tie is not vetoed, and one noisy row's spread
    does not excuse a quiet row's loss: each row is read against its OWN spreads.
    """
    every = _entry(**SLOW_PAIR, more_rows=((1.30, 0.55), (1.50, 0.60)))
    assert every["veto_margin"] == pytest.approx(0.97 - 0.55 / 1.30, abs=1e-6)
    assert _ask(_record([every]), "fused pair D", bracketed=True).vetoed
    one_tie = _entry(**SLOW_PAIR, more_rows=((1.00, 0.99),))
    assert one_tie["veto_margin"] < 0.0
    verdict = _ask(_record([one_tie]), "fused pair D", bracketed=True)
    assert verdict.outcome == "not_vetoed"
    assert verdict.evidence["ratio_min"] == pytest.approx(0.357143)
    assert verdict.evidence["ratio_max"] == pytest.approx(0.99)


# ---------------------------------------------------------------------------
# Degenerate
# ---------------------------------------------------------------------------

def test_no_record_and_an_empty_record_leave_the_longer_span_winning():
    for record in (None, _record([])):
        verdict = _ask(record, "fused pair D", bracketed=True)
        assert verdict.outcome == "no_record" and not verdict.vetoed
    assert _ask(None, "fused pair D", bracketed=True).reason == "no_record_supplied"
    assert _ask(_record([]), "fused pair D",
                bracketed=True).reason == "class_not_in_record"


def test_a_missing_key_is_no_record_on_every_axis():
    record = _record([_entry(**SLOW_PAIR)])
    assert _ask(record, "fused pair D (folded)", bracketed=True).outcome == "no_record"
    assert _ask(record, "fused pair D", bracketed=True,
                complex_storage=True).outcome == "no_record"
    assert _ask(record, "fused pair D", bracketed=True,
                dimensions=3).outcome == "no_record"
    assert _ask(record, "fused pair D", bracketed=True,
                susceptibilities=1).outcome == "no_record"


def test_a_key_marked_unmeasured_is_no_record_and_says_why():
    entry = _entry(**SLOW_PAIR, status="unmeasured", reason="1 row, minimum 2", rows=1)
    verdict = _ask(_record([entry]), "fused pair D", bracketed=True)
    assert verdict.outcome == "no_record"
    assert verdict.reason == "key_unmeasured"
    assert verdict.evidence["unmeasured_because"] == "1 row, minimum 2"
    # the numbers of the row that did land still travel with the verdict
    assert verdict.evidence["ratio_max"] == pytest.approx(0.357143)


def test_a_ratio_inside_the_spread_is_not_vetoed():
    inside = _entry(**dict(SLOW_PAIR, fused=1.00, baseline=0.98))
    verdict = _ask(_record([inside]), "fused pair D", bracketed=True)
    assert verdict.outcome == "not_vetoed"
    assert verdict.evidence["ratio_max"] == pytest.approx(0.98)


def test_a_record_for_another_table_is_never_applied():
    record = _record([_entry(**SLOW_PAIR)], table="cuda")
    verdict = _ask(record, "fused pair D", bracketed=True, table="triton")
    assert verdict.outcome == "no_record"
    assert verdict.reason == "record_is_for_another_table"
    with pytest.raises(dp.RecordRefused) as refusal:
        dp.validate(record, expect_table="triton")
    assert refusal.value.code == "wrong_table"


def test_a_record_cut_on_another_repair_route_is_never_applied():
    record = _record([_entry(**SLOW_PAIR)])
    verdict = _ask(record, "fused pair D", bracketed=True, route=OTHER_ROUTE)
    assert verdict.outcome == "no_record"
    assert verdict.reason == "record_is_for_another_repair_route"
    assert verdict.evidence["record_route"] == ROUTE
    assert verdict.evidence["live_route"] == OTHER_ROUTE
    assert verdict.evidence["differs_on"] == ["deposit_repair_sha256"]


def test_one_digest_carries_two_routes_and_the_consult_tells_them_apart():
    """The corpus's own hazard: ``e4f88c1e`` ran 3-per-step full AND 1-per-step
    restricted. The bracket is the dominant term of every measured veto, so a record
    cut on one of those routes says nothing about a tree running the other."""
    record = _record([_entry(**SLOW_PAIR)])
    assert _ask(record, "fused pair D", bracketed=True).vetoed
    for fact, value, spelling in (("repairs_per_bracketed_step", 1.0, "1-per-step"),
                                  ("restricted", True, "restricted"),
                                  ("fell_back", True, "fell-back"),
                                  ("index", "3-tuple", "3-tuple")):
        live = dict(LIVE_ROUTE, **{fact: value})
        verdict = _ask(record, "fused pair D", bracketed=True, live_route=live)
        assert verdict.outcome == "no_record", (fact, verdict.as_dict())
        assert verdict.reason == "record_is_for_another_repair_route"
        assert verdict.evidence["differs_on"] == [fact]
        assert spelling in verdict.evidence["live_route_id"]
        assert verdict.evidence["record_route_id"] == record["route"]["id"]


def test_a_live_route_the_caller_cannot_state_is_no_record():
    record = _record([_entry(**SLOW_PAIR)])
    for live in (None, {}, dict(LIVE_ROUTE, repairs_per_bracketed_step=None),
                 dict(LIVE_ROUTE, restricted=None), "e4f88c1e"):
        verdict = _ask(record, "fused pair D", bracketed=True, live_route=live)
        assert verdict.outcome == "no_record", live
        assert verdict.reason == "live_repair_route_not_stated", live


def test_a_clean_key_is_asked_only_for_the_repair_digest():
    """A clean product is priced only by rows with no bracket anywhere in the plan
    (the cutter's attribution rule), so its number holds no bracket term: what the
    bracket did per step on the live tree cannot move it, and the digest still must
    match, because the module is in the step either way."""
    record = _record([_entry(**FAST_PAIR)])
    live = dict(LIVE_ROUTE, repairs_per_bracketed_step=1.0, restricted=True)
    assert _ask(record, "fused pair B", bracketed=False,
                live_route=live).outcome == "not_vetoed"
    assert _ask(record, "fused pair B", bracketed=False,
                live_route=dict(live, deposit_repair_sha256=OTHER_ROUTE)
                ).reason == "record_is_for_another_repair_route"
    # the digest alone carries a clean key ...
    assert _ask(record, "fused pair B", bracketed=False,
                live_route={"deposit_repair_sha256": ROUTE}
                ).outcome == "not_vetoed"
    # ... and is not enough for a BRACKETED one, which names what it was not told
    bracketed = _ask(_record([_entry(**SLOW_PAIR)]), "fused pair D", bracketed=True,
                     live_route={"deposit_repair_sha256": ROUTE})
    assert bracketed.reason == "live_repair_route_not_stated"
    assert bracketed.evidence["not_stated"] == [       # in ROUTE_FACTS order
        "index", "repairs_per_bracketed_step", "restricted", "fell_back"]


def test_a_record_cut_on_another_fused_pair_emitter_is_never_applied():
    """Plan step 3 is 'repair the kernel, re-time, let it back in on merit'. Until the
    re-cut, the old record must not veto the repaired kernel."""
    record = _record([_entry(**SLOW_PAIR)])
    assert _ask(record, "fused pair D", bracketed=True).vetoed
    verdict = _ask(record, "fused pair D", bracketed=True,
                   emitters={"fused_pairs": OTHER_PAIRS_SHA})
    assert verdict.outcome == "no_record"
    assert verdict.reason == "record_is_for_another_emitter"
    assert verdict.evidence["differs_on"] == ["fused_pairs"]
    assert verdict.evidence["record_emitters"]["fused_pairs"] == PAIRS_SHA
    assert verdict.evidence["live_emitters"]["fused_pairs"] == OTHER_PAIRS_SHA


def test_an_emitter_the_caller_cannot_state_is_no_record():
    record = _record([_entry(**SLOW_PAIR)])
    for live in (None, {}, {"fused_pairs": None}, {"fused_polarization_pair": PAIRS_SHA}):
        verdict = _ask(record, "fused pair D", bracketed=True, emitters=live)
        assert verdict.outcome == "no_record", live
        assert verdict.reason == "live_emitters_not_stated", live
        assert verdict.evidence["not_stated"] == ["fused_pairs"]


def test_an_emitter_the_record_could_not_pin_is_named_and_not_compared():
    """A source some admitted rows recorded and others did not cannot be compared, and
    the record says so by name rather than pooling the two as one value."""
    barrier = {"sources": ["fused_pairs", "fused_polarization_pair"],
               "digests": {"fused_pairs": [PAIRS_SHA],
                           "fused_polarization_pair": [OTHER_PAIRS_SHA,
                                                       dp.NOT_RECORDED]},
               "rows_not_recording": {"fused_pairs": 0, "fused_polarization_pair": 17},
               "allowed_unpinned": ["fused_polarization_pair"]}
    record = _record([_entry(**SLOW_PAIR)], barrier=barrier)
    assert dp.compared_emitters(record) == {"fused_pairs": PAIRS_SHA}
    verdict = _ask(record, "fused pair D", bracketed=True)
    assert verdict.vetoed
    assert verdict.evidence["emitters_compared"] == ["fused_pairs"]
    assert verdict.evidence["emitters_not_compared"] == ["fused_polarization_pair"]


def test_the_live_emitter_digests_are_read_off_the_tree_by_one_function():
    live = dp.live_subject_sha256("cuda")
    assert sorted(live) == ["fused_pairs", "fused_polarization_pair"]
    assert sorted(dp.live_subject_sha256("triton")) == ["fused_pairs"]
    for source, digest in live.items():
        path = os.path.join(HERE, dp.SUBJECT_FILES[source])
        assert digest == dp.file_sha256(path), source


def test_a_baseline_that_is_not_what_the_candidate_displaces_is_no_record():
    record = _record([_entry(**SLOW_PAIR)])
    verdict = _ask(record, "fused pair D", bracketed=True,
                   displaces=("PML", "off-diagonal"))
    assert verdict.outcome == "no_record"
    assert verdict.reason == "baseline_is_not_what_the_candidate_displaces"
    # order is the slot order of the caller's span and carries no meaning here
    assert _ask(record, "fused pair D", bracketed=True,
                displaces=("ordinary", "PML")).outcome == "vetoed"


def _broken(mutate):
    record = _record([_entry(**SLOW_PAIR)])
    body = copy.deepcopy(record)
    mutate(body)
    return body


@pytest.mark.parametrize("mutate, code", [
    (lambda r: r.pop("keys"), "malformed_record"),
    (lambda r: r.pop("table"), "malformed_record"),
    (lambda r: r.pop("route"), "malformed_record"),
    (lambda r: r.__setitem__("keys", []), "malformed_record"),
    (lambda r: r.__setitem__("schema", "something/else"), "unsupported_schema"),
    (lambda r: r.__setitem__("table", "opencl"), "malformed_record"),
    (lambda r: r.__setitem__("recorded_utc", "yesterday"), "malformed_record"),
    # THE ROUTE IS AN IDENTITY, NOT A DIGEST: every fact of it is validated, and its
    # id is tied to those facts so a record cannot be handed a route it did not run.
    (lambda r: r["route"].pop("index"), "malformed_record"),
    (lambda r: r["route"].pop("repairs_per_bracketed_step"), "malformed_record"),
    (lambda r: r["route"].pop("id"), "malformed_record"),
    (lambda r: r["route"].__setitem__("id", "e4f88c1e4fc8:linear:1-per-step:full"),
     "malformed_record"),
    (lambda r: r["route"].__setitem__("restricted", "no"), "malformed_record"),
    (lambda r: r["route"].__setitem__("index", "sideways"), "malformed_record"),
    (lambda r: r.pop("subject_barrier"), "malformed_record"),
    (lambda r: r["subject_barrier"].__setitem__("sources", "fused_pairs"),
     "malformed_record"),
    (lambda r: r["subject_barrier"]["digests"].pop("fused_pairs"), "malformed_record"),
])
def test_a_malformed_record_refuses_by_name(mutate, code):
    body = _broken(mutate)
    body.pop("record_sha256", None)
    with pytest.raises(dp.RecordRefused) as refusal:
        dp.validate(dp.seal(body))
    assert refusal.value.code == code


@pytest.mark.parametrize("field, value", [
    ("status", "probably"),
    ("bracketed", "yes"),
    ("band", {"lo_cells": 10, "hi_cells": 5}),
    ("fused_ms_per_step", -1.0),
    ("ratio_max", None),
    ("slower_beyond_spread", False),      # disagrees with its own measurements
    ("veto_margin", 0.5),                 # so does this
    ("measurements", []),                 # numbers with no row behind them
    ("displaces", "PML"),
    ("product", "fused pair B"),          # disagrees with its own key
])
def test_a_malformed_entry_refuses_by_name(field, value):
    record = _record([_entry(**SLOW_PAIR)])
    body = copy.deepcopy(record)
    body.pop("record_sha256")
    next(iter(body["keys"].values()))[field] = value
    with pytest.raises(dp.RecordRefused) as refusal:
        dp.validate(dp.seal(body))
    assert refusal.value.code == "malformed_record"
    assert field in refusal.value.detail or "key" in refusal.value.detail


def test_overlapping_bands_in_one_class_refuse_by_name():
    first = _entry(**SLOW_PAIR)
    second = _entry(**dict(SLOW_PAIR, lo=50000, hi=200000))
    with pytest.raises(dp.RecordRefused) as refusal:
        dp.validate(_record([first, second]))
    assert refusal.value.code == "malformed_record"
    assert "overlap" in refusal.value.detail


def test_an_edited_record_refuses_by_name():
    record = _record([_entry(**SLOW_PAIR)])
    next(iter(record["keys"].values()))["fused_ms_per_step"] = 0.10
    with pytest.raises(dp.RecordRefused) as refusal:
        dp.validate(record)
    assert refusal.value.code == "record_digest_mismatch"
    unsealed = _record([_entry(**SLOW_PAIR)])
    unsealed.pop("record_sha256")
    with pytest.raises(dp.RecordRefused) as refusal:
        dp.validate(unsealed)
    assert refusal.value.code == "record_digest_mismatch"


def test_the_consult_validates_what_it_is_handed():
    record = _record([_entry(**SLOW_PAIR)])
    record["table"] = "cuda"          # edited after sealing
    with pytest.raises(dp.RecordRefused):
        _ask(record, "fused pair D", bracketed=True)


def test_an_unknown_mode_refuses_by_name():
    with pytest.raises(dp.ConsultRefused) as refusal:
        _ask(_record([_entry(**SLOW_PAIR)]), "fused pair D", bracketed=True,
             mode="fastest")
    assert refusal.value.code == "unknown_mode"


def test_span_mode_is_todays_behaviour_and_touches_nothing(tmp_path):
    record = _record([_entry(**SLOW_PAIR)])
    assert _ask(record, "fused pair D", bracketed=True).vetoed
    span = _ask(record, "fused pair D", bracketed=True, mode="span")
    assert span.outcome == "no_record" and not span.vetoed
    assert span.reason == "span_mode_record_not_consulted"
    # no I/O in span mode: a path that does not exist, and a file that is not a record
    missing = str(tmp_path / "absent" / "timing.json")
    garbage = tmp_path / "timing.json"
    garbage.write_text("{ not json")
    for path in (missing, str(garbage)):
        verdict = dp.consult_path(path, table="triton", candidate="fused pair D",
                                  displaces=("PML", "ordinary"), run_shape=_shape(),
                                  bracketed=True, mode="span",
                                  repair_route=LIVE_ROUTE,
                                  subject_sha256=LIVE_EMITTERS)
        assert verdict.outcome == "no_record" and not verdict.vetoed
    # span mode does not even validate a record handed to it directly
    assert not _ask({"not": "a record"}, "fused pair D", bracketed=True,
                    mode="span").vetoed


def test_measured_mode_reads_the_path_it_is_given(tmp_path):
    path = tmp_path / "timing.json"
    path.write_text(dp.dumps(_record([_entry(**SLOW_PAIR)])))
    asked = dict(table="triton", candidate="fused pair D", displaces=("PML", "ordinary"),
                 run_shape=_shape(), bracketed=True, mode="measured",
                 repair_route=LIVE_ROUTE, subject_sha256=LIVE_EMITTERS)
    assert dp.consult_path(str(path), **asked).vetoed
    absent = dp.consult_path(str(tmp_path / "none.json"), **asked)
    assert absent.outcome == "no_record" and absent.reason == "no_record_supplied"
    path.write_text("{ not json")
    with pytest.raises(dp.RecordRefused) as refusal:
        dp.consult_path(str(path), **asked)
    assert refusal.value.code == "unreadable_record"


# ---------------------------------------------------------------------------
# Scaling: the band lookup
# ---------------------------------------------------------------------------

def _banded():
    slow = _entry(**SLOW_PAIR)                                     # [1000, 100000]
    gap = _entry(**dict(SLOW_PAIR, lo=100001, hi=499999), status="unmeasured",
                 reason="the verdict changes between 100000 and 500000 cells and no "
                        "row lies between them", rows=0)
    fast = _entry(**dict(SLOW_PAIR, lo=500000, hi=500000, fused=2.0, baseline=2.2))
    return _record([slow, gap, fast])


@pytest.mark.parametrize("cells, outcome, reason", [
    (999, "no_record", "outside_the_measured_bands"),
    (1000, "vetoed", "slower_beyond_spread"),
    (1001, "vetoed", "slower_beyond_spread"),
    (99999, "vetoed", "slower_beyond_spread"),
    (100000, "vetoed", "slower_beyond_spread"),
    (100001, "no_record", "key_unmeasured"),
    (499999, "no_record", "key_unmeasured"),
    (500000, "not_vetoed", "not_slower_beyond_spread"),
    (500001, "no_record", "outside_the_measured_bands"),
])
def test_the_band_lookup_at_below_and_above_each_edge(cells, outcome, reason):
    verdict = _ask(_banded(), "fused pair D", bracketed=True, cells=cells)
    assert (verdict.outcome, verdict.reason) == (outcome, reason)


def test_nothing_is_extrapolated_past_the_measured_sizes():
    record = _banded()
    for cells in (1, 10, 999, 500001, 10**7, 10**9):
        verdict = _ask(record, "fused pair D", bracketed=True, cells=cells)
        assert verdict.outcome == "no_record"
        assert verdict.evidence["measured_bands"] == [
            [1000, 100000], [100001, 499999], [500000, 500000]]


def test_the_band_lookup_is_monotone_in_cells():
    record = _banded()
    sizes = [1, 500, 1000, 5000, 100000, 100001, 300000, 499999, 500000, 500001, 10**8]
    found = []
    for cells in sizes:
        verdict = _ask(record, "fused pair D", bracketed=True, cells=cells)
        band = verdict.evidence.get("band")
        if band is not None:
            assert band["lo_cells"] <= cells <= band["hi_cells"]
            found.append(band["lo_cells"])
    assert found == sorted(found), "a larger run landed in an earlier band"
    assert len(set(found)) == 3


def test_cells_come_from_the_run_shape_by_one_function():
    assert dp.cells_of([200, 120, 1]) == 24000
    assert dp.cells_of((48, 48, 48)) == 110592
    assert dp.cells_of([1, 1, 640]) == 640
    with pytest.raises(dp.ConsultRefused) as refusal:
        dp.cells_of([])
    assert refusal.value.code == "unreadable_run_shape"
    record = _record([_entry(**SLOW_PAIR)])
    shape = {"dimensions": 2, "grid_shape": (200, 120, 1), "complex_storage": False,
             "susceptibilities": 0}
    verdict = dp.consult(record, table="triton", candidate="fused pair D",
                         displaces=("PML", "ordinary"), run_shape=shape, bracketed=True,
                         mode="measured", repair_route=LIVE_ROUTE,
                         subject_sha256=LIVE_EMITTERS)
    assert verdict.vetoed and verdict.evidence["cells"] == 24000


def test_a_run_shape_missing_an_axis_refuses_by_name():
    record = _record([_entry(**SLOW_PAIR)])
    for missing in ("dimensions", "grid_shape"):
        shape = _shape()
        shape.pop(missing)
        with pytest.raises(dp.ConsultRefused) as refusal:
            dp.consult(record, table="triton", candidate="fused pair D",
                       displaces=("PML", "ordinary"), run_shape=shape, bracketed=True,
                       mode="measured", repair_route=LIVE_ROUTE,
                       subject_sha256=LIVE_EMITTERS)
        assert refusal.value.code == "unreadable_run_shape"


# ---------------------------------------------------------------------------
# Shape / serialization
# ---------------------------------------------------------------------------

def test_a_record_round_trips_through_its_own_bytes(tmp_path):
    record = _banded()
    text = dp.dumps(record)
    assert json.loads(text) == record
    path = tmp_path / "timing.json"
    path.write_text(text)
    loaded = dp.load_record(str(path), expect_table="triton")
    assert loaded == record
    assert dp.dumps(loaded) == text


def test_the_bytes_do_not_depend_on_insertion_order():
    forward = _record([_entry(**SLOW_PAIR), _entry(**FAST_PAIR)])
    backward = _record([_entry(**FAST_PAIR), _entry(**SLOW_PAIR)])
    assert dp.dumps(forward) == dp.dumps(backward)
    assert forward["record_sha256"] == backward["record_sha256"]
    shuffled = {key: forward[key] for key in reversed(list(forward))}
    assert dp.dumps(shuffled) == dp.dumps(forward)
    assert dp.dumps(forward).endswith("\n")


def test_sealing_is_idempotent_and_covers_every_other_field():
    record = _record([_entry(**SLOW_PAIR)])
    assert dp.seal(record)["record_sha256"] == record["record_sha256"]
    moved = copy.deepcopy(record)
    moved["recorded_utc"] = "2026-09-20T00:00:00Z"
    assert dp.seal(moved)["record_sha256"] != record["record_sha256"]


def test_a_verdict_serializes_for_a_route_gate():
    verdict = _ask(_record([_entry(**SLOW_PAIR)]), "fused pair D", bracketed=True)
    payload = verdict.as_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert payload["outcome"] == "vetoed" and payload["mode"] == "measured"
    assert payload["candidate"] == "fused pair D"
    assert payload["displaces"] == ["PML", "ordinary"]
    assert set(payload) == {"outcome", "reason", "mode", "table", "candidate",
                            "displaces", "key", "evidence"}


def test_the_key_spelling_is_one_function_and_reads_back():
    key = dp.key_string("fused pair D (folded dispersive)", "real", 2, 1, True,
                        4880, 203320)
    assert key == "fused pair D (folded dispersive)|real|2|1|bracketed|4880-203320"
    assert dp.parse_key(key) == {
        "product": "fused pair D (folded dispersive)", "storage": "real",
        "dimensions": 2, "susceptibilities": 1, "bracketed": True,
        "band": {"lo_cells": 4880, "hi_cells": 203320}}
    clean = dp.key_string("cuda:fused magnetic pair", "complex", 3, 0, False, 64000, 64000)
    assert clean == "cuda:fused magnetic pair|complex|3|0|clean|64000-64000"
    assert dp.parse_key(clean)["bracketed"] is False
    with pytest.raises(dp.RecordRefused):
        dp.parse_key("fused pair D|real|2|0|sometimes|1-2")


def test_the_route_id_is_one_spelling_and_names_every_fact_it_turns_on():
    full = dp.route_id(LIVE_ROUTE)
    assert full == "aaaaaaaaaaaa:linear:3-per-step:full"
    assert dp.route_id(dict(LIVE_ROUTE, restricted=True,
                            repairs_per_bracketed_step=1.0)) == (
        "aaaaaaaaaaaa:linear:1-per-step:restricted")
    assert dp.route_id(dict(LIVE_ROUTE, fell_back=True)).endswith(":fell-back")
    # a program in which nothing was bracketed states no bracket facts and says so
    nothing = {"deposit_repair_sha256": ROUTE, "index": None,
               "repairs_per_bracketed_step": None, "restricted": False,
               "fell_back": False}
    assert dp.route_id(nothing) == "aaaaaaaaaaaa:no-bracketed-row"
    assert dp.route_facts_missing(LIVE_ROUTE) == []
    assert dp.route_facts_missing({"deposit_repair_sha256": ROUTE}) == [
        "fell_back", "index", "repairs_per_bracketed_step", "restricted"]
    assert dp.route_facts_missing(None) == list(dp.ROUTE_FACTS)


def test_the_shape_class_reads_the_axes_the_release_tables_compute():
    shape = dp.shape_class({"dimensions": 3, "grid_shape": [40, 40, 40],
                            "complex_storage": True, "susceptibilities": 2},
                           bracketed=True)
    assert shape == dp.ShapeClass(storage="complex", dimensions=3, susceptibilities=2,
                                  bracketed=True, cells=64000)
    # an absent susceptibility count is the zero the run-shape reader omits
    plain = dp.shape_class({"dimensions": 1, "grid_shape": [1, 1, 640]}, bracketed=False)
    assert plain.storage == "real" and plain.susceptibilities == 0


# ---------------------------------------------------------------------------
# Realistic: the records shipped beside the kernel tables
# ---------------------------------------------------------------------------

DRIVE_CEILING = 110592          # the largest DRIVE witness, 48 cubed
SWEEP_TOP = 5400000

SHIPPED = {
    "triton": (os.path.join(HERE, "triton_kernels", "timing.json"),
               "fused pair D", "fused pair B"),
    "cuda": (os.path.join(HERE, "cuda_kernels", "timing.json"),
             "cuda:fused electric pair", "cuda:fused magnetic pair"),
}


def _shipped(table):
    # A TRACKED FILE, so its absence is a failure and never a skip: the record ships
    # beside the kernel table it describes.
    path = SHIPPED[table][0]
    assert os.path.exists(path), f"{path} is not in the tree"
    return dp.load_record(path, expect_table=table)


def _ask_entry(record, table, entry, cells):
    return dp.consult(
        record, table=table, candidate=entry["product"], displaces=entry["displaces"],
        run_shape={"dimensions": entry["dimensions"], "grid_shape": [cells, 1, 1],
                   "complex_storage": entry["storage"] == "complex",
                   "susceptibilities": entry["susceptibilities"]},
        bracketed=entry["bracketed"], mode="measured",
        repair_route=record["route"], subject_sha256=dp.compared_emitters(record))


@pytest.mark.parametrize("table", sorted(SHIPPED))
def test_every_two_pair_bracketed_plan_at_drive_sizes_is_slower_than_its_singles(table):
    record = _shipped(table)
    seen = measured = 0
    for entry in record["keys"].values():
        for row in entry.get("measurements", []):
            if not (entry["bracketed"] and row["co_fused"]
                    and row["cells"] <= DRIVE_CEILING):
                continue
            seen += 1
            assert row["ratio"] < 1.0, (entry["product"], row["case"], row["ratio"])
            if entry["status"] == "measured":
                measured += 1
                verdict = _ask_entry(record, table, entry, row["cells"])
                assert verdict.vetoed, (entry["product"], row["case"],
                                        verdict.as_dict())
    assert seen >= 4 and measured >= 2, (seen, measured)


#: Clean-seam rows at DRIVE sizes in each shipped record. The CUDA record has NONE,
#: and that is a fact about the corpus rather than about the products: both of its
#: clean-seam rows timed plans whose slots the CUDA and the Triton tables BOTH served,
#: and a mixed-table plan is not a measurement of either table's plan (the cutter's
#: admission rule). A count, not a floor, so a re-cut that loses them says so.
CLEAN_DRIVE_ROWS = {"triton": 5, "cuda": 0}


@pytest.mark.parametrize("table", sorted(SHIPPED))
def test_the_clean_seam_one_pair_rows_are_not_vetoed(table):
    record = _shipped(table)
    seen = 0
    for entry in record["keys"].values():
        for row in entry.get("measurements", []):
            if entry["bracketed"] or row["co_fused"] or row["cells"] > DRIVE_CEILING:
                continue
            seen += 1
            verdict = _ask_entry(record, table, entry, row["cells"])
            assert not verdict.vetoed, (entry["product"], row["case"],
                                        verdict.as_dict())
    assert seen == CLEAN_DRIVE_ROWS[table], seen


@pytest.mark.parametrize("table", sorted(SHIPPED))
def test_at_the_top_of_the_sweep_the_two_pair_plans_are_faster(table):
    record = _shipped(table)
    singles = ("cuda:PML", "cuda:ordinary") if table == "cuda" else ("PML", "ordinary")

    def ask(product, grid_shape):
        return dp.consult(
            record, table=table, candidate=product, displaces=singles,
            run_shape={"dimensions": 2, "grid_shape": grid_shape,
                       "complex_storage": False, "susceptibilities": 0},
            bracketed=True, mode="measured", repair_route=record["route"],
            subject_sha256=dp.compared_emitters(record))

    for product in SHIPPED[table][1:]:
        top = ask(product, [3000, 1800, 1])
        assert top.outcome == "not_vetoed", top.as_dict()
        assert top.evidence["cells"] == SWEEP_TOP
        at_the_top = [row for row in record["keys"][top.key]["measurements"]
                      if row["cells"] == SWEEP_TOP]
        assert len(at_the_top) >= 2
        assert all(row["ratio"] > 1.0 and row["co_fused"] for row in at_the_top), top.key
        # and the same product at the smallest sweep size is vetoed
        small = ask(product, [200, 120, 1])
        assert small.vetoed, small.as_dict()
        # between the last size that loses and the first that wins there is no row,
        # and the record says so instead of guessing which side a run falls on
        lost = record["keys"][small.key]["band"]["hi_cells"]
        gap = ask(product, [lost + 1, 1, 1])
        assert gap.outcome == "no_record" and gap.reason == "key_unmeasured", gap.as_dict()
        # EITHER no row lies between the two verdicts, OR the sizes that do were each
        # measured once and so set no band edge. Both are stated; neither is guessed.
        because = gap.evidence["unmeasured_because"]
        assert "verdict changes" in because or "set a band edge" in because, because


@pytest.mark.parametrize("table", sorted(SHIPPED))
def test_a_shipped_record_answers_no_record_for_a_tree_on_another_route(table):
    record = _shipped(table)
    live = dict(record["route"], deposit_repair_sha256=dp.repair_route_sha256())
    entry = next(e for e in record["keys"].values() if e["status"] == "measured")
    verdict = dp.consult(
        record, table=table, candidate=entry["product"], displaces=entry["displaces"],
        run_shape={"dimensions": entry["dimensions"],
                   "grid_shape": [entry["band"]["lo_cells"], 1, 1],
                   "complex_storage": entry["storage"] == "complex",
                   "susceptibilities": entry["susceptibilities"]},
        bracketed=entry["bracketed"], mode="measured", repair_route=live,
        subject_sha256=dp.compared_emitters(record))
    if live["deposit_repair_sha256"] == record["route"]["deposit_repair_sha256"]:
        assert verdict.outcome in ("vetoed", "not_vetoed")
    else:
        assert verdict.reason == "record_is_for_another_repair_route"


@pytest.mark.parametrize("table", sorted(SHIPPED))
def test_span_mode_never_vetoes_anything_a_shipped_record_would(table):
    record = _shipped(table)
    for entry in record["keys"].values():
        for cells in (entry["band"]["lo_cells"], entry["band"]["hi_cells"]):
            verdict = dp.consult(
                record, table=table, candidate=entry["product"],
                displaces=entry["displaces"],
                run_shape={"dimensions": entry["dimensions"], "grid_shape": [cells, 1, 1],
                           "complex_storage": entry["storage"] == "complex",
                           "susceptibilities": entry["susceptibilities"]},
                bracketed=entry["bracketed"], mode="span",
                repair_route=record["route"],
                subject_sha256=dp.compared_emitters(record))
            assert not verdict.vetoed


# ---------------------------------------------------------------------------
# Structure: unwired, and inside the environment-variable contract
# ---------------------------------------------------------------------------

def _tree():
    with open(os.path.join(HERE, "dispatch_preference.py"), encoding="utf-8") as handle:
        return ast.parse(handle.read())


def test_the_consult_reads_no_environment_variable():
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Attribute):
            assert node.attr not in ("environ", "getenv", "putenv"), node.lineno
        if isinstance(node, ast.Name):
            assert node.id != "getenv", node.lineno


def test_the_consult_imports_nothing_from_this_package():
    for node in ast.walk(_tree()):
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"relative import at line {node.lineno}"
            assert not (node.module or "").startswith("meep_gpu"), node.lineno
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("meep_gpu"), node.lineno


def test_no_dispatch_module_imports_the_consult_yet():
    """The wiring is its own re-certification round; until then nothing reaches it."""
    importers = []
    for folder, _dirs, files in os.walk(HERE):
        for name in files:
            if not name.endswith(".py") or name.startswith("test_"):
                continue
            if name == "dispatch_preference.py":
                continue
            with open(os.path.join(folder, name), encoding="utf-8") as handle:
                text = handle.read()
            if "dispatch_preference" in text:
                importers.append(os.path.relpath(os.path.join(folder, name), HERE))
    assert importers == [], importers
