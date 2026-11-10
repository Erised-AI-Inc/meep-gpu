"""Host tests for the dispatch-preference consult and its timing-record format.

Written before ``dispatch_preference.py`` (the repository's test-first rule for a new
decision engine). Five groups, in the order that rule names them:

* KNOWN VALUE -- a hand-built record vetoes exactly the candidate it measured slower
  beyond its spreads, and nothing else;
* DEGENERATE -- no record, an empty one, a missing key, a key marked unmeasured, a
  ratio inside the spread, a record for another table, a record cut on another compute
  capability, a device whose capability the caller cannot read, a record cut on another
  repair route (the WHOLE route identity, not its digest alone), a record cut on another
  fused-pair emitter, a live route or emitter the caller cannot state, a baseline that
  is not what the candidate would displace, and every malformed record refusing BY NAME;
* SCALING -- the band lookup at, below and above each edge, and monotone in cells;
* SHAPE / SERIALIZATION -- round trip, deterministic bytes, a verdict that serializes;
* REALISTIC -- the shipped records assert three facts: two-pair bracketed plans lose to
  their singles at DRIVE sizes, clean-seam one-pair rows are not vetoed, two-pair plans win at the top;
  every shipped record is at a name the reader opens (its architecture's representative,
  or its own card's name beside it), agrees with that name and holds on this tree, and
  its architecture is one the nearest rule can rank; every table reads one vendor's
  prior, from the nearest timed architecture, and a release ships a timed NVIDIA record
  and a timed Apple record.
* NAMES AND NEAREST -- a card's slug and its record's name, a name that disagrees with
  its content refused, and the nearest timed architecture of each vendor (compute
  capability, chip generation, ties, a spelling the rule cannot read).

The consult is WIRED at the composer rung (``fastpath._apply_preference``, built
in 1.0.1): the last group of structural tests pins that only ``fastpath`` imports
it and that it reads no environment variable, which is this package's environment-variable
contract (only ``fastpath.py`` and ``subnormal_policy.py`` read one). The wiring is tested in
``test_dispatch_preference_wiring.py``.
"""

from __future__ import annotations

import ast
import copy
import io
import json
import os
import pathlib
import tokenize

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

#: One record prices one architecture, so every record below stamps the capability its
#: rows ran on and every consult below states the capability it is asking about. The
#: hand-built records stamp the architecture the shipped ones were cut on, which is the
#: one :data:`SHIPPED` names its files for; ``OTHER_CAPABILITY`` is a card no record
#: here was cut on.
CAPABILITY = "8.6"
OTHER_CAPABILITY = "9.0"

#: The one host row a hand-built record carries. ``compute_capability`` is what the ROWS
#: stamped from the device they ran on, and the only field the consult reads off it.
HOST = {"device": "EXAMPLE GPU", "compute_capability": CAPABILITY}


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


def _record(entries, table="triton", route=ROUTE, barrier=None, host=None,
            **route_facts):
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
        "host": [dict(HOST)] if host is None else [dict(row) for row in host],
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
         mode="measured", route=ROUTE, live_route=AS_TIMED, emitters=AS_TIMED,
         capability=CAPABILITY, **shape):
    if live_route is AS_TIMED:
        live_route = dict(LIVE_ROUTE, deposit_repair_sha256=route)
    return dp.consult(record, table=table, capability=capability, candidate=candidate,
                      displaces=displaces,
                      run_shape=_shape(**shape), bracketed=bracketed, mode=mode,
                      repair_route=live_route,
                      subject_sha256=dict(LIVE_EMITTERS) if emitters is AS_TIMED
                      else emitters)


def _capability(record):
    """The ONE capability a record's rows stamp, read off the record itself.

    Asked of every shipped record below rather than typed beside it, so a re-cut on
    another card is a consult about that card and never a comparison against a
    capability the test spelled for it. :func:`dp.load_record` has already refused a
    record that names more or fewer than one.
    """
    capability, = dp.record_capabilities(record)
    return capability


def _tmp_record(tmp_path, table="triton", capability=CAPABILITY):
    """A writable record path under ``tmp_path``, named the ONE name it may have.

    The name is :func:`dp.record_path`'s even here, so a test cannot read a file under
    a name nothing in the package looks up.
    """
    path = pathlib.Path(dp.record_path(table, capability, root=str(tmp_path)))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


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


def test_a_record_cut_on_another_architecture_is_never_applied():
    """One record prices one architecture. The ratio a record carries was measured on
    one card's instruction set and occupancy; applying it to another card is a false
    veto, so the capability is a barrier exactly like the repair route and the
    emitters."""
    record = _record([_entry(**SLOW_PAIR)])
    assert _ask(record, "fused pair D", bracketed=True).vetoed
    verdict = _ask(record, "fused pair D", bracketed=True,
                   capability=OTHER_CAPABILITY)
    assert verdict.outcome == "no_record" and not verdict.vetoed
    assert verdict.reason == "record_is_for_another_capability"
    assert verdict.evidence["record_capability"] == [CAPABILITY]
    assert verdict.evidence["live_capability"] == OTHER_CAPABILITY
    assert verdict.evidence["record_sha256"] == record["record_sha256"]


def test_a_device_whose_capability_cannot_be_read_is_no_record():
    """A run that cannot say which architecture it is on may not be priced from a card
    it may not be: ``None`` is the mismatch it might be, never today's record."""
    record = _record([_entry(**SLOW_PAIR)])
    verdict = _ask(record, "fused pair D", bracketed=True, capability=None)
    assert verdict.outcome == "no_record" and not verdict.vetoed
    assert verdict.reason == "capability_unread"
    assert verdict.evidence["record_capability"] == [CAPABILITY]
    assert "live_capability" not in verdict.evidence


def test_a_record_naming_two_architectures_or_none_refuses_by_name():
    """The refusal is :func:`dp.validate`'s, so a mixed record cannot even be
    consulted: one number pooled over two architectures is the number a veto is read
    off, and an unstamped row counts as a value of its own so that it cannot hide
    inside a record that otherwise names one card."""
    for host in ([dict(HOST), dict(HOST, compute_capability=OTHER_CAPABILITY)],
                 [{"device": "EXAMPLE GPU"}],
                 [dict(HOST, compute_capability="")],
                 [dict(HOST), {"device": "EXAMPLE GPU"}]):
        record = _record([_entry(**SLOW_PAIR)], host=host)
        with pytest.raises(dp.RecordRefused) as refusal:
            dp.validate(record)
        assert refusal.value.code == "host_names_no_single_capability", host
    # an unstamped row is KEPT as a distinct value rather than dropped, which is what
    # makes the count wrong and the refusal fire
    unstamped = _record([_entry(**SLOW_PAIR)], host=[dict(HOST),
                                                     {"device": "EXAMPLE GPU"}])
    assert dp.record_capabilities(unstamped) == (
        CAPABILITY, "<no compute capability stamped>")
    assert dp.record_capabilities(_record([_entry(**SLOW_PAIR)])) == (CAPABILITY,)


def test_the_record_for_one_table_and_one_capability_has_one_spelling(tmp_path):
    """:func:`dp.record_path` is where a record lives, for the cutter and the consult
    alike: the capability is in the file name, so a directory listing says which
    architectures have been timed."""
    assert dp.record_path("triton", CAPABILITY) == os.path.join(
        HERE, "triton_kernels", "timing_cc86.json")
    assert dp.record_path("cuda", "9.0", root=str(tmp_path)) == os.path.join(
        str(tmp_path), "cuda_kernels", "timing_cc90.json")
    with pytest.raises(dp.ConsultRefused) as refusal:
        dp.record_path("opencl", CAPABILITY)
    assert refusal.value.code == "unknown_table"
    # the capability has ONE spelling; the others CuPy hands out are refused by name
    # rather than normalised by a second reader that could disagree with fastpath's
    for spelling in ("86", "sm_86", "8", "8.6.0", "", "eight.six"):
        with pytest.raises(dp.ConsultRefused) as refusal:
            dp.record_path("triton", spelling)
        assert refusal.value.code == "unreadable_capability", spelling


def test_the_metal_record_is_named_for_its_apple_gpu_family(tmp_path):
    """The Metal table's architecture is the Apple GPU family its rows stamp in the
    capability field (``applegpu_g13s``). Its record takes that name, is read, and is
    applied to that family alone; the other spelling is refused on either side."""
    assert dp.record_path("metal", "applegpu_g13s", root=str(tmp_path)) == os.path.join(
        str(tmp_path), "metal_kernels", "timing_applegpu_g13s.json")
    for table, spelling in (("metal", CAPABILITY), ("triton", "applegpu_g13s"),
                            ("metal", "applegpu-g13s"), ("metal", "")):
        with pytest.raises(dp.ConsultRefused) as refusal:
            dp.record_path(table, spelling)
        assert refusal.value.code == "unreadable_capability", (table, spelling)
    apple = dict(HOST, device="Apple M1 Max", compute_capability="applegpu_g13s")
    record = _record([_entry(**SLOW_PAIR)], table="metal", host=[apple])
    dp.validate(record, expect_table="metal")
    verdict = _ask(record, "fused pair D", bracketed=True, table="metal",
                   capability="applegpu_g13s")
    assert verdict.outcome == "vetoed", verdict.as_dict()
    assert _ask(record, "fused pair D", bracketed=True, table="metal",
                capability="applegpu_g15s").outcome == "no_record"
    # a Metal record stamped with a compute capability, or an NVIDIA one stamped with
    # an Apple family, names no architecture its table can have
    for table, host in (("metal", dict(HOST)), ("triton", apple)):
        with pytest.raises(dp.RecordRefused) as refusal:
            dp.validate(_record([_entry(**SLOW_PAIR)], table=table, host=[host]))
        assert refusal.value.code == "host_names_no_single_capability", table
    from meep_gpu import metal_runs  # noqa: PLC0415 - one spelling, held equal
    assert dp.APPLE_FAMILY.pattern == metal_runs.ARCHITECTURE_KEY.pattern


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
    # The route is an identity, not a digest: every fact of it is validated, and its
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
    missing = dp.record_path("triton", CAPABILITY, root=str(tmp_path / "absent"))
    garbage = _tmp_record(tmp_path)
    garbage.write_text("{ not json")
    for path in (missing, str(garbage)):
        verdict = dp.consult_path(path, table="triton", capability=CAPABILITY,
                                  candidate="fused pair D",
                                  displaces=("PML", "ordinary"), run_shape=_shape(),
                                  bracketed=True, mode="span",
                                  repair_route=LIVE_ROUTE,
                                  subject_sha256=LIVE_EMITTERS)
        assert verdict.outcome == "no_record" and not verdict.vetoed
    # span mode does not even validate a record handed to it directly
    assert not _ask({"not": "a record"}, "fused pair D", bracketed=True,
                    mode="span").vetoed


def test_measured_mode_reads_the_path_it_is_given(tmp_path):
    path = _tmp_record(tmp_path)
    path.write_text(dp.dumps(_record([_entry(**SLOW_PAIR)])))
    asked = dict(table="triton", capability=CAPABILITY, candidate="fused pair D",
                 displaces=("PML", "ordinary"),
                 run_shape=_shape(), bracketed=True, mode="measured",
                 repair_route=LIVE_ROUTE, subject_sha256=LIVE_EMITTERS)
    assert dp.consult_path(str(path), **asked).vetoed
    # a table whose architecture has never been timed reads no record at all, which is
    # today's behaviour rather than another architecture's number
    absent = dp.consult_path(
        str(_tmp_record(tmp_path, capability=OTHER_CAPABILITY)), **asked)
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
    verdict = dp.consult(record, table="triton", capability=CAPABILITY,
                         candidate="fused pair D",
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
            dp.consult(record, table="triton", capability=CAPABILITY,
                       candidate="fused pair D",
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
    path = _tmp_record(tmp_path)
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

#: The compute capability the shipped records were cut on, and the architecture whose
#: name their files carry: each is cc 8.6's representative, the record every cc 8.6 card
#: without a record of its own reads, cut on one card that the record itself names
#: (``host[].device``; read it off the record, never off this comment).
#: :func:`dp.record_path` spells the path, so these tests cannot read a record under a
#: name the package does not look up.
SHIPPED_CAPABILITY = "8.6"

SHIPPED = {
    "triton": (dp.record_path("triton", SHIPPED_CAPABILITY),
               "fused pair D", "fused pair B"),
    "cuda": (dp.record_path("cuda", SHIPPED_CAPABILITY),
             "cuda:fused electric pair", "cuda:fused magnetic pair"),
}


def _shipped(table):
    # A TRACKED FILE, so its absence is a failure and never a skip: the record ships
    # beside the kernel table it describes, under the name of the architecture it prices.
    path = SHIPPED[table][0]
    assert os.path.exists(path), f"{path} is not in the tree"
    record = dp.load_record(path, expect_table=table)
    assert _capability(record) == SHIPPED_CAPABILITY, (
        f"{path} is named for compute capability {SHIPPED_CAPABILITY} and its rows "
        f"were timed on {_capability(record)}")
    return record


def _ask_entry(record, table, entry, cells):
    return dp.consult(
        record, table=table, capability=_capability(record),
        candidate=entry["product"], displaces=entry["displaces"],
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


#: Clean-seam rows at DRIVE sizes in each shipped record. A count, not a floor, so a
#: re-cut that loses them says so. The records are the cuts of one RTX 3090's DRIVE
#: and size-sweep rows under the between-pass gate. Triton 6:
#: ``fused pair B`` on offdiag_2d (the DRIVE unit, 24,000 cells) and pml_3d (110,592),
#: ``fused pair B (folded complex)`` on the two folded complex off-diagonal cases,
#: ``fused pair B (nonlinear)`` on nonlinear_1d and ``fused pair D (complex conductive
#: no-PML)`` on complex_no_pml_3d. CUDA 2: ``cuda:fused magnetic pair`` on offdiag_2d
#: and pml_3d. The cut from the 2026-10-27 rows counts 6, not 8: the size sweep's two
#: passes of offdiag_2d at 24,000 cells (resolution 20) read a fused between-pass
#: spread of 0.0587, above the cutter's 0.05 gate, so both are in the record's
#: excluded list; the cut before it admitted them. The 2026-09-21 cut counted 3 and 0: its
#: Triton offdiag_2d and folded_complex_offdiag_2d rows failed the window floor the
#: between-pass gate retired, and its CUDA clean rows timed plans both NVIDIA tables
#: served, which the cutter's same-table rule leaves out.
CLEAN_DRIVE_ROWS = {"triton": 6, "cuda": 2}


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
            record, table=table, capability=_capability(record), candidate=product,
            displaces=singles,
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
        record, table=table, capability=_capability(record),
        candidate=entry["product"], displaces=entry["displaces"],
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
                record, table=table, capability=_capability(record),
                candidate=entry["product"],
                displaces=entry["displaces"],
                run_shape={"dimensions": entry["dimensions"], "grid_shape": [cells, 1, 1],
                           "complex_storage": entry["storage"] == "complex",
                           "susceptibilities": entry["susceptibilities"]},
                bracketed=entry["bracketed"], mode="span",
                repair_route=record["route"],
                subject_sha256=dp.compared_emitters(record))
            assert not verdict.vetoed


def test_every_shipped_record_is_at_a_name_the_reader_opens_and_agrees_with_it():
    """The package's records as the dispatch path reads them.

    A card reads its own record, ``dp.record_path(table, architecture, card=<card>)``,
    when the package ships one, and every other card of the architecture reads the
    representative at ``dp.record_path(table, architecture)``. So every shipped file is
    at one of those two names for its own table, architecture and card, the reader
    accepts its name (a card's name agrees with the rows, ``dp.load_record``), it names
    one device, no card has two files per table and architecture, and the
    representatives of the NVIDIA tables of one architecture were cut on one card.
    """
    seen = {}
    representatives = {}
    for path in dp.record_files():
        record = dp.load_record(path)
        capability = _capability(record)
        devices = dp.record_devices(record)
        assert len(devices) == 1, (path, devices)
        card = devices[0]
        table = record["table"]
        bare = dp.record_path(table, capability)
        own = dp.record_path(table, capability, card=card)
        assert path in (bare, own), (
            f"{path} is neither the representative's name {os.path.basename(bare)} nor "
            f"its card's own name {os.path.basename(own)} for the {table} record of "
            f"{capability}")
        slug = (table, capability, dp.card_slug(card))
        assert slug not in seen, (path, seen.get(slug))
        seen[slug] = path
        if path == bare and table in ("triton", "cuda"):
            representatives.setdefault(capability, set()).add(card)
    for capability, cards in representatives.items():
        assert len(cards) == 1, (capability, sorted(cards))


def test_every_shipped_records_architecture_is_ranked_by_the_nearest_rule():
    """A device with no record of its own reads the prior of its vendor's nearest timed
    architecture, and a timed spelling the rule cannot read a distance from refuses the
    prior for every such device of that vendor. So every shipped record's architecture
    is one :func:`dp.architecture_number` reads."""
    unread = []
    for path in dp.record_files():
        record = dp.load_record(path)
        vendor = dp.VENDOR_BY_TABLE[record["table"]]
        if dp.architecture_number(vendor, _capability(record)) is None:
            unread.append(f"{os.path.relpath(path, HERE)}: {_capability(record)}")
    assert not unread, unread


def test_every_shipped_record_holds_on_this_tree():
    """A shipped record answers a consult only while the deposit-repair module and the
    fused-pair emitters it was cut on are the tree's (the consult's own barriers); one that
    does not hold prices nothing and only claims to. Each failing file is named with its
    recorded and live digests."""
    live_route = dp.repair_route_sha256()
    failing = []
    for path in dp.record_files():
        record = dp.load_record(path)
        live = dp.live_subject_sha256(record["table"])
        why = []
        recorded = record["route"]["deposit_repair_sha256"]
        if recorded != live_route:
            why.append(f"route {record['route']['id']} against the live deposit_repair.py "
                       f"{live_route[:12]}")
        why += [f"{source} {digest[:12]} against the live {str(live.get(source))[:12]}"
                for source, digest in sorted(dp.compared_emitters(record).items())
                if live.get(source) != digest]
        if why:
            failing.append(f"{os.path.relpath(path, HERE)}: " + "; ".join(why))
    assert not failing, "\n".join(failing)


def test_every_table_reads_the_prior_of_one_vendor():
    """The vendor follows the architecture's spelling: a compute capability is NVIDIA's,
    an Apple GPU family Apple's, and every table has exactly one."""
    assert set(dp.VENDOR_BY_TABLE) == set(dp.TABLES)
    assert dp.VENDORS == ("apple", "nvidia")
    for table, vendor in dp.VENDOR_BY_TABLE.items():
        spelling = dp.ARCHITECTURE_BY_TABLE[table]
        assert vendor == {dp.CAPABILITY: "nvidia", dp.APPLE_FAMILY: "apple"}[spelling], table


def test_the_package_ships_an_nvidia_and_an_apple_timing_record():
    """A release ships at least one timed NVIDIA record (Triton or hand-written CUDA
    table) and one timed Apple record (Metal table): each loads, holds on this tree and
    prices at least one key (:func:`dp.timed_vendors`, the rule's one home, which the
    public export asks of the exported tree too). No vendor's GPUs are left with neither a
    record of their own nor a prior of their vendor."""
    timed = dp.timed_vendors()
    assert set(timed) == set(dp.VENDORS) == {"apple", "nvidia"}
    missing = sorted(vendor for vendor, block in timed.items() if not block["records"])
    assert not missing, "; ".join(
        f"no timed {vendor} record ships ({timed[vendor]['why_none']})" for vendor in missing)


#: The barrier sources a staging package below writes, by table, with fixed texts.
_STAGED_SOURCES = {"deposit_repair.py": "REPAIR = 1\n",
                   "cuda_kernels/fused_pairs.py": "PAIRS = 1\n",
                   "cuda_kernels/fused_polarization_pair.py": "POLARIZATION = 1\n",
                   "metal_dispatch.py": "METAL = 1\n",
                   "metal_kernels/launch.py": "LAUNCH = 1\n",
                   "metal_kernels/device.py": "DEVICE = 1\n"}


def _staged_package(root):
    """A package directory under ``root`` holding the barrier sources alone."""
    for name, text in _STAGED_SOURCES.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def _staged_record(root, table, architecture, *, route=None, moved=(), measured=True):
    """A sealed record of ``table`` cut on the staged sources, placed where the reader
    looks; ``route`` another deposit-repair digest, ``moved`` emitters recorded at another
    digest, ``measured`` False writes its one key unmeasured."""
    live = dp.live_subject_sha256(table, str(root))
    digests = {source: [OTHER_PAIRS_SHA if source in moved else digest]
               for source, digest in live.items()}
    barrier = {"sources": sorted(live), "digests": digests,
               "rows_not_recording": {source: 0 for source in live},
               "allowed_unpinned": []}
    entry = _entry(**dict(SLOW_PAIR, status="measured" if measured else "unmeasured"))
    record = _record([entry], table=table,
                     route=route or dp.repair_route_sha256(str(root / "deposit_repair.py")),
                     barrier=barrier,
                     host=[{"device": "EXAMPLE GPU", "compute_capability": architecture}])
    path = pathlib.Path(dp.record_path(table, architecture, root=str(root)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dp.dumps(record), encoding="utf-8")
    return path


def test_a_timed_vendor_has_a_record_that_holds_and_measures_a_key(tmp_path):
    """:func:`dp.timed_vendors` on staging packages: a record cut on another
    deposit-repair route, one whose emitters moved since it was cut and one that measures
    no key each count for nothing, each named with why; a holding Metal record with a
    measured key names apple and a holding Triton one names nvidia."""
    cases = {"route": dict(route=OTHER_ROUTE), "emitters": dict(moved=("fused_pairs",)),
             "unmeasured": dict(measured=False)}
    for name, change in cases.items():
        root = _staged_package(tmp_path / name)
        _staged_record(root, "metal", "applegpu_g13s", **change)
        timed = dp.timed_vendors(str(root))
        assert timed["apple"]["records"] == [], (name, timed)
        assert [item["path"] for item in timed["apple"]["not_timed"]] == [
            "metal_kernels/timing_applegpu_g13s.json"], (name, timed)
        assert timed["apple"]["why_none"].startswith("no apple record is timed"), timed
    why = {name: dp.timed_vendors(str(tmp_path / name))["apple"]["not_timed"][0]["why"]
           for name in cases}
    assert why["route"].startswith("cut on deposit-repair route"), why
    assert why["emitters"] == "its emitters fused_pairs moved since it was cut", why
    assert why["unmeasured"] == "it measures no key", why
    root = _staged_package(tmp_path / "holds")
    timed = dp.timed_vendors(str(root))
    assert {vendor: block["why_none"] for vendor, block in timed.items()} == {
        "apple": "no apple record ships", "nvidia": "no nvidia record ships"}
    _staged_record(root, "metal", "applegpu_g13s")
    _staged_record(root, "triton", CAPABILITY)
    timed = dp.timed_vendors(str(root))
    assert timed["apple"] == {"records": ["metal_kernels/timing_applegpu_g13s.json"],
                              "not_timed": [], "why_none": None}, timed
    assert timed["nvidia"]["records"] == ["triton_kernels/timing_cc86.json"], timed
    # A record in another table's folder is not that table's record
    misplaced = _staged_package(tmp_path / "misplaced")
    path = _staged_record(misplaced, "metal", "applegpu_g13s")
    (misplaced / "triton_kernels").mkdir()
    path.rename(misplaced / "triton_kernels" / "timing_cc86.json")
    timed = dp.timed_vendors(str(misplaced))
    assert timed["nvidia"]["records"] == [] and timed["apple"]["records"] == [], timed
    assert timed["nvidia"]["not_timed"][0]["why"].startswith("refused"), timed


# ---------------------------------------------------------------------------
# Names: one record per card
# ---------------------------------------------------------------------------

#: The slugs of real cards, and two that must stay apart.
CARD_SLUGS = {
    "NVIDIA RTX A6000": "nvidia_rtx_a6000",
    "NVIDIA GeForce RTX 3090": "nvidia_geforce_rtx_3090",
    "NVIDIA GeForce RTX 4080": "nvidia_geforce_rtx_4080",
    "NVIDIA H200": "nvidia_h200",
    "Apple M1 Max": "apple_m1_max",
    "NVIDIA GeForce RTX 3090 Ti": "nvidia_geforce_rtx_3090_ti",
    "NVIDIA A100-SXM4-80GB": "nvidia_a100_sxm4_80gb",
}

#: (card, architecture, the card record's name beside the representative)
CARD_RECORDS = (
    ("NVIDIA RTX A6000", "8.6", "timing_cc86-nvidia_rtx_a6000.json"),
    ("NVIDIA GeForce RTX 3090", "8.6", "timing_cc86-nvidia_geforce_rtx_3090.json"),
    ("NVIDIA GeForce RTX 4080", "8.9", "timing_cc89-nvidia_geforce_rtx_4080.json"),
    ("NVIDIA H200", "9.0", "timing_cc90-nvidia_h200.json"),
    ("Apple M1 Max", "applegpu_g13s", "timing_applegpu_g13s-apple_m1_max.json"),
)


def test_a_card_is_named_by_one_deterministic_slug():
    for name, slug in CARD_SLUGS.items():
        assert dp.card_slug(name) == slug, name
        assert dp.card_slug(slug) == slug, "the rule is idempotent"
        assert dp.CARD_SLUG.match(slug), slug
    assert (dp.card_slug("NVIDIA GeForce RTX 3090 Ti")
            != dp.card_slug("NVIDIA GeForce RTX 3090")), "a 3090 Ti is not a 3090"
    # doubled spaces, dashes and padding collapse to the one spelling
    assert dp.card_slug("  NVIDIA--GeForce   RTX 3090 - ") == "nvidia_geforce_rtx_3090"
    for name in ("", "  ", "---", None):
        with pytest.raises(dp.RecordRefused) as refused:
            dp.card_slug(name)
        assert refused.value.code == "unnamed_card", name


def test_a_card_record_is_named_for_its_architecture_and_its_card(tmp_path):
    for card, architecture, name in CARD_RECORDS:
        tables = ("metal",) if architecture.startswith("applegpu_") else ("cuda", "triton")
        for table in tables:
            assert os.path.basename(dp.record_path(table, architecture, card=card)) == name
            assert dp.record_path(table, architecture, card=dp.card_slug(card)) == (
                dp.record_path(table, architecture, card=card)), "a slug names its card"
            # it reads back as one (architecture, card), and the bare name as the
            # architecture's representative
            assert dp.record_name_parts(name) == (architecture, dp.card_slug(card))
            bare = os.path.basename(dp.record_path(table, architecture))
            assert dp.record_name_parts(bare) == (architecture, None)
            assert name.startswith(bare[:-len(".json")] + dp.CARD_SEPARATOR)
            # the reader's glob lists it beside the others
            path = tmp_path / f"{table}_kernels" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
            assert str(path) in dp.record_files(str(tmp_path))
    with pytest.raises(dp.RecordRefused) as refused:
        dp.record_path("triton", CAPABILITY, card=" - ")
    assert refused.value.code == "unnamed_card"
    # a name of no record reads back as nothing
    for name in ("timing.json", "timing_cc9.json", "timing_cc086.json", "timing_x.json",
                 "timing_cc86.report.txt", "timing_cc86-.json", "timing_cc86-Example.json",
                 "timing_cc86-_x.json", "timing_cc86-a-b.json"):
        assert dp.record_name_parts(name) is None, name


def test_a_record_whose_name_names_another_card_is_refused(tmp_path):
    """A card's own record is read by that card alone, so a file at a card's name whose
    rows ran on another card -- or on two -- is refused by name. The representative's
    bare name, and a name of no record, hold any one card's record."""
    record = _record([_entry(**SLOW_PAIR)])               # cut on EXAMPLE GPU
    folder = tmp_path / "triton_kernels"
    folder.mkdir()
    for name in ("timing_cc86.json", "timing_cc86-example_gpu.json", "candidate.json"):
        (folder / name).write_text(dp.dumps(record))
        assert dp.load_record(str(folder / name))[dp.DIGEST_FIELD] == record[dp.DIGEST_FIELD]
    two = _record([_entry(**SLOW_PAIR)], host=[HOST, dict(HOST, device="OTHER GPU")])
    for name, document in (("timing_cc86-another_gpu.json", record),
                           ("timing_cc86-example_gpu.json", two)):
        (folder / name).write_text(dp.dumps(document))
        with pytest.raises(dp.RecordRefused) as refused:
            dp.load_record(str(folder / name))
        assert refused.value.code == "record_named_for_another_card", name
    assert dp.record_devices(two) == ("EXAMPLE GPU", "OTHER GPU")


# ---------------------------------------------------------------------------
# The nearest timed architecture
# ---------------------------------------------------------------------------

def test_the_distance_is_the_compute_capability_or_the_chip_generation():
    assert dp.architecture_number("nvidia", "8.6") == 86
    assert dp.architecture_number("nvidia", "9.0") == 90
    assert dp.architecture_number("nvidia", "12.1") == 121
    assert dp.architecture_number("apple", "applegpu_g13s") == 13
    assert dp.architecture_number("apple", "applegpu_g15p") == 15
    for vendor, spelling in (("nvidia", "86"), ("nvidia", "8.10"), ("nvidia", None),
                             ("nvidia", "applegpu_g13s"), ("apple", "8.6"),
                             ("apple", "applegpu_test"), ("apple", "applegpu_g13s_m2"),
                             ("apple", "applegpu_g13"), ("apple", "applegpu_G13S"),
                             ("apple", None), ("amd", "8.6")):
        assert dp.architecture_number(vendor, spelling) is None, (vendor, spelling)


#: The basis a verdict names when the prior came from the live architecture itself.
OWN = "own architecture"


@pytest.mark.parametrize("live, timed, nearest", [
    ("8.6", ("8.6", "8.9"), ("8.6", OWN, 0)),
    ("8.9", ("8.6", "8.9"), ("8.9", OWN, 0)),
    ("8.7", ("8.6", "8.9"), ("8.6", "nearest compute capability", 1)),
    ("8.8", ("8.6", "8.9"), ("8.9", "nearest compute capability", 1)),
    ("9.0", ("8.6", "8.9"), ("8.9", "nearest compute capability", 1)),
    ("7.5", ("8.6", "8.9"), ("8.6", "nearest compute capability", 11)),
    ("12.0", ("8.6", "8.9"), ("8.9", "nearest compute capability", 31)),
    # a tie: the lower compute capability wins
    ("8.7", ("8.6", "8.8"), ("8.6", "nearest compute capability", 1)),
    ("8.0", ("7.5", "8.5"), ("7.5", "nearest compute capability", 5)),
])
def test_an_nvidia_device_reads_the_nearest_compute_capability(live, timed, nearest):
    assert dp.OWN_ARCHITECTURE == OWN
    assert dp.NEAREST_BASIS == {"nvidia": "nearest compute capability",
                                "apple": "closest chip generation"}
    assert dp.nearest_architecture("nvidia", live, list(timed)) == nearest
    assert dp.nearest_architecture("nvidia", live, list(reversed(timed))) == nearest


@pytest.mark.parametrize("live, timed, nearest", [
    ("applegpu_g13s", ("applegpu_g13s", "applegpu_g15p"),
     ("applegpu_g13s", OWN, 0)),
    ("applegpu_g16x", ("applegpu_g13s", "applegpu_g15p"),
     ("applegpu_g15p", "closest chip generation", 1)),
    # one generation from each: the lower generation wins
    ("applegpu_g14x", ("applegpu_g13s", "applegpu_g15p"),
     ("applegpu_g13s", "closest chip generation", 1)),
    # the architecture itself before another family of its generation
    ("applegpu_g13s", ("applegpu_g13g", "applegpu_g13s"),
     ("applegpu_g13s", OWN, 0)),
    # one generation, neither its own: the first spelling in order
    ("applegpu_g13x", ("applegpu_g13s", "applegpu_g13g"),
     ("applegpu_g13g", "closest chip generation", 0)),
])
def test_an_apple_gpu_reads_the_closest_chip_generation(live, timed, nearest):
    assert dp.nearest_architecture("apple", live, list(timed)) == nearest
    assert dp.nearest_architecture("apple", live, list(reversed(timed))) == nearest


def test_a_spelling_the_nearest_rule_cannot_read_is_refused_by_name():
    assert dp.nearest_architecture("nvidia", "8.6", []) is None
    for vendor, live, timed, code in (
            ("nvidia", None, ["8.6"], "live_architecture_unparsed"),
            ("nvidia", "86", ["8.6"], "live_architecture_unparsed"),
            ("apple", "applegpu_test", ["applegpu_g13s"], "live_architecture_unparsed"),
            ("apple", None, ["applegpu_g13s"], "live_architecture_unparsed"),
            ("apple", "applegpu_g14x", ["applegpu_g13s", "applegpu_other"],
             "prior_architecture_unparsed"),
            ("apple", "applegpu_g13s", ["applegpu_g13s", "applegpu_g13s_m2"],
             "prior_architecture_unparsed"),
            ("amd", "8.6", ["8.6"], "unknown_vendor")):
        with pytest.raises(dp.ConsultRefused) as refused:
            dp.nearest_architecture(vendor, live, timed)
        assert refused.value.code == code, (vendor, live, timed)


def _architecture_record(table, architecture, entries, device):
    return _record(entries, table=table, route=ROUTE,
                   barrier={"sources": ["fused_pairs"], "digests": {"fused_pairs": [PAIRS_SHA]},
                            "rows_not_recording": {"fused_pairs": 0},
                            "allowed_unpinned": []},
                   host=[{"device": device, "compute_capability": architecture}])


def _bracketed(lo, hi, fused, baseline):
    return _entry(**dict(SLOW_PAIR, lo=lo, hi=hi, fused=fused, baseline=baseline))


def test_the_prior_is_derived_per_architecture_and_read_from_the_nearest():
    """Two NVIDIA architectures, each with its own crossover: a device reads the nearest
    one's, never the two pooled, and the evidence names the live architecture, the one
    the prior came from, the basis and the records of that architecture alone."""
    slow, keep = (1.40, 0.50), (0.48, 0.50)
    cc86 = _architecture_record("triton", "8.6", [_bracketed(24000, 2400000, *slow),
                                                 _bracketed(5400000, 5400000, *keep)],
                                "CARD 86")
    cc90 = _architecture_record("cuda", "9.0", [_bracketed(24000, 1014000, *slow),
                                               _bracketed(2400000, 2400000, *keep)],
                                "CARD 90")
    held = _architecture_record("triton", "9.0", [_entry(**dict(
        SLOW_PAIR, status="unmeasured", reason="1 row, minimum 2"))], "CARD 90")
    prior = dp.portable_prior([("a.json", cc86), ("b.json", cc90), ("c.json", held)],
                              repair_route_sha256=ROUTE,
                              subject_sha256_by_table={"triton": LIVE_EMITTERS,
                                                       "cuda": {**LIVE_EMITTERS,
                                                                "fused_polarization_pair":
                                                                PAIRS_SHA}})
    nvidia = prior["by_vendor"]["nvidia"]
    assert {arch: block["bracketed_veto_up_to_cells"]
            for arch, block in nvidia["by_architecture"].items()} == {"8.6": 2400000,
                                                                       "9.0": 1014000}
    assert [item["path"] for item in nvidia["by_architecture"]["9.0"]["derived_from"]] == [
        "b.json"]
    assert prior["not_used"] == [{"path": "c.json", "why": "it measures no key"}]
    for live, outcome, source, basis in (("8.6", "vetoed", "8.6", OWN),
                                         ("8.7", "vetoed", "8.6",
                                          "nearest compute capability"),
                                         ("8.9", "no_record", "9.0",
                                          "nearest compute capability")):
        for table in ("triton", "cuda"):
            verdict = dp.consult_prior(prior, table=table, architecture=live,
                                       candidate="a fused pair", bracketed=True,
                                       cells=1200000, mode="measured")
            assert verdict.outcome == outcome, (live, table, verdict.as_dict())
            evidence = verdict.evidence
            assert (evidence["live_architecture"], evidence["prior_architecture"],
                    evidence["basis"]) == (live, source, basis)
            assert {item["architecture"][0] for item in evidence["derived_from"]} == {source}
    unread = dp.consult_prior(prior, table="triton", architecture="sm_89",
                              candidate="a fused pair", bracketed=True, cells=1200000,
                              mode="measured")
    assert (unread.outcome, unread.reason) == ("no_record", "live_architecture_unparsed")
    assert unread.evidence["prior_architecture"] is None


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


def _imports_the_consult(text: str) -> bool:
    """Any import of the module in a spelling the parser resolves: ``import``, ``from``,
    or ``importlib.import_module`` / ``__import__`` with the name as a string constant."""
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Import) and any(
                "dispatch_preference" in alias.name for alias in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and (
                "dispatch_preference" in (node.module or "")
                or any(alias.name == "dispatch_preference" for alias in node.names)):
            return True
        if isinstance(node, ast.Call) and getattr(
                node.func, "attr", getattr(node.func, "id", None)) in (
                "import_module", "__import__") and any(
                isinstance(arg, ast.Constant) and "dispatch_preference" in str(arg.value)
                for arg in node.args):
            return True
    return False


def _names_the_consult(relative: str, text: str) -> list:
    """The lines where a package module's text names the module, comments included,
    outside the one exempt mention. This scan finds what no import parser resolves (a
    computed name, an f-string, a ``sys.modules`` lookup) and a run-time read of a route
    run's statement of the default, the record field that shares the name. The exemption
    is one position in one file: in ``metal_runs.py``, the string element of the
    module-level ``DISPATCH_RUN_FIELDS`` tuple that names that statement as a route-run
    field, which is how the record's shape check refuses it beside the digests."""
    exempt = set()
    if relative == "metal_runs.py":
        for node in ast.parse(text).body:
            targets = (node.targets if isinstance(node, ast.Assign)
                       else [node.target] if isinstance(node, ast.AnnAssign) else [])
            if targets and isinstance(node.value, ast.Tuple) and any(
                    isinstance(target, ast.Name) and target.id == "DISPATCH_RUN_FIELDS"
                    for target in targets):
                exempt.update((element.lineno, element.col_offset)
                              for element in node.value.elts
                              if isinstance(element, ast.Constant)
                              and element.value == "dispatch_preference")
    tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    seen = sum(token.string.count("dispatch_preference") for token in tokens)
    if seen != text.count("dispatch_preference"):
        # Never fewer mentions than the text holds: a token form this scan does not
        # read reports the module rather than hiding a mention.
        return [0]
    return [token.start[0] for token in tokens
            if "dispatch_preference" in token.string
            and not (token.type == tokenize.STRING and token.start in exempt
                     and ast.literal_eval(token.string) == "dispatch_preference")]


def test_only_the_composer_rung_imports_the_consult():
    """Wired in 1.0.1 through ``fastpath._apply_preference`` alone.

    ``metal_dispatch`` reaches it through that same function rather than importing the
    consult, so one place reads the records and one place decides; a second importer
    would be a second reading of the same evidence. Two scans of every non-test module,
    either of which reports it: :func:`_imports_the_consult` and the text scan
    :func:`_names_the_consult`, whose one exemption is the route-run field list.
    """
    # Armed: each import spelling, and the four spellings no parser resolves, are
    # reported in a module that is not the exemption's.
    imports = ("from . import dispatch_preference as dp",
               "from meep_gpu import dispatch_preference",
               "from meep_gpu.dispatch_preference import consult",
               "import meep_gpu.dispatch_preference",
               "importlib.import_module('meep_gpu.dispatch_preference')")
    unresolved = (
        'importlib.import_module(f"{__package__}.dispatch_preference")',
        'getattr(importlib, "import_module")("meep_gpu.dispatch_preference")',
        'sys.modules.get("meep_gpu." + "dispatch_preference")',
        'ledger["driver_dispatch"]["runs"][key]["dispatch_preference"]["vetoes_applied"]')
    for planted in imports:
        assert _imports_the_consult(planted), planted
    for planted in imports + unresolved:
        assert _names_the_consult("route_stamps.py", planted), planted
    # The exemption is that one element: anything else in metal_runs.py that names the
    # module is reported, and the same list spelled in another module is not exempt.
    with open(os.path.join(HERE, "metal_runs.py"), encoding="utf-8") as handle:
        field_list = handle.read()
    assert field_list.count("dispatch_preference") == 1
    assert _names_the_consult("metal_runs.py", field_list) == []
    for planted in imports + unresolved + ('FIELDS = ("lift", "dispatch_preference")',
                                            "# reads the dispatch_preference statement"):
        assert _names_the_consult("metal_runs.py", f"{field_list}\n{planted}\n"), planted
    assert _names_the_consult(
        "route_stamps.py", 'DISPATCH_RUN_FIELDS = ("lift", "dispatch_preference")')

    importers = []
    for folder, _dirs, files in os.walk(HERE):
        for name in files:
            if not name.endswith(".py") or name.startswith("test_"):
                continue
            if name == "dispatch_preference.py":
                continue
            with open(os.path.join(folder, name), encoding="utf-8") as handle:
                text = handle.read()
            relative = os.path.relpath(os.path.join(folder, name), HERE)
            if _imports_the_consult(text) or _names_the_consult(relative, text):
                importers.append(relative)
    assert importers == ["fastpath.py"], importers
