"""Two architectures gated on one commit leave the same ledger in either binding order.

A round on a second architecture is gated on its own card, and the rebind that records
it runs later on a laptop. Whichever architecture is bound first, the other is bound
onto the same bytes afterwards, and the ledger that results must not depend on which
came first: a record that differs by binding order is a record whose content is an
accident of scheduling rather than a fact about a run.

The rebind's carry rule is what decides it. A rebind may carry a per-run fact the fresh
artifact does not state (a step budget, a curated narrative) from the same
architecture's previous record. It used to do so whenever the bytes had not moved IN
THAT PASS, so binding 9.0 first and 8.6 second carried the old 8.6 record's facts --
measured on other bytes -- into the new one. These tests drive the rebind's own
:func:`rebind_triton_welds.carried_run_facts` with the real
:func:`fastpath.bind_capability`, in the order the tool calls them.
"""

from __future__ import annotations

import copy
import json
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rebind_triton_welds as rebind  # noqa: E402

from meep_gpu import fastpath  # noqa: E402

OLD = "a" * 64          # the bytes the shipped 8.6 records were cut on
NEW = "b" * 64          # the bytes of the commit both architectures are gated on


def shipped_entry() -> dict:
    """An entry as the release ships it: one 8.6 record, live on the OLD bytes."""
    entry = {"source_sha256": {"kernels.py": OLD}, "status": "PASS"}
    fastpath.bind_capability(
        entry, bound_before=None, capability="8.6",
        run={"host": "the original A6000 run", "records": "old/",
             "step_budget": "measured on the OLD bytes"})
    assert fastpath.live_capabilities(entry) == ("8.6",)
    return entry


def bind(entry: dict, capability: str, host: str, supersede=()) -> tuple:
    """One rebind of one weld, in the order ``rebind_triton_welds`` performs it.

    The run states no step budget, so anything the record ends up with was CARRIED. A
    refusal puts the entry back as it was, as the tool does: the digests are refreshed
    before the bind, so leaving them would make a retry see bytes that never moved.
    """
    before = copy.deepcopy(entry)
    bound_before = fastpath.bound_digest(entry)
    entry["source_sha256"] = {"kernels.py": NEW}            # refreshed from the run
    run = {"host": host, "records": f"{capability}/"}
    run.update(rebind.carried_run_facts(entry, capability, rebind.CARRIED_WITHIN_CAPABILITY))
    try:
        return fastpath.bind_capability(entry, bound_before=bound_before,
                                        capability=capability, run=run, supersede=supersede)
    except fastpath.CapabilityRecordError:
        entry.clear()
        entry.update(before)
        raise


def test_the_architecture_already_recorded_binds_first_with_no_supersede():
    entry = shipped_entry()
    assert bind(entry, "8.6", "the A6000 re-run") == ()
    assert bind(entry, "9.0", "the H100 run") == ()
    assert fastpath.live_capabilities(entry) == ("8.6", "9.0")


def test_the_new_architecture_first_needs_the_old_one_named_and_ends_the_same():
    """9.0 first is a legitimate order, and it is the one a rented card makes likely."""
    first = shipped_entry()
    bind(first, "8.6", "the A6000 re-run")
    bind(first, "9.0", "the H100 run")

    second = shipped_entry()
    with pytest.raises(fastpath.CapabilityStale) as refused:
        bind(second, "9.0", "the H100 run")
    assert refused.value.names == ("8.6",)
    assert bind(second, "9.0", "the H100 run", supersede=("8.6",)) == ("8.6",)
    # Until 8.6 is re-bound, the WORKING ledger does not admit it: never commit here.
    assert fastpath.live_capabilities(second) == ("9.0",)
    assert bind(second, "8.6", "the A6000 re-run") == ()

    assert fastpath.live_capabilities(second) == ("8.6", "9.0")
    assert json.dumps(second, sort_keys=True) == json.dumps(first, sort_keys=True), (
        "the two binding orders of one commit left different ledgers")


def test_a_fact_measured_on_other_bytes_is_never_carried_in_either_order():
    for order in (("8.6", "9.0"), ("9.0", "8.6")):
        entry = shipped_entry()
        for capability in order:
            bind(entry, capability, f"{capability} host",
                 supersede=("8.6",) if order[0] == "9.0" and capability == "9.0" else ())
        assert "step_budget" not in entry[fastpath.RUNS]["8.6"], order


def test_a_fact_is_carried_when_the_same_architecture_reruns_on_the_same_bytes():
    """The carry still does its job: a re-run on unchanged bytes keeps what it measured."""
    entry = {"source_sha256": {"kernels.py": NEW}, "status": "PASS"}
    fastpath.bind_capability(entry, bound_before=None, capability="8.6",
                             run={"host": "h", "step_budget": "measured on NEW"})
    bind(entry, "8.6", "the same card, again")
    assert entry[fastpath.RUNS]["8.6"]["step_budget"] == "measured on NEW"


def test_the_carry_reads_only_a_live_record_and_only_the_fields_named():
    entry = shipped_entry()
    assert rebind.carried_run_facts(entry, "8.6", ("step_budget",)) == {
        "step_budget": "measured on the OLD bytes"}
    assert rebind.carried_run_facts(entry, "8.6", ("host",)) == {
        "host": "the original A6000 run"}
    assert rebind.carried_run_facts(entry, "9.0", ("step_budget",)) == {}
    moved = copy.deepcopy(entry)
    moved["source_sha256"] = {"kernels.py": NEW}
    assert rebind.carried_run_facts(moved, "8.6", ("step_budget",)) == {}
