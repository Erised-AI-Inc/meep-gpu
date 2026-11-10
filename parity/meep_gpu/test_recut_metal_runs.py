"""The Metal route record keeps one route run per GPU architecture.

``recut_driver_dispatch_record.py --backend metal`` reads the GPU architecture off
every leg's ``provenance.apple_gpu``, refuses a campaign directory other than
``metal_runs.route_campaign(route_stamps.route_gate("metal"), <architecture>)``, and
files the
route run (its legs, records, timestamp, the run's ``released_fused_arms`` subkeys,
the residency, the lift and the run's statement of the default) under
``runs[<architecture>]``. A second Mac's cut on the same bytes joins the first and
leaves the first's run, its statement of the default included, as it was; a cut that
would change another architecture's run, or write a statement beside the digests, is
refused by name; a cut on moved bytes refuses to strand another architecture's run unless
that architecture is superseded by name; a record still in the one-run shape, or carrying
one shared statement of the default, is refused. Driven here over campaigns built in the
test, against the live tree's digests; no GPU and no results tree are read.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gate_dispatch_fused_route as route_gate  # noqa: E402
import recut_driver_dispatch_record as recut  # noqa: E402

from meep_gpu import fastpath, metal_dispatch, metal_runs, route_stamps  # noqa: E402
from meep_gpu import test_metal_weld_contract as metal_contract  # noqa: E402

#: The stamp these campaigns run under: pinned, so this file does not move with the
#: registry's state (the placeholder until a freeze sets it), and never spent.
STAMP = "2099-01-01_test"
G13S = "applegpu_g13s"
G15S = "applegpu_g15s"
LEGS = ("harness_flush", "harness_keep", "shipped", "band_witness")
BOUND = recut.BACKENDS["metal"]["bound"]


def _live(name: str) -> str:
    return hashlib.sha256(recut._resolve("metal", name).read_bytes()).hexdigest()  # noqa: SLF001


def recorded_default(dispatching: bool, vetoes=()) -> dict:
    """A leg's ``dispatch_preference`` as the route gate writes it, on the live records,
    with the default vetoing ``vetoes`` (product labels) on its one case."""
    rows = ([{"control": "default-preference", "case": "pml_2d",
              "verdict": route_gate.DEFAULT_PREFERENCE_RECORDED, "records_read": {},
              "prior_derived_from": [],
              "vetoes_applied": [{"label": label} for label in vetoes]}]
            if dispatching else [])
    return route_gate.dispatch_preference_summary(rows, route_gate.shipped_timing_records())


def _campaign(results: pathlib.Path, architecture: str, *, legs_on=None, vetoes=()) -> str:
    """A released four-leg Metal route campaign on the live bytes, as the gate writes it."""
    name = metal_runs.route_campaign(route_stamps.route_gate("metal"), architecture)
    released = metal_dispatch.METAL_RELEASED_FUSED_ARMS
    cases = sorted({case for arm_cases in released.values() for case in arm_cases})
    arms_driven = {case: {f"slot{index}": arm
                          for index, arm in enumerate(sorted(released))
                          if case in released[arm]} for case in cases}
    for leg in LEGS:
        directory = results / name / leg
        directory.mkdir(parents=True)
        dispatching = leg not in recut.NON_DISPATCHING_LEGS
        artifact = {
            "release": {"released": True},
            "provenance": {
                "source_sha256": {recut._leg_key("metal", key): _live(key)  # noqa: SLF001
                                  for key in BOUND},
                "apple_gpu": {"architecture": (legs_on or {}).get(leg, architecture),
                              "name": "an Apple GPU", "error": None},
                "lift": {"prefer_gpu": True, "driver_gpu": "metal"},
            },
            "arms_driven": arms_driven if dispatching else {},
            # The timing records the tree ships, stated as the gate states them, with one
            # case's default recorded on a dispatching leg.
            "dispatch_preference": recorded_default(dispatching, vetoes),
            "residency": ({case: {"mode": "held", "invariant": "held between launches"}
                           for case in cases} if dispatching else {}),
        }
        (directory / "gate.json").write_text(json.dumps(artifact))
        rows = [{"case": case, "legs": {"fused": {"plan": {
            "composition": {"tables_dispatched": ["metal"]},
            "environment": {"device_certified": True, "torch_certified": True,
                            "frontend_certified": True,
                            # As the plan writes it since 2026-10-05: the
                            # architecture is also the key the certification quote
                            # reads (``metal_dispatch.environment_block``).
                            "device": {"architecture": architecture,
                                       "compute_capability": architecture}}}}}}
            for case in cases]
        (directory / "cases.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows))
    return name


def _record(path: pathlib.Path, *, digest=None, runs=None, one_run_shape=False) -> None:
    """A Metal ledger holding only a migrated driver_dispatch record."""
    record = {
        "table": "metal",
        "source_sha256": {key: digest or _live(key) for key in BOUND},
        "code_sha256": {},
        "exclusions": {"released_fused_arms": {}},
        "runs": runs or {},
    }
    if runs:
        for architecture, run in runs.items():
            run["bound_sha256"] = fastpath.bound_digest(record)
    if one_run_shape:
        del record["runs"]
        record["residency"] = {"mode": "held"}
    path.write_text(json.dumps({"driver_dispatch": record}, indent=2, sort_keys=True)
                    + "\n")


@pytest.fixture
def tree(tmp_path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    ledger = tmp_path / "fingerprints.json"
    monkeypatch.setattr(recut, "RESULTS", results)
    monkeypatch.setitem(recut.BACKENDS, "metal",
                        dict(recut.BACKENDS["metal"], record=ledger))
    monkeypatch.setattr(route_stamps, "ROUND", STAMP)
    monkeypatch.setattr(route_stamps, "REROUTED", {})
    return results, ledger


def _cut(name: str, *extra: str) -> int:
    return recut.main(["--backend", "metal", "--run", name, "--write", *extra])


def _dispatch(ledger: pathlib.Path) -> dict:
    return json.loads(ledger.read_text())["driver_dispatch"]


def test_a_cut_files_the_route_run_under_the_architecture_its_legs_recorded(tree):
    results, ledger = tree
    _record(ledger)
    name = _campaign(results, G13S)
    assert _cut(name) == 0
    record = _dispatch(ledger)
    assert metal_runs.live_architectures(record) == (G13S,)
    assert metal_runs.shape_reasons(record, run_fields=metal_runs.DISPATCH_RUN_FIELDS) == []
    run = record["runs"][G13S]
    assert run["released_fused_arms"]["gate"] == name
    assert run["records"] == f"apps/api/parity/meep_gpu/results/{name}"
    assert f"results/{name}/" in run["released_fused_arms"]["artifact"]
    assert run["residency"]["mode"] == "held"
    assert run["lift"]["per_leg"]["shipped"]["driver_gpu"] == "metal"
    assert run["status"] == "PASS"
    # The verdict is the run's alone: every cut files ``status`` under ``runs[<key>]``
    # and writes none beside the digests, and the record it writes is one the Metal
    # weld contract's route-record rule accepts, over that contract's own measurements.
    assert "status" not in record
    assert metal_contract._dispatch_record_problems({"driver_dispatch": record}) == []  # noqa: SLF001
    released = record["exclusions"]["released_fused_arms"]
    assert not {"gate", "artifact", "what_was_measured"} & set(released)
    assert set(released["arms"]) == set(metal_dispatch.METAL_RELEASED_FUSED_ARMS)
    assert "residency" not in record and "lift" not in record
    # The entry-level copies say what they were read from and when: the live runs, as
    # of this cut.
    assert "live per-architecture runs" in record["environments"]["read_from"]
    assert "as of this cut" in record["environments"]["read_from"]
    assert "host strings" not in record["toolchain"]["read_from"]
    assert all(len(pair) == 2 for pair in record["toolchain"]["validated"])


def test_a_second_architecture_on_the_same_bytes_joins_the_first(tree):
    results, ledger = tree
    _record(ledger)
    assert _cut(_campaign(results, G13S)) == 0
    first = _dispatch(ledger)["runs"][G13S]
    assert _cut(_campaign(results, G15S)) == 0
    record = _dispatch(ledger)
    assert metal_runs.live_architectures(record) == (G13S, G15S)
    assert record["runs"][G13S] == first


def test_two_architectures_whose_defaults_differ_keep_their_own_statements(
        tree, monkeypatch, capsys):
    """Two stand-in Apple GPUs whose defaults differ: the first's default gives way on its
    one case, the second's does not. Each run keeps its own statement of the default, the
    first's comes through the second's cut byte for byte, and no statement sits beside the
    digests. Then the fault the shared statement was, planted in the writer: a bind that
    also files the cut's statement in the other run is refused by name, nothing written."""
    results, ledger = tree
    _record(ledger)
    assert _cut(_campaign(results, G13S, vetoes=("fused pair D",))) == 0
    first = _dispatch(ledger)["runs"][G13S]
    assert first["dispatch_preference"]["vetoes_applied"] == {"pml_2d": ["fused pair D"]}
    second = _campaign(results, G15S)
    assert _cut(second) == 0
    record = _dispatch(ledger)
    assert recut.PREFERENCE_KEY not in record
    assert record["runs"][G13S] == first
    statement = record["runs"][G15S]["dispatch_preference"]
    assert statement["vetoes_applied"] == {}
    assert statement["run"] == second == record["runs"][G15S]["released_fused_arms"]["gate"]
    assert first["dispatch_preference"]["run"] != second
    # The same cut again replaces only its own run.
    assert _cut(second) == 0
    assert _dispatch(ledger)["runs"][G13S] == first

    real = metal_runs.bind_architecture

    def shared(entry, **kwargs):
        staled = real(entry, **kwargs)
        for run in entry[metal_runs.RUNS].values():
            run[recut.PREFERENCE_KEY] = kwargs["run"][recut.PREFERENCE_KEY]
        return staled

    monkeypatch.setattr(metal_runs, "bind_architecture", shared)
    before = ledger.read_text()
    assert _cut(second) == 1
    error = capsys.readouterr().err
    assert "cut_would_change_another_architecture" in error
    assert f"runs['{G13S}'] fields ['dispatch_preference']" in error
    assert ledger.read_text() == before


def test_a_cut_that_would_write_a_statement_beside_the_digests_is_refused(
        tree, monkeypatch, capsys):
    """The tool checks the record it writes as it checks the one it reads: a planted
    writer that files the cut's statement in its own run and also beside the digests is
    refused by name, nothing written."""
    results, ledger = tree
    _record(ledger)
    real = metal_runs.bind_architecture

    def beside(entry, **kwargs):
        staled = real(entry, **kwargs)
        entry[recut.PREFERENCE_KEY] = kwargs["run"][recut.PREFERENCE_KEY]
        return staled

    monkeypatch.setattr(metal_runs, "bind_architecture", beside)
    before = ledger.read_text()
    assert _cut(_campaign(results, G13S)) == 1
    error = capsys.readouterr().err
    assert "cut_would_write_route_run_fields_at_entry_level" in error
    assert "'dispatch_preference'" in error
    assert ledger.read_text() == before


def test_a_record_carrying_one_shared_statement_of_the_default_is_refused(tree, capsys):
    """The shape every cut overwrote: refused by name before any leg is read."""
    results, ledger = tree
    _record(ledger)
    document = json.loads(ledger.read_text())
    document["driver_dispatch"][recut.PREFERENCE_KEY] = recorded_default(True)
    ledger.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    before = ledger.read_text()
    assert _cut(_campaign(results, G13S)) == 2
    error = capsys.readouterr().err
    assert "record_carries_route_run_fields_at_entry_level" in error
    assert "'dispatch_preference'" in error
    assert ledger.read_text() == before


def test_a_cut_on_moved_bytes_names_the_architecture_it_would_strand(tree, capsys):
    results, ledger = tree
    _record(ledger, digest="0" * 64, runs={G15S: {"status": "PASS"}})
    before = ledger.read_text()
    name = _campaign(results, G13S)
    assert _cut(name) == 1
    assert ledger.read_text() == before
    assert "applegpu_g15s" in capsys.readouterr().err
    assert _cut(name, "--supersede", G15S) == 0
    assert metal_runs.live_architectures(_dispatch(ledger)) == (G13S,)


def test_a_run_directory_that_is_not_this_architectures_campaign_is_refused(tree, capsys):
    results, ledger = tree
    _record(ledger)
    name = _campaign(results, G13S)
    renamed = results / metal_runs.route_campaign(route_stamps.route_gate("metal"), G15S)
    (results / name).rename(renamed)
    assert _cut(renamed.name) == 2
    assert "run_is_not_this_architecture_campaign" in capsys.readouterr().err


def test_legs_that_name_two_architectures_are_refused(tree, capsys):
    results, ledger = tree
    _record(ledger)
    name = _campaign(results, G13S, legs_on={"harness_keep": G15S})
    assert _cut(name) == 1
    assert "legs_do_not_name_one_gpu_architecture" in capsys.readouterr().err


def test_a_record_in_the_one_run_shape_is_refused(tree, capsys):
    results, ledger = tree
    _record(ledger, one_run_shape=True)
    assert _cut(_campaign(results, G13S)) == 2
    assert "run migrate_metal_runs.py first" in capsys.readouterr().err


# The campaign name is the registry's, and the cut keys on it (meep_gpu/route_stamps.py).
# Each of these must refuse with nothing written; each is a campaign the legs would
# otherwise file cleanly (the first test above cuts the same campaign shape).


def test_a_cut_under_the_placeholder_stamp_is_refused_by_name(tree, monkeypatch, capsys):
    results, ledger = tree
    _record(ledger)
    monkeypatch.setattr(route_stamps, "ROUND", route_stamps.PLACEHOLDER)
    name = _campaign(results, G13S)
    before = ledger.read_text()
    assert _cut(name) == 2
    assert "route_stamp_is_unset" in capsys.readouterr().err
    assert ledger.read_text() == before


def test_a_spent_campaign_is_refused_by_name(tree, monkeypatch, capsys):
    results, ledger = tree
    _record(ledger)
    name = _campaign(results, G13S)
    monkeypatch.setitem(route_stamps.SPENT, "metal",
                        route_stamps.SPENT["metal"] + (name,))
    before = ledger.read_text()
    assert _cut(name) == 2
    assert "route_campaign_is_spent" in capsys.readouterr().err
    assert ledger.read_text() == before


def test_a_campaign_of_an_earlier_stamp_is_refused_once_the_stamp_moves(
        tree, monkeypatch, capsys):
    """A table re-routed alone: the registry names a new Metal stamp, and the run the
    old one spelled is no longer the release's, whatever it measured."""
    results, ledger = tree
    _record(ledger)
    name = _campaign(results, G13S)
    monkeypatch.setattr(route_stamps, "REROUTED", {"metal": "2099-01-02_test"})
    before = ledger.read_text()
    assert _cut(name) == 2
    assert "run_is_not_this_architecture_campaign" in capsys.readouterr().err
    assert ledger.read_text() == before
