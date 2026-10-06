"""Per-architecture run records on the Metal ledger: the write rules and what they decide.

The Metal half of ``test_capability_records.py``. A Metal weld entry keeps its digests
and one run record per Apple GPU architecture under ``runs[<architecture>]``; a run is
live while its ``bound_sha256`` is ``fastpath.bound_digest`` of the entry. What is held
here: the bound covers the bytes and nothing else, a second architecture on unchanged
bytes JOINS the first, a write on moved bytes refuses to strand the others unless they
are superseded by name, the key is an ``applegpu_*`` architecture and nothing else, the
one-run shape is refused by name, the admitted set is an intersection, and a dispatched
Metal arm quotes the run its weld recorded on THIS GPU. Stdlib and the package; no GPU.
"""

from __future__ import annotations

import copy

import pytest

from . import fastpath, metal_dispatch, metal_runs
from . import weld_record_walk as walk

PIN = "meep_gpu/metal_kernels/launch.py"
G13S = "applegpu_g13s"
G15S = "applegpu_g15s"


def host(architecture: str, torch: str = "2.10.0",
         frontend: str = "metalfe-32023.850.10") -> str:
    return f"this machine: Apple MPS device {architecture}, torch {torch}, {frontend}"


def run_for(architecture: str, **extra) -> dict:
    named = metal_runs.environment_of(host(architecture))
    run = {"host": host(architecture), "torch": named["torch"],
           "metal_frontend": named["metal_frontend"],
           "records": f"apps/api/parity/meep_gpu/results/metal_x_{architecture}/ - 1 records",
           "recorded_utc": "2026-10-05T00:00:00Z", "artifact_sha256": "c" * 64,
           "subnormal_policy": "flush - native and uncontrollable on MPS",
           "environment_read_from": f"metal_environment_{architecture}/start.json",
           "verdict_read_from": "gate.json:release.released"}
    run.update(extra)
    return run


def weld(*architectures: str, digest: str = "a" * 64) -> dict:
    entry = {"source_sha256": {PIN: digest}, "code_sha256": {PIN: "b" * 64},
             "status": "PASS", "purpose": "a curated claim"}
    for architecture in architectures:
        metal_runs.bind_architecture(entry, bound_before=None, architecture=architecture,
                                     run=run_for(architecture))
    return entry


# ---------------------------------------------------------------------------
# The bound, liveness and the staleness rule
# ---------------------------------------------------------------------------


def test_the_bound_is_over_the_bytes_the_run_certified_and_nothing_else():
    entry = weld(G13S)
    assert entry["runs"][G13S]["bound_sha256"] == fastpath.bound_digest(entry)
    moved = copy.deepcopy(entry)
    moved["code_sha256"][PIN] = "d" * 64
    moved["purpose"] = "another sentence"
    moved["device_sha256"] = {PIN: {"kind": "metal", "digests": {"k": "e" * 64}}}
    assert metal_runs.live_architectures(moved) == (G13S,), (
        "only fastpath.BOUND_FIELDS decide liveness; the device and code tiers are the "
        "weld contract's to check")


def test_a_run_stops_being_live_when_the_bytes_it_certified_move():
    entry = weld(G13S)
    entry["source_sha256"][PIN] = "f" * 64
    assert metal_runs.live_architectures(entry) == ()
    assert metal_runs.live_run(entry, G13S) is None


def test_a_second_architecture_on_the_same_bytes_joins_the_first():
    entry = weld(G13S)
    first = copy.deepcopy(entry["runs"][G13S])
    staled = metal_runs.bind_architecture(
        entry, bound_before=fastpath.bound_digest(entry), architecture=G15S,
        run=run_for(G15S))
    assert staled == ()
    assert metal_runs.live_architectures(entry) == (G13S, G15S)
    assert entry["runs"][G13S] == first


def test_a_run_on_bytes_that_moved_refuses_rather_than_stranding_the_others():
    entry = weld(G13S)
    before = fastpath.bound_digest(entry)
    entry["source_sha256"][PIN] = "f" * 64
    with pytest.raises(metal_runs.ArchitectureStale) as refused:
        metal_runs.bind_architecture(entry, bound_before=before, architecture=G15S,
                                     run=run_for(G15S))
    assert refused.value.names == (G13S,)
    assert "--supersede applegpu_g13s" in str(refused.value)
    assert isinstance(refused.value, fastpath.CapabilityRecordError)
    staled = metal_runs.bind_architecture(entry, bound_before=before, architecture=G15S,
                                          run=run_for(G15S), supersede=(G13S,))
    assert staled == (G13S,)
    assert metal_runs.live_architectures(entry) == (G15S,)
    assert G13S in entry["runs"], "a superseded run is kept as history, not deleted"


def test_the_architecture_being_written_is_never_in_its_own_stale_set():
    entry = weld(G13S)
    before = fastpath.bound_digest(entry)
    entry["source_sha256"][PIN] = "f" * 64
    assert metal_runs.bind_architecture(entry, bound_before=before, architecture=G13S,
                                        run=run_for(G13S)) == ()
    assert metal_runs.live_architectures(entry) == (G13S,)


def test_a_field_that_describes_the_bytes_is_refused_inside_a_run_record():
    entry = weld()
    with pytest.raises(metal_runs.ArchitectureRecordError, match="not run fields"):
        metal_runs.bind_architecture(entry, bound_before=None, architecture=G13S,
                                     run=run_for(G13S, source_sha256={PIN: "a" * 64}))


def test_an_entry_that_pins_no_bytes_is_refused_rather_than_bound_to_nothing():
    with pytest.raises(fastpath.CapabilityRecordError):
        metal_runs.bind_architecture({"status": "PASS"}, bound_before=None,
                                     architecture=G13S, run=run_for(G13S))


@pytest.mark.parametrize("key", ["8.6", "Apple M1 Max", "applegpu-g13s", "APPLEGPU_G13S",
                                 "", None, "applegpu_"])
def test_a_key_that_is_not_a_gpu_architecture_is_refused(key):
    with pytest.raises(metal_runs.ArchitectureRecordError, match="not a GPU architecture"):
        metal_runs.require_architecture(key)
    with pytest.raises(metal_runs.ArchitectureRecordError):
        metal_runs.bind_architecture(weld(), bound_before=None, architecture=key,
                                     run={})


def test_a_supersede_list_is_held_to_the_same_key_rule():
    entry = weld(G13S)
    with pytest.raises(metal_runs.ArchitectureRecordError):
        metal_runs.bind_architecture(entry, bound_before=None, architecture=G15S,
                                     run=run_for(G15S), supersede=("8.6",))


# ---------------------------------------------------------------------------
# The shape, refused by name
# ---------------------------------------------------------------------------


def test_the_per_architecture_shape_reads_clean():
    assert metal_runs.shape_reasons(weld(G13S, G15S)) == []


def test_the_one_run_shape_it_replaced_is_refused_by_name():
    entry = {"source_sha256": {PIN: "a" * 64}, "host": host(G13S),
             "recorded_utc": "2026-10-01T00:00:00Z"}
    reasons = metal_runs.shape_reasons(entry)
    assert any("run fields beside the digests" in reason for reason in reasons), reasons
    assert any("no 'runs' record" in reason for reason in reasons), reasons


def test_a_run_keyed_by_something_other_than_an_architecture_is_refused():
    entry = weld(G13S)
    entry["runs"]["8.6"] = dict(entry["runs"][G13S])
    assert any("'8.6'] is not a GPU architecture key" in reason
               for reason in metal_runs.shape_reasons(entry))


@pytest.mark.parametrize("field,value,says", [
    ("host", host(G15S), "host names architecture 'applegpu_g15s'"),
    ("torch", "2.11.0", "torch is '2.11.0' and its host line names '2.10.0'"),
    ("metal_frontend", "32023.851.1", "metal_frontend is '32023.851.1'"),
])
def test_a_run_whose_fields_disagree_with_its_host_line_is_refused(field, value, says):
    """The line a reader sees and the fields the admission compares are one claim."""
    entry = weld(G13S)
    entry["runs"][G13S][field] = value
    reasons = metal_runs.shape_reasons(entry)
    assert any(says in reason for reason in reasons), reasons


def test_the_frontend_is_compared_in_the_one_spelling_the_admission_uses():
    entry = weld(G13S)
    entry["runs"][G13S]["metal_frontend"] = "metalfe-32023.850.10"
    assert metal_runs.shape_reasons(entry) == []


def test_the_run_fields_name_the_environment_the_admission_reads():
    assert metal_runs.ENVIRONMENT_FIELDS == ("torch", "metal_frontend")
    assert set(metal_runs.ENVIRONMENT_FIELDS) < set(metal_dispatch.ENVIRONMENT_FACTS)
    assert "environment_read_from" in metal_runs.RUN_FIELDS
    assert not set(metal_runs.RUN_FIELDS) & set(fastpath.BOUND_FIELDS)
    assert "status" not in metal_runs.RUN_FIELDS, "status describes the weld, not a run"


def test_the_walker_the_dispatcher_and_the_writers_agree_on_where_a_run_lives():
    assert metal_runs.RUNS == fastpath.RUNS == walk.RUNS
    found = walk.census({"g": weld(G13S, G15S)})
    rules = sorted(f.rule for f in found if "bound_sha256" in f.trail)
    assert rules == ["run_bound_digest", "run_bound_digest"], rules
    assert not [f for f in found if f.rule == walk.UNCLASSIFIED]


# ---------------------------------------------------------------------------
# The intersection, and the route campaign's name
# ---------------------------------------------------------------------------


def test_the_table_claims_only_what_every_cited_weld_ran_on():
    ledger = {"a": weld(G13S, G15S), "b": weld(G13S)}
    report = metal_runs.admission_report(ledger, ["a", "b"])
    assert report["admitted"] == (G13S,)
    assert report["by_key"]["a"]["live"] == [G13S, G15S]
    assert metal_runs.admission_report({"a": weld(G15S), "b": weld(G13S)},
                                       ["a", "b"])["admitted"] == ()


def test_a_cited_weld_with_no_record_or_the_old_shape_blocks_by_name():
    old = {"source_sha256": {PIN: "a" * 64}, "host": host(G13S)}
    report = metal_runs.admission_report({"a": weld(G13S), "b": old}, ["a", "b", "c"])
    assert report["admitted"] == ()
    assert "no 'runs' record" in report["by_key"]["b"]["problem"]
    assert report["by_key"]["c"]["problem"] == "no entry under 'c'"


def test_an_unreadable_ledger_is_unknown():
    assert metal_runs.admission_report({}, ["a"])["admitted"] is None


@pytest.mark.parametrize("architecture,expected", [
    (G13S, "dispatch_metal_route_2026-10-05_perarch_applegpu_g13s"),
    (G15S, "dispatch_metal_route_2026-10-05_perarch_applegpu_g15s"),
])
def test_a_route_campaign_names_the_architecture_it_ran_on(architecture, expected):
    assert metal_runs.route_campaign("dispatch_metal_route_2026-10-05_perarch",
                                     architecture) == expected
    with pytest.raises(metal_runs.ArchitectureRecordError):
        metal_runs.route_campaign("dispatch_metal_route_x", "8.6")


def test_the_weld_keys_are_the_entries_that_pin_bytes():
    ledger = {"driver_dispatch": {"source_sha256": {}}, "metal_kernels": {},
              "_architecture_runs_migration": {}, "a": weld(G13S), "b": {"note": 1}}
    assert metal_runs.weld_keys(ledger) == ["a"]


# ---------------------------------------------------------------------------
# The certification quote a dispatched Metal arm carries (option B)
# ---------------------------------------------------------------------------


@pytest.fixture
def quoted_ledger():
    """A Metal ledger, installed where ``fastpath._certification_for`` reads it."""
    arm, (family, gate) = next(iter(sorted(metal_dispatch.ARM_CERTIFICATION.items())))
    entry = weld(G13S)
    metal_runs.bind_architecture(entry, bound_before=None, architecture=G15S,
                                 run=run_for(G15S, recorded_utc="2026-10-09T00:00:00Z"))
    saved = fastpath._FINGERPRINTS.get("metal_kernels")  # noqa: SLF001
    fastpath._FINGERPRINTS["metal_kernels"] = {gate: entry}  # noqa: SLF001
    yield arm, family
    if saved is None:
        fastpath._FINGERPRINTS.pop("metal_kernels", None)  # noqa: SLF001
    else:
        fastpath._FINGERPRINTS["metal_kernels"] = saved  # noqa: SLF001


def test_a_metal_arm_quotes_the_run_its_weld_recorded_on_this_gpu(quoted_ledger):
    """What ``environment_block`` writing the architecture as ``compute_capability`` buys.

    ``fastpath._finish`` hands the device block's ``compute_capability`` to
    ``_certification_for``; on a Metal plan that is the GPU architecture, so the arm
    quotes THAT architecture's run -- its host line and its timestamp -- and no other.
    """
    arm, family = quoted_ledger
    for architecture, when in ((G13S, "2026-10-05T00:00:00Z"),
                               (G15S, "2026-10-09T00:00:00Z")):
        quote = fastpath._certification_for(arm, "metal",  # noqa: SLF001
                                            capability=architecture)
        assert quote["family"] == family
        assert quote["capability"] == architecture
        assert quote["host"] == host(architecture)
        assert quote["recorded_utc"] == when
        assert quote["capabilities_live"] == [G13S, G15S]


def test_a_gpu_no_weld_ran_on_quotes_no_other_gpus_run(quoted_ledger):
    arm, _family = quoted_ledger
    quote = fastpath._certification_for(arm, "metal",  # noqa: SLF001
                                        capability="applegpu_g16x")
    assert "host" not in quote and "recorded_utc" not in quote, quote
    assert "applegpu_g16x" in quote["run_record"]
    unread = fastpath._certification_for(arm, "metal", capability=None)  # noqa: SLF001
    assert "host" not in unread and "could not be read" in unread["run_record"]
