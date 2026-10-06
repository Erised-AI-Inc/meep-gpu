"""The Metal environment: which GPU, torch and Metal frontend, and whether each is certified.

A Metal environment is three facts, each of which changes the code that runs: the
GPU architecture Metal compiles for, the torch that supplies ``compile_shader``,
and the Metal frontend of the macOS build. Each cited weld keeps one run record per
GPU architecture (``runs[<architecture>]``), so the GPU architecture is certified when
every weld the Metal table cites has a live run for it, and torch and the frontend
when every such run recorded this host's. A fact is uncertified when a readable weld
contradicts it, and unjudged when it was not read or a weld's record cannot judge it.
These tests need no GPU except the one that reads this host's architecture.
"""

from __future__ import annotations

import re
import sys

import pytest

from conftest import requires_resource_skip

from meep_gpu import backends, fastpath, metal_dispatch, metal_runs
from meep_gpu.metal_kernels import device

STAMPED = "this machine: Apple MPS device applegpu_g13s, torch 2.10.0, metalfe-32023.850.10"


@pytest.mark.parametrize("host,expected", [
    (STAMPED, {"architecture": "applegpu_g13s", "torch": "2.10.0",
               "metal_frontend": "32023.850.10"}),
    ("this machine: Apple MPS device, torch 2.10.0, metalfe-32023.850.10",
     {"architecture": None, "torch": "2.10.0", "metal_frontend": "32023.850.10"}),
    ("this machine: Apple MPS device, torch 2.10.0",
     {"architecture": None, "torch": "2.10.0", "metal_frontend": None}),
    ("this machine: Apple MPS device, torch ?, ?",
     {"architecture": None, "torch": None, "metal_frontend": None}),
    ("this Mac (Apple silicon, MPS); metal",
     {"architecture": None, "torch": None, "metal_frontend": None}),
    (None, {"architecture": None, "torch": None, "metal_frontend": None}),
])
def test_a_host_line_records_the_facts_it_names_and_none_it_does_not(host, expected):
    assert metal_dispatch.host_environment(host) == expected


def _cite(monkeypatch, *welds):
    """Substitute the cited welds' LIVE runs: one argument per weld.

    A weld is an environment (one live run), a tuple of environments (one live run
    per architecture) or ``None`` (an entry that cannot be read as runs).
    """
    def runs(weld):
        if weld is None:
            return None
        return tuple(dict(environment) for environment in
                     (weld if isinstance(weld, tuple) else (weld,)))

    monkeypatch.setattr(metal_dispatch, "cited_environments", lambda ledger=None: tuple(
        (f"metal_{index}_device_gate", runs(weld)) for index, weld in enumerate(welds)))


FULL = {"architecture": "applegpu_g13s", "torch": "2.10.0", "metal_frontend": "32023.850.10"}
SECOND = {"architecture": "applegpu_g15s", "torch": "2.10.0",
          "metal_frontend": "32023.850.10"}


@pytest.mark.parametrize("recorded,fact,value,verdict", [
    ([FULL, FULL], "architecture", "applegpu_g13s", True),
    ([FULL, FULL], "architecture", "applegpu_g15p", False),
    ([FULL, dict(FULL, architecture="applegpu_g15p")], "architecture", "applegpu_g13s",
     False),
    ([FULL, None], "architecture", "applegpu_g13s", None),
    ([FULL, None], "architecture", "applegpu_g15p", False),
    ([FULL, ()], "architecture", "applegpu_g13s", False),
    ([FULL], "architecture", None, None),
    ([], "torch", "2.10.0", None),
    ([FULL], "metal_frontend", "metalfe-32023.850.10", True),
    ([FULL], "metal_frontend", "32023.850.10", True),
    ([FULL], "metal_frontend", "metalfe-32023.851.1", False),
])
def test_a_fact_is_certified_only_where_every_cited_weld_recorded_it(
        monkeypatch, recorded, fact, value, verdict):
    """``()`` is a readable weld with no live run: it certifies no architecture."""
    _cite(monkeypatch, *recorded)
    architecture = None if fact == "architecture" else "applegpu_g13s"
    assert metal_dispatch.environment_verdict(fact, value, architecture) is verdict


def test_a_second_architecture_on_every_weld_is_certified_beside_the_first(monkeypatch):
    _cite(monkeypatch, (FULL, SECOND), (FULL, SECOND))
    for architecture in ("applegpu_g13s", "applegpu_g15s"):
        assert metal_dispatch.environment_verdict("architecture", architecture) is True
        assert metal_dispatch.environment_verdict("torch", "2.10.0", architecture) is True


def test_a_partial_round_certifies_nothing_new_and_takes_nothing_away(monkeypatch):
    """The intersection rule: one cited weld short and the second GPU is not certified."""
    _cite(monkeypatch, (FULL, SECOND), FULL)
    assert metal_dispatch.environment_verdict("architecture", "applegpu_g15s") is False
    assert metal_dispatch.environment_verdict("architecture", "applegpu_g13s") is True
    judged = metal_dispatch.environment_judgements("architecture", "applegpu_g15s")
    assert [verdict for _gate, verdict in judged] == [True, False]


def test_a_partial_round_does_not_name_the_second_architecture_as_certified(monkeypatch):
    """What a NOTE line calls certified obeys the intersection rule, as the verdict does.

    Every architecture a live run recorded is a candidate; only one every cited weld
    has a run for is listed. Otherwise a GPU one weld short of certified would read
    ``(certified: applegpu_g13s, applegpu_g15s)`` beside its own NOT CERTIFIED.
    """
    _cite(monkeypatch, (FULL, SECOND), FULL)
    assert metal_dispatch.environment_verdict("architecture", "applegpu_g15s") is False
    assert metal_dispatch.certified_values("architecture") == ["applegpu_g13s"]
    assert metal_dispatch.certified_values("torch", "applegpu_g15s") == []
    assert metal_dispatch.certified_values("torch", "applegpu_g13s") == ["2.10.0"]
    _cite(monkeypatch, (FULL, SECOND), (FULL, SECOND))
    assert metal_dispatch.certified_values("architecture") == [
        "applegpu_g13s", "applegpu_g15s"]
    assert metal_dispatch.certified_values("torch", "applegpu_g15s") == ["2.10.0"]
    # RUNS THAT DISAGREE CERTIFY NEITHER VALUE: each is contradicted by the other.
    _cite(monkeypatch, FULL, dict(FULL, torch="2.11.0"))
    assert metal_dispatch.certified_values("torch", "applegpu_g13s") == []


def test_torch_and_the_frontend_are_judged_against_this_architectures_runs(monkeypatch):
    """Each architecture's run records its own torch: one GPU's update is not the other's."""
    newer = dict(SECOND, torch="2.11.0")
    _cite(monkeypatch, (FULL, newer), (FULL, newer))
    assert metal_dispatch.environment_verdict("torch", "2.11.0", "applegpu_g15s") is True
    assert metal_dispatch.environment_verdict("torch", "2.11.0", "applegpu_g13s") is False
    assert metal_dispatch.environment_verdict("torch", "2.10.0", "applegpu_g13s") is True
    # AN ARCHITECTURE NO WELD RAN ON has no run to read torch from: not judged, and
    # its own verdict is the one that says it is not certified.
    assert metal_dispatch.environment_verdict("torch", "2.10.0", "applegpu_g16x") is None
    # AN UNREAD ARCHITECTURE is judged against every live run.
    assert metal_dispatch.environment_verdict("torch", "2.10.0") is False
    _cite(monkeypatch, (FULL, SECOND), FULL)
    assert metal_dispatch.environment_verdict("torch", "2.10.0") is True


def test_the_recorded_environments_count_their_welds_most_common_first(monkeypatch):
    unreadable = {"architecture": None, "torch": None, "metal_frontend": None}
    _cite(monkeypatch, None, FULL, (FULL, SECOND))
    assert metal_dispatch.recorded_environments() == [
        dict(FULL, welds=2), dict(SECOND, welds=1), dict(unreadable, welds=1)]


def _ledger_with_runs(**runs_by_gate):
    """A ledger holding every cited weld, each with the named live runs."""
    ledger = {}
    for gate in metal_runs.cited_keys():
        entry = {"source_sha256": {"meep_gpu/metal_kernels/launch.py": "a" * 64},
                 "status": "PASS"}
        for environment in runs_by_gate.get(gate, (FULL,)):
            metal_runs.bind_architecture(
                entry, bound_before=None, architecture=environment["architecture"],
                run={"torch": environment["torch"],
                     "metal_frontend": environment["metal_frontend"]})
        ledger[gate] = entry
    return ledger


def test_a_run_whose_bytes_moved_is_not_offered_to_the_admission():
    """ONE LIVENESS ANSWER: ``fastpath.live_capabilities`` over the entry's bound."""
    ledger = _ledger_with_runs()
    assert metal_dispatch.environment_verdict(
        "architecture", "applegpu_g13s", ledger=ledger) is True
    gate = metal_runs.cited_keys()[0]
    ledger[gate]["source_sha256"]["meep_gpu/metal_kernels/launch.py"] = "b" * 64
    assert dict(metal_dispatch.cited_environments(ledger=ledger))[gate] == ()
    assert metal_dispatch.environment_verdict(
        "architecture", "applegpu_g13s", ledger=ledger) is False


def test_an_entry_in_the_one_run_shape_is_not_read_both_ways():
    """A weld still holding its host line beside its digests is unreadable, not certified."""
    ledger = _ledger_with_runs()
    gate = metal_runs.cited_keys()[0]
    ledger[gate] = {"source_sha256": {"x.py": "a" * 64}, "host": STAMPED}
    assert dict(metal_dispatch.cited_environments(ledger=ledger))[gate] is None
    assert metal_dispatch.environment_verdict(
        "architecture", "applegpu_g13s", ledger=ledger) is None


def test_the_shipped_ledger_cites_a_readable_entry_for_every_gate():
    cited = metal_dispatch.cited_environments()
    gates = {gate for _family, gate in metal_dispatch.ARM_CERTIFICATION.values()}
    assert [gate for gate, _runs in cited] == sorted(gates)
    assert metal_dispatch.unresolved_certification_rows() == ()
    unreadable = sorted(gate for gate, runs in cited if not runs)
    assert not unreadable, (
        f"{len(unreadable)} cited Metal welds hold no live per-architecture run: "
        f"{unreadable[:4]} (parity/meep_gpu/migrate_metal_runs.py moves a one-run "
        f"ledger; a stale run is repaired by re-running the gate)")


@pytest.mark.parametrize("value,restricted", [
    (None, False), ("1", False), ("0", True), ("true", False), ("", False)])
def test_only_zero_restricts_the_metal_kernels_to_certified_environments(
        monkeypatch, value, restricted):
    if value is None:
        monkeypatch.delenv(fastpath.UNCERTIFIED_SWITCH, raising=False)
    else:
        monkeypatch.setenv(fastpath.UNCERTIFIED_SWITCH, value)
    assert metal_dispatch.certified_only() is restricted


@pytest.mark.parametrize("value,certified", [
    (None, True), ("0", True), ("1", False), ("true", False)])
def test_fast_math_is_certified_only_unset_or_zero(value, certified):
    """Every weld ran with it unset; ``0`` measured identical, ``1`` did not."""
    block = metal_dispatch.environment_block(None, {"fast_math": value})
    assert block["fast_math_certified"] is certified


@pytest.mark.parametrize("architecture", ["applegpu_g13s", "applegpu_g15s", None])
def test_the_architecture_is_written_as_the_key_the_certification_quote_reads(
        architecture):
    """Option B: ``fastpath._certification_for`` reads ``device.compute_capability``.

    The Metal ledger keys its runs by GPU architecture where the NVIDIA ledgers key
    theirs by compute capability, so the device block offers the architecture under
    that key, and a dispatched arm quotes the run its weld recorded on THIS GPU. An
    unread architecture is not spelled as one: the quote then names no run and says
    why.
    """
    block = metal_dispatch.environment_block(None, {"architecture": architecture})
    if architecture is None:
        assert "compute_capability" not in block["device"], block["device"]
    else:
        assert block["device"]["compute_capability"] == architecture
        assert block["device"]["architecture"] == architecture


@pytest.mark.parametrize("architecture,name,supported", [
    ("applegpu_g15p", "Apple M3 Pro", True),
    ("applegpu_g13s", None, True),
    ("other_gpu_arch", "Apple M1 Max", False),
    (None, "Apple M1 Max", True),
    (None, "AMD Radeon Pro 5500M", False),
    (None, None, None),
])
def test_every_apple_gpu_is_supported_whether_or_not_it_is_certified(
        architecture, name, supported):
    """Supported reads the GPU's maker; certified is what the welds measured."""
    toolchain = {"architecture": architecture, "device_name": name}
    assert metal_dispatch.device_supported(toolchain) is supported
    assert metal_dispatch.environment_block(None, toolchain)["device_supported"] is supported


@pytest.fixture
def _fresh_identity(monkeypatch):
    monkeypatch.setattr(device, "_APPLE_GPU", None)
    yield
    monkeypatch.setattr(device, "_APPLE_GPU", None)


def test_the_gpu_identity_off_macos_is_unread_and_says_why(monkeypatch, _fresh_identity):
    monkeypatch.setattr(sys, "platform", "linux")
    assert device.apple_gpu_identity() == {
        "name": None, "architecture": None, "error": "not macOS (linux)"}


def test_the_gpu_identity_is_read_once_and_a_caller_cannot_change_it(_fresh_identity):
    first = device.apple_gpu_identity()
    first["architecture"] = "edited"
    assert device.apple_gpu_identity()["architecture"] != "edited"
    assert set(device.apple_gpu_identity()) == {"name", "architecture", "error"}


def test_this_apple_gpu_names_its_architecture(_fresh_identity):
    if backends.available_gpu() != "metal":
        requires_resource_skip("mps_device", "the architecture is read from an Apple GPU")
    identity = device.apple_gpu_identity()
    assert identity["error"] is None, identity
    assert re.fullmatch(r"applegpu_[a-z0-9_]+", identity["architecture"]), identity
    assert identity["name"], identity


# --- the fastpath hooks the Metal half reports through --------------------------------

def _admitted(*entries):
    record = {}
    for table, because in entries:
        fastpath._admit_uncertified(record, table, "torch", "0.0.0", ["2.10.0"],  # noqa: SLF001
                                    because=because)
    return record


def test_an_opted_in_admission_records_and_says_what_it_always_did():
    record = _admitted(("triton", None))
    assert record["uncertified"]["admitted"] == [
        {"table": "triton", "what": "torch", "read": "0.0.0", "certified": ["2.10.0"]}]
    note = fastpath._uncertified_note(record)  # noqa: SLF001
    assert note.endswith(f". They were dispatched because {fastpath.UNCERTIFIED_SWITCH}=1; "
                         "compare the results with a prefer_gpu=False run of the same "
                         "simulation before relying on them"), note


@pytest.mark.parametrize("reason", ["SUPPORTED_BECAUSE", "UNCERTIFIED_BECAUSE"])
def test_a_table_that_admits_by_default_says_why_in_the_note(reason):
    because = getattr(metal_dispatch, reason).format(switch=fastpath.UNCERTIFIED_SWITCH)
    note = fastpath._uncertified_note(_admitted(("metal", because)))  # noqa: SLF001
    assert f"They were dispatched because {because};" in note, note
    assert f"{fastpath.UNCERTIFIED_SWITCH}=1" not in note, note


@pytest.mark.parametrize("device_certified,verdict", [
    ("absent", True), (True, True), (None, None)])
def test_the_metal_verdict_reads_the_gpu_where_the_record_names_one(device_certified,
                                                                    verdict):
    environment = {"torch_certified": True, "frontend_certified": True}
    if device_certified != "absent":
        environment["device_certified"] = device_certified
    record = {"environment": environment, "uncertified": {"admitted": []}}
    assert fastpath._certified_verdict(record, ["metal"]) is verdict  # noqa: SLF001
