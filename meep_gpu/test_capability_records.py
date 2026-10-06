"""One certification record per GPU architecture: is a second one ADDITIVE?

WHAT THIS FILE IS FOR. Each weld in the NVIDIA ledgers keeps one record per compute
capability, holding the facts of the run that certified that architecture and the digest
of the bytes it certified. Everything below is about the three ways that can go wrong:

* a second architecture's run REPLACES the first instead of joining it, which is how a
  table that was certified on an A6000 comes to claim an H100 it never ran on;
* a run's record outlives the bytes it certified, so an edit to a kernel silently keeps
  every architecture admitted;
* the table claims an architecture only SOME of its families ran on, so an arm served by
  one of the others dispatches on a card no gate ever measured.

Every test here runs on a laptop with no GPU: the ledgers are copied in memory and the
device is a stub, because what is under test is the bookkeeping, not a kernel.
"""

from __future__ import annotations

import copy
import json
import pathlib
import types

import pytest

import meep_gpu.fastpath as fastpath
import meep_gpu.fastpath_cuda as fastpath_cuda
from meep_gpu import weld_record_walk as walk

from .test_dispatch_contract import (CountingPlan, assert_the_family_block_is_its_artifacts,
                                     assert_the_family_block_names_its_own_run,
                                     certified_envelope_fields, certified_envelope_grid,
                                     certified_envelope_pml, cuda_step_plan,
                                     cupy_like_grid_with_device, install_composer,
                                     install_cuda_composer, step_plan, stub_triton)


@pytest.fixture(autouse=True)
def _opted_in_and_quiet(monkeypatch):
    """The ladder's first two rungs defaulted to "not vetoed, opted in".

    The suite pins dispatch OFF (its conftest does, so the package tests exercise the
    array path), and every test below is about a rung underneath that one.
    """
    monkeypatch.delenv(fastpath.FUSED_KILL_SWITCH, raising=False)
    monkeypatch.setenv(fastpath.DISPATCH_ENABLE, "1")
    monkeypatch.delenv(fastpath.DISPATCH_LOG, raising=False)
    monkeypatch.delenv(fastpath.WARM_SWITCH, raising=False)
    monkeypatch.delenv("MEEP_GPU_ALLOW_UNCERTIFIED", raising=False)
    fastpath.reset_dispatch_announcements()
    yield
    fastpath.reset_dispatch_announcements()

A_DIGEST = "a" * 64
B_DIGEST = "b" * 64


def entry_on(bytes_digest: str = A_DIGEST) -> dict:
    """A weld entry that pins one file, with no run recorded yet."""
    return {"source_sha256": {"kernels.py": bytes_digest}, "status": "PASS"}


def run_on(capability: str) -> dict:
    return {"host": f"a host, cc {capability}", "records": f"results/run_cc{capability}",
            "recorded_utc": "2026-09-30T00:00:00Z"}


def certify(entry: dict, capability: str, **kwargs) -> tuple:
    """Bind ``capability`` to ``entry``'s current bytes."""
    return fastpath.bind_capability(entry, capability=capability,
                                    run=run_on(capability), **kwargs)


# ---------------------------------------------------------------------------
# What a run record is bound to
# ---------------------------------------------------------------------------


def test_the_bound_is_over_the_bytes_the_run_certified_and_nothing_else():
    """The digest of the DIGESTS, so "do these records still describe the shipped
    bytes" is one comparison rather than a re-reading of the tree."""
    entry = entry_on()
    first = fastpath.bound_digest(entry)
    # A run fact is not part of it: two runs of the same bytes bind the same digest.
    entry["host"] = "another host"
    entry["recorded_utc"] = "2026-01-01T00:00:00Z"
    assert fastpath.bound_digest(entry) == first
    # Nor is anything recomputed from the checkout at bind time.
    entry["code_sha256"] = {"kernels.py": B_DIGEST}
    entry["device_sha256"] = {"plain_curl_step": B_DIGEST}
    assert fastpath.bound_digest(entry) == first
    # The pinned bytes are.
    entry["source_sha256"]["kernels.py"] = B_DIGEST
    assert fastpath.bound_digest(entry) != first


def test_an_entry_that_pins_no_bytes_is_refused_rather_than_bound_to_nothing():
    """A record bound to an empty set would be live forever, through any edit."""
    with pytest.raises(fastpath.CapabilityRecordError, match="binds no bytes"):
        fastpath.bound_digest({"status": "PASS", "host": "a host"})


def test_a_record_stops_being_live_when_the_bytes_it_certified_move():
    entry = entry_on()
    certify(entry, "8.6", bound_before=None)
    assert fastpath.live_capabilities(entry) == ("8.6",)
    entry["source_sha256"]["kernels.py"] = B_DIGEST
    assert fastpath.live_capabilities(entry) == ()
    assert "every recorded run binds bytes that have since moved" in \
        fastpath.capability_report({"g": entry}, ["g"])["by_key"]["g"]["problem"]


# ---------------------------------------------------------------------------
# Adding a second architecture
# ---------------------------------------------------------------------------


def test_a_second_architecture_on_the_same_bytes_joins_the_first():
    """THE WHOLE POINT. Certifying 9.0 must not un-certify 8.6."""
    entry = entry_on()
    certify(entry, "8.6", bound_before=None)
    before = copy.deepcopy(entry["runs"]["8.6"])
    certify(entry, "9.0", bound_before=fastpath.bound_digest(entry))
    assert fastpath.live_capabilities(entry) == ("8.6", "9.0")
    assert entry["runs"]["8.6"] == before, "the first architecture's record moved"
    assert entry["runs"]["9.0"]["host"].endswith("cc 9.0")


def test_a_run_on_bytes_that_moved_refuses_rather_than_stranding_the_others():
    """A rebind on changed bytes invalidates every other architecture's evidence,
    and that is a decision a person makes: the refusal names what would be lost."""
    entry = entry_on()
    certify(entry, "8.6", bound_before=None)
    bound = fastpath.bound_digest(entry)
    entry["source_sha256"]["kernels.py"] = B_DIGEST          # a kernel was edited
    before = copy.deepcopy(entry["runs"])
    with pytest.raises(fastpath.CapabilityStale) as refused:
        certify(entry, "9.0", bound_before=bound)
    assert refused.value.names == ("8.6",)
    assert "re-run those capabilities" in str(refused.value)
    assert "9.0" not in entry.get("runs", {}), "a refused write left a record behind"
    # A refusal changes NOTHING. Not writing 9.0 is half of it; the other half is that
    # the record it refused to strand comes out byte-identical, so a refusal can never
    # be the thing that damages the architecture it is protecting.
    assert entry["runs"] == before, "a refused write altered another architecture"
    # Named, it goes through, and the superseded architecture is no longer live.
    assert certify(entry, "9.0", bound_before=bound, supersede=("8.6",)) == ("8.6",)
    assert fastpath.live_capabilities(entry) == ("9.0",)


def test_the_capability_being_written_is_never_in_its_own_stale_set():
    """Re-running 8.6 on new bytes is the ordinary case and must not need --supersede."""
    entry = entry_on()
    certify(entry, "8.6", bound_before=None)
    bound = fastpath.bound_digest(entry)
    entry["source_sha256"]["kernels.py"] = B_DIGEST
    assert certify(entry, "8.6", bound_before=bound) == ()
    assert fastpath.live_capabilities(entry) == ("8.6",)


def test_a_field_that_describes_the_bytes_is_refused_inside_a_run_record():
    entry = entry_on()
    with pytest.raises(fastpath.CapabilityRecordError, match="not run fields"):
        fastpath.bind_capability(entry, bound_before=None, capability="8.6",
                                 run={"source_sha256": {"kernels.py": A_DIGEST}})


@pytest.mark.parametrize("capability", ["86", "8_6", "", "sm_86", "8.6 "])
def test_a_capability_that_is_not_the_one_spelling_is_refused(capability):
    """CuPy answers '86' and '8.6'; a record keyed by both would hide one of them."""
    with pytest.raises(fastpath.CapabilityRecordError, match="capability"):
        fastpath.bind_capability(entry_on(), bound_before=None,
                                 capability=capability, run=run_on("8.6"))


# ---------------------------------------------------------------------------
# What the table claims
# ---------------------------------------------------------------------------


def two_welds(second_capability: str = None) -> dict:
    ledger = {"g1": entry_on(), "g2": entry_on()}
    for entry in ledger.values():
        certify(entry, "8.6", bound_before=None)
    if second_capability:
        certify(ledger["g1"], second_capability,
                bound_before=fastpath.bound_digest(ledger["g1"]))
    return ledger


def test_a_table_claims_only_what_every_family_it_dispatches_ran_on():
    """An arm can be served from any certified family, so the answer is the
    INTERSECTION: a capability one family never ran on is one the table cannot claim."""
    ledger = two_welds("9.0")
    assert fastpath.table_capabilities(ledger, ["g1"]) == ("8.6", "9.0")
    assert fastpath.table_capabilities(ledger, ["g1", "g2"]) == ("8.6",)
    report = fastpath.capability_report(ledger, ["g1", "g2"])
    assert report["by_key"]["g2"]["live"] == ["8.6"] and not report["by_key"]["g2"]["problem"]


def test_a_cited_weld_with_no_record_at_all_blocks_the_whole_table_by_name():
    ledger = two_welds()
    report = fastpath.capability_report(ledger, ["g1", "nowhere"])
    assert report["admitted"] == ()
    assert "no entry under 'nowhere'" in report["by_key"]["nowhere"]["problem"]


def test_an_unreadable_ledger_is_unknown_and_an_empty_one_refuses():
    """The two empties are different facts and the device rung needs them apart."""
    assert fastpath.table_capabilities({"_unreadable": "no such file"}, ["g1"]) is None
    # None is "the ledger could not be read" and () is "it reads and shares nothing";
    # the first is unknown and the second refuses.
    assert fastpath._device_is_validated({"compute_capability": "8.6"}, None) is None
    assert fastpath._device_is_validated({"compute_capability": "8.6"}, ()) is False
    assert fastpath._device_is_validated({"unreadable": "x"}, ("8.6",)) is None
    assert fastpath._device_is_validated({"compute_capability": "90"}, ("9.0",)) is True


def test_an_entry_in_the_shape_this_replaced_is_refused_by_name():
    """No reader accepts both shapes: "which bytes did this run certify" would then
    have two answers, which is the defect the per-capability record closes."""
    old = {"source_sha256": {"kernels.py": A_DIGEST}, "host": "a host",
           "records": "results/x", "recorded_utc": "2026-01-01T00:00:00Z"}
    reasons = fastpath.retired_shape_reasons(old)
    assert any("run fields beside the digests" in reason for reason in reasons)
    assert any("no 'runs' record" in reason for reason in reasons)
    assert fastpath.retired_shape_reasons(
        {"source_sha256": {"k": A_DIGEST},
         "runs": {"8.6": {"bound_sha256": A_DIGEST}}}) == []


def test_a_run_record_keyed_by_something_other_than_a_capability_is_refused():
    entry = {"source_sha256": {"k": A_DIGEST},
             "runs": {"sm_90": {"bound_sha256": A_DIGEST}}}
    assert any("not a normalised capability" in reason
               for reason in fastpath.retired_shape_reasons(entry))


# ---------------------------------------------------------------------------
# The shipped ledgers
# ---------------------------------------------------------------------------


def test_both_nvidia_tables_derive_the_architecture_they_were_certified_on():
    """DERIVED from the welds, not typed: the hand-typed key it replaced was written
    by no tool, so a round on another architecture left it saying 8.6 regardless."""
    assert fastpath.validated_compute_capabilities() == ("8.6",)
    assert fastpath_cuda.validated_compute_capabilities() == ("8.6",)
    for table in ("triton", fastpath.CUDA_TABLE):
        report = fastpath.capability_admission(table)
        blocking = {key: state["problem"] for key, state in report["by_key"].items()
                    if state["problem"]}
        assert not blocking, blocking
        assert report["admitted"] == ("8.6",)


def test_no_shipped_nvidia_weld_is_left_in_the_shape_this_replaced():
    for table in ("triton", fastpath.CUDA_TABLE):
        ledger = fastpath._fingerprints(table)
        cited = {gate for _family, gate in
                 fastpath._arm_certification_map(table).values()}
        for key in sorted(cited):
            assert fastpath.retired_shape_reasons(ledger[key]) == [], key


def test_the_walker_and_the_dispatcher_agree_on_where_a_run_record_lives():
    """The walker is a standalone classifier and spells the key itself."""
    assert walk.RUNS == fastpath.RUNS


def test_the_family_record_binds_the_bytes_its_nine_families_re_ran():
    """It pins nothing of its own otherwise, so its records could never go stale --
    and 22 arms cite it, which would have made the whole container decoration."""
    entry = fastpath._fingerprints("triton")[fastpath.FAMILY_RECERT_GATE]
    assert entry["source_sha256_at_recert"], "it binds no bytes"
    assert fastpath.live_capabilities(entry) == ("8.6",)
    moved = copy.deepcopy(entry)
    name = sorted(moved["source_sha256_at_recert"])[0]
    moved["source_sha256_at_recert"][name] = B_DIGEST
    assert fastpath.live_capabilities(moved) == ()


# ---------------------------------------------------------------------------
# What the plan quotes, and what it dispatches
# ---------------------------------------------------------------------------


def test_a_family_quotes_the_run_that_certified_THIS_architecture():
    """The plan quotes the 8.6 run's host, timestamp and the family's OWN block.

    The budget used to be pinned as the literal "78/78" of the retired 2026-08-14
    transcription. The family re-cut now copies each gate's budget from its artifact,
    so what is asserted is the chain: the quoted budget and run id are the complex
    family's block inside ``runs["8.6"]``, the complex gate states a budget (a
    non-empty block), and the block names its own run. That the block is what the
    gate's artifact states is the next test's, which needs the evidence archive.
    """
    entry = fastpath._certification_for("complex", capability="8.6")
    assert entry["capability"] == "8.6" and entry["capabilities_live"] == ["8.6"]
    assert "A6000" in entry["host"] and entry["recorded_utc"]
    run = fastpath._fingerprints("triton")[fastpath.FAMILY_RECERT_GATE][fastpath.RUNS]["8.6"]
    block = run["families"]["complex"]
    assert entry["host"] == run["host"] and entry["recorded_utc"] == run["recorded_utc"]
    assert entry["run_id"] == block["run_id"]
    assert entry["step_budget"] == block["step_budget"], (
        "the per-family budget comes from the family's block in this architecture's run")
    assert isinstance(entry["step_budget"], dict) and entry["step_budget"], (
        f"the complex gate states its budget, and the block reads "
        f"{entry['step_budget']!r}")
    assert_the_family_block_names_its_own_run("complex", block, run)


def test_the_quoted_family_block_is_the_artifact_of_its_own_run():
    """The complex family's block, read back from the gate run it names.

    The artifact and its log hash to the block's pins, the campaign row agrees on exit
    code, release, artifact digest, start time and GPU index, and the budget is the
    one the artifact states -- the checks ``test_dispatch_contract`` applies to all
    nine families. Declared under ``[evidence_archive]`` in
    ``tools/ci/declared_resources.txt``: it reads the round's own artifacts, which a
    checkout holds only with the archive restored.
    """
    run = fastpath._fingerprints("triton")[fastpath.FAMILY_RECERT_GATE][fastpath.RUNS]["8.6"]
    assert_the_family_block_is_its_artifacts(
        "complex", run["families"]["complex"],
        pathlib.Path(fastpath.__file__).resolve().parent)


def test_a_family_quotes_no_run_for_an_architecture_it_was_not_certified_on():
    """Quoting 8.6's host beside a plan on another card is the mismatch this closes."""
    entry = fastpath._certification_for("complex", capability="9.0")
    assert "capability" not in entry and "host" not in entry
    assert "no live run on compute capability 9.0" in entry["run_record"]
    unread = fastpath._certification_for("complex", capability=None)
    assert "could not be read" in unread["run_record"]


def test_the_cuda_table_answers_the_same_way():
    entry = fastpath_cuda.certification_for("mirror fill", capability="8.6")
    assert entry["capability"] == "8.6" and entry["host"]
    miss = fastpath_cuda.certification_for("mirror fill", capability="9.0")
    assert "capability" not in miss and "no live run" in miss["run_record"]


def ledger_also_certified_on(table: str, capability: str, skip: str = None) -> dict:
    """A copy of a shipped ledger with ``capability`` certified on the same bytes."""
    ledger = copy.deepcopy(dict(fastpath._fingerprints(table)))
    cited = {gate for _family, gate in fastpath._arm_certification_map(table).values()}
    for key in sorted(cited):
        if key == skip:
            continue
        fastpath.bind_capability(
            ledger[key], bound_before=fastpath.bound_digest(ledger[key]),
            capability=capability, run=run_on(capability))
    return ledger


@pytest.fixture
def certified_on_9_0(monkeypatch):
    """Both NVIDIA ledgers with a 9.0 run recorded for every cited weld."""
    def install(skip_triton=None, skip_cuda=None):
        ledgers = {
            "triton_kernels": ledger_also_certified_on("triton", "9.0", skip_triton),
            "cuda_kernels": ledger_also_certified_on(fastpath.CUDA_TABLE, "9.0",
                                                     skip_cuda),
        }
        monkeypatch.setattr(fastpath, "_FINGERPRINTS", ledgers)
        return ledgers
    return install


def test_a_card_every_family_was_re_run_on_dispatches_with_no_opt_in(
        monkeypatch, certified_on_9_0):
    """The round that certifies 9.0 writes records; nothing else has to change."""
    certified_on_9_0()
    assert fastpath.validated_compute_capabilities() == ("8.6", "9.0")
    stub_triton(monkeypatch)
    install_composer(monkeypatch, step_plan({"step_B": CountingPlan("b")},
                                            {"step_B": "PML"}))
    grid = cupy_like_grid_with_device(capability=(9, 0), name=b"NVIDIA H100 PCIe")
    plan = fastpath.plan_fast_path(object(), object(), grid)
    assert plan is not None, fastpath.last_dispatch_report().get("refused_because")
    environment = plan.report()["environment"]
    assert environment["device"]["compute_capability"] == "9.0"
    assert environment["device_certified"] is True
    assert environment["device_certified_by_table"]["triton"] is True
    assert plan.report()["uncertified"]["admitted"] == []
    assert plan.report()["certified"] is True
    quoted = plan.report()["families"]["PML"]
    assert quoted["capability"] == "9.0", quoted.get("run_record")


def test_the_cuda_ledger_has_one_reader():
    """Admission and the CUDA certification quote read the SAME object.

    Two caches of one file agree only until either is replaced, and the two answers
    they feed have to agree: admission derives the architectures this table runs on,
    and :func:`fastpath_cuda.certification_for` quotes the run for the architecture a
    plan is running on.
    """
    assert fastpath_cuda.fingerprints() is fastpath._fingerprints(fastpath.CUDA_TABLE)


def test_a_cuda_arm_admitted_on_a_new_architecture_quotes_that_architectures_run(
        certified_on_9_0):
    """The CUDA half of the plan's quote, on the round that certified 9.0.

    The Triton family quote above was the only one this file checked, and the CUDA
    quote read its own cache of the ledger: under a round that certified 9.0 it saw
    the 8.6-only file and told the plan "no live run on compute capability 9.0 ...
    admitted by the opt-in" -- a false statement in the dispatch report of a card that
    no opt-in admitted.
    """
    certified_on_9_0()
    cited = {gate for _family, gate in
             fastpath._arm_certification_map(fastpath.CUDA_TABLE).values()}
    arm = next(label for label, (_family, gate)
               in sorted(fastpath_cuda.CUDA_ARM_CERTIFICATION.items()) if gate in cited)
    quoted = fastpath_cuda.certification_for(arm, capability="9.0")
    assert "9.0" in quoted["capabilities_live"], quoted
    assert quoted.get("capability") == "9.0", quoted.get("run_record")
    assert "run_record" not in quoted, quoted["run_record"]
    # and the architecture that was already certified still quotes its own run
    assert fastpath_cuda.certification_for(arm, capability="8.6").get("capability") == "8.6"


def test_one_family_short_refuses_the_card_and_names_the_weld(
        monkeypatch, certified_on_9_0):
    """A partial round certifies nothing: the arm it did not cover could be the one
    serving a slot. Restricted to certified identities, the card is refused and the
    refusal names what would have to be re-run."""
    monkeypatch.setenv(fastpath.UNCERTIFIED_SWITCH, "0")
    skipped = "bit_identity_gate"
    certified_on_9_0(skip_triton=skipped)
    assert fastpath.validated_compute_capabilities() == ("8.6",)
    stub_triton(monkeypatch)
    install_composer(monkeypatch, step_plan({"step_B": CountingPlan("b")},
                                            {"step_B": "PML"}))
    grid = cupy_like_grid_with_device(capability=(9, 0), name=b"NVIDIA H100 PCIe")
    assert fastpath.plan_fast_path(object(), object(), grid) is None
    record = fastpath.last_dispatch_report()
    triton = record["tables"]["triton"]
    assert triton["candidate"] is False and "9.0" in triton["refused_because"]
    assert triton["refused_because"].endswith(fastpath.CERTIFIED_ONLY_TAIL)
    assert skipped in triton["welds_without_a_live_run_here"]
    # ...and the architecture that WAS fully certified still dispatches.
    assert fastpath.plan_fast_path(object(), object(),
                                   cupy_like_grid_with_device()) is not None


def test_one_family_short_drops_the_table_by_default_when_the_other_is_certified(
        monkeypatch, certified_on_9_0):
    """By default the partially re-run table is SUPPORTED on 9.0, but the hand-CUDA
    table is certified there, so rung 4e drops the Triton table by name, with the
    weld that would have to be re-run, and the hand-CUDA table runs alone, certified."""
    skipped = "bit_identity_gate"
    certified_on_9_0(skip_triton=skipped)
    stub_triton(monkeypatch)
    monkeypatch.delenv(fastpath.FUSE_ARMS_SWITCH, raising=False)
    install_composer(monkeypatch, step_plan({"step_B": CountingPlan("b")},
                                            {"step_B": "PML"}))
    install_cuda_composer(monkeypatch, cuda_step_plan())
    grid = certified_envelope_grid()
    grid.xp.cuda = cupy_like_grid_with_device(capability=(9, 0),
                                              name=b"NVIDIA H100 PCIe").xp.cuda
    plan = fastpath.plan_fast_path(certified_envelope_fields(), certified_envelope_pml(),
                                   grid)
    assert plan is not None, fastpath.last_dispatch_report()["refused_because"]
    record = plan.report()
    assert record["decision"] == "dispatched" and record["step_path"] == "fused"
    assert record["composition"]["tables_dispatched"] == [fastpath.CUDA_TABLE]
    assert sorted(plan.slots) == ["step_D", "update_E"]
    assert record["certified"] is True
    triton = record["tables"]["triton"]
    assert triton["candidate"] is False
    assert triton["dropped_for_a_certified_table"] == [fastpath.CUDA_TABLE]
    assert skipped in triton["welds_without_a_live_run_here"]
    assert record["uncertified"]["admitted"] == []
    (dropped,) = record["uncertified"]["dropped"]
    assert (dropped["table"], dropped["read"]) == ("triton", "9.0")


def test_a_supported_card_no_family_ran_on_is_refused_only_when_restricted(
        monkeypatch, certified_on_9_0):
    certified_on_9_0()
    stub_triton(monkeypatch)
    install_composer(monkeypatch, step_plan({"step_B": CountingPlan("b")},
                                            {"step_B": "PML"}))
    grid = cupy_like_grid_with_device(capability=(8, 9), name=b"NVIDIA L40S")
    monkeypatch.setenv(fastpath.UNCERTIFIED_SWITCH, "0")
    assert fastpath.plan_fast_path(object(), object(), grid) is None
    assert "8.9" in fastpath.last_dispatch_report()["refused_because"]
    # By default it runs, uncertified, and names the welds no run covers.
    monkeypatch.delenv(fastpath.UNCERTIFIED_SWITCH)
    plan = fastpath.plan_fast_path(object(), object(), grid)
    assert plan is not None, fastpath.last_dispatch_report()["refused_because"]
    assert plan.report()["certified"] is False
    served = plan.report()["uncertified"]["served"]
    assert served and all(entry["welds_without_a_live_run_here"] for entry in served)


def test_an_unsupported_card_no_family_ran_on_is_still_refused(monkeypatch,
                                                              certified_on_9_0):
    certified_on_9_0()
    stub_triton(monkeypatch)
    install_composer(monkeypatch, step_plan({"step_B": CountingPlan("b")},
                                            {"step_B": "PML"}))
    grid = cupy_like_grid_with_device(capability=(10, 0), name=b"another card")
    assert fastpath.plan_fast_path(object(), object(), grid) is None
    reason = fastpath.last_dispatch_report()["refused_because"]
    assert "10.0" in reason and reason.endswith(fastpath.UNSUPPORTED_HINT), reason


# ---------------------------------------------------------------------------
# The campaign a record describes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("capability,expected", [("8.6", "_cc86"), ("9.0", "_cc90"),
                                                 ("12.0", "_cc120")])
def test_a_route_campaign_names_the_architecture_it_ran_on(capability, expected):
    """One stamp per release, one campaign per architecture under it, so a reader can
    see which card a route record describes from the directory name."""
    assert fastpath.route_campaign("dispatch_fused_route_2026-09-30_091", capability) \
        == f"dispatch_fused_route_2026-09-30_091{expected}"
    with pytest.raises(fastpath.CapabilityRecordError):
        fastpath.route_campaign("s", "86")


def test_the_licence_belongs_to_the_table_that_composes_first():
    assert fastpath.primary_table("8.6") == fastpath.NVIDIA_TABLE_PRECEDENCE[0]
    assert fastpath.primary_table("9.0") is None


def test_an_expansion_probe_from_another_architecture_licenses_nothing_here():
    """Which arm a platform's compiler takes is a per-architecture measurement, and
    the table it keys on names the backend and the machine but not the card."""
    probe = {"environment": {"backend": "cupy", "machine": "x86_64",
                             "compute_capability": "8.6"}}
    assert fastpath._probe_capability_reasons(probe, "8.6") == []
    assert fastpath._probe_capability_reasons(probe, None) == []
    assert fastpath._probe_capability_reasons({"environment": {}}, "9.0") == []
    dropped = fastpath._probe_capability_reasons(probe, "9.0")
    assert len(dropped) == 1 and "licenses nothing here" in dropped[0]
    assert fastpath._probe_capability_reasons(
        {"environment": {"cc": "86"}}, "8.6") == []


def test_the_migration_is_recorded_in_both_ledgers_and_states_no_digest():
    """A record of what moved, so a reader is not left to infer it from the shape."""
    import re  # noqa: PLC0415

    for table in ("triton", fastpath.CUDA_TABLE):
        note = fastpath._fingerprints(table).get("_capability_runs_migration")
        assert isinstance(note, dict), table
        assert note["capabilities"] == ["8.6"] and note["entries"] > 0
        assert note["capability_was_read_from"].startswith("the ")
        assert not re.findall(r"\b[0-9a-f]{64}\b", json.dumps(note))


def _printed_capabilities(row: str):
    """Every compute capability a README row's capability cell names.

    Reads the SECOND cell only, so a toolchain version in the next cell ("CuPy
    13.5.1") cannot be read as an architecture. Inside it, a capability is the
    ``N.N`` shape this ledger keys a record by, standing alone: not preceded or
    followed by a digit or a dot, which is what keeps Triton's "3.1.0" in the same
    cell from reading as "3.1". Any wording that lists them is read, so a row that
    says "8.6, 9.0" or "compute capabilities 8.6 and 9.0" states both -- the first
    form of this parser read only the number directly after the phrase, which
    turned the first round on a second architecture into a README no wording could
    satisfy.
    """
    import re  # noqa: PLC0415

    cells = row.split("|")
    assert len(cells) >= 4, f"not a three-column table row: {row}"
    return set(re.findall(r"(?<![\d.])([1-9][0-9]*\.[0-9])(?![\d.])", cells[2]))


def _readme_mismatches(readme: str):
    """``(table, printed, derived)`` for each NVIDIA README row that disagrees.

    Each row is keyed to its table by what the row SAYS, never by its position. A
    pairing by sort order put the Triton row against the CUDA table, which passed only
    while both tables admitted the same architectures -- and the first round that
    certifies one table and not the other is exactly the round this weld exists for.
    """
    rows = [line for line in readme.splitlines()
            if line.startswith("| NVIDIA,") and "compute capabilit" in line]
    assert len(rows) == 2, rows
    by_table = {}
    for row in rows:
        named = [table for table, label in (("triton", "Triton kernels"),
                                            (fastpath.CUDA_TABLE, "CUDA kernels"))
                 if label in row]
        assert len(named) == 1, f"this README row names no single table: {row}"
        assert named[0] not in by_table, f"two README rows name {named[0]}: {rows}"
        by_table[named[0]] = row
    assert set(by_table) == {"triton", fastpath.CUDA_TABLE}, sorted(by_table)
    wrong = []
    for table, row in sorted(by_table.items()):
        printed = _printed_capabilities(row)
        derived = set(fastpath.capability_admission(table)["admitted"] or ())
        if printed != derived:
            wrong.append((table, sorted(printed), sorted(derived)))
    return wrong


def test_the_readme_states_the_architectures_the_ledgers_actually_certify():
    """A doc-code weld, because a DECLARATION drifting from its evidence is the
    whole defect the per-architecture records closed: the list the README prints has
    to be the list the ledgers derive, or a reader is told something no run measured.
    """
    import pathlib  # noqa: PLC0415

    readme = (pathlib.Path(__file__).resolve().parents[1] / "README.md").read_text(
        encoding="utf-8")
    assert _readme_mismatches(readme) == [], (
        "README prints, per table, what the ledger does not derive "
        "(table, printed, derived): " + repr(_readme_mismatches(readme)))


def test_the_readme_weld_pairs_each_row_with_its_own_table_when_the_tables_differ(
        monkeypatch):
    """The case the weld is FOR: one table certified on an architecture the other is not.

    A round on a rented card can run the Triton fleet and not the CUDA families, so
    Triton gaining an architecture CUDA lacks is the likely shape of the next round,
    not an edge case. Driven with that asymmetry, a correct README has to pass and a
    README that swaps the two rows has to FAIL -- the swap is precisely what a
    sort-order pairing silently accepted.
    """
    admitted = {"triton": ("8.6", "9.0"), fastpath.CUDA_TABLE: ("8.6",)}
    monkeypatch.setattr(fastpath, "capability_admission",
                        lambda table="triton": {"admitted": admitted[table],
                                                "by_key": {}})
    triton = "| NVIDIA, Triton kernels | compute capability 8.6, 9.0 with Triton 3.1.0 | x |"
    cuda = "| NVIDIA, hand-written CUDA kernels | compute capability 8.6 | x |"

    assert _readme_mismatches(f"{triton}\n{cuda}\n") == []
    assert _readme_mismatches(f"{cuda}\n{triton}\n") == [], "row ORDER must not matter"

    swapped = ("| NVIDIA, Triton kernels | compute capability 8.6 with Triton 3.1.0 | x |"
               "\n| NVIDIA, hand-written CUDA kernels | compute capability 8.6, 9.0 | x |")
    wrong = _readme_mismatches(swapped)
    assert sorted(table for table, _printed, _derived in wrong) == sorted(
        ["triton", fastpath.CUDA_TABLE]), wrong


@pytest.mark.parametrize("cell, expected", [
    ("compute capability 8.6 with Triton 3.1.0", {"8.6"}),
    ("compute capability 8.6, 9.0 with Triton 3.1.0", {"8.6", "9.0"}),
    ("compute capabilities 8.6 and 9.0", {"8.6", "9.0"}),
    ("compute capabilities 8.6, 9.0 and 12.0", {"8.6", "9.0", "12.0"}),
    ("compute capability 9.0", {"9.0"}),
])
def test_the_readme_parser_reads_every_architecture_a_row_lists(cell, expected):
    """Every wording a person would use to list architectures is read in full.

    And a toolchain version is never read as one: Triton's "3.1.0" sits in the same
    cell and must not come back as "3.1", and the CuPy version in the next cell is
    not scanned at all.
    """
    row = f"| NVIDIA, Triton kernels | {cell} | one card, CuPy 13.5.1 |"
    assert _printed_capabilities(row) == expected, (cell, _printed_capabilities(row))
