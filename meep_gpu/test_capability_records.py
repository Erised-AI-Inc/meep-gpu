"""One certification record per GPU architecture: is a second one ADDITIVE?

What this file is for. Each weld in the NVIDIA ledgers keeps one record per compute
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
                                     assert_the_host_is_a_real_card,
                                     certified_envelope_fields, certified_envelope_grid,
                                     certified_envelope_pml, cuda_step_plan,
                                     cupy_like_grid_with_device, failing_triton_probe,
                                     install_composer, install_cuda_composer, step_plan,
                                     stub_triton)


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
    monkeypatch.delenv(fastpath.HARDWARE_SWITCH, raising=False)
    monkeypatch.delenv(fastpath.RETIRED_HARDWARE_SWITCH, raising=False)
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
    """The whole point. Certifying 9.0 must not un-certify 8.6."""
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
    by no tool, so a certification on another architecture left it saying 8.6 regardless."""
    # Retires 8.6 alone: the RTX 4080 SUPER's runs bound into these ledgers certify 8.9 too.
    assert fastpath.validated_compute_capabilities() == ("8.6", "8.9")
    assert fastpath_cuda.validated_compute_capabilities() == ("8.6", "8.9")
    for table in ("triton", fastpath.CUDA_TABLE):
        report = fastpath.capability_admission(table)
        blocking = {key: state["problem"] for key, state in report["by_key"].items()
                    if state["problem"]}
        assert not blocking, blocking
        # Retires 8.6 alone: each NVIDIA table admits 8.9 from its own 8.9 runs.
        assert report["admitted"] == ("8.6", "8.9")


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
    # Retires 8.6 alone: the RTX 4080 SUPER re-ran the nine families, so 8.9 is live here too.
    assert fastpath.live_capabilities(entry) == ("8.6", "8.9")
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
    # Retires ["8.6"]: 8.9 is live on this entry too; the quoted run stays 8.6's.
    assert entry["capability"] == "8.6" and entry["capabilities_live"] == ["8.6", "8.9"]
    assert entry["recorded_utc"]
    run = fastpath._fingerprints("triton")[fastpath.FAMILY_RECERT_GATE][fastpath.RUNS]["8.6"]
    assert_the_host_is_a_real_card(entry["host"], run, "8.6")
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
    ``tools/ci/declared_resources.txt``: it reads the certification run's own artifacts, which a
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


def record_certified_stack(ledger: dict, capability: str, **values: str) -> dict:
    """``ledger``, a copy of the Triton ledger, with ``capability``'s certified NVIDIA stack
    recorded as the bind writes it (``record_nvidia_stack.py``): the stand-in host's 8.6
    stack, with ``values`` read in its place, axis by axis."""
    stack = {axis: list(listed)
             for axis, listed in fastpath.certified_nvidia_stack("8.6").items()}
    stack.update({axis: [value] for axis, value in values.items()})
    entry = copy.deepcopy(ledger[fastpath.CERTIFIED_STACK_ENTRY])
    entry[fastpath.STACK_BY_CAPABILITY][capability] = {
        "read_from": [f"parity/meep_gpu/results/chunk_cc{capability.replace('.', '')}"
                      f"_c0/stack/gates_0.json"],
        "recorded_by": "parity/meep_gpu/record_nvidia_stack.py",
        "recorded_utc": "2026-09-30T00:00:00Z", "stack": stack}
    ledger[fastpath.CERTIFIED_STACK_ENTRY] = entry
    return ledger


def ledger_also_certified_on(table: str, capability: str, skip: str = None,
                             stack: bool = True) -> dict:
    """A copy of a shipped ledger with ``capability`` certified on the same bytes.

    The Triton ledger also gets ``capability``'s certified NVIDIA stack (``stack``), which
    the bind of a capability writes there whichever table it binds."""
    ledger = copy.deepcopy(dict(fastpath._fingerprints(table)))
    cited = {gate for _family, gate in fastpath._arm_certification_map(table).values()}
    for key in sorted(cited):
        if key == skip:
            continue
        fastpath.bind_capability(
            ledger[key], bound_before=fastpath.bound_digest(ledger[key]),
            capability=capability, run=run_on(capability))
    if stack and table == "triton":
        record_certified_stack(ledger, capability)
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
    """The certification run that certifies 9.0 writes records; nothing else has to change."""
    certified_on_9_0()
    # Retires ("8.6", "9.0"): 9.0 joins the shipped 8.6 and 8.9, which these ledgers certify.
    assert fastpath.validated_compute_capabilities() == ("8.6", "8.9", "9.0")
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
    """The CUDA half of the plan's quote, after a certification run on 9.0.

    The Triton family quote above was the only one this file checked, and the CUDA
    quote read its own cache of the ledger: after a certification run on 9.0 it saw
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
    """A partial certification run certifies nothing: the arm it did not cover could be the one
    serving a slot. Restricted to certified identities, the card is refused and the
    refusal names what would have to be re-run."""
    monkeypatch.setenv(fastpath.HARDWARE_SWITCH, fastpath.HARDWARE_CERTIFIED)
    skipped = "bit_identity_gate"
    certified_on_9_0(skip_triton=skipped)
    # Retires 8.6 alone: the partial 9.0 run adds nothing to the shipped 8.6 and 8.9.
    assert fastpath.validated_compute_capabilities() == ("8.6", "8.9")
    stub_triton(monkeypatch)
    install_composer(monkeypatch, step_plan({"step_B": CountingPlan("b")},
                                            {"step_B": "PML"}))
    grid = cupy_like_grid_with_device(capability=(9, 0), name=b"NVIDIA H100 PCIe")
    assert fastpath.plan_fast_path(object(), object(), grid) is None
    record = fastpath.last_dispatch_report()
    triton = record["tables"]["triton"]
    assert triton["candidate"] is False and "9.0" in triton["refused_because"]
    assert triton["refused_because"].endswith(fastpath.CERTIFIED_SCOPE_TAIL)
    assert skipped in triton["welds_without_a_live_run_here"]
    # ...and the architecture that WAS fully certified still dispatches.
    assert fastpath.plan_fast_path(object(), object(),
                                   cupy_like_grid_with_device()) is not None


def test_one_family_short_composes_beside_the_certified_table_by_default(
        monkeypatch, certified_on_9_0):
    """By default the partially re-run table is SUPPORTED on 9.0 and the hand-CUDA
    table is certified there; both compose (there is no rung 4e), the run is
    uncertified on the Triton table, and its entry names the weld that would have to be
    re-run."""
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
    assert record["composition"]["tables_dispatched"] == [fastpath.CUDA_TABLE, "triton"]
    assert sorted(plan.slots) == ["step_B", "step_D", "update_E"]
    assert record["certified"] is False
    assert record["tables"]["triton"] == {"candidate": True, "refused_because": None}
    assert record["uncertified"]["dropped"] == []
    (served,) = record["uncertified"]["served"]
    assert (served["table"], served["read"]) == ("triton", "9.0")
    assert skipped in served["welds_without_a_live_run_here"]


def test_a_supported_card_no_family_ran_on_is_refused_only_when_restricted(
        monkeypatch, certified_on_9_0):
    certified_on_9_0()
    stub_triton(monkeypatch)
    install_composer(monkeypatch, step_plan({"step_B": CountingPlan("b")},
                                            {"step_B": "PML"}))
    # 8.7 retires the L40S's 8.9 as the uncertified example: 8.9 is certified now.
    grid = cupy_like_grid_with_device(capability=(8, 7), name=b"NVIDIA Jetson AGX Orin")
    monkeypatch.setenv(fastpath.HARDWARE_SWITCH, fastpath.HARDWARE_CERTIFIED)
    assert fastpath.plan_fast_path(object(), object(), grid) is None
    assert "8.7" in fastpath.last_dispatch_report()["refused_because"]
    # By default it runs, uncertified, and names the welds no run covers.
    monkeypatch.delenv(fastpath.HARDWARE_SWITCH)
    plan = fastpath.plan_fast_path(object(), object(), grid)
    assert plan is not None, fastpath.last_dispatch_report()["refused_because"]
    assert plan.report()["certified"] is False
    served = plan.report()["uncertified"]["served"]
    assert served and all(entry["welds_without_a_live_run_here"] for entry in served)


def test_an_unsupported_card_no_family_ran_on_is_still_refused(monkeypatch,
                                                              certified_on_9_0):
    """A card the installed NVRTC cannot target and the installed Triton cannot launch
    on: the hand-CUDA table refused by name, the Triton table dropped by its probe."""
    certified_on_9_0()
    stub_triton(monkeypatch, probe=failing_triton_probe([(10, 0)]))
    install_composer(monkeypatch, step_plan({"step_B": CountingPlan("b")},
                                            {"step_B": "PML"}))
    grid = cupy_like_grid_with_device(capability=(10, 0), name=b"another card")
    assert fastpath.plan_fast_path(object(), object(), grid) is None
    record = fastpath.last_dispatch_report()
    reason = record["refused_because"]
    assert "10.0" in reason, reason
    cuda = record["tables"][fastpath.CUDA_TABLE]["refused_because"]
    assert "10.0" in cuda and cuda.endswith(fastpath.UNSUPPORTED_HINT), cuda
    assert "cannot compile and launch" in record["tables"]["triton"]["refused_because"]


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
    turned the first certification on a second architecture into a README no wording could
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
    while both tables admitted the same architectures -- and the first certification that
    covers one table and not the other is exactly the case this weld exists for.
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

    A certification on a rented card can run the Triton fleet and not the CUDA families, so
    Triton gaining an architecture CUDA lacks is the likely shape of the next one,
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


# ---------------------------------------------------------------------------
# What the documentation states is certified, read against the ledgers
# ---------------------------------------------------------------------------
#
# The README's certified-identity rows have their weld above. The tests below hold the
# other places the same facts are stated, each of which the certification of another
# compute capability moves. Every expected value is derived from the ledgers or from the
# shipped timing records, never typed: a later certification needs no edit here, and a
# sentence it leaves behind fails by name.

REPOSITORY = pathlib.Path(__file__).resolve().parents[1]

#: A compute capability as the ledgers key one, standing alone in prose.
CAPABILITY_IN_TEXT = r"(?<![\d.])([1-9][0-9]*\.[0-9])(?![\d.])"

NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                "seven": 7, "eight": 8, "nine": 9, "ten": 10}

#: The sentences that count the certified compute capabilities.
COUNT_STATEMENTS = (
    r"\b(\w+) (?:NVIDIA )?compute capabilit(?:y|ies) and \w+ Apple environments?\b",
    r"\b(\w+) compute capabilit(?:y|ies) certified\b",
    r"\bcertified on (\w+) identit(?:y|ies) per table\b",
    # The kernel-dispatch guide's bullet read "one identity certified" before 8.9.
    r"\b(\w+) identit(?:y|ies) certified\b",
)


def _admitted(table: str) -> set:
    return set(fastpath.capability_admission(table)["admitted"] or ())


def _certified_capabilities() -> set:
    """Every compute capability an NVIDIA table admits."""
    admitted = set()
    for table in fastpath.NVIDIA_TABLE_PRECEDENCE:
        admitted |= _admitted(table)
    return admitted


def _flat(text: str) -> str:
    """``text`` with every run of whitespace one space, so a wrapped sentence reads whole."""
    return " ".join(text.split())


def _documentation() -> dict:
    """The pages a reader consults for what this release certifies, flattened: README.md,
    INSTALL.md, AGENTS.md and the manual under docs/. Not CHANGELOG.md, whose entries
    state what each change did when it was made."""
    pages = [REPOSITORY / name for name in ("README.md", "INSTALL.md", "AGENTS.md")]
    pages += sorted((REPOSITORY / "docs").rglob("*.md"))
    return {page.relative_to(REPOSITORY).as_posix(): _flat(page.read_text(encoding="utf-8"))
            for page in pages}


def test_install_states_the_architectures_the_ledgers_actually_certify():
    """INSTALL.md's certified-identity table states the README's rows again: each NVIDIA row
    names exactly the compute capabilities its table's welds admit."""
    install = (REPOSITORY / "INSTALL.md").read_text(encoding="utf-8")
    wrong = []
    for table, label in (("triton", "Triton (NVIDIA)"),
                         (fastpath.CUDA_TABLE, "hand-written CUDA (NVIDIA)")):
        rows = [line for line in install.splitlines() if line.startswith(f"| {label} |")]
        assert len(rows) == 1, f"INSTALL.md holds {len(rows)} rows for {label}: {rows}"
        printed = _printed_capabilities(rows[0])
        if printed != _admitted(table):
            wrong.append((table, sorted(printed), sorted(_admitted(table))))
    assert wrong == [], ("INSTALL.md prints, per table, what the ledger does not derive "
                         f"(table, printed, derived): {wrong}")


def test_every_count_of_certified_compute_capabilities_is_the_ledgers():
    """A sentence that counts the certified compute capabilities counts what the welds admit.

    A count can sit under a table that names one capability more and still read the old
    number. The count must be read in README.md and in INSTALL.md, so a rewording this no
    longer reads fails here instead of escaping it.
    """
    import re  # noqa: PLC0415

    expected = len(_certified_capabilities())
    found, wrong = set(), []
    for name, text in _documentation().items():
        for pattern in COUNT_STATEMENTS:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                word = match.group(1).lower()
                count = int(word) if word.isdigit() else NUMBER_WORDS.get(word)
                found.add(name)
                if count != expected:
                    wrong.append((name, match.group(0)))
    assert {"README.md", "INSTALL.md"} <= found, (
        "no count of the certified compute capabilities was read in README.md and "
        f"INSTALL.md (read in {sorted(found)}); a reworded count must still be read here")
    assert wrong == [], (
        f"the welds admit {expected} compute capabilities and these sentences count "
        f"otherwise: {wrong}")


def test_the_new_gpu_guide_shows_the_admission_the_ledgers_derive():
    """The guide shows what the admission check prints on this release, and what it prints
    after the bind of a new capability: the first is the welds' admission, the second that
    admission with exactly one capability added."""
    import ast  # noqa: PLC0415
    import re  # noqa: PLC0415

    guide = _flat((REPOSITORY / "docs" / "development" / "certify-a-new-gpu.md").read_text(
        encoding="utf-8"))
    derived = {table: _admitted(table) for table in ("triton", fastpath.CUDA_TABLE)}
    shown = re.findall(r"\bOn [^`]{1,30}? it prints `(\{[^`]*\})`", guide)
    assert len(shown) == 1, f"the guide shows {len(shown)} outputs of the admission check: {shown}"
    printed = {table: set(listed) for table, listed in ast.literal_eval(shown[0]).items()}
    assert printed == derived, (
        f"the guide prints {shown[0]} for this release; the welds admit {derived}")
    after = re.findall(r"both tables admitting your capability beside ([^(]*)\(`(\{[^`]*\})`\)",
                       guide)
    assert after, "the guide no longer shows what the admission check prints after a bind"
    for beside, example in after:
        assert set(re.findall(CAPABILITY_IN_TEXT, beside)) == set().union(*derived.values()), (
            f"the guide names {beside!r} as the capabilities already admitted; the welds "
            f"admit {derived}")
        for table, listed in ast.literal_eval(example).items():
            added = set(listed) - derived[table]
            assert derived[table] <= set(listed) and len(added) == 1, (
                f"after a bind the guide shows {table} admitting {listed}, which is not "
                f"this release's {sorted(derived[table])} with one capability added")


def test_every_certified_capability_has_its_certified_stack_recorded():
    """A capability the welds admit has the stack its own runs read recorded beside it.
    Admitted without one, ``MEEP_GPU_HARDWARE=certified`` refuses it by name while every
    page states it certified."""
    missing = sorted(capability for capability in _certified_capabilities()
                     if fastpath.certified_nvidia_stack(capability) is None)
    assert missing == [], (
        f"compute capabilities {missing} are admitted with no certified stack recorded "
        f"({fastpath.CERTIFIED_STACK_ENTRY} in the Triton ledger)")


def test_the_readme_names_the_cuda_drivers_each_capability_was_certified_on():
    """The README's certified-on cell has a part for each compute capability the Triton table
    admits, and each part names the CUDA drivers that capability's certified stack lists
    (``13.4`` is driver ``13040``) and no other."""
    import re  # noqa: PLC0415

    readme = (REPOSITORY / "README.md").read_text(encoding="utf-8")
    rows = [line for line in readme.splitlines()
            if line.startswith("| NVIDIA, Triton kernels |")]
    assert len(rows) == 1, rows
    printed = {}
    for part in rows[0].split("|")[3].split(";"):
        head = re.match(r"\s*([1-9][0-9]*\.[0-9]): (.*)", part)
        if head:
            printed[head.group(1)] = {str(int(major) * 1000 + int(minor) * 10)
                                      for major, minor in re.findall(r"driver (\d+)\.(\d+)",
                                                                     head.group(2))}
    assert set(printed) == _admitted("triton"), (
        f"the certified-on cell has a part for {sorted(printed)}; the Triton table admits "
        f"{sorted(_admitted('triton'))}")
    for capability, drivers in sorted(printed.items()):
        recorded = set((fastpath.certified_nvidia_stack(capability) or {}).get("cuda_driver", ()))
        assert drivers == recorded, (
            f"the README names CUDA drivers {sorted(drivers)} for {capability}; its certified "
            f"stack lists {sorted(recorded)}")


def _timed_nvidia_capabilities() -> list:
    """The compute capabilities the shipped NVIDIA timing records time on this tree."""
    from meep_gpu import dispatch_preference as dp  # noqa: PLC0415

    root = pathlib.Path(dp.__file__).resolve().parent
    return sorted({capability for name in dp.timed_vendors()["nvidia"]["records"]
                   for capability in dp.record_capabilities(dp.load_record(str(root / name)))})


def test_a_certified_capability_no_record_times_is_stated_as_reading_the_nearest_one():
    """Certified is not timed. A certified capability that no shipped record times reads the
    record of the nearest timed capability (``dispatch_preference.nearest_architecture``).
    The README says so for each such capability, and every page that says it names the
    capability the rule reads; a record cut for that capability later makes the sentence
    false, and this fails on it by name."""
    import re  # noqa: PLC0415

    from meep_gpu import dispatch_preference as dp  # noqa: PLC0415

    timed = _timed_nvidia_capabilities()
    assert timed, "no shipped NVIDIA timing record times its capability on this tree"
    reads = {capability: dp.nearest_architecture("nvidia", capability, timed)[0]
             for capability in _certified_capabilities() if capability not in timed}
    stated, wrong = {}, []
    for name, text in _documentation().items():
        for capability, read in re.findall(
                r"[Cc]ompute capability ([1-9][0-9]*\.[0-9]) has no record of its own: its "
                r"default reads ([1-9][0-9]*\.[0-9])'s", text):
            stated.setdefault(capability, set()).add(name)
            if reads.get(capability) != read:
                wrong.append((name, capability, read,
                              reads.get(capability, "a record of its own")))
    assert wrong == [], (
        "these sentences name the record a capability reads wrongly (page, capability, "
        f"stated, what the rule reads): {wrong}")
    unstated = sorted(capability for capability in reads
                      if "README.md" not in stated.get(capability, ()))
    assert unstated == [], (
        f"compute capabilities {unstated} are certified with no timing record of their own, "
        "and the README does not say whose record their default reads")


def test_the_notes_the_docs_quote_name_the_certified_capabilities():
    """The docs quote the NOTE a supported, uncertified NVIDIA device prints. The capability it
    reads must be one no table certifies, and the list after ``certified:`` a table's
    admission, which is what the NOTE prints; certifying the capability an example quotes
    makes the example false."""
    import re  # noqa: PLC0415

    quoted = [(name, read, listed) for name, text in _documentation().items()
              for read, listed in re.findall(
                  r"NOT CERTIFIED on this host: GPU compute capability ([1-9][0-9]*\.[0-9]) "
                  r"\(certified: ([^)]*)\)", text)]
    assert any(name == "README.md" for name, _read, _listed in quoted), (
        f"the README no longer quotes the NOTE (quoted in {sorted({q[0] for q in quoted})})")
    tables = [_admitted(table) for table in fastpath.NVIDIA_TABLE_PRECEDENCE]
    wrong = [(name, read, listed) for name, read, listed in quoted
             if read in _certified_capabilities()
             or set(re.findall(CAPABILITY_IN_TEXT, listed)) not in tables]
    assert wrong == [], (
        "these quoted NOTEs name a certified capability as uncertified, or a certified list "
        f"no table admits (page, read, certified list); admitted {tables}: {wrong}")


ROUTE_TABLES = ("triton", fastpath.CUDA_TABLE, "metal")


def _route_record(table: str) -> dict:
    """A table's ``driver_dispatch`` record, as the package ships it."""
    return json.loads((pathlib.Path(fastpath.__file__).resolve().parent / f"{table}_kernels"
                       / "fingerprints.json").read_text(encoding="utf-8"))["driver_dispatch"]


def _route_runs() -> list:
    """``(table, key)`` for every live route run of the three route records."""
    return [(table, key) for table in ROUTE_TABLES
            for key in fastpath.live_capabilities(_route_record(table))]


def pytest_generate_tests(metafunc):
    """The live route runs, read when the tests are collected and not when this module is
    imported: other test modules import its helpers in processes that must not open a
    ledger they do not use (``test_ledger_scope``)."""
    if "route_run" in metafunc.fixturenames:
        runs = _route_runs()
        metafunc.parametrize("route_run", runs, ids=[f"{table}-{key}" for table, key in runs])


@pytest.mark.parametrize("table", ROUTE_TABLES)
def test_every_route_run_states_its_own_default_and_no_record_shares_one(table):
    """The statement of the default (the timing records a route run's legs ran with, what
    the default read and vetoed, the carry log) is read off one run's legs, so each run
    keeps its own under ``runs[<key>].dispatch_preference``, and its ``run`` names that
    run's own campaign. A statement beside the digests was one copy every cut rewrote: the
    compute capability 8.9 cut replaced 8.6's, and an Apple GPU architecture's cut replaced
    another's vetoes with its own. No record may carry one there again."""
    record = _route_record(table)
    assert "dispatch_preference" not in record, (
        f"the {table} route record carries a dispatch_preference beside its digests; each "
        "run's statement belongs in runs[<key>].dispatch_preference")
    assert fastpath.live_capabilities(record), f"the {table} route record has no live run"
    for key, run in sorted(record[fastpath.RUNS].items()):
        statement = run.get("dispatch_preference")
        assert isinstance(statement, dict) and isinstance(
            statement.get("shipped_records"), dict), (
            f"{table} runs[{key!r}] states no default of its own")
        assert statement.get("run") == run["released_fused_arms"]["gate"], (
            f"{table} runs[{key!r}] holds the statement of {statement.get('run')!r}, not of "
            f"its own campaign {run['released_fused_arms']['gate']!r}")


def test_every_route_run_pins_every_timing_record_the_package_ships(route_run):
    """A route run states the default on the timing records the tree shipped when it was
    cut, and pins each by the sha256 of its bytes
    (``runs[<key>].dispatch_preference.shipped_records``): the cut refuses a leg that ran on
    another set unless a carry takes it. The weld walk checks the bytes of each pin and the
    weld contracts count the pins against a floor; this holds the set per run, so a record
    added or removed after an architecture's cut leaves that architecture's run owing a
    re-cut or a carry, which another architecture's cut cannot pay."""
    from meep_gpu import dispatch_preference as dp  # noqa: PLC0415

    table, key = route_run
    root = pathlib.Path(dp.__file__).resolve().parent
    shipped = {f"meep_gpu/{pathlib.Path(path).relative_to(root).as_posix()}"
               for path in dp.record_files()}
    run = _route_record(table)[fastpath.RUNS][key]
    pinned = set(run["dispatch_preference"]["shipped_records"])
    assert pinned == shipped, (
        f"the {table} route run runs[{key!r}] pins {sorted(pinned)}; the package ships "
        f"{sorted(shipped)}")


def test_the_dispatch_guide_names_the_go_no_go_of_each_licensed_capability():
    """The kernel-dispatch guide's certification note says the record behind dispatch on by
    default was re-cut from a go/no-go. The Triton record carries one licence per compute
    capability whose primary table it is (``runs[<capability>].dispatch_by_default_licence``),
    each naming the card it ran on. Here each licence must be present, every verdict a licensing
    one and at least one case dispatched; the transcription's own rule (at least
    ``LICENCE_MIN_DISPATCHED`` dispatched, a clean null control over the same cases) is held
    field by field in test_dispatch_contract.py. The note names each licence's card, reads the
    go/no-go per capability when there is more than one, and reads each as GO."""
    import re  # noqa: PLC0415

    record = json.loads((REPOSITORY / "meep_gpu" / "triton_kernels" / "fingerprints.json")
                        .read_text(encoding="utf-8"))["driver_dispatch"]
    licensed = sorted(capability for capability in _certified_capabilities()
                      if fastpath.primary_table(capability) == "triton")
    assert licensed, "no admitted compute capability has the Triton table as its primary table"
    cards = {}
    for capability in licensed:
        licence = ((record.get(fastpath.RUNS) or {}).get(capability) or {}).get(
            "dispatch_by_default_licence")
        assert licence, f"the Triton record carries no go/no-go licence for {capability}"
        verdicts = set(licence["verdicts"].values())
        assert verdicts <= {"PASS-DISPATCHED", "PASS-FELL-BACK"} and (
            "PASS-DISPATCHED" in verdicts), (capability, licence["verdicts"])
        cards[capability] = " ".join(word for word in licence["device"]["name"].split()
                                     if word not in ("NVIDIA", "GeForce"))
    guide = (REPOSITORY / "docs" / "guides" / "kernel-dispatch.md").read_text(encoding="utf-8")
    start = guide.index("> **Note: Certification status of this release**")
    note = _flat(" ".join(line.lstrip(">") for line in guide[start:].split("\n\n## ")[0]
                          .splitlines()))
    unnamed = sorted((capability, card) for capability, card in cards.items()
                     if f"on one {card}" not in note)
    assert unnamed == [], (
        f"the certification note names no card for these licences (capability, card): {unnamed}")
    read = re.findall(r"from (its|each compute capability's) go/no-go, (which read GO|each GO)\b",
                      note)
    expected = (("each compute capability's", "each GO") if len(licensed) > 1
                else ("its", "which read GO"))
    assert read == [expected], (
        f"the note reads the go/no-go as {read}; the Triton record carries {len(licensed)} "
        f"licences ({licensed}), each a GO, so it reads {expected}")
