"""The harness half of the NVIDIA flush licence: who writes the evidence, who reads it.

``fastpath.policy_evidence_reasons`` admits a caller-installed ``flush`` on an NVIDIA
table only on ``runs[cc].certified_policies`` (``fastpath.CERTIFIED_POLICIES``). The
field has exactly four writers and two readers, all pinned here:

* ``rebind_cuda_welds.certified_policies`` -- every CUDA weld's legs come in keep and
  flush pairs, so the field is read off each released leg's own install stamp;
* ``rebind_triton_welds.attained_policy`` and ``second_policy_evidence`` -- the Triton
  fleet cuts one policy per campaign, so a flush campaign of the SAME gates on the
  SAME bytes and capability adds ``flush`` beside the bound run, and anything else
  leaves the weld bound under its first policy only, with the reason printed;
* ``recut_composition_records.second_policy_leg`` and ``family_policy_evidence`` --
  the composition and family records 33 Triton arms cite are re-cut by their own
  tool, which records ``flush`` from a flush composition run of the same legs, and
  for the family record only when all nine families released under it;
* ``gate_dispatch_fused_route.policy_leg_role`` -- a route leg that installed an
  admitted policy is a DISPATCH leg held to every requirement, not a refusal leg;
* ``recut_driver_dispatch_record.is_dispatching_witness`` -- such a leg is asked which
  arms it drove.

Host-only: no GPU, no MEEP, no ledger is read or written.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from meep_gpu import fastpath  # noqa: E402


# ---------------------------------------------------------------------------
# rebind_cuda_welds.certified_policies
# ---------------------------------------------------------------------------


def _cuda_leg(name):
    return {"subnormal_policy_install": {"policy": name}}


def test_cuda_keep_and_flush_legs_certify_both_policies():
    import rebind_cuda_welds as R

    found = R.certified_policies(
        ["keep/gate.json", "flush/gate.json"],
        [_cuda_leg("ieee_keep_ftz_stripped"), _cuda_leg("meep_x86_flush")])
    assert found == {"flush": ["flush/gate.json"], "keep": ["keep/gate.json"]}


def test_cuda_a_leg_whose_stamp_names_no_policy_contributes_nothing():
    """An unreadable stamp must NARROW the licence, never widen it."""
    import rebind_cuda_welds as R

    found = R.certified_policies(
        ["keep/gate.json", "flush/gate.json"],
        [_cuda_leg("ieee_keep_ftz_stripped"), {"subnormal_policy_install": "flush?"}])
    assert found == {"keep": ["keep/gate.json"]}
    assert R.certified_policies(["a"], [{}]) == {}


def test_cuda_the_field_is_a_slot_field():
    import rebind_cuda_welds as R

    assert fastpath.CERTIFIED_POLICIES in R.SLOT_FIELDS


# ---------------------------------------------------------------------------
# rebind_triton_welds: the attained policy and the second-policy evidence
# ---------------------------------------------------------------------------


def _triton_artifact(policy, released=True, sources=None):
    name = {"keep": "ieee_keep_ftz_stripped", "flush": "meep_x86_flush"}[policy]
    return {"canonical_verdict": {"released": released, "reasons": []},
            "subnormal_policy": {"installed": True, "policy": name, "resolved": policy},
            "imported_source_sha256": dict(sources or {"a.py": "1" * 64})}


def test_triton_attained_policy_reads_the_resolved_stamp():
    import rebind_triton_welds as R

    assert R.attained_policy(_triton_artifact("keep")) == "keep"
    assert R.attained_policy(_triton_artifact("flush")) == "flush"
    assert R.attained_policy({}) is None


def test_triton_the_field_is_a_slot_field():
    import rebind_triton_welds as R

    assert fastpath.CERTIFIED_POLICIES in R.SLOT_FIELDS


@pytest.fixture
def flush_root(tmp_path, monkeypatch):
    """One flush campaign root holding one key's gate directory."""
    import rebind_triton_welds as R

    key = sorted(R.CAMPAIGN_DIRS)[0]
    root = tmp_path / "triton_fleet_flush"
    (root / R.CAMPAIGN_DIRS[key]).mkdir(parents=True)
    monkeypatch.setattr(R, "run_capability", lambda fresh, dirs: ("8.6", {}))

    def write(artifact):
        (root / R.CAMPAIGN_DIRS[key] / "gate.json").write_text(json.dumps(artifact))

    return R, key, root, write


def _second(R, key, root, imported, capability="8.6"):
    return R.second_policy_evidence(key, {}, [root], {root: "results/flush"}, capability,
                                    imported, ["a.py"], "flush")


def test_triton_a_released_flush_run_on_the_same_bytes_is_evidence(flush_root):
    R, key, root, write = flush_root
    write(_triton_artifact("flush"))
    evidence, why = _second(R, key, root, {"a.py": "1" * 64})
    assert why == "" and evidence is not None
    assert len(evidence["artifact_sha256"]) == 64
    assert evidence["verdict_read_from"].startswith(f"results/flush/{R.CAMPAIGN_DIRS[key]}/")


def test_triton_an_unreleased_flush_run_is_refused(flush_root):
    R, key, root, write = flush_root
    write(_triton_artifact("flush", released=False))
    evidence, why = _second(R, key, root, {"a.py": "1" * 64})
    assert evidence is None and "not released" in why


def test_triton_a_flush_campaign_that_attained_keep_is_refused(flush_root):
    R, key, root, write = flush_root
    write(_triton_artifact("keep"))
    evidence, why = _second(R, key, root, {"a.py": "1" * 64})
    assert evidence is None and "attained 'keep'" in why


def test_triton_a_flush_run_on_other_bytes_is_refused(flush_root):
    R, key, root, write = flush_root
    write(_triton_artifact("flush", sources={"a.py": "2" * 64}))
    evidence, why = _second(R, key, root, {"a.py": "1" * 64})
    assert evidence is None and "other bytes" in why


def test_triton_a_flush_run_on_another_capability_is_refused(flush_root, monkeypatch):
    R, key, root, write = flush_root
    write(_triton_artifact("flush"))
    monkeypatch.setattr(R, "run_capability", lambda fresh, dirs: ("9.0", {}))
    evidence, why = _second(R, key, root, {"a.py": "1" * 64})
    assert evidence is None and "'9.0'" in why


def test_triton_no_flush_directory_is_named(tmp_path):
    import rebind_triton_welds as R

    key = sorted(R.CAMPAIGN_DIRS)[0]
    evidence, why = R.second_policy_evidence(key, {}, [tmp_path], {tmp_path: "x"},
                                             "8.6", {}, [], "flush")
    assert evidence is None and "no flush campaign holds" in why


def test_triton_flush_arguments_belong_to_their_own_modes():
    import rebind_triton_welds as R

    with pytest.raises(SystemExit, match="--bit-identity-flush belongs"):
        R.main(["--bit-identity-flush", "x.json"])
    with pytest.raises(SystemExit, match="separate"):
        R.main(["--bind-bit-identity", "x.json", "--flush-campaign", "y"])


# ---------------------------------------------------------------------------
# gate_dispatch_fused_route.policy_leg_role
# ---------------------------------------------------------------------------


def test_route_leg_roles_follow_the_admitted_table():
    import gate_dispatch_fused_route as G

    assert G.policy_leg_role(None, "triton") == "shipped"
    assert G.policy_leg_role("keep", "triton") == "certified"
    assert G.policy_leg_role("keep", "cuda") == "certified"
    assert G.policy_leg_role("flush", "triton") == "admitted"
    assert G.policy_leg_role("flush", "cuda") == "admitted"
    assert G.policy_leg_role("flush", "metal") == "certified"
    assert G.policy_leg_role("keep", "metal") == "refusal"


# ---------------------------------------------------------------------------
# recut_driver_dispatch_record.is_dispatching_witness
# ---------------------------------------------------------------------------


def test_recut_an_admitted_flush_leg_is_a_witness():
    import recut_driver_dispatch_record as C

    assert C.is_dispatching_witness("cuda_flush", {"admitted_policy_installed": "flush"})
    assert C.is_dispatching_witness("flush", {"admitted_policy_installed": "flush"})


def test_recut_a_refusal_leg_is_not_a_witness():
    import recut_driver_dispatch_record as C

    assert not C.is_dispatching_witness("cuda_flush", {})
    assert not C.is_dispatching_witness("flush", {"uncertified_policy_installed": "flush"})
    assert not C.is_dispatching_witness("harness_keep", {})
    assert not C.is_dispatching_witness("band_witness", {})


def test_recut_an_unreleased_optional_leg_is_not_a_witness():
    import recut_driver_dispatch_record as C

    assert not C.is_dispatching_witness("cuda_alone", {}, ("cuda_alone",))
    assert C.is_dispatching_witness("cuda_shipped", {}, ("cuda_alone",))


# ---------------------------------------------------------------------------
# recut_composition_records: composition and family records
# ---------------------------------------------------------------------------


def test_composition_the_field_is_a_slot_field_of_both_modes():
    import recut_composition_records as R

    assert R.CERTIFIED_POLICIES == fastpath.CERTIFIED_POLICIES
    assert R.CERTIFIED_POLICIES in R.SLOT_FIELDS
    assert R.CERTIFIED_POLICIES in R.FAMILY_SLOT_FIELDS
    assert R.CERTIFIED_POLICIES in fastpath.RUN_FIELDS


def _composition_leg(policy, released=True, digest="1" * 64, pinned="launch.py"):
    import recut_composition_records as R

    name = {"keep": "ieee_keep_ftz_stripped", "flush": "meep_x86_flush"}[policy]
    return {"summary": {"status": "passed" if released else "failed"},
            "subnormal_policy": {"installed": True, "policy": name, "resolved": policy},
            "imported_source_sha256": {R.pinned_path(pinned): digest}}


@pytest.fixture
def flush_run(tmp_path, monkeypatch):
    """One flush composition run holding one leg directory."""
    import recut_composition_records as R

    run = tmp_path / "triton_composition_flush"
    (run / "leg_a").mkdir(parents=True)
    monkeypatch.setattr(R, "leg_capability", lambda payload, dirs: ("8.6", {}))

    def write(payload):
        (run / "leg_a" / "gate.json").write_text(json.dumps(payload))

    def ask(capability="8.6", fresh="1" * 64):
        return R.second_policy_leg(run, "results/flush", ["leg_a"], capability,
                                   ["launch.py"], {"launch.py": fresh}, "flush")

    return R, write, ask


def test_composition_attained_policy_reads_the_resolved_stamp():
    import recut_composition_records as R

    assert R.attained_policy(_composition_leg("keep")) == "keep"
    assert R.attained_policy(_composition_leg("flush")) == "flush"
    assert R.attained_policy({"summary": {"status": "passed"}}) is None


def test_composition_a_released_flush_leg_on_the_same_bytes_is_evidence(flush_run):
    R, write, ask = flush_run
    write(_composition_leg("flush"))
    evidence, why = ask()
    assert why == "" and evidence is not None
    assert evidence["verdict_read_from"] == "results/flush/leg_a/gate.json:summary.status"


def test_composition_an_unreleased_flush_leg_is_refused(flush_run):
    R, write, ask = flush_run
    write(_composition_leg("flush", released=False))
    evidence, why = ask()
    assert evidence is None and "released=False" in why


def test_composition_a_flush_leg_that_attained_keep_is_refused(flush_run):
    R, write, ask = flush_run
    write(_composition_leg("keep"))
    evidence, why = ask()
    assert evidence is None and "attained 'keep'" in why


def test_composition_a_flush_leg_on_other_bytes_is_refused(flush_run):
    R, write, ask = flush_run
    write(_composition_leg("flush", digest="2" * 64))
    evidence, why = ask()
    assert evidence is None and "other bytes for ['launch.py']" in why


def test_composition_a_flush_leg_that_does_not_record_a_pinned_file_is_refused(flush_run):
    R, write, ask = flush_run
    write(_composition_leg("flush", pinned="other_module.py"))
    evidence, why = ask()
    assert evidence is None and "absent ['launch.py']" in why


def test_composition_a_flush_leg_on_another_capability_is_refused(flush_run):
    R, write, ask = flush_run
    write(_composition_leg("flush"))
    evidence, why = ask(capability="9.0")
    assert evidence is None and "'9.0'" in why


def test_composition_a_missing_flush_leg_is_named(tmp_path):
    import recut_composition_records as R

    evidence, why = R.second_policy_leg(tmp_path, "x", ["leg_a"], "8.6", [], {}, "flush")
    assert evidence is None and "the flush run" in why


def _family_root(tmp_path, monkeypatch, attained, families=("complex", "offdiag"),
                 missing=()):
    import recut_composition_records as R

    monkeypatch.setattr(R, "leg_capability", lambda payload, dirs: ("8.6", {}))
    monkeypatch.setattr(R, "HERE", tmp_path)
    real = "meep_gpu/__init__.py"
    digest = R.sha256_of(R.API / real)
    root = tmp_path / "fleet_flush"
    rows = []
    for family in families:
        if family in missing:
            continue
        (root / family).mkdir(parents=True)
        payload = {"canonical_verdict": {"released": True, "reasons": []},
                   "subnormal_policy": {"installed": True, "resolved": attained[family],
                                        "policy": {"keep": "ieee_keep_ftz_stripped",
                                                   "flush": "meep_x86_flush"}[attained[family]]},
                   "imported_source_sha256": {real: digest}}
        (root / family / "gate.json").write_text(json.dumps(payload))
        rows.append({"gate": family, "exit_code": 0, "released": True})
    (root / "campaign.json").write_text(json.dumps({"rows": rows}))
    return R


def test_family_flush_is_recorded_when_every_family_released_under_it(tmp_path, monkeypatch):
    R = _family_root(tmp_path, monkeypatch, {"complex": "flush", "offdiag": "flush"})
    evidence, reasons = R.family_policy_evidence(["fleet_flush"], ["complex", "offdiag"],
                                                 "8.6", "flush")
    assert reasons == [] and sorted(evidence["families"]) == ["complex", "offdiag"]


def test_family_one_family_short_records_no_flush(tmp_path, monkeypatch):
    R = _family_root(tmp_path, monkeypatch, {"complex": "flush", "offdiag": "flush"},
                     missing=("offdiag",))
    evidence, reasons = R.family_policy_evidence(["fleet_flush"], ["complex", "offdiag"],
                                                 "8.6", "flush")
    assert evidence is None and any(r.startswith("offdiag:") for r in reasons)


def test_family_a_family_that_attained_keep_records_no_flush(tmp_path, monkeypatch):
    R = _family_root(tmp_path, monkeypatch, {"complex": "flush", "offdiag": "keep"})
    evidence, reasons = R.family_policy_evidence(["fleet_flush"], ["complex", "offdiag"],
                                                 "8.6", "flush")
    assert evidence is None and reasons == ["offdiag: attained 'keep'"]


def test_composition_flush_arguments_belong_to_their_own_modes():
    import recut_composition_records as R

    with pytest.raises(SystemExit, match="--flush-run belongs to --run"):
        R.main(["--family-recert", "x", "--flush-run", "y"])
    with pytest.raises(SystemExit, match="--flush-family-recert belongs"):
        R.main(["--run", "x", "--flush-family-recert", "y"])


# ---------------------------------------------------------------------------
# gate_dispatch_fused_route: the admitted leg's per-case refusal is read by name
# ---------------------------------------------------------------------------


def _evidence_refusal():
    """Rung 8b's own text for a caller-installed flush with no per-arm evidence."""
    import types

    stub = types.SimpleNamespace(policy_is_installed=lambda: True,
                                 get_subnormal_policy=lambda: "flush",
                                 policy_epoch=lambda: 1)
    xp = types.SimpleNamespace(__name__="cupy")
    return fastpath._drive_one_policy({}, xp, stub, "triton", arms=["PML"],
                                      capability="8.6")


def test_route_admitted_needles_match_the_evidence_refusal(monkeypatch):
    import gate_dispatch_fused_route as G

    monkeypatch.setattr(fastpath, "policy_evidence_reasons",
                        lambda arms, table, capability, policy: ["PML: no flush run"])
    reason = _evidence_refusal()
    assert reason, "rung 8b admitted a flush plan with an arm lacking evidence"
    assert all(needle in reason for needle in G.ADMITTED_EVIDENCE_NEEDLES), reason
    assert repr("flush") in reason


def test_route_a_plain_policy_refusal_does_not_pass_an_admitted_leg():
    """The refusal for a policy the table never dispatches under is not the evidence
    refusal, so an admitted leg cannot pass a case on it."""
    import types

    import gate_dispatch_fused_route as G

    stub = types.SimpleNamespace(policy_is_installed=lambda: True,
                                 get_subnormal_policy=lambda: "keep",
                                 policy_epoch=lambda: 1)
    reason = fastpath._drive_one_policy({}, types.SimpleNamespace(__name__="torch"),
                                        stub, fastpath.METAL_TABLE, arms=["PML"],
                                        capability=None)
    assert reason and "cannot be swapped in place" in reason
    assert not all(needle in reason for needle in G.ADMITTED_EVIDENCE_NEEDLES), reason
