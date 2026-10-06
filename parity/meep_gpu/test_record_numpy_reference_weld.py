"""``record_numpy_reference_weld.py`` writes one digest, and only for a run that states the record.

The tool runs the NumPy validator the D/E record names and moves
``fused_electric_gate.numpy_reference_weld.probe_sha256`` to the validator's sha256.
These tests drive it on a scratch tree (a synthetic ledger and a stand-in validator
that writes a chosen artifact), so each refusal is exercised without the evidence
archive and without the real validator's runtime. Every refusal test also checks that
the ledger's bytes did not move. The last three tests read the shipped tree.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import record_numpy_reference_weld as w1  # noqa: E402

from meep_gpu import fastpath  # noqa: E402

VALIDATOR = "parity/meep_gpu/validate_stand_in.py"
HELPER = "loaded_by_the_stand_in.py"
OLD = "0" * 64
RAN, SKIPPED = 6, 2

STAND_IN = '''\
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HELPER = os.path.join(HERE, "{helper}")
parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
args = parser.parse_args()
for index in range({cases}):
    print(f"[ref] case {{index + 1}}/{cases}", flush=True)
if {edit_helper}:
    with open(HELPER, "a", encoding="utf-8") as handle:
        handle.write("# edited while the run was in flight\\n")
if {write_artifact}:
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({{"check": {check!r}, "backend": {backend!r},
                    "cases": [{{}}] * {cases}, "summary": {summary!r}}}, handle)
sys.exit({returncode})
'''


def stand_in(*, ran=RAN, identical=RAN, skipped=SKIPPED, passed=True, cases=None,
             check="pml_reference_vs_stepping", backend="numpy", returncode=0,
             edit_helper=False, write_artifact=True) -> str:
    summary = {"ran": ran, "identical": identical, "skipped": skipped, "pass": passed}
    return STAND_IN.format(helper=HELPER, cases=RAN + SKIPPED if cases is None else cases,
                           edit_helper=edit_helper, write_artifact=write_artifact,
                           check=check, backend=backend, summary=summary,
                           returncode=returncode)


def ledger_for(weld: dict) -> dict:
    """A ledger with the D/E entry, bound on 8.6 as the shipped one is, and one other weld."""
    gate = {"kernel": "fused_curl_constitutive_D", "probe": "parity/meep_gpu/gate_stand_in.py",
            "probe_sha256": "a" * 64, "numpy_reference_weld": weld}
    other = {"source_sha256": {"meep_gpu/triton_kernels/launch.py": "b" * 64}}
    for entry in (gate, other):
        fastpath.bind_capability(entry, bound_before=None, capability="8.6",
                                 run={"host": "a test run", "records": "runs/",
                                      "artifact_sha256": "c" * 64})
    return {"fused_electric_gate": gate, "other_device_gate": other,
            "kernel_source_sha256": "d" * 64}


def make_tree(tmp_path, source: str, *, weld_changes=None, indent=2) -> pathlib.Path:
    root = tmp_path / "tree"
    (root / "meep_gpu" / "triton_kernels").mkdir(parents=True)
    harness = root / "parity" / "meep_gpu"
    harness.mkdir(parents=True)
    (harness / HELPER).write_text("LOADED = True\n", encoding="utf-8")
    (root / VALIDATOR).write_text(source, encoding="utf-8")
    weld = {"probe": VALIDATOR, "probe_sha256": OLD, "structurally_invalid_skipped": SKIPPED,
            "valid_cases_bit_identical": f"{RAN}/{RAN}"}
    weld.update(weld_changes or {})
    weld = {key: value for key, value in weld.items() if value is not None}
    ledger = ledger_for(weld)
    text = json.dumps(ledger, indent=indent, sort_keys=True) + "\n"
    (root / w1.LEDGER_RELATIVE).write_text(text, encoding="utf-8")
    return root


def ledger_bytes(root: pathlib.Path) -> bytes:
    return (root / w1.LEDGER_RELATIVE).read_bytes()


def validator_digest(root: pathlib.Path) -> str:
    return w1.sha256_of(root / VALIDATOR)


def test_the_report_runs_the_validator_and_writes_nothing(tmp_path):
    root = make_tree(tmp_path, stand_in())
    before = ledger_bytes(root)
    report = w1.record(root)
    assert report["status"] == "stale" and report["written"] is False
    assert report["before"] == OLD and report["after"] == validator_digest(root)
    assert report["summary"]["ran"] == RAN and report["summary"]["skipped"] == SKIPPED
    assert set(report["watched"]) == {"validate_stand_in.py", HELPER}
    assert ledger_bytes(root) == before


def test_the_write_moves_only_the_leaf_and_leaves_every_record_live(tmp_path):
    root = make_tree(tmp_path, stand_in())
    before = json.loads(ledger_bytes(root))
    out = tmp_path / "kept" / "artifact.json"
    report = w1.record(root, write=True, validator_out=out)
    after = json.loads(ledger_bytes(root))
    assert report["written"] is True and out.is_file()
    assert report["artifact_sha256"] == w1.sha256_of(out)
    assert after["fused_electric_gate"]["numpy_reference_weld"]["probe_sha256"] == \
        validator_digest(root)
    assert w1.changed_leaves(before, after) == [
        ("fused_electric_gate", "numpy_reference_weld", "probe_sha256")]
    for key in ("fused_electric_gate", "other_device_gate"):
        assert fastpath.live_capabilities(after[key]) == ("8.6",)
    assert ledger_bytes(root).decode("utf-8") == w1.canonical(after)


def test_a_current_digest_is_not_rewritten(tmp_path):
    root = make_tree(tmp_path, stand_in())
    w1.record(root, write=True)
    before = ledger_bytes(root)
    report = w1.record(root, write=True)
    assert report["status"] == "current" and report["written"] is False
    assert ledger_bytes(root) == before


@pytest.mark.parametrize("changes, named", [
    ({"identical": RAN - 1}, "summary.identical"),
    ({"skipped": SKIPPED + 1}, "summary.skipped"),
    ({"ran": RAN + 1, "identical": RAN + 1}, "summary.ran"),
    ({"passed": False}, "summary.pass"),
    ({"check": "another_check"}, "check is"),
    ({"backend": "cupy"}, "backend is"),
    ({"cases": RAN}, "cases, not ran"),
    ({"returncode": 1}, "exited 1"),
    ({"write_artifact": False}, "wrote no artifact"),
    ({"edit_helper": True}, "changed while the validator ran"),
], ids=["identical_short", "skipped_differs", "ran_differs", "not_a_pass", "another_check",
        "another_backend", "cases_missing", "nonzero_exit", "no_artifact",
        "script_edited_during_the_run"])
def test_a_run_that_does_not_state_the_record_is_refused(tmp_path, changes, named):
    root = make_tree(tmp_path, stand_in(**changes))
    before = ledger_bytes(root)
    with pytest.raises(w1.Refusal, match=named):
        w1.record(root, write=True)
    assert ledger_bytes(root) == before


@pytest.mark.parametrize("weld_changes, named", [
    ({"valid_cases_bit_identical": f"{RAN - 1}/{RAN}"}, "claims every case"),
    ({"valid_cases_bit_identical": "all"}, "not 'N/N'"),
    ({"structurally_invalid_skipped": "44"}, "not a count"),
    ({"probe": "meep_gpu/validate_stand_in.py"}, "not a script of parity/meep_gpu"),
    ({"probe": "parity/meep_gpu/absent.py"}, "not in the tree"),
    ({"extra": "a field the tool does not know"}, "does not reshape it"),
    ({"probe_sha256": None}, "does not reshape it"),
], ids=["claim_not_a_full_pass", "claim_not_a_ratio", "skipped_not_a_count",
        "probe_outside_the_harness", "probe_absent", "extra_field", "digest_field_missing"])
def test_a_record_the_tool_cannot_keep_is_refused_before_the_run(tmp_path, weld_changes,
                                                                  named):
    root = make_tree(tmp_path, stand_in(), weld_changes=weld_changes)
    before = ledger_bytes(root)
    with pytest.raises(w1.Refusal, match=named):
        w1.record(root, write=True)
    assert ledger_bytes(root) == before


def test_a_ledger_in_another_serialisation_is_refused(tmp_path):
    root = make_tree(tmp_path, stand_in(), indent=1)
    before = ledger_bytes(root)
    with pytest.raises(w1.Refusal, match="does not round-trip"):
        w1.record(root, write=True)
    assert ledger_bytes(root) == before


def test_the_shipped_ledger_is_in_the_binders_serialisation():
    path = w1._API / w1.LEDGER_RELATIVE
    assert w1.canonical(json.loads(path.read_text(encoding="utf-8"))) == \
        path.read_text(encoding="utf-8")


def test_the_shipped_block_names_a_harness_script_and_claims_a_full_pass():
    ledger = json.loads((w1._API / w1.LEDGER_RELATIVE).read_text(encoding="utf-8"))
    weld = ledger[w1.GATE_KEY][w1.WELD_KEY]
    assert set(weld) == set(w1.WELD_FIELDS)
    assert w1.validator_path(w1._API, weld).name == "validate_pml_reference_vs_stepping.py"
    ran, skipped = w1.claimed_counts(weld)
    assert ran > 0 and skipped >= 0


def test_the_watch_reads_the_scripts_the_shipped_validator_loads():
    """If the pattern stopped matching, the watch would silently shrink to the validator."""
    validator = w1._API / "parity" / "meep_gpu" / "validate_pml_reference_vs_stepping.py"
    assert {path.name for path in w1.loaded_by_path(validator)} == {
        "probe_fused_kernel_bit_identity.py", "gate_triton_fused_electric.py"}
