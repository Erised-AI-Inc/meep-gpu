"""The one-time move of each weld's run evidence under the architecture it ran on.

The tool has already run on the shipped ledgers, so what is checked here is the tool
itself, on ledgers built in the test: that the move is REVERSIBLE (nothing of the old
entry is lost and nothing is invented), that every refusal fires and writes nothing, and
that a second run is refused rather than nesting one record inside another.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import migrate_capability_records as migrate  # noqa: E402

from meep_gpu import fastpath  # noqa: E402

A_DIGEST = "a" * 64
B_DIGEST = "b" * 64


def weld(capability: str = "8.6", **extra) -> dict:
    """One entry in the shape this tool migrates: a run's facts beside the digests."""
    entry = {
        "source_sha256": {"kernels.py": A_DIGEST},
        "code_sha256": {"kernels.py": B_DIGEST},
        "status": "PASS",
        "purpose": "a curated claim about what this gate means",
        "host": f"a host, NVIDIA RTX A6000, cc {capability}; CuPy 13.5.1; Triton 3.1.0",
        "records": "apps/api/parity/meep_gpu/results/a_campaign/",
        "recorded_utc": "2026-09-25T16:09:24Z",
        "artifact_sha256": A_DIGEST,
        "verdict_read_from": "gate.json:canonical_verdict.released",
    }
    entry.update(extra)
    return entry


def ledger_file(tmp_path: pathlib.Path, document: dict) -> pathlib.Path:
    """A ledger on disk in the formatting the shipped ones use."""
    path = tmp_path / "fingerprints.json"
    path.write_text(json.dumps(document, indent=2, ensure_ascii=True) + "\n",
                    encoding="utf-8")
    return path


def run(path: pathlib.Path, blocks=None, keys=None):
    """``migrate.migrate`` over a ledger whose owned keys are stated by the test."""
    document, _text = migrate._load(path)
    names = keys if keys is not None else [
        name for name, value in document.items()
        if isinstance(value, dict) and not name.startswith("_")
        and name != "driver_dispatch"]
    return migrate.migrate_document(document, path, table="triton",
                                    blocks=blocks or {}, stamp="2026_09_30",
                                    keys=names)


# ---------------------------------------------------------------------------
# What the move preserves
# ---------------------------------------------------------------------------


def test_the_move_loses_nothing_and_invents_nothing(tmp_path):
    """The inverse: the entry plus its record's fields is the entry it started as."""
    before = {"g": weld()}
    path = ledger_file(tmp_path, before)
    text, refusals, _report = run(path)
    assert refusals == [], refusals
    after = json.loads(text)["g"]
    rebuilt = {name: value for name, value in after.items() if name != "runs"}
    record = dict(after["runs"]["8.6"])
    assert record.pop("bound_sha256")
    rebuilt.update(record)
    assert rebuilt == before["g"]


def test_the_digests_stay_beside_the_entry_and_the_run_facts_move(tmp_path):
    text, refusals, _ = run(ledger_file(tmp_path, {"g": weld()}))
    assert refusals == []
    entry = json.loads(text)["g"]
    assert set(entry["runs"]) == {"8.6"}
    for stays in ("source_sha256", "code_sha256", "status", "purpose"):
        assert stays in entry, stays
    for moves in ("host", "records", "recorded_utc", "artifact_sha256",
                  "verdict_read_from"):
        assert moves not in entry and moves in entry["runs"]["8.6"], moves
    assert fastpath.live_capabilities(entry) == ("8.6",)


def test_the_architecture_is_read_from_the_run_not_typed(tmp_path):
    """A ledger whose host line says 9.0 migrates to 9.0, not to the tool's idea."""
    text, refusals, _ = run(ledger_file(tmp_path, {"g": weld("9.0")}))
    assert refusals == []
    assert set(json.loads(text)["g"]["runs"]) == {"9.0"}


def test_an_entry_the_tool_does_not_own_is_byte_identical_afterwards(tmp_path):
    document = {"g": weld(), "_prose": {"note": "not a weld"},
                "untouched": weld()}
    text, refusals, _ = run(document and ledger_file(tmp_path, document), keys=["g"])
    assert refusals == []
    after = json.loads(text)
    assert after["_prose"] == document["_prose"]
    assert after["untouched"] == document["untouched"]


# ---------------------------------------------------------------------------
# What it refuses
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("host,why", [
    ("a host with no architecture in it", "not exactly one"),
    ("a host, cc 8.6, and also cc 9.0", "not exactly one"),
])
def test_a_host_line_that_does_not_name_one_architecture_refuses(tmp_path, host, why):
    text, refusals, _ = run(ledger_file(tmp_path, {"g": weld(host=host)}))
    assert text is None and any(why in refusal for refusal in refusals), refusals


def test_an_entry_that_records_no_run_refuses(tmp_path):
    entry = weld()
    del entry["records"]
    text, refusals, _ = run(ledger_file(tmp_path, {"g": entry}))
    assert text is None and any("records no records" in r for r in refusals), refusals


def test_an_entry_that_pins_no_bytes_refuses(tmp_path):
    entry = weld()
    del entry["source_sha256"]
    text, refusals, _ = run(ledger_file(tmp_path, {"g": entry}))
    assert text is None and any("binds none of" in r for r in refusals), refusals


def test_one_refusal_stops_the_whole_ledger(tmp_path):
    """All or nothing: a half-migrated ledger is a shape no reader accepts."""
    text, refusals, _ = run(ledger_file(
        tmp_path, {"good": weld(), "bad": weld(host="no architecture here")}))
    assert text is None and len(refusals) == 1


def test_a_ledger_that_is_already_migrated_refuses(tmp_path):
    text, refusals, _ = run(ledger_file(tmp_path, {"g": weld()}))
    assert refusals == []
    again = tmp_path / "migrated.json"
    again.write_text(text, encoding="utf-8")
    text2, refusals2, _ = run(again)
    assert text2 is None
    assert any("already migrated" in refusal for refusal in refusals2), refusals2


def test_a_ledger_this_tool_would_reformat_refuses(tmp_path):
    """It must not be the change that re-indents a 600 KB record."""
    path = tmp_path / "fingerprints.json"
    path.write_text(json.dumps({"g": weld()}, indent=4) + "\n", encoding="utf-8")
    text, refusals, _ = run(path)
    assert text is None
    assert any("round-trip" in refusal for refusal in refusals), refusals


# ---------------------------------------------------------------------------
# The route record
# ---------------------------------------------------------------------------


def test_the_route_record_keeps_its_shape_and_dates_its_old_run(tmp_path):
    """Its run cannot be keyed: no device is readable from either table's record,
    so the old run becomes dated history and the next campaign writes the record."""
    dispatch = {"table": "triton", "source_sha256": {"fastpath.py": A_DIGEST},
                "dispatch_by_default": True, "runnable_slots": ["step_B"],
                "status": "PASS", "records": "results/a_route_campaign",
                "recorded_utc": "2026-09-20T00:00:00Z",
                "validated_compute_capabilities": []}
    text, refusals, report = run(ledger_file(
        tmp_path, {"g": weld(), "driver_dispatch": dispatch}), keys=["g"])
    assert refusals == [], refusals
    entry = json.loads(text)["driver_dispatch"]
    assert entry["runs"] == {}
    assert entry["dispatch_by_default"] is True and entry["table"] == "triton"
    assert "validated_compute_capabilities" not in entry
    history = [value for key, value in entry.items()
               if key.startswith("_route_run_before_capability_records")]
    assert len(history) == 1
    assert history[0]["records"] == "results/a_route_campaign"
    assert "why" in " ".join(history[0]).lower() or history[0]["_why_this_is_history"]


def test_the_shipped_ledgers_carry_the_note_and_no_retired_key():
    """What actually happened to the records that ship, not to a fixture."""
    for table in ("triton", fastpath.CUDA_TABLE):
        ledger = fastpath._fingerprints(table)
        assert ledger["_capability_runs_migration"]["capabilities"] == ["8.6"]
        assert "validated_compute_capabilities" not in ledger
        assert isinstance(ledger["driver_dispatch"]["runs"], dict)
