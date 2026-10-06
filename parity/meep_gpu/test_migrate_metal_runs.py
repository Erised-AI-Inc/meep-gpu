"""The one-time move of each Metal weld's run under the GPU architecture it ran on.

The tool is checked on ledgers built in the test -- the move is REVERSIBLE (nothing of
the old entry is lost and nothing is invented), every refusal fires and writes nothing,
and a second run is refused rather than nesting one record inside another -- and on the
shipped ledger, in whichever state it is in: before the move, the report mode's
verification must hold over all of it; after it, the tool must refuse to run again.
"""

from __future__ import annotations

import json
import pathlib
import re
import shutil
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import migrate_metal_runs as migrate  # noqa: E402

from meep_gpu import fastpath, metal_dispatch, metal_runs  # noqa: E402

A_DIGEST = "a" * 64
B_DIGEST = "b" * 64
HOST = "this machine: Apple MPS device applegpu_g13s, torch 2.10.0, metalfe-32023.850.10"


def weld(host: str = HOST, **extra) -> dict:
    """One entry in the shape this tool migrates: a run's facts beside the digests."""
    entry = {
        "source_sha256": {"meep_gpu/metal_kernels/launch.py": A_DIGEST},
        "code_sha256": {"meep_gpu/metal_kernels/launch.py": B_DIGEST},
        "status": "PASS",
        "purpose": "a curated claim about what this gate means",
        "host": host,
        "records": "apps/api/parity/meep_gpu/results/metal_x_2026-10-04_g13s/ - 3 records",
        "recorded_utc": "2026-10-05T04:19:22Z",
        "artifact_sha256": "c" * 64,
        "environment_read_from": "metal_environment_2026-10-04_g13s/start.json",
        "subnormal_policy": "flush - native and uncontrollable on MPS",
        "verdict_read_from": "metal_x_2026-10-04_g13s/gate.json:release.released",
    }
    entry.update(extra)
    return entry


def dispatch_record() -> dict:
    """A Metal driver_dispatch record in the shape the route recut wrote before."""
    return {
        "table": "metal",
        "source_sha256": {"meep_gpu/metal_dispatch.py": A_DIGEST},
        "code_sha256": {"meep_gpu/metal_dispatch.py": B_DIGEST},
        "exclusions": {"released_fused_arms": {
            "arms": ["fused pair B"], "gate": "dispatch_metal_route_x",
            "artifact": "apps/api/parity/meep_gpu/results/dispatch_metal_route_x/",
            "what_was_measured": "the arm launched"}},
        "residency": {"mode": "held", "per_case": {"shipped": {"pml_2d": {"held": []}}}},
        "lift": {"per_leg": {"shipped": {"prefer_gpu": True}}},
        "subnormal": {"certification_policy": "flush"},
        "environments": {"recorded": []},
    }


def text_of(document: dict) -> str:
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def run(document: dict, cited=None):
    keys = metal_runs.weld_keys(document)
    return migrate.migrate_document(document, text_of(document), stamp="2026_10_05",
                                    keys=keys, cited=keys if cited is None else cited)


# ---------------------------------------------------------------------------
# What the move preserves
# ---------------------------------------------------------------------------


def test_the_move_loses_nothing_and_invents_nothing():
    """The inverse: the entry plus its run's fields is the entry it started as."""
    before = weld()
    text, refusals, _report = run({"g": before})
    assert refusals == [], refusals
    after = json.loads(text)["g"]
    rebuilt = {name: value for name, value in after.items() if name != "runs"}
    record = dict(after["runs"]["applegpu_g13s"])
    assert record.pop("bound_sha256") == fastpath.bound_digest(after)
    assert (record.pop("torch"), record.pop("metal_frontend")) == ("2.10.0", "32023.850.10")
    rebuilt.update(record)
    assert rebuilt == before


def test_the_digests_stay_beside_the_entry_and_the_run_facts_move():
    text, refusals, _report = run({"g": weld()})
    assert refusals == []
    after = json.loads(text)["g"]
    assert set(after) == {"source_sha256", "code_sha256", "status", "purpose", "runs"}
    assert metal_runs.shape_reasons(after) == []
    assert metal_runs.live_architectures(after) == ("applegpu_g13s",)


def test_the_architecture_is_read_from_the_host_line_not_typed():
    other = HOST.replace("applegpu_g13s", "applegpu_g15p")
    text, refusals, report = run({"g": weld(), "h": weld(other)})
    assert refusals == []
    assert report["architectures"] == ["applegpu_g13s", "applegpu_g15p"]
    assert list(json.loads(text)["h"]["runs"]) == ["applegpu_g15p"]


def test_an_entry_the_tool_does_not_own_is_byte_identical_afterwards():
    keep = {"host_sha256": {"launch.py": A_DIGEST}, "validated_torch_versions": ["2.10.0"]}
    text, refusals, _report = run({"g": weld(), "metal_kernels": keep})
    assert refusals == []
    assert json.loads(text)["metal_kernels"] == keep


def test_every_digest_is_kept_and_only_the_bounds_are_added():
    text, refusals, report = run({"g": weld(), "h": weld(), "driver_dispatch":
                                  dispatch_record()})
    assert refusals == []
    assert report["digests_after"] - report["digests_before"] == 2
    assert report["runs_written"] == 2


def test_the_route_record_keeps_its_shape_and_dates_its_old_run():
    text, refusals, report = run({"g": weld(), "driver_dispatch": dispatch_record()})
    assert refusals == []
    record = json.loads(text)["driver_dispatch"]
    assert record["runs"] == {}
    history = record[f"{migrate.HISTORY_PREFIX}2026_10_05"]
    assert set(history) == {"residency", "lift", "released_fused_arms",
                            "_why_this_is_history"}
    assert history["released_fused_arms"]["gate"] == "dispatch_metal_route_x"
    assert record["exclusions"]["released_fused_arms"] == {"arms": ["fused pair B"]}
    assert record["source_sha256"] == dispatch_record()["source_sha256"]
    assert record["subnormal"] == dispatch_record()["subnormal"]
    assert not re.search(r"\b[0-9a-f]{64}\b", json.dumps(history))
    assert report["driver_dispatch_moved"] == sorted(history)


def test_the_route_records_move_is_checked_reversible(monkeypatch):
    """A move of driver_dispatch that loses a field is refused, with nothing written."""
    document = {"metal_x_device_gate": weld(), "driver_dispatch": dispatch_record()}
    text, refusals, _report = run(document)
    assert refusals == [] and text is not None
    migrated = json.loads(text)["driver_dispatch"]
    assert migrate._unmigrate_dispatch(migrated) == dispatch_record()

    real = migrate._migrate_dispatch

    def lossy(entry, stamp):
        entry, moved = real(entry, stamp)
        history = entry[f"{migrate.HISTORY_PREFIX}{stamp}"]
        del history["lift"]
        return entry, moved

    monkeypatch.setattr(migrate, "_migrate_dispatch", lossy)
    text, refusals, _report = run(document)
    assert text is None
    assert any(refusal.startswith("driver_dispatch: not reversible (lost ['lift']")
               for refusal in refusals), refusals


def test_the_note_names_what_moved_and_states_no_digest():
    text, refusals, _report = run({"g": weld()})
    assert refusals == []
    note = json.loads(text)[migrate.NOTE_KEY]
    assert note["entries"] == 1 and note["architectures"] == ["applegpu_g13s"]
    assert note["certified_before"] == {"architecture": "applegpu_g13s",
                                        "torch": "2.10.0",
                                        "metal_frontend": "32023.850.10"}
    assert not re.search(r"\b[0-9a-f]{64}\b", json.dumps(note))


# ---------------------------------------------------------------------------
# What it refuses, with nothing written
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("line,why", [
    ("this machine: Apple MPS device, torch 2.10.0, metalfe-32023.850.10",
     "names no ['architecture']"),
    ("this machine: Apple MPS device applegpu_g13s, torch ?, metalfe-32023.850.10",
     "names no ['torch']"),
    ("this machine: Apple MPS device applegpu_g13s, torch 2.10.0",
     "names no ['metal_frontend']"),
    ("this machine: Apple MPS device applegpu_G13S, torch 2.10.0, metalfe-1.2",
     "not a GPU architecture key"),
])
def test_a_host_line_that_does_not_name_one_environment_refuses(line, why):
    text, refusals, _report = run({"g": weld(line)})
    assert text is None
    assert any(why in refusal for refusal in refusals), refusals


def test_an_entry_that_records_no_run_refuses():
    entry = weld()
    del entry["records"]
    text, refusals, _report = run({"g": entry})
    assert text is None and "records no ['records']" in refusals[0], refusals


def test_an_entry_that_pins_no_bytes_refuses():
    entry = weld()
    del entry["source_sha256"]
    entry["artifact_only"] = True
    text, refusals, _report = migrate.migrate_document(
        {"g": entry}, text_of({"g": entry}), stamp="s", keys=["g"], cited=["g"])
    assert text is None and "binds none of" in refusals[0], refusals


def test_one_refusal_stops_the_whole_ledger():
    text, refusals, _report = run({"g": weld(), "h": weld("no line")})
    assert text is None and len(refusals) == 1, refusals


def test_a_ledger_that_is_already_migrated_refuses():
    text, _refusals, _report = run({"g": weld()})
    again = json.loads(text)
    assert migrate.migrate_document(again, text, stamp="s")[1] == [
        f"already migrated (it carries {migrate.NOTE_KEY!r} or a 'runs' record)"]


def test_a_ledger_this_tool_would_reformat_refuses():
    document = {"g": weld()}
    text = json.dumps(document, indent=4) + "\n"
    out, refusals, _report = migrate.migrate_document(document, text, stamp="s",
                                                      keys=["g"], cited=["g"])
    assert out is None and "would reformat" in refusals[0]


# ---------------------------------------------------------------------------
# The shipped ledger, in whichever state it is in
# ---------------------------------------------------------------------------


def test_the_shipped_ledger_migrates_entry_by_entry_or_already_has():
    """Before the move: the report mode verifies all of it. After: it refuses to run again.

    Before, every check the tool makes before writing must hold over the shipped
    ledger -- every digest kept, one live run per weld bound by
    ``fastpath.bound_digest``, each host line carried byte for byte, and the cited
    welds certifying exactly the environment their host lines recorded. After, the
    note is present and the tool refuses a second run by name.
    """
    text = migrate.LEDGER.read_text(encoding="utf-8")
    ledger = json.loads(text)
    out, refusals, report = migrate.migrate_document(ledger, text, stamp="2026_10_05")
    if migrate.NOTE_KEY in ledger:
        assert out is None and "already migrated" in refusals[0], refusals
        return
    assert refusals == [], refusals[:5]
    welds = metal_runs.weld_keys(ledger)
    assert report["entries"] == len(welds) >= 62
    assert report["runs_written"] == len(welds)
    assert report["digests_after"] - report["digests_before"] == len(welds)
    assert report["walker_run_bound_digests"] == len(welds)
    assert report["admitted_before"] is not None, (
        "the cited welds' host lines do not name one environment")
    assert report["admission_after"] == {"architecture": True, "torch": True,
                                         "metal_frontend": True}
    migrated = json.loads(out)
    for key in welds:
        entry = migrated[key]
        (architecture,) = metal_runs.live_architectures(entry)
        assert entry["runs"][architecture]["host"] == ledger[key]["host"], key


def test_the_tool_writes_a_copy_of_the_shipped_ledger_and_then_refuses_it(
        tmp_path, monkeypatch, capsys):
    """``main`` end to end on a copy: report mode writes nothing, --write writes once."""
    copy = tmp_path / "fingerprints.json"
    shutil.copyfile(migrate.LEDGER, copy)
    monkeypatch.setattr(migrate, "LEDGER", copy)
    monkeypatch.setattr(migrate, "_API", tmp_path)
    before = copy.read_bytes()
    if migrate.NOTE_KEY in json.loads(before):
        assert migrate.main([]) == 1
        assert copy.read_bytes() == before
        return
    assert migrate.main([]) == 0
    assert copy.read_bytes() == before
    assert "report only" in capsys.readouterr().out
    assert migrate.main(["--write"]) == 0
    written = json.loads(copy.read_text(encoding="utf-8"))
    assert migrate.NOTE_KEY in written
    assert metal_runs.admission_report(written, metal_runs.cited_keys())["admitted"] == (
        written[migrate.NOTE_KEY]["certified_before"]["architecture"],)
    assert dict(metal_dispatch.cited_environments(ledger=written)).keys() == set(
        metal_runs.cited_keys())
    assert migrate.main([]) == 1
