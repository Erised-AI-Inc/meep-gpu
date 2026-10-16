"""The Metal round certifies a new Apple GPU through the same chunked round, by a flag that says so.

``metal_round.py`` runs the Metal fleet chunk by chunk (``rounds/1.0/chunks.py``). These tests
pin, host-only (no GPU, no MEEP), what a second Apple GPU needs from it:

* the preflight of a chunk holds the index's device and mode to this Mac's architecture and
  to the Metal ledger both ways, and the flag is refused without a chunk and on the route
  stage (:func:`metal_round.architecture_problems`, :func:`metal_round.apply_chunk`);
* every stage that runs gates reads this Mac's architecture (the fleet too, so a fleet
  launched without its preflight is held to it), and an unchunked round runs only on an
  architecture the ledger admits (:meth:`metal_round.Round.device_problems`);
* ``status`` over an index reads each gate under the newest launch that ran it, never the
  plan, so a gate run again alone is read under its new run;
* the MEEP corpus rule is ``legs.py``'s, one home for both round drivers.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND = HERE / "rounds" / "1.0"
for path in (str(ROUND), str(HERE), str(HERE.parents[1])):
    if path not in sys.path:
        sys.path.insert(0, path)

import chunks as C  # noqa: E402
import legs as L  # noqa: E402
import metal_round  # noqa: E402

ROUND_STAMP = "2099-03-04_10"
M1, M3 = "applegpu_g13s", "applegpu_g15s"


def _index(device: str, mode: str) -> dict:
    units = C.metal_units(HERE.parents[1])
    return C.make_index("metal", ROUND_STAMP, C.partition(units, 3), {"rule": "test"},
                        device=device, device_mode=mode)


def test_a_chunk_of_another_architecture_or_mode_is_refused_by_name():
    certified, new = _index(M1, "certified"), _index(M3, "new")
    assert metal_round.architecture_problems(certified, M1, [M1], False) == []
    assert metal_round.architecture_problems(new, M3, [M1], True) == []
    assert "this Mac is applegpu_g15s" in metal_round.architecture_problems(
        certified, M3, [M1], False)[0]
    assert any("pass --new-architecture" in p
               for p in metal_round.architecture_problems(new, M3, [M1], False))
    assert any("already admits applegpu_g13s" in p
               for p in metal_round.architecture_problems(_index(M1, "new"), M1, [M1], True))
    assert any("gated with --new-architecture" in p
               for p in metal_round.architecture_problems(_index(M3, "certified"), M3, [M1],
                                                          False))
    assert "could not be read" in metal_round.architecture_problems(new, None, [M1], True)[0]
    assert any("admission could not be read" in p
               for p in metal_round.architecture_problems(new, M3, None, True))


def _args(**kw) -> argparse.Namespace:
    base = dict(stage="fleet", stamp=None, chunk_index="", chunk="", only="", logs="",
                results="", repo=str(HERE.parents[1]), new_architecture=False)
    base.update(kw)
    return argparse.Namespace(**base)


def test_the_flag_is_refused_without_a_chunk_and_on_the_route_stage(tmp_path):
    assert metal_round.apply_chunk(_args(stamp="s", new_architecture=True)) == \
        metal_round.NEW_ARCHITECTURE_UNCHUNKED
    path = tmp_path / "index.json"
    C.write_index(path, _index(M3, "new"))
    assert "fleet mode" in metal_round.apply_chunk(
        _args(stage="route", chunk_index=str(path), chunk="m0", new_architecture=True))
    chunk = _args(chunk_index=str(path), chunk="m1", new_architecture=True)
    assert metal_round.apply_chunk(chunk) is None
    assert chunk.stamp == f"{ROUND_STAMP}_{M3}_m1"


def test_status_reads_each_gate_under_its_newest_launch(tmp_path):
    index = _index(M1, "certified")
    results = tmp_path / "results"
    first = index["chunks"][1]
    gate = C.metal_gates(first)[0]
    again = C.new_chunk(index, [{"id": f"metal/{gate}", "expect_s": 1}], "one gate again")
    for chunk, utc in ((first, "2099-03-04T01:00:00Z"), (again, "2099-03-04T02:00:00Z")):
        record = results / C.launch_root(chunk["stamp"]) / "LAUNCH.json"
        record.parent.mkdir(parents=True)
        record.write_text(json.dumps({
            "schema": C.LAUNCH_SCHEMA, "machine": "metal", "device": M1,
            "round": ROUND_STAMP, "chunk": chunk["id"], "stamp": chunk["stamp"],
            "units": sorted(chunk["units"]), "tip": "0" * 40, "tree": "0" * 40, "utc": utc}))
    path = tmp_path / "index.json"
    C.write_index(path, index)
    status = _args(stage="status", chunk_index=str(path), results=str(results))
    assert metal_round.apply_chunk(status) is None
    assert status.stamp_of[gate] == again["stamp"]
    others = [g for g in C.metal_gates(first) if g != gate]
    assert all(status.stamp_of[g] == first["stamp"] for g in others)


def test_the_corpus_rule_has_one_home():
    for name in ("resolve_corpus", "read_corpus_record", "verify_corpus", "sha256_of"):
        assert getattr(metal_round, name) is getattr(L, name), name
    assert (metal_round.BUILD_RECORD, metal_round.BUILD_RECORD_DIR) == (L.BUILD_RECORD,
                                                                        L.BUILD_RECORD_DIR)


def _round(tmp_path: Path, index=None, new_architecture=False) -> "metal_round.Round":
    chunk = index["chunks"][0] if index else None
    args = argparse.Namespace(
        repo=str(tmp_path / "repo"), results=str(tmp_path / "results"),
        logs=str(tmp_path / "logs"), stamp=chunk["stamp"] if chunk else "t",
        python=sys.executable, meep_source="", dry_run_allow=False,
        new_architecture=new_architecture, chunk_manifest=chunk, chunk_index_data=index)
    (tmp_path / "repo").mkdir(parents=True, exist_ok=True)
    return metal_round.Round(args)


def _answer(monkeypatch, architecture, admitted) -> None:
    """This Mac's architecture and the Metal ledger's admission, as the checkout answers."""
    real = metal_round.subprocess.run

    def run(command, *args, **kwargs):
        if metal_round.ADMISSION_PROGRAM in command:
            return subprocess.CompletedProcess(command, 0, json.dumps(
                {"arch": architecture, "admitted": admitted}) + "\n", "")
        return real(command, *args, **kwargs)
    monkeypatch.setattr(metal_round.subprocess, "run", run)


def test_an_unchunked_round_runs_only_on_an_architecture_the_ledger_admits(tmp_path,
                                                                           monkeypatch):
    rnd = _round(tmp_path)
    _answer(monkeypatch, M1, [M1])
    assert rnd.device_problems("preflight") == []
    _answer(monkeypatch, M3, [M1])
    found = rnd.device_problems("preflight")
    assert len(found) == 1 and M3 in found[0] and "--new-architecture" in found[0]
    _answer(monkeypatch, None, [M1])
    assert "could not be read" in rnd.device_problems("fleet")[0]
    _answer(monkeypatch, M3, None)
    assert "admission could not be read" in rnd.device_problems("fleet")[0]
    chunked = _round(tmp_path, _index(M3, "new"), new_architecture=True)
    _answer(monkeypatch, M3, [M1])
    assert chunked.device_problems("preflight") == []
    chunked.args.new_architecture = False
    assert any("pass --new-architecture" in p for p in chunked.device_problems("fleet"))


def test_the_fleet_stage_is_held_to_the_architecture_without_its_preflight(tmp_path,
                                                                         monkeypatch):
    """A fleet launched alone, on an M3 under an index of the certified M1, stops before it
    writes a launch record or runs a gate; so does an unchunked fleet on the M3."""
    monkeypatch.setattr(metal_round.Round, "tree_is_the_commit", lambda self, stage: None)
    monkeypatch.setattr(metal_round.Round, "corpus", lambda self: {
        "root": None, "record": None, "problems": [], "notes": []})
    monkeypatch.setattr(metal_round.Round, "lifting_gates", lambda self: [])
    _answer(monkeypatch, M3, [M1])
    index = _index(M1, "certified")
    rnd = _round(tmp_path, index)
    assert rnd.stage_fleet() == 3
    log = (tmp_path / "logs" / "round.log").read_text(encoding="utf-8")
    assert "this Mac is applegpu_g15s" in log
    assert not (tmp_path / "results" / C.launch_root(index["chunks"][0]["stamp"])).exists()
    assert not (tmp_path / "results" / f"metal_environment_{index['chunks'][0]['stamp']}"
                ).exists()
    unchunked = _round(tmp_path / "unchunked")
    assert unchunked.stage_fleet() == 3
    assert "--new-architecture" in (tmp_path / "unchunked" / "logs" / "round.log").read_text(
        encoding="utf-8")
