"""The array-path oracle's minting tool: it writes only the leaves, and refuses by name.

Every test writes into copies of the three ledgers under ``tmp_path``; the tree's own
ledgers are only read. No test here asks whether the COMMITTED leaves describe this
tree: that is a certification question, asked by the three weld contracts, and an
array-path edit makes it red by design until the tables are certified again.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import record_array_path_oracle as tool  # noqa: E402

from meep_gpu import array_path_oracle as oracle  # noqa: E402

ROOT = tool._API


def _copies(tmp_path: Path, strip: bool = False) -> dict:
    """The three ledgers copied under ``tmp_path``, optionally with every leaf removed."""
    paths = {}
    for table, relative in tool.LEDGERS.items():
        target = tmp_path / f"{table}.json"
        shutil.copyfile(ROOT / relative, target)
        if strip:
            ledger = json.loads(target.read_text(encoding="utf-8"))
            for entry in ledger.values():
                if isinstance(entry, dict):
                    entry.pop(oracle.LEAF_KEY, None)
            target.write_text(tool.canonical(ledger), encoding="utf-8")
        paths[table] = target
    return paths


def test_the_tool_writes_only_the_leaves_and_is_idempotent(tmp_path):
    paths = _copies(tmp_path, strip=True)
    stripped = {table: json.loads(path.read_text(encoding="utf-8"))
                for table, path in paths.items()}
    report = tool.record(ROOT, write=True, ledgers=paths)
    assert report["written"] is True
    for table, path in paths.items():
        written = json.loads(path.read_text(encoding="utf-8"))
        moved = tool.changed_leaves(stripped[table], written)
        assert moved and all(trail[1] == oracle.LEAF_KEY for trail in moved), table
        assert tool.entry_bindings(written) == tool.entry_bindings(stripped[table]), table
        assert oracle.ledger_problems(written, oracle.live(), table) == []
        assert report["tables"][table]["problems_after"] == []
        assert path.read_text(encoding="utf-8") == tool.canonical(written)
    again = tool.record(ROOT, write=True, ledgers=paths)
    assert again["status"] == "current" and again["written"] is False


def test_the_tool_refuses_a_fixture_that_misses_a_declared_class(tmp_path, monkeypatch):
    paths = _copies(tmp_path, strip=True)
    before = {table: path.read_bytes() for table, path in paths.items()}
    monkeypatch.setattr(oracle, "CASES", tuple(
        dataclasses.replace(spec, pml=None) if spec.name == "pml_3d" else spec
        for spec in oracle.CASES))
    with pytest.raises(tool.Refusal, match="pml_3d/curl_B/Bx/recurrence"):
        tool.record(ROOT, write=True, ledgers=paths)
    assert {table: path.read_bytes() for table, path in paths.items()} == before


def test_the_tool_refuses_a_ledger_it_would_reformat(tmp_path):
    paths = _copies(tmp_path)
    ledger = json.loads(paths["metal"].read_text(encoding="utf-8"))
    paths["metal"].write_text(json.dumps(ledger, indent=1, sort_keys=True) + "\n",
                              encoding="utf-8")
    with pytest.raises(tool.Refusal, match="does not round-trip"):
        tool.record(ROOT, write=True, ledgers=paths)


def test_verify_names_every_table_when_the_array_path_moves(tmp_path, monkeypatch):
    from meep_gpu import stepping

    paths = _copies(tmp_path, strip=True)
    tool.record(ROOT, write=True, ledgers=paths)
    assert tool.verify(ROOT, ledgers=paths)["verified"] is True

    original = stepping._curl_from_operands

    def regrouped(operands, dtdx, presence, scratch=None):  # The pairwise grouping.
        if presence != (True, True):
            return original(operands, dtdx, presence, scratch=scratch)
        return dtdx * ((operands.shifted_first - operands.first)
                       + (operands.second - operands.shifted_second))

    monkeypatch.setattr(stepping, "_curl_from_operands", regrouped)
    report = tool.verify(ROOT, ledgers=paths)
    assert report["verified"] is False
    for table in tool.LEDGERS:
        assert any(problem.startswith(f"{table}: the array path's behaviour moved")
                   for problem in report["tables"][table]), report["tables"][table]
