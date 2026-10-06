"""Every route cut files its verdict under ``runs[<key>]`` and nowhere else.

A table's ``driver_dispatch`` record is not a weld. ``recut_driver_dispatch_record.py``
files each route run under ``runs[<key>]`` (a compute capability on the NVIDIA tables,
a GPU architecture on Metal) with that run's ``status``, and the three weld contracts
check the record through those runs with one rule
(``weld_record_walk.route_record_problems``): at least one live run, every live run
PASS, every pinned file matching the tree, and no route-run field, ``status`` among
them, beside the digests.

So the writer is held to the same rule here: a CUDA cut files PASS under the
capability and none beside the digests, the record it writes satisfies the CUDA
contract's own check, and the cut of every backend refuses a record that carries an
entry-level ``status``. The Metal cut is driven end to end in
``test_recut_metal_runs.py``.

Driven over a CUDA route campaign built in the test against the live tree's digests,
and over records built in the test; no GPU and no results tree are read.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import recut_driver_dispatch_record as recut  # noqa: E402

from meep_gpu import fastpath, fastpath_cuda  # noqa: E402
from meep_gpu import test_cuda_weld_contract as contract  # noqa: E402

CAPABILITY = "8.6"
CUDA_LEGS = recut.BACKENDS["cuda"]["required_legs"]


def _live(name: str) -> str:
    return hashlib.sha256(recut._resolve("cuda", name).read_bytes()).hexdigest()  # noqa: SLF001


def _campaign(results: pathlib.Path) -> str:
    """A released five-leg CUDA route campaign on the live bytes, as the gate writes it."""
    name = fastpath.route_campaign(fastpath_cuda.CUDA_DRIVER_ROUTE_FUSED_GATE, CAPABILITY)
    released = fastpath_cuda.CUDA_RELEASED_FUSED_ARMS
    cases = sorted({case for arm_cases in released.values() for case in arm_cases})
    arms_driven = {case: {f"slot{index}": arm
                          for index, arm in enumerate(sorted(released))
                          if case in released[arm]} for case in cases}
    for leg in CUDA_LEGS:
        directory = results / name / leg
        directory.mkdir(parents=True)
        dispatching = leg not in recut.NON_DISPATCHING_LEGS
        artifact = {
            "release": {"released": True},
            "provenance": {
                "source_sha256": {recut._leg_key("cuda", key): _live(key)  # noqa: SLF001
                                  for key in recut._cuda_bound_files()},  # noqa: SLF001
                "device": {"compute_capability": CAPABILITY},
            },
            "arms_driven": arms_driven if dispatching else {},
            "arbitration": ({case: ["INCUMBENT-YIELDED", "ARBITRATION-DEFAULT-HELD"]
                             for case in cases} if dispatching else {}),
        }
        (directory / "gate.json").write_text(json.dumps(artifact))
        rows = [{"case": case, "legs": {"fused": {"plan": {
            "composition": {"tables_dispatched": ["cuda"]},
            "environment": {"device": {"compute_capability": CAPABILITY},
                            "device_certified_by_table": {"cuda": True}}}}}}
            for case in cases]
        (directory / "cases.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows))
    return name


def _ledger(path: pathlib.Path) -> None:
    """A CUDA ledger holding only a migrated driver_dispatch record: no runs, no status."""
    record = {
        "table": "cuda",
        "source_sha256": {key: _live(key) for key in recut._cuda_bound_files()},  # noqa: SLF001
        "code_sha256": {},
        "exclusions": {"released_fused_arms": {}},
        "runs": {},
    }
    path.write_text(json.dumps({"driver_dispatch": record}, indent=2, sort_keys=True)
                    + "\n")


@pytest.fixture
def tree(tmp_path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    ledger = tmp_path / "fingerprints.json"
    monkeypatch.setattr(recut, "RESULTS", results)
    monkeypatch.setitem(recut.BACKENDS, "cuda",
                        dict(recut.BACKENDS["cuda"], record=ledger))
    # WHICH TABLE LICENSES THE DEFAULT is read from the shipped ledgers' admissions;
    # pinned here to the precedence's first table so this file does not move with the
    # certification state of the tree it runs on. The CUDA cut then neither needs nor
    # takes a licence.
    monkeypatch.setattr(fastpath, "primary_table", lambda capability: "triton")
    return results, ledger


def _dispatch(ledger: pathlib.Path) -> dict:
    return json.loads(ledger.read_text())["driver_dispatch"]


def _cut(name: str) -> int:
    return recut.main(["--backend", "cuda", "--run", name, "--write"])


def test_a_cuda_cut_files_its_verdict_under_the_capability_and_none_beside_the_digests(
        tree):
    results, ledger = tree
    _ledger(ledger)
    assert _cut(_campaign(results)) == 0
    record = _dispatch(ledger)
    assert record["runs"][CAPABILITY]["status"] == "PASS"
    assert "status" not in record
    assert fastpath.live_capabilities(record) == (CAPABILITY,)


def test_the_record_a_cut_writes_is_one_the_cuda_contract_accepts(tree):
    """The contract's own rule over the contract's own measurements, so this file
    cannot pass on a reading of the rule the contract does not share. A second cut
    over the record the first wrote is accepted too: nothing the cut writes is a
    field its own shape check refuses."""
    results, ledger = tree
    _ledger(ledger)
    name = _campaign(results)
    assert _cut(name) == 0
    first = _dispatch(ledger)
    assert contract._dispatch_record_problems({"driver_dispatch": first}) == []  # noqa: SLF001
    assert _cut(name) == 0
    second = _dispatch(ledger)
    assert contract._dispatch_record_problems({"driver_dispatch": second}) == []  # noqa: SLF001
    assert "status" not in second


@pytest.mark.parametrize("backend", ["cuda", "triton", "metal"])
def test_every_cut_refuses_a_record_carrying_an_entry_level_status(
        backend, tmp_path, monkeypatch, capsys):
    """``status`` is a route-run field on every table (``fastpath.DISPATCH_RUN_FIELDS``,
    ``metal_runs.DISPATCH_RUN_FIELDS``), so a record carrying one beside its digests is
    refused, with nothing written, before any leg's artifact is read. The legs are only
    named here for that reason."""
    results = tmp_path / "results"
    ledger = tmp_path / "fingerprints.json"
    monkeypatch.setattr(recut, "RESULTS", results)
    monkeypatch.setitem(recut.BACKENDS, backend,
                        dict(recut.BACKENDS[backend], record=ledger))
    name = f"{recut.BACKENDS[backend]['run_prefix']}_a_campaign"
    for leg in recut.BACKENDS[backend]["required_legs"]:
        (results / name / leg).mkdir(parents=True)
        (results / name / leg / "gate.json").write_text("{}")
    record = {"table": backend, "source_sha256": {"meep_gpu/fastpath.py": "0" * 64},
              "exclusions": {"released_fused_arms": {}}, "runs": {}, "status": "PASS"}
    ledger.write_text(json.dumps({"driver_dispatch": record}))
    before = ledger.read_text()
    assert recut.main(["--backend", backend, "--run", name, "--write"]) == 2
    error = capsys.readouterr().err
    assert "record_carries_route_run_fields_at_entry_level" in error
    assert "'status'" in error
    assert ledger.read_text() == before
