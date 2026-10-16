"""Chunked certification: any set of units runs as a chunk, and each unit binds from its own runs.

``rounds/1.0/chunks.py`` lets a round's gate units run as chunks, each under its own one-shot
stamp, on any slot and at any time. These tests pin the properties that make that safe
(host-only: no GPU, no MEEP):

* TWO CHUNKS NEVER COLLIDE ON A NAME: the default partitions of both machines cover every unit,
  their chunk plans together run exactly the legs the unchunked round runs, every root a chunk
  writes is its own and no other chunk's root extends it by ``_``, at eleven chunks (``_c1``
  against ``_c10``), across added and re-partitioned chunks, across two devices of one round,
  and against a foreign root in ``results/``;
* A UNIT IS JUDGED BY ITS OWN RUNS: ``verify`` takes, per unit, the newest run whose import
  record holds on the release tree and whose verdicts pass, reads the units of a run from its
  launch record and never from the index's plan, so re-partitioning the tree's units into any
  chunk sets moves no verdict, and a unit run again alone replaces only its own evidence;
* A NEW DEVICE IS NEVER SILENT: an index names its device and mode, a launch on another device
  is refused by name, and a new device's bind re-points no board;
* THE BIND BINDS THE SELECTION: one call per selected stamp, with the keys selected there where
  the tool takes keys, and a run a tool would bind against the selection named on a STOP line;
  the Triton ``--only`` names only keys the Triton ledger holds, a selected unit whose key it
  lacks named on a REFUSED line, so every line the tool reads passes its ``--only`` guard (on
  a copy of the R27 round's own index and selection, whose bind named two such keys);
* A BIND REFUSES AN ARTIFACT WHOSE IMPORT RECORD NO LONGER MATCHES THE TREE: ``chunks.py
  verify``, the three rebind tools, ``repoint_board_bindings.py --stamps``, and every writer of
  a round record, through the one rule in ``artifact_drift.py``.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROUND = HERE / "rounds" / "1.0"
for path in (str(ROUND), str(HERE), str(REPO)):
    if path not in sys.path:
        sys.path.insert(0, path)

import artifact_drift  # noqa: E402
import chunks as C  # noqa: E402
import legs as L  # noqa: E402
import metal_round  # noqa: E402
import round as round_driver  # noqa: E402

ROUND_STAMP = "2099-03-04_10"
DEVICE = {"nvidia": "cc86", "metal": "applegpu_g13s"}
NEW_DEVICE = {"nvidia": "cc90", "metal": "applegpu_g15s"}


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _index(machine: str, n: int = None, units=None, device=None, mode="certified") -> dict:
    units = C.units_for(REPO, machine) if units is None else units
    bins = C.partition(units, n or C.DEFAULT_CHUNKS[machine])
    return C.make_index(machine, ROUND_STAMP, bins, {"rule": "test"},
                        device=device or DEVICE[machine], device_mode=mode)


# ------------------------------------------------------------------- partitions --------
@pytest.mark.parametrize("machine", C.MACHINES)
def test_the_default_partition_covers_every_unit_once_and_balances_the_walls(machine):
    units = C.units_for(REPO, machine)
    index = _index(machine)
    assert len(index["chunks"]) == C.DEFAULT_CHUNKS[machine]
    placed = [u for chunk in index["chunks"] for u in chunk["units"]]
    assert sorted(placed) == sorted(u["id"] for u in units) and len(set(placed)) == len(placed)
    walls = [chunk["expect_s"] for chunk in index["chunks"]]
    # The LPT bound: no chunk exceeds another by more than the largest unit.
    assert max(walls) - min(walls) <= max(int(u["expect_s"]) for u in units), walls
    assert sum(walls) == sum(int(u["expect_s"]) for u in units)
    assert index["next_k"] == len(index["chunks"])


def test_every_unit_names_its_scripts_and_both_policies_travel_together():
    units = C.nvidia_units(REPO)
    for unit in units:
        assert unit["scripts"], unit["id"]
        assert all((REPO / s).is_file() for s in unit["scripts"]), unit["id"]
        if unit["kind"] in ("triton_fleet", "triton_composition", "triton_stencil",
                            "triton_bitid", "cuda_ade"):
            assert unit["policies"] == ["flush", "keep"], unit["id"]
    cuda = {u["subdir"] for u in units if u["kind"] == "cuda_subdir"}
    assert cuda == {leg["subdir"] for leg in L.cuda_legs(REPO, REPO, "x") if leg["subdir"]}


def _key(step: dict) -> list:
    """A stamp-free key per leg a step runs, so chunked and unchunked plans compare."""
    kind = step["kind"]
    if kind == "gate":
        args = list(step["args"])
        cut = args.index("--out")
        return [("gate", step["id"], tuple(args[:cut] + args[cut + 2:]), step["leg"])]
    if kind == "fleet":
        return [("fleet", step["root"], step["policy"], bool(step.get("manifest")), g)
                for g in step["gates"]]
    if kind == "stencil":
        return [("stencil", step["root"], step["family"], step["policy"])]
    if kind == "bitid":
        return [("bitid", step["root"], step["policy"], step["experiments"])]
    if kind == "hd":
        return [("hd", step["root"], step["gate"], step["policy"])]
    raise AssertionError(kind)


def test_the_chunk_plans_together_run_exactly_the_legs_of_the_unchunked_round(tmp_path):
    results = tmp_path / "results"
    whole, _summary = L.plan_lanes(REPO, results, "S", gpus=4, tables=("cuda", "triton"))
    want = sorted(k for lane in whole for step in lane for k in _key(step))
    got = []
    index = _index("nvidia")
    for chunk in index["chunks"]:
        plan, summary = C.nvidia_plan(REPO, results, chunk, 1)
        assert len(plan) == 1 and summary["wall_s"] == sum(
            int(s["expect_s"]) for s in plan[0])
        got += [k for step in plan[0] for k in _key(step)]
    assert sorted(got) == want


# ------------------------------------------------------------------ the round stamp ------
def test_the_round_is_read_from_the_route_stamp_registry():
    registry = C.load_registry(REPO)
    assert registry is not None, f"the checkout has no {C.REGISTRY}"
    for machine, tables in C.REGISTRY_TABLES.items():
        stamps = {registry.stamp(t) for t in tables}
        stamp, where = C.registry_round_stamp(REPO, machine)
        if stamps == {registry.PLACEHOLDER}:
            assert stamp is None and "the freeze sets it" in where
        else:
            assert stamp == stamps.pop() if len(stamps) == 1 else stamp is None
        round_stamp = stamp or ROUND_STAMP
        index = C.make_index(machine, round_stamp, [], {"rule": "test"}, device=DEVICE[machine])
        assert [p for p in C.round_problems(REPO, index) if "already run" not in p] == []


def _registry(root: Path, round_stamp: str, spent=()) -> None:
    (root / "meep_gpu").mkdir(parents=True, exist_ok=True)
    (root / C.REGISTRY).write_text(
        "PLACEHOLDER = 'UNSET'\n"
        f"ROUND = {round_stamp!r}\n"
        "REROUTED = {}\n"
        "ROUTE_PREFIXES = {'triton': 'dispatch_fused_route_', "
        "'cuda': 'dispatch_fused_route_cuda_', 'metal': 'dispatch_metal_route_'}\n"
        f"SPENT = {{'triton': {tuple(spent)!r}, 'cuda': (), 'metal': ()}}\n"
        "def stamp(table):\n    return REROUTED.get(table, ROUND)\n")


def test_the_round_fits_the_registry_or_is_refused_by_name(tmp_path):
    index = C.make_index("nvidia", ROUND_STAMP, [], {"rule": "test"}, device="cc86")
    assert "no route-stamp registry" in C.round_problems(tmp_path, index)[0]
    _registry(tmp_path, "UNSET")
    assert C.round_problems(tmp_path, index) == []                 # before the freeze
    assert C.registry_round_stamp(tmp_path, "nvidia")[0] is None
    _registry(tmp_path, ROUND_STAMP)
    assert C.round_problems(tmp_path, index) == []
    assert C.registry_round_stamp(tmp_path, "nvidia")[0] == ROUND_STAMP
    _registry(tmp_path, "2099-03-05_10")
    assert "names 2099-03-05_10 for the triton table" in C.round_problems(tmp_path, index)[0]
    # One route campaign per stamp AND DEVICE: a spent cc86 campaign stops a cc86 index of
    # the round and leaves a cc90 index of it free.
    _registry(tmp_path, "UNSET", spent=(f"dispatch_fused_route_{ROUND_STAMP}_cc86",))
    assert "has already run" in C.round_problems(tmp_path, index)[0]
    new = C.make_index("nvidia", ROUND_STAMP, [], {"rule": "test"}, device="cc90",
                       device_mode="new")
    assert C.round_problems(tmp_path, new) == []


# ------------------------------------------------------------------- collisions --------
def _assert_disjoint(*indices: dict) -> None:
    named = {}
    for index in indices:
        for chunk in C.allocated(index):
            roots = C.chunk_roots(chunk, index["machine"])
            assert C.launch_root(chunk["stamp"]) in roots
            assert chunk["stamp"] not in named, chunk["stamp"]
            named[chunk["stamp"]] = set(roots)
    for a, roots_a in named.items():
        for b, roots_b in named.items():
            if a == b:
                continue
            assert not roots_a & roots_b, (a, b, sorted(roots_a & roots_b)[:3])
            assert not [(x, y) for x in roots_a for y in roots_b if y.startswith(x + "_")], (a, b)


@pytest.mark.parametrize("machine", C.MACHINES)
def test_two_chunks_never_collide_on_a_name(machine):
    index = _index(machine)
    assert C.collisions(index, repo=REPO) == []
    _assert_disjoint(index)


@pytest.mark.parametrize("machine", C.MACHINES)
def test_two_devices_of_one_round_never_collide_on_a_name(machine):
    """The H200 and the A6000 (an M3 and an M1 Max) certify the same round side by side: the
    device in every stamp keeps each one's roots its own, and neither extends the other's."""
    certified = _index(machine)
    new = _index(machine, device=NEW_DEVICE[machine], mode="new")
    assert all(f"_{NEW_DEVICE[machine]}_" in c["stamp"] for c in new["chunks"])
    assert C.collisions(new, repo=REPO) == []
    _assert_disjoint(certified, new)
    names = [(n, s) for index in (certified, new) for n, s in C._roots_of_index(index)]
    assert C.name_collisions(names) == []


def test_a_chunks_roots_are_exactly_the_roots_its_plan_writes(tmp_path):
    results = tmp_path / "results"
    for chunk in _index("nvidia")["chunks"]:
        plan, _summary = C.nvidia_plan(REPO, results, chunk, 2)
        subs = C.stamp_subs(results, chunk["stamp"])
        written = {Path(C._sub(s, subs).get("artifact") or C._sub(s, subs)["root"])
                   .relative_to(results).parts[0] for lane in plan for s in lane}
        assert written | {C.launch_root(chunk["stamp"])} == set(C.chunk_roots(chunk, "nvidia"))
        assert all(chunk["stamp"] in name for name in written)


def test_eleven_chunks_and_added_chunks_collide_with_nothing(tmp_path):
    index = _index("nvidia", 11)
    assert {c["id"] for c in index["chunks"]} >= {"c1", "c10"}
    assert C.collisions(index, repo=REPO) == []
    path = tmp_path / "index.json"
    C.write_index(path, index)
    assert C.main(["relaunch", "--index", str(path), "--chunk", "c1", "--why", "a test"]) == 0
    unit = index["chunks"][2]["units"][0]
    assert C.main(["add", "--index", str(path), "--repo", str(REPO), "--units", unit,
                   "--why", "one unit again, alone"]) == 0
    again = C.read_index(path)
    c11, c12 = C.find_chunk(again, "c11"), C.find_chunk(again, "c12")
    assert c11["stamp"] == f"{ROUND_STAMP}_cc86_c11"
    assert c11["units"] == C.find_chunk(again, "c1")["units"] and c12["units"] == [unit]
    assert again["next_k"] == 13
    assert C.collisions(again, repo=REPO) == []
    _assert_disjoint(again)
    assert not Path(str(path) + ".lock").exists()


def test_a_shared_stamp_a_reused_id_a_missing_unit_and_a_bad_stamp_are_refused():
    index = _index("metal")
    shared = copy.deepcopy(index)
    shared["chunks"][1]["units"].append(shared["chunks"][0]["units"][0])
    assert C.collisions(shared, repo=REPO) == []          # a unit in two chunks: two runs
    same = copy.deepcopy(index)
    same["chunks"][1]["stamp"] = same["chunks"][0]["stamp"]
    problems = C.collisions(same)
    assert any(f"not '{ROUND_STAMP}_applegpu_g13s_m1'" in p for p in problems), problems
    reused = copy.deepcopy(index)
    reused["chunks"][2]["id"] = "m1"
    reused["chunks"][2]["stamp"] = reused["chunks"][1]["stamp"]
    assert any("chunk id m1 is current and current" in p for p in C.collisions(reused))
    beyond = copy.deepcopy(index)
    beyond["next_k"] = 2
    assert any("not below next_k" in p for p in C.collisions(beyond))
    gone = copy.deepcopy(index)
    gone["chunks"][2]["units"].pop()
    assert any("in no chunk" in p for p in C.collisions(gone, repo=REPO))
    prefixed = copy.deepcopy(index)
    prefixed["chunks"][1]["stamp"] = f"{ROUND_STAMP}_applegpu_g13s_m0_r2"
    assert any(f"not '{ROUND_STAMP}_applegpu_g13s_m1'" in p for p in C.collisions(prefixed))
    twice = copy.deepcopy(index)
    twice["chunks"][0]["units"].append(twice["chunks"][0]["units"][0])
    assert any("names a unit twice" in p for p in C.collisions(twice))


def test_an_index_of_the_first_chunk_tooling_is_refused_by_name(tmp_path):
    old = {"schema": "meep_gpu.cert_chunks/1", "machine": "metal", "round": ROUND_STAMP,
           "chunks": []}
    path = tmp_path / "index.json"
    path.write_text(json.dumps(old))
    with pytest.raises(SystemExit, match="retired_index_schema"):
        C.read_index(path)
    bad = dict(_index("nvidia"), device="sm_90")
    assert any("not a nvidia device" in p for p in C.collisions(bad))


def test_the_prefix_case_is_a_collision():
    assert C.name_collisions([("triton_fleet_R_c1", "R_c1"),
                              ("triton_fleet_R_c1_flush", "R_c1")]) == []
    problems = C.name_collisions([("triton_fleet_R_c1", "R_c1"),
                                  ("triton_fleet_R_c1_x", "R_c1_x")])
    assert len(problems) == 1 and "starts with root triton_fleet_R_c1" in problems[0]
    assert C.name_collisions([("triton_fleet_R_c1", "R_c1"),
                              ("triton_fleet_R_c10", "R_c10")]) == []


def test_a_foreign_root_a_bind_would_read_as_a_chunks_is_refused_either_way(tmp_path):
    index = _index("nvidia")
    chunk = index["chunks"][0]
    root = sorted(r for r in C.chunk_roots(chunk, "nvidia") if r.startswith("triton_fleet_"))[0]
    results = tmp_path / "results"
    (results / f"{root}_r2").mkdir(parents=True)          # a resume continuation: its own
    assert C.collisions(index, results=results) == []
    (results / f"{root}_other").mkdir()                   # another run under its name
    problems = C.collisions(index, results=results)
    assert len(problems) == 1 and f"{root}_other" in problems[0]
    shutil.rmtree(results / f"{root}_other")
    # The other way round: a run of the round's bare stamp, which a chunk root extends.
    (results / f"triton_fleet_{ROUND_STAMP}").mkdir()
    problems = C.collisions(index, results=results)
    assert problems and all(f"triton_fleet_{ROUND_STAMP} followed by '_'" in p
                            for p in problems), problems


# -------------------------------------------------------------------- readiness --------
#: Two Metal gates that import no shared kit (each one's own script is outside the other's
#: closure, which the tests check rather than assume).
GATE_A, GATE_B = "gate_metal_spine_byte_neutrality", "gate_metal_whole_step"


def _metal_pair_index() -> dict:
    units = {u["id"]: u for u in C.metal_units(REPO)}
    a, b = units[f"metal/{GATE_A}"], units[f"metal/{GATE_B}"]
    return C.make_index("metal", ROUND_STAMP, [[a], [b]], {"rule": "test"},
                        device="applegpu_g13s")


PRIOR = {"_stamp": "2099-01-01_10", "_machine": "metal"}


def _closures(overrides=None):
    ic = C.load_import_closure(REPO)
    resolver = ic.Resolver(REPO, overrides=overrides)
    pins = C.ledger_pins(REPO, ic)
    units = {u["id"]: u for u in C.metal_units(REPO)}
    return [set(C.unit_closure(units[f"metal/{g}"], resolver, pins)) for g in (GATE_A, GATE_B)]


def test_a_chunk_whose_closure_holds_a_changed_file_is_not_ready():
    index = _metal_pair_index()
    gate = f"parity/meep_gpu/{GATE_A}.py"
    closure_a, closure_b = _closures()
    assert gate in closure_a and gate not in closure_b
    report = C.readiness(REPO, index, [gate], PRIOR, say=lambda _line: None)
    assert report["chunks"]["m0"]["state"] == "NOT READY"
    assert report["chunks"]["m1"]["state"] == "READY"
    assert report["ready"] == ["m1"]
    assert any(f"run closure of {gate}" in why for why in report["chunks"]["m0"]["hits"][gate])


def test_a_chunk_whose_closure_gained_a_changed_file_is_reported_not_ready():
    """The planted control: an import of the changed gate added to the OTHER chunk's gate."""
    index = _metal_pair_index()
    gate = f"parity/meep_gpu/{GATE_A}.py"
    other = f"parity/meep_gpu/{GATE_B}.py"
    planted = {other: (REPO / other).read_text(encoding="utf-8") + f"\nimport {GATE_A}\n"}
    assert gate in _closures(planted)[1]
    report = C.readiness(REPO, index, [gate], PRIOR, say=lambda _line: None,
                         overrides=planted)
    assert report["chunks"]["m1"]["state"] == "NOT READY"
    assert any(f"run closure of {other}" in why for why in report["chunks"]["m1"]["hits"][gate])
    assert report["ready"] == []


def test_a_file_only_the_previous_rounds_record_names_still_holds_a_chunk_back():
    index = _metal_pair_index()
    planted = "meep_gpu/only_in_a_record.py"
    prior = dict(PRIOR, **{f"metal/{GATE_B}": {"files": [planted], "artifacts": 1}})
    report = C.readiness(REPO, index, [planted], prior, say=lambda _line: None)
    assert report["chunks"]["m1"]["state"] == "NOT READY"
    assert report["chunks"]["m1"]["hits"][planted] == [
        f"metal/{GATE_B}: recorded by the previous round's artifact"]
    assert report["chunks"]["m0"]["state"] == "READY"


def test_readiness_refuses_the_static_closure_alone_while_it_misses_lazy_reaches():
    ic = C.load_import_closure(REPO)
    gap = C.lazy_gap_open(ic.Resolver(REPO))
    index = _metal_pair_index()
    if gap:
        with pytest.raises(SystemExit, match="_LAZY"):
            C.readiness(REPO, index, [], None, say=lambda _line: None)
    else:
        assert C.readiness(REPO, index, [], None, say=lambda _line: None)["lazy_gap_open"] is False


def test_changing_patterns_match_exact_paths_directories_and_globs():
    assert C.matches("meep_gpu/sources.py", ["meep_gpu/sources.py"])
    assert C.matches("tools/ci/x.py", ["tools/ci/"])
    assert C.matches("meep_gpu/cuda_kernels/deposit_repair.py", ["meep_gpu/*_kernels/*deposit*"])
    assert not C.matches("meep_gpu/sources_old.py", ["meep_gpu/sources.py", "tools/"])


# --------------------------------------------------------- the rule and verify ---------
def test_the_drift_rule_reads_both_record_maps_and_skips_non_release_keys(tmp_path):
    (tmp_path / "meep_gpu").mkdir()
    live = tmp_path / "meep_gpu" / "a.py"
    live.write_text("A = 1\n")
    payload = {"imported_source_sha256": {"meep_gpu/a.py": _sha(live),
                                          "parity/meep_gpu/results/x/y.py": "0" * 64,
                                          "elsewhere/z.py": "0" * 64,
                                          "meep_gpu/b.py": "not a digest"},
               "source_sha256": {"meep_gpu/gone.py": "1" * 64}}
    assert set(artifact_drift.records(payload)["imported_source_sha256"]) == {"meep_gpu/a.py"}
    assert artifact_drift.artifact_moved(payload, tmp_path) == ["meep_gpu/gone.py"]
    live.write_text("A = 2\n")
    assert artifact_drift.artifact_moved(payload, tmp_path) == ["meep_gpu/a.py",
                                                                 "meep_gpu/gone.py"]
    text = artifact_drift.refusal("x/gate.json", ["meep_gpu/a.py", "meep_gpu/gone.py"], tmp_path)
    assert "meep_gpu/a.py (other bytes)" in text and "meep_gpu/gone.py (absent)" in text


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "user.email=fixture",
                           "-c", "user.name=t", "-c", "commit.gpgsign=false", *args],
                          capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture()
def release(tmp_path):
    """A release tree holding one Metal gate and one kernel module, a git repository, the
    tip's closure reader, and a results tree outside it."""
    repo = tmp_path / "repo"
    harness = repo / "parity" / "meep_gpu"
    harness.mkdir(parents=True)
    (repo / "meep_gpu").mkdir()
    (repo / "meep_gpu" / "__init__.py").write_text('_SUBMODULES = frozenset(("kernel",))\n')
    (repo / "meep_gpu" / "kernel.py").write_text("K = 1\n")
    (harness / "gate_metal_fake.py").write_text("from meep_gpu import kernel\n")
    shutil.copy(HERE / "import_closure.py", harness / "import_closure.py")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "tip")
    unit = {u["id"]: u for u in C.metal_units(repo)}["metal/gate_metal_fake"]
    index = C.make_index("metal", ROUND_STAMP, [[unit]], {"rule": "test"},
                         device="applegpu_g13s")
    results = tmp_path / "results"
    artifact = results / f"metal_fake_{index['chunks'][0]['stamp']}" / "gate.json"
    artifact.parent.mkdir(parents=True)
    _environment(results, index["chunks"][0]["stamp"])
    return repo, results, index, artifact


def _environment(results: Path, stamp: str, architecture: str = "applegpu_g13s") -> None:
    """The environment record a Metal launch writes (``metal_environment_<stamp>``)."""
    directory = results / f"metal_environment_{stamp}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "start.json").write_text(json.dumps({"architecture": architecture}))


UID = "metal/gate_metal_fake"


def _released(record: dict) -> str:
    return json.dumps({"release": {"released": True}, **record})


def test_verify_refuses_an_artifact_whose_import_record_no_longer_matches_the_tree(release):
    repo, results, index, artifact = release
    kernel = repo / "meep_gpu" / "kernel.py"
    artifact.write_text(_released({"imported_source_sha256": {
        "meep_gpu/kernel.py": _sha(kernel),
        "parity/meep_gpu/gate_metal_fake.py": _sha(repo / "parity/meep_gpu/gate_metal_fake.py")}}))
    say = []
    assert C.verify(repo, results, index, say=say.append)["units"][UID]["state"] == \
        "NOT LAUNCHED"
    assert C.write_launch_record(results, index, index["chunks"][0], repo) is None
    assert C.verify(repo, results, index, say=say.append)["ok"]
    kernel.write_text("K = 2\n")
    report = C.verify(repo, results, index, say=say.append)
    row = report["units"][UID]
    assert not report["ok"] and row["state"] == "REFUSED"
    assert "meep_gpu/kernel.py (other bytes)" in row["why"]
    kernel.unlink()
    assert "meep_gpu/kernel.py (absent)" in C.verify(repo, results, index, say=say.append)[
        "units"][UID]["why"]


def test_an_artifact_with_no_record_is_judged_by_its_launch_tip(release):
    repo, results, index, artifact = release
    chunk = index["chunks"][0]
    artifact.write_text(_released({}))
    say = []
    assert C.write_launch_record(results, index, chunk, repo) is None
    assert C.write_launch_record(results, index, chunk, repo) is None        # a resume
    assert C.verify(repo, results, index, say=say.append)["ok"]
    (repo / "meep_gpu" / "kernel.py").write_text("K = 2\n")
    _git(repo, "commit", "-qam", "a later landing moves the kernel")
    row = C.verify(repo, results, index, say=say.append)["units"][UID]
    assert row["state"] == "REFUSED" and "meep_gpu/kernel.py" in row["why"]
    assert "relaunch or add" in C.write_launch_record(results, index, chunk, repo)
    # A curated record alone (a probe's source_sha256) holds, yet the closure moved: refused.
    gate = "parity/meep_gpu/gate_metal_fake.py"
    artifact.write_text(_released({"source_sha256": {gate: _sha(repo / gate)}}))
    row = C.verify(repo, results, index, say=say.append)["units"][UID]
    assert row["state"] == "REFUSED" and "curated source_sha256 only" in row["why"]
    assert "meep_gpu/kernel.py" in row["why"]


def test_a_unit_run_again_alone_binds_from_its_newest_run_that_holds(release, tmp_path):
    """The unit's first run went stale; it runs again alone in a new chunk; the first chunk's
    launch record names it, and nothing about that refuses anything: the new run is chosen."""
    repo, results, index, artifact = release
    kernel = repo / "meep_gpu" / "kernel.py"
    record = {"imported_source_sha256": {"meep_gpu/kernel.py": _sha(kernel)}}
    artifact.write_text(_released(record))
    assert C.write_launch_record(results, index, index["chunks"][0], repo) is None
    kernel.write_text("K = 3\n")
    _git(repo, "commit", "-qam", "a landing moves the kernel")
    path = tmp_path / "index.json"
    C.write_index(path, index)
    assert C.main(["add", "--index", str(path), "--repo", str(repo), "--units", UID,
                   "--why", "the kernel moved"]) == 0
    index = C.read_index(path)
    again = C.find_chunk(index, "m1")
    second = results / f"metal_fake_{again['stamp']}" / "gate.json"
    second.parent.mkdir(parents=True)
    _environment(results, again["stamp"])
    second.write_text(_released({"imported_source_sha256": {"meep_gpu/kernel.py": _sha(kernel)}}))
    assert C.write_launch_record(results, index, again, repo) is None
    report = C.verify(repo, results, index, say=lambda _l: None)
    row = report["units"][UID]
    assert report["ok"] and row["selected"] == again["stamp"]
    assert [c["state"] for c in row["candidates"]] == ["REFUSED", "HOLDS"]
    # A newer run that did not release leaves the older one that holds chosen.
    kernel.write_text("K = 1\n")                  # the first run holds again
    second.write_text(json.dumps({"release": {"released": False},
                                  "imported_source_sha256": {"meep_gpu/kernel.py": _sha(kernel)}}))
    row = C.verify(repo, results, index, say=lambda _l: None)["units"][UID]
    assert row["selected"] == index["chunks"][0]["stamp"]
    assert row["candidates"][1]["state"] == "NOT RELEASED"
    # Two runs that both hold: the newer launch is chosen, and both stay candidates.
    second.write_text(_released({"imported_source_sha256": {"meep_gpu/kernel.py": _sha(kernel)}}))
    row = C.verify(repo, results, index, say=lambda _l: None)["units"][UID]
    assert [c["state"] for c in row["candidates"]] == ["HOLDS", "HOLDS"]
    assert row["selected"] == again["stamp"]


def test_a_launch_under_a_retired_chunks_stamp_is_still_a_run_of_its_units(release):
    """``repartition`` retires only chunks with no launch record, but a launcher reads the
    index once and takes no index lock, so its launch record can land after a re-partition
    retired its chunk: that run still counts, so a re-partition never removes a run."""
    repo, results, index, artifact = release
    artifact.write_text(_released({"imported_source_sha256": {
        "meep_gpu/kernel.py": _sha(repo / "meep_gpu" / "kernel.py")}}))
    chunk = index["chunks"][0]
    assert C.write_launch_record(results, index, chunk, repo) is None
    index["chunks"].remove(chunk)
    index["retired"].append(dict(chunk, retired_why="a re-partition"))
    report = C.verify(repo, results, index, say=lambda _l: None)
    assert report["ok"] and report["units"][UID]["selected"] == chunk["stamp"]


def test_a_launch_on_another_device_is_refused_by_name(release):
    repo, results, index, artifact = release
    artifact.write_text(_released({"imported_source_sha256": {
        "meep_gpu/kernel.py": _sha(repo / "meep_gpu" / "kernel.py")}}))
    stamp = index["chunks"][0]["stamp"]
    assert C.write_launch_record(results, index, index["chunks"][0], repo) is None
    environment = results / f"metal_environment_{stamp}"
    (environment / "start.json").write_text(json.dumps({"architecture": "applegpu_g15s"}))
    row = C.verify(repo, results, index, say=lambda _l: None)["units"][UID]
    assert row["state"] == "REFUSED" and "ran on ['applegpu_g15s']" in row["why"]
    (environment / "start.json").write_text(json.dumps({"architecture": "applegpu_g13s"}))
    assert C.verify(repo, results, index, say=lambda _l: None)["ok"]
    # A launch record of another device, under this index's stamp, is not this chunk's.
    launch = results / C.launch_root(stamp) / "LAUNCH.json"
    launch.write_text(json.dumps(dict(json.loads(launch.read_text()), device="applegpu_g15s")))
    row = C.verify(repo, results, index, say=lambda _l: None)["units"][UID]
    assert row["state"] == "REFUSED" and "names device 'applegpu_g15s'" in row["why"]


def test_a_run_whose_device_cannot_be_read_is_never_taken(release, tmp_path):
    """The device is read, never assumed from a stamp's name: a Metal launch with no
    environment record, and an NVIDIA launch none of whose artifacts stamps a device, are
    refused; an NVIDIA unit that stamps none is judged by the rest of its own launch."""
    repo, results, index, artifact = release
    artifact.write_text(_released({"imported_source_sha256": {
        "meep_gpu/kernel.py": _sha(repo / "meep_gpu" / "kernel.py")}}))
    stamp = index["chunks"][0]["stamp"]
    assert C.write_launch_record(results, index, index["chunks"][0], repo) is None
    shutil.rmtree(results / f"metal_environment_{stamp}")
    row = C.verify(repo, results, index, say=lambda _l: None)["units"][UID]
    assert row["state"] == "REFUSED" and "cannot be read" in row["why"]
    assert row["candidates"][0]["tool"] is False         # rebind_metal_welds.py refuses it
    _environment(results, stamp)
    assert C.verify(repo, results, index, say=lambda _l: None)["ok"]
    # NVIDIA: two units of one launch, one stamping its capability and one stamping none.
    units = {u["id"]: u for u in C.units_for(REPO, "nvidia")}
    pair = ["triton/fleet/source_seams", "triton/bit_identity"]
    nvidia = _index("nvidia", units=[units[u] for u in pair], n=1)
    out = tmp_path / "nvidia_results"
    chunk = nvidia["chunks"][0]
    _plant(units[pair[0]], chunk["stamp"], out, holds=True, capability=None)
    _launch(out, nvidia, chunk, pair, "2099-03-04T01:00:00Z")

    def judged(uid):
        return C.verify(REPO, out, nvidia, say=lambda _l: None)["units"][uid]

    row = judged(pair[0])
    assert row["state"] == "REFUSED" and "no artifact of launch" in row["why"]
    _plant(units[pair[1]], chunk["stamp"], out, holds=True, capability="9.0")
    row = judged(pair[0])
    assert row["state"] == "REFUSED" and "ran on ['cc90']" in row["why"]
    _plant(units[pair[1]], chunk["stamp"], out, holds=True, capability="8.6")
    row = judged(pair[0])
    assert row["state"] == "OK", row["why"]
    assert row["candidates"][0]["device_from"].startswith("the other artifacts of launch")


def test_a_triton_unit_alone_is_judged_by_its_campaigns_device_record(tmp_path):
    """A Triton unit whose artifacts stamp no device (``triton/fleet/source_seams``) runs
    again alone: the campaign root's ``device.json`` (``drive_triton_weld_gates.py`` writes
    it, the Triton rebinds read it) names its card, so the run is judged by its own record,
    and a record naming another card, or one that cannot be read, refuses it."""
    units = {u["id"]: u for u in C.units_for(REPO, "nvidia")}
    uid = "triton/fleet/source_seams"
    index = _index("nvidia", units=[units[uid]], n=1)
    results = tmp_path / "results"
    chunk = index["chunks"][0]
    _plant(units[uid], chunk["stamp"], results, holds=True, capability=None)
    _launch(results, index, chunk, [uid], "2099-03-04T01:00:00Z")

    def judged():
        return C.verify(REPO, results, index, say=lambda _l: None)["units"][uid]

    row = judged()
    assert row["state"] == "REFUSED" and "no artifact of launch" in row["why"]
    roots = sorted({d.parent for d in C.unit_dirs(units[uid], chunk["stamp"], results)})
    assert len(roots) == 2                                  # the keep and flush roots
    for root in roots:
        (root / C.TRITON_DEVICE_RECORD).write_text(json.dumps({"compute_capability": "8.6"}))
    row = judged()
    assert row["state"] == "OK", row["why"]
    assert row["candidates"][0]["device_from"] == "the campaign's device.json"
    (roots[0] / C.TRITON_DEVICE_RECORD).write_text(json.dumps({"compute_capability": "9.0"}))
    row = judged()
    assert row["state"] == "REFUSED" and "ran on ['cc90']" in row["why"]
    (roots[0] / C.TRITON_DEVICE_RECORD).write_text("{")
    row = judged()
    assert row["state"] == "REFUSED" and "unreadable:" in row["why"]
    # A CUDA unit never reads one: the CUDA rebind reads each artifact's own record only.
    cuda = next(u for u in units.values() if u["kind"] == "cuda_subdir")
    campaign = results / f"cuda_regate_{chunk['stamp']}_g0"
    (campaign / cuda["subdir"]).mkdir(parents=True)
    (campaign / C.TRITON_DEVICE_RECORD).write_text(json.dumps({"compute_capability": "9.0"}))
    assert C.campaign_devices(cuda, chunk["stamp"], results) == []


# ------------------------------------------------ re-partitioning moves no verdict ------
#: A file every planted record names, at its live bytes or at other bytes.
RECORDED = "meep_gpu/route_stamps.py"


def _plant(unit: dict, stamp: str, results: Path, holds: bool, released: bool = True,
           capability: str = "8.6") -> None:
    """The unit's artifacts under ``stamp``, laid out as the round writes them: a released
    gate verdict at every leg it binds, recording ``RECORDED`` at its live bytes or not, and,
    on NVIDIA, the compute capability it ran on (``capability=None``: none stamped)."""
    digest = _sha(REPO / RECORDED) if holds else "0" * 64
    record = {"canonical_verdict": {"released": released},
              "imported_source_sha256": {RECORDED: digest}}
    if capability and unit["kind"] != "metal_gate":
        record["stamped_device"] = {"compute_capability": capability}
    payload = json.dumps(record)
    for _leg, path in C.unit_verdict_paths(unit, stamp, results):
        target = path if path.suffix == ".json" else path / "gate.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload)


def _launch(results: Path, index: dict, chunk: dict, units, utc: str,
            environment: bool = True) -> None:
    """The chunk's launch record and, on Metal, the environment record its fleet writes."""
    if index["machine"] == "metal" and environment:
        _environment(results, chunk["stamp"], index["device"])
    path = results / C.launch_root(chunk["stamp"]) / "LAUNCH.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": C.LAUNCH_SCHEMA, "machine": index["machine"],
                                "device": index["device"], "round": index["round"],
                                "chunk": chunk["id"], "stamp": chunk["stamp"],
                                "units": sorted(units), "tip": "0" * 40, "tree": "0" * 40,
                                "utc": utc}))


def _summary(report: dict) -> tuple:
    return (report["counts"], {uid: (row["state"], row["selected"])
                               for uid, row in report["units"].items()})


@pytest.mark.parametrize("machine", C.MACHINES)
def test_re_partitioning_the_trees_units_into_any_chunk_sets_moves_no_verdict(machine, tmp_path):
    """Every unit of the tree (62 Metal and 123 NVIDIA at the 1.0 freeze) is run in the
    default partition; a few of those runs went stale and ran again alone in a later chunk,
    one did not, and one unit never launched. Then the index is re-partitioned at random,
    five times: its plan's unit lists reshuffled across the launched chunks, new chunks
    added, unlaunched ones retired. ``verify`` holds the same N of D, unit by unit, every
    time, because it reads the units of a run from the run's launch record."""
    units = {u["id"]: u for u in C.units_for(REPO, machine)}
    results = tmp_path / "results"
    index = _index(machine)
    rng = random.Random(1000 + len(units))
    stale = set(rng.sample(sorted(units), 6))
    rerun = set(sorted(stale)[:4])
    never = sorted(set(units) - stale)[0]
    # Two units whose artifacts stamp no device, as two Triton fleet gates do: each is
    # judged by what the other artifacts of its own launch stamp.
    mute = set(sorted(set(units) - stale - {never})[-2:])
    for k, chunk in enumerate(index["chunks"]):
        ran = [u for u in chunk["units"] if u != never]
        for uid in ran:
            _plant(units[uid], chunk["stamp"], results, holds=uid not in stale,
                   capability=None if uid in mute else "8.6")
        _launch(results, index, chunk, ran, f"2099-03-04T0{k}:00:00Z")
    again = C.new_chunk(index, [units[u] for u in sorted(rerun)], "the stale ones again")
    for uid in rerun:
        _plant(units[uid], again["stamp"], results, holds=True)
    _launch(results, index, again, rerun, "2099-03-04T09:00:00Z")
    first = C.verify(REPO, results, index, say=lambda _l: None)
    d = len(units)
    assert first["counts"] == {"OK": d - 3, "REFUSED": 2, "ABSENT": 0, "NOT LAUNCHED": 1}
    for uid in mute:
        chosen = first["units"][uid]["candidates"][-1]
        assert chosen["devices"] == [DEVICE[machine]], chosen
        if machine == "nvidia":
            assert chosen["device_from"].startswith("the other artifacts of launch")
    assert {uid for uid in rerun if first["units"][uid]["selected"] == again["stamp"]} == rerun
    assert first["units"][never]["state"] == "NOT LAUNCHED"
    for seed in range(5):
        shuffled = copy.deepcopy(index)
        pick = random.Random(seed)
        pool = sorted(units)
        pick.shuffle(pool)
        cuts = sorted(pick.sample(range(1, len(pool)), len(shuffled["chunks"]) - 1))
        for chunk, lo, hi in zip(shuffled["chunks"], [0] + cuts, cuts + [len(pool)]):
            chunk["units"] = sorted(pool[lo:hi])
        for _ in range(pick.randint(1, 3)):
            C.new_chunk(shuffled, [units[u] for u in pick.sample(pool, pick.randint(1, 9))],
                        "a re-partition")
        retired = shuffled["chunks"].pop()
        shuffled["retired"].append(dict(retired, retired_why="a re-partition"))
        assert C.collisions(shuffled) == []
        report = C.verify(REPO, results, shuffled, say=lambda _l: None)
        assert _summary(report) == _summary(first), seed


def test_the_first_toolings_failure_is_gone_a_launch_whose_units_the_plan_moved_still_holds(
        release, tmp_path):
    """FIX_REVIEW measured it: nine units moved into a chunk of their own made the chunks they
    left refuse their whole launch ("names other units"). The plan no longer judges a run."""
    repo, results, index, artifact = release
    artifact.write_text(_released({"imported_source_sha256": {
        "meep_gpu/kernel.py": _sha(repo / "meep_gpu" / "kernel.py")}}))
    assert C.write_launch_record(results, index, index["chunks"][0], repo) is None
    moved = copy.deepcopy(index)
    moved["chunks"][0]["units"] = []
    C.new_chunk(moved, [{"id": UID, "expect_s": 1}], "moved into a chunk of its own")
    assert C.verify(repo, results, moved, say=lambda _l: None)["ok"]


# ---------------------------------------------------------------- the multi-stamp tools ---
ENVIRONMENT = {"architecture": "applegpu_g13s", "torch": "2.10.0",
               "metal_frontend": "metalfe-32023.850.10"}
KEPT = "meep_gpu/kept.py"
OTHER = "meep_gpu/other.py"
S0, S1 = f"{ROUND_STAMP}_applegpu_g13s_m0", f"{ROUND_STAMP}_applegpu_g13s_m1"


@pytest.fixture()
def metal_rebind(tmp_path, monkeypatch):
    import metal_environment  # noqa: PLC0415
    import rebind_metal_welds as R  # noqa: PLC0415
    from meep_gpu import metal_runs  # noqa: PLC0415
    from meep_gpu.code_identity import code_digest_of_path  # noqa: PLC0415

    api = tmp_path / "api"
    (api / "meep_gpu").mkdir(parents=True)
    for name in (KEPT, OTHER):
        (api / name).write_text(f"# {name}\n", encoding="utf-8")
    results = api / "parity" / "meep_gpu" / "results"
    environment = dict(ENVIRONMENT, **{metal_environment.FAST_MATH: None})
    for stamp in (S0, S1):
        directory = results / f"metal_environment_{stamp}"
        directory.mkdir(parents=True)
        for name in ("start", "end"):
            (directory / f"{name}.json").write_text(json.dumps(environment))
    ledger = api / "meep_gpu" / "fingerprints.json"
    monkeypatch.setattr(R, "_API", api)
    monkeypatch.setattr(R, "LEDGER", ledger)
    monkeypatch.setattr(R, "RESULTS", results)
    monkeypatch.setattr(metal_environment, "RESULTS", results)

    def entry() -> dict:
        made = {"source_sha256": {KEPT: _sha(api / KEPT)},
                "code_sha256": {KEPT: code_digest_of_path(api / KEPT)},
                "status": "PASS", "purpose": "a weld"}
        host = metal_environment.host_line(environment)
        named = metal_runs.environment_of(host)
        metal_runs.bind_architecture(made, bound_before=None, architecture="applegpu_g13s", run={
            "host": host, "torch": named["torch"], "metal_frontend": named["metal_frontend"],
            "records": "apps/api/parity/meep_gpu/results/metal_a_earlier/ - 1 records",
            "recorded_utc": "2099-01-01T00:00:00Z", "artifact_sha256": "c" * 64,
            "subnormal_policy": "flush" + R.POLICY_SUFFIX,
            "environment_read_from": "metal_environment_earlier/start.json",
            "verdict_read_from": "metal_a_earlier/gate.json:release.released"})
        return made

    def artifact(family: str, stamp: str, record: dict) -> None:
        path = results / f"metal_{family}_{stamp}" / "gate.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"release": {"released": True}, "records": 1,
                                    "imported_source_sha256": record}))

    ledger.write_text(json.dumps({"metal_a_device_gate": entry(),
                                  "metal_b_device_gate": entry()}, indent=2, sort_keys=True)
                      + "\n", encoding="utf-8")
    return R, api, results, ledger, artifact


def test_the_metal_rebind_reads_every_chunk_stamp(metal_rebind, capsys):
    R, api, results, ledger, artifact = metal_rebind
    artifact("a", S0, {KEPT: _sha(api / KEPT)})
    artifact("b", S1, {KEPT: _sha(api / KEPT), OTHER: _sha(api / OTHER)})
    assert R.main(["--stamps", f"{S0},{S1}", "--write"]) == 0
    out = capsys.readouterr().out
    assert "2 rebound, 0 skipped, 0 without an artifact" in out
    runs = {key: entry["runs"]["applegpu_g13s"]
            for key, entry in json.loads(ledger.read_text()).items()}
    assert runs["metal_a_device_gate"]["environment_read_from"] == \
        f"metal_environment_{S0}/start.json"
    assert runs["metal_b_device_gate"]["environment_read_from"] == \
        f"metal_environment_{S1}/start.json"
    assert f"metal_b_{S1}/" in runs["metal_b_device_gate"]["records"]


def test_the_metal_rebind_one_stamp_at_a_time_ends_on_the_newest_run(metal_rebind, capsys):
    """How ``chunks.py bind`` drives it: one ``--stamp`` per selected stamp in launch order,
    so a unit run twice ends bound to its later run, and a stale run under a later stamp is
    skipped by the tool's own rule, leaving the earlier binding."""
    R, api, results, ledger, artifact = metal_rebind
    artifact("a", S0, {KEPT: _sha(api / KEPT)})
    artifact("a", S1, {KEPT: _sha(api / KEPT)})
    artifact("b", S0, {KEPT: _sha(api / KEPT)})
    artifact("b", S1, {KEPT: _sha(api / KEPT), OTHER: "0" * 64})   # stale under S1
    assert R.main(["--stamp", S0, "--write"]) == 0
    R.main(["--stamp", S1, "--write"])
    runs = {key: entry["runs"]["applegpu_g13s"]
            for key, entry in json.loads(ledger.read_text()).items()}
    assert f"metal_a_{S1}/" in runs["metal_a_device_gate"]["records"]
    assert f"metal_b_{S0}/" in runs["metal_b_device_gate"]["records"]
    capsys.readouterr()


def test_the_metal_rebind_refuses_a_moved_record_and_two_runs_of_one_gate(metal_rebind, capsys):
    R, api, results, ledger, artifact = metal_rebind
    before = ledger.read_text()
    artifact("a", S0, {KEPT: _sha(api / KEPT)})
    artifact("a", S1, {KEPT: _sha(api / KEPT)})
    artifact("b", S1, {KEPT: _sha(api / KEPT), OTHER: "0" * 64})   # OTHER moved since the run
    assert R.main(["--stamps", f"{S0},{S1}", "--write"]) == 1
    out = capsys.readouterr().out
    assert "SKIPPED  metal_a_device_gate: metal_a_<stamp>/ exists under 2 of the stamps" in out
    assert (f"SKIPPED  metal_b_device_gate: metal_b_{S1}/gate.json records 1 file(s) the "
            f"release tree does not hold at the recorded bytes: {OTHER} (other bytes)") in out
    assert "0 rebound" in out and ledger.read_text() == before


def test_the_metal_rebind_refuses_a_stamp_list_naming_two_environments(metal_rebind):
    R, api, results, ledger, artifact = metal_rebind
    other = dict(ENVIRONMENT, torch="9.9.9")
    import metal_environment  # noqa: PLC0415
    for name in ("start", "end"):
        (results / f"metal_environment_{S1}" / f"{name}.json").write_text(
            json.dumps(dict(other, **{metal_environment.FAST_MATH: None})))
    with pytest.raises(SystemExit, match="environments differ"):
        R.main(["--stamps", f"{S0},{S1}"])


def test_the_board_repoint_reads_every_chunk_stamp_and_never_chooses_between_runs(tmp_path):
    import repoint_board_bindings as tool  # noqa: PLC0415
    api, results = tmp_path, tmp_path / "results"
    source = api / "meep_gpu" / "metal_kernels" / "one.py"
    source.parent.mkdir(parents=True)
    source.write_text("# one\n")

    def run(stamp: str, digest: str) -> None:
        (results / f"metal_one_gate_{stamp}").mkdir(parents=True, exist_ok=True)
        (results / f"metal_one_gate_{stamp}" / "gate.json").write_text(json.dumps(
            {"release": {"released": True}, "source_sha256": {str(source): digest}}))

    text = f'{tool.METAL_ANCHOR}    "a": ("gate", "metal_one_gate_2099-01-01_a"),\n}}\n'
    for stamp in ("2099-01-01_a", S0, S1):       # every run on the one certified Mac
        _environment(results, stamp)
    run(S1, _sha(source))
    measured = tool.measure_metal(text, results, f"{S0},{S1}", api, stamps=[S0, S1])
    row = measured["rows"]["metal_one_gate_2099-01-01_a"]
    assert row["state"] == "repointed" and row["chosen"] == f"metal_one_gate_{S1}"
    written, _summary = tool.rewrite_metal(text, measured, today="2099-03-04")
    assert f"metal_one_gate_{S1}" in written and "the chunk stamps" in written
    run(S1, "0" * 64)
    row = tool.measure_metal(text, results, "x", api, stamps=[S0, S1])["rows"][
        "metal_one_gate_2099-01-01_a"]
    assert row["state"] == "unresolved" and "moved" in row["why_not"]
    run(S1, _sha(source))
    run(S0, _sha(source))
    row = tool.measure_metal(text, results, "x", api, stamps=[S0, S1])["rows"][
        "metal_one_gate_2099-01-01_a"]
    assert row["state"] == "unresolved" and "two runs of one gate" in row["why_not"]


# --------------------------------------------------------------------------- binds ------
def _selected_world(machine: str, tmp_path: Path):
    """The default partition launched with every unit holding, then three units run again
    alone: one of them stale in its first run. The selection verify writes."""
    units = {u["id"]: u for u in C.units_for(REPO, machine)}
    results = tmp_path / "results"
    index = _index(machine)
    kinds = ("cuda_subdir", "triton_fleet", "triton_stencil") if machine == "nvidia" else (
        "metal_gate",)
    picked = [uid for uid, u in sorted(units.items()) if u["kind"] in kinds][:3]
    for k, chunk in enumerate(index["chunks"]):
        for uid in chunk["units"]:
            _plant(units[uid], chunk["stamp"], results, holds=uid != picked[0])
        _launch(results, index, chunk, chunk["units"], f"2099-03-04T0{k}:00:00Z")
    again = C.new_chunk(index, [units[u] for u in picked], "three units again")
    for uid in picked:
        _plant(units[uid], again["stamp"], results, holds=True)
    _launch(results, index, again, picked, "2099-03-04T09:00:00Z")
    report = C.verify(REPO, results, index, say=lambda _l: None)
    assert report["ok"], report["counts"]
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps(C.selection_of(report, index, REPO)))
    return index, results, units, picked, again, selection


def test_the_metal_bind_calls_each_selected_stamp_once_in_launch_order(tmp_path):
    index, results, units, picked, again, selection = _selected_world("metal", tmp_path)
    lines = C.bind_commands(index, results, "index.json", str(selection), REPO)
    assert lines[1].startswith("python rounds/1.0/chunks.py verify")
    calls = [line for line in lines if line.startswith("python rebind_metal_welds.py")]
    assert calls == [f"python rebind_metal_welds.py --stamp {c['stamp']}"
                     for c in index["chunks"]]
    assert not [line for line in lines if "STOP" in line]
    # A later run the tool would bind and the selection does not take is named, never bound.
    sel = json.loads(selection.read_text())
    uid = picked[1]
    sel["selected"][uid] = sel["units"][uid]["candidates"][0]["stamp"]
    selection.write_text(json.dumps(sel))
    lines = C.bind_commands(index, results, "index.json", str(selection), REPO)
    stops = [line for line in lines if line.startswith("# STOP rebind_metal_welds.py")]
    assert len(stops) == 1 and uid in stops[0] and again["stamp"] in stops[0]
    assert f"mv results/{units[uid]['family']}_{again['stamp']}" in stops[0]
    # A STOP WITHHOLDS THE BIND: no command that writes a ledger is printed beside it.
    assert C.bind_withheld(lines) and lines[-1].startswith(C.WITHHELD)
    assert not [line for line in lines if line.startswith("python rebind_")]


def _set_aside(lines, results: Path) -> int:
    """Run every set-aside command the STOP lines print, from the directory holding
    ``results/``, as an operator would; returns how many ran."""
    commands = [line.split("set it aside first: ", 1)[1] for line in lines
                if line.startswith("# STOP") and "set it aside first: " in line]
    for command in commands:
        subprocess.run(["sh", "-c", command], cwd=str(results.parent), check=True)
    return len(commands)


def _reselect(index: dict, results: Path, selection: Path) -> dict:
    report = C.verify(REPO, results, index, say=lambda _l: None)
    selection.write_text(json.dumps(C.selection_of(report, index, REPO)))
    return report


def test_a_run_of_no_launch_under_a_selected_stamp_withholds_the_bind(tmp_path):
    """A unit's directory under a stamp whose launch record does not list it was never
    judged; the stamp-wide rebind would bind it, so it is named and the bind withheld."""
    index, results, units, picked, again, selection = _selected_world("metal", tmp_path)
    stray = next(uid for uid in sorted(units) if uid not in again["units"])
    _plant(units[stray], again["stamp"], results, holds=True)
    lines = C.bind_commands(index, results, "index.json", str(selection), REPO)
    stops = [line for line in lines if line.startswith("# STOP rebind_metal_welds.py")]
    assert len(stops) == 1 and "launch record does not list it" in stops[0], lines
    assert C.bind_withheld(lines)
    assert _set_aside(lines, results) == 1
    assert (results / f"set_aside_{again['stamp']}" /
            f"{units[stray]['family']}_{again['stamp']}").is_dir()
    _reselect(index, results, selection)
    lines = C.bind_commands(index, results, "index.json", str(selection), REPO)
    assert not C.bind_withheld(lines)


def test_the_binds_first_command_refuses_a_selection_the_results_no_longer_give(tmp_path):
    """The bind is printed from one selection and runs later: its first command judges
    again and stops when the choice, or the bytes it binds, moved since; it writes nothing."""
    index, results, units, picked, again, selection = _selected_world("metal", tmp_path)
    path = tmp_path / "index.json"
    C.write_index(path, index)
    before = selection.read_bytes()
    argv = ["verify", "--index", str(path), "--repo", str(REPO), "--results", str(results),
            "--against", str(selection)]
    assert C.main(argv) == 0
    lines = C.bind_commands(index, results, str(path), str(selection), REPO)
    assert f"--against {selection}" in lines[1]
    # Other bytes in a selected run's artifact: the selection no longer describes the bind.
    uid = picked[1]
    gate = next(results.glob(f"{units[uid]['family']}_{again['stamp']}/*.json"))
    gate.write_text(gate.read_text() + "\n")
    assert C.main(argv) == 1
    # A newer run of a unit moves its choice.
    shutil.rmtree(results / f"{units[uid]['family']}_{again['stamp']}")
    assert C.main(argv) == 1
    assert selection.read_bytes() == before


def test_the_nvidia_bind_binds_the_selection_and_nothing_else(tmp_path):
    index, results, units, picked, again, selection = _selected_world("nvidia", tmp_path)
    # The one CUDA unit whose first run went stale sits in a campaign the selection records
    # for its other units: record_cuda_regate.py records whole campaigns, so it is named,
    # and nothing that writes a ledger is printed until it is set aside.
    stale = picked[0]
    first = json.loads(selection.read_text())["units"][stale]["candidates"][0]["stamp"]
    lines = C.bind_commands(index, results, "index.json", str(selection), REPO)
    stops = [line for line in lines if line.startswith("# STOP")]
    assert len(stops) == 1 and stops[0].startswith("# STOP record_cuda_regate.py")
    assert units[stale]["subdir"] in stops[0] and f"set_aside_{first}" in stops[0]
    assert C.bind_withheld(lines) and not [line for line in lines if line.startswith(
        ("python rebind", "python record", "python recut"))]
    assert _set_aside(lines, results) == 1
    assert _reselect(index, results, selection)["ok"]
    lines = C.bind_commands(index, results, "index.json", str(selection), REPO)
    assert not C.bind_withheld(lines) and not [line for line in lines if "STOP" in line]
    text = "\n".join(lines)
    sel = json.loads(selection.read_text())["selected"]
    # One Triton call per stamp the selection draws a welded fleet or stencil unit from whose
    # key the Triton ledger holds, with --only the keys of the units selected there.
    held = _ledger_keys()
    keyed = {directory for key, directory in C._literal_dict(
        HERE / "rebind_triton_welds.py", "CAMPAIGN_DIRS").items() if key in held}
    triton = [line for line in lines if line.startswith("python rebind_triton_welds.py --campaign")]
    stamps = C._ordered(json.loads(selection.read_text()),
                        (sel[u] for u, unit in units.items()
                         if unit["kind"] in ("triton_fleet", "triton_stencil")
                         and (unit.get("gate") or unit.get("family")) in keyed))
    assert len(triton) == len(stamps) and all(" --only " in line for line in triton)
    assert all(set(_only(line)) <= held for line in triton), triton
    for line, stamp in zip(triton, stamps):
        assert all(stamp in part for part in line.split() if part.startswith("results/"))
    # The CUDA campaigns the selection draws from, recorded then rebound in launch order.
    cuda = [line for line in lines if line.startswith("python record_cuda_regate.py")]
    assert cuda and all(f"cuda_regate_{s}_g0" in "".join(cuda) for s in
                        {sel[u] for u in units if units[u]["kind"] == "cuda_subdir"})
    assert f"results/triton_bit_identity_{sel['triton/bit_identity']}/gate.json" in text
    assert f"--run results/triton_composition_{sel['triton/composition']} " in text
    assert f"results/cuda_ade_recut_{sel['cuda/ade_recut']}" in text
    assert "--family-recert" in text
    assert lines[-1] == C.bind_plan_last(False)
    assert all(line.endswith(" --write") for line in C.bind_commands(
        index, results, "index.json", str(selection), REPO, write=True)[2:] if
        line.startswith("python ") and "chunks.py verify" not in line)


def test_the_family_recert_roots_put_each_familys_selected_root_first():
    order = C._topological(["a", "b", "c"], {"c": {"a"}, "a": set(), "b": set()})
    assert order == ["b", "c", "a"]
    assert C._topological(["a", "b"], {"a": {"b"}, "b": {"a"}}) is None


def test_a_new_devices_bind_re_points_no_board(tmp_path):
    index, results, units, picked, again, selection = _selected_world("metal", tmp_path)
    sel = json.loads(selection.read_text())
    assert C.records_commands(index, sel, results, REPO)[-1].startswith(
        "python repoint_board_bindings.py metal --stamps ")
    new = dict(index, device="applegpu_g15s", device_mode="new")
    lines = C.records_commands(new, dict(sel, device="applegpu_g15s"), results, REPO)
    assert len(lines) == 1 and lines[0].startswith("no board re-point")


def test_legs_bind_takes_the_chunk_index_and_its_selection_and_refuses_a_metal_one(
        tmp_path, capsys):
    index, results, units, picked, again, selection = _selected_world("nvidia", tmp_path)
    path = tmp_path / "index.json"
    C.write_index(path, index)
    with pytest.raises(SystemExit):
        L.main(["bind", "--chunks", str(path), "--results", str(results)])
    argv = ["bind", "--chunks", str(path), "--results", str(results),
            "--selection", str(selection)]
    assert L.main(argv) == 1                      # a STOP stands: the bind is withheld
    out = capsys.readouterr().out.splitlines()
    assert out[-1].startswith(C.WITHHELD) and _set_aside(out, results) == 1
    _reselect(index, results, selection)
    assert L.main(argv) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("# the bind of selection") and "chunks.py verify" in out[1]
    assert "# records time (RECORDS stage), after the bind:" in out
    metal = tmp_path / "metal.json"
    C.write_index(metal, _index("metal"))
    with pytest.raises(SystemExit):
        L.main(["bind", "--chunks", str(metal), "--selection", str(selection)])


def test_the_chunked_bind_ends_as_the_unchunked_bind_does():
    """The label pass runs after every re-bind in both binds, so the guard's ledger leg
    reads the same ledgers either way."""
    for write in (False, True):
        unchunked = L.bind_plan(Path("results"), ROUND_STAMP, 1, write=write)
        assert C.bind_plan_last(write) == unchunked[-1]
        assert "neutralize_ledger_labels.py" in unchunked[-1]


# ------------------------------ the Triton --only names only keys the ledger holds ------
#: The R27 round's NVIDIA bind: byte copies of its index and of the selection its verify
#: wrote, and the lines ``legs.py bind --chunks`` printed from them on the frozen 1.0 tree,
#: the two paths it was given written ``<index>`` and ``<selection>``. That print named, in
#: the ``--only`` of two stamps, a ``CAMPAIGN_DIRS`` key the Triton ledger never held, and
#: ``rebind_triton_welds.py`` refused each of the two lines whole, binding none of its keys.
R27 = HERE / "fixtures" / "round_chunks_r27"
R27_SHA256 = {
    "INDEX_nvidia.json": "32d89cce2ce7d7847eb33b7753823357b3c03c213dd04cb1b8b3ed6c44c96b94",
    "SELECTION_nvidia.json": "537f0d1e094f9e0b09861b615c391515445e9f51eb418648eb3d6c96c02be869"}
#: The ledger ``rebind_triton_welds.py`` binds, spelled here rather than read off ``chunks.py``.
TRITON_LEDGER = REPO / "meep_gpu" / "triton_kernels" / "fingerprints.json"
TRITON_REBIND = "python rebind_triton_welds.py --campaign"
REFUSED = "# REFUSED rebind_triton_welds.py --only: "
_ROOT_STAMP = re.compile(r"results/triton_(?:fleet|offdiag_stencil_welds)_"
                         r"(\d{4}-\d\d-\d\d_\d+_cc\d+_c\d+)(?:_|$)")


def _ledger_keys() -> set:
    return set(json.loads(TRITON_LEDGER.read_text(encoding="utf-8")))


def _only(line: str) -> list:
    """The keys of a printed Triton line's ``--only``."""
    return line.rsplit(" --only ", 1)[1].split(" ", 1)[0].split(",")


def _line_stamp(line: str) -> str:
    """The stamp of a printed Triton line, read off its first root."""
    return _ROOT_STAMP.search(line.split(" --only ", 1)[0].split()[3]).group(1)


def _campaign_dirs_by_dir() -> dict:
    by_dir: dict = {}
    for key, directory in C._literal_dict(HERE / "rebind_triton_welds.py",
                                          "CAMPAIGN_DIRS").items():
        by_dir.setdefault(directory, []).append(key)
    return by_dir


def _r27_bind(tmp_path: Path, write: bool = False):
    """The R27 selection's bind as ``legs.py bind --chunks`` prints it (head, lines, records
    lines), from an empty ``results/``, and the round's own print. The round's results held
    no ``_r`` continuation, one CUDA campaign per stamp and no run a tool would bind against
    the selection, so the roots its print names follow from the stamps alone: the tree the
    round ran prints it line for line from an empty ``results/``."""
    results = tmp_path / "results"
    results.mkdir(exist_ok=True)
    index = C.read_index(R27 / "INDEX_nvidia.json")
    path = R27 / "SELECTION_nvidia.json"
    lines = C.bind_commands(index, results, "<index>", str(path), REPO, write=write)
    lines += C.records_lines(index, C.read_selection(path, index), results, REPO)
    printed = (R27 / "BIND_nvidia.txt").read_text(encoding="utf-8").splitlines()
    return [line.replace(str(path), "<selection>") for line in lines], printed


def _expected_bind(printed, selection: dict, held: set) -> list:
    """The round's print with each Triton line's ``--only`` cut to the keys ``held``, the
    selected units whose keys it drops named before it (``("refused", unit, stamp, keys)``),
    and a line left with no key not printed; every other line exactly as printed."""
    by_dir = _campaign_dirs_by_dir()
    refusals: dict = {}
    for uid, unit in sorted({u["id"]: u for u in C.units_for(REPO, "nvidia")}.items()):
        stamp = selection["selected"].get(uid)
        if stamp is None or unit["kind"] not in ("triton_fleet", "triton_stencil"):
            continue
        lacking = [key for key in by_dir.get(unit.get("gate") or unit.get("family"), [])
                   if key not in held]
        if lacking:
            refusals.setdefault(stamp, []).append((uid, lacking))
    out = []
    for line in printed:
        if not line.startswith(TRITON_REBIND):
            out.append(line)
            continue
        stamp = _line_stamp(line)
        out += [("refused", uid, stamp, lacking) for uid, lacking in refusals.get(stamp, ())]
        keys = [key for key in _only(line) if key in held]
        if keys:
            out.append(line.split(" --only ", 1)[0] + " --only " + ",".join(keys))
    return out


def _assert_bind(lines, expected) -> None:
    assert len(lines) == len(expected), "\n".join(map(str, lines))
    for line, want in zip(lines, expected):
        if isinstance(want, tuple):
            _refused, uid, stamp, lacking = want
            assert line.startswith(REFUSED) and "STOP" not in line, line
            assert f" {uid}, selected under {stamp}, welds {', '.join(lacking)}, " in line, line
        else:
            assert line == want


class _PastTheOnlyGuard(Exception):
    """Raised by the first step ``rebind_triton_welds.main`` takes after its ``--only`` guard."""


def _run_to_the_only_guard(monkeypatch, tmp_path: Path, line: str):
    """``rebind_triton_welds.py`` on one printed line, against the checkout's own Triton ledger,
    as far as its ``--only`` guard: the roots the line names exist (empty) under a planted
    harness, and the tool's first step after the guard raises, so nothing past it runs.
    Returns what ended the run: the tool's SystemExit when its guard refuses the line."""
    import rebind_triton_welds as RT  # noqa: PLC0415
    assert RT.LEDGER.resolve() == TRITON_LEDGER.resolve()
    api = (tmp_path / "api").resolve()
    here = api / "parity" / "meep_gpu"
    argv = shlex.split(line)[2:]
    for arg in argv:
        if arg.startswith("results/"):
            (here / arg).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(RT, "_HERE", here)
    monkeypatch.setattr(RT, "_API", api)

    def past(*_args, **_kwargs):
        raise _PastTheOnlyGuard()

    monkeypatch.setattr(RT, "retire_welds", past)
    with pytest.raises((SystemExit, _PastTheOnlyGuard)) as ended:
        RT.main(argv)
    return ended.value


def test_the_r27_fixtures_are_the_rounds_own_index_selection_and_print():
    for name, digest in R27_SHA256.items():
        assert _sha(R27 / name) == digest, name
    printed = (R27 / "BIND_nvidia.txt").read_text(encoding="utf-8").splitlines()
    assert printed[0].startswith("# the bind of selection <selection> (sha256 537f0d1e094f9e0b): "
                                 "123 unit(s) from 10 stamp(s)")
    assert len(printed) == 26 and sum(line.startswith(TRITON_REBIND) for line in printed) == 7


def test_the_bind_reads_the_ledger_the_triton_rebind_binds(tmp_path):
    import rebind_triton_welds as RT  # noqa: PLC0415
    rel, keys = C.triton_ledger_keys(REPO)
    assert (REPO / rel).resolve() == RT.LEDGER.resolve() == TRITON_LEDGER.resolve()
    assert keys == _ledger_keys()
    # A checkout with no Triton ledger is refused by name: no --only can be judged.
    harness = tmp_path / "parity" / "meep_gpu"
    harness.mkdir(parents=True)
    shutil.copy(HERE / "import_closure.py", harness / "import_closure.py")
    with pytest.raises(SystemExit, match=f"has no Triton ledger at {rel}"):
        C.triton_ledger_keys(tmp_path)


def test_the_r27_bind_names_only_keys_the_ledger_holds_and_nothing_else_moves(tmp_path):
    """Every line of the round's print is printed again byte for byte except its Triton
    lines: each names only the keys the ledger holds, after a REFUSED line per selected unit
    whose key the ledger lacks, and a REFUSED line withholds nothing."""
    lines, printed = _r27_bind(tmp_path)
    selection = json.loads((R27 / "SELECTION_nvidia.json").read_text(encoding="utf-8"))
    _assert_bind(lines, _expected_bind(printed, selection, _ledger_keys()))
    assert not C.bind_withheld(lines)
    written, _printed = _r27_bind(tmp_path, write=True)
    assert [line for line in written if line.startswith(REFUSED)] == [
        line for line in lines if line.startswith(REFUSED)]
    assert [line for line in written if line.startswith(TRITON_REBIND)] == [
        line + " --write" for line in lines if line.startswith(TRITON_REBIND)]


@pytest.mark.parametrize("dropped", [
    ("triton_conductivity_device_gate",),
    ("triton_complex_device_gate", "triton_offdiag_fused_electric_pair_device_gate")])
def test_a_key_the_ledger_lacks_leaves_its_line_and_a_stamp_left_with_none_prints_none(
        tmp_path, monkeypatch, dropped):
    """The same bind against a ledger lacking more keys: c10's only key, so c10 prints no
    Triton line; or one of c0's ten keys and c9's one stencil key, so c0's line names nine
    and c9 prints none."""
    rel, keys = C.triton_ledger_keys(REPO)
    monkeypatch.setattr(C, "triton_ledger_keys", lambda repo: (rel, keys - set(dropped)))
    lines, printed = _r27_bind(tmp_path)
    selection = json.loads((R27 / "SELECTION_nvidia.json").read_text(encoding="utf-8"))
    _assert_bind(lines, _expected_bind(printed, selection, _ledger_keys() - set(dropped)))
    gone = {"2026-10-15_10_cc86_c10"} if len(dropped) == 1 else {"2026-10-15_10_cc86_c9"}
    stamps = {_line_stamp(line) for line in lines if line.startswith(TRITON_REBIND)}
    assert stamps == {_line_stamp(line) for line in printed
                      if line.startswith(TRITON_REBIND)} - gone
    assert not any(key in line for key in dropped for line in lines
                   if line.startswith(TRITON_REBIND))


def test_the_rounds_two_refused_lines_are_cut_to_what_the_tools_guard_takes(tmp_path,
                                                                             monkeypatch):
    """The R27 case on the 1.0 ledger, which holds neither ``triton_bfast_device_gate`` nor
    ``triton_special_kz_device_gate`` (whether to seed them is a ruling, not a bind): the
    round's c8 and c11 lines, run as printed, end at the tool's guard naming that key; the
    bind printed now gives c8 14 of its 15 keys and c11 3 of its 4, its campaign arguments
    unchanged -- the two lines the round ran by hand instead -- and the tool's guard takes
    every Triton line it prints."""
    lines, printed = _r27_bind(tmp_path)
    held = _ledger_keys()
    old = {_line_stamp(line): line for line in printed if line.startswith(TRITON_REBIND)}
    new = {_line_stamp(line): line for line in lines if line.startswith(TRITON_REBIND)}
    c8, c11 = "2026-10-15_10_cc86_c8", "2026-10-15_10_cc86_c11"
    assert {stamp for stamp, line in old.items() if set(_only(line)) - held} == {c8, c11}
    for stamp, key in ((c8, "triton_bfast_device_gate"), (c11, "triton_special_kz_device_gate")):
        ended = _run_to_the_only_guard(monkeypatch, tmp_path, old[stamp])
        assert isinstance(ended, SystemExit), ended
        assert str(ended) == f"--only names ['{key}'], absent from the ledger"
        assert _only(new[stamp]) == [k for k in _only(old[stamp]) if k != key]
        assert new[stamp].split(" --only ")[0] == old[stamp].split(" --only ")[0]
    assert len(_only(new[c8])) == 14 and ",".join(_only(new[c11])) == (
        "triton_conductive_fused_electric_pair_device_gate,triton_fused_ade_chain_device_gate,"
        "triton_fused_ade_state_device_gate")
    refused = [line for line in lines if line.startswith(REFUSED)]
    assert len(refused) == 2
    assert (f" triton/fleet/bfast, selected under {c8}, welds triton_bfast_device_gate, "
            in refused[0])
    assert (f" triton/fleet/special_kz, selected under {c11}, welds "
            f"triton_special_kz_device_gate, " in refused[1])
    assert set(new) == set(old) and len(new) == 7
    for stamp, line in sorted(new.items()):
        ended = _run_to_the_only_guard(monkeypatch, tmp_path, line)
        assert isinstance(ended, _PastTheOnlyGuard), (stamp, ended)


def test_legs_bind_prints_the_r27_bind_and_exits_0(tmp_path, capsys):
    """``legs.py bind --chunks``, the command the round runs, on the copies: no line withheld,
    and the lines the bind above prints."""
    results = tmp_path / "results"
    results.mkdir()
    index, selection = R27 / "INDEX_nvidia.json", R27 / "SELECTION_nvidia.json"
    assert L.main(["bind", "--chunks", str(index), "--selection", str(selection),
                   "--results", str(results)]) == 0
    out = [line.replace(str(index), "<index>").replace(str(selection), "<selection>")
           for line in capsys.readouterr().out.splitlines()]
    lines, printed = _r27_bind(tmp_path)
    assert out == lines and not C.bind_withheld(out) and out[-1] == printed[-1]


def test_a_unit_run_again_alone_whose_key_the_ledger_lacks_prints_no_triton_line(
        tmp_path, monkeypatch):
    """A stamp whose only welded unit's key the ledger lacks prints no Triton line and names
    the unit on a REFUSED line, which withholds nothing; with the key in the ledger the same
    stamp prints its line with that key."""
    index, results, units, picked, again, selection = _selected_world("nvidia", tmp_path)
    assert _set_aside(C.bind_commands(index, results, "index.json", str(selection), REPO),
                      results) == 1
    by_dir, held = _campaign_dirs_by_dir(), _ledger_keys()
    uid = next(u for u, unit in sorted(units.items()) if unit["kind"] == "triton_fleet"
               and by_dir.get(unit["gate"]) and set(by_dir[unit["gate"]]) <= held)
    welds = sorted(by_dir[units[uid]["gate"]])
    alone = C.new_chunk(index, [units[uid]], "one welded unit again")
    _plant(units[uid], alone["stamp"], results, holds=True)
    _launch(results, index, alone, [uid], "2099-03-04T10:00:00Z")
    assert _reselect(index, results, selection)["ok"]
    assert json.loads(selection.read_text())["selected"][uid] == alone["stamp"]

    def own(lines):
        return [line for line in lines
                if line.startswith(TRITON_REBIND) and _line_stamp(line) == alone["stamp"]]

    lines = C.bind_commands(index, results, "index.json", str(selection), REPO)
    assert [_only(line) for line in own(lines)] == [welds]
    rel, keys = C.triton_ledger_keys(REPO)
    monkeypatch.setattr(C, "triton_ledger_keys", lambda repo: (rel, keys - set(welds)))
    lines = C.bind_commands(index, results, "index.json", str(selection), REPO)
    assert own(lines) == [] and not C.bind_withheld(lines)
    named = [line for line in lines
             if line.startswith(REFUSED) and f" {uid}, selected under {alone['stamp']}, " in line]
    assert len(named) == 1 and f"welds {', '.join(welds)}, " in named[0], lines


# ----------------------------------------------------------------- the round drivers ----
def _round_args(**kw) -> argparse.Namespace:
    base = dict(stage="gates", stamp=None, chunk_index="", chunk="", home="/h", gpu_ids="",
                capability="8.6")
    base.update(kw)
    return argparse.Namespace(**base)


def test_the_nvidia_round_runs_a_chunk_under_its_own_stamp_and_home(tmp_path):
    path = tmp_path / "index.json"
    C.write_index(path, _index("nvidia"))
    args = _round_args(chunk_index=str(path), chunk="c1", home=str(tmp_path / "lanes"))
    assert round_driver.apply_chunk(args) is None
    assert args.stamp == f"{ROUND_STAMP}_cc86_c1"
    assert round_driver.lanes_home(args) == (tmp_path / "lanes" / f"{ROUND_STAMP}_cc86_c1").resolve()
    assert "is not chunk c1's stamp" in round_driver.apply_chunk(
        _round_args(chunk_index=str(path), chunk="c1", stamp=ROUND_STAMP))
    assert "gate stage only" in round_driver.apply_chunk(
        _round_args(stage="route", chunk_index=str(path), chunk="c1"))
    assert "--stamp is required" in round_driver.apply_chunk(_round_args())
    for typo in ("9", "8.60", "sm_90", "9,0"):
        assert "--capability is refused" in round_driver.apply_chunk(
            _round_args(stamp="s", capability=typo)), typo
    assert "absolute path" in round_driver.apply_chunk(
        _round_args(chunk_index="index.json", chunk="c1"))
    subs = round_driver.subs_for(args, tmp_path, {})
    assert subs["@FLEET@"] == str(tmp_path / f"triton_fleet_{ROUND_STAMP}_cc86_c1")
    assert subs == C.stamp_subs(tmp_path, args.stamp)
    lane = argparse.Namespace(repo="/r", python="py", lock_dir="/locks", max_locks=4,
                              stamp=args.stamp, avoid="", libcuda_stub="", corpus_root="",
                              timeout_floor=[])
    argv = round_driver.lane_argv(lane, Path("p"), Path("d"), Path("r"), subs, 3)
    assert argv[argv.index("--gpu") + 1] == "3" and "--lock-dir" in argv


def test_a_chunk_of_another_device_or_mode_is_refused_by_name():
    index = _index("nvidia")
    assert L.index_device_problems(index, "8.6", False) == []
    assert "this card is cc90" in L.index_device_problems(index, "9.0", False)[0]
    assert "pass --new-capability" in L.index_device_problems(
        dict(index, device="cc90", device_mode="new"), "9.0", False)[0]
    assert "drop --new-capability" in L.index_device_problems(index, "8.6", True)[0]
    assert "metal index" in L.index_device_problems(_index("metal"), "8.6", False)[0]


def test_the_launch_lock_keeps_a_second_launcher_of_a_chunk_out(tmp_path):
    first = round_driver.LaunchLock(tmp_path, "first")
    assert first.take() is None
    assert "this chunk is running" in round_driver.LaunchLock(tmp_path, "second").take()
    first.release()
    assert not (tmp_path / "LAUNCH.lock").exists()
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    (tmp_path / "LAUNCH.lock").mkdir()
    (tmp_path / "LAUNCH.lock" / "pid").write_text(f"{gone.pid}\n")
    taken = round_driver.LaunchLock(tmp_path, "third")
    assert taken.take() is None
    taken.release()


def test_the_metal_round_runs_a_chunk_under_its_own_stamp(tmp_path):
    path = tmp_path / "metal.json"
    index = _index("metal")
    C.write_index(path, index)
    chunk = index["chunks"][1]

    def args(**kw):
        base = dict(stage="fleet", stamp=None, chunk_index=str(path), chunk="m1", only="",
                    logs="")
        base.update(kw)
        return argparse.Namespace(**base)

    a = args()
    assert metal_round.apply_chunk(a) is None
    assert a.stamp == chunk["stamp"] and a.only.split(",") == C.metal_gates(chunk)
    assert set(a.stamp_of) == set(C.metal_gates(chunk))
    gates = C.metal_gates(chunk)
    narrowed = args(only=gates[0])
    assert metal_round.apply_chunk(narrowed) is None and narrowed.only == gates[0]
    assert "not chunked" in metal_round.apply_chunk(args(stage="route"))
    assert "is not chunk m1's stamp" in metal_round.apply_chunk(args(stamp="2099-01-01_x"))
    status = args(stage="status", chunk="")
    assert metal_round.apply_chunk(status) is None
    assert len(status.stamp_of) == len(C.metal_units(REPO))
    assert Path(status.logs).parent == path.resolve().parent
    assert "status stage only" in metal_round.apply_chunk(args(chunk=""))


# ------------------------------------- the Triton and CUDA rebinds apply the same rule ---
import test_rebind_cuda_pin_retirement as cuda_plant  # noqa: E402
import test_rebind_unimported_pins as planted  # noqa: E402


@pytest.fixture()
def triton_rebind(tmp_path, monkeypatch):
    """The planted Triton root of ``test_rebind_unimported_pins``."""
    import rebind_triton_welds as RT  # noqa: PLC0415
    api = tmp_path / "api"
    planted._plant_tree(api, "from meep_gpu import grid\n")
    for rel, text in ((planted.TGATE, "from meep_gpu import grid\n"
                                      "from meep_gpu.triton_kernels import planted\n"),
                      (planted.TKERNEL, "K = 2\n"), ("meep_gpu/triton_kernels/__init__.py", "")):
        (api / rel).parent.mkdir(parents=True, exist_ok=True)
        (api / rel).write_text(text, encoding="utf-8")
    ledger = api / "meep_gpu" / "triton_kernels" / "fingerprints.json"
    monkeypatch.setattr(RT, "_API", api)
    monkeypatch.setattr(RT, "LEDGER", ledger)
    monkeypatch.setitem(RT.CAMPAIGN_DIRS, planted.TKEY, "planted")
    monkeypatch.setattr(RT, "RETIRED_UNIMPORTED_PINS", {})
    monkeypatch.setattr(RT, "toolchain_refusal", lambda identity, ledger: None)
    ledger.write_text(json.dumps({planted.TKEY: planted._triton_entry(api)}))
    return RT, api, ledger


@pytest.mark.parametrize("moved", [False, True])
def test_the_triton_rebind_refuses_a_run_whose_record_names_a_moved_file(triton_rebind, capsys,
                                                                         moved):
    RT, api, ledger = triton_rebind
    before = ledger.read_text()
    record = planted._triton_live(api, planted.DRIVER)       # recorded, not curated
    if moved:
        record[planted.DRIVER] = "0" * 64
    root = planted._triton_campaign(api, record)
    rc = RT.main(["--campaign", str(root), "--write"])
    out = capsys.readouterr().out
    if moved:
        assert rc == 1 and ledger.read_text() == before
        assert (f"SKIPPED   {planted.TKEY}: planted/gate.json records 1 file(s) the release "
                f"tree does not hold at the recorded bytes: {planted.DRIVER} (other bytes)") in out
    else:
        assert rc == 0 and f"rebound   {planted.TKEY}" in out, out


@pytest.mark.parametrize("moved", [False, True])
def test_the_cuda_rebind_refuses_legs_whose_record_names_a_moved_file(tmp_path, monkeypatch,
                                                                      capsys, moved):
    plant = cuda_plant.Plant(tmp_path, monkeypatch)
    other = "meep_gpu/other_planted.py"
    (plant.api / other).write_text("O = 1\n", encoding="utf-8")
    text = plant.write_ledger({cuda_plant.WELD: plant.entry(live=True)})
    record = plant.imports(other)                            # recorded, not curated
    if moved:
        record[other] = "0" * 64
    root = plant.campaign({cuda_plant.WELD: (record, record, True)})
    plant.run(root, "--write")
    out = capsys.readouterr().out
    if moved:
        assert plant.text() == text
        assert (f"records 1 file(s) the release tree does not hold at the recorded bytes: "
                f"{other} (other bytes)") in out and f"SKIPPED  {cuda_plant.WELD}:" in out
    else:
        assert f"rebound  {cuda_plant.WELD}: runs[{cuda_plant.CC}]" in out, out


# ------------------------------------------------------------- the artifact layout ------
def test_a_units_artifacts_are_found_where_the_round_writes_them(tmp_path):
    """The layout ``unit_dirs`` reads is the one the plans write: shard roots with their
    ``_flush`` twins and resume continuations, the chunk's CUDA campaign, the H->D roots,
    the composition roots, the Metal directories."""
    units = {u["id"]: u for u in C.nvidia_units(REPO)}
    stamp = f"{ROUND_STAMP}_cc86_c0"
    results = tmp_path / "results"
    t2 = next(u for u in units.values() if u["kind"] == "triton_fleet" and u["shard"] == "t2_rest")
    cuda = next(u for u in units.values() if u["kind"] == "cuda_subdir")
    hd = next(u for u in units.values() if u["kind"] == "triton_hd")
    comp = units["triton/composition"]
    want = {
        t2["id"]: results / f"triton_fleet_{stamp}_flush_t2_rest_r2" / t2["gate"],
        cuda["id"]: results / f"cuda_regate_{stamp}_g0" / cuda["subdir"],
        hd["id"]: results / f"triton_{hd['gate']}_{stamp}",
        comp["id"]: results / f"triton_composition_{stamp}_flush" / comp["gates"][0],
    }
    for directory in want.values():
        directory.mkdir(parents=True)
        (directory / "gate.json").write_text(json.dumps({"imported_source_sha256": {
            "parity/meep_gpu/gate_x.py": "0" * 64}}))
    for uid, directory in want.items():
        assert C.unit_dirs(units[uid], stamp, results) == [directory], uid
        assert [p for p, _payload in C.unit_records(units[uid], stamp, results)] == [
            directory / "gate.json"], uid
        assert C.unit_dirs(units[uid], f"{ROUND_STAMP}_cc86_c1", results) == [], uid
    metal = {u["id"]: u for u in C.metal_units(REPO)}["metal/gate_metal_pml"]
    (results / f"metal_pml_{ROUND_STAMP}_m0").mkdir()
    assert C.unit_dirs(metal, f"{ROUND_STAMP}_m0", results) == [results / f"metal_pml_{ROUND_STAMP}_m0"]


def test_every_units_verdicts_lie_under_its_own_directories(tmp_path):
    """What ``verify`` reads a verdict from is an artifact the unit's own runs write, under
    the unit's own directories, so a verdict never comes from another unit's run."""
    results = tmp_path / "results"
    for machine in C.MACHINES:
        for unit in C.units_for(REPO, machine):
            stamp = f"{ROUND_STAMP}_{DEVICE[machine]}_{C.LETTER[machine]}0"
            _plant(unit, stamp, results, holds=True)
            dirs = C.unit_dirs(unit, stamp, results)
            for _leg, path in C.unit_verdict_paths(unit, stamp, results):
                assert any(path == d or d in path.parents for d in dirs), (unit["id"], path)


def test_the_previous_rounds_records_are_read_per_unit(release):
    repo, results, index, artifact = release
    previous = "2099-01-01_10"
    old = results / f"metal_fake_{previous}" / "gate.json"
    old.parent.mkdir(parents=True)
    old.write_text(json.dumps({"imported_source_sha256": {"meep_gpu/kernel.py": "1" * 64}}))
    found = C.prior_records(repo, results, previous, "metal", say=lambda _line: None)
    assert found["metal/gate_metal_fake"] == {"files": ["meep_gpu/kernel.py"], "artifacts": 1}
    assert found["_stamp"] == previous


# --------------------------------------------------------------------- the command line ---
def test_extend_places_a_new_unit_in_a_new_chunk_and_plan_writes_its_lanes(tmp_path, capsys):
    index = _index("nvidia")
    dropped = index["chunks"][3]["units"].pop()                 # as if a lane added a gate
    path = tmp_path / "index.json"
    C.write_index(path, index)
    assert any("in no chunk" in p for p in C.collisions(C.read_index(path), repo=REPO))
    assert C.main(["extend", "--index", str(path), "--repo", str(REPO)]) == 0
    extended = C.read_index(path)
    assert extended["chunks"][-1]["id"] == "c4" and extended["chunks"][-1]["units"] == [dropped]
    assert extended["chunks"][-1]["stamp"] == f"{ROUND_STAMP}_cc86_c4"
    assert C.collisions(extended, repo=REPO) == []
    assert C.main(["extend", "--index", str(path), "--repo", str(REPO)]) == 0   # nothing left
    assert len(C.read_index(path)["chunks"]) == 5
    out = tmp_path / "plan"
    assert C.main(["plan", "--index", str(path), "--chunk", "c0", "--repo", str(REPO),
                   "--results", str(tmp_path / "results"), "--out", str(out), "--lanes", "2"]) == 0
    summary = json.loads((out / "lanes_summary.json").read_text())
    assert summary["stamp"] == f"{ROUND_STAMP}_cc86_c0" and len(summary["lanes"]) == 2
    steps = [s for lane in sorted(out.glob("lane_*.json"))
             for s in json.loads(lane.read_text())]
    assert sum(int(s["expect_s"]) for s in steps) == C.find_chunk(extended, "c0")["expect_s"]


def test_repartition_retires_only_unlaunched_chunks_and_never_reuses_an_id(tmp_path):
    index = _index("metal")
    path = tmp_path / "index.json"
    C.write_index(path, index)
    results = tmp_path / "results"
    _launch(results, index, index["chunks"][0], index["chunks"][0]["units"],
            "2099-03-04T00:00:00Z")
    assert C.main(["repartition", "--index", str(path), "--repo", str(REPO), "--results",
                   str(results), "--chunks", "5", "--why", "a finer split"]) == 0
    again = C.read_index(path)
    assert [c["id"] for c in again["chunks"]] == ["m0", "m3", "m4", "m5", "m6", "m7"]
    assert sorted(c["id"] for c in again["retired"]) == ["m1", "m2"]
    assert C.collisions(again, repo=REPO) == []
    _assert_disjoint(again)


def test_add_runs_the_units_a_verify_report_did_not_hold(tmp_path):
    index = _index("metal")
    path = tmp_path / "index.json"
    C.write_index(path, index)
    report = tmp_path / "verify.json"
    first, second = sorted(u for c in index["chunks"] for u in c["units"])[:2]
    report.write_text(json.dumps({"units": {first: {"state": "REFUSED"},
                                            second: {"state": "OK"}}}))
    assert C.main(["add", "--index", str(path), "--repo", str(REPO), "--from-verify",
                   str(report), "--why", "refused by verify"]) == 0
    assert C.read_index(path)["chunks"][-1]["units"] == [first]
    with pytest.raises(SystemExit, match="no unit"):
        C.main(["add", "--index", str(path), "--repo", str(REPO), "--units", "metal/nothing",
                "--why", "x"])


def test_partition_writes_the_device_and_its_mode(tmp_path, capsys):
    out = tmp_path / "index.json"
    round_stamp = C.registry_round_stamp(REPO, "nvidia")[0] or ROUND_STAMP
    assert C.main(["partition", "--repo", str(REPO), "--machine", "nvidia", "--capability",
                   "9.0", "--new-device", "--round", round_stamp, "--out", str(out)]) == 0
    index = C.read_index(out)
    assert (index["device"], index["device_mode"]) == ("cc90", "new")
    assert all(c["stamp"].startswith(f"{round_stamp}_cc90_c") for c in index["chunks"])
    with pytest.raises(SystemExit):
        C.main(["partition", "--repo", str(REPO), "--machine", "nvidia", "--architecture",
                "applegpu_g13s", "--round", ROUND_STAMP, "--out", str(tmp_path / "x.json")])
    with pytest.raises(SystemExit):
        C.main(["partition", "--repo", str(REPO), "--machine", "metal", "--capability", "8.6",
                "--round", ROUND_STAMP, "--out", str(tmp_path / "y.json")])


def test_ready_reads_a_changing_file_list_and_the_files_moved_since_a_commit(release, capsys):
    repo, results, index, artifact = release
    tip = _git(repo, "rev-parse", "HEAD")
    path = repo.parent / "index.json"
    C.write_index(path, index)
    prior = repo.parent / "prior.json"
    prior.write_text(json.dumps({"_stamp": "2099-01-01_10"}))
    listed = repo.parent / "changing.txt"
    listed.write_text("# open lanes\nmeep_gpu/kernel.py   # the kernel\n")
    out = repo.parent / "ready.json"
    assert C.main(["ready", "--index", str(path), "--repo", str(repo), "--changing-file",
                   str(listed), "--prior-records", str(prior), "--json", str(out)]) == 0
    assert json.loads(out.read_text())["chunks"]["m0"]["state"] == "NOT READY"
    assert "READY_CHUNKS - (0 of 1 chunks)" in capsys.readouterr().out
    assert C.main(["ready", "--index", str(path), "--repo", str(repo), "--since", tip,
                   "--prior-records", str(prior), "--json", str(out)]) == 0
    assert json.loads(out.read_text())["ready"] == ["m0"]
    (repo / "meep_gpu" / "kernel.py").write_text("K = 3\n")
    _git(repo, "commit", "-qam", "a landing moves the kernel")
    assert C.main(["ready", "--index", str(path), "--repo", str(repo), "--since", tip,
                   "--prior-records", str(prior), "--json", str(out)]) == 0
    report = json.loads(out.read_text())
    assert report["changing"] == ["meep_gpu/kernel.py"] and report["ready"] == []


def test_a_launched_units_closure_that_moved_since_its_launch_tip_is_stale(release):
    repo, results, index, artifact = release
    chunk = index["chunks"][0]
    prior = {"_stamp": "2099-01-01_10"}
    quiet = lambda _line: None  # noqa: E731
    assert C.staleness(repo, index, results, prior, say=quiet)["chunks"]["m0"]["state"] == \
        "NOT LAUNCHED"
    assert C.write_launch_record(results, index, chunk, repo) is None
    assert C.staleness(repo, index, results, prior, say=quiet)["chunks"]["m0"]["state"] == "CURRENT"
    (repo / "notes.txt").write_text("a landing outside the closure\n")
    _git(repo, "add", "notes.txt")
    _git(repo, "commit", "-qm", "a landing outside the closure")
    assert C.staleness(repo, index, results, prior, say=quiet)["chunks"]["m0"]["state"] == "CURRENT"
    (repo / "meep_gpu" / "kernel.py").write_text("K = 4\n")
    _git(repo, "commit", "-qam", "a landing inside the closure")
    report = C.staleness(repo, index, results, prior, say=quiet)
    assert report["stale"] == ["m0"] and report["stale_units"] == [UID]
    assert report["chunks"]["m0"]["stale_units"] == {UID: ["meep_gpu/kernel.py"]}


# ------------------------------------------- every round-record writer applies the rule ---
def _plant_api(tmp_path: Path):
    """A repository root holding one package module the planted legs record."""
    api = tmp_path / "api"
    (api / "meep_gpu").mkdir(parents=True)
    (api / "parity" / "meep_gpu" / "results").mkdir(parents=True)
    kernel = api / "meep_gpu" / "kernel.py"
    kernel.write_text("K = 1\n")
    return api, kernel


def _recorded(kernel: Path) -> dict:
    return {"imported_source_sha256": {"meep_gpu/kernel.py": _sha(kernel)}}


def test_the_cuda_regate_record_refuses_legs_whose_record_names_a_moved_file(
        tmp_path, monkeypatch, capsys):
    import record_cuda_regate as RR  # noqa: PLC0415
    api, kernel = _plant_api(tmp_path)
    campaign = api / "parity" / "meep_gpu" / "results" / f"cuda_regate_{ROUND_STAMP}_cc86_c1_g0"
    legs = ["flush/gate.json", "keep/gate.json"]
    for leg in legs:
        (campaign / "fam" / leg).parent.mkdir(parents=True)
        (campaign / "fam" / leg).write_text(json.dumps(_recorded(kernel)))
    rows = [{"block": "a_block", "directory": "fam", "legs": legs}]
    monkeypatch.setattr(RR, "_API", api)
    assert RR.drift_refusals(campaign, rows) == []
    kernel.write_text("K = 2\n")
    (why,) = RR.drift_refusals(campaign, rows)
    assert why.startswith("fam/ legs") and "(for a_block)" in why
    assert "meep_gpu/kernel.py (other bytes)" in why
    # main() asks it of the campaign it records.
    monkeypatch.setattr(RR, "_record", lambda: {})
    monkeypatch.setattr(RR, "record_disagrees_with_its_own_directories", lambda record: [])
    monkeypatch.setattr(RR, "survey", lambda campaign, record: (rows, []))
    assert RR.main(["--campaign", str(campaign)]) == 1
    assert "meep_gpu/kernel.py (other bytes)" in capsys.readouterr().out
    kernel.unlink()
    assert "meep_gpu/kernel.py (absent)" in RR.drift_refusals(campaign, rows)[0]


def test_the_ade_recut_refuses_a_leg_whose_record_names_a_moved_file_unless_declared(
        tmp_path, monkeypatch):
    import record_cuda_ade_recut as RA  # noqa: PLC0415
    api, kernel = _plant_api(tmp_path)
    monkeypatch.setattr(RA, "_API", api)
    campaign = tmp_path / f"cuda_ade_recut_{ROUND_STAMP}_cc86_c1"
    payloads = {leg: _recorded(kernel) for leg in RA.POLICY_LEGS}
    assert RA.import_drift(campaign, payloads, {}) == []
    kernel.write_text("K = 2\n")
    refused = RA.import_drift(campaign, payloads, {})
    assert [line.split(" ")[0] for line in refused] == [
        f"{campaign.name}/{leg}/gate.json" for leg in sorted(RA.POLICY_LEGS)]
    assert all("meep_gpu/kernel.py (other bytes)" in line for line in refused)
    assert RA.import_drift(campaign, payloads, {"meep_gpu/kernel.py": _sha(kernel)}) == []


@pytest.fixture()
def composition(tmp_path, monkeypatch):
    """``recut_composition_records`` over a planted root, its leg readers stubbed so only
    the release-tree rule decides."""
    import recut_composition_records as R  # noqa: PLC0415
    api, kernel = _plant_api(tmp_path)
    monkeypatch.setattr(R, "API", api)
    monkeypatch.setattr(R, "released", lambda payload: (True, "planted"))
    monkeypatch.setattr(R, "attained_policy", lambda payload: "flush")
    monkeypatch.setattr(R, "leg_capability", lambda payload, directories: ("8.6", {}))
    monkeypatch.setattr(R, "read_staged", lambda run: None)
    monkeypatch.setattr(R, "leg_digests", lambda payload, staged, directory: (
        {R.pinned_path("launch.py"): "1" * 64}, "planted"))
    monkeypatch.setattr(R, "campaign_row", lambda root, gate: (
        {"exit_code": 0, "released": True}, ""))
    return R, api, kernel


def test_the_composition_recut_refuses_a_flush_leg_whose_record_names_a_moved_file(
        composition, tmp_path):
    R, api, kernel = composition
    run = tmp_path / f"triton_composition_{ROUND_STAMP}_cc86_c1_flush"
    (run / "leg").mkdir(parents=True)
    (run / "leg" / "gate.json").write_text(json.dumps(_recorded(kernel)))

    def second():
        return R.second_policy_leg(run, "results/flush", ["leg"], "8.6", ["launch.py"],
                                   {"launch.py": "1" * 64}, "flush")
    evidence, why = second()
    assert evidence is not None and why == "", why
    kernel.write_text("K = 2\n")
    evidence, why = second()
    assert evidence is None
    assert why.startswith("the flush run: results/flush/leg/gate.json records 1 file(s)")
    assert R.leg_drift(_recorded(kernel), "x") is None


def test_the_family_recert_refuses_a_family_whose_record_names_a_moved_file(
        composition, tmp_path):
    R, api, kernel = composition
    root = tmp_path / f"triton_fleet_{ROUND_STAMP}_cc86_c1_flush"
    (root / "fam").mkdir(parents=True)
    (root / "fam" / "gate.json").write_text(json.dumps(_recorded(kernel)))
    evidence, reasons = R.family_policy_evidence([str(root)], ["fam"], "8.6", "flush")
    assert evidence is not None and reasons == [], reasons
    kernel.unlink()
    evidence, reasons = R.family_policy_evidence([str(root)], ["fam"], "8.6", "flush")
    assert evidence is None and len(reasons) == 1
    assert reasons[0].startswith(f"fam: {root}/fam/gate.json records 1 file(s)")
    assert "meep_gpu/kernel.py (absent)" in reasons[0]


from test_recut_route_status import _campaign as _route_campaign  # noqa: E402
from test_recut_route_status import _cut as _route_cut  # noqa: E402
from test_recut_route_status import _ledger as _route_ledger  # noqa: E402
from test_recut_route_status import tree  # noqa: E402,F401 - the fixture


@pytest.mark.parametrize("moved", [False, True])
def test_the_route_record_refuses_a_leg_whose_record_names_a_moved_file(tree, capsys, moved):
    results, ledger = tree
    _route_ledger(ledger)
    name = _route_campaign(results)
    leg = results / name / sorted(p.name for p in (results / name).iterdir())[0] / "gate.json"
    artifact = json.loads(leg.read_text())
    fastpath_py = REPO / "meep_gpu" / "fastpath.py"
    artifact["imported_source_sha256"] = {
        "meep_gpu/fastpath.py": "0" * 64 if moved else _sha(fastpath_py)}
    leg.write_text(json.dumps(artifact))
    rc = _route_cut(name)
    err = capsys.readouterr().err
    if moved:
        assert rc == 1, err
        assert (f"{name}/{leg.parent.name}/gate.json records 1 file(s) the release tree does "
                f"not hold at the recorded bytes: meep_gpu/fastpath.py (other bytes)") in err
    else:
        assert rc == 0, err
