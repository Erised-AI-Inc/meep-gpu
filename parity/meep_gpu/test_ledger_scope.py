"""A run depends on the ledgers of the kernel tables its device can read, and no others.

Two halves, and the second is sound only because of the first:

1. THE CLAIM, on a process. A child process with an audit hook on ``open``, installed
   before anything else is imported, loads and plans one platform's dispatch path through
   ``fastpath.plan_fast_path``. It runs on any host with no device: the Metal child
   drives the real composer and builders over the composition matrix with a stand-in
   ``torch`` that reaches no device; the NVIDIA child drives the real ladder with the
   stand-in device and composers ``test_nvidia_fast_default`` uses. Each child has to show
   it measured something: plans that dispatched, its own ledgers opened, and, on the Metal
   child, the Triton module imported whose import is why a bench records those ledgers.

2. THE TOOLS (``plan_carry.py``, ``timing_rows.py``), with planted ledger moves in both
   directions: a move of a ledger the run's device reads still re-runs or owes a check,
   and a move of the other platform's does not, on rows as they are recorded.

3. THE SLICE, on the shipped ledgers and the real bind writers: another device's bind leaves
   a row standing, a move of what a decision reads does not, every field of the slice moves
   the package's own readers and every field outside it moves nothing, and whatever cannot
   be proved (a state not archived, a spelling no run keys, a file read as text) is compared
   whole under a reason of its own. Existing fixtures hold no archive and no constants, so
   they compare on whole digests, as the first two halves ask.
"""
from __future__ import annotations

import ast
import collections
import copy
import gzip
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import zlib

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for _path in (HERE, REPO_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import ledger_scope  # noqa: E402
import plan_carry as pc  # noqa: E402
import plan_key  # noqa: E402
import rebind_triton_welds  # noqa: E402
import test_plan_carry as carry_fixtures  # noqa: E402
import test_timing_rows as row_fixtures  # noqa: E402
import timing_rows as TR  # noqa: E402
from meep_gpu import (dispatch_preference, fastpath, fastpath_cuda, metal_dispatch,  # noqa: E402
                      metal_runs)

TRITON = "meep_gpu/triton_kernels/fingerprints.json"
CUDA = "meep_gpu/cuda_kernels/fingerprints.json"
CUDA_NARRATIVE = "meep_gpu/cuda_kernels/certification.json"
METAL = "meep_gpu/metal_kernels/fingerprints.json"
NVIDIA_LEDGERS = (TRITON, CUDA, CUDA_NARRATIVE)
APPLE = "applegpu_g13s"

# ------------------------------------------------------------------------ the process ----

#: Installed first in each child: every ledger opened, by repository-relative name.
HOOK = r"""
import json, os, sys
OPENED = {}
def _hook(event, args):
    if event == "open" and args and isinstance(args[0], str):
        head, name = os.path.split(args[0])
        if name in ("fingerprints.json", "certification.json"):
            key = "meep_gpu/" + os.path.basename(head) + "/" + name
            OPENED[key] = OPENED.get(key, 0) + 1
sys.addaudithook(_hook)
"""

METAL_CHILD = HOOK + r"""
import types

class Stand:
    '''Answers everything with itself: no tensor exists and no device is reached.'''
    shape = (1,)
    dtype = "float32"
    ndim = 1
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return self
    def __call__(self, *args, **kwargs):
        return self
    def __getitem__(self, index):
        return self
    def __setitem__(self, index, value):
        pass
    def __iter__(self):
        return iter(())
    def __len__(self):
        return 1
    def __bool__(self):
        return True
    def __int__(self):
        return 1
    def __float__(self):
        return 1.0
    def __index__(self):
        return 1
    def __add__(self, other):
        return self
    __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = __truediv__ = __add__
    def __hash__(self):
        return 1
    def __eq__(self, other):
        return other is self

STAND = Stand()
torch = types.ModuleType("torch")
torch.__version__ = "stand-in"
def _attribute(name):
    if name.startswith("__"):
        raise AttributeError(name)
    return STAND
torch.__getattr__ = _attribute
torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(
    is_built=lambda: True, is_available=lambda: True))
torch.mps = types.SimpleNamespace(compile_shader=lambda source: STAND, synchronize=lambda: None)
sys.modules["torch"] = torch

sys.path.insert(0, os.path.join(os.getcwd(), "parity", "meep_gpu"))
import metal_composition_matrix as matrix
matrix.prepare_environment()
from meep_gpu import fastpath, metal_dispatch
# The environment the ledger certifies, so MEEP_GPU_HARDWARE=certified reaches the composer.
cited = next(run for _, runs in metal_dispatch.cited_environments() for run in runs or ())
metal_dispatch.metal_toolchain = lambda: {
    "importable": True, "version": cited.get("torch"), "built": True, "available": True,
    "compile_shader": True, "metal_frontend": cited.get("metal_frontend"),
    "architecture": cited.get("architecture"), "device_name": "stand-in", "fast_math": None,
    "machine": "arm64"}
tables, decisions = set(), {}
for row in matrix.MATRIX:
    fields, pml = row[1]()
    plan = fastpath.plan_fast_path(fields, pml, getattr(fields, "grid", None), sources=[])
    report = plan.report() if plan is not None else fastpath.last_dispatch_report()
    tables.add(report.get("table"))
    decisions[report.get("decision")] = decisions.get(report.get("decision"), 0) + 1
print(json.dumps({"opened": OPENED, "tables": sorted(map(str, tables)), "decisions": decisions,
                  "cases": len(matrix.MATRIX),
                  "triton_module_imported": "meep_gpu.triton_kernels.launch" in sys.modules}))
"""

NVIDIA_CHILD = HOOK + r"""
import pytest
from meep_gpu import fastpath, fastpath_cuda
from meep_gpu import test_nvidia_fast_default as stand_ins
patch = pytest.MonkeyPatch()
patch.setenv(fastpath.DISPATCH_ENABLE, "1")
plan = stand_ins._two_tables(patch)
report = plan.report() if plan is not None else fastpath.last_dispatch_report()
for module in ("meep_gpu.cuda_kernels.registry", "meep_gpu.cuda_kernels.arms",
               "meep_gpu.triton_kernels.launch", "meep_gpu.triton_kernels"):
    __import__(module)
print(json.dumps({"opened": OPENED, "table": report.get("table"),
                  "decision": report.get("decision"),
                  "tables_dispatched": (report.get("composition") or {}).get(
                      "tables_dispatched")}))
"""


def child(source, tmp_path, **environment):
    """Run ``source`` in a fresh process at the repository root; its last stdout line."""
    env = {name: value for name, value in os.environ.items()
           if not name.startswith(("MEEP_GPU_", "CUPY_", "TRITON_", "PYTORCH_"))}
    env.update(environment, PYTHONPATH=REPO_ROOT, PYTHONDONTWRITEBYTECODE="1",
               CUPY_CACHE_DIR=str(tmp_path / "cupy-cache"))
    flags = ["-s"] if sys.flags.no_user_site else []
    done = subprocess.run([sys.executable, *flags, "-c", source], cwd=REPO_ROOT, env=env,
                          text=True, capture_output=True, timeout=300, check=False)
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("scope", ["default", "certified", "any"])
def test_the_metal_dispatch_path_opens_no_nvidia_ledger(tmp_path, scope):
    environment = {} if scope == "default" else {"MEEP_GPU_HARDWARE": scope}
    seen = child(METAL_CHILD, tmp_path, **environment)
    assert seen["tables"] == ["metal"], seen
    # Not vacuous: it read its own ledger, planned, and held the module a bench keys on.
    assert seen["opened"].get(METAL, 0) >= 1, seen
    assert seen["triton_module_imported"] is True, seen
    assert seen["decisions"].get("dispatched", 0) > 0, seen
    assert not set(seen["opened"]) & set(NVIDIA_LEDGERS), seen


@pytest.mark.parametrize("scope", ["default", "certified", "any"])
def test_the_nvidia_dispatch_path_opens_no_metal_ledger(tmp_path, scope):
    environment = {} if scope == "default" else {"MEEP_GPU_HARDWARE": scope}
    seen = child(NVIDIA_CHILD, tmp_path, **environment)
    assert seen["decision"] == "dispatched", seen
    assert seen["tables_dispatched"] == ["cuda", "triton"], seen
    assert seen["opened"].get(TRITON, 0) >= 1 and seen["opened"].get(CUDA, 0) >= 1, seen
    assert METAL not in seen["opened"], seen
    assert CUDA_NARRATIVE not in seen["opened"], seen


# ---------------------------------------------------------------------- the one fact ----

@pytest.mark.parametrize("device, read, not_read", [
    ("cc86", NVIDIA_LEDGERS, (METAL,)),
    ("8.6", NVIDIA_LEDGERS, (METAL,)),
    ("cc90", NVIDIA_LEDGERS, (METAL,)),
    (APPLE, (METAL,), NVIDIA_LEDGERS),
    ("applegpu_g15p", (METAL,), NVIDIA_LEDGERS),
])
def test_a_device_reads_its_own_platform_s_ledgers_only(device, read, not_read):
    assert all(ledger_scope.reads_ledger(name, device) for name in read)
    assert not any(ledger_scope.reads_ledger(name, device) for name in not_read)


@pytest.mark.parametrize("device", [None, "", "some future GPU", "cc", "applegpu"])
def test_a_device_that_cannot_be_placed_is_compared_on_every_ledger(device):
    assert all(ledger_scope.reads_ledger(name, device)
               for name in NVIDIA_LEDGERS + (METAL,))


@pytest.mark.parametrize("device, name, platform", [
    (APPLE, TRITON, "nvidia"), (APPLE, CUDA, "nvidia"), ("cc86", METAL, "metal")])
def test_a_recorded_platform_outranks_the_device_label(device, name, platform):
    assert not ledger_scope.reads_ledger(name, device, (None,))
    assert ledger_scope.reads_ledger(name, device, (platform,))


def test_a_name_that_is_no_table_s_ledger_is_never_set_aside():
    for name in ("<no ledgers statement>", "meep_gpu/fastpath.py", "fingerprints.json",
                 "meep_gpu/other_kernels/fingerprints.json",
                 "meep_gpu/metal_kernels/timing_applegpu_g13s.json",
                 "meep_gpu/triton_kernels/timing_cc86.json"):
        assert ledger_scope.reads_ledger(name, APPLE)
        assert ledger_scope.reads_ledger(name, "cc86")


def test_every_ledger_a_row_records_belongs_to_a_platform():
    assert TR.DATA_FILES == frozenset(NVIDIA_LEDGERS + (METAL,))
    assert all(ledger_scope.reads_ledger(name, "cc86") != ledger_scope.reads_ledger(name, APPLE)
               for name in TR.DATA_FILES)


# ----------------------------------------------------------- plan_carry: planted moves ----

def plant(root, name, value="moved"):
    """A move of ``name``'s view: an entry other than ``driver_dispatch``."""
    path = root / name
    document = json.loads(path.read_text()) if path.exists() else {}
    document["planted"] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))


@pytest.fixture()
def tree(tmp_path):
    root = tmp_path / "repo"
    for name in carry_fixtures.CODE:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(f"# {name}\n")
    for name in (TRITON, CUDA, METAL):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(json.dumps({"arm": {"x": 1}, "driver_dispatch": {}}))
    (root / "meep_gpu/triton_kernels/timing_cc86.json").write_text('{"record_sha256": "a"}')
    return root


def on_device(payload, device):
    """A recorded run of ``device``, stamped as each platform's recorders stamp it."""
    if device.startswith("cc"):
        payload["stamped_device"] = {"compute_capability": f"{device[2]}.{device[3:]}"}
        return payload
    payload["stamped_device"] = {"unreadable": "cupy is not imported in this process"}
    payload["apple_gpu"] = {"architecture": device}
    if "table" in payload:  # an L3 row: its table and plan key name Metal
        payload["table"] = "metal"
        payload[plan_key.KEY_FIELD] = plan_key.key(
            {**carry_fixtures.report("fused pair D"), "table": "metal"}, 1)
    else:  # a route leg of the Metal gate names no backend
        payload["gate"] = "dispatch_metal_route"
        payload["provenance"].pop("backend")
    return payload


def route_leg(root, device, **kwargs):
    return on_device(carry_fixtures.leg(root, **kwargs), device)


def l3_row(root, device, **kwargs):
    run = on_device(carry_fixtures.l3_row(root), device)
    run.update(kwargs)
    return run


def route_pass(root, device):
    return on_device(carry_fixtures.plan_pass(root), device)


def l3_plan_row(root, device):
    return l3_row(root, device, plan_only=True)


def judged(kind, root, device, run, passed=None):
    judge = pc.judge_route_leg if kind == "route" else pc.judge_l3_row
    return judge(run, root, plan_key.plan_inputs(str(root)), device, passed)


RUNS = {"route": route_leg, "l3": l3_row}
PASSES = {"route": route_pass, "l3": l3_plan_row}
DIRECTIONS = [("cc86", TRITON), ("cc86", CUDA), (APPLE, METAL)]
OTHERS = [("cc86", METAL), (APPLE, TRITON), (APPLE, CUDA)]


@pytest.mark.parametrize("kind", sorted(RUNS))
@pytest.mark.parametrize("device, name", DIRECTIONS)
def test_a_moved_ledger_the_device_reads_still_re_runs(tree, kind, device, name):
    run = RUNS[kind](tree, device)
    assert judged(kind, tree, device, run)["state"] == pc.CARRY_DIGEST
    plant(tree, name)
    verdict = judged(kind, tree, device, run)
    assert verdict["state"] == pc.RERUN_INPUTS and verdict["moved"] == [name], verdict


@pytest.mark.parametrize("kind", sorted(RUNS))
@pytest.mark.parametrize("device, name", OTHERS)
def test_a_moved_ledger_of_the_other_platform_carries_the_run_as_recorded(
        tree, kind, device, name):
    run = RUNS[kind](tree, device)
    recorded = copy.deepcopy(run)
    plant(tree, name)
    assert judged(kind, tree, device, run)["state"] == pc.CARRY_DIGEST
    assert run == recorded, "a run is read as recorded and never rewritten"
    assert plan_key.inputs_moved(run[plan_key.INPUTS_FIELD],
                                 plan_key.plan_inputs(str(tree))) == {"ledgers": [name]}, (
        "the recorder's own statement still names every table's view")


@pytest.mark.parametrize("device, table, name", [
    ("cc86", "metal", METAL), (APPLE, "triton", TRITON), (APPLE, "cuda", CUDA)])
def test_an_l3_row_of_the_other_platform_s_table_is_compared_on_that_ledger(
        tree, device, table, name):
    run = l3_row(tree, device, table=table)
    assert judged("l3", tree, device, run)["state"] == pc.CARRY_DIGEST
    plant(tree, name)
    verdict = judged("l3", tree, device, run)
    assert verdict["state"] == pc.RERUN_INPUTS and verdict["moved"] == [name], verdict


@pytest.mark.parametrize("device, payload, name", [
    ("cc86", {"gate": "dispatch_metal_route"}, METAL),
    (APPLE, {"provenance": {"backend": "triton"}}, TRITON),
    (APPLE, {"provenance": {"backend": "cuda"}}, CUDA)])
def test_a_route_leg_of_the_other_platform_s_table_is_compared_on_that_ledger(
        tree, device, payload, name):
    run = route_leg(tree, device)
    for field, value in payload.items():
        run[field] = {**run[field], **value} if isinstance(value, dict) and field in run else value
    plant(tree, name)
    verdict = judged("route", tree, device, run)
    assert verdict["state"] == pc.RERUN_INPUTS and verdict["moved"] == [name], verdict


@pytest.mark.parametrize("device, platform, name", [
    ("cc86", "metal", METAL), (APPLE, "nvidia", TRITON)])
def test_a_recorded_plan_of_the_other_platform_is_compared_on_that_ledger(
        tree, device, platform, name):
    run = l3_row(tree, device)
    run.pop("table", None)
    run[plan_key.KEY_FIELD] = plan_key.key(
        {"table": "metal" if platform == "metal" else "triton", "decision": "dispatched",
         "arms": {}}, 1)
    assert run[plan_key.KEY_FIELD]["plan"]["selected_plan"]["platform"] == platform
    assert pc.platforms_recorded(run) == [platform]
    plant(tree, name)
    verdict = judged("l3", tree, device, run)
    assert verdict["state"] == pc.RERUN_INPUTS and verdict["moved"] == [name], verdict


@pytest.mark.parametrize("kind", sorted(RUNS))
def test_a_move_on_both_platforms_names_only_the_one_the_device_reads(tree, kind):
    run = RUNS[kind](tree, APPLE)
    for name in (TRITON, CUDA, METAL):
        plant(tree, name)
    assert judged(kind, tree, APPLE, run)["moved"] == [METAL]


@pytest.mark.parametrize("kind", sorted(RUNS))
@pytest.mark.parametrize("device", ["cc86", APPLE])
def test_a_run_whose_statement_holds_no_ledger_map_is_never_carried(tree, kind, device):
    """Nothing of the device's own: the other platform's names must not hide the gap."""
    own = [TRITON, CUDA] if device == "cc86" else [METAL]
    run = RUNS[kind](tree, device)
    for name in own:
        (tree / name).unlink()
    run[plan_key.INPUTS_FIELD]["ledgers"] = "not a map"
    assert judged(kind, tree, device, run)["state"] == pc.RERUN_INPUTS


@pytest.mark.parametrize("kind", sorted(RUNS))
@pytest.mark.parametrize("device, name", DIRECTIONS)
def test_a_run_that_did_not_state_a_ledger_its_device_reads_re_runs(tree, kind, device, name):
    run = RUNS[kind](tree, device)
    del run[plan_key.INPUTS_FIELD]["ledgers"][name]
    assert judged(kind, tree, device, run)["state"] == pc.RERUN_INPUTS


@pytest.mark.parametrize("kind", sorted(RUNS))
@pytest.mark.parametrize("device, name", OTHERS)
def test_a_plan_pass_stands_across_a_move_of_the_other_platform_s_ledger(
        tree, kind, device, name):
    run = RUNS[kind](tree, device)
    carry_fixtures.move_record(tree)
    passed = PASSES[kind](tree, device)
    plant(tree, name)
    verdict = judged(kind, tree, device, run, passed)
    assert verdict["state"] == pc.CARRY_PLAN, verdict


@pytest.mark.parametrize("kind", sorted(RUNS))
@pytest.mark.parametrize("device, name", DIRECTIONS)
def test_a_plan_pass_on_another_view_of_a_ledger_the_device_reads_cannot_stand(
        tree, kind, device, name):
    passed = PASSES[kind](tree, device)  # before the move: it read the old view
    plant(tree, name)
    run = RUNS[kind](tree, device)       # after it: the run itself holds
    carry_fixtures.move_record(tree)
    verdict = judged(kind, tree, device, run, passed)
    assert verdict["state"] == pc.OWED_PLAN, verdict
    assert "other planner data" in verdict["why"]


@pytest.mark.parametrize("kind", sorted(RUNS))
def test_a_moved_timing_record_is_owed_a_plan_pass_on_either_platform(tree, kind):
    """Only the ledgers are narrowed: the timing records are compared as recorded."""
    for device in ("cc86", APPLE):
        run = RUNS[kind](tree, device)
        (tree / "meep_gpu/metal_kernels/timing_applegpu_g13s.json").write_text(device)
        assert judged(kind, tree, device, run)["state"] == pc.OWED_PLAN
        (tree / "meep_gpu/metal_kernels/timing_applegpu_g13s.json").unlink()


# ---------------------------------------------------------- timing rows: planted moves ----

METAL_ARM = ("step_D", "metal", "fused electric D/E pair", "metal_electric_gate")


def recorded_ledgers(root, names):
    return {name: TR.artifact_drift._sha256(pathlib.Path(root) / name)  # noqa: SLF001
            for name in names}


@pytest.fixture()
def release(tmp_path):
    root = row_fixtures.release_tree(tmp_path)
    for name in (CUDA, METAL):
        path = os.path.join(root, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as handle:
            json.dump({}, handle)
    return root


def timed_row(root, architecture, names, cells=512000, launched=True):
    arms = () if not launched else (
        (METAL_ARM,) if architecture.startswith("applegpu") else (row_fixtures.ELECTRIC,))
    row = row_fixtures.ladder_row(cells, "s1", "2026-10-20T10:00:00Z", arms=arms,
                                  architecture=architecture)
    if not launched:
        row["launch_counters"] = []
    row["processes"][0][TR.DATA_KEY] = recorded_ledgers(root, names)
    return row


def rewrite(root, name):
    with open(os.path.join(root, name), "w") as handle:
        json.dump({"new_run": {"source_sha256": {}}}, handle)


ALL_FOUR = NVIDIA_LEDGERS + (METAL,)
#: (architecture, the ledgers the process recorded, a move of one its device reads).
OWN_MOVES = [(APPLE, ALL_FOUR, METAL), ("8.6", ALL_FOUR, TRITON), ("8.6", ALL_FOUR, CUDA),
             ("cc86", ALL_FOUR, TRITON), ("8.6", NVIDIA_LEDGERS, TRITON)]
OTHER_MOVES = [(APPLE, ALL_FOUR, TRITON), (APPLE, ALL_FOUR, CUDA),
               (APPLE, ALL_FOUR, CUDA_NARRATIVE), ("8.6", ALL_FOUR, METAL),
               ("cc86", ALL_FOUR, METAL)]


@pytest.mark.parametrize("architecture, names, moved", OWN_MOVES)
def test_a_ledger_the_row_reads_still_owes_a_check_or_a_re_time(
        tmp_path, release, architecture, names, moved):
    row = timed_row(release, architecture, names)
    blind = timed_row(release, architecture, names, cells=2097152, launched=False)
    records = TR.load_store(row_fixtures.store_of(tmp_path, [row, blind]))
    assert TR.bill(records, [moved], release) == {"retime": [blind["key"]],
                                                  "check": [row["key"]], "untouched": []}
    rewrite(release, moved)
    selection = TR.select(records, release)
    assert selection[row["key"]]["owed"] == "a dispatch check"
    assert selection[row["key"]]["assessments"][0]["status"] == "check_owed"
    assert moved in selection[row["key"]]["assessments"][0]["exempt"]
    assert selection[blind["key"]]["owed"] == "a re-time"


@pytest.mark.parametrize("architecture, names, moved", OTHER_MOVES)
def test_a_ledger_of_the_other_platform_leaves_the_row_valid(
        tmp_path, release, architecture, names, moved):
    row = timed_row(release, architecture, names)
    blind = timed_row(release, architecture, names, cells=2097152, launched=False)
    records = TR.load_store(row_fixtures.store_of(tmp_path, [row, blind]))
    assert TR.bill(records, [moved], release) == {
        "retime": [], "check": [], "untouched": sorted([row["key"], blind["key"]])}
    rewrite(release, moved)
    selection = TR.select(records, release)
    assert selection[row["key"]]["owed"] == "nothing"
    assessment = selection[row["key"]]["assessments"][0]
    assert assessment["status"] == "valid" and assessment["moved"] == {}
    assert selection[blind["key"]]["assessments"][0]["status"] == "valid"
    # the digest stays in the record, and a row recording no digest at all is still refused
    assert moved in row["processes"][0][TR.DATA_KEY]


def test_a_row_recording_no_ledger_digests_is_still_refused(release):
    row = timed_row(release, APPLE, ALL_FOUR)
    row["processes"][0].pop(TR.DATA_KEY)
    assert any("no ledger digests" in why for why in TR.refusals(row))


@pytest.mark.parametrize("architecture, names", [
    (APPLE, NVIDIA_LEDGERS), ("8.6", (METAL,)), ("cc86", (METAL,))])
def test_a_row_recording_no_ledger_of_its_own_platform_is_refused(
        release, architecture, names):
    row = timed_row(release, architecture, names)
    assert any("no ledger of its own platform" in why for why in TR.refusals(row))


@pytest.mark.parametrize("architecture, names", [
    (APPLE, (METAL,)), (APPLE, ALL_FOUR), ("8.6", (TRITON,)), ("8.6", (CUDA,)), ("8.6", ALL_FOUR),
    ("some future GPU", ())])
def test_a_row_recording_a_ledger_of_its_own_platform_is_not_refused_for_it(
        release, architecture, names):
    row = timed_row(release, architecture, names)
    assert not any("no ledger of its own platform" in why for why in TR.refusals(row))


@pytest.mark.parametrize("architecture, arm, backend, moved", [
    (APPLE, "cuda:fused magnetic pair", "cuda", CUDA),
    ("8.6", METAL_ARM, "metal", METAL)])
def test_a_row_that_launched_the_other_platform_s_table_owes_on_its_ledger(
        tmp_path, release, architecture, arm, backend, moved):
    """The launch counters outrank the architecture label: the row did read that ledger."""
    row = timed_row(release, architecture, ALL_FOUR)
    for block in row["launch_counters"]:
        for counter in block["slots"].values():
            counter.update(arm=arm, backend=backend)
    records = TR.load_store(row_fixtures.store_of(tmp_path, [row]))
    assert TR.bill(records, [moved], release)["check"] == [row["key"]]
    rewrite(release, moved)
    assert moved in TR.moved_files(row, release)


def test_a_row_of_an_unplaced_architecture_is_compared_on_every_ledger(tmp_path, release):
    row = timed_row(release, "some future GPU", ALL_FOUR)
    records = TR.load_store(row_fixtures.store_of(tmp_path, [row]))
    for name in ALL_FOUR:
        assert TR.bill(records, [name], release)["check"] == [row["key"]]


def test_a_dispatch_check_witnesses_across_a_move_of_the_other_platform_s_ledger(
        tmp_path, release):
    """The check itself recorded the NVIDIA digests, and they moved after it: it stands."""
    row = timed_row(release, APPLE, ALL_FOUR)
    rewrite(release, METAL)  # the row owes a check: its own ledger moved
    check = timed_row(release, APPLE, ALL_FOUR)
    check.update(kind="check", key=TR.check_key(row["key"]))
    check["session"] = {"id": "s2", "started_utc": "2026-10-21T10:00:00Z",
                        "ended_utc": "2026-10-21T10:00:00Z"}
    records = TR.load_store(row_fixtures.store_of(tmp_path, [row, check]))
    rewrite(release, TRITON)
    selection = TR.select(records, release)
    assert selection[row["key"]]["assessments"][0]["status"] == "valid", (
        selection[row["key"]]["assessments"][0])


# --------------------------------------------------------- the shipped ledgers: helpers ----

STATES_ROOT = os.path.join(REPO_ROOT, ledger_scope.STATES_DIR)
TABLE_LEDGER = dict(ledger_scope.LEDGER_OF_TABLE)
G13 = APPLE
G15 = "applegpu_g15s"
#: The Apple M4's architecture, certified beside G13 on the same bytes.
G16 = "applegpu_g16g"
NVIDIA_FILES = ("meep_gpu/fastpath.py", "meep_gpu/fastpath_cuda.py")
METAL_FILE = "meep_gpu/metal_dispatch.py"
TIMING_NAMES = ("meep_gpu/triton_kernels/timing_cc86.json",
                "meep_gpu/cuda_kernels/timing_cc86.json",
                "meep_gpu/metal_kernels/timing_applegpu_g13s.json")
#: Linked into a judged tree: read by the comparison, written by no test.
LINKED_GLOBS = ("meep_gpu/*.py", "meep_gpu/*_kernels/*.py", "meep_gpu/*_kernels/timing_*.json",
                "meep_gpu/*_kernels/*.json")


def _link(root, name, target=None):
    path = pathlib.Path(root) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(target or os.path.join(REPO_ROOT, name))


@pytest.fixture()
def shipped(tmp_path):
    """A judged tree holding the shipped ledgers, constants, kernels, timing records and
    archive, by link. A test replaces what it moves with :func:`write_json`."""
    root = tmp_path / "repo"
    for pattern in LINKED_GLOBS:
        for path in sorted(pathlib.Path(REPO_ROOT).glob(pattern)):
            rel = path.relative_to(REPO_ROOT).as_posix()
            if not (root / rel).exists():
                _link(root, rel)
    _link(root, ledger_scope.STATES_DIR)
    _link(root, carry_fixtures.CODE[1])
    return root


def read_ledger(root, table):
    return json.loads((pathlib.Path(root) / TABLE_LEDGER[table]).read_text())


def write_json(root, name, document):
    path = pathlib.Path(root) / name
    if path.is_symlink():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return path


def own_archive(root):
    """Replace the link to the shipped archive with a copy this test may write to."""
    link = pathlib.Path(root) / ledger_scope.STATES_DIR
    if link.is_symlink():
        link.unlink()
        shutil.copytree(STATES_ROOT, link)
    return link


def sha_of(root, name):
    return TR.artifact_drift._sha256(pathlib.Path(root) / name)  # noqa: SLF001


def closure_of(root, device):
    files = (METAL_FILE,) if str(device).startswith("applegpu") else NVIDIA_FILES
    return {name: sha_of(root, name) for name in files}


def shipped_l3(root, device="cc86", triton="3.1.0", **extra):
    run = on_device(carry_fixtures.l3_row(root), device)
    run["imported_source_sha256"].update(closure_of(root, device))
    run[plan_key.TOOLCHAIN_FIELD] = dict(run[plan_key.TOOLCHAIN_FIELD], triton=triton)
    run.update(extra)
    return run


def shipped_route(root, device="cc86", triton="3.1.0"):
    run = on_device(carry_fixtures.leg(root), device)
    run["imported_source_sha256"].update(closure_of(root, device))
    run["provenance"]["triton"] = triton
    return run


SHIPPED_RUNS = {"l3": shipped_l3, "route": shipped_route}


def shipped_timing(root, architecture="8.6", *, kind="ladder", key=None,
                   ended="2026-10-20T10:00:00Z", session="s1", triton="3.1.0"):
    """A timing row of ``architecture``, cut on the tree as it stands."""
    apple = architecture.startswith("applegpu")
    row = row_fixtures.ladder_row(512000, session, ended, architecture=architecture,
                                  arms=(METAL_ARM,) if apple else (row_fixtures.ELECTRIC,))
    process = row["processes"][0]
    process[TR.IMPORTED_KEY] = closure_of(root, architecture)
    process[TR.DATA_KEY] = {name: sha_of(root, name) for name in
                            ((METAL,) if apple else NVIDIA_LEDGERS)}
    process["imported_libraries"] = {} if apple else {"triton": triton}
    if kind == "check":
        row.update(kind="check", key=TR.check_key(key))
    return row


def run_template(entry, capability):
    run = dict(entry["runs"][capability])
    run.pop("bound_sha256", None)
    return run


def bind_nvidia(root, capability, *, source="8.6", tables=("triton", "cuda"), stack=True,
                moved_pin=None, supersede=(), host=None):
    """A bind of ``capability`` through the real writer: every run-bearing entry gets a run,
    copied from ``source``'s, and its own stack. ``moved_pin`` edits one pin first, as a
    round on moved bytes does."""
    for table in tables:
        ledger = read_ledger(root, table)
        for entry in ledger.values():
            if not (isinstance(entry, dict) and isinstance(entry.get("runs"), dict)
                    and source in entry["runs"] and any(f in entry for f in fastpath.BOUND_FIELDS)
                    and isinstance(entry.get("source_sha256"), dict)):
                continue
            before = fastpath.bound_digest(entry)
            if moved_pin and moved_pin in entry["source_sha256"]:
                entry["source_sha256"][moved_pin] = "0" * 64
            run = run_template(entry, source)
            if host:
                run["host"] = host
            fastpath.bind_capability(entry, bound_before=before, capability=capability, run=run,
                                     run_fields=tuple(run), supersede=supersede)
        if table == "triton" and stack:
            records = ledger["certified_nvidia_stack"]["by_capability"]
            records.setdefault(capability, copy.deepcopy(records["8.6"]))
        write_json(root, TABLE_LEDGER[table], ledger)


def bind_metal(root, architecture, *, source=G13, torch=None, moved_pin=None, supersede=()):
    ledger = read_ledger(root, "metal")
    for entry in ledger.values():
        if not (isinstance(entry, dict) and isinstance(entry.get("runs"), dict)
                and source in entry["runs"] and isinstance(entry.get("source_sha256"), dict)):
            continue
        before = fastpath.bound_digest(entry)
        if moved_pin and moved_pin in entry["source_sha256"]:
            entry["source_sha256"][moved_pin] = "0" * 64
        run = run_template(entry, source)
        if torch:
            run["torch"] = torch
        metal_runs.bind_architecture(entry, bound_before=before, architecture=architecture,
                                     run=run, run_fields=tuple(run), supersede=supersede)
    write_json(root, TABLE_LEDGER["metal"], ledger)


def judged_on(kind, root, device, run, passed=None):
    return judged(kind, pathlib.Path(root), device, run, passed)


def only_why(verdict, name):
    return (verdict.get("compared_whole") or {}).get(name, {}).get("why")


def cited_of(root):
    return ledger_scope.cited_welds(root)


# ----------------------------------------------------------- binds through the real writers ----

@pytest.mark.parametrize("kind", sorted(SHIPPED_RUNS))
def test_a_9_0_bind_on_unchanged_bytes_leaves_8_6_rows(shipped, kind):
    run = SHIPPED_RUNS[kind](shipped)
    bind_nvidia(shipped, "9.0")
    verdict = judged_on(kind, shipped, "cc86", run)
    assert verdict["state"] == pc.CARRY_SLICE, verdict
    aside = verdict["proof"]["set_aside"]
    assert set(aside) == {TRITON, CUDA}
    assert all(entry["why"] == "slice_equal" and entry["slice_sha256"] for entry in aside.values())


def test_a_9_0_row_stands_when_8_6_binds_beside_it_and_one_from_before_9_0_does_not(shipped):
    before = shipped_l3(shipped, "cc90")
    bind_nvidia(shipped, "9.0")
    for table in ("triton", "cuda"):
        ledger_scope.archive_state(own_archive(shipped).parents[2], table,
                                   shipped / TABLE_LEDGER[table])
    run = shipped_l3(shipped, "cc90")
    bind_nvidia(shipped, "8.6", source="9.0", stack=False, host="the GPU host: another card")
    assert judged_on("l3", shipped, "cc90", run)["state"] == pc.CARRY_SLICE
    verdict = judged_on("l3", shipped, "cc90", before)
    assert verdict["state"] == pc.RERUN_INPUTS
    assert only_why(verdict, TRITON) == "device_names_no_run"


def test_an_8_9_bind_after_9_0_leaves_both(shipped):
    bind_nvidia(shipped, "9.0")
    for table in ("triton", "cuda"):
        ledger_scope.archive_state(own_archive(shipped).parents[2], table,
                                   shipped / TABLE_LEDGER[table])
    on_90 = shipped_l3(shipped, "cc90")
    on_86 = shipped_l3(shipped, "cc86")
    bind_nvidia(shipped, "8.9")
    assert judged_on("l3", shipped, "cc90", on_90)["state"] == pc.CARRY_SLICE
    assert judged_on("l3", shipped, "cc86", on_86)["state"] == pc.CARRY_SLICE


@pytest.mark.parametrize("kind", sorted(SHIPPED_RUNS))
@pytest.mark.parametrize("torch", [None, "9.9.9"])
def test_a_g15s_bind_leaves_g13s_rows(shipped, kind, torch):
    run = SHIPPED_RUNS[kind](shipped, G13)
    bind_metal(shipped, G15, torch=torch)
    verdict = judged_on(kind, shipped, G13, run)
    assert verdict["state"] == pc.CARRY_SLICE, verdict
    assert set(verdict["proof"]["set_aside"]) == {METAL}


@pytest.mark.parametrize("torch", [None, "9.9.9"])
def test_a_g13s_write_leaves_g15s_rows(shipped, torch):
    bind_metal(shipped, G15)
    ledger_scope.archive_state(own_archive(shipped).parents[2], "metal",
                               shipped / TABLE_LEDGER["metal"])
    run = shipped_l3(shipped, G15)
    ledger = read_ledger(shipped, "metal")
    for entry in ledger.values():
        if isinstance(entry, dict) and G13 in (entry.get("runs") or {}):
            entry["runs"][G13]["recorded_utc"] = "moved"
            if torch:
                entry["runs"][G13]["torch"] = torch
    write_json(shipped, TABLE_LEDGER["metal"], ledger)
    assert judged_on("l3", shipped, G15, run)["state"] == pc.CARRY_SLICE


@pytest.mark.parametrize("kind", sorted(SHIPPED_RUNS))
def test_a_superseded_bind_voids_the_other_generation_on_nvidia(shipped, kind):
    run = SHIPPED_RUNS[kind](shipped)
    # Supersedes 8.9 beside 8.6: the moved bytes strand both live runs, which must be named.
    bind_nvidia(shipped, "9.0", moved_pin="meep_gpu/fastpath.py", supersede=("8.6", "8.9"))
    verdict = judged_on(kind, shipped, "cc86", run)
    assert verdict["state"] == pc.RERUN_INPUTS, verdict
    assert only_why(verdict, TRITON) == "slice_moved"
    assert "meep_gpu/fastpath.py" in verdict["compared_whole"][TRITON]["pins_moved"]


@pytest.mark.parametrize("kind", sorted(SHIPPED_RUNS))
def test_a_superseded_bind_voids_the_other_generation_on_metal(shipped, kind):
    run = SHIPPED_RUNS[kind](shipped, G13)
    # Supersedes G16 beside G13: the moved bytes strand both live runs, which must be named.
    bind_metal(shipped, G15, moved_pin="meep_gpu/fastpath.py", supersede=(G13, G16))
    verdict = judged_on(kind, shipped, G13, run)
    assert verdict["state"] == pc.RERUN_INPUTS, verdict
    assert only_why(verdict, METAL) == "slice_moved"


def test_a_harness_only_void_reads_as_one(shipped):
    run = shipped_l3(shipped)
    pin = "parity/meep_gpu/probe_fused_kernel_bit_identity.py"
    # Supersedes 8.9 beside 8.6: the moved bytes strand both live runs, which must be named.
    bind_nvidia(shipped, "9.0", tables=("cuda",), moved_pin=pin, supersede=("8.6", "8.9"))
    verdict = judged_on("l3", shipped, "cc86", run)
    assert verdict["state"] == pc.RERUN_INPUTS
    entry = verdict["compared_whole"][CUDA]
    assert entry["pins_moved"] == [pin] and entry["harness_only"] is True


def test_a_check_cut_before_a_bind_stays_a_witness(tmp_path, shipped):
    """A dispatch check compares launched arms; another device's bind moves nothing the
    check or its row read, so neither owes anything more."""
    row = shipped_timing(shipped)
    check = shipped_timing(shipped, kind="check", key=row["key"], session="s2",
                           ended="2026-10-21T10:00:00Z")
    records = TR.load_store(row_fixtures.store_of(tmp_path, [row, check]))
    bind_nvidia(shipped, "9.0")
    assessment = TR.select(records, str(shipped))[row["key"]]["assessments"][0]
    assert assessment["status"] == "valid" and assessment["moved"] == {}, assessment
    assert set(assessment["set_aside"]) >= {TRITON, CUDA}
    # and the file moved, so a whole-file compare would have owed a check
    assert sha_of(shipped, TRITON) != row["processes"][0][TR.DATA_KEY][TRITON]
    assert not TR.moved_files(check, str(shipped))


def test_the_unread_ledger_owes_a_placed_row_nothing(tmp_path, shipped):
    """``fastpath_cuda.certification`` has no caller on a dispatch path: the audit hook above
    sees it opened 0 times on either platform's plan."""
    row = shipped_timing(shipped)
    records = TR.load_store(row_fixtures.store_of(tmp_path, [row]))
    assert TR.bill(records, [CUDA_NARRATIVE], str(shipped))["untouched"] == [row["key"]]
    write_json(shipped, CUDA_NARRATIVE, {"a_block": {"moved": 1}})
    assert CUDA_NARRATIVE not in TR.moved_files(row, str(shipped))
    assessment = TR.select(records, str(shipped))[row["key"]]["assessments"][0]
    assert assessment["status"] == "valid"
    assert assessment["set_aside"][CUDA_NARRATIVE]["why"] == "unread"


def test_a_row_cut_on_a_state_the_device_moved_in_owes_a_check(tmp_path, shipped):
    row = shipped_timing(shipped)
    records = TR.load_store(row_fixtures.store_of(tmp_path, [row]))
    # Supersedes 8.9: re-binding 8.6 on moved bytes strands 8.9's live run, which must be named.
    bind_nvidia(shipped, "8.6", moved_pin="meep_gpu/cuda_kernels/step_curl_kernels.py",
                tables=("cuda",), supersede=("8.9",))
    selection = TR.select(records, str(shipped))[row["key"]]
    assert selection["owed"] == "a dispatch check"
    assert selection["assessments"][0]["compared_whole"][CUDA]["why"] == "slice_moved"


def test_a_bind_of_the_row_s_own_generation_that_keeps_every_value_stands(shipped):
    """Own-capability writes outside the slice void nothing: the tools ask what the device
    reads, whichever device wrote it."""
    run = shipped_l3(shipped)
    ledger = read_ledger(shipped, "triton")
    for entry in ledger.values():
        if isinstance(entry, dict) and "8.6" in (entry.get("runs") or {}):
            entry["runs"]["8.6"]["host"] = "the GPU host: another card"
            entry["array_path_oracle"] = {"moved": 1}
    write_json(shipped, TRITON, ledger)
    assert judged_on("l3", shipped, "cc86", run)["state"] == pc.CARRY_SLICE


@pytest.mark.parametrize("driver_added, policies, state", [
    (False, ("flush", "keep"), pc.CARRY_SLICE), (False, ("keep",), pc.RERUN_INPUTS),
    (True, ("flush", "keep"), pc.RERUN_INPUTS)])
def test_a_second_8_6_card_stands_only_when_it_changes_no_value(
        shipped, driver_added, policies, state):
    """A second card with both policies and no new stack value carries. Keep only, or a
    driver added to the stack, moves the capability's own slice, and the row re-runs: the
    record names no driver, so it cannot say it ran on an old one. That over-void stays."""
    run = shipped_l3(shipped)
    ledger = read_ledger(shipped, "triton")
    for entry in ledger.values():
        own = (entry.get("runs") or {}).get("8.6") if isinstance(entry, dict) else None
        if isinstance(own, dict) and isinstance(own.get("certified_policies"), dict):
            own["host"] = "the GPU host: another card"
            held = own["certified_policies"]
            own["certified_policies"] = {k: v for k, v in held.items() if k in policies}
    if driver_added:
        ledger["certified_nvidia_stack"]["by_capability"]["8.6"]["stack"]["cuda_driver"].append(
            "99999")
    write_json(shipped, TRITON, ledger)
    assert judged_on("l3", shipped, "cc86", run)["state"] == state


# -------------------------------------------------------------- one mutation per slice field ----

def _keep_policy(ledger, key):
    ledger[key]["runs"]["8.6"]["certified_policies"].pop("flush")


def _first_with(ledger, cited, field):
    return next(key for key in cited if field in ledger[key])


def _bound_mutation(field):
    def apply(ledgers, cited):
        key = _first_with(ledgers["triton"], cited["triton"], field)
        value = ledgers["triton"][key][field]
        ledgers["triton"][key][field] = (dict(value, moved="x") if isinstance(value, dict)
                                         else f"{value}-moved")
    return apply


def _axis_mutation(axis):
    def apply(ledgers, cited):
        ledgers["triton"]["certified_nvidia_stack"]["by_capability"]["8.6"]["stack"][axis] = ["x"]
    return apply


def _own_policy_weld(ledgers, cited):
    def holds_flush(key):
        return "flush" in (ledgers["triton"][key]["runs"]["8.6"].get("certified_policies") or {})
    return next(key for key in cited["triton"] if holds_flush(key))


MUST_MOVE = {
    **{f"bound {field}": ("triton", "8.6", _bound_mutation(field))
       for field in ledger_scope.BOUND_FIELDS},
    "run_bound": ("cuda", "8.6", lambda d, c: d["cuda"][c["cuda"][0]]["runs"]["8.6"].__setitem__(
        "bound_sha256", "moved")),
    "policy key": ("triton", "8.6", lambda d, c: _keep_policy(d["triton"], _own_policy_weld(d, c))),
    "own run deleted": ("cuda", "8.6", lambda d, c: d["cuda"][c["cuda"][0]]["runs"].pop("8.6")),
    "weld deleted": ("cuda", "8.6", lambda d, c: d["cuda"].pop(c["cuda"][0])),
    **{f"stack {axis}": ("triton", "8.6", _axis_mutation(axis))
       for axis in ledger_scope.NVIDIA_STACK_AXES},
    "stack record deleted": ("triton", "8.6", lambda d, c: d["triton"][
        "certified_nvidia_stack"]["by_capability"].pop("8.6")),
    "list membership": ("triton", "8.6", lambda d, c: d["triton"].__setitem__(
        "validated_triton_versions", [])),
    "other entry's bound": ("cuda", "8.6", lambda d, c: d["cuda"]["ade_2026-08-19"][
        "source_sha256"].__setitem__("meep_gpu/cuda_kernels/ade_kernels.py", "0" * 64)),
    "metal bound": ("metal", G13, lambda d, c: d["metal"][c["metal"][0]][
        "source_sha256"].__setitem__("moved", "x")),
    "metal run_bound": ("metal", G13, lambda d, c: d["metal"][c["metal"][0]]["runs"][
        G13].__setitem__("bound_sha256", "moved")),
    "metal torch": ("metal", G13, lambda d, c: d["metal"][c["metal"][0]]["runs"][G13].__setitem__(
        "torch", "9.9.9")),
    "metal frontend": ("metal", G13, lambda d, c: d["metal"][c["metal"][0]]["runs"][
        G13].__setitem__("metal_frontend", "metalfe-9.9")),
    "metal runs emptied": ("metal", G13, lambda d, c: d["metal"][c["metal"][0]].__setitem__(
        "runs", {})),
    "metal runs deleted": ("metal", G13, lambda d, c: d["metal"][c["metal"][0]].pop("runs")),
}


def _set_own(table, device, field):
    def apply(ledgers, cited):
        ledgers[table][cited[table][0]]["runs"][device][field] = "moved"
    return apply


def _set_entry(table, field):
    def apply(ledgers, cited):
        ledgers[table][cited[table][0]][field] = {"moved": 1}
    return apply


def _policy_value(ledgers, cited):
    held = ledgers["triton"][_own_policy_weld(ledgers, cited)]["runs"]["8.6"]["certified_policies"]
    held["flush"] = "moved"


def _uncited_narrative(ledgers, cited):
    key = next(key for key in sorted(ledgers["cuda"])
               if key not in cited["cuda"] and key != "driver_dispatch"
               and isinstance(ledgers["cuda"][key], dict))
    ledgers["cuda"][key]["purpose"] = "moved"


MUST_STAY = {
    "another capability's run": ("triton", "8.6", lambda d, c: d["triton"][c["triton"][0]][
        "runs"].__setitem__("9.0", {"bound_sha256": "x", "certified_policies": {}})),
    "another capability's stack": ("triton", "8.6", lambda d, c: d["triton"][
        "certified_nvidia_stack"]["by_capability"].__setitem__("9.0", {"stack": {"cupy": ["x"]}})),
    "own stack read_from": ("triton", "8.6", lambda d, c: d["triton"][
        "certified_nvidia_stack"]["by_capability"]["8.6"].__setitem__("read_from", "moved")),
    "driver_dispatch": ("triton", "8.6", lambda d, c: d["triton"].__setitem__(
        "driver_dispatch", {"source_sha256": {"moved": "x"}})),
    "array_path_oracle": ("cuda", "8.6", _set_entry("cuda", "array_path_oracle")),
    "purpose": ("cuda", "8.6", _set_entry("cuda", "purpose")),
    "status": ("cuda", "8.6", lambda d, c: d["cuda"][c["cuda"][0]].__setitem__("status", "FAIL")),
    "code_sha256": ("cuda", "8.6", _set_entry("cuda", "code_sha256")),
    "device_sha256": ("cuda", "8.6", _set_entry("cuda", "device_sha256")),
    **{f"own run {field}": ("cuda", "8.6", _set_own("cuda", "8.6", field))
       for field in ("host", "records", "subnormal_policy", "recorded_utc", "artifact_sha256")},
    "a policy's value, its key kept": ("triton", "8.6", _policy_value),
    "an uncited entry's narrative": ("cuda", "8.6", _uncited_narrative),
    "metal_kernels": ("metal", G13, lambda d, c: d["metal"].__setitem__("metal_kernels", {"m": 1})),
    "another architecture's run": ("metal", G13, lambda d, c: d["metal"][c["metal"][0]][
        "runs"].__setitem__(G15, {"bound_sha256": "x", "torch": "9", "metal_frontend": "9"})),
    "an append, the row on 3.1.0": ("triton", "8.6", lambda d, c: d["triton"][
        "validated_triton_versions"].append("3.2.0")),
}


@pytest.fixture(scope="module")
def real():
    cited = ledger_scope.cited_welds(REPO_ROOT)
    return {table: read_ledger(REPO_ROOT, table) for table in TABLE_LEDGER}, cited


def _slice(ledgers, cited, table, device, triton="3.1.0"):
    return ledger_scope.ledger_slice(ledgers[table], table, device, cited,
                                     triton_version=triton if table == "triton" else None)


#: The environment every cited Metal weld's own run names on the shipped ledger.
_SHIPPED_METAL = next(run for _gate, runs in metal_dispatch.cited_environments() for run in runs)
SHIPPED_TORCH, SHIPPED_FRONTEND = _SHIPPED_METAL["torch"], _SHIPPED_METAL["metal_frontend"]


def package_decisions(ledgers, table, device, monkeypatch):
    """What the package's own readers answer on these ledger states."""
    monkeypatch.setattr(fastpath, "_FINGERPRINTS",
                        {"triton_kernels": ledgers["triton"], "cuda_kernels": ledgers["cuda"]})
    if table == "metal":
        return [metal_dispatch.environment_judgements(fact, value, device, ledger=ledgers["metal"])
                for fact, value in (("architecture", device), ("torch", SHIPPED_TORCH),
                                    ("metal_frontend", SHIPPED_FRONTEND))] + [
            [(gate, None if runs is None else len(runs))
             for gate, runs in metal_dispatch.cited_environments(ledger=ledgers["metal"])]]
    out = {"3.1.0 listed": "3.1.0" in fastpath.validated_triton_versions(),
           "stack": fastpath.certified_nvidia_stack(device)}
    for name in ("triton", "cuda"):
        admitted = fastpath.capability_admission(name)["admitted"]
        out[f"{name} admitted"] = device in (admitted or ())
        out[f"{name} flush"] = fastpath.policy_evidence_reasons(
            sorted(fastpath._arm_certification_map(name)), name, device, "flush")
    return out


@pytest.mark.parametrize("name", sorted(MUST_MOVE))
def test_a_slice_field_moves_the_slice_and_what_the_package_reads(real, name, monkeypatch):
    ledgers, cited = real
    table, device, mutate = MUST_MOVE[name]
    changed = copy.deepcopy(ledgers)
    mutate(changed, cited)
    assert _slice(changed, cited, table, device) != _slice(ledgers, cited, table, device), name
    if name == "other entry's bound":
        # Read at emission, by no reader of the ledger: the file itself is the read, and
        # the slice stands in for it (test_a_re_gate_of_a_file_read_as_text_moves_the_slice).
        assert fastpath.bound_digest(changed["cuda"]["ade_2026-08-19"]) != fastpath.bound_digest(
            ledgers["cuda"]["ade_2026-08-19"])
        return
    assert (package_decisions(changed, table, device, monkeypatch)
            != package_decisions(ledgers, table, device, monkeypatch)), name


def test_an_append_with_no_version_known_moves_the_slice_and_the_list(real, monkeypatch):
    ledgers, cited = real
    changed = copy.deepcopy(ledgers)
    changed["triton"]["validated_triton_versions"].append("3.2.0")
    assert (_slice(changed, cited, "triton", "8.6", None)
            != _slice(ledgers, cited, "triton", "8.6", None))
    monkeypatch.setattr(fastpath, "_FINGERPRINTS", {"triton_kernels": changed["triton"]})
    assert "3.2.0" in fastpath.validated_triton_versions()


@pytest.mark.parametrize("name", sorted(MUST_STAY))
def test_an_unread_field_leaves_the_slice_and_what_the_package_decides(real, name, monkeypatch):
    ledgers, cited = real
    table, device, mutate = MUST_STAY[name]
    changed = copy.deepcopy(ledgers)
    mutate(changed, cited)
    assert _slice(changed, cited, table, device) == _slice(ledgers, cited, table, device), name
    assert (package_decisions(changed, table, device, monkeypatch)
            == package_decisions(ledgers, table, device, monkeypatch)), name


# ------------------------------------------------------------------------- the drift guards ----

def test_the_tool_s_spellings_are_the_package_s(real):
    ledgers, cited = real
    assert ledger_scope.BOUND_FIELDS == tuple(fastpath.BOUND_FIELDS)
    assert ledger_scope.NVIDIA_STACK_AXES == tuple(fastpath.NVIDIA_STACK_AXES)
    assert ledger_scope.STACK_ENTRY == fastpath.CERTIFIED_STACK_ENTRY
    assert ledger_scope.STACK_BY_CAPABILITY == fastpath.STACK_BY_CAPABILITY
    assert ledger_scope._CAPABILITY_KEY.pattern == fastpath._CAPABILITY_KEY.pattern
    assert ledger_scope._APPLE_KEY.pattern == dispatch_preference.APPLE_FAMILY.pattern
    for value in (None, "", "metalfe-32023.850.10", "32023.850.10", " metalfe-1 "):
        assert ledger_scope.frontend_key(value) == metal_dispatch.frontend_key(value)
    for table, ledger in ledgers.items():
        for entry in ledger.values():
            if not isinstance(entry, dict):
                continue
            try:
                expected = fastpath.bound_digest(entry)
            except fastpath.CapabilityRecordError:
                expected = None
            assert ledger_scope.bound_digest(entry) == expected


def test_the_cited_welds_read_from_source_are_the_imported_constants(real):
    _ledgers, cited = real
    assert set(cited["triton"]) == {gate for _f, gate in fastpath.ARM_CERTIFICATION.values()}
    assert set(cited["cuda"]) == {
        gate for _f, gate in fastpath_cuda.CUDA_ARM_CERTIFICATION.values()}
    assert set(cited["metal"]) == {gate for _f, gate in metal_dispatch.ARM_CERTIFICATION.values()}


def test_a_view_in_the_tool_is_the_view_a_row_records(real):
    ledgers, _cited = real
    for table, ledger in ledgers.items():
        assert ledger_scope.view_sha256(ledger) == plan_key.ledger_view_sha256(
            os.path.join(REPO_ROOT, TABLE_LEDGER[table]))


#: What a decision reads by value, per cited weld (``runs.<own>``, ``runs.<other>``), and what
#: only quotes a value into the record. A new field read outside these fails the test below,
#: which is what keeps the slice the read set.
NVIDIA_READ = {
    "decides": {"validated_triton_versions", "certified_nvidia_stack.by_capability.<own>.stack",
                "<weld>.source_sha256", "<weld>.source_sha256_at_recert", "<weld>.probe",
                "<weld>.probe_sha256", "<weld>.adapter", "<weld>.adapter_sha256", "<weld>.runs",
                "<weld>.runs.<own>", "<weld>.runs.<own>.bound_sha256",
                "<weld>.runs.<own>.certified_policies"},
    "reads without deciding": {"<weld>.runs.<other>", "<weld>.runs.<other>.bound_sha256",
                               "certified_nvidia_stack", "certified_nvidia_stack.by_capability",
                               "certified_nvidia_stack.by_capability.<own>",
                               "certified_nvidia_stack.by_capability.<other>",
                               "certified_nvidia_stack.by_capability.<other>.stack"},
    "quotes": {"<weld>", "<weld>.families", "<weld>.purpose", "<weld>.status", "<weld>.kernels",
               "<weld>.kernel_module", "<weld>.legs",
               "<weld>.runs.<own>.families",
               "<weld>.runs.<own>.host", "<weld>.runs.<own>.records",
               "<weld>.runs.<own>.recorded_utc", "<weld>.runs.<own>.subnormal_policy",
               "<other entry>"},
}
METAL_READ = {
    "decides": {"<weld>.source_sha256", "<weld>.runs", "<weld>.runs.<own>",
                "<weld>.runs.<own>.bound_sha256", "<weld>.runs.<own>.torch",
                "<weld>.runs.<own>.metal_frontend"},
    "reads without deciding": {"<weld>.runs.<other>", "<weld>.runs.<other>.bound_sha256",
                               "<weld>.runs.<other>.torch",
                               "<weld>.runs.<other>.metal_frontend"},
    "quotes": {"<weld>", "<weld>.purpose", "<weld>.status", "<weld>.runs.<own>.host",
               "<weld>.runs.<own>.records", "<weld>.runs.<own>.recorded_utc",
               "<weld>.runs.<own>.subnormal_policy"},
}

READS_COMMON = r"""
import copy, json, os, random, re, sys
sys.path.append(os.path.join(os.getcwd(), "parity", "meep_gpu"))
import selected_plan
class Log:
    def __init__(self):
        self.paths = set()
class Rec(dict):
    '''A dict that logs every read at its path; __iter__ overridden so dict(x) and the JSON
    encoder take the slow path through it.'''
    def __init__(self, data, log, path=()):
        super().__init__(data)
        self._log, self._path = log, path
    def _child(self, key, value):
        path = self._path + (str(key),)
        self._log.paths.add(path)
        return Rec(value, self._log, path) if isinstance(value, dict) else value
    def __getitem__(self, key):
        return self._child(key, super().__getitem__(key))
    def get(self, key, default=None):
        return self._child(key, super().__getitem__(key)) if super().__contains__(key) else default
    def __iter__(self):
        self._log.paths.add(self._path)
        return super().__iter__()
    def keys(self):
        self._log.paths.add(self._path)
        return super().keys()
    def items(self):
        self._log.paths.add(self._path)
        return [(k, self._child(k, v)) for k, v in super().items()]
    def values(self):
        self._log.paths.add(self._path)
        return [self._child(k, v) for k, v in super().items()]
    __hash__ = None
def fold(path, cited, own):
    '''One read as a label: welds and capabilities folded, leaves under a value cut.'''
    head = path[0]
    if head not in cited and head not in ("validated_triton_versions", "certified_nvidia_stack"):
        return "<other entry>"
    out = ["<weld>" if head in cited else head]
    for part in path[1:]:
        last = out[-1]
        if last in ("runs", "by_capability"):
            out.append("<own>" if part == own else "<other>")
        elif last in ("<weld>", "certified_nvidia_stack", "<own>", "<other>"):
            out.append(part)
        else:
            break
    return ".".join(out)
"""

NVIDIA_READS_CHILD = READS_COMMON + r"""
import pytest
from meep_gpu import backends, fastpath
from meep_gpu import subnormal_policy as policy_module
from meep_gpu import test_nvidia_fast_default as stand_ins
from meep_gpu.test_dispatch_contract import (CountingPlan, certified_envelope_fields,
    certified_envelope_pml, cuda_step_plan, install_composer, install_cuda_composer, step_plan,
    stub_triton)
FOLDERS = {"triton": "triton_kernels", "cuda": "cuda_kernels"}
IDENTITY = {"cupy": "13.5.1", "nvrtc": "11.8", "cuda_runtime": "11080", "cuda_driver": "12030",
            "host": "Linux x86_64", "numpy": "2.2"}
def ladder(scope, ledgers):
    patch = pytest.MonkeyPatch()
    try:
        for name in stand_ins.SWITCHES:
            patch.delenv(name, raising=False)
        patch.setenv(fastpath.DISPATCH_ENABLE, "1")
        policy_module._reset_for_tests()
        fastpath._POLICY_INSTALLED_BY_DISPATCH = None
        fastpath.reset_dispatch_announcements()
        if backends.subnormals_flushed():
            policy_module._drive_host_fpu(False)
        patch.setattr(fastpath, "_FINGERPRINTS", {FOLDERS[t]: ledgers[t] for t in FOLDERS})
        if scope != "default":
            patch.setenv(fastpath.HARDWARE_SWITCH, scope)
        stub_triton(patch, "3.1.0", stack=False)
        patch.setattr(fastpath, "_nvidia_stack_identity", lambda xp: dict(IDENTITY))
        install_composer(patch, step_plan({"step_B": CountingPlan("b")}, {"step_B": "PML"}))
        install_cuda_composer(patch, cuda_step_plan())
        plan = fastpath.plan_fast_path(certified_envelope_fields(), certified_envelope_pml(),
                                       stand_ins._grid((8, 6)))
        report = plan.report() if plan is not None else fastpath.last_dispatch_report()
        environment = report.get("environment") or {}
        return {"decision": report.get("decision"), "plan": selected_plan.selected_plan(report),
                "certified": report.get("certified"),
                "tables": {k: bool(v.get("candidate"))
                           for k, v in (report.get("tables") or {}).items()},
                "device_certified_by_table": environment.get("device_certified_by_table"),
                "triton_certified": environment.get("triton_certified"),
                "stack_certified": (environment.get("stack") or {}).get("certified")}
    finally:
        patch.undo()
def pure(ledgers):
    patch = pytest.MonkeyPatch()
    try:
        patch.setattr(fastpath, "_FINGERPRINTS", {FOLDERS[t]: ledgers[t] for t in FOLDERS})
        out = {"versions": list(fastpath.validated_triton_versions()),
               "stack": fastpath.certified_nvidia_stack("8.6")}
        for table in FOLDERS:
            admitted = fastpath.capability_admission(table)["admitted"]
            out[table] = [fastpath._device_is_validated({"compute_capability": "8.6"}, admitted),
                          fastpath.policy_evidence_reasons(
                              sorted(fastpath._arm_certification_map(table)), table, "8.6",
                              "flush")]
        return out
    finally:
        patch.undo()
def decisions(ledgers):
    return json.loads(json.dumps({"pure": pure(ledgers), "default": ladder("default", ledgers),
                                  "certified": ladder("certified", ledgers)}, default=str))
LEDGERS = {t: json.load(open(f"meep_gpu/{FOLDERS[t]}/fingerprints.json")) for t in FOLDERS}
CITED = {t: sorted({g for _f, g in fastpath._arm_certification_map(t).values()}) for t in FOLDERS}
for table, ledger in LEDGERS.items():
    for entry in ledger.values():
        if isinstance(entry, dict) and "8.6" in (entry.get("runs") or {}):
            entry["runs"]["9.0"] = dict(entry["runs"]["8.6"])
LEDGERS["triton"]["certified_nvidia_stack"]["by_capability"]["9.0"] = copy.deepcopy(
    LEDGERS["triton"]["certified_nvidia_stack"]["by_capability"]["8.6"])
base = decisions(LEDGERS)
logs = {t: Log() for t in FOLDERS}
logged = decisions({t: Rec(LEDGERS[t], logs[t]) for t in FOLDERS})
reads = {t: sorted({fold(p, set(CITED[t]), "8.6") for p in logs[t].paths if p}) for t in FOLDERS}
def strip(ledgers):
    out = {}
    for table in FOLDERS:
        small = {}
        for key in CITED[table]:
            entry = ledgers[table].get(key)
            if not isinstance(entry, dict):
                continue
            part = {f: entry[f] for f in fastpath.BOUND_FIELDS if f in entry}
            run = (entry.get("runs") or {}).get("8.6")
            part["runs"] = {}
            if isinstance(run, dict):
                part["runs"]["8.6"] = {f: run[f] for f in ("bound_sha256",
                                                           fastpath.CERTIFIED_POLICIES)
                                       if f in run}
            small[key] = part
        if table == "triton":
            small["validated_triton_versions"] = ledgers[table]["validated_triton_versions"]
            small["certified_nvidia_stack"] = {"by_capability": {"8.6": {"stack": ledgers[table][
                "certified_nvidia_stack"]["by_capability"]["8.6"]["stack"]}}}
        out[table] = small
    return out
sufficient = decisions(strip(LEDGERS)) == base
rng = random.Random(20261102)
fuzz = []
for round_number in range(5):
    damaged = copy.deepcopy(LEDGERS)
    for table in FOLDERS:
        for key, entry in damaged[table].items():
            if not isinstance(entry, dict):
                continue
            if key in CITED[table]:
                for field in ("purpose", "status", "code_sha256", "device_sha256"):
                    if rng.random() < 0.5:
                        entry[field] = f"fuzz {rng.random()}"
                other = entry["runs"].get("9.0")
                if other is not None and rng.random() < 0.5:
                    entry["runs"]["9.0"] = rng.choice([{"bound_sha256": "fuzz"}, "not a run"])
            elif key not in ("validated_triton_versions", "certified_nvidia_stack"):
                entry["fuzz"] = rng.random()
    fuzz.append(decisions(damaged)["certified"] == base["certified"]
                and decisions(damaged)["default"]["plan"] == base["default"]["plan"])
print(json.dumps({"reads": reads, "equal_under_the_log": logged == base, "sufficient": sufficient,
                  "fuzz": fuzz, "decision": base["default"]["decision"]}))
"""

METAL_READS_CHILD = METAL_CHILD.rsplit("tables, decisions = set(), {}", 1)[0] + READS_COMMON + r"""
LEDGER = json.load(open("meep_gpu/metal_kernels/fingerprints.json"))
for entry in LEDGER.values():
    if isinstance(entry, dict) and "applegpu_g13s" in (entry.get("runs") or {}):
        entry["runs"]["applegpu_g15s"] = dict(entry["runs"]["applegpu_g13s"], torch="9.9.9")
CITED = {gate for _f, gate in metal_dispatch.ARM_CERTIFICATION.values()}
g13 = next(run for _, runs in metal_dispatch.cited_environments(ledger=LEDGER) for run in runs or ()
           if run["architecture"] == "applegpu_g13s")
metal_dispatch.metal_toolchain = lambda: {
    "importable": True, "version": g13.get("torch"), "built": True, "available": True,
    "compile_shader": True, "metal_frontend": g13.get("metal_frontend"),
    "architecture": "applegpu_g13s", "device_name": "stand-in", "fast_math": None,
    "machine": "arm64"}
def plans(ledger):
    metal_dispatch._FINGERPRINTS = ledger
    fastpath._FINGERPRINTS = {"metal_kernels": ledger}
    out = []
    for scope in (None, "certified"):
        if scope:
            os.environ[fastpath.HARDWARE_SWITCH] = scope
        else:
            os.environ.pop(fastpath.HARDWARE_SWITCH, None)
        for row in matrix.MATRIX[::4]:
            fields, pml = row[1]()
            plan = fastpath.plan_fast_path(fields, pml, getattr(fields, "grid", None), sources=[])
            report = plan.report() if plan is not None else fastpath.last_dispatch_report()
            environment = report.get("environment") or {}
            out.append([report.get("decision"), selected_plan.selected_plan(report),
                        report.get("certified"), environment.get("device_certified"),
                        environment.get("torch_certified"), environment.get("frontend_certified")])
    return json.loads(json.dumps(out, default=str))
base = plans(LEDGER)
log = Log()
logged = plans(Rec(LEDGER, log))
reads = sorted({fold(p, CITED, "applegpu_g13s") for p in log.paths if p})
small = {}
for key in CITED:
    entry = LEDGER[key]
    run = entry["runs"]["applegpu_g13s"]
    small[key] = {"source_sha256": entry["source_sha256"], "runs": {"applegpu_g13s": {
        f: run[f] for f in ("bound_sha256", "torch", "metal_frontend") if f in run}}}
print(json.dumps({"reads": reads, "equal_under_the_log": logged == base,
                  "sufficient": plans(small) == base, "rows": len(base),
                  "dispatched": sum(1 for row in base if row[0] == "dispatched")}))
"""


def test_the_read_set_is_the_slice_on_nvidia(tmp_path):
    seen = child(NVIDIA_READS_CHILD, tmp_path)
    assert seen["decision"] == "dispatched", seen
    assert seen["equal_under_the_log"] is True
    allowed = set().union(*NVIDIA_READ.values())
    for table, reads in seen["reads"].items():
        assert not set(reads) - allowed, (table, sorted(set(reads) - allowed))
    assert {"<weld>.runs.<own>.bound_sha256", "<weld>.source_sha256"} <= set(seen["reads"]["cuda"])
    assert "validated_triton_versions" in seen["reads"]["triton"]


def test_the_read_set_is_the_slice_on_metal(tmp_path):
    seen = child(METAL_READS_CHILD, tmp_path)
    assert seen["dispatched"] > 0, seen
    assert seen["equal_under_the_log"] is True
    assert not set(seen["reads"]) - set().union(*METAL_READ.values()), seen["reads"]
    assert METAL_READ["decides"] <= set(seen["reads"])


def test_the_slice_s_sources_are_enough_and_the_rest_decides_nothing(tmp_path):
    seen = child(NVIDIA_READS_CHILD, tmp_path)
    assert seen["sufficient"] is True
    assert seen["fuzz"] == [True] * 5


def test_the_slice_s_sources_are_enough_on_metal(tmp_path):
    assert child(METAL_READS_CHILD, tmp_path)["sufficient"] is True


# ----------------------------------------------------------------------------- the archive ----

def test_every_archived_state_is_what_its_name_says():
    found = sorted(path for path in pathlib.Path(STATES_ROOT).glob("*/*")
                   if path.name != ".DS_Store")
    assert found, "no ledger state is archived"
    for path in found:
        assert path.parent.name in TABLE_LEDGER, path
        assert path.name.endswith(".json.gz"), path
        packed = path.read_bytes()
        assert packed[:2] == b"\x1f\x8b" and packed[4:8] == b"\0\0\0\0", f"{path}: mtime in header"
        assert packed[3] == 0, f"{path}: a name or comment in the header"
        raw = gzip.decompress(packed)
        assert hashlib.sha256(raw).hexdigest() == path.name.split(".")[0], path
        assert isinstance(json.loads(raw), dict), path


def test_every_live_ledger_is_archived():
    """A commit that moves a ledger carries its state: ``ledger_scope.py archive``."""
    for table, name in TABLE_LEDGER.items():
        digest = sha_of(REPO_ROOT, name)
        assert (pathlib.Path(STATES_ROOT) / table / f"{digest}.json.gz").is_file(), (
            f"{name}: python parity/meep_gpu/ledger_scope.py archive --table {table} {name}")


def test_archiving_is_idempotent_and_refuses_what_does_not_parse(tmp_path):
    source = tmp_path / "ledger.json"
    source.write_text('{"a": 1}\n')
    first = ledger_scope.archive_state(tmp_path, "cuda", source)
    packed = first.read_bytes()
    assert ledger_scope.archive_state(tmp_path, "cuda", source) == first
    assert first.read_bytes() == packed
    source.write_text("not json")
    with pytest.raises(ValueError):
        ledger_scope.archive_state(tmp_path, "cuda", source)
    source.write_text("[1]")
    with pytest.raises(ValueError):
        ledger_scope.archive_state(tmp_path, "cuda", source)
    with pytest.raises(ValueError):
        ledger_scope.archive_state(tmp_path, "other", source)


def test_two_raw_states_of_one_view_answer_the_same(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps({"w": {"source_sha256": {}}, "driver_dispatch": {"x": 1}}))
    b.write_text(json.dumps({"w": {"source_sha256": {}}, "driver_dispatch": {"x": 2}}))
    ledger_scope.archive_state(tmp_path, "cuda", a)
    ledger_scope.archive_state(tmp_path, "cuda", b)
    states = ledger_scope.LedgerStates(tmp_path)
    view = ledger_scope.view_sha256(json.loads(a.read_text()))
    assert states.view("cuda", view)["w"] == {"source_sha256": {}}


def _moved_cuda(root):
    """A write outside any slice: every comparison below turns on the state's lookup."""
    ledger = read_ledger(root, "cuda")
    ledger["planted_narrative"] = {"purpose": "moved"}
    write_json(root, CUDA, ledger)


def _rename_state(root, digest, table="cuda", to=None):
    directory = own_archive(root) / table
    return directory / f"{digest}.json.gz", directory / f"{to or '0' * 64}.json.gz"


def test_a_digest_no_state_answers_for_compares_whole_by_name(shipped):
    run = shipped_l3(shipped)
    run[plan_key.INPUTS_FIELD]["ledgers"][CUDA] = "a" * 64
    verdict = judged_on("l3", shipped, "cc86", run)
    assert verdict["state"] == pc.RERUN_INPUTS
    assert only_why(verdict, CUDA) == "ledger_state_not_archived"


def test_a_renamed_state_answers_for_nothing(shipped):
    run = shipped_timing(shipped)
    digest = run["processes"][0][TR.DATA_KEY][CUDA]
    _moved_cuda(shipped)
    assert not TR.moved_files(run, str(shipped))
    source, target = _rename_state(shipped, digest)
    source.rename(target)
    assert CUDA in TR.moved_files(run, str(shipped))
    run["processes"][0][TR.DATA_KEY][CUDA] = target.name.split(".")[0]
    verdict = ledger_scope.ledger_moved(
        CUDA, target.name.split(".")[0], form="raw", device="8.6",
        imports=closure_of(shipped, "8.6"), triton_version="3.1.0", root=shipped)
    assert verdict.moved and verdict.why == "ledger_state_refused"


def test_an_edited_state_is_refused_by_name(shipped):
    run = shipped_timing(shipped)
    digest = run["processes"][0][TR.DATA_KEY][CUDA]
    _moved_cuda(shipped)
    path, _ = _rename_state(shipped, digest)
    edited = json.loads(gzip.decompress(path.read_bytes()))
    edited["planted"] = 1
    path.write_bytes(gzip.compress(json.dumps(edited).encode()))
    verdict = ledger_scope.ledger_moved(
        CUDA, digest, form="raw", device="8.6", imports=closure_of(shipped, "8.6"),
        triton_version="3.1.0", root=shipped)
    assert verdict.moved and verdict.why == "ledger_state_refused"


def test_a_state_under_another_table_answers_for_nothing(shipped):
    run = shipped_timing(shipped)
    digest = run["processes"][0][TR.DATA_KEY][CUDA]
    _moved_cuda(shipped)
    source, _ = _rename_state(shipped, digest)
    source.rename(source.parent.parent / "triton" / source.name)
    verdict = ledger_scope.ledger_moved(
        CUDA, digest, form="raw", device="8.6", imports=closure_of(shipped, "8.6"),
        triton_version="3.1.0", root=shipped)
    assert verdict.moved and verdict.why == "ledger_state_not_archived"


def test_a_raw_digest_in_a_view_slot_resolves_to_nothing(shipped):
    raw = sha_of(shipped, CUDA)
    _moved_cuda(shipped)
    verdict = ledger_scope.ledger_moved(
        CUDA, raw, form="view", device="8.6", imports=closure_of(shipped, "8.6"),
        triton_version="3.1.0", root=shipped)
    assert verdict.moved and verdict.why == "ledger_state_not_archived"


# --------------------------------------------------------- planted records a reviewer tries ----

@pytest.mark.parametrize("recorded", [{"error": "OSError()"}, None, "short", 7])
def test_a_slot_holding_no_digest_compares_whole(shipped, recorded):
    run = shipped_l3(shipped)
    _moved_cuda(shipped)
    if recorded is None:
        del run[plan_key.INPUTS_FIELD]["ledgers"][CUDA]
    else:
        run[plan_key.INPUTS_FIELD]["ledgers"][CUDA] = recorded
    verdict = judged_on("l3", shipped, "cc86", run)
    assert verdict["state"] == pc.RERUN_INPUTS
    assert only_why(verdict, CUDA) == "ledger_digest_unrecorded"


@pytest.mark.parametrize("content", ["not json", None, "[1, 2]"])
def test_an_unreadable_live_ledger_compares_whole(shipped, content):
    run = shipped_timing(shipped)
    path = shipped / CUDA
    path.unlink()
    if content is not None:
        path.write_text(content)
    verdict = ledger_scope.ledger_moved(
        CUDA, run["processes"][0][TR.DATA_KEY][CUDA], form="raw", device="8.6",
        imports=closure_of(shipped, "8.6"), triton_version="3.1.0", root=shipped)
    assert verdict.moved and verdict.why == "live_ledger_unreadable"


def test_a_ledger_marked_unreadable_compares_whole(shipped):
    """``capability_report`` reads that key as a ledger it could not take and every verdict
    goes unknown, whatever the rest says, so it can never be slice-equal."""
    run = shipped_l3(shipped)
    ledger = read_ledger(shipped, "cuda")
    ledger[ledger_scope.UNREADABLE_KEY] = "planted"
    write_json(shipped, CUDA, ledger)
    verdict = judged_on("l3", shipped, "cc86", run)
    assert verdict["state"] == pc.RERUN_INPUTS
    assert only_why(verdict, CUDA) == "ledger_marked_unreadable"


@pytest.mark.parametrize("kind", sorted(SHIPPED_RUNS))
def test_a_pin_that_states_it_drifted_never_reads_as_carried(shipped, kind):
    """The real writer: ``rebind_cuda_welds`` takes ``source_sha256`` from the artifact and
    writes ``source_drift`` and ``status: DRIFTED`` when the file has since moved. Bound and
    run bound stay equal, so the slice does, and only the statement shows it."""
    run = SHIPPED_RUNS[kind](shipped)
    ledger = read_ledger(shipped, "cuda")
    key = cited_of(shipped)["cuda"][0]
    before = copy.deepcopy(ledger)
    pin = "meep_gpu/cuda_kernels/step_curl_kernels.py"
    ledger[key].update(source_drift=[pin], status="DRIFTED")
    assert (ledger_scope.ledger_slice(ledger, "cuda", "8.6", cited_of(shipped))
            == ledger_scope.ledger_slice(before, "cuda", "8.6", cited_of(shipped)))
    write_json(shipped, CUDA, ledger)
    verdict = judged_on(kind, shipped, "cc86", run)
    assert verdict["state"] == pc.RERUN_INPUTS, verdict
    assert only_why(verdict, CUDA) == "pin_drift_stated"
    assert verdict["compared_whole"][CUDA]["entries"] == [key]


def test_a_pin_that_states_it_drifted_owes_a_timing_row_a_check(tmp_path, shipped):
    row = shipped_timing(shipped)
    records = TR.load_store(row_fixtures.store_of(tmp_path, [row]))
    ledger = read_ledger(shipped, "cuda")
    ledger[cited_of(shipped)["cuda"][0]].update(source_drift=["x.py"], status="DRIFTED")
    write_json(shipped, CUDA, ledger)
    assessment = TR.select(records, str(shipped))[row["key"]]["assessments"][0]
    assert assessment["status"] == "check_owed" and CUDA in assessment["exempt"]
    assert assessment["compared_whole"][CUDA]["why"] == "pin_drift_stated"


@pytest.mark.parametrize("statement, reads", [({"source_drift": ["x.py"]}, True),
                                              ({"status": "DRIFTED"}, True),
                                              ({"status": "STALE"}, True),
                                              ({"status": "PASS", "source_drift": []}, False),
                                              ({}, False)])
def test_every_status_but_pass_states_a_drift(statement, reads):
    assert ledger_scope.stated_drift({"weld": statement}) == (["weld"] if reads else [])


def test_a_real_drifted_status_reaches_no_cited_weld_at_the_tip(real):
    ledgers, cited = real
    for table, ledger in ledgers.items():
        assert not ledger_scope.stated_drift(ledger), table


def test_an_import_record_lacking_the_constants_compares_whole(shipped):
    run = shipped_l3(shipped)
    del run["imported_source_sha256"]["meep_gpu/fastpath_cuda.py"]
    _moved_cuda(shipped)
    verdict = judged_on("l3", shipped, "cc86", run)
    assert only_why(verdict, CUDA) == "cited_source_not_in_closure"


def test_constants_at_other_bytes_compare_whole(shipped):
    run = shipped_timing(shipped)
    run["processes"][0][TR.IMPORTED_KEY]["meep_gpu/fastpath_cuda.py"] = "0" * 64
    _moved_cuda(shipped)
    verdict = ledger_scope.ledger_moved(
        CUDA, run["processes"][0][TR.DATA_KEY][CUDA], form="raw", device="8.6",
        imports=run["processes"][0][TR.IMPORTED_KEY], triton_version=None, root=shipped)
    assert verdict.moved and verdict.why == "cited_source_not_in_closure"


def test_constants_that_cannot_be_read_compare_whole(shipped):
    run = shipped_l3(shipped)
    _moved_cuda(shipped)
    path = shipped / "meep_gpu/fastpath_cuda.py"
    path.unlink()
    path.write_text("CUDA_ARM_CERTIFICATION = build()\n")
    run["imported_source_sha256"]["meep_gpu/fastpath_cuda.py"] = sha_of(
        shipped, "meep_gpu/fastpath_cuda.py")
    verdict = judged_on("l3", shipped, "cc86", run)
    assert only_why(verdict, CUDA) == "cited_unreadable"


def test_no_triton_version_keeps_the_whole_list(shipped):
    run = shipped_l3(shipped)
    del run[plan_key.TOOLCHAIN_FIELD]["triton"]
    ledger = read_ledger(shipped, "triton")
    ledger["validated_triton_versions"].append("3.2.0")
    write_json(shipped, TRITON, ledger)
    assert judged_on("l3", shipped, "cc86", run)["state"] == pc.RERUN_INPUTS


def test_a_row_on_3_1_0_stands_an_append_and_one_on_3_2_0_does_not(shipped):
    held = shipped_l3(shipped, triton="3.1.0")
    other = shipped_l3(shipped, triton="3.2.0")
    ledger = read_ledger(shipped, "triton")
    ledger["validated_triton_versions"].append("3.2.0")
    write_json(shipped, TRITON, ledger)
    assert judged_on("l3", shipped, "cc86", held)["state"] == pc.CARRY_SLICE
    assert judged_on("l3", shipped, "cc86", other)["state"] == pc.RERUN_INPUTS


def test_a_process_recording_only_the_unread_ledger_is_refused(release):
    row = timed_row(release, "8.6", (CUDA_NARRATIVE,))
    assert any("no ledger of its own platform" in why for why in TR.refusals(row))


def test_a_row_stamped_another_device_is_refused_before_any_slice(shipped):
    run = shipped_l3(shipped, "cc90")
    _moved_cuda(shipped)
    assert judged_on("l3", shipped, "cc86", run)["state"] == pc.RERUN_DEVICE


@pytest.mark.parametrize("device", ["8.60", "08.6", "cc086", "cc860", "cc8", "86", "ga100", None])
def test_a_spelling_that_keys_no_run_is_compared_on_the_whole_digest(shipped, device):
    run = shipped_timing(shipped)
    digest = run["processes"][0][TR.DATA_KEY][CUDA]
    _moved_cuda(shipped)
    verdict = ledger_scope.ledger_moved(
        CUDA, digest, form="raw", device=device, imports=closure_of(shipped, "8.6"),
        triton_version="3.1.0", root=shipped)
    assert verdict.moved and verdict.why in ("device_not_placed", "device_names_no_run"), verdict
    narrative = ledger_scope.ledger_moved(
        CUDA_NARRATIVE, "0" * 64, form="raw", device=device, imports=closure_of(shipped, "8.6"),
        triton_version="3.1.0", root=shipped)
    assert narrative.moved, narrative


@pytest.mark.parametrize("device, architecture", [
    ("cc86", "8.6"), ("8.6", "8.6"), ("cc100", "10.0"), ("cc120", "12.0"), (G13, G13),
    ("8.60", None), ("08.6", None), ("cc086", None), ("cc8", None), ("86", None), ("", None),
    (None, None), ("applegpu", None), ("Applegpu_g13s", None)])
def test_the_architecture_of_a_device(device, architecture):
    assert ledger_scope.architecture_of_device(device) == architecture


def test_a_device_this_ledger_never_ran_is_not_slice_equal(shipped):
    """The slice of a device no run names holds only ``None`` for its run fields, on both
    sides: that proves nothing, whatever moved."""
    # cc87 retires cc89 as the device no run names: the shipped ledgers now hold 8.9 runs.
    row = shipped_l3(shipped, "cc87")
    _moved_cuda(shipped)
    verdict = judged_on("l3", shipped, "cc87", row)
    assert verdict["state"] == pc.RERUN_INPUTS
    assert only_why(verdict, CUDA) == "device_names_no_run"


def test_a_launch_on_the_other_platform_s_table_compares_whole(shipped):
    run = shipped_l3(shipped, G13, table="cuda")
    ledger = read_ledger(shipped, "cuda")
    ledger["planted_narrative"] = {"purpose": "moved"}
    write_json(shipped, CUDA, ledger)
    verdict = judged_on("l3", shipped, G13, run)
    assert verdict["state"] == pc.RERUN_INPUTS
    assert only_why(verdict, CUDA) == "other_platform_recorded"


@pytest.mark.parametrize("empty_side", ["recorded", "live"])
def test_metal_runs_absent_and_empty_are_not_slice_equal(shipped, empty_side):
    """``metal_dispatch`` reads no ``runs`` as unknown and empty ``runs`` as uncertified."""
    key = cited_of(shipped)["metal"][0]
    ledger = read_ledger(shipped, "metal")
    other = copy.deepcopy(ledger)
    ledger[key]["runs"] = {}
    other[key].pop("runs")
    cited = cited_of(shipped)
    assert (ledger_scope.ledger_slice(ledger, "metal", G13, cited)
            != ledger_scope.ledger_slice(other, "metal", G13, cited))
    first, second = (ledger, other) if empty_side == "recorded" else (other, ledger)
    source = write_json(shipped, TABLE_LEDGER["metal"], first)
    ledger_scope.archive_state(own_archive(shipped).parents[2], "metal", source)
    run = shipped_l3(shipped, G13)
    write_json(shipped, TABLE_LEDGER["metal"], second)
    verdict = judged_on("l3", shipped, G13, run)
    assert verdict["state"] == pc.RERUN_INPUTS and only_why(verdict, METAL) == "slice_moved"


# ------------------------------------------------------------------- files read as text ----

#: A file read as text that no entry of its platform's ledgers pins. The list may only
#: shrink: a re-gate that widens a weld's pins moves ledger bytes, and that closes it.
UNPINNED_TEXT_READS = {"meep_gpu/triton_kernels/bfast_curl.py"}


def scan_text_reads(source):
    """Sibling files a module reads off disk rather than imports: a ``*.py`` literal inside
    ``with_name`` or a helper named for reading one, or in a path join inside a function that
    opens it. A load by path (``spec_from_file_location``) is an import, in the record. A name
    built at run time is not seen: ``_literal`` above is the one helper, by name."""
    tree = ast.parse(source)
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    found = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and re.fullmatch(r"[a-z0-9_]+\.py", node.value)):
            continue
        calls, function, cursor = [], None, parents.get(node)
        while cursor is not None:
            if isinstance(cursor, ast.Call):
                called = cursor.func
                calls.append(getattr(called, "attr", getattr(called, "id", "?")))
            if isinstance(cursor, ast.FunctionDef) and function is None:
                function = cursor
            cursor = parents.get(cursor)
        if "spec_from_file_location" in calls:
            continue
        opened = function is not None and any(
            isinstance(inner, ast.Call)
            and getattr(inner.func, "attr", getattr(inner.func, "id", "")) in ("open", "read_text")
            for inner in ast.walk(function))
        if "with_name" in calls or "_literal" in calls or ("join" in calls and opened):
            found.add(node.value)
    return found


def test_the_scan_sees_a_text_read_and_not_a_load_by_path():
    assert scan_text_reads('p = Path(__file__).with_name("a_kernels.py")\n') == {"a_kernels.py"}
    assert scan_text_reads(
        'def f():\n    with open(os.path.join(here, "b.py")) as h:\n        return h.read()\n'
    ) == {"b.py"}
    assert not scan_text_reads(
        'def f():\n    s = spec_from_file_location("c", os.path.join(here, "c.py"))\n')
    assert not scan_text_reads('x = "d.py"\n')


def test_every_text_read_is_in_the_table():
    found = {}
    for path in sorted(pathlib.Path(REPO_ROOT).glob("meep_gpu/*_kernels/*.py")):
        if path.name.startswith("test_"):
            continue
        targets = scan_text_reads(path.read_text(encoding="utf-8"))
        if targets:
            found[path.relative_to(REPO_ROOT).as_posix()] = tuple(sorted(targets))
    assert found == dict(ledger_scope.TEXT_READS)


def test_every_file_read_as_text_is_pinned_or_named(real):
    ledgers, _cited = real
    unpinned = set()
    for reader, targets in ledger_scope.TEXT_READS.items():
        platform = ledger_scope.platform_of_table(reader.split("/")[1][: -len("_kernels")])
        pinned = collections.defaultdict(set)
        for table, ledger in ledgers.items():
            if ledger_scope.platform_of_table(table) != platform:
                continue
            for key, entry in ledger.items():
                if (key != "driver_dispatch" and isinstance(entry, dict)
                        and isinstance(entry.get("source_sha256"), dict)):
                    for pin, digest in entry["source_sha256"].items():
                        pinned[pin].add(digest)
        for name in targets:
            target = f"{reader.rsplit('/', 1)[0]}/{name}"
            assert os.path.isfile(os.path.join(REPO_ROOT, target)), target
            if target not in pinned:
                unpinned.add(target)
            else:
                assert sha_of(REPO_ROOT, target) in pinned[target], target
    assert unpinned == UNPINNED_TEXT_READS


def test_a_re_gate_of_a_file_read_as_text_moves_the_slice(shipped):
    """The kernel text a fused emitter reads off disk (``fused_polarization_pair`` reads
    ``ade_kernels.py``) is in no import record and pinned by a weld no arm cites. Its re-gate
    must still void a row whose closure holds the reader and not the file."""
    reader = "meep_gpu/cuda_kernels/fused_polarization_pair.py"
    run = shipped_l3(shipped)
    run["imported_source_sha256"][reader] = sha_of(shipped, reader)
    ledger = read_ledger(shipped, "cuda")
    entry = ledger["ade_2026-08-19"]
    assert "ade_2026-08-19" not in cited_of(shipped)["cuda"]
    before = fastpath.bound_digest(entry)
    target = shipped / "meep_gpu/cuda_kernels/ade_kernels.py"
    edited = target.read_text() + "\n# moved\n"
    target.unlink()
    target.write_text(edited)
    entry["source_sha256"]["meep_gpu/cuda_kernels/ade_kernels.py"] = sha_of(
        shipped, "meep_gpu/cuda_kernels/ade_kernels.py")
    new_run = run_template(entry, "8.6")
    # Supersedes 8.9: re-binding 8.6 on moved bytes strands 8.9's live run, which must be named.
    fastpath.bind_capability(entry, bound_before=before, capability="8.6", run=new_run,
                             run_fields=tuple(new_run), supersede=("8.9",))
    write_json(shipped, CUDA, ledger)
    verdict = judged_on("l3", shipped, "cc86", run)
    assert verdict["state"] == pc.RERUN_INPUTS, verdict
    assert only_why(verdict, CUDA) == "slice_moved"
    assert "other_bound.ade_2026-08-19" in verdict["compared_whole"][CUDA]["fields"]


def test_a_file_read_as_text_that_nothing_pins_compares_whole(shipped):
    """No record names it and no bound digest holds it, so a slice cannot speak for it."""
    reader = "meep_gpu/triton_kernels/conductive_bfast_fused_hd_pair.py"
    run = shipped_l3(shipped)
    run["imported_source_sha256"][reader] = sha_of(shipped, reader)
    _moved_cuda(shipped)
    verdict = judged_on("l3", shipped, "cc86", run)
    assert verdict["state"] == pc.RERUN_INPUTS
    assert only_why(verdict, CUDA) == "text_read_unpinned"
    assert verdict["compared_whole"][CUDA]["files"] == sorted(UNPINNED_TEXT_READS)
    # ... and once the record names it, the closure speaks for it
    run["imported_source_sha256"]["meep_gpu/triton_kernels/bfast_curl.py"] = sha_of(
        shipped, "meep_gpu/triton_kernels/bfast_curl.py")
    assert judged_on("l3", shipped, "cc86", run)["state"] == pc.CARRY_SLICE


def test_a_file_read_as_text_edited_and_never_re_gated_compares_whole(shipped):
    """Another device's bind moves the file, and an edit nothing re-gated moves no entry, so
    every slice stays equal. The live bytes against the pins are what show it."""
    reader = "meep_gpu/cuda_kernels/fused_polarization_pair.py"
    target = "meep_gpu/cuda_kernels/ade_kernels.py"
    run = shipped_l3(shipped)
    run["imported_source_sha256"][reader] = sha_of(shipped, reader)
    without_the_reader = shipped_l3(shipped)
    edited = (shipped / target).read_text() + "\n# moved\n"
    (shipped / target).unlink()
    (shipped / target).write_text(edited)
    bind_nvidia(shipped, "9.0")
    verdict = judged_on("l3", shipped, "cc86", run)
    assert verdict["state"] == pc.RERUN_INPUTS
    assert only_why(verdict, CUDA) == "text_read_left_its_pin"
    assert verdict["compared_whole"][CUDA]["files"] == [target]
    assert judged_on("l3", shipped, "cc86", without_the_reader)["state"] == pc.CARRY_SLICE
    # ... and a process that imported the file is the closure's to judge
    run["imported_source_sha256"][target] = "0" * 64
    assert judged_on("l3", shipped, "cc86", run)["state"] == pc.RERUN_CLOSURE


def test_a_reader_of_the_other_platform_s_table_asks_nothing(shipped):
    """The Metal arm table imports a Triton module and never emits from it."""
    reader = "meep_gpu/triton_kernels/conductive_bfast_fused_hd_pair.py"
    run = shipped_l3(shipped, G13)
    run["imported_source_sha256"][reader] = sha_of(shipped, reader)
    ledger = read_ledger(shipped, "metal")
    ledger["planted_narrative"] = {"purpose": "moved"}
    write_json(shipped, METAL, ledger)
    assert judged_on("l3", shipped, G13, run)["state"] == pc.CARRY_SLICE


# ------------------------------------------------------------------------- timing records ----

CC90 = "meep_gpu/triton_kernels/timing_cc90.json"
CC90_CUDA = "meep_gpu/cuda_kernels/timing_cc90.json"
CC89 = "meep_gpu/triton_kernels/timing_cc89.json"
G15_RECORD = "meep_gpu/metal_kernels/timing_applegpu_g15s.json"
CC86_RECORD = "meep_gpu/triton_kernels/timing_cc86.json"
G13_RECORD = "meep_gpu/metal_kernels/timing_applegpu_g13s.json"


def plant_timing(root, name, source, stamp):
    """A sealed record: the real ``source``, its rows stamped ``stamp``."""
    record = json.loads((pathlib.Path(root) / source).read_text())
    record.pop(dispatch_preference.DIGEST_FIELD)
    for host in record["host"]:
        host["compute_capability"] = stamp
    return write_json(root, name, dispatch_preference.seal(record))


@pytest.mark.parametrize("kind", sorted(SHIPPED_RUNS))
def test_a_new_record_of_another_architecture_leaves_a_row_whose_own_holds(shipped, kind):
    run = SHIPPED_RUNS[kind](shipped)
    plant_timing(shipped, CC90, CC86_RECORD, "9.0")
    verdict = judged_on(kind, shipped, "cc86", run)
    assert verdict["state"] == pc.CARRY_SLICE, verdict
    assert verdict["proof"]["set_aside"][CC90]["why"] == "own_architecture_holds"


@pytest.mark.parametrize("kind", sorted(SHIPPED_RUNS))
@pytest.mark.parametrize("device, name, source, stamp", [
    ("cc86", G15_RECORD, G13_RECORD, "applegpu_g15s"), (G13, CC90, CC86_RECORD, "9.0")])
def test_another_vendor_s_new_record_is_not_compared(shipped, kind, device, name, source, stamp):
    run = SHIPPED_RUNS[kind](shipped, device)
    plant_timing(shipped, name, source, stamp)
    verdict = judged_on(kind, shipped, device, run)
    assert verdict["state"] == pc.CARRY_SLICE, verdict
    assert verdict["proof"]["set_aside"][name]["why"] == "other_vendor"


def test_another_vendor_s_record_is_compared_on_a_run_that_planned_there(shipped):
    run = shipped_l3(shipped, G13, table="cuda")
    plant_timing(shipped, CC90, CC86_RECORD, "9.0")
    verdict = judged_on("l3", shipped, G13, run)
    assert verdict["state"] == pc.OWED_PLAN
    assert only_why(verdict, CC90) == "other_platform_recorded"


def test_an_own_record_moved_beside_another_names_only_the_own(shipped):
    run = shipped_l3(shipped)
    plant_timing(shipped, CC90, CC86_RECORD, "9.0")
    record = json.loads((shipped / CC86_RECORD).read_text())
    record.pop(dispatch_preference.DIGEST_FIELD)
    record["host"] = copy.deepcopy(record["host"])
    write_json(shipped, CC86_RECORD, dispatch_preference.seal(dict(record, note="moved")))
    verdict = judged_on("l3", shipped, "cc86", run)
    assert verdict["state"] == pc.OWED_PLAN
    assert only_why(verdict, CC86_RECORD) == "own_architecture"
    # The other table's own-architecture record still holds, so 9.0's is set aside.
    assert verdict["moved"] == [CC86_RECORD]


def test_an_own_record_absent_when_the_row_ran_and_present_now_owes_a_plan(shipped):
    bind_nvidia(shipped, "9.0")
    for table in ("triton", "cuda"):
        ledger_scope.archive_state(own_archive(shipped).parents[2], table,
                                   shipped / TABLE_LEDGER[table])
    run = shipped_l3(shipped, "cc90")
    plant_timing(shipped, CC90, CC86_RECORD, "9.0")
    verdict = judged_on("l3", shipped, "cc90", run)
    assert verdict["state"] == pc.OWED_PLAN and only_why(verdict, CC90) == "own_architecture"


def test_a_name_no_record_spells_is_compared(shipped):
    run = shipped_l3(shipped)
    write_json(shipped, "meep_gpu/triton_kernels/timing_foo.json", {"record_sha256": "a"})
    verdict = judged_on("l3", shipped, "cc86", run)
    assert verdict["state"] == pc.OWED_PLAN
    assert only_why(verdict, "meep_gpu/triton_kernels/timing_foo.json") == "timing_name_unread"


def test_a_record_that_will_not_load_is_compared(shipped):
    run = shipped_l3(shipped)
    write_json(shipped, CC90, {"record_sha256": "a"})
    verdict = judged_on("l3", shipped, "cc86", run)
    assert verdict["state"] == pc.OWED_PLAN and only_why(verdict, CC90) == "timing_record_refused"


def test_a_file_named_for_one_architecture_and_stamped_another_is_refused_by_name(shipped):
    """The prior pools by the rows' stamp, so a file named ``cc90`` whose rows stamp 8.6 feeds
    the 8.6 prior; a name is never taken on its word."""
    run = shipped_l3(shipped)
    plant_timing(shipped, CC90, CC86_RECORD, "8.6")
    record = dispatch_preference.load_record(str(shipped / CC90))
    assert dispatch_preference.record_capabilities(record) == ("8.6",)
    verdict = judged_on("l3", shipped, "cc86", run)
    assert verdict["state"] == pc.OWED_PLAN
    assert only_why(verdict, CC90) == "timing_name_disagrees_with_stamp"
    assert verdict["compared_whole"][CC90]["stamped"] == ["8.6"]


def test_a_record_of_the_wrong_vendor_s_table_is_refused_by_name(shipped):
    run = shipped_l3(shipped)
    name = "meep_gpu/metal_kernels/timing_cc90.json"
    plant_timing(shipped, name, G13_RECORD, "9.0")
    verdict = judged_on("l3", shipped, "cc86", run)
    assert verdict["state"] == pc.OWED_PLAN
    assert only_why(verdict, name) == "timing_name_on_another_table"


@pytest.mark.parametrize("spelled", ["applegpu_g16", "applegpu_m5", "applegpu_g99zz"])
def test_a_record_whose_distance_cannot_be_read_is_compared(shipped, spelled):
    """It loads and times its architecture, and the nearest rule then refuses every Apple
    consult: ``prior_architecture_unparsed`` before the own-architecture check."""
    run = shipped_l3(shipped, G13)
    name = f"meep_gpu/metal_kernels/timing_{spelled}.json"
    plant_timing(shipped, name, G13_RECORD, spelled)
    assert dispatch_preference.architecture_number("apple", spelled) is None
    verdict = judged_on("l3", shipped, G13, run)
    assert verdict["state"] == pc.OWED_PLAN
    assert only_why(verdict, name) == "timing_architecture_unreadable"


@pytest.mark.parametrize("record", ["refused", "absent", "moved"])
def test_another_architecture_s_record_is_compared_when_the_own_does_not_hold(shipped, record):
    run = shipped_l3(shipped)
    plant_timing(shipped, CC90, CC86_RECORD, "9.0")
    for name in (CC86_RECORD, "meep_gpu/cuda_kernels/timing_cc86.json"):
        path = shipped / name
        if record == "refused":
            path.unlink()
            path.write_text('{"record_sha256": "a"}')
        elif record == "absent":
            path.unlink()
        else:
            document = json.loads(path.read_text())
            document.pop(dispatch_preference.DIGEST_FIELD)
            write_json(shipped, name, dispatch_preference.seal(dict(document, note="moved")))
    verdict = judged_on("l3", shipped, "cc86", run)
    assert verdict["state"] == pc.OWED_PLAN
    assert only_why(verdict, CC90) == "own_architecture_not_held"


def test_an_own_record_that_measures_no_key_does_not_hold(shipped):
    run = shipped_l3(shipped)
    plant_timing(shipped, CC90, CC86_RECORD, "9.0")
    for name in (CC86_RECORD, "meep_gpu/cuda_kernels/timing_cc86.json"):
        document = json.loads((shipped / name).read_text())
        document.pop(dispatch_preference.DIGEST_FIELD)
        document["keys"] = {key: dict(entry, status="unmeasured") for key, entry in
                            document["keys"].items()}
        try:
            sealed = dispatch_preference.seal(document)
        except dispatch_preference.RecordRefused:  # the format refuses it: as good as absent
            sealed = {"record_sha256": "a"}
        write_json(shipped, name, sealed)
    assert only_why(judged_on("l3", shipped, "cc86", run), CC90) == "own_architecture_not_held"


@pytest.mark.parametrize("device, why", [("cc86", "own_architecture"),
                                         ("cc89", "own_architecture_not_held"),
                                         ("cc90", "own_architecture_holds")])
def test_a_card_s_own_record_reads_by_its_stamp(shipped, device, why):
    """A second 8.6 card's file pools into the 8.6 prior, so an 8.6 row compares it; 8.9 has
    no record and reads the nearest, 8.6; 9.0's own record answers for 9.0."""
    if device != "cc86":
        bind_nvidia(shipped, device[2] + "." + device[3])
        for table in ("triton", "cuda"):
            ledger_scope.archive_state(own_archive(shipped).parents[2], table,
                                       shipped / TABLE_LEDGER[table])
    if device == "cc90":
        plant_timing(shipped, CC90, CC86_RECORD, "9.0")
    run = shipped_l3(shipped, device)
    source = dispatch_preference.load_record(str(shipped / CC86_RECORD))
    slug = dispatch_preference.card_slug(dispatch_preference.record_devices(source)[0])
    name = f"meep_gpu/triton_kernels/timing_cc86-{slug}.json"
    plant_timing(shipped, name, CC86_RECORD, "8.6")
    verdict = judged_on("l3", shipped, device, run)
    if why == "own_architecture_holds":
        assert verdict["state"] == pc.CARRY_SLICE
        assert verdict["proof"]["set_aside"][name]["why"] == why
    else:
        assert verdict["state"] == pc.OWED_PLAN and only_why(verdict, name) == why


def test_an_8_9_row_with_no_8_9_record_compares_9_0_s(shipped):
    """8.9's nearest timed architecture is 9.0 once it is timed: it does read that record."""
    bind_nvidia(shipped, "8.9")
    for table in ("triton", "cuda"):
        ledger_scope.archive_state(own_archive(shipped).parents[2], table,
                                   shipped / TABLE_LEDGER[table])
    run = shipped_l3(shipped, "cc89")
    plant_timing(shipped, CC90, CC86_RECORD, "9.0")
    verdict = judged_on("l3", shipped, "cc89", run)
    assert verdict["state"] == pc.OWED_PLAN
    assert only_why(verdict, CC90) == "own_architecture_not_held"


def test_a_tree_with_no_preference_module_compares_every_record(tree):
    run = route_leg(tree, "cc86")
    (tree / "meep_gpu/metal_kernels/timing_applegpu_g13s.json").write_text("x")
    verdict = judged("route", tree, "cc86", run)
    assert verdict["state"] == pc.OWED_PLAN


# ---------------------------------------------- the other vendor's record, under the planner ----

#: The real planner on the stand-in composer of the package's own wiring test, reading the
#: shipped records from a directory of their own: the Apple file's contents move and every plan
#: of three NVIDIA devices is compared, at two sizes either side of the prior's ceiling.
APPLE_RECORD_CHILD = r"""
import copy, json, os, pathlib, shutil, sys, tempfile
OPENED = {}
def _hook(event, args):
    if event == "open" and args and isinstance(args[0], str):
        head, name = os.path.split(args[0])
        if name.startswith("timing_") or name in ("fingerprints.json", "certification.json"):
            key = os.path.basename(head) + "/" + name
            OPENED[key] = OPENED.get(key, 0) + 1
sys.addaudithook(_hook)
import pytest
sys.path.append(os.path.join(os.getcwd(), "parity", "meep_gpu"))
import selected_plan
from meep_gpu import dispatch_preference as dp, fastpath, subnormal_policy as policy_module
from meep_gpu import test_dispatch_preference_wiring as wiring
APPLE = "metal_kernels/timing_applegpu_g13s.json"
RECORDS = ("triton_kernels/timing_cc86.json", "cuda_kernels/timing_cc86.json", APPLE)
PACKAGE = pathlib.Path("meep_gpu").resolve()
SHIPPED = json.loads((PACKAGE / APPLE).read_text())
def sealed(**changes):
    document = {key: value for key, value in copy.deepcopy(SHIPPED).items()
                if key != dp.DIGEST_FIELD}
    document.update(changes)
    return dp.seal(document)
def stamped(architecture):
    return sealed(host=[dict(SHIPPED["host"][0], compute_capability=architecture)])
def of_table(table, architecture, entry=None):
    # By default one bracketed keep from 1,000 cells up, which lowers the pool's keep floor below
    # every veto.
    facts = wiring._route_facts()
    digest = facts.pop("deposit_repair_sha256")
    return wiring._record([entry or wiring._keep_d(lo=1000, hi=2400000)], table=table, route=digest,
                          barrier=wiring._barrier(table),
                          host=[{"device": "STAND-IN", "compute_capability": architecture}],
                          **facts)
barrier = copy.deepcopy(SHIPPED["subject_barrier"])
barrier["sources"][0] = []
VARIANTS = {"shipped": None, "keep_everywhere": of_table("metal", "applegpu_g13s"),
            "veto_everywhere": of_table("metal", "applegpu_g13s",
                                        wiring._slow_d(lo=1000, hi=90000000)),
            "no_keys": sealed(keys={}),
            "stamped_g15s": stamped("applegpu_g15s"), "stamped_m5": stamped("applegpu_m5"),
            "unsealed": dict(SHIPPED, cut={"edited": True}), "unparseable": b"{not json",
            "not_utf8": b"\xff\xfe", "empty": {}, "a_list": [], "absent": "absent",
            "a_directory": "directory", "raises": sealed(subject_barrier=barrier),
            "triton_8_6": of_table("triton", "8.6"), "cuda_9_0": of_table("cuda", "9.0")}
SHAPES = {"24000": None, "1200000": wiring.BETWEEN}
def plan(root, capability, shape):
    patch = pytest.MonkeyPatch()
    try:
        policy_module._reset_for_tests()
        fastpath.reset_dispatch_announcements()
        fastpath._POLICY_INSTALLED_BY_DISPATCH = None
        for name in (fastpath.FUSED_KILL_SWITCH, fastpath.SUBNORMAL_INSTALL_SWITCH,
                     fastpath.DISPATCH_LOG, fastpath.FUSE_ARMS_SWITCH,
                     fastpath.BACKEND_PREFERENCE_SWITCH, fastpath.KERNEL_TABLE_SWITCH,
                     fastpath.HARDWARE_SWITCH, fastpath.PREFERENCE_SWITCH,
                     policy_module.POLICY_ENV):
            patch.delenv(name, raising=False)
        patch.setenv(fastpath.DISPATCH_ENABLE, "1")
        patch.setenv(fastpath.WARM_SWITCH, "0")
        patch.setattr(fastpath, "TIMING_RECORD_ROOT", str(root))
        _plan, report = wiring._freeze(patch, wiring.Composer(), capability=capability, shape=shape)
    finally:
        patch.undo()
    prior = (report.get("preference") or {}).get("prior") or {}
    candidates = (report.get("preference") or {}).get("candidates") or ()
    item = next((one for one in candidates if one.get("label") == "fused pair D"), {})
    return {"plan": selected_plan.selected_plan_sha256(report), "decision": report.get("decision"),
            "refused": str(report.get("refused_because") or ""),
            "step_D": wiring._arms(report).get("step_D"), "rule": item.get("rule"),
            "prior_nvidia": json.dumps((prior.get("by_vendor") or {}).get("nvidia"),
                                       sort_keys=True, default=str),
            "reads_apple": any(entry.get("path", "").endswith("applegpu_g13s.json")
                               for entry in prior.get("derived_from") or ())}
out = {}
with tempfile.TemporaryDirectory() as tmp:
    root = pathlib.Path(tmp)
    for rel in RECORDS:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PACKAGE / rel, root / rel)
    target = root / APPLE
    for name, content in VARIANTS.items():
        if target.is_dir():
            target.rmdir()
        elif target.exists():
            target.unlink()
        if content == "directory":
            target.mkdir()
        elif content is None:
            shutil.copyfile(PACKAGE / APPLE, target)
        elif isinstance(content, bytes):
            target.write_bytes(content)
        elif content != "absent":
            target.write_text(json.dumps(content))
        out[name] = {f"{major}.{minor}|{size}": plan(root, (major, minor), shape)
                     for major, minor in ((8, 6), (8, 9), (9, 0))
                     for size, shape in SHAPES.items()}
print(json.dumps({"opened": OPENED, "plans": out}))
"""

#: Contents the planner reads and that reach no NVIDIA decision.
APPLE_FILE_MOVES_NO_PLAN = ("keep_everywhere", "veto_everywhere", "no_keys", "stamped_g15s",
                            "stamped_m5", "unsealed", "unparseable", "not_utf8", "empty", "a_list",
                            "absent", "a_directory")


@pytest.fixture(scope="module")
def apple_plans(tmp_path_factory):
    return child(APPLE_RECORD_CHILD, tmp_path_factory.mktemp("apple_record"))


def test_an_nvidia_plan_opens_the_apple_record_and_the_prior_uses_it(apple_plans):
    """Not vacuous: the file is opened on this path, and when it is the shipped record it is
    used, so the vendor filter is what stands between it and the plan."""
    assert apple_plans["opened"]["metal_kernels/timing_applegpu_g13s.json"] >= 1
    shipped = apple_plans["plans"]["shipped"]
    assert all(one["decision"] == "dispatched" for one in shipped.values()), shipped
    assert any(one["reads_apple"] for one in shipped.values()), shipped
    assert len({one["plan"] for one in shipped.values()}) > 1, "both sizes make one plan"
    for variant in ("keep_everywhere", "veto_everywhere"):
        assert any(one["reads_apple"] for one in apple_plans["plans"][variant].values()), variant


@pytest.mark.parametrize("variant", APPLE_FILE_MOVES_NO_PLAN)
def test_what_the_apple_file_holds_moves_no_nvidia_plan(apple_plans, variant):
    shipped = apple_plans["plans"]["shipped"]
    moved = apple_plans["plans"][variant]
    assert {case: one["plan"] for case, one in moved.items()} == {
        case: one["plan"] for case, one in shipped.items()}
    assert {case: one["prior_nvidia"] for case, one in moved.items()} == {
        case: one["prior_nvidia"] for case, one in shipped.items()}


def test_an_apple_record_that_moved_is_set_aside_for_an_nvidia_row(shipped):
    """The same move through the tool: the plan does not move, so neither does the row."""
    run = shipped_l3(shipped)
    record = json.loads((shipped / G13_RECORD).read_text())
    record.pop(dispatch_preference.DIGEST_FIELD)
    record["keys"].pop(sorted(record["keys"])[0])
    write_json(shipped, G13_RECORD, dispatch_preference.seal(record))
    verdict = judged_on("l3", shipped, "cc86", run)
    assert verdict["state"] == pc.CARRY_SLICE, verdict
    assert verdict["proof"]["set_aside"][G13_RECORD]["why"] == "other_vendor"


@pytest.mark.parametrize("variant, device", [("triton_8_6", "8.9"), ("triton_8_6", "9.0"),
                                             ("cuda_9_0", "8.9"), ("cuda_9_0", "9.0")])
def test_a_record_of_an_nvidia_table_in_the_apple_file_moves_an_nvidia_plan(
        apple_plans, variant, device):
    """The reader takes the file with no table expected and pools it by what it holds. This is
    the comparison's power, and why the tool below reads the bytes and not the name."""
    case = f"{device}|24000"
    assert (apple_plans["plans"][variant][case]["plan"]
            != apple_plans["plans"]["shipped"][case]["plan"])
    assert (apple_plans["plans"][variant][case]["prior_nvidia"]
            != apple_plans["plans"]["shipped"][case]["prior_nvidia"])


def test_a_record_of_an_nvidia_table_in_the_apple_file_is_compared_by_name(shipped):
    run = shipped_l3(shipped)
    plant_timing(shipped, G13_RECORD, CC86_RECORD, "8.6")
    assert dispatch_preference.load_record(str(shipped / G13_RECORD))["table"] == "triton"
    verdict = judged_on("l3", shipped, "cc86", run)
    assert verdict["state"] == pc.OWED_PLAN
    assert only_why(verdict, G13_RECORD) == "timing_record_of_another_table"
    assert verdict["compared_whole"][G13_RECORD]["table"] == "triton"


def test_a_sealed_record_the_reader_raises_on_refuses_the_nvidia_planner(apple_plans):
    """``validate`` raises unnamed on a list among the barrier's sources, ``the_prior`` takes
    only a named refusal, and the planner refuses wherever the prior is asked. A defect of the
    reader, which stays where it is: the tool below does not set this file aside."""
    moved = apple_plans["plans"]["raises"]
    refused = [one for one in moved.values() if one["decision"] != "dispatched"]
    assert refused and all("TypeError" in one["refused"] for one in refused), moved


def test_a_sealed_record_the_reader_raises_on_is_compared_by_name(shipped):
    run = shipped_l3(shipped)
    record = json.loads((shipped / G13_RECORD).read_text())
    record.pop(dispatch_preference.DIGEST_FIELD)
    record["subject_barrier"]["sources"][0] = []
    write_json(shipped, G13_RECORD, dispatch_preference.seal(record))
    with pytest.raises(TypeError):
        dispatch_preference.load_record(str(shipped / G13_RECORD))
    verdict = judged_on("l3", shipped, "cc86", run)
    assert verdict["state"] == pc.OWED_PLAN
    assert only_why(verdict, G13_RECORD) == "timing_record_refused"


@pytest.mark.parametrize("damage, why", [("unparseable", "timing_record_refused"),
                                         ("unsealed", "timing_record_refused"),
                                         ("absent", "timing_record_absent"),
                                         ("stamped_g15s", "timing_name_disagrees_with_stamp")])
def test_what_is_no_apple_record_of_this_name_is_compared_though_no_plan_reads_it(
        shipped, damage, why):
    """No plan moves on these (above). The tool cannot tell them from what would move one, so
    it compares them, under a reason of its own."""
    run = shipped_l3(shipped)
    path = shipped / G13_RECORD
    if damage == "stamped_g15s":
        plant_timing(shipped, G13_RECORD, G13_RECORD, "applegpu_g15s")
    elif damage == "unsealed":
        write_json(shipped, G13_RECORD, dict(json.loads(path.read_text()), cut={"edited": True}))
    else:
        path.unlink()
        if damage == "unparseable":
            path.write_text("{not json")
    verdict = judged_on("l3", shipped, "cc86", run)
    assert verdict["state"] == pc.OWED_PLAN and only_why(verdict, G13_RECORD) == why


# --------------------------------------------------------------- plan passes and the re-cut ----

def test_a_plan_pass_on_another_slice_equal_view_stands(shipped):
    """Cut after the timing record moved and before another device's bind: its view is not
    the tree's, and nothing it read moved."""
    run = shipped_route(shipped)
    record = json.loads((shipped / CC86_RECORD).read_text())
    record.pop(dispatch_preference.DIGEST_FIELD)
    write_json(shipped, CC86_RECORD, dispatch_preference.seal(dict(record, note="moved")))
    passed = on_device(carry_fixtures.plan_pass(shipped), "cc86")
    passed["imported_source_sha256"].update(closure_of(shipped, "cc86"))
    passed["provenance"]["triton"] = "3.1.0"
    bind_nvidia(shipped, "9.0")
    assert passed[plan_key.INPUTS_FIELD]["ledgers"] != plan_key.plan_inputs(str(shipped))["ledgers"]
    verdict = judged_on("route", shipped, "cc86", run, passed)
    assert verdict["state"] == pc.CARRY_PLAN, verdict
    assert set(verdict["proof"]["set_aside"]) == {TRITON, CUDA}


def test_a_plan_pass_on_a_view_that_moved_what_the_device_reads_cannot_stand(shipped):
    run = shipped_route(shipped)
    record = json.loads((shipped / CC86_RECORD).read_text())
    record.pop(dispatch_preference.DIGEST_FIELD)
    write_json(shipped, CC86_RECORD, dispatch_preference.seal(dict(record, note="moved")))
    passed = on_device(carry_fixtures.plan_pass(shipped), "cc86")
    passed["imported_source_sha256"].update(closure_of(shipped, "cc86"))
    passed["provenance"]["triton"] = "3.1.0"
    ledger = read_ledger(shipped, "cuda")
    key = cited_of(shipped)["cuda"][0]
    ledger[key]["runs"]["8.6"]["bound_sha256"] = "moved"
    write_json(shipped, CUDA, ledger)
    live = plan_key.plan_inputs(str(shipped))["ledgers"]
    run[plan_key.INPUTS_FIELD]["ledgers"][CUDA] = live[CUDA]
    verdict = judged_on("route", shipped, "cc86", run, passed)
    assert verdict["state"] == pc.OWED_PLAN and "other planner data" in verdict["why"]


def test_a_carried_leg_re_verifies_through_the_carry_record(shipped, tmp_path):
    source = tmp_path / "campaign" / "leg"
    source.mkdir(parents=True)
    run = shipped_route(shipped)
    run["release"] = {"released": True, "reasons": []}
    (source / "gate.json").write_text(json.dumps(run))
    bind_nvidia(shipped, "9.0")
    carry = {"schema": pc.SCHEMA, "kind": "route", "device": "cc86",
             "legs": {"leg": {"state": pc.CARRY_SLICE, "files": pc.tree_files(source)}}}
    carried, problems = pc.verify_route_carry(carry, source.parent, shipped)
    assert problems == [] and carried["leg"]["state"] == pc.CARRY_SLICE
    # Supersedes 8.9 beside 8.6: the moved bytes strand both live runs, which must be named.
    bind_nvidia(shipped, "9.0", moved_pin="meep_gpu/fastpath.py", supersede=("8.6", "8.9"))
    carried, problems = pc.verify_route_carry(carry, source.parent, shipped)
    assert not carried and any("carried_leg_no_longer_holds" in line for line in problems)


# ---------------------------------------------------------- what a bind may not do by itself ----

def test_every_listed_triton_version_has_a_run_on_every_admitted_capability(real):
    """The list is shared and is the only Triton gate at dispatch, so a version added for one
    capability reads certified on all of them. The reader cannot be narrowed here; a ledger
    that would make it false is refused. The cost: a new Triton needs every capability
    re-run before any is certified on it."""
    ledgers, cited = real
    assert not triton_list_problems(ledgers["triton"], cited["triton"])


def triton_list_problems(ledger, cited):
    listed = [str(version) for version in ledger.get("validated_triton_versions") or ()]
    version_in_host = re.compile(r"Triton\s+([0-9][^\s;,)]*)")

    def version_of(run):
        return run.get("triton_version") or next(iter(
            version_in_host.findall(str(run.get("host")))), None)

    problems = []
    for capability in fastpath.table_capabilities(ledger, list(cited)):
        seen = {key: version_of(ledger[key]["runs"][capability]) for key in cited}
        for version in listed:
            short = [key for key in cited if seen[key] != version]
            if short:
                problems.append(f"{version} is listed and {len(short)} of {len(cited)} cited welds "
                                f"have no {capability} run on it")
        problems += [f"{key}: the {capability} run names {version}, not listed"
                     for key, version in seen.items() if version not in listed]
    return problems


def test_the_list_guard_names_a_version_no_run_covers(real):
    ledgers, cited = real
    widened = copy.deepcopy(ledgers["triton"])
    widened["validated_triton_versions"].append("3.2.0")
    assert any("3.2.0 is listed" in line for line in triton_list_problems(widened, cited["triton"]))
    emptied = copy.deepcopy(ledgers["triton"])
    emptied["validated_triton_versions"] = []
    assert any("not listed" in line for line in triton_list_problems(emptied, cited["triton"]))


def _ledger_with_cupy(*by_capability):
    bound = {"source_sha256": {"a.py": "0" * 64}}
    digest = fastpath.bound_digest(bound)
    return {"weld": dict(bound, runs={capability: {"bound_sha256": digest, "cupy_version": cupy}
                                      for capability, cupy in by_capability})}


def test_another_capability_s_cupy_widens_nothing_for_this_one(monkeypatch):
    monkeypatch.setattr(rebind_triton_welds.fastpath, "validated_triton_versions",
                        lambda: ("3.1.0",))
    ledger = _ledger_with_cupy(("8.6", "13.5.1"), ("9.0", "14.0"))
    assert rebind_triton_welds.recorded_cupy_versions(ledger, "8.6") == ("13.5.1",)
    run = {"triton": "3.1.0", "cupy": "14.0", "compute_capability": "8.6"}
    assert "outside" in rebind_triton_welds.toolchain_refusal(run, ledger)
    assert rebind_triton_welds.toolchain_refusal(dict(run, cupy="13.5.1"), ledger) is None


def test_a_first_bind_is_held_to_the_cupy_the_table_stands_on(monkeypatch):
    monkeypatch.setattr(rebind_triton_welds.fastpath, "validated_triton_versions",
                        lambda: ("3.1.0",))
    ledger = _ledger_with_cupy(("8.6", "13.5.1"))
    assert rebind_triton_welds.recorded_cupy_versions(ledger, "9.0") == ("13.5.1",)
    assert rebind_triton_welds.recorded_cupy_versions(ledger) == ("13.5.1",)


def test_a_bind_that_drops_a_policy_the_run_holds_is_refused_by_name():
    entry = {"source_sha256": {"a.py": "0" * 64},
             "runs": {"8.6": {"certified_policies": {"keep": {}, "flush": {}}}}}
    keep_only = {"certified_policies": {"keep": {}}}
    with pytest.raises(fastpath.CapabilityRecordError, match="policy_dropped.*flush"):
        rebind_triton_welds.refuse_policy_drop(entry, "8.6", keep_only, False)
    rebind_triton_welds.refuse_policy_drop(entry, "8.6", keep_only, True)
    rebind_triton_welds.refuse_policy_drop(entry, "8.6", {"certified_policies": {
        "keep": {}, "flush": {}}}, False)
    rebind_triton_welds.refuse_policy_drop(entry, "9.0", keep_only, False)
    assert ledger_scope.policies_dropped(None, keep_only, "certified_policies") == []
    assert ledger_scope.policies_dropped(entry["runs"]["8.6"], {}, "certified_policies") == [
        "flush", "keep"]


@pytest.mark.parametrize("device", ["cc860", "86.0", "8.60", "08.6", "cc8"])
def test_a_spelling_that_keys_no_run_compares_every_timing_record(shipped, device):
    """Well formed or not, a spelling no ledger runs is not placed, on another vendor's
    record too."""
    plant_timing(shipped, G15_RECORD, G13_RECORD, "applegpu_g15s")
    verdict = ledger_scope.timing_record_moved(
        G15_RECORD, device, {name: "same" for name in TIMING_NAMES},
        {name: "same" for name in TIMING_NAMES}, shipped)
    assert verdict.moved and verdict.why in ("device_not_placed", "device_names_no_run")


# ------------------------------------------------- what the readers refuse, the tools refuse ----

ENCODINGS = {"bom": lambda raw: b"\xef\xbb\xbf" + raw,
             "utf16": lambda raw: raw.decode("utf-8").encode("utf-16"),
             "utf16le": lambda raw: raw.decode("utf-8").encode("utf-16-le"),
             "utf32": lambda raw: raw.decode("utf-8").encode("utf-32")}


def write_over(root, name, raw):
    """Replace one linked file of a judged tree; the file the link names stays as it is."""
    path = pathlib.Path(root) / name
    if path.is_symlink():
        path.unlink()
    path.write_bytes(raw)
    return path


def re_encoded(root, name, encoding, moved):
    """The live ledger in ``encoding``, also moved outside the slice when ``moved``."""
    document = json.loads((pathlib.Path(root) / name).read_text(encoding="utf-8"))
    if moved:
        document["planted_narrative"] = {"purpose": "moved"}
    text = json.dumps(document, indent=2, sort_keys=True) + "\n"
    write_over(root, name, ENCODINGS[encoding](text.encode("utf-8")))


@pytest.mark.parametrize("moved", [False, True], ids=["same_view", "slice_equal"])
@pytest.mark.parametrize("encoding", sorted(ENCODINGS))
@pytest.mark.parametrize("kind, device, name", [("l3", "cc86", TRITON), ("l3", "cc86", CUDA),
                                                ("l3", G13, METAL), ("route", "cc86", TRITON),
                                                ("route", G13, METAL)])
def test_a_ledger_the_package_cannot_read_is_never_set_aside(shipped, kind, device, name,
                                                             encoding, moved):
    """Every package reader takes ``open(encoding="utf-8")``, which refuses a byte-order mark
    and UTF-16 or UTF-32: that device's table goes unreadable and its plans change. So the
    comparison must not read such a file as the same ledger, or as a slice-equal one."""
    run = SHIPPED_RUNS[kind](shipped, device)
    re_encoded(shipped, name, encoding, moved)
    assert "error" in plan_key.plan_inputs(str(shipped))["ledgers"][name]
    verdict = judged_on(kind, shipped, device, run)
    assert verdict["state"] == pc.RERUN_INPUTS, verdict
    assert only_why(verdict, name) == "live_ledger_unreadable"


@pytest.mark.parametrize("encoding", sorted(ENCODINGS))
def test_a_timing_row_owes_on_a_ledger_the_package_cannot_read(shipped, encoding):
    row = shipped_timing(shipped)
    re_encoded(shipped, TRITON, encoding, moved=True)
    compared = TR.ledger_comparison(row, row["processes"][0], str(shipped))
    assert compared[TRITON].moved and compared[TRITON].why == "live_ledger_unreadable"


@pytest.mark.parametrize("encoding", sorted(ENCODINGS))
def test_a_state_the_package_could_not_read_answers_for_nothing(shipped, encoding):
    """Such bytes were never a ledger a run read as one, so no row carries on them."""
    raw = ENCODINGS[encoding](b'{"a": 1}\n')
    digest = hashlib.sha256(raw).hexdigest()
    directory = own_archive(shipped) / "cuda"
    (directory / f"{digest}.json.gz").write_bytes(gzip.compress(raw, mtime=0))
    verdict = ledger_scope.ledger_moved(
        CUDA, digest, form="raw", device="8.6", imports=closure_of(shipped, "8.6"),
        triton_version="3.1.0", root=shipped)
    assert verdict.moved and verdict.why == "ledger_state_refused"


@pytest.mark.parametrize("encoding", sorted(ENCODINGS))
def test_archiving_refuses_what_the_package_cannot_read(tmp_path, encoding):
    source = tmp_path / "ledger.json"
    source.write_bytes(ENCODINGS[encoding](b'{"a": 1}\n'))
    with pytest.raises(ValueError):
        ledger_scope.archive_state(tmp_path, "cuda", source)
    assert not (tmp_path / ledger_scope.STATES_DIR).exists()


@pytest.mark.parametrize("answer", [{"error": "the reader refused it"}, "0" * 64, None])
def test_a_ledger_the_recorder_read_otherwise_is_not_set_aside(shipped, answer):
    """``plan_key`` said it moved, so a digest equal on the tool's own read is no answer."""
    run = shipped_l3(shipped)
    live = plan_key.plan_inputs(str(shipped))
    live["ledgers"][CUDA] = answer
    verdict = pc.judge_l3_row(run, shipped, live, "cc86", None)
    assert verdict["state"] == pc.RERUN_INPUTS, verdict
    assert CUDA in verdict["moved"]


# ------------------------------------------------------ a stale pin is no pin that holds ----

ADE_READER = "meep_gpu/cuda_kernels/fused_polarization_pair.py"
ADE_TARGET = "meep_gpu/cuda_kernels/ade_kernels.py"


def row_beside_a_stale_weld(root, dead_pin):
    """A row of a ledger state whose uncited entry pins the file a reader takes as text at
    other bytes, and those bytes, as a weld left behind by an older tree does. ``dead_pin``:
    the entry pins another package file at bytes the tree no longer holds."""
    target = pathlib.Path(root) / ADE_TARGET
    older = target.read_bytes() + b"\n# older\n"
    pins = {ADE_TARGET: hashlib.sha256(older).hexdigest()}
    if dead_pin:
        pins[NVIDIA_FILES[1]] = "0" * 64
    ledger = read_ledger(root, "cuda")
    ledger["stale_weld"] = {"source_sha256": pins, "status": "PASS"}
    write_json(root, CUDA, ledger)
    own_archive(root)
    ledger_scope.archive_state(root, "cuda", pathlib.Path(root) / CUDA)
    run = shipped_l3(root)
    run["imported_source_sha256"][ADE_READER] = sha_of(root, ADE_READER)
    return run, older


def test_a_file_read_as_text_set_back_to_a_stale_pin_compares_whole(shipped):
    """That weld's run still binds its digest, so it reads as live: what shows it stale is
    the other package file it pins, which is not at those bytes either."""
    run, older = row_beside_a_stale_weld(shipped, dead_pin=True)
    bind_nvidia(shipped, "9.0")
    assert judged_on("l3", shipped, "cc86", run)["state"] == pc.CARRY_SLICE
    write_over(shipped, ADE_TARGET, older)
    verdict = judged_on("l3", shipped, "cc86", run)
    assert verdict["state"] == pc.RERUN_INPUTS, verdict
    assert only_why(verdict, CUDA) == "text_read_left_its_pin"
    assert verdict["compared_whole"][CUDA]["files"] == [ADE_TARGET]


def test_a_file_read_as_text_whose_entries_disagree_compares_whole(shipped):
    """An entry pinning that file alone cannot be shown stale, so its pin and the others'
    both stand, and neither set of bytes is the one a run read."""
    run, older = row_beside_a_stale_weld(shipped, dead_pin=False)
    bind_nvidia(shipped, "9.0")
    for _ in range(2):
        verdict = judged_on("l3", shipped, "cc86", run)
        assert verdict["state"] == pc.RERUN_INPUTS, verdict
        assert only_why(verdict, CUDA) == "text_read_left_its_pin"
        write_over(shipped, ADE_TARGET, older)


def test_every_shipped_file_read_as_text_holds_at_one_pin(shipped):
    """The real stale weld does not reach a row at the tip, and no other such weld exists: a
    correct re-gate of it may empty the set, and nothing may join it unread."""
    tree = ledger_scope._Tree(shipped)  # noqa: SLF001
    holding = tree.pins_holding("nvidia")[1]
    for reader, targets in ledger_scope.TEXT_READS.items():
        for name in targets:
            target = f"{reader.rsplit('/', 1)[0]}/{name}"
            if target not in UNPINNED_TEXT_READS:
                assert holding[target] == {sha_of(shipped, target)}, target
    stale = {key for key, entry in read_ledger(shipped, "triton").items()
             if isinstance(entry, dict) and isinstance(entry.get("source_sha256"), dict)
             and entry["source_sha256"].get("meep_gpu/triton_kernels/kernels.py")
             not in (None, sha_of(shipped, "meep_gpu/triton_kernels/kernels.py"))}
    assert stale <= {"triton_cylindrical_device_gate"}


# --------------------------------------------------------------- labels that say nothing ----

NO_VERSION = ["", "unknown", "?", "not-imported", 0, "3", True]


@pytest.mark.parametrize("label", NO_VERSION)
def test_a_triton_label_that_is_no_version_keeps_the_whole_list(label):
    listed = {ledger_scope.VERSIONS_ENTRY: ["3.1.0"]}
    assert ledger_scope._versions_slice(listed, label) == {  # noqa: SLF001
        "validated_list": ["3.1.0"]}


@pytest.mark.parametrize("version", ["3.1.0", "3.1", "3.2.0rc1", "3.1.0+local"])
def test_a_triton_version_is_read_for_its_membership(version):
    listed = {ledger_scope.VERSIONS_ENTRY: ["3.1.0"]}
    assert ledger_scope._versions_slice(listed, version) == {  # noqa: SLF001
        "read": version, "validated": version == "3.1.0"}


@pytest.mark.parametrize("label", NO_VERSION)
def test_a_list_move_voids_a_row_whose_triton_label_is_no_version(shipped, label):
    run = shipped_l3(shipped, triton=label)
    row = shipped_timing(shipped, triton=label)
    ledger = read_ledger(shipped, "triton")
    ledger[ledger_scope.VERSIONS_ENTRY] = ledger[ledger_scope.VERSIONS_ENTRY] + ["9.9.9"]
    write_json(shipped, TRITON, ledger)
    verdict = judged_on("l3", shipped, "cc86", run)
    assert verdict["state"] == pc.RERUN_INPUTS, verdict
    assert only_why(verdict, TRITON) == "slice_moved"
    assert TRITON in TR.moved_files(row, str(shipped))


def test_two_triton_statements_that_disagree_keep_the_whole_list():
    both = {plan_key.TOOLCHAIN_FIELD: {"triton": "3.1.0"}, "provenance": {"triton": "3.2.0"}}
    assert pc.triton_version_of(both) is None
    both["provenance"]["triton"] = "3.1.0"
    assert pc.triton_version_of(both) == "3.1.0"
    del both["provenance"]["triton"]
    assert pc.triton_version_of(both) == "3.1.0"


@pytest.mark.parametrize("kind", ["ladder", "check"])
def test_a_row_whose_key_names_another_architecture_is_refused(shipped, kind):
    """A wrong stamp would cut the slice for the wrong device. The key is the row's second
    statement of it, written by the tool that keyed the unit."""
    row = shipped_timing(shipped, "8.6", kind=kind, key=TR.ladder_key("8.6", "pml_3d", 512000, "f"))
    stated = [reason for reason in TR.refusals(row, True) if reason.startswith("architecture")]
    assert stated == []
    row["architecture"] = "9.0"
    stated = [reason for reason in TR.refusals(row, True) if reason.startswith("architecture")]
    assert len(stated) == 1 and "8.6" in stated[0] and "9.0" in stated[0]
    assert TR.assess(row, str(shipped), TR.Attribution(str(shipped)),
                     allow_unrecorded_environment=True)["status"] == "refused"


def test_a_key_that_spells_the_architecture_as_a_device_is_not_refused(shipped):
    row = shipped_timing(shipped, "8.6")
    row["architecture"] = "cc86"
    assert not [reason for reason in TR.refusals(row, True) if reason.startswith("architecture")]


# ----------------------------------------------------------- failures compare, by name ----

@pytest.mark.parametrize("form", ["raw", "view"])
def test_a_state_that_will_not_inflate_is_refused_by_name(shipped, form):
    packed = gzip.compress(b'{"a": 1}' * 100, mtime=0)
    broken = packed[:10] + b"\xff" * 8 + packed[18:]
    with pytest.raises(zlib.error):
        gzip.decompress(broken)
    digest = "a" * 64
    (own_archive(shipped) / "cuda" / f"{digest}.json.gz").write_bytes(broken)
    _moved_cuda(shipped)
    verdict = ledger_scope.ledger_moved(
        CUDA, digest, form=form, device="8.6", imports=closure_of(shipped, "8.6"),
        triton_version="3.1.0", root=shipped)
    assert verdict.moved and verdict.why == "ledger_state_refused"


@pytest.mark.parametrize("form", ["raw", "view"])
def test_a_state_nested_past_the_parser_is_refused_by_name(shipped, form):
    raw = b"[" * 200000
    digest = hashlib.sha256(raw).hexdigest()
    (own_archive(shipped) / "cuda" / f"{digest}.json.gz").write_bytes(gzip.compress(raw, mtime=0))
    _moved_cuda(shipped)
    verdict = ledger_scope.ledger_moved(
        CUDA, digest, form=form, device="8.6", imports=closure_of(shipped, "8.6"),
        triton_version="3.1.0", root=shipped)
    assert verdict.moved and verdict.why == "ledger_state_refused"


def test_a_live_ledger_nested_past_the_parser_reads_as_unreadable(shipped):
    recorded = sha_of(shipped, CUDA)
    write_over(shipped, CUDA, b"[" * 200000)
    verdict = ledger_scope.ledger_moved(
        CUDA, recorded, form="raw", device="8.6", imports=closure_of(shipped, "8.6"),
        triton_version="3.1.0", root=shipped)
    assert verdict.moved and verdict.why == "live_ledger_unreadable"


@pytest.mark.parametrize("value", [["a"], {"a": 1}, 7, None])
@pytest.mark.parametrize("weld", ["ade_2026-08-19", "cited"])
def test_a_pin_that_is_no_digest_moves_the_slice_and_raises_nothing(shipped, weld, value):
    plain = shipped_l3(shipped)
    reading = shipped_l3(shipped)
    reading["imported_source_sha256"][ADE_READER] = sha_of(shipped, ADE_READER)
    key = cited_of(shipped)["cuda"][0] if weld == "cited" else weld
    ledger = read_ledger(shipped, "cuda")
    ledger[key]["source_sha256"]["meep_gpu/planted.py"] = value
    write_json(shipped, CUDA, ledger)
    verdict = judged_on("l3", shipped, "cc86", plain)
    assert verdict["state"] == pc.RERUN_INPUTS, verdict
    assert only_why(verdict, CUDA) == "slice_moved"
    # ... and a row holding a text reader asks the pins first: that entry no longer holds
    verdict = judged_on("l3", shipped, "cc86", reading)
    assert verdict["state"] == pc.RERUN_INPUTS, verdict
    assert only_why(verdict, CUDA) in ("slice_moved", "text_read_left_its_pin")


def test_any_other_failure_in_the_ledger_comparison_compares_whole(shipped, monkeypatch):
    def fails(*_args, **_kwargs):
        raise TypeError("planted")

    _moved_cuda(shipped)
    monkeypatch.setattr(ledger_scope, "ledger_slice", fails)
    verdict = ledger_scope.ledger_moved(
        CUDA, sha_of(REPO_ROOT, CUDA), form="raw", device="8.6",
        imports=closure_of(shipped, "8.6"), triton_version="3.1.0", root=shipped)
    assert verdict.moved and verdict.why == "comparison_failed"
    assert "planted" in verdict.proof["detail"]


def test_any_other_failure_in_the_timing_comparison_compares_whole(shipped, monkeypatch):
    class Fails:
        def record_name_parts(self, _name):
            raise TypeError("planted")

    monkeypatch.setattr(ledger_scope._Tree, "preference", lambda _self: Fails())  # noqa: SLF001
    verdict = ledger_scope.timing_record_moved(
        G13_RECORD, "8.6", {}, {}, shipped)
    assert verdict.moved and verdict.why == "comparison_failed"
    assert "planted" in verdict.proof["detail"]


# ---------------------------------------------------------------------------- the line limit ----

@pytest.mark.parametrize("name", ["ledger_scope.py", "test_ledger_scope.py"])
def test_these_files_keep_to_the_line_limit(name):
    text = pathlib.Path(HERE, name).read_text(encoding="utf-8")
    over = [number for number, line in enumerate(text.splitlines(), 1) if len(line) > 100]
    assert not over, f"{name}: lines over 100 columns: {over}"
