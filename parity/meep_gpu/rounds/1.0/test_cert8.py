"""Host tests for the release timing driver (``cert8.py``). No device, no timing.

The driver's units are commands of the release's own instruments; these tests read
them, check the release ladder accepts every one (its ``--dry-run`` starts no process),
run a session on stand-ins (a runner that writes what the instruments write, a reader
that plays the host), and check the gates the owner ruled: no stage gated on power,
MEEP's spread recorded and never gated, an intrusion that stops and re-runs a unit.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
for _path in (str(HERE), str(HARNESS), str(REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import cert8  # noqa: E402
import test_cut_timing_record as fixtures  # noqa: E402
import timing_rows as TR  # noqa: E402

NVIDIA = str(HERE / "cert8_catalog_nvidia.json")
APPLE = str(HERE / "cert8_catalog_apple.json")
DRIVE = {"triton": ["pml_2d", "pml_3d"], "cuda": ["pml_2d"]}


def args_for(tmp_path, **extra):
    values = dict(python=sys.executable, run_root=str(tmp_path / "runs"), session="s1",
                  tree=str(REPO_ROOT), gpu="0", bench_env=[], ladder_params=None,
                  store=str(tmp_path / "store"), role="release", no_variance=None,
                  environment_record=None, hold_s=0.0, watch_every_s=0.05)
    values.update(extra)
    return types.SimpleNamespace(**values)


# ---------------------------------------------------------------------------
# The catalogs and their keys
# ---------------------------------------------------------------------------

def test_the_shipped_catalogs_name_the_rows_of_the_1_0_round():
    nvidia = cert8.load_catalog(NVIDIA)
    keys = cert8.catalog_keys(nvidia, drive=lambda table: DRIVE[table])
    ladder = [k for k in keys if k.startswith("ladder|")]
    assert ladder == [f"ladder|8.6|pml_3d|{c}|priority" for c in (512000, 2097152, 7077888)]
    fused = [k for k in keys if k.startswith("fused|")]
    # DRIVE cases, then the sweep: triton 3 cases x 5 sizes x 2 passes, cuda 2 x 5 x 2
    assert len(fused) == 3 + 30 + 20
    assert "fused|8.6|triton|offdiag_2d|res300|p2" in fused
    apple = cert8.load_catalog(APPLE)
    keys = cert8.catalog_keys(apple)
    assert len(keys) == 5 and all(k.startswith("ladder|applegpu_g13s|pml_3d|") for k in keys)
    with pytest.raises(cert8.Refused):
        cert8.catalog_keys(nvidia)          # "release" DRIVE cases need their reader


def test_no_unit_is_gated_on_power(tmp_path):
    """Owner ruling: never gate a stage on AC power. Every ladder unit, the variance
    precondition and every ladder dispatch check pass ``--require-ac-power 0``."""
    for path in (NVIDIA, APPLE):
        catalog = cert8.load_catalog(path)
        keys = [k for k in cert8.catalog_keys(catalog, drive=lambda t: DRIVE[t])
                if k.startswith("ladder|")]
        units = cert8.build_units(args_for(tmp_path), catalog,
                                  keys + [TR.check_key(keys[0])])
        units.append(cert8.variance_unit(args_for(tmp_path), catalog))
        for unit in units:
            assert unit.argv[unit.argv.index("--require-ac-power") + 1] == "0", unit.name
            assert unit.argv.count("--require-ac-power") == 1


def test_ladder_rows_of_one_case_and_tier_share_one_ladder(tmp_path):
    catalog = cert8.load_catalog(APPLE)
    keys = cert8.catalog_keys(catalog)
    units = cert8.build_units(args_for(tmp_path), catalog, keys)
    assert [(u.name, len(u.keys)) for u in units] == [
        ("u01_ladder_pml_3d_priority", 3), ("u02_ladder_pml_3d_rest", 2)]
    # one row alone is one size
    one = cert8.build_units(args_for(tmp_path), catalog, [keys[1]])
    sizes = one[0].argv[one[0].argv.index("--priority-sizes") + 1]
    assert sizes == str(TR.parse_key(keys[1])["cells"])


def ladder_dry_run(unit):
    """The release ladder's own reading of a unit's command; no process starts."""
    argv = unit.argv[2:] + ["--dry-run", "--json", "--platform", "linux",
                            "--physical-cores", "48", "--logical-cpus", "96",
                            "--sockets", "2", "--threads-per-core", "2",
                            "--memory-total-gb", "1007", "--mpi-impl", "openmpi",
                            "--meep-config-log", "none", "--account", "user"]
    done = subprocess.run([sys.executable, "-u"] + argv, cwd=str(REPO_ROOT),
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    return json.loads(done.stdout[done.stdout.index("{"):])


def test_the_release_ladder_accepts_every_unit_and_plans_exactly_its_rows(tmp_path):
    catalog = cert8.load_catalog(NVIDIA)
    keys = [k for k in cert8.catalog_keys(catalog, drive=lambda t: DRIVE[t])
            if k.startswith("ladder|")]
    args = args_for(tmp_path)
    for unit in cert8.build_units(args, catalog, [keys[1], TR.check_key(keys[2])]):
        plan = ladder_dry_run(unit)
        tier = "preflight" if unit.kind == "check" else "priority"
        planned = {(row["case"], row["cells"]) for row in plan["rows"]
                   if row["tier"] == tier and row["kind"] in ("gpu", "meep")}
        rows = [TR.parse_key(TR.parse_key(k)["target"]) if unit.kind == "check"
                else TR.parse_key(k) for k in unit.keys]
        want = {(row["case"], row["cells"]) for row in rows}
        assert planned == want, (unit.name, planned)
        assert plan["resolved"]["out"] == unit.out
        assert int(plan["params"]["REQUIRE_AC_POWER"]["value"]) == 0
    # the variance precondition: one point, one route, one MEEP rank count, five passes
    variance = cert8.variance_unit(args, catalog)
    plan = ladder_dry_run(variance)
    timed = [row for row in plan["rows"] if row["tier"] == "priority"
             and row["kind"] in ("gpu", "meep")]
    assert {(row["kind"], row["cells"], row.get("route"), row.get("ranks"))
            for row in timed} == {("gpu", 512000, "default", None),
                                  ("meep", 512000, None, 32)}
    assert sorted({row["pass"] for row in timed}) == [1, 2, 3, 4, 5]
    assert plan["resolved"]["controls"] == []


def test_a_fused_unit_and_its_dispatch_check_run_the_same_bench_case(tmp_path):
    catalog = cert8.load_catalog(NVIDIA)
    key = TR.fused_key("8.6", "cuda", "pml_2d", "res60", 2)
    unit, check = cert8.build_units(args_for(tmp_path), catalog, [key, TR.check_key(key)])
    assert unit.argv[2] == "parity/meep_gpu/bench_fused_products.py"
    assert unit.argv[unit.argv.index("--res") + 1] == "60"
    assert unit.env["MEEP_GPU_BACKEND_PREFERENCE"] == "cuda"
    assert unit.env["MEEP_GPU_HARDWARE"] == "certified"
    out = unit.argv.index("--out")
    assert check.argv[:out + 1] == unit.argv[:out + 1]
    assert check.argv[out + 2:len(unit.argv)] == unit.argv[out + 2:]
    assert check.argv[len(unit.argv):] == cert8.CHECK_BENCH_FLAGS


def test_the_sweep_runs_odd_passes_up_and_even_passes_down(tmp_path):
    catalog = cert8.load_catalog(NVIDIA)
    keys = [k for k in cert8.catalog_keys(catalog, drive=lambda t: DRIVE[t])
            if k.startswith("fused|") and "|cuda|pml_2d|res" in k]
    units = cert8.build_units(args_for(tmp_path), catalog, keys)
    order = [(TR.parse_key(u.keys[0])["pass"], TR.parse_key(u.keys[0])["size"])
             for u in units]
    assert order == [(1, f"res{r}") for r in (20, 60, 130, 200, 300)] + \
        [(2, f"res{r}") for r in (300, 200, 130, 60, 20)]


def test_a_key_outside_the_catalog_is_refused(tmp_path):
    code = cert8.main(["plan", "--catalog", APPLE, "--store", str(tmp_path / "s"),
                       "--run-root", str(tmp_path / "runs"),
                       "--rows", "ladder|applegpu_g13s|pml_3d|999|priority"])
    assert code == cert8.EXIT_REFUSED


# ---------------------------------------------------------------------------
# The gates
# ---------------------------------------------------------------------------

def test_the_watcher_stops_a_unit_on_two_intrusive_readings_in_a_row():
    seen = []
    watcher = cert8.Watcher(lambda: {}, on_intrusion=seen.append)
    readings = [[], ["a"], [], ["b"], ["b"], ["b"], []]
    hits = [watcher.observe({"utc": str(i), "intrusions": r})
            for i, r in enumerate(readings)]
    assert hits == [False, False, False, False, True, True, False]
    assert len(seen) == 1 and seen[0]["what"] == ["b"] and seen[0]["first_seen"] == "3"


def test_the_selftest_passes():
    assert cert8.selftest() == 0


def variance_ledger(path, gpu, meep):
    rows = []
    for index, (g, m) in enumerate(zip(gpu, meep), 1):
        rows.append({"ledger": "gpu", "tier": "priority", "route": "default",
                     "pass": index, "measured": {"verdict": "TIMED", "legs": {
                         "fused": {"mcell_steps_per_s": g, "spread": 0.01}}}})
        rows.append({"ledger": "meep", "tier": "priority", "ranks": 32, "pass": index,
                     "measured": {"mcell_steps_per_s": m, "spread": 0.02}})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def test_the_variance_precondition_gates_the_gpu_and_records_meep(tmp_path):
    """MEEP's repeat spread of 12 % is recorded and refuses nothing; the GPU route's
    decides (owner ruling)."""
    ledger = variance_ledger(tmp_path / "go.jsonl", [747.0, 746.0, 742.6, 738.5, 735.0],
                             [500.0, 560.0, 520.0, 540.0, 510.0])
    verdict = cert8.variance_gate(ledger, 32, "default")
    assert verdict["ok"] is True and verdict["gated_sides"] == ["gpu"]
    assert verdict["sides"]["meep@32"]["spread"] == pytest.approx(60.0 / 520.0)
    assert TR.MEEP_SPREAD_RULE == verdict["meep_rule"]
    wide = variance_ledger(tmp_path / "no.jsonl", [747.0, 690.0, 742.6, 738.5, 735.0],
                           [500.0] * 5)
    assert cert8.variance_gate(wide, 32, "default")["ok"] is False


def test_the_quiet_window_needs_two_clear_readings():
    readings = iter([{"intrusions": ["x"]}, {"intrusions": []}, {"intrusions": ["y"]},
                     {"intrusions": []}, {"intrusions": []}])
    slept = []
    window = cert8.quiet_window(lambda: next(readings), hold=5.0, log=lambda m: None,
                                sleep=slept.append)
    assert len(window) == 2 and slept == [60.0, 5.0, 60.0, 5.0]


# ---------------------------------------------------------------------------
# A session on stand-ins
# ---------------------------------------------------------------------------

def bench_line(case, table, arch):
    row = fixtures.bench_row(case=case, table=table)
    for plan in row["plans"].values():
        plan["environment"]["device"]["compute_capability"] = arch
        plan["launch_counters"] = {"step_D": {"arm": "fused pair D", "backend": table,
                                              "dispatches": 6}}
        plan["families"] = {"fused pair D": {"family": "fe", "gate": "g"}}
    row[TR.IMPORTED_KEY] = {"meep_gpu/driver.py": TR.artifact_drift._sha256(
        REPO_ROOT / "meep_gpu" / "driver.py")}
    row[TR.DATA_KEY] = {}
    return json.dumps(row) + "\n"


def test_a_session_times_one_row_records_it_and_reruns_an_intruded_unit(tmp_path):
    catalog = cert8.load_catalog(NVIDIA)
    key = TR.fused_key("8.6", "triton", "pml_2d", "res20", 1)
    args = args_for(tmp_path, no_variance="fused rows only")
    environment = tmp_path / "build_record.json"
    environment.write_text(json.dumps({"platform": "linux-64"}))
    args.environment_record = str(environment)
    units = cert8.build_units(args, catalog, [key])
    calls = []

    def runner(unit, stopped):
        calls.append(unit.name)
        os.makedirs(unit.out, exist_ok=True)
        with open(os.path.join(unit.out, "rows.jsonl"), "a") as handle:
            handle.write(bench_line("pml_2d", "triton", "8.6"))
        return 0

    script = iter([{"intrusions": []}, {"intrusions": []},          # the quiet window
                   {"intrusions": ["busy"]}, {"intrusions": ["busy"]},  # attempt 1
                   {"intrusions": []}, {"intrusions": []}]           # quiet again
                  + [{"intrusions": []}] * 50)
    minted = []
    session = cert8.Session(args, catalog, reader=lambda: dict(next(script), utc="t"),
                            runner=lambda unit, stopped: (runner(unit, stopped),
                                                          __import__("time").sleep(0.3))[0],
                            minter=lambda a: minted.append(1) or str(tmp_path / "keep"))
    assert units[0].env["CUPY_CACHE_DIR"] == cert8.KEEP_CACHE_PENDING
    assert session.run(units) == 0
    assert minted == [1] and units[0].env["CUPY_CACHE_DIR"] == str(tmp_path / "keep")
    assert units[0].env["MEEP_GPU_SUBNORMAL_POLICY"] == "keep"
    assert calls == [units[0].name, units[0].name], "the intruded unit runs again"
    aside = [p for p in os.listdir(os.path.dirname(units[0].out)) if "intruded" in p]
    assert len(aside) == 1, "the intruded output is set aside, never deleted"
    records = TR.load_store(args.store)
    assert [r["key"] for r in records] == [key]
    record = records[0]
    assert record["gates"]["watcher"]["intrusions"] == []
    assert record["gates"]["watcher"]["first"] and record["gates"]["watcher"]["last"], \
        "R18: the readings at the start and the end of the unit are kept whole"
    assert record["gates"]["power"]["gated"] is False
    assert len(record["gates"]["quiet"]) == 2, "the window this unit opened on"
    assert record["unit"]["rc"] == 0
    assert record["unit"]["env"]["MEEP_GPU_SUBNORMAL_POLICY"] == "keep"
    assert record["unit"]["env"]["MEEP_GPU_DISPATCH"] is None, "unset, by name"
    assert TR.refusals(record) == []
    assert record["environment"]["reference"]["name"] == "build_record.json"
    assert record["processes"][0][TR.IMPORTED_KEY]
    progress = open(os.path.join(args.run_root, "s1", "progress.log")).read()
    assert "INTRUSION" in progress and "1 record(s) written" in progress
    # resume: the marker skips the finished unit
    again = cert8.Session(args, catalog, reader=lambda: {"intrusions": [], "utc": "t"},
                          runner=lambda unit, stopped: calls.append("again") or 0,
                          minter=lambda a: str(tmp_path / "keep"))
    assert again.run(units) == 0 and "again" not in calls


def test_a_unit_that_failed_is_not_marked_done(tmp_path):
    catalog = cert8.load_catalog(NVIDIA)
    key = TR.fused_key("8.6", "triton", "pml_2d", "res20", 1)
    args = args_for(tmp_path, no_variance="fused rows only", environment_record=None,
                    allow_unrecorded_environment=True)
    units = cert8.build_units(args, catalog, [key])
    session = cert8.Session(args, catalog, reader=lambda: {"intrusions": [], "utc": "t"},
                            runner=lambda unit, stopped: 3,
                            minter=lambda a: str(tmp_path / "keep"))
    assert session.run(units) == cert8.EXIT_UNIT
    assert not os.path.exists(session.marker(units[0]))


def test_owed_keys_retime_the_stale_and_check_the_exempt(tmp_path):
    catalog = cert8.load_catalog(APPLE)
    keys = cert8.catalog_keys(catalog)
    store = str(tmp_path / "store")
    environment = {"reference": {"name": "r.json", "sha256": "0" * 64}}
    for key in keys[:2]:
        clear = {"utc": "a", "intrusions": []}
        record = TR.base_record("ladder", key, session={"id": "s1", "started_utc": "a",
                                                        "ended_utc": "b"},
                                role="release", architecture="applegpu_g13s",
                                unit={"rc": 0}, environment=environment,
                                gates={"quiet": [clear, clear],
                                       "watcher": {"readings": 3, "intrusions": []}})
        record["processes"] = [{"role": "gpu:metal:p1", "ran_package": True,
                                TR.DATA_KEY: {},
                                TR.IMPORTED_KEY: {"meep_gpu/driver.py": (
                                    "0" * 64 if key == keys[0] else
                                    TR.artifact_drift._sha256(
                                        REPO_ROOT / "meep_gpu" / "driver.py"))}}]
        record["legs"] = {"gpu": {"metal": {"median": 600.0}}, "meep": {}}
        TR.write_record(store, record)
    retime, check = cert8.owed_keys(catalog, store, str(REPO_ROOT))
    assert keys[0] in retime and keys[1] not in retime
    assert set(retime) == set(keys) - {keys[1]} and check == []
    # R27: a session in a rebuilt reference environment owes every row of the old one,
    # the valid one included; in the environment the rows ran in, it owes the same
    rebuilt = tmp_path / "rebuilt_build_record.json"
    rebuilt.write_text("{}")
    retime, _ = cert8.owed_keys(catalog, store, str(REPO_ROOT),
                                environment_record=str(rebuilt))
    assert set(retime) == set(keys)
    args = args_for(tmp_path, rows=None, owed=True, release_root=str(REPO_ROOT),
                    allow_unrecorded_environment=False, store=store,
                    environment_record=str(rebuilt))
    assert set(cert8._keys_asked(args, catalog)) == set(keys)
    # plan and run read the same flag, so the plan of --owed is the run's
    for command in ("plan", "run"):
        parsed = cert8.build_parser().parse_args(
            [command, "--catalog", APPLE, "--store", store, "--run-root", "r", "--owed",
             "--environment-record", str(rebuilt)])
        assert parsed.environment_record == str(rebuilt)


# ---------------------------------------------------------------------------
# R5 and R18 as the session holds them
# ---------------------------------------------------------------------------

def test_a_unit_is_never_resumed_across_processes(tmp_path):
    """R5: a row is timed in one process under one watcher. A ladder output an earlier
    process left (it stopped before its plan finished) is set aside, never resumed, and
    the unit runs whole; the same holds for the variance precondition."""
    catalog = cert8.load_catalog(APPLE)
    keys = cert8.catalog_keys(catalog)
    args = args_for(tmp_path, no_variance="the ladder alone")
    units = cert8.build_units(args, catalog, keys[:1])
    os.makedirs(units[0].out)
    with open(os.path.join(units[0].out, "ladder_rows.jsonl"), "w") as handle:
        handle.write(json.dumps({"ledger": "meep", "tier": "priority", "pass": 1}) + "\n")
    seen = []
    session = cert8.Session(args, catalog, reader=lambda: {"intrusions": [], "utc": "t"},
                            runner=lambda unit, stopped: seen.append(list(unit.argv)) or 0,
                            minter=lambda a: str(tmp_path / "keep"))
    session.run(units)
    assert seen and all("--resume" not in argv for argv in seen)
    parent = os.path.dirname(units[0].out)
    aside = [name for name in os.listdir(parent)
             if name.startswith(os.path.basename(units[0].out) + ".set_aside_")]
    assert len(aside) == 1, "the earlier output is set aside, never deleted"
    assert os.path.exists(os.path.join(parent, aside[0], "ladder_rows.jsonl"))


def test_a_quiet_window_opens_every_unit(tmp_path):
    """The lane procedure gated every unit, not only the session: each unit reads its
    own window (two readings) before it starts, and records it."""
    catalog = cert8.load_catalog(NVIDIA)
    args = args_for(tmp_path, no_variance="fused rows only", environment_record=None,
                    allow_unrecorded_environment=True)
    keys = [TR.fused_key("8.6", "triton", "pml_2d", f"res{r}", 1) for r in (20, 60)]
    units = cert8.build_units(args, catalog, keys)
    windows = []

    def gate():
        windows.append(1)
        return {"intrusions": [], "utc": "t"}

    session = cert8.Session(args, catalog, reader=lambda: {"intrusions": [], "utc": "t"},
                            gate_reader=gate, runner=lambda unit, stopped: 3,
                            minter=lambda a: "keep")
    session.run(units)
    assert len(windows) == 2 * len(units)


def test_other_work_of_this_account_intrudes():
    """The lane procedure's no-other-work-of-ours rule (R18): a harness, test or MPI
    process of this account outside the session, or a busy Python process, intrudes;
    the session's own processes, a waiter's idle shell and other accounts do not."""
    me = 501
    table = {
        10: (1, me, 0.0, "python -u cert8.py run"),               # the driver
        11: (10, me, 95.0, "python -u parity/meep_gpu/bench_fused_products.py"),
        12: (1, me, 300.0, "mpirun -np 8 python bench_meep_identical_case.py"),
        13: (1, me, 50.0, "python -m pytest -q parity/meep_gpu"),
        14: (1, me, 0.1, "/bin/zsh slot.sh mac_cpu log -- zsh leg.sh"),
        15: (1, me, 40.0, "python somebody_elses_notebook.py"),
        16: (1, 777, 400.0, "python -m pytest"),                  # another account
        17: (1, me, 0.0, "python /runs/s1/ladder_x/tree/bench_meep_identical_case.py"),
        18: (1, me, 2.0, "/usr/bin/zsh -l"),
    }
    mine = cert8.own_pids(table, 10, markers=("/runs/s1/",))
    assert mine == {10, 11, 17}
    found = cert8.other_work(table, mine, me)
    assert [line.split(": ")[1].split()[0] for line in found] == ["12", "13", "15"]
    assert "MPI" in found[0] and "busy Python" in found[2]


def test_a_busy_build_or_tool_of_this_account_intrudes_and_shares_are_measured():
    """R18 as the 1.0 lanes ran it: a build, compiler or bulk tool of the account that
    burns CPU intrudes on either platform (an environment build started beside a
    ladder); on Linux any busy process of the account does, as the GPU host's watcher
    had it, while macOS reads only the kinds named (the account runs the desktop there).
    The share is the CPU time used BETWEEN readings: ``ps`` %cpu is a lifetime average
    on Linux, so a long-idle process that turns busy reads low on it."""
    me = 501
    table = {
        10: (1, me, 0.0, "python -u cert8.py run"),
        21: (1, me, 95.0, "/usr/bin/make -j 32"),
        22: (1, me, 60.0, "/opt/cuda/bin/ptxas -arch=sm_86 k.ptx"),
        23: (1, me, 5.0, "/usr/bin/make -j 32"),                 # idle: not busy
        24: (1, me, 80.0, "/usr/lib/app/renderer --type=gpu"),   # a busy non-tool
        25: (1, 777, 400.0, "/usr/bin/make -j 96"),              # another account
    }
    mine = cert8.own_pids(table, 10)
    linux = cert8.other_work(table, mine, me, platform="linux")
    assert [line.split(": ")[1].split()[0] for line in linux] == ["21", "22", "24"]
    assert "build or tool" in linux[0] and "a busy process (80 %" in linux[2]
    mac = cert8.other_work(table, mine, me, platform="darwin")
    assert [line.split(": ")[1].split()[0] for line in mac] == ["21", "22"]
    # the shares between two readings, not the lifetime average
    assert cert8.cpu_seconds("1-02:03:04") == 93784.0
    assert cert8.cpu_seconds("01:02:03") == 3723.0
    assert cert8.cpu_seconds("12:34.50") == 754.5
    assert cert8.cpu_seconds("garbage") is None
    shares = cert8.CpuShares()
    assert shares.update({7: ("make -j 32", 100.0), 8: ("sleep 9", 1.0)}, 1000.0) == {}
    later = shares.update({7: ("make -j 32", 160.0), 8: ("sleep 9", 1.0),
                           9: ("new", 5.0)}, 1120.0)
    assert later == {7: 50.0, 8: 0.0}
    # a reused pid with another command is a new process, not a share
    assert shares.update({7: ("other", 200.0)}, 1240.0) == {}


def test_a_box_reading_measures_each_process_between_readings(monkeypatch):
    """The watcher and the quiet window read the box through one share meter: a build
    of this account whose ``ps`` %cpu reads 0 (a lifetime average on Linux) but which
    used 30 s of CPU since the last reading intrudes on the second reading."""
    me = os.getuid()
    used = {"cpu": 0.0}

    def ps(argv, **_kwargs):
        assert argv[0] == "ps", argv
        used["cpu"] += 30.0
        minutes, seconds = divmod(used["cpu"], 60.0)
        return types.SimpleNamespace(
            stdout=f"  99901     1 {me}  0.0 {int(minutes)}:{seconds:05.2f} "
                   "/usr/bin/make -j 32\n")

    host = types.SimpleNamespace(
        memory=lambda: {}, loadavg=lambda: None, load1=lambda: 0.0,
        metal_neighbours=lambda: [], darwin_daemons=lambda: [],
        cpu_state_darwin=lambda: None, apple_gpu_state=lambda: None,
        compute_apps=lambda _gpu: [], gpu_uuids=lambda: {}, run_text=lambda _argv: "",
        as_uuid=lambda gpu, _uuids: gpu, uid_of_pid=lambda _pid: me)
    monkeypatch.setitem(sys.modules, "timing_host", host)
    monkeypatch.setattr(cert8.subprocess, "run", ps)
    monkeypatch.setattr(cert8, "SHARES", cert8.CpuShares())
    first = cert8.host_reading(None, os.getpid())
    assert not [line for line in first["intrusions"] if "99901" in line]
    time.sleep(0.05)
    second = cert8.host_reading(None, os.getpid())
    assert [line for line in second["intrusions"] if "99901" in line
            and "build or tool" in line]


def test_the_window_holds_the_ladders_load_and_memory_thresholds():
    plenty = {"available_gib": 600.0, "total_gib": 1007.5}
    assert cert8.gate_reasons("linux", 3.0, plenty, 96) == []
    assert cert8.gate_reasons("linux", 10.0, plenty, 96)[0].startswith("load1 10.0 >= 10")
    tight = {"available_gib": 200.0, "total_gib": 1007.5}
    assert "available memory 200.0 GiB" in cert8.gate_reasons("linux", 1.0, tight, 96)[0]
    assert cert8.gate_reasons("linux", None, plenty, 96) == ["load average unreadable"]
    # macOS: the ladder calibrates its own CPU threshold per row; memory and the
    # background daemons are read here, and load is recorded only
    mac = {"available_gib": 20.0, "total_gib": 32.0}
    assert cert8.gate_reasons("darwin", 9.0, mac, 10, []) == []
    assert cert8.gate_reasons("darwin", 1.0, mac, 10, ["mds_stores"])[0].startswith(
        "background daemons busy")
