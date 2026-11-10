"""Host tests for the release timing driver (``cert8.py``). No device, no timing.

The driver's units are commands of the release's own instruments; these tests read
them, check the release ladder accepts every one (its ``--dry-run`` starts no process),
run a session on stand-ins (a runner that writes what the instruments write, a reader
that plays the host), and check the gates the release procedure fixes
(``docs/development/timing.md``, "Running rows"): no stage gated on power, MEEP's spread
recorded and never gated, an intrusion that stops and re-runs a unit.
"""

from __future__ import annotations

import functools
import hashlib
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
import neutral_ids  # noqa: E402
import test_cut_timing_record as fixtures  # noqa: E402
import timing_rows as TR  # noqa: E402

NVIDIA = str(HERE / "cert8_catalog_nvidia.json")
APPLE = str(HERE / "cert8_catalog_apple.json")
DRIVE = {"triton": ["pml_2d", "pml_3d"], "cuda": ["pml_2d"],
         "metal": ["pml_2d", "magnetic_seam_2d"]}


def ladder_only(catalog):
    """A catalog's ladder rows alone. The shipped Apple catalog's fused table takes the
    release's DRIVE cases, which only the bench's estimate reads, so a test of ladder
    rows reads the catalog without it."""
    return dict(catalog, fused={"tables": {}})


#: The rank blocks the GPU host's 48 cores give (16, 24, 32, 48 | 8, 1 | 64), as a site file
#: declares them: a split session's MEEP halves take their ceiling (64) from it.
SITE_RANKS = "16,24,32,48|8,1|64"


def site_params(tmp_path, ranks=SITE_RANKS):
    """A site's ``--ladder-params`` file declaring its rank blocks (``ranks`` None: none)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "site.params"
    path.write_text("" if ranks is None else f"RANKS={ranks}\n")
    return str(path)


def args_for(tmp_path, **extra):
    """A session's arguments. A split session (``split_ladder``) is given a site file that
    declares the GPU host's rank blocks unless ``ladder_params`` is named: its MEEP halves
    take the host's MEEP rank ceiling from it (``cert8.split_rank_ceiling``)."""
    values = dict(python=sys.executable, run_root=str(tmp_path / "runs"), session="s1",
                  tree=str(REPO_ROOT), gpu="0", bench_env=[], ladder_params=None,
                  store=str(tmp_path / "store"), role="release", no_variance=None,
                  environment_record=None, hold_s=0.0, watch_every_s=0.05)
    if extra.get("split_ladder") and "ladder_params" not in extra:
        values["ladder_params"] = site_params(tmp_path / "site")
    values.update(extra)
    return types.SimpleNamespace(**values)


# ---------------------------------------------------------------------------
# The catalogs and their keys
# ---------------------------------------------------------------------------

#: The identical-case cases of the extended ladder: pml_3d (continuity with the 1.0
#: round) and the three cases it adds (dispersive, monitor-heavy, source on a fused seam).
EXTENDED_CASES = ("pml_3d", "pml_3d_lorentz", "pml_3d_monitors", "pml_3d_hz")


def test_the_shipped_catalogs_name_the_rows_of_the_extended_ladder():
    nvidia = cert8.load_catalog(NVIDIA)
    keys = cert8.catalog_keys(nvidia, drive=lambda table: DRIVE[table])
    ladder = [k for k in keys if k.startswith("ladder|")]
    assert ladder == [f"ladder|8.6|{case}|{c}|priority" for case in EXTENDED_CASES
                      for c in (512000, 2097152, 7077888)]
    fused = [k for k in keys if k.startswith("fused|")]
    # DRIVE cases, then the sweep: triton 3 cases x 5 sizes x 2 passes, cuda 2 x 5 x 2
    assert len(fused) == 3 + 30 + 20
    assert "fused|8.6|triton|offdiag_2d|res300|p2" in fused
    apple = cert8.load_catalog(APPLE)
    keys = cert8.catalog_keys(apple, drive=lambda table: DRIVE[table])
    ladder = [k for k in keys if k.startswith("ladder|")]
    assert len(ladder) == 4 * 5 and all(k.startswith("ladder|applegpu_g13s|") for k in ladder)
    assert {TR.parse_key(k)["case"] for k in ladder} == set(EXTENDED_CASES)
    # DRIVE cases, then the sweep: metal 3 cases x 5 sizes x 2 passes
    fused = [k for k in keys if k.startswith("fused|")]
    assert keys == ladder + fused and len(fused) == 2 + 30
    assert "fused|applegpu_g13s|metal|offdiag_2d|res300|p2" in fused
    for catalog in (nvidia, apple):
        with pytest.raises(cert8.Refused):
            cert8.catalog_keys(catalog)     # "release" DRIVE cases need their reader


def test_no_unit_is_gated_on_power(tmp_path):
    """No stage is gated on AC power (``docs/development/timing.md``, "Running rows").
    Every ladder unit, the variance precondition and every ladder dispatch check pass
    ``--require-ac-power 0``."""
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
    catalog = ladder_only(cert8.load_catalog(APPLE))
    keys = [k for k in cert8.catalog_keys(catalog) if TR.parse_key(k)["case"] == "pml_3d"]
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


def test_the_release_ladder_accepts_every_unit_and_plans_exactly_its_rows(tmp_path,
                                                                           monkeypatch):
    import timing_host  # noqa: PLC0415
    # The variance rank count is the host's (2P/3): read here as the GPU host's 48 cores.
    monkeypatch.setattr(timing_host, "topology", lambda *a, **k: {"physical_cores": 48})
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


def test_a_ladder_collected_by_hand_reads_no_drive_list(tmp_path, monkeypatch):
    """``collect ladder`` without ``--key`` records the catalog's ladder rows, on either
    catalog: a ladder's keys need no DRIVE list, so the release's DRIVE cases a fused
    table takes are never asked for (no bench estimate runs)."""
    collected = []
    monkeypatch.setattr(cert8, "collect",
                        lambda unit, **_kwargs: collected.append(list(unit.keys)) or ["row"])
    for catalog in (NVIDIA, APPLE):
        code = cert8.main(["collect", "ladder", "--catalog", catalog,
                           "--store", str(tmp_path / "store"), "--session", "s1",
                           "--dir", str(tmp_path / "ladder")])
        assert code == 0, catalog
        ladder = cert8.catalog_keys(ladder_only(cert8.load_catalog(catalog)))
        assert ladder and all(key.startswith("ladder|") for key in ladder), catalog
        assert collected[-1] == ladder, catalog
    assert len(collected) == 2


def metal_drive_rows():
    """The Metal route gate's DRIVE rows, read off its source (the gate imports the device
    stack). The bench times the ones that expect dispatch: the release's DRIVE cases."""
    import ast  # noqa: PLC0415

    tree = ast.parse((HARNESS / "gate_dispatch_metal_route.py").read_text(encoding="utf-8"))
    node, = [node for node in tree.body if isinstance(node, ast.AnnAssign)
             and getattr(node.target, "id", None) == "DRIVE"]
    return ast.literal_eval(node.value)


def test_the_apple_catalog_times_the_metal_fused_products(tmp_path):
    """The Apple catalog names the rows a Metal dispatch-preference record is cut from, as
    the NVIDIA catalog names its tables': every DRIVE case the release dispatches on the
    Metal table, and a size sweep of two passes at five sizes, with the NVIDIA catalog's
    bench and sweep arguments. Every sweep case is a Metal DRIVE case that expects dispatch,
    so each sweep row times a Metal fused product."""
    catalog = cert8.load_catalog(APPLE)
    nvidia = cert8.load_catalog(NVIDIA)["fused"]
    fused = catalog["fused"]
    assert set(fused["tables"]) == {"metal"} and fused["tables"]["metal"]["drive"] == "release"
    assert (fused["bench_args"], fused["sweep_args"]) == (nvidia["bench_args"],
                                                          nvidia["sweep_args"])
    sweep = fused["tables"]["metal"]["sweep"]
    assert sweep == {"cases": ["pml_2d", "magnetic_seam_2d", "offdiag_2d"],
                     "res": [20, 60, 130, 200, 300], "passes": 2}
    rows = metal_drive_rows()
    release = [case for case, spec in rows.items() if spec.get("expect") == "dispatch"]
    for case in sweep["cases"]:
        assert rows[case]["expect"] == "dispatch" and rows[case]["reached_by"] == "release"
    keys = cert8.catalog_keys(catalog, drive=lambda table: {"metal": release}[table])
    units = cert8.build_units(args_for(tmp_path), catalog, keys)
    fused_units = [unit for unit in units if unit.kind == "fused"]
    drive = [unit for unit in fused_units if TR.parse_key(unit.keys[0])["size"] == "drive"]
    assert len(release) == len(drive) == 42 and len(fused_units) == 42 + 30
    assert len(units) == 8 + 72        # the ladder's 8 units (4 cases x 2 tiers) and these
    for unit in fused_units:
        argv = unit.argv
        assert argv[argv.index("--drive-table") + 1] == "metal", unit.name
        if TR.parse_key(unit.keys[0])["size"] != "drive":
            assert argv[argv.index("--repeats") + 1] == "4", unit.name
            assert argv[argv.index("--target-seconds") + 1] == "1.5", unit.name


def test_a_metal_fused_unit_runs_on_certified_hardware(tmp_path):
    """A Metal fused unit and its dispatch check run with ``MEEP_GPU_HARDWARE=certified``,
    as an NVIDIA one does: an identity the bound ledger does not certify is refused by
    name instead of timed. The ladder's own Metal rows are not given the switch."""
    catalog = cert8.load_catalog(APPLE)
    key = TR.fused_key("applegpu_g13s", "metal", "pml_2d", "res60", 1)
    unit, check = cert8.build_units(args_for(tmp_path), catalog, [key, TR.check_key(key)])
    for item in (unit, check):
        assert item.env["MEEP_GPU_HARDWARE"] == "certified", item.name
        assert item.env["KMP_DUPLICATE_LIB_OK"] == "TRUE", item.name
        assert item.env["PYTORCH_ENABLE_MPS_FALLBACK"] is None, item.name
        assert item.env["MEEP_GPU_BACKEND_PREFERENCE"] is None, item.name
        assert "CUDA_VISIBLE_DEVICES" not in item.env, item.name
    ladder = cert8.build_units(args_for(tmp_path), ladder_only(catalog),
                               ["ladder|applegpu_g13s|pml_3d|512000|priority"])
    assert all("MEEP_GPU_HARDWARE" not in u.env for u in ladder)


def _fused_only_session(tmp_path, monkeypatch, spreads, **extra):
    """A fused-only session on stand-ins: its runner notes each unit it runs, and writes
    the variance ledger (GPU rates ``spreads``) when the precondition runs."""
    import timing_host  # noqa: PLC0415
    # The variance rank count is the host's (2P/3): read here as the GPU host's 48 cores.
    monkeypatch.setattr(timing_host, "topology", lambda *a, **k: {"physical_cores": 48})
    catalog = cert8.load_catalog(NVIDIA)
    args = args_for(tmp_path, allow_unrecorded_environment=True, **extra)
    units = cert8.build_units(args, catalog, [TR.fused_key("8.6", "cuda", "pml_2d",
                                                           "res60", 1)])
    ran = []

    def runner(unit, stopped):
        ran.append(unit.kind)
        os.makedirs(unit.out, exist_ok=True)
        if unit.kind == "variance":
            variance_ledger(Path(unit.out) / "ladder_rows.jsonl", spreads, [500.0] * 5)
        return 0

    session = cert8.Session(args, catalog, reader=lambda: {"intrusions": [], "utc": "t"},
                            runner=runner, minter=lambda a: str(tmp_path / "keep"))
    return args, units, session, ran


def test_a_fused_only_session_runs_the_variance_precondition_when_asked(tmp_path,
                                                                        monkeypatch):
    """A fused-only session runs no precondition unless asked; with ``--variance`` it runs
    the catalog's before its first unit, writes its verdict, refuses on NO-GO with the
    variance exit status before any unit runs, and runs its units after a GO. Given with
    ``--no-variance`` it is refused by name."""
    go = [747.0, 746.0, 742.6, 738.5, 735.0]
    no_go = [747.0, 690.0, 742.6, 738.5, 735.0]
    _args, units, session, ran = _fused_only_session(tmp_path / "plain", monkeypatch, go)
    session.run(units)
    assert ran == ["fused"]
    args, units, session, ran = _fused_only_session(tmp_path / "go", monkeypatch, go,
                                                    variance=True)
    session.run(units)
    assert ran == ["variance", "fused"]
    verdict = json.load(open(os.path.join(args.run_root, "s1", "variance.json")))
    assert verdict["verdict"] == "GO"
    args, units, session, ran = _fused_only_session(tmp_path / "nogo", monkeypatch, no_go,
                                                    variance=True)
    with pytest.raises(cert8.Refused) as refusal:
        session.run(units)
    assert refusal.value.code == cert8.EXIT_VARIANCE and ran == ["variance"]
    _args, units, session, ran = _fused_only_session(tmp_path / "both", monkeypatch, go,
                                                     variance=True, no_variance="why")
    with pytest.raises(cert8.Refused) as refusal:
        session.run(units)
    assert "--no-variance" in refusal.value.message and ran == []
    parsed = cert8.build_parser().parse_args(["run", "--catalog", NVIDIA, "--store", "s",
                                              "--run-root", "r", "--variance"])
    assert parsed.variance is True


def test_a_session_on_macos_holds_the_mac_awake(tmp_path):
    """On macOS a session holds the Mac awake for its life (``caffeinate -i -s -w`` on its
    own pid, the ladder's method), says so in its log without the pid, and stops it at its
    end; on a Mac without ``caffeinate`` it says so and runs on; off macOS it starts
    nothing. Nothing about power gates it."""
    class Child:
        terminated = False

        def terminate(self):
            self.terminated = True

    child, started = Child(), []

    def start(argv, **_kwargs):
        started.append(list(argv))
        return child

    present = lambda name: f"/usr/bin/{name}"  # noqa: E731
    process, line = cert8.keep_awake(4242, platform="darwin", which=present, start=start)
    assert process is child and started == [["caffeinate", "-i", "-s", "-w", "4242"]]
    assert line == "keep-awake: caffeinate -i -s -w <this session>"
    assert cert8.keep_awake(4242, platform="darwin", which=lambda name: None,
                            start=start) == (None, "keep-awake: caffeinate is absent, so "
                                                   "the Mac is not held awake; the session "
                                                   "runs on")
    assert cert8.keep_awake(4242, platform="linux", which=present, start=start) == (None, "")
    assert len(started) == 1
    args = args_for(tmp_path)
    asked = []
    session = cert8.Session(args, cert8.load_catalog(NVIDIA),
                            reader=lambda: {"intrusions": [], "utc": "t"},
                            runner=lambda unit, stopped: 0, minter=lambda a: "k",
                            awake=lambda pid: (asked.append(pid) or child, line))
    assert session.run([]) == 0
    assert asked == [os.getpid()] and child.terminated
    log = Path(args.run_root, "s1", "progress.log").read_text()
    assert line in log


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
    """A variance ladder's ledger; ``meep`` None: the GPU route alone (a split session's)."""
    rows = []
    for index, g in enumerate(gpu, 1):
        rows.append({"ledger": "gpu", "tier": "priority", "route": "default",
                     "pass": index, "measured": {"verdict": "TIMED", "legs": {
                         "fused": {"mcell_steps_per_s": g, "spread": 0.01}}}})
        if meep is not None:
            rows.append({"ledger": "meep", "tier": "priority", "ranks": 32, "pass": index,
                         "measured": {"mcell_steps_per_s": meep[index - 1], "spread": 0.02}})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def test_the_variance_precondition_gates_the_gpu_and_records_meep(tmp_path):
    """MEEP's repeat spread of 12 % is recorded and refuses nothing; the GPU route's
    decides."""
    ledger = variance_ledger(tmp_path / "go.jsonl", [747.0, 746.0, 742.6, 738.5, 735.0],
                             [500.0, 560.0, 520.0, 540.0, 510.0])
    verdict = cert8.variance_gate(ledger, 32, "default")
    assert verdict["ok"] is True and verdict["gated_sides"] == ["gpu"]
    assert verdict["sides"]["meep@32"]["spread"] == pytest.approx(60.0 / 520.0)
    assert TR.MEEP_SPREAD_RULE == verdict["meep_rule"]
    wide = variance_ledger(tmp_path / "no.jsonl", [747.0, 690.0, 742.6, 738.5, 735.0],
                           [500.0] * 5)
    assert cert8.variance_gate(wide, 32, "default")["ok"] is False
    # The GPU route alone (a split session's precondition): GO on the same rule, and no
    # MEEP side recorded at all.
    alone = variance_ledger(tmp_path / "gpu.jsonl", [747.0, 746.0, 742.6, 738.5, 735.0],
                            None)
    verdict = cert8.variance_gate(alone, None, "default")
    assert verdict["ok"] is True and set(verdict["sides"]) == {"gpu"}
    assert verdict["meep_ranks"] is None and verdict["gated_sides"] == ["gpu"]


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
        "the watcher's readings at the start and the end of the unit are kept whole"
    assert record["gates"]["power"]["gated"] is False
    assert len(record["gates"]["quiet"]) == 2, "the window this unit opened on"
    assert record["unit"]["rc"] == 0
    assert record["unit"]["env"]["MEEP_GPU_SUBNORMAL_POLICY"] == "keep"
    assert record["unit"]["env"]["MEEP_GPU_DISPATCH"] is None, "unset, by name"
    assert TR.refusals(record) == []
    # the record leaves the machine: the unit's command and environment are written as
    # neutral_ids spells them (the interpreter's path under its root's label)
    assert neutral_ids.residue(record["unit"]) == [] and neutral_ids.residue(
        record["gates"]) == []
    assert record["unit"]["argv"] == neutral_ids.argv(units[0].argv)
    assert record["environment"]["reference"]["name"] == "build_record.json"
    assert record["processes"][0][TR.IMPORTED_KEY]
    progress = open(os.path.join(args.run_root, "s1", "progress.log")).read()
    assert "INTRUSION" in progress and "1 record(s) written" in progress
    assert neutral_ids.residue(progress) == [], "the session's log leaves the machine"
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


def _bench_unit_session(tmp_path, verdict, rc):
    """One fused unit run by a stand-in bench that writes one row of ``verdict`` and
    exits ``rc``; returns the session's exit status, whether the unit was marked, and
    how many records the store holds."""
    catalog = cert8.load_catalog(NVIDIA)
    key = TR.fused_key("8.6", "triton", "pml_2d", "res20", 1)
    args = args_for(tmp_path, no_variance="fused rows only", environment_record=None,
                    allow_unrecorded_environment=True)
    units = cert8.build_units(args, catalog, [key])

    def runner(unit, stopped):
        row = json.loads(bench_line("pml_2d", "triton", "8.6"))
        row["verdict"] = verdict
        if verdict != "TIMED":
            row["reportable"] = False
            row["floors"] = dict(row["floors"], windows_steady=False)
        os.makedirs(unit.out, exist_ok=True)
        with open(os.path.join(unit.out, "rows.jsonl"), "a") as handle:
            handle.write(json.dumps(row) + "\n")
        return rc

    session = cert8.Session(args, catalog, reader=lambda: {"intrusions": [], "utc": "t"},
                            runner=runner, minter=lambda a: str(tmp_path / "keep"))
    status = session.run(units)
    return status, os.path.exists(session.marker(units[0])), len(TR.load_store(args.store))


def test_a_bench_unit_whose_row_is_struck_by_its_floors_is_done(tmp_path):
    """The bench exits 0 on a row struck by its floors (``bench_fused_products.
    exit_status``): the measurement is finished and its record kept, so the unit is
    marked and a resume of the session does not time it again."""
    status, marked, records = _bench_unit_session(tmp_path, "TIMED-NOT-REPORTABLE", 0)
    assert (status, marked, records) == (0, True, 1)


@pytest.mark.parametrize("verdict, rc", [("ERROR", 1), ("SKIP-TOO-LARGE", 1),
                                         ("ERROR", 0), ("TIMED-NOT-REPORTABLE", 1)])
def test_a_bench_unit_is_marked_only_on_exit_0_with_a_timed_row(tmp_path, verdict, rc):
    """An errored case runs again, and so does a unit whose exit status and row
    disagree; what it wrote stays in the store either way."""
    status, marked, records = _bench_unit_session(tmp_path, verdict, rc)
    assert (status, marked, records) == (cert8.EXIT_UNIT, False, 1)
    progress = open(os.path.join(str(tmp_path / "runs"), "s1", "progress.log")).read()
    assert "not marked done" in progress


def test_the_driver_and_the_bench_name_the_same_timed_verdicts():
    """The bench's exit status and the driver's done rule read one list of verdicts.
    Read off the bench's source, which imports the device stack."""
    import ast  # noqa: PLC0415

    tree = ast.parse((HARNESS / "bench_fused_products.py").read_text(encoding="utf-8"))
    found = [ast.literal_eval(node.value) for node in tree.body
             if isinstance(node, ast.Assign)
             and [getattr(t, "id", None) for t in node.targets] == ["TIMED_VERDICTS"]]
    assert found == [cert8.TIMED_VERDICTS]


def test_owed_keys_retime_the_stale_and_check_the_exempt(tmp_path):
    catalog = ladder_only(cert8.load_catalog(APPLE))
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
    # A session in a rebuilt reference environment owes every row of the old one,
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
# The same-session rule and the idle-host rules, as the session holds them
# ---------------------------------------------------------------------------

def test_a_unit_is_never_resumed_across_processes(tmp_path):
    """The same-session rule: a row is timed in one process under one watcher. A ladder
    output an earlier process left (it stopped before its plan finished) is set aside,
    never resumed, and the unit runs whole; the same holds for the variance precondition."""
    catalog = ladder_only(cert8.load_catalog(APPLE))
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
    """The quiet window gates every unit, not only the session: each unit reads its own
    window (two readings) before it starts, and records it."""
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
    """The quiet window's no-other-work-of-the-account rule (``docs/development/timing.md``,
    "Running rows"): a harness, test or MPI process of this account outside the session, or
    a busy Python process, intrudes; the session's own processes, a waiter's idle shell and
    other accounts do not."""
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
    # a reading is recorded: each process named by its program and the harness name that
    # matched, never by its pid or its arguments
    assert [line.split(": ", 1)[1] for line in found] == [
        "mpirun bench_meep_identical_case.py", "python pytest", "python"]
    assert "MPI" in found[0] and "busy Python" in found[2]


def test_other_work_records_no_argument_of_another_process():
    """A process's arguments can carry a remote account, an e-mail address or a repository
    URL, which no path rule recognises: an intrusion names the program and the harness name
    that qualified it, nothing else."""
    me, at = 501, "@"       # the address sign from a piece: a whole address is a guard hit
    table = {
        10: (1, me, 0.0, "python -u cert8.py run"),
        31: (1, me, 90.0, f"rsync -a results/ someone{at}remote.example.org:/data/x"),
        32: (1, me, 60.0, f"sshd: someone{at}pts/3"),
        33: (1, me, 5.0, f"python -u parity/meep_gpu/gate_metal_pml.py --notify someone{at}example.org"),
        34: (1, me, 0.0, f"zsh fleet_launch.sh git{at}github.example.org:someone/repo.git"),
        35: (1, me, 30.0, f"python3.12 tool.py https://someone:token{at}example.org/repo"),
        # a shell is named by the script that qualified it, not by a word of its own path
        36: (1, me, 0.0, "/opt/census/bin/zsh fleet_launch.sh x"),
    }
    found = cert8.other_work(table, cert8.own_pids(table, 10), me, platform="linux")
    assert [line.split(": ", 1)[1] for line in found] == [
        "rsync", "sshd:", "python gate_metal_pml.py", "zsh fleet_launch", "python3.12",
        "zsh fleet_launch"]
    for line in found:
        assert "someone" not in line and "@" not in line and "example" not in line, line


def test_a_busy_build_or_tool_of_this_account_intrudes_and_shares_are_measured():
    """The no-other-work rule as the 1.0 timing run applied it: a build, compiler or bulk tool
    of the account that burns CPU intrudes on either platform (an environment build started
    beside a ladder); on Linux any busy process of the account does, as the GPU host's watcher
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
    assert [line.split(": ", 1)[1] for line in linux] == ["make", "ptxas", "renderer"]
    assert "build or tool" in linux[0] and "a busy process (80 %" in linux[2]
    mac = cert8.other_work(table, mine, me, platform="darwin")
    assert [line.split(": ", 1)[1] for line in mac] == ["make", "ptxas"]
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
    assert not [line for line in first["intrusions"] if line.endswith(": make")]
    time.sleep(0.05)
    second = cert8.host_reading(None, os.getpid())
    assert [line for line in second["intrusions"] if line.endswith(": make")
            and "build or tool" in line]
    assert not [line for line in second["intrusions"] if "-j 32" in line], \
        "a reading is recorded: no argument"
    assert not [line for line in second["intrusions"] if "99901" in line], \
        "a reading is recorded: no pid"


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


# ---------------------------------------------------------------------------
# The extended ladder: portable catalogs, the new cases, preference legs
# ---------------------------------------------------------------------------

def test_a_portable_catalog_takes_another_architecture_by_name(tmp_path):
    catalog = cert8.load_catalog(NVIDIA, "8.9")
    keys = cert8.catalog_keys(catalog, drive=lambda table: DRIVE[table])
    assert keys and all(TR.parse_key(k)["architecture"] == "8.9" for k in keys
                        if not k.startswith("check|"))
    assert catalog["architecture_of_catalog"] == "8.6"
    assert cert8.load_catalog(APPLE, "applegpu_g14x")["architecture"] == "applegpu_g14x"
    fixed = tmp_path / "fixed.json"
    body = json.loads(Path(NVIDIA).read_text())
    body.pop("portable")
    fixed.write_text(json.dumps(body))
    with pytest.raises(cert8.Refused) as refused:
        cert8.load_catalog(str(fixed), "8.9")
    assert "is not portable" in refused.value.message
    assert cert8.load_catalog(str(fixed), "8.6")["architecture"] == "8.6"


def test_the_variance_rank_count_may_name_the_host_cores():
    assert cert8.variance_ranks({"meep_ranks": 32}) == 32
    assert cert8.variance_ranks({"meep_ranks": "2P/3"}, physical=48) == 32
    assert cert8.variance_ranks({"meep_ranks": "P"}, physical=8) == 8
    assert cert8.variance_ranks({"meep_ranks": "P/2"}, physical=10) == 5
    with pytest.raises(cert8.Refused):
        cert8.variance_ranks({"meep_ranks": "half"}, physical=8)
    # The shipped catalogs keep the 1.0 certification round's counts on the 1.0 hosts.
    nvidia = cert8.load_catalog(NVIDIA)["ladder"]["variance"]
    apple = cert8.load_catalog(APPLE)["ladder"]["variance"]
    assert cert8.variance_ranks(nvidia, physical=48) == 32
    assert cert8.variance_ranks(apple, physical=8) == 8


def test_every_extended_case_unit_runs_its_case_and_the_preference_legs_only_when_asked(
        tmp_path):
    catalog = cert8.load_catalog(NVIDIA)
    keys = [k for k in cert8.catalog_keys(catalog, drive=lambda t: DRIVE[t])
            if k.startswith("ladder|")]
    units = cert8.build_units(args_for(tmp_path), catalog, keys)
    assert sorted(u.name for u in units) == sorted(
        f"u{i:02d}_ladder_{case}_priority" for i, case in
        enumerate(sorted(EXTENDED_CASES), 1))
    for unit in units:
        argv = unit.argv
        assert argv[argv.index("--routes") + 1] == "default,cuda"
        assert argv[argv.index("--threaded-reference") + 1] == "1"
        assert argv[argv.index("--leave-free-share") + 1] == "0.25"
        assert "context" in argv[argv.index("--controls") + 1]
    asked = cert8.build_units(args_for(tmp_path, preference_legs=True), catalog, keys[:1])
    argv = asked[0].argv
    assert argv[argv.index("--routes") + 1] == "default,cuda,default_prefoff,cuda_prefoff"


def test_the_release_ladder_plans_the_seam_case_through_the_wrapper(tmp_path, monkeypatch):
    import timing_host  # noqa: PLC0415
    monkeypatch.setattr(timing_host, "topology", lambda *a, **k: {"physical_cores": 48})
    catalog = cert8.load_catalog(NVIDIA)
    key = TR.ladder_key("8.6", "pml_3d_hz", 512000, "priority")
    unit = cert8.build_units(args_for(tmp_path), catalog, [key])[0]
    plan = ladder_dry_run(unit)
    gpu = [row for row in plan["rows"] if row["kind"] == "gpu" and row["tier"] == "priority"]
    assert {row["route"] for row in gpu} == {"default", "cuda"}
    assert all(row["expected_repair"] == "B" for row in gpu)
    assert all(row["argv"][2].endswith("bench_timing_case.py") for row in gpu)
    meep = [row for row in plan["rows"] if row["kind"] == "meep"]
    assert max(row["ranks"] * row["configuration"]["threads_per_rank"] for row in meep) <= 72
    hybrid = [row for row in meep if row["configuration"]["threads_per_rank"] > 1]
    assert hybrid and all(row["control"] is None for row in hybrid)
    assert "context" in plan["resolved"]["controls_omitted"]


# ---------------------------------------------------------------------------
# What a reading writes about the machine (neutral_ids)
# ---------------------------------------------------------------------------

HOST = "gpu-node-17.cluster.example.edu"
LOGIN = "fixtureuser"
HOME = "/" + "ho" + "me" + "/fixtureuser"   # from pieces: a full home path is a hygiene-guard hit
REAL_A = "GPU-3f1c2a9e-5b7d-4e21-9a0c-6d8e2f4b1a37"
REAL_B = "GPU-9b2e7c41-0d6a-4f83-b5e9-1c7a3d8f6e02"
FOREIGN_PID = 271828


@pytest.fixture
def machine(monkeypatch):
    """This process as the fixture machine: its host names, login and home."""
    monkeypatch.setattr(neutral_ids.socket, "gethostname", lambda: HOST)
    monkeypatch.setattr(neutral_ids, "_node", lambda: HOST)
    monkeypatch.setattr(neutral_ids.getpass, "getuser", lambda: LOGIN)
    monkeypatch.setenv("HOME", HOME)
    neutral_ids.reset()
    yield neutral_ids
    neutral_ids.reset()


def assert_neutral(record, *tokens):
    assert neutral_ids.residue(record) == []
    text = json.dumps(record, sort_keys=True, default=str)
    for token in (HOST, HOST.split(".")[0], LOGIN, HOME, REAL_A, REAL_B,
                  str(FOREIGN_PID)) + tokens:
        assert str(token) not in text, (token, text[:400])


def _host_stub(me, apps, neighbours=()):
    return types.SimpleNamespace(
        memory=lambda: {"available_gib": 900.0, "total_gib": 1000.0},
        loadavg=lambda: [0.1, 0.1, 0.1], load1=lambda: 0.1,
        metal_neighbours=lambda: list(neighbours), darwin_daemons=lambda: [],
        cpu_state_darwin=lambda: None, apple_gpu_state=lambda: None,
        compute_apps=lambda _gpu: [dict(app) for app in apps],
        gpu_uuids=lambda: {"0": REAL_A, "1": REAL_B},
        run_text=lambda _argv: f"{REAL_A}, 0\n{REAL_B}, 0\n",
        as_uuid=lambda gpu, uuids: uuids.get(gpu, gpu),
        uid_of_pid=lambda pid: me if pid in (4242, 4343) else me + 1,
        default_mem_min_gib=lambda total: 250.0, default_cpu_max=lambda n: 9)


def test_a_gpu_reading_compares_raw_and_writes_cards_and_memory(machine, monkeypatch):
    """The reading compares pids, uids and UUIDs raw and keeps its verdicts; what it
    records names a GPU process by its card id and memory, never its pid or UUID."""
    me = os.getuid()
    apps = [{"gpu_uuid": REAL_A, "pid": FOREIGN_PID, "used_memory": "800 MiB"},
            {"gpu_uuid": REAL_B, "pid": 4242, "used_memory": "100 MiB"},
            {"gpu_uuid": REAL_B, "pid": FOREIGN_PID + 1, "used_memory": "4970 MiB"}]
    monkeypatch.setitem(sys.modules, "timing_host", _host_stub(me, apps))
    monkeypatch.setattr(cert8, "_process_table", lambda shares: {
        FOREIGN_PID + 2: (1, me, 50.0, f"python -m pytest -q {HOME}/meep-gpu/parity")})
    monkeypatch.setattr(cert8.sys, "platform", "linux")
    reading = cert8.host_reading("0", os.getpid())
    card_a, card_b = neutral_ids.card_id(REAL_A), neutral_ids.card_id(REAL_B)
    assert reading["intrusions"] == [
        "other work of this account (a harness, test or round process): python pytest",
        f"a foreign compute process on the device: {card_a}, 800 MiB",
        f"another GPU process of this account: {card_b}, 100 MiB"]
    assert reading["recorded"]["idle_neighbours"] == [
        {"gpu": card_b, "used_memory": "4970 MiB", "utilization": 0}]
    assert_neutral(reading, str(FOREIGN_PID + 1), str(FOREIGN_PID + 2), "4242")


def test_an_apple_reading_writes_its_neighbours_without_pids(machine, monkeypatch):
    me = os.getuid()
    lines = [f"{FOREIGN_PID} 1 {me + 1} python -u {HOME}/x/parity/meep_gpu/gate_metal_bfast.py",
             f"{FOREIGN_PID + 1} 1 {me} zsh {HOME}/x/recut_metal_gates.sh a b"]
    monkeypatch.setitem(sys.modules, "timing_host", _host_stub(me, [], lines))
    monkeypatch.setattr(cert8, "_process_table", lambda shares: {
        FOREIGN_PID + 5: (1, me + 1, 0.0, "idle")})
    monkeypatch.setattr(cert8.sys, "platform", "darwin")
    reading = cert8.host_reading(None, os.getpid())
    assert reading["intrusions"] == [
        "another process driving the Apple GPU: other python gate_metal_bfast.py",
        "another process driving the Apple GPU: own zsh recut_metal_gates.sh"]
    assert_neutral(reading, str(FOREIGN_PID + 1), str(me + 1))


def test_a_host_named_like_a_catalog_word_is_refused_before_a_unit_runs(machine, monkeypatch,
                                                                       tmp_path, capsys):
    import timing_host  # noqa: PLC0415

    catalog = cert8.load_catalog(NVIDIA)
    assert "triton" in cert8.catalog_words(catalog)
    key = TR.fused_key("8.6", "triton", "pml_2d", "res20", 1)
    argv = ["run", "--catalog", NVIDIA, "--store", str(tmp_path / "store"),
            "--run-root", str(tmp_path / "runs"), "--rows", key]
    monkeypatch.setattr(neutral_ids.socket, "gethostname", lambda: "pml_2d")
    monkeypatch.setattr(neutral_ids, "_node", lambda: "pml_2d")
    neutral_ids.reset()
    assert "pml_2d" in cert8.catalog_words(catalog)
    assert cert8.main(argv) == cert8.EXIT_REFUSED
    out = capsys.readouterr().out
    assert timing_host.VOCABULARY_REFUSAL in out
    assert not os.path.exists(tmp_path / "runs"), "refused before anything is written"


# ---------------------------------------------------------------------------
# A WSL2 host: the Windows side joins every reading
# ---------------------------------------------------------------------------

import timing_host as real_host  # noqa: E402

AT_LOCK = "1695, 9501, P2, 0x0000000000000001, 117.75, 55"
OFF_LOCK = "1485, 5001, P3, 0x0000000000000001, 90.00, 55"
WIN_QUIET = "cpus 24\nsample 7.6 0 1\nsample 5.1 7.8 1\n"
WIN_BUSY = "cpus 24\nsample 30.0 0 1\nsample 25.0 7.8 1\n"
WSL2 = {"wsl2": True, "why": "the kernel is a Microsoft build for WSL"}
#: The VM affinity mask the lab desktop qualified under, and the VM's full mask.
MASK, FULL = 0x999988, 0xFFFFFF


def windows_observed(gpu=AT_LOCK, load=WIN_QUIET, unreadable=(), mask=FULL):
    return {"utc": "t", "gpu": real_host.parse_windows_gpu(gpu)[0] if gpu else None,
            "cpu": real_host.parse_windows_load(load) if load else None,
            "vm_affinity": (real_host.parse_vm_affinity(f"affinity {mask} 1")
                            if mask is not None else None),
            "unreadable": list(unreadable), "seconds": 0.0}


class Setter:
    """The interop set of the VM's mask, as the lab desktop answered it: elevated, the set
    takes and reads back; ``denied``, the read gives 0 and the set is refused."""

    def __init__(self, before=FULL, denied=False):
        self.before, self.denied, self.calls = before, denied, []

    def __call__(self, mask):
        self.calls.append(mask)
        if self.denied:
            return {"ok": False, "declared": f"{mask:#x}", "processes": 1, "before": "0x0",
                    "set": "error Win32Exception 5", "read_back": "0x0",
                    "why": "the set was refused: " + real_host.WIN32_ERRORS[5]}
        before, self.before = self.before, mask
        return {"ok": True, "declared": f"{mask:#x}", "processes": 1, "before": f"{before:#x}",
                "set": "ok", "read_back": f"{mask:#x}", "why": None}


def wsl2_box(monkeypatch, me):
    """The box of a WSL2 host as host_reading reads it: an idle VM, no GPU process
    listed (WSL lists none), the platform Linux."""
    stub = _host_stub(me, [])
    for name in ("windows_summary", "windows_reasons", "windows_reading", "WindowsRule",
                 "set_vm_affinity"):
        setattr(stub, name, getattr(real_host, name))
    monkeypatch.setitem(sys.modules, "timing_host", stub)
    monkeypatch.setattr(cert8, "_process_table", lambda shares: {
        FOREIGN_PID + 5: (1, me + 1, 0.0, "idle")})
    monkeypatch.setattr(cert8.sys, "platform", "linux")


def test_on_a_native_host_the_windows_rule_is_skipped_by_name_and_reads_nothing(
        machine, monkeypatch, tmp_path):
    """Native Linux and macOS: no Windows program runs, no reading changes, the watcher's
    rule is the two-in-a-row rule, and the session's gates say the rule was skipped and why
    (a declared lock is recorded as not checked)."""
    def no_windows(*_args, **_kwargs):
        raise AssertionError("a Windows reading on a native host")

    monkeypatch.setattr(real_host, "windows_reading", no_windows)
    for host in ({"wsl2": False, "why": "native Linux: the kernel is not a Microsoft build"},
                 {"wsl2": False, "why": "not Linux (darwin)"}):
        monkeypatch.setattr(real_host, "wsl_host", lambda host=host: host)
        side = cert8.windows_side(args_for(tmp_path, gpu_clock_lock="1695,9501"))
        assert side.applied is False
        assert side.describe() == {"applied": False, "skipped": f"not a WSL2 host: {host['why']}",
                                   "declared_lock_not_checked": {"graphics_mhz": 1695,
                                                                 "memory_mhz": 9501}}
        assert cert8.windows_side(args_for(tmp_path)).describe() == {
            "applied": False, "skipped": f"not a WSL2 host: {host['why']}"}
    me = os.getuid()
    wsl2_box(monkeypatch, me)
    reading = cert8.host_reading("0", os.getpid(), gate=True, windows=side)
    assert reading == cert8.host_reading("0", os.getpid(), gate=True) | {"utc": reading["utc"]}
    assert "windows" not in reading["recorded"] and "decisive" not in reading
    watcher = cert8.Watcher(lambda: {})
    hits = [watcher.observe({"utc": str(i), "intrusions": r, "recorded": {}})
            for i, r in enumerate([["a"], [], ["b"], ["b"]])]
    assert hits == [False, False, False, True] and "windows" not in watcher.stop()


def test_a_wsl2_reading_records_the_windows_side_and_names_what_fails(machine, monkeypatch):
    me = os.getuid()
    wsl2_box(monkeypatch, me)
    state = {"observed": windows_observed()}
    side = cert8.WindowsSide(WSL2, real_host.WindowsRule(1695, 9501),
                             reader=lambda: state["observed"])
    reading = cert8.host_reading("0", os.getpid(), windows=side)
    assert reading["intrusions"] == [] and "decisive" not in reading
    recorded = reading["recorded"]["windows"]
    assert (recorded["gpu"]["graphics_mhz"], recorded["gpu"]["memory_mhz"],
            recorded["gpu"]["pstate"], recorded["gpu"]["reasons"]) == (1695, 9501, "P2", "0x1")
    assert recorded["cpu"]["outside_vm_busy_cpus"] == pytest.approx(1.15, abs=0.01)
    state["observed"] = windows_observed(OFF_LOCK)
    off = cert8.host_reading("0", os.getpid(), windows=side)
    assert off["intrusions"] == off["decisive"] == [
        "the card left the declared clock lock: graphics 1485 MHz, memory 5001 MHz, "
        "against 1695 / 9501"]
    state["observed"] = windows_observed(None, None, ["nvidia-smi.exe: no answer in 30 s"])
    lost = cert8.host_reading("0", os.getpid(), windows=side)
    assert lost["decisive"] == ["Windows side unreadable: nvidia-smi.exe: no answer in 30 s"]
    assert_neutral([reading, off, lost])


def _strike_session(tmp_path, monkeypatch, once, mask=None):
    """A session of one fused row on a WSL2 host whose Windows side reads ``once`` on ONE
    watcher reading of the first attempt, quiet otherwise (with ``mask`` declared, the VM
    holds it on every other reading)."""
    me = os.getuid()
    wsl2_box(monkeypatch, me)
    catalog = cert8.load_catalog(NVIDIA)
    key = TR.fused_key("8.6", "triton", "pml_2d", "res20", 1)
    args = args_for(tmp_path, no_variance="fused rows only")
    environment = tmp_path / "build_record.json"
    environment.write_text(json.dumps({"platform": "linux-64"}))
    args.environment_record = str(environment)
    units = cert8.build_units(args, catalog, [key])
    state = {"once": None, "served": 0}

    def windows_reader():
        if state["once"] is not None:
            observed, state["once"] = state["once"], None
            state["served"] += 1
            return observed
        return windows_observed(mask=mask or FULL)

    setter = Setter()
    side = cert8.WindowsSide(WSL2, real_host.WindowsRule(1695, 9501, vm_affinity=mask),
                             reader=windows_reader, setter=setter)
    calls = []

    def runner(unit, stopped):
        calls.append(unit.name)
        os.makedirs(unit.out, exist_ok=True)
        with open(os.path.join(unit.out, "rows.jsonl"), "a") as handle:
            handle.write(bench_line("pml_2d", "triton", "8.6"))
        if len(calls) == 1:
            state["once"] = once
            deadline = time.time() + 10
            while not stopped() and time.time() < deadline:
                time.sleep(0.01)
            return -15 if stopped() else 0
        time.sleep(0.3)
        return 0

    session = cert8.Session(args, catalog, runner=runner, minter=lambda a: str(tmp_path / "k"),
                            windows=side)
    assert session.run(units) == 0
    assert state["served"] == 1, "one reading, and only one, read the departure"
    assert setter.calls == ([mask, FULL] if mask else []), "set first, restored at the end"
    return args, units, calls


@pytest.mark.parametrize("once, named, mask", [
    (windows_observed(OFF_LOCK), "the card left the declared clock lock", None),
    (windows_observed(load=WIN_BUSY), "Windows busy outside the WSL VM: 5.92 logical CPUs >= 2",
     None),
    (windows_observed(mask=FULL), "the WSL VM's CPU affinity is 0xffffff, not the declared "
                                  "0x999988", MASK),
], ids=["clock_left_the_lock", "busy_windows", "vm_mask_changed"])
def test_one_reading_off_the_lock_or_of_a_busy_windows_strikes_the_unit(
        machine, tmp_path, monkeypatch, once, named, mask):
    """The card leaving its lock, Windows busy outside the VM, or the VM's affinity mask
    moved off the declared one, on ONE watcher reading, strikes the unit: it is stopped,
    its output set aside, and it runs again on a quiet window; only the clean attempt is
    recorded, with the Windows side in its gates."""
    args, units, calls = _strike_session(tmp_path, monkeypatch, once, mask)
    assert calls == [units[0].name, units[0].name], "the struck unit runs again"
    aside = [p for p in os.listdir(os.path.dirname(units[0].out)) if "intruded" in p]
    assert len(aside) == 1, "the struck output is set aside, never collected"
    progress = open(os.path.join(args.run_root, "s1", "progress.log")).read()
    assert "INTRUSION" in progress and named in progress
    assert "WSL2 host: the Windows side joins every reading (clock lock 1695 / 9501 MHz" in progress
    records = TR.load_store(args.store)
    assert len(records) == 1 and TR.refusals(records[0]) == []
    gates = records[0]["gates"]
    assert gates["watcher"]["intrusions"] == []
    assert gates["windows"]["applied"] is True and gates["windows"]["lock"] == {
        "graphics_mhz": 1695, "memory_mhz": 9501}
    for reading in gates["quiet"] + [gates["watcher"]["first"], gates["watcher"]["last"]]:
        windows = reading["recorded"]["windows"]
        assert (windows["gpu"]["graphics_mhz"], windows["gpu"]["pstate"],
                windows["gpu"]["reasons"]) == (1695, "P2", "0x1")
        assert windows["cpu"]["outside_vm_busy_cpus"] < 2
    summary = gates["watcher"]["windows"]
    assert summary["graphics_mhz"] == [1695] and summary["memory_mhz"] == [9501]
    assert summary["pstates"] == ["P2"] and summary["reasons"] == ["0x1"]
    assert summary["readings"] == gates["watcher"]["readings"]
    assert summary["vm_affinity"] == [f"{mask or FULL:#x}"]
    assert_neutral(records[0]["gates"])


def test_a_wsl2_host_whose_windows_side_cannot_be_read_is_refused_by_name(tmp_path):
    """Interop off: the session refuses before any unit, naming what it could not read,
    rather than time a unit whose gate saw the VM alone."""
    catalog = cert8.load_catalog(NVIDIA)
    args = args_for(tmp_path, no_variance="fused rows only", allow_unrecorded_environment=True)
    units = cert8.build_units(args, catalog, [TR.fused_key("8.6", "triton", "pml_2d", "res20", 1)])
    side = cert8.WindowsSide(WSL2, real_host.WindowsRule(1695, 9501),
                             reader=lambda: real_host.windows_reading(program=lambda n: None))
    ran = []
    clear = lambda: {"intrusions": [], "utc": "t"}  # noqa: E731 - the VM's side is quiet
    session = cert8.Session(args, catalog, reader=clear, gate_reader=clear,
                            runner=lambda unit, stopped: ran.append(1) or 0,
                            minter=lambda a: "k", windows=side)
    with pytest.raises(cert8.Refused) as refusal:
        session.run(units)
    assert "a WSL2 host whose Windows side cannot be read: nvidia-smi.exe not found" in str(
        refusal.value.message) and "powershell.exe not found" in str(refusal.value.message)
    assert ran == [] and TR.load_store(args.store) == []
    # a caller that runs one unit alone (the variance precondition of a fused-only
    # session, run on its own) is refused the same way, before the unit's window
    alone = cert8.Session(args, catalog, reader=clear, gate_reader=clear,
                          runner=lambda unit, stopped: ran.append(1) or 0,
                          minter=lambda a: "k", windows=side)
    with pytest.raises(cert8.Refused) as refusal:
        alone.run_unit(units[0])
    assert "Windows side cannot be read" in refusal.value.message and ran == []


def test_the_variance_record_carries_the_windows_readings(machine, tmp_path, monkeypatch):
    me = os.getuid()
    wsl2_box(monkeypatch, me)
    monkeypatch.setattr(sys.modules["timing_host"], "topology",
                        lambda *a, **k: {"physical_cores": 48}, raising=False)
    catalog = cert8.load_catalog(NVIDIA)
    args = args_for(tmp_path, allow_unrecorded_environment=True)
    units = cert8.build_units(args, catalog, ["ladder|8.6|pml_3d|512000|priority"])
    side = cert8.WindowsSide(WSL2, real_host.WindowsRule(), reader=windows_observed)

    def runner(unit, stopped):
        if unit.kind != "variance":
            return 3
        os.makedirs(unit.out, exist_ok=True)
        variance_ledger(Path(unit.out) / "ladder_rows.jsonl",
                        [747.0, 746.0, 742.6, 738.5, 735.0], [500.0] * 5)
        time.sleep(0.2)
        return 0

    session = cert8.Session(args, catalog, runner=runner, minter=lambda a: "k", windows=side)
    session.run(units)
    verdict = json.load(open(os.path.join(args.run_root, "s1", "variance.json")))
    assert verdict["verdict"] == "GO"
    windows = verdict["windows"]
    assert windows["applied"] is True and windows["lock"] is None
    assert [r["gpu"]["graphics_mhz"] for r in windows["quiet"]] == [1695, 1695]
    assert windows["watcher"]["graphics_mhz"] == [1695] and windows["watcher"]["readings"] >= 1
    assert_neutral(windows)


def test_the_clock_lock_is_declared_by_option_or_environment(monkeypatch, tmp_path):
    base = ["run", "--catalog", NVIDIA, "--store", "s", "--run-root", "r"]
    monkeypatch.setenv("MGPU_TIMING_GPU_CLOCK_LOCK", "1695,9501")
    monkeypatch.setenv("MGPU_TIMING_WINDOWS_CPU_MAX", "3")
    args = cert8.build_parser().parse_args(base)
    assert (args.gpu_clock_lock, args.windows_cpu_max) == ("1695,9501", "3")
    args = cert8.build_parser().parse_args(base + ["--gpu-clock-lock", "1710,9501"])
    monkeypatch.setattr(real_host, "wsl_host", lambda: {"wsl2": False, "why": "x"})
    rule = cert8.windows_side(args).rule
    assert (rule.lock_graphics_mhz, rule.lock_memory_mhz, rule.cpu_max) == (1710, 9501, 3.0)
    for lock, cpu_max, named in (("1695", None, "GRAPHICS,MEMORY"),
                                 ("1695,9501", "0", "a threshold of busy CPUs above 0"),
                                 ("1695,9501", "lots", "could not convert")):
        with pytest.raises(cert8.Refused) as refusal:
            cert8.windows_side(args_for(tmp_path, gpu_clock_lock=lock, windows_cpu_max=cpu_max))
        assert "--windows-cpu-max" in refusal.value.message and named in refusal.value.message


# ---------------------------------------------------------------------------
# A WSL2 host: the VM affinity mask
# ---------------------------------------------------------------------------

def _mask_session(tmp_path, monkeypatch, keys, setter, **extra):
    """A session on a WSL2 host with the mask declared; its runner notes each unit with the
    mask the VM held while it ran."""
    me = os.getuid()
    wsl2_box(monkeypatch, me)
    monkeypatch.setattr(sys.modules["timing_host"], "topology",
                        lambda *a, **k: {"physical_cores": 48}, raising=False)
    # A quiet window that cannot open gives up in a second here, so a session that would
    # wait on a VM off its mask fails at once rather than after the driver's 12 hours.
    monkeypatch.setattr(cert8, "quiet_window", functools.partial(
        cert8.quiet_window, give_up=1.0, sleep=lambda _s: time.sleep(0.01)))
    catalog = cert8.load_catalog(NVIDIA)
    args = args_for(tmp_path, allow_unrecorded_environment=True, **extra)
    units = cert8.build_units(args, catalog, keys)
    side = cert8.WindowsSide(WSL2, real_host.WindowsRule(1695, 9501, vm_affinity=MASK),
                             reader=lambda: windows_observed(mask=setter.before), setter=setter)
    events = []

    def runner(unit, stopped):
        events.append(("unit", unit.kind, setter.before))
        os.makedirs(unit.out, exist_ok=True)
        if unit.kind == "variance":
            # Under the mask the precondition is a GPU half: its ladder times no MEEP row.
            variance_ledger(Path(unit.out) / "ladder_rows.jsonl",
                            [747.0, 746.0, 742.6, 738.5, 735.0],
                            None if unit.side == "gpu" else [500.0] * 5)
        else:
            with open(os.path.join(unit.out, "rows.jsonl"), "a") as handle:
                handle.write(bench_line("pml_2d", "triton", "8.6"))
        time.sleep(0.2)
        return 0

    session = cert8.Session(args, catalog, runner=runner, minter=lambda a: str(tmp_path / "k"),
                            windows=side)
    return args, units, session, events


def test_a_declared_mask_is_set_before_the_first_unit_recorded_and_restored(
        machine, tmp_path, monkeypatch):
    """The session sets the mask through interop before its first unit, reads it back,
    times the unit under it, records it as set and as read in the row, and gives the VM's
    process its earlier mask back at its end."""
    setter = Setter()
    key = TR.fused_key("8.6", "triton", "pml_2d", "res20", 1)
    args, units, session, events = _mask_session(tmp_path, monkeypatch, [key], setter,
                                                 no_variance="fused rows only")
    assert session.run(units) == 0
    assert setter.calls == [MASK, FULL] and events == [("unit", "fused", MASK)], \
        "set before the unit, restored after it"
    record = TR.load_store(args.store)[0]
    windows = record["gates"]["windows"]
    assert windows["vm_affinity"] == "0x999988"
    assert windows["vm_affinity_set"] == {"ok": True, "declared": "0x999988", "processes": 1,
                                          "before": "0xffffff", "set": "ok",
                                          "read_back": "0x999988", "why": None}
    for reading in record["gates"]["quiet"]:
        assert reading["recorded"]["windows"]["vm_affinity"]["mask"] == "0x999988"
    assert record["gates"]["watcher"]["windows"]["vm_affinity"] == ["0x999988"]
    progress = open(os.path.join(args.run_root, "s1", "progress.log")).read()
    assert "VM affinity: declared 0x999988, before 0xffffff, set ok, read back 0x999988" in progress
    assert "VM affinity restored: set ok, read back 0xffffff" in progress
    assert TR.refusals(record, allow_unrecorded_environment=True) == []
    assert_neutral(record["gates"])


def test_a_mask_that_does_not_read_back_refuses_the_session_by_name(machine, tmp_path,
                                                                     monkeypatch):
    """An unelevated session: the read gives 0 and the set is denied. Nothing runs, and
    the refusal names both."""
    setter = Setter(denied=True)
    key = TR.fused_key("8.6", "triton", "pml_2d", "res20", 1)
    args, units, session, events = _mask_session(tmp_path, monkeypatch, [key], setter,
                                                 no_variance="fused rows only")
    with pytest.raises(cert8.Refused) as refusal:
        session.run(units)
    message = refusal.value.message
    assert "the WSL VM's CPU affinity could not be held at the declared 0x999988" in message
    assert "access denied" in message and "elevated context" in message
    assert "read back 0x0" in message
    assert events == [] and setter.calls == [MASK], "nothing ran, and nothing to restore"
    assert TR.load_store(args.store) == []


def test_meep_rows_are_never_timed_under_a_declared_mask(machine, tmp_path, monkeypatch):
    """A ladder row and a ladder row's dispatch check time MEEP beside the GPU legs in one
    ladder process: with a mask declared the session is refused by name before the mask is
    set. The variance precondition, which qualifies the host under the mask, is admitted
    as a GPU half: it times the GPU route alone, its record carries the mask, and it
    records no MEEP side."""
    setter = Setter()
    ladder = "ladder|8.6|pml_3d|512000|priority"
    fused = TR.fused_key("8.6", "triton", "pml_2d", "res20", 1)
    for index, keys in enumerate(([ladder, fused], [TR.check_key(ladder)])):
        args, units, session, events = _mask_session(tmp_path / f"s{index}", monkeypatch,
                                                      keys, setter)
        with pytest.raises(cert8.Refused) as refusal:
            session.run(units)
        assert "MEEP is never timed under it" in refusal.value.message
        assert "Time them in a session without --vm-affinity" in refusal.value.message
        with pytest.raises(cert8.Refused):
            session.run_unit([u for u in units if cert8.is_ladder_unit(u)][0])
        assert events == [] and setter.calls == [], "refused before the mask is set"
    args, units, session, events = _mask_session(tmp_path / "variance", monkeypatch,
                                                 [fused], setter)
    verdict = session.variance_precondition()
    session.close_windows()
    assert verdict["verdict"] == "GO" and events == [("unit", "variance", MASK)]
    assert setter.calls == [MASK, FULL]
    stored = json.load(open(os.path.join(args.run_root, "s1", "variance.json")))
    assert stored["windows"]["vm_affinity_set"]["read_back"] == "0x999988"
    assert stored["windows"]["watcher"]["vm_affinity"] == ["0x999988"]
    assert [r["vm_affinity"]["mask"] for r in stored["windows"]["quiet"]] == ["0x999988"] * 2
    assert stored["meep_side"].startswith("not run: under a declared VM affinity mask")
    assert set(stored["sides"]) == {"gpu"} and stored["meep_ranks"] is None
    # a unit timed after the restore sets the mask again before it runs
    assert session.run(units) == 0
    assert setter.calls == [MASK, FULL, MASK, FULL] and events[-1] == ("unit", "fused", MASK)


def test_on_a_native_host_a_declared_mask_is_skipped_by_name_and_sets_nothing(
        machine, tmp_path, monkeypatch):
    """Native Linux or macOS: the declared mask is recorded as not checked, nothing is set,
    and ladder units are not refused (MEEP runs as before)."""
    setter = Setter()
    monkeypatch.setattr(real_host, "wsl_host",
                        lambda: {"wsl2": False, "why": "native Linux: the kernel is not a "
                                                       "Microsoft build"})
    side = cert8.windows_side(args_for(tmp_path, vm_affinity="0x999988"))
    side.setter = setter
    assert side.masked is False and side.describe() == {
        "applied": False,
        "skipped": "not a WSL2 host: native Linux: the kernel is not a Microsoft build",
        "declared_vm_affinity_not_checked": "0x999988"}
    catalog = cert8.load_catalog(NVIDIA)
    args = args_for(tmp_path, allow_unrecorded_environment=True)
    monkeypatch.setattr(real_host, "topology", lambda *a, **k: {"physical_cores": 48})
    units = cert8.build_units(args, catalog, ["ladder|8.6|pml_3d|512000|priority"])
    clear = lambda: {"intrusions": [], "utc": "t"}  # noqa: E731
    session = cert8.Session(args, catalog, reader=clear, gate_reader=clear,
                            runner=lambda unit, stopped: 0, minter=lambda a: "k", windows=side)
    session.refuse_meep(units)
    assert session.open_windows()["declared_vm_affinity_not_checked"] == "0x999988"
    session.close_windows()
    assert setter.calls == []


# ---------------------------------------------------------------------------
# The split ladder: a ladder unit timed as a GPU half under the mask and a MEEP half
# with it cleared
# ---------------------------------------------------------------------------

SPLIT_KEYS = ["ladder|8.6|pml_3d|512000|priority", "ladder|8.6|pml_3d_hz|512000|priority",
              "ladder|8.6|pml_3d_lorentz|512000|priority"]


def test_the_split_ladder_plans_two_halves_per_ladder_unit_in_alternating_order(tmp_path):
    """Without ``--split-ladder`` a ladder unit is one process, as before. With it, each
    ladder unit runs no process of its own: its two halves are the unit's command with
    ``--ranks none`` (the GPU half) or ``--routes none`` (the MEEP half) appended under
    their own stamps, GPU first on the first unit, MEEP first on the second, and so on; a
    ladder row's dispatch check is a GPU half alone."""
    catalog = cert8.load_catalog(NVIDIA)
    keys = SPLIT_KEYS + [TR.check_key(SPLIT_KEYS[0])]
    site = site_params(tmp_path / "site")
    whole = cert8.build_units(args_for(tmp_path, ladder_params=site), catalog, keys)
    split = cert8.build_units(args_for(tmp_path, split_ladder=True, ladder_params=site),
                              catalog, keys)
    assert all(not u.children and u.side is None for u in whole)
    assert not any({"split", "side"} & set(u.describe()) for u in whole)
    assert not any(a in ("none",) for u in whole for a in u.argv)
    assert [(u.name, u.kind, u.keys) for u in split] == [(u.name, u.kind, u.keys) for u in whole]
    ladders = [u for u in split if u.kind == "ladder"]
    assert [[c.side for c in u.children] for u in ladders] == [
        ["gpu", "meep"], ["meep", "gpu"], ["gpu", "meep"]]
    # the MEEP half is handed the host's MEEP rank ceiling: the site's largest rank block
    extra = {"gpu": ["--ranks", "none"],
             "meep": ["--meep-rank-ceiling", "64", "--routes", "none"]}
    for unit, before in zip(ladders, whole):
        assert unit.argv == [] and unit.out == before.out
        stamp = before.argv[before.argv.index("--stamp") + 1]
        for child in unit.children:
            assert child.argv == [f"{stamp}_{child.side}" if a == stamp else a
                                  for a in before.argv] + extra[child.side]
            assert child.out == f"{before.out}_{child.side}" and child.keys == unit.keys
            assert child.kind == "ladder" and not child.children
        assert unit.describe()["split"]["order"] == [c.side for c in unit.children]
    check, = [u for u in split if u.kind == "check"]
    assert check.side == "gpu" and check.argv[-2:] == ["--ranks", "none"]
    assert check.argv[:-2] == whole[-1].argv


def test_the_two_halves_plan_the_whole_unit_between_them(tmp_path, monkeypatch):
    """The release ladder's own dry run of a unit and of its two halves: the GPU half
    plans the unit's GPU rows and nothing that starts MEEP, the MEEP half its MEEP rows,
    the preflight digest and the harness-lift digests and no GPU row, and between them
    they plan every row of the unit, each with the command, working directory and
    environment the whole unit gives it."""
    catalog = cert8.load_catalog(NVIDIA)
    site = site_params(tmp_path / "site")
    whole, = cert8.build_units(args_for(tmp_path, ladder_params=site), catalog, SPLIT_KEYS[:1])
    split, = cert8.build_units(args_for(tmp_path, split_ladder=True, ladder_params=site),
                               catalog, SPLIT_KEYS[:1])
    stamp = whole.argv[whole.argv.index("--stamp") + 1]
    planned = {"whole": ladder_dry_run(whole)}
    for child in split.children:
        text = json.dumps(ladder_dry_run(child)).replace(f"{stamp}_{child.side}", stamp)
        planned[child.side] = json.loads(text)
    rows = {name: {row["key"]: row for row in plan["rows"]} for name, plan in planned.items()}
    kinds = {name: {row["kind"] for row in by_key.values()} for name, by_key in rows.items()}
    assert kinds["gpu"] == {"gpu", "control"} and kinds["meep"] == {"meep", "digest", "lift"}
    assert kinds["whole"] == kinds["gpu"] | kinds["meep"]
    assert not set(rows["gpu"]) & set(rows["meep"])
    assert set(rows["gpu"]) | set(rows["meep"]) == set(rows["whole"])
    for side in ("gpu", "meep"):
        for key, row in rows[side].items():
            ours = rows["whole"][key]
            assert (row["argv"], row["cwd"], row["env"]) == (ours["argv"], ours["cwd"],
                                                             ours["env"]), (side, key)
    resolved = planned["gpu"]["resolved"]
    assert resolved["rank_blocks"] == {"A": [], "B": [], "C": []}
    assert resolved["preflight_ranks"] == [] and resolved["probe_ranks"] == []
    assert resolved["controls"] == [] and all(
        "--ranks none" in why for why in resolved["controls_omitted"].values())
    assert planned["meep"]["resolved"]["routes"] == []
    assert planned["meep"]["resolved"]["meep_rank_ceiling"] == 64
    assert "meep_rank_ceiling" not in planned["whole"]["resolved"]


def ladder_half(unit, side_utc):
    """What a split half's ladder writes (``test_timing_rows.write_ladder``), its rows
    stamped now so they fall inside the session."""
    import test_timing_rows as rows  # noqa: PLC0415
    parsed = TR.parse_key(unit.keys[0])
    if unit.kind == "check":
        parsed = TR.parse_key(parsed["target"])
    rows.write_ladder(Path(unit.out), gpu=unit.side == "gpu", meep=unit.side == "meep",
                      utc=side_utc, case=parsed["case"], cells=parsed["cells"],
                      tier="preflight" if unit.kind == "check" else parsed["tier"],
                      architecture="8.6")
    # Every ladder probes the MEEP builds it names; the MEEP half's are the MEEP rows'.
    builds = Path(unit.out) / "meep_builds"
    builds.mkdir(exist_ok=True)
    (builds / "reference.json").write_text(json.dumps({"half": unit.side}))


def _split_session(tmp_path, monkeypatch, keys, setter, reader=None, fail=None, **extra):
    """A split session on a WSL2 host with the mask declared: the runner writes each half's
    ladder (or the variance ledger) and notes each run with the mask the VM held."""
    me = os.getuid()
    wsl2_box(monkeypatch, me)
    monkeypatch.setattr(sys.modules["timing_host"], "topology",
                        lambda *a, **k: {"physical_cores": 48}, raising=False)
    monkeypatch.setattr(sys.modules["timing_host"], "parse_affinity_mask",
                        real_host.parse_affinity_mask, raising=False)
    monkeypatch.setattr(cert8, "quiet_window", functools.partial(
        cert8.quiet_window, give_up=1.0, sleep=lambda _s: time.sleep(0.01)))
    catalog = cert8.load_catalog(NVIDIA)
    args = args_for(tmp_path, allow_unrecorded_environment=True, split_ladder=True, **extra)
    units = cert8.build_units(args, catalog, keys)
    side = cert8.WindowsSide(WSL2, real_host.WindowsRule(1695, 9501, vm_affinity=MASK),
                             reader=reader or (lambda: windows_observed(mask=setter.before)),
                             setter=setter)
    events = []

    def runner(unit, stopped):
        events.append((unit.name, unit.side, setter.before))
        os.makedirs(unit.out, exist_ok=True)
        if unit.kind == "variance":
            # A split session's precondition is a GPU half: its ladder times no MEEP row.
            variance_ledger(Path(unit.out) / "ladder_rows.jsonl",
                            [747.0, 746.0, 742.6, 738.5, 735.0],
                            None if unit.side == "gpu" else [500.0] * 5)
        else:
            ladder_half(unit, cert8.utc())
        deadline = time.time() + 0.3
        while not stopped() and time.time() < deadline:
            time.sleep(0.01)
        if stopped():
            return -15
        return (fail or {}).get(unit.name, 0)

    session = cert8.Session(args, catalog, runner=runner, minter=lambda a: str(tmp_path / "k"),
                            windows=side)
    return args, units, session, events


def test_a_split_session_times_each_half_under_its_mask_and_composes_one_row(
        machine, tmp_path, monkeypatch):
    """Each half has its mask set and read back before its quiet window: the declared
    mask for the GPU half (and the variance precondition and a dispatch check), every one
    of Windows' 24 logical CPUs for the MEEP half; the first half alternates unit by unit,
    the session's earlier mask is given back at its end, and each ladder row is one record
    composed from both halves that records the two masks, the order and both runs."""
    setter = Setter()
    keys = SPLIT_KEYS[:2] + [TR.check_key(SPLIT_KEYS[0])]
    args, units, session, events = _split_session(tmp_path, monkeypatch, keys, setter)
    assert session.run(units) == 0
    names = [u.name for u in units]
    assert events == [("variance", "gpu", MASK),
                      (f"{names[0]}_gpu", "gpu", MASK), (f"{names[0]}_meep", "meep", FULL),
                      (f"{names[1]}_meep", "meep", FULL), (f"{names[1]}_gpu", "gpu", MASK),
                      (names[2], "gpu", MASK)]
    assert setter.calls == [MASK, MASK, MASK, FULL, FULL, MASK, MASK, FULL], \
        "set at the start, held before every run, restored to the earlier mask at the end"
    records = {r["key"]: r for r in TR.load_store(args.store)}
    assert sorted(records) == sorted(keys)
    for index, key in enumerate(SPLIT_KEYS[:2]):
        record = records[key]
        split = record["split"]
        order = ["gpu", "meep"] if index == 0 else ["meep", "gpu"]
        assert split["order"] == order and [h["side"] for h in split["halves"]] == order
        masks = {h["side"]: h["vm_affinity"] for h in split["halves"]}
        assert masks["gpu"] == {"declared": "0x999988", "before": "0xffffff" if index else
                                "0x999988", "set": "ok", "read_back": "0x999988",
                                "read": ["0x999988"]}
        assert masks["meep"]["declared"] == masks["meep"]["read_back"] == "0xffffff"
        assert masks["meep"]["read"] == ["0xffffff"] and masks["meep"]["set"] == "ok"
        for half in split["halves"]:
            assert half["rc"] == 0 and half["attempt"] == 1 and half["elapsed_s"] >= 0
            assert half["started_utc"] <= half["ended_utc"] and len(half["quiet_utc"]) == 2
            assert half["ladder"].endswith("_" + half["side"]) and half["watcher"]["readings"]
        assert [h["side"] for h in record["unit"]["halves"]] == order
        assert record["unit"]["argv"] == [] and all(
            h["argv"][-2:] == (["--ranks", "none"] if h["side"] == "gpu" else
                               ["--routes", "none"]) for h in record["unit"]["halves"])
        assert len(record["gates"]["quiet"]) == 4, "both halves' windows"
        assert record["gates"]["watcher"]["windows"]["vm_affinity"] == ["0x999988", "0xffffff"]
        assert record["gates"]["split_ladder"] == {
            "gpu": "0x999988", "meep": "0xffffff",
            "meep_source": "cleared: every one of Windows' 24 logical CPUs"}
        assert set(record["legs"]["gpu"]) == {"cuda"} and set(record["legs"]["meep"]) == {
            "meep@32"}
        assert record["environment"]["meep_builds"] == {"reference.json": hashlib.sha256(
            json.dumps({"half": "meep"}).encode()).hexdigest()}
        assert TR.refusals(record, allow_unrecorded_environment=True) == []
        assert_neutral(record["gates"])
    check = records[keys[2]]
    assert check["unit"]["side"] == "gpu" and check["unit"]["argv"][-2:] == ["--ranks", "none"]
    assert "split" not in check and check["legs"]["meep"] == {}
    assert TR.refusals(check, allow_unrecorded_environment=True) == []
    progress = open(os.path.join(args.run_root, "s1", "progress.log")).read()
    assert ("split ladder: GPU halves and every other unit under 0x999988, MEEP halves under "
            "0xffffff (cleared: every one of Windows' 24 logical CPUs)") in progress
    assert f"VM affinity for {names[0]}_meep: declared 0xffffff, before 0x999988, set ok, " \
           "read back 0xffffff" in progress
    assert "VM affinity restored: set ok, read back 0xffffff" in progress


def test_a_mask_moved_during_one_half_strikes_that_half_alone(machine, tmp_path, monkeypatch):
    """ONE watcher reading of the MEEP half that sees the GPU mask strikes the MEEP half:
    it is set aside and run again under its own mask; the GPU half is not run again, and
    the row records the clean attempt."""
    setter = Setter()
    state = {"once": None, "served": 0}

    def reader():
        if state["once"] is not None:
            observed, state["once"] = state["once"], None
            state["served"] += 1
            return observed
        return windows_observed(mask=setter.before)

    args, units, session, events = _split_session(tmp_path, monkeypatch, SPLIT_KEYS[:1],
                                                  setter, reader=reader,
                                                  no_variance="a split smoke")
    meep = units[0].children[1]
    original = session.runner

    def runner(unit, stopped):
        if unit is meep and not state["served"]:
            state["once"] = windows_observed(mask=MASK)
        return original(unit, stopped)

    session.runner = runner
    assert session.run(units) == 0 and state["served"] == 1
    assert [e[0] for e in events] == [units[0].children[0].name, meep.name, meep.name]
    aside = [p for p in os.listdir(os.path.dirname(meep.out)) if "intruded" in p]
    assert len(aside) == 1 and aside[0].startswith(os.path.basename(meep.out))
    progress = open(os.path.join(args.run_root, "s1", "progress.log")).read()
    assert "the WSL VM's CPU affinity is 0x999988, not the 0xffffff held for MEEP" in progress
    record, = TR.load_store(args.store)
    half = {h["side"]: h for h in record["split"]["halves"]}
    assert half["meep"]["attempt"] == 2 and half["gpu"]["attempt"] == 1
    assert half["meep"]["vm_affinity"]["read"] == ["0xffffff"]
    assert TR.refusals(record, allow_unrecorded_environment=True) == []


def test_a_half_that_fails_ends_the_unit_and_its_rows_are_refused(machine, tmp_path,
                                                                    monkeypatch):
    """A half that exits non-zero: the other half is not run (an earlier output of it is
    set aside), the unit is not marked done, and its rows say why they never enter."""
    setter = Setter()
    catalog = cert8.load_catalog(NVIDIA)
    unit, = cert8.build_units(args_for(tmp_path, split_ladder=True), catalog, SPLIT_KEYS[:1])
    first, second = unit.children
    os.makedirs(second.out)   # the second half's output of an earlier run
    args, units, session, events = _split_session(
        tmp_path, monkeypatch, SPLIT_KEYS[:1], setter, no_variance="a split smoke",
        fail={first.name: 3})
    assert session.run(units) == cert8.EXIT_UNIT
    assert [e[0] for e in events] == [first.name]
    assert not os.path.exists(second.out) and any(
        p.startswith(os.path.basename(second.out) + ".set_aside")
        for p in os.listdir(os.path.dirname(second.out)))
    assert not os.path.exists(session.marker(units[0]))
    record, = TR.load_store(args.store)
    assert record["split"]["not_run"] == {"side": "meep", "why": "the gpu half exited rc 3"}
    why = TR.refusals(record, allow_unrecorded_environment=True)
    assert any("the halves run were ['gpu'], not one GPU half and one MEEP half (the gpu "
               "half exited rc 3)" in reason for reason in why), why
    assert any("the ladder unit exited rc 3" in reason for reason in why)


@pytest.mark.parametrize("case", ["native", "no_mask", "malformed", "same_mask",
                                  "uncounted", "denied"])
def test_a_split_session_is_refused_by_name_before_it_times_anything(
        machine, tmp_path, monkeypatch, case):
    """``--split-ladder`` with no mask to split around (a native host, or none declared), a
    MEEP mask that does not parse, is the declared mask or cannot be derived (Windows'
    logical CPUs not counted), and a MEEP half whose mask does not read back: each is
    refused by name, and no half runs under the wrong mask."""
    setter, extra, reader = Setter(), {}, None
    if case == "malformed":
        extra["meep_vm_affinity"] = "0xZZ"
    elif case == "same_mask":
        extra["meep_vm_affinity"] = "0x999988"
    elif case == "uncounted":
        reader = lambda: windows_observed(load=None, mask=setter.before)  # noqa: E731
    args, units, session, events = _split_session(tmp_path, monkeypatch, SPLIT_KEYS[:1],
                                                  setter, reader=reader,
                                                  no_variance="a split smoke", **extra)
    if case in ("native", "no_mask"):
        rule = real_host.WindowsRule(1695, 9501) if case == "no_mask" else session.windows.rule
        session.windows = cert8.WindowsSide(
            WSL2 if case == "no_mask" else {"wsl2": False, "why": "native Linux"}, rule,
            reader=session.windows.reader, setter=setter)
    if case == "denied":
        allowed = setter

        def setter_denied(mask):
            return Setter(denied=True)(mask) if mask == FULL else allowed(mask)
        session.windows.setter = setter_denied
    with pytest.raises(cert8.Refused) as refusal:
        session.run(units)
    message = refusal.value.message
    named = {"native": "this session holds none: not a WSL2 host: native Linux. Run without "
                       "--split-ladder",
             "no_mask": "this session holds none: no VM affinity mask is declared",
             "malformed": "--meep-vm-affinity: an affinity mask is a positive integer",
             "same_mask": "the MEEP halves' mask 0x999988 is the declared VM affinity mask",
             "uncounted": "the first Windows reading did not count them",
             "denied": "could not be held at 0xffffff for " + units[0].children[1].name}[case]
    assert named in message, message
    if case == "denied":
        assert [e[1] for e in events] == ["gpu"], "the GPU half ran; no MEEP under any mask"
    else:
        assert events == [] and setter.calls == [], setter.calls
    assert TR.load_store(args.store) == []


def test_a_declared_meep_mask_holds_the_meep_halves(machine, tmp_path, monkeypatch):
    """``--meep-vm-affinity``: the MEEP half runs under the mask declared for it, set, read
    back and recorded as declared, and the session still restores the earlier mask."""
    setter = Setter()
    args, units, session, events = _split_session(tmp_path, monkeypatch, SPLIT_KEYS[:1],
                                                  setter, no_variance="a split smoke",
                                                  meep_vm_affinity="0xfffff0")
    assert session.run(units) == 0
    assert [(e[1], e[2]) for e in events] == [("gpu", MASK), ("meep", 0xFFFFF0)]
    assert setter.calls == [MASK, MASK, 0xFFFFF0, FULL]
    record, = TR.load_store(args.store)
    assert record["gates"]["split_ladder"] == {"gpu": "0x999988", "meep": "0xfffff0",
                                               "meep_source": "declared with --meep-vm-affinity"}
    mask = {h["side"]: h["vm_affinity"] for h in record["split"]["halves"]}["meep"]
    assert (mask["declared"], mask["set"], mask["read_back"], mask["read"]) == (
        "0xfffff0", "ok", "0xfffff0", ["0xfffff0"])
    assert TR.refusals(record, allow_unrecorded_environment=True) == []


def test_meep_under_a_mask_is_admitted_only_as_a_split_half(machine, tmp_path, monkeypatch):
    """Under a declared mask a whole ladder unit and a whole dispatch check of a ladder row
    are refused (as before), and a split unit, its halves and a split session's check are
    admitted; a split unit is never run as one process."""
    setter = Setter()
    keys = SPLIT_KEYS[:1] + [TR.check_key(SPLIT_KEYS[0])]
    args, units, session, events = _split_session(tmp_path, monkeypatch, keys, setter,
                                                  no_variance="a split smoke")
    session.refuse_meep(units + units[0].children)
    with pytest.raises(cert8.Refused) as refusal:
        session.run_unit(units[0])
    assert "is a split ladder unit: its halves run one by one" in refusal.value.message
    whole = cert8.build_units(args_for(tmp_path / "w"), cert8.load_catalog(NVIDIA), keys)
    session.split = False
    for unit in whole + units[0].children[1:]:
        with pytest.raises(cert8.Refused) as refusal:
            session.refuse_meep([unit])
        assert "or split each ladder unit into a GPU half under the mask and a MEEP half " \
               "without it (--split-ladder)" in refusal.value.message
    session.refuse_meep(units[0].children[:1] + units[1:])
    assert events == [] and setter.calls == []


def test_a_half_records_every_mask_its_readings_saw(machine, tmp_path, monkeypatch):
    """The masks a half records as read are its quiet window's and every watcher reading's,
    so a row whose half saw a mask other than its own is refused by name."""
    setter = Setter()
    args, units, session, events = _split_session(tmp_path, monkeypatch, SPLIT_KEYS[:1],
                                                  setter, no_variance="a split smoke")
    original = session.run_unit

    def run_unit(child):
        rc, watch, quiet = original(child)
        if child.side == "meep":
            watch = dict(watch, windows=dict(watch["windows"],
                                             vm_affinity=["0x999988", "0xffffff"]))
        return rc, watch, quiet

    session.run_unit = run_unit
    assert session.run(units) == 0
    record, = TR.load_store(args.store)
    half = {h["side"]: h for h in record["split"]["halves"]}
    assert half["meep"]["vm_affinity"]["read"] == ["0x999988", "0xffffff"]
    assert half["gpu"]["vm_affinity"]["read"] == ["0x999988"]
    assert any("the readings of the meep half saw the VM affinity ['0x999988', '0xffffff']"
               in why for why in TR.refusals(record, allow_unrecorded_environment=True))


def test_merged_windows_summaries_add_counts_join_values_and_keep_maxima():
    one = {"readings": 3, "unreadable": 0, "graphics_mhz": [1695], "vm_affinity": ["0x999988"],
           "power_w_max": 210.0, "outside_vm_busy_cpus_max": None, "busy_cpus_max": 1.2}
    two = {"readings": 2, "unreadable": 1, "graphics_mhz": [1695], "vm_affinity": ["0xffffff"],
           "power_w_max": 90.0, "outside_vm_busy_cpus_max": 0.4, "busy_cpus_max": None}
    assert cert8.merge_windows_summaries([one, two]) == {
        "readings": 5, "unreadable": 1, "graphics_mhz": [1695],
        "vm_affinity": ["0x999988", "0xffffff"], "power_w_max": 210.0,
        "outside_vm_busy_cpus_max": 0.4, "busy_cpus_max": 1.2}


# ---------------------------------------------------------------------------
# The variance precondition under a declared mask: the GPU route alone
# ---------------------------------------------------------------------------

def test_under_a_declared_mask_the_variance_precondition_times_the_gpu_route_alone(
        machine, tmp_path, monkeypatch):
    """The precondition gates the GPU route's repeat spread alone, and MEEP is never timed
    under a declared mask: in a session that declares one, split or not, it is a GPU half
    (``--ranks none``, run under the mask) and records no MEEP side. Without a mask it is
    the unit it was, MEEP at the catalog's rank count included."""
    monkeypatch.setattr(real_host, "topology", lambda *a, **k: {"physical_cores": 48})
    catalog = cert8.load_catalog(NVIDIA)
    whole = cert8.variance_unit(args_for(tmp_path), catalog)
    for split in (cert8.variance_unit(args_for(tmp_path, split_ladder=True, ladder_params=None),
                                      catalog),
                  cert8.variance_unit(args_for(tmp_path), catalog, masked=True)):
        assert split.side == "gpu" and split.argv[-6:] == ["--routes", "default", "--ranks",
                                                           "none", "--controls", "none"]
        assert split.argv[:-6] == whole.argv[:-6] and split.out == whole.out
        assert split.describe()["side"] == "gpu"
    assert whole.side is None and whole.argv[-6:] == ["--routes", "default", "--ranks",
                                                      "32||", "--controls", "none"]
    assert "side" not in whole.describe()
    setter = Setter()
    args, units, session, events = _split_session(tmp_path / "s", monkeypatch, SPLIT_KEYS[:1],
                                                  setter)
    assert session.run(units) == 0
    assert events[0] == ("variance", "gpu", MASK), "a GPU half, under the declared mask"
    stored = json.load(open(os.path.join(args.run_root, "s1", "variance.json")))
    assert stored["verdict"] == "GO" and set(stored["sides"]) == {"gpu"}
    assert stored["meep_ranks"] is None
    assert stored["meep_side"].startswith("not run: under a declared VM affinity mask the "
                                          "precondition times the GPU route alone")
    record, = TR.load_store(args.store)
    assert record["gates"]["variance"]["verdict"] == "GO"
    assert set(record["gates"]["variance"]["sides"]) == {"gpu"}
    assert TR.refusals(record, allow_unrecorded_environment=True) == []


@pytest.mark.parametrize("split", [True, False])
def test_a_precondition_that_would_time_meep_under_a_mask_is_refused_by_name(
        machine, tmp_path, monkeypatch, split):
    """A session under a declared mask, split or not, whose precondition is not a GPU half
    would time MEEP under the mask: it is refused by name before anything is set or run."""
    setter = Setter()
    fused = TR.fused_key("8.6", "triton", "pml_2d", "res20", 1)
    args, units, session, events = (
        _split_session(tmp_path, monkeypatch, SPLIT_KEYS[:1], setter) if split else
        _mask_session(tmp_path, monkeypatch, [fused], setter))
    whole = cert8.variance_unit(args_for(tmp_path / "w"), session.catalog)
    monkeypatch.setattr(cert8, "variance_unit", lambda a, c, masked=False: whole)
    with pytest.raises(cert8.Refused) as refusal:
        session.variance_precondition()
    assert "the variance precondition (variance) would time MEEP under the declared VM " \
           "affinity mask 0x999988" in refusal.value.message
    assert events == [] and setter.calls == []


# ---------------------------------------------------------------------------
# The host's MEEP rank ceiling: the MEEP half of a split session
# ---------------------------------------------------------------------------

def test_a_split_meep_half_is_held_to_the_rank_blocks_its_site_declares(tmp_path):
    """The MEEP half of a split unit is handed the largest rank count of the rank blocks
    the session's site file declares (8 on the 8-performance-core desktop, RANKS=8|1|), as
    ``--meep-rank-ceiling``; the GPU half, the dispatch check, a session without the split
    and a split session of fused rows alone are not, and need no rank blocks."""
    catalog = cert8.load_catalog(NVIDIA)
    keys = SPLIT_KEYS[:1] + [TR.check_key(SPLIT_KEYS[0])]
    site = site_params(tmp_path / "desk", "4,8|1|")
    unit, check = cert8.build_units(args_for(tmp_path, split_ladder=True, ladder_params=site),
                                    catalog, keys)
    halves = {child.side: child.argv for child in unit.children}
    assert halves["meep"][-4:] == ["--meep-rank-ceiling", "8", "--routes", "none"]
    assert "--meep-rank-ceiling" not in halves["gpu"] and "--meep-rank-ceiling" not in check.argv
    assert halves["meep"][halves["meep"].index("--params") + 1] == site
    whole = cert8.build_units(args_for(tmp_path, ladder_params=site), catalog, keys)
    assert not any("--meep-rank-ceiling" in u.argv for u in whole)
    fused = TR.fused_key("8.6", "triton", "pml_2d", "res20", 1)
    alone, = cert8.build_units(args_for(tmp_path, split_ladder=True, ladder_params=None),
                               catalog, [fused])
    assert alone.kind == "fused" and "--meep-rank-ceiling" not in alone.argv


@pytest.mark.parametrize("site, named", [
    ("none-given", "and no --ladder-params was given"),
    (None, "its site file site.params declares none (RANKS=8|1| on an 8-performance-core "
           "desktop)"),
    ("none", "its site file site.params declares none"),
    ("8|1", "three blocks A|B|C are needed"),
])
def test_a_split_session_whose_site_declares_no_rank_blocks_is_refused_by_name(
        tmp_path, site, named):
    """A split session's MEEP halves are held to the rank blocks its site file declares;
    with none declared a ladder would derive them from the cores it detects (12 under WSL2
    on the 8-performance-core desktop), so the session is refused by name, in plan and in
    run alike, before anything is set."""
    params = None if site == "none-given" else site_params(tmp_path / "s", site)
    with pytest.raises(cert8.Refused) as refusal:
        cert8.build_units(args_for(tmp_path, split_ladder=True, ladder_params=params),
                          cert8.load_catalog(NVIDIA), SPLIT_KEYS[:1])
    assert named in refusal.value.message, refusal.value.message


# ---------------------------------------------------------------------------
# What a row record names: no machine's own path
# ---------------------------------------------------------------------------

#: Another machine's home, as the lab's build record names it (from pieces: a full home path
#: in this file is a hygiene-guard hit).
LAB_HOME = "/" + "ho" + "me" + "/labaccount"


def lab_build_record(path):
    """A reference environment's build record as the lab desktop wrote it: its prefix under
    that machine's home in a dozen places, a map keyed by image path, a system library, and
    a URL."""
    prefix = f"{LAB_HOME}/miniforge3/envs/ref"
    image = f"{prefix}/lib/libmeep.so.39.0.0"
    record = {"configure_line": f"./configure --prefix={prefix} "
                                f"--with-libctl={prefix}/share/libctl CC={prefix}/bin/cc",
              "config_log": f"{prefix}/share/meep-gpu-build/config.log",
              "makefile_flags": {"CPPFLAGS": f"-I{prefix}/include",
                                 "LDFLAGS": f"-L{prefix}/lib -Wl,-rpath,{prefix}/lib"},
              "python": f"{prefix}/bin/python", "launcher": f"{prefix}/bin/mpirun",
              "launcher_version": "mpirun (Open MPI) 5.0.8\n\nReport bugs to "
                                  "https://www.open-mpi.org/community/help/\n",
              "sha256": {image: "ab" * 32}, "openmp_linked": {image: {"runtimes": ["libgomp"]}},
              "probe": {"prefix": prefix, "libmeep_present": [image]},
              "fma_census": {"tool": {"path": f"{LAB_HOME}/meep-gpu/tools/fma_census.py",
                                      "sha256": "cd" * 32}},
              "libm": {"path": "/usr/lib/x86_64-linux-gnu/libm.so.6", "sha256": "ef" * 32},
              "fp_contract": "off", "openmp": True, "mpi_implementation": "openmpi"}
    path.write_text(json.dumps(record, indent=1))
    return str(path), record


@pytest.fixture
def home_at_tmp(monkeypatch, tmp_path):
    """This machine, its home moved to the test's directory: the session's run root and
    store then lie under a root ``neutral_ids`` labels on any platform. The temporary root
    a test directory is made under is labelled by its shape on macOS and not on a Linux
    runner, where a record written under it would name the runner's temporary directory."""
    monkeypatch.setenv("HOME", str(tmp_path))
    neutral_ids.reset()
    yield tmp_path
    neutral_ids.reset()


def test_a_row_record_names_no_local_path_and_keeps_its_build_facts(home_at_tmp, tmp_path,
                                                                    monkeypatch):
    """Every record a session writes, a split ladder row and its dispatch check included,
    carries the reference environment's build record with every path in it under its
    root's label: no absolute path outside the system directories, no home directory. The
    digest is over the file as written, and the facts stay. The hygiene guard reads the
    store clean. This machine is itself here (no fixture machine): the guard reads its
    real account, which the records must not name either."""
    import importlib.util  # noqa: PLC0415

    environment, raw = lab_build_record(tmp_path / "build_record.json")
    setter = Setter()
    keys = SPLIT_KEYS[:1] + [TR.check_key(SPLIT_KEYS[0])]
    args, units, session, events = _split_session(tmp_path, monkeypatch, keys, setter,
                                                  environment_record=environment)
    assert session.run(units) == 0
    records = TR.load_store(args.store)
    assert len(records) == 2
    for record in records:
        assert TR.local_paths(TR.public(record)) == [], record["key"]
        reference = record["environment"]["reference"]
        assert reference["sha256"] == TR.file_sha256(environment)
        facts = reference["record"]
        assert (facts["fp_contract"], facts["openmp"], facts["mpi_implementation"]) == (
            "off", True, "openmpi")
        assert facts["sha256"] == {"<home>/miniforge3/envs/ref/lib/libmeep.so.39.0.0": "ab" * 32}
        assert facts["makefile_flags"]["CPPFLAGS"] == "-I<home>/miniforge3/envs/ref/include"
        assert facts["libm"] == raw["libm"] and facts["launcher_version"] == raw[
            "launcher_version"], "a system library and a URL are no machine's own path"
        assert "labaccount" not in json.dumps(record)
    verdict = json.load(open(os.path.join(args.run_root, "s1", "variance.json")))
    assert TR.local_paths(verdict) == [], "the session's variance record leaves the machine too"
    spec = importlib.util.spec_from_file_location(
        "check_hygiene_records", REPO_ROOT / "tools" / "ci" / "check_hygiene.py")
    guard = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, guard)
    spec.loader.exec_module(guard)
    assert guard.main(["--legs", "records", "--records", args.store]) == 0


@pytest.mark.parametrize("prefix, account", [
    ("/scratch/proj42/envs/ref", False),                 # a site's scratch filesystem
    ("/project/projectdirs/m4242/alice/envs/ref", True),  # a group's project directory
    ("/mnt/d/envs/ref", False),                          # another drive under WSL2
    ("/data/alice/envs/ref", True),                      # a data disk
])
def test_a_build_record_under_no_labelled_root_names_no_path(monkeypatch, tmp_path, prefix,
                                                             account):
    """A reference environment that lives under no root ``neutral_ids`` labels (an HPC
    site's scratch or project filesystem, another drive, a data disk): every such path is
    written as ``<path>/`` and its last component, the account inside it never written,
    and the facts, the system library and the URL kept. The digest is over the file."""
    monkeypatch.setattr(neutral_ids, "login", lambda: "alice")
    monkeypatch.setattr(neutral_ids, "host_names", lambda: ["nid001234"])
    neutral_ids.reset()
    image = f"{prefix}/lib/libmeep.so.39.0.0"
    raw = {"python": f"{prefix}/bin/python",
           "configure_line": f"./configure --prefix={prefix} CC={prefix}/bin/cc",
           "makefile_flags": {"LDFLAGS": f"-L{prefix}/lib -Wl,-rpath,{prefix}/lib"},
           "sha256": {image: "ab" * 32, f"{prefix}/lib/python3.12/meep/_meep.so": "cd" * 32},
           "libm": {"path": "/usr/lib/x86_64-linux-gnu/libm.so.6"},
           "launcher_version": "Report bugs to https://www.open-mpi.org/community/help/",
           "fp_contract": "off"}
    path = tmp_path / "build_record.json"
    path.write_text(json.dumps(raw))
    block = cert8.environment_block(str(path))
    neutral_ids.reset()
    facts = block["reference"]["record"]
    assert TR.local_paths(facts) == [] and "alice" not in json.dumps(facts)
    assert facts["python"] == "<path>/python"
    assert facts["configure_line"] == "./configure --prefix=<path>/ref CC=<path>/cc"
    assert facts["makefile_flags"]["LDFLAGS"] == "-L<path>/lib -Wl,-rpath,<path>/lib"
    assert facts["sha256"] == {"<path>/libmeep.so.39.0.0": "ab" * 32,
                               "<path>/_meep.so": "cd" * 32}
    assert (facts["libm"], facts["launcher_version"], facts["fp_contract"]) == (
        raw["libm"], raw["launcher_version"], "off")
    assert block["reference"]["sha256"] == TR.file_sha256(str(path))


def test_a_build_record_whose_keys_would_merge_is_refused_by_name(monkeypatch, tmp_path):
    """Two images at one name under two roots would become one key once written under their
    labels, and one digest would be lost: refused by name, by the session's own reading of
    the build record before anything runs (``main``) and at collect alike."""
    monkeypatch.setattr(neutral_ids, "host_names", lambda: [])
    neutral_ids.reset()
    raw = {"sha256": {"/scratch/a/lib/libmeep.so": "ab" * 32,
                      "/scratch/b/lib/libmeep.so": "cd" * 32}}
    path = tmp_path / "build_record.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(cert8.Refused) as refusal:
        cert8.environment_block(str(path))
    neutral_ids.reset()
    assert "holds 3 keys and 2 once its paths are written under their labels" in \
        refusal.value.message
    assert "/scratch" not in refusal.value.message


def test_a_build_record_naming_a_path_no_label_can_write_is_refused_by_name(monkeypatch,
                                                                          tmp_path):
    """A path the writer has no label for (one under ``~/``) would leave the machine as it
    is: the build record is refused by name, the path not repeated."""
    monkeypatch.setattr(neutral_ids, "host_names", lambda: [])
    neutral_ids.reset()
    path = tmp_path / "build_record.json"
    path.write_text(json.dumps({"config_log": "~/meep-1.35.0/config.log", "openmp": True}))
    with pytest.raises(cert8.Refused) as refusal:
        cert8.environment_block(str(path))
    neutral_ids.reset()
    assert "still names 1 local path(s) once its paths are written under their labels" in \
        refusal.value.message
    assert "meep-1.35.0" not in refusal.value.message


def test_a_run_refuses_a_build_record_it_cannot_write_before_anything_runs(
        monkeypatch, tmp_path, capsys):
    """``cert8.py run`` reads the reference build record once before the session starts: one
    every row would fail to write is refused by name before the toolchain is exported, a
    mask is set or a unit runs."""
    path = tmp_path / "build_record.json"
    path.write_text(json.dumps({"sha256": {"/scratch/a/lib/libmeep.so": "ab" * 32,
                                           "/scratch/b/lib/libmeep.so": "cd" * 32}}))
    started = []
    monkeypatch.setattr(cert8, "export_launcher_toolchain",
                        lambda python: started.append("toolchain"))
    monkeypatch.setattr(cert8, "Session", lambda *a, **k: started.append("session"))
    code = cert8.main(["run", "--catalog", NVIDIA, "--store", str(tmp_path / "store"),
                       "--run-root", str(tmp_path / "runs"), "--session", "s1",
                       "--python", sys.executable, "--gpu", "0",
                       "--environment-record", str(path), "--rows", SPLIT_KEYS[0]])
    assert code == cert8.EXIT_REFUSED and started == []
    assert "REFUSED the reference environment's build record holds 3 keys and 2" in \
        capsys.readouterr().out


def test_a_build_record_path_continuing_a_label_is_written_as_it_continues(home_at_tmp,
                                                                          tmp_path):
    """A configure line that doubles a slash after its home (``$HOME//miniforge3``, as a
    GPU host's did) is the home's path continued: written under the home's label, never
    as a path under no root."""
    path = tmp_path / "build_record.json"
    path.write_text(json.dumps({"configure_line": f"./configure --prefix={LAB_HOME}//"
                                                  "miniforge3/envs/ref --enable-single"}))
    facts = cert8.environment_block(str(path))["reference"]["record"]
    assert facts["configure_line"] == (
        "./configure --prefix=<home>//miniforge3/envs/ref --enable-single")
