"""The timing ladder's plan, built without a GPU, MEEP or a process.

The central check: under the A6000 host's profile (``fixtures/timing/
a6000_2026-09-28_b.params``) the ladder plans the 111 rows of the 2026-09-28 ladder
in the same order, under the same keys, with the same commands, working directories
and environments as ``fixtures/timing/ladder_2026-09-28_b_commands.json`` records
they ran with, up to the equivalence ``docs/development/timing.md`` states: paths,
stamp and times compared through placeholders, the box string by its key set (with
``busy_cores`` renamed ``busy_cpus``), and the named differences of the environment.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import timing_host  # noqa: E402
import timing_ladder as ladder  # noqa: E402

FIXTURES = os.path.join(HERE, "fixtures", "timing")
PARAMS = os.path.join(FIXTURES, "a6000_2026-09-28_b.params")
COMMANDS = os.path.join(FIXTURES, "ladder_2026-09-28_b_commands.json")
#: The launching environment the comparison is made under: a shell that exports
#: KMP_DUPLICATE_LIB_OK, as the 2026-09-28 reproduction's does (named difference 7).
LAUNCHING_ENV = {"PATH": "/usr/bin:/bin", "HOME": "/home/user",
                 "KMP_DUPLICATE_LIB_OK": "TRUE"}
#: Variables the released ladder unsets that the 2026-09-28 ladder did not (named
#: differences: the uncertified opt-in is never inherited; a MEEP row no longer
#: inherits a GPU row's caches or an OpenMP placement).
ADDED_UNSETS = {"MEEP_GPU_ALLOW_UNCERTIFIED", "TRITON_CACHE_DIR", "CUPY_CACHE_DIR",
                "OMP_PROC_BIND", "OMP_PLACES"}
#: Variables the released ladder SETS that the 2026-09-28 ladder left unset (named
#: difference: an NVIDIA row restricts the kernels to certified identities, which is
#: what unset meant on NVIDIA before supported-uncertified devices ran by default).
ADDED_SETS = {"MEEP_GPU_ALLOW_UNCERTIFIED": "0"}


def config_of(argv, environ=None):
    args = ladder.build_parser().parse_args(argv)
    return ladder.build_config(args, dict(LAUNCHING_ENV if environ is None else environ))


def plan_of(argv, environ=None):
    config = config_of(argv, environ)
    return config, ladder.plan_record(config, ladder.build_plan(config))


def refusal(argv, environ=None) -> str:
    with pytest.raises(SystemExit) as caught:
        config = config_of(argv, environ)
        ladder.build_plan(config)
    return str(caught.value)


A6000_PROFILE = ["--params", PARAMS]
MAC = ["--stamp", "t", "--python", "/opt/env/bin/python", "--meep-config-log", "none",
       "--platform", "darwin", "--physical-cores", "8", "--logical-cpus", "10",
       "--memory-total-gb", "32", "--mpi-impl", "openmpi", "--tree-layout", "release",
       "--tree-src", "/opt/meep-gpu", "--run-root", "/opt/runs"]
LINUX = ["--stamp", "t", "--python", "/opt/env/bin/python", "--meep-config-log", "none",
         "--platform", "linux", "--physical-cores", "48", "--logical-cpus", "96",
         "--memory-total-gb", "1000", "--mpi-impl", "openmpi", "--tree-layout", "release",
         "--tree-src", "/opt/meep-gpu", "--run-root", "/opt/runs", "--gpu", "0"]


# ---------------------------------------------------------------------------
# The 2026-09-28 reproduction
# ---------------------------------------------------------------------------

def _placeholders(record):
    resolved = record["resolved"]
    out = resolved["out"]
    run_root = os.path.dirname(out)
    stamp = os.path.basename(out)[len("ladder_"):]
    python = record["params"]["PYTHON"]["value"]
    return [(resolved["harness"], "<HARNESS>"), (resolved["api"], "<API>"),
            (resolved["drivers"], "<DRV>"), (python, "<PY>"),
            (resolved["launcher"], "<MPIRUN>"), (run_root, "<RUN_ROOT>"),
            (stamp, "<STAMP>")]


def _to_placeholders(text, pairs):
    for value, name in pairs:
        text = text.replace(value, name)
    return text


def _effective(delta, base):
    env = dict(base)
    for name in delta["unset"]:
        env.pop(name, None)
    env.update(delta["set"])
    if "LD_LIBRARY_PATH" in env:
        env["LD_LIBRARY_PATH"] = env["LD_LIBRARY_PATH"].rstrip(":")
    return env


def test_the_a6000_profile_plans_the_111_rows_of_2026_09_28():
    with open(COMMANDS, "r", encoding="utf-8") as handle:
        fixture = json.load(handle)
    config, record = plan_of(A6000_PROFILE)
    pairs = _placeholders(record)
    rows = record["rows"]
    assert len(rows) == len(fixture["rows"]) == 111
    assert record["resolved"]["rank_blocks"] == {"A": [16, 24, 32, 48], "B": [8, 1],
                                                 "C": [64]}
    base_old = dict(LAUNCHING_ENV, **fixture["process_exports"])
    base_new = ladder.process_environment(dict(LAUNCHING_ENV))
    new_keys = record["resolved"]["box_keys"]
    mismatches = []
    compared = 0
    for ours, theirs in zip(rows, fixture["rows"]):
        where = f"row {theirs['index']} {theirs['key']}"
        if (ours["index"], ours["kind"], ours["tier"]) != (
                theirs["index"], theirs["kind"], theirs["tier"]):
            mismatches.append(f"{where}: planned {ours['kind']}/{ours['tier']}")
            continue
        if ours["key"] != theirs["key"]:
            mismatches.append(f"{where}: key {ours['key']}")
        if theirs["argv"] is None:
            assert ours["argv"] is None, where
            if theirs.get("reads"):
                assert [_to_placeholders(p, pairs) for p in ours["reads"]] == \
                    theirs["reads"], where
            continue
        argv = [_to_placeholders(a, pairs) for a in ours["argv"]]
        compared += 1
        if argv != theirs["argv"]:
            diff = [(i, a, b) for i, (a, b) in enumerate(zip(argv, theirs["argv"]))
                    if a != b][:3]
            mismatches.append(f"{where}: argv differs {diff} (lengths {len(argv)}, "
                              f"{len(theirs['argv'])})")
        if _to_placeholders(ours["cwd"], pairs) != theirs["cwd"]:
            mismatches.append(f"{where}: cwd {ours['cwd']}")
        old = fixture["env"][theirs["env"]]
        new = {"set": {k: _to_placeholders(v, pairs) for k, v in ours["env"]["set"].items()},
               "unset": ours["env"]["unset"]}
        old_set = {k: v.replace("<BASE_PYTHONPATH>", "").replace(
            "<BASE_LD_LIBRARY_PATH>", "") for k, v in old["set"].items()}
        effective_new = _effective(new, base_new)
        for name, value in ADDED_SETS.items():
            if effective_new.get(name) == value:
                effective_new.pop(name)
        if effective_new != _effective(
                {"set": old_set, "unset": old["unset"]}, base_old):
            mismatches.append(f"{where}: environment differs")
        extra = set(new["unset"]) - set(old["unset"])
        if not extra <= ADDED_UNSETS:
            mismatches.append(f"{where}: unsets {sorted(extra - ADDED_UNSETS)} beyond the "
                              "named differences")
        if theirs.get("expected_launched"):
            assert ours["expected_launched"] == theirs["expected_launched"], where
        if theirs.get("box_keys"):
            renamed = ["busy_cpus" if k == "busy_cores" else k for k in theirs["box_keys"]]
            assert renamed == new_keys, (where, renamed, new_keys)
    assert not mismatches, "\n".join(mismatches[:20])
    assert compared == 101  # 20 GPU rows, 80 MEEP rows and the preflight digest


def test_the_a6000_profile_budget_is_the_2026_09_28_budget():
    _config, record = plan_of(A6000_PROFILE)
    tiers = record["budget_by_tier"]
    assert [(t, tiers[t]["rows"], tiers[t]["row_s"], tiers[t]["gate_s"]) for t in tiers] == [
        ("preflight", 6, 427, 474), ("bindprobe", 9, 193, 304),
        ("priority", 90, 10728, 5634), ("bindcheck", 6, 155, 228)]


def test_new_options_are_absent_at_their_defaults():
    _config, record = plan_of(A6000_PROFILE)
    new_flags = {"--build-label", "--build-record", "--threads-per-rank",
                 "--split-chunks-evenly", "--digest-file"}
    for row in record["rows"]:
        assert not new_flags & set(row["argv"] or []), row["key"]


# ---------------------------------------------------------------------------
# Ranks, keys, cases, controls
# ---------------------------------------------------------------------------

def test_default_rank_blocks():
    assert ladder.default_rank_blocks(48, 96, True) == ([16, 24, 32, 48], [8, 1], [64])
    assert ladder.default_rank_blocks(8, 10, False) == ([3, 4, 5, 8], [1], [10])
    assert ladder.default_rank_blocks(8, 8, False) == ([3, 4, 5, 8], [1], [])


@pytest.mark.parametrize("allow,nvidia,metal", [("0", "0", None), ("1", "1", "1")])
def test_an_nvidia_row_runs_certified_identities_only_unless_asked(allow, nvidia, metal):
    """``0`` exports MEEP_GPU_ALLOW_UNCERTIFIED=0 to the NVIDIA rows and leaves the
    Metal rows the Metal default; ``1`` exports 1 to both; no MEEP row inherits it."""
    _config, record = plan_of(A6000_PROFILE + ["--allow-uncertified", allow])
    gpu = [r for r in record["rows"] if r["kind"] == "gpu"]
    assert gpu and all(r["env"]["set"].get("MEEP_GPU_ALLOW_UNCERTIFIED") == nvidia
                       for r in gpu), [r["key"] for r in gpu]
    _config, record = plan_of(MAC + ["--allow-uncertified", allow])
    gpu = [r for r in record["rows"] if r["kind"] == "gpu"]
    assert gpu and all(r["env"]["set"].get("MEEP_GPU_ALLOW_UNCERTIFIED") == metal
                       for r in gpu), [r["key"] for r in gpu]
    if metal is None:
        assert all("MEEP_GPU_ALLOW_UNCERTIFIED" in r["env"]["unset"] for r in gpu)
    meep = [r for r in record["rows"] if r["kind"] == "meep"]
    assert meep and all("MEEP_GPU_ALLOW_UNCERTIFIED" in r["env"]["unset"] for r in meep)


def test_mac_profile_plans_unbound_rows_and_omits_bound_controls():
    config, record = plan_of(MAC)
    resolved = record["resolved"]
    assert resolved["rank_blocks"] == {"A": [3, 4, 5, 8], "B": [1], "C": [10]}
    assert resolved["routes"] == ["metal"]
    assert set(resolved["controls_omitted"]) == {"native", "threads2", "threads4",
                                                 "hwthread"}
    assert resolved["controls"] == ["split"]
    meep = [r for r in record["rows"] if r["kind"] == "meep"]
    assert meep and all(r["binding"] == ("unbound-ecores" if r["ranks"] > 8 else "unbound")
                        for r in meep)
    assert all("--bind-to" in r["argv"] and r["argv"][r["argv"].index("--bind-to") + 1]
               == "none" for r in meep)
    assert not [r for r in record["rows"] if r["tier"] in ("bindprobe", "bindcheck")]
    split = [r for r in meep if r["control"] == "split"]
    assert split and all("--split-chunks-evenly" in r["argv"] for r in split)
    assert all(r["out"].endswith("meep_splitcost") for r in split)


def test_key_carries_the_case_only_when_several_are_planned():
    _c, one = plan_of(LINUX + ["--tiers", "priority", "--passes", "1", "--controls",
                               "none", "--priority-sizes", "512000"])
    _c, two = plan_of(LINUX + ["--tiers", "priority", "--passes", "1", "--controls",
                               "none", "--priority-sizes", "512000", "--cases",
                               "pml_3d,thin_pml_3d"])
    assert one["rows"][0]["key"] == "priority_gpu_default_res20_p1"
    keys = [r["key"] for r in two["rows"]]
    assert "priority_pml_3d_gpu_default_res20_p1" in keys
    assert "priority_thin_pml_3d_gpu_default_res10_p1" in keys
    meep = next(r for r in two["rows"] if r["kind"] == "meep" and r["case"] == "thin_pml_3d")
    assert "--row-id" in meep["argv"]
    assert meep["argv"][meep["argv"].index("--row-id") + 1].startswith(
        "priority_thin_pml_3d_meep_r16_")


def test_a_new_case_runs_through_the_wrapper_and_a_builtin_does_not():
    config, record = plan_of(LINUX + ["--tiers", "priority", "--passes", "1",
                                      "--controls", "none", "--priority-sizes", "512000",
                                      "--cases", "pml_3d,thin_pml_3d_hz"])
    gpu = [r for r in record["rows"] if r["kind"] == "gpu"]
    builtin = [r for r in gpu if r["case"] == "pml_3d"]
    new = [r for r in gpu if r["case"] == "thin_pml_3d_hz"]
    assert all(r["argv"][2] == "parity/meep_gpu/bench_fused_products.py" for r in builtin)
    assert all(r["argv"][2].endswith("bench_timing_case.py") for r in new)
    assert all(r["expected_repair"] == "B" for r in new)
    assert all(r["res"] == 10 for r in new)


def test_linux_controls_carry_their_configuration():
    environ = dict(LAUNCHING_ENV)
    config, record = plan_of(LINUX + ["--tiers", "priority", "--passes", "1",
                                      "--priority-sizes", "512000", "--meep-build",
                                      "native=/opt/native/bin/python"], environ)
    controls = {}
    for row in record["rows"]:
        if row["kind"] == "meep" and row["control"]:
            controls.setdefault(row["control"], []).append(row)
    assert {c: sorted(r["ranks"] for r in rows) for c, rows in controls.items()} == {
        "native": [32, 48], "threads2": [24], "threads4": [12], "hwthread": [64, 96],
        "split": [32, 48]}
    t2 = controls["threads2"][0]
    assert t2["binding"] == "bound-core-by-package-pe2"
    assert "--map-by" in t2["argv"] and "package:PE=2" in t2["argv"]
    assert t2["argv"][t2["argv"].index("--threads-per-rank") + 1] == "2"
    assert t2["env"]["set"]["OMP_NUM_THREADS"] == "2"
    assert t2["env"]["set"]["OMP_PROC_BIND"] == "close"
    native = controls["native"][0]
    assert native["argv"][0] == "/opt/native/bin/mpirun"
    assert "/opt/native/bin/python" in native["argv"]
    assert native["argv"][native["argv"].index("--build-label") + 1] == "native"
    assert controls["hwthread"][0]["binding"] == "bound-hwthread"
    outs = {r["out"] for rows in controls.values() for r in rows}
    assert all(os.path.basename(o).startswith("meep_") for o in outs)


def test_digests_tier_lists_every_planned_case_and_size_once():
    _c, record = plan_of(LINUX + ["--tiers", "digests priority rest", "--passes", "1",
                                  "--controls", "none", "--priority-sizes", "512000",
                                  "--rest-sizes", "512000,110592", "--cases",
                                  "pml_3d,thin_pml_3d"])
    lifts = [(r["case"], r["cells"]) for r in record["rows"] if r["kind"] == "lift"]
    assert lifts == [("pml_3d", 512000), ("thin_pml_3d", 512000), ("pml_3d", 110592),
                     ("thin_pml_3d", 110592)]
    lift = next(r for r in record["rows"] if r["kind"] == "lift")
    assert "--digest-file" in lift["argv"]
    assert lift["argv"][lift["argv"].index("--digest-file") + 1].endswith(
        "digests/pml_3d_res20.sha256")


def test_srun_launcher_and_its_flags(monkeypatch):
    # The refusal is the one a host whose socket count cannot be read gets. The ladder
    # reads the count whenever the platform asked is this host's, so on a Linux host
    # the plan would carry that host's sockets: the topology is withheld here.
    monkeypatch.setattr(ladder, "detect_topology", lambda _p: {"method": None})
    srun = LINUX + ["--tiers", "priority", "--passes", "1", "--controls", "none",
                    "--priority-sizes", "512000", "--meep-launcher", "srun",
                    "--srun-mpi", "pmix", "--routes", "none"]
    assert "pass --sockets" in refusal(srun)
    _c, record = plan_of(srun + ["--sockets", "2"])
    meep = [r for r in record["rows"] if r["kind"] == "meep"]
    first = meep[0]["argv"]
    assert first[:4] == ["srun", "--mpi=pmix", "-n", "16"]
    # BALANCED ACROSS SOCKETS, as the Open MPI rows were: 8 + 8, not 16 on one socket.
    assert "--cpu-bind=verbose,cores" in first and "--ntasks-per-socket=8" in first
    assert "--ntasks-per-core=1" in first and "--distribution=block:block" in first
    r48 = next(r for r in meep if r["ranks"] == 48)
    assert "--ntasks-per-socket=24" in r48["argv"]
    over = next(r for r in meep if r["ranks"] == 64)
    assert "--cpu-bind=verbose,none" in over["argv"] and "--ntasks-per-core=2" in over["argv"]
    assert not any(a.startswith("--hint") for r in meep for a in r["argv"])
    assert not [r for r in record["rows"] if r["kind"] in ("gpu", "control")]


def test_another_build_under_srun_starts_through_srun():
    _c, record = plan_of(LINUX + ["--tiers", "priority", "--passes", "1", "--sockets", "2",
                                  "--priority-sizes", "512000", "--meep-launcher", "srun",
                                  "--srun-mpi", "pmix", "--routes", "none", "--controls",
                                  "native", "--meep-build", "native=/opt/native/bin/python"])
    native = [r for r in record["rows"] if r.get("control") == "native"]
    assert native and all(r["argv"][0] == "srun" for r in native)
    assert record["resolved"]["builds"]["native"]["mpi_impl"] == "srun"


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

def test_refusals_by_name():
    no_gpu = [a for a in LINUX if a not in ("--gpu", "0")]
    assert "--gpu is required" in refusal(no_gpu)
    no_record = [a for a in LINUX]
    index = no_record.index("--meep-config-log")
    del no_record[index:index + 2]
    assert "no build record" in refusal(no_record)
    inside = LINUX + ["--run-root", "/opt/meep-gpu/runs"]
    assert "inside the tree under test" in refusal(inside)
    assert "Open MPI cannot bind" in refusal(MAC + ["--binding", "bound"])
    assert "exceed the 6 physical cores" in refusal(
        [a if a not in ("48", "96") else {"48": "6", "96": "12"}[a] for a in LINUX]
        + ["--controls", "threads4"])
    assert "no integer resolution gives pml_3d exactly 500,000" in refusal(
        LINUX + ["--sizes", "500000"])
    assert "--stamp is required" in refusal(["--python", "x", "--meep-config-log", "none"])
    assert "device 4 is in --refuse-gpus" in refusal(
        [a if a != "0" else "4" for a in LINUX] + ["--refuse-gpus", "4,5"])
    assert "srun needs --srun-mpi" in refusal(LINUX + ["--meep-launcher", "srun"])
    assert "ROUTE:CASE=TABLES" in refusal(LINUX + ["--expect-launched", "cuda=pml_3d"])


def test_nice_is_refused_live_without_allow_nice(monkeypatch, tmp_path):
    config = config_of(LINUX + ["--run-root", str(tmp_path / "runs")])
    monkeypatch.setattr(os, "nice", lambda _n: 5)
    problems = ladder.live_checks(config, resume=False, collect=True)
    assert any("nice 5" in p for p in problems), problems
    allowed = config_of(LINUX + ["--run-root", str(tmp_path / "runs"), "--allow-nice", "1"])
    assert not any("nice" in p for p in ladder.live_checks(allowed, False, collect=True))


def test_a_dry_run_starts_no_process(monkeypatch, capsys):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("the dry run started a process")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(os, "system", forbidden)
    assert ladder.main(A6000_PROFILE + ["--dry-run", "--json"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["total"]["rows"] == 111
    assert ladder.main(MAC + ["--dry-run"]) == 0
    assert "DRY RUN DONE" in capsys.readouterr().out


def test_params_precedence(tmp_path):
    params = tmp_path / "p.env"
    params.write_text("PASSES=2\nGATE_CPU_MAX=7\n")
    config = config_of(LINUX + ["--params", str(params), "--passes", "4"],
                       dict(LAUNCHING_ENV, MGPU_TIMING_GATE_CPU_MAX="9",
                            MGPU_TIMING_MONITOR_EVERY_S="5"))
    assert (config.values["PASSES"], config.sources["PASSES"]) == ("4", "flag")
    assert (config.values["GATE_CPU_MAX"], config.sources["GATE_CPU_MAX"]) == (
        "7", "params-file")
    assert config.sources["MONITOR_EVERY_S"] == "environment"
    assert config.sources["GATE_MEM_MIN_GB"] == "derived"
    assert config.gate.mem_min_gib == 250.0
    site = config_of([a for a in LINUX if a not in ("--python", "/opt/env/bin/python")],
                     dict(LAUNCHING_ENV, MGPU_SITE_PYTHON="/site/python"))
    assert (site.python, site.sources["PYTHON"]) == ("/site/python", "site:MGPU_SITE_PYTHON")


def test_expect_launched_parsing_and_preflight_pinning():
    parsed = ladder.parse_expect_launched(["default:pml_3d=triton", "cuda:pml_3d=cuda+triton"])
    assert parsed == {("default", "pml_3d"): "triton", ("cuda", "pml_3d"): "cuda+triton"}
    config = config_of(LINUX)
    assert ladder.expected_launched(config, "cuda", "pml_3d", {}) is None
    assert ladder.expected_launched(config, "cuda", "pml_3d",
                                    {("cuda", "pml_3d"): "cuda"}) == "cuda"
    entry = {"measured": {"what_launched": "cuda", "monitors_mode": "attached"},
             "deposit_repair": {"ok": True}}
    summary = ("TIMED-NOT-REPORTABLE cells=110592 launched=cuda fused 1.0 Mcell-steps/s "
               "bit_identical=True lift[]")
    assert ladder.preflight_gpu_verdict(entry, summary, None) == (True, "cuda")
    assert ladder.preflight_gpu_verdict(entry, summary, "cuda+triton") == (False, "cuda")
    unrepaired = dict(entry, deposit_repair={"ok": False})
    assert ladder.preflight_gpu_verdict(unrepaired, summary, None)[0] is False
    detached = dict(entry, measured={"what_launched": "cuda", "monitors_mode": "detached"})
    assert ladder.preflight_gpu_verdict(detached, summary, None)[0] is False
    hot = dict(entry, conditions=["cpu_state_after: CPU speed limit 80 (thermal)"])
    assert ladder.preflight_gpu_verdict(hot, summary, None)[0] is False
    array = {"measured": {"what_launched": "nothing (array path)"},
             "deposit_repair": {"ok": True}}
    assert ladder.preflight_gpu_verdict(
        array, summary.replace("launched=cuda", "launched=nothing (array path)"),
        None)[0] is False


def test_run_process_times_out_with_term_then_kill(tmp_path):
    log = tmp_path / "row.log"
    started = time.time()
    rc, timed_out = ladder.run_process(
        [sys.executable, "-c", "import time; print('up', flush=True); time.sleep(30)"],
        str(tmp_path), dict(os.environ), str(log), 1.0, kill_after=2.0)
    assert (rc, timed_out) == (124, True)
    assert time.time() - started < 10
    assert "up" in log.read_text()
    rc, timed_out = ladder.run_process([sys.executable, "-c", "raise SystemExit(3)"],
                                       str(tmp_path), dict(os.environ), str(log), 10.0)
    assert (rc, timed_out) == (3, False)


def test_a_recorder_that_raises_does_not_stop_the_ladder(tmp_path):
    config = config_of(LINUX + ["--run-root", str(tmp_path / "runs")])
    plan = ladder.build_plan(config)
    run = ladder.Ladder(config, plan, resume=False)
    os.makedirs(run.out)

    def broken(*_args, **_kwargs):
        raise TypeError("unsupported format string passed to NoneType.__format__")

    entry, summary = run.record(broken, plan[0])
    assert summary.startswith("RECORDER ERROR TypeError")
    noted = [json.loads(line) for line in open(run.ledger)]
    assert noted[-1]["ledger"] == "recorder_error" and noted[-1]["run_dir"] == run.out


def test_box_keys_match_the_box_string_keys():
    text = ("load=1.0 2.0 3.0 busy_cpus=0.5 free_gb=981 gpu0=3 MiB, 0 % "
            "others=0,0, account=0")
    assert timing_host.box_keys(text) == ["load", "busy_cpus", "free_gb", "gpu0",
                                          "others", "account"]



# ---------------------------------------------------------------------------
# Environments, instruments and the live path
# ---------------------------------------------------------------------------

def test_inherited_switches_are_unset_and_darwin_kmp_is_per_row_kind():
    environ = dict(LAUNCHING_ENV, MEEP_GPU_METAL_RESIDENCY="shipped",
                   MEEP_GPU_WARM="1", PYTORCH_ENABLE_MPS_FALLBACK="1",
                   PYTORCH_MPS_HIGH_WATERMARK_RATIO="0.0")
    environ.pop("KMP_DUPLICATE_LIB_OK")
    _c, record = plan_of(MAC + ["--tiers", "priority", "--passes", "1", "--controls",
                                "none", "--priority-sizes", "512000"], environ)
    gpu = next(r for r in record["rows"] if r["kind"] == "gpu")
    meep = next(r for r in record["rows"] if r["kind"] == "meep")
    for name in ("MEEP_GPU_METAL_RESIDENCY", "MEEP_GPU_WARM",
                 "PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_HIGH_WATERMARK_RATIO"):
        assert name in gpu["env"]["unset"], name
    assert "MEEP_GPU_METAL_RESIDENCY" in meep["env"]["unset"]
    assert gpu["env"]["set"]["KMP_DUPLICATE_LIB_OK"] == "TRUE"
    assert "KMP_DUPLICATE_LIB_OK" in meep["env"]["unset"]
    assert "MEEP_GPU_SUBNORMAL_POLICY" not in gpu["env"]["set"]
    assert meep["argv"][meep["argv"].index("--table") + 1] == "metal"


def test_linux_rows_never_set_kmp():
    _c, record = plan_of(LINUX + ["--tiers", "priority", "--passes", "1", "--controls",
                                  "none", "--priority-sizes", "512000"])
    for row in record["rows"]:
        if row["env"]:
            assert "KMP_DUPLICATE_LIB_OK" not in row["env"]["set"], row["key"]
            assert "KMP_DUPLICATE_LIB_OK" not in row["env"]["unset"], row["key"]


def test_gpu_rows_must_time_with_monitors_attached():
    detached = "--repeats 6 --target-seconds 2.0 --memory-budget-bytes 1000"
    assert "--monitors attached" in refusal(LINUX + ["--gpu-flags", detached])
    config = config_of(LINUX + ["--gpu-flags", detached, "--routes", "none"])
    assert config.gpu_flags == detached


def test_darwin_instrument_and_array_control():
    config = config_of(MAC)
    assert config.memory_budget == int(0.6 * 32 * 1024 ** 3)
    assert f"--memory-budget-bytes {config.memory_budget}" in config.gpu_flags
    _c, record = plan_of(MAC + ["--tiers", "preflight", "--cases", "pml_3d"])
    preflight = next(r for r in record["rows"] if r["kind"] == "gpu")
    assert str(config.memory_budget) in preflight["argv"]
    assert "--array-control row with the metal route" in refusal(
        MAC + ["--array-control", "row"])
    assert config.gate.gpu_metric == "busy"


def test_gpu_gate_metric_follows_the_scope():
    assert config_of(LINUX).gate.gpu_metric == "load1"
    assert config_of(LINUX + ["--gate-scope", "allocation"]).gate.gpu_metric == "busy"
    assert "load average is host-wide" in refusal(
        LINUX + ["--gate-scope", "allocation", "--gate-gpu-metric", "load1"])


def test_hwthread_key_is_not_doubled():
    _c, record = plan_of(LINUX + ["--tiers", "priority", "--passes", "1",
                                  "--priority-sizes", "512000", "--controls", "hwthread"])
    keys = [r["key"] for r in record["rows"] if r.get("control") == "hwthread"]
    assert keys == ["priority_meep_r64_bound-hwthread_res20_p1",
                    "priority_meep_r96_bound-hwthread_res20_p1"]
    outs = {r["out"] for r in record["rows"] if r.get("control") == "hwthread"}
    assert all(o.endswith("meep_bound-hwthread") for o in outs)


def test_a_restricted_affinity_passes_its_scope_to_the_bench(monkeypatch):
    monkeypatch.setattr(ladder, "detect_topology", lambda _p: {
        "method": "sysfs", "physical_cores": 24, "logical_cpus": 48, "sockets": 1,
        "threads_per_core": 2, "restricted_by_affinity": True, "core_of_cpu": {}})
    _c, record = plan_of([a for a in LINUX if a not in ("48", "96", "--physical-cores",
                                                        "--logical-cpus")]
                         + ["--tiers", "priority", "--passes", "1", "--controls", "none",
                            "--priority-sizes", "512000"])
    meep = next(r for r in record["rows"] if r["kind"] == "meep")
    assert meep["argv"][meep["argv"].index("--scope-cpus") + 1] == "48"
    monkeypatch.undo()
    _c, plain = plan_of(LINUX + ["--tiers", "priority", "--passes", "1", "--controls",
                                 "none", "--priority-sizes", "512000"])
    assert all("--scope-cpus" not in (r["argv"] or []) for r in plain["rows"])


def test_gpu_resolution_by_uuid():
    uuids = {"0": "GPU-aaaa", "1": "GPU-bbbb"}
    config = config_of(LINUX + ["--gpu", "auto"])
    problems = []
    ladder.resolve_gpu(config, problems.append, uuids=uuids, inherited="1")
    assert not problems and config.gpu == "GPU-bbbb" and config.gate.gpu == "GPU-bbbb"
    many = config_of(LINUX + ["--gpu", "auto"])
    ladder.resolve_gpu(many, problems.append, uuids=uuids, inherited=None)
    assert problems and "exactly one allocated GPU" in problems[-1]
    by_uuid = config_of([a if a != "0" else "GPU-bbbb" for a in LINUX])
    problems = []
    ladder.resolve_gpu(by_uuid, problems.append, uuids=uuids, inherited="1")
    assert not problems and by_uuid.gpu == "GPU-bbbb"
    outside = config_of(LINUX)
    ladder.resolve_gpu(outside, problems.append, uuids=uuids, inherited="GPU-bbbb")
    assert problems and "outside the inherited" in problems[-1]
    refused = config_of(LINUX + ["--gpu", "auto", "--refuse-gpus", "1"])
    problems = []
    ladder.resolve_gpu(refused, problems.append, uuids=uuids, inherited="1")
    assert problems and "--refuse-gpus" in problems[-1]


def test_live_checks_refuse_a_missing_launcher_and_an_unknown_account(tmp_path):
    python = tmp_path / "bin" / "python"
    python.parent.mkdir()
    python.write_text("")
    (tmp_path / "tree").mkdir()
    config = config_of(["--stamp", "t", "--python", str(python), "--meep-config-log", "none",
                        "--platform", ladder.detect_platform(), "--physical-cores", "8",
                        "--logical-cpus", "8", "--memory-total-gb", "32", "--mpi-impl",
                        "openmpi", "--tree-layout", "release", "--tree-src",
                        str(tmp_path / "tree"),
                        "--run-root", str(tmp_path / "runs"), "--routes", "none",
                        "--account", "no-such-user-zz9", "--meep-build",
                        f"native={python},{tmp_path / 'nowhere' / 'mpirun'}"])
    problems = ladder.live_checks(config, resume=False, collect=True, environ={})
    assert any("names no user on this host" in p for p in problems), problems
    assert any("launcher of build native" in p and "does not exist" in p
               for p in problems), problems


def test_a_row_that_never_started_is_struck_without_a_done_marker(tmp_path):
    config = config_of(LINUX + ["--run-root", str(tmp_path / "runs"), "--tiers",
                                "priority", "--passes", "1", "--controls", "none",
                                "--priority-sizes", "512000", "--routes", "none",
                                "--gate-hold-meep-s", "0"])
    plan = ladder.build_plan(config)
    run = ladder.Ladder(config, plan, resume=False)
    os.makedirs(os.path.join(run.out, "done"))
    run.gate = lambda kind, label: 0.0
    run.code_digest = lambda: "c" * 64
    row = plan[0]
    config.builds["reference"].launcher = str(tmp_path / "no-such-mpirun")
    run.do_meep(row)
    noted = [json.loads(line) for line in open(run.ledger)]
    assert noted[-1]["ledger"] == "not_started" and "not started" in noted[-1]["struck"]
    assert noted[-1]["host"] == run.host
    assert not os.listdir(os.path.join(run.out, "done"))


def test_a_resume_on_another_host_is_refused(tmp_path, monkeypatch):
    config = config_of(LINUX + ["--run-root", str(tmp_path / "runs"), "--routes", "none"])
    run = ladder.Ladder(config, ladder.build_plan(config), resume=False)
    os.makedirs(run.out)
    run.check_identity()
    again = ladder.Ladder(config, ladder.build_plan(config), resume=True)
    again.check_identity()
    moved = ladder.Ladder(config, ladder.build_plan(config), resume=True)
    moved.host = "another-node"
    with pytest.raises(SystemExit):
        moved.check_identity()


def test_launcher_probe_checks_every_mode_the_plan_uses():
    config = config_of(LINUX + ["--controls", "hwthread,threads2", "--sockets", "2"])
    modes = ladder.placement_modes(config)
    assert [m for m, _n, _t in modes] == ["bound", "unbound", "over", "bound-hwthread",
                                          "bound-pe"]
    assert dict((m, n) for m, n, _t in modes)["bound"] == 48
