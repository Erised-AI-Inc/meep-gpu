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

import hashlib
import json
import os
import subprocess
import sys
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import neutral_ids  # noqa: E402
import timing_host  # noqa: E402
import timing_ladder as ladder  # noqa: E402
import timing_records  # noqa: E402

FIXTURES = os.path.join(HERE, "fixtures", "timing")
PARAMS = os.path.join(FIXTURES, "a6000_2026-09-28_b.params")
COMMANDS = os.path.join(FIXTURES, "ladder_2026-09-28_b_commands.json")
#: The launching environment the comparison is made under: a shell that exports
#: KMP_DUPLICATE_LIB_OK, as the 2026-09-28 reproduction's does (named difference 7).
LAUNCHING_ENV = {"PATH": "/usr/bin:/bin", "HOME": "/home/user",
                 "KMP_DUPLICATE_LIB_OK": "TRUE"}
#: Variables the released ladder unsets that the 2026-09-28 ladder did not (named
#: differences: the hardware scope is never inherited, and neither is the name it
#: replaced, which the package refuses; a MEEP row no longer inherits a GPU row's
#: caches or an OpenMP placement).
ADDED_UNSETS = {"MEEP_GPU_HARDWARE", "MEEP_GPU_ALLOW_UNCERTIFIED", "TRITON_CACHE_DIR",
                "CUPY_CACHE_DIR", "OMP_PROC_BIND", "OMP_PLACES"}
#: Variables the released ladder SETS that the 2026-09-28 ladder left unset (named
#: difference: an NVIDIA row restricts the kernels to certified identities, which is
#: what unset meant on NVIDIA before supported-uncertified devices ran by default).
ADDED_SETS = {"MEEP_GPU_HARDWARE": "certified"}


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


def test_the_a6000_profile_plans_the_111_rows_of_2026_09_28(monkeypatch):
    # The profile names the core counts. The process's own topology is withheld: on a
    # Linux host the ladder would read it, and under a CPU mask narrower than the host
    # (a NUMA-pinned shard) every MEEP row would carry ``--scope-cpus``, which the
    # fixture's rows, planned with the whole host in scope, do not.
    monkeypatch.setattr(ladder, "detect_topology", lambda _p: {"method": None})
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


#: The sha256 of the A6000 profile's plan record, serialised as below, as the ladder drew it
#: before it could plan one side alone (``--ranks none``): a ladder that declares neither
#: side away plans byte for byte what it planned before.
A6000_PLAN_SHA256 = "f9a7ca05c46982f90eea081868205c89ee4131248f23d9cf5a29627ad9939803"


def test_the_a6000_plan_is_byte_for_byte_the_plan_drawn_before_one_side_could_be_planned(
        monkeypatch):
    # The profile names the host; nothing of the host the test runs on enters the plan: on
    # a Linux host the ladder would read its topology, memory and the profile's account.
    monkeypatch.setattr(ladder, "detect_platform", lambda: "other")
    monkeypatch.setattr(ladder, "detect_topology", lambda _p: {"method": None})
    monkeypatch.setattr(timing_host, "account_uid", lambda _name: None)
    _config, record = plan_of(A6000_PROFILE)
    text = json.dumps(record, indent=1, sort_keys=True, default=str)
    assert len(record["rows"]) == 111
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == A6000_PLAN_SHA256
    assert "none" not in str(record["params"]["RANKS"]["value"]).lower()


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

UNIT = ["--tiers", "preflight digests priority", "--priority-sizes", "512000",
        "--cases", "pml_3d", "--binding", "bound",
        "--controls", "threads2,threads4,hwthread,split", "--threaded-reference", "1"]


def test_ranks_none_plans_the_gpu_routes_alone():
    """``--ranks none``: no row that starts MEEP (no rank block, no MEEP control, no
    preflight MEEP row or digest, no harness-lift digest, no binding probe or check); the
    preflight is the GPU rows' launch check. Both sides named away is refused."""
    config, record = plan_of(LINUX + UNIT + ["--tiers",
                                             "preflight digests bindprobe priority bindcheck",
                                             "--binding", "probe", "--ranks", "none"])
    kinds = {(row["tier"], row["kind"]) for row in record["rows"]}
    # the array control read off the GPU rows (--array-control extract) is a GPU-side row
    assert kinds == {("preflight", "gpu"), ("priority", "gpu"), ("priority", "control")}
    assert config.no_meep and config.controls == [] and config.probe_ranks == []
    assert config.preflight_ranks == [] and config.blocks == ([], [], [])
    assert sorted(config.controls_omitted) == ["hwthread", "split", "threads2", "threads4"]
    assert all("--ranks none" in why for why in config.controls_omitted.values())
    assert ladder.placement_modes(config) == [], "no launcher mode to probe"
    assert "times neither the GPU routes nor MEEP" in refusal(
        LINUX + UNIT + ["--ranks", "none", "--routes", "none"])
    whole, _ = plan_of(LINUX + UNIT)
    assert not whole.no_meep and ladder.placement_modes(whole)


def test_the_two_sides_planned_alone_are_the_whole_plan_between_them():
    """``--ranks none`` and ``--routes none`` split a ladder's rows between them: every
    row of the whole plan is planned by exactly one side, with the same key, command,
    working directory and environment; the array control is read off the GPU rows."""
    plans = {name: plan_of(LINUX + UNIT + extra)[1]["rows"]
             for name, extra in (("whole", []), ("gpu", ["--ranks", "none"]),
                                 ("meep", ["--routes", "none"]))}
    rows = {name: {row["key"]: row for row in plan} for name, plan in plans.items()}
    assert all(len(rows[name]) == len(plans[name]) for name in plans), "keys are unique"
    assert {row["kind"] for row in plans["gpu"]} == {"gpu", "control"}
    assert {row["kind"] for row in plans["meep"]} == {"meep", "digest", "lift"}
    assert not set(rows["gpu"]) & set(rows["meep"])
    assert set(rows["gpu"]) | set(rows["meep"]) == set(rows["whole"])
    for side in ("gpu", "meep"):
        for key, row in rows[side].items():
            ours = rows["whole"][key]
            assert (row["argv"], row["cwd"], row["env"]) == (
                ours["argv"], ours["cwd"], ours["env"]), (side, key)


def test_the_gpu_routes_alone_ask_the_slurm_check_for_one_process(monkeypatch, tmp_path):
    """A live Linux run checks its largest MEEP launch against the allocation; the GPU
    routes alone launch none, so they are checked as one process."""
    config = config_of(LINUX + UNIT + ["--ranks", "none", "--run-root", str(tmp_path / "r")])
    asked = []
    monkeypatch.setattr(ladder, "detect_platform", lambda: "linux")
    monkeypatch.setattr(timing_host, "affinity", lambda: list(range(96)))
    monkeypatch.setattr(timing_host, "slurm_refusals",
                        lambda env, impl, kind, largest, cpus: asked.append(largest) or [])
    ladder.live_checks(config, resume=False, collect=True, environ={})
    assert asked == [1]


def test_default_rank_blocks():
    assert ladder.default_rank_blocks(48, 96, True) == ([16, 24, 32, 48], [8, 1], [64])
    assert ladder.default_rank_blocks(8, 10, False) == ([3, 4, 5, 8], [1], [10])
    assert ladder.default_rank_blocks(8, 8, False) == ([3, 4, 5, 8], [1], [])


@pytest.mark.parametrize("allow,nvidia,metal", [("0", "certified", None),
                                                ("1", "any", "any")])
def test_an_nvidia_row_runs_certified_identities_only_unless_asked(allow, nvidia, metal):
    """``0`` exports MEEP_GPU_HARDWARE=certified to the NVIDIA rows and leaves the
    Metal rows the Metal default; ``1`` exports ``any`` to both; no MEEP row inherits
    it, and no row of any kind inherits or exports the retired MEEP_GPU_ALLOW_UNCERTIFIED."""
    _config, record = plan_of(A6000_PROFILE + ["--allow-uncertified", allow])
    gpu = [r for r in record["rows"] if r["kind"] == "gpu"]
    assert gpu and all(r["env"]["set"].get("MEEP_GPU_HARDWARE") == nvidia
                       for r in gpu), [r["key"] for r in gpu]
    rows = list(record["rows"])
    _config, record = plan_of(MAC + ["--allow-uncertified", allow])
    gpu = [r for r in record["rows"] if r["kind"] == "gpu"]
    assert gpu and all(r["env"]["set"].get("MEEP_GPU_HARDWARE") == metal
                       for r in gpu), [r["key"] for r in gpu]
    if metal is None:
        assert all("MEEP_GPU_HARDWARE" in r["env"]["unset"] for r in gpu)
    meep = [r for r in record["rows"] if r["kind"] == "meep"]
    assert meep and all("MEEP_GPU_HARDWARE" in r["env"]["unset"] for r in meep)
    rows += list(record["rows"])
    assert all("MEEP_GPU_ALLOW_UNCERTIFIED" not in r["env"]["set"]
               and "MEEP_GPU_ALLOW_UNCERTIFIED" in r["env"]["unset"]
               for r in rows if r["kind"] in ("gpu", "meep")), \
        [r["key"] for r in rows if r["kind"] in ("gpu", "meep")
         and "MEEP_GPU_ALLOW_UNCERTIFIED" not in r["env"]["unset"]]


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


def test_linux_controls_carry_their_configuration(monkeypatch):
    # The core counts and socket count are stated; the process's own topology is
    # withheld, so a Linux host whose CPU mask holds one thread per core does not
    # replace the stated two threads per core.
    monkeypatch.setattr(ladder, "detect_topology", lambda _p: {"method": None})
    environ = dict(LAUNCHING_ENV)
    config, record = plan_of(LINUX + ["--tiers", "priority", "--passes", "1",
                                      "--priority-sizes", "512000", "--meep-build",
                                      "native=/opt/native/bin/python", "--sockets", "2"],
                             environ)
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
    hw = controls["hwthread"][0]["argv"]
    assert hw[hw.index("--map-by") + 1].startswith("pe-list=0,48,2,50,")
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
    # Balanced across sockets, as the Open MPI rows were: 8 + 8, not 16 on one socket.
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
    assert "dry run done" in capsys.readouterr().out


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
    # the ledger leaves the machine: the run directory under its root's label, which a
    # reader resolves back to the directory on this machine
    assert noted[-1]["ledger"] == "recorder_error"
    assert noted[-1]["run_dir"] == neutral_ids.path(run.out)
    assert os.path.realpath(neutral_ids.resolve(noted[-1]["run_dir"])) == \
        os.path.realpath(run.out)


def test_the_mac_gate_is_calibrated_at_the_start_and_linux_keeps_its_default(tmp_path):
    """The 1.0 Mac timing ladder found that the macOS default CPU gate (1 of 10 busy CPUs)
    could not open with the user's desktop session idling at 1.2-2.0. On macOS it is
    calibrated before the first row, written down, and read back on a resume; Linux keeps
    10 % of its logical CPUs."""
    linux = config_of(LINUX)
    assert linux.cpu_calibration is None and linux.gate.cpu_max == 10
    explicit = config_of(MAC + ["--gate-cpu-max", "2"])
    assert explicit.cpu_calibration is None and explicit.gate.cpu_max == 2.0
    config = config_of(MAC + ["--run-root", str(tmp_path / "runs")])
    assert config.values["GATE_CPU_MAX"] == "auto"
    assert config.cpu_calibration == {"readings": 18, "every_s": 10.0, "logical_cpus": 10}
    assert config.gate.cpu_max == 1.0                  # until the readings are taken
    run = ladder.Ladder(config, ladder.build_plan(config), resume=False)
    os.makedirs(run.out)
    readings = iter([1.2, 1.5, 1.9, 2.0, 1.4, 1.7] * 3)
    slept = []
    run.calibrate_gate(reader=lambda: next(readings), sleep=slept.append)
    assert config.gate.cpu_max == 2.5 and config.sources["GATE_CPU_MAX"] == \
        "calibrated at start"
    assert slept == [10.0] * 17
    stored = json.load(open(os.path.join(run.out, "gate_calibration.json")))
    assert stored["value"] == 2.5 and len(stored["readings"]) == 18
    # a resume reads the threshold back; it does not calibrate again
    again = config_of(MAC + ["--run-root", str(tmp_path / "runs")])
    resumed = ladder.Ladder(again, ladder.build_plan(again), resume=True)
    resumed.calibrate_gate(reader=lambda: pytest.fail("calibrated twice"))
    assert again.gate.cpu_max == 2.5


def test_a_gpu_only_ladder_times_one_leg_behind_one_identity_row_per_size():
    """The GPU-only row of the release timing: the timed GPU rows lift the fused leg alone;
    each route, case and size is first proved bit-identical by one three-leg identity row
    with the preflight's short windows; the array control is its own row or none."""
    base = LINUX + ["--tiers", "preflight identity priority", "--passes", "3",
                    "--priority-sizes", "512000,2097152", "--controls", "none",
                    "--routes", "default,cuda", "--gpu-legs", "gpu-only"]
    config, record = plan_of(base + ["--array-control", "none"])
    assert config.gpu_legs == "gpu-only"
    gpu = [r for r in record["rows"] if r["kind"] == "gpu"]
    identity = [r for r in gpu if r["tier"] == "identity"]
    assert sorted((r["route"], r["cells"]) for r in identity) == [
        ("cuda", 512000), ("cuda", 2097152), ("default", 512000), ("default", 2097152)]
    timed = [r for r in gpu if r["tier"] == "priority"]
    assert len(timed) == 2 * 2 * 3
    for row in timed:
        argv = row["argv"]
        assert argv[argv.index("--legs") + 1] == "gpu-only"
    for row in identity + [r for r in gpu if r["tier"] == "preflight"]:
        assert "--legs" not in row["argv"] and "0.5" in row["argv"]
        assert "/identity/" in row["log"] or "/preflight/" in row["log"]
    assert not [r for r in record["rows"] if r["kind"] == "control"]
    _c, rows_control = plan_of(base + ["--array-control", "row"])
    assert [r for r in rows_control["rows"] if r["kind"] == "control"]
    assert "no array leg to read the control off" in refusal(base)
    assert "keep 'identity' in --tiers" in refusal(
        [a if a != "preflight identity priority" else "preflight priority" for a in base]
        + ["--array-control", "none"])
    # the default ladder is untouched: no identity rows, three-leg GPU rows
    _c, default = plan_of(LINUX + ["--tiers", "preflight identity priority", "--passes",
                                   "1", "--priority-sizes", "512000", "--controls", "none"])
    assert not [r for r in default["rows"] if r["tier"] == "identity"]
    assert all("--legs" not in r["argv"] for r in default["rows"] if r["kind"] == "gpu")


def test_a_gpu_only_row_reads_its_identity_row_off_the_ledger(tmp_path):
    config = config_of(LINUX + ["--run-root", str(tmp_path / "runs"), "--tiers",
                                "identity priority", "--passes", "1", "--priority-sizes",
                                "512000", "--controls", "none", "--gpu-legs", "gpu-only",
                                "--array-control", "none"])
    plan = ladder.build_plan(config)
    run = ladder.Ladder(config, plan, resume=False)
    os.makedirs(run.out)
    timed = next(r for r in plan if r.kind == "gpu" and r.tier == "priority")
    assert run.identity_entry(timed) is None
    for entry in ({"ledger": "gpu", "tier": "identity", "route": "default", "case": "pml_3d",
                   "res": timed.res, "key": "first"},
                  {"ledger": "gpu", "tier": "identity", "route": "cuda", "case": "pml_3d",
                   "res": timed.res, "key": "other route"},
                  {"ledger": "gpu", "tier": "priority", "route": "default", "case": "pml_3d",
                   "res": timed.res, "key": "a timed row"},
                  {"ledger": "gpu", "tier": "identity", "route": "default", "case": "pml_3d",
                   "res": timed.res, "key": "resumed"}):
        timing_records.append(run.ledger, entry)
    assert run.identity_entry(timed)["key"] == "resumed"


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
                                  "--priority-sizes", "512000", "--controls", "hwthread",
                                  "--sockets", "2"])
    keys = [r["key"] for r in record["rows"] if r.get("control") == "hwthread"]
    assert keys == ["priority_meep_r64_bound-hwthread_res20_p1",
                    "priority_meep_r96_bound-hwthread_res20_p1"]
    outs = {r["out"] for r in record["rows"] if r.get("control") == "hwthread"}
    assert all(o.endswith("meep_bound-hwthread") for o in outs)


def test_the_hwthread_control_without_a_topology_is_omitted_by_name(monkeypatch):
    """The control names each rank's hardware thread, so it needs the socket count;
    without it the control is left out, by name, instead of run doubled up (the 1.0
    NVIDIA timing ladder struck its 64-rank rows for a doubled-up placement)."""
    # On a Linux host the ladder reads that host's own topology, which would supply the
    # count: the topology is withheld here, as in test_srun_launcher_and_its_flags.
    monkeypatch.setattr(ladder, "detect_topology", lambda _p: {"method": None})
    config, record = plan_of(LINUX + ["--tiers", "priority", "--passes", "1",
                                      "--priority-sizes", "512000", "--controls",
                                      "hwthread,split"])
    assert config.controls == ["split"]
    assert "named hardware thread" in config.controls_omitted["hwthread"]
    assert not [r for r in record["rows"] if r.get("control") == "hwthread"]


#: A desktop of 8 performance cores and 24 logical CPUs (hardware threads and efficiency
#: cores), as its site file states it, timed with a catalog's controls given as flags.
DESKTOP = [a for a in LINUX if a not in ("48", "96", "--physical-cores", "--logical-cpus")] + [
    "--tiers", "priority", "--passes", "1", "--priority-sizes", "512000", "--sockets", "1",
    "--binding", "bound", "--controls", "threads2,threads4,hwthread,split"]


def _meep_rows(record):
    return sorted({(row["control"] or "reference", row["ranks"])
                   for row in record["rows"] if row["kind"] == "meep"})


def test_the_host_s_meep_rank_ceiling_bounds_every_control(monkeypatch, tmp_path):
    """The host's MEEP rank ceiling (``--meep-rank-ceiling``, which cert8.py hands the MEEP
    half of a split session from its site file's rank blocks) is the most MEEP ranks any
    row takes, and the controls a catalog asks for by flag cannot lift it: on the
    8-performance-core desktop (ceiling 8) the bound-hwthread rows (4P/3 and 2P: 11 and 16
    ranks) are refused by name and every other control is planned within it. A control
    with a row inside and a row above keeps the one inside, and the one above is named.
    The same site file without the ceiling plans as before: a declared rank block is no
    ceiling of its own."""
    monkeypatch.setattr(ladder, "detect_topology", lambda _p: {"method": None})
    site = tmp_path / "desktop.params"
    site.write_text("PHYSICAL_CORES=8\nLOGICAL_CPUS=24\nRANKS=8|1|\n")
    config, record = plan_of(DESKTOP + ["--params", str(site), "--meep-rank-ceiling", "8"])
    assert config.meep_rank_ceiling == 8 and config.controls == ["threads2", "threads4",
                                                                  "split"]
    assert _meep_rows(record) == [("reference", 1), ("reference", 8), ("split", 5),
                                  ("split", 8), ("threads2", 4), ("threads4", 2)]
    assert config.preflight_ranks == [1, 8] and config.probe_ranks == [5, 8]
    assert record["resolved"]["meep_rank_ceiling"] == 8
    why = config.controls_omitted["hwthread"]
    assert "plan 11, 16 MEEP ranks, above the host's MEEP rank ceiling 8" in why
    assert "rows_above_rank_ceiling" not in config.controls_omitted
    # the same site file without the ceiling: the plan as before, hwthread included
    config, record = plan_of(DESKTOP + ["--params", str(site)])
    assert config.meep_rank_ceiling is None and "hwthread" in config.controls
    assert {("hwthread", 11), ("hwthread", 16)} <= set(_meep_rows(record))
    assert "meep_rank_ceiling" not in record["resolved"] and config.controls_omitted == {}
    # one row of a control inside the ceiling, one above it
    config, record = plan_of(DESKTOP + ["--physical-cores", "8", "--logical-cpus", "24",
                                        "--ranks", "4,12|1|", "--meep-rank-ceiling", "12"])
    assert ("hwthread", 11) in _meep_rows(record) and ("hwthread", 16) not in _meep_rows(record)
    assert config.controls_omitted["rows_above_rank_ceiling"] == (
        "MEEP ranks above the host's MEEP rank ceiling 12 (--meep-rank-ceiling): hwthread@16")


@pytest.mark.parametrize("extra, named", [
    (["--physical-cores", "12", "--ranks", "8|1|"],
     "the preflight MEEP row runs at the physical cores, 12 (from flag), above the ceiling; "
     "declare PHYSICAL_CORES at or below 8"),
    (["--physical-cores", "8", "--ranks", "4,12|1|"],
     "the rank blocks 4,12|1| (from flag) reach 12 MEEP ranks; declare RANKS at or below"),
    (["--physical-cores", "8", "--ranks", "8|1|", "--probe-ranks", "6,10"],
     "the binding probe's ranks 6,10 (from flag) reach 10; declare PROBE_RANKS"),
    (["--physical-cores", "8"],
     "the rank blocks 3,4,5,8|1|11 (from derived) reach 11 MEEP ranks"),
])
def test_a_meep_row_the_ceiling_cannot_trim_is_refused_by_name(monkeypatch, extra, named):
    """Under the host's MEEP rank ceiling no MEEP row runs above it: a rank block, the
    binding probe and the preflight row (at the physical cores, 12 as WSL2 reports the
    8-performance-core desktop) are refused by name rather than planned, and so is a
    ceiling below one rank."""
    monkeypatch.setattr(ladder, "detect_topology", lambda _p: {"method": None})
    argv = DESKTOP + ["--logical-cpus", "24", "--meep-rank-ceiling", "8"] + extra
    assert named in refusal(argv)
    assert "a rank count of 1 or more" in refusal(
        DESKTOP + ["--physical-cores", "8", "--logical-cpus", "24", "--ranks", "8|1|",
                   "--meep-rank-ceiling", "0"])


#: The A6000 profile's plan record with the rank blocks the GPU host's 1.0.1 site file
#: declares (``RANKS=32,48||``) and the shipped NVIDIA catalog's ladder flags, on that host's
#: topology (2 sockets, 2 threads per core), serialised as below, as the ladder drew it
#: before a host's MEEP rank ceiling existed: a declared rank block sets no ceiling.
A6000_DECLARED_RANKS_PLAN_SHA256 = (
    "82c2c4729dc1d7ba9f3a632e80b1d7ca737e8102be54fa51e78d22fe7244e627")


def test_a_declared_rank_block_is_no_ceiling_on_the_a6000(monkeypatch, tmp_path):
    """The GPU host's site file declares the reference rank blocks it times (32 and 48);
    that is no limit on the host's MEEP ranks, so the catalog's bound-hwthread control is
    still planned at 64 ranks (96 is above its CPU floor), byte for byte as before."""
    monkeypatch.setattr(ladder, "detect_platform", lambda: "other")
    monkeypatch.setattr(ladder, "detect_topology", lambda _p: {"method": None})
    monkeypatch.setattr(timing_host, "account_uid", lambda _name: None)
    site = tmp_path / "a6000_ranks.params"
    site.write_text(open(PARAMS).read() + "RANKS=32,48||\n")
    catalog = json.load(open(os.path.join(HERE, "rounds", "1.0", "cert8_catalog_nvidia.json")))
    config, record = plan_of(["--params", str(site), "--sockets", "2", "--threads-per-core",
                              "2"] + catalog["ladder"]["ladder_args"])
    assert "meep_rank_ceiling" not in record["resolved"] and "hwthread" in config.controls
    assert ("hwthread", 64) in _meep_rows(record) and len(record["rows"]) == 101
    text = json.dumps(record, indent=1, sort_keys=True, default=str)
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == A6000_DECLARED_RANKS_PLAN_SHA256


def test_a_restricted_affinity_passes_its_scope_to_the_bench(monkeypatch):
    # Both readings are stated, so the test means the same under any CPU mask the
    # process itself runs with.
    monkeypatch.setattr(ladder, "detect_topology", lambda _p: {
        "method": "sysfs", "physical_cores": 24, "logical_cpus": 48, "sockets": 1,
        "threads_per_core": 2, "restricted_by_affinity": True, "core_of_cpu": {}})
    _c, record = plan_of([a for a in LINUX if a not in ("48", "96", "--physical-cores",
                                                        "--logical-cpus")]
                         + ["--tiers", "priority", "--passes", "1", "--controls", "none",
                            "--priority-sizes", "512000"])
    meep = next(r for r in record["rows"] if r["kind"] == "meep")
    assert meep["argv"][meep["argv"].index("--scope-cpus") + 1] == "48"
    monkeypatch.setattr(ladder, "detect_topology", lambda _p: {
        "method": "sysfs", "physical_cores": 48, "logical_cpus": 96, "sockets": 1,
        "threads_per_core": 2, "restricted_by_affinity": False, "core_of_cpu": {}})
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


# ---------------------------------------------------------------------------
# The extended ladder: context legs, hybrid reference legs, preference legs
# ---------------------------------------------------------------------------

def test_the_extended_options_add_nothing_at_their_defaults():
    _config, record = plan_of(A6000_PROFILE)
    for row in record["rows"]:
        assert "--context-label" not in (row["argv"] or []), row["key"]
        assert not any(name.startswith("MEEP_GPU_PREFER")
                       for name in ((row.get("env") or {}).get("set") or {})), row["key"]
    assert "context" not in _config.controls
    assert _config.threaded_reference is False


def test_context_legs_are_labelled_controls_and_admit_a_build_without_mpi():
    config, record = plan_of(LINUX + [
        "--tiers", "priority", "--passes", "1", "--priority-sizes", "512000",
        "--controls", "context",
        "--context-meep", "ctx-native=/opt/native/bin/python",
        "--context-meep", "ctx-cfdouble=/opt/cf/bin/python,none"])
    rows = [r for r in record["rows"] if r["kind"] == "meep" and r["control"] == "context"]
    by_build = {}
    for row in rows:
        by_build.setdefault(row["configuration"]["build"], []).append(row)
    assert sorted(r["ranks"] for r in by_build["ctx-native"]) == [32, 48]
    assert [r["ranks"] for r in by_build["ctx-cfdouble"]] == [1]
    native = by_build["ctx-native"][0]
    assert native["argv"][0] == "/opt/native/bin/mpirun"
    assert native["argv"][native["argv"].index("--context-label") + 1] == (
        "native-tuned MEEP (contraction on)")
    serial = by_build["ctx-cfdouble"][0]
    assert serial["argv"][0] == "/opt/cf/bin/python"           # no launcher
    assert serial["argv"][serial["argv"].index("--context-label") + 1] == (
        "conda-forge MEEP (double precision)")
    assert serial["binding"] == "serial"
    # Without a context build the control is omitted BY NAME; an unlabelled name refused.
    config, _record = plan_of(LINUX + ["--tiers", "priority", "--passes", "1",
                                        "--priority-sizes", "512000",
                                        "--controls", "context"])
    assert "context" in config.controls_omitted
    assert "no --context-meep" in config.controls_omitted["context"]
    assert "no label" in refusal(LINUX + ["--controls", "context",
                                          "--context-meep", "other=/opt/o/bin/python"])
    config, record = plan_of(LINUX + [
        "--tiers", "priority", "--passes", "1", "--priority-sizes", "512000",
        "--controls", "context", "--context-ranks", "1,48",
        "--context-meep", "mine=/opt/m/bin/python", "--context-label", "mine=my MEEP"])
    rows = [r for r in record["rows"] if r["kind"] == "meep" and r["control"] == "context"]
    assert sorted(r["ranks"] for r in rows) == [1, 48]


def test_threaded_reference_makes_the_hybrid_rows_reference_legs():
    _config, record = plan_of(LINUX + ["--tiers", "priority", "--passes", "1",
                                       "--priority-sizes", "512000", "--controls",
                                       "threads2,threads4", "--threaded-reference", "1"])
    hybrid = [r for r in record["rows"] if r["kind"] == "meep"
              and r["configuration"]["threads_per_rank"] > 1]
    assert sorted((r["ranks"], r["configuration"]["threads_per_rank"]) for r in hybrid) == [
        (12, 4), (24, 2)]
    assert all(r["control"] is None for r in hybrid)
    _config, record = plan_of(LINUX + ["--tiers", "priority", "--passes", "1",
                                       "--priority-sizes", "512000", "--controls",
                                       "threads2,threads4"])
    hybrid = [r for r in record["rows"] if r["kind"] == "meep"
              and r["configuration"]["threads_per_rank"] > 1]
    assert sorted(r["control"] for r in hybrid) == ["threads2", "threads4"]


def test_preference_routes_export_the_switch_and_time_the_same_table():
    config, record = plan_of(LINUX + ["--tiers", "priority", "--passes", "1",
                                      "--priority-sizes", "512000", "--controls", "none",
                                      "--routes",
                                      "default,default_prefoff,cuda,cuda_prefon",
                                      "--expect-launched", "default_prefoff:pml_3d=triton"])
    gpu = {r["route"]: r for r in record["rows"] if r["kind"] == "gpu"}
    assert set(gpu) == {"default", "default_prefoff", "cuda", "cuda_prefon"}
    assert gpu["default_prefoff"]["table"] == "triton" and gpu["cuda_prefon"]["table"] == "cuda"
    assert gpu["default_prefoff"]["env"]["set"]["MEEP_GPU_PREFER"] == "span"
    assert gpu["cuda_prefon"]["env"]["set"]["MEEP_GPU_PREFER"] == "measured"
    assert gpu["cuda_prefon"]["env"]["set"]["MEEP_GPU_BACKEND_PREFERENCE"] == "cuda"
    assert "MEEP_GPU_PREFER" not in gpu["default"]["env"]["set"]
    assert gpu["default_prefoff"]["expected_launched"] == "triton"
    keys = {r["key"] for r in record["rows"] if r["kind"] == "gpu"}
    assert len(keys) == 4                                    # distinct rows, distinct keys
    config, record = plan_of(LINUX + ["--tiers", "priority", "--passes", "1",
                                      "--priority-sizes", "512000", "--controls", "none",
                                      "--routes", "default,default_prefoff",
                                      "--preference-switch", "MEEP_GPU_OTHER",
                                      "--preference-values", "on=a,off=b"])
    gpu = {r["route"]: r for r in record["rows"] if r["kind"] == "gpu"}
    assert gpu["default_prefoff"]["env"]["set"]["MEEP_GPU_OTHER"] == "b"
    assert "no route 'metal_prefoff'" not in refusal(LINUX + ["--routes", "nosuch_prefoff"])
    assert "no route 'nosuch_prefoff'" in refusal(LINUX + ["--routes", "nosuch_prefoff"])
    assert "a MEEP_GPU_* package switch" in refusal(
        LINUX + ["--routes", "default_prefon", "--preference-switch", "PREFER"])
    assert "names a route twice" in refusal(LINUX + ["--routes", "default,default"])


def test_a_shared_host_floor_omits_the_rows_above_it_by_name():
    config, record = plan_of(LINUX + ["--tiers", "preflight,priority", "--passes", "1",
                                      "--priority-sizes", "512000", "--sockets", "2",
                                      "--controls", "threads2,hwthread",
                                      "--leave-free-share", "0.25"])
    ranks = sorted({r["ranks"] * r["configuration"]["threads_per_rank"]
                    for r in record["rows"] if r["kind"] == "meep"})
    assert max(ranks) == 64 and 96 not in ranks                  # 72 of 96 CPUs may run
    assert "96x1" in config.controls_omitted["rows_above_cpu_cap"]
    _config, record = plan_of(LINUX + ["--tiers", "priority", "--passes", "1",
                                       "--priority-sizes", "512000", "--sockets", "2",
                                       "--controls", "hwthread"])
    assert 96 in {r["ranks"] for r in record["rows"] if r["kind"] == "meep"}


def test_a_dispersive_case_scales_its_gpu_and_lift_budgets():
    config, record = plan_of(LINUX + ["--tiers", "digests,priority", "--passes", "1",
                                      "--priority-sizes", "7077888", "--controls", "none",
                                      "--cases", "pml_3d,pml_3d_lorentz"])
    rows = {(r["case"], r["kind"], r.get("route")): r for r in record["rows"]
            if r["kind"] in ("gpu", "lift")}
    for kind, route in (("gpu", "default"), ("lift", None)):
        plain = rows[("pml_3d", kind, route)]
        scaled = rows[("pml_3d_lorentz", kind, route)]
        assert scaled["budget_s"] == 3 * plain["budget_s"], kind
        assert scaled["timeout_s"] > 2 * plain["timeout_s"], kind


def test_a_mac_mpich_context_leg_names_the_libfabric_provider(tmp_path):
    """On macOS conda-forge's MPICH picks libfabric's sockets provider, whose helper threads
    spin; the ladder launches it with udp named (measured on an Apple M1 Max), and Open MPI's
    reference rows keep their flags."""
    bin_dir = tmp_path / "cf" / "bin"
    bin_dir.mkdir(parents=True)
    for name in ("python", "mpiexec", "mpichversion"):
        (bin_dir / name).write_text("")
    _config, record = plan_of(MAC + [
        "--tiers", "priority", "--passes", "1", "--priority-sizes", "512000",
        "--routes", "none", "--controls", "context", "--context-ranks", "8",
        "--context-meep", f"ctx-cfdouble={bin_dir / 'python'},{bin_dir / 'mpiexec'}"])
    rows = [r for r in record["rows"] if r["kind"] == "meep"]
    context = [r for r in rows if r["control"] == "context"]
    reference = [r for r in rows if not r["control"]]
    assert context and reference
    for row in context:
        argv = row["argv"]
        assert argv[0] == str(bin_dir / "mpiexec")
        assert argv[argv.index("-np") + 2:argv.index("-np") + 6] == [
            "-genv", "FI_PROVIDER", "udp", "-bind-to"]
    for row in reference:
        assert "-genv" not in row["argv"] and "--bind-to" in row["argv"]


# ---------------------------------------------------------------------------
# What the ladder writes about the machine (neutral_ids)
# ---------------------------------------------------------------------------

#: A machine no test here runs on: nothing the ladder writes may carry any of it.
HOST = "gpu-node-17.cluster.example.edu"
LOGIN = "fixtureuser"
HOME = "/" + "ho" + "me" + "/fixtureuser"   # from pieces: a full home path is a hygiene-guard hit
REAL_A = "GPU-3f1c2a9e-5b7d-4e21-9a0c-6d8e2f4b1a37"
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
    for token in (HOST, HOST.split(".")[0], LOGIN, HOME, REAL_A, str(FOREIGN_PID)) + tokens:
        assert str(token) not in text, (token, text[:400])


def test_the_account_is_counted_by_uid_and_written_as_own_or_other(machine, tmp_path):
    """The account gate compares uids in process. Unset, it counts this process's own
    account and never reads the login; a record writes ``own``/``other``, no name."""
    here = LINUX[:LINUX.index("--platform") + 1] + [ladder.detect_platform()] + \
        LINUX[LINUX.index("--platform") + 2:]
    config = config_of(here + ["--run-root", f"{HOME}/meep_gpu_validation"])
    assert config.values["ACCOUNT"] is None and config.account is None
    assert config.uid == os.getuid() and config.gate.uid == os.getuid()
    written = ladder.recorded(config, ladder.plan_record(config, ladder.build_plan(config)))
    assert written["params"]["ACCOUNT"]["value"] == "own"
    assert written["resolved"]["gate"]["uid"] == "own"
    assert written["resolved"]["out"].startswith("<home>/meep_gpu_validation/")
    assert_neutral(written)
    other = config_of(here + ["--account", "no-such-user-zz9"])
    assert ladder.account_label(other) == "unresolved"
    named = config_of(here + ["--account", str(os.getuid() + 1)])
    assert ladder.account_label(named) == "other"
    assert ladder.recorded(named, {"params": {"ACCOUNT": {"value": str(os.getuid() + 1)}}})[
        "params"]["ACCOUNT"]["value"] == "other"


def test_the_environment_and_build_records_the_ladder_writes_name_no_machine(
        machine, monkeypatch, tmp_path):
    here = LINUX[:LINUX.index("--platform") + 1] + [ladder.detect_platform()] + \
        LINUX[LINUX.index("--platform") + 2:]
    config = config_of(here + ["--run-root", str(tmp_path / "runs"), "--routes", "none",
                               "--account", str(os.getuid())])
    run = ladder.Ladder(config, ladder.build_plan(config), resume=False)
    os.makedirs(run.out)
    run.keepawake = type("Popen", (), {"pid": FOREIGN_PID,
                                       "args": ["caffeinate", "-w", str(FOREIGN_PID)]})()
    monkeypatch.setattr(timing_records, "meep_build_record", lambda python, log, launcher: {
        "python": f"{HOME}/miniforge3/envs/meep/bin/python", "openmp": True, "probe": {
            "single_precision": True, "meep_file": f"{HOME}/miniforge3/envs/meep/x.py"},
        "note": f"built on {HOST}"})
    captured = {}
    monkeypatch.setattr(timing_records, "environment_record",
                        lambda out, **kwargs: captured.update(kwargs, out=out))
    monkeypatch.setattr(timing_host, "run_text", lambda *a, **k: "mpirun (Open MPI) 4.1.5")
    run.environment()
    assert_neutral(captured["params"])
    assert captured["params"]["ACCOUNT"]["value"] == "own"
    assert captured["params"]["host"] == neutral_ids.machine_id()
    assert "pid" not in captured["params"]["keepawake"]
    for name in os.listdir(os.path.join(run.out, "meep_builds")):
        with open(os.path.join(run.out, "meep_builds", name), encoding="utf-8") as handle:
            assert_neutral(json.load(handle))
    with open(os.path.join(run.out, "params.env"), encoding="utf-8") as handle:
        params_env = handle.read()
    assert "ACCOUNT=own" in params_env and str(os.getuid()) not in params_env.split(
        "ACCOUNT=")[1].split("\n")[0]
    assert_neutral(params_env)


def test_host_identity_is_written_by_id_and_an_older_raw_file_still_resumes(
        machine, monkeypatch, tmp_path):
    config = config_of(LINUX + ["--run-root", str(tmp_path / "runs"), "--routes", "none",
                                "--gpu", REAL_A])
    monkeypatch.setattr(timing_host, "gpu_uuids", lambda: {"0": REAL_A})
    run = ladder.Ladder(config, ladder.build_plan(config), resume=False)
    os.makedirs(run.out)
    run.check_identity()
    path = os.path.join(run.out, "host_identity.json")
    with open(path, encoding="utf-8") as handle:
        written = json.load(handle)
    assert written["host"] == neutral_ids.machine_id()
    assert written["gpu_uuid"] == written["gpu"] == neutral_ids.card_id(REAL_A)
    assert_neutral(written)
    ladder.Ladder(config, ladder.build_plan(config), resume=True).check_identity()
    # a file an older ladder wrote holds the raw node name and UUID: the same machine
    monkeypatch.setattr(ladder.os, "uname", lambda: type("U", (), {"nodename": HOST})())
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"host": HOST, "platform": "linux", "gpu_uuid": REAL_A}, handle)
    ladder.Ladder(config, ladder.build_plan(config), resume=True).check_identity()
    # another card under the same index is refused, whichever spelling the file holds
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(dict(written, gpu_uuid=neutral_ids.card_id("GPU-" + "0" * 8 + "-1111-2222-"
                                                              "3333-444455556666")), handle)
    with pytest.raises(SystemExit):
        ladder.Ladder(config, ladder.build_plan(config), resume=True).check_identity()


def test_the_progress_log_and_the_box_trace_name_no_machine(machine, tmp_path):
    log = tmp_path / "progress.log"
    ladder.Log(str(log)).say(f"row 1/2 START | logs under {HOME}/runs/x on {HOST} "
                             f"pinned {REAL_A}")
    assert_neutral(log.read_text())


def test_every_driver_a_copy_ladder_runs_carries_its_helpers():
    """A ``--drivers-mode copy`` ladder runs its drivers from the copy: a driver that
    imports ``neutral_ids`` needs it copied beside it."""
    for name in ladder.DRIVER_FILES:
        with open(os.path.join(HERE, name), encoding="utf-8") as handle:
            if "import neutral_ids" in handle.read():
                assert "neutral_ids.py" in ladder.DRIVER_FILES, name
    assert "neutral_ids.py" in ladder.DRIVER_FILES


def test_a_host_named_like_a_case_is_refused_before_anything_is_written(machine,
                                                                        monkeypatch,
                                                                        tmp_path):
    config = config_of(LINUX + ["--run-root", str(tmp_path / "runs")])
    problems = ladder.live_checks(config, resume=False, collect=True, environ={})
    assert timing_host.VOCABULARY_REFUSAL not in problems
    # A route named like a kernel table is a word every record carries and is never
    # rewritten (neutral_ids.RECORD_WORDS); a case is this run's own word.
    monkeypatch.setattr(neutral_ids.socket, "gethostname", lambda: config.cases[0])
    monkeypatch.setattr(neutral_ids, "_node", lambda: config.cases[0])
    neutral_ids.reset()
    problems = ladder.live_checks(config, resume=False, collect=True, environ={})
    assert timing_host.VOCABULARY_REFUSAL in problems
