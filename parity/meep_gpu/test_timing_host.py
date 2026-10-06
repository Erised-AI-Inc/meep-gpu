"""The timing ladder's host readers, fed synthetic text; no GPU, no MEEP, no root.

The first test is the regression test of the 2026-09-28 defect: that ladder's MEEP
quiet gate multiplied a busy FRACTION by ``nproc``, which honours ``OMP_NUM_THREADS=1``,
so on a 96-CPU host the reading could not exceed 1.0 against a threshold of 10 and the
gate could never fail. The busy count here is summed per CPU line and must read the
same whatever the thread environment says.
"""

from __future__ import annotations

import os
import re
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import timing_host as th  # noqa: E402

FIXTURES = os.path.join(HERE, "fixtures", "timing")


def fixture(name: str) -> str:
    with open(os.path.join(FIXTURES, name), "r", encoding="utf-8") as handle:
        return handle.read()


# ---------------------------------------------------------------------------
# CPU busy
# ---------------------------------------------------------------------------

def test_busy_is_summed_per_cpu_whatever_omp_num_threads_says(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    first = th.parse_proc_stat(fixture("proc_stat_96cpu_a.txt"))
    second = th.parse_proc_stat(fixture("proc_stat_96cpu_b.txt"))
    reading = th.busy_from_samples(first, second)
    assert reading["cpus"] == 96
    assert reading["busy_cpus"] == pytest.approx(24.0)
    assert reading["aggregate_cross_check"] == pytest.approx(24.0)
    assert reading["scope"] == "host"
    # The 2026-09-28 reading of the same two samples: the aggregate busy fraction times
    # nproc, which reads 1 under OMP_NUM_THREADS=1.
    fraction = 1.0 - ((second["cpu"][0] - first["cpu"][0])
                      / float(second["cpu"][1] - first["cpu"][1]))
    assert 1 * fraction < 10 <= reading["busy_cpus"]


def test_guest_ticks_do_not_count_twice_and_iowait_counts_as_idle():
    base = "cpu0 100 0 0 1000 0 0 0 0 0 0\n"
    guest = "cpu0 200 0 0 1000 0 0 0 0 100 50\n"     # 100 user ticks, all of them guest
    reading = th.busy_from_samples(th.parse_proc_stat(base), th.parse_proc_stat(guest))
    assert reading["busy_cpus"] == pytest.approx(1.0)
    waiting = "cpu0 100 0 0 1050 50 0 0 0 0 0\n"
    reading = th.busy_from_samples(th.parse_proc_stat(base), th.parse_proc_stat(waiting))
    assert reading["busy_cpus"] == pytest.approx(0.0)


def test_unreadable_or_inconsistent_samples_are_none_and_not_quiet():
    a = th.parse_proc_stat("cpu0 1 0 0 10 0 0 0 0 0 0\ncpu1 1 0 0 10 0 0 0 0 0 0\n")
    same = th.busy_from_samples(a, a)
    assert same["busy_cpus"] is None and "did not advance" in same["why"]
    fewer = th.parse_proc_stat("cpu0 2 0 0 20 0 0 0 0 0 0\n")
    changed = th.busy_from_samples(a, fewer)
    assert changed["busy_cpus"] is None and "changed" in changed["why"]
    config = th.GateConfig(platform="linux", gpu=None, uid=0, cpu_max=10,
                           mem_min_gib=1, account_max=0, cap=None, meep_metric="busy",
                           check_device=False)
    ok, observed = th.quiet(config, "meep", busy_reader=lambda: same,
                            memory_reader=lambda: {"available_gib": 100.0})
    assert ok is False
    assert any("cpu unreadable" in reason for reason in observed["reasons"])


def test_allocation_scope_sums_only_the_allowed_cpus():
    first = th.parse_proc_stat(fixture("proc_stat_96cpu_a.txt"))
    second = th.parse_proc_stat(fixture("proc_stat_96cpu_b.txt"))
    inside = th.busy_from_samples(first, second, scope=range(0, 10))
    assert (inside["cpus"], inside["busy_cpus"]) == (10, pytest.approx(5.0))
    outside = th.busy_from_samples(first, second, scope=range(48, 58))
    assert outside["busy_cpus"] == pytest.approx(0.0)
    missing = th.busy_from_samples(first, second, scope=[200])
    assert missing["busy_cpus"] is None


def test_mach_ticks():
    reading = th.busy_from_ticks((100, 50, 1000, 0), (400, 150, 1500, 100), 10)
    # 1000 ticks elapsed, 500 idle: half of ten CPUs busy, nice counted as busy.
    assert reading["busy_cpus"] == pytest.approx(5.0)
    assert th.busy_from_ticks((1, 1, 1, 1), (1, 1, 1, 1), 10)["busy_cpus"] is None


def test_live_mach_reader_on_darwin_and_refusal_elsewhere():
    if sys.platform == "darwin":
        ticks = th.mach_cpu_ticks()
        assert len(ticks) == 4 and all(isinstance(t, int) and t >= 0 for t in ticks)
    else:
        with pytest.raises(RuntimeError, match="only on macOS"):
            th.mach_cpu_ticks()


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------

def _sysfs(root, sockets=2, cores=24, threads=2, offline=()):
    base = root / "devices" / "system" / "cpu"
    total = sockets * cores * threads
    cpu = 0
    for thread in range(threads):
        for socket in range(sockets):
            for core in range(cores):
                topology = base / f"cpu{cpu}" / "topology"
                topology.mkdir(parents=True)
                (topology / "physical_package_id").write_text(f"{socket}\n")
                (topology / "core_id").write_text(f"{core}\n")
                cpu += 1
    online = [str(c) for c in range(total) if c not in offline]
    (base / "online").write_text(",".join(online) + "\n")
    (base / "smt").mkdir()
    (base / "smt" / "active").write_text("1\n")
    for node in range(sockets):
        path = root / "devices" / "system" / "node" / f"node{node}"
        path.mkdir(parents=True)
        (path / "cpulist").write_text(f"{node * cores}-{node * cores + cores - 1},"
                                      f"{total // 2 + node * cores}-"
                                      f"{total // 2 + node * cores + cores - 1}\n")
    return root


def test_sysfs_topology_two_sockets(tmp_path):
    root = _sysfs(tmp_path)
    found = th.topology_linux(str(root))
    assert (found["physical_cores"], found["logical_cpus"], found["sockets"],
            found["threads_per_core"], found["smt_active"]) == (48, 96, 2, 2, True)
    assert found["numa_cpulists"]["node0"] == "0-23,48-71"
    assert found["package_of_cpu"]["0"] == 0 and found["package_of_cpu"]["24"] == 1
    allocation = th.topology_linux(str(root), allowed=list(range(0, 12)) + list(range(48, 60)))
    assert (allocation["physical_cores"], allocation["logical_cpus"]) == (12, 24)


def test_cpuinfo_fallback_and_cpulist():
    found = th.topology_from_cpuinfo(fixture("cpuinfo_2s2c2t.txt"))
    assert (found["physical_cores"], found["logical_cpus"], found["sockets"]) == (4, 8, 2)
    assert th.parse_cpulist("0-2,5,7-8") == [0, 1, 2, 5, 7, 8]


def test_sysctl_text_and_apple_topology():
    text = ("hw.physicalcpu: 10\nhw.logicalcpu: 10\nhw.perflevel0.physicalcpu: 8\n"
            "hw.perflevel1.physicalcpu: 2\n")
    found = th.topology_darwin(th.parse_sysctl(text))
    assert (found["physical_cores"], found["logical_cpus"], found["efficiency_cores"]) == (
        8, 10, 2)


# ---------------------------------------------------------------------------
# Launchers and bindings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text, implementation", [
    ("mpirun (Open MPI) 5.0.10\n\nReport bugs to https://www.open-mpi.org/community/help/",
     "openmpi"),
    ("HYDRA build details:\n    Version: 4.1.2", "mpich"),
    ("Intel(R) MPI Library for Linux* OS, Version 2021.10", "intelmpi"),
    ("slurm 23.02.7", "srun"),
    ("something else", "unknown"),
    (None, "unknown"),
])
def test_mpi_implementation(text, implementation):
    assert th.mpi_implementation(text) == implementation


def test_flag_templates_and_refusals():
    assert th.binding_flags("openmpi", "bound") == ("--map-by package --bind-to core",
                                                     "bound-core-by-package")
    assert th.binding_flags("openmpi", "over") == ("--use-hwthread-cpus --bind-to none",
                                                    "unbound-hwthreads")
    assert th.binding_flags("openmpi", "bound-pe", 4) == (
        "--map-by package:PE=4 --bind-to core", "bound-core-by-package-pe4")
    assert th.binding_flags("srun", "bound", ranks=16, sockets=2)[0] == (
        "--cpu-bind=cores --ntasks-per-core=1 --ntasks-per-socket=8 "
        "--distribution=block:block")
    with pytest.raises(SystemExit, match="pass --sockets"):
        th.binding_flags("srun", "bound", ranks=16)
    over = th.binding_flags("srun", "over", ranks=64, threads_per_core=2)[0]
    assert over == "--cpu-bind=none --ntasks-per-core=2" and "--hint" not in over
    assert th.binding_flags("srun", "bound-hwthread", ranks=64, sockets=2,
                            threads_per_core=2)[0] == (
        "--cpu-bind=threads --ntasks-per-core=2 --ntasks-per-socket=32 "
        "--distribution=block:block:cyclic")
    assert th.binding_flags("srun", "bound-pe", 2, ranks=24, sockets=2,
                            threads_per_core=2)[0].startswith("--cpus-per-task=4 ")
    assert th.binding_flags("openmpi", "over", platform="darwin") == (
        "--bind-to none", "unbound-ecores")
    assert th.binding_flags("openmpi", "unbound", platform="darwin")[0] == "--bind-to none"
    with pytest.raises(SystemExit, match="Open MPI cannot bind"):
        th.binding_flags("openmpi", "bound", platform="darwin")
    with pytest.raises(SystemExit, match="no flag template"):
        th.binding_flags("mpich", "bound")
    assert th.binding_flags("mpich", "bound", explicit={"bound": "-bind-to core"})[0] == \
        "-bind-to core"
    assert th.with_report("openmpi", "--bind-to none") == "--bind-to none --report-bindings"
    assert th.with_report("srun", "--cpu-bind=cores --distribution=block:block") == \
        "--cpu-bind=verbose,cores --distribution=block:block"
    assert th.binding_mode_for(64, "bound", 48) == "over"
    assert th.binding_mode_for(24, "bound", 48, threads=2) == "bound-pe"
    assert th.binding_mode_for(32, "unbound", 48) == "unbound"


def test_binding_report_lines_read_both_launchers():
    text = ("[gpu-host:308538] Rank 3 bound to package[0][core:L3]\n"
            "[gpu-host:309064] Rank 1 is not bound (or bound to all available processors)\n"
            "cpu-bind=MASK - node01, task  0  0 [12345]: mask 0x1 set\n"
            "unrelated line\n")
    assert len(th.binding_report_lines(text)) == 3


# ---------------------------------------------------------------------------
# Accounts, devices, memory, GPU and CPU state
# ---------------------------------------------------------------------------

def test_account_processes_match_by_uid_not_name():
    status_short = "Name:\tpython\nUid:\t1001\t1001\t1001\t1001\n"
    status_long = "Name:\tpython\nUid:\t1002\t1002\t1002\t1002\n"
    uids = {10: th.parse_status_uid(status_short), 11: th.parse_status_uid(status_long),
            12: th.parse_status_uid(status_long)}
    assert th.count_account_processes([10, 11, 12], 1001, uids.get) == 1
    assert th.count_account_processes([10, 11, 12], 1002, uids.get) == 2
    assert th.count_account_processes([10], None, uids.get) == 0


def test_compute_apps_and_device_neighbours():
    apps = th.parse_compute_apps("GPU-aaaa, 4242, 4970 MiB\nGPU-bbbb, 99, 10 MiB\n")
    assert [a["pid"] for a in apps] == [4242, 99]
    ps = ("100 1 501 /bin/zsh\n"
          "200 100 501 python -u parity/meep_gpu/bench_fused_products.py --drive-table metal\n"
          "300 1 501 bash recut_metal_gates.sh round\n"
          "400 1 502 vim bench_fused_products.py.notes\n")
    found = th.foreign_device_processes(ps, own_pid=200)
    assert len(found) == 1 and "recut_metal_gates.sh" in found[0]


def test_device_pattern_is_a_superset_of_the_bench_pattern():
    with open(os.path.join(HERE, "bench_fused_products.py"), "r", encoding="utf-8") as handle:
        text = handle.read()
    match = re.search(r"_DEVICE_NEIGHBOURS = re\.compile\(\s*(.*?)\)\n", text, re.S)
    assert match, "bench_fused_products._DEVICE_NEIGHBOURS moved"
    spelled = "".join(eval(part) for part in re.findall(r'r"[^"]*"', match.group(1)))  # noqa: S307

    def names(pattern):
        inner = re.search(r"\(\?:\^\|\[/\\s\]\)\(\?:(.*)\)\(\?:\\s\|\$\)$", pattern)
        assert inner, pattern
        return set(inner.group(1).split("|"))

    bench, ours = names(spelled), names(th.DEVICE_NEIGHBOURS.pattern)
    assert bench <= ours, sorted(bench - ours)
    assert {r"bench_timing_case\.py", r"gpu_array_control\.py", r"timing_ladder\.py"} <= ours
    ps = ("100 1 501 /bin/zsh\n"
          "200 1 501 python -u /x/parity/meep_gpu/bench_timing_case.py --drive-table metal\n"
          "300 1 501 python -u /x/parity/meep_gpu/gpu_array_control.py --case pml_3d\n"
          "400 1 501 python -u /x/parity/meep_gpu/timing_ladder.py --stamp s\n")
    assert len(th.foreign_device_processes(ps, own_pid=999)) == 3


def test_memory_parsers():
    meminfo = th.parse_meminfo(fixture("meminfo_linux.txt"))
    assert meminfo["MemAvailable"] == 1029012345
    parsed = th.parse_vm_stat(fixture("vm_stat_apple.txt"))
    assert parsed["page_size"] in (4096, 16384)
    assert th.available_gib_from_vm_stat(parsed) > 0
    assert th.available_gib_from_vm_stat({"pages_free": 10}) is None


def test_nvidia_query_fields_old_and_new_driver():
    new = th.nvidia_query_fields("... clocks_event_reasons.active ...")
    old = th.nvidia_query_fields("... clocks_throttle_reasons.active ...")
    assert "clocks_event_reasons.active" in new
    assert "clocks_throttle_reasons.active" in old
    rows = th.parse_gpu_csv(["index", "clocks.sm"], "0, 1800 MHz\n1, 210 MHz\n")
    assert rows[0] == {"index": "0", "clocks.sm": "1800 MHz"}


def test_cpu_state_reads_governor_turbo_and_frequencies(tmp_path):
    base = tmp_path / "devices" / "system" / "cpu"
    for cpu, (governor, frequency) in enumerate((("performance", 3000000),
                                                 ("performance", 2900000),
                                                 ("powersave", 1000000))):
        policy = base / f"cpu{cpu}" / "cpufreq"
        policy.mkdir(parents=True)
        (policy / "scaling_governor").write_text(governor + "\n")
        (policy / "scaling_cur_freq").write_text(f"{frequency}\n")
        (policy / "cpuinfo_max_freq").write_text("3900000\n")
        (policy / "scaling_driver").write_text("intel_pstate\n")
    (base / "intel_pstate").mkdir()
    (base / "intel_pstate" / "no_turbo").write_text("0\n")
    state = th.cpu_state_linux(str(tmp_path), cpus=[0, 1, 2])
    assert state["governors"] == {"performance": 2, "powersave": 1}
    assert state["intel_pstate_no_turbo"] == "0"
    assert state["scaling_cur_freq_khz"] == {"min": 1000000, "median": 2900000,
                                             "max": 3000000, "cpus": 3}
    assert state["cpuinfo_max_freq_khz"] == 3900000


def test_power_parsers():
    assert th.parse_power("Now drawing from 'AC Power'\n -InternalBattery-0 100%")[
        "on_ac_power"] is True
    assert th.parse_power("Now drawing from 'Battery Power'")["on_ac_power"] is False
    assert th.parse_powermode(" powermode            1\n") == 1


def test_gate_holds_two_readings_and_gives_up_by_name():
    config = th.GateConfig(platform="linux", gpu=None, uid=0, cpu_max=10, mem_min_gib=1,
                           account_max=0, cap=2, meep_metric="busy", hold_meep_s=30,
                           retry_s=15, give_up_s=100, check_device=True)
    clock = [0.0]

    def sleep(seconds):
        clock[0] += seconds

    readers = {"busy_reader": lambda: {"busy_cpus": 0.5},
               "memory_reader": lambda: {"available_gib": 10.0},
               "device_reader": lambda: {"device_free": True, "account_gpu_processes": 0}}
    waited = th.gate(config, "meep", "row", lambda _m: None, lambda: "box", sleep=sleep,
                     clock=lambda: clock[0], **readers)
    assert waited == 30.0
    busy = dict(readers, busy_reader=lambda: {"busy_cpus": 12.0})
    with pytest.raises(th.GateGaveUp):
        th.gate(config, "meep", "row", lambda _m: None, lambda: "box", sleep=sleep,
                clock=lambda: clock[0], **busy)
    capped = dict(readers, device_reader=lambda: {"device_free": True,
                                                  "account_gpu_processes": 2})
    ok, observed = th.quiet(th.GateConfig(**{**config.__dict__, "account_max": 5}), "meep",
                            **capped)
    assert ok is False and any("cap" in r for r in observed["reasons"])


def test_default_thresholds():
    assert th.default_cpu_max(96) == 10
    assert th.default_cpu_max(10) == 1
    assert th.default_mem_min_gib(1007.6) == pytest.approx(251.9)


# ---------------------------------------------------------------------------
# Placement, Slurm, macOS conditions, GPU identity
# ---------------------------------------------------------------------------

#: Two packages x 2 cores x 2 hardware threads: CPUs 0-3 on package 0, 4-7 on 1;
#: CPU c and c+... are siblings as cpuinfo_2s2c2t.txt lays them out.
CORES = {"0": "0:0", "1": "0:0", "2": "0:1", "3": "0:1",
         "4": "1:0", "5": "1:0", "6": "1:1", "7": "1:1"}


def test_affinity_lines_parse():
    text = ("AFFINITY node1 101 [0]\nnoise\nAFFINITY node1 102 [2, 3]\n"
            "AFFINITY node1 103 None\n")
    assert th.parse_affinity_lines(text) == [[0], [2, 3], None]


def test_placement_bound_balanced_and_unbalanced():
    assert th.check_placement("bound", [[0], [2], [4], [6]], 4, CORES, None) == []
    one_socket = th.check_placement("bound", [[0], [2]], 2, CORES, None)
    assert any("not balanced across packages" in p for p in one_socket), one_socket
    shared = th.check_placement("bound", [[0], [1]], 2, CORES, None)
    assert any("share physical cores" in p for p in shared), shared
    wide = th.check_placement("bound", [[0, 2], [4]], 2, CORES, None)
    assert any("more than one physical core" in p for p in wide), wide
    short = th.check_placement("bound", [[0]], 2, CORES, None)
    assert any("1 of 2 ranks answered" in p for p in short), short


def test_placement_hwthread_spreads_over_cores_first():
    assert th.check_placement("bound-hwthread", [[0], [4], [2], [6]], 4, CORES, None) == []
    packed = th.check_placement("bound-hwthread", [[0], [1], [4], [5]], 4, CORES, None)
    assert any("cores are left idle" in p for p in packed), packed


def test_placement_pe_and_unbound():
    assert th.check_placement("bound-pe", [[0, 1, 2, 3], [4, 5, 6, 7]], 2, CORES, None,
                              threads=2) == []
    narrow = th.check_placement("unbound", [list(range(8)), [0, 1]], 2, CORES,
                                list(range(8)))
    assert any("confined to fewer CPUs" in p for p in narrow), narrow
    assert th.check_placement("over", [list(range(8))] * 3, 3, CORES, list(range(8))) == []


def test_slurm_refusals():
    assert th.slurm_refusals({}, "openmpi", "mpirun", 64, 8) == []
    login = {"SLURM_JOB_ID": "7", "SLURM_CPUS_ON_NODE": "96", "SLURM_TASKS_PER_NODE": "96"}
    found = th.slurm_refusals(login, "openmpi", "mpirun", 64, 96)
    assert len(found) == 1 and "not on the allocated node" in found[0]
    environ = {"SLURM_JOB_ID": "7", "SLURM_CPUS_ON_NODE": "96",
               "SLURM_TASKS_PER_NODE": "48(x1)", "SLURMD_NODENAME": "gpu01"}
    problems = th.slurm_refusals(environ, "openmpi", "mpirun", 64, 96)
    assert len(problems) == 1 and "48 task slots" in problems[0]
    assert th.slurm_refusals(environ, "srun", "srun", 64, 96) == []
    step = dict(environ, SLURM_STEP_ID="0", SLURM_TASKS_PER_NODE="96")
    found = th.slurm_refusals(step, "openmpi", "mpirun", 64, 2)
    assert any("job step 0" in p for p in found) and any("fraction" in p for p in found)
    batch = dict(environ, SLURM_STEP_ID="batch", SLURM_TASKS_PER_NODE="96")
    assert th.slurm_refusals(batch, "openmpi", "mpirun", 64, 96) == []
    assert th.first_node_count("48(x2),12") == 48


def test_darwin_gate_reasons_respect_the_ac_power_choice():
    config = th.GateConfig(platform="darwin", gpu=None, uid=0, cpu_max=1, mem_min_gib=1,
                           account_max=0, cap=None, meep_metric="busy", gpu_metric="busy")
    quiet_readers = {"busy_reader": lambda: {"busy_cpus": 0.2},
                     "memory_reader": lambda: {"available_gib": 10.0},
                     "device_reader": lambda: {"device_free": True,
                                               "account_gpu_processes": 0},
                     "power_reader": lambda: {"on_ac_power": True, "low_power_mode": False,
                                              "cpu_speed_limit": None},
                     "daemon_reader": lambda: []}
    assert th.quiet(config, "meep", **quiet_readers)[0] is True
    battery = dict(quiet_readers, power_reader=lambda: {
        "on_ac_power": False, "power_source": "Battery Power", "low_power_mode": True,
        "cpu_speed_limit": None})
    ok, observed = th.quiet(config, "meep", **battery)
    assert ok is False and "on Battery Power" in observed["reasons"]
    assert "low power mode" in observed["reasons"]
    allowed = th.GateConfig(**{**config.__dict__, "require_ac_power": False})
    assert th.quiet(allowed, "meep", **battery)[0] is True
    hot = dict(quiet_readers, power_reader=lambda: {"on_ac_power": True,
                                                    "cpu_speed_limit": 80})
    assert any("CPU speed limit 80" in r for r in th.quiet(config, "meep", **hot)[1]["reasons"])
    daemons = dict(quiet_readers, daemon_reader=lambda: ["501 mds_stores 55%"])
    assert th.quiet(config, "meep", **daemons)[0] is False
    unread = dict(quiet_readers, daemon_reader=lambda: None)
    assert th.quiet(config, "meep", **unread)[0] is False
    gated = th.GateConfig(**{**config.__dict__, "apple_gpu_util_max": 5.0})
    busy_gpu = dict(quiet_readers, apple_gpu_reader=lambda: {
        "state": {"device_utilization_percent": 20}})
    assert th.quiet(gated, "gpu", **busy_gpu)[0] is False
    assert th.quiet(gated, "meep", **busy_gpu)[0] is True


def test_darwin_daemon_and_ioreg_parsers():
    rows = th.parse_ps_cpu("  1   0.0 /sbin/launchd\n 77  55.1 /System/Library/Frameworks/"
                           "CoreServices.framework/Frameworks/Metadata.framework/Support/"
                           "mds_stores\n 78  90.0 /usr/bin/python3\n 79 12.0 mdworker_shared\n")
    found = th.busy_daemons(rows)
    assert len(found) == 2 and "mds_stores" in found[0] and "mdworker_shared" in found[1]
    text = ('  "PerformanceStatistics" = {"Alloc system memory"=12658720768,'
            '"Renderer Utilization %"=19,"Device Utilization %"=20,'
            '"In use system memory"=326942720}\n  "model" = "Apple M1 Max"\n'
            '  "gpu-core-count" = 32\n')
    parsed = th.parse_ioreg_accelerator(text)
    assert parsed["model"] == "Apple M1 Max" and parsed["gpu_core_count"] == 32
    assert parsed["device_utilization_percent"] == 20
    assert parsed["in_use_system_memory_bytes"] == 326942720


def test_gpu_identity_by_uuid():
    uuids = th.parse_index_uuid("0, GPU-aaaa-1111\n1, GPU-bbbb-2222\n")
    assert uuids == {"0": "GPU-aaaa-1111", "1": "GPU-bbbb-2222"}
    assert th.as_uuid("1", uuids) == "GPU-bbbb-2222"
    assert th.as_uuid("GPU-cccc", uuids) == "GPU-cccc"
    assert th.as_uuid("7", uuids) is None


def test_a_gpu_pinned_by_uuid_is_keyed_by_its_index_in_the_box(monkeypatch):
    monkeypatch.setattr(th, "gpu_uuids", lambda: {"0": "GPU-aaaa", "3": "GPU-dddd"})
    th._UUID_INDEX.clear()
    assert th.box_label("GPU-dddd") == "3"
    assert th.box_label("2") == "2"
    assert th.box_label("GPU-ffff") == "GPU-ffff"
    th._UUID_INDEX.clear()
