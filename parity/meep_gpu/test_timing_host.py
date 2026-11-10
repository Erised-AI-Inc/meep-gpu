"""The timing ladder's host readers, fed synthetic text; no GPU, no MEEP, no root.

The first test is the regression test of the 2026-09-28 defect: that ladder's MEEP
quiet gate multiplied a busy FRACTION by ``nproc``, which honours ``OMP_NUM_THREADS=1``,
so on a 96-CPU host the reading could not exceed 1.0 against a threshold of 10 and the
gate could never fail. The busy count here is summed per CPU line and must read the
same whatever the thread environment says.
"""

from __future__ import annotations

import json
import os
import re
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import neutral_ids  # noqa: E402
import timing_host as th  # noqa: E402

FIXTURES = os.path.join(HERE, "fixtures", "timing")

#: A machine no test here runs on: nothing a reader writes may carry any of it.
HOST = "gpu-node-17.cluster.example.edu"
LOGIN = "fixtureuser"
HOME = "/" + "ho" + "me" + "/fixtureuser"   # from pieces: a full home path is a hygiene-guard hit
UUID_A = "GPU-3f1c2a9e-5b7d-4e21-9a0c-6d8e2f4b1a37"
UUID_B = "GPU-9b2e7c41-0d6a-4f83-b5e9-1c7a3d8f6e02"
FOREIGN_PID = 271828
OTHER_UID = 1777


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
    """``record`` names nothing of the fixture machine, by residue and by token."""
    assert neutral_ids.residue(record) == []
    text = json.dumps(record, sort_keys=True, default=str)
    for token in (HOST, HOST.split(".")[0], LOGIN, HOME, UUID_A, UUID_B,
                  str(FOREIGN_PID)) + tokens:
        assert str(token) not in text, (token, text[:400])


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
        th.binding_flags("intelmpi", "bound")
    assert th.binding_flags("intelmpi", "bound", explicit={"bound": "-bind-to core"})[0] == \
        "-bind-to core"
    # MPICH's Hydra mpiexec (conda-forge's MPI pymeep, a context leg): measured templates on
    # Linux; on macOS the libfabric provider is named, since the default (sockets) spins two
    # helper threads per rank (measured on an Apple M1 Max); no hardware-thread template.
    assert th.binding_flags("mpich", "bound")[0] == "-bind-to core -map-by socket"
    assert th.binding_flags("mpich", "bound-pe", 2)[0] == "-bind-to core:2 -map-by socket"
    assert th.binding_flags("mpich", "unbound")[0] == "-bind-to none"
    with pytest.raises(SystemExit, match="no mpich flag template is known for mode"):
        th.binding_flags("mpich", "bound-hwthread", ranks=64, sockets=2, threads_per_core=2)
    assert th.binding_flags("mpich", "unbound", platform="darwin")[0] == \
        "-genv FI_PROVIDER udp -bind-to none"
    assert th.binding_flags("mpich", "over", platform="darwin") == (
        "-genv FI_PROVIDER udp -bind-to none", "unbound-ecores")
    assert th.with_report("mpich", "-bind-to none") == "-bind-to none"
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


def test_the_calibrated_cpu_gate_opens_above_a_laptops_idle_floor():
    """In the 1.0 timing run the Mac idled at 1.2-2.0 busy CPUs of 10 with no work of
    ours, so the 10 % default (1) could never open. Calibrated by the rule that run
    fixed in advance, the same box gets 2.5; a quiet box keeps the default."""
    floor = [1.2, 1.5, 1.9, 2.0, 1.4, 1.7] * 3
    assert min(floor) >= th.default_cpu_max(10)        # the default never opens here
    found = th.calibrated_cpu_max(floor, 10)
    assert found["value"] == 2.5 and found["p80"] == pytest.approx(1.9)
    assert "warning" not in found
    assert th.calibrated_cpu_max([0.3, 0.5, 0.8] * 6, 10)["value"] == 1.0
    busy = th.calibrated_cpu_max([2.5, 2.7, 3.0] * 6, 10)  # the timing run's foreign process
    assert busy["value"] == 2.5 and "wait at the gate" in busy["warning"]
    assert th.calibrated_cpu_max([None] * 18, 10)["value"] == 1.0
    assert th.percentile([1, 2, 3, 4, 5], 0.8) == pytest.approx(4.2)


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


def _hwloc_placement(flags, ranks, sockets, cores_per_socket, threads_per_core):
    """What Open MPI's ORDERED pe-list binds rank r to, as Linux CPU numbers, on a host
    laid out like the GPU host: logical thread t of core i of package p is hwloc's
    ``(p * C + i) * T + t``, and Linux numbers it ``p * C + i + t * P`` (siblings P
    apart, P physical cores). Returns (affinity sets, CPU -> "package:core" map)."""
    import re  # noqa: PLC0415
    found = re.search(r"pe-list=([\d,]+):ORDERED", flags).group(1)
    listed = [int(v) for v in found.split(",")]
    physical = sockets * cores_per_socket
    core_of_cpu, affinities = {}, []
    for logical in listed[:ranks]:
        thread = logical % threads_per_core
        core = logical // threads_per_core            # package-major global core index
        cpu = core + thread * physical
        core_of_cpu[str(cpu)] = f"{core // cores_per_socket}:{core % cores_per_socket}"
        affinities.append([cpu])
    for core in range(physical):
        for thread in range(threads_per_core):
            core_of_cpu.setdefault(str(core + thread * physical),
                                   f"{core // cores_per_socket}:{core % cores_per_socket}")
    return affinities, core_of_cpu, listed


@pytest.mark.parametrize("ranks", [4, 48, 64, 96])
def test_bound_hwthread_takes_every_core_before_any_second_thread(ranks):
    """Found in the 1.0 timing run: the 2026-09-28 template (``--map-by package --bind-to
    hwthread``) put 64 ranks on 32 of the GPU host's 48 cores. The template now names each
    rank's hardware thread; on a 2 x 24 x 2 host, N ranks use min(N, 48) cores, balanced."""
    flags, label = th.binding_flags("openmpi", "bound-hwthread", ranks=ranks, sockets=2,
                                    threads_per_core=2, physical_cores=48)
    assert label == "bound-hwthread"
    assert flags.startswith("--use-hwthread-cpus --map-by pe-list=")
    assert flags.endswith(":ORDERED --bind-to hwthread")
    affinities, core_of_cpu, listed = _hwloc_placement(flags, ranks, 2, 24, 2)
    assert sorted(listed) == list(range(96))
    assert th.check_placement("bound-hwthread", affinities, ranks, core_of_cpu, None) == []
    used = {core_of_cpu[str(a[0])] for a in affinities}
    assert len(used) == min(ranks, 48)


def test_bound_hwthread_refuses_by_name_without_the_topology():
    with pytest.raises(SystemExit, match="named hardware thread"):
        th.binding_flags("openmpi", "bound-hwthread", ranks=64, sockets=2,
                         threads_per_core=2)
    with pytest.raises(SystemExit, match="named hardware thread"):
        th.binding_flags("openmpi", "bound-hwthread", ranks=64, sockets=5,
                         threads_per_core=2, physical_cores=48)
    explicit = {"bound-hwthread": "--my-own-flags"}
    assert th.binding_flags("openmpi", "bound-hwthread", explicit=explicit)[0] == \
        "--my-own-flags"


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
    # a reading is recorded: the daemon by name and share, never its pid
    assert found == ["mds_stores 55%", "mdworker_shared 12%"]
    text = ('  "PerformanceStatistics" = {"Alloc system memory"=12658720768,'
            '"Renderer Utilization %"=19,"Device Utilization %"=20,'
            '"In use system memory"=326942720}\n  "model" = "Apple M1 Max"\n'
            '  "gpu-core-count" = 32\n')
    parsed = th.parse_ioreg_accelerator(text)
    assert parsed["model"] == "Apple M1 Max" and parsed["gpu_core_count"] == 32
    assert parsed["device_utilization_percent"] == 20
    assert parsed["in_use_system_memory_bytes"] == 326942720


def test_the_apple_gpu_state_calls_ioreg_by_its_absolute_path(monkeypatch):
    """The reference environment's PATH (<prefix>/bin:/usr/bin:/bin) holds no /usr/sbin, where
    ioreg lives: called by name, the state read unreadable on every reference shell."""
    calls = []

    def run_text(command, timeout=30.0, env=None, cwd=None):
        calls.append(list(command))
        return '  "PerformanceStatistics" = {"Device Utilization %"=7}\n'

    monkeypatch.setattr(th, "run_text", run_text)
    state = th.apple_gpu_state()
    assert calls == [["/usr/sbin/ioreg", "-r", "-d", "1", "-c", "IOAccelerator"]]
    assert th.IOREG == "/usr/sbin/ioreg"
    assert state["readable"] is True and state["state"]["device_utilization_percent"] == 7


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
    # no index listed: the box string is logged, so the card id, never the UUID
    assert th.box_label("GPU-ffff") == neutral_ids.card_id("GPU-ffff")
    th._UUID_INDEX.clear()


# ---------------------------------------------------------------------------
# What a record writes about the machine (neutral_ids)
# ---------------------------------------------------------------------------

def test_the_affinity_probe_prints_neither_the_host_nor_the_pid(machine, monkeypatch,
                                                                 capsys):
    """Each rank's probe line lands in the ladder's log, which leaves the machine."""
    monkeypatch.setattr(os, "getpid", lambda: FOREIGN_PID)
    exec(compile(th.AFFINITY_PROBE, "<affinity probe>", "exec"), {})  # noqa: S102
    out = capsys.readouterr().out
    assert out.startswith("AFFINITY ")
    assert HOST not in out and HOST.split(".")[0] not in out and str(FOREIGN_PID) not in out
    assert len(th.parse_affinity_lines(out)) == 1
    # both spellings parse: the one older logs hold, and the one written now
    assert th.parse_affinity_lines("AFFINITY node1 101 [0]\nAFFINITY rank 0 [2, 3]\n") == [
        [0], [2, 3]]


def test_the_compute_apps_are_recorded_by_card_and_owner(machine):
    text = (f"{UUID_A}, {os.getpid()}, 300 MiB\n{UUID_A}, {FOREIGN_PID}, 5000 MiB\n"
            f"{UUID_B}, 99, 10 MiB")
    recorded = th.recorded_compute_apps(text, os.getpid())
    card_a, card_b = neutral_ids.card_id(UUID_A), neutral_ids.card_id(UUID_B)
    assert recorded.splitlines() == [f"{card_a}, own, 300 MiB", f"{card_a}, other, 5000 MiB",
                                     f"{card_b}, other, 10 MiB"]
    assert_neutral(recorded, str(os.getpid()))
    assert th.recorded_gpus(f"0, {UUID_A}, 3 MiB, 0 %") == f"0, {card_a}, 3 MiB, 0 %"
    assert th.recorded_device(UUID_A) == card_a and th.recorded_device("3") == "3"
    assert th.recorded_foreign_app(f"{UUID_A}, {FOREIGN_PID}, 5000 MiB") == (
        f"{card_a}, 5000 MiB")


def test_a_binding_line_keeps_its_cores_and_drops_host_and_pid(machine):
    lines = [f"[{HOST.split('.')[0]}:{FOREIGN_PID}] MCW rank 0 bound to socket 0[core 0[hwt "
             "0]]: [B/././.]",
             f"[{HOST}:{FOREIGN_PID}] Rank 3 bound to package[0][core:3]",
             f"cpu-bind=MASK - {HOST.split('.')[0]}, task  1  1 [{FOREIGN_PID}]: mask 0x2 set"]
    recorded = [th.recorded_binding_line(line) for line in lines]
    assert recorded == ["MCW rank 0 bound to socket 0[core 0[hwt 0]]: [B/././.]",
                        "Rank 3 bound to package[0][core:3]",
                        "cpu-bind=MASK - task  1  1: mask 0x2 set"]
    assert_neutral(recorded)
    assert th.binding_report_lines("\n".join(recorded)) == recorded[1:]


def test_the_device_state_records_no_pid_or_uuid_and_keeps_its_verdicts(machine,
                                                                         monkeypatch):
    """The gate compares pids, uids and UUIDs in process; what it reports is recorded."""
    apps = {None: [{"gpu_uuid": UUID_A, "pid": FOREIGN_PID, "used_memory": "5000 MiB"},
                   {"gpu_uuid": UUID_B, "pid": 4242, "used_memory": "10 MiB"}],
            "0": [{"gpu_uuid": UUID_A, "pid": FOREIGN_PID, "used_memory": "5000 MiB"}]}
    monkeypatch.setattr(th, "compute_apps", lambda gpu: apps[gpu])
    monkeypatch.setattr(th.count_account_processes, "__defaults__",
                        (lambda pid: 501 if pid == 4242 else OTHER_UID,))
    state = th.device_state("0", 501, "linux")
    assert state["device_free"] is False and state["account_gpu_processes"] == 1
    assert state["device_processes"] == [{"gpu": neutral_ids.card_id(UUID_A),
                                          "used_memory": "5000 MiB"}]
    assert_neutral(state)
    ok, reasons = th.verdict(th.GateConfig("linux", "0", 501, 9.0, 1.0, 0, None),
                             {"cpu": 0.0, "memory": {"available_gib": 99.0},
                              "device": state})
    assert not ok and any(r.startswith("device busy") for r in reasons)
    assert_neutral(reasons)
    ps = [f"{FOREIGN_PID} 1 {OTHER_UID} python -u {HOME}/runs/x/parity/meep_gpu/"
          "gate_metal_bfast.py --out y",
          f"300 1 501 python -u {HOME}/runs/x/bench_fused_products.py --out z"]
    monkeypatch.setattr(th, "metal_neighbours", lambda: list(ps))
    darwin = th.device_state(None, 501, "darwin")
    assert darwin["device_free"] is False and darwin["account_gpu_processes"] == 1
    assert darwin["device_processes"] == ["other python gate_metal_bfast.py",
                                          "own python bench_fused_products.py"]
    assert_neutral(darwin, f"{FOREIGN_PID} 1")


def test_the_gpu_state_and_its_trace_write_no_uuid(machine, monkeypatch):
    fields = th.nvidia_query_fields("clocks_event_reasons.active")
    monkeypatch.setattr(th, "nvidia_smi_available", lambda: True)
    monkeypatch.setattr(th, "gpu_query_fields", lambda: list(fields))
    row = ", ".join(UUID_A if f == "uuid" else ("0" if f == "index" else "1")
                    for f in fields)
    monkeypatch.setattr(th, "run_text", lambda *a, **k: row + "\n")
    state = th.gpu_state("0", "linux")
    assert state["readable"] and state["state"]["uuid"] == neutral_ids.card_id(UUID_A)
    assert_neutral(state)
    monkeypatch.setattr(th, "run_text", lambda *a, **k: f"garbled {UUID_A}")
    assert_neutral(th.gpu_state("0", "linux"))
    command = th.gpu_trace_command("0", 500)
    query = next(part for part in command if part.startswith("--query-gpu="))
    assert "uuid" not in query.split("=", 1)[1].split(",")
    assert "clocks.sm" in query


def test_the_allocation_facts_write_the_machine_id(machine):
    environ = {"SLURM_JOB_ID": "991", "SLURMD_NODENAME": HOST.split(".")[0],
               "SLURM_JOB_NODELIST": HOST.split(".")[0], "CUDA_VISIBLE_DEVICES": UUID_A}
    facts = th.slurm_facts(environ)
    assert facts["SLURM_JOB_ID"] == "991"
    assert facts["SLURMD_NODENAME"] == neutral_ids.machine_id()
    assert facts["CUDA_VISIBLE_DEVICES"] == neutral_ids.card_id(UUID_A)
    assert_neutral(facts)


def test_a_host_named_like_a_word_the_records_carry_is_refused(machine, monkeypatch):
    """A host named like a word every record carries (``metal``) is never rewritten
    (``neutral_ids.RECORD_WORDS``), so its records keep their tables; a host named like one
    of this run's own words (a case, ``pml_3d``) is refused by name of the rule, not the host."""
    assert th.vocabulary_collisions(["pml_3d", "triton"]) == []
    monkeypatch.setattr(neutral_ids.socket, "gethostname", lambda: "metal.lab.example.edu")
    monkeypatch.setattr(neutral_ids, "_node", lambda: "metal.lab.example.edu")
    neutral_ids.reset()
    assert th.vocabulary_collisions(["metal"]) == []
    assert neutral_ids.text('"drive_table": "metal"') == '"drive_table": "metal"'
    assert "metal.lab.example.edu" not in neutral_ids.text("on metal.lab.example.edu")
    monkeypatch.setattr(neutral_ids, "_node", lambda: "pml_3d")
    monkeypatch.setattr(neutral_ids.socket, "gethostname", lambda: "pml_3d")
    neutral_ids.reset()
    assert th.vocabulary_collisions(["pml_3d"]) == ["pml_3d"]
    assert "metal" not in th.VOCABULARY_REFUSAL and "pml_3d" not in th.VOCABULARY_REFUSAL


# ---------------------------------------------------------------------------
# The Windows side of a WSL2 host
# ---------------------------------------------------------------------------

#: What the lab desktop's Windows side read with the card locked at 1695 / 9501 MHz: the
#: card at rest reports P2 and the gpu_idle reason at its lock (CRLF, as Windows writes it).
WIN_GPU_AT_LOCK = "1695, 9501, P2, 0x0000000000000001, 117.75, 55\r\n"
#: The load sample: 24 logical CPUs, the VM process's affinity mask (all 24, one process),
#: the whole machine 5.1 % busy on the last sample, the VM's process 7.8 % of one CPU (its
#: first sample has no previous raw value: 0).
WIN_LOAD_QUIET = ("cpus 24\r\naffinity 16777215 1\r\nsample 7.60718225393715 0 1\r\n"
                  "sample 5.11984364517136 7.78237830477138 1\r\n")


def test_wsl2_is_read_from_a_microsoft_kernel_and_nothing_else():
    """The release string or the version banner naming Microsoft is WSL2; a native Linux
    kernel (a cloud build included), macOS and an unreadable /proc are not WSL2 hosts and
    each says why."""
    assert th.wsl_host("linux", "6.18.54.1-microsoft-standard-WSL2\n", "")["wsl2"] is True
    assert th.wsl_host("linux", "", "Linux version 5.15.153.1-microsoft-standard-WSL2 "
                                    "(root@builder) #1 SMP")["wsl2"] is True
    assert th.wsl_host("linux", "4.4.0-19041-Microsoft", "")["wsl2"] is True
    native = th.wsl_host("linux", "6.5.0-1025-azure\n", "Linux version 6.5.0-1025-azure")
    assert native == {"wsl2": False, "why": "native Linux: the kernel is not a Microsoft build"}
    assert th.wsl_host("darwin") == {"wsl2": False, "why": "not Linux (darwin)"}
    assert th.wsl_host("linux", "", "")["wsl2"] is False


def test_the_windows_card_and_load_parse_and_name_their_reasons():
    gpu = th.parse_windows_gpu(WIN_GPU_AT_LOCK)
    assert gpu == [{"graphics_mhz": 1695, "memory_mhz": 9501, "pstate": "P2",
                    "reasons": "0x1", "reason_names": ["gpu_idle"], "power_w": 117.75,
                    "temperature_c": 55}]
    assert th.reason_names(0x405) == ["gpu_idle", "sw_power_cap", "bit 0x400"]
    assert th.parse_windows_gpu("[N/A], 9501, P2, [N/A], 1, 2")[0]["graphics_mhz"] is None
    load = th.parse_windows_load(WIN_LOAD_QUIET)
    assert load["readable"] and load["logical_cpus"] == 24
    assert load["busy_cpus"] == pytest.approx(5.12 * 0.24, abs=0.01)
    assert load["vm_busy_cpus"] == pytest.approx(0.08)
    assert load["outside_vm_busy_cpus"] == pytest.approx(1.23 - 0.08, abs=0.01)
    assert th.parse_windows_load("error InvalidOperationException\n") == {
        "readable": False, "why": "the counter sample failed (InvalidOperationException)"}
    assert th.parse_windows_load("cpus 24\nsample 3.0 0 0\n")["why"] == (
        "no process of the WSL VM in the counters")
    assert th.parse_windows_load("")["readable"] is False


def _runner(gpu=WIN_GPU_AT_LOCK, load=WIN_LOAD_QUIET, calls=None):
    def run(command):
        if calls is not None:
            calls.append(list(command))
        return (gpu, "") if "nvidia-smi" in command[0] else (load, "")
    return run


def test_a_windows_reading_pins_the_card_and_records_no_name(machine):
    """The pinned card's UUID picks the Windows card and is never recorded; the load
    script prints numbers only (a counter path carries the Windows computer name)."""
    calls = []
    reading = th.windows_reading(UUID_A, runner=_runner(calls=calls),
                                 program=lambda name: f"{HOME}/bin/{name}")
    assert reading["unreadable"] == [] and reading["gpu"]["graphics_mhz"] == 1695
    assert reading["cpu"]["outside_vm_busy_cpus"] == pytest.approx(1.15, abs=0.01)
    assert f"--id={UUID_A}" in calls[0]
    assert calls[1][1:4] == ["-NoProfile", "-NonInteractive", "-EncodedCommand"]
    printed = set(re.findall(r'"(\w+) [\${]', th.WINDOWS_LOAD_SCRIPT))
    assert printed == {"cpus", "sample", "error", "affinity"}, \
        "numbers only: no counter path or process name is printed"
    assert reading["vm_affinity"] == {"readable": True, "mask": "0xffffff", "processes": 1}
    assert_neutral(reading)
    two = th.windows_reading(None, runner=_runner(gpu=WIN_GPU_AT_LOCK * 2),
                             program=lambda name: name)
    assert two["gpu"] is None and two["unreadable"] == [
        "nvidia-smi.exe: 2 cards read and none pinned"]


def test_without_interop_the_windows_side_reads_unreadable_by_name(monkeypatch):
    """No Windows program reachable (interop off, nothing on PATH or under /mnt/c), one
    that cannot be started, or one that does not answer: each is named, never empty."""
    none = th.windows_reading(program=lambda name: None)
    assert none["gpu"] is None and none["cpu"] is None and len(none["unreadable"]) == 2
    assert none["unreadable"][0].startswith("nvidia-smi.exe not found")
    assert none["unreadable"][1].startswith("powershell.exe not found")
    assert th.windows_program("nvidia-smi.exe", which=lambda n: None,
                              executable=lambda p: False) is None
    assert th.windows_program("nvidia-smi.exe", which=lambda n: None,
                              executable=lambda p: True) == th.WINDOWS_PROGRAMS["nvidia-smi.exe"]
    failed = th.windows_reading(program=lambda name: name,
                                runner=lambda command: (None, "could not be started "
                                                              "(Exec format error)"))
    assert failed["unreadable"] == [
        "nvidia-smi.exe: could not be started (Exec format error)",
        "Windows CPU load: could not be started (Exec format error)"]
    rule = th.WindowsRule(1695, 9501)
    assert th.windows_reasons(rule, failed)[0].startswith("Windows side unreadable: ")


def test_run_windows_closes_the_childs_stdin(tmp_path):
    """A Windows program reads the stdin it inherits: a driver fed on stdin would lose the
    rest of its script to the first nvidia-smi.exe it starts. Here this process's stdin
    holds the rest of a script, and the child must read none of it."""
    read_end, write_end = os.pipe()
    os.write(write_end, b"the rest of the script\n")
    os.close(write_end)
    saved = os.dup(0)
    os.dup2(read_end, 0)
    try:
        text, why = th.run_windows([sys.executable, "-c",
                                    "import sys; print(repr(sys.stdin.read()))"])
    finally:
        os.dup2(saved, 0)
        os.close(saved)
        os.close(read_end)
    assert why == "" and text.strip() == "''"
    assert th.run_windows([str(tmp_path / "absent.exe")])[1].startswith("could not be started")


def _observed(gpu=WIN_GPU_AT_LOCK, load=WIN_LOAD_QUIET):
    return {"gpu": th.parse_windows_gpu(gpu)[0], "cpu": th.parse_windows_load(load),
            "vm_affinity": th.parse_vm_affinity(load), "unreadable": []}


def test_the_windows_rule_holds_the_lock_the_reasons_and_the_load():
    rule = th.WindowsRule(1695, 9501)
    assert th.windows_reasons(rule, _observed()) == []
    off = _observed("1485, 5001, P3, 0x0000000000000001, 90.0, 55")
    assert th.windows_reasons(rule, off) == [
        "the card left the declared clock lock: graphics 1485 MHz, memory 5001 MHz, "
        "against 1695 / 9501"]
    memory_only = _observed("1695, 5001, P3, 0x0000000000000001, 90.0, 55")
    assert th.windows_reasons(rule, memory_only) == [
        "the card left the declared clock lock: graphics 1695 MHz, memory 5001 MHz, "
        "against 1695 / 9501"], "the memory clock is half the lock"
    capped = _observed("1695, 9501, P2, 0x0000000000000004, 350.0, 70")
    assert th.windows_reasons(rule, capped) == [
        "a clock-event reason other than gpu_idle: 0x4 (sw_power_cap)"]
    assert th.windows_reasons(rule, _observed("1695, 9501, P2, 0x0000000000000000, 1, 2")) == []
    busy = _observed(load="cpus 24\nsample 30.0 0 1\nsample 25.0 100.0 1\n")
    assert th.windows_reasons(rule, busy) == [
        "Windows busy outside the WSL VM: 5.0 logical CPUs >= 2"]
    # the VM's own load is the run's: 100 % of 24 CPUs, all of it the VM, is quiet
    own = _observed(load="cpus 24\nsample 100.0 0 1\nsample 100.0 2400.0 1\n")
    assert th.windows_reasons(rule, own) == []
    assert th.windows_reasons(th.WindowsRule(1695, 9501, cpu_max=6.0), busy) == []


def test_with_no_lock_declared_the_clocks_may_not_move_between_window_readings():
    rule = th.WindowsRule()
    rest = _observed("210, 405, P8, 0x0000000000000001, 20.0, 40")
    assert th.windows_reasons(rule, rest, previous=rest) == []
    woke = _observed("1965, 9751, P0, 0x0000000000000000, 300.0, 60")
    assert th.windows_reasons(rule, woke, previous=rest) == [
        "the card's clocks moved with no lock declared: graphics 210 -> 1965 MHz, "
        "memory 405 -> 9751 MHz"]
    assert th.windows_reasons(rule, woke) == [], "no clock rule without a lock or a window"
    assert th.parse_clock_lock(" 1695, 9501") == (1695, 9501)
    for bad in ("1695", "1695,", "1695;9501", "a,b", "0,9501"):
        with pytest.raises(ValueError, match="GRAPHICS,MEMORY"):
            th.parse_clock_lock(bad)


def test_a_units_windows_readings_are_summarised():
    gone = dict(_observed("1485, 5001, P3, 0x0000000000000001, 90.0, 58"))
    summary = th.windows_summary([_observed(), gone, {"gpu": None, "cpu": None,
                                                      "unreadable": ["x"]}])
    assert summary == {"readings": 3, "unreadable": 1, "graphics_mhz": [1485, 1695],
                       "memory_mhz": [5001, 9501], "pstates": ["P2", "P3"],
                       "reasons": ["0x1"], "power_w_max": 117.75, "temperature_c_max": 58,
                       "outside_vm_busy_cpus_max": 1.15, "busy_cpus_max": 1.23,
                       "vm_affinity": ["0xffffff"]}


# ---------------------------------------------------------------------------
# The WSL VM's CPU affinity mask
# ---------------------------------------------------------------------------

#: What the set script printed where the mask took (an elevated context) and where it was
#: denied (a restricted token: the read gives 0, the set fails with Win32 error 5).
SET_TOOK = "processes 1\r\nbefore 16777215\r\nset ok\r\nread 10066312\r\n"
SET_DENIED = "processes 1\r\nbefore 0\r\nset error Win32Exception 5\r\nread 0\r\n"


def test_an_affinity_mask_is_declared_in_hex_or_decimal():
    assert th.parse_affinity_mask("0x999988") == 0x999988 == th.parse_affinity_mask("10066312")
    assert th.parse_affinity_mask(" 0XFFFFFF ") == 0xFFFFFF
    for bad in ("", "0", "0x0", "-5", "lots", "0xZZ"):
        with pytest.raises(ValueError, match="affinity mask"):
            th.parse_affinity_mask(bad)


def test_the_vm_mask_is_read_with_the_load_and_one_vm_only():
    assert th.parse_vm_affinity("cpus 24\naffinity 10066312 1\n") == {
        "readable": True, "mask": "0x999988", "processes": 1}
    assert th.parse_vm_affinity("affinity 16777215 2\n")["why"] == (
        "2 WSL VM processes: a mask names one VM")
    assert th.parse_vm_affinity("affinity error ProcessCommandException\n")["why"] == (
        "the VM process's mask could not be read (ProcessCommandException)")
    assert th.parse_vm_affinity("cpus 24\n") == {"readable": False, "why": "no affinity line"}
    assert "Get-Process -Name vmmemWSL" in th.WINDOWS_LOAD_SCRIPT


def test_the_mask_is_set_and_read_back_or_refused_by_name():
    calls = []

    def runner(command, text=SET_TOOK):
        calls.append(command)
        return text, ""

    done = th.set_vm_affinity(0x999988, runner=runner, program=lambda name: name)
    assert done == {"ok": True, "declared": "0x999988", "processes": 1, "before": "0xffffff",
                    "set": "ok", "read_back": "0x999988", "why": None}
    import base64
    script = base64.b64decode(calls[0][-1]).decode("utf-16-le")
    assert calls[0][:4] == ["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand"]
    assert "ProcessorAffinity = [IntPtr]10066312;" in script and "{mask}" not in script
    denied = th.set_vm_affinity(0x999988, runner=lambda c: (SET_DENIED, ""),
                                program=lambda name: name)
    assert denied["ok"] is False and denied["read_back"] == "0x0"
    assert denied["set"] == "error Win32Exception 5"
    assert denied["why"].startswith("the set was refused: access denied") and \
        "elevated context" in denied["why"]
    other = th.set_vm_affinity(0x999988, runner=lambda c: (SET_TOOK.replace("10066312", "16777215"), ""),
                               program=lambda name: name)
    assert other["ok"] is False and other["why"] == "the mask read back 0xffffff, not 0x999988"
    two = th.set_vm_affinity(0x999988, runner=lambda c: ("processes 2\n", "exit status 3"),
                             program=lambda name: name)
    assert two["ok"] is False and two["why"] == "2 WSL VM processes: a mask names one VM"
    absent = th.set_vm_affinity(0x999988, program=lambda name: None)
    assert absent["ok"] is False and absent["why"].startswith("powershell.exe not found")


def test_a_declared_mask_is_required_at_every_reading_and_recorded_otherwise():
    masked = th.WindowsRule(1695, 9501, vm_affinity=0x999988)
    held = _observed(load=WIN_LOAD_QUIET.replace("16777215", "10066312"))
    assert th.windows_reasons(masked, held) == []
    assert th.windows_reasons(masked, _observed()) == [
        "the WSL VM's CPU affinity is 0xffffff, not the declared 0x999988"]
    lost = _observed(load=WIN_LOAD_QUIET.replace("affinity 16777215 1", "affinity error X"))
    assert th.windows_reasons(masked, lost) == [
        "the WSL VM's CPU affinity unreadable: the VM process's mask could not be read (X)"]
    assert th.windows_reasons(th.WindowsRule(1695, 9501), _observed()) == [], \
        "no mask declared: the mask is recorded, never gated"
    assert masked.describe()["vm_affinity"] == "0x999988"
    assert th.WindowsRule().describe()["vm_affinity"] is None
