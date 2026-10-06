"""Host readers for the timing ladder. Standard library only.

Every reader comes in two halves: a PURE parser that takes text (``/proc/stat``, a
sysfs tree, ``vm_stat``, ``nvidia-smi`` CSV, a launcher's ``--version``) and returns
numbers, and a thin reader that fetches the text on this host. The tests feed the
parsers synthetic text; the ladder calls the readers.

THE CPU-QUIET GATE, AND THE DEFECT IT REPLACES. The 2026-09-28 ladder's ``busy()``
read the aggregate ``cpu`` line of ``/proc/stat`` and multiplied the busy fraction by
``$(nproc)``. ``nproc`` honours ``OMP_NUM_THREADS``, which the ladder had set to 1, so
the reading was at most 1.0 against a threshold of 10 and the MEEP rows' gate could
never fail. Here the busy count is summed per logical CPU over the ``cpuN`` lines
present in both samples (host scope) or over the CPUs of this process's affinity
(allocation scope); nothing reads ``nproc``, ``os.cpu_count()`` or the environment for
it. A sample that cannot be formed -- an unreadable file, a CPU set that changed
between the two reads, a counter that did not advance -- reads ``None``, and ``None``
is NOT quiet: the 2026-09-28 reader printed 0 in that case, which read as quiet.

On macOS the same quantity is read from the Mach host's CPU ticks
(``host_statistics(HOST_CPU_LOAD_INFO)``), in host scope only: the P/E mapping of CPU
indices is not exposed.
"""

from __future__ import annotations

import ctypes
import dataclasses
import glob
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

#: Seconds between the two ``/proc/stat`` (or Mach tick) samples of one busy reading.
SAMPLE_S = 3.0

#: The ``/proc/stat`` per-CPU fields, in order. ``guest`` and ``guest_nice`` are already
#: counted inside ``user`` and ``nice`` by the kernel, so they are not added again.
PROC_STAT_FIELDS = ("user", "nice", "system", "idle", "iowait", "irq", "softirq",
                    "steal", "guest", "guest_nice")
_TOTAL_FIELDS = 8

#: Other processes driving an Apple GPU, matched against a process's command line. A
#: SUPERSET of ``bench_fused_products._DEVICE_NEIGHBOURS`` (a test checks that every
#: script the bench matches is matched here): the bench's own pattern, plus the
#: drivers this ladder starts on the device (``bench_timing_case.py``,
#: ``gpu_array_control.py``) and another ladder (``timing_ladder.py``). Spelled here so
#: this module imports nothing from the harness. This process, its ancestors and its
#: children are excluded by number, so a ladder never counts itself or its own row.
DEVICE_NEIGHBOURS = re.compile(
    r"(?:^|[/\s])(?:gate_dispatch_metal_route\.py|recut_metal_gates\.sh|"
    r"gate_metal_\w+\.py|probe_metal_\w+\.py|metal_gate_runner\.py|"
    r"run_metal_dispatch_campaign\.sh|bench_fused_products\.py|"
    r"bench_timing_case\.py|gpu_array_control\.py|timing_ladder\.py)(?:\s|$)")

#: macOS background work that moves a timing, matched against ``ps``'s command name:
#: Spotlight indexing, backups, software update, photo and media analysis, XProtect.
#: The Mac quiet gate of record (``night3_uniform.sh``) waited on the same names.
DARWIN_DAEMONS = re.compile(
    r"(?:^|/)(?:mds_stores|mdworker\w*|spotlightknowledged|backupd|softwareupdate\w*|"
    r"photoanalysisd|mediaanalysisd|XProtect\w*)(?:\s|$)")
#: The CPU percentage above which one of :data:`DARWIN_DAEMONS` makes the host not quiet.
DARWIN_DAEMON_CPU_MAX = 10.0


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def run_text(command: Sequence[str], timeout: float = 30.0,
             env: Optional[Mapping[str, str]] = None,
             cwd: Optional[str] = None) -> Optional[str]:
    """stdout+stderr of a command, or ``None`` when it cannot be run.

    ``cwd`` matters for ``python -c``: Python puts the working directory BEFORE
    ``PYTHONPATH`` on ``sys.path``, so a ``-c`` probe run from a repository root imports
    that root's ``meep_gpu`` whatever ``PYTHONPATH`` names.
    """
    try:
        done = subprocess.run(list(command), capture_output=True, text=True,
                              timeout=timeout, check=False, cwd=cwd,
                              env=dict(env) if env is not None else None)
    except (OSError, subprocess.SubprocessError):
        return None
    return (done.stdout or "") + (done.stderr or "")


def read_file(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return None


# ---------------------------------------------------------------------------
# CPU busy: Linux
# ---------------------------------------------------------------------------

def parse_proc_stat(text: str) -> Dict[str, Tuple[int, int]]:
    """``cpu`` and every ``cpuN`` line -> (idle ticks, total ticks).

    total = user + nice + system + idle + iowait + irq + softirq + steal;
    idle = idle + iowait. Missing trailing fields read 0.
    """
    out: Dict[str, Tuple[int, int]] = {}
    for line in (text or "").splitlines():
        parts = line.split()
        if not parts or not parts[0].startswith("cpu"):
            continue
        name = parts[0]
        if name != "cpu" and not name[3:].isdigit():
            continue
        try:
            values = [int(v) for v in parts[1:1 + len(PROC_STAT_FIELDS)]]
        except ValueError:
            continue
        values += [0] * (len(PROC_STAT_FIELDS) - len(values))
        total = sum(values[:_TOTAL_FIELDS])
        idle = values[3] + values[4]
        out[name] = (idle, total)
    return out


def busy_from_samples(first: Mapping[str, Tuple[int, int]],
                      second: Mapping[str, Tuple[int, int]],
                      scope: Optional[Iterable[int]] = None) -> Dict[str, Any]:
    """Logical CPUs busy between two parsed ``/proc/stat`` samples.

    ``busy_cpus`` = sum over CPUs i in scope of (1 - d_idle_i / d_total_i). Host scope
    (``scope`` None) is every ``cpuN`` line present in BOTH samples; allocation scope
    is the CPUs named. ``None`` -- not quiet -- when the CPU set changed between the
    samples, a CPU in scope is missing, or any counter in scope did not advance.
    """
    reading: Dict[str, Any] = {"busy_cpus": None, "cpus": 0,
                               "scope": "host" if scope is None else "allocation",
                               "aggregate_cross_check": None, "why": None}
    names_a = {n for n in first if n != "cpu"}
    names_b = {n for n in second if n != "cpu"}
    if not names_a or not names_b:
        reading["why"] = "no per-CPU lines in one of the samples"
        return reading
    if names_a != names_b:
        reading["why"] = (f"the CPU set changed between the samples "
                          f"({len(names_a)} then {len(names_b)} lines)")
        return reading
    if scope is None:
        chosen = sorted(names_a, key=lambda n: int(n[3:]))
    else:
        chosen = [f"cpu{int(i)}" for i in sorted(set(scope))]
        missing = [n for n in chosen if n not in names_a]
        if missing:
            reading["why"] = f"CPUs in scope missing from /proc/stat: {missing[:8]}"
            return reading
    busy = 0.0
    for name in chosen:
        d_idle = second[name][0] - first[name][0]
        d_total = second[name][1] - first[name][1]
        if d_total <= 0:
            reading["why"] = f"{name}'s counters did not advance (d_total {d_total})"
            return reading
        busy += 1.0 - min(1.0, max(0.0, d_idle / float(d_total)))
    reading["busy_cpus"] = round(busy, 2)
    reading["cpus"] = len(chosen)
    if "cpu" in first and "cpu" in second:
        d_idle = second["cpu"][0] - first["cpu"][0]
        d_total = second["cpu"][1] - first["cpu"][1]
        if d_total > 0:
            reading["aggregate_cross_check"] = round(
                len(names_a) * (1.0 - d_idle / float(d_total)), 2)
    return reading


def affinity() -> Optional[List[int]]:
    """This process's CPU affinity, where the platform exposes it."""
    getter = getattr(os, "sched_getaffinity", None)
    if getter is None:
        return None
    try:
        return sorted(int(c) for c in getter(0))
    except OSError:
        return None


def busy_cpus_linux(scope: str = "host", sample_s: float = SAMPLE_S,
                    path: str = "/proc/stat",
                    sleep: Callable[[float], None] = time.sleep) -> Dict[str, Any]:
    first_text = read_file(path)
    sleep(sample_s)
    second_text = read_file(path)
    if first_text is None or second_text is None:
        return {"busy_cpus": None, "cpus": 0, "scope": scope,
                "aggregate_cross_check": None, "why": f"{path} is unreadable"}
    cpus = affinity() if scope == "allocation" else None
    if scope == "allocation" and cpus is None:
        return {"busy_cpus": None, "cpus": 0, "scope": scope,
                "aggregate_cross_check": None,
                "why": "allocation scope asked and this platform has no affinity"}
    return busy_from_samples(parse_proc_stat(first_text), parse_proc_stat(second_text),
                             cpus)


# ---------------------------------------------------------------------------
# CPU busy: macOS
# ---------------------------------------------------------------------------

#: ``HOST_CPU_LOAD_INFO`` and its count of ``natural_t`` (mach/host_info.h).
HOST_CPU_LOAD_INFO = 3
HOST_CPU_LOAD_INFO_COUNT = 4

#: The library and the host port, taken ONCE per process: every ``mach_host_self()``
#: call adds a send right to the port, so a reader that took one per reading would
#: leak a right per gate reading over a ladder of thousands.
_MACH_HOST: Dict[str, Any] = {}


def _mach_host() -> Tuple[Any, int]:
    if "host" not in _MACH_HOST:
        library = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        library.mach_host_self.restype = ctypes.c_uint
        _MACH_HOST["library"] = library
        _MACH_HOST["host"] = int(library.mach_host_self())
    return _MACH_HOST["library"], _MACH_HOST["host"]


def mach_cpu_ticks() -> Tuple[int, int, int, int]:
    """(user, system, idle, nice) ticks summed over the host's CPUs, from the kernel.

    Refused by name off macOS: the Mach host interface exists only there.
    """
    if sys.platform != "darwin":
        raise RuntimeError(f"mach_cpu_ticks reads the Mach host interface, which "
                           f"exists only on macOS; this platform is {sys.platform!r}")
    library, host = _mach_host()
    info = (ctypes.c_uint * HOST_CPU_LOAD_INFO_COUNT)()
    count = ctypes.c_uint(HOST_CPU_LOAD_INFO_COUNT)
    status = library.host_statistics(ctypes.c_uint(host), ctypes.c_int(HOST_CPU_LOAD_INFO),
                                     ctypes.byref(info), ctypes.byref(count))
    if status != 0:
        raise RuntimeError(f"host_statistics returned {status}")
    return int(info[0]), int(info[1]), int(info[2]), int(info[3])


def busy_from_ticks(first: Sequence[int], second: Sequence[int],
                    logical_cpus: int) -> Dict[str, Any]:
    """Logical CPUs busy from two Mach tick samples (user, system, idle, nice).

    ``nice`` counts as busy: a niced neighbour still occupies a core.
    """
    reading: Dict[str, Any] = {"busy_cpus": None, "cpus": logical_cpus,
                               "scope": "host", "aggregate_cross_check": None,
                               "why": None}
    deltas = [int(b) - int(a) for a, b in zip(first, second)]
    total = sum(deltas)
    if len(deltas) != 4 or total <= 0 or logical_cpus <= 0:
        reading["why"] = f"the tick counters did not advance (d_total {total})"
        return reading
    reading["busy_cpus"] = round(logical_cpus * (1.0 - deltas[2] / float(total)), 2)
    reading["ticks_per_second_per_cpu"] = None
    return reading


def busy_cpus_darwin(sample_s: float = SAMPLE_S, logical_cpus: Optional[int] = None,
                     sleep: Callable[[float], None] = time.sleep) -> Dict[str, Any]:
    try:
        first = mach_cpu_ticks()
        started = time.monotonic()
        sleep(sample_s)
        second = mach_cpu_ticks()
        elapsed = time.monotonic() - started
    except Exception as error:  # noqa: BLE001 - fail closed, by name
        return {"busy_cpus": None, "cpus": 0, "scope": "host",
                "aggregate_cross_check": None, "why": f"{type(error).__name__}: {error}"}
    if logical_cpus is None:
        logical_cpus = int(darwin_sysctl().get("hw.logicalcpu") or 0)
    reading = busy_from_ticks(first, second, logical_cpus)
    if elapsed > 0 and logical_cpus:
        reading["ticks_per_second_per_cpu"] = round(
            sum(b - a for a, b in zip(first, second)) / elapsed / logical_cpus, 1)
    return reading


def busy_cpus(scope: str = "host", sample_s: float = SAMPLE_S,
              platform: Optional[str] = None) -> Dict[str, Any]:
    """Logical CPUs busy over ``sample_s`` seconds, on this platform."""
    platform = platform or sys.platform
    if platform == "darwin":
        if scope != "host":
            return {"busy_cpus": None, "cpus": 0, "scope": scope,
                    "aggregate_cross_check": None,
                    "why": "macOS exposes no per-CPU ticks for an allocation scope"}
        return busy_cpus_darwin(sample_s)
    return busy_cpus_linux(scope, sample_s)


def load1() -> Optional[float]:
    try:
        return round(os.getloadavg()[0], 2)
    except (OSError, AttributeError):
        return None


def loadavg() -> Optional[List[float]]:
    try:
        return [round(v, 2) for v in os.getloadavg()]
    except (OSError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

GIB = 1024 ** 3


def parse_meminfo(text: str) -> Dict[str, int]:
    """``/proc/meminfo`` -> {key: kB}."""
    out: Dict[str, int] = {}
    for line in (text or "").splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            out[key.strip()] = int(parts[0])
    return out


def parse_vm_stat(text: str) -> Dict[str, int]:
    """``vm_stat`` -> {"page_size": bytes, "<field>": pages}."""
    out: Dict[str, int] = {}
    size = re.search(r"page size of (\d+) bytes", text or "")
    if size:
        out["page_size"] = int(size.group(1))
    for line in (text or "").splitlines():
        match = re.match(r'^\s*"?([A-Za-z][^:"]*)"?:\s+(\d+)\.?\s*$', line)
        if match:
            out[match.group(1).strip().lower().replace(" ", "_").replace("-", "_")] = \
                int(match.group(2))
    return out


def available_gib_from_vm_stat(parsed: Mapping[str, int]) -> Optional[float]:
    """free + inactive + speculative + purgeable pages, in GiB: an APPROXIMATION of
    what Linux calls MemAvailable, named as one wherever it is recorded."""
    size = parsed.get("page_size")
    if not size:
        return None
    pages = sum(int(parsed.get(name, 0)) for name in
                ("pages_free", "pages_inactive", "pages_speculative", "pages_purgeable"))
    return round(pages * size / GIB, 1)


def memory(platform: Optional[str] = None) -> Dict[str, Any]:
    """Available and total memory in GiB (``free -g``'s unit), and how each was read."""
    platform = platform or sys.platform
    if platform == "darwin":
        parsed = parse_vm_stat(run_text(["vm_stat"]) or "")
        total = darwin_sysctl().get("hw.memsize")
        level = sysctl_value("kern.memorystatus_level")
        total_gib = round(int(total) / GIB, 1) if total else None
        # The kernel's own percentage of memory available, beside the vm_stat sum:
        # the vm_stat sum counts purgeable pages twice and dirty inactive pages as
        # free, so the two disagree; the gate reads the vm_stat sum and both are kept.
        return {"available_gib": available_gib_from_vm_stat(parsed),
                "total_gib": total_gib,
                "memorystatus_level_percent": int(level) if level and level.isdigit()
                else None,
                "available_gib_memorystatus": (round(total_gib * int(level) / 100.0, 1)
                                               if total_gib and level and level.isdigit()
                                               else None),
                "method": "vm_stat free+inactive+speculative+purgeable (approximation); "
                          "kern.memorystatus_level beside it"}
    parsed = parse_meminfo(read_file("/proc/meminfo") or "")
    available = parsed.get("MemAvailable")
    total = parsed.get("MemTotal")
    return {"available_gib": round(available * 1024 / GIB, 1) if available else None,
            "total_gib": round(total * 1024 / GIB, 1) if total else None,
            "method": "/proc/meminfo MemAvailable"}


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------

def parse_cpulist(text: str) -> List[int]:
    """``0-23,48-71`` -> [0, ..., 23, 48, ..., 71]."""
    out: List[int] = []
    for part in (text or "").strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            low, high = part.split("-", 1)
            out.extend(range(int(low), int(high) + 1))
        else:
            out.append(int(part))
    return out


def topology_linux(sys_root: str = "/sys", allowed: Optional[Iterable[int]] = None,
                   cpuinfo_text: Optional[str] = None) -> Dict[str, Any]:
    """Physical cores and logical CPUs in this process's scope, from sysfs.

    A physical core is a unique (physical_package_id, core_id) over the ONLINE CPUs,
    intersected with ``allowed`` (the affinity: a Slurm allocation's cpuset inside a
    job). Falls back to ``/proc/cpuinfo`` when sysfs carries no topology.
    """
    base = os.path.join(sys_root, "devices", "system", "cpu")
    online_text = read_file(os.path.join(base, "online"))
    online = set(parse_cpulist(online_text)) if online_text else None
    allowed_set = set(allowed) if allowed is not None else None
    cores: Dict[Tuple[int, int], List[int]] = {}
    logical: List[int] = []
    for path in sorted(glob.glob(os.path.join(base, "cpu[0-9]*"))):
        name = os.path.basename(path)
        if not name[3:].isdigit():
            continue
        index = int(name[3:])
        if online is not None and index not in online:
            continue
        if allowed_set is not None and index not in allowed_set:
            continue
        package = read_file(os.path.join(path, "topology", "physical_package_id"))
        core = read_file(os.path.join(path, "topology", "core_id"))
        if package is None or core is None:
            continue
        logical.append(index)
        cores.setdefault((int(package), int(core)), []).append(index)
    record: Dict[str, Any] = {"method": "sysfs"}
    if not cores and cpuinfo_text is not None:
        record = topology_from_cpuinfo(cpuinfo_text, allowed_set)
    elif not cores:
        text = read_file("/proc/cpuinfo")
        if text is not None:
            record = topology_from_cpuinfo(text, allowed_set)
        else:
            return {"method": None, "physical_cores": None, "logical_cpus": None,
                    "why": "no sysfs topology and no /proc/cpuinfo"}
    else:
        record.update({
            "physical_cores": len(cores), "logical_cpus": len(logical),
            "sockets": len({package for package, _core in cores}),
            "threads_per_core": max(len(v) for v in cores.values()),
            "cpus": logical,
            "package_of_cpu": {str(cpu): package for (package, _core), cpus in
                               sorted(cores.items()) for cpu in cpus},
            # (package, core) per CPU, so a rank's affinity can be read as the
            # PHYSICAL cores it occupies (the launcher probe, the hwthread strike).
            "core_of_cpu": {str(cpu): f"{package}:{core}" for (package, core), cpus in
                            sorted(cores.items()) for cpu in cpus},
        })
    smt = read_file(os.path.join(base, "smt", "active"))
    record["smt_active"] = None if smt is None else smt.strip() == "1"
    numa: Dict[str, str] = {}
    for path in sorted(glob.glob(os.path.join(sys_root, "devices", "system", "node",
                                              "node[0-9]*", "cpulist"))):
        numa[os.path.basename(os.path.dirname(path))] = (read_file(path) or "").strip()
    record["numa_cpulists"] = numa
    record["online_cpus"] = len(online) if online is not None else None
    # NARROWER THAN THE HOST: this process's affinity holds fewer CPUs than are
    # online (a Slurm job's cpuset, a taskset). A rank's "bound" flag is then judged
    # against this scope, not against os.cpu_count().
    record["restricted_by_affinity"] = bool(
        allowed_set is not None and online is not None
        and len(online & allowed_set) < len(online))
    return record


def topology_from_cpuinfo(text: str, allowed: Optional[set] = None) -> Dict[str, Any]:
    cores: Dict[Tuple[int, int], List[int]] = {}
    logical: List[int] = []
    processor = package = core = None
    for line in (text or "").splitlines() + [""]:
        if not line.strip():
            if processor is not None and (allowed is None or processor in allowed):
                logical.append(processor)
                cores.setdefault((package or 0, core if core is not None else processor),
                                 []).append(processor)
            processor = package = core = None
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key == "processor" and value.isdigit():
            processor = int(value)
        elif key == "physical id" and value.isdigit():
            package = int(value)
        elif key == "core id" and value.isdigit():
            core = int(value)
    return {"method": "/proc/cpuinfo", "physical_cores": len(cores) or None,
            "logical_cpus": len(logical) or None,
            "sockets": len({p for p, _c in cores}) or None,
            "threads_per_core": max((len(v) for v in cores.values()), default=None),
            "cpus": logical}


#: The keys of :data:`SYSCTL_KEYS` whose values are text; the rest are integers.
SYSCTL_TEXT_KEYS = ("hw.model", "machdep.cpu.brand_string")

SYSCTL_KEYS = ("hw.physicalcpu", "hw.logicalcpu", "hw.perflevel0.physicalcpu",
               "hw.perflevel1.physicalcpu", "hw.memsize", "hw.pagesize", "hw.model",
               "machdep.cpu.brand_string")


def parse_sysctl(text: str) -> Dict[str, str]:
    """``sysctl a b c`` ("name: value" lines) -> {name: value}."""
    out: Dict[str, str] = {}
    for line in (text or "").splitlines():
        key, sep, value = line.partition(":")
        if sep:
            out[key.strip()] = value.strip()
    return out


def sysctl_value(name: str) -> Optional[str]:
    """One ``sysctlbyname`` value through ``ctypes``: no process is started, so a dry run
    can read the topology. Integers come back as decimal text; ``None`` off macOS."""
    if sys.platform != "darwin":
        return None
    try:
        library = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        size = ctypes.c_size_t(0)
        if library.sysctlbyname(name.encode(), None, ctypes.byref(size), None, 0) != 0:
            return None
        buffer = ctypes.create_string_buffer(size.value)
        if library.sysctlbyname(name.encode(), buffer, ctypes.byref(size), None, 0) != 0:
            return None
    except (OSError, AttributeError):
        return None
    raw = buffer.raw[:size.value]
    if name in SYSCTL_TEXT_KEYS:
        return raw.rstrip(b"\x00").decode("utf-8", errors="replace")
    return str(int.from_bytes(raw, sys.byteorder))


def darwin_sysctl() -> Dict[str, str]:
    """The topology keys, read with ``sysctlbyname`` (no subprocess)."""
    if sys.platform != "darwin":
        return {}
    out: Dict[str, str] = {}
    for key in SYSCTL_KEYS:
        value = sysctl_value(key)
        if value is not None:
            out[key] = value
    return out


def topology_darwin(values: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """P-cores are the rank basis on Apple silicon; the E-cores are counted beside."""
    values = dict(values if values is not None else darwin_sysctl())

    def number(key: str) -> Optional[int]:
        value = values.get(key)
        return int(value) if value and str(value).isdigit() else None

    performance = number("hw.perflevel0.physicalcpu")
    physical = number("hw.physicalcpu")
    return {"method": "sysctl",
            "physical_cores": performance or physical,
            "logical_cpus": number("hw.logicalcpu"),
            "all_physical_cores": physical,
            "performance_cores": performance,
            "efficiency_cores": number("hw.perflevel1.physicalcpu"),
            "sockets": 1, "threads_per_core": 1, "smt_active": False,
            "model": values.get("hw.model"),
            "cpu": values.get("machdep.cpu.brand_string")}


def topology(platform: Optional[str] = None) -> Dict[str, Any]:
    platform = platform or sys.platform
    if platform == "darwin":
        return topology_darwin()
    return topology_linux(allowed=affinity())


SLURM_VARIABLES = ("SLURM_JOB_ID", "SLURMD_NODENAME", "SLURM_CPUS_ON_NODE", "SLURM_JOB_CPUS_PER_NODE",
                   "SLURM_NTASKS", "SLURM_TASKS_PER_NODE", "SLURM_CPUS_PER_TASK",
                   "SLURM_JOB_GPUS", "SLURM_JOB_NODELIST", "SLURM_STEP_ID",
                   "SLURM_JOB_NUM_NODES", "CUDA_VISIBLE_DEVICES")


def slurm_facts(environ: Optional[Mapping[str, str]] = None) -> Dict[str, Optional[str]]:
    environ = os.environ if environ is None else environ
    return {name: environ.get(name) for name in SLURM_VARIABLES}


def first_node_count(text: Optional[str]) -> Optional[int]:
    """The first node's count from a Slurm per-node list: ``96``, ``48(x2)``, ``2,1``."""
    match = re.match(r"\s*(\d+)", text or "")
    return int(match.group(1)) if match else None


def slurm_refusals(environ: Mapping[str, str], implementation: str, launcher_kind: str,
                   max_ranks: int, cpus_in_scope: Optional[int]) -> List[str]:
    """What a ladder inside a Slurm allocation must refuse before it starts, by name.

    * A shell that holds an allocation but is not on its node (an ``salloc`` shell on
      the login node: no ``SLURMD_NODENAME``).
    * A ladder started INSIDE a job step (``srun ... timing_ladder.py``): the step's
      affinity is narrow, every rank count is planned against it, and an ``mpirun``
      started there nests a second launcher inside the step.
    * A ladder whose CPUs in scope are well below the node's (under 90 % of
      ``SLURM_CPUS_ON_NODE``): the rank blocks would be planned against a fraction of
      the node.
    * ``mpirun`` in an allocation with fewer task slots on the node than the largest
      planned rank count: Open MPI takes its slot count from
      ``SLURM_TASKS_PER_NODE`` and the first larger row would fail.
    Outside Slurm (no ``SLURM_JOB_ID``) nothing is refused.
    """
    problems: List[str] = []
    if not environ.get("SLURM_JOB_ID"):
        return problems
    if not environ.get("SLURMD_NODENAME"):
        # An salloc shell stays on the login node unless the site runs an interactive
        # step: the gate and the GPU rows would read the wrong machine while mpirun
        # starts the ranks on the allocated node. Every process on an allocated node
        # (batch script, interactive step, srun step) carries SLURMD_NODENAME.
        problems.append("this shell holds a Slurm allocation (SLURM_JOB_ID="
                        f"{environ.get('SLURM_JOB_ID')}) but is not on the allocated node "
                        "(no SLURMD_NODENAME): start the ladder from the batch script or a "
                        "shell on the node")
    step = str(environ.get("SLURM_STEP_ID") or "")
    if step.isdigit():
        problems.append(f"this ladder runs inside Slurm job step {step} (started under "
                        "srun): the step's CPU affinity is narrow and a launcher started "
                        "there nests inside it; start the ladder from the batch script or "
                        "the salloc shell, not under srun")
    on_node = first_node_count(environ.get("SLURM_CPUS_ON_NODE"))
    if on_node and cpus_in_scope is not None and cpus_in_scope < 0.9 * on_node:
        problems.append(f"this process may use {cpus_in_scope} CPUs and the allocation "
                        f"holds {on_node} on this node (SLURM_CPUS_ON_NODE); the rank "
                        "blocks would be planned against a fraction of the node")
    if launcher_kind != "srun" and implementation in ("openmpi", "unknown"):
        tasks = first_node_count(environ.get("SLURM_TASKS_PER_NODE"))
        if tasks is not None and tasks < max_ranks:
            problems.append(f"the allocation gives {tasks} task slots on this node "
                            f"(SLURM_TASKS_PER_NODE={environ.get('SLURM_TASKS_PER_NODE')}) "
                            f"and the plan starts {max_ranks} ranks under mpirun; allocate "
                            f"with --ntasks-per-node={max_ranks} or more")
    return problems


# ---------------------------------------------------------------------------
# MPI launchers and their binding flags
# ---------------------------------------------------------------------------

def mpi_implementation(version_text: Optional[str]) -> str:
    """``openmpi``, ``mpich``, ``intelmpi``, ``srun`` or ``unknown``, from ``--version``."""
    text = version_text or ""
    if re.search(r"Open MPI|OpenRTE|open-mpi\.org", text, re.I):
        return "openmpi"
    if re.search(r"Intel\(R\) MPI", text):
        return "intelmpi"
    if re.search(r"HYDRA|MPICH", text):
        return "mpich"
    if re.match(r"\s*slurm\s+\d", text, re.I):
        return "srun"
    return "unknown"


#: Binding modes, their labels as rows record them, and the flags per implementation.
#: ``bound``/``unbound``/``over`` are the 2026-09-28 modes and strings. ``over`` is the
#: mode of a rank count above the physical cores in scope.
BINDING_LABELS = {
    "bound": "bound-core-by-package",
    "unbound": "unbound",
    "over": "unbound-hwthreads",
    "bound-hwthread": "bound-hwthread",
    "bound-pe": "bound-core-by-package-pe{threads}",
}

#: Placeholders a template may carry, filled per row by :func:`binding_flags`:
#: ``{threads}`` OpenMP threads per rank; ``{per_socket}`` ceil(ranks / sockets);
#: ``{threads_per_core}`` hardware threads per core; ``{cpus_per_task}`` threads x
#: threads per core (a rank's T threads on T whole cores).
#:
#: THE srun TEMPLATES, per the Slurm ``srun`` manual, and why each flag:
#: * ``--cpu-bind`` given on the command line OVERRIDES ``--hint``, so no template
#:   uses ``--hint``; ``--ntasks-per-core`` states the per-core task count instead.
#: * ``--distribution=block:block`` alone fills one socket before the next, so 16 ranks
#:   on a 2 x 24-core node would share one socket's memory bandwidth; the Open MPI
#:   rows split ranks evenly across packages, contiguously (ranks 0-7 on package 0,
#:   8-15 on package 1, as the 2026-09-28 binding report reads). ``--ntasks-per-socket
#:   ={per_socket}`` with block:block is that placement.
#: * ``bound-hwthread`` adds a third distribution level, ``cyclic``, so consecutive
#:   ranks take consecutive CORES before a core's second hardware thread.
#: The srun path has not run live on a cluster; ``--selftest --launcher-probe`` checks
#: each mode's placement against the topology before anything is timed.
FLAG_TEMPLATES: Dict[str, Dict[str, str]] = {
    "openmpi": {
        "bound": "--map-by package --bind-to core",
        "unbound": "--bind-to none",
        "over": "--use-hwthread-cpus --bind-to none",
        "bound-hwthread": "--use-hwthread-cpus --map-by package --bind-to hwthread",
        "bound-pe": "--map-by package:PE={threads} --bind-to core",
    },
    "srun": {
        "bound": "--cpu-bind=cores --ntasks-per-core=1 --ntasks-per-socket={per_socket} "
                 "--distribution=block:block",
        "unbound": "--cpu-bind=none",
        "over": "--cpu-bind=none --ntasks-per-core={threads_per_core}",
        "bound-hwthread": "--cpu-bind=threads --ntasks-per-core={threads_per_core} "
                          "--ntasks-per-socket={per_socket} "
                          "--distribution=block:block:cyclic",
        "bound-pe": "--cpus-per-task={cpus_per_task} --cpu-bind=cores --ntasks-per-core=1 "
                    "--ntasks-per-socket={per_socket} --distribution=block:block",
    },
}

#: The modes in which a rank is pinned to a fixed set of CPUs.
BOUND_MODES = ("bound", "bound-hwthread", "bound-pe")


def binding_mode_for(ranks: int, asked: str, physical_cores: int, threads: int = 1) -> str:
    """The mode a MEEP row of ``ranks`` ranks takes when ``asked`` was chosen.

    The 2026-09-28 rule: above the physical cores the ``over`` mode, whatever was
    asked (unless ``bound-hwthread`` was asked explicitly); otherwise the asked mode;
    a thread count above one is the ``bound-pe`` mode when the asked mode is bound.
    """
    if asked == "bound-hwthread":
        return "bound-hwthread"
    if ranks > physical_cores:
        return "over"
    if threads > 1 and asked == "bound":
        return "bound-pe"
    return asked


#: The label of a macOS row above the performance cores: its ranks run on the
#: efficiency cores too, which is not the same configuration as the rows below it.
DARWIN_OVER_LABEL = "unbound-ecores"


def binding_flags(implementation: str, mode: str, threads: int = 1,
                  platform: str = "linux",
                  explicit: Optional[Mapping[str, str]] = None,
                  ranks: int = 1, sockets: Optional[int] = None,
                  threads_per_core: Optional[int] = None) -> Tuple[str, str]:
    """(flags, label) for a binding mode, or SystemExit naming why it is refused.

    ``explicit`` maps a mode to a flag string the operator gave, which wins over any
    template. On macOS only the unbound modes exist: Open MPI cannot bind there. A
    template that needs the socket count (``{per_socket}``) is refused by name when it
    is unknown.
    """
    label = BINDING_LABELS[mode].format(threads=threads)
    per_core = int(threads_per_core or 1)
    values = {"threads": threads, "threads_per_core": per_core,
              "cpus_per_task": int(threads) * per_core,
              "per_socket": (int(math.ceil(int(ranks) / float(sockets)))
                             if sockets else None)}
    if explicit and mode in explicit and explicit[mode] is not None:
        return str(explicit[mode]).format(**values), label
    if platform == "darwin" and mode not in ("unbound", "over"):
        raise SystemExit(f"REFUSING: binding mode {mode!r} on macOS: Open MPI cannot bind "
                         "ranks to cores there; use --binding unbound")
    if platform == "darwin":
        return FLAG_TEMPLATES["openmpi"]["unbound"], (
            DARWIN_OVER_LABEL if mode == "over" else BINDING_LABELS["unbound"])
    templates = FLAG_TEMPLATES.get(implementation)
    if templates is None:
        raise SystemExit(f"REFUSING: the MPI launcher is {implementation!r} and no flag "
                         f"template is known for it; give the flags for mode {mode!r} "
                         "explicitly (--flags-bound, --flags-unbound, --flags-over, ...)")
    template = templates[mode]
    if "{per_socket}" in template and values["per_socket"] is None:
        raise SystemExit(f"REFUSING: the {implementation} template of mode {mode!r} "
                         "balances ranks across sockets and the socket count is unknown; "
                         "pass --sockets")
    return template.format(**values), label


def with_report(implementation: str, flags: str) -> str:
    """The flags with the launcher's own binding report switched on."""
    if implementation == "openmpi":
        return (flags + " --report-bindings").strip()
    if implementation == "srun":
        if "--cpu-bind=" in flags:
            return re.sub(r"--cpu-bind=(\S+)", r"--cpu-bind=verbose,\1", flags, count=1)
        return (flags + " --cpu-bind=verbose").strip()
    return flags


def binding_report_lines(text: str) -> List[str]:
    """The launcher's per-rank binding lines: Open MPI ``Rank N bound to`` / ``is not
    bound``, and Slurm's ``cpu-bind=... - host, task N``."""
    out = []
    for line in (text or "").splitlines():
        if ("Rank" in line and "bound" in line) or re.search(r"cpu[-_]bind", line) \
                and re.search(r"task\s+\d+", line):
            out.append(line.strip())
    return out


#: The one line each rank of ``--launcher-probe`` prints: ``AFFINITY <host> <pid> [cpus]``.
AFFINITY_PROBE = ("import os, socket; a = sorted(os.sched_getaffinity(0)) if "
                  "hasattr(os, 'sched_getaffinity') else None; "
                  "print('AFFINITY', socket.gethostname(), os.getpid(), a, flush=True)")


def parse_affinity_lines(text: str) -> List[Optional[List[int]]]:
    """Each rank's CPU set from the probe's ``AFFINITY`` lines (``None``: unreadable)."""
    out: List[Optional[List[int]]] = []
    for line in (text or "").splitlines():
        match = re.match(r"\s*AFFINITY\s+\S+\s+\d+\s+(.*)$", line)
        if not match:
            continue
        found = match.group(1).strip()
        out.append(None if found == "None" else [int(v) for v in re.findall(r"\d+", found)])
    return out


def check_placement(mode: str, affinities: Sequence[Optional[Sequence[int]]], ranks: int,
                    core_of_cpu: Mapping[str, str], allocation: Optional[Sequence[int]],
                    threads: int = 1) -> List[str]:
    """Every way the ranks' CPU sets differ from what ``mode`` promises, by name.

    ``core_of_cpu`` maps a CPU to ``"<package>:<core>"`` (``topology_linux``);
    ``allocation`` is the CPUs the ladder itself may use. The rules:

    * every rank answered, with a readable set;
    * ``bound``: each rank on ONE physical core, no core shared, ranks balanced across
      packages (counts differ by at most one);
    * ``bound-hwthread``: each rank on one hardware thread, none shared, and at least
      min(ranks, physical cores) distinct cores used;
    * ``bound-pe``: each rank on exactly ``threads`` whole cores, no core shared;
    * ``unbound``/``over``: every rank's set equals the allocation.
    """
    problems: List[str] = []
    if len(affinities) != ranks:
        problems.append(f"{len(affinities)} of {ranks} ranks answered")
    sets = [list(a) for a in affinities if a is not None]
    if len(sets) != len(affinities):
        problems.append(f"{len(affinities) - len(sets)} rank(s) could not read their "
                        "affinity")
    if not sets:
        return problems
    unknown = sorted({c for s in sets for c in s if str(c) not in core_of_cpu})
    if mode in BOUND_MODES and unknown:
        problems.append(f"CPUs {unknown[:8]} are missing from the topology's core map")
        return problems

    def cores(cpus: Sequence[int]) -> List[str]:
        return sorted({core_of_cpu[str(c)] for c in cpus})

    def balance(per_rank_cores: List[List[str]]) -> None:
        packages: Dict[str, int] = {}
        for found in per_rank_cores:
            for package in {c.split(":")[0] for c in found}:
                packages[package] = packages.get(package, 0) + 1
        every = sorted({c.split(":")[0] for c in core_of_cpu.values()})
        counts = [packages.get(p, 0) for p in every]
        if len(every) > 1 and counts and max(counts) - min(counts) > 1:
            problems.append("ranks are not balanced across packages: "
                            + ", ".join(f"package {p} {packages.get(p, 0)}" for p in every))

    if mode == "bound":
        per_rank = [cores(s) for s in sets]
        wide = [i for i, found in enumerate(per_rank) if len(found) != 1]
        if wide:
            problems.append(f"{len(wide)} rank(s) span more than one physical core "
                            f"(first: {per_rank[wide[0]]})")
        flat = [c for found in per_rank for c in found]
        if len(set(flat)) != len(flat):
            problems.append(f"ranks share physical cores: {len(flat)} rank-cores on "
                            f"{len(set(flat))} distinct cores")
        balance(per_rank)
    elif mode == "bound-hwthread":
        wide = [s for s in sets if len(s) != 1]
        if wide:
            problems.append(f"{len(wide)} rank(s) hold more than one hardware thread")
        flat = [c for s in sets for c in s]
        if len(set(flat)) != len(flat):
            problems.append("ranks share hardware threads")
        used = {core_of_cpu[str(c)] for c in flat}
        physical = len(set(core_of_cpu.values()))
        if len(used) < min(len(sets), physical):
            problems.append(f"the ranks use {len(used)} distinct physical cores of "
                            f"{physical}: cores are left idle while others run two ranks")
        balance([cores(s) for s in sets])
    elif mode == "bound-pe":
        per_rank = [cores(s) for s in sets]
        wrong = [found for found in per_rank if len(found) != threads]
        if wrong:
            problems.append(f"{len(wrong)} rank(s) do not hold exactly {threads} cores "
                            f"(first: {wrong[0]})")
        flat = [c for found in per_rank for c in found]
        if len(set(flat)) != len(flat):
            problems.append("ranks share physical cores")
        balance(per_rank)
    elif mode in ("unbound", "over"):
        if allocation is None:
            problems.append("the allocation's CPUs are unknown")
        else:
            want = sorted(set(allocation))
            narrow = [s for s in sets if sorted(set(s)) != want]
            if narrow:
                problems.append(f"{len(narrow)} rank(s) are confined to fewer CPUs than "
                                f"the allocation's {len(want)} (first: {len(narrow[0])} "
                                "CPUs)")
    return problems


# ---------------------------------------------------------------------------
# Account processes and device occupancy
# ---------------------------------------------------------------------------

def account_uid(account: Optional[str]) -> Optional[int]:
    if account is None:
        return os.getuid() if hasattr(os, "getuid") else None
    if str(account).isdigit():
        return int(account)
    try:
        import pwd  # noqa: PLC0415 - not on every platform
        return int(pwd.getpwnam(account).pw_uid)
    except (ImportError, KeyError):
        return None


def parse_status_uid(text: str) -> Optional[int]:
    """The real uid from ``/proc/<pid>/status`` (``Uid:\\treal\\teffective...``)."""
    for line in (text or "").splitlines():
        if line.startswith("Uid:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1])
    return None


def uid_of_pid(pid: int, proc_root: str = "/proc") -> Optional[int]:
    text = read_file(os.path.join(proc_root, str(pid), "status"))
    if text is not None:
        return parse_status_uid(text)
    answer = run_text(["ps", "-o", "uid=", "-p", str(pid)], timeout=10)
    if answer and answer.strip().split() and answer.strip().split()[0].isdigit():
        return int(answer.strip().split()[0])
    return None


def nvidia_smi_available() -> bool:
    return shutil.which("nvidia-smi") is not None


def parse_compute_apps(text: str) -> List[Dict[str, Any]]:
    """``--query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader`` rows."""
    out = []
    for line in (text or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[1].isdigit():
            out.append({"gpu_uuid": parts[0], "pid": int(parts[1]),
                        "used_memory": parts[2] if len(parts) > 2 else None})
    return out


def parse_index_uuid(text: str) -> Dict[str, str]:
    """``nvidia-smi --query-gpu=index,uuid --format=csv,noheader`` -> {index: uuid}."""
    out: Dict[str, str] = {}
    for line in (text or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].startswith("GPU-"):
            out[parts[0]] = parts[1]
    return out


def gpu_uuids() -> Optional[Dict[str, str]]:
    """{index: uuid} of every GPU ``nvidia-smi`` lists here; ``None`` when unreadable."""
    if not nvidia_smi_available():
        return None
    text = run_text(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                    timeout=30)
    return None if text is None else parse_index_uuid(text)


def as_uuid(device: Optional[str], uuids: Mapping[str, str]) -> Optional[str]:
    """A device index or UUID as a UUID (``None`` when an index is not listed)."""
    if device is None:
        return None
    device = str(device).strip()
    return device if device.startswith("GPU-") else uuids.get(device)


def compute_apps(gpu: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
    """Compute processes on every GPU, or on ``gpu``; ``None`` when unreadable."""
    if not nvidia_smi_available():
        return None
    command = ["nvidia-smi"]
    if gpu is not None:
        command.append(f"--id={gpu}")
    command += ["--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"]
    text = run_text(command, timeout=30)
    if text is None:
        return None
    return parse_compute_apps(text)


def count_account_processes(pids: Iterable[int], uid: Optional[int],
                            uid_reader: Callable[[int], Optional[int]] = uid_of_pid) -> int:
    """How many of ``pids`` belong to ``uid``. Matched by number, never by name: the
    2026-09-28 ``grep -c`` counted a longer account name that begins with the short one
    and missed a user name ``ps`` truncated."""
    if uid is None:
        return 0
    return sum(1 for pid in pids if uid_reader(int(pid)) == uid)


def foreign_device_processes(ps_text: str, own_pid: int,
                             pattern: "re.Pattern[str]" = DEVICE_NEIGHBOURS) -> List[str]:
    """Every OTHER process driving the Apple GPU, from ``ps -axo pid=,ppid=,uid=,command=``.

    This process, its ancestors and its children are excluded by number (a launching
    shell carries the bench's own command line).
    """
    table: List[Tuple[int, int, str, str]] = []
    for line in (ps_text or "").splitlines():
        parts = line.split(None, 3)
        if len(parts) < 4 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        table.append((int(parts[0]), int(parts[1]), parts[2], parts[3]))
    parent = {pid: ppid for pid, ppid, _uid, _command in table}
    mine = {own_pid}
    cursor = own_pid
    while cursor in parent and parent[cursor] not in mine:
        cursor = parent[cursor]
        mine.add(cursor)
    return [f"{pid} {ppid} {uid} {command[:200]}" for pid, ppid, uid, command in table
            if pid not in mine and ppid != own_pid and pattern.search(command)]


def metal_neighbours() -> Optional[List[str]]:
    text = run_text(["ps", "-axo", "pid=,ppid=,uid=,command="], timeout=15)
    if text is None:
        return None
    return foreign_device_processes(text, os.getpid())


def device_state(gpu: Optional[str], uid: Optional[int],
                 platform: Optional[str] = None) -> Dict[str, Any]:
    """Who is on the device and how many GPU processes the account holds.

    NVIDIA: zero compute apps on the pinned device is "free"; the account count is the
    account's compute processes on EVERY device. macOS: no foreign harness process
    driving the Apple GPU is "free"; the account count is how many of those are the
    account's.
    """
    platform = platform or sys.platform
    if platform == "darwin":
        found = metal_neighbours()
        if found is None:
            return {"device_free": None, "account_gpu_processes": None,
                    "device_processes": None, "why": "ps is unreadable"}
        mine = [line for line in found if line.split()[2:3] == [str(uid)]]
        return {"device_free": not found, "account_gpu_processes": len(mine),
                "device_processes": found, "how": "ps scan for Metal harness processes"}
    every = compute_apps(None)
    if every is None:
        return {"device_free": None, "account_gpu_processes": None,
                "device_processes": None, "why": "nvidia-smi is unreadable or absent"}
    pinned = compute_apps(gpu) if gpu is not None else []
    return {"device_free": pinned is not None and len(pinned) == 0,
            "account_gpu_processes": count_account_processes(
                [app["pid"] for app in every], uid),
            "device_processes": pinned, "how": "nvidia-smi compute apps"}


def gpu_brief(gpu: Optional[str]) -> Tuple[str, str]:
    """``gpu<N>=<memory.used>, <utilization>`` and every GPU's utilisation, as 09-28."""
    if gpu is None or not nvidia_smi_available():
        return "", ""
    own = (run_text(["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used,utilization.gpu",
                     "--format=csv,noheader"]) or "").strip()
    others = (run_text(["nvidia-smi", "--query-gpu=utilization.gpu",
                        "--format=csv,noheader,nounits"]) or "")
    return own, ",".join(line.strip() for line in others.splitlines() if line.strip()) + ","


# ---------------------------------------------------------------------------
# CPU and GPU state around rows
# ---------------------------------------------------------------------------

def cpu_state_linux(sys_root: str = "/sys", cpus: Optional[Iterable[int]] = None
                    ) -> Dict[str, Any]:
    """Governors, driver, turbo or boost, EPP, current frequencies over ``cpus``."""
    base = os.path.join(sys_root, "devices", "system", "cpu")
    if cpus is None:
        cpus = [int(os.path.basename(p)[3:]) for p in glob.glob(os.path.join(base, "cpu[0-9]*"))
                if os.path.basename(p)[3:].isdigit()]
    governors: Dict[str, int] = {}
    epp: Dict[str, int] = {}
    drivers: Dict[str, int] = {}
    frequencies: List[int] = []
    maximum: List[int] = []
    for index in sorted(cpus):
        policy = os.path.join(base, f"cpu{index}", "cpufreq")
        for name, bucket in (("scaling_governor", governors),
                             ("energy_performance_preference", epp),
                             ("scaling_driver", drivers)):
            value = read_file(os.path.join(policy, name))
            if value is not None:
                bucket[value.strip()] = bucket.get(value.strip(), 0) + 1
        current = read_file(os.path.join(policy, "scaling_cur_freq"))
        if current and current.strip().isdigit():
            frequencies.append(int(current.strip()))
        top = read_file(os.path.join(policy, "cpuinfo_max_freq"))
        if top and top.strip().isdigit():
            maximum.append(int(top.strip()))
    no_turbo = read_file(os.path.join(base, "intel_pstate", "no_turbo"))
    boost = read_file(os.path.join(base, "cpufreq", "boost"))
    hugepages = read_file(os.path.join(sys_root, "kernel", "mm", "transparent_hugepage",
                                       "enabled"))
    return {
        "governors": governors, "scaling_driver": drivers,
        "energy_performance_preference": epp,
        "intel_pstate_no_turbo": None if no_turbo is None else no_turbo.strip(),
        "cpufreq_boost": None if boost is None else boost.strip(),
        "scaling_cur_freq_khz": ({"min": min(frequencies),
                                  "median": int(statistics.median(frequencies)),
                                  "max": max(frequencies), "cpus": len(frequencies)}
                                 if frequencies else None),
        "cpuinfo_max_freq_khz": max(maximum) if maximum else None,
        "transparent_hugepage": None if hugepages is None else hugepages.strip(),
    }


def parse_power(text: str) -> Dict[str, Any]:
    """``pmset -g batt``: the power source and the charge."""
    source = re.search(r"Now drawing from '([^']+)'", text or "")
    charge = re.search(r"(\d+)%", text or "")
    name = source.group(1) if source else None
    return {"power_source": name,
            "on_ac_power": None if name is None else name == "AC Power",
            "battery_percent": int(charge.group(1)) if charge else None}


def parse_powermode(text: str) -> Optional[int]:
    """``pmset -g``: ``powermode`` (0 automatic, 1 low, 2 high) or ``lowpowermode``."""
    found = re.search(r"^\s*(?:powermode|lowpowermode)\s+(\d+)\s*$", text or "",
                      re.MULTILINE)
    return int(found.group(1)) if found else None


def cpu_state_darwin() -> Dict[str, Any]:
    power = run_text(["pmset", "-g", "batt"]) or ""
    settings = run_text(["pmset", "-g"]) or ""
    thermal = run_text(["pmset", "-g", "therm"]) or ""
    mode = parse_powermode(settings)
    limit = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", thermal)
    return {**parse_power(power), "powermode": mode,
            "low_power_mode": None if mode is None else mode == 1,
            "cpu_speed_limit": int(limit.group(1)) if limit else None,
            "thermal_report": " | ".join(l.strip() for l in thermal.splitlines()
                                         if l.strip())[:240],
            "frequency": "unreadable without privileges"}


def cpu_state(platform: Optional[str] = None,
              cpus: Optional[Iterable[int]] = None) -> Dict[str, Any]:
    platform = platform or sys.platform
    state = {"utc": utc(), "loadavg": loadavg()}
    state.update(cpu_state_darwin() if platform == "darwin" else cpu_state_linux(cpus=cpus))
    return state


def darwin_condition_reasons(state: Optional[Mapping[str, Any]],
                             require_ac_power: bool = True) -> List[str]:
    """Why a Mac's power or thermal state disqualifies a timing, from :func:`cpu_state_darwin`
    or the bench's ``host_state``: on battery or an unreadable power source (when AC
    power is required), low power mode (when required), a CPU speed limit below 100.
    A missing speed limit is "no thermal warning recorded", not a limit."""
    reasons: List[str] = []
    if not state:
        return ["the power and thermal state was not read"]
    if require_ac_power:
        if state.get("on_ac_power") is None:
            reasons.append("the power source is unreadable")
        elif state.get("on_ac_power") is False:
            reasons.append(f"on {state.get('power_source') or 'battery'}")
        if state.get("low_power_mode") is True:
            reasons.append("low power mode")
    limit = state.get("cpu_speed_limit")
    if limit is not None and int(limit) < 100:
        reasons.append(f"CPU speed limit {limit} (thermal)")
    return reasons


def parse_ps_cpu(text: str) -> List[Tuple[int, float, str]]:
    """``ps -axo pid=,%cpu=,comm=`` -> [(pid, cpu percent, command)]."""
    out: List[Tuple[int, float, str]] = []
    for line in (text or "").splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        try:
            out.append((int(parts[0]), float(parts[1]), parts[2].strip()))
        except ValueError:
            continue
    return out


def busy_daemons(rows: Iterable[Tuple[int, float, str]],
                 pattern: "re.Pattern[str]" = DARWIN_DAEMONS,
                 threshold: float = DARWIN_DAEMON_CPU_MAX) -> List[str]:
    """The background daemons above ``threshold`` per cent CPU, as ``pid command cpu%``."""
    return [f"{pid} {os.path.basename(command)[:80]} {cpu:.0f}%" for pid, cpu, command in rows
            if cpu > threshold and pattern.search(command)]


def darwin_daemons() -> Optional[List[str]]:
    text = run_text(["ps", "-axo", "pid=,%cpu=,comm="], timeout=15)
    return None if text is None else busy_daemons(parse_ps_cpu(text))


def parse_ioreg_accelerator(text: str) -> Dict[str, Any]:
    """``ioreg -r -d 1 -c IOAccelerator``: the Apple GPU's model, core count and the
    ``PerformanceStatistics`` utilisation and memory fields. Readable without root;
    clocks, temperature and power are not exposed here (``powermetrics`` needs root)."""
    out: Dict[str, Any] = {}
    model = re.search(r'"model"\s*=\s*"([^"]*)"', text or "")
    cores = re.search(r'"gpu-core-count"\s*=\s*(\d+)', text or "")
    out["model"] = model.group(1) if model else None
    out["gpu_core_count"] = int(cores.group(1)) if cores else None
    statistics_block = re.search(r'"PerformanceStatistics"\s*=\s*\{([^}]*)\}', text or "")
    fields = {}
    if statistics_block:
        for key, value in re.findall(r'"([^"]+)"=(\d+)', statistics_block.group(1)):
            fields[key] = int(value)
    for key, name in (("Device Utilization %", "device_utilization_percent"),
                      ("Renderer Utilization %", "renderer_utilization_percent"),
                      ("Tiler Utilization %", "tiler_utilization_percent"),
                      ("In use system memory", "in_use_system_memory_bytes"),
                      ("Alloc system memory", "alloc_system_memory_bytes")):
        out[name] = fields.get(key)
    return out


def apple_gpu_state() -> Dict[str, Any]:
    text = run_text(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"], timeout=15)
    if text is None:
        return {"utc": utc(), "readable": False, "why": "ioreg is unreadable"}
    found = parse_ioreg_accelerator(text)
    return {"utc": utc(), "readable": found.get("device_utilization_percent") is not None,
            "how": "ioreg IOAccelerator PerformanceStatistics",
            "not_recorded": "clocks, temperature and power: powermetrics needs root",
            "state": found}


#: The GPU facts read before and after each GPU row. The throttle field changed name
#: between driver generations; :func:`nvidia_query_fields` picks the one this driver has.
GPU_FIELDS = ("index", "uuid", "clocks.sm", "clocks.mem", "clocks.max.sm",
              "clocks.applications.graphics", "temperature.gpu", "power.draw",
              "enforced.power.limit", "pstate", "{reasons}", "utilization.gpu",
              "memory.used")


def nvidia_query_fields(help_text: Optional[str]) -> List[str]:
    reasons = ("clocks_event_reasons.active"
               if "clocks_event_reasons.active" in (help_text or "")
               else "clocks_throttle_reasons.active")
    return [field.format(reasons=reasons) for field in GPU_FIELDS]


def parse_gpu_csv(fields: Sequence[str], text: str) -> List[Dict[str, str]]:
    rows = []
    for line in (text or "").splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) == len(fields):
            rows.append(dict(zip(fields, cells)))
    return rows


_QUERY_FIELDS_CACHE: Dict[str, List[str]] = {}


def gpu_query_fields() -> List[str]:
    if "fields" not in _QUERY_FIELDS_CACHE:
        _QUERY_FIELDS_CACHE["fields"] = nvidia_query_fields(
            run_text(["nvidia-smi", "--help-query-gpu"], timeout=30))
    return _QUERY_FIELDS_CACHE["fields"]


def gpu_state(gpu: Optional[str], platform: Optional[str] = None) -> Dict[str, Any]:
    """The pinned NVIDIA GPU's clocks, temperature, power and throttle reasons, or on
    macOS the Apple GPU's utilisation (:func:`apple_gpu_state`)."""
    if (platform or sys.platform) == "darwin":
        return apple_gpu_state()
    if gpu is None or not nvidia_smi_available():
        return {"utc": utc(), "readable": False}
    fields = gpu_query_fields()
    text = run_text(["nvidia-smi", f"--id={gpu}", "--query-gpu=" + ",".join(fields),
                     "--format=csv,noheader"], timeout=30)
    rows = parse_gpu_csv(fields, text or "")
    return {"utc": utc(), "readable": bool(rows), "fields": fields,
            "state": rows[0] if rows else None,
            "raw": None if rows else (text or "")[-400:]}


def gpu_trace_command(gpu: str, every_ms: int = 1000) -> List[str]:
    return ["nvidia-smi", f"--id={gpu}", "--query-gpu=timestamp," + ",".join(gpu_query_fields()),
            "--format=csv,noheader", f"-lms={every_ms}"]


# ---------------------------------------------------------------------------
# The quiet gate
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class GateConfig:
    """Every threshold the gate reads, resolved once by the ladder."""
    platform: str
    gpu: Optional[str]
    uid: Optional[int]
    cpu_max: float
    mem_min_gib: float
    account_max: int
    cap: Optional[int]
    gpu_metric: str = "load1"
    meep_metric: str = "busy"
    hold_gpu_s: float = 120.0
    hold_meep_s: float = 30.0
    retry_s: float = 15.0
    give_up_s: float = 43200.0
    scope: str = "host"
    sample_s: float = SAMPLE_S
    check_device: bool = True
    #: macOS only: a gate reading on battery, in low power mode (both when this is
    #: set) or under a CPU speed limit is not quiet; busy background daemons are not
    #: quiet; and, when set, an Apple GPU above this Device Utilization % is not quiet.
    require_ac_power: bool = True
    check_daemons: bool = True
    apple_gpu_util_max: Optional[float] = None


def default_cpu_max(logical_cpus_in_scope: Optional[int]) -> int:
    """10 % of the logical CPUs in scope, at least one."""
    return max(1, int(round(0.10 * float(logical_cpus_in_scope or 0))))


def default_mem_min_gib(total_gib: Optional[float]) -> float:
    return round(0.25 * float(total_gib or 0.0), 1)


def reading(config: GateConfig, kind: str,
            busy_reader: Optional[Callable[[], Dict[str, Any]]] = None,
            memory_reader: Optional[Callable[[], Dict[str, Any]]] = None,
            device_reader: Optional[Callable[[], Dict[str, Any]]] = None,
            load_reader: Optional[Callable[[], Optional[float]]] = None,
            power_reader: Optional[Callable[[], Dict[str, Any]]] = None,
            daemon_reader: Optional[Callable[[], Optional[List[str]]]] = None,
            apple_gpu_reader: Optional[Callable[[], Dict[str, Any]]] = None
            ) -> Dict[str, Any]:
    """One reading of everything ``quiet`` compares, each through an injectable reader."""
    metric = config.gpu_metric if kind == "gpu" else config.meep_metric
    if config.platform == "darwin":
        out_darwin: Dict[str, Any] = {
            "power": (power_reader or cpu_state_darwin)(),
            "daemons": ((daemon_reader or darwin_daemons)() if config.check_daemons
                        else []),
        }
        if config.apple_gpu_util_max is not None and kind == "gpu":
            out_darwin["apple_gpu"] = (apple_gpu_reader or apple_gpu_state)()
    else:
        out_darwin = {}
    out: Dict[str, Any] = {"utc": utc(), "kind": kind, "metric": metric}
    if metric == "busy":
        busy = (busy_reader or (lambda: busy_cpus(config.scope, config.sample_s,
                                                  config.platform)))()
        out["busy"] = busy
        out["cpu"] = busy.get("busy_cpus")
    else:
        out["cpu"] = (load_reader or load1)()
    out["memory"] = (memory_reader or (lambda: memory(config.platform)))()
    if config.check_device:
        out["device"] = (device_reader or (lambda: device_state(
            config.gpu, config.uid, config.platform)))()
    else:
        out["device"] = {"device_free": True, "account_gpu_processes": 0,
                         "how": "not checked"}
    out.update(out_darwin)
    return out


def verdict(config: GateConfig, observed: Mapping[str, Any]) -> Tuple[bool, List[str]]:
    """Quiet or not, and every reason it is not. A value that could not be read is a
    reason, never a pass."""
    reasons: List[str] = []
    cpu = observed.get("cpu")
    if cpu is None:
        why = ((observed.get("busy") or {}).get("why") if observed.get("metric") == "busy"
               else "load average unreadable")
        reasons.append(f"cpu unreadable ({why})")
    elif not float(cpu) < float(config.cpu_max):
        reasons.append(f"{observed.get('metric')} {cpu} >= {config.cpu_max}")
    available = (observed.get("memory") or {}).get("available_gib")
    if available is None:
        reasons.append("available memory unreadable")
    elif not float(available) > float(config.mem_min_gib):
        reasons.append(f"available memory {available} GiB <= {config.mem_min_gib}")
    device = observed.get("device") or {}
    account = device.get("account_gpu_processes")
    if account is None:
        reasons.append(f"account GPU processes unreadable ({device.get('why')})")
    else:
        if int(account) > int(config.account_max):
            reasons.append(f"account GPU processes {account} > {config.account_max}")
        if config.cap is not None and not int(account) < int(config.cap):
            reasons.append(f"account GPU processes {account} would pass the cap "
                           f"{config.cap}")
    free = device.get("device_free")
    if free is None:
        reasons.append(f"device occupancy unreadable ({device.get('why')})")
    elif not free:
        reasons.append(f"device busy: {device.get('device_processes')}")
    if config.platform == "darwin":
        reasons += darwin_condition_reasons(observed.get("power"), config.require_ac_power)
        daemons = observed.get("daemons")
        if daemons is None:
            reasons.append("background daemons unreadable (ps)")
        elif daemons:
            reasons.append(f"background daemons busy: {daemons}")
        if config.apple_gpu_util_max is not None and "apple_gpu" in observed:
            util = ((observed.get("apple_gpu") or {}).get("state") or {}).get(
                "device_utilization_percent")
            if util is None:
                reasons.append("Apple GPU utilisation unreadable (ioreg)")
            elif not float(util) <= float(config.apple_gpu_util_max):
                reasons.append(f"Apple GPU Device Utilization {util} % > "
                               f"{config.apple_gpu_util_max}")
    return not reasons, reasons


def quiet(config: GateConfig, kind: str, **readers: Any) -> Tuple[bool, Dict[str, Any]]:
    observed = reading(config, kind, **readers)
    ok, reasons = verdict(config, observed)
    observed["quiet"] = ok
    observed["reasons"] = reasons
    return ok, observed


class GateGaveUp(RuntimeError):
    """The box did not go quiet within ``give_up_s``."""


def gate(config: GateConfig, kind: str, label: str, say: Callable[[str], None],
         box: Callable[[], str], sleep: Callable[[float], None] = time.sleep,
         clock: Callable[[], float] = time.time, **readers: Any) -> float:
    """Quiet, hold, quiet again; retry every ``retry_s``; log every 600 s. Returns the
    seconds waited; raises :class:`GateGaveUp` at ``give_up_s``."""
    hold = config.hold_gpu_s if kind == "gpu" else config.hold_meep_s
    started = clock()
    next_log = 0.0
    while True:
        ok, observed = quiet(config, kind, **readers)
        if ok:
            sleep(hold)
            ok, observed = quiet(config, kind, **readers)
            if ok:
                return clock() - started
        waited = clock() - started
        if waited >= next_log:
            say(f"not quiet before {label} (waited {waited:.0f}s): "
                f"{'; '.join(observed.get('reasons') or [])} | {box()}")
            next_log = waited + 600.0
        if waited >= config.give_up_s:
            raise GateGaveUp(f"gate gave up before {label} after {waited:.0f}s")
        sleep(config.retry_s)


def box_string(config: GateConfig, busy: Optional[Dict[str, Any]] = None) -> str:
    """One line of host state, keyed as the 2026-09-28 box string with ``busy_cores``
    renamed ``busy_cpus`` (the unit is logical CPUs busy)."""
    load = loadavg()
    busy = busy if busy is not None else busy_cpus(config.scope, config.sample_s,
                                                    config.platform)
    mem = memory(config.platform)
    device = device_state(config.gpu, config.uid, config.platform)
    parts = [f"load={' '.join(str(v) for v in load) if load else 'unreadable'}",
             f"busy_cpus={busy.get('busy_cpus') if busy.get('busy_cpus') is not None else 'unreadable'}",
             f"free_gb={mem.get('available_gib') if mem.get('available_gib') is not None else 'unreadable'}"]
    if config.platform == "darwin":
        parts.append(f"metal_neighbours={len(device.get('device_processes') or [])}")
    else:
        own, others = gpu_brief(config.gpu)
        parts.append(f"gpu{box_label(config.gpu)}={own}")
        parts.append(f"others={others}")
    parts.append(f"account={device.get('account_gpu_processes')}")
    return " ".join(parts)


_UUID_INDEX: Dict[str, str] = {}


def box_label(gpu: Optional[str]) -> str:
    """The pinned GPU as the box string keys it: its INDEX (``gpu0=``), as on
    2026-09-28 and as ``audit_timing_validity.py`` parses it, also when the ladder pins
    it by UUID (``--gpu auto`` resolves to one). The UUID itself when no index is
    listed for it."""
    if gpu is None or not str(gpu).startswith("GPU-"):
        return str(gpu)
    if gpu not in _UUID_INDEX:
        found = {uuid: index for index, uuid in (gpu_uuids() or {}).items()}
        if gpu in found:
            _UUID_INDEX[gpu] = found[gpu]
    return _UUID_INDEX.get(gpu, str(gpu))


def box_keys(text: str) -> List[str]:
    """The key set of a box string (``load``, ``busy_cpus``, ...), in order."""
    return re.findall(r"(?:^|\s)([A-Za-z_][A-Za-z0-9_]*)=", text or "")


def nice_level() -> int:
    return os.nice(0)


def selftest_readers(platform: Optional[str] = None) -> Dict[str, Any]:
    """Every reader once, read-only: what ``--selftest`` prints."""
    platform = platform or sys.platform
    out: Dict[str, Any] = {"platform": platform, "nice": nice_level(),
                           "topology": topology(platform), "load1": load1(),
                           "memory": memory(platform),
                           "busy_host": busy_cpus("host", SAMPLE_S, platform),
                           "slurm": slurm_facts(),
                           "cpu_state": cpu_state(platform)}
    if platform != "darwin":
        out["busy_allocation"] = busy_cpus("allocation", SAMPLE_S, platform)
    else:
        out["busy_daemons"] = darwin_daemons()
        out["apple_gpu"] = apple_gpu_state()
        out["darwin_process_priority"] = darwin_process_priority()
    return json.loads(json.dumps(out, default=str))


def darwin_process_priority() -> Optional[int]:
    """``getpriority(PRIO_DARWIN_PROCESS, 0)``: non-zero confines this process and its
    children to the efficiency cores (background QoS). ``None`` where unreadable."""
    which = getattr(os, "PRIO_DARWIN_PROCESS", None)
    if which is None or sys.platform != "darwin":
        return None
    try:
        return int(os.getpriority(which, 0))
    except OSError:
        return None
