#!/usr/bin/env python
"""The identical-case timing ladder: the package's GPU routes and stock MEEP, interleaved.

WHAT A LADDER IS. One run on one host, in one quiet window, that times the package's
GPU routes (``bench_fused_products.py``, or ``bench_timing_case.py`` for a case of
``timing_cases.py``) and stock MEEP (``bench_meep_identical_case.py`` under ``mpirun`` or
``srun``) on the SAME simulation, interleaved per size and pass so that drift lands on
both sides alike, every row behind a quiet gate and recorded in one ledger
(``ladder_rows.jsonl``) with its verdict. The tables are built afterwards from that
ledger (``build_identical_case_table.py``) and nothing is transcribed.

THE SEQUENCE, per (tier, size, pass, case):

    GPU route 1 -> MEEP block A -> GPU route 2 -> MEEP block B -> array control
    -> MEEP block C -> MEEP controls

THE ORDER OF WORK, tiers outermost, then passes, then sizes, then cases:

    preflight  every row kind once at the preflight size with short windows; a GPU row
               that does not launch what its route expects, bit-identically, with the
               deposit-repair bracket its case declares, stops the ladder here
    digests    the harness lift's geometry digest per planned (case, size): the digest
               of record every MEEP row at that size must carry
    bindprobe  MEEP at the probe ranks, bound against unbound, two passes; a rule fixed
               in advance chooses the binding of every later MEEP row
    priority   the priority sizes, every pass
    bindcheck  the binding not chosen, at the priority sizes, one pass
    rest       the remaining sizes, every pass
    largest    the largest size, every pass

THE GATE stands in front of every row that starts a process: CPU quiet (load1 for GPU
rows; for MEEP rows the busy logical CPUs over a 3 s sample, summed per CPU,
``timing_host.busy_from_samples``), available memory, the account's GPU processes, and
the device free, held for two readings. A reading that cannot be taken is not quiet.

USAGE

    timing_ladder.py --stamp S --python PY --gpu 0 --meep-config-log L [...]   live
    timing_ladder.py ... --dry-run [--json]       the plan: no process, no device, no file
    timing_ladder.py ... --selftest [--launcher-probe]   every reader once, read-only

Every parameter is a flag, a ``KEY=VALUE`` line of ``--params FILE``, an environment
variable ``MGPU_TIMING_<KEY>``, a site variable ``MGPU_SITE_*`` where one applies, or a
detected or built-in default, in that order of precedence; ``environment.json`` records
each resolved value and where it came from. ``docs/development/timing.md`` is the
manual.

PROGRESS. ``<run root>/ladder_<stamp>/progress.log`` carries one flushed line per row
start and end; ``box_trace.log`` one line of host state every ``--monitor-every-s``
seconds; the ledger one JSON line per row as it lands, so an interrupted ladder keeps
every row before the interruption and ``--resume`` continues it.
"""

from __future__ import annotations

import argparse
import dataclasses
import getpass
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import timing_cases  # noqa: E402
import timing_host  # noqa: E402
import timing_records  # noqa: E402

TIERS = ("preflight", "digests", "bindprobe", "priority", "bindcheck", "rest", "largest")
CONTROLS = ("native", "threads2", "threads4", "hwthread", "split")
ROUTE_TABLES = {"default": "triton", "cuda": "cuda", "metal": "metal"}
NVIDIA_ROUTES = ("default", "cuda")
#: The drivers a ``--drivers-mode copy`` ladder copies into its run, by name.
DRIVER_FILES = ("timing_ladder.py", "timing_host.py", "timing_records.py",
                "timing_cases.py", "bench_meep_identical_case.py",
                "digest_harness_lift.py", "gpu_array_control.py", "bench_timing_case.py",
                "build_identical_case_table.py", "audit_timing_validity.py")
PREFLIGHT_GPU_FLAGS = ("--repeats 2 --target-seconds 0.5 --monitors attached "
                       "--memory-budget-bytes 30000000000")
#: The GPU rows' bench flags on an NVIDIA host (the 2026-09-28 instrument). On macOS the
#: memory budget is derived from the host's memory instead (:func:`darwin_gpu_flags`).
LINUX_GPU_FLAGS = ("--repeats 6 --target-seconds 2.0 --monitors attached "
                   "--memory-budget-bytes 30000000000")
#: The share of a Mac's memory the frozen legs of one GPU row may take: the A6000
#: budget (30 GB) would be 94 % of a 32 GB Mac and more than a 16 or 24 GB one has.
DARWIN_MEMORY_BUDGET_SHARE = 0.6
#: Seconds per GPU row by cell count: the 2026-09-26 campaign's measured row times.
GPU_SECONDS_BY_CELLS = {110_592: 60, 262_144: 72, 512_000: 97, 1_124_864: 158,
                        2_097_152: 250, 4_096_000: 480, 7_077_888: 950,
                        11_239_424: 1400}
#: Stock MEEP's standing rates by rank count, for the budget only.
MEEP_RATE_BY_RANKS = {1: 42, 8: 260, 16: 400, 24: 450, 32: 500, 48: 500}
#: Variables a row's environment never inherits from the launching shell. Beyond these
#: named ones, EVERY inherited ``MEEP_GPU_*`` the ladder does not set itself is unset
#: (``MEEP_GPU_METAL_RESIDENCY``, ``MEEP_GPU_KERNEL_TABLE``, ``MEEP_GPU_WARM``, the
#: probe switches, ...), and recorded in the row's environment delta.
PACKAGE_SWITCHES = ("MEEP_GPU_DISPATCH", "MEEP_GPU_FUSED", "MEEP_GPU_FUSE_ARMS",
                    "MEEP_GPU_BACKEND_PREFERENCE", "MEEP_GPU_DISPATCH_LOG")
#: PyTorch switches that change what a Metal row runs (the CPU fallback for operations
#: MPS lacks, the MPS allocator's limits): unset for Metal rows, recorded.
PYTORCH_MPS_SWITCHES = ("PYTORCH_ENABLE_MPS_FALLBACK",)
PYTORCH_MPS_PREFIX = "PYTORCH_MPS_"
EXIT_REFUSED = 3
EXIT_GATE = 4
EXIT_PREFLIGHT = 5
EXIT_DIGEST = 6
EXIT_NICE = 7
EXIT_SIGNAL = 8
#: The rc a row reads when its process never started (the launcher or interpreter
#: could not be executed); such a row is struck and gets no ``done`` marker.
RC_NOT_STARTED = 127


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def refuse(message: str, code: int = EXIT_REFUSED) -> "SystemExit":
    """A refusal by name. Raised, so a test can read the message."""
    error = SystemExit(f"REFUSING: {message}")
    error.code = code  # type: ignore[attr-defined]
    error.refusal = message  # type: ignore[attr-defined]
    return error


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Param:
    name: str
    kind: str                     # str | int | float | list | bool | path
    default: Any = None
    help: str = ""
    choices: Optional[Tuple[str, ...]] = None
    site: Optional[str] = None    # MGPU_SITE_* variable supplying a default
    repeat: bool = False

    @property
    def flag(self) -> str:
        return "--" + self.name.lower().replace("_", "-")

    @property
    def dest(self) -> str:
        return self.name.lower()


PARAMS: Tuple[Param, ...] = (
    Param("STAMP", "str", None, "names this run: <run root>/ladder_<stamp>"),
    Param("PYTHON", "path", None, "interpreter of the GPU rows, the digests and the "
          "reference MEEP; it must import single-precision MEEP", site="MGPU_SITE_PYTHON"),
    Param("GPU", "str", None, "device index of the NVIDIA routes, or 'auto' when exactly "
          "one GPU is allocated"),
    Param("REFUSE_GPUS", "list", "", "device indices this ladder must never use"),
    Param("MEEP_LAUNCHER", "str", "mpirun", "how MEEP ranks start",
          choices=("mpirun", "srun")),
    Param("MPIRUN", "path", None, "the launcher binary; mpirun beside --python by "
          "default, or srun"),
    Param("SRUN_MPI", "str", None, "srun's --mpi value (pmix, pmi2, ...); required with "
          "--meep-launcher srun"),
    Param("MEEP_CONFIG_LOG", "str", None, "the reference MEEP build's config.log, or "
          "'none'"),
    Param("MEEP_BUILD_RECORD", "str", None, "the reference build's build_record.json "
          "(build_meep_133_native.sh / timing_records.py meep-build-record), or 'none'"),
    Param("MEEP_BUILD", "list", "", "another MEEP build, NAME=PYTHON[,LAUNCHER[,RECORD]]; "
          "repeatable", repeat=True),
    Param("RUN_ROOT", "path", None, "where ladder_<stamp>/ goes; refused inside any tree "
          "the ladder reads"),
    Param("TREE_SRC", "path", None, "the tree under test; this repository by default"),
    Param("TREE_MODE", "str", "snapshot", "snapshot copies the tree into the run; direct "
          "runs it in place", choices=("snapshot", "direct")),
    Param("TREE_LAYOUT", "str", "auto", "release (meep_gpu/ at the root) or development "
          "(apps/api/meep_gpu/); found by name when auto",
          choices=("auto", "release", "development")),
    Param("SNAPSHOT_EXCLUDE", "list",
          "__pycache__ *.pyc .git .DS_Store parity/meep_gpu/results",
          "names (fnmatch) and relative paths the snapshot and its manifest leave out"),
    Param("DRIVERS_MODE", "str", None, "tree: the drivers are the snapshot's own "
          "parity/meep_gpu; copy: they are copied into <out>/drivers",
          choices=("tree", "copy")),
    Param("LIBCUDA_STUB", "path", None, "directory holding libcuda.so.1 for Triton"),
    Param("CORPUS_ROOT", "path", None, "MEEP's python/ source tree for the corpus; "
          "$MGPU_SITE_MEEP_SOURCE/python when it exists"),
    Param("EXTRA_PYTHONPATH", "list", "", "directories added after the harness on the "
          "GPU rows' PYTHONPATH; {harness} and {api} name the tree that runs"),
    Param("ALLOW_UNCERTIFIED", "int", 0, "1 exports MEEP_GPU_ALLOW_UNCERTIFIED=1 to the "
          "GPU rows; 0 exports MEEP_GPU_ALLOW_UNCERTIFIED=0 to the NVIDIA rows (certified "
          "devices and toolchains only) and unsets it for every other row",
          choices=("0", "1")),
    Param("SUBNORMAL_POLICY", "str", None, "MEEP_GPU_SUBNORMAL_POLICY for the GPU rows "
          "('keep' on Linux, the package default on macOS; 'default' leaves it unset)"),
    Param("ROUTES", "list", None, "GPU routes: default, cuda, metal, or none"),
    Param("EXPECT_LAUNCHED", "list", "", "ROUTE:CASE=TABLES the preflight must launch; "
          "repeatable. Absent: the preflight row pins it", repeat=True),
    Param("CASES", "list", "pml_3d", "timing cases (timing_cases.py)"),
    Param("PRIORITY_RES", "list", None, "priority sizes as resolutions of the first case"),
    Param("REST_RES", "list", None, "rest sizes as resolutions of the first case"),
    Param("LARGEST_RES", "list", None, "largest sizes as resolutions of the first case"),
    Param("PROBE_RES", "int", None, "binding-probe size as a resolution of the first case"),
    Param("PREFLIGHT_RES", "int", None, "preflight size as a resolution of the first case"),
    Param("SIZES", "list", None, "priority sizes in cells (replaces --priority-res)"),
    Param("REST_SIZES", "list", "1124864,4096000,262144,110592", "rest sizes in cells"),
    Param("LARGEST_SIZES", "list", "11239424", "largest sizes in cells"),
    Param("PRIORITY_SIZES", "list", "512000,2097152,7077888", "priority sizes in cells"),
    Param("PROBE_SIZE", "int", 2097152, "binding-probe size in cells"),
    Param("PREFLIGHT_SIZE", "int", 110592, "preflight size in cells"),
    Param("PLATFORM", "str", None, "linux or darwin; detected", choices=("linux", "darwin")),
    Param("MPI_IMPL", "str", None, "openmpi, mpich, intelmpi, srun or unknown; detected"),
    Param("PHYSICAL_CORES", "int", None, "physical cores in scope (P-cores on Apple "
          "silicon); detected"),
    Param("LOGICAL_CPUS", "int", None, "logical CPUs in scope; detected"),
    Param("SOCKETS", "int", None, "CPU packages in scope; detected (srun's bound modes "
          "balance ranks across them)"),
    Param("THREADS_PER_CORE", "int", None, "hardware threads per core; detected, or "
          "logical / physical"),
    Param("MEMORY_TOTAL_GB", "float", None, "total memory in GiB; detected"),
    Param("RANKS", "str", None, "MEEP rank blocks A|B|C, comma lists"),
    Param("BINDING", "str", "probe", "probe, bound or unbound",
          choices=("probe", "bound", "unbound")),
    Param("PROBE_RANKS", "list", None, "rank counts of the binding probe"),
    Param("FLAGS_BOUND", "str", None, "launcher flags of the bound mode"),
    Param("FLAGS_UNBOUND", "str", None, "launcher flags of the unbound mode"),
    Param("FLAGS_OVER", "str", None, "launcher flags above the physical cores"),
    Param("FLAGS_BOUND_HWTHREAD", "str", None, "launcher flags of bound-hwthread"),
    Param("FLAGS_BOUND_PE", "str", None, "launcher flags of T threads per rank "
          "({threads} is substituted)"),
    Param("CONTROLS", "list", "native,threads2,threads4,hwthread,split",
          "MEEP controls after block C, or none"),
    Param("ACCOUNT", "str", None, "the account whose GPU processes are counted; this "
          "user by default; matched by uid"),
    Param("ACCOUNT_MAX", "int", 0, "account GPU processes a row may start beside"),
    Param("CAP", "int", None, "account GPU processes above which the guard kills this "
          "ladder's row; off by default"),
    Param("GATE_MEM_MIN_GB", "float", None, "available memory a row needs, GiB; 25 % of "
          "total by default"),
    Param("GATE_CPU_MAX", "float", None, "CPU metric below which a row may start; 10 % of "
          "the logical CPUs in scope by default"),
    Param("GATE_GPU_METRIC", "str", None, "load1 on a Linux host; busy on macOS and with "
          "--gate-scope allocation (a host-wide load average sees other jobs)",
          choices=("load1", "busy")),
    Param("GATE_MEEP_METRIC", "str", "busy", "", choices=("load1", "busy")),
    Param("GATE_HOLD_GPU_S", "float", 120.0, ""),
    Param("GATE_HOLD_MEEP_S", "float", 30.0, ""),
    Param("GATE_RETRY_S", "float", 15.0, ""),
    Param("GATE_GIVE_UP_S", "float", 43200.0, ""),
    Param("GATE_SCOPE", "str", "host", "busy CPUs over the host or over this process's "
          "allocation", choices=("host", "allocation")),
    Param("GATE_DARWIN_DAEMONS", "int", 1, "macOS: Spotlight, backup, update and media "
          "daemons above 10 % CPU make the host not quiet", choices=("0", "1")),
    Param("GATE_APPLE_GPU_UTIL_MAX", "float", None, "macOS: an Apple GPU Device "
          "Utilization % (ioreg) above which a GPU row may not start; recorded only "
          "when unset"),
    Param("MONITOR_EVERY_S", "float", 15.0, "box trace interval"),
    Param("GPU_TRACE_S", "float", 1.0, "GPU state sampling interval during GPU rows"),
    Param("CPU_TRACE_S", "float", 1.0, "CPU state sampling interval during rows"),
    Param("PASSES", "int", 3, ""),
    Param("TIERS", "list", " ".join(TIERS), ""),
    Param("ARRAY_CONTROL", "str", "extract", "", choices=("extract", "row")),
    Param("GPU_FLAGS", "str", None, "bench_fused_products.py flags; must time with "
          "--monitors attached. Linux: the 2026-09-28 flags; macOS: the same with a memory "
          "budget of 0.6 x the host's memory"),
    Param("ALLOW_NICE", "int", 0, "", choices=("0", "1")),
    Param("REQUIRE_AC_POWER", "int", 1, "macOS: refuse on battery or in low power mode",
          choices=("0", "1")),
)
PARAM_BY_NAME = {p.name: p for p in PARAMS}


def _split_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out: List[str] = []
        for item in value:
            out.extend(_split_list(item))
        return out
    return [part for part in re.split(r"[,\s]+", str(value).strip()) if part]


def read_params_file(path: Optional[str]) -> Dict[str, str]:
    """``KEY=VALUE`` lines; ``#`` comments; a repeated key accumulates (for repeatables)."""
    if not path:
        return {}
    values: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            key, sep, value = line.partition("=")
            key = key.strip().upper()
            if not sep or key not in PARAM_BY_NAME:
                raise refuse(f"{path}:{number}: {line!r} is not KEY=VALUE of a known "
                             "parameter")
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            if PARAM_BY_NAME[key].repeat and key in values:
                values[key] = values[key] + "\n" + value
            else:
                values[key] = value
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    for param in PARAMS:
        if param.repeat:
            parser.add_argument(param.flag, dest=param.dest, action="append", default=None,
                                help=param.help)
        else:
            parser.add_argument(param.flag, dest=param.dest, default=None, help=param.help)
    parser.add_argument("--params", default=None, help="KEY=VALUE file of parameters")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument("--json", action="store_true", help="with --dry-run: the plan "
                        "as JSON")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--launcher-probe", action="store_true", dest="launcher_probe")
    parser.add_argument("--resume", action="store_true")
    return parser


def resolve_params(args: argparse.Namespace, environ: Dict[str, str]
                   ) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Every parameter's value and its source: flag, params-file, environment, site,
    or default (detected and derived values are added by :func:`build_config`)."""
    from_file = read_params_file(args.params)
    values: Dict[str, Any] = {}
    sources: Dict[str, str] = {}
    for param in PARAMS:
        given = getattr(args, param.dest)
        if given is not None:
            values[param.name] = given
            sources[param.name] = "flag"
        elif param.name in from_file:
            raw = from_file[param.name]
            values[param.name] = raw.split("\n") if param.repeat else raw
            sources[param.name] = "params-file"
        elif f"MGPU_TIMING_{param.name}" in environ:
            raw = environ[f"MGPU_TIMING_{param.name}"]
            values[param.name] = raw.split("\n") if param.repeat else raw
            sources[param.name] = "environment"
        elif param.site and environ.get(param.site):
            values[param.name] = environ[param.site]
            sources[param.name] = f"site:{param.site}"
        else:
            values[param.name] = param.default
            sources[param.name] = "default"
        if param.choices and values[param.name] is not None \
                and str(values[param.name]) not in param.choices:
            raise refuse(f"{param.flag} {values[param.name]!r}: one of "
                         f"{', '.join(param.choices)}")
    return values, sources


def _int(value: Any, name: str) -> int:
    try:
        return int(str(value))
    except ValueError:
        raise refuse(f"--{name.lower().replace('_', '-')} {value!r} is not an integer")


def _float(value: Any, name: str) -> float:
    try:
        return float(str(value))
    except ValueError:
        raise refuse(f"--{name.lower().replace('_', '-')} {value!r} is not a number")


# ---------------------------------------------------------------------------
# Detection (no process is started: files and ctypes only)
# ---------------------------------------------------------------------------

def detect_platform() -> str:
    return "darwin" if sys.platform == "darwin" else "linux"


def detect_topology(platform: str) -> Dict[str, Any]:
    if platform != detect_platform():
        return {"method": None}
    return timing_host.topology(platform)


def detect_memory_total(platform: str) -> Optional[float]:
    if platform != detect_platform():
        return None
    if platform == "darwin":
        total = timing_host.sysctl_value("hw.memsize")
        return round(int(total) / timing_host.GIB, 1) if total else None
    parsed = timing_host.parse_meminfo(timing_host.read_file("/proc/meminfo") or "")
    return round(parsed["MemTotal"] * 1024 / timing_host.GIB, 1) if "MemTotal" in parsed \
        else None


def detect_mpi_impl_from_files(launcher: str, kind: str) -> str:
    """The MPI implementation from the files beside the launcher; no process started.
    The live run reads ``--version`` instead and records both."""
    if kind == "srun" or os.path.basename(launcher) == "srun":
        return "srun"
    directory = os.path.dirname(os.path.abspath(launcher)) if os.sep in launcher else ""
    if directory:
        if os.path.exists(os.path.join(directory, "ompi_info")):
            return "openmpi"
        if os.path.exists(os.path.join(directory, "mpichversion")) or \
                os.path.exists(os.path.join(directory, "hydra_pmi_proxy")):
            return "mpich"
        if os.path.exists(os.path.join(directory, "impi_info")):
            return "intelmpi"
    return "unknown"


# ---------------------------------------------------------------------------
# The configuration
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class MeepBuild:
    name: str
    python: str
    launcher: str
    record: Optional[str]          # build_record.json, or None
    config_log: Optional[str]      # config.log, or None
    unrecorded: bool = False
    #: The MPI implementation of THIS build's launcher (per build: a native build may
    #: bring its own mpirun); ``srun`` under the srun launcher.
    mpi_impl: str = "unknown"


@dataclasses.dataclass
class Config:
    values: Dict[str, Any]
    sources: Dict[str, str]
    platform: str
    stamp: str
    out: str
    run_root: str
    tree_src: str
    tree_layout: str
    tree: str
    api: str
    harness: str
    drv: str
    drivers_mode: str
    python: str
    launcher_kind: str
    launcher: str
    mpi_impl: str
    builds: Dict[str, MeepBuild]
    gpu: Optional[str]
    routes: List[str]
    cases: List[str]
    sizes: Dict[str, List[int]]         # tier -> cells
    physical: int
    logical: int
    smt: bool
    sockets: Optional[int]
    threads_per_core: int
    #: The detected topology (``core_of_cpu``, ``cpus``) when it was read on this host.
    topology: Dict[str, Any]
    #: This process's affinity is narrower than the host's online CPUs.
    restricted: bool
    blocks: Tuple[List[int], List[int], List[int]]
    probe_ranks: List[int]
    preflight_ranks: List[int]
    controls: List[str]
    controls_omitted: Dict[str, str]
    explicit_flags: Dict[str, str]
    expect_launched: Dict[Tuple[str, str], str]
    gate: timing_host.GateConfig
    tiers: List[str]
    passes: int
    binding: str
    array_control: str
    gpu_flags: str
    subnormal_policy: Optional[str]
    allow_uncertified: int
    account: str
    uid: Optional[int]
    extra_pythonpath: List[str]
    snapshot_exclude: List[str]
    libcuda_stub: Optional[str]
    corpus_root: Optional[str]
    base_env: Dict[str, str]
    memory_budget: Optional[int] = None

    def res(self, case: str, cells: int) -> int:
        return timing_cases.res_for_cells(case, cells)

    def table(self, route: str) -> str:
        return ROUTE_TABLES[route]

    def flag_values(self) -> Dict[str, Any]:
        """What :func:`timing_host.binding_flags` needs beyond the mode."""
        return {"sockets": self.sockets, "threads_per_core": self.threads_per_core}

    @property
    def metal_only(self) -> bool:
        return bool(self.routes) and all(route == "metal" for route in self.routes)


def default_rank_blocks(physical: int, logical: int,
                        smt: bool) -> Tuple[List[int], List[int], List[int]]:
    """A = [P/3, P/2, 2P/3, P] rounded; B = [8 when 8 < P/3, then 1]; C = [4P/3] on an
    SMT host, [logical] when the host has more logical CPUs than P without SMT (Apple
    E-cores). P = 48 with 96 logical: 16, 24, 32, 48 | 8, 1 | 64 (the 2026-09-28 lists)."""
    p = int(physical)
    a: List[int] = []
    for value in (p / 3.0, p / 2.0, 2.0 * p / 3.0, float(p)):
        n = max(1, int(round(value)))
        if n not in a:
            a.append(n)
    b = ([8] if 8 < p / 3.0 else []) + [1]
    b = [n for n in b if n not in a]
    c: List[int] = []
    if logical > p and smt:
        c = [int(round(4.0 * p / 3.0))]
    elif logical > p:
        c = [int(logical)]
    return a, b, c


def parse_rank_blocks(text: str) -> Tuple[List[int], List[int], List[int]]:
    parts = str(text).split("|")
    if len(parts) != 3:
        raise refuse(f"--ranks {text!r}: three blocks A|B|C are needed (a block may be "
                     "empty)")
    blocks = []
    for part in parts:
        try:
            blocks.append([int(v) for v in _split_list(part)])
        except ValueError:
            raise refuse(f"--ranks {text!r}: rank counts are integers")
    return blocks[0], blocks[1], blocks[2]


def parse_expect_launched(entries: Sequence[str]) -> Dict[Tuple[str, str], str]:
    """``ROUTE:CASE=TABLES`` -> {(route, case): tables}; refused by name when malformed."""
    out: Dict[Tuple[str, str], str] = {}
    for entry in _split_list(entries):
        match = re.fullmatch(r"([a-z]+):([A-Za-z0-9_]+)=([a-z+]+)", entry)
        if not match:
            raise refuse(f"--expect-launched {entry!r}: the form is ROUTE:CASE=TABLES, "
                         "e.g. cuda:pml_3d=cuda or default:pml_3d=triton")
        route, case, tables = match.groups()
        if route not in ROUTE_TABLES:
            raise refuse(f"--expect-launched {entry!r}: no route {route!r}")
        out[(route, case)] = tables
    return out


def parse_builds(entries: Sequence[str]) -> List[Tuple[str, str, Optional[str], Optional[str]]]:
    out = []
    for entry in entries or []:
        if not entry:
            continue
        name, sep, rest = str(entry).partition("=")
        if not sep or not name or not rest:
            raise refuse(f"--meep-build {entry!r}: the form is NAME=PYTHON[,LAUNCHER[,RECORD]]")
        parts = rest.split(",")
        out.append((name.strip(), parts[0].strip(),
                    parts[1].strip() if len(parts) > 1 and parts[1].strip() else None,
                    parts[2].strip() if len(parts) > 2 and parts[2].strip() else None))
    return out


def repository_root() -> str:
    root = timing_cases.repository_root_of(HERE)
    return root or os.path.dirname(os.path.dirname(HERE))


def _inside(path: str, root: str) -> bool:
    path = os.path.abspath(path).rstrip(os.sep) + os.sep
    root = os.path.abspath(root).rstrip(os.sep) + os.sep
    return path.startswith(root)


def build_config(args: argparse.Namespace, environ: Optional[Dict[str, str]] = None,
                 live: bool = False) -> Config:
    """Resolve every parameter, and refuse by name what the plan cannot be built from.

    Refusals that need only the arguments fire here, in a dry run too. Refusals that
    need the filesystem or a device (a missing stub, an existing run directory, nice)
    belong to :func:`live_checks`.
    """
    environ = dict(os.environ if environ is None else environ)
    values, sources = resolve_params(args, environ)

    def detected(name: str, value: Any) -> Any:
        if values.get(name) is None and value is not None:
            values[name] = value
            sources[name] = "detected"
        return values.get(name)

    def derived(name: str, value: Any) -> Any:
        if values.get(name) is None:
            values[name] = value
            sources[name] = "derived"
        return values.get(name)

    stamp = values["STAMP"]
    if not stamp:
        raise refuse("--stamp is required: it names the run directory and every row id")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(stamp)):
        raise refuse(f"--stamp {stamp!r}: letters, digits, '.', '_' and '-' only")
    platform = detected("PLATFORM", detect_platform())
    python = values["PYTHON"]
    if not python:
        raise refuse("--python is required (or MGPU_SITE_PYTHON): the GPU rows, the "
                     "digests and the reference MEEP run in it")

    # ---- the tree and the drivers -------------------------------------------------
    this_repo = repository_root()
    tree_src = os.path.abspath(derived("TREE_SRC", this_repo))
    layout = values["TREE_LAYOUT"]
    if layout == "auto":
        found = timing_records.layout_of(tree_src) if os.path.isdir(tree_src) else None
        if found is None:
            raise refuse(f"the layout of {tree_src} cannot be read (it holds neither "
                         "meep_gpu/ + parity/meep_gpu/ nor apps/api/...); pass "
                         "--tree-layout release|development")
        layout = found
        values["TREE_LAYOUT"] = found
        sources["TREE_LAYOUT"] = "detected"
    site_root = environ.get("MGPU_SITE_STAGING_ROOT") or os.path.join(
        os.path.expanduser("~"), "meep_gpu_validation")
    if values["RUN_ROOT"] is None:
        values["RUN_ROOT"] = os.path.join(site_root, "timing")
        sources["RUN_ROOT"] = ("site:MGPU_SITE_STAGING_ROOT"
                               if environ.get("MGPU_SITE_STAGING_ROOT") else "default")
    run_root = os.path.abspath(os.path.expanduser(values["RUN_ROOT"]))
    out = os.path.join(run_root, f"ladder_{stamp}")
    for root, what in ((tree_src, "the tree under test"), (this_repo, "this repository"),
                       (HERE, "the drivers' directory")):
        if _inside(out, root):
            raise refuse(f"the run directory {out} is inside {what} ({root}); every row, "
                         "cache and log of a ladder lives outside the trees it reads")
    tree = os.path.join(out, "tree") if values["TREE_MODE"] == "snapshot" else tree_src
    api = tree if layout == "release" else os.path.join(tree, "apps", "api")
    harness = os.path.join(api, "parity", "meep_gpu")
    drivers_mode = derived("DRIVERS_MODE",
                           "tree" if os.path.abspath(tree_src) == this_repo else "copy")
    drv = harness if drivers_mode == "tree" else os.path.join(out, "drivers")
    if drivers_mode == "tree" and layout == "development":
        raise refuse("--drivers-mode tree with a development-layout tree: that tree does "
                     "not carry these drivers; use --drivers-mode copy")

    # ---- topology and ranks ----------------------------------------------------------
    # Read whenever the platform asked is this host's (files and ctypes, no process):
    # explicit counts still win, and the socket count and core map come from here.
    topology = detect_topology(platform)
    physical = _int(detected("PHYSICAL_CORES", topology.get("physical_cores")) or 0,
                    "PHYSICAL_CORES")
    logical = _int(detected("LOGICAL_CPUS", topology.get("logical_cpus")) or 0,
                   "LOGICAL_CPUS")
    if physical <= 0 or logical <= 0:
        raise refuse("the physical cores and logical CPUs in scope could not be read; pass "
                     "--physical-cores and --logical-cpus")
    if logical < physical:
        raise refuse(f"--logical-cpus {logical} is below --physical-cores {physical}")
    smt = platform != "darwin" and logical > physical
    sockets_value = detected("SOCKETS", topology.get("sockets"))
    sockets = _int(sockets_value, "SOCKETS") if sockets_value is not None else None
    per_core = detected("THREADS_PER_CORE", topology.get("threads_per_core"))
    threads_per_core = (_int(per_core, "THREADS_PER_CORE") if per_core is not None
                        else max(1, logical // physical) if platform != "darwin" else 1)
    if per_core is None:
        values["THREADS_PER_CORE"] = threads_per_core
        sources["THREADS_PER_CORE"] = "derived"
    flag_values = {"sockets": sockets, "threads_per_core": threads_per_core}
    if values["RANKS"]:
        blocks = parse_rank_blocks(values["RANKS"])
    else:
        blocks = default_rank_blocks(physical, logical, smt)
        values["RANKS"] = "|".join(",".join(str(n) for n in b) for b in blocks)
        sources["RANKS"] = "derived"
    probe = [int(v) for v in _split_list(values["PROBE_RANKS"])] or \
        [max(1, int(round(2.0 * physical / 3.0))), physical]
    if values["PROBE_RANKS"] is None:
        values["PROBE_RANKS"] = ",".join(str(n) for n in probe)
        sources["PROBE_RANKS"] = "derived"
    preflight_ranks = [1, physical] + ([max(blocks[2])] if blocks[2] else [])
    preflight_ranks = list(dict.fromkeys(preflight_ranks))
    every_rank = sorted(set(blocks[0] + blocks[1] + blocks[2] + probe))
    if every_rank and max(every_rank) > logical:
        raise refuse(f"{max(every_rank)} MEEP ranks exceed the {logical} logical CPUs in "
                     "scope")

    # ---- the launcher and its flags --------------------------------------------------
    largest_ranks = max(every_rank) if every_rank else 1
    kind = values["MEEP_LAUNCHER"]
    launcher = values["MPIRUN"] or (
        "srun" if kind == "srun" else os.path.join(os.path.dirname(python), "mpirun"))
    if values["MPIRUN"] is None:
        values["MPIRUN"] = launcher
        sources["MPIRUN"] = "derived"
    if kind == "srun" and not values["SRUN_MPI"]:
        raise refuse("--meep-launcher srun needs --srun-mpi (pmix, pmi2, ...): srun must "
                     "be told how to wire the ranks' MPI")
    mpi_impl = detected("MPI_IMPL", detect_mpi_impl_from_files(launcher, kind))
    if platform == "darwin" and values["BINDING"] != "unbound":
        if sources["BINDING"] != "default":
            raise refuse(f"--binding {values['BINDING']} on macOS: Open MPI cannot bind "
                         "ranks to cores there; only --binding unbound runs")
        values["BINDING"] = "unbound"
        sources["BINDING"] = "derived"
    explicit = {mode: values[name] for mode, name in (
        ("bound", "FLAGS_BOUND"), ("unbound", "FLAGS_UNBOUND"), ("over", "FLAGS_OVER"),
        ("bound-hwthread", "FLAGS_BOUND_HWTHREAD"), ("bound-pe", "FLAGS_BOUND_PE"))
        if values[name] is not None}
    modes_needed = {"bound", "unbound"} if platform != "darwin" else {"unbound"}
    if blocks[2]:
        modes_needed.add("over")
    for mode in sorted(modes_needed):
        timing_host.binding_flags(mpi_impl if kind != "srun" else "srun", mode, 1, platform,
                                  explicit, ranks=largest_ranks, **flag_values)

    # ---- MEEP builds -----------------------------------------------------------------
    config_log = values["MEEP_CONFIG_LOG"]
    record = values["MEEP_BUILD_RECORD"]
    if config_log is None and record is None:
        raise refuse("no build record for the reference MEEP: pass --meep-config-log "
                     "<config.log>, --meep-build-record <build_record.json>, or "
                     "--meep-config-log none (the column is then labelled 'build flags "
                     "unrecorded')")
    builds = {"reference": MeepBuild(
        "reference", python, launcher,
        None if record in (None, "none") else record,
        None if config_log in (None, "none") else config_log,
        unrecorded=(config_log in (None, "none") and record in (None, "none")),
        mpi_impl=mpi_impl if kind != "srun" else "srun")}
    for name, build_python, build_launcher, build_record in parse_builds(
            values["MEEP_BUILD"]):
        if name in builds:
            raise refuse(f"--meep-build {name}: the name is already used")
        # UNDER srun every build starts through the same srun (a build's directory
        # holds no srun); under mpirun a build brings the mpirun beside its interpreter.
        build_launcher = build_launcher or (
            launcher if kind == "srun" else os.path.join(os.path.dirname(build_python),
                                                         "mpirun"))
        build_impl = "srun" if kind == "srun" else detect_mpi_impl_from_files(
            build_launcher, kind)
        if kind != "srun" and build_impl == "unknown" and mpi_impl != "unknown" and \
                sources["MPI_IMPL"] != "detected":
            build_impl = mpi_impl
        builds[name] = MeepBuild(
            name, build_python, build_launcher, build_record, None,
            unrecorded=build_record is None, mpi_impl=build_impl)
        for mode in sorted(modes_needed):
            timing_host.binding_flags(build_impl, mode, 1, platform, explicit,
                                      ranks=largest_ranks, **flag_values)

    # ---- routes, cases, sizes ----------------------------------------------------------
    routes = _split_list(values["ROUTES"]) if values["ROUTES"] is not None else (
        ["metal"] if platform == "darwin" else ["default", "cuda"])
    if values["ROUTES"] is None:
        values["ROUTES"] = ",".join(routes)
        sources["ROUTES"] = "derived"
    if routes == ["none"]:
        routes = []
    for route in routes:
        if route not in ROUTE_TABLES:
            raise refuse(f"--routes: no route {route!r} (default, cuda, metal, none)")
        if route == "metal" and platform != "darwin":
            raise refuse("the metal route runs on an Apple GPU host only")
        if route in NVIDIA_ROUTES and platform == "darwin":
            raise refuse(f"the {route} route drives an NVIDIA table; this host is macOS")
    gpu = values["GPU"]
    nvidia = [r for r in routes if r in NVIDIA_ROUTES]
    refused_gpus = _split_list(values["REFUSE_GPUS"])
    if nvidia and gpu is None:
        raise refuse("--gpu is required when an NVIDIA route is planned: the device index "
                     "every GPU row pins with CUDA_VISIBLE_DEVICES")
    if gpu is not None and str(gpu) in refused_gpus:
        raise refuse(f"device {gpu} is in --refuse-gpus ({', '.join(refused_gpus)})")
    if gpu is not None and gpu != "auto" and not re.fullmatch(r"\d+|GPU-[0-9a-fA-F-]+",
                                                              str(gpu)):
        raise refuse(f"--gpu {gpu!r}: a device index, a GPU UUID, or auto")
    cases = _split_list(values["CASES"])
    if not cases:
        raise refuse("--cases names no case")
    for case in cases:
        timing_cases.declared(case)
    first = cases[0]

    def res_list(name: str) -> Optional[List[int]]:
        if values[name] is None:
            return None
        return [timing_cases.cells(first, _int(v, name)) for v in _split_list(values[name])]

    sizes: Dict[str, List[int]] = {}
    priority = res_list("PRIORITY_RES")
    if priority is None and values["SIZES"] is not None:
        priority = [_int(v, "SIZES") for v in _split_list(values["SIZES"])]
    if priority is None:
        priority = [_int(v, "PRIORITY_SIZES") for v in _split_list(values["PRIORITY_SIZES"])]
    sizes["priority"] = priority
    sizes["rest"] = res_list("REST_RES") or [
        _int(v, "REST_SIZES") for v in _split_list(values["REST_SIZES"])]
    sizes["largest"] = res_list("LARGEST_RES") or [
        _int(v, "LARGEST_SIZES") for v in _split_list(values["LARGEST_SIZES"])]
    sizes["probe"] = ([timing_cases.cells(first, _int(values["PROBE_RES"], "PROBE_RES"))]
                      if values["PROBE_RES"] is not None
                      else [_int(values["PROBE_SIZE"], "PROBE_SIZE")])
    sizes["preflight"] = ([timing_cases.cells(first, _int(values["PREFLIGHT_RES"],
                                                          "PREFLIGHT_RES"))]
                          if values["PREFLIGHT_RES"] is not None
                          else [_int(values["PREFLIGHT_SIZE"], "PREFLIGHT_SIZE")])
    tiers = _split_list(values["TIERS"])
    for tier in tiers:
        if tier not in TIERS:
            raise refuse(f"--tiers: no tier {tier!r} ({', '.join(TIERS)})")
    used = {"preflight": sizes["preflight"], "bindprobe": sizes["probe"],
            "priority": sizes["priority"], "bindcheck": sizes["priority"],
            "rest": sizes["rest"], "largest": sizes["largest"]}
    for tier, tier_cells in used.items():
        if tier not in tiers:
            continue
        for cells in tier_cells:
            for case in (cases if tier not in ("bindprobe",) else [first]):
                timing_cases.res_for_cells(case, cells)

    # ---- controls ------------------------------------------------------------------
    controls = [c for c in _split_list(values["CONTROLS"]) if c != "none"]
    omitted: Dict[str, str] = {}
    for control in controls:
        if control not in CONTROLS:
            raise refuse(f"--controls: no control {control!r} ({', '.join(CONTROLS)})")
    kept: List[str] = []
    for control in controls:
        why = None
        if control == "native" and "native" not in builds:
            why = "no --meep-build native=PYTHON[,LAUNCHER[,RECORD]] was given"
        elif control in ("threads2", "threads4"):
            threads = 2 if control == "threads2" else 4
            ranks = max(1, int(round(physical / float(threads))))
            if platform == "darwin":
                why = (f"{threads} OpenMP threads per rank: the reference macOS MEEP "
                       "builds link no OpenMP runtime of their own (timing_records.py "
                       "meep-build-record reports it), so the bench would refuse the "
                       "rows; and Open MPI cannot bind ranks on macOS")
            elif ranks * threads > physical:
                raise refuse(f"control {control}: {ranks} ranks x {threads} threads exceed "
                             f"the {physical} physical cores in scope")
        elif control == "hwthread" and not smt:
            why = "the host has no SMT: there are no hardware threads above the cores"
        elif control == "hwthread" and platform == "darwin":
            why = "bound-hwthread is a bound mode, and Open MPI cannot bind on macOS"
        if why:
            omitted[control] = why
        else:
            kept.append(control)
    for control in kept:
        mode = {"threads2": "bound-pe", "threads4": "bound-pe",
                "hwthread": "bound-hwthread"}.get(control)
        if mode:
            timing_host.binding_flags(mpi_impl if kind != "srun" else "srun", mode, 2,
                                      platform, explicit, ranks=largest_ranks,
                                      **flag_values)

    # ---- gates ---------------------------------------------------------------------
    memory_total = values["MEMORY_TOTAL_GB"]
    if memory_total is None:
        memory_total = detect_memory_total(platform)
        if memory_total is not None:
            values["MEMORY_TOTAL_GB"] = memory_total
            sources["MEMORY_TOTAL_GB"] = "detected"
    cpu_max = values["GATE_CPU_MAX"]
    if cpu_max is None:
        cpu_max = timing_host.default_cpu_max(logical)
        values["GATE_CPU_MAX"] = cpu_max
        sources["GATE_CPU_MAX"] = "derived"
    mem_min = values["GATE_MEM_MIN_GB"]
    if mem_min is None:
        if memory_total is None:
            raise refuse("total memory could not be read; pass --gate-mem-min-gb")
        mem_min = timing_host.default_mem_min_gib(_float(memory_total, "MEMORY_TOTAL_GB"))
        values["GATE_MEM_MIN_GB"] = mem_min
        sources["GATE_MEM_MIN_GB"] = "derived"
    account = derived("ACCOUNT", getpass.getuser())
    uid = timing_host.account_uid(account) if platform == detect_platform() else None
    cap = values["CAP"]
    scope = values["GATE_SCOPE"]
    if platform == "darwin" and scope != "host":
        raise refuse("--gate-scope allocation on macOS: the Mach CPU ticks are host-wide")
    if values["GATE_GPU_METRIC"] is None:
        # A HOST-WIDE LOAD AVERAGE SEES EVERY JOB ON A SHARED NODE, so with an
        # allocation scope the GPU rows gate on the allocation's busy CPUs; on a Mac
        # the load average runs 2.5-2.8 times the busy CPUs and the per-CPU Mach tick
        # reading is the calibrated one.
        values["GATE_GPU_METRIC"] = ("busy" if platform == "darwin" or scope == "allocation"
                                     else "load1")
        sources["GATE_GPU_METRIC"] = "derived"
    elif values["GATE_GPU_METRIC"] == "load1" and scope == "allocation":
        raise refuse("--gate-gpu-metric load1 with --gate-scope allocation: the load "
                     "average is host-wide, so a GPU row would wait on other jobs; use "
                     "--gate-gpu-metric busy")
    if values["GATE_MEEP_METRIC"] == "load1" and scope == "allocation":
        raise refuse("--gate-meep-metric load1 with --gate-scope allocation: the load "
                     "average is host-wide; use busy")
    apple_util = values["GATE_APPLE_GPU_UTIL_MAX"]
    gate = timing_host.GateConfig(
        platform=platform, gpu=None if gpu in (None, "auto") else str(gpu), uid=uid,
        cpu_max=_float(cpu_max, "GATE_CPU_MAX"),
        mem_min_gib=_float(mem_min, "GATE_MEM_MIN_GB"),
        account_max=_int(values["ACCOUNT_MAX"], "ACCOUNT_MAX"),
        cap=None if cap in (None, "", "off") else _int(cap, "CAP"),
        gpu_metric=values["GATE_GPU_METRIC"], meep_metric=values["GATE_MEEP_METRIC"],
        hold_gpu_s=_float(values["GATE_HOLD_GPU_S"], "GATE_HOLD_GPU_S"),
        hold_meep_s=_float(values["GATE_HOLD_MEEP_S"], "GATE_HOLD_MEEP_S"),
        retry_s=_float(values["GATE_RETRY_S"], "GATE_RETRY_S"),
        give_up_s=_float(values["GATE_GIVE_UP_S"], "GATE_GIVE_UP_S"),
        scope=scope, check_device=bool(routes),
        require_ac_power=bool(int(values["REQUIRE_AC_POWER"])),
        check_daemons=bool(int(values["GATE_DARWIN_DAEMONS"])),
        apple_gpu_util_max=(None if apple_util in (None, "", "off")
                            else _float(apple_util, "GATE_APPLE_GPU_UTIL_MAX")))

    # ---- the GPU rows' instrument -----------------------------------------------------
    memory_budget = None
    if values["GPU_FLAGS"] is None:
        if platform == "darwin":
            if memory_total is None:
                raise refuse("total memory could not be read to size the GPU rows' memory "
                             "budget; pass --gpu-flags or --memory-total-gb")
            memory_budget = darwin_memory_budget(_float(memory_total, "MEMORY_TOTAL_GB"))
            values["GPU_FLAGS"] = darwin_gpu_flags(memory_budget)
        else:
            values["GPU_FLAGS"] = LINUX_GPU_FLAGS
        sources["GPU_FLAGS"] = "derived"
    if routes:
        flags = shlex.split(values["GPU_FLAGS"])
        attached = ("--monitors" in flags and flags.index("--monitors") + 1 < len(flags)
                    and flags[flags.index("--monitors") + 1] == "attached") or \
            "--monitors=attached" in flags
        if not attached:
            raise refuse(f"--gpu-flags {values['GPU_FLAGS']!r} does not time with "
                         "--monitors attached: every MEEP row accumulates its flux monitor, "
                         "so a GPU row timed without one is a different workload")
    if values["ARRAY_CONTROL"] == "row" and "metal" in routes:
        raise refuse("--array-control row with the metal route: gpu_array_control.py "
                     "drives the NVIDIA tables only; on macOS the control is read off "
                     "each GPU row's own array leg (--array-control extract)")

    subnormal = derived("SUBNORMAL_POLICY", "keep" if platform != "darwin" else "default")
    corpus = values["CORPUS_ROOT"]
    if corpus is None and environ.get("MGPU_SITE_MEEP_SOURCE"):
        candidate = os.path.join(environ["MGPU_SITE_MEEP_SOURCE"], "python")
        if os.path.isdir(candidate):
            corpus = candidate
            values["CORPUS_ROOT"] = corpus
            sources["CORPUS_ROOT"] = "site:MGPU_SITE_MEEP_SOURCE"

    base_env = process_environment(environ)
    config = Config(
        values=values, sources=sources, platform=platform, stamp=stamp, out=out,
        run_root=run_root, tree_src=tree_src, tree_layout=layout, tree=tree, api=api,
        harness=harness, drv=drv, drivers_mode=drivers_mode, python=python,
        launcher_kind=kind, launcher=launcher, mpi_impl=mpi_impl, builds=builds,
        gpu=None if gpu is None else str(gpu), routes=routes, cases=cases, sizes=sizes,
        physical=physical, logical=logical, smt=smt, blocks=blocks, probe_ranks=probe,
        preflight_ranks=preflight_ranks, controls=kept, controls_omitted=omitted,
        explicit_flags=explicit, expect_launched=parse_expect_launched(
            values["EXPECT_LAUNCHED"]),
        gate=gate, tiers=tiers, passes=_int(values["PASSES"], "PASSES"),
        binding=values["BINDING"], array_control=values["ARRAY_CONTROL"],
        gpu_flags=values["GPU_FLAGS"], subnormal_policy=(None if subnormal == "default"
                                                        else subnormal),
        allow_uncertified=_int(values["ALLOW_UNCERTIFIED"], "ALLOW_UNCERTIFIED"),
        account=account, uid=uid, extra_pythonpath=_split_list(values["EXTRA_PYTHONPATH"]),
        snapshot_exclude=_split_list(values["SNAPSHOT_EXCLUDE"]),
        libcuda_stub=values["LIBCUDA_STUB"], corpus_root=corpus, base_env=base_env,
        sockets=sockets, threads_per_core=threads_per_core,
        topology=topology if topology.get("method") else {},
        restricted=bool(topology.get("restricted_by_affinity")),
        memory_budget=memory_budget)
    return config


def darwin_memory_budget(total_gib: float) -> int:
    """Bytes the frozen legs of one Metal GPU row may take: 0.6 of the host's memory."""
    return int(DARWIN_MEMORY_BUDGET_SHARE * float(total_gib) * timing_host.GIB)


def darwin_gpu_flags(budget: int) -> str:
    return ("--repeats 6 --target-seconds 2.0 --monitors attached "
            f"--memory-budget-bytes {int(budget)}")


# ---------------------------------------------------------------------------
# Environments
# ---------------------------------------------------------------------------

#: What the ladder exports for itself and every row. ``KMP_DUPLICATE_LIB_OK`` is not
#: among them: on Linux it is inherited from the launching shell and recorded; on macOS
#: the GPU rows set it and the MEEP rows unset it (:data:`DARWIN_GPU_EXPORTS`).
PROCESS_EXPORTS = {"MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1",
                   "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1",
                   "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                   "NUMEXPR_NUM_THREADS": "1"}


def process_environment(environ: Dict[str, str]) -> Dict[str, str]:
    base = dict(environ)
    base.update(PROCESS_EXPORTS)
    return base


def apply_process_environment() -> None:
    """The ladder's own environment, applied first in every mode that reads the host,
    so ``--selftest`` reads under the environment the live run reads under."""
    os.environ.update(PROCESS_EXPORTS)


@dataclasses.dataclass
class EnvDelta:
    set: Dict[str, str]
    unset: List[str]

    def apply(self, base: Dict[str, str]) -> Dict[str, str]:
        env = dict(base)
        for name in self.unset:
            env.pop(name, None)
        env.update(self.set)
        return env

    def as_dict(self) -> Dict[str, Any]:
        return {"set": dict(self.set), "unset": sorted(self.unset)}


#: macOS, GPU rows only. The pip ``torch`` wheel bundles its own ``libomp`` and the
#: environment's OpenBLAS links conda's, so ``import torch`` beside NumPy aborts
#: ("OMP: Error #15") unless two OpenMP runtimes are allowed; every Metal row imports
#: torch. The reference Mac runs set the same (``build_meep_133_macos.sh``). Set on the
#: GPU rows' environment delta, so every row records it; the MEEP rows, which load one
#: runtime at most, unset it.
DARWIN_GPU_EXPORTS = {"KMP_DUPLICATE_LIB_OK": "TRUE"}


def inherited_switches(base: Dict[str, str], keep: Sequence[str] = ()) -> List[str]:
    """Every ``MEEP_GPU_*`` variable of the launching environment beyond the named
    :data:`PACKAGE_SWITCHES` and ``keep``: a row never runs a package switch the ladder
    did not set (``MEEP_GPU_METAL_RESIDENCY=shipped`` would time another transport)."""
    return sorted(name for name in base if name.startswith("MEEP_GPU_")
                  and name not in PACKAGE_SWITCHES and name not in keep)


def gpu_env(config: Config, route: str, keep_cache: str) -> EnvDelta:
    base_pythonpath = config.base_env.get("PYTHONPATH", "")
    base_ld = config.base_env.get("LD_LIBRARY_PATH", "")
    unset = list(PACKAGE_SWITCHES)
    sets: Dict[str, str] = {}
    pythonpath = [config.api, config.harness] + [
        entry.replace("{harness}", config.harness).replace("{api}", config.api)
        for entry in config.extra_pythonpath]
    if base_pythonpath:
        pythonpath.append(base_pythonpath)
    sets["PYTHONPATH"] = ":".join(pythonpath)
    # AN NVIDIA ROW TIMES A CERTIFIED DEVICE AND TOOLCHAIN OR NOTHING. The package
    # runs a supported, uncertified NVIDIA identity by default; ``0`` keeps such a row
    # on the refusal it got before that default, so it cannot enter a figure as
    # though certified. A Metal row keeps the Metal default (unset), as before.
    if config.allow_uncertified:
        sets["MEEP_GPU_ALLOW_UNCERTIFIED"] = "1"
    elif route in NVIDIA_ROUTES:
        sets["MEEP_GPU_ALLOW_UNCERTIFIED"] = "0"
    else:
        unset.append("MEEP_GPU_ALLOW_UNCERTIFIED")
    if route in NVIDIA_ROUTES:
        sets["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
        sets["CUPY_ACCELERATORS"] = ""
        sets["TRITON_CACHE_DIR"] = os.path.join(config.out, "cache", "triton")
        sets["CUPY_CACHE_DIR"] = keep_cache
        if config.libcuda_stub:
            sets["TRITON_LIBCUDA_PATH"] = config.libcuda_stub
            sets["LD_LIBRARY_PATH"] = ":".join(p for p in (config.libcuda_stub, base_ld) if p)
        if config.corpus_root:
            sets["MEEP_GPU_CORPUS_ROOT"] = config.corpus_root
    elif config.corpus_root:
        sets["MEEP_GPU_CORPUS_ROOT"] = config.corpus_root
    if config.subnormal_policy:
        sets["MEEP_GPU_SUBNORMAL_POLICY"] = config.subnormal_policy
    if route == "cuda":
        sets["MEEP_GPU_BACKEND_PREFERENCE"] = "cuda"
        unset.remove("MEEP_GPU_BACKEND_PREFERENCE")
    unset += [name for name in inherited_switches(config.base_env, list(sets))
              if name not in unset]
    if route == "metal":
        unset += [name for name in sorted(config.base_env)
                  if (name in PYTORCH_MPS_SWITCHES or name.startswith(PYTORCH_MPS_PREFIX))
                  and name not in unset]
    if config.platform == "darwin":
        sets.update(DARWIN_GPU_EXPORTS)
    return EnvDelta(sets, unset)


def meep_env(config: Config, threads: int = 1) -> EnvDelta:
    sets = {"CUDA_VISIBLE_DEVICES": "",
            "PYTHONPATH": config.base_env.get("PYTHONPATH", ""),
            "LD_LIBRARY_PATH": config.base_env.get("LD_LIBRARY_PATH", ""),
            "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1"}
    if threads > 1:
        sets.update({"OMP_PROC_BIND": "close", "OMP_PLACES": "cores"})
    unset = list(PACKAGE_SWITCHES) + ["TRITON_LIBCUDA_PATH", "MEEP_GPU_SUBNORMAL_POLICY",
                                      "MEEP_GPU_CORPUS_ROOT", "CUPY_ACCELERATORS",
                                      "MEEP_GPU_ALLOW_UNCERTIFIED", "TRITON_CACHE_DIR",
                                      "CUPY_CACHE_DIR"]
    if threads == 1:
        unset += ["OMP_PROC_BIND", "OMP_PLACES"]
    unset += [name for name in inherited_switches(config.base_env) if name not in unset]
    if config.platform == "darwin":
        unset += [name for name in DARWIN_GPU_EXPORTS if name not in unset]
    return EnvDelta(sets, unset)


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Row:
    tier: str
    kind: str                        # gpu | meep | control | digest | lift | decide
    case: Optional[str] = None
    cells: Optional[int] = None
    res: Optional[int] = None
    pass_index: int = 0
    route: Optional[str] = None
    ranks: Optional[int] = None
    binding: Optional[str] = None    # asked: bound | unbound | chosen | other | mode
    build: str = "reference"
    threads: int = 1
    split: Optional[bool] = None
    control: Optional[str] = None
    index: int = 0
    of: int = 0

    def configuration_label(self) -> str:
        parts = []
        if self.build != "reference":
            parts.append(self.build)
        if self.threads > 1:
            parts.append(f"t{self.threads}")
        if self.binding == "bound-hwthread":
            parts.append("bound-hwthread")
        if self.split is False:
            parts.append("splitcost")
        elif self.split is True:
            parts.append("spliteven")
        return "-".join(parts)


def build_plan(config: Config) -> List[Row]:
    plan: List[Row] = []
    a, b, c = config.blocks

    def sequence(tier: str, case: str, cells: int, pass_index: int) -> None:
        res = config.res(case, cells)

        def meep(ranks: int, **extra: Any) -> None:
            plan.append(Row(tier, "meep", case, cells, res, pass_index, ranks=ranks,
                            binding=extra.pop("binding", "chosen"), **extra))

        routes = list(config.routes)
        if routes:
            plan.append(Row(tier, "gpu", case, cells, res, pass_index, route=routes[0]))
        for n in a:
            meep(n)
        for route in routes[1:]:
            plan.append(Row(tier, "gpu", case, cells, res, pass_index, route=route))
        for n in b:
            meep(n)
        if routes:
            plan.append(Row(tier, "control", case, cells, res, pass_index))
        for n in c:
            meep(n)
        p = config.physical
        for control in config.controls:
            if control == "native":
                for n in dict.fromkeys([max(1, int(round(2 * p / 3.0))), p]):
                    meep(n, build="native", control=control)
            elif control in ("threads2", "threads4"):
                threads = 2 if control == "threads2" else 4
                meep(max(1, int(round(p / float(threads)))), threads=threads,
                     control=control, binding="bound")
            elif control == "hwthread":
                for n in dict.fromkeys([int(round(4 * p / 3.0)), 2 * p]):
                    if n <= config.logical:
                        meep(n, binding="bound-hwthread", control=control)
            elif control == "split":
                for n in dict.fromkeys([max(1, int(round(2 * p / 3.0))), p]):
                    meep(n, split=False, control=control)

    first = config.cases[0]
    for tier in config.tiers:
        if tier == "preflight":
            cells = config.sizes["preflight"][0]
            for case in config.cases:
                res = config.res(case, cells)
                for route in config.routes:
                    plan.append(Row("preflight", "gpu", case, cells, res, 0, route=route))
                if config.routes and config.array_control == "row":
                    plan.append(Row("preflight", "control", case, cells, res, 0))
                for n in config.preflight_ranks:
                    plan.append(Row("preflight", "meep", case, cells, res, 0, ranks=n,
                                    binding="bound"))
                plan.append(Row("preflight", "digest", case, cells, res, 0))
        elif tier == "digests":
            seen = set()
            for name in ("probe", "priority", "rest", "largest"):
                if {"probe": "bindprobe", "priority": "priority", "rest": "rest",
                        "largest": "largest"}[name] not in config.tiers:
                    continue
                for cells in config.sizes[name]:
                    for case in (config.cases if name != "probe" else [first]):
                        if (case, cells) in seen:
                            continue
                        seen.add((case, cells))
                        plan.append(Row("digests", "lift", case, cells,
                                        config.res(case, cells), 0))
        elif tier == "bindprobe":
            if config.binding != "probe" or config.platform == "darwin":
                continue
            cells = config.sizes["probe"][0]
            res = config.res(first, cells)
            for pass_index in (1, 2):
                for label in ("bound", "unbound"):
                    for n in config.probe_ranks:
                        plan.append(Row("bindprobe", "meep", first, cells, res, pass_index,
                                        ranks=n, binding=label))
            plan.append(Row("bindprobe", "decide", first, cells, res, 0))
        elif tier == "bindcheck":
            if config.platform == "darwin" or config.binding == "unbound":
                continue
            for cells in config.sizes["priority"]:
                for case in config.cases:
                    for n in config.probe_ranks:
                        plan.append(Row("bindcheck", "meep", case, cells,
                                        config.res(case, cells), 1, ranks=n,
                                        binding="other"))
        else:
            for pass_index in range(1, config.passes + 1):
                for cells in config.sizes[tier]:
                    for case in config.cases:
                        sequence(tier, case, cells, pass_index)
    for index, row in enumerate(plan, 1):
        row.index = index
        row.of = len(plan)
    return plan


# ---------------------------------------------------------------------------
# Row facts: keys, bindings, parameters, commands, budgets
# ---------------------------------------------------------------------------

def meep_params(row: Row) -> Tuple[int, float, int]:
    """(windows, target seconds, warm-up wall cap) of a MEEP row: the 2026-09-28 rule,
    with "resolution 40 and above" read as 4,096,000 cells and above."""
    if row.tier == "preflight":
        return 3, 0.5, 5
    if row.tier in ("bindprobe", "bindcheck"):
        return 3, 2.0, 20
    if row.ranks == 1 and (row.cells or 0) >= 4_096_000:
        return 3, 3.0, 30
    return 6, 2.0, 20


def gpu_estimate(cells: int) -> int:
    if cells in GPU_SECONDS_BY_CELLS:
        return GPU_SECONDS_BY_CELLS[cells]
    known = sorted(GPU_SECONDS_BY_CELLS.items())
    if cells <= known[0][0]:
        return known[0][1]
    for (c0, s0), (c1, s1) in zip(known, known[1:]):
        if c0 <= cells <= c1:
            return int(round(s0 + (s1 - s0) * (cells - c0) / float(c1 - c0)))
    return int(round(known[-1][1] * cells / float(known[-1][0])))


def meep_estimate(row: Row) -> int:
    windows, target, cap = meep_params(row)
    cells = float(row.cells or 0)
    mcells = cells / 1e6
    n = int(row.ranks or 1)
    rate = MEEP_RATE_BY_RANKS.get(n, 450)
    limit = 60 + 2200 * mcells
    if n > 1 and rate > limit:
        rate = limit
    per_step = cells / (rate * 1e6)
    init = 18 * mcells / n * 1.3 + 1
    warm = min(400 * per_step, cap)
    return int(10 + init + 0.3 * mcells + warm + 1 + windows * target + 3)


def row_budget(config: Config, row: Row) -> int:
    if row.kind == "gpu":
        return 180 if row.tier == "preflight" else gpu_estimate(row.cells or 0)
    if row.kind == "meep":
        return meep_estimate(row)
    if row.kind == "control":
        return (gpu_estimate(row.cells or 0) // 3 + 15) if config.array_control == "row" \
            else 1
    if row.kind == "digest":
        return 15
    if row.kind == "lift":
        return lift_timeout(row.cells or 0) // 4
    return 1


def lift_timeout(cells: int) -> int:
    """max(600 s, 4 x (30 s per Mcell)): the res-48 lift took 214.8 s on the A6000 host."""
    return int(max(600, 4 * 30 * cells / 1e6))


def row_timeout(config: Config, row: Row) -> int:
    budget = row_budget(config, row)
    if row.kind in ("gpu", "control"):
        return budget * 3 + 900
    if row.kind == "meep":
        return budget * 4 + 300
    if row.kind == "digest":
        return 600
    if row.kind == "lift":
        return lift_timeout(row.cells or 0)
    return 60


def gate_kind(config: Config, row: Row) -> Optional[str]:
    if row.kind == "gpu":
        return "gpu"
    if row.kind == "meep":
        return "meep"
    if row.kind == "control" and config.array_control == "row":
        return "gpu"
    return None


def gate_estimate(config: Config, kind: Optional[str]) -> int:
    if kind is None:
        return 0
    hold = config.gate.hold_gpu_s if kind == "gpu" else config.gate.hold_meep_s
    metric = config.gate.gpu_metric if kind == "gpu" else config.gate.meep_metric
    return int(hold + (60 if metric == "load1" else 8))


def binding_mode(config: Config, row: Row, chosen: str) -> str:
    """The binding mode a MEEP row runs under, given the binding chosen so far."""
    asked = row.binding or "chosen"
    if config.platform == "darwin":
        # Never bound on macOS; above the performance cores the ranks run on the
        # efficiency cores too, which the row's label says (``unbound-ecores``).
        return "over" if int(row.ranks or 1) > config.physical else "unbound"
    if asked == "chosen":
        asked = chosen
    elif asked == "other":
        asked = "unbound" if chosen == "bound" else "bound"
    if asked == "bound-hwthread":
        return "bound-hwthread"
    return timing_host.binding_mode_for(int(row.ranks or 1), asked, config.physical,
                                        row.threads)


def row_flags(config: Config, row: Row, chosen: str) -> Tuple[str, str, str]:
    """(mode, flags, label) of a MEEP row, with the flags of ITS build's launcher."""
    mode = binding_mode(config, row, chosen)
    impl = config.builds[row.build].mpi_impl if config.launcher_kind != "srun" else "srun"
    flags, label = timing_host.binding_flags(impl, mode, row.threads, config.platform,
                                             config.explicit_flags,
                                             ranks=int(row.ranks or 1),
                                             **config.flag_values())
    return mode, flags, label


def row_key(config: Config, row: Row, label: Optional[str] = None) -> str:
    tier = row.tier + (f"_{row.case}" if len(config.cases) > 1 and row.case else "")
    if row.kind == "gpu":
        return f"{tier}_gpu_{row.route}_res{row.res}_p{row.pass_index}"
    if row.kind == "meep":
        # The configuration label, less any part the binding label already carries
        # (a bound-hwthread row would otherwise read ..._bound-hwthread_bound-hwthread_...).
        config_label = row.configuration_label()
        if label == "bound-hwthread":
            config_label = "-".join(part for part in config_label.split("bound-hwthread")
                                    if part.strip("-")).strip("-")
        return (f"{tier}_meep_r{row.ranks}_{label}"
                + (f"_{config_label}" if config_label else "")
                + f"_res{row.res}_p{row.pass_index}")
    if row.kind == "control":
        return f"{tier}_control_array_res{row.res}_p{row.pass_index}"
    if row.kind == "digest":
        return f"{tier}_digest"
    if row.kind == "lift":
        return f"{tier}_lift_res{row.res}"
    return f"{tier}_decide"


def gpu_out(config: Config, row: Row, route: Optional[str] = None) -> str:
    route = route or row.route
    if row.tier == "preflight":
        return os.path.join(config.out, "preflight", f"gpu_{route}", f"{row.case}_res{row.res}")
    return os.path.join(config.out, f"gpu_{route}", f"pass{row.pass_index}",
                        f"{row.case}_res{row.res}")


def meep_out(config: Config, row: Row, mode: str) -> str:
    if row.tier == "preflight":
        return os.path.join(config.out, "preflight", "meep")
    if row.tier == "bindprobe":
        return os.path.join(config.out, "bindprobe", row.binding or "bound")
    if row.tier == "bindcheck":
        return os.path.join(config.out, "bindcheck",
                            "unbound" if mode in ("unbound", "over") else "bound")
    label = row.configuration_label()
    return os.path.join(config.out, "meep" + (f"_{label}" if label else ""))


@dataclasses.dataclass
class Command:
    argv: List[str]
    cwd: str
    env: EnvDelta
    log: str
    timeout: int
    out: Optional[str] = None
    label: Optional[str] = None
    mode: Optional[str] = None
    flags: Optional[str] = None
    row_id: Optional[str] = None
    launcher_line: Optional[str] = None
    build: Optional[MeepBuild] = None


def preflight_gpu_flags(config: Config) -> str:
    """The preflight's short windows; on macOS with the host's memory budget."""
    if config.memory_budget is None:
        return PREFLIGHT_GPU_FLAGS
    return re.sub(r"--memory-budget-bytes \d+",
                  f"--memory-budget-bytes {config.memory_budget}", PREFLIGHT_GPU_FLAGS)


def gpu_command(config: Config, row: Row, keep_cache: str, utc_stamp: str) -> Command:
    out = gpu_out(config, row)
    flags = preflight_gpu_flags(config) if row.tier == "preflight" else config.gpu_flags
    table = config.table(row.route or "default")
    common = ["--drive-table", table, "--out", out, "--gpu", "0", "--case", str(row.case),
              "--res", str(row.res)] + shlex.split(flags)
    if timing_cases.is_builtin(str(row.case)):
        argv = [config.python, "-u", "parity/meep_gpu/bench_fused_products.py"] + common
    else:
        argv = [config.python, "-u", os.path.join(config.drv, "bench_timing_case.py"),
                "--harness-root", config.harness] + common
    return Command(argv, config.api, gpu_env(config, row.route or "default", keep_cache),
                   os.path.join(out, f"run_{utc_stamp}.log"), row_timeout(config, row),
                   out=out, flags=flags)


def meep_command(config: Config, row: Row, chosen: str, utc_stamp: str,
                 box_before: str) -> Command:
    mode, flags, label = row_flags(config, row, chosen)
    key = row_key(config, row, label)
    row_id = f"{key}_{config.stamp}"
    windows, target, cap = meep_params(row)
    build = config.builds[row.build]
    outdir = meep_out(config, row, mode)
    log = os.path.join(config.out, "logs", f"{key}.{utc_stamp}.log")
    impl = build.mpi_impl if config.launcher_kind != "srun" else "srun"
    report_flags = timing_host.with_report(impl, flags)
    if config.launcher_kind == "srun":
        launch = [build.launcher, f"--mpi={config.values['SRUN_MPI']}", "-n",
                  str(row.ranks)] + shlex.split(report_flags)
    else:
        launch = [build.launcher, "-np", str(row.ranks)] + shlex.split(report_flags)
    bench = [build.python, "-u", os.path.join(config.drv, "bench_meep_identical_case.py"),
             "--harness-root", config.harness, "--case", str(row.case), "--res", str(row.res)]
    knobs = ["--windows", str(windows), "--target-seconds", str(target), "--warm-steps",
             "400", "--warm-seconds-cap", str(cap), "--stepper", "fields_step"]
    threads = row.threads
    line = " ".join([f"OMP_NUM_THREADS={threads}", "CUDA_VISIBLE_DEVICES="] + launch
                    + bench + knobs)
    argv = launch + bench + [
        "--out", outdir, "--pass-index", str(row.pass_index),
        "--passes-planned", str(config.passes), "--tag", row.tier, "--row-id", row_id,
        "--expect-ranks", str(row.ranks)] + knobs + [
        "--launcher-line", line, "--binding-label", label, "--bindings-report", log]
    config_log = build.config_log if build.name == "reference" else None
    if config_log:
        argv += ["--meep-config-log", config_log]
    argv += ["--box-before", box_before]
    # NEW OPTIONS ONLY WHERE THEY DIFFER FROM THE 2026-09-28 BEHAVIOUR.
    if build.name != "reference":
        argv += ["--build-label", build.name]
    if threads != 1:
        # A THREADED ROW READS WHETHER MEEP LINKS OPENMP FROM THE RECORD THIS LADDER
        # PROBED ITSELF (it carries a given record inside it, under build_record_file).
        argv += ["--build-record", probed_record_path(config, build.name)]
    elif build.record:
        argv += ["--build-record", build.record]
    if threads != 1:
        argv += ["--threads-per-rank", str(threads)]
    if row.split is not None:
        argv += ["--split-chunks-evenly", "true" if row.split else "false"]
    if config.restricted:
        argv += ["--scope-cpus", str(config.logical)]
    if config.metal_only:
        argv += ["--table", "metal"]
    return Command(argv, config.out, meep_env(config, threads), log,
                   row_timeout(config, row), out=outdir, label=label, mode=mode,
                   flags=flags, row_id=row_id, launcher_line=line, build=build)


def probed_record_path(config: Config, build: str) -> str:
    """Where the ladder writes the build record it probed from ``build``'s interpreter."""
    return os.path.join(config.out, "meep_builds", f"{build}.json")


def table_args(config: Config) -> List[str]:
    """``--table metal`` for a macOS ladder (the Metal gate's ``CASES``); nothing else."""
    return ["--table", "metal"] if config.metal_only else []


def digest_command(config: Config, row: Row) -> Command:
    out = os.path.join(config.out, "preflight", "digest")
    compare = [os.path.join(config.out, "preflight", "meep", f"rows_r{n}.jsonl")
               for n in config.preflight_ranks]
    argv = [config.python, "-u", os.path.join(config.drv, "digest_harness_lift.py"),
            "--harness-root", config.harness, "--case", str(row.case), "--res",
            str(row.res), "--out", out] + table_args(config) + ["--compare"] + compare
    log_name = "preflight_digest.log" if len(config.cases) == 1 else \
        f"preflight_digest_{row.case}.log"
    return Command(argv, config.out, meep_env(config), os.path.join(
        config.out, "logs", log_name), 600, out=out)


def lift_command(config: Config, row: Row) -> Command:
    out = os.path.join(config.out, "digests_lift")
    argv = [config.python, "-u", os.path.join(config.drv, "digest_harness_lift.py"),
            "--harness-root", config.harness, "--case", str(row.case), "--res",
            str(row.res), "--out", out, "--digest-file",
            digest_file(config, str(row.case), int(row.res or 0))] + table_args(config)
    return Command(argv, config.out, meep_env(config), os.path.join(
        config.out, "logs", f"{row_key(config, row)}.log"), row_timeout(config, row),
        out=out)


def control_command(config: Config, row: Row, keep_cache: str, utc_stamp: str) -> Command:
    out = (os.path.join(config.out, "preflight", "array_control", f"{row.case}_res{row.res}")
           if row.tier == "preflight" else
           os.path.join(config.out, "array_control", f"pass{row.pass_index}",
                        f"{row.case}_res{row.res}"))
    argv = [config.python, "-u", os.path.join(config.drv, "gpu_array_control.py"),
            "--harness-root", config.harness, "--case", str(row.case), "--res",
            str(row.res), "--out", out, "--gpu", "0", "--repeats", "6",
            "--target-seconds", "2.0", "--forced-by", "dispatch"]
    route = "default" if "default" in config.routes else (config.routes or ["default"])[0]
    return Command(argv, config.api, gpu_env(config, route, keep_cache),
                   os.path.join(out, f"run_{utc_stamp}.log"), row_timeout(config, row),
                   out=out)


def preflight_gpu_verdict(entry: Dict[str, Any], summary: str,
                          expect: Optional[str]) -> Tuple[bool, Optional[str]]:
    """Whether a preflight GPU row lets the ladder go on, and what it launched.

    The 2026-09-28 rule, kept: the harness timed it (``TIMED``, with or without the
    spread floor -- a preflight window is half a second), it launched a kernel table,
    and its fused leg is bit-identical to the array leg. Added: it launched what the
    route expects when an expectation was given (otherwise this row PINS it for every
    later row), the fused leg carries exactly the deposit-repair bracket the case
    declares, it timed with its monitors attached, and on a Mac no power, thermal or
    neighbour condition was recorded around it.
    """
    measured = entry.get("measured") or {}
    launched = measured.get("what_launched")
    ok = (summary.startswith("TIMED") and f"launched={launched} fused " in summary
          and "bit_identical=True" in summary
          and launched not in (None, "nothing (array path)")
          and (expect is None or launched == expect)
          and (entry.get("deposit_repair") or {}).get("ok") is True
          and measured.get("monitors_mode") == "attached"
          and not entry.get("conditions"))
    return ok, launched


def digest_file(config: Config, case: str, res: int) -> str:
    return os.path.join(config.out, "digests", f"{case}_res{res}.sha256")


def expected_launched(config: Config, route: str, case: str,
                      pinned: Dict[Tuple[str, str], str]) -> Optional[str]:
    return config.expect_launched.get((route, case)) or pinned.get((route, case))


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def describe_row(config: Config, row: Row, chosen: str = "bound") -> Dict[str, Any]:
    """Everything the plan says about a row, as the dry run prints it."""
    entry: Dict[str, Any] = {
        "index": row.index, "of": row.of, "tier": row.tier, "kind": row.kind,
        "case": row.case, "res": row.res, "cells": row.cells, "pass": row.pass_index,
        "route": row.route, "ranks": row.ranks, "control": row.control,
        "configuration": {"build": row.build, "threads_per_rank": row.threads,
                          "split_chunks_evenly": ("default" if row.split is None
                                                  else row.split)},
        "gate": gate_kind(config, row), "gate_hold_s": gate_estimate(config,
                                                                    gate_kind(config, row)),
        "budget_s": row_budget(config, row), "timeout_s": row_timeout(config, row),
        "argv": None, "cwd": None, "env": None,
    }
    command = None
    if row.kind == "gpu":
        command = gpu_command(config, row, "<KEEP_CACHE>", "<UTC>")
        entry["table"] = config.table(row.route or "default")
        entry["expected_launched"] = config.expect_launched.get((row.route, row.case))
        entry["expected_repair"] = timing_cases.expected_repair(str(row.case))
    elif row.kind == "meep":
        command = meep_command(config, row, chosen, "<UTC>", "<BOX>")
        entry["binding"] = command.label
        entry["binding_mode"] = command.mode
        entry["windows"], entry["target_seconds"], entry["warm_seconds_cap"] = \
            meep_params(row)
        entry["out"] = command.out
    elif row.kind == "digest":
        command = digest_command(config, row)
    elif row.kind == "lift":
        command = lift_command(config, row)
    elif row.kind == "control":
        if config.array_control == "row":
            command = control_command(config, row, "<KEEP_CACHE>", "<UTC>")
        else:
            entry["reads"] = [gpu_out(config, row, route) for route in config.routes]
    entry["key"] = row_key(config, row, command.label if command else None)
    if command:
        entry["argv"] = command.argv
        entry["cwd"] = command.cwd
        entry["env"] = command.env.as_dict()
        entry["log"] = command.log
    return entry


def plan_record(config: Config, plan: List[Row]) -> Dict[str, Any]:
    rows = [describe_row(config, row) for row in plan]
    budget: Dict[str, Dict[str, int]] = {}
    for entry in rows:
        tier = budget.setdefault(entry["tier"], {"rows": 0, "row_s": 0, "gate_s": 0})
        tier["rows"] += 1
        tier["row_s"] += entry["budget_s"]
        tier["gate_s"] += entry["gate_hold_s"]
    per_size: Dict[str, Dict[str, int]] = {}
    for entry in rows:
        if entry["tier"] in ("priority", "rest", "largest"):
            key = f"{entry['case']}@{entry['cells']}"
            size = per_size.setdefault(key, {"rows": 0, "row_s": 0, "gate_s": 0})
            size["rows"] += 1
            size["row_s"] += entry["budget_s"]
            size["gate_s"] += entry["gate_hold_s"]
    # A GPU pinned by UUID is keyed by its index in the box string; a dry run starts no
    # process to look the index up, so it names the key by what it will hold.
    gpu_key = (f"gpu<index of {config.gpu}>" if str(config.gpu).startswith("GPU-")
               else f"gpu{config.gpu}")
    linux_box = ["load", "busy_cpus", "free_gb", gpu_key, "others", "account"]
    darwin_box = ["load", "busy_cpus", "free_gb", "metal_neighbours", "account"]
    return {
        "params": {name: {"value": config.values.get(name),
                          "source": config.sources.get(name)}
                   for name in sorted(config.values)},
        "resolved": {
            "platform": config.platform, "out": config.out, "tree": config.tree,
            "api": config.api, "harness": config.harness, "drivers": config.drv,
            "drivers_mode": config.drivers_mode, "tree_layout": config.tree_layout,
            "launcher": config.launcher, "launcher_kind": config.launcher_kind,
            "mpi_implementation": config.mpi_impl,
            "physical_cores": config.physical, "logical_cpus": config.logical,
            "smt": config.smt,
            "rank_blocks": {"A": config.blocks[0], "B": config.blocks[1],
                            "C": config.blocks[2]},
            "probe_ranks": config.probe_ranks, "preflight_ranks": config.preflight_ranks,
            "routes": config.routes, "cases": timing_cases.describe(config.cases),
            "sizes_cells": config.sizes, "controls": config.controls,
            "controls_omitted": config.controls_omitted,
            "builds": {name: dataclasses.asdict(build)
                       for name, build in config.builds.items()},
            "expect_launched": {f"{r}:{c}": t for (r, c), t in
                                config.expect_launched.items()},
            "gate": dataclasses.asdict(config.gate),
            "box_keys": darwin_box if config.platform == "darwin" else linux_box,
            "kmp_duplicate_lib_ok": (
                "GPU rows set TRUE (torch beside NumPy), MEEP rows unset it"
                if config.platform == "darwin"
                else "inherited from the launching shell, never set"),
            "sockets": config.sockets, "threads_per_core": config.threads_per_core,
            "restricted_by_affinity": config.restricted,
            "gpu_flags": config.gpu_flags,
        },
        "budget_by_tier": budget, "budget_by_size": per_size,
        "total": {"rows": len(rows), "row_s": sum(e["budget_s"] for e in rows),
                  "gate_s": sum(e["gate_hold_s"] for e in rows)},
        "rows": rows,
    }


def print_plan(record: Dict[str, Any], say: Callable[[str], None]) -> None:
    resolved = record["resolved"]
    say(f"DRY RUN: {record['total']['rows']} rows; nothing runs, no device is read, "
        f"nothing is written; rows would go under {resolved['out']}")
    say("=== PARAMETERS (value <- source) ===")
    for name, entry in record["params"].items():
        value = entry["value"]
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value)
        say(f"  {name} = {value!r} <- {entry['source']}")
    say("=== RESOLVED ===")
    for name in ("platform", "tree", "api", "harness", "drivers", "drivers_mode",
                 "tree_layout", "launcher", "mpi_implementation", "physical_cores",
                 "logical_cpus", "smt", "rank_blocks", "probe_ranks", "preflight_ranks",
                 "routes", "sizes_cells", "controls", "controls_omitted",
                 "expect_launched", "box_keys", "kmp_duplicate_lib_ok", "sockets",
                 "threads_per_core", "restricted_by_affinity", "gpu_flags"):
        say(f"  {name}: {resolved[name]}")
    for case in resolved["cases"]:
        say(f"  case {case['name']}: cell {case['cell']} pml {case['pml']} (PML fraction "
            f"{case['pml_fraction_nominal']}), source {case['source_component']}, "
            f"template {case['template']}, deposit repair {case['expected_repair']}")
    gate = resolved["gate"]
    say(f"  gate: GPU rows {gate['gpu_metric']} < {gate['cpu_max']}, MEEP rows "
        f"{gate['meep_metric']} < {gate['cpu_max']} ({gate['scope']} scope), two readings "
        f"{gate['hold_gpu_s']:.0f} s / {gate['hold_meep_s']:.0f} s apart; memory > "
        f"{gate['mem_min_gib']} GiB; account GPU processes <= {gate['account_max']}"
        + (f", cap {gate['cap']}" if gate["cap"] is not None else "")
        + ("; device free" if gate["check_device"] else ""))
    say("=== BUDGET (estimates, seconds) ===")
    for key, size in record["budget_by_size"].items():
        say(f"  size {key}: {size['rows']} rows, rows {size['row_s']} s + gates "
            f"{size['gate_s']} s")
    cumulative = 0
    for tier, entry in record["budget_by_tier"].items():
        cumulative += entry["row_s"] + entry["gate_s"]
        say(f"  tier {tier}: {entry['rows']} rows, rows {entry['row_s']} s + gates "
            f"{entry['gate_s']} s; cumulative {cumulative / 3600.0:.2f} h")
    total = record["total"]
    say(f"  TOTAL: {total['rows']} rows, rows {total['row_s'] / 3600.0:.2f} h + gates "
        f"{total['gate_s'] / 3600.0:.2f} h, before any wait for a quiet box")
    last = None
    for entry in record["rows"]:
        if entry["tier"] != last:
            say(f"=== tier {entry['tier']} ===")
            last = entry["tier"]
        what = {"gpu": f"GPU {entry['route']} route (table {entry.get('table')})",
                "meep": f"MEEP ranks={entry['ranks']} binding={entry.get('binding')}",
                "control": "ARRAY CONTROL", "digest": "GEOMETRY DIGEST (harness lift vs "
                "preflight MEEP rows)", "lift": "HARNESS-LIFT DIGEST OF RECORD",
                "decide": "DECIDE the binding"}[entry["kind"]]
        configuration = entry["configuration"]
        extra = ""
        if entry["kind"] == "meep" and (configuration["build"] != "reference"
                                        or configuration["threads_per_rank"] != 1
                                        or configuration["split_chunks_evenly"] != "default"):
            extra = (f" build={configuration['build']} threads="
                     f"{configuration['threads_per_rank']} split="
                     f"{configuration['split_chunks_evenly']}")
        say(f"row {entry['index']}/{entry['of']} [{entry['tier']}] {entry['key']} {what} "
            f"case={entry['case']} res={entry['res']} cells={entry['cells']} "
            f"pass={entry['pass']}{extra} budget={entry['budget_s']}s "
            f"gate={entry['gate'] or 'none'}(+{entry['gate_hold_s']}s)")
        if entry["env"]:
            delta = entry["env"]
            say("    env: " + " ".join(f"{k}={v}" for k, v in sorted(delta["set"].items()))
                + (" unset " + " ".join(delta["unset"]) if delta["unset"] else ""))
        if entry["argv"]:
            say(f"    cd {entry['cwd']} && " + " ".join(shlex.quote(a) for a in entry["argv"]))
        elif entry.get("reads"):
            say(f"    reads the array legs of {', '.join(entry['reads'])}; no process")
    say(f"DRY RUN DONE: {record['total']['rows']} rows planned")


# ---------------------------------------------------------------------------
# Live
# ---------------------------------------------------------------------------

class Log:
    def __init__(self, path: Optional[str]) -> None:
        self.path = path
        self.lock = threading.Lock()

    def say(self, message: str) -> None:
        line = f"{utc()} [ladder] {message}"
        with self.lock:
            print(line, flush=True)
            if self.path:
                with open(self.path, "a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
                    handle.flush()


class NotStarted(RuntimeError):
    """A row's process could not be executed (a missing launcher or interpreter)."""


def run_process(argv: Sequence[str], cwd: str, env: Dict[str, str], log: str,
                limit: float, pid_file: Optional[str] = None, kind: str = "cpu",
                kill_after: float = 60.0) -> Tuple[int, bool]:
    """``setsid timeout --signal=TERM --kill-after=60 <limit>``, in Python.

    The row runs in its own process group (so the guard and a signal reach every rank);
    at ``limit`` seconds the group gets TERM, and KILL ``kill_after`` seconds later.
    Returns (rc, timed_out); a timed-out row reads rc 124, as ``timeout`` reported it.
    A process that cannot be executed raises :class:`NotStarted`, its reason written
    into the log first.
    """
    os.makedirs(os.path.dirname(os.path.abspath(log)), exist_ok=True)
    with open(log, "ab") as handle, open(os.devnull, "rb") as nothing:
        try:
            process = subprocess.Popen(list(argv), cwd=cwd, env=env, stdout=handle,
                                       stderr=subprocess.STDOUT, stdin=nothing,
                                       start_new_session=True)
        except OSError as error:
            reason = f"{type(error).__name__}: {error} (argv[0] {argv[0] if argv else None!r})"
            handle.write(f"NOT STARTED: {reason}\n".encode("utf-8"))
            raise NotStarted(reason) from error
        if pid_file:
            with open(pid_file, "w", encoding="utf-8") as pids:
                pids.write(f"{kind} {process.pid}\n")
        timed_out = False
        try:
            rc = process.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            timed_out = True
            _signal_group(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=kill_after)
            except subprocess.TimeoutExpired:
                _signal_group(process.pid, signal.SIGKILL)
                process.wait()
            rc = 124
        finally:
            if pid_file:
                with open(pid_file, "w", encoding="utf-8"):
                    pass
    if rc in (-signal.SIGKILL, 137):
        timed_out = True
    return rc, timed_out


def _signal_group(pid: int, which: int) -> None:
    for target in (lambda: os.killpg(pid, which), lambda: os.kill(pid, which)):
        try:
            target()
        except (ProcessLookupError, PermissionError, OSError):
            pass


class Ladder:
    """The live run: snapshot, gates, rows, records, guard and monitor."""

    def __init__(self, config: Config, plan: List[Row], resume: bool) -> None:
        self.config = config
        self.plan = plan
        self.resume = resume
        self.out = config.out
        self.ledger = os.path.join(self.out, "ladder_rows.jsonl")
        self.log = Log(os.path.join(self.out, "progress.log"))
        self.pid_file = os.path.join(self.out, "current_row.pid")
        self.chosen = "bound" if config.binding == "probe" else config.binding
        self.pinned: Dict[Tuple[str, str], str] = {}
        self.keep_cache = os.path.join(self.out, "cache", "cupy")
        self.first_code: Optional[str] = None
        self.stop = threading.Event()
        self.base_env = dict(config.base_env)
        self.host = os.uname().nodename
        self.keepawake: Optional[subprocess.Popen] = None

    # ---- helpers -------------------------------------------------------------------
    def say(self, message: str) -> None:
        self.log.say(message)

    def die(self, message: str, code: int) -> "SystemExit":
        self.say(f"STOP: {message}")
        error = SystemExit(code)
        return error

    def done_path(self, key: str) -> str:
        return os.path.join(self.out, "done", key)

    def box(self) -> str:
        return timing_host.box_string(self.config.gate)

    def code_digest(self) -> str:
        try:
            _lines, digest = timing_records.manifest(self.config.tree, True)
        except SystemExit as error:
            raise self.die(str(error), EXIT_REFUSED)
        if self.first_code is None:
            self.first_code = digest
        elif digest != self.first_code:
            self.say(f"CODE DIGEST CHANGED: the tree that runs now hashes {digest}, the "
                     f"ladder started on {self.first_code}; every row records its own")
        return digest

    def gate(self, kind: str, label: str) -> float:
        try:
            return timing_host.gate(self.config.gate, kind, label, self.say, self.box)
        except timing_host.GateGaveUp as error:
            raise self.die(str(error), EXIT_GATE)

    def launch(self, command: "Command", kind: str) -> Tuple[int, bool, Optional[str]]:
        """Run a row's process: (rc, timed_out, why it never started or None)."""
        try:
            rc, timed_out = run_process(command.argv, command.cwd,
                                        command.env.apply(self.base_env), command.log,
                                        command.timeout, self.pid_file, kind)
        except NotStarted as error:
            return RC_NOT_STARTED, False, str(error)
        return rc, timed_out, None

    def not_started(self, row: Row, key: str, why: str, **extra: Any) -> None:
        """A row whose process never started: struck in the ledger, NO done marker, so a
        resume with the path fixed runs it."""
        timing_records.record_note(self.ledger, "not_started", f"not started: {why}",
                                   self.fields(row, key=key, **extra))
        self.say(f"row {row.index}/{row.of} {key} NOT STARTED: {why}")

    def cpu_trace(self, key: str, kind: str = "gpu") -> Tuple[threading.Event,
                                                                threading.Thread]:
        """CPU state sampled during a row. A MEEP row is sampled at the box monitor's
        interval: reading every CPU's cpufreq files and fsyncing a line each second
        preempts unbound ranks (and on older kernels interrupts every CPU)."""
        stop = threading.Event()
        path = os.path.join(self.out, "cpu_trace", f"{key}.jsonl")
        every = float(self.config.values["CPU_TRACE_S"])
        if kind == "meep":
            every = max(every, float(self.config.values["MONITOR_EVERY_S"]))
        if self.config.platform == "darwin":
            every = max(every, 15.0)

        def loop() -> None:
            while not stop.is_set():
                timing_records.append(path, timing_host.cpu_state(self.config.platform))
                stop.wait(every)

        thread = threading.Thread(target=loop, daemon=True)
        thread.start()
        return stop, thread

    def apple_gpu_trace(self, key: str) -> Tuple[threading.Event, threading.Thread]:
        """macOS: the Apple GPU's ioreg utilisation during a GPU row, every 15 s at least
        (clocks, temperature and power need root and are not recorded)."""
        stop = threading.Event()
        path = os.path.join(self.out, "gpu_trace", f"{key}.jsonl")
        every = max(float(self.config.values["GPU_TRACE_S"]), 15.0)

        def loop() -> None:
            while not stop.is_set():
                timing_records.append(path, timing_host.apple_gpu_state())
                stop.wait(every)

        thread = threading.Thread(target=loop, daemon=True)
        thread.start()
        return stop, thread

    def gpu_trace(self, key: str) -> Optional[subprocess.Popen]:
        if self.config.gpu is None or not timing_host.nvidia_smi_available():
            return None
        path = os.path.join(self.out, "gpu_trace", f"{key}.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        every = int(1000 * float(self.config.values["GPU_TRACE_S"]))
        handle = open(path, "ab")
        return subprocess.Popen(timing_host.gpu_trace_command(self.config.gpu, every),
                                stdout=handle, stderr=subprocess.STDOUT,
                                start_new_session=True)

    def record(self, recorder: Callable[..., Tuple[Dict[str, Any], str]], row: Row,
               *args: Any, **kwargs: Any) -> Tuple[Dict[str, Any], str]:
        """A recorder called in process, as ``ladder_tools.py``'s ``main`` once guarded
        it in a subprocess: whatever it raises is written to the ledger as a note and
        returned as a summary, and never stops the ladder."""
        try:
            return recorder(*args, **kwargs)
        except Exception as error:  # noqa: BLE001 - a recorder never stops the ladder
            reason = f"RECORDER ERROR {type(error).__name__}: {error}"
            try:
                timing_records.record_note(self.ledger, "recorder_error", reason,
                                           self.fields(row, recorder=getattr(
                                               recorder, "__name__", str(recorder))))
            except Exception:  # noqa: BLE001
                pass
            return {"struck": reason, "measured": None}, reason

    def finish(self, key: str, row: Row, started: float, rc: int, timed_out: bool,
               waited: float, summary: str) -> None:
        self.say(f"row {row.index}/{row.of} {key} DONE rc={rc} timed_out={int(timed_out)} "
                 f"{int(time.time() - started)}s gate_waited={int(waited)}s | {summary} | "
                 f"after: {self.box()}")
        with open(self.done_path(key), "w", encoding="utf-8"):
            pass

    def fields(self, row: Row, **extra: Any) -> Dict[str, Any]:
        out = {"index": row.index, "of": row.of, "tier": row.tier, "case": row.case,
               "res": row.res, "cells_planned": row.cells, "pass": row.pass_index,
               "run_dir": self.out, "host": self.host, "platform": self.config.platform,
               "require_ac_power": self.config.gate.require_ac_power,
               "physical_cores": self.config.physical}
        out.update(extra)
        return out

    # ---- rows ------------------------------------------------------------------------
    def do_gpu(self, row: Row) -> None:
        config = self.config
        key = row_key(config, row)
        if os.path.exists(self.done_path(key)):
            self.say(f"row {row.index}/{row.of} {key} already done: skipped")
            return
        waited = self.gate("gpu", key)
        code = self.code_digest()
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        command = gpu_command(config, row, self.keep_cache, stamp)
        os.makedirs(command.out or self.out, exist_ok=True)
        before = self.box()
        state_before = timing_host.gpu_state(config.gpu, config.platform)
        cpu_before = timing_host.cpu_state(config.platform)
        started = time.time()
        expect = expected_launched(config, str(row.route), str(row.case), self.pinned)
        self.say(f"row {row.index}/{row.of} {key} START cells={row.cells} pass="
                 f"{row.pass_index} budget={row_budget(config, row)}s gate_waited="
                 f"{int(waited)}s code={code[:12]} expect={expect} | {before}")
        trace_stop, trace = self.cpu_trace(key, "gpu")
        sampler = self.gpu_trace(key)
        apple = self.apple_gpu_trace(key) if config.platform == "darwin" else None
        try:
            rc, timed_out, never = self.launch(command, "gpu")
        finally:
            trace_stop.set()
            if sampler is not None:
                sampler.terminate()
            if apple is not None:
                apple[0].set()
        if never is not None:
            self.not_started(row, key, never, kind="gpu", route=row.route,
                             code_digest=code)
            if row.tier == "preflight":
                raise self.die(f"the preflight GPU {row.route} row never started: {never}",
                               EXIT_PREFLIGHT)
            return
        entry, summary = self.record(
            timing_records.record_gpu, row, self.ledger, command.out or "", command.log,
            expect, timing_cases.expected_repair(str(row.case)), True,
            self.fields(row, kind="gpu", route=row.route, table_asked=config.table(
                str(row.route)), rc=rc, timed_out=int(timed_out),
                budget_s=row_budget(config, row), elapsed_s=int(time.time() - started),
                gate_waited_s=int(waited), code_digest=code, box_before=before,
                preference="cuda" if row.route == "cuda" else "unset", gpu=config.gpu,
                flags=command.flags, key=key),
            {"gpu_state_before": state_before,
             "gpu_state_after": timing_host.gpu_state(config.gpu, config.platform),
             "cpu_state_before": cpu_before,
             "cpu_state_after": timing_host.cpu_state(config.platform),
             "allow_uncertified": config.allow_uncertified,
             "argv": command.argv, "env": command.env.as_dict()})
        self.finish(key, row, started, rc, timed_out, waited, summary)
        if row.tier == "preflight":
            ok, launched = preflight_gpu_verdict(entry, summary, expect)
            if not ok:
                raise self.die(f"the preflight GPU {row.route} row on {row.case} did not "
                               f"launch {expect or 'a kernel table'} bit-identically with "
                               f"the declared deposit-repair route: {summary} (see "
                               f"{command.log})", EXIT_PREFLIGHT)
            if expect is None:
                self.pinned[(str(row.route), str(row.case))] = str(launched)
                self.say(f"expected launch PINNED by the preflight: {row.route}:{row.case}"
                         f"={launched}")
                with open(os.path.join(self.out, "expected_launched.json"), "w",
                          encoding="utf-8") as handle:
                    json.dump({f"{r}:{c}": t for (r, c), t in self.pinned.items()},
                              handle, indent=2, sort_keys=True)

    def do_meep(self, row: Row) -> None:
        config = self.config
        mode, _flags, label = row_flags(config, row, self.chosen)
        key = row_key(config, row, label)
        if os.path.exists(self.done_path(key)):
            self.say(f"row {row.index}/{row.of} {key} already done: skipped")
            return
        waited = self.gate("meep", key)
        code = self.code_digest()
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        before = self.box()
        command = meep_command(config, row, self.chosen, stamp, before)
        os.makedirs(command.out or self.out, exist_ok=True)
        cpu_before = timing_host.cpu_state(config.platform)
        started = time.time()
        self.say(f"row {row.index}/{row.of} {key} START ranks={row.ranks} binding={label} "
                 f"build={row.build} threads={row.threads} cells={row.cells} pass="
                 f"{row.pass_index} budget={row_budget(config, row)}s gate_waited="
                 f"{int(waited)}s | {before}")
        trace_stop, _trace = self.cpu_trace(key, "meep")
        try:
            rc, timed_out, never = self.launch(command, "cpu")
        finally:
            trace_stop.set()
        if never is not None:
            self.not_started(row, key, never, kind="meep", ranks=row.ranks, binding=label,
                             build=row.build, control=row.control, code_digest=code)
            if row.tier == "preflight":
                raise self.die(f"the preflight MEEP row at {row.ranks} ranks never "
                               f"started: {never}", EXIT_PREFLIGHT)
            return
        rows_file = os.path.join(command.out or "", f"rows_r{row.ranks}.jsonl")
        entry, summary = self.record(
            timing_records.record_meep, row, self.ledger, rows_file, str(command.row_id), command.log,
            digest_file(config, str(row.case), int(row.res or 0)), None,
            self.fields(row, kind="meep", ranks=row.ranks, binding=label, rc=rc,
                        timed_out=int(timed_out), budget_s=row_budget(config, row),
                        elapsed_s=int(time.time() - started), gate_waited_s=int(waited),
                        code_digest=code, box_before=before,
                        launcher_line=command.launcher_line, key=key,
                        control=row.control, build=row.build),
            {"cpu_state_before": cpu_before,
             "cpu_state_after": timing_host.cpu_state(config.platform),
             "binding_mode": mode, "flags": command.flags,
             "kmp_duplicate_lib_ok_inherited": self.base_env.get("KMP_DUPLICATE_LIB_OK"),
             "argv": command.argv, "env": command.env.as_dict()})
        self.finish(key, row, started, rc, timed_out, waited, summary)
        if row.tier == "preflight" and not (summary.startswith("MEASURED")
                                            and "monitor_stepped=True" in summary
                                            and not entry.get("struck")):
            raise self.die(f"the preflight MEEP row at {row.ranks} ranks did not measure "
                           f"with its monitor stepped: {summary} (see {command.log})",
                           EXIT_PREFLIGHT)

    def do_control(self, row: Row) -> None:
        config = self.config
        key = row_key(config, row)
        if config.array_control != "row":
            sources = [gpu_out(config, row, route) for route in config.routes]
            _entry, summary = self.record(
                timing_records.record_control, row,
                self.ledger, os.path.join(self.out, "array_control", "rows.jsonl"), sources,
                config.harness, False, self.fields(row, kind="control", key=key),
                timing_records.ARRAY_ENV_GATES[config.table(config.routes[0])]
                if config.routes else "gate_dispatch_fused_route.py")
            self.say(f"row {row.index}/{row.of} {key} DONE | {summary}")
            return
        if os.path.exists(self.done_path(key)):
            self.say(f"row {row.index}/{row.of} {key} already done: skipped")
            return
        waited = self.gate("gpu", key)
        code = self.code_digest()
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        command = control_command(config, row, self.keep_cache, stamp)
        os.makedirs(command.out or self.out, exist_ok=True)
        before = self.box()
        started = time.time()
        self.say(f"row {row.index}/{row.of} {key} START | {before}")
        rc, timed_out, never = self.launch(command, "gpu")
        if never is not None:
            self.not_started(row, key, never, kind="control", code_digest=code)
            return
        _entry, summary = self.record(
            timing_records.record_control, row, self.ledger, os.path.join(self.out, "array_control", "rows.jsonl"),
            [command.out or ""], config.harness, True,
            self.fields(row, kind="control", rc=rc, timed_out=int(timed_out),
                        budget_s=row_budget(config, row),
                        elapsed_s=int(time.time() - started), gate_waited_s=int(waited),
                        code_digest=code, box_before=before, key=key))
        self.finish(key, row, started, 0, timed_out, waited, summary)

    def do_digest(self, row: Row) -> None:
        config = self.config
        command = digest_command(config, row)
        code = self.code_digest()
        started = time.time()
        rc, timed_out, never = self.launch(command, "cpu")
        if never is not None:
            raise self.die(f"the preflight digest never started: {never}", EXIT_DIGEST)
        verdict = None
        try:
            with open(command.log, "r", encoding="utf-8", errors="replace") as handle:
                verdict = [line.strip() for line in handle if line.startswith("VERDICT")][-1:]
        except OSError:
            verdict = None
        text = verdict[0] if verdict else "no verdict line"
        timing_records.record_note(self.ledger, "digest", text,
                                   self.fields(row, rc=rc, timed_out=int(timed_out),
                                               elapsed_s=int(time.time() - started),
                                               code_digest=code))
        self.say(f"row {row.index}/{row.of} {row_key(config, row)} DONE rc={rc} | {text}")
        if rc != 0:
            raise self.die(f"the MEEP bench and the harness lift do not carry the same "
                           f"geometry digest for {row.case} at res {row.res} (see "
                           f"{command.log})", EXIT_DIGEST)

    def do_lift(self, row: Row) -> None:
        config = self.config
        key = row_key(config, row)
        if os.path.exists(self.done_path(key)):
            self.say(f"row {row.index}/{row.of} {key} already done: skipped")
            return
        command = lift_command(config, row)
        code = self.code_digest()
        started = time.time()
        self.say(f"row {row.index}/{row.of} {key} START {row.case} res {row.res} "
                 f"({row.cells:,} cells) timeout {command.timeout}s")
        rc, timed_out, never = self.launch(command, "cpu")
        if never is not None:
            raise self.die(f"the harness lift for {row.case} at res {row.res} never "
                           f"started: {never}", EXIT_DIGEST)
        text = "no verdict line"
        try:
            with open(command.log, "r", encoding="utf-8", errors="replace") as handle:
                lines = [line.strip() for line in handle if line.startswith("VERDICT")]
                text = lines[-1] if lines else text
        except OSError:
            pass
        timing_records.record_note(self.ledger, "lift", text,
                                   self.fields(row, rc=rc, timed_out=int(timed_out),
                                               elapsed_s=int(time.time() - started),
                                               key=key, code_digest=code,
                                               digest_file=digest_file(
                                                   config, str(row.case), int(row.res or 0))))
        if rc != 0:
            # A LIFT THAT DID NOT SETTLE ITS SIZE'S DIGEST stops the ladder: its driver
            # disagreed with the MEEP object, or it found a digest of record (a MEEP
            # row's, taken first) that is not its own. Every ratio at that size would
            # divide two different simulations.
            raise self.die(f"the harness lift for {row.case} at res {row.res} did not "
                           f"settle the digest of record: {text} (see {command.log})",
                           EXIT_DIGEST)
        self.finish(key, row, started, rc, timed_out, 0, text)

    def do_decide(self, row: Row) -> None:
        chosen, record = timing_records.decide_binding(
            os.path.join(self.out, "bindprobe"), ["bound", "unbound"], "bound",
            os.path.join(self.out, "binding_decision.json"))
        self.chosen = chosen if chosen in ("bound", "unbound") else "bound"
        with open(os.path.join(self.out, "chosen_binding"), "w", encoding="utf-8") as handle:
            handle.write(self.chosen + "\n")
        self.say(f"row {row.index}/{row.of} bindprobe_decide DONE | MEEP rows at or below "
                 f"{self.config.physical} ranks use binding '{self.chosen}' "
                 f"({record['reason']})")

    # ---- the run ---------------------------------------------------------------------
    def snapshot(self) -> None:
        config = self.config
        if config.values["TREE_MODE"] != "snapshot" or \
                os.path.exists(os.path.join(self.out, "tree_manifest.sha256")):
            return
        excluded = config.snapshot_exclude
        names = [e for e in excluded if "/" not in e]
        paths = [e for e in excluded if "/" in e]
        for attempt in (1, 2, 3):
            _l1, m1 = timing_records.manifest(config.tree_src, False, paths, names)
            if os.path.exists(config.tree):
                shutil.rmtree(config.tree)

            def ignore(directory: str, entries: List[str]) -> List[str]:
                relative = os.path.relpath(directory, config.tree_src)
                skipped = set()
                for entry in entries:
                    path = os.path.normpath(os.path.join(relative, entry))
                    if any(_fnmatch(entry, pattern) for pattern in names) or \
                            any(path == p.strip("/") for p in paths):
                        skipped.add(entry)
                return sorted(skipped)

            shutil.copytree(config.tree_src, config.tree, symlinks=True, ignore=ignore)
            lines, m2 = timing_records.manifest(config.tree, False, paths, names)
            _l3, m3 = timing_records.manifest(config.tree_src, False, paths, names)
            with open(os.path.join(self.out, "tree_manifest.listing"), "w",
                      encoding="utf-8") as handle:
                handle.write("\n".join(lines) + "\n")
            self.say(f"snapshot attempt {attempt}: source before [{m1}] copy [{m2}] source "
                     f"after [{m3}]")
            if m1 == m2 == m3:
                with open(os.path.join(self.out, "tree_manifest.sha256"), "w",
                          encoding="utf-8") as handle:
                    handle.write(m2 + "\n")
                return
            self.say("the source moved during the copy; waiting 60 s")
            time.sleep(60)
        raise self.die("the tree under test did not hold still for a snapshot in 3 "
                       "attempts", EXIT_REFUSED)

    def copy_drivers(self) -> None:
        config = self.config
        if config.drivers_mode != "copy":
            return
        os.makedirs(config.drv, exist_ok=True)
        digests = {}
        for name in DRIVER_FILES:
            source = os.path.join(HERE, name)
            target = os.path.join(config.drv, name)
            if not os.path.exists(target):
                shutil.copy2(source, target)
            digests[name] = timing_records.sha256_file(target)
            if self.resume and digests[name] != timing_records.sha256_file(source):
                self.say(f"driver {name} in this run differs from {source}; the run keeps "
                         "its own copy")
        with open(os.path.join(self.out, "drivers.sha256"), "w", encoding="utf-8") as handle:
            for name, digest in sorted(digests.items()):
                handle.write(f"{digest}  {name}\n")

    def mint_keep_cache(self) -> None:
        config = self.config
        if not any(route in NVIDIA_ROUTES for route in config.routes):
            return
        if config.subnormal_policy != "keep":
            self.say(f"subnormal policy {config.subnormal_policy}: no keep cache minted")
            return
        self.gate("gpu", "the keep-cache mint")
        env = dict(self.base_env)
        env.update({"CUDA_VISIBLE_DEVICES": str(config.gpu), "PYTHONPATH": config.api,
                    "CUPY_CACHE_DIR": os.path.join(self.out, "cache", "cupy")})
        code = ("from meep_gpu import subnormal_policy as sp\n"
                "r = sp.point_cupy_cache_at_keep_policy(); p = r.get('provenance') or {}\n"
                "print(r['after'] if (not p.get('reasons')) and p.get('marked') else "
                "'REFUSED ' + repr(p.get('reasons')))")
        # FROM THE SNAPSHOT'S ROOT: python -c puts the working directory first on
        # sys.path, so run from the ladder's own directory it would import the live
        # tree's meep_gpu whatever PYTHONPATH says.
        text = timing_host.run_text([config.python, "-c", code], timeout=600, env=env,
                                    cwd=config.api) or ""
        lines = [line for line in text.splitlines() if line.strip()]
        cache = lines[-1].strip() if lines else ""
        if "ftz_stripped" not in cache:
            raise self.die(f"the keep cache was not minted: {cache or text[-400:]}", 4)
        if not _inside(cache, self.out):
            raise self.die(f"the keep cache {cache} is outside this run", 4)
        self.keep_cache = cache
        self.say(f"keep cache: {cache}")

    def guard(self) -> None:
        config = self.config
        if config.gate.cap is None:
            return
        previous = 0
        path = os.path.join(self.out, "capguard.log")
        while not self.stop.is_set():
            device = timing_host.device_state(config.gate.gpu, config.uid, config.platform)
            count = device.get("account_gpu_processes")
            count = 0 if count is None else int(count)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(f"{utc()} account={count}\n")
            if count > config.gate.cap and previous > config.gate.cap:
                kind, pid = "", ""
                try:
                    with open(self.pid_file, "r", encoding="utf-8") as handle:
                        kind, _, pid = handle.read().strip().partition(" ")
                except OSError:
                    pass
                if kind == "gpu" and pid.isdigit():
                    with open(path, "a", encoding="utf-8") as handle:
                        handle.write(f"{utc()} CAP BREACH ({previous} then {count}): killing "
                                     f"this ladder's GPU row (pid {pid}) and the ladder\n")
                    _signal_group(int(pid), signal.SIGTERM)
                    time.sleep(10)
                    _signal_group(int(pid), signal.SIGKILL)
                    os.kill(os.getpid(), signal.SIGTERM)
                    return
                with open(path, "a", encoding="utf-8") as handle:
                    handle.write(f"{utc()} over the cap ({previous} then {count}) with no "
                                 "GPU row of this ladder running: nothing of ours to kill\n")
            previous = count
            self.stop.wait(60)

    def monitor(self) -> None:
        every = float(self.config.values["MONITOR_EVERY_S"])
        path = os.path.join(self.out, "box_trace.log")
        while not self.stop.is_set():
            try:
                with open(self.pid_file, "r", encoding="utf-8") as handle:
                    current = handle.read().strip()
            except OSError:
                current = ""
            line = f"{utc()} {self.box()} row=[{current}]"
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            self.stop.wait(every)

    def environment(self) -> None:
        config = self.config
        builds: Dict[str, Any] = {}
        threaded = {row.build for row in self.plan if row.kind == "meep" and row.threads > 1}
        for name, build in config.builds.items():
            record = timing_records.meep_build_record(build.python, build.config_log,
                                                      build.launcher)
            if build.record:
                try:
                    with open(build.record, "r", encoding="utf-8") as handle:
                        record["build_record_file"] = json.load(handle)
                except (OSError, ValueError) as error:
                    raise self.die(f"the build record {build.record} of {name} does not "
                                   f"read: {error}", EXIT_REFUSED)
            record["unrecorded"] = build.unrecorded
            record["mpi_implementation_used"] = build.mpi_impl
            probe = record.get("probe") or {}
            if probe.get("single_precision") is not True:
                raise self.die(f"the MEEP build {name} ({build.python}) is not single "
                               f"precision: {probe or record.get('probe_output')}",
                               EXIT_REFUSED)
            if name in threaded and record.get("openmp") is not True:
                raise self.die(f"a threaded control (threads2/threads4) is planned for the "
                               f"MEEP build {name}, and its own libraries link no OpenMP "
                               f"runtime (openmp={record.get('openmp')!r}, "
                               f"{record.get('openmp_linked')}); every such row would be "
                               "refused; drop those controls with --controls", EXIT_REFUSED)
            path = probed_record_path(config, name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2, sort_keys=True, default=str)
            builds[name] = record
        params = {name: {"value": config.values.get(name),
                         "source": config.sources.get(name)} for name in config.values}
        with open(os.path.join(self.out, "params.env"), "w", encoding="utf-8") as handle:
            for name, entry in sorted(params.items()):
                value = entry["value"]
                if isinstance(value, list):
                    value = ",".join(str(v) for v in value)
                handle.write(f"{name}={'' if value is None else value}  # {entry['source']}\n")
        package_env = gpu_env(config, config.routes[0], self.keep_cache).apply(
            self.base_env) if config.routes else None
        timing_records.environment_record(
            os.path.join(self.out, "environment.json"), tree=config.tree, drivers=config.drv,
            python=config.python, builds=builds, gpu=config.gpu, params=dict(
                params, resolved=plan_record(config, self.plan)["resolved"],
                mpi_implementation_from_version=timing_host.mpi_implementation(
                    timing_host.run_text([config.launcher, "--version"])),
                keep_cache=self.keep_cache,
                keepawake=(None if self.keepawake is None else
                           {"pid": self.keepawake.pid,
                            "argv": list(self.keepawake.args)}),
                host=self.host), platform=config.platform,
            package_env=package_env, package_cwd=config.api)

    def identity(self) -> Dict[str, Any]:
        """The host and the pinned GPU this ladder runs on."""
        found: Dict[str, Any] = {"host": self.host, "platform": self.config.platform,
                                 "gpu": self.config.gpu, "gpu_uuid": None}
        if self.config.gpu is not None and self.config.platform != "darwin":
            found["gpu_uuid"] = timing_host.as_uuid(self.config.gpu,
                                                    timing_host.gpu_uuids() or {})
        return found

    def check_identity(self) -> None:
        """ONE HOST AND ONE GPU PER LADDER. The first start writes them; a ``--resume``
        that lands on another node (a resubmitted Slurm job), or on another device
        under the same index, is refused by name rather than mixing hosts in a ledger."""
        path = os.path.join(self.out, "host_identity.json")
        now = self.identity()
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(dict(now, utc=utc()), handle, indent=2, sort_keys=True)
            return
        with open(path, "r", encoding="utf-8") as handle:
            first = json.load(handle)
        differs = [f"{key} {first.get(key)!r} -> {now.get(key)!r}"
                   for key in ("host", "platform", "gpu_uuid")
                   if first.get(key) != now.get(key)]
        if differs:
            raise self.die("this ladder started on another host or GPU than the one it "
                           f"resumes on ({'; '.join(differs)}); a ladder's rows come "
                           "from one host and one device -- start a new stamp",
                           EXIT_REFUSED)

    def keep_awake(self) -> None:
        """macOS: hold the Mac awake for the life of this process (``caffeinate -i -s
        -w <pid>``: no idle sleep, no system sleep on AC power), and record it."""
        if self.config.platform != "darwin" or shutil.which("caffeinate") is None:
            return
        self.keepawake = subprocess.Popen(
            ["caffeinate", "-i", "-s", "-w", str(os.getpid())],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.say(f"keep-awake: caffeinate -i -s -w {os.getpid()} (pid {self.keepawake.pid})")

    def run(self) -> int:
        config = self.config
        os.makedirs(self.out, exist_ok=True)
        for name in ("done", "logs", "digests", os.path.join("cache", "triton")):
            os.makedirs(os.path.join(self.out, name), exist_ok=True)
        with open(self.pid_file, "w", encoding="utf-8"):
            pass
        self.say(f"LADDER START stamp={config.stamp} pid={os.getpid()} nice={os.nice(0)} "
                 f"rows={len(self.plan)} drivers={config.drv} host={self.host} "
                 f"gpu={config.gpu}")
        self.check_identity()
        self.keep_awake()
        record = plan_record(config, self.plan)
        with open(os.path.join(self.out, "plan.json"), "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=1, sort_keys=True, default=str)
        self.say(f"box: {self.box()}")
        self.snapshot()
        self.copy_drivers()
        if not os.path.isfile(os.path.join(config.harness, "bench_fused_products.py")):
            raise self.die(f"no harness under {config.harness}", EXIT_REFUSED)
        if not os.path.isfile(os.path.join(config.drv, "bench_meep_identical_case.py")):
            raise self.die(f"no timing drivers under {config.drv}; a development-layout tree "
                           "needs --drivers-mode copy", EXIT_REFUSED)
        try:
            _l, start_code = timing_records.manifest(config.tree_src, True)
            self.say(f"tree under test: code digest at start {start_code}")
        except SystemExit as error:
            raise self.die(str(error), EXIT_REFUSED)
        self.mint_keep_cache()
        self.environment()
        pinned_path = os.path.join(self.out, "expected_launched.json")
        if os.path.exists(pinned_path):
            with open(pinned_path, "r", encoding="utf-8") as handle:
                for name, tables in json.load(handle).items():
                    route, _, case = name.partition(":")
                    self.pinned[(route, case)] = tables
        chosen_path = os.path.join(self.out, "chosen_binding")
        if os.path.exists(chosen_path):
            with open(chosen_path, "r", encoding="utf-8") as handle:
                self.chosen = handle.read().strip() or self.chosen
        threads = [threading.Thread(target=self.guard, daemon=True),
                   threading.Thread(target=self.monitor, daemon=True)]
        for thread in threads:
            thread.start()

        def on_signal(number: int, _frame: Any) -> None:
            kind, pid = "", ""
            try:
                with open(self.pid_file, "r", encoding="utf-8") as handle:
                    kind, _, pid = handle.read().strip().partition(" ")
            except OSError:
                pass
            if pid.isdigit():
                _signal_group(int(pid), signal.SIGTERM)
            self.say(f"TERMINATED by signal {number}")
            raise SystemExit(EXIT_SIGNAL)

        signal.signal(signal.SIGTERM, on_signal)
        signal.signal(signal.SIGINT, on_signal)
        last = None
        try:
            for row in self.plan:
                if row.tier != last:
                    self.say(f"=== tier {row.tier} ===")
                    last = row.tier
                {"gpu": self.do_gpu, "meep": self.do_meep, "control": self.do_control,
                 "digest": self.do_digest, "lift": self.do_lift,
                 "decide": self.do_decide}[row.kind](row)
        finally:
            self.stop.set()
        _l, end_code = timing_records.manifest(config.tree_src, True)
        self.say(f"tree under test: code digest at end {end_code} (at start {start_code})")
        self.say(f"LADDER DONE {len(self.plan)} rows | {self.box()}")
        return 0


def _fnmatch(name: str, pattern: str) -> bool:
    import fnmatch  # noqa: PLC0415
    return fnmatch.fnmatch(name, pattern)


# ---------------------------------------------------------------------------
# Live checks and the self-test
# ---------------------------------------------------------------------------

def source_tree_config(config: Config) -> Config:
    """The configuration with the API and harness of the SOURCE tree (the snapshot does
    not exist before the run), for probes taken by the live checks."""
    api = config.tree_src if config.tree_layout == "release" else os.path.join(
        config.tree_src, "apps", "api")
    return dataclasses.replace(config, api=api,
                               harness=os.path.join(api, "parity", "meep_gpu"))


def resolve_gpu(config: Config, problem: Callable[[str], None],
                uuids: Optional[Dict[str, str]] = None,
                inherited: Optional[str] = None) -> None:
    """The pinned NVIDIA GPU, checked BY UUID, and ``--gpu auto`` resolved.

    Every device name -- ``--gpu``, an inherited ``CUDA_VISIBLE_DEVICES`` (a Slurm
    allocation's, index or UUID), ``--refuse-gpus`` -- is mapped to its UUID through
    ``nvidia-smi --query-gpu=index,uuid`` before it is compared, so an index and a UUID
    of one device agree and two devices that share an index across views do not.
    ``auto`` becomes the UUID of the one allocated device, written into the
    configuration and the gate before any row runs (a dry run keeps the literal).
    """
    if not any(route in NVIDIA_ROUTES for route in config.routes):
        return
    if uuids is None:
        uuids = timing_host.gpu_uuids()
    if uuids is None:
        problem("an NVIDIA route is planned and nvidia-smi does not list the GPUs")
        return
    listed = set(uuids.values())
    allocated_names = _split_list(inherited) if inherited else sorted(uuids, key=int)
    allocated = [timing_host.as_uuid(name, uuids) for name in allocated_names]
    unknown = [name for name, uuid in zip(allocated_names, allocated)
               if uuid is None or uuid not in listed]
    if unknown:
        problem(f"the inherited CUDA_VISIBLE_DEVICES={inherited} names {unknown}, which "
                f"nvidia-smi does not list ({uuids})")
        return
    if config.gpu == "auto":
        if len(allocated) != 1:
            problem(f"--gpu auto needs exactly one allocated GPU; this host or allocation "
                    f"shows {allocated_names}")
            return
        chosen = str(allocated[0])
    else:
        chosen = timing_host.as_uuid(config.gpu, uuids) or ""
        if chosen not in listed:
            problem(f"--gpu {config.gpu} is not a device nvidia-smi lists ({uuids})")
            return
        if inherited and chosen not in allocated:
            problem(f"--gpu {config.gpu} ({chosen}) is outside the inherited "
                    f"CUDA_VISIBLE_DEVICES {inherited} (a Slurm allocation's GPUs)")
            return
    refused = {timing_host.as_uuid(name, uuids)
               for name in _split_list(config.values.get("REFUSE_GPUS"))}
    if chosen in refused:
        problem(f"device {chosen} is in --refuse-gpus ({config.values.get('REFUSE_GPUS')})")
        return
    if config.gpu == "auto":
        config.gpu = chosen
        config.gate.gpu = chosen
        config.sources["GPU"] = f"{config.sources.get('GPU')} (auto -> {chosen})"


#: Run under the GPU rows' environment by the live checks: which GPU route the
#: package resolves here. Reads device presence only; allocates nothing on it.
BACKEND_PROBE = ("from meep_gpu import backends; "
                 "print('BACKEND', backends.available_gpu(), flush=True)")


def backend_probe(config: Config) -> Tuple[Optional[str], str]:
    """(``cuda``/``metal``/``None``, output) of :data:`BACKEND_PROBE` under the first
    route's GPU-row environment, from the source tree's root."""
    source = source_tree_config(config)
    env = gpu_env(source, config.routes[0], os.path.join(config.out, "cache", "cupy")
                  ).apply(config.base_env)
    text = timing_host.run_text([config.python, "-c", BACKEND_PROBE], timeout=300,
                                env=env, cwd=source.api) or ""
    found = re.search(r"^BACKEND (\S+)", text, re.M)
    value = found.group(1) if found else None
    return (None if value in (None, "None") else value), text


def live_checks(config: Config, resume: bool, collect: bool = False,
                environ: Optional[Dict[str, str]] = None) -> List[str]:
    """Refusals that need the filesystem or the host. ``collect`` returns them instead
    of raising the first, for ``--selftest``. Resolves ``--gpu auto`` (a live run's
    configuration then names the device by UUID) and a launcher whose implementation
    only its ``--version`` names."""
    environ = dict(os.environ if environ is None else environ)
    problems: List[str] = []

    def problem(message: str) -> None:
        if not collect:
            raise refuse(message)
        problems.append(message)

    nice = os.nice(0)
    if nice != 0 and not int(config.values["ALLOW_NICE"]):
        problem(f"this process runs at nice {nice}; launch it un-niced or pass "
                "--allow-nice 1 (zsh starts a background job at nice 5 unless "
                "'setopt no_bg_nice'; or launch from bash)")
    if config.platform == "darwin":
        priority = timing_host.darwin_process_priority()
        if priority not in (None, 0):
            problem(f"this process runs at darwin priority {priority} (background QoS): "
                    "its ranks would be confined to the efficiency cores; launch it from "
                    "a foreground session")
    if os.path.exists(config.out) and not resume:
        problem(f"{config.out} exists; --resume continues it")
    if not os.path.exists(config.python):
        problem(f"--python {config.python} does not exist")
    for name, build in config.builds.items():
        if config.launcher_kind == "srun" or os.sep not in build.launcher:
            if shutil.which(build.launcher) is None:
                problem(f"the launcher of build {name}, {build.launcher}, is not on PATH")
        elif not os.path.exists(build.launcher):
            problem(f"the launcher of build {name}, {build.launcher}, does not exist")
        if build.record and not os.path.isfile(build.record):
            problem(f"the build record of {name}, {build.record}, does not exist")
        if build.config_log and not os.path.isfile(build.config_log):
            problem(f"the config.log of {name}, {build.config_log}, does not exist")
        if name != "reference" and not os.path.exists(build.python):
            problem(f"the interpreter of build {name}, {build.python}, does not exist")
        if config.launcher_kind != "srun":
            # THE LAUNCHER'S OWN --version decides; the files beside it are a dry run's
            # stand-in. A disagreement is refused; a launcher the files could not name
            # takes the implementation its --version names.
            version = timing_host.mpi_implementation(
                timing_host.run_text([build.launcher, "--version"]))
            if version != "unknown" and build.mpi_impl not in ("unknown", version):
                problem(f"the launcher of build {name}, {build.launcher}, reports "
                        f"{version} in --version and the files beside it say "
                        f"{build.mpi_impl}; pass --mpi-impl")
            elif version != "unknown" and build.mpi_impl == "unknown":
                build.mpi_impl = version
                if name == "reference":
                    config.mpi_impl = version
    if config.platform == detect_platform() and config.uid is None:
        problem(f"--account {config.account!r} names no user on this host: the account "
                "gate and the cap guard would count nothing and pass forever")
    if not os.path.isdir(config.tree_src):
        problem(f"the tree under test {config.tree_src} does not exist")
    if any(route in NVIDIA_ROUTES for route in config.routes):
        if config.libcuda_stub and not os.path.isfile(
                os.path.join(config.libcuda_stub, "libcuda.so.1")):
            problem(f"--libcuda-stub {config.libcuda_stub} holds no libcuda.so.1")
        if not timing_host.nvidia_smi_available():
            problem("an NVIDIA route is planned and nvidia-smi is not on PATH")
        else:
            resolve_gpu(config, problem, inherited=environ.get("CUDA_VISIBLE_DEVICES"))
    if config.platform == "linux" and config.platform == detect_platform():
        affinity = timing_host.affinity()
        for message in timing_host.slurm_refusals(
                environ, config.builds["reference"].mpi_impl, config.launcher_kind,
                max(config.blocks[0] + config.blocks[1] + config.blocks[2]
                    + config.probe_ranks + config.preflight_ranks),
                len(affinity) if affinity is not None else None):
            problem(message)
    if config.platform == "darwin" and int(config.values["REQUIRE_AC_POWER"]):
        state = timing_host.cpu_state_darwin()
        if state.get("on_ac_power") is False:
            problem(f"the Mac is on {state.get('power_source')}; timing on battery is "
                    "refused (--require-ac-power 0 to override)")
        if state.get("low_power_mode"):
            problem("the Mac is in low power mode; timing there is refused "
                    "(--require-ac-power 0 to override)")
    if config.routes and os.path.isdir(config.tree_src) and os.path.exists(config.python) \
            and config.gpu != "auto":
        want = "metal" if config.metal_only else "cuda"
        found, text = backend_probe(config)
        if found != want:
            problem(f"under the GPU rows' environment the package resolves the GPU route "
                    f"{found!r}, and the planned routes need {want!r}: "
                    f"{text.strip()[-300:] or 'no output'}")
    return problems


def placement_modes(config: Config) -> List[Tuple[str, int, int]]:
    """(mode, ranks, threads) the launcher probe checks: every mode the plan uses."""
    a, b, c = config.blocks
    largest = max(a + b + c + config.probe_ranks)
    modes: List[Tuple[str, int, int]] = []
    if config.platform == "darwin":
        return [("unbound", min(largest, config.logical), 1)]
    modes.append(("bound", min(largest, config.physical), 1))
    modes.append(("unbound", min(largest, config.physical), 1))
    if c:
        modes.append(("over", max(c), 1))
    if "hwthread" in config.controls:
        modes.append(("bound-hwthread", int(round(4 * config.physical / 3.0)), 1))
    for control, threads in (("threads2", 2), ("threads4", 4)):
        if control in config.controls:
            modes.append(("bound-pe", max(1, int(round(config.physical / float(threads)))),
                          threads))
    return modes


def launcher_probe(config: Config) -> List[str]:
    """Start the launcher in every binding mode the plan uses, read each rank's CPU set,
    and check it against what the mode promises (:func:`timing_host.check_placement`).
    Returns every failure, by name."""
    failures: List[str] = []
    core_map = (config.topology or {}).get("core_of_cpu") or {}
    allocation = timing_host.affinity()
    impl = config.builds["reference"].mpi_impl if config.launcher_kind != "srun" else "srun"
    for mode, ranks, threads in placement_modes(config):
        try:
            flags, label = timing_host.binding_flags(
                impl, mode, threads, config.platform, config.explicit_flags, ranks=ranks,
                **config.flag_values())
        except SystemExit as error:
            failures.append(f"{mode}: {error}")
            print(f"selftest launcher-probe {mode}: {error}")
            continue
        if config.launcher_kind == "srun":
            argv = [config.launcher, f"--mpi={config.values['SRUN_MPI']}", "-n", str(ranks)]
        else:
            argv = [config.launcher, "-np", str(ranks)]
        argv += shlex.split(flags) + [config.python, "-c", timing_host.AFFINITY_PROBE]
        text = timing_host.run_text(argv, timeout=180) or ""
        affinities = timing_host.parse_affinity_lines(text)
        if config.platform == "darwin":
            found = ([] if len(affinities) == ranks
                     else [f"{len(affinities)} of {ranks} ranks answered"])
        elif not core_map and mode in timing_host.BOUND_MODES:
            found = [f"the topology's core map is unavailable (pass no "
                     f"--physical-cores/--logical-cpus override that hides it?): "
                     f"{len(affinities)} of {ranks} ranks answered, placement unchecked"]
        else:
            found = timing_host.check_placement(mode, affinities, ranks, core_map,
                                                allocation, threads)
        verdict = "PLACEMENT OK" if not found else "PLACEMENT FAILED: " + "; ".join(found)
        print(f"selftest launcher-probe {label} -np {ranks}: {len(affinities)} of {ranks} "
              f"ranks answered; {verdict}"
              + ("" if affinities else f"; output: {text[-600:]!r}"))
        failures += [f"{label} -np {ranks}: {why}" for why in found]
    return failures


def selftest(config: Config, launcher_probe_too: bool) -> int:
    apply_process_environment()
    print(f"selftest platform={config.platform} host={os.uname().nodename} "
          f"nice={os.nice(0)} OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
    readers = timing_host.selftest_readers(config.platform)
    for name, value in readers.items():
        print(f"selftest {name}: {json.dumps(value, default=str)}")
    problems = live_checks(config, resume=False, collect=True)
    device = timing_host.device_state(config.gate.gpu, config.uid, config.platform)
    print(f"selftest account={config.account} uid={config.uid} gpu={config.gpu} device: "
          f"{json.dumps(device, default=str)}")
    for name, build in config.builds.items():
        version = timing_host.run_text([build.launcher, "--version"]) or ""
        print(f"selftest launcher of {name} {build.launcher}: implementation "
              f"{build.mpi_impl!r} (from --version "
              f"{timing_host.mpi_implementation(version)!r}); "
              f"{(version.strip().splitlines() or ['unreadable'])[0]}")
    impl = config.builds["reference"].mpi_impl if config.launcher_kind != "srun" else "srun"
    for mode in ("bound", "unbound", "over", "bound-hwthread", "bound-pe"):
        try:
            flags, label = timing_host.binding_flags(
                impl, mode, 2, config.platform, config.explicit_flags,
                ranks=config.physical, **config.flag_values())
            print(f"selftest binding {mode} -> {label}: {flags}")
        except SystemExit as error:
            print(f"selftest binding {mode} -> {error}")
    for name, build in config.builds.items():
        for path, what in ((build.record, "build record"), (build.config_log, "config.log")):
            if path:
                print(f"selftest build {name} {what}: "
                      f"{'present' if os.path.isfile(path) else 'MISSING'} {path}")
        if build.unrecorded:
            print(f"selftest build {name}: build flags unrecorded")
    if config.routes:
        found, text = backend_probe(config) if config.gpu != "auto" else (None, "")
        print(f"selftest GPU route the package resolves under the GPU rows' environment: "
              f"{found!r}" + ("" if found else f" ({text.strip()[-300:]!r})"))
    if config.gpu not in (None, "auto") or config.platform == "darwin":
        print(f"selftest gpu state: "
              f"{json.dumps(timing_host.gpu_state(config.gpu, config.platform))}")
    for problem in problems:
        print(f"selftest WOULD REFUSE: {problem}")
    for kind in ("gpu", "meep"):
        ok, observed = timing_host.quiet(config.gate, kind)
        print(f"selftest quiet({kind})={'yes' if ok else 'no'} cpu={observed.get('cpu')} "
              f"metric={observed.get('metric')} reasons={observed.get('reasons')}")
    print(f"selftest box: {timing_host.box_string(config.gate)}")
    if launcher_probe_too:
        failures = launcher_probe(config)
        print(f"selftest launcher-probe: {len(failures)} failure(s)")
        if failures:
            return EXIT_REFUSED
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = build_config(args)
        plan = build_plan(config)
    except SystemExit as error:
        print(str(error), file=sys.stderr, flush=True)
        return error.code if isinstance(error.code, int) else EXIT_REFUSED
    if args.dry_run:
        record = plan_record(config, plan)
        if args.json:
            print(json.dumps(record, indent=1, sort_keys=True, default=str))
        else:
            print_plan(record, lambda message: print(f"{utc()} [dry-run] {message}",
                                                     flush=True))
        return 0
    if args.selftest:
        return selftest(config, args.launcher_probe)
    apply_process_environment()
    try:
        live_checks(config, args.resume)
    except SystemExit as error:
        print(str(error), file=sys.stderr, flush=True)
        nice_refusal = "runs at nice" in str(error)
        return EXIT_NICE if nice_refusal else EXIT_REFUSED
    ladder = Ladder(config, plan, args.resume)
    try:
        return ladder.run()
    except SystemExit as error:
        return int(error.code) if isinstance(error.code, int) else EXIT_REFUSED


if __name__ == "__main__":
    raise SystemExit(main())
