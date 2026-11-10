#!/usr/bin/env python3
"""Check a Linux host before a certification round's gates, and change nothing on it.

Run it from the clone, with the reference environment active, on the machine (or inside
the batch job) that will run the gates (``docs/development/certify-a-new-gpu.md``,
"Preflight")::

    python parity/meep_gpu/rounds/1.0/preflight.py --capability 9.0 --expect "$EXPECT" \\
        2>&1 | tee "$ROUND_DIR/preflight.txt"

Every line starts with PASS, FAIL, WARN or INFO, and the last reads ``PREFLIGHT OK (0
FAIL)`` or ``PREFLIGHT REFUSED (<n> FAIL)``; the exit status is 1 when any line is FAIL. A
WARN does not stop the gates; it names something the round's later stages, or the person
reading the results, need to know.

What it checks, in order:

* the platform: Linux on x86_64, glibc 2.28 or later; whether the host is WSL2 (the
  WSL2-only checks below run only there) and whether this process is inside a batch job;
* where the work lives: the clone on a Linux file system (under WSL2, never under
  ``/mnt``), the free disk under it (``--disk-gib``, 40 GiB by default: what the round still
  needs once the environment is built) and the memory this process may use
  (``--memory-gib``, by default 24 GiB for one card and 12 GiB more for each further card);
* the tools the round calls, and a shell that exports none of the variables the round
  sets itself (``MEEP_GPU_*``, ``TRIDENT_FDTD_*``, ``CUPY_*``, ``TRITON_*``);
* the commit: ``--expect`` is a full commit hash, the clone's HEAD is that commit and its
  tracked files are unchanged;
* the reference environment: the interpreter running this check is one
  ``build_meep_reference_linux.sh`` built (its ``share/meep-gpu-build/reference_build.json``),
  and the link to the driver's ``libcuda.so.1`` that Triton's launcher is compiled against
  exists (the build makes it only on a host with an NVIDIA driver);
* the cards: the ones the round will run on (``--gpu-ids``, else the cards this process
  may use, read as :func:`card_ids` says), each of compute capability ``--capability``,
  none partitioned into MIG slices; the CUDA version the driver supplies, against the one
  ``--capability`` was certified on, read off the clone's Triton ledger (its certified NVIDIA
  stack, ``certified_nvidia_stack``, which lists every driver a capability's certified runs
  used; ``12030``, CUDA 12.3, for 8.6): the gates run on any driver, the bind records the
  one they run beside (for a certified capability, beside its listed ones, when the lanes
  read its recorded value on every other axis), and the route stage's admitted-policy legs
  run only beside a listed one; under WSL2, that the Windows driver supplies CUDA and no Linux
  driver package is installed;
* idle, by the round's own card rule: ``display_card.py`` beside this file, run once per
  card, all cards at once, reading each twice ``--poll`` seconds apart as a lane does. A
  card that drives no display must read idle by the idle rule; a card that drives a display
  (or any card of a WSL2 host) by the display rule. A card refused here is refused by the
  round;
* under WSL2, the Windows display watchdog's ``TdrDelay`` (read only, through Windows
  interop).

``--cards-only`` prints the cards alone (``0,1,2,3``, as :func:`card_ids` reads them) and
checks nothing, for a batch script to hand to ``run_chunks.py --gpu-ids``; it exits 1 when
the cards cannot be read.

It writes nothing and starts nothing on a card: ``nvidia-smi`` reads and the driver's
version call are all it asks of the GPU.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

ROUND_DIR = Path(__file__).resolve().parent
if str(ROUND_DIR) not in sys.path:
    sys.path.insert(0, str(ROUND_DIR))

import display_card  # noqa: E402

#: The checkout this file belongs to (``parity/meep_gpu/rounds/1.0/preflight.py``).
REPO = ROUND_DIR.parents[3]
#: Where the clone records the certified NVIDIA stack of each compute capability (the bind
#: writes it: ``record_nvidia_stack.py``), and the keys inside that ledger.
STACK_LEDGER = ("meep_gpu", "triton_kernels", "fingerprints.json")
STACK_ENTRY, STACK_BY_CAPABILITY = "certified_nvidia_stack", "by_capability"
GLIBC_MIN = (2, 28)
#: Free disk the round still needs under the clone once the reference environment is built:
#: the guide asks for about 60 GiB before the build, which takes 9.1 GB for the environment
#: and 4.8 GB for its package cache.
DISK_GIB = 40
MEMORY_GIB_ONE_CARD = 24
MEMORY_GIB_PER_FURTHER_CARD = 12
#: Prefixes of the variables the round sets for each child itself.
ROUND_PREFIXES = ("MEEP_GPU_", "TRIDENT_FDTD_", "CUPY_", "TRITON_")
#: File systems a WSL2 clone may live on (the Linux disk, never a Windows drive).
LINUX_FILESYSTEMS = ("ext4", "xfs", "btrfs")
WSL_LIB = "/usr/lib/wsl/lib"
BUILD_RECORD = ("share", "meep-gpu-build", "reference_build.json")
LIBCUDA_RECORD = ("share", "meep-gpu-build", "libcuda.json")
LINUX_DRIVER_PACKAGES = ("nvidia-driver", "nvidia-utils", "libnvidia-compute", "cuda-drivers",
                         "nvidia-dkms")


class Report:
    """The lines of the check, printed as they are found."""

    def __init__(self, write: Callable[[str], None] = print) -> None:
        self.write = write
        self.fails = 0
        self.lines: List[str] = []

    def __call__(self, level: str, text: str) -> None:
        line = f"{level:<5} {text}"
        self.lines.append(line)
        if level == "FAIL":
            self.fails += 1
        self.write(line)


class Host:
    """What the check reads off the machine. Tests replace it; nothing here writes."""

    def __init__(self, environ: Optional[Mapping[str, str]] = None) -> None:
        self.environ: Mapping[str, str] = dict(os.environ if environ is None else environ)

    def system(self) -> str:
        return platform.system()

    def machine(self) -> str:
        return platform.machine()

    def glibc(self) -> Optional[str]:
        try:
            text = os.confstr("CS_GNU_LIBC_VERSION") or ""
        except (AttributeError, ValueError, OSError):
            text = ""
        match = re.search(r"(\d+\.\d+)", text)
        return match.group(1) if match else None

    def wsl2(self) -> bool:
        return display_card.wsl2_host()

    def read(self, path: str) -> Optional[str]:
        try:
            return Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

    def free_gib(self, path: Path) -> Optional[float]:
        try:
            stat = os.statvfs(path)
        except OSError:
            return None
        return stat.f_bavail * stat.f_frsize / 2 ** 30

    def cpus(self) -> int:
        try:
            return len(os.sched_getaffinity(0))
        except AttributeError:
            return os.cpu_count() or 1

    def which(self, name: str) -> Optional[str]:
        return shutil.which(name, path=os.pathsep.join(
            [self.environ.get("PATH", ""), "/sbin", "/usr/sbin"]))

    def run(self, argv: Sequence[str], cwd: Optional[Path] = None) -> Tuple[int, str]:
        """``(exit status, stdout)``; a program that cannot run answers ``(127, why)``."""
        env = {k: v for k, v in self.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
        try:
            done = subprocess.run(list(argv), capture_output=True, text=True, timeout=120,
                                  cwd=cwd, env=env)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return 127, f"{argv[0]} unavailable: {exc}"
        return done.returncode, done.stdout

    def driver_version(self) -> Optional[int]:
        """cuDriverGetVersion through the driver's own library (12030 is CUDA 12.3)."""
        try:
            lib = ctypes.CDLL("libcuda.so.1")
        except OSError:
            return None
        value = ctypes.c_int(0)
        try:
            status = lib.cuDriverGetVersion(ctypes.byref(value))
        except AttributeError:
            return None
        return int(value.value) if status == 0 else None

    def prefix(self) -> Path:
        return Path(sys.prefix)

    def idle_reports(self, cards: Sequence[int], poll: int) -> Dict[int, Tuple[int, str]]:
        """``display_card.py --gpu <card> --poll <poll>`` for every card at once."""
        script = ROUND_DIR / "display_card.py"
        running = {card: subprocess.Popen(
            [sys.executable, "-s", str(script), "--gpu", str(card), "--poll", str(poll)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for card in cards}
        out = {}
        for card, process in running.items():
            text, _ = process.communicate()
            out[card] = (process.returncode, text or "")
        return out


# ------------------------------------------------------------------------- rules ----
def version_at_least(text: Optional[str], minimum: Tuple[int, int]) -> bool:
    match = re.match(r"(\d+)\.(\d+)", text or "")
    return bool(match) and (int(match.group(1)), int(match.group(2))) >= minimum


def mount_type(path: Path, mounts: Optional[str]) -> Optional[str]:
    """The file system type of the longest mount point holding ``path`` (``/proc/mounts``)."""
    best, kind = -1, None
    target = str(path)
    for line in (mounts or "").splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        point = fields[1].replace("\\040", " ")
        inside = target == point or target.startswith(point.rstrip("/") + "/") or point == "/"
        if inside and len(point) > best:
            best, kind = len(point), fields[2]
    return kind


def memory_gib(meminfo: Optional[str], cgroup_limits: Sequence[Optional[str]]) -> Optional[float]:
    """The memory this process may use: MemTotal, or a smaller cgroup limit (a batch job's)."""
    total = None
    match = re.search(r"^MemTotal:\s+(\d+)\s+kB", meminfo or "", re.M)
    if match:
        total = int(match.group(1)) * 1024
    for text in cgroup_limits:
        value = (text or "").strip()
        if value.isdigit() and int(value) > 0 and (total is None or int(value) < total):
            total = int(value)
    return None if total is None else total / 2 ** 30


def cgroup_memory_files(proc_cgroup: Optional[str]) -> List[str]:
    """The memory limit files of this process's cgroup, v2 and v1."""
    files = []
    for line in (proc_cgroup or "").splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        hierarchy, controllers, path = parts
        if hierarchy == "0" and controllers == "":
            files.append(f"/sys/fs/cgroup{path.rstrip('/')}/memory.max")
        elif "memory" in controllers.split(","):
            files.append(f"/sys/fs/cgroup/memory{path.rstrip('/')}/memory.limit_in_bytes")
    return files


def memory_needed_gib(cards: int) -> int:
    return MEMORY_GIB_ONE_CARD + MEMORY_GIB_PER_FURTHER_CARD * max(cards - 1, 0)


def export_problems(environ: Mapping[str, str]) -> List[str]:
    return sorted(name for name in environ if name.startswith(ROUND_PREFIXES))


def parse_listing(text: str) -> List[Dict[str, str]]:
    """``nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version`` rows."""
    rows = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5 and parts[0].isdigit():
            rows.append({"index": parts[0], "name": parts[1], "compute_cap": parts[2],
                         "memory_total": parts[3], "driver_version": parts[4]})
    return rows


def card_ids(listed: Sequence[int], visible: Optional[str],
             asked: Optional[str]) -> Tuple[List[int], str]:
    """The cards the round runs on, as the indices ``nvidia-smi`` lists, and how they were read.

    ``asked`` (``--gpu-ids``) wins. Otherwise ``CUDA_VISIBLE_DEVICES`` (``visible``) decides:
    unset, every card listed; a batch scheduler that confines the job to its cards sets it
    and ``nvidia-smi`` then lists only those (as many as it names), so the listed indices
    are the cards; one that does not confine it lists every card of the node and names the
    job's own by their indices. A mask naming devices by UUID, or one that matches neither
    reading, is refused: pass ``--gpu-ids``."""
    if asked:
        ids = [int(x) for x in asked.split(",") if x.strip()]
        missing = [i for i in ids if i not in listed]
        if missing:
            raise ValueError(f"--gpu-ids names {missing}, which nvidia-smi does not list "
                             f"(it lists {list(listed)})")
        return ids, "--gpu-ids"
    if visible is None:
        return list(listed), "every card nvidia-smi lists (CUDA_VISIBLE_DEVICES unset)"
    entries = [x.strip() for x in visible.split(",") if x.strip()]
    if not entries:
        raise ValueError("CUDA_VISIBLE_DEVICES is empty: it hides every card")
    if not all(x.isdigit() for x in entries):
        raise ValueError(f"CUDA_VISIBLE_DEVICES={visible} names devices by UUID or MIG "
                         f"name: pass --gpu-ids with the indices nvidia-smi lists")
    named = [int(x) for x in entries]
    if len(named) == len(listed):
        return list(listed), ("the cards this job may use (nvidia-smi lists as many as "
                              "CUDA_VISIBLE_DEVICES names)")
    if all(i in listed for i in named):
        return named, ("CUDA_VISIBLE_DEVICES (nvidia-smi lists every card of the node; the "
                       "job's are the ones it names)")
    raise ValueError(f"CUDA_VISIBLE_DEVICES={visible} does not match the cards nvidia-smi "
                     f"lists ({list(listed)}): pass --gpu-ids")


def certified_drivers(repo: Path, capability: str) -> Optional[Tuple[int, ...]]:
    """The CUDA driver versions ``capability`` was certified on, read off the clone's Triton
    ledger; None when it records no certified stack for it (a capability certified for the
    first time) or the ledger cannot be read."""
    try:
        ledger = json.loads(Path(repo).joinpath(*STACK_LEDGER).read_text(encoding="utf-8"))
        record = ledger[STACK_ENTRY][STACK_BY_CAPABILITY][capability]
        drivers = tuple(int(value) for value in record["stack"]["cuda_driver"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return drivers or None


def driver_line(version: Optional[int], capability: str,
                certified: Optional[Tuple[int, ...]]) -> Tuple[str, str]:
    """The driver's line, judged against ``certified``: the drivers ``capability`` was
    certified on (None when it has no certified stack yet)."""
    if version is None:
        return "WARN", ("the driver's CUDA version is unreadable (libcuda.so.1 did not answer "
                        "cuDriverGetVersion)")
    spelled = f"CUDA {version // 1000}.{(version % 1000) // 10} (driver version {version})"
    if certified is None:
        return "PASS", (f"the driver supplies {spelled}; compute capability {capability} has "
                        f"no certified stack yet, so the bind records the one its gates run "
                        f"beside, and its route stage runs beside the same driver")
    named = ", ".join(str(value) for value in certified)
    if version in certified:
        return "PASS", (f"the driver supplies {spelled}, one compute capability "
                        f"{capability} was certified on ({named})")
    return "WARN", (f"the driver supplies {spelled}, not one compute capability "
                    f"{capability} was certified on ({named}): the gates run on it, and the "
                    f"bind adds it to the capability's certified drivers only if every other "
                    f"axis of the stack its lanes read matches the record; until then the "
                    f"route stage's admitted-policy legs refuse every case beside it (the "
                    f"guide, 'The CUDA driver the route stage needs')")


def idle_decision(rc: int, report: str) -> Tuple[str, str, List[str]]:
    """``(level, decision, detail lines)`` from one ``display_card.py`` report."""
    lines = [line for line in report.splitlines() if line.strip()]
    detail = [line for line in lines if not line.startswith("DECISION")]
    decisions = [line for line in lines if line.startswith("DECISION")]
    if not decisions:
        return "FAIL", (f"the card rule did not decide (exit {rc}): "
                        f"{lines[-1] if lines else 'no output'}"), detail
    decision = decisions[-1][len("DECISION: "):] if decisions[-1].startswith("DECISION: ") \
        else decisions[-1]
    if rc == 0:
        return "PASS", decision, detail
    return "FAIL", decision + " (the LANE and FLEET lines above say why)", detail


# ------------------------------------------------------------------------ checks ----
def check_platform(host: Host, say: Report) -> bool:
    system, machine = host.system(), host.machine()
    if system != "Linux":
        say("FAIL", f"system {system}: this check is for Linux x86_64 hosts (an Apple GPU "
                    f"has its own steps in the guide)")
    if machine != "x86_64":
        say("FAIL", f"machine {machine}: x86_64 is needed")
    elif system == "Linux":
        say("PASS", "Linux x86_64")
    glibc = host.glibc()
    if version_at_least(glibc, GLIBC_MIN):
        say("PASS", f"glibc {glibc} (2.28 or later is needed)")
    else:
        say("FAIL", f"glibc {glibc or 'unreadable'}: the reference environment needs 2.28 or "
                    f"later")
    wsl2 = host.wsl2()
    say("INFO", "a WSL2 host (the WSL2 checks apply)" if wsl2 else "native Linux")
    release = host.read("/etc/os-release") or ""
    match = re.search(r'^PRETTY_NAME="?([^"\n]*)"?', release, re.M)
    if match:
        say("INFO", f"distribution: {match.group(1)}")
    if host.environ.get("SLURM_JOB_ID"):
        say("INFO", "inside a Slurm job (SLURM_JOB_ID is set)")
    elif host.environ.get("PBS_JOBID"):
        say("INFO", "inside a PBS job (PBS_JOBID is set)")
    return wsl2


def check_disk(host: Host, say: Report, repo: Path, wsl2: bool, disk_gib: int,
               memory_needed: int) -> None:
    where = str(repo)
    if wsl2 and where.startswith("/mnt/"):
        say("FAIL", f"the clone is under /mnt ({where}): under WSL2 keep it on the Linux "
                    f"disk, under your Linux home directory")
    kind = mount_type(repo, host.read("/proc/mounts"))
    if wsl2:
        if kind in LINUX_FILESYSTEMS:
            say("PASS", f"the clone is on {kind}")
        else:
            say("FAIL", f"the clone is on '{kind or 'unknown'}': under WSL2 clone and build on "
                        f"the Linux disk (ext4), never under /mnt")
    else:
        say("INFO", f"the clone is on {kind or 'an unreadable file system'}")
    free = host.free_gib(repo)
    if free is None:
        say("FAIL", "the free disk under the clone is unreadable")
    elif free >= disk_gib:
        say("PASS", f"free disk under the clone: {free:.0f} GiB ({disk_gib} GiB asked for)")
    else:
        say("FAIL", f"free disk under the clone: {free:.0f} GiB; after the build the round "
                    f"still needs about {disk_gib} GiB (results, lane logs and compile caches)")
    limits = [host.read(path) for path in cgroup_memory_files(host.read("/proc/self/cgroup"))]
    memory = memory_gib(host.read("/proc/meminfo"), limits)
    if memory is None:
        say("WARN", "the memory this process may use is unreadable")
    elif memory >= memory_needed:
        say("PASS", f"memory this process may use: {memory:.0f} GiB ({memory_needed} GiB "
                    f"asked for)")
    else:
        say("WARN", f"memory this process may use: {memory:.0f} GiB; ask for at least "
                    f"{memory_needed} GiB" + (" (memory= in %UserProfile%\\.wslconfig, then "
                                              "wsl --shutdown)" if wsl2 else
                                              " (a batch job: its memory request)"))
    say("INFO", f"processors this process may use: {host.cpus()}")


def check_tools(host: Host, say: Report) -> None:
    for tool in ("git", "tar", "gzip", "bash", "ldconfig"):
        where = host.which(tool)
        if where:
            say("PASS", f"found {tool}")
        else:
            say("FAIL", f"missing {tool}")


def check_shell(host: Host, say: Report) -> None:
    for name in export_problems(host.environ):
        say("FAIL", f"the shell exports {name}: unset it (the round sets what each gate needs)")
    if host.environ.get("PYTHONPATH"):
        say("WARN", "the shell sets PYTHONPATH: the round's drivers put the clone on the path "
                    "themselves; unset it")


def check_commit(host: Host, say: Report, repo: Path, expect: Optional[str]) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", expect or ""):
        say("FAIL", f"--expect is '{expect or ''}', not a 40-character commit hash: pass the "
                    f"commit the maintainers named")
        return
    rc, head = host.run(["git", "-C", str(repo), "rev-parse", "HEAD"])
    head = head.strip()
    if rc != 0:
        say("FAIL", f"git cannot read the clone's HEAD ({repo})")
    elif head != expect:
        say("FAIL", f"the clone's HEAD is {head}, not {expect}: check out the commit "
                    f"(git checkout --detach {expect})")
    else:
        say("PASS", f"the clone is at {expect}")
    rc, tree = host.run(["git", "-C", str(repo), "rev-parse", "HEAD^{tree}"])
    if rc == 0:
        say("INFO", f"tree {tree.strip()}")
    rc, status = host.run(["git", "-C", str(repo), "status", "--porcelain"])
    if rc != 0:
        say("FAIL", "git status did not run in the clone")
    elif status.strip():
        changed = [line for line in status.splitlines() if line.strip()]
        say("FAIL", f"{len(changed)} changed or untracked path(s) in the clone, e.g. "
                    f"{changed[0].strip()}: nothing tracked may change")
    else:
        say("PASS", "nothing changed in the clone (git status is empty)")


def check_environment(host: Host, say: Report) -> None:
    prefix = host.prefix()
    if not prefix.joinpath(*BUILD_RECORD).is_file():
        say("FAIL", f"this interpreter ({prefix}) is not a reference environment (no "
                    f"{'/'.join(BUILD_RECORD)}): activate the one "
                    f"build_meep_reference_linux.sh built, then run this check again")
        return
    say("PASS", f"the reference environment: {prefix}")
    try:
        record = json.loads(prefix.joinpath(*LIBCUDA_RECORD).read_text(encoding="utf-8"))
        directory = record.get("directory")
    except (OSError, ValueError, AttributeError):
        directory = None
    if directory and prefix.joinpath(directory, "libcuda.so").exists():
        say("PASS", f"the environment links Triton's launcher against the driver "
                    f"({directory}/libcuda.so)")
        return
    driver = driver_libcuda(host)
    if driver and Path(driver).with_name("libcuda.so").exists():
        say("WARN", f"the environment records no link to the driver's libcuda.so.1 (it was built "
                    f"on a host without an NVIDIA driver); Triton's launcher links against "
                    f"{Path(driver).with_name('libcuda.so')}, beside the driver's, instead")
        return
    say("FAIL", "the environment records no link to the driver's libcuda.so.1 (it was built on a "
                "host without an NVIDIA driver), and the driver's directory holds no libcuda.so, "
                "so Triton's launcher cannot link (cannot find -lcuda): on this host, run python "
                "parity/meep_gpu/build_meep_reference_record.py link-libcuda \"$CONDA_PREFIX\" "
                "\"$CONDA_PREFIX/share/meep-gpu-build/libcuda.json\"")


def driver_libcuda(host: Host) -> Optional[str]:
    """The driver's ``libcuda.so.1`` as the dynamic linker's cache names it, or None."""
    rc, cache = host.run([host.which("ldconfig") or "ldconfig", "-p"])
    return next((line.split("=>")[-1].strip() for line in cache.splitlines()
                 if line.strip().startswith("libcuda.so.1 ") and "x86-64" in line), None)


def check_cards(host: Host, say: Report, capability: str, asked: Optional[str], wsl2: bool,
                poll: int, repo: Path = REPO) -> List[int]:
    smi = host.which("nvidia-smi")
    if not smi:
        if wsl2 and Path(WSL_LIB, "nvidia-smi").exists():
            say("FAIL", f"nvidia-smi is in {WSL_LIB} but not on PATH: the round calls it by "
                        f"name; run sudo ln -s {WSL_LIB}/nvidia-smi /usr/local/bin/nvidia-smi")
        else:
            say("FAIL", "no nvidia-smi: no NVIDIA driver is reachable from this shell"
                        + (" (install the driver on Windows, not inside WSL2)" if wsl2 else ""))
        return []
    say("PASS", f"nvidia-smi at {smi}")
    libcuda = driver_libcuda(host)
    if wsl2:
        rc, packages = host.run(["dpkg", "-l"])
        found = [fields[1] for fields in (line.split() for line in packages.splitlines())
                 if len(fields) > 1 and fields[0] == "ii"
                 and fields[1].startswith(LINUX_DRIVER_PACKAGES)]
        if found:
            say("FAIL", f"Linux NVIDIA driver packages inside WSL2 ({' '.join(found)}): remove "
                        f"them; under WSL2 the Windows driver supplies CUDA")
        if libcuda and libcuda.startswith(WSL_LIB + "/"):
            say("PASS", f"libcuda.so.1 in the linker cache: {libcuda}")
        elif libcuda:
            say("WARN", f"libcuda.so.1 resolves to {libcuda}, not {WSL_LIB}: a Linux driver "
                        f"library may be shadowing the Windows one")
        else:
            say("FAIL", f"libcuda.so.1 is not in the linker cache: check that /etc/ld.so.conf.d/"
                        f"ld.wsl.conf names {WSL_LIB}, then sudo ldconfig")
    elif libcuda:
        say("PASS", f"libcuda.so.1 in the linker cache: {libcuda}")
    else:
        say("FAIL", "libcuda.so.1 is not in the linker cache: the NVIDIA driver is not installed "
                    "or not on the linker's path")
    rc, text = host.run(["nvidia-smi", "--query-gpu=index,name,compute_cap,memory.total,"
                         "driver_version", "--format=csv,noheader"])
    rows = parse_listing(text)
    say("INFO", "GPUs: " + ("; ".join(", ".join(row.values()) for row in rows) or "none listed"))
    if not rows:
        say("FAIL", "nvidia-smi lists no GPU")
        return []
    try:
        cards, how = card_ids([int(row["index"]) for row in rows],
                              host.environ.get("CUDA_VISIBLE_DEVICES"), asked)
    except ValueError as refused:
        say("FAIL", str(refused))
        return []
    say("INFO", f"the cards to run on: {','.join(map(str, cards))} ({how}); pass them as "
                f"--gpu-ids")
    by_index = {int(row["index"]): row for row in rows}
    for card in cards:
        row = by_index[card]
        if row["compute_cap"] == capability:
            say("PASS", f"GPU {card} ({row['name']}) compute capability {capability}")
        else:
            say("FAIL", f"GPU {card} ({row['name']}) compute capability '{row['compute_cap']}', "
                        f"not {capability}")
        rc, mig = host.run(["nvidia-smi", f"--id={card}", "--query-gpu=mig.mode.current",
                            "--format=csv,noheader"])
        if mig.strip().lower() == "enabled":
            say("FAIL", f"GPU {card} is partitioned into MIG slices: the round needs a whole card")
    level, text = driver_line(host.driver_version(), capability,
                              certified_drivers(repo, capability))
    say(level, text)
    reports = host.idle_reports(cards, poll)
    for card in cards:
        rc, report = reports.get(card, (1, ""))
        level, decision, detail = idle_decision(rc, report)
        for line in detail:
            say("INFO", f"GPU {card}: {line}")
        say(level, decision)
    return cards


def check_watchdog(host: Host, say: Report) -> None:
    if not host.which("reg.exe"):
        say("INFO", "Windows interop is off: read TdrDelay from Windows instead (the guide, "
                    "'The display watchdog')")
        return
    rc, text = host.run(["reg.exe", "query",
                         "HKLM\\SYSTEM\\CurrentControlSet\\Control\\GraphicsDrivers",
                         "/v", "TdrDelay"])
    value = next((line.split()[-1] for line in text.replace("\r", "").splitlines()
                  if "TdrDelay" in line), None)
    say("INFO", f"Windows TdrDelay: {value or 'not set (the Windows default, 2 s)'}")


def run(args: argparse.Namespace, host: Host, say: Report) -> int:
    say("INFO", f"meep-gpu certification preflight, capability {args.capability}")
    wsl2 = check_platform(host, say)
    asked = args.gpu_ids or None
    count = len([x for x in (asked or "").split(",") if x.strip()]) or 1
    check_disk(host, say, args.repo, wsl2, args.disk_gib,
               args.memory_gib or memory_needed_gib(count))
    check_tools(host, say)
    check_shell(host, say)
    check_commit(host, say, args.repo, args.expect)
    check_environment(host, say)
    check_cards(host, say, args.capability, asked, wsl2, args.poll, args.repo)
    if wsl2:
        check_watchdog(host, say)
    say.write("")
    say.write("PREFLIGHT OK (0 FAIL)" if say.fails == 0
              else f"PREFLIGHT REFUSED ({say.fails} FAIL)")
    return 0 if say.fails == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--capability", required=True,
                    help="the compute capability of the cards, major.minor (9.0)")
    ap.add_argument("--expect", default=None, help="the commit to run, 40 hexadecimal characters")
    ap.add_argument("--gpu-ids", default="",
                    help="the cards to run on, as nvidia-smi indexes them (default: read from "
                         "CUDA_VISIBLE_DEVICES and nvidia-smi)")
    ap.add_argument("--poll", type=int, default=60,
                    help="seconds between the two idle readings (a lane's default)")
    ap.add_argument("--disk-gib", type=int, default=DISK_GIB)
    ap.add_argument("--memory-gib", type=int, default=0,
                    help="default: 24 GiB for one card and 12 GiB more for each further card")
    ap.add_argument("--cards-only", action="store_true",
                    help="print the cards to run on (0,1,...) and check nothing")
    ap.add_argument("--repo", type=Path, default=REPO, help=argparse.SUPPRESS)
    return ap


def cards_only(args: argparse.Namespace, host: Host, write: Callable[[str], None]) -> int:
    rc, text = host.run(["nvidia-smi", "--query-gpu=index,name,compute_cap,memory.total,"
                         "driver_version", "--format=csv,noheader"])
    listed = [int(row["index"]) for row in parse_listing(text)]
    if not listed:
        write("no GPU listed by nvidia-smi")
        return 1
    try:
        cards, _how = card_ids(listed, host.environ.get("CUDA_VISIBLE_DEVICES"),
                               args.gpu_ids or None)
    except ValueError as refused:
        write(str(refused))
        return 1
    write(",".join(map(str, cards)))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cards_only:
        return cards_only(args, Host(), lambda line: print(line, flush=True))
    # Every line is written neutrally (no host name, account or local path): the report is
    # kept in ROUND_DIR and goes back with the results (pack_results.py).
    import round_status  # noqa: PLC0415 - beside this file; it loads neutral_ids.py

    neutral = round_status.neutral_ids()
    say = Report(lambda line: print(neutral.text(line), flush=True))
    return run(args, Host(), say)


if __name__ == "__main__":
    sys.exit(main())
