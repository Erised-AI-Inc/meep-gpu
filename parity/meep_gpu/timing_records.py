#!/usr/bin/env python
"""Recorders of the timing ladder. Standard library only.

Every recorder reads a row file defensively, extracts a handful of scalars, appends ONE
JSON line to the ladder's ledger and returns ONE summary line for the progress log. A
row that is missing or unparseable is recorded as such, by name; a recorder never
raises past :func:`main` and never stops the ladder.

Subcommands (``timing_ladder.py`` calls the same functions in process):

  manifest           sha256 of every file under a root -> a listing and its digest
  environment        the host, the MEEP builds, the packages and the knobs, as JSON
  record-gpu         one bench_fused_products row -> ledger (verdict, legs, what
                     launched, bit identity, the deposit-repair route, lift seconds)
  record-meep        one bench_meep_identical_case measurement row -> ledger
  record-control     the array-path control read off GPU rows' own array legs -> ledger
  record-note        a row that produced no measurement (timeout, refusal) -> ledger
  decide-binding     the binding probe's rule, applied to its rows
  meep-build-record  what a MEEP build is: version, precision, flags, libraries

Ported from the 2026-09-28 ladder's ``ladder_tools.py``. What changed, each by name:
``record-gpu`` read ``box_before.foreign_on_pinned_device``, a key no row carries, so the
field was always empty; it now reads ``foreign_compute_apps_on_pinned_gpu`` joined on the
pinned device's UUID. ``manifest --code-only`` hashed two fixed development-layout
prefixes and, on a release-layout root, hashed nothing and returned the digest of an
empty listing; it now finds the layout by name and refuses a root holding neither.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import timing_host  # noqa: E402

SKIP_DIRECTORIES = {"__pycache__", ".git"}
LIFT_LINE = re.compile(r"(\w+)/(\w+): lifted \(([^)]*)\) \(([0-9.]+) s\)")

#: The code a timing row can run, per layout: (package prefix, harness prefix).
LAYOUTS = {
    "release": ("meep_gpu/", "parity/meep_gpu/"),
    "development": ("apps/api/meep_gpu/", "apps/api/parity/meep_gpu/"),
}

#: The slots each seam's deposit-repair bracket occupies (leading, trailing).
REPAIR_SLOTS = {"B": ("step_B", "update_H"), "D": ("step_D", "update_E")}

#: The 2026-09-28 MEEP configuration: what a row that records none of its own ran.
REFERENCE_CONFIGURATION = {"build": "reference", "threads_per_rank": 1,
                           "split_chunks_evenly": "default", "stepper": "fields_step"}


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def parse_fields(pairs: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        parsed: Any = value
        for cast in (int, float):
            try:
                parsed = cast(value)
                break
            except ValueError:
                continue
        out[key] = parsed
    return out


def append(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def last_row(path: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """The last parseable JSON line of a rows file, or why there is none."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            lines = [line for line in handle if line.strip()]
    except OSError as error:
        return None, f"{type(error).__name__}: {error}"
    if not lines:
        return None, "the rows file is empty"
    try:
        return json.loads(lines[-1]), None
    except ValueError as error:
        return None, f"the last line does not parse: {error}"


def rows_of(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass
    return rows


def cells_of(shape: Optional[Sequence[int]]) -> Optional[int]:
    if not shape:
        return None
    cells = 1
    for count in shape:
        cells *= int(count)
    return cells


def rate(cells: Optional[int], seconds_per_step: Optional[float]) -> Optional[float]:
    if not cells or not seconds_per_step:
        return None
    return cells / seconds_per_step / 1e6


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------

def layout_of(root: str) -> Optional[str]:
    """``release`` or ``development``, found by name, or ``None``."""
    root = os.path.abspath(root)
    for name, (package, harness) in LAYOUTS.items():
        if (os.path.isdir(os.path.join(root, package.rstrip("/")))
                and os.path.isdir(os.path.join(root, harness.rstrip("/")))):
            return name
    return None


def manifest(root: str, code_only: bool,
             exclude: Sequence[str] = (),
             exclude_names: Sequence[str] = ()) -> Tuple[List[str], str]:
    """``sha256  relative/path`` for every file under ``root``, and the listing's digest.

    ``code_only`` walks the package and the harness of the root's layout and keeps every
    ``.py``/``.json`` of the package and every ``.py`` of the harness. A root holding
    neither layout is REFUSED (SystemExit) rather than hashed to an empty listing.
    ``exclude`` names relative paths left out; ``exclude_names`` names (``fnmatch``
    patterns) of files and directories left out wherever they sit -- the same two
    lists a snapshot copies by, so a snapshot and its source hash alike.
    """
    import fnmatch  # noqa: PLC0415

    def named(name: str) -> bool:
        return any(fnmatch.fnmatch(name, pattern) for pattern in exclude_names)

    lines: List[str] = []
    root = os.path.abspath(root)
    prefixes: Tuple[str, str] = ("", "")
    if code_only:
        layout = layout_of(root)
        if layout is None:
            raise SystemExit(f"REFUSING: {root} holds neither meep_gpu/ + parity/meep_gpu/ "
                             "nor apps/api/meep_gpu/ + apps/api/parity/meep_gpu/; a code "
                             "digest of it would be the digest of nothing")
        prefixes = LAYOUTS[layout]
        starts = [os.path.join(root, prefix.rstrip("/")) for prefix in prefixes]
    else:
        starts = [root]
    excluded = [e.strip("/") for e in exclude if e]
    for start in starts:
        if not os.path.isdir(start):
            continue
        for directory, names, files in os.walk(start):
            names[:] = sorted(n for n in names if n not in SKIP_DIRECTORIES
                              and not named(n))
            relative_dir = os.path.relpath(directory, root)
            if any(relative_dir == e or relative_dir.startswith(e + os.sep)
                   for e in excluded):
                names[:] = []
                continue
            for name in sorted(files):
                if name.endswith(".pyc") or named(name):
                    continue
                path = os.path.join(directory, name)
                relative = os.path.relpath(path, root)
                if code_only and not (
                        (relative.startswith(prefixes[0])
                         and relative.endswith((".py", ".json")))
                        or (relative.startswith(prefixes[1])
                            and relative.endswith(".py"))):
                    continue
                if os.path.islink(path) and not os.path.exists(path):
                    continue
                lines.append(f"{sha256_file(path)}  {relative}")
    lines = sorted(set(lines), key=lambda line: line.split("  ", 1)[1])
    digest = hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()
    return lines, digest


def command_manifest(args: Any) -> int:
    lines, digest = manifest(args.root, args.code_only, args.exclude or ())
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
    print(f"{digest} {len(lines)}")
    return 0


# ---------------------------------------------------------------------------
# environment and MEEP builds
# ---------------------------------------------------------------------------

PACKAGE_PROBE = r"""
import json, sys, os
out = {"python": sys.version.split()[0], "executable": sys.executable}
try:
    import meep as mp
    out["meep"] = mp.__version__
    out["meep_single_precision"] = bool(mp.is_single_precision())
    out["meep_with_mpi"] = bool(mp.with_mpi())
    out["meep_file"] = mp.__file__
    out["meep_extension"] = getattr(getattr(mp, "_meep", None), "__file__", None)
except Exception as error:
    out["meep"] = "unavailable: %s" % error
for name in ("numpy", "cupy", "triton", "mpi4py", "torch"):
    try:
        module = __import__(name)
        out[name] = getattr(module, "__version__", "present")
    except Exception as error:
        out[name] = "unavailable: %s" % type(error).__name__
print("PACKAGES " + json.dumps(out))
"""


def probe_packages(python: str, timeout: float = 300.0,
                   env: Optional[Dict[str, str]] = None,
                   cwd: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], str]:
    """The package versions, read in ``env`` (the GPU rows' environment, from the
    ladder: on macOS that is the environment in which ``torch`` imports beside MEEP)."""
    quiet = dict(os.environ if env is None else env)
    quiet["CUDA_VISIBLE_DEVICES"] = ""
    text = timing_host.run_text([python, "-c", PACKAGE_PROBE], timeout=timeout,
                                env=quiet, cwd=cwd) or ""
    for line in text.splitlines():
        if line.startswith("PACKAGES "):
            try:
                return json.loads(line[len("PACKAGES "):]), text
            except ValueError:
                break
    return None, text


def configure_line(config_log: Optional[str]) -> Optional[str]:
    if not config_log:
        return None
    try:
        with open(config_log, "r", encoding="utf-8", errors="replace") as handle:
            for _ in range(40):
                line = handle.readline()
                if not line:
                    break
                if line.lstrip().startswith("$ "):
                    return line.strip()[2:]
    except OSError:
        return None
    return None


def makefile_flags(config_log: Optional[str]) -> Dict[str, str]:
    flags: Dict[str, str] = {}
    if not config_log:
        return flags
    makefile = os.path.join(os.path.dirname(config_log), "Makefile")
    try:
        with open(makefile, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                for name in ("CXXFLAGS", "CFLAGS", "CXX", "CPPFLAGS", "LDFLAGS"):
                    if line.startswith(name + " ="):
                        flags[name] = line.split("=", 1)[1].strip()
    except OSError:
        pass
    if not flags:
        # A build directory that is gone leaves config.log alone; its output variables
        # carry the same values (``CXXFLAGS='-O2 ...'``).
        try:
            with open(config_log, "r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    match = re.match(r"^(CXXFLAGS|CFLAGS|CXX|CPPFLAGS|LDFLAGS)='(.*)'$",
                                     line.rstrip("\n"))
                    if match and match.group(1) not in flags:
                        flags[match.group(1)] = match.group(2)
        except OSError:
            pass
    return flags


OPENMP_RUNTIMES = ("libgomp", "libomp", "libiomp5")


def elf_needed(path: str) -> Optional[List[str]]:
    """The DIRECT ``DT_NEEDED`` entries of a 64-bit little-endian ELF file, read from its
    dynamic section; ``None`` when the file is not one (or cannot be read).

    ``ldd`` lists the TRANSITIVE closure, so a MEEP built without OpenMP that links an
    OpenBLAS built with it shows ``libgomp`` there; only the direct entries say what
    MEEP itself was built with.
    """
    import struct  # noqa: PLC0415
    try:
        with open(path, "rb") as handle:
            data = handle.read()
    except OSError:
        return None
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        return None
    shoff, = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum = struct.unpack_from("<HH", data, 0x3A)
    sections = []
    for index in range(shnum):
        base = shoff + index * shentsize
        if base + 64 > len(data):
            return None
        sh_type, = struct.unpack_from("<I", data, base + 4)
        sh_offset, sh_size = struct.unpack_from("<QQ", data, base + 0x18)
        sh_link, = struct.unpack_from("<I", data, base + 0x28)
        sections.append((sh_type, sh_offset, sh_size, sh_link))
    needed: List[str] = []
    for sh_type, offset, size, link in sections:
        if sh_type != 6 or link >= len(sections):  # SHT_DYNAMIC
            continue
        strtab = sections[link][1]
        for entry in range(offset, offset + size, 16):
            tag, value = struct.unpack_from("<qQ", data, entry)
            if tag == 0:
                break
            if tag == 1:  # DT_NEEDED
                end = data.index(b"\x00", strtab + value)
                needed.append(data[strtab + value:end].decode("utf-8", "replace"))
    return needed


def linked_libraries(path: str) -> Tuple[List[str], str]:
    """(the libraries ``path`` links DIRECTLY, how they were read).

    ``otool -L`` on macOS (direct by construction); the ELF dynamic section on Linux;
    ``ldd`` only as a last resort, labelled ``transitive``.
    """
    if sys.platform == "darwin":
        text = timing_host.run_text(["otool", "-L", path], timeout=30) or ""
        return [line.strip().split(" ")[0] for line in text.splitlines()[1:]
                if line.strip()], "otool -L (direct)"
    needed = elf_needed(path)
    if needed is not None:
        return needed, "ELF DT_NEEDED (direct)"
    text = timing_host.run_text(["ldd", path], timeout=30) or ""
    return [line.strip().split(" ")[0] for line in text.splitlines() if line.strip()], \
        "ldd (transitive)"


def linked_openmp(paths: Sequence[str]) -> Dict[str, Any]:
    """Which OpenMP runtime each library links DIRECTLY (:func:`linked_libraries`)."""
    out: Dict[str, Any] = {}
    for path in paths:
        libraries, how = linked_libraries(path)
        out[path] = {"runtimes": sorted({name for name in OPENMP_RUNTIMES
                                         if any(name in lib for lib in libraries)}),
                     "how": how}
    return out


#: Run in the build's interpreter: import MEEP, then list the images the process
#: ACTUALLY loaded (``/proc/self/maps`` on Linux, the dyld image list on macOS), so a
#: build record fingerprints the MEEP libraries that ran, not every ``libmeep*`` an
#: environment happens to hold.
BUILD_PROBE = r"""
import json, glob, os, sys
import meep as mp
loaded = set()
try:
    with open("/proc/self/maps") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) >= 6 and parts[5].startswith("/"):
                loaded.add(parts[5])
    how = "/proc/self/maps"
except OSError:
    import ctypes
    dyld = ctypes.CDLL(None)
    dyld._dyld_image_count.restype = ctypes.c_uint32
    dyld._dyld_get_image_name.restype = ctypes.c_char_p
    dyld._dyld_get_image_name.argtypes = [ctypes.c_uint32]
    for index in range(dyld._dyld_image_count()):
        name = dyld._dyld_get_image_name(index)
        if name:
            loaded.add(name.decode("utf-8", "replace"))
    how = "dyld image list"
here = os.path.dirname(mp.__file__)
meep_images = sorted(os.path.realpath(p) for p in loaded
                     if os.path.basename(p).startswith(("_meep", "libmeep")))
print("BUILD " + json.dumps({
    "meep": mp.__version__, "single_precision": bool(mp.is_single_precision()),
    "with_mpi": bool(mp.with_mpi()), "meep_file": mp.__file__,
    "loaded_meep_images": sorted(set(meep_images)), "loaded_by": how,
    "extensions_present": sorted(glob.glob(os.path.join(here, "_meep*.so"))),
    "libmeep_present": sorted(glob.glob(os.path.join(sys.prefix, "lib", "libmeep*"))),
    "openmp_runtimes_loaded": sorted({os.path.basename(p) for p in loaded
                                      if any(n in os.path.basename(p) for n in
                                             ("libgomp", "libomp", "libiomp5"))}),
    "prefix": sys.prefix}))
"""


def meep_build_record(python: str, config_log: Optional[str] = None,
                      launcher: Optional[str] = None) -> Dict[str, Any]:
    """What a MEEP build IS, read off the interpreter that imports it."""
    record: Dict[str, Any] = {"utc": utc(), "python": python, "launcher": launcher,
                              "config_log": config_log}
    quiet = dict(os.environ)
    quiet["CUDA_VISIBLE_DEVICES"] = ""
    text = timing_host.run_text([python, "-c", BUILD_PROBE], timeout=300, env=quiet) or ""
    facts = None
    for line in text.splitlines():
        if line.startswith("BUILD "):
            try:
                facts = json.loads(line[len("BUILD "):])
            except ValueError:
                facts = None
    record["probe"] = facts
    if facts is None:
        record["probe_output"] = text[-2000:]
        return record
    # ONLY THE IMAGES THE PROCESS LOADED: an environment can hold a stale libmeep.33
    # beside the libmeep.37 that runs, and a record that hashed both would describe
    # a mix of two builds.
    libraries = [p for p in facts.get("loaded_meep_images", []) if os.path.isfile(p)]
    record["sha256"] = {path: sha256_file(path) for path in libraries}
    record["openmp_linked"] = linked_openmp(libraries)
    # MEEP IS BUILT WITH OPENMP when one of ITS OWN images links a runtime directly; a
    # runtime another library brought into the process (OpenBLAS) does not count.
    record["openmp"] = (any(entry["runtimes"] for entry in record["openmp_linked"].values())
                        if libraries else None)
    record["openmp_runtimes_in_process"] = facts.get("openmp_runtimes_loaded")
    record["configure_line"] = configure_line(config_log)
    record["makefile_flags"] = makefile_flags(config_log)
    if launcher:
        record["launcher_version"] = (timing_host.run_text([launcher, "--version"]) or "")[:400]
        record["mpi_implementation"] = timing_host.mpi_implementation(
            record["launcher_version"])
    return record


def command_meep_build_record(args: Any) -> int:
    record = meep_build_record(args.python, args.config_log, args.launcher)
    for path in args.extra_json or []:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                record.setdefault("build", {}).update(json.load(handle))
        except (OSError, ValueError) as error:
            record.setdefault("build_unread", []).append(f"{path}: {error}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True, default=str)
    probe = record.get("probe") or {}
    print(f"MEEP build record: meep {probe.get('meep')} single "
          f"{probe.get('single_precision')} openmp {record.get('openmp')} -> {args.out}")
    if args.require_single and probe.get("single_precision") is not True:
        print("REFUSING: the MEEP build is not single precision")
        return 3
    return 0


def environment_record(out: str, *, tree: Optional[str], drivers: Optional[str],
                       python: Optional[str], builds: Dict[str, Dict[str, Any]],
                       gpu: Optional[str], params: Dict[str, Any],
                       topology: Optional[Dict[str, Any]] = None,
                       platform: Optional[str] = None,
                       package_env: Optional[Dict[str, str]] = None,
                       package_cwd: Optional[str] = None) -> Dict[str, Any]:
    """The host, every MEEP build, the packages and every resolved parameter.

    ``package_env`` is the environment the package versions are probed in: the GPU
    rows' own (on macOS ``torch`` imports beside MEEP only there)."""
    platform = platform or sys.platform
    record: Dict[str, Any] = {"utc": utc(), "params": params, "gpu": gpu,
                              "nice": os.nice(0), "platform": platform,
                              "kmp_duplicate_lib_ok_inherited":
                                  os.environ.get("KMP_DUPLICATE_LIB_OK")}
    record["uname"] = timing_host.run_text(["uname", "-a"])
    record["topology"] = topology or timing_host.topology(platform)
    record["slurm"] = timing_host.slurm_facts()
    record["cpu_state"] = timing_host.cpu_state(platform)
    if platform == "darwin":
        record["sysctl"] = timing_host.darwin_sysctl()
        record["apple_gpu"] = timing_host.apple_gpu_state()
        record["darwin_process_priority"] = timing_host.darwin_process_priority()
    else:
        record["lscpu"] = timing_host.run_text(["lscpu"])
        record["free_g"] = timing_host.run_text(["free", "-g"])
        record["loadavg"] = timing_host.read_file("/proc/loadavg")
    if timing_host.nvidia_smi_available():
        record["nvidia_smi_gpus"] = timing_host.run_text(
            ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,"
             "memory.used,utilization.gpu,compute_cap,persistence_mode,ecc.mode.current",
             "--format=csv,noheader"])
        record["nvidia_smi_apps"] = timing_host.run_text(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
             "--format=csv,noheader"])
        if gpu is not None:
            record["nvidia_smi_clock_performance"] = timing_host.run_text(
                ["nvidia-smi", "-i", str(gpu), "-q", "-d", "CLOCK,PERFORMANCE"])
            record["gpu_state"] = timing_host.gpu_state(gpu)
    if python:
        packages, text = probe_packages(python, env=package_env, cwd=package_cwd)
        record["packages_probed_in"] = ("the GPU rows' environment" if package_env
                                        else "the ladder's environment")
        record["packages"] = packages
        if packages is None:
            record["packages_probe_output"] = text[-2000:]
    record["meep_builds"] = builds
    if drivers:
        record["drivers"] = {
            name: sha256_file(os.path.join(drivers, name))
            for name in sorted(os.listdir(drivers))
            if os.path.isfile(os.path.join(drivers, name))}
    if tree:
        _, code = manifest(tree, True)
        record["tree"] = {"root": os.path.abspath(tree), "layout": layout_of(tree),
                          "code_digest": code}
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True, default=str)
    with open(out, "r", encoding="utf-8") as handle:
        json.load(handle)  # PROVED PARSEABLE before the ladder relies on it
    return record


# ---------------------------------------------------------------------------
# record-gpu
# ---------------------------------------------------------------------------

def lift_seconds(run_log: Optional[str]) -> Dict[str, float]:
    found: Dict[str, float] = {}
    if not run_log:
        return found
    try:
        with open(run_log, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                match = LIFT_LINE.search(line)
                if match:
                    found[match.group(2)] = float(match.group(4))
    except OSError:
        pass
    return found


def pinned_foreign(box: Optional[Dict[str, Any]]) -> Optional[List[str]]:
    """Compute apps on the PINNED device other than the row's own process, joined by
    UUID (``CUDA_VISIBLE_DEVICES`` index -> ``gpus`` -> ``compute_apps``).

    The row's own ``foreign_compute_apps_on_pinned_gpu`` compares an index with a UUID
    and so names a neighbour on ANY device; it is recorded beside this, not instead.
    """
    if not box:
        return None
    index = str(box.get("CUDA_VISIBLE_DEVICES", "")).split(",")[0].strip()
    uuid = index if index.startswith("GPU-") else None
    for line in str(box.get("gpus") or "").splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) >= 2 and cells[0] == index:
            uuid = cells[1]
    if uuid is None:
        return None
    own = str(box.get("pid"))
    found = []
    for line in str(box.get("compute_apps") or "").splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) >= 2 and cells[0] == uuid and cells[1] != own:
            found.append(line.strip())
    return found


def repair_check(row: Dict[str, Any], expected: Optional[str]) -> Dict[str, Any]:
    """The fused leg's deposit-repair route against the case's declaration.

    ``expected`` ``None``: no bracket may sit on the plan. ``"B"``/``"D"``: the bracket
    must occupy exactly that seam's two slots and must have repaired at least once.
    """
    route = (((row.get("per_leg") or {}).get("fused") or {})
             .get("deposit_repair_route") or {})
    slots = list(route.get("bracketed_slots") or [])
    repairs = int(route.get("linear_repairs") or 0) + int(route.get("cells_repairs") or 0)
    seams = sorted({seam for seam, pair in REPAIR_SLOTS.items()
                    if any(slot in pair for slot in slots)})
    if not route:
        ok, why = False, "the fused leg carries no deposit_repair_route record"
    elif expected is None:
        ok = not slots
        why = None if ok else f"a deposit-repair bracket sits on {slots}; none was declared"
    else:
        want = list(REPAIR_SLOTS[expected])
        ok = sorted(slots) == sorted(want) and repairs > 0
        why = None if ok else (f"declared the {expected}-seam bracket ({want}); the fused "
                               f"leg bracketed {slots} with {repairs} repairs")
    return {"expected": expected, "bracketed_slots": slots, "seams": seams,
            "repairs": repairs, "ok": ok, "why": why,
            "counters": {k: route.get(k) for k in
                         ("linear_saves", "linear_repairs", "cells_saves",
                          "cells_repairs", "restricted_sources", "restriction_fallbacks")
                         if k in route}}


def gpu_facts(row: Dict[str, Any], run_log: Optional[str]) -> Dict[str, Any]:
    cells = cells_of(row.get("grid_shape"))
    legs = row.get("per_leg") or {}
    substitution = row.get("substitution") or {}
    plans = row.get("plans") or {}
    box_before = row.get("box_before") or {}
    out: Dict[str, Any] = {
        "case": row.get("case"), "timing_case": row.get("timing_case"),
        "verdict": row.get("verdict"), "reportable": row.get("reportable"),
        "error": row.get("error"), "why": row.get("why"),
        "floors_failed": [k for k, v in (row.get("floors") or {}).items() if v is False],
        "grid_shape": row.get("grid_shape"), "cells": cells,
        "drive_table": row.get("drive_table"), "reached_by": row.get("reached_by"),
        "monitors_mode": row.get("monitors_mode"),
        "monitors_attached": (row.get("monitors_attached") or {}).get("fused"),
        "row_elapsed_s": row.get("elapsed_s"), "row_utc": row.get("utc"),
        "prefer_gpu": row.get("prefer_gpu"), "smoke": row.get("smoke"),
        "bit_identical_fused_vs_array": (row.get("bit_identity") or {}).get(
            "fused_vs_array_identical"),
        "bit_identity_words": (row.get("bit_identity") or {}).get("words"),
        "tables_dispatched": substitution.get("tables_dispatched"),
        "tables_by_leg": substitution.get("tables_by_leg"),
        "substitution_proved": substitution.get("proved"),
        "launch_witness_available": substitution.get("launch_witness_available"),
        "witnesses_installed": substitution.get("witnesses_installed"),
        "fused_arms": substitution.get("fused_arms"),
        "legs": {}, "lift_seconds": lift_seconds(run_log),
        "provenance_sha256": (row.get("provenance") or {}).get("sha256"),
        "load1_before": (box_before.get("loadavg") or [None])[0],
        "load1_after": ((row.get("box_after") or {}).get("loadavg") or [None])[0],
        "foreign_on_pinned_device_before": pinned_foreign(box_before),
        "foreign_on_pinned_device_after": pinned_foreign(row.get("box_after")),
        "foreign_compute_apps_on_pinned_gpu_before": box_before.get(
            "foreign_compute_apps_on_pinned_gpu"),
        "host_before": row.get("host_before"), "host_after": row.get("host_after"),
    }
    for label, leg in legs.items():
        per_step = leg.get("median_seconds_per_step")
        plan = plans.get(label) or {}
        out["legs"][label] = {
            "ms_per_step": None if per_step is None else per_step * 1e3,
            "mcell_steps_per_s": rate(cells, per_step),
            "spread": leg.get("spread"), "windows": leg.get("windows"),
            "steps_per_window": leg.get("steps_per_window"),
            "launches_per_step": leg.get("launches_per_step"),
            "new_kernels_inside_timed_region": leg.get("new_kernels_inside_timed_region"),
            "deposit_repair_route": leg.get("deposit_repair_route"),
            "step_path": plan.get("step_path"),
            "enable": plan.get("enable"),
            "launch_counters": plan.get("launch_counters"),
        }
    served = out["tables_dispatched"] or []
    out["what_launched"] = ("+".join(served) if served else "nothing (array path)")
    policy = row.get("subnormal_policy")
    if isinstance(policy, dict):
        out["subnormal_policy"] = {k: policy.get(k) for k in
                                   ("requested", "effective", "policy", "ftz_removed")
                                   if k in policy} or policy
    return out


def condition_strikes(entry: Dict[str, Any], row: Optional[Dict[str, Any]] = None
                      ) -> List[str]:
    """Why a row taken on a Mac is not a timing: the power or thermal state before or
    after it (the ladder's ``cpu_state_before``/``_after`` and the bench's
    ``host_before``/``_after``), and on a GPU row another process driving the Apple GPU
    (the bench's ``host_*.foreign_processes``). Applies only when the entry says its
    platform is ``darwin``; ``require_ac_power`` is the ladder's setting."""
    if entry.get("platform") != "darwin":
        return []
    require = bool(entry.get("require_ac_power", True))
    reasons: List[str] = []
    for label in ("cpu_state_before", "cpu_state_after"):
        for why in timing_host.darwin_condition_reasons(entry.get(label), require):
            reasons.append(f"{label}: {why}")
    for label in ("host_before", "host_after"):
        state = (row or {}).get(label)
        if row is None or state is None:
            continue
        for why in timing_host.darwin_condition_reasons(state, require):
            reasons.append(f"{label}: {why}")
        foreign = state.get("foreign_processes")
        if foreign:
            reasons.append(f"{label}: another process drives the Apple GPU: "
                           f"{foreign[:3]}")
    return reasons


def monitor_count(attached: Any) -> Optional[int]:
    """The monitors a GPU leg carried: the sum over the driver's monitor lists."""
    if not isinstance(attached, dict):
        return None
    return sum(int(v) for v in attached.values() if isinstance(v, (int, float)))


def record_gpu(ledger: str, row_dir: str, run_log: Optional[str],
               expect_launched: Optional[str], expect_repair: Optional[str],
               check_repair: bool, fields: Dict[str, Any],
               extra: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], str]:
    rows_path = os.path.join(row_dir, "rows.jsonl")
    row, why = last_row(rows_path)
    entry: Dict[str, Any] = {"ledger": "gpu", "utc": utc(), "rows_file": rows_path,
                             "run_log": run_log, **fields, **(extra or {})}
    if row is None:
        entry["measured"] = None
        entry["struck"] = f"no row: {why}"
        append(ledger, entry)
        return entry, f"NO ROW ({why})"
    facts = gpu_facts(row, run_log)
    entry["measured"] = facts
    entry["case"] = entry.get("case") or facts["case"]
    repair = repair_check(row, expect_repair) if check_repair else None
    entry["deposit_repair"] = repair
    fused = (facts["legs"].get("fused") or {})
    array = (facts["legs"].get("array") or {})
    unfused = (facts["legs"].get("unfused") or {})
    facts["monitor_count"] = monitor_count(facts.get("monitors_attached"))
    conditions = condition_strikes(entry, row)
    entry["conditions"] = conditions
    if facts["verdict"] != "TIMED":
        entry["struck"] = (f"{facts['verdict']}: floors failed "
                           f"{facts['floors_failed'] or facts.get('error')}")
    elif facts["monitors_mode"] != "attached":
        # THE MEEP SIDE ALWAYS ACCUMULATES ITS FLUX MONITOR: a GPU row timed without
        # one is a different workload and is never divided by a MEEP row.
        entry["struck"] = (f"monitors {facts['monitors_mode']}: a GPU row is timed with "
                           "its monitors attached, as the MEEP rows are")
    elif expect_launched and facts["what_launched"] != expect_launched:
        # A ROW THAT FELL TO ANOTHER ROUTE IS A NUMBER, NOT AN ERROR, unless it is read.
        entry["struck"] = (f"launched {facts['what_launched']}, and this route is "
                           f"{expect_launched}")
    elif facts["bit_identical_fused_vs_array"] is not True:
        entry["struck"] = "the fused leg is not bit-identical to the array leg"
    elif repair is not None and not repair["ok"]:
        entry["struck"] = f"deposit-repair route: {repair['why']}"
    elif conditions:
        entry["struck"] = "host conditions: " + "; ".join(conditions)
    entry["expected_launched"] = expect_launched
    append(ledger, entry)

    def number(value: Optional[float], spec: str) -> str:
        return "n/a" if value is None else format(value, spec)

    lifts = ",".join(f"{k}={v:.0f}s" for k, v in facts["lift_seconds"].items())
    summary = (f"{facts['verdict']} cells={facts['cells']} launched={facts['what_launched']} "
               f"fused {number(fused.get('mcell_steps_per_s'), '.1f')} Mcell-steps/s "
               f"({number(fused.get('ms_per_step'), '.4f')} ms, spread "
               f"{number(fused.get('spread'), '.3f')}, "
               f"{number(fused.get('launches_per_step'), '.2f')} launches/step) "
               f"singles {number(unfused.get('mcell_steps_per_s'), '.1f')} "
               f"array {number(array.get('mcell_steps_per_s'), '.1f')} "
               f"(array launches/step {number(array.get('launches_per_step'), '.2f')}) "
               f"bit_identical={facts['bit_identical_fused_vs_array']} lift[{lifts}]"
               + (f" repair={','.join(repair['seams']) or 'none'}"
                  f"(expected {repair['expected'] or 'none'})" if repair else "")
               + (f" STRUCK: {entry['struck']}" if entry.get("struck") else ""))
    return entry, summary


def command_record_gpu(args: Any) -> int:
    _entry, summary = record_gpu(
        args.ledger, args.row_dir, args.run_log, args.expect_launched,
        None if args.expect_repair in (None, "none") else args.expect_repair,
        args.expect_repair is not None, parse_fields(args.field))
    print(summary)
    return 0


# ---------------------------------------------------------------------------
# record-meep
# ---------------------------------------------------------------------------

def digest_of_record(digest_file: Optional[str], candidate: Optional[str],
                     row_id: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """The digest a MEEP row at this (case, resolution) must carry, and its source.

    PRECEDENCE: the harness lift's digest, written by the ladder's ``digests`` tier
    (``.source`` reads ``harness-lift``), when present. Otherwise the 2026-09-28 rule:
    the first MEEP measurement at the size sets it (``first-meep-row:<row_id>``), and
    every later row must carry the same one.
    """
    if not digest_file:
        return None, None
    source_file = digest_file[:-len(".sha256")] + ".source" if digest_file.endswith(
        ".sha256") else digest_file + ".source"
    try:
        with open(digest_file, "r", encoding="utf-8") as handle:
            value = handle.read().strip()
        try:
            with open(source_file, "r", encoding="utf-8") as handle:
                source = handle.read().strip()
        except OSError:
            source = "unrecorded"
        return value or None, source
    except OSError:
        pass
    if not candidate:
        return None, None
    os.makedirs(os.path.dirname(os.path.abspath(digest_file)), exist_ok=True)
    with open(digest_file, "w", encoding="utf-8") as handle:
        handle.write(str(candidate) + "\n")
    with open(source_file, "w", encoding="utf-8") as handle:
        handle.write(f"first-meep-row:{row_id}\n")
    return candidate, f"first-meep-row:{row_id}"


def write_lift_digest(digest_file: str, digest: str, origin: str) -> None:
    """The harness lift's digest becomes the digest of record for its size."""
    os.makedirs(os.path.dirname(os.path.abspath(digest_file)), exist_ok=True)
    with open(digest_file, "w", encoding="utf-8") as handle:
        handle.write(digest + "\n")
    with open(digest_file[:-len(".sha256")] + ".source", "w", encoding="utf-8") as handle:
        handle.write(f"harness-lift:{origin}\n")


def configuration_of(measurement: Dict[str, Any]) -> Dict[str, Any]:
    """The row's MEEP configuration; a row that records none ran the reference one."""
    found = dict(REFERENCE_CONFIGURATION)
    found.update({k: v for k, v in (measurement.get("configuration") or {}).items()
                  if v is not None})
    if measurement.get("stepper"):
        found["stepper"] = measurement["stepper"]
    return found


#: The CPU seconds per wall second a rank asked for T OpenMP threads must reach, per
#: thread: a single-threaded rank that busy-polls MPI already reads about 1.0, so the
#: floor at T = 2 must sit well above it.
THREADS_CPU_FLOOR_PER_THREAD = 0.75


def threads_verdict(threads: int, observed: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Whether a row asked for ``threads`` OpenMP threads per rank ran them.

    Two facts the bench records per rank, minimum over ranks:

    * ``omp_threads_added_min``: threads the process gained between MPI initialisation
      (before any OpenMP region, so the MPI library's helper threads are already in the
      baseline) and the end of the windows; it must be at least T - 1. A raw thread
      count cannot be used: the MPI helper threads alone put a single-threaded rank at
      two or more.
    * ``cpu_seconds_over_wall_min``: at least 0.75 T.

    A row recorded before the baseline existed falls back to the raw count against T.
    """
    if threads <= 1:
        return True, None
    added = observed.get("omp_threads_added_min")
    cpu = observed.get("cpu_seconds_over_wall_min") or 0.0
    floor = THREADS_CPU_FLOOR_PER_THREAD * threads
    if added is not None:
        ok = int(added) >= threads - 1 and float(cpu) >= floor
        counted = f"{added} threads added after MPI initialisation (floor {threads - 1})"
    else:
        ok = (observed.get("threads_min") or 0) >= threads and float(cpu) >= floor
        counted = f"{observed.get('threads_min')} threads (floor {threads})"
    if ok:
        return True, None
    return False, (f"{threads} threads per rank asked; observed {counted} and "
                   f"{cpu} CPU seconds per wall second (floor {floor:.2f})")


def cores_verdict(entry: Dict[str, Any], facts: Dict[str, Any]) -> Optional[str]:
    """A bound row whose ranks used fewer distinct physical cores than min(ranks, P):
    the ``bound-hwthread`` control filling both hardware threads of some cores while
    others idle cannot test what it is for. Read from the bench's per-rank affinity
    mapped to cores; ``None`` (no strike) when the map was not available."""
    mode = entry.get("binding_mode")
    binding = facts.get("binding") or {}
    used = binding.get("physical_cores_used")
    physical = entry.get("physical_cores")
    ranks = facts.get("ranks")
    if mode not in timing_host.BOUND_MODES or used is None or not physical or not ranks:
        return None
    want = min(int(ranks) * (int(facts.get("configuration", {}).get("threads_per_rank")
                                 or 1) if mode == "bound-pe" else 1), int(physical))
    if int(used) < want:
        return (f"the {ranks} ranks of this {mode} row used {used} distinct physical "
                f"cores of {physical}; at least {want} were expected, so cores sat idle "
                "while others ran two ranks")
    return None


def record_meep(ledger: str, rows_file: str, row_id: str, run_log: Optional[str],
                digest_file: Optional[str], expect_digest: Optional[str],
                fields: Dict[str, Any],
                extra: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], str]:
    entry: Dict[str, Any] = {"ledger": "meep", "utc": utc(), "rows_file": rows_file,
                             "run_log": run_log, "row_id": row_id, **fields,
                             **(extra or {})}
    found = [row for row in rows_of(rows_file) if row.get("row_id") == row_id]
    measurement = next((row for row in found if row.get("row") == "measurement"), None)
    geometry = next((row for row in found if row.get("row") == "geometry"), None)
    report_text = ""
    if run_log:
        try:
            with open(run_log, "r", encoding="utf-8", errors="replace") as handle:
                report_text = handle.read()
        except OSError:
            report_text = ""
    bound_lines = timing_host.binding_report_lines(report_text)[:256]
    entry["launcher_bindings"] = {"lines": len(bound_lines),
                                  "first": bound_lines[:2], "last": bound_lines[-1:]}
    if measurement is None:
        windows = [row for row in found if row.get("row") == "window"]
        entry["measured"] = None
        entry["windows_landed"] = len(windows)
        entry["geometry_digest"] = (geometry or {}).get("geometry_digest")
        entry["struck"] = (f"no measurement row for {row_id}: "
                           f"{len(windows)} window rows landed")
        append(ledger, entry)
        return entry, f"NO MEASUREMENT ({len(windows)} window rows landed)"
    environment = measurement.get("environment") or {}
    configuration = configuration_of(measurement)
    facts = {
        "case": measurement.get("case"),
        "cells": measurement.get("cells"), "ranks": measurement.get("ranks"),
        "mcell_steps_per_s": measurement.get("mcell_steps_per_s"),
        "mcell_steps_per_s_min": measurement.get("mcell_steps_per_s_min"),
        "mcell_steps_per_s_max": measurement.get("mcell_steps_per_s_max"),
        "ms_per_step": measurement.get("ms_per_step"),
        "spread": measurement.get("spread"),
        "spread_within_0.05": measurement.get("spread_within_0.05"),
        "windows": measurement.get("windows"),
        "steps_per_window": measurement.get("steps_per_window"),
        "stepper": measurement.get("stepper"),
        "build_seconds": measurement.get("build_seconds"),
        "init_seconds": measurement.get("init_seconds"),
        "warm_up": measurement.get("warm_up"),
        "monitors_attached": measurement.get("monitors_attached"),
        "monitor": measurement.get("monitor"),
        "geometry_digest": measurement.get("geometry_digest"),
        "meep": environment.get("meep"),
        "single_precision": environment.get("single_precision"),
        "binding": environment.get("binding"),
        "binding_label": environment.get("binding_label"),
        "threads": environment.get("threads"),
        "configuration": configuration,
        "threads_observed": measurement.get("threads_observed"),
        "chunks": measurement.get("chunks"),
        "bench_sha256": (environment.get("bench") or {}).get("sha256"),
        "builder": {k: v for k, v in (environment.get("builder") or {}).items()
                    if k != "modules"},
        "builder_modules": {k: (v or {}).get("sha256") for k, v in
                            ((environment.get("builder") or {}).get("modules")
                             or {}).items()},
    }
    entry["measured"] = facts
    entry["configuration"] = configuration
    expected, source = digest_of_record(digest_file, facts["geometry_digest"], row_id)
    expected = expect_digest or expected
    entry["geometry_digest_of_record"] = expected or facts["geometry_digest"]
    entry["geometry_digest_source"] = "argument" if expect_digest else source
    threads = int(configuration.get("threads_per_rank") or 1)
    observed = facts["threads_observed"] or {}
    threads_ok, threads_why = threads_verdict(threads, observed)
    cores_why = cores_verdict(entry, facts)
    conditions = condition_strikes(entry)
    entry["conditions"] = conditions
    if expected and facts["geometry_digest"] != expected:
        entry["struck"] = (f"geometry digest {facts['geometry_digest']} is not the "
                           f"digest of record {expected} "
                           f"(source: {entry['geometry_digest_source']})")
    elif (facts["monitor"] or {}).get("stepped") is not True:
        entry["struck"] = "the flux monitor shows no sign of having been stepped"
    elif facts["single_precision"] is not True:
        entry["struck"] = "the MEEP build is not single precision"
    elif not threads_ok:
        entry["struck"] = threads_why
    elif cores_why:
        entry["struck"] = cores_why
    elif conditions:
        entry["struck"] = "host conditions: " + "; ".join(conditions)
    append(ledger, entry)

    def number(value: Any, spec: str) -> str:
        return "n/a" if not isinstance(value, (int, float)) else format(value, spec)

    summary = (f"MEASURED case={facts['case']} cells={facts['cells']} ranks={facts['ranks']} "
               f"{number(facts['mcell_steps_per_s'], '.1f')} Mcell-steps/s "
               f"({number(facts['ms_per_step'], '.4f')} ms, spread "
               f"{number(facts['spread'], '.3f')}, "
               f"{facts['windows']} windows x {facts['steps_per_window']} steps) "
               f"init {number(facts['init_seconds'], '.1f')} s "
               f"binding={facts['binding_label']} "
               f"bound={(facts['binding'] or {}).get('bound')} "
               f"build={configuration.get('build')} threads={threads} "
               f"split={configuration.get('split_chunks_evenly')} "
               f"monitor_stepped={(facts['monitor'] or {}).get('stepped')} "
               f"digest={str(facts['geometry_digest'])[:16]}"
               + (f" STRUCK: {entry['struck']}" if entry.get("struck") else ""))
    return entry, summary


def command_record_meep(args: Any) -> int:
    _entry, summary = record_meep(args.ledger, args.rows_file, args.row_id, args.run_log,
                                  args.digest_file, args.expect_digest,
                                  parse_fields(args.field))
    print(summary)
    return 0


# ---------------------------------------------------------------------------
# record-control
# ---------------------------------------------------------------------------

#: The route gate whose ``ARRAY_ENV`` forces each table's array leg.
ARRAY_ENV_GATES = {"triton": "gate_dispatch_fused_route.py",
                   "cuda": "gate_dispatch_fused_route.py",
                   "metal": "gate_dispatch_metal_route.py"}


def array_env_of(harness_root: Optional[str],
                 gate_file: str = "gate_dispatch_fused_route.py") -> Any:
    """The array leg's environment, READ from the harness that ran, never re-typed,
    from the route gate of the table the GPU rows drove (``ARRAY_ENV_GATES``)."""
    if not harness_root:
        return None
    path = os.path.join(harness_root, gate_file)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        return None
    match = re.search(r"^ARRAY_ENV[^=]*=\s*(\{.*?\})", text, re.S | re.M)
    if not match:
        return None
    literal = match.group(1)
    try:
        import ast  # noqa: PLC0415
        return ast.literal_eval(literal)
    except Exception:  # noqa: BLE001
        return " ".join(literal.split())


def record_control(ledger: str, control_out: Optional[str], gpu_row_dirs: Sequence[str],
                   harness_root: Optional[str], standalone: bool,
                   fields: Dict[str, Any],
                   gate_file: str = "gate_dispatch_fused_route.py"
                   ) -> Tuple[Dict[str, Any], str]:
    forced = array_env_of(harness_root, gate_file)
    standalone_facts = None
    sources = []
    for directory in gpu_row_dirs:
        row, why = last_row(os.path.join(directory, "rows.jsonl"))
        if row is None:
            sources.append({"row_dir": directory, "measured": None, "why": why})
            continue
        if standalone:
            standalone_facts = {"forced_by": row.get("forced_by"),
                                "environment_of_the_leg": row.get("environment_of_the_leg"),
                                "row_verdict": row.get("verdict"),
                                "floors": row.get("floors"), "lift_s": row.get("lift_s"),
                                "error": row.get("error")}
        facts = gpu_facts(row, None)
        array = facts["legs"].get("array") or {}
        sources.append({
            "row_dir": directory, "row_verdict": facts["verdict"],
            "cells": facts["cells"], "drive_table": facts["drive_table"],
            "mcell_steps_per_s": array.get("mcell_steps_per_s"),
            "ms_per_step": array.get("ms_per_step"), "spread": array.get("spread"),
            "windows": array.get("windows"),
            "steps_per_window": array.get("steps_per_window"),
            "launches_per_step": array.get("launches_per_step"),
            "tables_dispatched_by_this_leg": (facts["tables_by_leg"] or {}).get("array"),
            "step_path": array.get("step_path"), "enable": array.get("enable"),
            "kernels_vetoed": (array.get("launches_per_step") == 0.0
                               and not (facts["tables_by_leg"] or {}).get("array")),
        })
    rates = [s["mcell_steps_per_s"] for s in sources if s.get("mcell_steps_per_s")]
    entry: Dict[str, Any] = {
        "ledger": "array_control", "utc": utc(), "mode": "extracted",
        "how_forced": {"environment_of_the_array_leg": forced,
                       "read_from": f"{gate_file[:-3]}.ARRAY_ENV in the tree that ran",
                       "process": "the array leg of each GPU row, timed in the same "
                                  "process as the fused leg, AB/BA interleaved"},
        "sources": sources,
        "mcell_steps_per_s_mean": (sum(rates) / len(rates)) if rates else None,
        **fields}
    if standalone:
        entry["mode"] = "standalone"
        entry["how_forced"] = {"standalone_row": standalone_facts,
                               "process": "one fresh process, one leg, lifted and timed "
                                          "by gpu_array_control.py"}
        forced = (standalone_facts or {}).get("environment_of_the_leg")
    vetoed = [s.get("kernels_vetoed") for s in sources if "kernels_vetoed" in s]
    if not rates:
        entry["struck"] = "no GPU row of this size and pass carried an array leg"
    elif not all(vetoed):
        entry["struck"] = "an array leg launched a kernel or named a table"
    elif standalone and (standalone_facts or {}).get("row_verdict") != "TIMED":
        entry["struck"] = (f"the standalone row is "
                           f"{(standalone_facts or {}).get('row_verdict')}: floors "
                           f"{(standalone_facts or {}).get('floors')}")
    append(ledger, entry)
    if control_out:
        append(control_out, entry)
    if rates:
        summary = (f"ARRAY CONTROL ({entry['mode']}, {len(rates)} row(s)) "
                   f"{entry['mcell_steps_per_s_mean']:.1f} Mcell-steps/s, forced by "
                   f"{forced}, kernels vetoed {all(vetoed)}")
    else:
        summary = "ARRAY CONTROL: no array leg to read"
    return entry, summary


def command_record_control(args: Any) -> int:
    _entry, summary = record_control(args.ledger, args.control_out, args.gpu_row_dir,
                                     args.harness_root, args.standalone,
                                     parse_fields(args.field), args.gate_file)
    print(summary)
    return 0


def record_note(ledger: str, kind: str, reason: str, fields: Dict[str, Any]) -> str:
    append(ledger, {"ledger": kind, "utc": utc(), "measured": None, "struck": reason,
                    **fields})
    return f"STRUCK {reason}"


def command_record_note(args: Any) -> int:
    print(record_note(args.ledger, args.kind, args.reason, parse_fields(args.field)))
    return 0


# ---------------------------------------------------------------------------
# decide-binding
# ---------------------------------------------------------------------------

def decide_binding(probe_dir: str, bindings: Sequence[str], default: str,
                   out: str) -> Tuple[str, Dict[str, Any]]:
    """THE RULE, FIXED BEFORE ANY DATA: per binding, the median over passes at each
    rank count, then the best rank count; the binding whose best is higher is used for
    every MEEP row at or below the physical core count. A tie, or a binding with no
    measurement, leaves the default."""
    table: Dict[str, Dict[str, Any]] = {}
    for label in bindings:
        directory = os.path.join(probe_dir, label)
        by_rank: Dict[int, List[float]] = {}
        try:
            names = sorted(os.listdir(directory))
        except OSError:
            names = []
        for name in names:
            if not (name.startswith("rows_r") and name.endswith(".jsonl")):
                continue
            for row in rows_of(os.path.join(directory, name)):
                if row.get("row") == "measurement":
                    by_rank.setdefault(int(row["ranks"]), []).append(
                        float(row["mcell_steps_per_s"]))
        medians = {ranks: statistics.median(values) for ranks, values in by_rank.items()}
        table[label] = {"passes": {str(k): v for k, v in by_rank.items()},
                        "median_by_ranks": {str(k): v for k, v in medians.items()},
                        "best": max(medians.values()) if medians else None}
    measured = {k: v["best"] for k, v in table.items() if v["best"] is not None}
    chosen = default
    reason = "default: not every binding was measured"
    if len(measured) == len(bindings) and measured:
        best = max(measured, key=lambda key: measured[key])
        others = [v for k, v in measured.items() if k != best]
        if not others or measured[best] > max(others):
            chosen = best
            reason = "the higher best-rank median over passes"
        else:
            reason = "default: a tie"
    record = {"utc": utc(), "rule": decide_binding.__doc__, "bindings": table,
              "default": default, "chosen": chosen, "reason": reason}
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
    return chosen, record


def command_decide_binding(args: Any) -> int:
    chosen, record = decide_binding(args.probe_dir, args.binding, args.default, args.out)
    measured = {k: v["best"] for k, v in record["bindings"].items()}
    print(f"binding probe: {json.dumps(measured)} -> {chosen} ({record['reason']})",
          file=sys.stderr)
    print(chosen)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    one = sub.add_parser("manifest")
    one.add_argument("--root", required=True)
    one.add_argument("--out", default=None)
    one.add_argument("--code-only", action="store_true", dest="code_only")
    one.add_argument("--exclude", action="append", default=[])
    one.set_defaults(run=command_manifest)

    one = sub.add_parser("record-gpu")
    one.add_argument("--ledger", required=True)
    one.add_argument("--row-dir", required=True, dest="row_dir")
    one.add_argument("--run-log", default=None, dest="run_log")
    one.add_argument("--expect-launched", default=None, dest="expect_launched")
    one.add_argument("--expect-repair", default=None, dest="expect_repair",
                     choices=("none", "B", "D"))
    one.add_argument("--field", action="append", default=[])
    one.set_defaults(run=command_record_gpu)

    one = sub.add_parser("record-meep")
    one.add_argument("--ledger", required=True)
    one.add_argument("--rows-file", required=True, dest="rows_file")
    one.add_argument("--row-id", required=True, dest="row_id")
    one.add_argument("--run-log", default=None, dest="run_log")
    one.add_argument("--expect-digest", default=None, dest="expect_digest")
    one.add_argument("--digest-file", default=None, dest="digest_file")
    one.add_argument("--field", action="append", default=[])
    one.set_defaults(run=command_record_meep)

    one = sub.add_parser("record-control")
    one.add_argument("--ledger", required=True)
    one.add_argument("--control-out", default=None, dest="control_out")
    one.add_argument("--gpu-row-dir", action="append", default=[], dest="gpu_row_dir")
    one.add_argument("--harness-root", default=None, dest="harness_root")
    one.add_argument("--standalone", action="store_true")
    one.add_argument("--gate-file", default="gate_dispatch_fused_route.py",
                     dest="gate_file", help="the route gate whose ARRAY_ENV forced the "
                     "array legs (gate_dispatch_metal_route.py for Metal rows)")
    one.add_argument("--field", action="append", default=[])
    one.set_defaults(run=command_record_control)

    one = sub.add_parser("record-note")
    one.add_argument("--ledger", required=True)
    one.add_argument("--kind", required=True)
    one.add_argument("--reason", required=True)
    one.add_argument("--field", action="append", default=[])
    one.set_defaults(run=command_record_note)

    one = sub.add_parser("decide-binding")
    one.add_argument("--probe-dir", required=True, dest="probe_dir")
    one.add_argument("--binding", action="append", default=[])
    one.add_argument("--default", default="bound")
    one.add_argument("--out", required=True)
    one.set_defaults(run=command_decide_binding)

    one = sub.add_parser("meep-build-record")
    one.add_argument("--python", required=True)
    one.add_argument("--config-log", default=None, dest="config_log")
    one.add_argument("--launcher", default=None)
    one.add_argument("--extra-json", action="append", default=[], dest="extra_json",
                     help="a JSON object of build facts to merge under 'build'")
    one.add_argument("--require-single", action="store_true", dest="require_single")
    one.add_argument("--out", required=True)
    one.set_defaults(run=command_meep_build_record)

    args = parser.parse_args(argv)
    try:
        return int(args.run(args) or 0)
    except SystemExit:
        raise
    except Exception as error:  # noqa: BLE001 - a recorder never stops the ladder
        print(f"RECORDER ERROR {type(error).__name__}: {error}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
