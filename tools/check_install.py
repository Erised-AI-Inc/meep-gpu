#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Check that MEEP and meep-gpu are installed and work together on this host.

Run it after installing, with the Python of the environment that will run your
simulations, from the repository root, as one process (not under mpirun):

    python tools/check_install.py

What it does, in order:

0. on a Mac with PyTorch installed, reads, before any process imports PyTorch,
   which OpenMP runtime PyTorch will load and which one NumPy and MEEP load; when
   they are two different files, every import order stops with 'OMP: Error #15',
   and the check stops here and names the fix instead;
1. prints the Python and NumPy versions;
2. imports MEEP and prints its version, whether it is a serial or an MPI build,
   and whether it is a single- or a double-precision build;
3. checks that this MEEP has every name the lift reads from it;
4. imports meep_gpu and prints which GPU route this host has;
5. builds one small simulation (a dielectric sphere in a 24 x 24 x 24 cell with
   absorbing boundaries and one flux monitor) and steps it three ways, each in
   its own process: with the package's default (prefer_gpu=True, this host's
   GPU route), on the NumPy reference, and with MEEP itself;
6. prints whether the device and the toolchain are on the certified list,
   whether compiled kernels served the default run, where that run stepped, and
   whether the three runs agree.

The exit status is 0 when everything that ran is in order. Otherwise the last
line states the reason and names the section of INSTALL.md to read, and the exit
status is 1.

Options:

    --require-gpu   fail when this host has no GPU route (the default is to
                    check the NumPy reference alone on such a host)
    --gpu-id N      the CUDA device to use (default 0); an Apple GPU is device 0
    --verbose       also print what each run wrote

Only the documented surface of meep_gpu is used: available_gpu,
missing_dependencies, gpu_compatibility, run_on_gpu, the result object, and the
driver's active_step_path and fast_path_report().
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import struct
import subprocess
import sys
import tempfile
import time

#: MEEP releases this package has been run against (INSTALL.md, "Tested MEEP
#: versions and builds", says what was run on each).
TESTED_MEEP_VERSIONS = ("1.33.0", "1.34.0")
#: The message of the linker when Triton cannot link its launcher against the
#: NVIDIA driver library (``-lcuda`` needs a file named ``libcuda.so``).
LIBCUDA_LINK_ERROR = re.compile(r"cannot find -lcuda|libcuda\.so[^\n]*(?:not found|cannot)")
# The largest relative difference accepted between two runs of the same
# simulation. The engine steps single precision. On the hosts this case has
# been run on, the reference differs from MEEP by 1.5e-6 (double-precision MEEP)
# and 2.5e-6 (single-precision MEEP) of the final field, and from the GPU run by
# 0. The band is wider than that, so it fails on a defect and not on rounding.
AGREEMENT_BAND = 1.0e-4
RUN_UNTIL = 20.0
LEG_TIMEOUT_S = 1200
MARK = "CHECK-INSTALL-JSON "

# Section titles of INSTALL.md, named in failure lines.
SECTION_NEED = "What you need"
SECTION_TESTED = "Tested MEEP versions and builds"
SECTION_EXISTING = "Installing into an existing MEEP environment"
SECTION_MPI = "MPI"
SECTION_PRECISION = "Precision"
SECTION_OPENMP = "One OpenMP runtime on an Apple silicon Mac"
SECTION_UNCERTIFIED = "GPUs that are not on the certified list"
SECTION_TROUBLE = "Troubleshooting"

# Every MEEP name the lift reads, by where it is read from. A MEEP release that
# lacks one of them cannot be lifted from.
MODULE_NAMES = (
    "ALL_COMPONENTS", "AUTOMATIC", "CYLINDRICAL", "NO_DIRECTION", "EVEN_Z", "ODD_Z",
    "X", "Y", "Z", "R", "P", "Low", "High", "ALL", "E_stuff",
    "Ex", "Ey", "Ez", "Er", "Ep", "Hx", "Hy", "Hz", "Hr", "Hp",
    "Dx", "Dy", "Dz", "Dr", "Dp", "Bx", "By", "Bz", "Br", "Bp",
    "Simulation", "Medium", "MaterialGrid", "Block", "Vector3", "Volume", "Mirror",
    "PML", "Absorber", "Source", "ContinuousSource", "GaussianSource", "CustomSource",
    "EigenModeSource", "GaussianBeamSource", "GaussianBeam3DSource",
    "GaussianBeam2DSource", "IndexedSource", "LorentzianSusceptibility",
    "DrudeSusceptibility",
    "fields", "structure", "grid_volume", "ivec", "iveccyl", "py_v3_to_vec",
    "component_name", "is_point_in_object", "get_equiv_sources", "count_processors",
    "is_single_precision", "get_realnum_size", "with_mpi", "am_master",
    "set_zero_subnormals", "__version__",
)
SIMULATION_MODULE_NAMES = (
    "NearToFarData", "FluxData", "DftFlux", "DftFields", "DftNear2Far", "DftForce",
    "DftEnergy",
)
FIELDS_CLASS_NAMES = ("get_chi1inv", "get_eps", "get_mu", "last_source_time")
FIELDS_INSTANCE_NAMES = ("gv", "user_volume", "is_real", "dt", "t")
STRUCTURE_CLASS_NAMES = ("get_chi1inv",)
GRID_VOLUME_NAMES = (
    "big_corner", "little_corner", "dim", "index", "iyee_shift", "loc", "owns",
)
SIMULATION_NAMES = (
    "Courant", "cell_size", "default_material", "dimensions", "fields", "structure",
    "geometry", "geometry_center", "extra_materials", "boundary_layers", "symmetries",
    "sources", "dft_objects", "k_point", "m", "resolution", "is_cylindrical",
    "accurate_fields_near_cylorigin", "bfast_scaled_k", "special_kz",
    "ensure_periodicity", "eps_averaging", "force_complex_fields", "run_index",
    "init_sim", "reset_meep", "round_time", "meep_time", "timestep", "get_array",
    "get_array_metadata", "get_epsilon", "get_mu", "get_field_point", "get_eigenmode",
    "get_eigenmode_coefficients", "get_farfields", "load_near2far_data",
    "load_flux_data", "load_minus_flux_data", "add_flux", "add_dft_fields",
    "add_near2far", "add_force", "add_energy", "_is_initialized",
)
DFT_FLUX_NAMES = ("swigobj", "regions", "freq")
DFT_CHUNK_NAMES = ("next_in_dft", "sn", "vc", "get_decimation_factor")
# Integer values of MEEP constants that the lift relies on.
PINNED_VALUES = {
    "High": 0, "Low": 1, "ALL": -1, "CYLINDRICAL": -2, "AUTOMATIC": -1,
    "X": 0, "Y": 1, "Z": 2, "R": 3, "P": 4,
}
GPU_DISTRIBUTIONS = ("torch", "triton", "cupy", "cupy-cuda11x", "cupy-cuda12x",
                     "cupy-cuda13x")


# --------------------------------------------------------------------------
# The child processes. Each imports NumPy, then MEEP, then meep_gpu, in that
# order, does one thing, and prints one line of JSON.
# --------------------------------------------------------------------------

def emit(payload) -> None:
    print(MARK + json.dumps(payload, default=str, sort_keys=True), flush=True)


def build_simulation(mp):
    """The check's simulation: 24 x 24 x 24 cells, one sphere, one flux monitor."""
    sim = mp.Simulation(
        cell_size=mp.Vector3(4, 4, 4),
        resolution=6,
        boundary_layers=[mp.PML(0.8)],
        geometry=[mp.Sphere(radius=0.8, material=mp.Medium(epsilon=9))],
        sources=[mp.Source(mp.GaussianSource(frequency=0.4, fwidth=0.4),
                           component=mp.Ez, center=mp.Vector3(-1.2, 0, 0))],
    )
    flux = sim.add_flux(0.4, 0.4, 3, mp.FluxRegion(center=mp.Vector3(0.9, 0, 0),
                                                   size=mp.Vector3(0, 2, 2)))
    return sim, flux


def distribution_versions():
    from importlib import metadata

    found = {}
    for name in GPU_DISTRIBUTIONS + ("meep-gpu", "scipy"):
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = None
        except Exception as exc:  # noqa: BLE001
            found[name] = f"unreadable ({exc!r})"
    return found


def sweep(target, names):
    return [name for name in names if not hasattr(target, name)]


# --------------------------------------------------------------------------
# OpenMP runtimes on a Mac. A process may hold one OpenMP runtime: when a second
# copy initializes, it stops the process with 'OMP: Error #15'. The PyTorch
# wheels on PyPI carry their own copy (torch/lib/libomp.dylib); a conda-forge
# environment has another (lib/libomp.dylib, package llvm-openmp), which the
# OpenMP build of OpenBLAS loads, and with it NumPy and MEEP. Whether PyTorch
# loads its own copy is read from the load commands of torch/lib/
# libtorch_cpu.dylib, without importing PyTorch:
#   @loader_path/libomp.dylib (PyTorch 2.14.0): always its own copy;
#   @rpath/libomp.dylib (PyTorch 2.10.0, and conda-forge's PyTorch): a copy with
#     that install name already in the process is used; otherwise the first
#     file on the library's run-path list.
# --------------------------------------------------------------------------

OPENMP_FILE = re.compile(r"^lib[ig]?omp[0-9]*\.dylib$")
_LC_REQ_DYLD = 0x80000000
_LOAD_COMMANDS = {
    0x0C: "link",                   # LC_LOAD_DYLIB
    0x18 | _LC_REQ_DYLD: "link",    # LC_LOAD_WEAK_DYLIB
    0x1F | _LC_REQ_DYLD: "link",    # LC_REEXPORT_DYLIB
    0x20: "link",                   # LC_LAZY_LOAD_DYLIB
    0x23 | _LC_REQ_DYLD: "link",    # LC_LOAD_UPWARD_DYLIB
    0x1C | _LC_REQ_DYLD: "rpath",   # LC_RPATH
}
_FAT_MAGIC = {0xCAFEBABE: 20, 0xCAFEBABF: 32}
_CPU_TYPE = {"arm64": 0x0100000C, "x86_64": 0x01000007}


def macho_links(path):
    """``(linked library names, run-path list)`` from a Mach-O file's load commands.

    Reads the header and the load commands only. Raises ValueError, OSError or
    struct.error on a file it cannot read.
    """
    with open(path, "rb") as handle:
        start = 0
        magic = struct.unpack(">I", handle.read(4))[0]
        if magic in _FAT_MAGIC:  # a universal file: take this machine's slice
            width = _FAT_MAGIC[magic]
            count = struct.unpack(">I", handle.read(4))[0]
            table = handle.read(width * count)
            wanted = _CPU_TYPE.get(platform.machine())
            for index in range(count):
                entry = table[index * width:(index + 1) * width]
                if struct.unpack(">I", entry[:4])[0] == wanted:
                    start = (struct.unpack(">Q", entry[8:16])[0] if width == 32
                             else struct.unpack(">I", entry[8:12])[0])
                    break
            else:
                raise ValueError("no slice for this machine")
        handle.seek(start)
        header = handle.read(32)
        if struct.unpack("<I", header[:4])[0] != 0xFEEDFACF:
            raise ValueError("not a 64-bit Mach-O file")
        count, size = struct.unpack("<II", header[16:24])
        commands = handle.read(size)
    links, rpaths = [], []
    position = 0
    for _ in range(count):
        command, length = struct.unpack_from("<II", commands, position)
        if length < 8 or position + length > len(commands):
            raise ValueError("malformed load command")
        kind = _LOAD_COMMANDS.get(command)
        if kind is not None:
            if length < 12:
                raise ValueError("malformed load command")
            offset = struct.unpack_from("<I", commands, position + 8)[0]
            text = commands[position + offset:position + length].split(b"\0", 1)[0]
            (links if kind == "link" else rpaths).append(text.decode("utf-8", "replace"))
        position += length
    return links, rpaths


def torch_openmp():
    """Which OpenMP runtime PyTorch will load, read from its files. Never imports torch.

    None when this is not a Mac or PyTorch is not installed. Otherwise a mapping:
    ``torch`` (version), ``reference`` (how libtorch_cpu names the runtime, None
    when it links none), ``image`` (the file it resolves to), ``by_install_name``
    (True for an @rpath reference, which a copy already in the process satisfies)
    and ``unread`` (why the files could not be read, when they could not).
    """
    if platform.system() != "Darwin":
        return None
    import importlib.util
    from importlib import metadata

    try:
        spec = importlib.util.find_spec("torch")
    except (ImportError, ValueError):
        spec = None
    if spec is None or not spec.submodule_search_locations:
        return None
    try:
        version = metadata.version("torch")
    except Exception:  # noqa: BLE001 - a version that cannot be read is reported as such
        version = None
    found = {"torch": version, "reference": None, "image": None,
             "by_install_name": False}
    core = os.path.join(list(spec.submodule_search_locations)[0], "lib",
                        "libtorch_cpu.dylib")
    if not os.path.exists(core):
        found["unread"] = "torch/lib/libtorch_cpu.dylib is not present"
        return found
    core = os.path.realpath(core)
    try:
        links, rpaths = macho_links(core)
    except (OSError, ValueError, struct.error) as exc:
        found["unread"] = f"{os.path.basename(core)}: {exc}"
        return found
    named = [link for link in links if OPENMP_FILE.match(os.path.basename(link))]
    if not named:
        return found
    reference = found["reference"] = named[0]
    here = os.path.dirname(core)

    def expand(path):
        return (path.replace("@loader_path", here)
                .replace("@executable_path", os.path.dirname(sys.executable)))

    if reference.startswith("@rpath/"):
        found["by_install_name"] = True
        tail = reference[len("@rpath/"):]
        candidates = [os.path.join(expand(entry), tail) for entry in rpaths]
    else:
        candidates = [expand(reference)]
    for candidate in candidates:
        if os.path.exists(candidate):
            found["image"] = os.path.realpath(candidate)
            break
    if found["image"] is None:
        found["unread"] = f"{reference} does not resolve to a file"
    return found


def loaded_openmp_images():
    """The OpenMP runtime files this process holds, from the dynamic loader's list."""
    import ctypes

    loader = ctypes.CDLL(None)
    loader._dyld_image_count.restype = ctypes.c_uint32
    loader._dyld_get_image_name.restype = ctypes.c_char_p
    loader._dyld_get_image_name.argtypes = [ctypes.c_uint32]
    images = set()
    for index in range(loader._dyld_image_count()):
        name = loader._dyld_get_image_name(index)
        if name and OPENMP_FILE.match(os.path.basename(name.decode("utf-8", "replace"))):
            images.add(os.path.realpath(name.decode("utf-8", "replace")))
    return sorted(images)


def child_openmp() -> int:
    """Import NumPy, then MEEP (never PyTorch) and report the OpenMP runtimes loaded."""
    imported = {}
    for name in ("numpy", "meep"):
        try:
            __import__(name)
            imported[name] = True
        except Exception as exc:  # noqa: BLE001 - reported by the facts run
            imported[name] = repr(exc)
    try:
        images = loaded_openmp_images()
    except Exception as exc:  # noqa: BLE001
        emit({"imported": imported, "error": repr(exc)})
        return 0
    emit({"imported": imported, "images": images})
    return 0


def judge_openmp(torch_side, loaded):
    """``(verdict, line)``: one, two, order or unread, and the line the check prints."""
    def short(path):
        prefix = sys.prefix.rstrip("/") + "/"
        return "<environment>/" + path[len(prefix):] if path.startswith(prefix) else path

    version = torch_side.get("torch") or "(version not read)"
    if torch_side.get("unread"):
        return "unread", f"not read ({torch_side['unread']})"
    image = torch_side.get("image")
    if torch_side.get("reference") is None:
        return "one", (f"PyTorch {version} links none; NumPy and MEEP load "
                       f"{len(loaded)}")
    if not loaded:
        return "one", (f"one: PyTorch {version}'s own ({short(image)}); NumPy and "
                       "MEEP load none")
    if image in loaded:
        return "one", f"one, shared by PyTorch {version}, NumPy and MEEP ({short(image)})"
    others = ", ".join(short(path) for path in loaded)
    if torch_side.get("by_install_name"):
        return "order", (f"two copies on disk: PyTorch {version}'s ({short(image)}) and "
                         f"the one NumPy and MEEP load ({others}); one is loaded only "
                         "when MEEP or NumPy is imported before PyTorch")
    return "two", (f"two: PyTorch {version} loads its own ({short(image)}) whatever "
                   f"was imported before it, and NumPy and MEEP load {others}")


OPENMP_FIX = ("conda install -c conda-forge --override-channels "
              "\"libopenblas=*=*pthreads*\"")


def child_facts() -> int:
    facts = {
        "python": platform.python_version(),
        "system": f"{platform.system()} {platform.machine()}",
        "environment_variables": {
            name: os.environ.get(name)
            for name in ("KMP_DUPLICATE_LIB_OK", "PYTHONPATH", "MEEP_GPU_DISPATCH",
                         "MEEP_GPU_FUSED", "MEEP_GPU_KERNEL_TABLE",
                         "MEEP_GPU_BACKEND_PREFERENCE", "TRITON_LIBCUDA_PATH",
                         "CUDA_VISIBLE_DEVICES", "CC")
        },
        "distributions": distribution_versions(),
    }
    try:
        import numpy

        facts["numpy"] = numpy.__version__
    except Exception as exc:  # noqa: BLE001
        facts["numpy_error"] = repr(exc)
        emit(facts)
        return 0
    try:
        import meep as mp
    except Exception as exc:  # noqa: BLE001
        facts["meep_error"] = repr(exc)
        emit(facts)
        return 0
    mp.verbosity(0)
    facts["meep"] = {
        "version": str(mp.__version__),
        "file": getattr(mp, "__file__", None),
        "single_precision": bool(mp.is_single_precision())
        if hasattr(mp, "is_single_precision") else None,
        "real_bytes": int(mp.get_realnum_size())
        if hasattr(mp, "get_realnum_size") else None,
        "mpi": bool(mp.with_mpi()) if hasattr(mp, "with_mpi") else None,
        "processes": int(mp.count_processors())
        if hasattr(mp, "count_processors") else None,
    }

    missing = {}
    total = 0

    def record(label, target, names):
        nonlocal total
        total += len(names)
        absent = sweep(target, names)
        if absent:
            missing[label] = absent

    try:
        record("meep", mp, MODULE_NAMES)
        import meep.simulation as simulation

        record("meep.simulation", simulation, SIMULATION_MODULE_NAMES)
        record("meep.fields", getattr(mp, "fields", None), FIELDS_CLASS_NAMES)
        record("meep.structure", getattr(mp, "structure", None), STRUCTURE_CLASS_NAMES)
        record("meep.grid_volume", getattr(mp, "grid_volume", None), GRID_VOLUME_NAMES)
        sim, flux = build_simulation(mp)
        record("a constructed Simulation", sim, SIMULATION_NAMES)
        sim.init_sim()
        record("an initialized simulation's fields", sim.fields, FIELDS_INSTANCE_NAMES)
        record("a flux monitor", flux, DFT_FLUX_NAMES)
        chunk = getattr(getattr(flux, "swigobj", None), "E", None)
        if chunk is None:
            total += len(DFT_CHUNK_NAMES)
            missing["a flux monitor's chunk"] = list(DFT_CHUNK_NAMES)
        else:
            record("a flux monitor's chunk", chunk, DFT_CHUNK_NAMES)
        facts["epsilon_dtype"] = str(sim.get_epsilon().dtype)
    except Exception as exc:  # noqa: BLE001
        facts["probe_error"] = repr(exc)
    facts["names"] = {
        "required": total,
        "present": total - sum(len(names) for names in missing.values()),
        "missing": missing,
    }
    facts["constants"] = {
        "checked": len(PINNED_VALUES),
        "mismatched": {
            name: {"expected": want, "found": repr(getattr(mp, name, None))}
            for name, want in PINNED_VALUES.items()
            if not hasattr(mp, name) or int(getattr(mp, name)) != want
        },
    }
    facts["sigma_reader"] = hasattr(getattr(mp, "fields", None),
                                    "get_susceptibility_sigma")

    try:
        import meep_gpu
    except Exception as exc:  # noqa: BLE001
        facts["package_error"] = repr(exc)
        emit(facts)
        return 0
    facts["package"] = {"file": getattr(meep_gpu, "__file__", None)}
    try:
        facts["gpu"] = {
            "route": meep_gpu.available_gpu(),
            "missing": list(meep_gpu.missing_dependencies()),
        }
    except Exception as exc:  # noqa: BLE001
        facts["gpu_error"] = repr(exc)
    emit(facts)
    return 0


def summarize_report(report):
    """The parts of driver.fast_path_report() this check prints."""
    if not isinstance(report, dict):
        return None
    environment = report.get("environment") or {}
    device = environment.get("device") or {}
    composition = report.get("composition") or {}
    return {
        "decision": report.get("decision"),
        "step_path": report.get("step_path"),
        "table": report.get("table"),
        "tables_dispatched": composition.get("tables_dispatched"),
        "refused_because": report.get("refused_because"),
        "torch": environment.get("torch"),
        "torch_certified": environment.get("torch_certified"),
        "metal_frontend": environment.get("metal_frontend"),
        "frontend_certified": environment.get("frontend_certified"),
        "triton": environment.get("triton"),
        "triton_certified": environment.get("triton_certified"),
        "triton_import_error": environment.get("triton_import_error"),
        "cupy": environment.get("backend_version")
        if environment.get("backend") == "cupy" else None,
        "device_name": device.get("name") or device.get("device_name"),
        "compute_capability": device.get("compute_capability"),
        "device_certified": environment.get("device_certified"),
        "device_certified_by_table": environment.get("device_certified_by_table"),
        "certified_compute_capabilities":
            environment.get("validated_compute_capabilities_by_table"),
        "certified_toolchains": environment.get("validated_toolchains"),
    }


def child_leg(leg: str, save: str, gpu_id: int) -> int:
    import numpy as np
    import meep as mp

    mp.verbosity(0)
    sim, flux = build_simulation(mp)
    payload = {"leg": leg}
    if leg == "meep":
        started = time.perf_counter()
        sim.run(until=RUN_UNTIL)
        payload["seconds"] = time.perf_counter() - started
        field = np.asarray(sim.get_array(component=mp.Ez))
        spectrum = np.asarray(mp.get_fluxes(flux), dtype=np.float64)
        payload["meep_time"] = float(sim.meep_time())
    else:
        from meep_gpu import gpu_compatibility, run_on_gpu

        verdict = gpu_compatibility(sim)
        if not verdict.supported:
            payload["error"] = ("the lift refused the check's own simulation: "
                                + "; ".join(str(reason) for reason in verdict.reasons))
            emit(payload)
            return 3
        started = time.perf_counter()
        if leg == "gpu":
            result = run_on_gpu(sim, until=RUN_UNTIL, gpu_id=gpu_id)
        else:
            result = run_on_gpu(sim, until=RUN_UNTIL, prefer_gpu=False)
        payload["seconds"] = time.perf_counter() - started
        try:
            field = np.asarray(result.get_array("Ez"))
            spectrum = np.asarray(result.get_flux_spectrum(flux), dtype=np.float64)
            payload["steps"] = int(result.steps)
            payload["meep_time"] = float(result.meep_time)
            payload["driver_gpu"] = result.driver.gpu
            payload["step_path"] = result.driver.active_step_path
            payload["report"] = summarize_report(result.driver.fast_path_report())
        finally:
            result.close()
    payload["field_dtype"] = str(field.dtype)
    payload["field_shape"] = list(field.shape)
    payload["field_max"] = float(np.abs(field).max())
    payload["finite"] = bool(np.isfinite(field).all() and np.isfinite(spectrum).all())
    np.savez(save, field=field, spectrum=spectrum)
    emit(payload)
    return 0


def child_main(arguments) -> int:
    if arguments.child == "facts":
        return child_facts()
    if arguments.child == "openmp":
        return child_openmp()
    try:
        return child_leg(arguments.child, arguments.save, arguments.gpu_id)
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        emit({"leg": arguments.child, "error": f"{type(exc).__name__}: {exc}"})
        return 3


# --------------------------------------------------------------------------
# The parent process. It imports neither MEEP nor meep_gpu.
# --------------------------------------------------------------------------

class Failure(Exception):
    def __init__(self, reason: str, section: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.section = section


def run_child(kind: str, arguments, extra=()):
    command = [sys.executable, os.path.abspath(__file__), "--child", kind,
               "--gpu-id", str(arguments.gpu_id), *extra]
    try:
        done = subprocess.run(command, capture_output=True, text=True,
                              timeout=LEG_TIMEOUT_S, cwd=tempfile.gettempdir(),
                              check=False)
    except subprocess.TimeoutExpired:
        raise Failure(f"the {kind} run did not finish in {LEG_TIMEOUT_S} seconds.",
                      SECTION_TROUBLE) from None
    payload = None
    for line in done.stdout.splitlines():
        if line.startswith(MARK):
            payload = json.loads(line[len(MARK):])
    if arguments.verbose:
        for stream, text in (("output", done.stdout), ("messages", done.stderr)):
            for line in text.splitlines():
                if line.strip() and not line.startswith(MARK):
                    print(f"      [{kind} {stream}] {line}")
    return payload, done


def explain_crash(kind: str, done) -> Failure:
    """``kind`` names the run: "the first run", "the run on MEEP itself", ..."""
    text = done.stderr + done.stdout
    if "OMP: Error #15" in text:
        return Failure(
            f"{kind} stopped with 'OMP: Error #15': two OpenMP runtimes were "
            "loaded into one process, PyTorch's own and the one NumPy and MEEP load. "
            f"In a conda-forge environment, {OPENMP_FIX} leaves one.",
            SECTION_OPENMP)
    last = [line for line in text.splitlines() if line.strip()]
    detail = last[-1].strip() if last else "no output"
    if done.returncode < 0:
        how = f"was stopped by signal {-done.returncode}"
    else:
        how = f"exited with status {done.returncode}"
    return Failure(f"{kind} {how} before reporting ({detail[:200]}).",
                   SECTION_TROUBLE)


def relative_difference(first, second) -> float:
    import numpy as np

    first = np.asarray(first, dtype=np.complex128 if np.iscomplexobj(first)
                       or np.iscomplexobj(second) else np.float64)
    second = np.asarray(second, dtype=first.dtype)
    if first.shape != second.shape:
        return float("inf")
    scale = float(np.linalg.norm(second.ravel()))
    if scale == 0.0:
        return float("inf")
    return float(np.linalg.norm((first - second).ravel())) / scale


def say(text: str = "") -> None:
    print(text, flush=True)


def describe_build(meep) -> str:
    kind = "MPI build" if meep.get("mpi") else "serial build (no MPI)"
    return f"{kind}, running as {meep.get('processes')} process(es)"


def describe_precision(meep) -> str:
    if meep.get("single_precision") is None:
        return "could not be read"
    word = "single" if meep["single_precision"] else "double"
    return f"{word} precision ({meep.get('real_bytes')}-byte real numbers)"


def mark(value) -> str:
    return {True: "on the certified list", False: "NOT on the certified list"}.get(
        value, "could not be read, so it is not refused")


def launched_processes():
    """The size of the MPI job this process was started in, when a launcher says so.

    Open MPI and MPICH both tell the processes they start how many there are,
    through the environment. Nothing is imported to read it.
    """
    for name in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE"):
        value = os.environ.get(name, "")
        if value.isdigit():
            return int(value)
    return None


def check(arguments) -> int:
    notes = []
    say("meep-gpu installation check")
    say("===========================")
    say()

    launched = launched_processes()
    if launched is not None and launched > 1:
        raise Failure(
            f"this check was started by an MPI launcher as one of {launched} "
            "processes; the package steps one process on one GPU. Start the check "
            "without mpirun.", SECTION_MPI)

    # Read before any process imports PyTorch: a process that loads two OpenMP
    # runtimes is stopped by the second one, so the prediction is made from files
    # and from a process that imports NumPy and MEEP only.
    openmp = None
    torch_side = torch_openmp()
    if torch_side is not None:
        payload, done = run_child("openmp", arguments)
        if payload is None:
            raise explain_crash("the first run", done)
        if "images" in payload:
            openmp = judge_openmp(torch_side, payload["images"])
        else:
            openmp = ("unread", f"not read ({payload.get('error')})")
        if openmp[0] == "two":
            say("OpenMP runtimes, read before PyTorch is imported")
            say(f"   {openmp[1]}")
            say()
            raise Failure(
                f"PyTorch {torch_side.get('torch')} and this environment load two "
                "different OpenMP runtimes, so every process that imports PyTorch "
                "stops with 'OMP: Error #15', whatever the import order. Keep this "
                "PyTorch and take the build of OpenBLAS that loads no OpenMP "
                f"runtime: {OPENMP_FIX}", SECTION_OPENMP)
        if openmp[0] == "order":
            notes.append(
                "PyTorch and this environment each carry an OpenMP runtime. This check "
                "imports MEEP first, so one is loaded; a script that imports torch "
                "before meep or numpy stops with 'OMP: Error #15'. To make the order "
                f"irrelevant: {OPENMP_FIX} (INSTALL.md, \"{SECTION_OPENMP}\").")

    facts, done = run_child("facts", arguments)
    if facts is None:
        raise explain_crash("the first run", done)

    say("1. Python and NumPy")
    say(f"   Python   {facts.get('python')} on {facts.get('system')}")
    if "numpy_error" in facts:
        say(f"   NumPy    not importable: {facts['numpy_error']}")
        raise Failure("NumPy cannot be imported in this environment.", SECTION_NEED)
    say(f"   NumPy    {facts.get('numpy')}")
    variables = facts.get("environment_variables") or {}
    if variables.get("KMP_DUPLICATE_LIB_OK"):
        notes.append("KMP_DUPLICATE_LIB_OK is set in this shell. Unset it: with it "
                     "set, a process that loads two OpenMP runtimes keeps running "
                     f"and can crash later (INSTALL.md, \"{SECTION_OPENMP}\").")
    if variables.get("PYTHONPATH"):
        notes.append("PYTHONPATH is set in this shell, so modules may come from "
                     "outside this environment.")
    say()

    say("2. MEEP")
    if "meep_error" in facts:
        say(f"   not importable: {facts['meep_error']}")
        raise Failure("MEEP cannot be imported by this Python.", SECTION_NEED)
    meep = facts["meep"]
    tested = meep["version"] in TESTED_MEEP_VERSIONS
    releases = " and ".join(TESTED_MEEP_VERSIONS)
    say(f"   version    {meep['version']}"
        + ("  (a tested version)" if tested
           else f"  (NOT a tested version; {releases} are)"))
    say(f"   build      {describe_build(meep)}")
    say(f"   precision  {describe_precision(meep)}")
    say(f"   location   {meep.get('file')}")
    if not tested:
        notes.append(f"MEEP {meep['version']} has not been tested with this package; "
                     f"{releases} have (INSTALL.md, \"{SECTION_TESTED}\").")
    if meep.get("single_precision") is False:
        notes.append("This MEEP is a double-precision build. The engine steps single "
                     "precision, so results come back as single-precision arrays "
                     f"(INSTALL.md, \"{SECTION_PRECISION}\").")
    if (meep.get("processes") or 1) > 1:
        raise Failure(
            f"this process is one of {meep['processes']} MPI processes; the package "
            "steps one process on one GPU. Start the script without mpirun.",
            SECTION_MPI)
    names = facts.get("names") or {}
    constants = facts.get("constants") or {}
    if "probe_error" in facts:
        say(f"   names the lift reads: the probe stopped: {facts['probe_error']}")
        raise Failure("this MEEP could not build and initialize the check's "
                      "simulation.", SECTION_TESTED)
    say(f"   names the lift reads      {names.get('present')} of "
        f"{names.get('required')} present")
    wrong = constants.get("mismatched") or {}
    say(f"   constants the lift relies on  "
        f"{constants.get('checked', 0) - len(wrong)} of {constants.get('checked')} "
        "have the expected value")
    say("   optional sigma-reader patch   "
        + ("present" if facts.get("sigma_reader") else "absent (stock MEEP; expected)"))
    if names.get("missing"):
        for where, absent in names["missing"].items():
            say(f"      missing from {where}: {', '.join(absent)}")
        count = names["required"] - names["present"]
        raise Failure(f"this MEEP lacks {count} of {names['required']} names the "
                      "lift reads.", SECTION_TESTED)
    if wrong:
        for name, pair in wrong.items():
            say(f"      {name}: expected {pair['expected']}, found {pair['found']}")
        raise Failure(f"{len(wrong)} of {constants['checked']} MEEP constants do not "
                      "have the value the lift relies on.", SECTION_TESTED)
    say()

    say("3. meep-gpu")
    if "package_error" in facts:
        say(f"   not importable: {facts['package_error']}")
        raise Failure("meep_gpu cannot be imported by this Python; install this "
                      "package into the environment that holds MEEP.",
                      SECTION_EXISTING)
    distributions = facts.get("distributions") or {}
    say(f"   version    {distributions.get('meep-gpu') or 'not installed as a distribution'}")
    say(f"   location   {facts['package'].get('file')}")
    say()

    say("4. GPU route of this host")
    if "gpu_error" in facts:
        raise Failure(f"the GPU probe raised: {facts['gpu_error']}", SECTION_TROUBLE)
    route = facts["gpu"]["route"]
    lacking = facts["gpu"]["missing"]
    words = {"cuda": "NVIDIA GPU, through CuPy", "metal": "Apple GPU, through PyTorch"}
    if route is None:
        say("   none: a default run (prefer_gpu=True) raises on this host")
        say(f"   what the GPU route of this platform lacks: {', '.join(lacking) or '-'}")
    else:
        say(f"   {words.get(route, route)}")
    for name in GPU_DISTRIBUTIONS:
        if distributions.get(name):
            say(f"   installed  {name} {distributions[name]}")
    if openmp is not None:
        say(f"   OpenMP runtimes  {openmp[1]}")
    cupies = [name for name in GPU_DISTRIBUTIONS
              if name.startswith("cupy") and distributions.get(name)]
    if len(cupies) > 1:
        raise Failure(f"{len(cupies)} CuPy packages are installed "
                      f"({', '.join(cupies)}); only one may be.", SECTION_TROUBLE)
    if route is None and arguments.require_gpu:
        raise Failure("this host has no GPU route and --require-gpu was given "
                      f"(missing: {', '.join(lacking) or 'unknown'}).", SECTION_TROUBLE)
    say()

    say("5. One small simulation, stepped in separate processes")
    say("   24 x 24 x 24 cells, a dielectric sphere, absorbing boundaries, one flux")
    say(f"   monitor, run to MEEP time {RUN_UNTIL:g}")
    legs = (["gpu"] if route else []) + ["reference", "meep"]
    labels = {"gpu": "the default run (prefer_gpu=True)",
              "reference": "the NumPy reference", "meep": "MEEP itself"}
    #: What section 5 prints while a leg runs, and names in a failure.
    places = {"gpu": "with the default, prefer_gpu=True (this host's GPU route)",
              "reference": "on the NumPy reference", "meep": "on MEEP itself"}
    results = {}
    with tempfile.TemporaryDirectory(prefix="meep-gpu-check-") as scratch:
        for leg in legs:
            hint = ("  (if compiled kernels serve it, the first run compiles them; allow a few minutes)"
                    if leg == "gpu" else "")
            say(f"   running {places[leg]} ...{hint}")
            path = os.path.join(scratch, f"{leg}.npz")
            payload, done = run_child(leg, arguments, ("--save", path))
            if payload is None:
                raise explain_crash(f"the run {places[leg]}", done)
            if payload.get("error"):
                for line in done.stderr.splitlines()[-12:]:
                    say(f"      {line}")
                section = SECTION_TROUBLE
                raise Failure(f"the run {places[leg]} raised: "
                              f"{payload['error'][:300]}", section)
            payload["announced"] = [line for line in done.stderr.splitlines()
                                    if line.startswith("meep_gpu:")]
            payload["libcuda_link_error"] = [
                line.strip() for line in done.stderr.splitlines()
                if LIBCUDA_LINK_ERROR.search(line)][:1]
            results[leg] = payload
            steps = f"{payload['steps']} steps, " if "steps" in payload else ""
            say(f"      finished: {steps}{payload['seconds']:.1f} s, field stored as "
                f"{payload['field_dtype']}")
            if not payload.get("finite"):
                raise Failure(f"the run {places[leg]} produced values that are "
                              "not finite.", SECTION_TROUBLE)

        import numpy as np

        arrays = {leg: np.load(os.path.join(scratch, f"{leg}.npz")) for leg in legs}
        comparisons = []
        if route:
            comparisons.append(("gpu", "reference"))
        comparisons.append(("reference", "meep"))
        differences = {}
        for first, second in comparisons:
            differences[(first, second)] = (
                relative_difference(arrays[first]["field"], arrays[second]["field"]),
                relative_difference(arrays[first]["spectrum"],
                                    arrays[second]["spectrum"]),
            )
    say()

    say("6. Result")
    kernels_expected = False
    if route:
        gpu = results["gpu"]
        report = gpu.get("report") or {}
        say(f"   GPU route        {words.get(route, route)}")
        if route == "metal":
            say(f"   PyTorch          {report.get('torch')}: "
                f"{mark(report.get('torch_certified'))}")
            say(f"   Metal frontend   {report.get('metal_frontend')}: "
                f"{mark(report.get('frontend_certified'))}")
            certified = (report.get("torch_certified") is not False
                         and report.get("frontend_certified") is not False)
        else:
            by_table = report.get("device_certified_by_table") or {}
            say(f"   device           {report.get('device_name') or 'name not read'}, "
                f"compute capability {report.get('compute_capability')}")
            say(f"      for the Triton kernels: {mark(by_table.get('triton'))}")
            say(f"      for the CUDA kernels:   {mark(by_table.get('cuda'))}")
            say(f"   CuPy             {report.get('cupy')}")
            if report.get("triton") is None:
                say("   Triton           not importable"
                    + (f": {report.get('triton_import_error')}"
                       if report.get("triton_import_error") else ""))
            else:
                say(f"   Triton           {report.get('triton')}: "
                    f"{mark(report.get('triton_certified'))}")
            certified = any(value is not False for value in by_table.values()) \
                if by_table else report.get("device_certified") is not False
        served = gpu.get("step_path") == "fused"
        tables = report.get("tables_dispatched") or (
            [report.get("table")] if report.get("table") else [])
        if served:
            say("   compiled kernels served the GPU run: yes"
                + (f" (kernel table: {', '.join(str(t) for t in tables)})"
                   if tables else ""))
        else:
            say("   compiled kernels served the GPU run: NO. The run took the array "
                "path.")
            say("      the default run stepped on "
                + ("NumPy, on the host CPU: the Apple GPU was not used"
                   if route == "metal" else "CuPy, on this host's GPU"))
            if report.get("refused_because"):
                say(f"      reason given by the package: "
                    f"{str(report['refused_because'])[:600]}")
        for line in gpu.get("announced") or []:
            say(f"      {line[:600]}")
        switched_off = (variables.get("MEEP_GPU_DISPATCH") == "0"
                        or variables.get("MEEP_GPU_FUSED") == "0")
        kernels_expected = certified and not served and not switched_off
        if not served and switched_off:
            notes.append("Kernel dispatch is switched off by MEEP_GPU_DISPATCH=0 or "
                         "MEEP_GPU_FUSED=0 in this shell.")
        if not served and not certified:
            notes.append("This device or toolchain is outside the certified list, so "
                         "the default run took the array path"
                         + (", NumPy on the host CPU" if route == "metal" else "")
                         + f" (INSTALL.md, \"{SECTION_UNCERTIFIED}\").")

    failures = []
    for (first, second), (field, spectrum) in differences.items():
        within = field <= AGREEMENT_BAND and spectrum <= AGREEMENT_BAND
        say(f"   {labels[first]} against {labels[second]}:")
        say(f"      final field differs by {field:.3e}, flux spectrum by "
            f"{spectrum:.3e} (relative); accepted up to {AGREEMENT_BAND:.0e}: "
            + ("agree" if within else "DO NOT AGREE"))
        if not within:
            failures.append((first, second))
    say()
    for note in notes:
        say(f"NOTE: {note}")
    if failures:
        first, second = failures[0]
        section = SECTION_PRECISION if "meep" in (first, second) else SECTION_TROUBLE
        raise Failure(f"{labels[first]} and {labels[second]} do not agree within "
                      f"{AGREEMENT_BAND:.0e}.", section)
    if kernels_expected and (results["gpu"].get("libcuda_link_error") or []):
        raise Failure("compiled kernels did not serve the run: Triton could not link "
                      "against the NVIDIA driver library ("
                      + results["gpu"]["libcuda_link_error"][0][:120]
                      + "). Do step 3 of Route 1 in this shell.", SECTION_TROUBLE)
    if kernels_expected:
        raise Failure("the device and toolchain are on the certified list, but "
                      "compiled kernels did not serve the run; it took the array "
                      "path.", SECTION_TROUBLE)
    if route and served:
        say("OK: MEEP and meep-gpu work together on this host's GPU.")
    elif route == "metal":
        say("OK: MEEP and meep-gpu work together. The default run took the array "
            "path, NumPy on the host CPU; the Apple GPU was not used.")
    elif route:
        say("OK: MEEP and meep-gpu work together on this host's GPU, on the array "
            "path (CuPy); compiled kernels did not serve the run.")
    else:
        say("OK: MEEP and meep-gpu work together on the NumPy reference. This host "
            "has no GPU route; pass prefer_gpu=False to every call.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check that MEEP and meep-gpu are installed and work together.")
    parser.add_argument("--require-gpu", action="store_true",
                        help="fail when this host has no GPU route")
    parser.add_argument("--gpu-id", type=int, default=0,
                        help="the CUDA device to use (default 0)")
    parser.add_argument("--verbose", action="store_true",
                        help="also print what each run wrote")
    parser.add_argument("--child", choices=("openmp", "facts", "gpu", "reference", "meep"),
                        help=argparse.SUPPRESS)
    parser.add_argument("--save", help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.child:
        return child_main(arguments)
    try:
        return check(arguments)
    except Failure as failure:
        reason = " ".join(failure.reason.split())
        if not reason.endswith((".", "!", "?")):
            reason += "."
        say()
        say(f"FAILED: {reason} Read INSTALL.md, \"{failure.section}\".")
        return 1


if __name__ == "__main__":
    sys.exit(main())
