"""Record the Metal environment a certification campaign runs in, and read it back.

A Metal weld is certified for one ENVIRONMENT: the GPU architecture Metal compiles
for (``MTLDevice.architecture.name``), the torch that supplies ``compile_shader``,
and the Metal frontend of the macOS build. ``metal_dispatch`` reads all three off
each cited weld's ``host`` line, so the line has to name them, and it can only name
what was read on the machine that ran the gates.

``recut_metal_gates.sh`` writes this record twice, before its first gate and after
its last, into ``results/metal_environment_<stamp>/``. ``rebind_metal_welds.py`` and
``mint_metal_weld.py`` refuse a campaign whose two records are missing, disagree, or
leave a fact unread, and write the weld's ``host`` line from them with
:func:`host_line`.

    python parity/meep_gpu/metal_environment.py --write results/metal_environment_<stamp>/start.json
    python parity/meep_gpu/metal_environment.py --check results/metal_environment_<stamp>
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Optional

_HERE = Path(__file__).resolve().parent


def _find_api_root(start: Path) -> Path:
    """BY NAME, never by ``parents[N]``: a moved harness must not resolve a wrong root."""
    for candidate in (start, *start.parents):
        if (candidate / "parity" / "meep_gpu").is_dir() and (candidate / "meep_gpu").is_dir():
            return candidate
    raise SystemExit(f"cannot locate the repository root above {start}")


_API = _find_api_root(_HERE)
if str(_API) not in sys.path:
    sys.path.insert(0, str(_API))

from meep_gpu.metal_kernels import device  # noqa: E402

RESULTS = _HERE / "results"
#: The facts a weld's certification depends on; a campaign record must read every one.
FACTS = ("architecture", "torch", "metal_frontend")
#: Read and recorded because it recompiles every Metal source in fast-math mode.
FAST_MATH = "PYTORCH_MPS_FAST_MATH"


def _sysctl(name: str) -> Optional[str]:
    try:
        import ctypes  # noqa: PLC0415

        libc = ctypes.CDLL(None)
        call = libc.sysctlbyname
        call.argtypes = [ctypes.c_char_p, ctypes.c_void_p,
                         ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p,
                         ctypes.c_size_t]
        call.restype = ctypes.c_int
        size = ctypes.c_size_t(0)
        if call(name.encode(), None, ctypes.byref(size), None, 0) != 0 or not size.value:
            return None
        buffer = ctypes.create_string_buffer(size.value)
        if call(name.encode(), buffer, ctypes.byref(size), None, 0) != 0:
            return None
        return buffer.value.decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - an unread fact is recorded as None
        return None


def read() -> Dict[str, Any]:
    """This host's Metal environment. Never raises; an unread fact is ``None``.

    torch's version is read from its installed metadata, so recording it does not
    import torch (and with it a second OpenMP runtime) into the recording process.
    """
    gpu = device.apple_gpu_identity()
    try:
        torch_version: Optional[str] = metadata.version("torch")
    except metadata.PackageNotFoundError:
        torch_version = None
    return {
        "architecture": gpu.get("architecture"),
        "device_name": gpu.get("name"),
        "architecture_error": gpu.get("error"),
        "torch": torch_version,
        "metal_frontend": device.metal_frontend_version(),
        "macos": _sysctl("kern.osproductversion"),
        "macos_build": _sysctl("kern.osversion"),
        "machine": platform.machine(),
        FAST_MATH: os.environ.get(FAST_MATH),
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def host_line(environment: Dict[str, Any]) -> str:
    """The ``host`` line a Metal weld records: every fact ``metal_dispatch`` parses."""
    return (f"this machine: Apple MPS device {environment['architecture']}, "
            f"torch {environment['torch']}, {environment['metal_frontend']}")


def campaign(stamp: str) -> Dict[str, Any]:
    """The environment a campaign ran in, from its two records. Raises SystemExit.

    Refuses a campaign with a missing record, a fact either record could not read,
    a fact that changed between the first gate and the last, or fast-math set.
    """
    directory = RESULTS / f"metal_environment_{stamp}"
    records = {}
    for name in ("start", "end"):
        path = directory / f"{name}.json"
        if not path.is_file():
            raise SystemExit(
                f"{path} is missing: the campaign's Metal environment was not recorded, "
                f"so no weld can say which GPU, torch and Metal frontend it ran on. "
                f"recut_metal_gates.sh writes it when given a stamp")
        records[name] = json.loads(path.read_text(encoding="utf-8"))
    for name, record in records.items():
        unread = [fact for fact in FACTS if not record.get(fact)]
        if unread:
            raise SystemExit(f"{directory}/{name}.json could not read {unread} "
                             f"({record.get('architecture_error') or 'no reason given'})")
        if record.get(FAST_MATH) not in (None, "0"):
            raise SystemExit(f"{directory}/{name}.json records {FAST_MATH}="
                             f"{record[FAST_MATH]!r}: the gates ran fast-math sources")
    moved = [fact for fact in FACTS if records["start"][fact] != records["end"][fact]]
    if moved:
        raise SystemExit(f"{moved} changed between the first gate and the last "
                         f"({ {fact: (records['start'][fact], records['end'][fact]) for fact in moved} })")
    return records["start"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", type=Path, help="record this host's environment here")
    action.add_argument("--check", metavar="STAMP",
                        help="read a campaign's two records and print its host line")
    args = parser.parse_args(argv)
    if args.write is not None:
        if args.write.exists():
            raise SystemExit(f"{args.write} exists; a campaign's record is written once")
        args.write.parent.mkdir(parents=True, exist_ok=True)
        environment = read()
        args.write.write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n",
                              encoding="utf-8")
        print(json.dumps(environment, sort_keys=True), flush=True)
        return 0
    print(host_line(campaign(args.check)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
