#!/usr/bin/env python3
"""Run the checkout's own Triton fleet driver, with per-gate timeouts that can only RISE.

The round runs ``parity/meep_gpu/drive_triton_weld_gates.py`` exactly as the 0.9.2
rounds did (its argv, its per-gate environment, its artifact layout, its
``campaign.json``), with one deviation, recorded in the driver log's first line:
``--timeout-floor gate=seconds`` raises that gate's timeout to at least ``seconds``.
It never lowers one. MEASURED: on the 1.0 tree a pre-certification pass timed out
``conductivity`` at its 3,600 s timeout and ``symmetry`` at 5,400 s with only two
gates on the card (0.9.x sequential fleets took 620-680 s and 330-390 s). The driver
itself carries 10,800 s for both since entry 99; the round's defaults below hold every
gate's timeout at 3x or more the larger of its two measured walls (the 1.0 round's
CERT-3 and the final pre-certification pass), which raises four gates.
A timed-out gate is a gate with no verdict; a longer timeout costs only wall time on a
gate that hangs.

A second deviation, only under ``--policy flush`` and only with
``--host-import-wrapper W``: each gate process is started as ``python -u W <repo>
<gate script> <args>``, which imports MEEP and then runs the gate's own ``__main__``
with its own argv. MEASURED: on the 1.0 tree a flush fleet gate that installs its
policy before anything imports MEEP refuses at startup ("the 'flush' subnormal policy
is UNATTAINABLE for the 'host' executor: MEEP is not imported in this process"),
because the host FTZ/DAZ bits are reachable only through MEEP. ``W`` must be a copy
OUTSIDE the checkout, so gate provenance never records it among the gate's imports.

THE ENVIRONMENT. The fleet driver builds each gate's environment from this process's
(``drive_triton_weld_gates.gate_environment``), so it is checked here before the driver is
imported (:func:`environment_problems`), and the launch is REFUSED by name when this
interpreter enables the user site directory (the driver is imported in process, and every
gate would inherit the setting) or when the environment carries a package switch
(``MEEP_GPU_*``, ``TRIDENT_FDTD_*``) other than the corpus root the round names: the driver
sets each gate's own. ``run_lane.py`` starts this launcher scrubbed and with
``PYTHONNOUSERSITE=1``, so in the round nothing is refused. Then ``PYTHONNOUSERSITE=1`` and
an empty ``CUPY_ACCELERATORS`` are set for every gate, as every CUDA child of the round has
them (``RUNBOOK.md``, section 3), and the Triton launcher's toolchain
(``legs.launcher_toolchain``): ``CC`` and ``TRITON_LIBCUDA_PATH`` as the caller set them
(``run_lane.py`` sets both), else the compiler and the libcuda link of the reference
environment this interpreter runs in. A reference environment without its compiler, or
without the libcuda link its build recorded, is refused by name before the driver starts:
Triton would compile every gate's launchers with the host's ``gcc``, and the fleet driver's
own default stub (``~/triton_libcuda_stub``) is not part of the environment.

usage: fleet_launch.py --repo R [--timeout-floor gate=s ...] [--host-import-wrapper W]
                       -- <fleet driver args...>
"""
from __future__ import annotations

import argparse
import os
import site
import sys
from pathlib import Path
from typing import List, Mapping

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import legs  # noqa: E402 - beside this file; the toolchain rule lives there

#: The round's default floors (seconds): 3x the larger of each gate's walls in the 1.0
#: round's CERT-3 NVIDIA (one gate at a time on a card) and in the final
#: pre-certification pass (four gates sharing a card), rounded up to a multiple of 300 s,
#: for the four gates whose driver timeout falls below that: ``conductivity`` and
#: ``symmetry`` 10,800 -> 15,900 s (walls 5,261 s and 5,275 s), ``folded_complex``
#: 10,800 -> 13,500 s (4,463 s), ``special_kz`` 7,200 -> 12,600 s (4,109 s). Every other
#: gate's driver timeout is already 3x its larger wall. A floor never lowers one.
DEFAULT_FLOORS = {"conductivity": 15900, "symmetry": 15900, "folded_complex": 13500,
                  "special_kz": 12600}

#: A package switch in the launcher's environment is refused by name: the fleet driver
#: sets each gate's own, and a gate inherits everything else.
SHAPING_PREFIXES = ("MEEP_GPU_", "TRIDENT_FDTD_")
#: The one package variable the round names and passes to the fleet (``run_lane.py``).
ROUND_NAMED = ("MEEP_GPU_CORPUS_ROOT",)
#: Set for every gate the fleet starts.
GATE_EXPORTS = {"PYTHONNOUSERSITE": "1", "CUPY_ACCELERATORS": ""}


def environment_problems(environ: Mapping[str, str], user_site_enabled: bool) -> List[str]:
    """Why the fleet may not start from this environment, each by name."""
    problems = []
    if user_site_enabled:
        problems.append("this interpreter enables the user site directory "
                        "(site.ENABLE_USER_SITE); the fleet driver is imported in this process "
                        "and every gate inherits it: start the launcher with "
                        "PYTHONNOUSERSITE=1, as run_lane.py does")
    exported = sorted(name for name in environ
                      if name.startswith(SHAPING_PREFIXES) and name not in ROUND_NAMED)
    if exported:
        problems.append(f"{exported} exported to the fleet; the fleet driver sets each gate's "
                        f"own package switches and a gate would inherit these")
    return problems


def parse_floors(items):
    floors = dict(DEFAULT_FLOORS)
    for item in items or ():
        name, _, value = item.partition("=")
        if not name or not value.isdigit():
            raise SystemExit(f"--timeout-floor takes gate=seconds, got {item!r}")
        floors[name] = int(value)
    return floors


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        raise SystemExit("usage: fleet_launch.py --repo R [--timeout-floor g=s] -- <driver args>")
    split = argv.index("--")
    ours, theirs = argv[:split], argv[split + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--timeout-floor", action="append", default=[])
    ap.add_argument("--host-import-wrapper", default="")
    args = ap.parse_args(ours)
    problems = environment_problems(os.environ, bool(site.ENABLE_USER_SITE))
    toolchain = legs.launcher_toolchain(
        sys.prefix if sys.platform.startswith(legs.LAUNCHER_PLATFORMS) else None,
        dict(os.environ))
    problems += toolchain["problems"]
    if problems:
        print("fleet_launch: REFUSED: " + "; ".join(problems), flush=True)
        return 3
    os.environ.update(GATE_EXPORTS)
    legs.launcher_environment(os.environ, toolchain)
    repo = Path(args.repo).resolve()
    harness = repo / "parity" / "meep_gpu"
    sys.path[:0] = [str(repo), str(harness)]
    import drive_triton_weld_gates as fleet  # noqa: PLC0415 - the checkout's own driver

    fleet.log("[round] fleet_launch: user site directory off; every gate starts with "
              + ", ".join(f"{k}={v!r}" for k, v in sorted(GATE_EXPORTS.items())))
    fleet.log(f"[round] fleet_launch: {legs.launcher_line(toolchain)}")
    raised = {}
    for name, floor in parse_floors(args.timeout_floor).items():
        gate = fleet.GATES.get(name)
        if gate is not None and gate.timeout < floor:
            raised[name] = (gate.timeout, floor)
            gate.timeout = floor
    fleet.log(f"[round] fleet_launch: timeouts raised (never lowered): "
              + (", ".join(f"{k} {a}->{b} s" for k, (a, b) in sorted(raised.items()))
                 or "none"))
    policy = theirs[theirs.index("--policy") + 1] if "--policy" in theirs else "keep"
    if policy == "flush" and args.host_import_wrapper:
        wrapper = str(Path(args.host_import_wrapper).resolve())
        if str(repo) in wrapper:
            raise SystemExit(f"{wrapper} is inside the checkout; gate provenance would "
                             f"record it")
        fleet.subprocess = _HostImportSubprocess(fleet.subprocess, wrapper, repo, harness)
        fleet.log(f"[round] fleet_launch: flush campaign; every gate starts through "
                  f"{wrapper} (MEEP imported first, for the host half of the policy)")
    return fleet.main(theirs)


class _HostImportSubprocess:
    """``subprocess`` with ``Popen`` rewriting a gate's argv to start through the wrapper."""

    def __init__(self, real, wrapper: str, repo: Path, harness: Path):
        self._real, self._wrapper, self._repo, self._harness = real, wrapper, repo, harness

    def __getattr__(self, name):
        return getattr(self._real, name)

    def Popen(self, argv, *a, **k):  # noqa: N802 - the name the driver calls
        argv = list(argv)
        if (len(argv) >= 3 and argv[1] == "-u" and argv[2].endswith(".py")
                and Path(argv[2]).resolve().parent == self._harness.resolve()):
            argv = [argv[0], "-u", self._wrapper, str(self._repo), *argv[2:]]
        return self._real.Popen(argv, *a, **k)


if __name__ == "__main__":
    sys.exit(main())
