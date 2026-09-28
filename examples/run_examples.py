"""Run every example leg by leg, compare the legs, and check the expected output.

    python examples/run_examples.py                 run and compare
    python examples/run_examples.py --check         also compare with examples/expected/
    python examples/run_examples.py --only sphere_flux_3d
    python examples/run_examples.py --record        rewrite examples/expected/ (maintainers)

``quickstart.py``, the README's first run, is run first on a host with a GPU
route, and its own lines are recorded and checked like the rest.

Each leg runs in its own process (see common.py). Per example, in order:

    gpu         this host's GPU; skipped, and said so, on a host without one
    array       the array path of the same GPU route, under the subnormal policy the
                gpu leg reported
    reference   the NumPy reference, under the host's default policy
    meep        MEEP itself

and three comparisons:

    gpu against array        every byte of the field and of each spectrum. The package's
                             contract is identity; a difference fails the run.
    gpu against reference    relative difference, reported. The two are identical where
                             the policies coincide and differ at single-precision
                             rounding level where they do not.
    reference against MEEP   relative difference against the example's stated band
                             (BANDS below). The bands were set from runs against a
                             single-precision build of MEEP 1.33.0; with any other
                             build the figure is reported and not judged.

Exit status: 0 when every judged comparison holds, 1 otherwise.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXPECTED = HERE / "expected"
#: Largest relative difference from MEEP that passes, per example: the L2 norm of
#: the difference over the L2 norm of MEEP's own result, for the final field and for
#: each flux spectrum. The engine and the MEEP build compared against both step
#: single precision, and the two differ by a few parts in a million OF THE FIELD'S
#: PEAK. The dispersive example ends after the pulse has left the cell, when the
#: field that remains is 0.3 % of its peak, so the same absolute difference is a
#: larger fraction of what is left; its band is wider for that reason and no other.
BANDS = {
    "sphere_flux_3d": 1.0e-4,
    "slab_flux_2d": 1.0e-4,
    "dispersive_slab_2d": 2.0e-3,
}
EXAMPLES = tuple(BANDS)
DISPATCH_ENABLE = "MEEP_GPU_DISPATCH"
PREFIXES = ("[result]", "[host]", "[timing]")
NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def say(line: str = "") -> None:
    print(line, flush=True)


def run_leg(name: str, leg: str, saved: Path, *, policy: str = "default") -> list[str]:
    """One leg in its own process. Returns its prefixed lines; raises if it failed."""
    command = [sys.executable, str(HERE / f"{name}.py"), "--leg", leg,
               "--save", str(saved), "--policy", policy]
    environment = dict(os.environ)
    if leg != "array":
        # The gpu leg runs with the shipped default. A value left in the caller's
        # environment would make this a different run from the one documented.
        environment.pop(DISPATCH_ENABLE, None)
    started = time.perf_counter()
    done = subprocess.run(command, cwd=str(HERE), env=environment, text=True,
                          capture_output=True)
    elapsed = time.perf_counter() - started
    lines = [line for line in done.stdout.splitlines() if line.startswith(PREFIXES)]
    # The package states on stderr which path served the run. It is host text.
    lines += [f"[host] {line}" for line in done.stderr.splitlines()
              if line.startswith("meep_gpu:")]
    if done.returncode != 0:
        sys.stdout.write(done.stdout)
        sys.stderr.write(done.stderr)
        raise SystemExit(f"{name} --leg {leg} exited with status {done.returncode}")
    step = host_value(lines, "step path")
    # Which path stepped the leg, so a gpu leg that took the array path says so.
    say(f"  {leg:<9} finished in {elapsed:6.1f} s"
        + (f"  (step path: {step})" if step else ""))
    return lines


def host_value(lines: list[str], key: str) -> str | None:
    for line in lines:
        if line.startswith(f"[host] {key}: "):
            return line.split(": ", 1)[1].strip()
    return None


def relative_difference(a: np.ndarray, b: np.ndarray) -> float:
    """The L2 norm of the difference over the L2 norm of ``b``, in double precision."""
    a = np.asarray(a, dtype=np.complex128 if np.iscomplexobj(a) else np.float64)
    b = np.asarray(b, dtype=np.complex128 if np.iscomplexobj(b) else np.float64)
    if a.shape != b.shape:
        return float("inf")
    scale = float(np.linalg.norm(b.ravel()))
    return float(np.linalg.norm((a - b).ravel())) / scale if scale else float("inf")


def compare_bytes(left: Path, right: Path) -> tuple[bool, list[str]]:
    a, b = np.load(left), np.load(right)
    detail, identical = [], True
    for key in a.files:
        same = (a[key].dtype == b[key].dtype and a[key].shape == b[key].shape
                and a[key].tobytes() == b[key].tobytes())
        identical = identical and same
        detail.append(f"{key} {'identical' if same else 'DIFFERENT'}")
    return identical, detail


def compare_relative(left: Path, right: Path) -> tuple[float, float]:
    """(field, worst spectrum) relative differences of ``left`` against ``right``."""
    a, b = np.load(left), np.load(right)
    field = relative_difference(a["field"], b["field"])
    spectra = [relative_difference(a[key], b[key]) for key in a.files if key != "field"]
    return field, (max(spectra) if spectra else 0.0)


def result_blocks(lines: list[str]) -> dict[str, list[str]]:
    """The [result] lines of each leg, keyed by the leg's name."""
    blocks: dict[str, list[str]] = {}
    pending: list[str] = []
    current: list[str] | None = None
    for line in lines:
        if not line.startswith("[result]"):
            continue
        if line.startswith("[result] example: "):
            pending, current = [line], None
        elif line.startswith("[result] leg: "):
            current = blocks.setdefault(line.split(": ", 1)[1].strip(), [])
            current.extend(pending + [line])
        elif current is not None:
            current.append(line)
    return blocks


def numbers_after_label(line: str) -> np.ndarray:
    return np.array([float(x) for x in NUMBER.findall(line.split(":", 1)[-1])])


def check_against_expected(name: str, produced: list[str], tolerance: float) -> list[str]:
    """Compare [result] lines with the recorded ones, leg by leg; returns mismatches.

    Only the legs this host ran are compared, so a host with no GPU checks its
    reference and meep legs against a recording that carries four. Text must match
    exactly. Numbers on a line must agree within ``tolerance`` of the largest
    magnitude on that line, which keeps a spectrum's near-zero entries from being
    judged against themselves.
    """
    path = EXPECTED / f"{name}.txt"
    if not path.exists():
        return [f"{path.name} is missing"]
    recorded = result_blocks(path.read_text(encoding="utf-8").splitlines())
    problems = []
    for leg, ours in result_blocks(produced).items():
        theirs = recorded.get(leg)
        if theirs is None:
            problems.append(f"{leg}: no recorded output for this leg")
            continue
        if len(theirs) != len(ours):
            problems.append(f"{leg}: {len(ours)} [result] lines produced, "
                            f"{len(theirs)} recorded")
        for want, got in zip(theirs, ours):
            if NUMBER.sub("#", want) != NUMBER.sub("#", got):
                problems.append(f"{leg}: text differs\n    recorded: {want}\n"
                                f"    produced: {got}")
                continue
            wanted, found = numbers_after_label(want), numbers_after_label(got)
            if wanted.size and not np.allclose(
                    found, wanted, rtol=0.0,
                    atol=tolerance * float(np.abs(wanted).max() or 1.0)):
                problems.append(f"{leg}: outside {tolerance:g}\n    recorded: {want}\n"
                                f"    produced: {got}")
    return problems


#: The lines quickstart.py prints itself, by their first word. Everything else on
#: its standard output is MEEP's own initialization report.
QUICKSTART_LINES = ("supported:", "steps:", "step path:", "flux:", "max |Ez|:")
#: Of those, the one that describes the host and is not compared.
QUICKSTART_HOST_LINES = ("step path:",)


def run_quickstart() -> list[str]:
    """quickstart.py in its own process; returns the lines the script printed."""
    environment = dict(os.environ)
    environment.pop(DISPATCH_ENABLE, None)
    started = time.perf_counter()
    done = subprocess.run([sys.executable, str(HERE / "quickstart.py")], cwd=str(HERE),
                          env=environment, text=True, capture_output=True)
    elapsed = time.perf_counter() - started
    if done.returncode != 0:
        sys.stdout.write(done.stdout)
        sys.stderr.write(done.stderr)
        raise SystemExit(f"quickstart.py exited with status {done.returncode}")
    say(f"  finished in {elapsed:6.1f} s")
    return [line for line in done.stdout.splitlines()
            if line.startswith(QUICKSTART_LINES)]


def check_quickstart(produced: list[str], tolerance: float) -> list[str]:
    path = EXPECTED / "quickstart.txt"
    if not path.exists():
        return [f"{path.name} is missing"]
    recorded = [line for line in path.read_text(encoding="utf-8").splitlines()
                if line.startswith(QUICKSTART_LINES)]
    problems = []
    if len(recorded) != len(produced):
        problems.append(f"{len(produced)} lines produced, {len(recorded)} recorded")
    for want, got in zip(recorded, produced):
        if want.startswith(QUICKSTART_HOST_LINES):
            continue
        if NUMBER.sub("#", want) != NUMBER.sub("#", got):
            problems.append(f"text differs\n    recorded: {want}\n    produced: {got}")
            continue
        wanted, found = numbers_after_label(want), numbers_after_label(got)
        if wanted.size and not np.allclose(
                found, wanted, rtol=0.0,
                atol=tolerance * float(np.abs(wanted).max() or 1.0)):
            problems.append(f"outside {tolerance:g}\n    recorded: {want}\n"
                            f"    produced: {got}")
    return problems


def header() -> list[str]:
    import platform

    import meep as mp
    import meep_gpu

    lines = [
        "# Recorded output of examples/run_examples.py --record.",
        f"# Host: {platform.system()} {platform.machine()}, "
        f"python {platform.python_version()}, numpy {np.__version__}.",
        f"# MEEP {mp.__version__}, single precision: {bool(mp.is_single_precision())}.",
        f"# GPU route: {meep_gpu.available_gpu()}.",
        "# [result] lines are compared by --check. [host] and [timing] lines are not:",
        "# they describe the recording host, and the wall times were taken on a",
        "# loaded machine and are not timings of record.",
    ]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", action="append", choices=EXAMPLES, default=None)
    parser.add_argument("--check", action="store_true",
                        help="compare [result] lines with examples/expected/")
    parser.add_argument("--record", action="store_true",
                        help="rewrite examples/expected/ from this run")
    parser.add_argument("--tolerance", type=float, default=1.0e-4,
                        help="relative tolerance of --check")
    arguments = parser.parse_args()

    import meep as mp
    import meep_gpu

    gpu = meep_gpu.available_gpu()
    single = bool(mp.is_single_precision())
    say(f"MEEP {mp.__version__}, single precision: {single}; GPU route: {gpu}")
    if gpu is None:
        say("No GPU route on this host: the gpu and array legs are skipped. "
            f"Missing: {list(meep_gpu.missing_dependencies())}")
    if not single:
        say("MEEP is a double-precision build. The engine steps single precision; "
            "the difference from MEEP is reported and not judged.")

    failures: list[str] = []
    if gpu is not None and not arguments.only:
        say()
        say("== quickstart")
        lines = run_quickstart()
        for line in lines:
            say(f"  {line}")
        if arguments.record:
            EXPECTED.mkdir(exist_ok=True)
            text = "\n".join(
                header()[:4]
                + ["# Lines are compared by --check, except the step path, which",
                   "# describes the recording host.", ""]
                + lines) + "\n"
            (EXPECTED / "quickstart.txt").write_text(text, encoding="utf-8")
            say(f"  recorded {EXPECTED.name}/quickstart.txt")
        if arguments.check:
            problems = check_quickstart(lines, arguments.tolerance)
            say(f"  [check] {len(problems)} mismatches against "
                f"{EXPECTED.name}/quickstart.txt")
            for problem in problems:
                say(f"    {problem}")
                failures.append(f"quickstart: {problem.splitlines()[0]}")

    for name in (arguments.only or EXAMPLES):
        say()
        say(f"== {name}")
        produced: list[str] = []
        with tempfile.TemporaryDirectory() as scratch:
            saved = {leg: Path(scratch) / f"{leg}.npz"
                     for leg in ("gpu", "array", "reference", "meep")}
            policy = "default"
            if gpu is not None:
                lines = run_leg(name, "gpu", saved["gpu"])
                produced += lines
                policy = host_value(lines, "subnormal policy") or "default"
                if policy not in ("flush", "keep"):
                    policy = "default"
                produced += run_leg(name, "array", saved["array"], policy=policy)
            produced += run_leg(name, "reference", saved["reference"])
            produced += run_leg(name, "meep", saved["meep"])

            compared = []
            if gpu is not None:
                identical, detail = compare_bytes(saved["gpu"], saved["array"])
                compared.append(
                    f"[compare] gpu against array (policy {policy}): "
                    f"{'identical bytes' if identical else 'DIFFERENT'} "
                    f"({', '.join(detail)})")
                if not identical:
                    failures.append(f"{name}: the gpu leg differs from the array path")
                field, spectrum = compare_relative(saved["gpu"], saved["reference"])
                compared.append(
                    f"[compare] gpu against reference (default policy): field "
                    f"relative difference {field:.3e}, spectrum {spectrum:.3e}")
            field, spectrum = compare_relative(saved["reference"], saved["meep"])
            judged, band = single, BANDS[name]
            verdict = ("not judged against" if not judged else
                       "within" if max(field, spectrum) <= band else "OUTSIDE")
            compared.append(
                f"[compare] reference against MEEP: field relative difference "
                f"{field:.3e}, spectrum {spectrum:.3e}; {verdict} the band {band:g}")
            if judged and max(field, spectrum) > band:
                failures.append(f"{name}: the reference differs from MEEP by more than "
                                f"{band:g}")
        for line in compared:
            say(f"  {line}")

        if arguments.record:
            EXPECTED.mkdir(exist_ok=True)
            text = "\n".join(header() + [""] + produced + [""] + compared) + "\n"
            (EXPECTED / f"{name}.txt").write_text(text, encoding="utf-8")
            say(f"  recorded {EXPECTED.name}/{name}.txt")
        if arguments.check:
            problems = check_against_expected(name, produced, arguments.tolerance)
            say(f"  [check] {len(problems)} mismatches against "
                f"{EXPECTED.name}/{name}.txt")
            for problem in problems:
                say(f"    {problem}")
                failures.append(f"{name}: {problem.splitlines()[0]}")

    say()
    if failures:
        say(f"FAILED: {len(failures)} judged comparisons did not hold")
        for failure in failures:
            say(f"  - {failure}")
        return 1
    say("PASSED: every judged comparison held")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
