"""DEVICE LEG DRIVER for :mod:`stress_cuda_end_to_end` — the same rows, the same
``run_row``, plus the three things a device run needs that a laptop run does not.

=============================================================================
WHY THIS FILE EXISTS RATHER THAN A FLAG
=============================================================================

``stress_cuda_end_to_end.py`` is complete as an experiment and is NOT edited
here: its ``subject_digest`` covers its own bytes, and a driver that mutated the
instrument it drives would move that digest for reasons unrelated to the kernels.
Everything below is ADDITIVE and lives outside it.

Three things are added, and each answers a question the NumPy leg cannot even
ask:

1. **THE CERTIFICATION SUBNORMAL POLICY, INSTALLED BEFORE THE FIRST COMPILE.**
   ``fastpath.CERTIFICATION_SUBNORMAL_POLICY`` is ``"keep"`` (fastpath.py:510) and
   the shipped dispatch gate refuses to dispatch under anything else
   (fastpath.py:1085-1092). CuPy appends ``-ftz=true`` to every NVRTC compile, so
   a stress run that installed nothing would compile the certified SOURCE into
   UNCERTIFIED BYTES and compare those. The order is load-bearing and was measured
   on device by the probe this calls: install-then-compile keeps a planted
   subnormal, compile-then-install flushes it and stays flushed
   (probe_fused_kernel_bit_identity.py:4347-4353).

2. **A DEVICE LAUNCH CENSUS, KEYED BY THE DEVICE KERNEL'S OWN NAME.** This is the
   floor the task set: a harness that silently failed to substitute would report
   THE ARRAY PATH AGREEING WITH ITSELF, which is the most expensive kind of pass
   available here. ``Substitution.dispatches`` counts WRAPPER invocations — it
   proves the driver's module globals were rebound, and it would count exactly the
   same if the wrapper's body never reached a GPU. The census counts
   ``cp.RawKernel.__call__`` and reports the count under the name the device
   function carries, so a green result is only readable together with a nonzero
   count against ``step_B_pml_real`` / ``step_D_pml_real`` / ``update_*_pml_real``.

   THE SEAM IS ``compile_cache.get_or_compile`` (compile_cache.py:182), because
   every kernel family reaches its ``cp.RawKernel`` through it and the three
   families this leg substitutes reach it by MODULE ATTRIBUTE
   (``compile_cache.get_or_compile(...)`` in step_curl_kernels.py:1198,
   constitutive_kernels.py:405, offdiag_constitutive_kernels.py:258), so rebinding
   the attribute is enough and no package byte changes. ``cp.RawKernel`` is a
   CuPy extension type whose ``__call__`` cannot be assigned, which is why the
   counter is a forwarding proxy rather than a method patch.

   **THE NAME IS READ OFF THE KERNEL, NOT OFF THE CALL SITE.** The whole occasion
   for this leg is a rename that moved 34 device kernel names, and a census keyed
   by the Python wrapper's name would have reported the same string before and
   after it. ``RawKernel.name`` is the ``extern "C" __global__`` symbol, so the
   census is also the measurement that the RENAMED kernels are the ones that ran.

3. **THE NVRTC BINARY OBSERVER** (``probe_fused_kernel_bit_identity``:4197),
   installed BEFORE the policy so the strip wraps it and it records the option
   tuple NVRTC was really given. It answers "did a compiler run at all, and with
   what", which a launch count alone does not: a cache hit above the seam launches
   without compiling.

=============================================================================
WHAT IT DOES NOT ADD
=============================================================================

The rows, the predicate consult, the plan, the wrappers, the inventory, the
checkpoints, the vacuity floors, the defect battery and the verdict are all
``stress_cuda_end_to_end``'s, unchanged and untouched. ``run_row`` is called
directly. If this driver disagreed with that module about anything, the module
wins.

=============================================================================
RUNNING IT
=============================================================================

Device (ONE verified-empty GPU, pinned by UUID; the cache dir MUST carry the
policy token ``ftz_stripped`` or ``"keep"`` refuses at install)::

    CUDA_VISIBLE_DEVICES=GPU-xxxx \\
    CUPY_CACHE_DIR=$OUT/cupy_cache/ftz_stripped \\
    MEEP_PYTHON_DIR=$HOME/meep_corpus/python \\
    PYTHONPATH=$API python -u \\
        parity/meep_gpu/stress_cuda_device_leg.py --backend cuda \\
        --steps 2000 --checkpoint-every 100 --out $OUT/run

Rule 7: one flushed line per row from the parent, one flushed line per checkpoint
from the child (``stress.say``), the per-row artifact rewritten atomically after
every checkpoint, and the launch census appended to it when the row ends.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_API = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _path in (_API, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)


# ---------------------------------------------------------------------------
# THE DEVICE LAUNCH CENSUS
# ---------------------------------------------------------------------------

class LaunchCensus:
    """Count every ``cp.RawKernel`` launch, under the DEVICE function's own name.

    Installed by rebinding ``compile_cache.get_or_compile`` to wrap whatever it
    hands back. The memo inside ``get_or_compile`` still holds the REAL kernel, so
    the proxy is per-call and never enters the cache; the proxies are themselves
    memoized by ``id`` of the real object so a long run does not build one per
    launch.

    A REFUSAL RATHER THAN A ZERO. ``install`` raises if the seam is not there to
    patch: a census that silently observed nothing would report ``{}`` for "the
    kernels never ran" and for "the counter was never wired", and only one of
    those is a finding about the kernels.
    """

    def __init__(self) -> None:
        self.launches: Dict[str, int] = {}
        self.constructions: Dict[str, int] = {}
        self._proxies: Dict[int, Any] = {}
        self._original: Optional[Any] = None
        self._module: Optional[Any] = None

    def _wrap(self, kernel: Any) -> Any:
        proxy = self._proxies.get(id(kernel))
        if proxy is not None:
            return proxy
        name = str(getattr(kernel, "name", None) or f"<unnamed {type(kernel).__name__}>")
        self.constructions[name] = self.constructions.get(name, 0) + 1
        census = self

        class _Counted:
            """Forwarding proxy: counts the call, forwards everything else."""

            def __call__(self, *args: Any, **kwargs: Any) -> Any:
                census.launches[name] = census.launches.get(name, 0) + 1
                return kernel(*args, **kwargs)

            def __getattr__(self, attribute: str) -> Any:
                return getattr(kernel, attribute)

            def __repr__(self) -> str:  # pragma: no cover - diagnostics only
                return f"<launch-counted {name} {kernel!r}>"

        proxy = _Counted()
        self._proxies[id(kernel)] = proxy
        return proxy

    def install(self) -> Dict[str, Any]:
        from meep_gpu.cuda_kernels import compile_cache  # noqa: PLC0415

        original = getattr(compile_cache, "get_or_compile", None)
        if not callable(original):
            raise RuntimeError(
                "meep_gpu.cuda_kernels.compile_cache has no get_or_compile to wrap; "
                "the launch census cannot be installed and a run without it cannot "
                "distinguish a substituted kernel from the array path agreeing with "
                "itself")

        def counted(key: Any, factory: Any) -> Any:
            return self._wrap(original(key, factory))

        counted.__name__ = "get_or_compile"
        compile_cache.get_or_compile = counted
        self._original, self._module = original, compile_cache
        return {"installed": True, "seam": "compile_cache.get_or_compile"}

    def uninstall(self) -> None:
        if self._original is not None and self._module is not None:
            self._module.get_or_compile = self._original
            self._original = None

    def report(self) -> Dict[str, Any]:
        return {
            "seam": "meep_gpu.cuda_kernels.compile_cache.get_or_compile",
            "counts_what": "cp.RawKernel.__call__, keyed by RawKernel.name — the "
                           "extern \"C\" __global__ symbol in the device source",
            "device_kernels_launched": dict(sorted(self.launches.items())),
            "total_launches": int(sum(self.launches.values())),
            "distinct_device_kernels": len(self.launches),
            "raw_kernel_constructions": dict(sorted(self.constructions.items())),
        }


# ---------------------------------------------------------------------------
# THE CHILD
# ---------------------------------------------------------------------------

def run_child(label: str, backend: str, steps: int, every: int, defect: str,
              out_dir: str, case_timeout: float, policy: str,
              examples_dir: Optional[str], tests_dir: Optional[str]) -> int:
    """One row, with the observer and the policy installed before the first compile."""
    import stress_cuda_end_to_end as stress  # noqa: PLC0415

    stress.resolve_corpus(examples_dir, tests_dir)
    stress._PROGRESS_PATH = os.path.join(out_dir, "progress.log")
    per_row_path = os.path.join(out_dir, "per_row", f"{label}.json")

    extras: Dict[str, Any] = {"backend_requested": backend}

    census = LaunchCensus()
    if backend == "cuda":
        import probe_fused_kernel_bit_identity as probe  # noqa: PLC0415

        # ORDER: observer first so the policy's strip wraps IT and it records the
        # option tuple NVRTC was really given (probe:4208-4216); policy second, and
        # both before anything has compiled.
        extras["nvrtc_observer"] = probe.install_nvrtc_binary_observer()
        extras["cupy_cache_dir"] = os.environ.get("CUPY_CACHE_DIR")
        extras["subnormal_policy_install"] = \
            probe.install_subnormal_policy_for_run(policy, _API)
        extras["subnormal_policy_stamp"] = probe.subnormal_policy_stamp(_API)
        extras["launch_census_install"] = census.install()
        extras["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            import cupy as cp  # noqa: PLC0415
            device = cp.cuda.Device()
            extras["device"] = {
                "id": int(device.id),
                "name": cp.cuda.runtime.getDeviceProperties(device.id)["name"]
                        .decode("utf-8", "replace"),
                "cupy": cp.__version__,
                "runtime": int(cp.cuda.runtime.runtimeGetVersion()),
            }
        except BaseException as exc:  # noqa: BLE001
            extras["device"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}

    try:
        record = stress.run_row(stress.ROWS_BY_LABEL[label], backend, steps, every,
                                defect, out_dir, case_timeout)
    finally:
        census.uninstall()

    if backend == "cuda":
        extras["launch_census"] = census.report()
        import probe_fused_kernel_bit_identity as probe  # noqa: PLC0415
        report = probe.nvrtc_binary_report()
        # The per-observation list is long and its digests are the part a reader
        # needs; the whole thing goes to a sidecar so the row artifact stays legible.
        observations = report.pop("observations", [])
        extras["nvrtc_binary_report"] = report
        with open(os.path.join(out_dir, "per_row", f"{label}.nvrtc.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(observations, handle, indent=2, default=str)

    # WHAT MAKES THE ROW READABLE AT ALL: the wrapper count and the device count,
    # side by side. Wrappers fired but no device kernel launched means the leg
    # measured the array path against itself.
    dispatched = sum(int(v) for v in
                     (record.get("plan", {}).get("dispatches") or {}).values())
    launched = int(extras.get("launch_census", {}).get("total_launches", 0))
    extras["substitution_reached_the_device"] = bool(backend != "cuda" or launched > 0)
    extras["wrapper_dispatches_total"] = dispatched
    record["device_leg"] = extras
    stress.save(record, per_row_path)
    stress.say(f"[{label}] wrapper_dispatches={dispatched} device_launches={launched} "
               f"kernels={sorted(extras.get('launch_census', {}).get('device_kernels_launched', {}))}")
    return 0 if not record.get("error") else 1


# ---------------------------------------------------------------------------
# THE PARENT
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    import stress_cuda_end_to_end as stress  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=("numpy", "cuda"), default="cuda")
    parser.add_argument("--rows", default="all")
    parser.add_argument("--examples-dir", default=None)
    parser.add_argument("--tests-dir", default=None)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--defect", default="none")
    parser.add_argument("--subnormal-policy", default="keep",
                        help="the policy INSTALLED before the first compile; "
                             "fastpath.CERTIFICATION_SUBNORMAL_POLICY is 'keep'")
    parser.add_argument("--out", required=True, help="a DIRECTORY")
    parser.add_argument("--timeout", type=float, default=7200.0)
    parser.add_argument("--case-timeout", type=float, default=600.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--child-row", default=None)
    args = parser.parse_args(argv)

    out_dir = os.path.abspath(args.out)
    os.makedirs(os.path.join(out_dir, "per_row"), exist_ok=True)
    stress._PROGRESS_PATH = os.path.join(out_dir, "progress.log")

    if args.child_row:
        return run_child(args.child_row, args.backend, args.steps,
                         args.checkpoint_every, args.defect, out_dir,
                         args.case_timeout, args.subnormal_policy,
                         args.examples_dir, args.tests_dir)

    examples_dir, tests_dir = stress.resolve_corpus(args.examples_dir, args.tests_dir)
    labels = (tuple(row["label"] for row in stress.ROWS) if args.rows == "all"
              else tuple(p.strip() for p in args.rows.split(",") if p.strip()))
    unknown = [label for label in labels if label not in stress.ROWS_BY_LABEL]
    if unknown:
        print(f"unknown row labels: {unknown}", flush=True)
        return 2

    results: Dict[str, Any] = {
        "what": "THE DEVICE LEG of the end-to-end stress: real corpus rows, the "
                "shipped CUDA kernels substituted at every admitted slot, the "
                "certification subnormal policy installed before the first compile, "
                "and every device launch counted under the device kernel's own name",
        "certifies": args.backend == "cuda",
        "driver": os.path.abspath(__file__),
        "instrument": os.path.abspath(stress.__file__),
        "backend": args.backend,
        "defect": args.defect,
        "defect_means": stress.DEFECTS[args.defect],
        "steps": args.steps,
        "checkpoint_every": args.checkpoint_every,
        "subnormal_policy_requested": args.subnormal_policy,
        "cupy_cache_dir": os.environ.get("CUPY_CACHE_DIR"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "corpus": {"examples_dir": examples_dir, "tests_dir": tests_dir},
        "rows_requested": list(labels),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "argv": list(sys.argv),
        "python": sys.version.split()[0],
        "subject_before": stress.subject_digest(_API),
        "rows": [],
    }
    artifact = os.path.join(out_dir, "stress_device.json")
    stress.save(results, artifact)

    environment = dict(os.environ)
    environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    environment["MPLBACKEND"] = "Agg"
    environment["PYTHONPATH"] = _API + os.pathsep + environment.get("PYTHONPATH", "")

    workdir = os.path.join(out_dir, "workdir")
    os.makedirs(workdir, exist_ok=True)
    for entry in sorted(os.listdir(examples_dir)):
        if entry.endswith((".py", ".ipynb")):
            continue
        link = os.path.join(workdir, entry)
        if not os.path.exists(link):
            try:
                os.symlink(os.path.join(examples_dir, entry), link)
            except OSError:
                pass

    stress.say(f"device leg: backend={args.backend} policy={args.subnormal_policy} "
               f"defect={args.defect} steps={args.steps} every={args.checkpoint_every} "
               f"rows={len(labels)} gpu={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    for index, label in enumerate(labels, start=1):
        per_row_path = os.path.join(out_dir, "per_row", f"{label}.json")
        if args.resume and os.path.exists(per_row_path):
            stress.say(f"  {index}/{len(labels)} {label}: resumed from disk")
            results["rows"].append(json.load(open(per_row_path, encoding="utf-8")))
            stress.save(results, artifact)
            continue
        command = [sys.executable, "-u", os.path.abspath(__file__),
                   "--child-row", label, "--backend", args.backend,
                   "--steps", str(args.steps),
                   "--checkpoint-every", str(args.checkpoint_every),
                   "--defect", args.defect, "--out", out_dir,
                   "--subnormal-policy", args.subnormal_policy,
                   "--case-timeout", str(args.case_timeout),
                   "--examples-dir", examples_dir, "--tests-dir", tests_dir]
        started = time.time()
        status = "ok"
        try:
            completed = subprocess.run(command, cwd=workdir, env=environment,
                                       timeout=args.timeout, check=False,
                                       stdout=None, stderr=subprocess.PIPE)
            stderr_text = (completed.stderr or b"").decode("utf-8", "replace")
        except subprocess.TimeoutExpired as expired:
            status = "timeout"
            stderr_text = ((expired.stderr or b"").decode("utf-8", "replace")
                           if expired.stderr else "")
        if os.path.exists(per_row_path):
            row_record = json.load(open(per_row_path, encoding="utf-8"))
        else:
            row_record = {"label": label,
                          "error": ("child died" if status == "ok" else "child timeout"),
                          "stderr_tail": stderr_text[-3000:]}
        if row_record.get("error") and stderr_text:
            row_record.setdefault("stderr_tail", stderr_text[-3000:])
        results["rows"].append(row_record)
        leg = row_record.get("device_leg", {})
        stress.say(
            f"  {index}/{len(labels)} {label}: "
            f"slots={row_record.get('plan', {}).get('slots_substituted')} "
            f"identical={row_record.get('identical_checkpoints')} "
            f"vacuous={row_record.get('vacuous_checkpoints')} "
            f"first_divergence={row_record.get('first_divergence_step')} "
            f"device_launches={leg.get('launch_census', {}).get('total_launches')} "
            f"{'ERROR ' + str(row_record.get('error'))[:160] if row_record.get('error') else ''}"
            f" ({time.time() - started:.1f} s)")
        stress.save(results, artifact)

    results["subject_after"] = stress.subject_digest(_API)
    results["subject_stable"] = (results["subject_after"]["manifest_sha256"]
                                 == results["subject_before"]["manifest_sha256"])
    results["verdict"] = stress.verdict(results["rows"], args.defect)

    # THE SUBSTITUTION FLOOR, taken over the whole run and reported beside the
    # verdict rather than left for a reader to assemble. A row that dispatched
    # wrappers but launched no device kernel is the array path agreeing with
    # itself, and it must not read as evidence.
    device_rows = {
        row.get("label"): {
            "wrapper_dispatches": row.get("device_leg", {}).get(
                "wrapper_dispatches_total"),
            "device_launches": row.get("device_leg", {}).get(
                "launch_census", {}).get("total_launches"),
            "device_kernels": sorted(row.get("device_leg", {}).get(
                "launch_census", {}).get("device_kernels_launched", {})),
        } for row in results["rows"]}
    results["substitution_census"] = device_rows
    results["rows_that_reached_the_device"] = sorted(
        label for label, entry in device_rows.items()
        if (entry.get("device_launches") or 0) > 0)
    results["rows_with_wrappers_but_no_device_launch"] = sorted(
        label for label, entry in device_rows.items()
        if (entry.get("wrapper_dispatches") or 0) > 0
        and not (entry.get("device_launches") or 0))
    results["not_established"] = list(getattr(stress, "_NOT_ESTABLISHED", ())) or [
        "Only the rows in stress_cuda_end_to_end.ROWS. 7 of 186.",
        "update_P, and every family with no CUDA predicate, is never substituted.",
        "Determinism: each row is ONE comparison against the array path. A race "
        "that resolves the same way twice is invisible here.",
        "Throughput: nothing here is a performance measurement.",
    ]
    stress.save(results, artifact)
    stress.say(f"verdict: {json.dumps(results['verdict'], default=str)}")
    stress.say(f"reached_device={results['rows_that_reached_the_device']}")
    stress.say(f"wrappers_without_device_launch="
               f"{results['rows_with_wrappers_but_no_device_launch']}")
    stress.say(f"subject_stable={results['subject_stable']} -> {artifact}")
    if args.defect == "none":
        return 0 if results["verdict"].get("null_confirmed") else 1
    return 0 if results["verdict"].get("caught") else 1


if __name__ == "__main__":
    raise SystemExit(main())
