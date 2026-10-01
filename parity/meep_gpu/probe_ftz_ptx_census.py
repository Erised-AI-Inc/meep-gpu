#!/usr/bin/env python3
"""DIRECT attribution: the ``.ftz`` census of BOTH executors' generated code, same sub-step.

WHY THIS EXISTS. The end-to-end dispatch proof measured that, as shipped
(``lift_simulation`` + ``MEEP_GPU_DISPATCH=1``, nothing installing a subnormal
policy), 6 of 7 dispatching cases diverge from the kill-switched array path, that
every differing element at the first divergent rung sits at or beside the float32
subnormal range, and that installing ONE uniform policy removes the divergence
entirely. It did NOT show the mechanism at instruction level; it named that as the
obvious next measurement and declined to claim it. This probe is that measurement.

THE CLAIM UNDER TEST, stated so it can FAIL::

    as shipped, the CuPy array-path kernels for a sub-step carry ``.ftz`` on their
    f32 arithmetic and the Triton kernel for the SAME sub-step does not; and
    installing a subnormal policy makes the two agree.

If the census does not show that, the divergence has a different cause and the
next phase's fix would be aimed wrong. That outcome is reported plainly rather
than smoothed over.

WHAT IS COMPARED, and why it is the same thing on both sides. One lifted
simulation, one device, one process per policy leg. The driver's five consulted
sub-steps (``step_B``, ``update_H``, ``step_D``, ``update_E``, ``update_P``) are
the unit: for each, the probe collects the generated code of whichever executor
would run it.

* CUPY side — the kill-switched leg (``MEEP_GPU_FUSED=0``) takes ONE step with
  the five array entry points wrapped in the ``driver`` module namespace, so every
  NVRTC compile is attributed to the sub-step it happened inside. The PTX is read
  from the SAME ``nvrtcProgram`` the shipped compile produced — the wrapper sits
  on ``_NVRTCProgram.compile``, BELOW both ``backends``' host-FPU guard and any
  policy strip, calls the original, and then asks that already-compiled program
  for its PTX. Same source, same options, same compile: nothing is recompiled with
  different flags to make it readable.
* TRITON side — the dispatching leg takes one step (which freezes the plan and
  warms every kernel), and then, per dispatched slot, drops EVERY live
  ``JITFunction`` cache and re-dispatches just that slot. What is in the caches
  afterwards is exactly that slot's kernels, and ``CompiledKernel.asm['ptx']`` is
  their PTX. Field values are irrelevant here and the driver is discarded.

Both caches are COLD per leg (a private ``CUPY_CACHE_DIR`` and ``TRITON_CACHE_DIR``),
because CuPy's disk cache is keyed above the seam this probe reads: a warm cache
serves a binary without ever calling the compiler, and the probe would census
nothing and report zero.

THE CENSUS IS THE PACKAGE'S OWN. ``subnormal_policy.audit_ptx`` decides what
counts as an audited f32 arithmetic instruction — predicated forms and packed
``.f32x2`` included, transport (``mov``/``ld``/``st``/``cvt``) excluded — so the
number here and the number the shipped compile-time audit enforces are the same
number, not two definitions that happen to agree today.

LEGS (one process each; a policy is chosen before the first compile or not at all)::

    shipped   nothing installs a policy. CuPy flushes by its own default,
              Triton keeps by its own. WHAT A USER GETS.
    keep      install_subnormal_policy("keep") before the lift.
    flush     install_subnormal_policy("flush") before the lift, with
              CUPY_ACCELERATORS='' set before CuPy imports (the module refuses
              otherwise, and the refusal names the variable).

Progress reporting: one flushed line per phase per sub-step, and the JSON artifact is
rewritten atomically as each sub-step lands, so an interrupted run keeps
everything up to the failure.

Run (the GPU host, ONE pinned GPU verified empty)::

    CUDA_VISIBLE_DEVICES=<idx> CUPY_CACHE_DIR=<private> TRITON_CACHE_DIR=<private> \\
        python -u parity/meep_gpu/probe_ftz_ptx_census.py \\
        --leg shipped --case pml_2d --out results/ftz_ptx_census/<leg>
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_API = os.path.abspath(os.path.join(HERE, "..", ".."))
if REPO_API not in sys.path:
    sys.path.insert(0, REPO_API)
if HERE not in sys.path:  # The case builders live beside this file, in the gate.
    sys.path.insert(0, HERE)

#: The driver's five consulted sub-steps, in step order.
SLOTS: Tuple[str, ...] = ("step_B", "update_H", "step_D", "update_E", "update_P")

_STARTED = time.time()


def say(message: str) -> None:  # progress reporting: unbuffered, one line, elapsed.
    print(f"[{time.time() - _STARTED:8.1f}s] {message}", flush=True)


# ---------------------------------------------------------------------------
# Artifact
# ---------------------------------------------------------------------------


def save(record: Dict[str, Any], path: str) -> None:
    """Atomic rewrite: tmp + fsync + replace, after every sub-step (progress reporting)."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    with os.fdopen(handle, "w") as sink:
        json.dump(record, sink, indent=2, sort_keys=True, default=str)
        sink.flush()
        os.fsync(sink.fileno())
    os.replace(tmp, path)


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as sink:
        sink.write(text)


# ---------------------------------------------------------------------------
# The CuPy seam: PTX out of the SAME compile the shipped path performs
# ---------------------------------------------------------------------------


class CupyCompileCapture:
    """Every NVRTC compile CuPy performs, with its final options and its PTX.

    Installed on ``_NVRTCProgram.compile``, which is BELOW both the host-FPU guard
    ``backends.guard_kernel_compilation`` installs on ``compile_using_nvrtc`` and
    the ``-ftz=true`` strip ``subnormal_policy`` installs at the same seam. What
    arrives here is therefore the option tuple NVRTC actually sees — the direct
    reading of "did this process pass ``-ftz=true``" — and ``nvrtc.getPTX`` on the
    program the original call just compiled is that compile's own generated code.
    """

    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []
        self.slot: Optional[str] = None
        self._original = None
        self._compiler = None

    def install(self) -> None:
        from cupy.cuda import compiler as C  # noqa: PLC0415

        self._compiler = C
        self._original = C._NVRTCProgram.compile
        original = self._original
        capture = self

        def compile(self, options=(), log_stream=None):  # noqa: A002 - mirrors CuPy's name
            result = original(self, options, log_stream)
            ptx, error = "", None
            try:
                raw = C.nvrtc.getPTX(self.ptr)
                ptx = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
            except Exception as exc:  # noqa: BLE001 - an unreadable program is recorded, not fatal
                error = repr(exc)
            # ``result[0]`` is a CUBIN only when ``method == 'cubin'``, and CuPy uses
            # BOTH in one process: measured on this host the kernels resolve
            # ``-arch=sm_86``/cubin while the empty-file include probe compiles
            # ``-arch=compute_86``/ptx. Handing that PTX text to a disassembler as if
            # it were a cubin produced a "file appears truncated" error that read
            # like a tool failure rather than a category error, so the method decides
            # and it is recorded per compile.
            method = str(getattr(self, "method", ""))
            payload = result[0] if isinstance(result, tuple) else None
            cubin = payload if (method == "cubin" and isinstance(payload, bytes)) else None
            capture.records.append({
                "slot": capture.slot,
                "name": str(getattr(self, "name", "")),
                "method": method,
                "options": [str(option) for option in tuple(options)],
                "ftz_true_in_options": any("-ftz=true" in str(o) or "--ftz=true" in str(o)
                                           for o in tuple(options)),
                "ftz_option_present": any("ftz" in str(o) for o in tuple(options)),
                "source": str(getattr(self, "src", "")),
                "ptx": ptx,
                "ptx_error": error,
                "cubin": cubin if isinstance(cubin, bytes) else None,
            })
            return result

        C._NVRTCProgram.compile = compile

    def uninstall(self) -> None:
        if self._original is not None and self._compiler is not None:
            self._compiler._NVRTCProgram.compile = self._original


# ---------------------------------------------------------------------------
# The Triton seam: every live JITFunction's compiled-kernel cache
# ---------------------------------------------------------------------------


def _jit_functions() -> List[Any]:
    """Every live ``JITFunction``. Triton keeps no registry; the GC is the registry."""
    import gc  # noqa: PLC0415
    import importlib  # noqa: PLC0415

    jit_module = importlib.import_module("triton.runtime.jit")
    jit_class = getattr(jit_module, "JITFunction", None)
    if jit_class is None:
        return []
    return [obj for obj in gc.get_objects() if isinstance(obj, jit_class)]


def clear_triton_caches() -> int:
    """Drop every in-process compiled-kernel cache; return how many entries went."""
    dropped = 0
    for function in _jit_functions():
        cache = getattr(function, "cache", None)
        if not isinstance(cache, dict):
            continue
        for per_device in list(cache.values()):
            if isinstance(per_device, dict):
                dropped += len(per_device)
                per_device.clear()
    return dropped


def collect_triton_kernels() -> List[Dict[str, Any]]:
    """Name, PTX and cubin of every compiled kernel currently cached in process."""
    out: List[Dict[str, Any]] = []
    for function in _jit_functions():
        cache = getattr(function, "cache", None)
        if not isinstance(cache, dict):
            continue
        name = str(getattr(function, "__name__", "") or getattr(function, "fn", ""))
        for per_device in cache.values():
            if not isinstance(per_device, dict):
                continue
            for compiled in per_device.values():
                asm = getattr(compiled, "asm", None)
                if not isinstance(asm, dict):
                    continue
                ptx = asm.get("ptx")
                if not ptx:
                    continue
                out.append({"name": name, "ptx": str(ptx),
                            "cubin": asm.get("cubin") if isinstance(asm.get("cubin"), bytes) else None,
                            "stages": sorted(asm.keys())})
    return out


# ---------------------------------------------------------------------------
# Census
# ---------------------------------------------------------------------------


def census(ptx: str) -> Dict[str, Any]:
    """The package's OWN audit, read for counts rather than for a verdict.

    ``audit_ptx(..., FLUSH)`` is asked because under flush the violation list is
    exactly the audited instructions WITHOUT ``.ftz`` — so one call yields both
    halves of the census. ``audited``/``with_ftz``/``by_opcode`` do not depend on
    which policy is passed.
    """
    from meep_gpu.subnormal_policy import FLUSH, audit_ptx  # noqa: PLC0415

    result = audit_ptx(ptx, FLUSH)
    return {"audited": int(result["audited"]),
            "with_ftz": int(result["with_ftz"]),
            "without_ftz": int(result["missing_ftz"]),
            "by_opcode": dict(result["by_opcode"])}


def combine(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = {"kernels": len(entries), "audited": 0, "with_ftz": 0, "without_ftz": 0,
             "by_opcode": {},
             "sass": {"kernels_read": 0, "fp32_arithmetic": 0, "with_ftz": 0,
                      "without_ftz": 0, "conversions_with_ftz": 0}}
    for entry in entries:
        stats = entry["census"]
        total["audited"] += stats["audited"]
        total["with_ftz"] += stats["with_ftz"]
        total["without_ftz"] += stats["without_ftz"]
        for opcode, count in stats["by_opcode"].items():
            total["by_opcode"][opcode] = total["by_opcode"].get(opcode, 0) + count
        sass = entry.get("sass") or {}
        if sass.get("read"):
            total["sass"]["kernels_read"] += 1
            for key in ("fp32_arithmetic", "with_ftz", "without_ftz", "conversions_with_ftz"):
                total["sass"][key] += int(sass.get(key, 0))
    total["by_opcode"] = dict(sorted(total["by_opcode"].items()))
    return total


#: SASS fp32 ARITHMETIC mnemonics. Conversions are deliberately absent for the same
#: reason ``audit_ptx`` excludes ``cvt``: measured on this hardware, a shipped-leg
#: ``pml_curl_step`` disassembles with exactly ONE ``.FTZ`` line and it is
#: ``F2I.FTZ.U32.TRUNC.NTZ`` — a float-to-int conversion whose FTZ is the
#: conversion's own modifier, not the denormal policy's. Counting it would report
#: "1 of 67 flush" for a kernel in which no arithmetic flushes at all.
SASS_FP32_ARITHMETIC = ("FADD", "FMUL", "FFMA", "FSETP", "FMNMX", "FSEL", "MUFU")
SASS_FP32_CONVERSION = ("F2I", "I2F", "F2F", "FRND")

_SASS_OPCODE = None


def sass_census(cubin: Optional[bytes], nvdisasm: Optional[str]) -> Optional[Dict[str, Any]]:
    """The SASS fp32 arithmetic census, when ``nvdisasm`` can read the cubin.

    PTX is sufficient for the claim; SASS is added when it is free because it is
    the code the SM actually runs, and because it is an INDEPENDENT reading — a
    PTX modifier that ptxas then ignored would show up here and nowhere else. A
    cubin ``nvdisasm`` refuses (an arch newer than this toolkit knows) is recorded
    as unread, never as zero.
    """
    global _SASS_OPCODE
    if not cubin or not nvdisasm:
        return None
    if _SASS_OPCODE is None:
        import re  # noqa: PLC0415

        _SASS_OPCODE = re.compile(r"/\*[0-9a-f]+\*/\s+(?:@!?\w+\s+)?([A-Z][A-Z0-9_.]*)")
    handle, path = tempfile.mkstemp(suffix=".cubin")
    try:
        with os.fdopen(handle, "wb") as sink:
            sink.write(cubin)
        # TWO DISASSEMBLERS, IN ORDER, and the fallback is load-bearing rather than
        # defensive: measured on this host, ``nvdisasm -c`` reads Triton's cubin and
        # refuses every NVRTC-produced one with "appears truncated" (a 2495-byte
        # file whose first four bytes are \x7fELF), while ``cuobjdump -sass`` reads
        # both. Reporting only nvdisasm's refusal would have left the CuPy side
        # with no SASS reading and made the asymmetry look like a property of the
        # code rather than of the tool.
        tools = [[nvdisasm, "-c", path],
                 [os.path.join(os.path.dirname(nvdisasm), "cuobjdump"), "-sass", path]]
        proc = None
        for command in tools:
            if not os.path.exists(command[0]):
                continue
            proc = subprocess.run(command, capture_output=True, text=True)
            if proc.returncode == 0 and proc.stdout.strip():
                break
        if proc is None or proc.returncode != 0 or not proc.stdout.strip():
            detail = (proc.stderr or proc.stdout or "").strip()[:400] if proc else "no disassembler"
            return {"read": False, "error": detail}
        arithmetic: Dict[str, int] = {}
        total = with_ftz = conversions_with_ftz = 0
        for line in proc.stdout.splitlines():
            match = _SASS_OPCODE.search(line)
            if not match:
                continue
            opcode = match.group(1)
            stem = opcode.split(".")[0]
            if stem in SASS_FP32_CONVERSION:
                conversions_with_ftz += int(".FTZ" in opcode)
                continue
            if stem not in SASS_FP32_ARITHMETIC:
                continue
            total += 1
            with_ftz += int(".FTZ" in opcode)
            arithmetic[opcode] = arithmetic.get(opcode, 0) + 1
        return {"read": True, "fp32_arithmetic": total, "with_ftz": with_ftz,
                "without_ftz": total - with_ftz,
                "conversions_with_ftz": conversions_with_ftz,
                "by_opcode": dict(sorted(arithmetic.items()))}
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Cases (the end-to-end gate's, imported so the configuration is literally the same)
# ---------------------------------------------------------------------------


def build_case(name: str, res: Optional[int]):
    import meep as mp  # noqa: PLC0415

    from gate_dispatch_end_to_end import CASES  # noqa: PLC0415

    try:
        mp.verbosity(0)
    except Exception:  # noqa: BLE001
        pass
    builder = CASES[name]
    return builder(mp) if res is None else builder(mp, res)


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------


def run_cupy_phase(case: str, res: Optional[int], gpu_id: int,
                   nvdisasm: Optional[str], out_dir: str) -> Dict[str, Any]:
    """One kill-switched step, with every NVRTC compile attributed to its sub-step."""
    import meep_gpu  # noqa: PLC0415
    from meep_gpu import driver as driver_module  # noqa: PLC0415

    os.environ["MEEP_GPU_DISPATCH"] = "1"
    os.environ["MEEP_GPU_FUSED"] = "0"  # The kill switch: the array path IS the reference.

    capture = CupyCompileCapture()
    capture.install()

    originals = {}
    for slot in SLOTS:
        original = getattr(driver_module, slot)
        originals[slot] = original

        def wrapped(*args, _slot=slot, _original=original, **kwargs):
            previous, capture.slot = capture.slot, _slot
            try:
                return _original(*args, **kwargs)
            finally:
                capture.slot = previous

        setattr(driver_module, slot, wrapped)

    try:
        sim, _monitors, _until = build_case(case, res)
        driver = meep_gpu.lift_simulation(sim, prefer_gpu=True, gpu_id=gpu_id)
        say(f"cupy: lifted {tuple(int(v) for v in driver.shape)}; stepping once")
        driver.step()
        try:
            driver.xp.cuda.runtime.deviceSynchronize()
        except Exception:  # noqa: BLE001 - a NumPy host has nothing to synchronize
            pass
        step_path = getattr(driver, "active_step_path", None)
    finally:
        for slot, original in originals.items():
            setattr(driver_module, slot, original)
        capture.uninstall()

    per_slot: Dict[str, Any] = {}
    unattributed: List[Dict[str, Any]] = []
    for index, record in enumerate(capture.records):
        stats = census(record["ptx"]) if record["ptx"] else {
            "audited": 0, "with_ftz": 0, "without_ftz": 0, "by_opcode": {}}
        entry = {"index": index,  # Names ptx/cupy_<slot>_<index>.ptx and src/…​.cu.
                 "name": record["name"], "options": record["options"],
                 "method": record["method"],
                 "ftz_true_in_options": record["ftz_true_in_options"],
                 "ptx_error": record["ptx_error"], "census": stats,
                 "sass": (sass_census(record["cubin"], nvdisasm) if record["cubin"]
                          else {"read": False, "reason": f"compile method {record['method']!r}: "
                                                         "no cubin to disassemble"})}
        slot = record["slot"]
        target = per_slot.setdefault(slot, []) if slot else unattributed
        if slot:
            target.append(entry)
        else:
            unattributed.append(entry)
        stem = f"cupy_{slot or 'other'}_{index:03d}"
        write_text(os.path.join(out_dir, "ptx", f"{stem}.ptx"), record["ptx"])
        write_text(os.path.join(out_dir, "src", f"{stem}.cu"), record["source"])

    every = [entry for entries in per_slot.values() for entry in entries] + unattributed
    return {
        "step_path": step_path,
        "compiles": len(capture.records),
        # ATTRIBUTION IS FIRST-COMPILE-WINS, and that is a real limit rather than a
        # bookkeeping detail: CuPy memoizes an elementwise kernel in process, so a
        # sub-step whose expression was already compiled by an earlier sub-step
        # compiles nothing and reports an empty slot. It does not weaken the claim —
        # ``whole_step`` and ``options`` below are over EVERY compile the step made,
        # and the per-slot blocks say which sub-step first asked for each.
        "per_slot": {slot: {"kernels": entries, "total": combine(entries)}
                     for slot, entries in sorted(per_slot.items())},
        "unattributed": {"kernels": unattributed, "total": combine(unattributed)},
        "whole_step": combine(every),
        "options_carried_ftz_true": sorted({
            bool(record["ftz_true_in_options"]) for record in capture.records}),
        "compiles_with_ftz_true": sum(1 for record in capture.records
                                      if record["ftz_true_in_options"]),
        "example_options": (capture.records[0]["options"] if capture.records else []),
    }


def run_triton_phase(case: str, res: Optional[int], gpu_id: int,
                     nvdisasm: Optional[str], out_dir: str,
                     artifact: Dict[str, Any], artifact_path: str) -> Dict[str, Any]:
    """One dispatching step to freeze the plan, then one slot at a time from cold caches."""
    import meep_gpu  # noqa: PLC0415

    os.environ["MEEP_GPU_DISPATCH"] = "1"
    os.environ.pop("MEEP_GPU_FUSED", None)

    sim, _monitors, _until = build_case(case, res)
    driver = meep_gpu.lift_simulation(sim, prefer_gpu=True, gpu_id=gpu_id)
    say(f"triton: lifted {tuple(int(v) for v in driver.shape)}; stepping once to freeze the plan")
    driver.step()
    try:
        driver.xp.cuda.runtime.deviceSynchronize()
    except Exception:  # noqa: BLE001
        pass

    report = driver.fast_path_report() or {}
    plan = driver._fast_path  # The frozen plan; the probe reads it, never edits it.
    result: Dict[str, Any] = {
        "step_path": getattr(driver, "active_step_path", None),
        "decision": report.get("decision"),
        "refusal": report.get("refusal"),
        "slots": list(report.get("slots") or []),
        "arms": dict(report.get("arms") or {}),
        "launch_counters": report.get("launch_counters"),
        "per_slot": {},
    }
    if plan is None:
        say("triton: NO PLAN — the dispatcher refused; nothing to census")
        return result
    result["slots"] = list(plan.slots)
    result["arms"] = dict(plan.arms)

    for slot in plan.slots:
        dropped = clear_triton_caches()
        error = None
        try:
            ran = plan.dispatch(slot, driver.fields)
        except Exception as exc:  # noqa: BLE001 - recorded, never repaired
            ran, error = False, repr(exc)
        try:
            driver.xp.cuda.runtime.deviceSynchronize()
        except Exception:  # noqa: BLE001
            pass
        kernels = collect_triton_kernels()
        entries = []
        for index, kernel in enumerate(kernels):
            entries.append({"name": kernel["name"], "stages": kernel["stages"],
                            "census": census(kernel["ptx"]),
                            "sass": sass_census(kernel["cubin"], nvdisasm)})
            write_text(os.path.join(out_dir, "ptx", f"triton_{slot}_{index:03d}.ptx"),
                       kernel["ptx"])
        result["per_slot"][slot] = {
            "arm": plan.arms.get(slot), "dispatched": bool(ran), "error": error,
            "cache_entries_dropped_first": dropped,
            "kernels": entries, "total": combine(entries)}
        total = result["per_slot"][slot]["total"]
        say(f"triton/{slot}: {total['kernels']} kernels, {total['audited']} audited f32, "
            f"{total['with_ftz']} with .ftz")
        result["whole_step"] = combine([entry for block in result["per_slot"].values()
                                        for entry in block["kernels"]])
        artifact["triton"] = result
        save(artifact, artifact_path)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def policy_leg(leg: str) -> Dict[str, Any]:
    """Install the leg's policy — or record, positively, that nothing was installed.

    MEEP IS IMPORTED FIRST, ON EVERY LEG. ``install_host_policy("flush")`` refuses
    outright without it — ``mp.set_zero_subnormals`` is the only exposure of this
    process's FTZ/DAZ bits the package will use — and importing it on only the leg
    that needs it would make the three legs differ in more than the policy. MEEP
    sets the host FTZ/DAZ bits at import on x86, so the ordering is also the one
    the shipped path has: every leg's ``build_case`` imports MEEP before anything
    compiles.
    """
    import meep as mp  # noqa: PLC0415

    from meep_gpu import backends  # noqa: PLC0415
    from meep_gpu import subnormal_policy as sp  # noqa: PLC0415

    before = bool(backends.subnormals_flushed())
    block = {"leg": leg, "meep": str(getattr(mp, "__version__", "unknown")),
             "host_flushing_after_meep_import": before}
    if leg == "shipped":
        block.update({"installed": False,
                      "policy_is_installed": sp.policy_is_installed(),
                      "would_resolve_to": sp.default_policy(),
                      "host_flushing_at_lift": bool(backends.subnormals_flushed()),
                      "note": "nothing calls install_subnormal_policy on the shipped path"})
        return block
    stamp = sp.install_subnormal_policy(leg)
    block.update({"installed": True, "stamp": stamp,
                  "policy_is_installed": sp.policy_is_installed(),
                  "host_flushing_at_lift": bool(backends.subnormals_flushed())})
    return block


def environment() -> Dict[str, Any]:
    import cupy  # noqa: PLC0415
    import triton  # noqa: PLC0415

    device = cupy.cuda.Device()
    props = cupy.cuda.runtime.getDeviceProperties(device.id)
    return {
        "host": platform.node(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cupy": cupy.__version__,
        "triton": triton.__version__,
        "device_name": props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"]),
        "compute_capability": f"{props['major']}.{props['minor']}",
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "CUPY_CACHE_DIR": os.environ.get("CUPY_CACHE_DIR"),
        "TRITON_CACHE_DIR": os.environ.get("TRITON_CACHE_DIR"),
        "CUPY_ACCELERATORS": os.environ.get("CUPY_ACCELERATORS"),
        "MEEP_GPU_SUBNORMAL_POLICY": os.environ.get("MEEP_GPU_SUBNORMAL_POLICY"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leg", required=True, choices=("shipped", "keep", "flush"))
    parser.add_argument("--case", default="pml_2d")
    parser.add_argument("--res", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--nvdisasm", default="/usr/local/cuda-11.6/bin/nvdisasm")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    artifact_path = os.path.join(out_dir, "census.json")
    nvdisasm = args.nvdisasm if args.nvdisasm and os.path.exists(args.nvdisasm) else None

    say(f"leg={args.leg} case={args.case} out={out_dir} nvdisasm={bool(nvdisasm)}")

    # The policy is chosen BEFORE the first compile, which means before the lift.
    policy = policy_leg(args.leg)
    say(f"policy: {json.dumps({k: v for k, v in policy.items() if k != 'stamp'})}")

    artifact: Dict[str, Any] = {
        "leg": args.leg, "case": args.case, "res": args.res,
        "policy": policy, "environment": environment(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    save(artifact, artifact_path)

    say("phase 1/2: CuPy array path (kill switch on)")
    artifact["cupy"] = run_cupy_phase(args.case, args.res, args.gpu, nvdisasm, out_dir)
    for slot, block in artifact["cupy"]["per_slot"].items():
        total = block["total"]
        say(f"cupy/{slot}: {total['kernels']} kernels, {total['audited']} audited f32, "
            f"{total['with_ftz']} with .ftz")
    save(artifact, artifact_path)

    say("phase 2/2: Triton dispatch")
    artifact["triton"] = run_triton_phase(args.case, args.res, args.gpu, nvdisasm,
                                          out_dir, artifact, artifact_path)

    # The comparison, per sub-step both executors covered, plus the whole step.
    verdict: Dict[str, Any] = {}
    rows: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
    for slot in SLOTS:
        cupy_block = artifact["cupy"]["per_slot"].get(slot)
        triton_block = artifact["triton"]["per_slot"].get(slot)
        if not cupy_block or not triton_block:
            continue
        rows.append((slot, cupy_block["total"], triton_block["total"]))
    if artifact["triton"].get("whole_step"):
        rows.append(("whole_step", artifact["cupy"]["whole_step"],
                     artifact["triton"]["whole_step"]))
    for slot, cupy_total, triton_total in rows:
        verdict[slot] = {
            "cupy_audited": cupy_total["audited"], "cupy_with_ftz": cupy_total["with_ftz"],
            "triton_audited": triton_total["audited"], "triton_with_ftz": triton_total["with_ftz"],
            "cupy_all_ftz": cupy_total["audited"] > 0 and cupy_total["without_ftz"] == 0,
            "triton_none_ftz": triton_total["audited"] > 0 and triton_total["with_ftz"] == 0,
            "triton_all_ftz": triton_total["audited"] > 0 and triton_total["without_ftz"] == 0,
            "agree": (cupy_total["audited"] > 0 and triton_total["audited"] > 0
                      and (cupy_total["without_ftz"] == 0) == (triton_total["without_ftz"] == 0)
                      and (cupy_total["with_ftz"] == 0) == (triton_total["with_ftz"] == 0)),
            "triton_sass": triton_total["sass"],
        }
    artifact["verdict"] = verdict
    artifact["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save(artifact, artifact_path)

    for slot, row in verdict.items():
        say(f"VERDICT {slot}: cupy {row['cupy_with_ftz']}/{row['cupy_audited']} ftz, "
            f"triton {row['triton_with_ftz']}/{row['triton_audited']} ftz, "
            f"agree={row['agree']}")
    say(f"DONE leg={args.leg} artifact={artifact_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
