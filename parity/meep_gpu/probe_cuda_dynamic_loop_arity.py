"""THE DYNAMIC-LOOP GATE, device leg: is a loop bit-identical to an unrolled sum?

Subject and rationale: :mod:`cuda_dynamic_loop_arity`. This file is the runner.

WHAT IT MEASURES, in the order it measures it:

1. ``ptx``            - NVRTC PTX for every variant at every swept arity, with a
                        census: does the dynamic kernel really contain a LOOP
                        (a backward branch), does the unrolled one really not,
                        does each carry exactly the float arithmetic the
                        transcription says, and is ``fma.rn.f32`` absent from
                        both under ``--fmad=false``. Needs NVRTC, and a VISIBLE
                        device to read the architecture off (``ptx_arch``) -- still
                        no FREE device and no CUDA context.

                        ONE CAVEAT, MEASURED RATHER THAN ASSUMED, AND IT LIMITS
                        WHAT THIS SECTION LICENSES. This section compiles under
                        OUR option tuple alone. ``cupy.RawKernel`` -- the path
                        section 2 launches through -- appends more, and the
                        addition that matters is ``-ftz=true``. Measured on
                        the GPU host, CuPy 13.5.1, by wrapping
                        ``compile_using_nvrtc`` and reading the tuple it really
                        received:

                            (... cccl includes ..., '--fmad=false', '--std=c++11',
                             ... include paths ..., '-ftz=true')

                        So ``any_ftz == 0`` here is a fact about THIS compile and
                        NOT about the binary section 2 ran. It does not touch
                        this gate's question -- flushing changes which values
                        become zero, not the ORDER of the additions, and the
                        headline compares two kernels that both carry the flag --
                        but a reader must not cite this census as evidence about
                        the launched binary's subnormal policy. The sibling
                        ``probe_cuda_ptx_transfer.py`` is where that is measured
                        at the seam the policy acts on.
2. ``bytes``          - the verdict. Every case launched under every variant from
                        the SAME host arrays; outputs compared as UINT32 WORDS.
                        Reports, per case: subject-vs-unrolled identity, each
                        control's catch rate, the trip counts the kernels
                        reported, the words each launch moved, and the array-path
                        reference's own agreement.
3. ``determinism``    - each variant relaunched N times on fresh device copies of
                        the same host arrays. A verdict from one launch cannot
                        distinguish "identical" from "identical this once".

EVERY LEG CARRIES ITS OWN FLOORS AND WILL FAIL RATHER THAN PASS VACUOUSLY:

* ``moved_fraction`` >= ``MOVED_WORD_FLOOR`` for every launch of every variant.
  Zero-init is a fixed point of this sub-step, so two kernels that wrote nothing
  would compare identical.
* every ARMED control caught on >= ``CONTROL_CATCH_FLOOR`` of the words, on the
  headline value class. If ``dynamic_reversed`` -- the same operations in the
  opposite order -- is NOT caught, then these operands cannot separate two
  association orders and the leg's "identical" is about the draw, not the code.
* the trip counts the dynamic kernels report must equal the arity they were
  given, everywhere, and the ``dynamic_short`` control must report one less per
  component.
* ``dynamic_notrips`` must be bit-identical to ``dynamic``, which is what makes
  the trip counter admissible as evidence rather than a perturbation.

Run (the GPU host, ONE verified-free GPU)::

    CUDA_VISIBLE_DEVICES=<gpu> python -u probe_cuda_dynamic_loop_arity.py \\
        --out results/<dir>/dynamic_loop_arity.json \\
        --jsonl results/<dir>/dynamic_loop_arity.jsonl \\
        --ptx-dir results/<dir>/ptx

Progress is a line per case on stdout, unbuffered, and every case is appended to
the JSONL as it lands, so an interrupted run keeps everything up to the failure
(the progress-reporting rule).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_API = os.path.abspath(os.path.join(HERE, "..", ".."))
if REPO_API not in sys.path:
    sys.path.insert(0, REPO_API)


def load_subject():
    """The subject module, loaded BY PATH so this probe stands alone.

    The sibling probes do the same, and for the same reason: a gate that needs
    the package importable is a gate that cannot run against a staged tree.
    """
    path = os.path.join(HERE, "cuda_dynamic_loop_arity.py")
    spec = importlib.util.spec_from_file_location("cuda_dynamic_loop_arity", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SUBJECT = load_subject()

def log(message: str) -> None:
    print(message, flush=True)


def ptx_arch() -> str:
    """``compute_NN`` for the device THIS PROCESS has open, read when asked.

    WHY NOT A CONSTANT (changed 2026-09-30). This was ``ARCH_PTX =
    "compute_86"``, bound as ``compile_to_ptx``'s default argument — so the value
    was fixed when the module was IMPORTED and the recorded ``arch`` was a typed
    claim rather than a measurement. Run on any other card, the probe would compile
    sm_86 PTX, census it, and record the whole section as if it described the device
    it was launching on: a census of the wrong architecture is not a weaker result,
    it is a false one, and PTX is exactly what differs between architectures.

    ``Device().compute_capability`` is CuPy's own undotted string (``"86"``,
    ``"90"``, ``"100"`` for cc 10.0), which is already the spelling NVRTC's
    ``--gpu-architecture=compute_NN`` wants, so it is concatenated rather than
    re-derived from major/minor ints — the same read ``gate_cuda_complex.py`` makes
    for its NVRTC guard. RAISES on a host with no readable device: the callers
    record that as a named section error, because a PTX census with no device behind
    it has nothing to say.
    """
    import cupy as cp  # noqa: PLC0415 - a device read, never at import time

    return "compute_" + str(cp.cuda.Device().compute_capability)


# ---------------------------------------------------------------------------
# PTX -- no device needed
# ---------------------------------------------------------------------------

def compile_to_ptx(code: str, options: Sequence[str], arch: str) -> str:
    """Real PTX text from NVRTC's own ``getPTX``.

    ``compile_using_nvrtc`` returns a CUBIN on CuPy 13.5.1, so counting opcodes
    in its result measures nothing. Needs the NVRTC library, not a context.

    ``arch`` IS REQUIRED and has no default: a default would be evaluated once at
    import and the caller could not tell a measured architecture from a stale one.
    The caller reads it from :func:`ptx_arch` and records the same value it
    compiles under.
    """
    from cupy.cuda import nvrtc  # noqa: PLC0415

    program = nvrtc.createProgram(code, "kernel.cu", [], [])
    try:
        nvrtc.compileProgram(program, tuple(options) + (f"--gpu-architecture={arch}",))
        text = nvrtc.getPTX(program).decode()
    finally:
        nvrtc.destroyProgram(program)
    if ".visible .entry" not in text:
        raise RuntimeError("NVRTC did not return PTX (no '.visible .entry')")
    return text


_LABEL = re.compile(r"^\s*\$?(L__\w+):", re.M)
_BRANCH = re.compile(r"\bbra(?:\.uni)?\s+\$?(L__\w+)")


def ptx_census(text: str) -> Dict[str, Any]:
    """What a PTX body IS, in the terms this gate needs.

    ``backward_branches`` is the loop detector and it is a STRUCTURAL read, not
    a heuristic: a branch whose target label appears EARLIER in the text than
    the branch itself is a back edge, which is what a loop is. An unrolled chain
    has none.
    """
    label_at = {match.group(1): match.start() for match in _LABEL.finditer(text)}
    backward = 0
    forward = 0
    for match in _BRANCH.finditer(text):
        target = label_at.get(match.group(1))
        if target is None:
            continue
        if target < match.start():
            backward += 1
        else:
            forward += 1
    # THE ROUNDING MODE IS IN THE OPCODE and the first cut of this census forgot
    # it: NVRTC emits `sub.rn.f32`, not `sub.f32`, so `\bsub\.f32\b` matched
    # nothing and the leg reported zero float arithmetic in a kernel full of it.
    # A census that reads zero for everything looks exactly like a census of a
    # kernel that does nothing. Optional `.rn`/`.ftz` now, and the counts are
    # asserted against the transcription's own term count.
    counts = {
        "backward_branches": backward,
        "forward_branches": forward,
        "sub_f32": len(re.findall(r"\bsub(?:\.rn)?(?:\.ftz)?\.f32\b", text)),
        "add_f32": len(re.findall(r"\badd(?:\.rn)?(?:\.ftz)?\.f32\b", text)),
        "mul_f32": len(re.findall(r"\bmul(?:\.rn)?(?:\.ftz)?\.f32\b", text)),
        "fma_rn_f32": len(re.findall(r"\bfma\.rn(?:\.ftz)?\.f32\b", text)),
        "any_ftz": len(re.findall(r"\b[a-z]+(?:\.[a-z0-9]+)*\.ftz\.f32\b", text)),
        "any_f64": len(re.findall(r"\.f64\b", text)),
        "ld_global_f32": len(re.findall(r"\bld\.global(?:\.nc)?\.f32\b", text)),
        "st_global_f32": len(re.findall(r"\bst\.global\.f32\b", text)),
        "ptx_lines": text.count("\n"),
    }
    counts["registers"] = sum(
        int(m.group(2)) for m in re.finditer(r"\.reg\s+\.([a-z0-9]+)\s+%\w+<(\d+)>", text))
    return counts


def unrolled_expectation(arity: int) -> Dict[str, int]:
    """The float opcodes the UNROLLED kernel must contain at a symmetric arity.

    Counted off the transcription, not off a dump:

      sub.rn.f32   3*arity   the pole subtractions, one per term per component
                 + 3         ``a - kms*prev``, the second accumulation
      add.rn.f32   3         ``f[idx] + kps*src``, the first accumulation
      mul.rn.f32   9         ``s*inv_eps``, ``kps*src``, ``kms*prev`` x3
      fma.rn.f32   0         --fmad=false; a contraction here would round once
                             where the array path rounds twice

    Checking it is what stops a preprocessor guard that never matched from
    passing as a kernel: a body with no pole subtractions in it would compare
    bit-identical to another body with no pole subtractions in it.
    """
    return {"sub_f32": 3 * arity + 3, "add_f32": 3, "mul_f32": 9, "fma_rn_f32": 0}


def section_ptx(results: Dict[str, Any], ptx_dir: Optional[str],
                arities: Sequence[int]) -> None:
    """PTX for every variant at every swept arity, plus the option-tuple pin."""
    log("== section ptx ==")
    out: Dict[str, Any] = {
        "arch": None,
        "options": list(SUBJECT.COMPILE_OPTIONS),
        "ftz_caveat": (
            "compiled under our option tuple ALONE. cupy.RawKernel -- the launch "
            "path section 'bytes' uses -- additionally appends '-ftz=true' "
            "(measured on the GPU host, CuPy 13.5.1, by wrapping "
            "compile_using_nvrtc). any_ftz==0 here is therefore a fact about "
            "THIS compile and not about the launched binary. It does not affect "
            "the association-order question, which is what this gate measures."),
        "census": {}, "errors": [], "failures": []}

    # THE ARCHITECTURE IS READ ONCE, HERE, AND RECORDED AS WHAT WAS COMPILED. A
    # failure is an ERROR row rather than a raised exception because this module's
    # ``main`` runs the byte verdict after this section and writes the artifact at
    # the end: losing a completed device leg to an unreadable identity would be the
    # expensive failure, and ``out["ok"]`` already turns any error into a non-zero
    # exit.
    try:
        arch = ptx_arch()
    except Exception as error:  # noqa: BLE001 - recorded, not raised
        out["errors"].append(
            f"the PTX architecture cannot be read from the device "
            f"({type(error).__name__}: {error}); a census compiled under a typed "
            f"architecture would describe a card this run never opened")
        out["ok"] = False
        log(f"   PTX ARCH UNREADABLE: {type(error).__name__}: {error}")
        results["ptx"] = out
        return
    out["arch"] = arch
    log(f"   arch read from the live device: {arch}")

    if ptx_dir:
        os.makedirs(ptx_dir, exist_ok=True)
    for variant in SUBJECT.ALL_VARIANTS:
        per_arity: Dict[str, Any] = {}
        # The dynamic variants carry no arity in their source; one compile is the
        # whole family, and saying so IS the cost measurement.
        sweep = arities if variant == "unrolled" else (None,)
        for arity in sweep:
            triple = (arity, arity, arity) if arity is not None else (0, 0, 0)
            code = SUBJECT.source_for(variant, triple)
            key = "any" if arity is None else str(arity)
            try:
                text = compile_to_ptx(code, SUBJECT.COMPILE_OPTIONS, arch)
            except Exception as error:  # noqa: BLE001 - recorded, not raised
                out["errors"].append(f"{variant}@{key}: {type(error).__name__}: {error}")
                log(f"   {variant:20s} np={key:>3s}  COMPILE FAILED: {error}")
                continue
            census = ptx_census(text)
            census["source_sha256"] = SUBJECT.source_digest(code)
            per_arity[key] = census
            if ptx_dir:
                with open(os.path.join(ptx_dir, f"{variant}_np{key}.ptx"), "w") as handle:
                    handle.write(text)

            # The two structural claims the byte verdict leans on, checked here
            # rather than eyeballed: the unrolled kernel has NO loop and exactly
            # the arithmetic the transcription says; the dynamic ones DO loop.
            if variant == "unrolled":
                if census["backward_branches"]:
                    out["failures"].append(
                        f"unrolled@{key}: {census['backward_branches']} backward "
                        f"branches -- the 'unrolled' side contains a loop")
                for opcode, expected in unrolled_expectation(int(key)).items():
                    if census[opcode] != expected:
                        out["failures"].append(
                            f"unrolled@{key}: {opcode}={census[opcode]}, transcription "
                            f"says {expected}")
            elif census["backward_branches"] == 0:
                out["failures"].append(
                    f"{variant}: no backward branch -- the compiler fully unrolled "
                    f"the runtime-bounded loop and there is no dynamic loop to test")

            log(f"   {variant:20s} np={key:>3s}  back_branches="
                f"{census['backward_branches']:2d}  sub={census['sub_f32']:3d} "
                f"add={census['add_f32']:2d} mul={census['mul_f32']:2d}  "
                f"fma.rn.f32={census['fma_rn_f32']}  ftz={census['any_ftz']}  "
                f"regs={census['registers']}")
        out["census"][variant] = per_arity
    out["ok"] = not out["failures"] and not out["errors"]
    for failure in out["failures"]:
        log(f"   PTX FAILURE: {failure}")
    results["ptx"] = out


# ---------------------------------------------------------------------------
# Device launches
# ---------------------------------------------------------------------------

def _device_state(case: Dict[str, Any]) -> Dict[str, Any]:
    """One fresh device copy of a case's host arrays.

    FRESH PER LAUNCH, always. The sub-step is in place -- it overwrites E and
    f_w_E -- so a second launch on the first launch's buffers would be measuring
    a different input. That is the "stale cached pointer" false-pass class this
    project has hit before.
    """
    import cupy as cp  # noqa: PLC0415

    state = case["state"]
    device = {name: cp.asarray(np.ascontiguousarray(array))
              for name, array in state.items()}
    device["kps"] = [cp.asarray(np.ascontiguousarray(v)) for v in case["kps_vectors"]]
    device["kms"] = [cp.asarray(np.ascontiguousarray(v)) for v in case["kms_vectors"]]
    shape = case["shape"]
    device["trips"] = cp.zeros(int(np.prod(shape)), dtype=cp.int32)
    return device


_KERNEL_CACHE: Dict[Tuple[str, Any], Any] = {}


def get_kernel(variant: str, arities: Tuple[int, int, int]):
    """Compile once per distinct binary, memoized on the SOURCE.

    The key is the source text, not the variant name, so a dynamic variant is
    compiled ONCE for the whole arity sweep and an unrolled one once per triple.
    That asymmetry is the cost model this gate exists to price, and the memo
    makes it countable: :func:`compile_counts` reports it.
    """
    import cupy as cp  # noqa: PLC0415

    code = SUBJECT.source_for(variant, arities)
    key = (variant, SUBJECT.source_digest(code))
    if key not in _KERNEL_CACHE:
        name = ("dispersive_update_E_unrolled" if variant == "unrolled"
                else "dispersive_update_E_dynamic")
        _KERNEL_CACHE[key] = cp.RawKernel(code, name, options=SUBJECT.COMPILE_OPTIONS)
    return _KERNEL_CACHE[key]


def compile_counts() -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for variant, _digest in _KERNEL_CACHE:
        counts[variant] = counts.get(variant, 0) + 1
    return counts


def launch(variant: str, case: Dict[str, Any], device: Dict[str, Any]) -> None:
    """One launch of one variant against one fresh device state."""
    import cupy as cp  # noqa: PLC0415

    shape = case["shape"]
    arities = case["arities"]
    n = int(np.prod(shape))
    blocks = (n + SUBJECT.THREADS - 1) // SUBJECT.THREADS
    kernel = get_kernel(variant, arities)

    fields = tuple(device[name] for name in
                   ("Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez",
                    "Dx", "Dy", "Dz",
                    "inv_eps_Ex", "inv_eps_Ey", "inv_eps_Ez"))
    tables = (device["kps"][0], device["kms"][0],
              device["kps"][1], device["kms"][1],
              device["kps"][2], device["kms"][2])

    if variant == "unrolled":
        poles = tuple(device[f"{prefix}{pole}"]
                      for prefix in ("Px", "Py", "Pz")
                      for pole in range(SUBJECT.MAX_POLES))
        args = fields + poles + (device["trips"],
                                 shape[0], shape[1], shape[2]) + tables
    else:
        # A DEVICE ARRAY OF POINTERS per component. This is the indirection a
        # runtime arity forces; eight named arguments cannot be indexed by p.
        tables_of_pointers = []
        for prefix in ("Px", "Py", "Pz"):
            values = np.array(
                [device[f"{prefix}{pole}"].data.ptr for pole in range(SUBJECT.MAX_POLES)],
                dtype=np.uint64)
            holder = cp.asarray(values)
            device.setdefault("_pointer_tables", []).append(holder)
            tables_of_pointers.append(holder)
        args = (fields + tuple(tables_of_pointers)
                + (int(arities[0]), int(arities[1]), int(arities[2]))
                + (device["trips"], shape[0], shape[1], shape[2]) + tables)

    kernel((blocks,), (SUBJECT.THREADS,), args)
    cp.cuda.runtime.deviceSynchronize()


def harvest(device: Dict[str, Any]) -> Dict[str, np.ndarray]:
    import cupy as cp  # noqa: PLC0415

    return {name: cp.asnumpy(device[name]) for name in SUBJECT.OUTPUT_NAMES}


def expected_trips(variant: str, arities: Tuple[int, int, int]) -> int:
    """What the kernel's own trip counter must report.

    The short control drops one iteration per component, and never below zero --
    ``for (p = 0; p < np - 1; ...)`` with np = 0 runs zero times, which is why
    the control is only ARMED at NP >= 1.
    """
    if variant == "dynamic_short":
        return sum(max(value - 1, 0) for value in arities)
    return sum(arities)


# ---------------------------------------------------------------------------
# The byte verdict
# ---------------------------------------------------------------------------

def run_case(descriptor: Dict[str, Any], repeats: int) -> Dict[str, Any]:
    """Every variant, one case, one verdict row."""
    case = SUBJECT.make_case(descriptor["shape"], descriptor["arities"],
                             descriptor["value_class"], descriptor["seed"])
    before = {name: np.array(case["state"][name], copy=True)
              for name in SUBJECT.OUTPUT_NAMES}

    outputs: Dict[str, Dict[str, np.ndarray]] = {}
    trips: Dict[str, Dict[str, int]] = {}
    moved: Dict[str, Dict[str, Any]] = {}
    repeat_identical: Dict[str, bool] = {}
    failures: List[str] = []

    for variant in SUBJECT.ALL_VARIANTS:
        device = _device_state(case)
        launch(variant, case, device)
        outputs[variant] = harvest(device)
        harvested_trips = device["trips"].get()
        trips[variant] = {"min": int(harvested_trips.min()),
                          "max": int(harvested_trips.max()),
                          "expected": expected_trips(variant, case["arities"])}
        moved[variant] = SUBJECT.moved_words(before, outputs[variant])
        floor = SUBJECT.MOVED_WORD_FLOOR_BY_CLASS[descriptor["value_class"]]
        if moved[variant]["moved_fraction"] < floor:
            failures.append(
                f"{variant}: moved {moved[variant]['moved_fraction']:.4f} of its words, "
                f"below the {floor} floor -- the leg is vacuous")
        # Determinism: relaunch on FRESH copies of the same host arrays.
        same = True
        for _ in range(max(repeats - 1, 0)):
            again = _device_state(case)
            launch(variant, case, again)
            if not SUBJECT.compare_words(outputs[variant], harvest(again))["identical"]:
                same = False
        repeat_identical[variant] = same
        if not same:
            failures.append(f"{variant}: not deterministic across {repeats} launches")

    # The trip-count claim, checked rather than asserted. `dynamic_notrips` writes
    # no counter by construction, so it is exempt BY NAME rather than by silence.
    for variant in SUBJECT.ALL_VARIANTS:
        if variant == "dynamic_notrips":
            continue
        record = trips[variant]
        if record["min"] != record["expected"] or record["max"] != record["expected"]:
            failures.append(
                f"{variant}: trip count {record['min']}..{record['max']} is not the "
                f"expected {record['expected']}")

    baseline = outputs["unrolled"]
    comparisons: Dict[str, Dict[str, Any]] = {
        variant: SUBJECT.compare_words(baseline, outputs[variant])
        for variant in SUBJECT.ALL_VARIANTS if variant != "unrolled"}

    # The null: the trip counter must not perturb the arithmetic.
    counter_inert = SUBJECT.compare_words(outputs["dynamic"],
                                          outputs["dynamic_notrips"])["identical"]
    if not counter_inert:
        failures.append("dynamic_notrips differs from dynamic: the trip counter is "
                        "perturbing the arithmetic and is not admissible evidence")

    # Controls. An ARMED control that is NOT caught is a defect in the LEG.
    # Scored on the ELIGIBLE arrays only -- the components whose own arity puts
    # them in the control's reach. See SUBJECT.eligible_outputs.
    control_rows: Dict[str, Any] = {}
    for control in SUBJECT.CONTROL_VARIANTS:
        armed_at = SUBJECT.CONTROL_ARMED_AT[control]
        eligible = SUBJECT.eligible_outputs(case["arities"], armed_at)
        armed = bool(eligible)
        result = comparisons[control]
        scoped = (SUBJECT.compare_words(baseline, outputs[control], eligible)
                  if eligible else None)
        caught_fraction = scoped["differing_fraction"] if scoped else 0.0
        caught_words = scoped["differing_words"] if scoped else 0
        control_rows[control] = {"armed": armed, "armed_at": armed_at,
                                 "eligible_arrays": list(eligible),
                                 "caught_fraction": caught_fraction,
                                 "caught_words": caught_words,
                                 "caught": bool(scoped and not scoped["identical"]),
                                 "caught_words_all_arrays": result["differing_words"],
                                 "max_ulp": result["max_ulp"]}
        if armed and descriptor["value_class"] == SUBJECT.HEADLINE_VALUE_CLASS:
            if caught_fraction < SUBJECT.CONTROL_CATCH_FLOOR:
                failures.append(
                    f"{control}: caught on {caught_fraction:.4f} of the eligible "
                    f"words, below the {SUBJECT.CONTROL_CATCH_FLOOR} floor -- these "
                    f"operands cannot separate association orders, so the verdict "
                    f"is about the draw")
            if caught_words < SUBJECT.CONTROL_CATCH_MINIMUM_WORDS:
                failures.append(
                    f"{control}: caught {caught_words} words, below the absolute "
                    f"{SUBJECT.CONTROL_CATCH_MINIMUM_WORDS}-word floor -- a fraction "
                    f"of a tiny case is not evidence")
        if armed and not control_rows[control]["caught"]:
            failures.append(
                f"{control}: ARMED at NP>={armed_at} and NOT CAUGHT at all -- the "
                f"comparison cannot fail, so it certifies nothing")

    reference = SUBJECT.reference_update_E(
        case["state"], case["arities"], case["kps_broadcast"], case["kms_broadcast"])
    reference_rows = {variant: SUBJECT.compare_words(reference, outputs[variant])
                      for variant in ("unrolled",) + SUBJECT.SUBJECT_VARIANTS}

    subject_identical = all(comparisons[v]["identical"] for v in SUBJECT.SUBJECT_VARIANTS)

    return {
        "key": SUBJECT.case_key(case["shape"], case["arities"],
                                descriptor["value_class"]),
        "shape": list(case["shape"]),
        "arities": list(case["arities"]),
        "value_class": descriptor["value_class"],
        "seed": descriptor["seed"],
        "operand_census": case["operand_census"],
        "subject_identical": subject_identical,
        "comparisons": comparisons,
        "controls": control_rows,
        "trips": trips,
        "moved": moved,
        "repeat_identical": repeat_identical,
        "counter_inert": counter_inert,
        "reference": {k: {"identical": v["identical"],
                          "differing_words": v["differing_words"],
                          "max_ulp": v["max_ulp"]}
                      for k, v in reference_rows.items()},
        "failures": failures,
        "ok": not failures,
    }


def section_bytes(results: Dict[str, Any], jsonl_path: Optional[str],
                  repeats: int, limit: Optional[int]) -> None:
    log("== section bytes ==")
    descriptors = SUBJECT.sweep_cases()
    if limit:
        descriptors = descriptors[:limit]
    rows: List[Dict[str, Any]] = []
    started = time.time()
    handle = open(jsonl_path, "a") if jsonl_path else None
    try:
        for index, descriptor in enumerate(descriptors, start=1):
            case_started = time.time()
            row = run_case(descriptor, repeats)
            rows.append(row)
            if handle:
                handle.write(json.dumps(row) + "\n")
                handle.flush()
            verdict = "IDENTICAL" if row["subject_identical"] else "DIVERGENT"
            caught = ",".join(
                f"{name.replace('dynamic_', '')}={data['caught_fraction']:.3f}"
                for name, data in row["controls"].items() if data["armed"])
            log(f"   case {index}/{len(descriptors)} {row['key']:44s} {verdict:9s} "
                f"moved={row['moved']['dynamic']['moved_fraction']:.3f} "
                f"controls[{caught}] "
                f"{'OK' if row['ok'] else 'FAIL: ' + '; '.join(row['failures'])} "
                f"({time.time() - case_started:.2f} s)")
    finally:
        if handle:
            handle.close()
    results["bytes"] = {
        "cases": len(rows),
        "elapsed_s": time.time() - started,
        "repeats": repeats,
        "compiles_per_variant": compile_counts(),
        "rows": rows,
    }
    summarize(results)


def summarize(results: Dict[str, Any]) -> None:
    rows = results["bytes"]["rows"]
    headline = [r for r in rows if r["value_class"] == SUBJECT.HEADLINE_VALUE_CLASS]
    by_arity: Dict[int, Dict[str, int]] = {}
    for row in rows:
        top = max(row["arities"])
        bucket = by_arity.setdefault(top, {"cases": 0, "identical": 0})
        bucket["cases"] += 1
        bucket["identical"] += int(row["subject_identical"])
    first_divergent = next((r["key"] for r in rows if not r["subject_identical"]), None)
    summary = {
        "cases": len(rows),
        "identical": sum(int(r["subject_identical"]) for r in rows),
        "headline_cases": len(headline),
        "headline_identical": sum(int(r["subject_identical"]) for r in headline),
        "ok": all(r["ok"] for r in rows),
        "failures": [f for r in rows for f in r["failures"]],
        "by_max_arity": by_arity,
        "first_divergent_case": first_divergent,
        "verdict": ("IDENTICAL" if all(r["subject_identical"] for r in rows)
                    else "DIVERGENT"),
    }
    results["summary"] = summary
    log("")
    log(f"VERDICT: {summary['verdict']}   "
        f"{summary['identical']}/{summary['cases']} cases bit-identical "
        f"(headline class {summary['headline_identical']}/{summary['headline_cases']})")
    if first_divergent:
        log(f"   first divergent case: {first_divergent}")
    if summary["failures"]:
        log(f"   LEG FAILURES ({len(summary['failures'])}):")
        for failure in summary["failures"][:20]:
            log(f"     - {failure}")


# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--jsonl", default=None)
    parser.add_argument("--ptx-dir", default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sections", default="ptx,bytes")
    arguments = parser.parse_args(argv)

    sections = [s.strip() for s in arguments.sections.split(",") if s.strip()]
    results: Dict[str, Any] = {
        "probe": "cuda_dynamic_loop_arity",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "subject_sha256": SUBJECT.file_digest(
            os.path.join(HERE, "cuda_dynamic_loop_arity.py")),
        "probe_sha256": SUBJECT.file_digest(os.path.abspath(__file__)),
        "options": list(SUBJECT.COMPILE_OPTIONS),
        "corpus_arities": SUBJECT.CORPUS_ARITIES,
    }
    try:
        import cupy as cp  # noqa: PLC0415
        results["cupy"] = cp.__version__
        results["device"] = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    except Exception as error:  # noqa: BLE001
        results["cupy_error"] = f"{type(error).__name__}: {error}"

    if "ptx" in sections:
        section_ptx(results, arguments.ptx_dir, SUBJECT.SWEEP_ARITIES)
    if "bytes" in sections:
        section_bytes(results, arguments.jsonl, arguments.repeats, arguments.limit)

    os.makedirs(os.path.dirname(os.path.abspath(arguments.out)), exist_ok=True)
    with open(arguments.out, "w") as handle:
        json.dump(results, handle, indent=1, default=str)
    log(f"wrote {arguments.out}")
    ok = (results.get("summary", {}).get("ok", True)
          and results.get("ptx", {}).get("ok", True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
