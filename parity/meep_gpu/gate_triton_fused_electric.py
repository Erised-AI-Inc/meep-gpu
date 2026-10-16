"""CUDA gate for the fused ``step_D -> zero_metal_D -> update_E`` Triton kernel.

The magnetic fused-pair gate predates the electric kernel and cannot imply it:
the D curl shifts backward, the two PML halves use the opposite Yee sublattices,
``zero_metal_D`` clears two components per wall, and ``update_E`` consumes
``D * inverse_epsilon``. This gate measures those facts together, in one launch.

Every base case compares all twelve arrays written by the pair against both:

* an independent array-order transcription pinned to ``stepping.py`` on NumPy;
* the existing two-kernel Triton composition with the wall pass between it.

The mutation legs include real defects and measured nulls. Results are rewritten
atomically after every case and every progress line is flushed (the progress-reporting rule).
Run only inside a one-GPU Slurm allocation.

THE CAMPAIGN'S SUBNORMAL POLICY. Until the 1.0 certification round this gate installed
no float32 subnormal policy and stated none: the CuPy oracle flushed (CuPy's default
``-ftz=true``) while Triton kept, so its ledger entry declared ``_mixed_policy`` and no
run could carry evidence for a policy. The two arms that cite it (``fused pair B`` and
``fused pair D``) were then refused on every flush dispatch, and with them every plan
the default precedence composes from them on the route gate's flush legs. The gate now
does what the five composition gates do (``composition_policy.py``, owner ruling R16):
before it builds anything it installs the policy ``MEEP_GPU_SUBNORMAL_POLICY`` names
(``keep`` when unset) on the host, CuPy and Triton, refuses by name when it is not
attained, stamps it as ``subnormal_policy``, and confirms at the end of the run that it
still holds. It lifts no MEEP simulation, so under ``flush`` it runs through the flush
fleet's MEEP-importing wrapper (``rounds/1.0/meep_host_import.py``), which
``run_lane.py`` always uses; a direct flush run without it refuses by name.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import re
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

try:
    import cupy as cp
except ImportError:  # The NumPy reference is imported by the laptop weld.
    cp = None

HERE = Path(__file__).resolve().parent
API_ROOT = HERE.parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))


def _load_shared():
    path = HERE / "probe_fused_kernel_bit_identity.py"
    spec = importlib.util.spec_from_file_location("fused_electric_shared", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


shared = _load_shared()

TARGETS = ("Dx", "Dy", "Dz")
CURL_AUX = ("fu_Dx", "fu_Dy", "fu_Dz")
SOURCES = ("Hx", "Hy", "Hz")
E_TARGETS = ("Ex", "Ey", "Ez")
E_AUX = ("f_w_Ex", "f_w_Ey", "f_w_Ez")
STATE_NAMES = TARGETS + CURL_AUX + E_TARGETS + E_AUX


def log(message: str) -> None:
    print(message, flush=True)


def save(results: Dict[str, Any], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
    _stamp_provenance(results)  # bytes THIS process imported; see gate_provenance
    temporary.write_text(json.dumps(results, indent=2), encoding="utf-8")
    os.replace(temporary, out)


def zero_metal_flags(boundaries: Sequence[str]) -> Tuple[bool, bool, bool]:
    return tuple(kind == shared.METALLIC for kind in boundaries)  # type: ignore[return-value]


def reference_zero_metal_D(state: Dict[str, Any], flags: Sequence[bool]) -> None:
    """The array path's tangential-D wall wipe, one stored low plane per wall."""
    for axis, flag in enumerate(flags):
        if not flag:
            continue
        for component_axis, component in enumerate(TARGETS):
            if component_axis != axis:
                state[component][shared._face(axis, 0)] = 0


def reference_fused_electric_step(
    xp: Any,
    arrays: Dict[str, Any],
    state: Dict[str, Any],
    curl_coefficients: Dict[str, Any],
    constitutive_coefficients: Dict[str, Any],
    dtdx: Any,
    boundaries: Sequence[str],
    zero_metal: Sequence[bool],
    grouping: str = "array_order",
    *,
    record: Sequence[int],
) -> None:
    """Independent transcription of the three passes in their driver order.

    ``step_D``, ``zero_metal_D``, ``update_E``, each half choosing its forms on the
    one chunk ``record`` (REQUIRED): the record the plans launched with, or, in the
    validator, the record of the region plan the array path stepped with.
    ``grouping="regrouped"`` re-associates both halves (the curl's
    ``pairwise_order`` and the constitutive's ``regrouped``).
    """
    regrouped = grouping == "regrouped"
    shared.reference_pml_step(
        xp,
        {name: arrays[name] for name in SOURCES},
        {name: state[name] for name in TARGETS},
        {name: state[name] for name in CURL_AUX},
        curl_coefficients,
        dtdx,
        "step_D",
        boundaries,
        "pairwise_order" if regrouped else "array_order",
        record=record,
    )
    reference_zero_metal_D(state, zero_metal)
    constitutive_arrays = {
        **{name: state[name] for name in TARGETS},
        **{f"inv_eps_{name}": arrays[f"inv_eps_{name}"] for name in E_TARGETS},
    }
    shared.reference_constitutive_step(
        "E",
        constitutive_arrays,
        state,
        constitutive_coefficients,
        "regrouped" if regrouped else "array_order",
        record=record,
    )


def make_arrays(xp: Any, shape: Tuple[int, int, int], seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    arrays = {
        name: xp.asarray(np.ascontiguousarray(
            rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)))
        for name in STATE_NAMES + SOURCES
    }
    for name in E_TARGETS:
        arrays[f"inv_eps_{name}"] = xp.asarray(np.ascontiguousarray(
            rng.uniform(0.2, 0.9, size=shape).astype(np.float32)))
    return arrays


def coefficients(source: str, shape, boundaries, dtdx, repo_root: str):
    if source == "synthetic":
        return (
            shared.synthetic_coefficients(cp, shape, False),
            shared.synthetic_constitutive_coefficients(cp, shape, True),
        )
    _, layer = shared.real_pml_layer(
        cp, shape, boundaries, repo_root, courant=float(dtdx))
    return (
        shared.layer_coefficients(layer, False),
        shared.layer_constitutive_coefficients(layer, True),
    )


def copy_state(arrays: Dict[str, Any]) -> Dict[str, Any]:
    copied = dict(arrays)
    for name in STATE_NAMES:
        copied[name] = arrays[name].copy()
    return copied


def compare_state(actual: Dict[str, Any], expected: Dict[str, Any]) -> Dict[str, Any]:
    return shared.combine({
        name: shared.bit_compare(actual[name], expected[name]) for name in STATE_NAMES
    })


def run_fused(
    arrays: Dict[str, Any],
    curl_flat: Dict[str, Any],
    constitutive_flat: Dict[str, Any],
    codes: Sequence[int],
    zero_metal: Sequence[bool],
    dtdx: float,
    kernel: Any = None,
    guard: bool = False,
    num_warps: Optional[int] = 1,
) -> None:
    from gate_chunk_records import every_cell_record
    from meep_gpu.triton_kernels.launch import plan_fused_pair_from_arrays

    # The fused and the unfused legs bind ONE division, every cell in a layer chunk:
    # the hand reference steps the recurrence on every cell.
    plan = plan_fused_pair_from_arrays(
        "D", arrays, curl_flat, constitutive_flat, codes, zero_metal, dtdx,
        kernel=kernel, num_warps=num_warps,
        chunk_bounds=every_cell_record(arrays[TARGETS[0]].shape))
    plan.run(guard=guard)


def run_unfused(
    arrays: Dict[str, Any],
    curl_flat: Dict[str, Any],
    constitutive_flat: Dict[str, Any],
    codes: Sequence[int],
    zero_metal: Sequence[bool],
    dtdx: float,
    guard: bool = False,
) -> None:
    from gate_chunk_records import every_cell_record
    from meep_gpu.triton_kernels.launch import (
        plan_constitutive_from_arrays,
        plan_from_arrays,
    )

    record = every_cell_record(arrays[TARGETS[0]].shape)
    plan_from_arrays("step_D", arrays, curl_flat, codes, dtdx,
                     chunk_bounds=record).run(guard=guard)
    reference_zero_metal_D(arrays, zero_metal)
    plan_constitutive_from_arrays("E", arrays, constitutive_flat,
                                  chunk_bounds=record).run(guard=guard)


def one_case(
    shape,
    boundaries,
    dtdx,
    coefficient_source,
    repo_root,
    *,
    kernel=None,
    mutation: Optional[str] = None,
    num_warps: Optional[int] = 1,
) -> Dict[str, Any]:
    case: Dict[str, Any] = {
        "shape": list(shape),
        "boundaries": list(boundaries),
        "dtdx": repr(dtdx),
        "coefficient_source": coefficient_source,
        "mutation": mutation,
        "num_warps": num_warps,
    }
    walled_invariant = [axis for axis in range(3)
                        if shape[axis] == 1 and boundaries[axis] == shared.METALLIC]
    if walled_invariant:
        case["skipped"] = f"metallic invariant axis {walled_invariant[0]}"
        return case

    curl_coefficients, constitutive_coefficients = coefficients(
        coefficient_source, shape, boundaries, dtdx, repo_root)
    arrays = make_arrays(cp, shape, shared.SEED + 71)
    flags = zero_metal_flags(boundaries)

    from gate_chunk_records import every_cell_record

    reference = {name: arrays[name].copy() for name in STATE_NAMES}
    reference_fused_electric_step(
        cp, arrays, reference, curl_coefficients, constitutive_coefficients,
        np.float32(dtdx), boundaries, flags,
        record=every_cell_record(arrays[TARGETS[0]].shape))

    kernel_curl = curl_coefficients
    kernel_constitutive = constitutive_coefficients
    kernel_codes = [0 if kind == shared.PERIODIC else 1 for kind in boundaries]
    kernel_flags = flags
    if mutation == "half_integer_curl_on_D":
        kernel_curl = shared.synthetic_coefficients(cp, shape, True)
    elif mutation == "integer_constitutive_on_E":
        kernel_constitutive = shared.synthetic_constitutive_coefficients(cp, shape, False)
    elif mutation == "swap_kps_kms":
        kernel_constitutive = shared._swapped_kps_kms(kernel_constitutive)
    elif mutation == "metallic_as_periodic":
        kernel_codes = [0, 0, 0]
    elif mutation == "drop_zero_metal_binding":
        kernel_flags = (False, False, False)

    fused = copy_state(arrays)
    unfused = copy_state(arrays)
    error = None
    try:
        run_fused(
            fused,
            shared.flatten_coefficients(kernel_curl),
            shared.flatten_coefficients(kernel_constitutive),
            kernel_codes,
            kernel_flags,
            dtdx,
            kernel=kernel,
            num_warps=num_warps,
        )
        run_unfused(
            unfused,
            shared.flatten_coefficients(curl_coefficients),
            shared.flatten_coefficients(constitutive_coefficients),
            [0 if kind == shared.PERIODIC else 1 for kind in boundaries],
            flags,
            dtdx,
        )
        cp.cuda.runtime.deviceSynchronize()
    except Exception as exc:  # noqa: BLE001 - the artifact owns the launch failure
        error = f"{type(exc).__name__}: {exc}"[:2000]
    case["launch_error"] = error
    if error is not None:
        case["vs_array"] = {"bit_identical": False, "differing_floats": -1}
        case["vs_two_kernels"] = {"bit_identical": False, "differing_floats": -1}
        return case

    case["vs_array"] = compare_state(fused, reference)
    case["vs_two_kernels"] = compare_state(fused, unfused)
    case["two_kernels_vs_array"] = compare_state(unfused, reference)
    return case


#: The shipped constitutive call (``forms.constitutive_form``), one per target: the
#: old field word ``f`` (a masked load), then ``prev``, ``src``, ``kps``, ``kms`` and the
#: dsigw membership that keys the select.
_CONSTITUTIVE_CALL = re.compile(
    r"a(?P<t>\d) = constitutive_form\((?P<f>tl\.load\(e(?P=t) \+ idx, "
    r"mask=live & dw(?P=t), other=0\.0\)),\s+prev(?P=t), src(?P=t), kp_(?P=t), "
    r"km_(?P=t), dw(?P=t)\)")


def _constant(name: str) -> int:
    """One of ``kernels.py``'s ``tl.constexpr`` globals as a plain int."""
    from meep_gpu.triton_kernels import kernels

    value = getattr(kernels, name)
    return int(getattr(value, "value", value))


def mutation_header() -> str:
    """The module text a mutated copy of the kernel compiles under.

    The kernel reads ``PERIODIC``/``METALLIC`` and the chunk record's two word
    offsets as ``tl.constexpr`` globals and calls the shared ``@triton.jit`` forms of
    ``triton_kernels/forms.py`` by their imported names. A copy compiled without them
    does not compile (Triton raises ``NameError`` at the first launch), and a launch
    error scores as a difference, so every must-differ mutation would read caught
    while measuring nothing. The constants are read from ``kernels.py``, not spelled.
    """
    return ("import triton\nimport triton.language as tl\n"
            "from meep_gpu.triton_kernels.forms import ("
            "axis_present, constitutive_form, curl_form, curl_stencil, record_in)\n"
            f"PERIODIC = tl.constexpr({_constant('PERIODIC')})\n"
            f"METALLIC = tl.constexpr({_constant('METALLIC')})\n"
            f"RECORD_HALF_INTEGER = tl.constexpr({_constant('RECORD_HALF_INTEGER')})\n"
            f"RECORD_PRESENCE = tl.constexpr({_constant('RECORD_PRESENCE')})\n\n")


#: The source mutations, each scored with its expectation in ``main``.
SOURCE_MUTATIONS = ("reload_D_from_memory", "reload_D_from_next_cell",
                    "inverse_epsilon_on_left", "drop_fw_store", "regroup_constitutive")


def mutated_source(name: str, source: str) -> Tuple[str, int]:
    """``source`` (the shipped kernel's text) with mutation ``name`` applied, and its
    site count. Every mutation rewrites one site per target, so 3 is the only count
    that arms it; the caller refuses any other."""
    hits = 0
    if name in ("reload_D_from_memory", "reload_D_from_next_cell"):
        for index in range(3):
            old = f"src{index} = v{index} * tl.load(ie{index} + idx, mask=live, other=0.0)"
            offset = "idx" if name == "reload_D_from_memory" else "(idx + 1) % n_elem"
            new = (f"src{index} = tl.load(f{index} + {offset}, mask=live, other=0.0) "
                   f"* tl.load(ie{index} + idx, mask=live, other=0.0)")
            count = source.count(old)
            source = source.replace(old, new)
            hits += count
    elif name == "inverse_epsilon_on_left":
        for index in range(3):
            old = f"src{index} = v{index} * tl.load(ie{index} + idx, mask=live, other=0.0)"
            new = f"src{index} = tl.load(ie{index} + idx, mask=live, other=0.0) * v{index}"
            count = source.count(old)
            source = source.replace(old, new)
            hits += count
    elif name == "drop_fw_store":
        # The auxiliary store is masked to the accumulation's cells (the auxiliary
        # contract); every cell is one under the gate's every-cell record, so dropping
        # it leaves f_w at its pre-step word everywhere.
        for index in range(3):
            old = f"    tl.store(w{index} + idx, src{index}, mask=live & dw{index})\n"
            count = source.count(old)
            source = source.replace(old, "")
            hits += count
    elif name == "regroup_constitutive":
        # INVERTED (KERNELS.md section 7.1): the shipped form is MEEP's
        # ``f + ((kps*src) - (kms*prev))`` (``forms.constitutive_form``), so the
        # mutant puts the two separate accumulations the kernel carried until 0.10
        # back at the call site, ``(f + kps*src) - kms*prev``, with the same select.
        hits = len(_CONSTITUTIVE_CALL.findall(source))
        source = _CONSTITUTIVE_CALL.sub(
            lambda m: (f"a{m['t']} = tl.where(dw{m['t']}, ({m['f']} + kp_{m['t']} * "
                       f"src{m['t']}) - km_{m['t']} * prev{m['t']}, src{m['t']})"),
            source)
    else:  # pragma: no cover - argparse constrains this
        raise ValueError(name)
    return source, hits


def mutate_kernel(name: str):
    """Compile one derived copy of the shipped D fused kernel from a real file."""
    from meep_gpu.triton_kernels import kernels

    source = textwrap.dedent(inspect.getsource(kernels.fused_curl_constitutive_D.fn))
    source, hits = mutated_source(name, source)
    if hits != 3:
        raise RuntimeError(f"mutation {name} hit {hits} sites, expected 3")

    handle = tempfile.NamedTemporaryFile(
        "w", suffix="_fused_D_mutation.py", delete=False, encoding="utf-8")
    handle.write(mutation_header() + source)
    handle.close()
    spec = importlib.util.spec_from_file_location(
        f"fused_D_mutation_{name}", handle.name)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import mutation {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.fused_curl_constitutive_D, handle.name, hits


def provenance(num_warps: Optional[int]) -> Dict[str, Any]:
    from meep_gpu.triton_kernels import coverage, kernels, launch
    import triton

    device = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(device.id)
    name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
    files = [Path(kernels.__file__), Path(launch.__file__), Path(coverage.__file__), Path(__file__)]
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "cupy": cp.__version__,
        "triton": triton.__version__,
        "driver": int(cp.cuda.runtime.driverGetVersion()),
        "runtime": int(cp.cuda.runtime.runtimeGetVersion()),
        "device": {"logical_id": int(device.id), "name": name},
        "num_warps": num_warps,
        "sources": {str(path.relative_to(API_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in files},
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repo-root", default=str(API_ROOT))
    parser.add_argument("--num-warps", type=int, default=1)
    args = parser.parse_args(argv)
    if cp is None:
        raise SystemExit("CuPy is required for the CUDA gate")
    from composition_policy import (  # noqa: PLC0415 - beside this file
        confirm_campaign_policy, stamp_campaign_policy)

    out = Path(args.out).resolve()
    results: Dict[str, Any] = {
        "gate": "triton_fused_electric",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_cases": [],
        "mutations": [],
    }
    # THE CAMPAIGN'S SUBNORMAL POLICY, installed and measured before the first kernel
    # compiles and stamped into the artifact (composition_policy.py). A refusal writes
    # ``summary.status = refused`` and the run measures nothing.
    refused = stamp_campaign_policy(results, cp)
    results["provenance"] = provenance(args.num_warps)
    save(results, out)
    if refused:
        log(f"FUSED ELECTRIC GATE REFUSED: {refused}")
        return 2

    combinations = [
        (source, shape, boundaries, dtdx)
        for source in ("synthetic", "real")
        for shape in shared.PML_SHAPES
        for boundaries in shared.PML_BOUNDARY_SETS
        for dtdx in shared.PML_DTDX
    ]
    started = time.time()
    for index, (source, shape, boundaries, dtdx) in enumerate(combinations, 1):
        case_started = time.time()
        case = one_case(
            shape, boundaries, dtdx, source, args.repo_root,
            num_warps=args.num_warps)
        case["seconds"] = round(time.time() - case_started, 3)
        results["base_cases"].append(case)
        verdict = case.get("vs_array", {})
        log(f"[base] {index}/{len(combinations)} {source} {shape} {boundaries} "
            f"dtdx={dtdx}: identical={verdict.get('bit_identical', False)} "
            f"diff={verdict.get('differing_floats', -1)} ({case['seconds']} s)")
        save(results, out)

    mutation_specs = (
        ("half_integer_curl_on_D", True, None),
        ("integer_constitutive_on_E", True, None),
        ("swap_kps_kms", True, None),
        ("metallic_as_periodic", True, None),
        ("drop_zero_metal_binding", True, None),
        ("reload_D_from_memory", False, "reload_D_from_memory"),
        ("reload_D_from_next_cell", True, "reload_D_from_next_cell"),
        ("inverse_epsilon_on_left", False, "inverse_epsilon_on_left"),
        ("drop_fw_store", True, "drop_fw_store"),
        ("regroup_constitutive", True, "regroup_constitutive"),
    )
    temporary_files = []
    for index, (name, must_differ, source_mutation) in enumerate(mutation_specs, 1):
        kernel = None
        hits = None
        if source_mutation is not None:
            kernel, temporary, hits = mutate_kernel(source_mutation)
            temporary_files.append(temporary)
        case = one_case(
            shared.PML_SHAPES[0], shared.PML_BOUNDARY_SETS[3], 0.35,
            "synthetic", args.repo_root, kernel=kernel,
            mutation=None if source_mutation is not None else name,
            num_warps=args.num_warps)
        identical = bool(case.get("vs_array", {}).get("bit_identical", False))
        row = {
            "name": name,
            "must_differ": must_differ,
            "caught": (not identical) if must_differ else identical,
            "identical": identical,
            "source_sites": hits,
            "case": case,
        }
        results["mutations"].append(row)
        log(f"[mutation] {index}/{len(mutation_specs)} {name}: "
            f"identical={identical} expected={'different' if must_differ else 'identical'} "
            f"caught={row['caught']}")
        save(results, out)

    # THE POLICY MUST STILL HOLD at the end of the run, on every executor and on the
    # host FPU; the stamp is re-taken with the counters the run accumulated.
    refused = confirm_campaign_policy(results)
    if refused:
        results["elapsed_seconds"] = round(time.time() - started, 2)
        results["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save(results, out)
        log(f"FUSED ELECTRIC GATE REFUSED: {refused}")
        return 2

    ran = [case for case in results["base_cases"] if not case.get("skipped")]
    base_pass = bool(ran) and all(
        case["vs_array"]["bit_identical"]
        and case["vs_two_kernels"]["bit_identical"]
        and case["two_kernels_vs_array"]["bit_identical"]
        for case in ran)
    mutation_pass = all(row["caught"] for row in results["mutations"])
    results["summary"] = {
        "ran": len(ran),
        "skipped": len(results["base_cases"]) - len(ran),
        "identical_vs_array": sum(
            int(case["vs_array"]["bit_identical"]) for case in ran),
        "identical_vs_two_kernels": sum(
            int(case["vs_two_kernels"]["bit_identical"]) for case in ran),
        "mutations_caught": sum(int(row["caught"]) for row in results["mutations"]),
        "mutations_total": len(results["mutations"]),
        "pass": base_pass and mutation_pass,
    }
    results["elapsed_seconds"] = round(time.time() - started, 2)
    results["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save(results, out)
    log(f"[done] base={results['summary']['identical_vs_array']}/{len(ran)} "
        f"mutations={results['summary']['mutations_caught']}/"
        f"{results['summary']['mutations_total']} pass={results['summary']['pass']} "
        f"({results['elapsed_seconds']} s) -> {out}")
    return 0 if results["summary"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
