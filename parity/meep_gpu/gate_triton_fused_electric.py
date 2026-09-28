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
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
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
) -> None:
    """Independent transcription of the three passes in their driver order."""
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
        "kernel_order" if regrouped else "array_order",
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
    from meep_gpu.triton_kernels.launch import plan_fused_pair_from_arrays

    plan = plan_fused_pair_from_arrays(
        "D", arrays, curl_flat, constitutive_flat, codes, zero_metal, dtdx,
        kernel=kernel, num_warps=num_warps)
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
    from meep_gpu.triton_kernels.launch import (
        plan_constitutive_from_arrays,
        plan_from_arrays,
    )

    plan_from_arrays("step_D", arrays, curl_flat, codes, dtdx).run(guard=guard)
    reference_zero_metal_D(arrays, zero_metal)
    plan_constitutive_from_arrays("E", arrays, constitutive_flat).run(guard=guard)


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

    reference = {name: arrays[name].copy() for name in STATE_NAMES}
    reference_fused_electric_step(
        cp, arrays, reference, curl_coefficients, constitutive_coefficients,
        np.float32(dtdx), boundaries, flags)

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


def mutate_kernel(name: str):
    """Compile one derived copy of the shipped D fused kernel from a real file."""
    from meep_gpu.triton_kernels import kernels

    source = textwrap.dedent(inspect.getsource(kernels.fused_curl_constitutive_D.fn))
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
        for index in range(3):
            old = f"    tl.store(w{index} + idx, src{index}, mask=live)\n"
            count = source.count(old)
            source = source.replace(old, "")
            hits += count
    elif name == "regroup_constitutive":
        for index in range(3):
            old = (f"a{index} = a{index} + kp_{index} * src{index}\n"
                   f"    a{index} = a{index} - km_{index} * prev{index}")
            new = (f"a{index} = a{index} + (kp_{index} * src{index} "
                   f"- km_{index} * prev{index})\n    a{index} = a{index}")
            count = source.count(old)
            source = source.replace(old, new)
            hits += count
    else:  # pragma: no cover - argparse constrains this
        raise ValueError(name)
    if hits != 3:
        raise RuntimeError(f"mutation {name} hit {hits} sites, expected 3")

    handle = tempfile.NamedTemporaryFile(
        "w", suffix="_fused_D_mutation.py", delete=False, encoding="utf-8")
    handle.write(
        "import triton\nimport triton.language as tl\n"
        "PERIODIC = tl.constexpr(0)\nMETALLIC = tl.constexpr(1)\n\n" + source)
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
    out = Path(args.out).resolve()
    results: Dict[str, Any] = {
        "gate": "triton_fused_electric",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": provenance(args.num_warps),
        "base_cases": [],
        "mutations": [],
    }
    save(results, out)

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
