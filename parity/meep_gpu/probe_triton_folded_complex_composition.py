"""Complete-driver composition gate for Phase B — the fold under complex storage.

The sub-step gate (``gate_triton_folded_complex.py``) certifies each kernel
against an in-file transcription. This complementary probe asks the COMPOSITION
question: with the folded curls, the two ghost-fill passes and the constitutive
delegations installed by monkeypatch — production dispatch untouched,
``plan_fast_path`` still returning ``None`` — does the complete CuPy driver state
reproduce the array-path driver EXACTLY after every step, for a STATED number of
steps, sources, PML, boundary fills and metal zeroing included?

THE DRIVER ORDER IS FIVE SLOTS PER HALF, NOT THREE (driver.py:3206-3226, read
from source; ``symmetry.py:44`` cites a stale ``driver.py:3163-3183`` and names
three):

    magnetic:  step_B -> inject -> fill_symmetry_bc_B -> zero_metal_B
                      -> fill_folded_far_ghosts_B -> update_H
    electric:  step_D -> inject -> fill_symmetry_bc_D -> zero_metal_D
                      -> fill_folded_far_ghosts_D -> update_E -> update_P

The two metallic passes sit BETWEEN the two fills, and the far-ghost fill is the
LAST thing before the constitutive sub-step. A composed plan MUST NOT fuse across
any of those seams — which is exactly why
``FoldedMirrorGhostFillComplexPlan`` exposes ``run_near`` and ``run_far``
separately rather than one fused per-axis launch: measured, fusing them reorders
the doubly-unowned corner of a component that is NEAR on one folded axis and FAR
on another, and complex float multiplication is not associative.

TWO ASSERTION LAYERS, because one is measurably blind. ``special_kz.py:311-320``
records that an inlined helper's spelling once made the STORED layer blind to a
class the PRODUCT layer caught, so every case is compared

* at the STORED layer, after ``update_H`` / ``update_E`` — the bytes a monitor
  would read; and
* at the PRODUCT layer, through a wrapper kernel that stores the parity helper's
  output words DIRECTLY, so the fill's complex product is visible before any
  downstream rounding can hide it.

EVERY SOURCE IS PROVEN NON-VACUOUS against a source-free control: the injected
bytes must DIFFER, or the case measured a quiet grid and its source clause is
worthless.

ZERO-INIT + SIGNED-ZERO CENSUS IS MANDATORY for a complex tranche. One case
initialises every array to ``+0.0`` under a thin absorber whose deepest
``kms = kappa - sigma`` goes NEGATIVE, and counts ``0x80000000`` words after
every step. A census of 0 FAILS the case as VACUOUS: random-seeded states are
provably blind to the signed-zero class, and jobs 2343/2345 proved that class
reachable at the driver level.

SUBNORMAL POLICY. This probe runs UNDER THE STRIPPED IEEE-KEEP POLICY — the ship
configuration — installed via ``gate_triton_complex.install_ftz_strip`` BEFORE
any CuPy compile, never re-implemented here and never as a ``-ftz=false`` user
option (NVRTC rejects the duplicate). A probe artifact handed in with
``--probe-artifact`` must certify that same policy and carry the PARITY pattern.

Usage (the GPU host, one clear device; the cache dir must carry the policy token)::

    CUDA_VISIBLE_DEVICES=0 \\
    CUPY_CACHE_DIR=$RUN_ROOT/results/cupy_cache_ftz_stripped_$JOB \\
        python -u probe_triton_folded_complex_composition.py \\
        --out results/triton_folded_complex_<date>/composition.json

Laptop (no CUDA/Triton): ``--self-check`` builds every case's drivers on NumPy,
steps the array path two steps per case, pins the plan/predicate surface, and
says in so many words that the Triton legs were skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
API_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
if API_ROOT not in sys.path:
    sys.path.insert(0, API_ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import gate_triton_complex as gate  # noqa: E402 - policy machinery + seeding
import gate_triton_folded_complex as fold_gate  # noqa: E402

SEED = gate.SEED
save = gate.save
log = gate.log

STATE_NAMES = gate.ALL_STATE
PRIMARY_NAMES = ("Bx", "By", "Bz", "Dx", "Dy", "Dz")
REQUIRED_STATE = set(STATE_NAMES)

BETA_EIGSRC = fold_gate.BETA_EIGSRC
BETA_GRATING_13_2 = fold_gate.BETA_GRATING_13_2
BETA_GRATING_17_7 = fold_gate.BETA_GRATING_17_7
KX_GRATING_13_2 = fold_gate.KX_GRATING_13_2
KX_GRATING_17_7 = fold_gate.KX_GRATING_17_7

#: The five slots per half a composed plan may not fuse across.
DRIVER_SLOTS_PER_HALF = (
    ("step_B", "inject", "fill_symmetry_bc_B", "zero_metal_B",
     "fill_folded_far_ghosts_B", "update_H"),
    ("step_D", "inject", "fill_symmetry_bc_D", "zero_metal_D",
     "fill_folded_far_ghosts_D", "update_E", "update_P"),
)


def gaussian_source(component: str) -> Dict[str, Any]:
    return {"component": component, "source_type": "gaussian",
            "frequency": 0.8, "fwidth": 0.4,
            "center": (0.0, 0.0, 0.0), "size": (0.0, 0.0, 0.0),
            "amplitude": 1.0}


def cw_source(component: str, amplitude: complex) -> Dict[str, Any]:
    return {"component": component, "frequency": 0.7,
            "center": (0.0, 0.0, 0.0), "size": (0.0, 0.0, 0.0),
            "amplitude": amplitude}


#: name, cell, resolution, symmetry, boundaries, k_point, beta, courant, pml,
#: sources, steps, zero_init.
#:
#: Every row exists because it is the only witness to something: the two
#: terminations, both count parities, both plane phases, a two-axis fold with
#: MIXED count parity, an in-plane Bloch phase beside a fold, both beta storages,
#: sources on each half, and the zero-init census case.
CASES: Tuple[Tuple[Any, ...], ...] = (
    ("f1_fold_y_periodic_even", (3.0, 3.0, 0.0), 12.0, (("Y", 1),), "periodic",
     (0.0, 0.0, 0.0), 0.0, 0.5, {"x": 5, "y": {"high": 5}}, (), 8, False),
    ("f2_fold_y_periodic_ODD", (3.0, 3.1, 0.0), 12.0, (("Y", 1),), "periodic",
     (0.0, 0.0, 0.0), 0.0, 0.35, {"x": 5, "y": {"high": 5}}, (), 8, False),
    ("f3_fold_y_periodic_odd_plane", (3.0, 3.0, 0.0), 12.0, (("Y", -1),),
     "periodic", (0.0, 0.0, 0.0), 0.0, 0.35, {"x": 5, "y": {"high": 5}}, (),
     8, False),
    ("f4_fold_y_metallic", (3.0, 3.0, 0.0), 12.0, (("Y", 1),),
     ("periodic", "metallic", "periodic"), (0.0, 0.0, 0.0), 0.0, 0.35,
     {"x": 5}, (), 8, False),
    ("f5_fold_xy_mixed_parity", (3.0, 3.1, 0.0), 12.0, (("X", 1), ("Y", -1)),
     "periodic", (0.0, 0.0, 0.0), 0.0, 0.35,
     {"x": {"high": 5}, "y": {"high": 5}}, (), 8, False),
    ("f6_electric_source", (3.0, 3.0, 0.0), 12.0, (("Y", 1),), "periodic",
     (0.0, 0.0, 0.0), 0.0, 0.35, {"x": 5, "y": {"high": 5}},
     (gaussian_source("Ez"),), 8, False),
    # Hy is EVEN under an even Y plane; Hx/Hz/Ey are odd and a point source on
    # the plane is refused outright by `sources._validate_symmetry_parity`, so
    # the magnetic seam is closed with the component the plane admits.
    ("f7_magnetic_source", (3.0, 3.0, 0.0), 12.0, (("Y", 1),), "periodic",
     (0.0, 0.0, 0.0), 0.0, 0.35, {"x": 5, "y": {"high": 5}},
     (cw_source("Hy", complex(0.75, 0.4)),), 8, False),
    ("f8_beta_fold_y", (3.0, 3.0, 0.0), 12.0, (("Y", 1),), "periodic",
     (0.0, 0.0, 0.0), BETA_EIGSRC, 0.35, {"x": 5, "y": {"high": 5}}, (), 8,
     False),
    ("f9_beta_fold_y_bloch_kx", (3.0, 3.0, 0.0), 12.0, (("Y", 1),), "periodic",
     (0.23, 0.0, 0.0), BETA_GRATING_13_2, 0.35, {"x": 5, "y": {"high": 5}},
     (), 8, False),
    ("f10_zero_init_census", (2.0, 2.1, 0.0), 10.0, (("Y", 1),), "periodic",
     (0.0, 0.0, 0.0), 0.0, 0.35, {"x": 3, "y": {"high": 3}}, (), 4, True),
)
CORE_BUDGET = sum(case[10] for case in CASES)


def build_driver(xp, spec, seed: int, prefer_gpu: bool, with_sources: bool):
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415
    from meep_gpu.grid import Mirror  # noqa: PLC0415

    (_name, cell, resolution, symmetry, boundaries, k_point, beta, courant,
     pml_spec, sources, _steps, zero_init) = spec
    driver = FdtdDriver(
        cell_size=cell, resolution=resolution, dimensions=2,
        force_complex_fields=True, courant=courant,
        symmetry=tuple(Mirror(axis, phase) for axis, phase in symmetry),
        boundaries=boundaries, k_point=k_point, beta=beta,
        prefer_gpu=prefer_gpu, gpu_id=0,
    )
    shape = driver.shape
    index = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    epsilon = np.ascontiguousarray(
        (1.45 + 0.30 * np.sin(index * np.float32(0.037))).astype(np.float32))
    driver.set_epsilon(xp.asarray(epsilon))
    driver.setup_pml(dict(pml_spec))
    if with_sources:
        for declaration in sources:
            driver.add_source(dict(declaration))
    if zero_init:
        # EVERY array +0.0. The absorber's deepest kms goes negative, which is
        # what produces -0.0 stores from a quiet grid; a census of 0 would make
        # the case vacuous.
        for name in STATE_NAMES:
            volume = getattr(driver.fields, name, None)
            if volume is not None:
                volume[...] = 0
        return driver
    rng = np.random.default_rng(seed)
    for primary in PRIMARY_NAMES:
        host = gate._seed_host_complex(tuple(shape), rng, scale=0.25)
        driver.set_field(primary, xp.asarray(host))
    return driver


def state(driver) -> Dict[str, Any]:
    return {name: getattr(driver.fields, name) for name in STATE_NAMES
            if getattr(driver.fields, name, None) is not None}


def compare(xp, left, right) -> Dict[str, Any]:
    a = np.ascontiguousarray(_host(xp, left)).view(np.uint32).ravel()
    b = np.ascontiguousarray(_host(xp, right)).view(np.uint32).ravel()
    return {"bit_identical": bool(np.array_equal(a, b)),
            "differing_floats": int(np.count_nonzero(a != b)),
            "total_floats": int(a.size)}


def _host(xp, array):
    return xp.asnumpy(array) if hasattr(xp, "asnumpy") else np.asarray(array)


def census(xp, driver) -> int:
    """``0x80000000`` words across the stored primaries and the six ``fu_*``."""
    return fold_gate.signed_zero_census(xp, state(driver))


# ---------------------------------------------------------------------------
# Plans, launch counters and the monkeypatch
# ---------------------------------------------------------------------------

class SlotCounter:
    """A monkeypatch that silently fell through for one slot would leave BOTH
    sides on the array path and every step would pass vacuously, so every
    patched slot proves it launched exactly once per complete step."""

    def __init__(self, call) -> None:
        self.call = call
        self.launches = 0

    def run(self) -> None:
        self.launches += 1
        self.call()


def build_plans(driver, record) -> Tuple[Dict[str, SlotCounter], Dict[str, Any]]:
    """Six patched slots per complete step, each its own plan or delegation.

    The two fill passes are SEPARATE slots — the driver runs ``zero_metal_*``
    between them, and under complex storage fusing them is byte-visible.
    """
    from meep_gpu.triton_kernels import folded_complex  # noqa: PLC0415

    fields, pml = driver.fields, driver.pml
    beta = float(getattr(driver.fields.grid, "beta", 0.0))
    curl = (folded_complex.plan_folded_beta_bloch_pml_curl if beta != 0.0
            else folded_complex.plan_folded_complex_pml_curl)
    plans: Dict[str, Any] = {
        "step_B": curl(fields, pml, "step_B", probe=record),
        "step_D": curl(fields, pml, "step_D", probe=record),
        "update_H": folded_complex.plan_folded_complex_constitutive(
            fields, pml, "H", probe=record),
        "update_E": folded_complex.plan_folded_complex_constitutive(
            fields, pml, "E", probe=record),
    }
    fills = {family: folded_complex.plan_folded_mirror_ghost_fill_complex(
        fields, family, probe=record) for family in ("B", "D")}
    missing = sorted([slot for slot, plan in plans.items() if plan is None]
                     + [f"fill_{f}" for f, plan in fills.items() if plan is None])
    if missing:
        reasons: Dict[str, Any] = {}
        for slot in missing:
            if slot.startswith("fill_"):
                reasons[slot] = list(
                    folded_complex.folded_mirror_ghost_fill_complex_coverage(
                        fields, slot[-1], probe=record).reasons)
            elif slot.startswith("step_"):
                verdict = (folded_complex.folded_beta_bloch_pml_curl_coverage
                           if beta != 0.0 else
                           folded_complex.folded_complex_pml_curl_coverage)
                reasons[slot] = list(verdict(fields, pml, slot,
                                             probe=record).reasons)
            else:
                reasons[slot] = list(
                    folded_complex.folded_complex_constitutive_coverage(
                        fields, pml, slot[-1], probe=record).reasons)
        raise AssertionError(f"plans refused for {missing}: "
                             + json.dumps(reasons, indent=1))

    counters = {
        "step_B": SlotCounter(plans["step_B"].run),
        "step_D": SlotCounter(plans["step_D"].run),
        "update_H": SlotCounter(plans["update_H"].run),
        "update_E": SlotCounter(plans["update_E"].run),
        "fill_symmetry_bc_B": SlotCounter(fills["B"].run_near),
        "fill_symmetry_bc_D": SlotCounter(fills["D"].run_near),
        "fill_folded_far_ghosts_B": SlotCounter(fills["B"].run_far),
        "fill_folded_far_ghosts_D": SlotCounter(fills["D"].run_far),
    }
    return counters, {}


def install(driver_module, counters: Dict[str, SlotCounter], owner):
    """Serve the patched slots from the plans, for the OWNER fields only.

    Sources, ``zero_metal_*`` and ``update_P`` stay the driver's own, and the
    five-slot order is untouched — this probe measures the composition, it does
    not rewrite it.
    """
    names = tuple(counters)
    originals = {name: getattr(driver_module, name) for name in names}

    def wrapper(name):
        def wrapped(fields, pml=None):
            if fields is owner:
                counters[name].run()
                return None
            original = originals[name]
            try:
                return original(fields, pml)
            except TypeError:
                return original(fields)
        return wrapped

    for name in names:
        setattr(driver_module, name, wrapper(name))
    return lambda: [setattr(driver_module, name, function)
                    for name, function in originals.items()]


# ---------------------------------------------------------------------------
# The PRODUCT layer — a wrapper kernel storing the parity helper's words
# ---------------------------------------------------------------------------

PRODUCT_PROBE_SOURCE = '''
@triton.jit
def parity_product_probe(z, out, n_elem, c_re, c_im,
                         EXPANSION: tl.constexpr, BLOCK: tl.constexpr):
    """Store ``_mul_imag_coefficient_left(c, z)``'s words DIRECTLY.

    The stored layer can be blind to a class the product layer catches
    (special_kz.py:311-320's measured lesson), so the fill's complex product is
    asserted here too, before any downstream rounding can hide it.
    """
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = idx < n_elem
    z_re = tl.load(z + 2 * idx, mask=live, other=0.0)
    z_im = tl.load(z + 2 * idx + 1, mask=live, other=0.0)
    o_re, o_im = _mul_imag_coefficient_left(c_re, c_im, z_re, z_im, EXPANSION)
    tl.store(out + 2 * idx, o_re, mask=live)
    tl.store(out + 2 * idx + 1, o_im, mask=live)
'''


def run_product_layer(xp, expansion: int) -> Dict[str, Any]:
    """The parity product's own words against the array path's, byte for byte."""
    import inspect  # noqa: PLC0415

    from meep_gpu.triton_kernels import special_kz  # noqa: PLC0415

    source = (inspect.getsource(special_kz._mul_imag_coefficient_left.fn)
              + "\n\n" + PRODUCT_PROBE_SOURCE)
    kernel = fold_gate.compile_mutated(source, "parity_product_probe")

    words = np.float32([0.0, -0.0, 1e-45, -1e-45, 7e-45, -7e-45, 1.5, -1.5,
                        3.4e38, -3.4e38, 0.75, -0.25, 1.0, -1.0, 1e-40, -1e-40])
    grid_re, grid_im = np.meshgrid(words, words, indexing="ij")
    plane = np.empty(grid_re.shape, dtype=np.complex64)
    plane.real = grid_re
    plane.imag = grid_im
    cases: List[Dict[str, Any]] = []
    for phase in (1, -1):
        for label, coefficient in (("near", phase), ("far", -phase)):
            device = xp.asarray(np.ascontiguousarray(plane))
            out = xp.zeros_like(device)
            n_elem = int(plane.size)
            c = np.complex64(coefficient)
            kernel[((n_elem + 255) // 256,)](
                _pointer(device), _pointer(out), n_elem,
                float(np.float32(c.real)), float(np.float32(c.imag)),
                EXPANSION=expansion, BLOCK=256,
                enable_fp_fusion=_fusion())
            xp.cuda.runtime.deviceSynchronize()
            reference = coefficient * plane          # what stepping.py does
            verdict = compare(xp, out, xp.asarray(np.ascontiguousarray(reference)))
            planewise = np.ascontiguousarray(plane).view(np.float32) * np.float32(
                coefficient)
            blind = compare(xp, xp.asarray(np.ascontiguousarray(reference)),
                            xp.asarray(planewise.view(np.complex64)))
            cases.append({"phase": phase, "fill": label, "verdict": verdict,
                          "planewise_would_differ_in": blind["differing_floats"]})
            log(f"[product] phase{phase:+d}/{label}: "
                f"identical={verdict['bit_identical']} "
                f"differing={verdict['differing_floats']}/{verdict['total_floats']} "
                f"planewise_would_differ_in={blind['differing_floats']}")
    return {"ran": len(cases),
            "identical": sum(int(c["verdict"]["bit_identical"]) for c in cases),
            "cases": cases}


def _pointer(array):
    from meep_gpu.triton_kernels.launch import CupyPointer  # noqa: PLC0415

    return CupyPointer(array.view(np.float32))


def _fusion():
    from meep_gpu.triton_kernels.kernels import ENABLE_FP_FUSION  # noqa: PLC0415

    return ENABLE_FP_FUSION


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------

def run_case(xp, record, spec, results: Dict[str, Any],
             out_path: str) -> Dict[str, Any]:
    from meep_gpu import driver as driver_module  # noqa: PLC0415

    (name, _cell, _resolution, _symmetry, _boundaries, _k, _beta, _courant,
     _pml, sources, steps, zero_init) = spec
    started = time.time()
    reference = build_driver(xp, spec, 20260812, prefer_gpu=True,
                             with_sources=True)
    candidate = build_driver(xp, spec, 20260812, prefer_gpu=True,
                             with_sources=True)
    control = (build_driver(xp, spec, 20260812, prefer_gpu=True,
                            with_sources=False) if sources else None)
    undo = lambda: None  # noqa: E731
    case: Dict[str, Any] = {"name": name, "steps": steps,
                            "step_budget_stated": steps,
                            "zero_init": zero_init}
    try:
        counters, refusals = build_plans(candidate, record)
        case["expected_refusals"] = refusals
        missing = REQUIRED_STATE - set(state(candidate))
        if missing:
            raise AssertionError(f"{name}: state inventory is short: "
                                 f"{sorted(missing)}")
        undo = install(driver_module, counters, candidate.fields)
        per_step: List[Dict[str, Any]] = []
        censuses: List[int] = []
        for step in range(steps):
            reference.step()
            candidate.step()
            if control is not None:
                control.step()
            left, right = state(candidate), state(reference)
            verdict = {n: compare(xp, left[n], right[n]) for n in sorted(left)}
            worst = sorted(((v["differing_floats"], n)
                            for n, v in verdict.items()), reverse=True)[:3]
            per_step.append({
                "step": step,
                "bit_identical": all(v["bit_identical"] for v in verdict.values()),
                "worst": [{"name": n, "differing": d} for d, n in worst],
                "launches": {slot: counter.launches
                             for slot, counter in counters.items()},
            })
            censuses.append(census(xp, reference))
            log(f"  step {step + 1}/{steps} [{name}] identical="
                f"{per_step[-1]['bit_identical']} census={censuses[-1]} "
                f"launches={sorted(set(c.launches for c in counters.values()))}")
        case["per_step"] = per_step
        case["census"] = censuses
        case["identical"] = all(entry["bit_identical"] for entry in per_step)

        expected = {slot: steps for slot in counters}
        actual = {slot: counter.launches for slot, counter in counters.items()}
        case["launch_counts"] = actual
        case["all_slots_launched_once_per_step"] = actual == expected
        if actual != expected:
            case["failure"] = (
                f"a patched slot did not launch once per step: expected "
                f"{expected}, got {actual} — the monkeypatch fell through and "
                f"both sides ran the array path")

        if control is not None:
            injected = {n: compare(xp, state(reference)[n], state(control)[n])
                        for n in PRIMARY_NAMES}
            differing = sum(v["differing_floats"] for v in injected.values())
            case["source_non_vacuous"] = differing > 0
            case["source_injected_words"] = differing
            if differing == 0:
                case["failure"] = ("the source injected NOTHING: the case "
                                   "measured a quiet grid and its source clause "
                                   "is worthless")
        if zero_init:
            case["census_vacuous"] = any(count == 0 for count in censuses)
            if case["census_vacuous"]:
                case["failure"] = ("a signed-zero census of 0 means the class "
                                   "was never REACHED; the case is VACUOUS, "
                                   "not passed")
    except Exception as exc:  # noqa: BLE001 - a refusal is a result, recorded
        case["error"] = f"{type(exc).__name__}: {exc}"
        case["identical"] = False
    finally:
        undo()
    case["seconds"] = round(time.time() - started, 3)
    return case


# ---------------------------------------------------------------------------
# Self-check (laptop): NumPy only, no Triton, no GPU
# ---------------------------------------------------------------------------

def self_check(out_path: str) -> int:
    """Build every case on NumPy, step the array path, and pin the surface.

    Says in so many words that the Triton legs were skipped — a self-check that
    reads like a pass is worse than no self-check.
    """
    from meep_gpu.triton_kernels import folded_complex  # noqa: PLC0415

    results: Dict[str, Any] = {
        "mode": "self_check",
        "note": ("NumPy only: NO Triton kernel ran, NO byte comparison against "
                 "a kernel was made, and NOTHING here licenses the composition. "
                 "This leg proves the cases BUILD and that the predicate surface "
                 "answers."),
        "driver_slots_per_half": [list(half) for half in DRIVER_SLOTS_PER_HALF],
        "cases": [],
        "core_budget": CORE_BUDGET,
    }
    record = {"backend": "cupy",
              "patterns": {name: "FMA_V1" for name in
                           set(folded_complex.PARITY_PROBE_PATTERNS)
                           | set(folded_complex.FOLDED_BETA_PROBE_PATTERNS)},
              "subnormal_policy": {"policy": gate.active_policy_name(),
                                   "nvrtc_calls": 1, "ftz_removed": 1}}
    failures = 0
    for index, spec in enumerate(CASES, start=1):
        started = time.time()
        name = spec[0]
        driver = build_driver(np, spec, 20260812, prefer_gpu=False,
                              with_sources=True)
        grid = driver.fields.grid
        codes, reasons = folded_complex.folded_axis_kinds(grid, driver.pml)
        entries = (folded_complex.ghost_fill_axis_entries(grid, "B")
                   if codes is not None else ())
        for _ in range(2):
            driver.step()
        # Only the CuPy clause may fire; anything else is a real refusal.
        beta = float(getattr(grid, "beta", 0.0))
        verdict = (folded_complex.folded_beta_bloch_pml_curl_coverage
                   if beta != 0.0 else
                   folded_complex.folded_complex_pml_curl_coverage)(
            driver.fields, driver.pml, "step_B", probe=record)
        blocking = [r for r in verdict.reasons if "array module" not in r]
        fill_verdict = folded_complex.folded_mirror_ghost_fill_complex_coverage(
            driver.fields, "B", probe=record)
        fill_blocking = [r for r in fill_verdict.reasons
                         if "array module" not in r]
        ok = not blocking and not fill_blocking and codes is not None
        failures += 0 if ok else 1
        results["cases"].append({
            "name": name, "shape": [int(n) for n in grid.shape],
            "stored": [int(grid.stored_cells(a)) for a in range(3)],
            "owned": [int(grid.owned_cells(a)) for a in range(3)],
            "codes": list(codes or ()), "fold_reasons": list(reasons),
            "entries": [{k: (list(v) if isinstance(v, tuple) else v)
                         for k, v in e.items()} for e in entries],
            "curl_blocking_reasons": blocking,
            "fill_blocking_reasons": fill_blocking,
            "ok": ok, "seconds": round(time.time() - started, 3)})
        log(f"case {index}/{len(CASES)} [self-check] {name} "
            f"shape={results['cases'][-1]['shape']} "
            f"codes={results['cases'][-1]['codes']} ok={ok}"
            + ("" if ok else f" blocking={blocking + fill_blocking}"))
        save(results, out_path)
    results["passed"] = failures == 0
    save(results, out_path)
    log(f"[self-check] {len(CASES) - failures}/{len(CASES)} cases build and "
        f"admit; NO Triton leg ran")
    return 0 if failures == 0 else 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def environment(xp) -> Dict[str, Any]:
    record: Dict[str, Any] = {"cupy": xp is not None}
    if xp is None:
        return record
    try:
        device = xp.cuda.runtime.getDeviceProperties(0)
        record["device"] = device["name"].decode()
    except Exception:  # noqa: BLE001
        record["device"] = "unknown"
    record["visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    record["cupy_cache_dir"] = os.environ.get("CUPY_CACHE_DIR")
    return record


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.path.join(
        HERE, "results", "triton_folded_complex", "composition.json"))
    parser.add_argument("--probe-artifact", default=None)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    if args.self_check:
        return self_check(args.out)

    try:
        import cupy as xp
    except ImportError:
        log("no cupy on this host; use --self-check for the laptop leg")
        return 2
    gate.install_ftz_strip()

    record: Optional[Dict[str, Any]] = None
    if args.probe_artifact and os.path.exists(args.probe_artifact):
        with open(args.probe_artifact, "r", encoding="utf-8") as handle:
            record = json.load(handle)
    else:
        record = fold_gate.measure_folded_expansion_record(xp, "cupy")

    from meep_gpu.triton_kernels import folded_complex  # noqa: PLC0415

    results: Dict[str, Any] = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "environment": environment(xp),
        "subnormal_policy": gate.policy_stamp("cupy"),
        "driver_slots_per_half": [list(half) for half in DRIVER_SLOTS_PER_HALF],
        "core_budget": CORE_BUDGET,
        "fusion_configuration": "enable_fp_fusion=ENABLE_FP_FUSION (fusion OFF)",
        "provenance": fold_gate.write_provenance(
            os.path.dirname(os.path.abspath(args.out))),
        "probe_patterns": record.get("patterns") if record else None,
    }
    save(results, args.out)

    reasons = gate.ftz_strip_license_reasons() + gate.probe_record_policy_reasons(record)
    # THE VERDICT, not the boolean: 'measured' against 'environment_default' is
    # the difference between this run having discriminated the arm and having
    # inherited it, and ``non_discriminating`` names the patterns whose MEASURED
    # blindness took them out of the agreement test. The BETA verdict is recorded
    # BESIDE it and does not gate anything here — K3b's licence is enforced where
    # it is bound, by ``folded_beta_bloch_pml_curl_coverage``'s own refusal
    # enumeration, and asserting it a second time up front would refuse a run the
    # predicate would have admitted.
    parity_verdict = folded_complex.parity_expansion_license(record)
    expansion = parity_verdict["expansion"]
    if expansion is None:
        reasons.append("the probe record licenses no EXPANSION for the parity "
                       "product; a guessed constexpr certifies nothing: "
                       + "; ".join(parity_verdict["refusals"]))
    results["license"] = {"licensed": not reasons, "reasons": reasons,
                          "parity": parity_verdict,
                          "beta_recorded_not_gated":
                              folded_complex.folded_beta_expansion_license(record)}
    save(results, args.out)
    log(f"[license] parity arm={parity_verdict['arm']} "
        f"basis={parity_verdict['basis']} discriminating="
        f"{sorted(parity_verdict['discriminating'])} non_discriminating="
        f"{parity_verdict['non_discriminating']}")
    if reasons:
        log("[license] REFUSED: " + "; ".join(reasons))
        return 1

    results["product_layer"] = run_product_layer(xp, expansion)
    save(results, args.out)

    cases: List[Dict[str, Any]] = []
    started_all = time.time()
    for index, spec in enumerate(CASES, start=1):
        case = run_case(xp, record, spec, results, args.out)
        cases.append(case)
        log(f"case {index}/{len(CASES)} [{case['name']}] "
            f"identical={case.get('identical')} "
            f"slots_ok={case.get('all_slots_launched_once_per_step')} "
            f"error={case.get('error')} "
            f"elapsed={round(time.time() - started_all, 1)} s "
            f"({case['seconds']} s)")
        results["cases"] = cases
        save(results, args.out)

    failures: List[str] = []
    for case in cases:
        if not case.get("identical"):
            failures.append(f"{case['name']}: composed state diverges "
                            f"({case.get('error') or case.get('failure')})")
        if case.get("all_slots_launched_once_per_step") is False:
            failures.append(f"{case['name']}: {case.get('failure')}")
        if case.get("source_non_vacuous") is False:
            failures.append(f"{case['name']}: the source was vacuous")
        if case.get("census_vacuous"):
            failures.append(f"{case['name']}: the signed-zero census was 0 — "
                            f"VACUOUS, not passed")
    product = results.get("product_layer", {})
    if product and product["identical"] != product["ran"]:
        failures.append("the parity PRODUCT layer diverges from the array path")

    # RE-STAMP: `policy_stamp` copies the strip's counters, so the stamp taken
    # when `results` was built froze them at 0 — before a single case had run.
    # Advertising the stripped policy beside a counter saying the strip never
    # fired is the same unevidenced claim this probe refuses elsewhere. Second
    # stamp per the sibling probe's pattern (probe_triton_nonlinear_composition.py
    # :818 then :845).
    results["subnormal_policy"] = gate.policy_stamp("cupy")
    results["summary"] = {
        "ran": len(cases),
        "identical": sum(int(bool(c.get("identical"))) for c in cases),
        "product_layer": f"{product.get('identical')}/{product.get('ran')}",
        "failures": failures,
        "passed": not failures,
    }
    results["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    save(results, args.out)
    log("[summary] " + json.dumps(results["summary"], indent=1, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
