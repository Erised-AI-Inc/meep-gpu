"""Bit-identity gate for the special_kz (grid.beta) Triton kernels.

The module under test is ``meep_gpu/triton_kernels/special_kz.py``, which the
Triton planner dispatches for a ``grid.beta`` run under its release entry (nothing here
changes that). This gate is the arbiter its docstring promises: every grouping
choice the kernels could not force by construction — the term's position between
curl and mask, the subtraction-as-negated-add, the unrotated center partner, the
HAS_BETA constexpr's identity arm — is HELD HERE, empirically, per platform.

CLONED from the complex tranche's certified harness (``gate_triton_complex.py``),
and deliberately IMPORTING it rather than restating: the subnormal
policy machinery (``install_ftz_strip`` / ``ftz_strip_license_reasons`` /
``policy_stamp``), the complex reference transcription (``complex_curl`` / ``complex_mask``
/ ``complex_constitutive_step``), the seeding (``_seed_host_complex``) and the
atomic writer all come from there, so this gate's verdict is about the beta
term and not about a re-implementation of the certified machinery.

SUBNORMAL POLICY — certification runs UNDER ``ieee_keep_ftz_stripped``. CuPy
unconditionally appends ``-ftz=true`` (cupy/cuda/compiler.py:552); the strip is
installed at the NVRTC seam BEFORE any CuPy compile, the cache dir must be
private and policy-suffixed, the strip counters are stamped into every
artifact, and a run whose strip cannot be confirmed exercised licenses nothing.
Never pass '-ftz=false' as user options (NVRTC duplicate-option hard error,
measured).

Legs, in order:

* ``reference``  — a SECOND transcription of the beta curl sub-steps (the
  laptop test file carries a third): the aligned curl (MEEP's left-to-right
  stencil by partner presence, the chunk forms keyed on the run's record) and
  then MEEP's beta pass after every component's curl (step_db.cpp:161-175)
  through its additive ladder (``fold_pass_reference``), and the record-keyed
  constitutive with MEEP's start of H, pinned against ``stepping.step_B``/
  ``step_D``/``update_H``/``update_E`` on real beta ``Grid``/``Fields``/``PML``
  objects over 4 full cycles per case: R1 (refl-angular-kz2d's beta, Courant
  0.5, PML x+y), R3 (metallic x, the grating's negative beta, Courant 0.34),
  R4 (odd 37x29x1 shape, Courant 0.4375), C1 (complex k = 0), C2 (the
  test_special_kz marquee: in-plane kx 0.9205 + beta -0.3907, Courant 0.35),
  C3 (Brillouin-edge x phase EXACTLY -1+0j + metallic y). Runs on NumPy (the
  laptop merge bar) and again on CuPy when a device is present.
* ``synthetic``  — the kernels against the in-gate reference on bare device
  arrays: real and complex families, metallic/periodic boundary products,
  both corpus beta signs, non-power-of-two Courants (0.34/0.4375 real, 0.35
  complex), guard on/off, seeds carrying signed-zero planes, each on two chunk
  records (``gate_triton_symmetry.RECORD_KINDS``: every cell in a layer chunk,
  and two runs per axis with coefficients exactly 1 outside them).
* ``identity``   — R5: the HAS_BETA=0 build byte-identical to the root
  kernels (``kernels.pml_curl_step`` / ``bloch_pml_curl_step``) on the same
  seeded state and record; and the engine-route plans bind HAS_BETA=1 only.
* ``refusals``   — R8 + C5 + R9, asserted WITH reasons, host-capable: real+
  offdiag+beta (predicate refusal AND stepping's own ValueError), beta on a
  3-D grid (Grid raises), beta+cylindrical (Grid raises), beta+BFAST,
  beta+chi2/chi3, no-PML+beta, conductivity cross-refusal in BOTH directions
  (this module refuses conductive targets; ``conductivity.py`` refuses beta),
  C5 complex+fold+beta recorded as a predicted null ('complex-by-fold
  inherited from the complex tranche, not a beta gap'), R9 folded-real
  recorded as Phase B deferred.
* ``mutations``  — armed and launch-counted, DISARMED/NEEDLE-MISSED are
  failures, each scored per family (an OR would let a family-specific miss
  hide behind the other family's catch) at its declared expectation: the beta
  pass's own mutations and the root forms' inversions and chunk-form battery
  (``fold_pass_mutations``, the same spellings every carrier of the pass is
  scored with); m2 the B and D sides' words exchanged (host, complex); m10 the
  real kernel launched on complex storage's word view. m11 (predicate
  mutations: drop the offdiag-real clause, drop the bfast clause, widen the
  boundary set) lives in ``meep_gpu/test_triton_special_kz.py`` where it runs
  at every merge, and is recorded here by reference. The earlier m8 (a
  per-element unrounded coefficient) and m9 (the complex product's folded
  zero cross terms) measured a complex PRODUCT the kernels no longer form:
  the beta pass is a quarter turn of the partner's parts by one host-rounded
  word, whose single rounding the host suite pins byte for byte
  (``test_the_complex_coefficient_matches_stepping_bytes``).
* ``engine``     — ``plan_beta_pml_curl`` / ``plan_beta_bloch_pml_curl`` /
  ``plan_beta_run_constitutive`` / ``plan_beta_run_complex_constitutive``
  from the engine's own objects against ``stepping`` on the reference grids,
  4 cycles, all 24/30 live arrays byte-compared per sub-step.

Discipline carried from prior defects: uint32 byte compare only (never
allclose); non-power-of-two Courant in every sweep; stated step budgets;
predicted nulls recorded WITH reasons; per-case flushed progress lines; atomic
JSON rewrite per case; own provenance record (``fingerprints.json`` is another
track's file and is only hashed). Correctness only: no throughput claims.

Usage (the GPU host, one clear device; cache dir private + policy-suffixed)::

    CUDA_VISIBLE_DEVICES=5 \\
    CUPY_CACHE_DIR=$RUN_ROOT/results/cupy_cache_ftz_stripped_$JOB \\
        python -u gate_triton_special_kz.py \\
        --out results/triton_special_kz_<date>/gate.json

Laptop (NumPy only; CUDA legs and the strip skip cleanly and say so)::

    python -u gate_triton_special_kz.py --legs reference,refusals \\
        --out /tmp/beta_gate_local.json
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import os
import sys
import tempfile
import textwrap
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_API = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_API not in sys.path:
    sys.path.insert(0, _REPO_API)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    import cupy as cp
except ImportError:  # Laptop leg: the reference transcription still validates.
    cp = None

try:
    import triton  # noqa: F401

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

# Shared machinery — never re-implemented here (see the module docstring).
import fold_pass_mutations as fold_passes  # noqa: E402
import fold_pass_reference as fold_ref  # noqa: E402
import gate_triton_complex as gate  # noqa: E402
import probe_fused_kernel_bit_identity as probe  # noqa: E402
from gate_chunk_records import every_cell_record, layer_record  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.triton_kernels import complex_fields  # noqa: E402
from meep_gpu.triton_kernels import special_kz  # noqa: E402

SEED = gate.SEED
log = gate.log
save = gate.save
bit_compare = probe.bit_compare
combine = probe.combine


def _symmetry_gate() -> Any:
    """``gate_triton_symmetry`` (the chunk-record reference, the records, the fitted
    coefficients), imported on first use: it imports CuPy at module level, and this
    module must import on a host without it (the laptop suite reads its records)."""
    import gate_triton_symmetry  # noqa: PLC0415

    return gate_triton_symmetry
_face = probe._face
PERIODIC = probe.PERIODIC
METALLIC = probe.METALLIC
CODE_OF = gate.CODE_OF

install_ftz_strip = gate.install_ftz_strip
ftz_strip_license_reasons = gate.ftz_strip_license_reasons
policy_stamp = gate.policy_stamp

FU_NAMES = gate.FU_NAMES
FW_NAMES = gate.FW_NAMES
FIELD_12 = gate.FIELD_12
ALL_STATE = gate.ALL_STATE

#: The corpus anchors' numbers — the sweeps run the values the unlock is about.
BETA_KZ2D = 0.3321611318837033
BETA_GRATING = -0.6850526103319672
BETA_SPECIAL_KZ = -0.39073112848927377
KX_SPECIAL_KZ = 0.9205048534524404

CURL_SUB_STEPS = ("step_B", "step_D")


def write_provenance(results_dir: str) -> str:
    """This gate's own provenance beside the shared one: the beta module and
    this file's hashes added to the tracked set (``fingerprints.json`` is
    another track's file and is only hashed, exactly as the complex gate
    records it)."""
    import meep_gpu  # noqa: PLC0415

    kernel_dir = os.path.join(os.path.dirname(os.path.abspath(meep_gpu.__file__)),
                              "triton_kernels")
    extra = {
        "beta_sha256": {
            "triton_kernels/special_kz.py": gate._digest(
                os.path.join(kernel_dir, "special_kz.py")),
            "parity/gate_triton_special_kz.py": gate._digest(
                os.path.abspath(__file__)),
        },
        "tranche": "special_kz (grid.beta) — Phase A",
    }
    composition = os.path.join(_HERE, "probe_triton_special_kz_composition.py")
    if os.path.exists(composition):
        extra["beta_sha256"]["parity/probe_triton_special_kz_composition.py"] = (
            gate._digest(composition))
    return gate.write_provenance(results_dir, extra=extra)


def real_beta_pml_step(xp: Any, arrays: Dict[str, Any],
                       coefficients: Dict[str, Any], dtdx: float,
                       sub_step: str, boundaries: Sequence[str],
                       beta: float, dt: float, record: Sequence[int],
                       words: Optional[Tuple[float, float]] = None) -> None:
    """One REAL beta curl sub-step, in place: the aligned real curl and its chunk
    forms (``gate_triton_symmetry.reference_pml_step`` keyed on ``record``), then
    MEEP's beta pass after every component's curl through its ladder
    (``fold_pass_reference.beta_pass``). ``words`` defaults to the host-rounded pair
    (``special_kz.beta_curl_coefficients``)."""
    _symmetry_gate().reference_pml_step(xp, arrays, coefficients, dtdx, sub_step,
                                        boundaries, (1, 1, 1), (None, None, None),
                                        record=record)
    magnetic = sub_step == "step_B"
    terms = probe.B_PML_TERMS if magnetic else probe.D_PML_TERMS
    if words is None:
        words = special_kz.beta_curl_coefficients(beta, dt, magnetic, False)
    fold_ref.beta_pass(xp, arrays, coefficients, terms, magnetic, words, record,
                       lambda increment, iyee: _symmetry_gate().reference_mask(
                           increment, iyee, boundaries))


def complex_beta_pml_step(xp: Any, arrays: Dict[str, Any],
                          coefficients: Dict[str, Any], dtdx: float,
                          sub_step: str, boundaries: Sequence[str],
                          phases: Sequence[Optional[complex]],
                          beta: float, dt: float, record: Sequence[int],
                          words: Optional[Tuple[float, float]] = None) -> None:
    """One COMPLEX beta curl sub-step, in place: the complex gate's aligned curl and
    chunk forms per part (``gate_triton_complex.complex_pml_step``), then MEEP's beta
    pass -- a quarter turn of the partner's parts by one host-rounded word -- after
    every component's curl (``fold_pass_reference.beta_pass``)."""
    gate.complex_pml_step(xp, arrays, coefficients, dtdx, sub_step, boundaries,
                          phases, record)
    magnetic = sub_step == "step_B"
    terms = probe.B_PML_TERMS if magnetic else probe.D_PML_TERMS
    if words is None:
        words = special_kz.beta_curl_coefficients(beta, dt, magnetic, True)
    fold_ref.beta_pass(xp, arrays, coefficients, terms, magnetic, words, record,
                       lambda increment, iyee: gate.complex_mask(
                           increment, iyee, boundaries))


# ---------------------------------------------------------------------------
# Reference grids — the corpus anchors' own numbers
# ---------------------------------------------------------------------------

REFERENCE_GRIDS: Tuple[Dict[str, Any], ...] = (
    # R1: refl-angular-kz2d's beta, 2-D PML x+y, Courant 0.5, source-free.
    {"name": "r1_kz2d_pml_xy", "cell": (2.0, 1.6, 0.0), "resolution": 10.0,
     "boundaries": "periodic", "k": (0.0, 0.0, 0.0), "beta": BETA_KZ2D,
     "courant": 0.5, "complex": False},
    # R3: metallic x / periodic y, the grating's negative beta, NP2 Courant.
    {"name": "r3_metallic_x_negative_beta", "cell": (2.0, 1.6, 0.0),
     "resolution": 10.0,
     "boundaries": ("metallic", "periodic", "periodic"),
     "k": (0.0, 0.0, 0.0), "beta": BETA_GRATING, "courant": 0.34,
     "complex": False},
    # R4: odd shape 37x29x1 with the 1-cell invariant axis, second NP2 Courant.
    {"name": "r4_odd_shape", "cell": (37.0, 29.0, 0.0), "resolution": 1.0,
     "boundaries": "periodic", "k": (0.0, 0.0, 0.0), "beta": 0.2,
     "courant": 0.4375, "complex": False},
    # C1: beta != 0, in-plane k = 0, complex storage, PML both axes.
    {"name": "c1_complex_k0", "cell": (2.0, 1.6, 0.0), "resolution": 10.0,
     "boundaries": "periodic", "k": (0.0, 0.0, 0.0), "beta": 0.2,
     "courant": 0.35, "complex": True},
    # C2 (marquee): the test_special_kz anchor — in-plane Bloch + beta.
    {"name": "c2_special_kz_marquee", "cell": (2.0, 1.6, 0.0),
     "resolution": 10.0, "boundaries": "periodic",
     "k": (KX_SPECIAL_KZ, 0.0, 0.0), "beta": BETA_SPECIAL_KZ,
     "courant": 0.35, "complex": True},
    # C3: Brillouin-edge phase EXACTLY -1+0j on x + metallic y.
    # resolution 10, Lx=2.0 -> nx_full=20; k=0.25 makes the edge branch exact.
    {"name": "c3_brillouin_edge_metallic_y", "cell": (2.0, 1.6, 0.0),
     "resolution": 10.0, "boundaries": ("periodic", "metallic", "periodic"),
     "k": (0.25, 0.0, 0.0), "beta": 0.2, "courant": 0.35, "complex": True},
)
REFERENCE_STEPS = 4


def _seed_host_real(shape: Tuple[int, int, int], rng,
                    scale: float = 1.0) -> np.ndarray:
    """Seeded float32 with a signed-zero plane: the beta term is a pure
    scalar-by-field product, so ±0 partners flow straight through it."""
    host = (scale * rng.uniform(-1.0, 1.0, size=shape)).astype(np.float32)
    live = [axis for axis in range(3) if shape[axis] > 1]
    if live:
        host[_face(live[0], 0)] = -0.0
        plane = host[_face(live[-1], -1)]
        plane[...] = np.float32(0.0)
    return host


def _build_reference_fields(xp, spec: Dict[str, Any]):
    grid = Grid(resolution=spec["resolution"], cell_size=spec["cell"],
                boundaries=spec["boundaries"], dimensions=2,
                courant=spec["courant"], k_point=tuple(spec["k"]),
                beta=spec["beta"], xp=xp)
    fields = Fields(grid=grid, force_complex_fields=bool(spec["complex"]))
    fields.enable_pml_storage()
    index = np.arange(int(np.prod(grid.shape)),
                      dtype=np.float32).reshape(grid.shape)
    epsilon = (1.45 + 0.30 * np.sin(index * np.float32(0.037))).astype(np.float32)
    inverse = (np.float32(1.0) / epsilon).astype(np.float32)
    fields.set_isotropic_epsilon_volume(xp.asarray(np.ascontiguousarray(epsilon)),
                                        xp.asarray(np.ascontiguousarray(inverse)))
    thickness = tuple((2, 2) if grid.shape[axis] >= 6 else (0, 0)
                      for axis in range(3))
    pml = PML(grid=grid, thickness=thickness)
    rng = np.random.default_rng(SEED + 29)
    for name in ALL_STATE:
        target = getattr(fields, name)
        if spec["complex"]:
            target[...] = xp.asarray(gate._seed_host_complex(tuple(grid.shape),
                                                             rng))
        else:
            target[...] = xp.asarray(_seed_host_real(tuple(grid.shape), rng))
    return grid, fields, pml


def run_reference_validation(results: Dict[str, Any], out_path: str, xp,
                             backend_name: str) -> Dict[str, Any]:
    """The transcription against stepping on real beta objects, 4 full cycles,
    byte-compared after every sub-step; sha256 chain for offline comparison."""
    cases: List[Dict[str, Any]] = []
    for spec in REFERENCE_GRIDS:
        started = time.time()
        grid, fields, pml = _build_reference_fields(xp, spec)
        kinds = stepping._boundary_kinds(grid, pml)
        phases = tuple(grid.bloch_phase(axis) for axis in range(3))
        dtdx = grid.dt / grid.dx
        record = layer_record(grid, pml)
        arrays = {name: getattr(fields, name).copy() for name in ALL_STATE}
        arrays.update({"inv_eps_" + name: fields.inverse_epsilon_for(name)
                       for name in ("Ex", "Ey", "Ez")})
        curl_tables = {sub: probe.layer_coefficients(pml, sub == "step_B")
                       for sub in CURL_SUB_STEPS}
        constitutive_tables = {
            side: probe.layer_constitutive_coefficients(pml, side == "E")
            for side in ("H", "E")}

        case: Dict[str, Any] = {
            "spec": {key: repr(value) for key, value in spec.items()},
            "backend": backend_name, "shape": list(grid.shape),
            "kinds": list(kinds), "phases": [repr(p) for p in phases],
            "beta": repr(grid.beta), "dtdx": repr(dtdx),
            "steps": REFERENCE_STEPS, "state_sha256_per_step": [],
            "first_divergence": None,
        }
        parts: Dict[str, Any] = {}
        for step in range(1, REFERENCE_STEPS + 1):
            for slot, engine_call, side in (
                    ("step_B", stepping.step_B, None),
                    ("update_H", stepping.update_H, "H"),
                    ("step_D", stepping.step_D, None),
                    ("update_E", stepping.update_E, "E")):
                engine_call(fields, pml)
                if slot in CURL_SUB_STEPS:
                    if spec["complex"]:
                        complex_beta_pml_step(xp, arrays, curl_tables[slot],
                                              dtdx, slot, kinds, phases,
                                              grid.beta, grid.dt, record)
                    else:
                        real_beta_pml_step(xp, arrays, curl_tables[slot], dtdx,
                                           slot, kinds, grid.beta, grid.dt,
                                           record)
                else:
                    # MEEP's start of H on the run's first update_H.
                    fold_ref.constitutive_step(
                        xp, arrays, constitutive_tables[side], side,
                        probe.CONSTITUTIVE_TERMS[side], record,
                        start_h=(side == "H" and step == 1))
                parts = {name: bit_compare(arrays[name], getattr(fields, name))
                         for name in ALL_STATE}
                if (not all(p["bit_identical"] for p in parts.values())
                        and case["first_divergence"] is None):
                    case["first_divergence"] = f"step {step} after {slot}"
            case["state_sha256_per_step"].append(
                gate._state_sha256(xp, {name: arrays[name] for name in ALL_STATE}))
        case["verdict"] = combine(parts)
        if case["first_divergence"] is not None:
            case["verdict"]["bit_identical"] = False
        case["seconds"] = round(time.time() - started, 3)
        cases.append(case)
        log(f"[reference:{backend_name}] {spec['name']} "
            f"shape={tuple(grid.shape)} kinds={kinds} beta={grid.beta!r} "
            f"identical={case['verdict']['bit_identical']} "
            f"(differing={case['verdict']['differing_floats']}/"
            f"{case['verdict']['total_floats']}) ({case['seconds']} s)")
        results.setdefault("reference", {})[backend_name] = {
            "ran": len(cases),
            "identical": sum(int(c["verdict"]["bit_identical"]) for c in cases),
            "steps_per_case": REFERENCE_STEPS,
            "cases": cases}
        save(results, out_path)
    return results["reference"][backend_name]


# ---------------------------------------------------------------------------
# The synthetic sweeps — kernels vs the in-gate reference, on device
# ---------------------------------------------------------------------------

REAL_SHAPES: Tuple[Tuple[int, int, int], ...] = ((7, 6, 1), (37, 29, 1),
                                                 (6, 5, 1))
COMPLEX_SHAPES: Tuple[Tuple[int, int, int], ...] = ((7, 6, 1), (6, 5, 1))

REAL_CONFIGS: Tuple[Dict[str, Any], ...] = (
    {"name": "periodic_xy", "boundaries": (PERIODIC, PERIODIC, PERIODIC)},
    {"name": "metallic_x", "boundaries": (METALLIC, PERIODIC, PERIODIC)},
    {"name": "metallic_xy", "boundaries": (METALLIC, METALLIC, PERIODIC)},
)
COMPLEX_CONFIGS: Tuple[Dict[str, Any], ...] = (
    {"name": "k0_unphased", "boundaries": (PERIODIC, PERIODIC, PERIODIC),
     "phases": (None, None, None)},
    {"name": "inplane_bloch_x",
     "boundaries": (PERIODIC, PERIODIC, PERIODIC),
     "phases": (gate._phase(KX_SPECIAL_KZ * 2.0), None, None)},
    {"name": "brillouin_edge_x_metallic_y",
     "boundaries": (PERIODIC, METALLIC, PERIODIC),
     "phases": (complex(-1.0, 0.0), None, None)},
)
REAL_BETAS = (BETA_KZ2D, BETA_GRATING)
REAL_DTDX = (0.34, 0.4375)
COMPLEX_BETAS = (BETA_SPECIAL_KZ, 0.2)
COMPLEX_DTDX = (0.35,)
BETA_DT = 0.05  # the physical dt the coefficient is built from in the sweeps


def _make_real_state(shape, names) -> Dict[str, Any]:
    rng = np.random.default_rng(SEED + 31)
    return {name: cp.asarray(_seed_host_real(tuple(shape), rng))
            for name in names}


def one_real_case(shape, config, beta, dtdx, sub_step, guard,
                  kernel=None, has_beta: int = 1,
                  record_kind: str = "split") -> Dict[str, Any]:
    case: Dict[str, Any] = {
        "family": "real", "shape": list(shape), "config": config["name"],
        "beta": repr(beta), "dtdx": repr(dtdx), "sub_step": sub_step,
        "guard": guard, "record_kind": record_kind,
    }
    names = FIELD_12 + FU_NAMES
    arrays = _make_real_state(shape, names)
    half_integer = sub_step == "step_B"
    record = _symmetry_gate().record_for(record_kind, shape)
    coefficients = probe.synthetic_coefficients(cp, tuple(shape), half_integer)
    if record_kind == "split":
        coefficients = _symmetry_gate().fit_coefficients(cp, coefficients, record,
                                                         half_integer, shape)
    flat = probe.flatten_coefficients(coefficients)
    reference = {name: array.copy() for name, array in arrays.items()}
    real_beta_pml_step(cp, reference, coefficients, dtdx, sub_step,
                       config["boundaries"], beta, BETA_DT, record)

    plus, minus = special_kz.beta_curl_coefficients(
        beta, BETA_DT, magnetic=half_integer, complex_storage=False)
    codes = [CODE_OF[b] for b in config["boundaries"]]
    plan = special_kz.plan_beta_pml_curl_from_arrays(
        sub_step, arrays, flat, codes, float(dtdx), plus, minus,
        kernel=kernel, has_beta=has_beta, chunk_bounds=record)
    plan.run(guard=guard)
    cp.cuda.runtime.deviceSynchronize()

    targets = ("Bx", "By", "Bz") if sub_step == "step_B" else ("Dx", "Dy", "Dz")
    checked = targets + tuple("fu_" + t for t in targets)
    case["verdict"] = combine({name: bit_compare(arrays[name], reference[name])
                               for name in checked})
    return case


def one_complex_case(shape, config, beta, dtdx, sub_step, guard,
                     kernel=None, magnetic_override=None,
                     has_beta: int = 1,
                     record_kind: str = "split") -> Dict[str, Any]:
    case: Dict[str, Any] = {
        "family": "complex", "shape": list(shape), "config": config["name"],
        "beta": repr(beta), "dtdx": repr(dtdx), "sub_step": sub_step,
        "guard": guard, "record_kind": record_kind,
        "phases": [repr(p) for p in config["phases"]],
    }
    names = FIELD_12 + FU_NAMES
    rng = np.random.default_rng(SEED + 37)
    arrays = {name: cp.asarray(gate._seed_host_complex(tuple(shape), rng))
              for name in names}
    half_integer = sub_step == "step_B"
    record = _symmetry_gate().record_for(record_kind, shape)
    coefficients = probe.synthetic_coefficients(cp, tuple(shape), half_integer)
    if record_kind == "split":
        coefficients = _symmetry_gate().fit_coefficients(cp, coefficients, record,
                                                         half_integer, shape)
    flat = probe.flatten_coefficients(coefficients)
    reference = {name: array.copy() for name, array in arrays.items()}
    complex_beta_pml_step(cp, reference, coefficients, dtdx, sub_step,
                          config["boundaries"], config["phases"], beta, BETA_DT,
                          record)

    magnetic = half_integer if magnetic_override is None else magnetic_override
    words = special_kz.beta_curl_coefficients(beta, BETA_DT, magnetic=magnetic,
                                              complex_storage=True)
    codes = [CODE_OF[b] for b in config["boundaries"]]
    plan = special_kz.plan_beta_bloch_pml_curl_from_arrays(
        sub_step, arrays, flat, codes, config["phases"], float(dtdx), words, kernel=kernel, has_beta=has_beta,
        chunk_bounds=record)
    plan.run(guard=guard)
    cp.cuda.runtime.deviceSynchronize()

    targets = ("Bx", "By", "Bz") if sub_step == "step_B" else ("Dx", "Dy", "Dz")
    checked = targets + tuple("fu_" + t for t in targets)
    case["verdict"] = combine({name: bit_compare(arrays[name], reference[name])
                               for name in checked})
    return case


def _summarize(cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    guarded = [c for c in cases if c.get("guard") is False]
    return {
        "ran": len(cases),
        "identical": sum(int(c["verdict"]["bit_identical"]) for c in cases),
        "guarded_ran": len(guarded),
        "guarded_identical": sum(int(c["verdict"]["bit_identical"])
                                 for c in guarded),
        "guarded_pass": bool(guarded) and all(c["verdict"]["bit_identical"]
                                              for c in guarded),
    }


def run_synthetic(results: Dict[str, Any], out_path: str,
                  label: str = "gate", guards=(False, True),
                  real_kernel=None, complex_kernel=None,
                  families=("real", "complex"),
                  complex_magnetic_override=None,
                  real_configs=REAL_CONFIGS, complex_configs=COMPLEX_CONFIGS,
                  record_kinds=None,
                  ) -> Dict[str, Any]:
    cases: List[Dict[str, Any]] = []
    if record_kinds is None:
        record_kinds = _symmetry_gate().RECORD_KINDS
    combos: List[Tuple] = []
    if "real" in families:
        combos += [("real", shape, config, beta, dtdx, sub_step, kind)
                   for shape in REAL_SHAPES for config in real_configs
                   for beta in REAL_BETAS for dtdx in REAL_DTDX
                   for sub_step in CURL_SUB_STEPS for kind in record_kinds]
    if "complex" in families:
        combos += [("complex", shape, config, beta, dtdx, sub_step, kind)
                   for shape in COMPLEX_SHAPES for config in complex_configs
                   for beta in COMPLEX_BETAS for dtdx in COMPLEX_DTDX
                   for sub_step in CURL_SUB_STEPS for kind in record_kinds]
    total = len(guards) * len(combos)
    index = 0
    for guard in guards:
        for family, shape, config, beta, dtdx, sub_step, kind in combos:
            index += 1
            started = time.time()
            if family == "real":
                case = one_real_case(shape, config, beta, dtdx, sub_step,
                                     guard, kernel=real_kernel, record_kind=kind)
            else:
                case = one_complex_case(shape, config, beta, dtdx, sub_step,
                                        guard,
                                        kernel=complex_kernel,
                                        magnetic_override=(
                                            (sub_step != "step_B")
                                            if complex_magnetic_override
                                            else None),
                                        record_kind=kind)
            case["seconds"] = round(time.time() - started, 3)
            cases.append(case)
            verdict = case["verdict"]
            log(f"[{label}] {index}/{total} guard={guard} {family} {sub_step} "
                f"{'x'.join(str(n) for n in shape)} {config['name']} {kind} "
                f"beta={beta!r}: identical={verdict['bit_identical']} "
                f"(differing={verdict['differing_floats']}/"
                f"{verdict['total_floats']}) ({case['seconds']} s)")
            summary = _summarize(cases)
            summary["cases"] = cases
            results.setdefault("synthetic", {})[label] = summary
            save(results, out_path)
    return results["synthetic"][label]


# ---------------------------------------------------------------------------
# R5 — the HAS_BETA=0 identity leg
# ---------------------------------------------------------------------------

def run_identity(results: Dict[str, Any], out_path: str) -> Dict[str, Any]:
    """HAS_BETA=0 must reproduce the certified kernels byte-for-byte on the
    same seeded state — the beta term is compiled OUT, not merely zeroed."""
    from meep_gpu.triton_kernels import launch  # noqa: PLC0415

    leg: Dict[str, Any] = {"cases": []}
    shape = (7, 6, 1)
    # Real family: beta kernel HAS_BETA=0 vs kernels.pml_curl_step, on synthetic
    # coefficient tables, so the certified plan binds the every-cell chunk record.
    from gate_chunk_records import every_cell_record  # noqa: PLC0415

    for sub_step in CURL_SUB_STEPS:
        names = FIELD_12 + FU_NAMES
        arrays = _make_real_state(shape, names)
        twin = {name: array.copy() for name, array in arrays.items()}
        coefficients = probe.synthetic_coefficients(cp, shape,
                                                    sub_step == "step_B")
        flat = probe.flatten_coefficients(coefficients)
        codes = [0, 1, 0]
        special_kz.plan_beta_pml_curl_from_arrays(
            sub_step, arrays, flat, codes, 0.34, 0.1, -0.1,
            has_beta=0, chunk_bounds=every_cell_record(shape)).run(guard=False)
        launch.plan_from_arrays(sub_step, twin, flat, codes, 0.34,
                                chunk_bounds=every_cell_record(shape)
                                ).run(guard=False)
        cp.cuda.runtime.deviceSynchronize()
        targets = ("Bx", "By", "Bz") if sub_step == "step_B" else ("Dx", "Dy", "Dz")
        checked = targets + tuple("fu_" + t for t in targets)
        verdict = combine({name: bit_compare(arrays[name], twin[name])
                           for name in checked})
        leg["cases"].append({"family": "real", "sub_step": sub_step,
                             "verdict": verdict})
        log(f"[identity] real {sub_step}: HAS_BETA=0 vs certified "
            f"identical={verdict['bit_identical']}")
    # Complex family: HAS_BETA=0 vs complex_fields.bloch_pml_curl_step.
    for sub_step in CURL_SUB_STEPS:
        names = FIELD_12 + FU_NAMES
        rng = np.random.default_rng(SEED + 41)
        arrays = {name: cp.asarray(gate._seed_host_complex(shape, rng))
                  for name in names}
        twin = {name: array.copy() for name, array in arrays.items()}
        coefficients = probe.synthetic_coefficients(cp, shape,
                                                    sub_step == "step_B")
        flat = probe.flatten_coefficients(coefficients)
        codes = [0, 0, 0]
        phases = (gate._phase(0.21), None, None)
        # NONZERO beta words deliberately (matching the real leg's 0.1/-0.1):
        # HAS_BETA=0 must compile the pass OUT, and only nonzero words make an
        # always-on HAS_BETA defect byte-visible -- a zero increment subtracted
        # is byte-neutral on every word (x - (+0) == x), so zero words would
        # prove nothing.
        special_kz.plan_beta_bloch_pml_curl_from_arrays(
            sub_step, arrays, flat, codes, phases, 0.35,
            (0.1, -0.3), has_beta=0,
            chunk_bounds=every_cell_record(shape)).run(guard=False)
        complex_fields.plan_complex_pml_curl_from_arrays(
            sub_step, twin, flat, codes, phases, 0.35,
            chunk_bounds=every_cell_record(shape)).run(guard=False)
        cp.cuda.runtime.deviceSynchronize()
        targets = ("Bx", "By", "Bz") if sub_step == "step_B" else ("Dx", "Dy", "Dz")
        checked = targets + tuple("fu_" + t for t in targets)
        verdict = combine({name: bit_compare(arrays[name], twin[name])
                           for name in checked})
        leg["cases"].append({"family": "complex", "sub_step": sub_step,
                             "verdict": verdict})
        log(f"[identity] complex {sub_step}: HAS_BETA=0 vs certified "
            f"identical={verdict['bit_identical']}")
    leg["identical"] = sum(int(c["verdict"]["bit_identical"])
                           for c in leg["cases"])
    leg["ran"] = len(leg["cases"])
    leg["engine_plans_bind_has_beta_1"] = (
        "plan_beta_pml_curl/plan_beta_bloch_pml_curl construct only behind a "
        "predicate requiring beta != 0 and always pass has_beta=1; the 0 arm "
        "exists for this leg alone")
    results["identity"] = leg
    save(results, out_path)
    return leg


# ---------------------------------------------------------------------------
# R8 / C5 / R9 — the refusal enumeration, asserted with reasons (host-capable)
# ---------------------------------------------------------------------------

def _mirror_pml(grid):
    return PML(grid=grid, thickness={"x": {"high": 2}, "y": 2})


def run_refusals(results: Dict[str, Any], out_path: str) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    failures: List[str] = []

    def record(name: str, ok: bool, detail: str) -> None:
        rows.append({"row": name, "as_expected": bool(ok), "detail": detail})
        if not ok:
            failures.append(f"refusal row {name}: {detail}")
        log(f"[refusals] {name}: as_expected={ok}")

    def build(beta=BETA_KZ2D, complex_storage=False, courant=0.35,
              skip_pml=False, **kwargs):
        grid = Grid(resolution=10.0, cell_size=(2.0, 1.6, 0.0), dimensions=2,
                    courant=courant, beta=beta, **kwargs)
        fields = Fields(grid=grid, force_complex_fields=complex_storage)
        fields.enable_pml_storage()
        if skip_pml:  # a mirrored axis refuses a low-face layer
            return fields, None
        thickness = tuple((2, 2) if grid.shape[axis] >= 6 else (0, 0)
                          for axis in range(3))
        return fields, PML(grid=grid, thickness=thickness)

    # (f) real + offdiag + beta: predicate refusal AND stepping's own raise.
    fields, pml = build()
    fields._chi1inv_offdiagonal = {"Ex": {"Ey": np.full(
        fields.grid.shape, 0.1, dtype=np.float32)}}
    reasons = special_kz.beta_pml_curl_coverage(fields, pml, "step_B").reasons
    hit = any("implicit-i" in r for r in reasons)
    try:
        shim_grid = type("G", (), {"beta": 0.25, "dt": 0.05, "xp": np})()
        shim = type("F", (), {"grid": shim_grid,
                              "has_offdiagonal_epsilon": True})()
        stepping._special_kz_beta_term(shim, np.ones((2, 2, 1), np.float32),
                                       +1.0, magnetic=True)
        raised = False
    except ValueError:
        raised = True
    record("real_offdiag_beta", hit and raised,
           f"predicate refused={hit} (stepping.py:800-810), "
           f"stepping raised={raised} (MEEP fields.cpp:548-549)")

    # (d)/(e) beta off 2-D Cartesian: the GRID refuses at construction.
    try:
        Grid(resolution=10.0, cell_size=(0.8, 0.8, 0.8), dimensions=3, beta=0.2)
        record("beta_on_3d_grid", False, "Grid ACCEPTED beta on 3-D")
    except ValueError as exc:
        record("beta_on_3d_grid", True, f"Grid raised: {exc}")
    try:
        Grid(resolution=10.0, cell_size=(0.8, 0.0, 0.8), dimensions=2,
             cylindrical=True, beta=0.2)
        record("beta_cylindrical", False, "Grid ACCEPTED beta on cylindrical")
    except ValueError as exc:
        record("beta_cylindrical", True, f"Grid raised: {exc}")

    # (a) beta + BFAST.
    fields, pml = build()
    fields.grid.bfast_scaled_k = (0.0, 0.0, 0.5)
    if fields.grid.bfast_active:
        reasons = special_kz.beta_pml_curl_coverage(fields, pml, "step_B").reasons
        record("beta_bfast", any("BFAST" in r for r in reasons), str(reasons))
    else:
        record("beta_bfast", True,
               "grid refused the planted bfast_scaled_k at resolution; the "
               "predicate clause is exercised by the laptop tests")

    # (b) beta + chi2/chi3 — planted through the class property, restored after.
    fields, pml = build()
    original = Fields.has_nonlinearity
    try:
        Fields.has_nonlinearity = property(lambda self: True)
        reasons = special_kz.beta_pml_curl_coverage(fields, pml, "step_B").reasons
    finally:
        Fields.has_nonlinearity = original
    record("beta_chi2_chi3", any("chi2/chi3" in r for r in reasons), str(reasons))

    # (h) no-PML + beta.
    fields, _pml = build()
    reasons = special_kz.beta_pml_curl_coverage(fields, None, "step_B").reasons
    record("no_pml_beta", any("PML" in r for r in reasons), str(reasons))

    # Conductivity cross-refusal, both directions.
    from meep_gpu.triton_kernels import conductivity  # noqa: PLC0415

    fields, pml = build()
    fields.set_d_conductivity(np.full(fields.grid.shape, 0.5, dtype=np.float32))
    mine = special_kz.beta_pml_curl_coverage(fields, pml, "step_D").reasons
    theirs = conductivity.conductive_pml_curl_coverage(fields, pml,
                                                       "step_D").reasons
    record("conductivity_cross_refusal",
           any("conductivity" in r for r in mine)
           and any("beta" in r for r in theirs),
           f"beta side: {[r for r in mine if 'conductivity' in r]}; "
           f"conductive side: {[r for r in theirs if 'beta' in r]}")

    # C5: complex + fold + beta — predicted null, the complex tranche's gap.
    fields, _unused = build(symmetry=("X",), complex_storage=True, beta=0.2,
                            skip_pml=True)
    pml = _mirror_pml(fields.grid)
    reasons = special_kz.beta_bloch_pml_curl_coverage(
        fields, pml, "step_B").reasons
    record("c5_complex_fold_beta_predicted_null",
           any("fold" in r.lower() or "mirror" in r for r in reasons),
           "predicted null with reason: complex-by-fold inherited from the "
           "complex tranche's composition gap, not a beta gap "
           f"(binary_grating/eigsrc_0 anchors); reasons={list(reasons)[:3]}")

    # R9: the folded REAL family is Phase B — recorded as deferred, by name.
    fields, _unused = build(symmetry=("X",), beta=0.2, skip_pml=True)
    pml = _mirror_pml(fields.grid)
    reasons = special_kz.beta_pml_curl_coverage(fields, pml, "step_B").reasons
    record("r9_folded_real_phase_b_deferred",
           any("Phase B" in r for r in reasons),
           "Phase B (folded real beta curl restating symmetry.py's clauses; "
           "unlocks test_eigsrc_kz_1_real_imag) is DEFERRED and refused by "
           f"name in Phase A; reasons={[r for r in reasons if 'Phase B' in r]}")

    leg = {"rows": rows, "failures": failures,
           "m11_note": ("predicate mutations (drop offdiag-real clause, drop "
                        "bfast clause, widen boundary set) run at every merge "
                        "in meep_gpu/test_triton_special_kz.py")}
    results["refusals"] = leg
    save(results, out_path)
    return leg


# ---------------------------------------------------------------------------
# Mutations — source-level (compiled, launch-counted) and host-level
# ---------------------------------------------------------------------------

_TEMPORARY: List[str] = []


def compile_mutated(source: str, kernel_name: str):
    """Compile a mutated copy of the shipped kernels from a real file on disk
    (Triton reads source via inspect). The constexpr codes are re-declared in
    the header because a @triton.jit body may not read a plain module global; the
    connection and the shared forms are IMPORTED, as the shipped module imports them.
    The module name does not start with ``triton``: Triton's dependency finder leaves
    every callee whose ``__module__`` starts with its package name out of the kernel's
    cache key."""
    header = (
        "import triton\nimport triton.language as tl\n"
        "from meep_gpu.triton_kernels.complex_fields import _connect\n"
        + fold_passes.MUTANT_IMPORTS
        + "PERIODIC = tl.constexpr(0)\nMETALLIC = tl.constexpr(1)\n\n")
    handle = tempfile.NamedTemporaryFile("w", suffix="_mutated_beta.py",
                                         delete=False, encoding="utf-8")
    handle.write(header + source)
    handle.close()
    _TEMPORARY.append(handle.name)
    spec = importlib.util.spec_from_file_location(
        "special_kz_mutant_" + str(len(_TEMPORARY)), handle.name)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = module  # type: ignore[union-attr]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return getattr(module, kernel_name)


def shipped_source() -> str:
    """Both kernels, as one mutable text. Every helper they call is a shared form
    (``forms.py``) or the connection (``complex_fields._connect``), imported by the
    mutant rather than mutated here."""
    functions = (special_kz.beta_pml_curl_step,
                 special_kz.beta_bloch_pml_curl_step)
    return "\n\n".join(textwrap.dedent(inspect.getsource(f.fn))
                       for f in functions)


#: Target 2's last stencil line in each kernel: the previous placement of the beta
#: term (inside the curl, before the forms) is planted after it.
REAL_CURL_ANCHOR = "curl2 = curl_stencil(b_x, b, a, a_y, dtdx, px, py)"
COMPLEX_CURL_ANCHOR = "curl2_im = curl_stencil(b_x_im, b2_im, a2_im, a_y_im, dtdx, px, py)"

#: name -> (rewrite, families, expectation, why). Each family's kernel is scored on its
#: own; an entry is scored only in the families whose kernel text it rewrites. The
#: beta pass's mutations and the root forms' are ``fold_pass_mutations``' shared
#: spellings (every carrier of the pass is scored with the same text).
SOURCE_MUTATIONS: Dict[str, Tuple[Any, Tuple[str, ...], str, str]] = {}
for _family, _entries in (
        ("real", fold_passes.beta_mutations(False, False, REAL_CURL_ANCHOR)
         + fold_passes.form_mutations(False)),
        ("complex", fold_passes.beta_mutations(True, False, COMPLEX_CURL_ANCHOR)
         + fold_passes.form_mutations(False))):
    for _name, _why, _expectation, _rewrite, _needs in _entries:
        _key = f"{_family}:{_name}"
        SOURCE_MUTATIONS[_key] = (_rewrite, (_family,), _expectation, _why)


class CountingKernel(gate.CountingKernel):
    pass


def run_mutations(results: Dict[str, Any], out_path: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    shipped = shipped_source()
    kernels = {"real": "beta_pml_curl_step", "complex": "beta_bloch_pml_curl_step"}

    for name, (transform, families, expectation, why) in SOURCE_MUTATIONS.items():
        mutated, hits = transform(shipped)
        if hits == 0 or mutated == shipped:
            out[name] = {"error": "NEEDLE MISSED: the mutation matched "
                                  "nothing; the needle drifted off the kernel",
                         "expectation": expectation}
            log(f"[mut] {name}: NEEDLE MISSED — nothing exercised")
            results["mutations"] = out
            save(results, out_path)
            continue
        entry: Dict[str, Any] = {"sites": hits, "families": list(families),
                                 "expectation": expectation, "why": why}
        caught_per_family: Dict[str, bool] = {}
        launches = 0
        for family in families:
            counter = CountingKernel(compile_mutated(mutated, kernels[family]))
            summary = run_synthetic(
                results, out_path, label="mutation:" + name,
                guards=(False,), families=(family,),
                **({"real_kernel": counter} if family == "real"
                   else {"complex_kernel": counter}))
            entry[family] = {"ran": summary["ran"], "identical": summary["identical"]}
            caught_per_family[family] = summary["identical"] < summary["ran"]
            launches += counter.launches
        entry["launches"] = launches
        entry["caught_per_family"] = caught_per_family
        entry["caught"] = bool(caught_per_family) and all(caught_per_family.values())
        entry["as_expected"] = entry["caught"] == (expectation == "caught")
        if launches == 0:
            entry["error"] = ("DISARMED: the mutated kernel never launched — "
                              "the leg measured nothing and FAILS")
        log(f"[mut] {name}: sites={hits} launches={launches} "
            f"caught_per_family={caught_per_family} expectation={expectation} "
            f"as_expected={entry['as_expected']}")
        out[name] = entry
        results["mutations"] = out
        save(results, out_path)

    # m2 (host): the B and D sides' words exchanged -- the plan built with the WRONG
    # magnetic flag, so every complex case takes the other side's quarter turn.
    summary = run_synthetic(results, out_path,
                            label="mutation:m2_pm_i_swapped", guards=(False,),
                            families=("complex",),
                            complex_magnetic_override=True)
    caught = summary["identical"] < summary["ran"]
    out["m2_pm_i_swapped"] = {
        "host": True, "ran": summary["ran"], "identical": summary["identical"],
        "expectation": "caught", "caught": caught, "as_expected": caught}
    log(f"[mut] m2_pm_i_swapped: identical={summary['identical']}/"
        f"{summary['ran']}")
    results["mutations"] = out
    save(results, out_path)


    # m10 (host): the REAL kernel launched on complex storage's word view.
    shape = (7, 6, 1)
    rng = np.random.default_rng(SEED + 43)
    names = FIELD_12 + FU_NAMES
    arrays = {name: cp.asarray(gate._seed_host_complex(shape, rng))
              for name in names}
    reference = {name: array.copy() for name, array in arrays.items()}
    coefficients = probe.synthetic_coefficients(cp, shape, True)
    record = every_cell_record(shape)
    complex_beta_pml_step(cp, reference, coefficients, 0.35, "step_B",
                          (PERIODIC, PERIODIC, PERIODIC), (None, None, None),
                          0.2, BETA_DT, record)
    word_views = {name: arrays[name].view(cp.float32) for name in names}
    word_coefficients = dict(coefficients)
    word_coefficients["kms_z"] = cp.ascontiguousarray(
        cp.repeat(coefficients["kms_z"].reshape(-1), 2)).reshape(1, 1, 2)
    word_coefficients["sinv_z"] = cp.ascontiguousarray(
        cp.repeat(coefficients["sinv_z"].reshape(-1), 2)).reshape(1, 1, 2)
    flat = probe.flatten_coefficients(word_coefficients)
    plus, minus = special_kz.beta_curl_coefficients(0.2, BETA_DT, True, False)
    plan = special_kz.plan_beta_pml_curl_from_arrays(
        "step_B", word_views, flat, (0, 0, 0), 0.35, plus, minus,
        chunk_bounds=every_cell_record((7, 6, 2)))
    plan.run(guard=False)
    cp.cuda.runtime.deviceSynchronize()
    checked = ("Bx", "By", "Bz", "fu_Bx", "fu_By", "fu_Bz")
    verdict = combine({name: bit_compare(arrays[name], reference[name])
                       for name in checked})
    out["m10_real_kernel_on_complex_storage"] = {
        "host": True, "ran": 1, "identical": int(verdict["bit_identical"]),
        "expectation": "caught", "caught": not verdict["bit_identical"],
        "as_expected": not verdict["bit_identical"]}
    log(f"[mut] m10_real_kernel_on_complex_storage: "
        f"caught={not verdict['bit_identical']}")

    out["m11_predicate_mutations"] = {
        "delegated": "meep_gpu/test_triton_special_kz.py "
                     "(test_m11_dropping_the_offdiag_real_clause_is_caught, "
                     "test_m11_dropping_the_bfast_clause_is_caught, "
                     "test_m11_widening_the_boundary_set_is_caught) — runs at "
                     "every merge on the laptop, no device required"}
    results["mutations"] = out
    save(results, out_path)
    return out


# ---------------------------------------------------------------------------
# The engine leg — engine-route plans against stepping, on CuPy
# ---------------------------------------------------------------------------

ENGINE_STEPS = 4


def run_engine(results: Dict[str, Any], out_path: str) -> Dict[str, Any]:
    cases: List[Dict[str, Any]] = []
    for spec in REFERENCE_GRIDS:
        started = time.time()
        grid, fields, pml = _build_reference_fields(cp, spec)
        reference = Fields(grid=grid,
                           force_complex_fields=bool(spec["complex"]))
        reference.enable_pml_storage()
        reference.set_isotropic_epsilon_volume(fields.epsilon_for("Ex"),
                                               fields.inverse_epsilon_for("Ex"))
        for name in ALL_STATE:
            getattr(reference, name)[...] = getattr(fields, name)

        case: Dict[str, Any] = {"spec": {k: repr(v) for k, v in spec.items()},
                                "shape": list(grid.shape),
                                "steps": ENGINE_STEPS,
                                "first_divergence": None}
        if spec["complex"]:
            plans = {
                "step_B": special_kz.plan_beta_bloch_pml_curl(
                    fields, pml, "step_B"),
                "update_H": special_kz.plan_beta_run_complex_constitutive(
                    fields, pml, "H"),
                "step_D": special_kz.plan_beta_bloch_pml_curl(
                    fields, pml, "step_D"),
                "update_E": special_kz.plan_beta_run_complex_constitutive(
                    fields, pml, "E"),
            }
        else:
            plans = {
                "step_B": special_kz.plan_beta_pml_curl(fields, pml, "step_B"),
                "update_H": special_kz.plan_beta_run_constitutive(
                    fields, pml, "H"),
                "step_D": special_kz.plan_beta_pml_curl(fields, pml, "step_D"),
                "update_E": special_kz.plan_beta_run_constitutive(
                    fields, pml, "E"),
            }
        missing = sorted(slot for slot, plan in plans.items() if plan is None)
        if missing:
            case["error"] = (f"the predicate refused a configuration the gate "
                             f"built: {missing}")
            cases.append(case)
            log(f"[engine] {spec['name']} REFUSED: {missing}")
            results["engine"] = {"ran": len(cases), "identical": 0,
                                 "cases": cases}
            save(results, out_path)
            continue
        parts: Dict[str, Any] = {}
        for step in range(1, ENGINE_STEPS + 1):
            for slot, engine_call in (("step_B", stepping.step_B),
                                      ("update_H", stepping.update_H),
                                      ("step_D", stepping.step_D),
                                      ("update_E", stepping.update_E)):
                plans[slot].run()
                engine_call(reference, pml)
                cp.cuda.runtime.deviceSynchronize()
                for name in ALL_STATE:
                    parts[name] = bit_compare(getattr(fields, name),
                                              getattr(reference, name))
                # Latch the FIRST diverging sub-step (the reference leg's
                # convention): a transient that healed byte-exactly before
                # the final snapshot must still fail, not pass silently.
                if (case["first_divergence"] is None
                        and not all(p["bit_identical"]
                                    for p in parts.values())):
                    case["first_divergence"] = f"step {step} after {slot}"
        case["verdict"] = combine(parts)
        if case["first_divergence"] is not None:
            case["verdict"]["bit_identical"] = False
        case["seconds"] = round(time.time() - started, 3)
        cases.append(case)
        log(f"[engine] {spec['name']} shape={tuple(grid.shape)} "
            f"identical={case['verdict']['bit_identical']} "
            f"(differing={case['verdict']['differing_floats']}/"
            f"{case['verdict']['total_floats']}) ({case['seconds']} s)")
        results["engine"] = {
            "ran": len(cases),
            "identical": sum(int(c.get("verdict", {}).get("bit_identical",
                                                          False))
                             for c in cases),
            "cases": cases}
        save(results, out_path)
    return results["engine"]


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------

DEFAULT_LEGS = "reference,refusals,synthetic,identity,mutations,engine"
DEVICE_LEGS = ("synthetic", "identity", "mutations", "engine")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--legs", default=DEFAULT_LEGS)
    args = parser.parse_args(argv)
    legs = tuple(name.strip() for name in args.legs.split(",") if name.strip())
    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."

    # The ship policy, installed BEFORE any CuPy compile (a clean no-op
    # without CuPy; raises at startup on a cache dir that could mix policies).
    install_ftz_strip()

    device_available = cp is not None and _TRITON_AVAILABLE
    results: Dict[str, Any] = {
        "seed": SEED,
        "tranche": "special_kz (grid.beta) — Phase A: plain real + plain complex",
        "granularity": "sub-step (the composition probe holds the whole-step "
                       "claim)",
        "correctness_only": "no throughput or timing claims in this artifact",
        "step_budgets": {"reference": REFERENCE_STEPS, "engine": ENGINE_STEPS,
                         "note": "bit-identity is claimed for exactly these "
                                 "stated budgets and no further"},
        "betas": {"real": [repr(v) for v in REAL_BETAS],
                  "complex": [repr(v) for v in COMPLEX_BETAS]},
        "host": {"cupy": None if cp is None else cp.__version__,
                 "triton": triton.__version__ if _TRITON_AVAILABLE else None,
                 "numpy": np.__version__,
                 "python": sys.version.split()[0]},
        "subnormal_policy": policy_stamp("cupy"),  # refreshed at the end
        "skipped_legs": {},
    }
    os.makedirs(out_dir, exist_ok=True)
    write_provenance(out_dir)
    save(results, args.out)

    failures: List[str] = []

    if "reference" in legs:
        summary = run_reference_validation(results, args.out, np, "numpy")
        if summary["identical"] != summary["ran"]:
            failures.append(f"reference transcription differs from stepping.py "
                            f"on numpy: {summary['identical']}/{summary['ran']}")
        if cp is not None:
            summary = run_reference_validation(results, args.out, cp, "cupy")
            if summary["identical"] != summary["ran"]:
                failures.append(f"reference transcription differs from "
                                f"stepping.py on cupy: "
                                f"{summary['identical']}/{summary['ran']}")
        else:
            log("[reference] cupy leg SKIPPED cleanly: no cupy on this host")

    if "refusals" in legs:
        leg = run_refusals(results, args.out)
        failures.extend(leg["failures"])

    for leg_name in DEVICE_LEGS:
        if leg_name not in legs:
            continue
        if not device_available:
            reason = ("no CUDA/triton on this host — this leg runs on the "
                      "measurement machine; nothing was measured here")
            results["skipped_legs"][leg_name] = reason
            log(f"[{leg_name}] SKIPPED cleanly: {reason}")
            save(results, args.out)
            continue
        if leg_name == "synthetic":
            summary = run_synthetic(results, args.out)
            if not summary["guarded_pass"]:
                failures.append(f"synthetic sweep: "
                                f"{summary['guarded_identical']}/"
                                f"{summary['guarded_ran']} guarded identical")
            log(f"[SUMMARY] synthetic guarded "
                f"{summary['guarded_identical']}/{summary['guarded_ran']}")
        elif leg_name == "identity":
            leg = run_identity(results, args.out)
            if leg["identical"] != leg["ran"]:
                failures.append(f"identity leg (R5): {leg['identical']}/"
                                f"{leg['ran']} HAS_BETA=0 cases identical")
        elif leg_name == "mutations":
            out = run_mutations(results, args.out)
            for name, entry in out.items():
                if name == "m11_predicate_mutations":
                    continue
                if entry.get("error"):
                    failures.append(f"mutation {name}: {entry['error']}")
                elif not entry.get("as_expected"):
                    failures.append(f"mutation {name}: caught={entry.get('caught')} "
                                    f"against expectation "
                                    f"{entry.get('expectation')}")
        elif leg_name == "engine":
            summary = run_engine(results, args.out)
            if summary["identical"] != summary["ran"]:
                failures.append(f"engine leg: {summary['identical']}/"
                                f"{summary['ran']} identical")

    if cp is not None:
        for reason in ftz_strip_license_reasons():
            message = "subnormal policy: " + reason
            if message not in failures and not any(
                    reason in existing for existing in failures):
                failures.append(message)
    results["subnormal_policy"] = policy_stamp("cupy")

    results["summary"] = {
        "status": "passed" if not failures else "FAILED",
        "failures": failures,
        "device_legs_ran": device_available,
        "certified_under_subnormal_policy":
            results["subnormal_policy"].get("policy"),
    }
    save(results, args.out)
    log(f"[done] {args.out} status={results['summary']['status']} "
        f"failures={failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
