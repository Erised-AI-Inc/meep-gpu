"""Bit-identity gate for the BFAST Triton kernel.

The module under test is ``meep_gpu/triton_kernels/bfast_curl.py`` — NOT wired
into production dispatch (``plan_fast_path`` keeps returning None; nothing
here changes that).

WHAT THIS GATE HOLDS, and what it CANNOT. Held here, empirically, per
platform — each one byte-visible in f32 and armed by a mutation or a leg: the
six host-rounded scalars and their cross-indexing (m1), the D-side negation
(m2), the absence of dtdx in the tail (m3), the SUM rather than the curl's
difference (m4), the ``total - 2.0*state`` advance (m5), the invariance guard
(m6), the fold SIGN (m7), the pre-store advance mask (m8), the in-place state
write (m9), the always-run-all-six-targets binding (the state is seeded
nonzero, so a zero-k target's ``-2*state`` advance moves bytes), and the
HAS_BFAST identity arm (the identity leg).

THE PASS FOLLOWS THE FORMS. MEEP's BFAST pass runs on each component right after
its curl (step_db.cpp:124-142): the state takes two roundings (``F = S - F_prev``
stored, ``raw = F - F_prev``), an absent partner is not read, MEEP's plain one-partner
line drops ``- F``, and ``raw`` goes through the pass's own additive ladder keyed on the
chunk record (``fold_pass_reference``). Its position after the forms is held by a
SOURCE assertion in ``meep_gpu/test_triton_bfast.py`` and by the transcription legs of
the fused products; the ladder and the state's roundings are held here.

CLONED from the certified tranches' harnesses (complex, special_kz, nonlinear,
offdiag) and deliberately IMPORTING the
shared machinery rather than restating: the subnormal policy
(``install_ftz_strip`` / ``ftz_strip_license_reasons`` / ``policy_stamp``),
the shared real-PML reference transcriptions
(``probe_fused_kernel_bit_identity``'s shift/curl/mask/recurrence and the
constitutive step), the seeding and the atomic writer all come from there, so
this gate's verdict is about the BFAST tail and not about a re-implementation
of the certified machinery. This family has NO complex product — real f32
only (Phase A).

SUBNORMAL POLICY — certification runs UNDER ``ieee_keep_ftz_stripped``. CuPy
unconditionally appends ``-ftz=true`` (cupy/cuda/compiler.py:552); the strip
is installed at the NVRTC seam BEFORE any CuPy compile, the cache dir must be
private and policy-suffixed, and the strip counters are stamped into every
artifact. This family's FTZ surface is unusually hot: ``f_bfast`` is the ONE
auxiliary where subnormals persist indefinitely (the marginally stable
``(-1)^n`` mode, stepping.py:868-874), which is why the subnormal leg seeds
the STATE with subnormals rather than only the fields.

Legs, in order:

* ``reference``  — a SECOND transcription of the BFAST curl sub-steps (the
  laptop test file carries a third), pinned against ``stepping.step_B``/
  ``step_D``/``update_H``/``update_E`` on real BFAST Grid/Fields/PML objects
  over 4 full cycles per case, states byte-compared too: R1 (the marquee's
  own class — dims=3, one-cell x/y, z-PML both faces, kx = 1.4 sin 35.7 deg,
  Courant (1-kx)/sqrt(3), natively non-power-of-two), R2 (full 3-D k, all
  components distinct — the k-indexing needle), R3 (DECLARED dims=2 — the
  invariance-guard needle), R4 (metallic x — the mask needle), R5 (metallic
  x AND y — TWO masked axes, m8's needle configuration and the one the D
  targets double-mask), R6 (DECLARED dims=1 — the D1 arm of the guard, where
  four flags are false at once). NumPy (the laptop merge bar) and again on
  CuPy when a device is present.
* ``refusals``   — asserted WITH reasons, host-capable: bfast+cylindrical
  (the GRID raises — upstream, grid.py:706-742), bfast+beta (cross-refusal
  in BOTH directions: this family names the fold order, special_kz refuses
  bfast), bfast+conductivity (both directions), bfast+complex and
  bfast+Bloch (the predicate refuses what Grid deliberately allows,
  grid.py:719-724), bfast+mirror fold, bfast+chi2/chi3, bfast without PML,
  non-cupy backend; and a nonzero k on a DECLARED-INVARIANT AXIS asserted
  ADMITTED (MEEP does not read the partner it multiplies, and neither do the
  array path and this kernel). Predicate-mutation tests live in
  ``meep_gpu/test_triton_bfast.py`` where they run at every merge — clause 11
  (bfast_active), the k_point pair, beta, conductivity and the boundary/mirror
  set.
* ``synthetic``  — the kernel against the in-gate reference on bare device
  arrays: randomized f32 operands AND randomized NONZERO f_bfast state (the
  only layer where the always-run-all-six binding and the ``-2*state``
  advance are byte-visible), both stride directions, every covered boundary
  kind (with and without masked rows), the marquee k and the full 3-D k,
  both non-power-of-two Courants, guard on/off; curl, fu AND state bytes
  compared, uint32 only.
* ``subnormal``  — the synthetic sweep with the state seeded WITH subnormals;
  bytes under the policy, in-run strip counters recorded.
* ``identity``   — the HAS_BFAST=0 build byte-identical to the certified
  ``kernels.pml_curl_step`` on the same seeded state, AND the states
  untouched (the tail compiled OUT, not merely zeroed), over four boundary
  triples covering the BCX, BCY and BCZ metallic arms, both shapes and both
  non-power-of-two dtdx. SELF-CHECKING: unlike every other leg this one
  compares two arrays rather than an array against a computed reference, so
  a no-launch would be trivially identical — both plans are launch-counted
  and the certified kernel must move bytes off its seeds.
* ``mutations``  — armed, launch-counted, DISARMED/NEEDLE-MISSED are
  failures, mutant-distinctness verified (renamed entry points + cache-key
  inequality — the measured stale-binary hazard), each at its declared
  expectation: the BFAST pass's mutations and the root forms' inversions and
  chunk-form battery are ``fold_pass_mutations``' shared spellings (the same text
  every carrier of the pass is scored with); m1 k-pair interchange (host; needle =
  the full-k configs, all components distinct); m2 D-side negation dropped (host;
  step_D combos); m6 invariance guard removed (host; needle = the DECLARED-2-D
  configs); m9 state written to a copy, not in place (host check: the engine plan
  must bind ``fields.f_bfast_*`` pointer-identically and the bytes must move
  THROUGH the held reference). Predicted nulls recorded WITH reasons, not armed.
* ``engine``     — ``plan_bfast_pml_curl`` / ``plan_bfast_run_constitutive``
  from the engine's own objects against ``stepping`` on the reference grids,
  4 cycles, all live arrays byte-compared per sub-step, states included.

Discipline carried from prior defects: uint32 byte compare only (never
allclose); non-power-of-two Courant in every sweep (the marquee's own
0.105679... is one of the two); stated step budgets; predicted nulls recorded
WITH reasons; per-case flushed progress lines; atomic JSON rewrite per case;
own provenance record. Correctness only: no throughput claims.

Usage (the GPU host, one clear device; cache dir private + policy-suffixed)::

    CUDA_VISIBLE_DEVICES=5 \\
    CUPY_CACHE_DIR=$RUN_ROOT/results/cupy_cache_ftz_stripped_$JOB \\
        python -u gate_triton_bfast.py \\
        --out results/triton_bfast_<date>/gate.json

Laptop (NumPy only; CUDA legs and the strip skip cleanly and say so)::

    python -u gate_triton_bfast.py --legs reference,refusals \\
        --out /tmp/bfast_gate_local.json
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import math
import os
import sys
import tempfile
import textwrap
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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
from meep_gpu.fields import BFAST_COMPONENTS, Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.triton_kernels import bfast_curl  # noqa: E402

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
FIELD_12 = gate.FIELD_12
ALL_STATE = gate.ALL_STATE

BFAST_ALL = tuple("f_bfast_" + name for name in BFAST_COMPONENTS)
CHECK_STATE = ALL_STATE + BFAST_ALL

#: The marquee's own numbers, computed the way MEEP's test computes them in
#: USER code (test_refl_angular.py:50-52) — the demand case this tranche is
#: for. Natively non-power-of-two.
KX_MARQUEE = 1.4 * math.sin(math.radians(35.7))
COURANT_MARQUEE = (1.0 - KX_MARQUEE) / math.sqrt(3.0)
FULL_K = (0.31, 0.17, 0.23)     # all distinct: the k-indexing needle
K_METALLIC = (0.4, 0.25, 0.0)

CURL_SUB_STEPS = ("step_B", "step_D")
SUB_TARGETS = {"step_B": ("Bx", "By", "Bz"), "step_D": ("Dx", "Dy", "Dz")}


def write_provenance(results_dir: str) -> str:
    """This gate's own provenance beside the shared one: the bfast module and
    this trio's hashes added to the tracked set (``fingerprints.json`` is
    another track's file and is only hashed)."""
    import meep_gpu  # noqa: PLC0415

    kernel_dir = os.path.join(os.path.dirname(os.path.abspath(meep_gpu.__file__)),
                              "triton_kernels")
    extra = {
        "bfast_sha256": {
            "triton_kernels/bfast_curl.py": gate._digest(
                os.path.join(kernel_dir, "bfast_curl.py")),
            "parity/gate_triton_bfast.py": gate._digest(
                os.path.abspath(__file__)),
        },
        "tranche": "BFAST (bfast_scaled_k) — Phase A: real-f32 split-field-PML curl",
    }
    composition = os.path.join(_HERE, "probe_triton_bfast_composition.py")
    if os.path.exists(composition):
        extra["bfast_sha256"]["parity/probe_triton_bfast_composition.py"] = (
            gate._digest(composition))
    return gate.write_provenance(results_dir, extra=extra)


# ---------------------------------------------------------------------------
# The in-gate reference — the BFAST tail added to the shared transcription
# ---------------------------------------------------------------------------
#
# SECOND transcription (the laptop test file's is the third). The k
# assignment, the guard, the D negation and the recurrence are computed INLINE
# from stepping.py:909-931's arithmetic rather than by calling
# bfast_curl.bfast_curl_coefficients, so the gate and the module cannot agree
# by sharing a defect in the host function.

def accumulator() -> Dict[str, Any]:
    """A running total over EVERY comparison a multi-step leg makes.

    ``combine`` describes ONE snapshot. Rebinding it inside the step loop and
    reading it after leaves the artifact describing only the LAST sub-step, so
    a divergence that self-heals reports ``0/N differing`` beside
    ``bit_identical: False``. These legs accumulate instead."""
    return {"bit_identical": True, "differing_floats": 0, "total_floats": 0,
            "comparisons": 0}


def accumulate(totals: Dict[str, Any],
               parts: Dict[str, Any]) -> Dict[str, Any]:
    for part in parts.values():
        totals["bit_identical"] = bool(totals["bit_identical"]
                                       and part["bit_identical"])
        totals["differing_floats"] += int(part["differing_floats"])
        totals["total_floats"] += int(part["total_floats"])
        totals["comparisons"] += 1
    return totals








def real_bfast_pml_step(xp: Any, arrays: Dict[str, Any],
                        coefficients: Dict[str, Any], dtdx: float,
                        sub_step: str, boundaries: Sequence[str],
                        bfast_k: Sequence[float], record: Sequence[int]) -> None:
    """One REAL BFAST curl sub-step, in place: the aligned real curl and its chunk
    forms (``gate_triton_symmetry.reference_pml_step`` keyed on ``record``), then
    MEEP's BFAST pass on each component -- the state's two roundings, the one-partner
    forms by the record's presence word, the plain branch on cells in neither the
    dsig nor the dsigu chunk -- through its ladder (``fold_pass_reference``). The
    pass reads the curl's sources only, so following the whole curl sub-step is the
    same word per cell as following each component's curl."""
    _symmetry_gate().reference_pml_step(xp, arrays, coefficients, dtdx, sub_step,
                                        boundaries, (1, 1, 1), (None, None, None),
                                        record=record)
    terms = probe.B_PML_TERMS if sub_step == "step_B" else probe.D_PML_TERMS
    backward = sub_step == "step_D"
    half_integer = not backward
    present = fold_ref.presence(record)

    def mask(values: Any, iyee: Sequence[int]) -> None:
        _symmetry_gate().reference_mask(values, iyee, boundaries)

    for target, g1, a1, g2, a2, dsig, dsigu, iyee in terms:
        shape = tuple(int(n) for n in arrays[target].shape)
        have = (present[a1], present[a2])
        k1, k2 = fold_ref.bfast_words(bfast_k, a1, a2, g1, g2, have, not backward)
        increment = fold_ref.bfast_increment(
            xp, arrays["f_bfast_" + target],
            probe.reference_shift(xp, arrays[g1], a1, boundaries[a1], backward),
            arrays[g1],
            probe.reference_shift(xp, arrays[g2], a2, boundaries[a2], backward),
            arrays[g2], k1, k2, have[0], have[1],
            fold_ref.plain_cells(xp, record, half_integer, dsig, dsigu, shape),
            iyee, mask)
        inside = fold_ref.membership(xp, record, half_integer, "xyz".index(dsigu),
                                     shape)
        fold_ref.ladder(xp, arrays[target], arrays["fu_" + target], increment,
                        coefficients["sinv_" + dsig], coefficients["sinv_" + dsigu],
                        inside)


# ---------------------------------------------------------------------------
# Reference grids — the marquee's own class plus the three needles
# ---------------------------------------------------------------------------

REFERENCE_GRIDS: Tuple[Dict[str, Any], ...] = (
    # R1: the marquee's class — dims=3, ONE-CELL x/y (all invariance flags
    # FALSE, grid.py:1165-1183), z-only cell, PML both z faces, kx bfast,
    # the natively-NP2 Courant.
    {"name": "r1_marquee_z_only", "cell": (0.1, 0.1, 4.8), "resolution": 10.0,
     "dimensions": 3, "boundaries": "periodic",
     "bfast": (KX_MARQUEE, 0.0, 0.0), "courant": COURANT_MARQUEE,
     "pml": ((0, 0), (0, 0), (2, 2))},
    # R2: full 3-D k, all components distinct — the k-indexing needle.
    {"name": "r2_full_k_3d", "cell": (1.2, 1.0, 1.4), "resolution": 10.0,
     "dimensions": 3, "boundaries": "periodic", "bfast": FULL_K,
     "courant": 0.35, "pml": ((0, 0), (0, 0), (3, 3))},
    # R3: DECLARED dims=2 — the invariance-guard needle (have_p kills k2 = kx
    # on By's Ez sum, which the curl's own difference would NOT zero).
    {"name": "r3_guard_2d", "cell": (2.0, 1.6, 0.0), "resolution": 10.0,
     "dimensions": 2, "boundaries": "periodic", "bfast": (0.4, 0.0, 0.0),
     "courant": 0.4375, "pml": ((2, 2), (2, 2), (0, 0))},
    # R4: metallic x — the ownership-mask needle (wall-row state bytes).
    {"name": "r4_metallic_x", "cell": (0.8, 0.6, 1.0), "resolution": 10.0,
     "dimensions": 3, "boundaries": ("metallic", "periodic", "periodic"),
     "bfast": K_METALLIC, "courant": 0.35, "pml": ((0, 0), (0, 0), (3, 3))},
    # R5: metallic x AND y — TWO masked axes. The synthetic, subnormal and
    # mutation legs all sweep metallic_xy and it is m8's needle, but without
    # this grid that configuration would be anchored only against this file's
    # own transcription, never against stepping.py.
    {"name": "r5_metallic_xy", "cell": (0.8, 0.6, 1.0), "resolution": 10.0,
     "dimensions": 3, "boundaries": ("metallic", "metallic", "periodic"),
     "bfast": K_METALLIC, "courant": 0.35, "pml": ((0, 0), (0, 0), (3, 3))},
    # R6: DECLARED dims=1 — the D1 arm of the invariance guard, where x AND y
    # are invariant and four flags are false at once. k rides z because
    # it is the D1 class whose k needs no invariant-axis partner, and
    # every one of the six scalars is gated to zero: the case pins the D1
    # guard arm and the pure -2*state advance. NOTE, and it is in the artifact
    # deliberately: MEEP's gv.has_field leaves Bx/Bz/Dy/Dz unallocated in D1
    # so step_db never steps them, where this engine steps all six. From the
    # PHYSICAL all-zero state that is invisible (advance = -2*0); the seeded
    # state below is a harness configuration, and this case's claim is
    # kernel == array path, not engine == MEEP-in-D1.
    {"name": "r6_dims1_z_only", "cell": (0.0, 0.0, 4.8), "resolution": 10.0,
     "dimensions": 1, "boundaries": "periodic", "bfast": (0.0, 0.0, 0.31),
     "courant": 0.35, "pml": ((0, 0), (0, 0), (2, 2)),
     "note": "D1: all six (k1,k2) gated to zero; MEEP steps only 2 of the 6 "
             "components here (gv.has_field) — kernel-vs-array-path only"},
)
REFERENCE_STEPS = 4


def _seed_host_real(shape: Tuple[int, int, int], rng,
                    scale: float = 1.0) -> np.ndarray:
    """Seeded float32 with a signed-zero plane."""
    host = (scale * rng.uniform(-1.0, 1.0, size=shape)).astype(np.float32)
    live = [axis for axis in range(3) if shape[axis] > 1]
    if live:
        host[_face(live[0], 0)] = -0.0
    return host


def _seed_host_state(shape: Tuple[int, int, int], rng,
                     subnormal: bool = False) -> np.ndarray:
    """Seeded NONZERO f_bfast state — mandatory (the only layer where the
    always-run-all-six binding and the -2*state advance are byte-visible) —
    with signed zeros, and DENSE subnormals on the subnormal leg (the
    marginally stable IIR never decays them, stepping.py:868-874)."""
    host = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    flat = host.reshape(-1)
    flat[5::13] = np.float32(-0.0)
    if subnormal:
        flat[0::3] = np.float32(1e-45)
        flat[1::7] = np.float32(-3e-44)
        flat[2::11] = np.float32(1e-40)
        flat[4::9] = np.float32(-1e-45)
    else:
        flat[0::17] = np.float32(1e-45)
        flat[3::19] = np.float32(-3e-44)
    return host


def _build_reference_fields(xp, spec: Dict[str, Any]):
    grid = Grid(resolution=spec["resolution"], cell_size=spec["cell"],
                boundaries=spec["boundaries"], dimensions=spec["dimensions"],
                courant=spec["courant"], bfast_scaled_k=tuple(spec["bfast"]),
                xp=xp)
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    index = np.arange(int(np.prod(grid.shape)),
                      dtype=np.float32).reshape(grid.shape)
    epsilon = (1.45 + 0.30 * np.sin(index * np.float32(0.037))).astype(np.float32)
    inverse = (np.float32(1.0) / epsilon).astype(np.float32)
    fields.set_isotropic_epsilon_volume(xp.asarray(np.ascontiguousarray(epsilon)),
                                        xp.asarray(np.ascontiguousarray(inverse)))
    pml = PML(grid=grid, thickness=tuple(spec["pml"]))
    rng = np.random.default_rng(SEED + 29)
    for name in ALL_STATE:
        getattr(fields, name)[...] = xp.asarray(
            _seed_host_real(tuple(grid.shape), rng))
    srng = np.random.default_rng(SEED + 31)
    for name in BFAST_ALL:
        getattr(fields, name)[...] = xp.asarray(
            _seed_host_state(tuple(grid.shape), srng))
    return grid, fields, pml


def run_reference_validation(results: Dict[str, Any], out_path: str, xp,
                             backend_name: str) -> Dict[str, Any]:
    """The transcription against stepping on real BFAST objects, 4 full
    cycles, byte-compared after every sub-step, STATES INCLUDED."""
    cases: List[Dict[str, Any]] = []
    for spec in REFERENCE_GRIDS:
        started = time.time()
        grid, fields, pml = _build_reference_fields(xp, spec)
        kinds = stepping._boundary_kinds(grid, pml)
        invariant = tuple(grid.is_invariant(axis) for axis in range(3))
        dtdx = grid.dt / grid.dx
        record = layer_record(grid, pml)
        arrays = {name: getattr(fields, name).copy() for name in CHECK_STATE}
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
            "kinds": list(kinds), "invariant": list(invariant),
            "bfast": [repr(v) for v in grid.bfast_scaled_k],
            "dtdx": repr(dtdx),
            "steps": REFERENCE_STEPS,
            "first_divergence": None,
        }
        # Accumulated over EVERY sub-step of every step, not rebound per slot:
        # a divergence at step 1 that self-heals by the last comparison must
        # still be counted, or the artifact reports "0/N differing" beside
        # bit_identical=False.
        totals = accumulator()
        for step in range(1, REFERENCE_STEPS + 1):
            for slot, engine_call, side in (
                    ("step_B", stepping.step_B, None),
                    ("update_H", stepping.update_H, "H"),
                    ("step_D", stepping.step_D, None),
                    ("update_E", stepping.update_E, "E")):
                engine_call(fields, pml)
                if slot in CURL_SUB_STEPS:
                    real_bfast_pml_step(xp, arrays, curl_tables[slot], dtdx,
                                        slot, kinds, grid.bfast_scaled_k, record)
                else:
                    # MEEP's start of H on the run's first update_H.
                    fold_ref.constitutive_step(
                        xp, arrays, constitutive_tables[side], side,
                        probe.CONSTITUTIVE_TERMS[side], record,
                        start_h=(side == "H" and step == 1))
                parts = {name: bit_compare(arrays[name], getattr(fields, name))
                         for name in CHECK_STATE}
                totals = accumulate(totals, parts)
                if (not all(p["bit_identical"] for p in parts.values())
                        and case["first_divergence"] is None):
                    case["first_divergence"] = f"step {step} after {slot}"
        case["verdict"] = totals
        case["comparisons"] = totals["comparisons"]
        if case["first_divergence"] is not None:
            case["verdict"]["bit_identical"] = False
        case["seconds"] = round(time.time() - started, 3)
        cases.append(case)
        log(f"[reference:{backend_name}] {spec['name']} "
            f"shape={tuple(grid.shape)} kinds={kinds} "
            f"identical={case['verdict']['bit_identical']} "
            f"(differing={case['verdict']['differing_floats']}/"
            f"{case['verdict']['total_floats']}) ({case['seconds']} s)")
        results.setdefault("reference", {})[backend_name] = {
            "ran": len(cases),
            "identical": sum(int(c["verdict"]["bit_identical"]) for c in cases),
            "steps_per_case": REFERENCE_STEPS,
            "states_compared": list(CHECK_STATE),
            "cases": cases}
        save(results, out_path)
    return results["reference"][backend_name]


# ---------------------------------------------------------------------------
# The synthetic sweeps — kernel vs the in-gate reference, on device
# ---------------------------------------------------------------------------

SHAPES: Tuple[Tuple[int, int, int], ...] = ((7, 6, 8), (6, 5, 9))

CONFIGS: Tuple[Dict[str, Any], ...] = (
    {"name": "periodic_xyz", "boundaries": (PERIODIC, PERIODIC, PERIODIC)},
    {"name": "metallic_x", "boundaries": (METALLIC, PERIODIC, PERIODIC)},
    {"name": "metallic_xy", "boundaries": (METALLIC, METALLIC, PERIODIC)},
)
METALLIC_CONFIGS = tuple(c for c in CONFIGS if c["name"] != "periodic_xyz")

K_SETS: Tuple[Tuple[str, Tuple[float, float, float]], ...] = (
    ("marquee_kx", (KX_MARQUEE, 0.0, 0.0)),
    ("full_k", FULL_K),
)
FULL_K_ONLY = (K_SETS[1],)

#: The m6 needle: a DECLARED dims=2 combo (invariant z) with an in-plane k —
#: the ONLY combos where dropping the guard is byte-visible.
GUARD_CONFIGS: Tuple[Dict[str, Any], ...] = (
    {"name": "guard_2d_declared",
     "boundaries": (PERIODIC, PERIODIC, PERIODIC),
     "shape": (7, 6, 1), "invariant": (False, False, True),
     "k": (0.4, 0.0, 0.0)},
    # A nonzero k on the declared-invariant axis: MEEP never reads the partner it
    # multiplies (the absent partner is not read), so it is admitted and stepped.
    {"name": "guard_2d_invariant_k",
     "boundaries": (PERIODIC, PERIODIC, PERIODIC),
     "shape": (7, 6, 1), "invariant": (False, False, True),
     "k": (0.4, 0.0, 0.3)},
)

DTDX_VALUES = (COURANT_MARQUEE, 0.35)  # both non-power-of-two; one is the marquee's own
LIVE_3D = (False, False, False)


def one_case(shape, config, k_label, bfast_k, invariant, dtdx, sub_step,
             guard, kernel=None, ks_override=None, has_bfast: int = 1,
             state_mode: str = "random", record_kind: str = "split") -> Dict[str, Any]:
    case: Dict[str, Any] = {
        "shape": list(shape), "config": config["name"], "k_set": k_label,
        "bfast": [repr(v) for v in bfast_k],
        "invariant": list(invariant), "dtdx": repr(dtdx),
        "sub_step": sub_step, "guard": guard, "state_mode": state_mode,
        "record_kind": record_kind,
    }
    names = FIELD_12 + FU_NAMES
    rng = np.random.default_rng(SEED + 51)
    arrays = {name: cp.asarray(_seed_host_real(tuple(shape), rng))
              for name in names}
    srng = np.random.default_rng(SEED + 53)
    for name in BFAST_ALL:
        arrays[name] = cp.asarray(_seed_host_state(
            tuple(shape), srng, subnormal=(state_mode == "subnormal")))
    half_integer = sub_step == "step_B"
    record = _symmetry_gate().record_for(record_kind, shape)
    coefficients = probe.synthetic_coefficients(cp, tuple(shape), half_integer)
    if record_kind == "split":
        coefficients = _symmetry_gate().fit_coefficients(cp, coefficients, record,
                                                         half_integer, shape)
    flat = probe.flatten_coefficients(coefficients)
    reference = {name: array.copy() for name, array in arrays.items()}
    real_bfast_pml_step(cp, reference, coefficients, dtdx, sub_step,
                        config["boundaries"], bfast_k, record)

    ks = bfast_curl.bfast_curl_coefficients(bfast_k, invariant,
                                            magnetic=half_integer)
    if ks_override is not None:
        ks = (ks_override(ks, bfast_k, invariant, sub_step)
              if callable(ks_override) else tuple(ks_override))
    codes = [CODE_OF[b] for b in config["boundaries"]]
    plan = bfast_curl.plan_bfast_pml_curl_from_arrays(
        sub_step, arrays, flat, codes, float(dtdx), ks,
        kernel=kernel, has_bfast=has_bfast, chunk_bounds=record)
    plan.run(guard=guard)
    cp.cuda.runtime.deviceSynchronize()

    targets = SUB_TARGETS[sub_step]
    checked = (targets + tuple("fu_" + t for t in targets)
               + bfast_curl.BFAST_STATE_NAMES[sub_step])
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
                  kernel=None, ks_override=None,
                  sub_steps=CURL_SUB_STEPS,
                  configs=CONFIGS, k_sets=K_SETS,
                  guard_configs=GUARD_CONFIGS,
                  state_mode: str = "random",
                  record_kinds=None) -> Dict[str, Any]:
    cases: List[Dict[str, Any]] = []
    if record_kinds is None:
        record_kinds = _symmetry_gate().RECORD_KINDS
    combos: List[Tuple] = []
    for shape in SHAPES:
        for config in configs:
            for k_label, bfast_k in k_sets:
                for dtdx in DTDX_VALUES:
                    for sub_step in sub_steps:
                        for kind in record_kinds:
                            combos.append((shape, config, k_label, bfast_k,
                                           LIVE_3D, dtdx, sub_step, kind))
    for config in guard_configs:
        for dtdx in DTDX_VALUES:
            for sub_step in sub_steps:
                for kind in record_kinds:
                    combos.append((config["shape"], config, config["name"],
                                   config["k"], config["invariant"], dtdx,
                                   sub_step, kind))
    total = len(guards) * len(combos)
    index = 0
    for guard in guards:
        for (shape, config, k_label, bfast_k, invariant, dtdx, sub_step,
             kind) in combos:
            index += 1
            started = time.time()
            case = one_case(shape, config, k_label, bfast_k, invariant, dtdx,
                            sub_step, guard, kernel=kernel,
                            ks_override=ks_override, state_mode=state_mode,
                            record_kind=kind)
            case["seconds"] = round(time.time() - started, 3)
            cases.append(case)
            verdict = case["verdict"]
            log(f"[{label}] {index}/{total} guard={guard} {sub_step} "
                f"{'x'.join(str(n) for n in shape)} {config['name']} "
                f"{k_label} {kind}: identical={verdict['bit_identical']} "
                f"(differing={verdict['differing_floats']}/"
                f"{verdict['total_floats']}) ({case['seconds']} s)")
            summary = _summarize(cases)
            summary["cases"] = cases
            results.setdefault("synthetic", {})[label] = summary
            save(results, out_path)
    return results["synthetic"][label]


def run_subnormal(results: Dict[str, Any], out_path: str) -> Dict[str, Any]:
    """The synthetic sweep with SUBNORMAL-seeded state — this family's hot
    FTZ surface (the state never decays), verified under the stripped
    policy, in-run strip counters recorded beside the verdict."""
    summary = run_synthetic(results, out_path, label="subnormal",
                            guards=(False,), state_mode="subnormal")
    summary["subnormal_policy"] = policy_stamp("cupy")
    summary["policy_problems"] = ftz_strip_license_reasons()
    results["synthetic"]["subnormal"] = summary
    save(results, out_path)
    return summary


# ---------------------------------------------------------------------------
# The HAS_BFAST=0 identity leg
# ---------------------------------------------------------------------------

#: The identity leg's own combos. A single boundary triple would leave the
#: BCX and BCZ metallic arms of the certified body unexercised by the ONLY leg
#: that certifies the HAS_BFAST=0 build; both shapes and both non-power-of-two
#: dtdx values ride along at no extra risk.
IDENTITY_CONFIGS: Tuple[Dict[str, Any], ...] = (
    {"name": "periodic_xyz", "shape": (7, 6, 8), "codes": [0, 0, 0],
     "dtdx": COURANT_MARQUEE},
    {"name": "metallic_x", "shape": (7, 6, 8), "codes": [1, 0, 0],
     "dtdx": 0.35},
    {"name": "metallic_y", "shape": (6, 5, 9), "codes": [0, 1, 0],
     "dtdx": COURANT_MARQUEE},
    {"name": "metallic_xz", "shape": (6, 5, 9), "codes": [1, 0, 1],
     "dtdx": 0.35},
)


def run_identity(results: Dict[str, Any], out_path: str) -> Dict[str, Any]:
    """HAS_BFAST=0 must reproduce the certified kernel byte-for-byte on the
    same seeded state, AND leave the states untouched — the tail is compiled
    OUT, not merely zeroed. The ks bound are NONZERO deliberately: only a
    nonzero pair makes an always-on HAS_BFAST defect byte-visible.

    SELF-CHECKING, which every other leg is by construction and this one is
    not: comparing two arrays that both took no launch is trivially identical
    and states_untouched is trivially True, so a leg that never launched would
    report a hollow pass. Both plans therefore run through a CountingKernel and
    both must move bytes off their seeds — the gate's own DISARMED discipline
    (three prior harness-disarm defects make it mandatory)."""
    from meep_gpu.triton_kernels import kernels, launch  # noqa: PLC0415

    leg: Dict[str, Any] = {"cases": []}
    for config in IDENTITY_CONFIGS:
        shape = tuple(config["shape"])
        for sub_step in CURL_SUB_STEPS:
            names = FIELD_12 + FU_NAMES
            rng = np.random.default_rng(SEED + 61)
            arrays = {name: cp.asarray(_seed_host_real(shape, rng))
                      for name in names}
            srng = np.random.default_rng(SEED + 63)
            for name in BFAST_ALL:
                arrays[name] = cp.asarray(_seed_host_state(shape, srng))
            state_before = {name: arrays[name].copy() for name in BFAST_ALL}
            twin = {name: arrays[name].copy() for name in names}
            seeds_before = {name: arrays[name].copy() for name in names}
            coefficients = probe.synthetic_coefficients(cp, shape,
                                                        sub_step == "step_B")
            flat = probe.flatten_coefficients(coefficients)
            codes = list(config["codes"])
            dtdx = float(config["dtdx"])
            candidate_counter = CountingKernel(bfast_curl.bfast_pml_curl_step)
            twin_counter = CountingKernel(kernels.pml_curl_step)
            # Synthetic coefficient tables: the every-cell chunk record, on both.
            bfast_curl.plan_bfast_pml_curl_from_arrays(
                sub_step, arrays, flat, codes, dtdx,
                (0.1, -0.2, 0.3, -0.4, 0.5, -0.6), has_bfast=0,
                kernel=candidate_counter,
                chunk_bounds=every_cell_record(shape)).run(guard=False)

            launch.plan_from_arrays(sub_step, twin, flat, codes, dtdx,
                                    kernel=twin_counter,
                                    chunk_bounds=every_cell_record(shape)
                                    ).run(guard=False)
            cp.cuda.runtime.deviceSynchronize()
            targets = SUB_TARGETS[sub_step]
            checked = targets + tuple("fu_" + t for t in targets)
            verdict = combine({name: bit_compare(arrays[name], twin[name])
                               for name in checked})
            states = combine({name: bit_compare(arrays[name],
                                                state_before[name])
                              for name in BFAST_ALL})
            moved = sum(int(not bit_compare(twin[name],
                                            seeds_before[name])["bit_identical"])
                        for name in checked)
            case = {"config": config["name"], "shape": list(shape),
                    "codes": codes, "dtdx": repr(dtdx), "sub_step": sub_step,
                    "verdict": verdict,
                    "states_untouched": states["bit_identical"],
                    "candidate_launches": candidate_counter.launches,
                    "certified_launches": twin_counter.launches,
                    "certified_arrays_moved": moved,
                    "of_checked": len(checked)}
            if candidate_counter.launches == 0 or twin_counter.launches == 0:
                case["error"] = ("DISARMED: a plan never launched — two "
                                 "untouched arrays compare identical and this "
                                 "leg would report a hollow pass")
            elif moved == 0:
                case["error"] = ("VACUOUS: the certified kernel moved no bytes "
                                 "off the seeds, so byte-identity here says "
                                 "nothing about the HAS_BFAST=0 build")
            leg["cases"].append(case)
            log(f"[identity] {config['name']}/{sub_step}: HAS_BFAST=0 vs "
                f"certified identical={verdict['bit_identical']} "
                f"states_untouched={states['bit_identical']} "
                f"launches={candidate_counter.launches}/"
                f"{twin_counter.launches} moved={moved}/{len(checked)}")
            results["identity"] = leg
            save(results, out_path)
    leg["identical"] = sum(int(c["verdict"]["bit_identical"]
                               and c["states_untouched"]
                               and not c.get("error"))
                           for c in leg["cases"])
    leg["ran"] = len(leg["cases"])
    leg["self_check"] = ("both plans launch-counted and the certified kernel "
                         "must move bytes off its seeds; either failure is "
                         "recorded as an error and fails the leg")
    leg["engine_plans_bind_has_bfast_1"] = (
        "plan_bfast_pml_curl constructs only behind a predicate requiring "
        "bfast_active and always passes has_bfast=1; the 0 arm exists for "
        "this leg alone")
    results["identity"] = leg
    save(results, out_path)
    return leg


# ---------------------------------------------------------------------------
# The refusal enumeration, asserted with reasons (host-capable)
# ---------------------------------------------------------------------------

def _np_build(cell=(0.1, 0.1, 4.8), dimensions=3,
              bfast=(KX_MARQUEE, 0.0, 0.0), courant=COURANT_MARQUEE,
              complex_storage=False, k_point=(0.0, 0.0, 0.0),
              skip_pml=False, pml_override=None, **grid_kwargs):
    grid = Grid(resolution=10.0, cell_size=cell, dimensions=dimensions,
                courant=courant, k_point=k_point, bfast_scaled_k=bfast,
                **grid_kwargs)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.enable_pml_storage()
    if skip_pml:
        return fields, None
    if pml_override is not None:
        return fields, PML(grid=grid, thickness=pml_override)
    thickness = tuple((2, 2) if grid.shape[axis] >= 6 else (0, 0)
                      for axis in range(3))
    return fields, PML(grid=grid, thickness=thickness)


def run_refusals(results: Dict[str, Any], out_path: str) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    failures: List[str] = []

    def record(name: str, ok: bool, detail: str) -> None:
        rows.append({"row": name, "as_expected": bool(ok), "detail": detail})
        if not ok:
            failures.append(f"refusal row {name}: {detail}")
        log(f"[refusals] {name}: as_expected={ok}")

    def reasons_of(fields, pml, sub_step="step_B"):
        return [r for r in bfast_curl.bfast_pml_curl_coverage(
            fields, pml, sub_step).reasons if "array module" not in r]

    # bfast + cylindrical: the GRID raises — upstream (grid.py:706-742, on
    # MEEP's own step_generic.cpp:376 missing '- F[i]').
    try:
        Grid(resolution=10.0, cell_size=(0.8, 0.0, 0.8), dimensions=2,
             cylindrical=True, bfast_scaled_k=(0.3, 0.0, 0.0))
        record("bfast_cylindrical", False, "Grid ACCEPTED bfast on cylindrical")
    except ValueError as exc:
        record("bfast_cylindrical", True, f"Grid raised: {exc}")

    # bfast + beta: cross-refusal in BOTH directions.
    from meep_gpu.triton_kernels import special_kz  # noqa: PLC0415

    fields, pml = _np_build(cell=(2.0, 1.6, 0.0), dimensions=2,
                            bfast=(0.4, 0.0, 0.0), courant=0.4375, beta=0.25)
    mine = reasons_of(fields, pml)
    theirs = special_kz.beta_pml_curl_coverage(fields, pml, "step_B").reasons
    record("bfast_beta_cross_refusal",
           any("fold order" in r or "byte-significant" in r for r in mine)
           and any("BFAST" in r for r in theirs),
           f"bfast side names the fold order: "
           f"{[r for r in mine if 'beta' in r or 'fold' in r]}; special_kz "
           f"side: {[r for r in theirs if 'BFAST' in r]}")

    # bfast + conductivity, both directions.
    from meep_gpu.triton_kernels import conductivity  # noqa: PLC0415

    fields, pml = _np_build()
    fields.set_d_conductivity(np.full(fields.grid.shape, 0.5, dtype=np.float32))
    mine = reasons_of(fields, pml, "step_D")
    theirs = conductivity.conductive_pml_curl_coverage(fields, pml,
                                                       "step_D").reasons
    record("bfast_conductivity_cross_refusal",
           any("conductivity" in r for r in mine)
           and any("BFAST" in r for r in theirs),
           f"bfast side: {[r for r in mine if 'conductivity' in r]}; "
           f"conductive side: {[r for r in theirs if 'BFAST' in r]}")

    # bfast + complex storage — refused by name (Phase A).
    fields, pml = _np_build(complex_storage=True)
    record("bfast_complex_storage",
           any("complex family" in r for r in reasons_of(fields, pml)),
           str(reasons_of(fields, pml)[:3]))

    # bfast + Bloch — the predicate refuses what Grid deliberately allows
    # (grid.py:719-724: independent constructor slots).
    grid = Grid(resolution=10.0, cell_size=(1.2, 1.0, 1.4), dimensions=3,
                courant=0.35, bfast_scaled_k=FULL_K, k_point=(0.25, 0.0, 0.0))
    took_both = grid.bfast_active and any(float(v) != 0.0 for v in grid.k_point)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_pml_storage()
    pml = PML(grid=grid, thickness={"z": 3})
    reasons = reasons_of(fields, pml)
    record("bfast_bloch_predicate_refuses_what_grid_allows",
           took_both and any("not exactly zero" in r or "k_point" in r
                             for r in reasons),
           f"grid took both slots={took_both}; reasons={reasons[:3]}")

    # bfast + a nonzero k on a DECLARED-invariant axis (clause 11a): ADMITTED. MEEP
    # does not read the partner such a k multiplies -- step_bfast takes a NULL
    # pointer and swaps it into the g2 slot (step_generic.cpp:342-346) -- and the
    # array path now takes the same one-partner forms (bit_parity_0_10 WP AP6b), as
    # does this kernel; the synthetic and engine legs step the class. The row
    # asserts the admission and that no reason still names the invariant axis.
    fields, pml = _np_build(cell=(2.0, 1.6, 0.0), dimensions=2,
                            bfast=(0.4, 0.0, 0.3), courant=0.4375)
    invariant_reasons = [r for r in reasons_of(fields, pml)
                         if "invariant" in r.lower()]
    record("bfast_k_on_an_invariant_axis_admitted",
           not invariant_reasons,
           f"kz on the invariant axis of a dims=2 grid: reasons naming an "
           f"invariant axis {invariant_reasons}")

    # bfast + mirror fold.
    fields, pml = _np_build(cell=(2.0, 1.6, 0.0), dimensions=2,
                            bfast=(0.4, 0.0, 0.0), courant=0.4375,
                            symmetry=("X",),
                            pml_override={"x": {"high": 2}, "y": 2})
    record("bfast_mirror_fold",
           any("mirror" in r or "folded" in r for r in reasons_of(fields, pml)),
           str([r for r in reasons_of(fields, pml)
                if "mirror" in r or "folded" in r]))

    # bfast + chi2/chi3 — planted through the class property, restored after.
    fields, pml = _np_build()
    original = Fields.has_nonlinearity
    try:
        Fields.has_nonlinearity = property(lambda self: True)
        reasons = reasons_of(fields, pml)
    finally:
        Fields.has_nonlinearity = original
    record("bfast_chi2_chi3", any("chi2/chi3" in r for r in reasons),
           str([r for r in reasons if "chi2" in r]))

    # bfast without PML — refused by name (and no_pml.py refuses bfast).
    fields, _none = _np_build(skip_pml=True)
    reasons = reasons_of(fields, None)
    record("bfast_no_pml", any("PML" in r for r in reasons),
           str([r for r in reasons if "PML" in r]))

    # non-cupy backend: the numpy-built marquee class is refused ONLY for the
    # backend on a laptop — exercised on either host by reading the full list.
    fields, pml = _np_build()
    full = bfast_curl.bfast_pml_curl_coverage(fields, pml, "step_B").reasons
    record("non_cupy_backend",
           any("not cupy" in r for r in full),
           f"backend clause fired on the numpy grid: "
           f"{[r for r in full if 'cupy' in r]}")

    leg = {"rows": rows, "failures": failures,
           "predicate_mutations_note": (
               "predicate mutations run at every merge in "
               "meep_gpu/test_triton_bfast.py: drop clause 11 (bfast_active — the "
               "inversion this whole product hangs on), drop the k_point "
               "pair (whose refused configuration makes the ARRAY PATH "
               "raise, so admitting it would answer where the engine will "
               "not), drop the beta clause, drop the conductivity clause, "
               "widen the boundary set")}
    results["refusals"] = leg
    save(results, out_path)
    return leg


# ---------------------------------------------------------------------------
# Mutations — source-level (compiled, renamed, launch-counted) and host-level
# ---------------------------------------------------------------------------

_TEMPORARY: List[str] = []


def compile_mutated(source: str, kernel_name: str):
    """Compile a mutated copy of the shipped kernel from a real file on disk
    (Triton reads source via inspect). The constexpr codes are re-declared in
    the header because a @triton.jit body may not read a plain module global; the
    shared forms are IMPORTED, as the shipped module imports them. The module name
    does not start with ``triton`` (Triton's dependency finder leaves such callees
    out of the kernel's cache key)."""
    header = (
        "import triton\nimport triton.language as tl\n"
        + fold_passes.MUTANT_IMPORTS
        + "PERIODIC = tl.constexpr(0)\nMETALLIC = tl.constexpr(1)\n\n")
    handle = tempfile.NamedTemporaryFile("w", suffix="_mutated_bfast.py",
                                         delete=False, encoding="utf-8")
    handle.write(header + source)
    handle.close()
    _TEMPORARY.append(handle.name)
    spec = importlib.util.spec_from_file_location(
        "bfast_mutant_" + str(len(_TEMPORARY)), handle.name)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = module  # type: ignore[union-attr]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return getattr(module, kernel_name)


def shipped_source() -> str:
    return textwrap.dedent(inspect.getsource(bfast_curl.bfast_pml_curl_step.fn))


#: name -> (rewrite, expectation, why): the BFAST pass's mutations and the root forms'
#: (KERNELS.md 7.1's inversions and 7.2's chunk-form battery), ``fold_pass_mutations``'
#: shared spellings. The synthetic sweep each is scored on carries every ``need``
#: (walled configs, both records' layers, the declared-2-D one-partner configs and
#: the invariant-axis k), so a catch anywhere in it is a catch.
SOURCE_MUTATIONS: Dict[str, Tuple[Callable[[str], Tuple[str, int]], str, str]] = {
    name: (rewrite, expectation, why)
    for name, why, expectation, rewrite, _needs in
    fold_passes.bfast_mutations() + fold_passes.form_mutations(False)}


class CountingKernel(gate.CountingKernel):
    pass


def _mutant_distinctness(name: str, mutant_kernel) -> Dict[str, Any]:
    """Platform fact: Triton's cache can serve a STALE binary to a
    renamed-but-source-similar entry point (measured) — so every mutant gets
    its OWN entry-point name, and the cache keys must differ from the true
    kernel's. Recorded per mutation; equality is a leg failure."""
    true_key = bfast_curl.bfast_pml_curl_step.cache_key
    mut_key = mutant_kernel.cache_key
    return {
        "entry_point": mutant_kernel.fn.__name__,
        "true_cache_key": str(true_key)[:16],
        "mut_cache_key": str(mut_key)[:16],
        "cache_keys_differ": bool(str(true_key) != str(mut_key)),
    }


def _renamed(source: str, suffix: str) -> Tuple[str, str]:
    name = f"bfast_pml_curl_step__{suffix}"
    return source.replace("def bfast_pml_curl_step(", f"def {name}(", 1), name


def run_mutations(results: Dict[str, Any], out_path: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    shipped = shipped_source()

    for name, (transform, expectation, why) in SOURCE_MUTATIONS.items():
        mutated, hits = transform(shipped)
        if hits == 0 or mutated == shipped:
            out[name] = {"error": "NEEDLE MISSED: the mutation matched "
                                  "nothing; the needle drifted off the kernel"}
            log(f"[mut] {name}: NEEDLE MISSED — nothing exercised")
            results["mutations"] = out
            save(results, out_path)
            continue
        renamed_source, entry = _renamed(mutated, name.split("_")[0] + "_"
                                         + str(len(out)))
        counter = CountingKernel(compile_mutated(renamed_source, entry))
        distinctness = _mutant_distinctness(name, counter.kernel)
        summary = run_synthetic(results, out_path,
                                label="mutation:" + name, guards=(False,),
                                kernel=counter)
        caught = summary["identical"] < summary["ran"]
        entry_row: Dict[str, Any] = {
            "sites": hits, "launches": counter.launches,
            "ran": summary["ran"], "identical": summary["identical"],
            "expectation": expectation, "why": why,
            "mutant_distinctness": distinctness,
            "caught": caught, "as_expected": caught == (expectation == "caught"),
        }
        if counter.launches == 0:
            entry_row["error"] = ("DISARMED: the mutated kernel never "
                                  "launched — the leg measured nothing")
        if not distinctness["cache_keys_differ"]:
            entry_row["error"] = ("STALE-BINARY HAZARD: the mutant shares the "
                                  "true kernel's cache key (the measured "
                                  "aliasing class)")
        log(f"[mut] {name}: sites={hits} launches={counter.launches} "
            f"identical={summary['identical']}/{summary['ran']} "
            f"caught={entry_row['caught']} expectation={expectation}")
        out[name] = entry_row
        results["mutations"] = out
        save(results, out_path)

    # m1 (host): the k pair interchanged per term — k1 indexed by the SAME
    # partner's own axis instead of the OTHER's (the trap at :787-796). Armed
    # on the full-k configs, where every component is distinct.
    true_counter = CountingKernel(bfast_curl.bfast_pml_curl_step)
    summary = run_synthetic(
        results, out_path, label="mutation:m1_k_pair_interchanged",
        guards=(False,), kernel=true_counter,
        ks_override=lambda ks, *_: (ks[1], ks[0], ks[3], ks[2], ks[5], ks[4]),
        k_sets=FULL_K_ONLY, guard_configs=())
    out["m1_k_pair_interchanged"] = {
        "host": True, "launches": true_counter.launches,
        "ran": summary["ran"], "identical": summary["identical"],
        "needle": "full_k configs (distinct kx, ky, kz)",
        "caught": summary["identical"] < summary["ran"],
        "expectation": "caught",
        "as_expected": summary["identical"] < summary["ran"]}
    if true_counter.launches == 0:
        out["m1_k_pair_interchanged"]["error"] = "DISARMED: never launched"
    log(f"[mut] m1: identical={summary['identical']}/{summary['ran']}")
    results["mutations"] = out
    save(results, out_path)

    # m2 (host): the D-side negation dropped — the plan built with the
    # MAGNETIC side's scalars on step_D (stepping.py:913-914 inverted).
    m2_counter = CountingKernel(bfast_curl.bfast_pml_curl_step)
    summary = run_synthetic(
        results, out_path, label="mutation:m2_d_negation_dropped",
        guards=(False,), kernel=m2_counter, sub_steps=("step_D",),
        ks_override=lambda ks, k, inv, sub: bfast_curl.bfast_curl_coefficients(
            k, inv, magnetic=True))
    out["m2_d_negation_dropped"] = {
        "host": True, "launches": m2_counter.launches,
        "ran": summary["ran"], "identical": summary["identical"],
        "caught": summary["identical"] < summary["ran"],
        "expectation": "caught",
        "as_expected": summary["identical"] < summary["ran"]}
    if m2_counter.launches == 0:
        out["m2_d_negation_dropped"]["error"] = "DISARMED: never launched"
    log(f"[mut] m2: identical={summary['identical']}/{summary['ran']}")
    results["mutations"] = out
    save(results, out_path)

    # m6 (host): the invariance guard removed — scalars built with the
    # all-live table. Needle = the DECLARED-2-D config, the ONLY combos where
    # the guard is byte-visible; on the 3-D combos every flag is already
    # False and the mutant is the true kernel (predicted null, recorded).
    m6_counter = CountingKernel(bfast_curl.bfast_pml_curl_step)
    summary = run_synthetic(
        results, out_path, label="mutation:m6_invariance_guard_removed",
        guards=(False,), kernel=m6_counter,
        ks_override=lambda ks, k, inv, sub: bfast_curl.bfast_curl_coefficients(
            k, (False, False, False), magnetic=(sub == "step_B")),
        configs=(), k_sets=())
    out["m6_invariance_guard_removed"] = {
        "host": True, "launches": m6_counter.launches,
        "ran": summary["ran"], "identical": summary["identical"],
        "needle": "guard_2d_declared (dims=2, kx): have_p must kill k2=kx on "
                  "the Ez/Hz sum that the curl's own difference would not "
                  "zero (stepping.py:889-898)",
        "predicted_null_on_3d": "on every dims=3 combo the invariance table "
                                "is all-live and the mutant binds the true "
                                "scalars — byte-identical BY CONSTRUCTION, "
                                "so those combos are not armed",
        "caught": summary["identical"] < summary["ran"],
        "expectation": "caught",
        "as_expected": summary["identical"] < summary["ran"]}
    if m6_counter.launches == 0:
        out["m6_invariance_guard_removed"]["error"] = "DISARMED: never launched"
    log(f"[mut] m6: identical={summary['identical']}/{summary['ran']}")
    results["mutations"] = out
    save(results, out_path)

    # m9 (host): the state written to a copy, not in place. The engine plan
    # must bind fields.f_bfast_* POINTER-IDENTICALLY, and the bytes must move
    # through the held reference — the driver's sync backup/restore
    # (driver.py:4126-4135) depends on exactly this; the composition probe's
    # sync case holds the driver half.
    spec = REFERENCE_GRIDS[1]  # r2_full_k_3d
    grid, fields, pml = _build_reference_fields(cp, spec)
    plan = bfast_curl.plan_bfast_pml_curl(fields, pml, "step_B")
    if plan is None:
        out["m9_state_written_to_copy"] = {
            "error": "the predicate refused the engine-built r2 configuration"}
    else:
        held = fields.f_bfast_Bx
        pointer_identical = all(
            pointer.array is getattr(fields, state_name)
            for pointer, state_name in zip(
                plan._states, bfast_curl.BFAST_STATE_NAMES["step_B"]))
        before = held.copy()
        plan.run(guard=False)
        cp.cuda.runtime.deviceSynchronize()
        moved = not bit_compare(held, before)["bit_identical"]
        still_same_object = fields.f_bfast_Bx is held
        out["m9_state_written_to_copy"] = {
            "host": True, "launches": 1,
            "pointer_identical_binding": pointer_identical,
            "bytes_moved_through_held_reference": moved,
            "attribute_still_the_same_object": still_same_object,
            "caught": pointer_identical and moved and still_same_object,
            "expectation": "caught",
            "as_expected": pointer_identical and moved and still_same_object}
    log(f"[mut] m9: {out['m9_state_written_to_copy']}")
    results["mutations"] = out
    save(results, out_path)

    out["predicted_nulls"] = {
        "negate_then_round_vs_round_then_negate": (
            "the D side negates k in host f64 then rounds once; f64 negation and "
            "f32 rounding commute EXACTLY, so the two spellings are the same bytes "
            "on every k -- recorded, not armed"),
        "subtract_the_negated_increment_vs_add_the_raw": (
            "the ladder subtracts the NEGATED raw word, the array path's spelling "
            "and MEEP's f += raw word for word (x - (-y) is x + y in IEEE-754, "
            "signed zeros included), while x - (+0) keeps an unowned cell's -0 -- "
            "recorded, not armed (the hazard would be Triton UNARY minus, which "
            "the pass never uses: the negation is spelled * -1.0)"),
        "shifted_plus_center_vs_center_plus_shifted": (
            "IEEE addition is commutative, so each sum's operand order changes no "
            "byte. Held by test_triton_bfast.py's source assertion on the operand "
            "order; recorded, not armed"),
    }
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
        reference = Fields(grid=grid, force_complex_fields=False)
        reference.enable_pml_storage()
        reference.set_isotropic_epsilon_volume(fields.epsilon_for("Ex"),
                                               fields.inverse_epsilon_for("Ex"))
        for name in CHECK_STATE:
            getattr(reference, name)[...] = getattr(fields, name)

        case: Dict[str, Any] = {"spec": {k: repr(v) for k, v in spec.items()},
                                "shape": list(grid.shape),
                                "steps": ENGINE_STEPS,
                                "first_divergence": None}
        plans = {
            "step_B": bfast_curl.plan_bfast_pml_curl(fields, pml, "step_B"),
            "update_H": bfast_curl.plan_bfast_run_constitutive(fields, pml, "H"),
            "step_D": bfast_curl.plan_bfast_pml_curl(fields, pml, "step_D"),
            "update_E": bfast_curl.plan_bfast_run_constitutive(fields, pml, "E"),
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
        totals = accumulator()   # every sub-step counted, not just the last
        for step in range(1, ENGINE_STEPS + 1):
            for slot, engine_call in (("step_B", stepping.step_B),
                                      ("update_H", stepping.update_H),
                                      ("step_D", stepping.step_D),
                                      ("update_E", stepping.update_E)):
                plans[slot].run()
                engine_call(reference, pml)
                cp.cuda.runtime.deviceSynchronize()
                parts = {name: bit_compare(getattr(fields, name),
                                           getattr(reference, name))
                         for name in CHECK_STATE}
                totals = accumulate(totals, parts)
                if (case["first_divergence"] is None
                        and not all(p["bit_identical"]
                                    for p in parts.values())):
                    case["first_divergence"] = f"step {step} after {slot}"
        case["verdict"] = totals
        case["comparisons"] = totals["comparisons"]
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

DEFAULT_LEGS = "reference,refusals,synthetic,subnormal,identity,mutations,engine"
DEVICE_LEGS = ("synthetic", "subnormal", "identity", "mutations", "engine")


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
        "tranche": "BFAST (bfast_scaled_k) — Phase A: real-f32 "
                   "split-field-PML curl",
        "granularity": "sub-step (the composition probe holds the whole-step "
                       "claim)",
        "correctness_only": "no throughput or timing claims in this artifact",
        "step_budgets": {"reference": REFERENCE_STEPS, "engine": ENGINE_STEPS,
                         "note": "bit-identity is claimed for exactly these "
                                 "stated budgets and no further"},
        "marquee": {"kx": repr(KX_MARQUEE), "courant": repr(COURANT_MARQUEE),
                    "computed_as": "1.4*sin(35.7 deg); (1 - kx)/sqrt(3) — "
                                   "MEEP test_refl_angular.py:50-52's own "
                                   "user-code computation"},
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
        elif leg_name == "subnormal":
            summary = run_subnormal(results, args.out)
            if summary["identical"] != summary["ran"]:
                failures.append(f"subnormal-state sweep: "
                                f"{summary['identical']}/{summary['ran']} "
                                f"identical")
            if summary["policy_problems"]:
                failures.append("subnormal leg policy: "
                                + "; ".join(summary["policy_problems"]))
        elif leg_name == "identity":
            leg = run_identity(results, args.out)
            if leg["identical"] != leg["ran"]:
                failures.append(f"identity leg: {leg['identical']}/"
                                f"{leg['ran']} HAS_BFAST=0 cases identical "
                                f"with untouched states")
            for case in leg["cases"]:
                if case.get("error"):
                    failures.append(f"identity {case['config']}/"
                                    f"{case['sub_step']}: {case['error']}")
        elif leg_name == "mutations":
            out = run_mutations(results, args.out)
            for name, entry in out.items():
                if name == "predicted_nulls":
                    continue
                if entry.get("error"):
                    failures.append(f"mutation {name}: {entry['error']}")
                elif not entry.get("as_expected", entry.get("caught")):
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
