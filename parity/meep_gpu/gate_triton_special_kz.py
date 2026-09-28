"""Bit-identity gate for the special_kz (grid.beta) Triton kernels.

The module under test is ``meep_gpu/triton_kernels/special_kz.py``, which the
Triton planner dispatches for a ``grid.beta`` run under its release entry (nothing here
changes that). This gate is the arbiter its docstring promises: every grouping
choice the kernels could not force by construction — the term's position between
curl and mask, the subtraction-as-negated-add, the imaginary-coefficient-left
product's EXPANSION arm, the unrotated center partner, the HAS_BETA constexpr's
identity arm — is HELD HERE, empirically, per platform.

CLONED from the complex tranche's harness (``gate_triton_complex.py``, job
2330), and deliberately IMPORTING it rather than restating: the subnormal
policy machinery (``install_ftz_strip`` / ``ftz_strip_license_reasons`` /
``policy_stamp`` / ``probe_record_policy_reasons``), the exact-fma32 expansion
probe, the complex reference transcription (``complex_curl`` / ``complex_mask``
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

* ``expansion``  — the complex tranche's four patterns PLUS the NEW
  imaginary-coefficient-left pattern (``special_kz.BETA_PROBE_PATTERN``): the
  platform bytes of ``complex64 scalar (signed-zero real, ±imag) * complex64
  array`` with the coefficient LEFT (stepping.py:811), classified against
  exact-fma candidates on random + signed-zero + underflow vectors, per
  coefficient value, all required to agree. Writes the probe artifact the
  complex beta curl binds to (``beta_expansion_from_probe`` refuses without
  the new pattern, §4(k)).
* ``reference``  — a SECOND transcription of the beta curl sub-steps (the
  laptop test file carries a third), pinned against ``stepping.step_B``/
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
  complex), guard on/off, seeds carrying signed-zero planes (the beta term is
  a pure scalar-by-field product — signed zeros flow straight through).
* ``identity``   — R5: the HAS_BETA=0 build byte-identical to the certified
  plain kernels (``kernels.pml_curl_step`` / ``bloch_pml_curl_step``) on the
  same seeded state; and the engine-route plans bind HAS_BETA=1 only.
* ``refusals``   — R8 + C5 + R9, asserted WITH reasons, host-capable: real+
  offdiag+beta (predicate refusal AND stepping's own ValueError), beta on a
  3-D grid (Grid raises), beta+cylindrical (Grid raises), beta+BFAST,
  beta+chi2/chi3, no-PML+beta, conductivity cross-refusal in BOTH directions
  (this module refuses conductive targets; ``conductivity.py`` refuses beta),
  C5 complex+fold+beta recorded as a predicted null ('complex-by-fold
  inherited from the complex tranche, not a beta gap'), R9 folded-real
  recorded as Phase B deferred.
* ``mutations``  — armed and launch-counted, DISARMED/NEEDLE-MISSED are
  failures, and a multi-family mutation must be caught in EACH armed family
  (an OR would let a family-specific miss hide behind the other family's
  catch): m1 sign swap between the two targets; m2 ±i swap between the B
  and D sides (host, complex); m3 term scaled by dtdx (both families); m4
  partner read from an IN-PLANE shifted operand — b_x/a_y, deliberately NOT
  the z-shifted one: every sweep shape has nz = 1 with z periodic, so
  oz == idx and a b_z/a_z mutant is byte-identical to the true kernel and
  measures nothing; m5 wrap-phase rotation applied to the partner (complex,
  phased config); m6 final negation dropped; m7 beta applied to target 2
  (both families); m8 per-element f64 coefficient instead of
  host-rounded-once (the UNROUNDED coefficient smuggled through the two fp32
  slots as a hi/lo pair — the f64-annotation route is dead on Triton 3.1.0,
  whose jit.py honors int/uint/bool annotations only; job 2332 measured it
  as a 72/72-identical false needle — with the PER-COMBO pair bound so every
  combo's catch isolates the double rounding rather than a wrong-beta
  constant); m9 signed-zero cross terms folded away (complex) — caught at
  the PRODUCT layer (a wrapper kernel stores the helper's own words, welded
  per word to the licensed arm's exact-fma32 candidate AND to the fold's
  emulation) AND, since the 2026-08-12 complex-helper spelling fix, at the
  STORED layer too: the inlined complex_fields helpers' addends were unary
  ``-`` until then, and Triton's ``0.0 - x`` lowering canonicalized every ±0
  addend before a store (measured, job 2332's predicted-4-got-0 engineered
  leg — kept as the record of the old spelling); with the ``* -1.0``
  spelling the engineered leg re-asserts the restored IEEE-negation
  prediction (4 words under either arm, device recut to confirm); m10 the
  real kernel launched on complex storage's word view. m11 (predicate
  mutations: drop the offdiag-real clause, drop
  the bfast clause, widen the boundary set) lives in
  ``meep_gpu/test_triton_special_kz.py`` where it runs at every merge, and
  is recorded here by reference.
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

    python -u gate_triton_special_kz.py --legs expansion,reference,refusals \\
        --out /tmp/beta_gate_local.json
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
import gate_triton_complex as gate  # noqa: E402
import probe_fused_kernel_bit_identity as probe  # noqa: E402

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
_face = probe._face
PERIODIC = probe.PERIODIC
METALLIC = probe.METALLIC
CODE_OF = gate.CODE_OF

install_ftz_strip = gate.install_ftz_strip
ftz_strip_license_reasons = gate.ftz_strip_license_reasons
policy_stamp = gate.policy_stamp
probe_record_policy_reasons = gate.probe_record_policy_reasons

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


# ---------------------------------------------------------------------------
# The expansion leg — base patterns + the imaginary-coefficient-left pattern
# ---------------------------------------------------------------------------

#: The coefficient values the new pattern is measured with: the plans' own
#: words for the corpus betas on both sides and both signs, so every signed-
#: zero real word the engine can bind is among the measured coefficients.
def _beta_coefficient_scalars() -> List[np.complex64]:
    scalars: List[np.complex64] = []
    for beta, dt in ((BETA_KZ2D, 1.0 / 24.0), (BETA_GRATING, 0.034),
                     (BETA_SPECIAL_KZ, 0.035), (0.2, 0.05)):
        for magnetic in (True, False):
            for pair in special_kz.beta_curl_coefficients(
                    beta, dt, magnetic, complex_storage=True):
                value = np.empty((), dtype=np.complex64)
                value.real = np.float32(pair[0])  # plane assignment keeps -0.0
                value.imag = np.float32(pair[1])
                scalars.append(value)
    return scalars


def _beta_pattern_vectors(rng) -> Tuple[np.ndarray, np.ndarray]:
    """(z_re, z_im) rows: random + the signed-zero and subnormal rows on which
    the c_re cross terms are visible at the PRODUCT layer — the premise behind
    accepting AMBIGUOUS_BOTH for this pattern (and the reason m9's catch lives
    at the product layer: at the STORED layer those ±0 differences are
    absorbed — on random seeds by ``x - (±0) == x``, and on ALL states on this
    platform by Triton's ``0.0 - x`` unary-minus lowering in the inlined
    helpers, measured by job 2332)."""
    zr = list(rng.uniform(-2.0, 2.0, gate.RANDOM_VECTORS))
    zi = list(rng.uniform(-2.0, 2.0, gate.RANDOM_VECTORS))
    for zero in (0.0, -0.0):
        for other in (1.0, -1.0, 0.75, -0.5, 0.0, -0.0):
            zr.append(zero); zi.append(other)  # noqa: E702
            zr.append(other); zi.append(zero)  # noqa: E702
    for small in (1e-45, -1e-45, 3e-44, -3e-44, 1e-40, -1e-40):
        for other in (1.0, -1.0, 0.0, -0.0):
            zr.append(small); zi.append(other)  # noqa: E702
            zr.append(other); zi.append(small)  # noqa: E702
    # WORD-PRESERVING, not np.asarray: the six ``small`` values above are
    # subnormal, and ``np.asarray([1e-45], np.float32)`` is a float64 -> float32
    # CONVERSION that a flushing HOST answers with zero — destroying them before
    # the device is asked anything. Same repair, same reason, as the base gate's
    # _zero_imag_operands. Measured: 48 of this set's 4240 operand words are
    # subnormal and were being lost this way under 'flush'.
    out_zr = gate._asarray_float32(zr)
    out_zi = gate._asarray_float32(zi)
    n_zr, n_zi = _normal_underflow_rows()
    return (np.concatenate([out_zr, n_zr]), np.concatenate([out_zi, n_zi]))


def _normal_underflow_rows() -> Tuple[np.ndarray, np.ndarray]:
    """(z_re, z_im) rows with all-NORMAL operands and an underflowing product.

    WHAT THIS PATTERN CAN AND CANNOT SEPARATE, measured rather than assumed.
    The beta coefficient is pure imaginary, so ``c_re`` is a signed zero,
    ``c_re * zr`` is exactly ±0, and ``fma(±0, zr, X) == X``: FMA_V1 and NAIVE
    are bit-identical here for EVERY vector set, under either policy (measured
    2026-08-17: 0 words apart on all 2,120 pre-existing rows and on every row
    added below). :data:`AMBIGUOUS_BOTH` is therefore this pattern's permanent
    and correct verdict, not a weakness to be engineered away.

    What the rows below restore is the OTHER half of the licence rule. An
    ambiguity claim only licenses when the known-wrong ``FMA_V2_diagnostic`` arm
    is excluded — and V2 fuses the other product, so it separates only where the
    product underflows. Every such row in the set above reaches that state via a
    subnormal OPERAND, which the ship policy destroys on first use: measured, the
    diagnostic sits 64 words from FMA_V1 under 'keep' and **0 under 'flush'**,
    which is exactly why expansion_license refused special_kz on the 2026-08-17
    flush probe ("a comparison the known-wrong transcription passes has no power
    to license anything").

    These rows have no subnormal operand at all — only a subnormal PRODUCT — so
    they survive flush: with them the diagnostic separates by 256 words under
    'flush' and the pre-existing 64 under 'keep' is unchanged.

    The words are the base gate's :data:`gate._NORMAL_UNDERFLOWING_ZR` reused
    unchanged, not a new constant: this orientation's band is |z| normal with
    |z| * max|beta*dt| < FLT_MIN, i.e. |z| < 8.0323e-38, and all eight of that
    constant's words (1.175e-38 .. 2.351e-38) fall inside it. Built from WORDS,
    never routed back through a float64 -> float32 conversion.
    """
    zr: List[np.float32] = []
    zi: List[np.float32] = []
    for zr_word in gate._NORMAL_UNDERFLOWING_ZR:
        for imag in gate._NORMAL_UNDERFLOWING_ZI:
            zr.append(gate._float32_from_word(zr_word))
            zi.append(np.float32(imag))  # exact, never subnormal
    return (np.ascontiguousarray(np.array(zr, dtype=np.float32)),
            np.ascontiguousarray(np.array(zi, dtype=np.float32)))


def measure_beta_expansion_record(xp, backend_name: str) -> Dict[str, Any]:
    """The complex tranche's record extended with the beta coefficient pattern.

    Platform bytes: ``xp.multiply(coefficient_scalar, z)`` — the coefficient
    LEFT, exactly the S:784 orientation. Candidates: the FIRST operand's
    products fused (``gate._candidates_c8`` with the coefficient as the first
    operand) — the same FMA_V1 shape the phase rotation certified, measured
    here rather than assumed because the scalar-broadcast dispatch is its own
    compiled loop. Every coefficient value must classify identically; a
    disagreement across coefficients refuses by name.

    THE ENTRY CARRIES THE EVIDENCE, not only the verdict. Until 2026-08-15 the
    per-coefficient entry kept ``classified`` and ``mismatch_words`` and dropped
    ``licensable_arms_disagreement_words`` and ``matches``, which ``_classify``
    had already measured — harmless while this tranche's own loop accepted the
    word AMBIGUOUS_BOTH on trust, and not harmless now that the licence is the
    base family's rule, which requires an ambiguity claim to be backed by the
    measured arms-apart count on a positive vector set and by no DIAGNOSTIC arm
    having passed the same comparison. Dropping the evidence would make this
    gate's own honest record refuse itself.
    """
    record = gate.measure_expansion_record(xp, backend_name)
    rng = np.random.default_rng(SEED + 137)
    zr, zi = _beta_pattern_vectors(rng)
    z = gate._interleave_c8(zr, zi)
    per_coefficient: List[Dict[str, Any]] = []
    classes = set()
    for scalar0d in _beta_coefficient_scalars():
        # S:784 binds a NUMPY SCALAR (``dtype.type(coefficient)``) on both
        # array paths, and CuPy ufuncs reject 0-d numpy ndarrays outright
        # (job 2331's TypeError) — so the 0-d carrier collapses to its numpy
        # scalar here, byte-preserving (signed-zero real included).
        scalar = scalar0d[()]
        product = probe.to_host(xp.multiply(scalar, xp.asarray(z))
                                ).astype(np.complex64)
        platform = (np.ascontiguousarray(product.real),
                    np.ascontiguousarray(product.imag))
        candidates = gate._candidates_c8(
            np.full(zr.shape, scalar.real, np.float32),
            np.full(zr.shape, scalar.imag, np.float32), zr, zi)
        detail = gate._classify(platform, candidates)
        entry = {"coefficient": repr(complex(scalar)),
                 "coefficient_real_sign": math.copysign(1.0, float(scalar.real)),
                 **{k: detail[k] for k in detail}}
        per_coefficient.append(entry)
        classes.add(detail["classified"])
    verdict = classes.pop() if len(classes) == 1 else "DISAGREES_ACROSS_COEFFICIENTS"
    record["patterns"][special_kz.BETA_PROBE_PATTERN] = verdict
    record["detail"][special_kz.BETA_PROBE_PATTERN] = per_coefficient
    record["vectors"][special_kz.BETA_PROBE_PATTERN] = int(zr.size)
    # Restamp AFTER the extra measurement so the strip counters cover it too.
    record["subnormal_policy"] = gate.policy_stamp(backend_name)
    return record


def run_expansion(results: Dict[str, Any], out_path: str,
                  probe_artifact_path: str) -> Optional[Dict[str, Any]]:
    """Both backends measured; the CUPY record is what the kernels bind to.
    Policy first, then the EXTENDED license (``beta_expansion_license``).

    THE ARTIFACT RECORDS THE BASIS AND THE EXCLUSIONS BY NAME. ``basis`` is
    'measured' when a live pattern discriminated this run and
    'environment_default' when none did and the arm came from a prior
    measurement; ``discriminating`` names the patterns that voted and
    ``non_discriminating`` the ones whose measured blindness took them out of
    the agreement test, so a reader can see exactly what did and did not
    contribute rather than being handed an arm and a word.
    """
    leg: Dict[str, Any] = {}
    leg["numpy"] = measure_beta_expansion_record(np, "numpy")
    log(f"[expansion] numpy patterns: {leg['numpy']['patterns']}")
    cupy_record = None
    if cp is not None:
        cupy_record = measure_beta_expansion_record(cp, "cupy")
        log(f"[expansion] cupy patterns: {cupy_record['patterns']}")
        save(cupy_record, probe_artifact_path)
        leg["cupy"] = cupy_record
        leg["artifact"] = probe_artifact_path
        policy_problems = ftz_strip_license_reasons()
        leg["subnormal_policy"] = policy_stamp("cupy")
        if policy_problems:
            leg["refusal"] = "subnormal policy: " + "; ".join(policy_problems)
            leg["licensed_expansion"] = None
            log(f"[expansion] POLICY REFUSAL: {leg['refusal']}")
            cupy_record = None
        else:
            verdict = special_kz.beta_expansion_license(cupy_record)
            expansion = verdict["expansion"]
            leg["license"] = verdict
            leg["licensed_expansion"] = verdict["arm"]
            log(f"[expansion] licence: arm={verdict['arm']} "
                f"basis={verdict['basis']} discriminating="
                f"{sorted(verdict['discriminating'])} non_discriminating="
                f"{verdict['non_discriminating']} policy={verdict['policy']!r}")
            if verdict["basis"] == "environment_default":
                log(f"[expansion] DEFAULTED FROM ENVIRONMENT (not measured this "
                    f"run): {verdict['why_arbitrary']}")
            if expansion is None:
                leg["refusal"] = (
                    "the cupy record does not license a single EXPANSION "
                    "constexpr over the EXTENDED pattern set "
                    f"({special_kz.BETA_PROBE_PATTERNS}); the complex beta "
                    "curl may not launch — a disagreeing platform earns a new "
                    "arm designed from the per-coefficient mismatches above; "
                    "refused by name: " + "; ".join(verdict["refusals"]))
                log(f"[expansion] REFUSAL: {leg['refusal']}")
    else:
        leg["cupy"] = "skipped: cupy is not importable on this host"
        log("[expansion] cupy leg SKIPPED cleanly: no cupy on this host")
    results["expansion"] = leg
    save(results, out_path)
    return cupy_record


# ---------------------------------------------------------------------------
# The in-gate reference — the beta term added to the shared transcriptions
# ---------------------------------------------------------------------------
#
# SECOND transcription (the laptop test file's is the third). The term is
# computed INLINE from stepping.py:797-811's arithmetic rather than by calling
# special_kz.beta_curl_coefficients, so the gate and the module cannot agree by
# sharing a defect in the host function.

#: target index within a sub-step -> (which operand is the partner, sign):
#: index 0 pairs the SECOND source's center at +1 (Bx <- Ey / Dx <- Hy),
#: index 1 the FIRST source's center at -1 (By <- Ex / Dy <- Hx), index 2
#: nothing (stepping.py:384-391 / :467-474; step_db.cpp:148-176).
BETA_PARTNERS = {0: ("second", +1.0), 1: ("first", -1.0)}


def beta_term(partner: Any, sign: float, beta: float, dt: float,
              magnetic: bool) -> Any:
    """``-(dtype.type(sign*2*pi*beta*dt [* ±1j]) * partner)`` — S:770-784 inline."""
    coefficient: Any = sign * 2.0 * math.pi * beta * dt
    if partner.dtype.kind == "c":
        coefficient = coefficient * (1j if magnetic else -1j)
    return -(partner.dtype.type(coefficient) * partner)


def real_beta_pml_step(xp: Any, arrays: Dict[str, Any],
                       coefficients: Dict[str, Any], dtdx: float,
                       sub_step: str, boundaries: Sequence[str],
                       beta: float, dt: float) -> None:
    """One REAL beta curl sub-step, in place — the shared probe's real
    transcription with the term inserted between curl and mask."""
    terms = probe.B_PML_TERMS if sub_step == "step_B" else probe.D_PML_TERMS
    backward = sub_step == "step_D"
    magnetic = not backward
    for index, (target, g1, a1, g2, a2, dsig, dsigu, iyee) in enumerate(terms):
        curl = probe.reference_curl(xp, arrays[g1], arrays[g2], a1, a2, dtdx,
                                    backward, boundaries, "array_order")
        if index in BETA_PARTNERS:
            which, sign = BETA_PARTNERS[index]
            partner = arrays[g2] if which == "second" else arrays[g1]
            curl = curl + beta_term(partner, sign, beta, dt, magnetic)
        probe.reference_mask(curl, iyee, boundaries)
        probe.reference_recurrence(
            arrays[target], arrays["fu_" + target], curl,
            coefficients["kms_" + dsig], coefficients["sinv_" + dsig],
            coefficients["kms_" + dsigu], coefficients["sinv_" + dsigu])


def complex_beta_pml_step(xp: Any, arrays: Dict[str, Any],
                          coefficients: Dict[str, Any], dtdx: float,
                          sub_step: str, boundaries: Sequence[str],
                          phases: Sequence[Optional[complex]],
                          beta: float, dt: float) -> None:
    """One COMPLEX beta curl sub-step, in place — the complex gate's
    transcription with the ±i term inserted between curl and mask."""
    terms = probe.B_PML_TERMS if sub_step == "step_B" else probe.D_PML_TERMS
    backward = sub_step == "step_D"
    magnetic = not backward
    for index, term in enumerate(terms):
        target, g1, _a1, g2, _a2, dsig, dsigu, iyee = term
        curl = gate.complex_curl(xp, arrays, term, dtdx, backward, boundaries,
                                 phases)
        if index in BETA_PARTNERS:
            which, sign = BETA_PARTNERS[index]
            partner = arrays[g2] if which == "second" else arrays[g1]
            curl = curl + beta_term(partner, sign, beta, dt, magnetic)
        gate.complex_mask(curl, iyee, boundaries)
        probe.reference_recurrence(
            arrays[target], arrays["fu_" + target], curl,
            coefficients["kms_" + dsig], coefficients["sinv_" + dsig],
            coefficients["kms_" + dsigu], coefficients["sinv_" + dsigu])


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
                                              grid.beta, grid.dt)
                    else:
                        real_beta_pml_step(xp, arrays, curl_tables[slot], dtdx,
                                           slot, kinds, grid.beta, grid.dt)
                else:
                    probe.reference_constitutive_step(
                        side, arrays, arrays, constitutive_tables[side])
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
                  kernel=None, beta_override=None,
                  has_beta: int = 1) -> Dict[str, Any]:
    case: Dict[str, Any] = {
        "family": "real", "shape": list(shape), "config": config["name"],
        "beta": repr(beta), "dtdx": repr(dtdx), "sub_step": sub_step,
        "guard": guard,
    }
    names = FIELD_12 + FU_NAMES
    arrays = _make_real_state(shape, names)
    coefficients = probe.synthetic_coefficients(cp, tuple(shape),
                                                sub_step == "step_B")
    flat = probe.flatten_coefficients(coefficients)
    reference = {name: array.copy() for name, array in arrays.items()}
    real_beta_pml_step(cp, reference, coefficients, dtdx, sub_step,
                       config["boundaries"], beta, BETA_DT)

    plus, minus = special_kz.beta_curl_coefficients(
        beta, BETA_DT, magnetic=(sub_step == "step_B"), complex_storage=False)
    if beta_override is not None:
        # A callable receives the COMBO'S OWN beta (m8 binds the per-combo
        # unrounded pair); a tuple is bound as-is.
        plus, minus = (beta_override(beta) if callable(beta_override)
                       else beta_override)
    codes = [CODE_OF[b] for b in config["boundaries"]]
    plan = special_kz.plan_beta_pml_curl_from_arrays(
        sub_step, arrays, flat, codes, float(dtdx), plus, minus,
        kernel=kernel, has_beta=has_beta)
    plan.run(guard=guard)
    cp.cuda.runtime.deviceSynchronize()

    targets = ("Bx", "By", "Bz") if sub_step == "step_B" else ("Dx", "Dy", "Dz")
    checked = targets + tuple("fu_" + t for t in targets)
    case["verdict"] = combine({name: bit_compare(arrays[name], reference[name])
                               for name in checked})
    return case


def one_complex_case(shape, config, beta, dtdx, sub_step, guard, expansion,
                     kernel=None, magnetic_override=None,
                     has_beta: int = 1) -> Dict[str, Any]:
    case: Dict[str, Any] = {
        "family": "complex", "shape": list(shape), "config": config["name"],
        "beta": repr(beta), "dtdx": repr(dtdx), "sub_step": sub_step,
        "guard": guard,
        "phases": [repr(p) for p in config["phases"]],
    }
    names = FIELD_12 + FU_NAMES
    rng = np.random.default_rng(SEED + 37)
    arrays = {name: cp.asarray(gate._seed_host_complex(tuple(shape), rng))
              for name in names}
    coefficients = probe.synthetic_coefficients(cp, tuple(shape),
                                                sub_step == "step_B")
    flat = probe.flatten_coefficients(coefficients)
    reference = {name: array.copy() for name, array in arrays.items()}
    complex_beta_pml_step(cp, reference, coefficients, dtdx, sub_step,
                          config["boundaries"], config["phases"], beta, BETA_DT)

    magnetic = (sub_step == "step_B") if magnetic_override is None \
        else magnetic_override
    words = special_kz.beta_curl_coefficients(beta, BETA_DT, magnetic=magnetic,
                                              complex_storage=True)
    codes = [CODE_OF[b] for b in config["boundaries"]]
    plan = special_kz.plan_beta_bloch_pml_curl_from_arrays(
        sub_step, arrays, flat, codes, config["phases"], float(dtdx),
        int(expansion), words, kernel=kernel, has_beta=has_beta)
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


def run_synthetic(results: Dict[str, Any], out_path: str, expansion: int,
                  label: str = "gate", guards=(False, True),
                  real_kernel=None, complex_kernel=None,
                  families=("real", "complex"),
                  real_beta_override=None, complex_magnetic_override=None,
                  real_configs=REAL_CONFIGS, complex_configs=COMPLEX_CONFIGS,
                  ) -> Dict[str, Any]:
    cases: List[Dict[str, Any]] = []
    combos: List[Tuple] = []
    if "real" in families:
        combos += [("real", shape, config, beta, dtdx, sub_step)
                   for shape in REAL_SHAPES for config in real_configs
                   for beta in REAL_BETAS for dtdx in REAL_DTDX
                   for sub_step in CURL_SUB_STEPS]
    if "complex" in families:
        combos += [("complex", shape, config, beta, dtdx, sub_step)
                   for shape in COMPLEX_SHAPES for config in complex_configs
                   for beta in COMPLEX_BETAS for dtdx in COMPLEX_DTDX
                   for sub_step in CURL_SUB_STEPS]
    total = len(guards) * len(combos)
    index = 0
    for guard in guards:
        for family, shape, config, beta, dtdx, sub_step in combos:
            index += 1
            started = time.time()
            if family == "real":
                case = one_real_case(shape, config, beta, dtdx, sub_step,
                                     guard, kernel=real_kernel,
                                     beta_override=real_beta_override)
            else:
                case = one_complex_case(shape, config, beta, dtdx, sub_step,
                                        guard, expansion,
                                        kernel=complex_kernel,
                                        magnetic_override=(
                                            (sub_step != "step_B")
                                            if complex_magnetic_override
                                            else None))
            case["seconds"] = round(time.time() - started, 3)
            cases.append(case)
            verdict = case["verdict"]
            log(f"[{label}] {index}/{total} guard={guard} {family} {sub_step} "
                f"{'x'.join(str(n) for n in shape)} {config['name']} "
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

def run_identity(results: Dict[str, Any], out_path: str,
                 expansion: int) -> Dict[str, Any]:
    """HAS_BETA=0 must reproduce the certified kernels byte-for-byte on the
    same seeded state — the beta term is compiled OUT, not merely zeroed."""
    from meep_gpu.triton_kernels import launch  # noqa: PLC0415

    leg: Dict[str, Any] = {"cases": []}
    shape = (7, 6, 1)
    # Real family: beta kernel HAS_BETA=0 vs kernels.pml_curl_step.
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
            has_beta=0).run(guard=False)
        launch.plan_from_arrays(sub_step, twin, flat, codes, 0.34
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
        # HAS_BETA=0 must compile the term OUT, and only nonzero words make an
        # always-on HAS_BETA defect byte-visible — subtracting a
        # zero-coefficient product is byte-neutral on every word that is not
        # itself ±0 (x - (±0) == x), so zero words would prove nothing.
        special_kz.plan_beta_bloch_pml_curl_from_arrays(
            sub_step, arrays, flat, codes, phases, 0.35, expansion,
            ((0.1, -0.2), (-0.3, 0.4)), has_beta=0).run(guard=False)
        complex_fields.plan_complex_pml_curl_from_arrays(
            sub_step, twin, flat, codes, phases, 0.35, expansion
            ).run(guard=False)
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
        fields, pml, "step_B", probe=None).reasons
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
    the header because a @triton.jit body may not read a plain module global."""
    header = (
        "import triton\nimport triton.language as tl\n"
        "PERIODIC = tl.constexpr(0)\nMETALLIC = tl.constexpr(1)\n"
        "NAIVE = tl.constexpr(0)\nFMA_V1 = tl.constexpr(1)\n\n")
    handle = tempfile.NamedTemporaryFile("w", suffix="_mutated_beta.py",
                                         delete=False, encoding="utf-8")
    handle.write(header + source)
    handle.close()
    _TEMPORARY.append(handle.name)
    spec = importlib.util.spec_from_file_location(
        "triton_mutated_beta_" + str(len(_TEMPORARY)), handle.name)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = module  # type: ignore[union-attr]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return getattr(module, kernel_name)


def shipped_source() -> str:
    """Helpers first (they may be the mutation target), then both kernels.
    The complex kernel calls complex_fields' certified helpers, so those are
    included verbatim ahead of it."""
    functions = (complex_fields._rotate_field_left,
                 complex_fields._mul_field_left,
                 complex_fields._mul_coefficient_left,
                 special_kz._mul_imag_coefficient_left,
                 special_kz.beta_pml_curl_step,
                 special_kz.beta_bloch_pml_curl_step)
    return "\n\n".join(textwrap.dedent(inspect.getsource(f.fn))
                       for f in functions)


REAL_TERM_0 = "curl0 = curl0 - (beta_plus * b)"
REAL_TERM_1 = "curl1 = curl1 - (beta_minus * a)"
COMPLEX_TERM_0 = "_mul_imag_coefficient_left(bp_re, bp_im, b_re, b_im, EXPANSION)"
COMPLEX_TERM_1 = "_mul_imag_coefficient_left(bm_re, bm_im, a_re, a_im, EXPANSION)"


def _swap_two(source: str, first: str, second: str) -> Tuple[str, int]:
    if first not in source or second not in source:
        return source, 0
    marker = "__MUTATION_TMP__"
    source = source.replace(first, marker)
    source = source.replace(second, first)
    source = source.replace(marker, second)
    return source, 2


def mutate_m1_sign_swap(source: str) -> Tuple[str, int]:
    """m1: the X/Y target signs swapped — target 0 takes the minus coefficient."""
    hits = 0
    source, n = _swap_two(source, "(beta_plus * b)", "(beta_minus * a)")
    hits += n
    source, n = _swap_two(source, "(bp_re, bp_im, b_re, b_im",
                          "(bm_re, bm_im, a_re, a_im")
    hits += n
    return source, hits


def mutate_m3_dtdx_scaled(source: str) -> Tuple[str, int]:
    """m3: the analytic term wrongly scaled by dtdx (stepping.py:760-762) —
    planted in BOTH families (the complex needles scale the helper's product
    words before the subtraction)."""
    hits = 0
    for needle, replacement in (
            (REAL_TERM_0, "curl0 = curl0 - (dtdx * (beta_plus * b))"),
            (REAL_TERM_1, "curl1 = curl1 - (dtdx * (beta_minus * a))"),
            ("curl0_re = curl0_re - t_re", "curl0_re = curl0_re - (dtdx * t_re)"),
            ("curl0_im = curl0_im - t_im", "curl0_im = curl0_im - (dtdx * t_im)"),
            ("curl1_re = curl1_re - t_re", "curl1_re = curl1_re - (dtdx * t_re)"),
            ("curl1_im = curl1_im - t_im", "curl1_im = curl1_im - (dtdx * t_im)")):
        if needle in source:
            source = source.replace(needle, replacement)
            hits += 1
    return source, hits


def mutate_m4_shifted_partner(source: str) -> Tuple[str, int]:
    """m4: the partner read from an IN-PLANE shifted operand instead of the
    center. The needles aim at ``b_x``/``a_y`` — genuinely different cells on
    every sweep shape — and deliberately NOT at the z-shifted operand: every
    sweep shape has nz = 1 with z PERIODIC, so ``oz == idx`` and a
    ``b_z``/``a_z`` mutant is byte-identical to the true kernel; a needle
    there measures nothing and fails the gate on a correct kernel."""
    hits = 0
    for needle, replacement in (
            ("(beta_plus * b)", "(beta_plus * b_x)"),
            ("(beta_minus * a)", "(beta_minus * a_y)"),
            ("(bp_re, bp_im, b_re, b_im", "(bp_re, bp_im, b_x_re, b_x_im"),
            ("(bm_re, bm_im, a_re, a_im", "(bm_re, bm_im, a_y_re, a_y_im")):
        if needle in source:
            source = source.replace(needle, replacement)
            hits += 1
    return source, hits


def mutate_m5_rotated_partner(source: str) -> Tuple[str, int]:
    """m5: the wrap-phase rotation applied to the beta partner (the partner is
    a CENTER operand and must never be rotated) — armed on a phased config."""
    replacement = (
        "_mul_imag_coefficient_left(bp_re, bp_im, "
        "tl.math.fma(b_re, pxr, -(b_im * pxi)), "
        "tl.math.fma(b_re, pxi, b_im * pxr), EXPANSION)")
    if COMPLEX_TERM_0 not in source:
        return source, 0
    return source.replace(COMPLEX_TERM_0, replacement), 1


def mutate_m6_negation_dropped(source: str) -> Tuple[str, int]:
    """m6: the product ADDED instead of the negated-add (curl + c*g)."""
    hits = 0
    for needle, replacement in (
            (REAL_TERM_0, "curl0 = curl0 + (beta_plus * b)"),
            (REAL_TERM_1, "curl1 = curl1 + (beta_minus * a)"),
            ("curl0_re = curl0_re - t_re", "curl0_re = curl0_re + t_re"),
            ("curl0_im = curl0_im - t_im", "curl0_im = curl0_im + t_im")):
        if needle in source:
            source = source.replace(needle, replacement)
            hits += 1
    return source, hits


COMPLEX_TERM_1_LINE = "curl1_im = curl1_im - t_im"


def mutate_m7_beta_on_target2(source: str) -> Tuple[str, int]:
    """m7: the term applied to Bz/Dz, which step_db.cpp:148-176 never touches
    — planted in BOTH families."""
    hits = 0
    if REAL_TERM_1 in source:
        source = source.replace(
            REAL_TERM_1,
            REAL_TERM_1 + "\n        curl2 = curl2 - (beta_plus * c)")
        hits += 1
    if COMPLEX_TERM_1_LINE in source:
        source = source.replace(
            COMPLEX_TERM_1_LINE,
            COMPLEX_TERM_1_LINE
            + "\n        t2b_re, t2b_im = _mul_imag_coefficient_left("
              "bp_re, bp_im, c_re, c_im, EXPANSION)"
            + "\n        curl2_re = curl2_re - t2b_re"
            + "\n        curl2_im = curl2_im - t2b_im")
        hits += 1
    return source, hits


def mutate_m8_f64_coefficient(source: str) -> Tuple[str, int]:
    """m8: per-element f64 coefficient instead of host-rounded-once.

    The original arming route — re-typing the scalar arguments
    ``tl.float64`` — is DEAD on Triton 3.1.0: ``jit.py``'s ``annotation_type``
    honors int/uint/bool annotations only, so a Python float argument is
    always fp32 at the ABI. The unrounded coefficient was therefore rounded to
    f32 at the call boundary, and ``round_f32(f64(c32) * f64(b))`` of an
    ALREADY-ROUNDED coefficient is the single rounding of the exact product —
    byte-identical to the true kernel (job 2332: 72/72 identical launches; a
    false needle of the m4-z-shift class). The rework smuggles the UNROUNDED
    coefficient through the two existing fp32 slots as a hi/lo pair — the leg
    binds ``beta_plus = hi = f32(c)``, ``beta_minus = lo = f32(c - hi)``, the
    kernel rebuilds ``c`` in f64 to ~49 bits (the true minus coefficient is
    the exact negation of plus, S:770 with sign=-1) — so the per-element
    product really is taken against the unrounded value and every combo's
    catch isolates the double rounding."""
    hits = 0
    for needle, replacement in (
            (REAL_TERM_0,
             "curl0 = curl0 - (((beta_plus.to(tl.float64) + "
             "beta_minus.to(tl.float64)) * b.to(tl.float64))"
             ".to(tl.float32))"),
            (REAL_TERM_1,
             "curl1 = curl1 - (((0.0 - (beta_plus.to(tl.float64) + "
             "beta_minus.to(tl.float64))) * a.to(tl.float64))"
             ".to(tl.float32))")):
        if needle in source:
            source = source.replace(needle, replacement)
            hits += 1
    return source, hits


def mutate_m9_zero_cross_folded(source: str) -> Tuple[str, int]:
    """m9: the signed-zero c_re cross terms folded away — exactly the
    constant-fold the compiler must not perform. NOT in the sweep table (the
    fold is byte-neutral on random seeds). :func:`run_m9_product_and_stored`
    catches this mutant at the PRODUCT layer — a wrapper kernel storing the
    helper's own output words (the byte-visible layer, job 2329's lesson) —
    and ALSO at the engineered stored-layer launch, whose expectation was
    restored 2026-08-12 to the original IEEE-negation prediction after the
    inlined complex_fields helpers dropped their unary-minus addends (job
    2332's predicted-4-got-0 stands as the record of the old spelling, whose
    ``0.0 - x`` lowering canonicalized every ±0 addend before a store)."""
    hits = 0
    for needle, replacement in (
            ("tl.math.fma(c_re, z_re, (c_im * z_im) * -1.0)",
             "(c_im * z_im) * -1.0"),
            ("tl.math.fma(c_re, z_im, c_im * z_re)", "(c_im * z_re)"),
            ("(c_re * z_re) - (c_im * z_im)", "0.0 - (c_im * z_im)"),
            ("(c_re * z_im) + (c_im * z_re)", "0.0 + (c_im * z_re)")):
        if needle in source:
            source = source.replace(needle, replacement)
            hits += 1
    return source, hits


#: name -> (transform, families the catch is demanded in — EACH must catch).
SOURCE_MUTATIONS = {
    "m1_sign_swap": (mutate_m1_sign_swap, ("real", "complex")),
    "m3_term_scaled_by_dtdx": (mutate_m3_dtdx_scaled, ("real", "complex")),
    "m4_partner_from_shifted_operand": (mutate_m4_shifted_partner,
                                        ("real", "complex")),
    "m5_wrap_rotation_on_partner": (mutate_m5_rotated_partner, ("complex",)),
    "m6_negation_dropped": (mutate_m6_negation_dropped, ("real", "complex")),
    "m7_beta_on_target2": (mutate_m7_beta_on_target2, ("real", "complex")),
}

#: m5 only diverges where a phase actually rotates, so its complex sweep runs
#: the phased configs only. m9 is NOT in the table above: the signed-zero fold
#: is byte-neutral on the random-seeded sweeps (see run_m9_product_and_stored), so it
#: runs as a dedicated engineered-state leg instead.
_PHASED_COMPLEX = tuple(c for c in COMPLEX_CONFIGS
                        if c["name"] != "k0_unphased")


class CountingKernel(gate.CountingKernel):
    pass


# ---------------------------------------------------------------------------
# m9 — product-layer catch + the stored-layer engineered leg
# ---------------------------------------------------------------------------
#
# The fold's only effect is a ±0 sign flip in the helper's output words. The
# ORIGINAL plan — an engineered coincidence state (a ±0 partner word, an
# exactly ``-0`` curl word, an exactly ``-0`` fu word, composed so the flip
# reaches ``n = ((fu*km) - curl)*sinv``; NumPy emulation under IEEE negation
# predicted exactly 4 differing stored words per arm) — was measured by job
# 2332 at ZERO differing stored words. Mechanism, verified against the
# installed Triton 3.1.0: unary minus lowers as ``0.0 - x``
# (language/semantic.py:386-391), and the INLINED complex_fields helpers'
# addends were spelled unary ``-`` at the time (the spelling carried until
# 2026-08-12), turning every ±0 addend into ``+0`` — an exactly ``-0`` curl
# word could not even be FORMED from zero operands on device ( ``(dtdx * -0)
# + (0-(0*t_im)) = -0 + +0 = +0`` ), and any surviving ±0 difference was
# laundered again at the next helper multiply before a store. That measured
# absorption moved the catch to the layer where the defect IS byte-visible
# (job 2329's lesson): a wrapper kernel that stores
# ``_mul_imag_coefficient_left``'s own output words.
#
# The wrapper leg doubles as the helper's FAITHFULNESS pin: the true
# kernel's product words must equal the licensed arm's exact-fma32 candidate
# bytes (``gate._candidates_c8`` with the coefficient as the FIRST operand)
# on the full signed-zero/subnormal vector set. That pin is what makes the
# module's ``* -1.0`` negation spelling load-bearing: the unary-minus
# spelling fails it on the ``c_re·z_re = +0``-with-``-0``-addend rows.
#
# 2026-08-12: the complex tranche's zero-init composition reachability
# analysis showed the array path stores ``-0.0`` words the canonicalized
# helpers cannot reproduce, and complex_fields' three inlined helpers
# (_rotate_field_left / _mul_field_left / _mul_coefficient_left) now carry
# the ``* -1.0`` spelling too. Both absorption mechanisms above were exactly
# those three addends, so the engineered state's flip once again propagates
# sign-exactly to the stores and the stored-layer expectation REVERTS to the
# original IEEE-negation prediction (4 words per arm). The engineered leg is
# KEPT and launched with that restored expectation; a count off it means
# either the platform lowering or the helper spelling changed — fail and
# re-derive. Job 2332's 0-word measurement remains the record of the OLD
# spelling.

M9_SHAPE = (4, 3, 4)  # nz > 1 ON PURPOSE: the -0 curl addend needs b_z != b
                      # (kernel-vs-kernel probe; no 2-D semantics claimed).
#: ±0 real word, nonzero imaginary — the WORD CLASS the engine binds (the
#: minus side's real word is -0), so the launch runs the same ±0-real
#: scalar-argument shape production binds; a nonzero real word would compile
#: and launch a different value class than the one the fold hazard lives in.
M9_WORDS = ((-0.0, 0.7), (-0.0, 0.7))
#: Stored-layer expectation PER LICENSED ARM — the original IEEE-negation
#: host prediction (4 words per arm), RESTORED 2026-08-12: the complex
#: helpers' ``* -1.0`` spelling fix removed both absorption mechanisms that
#: job 2332 measured under the old unary-minus spelling (predicted-4-got-0,
#: kept above as the record of that spelling). FMA_V1: 4 is the host
#: prediction pending the device recut — the leg fails loudly if the device
#: disagrees, which would mean the lowering or the helper spelling moved
#: again. NAIVE: unchanged host prediction (this platform licenses FMA_V1,
#: so the NAIVE row remains a prediction, not a device record).
M9_STORED_EXPECTED = {"FMA_V1": 4, "NAIVE": 4}

M9_WRAPPER_SRC = '''

@triton.jit
def beta_product_probe(z, out, n_elem,
                       c_re, c_im,
                       EXPANSION: tl.constexpr, BLOCK: tl.constexpr):
    """Stores _mul_imag_coefficient_left's own words — the byte-visible
    layer for the m9 fold on this platform."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    live = offs < n_elem
    z_re = tl.load(z + 2 * offs, mask=live, other=0.0)
    z_im = tl.load(z + 2 * offs + 1, mask=live, other=0.0)
    out_re, out_im = _mul_imag_coefficient_left(c_re, c_im, z_re, z_im,
                                                EXPANSION)
    tl.store(out + 2 * offs, out_re, mask=live)
    tl.store(out + 2 * offs + 1, out_im, mask=live)
'''


def _m9_engineered_state() -> Dict[str, Any]:
    """The coincidence state, host side — every word an exact signed zero."""
    state = {name: np.zeros(M9_SHAPE, dtype=np.complex64)
             for name in FIELD_12 + FU_NAMES}
    state["Ey"].real[:, :, 0::2] = -0.0   # b center -0 at even k, b_z +0
    state["Ex"].real[:, :, 1::2] = -0.0   # a center +0 at even k, a_z -0
    state["Ez"].real[0::2, 1::2, :] = -0.0  # -0 at (i even, j odd): c_y for
    #                                         target 0, center for target 1
    state["fu_Bx"].real[...] = -0.0       # the q-chains' -0 (im stays +0)
    state["fu_By"].real[...] = -0.0
    return state


def _interleave_words(re: np.ndarray, im: np.ndarray) -> np.ndarray:
    out = np.empty(2 * re.size, np.float32)
    out[0::2] = re
    out[1::2] = im
    return out


def _m9_fold_emulation(arm: str, c_im32: np.float32, zr: np.ndarray,
                       zi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """NumPy bytes of the FOLDED helper, per arm — f32 single rounding via
    gate.mul32; the FMA arm's ``* -1.0`` is a sign-exact negation, the NAIVE
    arm's literal ``0.0 -``/``0.0 +`` are binary ops that launder ±0, and the
    emulation reproduces each spelling exactly."""
    n = zr.size
    out_re = np.empty(n, np.float32)
    out_im = np.empty(n, np.float32)
    zero = np.float32(0.0)
    for i in range(n):
        p = gate.mul32(c_im32, zi[i])
        t = gate.mul32(c_im32, zr[i])
        if arm == "FMA_V1":
            out_re[i] = np.float32(-float(p))
            out_im[i] = t
        else:
            out_re[i] = np.float32(zero - p)
            out_im[i] = np.float32(zero + t)
    return out_re, out_im


def run_m9_product_and_stored(expansion: int) -> Dict[str, Any]:
    """m9 at the product layer (armed, launch-counted, predicted per word) +
    the stored-layer engineered launch against its restored IEEE prediction.

    Product leg, per coefficient word class: the true wrapper must reproduce
    the licensed arm's exact-fma32 candidate bytes (the faithfulness pin —
    this is where the module's ``* -1.0`` negation is load-bearing), the
    mutant wrapper must reproduce the fold's own NumPy emulation, and the two
    must differ in EXACTLY the predicted words, with a nonzero total.

    Stored leg: the job-2332 engineered state, true vs mutant full kernel,
    expected exactly M9_STORED_EXPECTED[licensed arm] differing stored words
    (4 per arm — the original IEEE-negation host prediction, restored
    2026-08-12 when the inlined complex_fields helpers dropped their
    unary-minus addends; job 2332 measured 0 under the old spelling's
    canonicalization), re-asserted so a Triton or helper-spelling change
    fails loudly here instead of silently shifting the byte contract."""
    shipped = shipped_source()
    mutated, hits = mutate_m9_zero_cross_folded(shipped)
    if hits < 4:
        return {"error": f"NEEDLE MISSED ({hits}/4 sites)"}
    arm = "FMA_V1" if int(expansion) == 1 else "NAIVE"  # header codes :963

    entry: Dict[str, Any] = {"sites": hits, "families": ["complex"],
                             "licensed_arm": arm}

    # --- product layer -----------------------------------------------------
    from meep_gpu.triton_kernels.launch import CupyPointer  # noqa: PLC0415

    rng = np.random.default_rng(SEED + 141)
    zr, zi = _beta_pattern_vectors(rng)
    z_dev = cp.asarray(gate._interleave_c8(zr, zi)).view(cp.float32)
    n = int(zr.size)
    block = 256
    grid = ((n + block - 1) // block,)
    true_wrapper = compile_mutated(shipped + M9_WRAPPER_SRC,
                                   "beta_product_probe")
    # The mutant wrapper gets its OWN function name: an entry point whose
    # source text differs from the true wrapper's in every byte-addressed
    # cache and registry, so no same-name aliasing at any layer can serve the
    # true binary for the mutant launch (job 2334 measured mutant bytes ==
    # true bytes while the two PTX provably differ — diagnostics below).
    mut_wrapper_src = (M9_WRAPPER_SRC
                       .replace("beta_product_probe", "beta_product_probe_m9fold"))
    counter = CountingKernel(compile_mutated(mutated + mut_wrapper_src,
                                             "beta_product_probe_m9fold"))
    helper_true_src = inspect.getsource(
        true_wrapper.fn.__globals__["_mul_imag_coefficient_left"].fn)
    helper_mut_src = inspect.getsource(
        counter.kernel.fn.__globals__["_mul_imag_coefficient_left"].fn)
    diagnostics: Dict[str, Any] = {
        "same_jit_object": true_wrapper is counter.kernel,
        "true_cache_key": true_wrapper.cache_key[:16],
        "mut_cache_key": counter.kernel.cache_key[:16],
        "true_helper_has_fma": "tl.math.fma(c_re, z_re" in helper_true_src,
        "mut_helper_has_fma": "tl.math.fma(c_re, z_re" in helper_mut_src,
    }
    log(f"[mut] m9 diagnostics pre-launch: {diagnostics}")
    rows: List[Dict[str, Any]] = []
    total_differing = 0
    total_predicted = 0
    product_ok = True
    for scalar0d in _beta_coefficient_scalars():
        c_re32 = np.float32(scalar0d.real)
        c_im32 = np.float32(scalar0d.imag)
        out_true = cp.full((2 * n,), np.float32(3.0), dtype=cp.float32)
        out_mut = cp.full((2 * n,), np.float32(3.0), dtype=cp.float32)
        true_wrapper[grid](CupyPointer(z_dev), CupyPointer(out_true), n,
                           float(c_re32), float(c_im32),
                           EXPANSION=int(expansion), BLOCK=block)
        counter[grid](CupyPointer(z_dev), CupyPointer(out_mut), n,
                      float(c_re32), float(c_im32),
                      EXPANSION=int(expansion), BLOCK=block)
        cp.cuda.runtime.deviceSynchronize()
        host_true = probe.to_host(out_true)
        host_mut = probe.to_host(out_mut)
        cand_re, cand_im = gate._candidates_c8(
            np.full(n, c_re32, np.float32), np.full(n, c_im32, np.float32),
            zr, zi)[arm]
        expected_true = _interleave_words(cand_re, cand_im)
        fold_re, fold_im = _m9_fold_emulation(arm, c_im32, zr, zi)
        expected_mut = _interleave_words(fold_re, fold_im)
        u = np.uint32
        differing = int(np.count_nonzero(host_true.view(u)
                                         != host_mut.view(u)))
        predicted = int(np.count_nonzero(expected_true.view(u)
                                         != expected_mut.view(u)))
        row = {
            "coefficient": repr(complex(scalar0d[()])),
            "differing_words": differing,
            "predicted_words": predicted,
            "true_matches_candidate": bool(
                np.array_equal(host_true.view(u), expected_true.view(u))),
            "mutant_matches_fold_emulation": bool(
                np.array_equal(host_mut.view(u), expected_mut.view(u))),
        }
        if predicted and "poison_rows" not in diagnostics:
            u32 = np.uint32
            indices = np.nonzero(expected_true.view(u32)
                                 != expected_mut.view(u32))[0][:4]
            diagnostics["poison_rows"] = [
                {"word_index": int(i),
                 "z": (repr(float(zr[i // 2])), repr(float(zi[i // 2]))),
                 "candidate": hex(int(expected_true.view(u32)[i])),
                 "fold_emulation": hex(int(expected_mut.view(u32)[i])),
                 "device_true": hex(int(host_true.view(u32)[i])),
                 "device_mutant": hex(int(host_mut.view(u32)[i]))}
                for i in indices]
        rows.append(row)
        total_differing += differing
        total_predicted += predicted
        product_ok = (product_ok and row["true_matches_candidate"]
                      and row["mutant_matches_fold_emulation"]
                      and differing == predicted)

    def _ptx_fma_counts(jitfn) -> List[int]:
        counts: List[int] = []
        try:
            for per_device in jitfn.cache.values():
                for compiled in per_device.values():
                    asm = getattr(compiled, "asm", {})
                    ptx = asm.get("ptx", "") if hasattr(asm, "get") else ""
                    counts.append(int(ptx.count("fma.rn.f32")))
        except Exception as error:  # noqa: BLE001 — diagnostics must not kill the leg
            counts.append(-1)
            diagnostics["ptx_probe_error"] = repr(error)
        return counts

    diagnostics["true_ptx_fma_counts"] = _ptx_fma_counts(true_wrapper)
    diagnostics["mut_ptx_fma_counts"] = _ptx_fma_counts(counter.kernel)
    log(f"[mut] m9 diagnostics post-launch: "
        f"true_ptx_fma={diagnostics['true_ptx_fma_counts']} "
        f"mut_ptx_fma={diagnostics['mut_ptx_fma_counts']} "
        f"poison={diagnostics.get('poison_rows')}")
    entry["product_layer"] = {
        "vectors": n, "coefficients": len(rows), "rows": rows,
        "total_differing_words": total_differing,
        "total_predicted_words": total_predicted,
        "launches": counter.launches,
        "diagnostics": diagnostics,
        "faithfulness_pin": "true wrapper bytes == licensed-arm exact-fma32 "
                            "candidate on every vector and coefficient",
    }
    product_caught = (counter.launches > 0 and total_differing > 0
                      and product_ok)
    log(f"[mut] m9 product layer: arm={arm} launches={counter.launches} "
        f"differing={total_differing} (predicted {total_predicted}) "
        f"faithful={product_ok} CAUGHT={product_caught}")

    # --- stored layer (restored IEEE prediction, 2026-08-12) ----------------
    stored_counter = CountingKernel(compile_mutated(
        mutated, "beta_bloch_pml_curl_step"))
    host = _m9_engineered_state()
    names = FIELD_12 + FU_NAMES
    true_arrays = {name: cp.asarray(host[name]) for name in names}
    mutant_arrays = {name: array.copy() for name, array in true_arrays.items()}
    coefficients = probe.synthetic_coefficients(cp, M9_SHAPE, True)
    flat = probe.flatten_coefficients(coefficients)
    special_kz.plan_beta_bloch_pml_curl_from_arrays(
        "step_B", true_arrays, flat, (0, 0, 0), (None, None, None), 0.35,
        int(expansion), M9_WORDS).run(guard=False)
    special_kz.plan_beta_bloch_pml_curl_from_arrays(
        "step_B", mutant_arrays, flat, (0, 0, 0), (None, None, None), 0.35,
        int(expansion), M9_WORDS, kernel=stored_counter).run(guard=False)
    cp.cuda.runtime.deviceSynchronize()
    checked = ("Bx", "By", "Bz", "fu_Bx", "fu_By", "fu_Bz")
    verdict = combine({name: bit_compare(mutant_arrays[name],
                                         true_arrays[name])
                       for name in checked})
    stored_expected = M9_STORED_EXPECTED[arm]
    stored_ok = (verdict["differing_floats"] == stored_expected
                 and stored_counter.launches > 0)
    entry["stored_layer"] = {
        "expectation_basis": "restored IEEE-negation host prediction "
                             "(2026-08-12): the inlined complex_fields "
                             "helpers now spell their addends * -1.0, so "
                             "the fold's ±0 flip propagates to the stores; "
                             "job 2332's 0-word absorption stands as the "
                             "record of the old unary-minus spelling",
        "launches": stored_counter.launches,
        "differing_words": verdict["differing_floats"],
        "expected_words": stored_expected,
        "as_expected": stored_ok,
    }
    log(f"[mut] m9 stored layer (restored IEEE prediction): "
        f"launches={stored_counter.launches} "
        f"differing={verdict['differing_floats']} "
        f"(expected {stored_expected}) as_expected={stored_ok}")

    entry["launches"] = counter.launches + stored_counter.launches
    entry["caught"] = product_caught and stored_ok
    if counter.launches == 0 or stored_counter.launches == 0:
        entry["error"] = "DISARMED: a mutated kernel never launched"
    return entry


def run_mutations(results: Dict[str, Any], out_path: str,
                  expansion: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    shipped = shipped_source()

    for name, (transform, families) in SOURCE_MUTATIONS.items():
        mutated, hits = transform(shipped)
        if hits == 0:
            out[name] = {"error": "NEEDLE MISSED: the mutation matched "
                                  "nothing; the needle drifted off the kernel"}
            log(f"[mut] {name}: NEEDLE MISSED — nothing exercised")
            results["mutations"] = out
            save(results, out_path)
            continue
        entry: Dict[str, Any] = {"sites": hits, "families": list(families)}
        caught_per_family: Dict[str, bool] = {}
        launches = 0
        if "real" in families:
            counter = CountingKernel(compile_mutated(mutated,
                                                     "beta_pml_curl_step"))
            summary = run_synthetic(results, out_path, expansion,
                                    label="mutation:" + name + ":real",
                                    guards=(False,), real_kernel=counter,
                                    families=("real",))
            entry["real"] = {"ran": summary["ran"],
                             "identical": summary["identical"]}
            caught_per_family["real"] = summary["identical"] < summary["ran"]
            launches += counter.launches
        if "complex" in families:
            counter = CountingKernel(compile_mutated(
                mutated, "beta_bloch_pml_curl_step"))
            configs = (_PHASED_COMPLEX
                       if name == "m5_wrap_rotation_on_partner"
                       else COMPLEX_CONFIGS)
            summary = run_synthetic(results, out_path, expansion,
                                    label="mutation:" + name + ":complex",
                                    guards=(False,), complex_kernel=counter,
                                    families=("complex",),
                                    complex_configs=configs)
            entry["complex"] = {"ran": summary["ran"],
                                "identical": summary["identical"]}
            caught_per_family["complex"] = summary["identical"] < summary["ran"]
            launches += counter.launches
        entry["launches"] = launches
        entry["caught_per_family"] = caught_per_family
        # EACH armed family must catch on its own: an OR over families would
        # let a family-specific miss hide behind the other family's catch.
        entry["caught"] = (bool(caught_per_family)
                           and all(caught_per_family.values()))
        if launches == 0:
            entry["error"] = ("DISARMED: the mutated kernel never launched — "
                              "the leg measured nothing and FAILS")
        log(f"[mut] {name}: sites={hits} launches={launches} "
            f"caught_per_family={caught_per_family} CAUGHT={entry['caught']}")
        out[name] = entry
        results["mutations"] = out
        save(results, out_path)

    # m2 (host): the ±i swapped between the B and D sides — the plan built
    # with the WRONG magnetic flag, so every complex case runs the other
    # side's coefficient.
    summary = run_synthetic(results, out_path, expansion,
                            label="mutation:m2_pm_i_swapped", guards=(False,),
                            families=("complex",),
                            complex_magnetic_override=True)
    out["m2_pm_i_swapped"] = {
        "host": True, "ran": summary["ran"],
        "identical": summary["identical"],
        "caught": summary["identical"] < summary["ran"]}
    log(f"[mut] m2_pm_i_swapped: identical={summary['identical']}/"
        f"{summary['ran']}")
    results["mutations"] = out
    save(results, out_path)

    # m8 (source + host): the hi/lo mutant launched with the UNROUNDED
    # coefficient split across the two fp32 slots (the f64-annotation route is
    # dead on Triton 3.1.0 — see mutate_m8_f64_coefficient). The pair is built
    # PER COMBO from the combo's OWN beta: a fixed BETA_KZ2D pair would make
    # every BETA_GRATING combo diverge for the wrong reason (a wrong-beta
    # constant, not the double-rounding m8 plants), diluting the catch's
    # attribution to the defect it claims. hi IS the reference's own rounded
    # coefficient, so any divergence is the double rounding and nothing else.
    mutated, hits = mutate_m8_f64_coefficient(shipped)
    if hits < 2:
        out["m8_f64_coefficient"] = {"error": f"NEEDLE MISSED ({hits}/2 sites)"}
    else:
        counter = CountingKernel(compile_mutated(mutated, "beta_pml_curl_step"))

        def hilo(beta: float) -> Tuple[float, float]:
            coefficient = 2.0 * math.pi * beta * BETA_DT
            hi = np.float32(coefficient)
            lo = np.float32(coefficient - float(hi))
            return (float(hi), float(lo))

        summary = run_synthetic(results, out_path, expansion,
                                label="mutation:m8_f64_coefficient",
                                guards=(False,), real_kernel=counter,
                                families=("real",),
                                real_beta_override=hilo)
        out["m8_f64_coefficient"] = {
            "sites": hits, "launches": counter.launches,
            "ran": summary["ran"], "identical": summary["identical"],
            "per_combo_unrounded": "the mutant binds each combo's own beta "
                                   "unrounded, so a catch isolates the "
                                   "double-rounding defect on every combo",
            "families_note": "real-only by design: the complex product's "
                             "rounding discipline is held by the measured "
                             "EXPANSION probe contract; a per-element f64 "
                             "complex emulation would replace the helper "
                             "wholesale rather than plant a single defect",
            "caught": summary["identical"] < summary["ran"]}
        if counter.launches == 0:
            out["m8_f64_coefficient"]["error"] = "DISARMED: never launched"
        log(f"[mut] m8_f64_coefficient: launches={counter.launches} "
            f"identical={summary['identical']}/{summary['ran']}")
    results["mutations"] = out
    save(results, out_path)

    # m9 (source, dedicated leg): byte-neutral on the random-seeded sweeps, so
    # it is caught at the product layer, and — since the 2026-08-12 complex-
    # helper spelling fix removed the stored-layer absorption job 2332
    # measured — at the engineered stored leg's restored IEEE prediction.
    out["m9_zero_cross_terms_folded"] = run_m9_product_and_stored(expansion)
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
    complex_beta_pml_step(cp, reference, coefficients, 0.35, "step_B",
                          (PERIODIC, PERIODIC, PERIODIC), (None, None, None),
                          0.2, BETA_DT)
    word_views = {name: arrays[name].view(cp.float32) for name in names}
    word_coefficients = dict(coefficients)
    word_coefficients["kms_z"] = cp.ascontiguousarray(
        cp.repeat(coefficients["kms_z"].reshape(-1), 2)).reshape(1, 1, 2)
    word_coefficients["sinv_z"] = cp.ascontiguousarray(
        cp.repeat(coefficients["sinv_z"].reshape(-1), 2)).reshape(1, 1, 2)
    flat = probe.flatten_coefficients(word_coefficients)
    plus, minus = special_kz.beta_curl_coefficients(0.2, BETA_DT, True, False)
    plan = special_kz.plan_beta_pml_curl_from_arrays(
        "step_B", word_views, flat, (0, 0, 0), 0.35, plus, minus)
    plan.run(guard=False)
    cp.cuda.runtime.deviceSynchronize()
    checked = ("Bx", "By", "Bz", "fu_Bx", "fu_By", "fu_Bz")
    verdict = combine({name: bit_compare(arrays[name], reference[name])
                       for name in checked})
    out["m10_real_kernel_on_complex_storage"] = {
        "host": True, "ran": 1, "identical": int(verdict["bit_identical"]),
        "caught": not verdict["bit_identical"]}
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


def run_engine(results: Dict[str, Any], out_path: str,
               record: Dict[str, Any]) -> Dict[str, Any]:
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
                    fields, pml, "step_B", probe=record),
                "update_H": special_kz.plan_beta_run_complex_constitutive(
                    fields, pml, "H", probe=record),
                "step_D": special_kz.plan_beta_bloch_pml_curl(
                    fields, pml, "step_D", probe=record),
                "update_E": special_kz.plan_beta_run_complex_constitutive(
                    fields, pml, "E", probe=record),
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

DEFAULT_LEGS = "expansion,reference,refusals,synthetic,identity,mutations,engine"
DEVICE_LEGS = ("synthetic", "identity", "mutations", "engine")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--legs", default=DEFAULT_LEGS)
    parser.add_argument("--probe-artifact", default=None,
                        help="where to write the expansion probe JSON "
                             "(default: probe.json beside --out)")
    args = parser.parse_args(argv)
    legs = tuple(name.strip() for name in args.legs.split(",") if name.strip())
    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    probe_artifact = args.probe_artifact or os.path.join(out_dir, "probe.json")

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
    record: Optional[Dict[str, Any]] = None
    expansion: Optional[int] = None

    if "expansion" in legs:
        record = run_expansion(results, args.out, probe_artifact)
        if cp is not None and record is None:
            failures.append("expansion leg refused: "
                            + str(results["expansion"].get("refusal")))
        elif record is not None:
            expansion = special_kz.beta_expansion_from_probe(record)
            if expansion is None:
                failures.append("expansion probe refused: no single constexpr "
                                "is licensed over the extended pattern set")
    elif device_available:
        record = complex_fields.load_expansion_probe(probe_artifact)
        policy_problems = probe_record_policy_reasons(record)
        if policy_problems:
            failures.append("probe artifact policy: " + "; ".join(policy_problems))
            record = None
        expansion = special_kz.beta_expansion_from_probe(record)

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
        if expansion is None:
            reason = ("the expansion probe licensed no constexpr; the complex "
                      "kernel may not be launched against an unmeasured "
                      "platform")
            results["skipped_legs"][leg_name] = reason
            log(f"[{leg_name}] REFUSED: {reason}")
            save(results, args.out)
            continue
        if leg_name == "synthetic":
            summary = run_synthetic(results, args.out, expansion)
            if not summary["guarded_pass"]:
                failures.append(f"synthetic sweep: "
                                f"{summary['guarded_identical']}/"
                                f"{summary['guarded_ran']} guarded identical")
            log(f"[SUMMARY] synthetic guarded "
                f"{summary['guarded_identical']}/{summary['guarded_ran']}")
        elif leg_name == "identity":
            leg = run_identity(results, args.out, expansion)
            if leg["identical"] != leg["ran"]:
                failures.append(f"identity leg (R5): {leg['identical']}/"
                                f"{leg['ran']} HAS_BETA=0 cases identical")
        elif leg_name == "mutations":
            out = run_mutations(results, args.out, expansion)
            for name, entry in out.items():
                if name == "m11_predicate_mutations":
                    continue
                if entry.get("error"):
                    failures.append(f"mutation {name}: {entry['error']}")
                elif not entry.get("caught"):
                    failures.append(f"mutation {name} was NOT caught")
        elif leg_name == "engine":
            summary = run_engine(results, args.out, record)
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
        "device_legs_ran": device_available and expansion is not None,
        "certified_under_subnormal_policy":
            results["subnormal_policy"].get("policy"),
    }
    save(results, args.out)
    log(f"[done] {args.out} status={results['summary']['status']} "
        f"failures={failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
