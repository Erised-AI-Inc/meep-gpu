"""Which complex-multiply arm does the REFERENCE take on this host?

THE QUESTION IS INVERTED RELATIVE TO THE TRITON TRACK, and that inversion is the
whole reason this probe exists in its own file.

On Triton the device's lowering was the unknown: the kernel was written twice
(``NAIVE`` and ``FMA_V1``) and the probe asked which one CuPy's compiled dispatch
reproduced. On Metal the shader spells exactly what it is told to, under
``#pragma clang fp contract(off)`` — four rounded products and two rounded adds if
that is what is written, an explicit ``fma`` if that is what is written (the
directive forbids IMPLICIT contraction only, measured). So the open question is
about the ORACLE: the claim is byte-identity to ``stepping.py``, ``stepping.py``
runs NumPy on this host, and what NumPy's complex64 multiply does is a property of
the installed build.

MEASURED HERE (numpy 2.4.3, arm64, 2026-08-15) and the answer is not the obvious
one:

    numpy's complex64 product is the FUSED form, and the fusion is
    ORIENTATION-SENSITIVE. For ``left * right`` it computes

        re = fma(left_re, right_re, -(left_im * right_im))
        im = fma(left_re, right_im,  (left_im * right_re))

    — 0/16,384 differing words — while the four-rounded-product form misses
    thousands. Which factor is FUSED depends on which operand the array path
    writes on the LEFT, which is why the call-site orientation table below is
    normative and every orientation is probed separately rather than once.

THE FIFTH PATTERN IS THIS TRANCHE'S OWN. ``stepping.py:811`` multiplies a complex64
COEFFICIENT — real word a SIGNED ZERO from Python's own complex multiply, imaginary
word ``±2*pi*beta*dt`` — by a field array, coefficient on the LEFT. None of the base
four covers a general scalar-left coefficient, so an artifact cut before this
tranche does not license this family and ``special_kz.beta_expansion_from_probe``
refuses it BY NAME.

``AMBIGUOUS_BOTH`` is expected on that fifth pattern and is ACCEPTED there and
nowhere else, by arithmetic rather than leniency: with ``c_re`` exactly ±0 the
fused arm's extra product is EXACT, so both arms produce identical bytes and the
pattern cannot prefer an arm. It still gates ``NEITHER``.

WHAT THIS PROBE DOES NOT MEASURE, stated: the subnormal band. Every vector here is
seeded in the normal range, because the claim this artifact licenses is itself made
under a checked subnormal-free precondition (:mod:`meep_gpu.metal_kernels.preconditions`).
A probe that mixed the two questions would answer neither.

    python -u probe_metal_beta_expansion.py --out results/metal_special_kz/expansion.json
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
API_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
if API_ROOT not in sys.path:
    sys.path.insert(0, API_ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import metal_gate_kit as kit  # noqa: E402

from meep_gpu.metal_kernels import special_kz  # noqa: E402

SEED = 20260815
VECTOR_CELLS = 8192


def _words(array: Any) -> Any:
    return np.ascontiguousarray(array, dtype=np.float32).reshape(-1).view(np.uint32)


def _fma(a: Any, b: Any, addend: Any) -> Any:
    """An exact-enough f32 fma: the product is EXACT in float64 (24+24 <= 53 bits).

    The subsequent add rounds twice (to float64, then to float32), which can in
    principle differ from a true fma. It is used only to CLASSIFY, and the
    classification is corroborated by the device: the shipped shader's ``fma()``
    reproduces the reference on the same vectors, which is the measurement that
    actually licenses the arm.
    """
    return (np.float64(a) * np.float64(b) + np.float64(addend)).astype(np.float32)


def _arms(left_re, left_im, right_re, right_im) -> Dict[str, Tuple[Any, Any]]:
    """The two transcription arms, with LEFT and RIGHT named rather than assumed."""
    p_rr = np.float32(left_re * right_re)
    p_ii = np.float32(left_im * right_im)
    p_ri = np.float32(left_re * right_im)
    p_ir = np.float32(left_im * right_re)
    return {
        "NAIVE": (np.float32(p_rr - p_ii), np.float32(p_ri + p_ir)),
        "FMA_V1": (_fma(left_re, right_re, -p_ii.astype(np.float64)),
                   _fma(left_re, right_im, p_ir.astype(np.float64))),
    }


def _classify(product: Any, left_re, left_im, right_re, right_im
              ) -> Tuple[str, Dict[str, Any]]:
    got = np.asarray(product, dtype=np.complex64)
    detail: Dict[str, Any] = {}
    misses: Dict[str, int] = {}
    for arm, (arm_re, arm_im) in _arms(left_re, left_im, right_re, right_im).items():
        bad = ((_words(got.real) != _words(arm_re))
               | (_words(got.imag) != _words(arm_im)))
        misses[arm] = int(bad.sum())
        index = np.flatnonzero(bad)[:3]
        detail[arm] = {
            "differing_words": misses[arm],
            "example_cells": [
                {"index": int(i),
                 "reference": [float(got.real.reshape(-1)[i]),
                               float(got.imag.reshape(-1)[i])],
                 "arm": [float(np.asarray(arm_re).reshape(-1)[i]),
                         float(np.asarray(arm_im).reshape(-1)[i])]}
                for i in index],
        }
    if misses["NAIVE"] == 0 and misses["FMA_V1"] == 0:
        return "AMBIGUOUS_BOTH", detail
    if misses["NAIVE"] == 0:
        return "NAIVE", detail
    if misses["FMA_V1"] == 0:
        return "FMA_V1", detail
    return "NEITHER", detail


def _vectors(rng) -> Tuple[Any, Any]:
    """Random normals plus an exhaustive signed-zero corner block.

    NO SUBNORMALS: the claim this artifact licenses is made under a checked
    subnormal-free precondition, and mixing the two questions would answer
    neither. The signed-zero block is exhaustive over the four (re, im) sign
    combinations because random data never produces one.
    """
    real = rng.standard_normal(VECTOR_CELLS).astype(np.float32)
    imag = rng.standard_normal(VECTOR_CELLS).astype(np.float32)
    corners = np.float32([0.0, -0.0, 0.0, -0.0, 1e-20, -1e-20, 3.0, -3.0])
    corner_imag = np.float32([0.0, 0.0, -0.0, -0.0, -1e-20, 1e-20, -0.5, 0.5])
    real[:corners.size] = corners
    imag[:corner_imag.size] = corner_imag
    return real, imag


def _complex_from_parts(real: Any, imag: Any) -> Any:
    """``real + 1j*imag`` DOES NOT PRESERVE SIGNED ZEROS — build the words directly.

    Measured while writing this probe and recorded because it silently corrupted an
    earlier run's needles: ``zr + 1j*zi`` multiplies through complex128 and
    canonicalizes ``-0.0`` on the imaginary half, so three of the four signed-zero
    corners never reached the classifier at all. Writing the two planes into a
    complex64 view keeps every bit.
    """
    out = np.zeros(real.size, dtype=np.complex64)
    view = out.view(np.float32).reshape(-1, 2)
    view[:, 0] = real
    view[:, 1] = imag
    return out


def measure(record_backend: str = special_kz.PROBE_BACKEND) -> Dict[str, Any]:
    rng = np.random.default_rng(SEED)
    real, imag = _vectors(rng)
    field = _complex_from_parts(real, imag)

    patterns: Dict[str, str] = {}
    detail: Dict[str, Any] = {}

    # 1. the Bloch phase rotation: `shifted[plane] *= phase` (S:1862) -- FIELD LEFT.
    phase = np.complex64(complex(np.cos(0.7), np.sin(0.7)))
    patterns["c8_mul_c8"], detail["c8_mul_c8"] = _classify(
        (field * phase).astype(np.complex64), real, imag,
        np.float32(phase.real), np.float32(phase.imag))

    # 2. `fu *= kms` / `field *= kms_u` (S:1929-1935) -- FIELD LEFT, real scalar.
    kms = np.float32(0.8137)
    patterns["c8_mul_f4_field_left"], detail["c8_mul_f4_field_left"] = _classify(
        (field * kms).astype(np.complex64), real, imag, kms, np.float32(0.0))

    # 3. `kps * fw` / `kms * fw_previous` (S:2086-2095) -- COEFFICIENT LEFT.
    patterns["f4_mul_c8_coefficient_left"], detail["f4_mul_c8_coefficient_left"] = (
        _classify((kms * field).astype(np.complex64), kms, np.float32(0.0),
                  real, imag))

    # 4. `xp.multiply(dtdx, total)` (S:1635) -- PYTHON FLOAT LEFT.
    dtdx = 0.35
    patterns["python_float_left"], detail["python_float_left"] = _classify(
        np.multiply(np.float32(dtdx), field).astype(np.complex64),
        np.float32(dtdx), np.float32(0.0), real, imag)

    # 5. THIS TRANCHE'S OWN: the beta coefficient (S:784) -- COEFFICIENT LEFT, with
    #    a SIGNED-ZERO real word. Every corpus beta, both signs, both sides, so the
    #    classification covers every coefficient the engine can actually bind.
    verdicts: List[str] = []
    per_coefficient: List[Dict[str, Any]] = []
    for beta, dt in ((0.3321611318837033, 1.0 / 24.0),
                     (-0.6850526103319672, 0.034),
                     (-0.3907, 0.0233)):
        for magnetic in (True, False):
            for words in special_kz.beta_curl_coefficients(
                    beta, dt, magnetic, complex_storage=True):
                coefficient = _complex_from_parts(np.float32([words[0]]),
                                                  np.float32([words[1]]))[0]
                verdict, entry = _classify(
                    (coefficient * field).astype(np.complex64),
                    np.float32(words[0]), np.float32(words[1]), real, imag)
                verdicts.append(verdict)
                per_coefficient.append({
                    "beta": beta, "dt": dt, "magnetic": magnetic,
                    "coefficient_words": [words[0], words[1]],
                    "verdict": verdict,
                    "naive_differing": entry["NAIVE"]["differing_words"],
                    "fma_differing": entry["FMA_V1"]["differing_words"]})
    unique = sorted(set(verdicts))
    patterns[special_kz.BETA_PROBE_PATTERN] = (
        unique[0] if len(unique) == 1 else "NEITHER")
    detail[special_kz.BETA_PROBE_PATTERN] = {
        "per_coefficient": per_coefficient,
        "distinct_verdicts": unique,
        "note": ("AMBIGUOUS_BOTH is the expected verdict: c_re is exactly +-0 so "
                 "the fused arm's extra product is exact and the two arms cannot "
                 "be told apart on this pattern. It still gates NEITHER."),
    }

    return {
        "backend": record_backend,
        "numpy": np.__version__,
        "seed": SEED,
        "vector_cells": VECTOR_CELLS,
        "patterns": patterns,
        "detail": detail,
        "subnormal_free_vectors": True,
        "orientation_note": ("the fused product is the LEFT operand's; every "
                             "pattern is probed in the orientation its call site "
                             "writes, and a single-orientation probe would "
                             "misclassify half of them"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()

    record = measure()
    record["environment"] = kit.environment_stamp()
    arm = special_kz.beta_expansion_from_probe(record)
    record["licensed_expansion"] = arm
    kit.save(record, arguments.out)

    for name, verdict in sorted(record["patterns"].items()):
        kit.log(f"[pattern] {name:<38} {verdict}")
    kit.log(f"[license] expansion arm = {arm!r}")
    if arm is None:
        neither = [name for name, verdict in record["patterns"].items()
                   if verdict == "NEITHER"]
        kit.log(f"REFUSED: no arm reproduces this host's reference for {neither}; "
                f"the complex beta curl may not be built here")
        return kit.EXIT_CANNOT_CERTIFY
    kit.log(f"EXPANSION PROBE WROTE {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
