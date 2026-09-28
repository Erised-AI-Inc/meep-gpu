"""Which complex-multiply arm does the REFERENCE take on the FOLD'S OWN orientation?

THE FAMILY THIS LICENSES EXISTS BECAUSE ONE MULTIPLY STOPPED BEING A SIGN FLIP.
Under real float32 storage the mirror fold's parity is ``+/-1 * word`` and the
certified Metal spelling is ``-x`` or a plain copy — a sign-bit operation, exact on
every input this host can hold. Under complex64 the array path spells
``phase * plane`` (stepping.py:1450-1451 near, :1529-1532 far) with a PYTHON INT and
a complex64 array, and numpy carries only ``'FF->F'`` complex loops, so it is the
FULL complex multiply by ``(+/-1.0, +0.0)`` WITH its zero cross terms. Measured on
this host over a 512-word engineered table, the sign-bit spelling misses 16 words at
BOTH parities, every one of them a signed zero the complex product canonicalizes.

SO THE FOLDED COMPLEX FILL NEEDS ITS OWN COEFFICIENT ORIENTATION MEASURED, and none
of the four patterns ``complex_fields`` probes is it:

* ``c8_mul_c8``                  general complex product, FIELD on the left
* ``c8_mul_c8_scalar_right``     the Bloch rotation, ``shifted[plane] *= phase``
* ``c8_mul_f4_field_left``       a REAL coefficient, field left
* ``f4_mul_c8_coefficient_left`` a REAL coefficient, coefficient left
* ``python_float_left``          a Python float on the left

The parity coefficient is a COMPLEX scalar on the LEFT whose real word is exactly
+/-1.0 and whose imaginary word is bitwise zero. ``special_kz`` established the
precedent that a new operand ORIENTATION earns its own pattern rather than inheriting
a verdict measured elsewhere; ``folded_complex.PARITY_PROBE_PATTERN`` follows it,
and an artifact cut before this tranche licenses nothing.

``AMBIGUOUS_BOTH`` IS THE EXPECTED VERDICT ON THE NEW PATTERN AND IS ACCEPTED THERE
BY ARITHMETIC, NOT BY LENIENCY: with ``c_im`` exactly ``+0.0`` the fused arm's extra
product is EXACT, so both arms produce identical bytes and the pattern cannot prefer
one. What it still gates is ``NEITHER`` — a platform whose bytes no transcription
reproduces refuses BY NAME rather than inheriting a verdict about a different
orientation.

WHAT THIS PROBE DOES NOT MEASURE, stated: the subnormal band. Every vector here is
seeded in the normal range, because the claim this artifact licenses is itself made
under a checked subnormal-free precondition — and for THIS family that precondition
is load-bearing in a way it was not for the real fold's fill. Measured on the same
host: ``c_mul(coefficient, plane)`` matches numpy 0/512 words on a band-free table
and 0/1024 on a plane scaled to 1e-30, but misses 756/1024 at 1e-38 (756 subnormal
operand words in) and 1024/1024 at 1e-40. The divergence count EQUALS the subnormal
operand word count at every scale. A probe that mixed the two questions would answer
neither.

    python -u probe_metal_folded_complex_expansion.py --out results/.../expansion.json
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

from meep_gpu.metal_kernels import complex_fields, folded_complex  # noqa: E402

SEED = 20260816
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
        detail[arm] = {
            "differing_words": misses[arm],
            "example_cells": [
                {"index": int(i),
                 "reference": [float(got.real.reshape(-1)[i]),
                               float(got.imag.reshape(-1)[i])],
                 "arm": [float(np.asarray(arm_re).reshape(-1)[i]),
                         float(np.asarray(arm_im).reshape(-1)[i])]}
                for i in np.flatnonzero(bad)[:3]],
        }
    if misses["NAIVE"] == 0 and misses["FMA_V1"] == 0:
        return complex_fields.AMBIGUOUS_BOTH, detail
    if misses["NAIVE"] == 0:
        return "NAIVE", detail
    if misses["FMA_V1"] == 0:
        return "FMA_V1", detail
    return "NEITHER", detail


def _complex_from_parts(real: Any, imag: Any) -> Any:
    """``real + 1j*imag`` DOES NOT PRESERVE SIGNED ZEROS — build the words directly.

    Recorded by ``probe_metal_beta_expansion`` after it silently corrupted that
    probe's needles: ``zr + 1j*zi`` multiplies through complex128 and canonicalizes
    ``-0.0`` on the imaginary half. Writing the two planes into a complex64 view
    keeps every bit, which for THIS family is the whole question — every word the
    parity product moves that a sign flip does not is a signed zero.
    """
    out = np.zeros(np.asarray(real).size, dtype=np.complex64)
    view = out.view(np.float32).reshape(-1, 2)
    view[:, 0] = real
    view[:, 1] = imag
    return out


def _vectors(rng) -> Tuple[Any, Any]:
    """Random normals plus an exhaustive signed-zero corner block. NO SUBNORMALS."""
    real = rng.standard_normal(VECTOR_CELLS).astype(np.float32)
    imag = rng.standard_normal(VECTOR_CELLS).astype(np.float32)
    corners = np.float32([0.0, -0.0, 0.0, -0.0, 1e-20, -1e-20, 3.0, -3.0])
    corner_imag = np.float32([0.0, 0.0, -0.0, -0.0, -1e-20, 1e-20, -0.5, 0.5])
    real[:corners.size] = corners
    imag[:corner_imag.size] = corner_imag
    return real, imag


def measure(record_backend: str = folded_complex.PROBE_BACKEND) -> Dict[str, Any]:
    rng = np.random.default_rng(SEED)
    real, imag = _vectors(rng)
    field = _complex_from_parts(real, imag)

    patterns: Dict[str, str] = {}
    detail: Dict[str, Any] = {}

    # THE BASE FOUR ARE `complex_fields`' OWN CALL SITES, re-measured here rather
    # than copied across from that family's artifact. The folded curl launches a
    # body built from the same helpers, so the same orientations have to agree; and
    # an artifact that carried only the new pattern would license a kernel on
    # evidence about one of its five multiplies.
    phase = np.complex64(complex(np.cos(0.7), np.sin(0.7)))
    patterns["c8_mul_c8"], detail["c8_mul_c8"] = _classify(
        (field * phase).astype(np.complex64), real, imag,
        np.float32(phase.real), np.float32(phase.imag))

    # `shifted[plane] *= phase` (S:1862): an ARRAY times a complex64 SCALAR.
    patterns["c8_mul_c8_scalar_right"], detail["c8_mul_c8_scalar_right"] = _classify(
        np.multiply(field, phase).astype(np.complex64), real, imag,
        np.float32(phase.real), np.float32(phase.imag))

    kms = np.float32(0.8137)
    patterns["c8_mul_f4_field_left"], detail["c8_mul_f4_field_left"] = _classify(
        (field * kms).astype(np.complex64), real, imag, kms, np.float32(0.0))

    patterns["f4_mul_c8_coefficient_left"], detail["f4_mul_c8_coefficient_left"] = (
        _classify((kms * field).astype(np.complex64), kms, np.float32(0.0),
                  real, imag))

    dtdx = 0.35
    patterns["python_float_left"], detail["python_float_left"] = _classify(
        np.multiply(np.float32(dtdx), field).astype(np.complex64),
        np.float32(dtdx), np.float32(0.0), real, imag)

    # 6. THIS TRANCHE'S OWN: the mirror parity, COEFFICIENT ON THE LEFT, spelled the
    #    way the array path spells it — `phase * plane` with `phase` a PYTHON INT, so
    #    the promotion numpy performs is part of what is being measured. Both mirror
    #    phases and both fills (near = +phase, far = -phase), which is every
    #    coefficient the engine can bind on this path.
    verdicts: List[str] = []
    per_coefficient: List[Dict[str, Any]] = []
    for declared in (1, -1):
        near_words, far_words = folded_complex.mirror_parity_coefficients(declared)
        for fill, words in (("near", near_words), ("far", far_words)):
            parity = int(round(words[0]))
            product = (parity * field).astype(np.complex64)
            verdict, entry = _classify(product, np.float32(words[0]),
                                       np.float32(words[1]), real, imag)
            verdicts.append(verdict)
            per_coefficient.append({
                "declared_phase": declared,
                "fill": fill,
                "coefficient_words": [words[0], words[1]],
                "imaginary_word_hex": f"0x{_words(np.float32([words[1]]))[0]:08x}",
                "verdict": verdict,
                "naive_differing": entry["NAIVE"]["differing_words"],
                "fma_differing": entry["FMA_V1"]["differing_words"]})
    unique = sorted(set(verdicts))
    patterns[folded_complex.PARITY_PROBE_PATTERN] = (
        unique[0] if len(unique) == 1 else "NEITHER")
    detail[folded_complex.PARITY_PROBE_PATTERN] = {
        "per_coefficient": per_coefficient,
        "distinct_verdicts": unique,
        "note": ("AMBIGUOUS_BOTH is the expected verdict: c_im is bitwise +0.0 so "
                 "the fused arm's extra product is exact and the two arms cannot be "
                 "told apart on this pattern. It still gates NEITHER."),
    }

    # THE NON-VACUITY FLOOR FOR THIS ARTIFACT. The whole family exists because the
    # parity product is NOT the sign flip the real fold ships, so the artifact
    # records that difference rather than leaving it in a docstring. If this is ever
    # zero, `folded_complex` should be deleted and `symmetry.mirror_ghost_fill`
    # admitted on complex storage instead.
    sign_flip: Dict[str, Any] = {}
    for declared in (1, -1):
        reference = (declared * field).astype(np.complex64)
        flipped = _words(field.view(np.float32)).copy()
        if declared == -1:
            flipped = flipped ^ np.uint32(0x80000000)
        candidate = flipped.view(np.float32).view(np.complex64)
        differ = int((_words(reference.view(np.float32))
                      != _words(candidate.view(np.float32))).sum())
        sign_flip[f"phase{declared:+d}"] = {
            "differing_words": differ,
            "total_words": int(_words(reference.view(np.float32)).size)}

    return {
        "backend": record_backend,
        "numpy": np.__version__,
        "seed": SEED,
        "vector_cells": VECTOR_CELLS,
        "patterns": patterns,
        "detail": detail,
        "sign_flip_vs_array_path": sign_flip,
        "subnormal_free_vectors": True,
        "orientation_note": ("the fused product is the LEFT operand's; every pattern "
                             "is probed in the orientation its call site writes, and "
                             "a single-orientation probe would misclassify half of "
                             "them"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()

    record = measure()
    record["environment"] = kit.environment_stamp()
    arm = folded_complex.expansion_from_probe(record)
    record["licensed_expansion"] = arm
    kit.save(record, arguments.out)

    for name, verdict in sorted(record["patterns"].items()):
        kit.log(f"[pattern] {name:<38} {verdict}")
    for name, entry in sorted(record["sign_flip_vs_array_path"].items()):
        kit.log(f"[sign-flip] {name}: the real fold's spelling misses "
                f"{entry['differing_words']}/{entry['total_words']} words")
    kit.log(f"[license] expansion arm = {arm!r}")
    if arm is None:
        neither = [name for name, verdict in record["patterns"].items()
                   if verdict == "NEITHER"]
        kit.log(f"[refusal] no arm licensed; NEITHER on {neither or 'none'}")
        return 1
    if not any(entry["differing_words"]
               for entry in record["sign_flip_vs_array_path"].values()):
        kit.log("[refusal] the sign-flip spelling matched the array path on every "
                "word: this family's whole reason to exist is unmeasured here")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
