"""What this host actually does with the cylindrical complex family's own spellings.

FOUR QUESTIONS, FOUR LEGS, ONE ARTIFACT. Each is a platform or array-path fact the
family would otherwise have to INHERIT from the Triton/CuPy track, and this project's
rule is that a remembered fact about another backend is a hypothesis, not a
measurement. Nothing here is copied forward; every number below is cut on this host.

1. ``expansion`` — WHICH COMPLEX-MULTIPLY ARM THE REFERENCE TAKES, over the FIVE
   orientations ``complex_fields`` already probes PLUS THE TWO THIS FAMILY ADDS:

   * ``c8_mul_c8_row_coefficient_left`` — ``stepping._cylindrical_imr_term`` (:724)
     spells ``factor * partner_values`` with ``factor`` an ``(nr, 1, 1)`` complex64
     ROW and ``partner_values`` an ``(nr, ny, nz)`` complex64 VOLUME. A broadcast
     complex-by-complex product with the COEFFICIENT ON THE LEFT. None of the base
     five is that call: they cover a general product with the FIELD left, an array
     times a complex SCALAR, a REAL coefficient in each orientation, and a Python
     float on the left.
   * ``c8_mul_c8_scalar_left`` — ``_cylindrical_axis_increment_B`` (:644) spells
     ``1j * (m*dtdx) * ez_off_axis``, i.e. a PYTHON COMPLEX on the LEFT of a
     complex64 array. Its real word is a SIGNED ZERO (``-0.0`` for m < 0), which is
     exactly the operand class a plane-wise shortcut gets wrong.

   ``special_kz`` and ``folded_complex`` each established that a new operand
   ORIENTATION earns its own pattern rather than inheriting a verdict measured
   elsewhere. This follows them, and an artifact cut before this tranche licenses
   nothing here.

2. ``negation`` — HOW ``-x`` LOWERS ON METAL, re-measured rather than inherited.
   The Triton cylindrical kernel spells every negated addend ``x * -1.0`` because
   *Triton* lowers unary minus as ``0.0 - x`` and canonicalizes signed zeros. THAT
   IS A FACT ABOUT TRITON. This leg compiles all three spellings and compares them
   against ``numpy``'s own unary negation on an EXHAUSTIVE signed-zero table, so the
   shipped spelling is chosen by this host's answer.

3. ``axis_identity`` — WHETHER THE r = 0 NEAR GHOST IS OBSERVABLE, measured on the
   ARRAY PATH on this host (NumPy, no device involved). ``stepping._shift_up``'s
   ``CYL_AXIS`` branch SHARES the ``METALLIC`` arm by source identity (:1781-1783 —
   one branch, three kinds), but the NEAR ghost (:1830-1842) writes the
   ``r_to_minus_r`` image and is a different rule. The claim the kernel rests on is
   that the ownership mask zeroes the only row that could consume it. That claim is
   about ``stepping.py``, not about a backend, so it is checkable here: run the real
   sub-step, then run it again with ``boundaries[0]`` forced to ``metallic`` — the
   arm the kernel actually compiles — and count differing words.

4. ``helper_agreement`` — WHETHER THE SHIPPED SHADER HELPER REPRODUCES THE
   REFERENCE on this family's two new orientations. Leg 1 classifies what numpy
   does; this launches ``templates.complex_helpers(arm)``'s ``c_mul`` on the device
   against the same vectors and counts differing uint32 words. Leg 1 alone would
   license an arm from a host-side model of the kernel; this is the kernel.

EVIDENCE CLASS, STATED AND NOT PAPERED OVER. ``torch.mps.compile_shader`` exposes no
AIR, no GPU ISA and no optimisation report, so legs 2 and 4 certify that the ANSWERS
agree, never that the INSTRUCTIONS do. They catch a wrong answer, not a wrong
instruction. That is this backend's standing certification gap against the Triton and
CUDA tracks and it stays stated on every claim built on top of these numbers.

NO SUBNORMALS ANYWHERE IN THE VECTORS. MPS flushes float32 subnormals natively with
no lever, so every claim this artifact licenses is made under a CHECKED
subnormal-free precondition; a probe that mixed the band question into the
orientation question would answer neither. The census is asserted, not assumed.

    python -u probe_metal_cylindrical_complex.py \\
        --out results/metal_cylindrical_complex_2026-08-16/expansion.json
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
API_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
if API_ROOT not in sys.path:
    sys.path.insert(0, API_ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import metal_gate_kit as kit  # noqa: E402

from meep_gpu.metal_kernels import (  # noqa: E402
    complex_fields,
    cylindrical_complex,
    templates,
)

SEED = 20260816
VECTOR_CELLS = 8192

#: The negation leg's table is EXHAUSTIVE over the operand classes that matter, not
#: sampled: random data discriminates none of the three spellings (measured 0/16384
#: by the complex tranche), and every word the wrong spelling moves is a signed zero.
_NEGATION_CORNERS: Tuple[float, ...] = (
    0.0, -0.0, 1.0, -1.0, 3.5, -3.5, 1e-20, -1e-20, 1e20, -1e20,
    float(np.float32(np.pi)), float(-np.float32(np.pi)),
)


def _words(array: Any) -> Any:
    return np.ascontiguousarray(array, dtype=np.float32).reshape(-1).view(np.uint32)


def _differing(left: Any, right: Any) -> int:
    return int(np.count_nonzero(_words(left) != _words(right)))


def _subnormal_words(array: Any) -> int:
    """How many float32 words in ``array`` sit in the subnormal band."""
    flat = np.ascontiguousarray(array, dtype=np.float32).reshape(-1)
    magnitude = np.abs(flat)
    return int(np.count_nonzero((magnitude > 0.0)
                                & (magnitude < np.float32(np.finfo(np.float32).tiny))))


def _fma(a: Any, b: Any, addend: Any) -> Any:
    """An exact-enough f32 fma: the product is EXACT in float64 (24+24 <= 53 bits).

    The subsequent add rounds twice, which can in principle differ from a true fma.
    It is used only to CLASSIFY; the classification is corroborated by leg 4, where
    the shipped shader's own ``fma()`` reproduces the reference on the same vectors.
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
    keeps every bit, which for THIS family matters twice over — the i*m/r row's real
    word is ``+0.0`` and the |m| = 1 scalar's is ``-0.0`` at negative m.
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


# ---------------------------------------------------------------------------
# Leg 1 — the expansion arm
# ---------------------------------------------------------------------------

def measure_expansion(record_backend: str = cylindrical_complex.PROBE_BACKEND
                      ) -> Dict[str, Any]:
    rng = np.random.default_rng(SEED)
    real, imag = _vectors(rng)
    field = _complex_from_parts(real, imag)

    patterns: Dict[str, str] = {}
    detail: Dict[str, Any] = {}

    # THE BASE FIVE ARE `complex_fields`' OWN CALL SITES, re-measured here rather
    # than read across from that family's artifact. This family's curl launches a
    # body built from the same helpers — the dtdx scalar-left multiply, the
    # `fu *= kms` family — so the same orientations have to agree, and an artifact
    # carrying only the two new patterns would license a kernel on evidence about
    # two of its five multiplies.
    phase = np.complex64(complex(np.cos(0.7), np.sin(0.7)))
    patterns["c8_mul_c8"], detail["c8_mul_c8"] = _classify(
        (field * phase).astype(np.complex64), real, imag,
        np.float32(phase.real), np.float32(phase.imag))
    patterns["c8_mul_c8_scalar_right"], detail["c8_mul_c8_scalar_right"] = _classify(
        np.multiply(field, phase).astype(np.complex64), real, imag,
        np.float32(phase.real), np.float32(phase.imag))
    kms = np.float32(0.8137)
    patterns["c8_mul_f4_field_left"], detail["c8_mul_f4_field_left"] = _classify(
        (field * kms).astype(np.complex64), real, imag, kms, np.float32(0.0))
    patterns["f4_mul_c8_coefficient_left"], detail["f4_mul_c8_coefficient_left"] = (
        _classify((kms * field).astype(np.complex64), kms, np.float32(0.0),
                  real, imag))
    dtdx = 0.37
    patterns["python_float_left"], detail["python_float_left"] = _classify(
        np.multiply(np.float32(dtdx), field).astype(np.complex64),
        np.float32(dtdx), np.float32(0.0), real, imag)

    # 6. THIS TRANCHE'S FIRST: the i*m/r coefficient as a BROADCAST ROW on the LEFT,
    #    built by the family's own transcription of `stepping`'s `_build` and
    #    multiplied exactly as :724 multiplies it. Swept over BOTH signs of m, BOTH
    #    call-site signs and BOTH targets of each sub-step, because the row's real
    #    word is laundered to +0.0 by the `(-1j) * X` spelling and a sign that
    #    escaped it would be visible only here.
    rows = 32
    volume = _complex_from_parts(
        rng.standard_normal(rows * 3 * 5).astype(np.float32),
        rng.standard_normal(rows * 3 * 5).astype(np.float32)).reshape(rows, 3, 5)
    row_verdicts: List[str] = []
    per_row: List[Dict[str, Any]] = []
    for m in (1, -1, 2, -2, 3, 5):
        for sub_step, terms in cylindrical_complex.IMR_TERMS.items():
            targets = cylindrical_complex.SUB_STEPS[sub_step]["targets"]
            for index, _register, sign in terms:
                factor = cylindrical_complex.imr_coefficient_row(
                    np, targets[index], sign, m, dtdx, rows, np.complex64)
                product = (factor * volume).astype(np.complex64)
                broadcast = np.broadcast_to(factor, volume.shape)
                verdict, entry = _classify(
                    product.reshape(-1),
                    np.ascontiguousarray(broadcast.real).reshape(-1),
                    np.ascontiguousarray(broadcast.imag).reshape(-1),
                    np.ascontiguousarray(volume.real).reshape(-1),
                    np.ascontiguousarray(volume.imag).reshape(-1))
                row_verdicts.append(verdict)
                per_row.append({
                    "m": m, "sub_step": sub_step, "target": targets[index],
                    "sign": sign, "verdict": verdict,
                    "real_word_hex": f"0x{_words(factor.real)[0]:08x}",
                    "real_words_all_plus_zero": bool(
                        np.all(_words(factor.real) == np.uint32(0))),
                    "naive_differing": entry["NAIVE"]["differing_words"],
                    "fma_differing": entry["FMA_V1"]["differing_words"]})
    unique_rows = sorted(set(row_verdicts))
    patterns[cylindrical_complex.IMR_ROW_PROBE_PATTERN] = (
        unique_rows[0] if len(unique_rows) == 1 else "NEITHER")
    detail[cylindrical_complex.IMR_ROW_PROBE_PATTERN] = {
        "per_row": per_row, "distinct_verdicts": unique_rows,
        "note": ("AMBIGUOUS_BOTH is the expected verdict and is accepted by "
                 "arithmetic rather than leniency: the row's REAL word is exactly "
                 "+0.0 at every m, so c_re*z_re is exact and the fused and "
                 "separately-rounded forms are the same single rounding. It still "
                 "gates NEITHER, and `real_words_all_plus_zero` is the datum that "
                 "makes the reason checkable rather than asserted.")}

    # 7. THIS TRANCHE'S SECOND: the |m| = 1 B-side scalar, a PYTHON COMPLEX on the
    #    LEFT whose real word is a SIGNED ZERO. Both signs of m, because the sign is
    #    the whole point — `1j * X` gives -0.0 for X < 0 where the i*m/r row's
    #    `(-1j) * X` gives +0.0 for both.
    scalar_verdicts: List[str] = []
    per_scalar: List[Dict[str, Any]] = []
    for m in (1, -1):
        _minus, (inc_re, inc_im) = cylindrical_complex.axis_increment_scalars(m, dtdx)
        scalar = _complex_from_parts(np.float32([inc_re]), np.float32([inc_im]))[0]
        product = (scalar * field).astype(np.complex64)
        verdict, entry = _classify(product, np.float32(inc_re), np.float32(inc_im),
                                   real, imag)
        scalar_verdicts.append(verdict)
        per_scalar.append({
            "m": m, "verdict": verdict,
            "real_word_hex": f"0x{_words(np.float32([inc_re]))[0]:08x}",
            "real_word_is_negative_zero": bool(
                _words(np.float32([inc_re]))[0] == np.uint32(0x80000000)),
            "naive_differing": entry["NAIVE"]["differing_words"],
            "fma_differing": entry["FMA_V1"]["differing_words"]})
    unique_scalars = sorted(set(scalar_verdicts))
    patterns[cylindrical_complex.AXIS_SCALAR_PROBE_PATTERN] = (
        unique_scalars[0] if len(unique_scalars) == 1 else "NEITHER")
    detail[cylindrical_complex.AXIS_SCALAR_PROBE_PATTERN] = {
        "per_scalar": per_scalar, "distinct_verdicts": unique_scalars}

    return {
        "backend": record_backend,
        "numpy": np.__version__,
        "seed": SEED,
        "vector_cells": VECTOR_CELLS,
        "patterns": patterns,
        "detail": detail,
        "subnormal_words_in_vectors": _subnormal_words(field.view(np.float32)),
        "orientation_note": ("the fused product is the LEFT operand's; every pattern "
                             "is probed in the orientation its call site writes, and "
                             "a single-orientation probe would misclassify half"),
    }


# ---------------------------------------------------------------------------
# Leg 2 — how negation lowers on THIS backend
# ---------------------------------------------------------------------------

_NEGATION_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

kernel void negate_probe(
    device float*       unary   [[buffer(0)]],
    device float*       mulneg  [[buffer(1)]],
    device float*       subzero [[buffer(2)]],
    device const float* src     [[buffer(3)]],
    constant uint&      n_elem  [[buffer(4)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n_elem) { return; }
    float x = src[idx];
    unary[idx]   = -x;
    mulneg[idx]  = x * -1.0f;
    subzero[idx] = 0.0f - x;
}
"""


def measure_negation(runner: "Runner") -> Dict[str, Any]:
    """Three spellings of negation against ``numpy``'s own, on this host.

    The shipped spelling is chosen by THIS table, not by the Triton track's. The
    reference is ``numpy.negative``, which is what ``stepping`` performs when it
    writes ``-axis_increment[1]`` (:372, :453) and ``-(factor * partner_values)``
    (:724) — a sign flip on every word including the zeros.
    """
    source = np.array(_NEGATION_CORNERS, dtype=np.float32)
    rng = np.random.default_rng(SEED + 1)
    padded = np.concatenate(
        [source, rng.standard_normal(1024 - source.size).astype(np.float32)])
    reference = np.negative(padded)

    out: Dict[str, Any] = {"total_words": int(padded.size), "modes": {}}
    for mode in templates.CONTRACT_MODES:
        filled = templates.substitute(
            _NEGATION_SOURCE, {"__CONTRACT__": templates.contraction_pragma(mode)})
        got = runner.launch_negation(filled, padded)
        out["modes"][mode] = {
            spelling: {
                "differing_words": _differing(values, reference),
                "example_indices": [
                    int(i) for i in np.flatnonzero(_words(values) != _words(reference))
                    [:4]],
                "example_source": [
                    float(padded[i]) for i in
                    np.flatnonzero(_words(values) != _words(reference))[:4]],
            }
            for spelling, values in got.items()}
        out["modes"][mode]["moved_words"] = int(
            sum(int(np.count_nonzero(_words(values) != _words(
                np.full(padded.size, np.float32(-7.5)))))
                for values in got.values()))
    out["subnormal_words_in_source"] = _subnormal_words(padded)
    out["shipped_spelling"] = "unary_minus"
    out["note"] = ("Triton spells negated addends `x * -1.0` because TRITON lowers "
                   "unary minus as `0.0 - x` and canonicalizes signed zeros. That "
                   "reason is about Triton. This table is what Metal does.")
    return out


# ---------------------------------------------------------------------------
# Leg 3 — is the r = 0 NEAR ghost observable? (array path, NumPy, no device)
# ---------------------------------------------------------------------------

def measure_axis_identity() -> Dict[str, Any]:
    """Does forcing ``boundaries[0] = 'metallic'`` change a single word?

    THE KERNEL COMPILES ``BCX = METALLIC`` ON THE r AXIS, and this leg is why it may.
    Two halves, and only the first is a reading:

    * the FAR ghost is a SOURCE IDENTITY — ``_shift_up`` (:1781-1783) has ONE branch
      covering ``MIRROR``, ``METALLIC`` and ``CYL_AXIS`` and writes the same hard
      zero for all three. Nothing to measure; a test asserts the branch text.
    * the NEAR ghost is NOT — ``_shift_down``'s ``CYL_AXIS`` branch (:1830-1842)
      writes the ``r_to_minus_r`` image where ``METALLIC`` writes zero. The kernel's
      claim is that ``_mask_non_owned_cells`` (:1898-1902) zeroes the only row that
      could consume it. That is a claim about ``stepping.py``, so it is MEASURED
      here by running the real sub-step twice.

    A NONZERO COUNT IS A LEGITIMATE OUTCOME and would mean this family must carry the
    near ghost rather than compile it away. The leg reports the count; it does not
    assume it.
    """
    from meep_gpu import stepping  # noqa: PLC0415
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415

    volumes = ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
               "Hx", "Hy", "Hz", "fu_Bx", "fu_By", "fu_Bz",
               "fu_Dx", "fu_Dy", "fu_Dz")

    def build(m: int, z_kind: str, seed: int):
        grid = Grid(resolution=1.0, cell_size=(18.0, 0.0, 22.0), cylindrical=True,
                    m=int(m), boundaries={"z": z_kind}, courant=0.37, xp=np)
        fields = Fields(grid=grid, force_complex_fields=True)
        shape = tuple(grid.shape)
        fields.set_isotropic_epsilon_volume(
            np.full(shape, np.float32(2.25)),
            np.full(shape, np.float32(1.0 / 2.25)))
        fields.enable_pml_storage()
        rng = np.random.default_rng(seed)
        for name in volumes:
            array = getattr(fields, name, None)
            if array is None:
                continue
            array.real[...] = rng.standard_normal(array.shape).astype(np.float32)
            array.imag[...] = rng.standard_normal(array.shape).astype(np.float32)
        return fields, PML(grid=grid, thickness={"x": (0, 4), "z": 4})

    def snapshot(fields) -> Dict[str, Any]:
        return {name: np.array(getattr(fields, name), copy=True)
                for name in volumes if getattr(fields, name, None) is not None}

    original = stepping._boundary_kinds
    rows: List[Dict[str, Any]] = []
    for m in (1, -1, 2, -2, 3, 5):
        for z_kind in ("metallic", "periodic"):
            for sub_step, runner in (("step_B", stepping.step_B),
                                     ("step_D", stepping.step_D)):
                seed = abs(hash((m, z_kind, sub_step))) % (2 ** 31)
                fields_a, pml_a = build(m, z_kind, seed)
                before = snapshot(fields_a)
                runner(fields_a, pml_a)
                after_axis = snapshot(fields_a)

                fields_b, pml_b = build(m, z_kind, seed)

                def forced(grid, pml, _original=original):
                    kinds = _original(grid, pml)
                    return (stepping.METALLIC,) + tuple(kinds[1:])

                stepping._boundary_kinds = forced
                try:
                    runner(fields_b, pml_b)
                finally:
                    stepping._boundary_kinds = original
                after_metallic = snapshot(fields_b)

                differing = sum(_differing(after_axis[name].view(np.float32),
                                           after_metallic[name].view(np.float32))
                                for name in after_axis)
                moved = sum(_differing(before[name].view(np.float32),
                                       after_axis[name].view(np.float32))
                            for name in after_axis)
                total = sum(int(after_axis[name].view(np.float32).size)
                            for name in after_axis)
                band = sum(_subnormal_words(after_axis[name].view(np.float32))
                           for name in after_axis)
                rows.append({"m": m, "z": z_kind, "sub_step": sub_step,
                             "differing_words": differing, "moved_words": moved,
                             "total_words": total, "subnormal_words": band})
                kit.log(f"[axis] m={m:+d} z={z_kind:<8} {sub_step}: "
                        f"differing={differing} moved={moved}/{total} "
                        f"subnormal={band}")

    return {
        "rows": rows,
        "rows_measured": len(rows),
        "total_differing": sum(row["differing_words"] for row in rows),
        "minimum_moved": min(row["moved_words"] for row in rows) if rows else 0,
        "subnormal_words": sum(row["subnormal_words"] for row in rows),
        "far_ghost": ("SOURCE IDENTITY, not measured: stepping._shift_up:1781-1783 "
                      "is one branch over (MIRROR, METALLIC, CYL_AXIS)"),
        "near_ghost": ("MEASURED: _shift_down's CYL_AXIS branch (:1830-1842) writes "
                       "the r_to_minus_r image where METALLIC writes zero; the "
                       "count above is whether any stored word can tell"),
    }


# ---------------------------------------------------------------------------
# Leg 4 — does the shipped shader helper reproduce the reference?
# ---------------------------------------------------------------------------

_HELPER_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

__HELPERS__

kernel void helper_probe(
    device float2*       out    [[buffer(0)]],
    device const float2* coef   [[buffer(1)]],
    device const float2* field  [[buffer(2)]],
    constant uint&       n_elem [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n_elem) { return; }
    out[idx] = c_mul(coef[idx], field[idx]);
}
"""


def measure_helper_agreement(runner: "Runner", arm: str) -> Dict[str, Any]:
    """``c_mul(coefficient, field)`` on the device against ``coefficient * field``.

    THE ORIENTATION IS THE POINT. ``stepping`` writes ``factor * partner_values``
    with the COEFFICIENT on the left, so the shader passes it as ``c_mul``'s FIRST
    argument. Swapping them is a different fused expansion and a different set of
    bytes; this leg is what makes "the helper serves this call site" a measurement.
    """
    rng = np.random.default_rng(SEED + 2)
    cells = 4096
    dtdx = 0.37
    rows: List[Dict[str, Any]] = []
    for m in (1, -1, 2, -2, 3, 5):
        for sub_step, terms in cylindrical_complex.IMR_TERMS.items():
            targets = cylindrical_complex.SUB_STEPS[sub_step]["targets"]
            for index, _register, sign in terms:
                factor = cylindrical_complex.imr_coefficient_row(
                    np, targets[index], sign, m, dtdx, cells, np.complex64
                ).reshape(-1)
                field = _complex_from_parts(
                    rng.standard_normal(cells).astype(np.float32),
                    rng.standard_normal(cells).astype(np.float32))
                reference = (factor * field).astype(np.complex64)
                for mode in templates.CONTRACT_MODES:
                    filled = templates.substitute(_HELPER_SOURCE, {
                        "__CONTRACT__": templates.contraction_pragma(mode),
                        "__HELPERS__": templates.complex_helpers(arm)})
                    got, moved = runner.launch_helper(filled, factor, field)
                    rows.append({
                        "m": m, "sub_step": sub_step, "target": targets[index],
                        "sign": sign, "contract": mode,
                        "differing_words": _differing(got.view(np.float32),
                                                      reference.view(np.float32)),
                        "moved_words": moved,
                        "total_words": int(got.view(np.float32).size),
                        "subnormal_words": _subnormal_words(got.view(np.float32))})
    kit.log(f"[helper] {len(rows)} rows, "
            f"{sum(r['differing_words'] for r in rows)} differing")
    return {
        "arm": arm,
        "rows": rows,
        "rows_measured": len(rows),
        "total_differing": sum(row["differing_words"] for row in rows),
        "minimum_moved": min(row["moved_words"] for row in rows) if rows else 0,
        "subnormal_words": sum(row["subnormal_words"] for row in rows),
        "evidence_class": ("BEHAVIOURAL. torch.mps.compile_shader exposes no AIR and "
                           "no ISA, so this certifies that the ANSWERS agree, never "
                           "that the instructions do"),
    }


# ---------------------------------------------------------------------------
# The device runner
# ---------------------------------------------------------------------------

class Runner:
    """Compiles on demand and launches one probe kernel against fresh device copies.

    EVERY OUTPUT BUFFER IS SENTINEL-FILLED, never zero-filled: a kernel that wrote
    nothing would otherwise compare equal to a reference of zeros, and a no-op
    agreeing with a no-op certifies nothing. ``moved`` is the vacuity floor.
    """

    SENTINEL = np.float32(-7.5)

    def __init__(self) -> None:
        import torch  # noqa: PLC0415

        self.torch = torch
        self._cache: Dict[str, Any] = {}
        self.compiles = 0
        self.launches = 0

    def library(self, source: str) -> Any:
        library = self._cache.get(source)
        if library is None:
            library = self.torch.mps.compile_shader(source)
            self._cache[source] = library
            self.compiles += 1
        return library

    def launch_negation(self, source: str, values: Any) -> Dict[str, Any]:
        torch = self.torch
        library = self.library(source)
        n = int(values.size)
        sentinel = np.full(n, self.SENTINEL, dtype=np.float32)
        outputs = [torch.from_numpy(sentinel.copy()).to("mps") for _ in range(3)]
        source_tensor = torch.from_numpy(
            np.ascontiguousarray(values, dtype=np.float32)).to("mps")
        library.negate_probe(*outputs, source_tensor, n)
        torch.mps.synchronize()
        self.launches += 1
        names = ("unary_minus", "mul_minus_one", "zero_minus")
        return {name: tensor.cpu().numpy()
                for name, tensor in zip(names, outputs)}

    def launch_helper(self, source: str, coefficient: Any, field: Any
                      ) -> Tuple[Any, int]:
        torch = self.torch
        library = self.library(source)
        n = int(field.size)
        sentinel = np.full(n * 2, self.SENTINEL, dtype=np.float32)
        out = torch.from_numpy(sentinel.copy().view(np.complex64)).to("mps")
        d_coef = torch.from_numpy(
            np.ascontiguousarray(coefficient, dtype=np.complex64)).to("mps")
        d_field = torch.from_numpy(
            np.ascontiguousarray(field, dtype=np.complex64)).to("mps")
        library.helper_probe(out, d_coef, d_field, n)
        torch.mps.synchronize()
        self.launches += 1
        got = out.cpu().numpy()
        moved = int(np.count_nonzero(
            _words(got.view(np.float32)) != _words(sentinel)))
        return got, moved


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--legs", default="all",
                        help="comma-separated subset of "
                             "expansion,negation,axis_identity,helper_agreement")
    arguments = parser.parse_args()
    wanted = ({"expansion", "negation", "axis_identity", "helper_agreement"}
              if arguments.legs == "all"
              else {name.strip() for name in arguments.legs.split(",")})

    record: Dict[str, Any] = {"legs_run": sorted(wanted)}
    failures: List[str] = []

    if "expansion" in wanted:
        kit.log("[leg] expansion")
        record.update(measure_expansion())
        for name, verdict in sorted(record["patterns"].items()):
            kit.log(f"[pattern] {name:<40} {verdict}")

    arm: Optional[str] = None
    if "patterns" in record:
        arm = cylindrical_complex.expansion_from_probe(record)
        record["licensed_expansion"] = arm
        kit.log(f"[license] expansion arm = {arm!r}")
        if arm is None:
            failures.append("no expansion arm is licensed by this artifact")

    runner: Optional[Runner] = None
    if wanted & {"negation", "helper_agreement"}:
        runner = Runner()
        kit.log(f"[device] torch {runner.torch.__version__} "
                f"mps={runner.torch.backends.mps.is_available()}")

    if "negation" in wanted and runner is not None:
        kit.log("[leg] negation")
        record["negation"] = measure_negation(runner)
        for mode, table in sorted(record["negation"]["modes"].items()):
            for spelling in ("unary_minus", "mul_minus_one", "zero_minus"):
                kit.log(f"[negation] contract={mode:<4} {spelling:<14} "
                        f"{table[spelling]['differing_words']}/"
                        f"{record['negation']['total_words']}")
        for mode, table in record["negation"]["modes"].items():
            if table["unary_minus"]["differing_words"]:
                failures.append(
                    f"the shipped `-x` spelling misses "
                    f"{table['unary_minus']['differing_words']} words under "
                    f"contract={mode}")

    if "axis_identity" in wanted:
        kit.log("[leg] axis_identity (array path, NumPy)")
        record["axis_identity"] = measure_axis_identity()
        kit.log(f"[axis] total differing = "
                f"{record['axis_identity']['total_differing']} over "
                f"{record['axis_identity']['rows_measured']} rows")
        if record["axis_identity"]["minimum_moved"] <= 0:
            failures.append("an axis_identity row moved no words: a no-op agreeing "
                            "with a no-op certifies nothing")

    if "helper_agreement" in wanted and runner is not None:
        if arm is None:
            failures.append("helper_agreement needs a licensed arm and has none")
        else:
            kit.log("[leg] helper_agreement")
            record["helper_agreement"] = measure_helper_agreement(runner, arm)
            if record["helper_agreement"]["minimum_moved"] <= 0:
                failures.append("a helper_agreement row moved no words")
            if record["helper_agreement"]["total_differing"]:
                failures.append(
                    f"the shipped c_mul helper misses "
                    f"{record['helper_agreement']['total_differing']} words against "
                    f"the reference on this family's own orientation")

    if runner is not None:
        record["device"] = {"compiles": runner.compiles, "launches": runner.launches}
    record["environment"] = kit.environment_stamp()
    record["failures"] = failures
    kit.save(record, arguments.out)
    kit.log(f"[artifact] {arguments.out}")
    for failure in failures:
        kit.log(f"[refusal] {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
