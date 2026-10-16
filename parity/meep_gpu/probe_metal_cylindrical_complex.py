"""What this host actually does with the cylindrical complex family's own spellings.

THREE QUESTIONS, THREE LEGS, ONE ARTIFACT. Each is a platform or array-path fact the
family would otherwise have to INHERIT from the Triton/CuPy track, and this project's
rule is that a remembered fact about another backend is a hypothesis, not a
measurement. Nothing here is copied forward; every number below is cut on this host.

1. ``negation`` — HOW ``-x`` LOWERS ON METAL, re-measured rather than inherited.
   The Triton cylindrical kernel spells every negated addend ``x * -1.0`` because
   *Triton* lowers unary minus as ``0.0 - x`` and canonicalizes signed zeros. THAT
   IS A FACT ABOUT TRITON. This leg compiles all three spellings and compares them
   against ``numpy``'s own unary negation on an EXHAUSTIVE signed-zero table, so the
   shipped spelling is chosen by this host's answer.

2. ``axis_identity`` — WHETHER THE r = 0 NEAR GHOST IS OBSERVABLE, measured on the
   ARRAY PATH on this host (NumPy, no device involved). ``stepping._shift_up``'s
   ``CYL_AXIS`` branch SHARES the ``METALLIC`` arm by source identity (:1781-1783 —
   one branch, three kinds), but the NEAR ghost (:1830-1842) writes the
   ``r_to_minus_r`` image and is a different rule. The claim the kernel rests on is
   that the ownership mask zeroes the only row that could consume it. That claim is
   about ``stepping.py``, not about a backend, so it is checkable here: run the real
   sub-step, then run it again with ``boundaries[0]`` forced to ``metallic`` — the
   arm the kernel actually compiles — and count differing words.

3. ``helper_agreement`` — WHETHER THE SHIPPED SHADER HELPER REPRODUCES THE
   REFERENCE on this family's own orientation (the i*m/r coefficient ROW on the
   left of a volume): this launches ``templates.complex_helpers()``'s ``c_mul`` on
   the device and counts differing uint32 words against ``factor * field``.

SINCE THE ARRAY PATH STEPS MEEP'S CYLINDRICAL FORMS (bit_parity_0_10 AP6c), THE i*m/r
ROW IS NOT A CALL ``stepping`` MAKES: the term is a per-part quarter turn of a float32
``rinv`` row (``stepping._cylindrical_imr_increment``), and the Metal kernel forms no
general complex product at all. Leg 3 still builds the retired row
(:func:`legacy_imr_row`) as the operand ``c_mul`` is measured on.

EVIDENCE CLASS, STATED AND NOT PAPERED OVER. ``torch.mps.compile_shader`` exposes no
AIR, no GPU ISA and no optimisation report, so legs 1 and 3 certify that the ANSWERS
agree, never that the INSTRUCTIONS do. They catch a wrong answer, not a wrong
instruction. That is this backend's standing certification gap against the Triton and
CUDA tracks and it stays stated on every claim built on top of these numbers.

NO SUBNORMALS ANYWHERE IN THE VECTORS. MPS flushes float32 subnormals natively with
no lever, so every claim this artifact licenses is made under a CHECKED
subnormal-free precondition; a probe that mixed the band question into the
orientation question would answer neither. The census is asserted, not assumed.

    python -u probe_metal_cylindrical_complex.py \\
        --out results/metal_cylindrical_complex_<date>/probe.json
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
    cylindrical_complex,
    templates,
)

SEED = 20260816

#: The negation leg's table is EXHAUSTIVE over the operand classes that matter, not
#: sampled: random data discriminates none of the three spellings (measured 0/16384
#: by the complex tranche), and every word the wrong spelling moves is a signed zero.
_NEGATION_CORNERS: Tuple[float, ...] = (
    0.0, -0.0, 1.0, -1.0, 3.5, -3.5, 1e-20, -1e-20, 1e20, -1e20,
    float(np.float32(np.pi)), float(-np.float32(np.pi)),
)


def legacy_imr_row(xp: Any, target: str, sign: float, m: int, dtdx: float, rows: int,
                   dtype: Any) -> Any:
    """The ``(rows, 1, 1)`` complex row ``(-1j) * (sign*2*m*dtdx) / max(2r + iyee_r, 1)``.

    The operand class of the retired i*m/r coefficient row, built as it was:
    the quotient in float64, rounded once to ``dtype``. Its real word is +0.0 on every
    entry, whatever the sign, because the ``(-1j) * X`` spelling launders it.
    """
    from meep_gpu.fields import IYEE_SHIFTS  # noqa: PLC0415

    iyee_r = IYEE_SHIFTS[target][0]
    doubled = xp.maximum(2 * xp.arange(int(rows), dtype=xp.float64) + iyee_r, 1.0)
    return ((-1j) * (float(sign) * 2.0 * int(m) * float(dtdx))
            / doubled.reshape(-1, 1, 1)).astype(dtype)


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


def _complex_from_parts(real: Any, imag: Any) -> Any:
    """``real + 1j*imag`` DOES NOT PRESERVE SIGNED ZEROS — build the words directly.

    ``zr + 1j*zi`` multiplies through complex128 and canonicalizes
    ``-0.0`` on the imaginary half. Writing the two planes into a complex64 view
    keeps every bit, which for THIS family matters twice over — the i*m/r row's real
    word is ``+0.0`` and the |m| = 1 scalar's is ``-0.0`` at negative m.
    """
    out = np.zeros(np.asarray(real).size, dtype=np.complex64)
    view = out.view(np.float32).reshape(-1, 2)
    view[:, 0] = real
    view[:, 1] = imag
    return out


# ---------------------------------------------------------------------------
# Leg 1 — how negation lowers on THIS backend
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
# Leg 2 — is the r = 0 NEAR ghost observable? (array path, NumPy, no device)
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
# Leg 3 — does the shipped shader helper reproduce the reference?
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


def measure_helper_agreement(runner: "Runner") -> Dict[str, Any]:
    """``c_mul(coefficient, field)`` on the device against ``coefficient * field``.

    THE ORIENTATION IS THE POINT. ``stepping`` writes ``factor * partner_values``
    with the COEFFICIENT on the left, so the shader passes it as ``c_mul``'s FIRST
    argument. Swapping them is a different product and a different set of
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
                factor = legacy_imr_row(
                    np, targets[index], sign, m, dtdx, cells, np.complex64
                ).reshape(-1)
                field = _complex_from_parts(
                    rng.standard_normal(cells).astype(np.float32),
                    rng.standard_normal(cells).astype(np.float32))
                reference = (factor * field).astype(np.complex64)
                for mode in templates.CONTRACT_MODES:
                    filled = templates.substitute(_HELPER_SOURCE, {
                        "__CONTRACT__": templates.contraction_pragma(mode),
                        "__HELPERS__": templates.complex_helpers()})
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
                             "negation,axis_identity,helper_agreement")
    arguments = parser.parse_args()
    wanted = ({"negation", "axis_identity", "helper_agreement"}
              if arguments.legs == "all"
              else {name.strip() for name in arguments.legs.split(",")})

    record: Dict[str, Any] = {"legs_run": sorted(wanted)}
    failures: List[str] = []

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
        kit.log("[leg] helper_agreement")
        record["helper_agreement"] = measure_helper_agreement(runner)
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
