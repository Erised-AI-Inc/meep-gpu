"""Host tests for the CUDA complex Dcyl H->D two-launch product.

Everything here runs without a device: ``complex_emitter`` and this family's emitter are
CuPy-free, so the whole lift is testable at the merge bar -- the four declared edits
INVERT to the certified prelude exactly, the certified body survives the capture, the
increment's multiply IS the FMA_V1 ``mul_field_left`` lifted from the arm block, the
divide IS the measured scaled algorithm, and both arms emit distinct ASCII sources with
one kernel each. The predicate is exercised on a NumPy-wearing-CuPy shim with a licence
of the shape the arbiter returns.

THE HOST IS NOT THE ORACLE FOR THE COMPLEX DIVIDE, and one test says so in numbers:
NumPy's ``complex64 / float32`` is ``a * (float32(1)/d)`` and the family's spelling is
CuPy's scaled algorithm, which differs from NumPy on the host -- that inversion is the
reason the family carries the CuPy spelling (0 of 14,387,104 words on device) and the
reason its gate arms NumPy's as a mutation.

Device identity is ``parity/meep_gpu/gate_cuda_cylindrical_fused_hd_pair.py``'s.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from meep_gpu import withdraw_hoist
from meep_gpu.cuda_kernels import complex_emitter, cylindrical_prefix
from meep_gpu.cuda_kernels import cylindrical_fused_hd_pair as family
from meep_gpu.cuda_kernels import fused_hd_pair
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid
from meep_gpu.pml import PML

_API_ROOT = Path(__file__).resolve().parents[1]
_BOARD = (_API_ROOT / "parity" / "meep_gpu" / "results" / "fusion_matrix_cuda_2026-09-06_hd"
          / "fusion_matrix_cuda.json")
_CELL = ("cuda_complex/complex", "cuda_cyl_complex/cylindrical complex")

#: A licence of the shape ``complex_fields.expansion_license`` returns, cut under keep.
_LICENCE = {"arm": "FMA_V1", "expansion": 1, "refusals": [], "basis": "measured",
            "policy": "ieee_keep_ftz_stripped", "policy_resolved": "keep"}


class _NumpyWearingCupysName:
    __name__ = "cupy"

    def __getattr__(self, item):
        return getattr(np, item)


def _dcyl(m: int = 1, complex_storage: bool = True, shape=(9, 1, 11), **kwargs):
    grid = Grid(resolution=1.0, cell_size=(float(shape[0]), 0.0, float(shape[2])),
                cylindrical=True, m=m, boundaries={"z": "metallic"}, courant=0.5,
                xp=_NumpyWearingCupysName(), **kwargs)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    return fields, PML(grid=grid, thickness={"x": (0, 2), "z": 2}), grid


# ---------------------------------------------------------------------------
# The declarations
# ---------------------------------------------------------------------------

def test_the_seam_span_slot_and_launch_count_are_the_real_siblings():
    assert family.SEAM == withdraw_hoist.SEAM
    assert family.REPLACES == withdraw_hoist.SEAM_SPAN == ("update_H", "step_D")
    assert family.SLOT == "update_H"
    assert family.LAUNCHES_PER_RUN == 3
    assert family.CARRIES_DEPOSIT_REPAIR is False
    assert family.HOISTS_THE_WITHDRAW is False
    assert family.INSTALLABLE is False
    assert "+2 launches" in family.INSTALLABLE_REASON
    assert family.PREFIX_SOURCE == cylindrical_prefix.PREFIX_COMPONENT["step_D"]
    assert family.PREFIX_IR0_VALUE == cylindrical_prefix.PREFIX_IR0["step_D"]
    assert family.SCRATCH_VOLUMES is fused_hd_pair.SCRATCH_VOLUMES


def test_the_kernel_is_in_exactly_one_set_and_launch_2_is_the_certified_complex_curl():
    assert family.KERNEL_NAME in family.CERTIFIED_KERNELS
    assert not family.UNCERTIFIED_KERNELS
    tree = ast.parse((Path(family.__file__).parent / "cylindrical_complex_kernels.py")
                     .read_text(encoding="utf-8"))
    certified = next(ast.literal_eval(node.value) for node in tree.body
                     if isinstance(node, (ast.Assign, ast.AnnAssign))
                     and getattr(getattr(node, "targets", [None])[0]
                                 if isinstance(node, ast.Assign) else node.target,
                                 "id", None) == "CERTIFIED_KERNELS")
    assert family.CURL_KERNEL_NAME == "cyl_step_D_pml_complex"
    assert family.CURL_KERNEL_NAME in certified


# ---------------------------------------------------------------------------
# The lift: the declared edits invert to the certified text
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm_name", sorted(complex_emitter.EXPANSIONS))
def test_the_four_prelude_edits_invert_to_the_certified_prelude(arm_name):
    arm = complex_emitter.EXPANSIONS[arm_name]
    certified = (complex_emitter._HEAD + complex_emitter._ARM_SOURCE[arm]  # noqa: SLF001
                 + complex_emitter._TAIL)  # noqa: SLF001
    emitted = family.constitutive_prelude(arm)
    assert emitted != certified
    inverted = emitted
    for pure, original in ((family._APPLY_F_STORE_PURE, family._APPLY_F_STORE),  # noqa: SLF001
                           (family._APPLY_FW_STORE_PURE, family._APPLY_FW_STORE),  # noqa: SLF001
                           (family._APPLY_PARAMETERS_PURE, family._APPLY_PARAMETERS),  # noqa: SLF001
                           (family._APPLY_SIGNATURE_PURE, family._APPLY_SIGNATURE)):  # noqa: SLF001
        assert inverted.count(pure) == 1
        inverted = inverted.replace(pure, original, 1)
    assert inverted == certified
    for entry in family.CONSTITUTIVE_LIFT_EDITS:
        assert set(entry) == {"line", "became", "why"} and entry["why"].strip()


@pytest.mark.parametrize("arm_name", sorted(complex_emitter.EXPANSIONS))
def test_the_certified_body_survives_the_capture_with_the_decode_kept(arm_name):
    arm = complex_emitter.EXPANSIONS[arm_name]
    prelude = (complex_emitter._HEAD + complex_emitter._ARM_SOURCE[arm]  # noqa: SLF001
               + complex_emitter._TAIL)  # noqa: SLF001
    body = complex_emitter.complex_source("update_H", arm)[len(prelude):]
    body = body.split("\n) {\n", 1)[1][: -len("}\n")]
    lifted = family.raw_update_H_cell_source(arm)
    assert "blockIdx.x" not in lifted
    assert "    int k = idx % nz;\n    int j = (idx / nz) % ny;\n    int i = idx / (ny * nz);\n" in lifted
    assert lifted.count("constitutive_apply_pure(") == 3
    assert "constitutive_apply(" not in lifted.replace("constitutive_apply_pure(", "")
    for line in body.splitlines():
        if line.strip() and "blockIdx" not in line and "return;" not in line \
                and "constitutive_apply(" not in line:
            assert line in lifted, line


def test_the_increment_multiply_is_the_fma_v1_mul_field_left_lifted_and_renamed():
    block = complex_emitter._ARM_SOURCE[complex_emitter.EXPANSIONS["FMA_V1"]]  # noqa: SLF001
    head = "__device__ __forceinline__ cf mul_field_left(cf z, float c) {\n"
    start = block.index(head)
    certified = block[start: block.index("}\n", start) + 2]
    lifted = family.mul_field_left_increment_source()
    assert certified.replace("mul_field_left(", "mul_field_left_increment(", 1) in lifted
    assert "__fmaf_rn(z.re, c, (z.im * 0.0f) * -1.0f)" in lifted
    assert family.INCREMENT_MULTIPLY_SOURCE_ARM == "FMA_V1"


def test_the_divide_is_the_measured_scaled_algorithm_with_every_zero_term_kept():
    divide = family.div_coefficient_source()
    for line in ("    float s = fabsf(d) + fabsf(0.0f);", "    float oos = 1.0f / s;",
                 "    float ars = z.re * oos;", "    float ais = z.im * oos;",
                 "    float brs = d * oos;", "    float bis = 0.0f * oos;",
                 "    s = (brs * brs) + (bis * bis);", "    oos = 1.0f / s;",
                 "    q.re = ((ars * brs) + (ais * bis)) * oos;",
                 "    q.im = ((ais * brs) - (ars * bis)) * oos;"):
        assert divide.count(line) == 1, line
    assert "z.re / d" not in divide and "z.re * r" not in divide


def test_the_host_is_not_the_oracle_for_the_complex_divide():
    """NumPy's ``complex64 / float32`` IS ``a * (float32(1)/d)``; the family's scaled
    spelling differs from NumPy on the host. The family carries CuPy's spelling because
    its oracle is CuPy (0 of 14,387,104 words on device), and its gate arms NumPy's."""
    rng = np.random.default_rng(3)
    rows, cols = 163, 175
    z = np.empty((rows, cols), dtype=np.complex64)
    z.real = rng.standard_normal((rows, cols)).astype(np.float32)
    z.imag = rng.standard_normal((rows, cols)).astype(np.float32)
    d = (np.arange(rows, dtype=np.float64) + 1.0).astype(np.float32).reshape(-1, 1)
    numpy_oracle = z / d
    reciprocal = z * (np.float32(1) / d)
    assert int(np.count_nonzero(reciprocal.view(np.uint32) != numpy_oracle.view(np.uint32))) == 0
    with np.errstate(all="ignore"):
        f32 = np.float32
        s = np.abs(d) + f32(0.0)
        oos = f32(1.0) / s
        ars, ais = z.real * oos, z.imag * oos
        brs, bis = d * oos, f32(0.0) * oos
        s2 = (brs * brs) + (bis * bis)
        oos2 = f32(1.0) / s2
        scaled = np.empty_like(z)
        scaled.real = ((ars * brs) + (ais * bis)) * oos2
        scaled.imag = ((ais * brs) - (ars * bis)) * oos2
    moved = int(np.count_nonzero(scaled.view(np.uint32) != numpy_oracle.view(np.uint32)))
    assert moved > 0, "the scaled spelling agreed with NumPy; the inversion would be gone"


def test_every_declared_increment_line_is_emitted_once_under_both_arms():
    for arm_name in sorted(complex_emitter.EXPANSIONS):
        source = family.kernel_source(arm_name)
        for entry in family.INCREMENT_SPELLING:
            assert set(entry) == {"line", "transcribes", "control"}
            assert source.count(entry["line"]) == 1, (arm_name, entry["line"])
        assert source.isascii()
        assert source.count("{") == source.count("}")
        assert source.count(f"void {family.KERNEL_NAME}(") == 1
        below = source.split("raw_update_H_cell(idx, weld, own_h, own_w);", 1)[1]
        for name in family.H_TARGETS:
            assert f"weld.{name}" not in below
        assert " / divisor" not in below and "1.0f / divisor" not in source
    assert family.kernel_source("NAIVE") != family.kernel_source("FMA_V1")
    assert sorted(family.device_sources()) == sorted(complex_emitter.EXPANSIONS)


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("m", (0, 1, 2, -3))
def test_the_predicate_admits_every_m_class_under_a_policy_matched_licence(m):
    fields, pml, grid = _dcyl(m=m)
    assert family.covers_cylindrical_fused_hd_pair(
        fields, pml, grid, (), _LICENCE, "keep") == (True, "covered")


def test_the_predicate_refuses_by_name():
    fields, pml, grid = _dcyl()
    for licence, policy, needle in ((None, "keep", "no expansion licence"),
                                    (_LICENCE, "flush", "cut under")):
        covered, reason = family.covers_cylindrical_fused_hd_pair(
            fields, pml, grid, (), licence, policy)
        assert not covered and needle in reason, reason
    covered, reason = family.covers_cylindrical_fused_hd_pair(
        fields, pml, grid, None, _LICENCE, "keep")
    assert not covered and "was not declared" in reason
    fields, pml, grid = _dcyl(m=0, complex_storage=False)
    covered, reason = family.covers_cylindrical_fused_hd_pair(
        fields, pml, grid, (), _LICENCE, "keep")
    assert not covered and "real float32" in reason
    grid = Grid(resolution=1.0, cell_size=(8.0, 8.0, 8.0), courant=0.5,
                boundaries=("periodic", "periodic", "periodic"), xp=_NumpyWearingCupysName())
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    covered, reason = family.covers_cylindrical_fused_hd_pair(
        fields, PML(grid=grid, thickness=2), grid, (), _LICENCE, "keep")
    assert not covered and "cylindrical" in reason


# ---------------------------------------------------------------------------
# The cell and the tables
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _BOARD.is_file(), reason=f"{_BOARD} is not in this checkout")
def test_the_cell_this_product_claims_is_the_boards_eighteen_rows():
    instances = json.loads(_BOARD.read_text(encoding="utf-8"))["h_to_d_seam"]["instances"]
    rows = [r for r in instances if (r["update_H"], r["step_D"]) == _CELL]
    assert len(rows) == 18, sorted(r["row"] for r in rows)
    withdraws = sorted(r["row"] for r in rows if r["withdraw_in_seam"])
    assert withdraws == ["examples:cylinder_cross_section.py", "examples:zone_plate.py"]
    assert sum(1 for r in rows if r["bucket"] == "buildable_not_built") == 16
    for text in ("18 seam-instances", "cylinder_cross_section.py", "zone_plate.py"):
        assert text in family.__doc__.replace("\n", " ")


def test_the_composer_carries_the_rows_this_product_needs_and_refuses_to_install_it():
    from meep_gpu.cuda_kernels import fused_pairs  # noqa: PLC0415

    assert fused_pairs.FUSED_PAIR_ARMS[family.FAMILY] == ("complex", "cylindrical complex")
    product = fused_pairs.FUSED_PRODUCTS[family.FAMILY]
    assert product["curl_slot"] == "update_H"
    assert fused_pairs.span_of(family.FAMILY, product) == family.REPLACES
    reason = fused_pairs._declared_uninstallable(family.FAMILY, product)  # noqa: SLF001
    assert reason is not None and family.FAMILY in reason
