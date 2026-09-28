"""Merge-bar tests for the Metal COMPLEX radial prefix scan (``cylindrical_complex_scan``).

WHAT THIS SUITE OWNS AND WHAT IT DOES NOT. The BYTES are the device gate's —
``parity/meep_gpu/gate_metal_cylindrical_complex_scan.py`` launches the kernel at every
corpus radial extent against ``stepping.cylindrical_rderiv_prefix`` and compares uint32
words. What lives here is everything true about this kernel WITHOUT a device, plus the
smallest non-vacuous device check, plus the gate's MEASURED COUNTS read from its artifact
so a re-cut that quietly changed them fails a test rather than passing unnoticed.

THE ONE FACT THIS FILE EXISTS TO PIN: the divide spelling INVERTS between the real and
the complex scan. The real kernel spells ``/`` and its own comment insists on it; numpy's
``complex64 / float32`` is its complex divide loop, which on every word that reaches the
prefix is the RECIPROCAL MULTIPLY ``z * (1.0f / d)``. A reader porting the real template
carries the ``/`` over and is wrong on ~90% of prefix words. The spelling is pinned as
source text here and armed as a mutation in the gate.
"""

from __future__ import annotations

import ast
import json
import os
import pathlib

import numpy as np
import pytest

from meep_gpu import stepping
from meep_gpu.metal_kernels import cylindrical_complex_scan as scan
from meep_gpu.metal_kernels import cylindrical_real as cyl
from meep_gpu.metal_kernels import shaders, templates

PACKAGE_DIR = pathlib.Path(__file__).resolve().parent
REPO = PACKAGE_DIR.parent
MODULE = PACKAGE_DIR / "metal_kernels" / "cylindrical_complex_scan.py"
GATE = REPO / "parity" / "meep_gpu" / "gate_metal_cylindrical_complex_scan.py"
#: The RELEASED cut. ``_cyl``, ``_cyl2`` and ``_cyl3`` beside it are earlier cuts refused
#: on gate-harness facts (a helper-definition miscount, a needle that renamed that
#: definition, a wall-row mutation scored over classes that absorb its one-word edit);
#: the kernel bytes are identical in all four.
ARTIFACT = (REPO / "parity" / "meep_gpu" / "results"
            / "metal_cylindrical_complex_scan_2026-09-06_cyl4" / "gate.json")

EXPANSIONS = tuple(templates.EXPANSION_ARMS)


def _has_mps() -> bool:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


requires_mps = pytest.mark.skipif(not _has_mps(), reason="no MPS device")


def _code_lines(source: str) -> str:
    """The source with ``//`` comment text removed — what the COMPILER sees."""
    kept = []
    for line in source.splitlines():
        head = line.split("//", 1)[0]
        if head.strip():
            kept.append(head)
    return "\n".join(kept)


# ---------------------------------------------------------------------------
# 1. The spelling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sub_step", ("step_B", "step_D"))
@pytest.mark.parametrize("expansion", EXPANSIONS)
def test_the_divide_is_the_reciprocal_multiply_and_never_the_true_divide(sub_step,
                                                                          expansion):
    """The INVERSE of ``test_metal_cylindrical_real``'s pin, for the reason the module
    docstring gives: numpy's complex64 / float32 is its complex divide loop."""
    code = _code_lines(scan.cylindrical_complex_prefix_source(sub_step, expansion))
    assert "return z * (1.0f / d);" in code
    assert "c_div_real(w - prev, divisor[i - 1])" in code
    assert "/ divisor" not in code, "the real scan's `/` carried onto the complex increment"
    assert "z / d" not in code
    assert "fast::" not in code


@pytest.mark.parametrize("sub_step", ("step_B", "step_D"))
def test_the_weight_multiply_keeps_the_field_on_the_left(sub_step):
    """``f_p * weights`` (stepping.py:1313, :1327): field LEFT, through the certified
    field-left helper; the coefficient-left helper is defined but never called on it."""
    code = _code_lines(scan.cylindrical_complex_prefix_source(sub_step, "FMA_V1"))
    src0, srci = scan.source_reads(sub_step)
    assert f"c_mul_field_left({src0}, weights[0])" in code
    assert f"c_mul_field_left({srci}, weights[i])" in code
    assert "c_mul_coefficient_left(w" not in code
    assert "c_mul_coefficient_left(src" not in code


def test_row_zero_is_an_exact_positive_complex_zero_and_the_sum_is_serial():
    code = _code_lines(scan.cylindrical_complex_prefix_source("step_D", "FMA_V1"))
    assert "float2 acc = float2(0.0f, 0.0f);" in code
    assert "out[base] = acc;" in code
    assert "acc = acc + inc;" in code
    assert "acc = inc + acc;" not in code


def test_the_wall_row_is_a_source_specialisation_on_the_b_side_only():
    b_code = _code_lines(scan.cylindrical_complex_prefix_source("step_B", "FMA_V1"))
    d_code = _code_lines(scan.cylindrical_complex_prefix_source("step_D", "FMA_V1"))
    assert "i < nxi - 1 ? src[i * nyz + base] : float2(0.0f, 0.0f)" in b_code
    assert "i < nxi - 1" not in d_code
    assert scan.scan_rows("step_B", 17) == 18 and scan.scan_rows("step_D", 17) == 17


# ---------------------------------------------------------------------------
# 2. The tables are the real family's, imported
# ---------------------------------------------------------------------------

def test_the_ir0_and_wall_row_tables_are_the_real_familys_own_objects():
    assert scan.PREFIX_IR0 is cyl.PREFIX_IR0
    assert scan.PREFIX_WALL_ROW is cyl.PREFIX_WALL_ROW
    assert scan.prefix_row_vectors is cyl.prefix_row_vectors
    assert scan.PREFIX_IR0 == {"step_B": 0.0, "step_D": 0.5}
    assert scan.PREFIX_WALL_ROW == {"step_B": True, "step_D": False}


def test_the_row_vectors_are_the_float32_ladders_complex_storage_binds():
    """``cylindrical_rderiv_prefix`` builds the weights from ``f_p.real.dtype``, which
    is float32 under complex64 — so the two families bind the SAME vectors."""
    for sub_step in ("step_B", "step_D"):
        weights, divisor = scan.prefix_row_vectors(sub_step, 23)
        rows = scan.scan_rows(sub_step, 23)
        expected_w, expected_d = stepping._cylindrical_rderiv_weights(
            np, rows, scan.PREFIX_IR0[sub_step], np.float32)
        assert weights.dtype == np.float32 and divisor.dtype == np.float32
        assert np.array_equal(weights.view(np.uint32),
                              np.ascontiguousarray(expected_w).reshape(-1).view(np.uint32))
        assert np.array_equal(divisor.view(np.uint32),
                              np.ascontiguousarray(expected_d).reshape(-1).view(np.uint32))


# ---------------------------------------------------------------------------
# 3. The structural contracts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sub_step", ("step_B", "step_D"))
def test_the_kernel_binds_eight_buffers_two_of_them_float2(sub_step):
    source = scan.cylindrical_complex_prefix_source(sub_step, "FMA_V1")
    indices = [int(part.split(")")[0]) for part in source.split("[[buffer(")[1:]]
    assert indices == list(range(scan.BINDINGS)) == list(range(8))
    signature = source.split("kernel void", 1)[1].split("{", 1)[0]
    assert signature.count("device float2*") == 1
    assert signature.count("device const float2*") == 1
    assert signature.count("device const float*") == 2


def test_scan_arguments_follow_the_signature_order():
    args = scan.scan_arguments("out", "src", "w", "d", (18, 1, 7))
    assert args == ("out", "src", "w", "d", 18, 1, 7, 7)


def test_the_contraction_directive_is_present_once_and_reached_through_templates():
    for expansion in EXPANSIONS:
        for source in scan.enumerate_cylindrical_complex_scan_sources(expansion).values():
            assert source.count(shaders.contraction_pragma(shaders.CONTRACT_OFF)) == 1
            assert "math_mode" not in source
        for source in scan.enumerate_cylindrical_complex_scan_sources(
                expansion, shaders.CONTRACT_FAST).values():
            assert source.count(shaders.contraction_pragma(shaders.CONTRACT_FAST)) == 1
    text = MODULE.read_text(encoding="utf-8")
    assert "pragma" not in text.replace("contraction_pragma", "")


def test_the_module_imports_no_torch_at_module_scope_and_registers_no_arm():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] != "torch" for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("torch")
    assert "def register_arms(" not in MODULE.read_text(encoding="utf-8")


def test_the_two_sources_per_arm_are_distinct_and_enumerated():
    for expansion in EXPANSIONS:
        sources = scan.enumerate_cylindrical_complex_scan_sources(expansion)
        assert set(sources) == {f"{scan.KERNEL}/step_B/{expansion}",
                                f"{scan.KERNEL}/step_D/{expansion}"}
        assert len(set(sources.values())) == 2


def test_the_weld_is_minted_and_nothing_is_declared_owed():
    """WIRED 2026-09-06: the gate-fleet partition reads ``WELD_OWED``; empty means the
    weld entry exists in ``fingerprints.json``, minted from the released artifact."""
    assert scan.WELD_OWED == ""
    welded = json.loads((PACKAGE_DIR / "metal_kernels" / "fingerprints.json").read_text())
    assert f"metal_{scan.FAMILY}_device_gate" in welded


# ---------------------------------------------------------------------------
# 4. The reference, measured on the host (the finding behind the spelling)
# ---------------------------------------------------------------------------

def test_numpys_complex_by_real_divide_is_the_reciprocal_multiply_on_random_data():
    rng = np.random.default_rng(11)
    z = (rng.standard_normal(4096) + 1j * rng.standard_normal(4096)).astype(np.complex64)
    d = np.float32(3.0)
    recip = np.float32(1) / d
    q = z / d
    expected = np.empty_like(z)
    expected.real = z.real * recip
    expected.imag = z.imag * recip
    componentwise = np.empty_like(z)
    componentwise.real = z.real / d
    componentwise.imag = z.imag / d
    assert np.array_equal(q.view(np.uint32), expected.view(np.uint32))
    assert np.count_nonzero(q.view(np.uint32) != componentwise.view(np.uint32)) > 1000


def test_the_signed_zero_refinement_is_real_on_the_increment_and_absorbed_by_the_prefix():
    """numpy's loop differs from the reciprocal multiply on the SIGN of exactly-zero
    increments; the running sum from +0.0 never stores that sign."""
    rng = np.random.default_rng(3)
    shape = (64, 1, 9)
    pick = rng.integers(0, 4, shape)
    plane = np.where(pick == 0, np.float32(0.0),
                     np.where(pick == 1, np.float32(-0.0),
                              rng.standard_normal(shape).astype(np.float32)))
    source = np.empty(shape, dtype=np.complex64)
    source.real = plane.astype(np.float32)
    source.imag = np.roll(plane, 3, axis=0).astype(np.float32)
    weights, divisor = stepping._cylindrical_rderiv_weights(np, shape[0], 0.5, np.float32)
    weighted = source * weights
    numpy_inc = np.zeros_like(source)
    numpy_inc[1:] = (weighted[1:] - weighted[:-1]) / divisor
    recip = np.float32(1) / divisor
    recip_inc = np.zeros_like(source)
    diff = weighted[1:] - weighted[:-1]
    recip_inc.real[1:] = (diff.real * recip).astype(np.float32)
    recip_inc.imag[1:] = (diff.imag * recip).astype(np.float32)
    increments_differ = np.count_nonzero(numpy_inc.view(np.uint32)
                                         != recip_inc.view(np.uint32))
    prefixes_differ = np.count_nonzero(np.cumsum(numpy_inc, axis=0).view(np.uint32)
                                       != np.cumsum(recip_inc, axis=0).view(np.uint32))
    assert increments_differ > 0, "the refinement must be real or the claim is vacuous"
    assert prefixes_differ == 0
    oracle = stepping.cylindrical_rderiv_prefix(np, source, 0.5)
    assert np.array_equal(np.cumsum(numpy_inc, axis=0).view(np.uint32),
                          oracle.view(np.uint32))


# ---------------------------------------------------------------------------
# 5. The smallest non-vacuous device check
# ---------------------------------------------------------------------------

@requires_mps
@pytest.mark.parametrize("expansion", EXPANSIONS)
def test_the_scan_reproduces_steppings_own_prefix_and_the_true_divide_does_not(expansion):
    from meep_gpu.metal_kernels.device import compile_source

    import torch

    rng = np.random.default_rng(2026)
    for sub_step in ("step_B", "step_D"):
        shape = (13, 1, 9)
        source = (rng.uniform(-1.0, 1.0, shape) + 1j * rng.uniform(-1.0, 1.0, shape)
                  ).astype(np.complex64)
        rows = scan.scan_rows(sub_step, shape[0])
        scanned = source
        if scan.PREFIX_WALL_ROW[sub_step]:
            scanned = np.zeros((rows,) + shape[1:], dtype=np.complex64)
            scanned[: shape[0]] = source
        expected = stepping.cylindrical_rderiv_prefix(np, scanned, scan.PREFIX_IR0[sub_step])
        weights, divisor = scan.prefix_row_vectors(sub_step, shape[0])

        def upload(array, dtype=None):
            return torch.from_numpy(np.ascontiguousarray(array, dtype=dtype).reshape(-1)
                                    ).to("mps")

        shipped_source = scan.cylindrical_complex_prefix_source(sub_step, expansion)
        sentinel = np.full((rows,) + shape[1:], -7.5 - 7.5j, dtype=np.complex64)
        for label, text in (("shipped", shipped_source),
                            ("true_divide", shipped_source.replace(
                                "return z * (1.0f / d);", "return z / d;"))):
            assert text != shipped_source or label == "shipped"
            entry = getattr(compile_source(text), scan.KERNEL)
            out = upload(sentinel)
            entry(*scan.scan_arguments(out, upload(source), upload(weights, np.float32),
                                       upload(divisor, np.float32), (rows,) + shape[1:]))
            torch.mps.synchronize()
            got = out.cpu().numpy().reshape(expected.shape)
            # A sentinel fill, so a kernel that never ran cannot compare equal.
            assert np.count_nonzero(got.view(np.uint32) != sentinel.view(np.uint32)) > 0
            differing = int(np.count_nonzero(got.view(np.uint32) != expected.view(np.uint32)))
            if label == "shipped":
                assert differing == 0, (sub_step, expansion, differing)
            else:
                assert differing > expected.size // 2, (
                    f"{sub_step}: the true-divide control did not bite ({differing})")


# ---------------------------------------------------------------------------
# 6. The measured counts, read from the released artifact
# ---------------------------------------------------------------------------

def _artifact():
    if not ARTIFACT.exists():
        pytest.skip(f"{ARTIFACT} is not on this host")
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_the_gate_exists_and_the_released_artifact_says_it_released():
    assert GATE.exists()
    record = _artifact()
    assert record["summary"]["status"] == "passed"
    assert record["release"]["released"] is True
    assert record["weld"]["passed"] is True


def test_the_artifact_measured_the_corpus_extents_at_both_arms_with_zero_differing():
    record = _artifact()
    summary = record["legs"]["prefix_summary"]
    assert summary["cases"] == 104
    assert summary["extents"] == 13
    assert summary["value_classes"] == ["uniform", "wide_dynamic", "cancelling",
                                        "signed_zero_mix"]
    assert summary["compared"] == 11_097_288
    assert summary["differing"] == 0
    assert summary["differing_other_arms"] == {"NAIVE": 0}
    assert summary["measured_expansion"] == "FMA_V1" == summary["probe_bound_expansion"]
    assert record["legs"]["corpus_shapes"]["checked"] == 18
    assert record["legs"]["corpus_shapes"]["mismatches"] == []
    assert record["summary"]["comparisons"] == 25_928_888


def test_the_artifact_caught_the_true_divide_and_measured_the_equivalences():
    record = _artifact()
    verdicts = record["summary"]["mutation_verdicts"]
    assert verdicts["true_divide_spelling"] == "CAUGHT 104/104"
    assert verdicts["blocked_scan_32"] == "CAUGHT 104/104"
    assert verdicts["weights_off_by_one"] == "CAUGHT 104/104"
    assert verdicts["divisor_off_by_one"] == "CAUGHT 104/104"
    assert verdicts["negative_zero_row_zero"] == "CAUGHT 104/104"
    assert verdicts["wall_row_dropped"] == "CAUGHT 26/26"
    for null in ("literal_numpy_divide_loop", "componentwise_weight_multiply",
                 "coefficient_left_orientation", "commuted_accumulator"):
        assert verdicts[null] == "CAUGHT 0/104", null
    # The contraction guard is LOAD-BEARING on this kernel — the multiply-add fuses.
    assert verdicts["contraction_fast"] == "CAUGHT 104/104"
    rows = {row["mutation"]: row for row in record["legs"]["mutations"]}
    assert rows["wall_row_dropped"]["caught_by_class"] == {
        "uniform": 13, "cancelling": 13, "wide_dynamic": 11, "signed_zero_mix": 12}
    assert all("nz1" in label for label in
               rows["wall_row_dropped"]["unscoped_cases_not_caught"])


def test_the_artifact_recorded_the_absorption_and_the_increment_refinement():
    record = _artifact()
    absorption = record["legs"]["absorption"]
    assert len(absorption) == 26
    assert all(row["reference_all_positive_zero"] and row["differing"] == 0
               for row in absorption)
    summary = record["legs"]["increment_sign_summary"]
    assert summary["increment_differing"]["reciprocal"] > 0
    assert summary["increment_differing"]["componentwise_multiply"] > 0
    assert summary["prefix_differing"] == {"reciprocal": 0, "componentwise_multiply": 0}
    assert record["legs"]["disarm"] == {"cases": 104, "differing": 0, "launches": 104,
                                        "passed": True}
