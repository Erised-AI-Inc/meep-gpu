"""Merge-bar tests for the Metal COMPLEX radial scan (``cylindrical_complex_scan``).

WHAT THIS SUITE OWNS AND WHAT IT DOES NOT. The BYTES are the device gate's —
``parity/meep_gpu/gate_metal_cylindrical_complex_scan.py`` launches the kernel against
``stepping.cylindrical_rderiv_difference`` and compares uint32 words. What lives here is
everything true about this kernel WITHOUT a device, plus the smallest non-vacuous device
check.

THE ONE FACT THIS FILE EXISTS TO PIN: the complex scan IS the real scan with ``float2``
registers. The array path forms every step of MEEP's per-chunk radial sum per part
(``part_products``): the weighted term, the row difference, the product with the
multiplied ``rinv`` and the running sum are each one rounded real operation per part,
so the complex kernel's column loop is the real loop with the declared type edits and
nothing else.
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
from meep_gpu.metal_kernels import shaders

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
# 1. The loop is the real loop with float2 registers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sub_step", ("step_B", "step_D"))
def test_the_column_loop_is_the_real_loop_with_only_the_declared_type_edits(sub_step):
    """Undo :data:`FLOAT2_EDITS` and the real family's own loop remains, byte for byte."""
    loop = scan.column_scan_loop(sub_step)
    undone = loop
    for old, new in scan.FLOAT2_EDITS:
        undone = undone.replace(new, old)
    assert undone == cyl._SCAN_LOOPS[sub_step]
    assert "float2 acc = float2(0.0f, 0.0f);" in loop


@pytest.mark.parametrize("sub_step", ("step_B", "step_D"))
def test_the_scan_multiplies_rinv_and_never_divides(sub_step):
    """MEEP stores ``rinv`` and MULTIPLIES by it; the only ``/`` is the index decode."""
    code = _code_lines(scan.cylindrical_complex_scan_source(sub_step))
    divides = [line for line in code.splitlines() if "/" in line]
    assert divides == ["    int j = int(idx) / nzi;"], divides
    assert "rinv[g" in code and "fast::" not in code


def test_the_source_reads_are_the_ones_a_recomputing_leader_replaces():
    for sub_step in ("step_B", "step_D"):
        loop = scan.column_scan_loop(sub_step)
        for read in scan.source_reads(sub_step):
            assert loop.count(read) == 1, (sub_step, read)


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
    """``cylindrical_rderiv_difference`` builds its rows from ``f_p.real.dtype``, float32
    under complex64 — so the two families bind the SAME vectors."""
    for sub_step in ("step_B", "step_D"):
        weights, rinv = scan.prefix_row_vectors(sub_step, 23)
        rows = scan.scan_rows(sub_step, 23)
        expected_w, expected_r = stepping._cylindrical_rderiv_rows(
            np, rows, scan.PREFIX_IR0[sub_step], np.float32)
        assert weights.dtype == np.float32 and rinv.dtype == np.float32
        assert np.array_equal(weights.view(np.uint32),
                              np.ascontiguousarray(expected_w).reshape(-1).view(np.uint32))
        assert np.array_equal(rinv.view(np.uint32),
                              np.ascontiguousarray(expected_r).reshape(-1).view(np.uint32))


# ---------------------------------------------------------------------------
# 3. The structural contracts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sub_step", ("step_B", "step_D"))
def test_the_kernel_binds_nine_buffers_two_of_them_float2(sub_step):
    source = scan.cylindrical_complex_scan_source(sub_step)
    indices = [int(part.split(")")[0]) for part in source.split("[[buffer(")[1:]]
    assert indices == list(range(scan.BINDINGS)) == list(range(9))
    signature = source.split("kernel void", 1)[1].split("{", 1)[0]
    assert signature.count("device float2*") == 1
    assert signature.count("device const float2*") == 1
    assert signature.count("device const float*") == 2
    assert signature.count("device const int*") == 1


def test_scan_arguments_follow_the_signature_order():
    args = scan.scan_arguments("out", "src", "w", "r", "s", (18, 1, 7))
    assert args == ("out", "src", "w", "r", "s", 18, 1, 7, 7)


def test_the_contraction_directive_is_present_once_and_reached_through_templates():
    for source in scan.enumerate_cylindrical_complex_scan_sources().values():
        assert source.count(shaders.contraction_pragma(shaders.CONTRACT_OFF)) == 1
        assert "math_mode" not in source
    for source in scan.enumerate_cylindrical_complex_scan_sources(
            shaders.CONTRACT_FAST).values():
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


def test_the_two_sources_are_enumerated_and_the_gate_exists():
    sources = scan.enumerate_cylindrical_complex_scan_sources()
    assert set(sources) == {f"{scan.KERNEL}/step_B", f"{scan.KERNEL}/step_D"}
    assert len(set(sources.values())) == 2
    assert GATE.exists()


def test_the_weld_is_minted_and_nothing_is_declared_owed():
    """The gate-fleet partition reads ``WELD_OWED``; empty means the weld entry exists in
    ``fingerprints.json`` (its pins drift until the certification round re-mints it)."""
    assert scan.WELD_OWED == ""
    welded = json.loads((PACKAGE_DIR / "metal_kernels" / "fingerprints.json").read_text())
    assert f"metal_{scan.FAMILY}_device_gate" in welded


# ---------------------------------------------------------------------------
# 4. The reference, on the host: the complex loop is two real loops
# ---------------------------------------------------------------------------

def _divided_starts(sub_step, rows, cuts):
    """A per-row start line with chunks starting at ``cuts`` (the walk's own rule)."""
    line = np.zeros(rows, dtype=np.int64)
    for cut in cuts:
        first = cut + 1 if sub_step == "step_D" else cut
        line[first:] = cut
    return line.reshape(-1, 1, 1)


def _oracle(sub_step, source, starts):
    rows = source.shape[0]
    if sub_step == "step_B":
        extended = np.zeros((rows + 1,) + source.shape[1:], dtype=source.dtype)
        extended[:rows] = source
        return stepping.cylindrical_rderiv_difference(np, extended, 0.0, starts, rows,
                                                      backward=False)
    return stepping.cylindrical_rderiv_difference(np, source, 0.5, starts, rows,
                                                  backward=True)


@pytest.mark.parametrize("sub_step", ("step_B", "step_D"))
def test_the_complex_difference_is_the_real_difference_per_part(sub_step):
    """The array path's complex difference IS two real differences, part by part, at one,
    two and three chunks: the fact that licenses a float2 copy of the real loop."""
    rng = np.random.default_rng(11)
    shape = (19, 1, 5)
    source = (rng.uniform(-1.0, 1.0, shape) + 1j * rng.uniform(-1.0, 1.0, shape)
              ).astype(np.complex64)
    for cuts in ((), (8,), (5, 12)):
        starts = _divided_starts(sub_step, shape[0], cuts) if cuts else None
        whole = _oracle(sub_step, source, starts)
        real = _oracle(sub_step, np.ascontiguousarray(source.real), starts)
        imag = _oracle(sub_step, np.ascontiguousarray(source.imag), starts)
        assert np.array_equal(np.ascontiguousarray(whole.real).view(np.uint32),
                              real.view(np.uint32)), cuts
        assert np.array_equal(np.ascontiguousarray(whole.imag).view(np.uint32),
                              imag.view(np.uint32)), cuts


# ---------------------------------------------------------------------------
# 5. The smallest non-vacuous device check
# ---------------------------------------------------------------------------

@requires_mps
def test_the_scan_reproduces_the_array_paths_difference_and_one_sum_does_not():
    from meep_gpu.metal_kernels.device import compile_source

    import torch

    rng = np.random.default_rng(2026)

    def upload(array, dtype=None):
        return torch.from_numpy(np.ascontiguousarray(array, dtype=dtype).reshape(-1)
                                ).to("mps")

    for sub_step in ("step_B", "step_D"):
        shape = (17, 1, 9)
        source = (rng.uniform(-1.0, 1.0, shape) + 1j * rng.uniform(-1.0, 1.0, shape)
                  ).astype(np.complex64)
        weights, rinv = scan.prefix_row_vectors(sub_step, shape[0])
        entry = getattr(compile_source(
            scan.cylindrical_complex_scan_source(sub_step)), scan.KERNEL)
        sentinel = np.full(shape, -7.5 - 7.5j, dtype=np.complex64)
        for cuts in ((), (6,), (4, 11)):
            starts = _divided_starts(sub_step, shape[0], cuts) if cuts else None
            expected = _oracle(sub_step, source, starts)
            line = cyl.scan_starts(sub_step, starts, shape[0])
            out = upload(sentinel)
            entry(*scan.scan_arguments(out, upload(source), upload(weights, np.float32),
                                       upload(rinv, np.float32), upload(line, np.int32),
                                       shape))
            torch.mps.synchronize()
            got = out.cpu().numpy().reshape(expected.shape)
            assert np.count_nonzero(got.view(np.uint32) != sentinel.view(np.uint32)) > 0
            differing = int(np.count_nonzero(got.view(np.uint32) != expected.view(np.uint32)))
            assert differing == 0, (sub_step, cuts, differing)
            if cuts:
                single = _oracle(sub_step, source, None)
                assert np.count_nonzero(got.view(np.uint32) != single.view(np.uint32)) > 0


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
