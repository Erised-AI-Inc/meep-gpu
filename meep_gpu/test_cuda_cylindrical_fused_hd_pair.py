"""Host tests for the CUDA complex Dcyl H->D product (three launches).

Everything here runs without a device: ``complex_emitter``, ``complex_fused_hd_pair``'s
pure helpers and this family's emitter are CuPy-free, so launch 1's text is testable at
the merge bar -- it is the imported pure prelude, argument pack and cell function plus
the scratch stores, ASCII, one kernel, one body under both arm names. The predicate is
exercised on a NumPy-wearing-CuPy shim with a licence of the shape the arbiter returns.
The word-for-word sequence (launch 1, the scan over the scratch Hp per stored part, the
rotation, launch 3) runs in a host interpreter in
``cuda_kernels/test_cylindrical_chunk_forms.py``.

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
from meep_gpu.cuda_kernels import cylindrical_prefix
from meep_gpu.cuda_kernels import cylindrical_fused_hd_pair as family
from meep_gpu.cuda_kernels import complex_fused_hd_pair, fused_hd_pair
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid
from meep_gpu.pml import PML

_API_ROOT = Path(__file__).resolve().parents[1]
_BOARD = (_API_ROOT / "parity" / "meep_gpu" / "results" / "fusion_matrix_cuda_2026-09-06_hd"
          / "fusion_matrix_cuda.json")
_CELL = ("cuda_complex/complex", "cuda_cyl_complex/cylindrical complex")



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
    for phrase in ("no launch saved", "three launches", "owner's design ruling",
                   "predicate admission"):
        assert phrase in family.INSTALLABLE_REASON, phrase
    assert family.PREFIX_SOURCE == cylindrical_prefix.PREFIX_COMPONENT["step_D"]
    assert family.SCRATCH_VOLUMES is fused_hd_pair.SCRATCH_VOLUMES


def test_the_kernel_is_in_exactly_one_set_and_launch_2_is_the_certified_complex_curl():
    # NEW with the per-chunk radial sum: UNCERTIFIED, by name and gate, until the
    # certification round re-cuts its predecessor's block.
    assert family.KERNEL_NAME in family.UNCERTIFIED_KERNELS
    assert family.KERNEL_NAME not in family.CERTIFIED_KERNELS
    assert "gate_cuda_cylindrical_fused_hd_pair.py" in family.UNCERTIFIED_KERNELS[
        family.KERNEL_NAME]
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
# Launch 1's text: imported, never retyped
# ---------------------------------------------------------------------------

def test_launch_1_is_the_pure_complex_cell_function_plus_the_scratch_stores():
    """The pure prelude, the argument pack and the cell function are
    :mod:`.complex_fused_hd_pair`'s, imported; this family adds the signature and the
    six scratch stores, at the thread's OWN cell only."""
    source = family.kernel_source()
    composed = (complex_fused_hd_pair.constitutive_prelude("plain")
                + complex_fused_hd_pair.weld_args_struct()
                + complex_fused_hd_pair.raw_update_H_cell_source()
                + family.signature())
    assert source.startswith(composed)
    assert source.endswith("    cf_store(f_w_Hz_out, idx, own_w[2]);\n}\n")
    assert source.count("raw_update_H_cell(idx") == 1
    assert "idx - nyz" not in source


def test_launch_1_is_ascii_and_one_kernel():
    """Pure ASCII is an NVRTC requirement, and no H target is read below the own-cell
    store."""
    source = family.kernel_source()
    assert source.isascii()
    assert source.count("{") == source.count("}")
    assert source.count(f"void {family.KERNEL_NAME}(") == 1
    below = source.split("raw_update_H_cell(idx, weld, own_h, own_w);", 1)[1]
    for target in family.H_TARGETS:
        assert f"weld.{target}" not in below, target
    assert "divisor[" not in source and "cumsum(" not in source
    assert sorted(family.device_sources()) == [family.KERNEL_NAME]


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("m", (0, 1, 2, -3))
def test_the_predicate_admits_every_m_class_under_a_policy_matched_licence(m):
    fields, pml, grid = _dcyl(m=m)
    assert family.covers_cylindrical_fused_hd_pair(
        fields, pml, grid, ()) == (True, "covered")


def test_the_predicate_refuses_by_name():
    fields, pml, grid = _dcyl()
    covered, reason = family.covers_cylindrical_fused_hd_pair(
        fields, pml, grid, None)
    assert not covered and "was not declared" in reason
    fields, pml, grid = _dcyl(m=0, complex_storage=False)
    covered, reason = family.covers_cylindrical_fused_hd_pair(
        fields, pml, grid, ())
    assert not covered and "real float32" in reason
    grid = Grid(resolution=1.0, cell_size=(8.0, 8.0, 8.0), courant=0.5,
                boundaries=("periodic", "periodic", "periodic"), xp=_NumpyWearingCupysName())
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    covered, reason = family.covers_cylindrical_fused_hd_pair(
        fields, PML(grid=grid, thickness=2), grid, ())
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
