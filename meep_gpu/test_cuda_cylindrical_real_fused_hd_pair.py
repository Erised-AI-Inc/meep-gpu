"""Host tests for the CUDA real m = 0 Dcyl H->D product (three launches).

WHAT CAN BE TESTED WITHOUT A DEVICE: the declarations, the seam facts, the predicate on
a NumPy-wearing-CuPy shim, and the scan tables the launcher binds for the D side. The
word-for-word sequence -- launch 1 into scratch, the column-serial scan over the scratch
Hp, the rotation, launch 3 -- runs in a host interpreter beside ``stepping``'s own
sequence in ``cuda_kernels/test_cylindrical_chunk_forms.py``. The emitter needs the
certified device strings, which import CuPy at scope; those tests declare the resource
and skip.

Device identity is ``parity/meep_gpu/gate_cuda_cylindrical_real_fused_hd_pair.py``'s.
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
from meep_gpu.cuda_kernels import cylindrical_real_fused_hd_pair as family
from meep_gpu.cuda_kernels import cylindrical_fused_hd_pair as complex_sibling
from meep_gpu.cuda_kernels import fused_hd_pair
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid
from meep_gpu.pml import PML

_API_ROOT = Path(__file__).resolve().parents[1]
_BOARD = (_API_ROOT / "parity" / "meep_gpu" / "results" / "fusion_matrix_cuda_2026-09-06_hd"
          / "fusion_matrix_cuda.json")
_CELL = ("cuda_constitutive/ordinary", "cuda_cylindrical/cylindrical")

_HAS_CERTIFIED_TEXT = not (family.constitutive_kernels is None
                           or family.cylindrical_kernels is None)
_emitter = pytest.mark.requires_resource("cuda-certified-device-text")


def _needs_certified_text() -> None:
    if not _HAS_CERTIFIED_TEXT:
        pytest.skip("[requires_resource][cuda-certified-device-text] constitutive_kernels / "
                    "cylindrical_kernels import CuPy at module scope")


class _NumpyWearingCupysName:
    __name__ = "cupy"

    def __getattr__(self, item):
        return getattr(np, item)


def _dcyl(m: int = 0, complex_storage: bool = False, shape=(9, 1, 11), **kwargs):
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

def test_the_seam_is_the_withdraw_hoists_and_the_span_is_the_measured_one():
    assert family.SEAM == withdraw_hoist.SEAM
    assert family.REPLACES == withdraw_hoist.SEAM_SPAN == ("update_H", "step_D")
    assert family.SLOT == "update_H"
    assert family.CARRIES_DEPOSIT_REPAIR is False
    assert family.HOISTS_THE_WITHDRAW is False
    assert withdraw_hoist._span_reasons(family.REPLACES) == ()  # noqa: SLF001


def _executable(module) -> str:
    """A module's code with comments and docstrings removed (``code_identity``'s rule)."""
    from meep_gpu.code_identity import strip_docstrings  # noqa: PLC0415

    return ast.unparse(strip_docstrings(ast.parse(
        Path(module.__file__).read_text(encoding="utf-8"))))


def test_three_launches_per_run_and_the_scan_is_counted_between_two_kernels():
    """Launch 1, the column-serial scan, launch 3: the scan is a LAUNCH, counted, and no
    host prefix pass survives."""
    assert family.LAUNCHES_PER_RUN == 3
    code = _executable(family)
    assert "launch_radial_scan(scan_kernel or get_radial_scan_kernel()" in code
    assert "scan_launches" in code
    # Call and index spellings, not bare words: the declarations' prose names the
    # 0.9.2 increment it no longer fuses.
    for gone in (".cumsum(", "divisor[", "/ divisor", "increment_source("):
        assert gone not in code, gone


def test_the_prefix_source_and_the_scan_side_are_read_off_the_shared_table():
    assert family.PREFIX_SOURCE == cylindrical_prefix.PREFIX_COMPONENT["step_D"] == "Hy"
    assert cylindrical_prefix.RADIAL_TARGET["step_D"] == "Dz"
    assert cylindrical_prefix.RADIAL_BACKWARD["step_D"] is True
    assert cylindrical_prefix.PREFIX_WALL_ROW["step_D"] is False
    assert "radial_scan_tables(fields, pml, 'step_D')" in _executable(family)


def test_the_product_declares_itself_uninstallable_on_a_cylindrical_launch_algebra():
    assert family.INSTALLABLE is False
    for phrase in ("no launch saved", "three launches", "column-serial scan",
                   "owner's design ruling", "predicate admission"):
        assert phrase in family.INSTALLABLE_REASON, phrase
    assert "PREDICATE ADMISSION" in family.WHAT_A_RELEASE_DOES_NOT_LICENSE
    assert "no timing" in family.WHAT_A_RELEASE_DOES_NOT_LICENSE.lower()


def test_the_kernel_is_in_exactly_one_certification_set_and_the_curl_is_the_certified_one():
    # NEW with the per-chunk radial sum: no record names it until the certification
    # round re-cuts its predecessor's block, so it is UNCERTIFIED, by name and gate.
    assert family.KERNEL_NAME in family.UNCERTIFIED_KERNELS
    assert family.KERNEL_NAME not in family.CERTIFIED_KERNELS
    assert family.KERNEL_NAME not in fused_hd_pair.CERTIFIED_KERNELS
    assert "gate_cuda_cylindrical_real_fused_hd_pair.py" in family.UNCERTIFIED_KERNELS[
        family.KERNEL_NAME]
    # The curl launch 2 runs is the CERTIFIED cylindrical kernel, read off that
    # module's syntax tree (it imports CuPy at scope).
    tree = ast.parse((Path(family.__file__).parent / "cylindrical_kernels.py")
                     .read_text(encoding="utf-8"))
    certified = next(ast.literal_eval(node.value) for node in tree.body
                     if isinstance(node, (ast.Assign, ast.AnnAssign))
                     and getattr(getattr(node, "targets", [None])[0]
                                 if isinstance(node, ast.Assign) else node.target,
                                 "id", None) == "CERTIFIED_KERNELS")
    assert family.CURL_KERNEL_NAME == "cyl_step_D_pml_real"
    assert family.CURL_KERNEL_NAME in certified


def test_the_scratch_volumes_and_targets_are_the_hd_welds_own():
    assert family.SCRATCH_VOLUMES is fused_hd_pair.SCRATCH_VOLUMES
    assert family.H_TARGETS is fused_hd_pair.H_TARGETS


# ---------------------------------------------------------------------------
# The scan the launcher binds
# ---------------------------------------------------------------------------

def test_the_scan_tables_are_the_d_sides():
    """Hp's stored rows with no wall row, Dz's rows as the target, a BACKWARD
    difference, and one start per target row."""
    fields, pml, _grid = _dcyl(shape=(16, 1, 20))
    tables = cylindrical_prefix.radial_scan_tables(fields, pml, "step_D")
    rows = int(fields.Hy.shape[0])
    assert tables["rows_stored"] == tables["rows_src"] == rows
    assert tables["n_target"] == int(fields.Dz.shape[0])
    assert tables["backward"] is True
    assert np.asarray(tables["starts"]).shape == (tables["n_target"],)
    assert np.asarray(tables["w"]).shape == (rows,)


def test_both_storages_run_the_one_scan_kernel():
    """The 0.9.2 increments diverged by storage (``/`` here, a scaled complex divide on
    the sibling). The per-chunk sum is one column-serial kernel over float32 words -- a
    complex64 volume is two columns per cell -- so both families launch the same text."""
    for module in (family, complex_sibling):
        code = _executable(module)
        assert "get_radial_scan_kernel()" in code and "launch_radial_scan(" in code
        assert "div_coefficient(" not in code and ".cumsum(" not in code


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------

def test_the_predicate_admits_a_real_m0_dcyl_run_and_refuses_the_rest_by_name():
    fields, pml, grid = _dcyl()
    assert family.covers_cylindrical_real_fused_hd_pair(fields, pml, grid, ()) == (True, "covered")
    covered, reason = family.covers_cylindrical_real_fused_hd_pair(fields, pml, grid, None)
    assert not covered and "was not declared" in reason
    covered, reason = family.covers_cylindrical_real_fused_hd_pair(fields, None, grid, ())
    assert not covered and "no active PML" in reason
    fields, pml, grid = _dcyl(complex_storage=True)
    covered, reason = family.covers_cylindrical_real_fused_hd_pair(fields, pml, grid, ())
    assert not covered and reason.startswith("constitutive half") and "complex64" in reason
    fields, pml, grid = _dcyl(m=1, complex_storage=True)
    covered, reason = family.covers_cylindrical_real_fused_hd_pair(fields, pml, grid, ())
    assert not covered


def test_the_predicate_refuses_a_cartesian_grid_through_the_curl_half():
    grid = Grid(resolution=1.0, cell_size=(8.0, 8.0, 8.0), courant=0.5,
                boundaries=("periodic", "periodic", "periodic"), xp=_NumpyWearingCupysName())
    fields = Fields(grid=grid)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    covered, reason = family.covers_cylindrical_real_fused_hd_pair(
        fields, PML(grid=grid, thickness=2), grid, ())
    assert not covered and reason.startswith("curl half") and "cylindrical" in reason


def test_an_integrated_electric_source_stands_no_withdraw_in_the_seam():
    """An engine integrated source enters at the constitutive update through
    ``f_minus_p`` (``sources.f_minus_p_begin``), so nothing stands between the two
    consults this product spans."""
    from meep_gpu.sources import ContinuousEnvelope, VolumeSource  # noqa: PLC0415

    fields, pml, grid = _dcyl()
    source = VolumeSource(grid=grid, component="Ez", center=(0.0, 0.0, 0.0),
                          size=(0.0, 0.0, 0.0),
                          envelope=ContinuousEnvelope(frequency=1.0, is_integrated=True))
    assert source.is_integrated and not hasattr(source, "withdraw")
    assert family.covers_cylindrical_real_fused_hd_pair(fields, pml, grid, [source]) == (
        True, "covered")


def test_a_foreign_standing_withdraw_is_refused_by_name():
    """A source class that brings its own integrated electric ``withdraw`` is refused:
    nothing measured a fused launch beside its body."""

    class _ForeignElectric:
        def withdraw(self, fields):  # never called: a predicate reads a source
            raise AssertionError("a coverage predicate called a source's withdraw")

        field_type = "D"
        component = "Ez"
        is_integrated = True
        _n_source_points = 3
        _applied_dipole = 0j

    fields, pml, grid = _dcyl()
    covered, reason = family.covers_cylindrical_real_fused_hd_pair(
        fields, pml, grid, [_ForeignElectric()])
    assert not covered and "standing integrated" in reason


# ---------------------------------------------------------------------------
# The cell, read from the board rather than transcribed
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _BOARD.is_file(), reason=f"{_BOARD} is not in this checkout")
def test_the_cell_this_product_claims_is_the_boards_three_rows():
    instances = json.loads(_BOARD.read_text(encoding="utf-8"))["h_to_d_seam"]["instances"]
    rows = [r for r in instances if (r["update_H"], r["step_D"]) == _CELL]
    assert len(rows) == 3, sorted(r["row"] for r in rows)
    assert all(r["bucket"] == "buildable_not_built" for r in rows)
    assert not any(r["withdraw_in_seam"] for r in rows)
    for text in ("3\nseam-instances", "test_pml_cyl_0_0_0"):
        assert text.replace("\n", " ") in family.__doc__.replace("\n", " ")


def test_the_composer_carries_the_rows_this_product_needs_and_refuses_to_install_it():
    """Wired 2026-09-07: the absorb row names the two certified halves' arms, the
    product row sits on the update_H seam, the span is REPLACES, and the composer
    refuses the product BY NAME from its own INSTALLABLE = False."""
    from meep_gpu.cuda_kernels import fused_pairs  # noqa: PLC0415

    assert fused_pairs.FUSED_PAIR_ARMS[family.FAMILY] == ("ordinary", "cylindrical")
    product = fused_pairs.FUSED_PRODUCTS[family.FAMILY]
    assert product["curl_slot"] == family.SLOT == "update_H"
    assert fused_pairs.span_of(family.FAMILY, product) == family.REPLACES
    reason = fused_pairs._declared_uninstallable(family.FAMILY, product)  # noqa: SLF001
    assert reason is not None and family.FAMILY in reason and "no launch saved" in reason
    assert fused_pairs.FUSED_PAIR_SEAMS["update_H"] == ("step_D", withdraw_hoist.SEAM)


# ---------------------------------------------------------------------------
# The emitter (device host only)
# ---------------------------------------------------------------------------

@_emitter
def test_the_emitted_source_is_the_three_imported_emitters_plus_the_scratch_stores():
    _needs_certified_text()
    source = family.kernel_source()
    composed = (fused_hd_pair.constitutive_prelude() + fused_hd_pair.weld_args_struct()
                + fused_hd_pair.raw_update_H_cell_source() + family.signature())
    assert source.startswith(composed)
    assert source.endswith("    f_w_Hz_out[idx] = own_w[2];\n}\n")
    assert source.isascii()
    assert source.count(f"void {family.KERNEL_NAME}(") == 1
    # The thread's OWN cell only: the radial neighbour's recompute went with the increment.
    assert source.count("raw_update_H_cell(idx") == 1
    assert "idx - nyz" not in source
    assert list(family.device_sources()) == [family.KERNEL_NAME]


@_emitter
def test_no_stale_magnetic_read_survives_below_the_own_cell_store():
    _needs_certified_text()
    source = family.kernel_source()
    below = source.split("raw_update_H_cell(idx, weld, own_h, own_w);", 1)[1]
    for name in family.H_TARGETS:
        assert f"{name}[" not in below and f"weld.{name}" not in below
