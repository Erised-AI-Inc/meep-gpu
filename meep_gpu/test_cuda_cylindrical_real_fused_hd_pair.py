"""Host tests for the CUDA real m = 0 Dcyl H->D two-launch product.

WHAT CAN BE TESTED WITHOUT A DEVICE: the declarations, the seam facts, the predicate on
a NumPy-wearing-CuPy shim, the array-path cache key the row vectors share with
``stepping``, and the INCREMENT SPELLING pinned against the shipped
``cylindrical_rderiv_prefix`` on NumPy (where the real family's ``/`` IS the oracle's
spelling and the reciprocal multiply is measurably not). The emitter needs the certified
device strings, which import CuPy at scope; those tests declare the resource and skip.

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

from meep_gpu import stepping, withdraw_hoist
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


def test_three_launches_per_run_and_the_scan_is_the_one_counted_beside_two_kernels():
    """Launch 1, ``xp.cumsum``, launch 2: the scan is COUNTED, never fused."""
    assert family.LAUNCHES_PER_RUN == 3
    source = Path(family.__file__).read_text(encoding="utf-8")
    assert "xp.cumsum(state[\"increment\"], axis=0, out=state[\"prefix\"])" in source
    assert "column_serial" not in source


def test_the_prefix_source_and_half_shift_are_read_off_the_shared_table():
    assert family.PREFIX_SOURCE == cylindrical_prefix.PREFIX_COMPONENT["step_D"] == "Hy"
    assert family.PREFIX_IR0_VALUE == cylindrical_prefix.PREFIX_IR0["step_D"] == 0.5


def test_the_product_declares_itself_uninstallable_on_a_cylindrical_launch_algebra():
    assert family.INSTALLABLE is False
    for phrase in ("7 launches", "+4 - 2 = +2", "launches_saved", "composition gate",
                   "predicate admission"):
        assert phrase in family.INSTALLABLE_REASON, phrase
    assert "PREDICATE ADMISSION" in family.WHAT_A_RELEASE_DOES_NOT_LICENSE
    assert "no timing" in family.WHAT_A_RELEASE_DOES_NOT_LICENSE.lower()


def test_the_kernel_is_in_exactly_one_certification_set_and_the_curl_is_the_certified_one():
    assert family.KERNEL_NAME in family.CERTIFIED_KERNELS
    assert family.KERNEL_NAME not in fused_hd_pair.CERTIFIED_KERNELS
    assert not family.UNCERTIFIED_KERNELS
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
# The measured spelling
# ---------------------------------------------------------------------------

def test_every_declared_increment_line_is_emitted_once_and_carries_a_control():
    source = family.increment_source()
    for entry in family.INCREMENT_SPELLING:
        assert set(entry) == {"line", "transcribes", "control"}, entry
        assert source.count(entry["line"]) == 1, entry["line"]
    assert "increment[idx] = diff / divisor[i - 1];" in source
    assert "1.0f /" not in source
    assert "raw_update_H_cell(idx - nyz, weld, halo_h, halo_w);" in source
    for name in family.H_TARGETS:
        assert f"{name}[" not in source and f"weld.{name}" not in source


def test_the_divide_inverts_between_the_two_families():
    """The real family spells ``/``; the complex sibling spells CuPy's scaled algorithm.
    Each is right on its own storage and wrong on the other's (measured on device)."""
    real = family.increment_source()
    complex_source = complex_sibling.increment_source()
    assert " / divisor[i - 1]" in real
    assert "cf_div_coefficient(diff, divisor[i - 1])" in complex_source
    assert " / divisor" not in complex_source
    divide = complex_sibling.div_coefficient_source()
    assert "float s = fabsf(d) + fabsf(0.0f);" in divide
    assert "q.re = ((ars * brs) + (ais * bis)) * oos;" in divide


def test_the_numpy_replica_of_the_real_spelling_reproduces_the_shipped_prefix():
    """On NumPy the increment's ``/`` reproduces ``stepping.cylindrical_rderiv_prefix``
    word for word and the reciprocal multiply does not -- the host half of the spelling
    argument, with a control that bites."""
    rng = np.random.default_rng(1)
    for nr, nz in ((16, 20), (163, 175), (40, 1)):
        hy = rng.standard_normal((nr, 1, nz)).astype(np.float32)
        counts = np.arange(nr, dtype=np.float64) + 0.5
        weights = counts.astype(np.float32)
        divisor = (counts[1:] - 0.5).astype(np.float32)
        increment = np.zeros_like(hy)
        wi = hy[1:] * weights[1:, None, None]
        wim1 = hy[:-1] * weights[:-1, None, None]
        increment[1:] = (wi - wim1) / divisor[:, None, None]
        shipped = stepping.cylindrical_rderiv_prefix(np, hy, 0.5)
        mine = np.cumsum(increment, axis=0, dtype=np.float32)
        assert int(np.count_nonzero(mine.view(np.uint32) != shipped.view(np.uint32))) == 0
        reciprocal = np.zeros_like(hy)
        reciprocal[1:] = (wi - wim1) * (np.float32(1) / divisor)[:, None, None]
        control = np.cumsum(reciprocal, axis=0, dtype=np.float32)
        moved = int(np.count_nonzero(control.view(np.uint32) != shipped.view(np.uint32)))
        assert moved > 0, (nr, nz, moved)


def test_the_row_vectors_are_the_array_paths_own_cached_ones():
    fields, _pml, _grid = _dcyl(shape=(16, 1, 20))
    weights, divisor = family.prefix_row_vectors(fields)
    assert weights.shape == (16,) and divisor.shape == (15,)
    cached = fields.scratch.constant(("cyl_rderiv", 16, 0.5, np.dtype(np.float32)),
                                     lambda: (None, None))
    assert cached[0] is not None, "the family did not populate stepping's own cache key"
    assert np.array_equal(cached[0].reshape(-1), weights)
    assert np.array_equal(cached[1].reshape(-1), divisor)
    assert weights[0] == np.float32(0.5) and divisor[0] == np.float32(1.0)


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


def test_a_standing_integrated_electric_withdraw_is_refused_by_name():
    from meep_gpu.sources import ContinuousEnvelope, VolumeSource  # noqa: PLC0415

    fields, pml, grid = _dcyl()
    source = VolumeSource(grid=grid, component="Ez", center=(0.0, 0.0, 0.0),
                          size=(0.0, 0.0, 0.0),
                          envelope=ContinuousEnvelope(frequency=1.0, is_integrated=True))
    covered, reason = family.covers_cylindrical_real_fused_hd_pair(fields, pml, grid, [source])
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
    assert reason is not None and family.FAMILY in reason and "+4 - 2" in reason
    assert fused_pairs.FUSED_PAIR_SEAMS["update_H"] == ("step_D", withdraw_hoist.SEAM)


# ---------------------------------------------------------------------------
# The emitter (device host only)
# ---------------------------------------------------------------------------

@_emitter
def test_the_emitted_source_is_the_four_imported_emitters_plus_the_increment():
    _needs_certified_text()
    source = family.kernel_source()
    composed = (fused_hd_pair.constitutive_prelude() + fused_hd_pair.weld_args_struct()
                + fused_hd_pair.raw_update_H_cell_source() + family.signature())
    assert source.startswith(composed)
    assert source.endswith(family.increment_source() + "}\n")
    assert source.isascii()
    assert source.count(f"void {family.KERNEL_NAME}(") == 1
    assert list(family.device_sources()) == [family.KERNEL_NAME]


@_emitter
def test_no_stale_magnetic_read_survives_below_the_own_cell_store():
    _needs_certified_text()
    source = family.kernel_source()
    below = source.split("raw_update_H_cell(idx, weld, own_h, own_w);", 1)[1]
    for name in family.H_TARGETS:
        assert f"{name}[" not in below and f"weld.{name}" not in below
