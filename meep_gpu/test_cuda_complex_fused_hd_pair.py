"""Host tests for the CUDA complex Cartesian H->D weld -- both variants.

Everything here runs WITHOUT a device: ``complex_emitter``, ``complex_folded_kernels``
and this family's emitter are all CuPy-free, so the whole lift is testable at the merge
bar. What is pinned:

* the four declared constitutive edits INVERT to the certified prelude character for
  character, on BOTH variants and BOTH arms;
* the certified ``cshift_dn`` survives the resolve with its branch, its ghost, its wrap
  arithmetic and the Bloch rotation ON THE WRAPPED LANE ONLY -- and the FOLDED variant
  arrives carrying ``complex_folded_kernels``' widened ghost, which is the fact that
  lets one product cover both board cells;
* every one of the six shifted magnetic taps and six own-cell loads is redirected, and
  no magnetic pointer survives readable in the curl half;
* the two variants emit DIFFERENT ASCII texts, one declared kernel each;
* the family performs no floating-point division -- the one arithmetic fact a reader
  coming from the cylindrical H->D products will look for and not find;
* the predicate is a CONJUNCTION of the two certified halves' own predicates, selected
  by ``variant_for``, and refuses the seam's withdraw rows by name.

THE HOST IS NOT THE ORACLE FOR THE COMPLEX MULTIPLY. NumPy's ``complex64 * float32``
and CuPy's differ on the sign of a flushed zero, which is why the emitted text carries
the FMA_V1 ``mul_field_left`` measured 0 of 4,005,000 words on device and why the gate
arms the naive four-product form on a PLANTED row-0 tiny-normal class. Device identity
is ``parity/meep_gpu/gate_cuda_complex_fused_hd_pair.py``'s.
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
from meep_gpu.cuda_kernels import complex_emitter, complex_folded_kernels
from meep_gpu.cuda_kernels import complex_fused_hd_pair as family
from meep_gpu.cuda_kernels import fused_hd_pair
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid, Mirror
from meep_gpu.pml import PML

_API_ROOT = Path(__file__).resolve().parents[1]
_BOARD = (_API_ROOT / "parity" / "meep_gpu" / "results"
          / "fusion_matrix_cuda_2026-09-07_cyl" / "fusion_matrix_cuda.json")
_CELL_PLAIN = ("cuda_complex/complex", "cuda_complex/complex")
_CELL_FOLDED = ("cuda_complex/complex", "cuda_complex_folded/folded complex")

#: A licence of the shape ``complex_fields.expansion_license`` returns, cut under keep.
_LICENCE = {"arm": "FMA_V1", "expansion": 1, "refusals": [], "basis": "measured",
            "policy": "ieee_keep_ftz_stripped", "policy_resolved": "keep"}
_POLICY = "keep"

_VARIANTS = ("plain", "folded")
_ARMS = ("FMA_V1", "NAIVE")


class _NumpyWearingCupysName:
    """NumPy answering to the name every predicate's backend check reads."""

    __name__ = "cupy"

    def __getattr__(self, item):
        return getattr(np, item)


def _grid(folded: bool = False, k_point=(0.0, 0.0, 0.0), boundaries=None):
    symmetry = (Mirror("Y", 1),) if folded else ()
    return Grid(resolution=1.0, cell_size=(9.0, 10.0, 11.0), courant=0.5,
                xp=_NumpyWearingCupysName(), k_point=tuple(k_point),
                boundaries=boundaries, symmetry=symmetry)


def _engine(folded: bool = False, **kwargs):
    grid = _grid(folded=folded, **kwargs)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    thickness = tuple((0, 2) if grid.is_mirrored(axis) else (2, 2) for axis in range(3))
    return fields, PML(grid=grid, thickness=thickness), grid


# ---------------------------------------------------------------------------
# The declarations
# ---------------------------------------------------------------------------

def test_the_seam_span_and_slot_are_the_real_siblings():
    assert family.SEAM == withdraw_hoist.SEAM
    assert family.REPLACES == withdraw_hoist.SEAM_SPAN == ("update_H", "step_D")
    assert family.SLOT == "update_H"
    assert family.CARRIES_DEPOSIT_REPAIR is False
    assert family.HOISTS_THE_WITHDRAW is False
    assert family.INSTALLABLE is False
    assert "can only TIE" in family.INSTALLABLE_REASON
    # The rotation and the scratch tuple are the Cartesian sibling's, REUSED rather
    # than re-spelled, so the two products cannot disagree about what a rotation
    # covers.
    assert family.SCRATCH_VOLUMES is fused_hd_pair.SCRATCH_VOLUMES
    assert family.H_TARGETS is fused_hd_pair.H_TARGETS
    assert family.rotate_into_fields.__module__.endswith("complex_fused_hd_pair")


def test_the_two_variants_and_their_kernel_names():
    assert family.VARIANTS == _VARIANTS
    assert set(family.KERNEL_NAMES) == set(_VARIANTS)
    assert family.KERNEL_NAME == family.KERNEL_NAMES["plain"]
    assert len(set(family.KERNEL_NAMES.values())) == 2


def test_both_kernel_names_are_in_exactly_one_partition_set():
    """The partition ``test_kernel_partition.py`` enforces package-wide, per module."""
    source = (Path(family.__file__)).read_text(encoding="utf-8")
    declared = set()
    for node in ast.parse(source).body:
        target = (node.targets[0] if isinstance(node, ast.Assign)
                  and len(node.targets) == 1 else None)
        if isinstance(target, ast.Name) and target.id == "UNCERTIFIED_KERNELS":
            declared = set(ast.literal_eval(node.value))
    assert declared == set()
    assert set(family.CERTIFIED_KERNELS) == set(family.KERNEL_NAMES.values())
    # Every CERTIFIED name is backed by a block a reader can check, which is what
    # ``test_every_certified_kernel_is_claimed_by_a_record_block`` enforces
    # package-wide. Checked against the raw record text, so a block nested anywhere
    # in it counts.
    record_text = (Path(family.__file__).parent / "certification.json").read_text(
        encoding="utf-8")
    for name in family.CERTIFIED_KERNELS:
        assert f'"{name}"' in record_text


def test_the_release_disclaimer_names_what_it_does_not_license():
    text = family.WHAT_A_RELEASE_DOES_NOT_LICENSE
    for phrase in ("PREDICATE ADMISSION", "INSTALLABLE = False", "NO TIMING",
                   "COUNT"):
        assert phrase in text


# ---------------------------------------------------------------------------
# The lift
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variant", _VARIANTS)
@pytest.mark.parametrize("arm", _ARMS)
def test_the_constitutive_lift_inverts_to_the_certified_prelude(variant, arm):
    """The four edits are ANCHORED, and undoing them gives back the certified text."""
    prelude = family.constitutive_prelude(variant, arm)
    certified = (complex_folded_kernels.folded_source("step_D", arm)
                 if variant == "folded"
                 else complex_emitter.complex_source("step_D", arm)
                 ).split('\nextern "C" __global__ void ', 1)[0]
    inverted = prelude
    for old, new in family.CONSTITUTIVE_LIFT_EDITS:
        assert prelude.count(new) == 1, f"the lifted prelude lost {old!r}"
        inverted = inverted.replace(new, old, 1)
    assert inverted == certified


@pytest.mark.parametrize("variant", _VARIANTS)
@pytest.mark.parametrize("arm", _ARMS)
def test_the_certified_update_H_body_survives_the_capture(variant, arm):
    """Every storing call is captured; the decode and the coefficient pairing stay."""
    del variant   # the constitutive body is variant-INDEPENDENT, which is the point
    body = family.raw_update_H_cell_source(arm)
    assert "constitutive_apply(" not in body
    assert body.count("constitutive_apply_pure(") == 3
    # The certified component-to-axis pairing, character for character.
    for component, axis in enumerate("xyz"):
        coordinate = "ijk"[component]
        assert (f"kps_{axis}[{coordinate}], kms_{axis}[{coordinate}]" in body)
    assert body.count("int k = idx % nz;") == 1
    assert "if (idx >= nx * ny * nz) return;" not in body


@pytest.mark.parametrize("variant", _VARIANTS)
@pytest.mark.parametrize("arm", _ARMS)
def test_the_resolved_shift_keeps_the_certified_branches(variant, arm):
    helper = family.cshift_dn_recompute_source(variant, arm)
    assert "if (ia > 0) return resolve_H(comp, idx - stride, weld);" in helper
    assert "resolve_H(comp, idx + (na - 1) * stride, weld)" in helper
    assert "if (ph) w = rotate_field_left(w, phase);" in helper
    assert "cf_load(g" not in helper
    # THE GHOST IS A CONSTANT, NOT A FIELD: it must not be recomputed.
    assert "return cf_zero();" in helper
    if variant == "folded":
        assert "bc == BC_METALLIC || bc == BC_MIRROR_PERIODIC" in helper
    else:
        assert "BC_MIRROR_PERIODIC" not in helper


def test_the_wrap_index_is_taken_with_balanced_parentheses():
    """The certified wrap spells its index ``idx + (na - 1) * stride``.

    A scan that stopped at the first ``)`` would cut it after ``(na - 1)`` and emit a
    kernel that reads a different cell -- which compiles and converges.
    """
    head, expression, tail = family._balanced_call(  # noqa: SLF001
        "x = cf_load(g, idx + (na - 1) * stride); y", "cf_load(g,")
    assert expression.strip() == "idx + (na - 1) * stride"
    assert head == "x = " and tail == "; y"


@pytest.mark.parametrize("variant", _VARIANTS)
@pytest.mark.parametrize("arm", _ARMS)
def test_every_magnetic_read_in_the_curl_half_is_redirected(variant, arm):
    head, tail = family.curl_pieces(variant, arm)
    assert "cshift_dn(" not in tail
    for pointer in family.CURL_SOURCE_NAMES:
        assert f"cf_load({pointer}," not in tail
    assert tail.count("cshift_dn_recompute(") == family.HALO_TAPS
    assert tail.count("own_h[") == family.OWN_LOAD_EDITS
    # The head is the certified preamble, the strides, the decode and the phases.
    assert "int idx = blockIdx.x * blockDim.x + threadIdx.x;" in head
    assert "const int sx = ny * nz;" in head
    assert "cf px; px.re = pxr; px.im = pxi;" in head


@pytest.mark.parametrize("variant", _VARIANTS)
@pytest.mark.parametrize("arm", _ARMS)
def test_the_emitted_source_is_ascii_with_one_declared_kernel(variant, arm):
    source = family.kernel_source(variant, arm)
    assert source.isascii()
    declarations = [line for line in source.splitlines()
                    if 'extern "C" __global__ void' in line]
    assert len(declarations) == 1
    assert family.KERNEL_NAMES[variant] in declarations[0]


def test_the_four_texts_are_all_different():
    texts = {(variant, arm): family.kernel_source(variant, arm)
             for variant in _VARIANTS for arm in _ARMS}
    assert len(set(texts.values())) == 4


def test_the_folded_text_carries_the_three_deltas_and_the_plain_one_does_not():
    plain = family.kernel_source("plain", "FMA_V1")
    folded = family.kernel_source("folded", "FMA_V1")
    assert "#define BC_MIRROR_PERIODIC 2" in folded
    assert "#define BC_MIRROR_PERIODIC" not in plain
    assert "bc == BC_METALLIC || bc == BC_MIRROR_PERIODIC" in folded
    # The mask splits by termination: the shift-0 arm on the folded code AND the
    # shift-1 top-plane arm, six clears across the three targets at minimum.
    assert folded.count("BC_MIRROR_PERIODIC && ") >= 6
    assert "BC_MIRROR_PERIODIC && " not in plain


def test_the_arm_block_is_lifted_whole_under_both_arms():
    for arm in _ARMS:
        block = complex_emitter._ARM_SOURCE[  # noqa: SLF001
            complex_emitter.EXPANSIONS[arm]]
        assert block.strip() in family.kernel_source("plain", arm)


def test_the_family_performs_no_floating_point_division():
    """The one arithmetic fact a reader from the cylindrical products will look for.

    CuPy's ``complex64 / float32`` is the SCALED complex/complex algorithm, not
    numpy's reciprocal multiply, and getting it wrong costs ~5.5 % of complex words.
    This family has no division site at all -- the reciprocal its recurrence needs is
    ``sinv``, computed host-side by ``PML`` -- so the only ``/`` in the emitted text
    is the index decode's integer one, checked here BY SHAPE rather than waved past.
    """
    allowed = {"int k = idx % nz;", "int j = (idx / nz) % ny;",
               "int i = idx / (ny * nz);"}
    for variant in _VARIANTS:
        for arm in _ARMS:
            for line in family.kernel_source(variant, arm).splitlines():
                code = line.split("//")[0].strip()
                assert "/" not in code or code in allowed, code


def test_the_weld_pack_binds_the_pre_launch_field_under_the_curl_templates_name():
    """``weld.Hx = g0`` -- there is no parameter called ``Hx``, only ``Hx_out``."""
    construction = family.weld_args_construction()
    assert "weld.Hx = g0;" in construction
    assert "weld.Hy = g1;" in construction
    assert "weld.Hz = g2;" in construction
    source = family.kernel_source("plain", "FMA_V1")
    start = source.index('extern "C" __global__')
    signature = source[start:source.index(") {", start)]
    assert "float* __restrict__ Hx_out" in signature
    assert " Hx," not in signature
    assert "const float* __restrict__ g0" in signature


def test_the_source_digest_moves_with_the_text():
    before = family.source_digest()
    assert isinstance(before, str) and len(before) == 64
    assert family.source_digest() == before
    assert set(family.device_sources()) == set(family.KERNEL_NAMES.values())


# ---------------------------------------------------------------------------
# The variant selector
# ---------------------------------------------------------------------------

def test_variant_for_reads_the_grid():
    assert family.variant_for(_grid(folded=False)) == "plain"
    assert family.variant_for(_grid(folded=True)) == "folded"


def test_variant_for_refuses_a_grid_with_no_fold_reader():
    class _G:
        xp = np

    with pytest.raises(ValueError, match="is_mirrored"):
        family.variant_for(_G())


@pytest.mark.parametrize("name", ("mirror", "cylindrical", "", None))
def test_an_unknown_variant_is_refused_rather_than_defaulted(name):
    with pytest.raises(ValueError, match="variant must be one of"):
        family.kernel_source(name, "FMA_V1")


def test_an_unknown_arm_is_refused_rather_than_defaulted():
    with pytest.raises(ValueError, match="expansion must be one of"):
        family.kernel_source("plain", "SOMETHING")


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------

def test_the_predicate_admits_a_plain_complex_run():
    fields, pml, grid = _engine()
    covered, reason = family.covers_complex_fused_hd_pair(
        fields, pml, grid, (), _LICENCE, _POLICY)
    assert covered, reason


def test_the_predicate_admits_a_folded_complex_run():
    fields, pml, grid = _engine(folded=True)
    covered, reason = family.covers_complex_fused_hd_pair(
        fields, pml, grid, (), _LICENCE, _POLICY)
    assert covered, reason


def test_the_predicate_refuses_without_a_licence_and_names_the_constitutive_half():
    fields, pml, grid = _engine()
    covered, reason = family.covers_complex_fused_hd_pair(fields, pml, grid, ())
    assert not covered
    assert reason.startswith("constitutive half:")
    assert "expansion licence" in reason


def test_the_predicate_refuses_an_undeclared_source_set():
    fields, pml, grid = _engine()
    covered, reason = family.covers_complex_fused_hd_pair(
        fields, pml, grid, None, _LICENCE, _POLICY)
    assert not covered
    assert "source set was not declared" in reason


def test_the_predicate_refuses_a_standing_integrated_electric_withdraw_by_name():
    class _Withdrawing:
        field_type = "electric"
        is_integrated = True
        _n_source_points = 1

        def withdraw(self, fields):  # noqa: ARG002
            return None

    fields, pml, grid = _engine()
    covered, reason = family.covers_complex_fused_hd_pair(
        fields, pml, grid, (_Withdrawing(),), _LICENCE, _POLICY)
    assert not covered
    assert "standing integrated" in reason
    assert "HOISTS_THE_WITHDRAW = False" in reason


def test_the_predicate_is_a_conjunction_of_the_two_certified_halves():
    """The refusal is PREFIXED with the half that produced it, on both variants."""
    from meep_gpu.cuda_kernels.coverage import (  # noqa: PLC0415
        covers_real_pml_complex_constitutive, covers_real_pml_complex_curl)

    fields, pml, grid = _engine()
    assert covers_real_pml_complex_constitutive(
        fields, pml, grid, "H", _LICENCE, _POLICY)[0]
    assert covers_real_pml_complex_curl(fields, pml, grid, "step_D",
                                        _LICENCE, _POLICY)[0]
    folded_fields, folded_pml, folded_grid = _engine(folded=True)
    # The certified PLAIN curl refuses a fold by name; the folded one admits it. The
    # product follows the grid, which is what makes one predicate serve two cells.
    assert not covers_real_pml_complex_curl(
        folded_fields, folded_pml, folded_grid, "step_D", _LICENCE, _POLICY)[0]
    assert complex_folded_kernels.covers_complex_folded_curl(
        folded_fields, folded_pml, folded_grid, "step_D", _LICENCE, _POLICY)[0]


def test_the_predicate_refuses_when_a_rotated_volume_is_missing():
    fields, pml, grid = _engine()
    fields.f_w_Hy = None
    covered, reason = family.covers_complex_fused_hd_pair(
        fields, pml, grid, (), _LICENCE, _POLICY)
    assert not covered
    assert "f_w_Hy is not allocated" in reason


# ---------------------------------------------------------------------------
# The board cells this product is built for
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _BOARD.is_file(), reason="the standing board is not on this host")
def test_the_two_cells_are_the_ones_the_board_calls_unbuilt():
    board = json.loads(_BOARD.read_text(encoding="utf-8"))
    instances = (board.get("h_to_d_seam") or {}).get("instances") or []
    counts = {}
    for record in instances:
        if record.get("bucket") != "buildable_not_built":
            continue
        cell = (record.get("update_H"), record.get("step_D"))
        counts[cell] = counts.get(cell, 0) + 1
    assert counts.get(_CELL_PLAIN) == 17
    assert counts.get(_CELL_FOLDED) == 5
    # The plain cell is the LARGEST unbuilt H->D cell on this backend, which is the
    # reason this product exists; if it stops being so the note in the module
    # docstring is stale.
    assert max(counts.values()) == 17


@pytest.mark.skipif(not _BOARD.is_file(), reason="the standing board is not on this host")
def test_no_row_of_either_cell_carries_a_standing_withdraw():
    """What the withdraw refusal costs on these two cells: nothing, MEASURED."""
    board = json.loads(_BOARD.read_text(encoding="utf-8"))
    instances = (board.get("h_to_d_seam") or {}).get("instances") or []
    mine = [record for record in instances
            if (record.get("update_H"), record.get("step_D"))
            in (_CELL_PLAIN, _CELL_FOLDED)
            and record.get("bucket") == "buildable_not_built"]
    assert len(mine) == 22
    assert not [record for record in mine if record.get("withdraw_in_seam")]
    assert not [record for record in mine
                if int(record.get("integrated_electric_sources") or 0)]


# ---------------------------------------------------------------------------
# Not wired, and this is where that is pinned
# ---------------------------------------------------------------------------

def test_this_family_is_wired_with_both_halves_of_its_row():
    """A product row WITHOUT its arm row (or the reverse) is the silent failure.

    ``install_fused_pairs`` refuses a family with no ``FUSED_PAIR_ARMS`` entry BY
    NAME, so half a wiring is a product that is never selected and nothing goes red.
    The predicate is additionally required to be DECLARED unregistered: a ``covers_*``
    that is neither in the arm table nor in ``NOT_REGISTERED`` is an undecided one.
    """
    from meep_gpu.cuda_kernels import fused_pairs, registry  # noqa: PLC0415

    assert family.FAMILY in fused_pairs.FUSED_PRODUCTS
    assert fused_pairs.FUSED_PRODUCTS[family.FAMILY]["curl_slot"] == "update_H"
    assert fused_pairs.FUSED_PAIR_ARMS[family.FAMILY] == ("complex", "complex")
    predicate = "complex_fused_hd_pair.covers_complex_fused_hd_pair"
    assert predicate in registry.NOT_REGISTERED
    # STILL REFUSED, on every configuration, and that is the point of the wiring:
    # the flag is now reachable by the composer rather than merely declared.
    assert fused_pairs._declared_uninstallable(  # noqa: SLF001
        family.FAMILY, fused_pairs.FUSED_PRODUCTS[family.FAMILY])
