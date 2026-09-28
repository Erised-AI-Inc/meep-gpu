"""Host-side contract for the BFAST Metal H->D weld.

WHAT THESE TESTS PIN AND WHAT THEY DELIBERATELY DO NOT. Byte identity against the array
path is the GATE's claim and needs a device
(``parity/meep_gpu/gate_metal_bfast_fused_hd_pair.py``); what a host suite can hold is
the LIFT -- that the emitted kernel is the certified BFAST curl's and the certified
constitutive's own text with exactly the declared edits -- the SIGNATURE arithmetic, the
Tustin SCALARS, the PREDICATE's inversions, and the declarations the composer and the
board read.

THE ONE THING THIS FAMILY CAN GET WRONG SILENTLY is the six host-rounded ``k1``/``k2``
scalars, and they are checked here against ``stepping``'s own term tables as float32
WORDS -- never ``allclose``, because a sign that flipped on a zero would compare equal.

Every test here runs without ``torch``. The two that need a device declare themselves
skipped rather than passing quietly.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import re
from typing import Any, Sequence, Tuple

import numpy as np
import pytest

from meep_gpu.metal_kernels import (
    bfast_curl, bfast_fused_hd_pair as family, fused_hd_pair as plain, shaders,
)
from meep_gpu.metal_kernels.device import MAX_BUFFER_BINDINGS
from meep_gpu.metal_kernels.launch import SUB_STEPS
from meep_gpu.triton_kernels.bfast_curl import (
    BFAST_STATE_NAMES, bfast_curl_coefficients,
)

CODES: Tuple[Tuple[int, int, int], ...] = ((0, 0, 0), (1, 1, 1), (1, 0, 1))


def sources() -> Sequence[Tuple[Tuple[int, ...], str]]:
    return [(codes, family.bfast_fused_hd_pair_source(codes)) for codes in CODES]


# ---------------------------------------------------------------------------
# The declarations the composer and the board read
# ---------------------------------------------------------------------------

def test_the_span_is_the_drivers_two_adjacent_consults():
    assert family.REPLACES == ("update_H", "step_D")
    assert family.SLOT == "update_H" == family.REPLACES[0]


def test_the_seam_name_is_spelled_through_withdraw_hoist():
    from meep_gpu import withdraw_hoist

    assert family.SEAM is withdraw_hoist.SEAM


def test_nothing_is_injected_in_this_seam_so_the_repair_flag_is_false():
    import inspect

    assert family.CARRIES_DEPOSIT_REPAIR is False
    body = inspect.getsource(family.metal_bfast_fused_hd_pair_coverage)
    code = "\n".join(line for line in body.splitlines()
                     if not line.strip().startswith("#"))
    assert "deposit_repair." not in code
    assert "seam_withdraw_reasons" in code


def test_the_withdraw_flag_and_its_wiring_move_together():
    assert family.HOISTS_THE_WITHDRAW is False
    assert not (family.HOISTS_THE_WITHDRAW and not family.INSTALLABLE)


def test_the_installable_reason_records_the_measurement_that_corrected_it():
    """THE CELL'S FIRST READING WAS WRONG AND THE CONSTANT SAYS SO.

    It is natural to assume both neighbours install here, since
    ``bfast_fused_magnetic_pair`` and ``bfast_fused_electric_pair`` are both built and
    gated. Measured through the shipped composer, only the electric one does -- the
    magnetic pair carries no ``FUSED_PAIR_ARMS`` row -- so this cell is a TIE and not
    a loss. A reason that still claimed "LOSS" would be a number nobody could
    reproduce.
    """
    assert family.INSTALLABLE is False
    reason = family.INSTALLABLE_REASON
    assert "TIE" in reason
    assert "bfast_fused_electric_pair" in reason
    assert "FUSED_PAIR_ARMS" in reason
    assert "no timing claim" in reason
    assert "LOSS" not in reason


def test_the_family_is_welded_and_the_declaration_it_replaced_is_kept():
    """The partition's other half: this family has a GATE and now a ledger entry.

    ``test_metal_weld_contract.test_every_metal_family_is_welded`` requires the
    declaring set and the welded set to be DISJOINT and to EXHAUST the fleet, so the
    two halves move together and each prevents a different failure: emptying
    ``WELD_OWED`` without the entry claims a released gate this tree cannot show, and
    cutting the entry without emptying the string leaves a welded family still
    declaring its weld is owed. BOTH are checked here rather than either being taken
    on trust.

    THE DECLARATION WAS EMPTIED ONCE BEFORE, ON 2026-09-08, BESIDE A HAND-WRITTEN
    LEDGER ENTRY, and both were reverted. That entry was not the minting tool's
    output: its ``code_sha256`` map held raw ``sha256`` values, which
    ``code_identity.code_digest_of_path`` -- the function ``mint_metal_weld.py:144``
    fills that field with -- cannot return, and its ``recorded_utc`` was rounded to
    the minute where the tool writes seconds. What released the family instead was a
    re-gate: emptying the string MOVES this module, the module is the first pin of its
    own weld, so the gate was re-run against the emptied bytes and
    ``mint_metal_weld.py`` bound that artifact. The reverted text is kept in
    ``_RETIRED_WELD_OWED`` rather than deleted, which is what the third assertion
    reads -- a weld that erased the record of the forgery it replaced would be worse
    evidence than one that keeps it.
    """
    from meep_gpu.metal_kernels import launch as metal_launch  # noqa: PLC0415

    assert family.WELD_OWED == "", (
        "the family holds a released device gate; WELD_OWED must be empty, and the "
        "weld contract's partition is what reads it")
    # The weld entries are TOP-LEVEL keys of fingerprints.json, beside the
    # "metal_kernels" digest block.
    assert "metal_bfast_fused_hd_pair_device_gate" in metal_launch.load_fingerprints(), (
        "WELD_OWED is empty and fingerprints.json carries no "
        "metal_bfast_fused_hd_pair_device_gate: one of the two is false")
    assert ("results/" in family._RETIRED_WELD_OWED
            and len(family._RETIRED_WELD_OWED) > 200), (
        "the retired declaration must still name the artifact and what the release "
        "owed; a weld does not license deleting the record it replaced")


# ---------------------------------------------------------------------------
# The signature
# ---------------------------------------------------------------------------

def test_the_shipped_signature_is_counted_off_its_own_source():
    assert family.shipped_signature_bindings() == family.PACKED_BINDINGS
    assert family.PACKED_BINDINGS == family.PACKED_POINTERS + 1
    assert family.PACKED_BINDINGS <= MAX_BUFFER_BINDINGS


def test_the_unpacked_shape_is_over_the_ceiling_which_is_why_the_pack_exists():
    assert family.UNPACKED_POINTER_BINDINGS > MAX_BUFFER_BINDINGS
    assert family.UNPACKED_POINTER_BINDINGS == family.UNPACKED_POINTERS + 1
    # The pack costs one pointer and saves the nine members it holds.
    saved = family.UNPACKED_POINTERS - family.PACKED_POINTERS
    assert saved == len(family.PACKED_VECTORS) - 1


def test_every_refuted_signature_declares_the_count_it_claims():
    for builder, count in (
            (family.refuted_unpacked_pointer_source,
             family.UNPACKED_POINTER_BINDINGS),
            (family.refuted_separate_scalar_source,
             family.SEPARATE_SCALAR_BINDINGS)):
        source = builder()
        assert source.count("[[buffer(") == count
        assert count > MAX_BUFFER_BINDINGS


def test_the_pack_holds_both_halves_coefficient_vectors_and_no_second_km_group():
    assert set(family.PACKED_VECTORS) == {
        "kmx", "sinvx", "kmy", "sinvy", "kmz", "sinvz", "kp0", "kp1", "kp2"}
    assert not {"km0", "km1", "km2"} & set(family.PACKED_VECTORS)
    assert SUB_STEPS["step_D"]["suffix"] == ""
    from meep_gpu.triton_kernels.coverage import CONSTITUTIVE_SIDES

    assert CONSTITUTIVE_SIDES["H"]["half_integer"] is False


def test_the_six_tustin_scalars_are_the_curls_own_signature_order():
    """The order is ``bfast_curl``'s buffer order AND ``bfast_curl_coefficients``'.

    Two orders that agree by accident would be a filter with two of its coefficients
    exchanged on every cell, which is smooth and wrong rather than a failure.
    """
    assert family.K_SCALARS == ("k1_0", "k2_0", "k1_1", "k2_1", "k1_2", "k2_2")
    certified = bfast_curl._BFAST_CURL_TEMPLATE  # noqa: SLF001
    positions = [certified.index(f"constant float&     {name}")
                 for name in family.K_SCALARS]
    assert positions == sorted(positions)
    # And the coefficient builder returns them in that same order, per target.
    values = bfast_curl_coefficients((0.2, -0.13, 0.07), (False, False, False),
                                     magnetic=False)
    assert len(values) == len(family.K_SCALARS)


def test_the_electric_side_is_the_negated_magnetic_side_as_words():
    """MEEP's ``if (ft == D_stuff) { k1 = -k1; k2 = -k2; }`` -- compared as WORDS.

    ``allclose`` would call ``-0.0`` and ``0.0`` equal, which is exactly the pair a
    dropped negation on a zero component produces.
    """
    for shear in ((0.2, -0.13, 0.07), (0.2, 0.0, 0.0), (0.0, 0.0, 0.0)):
        for invariant in ((False, False, False), (False, True, False)):
            electric = np.array(bfast_curl_coefficients(shear, invariant,
                                                        magnetic=False),
                                dtype=np.float32)
            magnetic = np.array(bfast_curl_coefficients(shear, invariant,
                                                        magnetic=True),
                                dtype=np.float32)
            assert np.array_equal(electric.view(np.uint32),
                                  (-magnetic).view(np.uint32)), (shear, invariant)


# ---------------------------------------------------------------------------
# The lift
# ---------------------------------------------------------------------------

def test_every_magnetic_read_is_a_register_or_a_recompute():
    for codes, source in sources():
        tail = family.bfast_welded_curl_tail(codes)
        for target in range(3):
            assert f"g{target}[" not in tail, (codes, target)
        assert "g0[" not in source.split("__CURL__")[-1]


def test_the_lift_redirects_exactly_the_nine_magnetic_loads():
    for codes, source in sources():
        assert source.count("h_cell(") == 1 + 1 + len(plain.HALO_TAPS)
        assert "float a   = own.a0;" in source
        for var, target in plain.HALO_TAPS:
            assert re.search(rf"float {var} = \w+ \? h_cell\([^)]*\)\.a{target} : 0\.0f;",
                             source), (codes, var)


def test_the_bfast_tail_survives_the_lift_untouched():
    """Its operands are the curl's OWN registers, so it contains no ``gN[`` at all.

    Every line of the certified tail must appear character for character in the welded
    body: this family adds no arithmetic to the Tustin filter, and a lift that touched
    one of those lines would be arithmetic it does not own.
    """
    for codes, _source in sources():
        certified = bfast_curl.bfast_curl_source(
            codes, bool(SUB_STEPS["step_D"]["backward"]), has_bfast=True)
        welded = family.bfast_welded_curl_tail(codes)
        for line in certified.splitlines():
            stripped = line.strip()
            if not re.match(r"(float (st|total|adv)\d|s\d\[ii\] = |curl\d = curl\d - "
                            r"adv\d;|adv\d = at_[xyz] \?)", stripped):
                continue
            assert stripped in welded, (codes, stripped)


def test_the_tustin_state_is_stepped_in_place_and_is_not_rotated():
    """``f_bfast_D`` is own-cell only, so in place is safe -- and it must NOT rotate.

    The driver's flux backup/restore around the magnetic half-step depends on the
    mirror wrapping ``fields.f_bfast_*`` itself; rotating it would hand the restore a
    buffer the engine no longer names.
    """
    assert set(BFAST_STATE_NAMES["step_D"]) & set(family.ROTATED_NAMES) == set()
    assert family.ROTATED_NAMES == plain.ROTATED_NAMES
    for codes, source in sources():
        for index in range(3):
            assert f"s{index}[ii] = st{index} + adv{index};" in source
            # read at the thread's own cell only -- no shifted index anywhere.
            assert not re.search(rf"s{index}\[o[xyz]\]", source), (codes, index)


def test_the_constitutive_half_is_the_plain_products_lift():
    for _codes, source in sources():
        assert plain.h_cell_function() in source


def test_the_ghost_ternary_and_its_guard_are_the_emitters_own():
    for codes, source in sources():
        certified = bfast_curl.bfast_curl_source(
            codes, bool(SUB_STEPS["step_D"]["backward"]), has_bfast=True)
        for line in certified.splitlines():
            stripped = line.strip()
            if stripped.startswith(("vx = ", "vy = ", "vz = ", "sj = ", "sk = ",
                                    "si = ", "int si = ")):
                assert stripped in source, (codes, stripped)
        assert source.count(" : 0.0f;") >= len(plain.HALO_TAPS)


def test_the_pack_prologue_recreates_every_member_under_its_own_name():
    for _codes, source in sources():
        for member in family.PACKED_VECTORS:
            assert (f"device const float* {member} = cpack + prm.off_{member};"
                    in source)


def test_the_params_struct_carries_the_scalars_and_one_offset_per_member():
    for _codes, source in sources():
        struct = source.split("struct Params {", 1)[1].split("};", 1)[0]
        for scalar in family.K_SCALARS:
            assert f"float {scalar};" in struct
        for member in family.PACKED_VECTORS:
            assert f"uint off_{member};" in struct
        assert struct.count("uint off_") == len(family.PACKED_VECTORS)


def test_the_wall_clear_is_not_carried():
    for _codes, source in sources():
        assert "zero_metal" not in source


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------

class _Grid:
    def __init__(self, **overrides: Any) -> None:
        self.shape = (6, 5, 4)
        self.dt = 0.035
        self.dx = 0.1
        self.beta = 0.0
        self.cylindrical = False
        self.bfast_active = True
        self.bfast_scaled_k = (0.2, -0.13, 0.07)
        self.dimensions = 3
        for key, value in overrides.items():
            setattr(self, key, value)

    def is_mirrored(self, axis: int) -> bool:  # noqa: ARG002
        return False

    def is_invariant(self, axis: int) -> bool:  # noqa: ARG002
        return False


def test_the_predicate_is_a_conjunction_of_the_two_bfast_halves():
    import inspect

    body = inspect.getsource(family.metal_bfast_fused_hd_pair_coverage)
    assert "bfast_run_constitutive_coverage(fields, pml, \"H\", residency)" in body
    assert "bfast_pml_curl_coverage(fields, pml, \"step_D\", residency)" in body
    assert "constitutive half:" in body
    assert "curl half:" in body


def test_the_predicate_refuses_a_fields_with_no_grid():
    verdict = family.metal_bfast_fused_hd_pair_coverage(object(), None, ())
    assert not verdict.covered


def test_an_undeclared_source_set_is_refused_rather_than_assumed_empty():
    class _Fields:
        grid = _Grid()

    verdict = family.metal_bfast_fused_hd_pair_coverage(_Fields(), None, None)
    assert not verdict.covered
    assert any("the source set was not declared" in reason
               for reason in verdict.reasons)


def test_the_fold_is_refused_by_name_naming_the_folded_bfast_cell():
    class _Folded(_Grid):
        def is_mirrored(self, axis: int) -> bool:
            return axis == 2

    class _Fields:
        grid = _Folded()

    verdict = family.metal_bfast_fused_hd_pair_coverage(_Fields(), None, ())
    assert not verdict.covered
    joined = " | ".join(verdict.reasons)
    assert "axis 2 is folded" in joined
    assert "folded BFAST" in joined


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_the_arm_is_registered_unwired_on_the_first_half_of_the_seam():
    from meep_gpu.metal_kernels import arms

    rows = [spec for spec in arms.registered("update_H")
            if spec.family == family.FAMILY]
    assert len(rows) == 1
    assert rows[0].wired is False
    assert rows[0].replaces == family.REPLACES
    assert rows[0].is_weld is True
    assert family.FAMILY not in [spec.family for spec in arms.registered("update_H")
                                if spec.wired]


def test_the_arm_refuses_every_slot_that_is_not_its_own():
    for slot in ("step_B", "step_D", "update_E", "update_P"):
        verdict = family._arm_coverage(object(), slot)  # noqa: SLF001
        assert not verdict.covered
        assert family._arm_plan(object(), slot) is None  # noqa: SLF001


# ---------------------------------------------------------------------------
# The device-dependent claims, declared skipped rather than passing quietly
# ---------------------------------------------------------------------------

def _mps() -> bool:
    try:
        import torch
    except ImportError:  # pragma: no cover
        return False
    return bool(torch.backends.mps.is_available())


@pytest.mark.skipif(not _mps(), reason="no MPS device on this host")
def test_the_shipped_signature_compiles_on_every_specialisation():
    from meep_gpu.metal_kernels.device import compile_source

    for _codes, source in sources():
        compile_source(source)


@pytest.mark.skipif(not _mps(), reason="no MPS device on this host")
def test_the_two_refuted_signatures_do_not_compile():
    from meep_gpu.metal_kernels.device import compile_source

    for builder in (family.refuted_unpacked_pointer_source,
                    family.refuted_separate_scalar_source):
        with pytest.raises(Exception):
            compile_source(builder())
