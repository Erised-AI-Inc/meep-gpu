"""Host-side contract for the CONDUCTIVE Metal H->D weld.

WHAT THESE TESTS PIN AND WHAT THEY DELIBERATELY DO NOT. Byte identity against the array
path is the GATE's claim and needs a device
(``parity/meep_gpu/gate_metal_conductive_fused_hd_pair.py``); what a host suite can hold
is the LIFT -- that the emitted kernel is the two certified emitters' own text with
exactly the declared edits -- the SIGNATURE arithmetic, the PREDICATE's inversions, and
the declarations the composer and the board read. Those are the places a later edit
turns a green gate into a stale one without failing anything.

Every test here runs without ``torch``: the emitters are pure string builders and the
predicates are pure Python over duck-typed objects. The two that need a device declare
themselves skipped rather than passing quietly.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import re
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pytest

from meep_gpu.metal_kernels import (
    coefficient_pack, conductive_fused_hd_pair as family, conductive_pml,
    fused_hd_pair as plain, shaders,
)
from meep_gpu.metal_kernels.device import MAX_BUFFER_BINDINGS
from meep_gpu.metal_kernels.launch import SUB_STEPS

#: The specialisations every text test walks: both boundary extremes and both lossy
#: shapes, because the lift's needles resolve against the emitted text and the emitted
#: text is what specialisation changes.
CODES: Tuple[Tuple[int, int, int], ...] = ((0, 0, 0), (1, 1, 1), (1, 0, 1))
LOSSY: Tuple[Tuple[bool, bool, bool], ...] = ((True, True, True),
                                              (True, False, True),
                                              (False, True, False))


def sources() -> Sequence[Tuple[Tuple[int, ...], Tuple[bool, ...], str]]:
    return [(codes, lossy, family.conductive_fused_hd_pair_source(codes, lossy))
            for codes in CODES for lossy in LOSSY]


# ---------------------------------------------------------------------------
# The declarations the composer and the board read
# ---------------------------------------------------------------------------

def test_the_span_is_the_drivers_two_adjacent_consults():
    assert family.REPLACES == ("update_H", "step_D")
    assert family.SLOT == "update_H"
    # THE SLOT IS THE FIRST HALF IN DRIVER ORDER, so a refusal is named on the slot
    # the fusion starts at.
    assert family.SLOT == family.REPLACES[0]


def test_the_seam_name_is_spelled_through_withdraw_hoist():
    from meep_gpu import withdraw_hoist

    assert family.SEAM is withdraw_hoist.SEAM


def test_nothing_is_injected_in_this_seam_so_the_repair_flag_is_false():
    """A FACT ABOUT THE DRIVER, not a choice -- and the complement is asserted too.

    ``CARRIES_DEPOSIT_REPAIR`` False is only honest if the predicate really does not
    consult ``deposit_repair``; a flag at False beside a live consult would be a
    product that refuses rows it could serve.
    """
    import inspect

    assert family.CARRIES_DEPOSIT_REPAIR is False
    body = inspect.getsource(family.metal_conductive_fused_hd_pair_coverage)
    # NO CALL, only the comment that says why there is none. The distinction matters:
    # a product that CONSULTED deposit_repair while declaring the flag False would
    # refuse rows it can serve, and one that declared True without bracketing its
    # launch would compute the constitutive half against a pre-injection field.
    code = "\n".join(line for line in body.splitlines()
                     if not line.strip().startswith("#"))
    assert "deposit_repair." not in code
    assert "seam_withdraw_reasons" in code


def test_the_withdraw_flag_and_its_wiring_move_together():
    """``HOISTS_THE_WITHDRAW`` True without the installer's wrapper is a silent defect.

    The only wiring that performs the hoist is ``launch._install_fused_pair``'s
    ``withdraw_hoist.SEAM`` branch, which is unreachable while ``INSTALLABLE`` is
    False. So True here must imply installable; the pair is pinned rather than each
    flag alone.
    """
    assert family.HOISTS_THE_WITHDRAW is False
    assert not (family.HOISTS_THE_WITHDRAW and not family.INSTALLABLE)


def test_the_installable_reason_is_a_measurement_not_a_policy():
    assert family.INSTALLABLE is False
    reason = family.INSTALLABLE_REASON
    # The three things a reader must be able to find: the arithmetic, the incumbent,
    # and what the verdict does NOT license.
    assert "4 - (installed pairs)" in reason
    assert "conductive_fused_electric_pair" in reason
    assert "no timing claim" in reason


def test_the_family_is_welded_and_the_declaration_it_replaced_is_kept():
    """The partition's other half: this family has a GATE and now a ledger entry.

    ``test_metal_weld_contract.test_every_metal_family_is_welded`` partitions the
    ``gate_metal_*.py`` fleet against ``fingerprints.json`` and requires the declaring
    set and the welded set to be DISJOINT and to EXHAUST it, so the two halves move
    together and each prevents a different failure: emptying ``WELD_OWED`` without the
    entry claims a released gate this tree cannot show, and cutting the entry without
    emptying the string leaves a welded family still declaring its weld is owed. BOTH
    are checked here rather than either being taken on trust.

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
    assert ("metal_conductive_fused_hd_pair_device_gate"
            in metal_launch.load_fingerprints()), (
        "WELD_OWED is empty and fingerprints.json carries no "
        "metal_conductive_fused_hd_pair_device_gate: one of the two is false")
    assert ("results/" in family._RETIRED_WELD_OWED
            and len(family._RETIRED_WELD_OWED) > 200), (
        "the retired declaration must still name the artifact and what the release "
        "owed; a weld does not license deleting the record it replaced")


# ---------------------------------------------------------------------------
# The signature — the whole reason this product exists
# ---------------------------------------------------------------------------

def test_the_shipped_signature_is_counted_off_its_own_source():
    assert family.shipped_signature_bindings() == family.PACKED_BINDINGS
    assert family.PACKED_BINDINGS == family.PACKED_POINTERS + 1
    assert family.PACKED_BINDINGS <= MAX_BUFFER_BINDINGS


def test_the_vectors_only_shape_is_over_the_ceiling_by_exactly_one():
    """THE MEASUREMENT THAT MAKES THE SECOND PACK LOAD-BEARING.

    With only the nine per-axis vectors packed the signature is one binding past the
    platform ceiling, so the conductivity-volume pack is a NECESSITY rather than a
    tidiness. The gate compiles the shape and requires the failure; this pins the
    arithmetic so a future edit that shrinks the signature cannot leave the constant
    claiming a refusal that no longer holds.
    """
    assert family.VECTORS_ONLY_BINDINGS == MAX_BUFFER_BINDINGS + 1
    assert family.UNPACKED_POINTER_BINDINGS == family.UNPACKED_POINTERS + 1
    assert family.UNPACKED_POINTERS > family.PACKED_POINTERS
    # The pack saves exactly the members it holds, minus the two pointers it costs.
    saved = family.UNPACKED_POINTERS - family.PACKED_POINTERS
    assert saved == len(family.PACKED_VECTORS) + len(family.PACKED_VOLUMES) - 2


def test_every_refuted_signature_declares_the_count_it_claims():
    """Each builder asserts its own binding count; calling them proves the claim.

    The builders are the gate's, and they raise on a mismatch. Running them here means
    a change to the touch-kernel emitter fails in the host suite rather than at the
    next device campaign.
    """
    for builder, count in (
            (family.refuted_vectors_only_source, family.VECTORS_ONLY_BINDINGS),
            (family.refuted_unpacked_pointer_source,
             family.UNPACKED_POINTER_BINDINGS),
            (family.refuted_separate_scalar_source,
             family.SEPARATE_SCALAR_BINDINGS)):
        source = builder()
        assert source.count("[[buffer(") == count
        assert count > MAX_BUFFER_BINDINGS


def test_the_two_packs_are_disjoint_and_single_geometry():
    """One pack per GEOMETRY, which is what the D->E twin's objection asked for.

    ``conductive_fused_electric_pair`` declined to fold volumes in beside per-axis
    vectors ("a mixed-geometry table for no binding need"); this seam has the binding
    need and answers the objection with two packs rather than one mixed table.
    """
    assert not set(family.PACKED_VECTORS) & set(family.PACKED_VOLUMES)
    assert len(set(family.PACKED_VECTORS)) == len(family.PACKED_VECTORS)
    assert len(set(family.PACKED_VOLUMES)) == len(family.PACKED_VOLUMES)
    # The vectors are the two halves' per-axis coefficients and nothing else.
    assert set(family.PACKED_VECTORS) == {
        "kmx", "sinvx", "kmy", "sinvy", "kmz", "sinvz", "kp0", "kp1", "kp2"}
    assert set(family.PACKED_VOLUMES) == {"cf0", "cf1", "cf2", "ci0", "ci1", "ci2"}


def test_the_constitutive_km_group_is_the_curls_and_that_is_asserted_not_assumed():
    """The shared ``kms`` group is the premise the packed signature rests on."""
    assert SUB_STEPS["step_D"]["suffix"] == ""
    from meep_gpu.triton_kernels.coverage import CONSTITUTIVE_SIDES

    assert CONSTITUTIVE_SIDES["H"]["half_integer"] is False
    # The constitutive's km0/km1/km2 are therefore NOT pack members: the lift renames
    # the reads to the curl's kmx/kmy/kmz instead of binding a second group.
    assert not {"km0", "km1", "km2"} & set(family.PACKED_VECTORS)


# ---------------------------------------------------------------------------
# The lift — the emitted kernel is the certified text
# ---------------------------------------------------------------------------

def test_every_magnetic_read_is_a_register_or_a_recompute():
    """No ``gN[`` spelling survives, on every specialisation.

    In the fused signature that pointer does not exist; a surviving read would not
    compile, but a read that moved to another pointer WOULD, and this is the check that
    the redirect is total rather than mostly total.
    """
    for codes, lossy, source in sources():
        tail = family.conductive_welded_curl_tail(codes, lossy)
        for target in range(3):
            assert f"g{target}[" not in tail, (codes, lossy, target)
        assert f"g0[" not in source.split("__CURL__")[-1]


def test_the_lift_redirects_exactly_the_nine_magnetic_loads():
    """Three own-cell loads to registers, six shifted loads to ``h_cell`` calls."""
    for codes, lossy, source in sources():
        assert source.count("h_cell(") == 1 + 1 + len(plain.HALO_TAPS)
        assert "float a = own.a0, b = own.a1, c = own.a2;" in source
        for var, target in plain.HALO_TAPS:
            assert re.search(rf"{var} = \w+ \? h_cell\([^)]*\)\.a{target} : 0\.0f",
                             source), (codes, lossy, var)


def test_the_ghost_ternary_and_its_guard_are_the_emitters_own():
    """The validity flag and the exact ``0.0f`` are PARSED, never retyped.

    A metallic ghost that read an exact ``0.0f`` must still read an exact ``0.0f``, and
    the periodic wrap must still be the emitter's own integer expression. Both are
    checked against the certified curl's own text rather than against a literal.
    """
    for codes, lossy, source in sources():
        certified = conductive_pml.conductive_pml_curl_source(
            codes, bool(SUB_STEPS["step_D"]["backward"]), lossy)
        for line in certified.splitlines():
            stripped = line.strip()
            if stripped.startswith("vx = ") or stripped.startswith("sj = ") \
                    or stripped.startswith("sk = ") or stripped.startswith("vy = ") \
                    or stripped.startswith("vz = ") or stripped.startswith("si = "):
                assert stripped in source, (codes, lossy, stripped)
        # And the ghost value is still an exact zero literal on every tap.
        assert source.count(" : 0.0f") >= len(plain.HALO_TAPS)


def test_the_conductive_tail_survives_the_lift_unedited():
    """The four-branch tail touches no magnetic pointer, so the lift must not touch it.

    Every line of the certified tail below the curl must appear in the welded body
    character for character; a lift that "tidied" one of them would be arithmetic this
    family does not own.
    """
    for codes, lossy, _source in sources():
        certified = conductive_pml.conductive_pml_curl_source(
            codes, bool(SUB_STEPS["step_D"]["backward"]), lossy)
        welded = family.conductive_welded_curl_tail(codes, lossy)
        for line in certified.splitlines():
            stripped = line.strip()
            # THE TAIL'S OWN LINES, NAMED. A prefix like "float c" would also catch
            # the curl's magnetic loads `float c_x = ...`, which the lift DOES
            # redirect -- and a test that demanded those survive unedited would
            # contradict the whole shape.
            if not re.match(r"(float [fuc]\d_(previous|new)|bool dsigu?\d|"
                            r"[fuc]\d\[ii\] = )", stripped):
                continue
            assert stripped in welded, (codes, lossy, stripped)


def test_the_decode_anchor_is_this_emitters_own_and_not_the_packages():
    """The conductive template declares ``j`` and ``i`` on ONE line.

    ``offdiag_weld_common.DECODE_END`` is the certified curl's two-line spelling, and
    using it here would splice a truncated body. The two must stay different, and this
    family's must still be present in the emitter's output.
    """
    from meep_gpu.metal_kernels import offdiag_weld_common

    assert family.DECODE_END != offdiag_weld_common.DECODE_END
    for codes, lossy, _source in sources():
        certified = conductive_pml.conductive_pml_curl_source(
            codes, bool(SUB_STEPS["step_D"]["backward"]), lossy)
        assert family.DECODE_END in certified


def test_the_constitutive_half_is_the_plain_products_lift():
    """The ``update_H`` body is :mod:`.fused_hd_pair`'s ``h_cell``, not a copy.

    Sharing the lift is what keeps the two welds' constitutive halves from drifting
    apart, so it is pinned as an identity of the emitted text rather than described.
    """
    for _codes, _lossy, source in sources():
        assert plain.h_cell_function() in source


def test_a_lossless_target_names_no_conductivity_member():
    """THE SENTINEL RULE, on the emitted text.

    The plan writes offset 0 for a lossless target's two pack members and stores
    nothing for them; the premise is that the lossless tail never indexes them.
    """
    for codes in CODES:
        for lossy in LOSSY:
            source = family.conductive_fused_hd_pair_source(codes, lossy)
            for member in family.unread_conductivity_members(lossy):
                assert f"{member}[" not in source, (codes, lossy, member)


def test_a_lossy_target_does_name_its_conductivity_members():
    """THE COMPLEMENT, which is what makes the test above worth having.

    A rule that held because NOTHING indexes those members would be vacuous.
    """
    for codes in CODES:
        for lossy in LOSSY:
            source = family.conductive_fused_hd_pair_source(codes, lossy)
            for index, on in enumerate(lossy):
                if not on:
                    continue
                assert f"cf{index}[" in source, (codes, lossy, index)
                assert f"ci{index}[" in source, (codes, lossy, index)


def test_the_pack_prologue_recreates_every_member_under_its_own_name():
    """That is why the lifted bodies need no edit."""
    for _codes, _lossy, source in sources():
        for member in family.PACKED_VECTORS:
            assert (f"device const float* {member} = cpack + prm.off_{member};"
                    in source)
        for member in family.PACKED_VOLUMES:
            assert (f"device const float* {member} = vpack + prm.off_{member};"
                    in source)


def test_the_params_struct_carries_one_offset_per_pack_member():
    for _codes, _lossy, source in sources():
        struct = source.split("struct Params {", 1)[1].split("};", 1)[0]
        for member in family.PACKED_VECTORS + family.PACKED_VOLUMES:
            assert f"uint off_{member};" in struct
        assert struct.count("uint off_") == (len(family.PACKED_VECTORS)
                                             + len(family.PACKED_VOLUMES))


def test_the_wall_clear_is_not_carried():
    """``zero_metal_B`` closes one seam earlier and ``zero_metal_D`` opens one later.

    A mask here would be a pass the driver runs again.
    """
    for _codes, _lossy, source in sources():
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
        self.bfast_active = False
        self.dimensions = 3
        self.k_point = (0.0, 0.0, 0.0)
        for key, value in overrides.items():
            setattr(self, key, value)

    def is_mirrored(self, axis: int) -> bool:  # noqa: ARG002
        return False


def _coverage_is_a_conjunction() -> None:
    """The predicate must never admit what either half refuses."""


def test_the_predicate_is_a_conjunction_of_the_two_halves():
    """Not a paraphrase of them -- the two functions are CALLED.

    A predicate that restated its halves' clauses would drift from them silently, so
    the test reads the source for the two call sites rather than trying to build a
    configuration for every clause.
    """
    import inspect

    body = inspect.getsource(family.metal_conductive_fused_hd_pair_coverage)
    assert "constitutive_coverage(fields, pml, \"H\", residency)" in body
    assert "metal_conductive_pml_curl_coverage(\n        fields, pml, \"step_D\"" in body
    assert "constitutive half:" in body
    assert "curl half:" in body


def test_the_predicate_refuses_a_fields_with_no_grid():
    verdict = family.metal_conductive_fused_hd_pair_coverage(object(), None, ())
    assert not verdict.covered
    assert any("grid" in reason for reason in verdict.reasons)


def test_an_undeclared_source_set_is_refused_rather_than_assumed_empty():
    """IGNORANCE IS NEVER AN EMPTY SET.

    ``Fields`` does not hold the source list, so a predicate that inferred "no
    withdraw stands" from not being told would be an over-covering refusal.
    """

    class _Fields:
        grid = _Grid()

    verdict = family.metal_conductive_fused_hd_pair_coverage(_Fields(), None, None)
    assert not verdict.covered
    assert any("the source set was not declared" in reason
               for reason in verdict.reasons)


def test_the_fold_is_refused_by_name_on_this_seam_for_this_seams_reason():
    """Neither fill runs between these two consults, so the fold's reason is the ARM."""

    class _Folded(_Grid):
        def is_mirrored(self, axis: int) -> bool:
            return axis == 1

    class _Fields:
        grid = _Folded()

    verdict = family.metal_conductive_fused_hd_pair_coverage(_Fields(), None, ())
    assert not verdict.covered
    joined = " | ".join(verdict.reasons)
    assert "axis 1 is folded" in joined
    assert "conductive PML curl" in joined


def test_unread_conductivity_members_names_exactly_the_lossless_pairs():
    assert family.unread_conductivity_members((True, True, True)) == ()
    assert family.unread_conductivity_members((True, False, True)) == ("cf1", "ci1")
    assert family.unread_conductivity_members((False, False, False)) == (
        "cf0", "ci0", "cf1", "ci1", "cf2", "ci2")
    with pytest.raises(ValueError):
        family.unread_conductivity_members((True, False))


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_the_arm_is_registered_unwired_on_the_first_half_of_the_seam():
    """REGISTERED SO IT IS ENUMERABLE, NOT SO IT IS SELECTABLE."""
    from meep_gpu.metal_kernels import arms

    rows = [spec for spec in arms.registered("update_H")
            if spec.family == family.FAMILY]
    assert len(rows) == 1
    assert rows[0].wired is False
    assert rows[0].replaces == family.REPLACES
    assert rows[0].is_weld is True
    # And it is skipped by the composer's own view of the table.
    labels = [spec.family for spec in arms.registered("update_H") if spec.wired]
    assert family.FAMILY not in labels


def test_the_arm_refuses_every_slot_that_is_not_its_own():
    for slot in ("step_B", "step_D", "update_E", "update_P"):
        verdict = family._arm_coverage(object(), slot)  # noqa: SLF001
        assert not verdict.covered
        assert any("cannot fill" in reason for reason in verdict.reasons)
        assert family._arm_plan(object(), slot) is None  # noqa: SLF001


# ---------------------------------------------------------------------------
# The two device-dependent claims, declared skipped rather than passing quietly
# ---------------------------------------------------------------------------

def _mps() -> bool:
    try:
        import torch
    except ImportError:  # pragma: no cover - a host without torch
        return False
    return bool(torch.backends.mps.is_available())


@pytest.mark.skipif(not _mps(), reason="no MPS device on this host")
def test_the_shipped_signature_compiles_on_every_specialisation():
    from meep_gpu.metal_kernels.device import compile_source

    for codes, lossy, source in sources():
        compile_source(source)  # a compile failure raises


@pytest.mark.skipif(not _mps(), reason="no MPS device on this host")
def test_the_three_refuted_signatures_do_not_compile():
    """THE CEILING IS A MEASUREMENT, not an argument.

    The gate makes the same three compiles; repeating them here means a toolchain that
    raised the ceiling fails the host suite too, instead of leaving three constants
    claiming a refusal nothing checks.
    """
    from meep_gpu.metal_kernels.device import compile_source

    for builder in (family.refuted_vectors_only_source,
                    family.refuted_unpacked_pointer_source,
                    family.refuted_separate_scalar_source):
        with pytest.raises(Exception):
            compile_source(builder())
