"""Laptop tests for the FOLDED Metal H->D weld (folded ``update_H`` into folded
``step_D``).

THE MERGE BAR, NOT THE CERTIFICATION. The byte claim owed by this family is a device
gate that runs complete driver steps and compares uint32 words
(``parity/meep_gpu/gate_metal_folded_fused_hd_pair.py``); these tests are what must stay
green on every change, and they are chosen for the defects a byte gate would catch LATE,
on a configuration nobody sweeps, or not at all:

* **THE CODES.** This family's single point of failure. ``folded_axis_kinds`` resolves
  four codes and the plain product's ``1 if kind == "metallic" else 0`` expression maps
  a fold to PERIODIC, which makes the ghost a WRAP and deletes the cell-0 mask. It is a
  smooth wrong answer on every folded axis and the emitter cannot catch it, so the
  source of the codes is asserted here, in three separate ways;
* **the lift.** The curl body is spliced from ``symmetry.folded_curl_source``'s own
  output rather than retyped, and "spliced" is a hypothesis until something compares the
  strings. The tests reconstruct the certified folded body independently and assert that
  the ONLY lines that moved are the nine magnetic loads, that the two FOLD-SPECIFIC
  blocks (the top-plane mask and the widened cell-0 mask) survive character for
  character, and that on an unfolded triple the emission reduces to
  :mod:`.fused_hd_pair`'s plus exactly one dead ``last_*`` declaration;
* **the phase blindness.** The parity is a compile-time SOURCE specialisation everywhere
  else in this family, so "this kernel does not carry it" is falsifiable and is asserted
  by digest at both parities, against a fill emitter whose source DOES differ -- which
  is what keeps the equality from being a coincidence of the harness;
* **the ghost.** On a fold the backward ghost is the METALLIC literal ``0.0f`` and NOT
  the array path's ``parity * H[2]``, which is legal only because the value is dead
  under a mask. The exact literal and the parsed guard are what a redirect is most
  likely to eat;
* **the binding ceiling**, which is an EQUALITY here: 31 of 31, no headroom, and the
  fold adds no argument. A future edit that adds a volume discovers it in this file;
* **the predicate's refusals**, every one answerable without a GPU, INCLUDING BOTH
  DIRECTIONS of the inverted fold clause -- disjointness is a property of a pair of
  predicates and neither one alone can carry it;
* **the declarations.** This product is registered UNWIRED and gated on the fold, holds
  a ``FUSED_PAIR_ARMS`` absorb row (2026-09-07) so the seam loop can ask it, declares
  ``INSTALLABLE = False`` on a measured arbitration, and declares ``WELD_OWED`` EMPTY
  because its ledger entry has landed. Each of those is a claim a reader will act on;
* **the scratch and the rotation**, pure host logic driven against a fake residency, so
  the disjointness refusal and the buffer swap are measured on a machine with no device.

Everything that needs a device is guarded and skipped, so this file is the merge bar on
a host with no MPS as well as on this one.

WHAT THIS FILE DOES NOT CLAIM. Nothing here is a byte-identity measurement and nothing
here licenses installing the product. The cell's own numbers -- 78 rows in the
``(folded -> folded)`` cell, of which this predicate reaches 75 and refuses 3 by name --
come from ``parity/meep_gpu/results/h_to_d_seam_2026-09-04/h_to_d_seam.jsonl`` and are
asserted against that record where it is present rather than restated.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from meep_gpu import withdraw_hoist
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid, Mirror
from meep_gpu.metal_kernels import (
    arms,
    folded_fused_hd_pair as folded,
    fused_hd_pair as plain,
    launch as metal_launch,
    offdiag_weld_common as weld,
    registry,
    shaders,
    symmetry,
    templates,
)
from meep_gpu.metal_kernels.device import MAX_BUFFER_BINDINGS
from meep_gpu.pml import PML
from meep_gpu.triton_kernels.coverage import CONSTITUTIVE_SIDES
from meep_gpu.triton_kernels.symmetry import (
    CODE_METALLIC as M, CODE_MIRROR_METALLIC as MM, CODE_MIRROR_PERIODIC as MP,
    CODE_PERIODIC as P,
)


def _torch_mps() -> bool:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False
    return bool(getattr(getattr(torch, "backends", None), "mps", None)
                and torch.backends.mps.is_available())


needs_mps = pytest.mark.skipif(not _torch_mps(),
                               reason="no MPS device on this host")

#: THE SPECIALISATIONS THE STRING TESTS RUN OVER, and they are the corpus cell's own
#: seven boundary triples plus three the cell does not carry: the unfolded ``(P,P,P)``
#: (where the emission must reduce to the plain product's), a single folded metallic
#: axis and a single folded periodic one. The ghost rule and the two masks are exactly
#: what specialisation changes, so a representative triple would prove nothing about
#: the others -- the top-plane mask is emitted on MIRROR_PERIODIC and NOT on
#: MIRROR_METALLIC, and a sweep on one termination says nothing about the other.
CODES = (
    (MM, MM, P),    # 36 of the cell's 78 rows
    (M, MM, P),     # 26 rows: a live non-folded wall beside a folded metallic axis
    (P, MP, P),     # 10 rows: the far pass and one top-plane mask line
    (MP, MP, P),    # 2 rows: two top-plane lines and a doubly-unowned corner
    (M, MM, MM),    # 2 rows, 3-D
    (P, MP, MP),    # 1 row, 3-D, odd full count
    (MM, MM, MM),   # 1 row, three folded metallic axes
    (P, P, P),      # NOT in the cell: the reduction to the plain product
    (MM, P, P),
    (MP, P, P),
)
MODES = shaders.CONTRACT_MODES

#: The cell's own numbers, asserted against the seam record where it is present.
CELL_ROWS = 78
CELL_REACH = 75
CELL_WITHDRAW_ROWS = 3


# ---------------------------------------------------------------------------
# The declarations
# ---------------------------------------------------------------------------

def test_the_span_is_the_seam_the_withdraw_module_owns():
    """``REPLACES`` and ``SEAM`` are read from :mod:`meep_gpu.withdraw_hoist`.

    A product whose span disagreed with that module's ``SEAM_SPAN`` would be refused by
    ``hoistable``'s span clause the moment it ever declared the hoist -- but only then.
    """
    assert folded.REPLACES == withdraw_hoist.SEAM_SPAN == ("update_H", "step_D")
    assert folded.SEAM == withdraw_hoist.SEAM == "H_to_D"
    assert folded.SLOT == "update_H" == folded.REPLACES[0]


def test_the_span_is_the_plain_products_and_so_is_the_signature_shape():
    """Same seam, same shape, and the constants are IMPORTED rather than re-spelled.

    The fold adds no kernel argument on this backend (symmetry.py:328-332), so this
    product's signature IS the plain one's -- and a second spelling of the four binding
    counts would be a second place for the same measurement to go stale.
    """
    assert folded.REPLACES == plain.REPLACES
    assert folded.ROTATED_NAMES is plain.ROTATED_NAMES
    for name in ("PACKED_BINDINGS", "SEPARATE_SCALAR_BINDINGS",
                 "UNSHARED_KMS_BINDINGS", "ONE_MORE_POINTER_BINDINGS"):
        assert getattr(folded, name) == getattr(plain, name), name
    for name in ("refuted_separate_scalar_source", "refuted_unshared_kms_source",
                 "refuted_one_more_pointer_source"):
        assert getattr(folded, name) is getattr(plain, name), name


def test_the_two_seam_declarations_are_both_false_and_say_why():
    """``CARRIES_DEPOSIT_REPAIR`` and ``HOISTS_THE_WITHDRAW`` are separate claims.

    Both False, for two DIFFERENT reasons -- the first because nothing is injected in
    this seam at all, the second because the wiring that would perform the hoist is
    unreachable for an uninstallable product. A product that declared one must not
    thereby acquire the other's licence.
    """
    assert folded.CARRIES_DEPOSIT_REPAIR is False
    assert folded.HOISTS_THE_WITHDRAW is False
    assert folded.INSTALLABLE is False
    text = Path(folded.__file__).read_text(encoding="utf-8")
    assert "nothing is injected" in text
    assert "_install_fused_pair" in text and "unreachable" in text


def test_the_family_is_welded_and_holds_its_absorb_row():
    """The two declarations a reader will act on, and both are partitions.

    WIRED 2026-09-07, so both flipped together. ``WELD_OWED`` empty is a CLAIM that
    ``fingerprints.json`` carries this family's weld entry --
    ``test_metal_weld_contract.test_every_metal_family_is_welded`` requires the
    declaring set and the welded set to be disjoint AND to exhaust the gate fleet, so
    an honest refusal and a forgotten weld stay distinguishable, and emptying the
    string without the entry fails there by name. The ``launch.FUSED_PAIR_ARMS`` row
    is the other half: it lets the seam loop ASK this product, which then refuses it
    on ``INSTALLABLE = False`` and, with that flag out of the way, on the released
    ``folded_fused_magnetic_pair`` holding ``update_H`` first.
    """
    assert folded.WELD_OWED == "", (
        "WELD_OWED must be empty once fingerprints.json carries "
        "metal_folded_fused_hd_pair_device_gate")
    # The weld entries are TOP-LEVEL keys of fingerprints.json, beside the
    # "metal_kernels" digest block.
    assert "metal_folded_fused_hd_pair_device_gate" in (
        metal_launch.load_fingerprints())
    assert metal_launch.FUSED_PAIR_ARMS[folded.FAMILY] == ("folded", "folded")
    assert "78 of 78" in folded.INSTALLABLE_REASON, (
        "the uninstallable declaration must carry the MEASURED join, not a policy")
    assert "no timing" in folded.INSTALLABLE_REASON


def test_the_module_is_imported_by_the_registry_and_says_so():
    """REGISTERED IN A NORMAL RUN SINCE 2026-09-07, AND DECLARED.

    ``registry.FAMILY_MODULES`` is what imports a family, so a module listed there
    contributes its arm to every composition's table -- ``wired=False``, so
    ``plan_step`` still cannot select it, and the seam loop reaches it through the
    absorb row instead. That is the state this round ships, and a reader must not have
    to discover it: the module says so and this test pins the pair, in both
    directions.
    """
    stem = Path(folded.__file__).stem
    text = Path(folded.__file__).read_text(encoding="utf-8")
    assert stem in registry.FAMILY_MODULES, (
        "the wiring landed the FAMILY_MODULES row; a module dropped back out of it "
        "must say so where a reader will see it")
    assert "not in ``registry.FAMILY_MODULES``" not in text and \
        "NOT in ``registry.FAMILY_MODULES``" not in text, (
            "the module is in FAMILY_MODULES now; its prose must stop saying "
            "nothing imports it")
    assert "IS in ``registry.FAMILY_MODULES``" in text


# ---------------------------------------------------------------------------
# THE CODES — this family's single point of failure
# ---------------------------------------------------------------------------

def test_the_plan_takes_its_codes_from_folded_axis_kinds_and_not_from_boundary_kinds():
    """Read off the module's own source, because the defect is a SMOOTH wrong answer.

    ``stepping._boundary_kinds`` reports ``"mirror"`` on a folded axis, and the plain
    product's ``1 if kind == "metallic" else 0`` maps that to 0 = PERIODIC: the backward
    ghost becomes a WRAP to the far plane AND ``templates.ownership_mask`` emits no
    cell-0 mask. Neither the emitter nor the compiler can catch it, since 0 and 1 are
    valid codes -- so the SOURCE of the codes is the thing to pin.
    """
    import inspect  # noqa: PLC0415

    body = inspect.getsource(folded.plan_metal_folded_fused_hd_pair)
    # THE EXECUTABLE LINES ONLY. The docstring and the comments NAME the defect on
    # purpose -- that is where a reader learns what not to do -- so a text search over
    # the whole function would refuse the module for documenting itself.
    body = body.split('"""', 2)[-1]
    code = "\n".join(line for line in body.splitlines()
                     if not line.strip().startswith("#"))
    assert "symmetry.folded_axis_kinds(grid, pml)" in code
    assert "_boundary_kinds" not in code, (
        "the plan builder must not reach for _boundary_kinds; that expression loses "
        "the MIRROR_METALLIC / MIRROR_PERIODIC split, which decides the top-plane mask")


def test_the_two_code_sources_disagree_on_every_folded_axis():
    """THE DEFECT IS REAL AND IS MEASURED HERE, not merely described.

    On a folded grid the two expressions produce DIFFERENT triples, and the difference
    is exactly the fold: ``folded_axis_kinds`` returns a MIRROR code where the plain
    expression returns PERIODIC.
    """
    from meep_gpu.stepping import _boundary_kinds  # noqa: PLC0415

    grid = _folded_grid(axes="Y")
    pml = _pml(grid)
    codes, reasons = symmetry.folded_axis_kinds(grid, pml)
    assert codes is not None, reasons
    plain_codes = tuple(1 if kind == "metallic" else 0
                        for kind in _boundary_kinds(grid, pml))
    folded_axes = [axis for axis in range(3) if int(codes[axis]) in symmetry.MIRROR_CODES]
    assert folded_axes, "the fixture is not folded; this test would prove nothing"
    for axis in folded_axes:
        assert plain_codes[axis] == 0, (
            "the plain expression maps a fold to PERIODIC -- that IS the defect")
        assert int(codes[axis]) != 0


@pytest.mark.parametrize("codes", [c for c in CODES
                                   if any(x in symmetry.MIRROR_CODES for x in c)])
def test_the_wrong_reduction_changes_the_ghost_and_deletes_the_cell_zero_mask(codes):
    """What the wrong codes DO to the emitted text, in the two places they show.

    Reducing the mirror codes to PERIODIC rather than METALLIC is the same defect
    expressed in the emitter: the ghost stops serving an exact literal and starts
    wrapping, and the cell-0 mask for that axis disappears. Both are asserted so a
    future change to either emitter cannot make the mutation inert without a red here.
    """
    backward = bool(metal_launch.SUB_STEPS["step_D"]["backward"])
    right = symmetry._reduced_codes(codes)  # noqa: SLF001
    wrong = tuple(shaders.PERIODIC if int(code) in symmetry.MIRROR_CODES
                  else (shaders.METALLIC if int(code) == M else shaders.PERIODIC)
                  for code in codes)
    assert right != wrong
    folded_axis = next(index for index, code in enumerate(codes)
                       if int(code) in symmetry.MIRROR_CODES)
    axis = "xyz"[folded_axis]
    # THE GHOST: the METALLIC reduction emits a VALIDITY FLAG, which the template's
    # load line turns into an exact `0.0f`; the PERIODIC one emits an integer WRAP to
    # the far plane and no flag at all, so the load reads a real cell there.
    right_ghost = templates.ghost(axis, right[folded_axis], backward)
    wrong_ghost = templates.ghost(axis, wrong[folded_axis], backward)
    assert right_ghost != wrong_ghost
    flag = ("vx", "vy", "vz")[folded_axis]
    shifted = ("si", "sj", "sk")[folded_axis]
    assert f"{flag} =" in right_ghost and "?" not in right_ghost, right_ghost
    assert f"{shifted} = ({shifted} < 0) ?" in wrong_ghost, wrong_ghost
    # THE MASK: the wrong reduction deletes the cell-0 lines for this axis, which is
    # the half of the defect a ghost comparison cannot see.
    right_mask = templates.ownership_mask(right, backward)
    wrong_mask = templates.ownership_mask(wrong, backward)
    at = ("at_x", "at_y", "at_z")[folded_axis]
    assert at in right_mask, right_mask
    assert at not in wrong_mask, wrong_mask


# ---------------------------------------------------------------------------
# The lift
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("codes", CODES)
def test_the_curl_lift_moves_exactly_the_nine_magnetic_loads(codes):
    """Line for line against the FOLDED emitter's own output.

    Three own-cell loads become registers and six halo loads become recomputes; every
    other line -- the guard, the decode, the ghost gather, the curl grouping, both
    masks, the split-field recurrence and the six stores -- must be character-identical.
    """
    backward = bool(metal_launch.SUB_STEPS["step_D"]["backward"])
    certified = symmetry.folded_curl_source(codes, backward)
    tail = certified.split(plain.DECODE_END, 1)[1]
    tail = tail[: -len("}\n")]
    welded = folded.folded_welded_curl_tail(codes)
    before, after = tail.splitlines(), welded.splitlines()
    assert len(before) == len(after), "the lift changed the LINE COUNT"
    changed = [index for index, (a, b) in enumerate(zip(before, after)) if a != b]
    assert len(changed) == 9, (
        f"{len(changed)} lines moved, expected the three own-cell loads and the six "
        f"halo taps: {[before[index] for index in changed]}")
    for index in changed:
        assert "g0[" in before[index] or "g1[" in before[index] \
            or "g2[" in before[index], before[index]


@pytest.mark.parametrize("codes", CODES)
def test_the_two_fold_blocks_survive_the_lift_character_for_character(codes):
    """The top-plane mask and the widened cell-0 mask are UNTOUCHED by the redirect.

    That is a property of what they touch rather than an exemption: both write
    ``curlN = flag ? 0.0f : curlN;`` and neither indexes a magnetic pointer, so the lift
    -- which moves only the nine magnetic loads -- cannot reach them. Asserting it is
    what keeps a future widening of the redirect from silently eating a mask.
    """
    backward = bool(metal_launch.SUB_STEPS["step_D"]["backward"])
    welded = folded.folded_welded_curl_tail(codes)
    top = symmetry.folded_top_plane_mask(codes, backward)
    cell_zero = symmetry.folded_cell_zero_mask(codes, backward)
    assert top in welded, "the top-plane mask did not survive the lift"
    assert cell_zero in welded, "the cell-0 mask did not survive the lift"
    if any(int(code) == MP for code in codes):
        assert "curl" in top, "a MIRROR_PERIODIC axis must emit a top-plane line"
    else:
        assert "curl" not in top, (
            "a top-plane mask on a MIRROR_METALLIC axis deletes a STEPPED cell "
            "(symmetry.py:533-536)")


@pytest.mark.parametrize("codes", CODES)
def test_no_magnetic_pointer_survives_anywhere_in_the_fused_kernel(codes):
    """In this signature ``g0``/``g1``/``g2`` do not exist.

    A missed redirect is a compile error only by luck; in the fused kernel the names
    ``f``, ``w`` and ``g`` all exist and mean something else, so a survivor is a smooth
    wrong answer. The check is on the WHOLE source, not the tail.
    """
    source = folded.folded_fused_hd_pair_source(codes)
    body = source.split("uint idx [[thread_position_in_grid]])", 1)[1]
    for target in range(3):
        assert f"g{target}[" not in body, target


@pytest.mark.parametrize("codes", CODES)
def test_every_recompute_lands_on_the_cell_the_certified_index_composed(codes):
    """The tap coordinates are PARSED from the emitter's index lines, never tabulated.

    ``ox = si * nyz + j * nzi + k`` is the emitter's own composition and
    ``offset_coordinates`` inverts it, so a change to how that index is spelled RAISES
    rather than quietly redirecting a tap one cell over.
    """
    backward = bool(metal_launch.SUB_STEPS["step_D"]["backward"])
    tail = symmetry.folded_curl_source(codes, backward).split(
        plain.DECODE_END, 1)[1][: -len("}\n")]
    offsets = plain.offset_coordinates(tail)
    assert offsets == {"ox": ("si", "j", "k"), "oy": ("i", "sj", "k"),
                       "oz": ("i", "j", "sk")}
    welded = folded.folded_welded_curl_tail(codes)
    for var, target in plain.HALO_TAPS:
        axis = "xyz".index(var[-1])
        expected = list(("i", "j", "k"))
        expected[axis] = ("si", "sj", "sk")[axis]
        call = f"h_cell({', '.join(expected)}, {plain.H_CELL_TAIL_ARGS}).a{target}"
        assert call in welded, (var, call)


@pytest.mark.parametrize("codes", CODES)
def test_the_folded_ghost_is_the_metallic_literal_and_not_a_mirror_read(codes):
    """ON A FOLD THE GHOST IS AN EXACT ``0.0f``, and that is the whole legality claim.

    ``_reduced_codes`` maps both mirror codes to METALLIC, so past a mirror plane the
    kernel serves a literal zero rather than the array path's ``parity * field[2]``
    (``stepping._shift_down``:1823-1826). That is legal ONLY because the value is dead
    under the cell-0 mask (symmetry.py:30-43) -- and it means no recompute is evaluated
    past the face at all, since the ternary's guard is false there.
    """
    welded = folded.folded_welded_curl_tail(codes)
    taps = [line for line in welded.splitlines() if "h_cell(" in line and " ? " in line]
    assert len(taps) == 6, taps
    for line in taps:
        assert line.rstrip().endswith(": 0.0f;"), (
            "the ghost must still serve an exact literal zero; a recompute there would "
            "evaluate h_cell at an out-of-range cell", line)
    for axis, code in enumerate(codes):
        if int(code) in symmetry.MIRROR_CODES or int(code) == M:
            flag = ("vx", "vy", "vz")[axis]
            assert f"bool {flag} = true" in welded or f"{flag} =" in welded, axis


def test_the_unfolded_emission_reduces_to_the_plain_products_plus_one_dead_line():
    """THE STRONGEST SINGLE STATEMENT ABOUT THE FOLD'S DEVICE-CODE DELTA.

    On ``(P, P, P)`` ``symmetry.folded_curl_source`` reduces to the certified curl, so
    this product's emission must reduce to :func:`.fused_hd_pair.fused_hd_pair_source`'s
    -- comments plus ONE dead ``bool last_x = ...`` declaration, and nothing removed.
    A diff rather than a claim: if the fold ever grows real device code on an unfolded
    grid, it is found here.
    """
    import difflib  # noqa: PLC0415

    diff = list(difflib.unified_diff(
        plain.fused_hd_pair_source((0, 0, 0)).splitlines(),
        folded.folded_fused_hd_pair_source((P, P, P)).splitlines(),
        lineterm="", n=0))
    added = [line[1:] for line in diff
             if line.startswith("+") and not line.startswith("+++")]
    removed = [line[1:] for line in diff
               if line.startswith("-") and not line.startswith("---")]
    code_added = [line for line in added
                  if line.strip() and not line.strip().startswith("//")]
    code_removed = [line for line in removed
                    if line.strip() and not line.strip().startswith("//")]
    assert not code_removed, code_removed
    assert len(code_added) == 1 and "bool last_x" in code_added[0], code_added


@pytest.mark.parametrize("codes", CODES)
@pytest.mark.parametrize("mode", MODES)
def test_the_source_is_ascii_and_carries_exactly_one_contraction_pragma(codes, mode):
    source = folded.folded_fused_hd_pair_source(codes, mode)
    assert all(ord(character) < 128 for character in source)
    pragma = shaders.contraction_pragma(mode)
    if pragma.strip():
        assert source.count(pragma) == 1


def test_the_constitutive_half_is_the_plain_products_lift_and_not_a_copy():
    """No second copy of the certified ``update_H`` body exists anywhere here.

    The fold adds NO kernel on ``update_H`` (symmetry.py:1662-1702 builds the certified
    ``ConstitutivePlan``), so the plain product's ``h_cell`` IS the folded arm's
    arithmetic character for character -- and a second lift would be a second place for
    the seven declared edits to drift.
    """
    text = Path(folded.__file__).read_text(encoding="utf-8")
    assert "_plain.h_cell_function" in text
    assert "CONSTITUTIVE_LIFT_EDITS" not in text.split('"""', 2)[-1], (
        "this module must not re-declare the constitutive lift edits")
    for codes in ((P, MP, P), (MM, MM, MM)):
        source = folded.folded_fused_hd_pair_source(codes)
        assert plain.h_cell_function() in source


# ---------------------------------------------------------------------------
# Phase blindness
# ---------------------------------------------------------------------------

def test_the_kernel_is_blind_to_the_mirror_parity():
    """The emitted source is the SAME BYTES at both parities, and the fill's is not.

    The parity is a compile-time source specialisation everywhere else in this family
    (``mirror_ghost_fill_source``), which is what makes this a measurement: the fill
    emitter's source DOES differ at the two parities on the same grid, and this
    kernel's does not, because it performs no fill and its ghost is a literal zero.
    """
    even = _folded_grid(axes="Y", phases=(1,))
    odd = _folded_grid(axes="Y", phases=(-1,))
    even_codes, _ = symmetry.folded_axis_kinds(even, _pml(even))
    odd_codes, _ = symmetry.folded_axis_kinds(odd, _pml(odd))
    assert tuple(even_codes) == tuple(odd_codes), (
        "the codes must not carry the parity either; if they did, this kernel would "
        "specialise on it after all")
    assert folded.folded_fused_hd_pair_source(even_codes) == \
        folded.folded_fused_hd_pair_source(odd_codes)
    # NON-VACUITY: the emitter that DOES carry the parity must differ on these grids.
    even_entries = symmetry.ghost_fill_axis_entries(even, "B", "near")
    odd_entries = symmetry.ghost_fill_axis_entries(odd, "B", "near")
    assert even_entries and odd_entries
    def fill(entry):
        return symmetry.mirror_ghost_fill_source(
            int(entry["axis"]), int(entry["phase"]), entry["pass"], entry["shifts"])

    assert fill(even_entries[0]) != fill(odd_entries[0]), (
        "the fill emitter no longer specialises on the parity, so this kernel's byte "
        "equality has stopped being evidence of anything")


# ---------------------------------------------------------------------------
# The signature
# ---------------------------------------------------------------------------

def test_the_shipped_signature_is_the_platform_ceiling_exactly_on_a_folded_triple():
    """31 of 31, no headroom, AND THE COUNT IS TAKEN ON A FOLD.

    Counting on the degenerate unfolded emission would miss a fold that had grown an
    argument, which is exactly the failure this constant exists to catch.
    """
    assert folded.shipped_signature_bindings() == folded.PACKED_BINDINGS \
        == MAX_BUFFER_BINDINGS == 31


@pytest.mark.parametrize("codes", CODES)
def test_the_fold_adds_no_kernel_argument_on_any_specialisation(codes):
    """The signature is byte-identical across every boundary triple.

    The mirror codes, the parity, the reflect row and the ghost-source row are the four
    things a fold could have needed as a binding, and each is zero: the codes are baked
    into the source, the parity never reaches this kernel, the far image is a fill
    outside the seam, and the near ghost's source is read only by the fill.
    """
    def signature(triple):
        source = folded.folded_fused_hd_pair_source(triple)
        head = source.split("kernel void fused_hd_pair_step(", 1)[1]
        return head.split("uint idx [[thread_position_in_grid]])", 1)[0]

    reference = signature((P, P, P))
    assert signature(codes) == reference
    assert reference.count("[[buffer(") == folded.PACKED_BINDINGS


def test_both_halves_read_one_sub_lattice_which_is_what_makes_thirty_pointers():
    """The shared ``kms`` group, asserted off the SHIPPED tables rather than assumed.

    ``SUB_STEPS['step_D']['suffix']`` is ``''`` and
    ``CONSTITUTIVE_SIDES['H']['half_integer']`` is False, so the curl's three ``kms``
    volumes and the constitutive's are the SAME THREE and are bound once. Unshared the
    signature is 34 and does not compile; bind the half-integer set instead and nothing
    fails, which is why it is also a gate mutation.
    """
    assert metal_launch.SUB_STEPS["step_D"]["suffix"] == ""
    assert CONSTITUTIVE_SIDES["H"]["half_integer"] is False
    assert folded.UNSHARED_KMS_BINDINGS > MAX_BUFFER_BINDINGS


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------

class _Residency:
    """The minimum a predicate needs to see: something exposing ``mirror``."""

    def mirror(self, name, host, constant=False):  # pragma: no cover - never called
        raise AssertionError("the predicate must not mirror anything")


def _folded_grid(axes="Y", phases=None, boundaries=None, **overrides):
    phases = (1,) * len(axes) if phases is None else tuple(phases)
    keywords = dict(resolution=10.0, cell_size=(1.6, 2.0, 1.2), boundaries=boundaries,
                    dimensions=3, courant=0.35, k_point=(0.0, 0.0, 0.0), xp=np,
                    symmetry=tuple(Mirror(name, int(phase))
                                   for name, phase in zip(axes, phases)))
    keywords.update(overrides)
    return Grid(**keywords)


def _pml(grid):
    thickness = []
    for index in range(3):
        if grid.shape[index] < 6:
            thickness.append((0, 0))
        elif grid.is_mirrored(index):
            thickness.append((0, 2))
        else:
            thickness.append((2, 2))
    return PML(grid=grid, thickness=tuple(thickness))


def _fields(grid, complex_storage=False):
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.enable_pml_storage()
    shape = grid.shape
    epsilon = np.full(shape, np.float32(2.0), dtype=np.float32)
    inverse = np.full(shape, np.float32(0.5), dtype=np.float32)
    fields.set_epsilon_volumes({c: epsilon for c in ("Ex", "Ey", "Ez")},
                               {c: inverse for c in ("Ex", "Ey", "Ez")}, {})
    return fields


def _covered_case(axes="Y", phases=None, boundaries=None):
    grid = _folded_grid(axes=axes, phases=phases, boundaries=boundaries)
    return _fields(grid), _pml(grid)


@pytest.fixture()
def flush_policy(monkeypatch):
    """ONE CANONICAL POLICY, and it is a precondition rather than a default."""
    monkeypatch.setenv("MEEP_GPU_SUBNORMAL_POLICY", "flush")


@needs_mps
@pytest.mark.parametrize("axes,boundaries", [
    ("Y", None),
    ("Y", {"x": "metallic", "y": "metallic"}),
    ("XY", {"x": "metallic", "y": "metallic"}),
    ("XY", None),
    ("YZ", {"x": "metallic", "y": "metallic", "z": "metallic"}),
    ("XYZ", {"x": "metallic", "y": "metallic", "z": "metallic"}),
])
def test_the_predicate_admits_the_cell_this_product_was_built_for(
        axes, boundaries, flush_policy):
    """Every termination and every fold count the corpus cell carries."""
    fields, pml = _covered_case(axes=axes, boundaries=boundaries)
    verdict = folded.metal_folded_fused_hd_pair_coverage(fields, pml, (), _Residency())
    assert verdict.covered, verdict.reasons


@needs_mps
def test_an_unfolded_grid_is_refused_naming_the_plain_product(flush_policy):
    """HALF OF THE INVERTED CLAUSE. The other half is the next test."""
    grid = Grid(resolution=10.0, cell_size=(1.2, 1.0, 0.9), boundaries="periodic",
                dimensions=3, courant=0.35, k_point=(0.0, 0.0, 0.0), xp=np)
    verdict = folded.metal_folded_fused_hd_pair_coverage(
        _fields(grid), PML(grid=grid, thickness=tuple((2, 2) for _ in range(3))),
        (), _Residency())
    assert not verdict.covered
    assert any("fused_hd_pair" in reason for reason in verdict.reasons), verdict.reasons


@needs_mps
def test_the_plain_product_refuses_this_products_cell(flush_policy):
    """THE OTHER HALF, and it is why disjointness cannot be a single module's claim.

    Both predicates admitting one row would make the numerical method depend on
    composer order. The clause is inverted, so exactly one can ever answer.
    """
    fields, pml = _covered_case()
    verdict = plain.metal_fused_hd_pair_coverage(fields, pml, (), _Residency())
    assert not verdict.covered
    assert any("folded" in reason for reason in verdict.reasons), verdict.reasons


def test_an_undeclared_source_list_is_refused_by_name():
    """Ignorance is never an empty set."""
    fields, pml = _covered_case()
    verdict = folded.metal_folded_fused_hd_pair_coverage(fields, pml, None,
                                                         _Residency())
    assert not verdict.covered
    assert any("was not declared" in reason for reason in verdict.reasons)


def test_a_plan_built_with_no_residency_is_refused_by_both_halves():
    """Two sub-steps mirroring one volume separately would each hold a private copy."""
    fields, pml = _covered_case()
    verdict = folded.metal_folded_fused_hd_pair_coverage(fields, pml, (), None)
    assert not verdict.covered
    named = [reason for reason in verdict.reasons if "residency" in reason]
    assert len(named) == 2, (
        "both halves must say it; one alone would mean the other stopped asking")
    assert named[0].startswith("folded constitutive half:")
    assert named[1].startswith("folded curl half:")


def test_the_constitutive_half_is_asked_first_so_the_null_update_H_names_itself(
        flush_policy):
    """THE ORDER IS THE DRIVER'S (driver.py:3311 before :3315).

    An inactive absorber makes ``update_H`` return before its first statement -- the
    single largest non-fusion reason on the board -- and that belongs to the
    constitutive side, so it must be the FIRST reason a reader sees.
    """
    grid = _folded_grid()
    fields = _fields(grid)
    verdict = folded.metal_folded_fused_hd_pair_coverage(fields, None, (),
                                                         _Residency())
    assert not verdict.covered
    assert verdict.reasons[0].startswith("folded constitutive half:"), verdict.reasons
    assert "no active PML" in verdict.reasons[0]


@needs_mps
def test_complex_storage_under_a_fold_is_refused_naming_the_other_family(flush_policy):
    """The folded-complex family's, and the parity is why: under complex64 the fill's
    sign flip becomes a full complex multiply with zero cross terms."""
    grid = _folded_grid()
    verdict = folded.metal_folded_fused_hd_pair_coverage(
        _fields(grid, complex_storage=True), _pml(grid), (), _Residency())
    assert not verdict.covered
    assert any("complex" in reason for reason in verdict.reasons), verdict.reasons


def test_a_keep_policy_run_is_refused_by_name_rather_than_run(monkeypatch):
    """MPS flushes denormals natively and exposes no lever."""
    monkeypatch.setenv("MEEP_GPU_SUBNORMAL_POLICY", "keep")
    fields, pml = _covered_case()
    verdict = folded.metal_folded_fused_hd_pair_coverage(fields, pml, (), _Residency())
    assert not verdict.covered
    assert any("keep" in reason for reason in verdict.reasons), verdict.reasons


@needs_mps
def test_a_standing_electric_withdraw_is_refused_by_name_while_the_flag_is_false(
        flush_policy):
    """3 of the cell's 78 rows, and the refusal names the flag that would lift it."""
    from meep_gpu.sources import GaussianEnvelope, VolumeSource  # noqa: PLC0415

    fields, pml = _covered_case()
    source = VolumeSource(grid=fields.grid, component="Ez",
                          center=(0.15, 0.1, 0.05), size=(0.0, 0.0, 0.0),
                          envelope=GaussianEnvelope(frequency=1.0, fwidth=0.5,
                                                    is_integrated=True))
    verdict = folded.metal_folded_fused_hd_pair_coverage(fields, pml, (source,),
                                                          _Residency())
    assert not verdict.covered
    assert any("HOISTS_THE_WITHDRAW = False" in reason
               for reason in verdict.reasons), verdict.reasons


@needs_mps
@pytest.mark.parametrize("component", ["Hy", "Ez"])
def test_a_source_without_a_standing_withdraw_is_not_this_seams_business(
        component, flush_policy):
    """A magnetic source is injected one seam earlier and a non-integrated electric
    one has the no-op withdraw; neither is refused."""
    from meep_gpu.sources import GaussianEnvelope, VolumeSource  # noqa: PLC0415

    fields, pml = _covered_case()
    integrated = component == "Hy"
    source = VolumeSource(grid=fields.grid, component=component,
                          center=(0.15, 0.1, 0.05), size=(0.0, 0.0, 0.0),
                          envelope=GaussianEnvelope(frequency=1.0, fwidth=0.5,
                                                    is_integrated=integrated))
    verdict = folded.metal_folded_fused_hd_pair_coverage(fields, pml, (source,),
                                                          _Residency())
    assert verdict.covered, verdict.reasons


def test_the_withdraw_clause_is_the_shared_modules_and_not_a_second_copy():
    """The decision lives in :mod:`meep_gpu.withdraw_hoist`; only the prose is local."""
    text = Path(folded.__file__).read_text(encoding="utf-8")
    assert "_withdraw_hoist.seam_withdraw_reasons" in text
    assert "hoists_the_withdraw=HOISTS_THE_WITHDRAW" in text
    assert "span=REPLACES" in text


# ---------------------------------------------------------------------------
# Registration and composition
# ---------------------------------------------------------------------------

def test_the_arm_is_registered_unwired_gated_on_the_fold_and_is_a_weld():
    """Enumerable, not selectable, and SILENT on an unfolded run.

    ``arms.arms_for`` skips an unwired row so ``plan_step`` cannot select it, while
    ``arms.registered`` still returns it -- which is what lets
    ``launch._neighbouring_seam_claimant`` see the row. The gate is the folded family's
    convention: on an unfolded run four other arms will speak on ``update_H`` and this
    one should not add a refusal about a fold nobody asked for.
    """
    rows = [spec for spec in arms.registered("update_H")
            if spec.family == folded.FAMILY]
    assert len(rows) == 1, rows
    row = rows[0]
    assert row.wired is False
    assert row.replaces == folded.REPLACES
    assert row.gate is symmetry._has_fold  # noqa: SLF001
    assert row.label == "fused H/D pair (folded)"


def test_the_gate_keeps_the_row_silent_on_an_unfolded_run():
    """The cheap read of the grid, driven in both directions."""
    class _Context:
        def __init__(self, grid):
            self.fields = type("F", (), {"grid": grid})()

    unfolded = Grid(resolution=10.0, cell_size=(1.2, 1.0, 0.9), boundaries="periodic",
                    dimensions=3, courant=0.35, xp=np)
    assert symmetry._has_fold(_Context(_folded_grid())) is True  # noqa: SLF001
    assert symmetry._has_fold(_Context(unfolded)) is False  # noqa: SLF001


def test_the_uninstallable_declaration_is_read_off_this_module():
    """``launch._declared_uninstallable`` reads the flag and reports the reason."""
    text = Path(metal_launch.__file__).read_text(encoding="utf-8")
    assert "_declared_uninstallable" in text
    assert "INSTALLABLE" in text
    assert isinstance(folded.INSTALLABLE_REASON, str)
    assert len(folded.INSTALLABLE_REASON) > 400, (
        "the declaration must carry the measurement, not a sentence")


# ---------------------------------------------------------------------------
# The scratch, the rotation and the plan
# ---------------------------------------------------------------------------

def test_the_launch_binds_every_scratch_first_then_every_pre_launch_buffer():
    """The signature order IS the argument order, and it is not negotiable."""
    source = folded.folded_fused_hd_pair_source((P, MP, P))
    head = source.split("kernel void fused_hd_pair_step(", 1)[1]
    head = head.split("uint idx [[thread_position_in_grid]])", 1)[0]
    names = [line.split("[[buffer(")[0].split()[-1] for line in head.splitlines()
             if "[[buffer(" in line]
    assert names[:6] == ["ho0", "ho1", "ho2", "wo0", "wo1", "wo2"]
    assert names[6:12] == ["hi0", "hi1", "hi2", "wi0", "wi1", "wi2"]
    assert names[-1] == "prm"
    assert len(names) == folded.PACKED_BINDINGS


def test_the_rotation_moves_the_engines_references_only_after_the_launch():
    """Pure host logic, driven against a fake residency on a machine with no device."""
    class _Tensor:
        def __init__(self, host):
            self.host = host

    class _FakeResidency:
        device = "cpu"

        def __init__(self):
            self.by_host = {}

        def mirror(self, name, host, constant=False, dtype=None):
            tensor = _Tensor(host)
            self.by_host[id(host)] = tensor
            return tensor

        def tensor_for_host(self, host):
            return self.by_host.get(id(host))

    class _Fields:
        pass

    fields = _Fields()
    arrays = {}
    for name in folded.ROTATED_NAMES:
        arrays[name] = np.zeros((2, 2, 2), dtype=np.float32)
        setattr(fields, name, arrays[name])
    residency = _FakeResidency()
    twins = {}
    for name in folded.ROTATED_NAMES:
        residency.mirror(name, getattr(fields, name))
        twin, _ = weld.scratch_twin(residency, name, getattr(fields, name))
        twins[name] = twin
    calls = []
    plan = weld.plan_scratch_weld(
        folded.FAMILY, residency, fields, rotated_names=folded.ROTATED_NAMES,
        static_args=(), functions={shaders.CONTRACT_OFF:
                                   lambda *args: calls.append(args)},
        volumes=(), shape=(2, 2, 2), codes=(P, MP, P), zero_metal=(0, 0, 0),
        row_mask=(), replaces=folded.REPLACES, twins=twins)
    before = {name: getattr(fields, name) for name in folded.ROTATED_NAMES}
    plan.run()
    assert len(calls) == 1
    assert plan.launches == 1
    for name in folded.ROTATED_NAMES:
        assert getattr(fields, name) is twins[name], (
            "the engine's reference must move to the freshly written buffer")
        assert plan.rotated[name] is before[name]


def test_an_aliased_scratch_pair_is_refused_rather_than_launched():
    """The whole design is that nothing written is read."""
    class _Tensor:
        pass

    shared = _Tensor()

    class _FakeResidency:
        device = "cpu"

        def mirror(self, name, host, constant=False, dtype=None):
            return shared

        def tensor_for_host(self, host):
            return shared

    class _Fields:
        pass

    fields = _Fields()
    for name in folded.ROTATED_NAMES:
        setattr(fields, name, np.zeros((2, 2, 2), dtype=np.float32))
    residency = _FakeResidency()
    twins = {name: np.zeros((2, 2, 2), dtype=np.float32)
             for name in folded.ROTATED_NAMES}
    plan = weld.plan_scratch_weld(
        folded.FAMILY, residency, fields, rotated_names=folded.ROTATED_NAMES,
        static_args=(), functions={shaders.CONTRACT_OFF: lambda *args: None},
        volumes=(), shape=(2, 2, 2), codes=(P, MP, P), zero_metal=(0, 0, 0),
        row_mask=(), replaces=folded.REPLACES, twins=twins)
    with pytest.raises(RuntimeError, match="ONE tensor"):
        plan.run()


# ---------------------------------------------------------------------------
# The cell — asserted against the seam record where it is present
# ---------------------------------------------------------------------------

def _seam_record():
    path = (Path(__file__).resolve().parents[1] / "parity" / "meep_gpu" / "results"
            / "h_to_d_seam_2026-09-04" / "h_to_d_seam.jsonl")
    if not path.is_file():
        return None
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def test_the_cell_is_seventy_eight_rows_of_which_three_carry_a_standing_withdraw():
    """READ FROM THE RECORD, never restated. The numbers in the module docstring and
    in ``INSTALLABLE_REASON`` are what a reader will act on, so they answer to the
    artifact rather than to each other."""
    rows = _seam_record()
    if rows is None:
        pytest.skip("the H->D seam record is not in this tree")
    cell = [row for row in rows
            if row["arms"]["metal"]["update_H"] == "folded"
            and row["arms"]["metal"]["step_D"] == "folded"]
    assert len(cell) == CELL_ROWS, len(cell)
    withdraw = [row["label"] for row in cell
                if row["h_to_d_seam"]["n_electric_withdraws_that_do_work"]]
    assert len(withdraw) == CELL_WITHDRAW_ROWS, withdraw
    assert len(cell) - len(withdraw) == CELL_REACH
    buckets = {row["arms"]["metal"]["would_be_bucket"] for row in cell}
    assert buckets == {"buildable_not_built", "withdraw_seam"}, buckets
    text = Path(folded.__file__).read_text(encoding="utf-8")
    assert f"**{CELL_ROWS} rows**" in text or f"is **{CELL_ROWS} rows**" in text
    assert f"**{CELL_REACH} of\nthe cell's {CELL_ROWS}**" in text or \
        f"{CELL_REACH} of" in text
    for label in withdraw:
        row_name = label.split(":", 1)[1]
        assert row_name in text, (
            f"{label} carries a standing in-seam withdraw and this predicate refuses "
            f"it; the module must name it")


def test_this_cell_is_larger_than_the_plain_products_and_the_module_says_so():
    """The reason this product exists: 78 rows against 49, the largest cell here."""
    rows = _seam_record()
    if rows is None:
        pytest.skip("the H->D seam record is not in this tree")
    from collections import Counter  # noqa: PLC0415

    counts = Counter((row["arms"]["metal"]["update_H"],
                      row["arms"]["metal"]["step_D"]) for row in rows)
    assert counts[("folded", "folded")] == CELL_ROWS
    assert counts[("folded", "folded")] == max(counts.values())
    assert counts[("folded", "folded")] > counts[("ordinary", "PML")]
