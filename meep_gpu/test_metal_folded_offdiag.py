"""Merge-bar tests for the Metal FOLDED off-diagonal ``update_E`` family.

This family is a COMPOSITION of two things that are each already certified on this
backend, which is what makes it cheap and what makes it dangerous: the two halves
are byte-identical alone, and whether they remain so together is the question. The
tests are organised around that.

**1. THE STRUCTURAL REDUCTION, CHECKED BY CHARACTER.** The claim the family rests on
is that it emits the CERTIFIED off-diagonal source everywhere the fold does not
reach, and its own body only where it does. On the Triton track the analogous claim
needed a PTX read to settle; ``torch.mps.compile_shader`` exposes no disassembly at
all, so string equality is the only place it can be held here — and it is a
STRONGER instrument than PTX for this particular claim, not a weaker one. Three
directions are pinned: an unfolded code triple must produce a string EQUAL to
``offdiag_update_e.offdiag_source``'s; an ODD mirror plane (whose ghost weight is
``-phase == +1``) must produce the certified COMPONENT blocks; and an EVEN plane
must differ in exactly the two named blocks and nowhere else.

**2. THE SIGN, DERIVED FROM ``fields.mirror_parity`` ITSELF** rather than from a
second reading of the Yee table, and the SPELLING required to be the measured one.
``-x`` for the negated lane and a plain copy for the un-negated one; ``0.0f - x``
and any runtime weight are mutations, not comments.

**3. THE FOLD'S OWN HAZARDS**, which the certified off-diagonal suite cannot see:

* THE MIRROR PLANE IS WRITTEN BY ONE PASS AND READ BY THE NEXT, so a fold bug can be
  byte-perfect per sub-step and wrong per step. The whole-step leg composes the
  folded curls, both mirror fills and THIS ``update_E`` through the driver's five
  passes per half and reports the FIRST DIVERGENT STEP;
* THE TWO TERMINATIONS ARE DIFFERENT GRIDS even though they compile the same body
  here, and that equality is a NULL CONTROL rather than an assumption;
* REACHABILITY IS SHARP AT SLOT LEVEL, NOT ROW LEVEL. A row can be live through its
  other slot and never touch the fold, and the emitter reduces to the certified body
  in exactly that case.

**4. THE DISJOINTNESS**, stated as three inversions because this family has three
neighbours: the unfolded off-diagonal arm, the folded diagonal arm, and every plain
arm.

Device-touching tests are skipped without MPS. EVERY behavioural leg asserts a
VACUITY FLOOR — words actually moved — because zero-init is a fixed point of the
constitutive sub-step and a no-op agreeing with a no-op is trivially identical.
"""

from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import pytest

_PARITY = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                       "parity", "meep_gpu")
_PARITY = os.path.abspath(_PARITY)
if _PARITY not in sys.path:
    sys.path.insert(0, _PARITY)

import metal_composition_matrix as matrix  # noqa: E402

from meep_gpu import fields as fields_module  # noqa: E402
from meep_gpu import stepping  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid, Mirror  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.metal_kernels import arms, device, preconditions  # noqa: E402
from meep_gpu.metal_kernels import shaders, subnormal  # noqa: E402
from meep_gpu.metal_kernels import folded_offdiag_update_e as folded  # noqa: E402
from meep_gpu.metal_kernels import offdiag_update_e as offdiag  # noqa: E402
from meep_gpu.metal_kernels import symmetry  # noqa: E402

ENVIRONMENT = matrix.prepare_environment()

P = folded.CODE_PERIODIC
M = folded.CODE_METALLIC
MM = folded.CODE_MIRROR_METALLIC
MP = folded.CODE_MIRROR_PERIODIC

#: The two terminations of a fold on Y over a periodic x — the shape the corpus's
#: folded off-diagonal rows carry, and its unmeasured twin.
PERIODIC_FOLD = (P, MP, P)
METALLIC_FOLD = (P, MM, P)

#: The volumes this sub-step touches, and the six it writes.
NAMES = ("Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez",
         "Dx", "Dy", "Dz", "Hx", "Hy", "Hz")
COMPARED = ("Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez")

#: Every volume a complete step touches, for the whole-step leg.
SNAPSHOT = ("Bx", "By", "Bz", "Hx", "Hy", "Hz", "Dx", "Dy", "Dz",
            "Ex", "Ey", "Ez", "fu_Bx", "fu_By", "fu_Bz",
            "fu_Dx", "fu_Dy", "fu_Dz", "f_w_Hx", "f_w_Hy", "f_w_Hz",
            "f_w_Ex", "f_w_Ey", "f_w_Ez")

DEFAULT_ROWS = {"Ex": ("Ey", "Ez"), "Ez": ("Ex",)}


def _mps_available() -> bool:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False
    return bool(getattr(getattr(torch.backends, "mps", None), "is_available",
                        lambda: False)())


requires_mps = pytest.mark.skipif(not _mps_available(),
                                  reason="no MPS device on this host")


def _words(a):
    return np.ascontiguousarray(a, dtype=np.float32).reshape(-1).view(np.uint32)


def _differing(a, b) -> int:
    """uint32 WORD equality, never allclose: this is a bit-identity family."""
    return int(np.count_nonzero(_words(a) != _words(b)))


def build(axes="Y", phases=(1,), boundaries=None, dims=2, extent=2.0,
          rows=None, seed=4):
    """A folded grid with off-diagonal rows installed through the PUBLIC installer.

    Deterministic in ``seed``, so the whole-step leg can build the reference a
    SECOND TIME rather than deep-copying — ``Grid`` holds the array module itself
    and a deepcopy raises.
    """
    size = [1.6, 1.6, 0.0] if dims == 2 else [1.3, 1.2, 1.1]
    for name in axes:
        size["XYZ".index(name)] = extent
    grid = Grid(resolution=10.0, cell_size=tuple(size), dimensions=dims,
                courant=0.35,
                symmetry=tuple(Mirror(name, int(p))
                               for name, p in zip(axes, phases)),
                boundaries=boundaries, xp=np)
    fields = Fields(grid=grid, force_complex_fields=False)
    shape = grid.shape
    rng = np.random.default_rng(seed)
    epsilon = (1.45 + 0.3 * rng.random(shape)).astype(np.float32)
    inverse = (np.float32(1.0) / epsilon).astype(np.float32)
    # SPATIALLY VARYING rows, deliberately: a uniform coefficient makes the
    # certified family's m1 hoist mutation invisible, and the same blindness would
    # apply to anything that moved the multiply out from between the two shifts.
    built = {row: {p: (0.03 * rng.standard_normal(shape)).astype(np.float32)
                   for p in partners}
             for row, partners in (rows or DEFAULT_ROWS).items()}
    fields.set_epsilon_volumes({c: epsilon for c in ("Ex", "Ey", "Ez")},
                               {c: inverse for c in ("Ex", "Ey", "Ez")}, built)
    fields.enable_pml_storage()
    folded_indices = {"XYZ".index(n) for n in axes}
    thickness = []
    for index in range(3):
        if grid.shape[index] < 6:
            thickness.append((0, 0))
        elif index in folded_indices:
            thickness.append((0, 2))
        else:
            thickness.append((2, 2))
    pml = PML(grid=grid, thickness=tuple(thickness))
    for name in NAMES:
        getattr(fields, name)[...] = (rng.standard_normal(shape) * 0.37
                                      ).astype(np.float32)
    return fields, pml


def _oracle(fields, pml):
    """``stepping.update_E`` on the same state, with the vacuity floor measured."""
    before = {n: np.array(getattr(fields, n), copy=True) for n in NAMES}
    stepping.update_E(fields, pml)
    after = {n: np.array(getattr(fields, n), copy=True) for n in COMPARED}
    moved = sum(_differing(before[n], after[n]) for n in COMPARED)
    for name, value in before.items():
        getattr(fields, name)[...] = value
    return after, moved


def _shipped_specialisation(fields, pml):
    codes, _ = symmetry.folded_axis_kinds(fields.grid, pml)
    weights = folded.mirror_ghost_weights(fields.grid)
    walls = offdiag.wall_mask_axes(fields.grid)
    negate = folded.negated_axes(codes, weights)
    rows = offdiag.row_volumes_for(fields)
    row_mask = tuple(int(value is not None) for value in rows)
    return row_mask, codes, walls, negate


# ---------------------------------------------------------------------------
# 1. The structural reduction — checked by character, in three directions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("codes", tuple(itertools.product((P, M), repeat=3)))
def test_an_unfolded_triple_emits_the_certified_source_character_for_character(codes):
    """THE WHOLE REDUCTION CLAIM, over every unfolded code triple and every row mask.

    With no mirror code the folded emitter must return
    ``offdiag_update_e.offdiag_source``'s string EXACTLY — not a similar one. This
    is what makes "the fold adds one ghost line and one negated lane, and nothing
    else" a property of the code rather than of a comment, and it is what the Triton
    twin could only establish with a PTX-verified-different binary.
    """
    walls = tuple(int(code == M) for code in codes)
    for mask in range(1, 1 << len(folded.ROW_SLOTS)):
        row_mask = tuple((mask >> bit) & 1 for bit in range(len(folded.ROW_SLOTS)))
        assert (folded.folded_offdiag_source(row_mask, codes, walls, (0, 0, 0))
                == offdiag.offdiag_source(row_mask, codes, walls)), (codes, row_mask)


@pytest.mark.parametrize("mirror", (MM, MP))
def test_an_odd_mirror_plane_emits_the_certified_component_blocks(mirror):
    """THE POLARITY INVERTS THE READING A READER ARRIVES WITH, so it is pinned.

    The ghost weight is ``-phase``. An ODD plane (``phase = -1``) therefore weighs
    ``+1``, the ghost lane needs no sign at all, and every component block is the
    CERTIFIED emitter's text; only the ghost block differs, because the DOWN index
    is redirected rather than masked. The EVEN plane is the expensive one.
    """
    codes = (P, mirror, P)
    row_mask = (1, 1, 1, 1, 1, 1)
    odd = folded.folded_offdiag_source(row_mask, codes, (0, 0, 0), (0, 0, 0))
    certified = offdiag.offdiag_source(row_mask, (P, M, P), (0, 0, 0))
    for component in range(3):
        block = offdiag._component_source(component, row_mask, (0, 0, 0))
        assert block in odd, component
        assert block in certified, component
    # ...and the ONE line that differs is the ghost redirect.
    assert "    dj = (j == 0) ? 2 : dj;" in odd
    assert "    dvy = (dj >= 0);" in certified
    assert (set(certified.splitlines()) ^ set(odd.splitlines())
            == {"    dvy = (dj >= 0);", "    dj = (j == 0) ? 2 : dj;"})


def test_an_even_plane_negates_only_the_terms_whose_partner_axis_is_folded():
    """REACHABILITY IS SHARP AT SLOT LEVEL, and the emitter is what makes it so.

    A fold on axis ``a`` is byte-visible iff some LIVE row slot takes ``E_a`` as its
    PARTNER. With every slot live and a fold on Y, exactly the two terms whose
    partner is Y (``Ex <- Ey`` and ``Ez <- Ey``) carry the negation; ``Ey``'s own two
    slots take Z and X as partners and are the certified text.
    """
    row_mask = (1, 1, 1, 1, 1, 1)
    even = folded.folded_offdiag_source(row_mask, PERIODIC_FOLD, (0, 0, 0),
                                        (0, 1, 0))
    # Component 1 is Ey: partners Z then X, neither folded -> certified verbatim.
    assert offdiag._component_source(1, row_mask, (0, 0, 0)) in even
    # Components 0 and 2 take Ey as a partner in exactly one slot each.
    assert even.count("at_y ? -dn_") == 2
    assert even.count("at_y ? -cn_") == 2
    assert "at_x ? -" not in even and "at_z ? -" not in even


def test_a_component_whose_live_slot_misses_the_fold_is_the_certified_block():
    """THE MEASURED COUNTEREXAMPLE TO A ROW-LEVEL READING.

    Fold X with the single live slot ``Ey <- Ez``: the live row is NOT ``Ex``, so a
    row-level test answers "reachable" — but the fold is in neither role for that
    slot and the emitted block must be the certified one. The Triton track measured
    the same configuration byte-identical between MIRROR and METALLIC codes.
    """
    row_mask = tuple(int((row, partner) == ("Ey", "Ez"))
                     for row, partner in folded.ROW_SLOTS)
    assert sum(row_mask) == 1
    source = folded.folded_offdiag_source(row_mask, (MP, P, P), (0, 0, 0),
                                          (1, 0, 0))
    for component in range(3):
        assert offdiag._component_source(component, row_mask, (0, 0, 0)) in source
    assert "-dn_" not in source and "-cn_" not in source


def test_the_folded_kernel_binds_exactly_what_the_certified_kernel_binds():
    """THE FOLD ADDS NO ARGUMENT, and that is a consequence of two facts rather than
    a coincidence: the own-axis up shift is not parity-weighted (so no reflect row)
    and the parity is a SOURCE specialisation on this backend (so no ghost weight).
    The Triton twin passes three runtime ``gw*`` scalars for the second reason alone.

    Also the 31-binding ceiling, asserted rather than left for a future edit to
    discover at compile time.
    """
    source = folded.folded_offdiag_source((1, 0, 0, 0, 0, 0), PERIODIC_FOLD,
                                          (0, 0, 0), (0, 1, 0))
    certified = offdiag.offdiag_source((1, 0, 0, 0, 0, 0), (P, M, P), (0, 0, 0))
    assert source.count("[[buffer(") == certified.count("[[buffer(")
    assert source.count("[[buffer(") == folded.BINDING_COUNT
    assert folded.BINDING_COUNT <= device.MAX_BUFFER_BINDINGS


def test_the_template_is_the_certified_module_attribute_not_a_copy():
    """A copied template is a template that drifts. This family substitutes into
    ``offdiag_update_e._TEMPLATE`` itself, so the kernel name, the index decode, the
    ``at_*`` predicates and the three PML tails cannot diverge from the certified
    kernel's by an edit here."""
    assert "_TEMPLATE" not in folded.__dict__
    source = folded.folded_offdiag_source((1, 0, 0, 0, 0, 0), PERIODIC_FOLD,
                                          (0, 0, 0), (0, 1, 0))
    assert "kernel void offdiag_constitutive_step(" in source


@pytest.mark.parametrize("code", (P, M))
def test_an_unfolded_axis_ghost_block_is_the_certified_emitters_output(code):
    """The certified ghost emitter is CALLED, not reimplemented, so an edit to the
    certified ghost rule reaches this family without a second edit."""
    for axis in "xyz":
        assert (folded._folded_two_way_ghost(axis, code)
                == offdiag._two_way_ghost(axis, code))


@pytest.mark.parametrize("mirror", (MM, MP))
def test_both_mirror_terminations_emit_the_same_ghost_block(mirror):
    """A PREDICTED NULL WITH A DERIVATION, carried as a control rather than collapsed.

    ``update_E`` has no ownership mask and, because ``_offdiagonal_terms`` calls
    ``_shift_up`` without ``component``/``reflect_row``, no reflect row either — so
    the MIRROR_METALLIC / MIRROR_PERIODIC split cannot reach this kernel. The codes
    are still carried, because ``folded_axis_kinds`` cross-checks the split against
    ``grid.is_metallic`` and refuses a disagreement.
    """
    for axis in "xyz":
        assert (folded._folded_two_way_ghost(axis, mirror)
                == folded._folded_two_way_ghost(axis, MM))
    row_mask = (1, 1, 1, 1, 1, 1)
    assert (folded.folded_offdiag_source(row_mask, (P, mirror, P), (0, 0, 0),
                                         (0, 1, 0))
            == folded.folded_offdiag_source(row_mask, METALLIC_FOLD, (0, 0, 0),
                                            (0, 1, 0)))


def test_the_mirror_ghost_block_masks_the_far_face_and_redirects_the_near_one():
    """The two documented deltas, spelled out, because getting either backwards is a
    plane of wrong values rather than a crash: DOWN is redirected to stored row
    ``MIRROR_SOURCE_INDEX`` (a live interior plane, not a mask) and UP is the
    METALLIC arm's mask verbatim."""
    block = folded._folded_two_way_ghost("y", MP)
    metallic = offdiag._two_way_ghost("y", M)
    assert f"? {folded.MIRROR_SOURCE_INDEX} :" in block
    assert "dvy" not in block, "a mirror ghost is a redirect, never a mask"
    assert metallic.splitlines()[1] in block


# ---------------------------------------------------------------------------
# 2. The parity, derived from the engine rather than restated
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("axis", (0, 1, 2))
@pytest.mark.parametrize("phase", (1, -1))
def test_the_ghost_weight_collapses_to_minus_phase(axis, phase):
    """``mirror_parity(c, axis, phase) == phase * (1 - 2*iyee[c][axis])``, and every
    D component has Yee shift 1 on its OWN axis, so the weight the partner-axis down
    shift carries is ``-phase`` for EVERY partner and EVERY axis. Derived through
    ``fields.mirror_parity`` itself here so the collapse cannot drift from it."""
    component = "D" + "xyz"[axis]
    assert fields_module.mirror_parity(component, axis, phase) == -phase
    assert fields_module.IYEE_SHIFTS[component][axis] == 1


@pytest.mark.parametrize("phase,expected", ((1, -1.0), (-1, 1.0)))
def test_mirror_ghost_weights_reads_the_grids_own_plane(phase, expected):
    fields, _pml = build(phases=(phase,))
    weights = folded.mirror_ghost_weights(fields.grid)
    assert weights[1] == expected
    assert weights[0] == weights[2] == 1.0


def test_an_unreadable_plane_gives_a_nan_weight_and_is_refused_by_name():
    """The weight decides the SOURCE on this backend, so an unreadable plane must not
    be allowed to select a specialisation by accident."""

    class NoPhase:
        def __init__(self, wrapped):
            object.__setattr__(self, "_wrapped", wrapped)

        def __getattr__(self, name):
            if name == "mirror_phase":
                return lambda axis: None
            return getattr(object.__getattribute__(self, "_wrapped"), name)

    fields, pml = build()
    weights = folded.mirror_ghost_weights(NoPhase(fields.grid))
    assert weights[1] != weights[1], weights


@pytest.mark.parametrize("codes,weights,expected", (
    ((P, MP, P), (1.0, -1.0, 1.0), (0, 1, 0)),
    ((P, MP, P), (1.0, 1.0, 1.0), (0, 0, 0)),
    ((MM, MP, P), (-1.0, -1.0, 1.0), (1, 1, 0)),
    # An unfolded axis is NEVER negated whatever the weight says: the ghost lane
    # does not exist there and the value is never read.
    ((P, P, P), (-1.0, -1.0, -1.0), (0, 0, 0)),
))
def test_negated_axes_is_one_derivation_of_code_and_weight(codes, weights,
                                                           expected):
    assert folded.negated_axes(codes, weights) == expected


def test_the_emitter_refuses_a_sign_on_an_unfolded_axis():
    with pytest.raises(ValueError, match="marked negated"):
        folded.folded_offdiag_source((1, 0, 0, 0, 0, 0), (P, P, P), (0, 0, 0),
                                     (0, 1, 0))


def test_the_emitter_refuses_a_wall_on_a_folded_axis():
    """``_mask_metallic_wall_coupling`` ABSTAINS on a mirrored axis (stepping.py:1282)
    and zeroing the fold plane the metallic way costs 2.0e-02. A folded METALLIC axis
    is ``is_metallic`` True AND ``is_mirrored`` True, so this is the configuration
    where conflating the wall declaration with the boundary code would bite."""
    with pytest.raises(ValueError, match="folded AND wall-masked"):
        folded.folded_offdiag_source((1, 0, 0, 0, 0, 0), METALLIC_FOLD, (0, 1, 0),
                                     (0, 0, 0))


def test_an_all_dead_row_mask_is_refused_toward_the_folded_diagonal_family():
    with pytest.raises(ValueError, match="no row slot survives"):
        folded.folded_offdiag_source((0,) * 6, PERIODIC_FOLD, (0, 0, 0), (0, 1, 0))


def test_the_wall_declaration_and_the_boundary_code_come_apart_on_a_fold():
    """WITHIN THE CERTIFIED FAMILY'S ADMITTED SPACE the two are LOCKED
    (``walls[a] == (codes[a] == METALLIC)``, measured there); a fold is exactly the
    configuration that breaks the lock, which is why the plan carries them
    separately rather than deriving one from the other."""
    fields, pml = build(boundaries={"y": "metallic"})
    codes, _ = symmetry.folded_axis_kinds(fields.grid, pml)
    walls = offdiag.wall_mask_axes(fields.grid)
    assert codes[1] == MM
    assert fields.grid.is_metallic(1) and fields.grid.is_mirrored(1)
    assert walls[1] == 0, "the mask abstains on a fold plane"


# ---------------------------------------------------------------------------
# 3. Constants and reachability
# ---------------------------------------------------------------------------

def test_every_shared_constant_is_the_parents_object_not_a_copy():
    """One home per fact. An import that silently became a copy is caught here."""
    assert folded.E_TERMS is offdiag.E_TERMS
    assert folded.ROW_SLOTS is offdiag.ROW_SLOTS
    assert folded.TRANSVERSE_PARTNERS is offdiag.TRANSVERSE_PARTNERS
    assert folded.WALL_MASK_AXES is offdiag.WALL_MASK_AXES
    assert folded.BINDING_COUNT == offdiag.BINDING_COUNT
    assert folded.HALF_INTEGER == offdiag.HALF_INTEGER
    assert folded.MIRROR_CODES == symmetry.MIRROR_CODES
    assert folded.MIRROR_SOURCE_INDEX == symmetry.MIRROR_SOURCE_INDEX
    assert folded.MIRROR_SOURCE_INDEX == stepping.MIRROR_SOURCE_INDEX


@pytest.mark.parametrize("axes,rows,reachable,dims", (
    ("X", {"Ey": ("Ex",)}, True, 2),
    ("X", {"Ey": ("Ez",)}, False, 2),
    ("X", {"Ex": ("Ey", "Ez")}, False, 2),
    ("Y", {"Ex": ("Ey",)}, True, 2),
    ("Y", {"Ez": ("Ex",)}, False, 2),
    ("Z", {"Ex": ("Ez",)}, True, 3),
))
def test_reachability_is_asked_at_slot_level(axes, rows, reachable, dims):
    """A fold on axis ``a`` is byte-visible iff some LIVE SLOT takes ``E_a`` as its
    PARTNER. The row-level reading ("some live row is not ``E_a``") answers
    differently on rows 2 and 3 and is wrong there."""
    fields, _pml = build(axes=axes, rows=rows, dims=dims,
                         extent=1.4 if dims == 3 else 2.0)
    got, notes = folded.mirror_arm_is_reachable(fields.grid, fields)
    assert got is reachable, notes
    assert notes


def test_the_specialisation_enumeration_carries_a_mirror_on_every_entry():
    """An unfolded triple emits the CERTIFIED family's source, so hashing it here
    would fingerprint that family's surface a second time under this name."""
    entries = folded.specialisations()
    assert entries
    assert all(any(code in folded.MIRROR_CODES for code in codes)
               for _rows, codes, _walls, _negate in entries)
    assert all(negate[axis] == 0 or codes[axis] in folded.MIRROR_CODES
               for _rows, codes, _walls, negate in entries for axis in range(3))
    assert all(walls[axis] == 0 or codes[axis] == M
               for _rows, codes, walls, _negate in entries for axis in range(3))
    assert len(entries) == 63 * (7 ** 3 - 3 ** 3)


# ---------------------------------------------------------------------------
# 4. The predicate — three inversions, and every refusal by name
# ---------------------------------------------------------------------------

def _verdict(fields, pml, residency=None):
    return folded.folded_offdiag_constitutive_coverage(
        fields, pml, device.Residency() if residency is None else residency)


@requires_mps
def test_the_routing_verdict_requires_a_fold_and_the_wide_one_does_not(monkeypatch):
    """THE FIRST INVERSION, against ``offdiag_update_e``, and the reason the two
    verdicts are separate functions.

    The WIDE verdict admits an unfolded grid so a gate can prove the reduction to the
    certified kernel; registering it would make every unfolded off-diagonal row
    ambiguous and would leave the numerical method depending on composer order.
    """
    monkeypatch.setenv(subnormal.POLICY_ENV, subnormal.FLUSH)
    fields, pml = matrix.cart(rows={"Ex": ("Ey",)})
    wide = folded.folded_offdiag_constitutive_coverage(fields, pml,
                                                       device.Residency())
    narrow = folded.folded_offdiag_composition_coverage(fields, pml,
                                                        device.Residency())
    assert wide.covered, wide.reasons
    assert not narrow.covered
    assert any("no mirror plane is active" in reason for reason in narrow.reasons)
    # ...and the certified arm admits exactly there, which is what makes it an
    # inversion rather than an omission.
    assert offdiag.offdiag_constitutive_coverage(fields, pml,
                                                 device.Residency()).covered


@requires_mps
def test_the_certified_offdiag_predicate_refuses_the_fold_by_name(monkeypatch):
    """The other side of the first inversion, asserted rather than assumed."""
    monkeypatch.setenv(subnormal.POLICY_ENV, subnormal.FLUSH)
    fields, pml = build()
    verdict = offdiag.offdiag_constitutive_coverage(fields, pml, device.Residency())
    assert not verdict.covered
    assert any("mirror plane" in reason for reason in verdict.reasons), verdict.reasons


@requires_mps
def test_the_folded_diagonal_predicate_refuses_a_live_row_by_name(monkeypatch):
    """THE SECOND INVERSION, against ``symmetry.folded_constitutive_coverage``.

    That predicate gates the ELEMENT-WISE constitutive body and refuses an
    off-diagonal row because the row product reads neighbours; this one requires a
    live row SLOT. The pair is disjoint through the installer's zero-row drop rather
    than through an inverted clause, which is why the composition matrix carries the
    planted disagreement — but on every REACHABLE configuration exactly one admits,
    and that is what is measured here.
    """
    monkeypatch.setenv(subnormal.POLICY_ENV, subnormal.FLUSH)
    fields, pml = build()
    diagonal = symmetry.folded_constitutive_coverage(fields, pml, "E",
                                                     device.Residency())
    assert not diagonal.covered
    assert any("off-diagonal chi1inv row" in reason for reason in diagonal.reasons)
    assert _verdict(fields, pml).covered, _verdict(fields, pml).reasons

    # ...and with no row installed the two swap.
    plain, plain_pml = matrix.folded()
    assert symmetry.folded_constitutive_coverage(plain, plain_pml, "E",
                                                 device.Residency()).covered
    bare = _verdict(plain, plain_pml)
    assert not bare.covered
    assert any("no off-diagonal chi1inv row survived" in reason
               for reason in bare.reasons), bare.reasons


@requires_mps
def test_the_curl_slots_are_the_folded_familys_and_this_one_claims_only_update_E():
    """No clause separates this family from the folded CURL arms; the SLOT does, and
    it is stated so a reader does not go looking for one."""
    assert folded.SLOT == "update_E"
    assert {spec.slot for spec in arms.registered()
            if spec.family == folded.FAMILY} == {"update_E"}


@requires_mps
@pytest.mark.parametrize("mutate,needle", (
    (lambda: matrix.folded(complex_storage=True, rows={"Ex": ("Ey",)}),
     "force_complex_fields"),
    (lambda: matrix.dispersive(matrix.folded(rows={"Ex": ("Ey",)})),
     "susceptibility is registered"),
    (lambda: matrix.nonlinear(matrix.folded(rows={"Ex": ("Ey",)})), "chi2/chi3"),
    (lambda: matrix.cylindrical(m=1), "cylindrical"),
))
def test_every_family_this_one_does_not_carry_is_refused_by_name(monkeypatch,
                                                                mutate, needle):
    monkeypatch.setenv(subnormal.POLICY_ENV, subnormal.FLUSH)
    fields, pml = mutate()
    verdict = _verdict(fields, pml)
    assert not verdict.covered
    assert any(needle in reason for reason in verdict.reasons), verdict.reasons


@requires_mps
def test_cylindrical_is_refused_twice(monkeypatch):
    """A THREE-WAY HAZARD, not a two-way one. ``_boundary_kinds`` puts ``is_axis``
    AHEAD of ``is_mirrored``, so a folded r axis would report CYL_AXIS and the fold
    would vanish silently; and ``_mirror_phases`` puts ``(-1)**grid.m`` into the SAME
    SLOT the mirror phase occupies. The grid flag alone is not the inversion."""
    monkeypatch.setenv(subnormal.POLICY_ENV, subnormal.FLUSH)
    fields, pml = matrix.cylindrical(m=1)
    verdict = _verdict(fields, pml)
    assert sum("cylindrical" in reason or "r = 0 axis" in reason
               for reason in verdict.reasons) >= 2, verdict.reasons


@requires_mps
def test_an_inactive_absorber_is_refused_toward_the_null_family(monkeypatch):
    """A SECOND, INDEPENDENT SEPARATION on this slot: ``no_pml_constitutive``
    requires an INACTIVE absorber and this family an ACTIVE one."""
    monkeypatch.setenv(subnormal.POLICY_ENV, subnormal.FLUSH)
    fields, pml = matrix.folded_no_pml()
    verdict = _verdict(fields, pml)
    assert not verdict.covered
    assert any("no active PML layer" in reason for reason in verdict.reasons)


@requires_mps
def test_the_keep_policy_refuses_the_plan_at_build_time(monkeypatch):
    """``keep`` is a REFUSAL on this executor, not a silent downgrade: Metal flushes
    float32 denormals natively and exposes no lever, and this arm64 host's DEFAULT
    resolves to keep — which is exactly why every other leg pins ``flush``."""
    monkeypatch.setenv(subnormal.POLICY_ENV, subnormal.KEEP)
    fields, pml = build()
    verdict = _verdict(fields, pml)
    assert not verdict.covered
    assert any("subnormal" in reason for reason in verdict.reasons), verdict.reasons
    assert folded.plan_folded_offdiag_constitutive(
        fields, pml, device.Residency()) is None


@requires_mps
def test_a_covered_verdict_never_meets_a_raising_builder(monkeypatch):
    """The builder's None-means-refused contract: wherever the predicate answers
    covered the plan must be built, never raised out of."""
    monkeypatch.setenv(subnormal.POLICY_ENV, subnormal.FLUSH)
    for kwargs in (dict(), dict(boundaries={"y": "metallic"}), dict(phases=(-1,)),
                   dict(axes="XY", phases=(1, -1)), dict(extent=2.1)):
        fields, pml = build(**kwargs)
        residency = device.Residency()
        verdict = _verdict(fields, pml, residency)
        assert verdict.covered, (kwargs, verdict.reasons)
        assert folded.plan_folded_offdiag_constitutive(
            fields, pml, residency) is not None, kwargs


# ---------------------------------------------------------------------------
# 5. BYTE IDENTITY against stepping.update_E
# ---------------------------------------------------------------------------

BYTE_CASES = (
    ("2d_fold_Y_even_periodic", dict(axes="Y", phases=(1,))),
    ("2d_fold_Y_odd_periodic", dict(axes="Y", phases=(-1,))),
    ("2d_fold_Y_even_metallic", dict(axes="Y", phases=(1,),
                                     boundaries={"y": "metallic"})),
    ("2d_fold_Y_odd_metallic", dict(axes="Y", phases=(-1,),
                                    boundaries={"y": "metallic"})),
    ("2d_fold_X_even", dict(axes="X", phases=(1,))),
    ("2d_fold_XY_mixed", dict(axes="XY", phases=(1, -1))),
    ("2d_fold_XY_both_even", dict(axes="XY", phases=(1, 1))),
    ("2d_fold_Y_odd_full_count", dict(axes="Y", phases=(1,), extent=2.1)),
    ("2d_fold_Y_with_metallic_wall", dict(axes="Y", phases=(1,),
                                          boundaries={"x": "metallic",
                                                      "y": "metallic"})),
    ("3d_fold_Z_even", dict(axes="Z", phases=(1,), dims=3, extent=1.4)),
    ("3d_fold_XYZ_mixed", dict(axes="XYZ", phases=(1, -1, 1), dims=3, extent=1.4)),
    ("2d_fold_X_slot_misses_the_fold", dict(axes="X", phases=(1,),
                                            rows={"Ey": ("Ez",)})),
    ("2d_fold_X_slot_crosses_the_fold", dict(axes="X", phases=(1,),
                                             rows={"Ey": ("Ex",)})),
    ("2d_fold_Y_all_six_slots", dict(axes="Y", phases=(1,),
                                     rows={"Ex": ("Ey", "Ez"),
                                           "Ey": ("Ez", "Ex"),
                                           "Ez": ("Ex", "Ey")})),
)


@requires_mps
@pytest.mark.parametrize("label,kwargs", BYTE_CASES, ids=[c[0] for c in BYTE_CASES])
def test_the_sub_step_reproduces_stepping_update_E_byte_for_byte(monkeypatch,
                                                                 label, kwargs):
    """THE FAMILY'S CENTRAL CLAIM, over both terminations, both parities, one and
    three folded axes, both full-count parities, 2-D and 3-D, and a row set whose
    live slot misses the fold entirely.

    Measured on this host 2026-08-16: 0 differing words on every case, with the
    oracle moving 864-8424 words per case (the VACUITY FLOOR, asserted below —
    zero-init is a fixed point of this sub-step, so a no-op agreeing with a no-op
    would otherwise pass).
    """
    monkeypatch.setenv(subnormal.POLICY_ENV, subnormal.FLUSH)
    fields, pml = build(**kwargs)
    after, moved = _oracle(fields, pml)
    assert moved > 0, "the oracle did nothing; identity here would be vacuous"

    residency = device.Residency()
    plan = folded.plan_folded_offdiag_constitutive(fields, pml, residency)
    assert plan is not None, _verdict(fields, pml, residency).reasons
    plan.run()
    residency.sync_out()

    assert plan.launches == 1, "the plan must have RUN for identity to mean anything"
    for name in COMPARED:
        assert _differing(getattr(fields, name), after[name]) == 0, (label, name)
    assert not residency.verify()


@requires_mps
def test_asking_for_an_unbuilt_contract_variant_raises(monkeypatch):
    """The guard selector must not fall back to the pinned source: a leg that
    measured the contraction guard through a silent fallback would certify nothing."""
    monkeypatch.setenv(subnormal.POLICY_ENV, subnormal.FLUSH)
    fields, pml = build()
    plan = folded.plan_folded_offdiag_constitutive(fields, pml, device.Residency())
    assert plan is not None
    assert plan.variants == (shaders.CONTRACT_OFF,)
    with pytest.raises(KeyError, match="holds no 'fast' variant"):
        plan.run(contract=shaders.CONTRACT_FAST)


# ---------------------------------------------------------------------------
# 6. THE WHOLE STEP — the real arbiter
# ---------------------------------------------------------------------------

WHOLE_STEP_CASES = (
    ("2d_fold_Y_even_periodic", dict(axes="Y", phases=(1,))),
    ("2d_fold_Y_odd_periodic", dict(axes="Y", phases=(-1,))),
    ("2d_fold_Y_even_metallic", dict(axes="Y", phases=(1,),
                                     boundaries={"y": "metallic"})),
    ("2d_fold_XY_mixed", dict(axes="XY", phases=(1, -1))),
    ("2d_fold_Y_odd_full_count", dict(axes="Y", phases=(1,), extent=2.1)),
    ("3d_fold_Z_even", dict(axes="Z", phases=(1,), dims=3, extent=1.4)),
)


@requires_mps
@pytest.mark.parametrize("label,kwargs", WHOLE_STEP_CASES,
                         ids=[c[0] for c in WHOLE_STEP_CASES])
def test_a_complete_step_composes_with_the_folded_curls_and_fills(monkeypatch,
                                                                  label, kwargs):
    """SIX GREEN SUB-STEP COMPARISONS SAY NOTHING ABOUT THE OBJECT THE ENGINE RUNS.

    Three failure classes live only in a complete step — a stale mirror, a seam, and
    an accumulating auxiliary (``fu_*``, ``f_w_*`` are STATE, and a kernel right for
    one launch and wrong forever after is identical in a single-launch leg) — and the
    fold adds a fourth: the mirror plane is WRITTEN by one pass and READ by the next,
    so a fold bug can be byte-perfect per sub-step and wrong per step.

    Compared per COMPLETE STEP over a stated budget, reporting the FIRST DIVERGENT
    STEP; per-slot launch counters prove each slot actually ran, so no slot can pass
    by not executing. Measured on this host 2026-08-16: 0 differing words at every
    step of 6, on every case.
    """
    monkeypatch.setenv(subnormal.POLICY_ENV, subnormal.FLUSH)
    budget = 6
    fields, pml = build(**kwargs)
    reference_fields, reference_pml = build(**kwargs)
    for name in SNAPSHOT:
        if getattr(fields, name, None) is not None:
            assert _differing(getattr(fields, name),
                              getattr(reference_fields, name)) == 0, name

    residency = device.Residency()
    plans = {
        "step_B": symmetry.plan_folded_pml_curl(fields, pml, "step_B", residency),
        "fill_B": symmetry.plan_mirror_ghost_fill(fields, "B", "fill_B", residency),
        "update_H": symmetry.plan_folded_constitutive(fields, pml, "H", residency),
        "step_D": symmetry.plan_folded_pml_curl(fields, pml, "step_D", residency),
        "fill_D": symmetry.plan_mirror_ghost_fill(fields, "D", "fill_D", residency),
        "update_E": folded.plan_folded_offdiag_constitutive(fields, pml, residency),
    }
    assert all(plan is not None for plan in plans.values()), {
        name for name, plan in plans.items() if plan is None}
    residency.sync_in()

    first_divergent = None
    per_step = []
    for step in range(budget):
        # The driver's own order, five passes per half (driver.py:3282-3306) with no
        # source and no pole: the two injections and update_P are absent.
        stepping.step_B(reference_fields, reference_pml)
        stepping.fill_symmetry_bc_B(reference_fields)
        stepping.fill_folded_far_ghosts_B(reference_fields)
        stepping.update_H(reference_fields, reference_pml)
        stepping.step_D(reference_fields, reference_pml)
        stepping.fill_symmetry_bc_D(reference_fields)
        stepping.fill_folded_far_ghosts_D(reference_fields)
        stepping.update_E(reference_fields, reference_pml)

        plans["step_B"].run()
        plans["fill_B"].run_near()
        plans["fill_B"].run_far()
        plans["update_H"].run()
        plans["step_D"].run()
        plans["fill_D"].run_near()
        plans["fill_D"].run_far()
        plans["update_E"].run()
        residency.sync_out()

        per_name = {name: _differing(getattr(fields, name),
                                     getattr(reference_fields, name))
                    for name in SNAPSHOT
                    if getattr(fields, name, None) is not None}
        total = sum(per_name.values())
        per_step.append(total)
        if total and first_divergent is None:
            first_divergent = (step, {k: v for k, v in per_name.items() if v})

    assert first_divergent is None, (label, "FIRST DIVERGENT STEP", first_divergent,
                                     "per-step differing words", per_step)
    moved = sum(_differing(np.zeros_like(getattr(reference_fields, name)),
                           getattr(reference_fields, name))
                for name in ("Bx", "By", "Bz", "Ex", "Ey", "Ez"))
    assert moved > 100, (label, moved, "VACUOUS whole-step leg")
    assert plans["update_E"].launches == budget
    assert plans["step_B"].launches == budget
    assert not residency.verify()


# ---------------------------------------------------------------------------
# 6b. THE PRECONDITION IS A WINDOW, NOT A SCALAR
# ---------------------------------------------------------------------------

def _census_step(fields, weights, codes, step):
    """One step's census over the operands this sub-step reads, ITS RESULTS, and the
    fold's OWN intermediate — the ghost-weighted pair on the mirror plane.

    THE FOLD PUTS A DEEP-PML PLANE ON THE FAR FACE, which is exactly where tiny
    magnitudes live, and the ghost lane READS stored row ``MIRROR_SOURCE_INDEX`` and
    WRITES the fold plane. A whole-volume aggregate cannot be moved by a 1e-40
    boundary plane, so the intermediate is censused on THOSE PLANES specifically.
    """
    window = preconditions.SubnormalWindow(
        first_step=step, last_step=step,
        per_array_words=1, per_intermediate_words=1)
    for name in ("Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
                 "f_w_Ex", "f_w_Ey", "f_w_Ez"):
        window.observe(name, getattr(fields, name), step=step)
    for name in ("Ex", "Ey", "Ez"):
        window.observe("inv_eps_" + name, fields.inverse_epsilon_for(name),
                       step=step)
    for (row, partner), value in zip(folded.ROW_SLOTS,
                                     offdiag.row_volumes_for(fields)):
        if value is not None:
            window.observe(f"chi1inv_offdiag:{row}:{partner}", value, step=step)
    # The fold's own intermediate: the ghosted pair `g[face 0] + w * g[row 2]`, on
    # the mirror plane of every folded axis, for every partner volume.
    for axis, code in enumerate(codes):
        if int(code) not in folded.MIRROR_CODES:
            continue
        weight = np.float32(weights[axis])
        for name in ("Dx", "Dy", "Dz"):
            volume = getattr(fields, name)
            face = [slice(None)] * 3
            face[axis] = 0
            source = [slice(None)] * 3
            source[axis] = folded.MIRROR_SOURCE_INDEX
            pair = (volume[tuple(face)]
                    + weight * volume[tuple(source)]).astype(np.float32)
            window.observe_intermediate(f"ghost_pair[{axis}]{name}", pair,
                                        step=step)
    return window


@requires_mps
@pytest.mark.parametrize("scale,fires", ((1.0, False), (1e-30, False),
                                         (1e-38, True)),
                         ids=("physical", "small_normal", "scaled_control"))
def test_the_subnormal_precondition_reports_a_window_not_a_count(monkeypatch,
                                                                 scale, fires):
    """THE PRECONDITION IS A RUN-AND-WINDOW FACT, so the leg reports the WINDOW.

    Band entry is not a family property: on the chi3 round a Q~20 narrow-band
    Gaussian turn-on swept ~40 decades and dragged the leading edge through the whole
    band from step 55 to step 3,726, then ran clean for 16,274 steps. So "this leg
    censused zero" licenses nothing about a run with a narrow-band source, and the
    honest report is the FIRST and LAST step the census fired, not a count.

    THREE ROWS, and the last is what makes the first two non-decorative. Measured on
    this host 2026-08-16 over 17,568 censused words per row: the PHYSICAL run is
    clean (window EMPTY, 0 subnormal words); a SMALL-NORMAL run eight decades down is
    still clean (window EMPTY, 0) — the cliff, not a threshold; and a run scaled to
    1e-38 FIRES at every step with 9,495 words in the band and a window of ``[0, 5]``.
    A precondition never demonstrated to fire is decorative.
    """
    monkeypatch.setenv(subnormal.POLICY_ENV, subnormal.FLUSH)
    budget = 6
    fields, pml = build(axes="Y", phases=(1,))
    reference_fields, reference_pml = build(axes="Y", phases=(1,))
    if scale != 1.0:
        for name in NAMES:
            for target in (fields, reference_fields):
                getattr(target, name)[...] = (
                    getattr(target, name) * np.float32(scale)).astype(np.float32)

    codes, _ = symmetry.folded_axis_kinds(fields.grid, pml)
    weights = folded.mirror_ghost_weights(fields.grid)

    residency = device.Residency()
    plan = folded.plan_folded_offdiag_constitutive(fields, pml, residency)
    assert plan is not None
    residency.sync_in()

    fired = []
    reports = []
    for step in range(budget):
        stepping.step_B(reference_fields, reference_pml)
        stepping.fill_symmetry_bc_B(reference_fields)
        stepping.fill_folded_far_ghosts_B(reference_fields)
        stepping.update_H(reference_fields, reference_pml)
        stepping.step_D(reference_fields, reference_pml)
        stepping.fill_symmetry_bc_D(reference_fields)
        stepping.fill_folded_far_ghosts_D(reference_fields)
        stepping.update_E(reference_fields, reference_pml)
        window = _census_step(reference_fields, weights, codes, step)
        report = window.report()
        reports.append(report)
        assert not report["vacuous"], (step, report["vacuity_reasons"])
        if not report["clean"]:
            fired.append(step)

    observed = sum(report["observed_words"] for report in reports)
    assert observed > 1000, ("VACUOUS census", observed)
    window_bounds = [fired[0], fired[-1]] if fired else None
    assert bool(fired) is fires, (scale, "window", window_bounds,
                                  "steps censused", budget)
    if fires:
        # THE WINDOW, not a scalar: first and last step the census fired.
        assert window_bounds is not None and len(window_bounds) == 2
        assert window_bounds[0] <= window_bounds[1]
    else:
        # The physical row is clean over the WHOLE window, and the refusal helper is
        # what a gate would call — exercised here so it is not merely available.
        merged = preconditions.SubnormalWindow(
            first_step=0, last_step=budget - 1, per_array_words=1,
            per_intermediate_words=1)
        merged.observe("Dx", reference_fields.Dx, step=budget - 1)
        merged.observe_intermediate(
            "ghost_pair", (reference_fields.Dy[:, 0, :]
                           + np.float32(weights[1])
                           * reference_fields.Dy[:, folded.MIRROR_SOURCE_INDEX, :]
                           ).astype(np.float32), step=budget - 1)
        preconditions.assert_clean_or_refuse(
            merged, "folded off-diagonal update_E, physical band")


# ---------------------------------------------------------------------------
# 7. MUTATIONS — a gate that cannot fail certifies nothing
# ---------------------------------------------------------------------------

def _run_mutant(fields, pml, source):
    from meep_gpu.metal_kernels.device import compile_source

    residency = device.Residency()
    plan = folded.plan_folded_offdiag_constitutive(fields, pml, residency)
    assert plan is not None
    plan._functions = {shaders.CONTRACT_OFF:
                       compile_source(source).offdiag_constitutive_step}
    residency.sync_in()
    plan.run()
    residency.sync_out()
    assert plan.launches == 1
    return plan


#: ``(label, mutate(source) -> source, build kwargs, must_be_caught)``. Every entry
#: is a defect a reader could plausibly write, and the NULLS are as load-bearing as
#: the catches: a leg that reported only catches could not tell a discriminating
#: measurement from a coincidence.
MUTATIONS = (
    # The mirror ghost is a REDIRECT to a live interior plane, not the metallic mask.
    ("mirror_ghost_becomes_metallic",
     lambda s: s.replace("    dj = (j == 0) ? 2 : dj;", "    dvy = (dj >= 0);"),
     dict(axes="Y", phases=(1,)), True),
    ("mirror_ghost_becomes_periodic",
     lambda s: s.replace("    dj = (j == 0) ? 2 : dj;",
                         "    dj = (dj < 0) ? (nyi - 1) : dj;"),
     dict(axes="Y", phases=(1,)), True),
    # MEEP's halved origin io = -2 maps the ghost at -1 onto stored cell 2.
    ("ghost_images_stored_row_one",
     lambda s: s.replace("(j == 0) ? 2 : dj", "(j == 0) ? 1 : dj"),
     dict(axes="Y", phases=(1,)), True),
    # The sign must apply on the GHOST LANE, not to the whole volume.
    ("negation_applied_to_the_whole_volume",
     lambda s: s.replace("(at_y ? -dn_", "(true ? -dn_").replace(
         "(at_y ? -cn_", "(true ? -cn_"),
     dict(axes="Y", phases=(1,)), True),
    # The wall mask must ABSTAIN on a fold plane (stepping.py:1282).
    ("fold_plane_masked_the_metallic_way",
     lambda s: s.replace("    float src0 = (gs0 * us0) + total0;",
                         "    total0 = at_y ? 0.0f : total0;\n"
                         "    float src0 = (gs0 * us0) + total0;"),
     dict(axes="Y", phases=(1,)), True),
    # The own-axis UP shift serves an exact 0 on a mirror axis; a wrap is wrong.
    # CAUGHT only where the folded axis is the OWN axis of a LIVE row, which is
    # why the row set is named and why the vacuous twin below is carried.
    ("own_axis_up_wraps_with_a_live_Ey_row",
     lambda s: s.replace("    uvy = (uj < nyi);", "    uj = (uj == nyi) ? 0 : uj;"),
     dict(axes="Y", phases=(1,), rows={"Ey": ("Ez", "Ex")}), True),
    ("own_axis_up_wraps_with_the_Ey_row_dead",
     lambda s: s.replace("    uvy = (uj < nyi);", "    uj = (uj == nyi) ? 0 : uj;"),
     dict(axes="Y", phases=(1,)), False),
    # The refuted spelling. NOT caught on ordinary normal data — the term's own sum
    # renormalises — which is exactly why the value-class leg below exists.
    ("refuted_spelling_zero_minus_x_on_normals",
     lambda s: s.replace("-dn_", "0.0f - dn_").replace("-cn_", "0.0f - cn_"),
     dict(axes="Y", phases=(1,)), False),
)


@requires_mps
@pytest.mark.parametrize("label,mutate,kwargs,caught", MUTATIONS,
                         ids=[m[0] for m in MUTATIONS])
def test_the_byte_leg_catches_the_defects_it_claims_to(monkeypatch, label, mutate,
                                                       kwargs, caught):
    """Measured on this host 2026-08-16, differing words per mutant: metallic ghost
    32, periodic ghost 32, wrong ghost row 32, whole-volume negation 352, fold plane
    wall-masked 32, own-axis up wrap 32 (with a live ``Ey`` row) and 0 (without one),
    ``0.0f - x`` 0 on normals.

    THE TWO NULLS ARE NAMED RATHER THAN OMITTED. The up-wrap pair is the same mutant
    on two row sets and is what makes the catch a discriminating measurement instead
    of a coincidence; the ``0.0f - x`` null is what the value-class leg then
    resolves.
    """
    monkeypatch.setenv(subnormal.POLICY_ENV, subnormal.FLUSH)
    fields, pml = build(**kwargs)
    after, moved = _oracle(fields, pml)
    assert moved > 0, "VACUOUS: the oracle did nothing"
    row_mask, codes, walls, negate = _shipped_specialisation(fields, pml)
    source = folded.folded_offdiag_source(row_mask, codes, walls, negate)
    mutant = mutate(source)
    assert mutant != source, (label, "the mutation did not apply; the leg is DISARMED")

    _run_mutant(fields, pml, mutant)
    differing = sum(_differing(getattr(fields, name), after[name])
                    for name in COMPARED)
    assert bool(differing) is caught, (label, differing)


@requires_mps
@pytest.mark.parametrize("label,plant,caught", (
    # `-x` and `0.0f - x` part company on +0.0, but `g[i] + ghost` washes the sign
    # back out unless g[i] is itself -0.0. A zeroed ghost plane ALONE is a null.
    ("plus_zero_on_the_ghost_plane_only", lambda a: a.__setitem__(
        (slice(None), 2, slice(None)), np.float32(0.0)), False),
    # The engineered lattice: row 0 negative zero, rows 1-2 as shown, so the sign
    # survives the pair sum, the coefficient multiply and the row sum into f_w.
    ("signed_zero_lattice_rows_0_1_2", None, True),
    # A subnormal ghost plane. `0.0f - x` flushes every subnormal on this backend.
    ("subnormal_ghost_plane", lambda a: a.__setitem__(
        (slice(None), 2, slice(None)), np.float32(1e-40)), True),
))
def test_the_refuted_spelling_is_refused_on_evidence_and_the_evidence_names_its_class(
        monkeypatch, label, plant, caught):
    """WHY ``0.0f - x`` IS A MUTATION AND NOT A COMMENT, and what its reach actually is.

    Measured on this host 2026-08-16 in this family's own term shape: on the GHOST
    VALUE alone ``0.0f - x`` misses 12 of 30 words (10 of 10 subnormals AND 2 of 4
    signed zeros) while ``-x`` and ``x * -1.0f`` are exact; added to a NORMAL partner
    every spelling agrees, because the addition renormalises.

    So the divergence needs the value to survive to the output, and the three rows
    here are that reach: a zeroed ghost plane alone is a NULL (the pair sum
    ``+0.0 + -0.0`` is ``+0.0`` either way), an engineered signed-zero lattice is
    CAUGHT in 5 words UNDER THE SUBNORMAL-FREE PRECONDITION, and a subnormal ghost
    plane is caught in 32. The middle row is why the spelling is refused rather than
    left to the precondition.
    """
    monkeypatch.setenv(subnormal.POLICY_ENV, subnormal.FLUSH)
    fields, pml = build(axes="Y", phases=(1,))
    for name in ("Dx", "Dy", "Dz"):
        array = getattr(fields, name)
        if plant is None:
            array[:, 0, :] = np.float32(-0.0)
            array[:, 1, :] = np.float32(-0.0)
            array[:, 2, :] = np.float32(0.0)
        else:
            plant(array)

    after, moved = _oracle(fields, pml)
    assert moved > 0, "VACUOUS: the oracle did nothing"
    row_mask, codes, walls, negate = _shipped_specialisation(fields, pml)
    source = folded.folded_offdiag_source(row_mask, codes, walls, negate)
    mutant = (source.replace("-dn_", "0.0f - dn_")
                    .replace("-cn_", "0.0f - cn_"))
    assert mutant != source
    _run_mutant(fields, pml, mutant)
    differing = sum(_differing(getattr(fields, name), after[name])
                    for name in COMPARED)
    assert bool(differing) is caught, (label, differing)


@requires_mps
@pytest.mark.parametrize("phases,flipped", (((1,), (0, 0, 0)), ((-1,), (0, 1, 0))))
def test_flipping_the_parity_is_caught_at_both_plane_phases(monkeypatch, phases,
                                                            flipped):
    """The sign is a SOURCE specialisation here, so the mutation is a different
    compiled body rather than a different passed word — and it must be caught at BOTH
    phases, because the even plane's defect is a missing negation and the odd plane's
    is a spurious one. Measured: 32 differing words each."""
    monkeypatch.setenv(subnormal.POLICY_ENV, subnormal.FLUSH)
    fields, pml = build(axes="Y", phases=phases)
    after, moved = _oracle(fields, pml)
    assert moved > 0
    row_mask, codes, walls, negate = _shipped_specialisation(fields, pml)
    assert negate != flipped, "the mutation must differ from the shipped build"
    mutant = folded.folded_offdiag_source(row_mask, codes, walls, flipped)
    _run_mutant(fields, pml, mutant)
    differing = sum(_differing(getattr(fields, name), after[name])
                    for name in COMPARED)
    assert differing > 0, (phases, "the parity flip was not caught")


# ---------------------------------------------------------------------------
# 8. Compilation and wiring
# ---------------------------------------------------------------------------

@requires_mps
def test_every_specialisation_the_corpus_drives_compiles(monkeypatch):
    """The corpus's folded off-diagonal rows carry TWO BC triples, both folded
    METALLIC and 2-D. Both parities and both terminations are compiled here, over
    every row mask, in both contraction modes — 63 x 4 x 2 = 504 shaders."""
    monkeypatch.setenv(subnormal.POLICY_ENV, subnormal.FLUSH)
    from meep_gpu.metal_kernels.device import compile_source

    triples = ((M, MM, P), (MM, MM, P), (P, MP, P), (MM, MM, MM))
    built = 0
    for mask in range(1, 1 << len(folded.ROW_SLOTS)):
        row_mask = tuple((mask >> bit) & 1 for bit in range(len(folded.ROW_SLOTS)))
        for codes in triples:
            walls = tuple(int(code == M) for code in codes)
            negate = tuple(int(code in folded.MIRROR_CODES) for code in codes)
            for mode in (shaders.CONTRACT_OFF, shaders.CONTRACT_FAST):
                compile_source(folded.folded_offdiag_source(
                    row_mask, codes, walls, negate, mode))
                built += 1
    assert built == 63 * len(triples) * 2


def test_the_contraction_mode_changes_the_corpus_digest():
    """The guard is a property of the SOURCE on this backend, so the two modes are
    two different compiled kernels and the fingerprint must say so."""
    off = folded.corpus_digest(shaders.CONTRACT_OFF)
    fast = folded.corpus_digest(shaders.CONTRACT_FAST)
    assert off["count"] == fast["count"] == len(folded.specialisations())
    assert off["sha256"] != fast["sha256"]


def test_the_family_is_in_the_registry_module_and_holds_exactly_one_arm():
    """A family in the tree but NOT in ``registry.FAMILY_MODULES`` is INVISIBLE to
    ``plan_step`` — a silent coverage loss rather than an error, because the composer
    would leave the slot on the array path, which is always correct and never
    complained about."""
    from meep_gpu.metal_kernels import registry

    assert "folded_offdiag_update_e" in registry.FAMILY_MODULES
    rows = [spec for spec in arms.registered() if spec.family == folded.FAMILY]
    assert len(rows) == 1
    row, = rows
    assert row.slot == "update_E"
    assert row.label == folded.LABEL
    assert row.wired
    assert row.gate is not None, "an ungated arm would speak on every unfolded run"


@requires_mps
def test_the_gate_asks_for_both_halves(monkeypatch):
    """THIS FAMILY IS AN INTERSECTION, so the gate asks BOTH questions: on an
    unfolded run with rows the certified arm speaks, and on a folded run without rows
    the folded constitutive arm does. In neither case does a reader want this
    family's full refusal list beside theirs."""
    monkeypatch.setenv(subnormal.POLICY_ENV, subnormal.FLUSH)

    class _Context:
        def __init__(self, fields):
            self.fields = fields

    fields, _pml = build()
    assert folded._has_fold_and_rows(_Context(fields))
    plain, _ = matrix.folded()
    assert not folded._has_fold_and_rows(_Context(plain))
    unfolded, _ = matrix.cart(rows={"Ex": ("Ey",)})
    assert not folded._has_fold_and_rows(_Context(unfolded))
    assert not folded._has_fold_and_rows(_Context(None))


@pytest.mark.parametrize("attribute", ("grid", "chi1inv_offdiagonal_for"))
def test_an_unreadable_state_gates_the_arm_out_rather_than_raising(attribute):
    """``plan_step`` evaluates a gate EAGERLY inside ``arms.arms_for``, which sits
    OUTSIDE the per-arm try/except — so a raising read here would escape a composer
    whose whole contract is that it never raises. Unreadable is not live: the arm
    goes silent and the slot falls to the array path, which is always correct."""

    class _Raising:
        def __init__(self, wrapped, name):
            object.__setattr__(self, "_wrapped", wrapped)
            object.__setattr__(self, "_name", name)

        def __getattr__(self, name):
            if name == object.__getattribute__(self, "_name"):
                raise RuntimeError(f"{name} is unreadable")
            return getattr(object.__getattribute__(self, "_wrapped"), name)

    class _Context:
        def __init__(self, fields):
            self.fields = fields

    fields, _pml = build()
    assert folded._has_fold_and_rows(_Context(fields))
    assert not folded._has_fold_and_rows(_Context(_Raising(fields, attribute)))
