"""Laptop tests for the Metal off-diagonal ``update_E`` family.

THE MERGE BAR, NOT THE CERTIFICATION. The byte claim is
``parity/meep_gpu/gate_metal_offdiag.py``'s nine legs and the composition probe's
three; these tests are what must stay green on every change, and they are chosen
for the defects a byte gate would catch LATE or not at all:

* the constant tables that BIND coefficients to partners and axes to masks. A
  mispaired slot is a half-cell registration error that stays smooth and plausible,
  and the tables are pinned against the ENGINE's own (``fields.IYEE_SHIFTS``,
  ``stepping.E_CONSTITUTIVE_TERMS``) rather than against a copy of themselves;
* the emitter's REFUSALS. A source that silently emitted an all-dead row mask would
  overlap the certified plain kernel; a substitution that silently left a
  placeholder would make a mutation leg report an uncaught defect;
* the BINDING COUNT against Metal's measured 31-buffer ceiling, so a future edit
  that adds a volume discovers the limit here rather than at compile time on a
  configuration nobody sweeps;
* the predicate's refusals, all of which are answerable WITHOUT a GPU;
* the registration's ``wired=False``, because a family that quietly became wired
  would change what ``plan_step`` launches.

Everything that needs a device is guarded and skipped, so this file is the merge
bar on a host with no MPS as well as on this one.
"""

from __future__ import annotations

import numpy as np
import pytest

from meep_gpu import stepping
from meep_gpu.fields import IYEE_SHIFTS, Fields
from meep_gpu.grid import Grid
from meep_gpu.metal_kernels import (
    arms,
    offdiag_update_e as offdiag,
    shaders,
    subnormal,
)
from meep_gpu.pml import PML


def _torch_mps() -> bool:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False
    return bool(getattr(getattr(torch, "backends", None), "mps", None)
                and torch.backends.mps.is_available())


needs_mps = pytest.mark.skipif(not _torch_mps(),
                               reason="no MPS device on this host")


# ---------------------------------------------------------------------------
# The constant tables, pinned against the engine's own
# ---------------------------------------------------------------------------

def test_e_terms_match_steppings_constitutive_order():
    """``E_TERMS`` is ``stepping.E_CONSTITUTIVE_TERMS`` with the axis as an index.

    The ORDER decides which coefficient vector each component's PML tail reads, and
    a permutation here is a silent half-cell error in the absorber profile.
    """
    expected = tuple((component, source, "xyz".index(axis))
                     for component, source, axis in stepping.E_CONSTITUTIVE_TERMS)
    assert offdiag.E_TERMS == expected


def test_wall_mask_axes_match_the_engines_yee_shifts():
    """The mask zeroes face 0 of every axis whose Yee shift is 0, ascending.

    Derived from ``fields.IYEE_SHIFTS`` rather than compared to a second copy of the
    table: ``_mask_metallic_wall_coupling`` (stepping.py:1279-1283) loops the axes
    and skips those with a nonzero shift, so this is the same question its ``if``
    asks.
    """
    for index, (component, _source, _axis) in enumerate(offdiag.E_TERMS):
        shifts = IYEE_SHIFTS[component]
        expected = tuple(axis for axis in range(3) if shifts[axis] == 0)
        assert offdiag.WALL_MASK_AXES[index] == expected, component


def test_transverse_partners_follow_meeps_cycle_direction():
    """Own axis + 1 then own axis + 2, X -> Y -> Z (stepping.py:1235-1237).

    The order BINDS COEFFICIENTS TO PARTNERS, so a swap makes every row read the
    wrong volume — which is the gate's m3 mutation and is invisible to any
    magnitude check.
    """
    for index, (_component, _source, own_axis) in enumerate(offdiag.E_TERMS):
        assert offdiag.TRANSVERSE_PARTNERS[index] == ((own_axis + 1) % 3,
                                                      (own_axis + 2) % 3)


def test_row_slots_are_the_partner_order_component_by_component():
    """The six slots are E_TERMS order x TRANSVERSE_PARTNERS order, and nothing else.

    The plan fills the kernel's six coefficient buffers by this table, so a slot in
    the wrong place mispairs a coefficient with a partner volume.
    """
    expected = []
    for index, (component, _source, _axis) in enumerate(offdiag.E_TERMS):
        for partner_axis in offdiag.TRANSVERSE_PARTNERS[index]:
            expected.append((component, offdiag.E_TERMS[partner_axis][0]))
    assert offdiag.ROW_SLOTS == tuple(expected)


def test_half_integer_is_the_e_side_sub_lattice():
    """``update_E`` reads kps_a_h/kms_a_h (stepping.py:1015). A swap is a half-cell error."""
    assert offdiag.HALF_INTEGER is True


# ---------------------------------------------------------------------------
# The emitter
# ---------------------------------------------------------------------------

FULL = (1, 1, 1, 1, 1, 1)


def test_the_binding_count_stays_under_metals_ceiling():
    """28 bindings, three below the measured 31-buffer limit.

    Metal's buffer attribute indices must be 0..30 and a 32nd is a COMPILE ERROR.
    The count is asserted here so an edit that adds a volume discovers the ceiling
    at the merge bar rather than at compile time on a specialisation nobody sweeps.
    """
    source = offdiag.offdiag_source(FULL, (0, 0, 0), (0, 0, 0))
    assert source.count("[[buffer(") == offdiag.BINDING_COUNT
    assert offdiag.BINDING_COUNT <= 31
    for index in range(offdiag.BINDING_COUNT):
        assert f"[[buffer({index})]]" in source, index


def test_an_all_dead_row_mask_is_refused_by_name():
    """That configuration is the certified plain kernel's; a second kernel for it
    would make the two families overlap on exactly the runs the install-time
    zero-row drop separates."""
    with pytest.raises(ValueError, match="no row slot survives"):
        offdiag.offdiag_source((0, 0, 0, 0, 0, 0), (0, 0, 0), (0, 0, 0))


def test_a_boundary_code_that_is_neither_periodic_nor_metallic_is_refused():
    with pytest.raises(ValueError, match="neither PERIODIC nor METALLIC"):
        offdiag.offdiag_source(FULL, (0, 0, 7), (0, 0, 0))


def test_the_emitted_source_carries_the_contraction_directive_exactly_once():
    """The one compile option the whole claim rests on, and only from shaders'
    single spelling."""
    for mode in shaders.CONTRACT_MODES:
        source = offdiag.offdiag_source(FULL, (1, 1, 0), (1, 1, 0), mode)
        assert source.count(shaders.contraction_pragma(mode)) == 1


def test_a_component_with_no_row_emits_the_plain_diagonal_product():
    """Docstring point 6: the 'none' arm is the certified kernel's body verbatim.

    Textual here and byte-measured in the gate's seam leg; the pair is what makes
    the disjointness claim checkable at two levels rather than asserted at one.
    """
    # Only slot 0 (Ex/Ey) lives, so components 1 and 2 take the none arm.
    source = offdiag.offdiag_source((1, 0, 0, 0, 0, 0), (0, 0, 0), (0, 0, 0))
    assert "    float src1 = gs1 * us1;" in source
    assert "    float src2 = gs2 * us2;" in source
    assert "total1" not in source
    assert "total2" not in source
    # And the live one is NOT the plain arm.
    assert "    float src0 = (gs0 * us0) + total0;" in source


def test_the_wall_mask_is_emitted_only_for_declared_axes_of_the_right_component():
    """Face 0, ascending axis order, only where the grid DECLARES metallic.

    A different question from the ghost codes: the mask asks
    ``is_metallic and not is_mirrored`` (stepping.py:1279-1283), which is why the
    plan carries ``wall_axes`` separately from ``boundary_codes``.
    """
    # Walls on x and y. Component 0 (Ex, shift (1,0,0)) masks y only; component 1
    # (Ey) masks x only; component 2 (Ez) masks x and y.
    source = offdiag.offdiag_source(FULL, (1, 1, 0), (1, 1, 0))
    assert "    total0 = at_y ? 0.0f : total0;" in source
    assert "    total0 = at_x ? 0.0f : total0;" not in source
    assert "    total1 = at_x ? 0.0f : total1;" in source
    assert "    total1 = at_y ? 0.0f : total1;" not in source
    assert "    total2 = at_x ? 0.0f : total2;" in source
    assert "    total2 = at_y ? 0.0f : total2;" in source
    # No declared wall: no branch at all, and a comment saying so rather than
    # silence.
    plain = offdiag.offdiag_source(FULL, (1, 1, 0), (0, 0, 0))
    assert "? 0.0f : total0;" not in plain
    assert "no declared metallic wall axis" in plain


def test_the_metallic_ghost_is_a_flag_and_the_periodic_one_is_a_wrap():
    """The rule is a TERNARY on a validity flag, never a clamped load.

    ``v ? p[o] : 0.0f`` delivers the exact ``+0.0`` past the wall that the array
    path's ``_shift_*`` writes into that plane, without dereferencing anything.
    """
    metallic = offdiag.offdiag_source(FULL, (1, 1, 1), (0, 0, 0))
    assert "    dvx = (di >= 0);" in metallic
    assert "    uvx = (ui < nxi);" in metallic
    periodic = offdiag.offdiag_source(FULL, (0, 0, 0), (0, 0, 0))
    assert "    di = (di < 0) ? (nxi - 1) : di;" in periodic
    assert "    ui = (ui == nxi) ? 0 : ui;" in periodic


#: The per-axis stride and home tokens, SPELLED A SECOND TIME on purpose.
#: ``_index`` assembles six flat-index triples from ``_STRIDE``/``_HOME``; a test
#: that called ``_index`` would agree with any table, including a wrong one. This
#: is the independent spelling the assembled indices are checked against.
_STRIDE_SPELLING = ("{} * nyz", "{} * nzi", "{}")
_HOME_SPELLING = ("i", "j", "k")
_DOWN_SPELLING = ("di", "dj", "dk")
_UP_SPELLING = ("ui", "uj", "uk")
_DVALID_SPELLING = ("dvx", "dvy", "dvz")
_UVALID_SPELLING = ("uvx", "uvy", "uvz")


def _flat_index(shifts):
    return " + ".join(_STRIDE_SPELLING[axis].format(shifts.get(axis,
                                                               _HOME_SPELLING[axis]))
                      for axis in range(3))


def test_all_six_row_slots_carry_the_indices_their_own_axes_imply():
    """Every slot's three flat indices, against an INDEPENDENT spelling.

    ``_index`` assembles the six (component, partner) triples from a stride table
    precisely because six hand-written index expressions is six chances to write
    ``dj`` where ``dk`` belongs — a half-cell registration error that stays smooth
    and plausible. Only slot 0 was pinned as text; the other five rested entirely
    on the byte gate. This walks all six and rebuilds each index from the axis
    tables rather than from ``_index``, so a wrong stride table fails here rather
    than agreeing with itself.

    The three that matter per slot: the partner pair reads DOWN the partner's axis,
    the far pair reads UP the component's OWN axis, and the corner composes the two.
    """
    for component, (_name, _source, own_axis) in enumerate(offdiag.E_TERMS):
        for offset in (0, 1):
            slot = 2 * component + offset
            mask = tuple(int(index == slot) for index in range(6))
            source = offdiag.offdiag_source(mask, (0, 0, 0), (0, 0, 0))
            partner_axis = offdiag.TRANSVERSE_PARTNERS[component][offset]
            tag = f"{component}{offset}"
            buffer = f"u{component}{offset + 1}"
            volume = f"g{partner_axis}"
            down = _DOWN_SPELLING[partner_axis]
            up = _UP_SPELLING[own_axis]
            dvalid = _DVALID_SPELLING[partner_axis]
            uvalid = _UVALID_SPELLING[own_axis]
            expected = [
                f"    float near_{tag} = {volume}[ii]",
                f"        + ({dvalid} ? {volume}"
                f"[{_flat_index({partner_axis: down})}] : 0.0f);",
                f"    float far_{tag} = ({uvalid} ? {volume}"
                f"[{_flat_index({own_axis: up})}] : 0.0f)",
                f"        + (({uvalid} && {dvalid}) ? {volume}"
                f"[{_flat_index({own_axis: up, partner_axis: down})}] : 0.0f);",
                f"    float unear_{tag} = {buffer}[ii];",
                f"    float ufar_{tag} = {uvalid} ? {buffer}"
                f"[{_flat_index({own_axis: up})}] : 0.0f;",
            ]
            for line in expected:
                assert line in source, (component, offset, line)


def test_the_wall_declaration_is_locked_to_the_boundary_codes_in_the_admitted_space():
    """``walls[a] == (codes[a] == METALLIC)`` on every grid this family admits.

    The plan carries ``wall_axes`` separately from ``boundary_codes`` because the
    mask asks ``is_metallic and not is_mirrored`` while the ghost rule asks
    ``_boundary_kinds``. Those are genuinely different questions — but this family
    REFUSES folds, and with mirrors gone the two answers coincide on every admitted
    grid. Pinned so the separation is understood as future-proofing rather than as
    a live degree of freedom, and so the round that admits folds has to face the
    divergence deliberately instead of discovering it as a wrong field.
    """
    for x in ("periodic", "metallic"):
        for y in ("periodic", "metallic"):
            for z in ("periodic", "metallic"):
                grid = _grid(boundaries=(x, y, z))
                pml = PML(grid=grid, thickness=tuple((2, 2) for _ in range(3)))
                kinds = stepping._boundary_kinds(grid, pml)
                codes = tuple(shaders.METALLIC if kind == "metallic"
                              else shaders.PERIODIC for kind in kinds)
                walls = offdiag.wall_mask_axes(grid)
                assert walls == tuple(int(code == shaders.METALLIC)
                                      for code in codes), (x, y, z, codes, walls)


def test_the_coefficient_multiply_sits_between_the_two_shifts():
    """THE one new arithmetic element of this family, pinned as source text.

    ``u`` is loaded at the component's OWN node for the near pair and at the node
    one cell UP its own axis for the far pair. A build that multiplied a four-point
    average by ``u[ii]`` would be the hoist — the gate's m1 — and would be
    algebraically equal on a uniform coefficient, which is exactly why the source
    text is pinned here as well as measured there.
    """
    source = offdiag.offdiag_source((1, 0, 0, 0, 0, 0), (0, 0, 0), (0, 0, 0))
    assert "    float unear_00 = u01[ii];" in source
    assert "    float ufar_00 = uvx ? u01[ui * nyz + j * nzi + k] : 0.0f;" in source
    assert ("    float term_00 = 0.25f * ((near_00 * unear_00)"
            " + (far_00 * ufar_00));") in source


def test_no_float_divides_and_no_min_or_max_builtin_is_called():
    """The measured Metal hazards, absent by construction and pinned as such.

    ``fast::divide`` produced 2471 mismatches on this host, and the ``min``/``max``
    builtins return the SECOND operand on two zeros and drop NaN, which disagrees
    with ``numpy.minimum``/``maximum``. This family needs none of them; the test is
    what stops a future edit from reaching for one.

    THE INDEX DIVISIONS ARE DELIBERATELY EXEMPT and the exemption is the point of
    checking float lines rather than every ``/``: ``ii / nzi`` is INTEGER and EXACT,
    so no float rounds on that path and two spellings of it cannot differ in bits.
    A float divide is the hazard; an int divide is the flat-index decode.
    """
    for codes in ((0, 0, 0), (1, 1, 1)):
        source = offdiag.offdiag_source(FULL, codes, codes)
        body = source.split("uint idx [[thread_position_in_grid]])", 1)[1]
        code = [line for line in body.splitlines()
                if not line.strip().startswith("//")]
        assert not any("fast::" in line for line in code)
        for line in code:
            if "float" in line or any(name in line for name in
                                      ("near_", "far_", "term_", "total", "src",
                                       "a0", "a1", "a2")):
                assert "/" not in line, f"a float divide appeared: {line!r}"
        for builtin in ("min(", "max(", "fmin(", "fmax("):
            assert not any(builtin in line for line in code), builtin


def test_the_specialisation_corpus_is_the_size_the_family_claims():
    """63 live row masks x 8 boundary triples x 8 wall triples."""
    assert len(offdiag.specialisations()) == 63 * 8 * 8
    digest = offdiag.corpus_digest()
    assert digest["count"] == 63 * 8 * 8
    assert len(digest["sha256"]) == 64
    # A DIGEST, not a certification: the two contraction modes must differ, or the
    # guard is not in the string it is supposed to be a property of.
    assert (offdiag.corpus_digest("off")["sha256"]
            != offdiag.corpus_digest("fast")["sha256"])


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------

def _grid(boundaries="periodic"):
    return Grid(resolution=10.0, cell_size=(1.2, 1.0, 0.9), boundaries=boundaries,
                dimensions=3, courant=0.35, k_point=(0.0, 0.0, 0.0), xp=np)


def _fields_with_rows(grid, rows=None):
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    shape = grid.shape
    epsilon = np.full(shape, np.float32(2.0), dtype=np.float32)
    inverse = np.full(shape, np.float32(0.5), dtype=np.float32)
    rng = np.random.default_rng(7)
    if rows is None:
        rows = {"Ex": {"Ey": (0.03 * rng.standard_normal(shape)).astype(np.float32)}}
    fields.set_epsilon_volumes({c: epsilon for c in ("Ex", "Ey", "Ez")},
                               {c: inverse for c in ("Ex", "Ey", "Ez")}, rows)
    return fields


class _Residency:
    """The minimum a predicate needs to see: something exposing ``mirror``."""

    def mirror(self, name, host, constant=False):  # pragma: no cover - never called
        raise AssertionError("the predicate must not mirror anything")


def test_a_plan_built_with_no_residency_is_refused_by_name():
    """Two sub-steps mirroring one volume separately each hold a private copy, and
    the second launch reads the first one's stale bytes."""
    grid = _grid()
    fields = _fields_with_rows(grid)
    pml = PML(grid=grid, thickness=tuple((2, 2) for _ in range(3)))
    verdict = offdiag.offdiag_constitutive_coverage(fields, pml, None)
    assert not verdict.covered
    assert any("residency" in reason for reason in verdict.reasons), verdict.reasons


def test_a_zero_row_run_is_refused_toward_the_plain_kernel():
    """The install-time drop (fields.py:1302-1303) makes the two families disjoint,
    and this predicate must not overlap ``constitutive_coverage(side='E')``."""
    grid = _grid()
    fields = _fields_with_rows(grid, rows={})
    pml = PML(grid=grid, thickness=tuple((2, 2) for _ in range(3)))
    verdict = offdiag.offdiag_constitutive_coverage(fields, pml, _Residency())
    assert not verdict.covered
    assert any("no off-diagonal chi1inv row survived" in reason
               for reason in verdict.reasons), verdict.reasons


def test_an_explicitly_zero_row_is_dropped_at_install_rather_than_carried():
    """The installer's own rule, restated as a test because this family's
    disjointness depends on it: a caller passing explicit zeros reduces byte for
    byte to the diagonal engine."""
    grid = _grid()
    zeros = np.zeros(grid.shape, dtype=np.float32)
    fields = _fields_with_rows(grid, rows={"Ex": {"Ey": zeros}})
    assert not fields.has_offdiagonal_epsilon


def test_an_inactive_absorber_is_refused():
    grid = _grid()
    fields = _fields_with_rows(grid)
    inert = PML(grid=grid, thickness=tuple((0, 0) for _ in range(3)))
    verdict = offdiag.offdiag_constitutive_coverage(fields, inert, _Residency())
    assert not verdict.covered
    assert any("PML" in reason for reason in verdict.reasons), verdict.reasons


def test_a_row_volume_that_aliases_an_output_is_refused_before_the_builder_raises():
    """The output-alias clause is beyond even the installer's reach.

    ``set_epsilon_volumes`` keeps the caller's array without copying
    (fields.py:1296), so a row that IS one of the E/f_w outputs arrives legally
    installed. The plan builder refuses the alias with a raise, so the PREDICATE
    must refuse it first — a covered verdict must never meet a ValueError.
    """
    grid = _grid()
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    shape = grid.shape
    epsilon = np.full(shape, np.float32(2.0), dtype=np.float32)
    inverse = np.full(shape, np.float32(0.5), dtype=np.float32)
    fields.Ex[...] = np.float32(0.02)  # non-zero, or the installer would drop it
    fields.set_epsilon_volumes({c: epsilon for c in ("Ex", "Ey", "Ez")},
                               {c: inverse for c in ("Ex", "Ey", "Ez")},
                               {"Ex": {"Ey": fields.Ex}})
    pml = PML(grid=grid, thickness=tuple((2, 2) for _ in range(3)))
    verdict = offdiag.offdiag_constitutive_coverage(fields, pml, _Residency())
    assert not verdict.covered
    assert any("aliases output" in reason for reason in verdict.reasons), \
        verdict.reasons


class _Planted:
    """A ``Fields`` whose off-diagonal rows were planted PAST the installer.

    ``set_epsilon_volumes`` normalizes MOST of this away — it casts to float32,
    refuses complex, refuses a shape mismatch and drops identically-zero rows — so
    those clauses in :func:`offdiag._row_reasons` need a row that arrives another
    way, and this is how a test constructs one.

    NOT ALL OF IT, AND THIS DOCSTRING USED TO SAY OTHERWISE. The cast is
    ``astype(..., copy=False)`` (fields.py:1296), whose default ``order="K"``
    PRESERVES Fortran order, so the CONTIGUITY clause is reachable through the
    front door — see
    :func:`test_an_f_contiguous_row_reaches_the_predicate_through_the_installer`,
    which measures it. The two ALIAS clauses are beyond the installer entirely,
    because it keeps the caller's array without copying.
    """

    def __init__(self, inner, rows):
        self._inner = inner
        self._rows = rows

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def chi1inv_offdiagonal_for(self, component):
        return self._rows.get(component, {})

    @property
    def has_offdiagonal_epsilon(self):
        return True


class _NoDtype:
    """Volume-shaped and answers no dtype — the fail-closed case."""

    shape = (12, 10, 9)


def test_a_row_volume_with_no_readable_dtype_is_REFUSED_rather_than_raising():
    """The predicate must not raise. MEASURED: it used to.

    ``_row_reasons`` read ``str(getattr(value, "dtype", ""))[0]``, which is an
    ``IndexError`` on a volume-shaped object answering no dtype. A raising
    predicate satisfies neither contract it sits between: ``plan_offdiag_constitutive``
    promises None-means-refused, and the arm table treats a raise as a refusal — so
    the same configuration would be reported two different ways depending on which
    caller reached it first.
    """
    grid = _grid()
    fields = _fields_with_rows(grid)
    pml = PML(grid=grid, thickness=tuple((2, 2) for _ in range(3)))
    verdict = offdiag.offdiag_constitutive_coverage(
        _Planted(fields, {"Ex": {"Ey": _NoDtype()}}), pml, _Residency())
    assert not verdict.covered
    assert any("exposes no dtype" in reason for reason in verdict.reasons), \
        verdict.reasons


@pytest.mark.parametrize("label,value,needle", [
    ("float64", np.full((12, 10, 9), 0.03, dtype=np.float64),
     "is not float32"),
    ("complex64", (np.full((12, 10, 9), 0.03) + 0.01j).astype(np.complex64),
     "is complex"),
    ("wrong_shape", np.full((12, 10, 8), np.float32(0.03), dtype=np.float32),
     "!= grid shape"),
    ("f_contiguous",
     np.asfortranarray(np.full((12, 10, 9), np.float32(0.03), dtype=np.float32)),
     "not C-contiguous"),
    ("scalar", np.float32(0.03), "is not a volume"),
])
def test_a_row_planted_past_the_installer_is_refused_by_name(label, value, needle):
    """Every ``_row_reasons`` clause, CONSTRUCTED and CALLED rather than read.

    Four of these five the installer normalizes away, which is why the clauses had
    no test: reading them is not evidence that they fire. The fifth —
    ``f_contiguous`` — the installer does NOT normalize away; the test below
    reaches it through the public installer instead, and this parametrisation keeps
    it as the planted case as well because both routes must be refused by the same
    name.
    """
    grid = _grid()
    fields = _fields_with_rows(grid)
    pml = PML(grid=grid, thickness=tuple((2, 2) for _ in range(3)))
    verdict = offdiag.offdiag_constitutive_coverage(
        _Planted(fields, {"Ex": {"Ey": value}}), pml, _Residency())
    assert not verdict.covered, label
    assert any(needle in reason for reason in verdict.reasons), verdict.reasons


def test_an_f_contiguous_row_reaches_the_predicate_through_the_installer():
    """The contiguity clause is a FRONT-DOOR refusal, not a planted-row one.

    ``_validated_offdiagonal_rows`` casts with ``astype(..., copy=False)``
    (fields.py:1296) and ``order="K"`` keeps Fortran order, so a row built with
    ``numpy.asfortranarray`` installs and arrives at the predicate F-ordered. Two
    assertions, and the first is the one that matters: if the installer ever starts
    normalizing, this test fails and says so rather than quietly becoming a second
    copy of the planted case.
    """
    grid = _grid()
    shape = grid.shape
    row = np.asfortranarray(
        (0.03 * np.random.default_rng(5).standard_normal(shape)).astype(np.float32))
    fields = _fields_with_rows(grid, rows={"Ex": {"Ey": row}})
    installed = fields.chi1inv_offdiagonal_for("Ex").get("Ey")
    assert installed is not None and not installed.flags["C_CONTIGUOUS"], (
        "the installer normalized Fortran order away; the front-door route this "
        "test measures no longer exists and the docstrings that describe it must "
        "be rewritten")
    pml = PML(grid=grid, thickness=tuple((2, 2) for _ in range(3)))
    verdict = offdiag.offdiag_constitutive_coverage(fields, pml, _Residency())
    assert not verdict.covered
    assert any("not C-contiguous" in reason for reason in verdict.reasons), \
        verdict.reasons


@pytest.mark.parametrize("target,needle", [
    ("Ex", "aliases output Ex"),
    ("Dy", "aliases source Dy"),
])
def test_a_row_volume_that_aliases_a_written_mirror_is_refused(target, needle):
    """Both alias hazards, CONSTRUCTED — and the second one is new.

    An OUTPUT alias makes the answer depend on thread schedule. A SOURCE alias is
    worse because it is invisible here: the residency registry keys mirrors by
    NAME, so one host array bound as ``Dy`` (written by ``step_D``) and again as
    ``chi1inv_offdiag:Ex:Ey`` (constant) becomes two device tensors, and the
    coefficient freezes at plan time. Measured on this host 2026-08-15: the single
    sub-step is byte-identical — which is why the byte gate could not see it — and
    two complete driver steps diverge by 19,620 words with
    ``Residency.verify`` reporting ``{'chi1inv_offdiag:Ex:Ey': 1080}``.
    """
    grid = _grid()
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    shape = grid.shape
    epsilon = np.full(shape, np.float32(2.0), dtype=np.float32)
    inverse = np.full(shape, np.float32(0.5), dtype=np.float32)
    getattr(fields, target)[...] = np.float32(0.02)  # or the installer drops it
    fields.set_epsilon_volumes({c: epsilon for c in ("Ex", "Ey", "Ez")},
                               {c: inverse for c in ("Ex", "Ey", "Ez")},
                               {"Ex": {"Ey": getattr(fields, target)}})
    pml = PML(grid=grid, thickness=tuple((2, 2) for _ in range(3)))
    verdict = offdiag.offdiag_constitutive_coverage(fields, pml, _Residency())
    assert not verdict.covered
    assert any(needle in reason for reason in verdict.reasons), verdict.reasons


def test_an_inverse_epsilon_that_aliases_a_source_is_refused():
    """The same hazard on the other read-only input.

    ``set_epsilon_volumes`` stores the inverse map with NO normalization at all
    (fields.py:1251-1252), so this arrives legally installed exactly as the row
    alias does.
    """
    grid = _grid()
    fields = _fields_with_rows(grid)
    shape = grid.shape
    epsilon = np.full(shape, np.float32(2.0), dtype=np.float32)
    inverse = np.full(shape, np.float32(0.5), dtype=np.float32)
    fields.set_epsilon_volumes(
        {c: epsilon for c in ("Ex", "Ey", "Ez")},
        {"Ex": fields.Dx, "Ey": inverse, "Ez": inverse},
        {r: dict(fields.chi1inv_offdiagonal_for(r)) for r in ("Ex", "Ey", "Ez")})
    pml = PML(grid=grid, thickness=tuple((2, 2) for _ in range(3)))
    verdict = offdiag.offdiag_constitutive_coverage(fields, pml, _Residency())
    assert not verdict.covered
    assert any("aliases source Dx" in reason for reason in verdict.reasons), \
        verdict.reasons


def test_an_isotropic_inverse_epsilon_install_is_still_admitted():
    """The alias clauses must not refuse the ONE aliasing that is legal.

    Three inverse-epsilon components aliasing one host array is the isotropic
    install; they are CONSTANT mirrors of a volume nothing writes, so they cannot
    go stale. A clause that refused them would have made the family's own sweep
    (which runs both the aliased and the three-distinct form) unbuildable.

    ASSERTED ON THE REASON SET, NOT ON ``covered``: this host's DEFAULT subnormal
    policy resolves to ``keep``, which the MPS executor refuses, so a coverage
    verdict here would measure the ambient environment rather than the clause.
    """
    grid = _grid()
    fields = _fields_with_rows(grid)
    assert (fields.inverse_epsilon_for("Ex")
            is fields.inverse_epsilon_for("Ez")), "the fixture is not aliased"
    pml = PML(grid=grid, thickness=tuple((2, 2) for _ in range(3)))
    verdict = offdiag.offdiag_constitutive_coverage(fields, pml, _Residency())
    assert not any("aliases" in reason for reason in verdict.reasons), \
        verdict.reasons


def test_the_plan_builder_returns_none_out_of_coverage():
    """None means REFUSED. A configuration this kernel does not carry must fall back
    to the array path, never raise into a caller that would otherwise have stepped
    correctly."""
    grid = _grid()
    fields = _fields_with_rows(grid, rows={})
    pml = PML(grid=grid, thickness=tuple((2, 2) for _ in range(3)))
    assert offdiag.plan_offdiag_constitutive(fields, pml, _Residency()) is None


def test_wall_mask_axes_reads_the_declaration_not_the_resolved_ghost_rule():
    """``is_metallic and not is_mirrored`` — the grid's own declaration.

    A metallic axis that is ALSO mirrored declares no wall for this mask (the fold
    plane genuinely carries coupling), and reading ``_boundary_kinds`` here instead
    would zero a plane MEEP steps.
    """
    walls = offdiag.wall_mask_axes(_grid(boundaries="metallic"))
    assert walls == (1, 1, 1)
    assert offdiag.wall_mask_axes(_grid()) == (0, 0, 0)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_the_arm_is_registered_and_wired_on_update_E_alone():
    """One arm, on ``update_E``, WIRED as of tranche 2.

    The slot matters as much as the flag: this family's whole effect is inside
    ``update_E`` (stepping.py:1001-1008), so an arm on any curl slot would be
    claiming a sub-step it does not implement.
    """
    specs = [s for s in arms.registered(offdiag.SLOT)
             if s.family == offdiag.FAMILY]
    assert len(specs) == 1, specs
    assert specs[0].wired is True
    assert specs[0].slot == "update_E"
    assert not [s for slot in ("step_B", "step_D", "update_H")
                for s in arms.registered(slot) if s.family == offdiag.FAMILY]


def test_re_registering_the_same_arm_is_refused():
    """Import order must never decide which kernel a slot launches."""
    with pytest.raises(ValueError, match="already registered"):
        arms.register(family=offdiag.FAMILY, slot=offdiag.SLOT, label="offdiag",
                      coverage=offdiag._arm_coverage, plan=offdiag._arm_plan,
                      prefix="offdiag: ", noun="off-diagonal constitutive",
                      wired=True)


# ---------------------------------------------------------------------------
# Device-backed
# ---------------------------------------------------------------------------

@needs_mps
def test_one_update_e_is_byte_identical_to_stepping(monkeypatch):
    """The claim in miniature, so a regression fails the merge bar and not only the
    gate. uint32 words, never ``allclose``, and the step is proven to have MOVED
    STATE — a no-op agreeing with a no-op is trivially identical.

    THE POLICY IS PINNED PER TEST, as the certified suite pins it. This arm64
    host's DEFAULT resolves to ``keep`` (MEEP's ``set_zero_subnormals`` is a no-op
    under ``#if HAVE_IMMINTRIN_H``, so ``match_meep`` measures keep), and the MPS
    executor cannot honour keep — Metal flushes denormals natively with no lever.
    Requesting ``flush`` explicitly is what the gate does and is the only policy
    under which this claim is measurable at all; a test that inherited the ambient
    policy would pass or fail on the shell's environment rather than on the code.
    """
    from meep_gpu.metal_kernels import device

    monkeypatch.setenv(subnormal.POLICY_ENV, subnormal.FLUSH)

    grid = Grid(resolution=10.0, cell_size=(1.2, 1.0, 0.9),
                boundaries=("metallic", "metallic", "periodic"), dimensions=3,
                courant=0.35, k_point=(0.0, 0.0, 0.0), xp=np)
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    shape = grid.shape
    rng = np.random.default_rng(20260815)
    epsilon = (1.45 + 0.3 * rng.random(shape)).astype(np.float32)
    inverse = (np.float32(1.0) / epsilon).astype(np.float32)
    rows = {"Ex": {"Ey": (0.03 * rng.standard_normal(shape)).astype(np.float32),
                   "Ez": (0.02 * rng.standard_normal(shape)).astype(np.float32)},
            "Ez": {"Ex": (0.04 * rng.standard_normal(shape)).astype(np.float32)}}
    fields.set_epsilon_volumes({c: epsilon for c in ("Ex", "Ey", "Ez")},
                               {c: inverse for c in ("Ex", "Ey", "Ez")}, rows)
    pml = PML(grid=grid, thickness=tuple((2, 2) for _ in range(3)))
    names = ("Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez",
             "Dx", "Dy", "Dz", "Hx", "Hy", "Hz")
    for name in names:
        getattr(fields, name)[...] = (
            rng.standard_normal(shape) * 0.37).astype(np.float32)

    compared = ("Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez")
    before = {n: np.array(getattr(fields, n), copy=True) for n in names}
    stepping.update_E(fields, pml)
    after = {n: np.array(getattr(fields, n), copy=True) for n in compared}

    def words(a):
        return np.ascontiguousarray(a, dtype=np.float32).reshape(-1).view(np.uint32)

    moved = sum(int(np.count_nonzero(words(before[n]) != words(after[n])))
                for n in compared)
    assert moved > 0, "the oracle did nothing; identity here would be vacuous"

    for name, value in before.items():
        getattr(fields, name)[...] = value
    residency = device.Residency()
    plan = offdiag.plan_offdiag_constitutive(fields, pml, residency)
    assert plan is not None, offdiag.offdiag_constitutive_coverage(
        fields, pml, residency).reasons
    plan.run()
    residency.sync_out()
    assert plan.launches == 1
    for name in compared:
        assert int(np.count_nonzero(
            words(getattr(fields, name)) != words(after[name]))) == 0, name
    assert not residency.verify()


@needs_mps
def test_asking_for_an_unbuilt_contract_variant_raises(monkeypatch):
    """The guard selector must not fall back to the pinned source: a gate leg that
    measured the guard through a silent fallback would certify nothing."""
    from meep_gpu.metal_kernels import device

    monkeypatch.setenv(subnormal.POLICY_ENV, subnormal.FLUSH)
    grid = _grid()
    fields = _fields_with_rows(grid)
    pml = PML(grid=grid, thickness=tuple((2, 2) for _ in range(3)))
    plan = offdiag.plan_offdiag_constitutive(fields, pml, device.Residency())
    assert plan is not None
    assert plan.variants == (shaders.CONTRACT_OFF,)
    with pytest.raises(KeyError, match="holds no 'fast' variant"):
        plan.run(contract=shaders.CONTRACT_FAST)

@needs_mps
def test_the_keep_policy_refuses_the_plan_at_build_time(monkeypatch):
    """``keep`` is a REFUSAL on this executor, not a silent downgrade.

    Metal flushes float32 denormals natively and exposes no lever — both denormal
    pragma spellings are compile errors on this toolchain — so a run resolved to
    ``keep`` is refused at PLAN time rather than run under a policy the device was
    never in. This host's default resolves to keep, which is exactly why the two
    tests above pin ``flush`` explicitly, and this test is the other half of that
    pair: it proves the refusal is real rather than assumed.
    """
    from meep_gpu.metal_kernels import device

    monkeypatch.setenv(subnormal.POLICY_ENV, subnormal.KEEP)
    grid = _grid()
    fields = _fields_with_rows(grid)
    pml = PML(grid=grid, thickness=tuple((2, 2) for _ in range(3)))
    verdict = offdiag.offdiag_constitutive_coverage(fields, pml, device.Residency())
    assert not verdict.covered
    assert any("subnormal" in reason for reason in verdict.reasons), verdict.reasons
    assert offdiag.plan_offdiag_constitutive(fields, pml, device.Residency()) is None
