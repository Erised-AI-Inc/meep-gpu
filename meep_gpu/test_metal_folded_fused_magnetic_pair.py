"""The folded fused magnetic Metal kernel, as claims a laptop can check without a GPU.

WHAT THIS SUITE OWNS AND WHAT IT DELIBERATELY DOES NOT. The BYTES are the device
gate's — ``parity/meep_gpu/gate_metal_folded_fused_magnetic_pair.py`` steps two
engines side by side for twelve complete steps and compares uint32 words, and no
assertion here duplicates that. What lives here is everything true about the family
WITHOUT a device: what the emitted source says, which configurations the predicate
refuses and by what name, that the arm is registered UNWIRED, and that the plan binds
the lattice each half is supposed to take.

THE FOUR CLAIMS THIS SUITE MAKES THAT THE GATE CANNOT.

1. That the product is invisible to ``plan_step`` — a property of the TABLE, and the
   whole reason this family may land beside the wired tree without touching it.
2. That the near-fill geometry is the B one and NOT the D one. The gate measures
   that the shipped kernel is right; only a table-level assertion says that a
   component is a ghost destination on its OWN axis alone, which is the single most
   likely slip in porting this product from :mod:`~meep_gpu.metal_kernels.folded_fused_pair`.
3. That the imaged ghost reads its coefficient pair at stored index 0 rather than
   reusing the source thread's. That is the one line the sibling does not carry, and
   on ordinary fixtures it is BYTE-INVISIBLE (the gate's ``moved_coefficient`` leg
   measures exactly that), so a source-level assertion is what keeps it honest.
4. That both halves are the certified emitters' own text, checked by re-running
   those emitters — the construction the module claims, made falsifiable.
"""

from __future__ import annotations

import pathlib

import pytest

PACKAGE_DIR = pathlib.Path(__file__).resolve().parent
REPO = PACKAGE_DIR.parent
GATE = REPO / "parity" / "meep_gpu" / "gate_metal_folded_fused_magnetic_pair.py"

from meep_gpu.metal_kernels import (  # noqa: E402
    arms, folded_fused_magnetic_pair as family, folded_fused_pair as sibling,
    fused_magnetic_pair as unfolded, registry, shaders, symmetry,
)
from meep_gpu.triton_kernels.symmetry import (  # noqa: E402
    CODE_METALLIC, CODE_MIRROR_METALLIC, CODE_MIRROR_PERIODIC, CODE_PERIODIC,
    TARGET_IYEE,
)

#: One folded axis, one wall, one free axis — the smallest triple that emits every
#: kind of line the family can emit except a second parity.
CODES = (CODE_MIRROR_METALLIC, CODE_MIRROR_METALLIC, CODE_METALLIC)
PHASES = (1, -1, 0)
WALLS = (False, False, True)


def source(codes=CODES, phases=PHASES, walls=WALLS,
           contract=shaders.CONTRACT_OFF) -> str:
    return family.folded_fused_magnetic_pair_source(codes, phases, walls, contract)


# ---------------------------------------------------------------------------
# 1. The source is a LIFT, not a transcription — so the lift is checkable
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("codes", (
    (CODE_MIRROR_METALLIC, CODE_PERIODIC, CODE_PERIODIC),
    (CODE_MIRROR_METALLIC, CODE_MIRROR_METALLIC, CODE_METALLIC),
    (CODE_MIRROR_METALLIC, CODE_MIRROR_METALLIC, CODE_MIRROR_METALLIC),
))
def test_the_curl_head_is_the_certified_folded_curl_bodys_own_bytes(codes):
    """The lifted head is a VERBATIM PREFIX of the certified folded curl body.

    This family does not retype the curl half; it cuts the certified emitter's own
    output at the split-field block and re-emits only below the cut. That makes "the
    fused ghost gather, curl grouping and cell-0 mask ARE the certified ones" a
    string comparison rather than a claim.
    """
    head = family.certified_curl_head(codes)
    certified = symmetry.folded_curl_source(codes, family.BACKWARD)
    body = certified.split("uint idx [[thread_position_in_grid]])\n{\n", 1)[1]
    assert head.strip()
    assert body.startswith(head)
    assert head in source(codes, (1, 1, 1), (False, False, False))


def test_the_curl_half_is_the_forward_direction():
    """``step_B`` shifts UP and ``step_D`` shifts DOWN. Getting this backwards is a
    whole-grid defect that still converges."""
    from meep_gpu.triton_kernels.launch import SUB_STEPS

    assert family.BACKWARD is False
    assert bool(SUB_STEPS[family.CURL_SUB_STEP]["backward"]) is family.BACKWARD
    assert "int si = i + 1, sj = j + 1, sk = k + 1;" in source()


def test_the_recurrence_statements_are_the_certified_ones():
    """Every ``p``/``n``/``v`` line is pulled out of the certified body by anchor.

    Not retyped: an anchor that stopped identifying exactly one line raises in
    :func:`certified_curl_statements` rather than splicing an arbitrary one. The
    ``n`` statement is the certified line verbatim; the ``p`` and ``v`` statements
    are the certified lines with EXACTLY the :data:`LIFT_EDITS` fragment replaced,
    and nothing else moved.
    """
    statements = family.certified_curl_statements(CODES)
    certified = symmetry.folded_curl_source(CODES, family.BACKWARD)
    lines = certified.splitlines()
    text = source()
    for target, (previous, stage, stepped) in enumerate(statements["recurrence"]):
        assert stage in lines
        for (prefix, old, _new), lifted in zip(family.LIFT_EDITS, (previous, stepped)):
            original = [line for line in lines if line.startswith(prefix.format(t=target))]
            assert len(original) == 1, original
            fragment = old.format(t=target)
            assert original[0].endswith(fragment), (original, fragment)
            assert lifted.startswith(original[0][: -len(fragment)])
            assert lifted != original[0]
        for line in (previous, stage, stepped):
            assert line.strip() in text
    assert statements["flags"] in lines
    for store in statements["aux_stores"]:
        assert store in lines and store in text


def test_the_dsig_only_form_is_the_recurrence_first_stage_with_fu_as_f():
    """The edited ``v`` spells the dsig-only form as the ``n`` statement with ``p := f``.

    Same operations, same order, same rounding: the certified ``v`` takes ``n`` there,
    and ``n`` was formed from ``p = f``. So the derived expression must be exactly the
    ``n`` statement's right-hand side with ``p{t}`` read as ``f{t}[ii]``.
    """
    statements = family.certified_curl_statements(CODES)
    for target, (_previous, stage, stepped) in enumerate(statements["recurrence"]):
        expression = stage.split(" = ", 1)[1].rstrip()[:-1]
        derived = "(" + expression.replace(f"p{target}", f"f{target}[ii]") + ")"
        assert stepped.rstrip().endswith(f": {derived};"), (stepped, derived)


def test_a_destination_thread_never_loads_the_field_its_source_writes():
    """Before the first ownership guard, no ``f{t}[ii]`` is read.

    The certified curl reads ``p = du ? fu : f`` at every cell; on a destination
    thread that ``f`` is the word its source thread writes in the same dispatch. The
    lift moves that read into the owned block, so the shared prologue loads only
    ``u{t}`` and the operands.
    """
    text = source((CODE_MIRROR_PERIODIC, CODE_MIRROR_PERIODIC, CODE_PERIODIC),
                  (1, -1, 0), (False, False, False))
    shared = text.split("    if (!(", 1)[0]
    for target in range(3):
        assert f"f{target}[ii]" not in shared, target
        assert f"float p{target} = du{target} ? u{target}[ii] : 0.0f;" in shared


def test_the_constitutive_statements_reproduce_the_certified_h_body():
    """The parameterised seven statements ARE ``shaders.constitutive_source('H')``'s.

    This family re-emits that body twice per folded component — the owned cell and
    the imaged ghost — so it cannot splice it wholesale the way the unfolded fused
    magnetic pair does. Rendering the template with the certified spellings must give
    the certified statements back, and the source builder raises if it does not.
    """
    for mode in shaders.CONTRACT_MODES:
        result = family.constitutive_transcription(mode)
        assert result["identical"], result
        assert [len(block) for block in result["certified"]] == [6, 6, 6]


def test_the_h_side_carries_no_inverse_mu():
    """``update_H``'s source is B with mu = 1 baked into the array path too, which is
    what makes this signature 27 pointers rather than the E side's 30."""
    text = source()
    assert "ie0" not in text and "inv_eps" not in text
    assert "float o0_src = v0;" in text


# ---------------------------------------------------------------------------
# 2. The B GEOMETRY, which is the whole difference from the D-side sibling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target", (0, 1, 2))
def test_a_component_is_a_ghost_destination_on_its_own_axis_alone(target):
    """``iyee[Bm][a] == 0`` iff ``a == m`` — the exact COMPLEMENT of the D family.

    The D side fills the two axes that are NOT the component's own and can therefore
    write up to three ghost cells per component; this side fills one axis and writes
    one. A family that inherited the sibling's ``near_fill_axes`` would image the
    wrong planes on every folded grid.
    """
    every = (CODE_MIRROR_METALLIC,) * 3
    assert family.near_fill_axes(every, target) == (target,)
    assert TARGET_IYEE[("Bx", "By", "Bz")[target]][target] == 0
    # The sibling's answer on the same codes, for the same target, is the complement.
    assert sibling.near_fill_axes(every, target) == tuple(
        axis for axis in range(3) if axis != target)


def test_no_configuration_gives_a_component_two_ghost_axes():
    """No composite corner can exist here, and the builder REFUSES one rather than
    assuming it cannot arise."""
    import itertools

    for codes in itertools.product((CODE_PERIODIC, CODE_METALLIC,
                                    CODE_MIRROR_METALLIC), repeat=3):
        for target in range(3):
            assert len(family.near_fill_axes(codes, target)) <= 1


def test_the_ghost_reads_its_own_coefficient_pair_at_stored_index_zero():
    """THE ONE LINE THE SIBLING DOES NOT CARRY.

    ``update_H`` indexes ``kps``/``kms`` on the component's own axis, and on this
    family the fill images along that same axis — so the source at stored 2 and the
    destination at stored 0 take DIFFERENT entries. The sibling's "the coefficient
    index does not move" is FALSE here.

    This is asserted at the SOURCE because it is byte-invisible on an ordinary
    fixture: a folded axis carries no absorber at the mirror plane, so a shallow
    layer leaves ``kps[0] == kps[2]`` exactly. The gate's ``moved_coefficient`` leg
    is where that observability is measured.
    """
    text = source()
    assert "float g0_n_kp = kp0[0], g0_n_km = km0[0];" in text
    assert "float g1_n_kp = kp1[0], g1_n_km = km1[0];" in text
    assert ("g0_n_acc = h0[g0_n_i] + ((g0_n_kp * g0_n_src) - "
            "(g0_n_km * g0_n_prev));") in text
    # and the OWNED cell still takes the source thread's own entry
    assert "o0_acc = h0[ii] + ((kp_0 * o0_src) - (km_0 * o0_prev));" in text


def test_the_near_ghost_selects_its_form_at_its_own_coordinate():
    """The H half's membership is on the component's OWN axis, which the near fill
    moves, so the ghost's flag is evaluated at stored 0 — through the shared
    emitter, reading the integer-lattice locals the lifted block declares — and the
    ghost's six statements read that flag, not the owner's."""
    text = source()
    assert ("const bool in_i_x_g0_n = ((0) >= cbi_x_a0 && (0) < cbi_x_a1) || "
            "((0) >= cbi_x_b0 && (0) < cbi_x_b1);") in text
    assert "float g0_n_prev = in_i_x_g0_n ? w0[g0_n_i] : 0.0f;" in text
    assert "if (in_i_x_g0_n) { w0[g0_n_i] = g0_n_src; }" in text
    # The owner selects on its own flag, lifted from the certified H body.
    assert "    const bool dw0 = in_i_x, dw1 = in_i_y, dw2 = in_i_z;" in text
    assert "float o0_prev = dw0 ? w0[ii] : 0.0f;" in text


def test_the_near_ghost_and_its_source_can_take_different_forms():
    """Why the near ghost's flag is evaluated at stored 0 and not read off its source.

    On a deep high-face layer the record's integer line on a folded axis is OUT at
    stored 0 and IN at stored ``NEAR_SOURCE_INDEX``: the ghost steps the plain form
    while its source accumulates. Keying the ghost at its source would accumulate at
    the ghost — a different word, which the gate's ``moved_coefficient`` leg arms and
    must catch on exactly this geometry. Built here from the engine's own objects.
    """
    from meep_gpu.chunk_map import ChunkLayout
    from meep_gpu.grid import Grid, Mirror
    from meep_gpu.metal_kernels import chunk_bounds
    from meep_gpu.pml import PML

    grid = Grid(resolution=10.0, cell_size=(0.8, 0.8, 1.2), dimensions=3,
                courant=0.35, boundaries={"x": "metallic", "y": "metallic",
                                          "z": "metallic"},
                symmetry=(Mirror("X", 1), Mirror("Y", -1)))
    pml = PML(grid=grid, thickness=((0, 4), (0, 4), (2, 2)))
    record = chunk_bounds.for_layout(ChunkLayout.one_rank(grid, pml), grid.shape)
    line = chunk_bounds.decode(record, family.CONSTITUTIVE_HALF_INTEGER, 0,
                               grid.shape[0])
    assert not line[0] and line[family.NEAR_SOURCE_INDEX], line
    # and the emitted ghost reads the flag it evaluates at 0, never the owner's
    text = source()
    ghost = text.split("        if (i == 2) {", 1)[1].split("        }", 1)[0]
    assert "in_i_x_g0_n" in ghost and "dw0" not in ghost


def test_the_ghost_is_written_by_the_source_thread_not_the_destination():
    """The ownership restructure, at the source level: the whole per-component tail
    sits inside ``coordinate != 0`` and the carry inside ``coordinate == 2``."""
    text = source()
    assert "    if (!(i == 0)) {" in text
    assert "        if (i == 2) {" in text
    assert "int g0_n_i = ii - 2 * nyz;" in text


@pytest.mark.parametrize("phase,spelling", ((1, "float g0_n_v = v0;"),
                                            (-1, "float g0_n_v = -v0;")))
def test_the_parity_is_a_compile_time_sign(phase, spelling):
    """``-x`` for odd and a plain copy for even — never a runtime weight, which is
    measured to flush every subnormal on this backend at BOTH signs."""
    assert spelling in source(CODES, (phase, 1, 0), WALLS)


def test_the_wall_clear_is_the_b_diagonal_and_is_imported_not_respelled():
    """Bx clears on an x wall, By on y, Bz on z — the table the unfolded fused
    magnetic pair already pins. Imported, so the two cannot drift."""
    assert family._ZERO_METAL_ROWS is unfolded._ZERO_METAL_ROWS
    text = source((CODE_MIRROR_METALLIC, CODE_METALLIC, CODE_METALLIC),
                  (1, 0, 0), (False, True, True))
    assert "v1 = at_y ? 0.0f : v1;" in text
    assert "v2 = at_z ? 0.0f : v2;" in text
    assert "v0 = at_x" not in text  # x is folded, and a folded axis is never walled


def test_the_imaged_ghost_never_takes_a_wall_line():
    """The fill and the wall clear are mutually exclusive PER COMPONENT here, so the
    D side's parity-then-clear ordering question is vacuous.

    Checked rather than inferred: an overlap would raise out of the carry builder.
    """
    text = source()
    ghost = text.split("        if (i == 2) {", 1)[1].split("        }", 1)[0]
    assert "at_x" not in ghost and "at_y" not in ghost and "at_z" not in ghost


def test_a_folded_axis_that_is_also_walled_is_refused_at_build_time():
    with pytest.raises(ValueError, match="both folded and walled"):
        source(CODES, PHASES, (True, False, True))


def test_a_folded_periodic_axis_is_CARRIED_not_refused():
    """THE FLIP OF 2026-08-20, pinned at the source.

    This test asserted the OPPOSITE until the far carry landed: a folded PERIODIC
    axis raised ``ValueError`` naming ``fill_folded_far_ghosts_B``, because that
    pass runs inside the seam (driver.py:3287) and no product imaged the top
    stored slot. The kernel images it now, so the same configuration must BUILD,
    and what is pinned is the text the carry emits:

      * the far ghost's destination is the TOP plane, built from the RUNTIME
        reflect row rather than a baked ``n - 2``;
      * its parity is ``mirror_parity`` on a SHIFT-1 component, i.e. ``-phase``,
        the opposite of the near fill's;
      * the folded curl's top-plane mask, which fires only on this code, is
        present — a kernel that stepped that plane and then imaged over it would
        leave ``fu`` carrying the unmasked recurrence.
    """
    text = source((CODE_MIRROR_PERIODIC, CODE_PERIODIC, CODE_PERIODIC), (1, 0, 0),
                  (False, False, False))
    # Bx has Yee shift 0 on x, so on a folded x axis it takes the NEAR fill; By and
    # Bz have shift 1 there and take the FAR one. That is fields.IYEE_SHIFTS, and
    # the two sets are complementary.
    assert "int g0_n_i = ii - 2 * nyz;" in text
    assert "int g1_i_i = ii + ((nxi - 1) - reflect_x) * nyz;" in text
    assert "int g2_i_i = ii + ((nxi - 1) - reflect_x) * nyz;" in text
    # parity: +phase on the shift-0 near ghost, -phase on the shift-1 far ones
    assert "float g0_n_v = v0;" in text
    assert "float g1_i_v = -v1;" in text
    # the top-plane mask, emitted only on MIRROR_PERIODIC
    assert "curl1 = last_x ? 0.0f : curl1;" in text
    assert "curl2 = last_x ? 0.0f : curl2;" in text
    # and the far ghost keeps its source thread's constitutive pair AND flag: it
    # moves along an axis that is NOT the one update_H indexes on
    assert ("g1_i_acc = h1[g1_i_i] + ((kp_1 * g1_i_src) - (km_1 * g1_i_prev));"
            in text)
    assert "float g1_i_prev = dw1 ? w1[g1_i_i] : 0.0f;" in text


def test_the_far_carry_composes_with_the_near_one_and_with_itself():
    """Two folded PERIODIC axes: a component can be a destination on BOTH.

    Bx is a NEAR destination on x and a FAR one on y, so the cell at stored 0 on x
    AND the top plane on y carries the PRODUCT of the two parities; Bz is a far
    destination on both and carries the corner. Neither exists on this family
    until the far carry does — a B component is a near destination on ONE axis.
    """
    codes = (CODE_MIRROR_PERIODIC, CODE_MIRROR_PERIODIC, CODE_PERIODIC)
    text = source(codes, (1, -1, 0), (False, False, False))
    # the composite near/far ghost, and its coefficient pair reloaded at index 0
    # because its NEAR half still moves the indexed axis
    assert "int g0_jn_i = ii + ((nyi - 1) - reflect_y) * nzi - 2 * nyz;" in text
    assert "float g0_jn_kp = kp0[0], g0_jn_km = km0[0];" in text
    # the far/far corner, at the product of the two parities: Bz has shift 1 on
    # both, so -(+1) * -(-1) = -1
    assert ("int g2_ij_i = ii + ((nxi - 1) - reflect_x) * nyz "
            "+ ((nyi - 1) - reflect_y) * nzi;") in text
    assert "float g2_ij_v = -v2;" in text
    # every destination is owned by exactly one thread, so the owning thread's own
    # cell must be excluded on EVERY plane it is a destination of
    assert "    if (!(i == 0 || last_y)) {" in text
    assert "    if (!(last_x || last_y)) {" in text


def test_the_destination_set_is_every_composition_and_nothing_else():
    """``carried_destinations`` is the ownership rule, checked against the Yee table.

    2^far * (near ? 2 : 1) - 1 cells per component: the thread's own cell is not a
    ghost, and everything else the fills leave is. Read off the same table the
    kernel branches on rather than restated.
    """
    codes = (CODE_MIRROR_PERIODIC, CODE_MIRROR_PERIODIC, CODE_MIRROR_PERIODIC)
    for target in range(3):
        near = family.near_fill_axes(codes, target)
        far = family.far_fill_axes(codes, target)
        assert len(near) == 1 and len(far) == 2
        assert not (set(near) & set(far))
        owned = family.carried_destinations(near, far)
        assert len(owned) == 2 ** len(far) * 2 - 1 == 7
        assert len(set(owned)) == len(owned)
        assert ((), False) not in owned


def test_an_unfolded_grid_belongs_to_the_other_family():
    with pytest.raises(ValueError, match="no axis is folded"):
        source((CODE_PERIODIC, CODE_METALLIC, CODE_PERIODIC), (0, 0, 0),
               (False, True, False))


@pytest.mark.parametrize("phases", ((0, -1, 0), (2, -1, 0), (None, -1, 0)))
def test_a_folded_axis_without_a_readable_parity_is_refused(phases):
    with pytest.raises(ValueError, match="mirror phase"):
        source(CODES, phases, WALLS)


def test_no_top_plane_mask_is_emitted():
    """Refusing MIRROR_PERIODIC retires ``folded_top_plane_mask``'s only live block,
    so a ``last_*`` in a select here would be a mask nothing asked for."""
    text = source()
    assert "curl0 = last_" not in text
    assert "curl1 = last_" not in text
    assert "curl2 = last_" not in text


# ---------------------------------------------------------------------------
# 3. The predicate: what it refuses, and by what NAME
# ---------------------------------------------------------------------------

class _Source:
    """The smallest thing the predicate reads: a declared ``field_type``."""

    def __init__(self, field_type: str) -> None:
        self.field_type = field_type


def test_an_undeclared_source_set_is_refused_by_name():
    """IGNORANCE IS NOT AN EMPTY SET. ``Fields`` does not hold the source list, so
    inferring "no sources" from not being told is the over-covering refusal this
    clause exists to prevent."""
    coverage = family.metal_folded_fused_magnetic_pair_coverage(object(), None, None)
    assert not coverage.covered
    assert any("was not declared" in reason for reason in coverage.reasons)


def test_a_magnetic_source_with_no_deposit_index_is_refused_by_name():
    """THE POLARITY, and what survives of it since the deposit became carryable.

    A magnetic current lands BETWEEN ``step_B`` and ``update_H``
    (driver.py:3283-3284); an electric one lands in the other half. Since 2026-08-28
    this family CARRIES an in-seam magnetic deposit, so the polarity no longer refuses
    by itself -- but ``_Source`` publishes no ``_point_ix``/``_point_iy``/``_point_iz``
    and a repair cannot save cells a source will not name, so THAT is refused, by name,
    and only for the seam's own field type.
    """
    magnetic = family.metal_folded_fused_magnetic_pair_coverage(
        object(), None, (_Source("B"),))
    assert any("does not publish the index it writes" in reason
               for reason in magnetic.reasons), magnetic.reasons
    electric = family.metal_folded_fused_magnetic_pair_coverage(
        object(), None, (_Source("D"),))
    assert not any("source 0" in reason for reason in electric.reasons)


def test_holding_the_shipped_flag_False_brings_the_polarity_refusal_straight_back(
        monkeypatch):
    """THE OTHER DIRECTION. Without it, the clause above would be consistent with the
    polarity having been dropped rather than carried -- the two answers are different
    strings and only the flag chooses between them."""
    monkeypatch.setattr(family, "CARRIES_DEPOSIT_REPAIR", False)
    magnetic = family.metal_folded_fused_magnetic_pair_coverage(
        object(), None, (_Source("B"),))
    assert any("is magnetic" in reason and "driver.py:3283-3284" in reason
               for reason in magnetic.reasons), magnetic.reasons
    electric = family.metal_folded_fused_magnetic_pair_coverage(
        object(), None, (_Source("D"),))
    assert not any("source 0" in reason for reason in electric.reasons)


def test_the_polarity_is_the_opposite_of_the_folded_d_side_pair():
    """The two folded fused pairs must not both refuse the same source, or one has
    copied the other's clause rather than read the driver."""
    d_side = sibling.metal_folded_fused_pair_coverage(object(), None, (_Source("B"),))
    assert not any("source 0" in reason for reason in d_side.reasons)


def test_the_predicate_never_raises_on_a_degenerate_object():
    """``plan_step``'s whole contract is that a refusal is a verdict, not an
    exception: a raising predicate would escape the composer's arm loop."""
    coverage = family.metal_folded_fused_magnetic_pair_coverage(object(), None, ())
    assert not coverage.covered and coverage.reasons
    assert family.plan_metal_folded_fused_magnetic_pair(
        object(), None, (), chunk_bounds=None) is None


def test_an_admitted_plan_without_a_record_is_refused_by_name(monkeypatch):
    """The record has no default: a plan the predicate admits but that is given no
    division raises the record's own refusal, before any volume is bound."""
    from meep_gpu.chunk_map import ChunkLayoutRefusal
    from meep_gpu.fields import Fields
    from meep_gpu.grid import Grid, Mirror
    from meep_gpu.pml import PML
    from meep_gpu.triton_kernels.coverage import Coverage

    grid = Grid(resolution=10.0, cell_size=(1.6, 3.0, 1.0), boundaries="metallic",
                symmetry=(Mirror("Y", 1),))
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    monkeypatch.setattr(family, "metal_folded_fused_magnetic_pair_coverage",
                        lambda *args, **kwargs: Coverage(True, ()))
    with pytest.raises(ChunkLayoutRefusal, match="MetalFoldedFusedMagneticPairPlan"):
        family.plan_metal_folded_fused_magnetic_pair(
            fields, PML(grid=grid, thickness=0.2), (), None, chunk_bounds=None)


def test_the_record_rides_in_params_and_costs_no_binding():
    """The chunk record is one ``Params`` member after the eight scalars: 132 bytes,
    the record at offset 32, the struct declared once and aliased as ``cb`` — and the
    binding count stays the unfolded pair's 28."""
    from meep_gpu.metal_kernels import chunk_bounds

    dtype = family.params_record_dtype()
    assert dtype.itemsize == 132
    assert dtype.fields["cb"][1] == 32
    assert dtype.fields["cb"][0] == chunk_bounds.record_dtype()
    text = source()
    assert text.count(chunk_bounds.struct_source()) == 1
    assert chunk_bounds.params_member() in text
    assert chunk_bounds.params_alias() in text
    assert "[[buffer(28)]]" not in text


def test_both_halves_predicates_are_conjoined_and_labelled():
    """A configuration either half refuses is refused here with THAT half's reasons,
    prefixed, so a reader can tell which side spoke."""
    coverage = family.metal_folded_fused_magnetic_pair_coverage(object(), None, ())
    prefixes = {reason.split(":")[0] for reason in coverage.reasons}
    assert "folded curl half" in prefixes
    assert "folded constitutive half" in prefixes


# ---------------------------------------------------------------------------
# 4. The table: registered, enumerable, and INVISIBLE to plan_step
# ---------------------------------------------------------------------------

def test_the_arm_is_registered_unwired_on_exactly_one_slot():
    rows = [spec for spec in arms.registered() if spec.family == family.FAMILY]
    assert len(rows) == 1, rows
    assert rows[0].slot == family.SLOT == "step_B"
    assert rows[0].wired is False, (
        "this product spans step_B, fill_B, zero_metal_B and update_H; wiring it "
        "would hand plan_step a slot assignment no composition rule has measured")


def test_plan_step_cannot_select_it():
    """``arms_for`` is the gate between the table and the composer. An unwired row
    stays ENUMERABLE — which is what lets a disjointness sweep see it — while never
    reaching ``_select_slot``."""
    context = arms.StepContext(object(), None, None, (), sources=())
    labels = [arm.label for arm in arms.arms_for(family.SLOT, context)]
    assert "folded fused B/H pair" not in labels
    assert "folded fused B/H pair" in [
        spec.label for spec in arms.registered(family.SLOT)]


def test_the_family_is_in_family_modules():
    """A family in the tree but not in ``registry`` is invisible to ``plan_step``,
    which is a silent coverage loss rather than an error."""
    assert family.FAMILY in registry.FAMILY_MODULES


def test_the_plan_declares_every_pass_it_performs():
    """``replaces_sub_steps`` is READ by the composer and by the whole-step walk.

    A plan that under-declared would have the seam passes it actually performs
    bracketed with a host sync, which is a stale mirror rather than a wrong number.
    """
    assert family.REPLACES == ("step_B", "fill_B", "zero_metal_B",
                              "fill_folded_far_ghosts_B", "update_H")
    plan = family.MetalFoldedFusedMagneticPairPlan
    assert plan.replaces_sub_steps == family.REPLACES
    assert plan.launches_per_run == 1
    assert plan.performs_device_work is True


# ---------------------------------------------------------------------------
# 5. The platform ceiling that decides the product's SHAPE
# ---------------------------------------------------------------------------

def test_the_signature_is_the_unfolded_pairs_signature_exactly():
    """The fold adds NO argument: the stored extent carries it and the PML
    coefficient vectors are already built at that extent. So the binding counts and
    the refuted source are IMPORTED rather than re-spelled."""
    assert family.SEPARATE_SCALAR_BINDINGS is unfolded.SEPARATE_SCALAR_BINDINGS
    assert family.PACKED_BINDINGS is unfolded.PACKED_BINDINGS
    assert family.refuted_separate_scalar_source is (
        unfolded.refuted_separate_scalar_source)


def test_the_shipped_packed_signature_fits_with_room():
    """27 pointers plus one packed struct is 28 — under the ceiling, and ONE dispatch
    for all three components."""
    from meep_gpu.metal_kernels.device import MAX_BUFFER_BINDINGS

    text = source()
    highest = max(int(chunk.split(")")[0])
                  for chunk in text.split("[[buffer(")[1:])
    assert highest + 1 == family.PACKED_BINDINGS == 28
    assert family.PACKED_BINDINGS <= MAX_BUFFER_BINDINGS
    assert family.SEPARATE_SCALAR_BINDINGS > MAX_BUFFER_BINDINGS


def test_the_contraction_guard_is_emitted_in_both_modes():
    """The guard is a property of the SOURCE on this backend, so each mode is a
    different compiled kernel and a plan built without one must raise rather than
    launch the pinned one."""
    off = source(contract=shaders.CONTRACT_OFF)
    fast = source(contract=shaders.CONTRACT_FAST)
    assert "contract(off)" in off
    assert off != fast


# ---------------------------------------------------------------------------
# 6. The gate exists, and it carries the leg the family's one novel line needs
# ---------------------------------------------------------------------------

def test_the_device_gate_is_present_and_routes_through_the_runner():
    assert GATE.is_file()
    text = GATE.read_text(encoding="utf-8")
    assert "run_current_measurement" in text, (
        "a Metal gate must run through metal_gate_runner so its artifact records "
        "the sources the PROCESS imported")
    assert "gate_provenance" in text


def test_the_gate_carries_the_moved_coefficient_observability_leg():
    """The family's one novel line is byte-invisible on an ordinary fixture, so a
    gate without this leg would report its mutation as uncaught and leave a reader to
    guess whether the line or the fixture was at fault."""
    text = GATE.read_text(encoding="utf-8")
    assert "leg_moved_coefficient" in text
    assert "MOVED_COEFFICIENT_EDIT" in text
    assert "prediction_holds" in text
