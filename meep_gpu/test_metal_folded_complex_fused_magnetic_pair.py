"""The folded COMPLEX fused magnetic Metal kernel, as claims a laptop can check without a GPU.

WHAT THIS SUITE OWNS AND WHAT IT DELIBERATELY DOES NOT. The BYTES are the device
gate's — ``parity/meep_gpu/gate_metal_folded_complex_fused_magnetic_pair.py`` steps
two engines side by side for twelve complete steps over eleven configurations and
compares uint32 words, and no assertion here duplicates that. What lives here is
everything true about the family WITHOUT a device: what the emitted source says, which
configurations the predicate refuses and by what name, that the arm is registered
UNWIRED, and that the plan binds the lattice each half is supposed to take.

THE FIVE CLAIMS THIS SUITE MAKES THAT THE GATE CANNOT.

1. That the product is invisible to ``plan_step`` — a property of the TABLE, and the
   whole reason this family may land beside the wired tree without touching it.
2. That the near-fill geometry is the B one and NOT the D one: a component is a ghost
   destination on its OWN axis alone. That is the single most likely slip in porting
   this product from :mod:`~meep_gpu.metal_kernels.folded_fused_pair`.
3. That the imaged ghost reads its coefficient pair at stored index 0 rather than
   reusing the source thread's. That is the one line the sibling does not carry, and
   on ordinary fixtures it is BYTE-INVISIBLE (the gate's ``moved_coefficient`` leg
   measures exactly that), so a source-level assertion is what keeps it honest.
4. That ALL THREE pieces are the certified emitters' own text, checked by re-running
   those emitters — the construction the module claims, made falsifiable. The third
   piece, the ghost's complex parity product, is the one no sibling family has.
5. That the parity is a RUNTIME WORD here and a SOURCE SPECIALISATION on the real
   fold. Two grids differing only in a mirror parity must compile to the SAME kernel
   on this family and to DIFFERENT kernels on that one — the inversion the module
   docstring states, asserted rather than left to a reader.
"""

from __future__ import annotations

import pathlib

import pytest

PACKAGE_DIR = pathlib.Path(__file__).resolve().parent
REPO = PACKAGE_DIR.parent
GATE = REPO / "parity" / "meep_gpu" / "gate_metal_folded_complex_fused_magnetic_pair.py"

from meep_gpu.metal_kernels import (  # noqa: E402
    arms, complex_fused_magnetic_pair as unfolded, folded_complex,
    folded_complex_fused_magnetic_pair as family,
    folded_fused_magnetic_pair as real_fold, registry, shaders,
)
from meep_gpu.triton_kernels.symmetry import (  # noqa: E402
    CODE_METALLIC, CODE_MIRROR_METALLIC, CODE_MIRROR_PERIODIC, CODE_PERIODIC,
    TARGET_IYEE,
)

#: The expansion arm every source in this suite is emitted under. A LITERAL rather
#: than a probe read: these tests are about what the emitter DECIDES, not about which
#: arm this host's NumPy takes, and reading a probe here would make the suite skip
#: silently on a machine with no artifact. The gate's ``expansion`` leg is where the
#: probe binding is measured.
EXPANSION = "FMA_V1"

#: Two folded axes with MIXED parities, one wall, no phased axis — the smallest
#: configuration that emits every kind of line the family can emit except a rotation.
CODES = (CODE_MIRROR_METALLIC, CODE_MIRROR_METALLIC, CODE_METALLIC)
PHASED = (0, 0, 0)
WALLS = (False, False, True)


def source(codes=CODES, phased=PHASED, walls=WALLS, expansion=EXPANSION,
           contract=shaders.CONTRACT_OFF) -> str:
    return family.folded_complex_fused_magnetic_pair_source(
        codes, phased, walls, expansion, contract)


def _mask_lines(text: str) -> str:
    """The body's TOP-PLANE CURL MASK lines, as text — empty when it carries none.

    Read off the emitted source rather than spelled, so this cannot agree with a
    kernel that spelled the mask wrongly; ``symmetry.folded_top_plane_mask`` is the
    emitter and the module splices its output.
    """
    return "\n".join(line for line in text.splitlines()
                     if line.strip().startswith("curl") and " = last_" in line)


# ---------------------------------------------------------------------------
# 1. The source is a LIFT, not a transcription — so the lift is checkable
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("codes,phased", (
    ((CODE_MIRROR_METALLIC, CODE_PERIODIC, CODE_PERIODIC), (0, 0, 0)),
    ((CODE_MIRROR_METALLIC, CODE_PERIODIC, CODE_PERIODIC), (0, 1, 1)),
    ((CODE_MIRROR_METALLIC, CODE_MIRROR_METALLIC, CODE_METALLIC), (0, 0, 0)),
    ((CODE_MIRROR_METALLIC, CODE_MIRROR_METALLIC, CODE_MIRROR_METALLIC), (0, 0, 0)),
))
def test_the_curl_head_is_the_certified_folded_bloch_curl_bodys_own_bytes(codes, phased):
    """The lifted head is a VERBATIM PREFIX of the certified folded Bloch curl body.

    This family does not retype the curl half; it cuts the certified emitter's own
    output at the split-field recurrence. A prefix check is the strongest statement
    available without re-deriving the emitter, and it covers the ghost gather, the
    per-axis Bloch rotation and the B-diagonal cell-0 ownership mask in one.

    The PHASED row matters on its own: an unphased axis emits NO multiply at all
    rather than one against ``1+0j``, so a lift that dropped the rotation block would
    still be a prefix on the ``(0, 0, 0)`` rows alone.
    """
    head = family.certified_curl_head(codes, phased, EXPANSION)
    certified = folded_complex.folded_bloch_curl_source(
        codes, family.BACKWARD, phased, EXPANSION)
    body = certified.split("uint idx [[thread_position_in_grid]])\n{\n", 1)[1]
    assert head.strip()
    assert body.startswith(head), (
        "the lifted curl head is no longer the certified folded Bloch curl body's own "
        "opening bytes; this family SPLICES that text and a divergence here means the "
        "fused kernel is quietly not the certified arithmetic")


def test_the_lifted_head_stops_before_the_recurrence():
    """The cut point is the split-field block, and nothing below it is spliced.

    The whole restructure lives below that line: the recurrence is re-emitted per
    component inside an ownership carve-out. A head that carried the recurrence would
    step every cell twice.
    """
    head = family.certified_curl_head(CODES, PHASED, EXPANSION)
    assert "--- split-field recurrence" not in head
    assert "u0[ii] =" not in head and "f0[ii] =" not in head


@pytest.mark.parametrize("contract", shaders.CONTRACT_MODES)
def test_the_constitutive_statements_reproduce_the_certified_complex_h_body(contract):
    """The parameterised seven statements ARE ``bloch_constitutive_source('H')``'s.

    This family re-emits the constitutive arithmetic twice per folded component — at
    the owned cell and at the imaged ghost — so it cannot splice that body wholesale.
    The measurement that keeps the parameterisation honest is rendering the template
    with the CERTIFIED spellings and requiring the result to be that body's own
    statements, and it must hold in BOTH contraction modes.
    """
    record = family.constitutive_transcription(EXPANSION, contract)
    assert record["identical"], (record["emitted"], record["certified"])
    assert [len(block) for block in record["certified"]] == [7, 7, 7]


def test_the_constitutive_multiply_is_the_coefficient_left_complex_form():
    """The accumulation is ``c_mul_coefficient_left``, not a real multiply.

    THE ONE ARITHMETIC DIFFERENCE from the real fold's identical-shaped template, and
    it is a different rounding rather than a spelling: ``stepping``'s complex path
    spells the dsigw products as the zero-imaginary complex form with the COEFFICIENT
    ON THE LEFT (S:2086-2087, :2093-2095). A template that carried the real twin's
    ``acc + kp * src`` would compile — ``float2 * float`` is legal Metal — and be a
    different number wherever a cross term is not exactly zero.
    """
    text = source()
    assert "c_mul_coefficient_left(kp_0, o0_src)" in text
    assert "c_mul_coefficient_left(km_0, o0_prev)" in text
    assert "+ kp_0 * o0_src" not in text and "- km_0 * o0_prev" not in text


@pytest.mark.parametrize("axis", (0, 1, 2))
def test_the_ghost_write_is_the_certified_folded_complex_fills_own_line(axis):
    """The third lift: the ghost's parity product is the certified FILL's own call.

    Neither curl nor constitutive, and without this check it would be the one
    hand-typed complex multiply in the family. The comparison is made under a STATED
    rename — the coefficient register, the source operand and the destination index —
    because the lifted curl head already owns the name ``c``.
    """
    record = family.near_fill_transcription(axis, EXPANSION)
    assert record["identical"], record["rows"]
    for row in record["rows"]:
        assert "c_mul(" in row["emitted"], row
        # THE COEFFICIENT IS ON THE LEFT, which is the array path's `phase * plane`.
        # The destination index carries the transcription's own SYNTHETIC tag — a
        # placeholder that exists only inside the comparison, so it is read off the
        # record rather than spelled. The emitter's real tags name the ghost's axes
        # (``g0_n``, ``g0_j``, ``g0_jn``), and pinning one of those here would be
        # pinning the emitter's naming instead of the certified line's SHAPE.
        prefix, call = row["emitted"].split(" = ", 1)
        tag = f"g{row['target']}_t"
        if row["role"] == "last":
            # THE LAST APPLICATION WRITES THE GHOST CELL, which is the certified fill's
            # one-line shape exactly.
            assert prefix == f"f{row['target']}[{tag}_i]", prefix
        else:
            # AN EARLIER ONE WRITES THE CARRY REGISTER, because the array path's
            # corresponding pass writes a memory cell the next pass then reads. Only
            # the FIRST link declares it.
            assert prefix == (("float2 " if row["role"] == "earlier" else "")
                              + f"{tag}_v"), (row["role"], prefix)
        assert call.startswith(f"c_mul(mp{axis},"), call

    # THE THREE ROLES ARE ALL PRESENT, or the loop above measured one shape and
    # reported on three.
    assert {row["role"] for row in record["rows"]} == {"last", "earlier", "continued"}


@pytest.mark.parametrize("axis", (0, 1, 2))
def test_the_far_ghost_write_is_the_certified_FAR_fills_own_line(axis):
    """The far pass's twin of the check above — and it is NOT the near line moved.

    THE FOURTH LIFT, carried since 2026-08-21. The certified far fill changes the
    DESTINATION row as well as the source (``last`` imaged from ``reflect_row``, not
    ``0`` imaged from ``2``) and it takes the OTHER of
    ``mirror_parity_coefficients``' two words, because ``mirror_parity`` is
    ``phase * (1 - 2*iyee)`` and a far destination has Yee shift 1 there. Asserting
    the register NAME is what separates the two: a carry that reused ``mp`` would be a
    plane of wrong signs and would still be a valid complex multiply.
    """
    record = family.far_fill_transcription(axis, EXPANSION)
    assert record["identical"], record["rows"]
    assert record["rows"], record
    for row in record["rows"]:
        call = row["emitted"].split(" = ", 1)[1]
        assert call.startswith(f"c_mul(fp{axis},"), call
        assert f"c_mul(mp{axis}," not in row["emitted"], row

    # AND THE TWO PASSES TOUCH DISJOINT COMPONENTS ON ONE AXIS. A B component's Yee
    # shift on a given axis is 0 or 1, never both, so no component is a near and a far
    # destination along the SAME axis — which is why the chain's entries are distinct
    # axes and `parity_chain` can order them at all.
    near = {row["target"] for row in family.near_fill_transcription(
        axis, EXPANSION)["rows"]}
    far = {row["target"] for row in record["rows"]}
    assert near and far and not (near & far), (near, far)


def test_an_empty_near_pass_is_refused_by_the_certified_emitter_not_measured_empty(
        monkeypatch):
    """A fill comparison over an empty row set reports as a pass; something must raise.

    THIS TEST DRIVES THE REAL PATH rather than restating the condition. Every axis
    images exactly one B component under the shipped Yee table, so the empty case is
    unreachable through it; the way to reach it is to hand the family a target triple
    whose Yee shift on the axis is 1 everywhere, which is what the monkeypatch does.

    WHAT IT MEASURES, and the answer is not the one the checker's own guard expects:
    :func:`.folded_complex.folded_mirror_fill_complex_source` refuses FIRST, by name —
    "an empty fill launch is a no-op that a before/after comparison reports as a pass"
    — so :func:`near_fill_transcription`'s own empty-row raise is a SECOND line of
    defence behind it. Both are kept; this pins which one fires, so a reader does not
    have to guess and a future change that removed the emitter's guard would surface
    here as a changed message rather than as silence.
    """
    from meep_gpu.metal_kernels import folded_complex as module

    assert all(TARGET_IYEE[name][0] == 1 for name in ("By", "Bz", "Dx")), (
        "the substitute triple no longer has shift 1 on axis 0; this test would "
        "then exercise a non-empty pass and measure nothing")
    monkeypatch.setitem(module.GHOST_FILL_FAMILIES, "B",
                        {"targets": ("By", "Bz", "Dx")})
    with pytest.raises(ValueError, match="an empty fill launch is a no-op"):
        family.near_fill_transcription(0, EXPANSION)


# ---------------------------------------------------------------------------
# 2. The B geometry, at the table level
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target,name", ((0, "Bx"), (1, "By"), (2, "Bz")))
def test_a_component_is_a_ghost_destination_on_its_own_axis_alone(target, name):
    """``iyee[Bm][a] == 0`` iff ``a == m`` — the B family's whole fill geometry.

    THE MOST LIKELY SLIP IN THE PORT. On the D family the near fill touches the two
    axes that are NOT the component's own; getting it backwards here would image the
    wrong plane on every folded run and would still compile, still run, and still look
    converged.
    """
    codes = (CODE_MIRROR_METALLIC,) * 3
    assert family.near_fill_axes(codes, target) == (target,)
    assert TARGET_IYEE[name][target] == 0
    assert all(TARGET_IYEE[name][axis] == 1 for axis in range(3) if axis != target)


def test_only_folded_axes_carry_a_ghost():
    """A PERIODIC or METALLIC axis is not a fill destination, whatever its shift."""
    assert family.near_fill_axes((CODE_PERIODIC, CODE_METALLIC, CODE_PERIODIC), 0) == ()
    assert family.near_fill_axes(
        (CODE_MIRROR_METALLIC, CODE_METALLIC, CODE_PERIODIC), 1) == ()
    # MIRROR_PERIODIC is matched by the axis reader even though the source builder
    # refuses it first, so a future far-fill carry changes one function.
    assert family.near_fill_axes(
        (CODE_MIRROR_PERIODIC, CODE_PERIODIC, CODE_PERIODIC), 0) == (0,)


def test_the_imaged_ghost_reads_its_own_coefficient_pair_at_index_zero():
    """The one line the electric sibling does not have, asserted at the source level.

    ``update_H`` indexes ``kps``/``kms`` on the component's own axis, which is the axis
    the fill images along — so the fill's source (stored 2) and destination (stored 0)
    take DIFFERENT entries. Reusing the source thread's pair applies the absorber
    profile of stored cell 2 to stored cell 0: smooth, converged and wrong inside the
    PML, invisible outside it. On the shared fixture's 2-cell layer the two entries are
    the SAME WORD, so no byte gate on that fixture can see the defect at all; this
    assertion and the gate's ``moved_coefficient`` leg are what keep it honest.
    """
    text = source()
    # ``_n`` is the NEAR carry's tag since the far carry landed: a ghost is named for
    # the axes it images along, and the reload belongs to the NEAR one alone — a far
    # destination shares the indexed axis with its source thread and must NOT reload.
    assert "float g0_n_kp = kp0[0], g0_n_km = km0[0];" in text
    assert "float g1_n_kp = kp1[0], g1_n_km = km1[0];" in text
    # And the OWNED cell still takes the thread's own entry.
    assert "c_mul_coefficient_left(kp_0, o0_src)" in text
    assert "c_mul_coefficient_left(g0_n_kp, g0_n_src)" in text

    # THE FAR SIDE IS THE OPPOSITE CLAIM AND IT IS ASSERTED HERE, not left to the
    # gate: a far ghost sits at the SAME coefficient index as the thread that owns it,
    # so reloading at 0 there would apply the mirror plane's absorber profile to the
    # top plane. The gate's `far_ghost_reloads_the_near_coefficient` mutation is the
    # byte-level twin of this line.
    periodic = source(codes=(CODE_MIRROR_PERIODIC, CODE_MIRROR_PERIODIC,
                             CODE_METALLIC), walls=(False, False, True))
    assert "c_mul_coefficient_left(kp_0, g0_j_src)" in periodic
    assert "float g0_j_kp = kp0[0]" not in periodic


def test_the_ghost_carve_out_makes_the_source_thread_the_owner():
    """``if (!(m == 0))`` on the destination and ``if (m == 2)`` on the source.

    The restructure exists to remove a cross-thread read: the cell the fill writes is
    OWNED BY THE THREAD THAT COMPUTES ITS SOURCE, so no thread forms a displacement
    from a word another thread is writing in the same launch.

    THE CARVE-OUT GREW A SECOND DISJUNCT WITH THE FAR CARRY, and that is asserted
    below rather than left to the gate: on a folded PERIODIC axis the TOP plane is a
    ghost destination too, so a carve-out that still named only cell 0 would let the
    top-plane thread store its own displacement over the image the source thread
    wrote — two unordered writes to one word.
    """
    text = source()
    assert "    if (!(i == 0)) {" in text and "        if (i == 2) {" in text
    assert "    if (!(j == 0)) {" in text and "        if (j == 2) {" in text
    assert "int g0_n_i = ii - 2 * nyz;" in text
    assert "int g1_n_i = ii - 2 * nzi;" in text

    periodic = source(codes=(CODE_MIRROR_PERIODIC, CODE_MIRROR_PERIODIC,
                             CODE_METALLIC), walls=(False, False, True))
    assert "    if (!(i == 0 || last_y)) {" in periodic, periodic
    assert "    if (!(j == 0 || last_x)) {" in periodic, periodic
    # Component 2's own axis is unfolded here, so it is a FAR destination on both
    # folded axes and a NEAR destination on none — its carve-out names no cell 0.
    assert "    if (!(last_x || last_y)) {" in periodic, periodic


def test_the_wall_clear_is_the_b_diagonal_and_writes_a_complex_zero():
    """Bx on an x wall, By on y, Bz on z — and BOTH planes.

    The array path assigns a COMPLEX zero (S:1896, :1902). A real ``0.0f`` would not
    even compile against a ``float2``, but the plane-wise question it stands for is
    live and the gate carries a mutation that clears only the real plane.
    """
    text = source(walls=(False, False, True))
    assert "v2 = at_z ? float2(0.0f, 0.0f) : v2;" in text
    assert "v0 = at_x ?" not in text and "v1 = at_y ?" not in text
    assert "0.0f : v2" not in text


# ---------------------------------------------------------------------------
# 3. The parity is a RUNTIME WORD — the inversion from the real fold
# ---------------------------------------------------------------------------

def test_two_parities_compile_to_the_same_source_here_and_differ_on_the_real_fold():
    """THE INVERSION, asserted rather than described.

    :mod:`.folded_fused_magnetic_pair` bakes the parity into the SOURCE, because
    ``symmetry._parity_spelling`` measured that a runtime float multiply flushes every
    subnormal on this backend at both signs. This family passes it as a host-rounded
    complex word, because the certified folded complex fill does — the parity is
    already inside a ``c_mul`` that flushes anyway, and passing it keeps the word
    host-rounded rather than synthesised in-kernel.

    The consequence is that ``phases`` is NOT a parameter of this family's source
    builder at all, which is what this test pins.
    """
    even = source()
    odd = source()
    assert even == odd
    real_even = real_fold.folded_fused_magnetic_pair_source(
        CODES, (1, 1, 0), WALLS)
    real_odd = real_fold.folded_fused_magnetic_pair_source(
        CODES, (1, -1, 0), WALLS)
    assert real_even != real_odd, (
        "the REAL fold's parity is a source specialisation; if that stopped being "
        "true, this family's opposite choice would need re-deriving rather than "
        "re-stating")


@pytest.mark.parametrize("phase,expected", ((1, (1.0, 0.0)), (-1, (-1.0, 0.0))))
def test_the_parity_word_is_host_rounded_and_has_a_bitwise_zero_imaginary(phase, expected):
    """``complex64(+-1)`` is exactly ``+-1.0`` with an imaginary word of ``+0.0``.

    Everything this family says about the ghost's complex product resting on the two
    expansion arms agreeing depends on that pattern: with ``c_im`` an exact ``+0.0``
    the fused arm's extra product is EXACT, so the two arms produce identical bytes
    there. The words come from :func:`.folded_complex.mirror_parity_coefficients`, the
    one place in the package where a mirror parity becomes a complex word.
    """
    import numpy

    codes = (CODE_MIRROR_METALLIC, CODE_PERIODIC, CODE_PERIODIC)
    words = family.parity_words(codes, (phase, 0, 0))
    assert words[0] == expected
    assert numpy.float32(words[0][1]).tobytes() == numpy.float32(0.0).tobytes()
    # An unfolded axis carries a word no emitted line reads.
    assert words[1] == (0.0, 0.0) and words[2] == (0.0, 0.0)


def test_an_undeclarable_parity_is_refused_rather_than_defaulted():
    """``mirror_parity_coefficients`` raises on anything but +-1."""
    with pytest.raises(ValueError, match=r"declared phase is \+1 or -1"):
        folded_complex.mirror_parity_coefficients(0)


# ---------------------------------------------------------------------------
# 4. Scope: what the source builder refuses, by name
# ---------------------------------------------------------------------------

def test_a_folded_periodic_axis_is_ADMITTED_and_its_far_fill_CARRIED():
    """``fill_folded_far_ghosts_B`` runs inside this seam and this carry now TAKES it.

    This test pinned the refusal until 2026-08-21. What it pins now is that the
    retirement moved all three things together — the emitter, the declaration and the
    hazard the refusal was also discharging. The last is the one a retirement gets
    wrong: refusing ``CODE_MIRROR_PERIODIC`` outright ALSO retired the folded curl's
    top-plane mask, so a carry that took the fill and left the mask would put a plane
    of stepped cells where the array path has unowned allocation slots.
    """
    periodic = (CODE_MIRROR_PERIODIC, CODE_MIRROR_PERIODIC, CODE_METALLIC)
    text = source(codes=periodic, walls=(False, False, True))

    # 1. THE DECLARATION, in the driver's own order (driver.py:3281-3289).
    assert family.REPLACES == ("step_B", "fill_B", "zero_metal_B",
                               "fill_folded_far_ghosts_B", "update_H")

    # 2. THE FAR FILL FIRES, on the axes the module's own table names and no others.
    far = tuple(family.far_fill_axes(periodic, target) for target in range(3))
    assert far == ((1,), (0,), (0, 1)), far
    assert "_fill_folded_far_ghosts" in text
    assert "== reflect_x" in text and "== reflect_y" in text

    # 3. THE TOP-PLANE MASK CAME WITH IT — the second obligation. Both directions:
    # present on the periodic body, absent on the metallic one where the top plane is
    # an owned cell and masking it would delete a real value.
    assert "curl0 = last_y ? float2(0.0f, 0.0f) : curl0;" in text
    assert "last_" not in _mask_lines(source())
    assert _mask_lines(text), text


def test_an_unfolded_grid_is_refused_and_names_the_sibling():
    """An unfolded complex grid belongs to :mod:`.complex_fused_magnetic_pair`."""
    with pytest.raises(ValueError, match="complex_fused_magnetic_pair"):
        source(codes=(CODE_PERIODIC, CODE_METALLIC, CODE_PERIODIC))


def test_a_folded_axis_that_is_also_walled_is_refused():
    """``_zero_metal`` skips a folded axis; the carry relies on the two being disjoint."""
    with pytest.raises(ValueError, match="both folded and walled"):
        source(walls=(True, False, False))


def test_the_top_plane_flags_are_voided_only_where_the_mask_is_empty():
    """The flags are declared unconditionally; whether they are READ is per-config.

    On a grid with no folded PERIODIC axis ``folded_top_plane_mask`` emits nothing and
    the three ``last_*`` flags must be explicitly voided — unread is a warning on some
    toolchains and, worse, reads as an omission. On a folded PERIODIC grid they are
    READ, by the mask and by the far carve-out, and a body that still voided them
    without reading them would be the half-done retirement this suite exists to catch.
    """
    metallic = source()
    assert "(void)last_x; (void)last_y; (void)last_z;" in metallic
    assert "no folded PERIODIC axis" in metallic
    assert _mask_lines(metallic) == ""

    periodic = source(codes=(CODE_MIRROR_PERIODIC, CODE_MIRROR_PERIODIC,
                             CODE_METALLIC), walls=(False, False, True))
    assert _mask_lines(periodic) != ""
    assert "no folded PERIODIC axis" not in periodic


# ---------------------------------------------------------------------------
# 5. The signature, the registration and the wiring
# ---------------------------------------------------------------------------

def test_the_signature_is_the_unfolded_complex_pairs_exactly():
    """27 pointers plus one packed struct; the fold adds no BINDING.

    The three mirror-parity words ride INSIDE the packed struct beside the three Bloch
    phases, so the fold costs nothing at the binding ceiling. The two constants are
    IMPORTED from the unfolded family rather than re-spelled, and this pins that they
    still describe this kernel.
    """
    text = source()
    assert family.PACKED_BINDINGS == unfolded.PACKED_BINDINGS == 28
    assert family.SEPARATE_SCALAR_BINDINGS == unfolded.SEPARATE_SCALAR_BINDINGS == 35
    assert "[[buffer(27)]]" in text and "[[buffer(28)]]" not in text
    assert "constant Params&     prm     [[buffer(27)]]" in text


def test_the_params_record_is_one_hundred_and_four_bytes_with_the_complex_first():
    """Metal aligns ``float2`` to 8 bytes; NINE of them, then four uints, a float, and
    the three int reflect rows: 72 + 20 + 12 = 104, already 8-byte aligned so Metal
    adds no tail padding.

    IT WAS 72 BEFORE THE FAR CARRY. The three FAR parity words and the three runtime
    reflect rows are what grew it, and they are what let the carry cost NO BINDING —
    so the itemsize and the binding count are the two halves of one claim and both are
    pinned here. Stated explicitly in the dtype so the record cannot drift into
    agreeing with Metal only by accident. The gate's ``binding_ceiling`` leg LAUNCHES
    this record and reads every field back, which is the measurement; this is the
    shape.
    """
    dtype = family.params_record_dtype()
    assert dtype.itemsize == family.PARAMS_ITEMSIZE == 104
    assert dtype.names == ("px", "py", "pz", "m0", "m1", "m2", "d0", "d1", "d2",
                           "nx", "ny", "nz", "n_elem", "dtdx", "rx", "ry", "rz")
    assert [dtype.fields[name][1] for name in dtype.names] == [
        0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 76, 80, 84, 88, 92, 96, 100]
    # THE COMPLEX MEMBERS COME FIRST AND THE SCALARS LAST — the packing rule the
    # unfolded sibling measured on this toolchain, restated as a boundary rather than
    # as a comment: no scalar may sit before the last float2.
    assert dtype.fields["d2"][1] < dtype.fields["nx"][1]

    # AND THE CARRY IS FREE AT THE CEILING. The reflect rows and far parities ride in
    # this record, so the shipped signature is the unfolded pair's exactly.
    assert family.PACKED_BINDINGS == unfolded.PACKED_BINDINGS == 28
    assert family.PACKED_BINDINGS < 31


def test_the_family_replaces_five_driver_passes_in_driver_order():
    """FIVE since 2026-08-21: ``fill_folded_far_ghosts_B`` is CARRIED, not refused.

    ``REPLACES`` is what the whole-step gate reads as ``covered_passes``. A carry whose
    declaration stayed at four claims one launch replaces four passes while the bytes
    replace five — the driver would then run the far fill again on the host, over the
    plane the kernel already imaged. The gate's ``refusal`` leg reads
    ``far_pass_is_in_replaces`` off this tuple and failed on exactly that.
    """
    assert family.REPLACES == ("step_B", "fill_B", "zero_metal_B",
                               "fill_folded_far_ghosts_B", "update_H")
    assert family.SLOT == "step_B"
    assert family.CURL_SUB_STEP == "step_B" and family.CONSTITUTIVE_SIDE == "H"
    assert family.BACKWARD is False
    assert family.NEAR_SOURCE_INDEX == 2


def test_the_arm_is_registered_and_unwired():
    """THE CLAIM THAT LETS THIS LAND BESIDE THE WIRED TREE.

    ``plan_step`` assigns at most one arm per slot and this product spans four, so
    there is no slot it could claim without a composition rule nothing has measured.
    Registered unwired keeps it ENUMERABLE for the disjointness sweep while
    ``arms.arms_for`` skips it.
    """
    registered = {arm.family: arm for arm in arms.registered()}
    assert family.FAMILY in registered, (
        "the family is not in the arm table; registry.py must import it")
    row = registered[family.FAMILY]
    assert row.wired is False
    assert row.slot == family.SLOT == "step_B"
    # ``arms_for`` is the gate between the table and the composer: an unwired row
    # stays ENUMERABLE — which is what lets a disjointness sweep see it — while never
    # reaching ``_select_slot``.
    context = arms.StepContext(object(), None, None, (), sources=())
    assert row.label not in [arm.label for arm in arms.arms_for(family.SLOT, context)]
    assert row.label in [spec.label for spec in arms.registered(family.SLOT)]


def test_the_family_is_named_in_the_registry_module_list():
    """A family in the tree but not in ``FAMILY_MODULES`` is invisible to the sweep."""
    assert "folded_complex_fused_magnetic_pair" in registry.FAMILY_MODULES


def test_every_other_fused_pair_is_still_unwired():
    """No existing arm's selection changed. The whole landing condition, restated."""
    registered = {arm.family: arm for arm in arms.registered()}
    for name in ("fused_magnetic_pair", "complex_fused_magnetic_pair",
                 "folded_fused_magnetic_pair", "folded_fused_pair",
                 "fused_dispersive_pair"):
        assert registered[name].wired is False


def test_the_gate_exists_and_names_the_family():
    """A product whose gate is absent is an unmeasured claim."""
    assert GATE.is_file()
    text = GATE.read_text(encoding="utf-8")
    assert "folded_complex_fused_magnetic_pair" in text
    # The dead-branch guard, and the digest seed rule, both pinned by name: a gate
    # that lost either would still pass its own legs.
    assert "PHASE_MUTATION_CASE" in text
    assert "hashlib.sha256" in text and "hash()" in text
