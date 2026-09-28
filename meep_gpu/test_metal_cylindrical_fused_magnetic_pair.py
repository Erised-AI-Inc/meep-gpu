"""The CYLINDRICAL COMPLEX fused magnetic Metal pair, as claims a laptop can check.

WHAT THIS SUITE OWNS AND WHAT IT DELIBERATELY DOES NOT. The BYTES are the device
gate's — ``parity/meep_gpu/gate_metal_cylindrical_fused_magnetic_pair.py`` steps two
engines side by side for twelve complete steps and compares uint32 words, and no
assertion here duplicates that. What lives here is everything true about the family
WITHOUT a device: what the emitted source says, which configurations the predicate
refuses and by what name, that the arm is registered UNWIRED, and the two
BYTE-INVISIBLE CHOICES a byte gate structurally cannot hold.

THE FOUR CLAIMS THIS SUITE MAKES THAT THE GATE CANNOT.

1. **The product is invisible to ``plan_step``** — a property of the TABLE, and the
   whole reason this family may land beside the wired tree without touching it.
2. **The wall table on a Dcyl grid can only ever name z**, so the emitted mask is a
   single line and the r and phi clears are never emitted. That is what makes the
   gate's wall mutations reachable and its r/phi ones absent-by-design rather than
   dead; a byte gate cannot assert the absence of a line.
3. **The binding budget is 31 of 31 with ZERO headroom**, by arithmetic on the
   counts. Whether the refuted signatures actually fail to COMPILE is a device
   measurement and belongs to the gate; that the counts this suite and that leg
   talk about are the same counts belongs here.
4. **The two byte-invisible choices are pinned by SOURCE TEXT.** The i*m/r operand
   order and the ``t0`` grouping are both what the array path computes and both
   produce identical bytes through this kernel — measured, 0 differing words — so a
   byte gate CANNOT hold them and saying it does would be the artifact overclaim the
   BFAST tranche names. They are held here instead.

THE ARM IS NOT ASSERTED HERE. Which complex-multiply expansion a platform takes is a
measured fact carried in a probe artifact, and a unit test that pinned ``FMA_V1``
would be a guess wearing a test's clothes. What IS asserted is that the family binds
through the CYLINDRICAL family's seven-pattern probe rather than the plain complex
family's five-pattern one, and that a five-pattern record licenses nothing here.
"""

from __future__ import annotations

import pathlib
import re

import pytest

PACKAGE_DIR = pathlib.Path(__file__).resolve().parent
REPO = PACKAGE_DIR.parent
GATE = (REPO / "parity" / "meep_gpu"
        / "gate_metal_cylindrical_fused_magnetic_pair.py")

family = pytest.importorskip(
    "meep_gpu.metal_kernels.cylindrical_fused_magnetic_pair")
cylindrical_complex = pytest.importorskip(
    "meep_gpu.metal_kernels.cylindrical_complex")
templates = pytest.importorskip("meep_gpu.metal_kernels.templates")
device = pytest.importorskip("meep_gpu.metal_kernels.device")
twin = pytest.importorskip(
    "meep_gpu.metal_kernels.complex_fused_magnetic_pair")

#: The specialisation most assertions read: metallic z, |m| = 1, the z wall live.
#: Thirteen of the sixteen corpus rows terminate z METALLIC, so this is the family's
#: ordinary configuration rather than a corner.
WALLED = dict(bcz=templates.METALLIC, m_arm=cylindrical_complex.M_ONE,
              zero_metal=(False, False, True))


def source(**overrides):
    keywords = dict(WALLED)
    keywords.update(overrides)
    return family.cylindrical_fused_magnetic_pair_source(
        keywords["bcz"], keywords["m_arm"], keywords["zero_metal"], "FMA_V1")


# ---------------------------------------------------------------------------
# The seam this product spans
# ---------------------------------------------------------------------------

def test_replaces_is_the_driver_order_and_is_declared_not_inferred():
    """``REPLACES`` is the driver's own pass list, in the driver's own order."""
    assert family.REPLACES == ("step_B", "zero_metal_B", "update_H")
    assert family.SLOT == "step_B" == family.CURL_SUB_STEP
    assert family.CONSTITUTIVE_SIDE == "H"


def test_the_arm_is_registered_unwired_and_plan_step_cannot_select_it():
    """THE WHOLE COMPOSITION STORY. This product spans three slots and ``plan_step``
    assigns at most one arm per slot, so it registers UNWIRED: enumerable by the
    disjointness sweep, invisible to ``arms.arms_for``, unreachable from a run."""
    from meep_gpu.metal_kernels import arms

    rows = [spec for spec in arms.registered() if spec.family == family.FAMILY]
    assert len(rows) == 1, rows
    assert rows[0].slot == family.SLOT
    assert rows[0].wired is False
    # ``arms_for`` filters on ``wired`` (arms.py:231-235), so an unwired row is
    # invisible to it whatever context it is handed — which is the property the
    # flag exists to give, asserted on the FILTER rather than on one context.
    assert all(spec.wired for spec in arms.registered(family.SLOT)
               if spec.family == family.FAMILY) is False
    assert family.FAMILY in {spec.family for spec in arms.registered(family.SLOT)}


def test_the_family_is_in_the_registry_module_list():
    """A family added to the tree but not to ``FAMILY_MODULES`` is invisible to
    ``plan_step``, which is a silent coverage loss rather than an error."""
    from meep_gpu.metal_kernels import registry

    assert "cylindrical_fused_magnetic_pair" in registry.FAMILY_MODULES


# ---------------------------------------------------------------------------
# The binding budget
# ---------------------------------------------------------------------------

def test_the_signature_binds_thirty_one_of_thirty_one_with_zero_headroom():
    """31 pointers-plus-struct against a ceiling of 31. Counted from the EMITTED
    TEXT, not from the constant, so the two cannot drift."""
    attributes = re.findall(r"\[\[buffer\((\d+)\)\]\]", source())
    assert len(attributes) == family.PACKED_BINDINGS == 31
    assert len(set(attributes)) == len(attributes), "a buffer index is reused"
    assert max(int(index) for index in attributes) == 30
    assert family.PACKED_BINDINGS == device.MAX_BUFFER_BINDINGS, (
        "this family sits AT the ceiling; a change to either number is a change to "
        "whether the product can be built at all")


def test_the_two_refuted_signatures_are_over_the_ceiling_by_arithmetic():
    """Whether they FAIL TO COMPILE is the gate's measurement; that they are over
    the ceiling at all is arithmetic and belongs here."""
    assert family.SEPARATE_SCALAR_BINDINGS == 39 > device.MAX_BUFFER_BINDINGS
    assert family.OVER_CEILING_BINDINGS == 32 > device.MAX_BUFFER_BINDINGS
    separate = re.findall(r"\[\[buffer\((\d+)\)\]\]",
                          family.refuted_separate_scalar_source())
    assert len(separate) == family.SEPARATE_SCALAR_BINDINGS
    over = re.findall(r"\[\[buffer\((\d+)\)\]\]",
                      family.refuted_thirty_second_binding())
    assert len(over) == family.OVER_CEILING_BINDINGS


def test_the_params_record_is_forty_bytes_with_the_float2_first():
    """Metal aligns ``float2`` to 8. With ``inc_b`` FIRST the natural offsets are
    Metal's and there is no internal padding — the complex twin measured what the
    other order costs, and it is a plausible complex number rather than garbage."""
    dtype = family.params_record_dtype()
    assert dtype.itemsize == family.PARAMS_ITEMSIZE == 40
    assert dtype.names[0] == "inc_b"
    assert dtype.names[-1] == "axis_coef"
    assert [dtype.fields[name][1] for name in dtype.names] == [
        0, 8, 12, 16, 20, 24, 28, 32, 36]
    struct = source().split("struct Params {", 1)[1].split("};", 1)[0]
    assert "float dtdx; float minus_dtdx; float axis_coef;" in struct


# ---------------------------------------------------------------------------
# The wall table
# ---------------------------------------------------------------------------

def test_only_the_z_wall_clear_is_ever_emitted():
    """A Dcyl grid cannot wall r or phi (``Grid`` builds ``metallic_axes`` as
    ``pair[1] == METALLIC and pair[0] != AXIS`` and r's pair is ``(AXIS, METALLIC)``),
    so the emitted mask is Bz's row and nothing else.

    THE ABSENCE IS THE POINT. A byte gate cannot assert that a line is missing, and
    a wall mutation armed on the r or phi row would rewrite text the emitter never
    emits and report UNCAUGHT while measuring nothing."""
    zero = templates.COMPLEX_ZERO
    text = source()
    assert f"v2 = at_z ? {zero} : v2;" in text
    assert f"v0 = at_x ? {zero} : v0;" not in text
    assert f"v1 = at_y ? {zero} : v1;" not in text


def test_a_periodic_z_emits_no_wall_clear_at_all():
    text = source(bcz=templates.PERIODIC, zero_metal=(False, False, False))
    assert "no walled axis clears a B component" in text
    assert "at_z ?" not in text.split("--- zero_metal_B")[1].split("f0[ii] = v0")[0]


@pytest.mark.parametrize("axis", family.FORBIDDEN_WALL_AXES)
def test_an_r_or_phi_wall_is_refused_by_the_emitter_by_name(axis):
    """Refused rather than emitted: the r = 0 clear is a row the per-|m| axis rules
    own, and writing a zero there is a value the array path never writes."""
    walls = [False, False, True]
    walls[axis] = True
    with pytest.raises(ValueError, match="AXIS"):
        source(zero_metal=tuple(walls))


def test_the_wall_clear_is_imported_from_the_twin_not_respelled():
    """ONE HOME FOR THE B-SIDE WALL TABLE. The diagonal (Bx on x, By on y, Bz on z)
    is the exact complement of the D side's, and a second spelling here is a second
    place for the two to drift."""
    text = pathlib.Path(family.__file__).read_text(encoding="utf-8")
    assert "complex_fused_magnetic_pair.zero_metal_mask(zero_metal)" in text
    assert "_ZERO_METAL_ROWS" not in text


# ---------------------------------------------------------------------------
# Transcription — both halves are the certified emitters' own bytes
# ---------------------------------------------------------------------------

def test_the_curl_half_is_the_certified_body_verbatim_around_the_wall_clear():
    text = source()
    curl = family.certified_cylindrical_curl_body(
        WALLED["bcz"], WALLED["m_arm"], "FMA_V1")
    head, tail = curl.split(family._CURL_STORE, 1)
    assert head in text
    assert (family._CURL_STORE + tail) in text


def test_the_constitutive_half_is_the_twins_lift_and_differs_only_in_the_seam():
    """``update_H`` carries NO cylindrical branch — measured by the cylindrical
    tranche at 480/480 rows, 0 differing words — so the constitutive half of this
    pair IS the complex twin's, lift and all."""
    text = source()
    certified = twin.certified_constitutive_body("FMA_V1").splitlines()
    spliced = [line for line in
               text.split("    // --- update_H (stepping.update_H", 1)[1].splitlines()
               if "// THE SEAM:" not in line]
    while spliced and (spliced[0].endswith("--")
                       or spliced[0].lstrip().startswith("//")):
        spliced.pop(0)
    while spliced and spliced[-1].strip() in ("", "}"):
        spliced.pop()
    assert len(certified) == len(spliced)
    changed = [(a, b) for a, b in zip(certified, spliced) if a != b]
    assert changed == [(f"    float2 src{t} = g{t}[ii];",
                        f"    float2 src{t} = v{t};") for t in range(3)]


def test_the_seam_replaces_the_reload_and_nothing_else():
    text = source()
    for target in range(3):
        assert f"float2 src{target} = v{target};" in text
        assert f"float2 src{target} = g{target}[ii];" not in text


# ---------------------------------------------------------------------------
# The byte-invisible choices — held HERE, because a byte gate cannot hold them
# ---------------------------------------------------------------------------

def test_the_imr_coefficient_stays_on_the_left():
    """``stepping._cylindrical_imr_term`` (:724) spells ``factor * partner_values``
    with the coefficient on the LEFT, and this kernel keeps that orientation.

    MEASURED BYTE-INVISIBLE, 2026-08-20: every entry of every i*m/r row this family
    binds has a real word of exactly ``+0.0`` (16/16 words ``0x00000000`` at
    m = +-1 and m = 3, on both targets), and with ``c_re = +-0.0`` the product
    ``c_re * z_re`` is exact — so ``c_mul(q, z)`` and ``c_mul(z, q)`` are the same
    single-rounding operation. The gate's ``signed_zero`` leg measures 0 differing
    words for the swap across twelve rows spanning both value classes and three
    lattice draws. So the choice is REAL and the byte gate cannot hold it; this
    assertion is what does."""
    text = source()
    assert "float2 m0 = c_mul(q0, c);" in text
    assert "float2 m2 = c_mul(q2, a);" in text
    assert "c_mul(c, q0)" not in text and "c_mul(a, q2)" not in text


def test_the_curl_grouping_keeps_the_invariant_axis_difference():
    """``stepping._curl_from_operands`` groups ``((A) + (B))`` and this kernel keeps
    it on all three terms.

    ``t0`` and ``t2`` each lead with the PHI SELF-DIFFERENCE, which is an exact
    ``+0.0`` on a one-cell axis, so flattening THEIR parens is byte-invisible
    (measured 2026-08-20: 0 differing words, against 528 for the identical edit on
    ``t1``, which the gate arms as a catch). Kept because it is what the array path
    computes, and pinned here because the gate cannot pin it."""
    text = source()
    assert "float2 t0 = ((c_y - c) + (b - b_z));" in text
    assert "float2 t1 = ((a_z - a) + (c - c_x));" in text
    assert "float2 t2 = ((b_x - b) + (a - a_y));" in text


def test_the_prefix_replaces_target_twos_curl_rather_than_adding_to_it():
    """``step_B`` :343-347 — Bz's WHOLE curl is the forward difference of the
    EXTENDED prefix: ONE subtract, ONE multiply. The four-operand grouping is a
    different float32 number per plane, and the gate catches the substitution
    (7,296 words)."""
    text = source()
    assert "curl2 = c_mul_coefficient_left(dtdx, pu - pd);" in text
    assert "float2 pu = pfx[ii + nyz];" in text


# ---------------------------------------------------------------------------
# The |m| arms compile different bodies
# ---------------------------------------------------------------------------

def test_the_three_m_arms_are_different_bodies():
    """``M_ONE`` emits an axis-row increment and no near-axis hold; ``M_MANY`` emits
    the hold and NO increment; ``M_ZERO`` (2026-09-04) emits neither, carries no
    i*m/r block at all, and clears ``Bx[r = 0]`` after the recurrence. That is why
    the gate's mutation set is split by case — a needle armed on the wrong arm
    rewrites text the kernel never emits."""
    zero = source(m_arm=cylindrical_complex.M_ZERO)
    one = source(m_arm=cylindrical_complex.M_ONE)
    many = source(m_arm=cylindrical_complex.M_MANY)
    assert "curl0 = at_x ? -inc : curl0;" in one
    assert "curl0 = at_x ? -inc : curl0;" not in many
    assert "curl0 = at_x ? -inc : curl0;" not in zero
    assert "bool near = (i < int(zrows));" in many
    assert "bool near = (i < int(zrows));" not in one
    assert "bool near = (i < int(zrows));" not in zero
    assert "float2 q0 = c0[i];" in one and "float2 q0 = c0[i];" in many
    assert "float2 q0 = c0[i];" not in zero
    assert "v0 = at_x ? float2(0.0f, 0.0f) : v0;" in zero
    assert "v0 = at_x ? float2(0.0f, 0.0f) : v0;" not in one
    assert "v0 = at_x ? float2(0.0f, 0.0f) : v0;" not in many
    # The seam consumes the CLEARED register: the clear precedes the store, and the
    # store precedes the constitutive half's first read of v0.
    assert (zero.index("v0 = at_x ? float2(0.0f, 0.0f) : v0;")
            < zero.index(family._CURL_STORE) < zero.index("float2 src0 = v0;"))


def test_the_enumeration_is_closed_at_six_sources():
    """``zero_rows`` is a RUNTIME UNIFORM, not a specialisation, which is what keeps
    this enumeration finite. The wall flag is a FUNCTION of ``bcz`` rather than a
    free axis, because ``zero_metal_axes`` cannot report a wall on a periodic axis.
    Four until 2026-09-04; the ``M_ZERO`` arm added two."""
    labels = family.enumerate_sources("FMA_V1")
    assert len(labels) == 6
    assert len(set(labels.values())) == 6, "two specialisations emit the same source"
    assert sum("/m0/" in label for label in labels) == 2
    for label in labels:
        assert "/zm1/" in label if "bcz1" in label else "/zm0/" in label


# ---------------------------------------------------------------------------
# Refusals, each by name
# ---------------------------------------------------------------------------

def _fixture(**keywords):
    matrix = pytest.importorskip("parity.meep_gpu.metal_composition_matrix") \
        if False else None
    import sys
    sys.path.insert(0, str(REPO / "parity" / "meep_gpu"))
    import metal_composition_matrix as mx  # noqa: PLC0415

    mx.prepare_environment()
    return mx.cylindrical(**keywords)


def test_an_undeclared_source_list_is_refused_ignorance_is_not_an_empty_set():
    fields, pml = _fixture(m=1)
    verdict = family.cylindrical_fused_magnetic_pair_coverage(fields, pml, None)
    assert not verdict.covered
    assert any("was not declared" in reason for reason in verdict.reasons)


def test_a_magnetic_source_is_refused_by_name_and_an_electric_one_is_not():
    """THE POLARITY IS THE WHOLE POINT. A magnetic source is injected BETWEEN the
    halves; an electric one is injected in the D/E seam and does not disqualify the
    pair — and on this family the electric case is every one of the sixteen corpus
    rows."""
    from meep_gpu.sources import ContinuousEnvelope, VolumeSource

    fields, pml = _fixture(m=1)

    def volume(component):
        return VolumeSource(grid=fields.grid, component=component,
                            center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                            envelope=ContinuousEnvelope(frequency=1.0))

    magnetic = family.cylindrical_fused_magnetic_pair_coverage(
        fields, pml, (volume("Hy"),))
    assert not magnetic.covered
    assert any("is magnetic" in reason and "driver.py:3283-3284" in reason
               for reason in magnetic.reasons)
    electric = family.cylindrical_fused_magnetic_pair_coverage(
        fields, pml, (volume("Ez"),))
    assert [reason for reason in electric.reasons
            if "source" in reason and "magnetic" in reason] == []


def test_m_zero_is_partitioned_by_storage_and_a_complex_m_zero_run_is_admitted():
    """A float32 m = 0 run is the cylindrical REAL product's row and is refused HERE
    by name on storage; a complex64 m = 0 run is THIS family's (the ``M_ZERO`` arm,
    2026-09-04) and is refused by the real product on the same clause the other way
    — so no slot is co-admitted and none falls between the two."""
    from meep_gpu.metal_kernels import cylindrical_real

    fields, pml = _fixture(m=0, complex_storage=False)
    verdict = family.cylindrical_fused_magnetic_pair_coverage(fields, pml, ())
    assert not verdict.covered
    assert any("force_complex_fields is not set" in reason for reason in verdict.reasons)

    fields, pml = _fixture(m=0, complex_storage=True)
    residency = device.Residency()
    verdict = family.cylindrical_fused_magnetic_pair_coverage(fields, pml, (),
                                                              residency)
    assert verdict.covered, verdict.reasons
    real = cylindrical_real.cylindrical_real_curl_coverage(fields, pml, "step_B",
                                                          residency)
    assert not real.covered
    assert any("force_complex_fields=True" in reason for reason in real.reasons)


def test_a_cartesian_grid_is_refused():
    import sys
    sys.path.insert(0, str(REPO / "parity" / "meep_gpu"))
    import metal_composition_matrix as mx  # noqa: PLC0415

    mx.prepare_environment()
    fields, pml = mx.cart()
    verdict = family.cylindrical_fused_magnetic_pair_coverage(fields, pml, ())
    assert not verdict.covered
    assert any("not cylindrical" in reason for reason in verdict.reasons)


def test_real_storage_is_refused():
    fields, pml = _fixture(m=1, complex_storage=False)
    verdict = family.cylindrical_fused_magnetic_pair_coverage(fields, pml, ())
    assert not verdict.covered


def test_the_predicate_inherits_both_halves_rather_than_restating_them():
    """A configuration either half refuses is refused here WITH THAT HALF'S REASONS,
    prefixed so a reader can tell which side said it."""
    fields, pml = _fixture(m=0, complex_storage=False)
    verdict = family.cylindrical_fused_magnetic_pair_coverage(fields, pml, ())
    assert any(reason.startswith("cylindrical curl half: ")
               for reason in verdict.reasons)
    assert any(reason.startswith("constitutive half: ")
               for reason in verdict.reasons)


# ---------------------------------------------------------------------------
# The expansion probe
# ---------------------------------------------------------------------------

def test_the_family_binds_through_the_cylindrical_seven_pattern_probe():
    """NOT the plain complex family's five-pattern artifact. This kernel performs
    SEVEN orientations — the base five plus the i*m/r row and the |m| = 1 scalar —
    and reading a record that measured five would licence two calls nothing
    classified."""
    text = pathlib.Path(family.__file__).read_text(encoding="utf-8")
    assert "cylindrical_complex.expansion_from_probe" in text
    assert "cylindrical_complex.load_expansion_probe" in text
    assert "complex_fields.load_expansion_probe" not in text
    assert len(cylindrical_complex.CYLINDRICAL_PROBE_PATTERNS) == 7


def test_a_missing_probe_refuses_rather_than_defaulting_to_an_arm():
    assert cylindrical_complex.expansion_from_probe(None) is None


# ---------------------------------------------------------------------------
# The gate exists and says what this suite says
# ---------------------------------------------------------------------------

def test_the_device_gate_exists_and_declares_the_same_binding_numbers():
    text = GATE.read_text(encoding="utf-8")
    assert "family.PACKED_BINDINGS" in text
    assert "family.OVER_CEILING_BINDINGS" in text
    assert "leg_signed_zero" in text
    assert "leg_needle_reachability" in text
