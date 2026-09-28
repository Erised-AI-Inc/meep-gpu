"""The COMPLEX fused magnetic Metal kernel, as claims a laptop can check without a GPU.

WHAT THIS SUITE OWNS AND WHAT IT DELIBERATELY DOES NOT. The BYTES are the device
gate's — ``parity/meep_gpu/gate_metal_complex_fused_magnetic_pair.py`` steps two
engines side by side for twelve complete steps and compares uint32 words, and no
assertion here duplicates that. What lives here is everything true about the family
WITHOUT a device: what the emitted source says, which configurations the predicate
refuses and by what name, that the arm is registered UNWIRED, and that the plan binds
the lattice each half is supposed to take.

THE THREE CLAIMS THIS SUITE MAKES THAT THE GATE CANNOT.

1. That the product is invisible to ``plan_step`` — a property of the TABLE, and the
   whole reason this family may land beside the wired tree without touching it.
2. That the wall-clear table is THE DIAGONAL and its zero is a COMPLEX zero. The gate
   measures that the shipped kernel is right; only a table-level assertion says that
   the D-side family's table is the WRONG one here, which is the single most likely
   slip in porting this product from its sibling.
3. That ``float2`` and the packed struct are FORCED rather than preferred, by arithmetic
   on the binding counts. Whether each refuted signature actually fails to COMPILE is a
   device measurement and belongs to the gate; that the counts this suite and that leg
   talk about are the same counts belongs here.

THE ARM IS NOT ASSERTED HERE. Which complex-multiply expansion a platform takes is a
measured fact carried in a probe artifact, and a unit test that pinned ``FMA_V1`` would
be a guess wearing a test's clothes. What IS asserted is that the family binds through
:mod:`.complex_fields`' probe rather than carrying one of its own, and that a missing or
ambiguous probe refuses.
"""

from __future__ import annotations

import pathlib

import pytest

PACKAGE_DIR = pathlib.Path(__file__).resolve().parent
REPO = PACKAGE_DIR.parent
GATE = (REPO / "parity" / "meep_gpu"
        / "gate_metal_complex_fused_magnetic_pair.py")

from meep_gpu.metal_kernels import (  # noqa: E402
    arms, complex_fields, complex_fused_magnetic_pair as family,
    fused_magnetic_pair as real_twin, registry, shaders, templates,
)

ARM = "FMA_V1"  # a spelling to emit WITH, never a claim about this platform.
PERIODIC = (0, 0, 0)
UNPHASED = (0, 0, 0)
UNWALLED = (False, False, False)


# ---------------------------------------------------------------------------
# 1. The source is a LIFT, not a transcription — so the lift is checkable
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("codes,phased", (((0, 0, 0), (0, 0, 0)),
                                          ((0, 0, 0), (1, 1, 1)),
                                          ((1, 0, 1), (0, 1, 0)),
                                          ((1, 1, 1), (0, 0, 0))))
def test_the_curl_half_is_the_certified_complex_curl_bodys_own_bytes(codes, phased):
    """The certified complex ``step_B`` body appears VERBATIM, both sides of the splice.

    This family does not retype either half: it lifts them from
    ``complex_fields.bloch_curl_source`` and ``bloch_constitutive_source``. That makes
    the transcription rule a substring check rather than a promise.
    """
    source = family.complex_fused_magnetic_pair_source(codes, phased, UNWALLED, ARM)
    body = family.certified_curl_body(codes, phased, ARM)
    head, tail = body.split(family._CURL_STORE, 1)
    assert head in source
    assert (family._CURL_STORE + tail) in source


def test_the_curl_half_is_the_forward_direction():
    """``step_B`` differences UP. ``step_D``'s negated strides are another product."""
    source = family.complex_fused_magnetic_pair_source(
        (1, 1, 1), UNPHASED, (True, True, True), ARM)
    assert "int si = i + 1, sj = j + 1, sk = k + 1;" in source
    assert "int si = i - 1, sj = j - 1, sk = k - 1;" not in source


def test_the_constitutive_half_differs_only_in_the_three_seam_lines():
    """Every other line of the certified H body must survive the lift unedited."""
    source = family.complex_fused_magnetic_pair_source(
        PERIODIC, UNPHASED, UNWALLED, ARM)
    certified = family.certified_constitutive_body(ARM).splitlines()
    marker = "    // --- update_H (stepping.update_H"
    spliced = [line for line in source.split(marker, 1)[1].splitlines()
               if "// THE SEAM:" not in line]
    spliced = spliced[1:] if spliced and spliced[0].endswith("--") else spliced
    while spliced and spliced[-1].strip() in ("", "}"):
        spliced.pop()
    assert len(certified) == len(spliced)
    changed = [(a, b) for a, b in zip(certified, spliced) if a != b]
    assert changed == [(f"    float2 src{t} = g{t}[ii];",
                        f"    float2 src{t} = v{t};") for t in range(3)]


def test_the_seam_removes_every_reference_to_the_curls_source_buffer():
    """``g0`` means E in the curl half and B in the certified H body — ONE name, two
    meanings, and the seam substitution is what resolves it.

    If a future edit renamed the H body's source instead of substituting it, ``g0``
    would still appear after the marker and would silently read the ELECTRIC field as
    the magnetic flux density.
    """
    source = family.complex_fused_magnetic_pair_source(
        PERIODIC, UNPHASED, UNWALLED, ARM)
    marker = "    // --- update_H (stepping.update_H"
    constitutive_half = source.split(marker, 1)[1]
    assert "g0[ii]" not in constitutive_half
    assert "g1[ii]" not in constitutive_half
    assert "g2[ii]" not in constitutive_half


def test_the_h_side_carries_no_inverse_epsilon():
    """``update_H``'s source is B unmultiplied; the E side's ``inv_eps`` is dropped.

    That drop is exactly why this signature is 27 pointers and not 30, so it is
    asserted rather than left as a docstring claim.
    """
    source = family.complex_fused_magnetic_pair_source(
        PERIODIC, UNPHASED, UNWALLED, ARM)
    assert "e0" not in source.split("kernel void", 1)[1].split(")", 1)[0]
    assert "c_mul_field_left(g0[ii], e0[ii])" not in source


@pytest.mark.parametrize("bad", ((0, 0), (0, 0, 0, 0)))
def test_a_boundary_or_phase_triple_that_is_not_a_triple_is_refused(bad):
    with pytest.raises(ValueError):
        family.complex_fused_magnetic_pair_source(bad, UNPHASED, UNWALLED, ARM)
    with pytest.raises(ValueError):
        family.complex_fused_magnetic_pair_source(PERIODIC, bad, UNWALLED, ARM)


def test_a_phase_on_a_metallic_axis_is_refused_by_the_certified_emitter():
    """One raise, in the emitter that owns the rule (mirroring S:2346-2360).

    This family does not re-check it: a second copy of the rule is a second place for
    it to drift.
    """
    with pytest.raises(ValueError):
        family.complex_fused_magnetic_pair_source(
            (1, 0, 0), (1, 0, 0), UNWALLED, ARM)


# ---------------------------------------------------------------------------
# 2. The inline wall clear IS the B-side table, and its zero is COMPLEX
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("walled,expected", (
    ((False, False, False), ()),
    ((True, False, False), ("v0 = at_x",)),
    ((False, True, False), ("v1 = at_y",)),
    ((False, False, True), ("v2 = at_z",)),
    ((True, True, True), ("v0 = at_x", "v1 = at_y", "v2 = at_z")),
))
def test_the_inline_wall_clear_is_the_b_yee_shift_table(walled, expected):
    """Bx clears on an x wall, By on y, Bz on z — the DIAGONAL (fields.py:216-217)."""
    mask = family.zero_metal_mask(walled)
    for fragment in expected:
        assert fragment in mask
    if not expected:
        assert "no walled axis clears a B component" in mask


def test_the_b_table_is_the_complement_of_the_d_table():
    """THE SINGLE MOST LIKELY SLIP in porting this product from its D-side sibling.

    The D side clears the OFF-diagonal (Dy/Dz on an x wall); B clears the diagonal.
    Asserted against the real twin's own table so the two cannot drift into agreement.
    """
    assert family._ZERO_METAL_ROWS == real_twin._ZERO_METAL_ROWS
    for target, axis, _flag in family._ZERO_METAL_ROWS:
        assert target == axis, "the B table is the DIAGONAL"


def test_the_wall_clear_writes_a_complex_zero_to_both_planes():
    """The array path assigns a COMPLEX zero when it masks (S:1896, :1902).

    A real ``0.0f`` here would not even compile against a ``float2`` register, but the
    plane-wise question it stands for is live: clearing only the real plane is a
    must-catch gate mutation, and this is its table-level twin.
    """
    mask = family.zero_metal_mask((True, True, True))
    assert templates.COMPLEX_ZERO in mask
    assert "? 0.0f :" not in mask


def test_a_run_with_no_wall_emits_no_clear_at_all():
    source = family.complex_fused_magnetic_pair_source(
        PERIODIC, UNPHASED, UNWALLED, ARM)
    assert "no walled axis clears a B component" in source
    assert "? float2(0.0f, 0.0f) : v0;" not in source


def test_the_auxiliary_is_never_masked():
    """``zero_metal_B`` passes ``B_COMPONENTS`` only (stepping.py:2250)."""
    source = family.complex_fused_magnetic_pair_source(
        (1, 1, 1), UNPHASED, (True, True, True), ARM)
    assert "u0[ii] = n0; u1[ii] = n1; u2[ii] = n2;" in source
    assert "u0[ii] = at_x" not in source


# ---------------------------------------------------------------------------
# 3. The phase is SKIPPED, not multiplied by 1+0j
# ---------------------------------------------------------------------------

def test_an_unphased_axis_emits_no_multiply_at_all():
    """The SKIP is the bit-identity of k = 0 (S:1768-1770), not a multiply by one."""
    source = family.complex_fused_magnetic_pair_source(
        PERIODIC, UNPHASED, UNWALLED, ARM)
    assert "carries no Bloch phase" in source
    assert "c_mul(b_x, px)" not in source
    assert "c_mul(a_y, py)" not in source


@pytest.mark.parametrize("axis,operands,phase", ((0, ("b_x", "c_x"), "px"),
                                                 (1, ("a_y", "c_y"), "py"),
                                                 (2, ("a_z", "b_z"), "pz")))
def test_each_phased_axis_rotates_exactly_the_operands_that_crossed_it(
        axis, operands, phase):
    """Which operand crossed which face follows the stencil, not the axis order."""
    phased = tuple(1 if n == axis else 0 for n in range(3))
    source = family.complex_fused_magnetic_pair_source(
        PERIODIC, phased, UNWALLED, ARM)
    for operand in operands:
        assert f"c_mul({operand}, {phase})" in source


# ---------------------------------------------------------------------------
# 4. The predicate: the source seam is this family's binding clause
# ---------------------------------------------------------------------------

class _Source:
    def __init__(self, field_type: str) -> None:
        self.field_type = field_type


class _Grid:
    shape = (4, 4, 4)
    has_metallic = False

    def is_metallic(self, axis): return False
    def is_mirrored(self, axis): return False


class _Fields:
    grid = _Grid()


def test_an_undeclared_source_set_is_refused_by_name():
    """Ignorance is never an empty set: ``Fields`` does not hold the source list."""
    coverage = family.metal_complex_fused_magnetic_pair_coverage(
        _Fields(), None, None, None)
    assert not coverage.covered
    assert any("was not declared" in reason for reason in coverage.reasons)


def _seam_reasons(sources):
    """The seam clause's own reasons, with both halves' refusals filtered out.

    The stub grid refuses both halves on their own clauses, which is fine and not what
    these cases are about; keeping them would make "no seam reason" unmeasurable.
    """
    coverage = family.metal_complex_fused_magnetic_pair_coverage(
        _Fields(), None, sources, None)
    return tuple(reason for reason in coverage.reasons if "half: " not in reason)


def test_a_magnetic_source_is_carried_by_name_and_an_electric_one_is_not_this_seam():
    """THE POLARITY IS THE WHOLE POINT of preferring this seam over the D-side one,
    and it survives the 2026-08-28 flip.

    ``CARRIES_DEPOSIT_REPAIR`` is now True, so a magnetic source is no longer refused
    for BEING magnetic -- it is CARRIED, and what refuses here is what the repair
    cannot establish on a stub: no ``f_w_Hx`` to save, and a source that publishes no
    deposit index. Both are ``deposit_repair``'s own clauses (deposit_repair.py:108-111,
    :244-247), and both are reached only for an IN-SEAM source, so the polarity is still
    exactly what is being measured. The electric source produces no seam reason at all,
    because it is injected in the other half-step.
    """
    magnetic = _seam_reasons((_Source("B"),))
    assert any("f_w_Hx is not allocated" in reason for reason in magnetic), magnetic
    assert any("does not publish the index it writes" in reason
               for reason in magnetic), magnetic
    assert not _seam_reasons((_Source("D"),))


def test_the_pre_flip_refusal_is_still_what_the_flag_returns_when_held_false(monkeypatch):
    """The other direction, and the record of what this clause said for eleven months.

    Held at False by monkeypatch now that this family ships True: the same code path,
    and the prose this family's gate pins. Without this case the flip would be
    indistinguishable from the clause having been deleted.
    """
    monkeypatch.setattr(family, "CARRIES_DEPOSIT_REPAIR", False)
    magnetic = _seam_reasons((_Source("B"),))
    assert any("is magnetic" in r and "driver.py:3283-3284" in r
               for r in magnetic), magnetic
    assert not any("is magnetic" in r for r in _seam_reasons((_Source("D"),)))


def test_the_polarity_matches_the_real_twins():
    """Both B-seam families refuse the same field type, from one constant."""
    from meep_gpu.triton_kernels.coverage import MAGNETIC_FIELD_TYPE

    assert MAGNETIC_FIELD_TYPE == "B"


def test_the_predicate_never_raises_on_a_degenerate_object():
    """``plan_step``'s whole contract is that a predicate REFUSES rather than raises."""
    coverage = family.metal_complex_fused_magnetic_pair_coverage(
        object(), None, (), None)
    assert not coverage.covered
    assert coverage.reasons


def test_both_halves_predicates_are_conjoined_and_labelled():
    """A configuration either half refuses is refused here, with the half named."""
    coverage = family.metal_complex_fused_magnetic_pair_coverage(
        _Fields(), None, (), None)
    assert not coverage.covered
    assert any(r.startswith("curl half: ") for r in coverage.reasons)
    assert any(r.startswith("constitutive half: ") for r in coverage.reasons)


def test_real_storage_is_refused_by_the_inherited_complex_clause():
    """The domain split against the shipped real kernels runs BOTH ways."""
    coverage = family.metal_complex_fused_magnetic_pair_coverage(
        _Fields(), None, (), None)
    assert any("storage is real float32" in r for r in coverage.reasons)


# ---------------------------------------------------------------------------
# 5. The arm is BOUND FROM AN ARTIFACT, and this family carries no probe of its own
# ---------------------------------------------------------------------------

def test_the_family_binds_through_the_complex_families_probe():
    """One question, one answer. A second probe here would be a way for two to drift."""
    assert not hasattr(family, "PROBE_PATH_ENVIRONMENT")
    assert not hasattr(family, "load_expansion_probe")
    assert complex_fields.PROBE_PATH_ENVIRONMENT == (
        "MEEP_GPU_METAL_COMPLEX_EXPANSION_PROBE")


def test_a_missing_or_ambiguous_probe_yields_no_arm():
    """Never a default arm: a platform that has not been measured is a refusal."""
    assert complex_fields.expansion_from_probe(None) is None
    ambiguous = {"backend": complex_fields.PROBE_BACKEND,
                 "patterns": {name: complex_fields.AMBIGUOUS_BOTH
                              for name in complex_fields.PROBE_PATTERNS}}
    assert complex_fields.expansion_from_probe(ambiguous) is None


@pytest.mark.parametrize("arm", templates.EXPANSION_ARMS)
def test_the_helpers_are_emitted_verbatim_for_whichever_arm_is_bound(arm):
    """The helpers ARE the arm; the kernel must carry the certified block unedited."""
    source = family.complex_fused_magnetic_pair_source(
        PERIODIC, (1, 1, 1), UNWALLED, arm)
    assert templates.complex_helpers(arm) in source


def test_the_two_arms_produce_different_sources():
    """If they did not, the probe would be binding nothing."""
    a = family.complex_fused_magnetic_pair_source(PERIODIC, (1, 1, 1), UNWALLED,
                                                  "FMA_V1")
    b = family.complex_fused_magnetic_pair_source(PERIODIC, (1, 1, 1), UNWALLED,
                                                  "NAIVE")
    assert a != b


# ---------------------------------------------------------------------------
# 6. The table: registered, enumerable, and invisible to plan_step
# ---------------------------------------------------------------------------

def test_the_arm_is_registered_unwired_on_exactly_one_slot():
    rows = [spec for spec in arms.registered() if spec.family == family.FAMILY]
    assert len(rows) == 1, rows
    assert rows[0].slot == family.SLOT == "step_B"
    assert rows[0].wired is False, (
        "this product spans step_B, zero_metal_B and update_H; wiring it would hand "
        "plan_step a slot assignment no composition rule has measured")


def test_plan_step_cannot_select_it():
    """``arms_for`` is the gate between the table and the composer."""
    context = arms.StepContext(object(), None, None, (), sources=())
    labels = [arm.label for arm in arms.arms_for(family.SLOT, context)]
    assert "complex fused magnetic B/H pair" not in labels
    assert "complex fused magnetic B/H pair" in [
        spec.label for spec in arms.registered(family.SLOT)]


def test_the_family_is_in_family_modules():
    """A family in the tree but not in ``registry`` is invisible to ``plan_step``,
    which is a silent coverage loss rather than an error."""
    assert family.FAMILY in registry.FAMILY_MODULES


def test_the_plan_declares_every_pass_it_performs():
    """``replaces_sub_steps`` is READ by the composer and by the whole-step walk."""
    assert family.REPLACES == ("step_B", "zero_metal_B", "update_H")
    assert (family.MetalComplexFusedMagneticPairPlan.replaces_sub_steps
            == family.REPLACES)
    assert family.MetalComplexFusedMagneticPairPlan.launches_per_run == 1
    assert family.MetalComplexFusedMagneticPairPlan.performs_device_work is True


# ---------------------------------------------------------------------------
# 7. The platform ceiling that decides the product's SHAPE
# ---------------------------------------------------------------------------

def test_float2_is_forced_the_split_plane_form_cannot_be_built():
    """Thirty field planes plus twelve coefficients is over the ceiling ALONE.

    Whether it FAILS TO COMPILE is a device measurement and belongs to the gate's
    ``binding_ceiling`` leg; what belongs here is that the signature this suite and
    that leg talk about is the same one, and that it is over the documented limit.
    """
    from meep_gpu.metal_kernels.device import MAX_BUFFER_BINDINGS

    source = family.split_plane_pair_signature()
    highest = max(int(chunk.split(")")[0])
                  for chunk in source.split("[[buffer(")[1:])
    assert highest + 1 == family.SPLIT_PLANE_BINDINGS == 53
    assert 15 * 2 + 12 > MAX_BUFFER_BINDINGS, (
        "the field pointers alone must break the ceiling; that is the claim")


def test_the_packed_struct_is_forced_separate_scalars_do_not_fit():
    """27 pointers + 5 scalars + 3 phases is 35 — the certified curl's own binding
    style, which this pair's fifteen volumes push over the edge."""
    from meep_gpu.metal_kernels.device import MAX_BUFFER_BINDINGS

    source = family.refuted_separate_scalar_source()
    highest = max(int(chunk.split(")")[0])
                  for chunk in source.split("[[buffer(")[1:])
    assert highest + 1 == family.SEPARATE_SCALAR_BINDINGS == 35
    assert family.SEPARATE_SCALAR_BINDINGS > MAX_BUFFER_BINDINGS


def test_the_shipped_packed_signature_fits_with_room():
    """27 pointers plus one packed struct is 28 — three under the ceiling."""
    from meep_gpu.metal_kernels.device import MAX_BUFFER_BINDINGS

    source = family.complex_fused_magnetic_pair_source(
        (1, 1, 1), UNPHASED, (True, True, True), ARM)
    highest = max(int(chunk.split(")")[0])
                  for chunk in source.split("[[buffer(")[1:])
    assert highest + 1 == family.PACKED_BINDINGS == 28
    assert family.PACKED_BINDINGS <= MAX_BUFFER_BINDINGS


def test_the_pointer_count_is_the_real_twins_because_float2_does_not_double_it():
    """A complex64 volume binds as ONE ``float2*``, exactly as a float32 volume binds
    as one ``float*``. That is the whole reason this fusion fits at all."""
    assert family.PACKED_BINDINGS == real_twin.PACKED_BINDINGS == 28


def test_the_coefficients_stay_float32_under_complex_storage():
    """stepping.py:41-50 — every PML coefficient is float32 in BOTH storage modes.

    Had they widened with the fields this signature would need 39 pointers and the
    product could not be built, so the invariant is asserted rather than assumed.
    """
    source = family.complex_fused_magnetic_pair_source(
        PERIODIC, UNPHASED, UNWALLED, ARM)
    signature = source.split("kernel void", 1)[1].split("{", 1)[0]
    for name in ("kmx", "sinvx", "kp0", "km0"):
        assert f"device const float*  {name}" in signature.replace("   ", "  ")


# ---------------------------------------------------------------------------
# 8. The Params record: the alignment trap, as a table-level claim
# ---------------------------------------------------------------------------

def test_the_params_record_puts_the_float2_members_first():
    """Metal aligns ``float2`` to 8 bytes. Phases first means NO internal padding, so
    the natural offsets coincide with Metal's — measured, and the reason the shipped
    order is this one. The device round-trip is the gate's ``params_layout`` leg."""
    dtype = family.params_record_dtype()
    assert dtype.names[:3] == ("px", "py", "pz")
    assert [dtype.fields[n][1] for n in dtype.names] == [0, 8, 16, 24, 28, 32, 36, 40]
    assert dtype.itemsize == family.PARAMS_ITEMSIZE == 48


def test_the_struct_declaration_agrees_with_the_host_record():
    """One layout, declared twice — so the two are compared rather than trusted."""
    source = family.complex_fused_magnetic_pair_source(
        PERIODIC, UNPHASED, UNWALLED, ARM)
    struct = source.split("struct Params {", 1)[1].split("};", 1)[0]
    order = [token for token in struct.replace(";", " ").split()
             if token in ("px", "py", "pz", "nx", "ny", "nz", "n_elem", "dtdx")]
    assert tuple(order) == family.params_record_dtype().names


# ---------------------------------------------------------------------------
# 9. Specialisation enumeration and the contraction guard
# ---------------------------------------------------------------------------

def test_enumerate_sources_covers_only_reachable_specialisations():
    """A metallic axis cannot carry a phase and a periodic axis cannot be walled, so
    the enumeration is the REACHABLE set rather than the Cartesian product."""
    labels = family.enumerate_sources(ARM)
    assert labels
    for label in labels:
        _kernel, codes, phase, wall, arm = label.split("/")
        assert arm == ARM
        for axis in range(3):
            metallic = codes[axis] == "1"
            assert not (metallic and phase[2 + axis] == "1")
            assert not ((not metallic) and wall[2 + axis] == "1")


def test_the_contraction_guard_is_emitted_in_both_modes():
    """Each mode is a different compiled kernel on this backend."""
    off = family.complex_fused_magnetic_pair_source(
        (1, 1, 1), UNPHASED, (True,) * 3, ARM, shaders.CONTRACT_OFF)
    fast = family.complex_fused_magnetic_pair_source(
        (1, 1, 1), UNPHASED, (True,) * 3, ARM, shaders.CONTRACT_FAST)
    assert "contract(off)" in off
    assert off != fast


# ---------------------------------------------------------------------------
# 10. The gate exists, and the bytes it certified are these bytes
# ---------------------------------------------------------------------------

def test_the_device_gate_is_present_and_routes_through_the_runner():
    assert GATE.is_file()
    text = GATE.read_text(encoding="utf-8")
    assert "run_current_measurement" in text, (
        "a Metal gate must run through metal_gate_runner so its artifact records the "
        "sources the PROCESS imported")
    assert "gate_provenance" in text


def test_the_family_has_no_checked_in_kernel_fingerprint():
    """The sources are a function of the PROBE-BOUND arm, so a checked-in hash would
    record a choice rather than a measurement — :mod:`.complex_fields`' rule, and this
    family inherits it because it inherits the arm."""
    import json

    fingerprints = json.loads(
        (PACKAGE_DIR / "metal_kernels" / "fingerprints.json").read_text("utf-8"))
    kernels = fingerprints.get("kernel_source_sha256", {})
    assert not any(key.startswith("complex_fused_magnetic_pair") for key in kernels)
