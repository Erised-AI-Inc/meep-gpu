"""The complex conductive D->E pair, as claims a laptop can check without a GPU.

WHAT THIS SUITE OWNS AND WHAT IT DELIBERATELY DOES NOT. The BYTES are the device
gate's — ``parity/meep_gpu/gate_metal_complex_conductive_fused_pair.py`` steps two
engines side by side for sixty complete seam steps and compares uint32 words — and
no assertion here duplicates that. What lives here is everything true about the
family WITHOUT a device: what the emitted source says, which configurations the
predicate refuses and BY WHAT NAME, that the arm is registered UNWIRED, and that
the signature respects the platform ceiling by ARITHMETIC.

THE TRANSCRIPTION TESTS PARSE, THEY DO NOT MIRROR. Each one CALLS the certified
emitter and searches its output; none re-implements a line.

THE CLAUSE THIS FAMILY LIVES OR DIES ON is the electric source. The driver injects
between the two halves (driver.py:3294-3299) and, on a conductive row, through the
``condinv``-scaled path at :3296 — so the whole cell exists only because its four
corpus rows carry a MAGNETIC source and no electric one. The POLARITY is asserted
in both directions here, and the seam's emptiness is measured on the real driver by
the gate's ``seam_is_empty`` leg and by
``results/metal_unbuilt_cells_reachability_2026-08-20/``.
"""

from __future__ import annotations

import pathlib
import re

import pytest

PACKAGE_DIR = pathlib.Path(__file__).resolve().parent
REPO = PACKAGE_DIR.parent
GATE = REPO / "parity" / "meep_gpu" / "gate_metal_complex_conductive_fused_pair.py"
REACHABILITY = (REPO / "parity" / "meep_gpu" / "results"
                / "metal_unbuilt_cells_reachability_2026-08-20" / "d_to_e_seam.json")

from meep_gpu.metal_kernels import (  # noqa: E402
    arms, complex_conductive_fused_pair as family, complex_no_pml_conductive,
    complex_no_pml_stored_e, registry, templates,
)
from meep_gpu.metal_kernels.device import MAX_BUFFER_BINDINGS  # noqa: E402

ARM = "FMA_V1"
PERIODIC = (0, 0, 0)
PHASED = (1, 1, 1)
CONDUCTIVE = (True, True, True)


def _source(codes=PERIODIC, phased=PHASED, conductive=CONDUCTIVE, poles=(1, 1, 1)):
    return family.complex_conductive_fused_pair_source(
        codes, phased, conductive, poles, ARM)


# ---------------------------------------------------------------------------
# 1. The curl half is the certified body, as ONE BLOCK
# ---------------------------------------------------------------------------

def test_the_curl_half_appears_as_one_contiguous_certified_block():
    """Stronger than a line-by-line check: the whole certified body, verbatim.

    ``complex_no_pml_conductive.complex_conductive_no_pml_curl_source`` is CALLED
    at the same specialisation and the span from the ghost-shift setup to the wall
    flags is required to appear in this family's output unchanged. A reordering
    inside that span fails here; a line-by-line check would not see it.
    """
    certified = complex_no_pml_conductive.complex_conductive_no_pml_curl_source(
        PERIODIC, True, PHASED, CONDUCTIVE, ARM)
    start = certified.index("    int si = i - 1")
    end = certified.index("    bool at_x = (i == 0), at_y = (j == 0), at_z = (k == 0);")
    block = certified[start:end]
    assert len(block) > 800, "the extracted certified block is implausibly short"
    assert block in _source(), "the curl half is not the certified body verbatim"


@pytest.mark.parametrize("index", (0, 1, 2))
def test_the_conductive_tail_is_the_certified_emitters_own_text(index):
    """On a CONDUCTIVE component NOTHING moves: the tail already names the value."""
    tail = complex_no_pml_conductive._tail(index, True)
    assert f"float2 value{index} = f{index}[ii];" in tail
    assert tail in _source(), (
        f"the conductive tail for target {index} is not the certified text")


def test_the_non_conductive_tail_is_exactly_the_documented_split():
    """The ONE line this family rewrites, and the rewrite is asserted to BE a split.

    The certified plain tail is a single line; fused it becomes a declaration and a
    store. The declaration is required to be that line with ``f0[ii] = `` replaced
    by ``float2 value0 = ``, so an edit that changed the arithmetic while keeping
    the shape fails here.
    """
    certified = complex_no_pml_conductive._tail(0, False)
    assert certified == "    f0[ii] = f0[ii] - curl0;"
    plain = _source(conductive=(False, True, True))
    assert certified.replace("f0[ii] = ", "float2 value0 = ") in plain
    assert "    f0[ii] = value0;" in plain
    assert certified not in plain, "the unsplit line is still present as well"


def test_the_constitutive_half_is_the_certified_e_body_under_the_named_renames():
    certified = complex_no_pml_stored_e.complex_stored_e_source(2, ARM)
    fused = _source(poles=(2, 2, 2))
    for wanted, emitted in (
            ("    float2 source = d_in[idx];", "        float2 source = value0;"),
            ("    source = source - p0[idx];", "        source = source - p0_0[idx];"),
            ("    source = source - p1[idx];", "        source = source - p0_1[idx];"),
            ("    e_out[idx] = c_mul_field_left(source, inv_e[idx]);",
             "        e0[idx] = c_mul_field_left(source, iv0[idx]);")):
        assert wanted in certified, f"the certified E body's {wanted!r} moved"
        assert emitted in fused, f"{wanted!r} is not transcribed; it is a re-derivation"


def test_d_in_is_never_bound_because_it_IS_the_curls_own_target():
    """The shared binding is the fusion's whole footprint, asserted from the signature."""
    signature = _source().split("uint idx [[thread_position_in_grid]]")[0]
    assert "d_in" not in signature
    for index in range(3):
        assert f"f{index}" in signature and f"e{index}" in signature


def test_the_helper_block_is_the_certified_emitters_own_text():
    assert templates.complex_helpers(ARM) in _source()


# ---------------------------------------------------------------------------
# 2. The branch census — this body is NOT straight line, and that is the point
# ---------------------------------------------------------------------------

def test_the_only_if_is_the_guard_and_every_other_branch_is_a_ternary():
    """The curl half carries ghost wraps and Bloch wrap planes; the count is pinned.

    This family cannot answer the dead-branch question the way the E->P chain does,
    so the branches are ENUMERATED instead and a mutation is required to sit on a
    line the scored case executes. Recording the counts per specialisation is what
    turns a future added branch into a changed census rather than a silent pass.
    """
    for codes, phased, conductive, wraps in (
            (PERIODIC, PHASED, CONDUCTIVE, 3),
            (PERIODIC, (0, 0, 0), CONDUCTIVE, 0),
            (PERIODIC, PHASED, (True, False, False), 3),
            ((1, 0, 0), (0, 1, 1), CONDUCTIVE, 2)):
        source = _source(codes, phased, conductive)
        assert len(re.findall(r"\bif\s*\(", source)) == 1
        assert not re.findall(r"\b(for|while)\s*\(", source)
        assert source.count("bool w") == wraps, (
            f"codes={codes} phased={phased}: {source.count('bool w')} wrap planes, "
            f"expected {wraps}")


def test_a_walled_axis_emits_the_ownership_mask_and_an_unwalled_one_emits_the_comment():
    """The mask is the certified emitter's, and its ABSENCE is what the wall clause buys."""
    assert "// no metallic axis: no ownership mask" in _source()
    walled = _source(codes=(1, 0, 0), phased=(0, 1, 1))
    assert "? float2(0.0f, 0.0f) :" in walled
    assert templates.ownership_mask((1, 0, 0), True,
                                    zero=templates.COMPLEX_ZERO) in walled


# ---------------------------------------------------------------------------
# 3. The binding arithmetic, against the EMITTED signature
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("poles", ((0, 0, 0), (1, 1, 1), (2, 1, 3), (4, 4, 4)))
def test_binding_count_equals_the_emitted_attribute_count(poles):
    expected = family.binding_count(poles)
    assert expected <= MAX_BUFFER_BINDINGS
    source = _source(poles=poles)
    indices = [int(m) for m in re.findall(r"\[\[buffer\((\d+)\)\]\]", source)]
    assert sorted(indices) == list(range(expected))


def test_an_over_ceiling_signature_is_refused_by_name_not_left_to_the_compiler():
    poles = (4, 4, 5)
    assert family.binding_count(poles) > MAX_BUFFER_BINDINGS
    with pytest.raises(ValueError) as raised:
        _source(poles=poles)
    assert "device.py:72" in str(raised.value)


def test_the_packed_params_is_what_buys_the_headroom_and_the_number_is_stated():
    """Without it, the certified curl's EIGHT separate scalars put two poles over.

    The arithmetic is done here rather than asserted in prose: the unpacked count
    is the fused pointers plus the certified curl's own scalar bindings.
    """
    unpacked_two_pole = (family.CURL_POINTERS + family.CONSTITUTIVE_POINTERS
                         + 2 * 3 + 8)
    assert unpacked_two_pole > MAX_BUFFER_BINDINGS
    assert family.binding_count((2, 2, 2)) <= MAX_BUFFER_BINDINGS
    assert family.params_words() == 11


def test_the_params_prologue_rebuilds_every_name_the_certified_emitters_use():
    """The prologue exists so GUARD, DECODE_IJK, ghost() and the phase blocks are
    emitted from the CERTIFIED emitters unchanged."""
    source = _source()
    for line in ("    uint nx = prm.nx;", "    uint ny = prm.ny;",
                 "    uint nz = prm.nz;", "    uint n_elem = prm.n_elem;",
                 "    float dtdx = prm.dtdx;",
                 "    float2 px = float2(prm.px_re, prm.px_im);",
                 "    float2 py = float2(prm.py_re, prm.py_im);",
                 "    float2 pz = float2(prm.pz_re, prm.pz_im);"):
        assert line in source
    assert templates.GUARD in source
    assert templates.DECODE_IJK in source


def test_the_pole_slots_are_not_padded_to_eight():
    signature = _source(poles=(1, 1, 1)).split("uint idx")[0]
    assert "p0_1" not in signature and "p0_7" not in signature


def test_a_phase_on_a_non_periodic_axis_is_refused_at_emit_time():
    with pytest.raises(ValueError) as raised:
        _source(codes=(1, 0, 0), phased=(1, 0, 0))
    assert "carries a phase but is not periodic" in str(raised.value)


# ---------------------------------------------------------------------------
# 4. The predicate: what it refuses, and by what name
# ---------------------------------------------------------------------------

def _grid(boundaries="periodic", symmetry=()):
    """A REAL grid, because the certified halves consult the array path for the
    per-axis ghost rule and a hand-rolled double cannot answer that. Device-free."""
    import numpy  # noqa: PLC0415

    from meep_gpu.grid import Grid  # noqa: PLC0415

    return Grid(resolution=8.0, cell_size=(1.0, 1.0, 1.0), boundaries=boundaries,
                dimensions=3, courant=0.35, symmetry=tuple(symmetry), xp=numpy)


class _Fields:
    """A minimal engine object: a real grid, nothing else allocated.

    Every certified half refuses it for its own reasons, which is fine — these
    tests assert on THIS family's clause names, and those are appended regardless.
    """

    def __init__(self, boundaries="periodic", symmetry=()):
        self.grid = _grid(boundaries, symmetry)
        self.polarizations = ()
        self.has_offdiagonal_epsilon = False
        self.has_nonlinearity = False


class _Source:
    def __init__(self, field_type):
        self.field_type = field_type


def _reasons(fields=None, sources=(), pml=None):
    return family.complex_conductive_fused_pair_coverage(
        _Fields() if fields is None else fields, pml, sources).reasons


def test_the_predicate_never_raises_on_a_degenerate_object():
    verdict = family.complex_conductive_fused_pair_coverage(object(), None, ())
    assert verdict.covered is False and verdict.reasons


def test_an_unreadable_boundary_rule_is_a_refusal_and_not_a_crash():
    """UNREADABLE IS A REFUSAL, and BOTH halves' clause lists obey that.

    ``_boundary_kinds`` returns ``None`` for a grid the array path will not resolve
    (coverage.py:105-121), and a clause that iterated it would raise where the
    contract says refuse. NOTHING IS STUBBED: the certified curl half refuses this
    same input by its own name out of ``complex_no_pml_conductive._base_reasons``,
    so the public predicate reaches this family's clause on the real path. The curl
    half's refusal is asserted here too — that is what makes the stub unnecessary,
    and it fails loudly if the shared base ever goes back to iterating the ``None``.
    """
    class _Unresolvable:
        shape = (8, 8, 8)

        def is_mirrored(self, axis):
            return False

    class _Broken(_Fields):
        def __init__(self):
            self.grid = _Unresolvable()
            self.polarizations = ()
            self.has_offdiagonal_epsilon = False
            self.has_nonlinearity = False

    verdict = family.complex_conductive_fused_pair_coverage(_Broken(), None, ())
    assert verdict.covered is False
    needle = "the per-axis boundary rule is unreadable"
    assert any(reason.startswith("curl half: ") and needle in reason
               for reason in verdict.reasons), verdict.reasons
    assert any(not reason.startswith("curl half: ") and needle in reason
               for reason in verdict.reasons), verdict.reasons


def test_both_halves_predicates_are_conjoined_and_labelled():
    reasons = _reasons()
    assert any(reason.startswith("curl half: ") for reason in reasons)
    assert any(reason.startswith("E half: ") for reason in reasons)


def test_an_undeclared_source_set_is_a_refusal_and_never_an_empty_set():
    reasons = family.complex_conductive_fused_pair_coverage(
        _Fields(), None, None).reasons
    assert any("the source set was not declared" in reason for reason in reasons)


def test_an_electric_source_is_refused_by_name_citing_both_driver_lines():
    """The clause the whole cell turns on, with the conductive path named too."""
    reasons = _reasons(sources=(_Source("D"),))
    assert any("driver.py:3294-3299" in reason for reason in reasons)
    assert any("driver.py:3296" in reason for reason in reasons)


def test_a_magnetic_source_does_NOT_trip_the_source_clause():
    """THE POLARITY, and it is what makes this cell's four corpus rows reachable.

    A magnetic source is deposited in the B/H half (driver.py:3283-3284), nowhere
    near this seam. Asserted as the ABSENCE of the electric clause rather than as
    full coverage, because the degenerate fixture is refused for other reasons.
    """
    reasons = _reasons(sources=(_Source("B"),))
    assert not any("is electric" in reason for reason in reasons)


def test_a_metallic_axis_is_refused_by_name_citing_zero_metal_D():
    fields = _Fields(boundaries=("metallic", "periodic", "periodic"))
    reasons = _reasons(fields)
    assert any("stepping.zero_metal_D runs inside this seam" in reason
               and "driver.py:3301" in reason for reason in reasons)


def test_a_folded_axis_is_refused_by_name_citing_both_D_side_fills():
    from meep_gpu.grid import Mirror  # noqa: PLC0415

    fields = _Fields(symmetry=(Mirror("Y", +1),))
    reasons = _reasons(fields)
    assert any("stepping.fill_symmetry_bc_D" in reason
               and "stepping.fill_folded_far_ghosts_D" in reason
               and "driver.py:3300-3302" in reason for reason in reasons)


def test_an_offdiagonal_row_is_refused_naming_the_stencil():
    """Restated here because THIS is the fact that makes the cell buildable at all."""
    fields = _Fields()
    fields.has_offdiagonal_epsilon = True
    reasons = _reasons(fields)
    assert any("STENCIL over the D volumes step_D writes in place" in reason
               and "stepping.py:1235-1253" in reason for reason in reasons)


# ---------------------------------------------------------------------------
# 5. Registration: the arm exists, is UNWIRED, and cannot be composed
# ---------------------------------------------------------------------------

def test_the_arm_is_registered_unwired_on_the_first_slot_it_spans():
    specs = [spec for spec in arms.registered() if spec.family == family.FAMILY]
    assert len(specs) == 1
    assert specs[0].slot == family.SLOT == "step_D"
    assert specs[0].wired is False
    assert family.REPLACES == ("step_D", "update_E")


def test_an_unwired_arm_is_invisible_to_the_composer():
    class _Context:
        fields = _Fields()
        pml = None
        residency = None
        contract_variants = ()
        extra: dict = {}
        sources = ()

    spec = next(s for s in arms.registered() if s.family == family.FAMILY)
    offered = {arm.label for arm in arms.arms_for("step_D", _Context())}
    assert spec.label not in offered
    assert offered, "no arm at all is offered; this assertion would be vacuous"


def test_the_module_is_named_in_the_registry_so_it_is_not_invisible():
    assert "complex_conductive_fused_pair" in registry.FAMILY_MODULES


def test_the_plan_declares_one_launch_per_run():
    """The whole-step arbiter asserts the exact per-cycle count on every filled slot."""
    assert family.MetalComplexConductiveFusedPairPlan.launches_per_run == 1
    assert family.MetalComplexConductiveFusedPairPlan.performs_device_work is True


# ---------------------------------------------------------------------------
# 6. The evidence this family's central claim rests on is PRESENT
# ---------------------------------------------------------------------------

def test_the_device_gate_exists_and_measures_the_seam_on_the_real_driver():
    assert GATE.exists()
    text = GATE.read_text()
    assert "complex_conductive_fused_pair as family" in text
    assert "def leg_seam_is_empty" in text, (
        "the seam's emptiness must be MEASURED on the driver, not read off a guard")
    assert "_inject_electric_through_conductivity" in text
    assert "gate_provenance" in text


def test_the_reachability_artifact_records_the_determination_and_its_control():
    """A determination whose control leg did not FIRE would be a blind instrument."""
    import json  # noqa: PLC0415

    assert REACHABILITY.exists(), (
        f"{REACHABILITY} is missing; the reachability of this whole cell rests on it")
    record = json.loads(REACHABILITY.read_text())
    assert record["determination"].startswith("REACHABLE")
    assert record["corpus_leg_seam_empty"] is True
    assert record["control_leg_fired"] is True
    assert record["corpus_leg_conductive_pass_ran"] is False
