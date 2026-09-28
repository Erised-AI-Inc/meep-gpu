"""Laptop tests for the Metal complex-field / Bloch family.

WHAT THIS FILE IS FOR, and what it deliberately is not. The BYTES are certified by
``parity/meep_gpu/gate_metal_complex.py``, which needs a GPU. These tests run
anywhere and pin the things a byte gate cannot:

1. **the predicate's clause list** — every refusal this family makes, by name, and
   the ADMISSIONS that are load-bearing (an absorber on a phased axis, off-diagonal
   epsilon on the curl);
2. **the probe contract** — the arm binds from a measured artifact or the family
   refuses, and every way a probe can fail to decide is a refusal rather than a
   default;
3. **source-text pins for the defects that are BYTE-INVISIBLE in float32** and are
   therefore recorded in the gate's ``predicted_nulls`` rather than mutated. A
   defect nobody can measure still has to be pinned somewhere, and source text is
   the honest place;
4. **the plan builder's None-only refusal contract**;
5. **the registration being ENUMERABLE BUT NOT WIRED**, which is what keeps this
   tranche out of dispatch while still letting the composition probe sweep it.

NO TORCH IS IMPORTED. Everything here is string and predicate work.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pytest

from meep_gpu.metal_kernels import arms, complex_fields as cx, shaders, templates

PACKAGE_DIR = os.path.dirname(os.path.abspath(cx.__file__))


# ---------------------------------------------------------------------------
# 1. Source shape and the specialisation
# ---------------------------------------------------------------------------

def test_every_shipped_complex_source_carries_the_contraction_directive():
    for arm in templates.EXPANSION_ARMS:
        for label, source in cx.enumerate_complex_sources(arm).items():
            assert shaders.contraction_pragma(shaders.CONTRACT_OFF) in source, label


def test_the_specialisation_enumerates_only_REACHABLE_phase_triples():
    """A metallic axis cannot carry a phase, so its flag is never enumerated.

    Fingerprinting a label that cannot be built would record a hash for a kernel no
    plan can emit, which is the quiet way a fingerprint file stops describing what
    ships.
    """
    labels = tuple(cx.enumerate_complex_sources("FMA_V1"))
    for label in labels:
        if not label.startswith("bloch_pml_curl_step/"):
            continue
        _, _, codes, phased, _ = label.split("/")
        for code, flag in zip(codes, phased[2:]):
            assert not (code == "1" and flag == "1"), label
    # All-periodic admits every triple; all-metallic admits only the unphased one.
    assert sum(1 for lab in labels if "/000/" in lab) == 2 * 8
    assert sum(1 for lab in labels if "/111/" in lab) == 2 * 1


def test_a_phase_on_a_metallic_axis_is_refused_at_source_time():
    with pytest.raises(ValueError, match="only a periodic wrap can carry a phase"):
        cx.bloch_curl_source((templates.METALLIC, 0, 0), False, (1, 0, 0), "FMA_V1")


def test_an_unphased_axis_emits_a_SKIP_and_no_rotation():
    """The k = 0 bit-identity is a SKIP, not a multiply by 1+0j (S:1768-1770)."""
    source = cx.bloch_curl_source((0, 0, 0), False, (0, 0, 0), "FMA_V1")
    assert source.count("SKIPPED") == 3
    # The three c_mul spellings in the helper block are definitions, not call sites.
    assert source.count("c_mul(") == 1


def test_the_ownership_mask_writes_a_COMPLEX_zero_to_both_planes():
    b_mask = cx.bloch_curl_source((templates.METALLIC,) * 3, False, (0, 0, 0),
                                  "FMA_V1")
    d_mask = cx.bloch_curl_source((templates.METALLIC,) * 3, True, (0, 0, 0),
                                  "FMA_V1")
    assert b_mask.count("? float2(0.0f, 0.0f) : curl") == 3
    assert d_mask.count("? float2(0.0f, 0.0f) : curl") == 6


def test_the_two_expansion_arms_really_differ():
    """A selector that selected nothing would make the probe decorative."""
    fused = cx.bloch_curl_source((0, 0, 0), False, (1, 0, 0), "FMA_V1")
    naive = cx.bloch_curl_source((0, 0, 0), False, (1, 0, 0), "NAIVE")
    assert fused != naive
    assert "fma(" in fused
    assert "fma(" not in templates.complex_helpers("NAIVE")


def test_the_curl_binds_23_buffers_and_the_split_form_would_need_35():
    """``float2`` is FORCED by the 31-binding ceiling, not preferred."""
    from meep_gpu.metal_kernels.device import MAX_BUFFER_BINDINGS

    source = cx.bloch_curl_source((0, 0, 0), False, (0, 0, 0), "FMA_V1")
    indices = [int(m) for m in re.findall(r"\[\[buffer\((\d+)\)\]\]", source)]
    assert indices == list(range(cx.CURL_BINDINGS))
    assert max(indices) < MAX_BUFFER_BINDINGS
    assert cx.split_plane_binding_count() == 35
    assert cx.split_plane_binding_count() > MAX_BUFFER_BINDINGS
    split = cx.split_plane_curl_signature()
    assert f"buffer({MAX_BUFFER_BINDINGS})" in split, (
        "the split signature must actually cross the ceiling, or the gate's "
        "compile-error leg would be measuring a kernel that fits")


def test_the_constitutive_binds_22_buffers_on_both_sides():
    for side in ("H", "E"):
        source = cx.bloch_constitutive_source(side, "FMA_V1")
        indices = [int(m) for m in re.findall(r"\[\[buffer\((\d+)\)\]\]", source)]
        assert indices == list(range(22)), side


def test_the_E_side_puts_D_on_the_LEFT():
    """S:982-984 writes ``source * inverse_epsilon``. The gate measured the two
    orientations EQUIVALENT for a real coefficient; the source still spells the
    array path's, and this test is what keeps that true after the equivalence
    stopped being enforced by a mutation."""
    source = cx.bloch_constitutive_source("E", "FMA_V1")
    assert "c_mul_field_left(g0[ii], e0[ii])" in source
    assert "c_mul_coefficient_left(e0[ii]" not in source


# ---------------------------------------------------------------------------
# 2. The three BYTE-INVISIBLE defects, pinned by source text
# ---------------------------------------------------------------------------
#
# Each of these appears in `gate_metal_complex.PREDICTED_NULLS` with the reason it
# cannot be measured. Pinning them here is what stops "not measured" becoming "not
# checked at all".

def test_complex_add_is_component_wise():
    """PREDICTED NULL: ``float2`` operators and two scalar ops emit the same rounds.

    MSL vector arithmetic is element-wise IEEE with no horizontal step, so no
    mutation can distinguish the spellings. The pin is that the curl's three
    differences stay ``float2`` expressions with their parens intact.
    """
    source = cx.bloch_curl_source((0, 0, 0), False, (0, 0, 0), "FMA_V1")
    assert "float2 t0 = ((c_y - c) + (b - b_z));" in source
    assert "float2 t1 = ((a_z - a) + (c - c_x));" in source
    assert "float2 t2 = ((b_x - b) + (a - a_y));" in source


def test_stores_are_one_float2_per_cell():
    """PREDICTED NULL: the planes do not alias, so store ORDER within a cell is
    unobservable. The pin is that there is one store per volume, not two."""
    source = cx.bloch_curl_source((0, 0, 0), False, (0, 0, 0), "FMA_V1")
    assert "u0[ii] = n0; u1[ii] = n1; u2[ii] = n2;" in source
    assert "f0[ii] = v0; f1[ii] = v1; f2[ii] = v2;" in source
    assert ".x =" not in source and ".y =" not in source


@pytest.mark.parametrize("axis,index,extent,phase,operands", [
    ("x", "i", "nxi", "px", ("b_x", "c_x")),
    ("y", "j", "nyi", "py", ("a_y", "c_y")),
    ("z", "k", "nzi", "pz", ("a_z", "b_z")),
])
def test_each_axis_rotates_ITS_OWN_operands_with_ITS_OWN_predicate(
        axis, index, extent, phase, operands):
    """Which operand crossed which face, per axis, pinned one axis at a time.

    THE Y BLOCK WAS UNMEASURED UNTIL 2026-08-15. Every gate case carried ``k_y = 0``
    and no synthetic phase set phased y, so ``_phase_block("y", ...)`` and the ``py``
    binding were emitted, fingerprinted and never launched against the oracle — a
    wrong index variable or a wrong operand pair there was invisible to the byte
    gate. The gate now carries ``periodic_ky``/``periodic_kxyz``, a per-axis
    synthetic single, a per-axis phase-coverage floor and mutations m6y/m7y/m14;
    this is the source-text half of the same pin.
    """
    flags = tuple(1 if a == axis else 0 for a in "xyz")
    source = cx.bloch_curl_source((0, 0, 0), False, flags, "FMA_V1")
    assert f"bool w{axis} = ({index} == {extent} - 1);" in source
    for operand in operands:
        assert f"{operand} = w{axis} ? c_mul({operand}, {phase}) : {operand};" in source
    # ... and no OTHER operand is rotated on this axis's predicate.
    for other in ("a_y", "a_z", "b_x", "b_z", "c_x", "c_y"):
        if other in operands:
            continue
        assert f"{other} = w{axis} ?" not in source
    backward = cx.bloch_curl_source((0, 0, 0), True, flags, "FMA_V1")
    assert f"bool w{axis} = ({index} == 0);" in backward


def test_phase_block_rotates_both_operands():
    """PREDICTED NULL: the two rotations are independent, so their ORDER is
    unobservable. The pin is that BOTH operands that crossed the face are rotated —
    dropping one is a real defect the gate's m6/m7 cover from another angle."""
    source = cx.bloch_curl_source((0, 0, 0), False, (1, 1, 1), "FMA_V1")
    for operand, phase in (("b_x", "px"), ("c_x", "px"), ("a_y", "py"),
                           ("c_y", "py"), ("a_z", "pz"), ("b_z", "pz")):
        assert f"{operand} = w" in source and f"c_mul({operand}, {phase})" in source


#: Every ternary condition the two complex kernels are allowed to select on, with
#: the exact integer declaration each one comes from.
_INTEGER_PREDICATES = {
    "vx": "bool vx = true, vy = true, vz = true;",
    "vy": "bool vx = true, vy = true, vz = true;",
    "vz": "bool vx = true, vy = true, vz = true;",
    "wx": "bool wx = (i ==", "wy": "bool wy = (j ==", "wz": "bool wz = (k ==",
    "at_x": "bool at_x = (i == 0), at_y = (j == 0), at_z = (k == 0);",
    "at_y": "bool at_x = (i == 0), at_y = (j == 0), at_z = (k == 0);",
    "at_z": "bool at_x = (i == 0), at_y = (j == 0), at_z = (k == 0);",
}


@pytest.mark.parametrize("codes", [(0, 0, 0), (1, 1, 1), (1, 0, 1)])
@pytest.mark.parametrize("backward", [False, True])
def test_no_select_in_these_kernels_branches_on_a_FIELD_VALUE(codes, backward):
    """DENORMALS-ARE-ZERO IS VISIBLE TO PREDICATES ON MPS, NOT ONLY TO ARITHMETIC.

    Measured and recorded in ``metal_kernels.subnormal``: ``x == 0.0f`` is TRUE for a
    subnormal ``x`` and ``x < 0.0f`` is FALSE for a negative subnormal one. So a
    select whose condition reads a FIELD would take a different branch on the device
    than the array path takes on the host, and it would do so INSIDE the band the
    precondition already excludes — which means the census could be clean and the
    branch still wrong, because the census bounds the ARITHMETIC and a taken branch
    is not arithmetic.

    That note was written for tranche 1's two kernels and this family adds a third
    select (the Bloch wrapped-lane rotation). The property is asserted rather than
    inherited: every ``?`` in either emitted source selects on a name from
    :data:`_INTEGER_PREDICATES`, and each of those is declared from an integer
    comparison on a loop index.
    """
    sources = [cx.bloch_curl_source(codes, backward, [1 if c == 0 else 0
                                                      for c in codes], "FMA_V1")]
    sources += [cx.bloch_constitutive_source(side, "FMA_V1") for side in ("H", "E")]
    for source in sources:
        conditions = re.findall(r"([A-Za-z_][A-Za-z_0-9.]*)\s*\?", source)
        assert conditions or "constitutive" in source, source[:200]
        for condition in conditions:
            assert condition in _INTEGER_PREDICATES, (
                f"{condition!r} is selected on and is not a known integer "
                f"predicate; on MPS a select on a FIELD VALUE takes the flushed "
                f"branch for a subnormal operand and the subnormal-free "
                f"precondition does not bound it")
            assert _INTEGER_PREDICATES[condition] in source, (
                f"{condition!r} is selected on but its integer declaration "
                f"{_INTEGER_PREDICATES[condition]!r} is not in the source")


# ---------------------------------------------------------------------------
# 3. The probe contract
# ---------------------------------------------------------------------------

_GOOD = {
    "backend": "numpy",
    "patterns": {"c8_mul_c8": "FMA_V1",
                 "c8_mul_c8_scalar_right": "FMA_V1",
                 "c8_mul_f4_field_left": cx.AMBIGUOUS_BOTH,
                 "f4_mul_c8_coefficient_left": cx.AMBIGUOUS_BOTH,
                 "python_float_left": cx.AMBIGUOUS_BOTH},
}


def test_a_good_probe_binds_the_arm():
    assert cx.expansion_from_probe(_GOOD) == "FMA_V1"


@pytest.mark.parametrize("record,why", [
    (None, "missing"),
    ({}, "empty"),
    ({"backend": "cupy", "patterns": _GOOD["patterns"]}, "the wrong backend"),
    ({"backend": "numpy", "patterns": {}}, "no patterns"),
    ({"backend": "numpy",
      "patterns": {**_GOOD["patterns"], "c8_mul_c8": "NEITHER"}}, "NEITHER"),
    ({"backend": "numpy",
      "patterns": {**_GOOD["patterns"], "c8_mul_c8": "NAIVE"}}, "disagreeing"),
    ({"backend": "numpy",
      "patterns": {p: cx.AMBIGUOUS_BOTH for p in cx.PROBE_PATTERNS}},
     "nothing discriminated"),
])
def test_a_probe_that_cannot_decide_refuses(record, why):
    """Every failure mode returns None. A default here would be a guess wearing an
    artifact's clothes."""
    assert cx.expansion_from_probe(record) is None, why


def test_the_probe_binds_to_the_backend_the_ENGINE_holds():
    """numpy, not cupy — the inversion of the Triton track, and the whole reason the
    probe measures the HOST rather than the device."""
    assert cx.PROBE_BACKEND == "numpy"


def test_an_unreadable_probe_is_treated_as_missing(tmp_path):
    bad = tmp_path / "not.json"
    bad.write_text("{ this is not json", encoding="utf-8")
    assert cx.load_expansion_probe(str(bad)) is None


# ---------------------------------------------------------------------------
# 4. Coverage — the clause list, by name
# ---------------------------------------------------------------------------

class _Grid:
    def __init__(self, **kwargs):
        self.xp = np
        self.shape = (6, 5, 4)
        self.k_point = kwargs.pop("k_point", (0.3, 0.0, 0.0))
        self.has_bloch = kwargs.pop("has_bloch", True)
        self.cylindrical = kwargs.pop("cylindrical", False)
        self.bfast_active = kwargs.pop("bfast_active", False)
        self.beta = kwargs.pop("beta", 0.0)
        self._kinds = kwargs.pop("kinds", ("periodic", "periodic", "periodic"))
        self._phases = kwargs.pop("phases", (complex(-1.0, 0.0), None, None))
        for key, value in kwargs.items():
            setattr(self, key, value)

    def has_symmetry(self):
        return False

    def is_mirrored(self, axis):
        return False

    def is_axis(self, axis):
        return False

    def is_metallic(self, axis):
        return self._kinds[axis] == "metallic"

    def bloch_phase(self, axis):
        return self._phases[axis]


def _reasons(**kwargs):
    """Every refusal this family makes for one synthetic configuration."""
    grid = _Grid(**kwargs.pop("grid", {}))

    class _Fields:
        pass

    fields = _Fields()
    fields.grid = grid
    fields.force_complex_fields = kwargs.pop("force_complex_fields", True)
    fields.stores_E = kwargs.pop("stores_E", True)
    fields.has_nonlinearity = kwargs.pop("has_nonlinearity", False)
    fields.has_polarizations = kwargs.pop("has_polarizations", False)
    fields.polarizations = ()
    fields.has_offdiagonal_epsilon = kwargs.pop("has_offdiagonal_epsilon", False)
    fields.condfac_for = kwargs.pop("condfac_for", lambda name: None)
    for key, value in kwargs.items():
        setattr(fields, key, value)
    verdict = cx.complex_pml_curl_coverage(fields, None, "step_B", None,
                                           probe=_GOOD)
    return verdict.reasons


def test_a_real_storage_run_is_refused_by_name():
    reasons = _reasons(force_complex_fields=False,
                       grid={"has_bloch": False, "k_point": (0.0, 0.0, 0.0)})
    assert any("storage is real float32" in r for r in reasons)


def test_beta_is_refused_as_grid_beta_and_NOT_as_a_bloch_phase():
    """refl-angular-kz2d.py and parallel-wvgs-force.py are ``grid.beta`` runs. If
    this family admitted them they would be stepped with no beta term at all, which
    is a silent wrong answer rather than a crash."""
    reasons = _reasons(grid={"beta": 0.7})
    assert any("that is grid.beta, not a Bloch phase" in r for r in reasons)


@pytest.mark.parametrize("kwargs,needle", [
    ({"has_nonlinearity": True}, "chi2/chi3 is installed"),
    ({"has_polarizations": True}, "complex-storage ADE is a later tranche"),
    ({"stores_E": False}, "E is recomputed from D"),
    ({"grid": {"cylindrical": True}}, "cylindrical (Dcyl)"),
    ({"grid": {"bfast_active": True}}, "BFAST is active"),
    ({"condfac_for": lambda name: object()}, "a conductivity is installed"),
    ({"condfac_for": None}, "fields does not expose condfac_for"),
])
def test_each_refusal_is_reported_by_name(kwargs, needle):
    assert any(needle in r for r in _reasons(**kwargs)), needle


def test_a_phase_on_a_metallic_axis_is_refused_by_the_predicate_too():
    reasons = _reasons(grid={"kinds": ("metallic", "periodic", "periodic"),
                             "phases": (complex(-1.0, 0.0), None, None),
                             "k_point": (0.5, 0.0, 0.0)})
    assert any("only a periodic wrap can carry a phase" in r for r in reasons)
    assert any("no lattice vector for the phase" in r for r in reasons)


def test_an_unreadable_phase_is_NOT_an_unphased_one():
    def raiser(axis):
        raise RuntimeError("boom")

    grid_kwargs = {"phases": (None, None, None)}
    grid = _Grid(**grid_kwargs)
    grid.bloch_phase = raiser

    class _Fields:
        pass

    fields = _Fields()
    fields.grid = grid
    fields.force_complex_fields = True
    fields.stores_E = True
    fields.has_nonlinearity = False
    fields.has_polarizations = False
    fields.polarizations = ()
    fields.condfac_for = lambda name: None
    reasons = cx.complex_pml_curl_coverage(fields, None, "step_B", None,
                                           probe=_GOOD).reasons
    assert any("an unreadable phase is not an unphased one" in r for r in reasons)


def test_no_probe_is_a_refusal_by_name(monkeypatch):
    """MONKEYPATCH, not a bare pop: the variable is PROCESS-WIDE DISPATCH STATE.

    This test popped ``$MEEP_GPU_METAL_COMPLEX_EXPANSION_PROBE`` and never put it
    back, and the leak was not cosmetic — it is what the predicate binds its
    multiply arm from, so every later test in the session saw the complex arms
    refuse. Measured: with this file ahead of it, ``test_metal_planner_composition``
    lost six tests (its own precondition check plus the five complex rows of the
    composition matrix) and each read as a planner defect. ``monkeypatch.delenv``
    restores at teardown, which is the only reason the deletion is safe here.
    """
    grid = _Grid()

    class _Fields:
        pass

    fields = _Fields()
    fields.grid = grid
    fields.force_complex_fields = True
    fields.stores_E = True
    fields.condfac_for = lambda name: None
    monkeypatch.delenv(cx.PROBE_PATH_ENVIRONMENT, raising=False)
    reasons = cx.complex_pml_curl_coverage(fields, None, "step_B", None).reasons
    assert any("expansion probe artifact" in r for r in reasons)


def test_offdiagonal_epsilon_is_admitted_by_the_curl_and_refused_by_update_E():
    """The per-sub-step split: the row product is constitutive-only (S:972-979)."""
    curl_reasons = _reasons(has_offdiagonal_epsilon=True)
    assert not any("off-diagonal" in r for r in curl_reasons)


def test_the_plan_builders_refuse_with_None_and_never_raise():
    class _Fields:
        pass

    fields = _Fields()
    fields.grid = _Grid()
    fields.force_complex_fields = False
    assert cx.plan_complex_pml_curl(fields, None, "step_B", None) is None
    assert cx.plan_complex_constitutive(fields, None, "H", None) is None


def test_an_unknown_sub_step_or_side_RAISES_rather_than_refusing():
    """A caller typo is a programming error and must not look like 'not covered'."""
    with pytest.raises(ValueError):
        cx.complex_pml_curl_coverage(object(), None, "step_Q")
    with pytest.raises(ValueError):
        cx.complex_constitutive_coverage(object(), None, "Q")


# ---------------------------------------------------------------------------
# 5. The phase encoding
# ---------------------------------------------------------------------------

def test_the_backward_sub_step_gets_the_CONJUGATE():
    """S:1818-1822. The classic sign error leaves every magnitude plausible."""
    phase = complex(np.exp(2j * np.pi * 0.3))
    forward_flags, forward = cx.phase_arguments((phase, None, None), backward=False)
    backward_flags, backward = cx.phase_arguments((phase, None, None), backward=True)
    assert forward_flags == backward_flags == (1, 0, 0)
    assert forward[0][0] == backward[0][0]
    assert forward[0][1] == -backward[0][1] != 0.0


def test_an_unphased_axis_encodes_1_0_under_flag_0():
    flags, values = cx.phase_arguments((None, None, None), backward=False)
    assert flags == (0, 0, 0)
    assert values == ((1.0, 0.0), (1.0, 0.0), (1.0, 0.0))


def test_the_brillouin_edge_is_EXACTLY_minus_one():
    flags, values = cx.phase_arguments((complex(-1.0, 0.0), None, None),
                                       backward=False)
    assert values[0] == (-1.0, 0.0)
    # And its conjugate is itself: the imaginary part is an exact zero, whose
    # negation is -0.0 and whose product with anything is a signed zero. That is the
    # one phase value at which a plane-wise collapse would be invisible on random
    # data, which is why the gate pins the edge as a case rather than sampling it.
    _, backward = cx.phase_arguments((complex(-1.0, 0.0), None, None), backward=True)
    assert backward[0][0] == -1.0
    assert np.signbit(np.float32(backward[0][1]))


def test_the_phase_is_rounded_to_complex64_BEFORE_splitting():
    phase = complex(np.exp(2j * np.pi * 0.123456789))
    _, values = cx.phase_arguments((phase, None, None), backward=False)
    rounded = np.complex64(phase)
    assert values[0] == (float(np.float32(rounded.real)),
                         float(np.float32(rounded.imag)))


# ---------------------------------------------------------------------------
# 6. Registration: four arms, WIRED as of tranche 2
# ---------------------------------------------------------------------------

def test_the_family_registers_four_wired_arms():
    """One arm per slot, all four wired — the tranche-2 state.

    They were ``wired=False`` for a tranche and the flip followed the composition
    sweep rather than preceding it. What this pins is the SHAPE: exactly one arm
    per slot from this family, so the family can never be the one that makes a slot
    self-ambiguous.
    """
    for slot in ("step_B", "step_D", "update_H", "update_E"):
        mine = [spec for spec in arms.registered(slot)
                if spec.family == cx.FAMILY]
        assert len(mine) == 1, slot
        assert mine[0].wired is True, slot


def test_a_wired_arm_is_bound_for_selection_on_every_slot_it_claims():
    """``arms_for`` skips unwired rows; every row of this family must survive it."""
    class _Context:
        fields = None
        pml = None
        residency = None
        contract_variants = ()
        sources = ()
        synced = ()
        extra = {}

    for slot in ("step_B", "step_D", "update_H", "update_E"):
        assert any(spec.family == cx.FAMILY for spec in arms.registered(slot))
        bound = arms.arms_for(slot, _Context())
        assert any("complex" in arm.label for arm in bound), slot


def test_the_beta_clause_is_what_keeps_this_family_off_a_beta_run(monkeypatch):
    """The disjointness that had to hold BEFORE this family could be wired.

    Not a reading of clause 8 — the predicate is called on a real complex BETA
    configuration and must refuse, naming beta. If it ever admits, a complex beta
    run becomes ambiguous on all four slots and every one falls to the array path.
    """
    monkeypatch.setenv("MEEP_GPU_SUBNORMAL_POLICY", "flush")
    import numpy as _np

    from meep_gpu.fields import Fields
    from meep_gpu.grid import Grid
    from meep_gpu.pml import PML

    grid = Grid(resolution=10.0, cell_size=(2.0, 1.6, 0.0), dimensions=2,
                courant=0.34, beta=0.33, xp=_np)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_pml_storage()
    pml = PML(grid=grid, thickness=((2, 2), (2, 2), (0, 0)))
    for probe_slot in ("step_B", "step_D"):
        verdict = cx.complex_pml_curl_coverage(fields, pml, probe_slot, None, None)
        assert not verdict.covered
        assert any("beta" in reason for reason in verdict.reasons), verdict.reasons
    for side in ("H", "E"):
        verdict = cx.complex_constitutive_coverage(fields, pml, side, None, None)
        assert not verdict.covered
        assert any("beta" in reason for reason in verdict.reasons), verdict.reasons


def test_registering_the_same_arm_twice_is_refused():
    with pytest.raises(ValueError, match="already registered"):
        cx.register_arms()


# ---------------------------------------------------------------------------
# 7. Over-coverage, CONSTRUCTED — real engine objects through the predicate
# ---------------------------------------------------------------------------
#
# READING A CLAUSE IS NOT EVIDENCE THAT IT FIRES. Section 4 above uses synthetic
# duck objects, which can only exercise the clauses those ducks happen to model.
# These build REAL Grid/Fields/PML objects with exactly one unsupported feature
# installed and assert the refusal by name. Torch is still not needed: a host
# without it simply contributes one extra backend reason, and every assertion here
# is "this reason is present", never "these are the only reasons".

@pytest.fixture(autouse=True)
def _flush_policy(monkeypatch):
    """THE ARM64 DEFAULT RESOLVES TO ``keep``, WHICH MPS CANNOT DELIVER.

    ``match_meep`` measures MEEP's own behaviour and MEEP's ``set_zero_subnormals``
    is a no-op on this host, so the process default is exactly the policy this
    executor has no lever for — and every predicate then refuses on clause 1 no
    matter what else is true. The gate requests ``flush`` explicitly before anything
    resolves a policy; these tests do the same, or the ADMISSION assertions below
    would pass for the wrong reason and the REFUSAL ones would be vacuous.
    """
    monkeypatch.setenv("MEEP_GPU_SUBNORMAL_POLICY", "flush")


def _engine_case(boundaries="periodic", k=(0.3, 0.0, 0.0), cell=(1.2, 1.0, 0.9),
                 complex_storage=True, pml_thickness=None, dimensions=3,
                 **grid_kwargs):
    from meep_gpu.fields import Fields
    from meep_gpu.grid import Grid
    from meep_gpu.pml import PML

    grid = Grid(resolution=10.0, cell_size=cell, boundaries=boundaries,
                dimensions=dimensions, courant=0.35, k_point=k, xp=np, **grid_kwargs)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.enable_pml_storage()
    count = int(np.prod(grid.shape))
    eps = (1.45 + 0.3 * np.sin(np.arange(count, dtype=np.float32) * 0.037)
           ).astype(np.float32).reshape(grid.shape)
    fields.set_isotropic_epsilon_volume(eps, (np.float32(1.0) / eps).astype(np.float32))
    thickness = ((2, 2), (2, 2), (2, 2)) if pml_thickness is None else pml_thickness
    return grid, fields, PML(grid=grid, thickness=thickness)


def _engine_reasons(fields, pml, slot):
    residency = type("_R", (), {"mirror": lambda *a, **k: None})()
    if slot in ("step_B", "step_D"):
        return cx.complex_pml_curl_coverage(fields, pml, slot, residency,
                                            probe=_GOOD)
    side = "H" if slot == "update_H" else "E"
    return cx.complex_constitutive_coverage(fields, pml, side, residency,
                                            probe=_GOOD)


def test_the_engine_baseline_is_ADMITTED_on_all_four_slots():
    """Without this every refusal below could be produced by an unrelated clause."""
    _, fields, pml = _engine_case()
    for slot in ("step_B", "step_D", "update_H", "update_E"):
        verdict = _engine_reasons(fields, pml, slot)
        assert verdict.covered, (slot, verdict.reasons)


def test_a_mirror_fold_is_refused_on_a_REAL_folded_grid():
    _, fields, pml = _engine_case(k=(0.0, 0.0, 0.0), symmetry=("X",),
                                  pml_thickness={"x": {"high": 2}, "y": (2, 2),
                                                 "z": (2, 2)})
    for slot in ("step_B", "update_E"):
        assert any("mirror" in r for r in _engine_reasons(fields, pml, slot).reasons)


def test_the_cylindrical_axis_is_refused_on_a_REAL_Dcyl_grid():
    _, fields, pml = _engine_case(cell=(1.2, 0.0, 0.9), k=(0.0, 0.0, 0.0),
                                  dimensions=2, cylindrical=True,
                                  pml_thickness={"x": (0, 2), "y": (0, 0),
                                                 "z": (2, 2)})
    for slot in ("step_B", "update_E"):
        assert any("cylindrical" in r
                   for r in _engine_reasons(fields, pml, slot).reasons)


@pytest.mark.parametrize("install,needle", [
    ("d_conductivity", "a conductivity is installed"),
    ("b_conductivity", "a conductivity is installed"),
    ("nonlinearity", "chi2/chi3 is installed"),
])
def test_an_installed_feature_is_refused_on_every_slot(install, needle):
    grid, fields, pml = _engine_case()
    volume = np.full(grid.shape, 0.2, dtype=np.float32)
    if install == "d_conductivity":
        fields.set_d_conductivity(volume)
    elif install == "b_conductivity":
        fields.set_b_conductivity(volume)
    else:
        fields.set_nonlinear_volumes({"Ex": volume}, {"Ex": volume})
    for slot in ("step_B", "step_D", "update_H", "update_E"):
        assert any(needle in r for r in _engine_reasons(fields, pml, slot).reasons), (
            slot, needle)


def test_an_offdiagonal_row_splits_the_slots_the_way_stepping_does():
    """S:972-979 — the row product is constitutive-only, and only on the E side."""
    grid, fields, pml = _engine_case()
    count = int(np.prod(grid.shape))
    diag = (1.45 + 0.3 * np.sin(np.arange(count, dtype=np.float32) * 0.037)
            ).astype(np.float32).reshape(grid.shape)
    fields.set_epsilon_volumes(
        {"Ex": diag, "Ey": diag, "Ez": diag},
        {"Ex": np.float32(1.0) / diag, "Ey": np.float32(1.0) / diag,
         "Ez": np.float32(1.0) / diag},
        chi1inv_offdiagonal={"Ex": {"Ey": np.full(grid.shape, 0.05,
                                                  dtype=np.float32)}})
    for slot in ("step_B", "step_D", "update_H"):
        assert _engine_reasons(fields, pml, slot).covered, slot
    refused = _engine_reasons(fields, pml, "update_E")
    assert not refused.covered
    assert any("off-diagonal chi1inv row" in r for r in refused.reasons)


def test_an_inactive_absorber_is_refused_as_no_absorber():
    _, fields, pml = _engine_case(pml_thickness=((0, 0), (0, 0), (0, 0)))
    assert any("no active PML layer" in r
               for r in _engine_reasons(fields, pml, "step_B").reasons)


@pytest.mark.parametrize("break_it,needle", [
    ("dtype", "is not complex64"),
    ("order", "not C-contiguous"),
    ("shape", "!= grid shape"),
    ("missing", "Ex is not allocated"),
])
def test_a_malformed_source_volume_is_refused_by_name(break_it, needle):
    _, fields, pml = _engine_case()
    if break_it == "dtype":
        fields.Ex = fields.Ex.astype(np.complex128)
    elif break_it == "order":
        fields.Ex = np.asfortranarray(fields.Ex)
    elif break_it == "shape":
        fields.Ex = np.zeros((3, 3, 3), dtype=np.complex64)
    else:
        fields.Ex = None
    assert any(needle in r for r in _engine_reasons(fields, pml, "step_B").reasons)


@pytest.mark.parametrize("break_it,needle", [
    ("short", "entries, axis x has"),
    ("float64", "is not float32"),
])
def test_a_malformed_PML_coefficient_vector_is_refused_by_name(break_it, needle):
    _, fields, pml = _engine_case()
    if break_it == "short":
        pml.kms_x_h = np.zeros(3, dtype=np.float32)
    else:
        pml.kms_x_h = pml.kms_x_h.astype(np.float64)
    assert any(needle in r for r in _engine_reasons(fields, pml, "step_B").reasons)


def test_the_GRID_itself_refuses_k_on_a_metallic_axis():
    """The predicate's clause 5 is reachable only from a duck object, and saying so
    is part of the record: a real ``Grid`` will not build the pairing at all."""
    from meep_gpu.grid import Grid

    with pytest.raises(ValueError, match="metallic"):
        Grid(resolution=10.0, cell_size=(1.2, 1.0, 0.9),
             boundaries=("metallic", "periodic", "periodic"), dimensions=3,
             courant=0.35, k_point=(0.5, 0.0, 0.0), xp=np)


# ---------------------------------------------------------------------------
# 8. The gate kit's mutation verdict — a PARTIAL needle miss is a miss
# ---------------------------------------------------------------------------

def _gate_module():
    import sys as _sys

    gate_dir = os.path.abspath(os.path.join(os.path.dirname(PACKAGE_DIR), os.pardir,
                                            "parity", "meep_gpu"))
    if gate_dir not in _sys.path:
        _sys.path.insert(0, gate_dir)
    import gate_metal_complex as gate  # noqa: PLC0415

    return gate


def test_the_diagnostic_arms_are_the_SHIPPED_helper_and_the_check_FIRES():
    """The exhaustive tables' ``shipped`` row must BE the emitted helper.

    Both halves, because a guard never shown to fire is decorative: on the tree as
    it stands the check passes, and with one character of the table perturbed it
    RAISES. Without it the 64- and 256-pattern tables certified a HAND COPY of the
    shader — an edit to ``templates._COMPLEX_HELPERS`` would have left the leg green
    while it measured the previous spelling.
    """
    gate = _gate_module()
    gate.EXPANSION = "FMA_V1"
    checked = gate._assert_diagnostic_arms_are_the_shipped_spelling()
    assert set(checked) == {"field_left_re", "field_left_im", "full_product_re",
                            "full_product_im"}

    original = gate._FULL_PRODUCT_ARMS["shipped"]
    gate._FULL_PRODUCT_ARMS["shipped"] = ("fma(z.x, p.x, -(z.y * p.y) )",
                                          original[1], original[2])
    try:
        with pytest.raises(AssertionError, match="HAND COPY"):
            gate._assert_diagnostic_arms_are_the_shipped_spelling()
    finally:
        gate._FULL_PRODUCT_ARMS["shipped"] = original
    assert gate._assert_diagnostic_arms_are_the_shipped_spelling()


def test_the_admitted_domain_leg_covers_what_the_predicate_admits():
    """The two configurations found ADMITTED and never launched are gate cases now.

    Pinned here rather than only in the gate because the hole they close is a
    PREDICATE/MATRIX drift, and a predicate change that widens the domain again
    should be visible on a laptop.
    """
    gate = _gate_module()
    labels = {row[0] for row in gate.ADMITTED_DOMAIN}
    assert labels == {"two_dimensional_collapsed_z", "absorber_absent_on_y"}
    for _label, kwargs, why in gate.ADMITTED_DOMAIN:
        assert why.strip(), _label
        assert "cell" in kwargs and "k_point" in kwargs
    assert "admitted_domain" in [name for name, _ in gate.LEGS]


def test_a_partially_missed_needle_is_NEEDLE_MISSED():
    """A mutation swept over two sub-steps whose needle matched in only one used to
    report ``CAUGHT 1/1`` and pass, silently dropping the half that was never
    planted. ``verdict`` now reads ANY miss as a miss."""
    import sys as _sys

    kit_dir = os.path.join(os.path.dirname(PACKAGE_DIR), os.pardir, "parity",
                           "meep_gpu")
    kit_dir = os.path.abspath(kit_dir)
    if kit_dir not in _sys.path:
        _sys.path.insert(0, kit_dir)
    import metal_gate_kit as kit

    assert kit.MutationHarness.verdict(True, 0, 0, 0) == "NEEDLE-MISSED"
    assert kit.MutationHarness.verdict(True, 1, 1, 1) == "NEEDLE-MISSED"
    assert kit.MutationHarness.verdict(False, 2, 0, 0) == "DISARMED"
    assert kit.MutationHarness.verdict(False, 2, 2, 2) == "CAUGHT 2/2"
