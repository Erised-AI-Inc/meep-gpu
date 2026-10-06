"""The complex/no-absorber hand-CUDA family: its bytes, its clauses, its hazards.

WHAT RUNS HERE AND WHAT IS OWED. A kernel has to be launched to be compared, and
this host has no CuPy, so the bit-identity question belongs to
``parity/meep_gpu/gate_cuda_complex_no_pml.py`` and its numbers land in
:data:`complex_no_pml_kernels.COMPLEX_NO_PML_ADMISSION`.
``test_admission_record_is_all_or_nothing`` refuses a half-filled record, so a
green suite here can never be mistaken for a device verdict.

What DOES run here is everything whose failure mode is a silent wrong answer that
no device would report:

1. **The curl is the certified one.** This family builds its four curl kernels by
   substituting a signature and three tail calls into
   ``complex_emitter._CURL_TEMPLATE``. The tests below re-derive the stencil,
   the ghost helper, the phase arguments and the ownership masks from
   ``complex_emitter``'s own tables and assert the emitted text carries exactly
   them -- so "nothing in the arithmetic moved" is checked rather than reviewed.

2. **The three arithmetic hazards, MEASURED on this host.** Each of the three
   claims the module makes about float32 -- the conductive order, the pole
   subtraction order, the zero-imaginary product on signed zeros -- is recomputed
   here in NumPy and its word count printed. A claim about rounding that no test
   evaluates is a claim, not a measurement.

3. **The predicate verdict table**, on REAL ``Grid``/``Fields``/``PML`` objects
   rather than dictionaries, because a dictionary answers the way the test author
   expected and the engine answers the way the engine does.

4. **The partition** -- no slot may be admitted by this family and by a shipped
   sibling at once. Two admitters leave a slot unselected and it falls back to
   the array path: a silent coverage LOSS, not an error.

5. **The platform requirements** -- ASCII, no division, no unary minus on a float
   path -- which killed sibling kernels at their first launch and are cheap to
   pin.
"""

from __future__ import annotations

import json
import pathlib
import re

import numpy
import pytest

from .. import stepping
from ..dispersion import PolarizationState, Susceptibility
from ..fields import Fields
from ..grid import Grid
from ..pml import PML
from . import complex_emitter, coverage
from . import complex_no_pml_kernels as family
from .test_certification_metadata import RELEASE_HOST_PLACEHOLDER, _artifact_stamps

HERE = pathlib.Path(__file__).parent
MODULE_PATH = HERE / "complex_no_pml_kernels.py"
STEPPING_PATH = HERE.parent / "stepping.py"

ARMS = ("NAIVE", "FMA_V1")
CURL_KEYS = ("step_B", "step_D", "step_B_conductive", "step_D_conductive")


class _NumpyWearingCupysName:
    """NumPy behind CuPy's ``__name__``, with NumPy's own dtype objects.

    The dtype identity matters: ``xp.complex64`` has to be the object the arrays
    actually carry, or a dtype clause tests the stand-in instead of the array.
    """

    __name__ = "cupy"

    def __getattr__(self, item):
        return getattr(numpy, item)


XP = _NumpyWearingCupysName()

#: A refusal-free licence for the four-pattern probe every complex family shares.
FOUR_PATTERN_LICENCE = {
    "arm": "FMA_V1", "expansion": 1, "basis": "measured",
    "policy_resolved": "keep",
    "refusals": (),
    "patterns": {"c8_mul_c8": "FMA_V1", "c8_mul_f4_field_left": "FMA_V1",
                 "f4_mul_c8_coefficient_left": "FMA_V1",
                 "python_float_left": "FMA_V1"},
}

#: The same licence over the SEVEN-pattern record, which classifies the
#: orientation ``update_P`` needs.
SEVEN_PATTERN_LICENCE = dict(
    FOUR_PATTERN_LICENCE,
    patterns=dict(FOUR_PATTERN_LICENCE["patterns"],
                  **{family.ADE_PROBE_PATTERN: "FMA_V1",
                     "c8_mul_c8_imaginary_coefficient_left": "AMBIGUOUS_BOTH",
                     "c8_mul_c8_parity_coefficient_left": "AMBIGUOUS_BOTH"}))

POLICY = "keep"


# ---------------------------------------------------------------------------
# Fixtures: REAL engine objects
# ---------------------------------------------------------------------------

def build(*, complex_storage=True, k_point=(0.2, 0.0, 0.0), storage=True,
          pml_thickness=None, pml_storage=False, conductivity=None,
          poles=0, offdiagonal=False, cell=(6.0, 6.0, 6.0), xp=XP):
    """A ``(fields, pml, grid)`` triple in this family's shape, or beside it."""
    grid = Grid(resolution=1.0, cell_size=cell, boundaries=("periodic",) * 3,
                xp=xp, k_point=k_point)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    if storage:
        fields.enable_field_storage()
    if pml_storage:
        fields.enable_pml_storage()
    if conductivity is not None:
        fields.set_d_conductivity(xp.full(grid.shape, numpy.float32(conductivity)))
        fields.set_b_conductivity(xp.full(grid.shape, numpy.float32(conductivity)))
    for _ in range(poles):
        fields.polarizations.append(_electric_state(grid, fields))
    if offdiagonal:
        fields.set_epsilon_volumes(
            {c: xp.full(grid.shape, numpy.float32(2.25))
             for c in ("Ex", "Ey", "Ez")},
            {c: xp.full(grid.shape, numpy.float32(1 / 2.25))
             for c in ("Ex", "Ey", "Ez")},
            {"Ex": {"Ey": xp.full(grid.shape, numpy.float32(0.1))}})
    layer = None if pml_thickness is None else PML(grid=grid,
                                                   thickness=pml_thickness)
    return fields, layer, grid


def _electric_state(grid, fields, sigma=0.35):
    return PolarizationState(
        Susceptibility(frequency=1.1, gamma=0.05),
        grid.xp.full(grid.shape, numpy.float32(sigma)),
        grid, fields._field_dtype())


@pytest.fixture()
def shipped():
    """The family's own shape: complex, inert layer, conductivity, one pole."""
    return build(conductivity=0.2, poles=1)


# ===========================================================================
# 1. THE CURL IS THE CERTIFIED ONE
# ===========================================================================

@pytest.mark.parametrize("key", CURL_KEYS)
@pytest.mark.parametrize("arm", ARMS)
def test_the_curl_stencil_is_the_shared_templates_own_text(key, arm):
    """Every stencil line in the emitted curl is ``complex_emitter``'s, verbatim.

    Not a spot check: the three per-target blocks of the shared template are
    extracted from the SIBLING and each is required to appear in the emitted
    source with only its tail line replaced. A stencil that had drifted -- a
    swapped source volume, a shifted axis, a lost phase argument -- is a smooth,
    converged, entirely wrong field that no compile error would report.
    """
    sub_step, conductive = family.CURL_KERNELS[key]
    emitted = family.kernel_source(key, arm)
    backward = complex_emitter.KERNELS[sub_step][1]
    shift = "cshift_dn" if backward else "cshift_up"

    reference = complex_emitter.complex_source(sub_step, arm)
    # The four stencil lines of each of the three target blocks, taken from the
    # certified source rather than restated here.
    blocks = re.findall(
        r"\n    \{\n(        cf f_1 = .*?)\n        cf curl = ", reference,
        flags=re.S)
    assert len(blocks) == 3, blocks
    for block in blocks:
        assert block in emitted, (
            f"a certified stencil block is missing from {key}/{arm}:\n{block}")
    assert emitted.count(
        "        cf curl = mul_coefficient_left(dtdx, "
        "cf_add(cf_sub(sf, f_1), cf_sub(f_2, ss)));") == 3
    body = emitted[emitted.index('extern "C" __global__ void'):]
    other = "cshift_up" if backward else "cshift_dn"
    assert body.count(shift + "(") == 6, "six ghost reads, all one direction"
    assert other + "(" not in body, "the opposite ghost rule reached the body"


@pytest.mark.parametrize("key", CURL_KEYS)
@pytest.mark.parametrize("arm", ARMS)
def test_the_ownership_masks_are_the_shared_tables(key, arm):
    """One ``curl = cf_zero()`` guard per masked axis, from ``_MASK_AXES``.

    ``fields.IYEE_SHIFTS`` gives each B target ONE masked axis and each D target
    TWO; getting that asymmetry backwards steps a plane MEEP does not own.
    """
    sub_step, _ = family.CURL_KERNELS[key]
    backward = complex_emitter.KERNELS[sub_step][1]
    emitted = family.kernel_source(key, arm)
    expected = sum(len(axes)
                   for axes in complex_emitter._MASK_AXES[backward])
    assert emitted.count("curl = cf_zero();") == expected
    for axes in complex_emitter._MASK_AXES[backward]:
        assert complex_emitter._mask_lines(axes) in emitted


@pytest.mark.parametrize("key", CURL_KEYS)
@pytest.mark.parametrize("arm", ARMS)
def test_no_split_field_token_survives_into_a_no_absorber_curl(key, arm):
    """No ``pml_apply`` call, no ``kms``/``sinv`` index, no ``fu`` pointer.

    The whole safety argument for building this family out of the split-field
    template is that the tail is replaced at every site. One missed site would
    step the absorber recurrence with whatever coefficients happened to be bound.
    """
    emitted = family.kernel_source(key, arm)
    body = emitted[emitted.index('extern "C" __global__ void'):]
    for pattern in family._FORBIDDEN_IN_CURL_BODY:
        found = re.search(pattern, body)
        assert found is None, f"{found.group(0)!r} survived into {key}/{arm}"


@pytest.mark.parametrize("arm", ARMS)
def test_the_dead_prelude_helpers_are_carried_whole(arm):
    """``pml_apply`` and ``constitutive_apply`` are compiled in and never called.

    Carried rather than forked out, for the reason ``no_pml_curl`` carries its
    dead ``pml_apply``: a fork is two sets of bytes that are equal only until
    someone edits one. Their presence is what gives the gate a site to arm four
    MUST-BE-UNCAUGHT nulls at, and their silence there is the measurement that
    this family really does not route through those recurrences.
    """
    for key in family.KERNEL_KEYS:
        source = family.kernel_source(key, arm)
        assert "__device__ __forceinline__ void pml_apply(" in source
        assert "__device__ __forceinline__ void constitutive_apply(" in source
        body = source[source.index('extern "C" __global__ void'):]
        assert re.search(r"(?<![A-Za-z0-9_])pml_apply\(", body) is None
        assert re.search(r"(?<![A-Za-z0-9_])constitutive_apply\(", body) is None


@pytest.mark.parametrize("arm", ARMS)
def test_the_conductive_arm_differs_from_the_plain_one_only_in_its_tail(arm):
    """Same stencil, same masks, same phases; two extra parameter triples and a tail.

    Pinned as a DIFF rather than as two independent readings, because the claim
    the family rests on is that the two arms share one curl. Both sources are
    reduced by deleting the tail lines and the conductive parameter block; what
    remains must be identical but for the kernel name.
    """
    for sub_step in ("step_B", "step_D"):
        plain = family.kernel_source(sub_step, arm)
        cond = family.kernel_source(sub_step + "_conductive", arm)
        # Counted in the KERNEL BODY: both helpers are also DECLARED in the
        # prelude, and "no_pml_apply(f" matches "no_pml_apply(float*".
        plain_body = plain[plain.index('extern "C" __global__ void'):]
        cond_body = cond[cond.index('extern "C" __global__ void'):]
        assert plain_body.count("no_pml_apply(f") == 3
        assert cond_body.count("conductive_apply(f") == 3
        assert "condfac_0[idx]" in cond_body and "condfac" not in plain_body

        def reduce(text, name):
            text = text.replace(name, "THE_KERNEL")
            text = re.sub(r"\n *(no_pml_apply|conductive_apply)\(f\d[^;]*;", "",
                          text)
            return text.replace(family._CURL_CONDUCTIVE_PARAMS, "")

        assert reduce(plain, family.kernel_name(sub_step)) == \
            reduce(cond, family.kernel_name(sub_step + "_conductive"))



def test_the_builder_refuses_a_template_it_can_no_longer_recognise(monkeypatch):
    """A renamed or restructured shared template fails HERE, loudly, at build.

    The alternative is a family that silently stops sharing the certified curl,
    which is the failure this whole construction exists to make impossible.
    """
    monkeypatch.setattr(complex_emitter, "_CURL_TEMPLATE",
                        complex_emitter._CURL_TEMPLATE.replace("pml_apply", "zzz"))
    with pytest.raises(RuntimeError, match="three pml_apply tail sites"):
        family.kernel_source("step_B", "FMA_V1")


# ===========================================================================
# 2. THE THREE ARITHMETIC HAZARDS, MEASURED HERE
# ===========================================================================

def test_the_conductive_order_is_load_bearing_and_the_spelling_is_not(capsys):
    """MEASURED, both halves, because only one of them is a hazard.

    The module says the ORDER of ``*= condfac; -= curl; *= condinv`` decides bytes
    and that writing the three as one nested C expression does not. Both are
    recomputed here rather than asserted, and both counts are printed.
    """
    rng = numpy.random.default_rng(7)
    n = 200_000
    f = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(numpy.complex64)
    curl = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(numpy.complex64)
    condfac = rng.random(n).astype(numpy.float32)
    condinv = rng.random(n).astype(numpy.float32)

    reference = f.copy()
    reference *= condfac          # stepping.py:1996
    reference -= curl             # stepping.py:1997
    reference *= condinv          # stepping.py:1998

    nested = ((f * condfac) - curl) * condinv
    reordered = ((f - curl) * condfac) * condinv

    same_tree = int(numpy.sum(
        reference.view(numpy.uint32) != nested.view(numpy.uint32)))
    swapped = int(numpy.sum(
        reference.view(numpy.uint32) != reordered.view(numpy.uint32)))
    words = reference.view(numpy.uint32).size
    with capsys.disabled():
        print(f"\n  conductive tail: one nested expression differs on "
              f"{same_tree}/{words} words; the reordered form on "
              f"{swapped}/{words}", flush=True)
    assert same_tree == 0, (
        f"the nested spelling is NOT the same expression tree after all "
        f"({same_tree}/{words} differ); the module says it is")
    assert swapped == words, (
        f"reassociating the conductive tail changed only {swapped}/{words} "
        f"words; the order clause rests on it changing them")
    # And the shipped device text carries the array path's order.
    source = family.kernel_source("step_B_conductive", "FMA_V1")
    helper = source[source.index("void conductive_apply("):]
    helper = helper[helper.index(") {"):]
    helper = helper[:helper.index("\n}")]
    assert helper.index("condfac") < helper.index("cf_sub") < helper.index("condinv"), (
        f"conductive_apply does not multiply, subtract, multiply in that "
        f"order:\n{helper}")


def test_the_poles_may_not_be_pre_summed(capsys):
    """``((D - P0) - P1)`` against ``D - (P0 + P1)``, measured in float32."""
    rng = numpy.random.default_rng(11)
    n = 200_000
    D = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(numpy.complex64)
    P0 = (rng.standard_normal(n) * 1e-3).astype(numpy.complex64)
    P1 = (rng.standard_normal(n) * 1e-3).astype(numpy.complex64)
    sequential = (D - P0) - P1
    presummed = D - (P0 + P1)
    differing = int(numpy.sum(
        sequential.view(numpy.uint32) != presummed.view(numpy.uint32)))
    words = sequential.view(numpy.uint32).size
    with capsys.disabled():
        print(f"  pole order: pre-summing differs on {differing}/{words} words",
              flush=True)
    assert differing > 0, (
        "pre-summing two poles changed no word on this fixture, so the "
        "registration-order clause has no measured basis here")
    source = family.kernel_source("update_E", "FMA_V1")
    helper = source[source.index("cf minus_poles("):]
    helper = helper[:helper.index("return s;")]
    assert helper.count("cf_sub(s, cf_load(") == family.MAX_POLES, (
        "minus_poles must subtract one pole at a time")


def test_the_real_coefficient_multiply_is_the_full_complex_product(capsys):
    """Plane-wise ``{re*c, im*c}`` is byte-wrong on signed zeros; measured.

    This is ``complex_emitter``'s fact, re-measured for the two coefficient sites
    this family adds (``condfac``, ``condinv``): they are ordinary float32
    volumes, so the natural implementation is to scale the two planes, and it is
    the wrong one.
    """
    patterns = numpy.array(
        [complex(0.0, -0.0), complex(-0.0, 0.0), complex(-0.0, -0.0),
         complex(0.0, 0.0)], dtype=numpy.complex64)
    c = numpy.full(patterns.shape, numpy.float32(1.5))
    full = patterns * c
    plane = (patterns.real * c + 1j * (patterns.imag * c)).astype(numpy.complex64)
    differing = int(numpy.sum(
        full.view(numpy.uint32) != plane.view(numpy.uint32)))
    with capsys.disabled():
        print(f"  zero-imaginary product: plane-wise differs on {differing}/"
              f"{full.view(numpy.uint32).size} words", flush=True)
    assert differing > 0
    source = family.kernel_source("step_B_conductive", "FMA_V1")
    assert "mul_field_left(cf_load(f, idx), condfac)" in source
    assert "mul_field_left(t, condinv)" in source


# ===========================================================================
# 3. THE TRANSCRIPTIONS ARE STILL WHERE THIS FILE SAYS THEY ARE
# ===========================================================================

@pytest.mark.parametrize("line,expected", [
    (539, "target -= curl"),
    (1996, "field *= condfac"),
    (1997, "field -= curl"),
    (1998, "field *= condinv"),
    (1022, "getattr(fields, component)[...] = constitutive"),
])
def test_every_cited_stepping_line_still_says_what_is_cited(line, expected):
    """A citation that has drifted into fiction is the defect rule 3 forbids.

    Checkable rather than reviewable: the cited line number is read back out of
    ``stepping.py`` and required to carry the cited text.
    """
    text = STEPPING_PATH.read_text(encoding="utf-8").splitlines()
    assert expected in text[line - 1], (
        f"stepping.py:{line} now reads {text[line - 1]!r}, not {expected!r}")


def test_the_array_path_tail_this_family_replaces_is_still_the_one_transcribed():
    """``_apply_conductive_update`` is still three statements in that order."""
    body = stepping._apply_conductive_update.__doc__
    assert "f[i] = ((1 - dt/2*cnd[i]) * f[i] - dtdx * curl) * cndinv[i]" in body
    source = STEPPING_PATH.read_text(encoding="utf-8")
    block = source[source.index("def _apply_conductive_update("):]
    block = block[:block.index("\ndef ", 1)]
    statements = [line.strip() for line in block.splitlines()
                  if line.strip().startswith("field")]
    assert statements == ["field *= condfac", "field -= curl", "field *= condinv"]


def test_max_poles_and_the_probe_pattern_match_the_triton_siblings():
    """Two tables restated in this package; both pinned against their originals.

    A corpus row with nine poles must be refused by the same number on both
    tracks, and the fifth expansion orientation must be spelled identically or
    a probe record classifying it would not be recognised.
    """
    triton_stored_e = (HERE.parent / "triton_kernels" /
                       "complex_no_pml_stored_e.py").read_text(encoding="utf-8")
    assert f"MAX_POLES = {family.MAX_POLES}" in triton_stored_e
    triton_ade = (HERE.parent / "triton_kernels" /
                  "complex_ade.py").read_text(encoding="utf-8")
    assert (f'COMPLEX_ADE_PROBE_PATTERN = "{family.ADE_PROBE_PATTERN}"'
            in triton_ade)


# ===========================================================================
# 4. THE PREDICATE VERDICT TABLE
# ===========================================================================

def _curl(fields, pml, grid, sub_step, licence=None):
    return family.covers_complex_no_pml_curl(
        fields, pml, grid, sub_step,
        license=FOUR_PATTERN_LICENCE if licence is None else licence,
        subnormal_policy=POLICY)


@pytest.mark.parametrize("sub_step", ("step_B", "step_D"))
def test_the_shipped_shape_is_admitted_at_both_curls(shipped, sub_step):
    fields, pml, grid = shipped
    covered, reason = _curl(fields, pml, grid, sub_step)
    assert covered, reason
    assert family.complex_no_pml_curl_arm(fields, sub_step) == "conductive"


@pytest.mark.parametrize("sub_step", ("step_B", "step_D"))
def test_a_lossless_complex_run_takes_the_plain_arm(sub_step):
    fields, pml, grid = build()
    covered, reason = _curl(fields, pml, grid, sub_step)
    assert covered, reason
    assert family.complex_no_pml_curl_arm(fields, sub_step) == "plain"


def test_an_active_absorber_is_refused_at_both_curls():
    """The partition clause. An active layer is the split-field family's."""
    fields, pml, grid = build(pml_thickness=2, pml_storage=True)
    for sub_step in ("step_B", "step_D"):
        covered, reason = _curl(fields, pml, grid, sub_step)
        assert not covered
        assert reason.startswith("an active absorber is installed")


def test_an_inert_layer_object_is_admitted_not_refused():
    """``PML(thickness=0)`` is INACTIVE, and a run carrying one belongs here.

    ``PML.is_active`` is ``any(low > 0 or high > 0)``, so a zero-thickness table
    steps bit-identically to ``pml=None``. A predicate that keyed on the object's
    presence rather than on ``is_active`` would give this run to nobody.
    """
    fields, pml, grid = build(pml_thickness=0)
    assert pml is not None and not pml.is_active
    for sub_step in ("step_B", "step_D"):
        covered, reason = _curl(fields, pml, grid, sub_step)
        assert covered, reason


def test_pml_storage_mode_under_an_inert_layer_is_refused():
    """``get_H`` would serve a stored H that ``update_H`` never writes."""
    fields, pml, grid = build(pml_storage=True, pml_thickness=0)
    covered, reason = _curl(fields, pml, grid, "step_D")
    assert not covered
    assert reason.startswith("Fields is in PML storage mode")


def test_real_storage_is_refused_by_the_inverted_clause():
    fields, pml, grid = build(complex_storage=False, k_point=(0.0, 0.0, 0.0))
    covered, reason = _curl(fields, pml, grid, "step_B")
    assert not covered
    assert "real float32 storage" in reason


def test_a_mixed_conductivity_sub_step_is_refused_rather_than_guessed():
    """One lossy target and two lossless is two tails inside one sub-step.

    Built by removing one component's table from the map the engine filled, which
    is the only way to reach a state ``set_d_conductivity`` will not produce.
    """
    fields, pml, grid = build(conductivity=0.2)
    fields._condfac.pop("Dy")
    covered, reason = _curl(fields, pml, grid, "step_D")
    assert not covered
    assert "disagree about conductivity" in reason
    with pytest.raises(ValueError, match="disagree about conductivity"):
        family.complex_no_pml_curl_arm(fields, "step_D")


def test_an_allocated_auxiliary_under_an_inert_layer_is_refused():
    """``fu_*`` allocated means something else built it and this step would stale it."""
    fields, pml, grid = build()
    fields.fu_Bx = grid.xp.zeros(grid.shape, dtype=numpy.complex64)
    covered, reason = _curl(fields, pml, grid, "step_B")
    assert not covered
    assert reason.startswith("fu_Bx is allocated")


def test_a_conductive_run_missing_its_condinv_table_is_refused():
    fields, pml, grid = build(conductivity=0.2)
    fields._condinv.pop("Bz")
    covered, reason = _curl(fields, pml, grid, "step_B")
    assert not covered
    assert "condinv for Bz" in reason


def test_the_curl_admits_dispersion_and_names_why(shipped):
    """The certified complex twin refuses it; the premise of that refusal is gone.

    The twin's reason is that a curl admitted into a step whose ``update_E`` and
    ``update_P`` are both refused buys a sub-step nothing can compose with. This
    family builds both, and the curl reads no ``P``.
    """
    fields, pml, grid = shipped
    assert fields.polarizations
    covered, _ = _curl(fields, pml, grid, "step_B")
    assert covered
    twin_covered, twin_reason = coverage.covers_real_pml_complex_curl(
        fields, pml, grid, "step_B", license=FOUR_PATTERN_LICENCE,
        subnormal_policy=POLICY)
    assert not twin_covered
    assert twin_reason  # and it is the layer, not the dispersion, that refuses


# --- update_E --------------------------------------------------------------

def _stored_e(fields, pml, grid, licence=None):
    return family.covers_complex_no_pml_stored_e(
        fields, pml, grid,
        license=FOUR_PATTERN_LICENCE if licence is None else licence,
        subnormal_policy=POLICY)


def test_update_E_is_admitted_on_the_shipped_shape(shipped):
    fields, pml, grid = shipped
    covered, reason = _stored_e(fields, pml, grid)
    assert covered, reason


def test_update_E_is_admitted_with_no_pole_at_all():
    """``stores_E`` True and no susceptibility still WRITES ``E = D * inv_eps``.

    ``update_E`` returns at stepping.py:983 only when ``stores_E`` is ALSO False.
    Refusing a pole-free run would give a real store to nobody.
    """
    fields, pml, grid = build()
    covered, reason = _stored_e(fields, pml, grid)
    assert covered, reason
    assert all(count == 0 for count in
               (len(v) for v in family.poles_per_component(fields).values()))


def test_update_E_refuses_the_off_diagonal_row_by_name():
    """The two slots this family does not build, refused rather than mis-served."""
    fields, pml, grid = build(offdiagonal=True)
    covered, reason = _stored_e(fields, pml, grid)
    assert not covered
    assert reason.startswith("off-diagonal chi1inv")
    assert family.WHAT_IS_NOT_BUILT["slots"] == 2


def test_update_E_refuses_a_run_with_no_stored_E():
    """That is ``no_pml_constitutive``'s NULL arm; a launch there is a wrong answer."""
    fields, pml, grid = build(storage=False)
    assert not fields.stores_E
    covered, reason = _stored_e(fields, pml, grid)
    assert not covered
    assert "E is recomputed from D rather than stored" in reason


def test_update_E_refuses_an_allocated_f_w():
    fields, pml, grid = build()
    fields.f_w_Ex = grid.xp.zeros(grid.shape, dtype=numpy.complex64)
    covered, reason = _stored_e(fields, pml, grid)
    assert not covered
    assert reason.startswith("f_w_Ex is allocated")


def test_update_E_refuses_more_poles_than_the_kernel_has_slots():
    fields, pml, grid = build(poles=family.MAX_POLES + 1)
    covered, reason = _stored_e(fields, pml, grid)
    assert not covered
    assert f"more than the kernel's MAX_POLES={family.MAX_POLES}" in reason


# --- update_P --------------------------------------------------------------

def _ade(fields, pml, grid, licence):
    return family.covers_complex_no_pml_ade_update_p(
        fields, pml, grid, license=licence, subnormal_policy=POLICY)


def test_update_P_refuses_the_four_pattern_probe_by_name(shipped):
    """The fifth orientation is not measured there, and may not be inferred.

    ``xp.multiply(P, c_now)`` is a complex64 array times a Python float. The
    four-pattern record classifies an array coefficient and a left scalar and
    neither is this.
    """
    fields, pml, grid = shipped
    covered, reason = _ade(fields, pml, grid, FOUR_PATTERN_LICENCE)
    assert not covered
    assert family.ADE_PROBE_PATTERN in reason


def test_update_P_is_admitted_under_the_seven_pattern_probe(shipped):
    fields, pml, grid = shipped
    covered, reason = _ade(fields, pml, grid, SEVEN_PATTERN_LICENCE)
    assert covered, reason


def test_update_P_refuses_a_probe_that_classifies_the_fifth_pattern_differently(
        shipped):
    """One kernel cannot be compiled under two arms."""
    fields, pml, grid = shipped
    licence = dict(SEVEN_PATTERN_LICENCE,
                   patterns=dict(SEVEN_PATTERN_LICENCE["patterns"],
                                 **{family.ADE_PROBE_PATTERN: "NAIVE"}))
    covered, reason = _ade(fields, pml, grid, licence)
    assert not covered
    assert "one kernel cannot be compiled under two arms" in reason


def test_update_P_refuses_a_run_with_no_polarization():
    fields, pml, grid = build()
    covered, reason = _ade(fields, pml, grid, SEVEN_PATTERN_LICENCE)
    assert not covered
    assert reason == "no polarization is registered; update_P is a no-op"


def test_update_P_pins_the_drive_identity_rather_than_its_existence(shipped):
    """Without a layer the drive MUST be the stored E, and that is checked as ``is``.

    ``Fields.drive_field`` returns ``f_w_<c>`` under an active layer and the
    stored E without one; the two agree exactly outside an absorber, so an
    existence check would admit a ``Fields`` whose storage mode disagreed with its
    layer -- the state that produces the wrong pointer.
    """
    fields, pml, grid = shipped
    assert fields.drive_field("Ex") is fields.Ex
    fields.f_w_Ex = grid.xp.zeros(grid.shape, dtype=numpy.complex64)
    fields._pml_active = True
    covered, reason = _ade(fields, pml, grid, SEVEN_PATTERN_LICENCE)
    assert not covered
    # The PML-storage clause fires first, which is the right refusal to name.
    assert "PML storage mode" in reason


# ===========================================================================
# 5. THE PARTITION
# ===========================================================================

SIBLING_CURLS = (
    ("cuda_complex", lambda f, p, g, s: coverage.covers_real_pml_complex_curl(
        f, p, g, s, license=FOUR_PATTERN_LICENCE, subnormal_policy=POLICY)),
    ("cuda_curl", lambda f, p, g, s: coverage.covers_real_pml_curl(f, p, g, s)),
)


@pytest.mark.parametrize("sub_step", ("step_B", "step_D"))
@pytest.mark.parametrize("name,sibling", SIBLING_CURLS,
                         ids=[entry[0] for entry in SIBLING_CURLS])
def test_no_curl_slot_is_admitted_by_this_family_and_a_sibling(
        sub_step, name, sibling):
    """Two admitters leave a slot UNSELECTED: a silent coverage loss, not an error.

    Swept over every configuration this file builds, admitted or refused, because
    the hazard is an overlap anywhere and not an overlap on the shipped shape.
    """
    configurations = {
        "shipped": build(conductivity=0.2, poles=1),
        "lossless": build(),
        "inert_layer": build(pml_thickness=0),
        "active_layer": build(pml_thickness=2, pml_storage=True),
        "real_storage": build(complex_storage=False, k_point=(0.0, 0.0, 0.0)),
        "no_stored_E": build(storage=False),
    }
    for label, (fields, pml, grid) in configurations.items():
        mine, _ = _curl(fields, pml, grid, sub_step)
        theirs, _ = sibling(fields, pml, grid, sub_step)
        assert not (mine and theirs), (
            f"{label}/{sub_step} is admitted by this family AND by {name}")


def test_update_E_never_overlaps_the_no_pml_null_arm():
    """The null arm covers ``update_E`` only where the array path RETURNS."""
    from .no_pml_constitutive import covers_no_pml_null_constitutive as null

    for label, (fields, pml, grid) in {
            "stored": build(), "unstored": build(storage=False),
            "shipped": build(conductivity=0.2, poles=1)}.items():
        mine, _ = _stored_e(fields, pml, grid)
        theirs, _ = null(fields, pml, grid, "E")
        assert not (mine and theirs), label


def test_update_P_never_overlaps_the_real_ade_family(shipped):
    fields, pml, grid = shipped
    mine, _ = _ade(fields, pml, grid, SEVEN_PATTERN_LICENCE)
    theirs, _ = coverage.covers_real_pml_ade_update_p(fields, pml, grid)
    assert mine and not theirs


# ===========================================================================
# 6. PLATFORM REQUIREMENTS AND THE RECORD
# ===========================================================================

@pytest.mark.parametrize("arm", ARMS)
def test_every_device_string_is_pure_ascii(arm):
    """NVRTC gets the source through a locale-encoded ``open``; two em-dashes in a
    comment killed a sibling kernel at its first launch."""
    for key in family.KERNEL_KEYS:
        source = family.kernel_source(key, arm)
        source.encode("ascii")  # raises on anything else


@pytest.mark.parametrize("arm", ARMS)
def test_no_division_and_no_unary_minus_on_a_float_path(arm):
    """Two platform facts, pinned so an edit cannot reintroduce either.

    Division: ptxas expands ``div.rn.f32`` into a sequence whose range checks
    carry ``.FTZ`` in SASS regardless of the PTX modifier. Unary minus: Triton
    lowers ``-x`` as ``0.0 - x`` and canonicalizes ``-0.0``; CUDA does not, so
    ``* -1.0f`` is kept only so the two bodies read as one transcription -- but an
    edit that introduced ``-x`` here would make them diverge on a later port.
    """
    for key in family.KERNEL_KEYS:
        source = family.kernel_source(key, arm)
        code = "\n".join(line.split("//")[0] for line in source.splitlines())
        # Every division that survives must be INTEGER index arithmetic. The one
        # PTX exception on record is ptxas expanding div.rn.f32 into a sequence
        # whose range checks carry .FTZ in SASS whatever the PTX modifier says;
        # int division cannot reach it, float division must not appear.
        divisions = re.findall(r"[^\s;{}()]+\s*/\s*[^\s;{}()]+", code)
        assert set(divisions) <= {"idx / nz", "idx /", "(idx / nz) % ny",
                                  "idx / (ny"}, (key, sorted(set(divisions)))
        assert re.search(r"[=(,]\s*-[A-Za-z_]", code) is None, key


def test_the_shipped_kernels_partition_into_certified_and_uncertified():
    """A kernel in neither set would ship, be wire-able, and be described by nothing."""
    shipped_names = family.shipped_kernel_names()
    certified = set(family.CERTIFIED_KERNELS)
    dead = set(family.UNCERTIFIED_KERNELS)
    assert not certified & dead
    assert certified | dead == shipped_names, {
        "in neither set": sorted(shipped_names - certified - dead),
        "named but not shipped": sorted((certified | dead) - shipped_names)}
    assert shipped_names == set(family.KERNEL_KEYS.values())


def test_admission_record_is_all_or_nothing():
    """Half a device record is worse than none: it reads as a verdict.

    Either every measured field is filled or every one is None. Nothing in the
    module may be read as a device verdict while ``host`` is None.
    """
    record = family.COMPLEX_NO_PML_ADMISSION
    measured = ("artifacts", "recorded_utc", "host", "device", "cases_scored",
                "single_launch_identical", "multi_step_identical",
                "source_mutations_caught", "source_mutations_null_confirmed",
                "source_mutations_escaped",
                "verdict_flips_against_planted_defect",
                "slots_before", "slots_after", "slots_gained")
    filled = [name for name in measured if record[name] is not None]
    assert filled in ([], list(measured)), {
        "filled": filled,
        "empty": [name for name in measured if record[name] is None]}


def test_the_recon_artifact_this_family_was_designed_from_is_on_disk():
    """The six-row measurement is the whole basis for the kernel split.

    A design read off a projection rather than off the objects is what the brief's
    own numbers got wrong; the artifact is checked in so a later round can see the
    same facts.
    """
    root = (HERE.parents[1] / "parity" / "meep_gpu" / "results"
            / "cuda_complex_no_pml_recon_2026-08-20" / "facts")
    if not root.is_dir():
        pytest.skip(f"no recon artifact on this host: {root}")
    rows = sorted(path.stem for path in root.glob("*.json"))
    assert len(rows) == 6, rows
    conductive = 0
    offdiagonal = 0
    for path in root.glob("*.json"):
        facts = json.loads(path.read_text(encoding="utf-8"))
        assert facts["fields"]["force_complex_fields"] is True
        assert facts["pml"]["is_none"] is True
        # THE MEASUREMENT THAT KILLED THE DERIVED-E ARM before it was written.
        assert facts["fields"]["stores_E"] is True
        assert facts["fields"]["Ex"] is not None
        conductive += bool(facts["fields"]["has_conductivity"])
        offdiagonal += bool(facts["fields"]["has_offdiagonal_epsilon"])
    assert conductive == 4, "four rows route the curl to the conductive tail"
    assert offdiagonal == 2, "two rows need the tensor row product this file refuses"


def test_the_corpus_digest_is_stable_across_calls():
    """One sha256 over all fourteen sources; a gate artifact quotes it."""
    assert family.corpus_digest() == family.corpus_digest()
    assert len(family.corpus_digest()) == 64


def test_the_mutation_seam_is_read_on_every_call_not_memoized():
    """A source memoized at first call hands back pre-mutation bytes forever.

    That is "a leg reporting a pass for a mutation it never applied", which the
    sibling track hit three times.
    """
    before = family.kernel_source("step_B", "FMA_V1")
    family.set_kernel_source("step_B", "FMA_V1", "// mutated\n" + before)
    try:
        assert family.kernel_source("step_B", "FMA_V1").startswith("// mutated")
    finally:
        family.set_kernel_source("step_B", "FMA_V1", None)
    assert family.kernel_source("step_B", "FMA_V1") == before


def test_the_arm_has_no_default_anywhere():
    """A wrong arm is a wrong answer, not a crash: both arms compile and run."""
    with pytest.raises(TypeError):
        family.kernel_source("step_B")  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="expansion must be"):
        family.kernel_source("step_B", "SOMETHING_ELSE")


#: ``parity/meep_gpu/results/`` is gitignored, so on a fresh checkout the four
#: tests below are DECLARED SKIPS and everything above still holds. Where the
#: directory IS present this is the half that catches a number taken from the
#: wrong run -- a real failure mode on this project and not a hypothetical one,
#: which is why ``gate_provenance`` exists. Same arrangement as
#: ``test_certification_metadata.py``'s artifact half.
# REPOINTED 2026-08-21 to the post-rename re-run. The 08-20 artifact keys its
# device digests by the OLD kernel names (fused_step_B_no_pml_complex, ...), so
# after the rename this test KeyError'd on the new name before it could compare
# anything. The re-run is the same gate on the shipped bytes: its keep leg
# reproduces all seven digests and the corpus digest against live emission under
# the arm it bound, which is what licenses reading the record off it.
# Its FLUSH leg did not release and is not read here — the device TEXT is
# policy-independent, so the keep leg alone settles what this test asks.
GATE_ARTIFACTS = (HERE.parents[1] / "parity" / "meep_gpu" / "results"
                  / "cuda_complex_no_pml_2026-08-21_rename")
CENSUS_ARTIFACTS = (HERE.parents[1] / "parity" / "meep_gpu" / "results"
                    / "cuda_predicate_coverage_2026-08-20_complex_no_pml")


def _gate_record():
    if not (GATE_ARTIFACTS / "keep" / "gate.json").is_file():
        pytest.skip(f"no gate artifact on this host: {GATE_ARTIFACTS}")
    return GATE_ARTIFACTS, json.loads(
        (GATE_ARTIFACTS / "keep" / "gate.json").read_text(encoding="utf-8"))


def test_the_recorded_verdict_is_the_artifact_on_disk():
    """Every headline number in the admission record, read back out of the gate.

    A record transcribed by hand from a run someone remembers is the failure
    ``gate_provenance`` exists to end from the other side. These are the numbers a
    reader would otherwise have to trust.
    """
    record = family.COMPLEX_NO_PML_ADMISSION
    root, gate = _gate_record()
    summary = gate["summary"]
    assert summary["released"] is True
    if record["host"] == RELEASE_HOST_PLACEHOLDER:
        # The release export writes this placeholder in place of the machine name a
        # run stamped, and the artifact is never edited, so the record is read by the
        # rule test_certification_metadata states for the placeholder: it stands for
        # one stamped machine, so the artifact's legs must stamp exactly one hostname.
        stamped = _artifact_stamps(root)["hostname"]
        assert len(stamped) == 1, (
            f"the record holds the release placeholder {record['host']!r}, which "
            f"names one machine, and {root.name} stamps {sorted(stamped)}")
    else:
        assert record["host"] == gate["host"]
    assert record["recorded_utc"] == gate["finished_utc"]
    assert record["cases_scored"] == summary["scored_cases"]
    assert record["expansion_arm"] == gate["expansion_arm"]["name"]
    assert record["multi_step_budget"] == 60

    scored = [c for c in gate["sweep"]["fmad_false"] if not c.get("skipped")]
    assert record["cases_skipped"] == len(gate["sweep"]["fmad_false"]) - len(scored)
    assert record["single_launch_identical"] == sum(
        1 for c in scored if c["single_launch"]["bit_identical"])
    assert record["multi_step_identical"] == sum(
        1 for c in scored if c.get("multi_step", {}).get("bit_identical"))
    assert record["single_launch_identical"] == len(scored), (
        "the record claims every scored case was identical")

    mutations = gate["source_mutations"]
    assert record["source_mutations_armed"] == sum(
        1 for v in mutations.values() if v.get("armed"))
    assert record["source_mutations_caught"] == sum(
        1 for v in mutations.values() if v.get("verdict") == "CAUGHT")
    assert record["source_mutations_null_confirmed"] == sum(
        1 for v in mutations.values() if v.get("verdict") == "NULL CONFIRMED")
    assert record["source_mutations_escaped"] == 0
    assert not [key for key, leg in mutations.items()
                if leg.get("verdict") not in ("CAUGHT", "NULL CONFIRMED")], (
        "every armed mutation must be CAUGHT or NULL CONFIRMED; an ESCAPED or "
        "UNACCOUNTED leg is a defect the battery could not see")
    assert record["verdict_flips_against_planted_defect"] is True
    flips = gate["verdict_flips_against_planted_defect"]
    assert flips["all_flipped"] is True and not flips["inapplicable"]


def test_the_recorded_kernel_digests_are_the_live_source():
    """THE DEVICE TEXT IS PINNED BY DIGEST, NOT THE FILE.

    A record written after a gate ran cannot also be inside the bytes that gate
    stamped, and pinning the whole file would make every comment edit read as an
    un-gated kernel change. What must not have moved is the seven device strings,
    so those are what the record carries -- and this recomputes them from
    :func:`kernel_source` under the arm the gate bound.
    """
    import hashlib

    record = family.COMPLEX_NO_PML_ADMISSION
    _root, gate = _gate_record()
    arm = gate["expansion_arm"]["code"]
    for key in family.KERNEL_KEYS:
        name = family.kernel_name(key)
        live = hashlib.sha256(
            family.kernel_source(key, arm).encode("utf-8")).hexdigest()
        assert record["kernel_source_sha256"][name] == live, (
            f"{name} has changed since the gate ran")
        assert gate["kernel_source_sha256"][name] == live
    assert record["corpus_digest"] == family.corpus_digest() == \
        gate["corpus_digest"]


def test_the_flush_leg_refused_and_the_record_says_so_first():
    """This round's gate refused flush, and the record must keep saying so first.

    The 2026-08-20 gate ran under 'keep' and refused 'flush' by name, because
    ``update_P``'s fifth expansion orientation was classified only by a keep-cut
    probe record and a licence is policy-conditional. That gap was closed
    2026-08-27 (``cuda_complex_no_pml_2026-08-27`` spans both policies), but the
    closure lives in ITS record — this round's artifact is immutable and still
    refuses, so the FIRST entry of ``what_it_does_not_license`` must still name
    the refusal (and now, honestly, the record that superseded it) rather than
    read as if this round's gate had licensed flush.
    """
    root, _gate = _gate_record()
    flush = json.loads((root / "flush" / "gate.json").read_text(encoding="utf-8"))
    assert "refused" in flush["status"]
    assert "subnormal policy" in flush["status"]
    assert "sweep" not in flush, "a refused leg must not carry measurements"
    first = family.COMPLEX_NO_PML_ADMISSION["what_it_does_not_license"][0]
    assert first.startswith("THE FLUSH POLICY")
    assert family.ADE_PROBE_PATTERN in first


def test_the_recorded_slot_delta_is_the_census_artifact():
    """+20 slots and +4 rows, read back out of the census this round ran.

    A slot delta nobody recomputed is the number this project has twice been
    burned by. The controlled comparison -- one analyzer, one record, the family
    in and out of UNION_FAMILIES -- is printed in the report and parsed here.
    """
    record = family.COMPLEX_NO_PML_ADMISSION
    report_path = CENSUS_ARTIFACTS / "coverage_report.txt"
    if not report_path.is_file():
        pytest.skip(f"no census artifact on this host: {CENSUS_ARTIFACTS}")
    report = report_path.read_text(encoding="utf-8")
    assert (f"without cuda_complex_no_pml : {record['slots_before']} / 759 slots, "
            f"{record['rows_covered_at_every_sub_step_before']} rows") in report
    assert (f"with    cuda_complex_no_pml : {record['slots_after']} / 759 slots, "
            f"{record['rows_covered_at_every_sub_step_after']} rows") in report
    assert f"DELTA                       : +{record['slots_gained']} slots, +4 rows" \
        in report
    assert "MATCHES the without-column" in report, (
        "the without-column must reproduce the closeout round's union, or the "
        "delta is a difference of two rounds rather than of one predicate")
    assert "disjointness: no slot is admitted by two families" in report
    assert f"cuda_complex_no_pml        {record['slots_gained']} slots" in report
