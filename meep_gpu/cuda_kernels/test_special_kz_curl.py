"""The special_kz (``grid.beta``) curl family: its predicate, its coefficient, its
device text, and a behavioural leg that actually steps a grid.

WHAT IS PINNED HERE AND WHAT IS NOT. The BYTE-IDENTITY VERDICT needs a device and
lives in ``parity/meep_gpu/gate_cuda_special_kz.py``. What this file pins is the
half of the family's silent-failure surface a laptop can reach:

* the PREDICATE, whose every refusal is a plausible, smooth, WRONG field if it
  leaks -- and whose ADMISSIONS must not overlap the certified pair's, because two
  families on one slot is a widening the union census reports as a finding;
* the COEFFICIENT, against ``stepping._special_kz_beta_term``'s own arithmetic on
  a real ``Grid``/``Fields`` pair -- not against a restatement of it;
* the DEVICE TEXT, against the CERTIFIED kernel it is supposed to be a copy of
  plus two lines. Comments stripped, the two files' code must be equal; a silent
  edit to either would otherwise only show on a device;
* the BEHAVIOUR, through the gate's own NumPy backend: the transcribed device tree
  is stepped against ``stepping.step_B``/``step_D`` on a real beta grid and
  compared as uint32 words, and two mutations are asserted to DIVERGE. A test that
  only checks verdict strings cannot tell a working kernel from one that returns
  its input.
"""

from __future__ import annotations

import ast
import math
import pathlib
import re
import types

import numpy
import pytest

from ..fields import Fields
from ..grid import Grid, Mirror
from ..pml import PML
from .. import stepping
from . import coverage, special_kz_curl


class _NumpyWearingCupysName(types.ModuleType):
    """NumPy behind CuPy's ``__name__`` -- the one clause a laptop cannot satisfy."""

    def __init__(self):
        super().__init__("cupy")

    def __getattr__(self, item):
        return getattr(numpy, item)


@pytest.fixture
def xp():
    return _NumpyWearingCupysName()


def build(xp, *, beta=0.31, cell=(12.0, 8.0, 0.0), boundaries=("periodic",) * 3,
          symmetry=(), force_complex_fields=False, dimensions=2, **kwargs):
    """A frozen (fields, pml, grid) triple with PML storage allocated.

    ``dimensions=2`` and a zero z extent are not a choice: ``Grid._resolve_beta``
    (grid.py:668-697) refuses beta on anything else, so these are the only grids a
    dispatch could hand this family.
    """
    grid = Grid(resolution=1.0, cell_size=cell, boundaries=boundaries,
                symmetry=symmetry, xp=xp, courant=0.5, dimensions=dimensions,
                beta=beta, **kwargs)
    thickness = tuple(
        (0, 0) if grid.shape[axis] < 6
        else (0, 2) if grid.is_mirrored(axis)
        else (2, 2)
        for axis in range(3))
    layer = PML(grid=grid, thickness=thickness)
    fields = Fields(grid=grid, force_complex_fields=force_complex_fields)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    return fields, layer, grid


SUB_STEPS = ("step_B", "step_D")
SIDES = ("H", "E")


# --------------------------------------------------------------------------
# The predicate
# --------------------------------------------------------------------------

def test_admits_a_real_beta_run_at_every_sub_step(xp):
    """The configuration the corpus actually has: real storage, active PML, beta.

    Both curl sub-steps AND both constitutive sides, because they are separate
    questions with separate answers -- the Metal track's tranche 6 ended with one
    row's two curls served and its two constitutive sides open.
    """
    fields, layer, grid = build(xp)
    for sub_step in SUB_STEPS:
        covered, reason = special_kz_curl.covers_special_kz_curl(
            fields, layer, grid, sub_step)
        assert covered, f"{sub_step}: {reason}"
    for side in SIDES:
        covered, reason = special_kz_curl.covers_special_kz_constitutive(
            fields, layer, grid, side)
        assert covered, f"update_{side}: {reason}"


def test_partitions_with_the_certified_family_in_both_directions(xp):
    """No slot may be admitted by two families; the union census calls that a
    FINDING rather than extra coverage.

    Measured both ways: on a beta run the certified predicate must refuse and this
    one admit; on a beta = 0 run the reverse.
    """
    beta_fields, beta_layer, beta_grid = build(xp, beta=0.31)
    plain_fields, plain_layer, plain_grid = build(xp, beta=0.0)
    for sub_step in SUB_STEPS:
        assert special_kz_curl.covers_special_kz_curl(
            beta_fields, beta_layer, beta_grid, sub_step)[0]
        certified, reason = coverage.covers_real_pml_curl(
            beta_fields, beta_layer, beta_grid, sub_step)
        assert not certified
        assert "special_kz" in reason

        assert coverage.covers_real_pml_curl(
            plain_fields, plain_layer, plain_grid, sub_step)[0]
        covered, reason = special_kz_curl.covers_special_kz_curl(
            plain_fields, plain_layer, plain_grid, sub_step)
        assert not covered
        assert "grid.beta is zero" in reason
    for side in SIDES:
        assert special_kz_curl.covers_special_kz_constitutive(
            beta_fields, beta_layer, beta_grid, side)[0]
        assert not coverage.covers_real_pml_constitutive(
            beta_fields, beta_layer, beta_grid, side)[0]
        assert coverage.covers_real_pml_constitutive(
            plain_fields, plain_layer, plain_grid, side)[0]
        assert not special_kz_curl.covers_special_kz_constitutive(
            plain_fields, plain_layer, plain_grid, side)[0]


def test_refuses_off_diagonal_epsilon_family_wide(xp):
    """stepping._special_kz_beta_term RAISES on it (stepping.py:800-810) and MEEP
    aborts (fields.cpp:548-549), so the refusal is the WHOLE family's, not the E
    side's: the run dies inside the first beta curl.

    The raise is measured, not quoted -- an admission here would be a kernel
    stepping a configuration the oracle refuses to define.
    """
    fields, layer, grid = build(xp)
    # ``has_offdiagonal_epsilon`` is a read-only property over
    # ``Fields._chi1inv_offdiagonal`` (fields.py:1313-1315), which the lifter fills
    # from the structure's surviving rows. One row is planted directly, which is
    # the same state a lifted anisotropic run reaches.
    row = grid.xp.full(grid.shape, numpy.float32(0.1), dtype=numpy.float32)
    fields._chi1inv_offdiagonal = {"Ex": {"Ey": row}}
    assert fields.has_offdiagonal_epsilon

    # THE ORACLE REFUSES THIS RUN, measured rather than quoted: the array path's
    # own beta term raises inside the first curl, so there is no such run to step.
    with pytest.raises(ValueError, match="needs complex fields"):
        stepping._special_kz_beta_term(
            fields, numpy.zeros(grid.shape, dtype=numpy.float32), 1.0,
            magnetic=True)

    for sub_step in SUB_STEPS:
        covered, reason = special_kz_curl.covers_special_kz_curl(
            fields, layer, grid, sub_step)
        assert not covered
        assert "off-diagonal epsilon with beta in REAL storage" in reason
    for side in SIDES:
        covered, reason = special_kz_curl.covers_special_kz_constitutive(
            fields, layer, grid, side)
        assert not covered
        assert "off-diagonal epsilon with beta in REAL storage" in reason


def test_refuses_complex_storage_by_name(xp):
    """Four of the six beta corpus rows are complex; none is this family's."""
    fields, layer, grid = build(xp, force_complex_fields=True)
    for sub_step in SUB_STEPS:
        covered, reason = special_kz_curl.covers_special_kz_curl(
            fields, layer, grid, sub_step)
        assert not covered
        assert "complex64 storage with beta" in reason


def test_refuses_a_grid_that_is_not_effectively_two_dimensional(xp):
    """RESTATED, never inferred from ``Grid._resolve_beta``'s constructor guard.

    A predicate that reasons "the constructor would have refused it" admits by
    argument from absence. The proxy below is a grid that reports beta on three
    dimensions, which is exactly what a lifted object could do if the guard ever
    moved.
    """
    fields, layer, grid = build(xp)

    class _ThreeDimensional:
        def __init__(self, real):
            object.__setattr__(self, "_real", real)

        @property
        def dimensions(self):
            return 3

        def __getattr__(self, item):
            return getattr(object.__getattribute__(self, "_real"), item)

    proxy = _ThreeDimensional(grid)
    covered, reason = special_kz_curl.covers_special_kz_curl(
        fields, layer, proxy, "step_B")
    assert not covered
    assert "is not the effective-2-D grid beta requires" in reason


def test_the_fold_is_admitted_at_both_terminations(xp):
    """One of the two real-storage beta corpus rows is a Y fold with a PERIODIC
    termination (``TestSpecialKz.test_eigsrc_kz_1_real_imag``, stored 212 > owned
    211). The certified pair carries both fold masks since 2026-08-20 and the beta
    term reuses a CENTRE register the fold never touches, so admission is inherited
    -- and the composition is MEASURED by the gate's folded arm, not argued here.
    """
    for boundaries, phase in ((("periodic", "periodic", "periodic"), 1),
                              (("periodic", "metallic", "periodic"), 1),
                              (("periodic", "periodic", "periodic"), -1)):
        fields, layer, grid = build(
            xp, beta=0.2, cell=(8.0, 16.0, 0.0), boundaries=boundaries,
            symmetry=(Mirror("Y", phase),))
        assert grid.is_mirrored(1)
        for sub_step in SUB_STEPS:
            covered, reason = special_kz_curl.covers_special_kz_curl(
                fields, layer, grid, sub_step)
            assert covered, f"{boundaries} phase={phase} {sub_step}: {reason}"


def test_refuses_a_sub_step_name_it_does_not_serve(xp):
    fields, layer, grid = build(xp)
    with pytest.raises(ValueError):
        special_kz_curl.covers_special_kz_curl(fields, layer, grid, "update_H")
    with pytest.raises(ValueError):
        special_kz_curl.covers_special_kz_constitutive(fields, layer, grid, "step_B")


def test_an_unreadable_grid_is_refused_not_admitted(xp):
    """FAIL CLOSED. A raise escaping a predicate is a CRASHED RUN where a "no" was
    the correct answer."""
    fields, layer, grid = build(xp)

    class _Unreadable:
        def __init__(self, real):
            object.__setattr__(self, "_real", real)

        @property
        def beta(self):
            raise RuntimeError("this grid cannot say")

        def __getattr__(self, item):
            return getattr(object.__getattribute__(self, "_real"), item)

    covered, reason = special_kz_curl.covers_special_kz_curl(
        fields, layer, _Unreadable(grid), "step_B")
    assert not covered
    assert "could not be asked for beta" in reason


# --------------------------------------------------------------------------
# The coefficient
# --------------------------------------------------------------------------

@pytest.mark.parametrize("beta", [0.3321611318837033, 0.2, -0.39073112848927377,
                                  -0.685, 1e-7, 12.5])
def test_coefficient_reproduces_steppings_own_increment_bit_for_bit(xp, beta):
    """The transcription measured against the ORACLE, not against a restatement.

    ``stepping._special_kz_beta_term`` is called on a real ``Fields`` and its
    returned increment compared, as uint32 words, against
    ``-(coefficient * partner)`` built from :func:`beta_curl_coefficients`. The
    B side's ``sign = +1`` call site and the D side's are both exercised, and the
    ``magnetic`` flag is passed BOTH ways at each sign, because under real storage
    it must not change the answer at all (the ``+-1j`` branch at
    stepping.py:798-799 is not taken).
    """
    fields, _layer, grid = build(xp, beta=beta)
    plus, minus = special_kz_curl.beta_curl_coefficients(grid.beta, grid.dt)
    rng = numpy.random.default_rng(4242)
    partner = rng.uniform(-1.0, 1.0, size=grid.shape).astype(numpy.float32)
    for sign, mine in ((+1.0, plus), (-1.0, minus)):
        for magnetic in (True, False):
            theirs = stepping._special_kz_beta_term(fields, partner, sign,
                                                    magnetic=magnetic)
            ours = -(numpy.float32(mine) * partner)
            assert numpy.array_equal(
                numpy.ascontiguousarray(theirs, dtype=numpy.float32).view(numpy.uint32),
                numpy.ascontiguousarray(ours, dtype=numpy.float32).view(numpy.uint32)), (
                f"beta={beta} sign={sign} magnetic={magnetic}")


def test_coefficient_is_the_float64_product_rounded_once(xp):
    """stepping.py:797 computes in float64 and :811 rounds ONCE. A double rounding
    (through float16, say) is a different word, and the gate arms it as a host
    mutation; this pins the shipped spelling."""
    plus, minus = special_kz_curl.beta_curl_coefficients(0.3321611318837033, 0.005)
    exact = 2.0 * math.pi * 0.3321611318837033 * 0.005
    assert plus == float(numpy.float32(exact))
    assert minus == float(numpy.float32(-exact))
    assert plus == -minus  # float32 negation is exact


# --------------------------------------------------------------------------
# The device text
# --------------------------------------------------------------------------

def _literal(name: str) -> str:
    """A module-level string literal from ``step_curl_kernels.py``, read WITHOUT
    importing it (that module imports ``cupy`` at scope)."""
    source = (pathlib.Path(special_kz_curl.__file__).with_name("step_curl_kernels.py")
              .read_text(encoding="utf-8"))
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == name
                and isinstance(node.value, ast.Constant)):
            return node.value.value
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == name
                and isinstance(node.value, ast.BinOp)):
            # ``_REAL_PML_PRELUDE + r'''...'''``
            return node.value.right.value
    raise AssertionError(f"{name} is no longer a module-level literal")


def _code_only(text: str) -> str:
    """Comment lines and blank lines removed, trailing space stripped."""
    out = []
    for line in text.splitlines():
        stripped = line.split("//")[0].rstrip()
        if stripped.strip():
            out.append(stripped)
    return "\n".join(out)


BETA_LINES = ("        curl = curl - (beta_plus * f2);",
              "        curl = curl - (beta_minus * f1);")


@pytest.mark.parametrize("side", ["B", "D"])
def test_kernel_is_the_certified_kernel_plus_exactly_the_beta_lines(side):
    """THE ANTI-DRIFT TEST, and the reason this family is safe to read at all.

    Strip the comments, delete the two beta statements and the two extra scalar
    parameters, rename the entry point -- and what is left must be the CERTIFIED
    ``step_{B,D}_pml_real`` byte for byte. The ghost rule, the stencil
    grouping, every mask line, the coefficient pairing and the recurrence are then
    not "believed to be the same"; they ARE the same text.
    """
    mine = special_kz_curl.kernel_source(f"step_{side}_special_kz_real")
    certified = (special_kz_curl._REAL_PML_PRELUDE
                 + _literal(f"_step_{side}_pml_real_kernel_code"))
    stripped = _code_only(mine)
    for line in BETA_LINES:
        assert line in stripped, f"the {side} kernel no longer carries {line!r}"
        stripped = stripped.replace(line + "\n", "")
    stripped = stripped.replace(
        "    int bc_x, int bc_y, int bc_z,\n    float beta_plus, float beta_minus\n",
        "    int bc_x, int bc_y, int bc_z\n")
    stripped = stripped.replace(f"step_{side}_special_kz_real",
                                f"step_{side}_pml_real")
    assert stripped == _code_only(certified)


@pytest.mark.parametrize("side", ["B", "D"])
def test_the_z_component_carries_no_beta_term(side):
    """MEEP's ``cc`` loop runs over ``d_c`` in {X, Y} only (step_db.cpp:148-176) and
    stepping.py:384-391 / :467-474 have no z branch. Two statements, not three."""
    stripped = _code_only(
        special_kz_curl.kernel_source(f"step_{side}_special_kz_real"))
    assert len(re.findall(r"curl = curl - \(beta_", stripped)) == 2


def test_the_shared_prelude_is_the_certified_one():
    """Read out of the sibling rather than copied, so the two cannot fork."""
    assert special_kz_curl._REAL_PML_PRELUDE == _literal("_REAL_PML_PRELUDE")
    assert "BC_MIRROR_PERIODIC 2" in special_kz_curl._REAL_PML_PRELUDE


def test_the_mutation_seam_round_trips():
    """``set_kernel_source`` is the ONE seam a gate mutates through; a second copy
    of the text anywhere would let a mutation leg report a pass for bytes it never
    applied."""
    name = special_kz_curl.SPECIAL_KZ_KERNELS[0]
    original = special_kz_curl.kernel_source(name)
    try:
        special_kz_curl.set_kernel_source(name, original + "\n// planted\n")
        assert special_kz_curl.kernel_source(name).endswith("// planted\n")
    finally:
        special_kz_curl.set_kernel_source(name, original)
    assert special_kz_curl.kernel_source(name) == original
    with pytest.raises(ValueError):
        special_kz_curl.kernel_source("no_such_kernel")


# --------------------------------------------------------------------------
# The behaviour
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gate():
    """The device gate's own harness, on its NumPy backend.

    Imported rather than re-transcribed: the gate owns the transcription of the
    device tree, and a second copy here would be a second thing to drift -- which
    is the exact failure mode these tests exist to catch everywhere else.
    """
    from parity.meep_gpu import gate_cuda_special_kz  # noqa: PLC0415

    return gate_cuda_special_kz


@pytest.mark.parametrize("label", ["plain_periodic", "plain_metallic_xy",
                                   "fold_Y_periodic", "fold_Y_metallic"])
@pytest.mark.parametrize("sub_step", SUB_STEPS)
def test_the_transcribed_tree_steps_a_beta_grid_identically(gate, label, sub_step):
    """The device tree, in float32, against ``stepping`` itself -- uint32 words.

    COMPILES NOTHING AND CERTIFIES NOTHING; the byte verdict is the device gate's.
    What this settles at the merge bar is that the beta term's OPERAND, SIGN,
    SCALE and POSITION reproduce the array path on a real grid, folded and not.
    The case carries its own non-vacuity floors: the array path must have moved,
    and the beta term must have moved the components ``stepping`` says carry one
    and no others.
    """
    spec = next(s for s in gate.BETA_SPECS if s["label"] == label)
    case = gate.one_case("numpy", spec, sub_step, gate.INEXACT_COURANT,
                         "uniform", "fmad_false")
    assert not case.get("skipped"), case.get("skipped")
    assert case["beta_term_is_live"]["meets_floor"], case["beta_term_is_live"]
    assert case["beta_term_is_live"]["component_pattern_matches_stepping"]
    assert case["single_launch"]["bit_identical"], case["single_launch"]
    assert case["multi_step"]["bit_identical"], case["multi_step"]


@pytest.mark.parametrize("mutation", ["swap_beta_coefficients",
                                      "zero_beta_coefficients",
                                      "beta_coefficient_scaled_by_dtdx"])
def test_a_planted_coefficient_defect_diverges(gate, mutation):
    """A comparator that cannot fail certifies nothing. Each of these is a defect
    the array path would not commit, and each must show as differing words."""
    spec = next(s for s in gate.BETA_SPECS if s["label"] == "plain_periodic")
    case = gate.one_case("numpy", spec, "step_B", gate.INEXACT_COURANT, "uniform",
                         "fmad_false", host_mutation=mutation)
    assert not case.get("skipped")
    assert not case["single_launch"]["bit_identical"], mutation


def test_the_null_coefficient_mutation_stays_identical(gate):
    """float64 negation and float32 rounding COMMUTE. Computing the minus
    coefficient as ``-plus`` in float64 and rounding once must be the same word --
    a battery of only-must-be-caught legs scores identically whether the comparator
    works or has degenerated into failing everything."""
    spec = next(s for s in gate.BETA_SPECS if s["label"] == "plain_periodic")
    case = gate.one_case("numpy", spec, "step_B", gate.INEXACT_COURANT, "uniform",
                         "fmad_false",
                         host_mutation="beta_coefficient_negated_from_plus")
    assert not case.get("skipped")
    assert case["single_launch"]["bit_identical"]


def test_the_gates_term_table_agrees_with_stepping(gate):
    """The NumPy leg consumes a transcription of the DEVICE source; if it drifted
    from ``stepping``'s own term tables the leg would compare a different kernel."""
    record = gate.check_terms_against_stepping()
    assert record["agreed"], record["disagreements"]
