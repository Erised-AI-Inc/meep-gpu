"""The BFAST (``grid.bfast_active``) curl family: its predicate, its k assignment,
its device text, and a behavioural leg that actually steps a grid.

WHAT IS PINNED HERE AND WHAT IS NOT. The BYTE-IDENTITY VERDICT needs a device and
lives in ``parity/meep_gpu/gate_cuda_bfast.py``. What this file pins is the half a
laptop can reach:

* the PREDICATE -- its admissions, which must not overlap the certified pair's,
  and its refusals, of which the FOLD is the one that costs nothing and is
  therefore the one most likely to be widened by argument;
* the k ASSIGNMENT, against ``stepping._bfast_term``'s own arithmetic on real
  ``Grid`` objects, INCLUDING an invariant axis, because ``have_p``/``have_m``
  zero a k there and the curl's difference would swallow the mistake while the
  BFAST sum would not;
* the DEVICE TEXT, against the CERTIFIED kernel it is supposed to be a copy of
  plus the named insert;
* the BEHAVIOUR, through the gate's own NumPy backend: the transcribed device tree
  is stepped against ``stepping`` on a real BFAST grid, and the ``f_bfast`` IIR
  STATE is compared beside the field, because a kernel that gets the field right
  and the state wrong is right for exactly one launch and wrong forever after.
"""

from __future__ import annotations

import ast
import pathlib
import re
import types

import numpy
import pytest

from ..fields import Fields
from ..grid import Grid, Mirror
from ..pml import PML
from .. import stepping
from . import bfast_curl, coverage, folded_pass


class _NumpyWearingCupysName(types.ModuleType):
    def __init__(self):
        super().__init__("cupy")

    def __getattr__(self, item):
        return getattr(numpy, item)


@pytest.fixture
def xp():
    return _NumpyWearingCupysName()


def build(xp, *, k=(0.81, -0.37, 0.52), cell=(8.0, 10.0, 12.0),
          boundaries=("periodic",) * 3, symmetry=(), force_complex_fields=False,
          **kwargs):
    grid = Grid(resolution=1.0, cell_size=cell, boundaries=boundaries,
                symmetry=symmetry, xp=xp, courant=0.5, bfast_scaled_k=k, **kwargs)
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

def test_admits_a_real_bfast_run_at_every_sub_step(xp):
    fields, layer, grid = build(xp)
    assert grid.bfast_active
    for sub_step in SUB_STEPS:
        covered, reason = bfast_curl.covers_bfast_curl(fields, layer, grid, sub_step)
        assert covered, f"{sub_step}: {reason}"
    for side in SIDES:
        covered, reason = bfast_curl.covers_bfast_constitutive(
            fields, layer, grid, side)
        assert covered, f"update_{side}: {reason}"


def test_partitions_with_the_certified_family_in_both_directions(xp):
    """Two families on one slot is a widening the union census reports as a
    FINDING; measured both ways."""
    bfast_fields, bfast_layer, bfast_grid = build(xp)
    plain_fields, plain_layer, plain_grid = build(xp, k=(0.0, 0.0, 0.0))
    assert not plain_grid.bfast_active
    for sub_step in SUB_STEPS:
        assert bfast_curl.covers_bfast_curl(
            bfast_fields, bfast_layer, bfast_grid, sub_step)[0]
        certified, reason = coverage.covers_real_pml_curl(
            bfast_fields, bfast_layer, bfast_grid, sub_step)
        assert not certified
        assert "BFAST" in reason

        assert coverage.covers_real_pml_curl(
            plain_fields, plain_layer, plain_grid, sub_step)[0]
        covered, reason = bfast_curl.covers_bfast_curl(
            plain_fields, plain_layer, plain_grid, sub_step)
        assert not covered
        assert "grid.bfast_active is False" in reason
    for side in SIDES:
        assert bfast_curl.covers_bfast_constitutive(
            bfast_fields, bfast_layer, bfast_grid, side)[0]
        assert not coverage.covers_real_pml_constitutive(
            bfast_fields, bfast_layer, bfast_grid, side)[0]
        assert not bfast_curl.covers_bfast_constitutive(
            plain_fields, plain_layer, plain_grid, side)[0]


def test_refuses_a_mirror_fold_by_name(xp):
    """THE REFUSAL THAT COSTS NOTHING, AND IS THEREFORE THE ONE TO PIN.

    The certified pair admits a fold since 2026-08-20 on a device verdict whose
    argument is that the folded ghost VALUES are dead -- a claim about a stencil
    that DIFFERENCES its ghosts. BFAST SUMS the same operands, so the claim does
    not transfer, and no corpus row pairs BFAST with a fold. A widening here would
    be pure argument.
    """
    fields, layer, grid = build(xp, cell=(8.0, 16.0, 12.0),
                                symmetry=(Mirror("Y", 1),))
    assert grid.is_mirrored(1)
    # AND NOBODY ELSE PICKS IT UP: the certified pair admits the fold but refuses
    # the BFAST, so a fold + BFAST slot is served by no family at all. That is the
    # honest state and it is asserted, because "refused here" would otherwise leave
    # open whether the slot quietly fell to the certified kernel.
    certified, certified_reason = coverage.covers_real_pml_curl(
        fields, layer, grid, "step_B")
    assert not certified
    assert "BFAST" in certified_reason
    for sub_step in SUB_STEPS:
        covered, reason = bfast_curl.covers_bfast_curl(fields, layer, grid, sub_step)
        assert not covered
        assert "mirror symmetry with BFAST" in reason


def test_refuses_complex_storage_by_name(xp):
    """``Fields._ensure_bfast_storage`` allocates ONE COMPLEX state per component
    (fields.py:632-651); these kernels index float32."""
    fields, layer, grid = build(xp, force_complex_fields=True)
    for sub_step in SUB_STEPS:
        covered, reason = bfast_curl.covers_bfast_curl(fields, layer, grid, sub_step)
        assert not covered
        assert "complex64 storage with BFAST" in reason


def test_refuses_a_missing_or_wrong_shaped_state(xp):
    """The state is an OUTPUT: the array path RAISES without it (stepping.py:917-920)
    and a kernel handed a differently shaped one corrupts memory."""
    fields, layer, grid = build(xp)
    saved = fields.f_bfast_By
    fields.f_bfast_By = None
    covered, reason = bfast_curl.covers_bfast_curl(fields, layer, grid, "step_B")
    assert not covered
    assert "f_bfast_By is not allocated" in reason
    fields.f_bfast_By = numpy.zeros((2, 2, 2), dtype=numpy.float32)
    covered, reason = bfast_curl.covers_bfast_curl(fields, layer, grid, "step_B")
    assert not covered
    assert "f_bfast_By" in reason
    fields.f_bfast_By = saved
    assert bfast_curl.covers_bfast_curl(fields, layer, grid, "step_B")[0]


def test_the_constitutive_sides_do_not_ask_for_the_state(xp):
    """``update_H``/``update_E`` never touch ``f_bfast``; requiring it would refuse
    a configuration on the absence of an array the sub-step does not read."""
    fields, layer, grid = build(xp)
    for name in ("f_bfast_Bx", "f_bfast_By", "f_bfast_Bz",
                 "f_bfast_Dx", "f_bfast_Dy", "f_bfast_Dz"):
        setattr(fields, name, None)
    for side in SIDES:
        covered, reason = bfast_curl.covers_bfast_constitutive(
            fields, layer, grid, side)
        assert covered, f"update_{side}: {reason}"
    for sub_step in SUB_STEPS:
        assert not bfast_curl.covers_bfast_curl(fields, layer, grid, sub_step)[0]


def test_refuses_a_sub_step_name_it_does_not_serve(xp):
    fields, layer, grid = build(xp)
    with pytest.raises(ValueError):
        bfast_curl.covers_bfast_curl(fields, layer, grid, "update_E")
    with pytest.raises(ValueError):
        bfast_curl.covers_bfast_constitutive(fields, layer, grid, "step_D")


def test_an_unreadable_grid_is_refused_not_admitted(xp):
    fields, layer, grid = build(xp)

    class _Unreadable:
        def __init__(self, real):
            object.__setattr__(self, "_real", real)

        def is_mirrored(self, axis):
            raise RuntimeError("this grid cannot say")

        def __getattr__(self, item):
            return getattr(object.__getattribute__(self, "_real"), item)

    covered, reason = bfast_curl.covers_bfast_curl(
        fields, layer, _Unreadable(grid), "step_B")
    assert not covered
    assert "could not answer a question this predicate has to ask" in reason


# --------------------------------------------------------------------------
# The k assignment
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cell,dimensions", [((8.0, 10.0, 12.0), None),
                                             ((8.0, 10.0, 0.0), 2)])
@pytest.mark.parametrize("k", [(0.8169576958985646, 0.0, 0.0),
                               (0.81, -0.37, 0.52),
                               (-0.63, 0.44, -0.21)])
def test_k_assignment_reproduces_steppings_own(xp, cell, dimensions, k):
    """Re-derived through ``stepping``'s OWN term tables and ``_bfast_axis``.

    The second parametrisation carries an INVARIANT axis, where
    ``have_p``/``have_m`` zero a k (stepping.py:909-912). That case is the reason
    the gates are host-side: the curl's DIFFERENCE swallows an invariant axis by
    itself and the BFAST SUM does not.
    """
    extra = {} if dimensions is None else {"dimensions": dimensions}
    _fields, _layer, grid = build(xp, k=k, cell=cell, **extra)
    for sub_step, source in (("step_B", stepping.B_CURL_TERMS),
                             ("step_D", stepping.D_CURL_TERMS)):
        mine = bfast_curl.bfast_curl_coefficients(grid, sub_step)
        assert len(mine) == 3
        magnetic = sub_step == "step_B"
        for index, term in enumerate(source):
            have_p = not grid.is_invariant(term.first_axis)
            have_m = not grid.is_invariant(term.second_axis)
            k1 = grid.bfast_scaled_k[stepping._bfast_axis(term.second)] if have_m else 0.0
            k2 = grid.bfast_scaled_k[stepping._bfast_axis(term.first)] if have_p else 0.0
            if not magnetic:
                k1, k2 = -k1, -k2
            assert mine[index] == (float(numpy.float32(k1)),
                                   float(numpy.float32(k2))), (
                f"{sub_step} {term.target}")


def test_the_term_table_matches_steppings(xp):
    """``BFAST_TERMS`` is a transcription of the device source; the k assignment
    reads ``first``/``second`` off it, and confusing them is silent."""
    for sub_step, source in (("step_B", stepping.B_CURL_TERMS),
                             ("step_D", stepping.D_CURL_TERMS)):
        for mine, theirs in zip(bfast_curl.BFAST_TERMS[sub_step], source):
            assert mine == (theirs.target, theirs.first, theirs.first_axis,
                            theirs.second, theirs.second_axis)


def test_bfast_axis_matches_steppings(xp):
    for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "Bx", "Dz"):
        assert bfast_curl._bfast_axis(component) == stepping._bfast_axis(component)


def test_the_d_side_is_the_b_side_negated(xp):
    """MEEP's ``if (ft == D_stuff) { k1 = -k1; k2 = -k2; }`` (step_db.cpp:136), which
    is what turns ``+d/dt(k x E)`` into ``-d/dt(k x H)``."""
    _f, _l, grid = build(xp)
    b = bfast_curl.bfast_curl_coefficients(grid, "step_B")
    d = bfast_curl.bfast_curl_coefficients(grid, "step_D")
    assert d == tuple((-k1, -k2) for k1, k2 in b)


# --------------------------------------------------------------------------
# The device text
# --------------------------------------------------------------------------

def _literal(name: str) -> str:
    source = (pathlib.Path(bfast_curl.__file__).with_name("step_curl_kernels.py")
              .read_text(encoding="utf-8"))
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == name):
            continue
        if isinstance(node.value, ast.Constant):
            return node.value.value
        if isinstance(node.value, ast.BinOp):
            return node.value.right.value
    raise AssertionError(f"{name} is no longer a module-level literal")


def _code_only(text: str) -> str:
    out = []
    for line in text.splitlines():
        stripped = line.split("//")[0].rstrip()
        if stripped.strip():
            out.append(stripped)
    return "\n".join(out)


#: The lines the BFAST pass adds to each component block (``folded_pass.bfast_insert``):
#: the ownership flag, its mask lines, the state step and its ladder.
_PASS_LINE_PATTERNS = (
    re.compile(r"int owned_cell = 1;"),
    re.compile(r"if \(.*\) owned_cell = 0;"),
    re.compile(r"float inc_bfast = bfast_pass\(fb_[BD][xyz], idx, sf, f1, ss, f2, "
               r"k1_[abc], k2_[abc], pr_[xyz], pr_[xyz], "
               r"!in_[hi]_[xyz] && !in_[hi]_[xyz], owned_cell\);"),
    re.compile(r"fold_apply\([BD][xyz], fu_[BD][xyz], idx, inc_bfast, .*\);"),
)


@pytest.mark.parametrize("side,letters", [("B", "Bx By Bz"), ("D", "Dx Dy Dz")])
def test_kernel_is_the_certified_kernel_plus_exactly_the_bfast_insert(side, letters):
    """THE ANTI-DRIFT TEST. Strip the comments, the fold helpers, the BFAST pass
    statements and the state/scalar parameters, rename the entry point, and what is
    left must be the CERTIFIED ``step_{B,D}_pml_real`` byte for byte.

    The pass is MEEP's own loop after the curl (step_db.cpp:124-142), so deleting it
    leaves the curl block untouched: this test also pins that the curl's own masks and
    absorber were not disturbed by the insert.
    """
    targets = letters.split()
    mine = _code_only(bfast_curl.kernel_source(f"step_{side}_bfast_real"))
    certified = _code_only(bfast_curl._REAL_PML_PRELUDE
                           + _literal(f"_step_{side}_pml_real_kernel_code"))
    helpers = _code_only(folded_pass.REAL_FOLD_HELPERS + folded_pass.BFAST_PASS_HELPER)
    assert mine.count(helpers) == 1
    mine = mine.replace(helpers + "\n", "", 1)
    kept = [line for line in mine.splitlines()
            if not any(pattern.fullmatch(line.strip()) for pattern in _PASS_LINE_PATTERNS)]
    stripped = "\n".join(kept)
    stripped = stripped.replace(
        "    float* __restrict__ fb_%s, float* __restrict__ fb_%s, "
        "float* __restrict__ fb_%s,\n" % tuple(targets), "")
    stripped = stripped.replace(
        "    float k1_a, float k2_a, float k1_b, float k2_b, float k1_c, float k2_c,\n",
        "")
    stripped = stripped.replace(f"step_{side}_bfast_real",
                                f"step_{side}_pml_real")
    assert stripped == certified


@pytest.mark.parametrize("side", ["B", "D"])
def test_every_component_carries_a_bfast_pass(side):
    """Unlike special_kz, whose z component gets none: MEEP's ``step_bfast`` runs
    over all three (step_db.cpp:129-142), each a pass of its own after the curl."""
    text = _code_only(bfast_curl.kernel_source(f"step_{side}_bfast_real"))
    assert len(re.findall(r"float inc_bfast = bfast_pass\(", text)) == 3
    assert len(re.findall(r"fold_apply\([BD][xyz], fu_[BD][xyz], idx, inc_bfast, ",
                          text)) == 3


@pytest.mark.parametrize("side", ["B", "D"])
def test_each_pass_follows_its_own_curl(side):
    """The pass reads the absorber's output: per component, the curl's ``pml_apply``
    comes first, then the state step, then its ladder. Reversed, the ladder would add
    the increment to a field the curl then overwrites."""
    text = _code_only(bfast_curl.kernel_source(f"step_{side}_bfast_real"))
    body = text[text.index('extern "C"'):]
    for target in (f"{side}x", f"{side}y", f"{side}z"):
        curl = body.index(f"pml_apply({target}, fu_{target}, idx, curl, ")
        state = body.index(f"bfast_pass(fb_{target}, idx, ")
        ladder = body.index(f"fold_apply({target}, fu_{target}, idx, inc_bfast, ")
        assert curl < state < ladder, target


def test_the_state_is_stored_on_owned_cells_only():
    """MEEP writes ``F`` only inside its owned-cell loop: the pass returns before the
    store on an unowned cell, and forms the increment from the register it loaded."""
    helper = folded_pass.BFAST_PASS_HELPER
    assert "    if (!owned) return 0.0f;\n    F[idx] = updated;\n" in helper
    assert "    return -(updated - fprev);\n" in helper


def test_the_shared_prelude_is_the_certified_one():
    assert bfast_curl._REAL_PML_PRELUDE == _literal("_REAL_PML_PRELUDE")


def test_the_mutation_seam_round_trips():
    name = bfast_curl.BFAST_KERNELS[0]
    original = bfast_curl.kernel_source(name)
    try:
        bfast_curl.set_kernel_source(name, original + "\n// planted\n")
        assert bfast_curl.kernel_source(name).endswith("// planted\n")
    finally:
        bfast_curl.set_kernel_source(name, original)
    assert bfast_curl.kernel_source(name) == original
    with pytest.raises(ValueError):
        bfast_curl.kernel_source("no_such_kernel")


# --------------------------------------------------------------------------
# The behaviour
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gate():
    from parity.meep_gpu import gate_cuda_bfast  # noqa: PLC0415

    return gate_cuda_bfast


@pytest.mark.parametrize("label", ["corpus_single_axis_k", "general_k_periodic",
                                   "general_k_metallic_xyz", "invariant_z_axis"])
@pytest.mark.parametrize("sub_step", SUB_STEPS)
def test_the_transcribed_tree_steps_a_bfast_grid_identically(gate, label, sub_step):
    """The device tree, in float32, against ``stepping`` itself -- uint32 words,
    FIELD, PML AUXILIARY AND ``f_bfast`` STATE.

    COMPILES NOTHING AND CERTIFIES NOTHING; the byte verdict is the device gate's.
    The case carries its own floors: the state must start nonzero (or the ``- F_prev``
    term is invisible at launch one), the array path must have moved, and the
    run must differ BOTH from a BFAST-off control and from a zero-k control -- the
    second is what makes the k assignment load-bearing rather than the recurrence
    alone.
    """
    spec = next(s for s in gate.BFAST_SPECS if s["label"] == label)
    case = gate.one_case("numpy", spec, sub_step, gate.INEXACT_COURANT,
                         "uniform", "fmad_false")
    assert not case.get("skipped"), case.get("skipped")
    assert case["state_starts_nonzero"]["meets_floor"]
    assert case["bfast_term_is_live"]["meets_floor"], case["bfast_term_is_live"]
    assert case["bfast_term_is_live"]["total_vs_zero_k"] > 0
    assert case["single_launch"]["bit_identical"], case["single_launch"]
    assert case["multi_step"]["bit_identical"], case["multi_step"]
    # The STATE specifically, named, so a future refactor that stopped comparing it
    # fails here rather than passing silently.
    for name in gate.SUB_STEP_ARRAYS[sub_step]["state"]:
        assert case["single_launch_per_array"][name] == 0


@pytest.mark.parametrize("mutation", ["swap_k1_k2_scalars", "zero_k_scalars",
                                      "negate_k_scalars",
                                      "k_indexed_by_derivative_axis"])
def test_a_planted_k_defect_diverges(gate, mutation):
    """A comparator that cannot fail certifies nothing.
    ``k_indexed_by_derivative_axis`` is MEEP's own documented easy mistake."""
    spec = next(s for s in gate.BFAST_SPECS if s["label"] == "general_k_periodic")
    case = gate.one_case("numpy", spec, "step_B", gate.INEXACT_COURANT, "uniform",
                         "fmad_false", host_mutation=mutation)
    assert not case.get("skipped")
    assert not case["single_launch"]["bit_identical"], mutation


@pytest.mark.parametrize("mutation", ["record_bound_off_by_one",
                                      "record_lattices_swapped"])
def test_a_corrupted_chunk_record_diverges(gate, mutation):
    """KERNELS.md section 7.2: the record the launch binds, corrupted while the
    oracle's division is untouched, must move a word -- the per-cell forms are chosen
    by the record and nothing else."""
    spec = next(s for s in gate.BFAST_SPECS if s["label"] == "general_k_periodic")
    case = gate.one_case("numpy", spec, "step_B", gate.INEXACT_COURANT, "uniform",
                         "fmad_false", host_mutation=mutation)
    assert not case.get("skipped"), case.get("skipped")
    assert not case["single_launch"]["bit_identical"], mutation


def test_the_null_k_mutation_stays_identical(gate):
    """Rounding an already-float32 value to float32 is the identity; a battery of
    only-must-be-caught legs cannot distinguish a working comparator from one that
    fails everything."""
    spec = next(s for s in gate.BFAST_SPECS if s["label"] == "general_k_periodic")
    case = gate.one_case("numpy", spec, "step_B", gate.INEXACT_COURANT, "uniform",
                         "fmad_false", host_mutation="k_scalars_rounded_again")
    assert not case.get("skipped")
    assert case["single_launch"]["bit_identical"]


def test_the_gates_term_table_agrees_with_stepping(gate):
    record = gate.check_terms_against_stepping()
    assert record["agreed"], record["disagreements"]
