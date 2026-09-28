"""Laptop contracts for the BFAST fused magnetic B/H pair on Triton.

Everything here runs on the NumPy merge-bar machine. The load-bearing one is the
TRANSCRIPTION leg: this product claims to be
``kernels.fused_curl_constitutive_B`` plus exactly the ``if HAS_BFAST:`` block
``bfast_curl.bfast_pml_curl_step`` adds to ``kernels.pml_curl_step``, and that
claim is checked here as two EXACT STATEMENT-LIST equalities against the shipped
sources rather than by reading the docstring.

THE COMPARISON IS PER TOP-LEVEL STATEMENT, NOT PER LINE, and that is the one
difference from the sibling beta suite. The BFAST tail CONTAINS lines that also
appear elsewhere in both bodies (``if BACKWARD:``, ``if BCX == METALLIC:``), so a
flat line subtraction would leave those orphans behind and could not express the
equality at all. An ``ast`` statement is the right grain: the tail is exactly ONE
``If`` node.

The device bytes are the gate's
(``parity/meep_gpu/probe_triton_bfast_fused_magnetic_pair.py``); nothing here
launches.
"""

from __future__ import annotations

import ast
import importlib
import pathlib
import sys
from typing import List

import numpy
import pytest

from meep_gpu.fields import Fields
from meep_gpu.grid import Grid
from meep_gpu.pml import PML
from meep_gpu.test_triton_kernels import PACKAGE_DIR

MODULE_NAME = "meep_gpu.triton_kernels.bfast_fused_magnetic_pair"
MODULE_PATH = PACKAGE_DIR / "bfast_fused_magnetic_pair.py"
API_ROOT = pathlib.Path(__file__).resolve().parents[1]
DRIVER_PATH = API_ROOT / "meep_gpu" / "driver.py"

#: A BFAST shear with all three components live, so no ``k1``/``k2`` word is zero
#: for a reason unrelated to the invariance gating.
BFAST_K = (0.13, -0.4, 0.07)


@pytest.fixture(scope="module")
def product():
    return importlib.import_module(MODULE_NAME)


class _Magnetic:
    field_type = "B"


class _Electric:
    field_type = "D"


def _segments(path: pathlib.Path, name: str) -> List[str]:
    """One shipped kernel's BODY as a list of top-level STATEMENT sources.

    Read from the FILE rather than imported: the merge bar has no Triton. The text
    is EXACT — never ``ast.unparse``d, because what is being compared includes the
    PARENTHESISATION and unparsing re-derives minimal parentheses, which would
    silently equate ``dtdx * ((c_y - c) + (b - b_z))`` with a different float32
    grouping.

    Continuation lines are dedented by the function's own ``col_offset`` so a body
    nested inside ``if triton is not None:`` compares equal to one at module level,
    and comment-only lines and trailing comments are dropped so a prose edit is not
    a transcription change.
    """
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    node = next((found for found in ast.walk(tree)
                 if isinstance(found, ast.FunctionDef) and found.name == name), None)
    assert node is not None, f"{name} is not defined in {path}"
    body = node.body
    if (isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    margin = node.col_offset
    out: List[str] = []
    for statement in body:
        segment = ast.get_source_segment(text, statement)
        assert segment is not None, f"{name}: unavailable source segment"
        lines = segment.splitlines()
        lines = [lines[0]] + [line[margin:] if line[:margin].strip() == ""
                              else line.lstrip() for line in lines[1:]]
        kept = [line.split("#", 1)[0].rstrip() for line in lines]
        out.append("\n".join(line for line in kept if line.strip()))
    return out


@pytest.fixture(scope="module")
def bodies():
    return {
        "fused": _segments(PACKAGE_DIR / "bfast_fused_magnetic_pair.py",
                           "bfast_fused_curl_constitutive_B"),
        "bfast_curl": _segments(PACKAGE_DIR / "bfast_curl.py",
                                "bfast_pml_curl_step"),
        "ordinary_fused": _segments(PACKAGE_DIR / "kernels.py",
                                    "fused_curl_constitutive_B"),
        "ordinary_curl": _segments(PACKAGE_DIR / "kernels.py", "pml_curl_step"),
    }


def _build(cell_size=(0.8, 0.8, 0.8), dimensions=3, boundaries=None,
           pml_thickness=2, complex_storage=False, bfast_scaled_k=BFAST_K,
           k_point=(0.0, 0.0, 0.0), seed=23, thickness=None, **grid_kwargs):
    """A real BFAST Grid/Fields/PML triple on NumPy, PML storage enabled.

    Default: real storage, an active PML, a live three-component shear, k = 0, no
    fold. Every refusal test perturbs exactly one clause off this.
    """
    grid = Grid(resolution=10.0, cell_size=cell_size, dimensions=dimensions,
                boundaries=boundaries, courant=0.35, k_point=k_point,
                bfast_scaled_k=bfast_scaled_k, **grid_kwargs)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.set_background_eps(2.25)
    fields.enable_pml_storage()
    rng = numpy.random.default_rng(seed)
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
                 "f_w_Hx", "f_w_Hy", "f_w_Hz",
                 "f_bfast_Bx", "f_bfast_By", "f_bfast_Bz"):
        array = getattr(fields, name, None)
        if array is not None:
            array[...] = rng.uniform(-0.4, 0.4, size=grid.shape).astype(numpy.float32)
    if thickness is None:
        thickness = tuple(
            (pml_thickness, pml_thickness) if grid.shape[axis] >= 6 else (0, 0)
            for axis in range(3))
    return fields, PML(grid=grid, thickness=thickness)


def _reasons(product, fields, pml, sources=()):
    verdict = product.bfast_fused_magnetic_pair_coverage(fields, pml, sources)
    return [r for r in verdict.reasons if "array module" not in r]


# ---------------------------------------------------------------------------
# THE TRANSCRIPTION — the claim this whole product rests on
# ---------------------------------------------------------------------------

def test_the_bfast_tail_is_exactly_one_statement(bodies):
    """The premise of the per-statement comparison, stated first."""
    tail = [s for s in bodies["bfast_curl"] if s.startswith("if HAS_BFAST:")]
    assert len(tail) == 1, tail
    delta = [s for s in bodies["bfast_curl"] if s not in bodies["ordinary_curl"]]
    assert delta == tail, [s[:60] for s in delta]


def test_removing_the_bfast_tail_reproduces_the_shipped_fused_pair(bodies):
    """``this kernel`` minus the tail IS ``fused_curl_constitutive_B``.

    An EXACT statement-list equality, in order. This is the diff a reviewer would
    do by hand, made mechanical.
    """
    without_tail = [s for s in bodies["fused"] if not s.startswith("if HAS_BFAST:")]
    assert without_tail == bodies["ordinary_fused"], (
        "not fused_curl_constitutive_B plus the tail; first difference at index "
        + str(next((i for i, (a, b) in enumerate(
            zip(without_tail, bodies["ordinary_fused"])) if a != b), "length")))


def test_removing_the_weld_reproduces_the_shipped_bfast_curl(bodies):
    """``this kernel`` minus the weld IS ``bfast_curl.bfast_pml_curl_step``.

    The removed set is DERIVED — whatever ``fused_curl_constitutive_B`` has that
    ``pml_curl_step`` does not — so a change to the weld moves this test with it
    rather than requiring a hand-maintained list.
    """
    weld = [s for s in bodies["ordinary_fused"] if s not in bodies["ordinary_curl"]]
    assert weld, "the weld is empty; this test compares nothing"
    without_weld = [s for s in bodies["fused"] if s not in weld]
    assert without_weld == bodies["bfast_curl"], (
        "not bfast_pml_curl_step plus the weld; first difference at index "
        + str(next((i for i, (a, b) in enumerate(
            zip(without_weld, bodies["bfast_curl"])) if a != b), "length")))


def test_every_statement_comes_from_a_shipped_body(bodies):
    """Nothing in this kernel was written here."""
    shipped = set(bodies["bfast_curl"]) | set(bodies["ordinary_fused"])
    invented = [s for s in bodies["fused"] if s not in shipped]
    assert invented == [], [s[:80] for s in invented]


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------

def test_the_package_does_not_import_the_module_eagerly():
    for name in list(sys.modules):
        if name == MODULE_NAME:
            del sys.modules[name]
    importlib.import_module("meep_gpu.triton_kernels")
    eagerly_imported = MODULE_NAME in sys.modules
    importlib.import_module(MODULE_NAME)  # restore for the other tests
    assert not eagerly_imported


def test_the_module_answers_coverage_but_the_kernel_fails_clearly_without_triton(product):
    if product.bfast_fused_curl_constitutive_B is not None:
        pytest.skip("triton is installed here; the absence arm is exercised on "
                    "the merge-bar box without it")
    fields, pml = _build()
    assert product.bfast_fused_magnetic_pair_coverage(fields, pml, ()).reasons
    with pytest.raises(ImportError, match="triton"):
        product.bfast_fused_curl_constitutive_B_kernel()


def test_backward_matches_the_sub_step_table(product):
    from meep_gpu.triton_kernels import launch  # noqa: PLC0415

    assert product.BACKWARD == int(launch.SUB_STEPS[product.CURL_SUB_STEP]["backward"])
    assert product.BACKWARD == 0


def test_replaces_is_the_drivers_own_call_order(product):
    text = DRIVER_PATH.read_text(encoding="utf-8")
    positions = []
    for name in product.REPLACES:
        index = text.find(f"{name}(self.fields")
        assert index > 0, f"driver.py never calls {name}(self.fields...)"
        positions.append(index)
    assert positions == sorted(positions), product.REPLACES


# ---------------------------------------------------------------------------
# Admission and refusals
# ---------------------------------------------------------------------------

def test_the_bfast_family_is_admitted(product):
    fields, pml = _build()
    assert _reasons(product, fields, pml, (_Electric(),)) == []


def test_a_walled_run_is_admitted(product):
    """The wall clear is carried inline, so a walled BFAST run must stay covered."""
    fields, pml = _build(boundaries="metallic")
    assert _reasons(product, fields, pml, ()) == []


@pytest.mark.parametrize("sources,needle", [
    (None, "was not declared"),
    ((_Magnetic(),), "is magnetic"),
    ((_Electric(), _Magnetic()), "is magnetic"),
])
def test_the_source_seam_refuses_by_name(product, sources, needle):
    fields, pml = _build()
    reasons = _reasons(product, fields, pml, sources)
    assert any(needle in reason for reason in reasons), reasons


def test_a_bfast_inactive_run_is_refused(product):
    """The inverted clause: a shear-free run is the ordinary fused pair's."""
    fields, pml = _build(bfast_scaled_k=(0.0, 0.0, 0.0))
    reasons = _reasons(product, fields, pml, ())
    assert any("BFAST" in reason or "bfast" in reason for reason in reasons), reasons


def test_complex_storage_is_refused(product):
    fields, pml = _build(complex_storage=True)
    reasons = _reasons(product, fields, pml, ())
    assert any("force_complex_fields" in reason for reason in reasons), reasons


def test_an_inactive_absorber_is_refused(product):
    fields, pml = _build(pml_thickness=0)
    reasons = _reasons(product, fields, pml, ())
    assert any("PML" in reason for reason in reasons), reasons


def test_a_fold_is_refused_by_name(product):
    from meep_gpu.grid import Mirror  # noqa: PLC0415

    fields, pml = _build(symmetry=(Mirror("Y", 1),),
                         thickness=((2, 2), (0, 2), (2, 2)))
    reasons = _reasons(product, fields, pml, ())
    assert any("mirror plane" in reason or "folded" in reason
               for reason in reasons), reasons


def test_a_grid_that_cannot_answer_the_invariance_question_is_refused(product):
    """The six k1/k2 words are gated on ``grid.is_invariant``."""
    fields, pml = _build()

    class _Blind:
        def __getattribute__(self, name):
            if name == "is_invariant":
                return None
            return getattr(fields.grid, name)

    class _Blinded:
        grid = _Blind()

        def __getattr__(self, name):
            return getattr(fields, name)

    reasons = _reasons(product, _Blinded(), pml, ())
    assert any("is_invariant" in reason for reason in reasons), reasons


# ---------------------------------------------------------------------------
# Disjointness from the ordinary fused pair, in BOTH directions
# ---------------------------------------------------------------------------

def test_this_arm_and_the_ordinary_fused_pair_never_co_admit(product):
    from meep_gpu.triton_kernels import coverage  # noqa: PLC0415

    for shear in (BFAST_K, (0.0, 0.0, 0.0)):
        fields, pml = _build(bfast_scaled_k=shear)
        mine = [r for r in product.bfast_fused_magnetic_pair_coverage(
            fields, pml, ()).reasons if "array module" not in r]
        theirs = [r for r in coverage.fused_pair_coverage(
            fields, pml, "B", ()).reasons if "array module" not in r]
        assert mine or theirs, (
            f"both predicates admitted a bfast_scaled_k={shear!r} run: the arm "
            f"table would leave the slot unselected")
        if shear == (0.0, 0.0, 0.0):
            assert mine, "a shear-free run must be refused here"
        else:
            assert mine == [] and theirs, "a BFAST run is this arm's"


# ---------------------------------------------------------------------------
# The weld is never wider than either half
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs,sources", [
    ({}, ()),
    ({}, (_Electric(),)),
    ({}, (_Magnetic(),)),
    ({}, None),
    ({"bfast_scaled_k": (0.0, 0.0, 0.0)}, ()),
    ({"complex_storage": True}, ()),
    ({"pml_thickness": 0}, ()),
    ({"boundaries": "metallic"}, ()),
    ({"k_point": (0.2, 0.0, 0.0)}, ()),
])
def test_the_weld_is_never_wider_than_the_halves(product, kwargs, sources):
    from meep_gpu.triton_kernels import bfast_curl  # noqa: PLC0415

    fields, pml = _build(**kwargs)
    weld = product.bfast_fused_magnetic_pair_coverage(fields, pml, sources)
    curl = bfast_curl.bfast_pml_curl_coverage(fields, pml, "step_B")
    magnetic = bfast_curl.bfast_run_constitutive_coverage(fields, pml, "H")
    assert (not weld.covered) or (curl.covered and magnetic.covered), (
        weld.reasons, curl.reasons, magnetic.reasons)


def test_the_plan_refuses_rather_than_raises_on_a_numpy_host(product):
    fields, pml = _build()
    assert product.plan_bfast_fused_magnetic_pair(fields, pml, ()) is None
    assert product.plan_bfast_fused_magnetic_pair(fields, pml, None) is None
