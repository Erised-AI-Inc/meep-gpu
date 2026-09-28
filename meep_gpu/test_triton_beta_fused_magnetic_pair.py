"""Laptop contracts for the REAL-beta fused magnetic B/H pair on Triton.

Everything here runs on the NumPy merge-bar machine. The load-bearing one is the
TRANSCRIPTION leg: this product claims to be
``kernels.fused_curl_constitutive_B`` plus exactly the three lines
``special_kz.beta_pml_curl_step`` adds to ``kernels.pml_curl_step``, and that claim
is checked here as two EXACT statement-list equalities against the shipped sources
rather than by reading the docstring. The device bytes are the gate's
(``parity/meep_gpu/probe_triton_beta_fused_magnetic_pair.py``); nothing here
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

MODULE_NAME = "meep_gpu.triton_kernels.beta_fused_magnetic_pair"
MODULE_PATH = PACKAGE_DIR / "beta_fused_magnetic_pair.py"
API_ROOT = pathlib.Path(__file__).resolve().parents[1]
DRIVER_PATH = API_ROOT / "meep_gpu" / "driver.py"

#: refl-angular-kz2d.py's own beta, to five places.
BETA_CORPUS = 0.3321611318837033

#: The three lines the beta term adds, and the ONLY thing that may separate this
#: kernel from ``kernels.fused_curl_constitutive_B``.
BETA_INSERT = (
    "if HAS_BETA:",
    "curl0 = curl0 - (beta_plus * b)",
    "curl1 = curl1 - (beta_minus * a)",
)

#: The wall clear plus the whole constitutive half — the ONLY thing that may
#: separate this kernel from ``special_kz.beta_pml_curl_step``. Derived below
#: rather than listed: it is whatever ``fused_curl_constitutive_B`` has and
#: ``pml_curl_step`` does not.


@pytest.fixture(scope="module")
def product():
    return importlib.import_module(MODULE_NAME)


class _Magnetic:
    field_type = "B"


class _Electric:
    field_type = "D"


def _statements(text: str) -> List[str]:
    """Executable lines: comments and blanks removed, indentation normalised.

    Indentation is DROPPED deliberately. What these comparisons are about is the
    arithmetic and its order; a body re-indented by one level is the same
    transcription, and a body whose statements differ is not.
    """
    out: List[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if line.strip():
            out.append(line.strip())
    return out


def _shipped_body(path: pathlib.Path, name: str) -> List[str]:
    """One shipped kernel's BODY statements, docstring and signature removed.

    Read from the FILE rather than imported: the merge bar has no Triton, so
    importing ``kernels.py`` raises and this check would only ever bite on a
    device run.

    The text is EXACT — never ``ast.unparse``d. What is being compared includes
    the PARENTHESISATION, and unparsing re-derives minimal parentheses, which
    would silently equate ``dtdx * ((c_y - c) + (b - b_z))`` with a different
    float32 grouping.
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
    segments = [ast.get_source_segment(text, statement) for statement in body]
    assert all(segment is not None for segment in segments), name
    return _statements("\n".join(segments))


@pytest.fixture(scope="module")
def bodies():
    return {
        "fused": _shipped_body(PACKAGE_DIR / "beta_fused_magnetic_pair.py",
                               "beta_fused_curl_constitutive_B"),
        "beta_curl": _shipped_body(PACKAGE_DIR / "special_kz.py",
                                   "beta_pml_curl_step"),
        "ordinary_fused": _shipped_body(PACKAGE_DIR / "kernels.py",
                                        "fused_curl_constitutive_B"),
        "ordinary_curl": _shipped_body(PACKAGE_DIR / "kernels.py",
                                       "pml_curl_step"),
    }


def _build(cell_size=(1.2, 1.0, 0.0), dimensions=2, boundaries=None,
           pml_thickness=2, complex_storage=False, beta=BETA_CORPUS,
           k_point=(0.0, 0.0, 0.0), seed=17, thickness=None, **grid_kwargs):
    """A real beta Grid/Fields/PML triple on NumPy, PML storage enabled.

    Default: the corpus family — 2-D Cartesian (which is the ONLY place MEEP
    allows beta, fields.cpp:546-547), real storage, an active PML, k = 0, no fold.
    Every refusal test perturbs exactly one clause off this.
    """
    grid = Grid(resolution=10.0, cell_size=cell_size, dimensions=dimensions,
                boundaries=boundaries, courant=0.35, k_point=k_point, beta=beta,
                **grid_kwargs)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.set_background_eps(2.25)
    fields.enable_pml_storage()
    rng = numpy.random.default_rng(seed)
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
                 "f_w_Hx", "f_w_Hy", "f_w_Hz"):
        getattr(fields, name)[...] = rng.uniform(
            -0.4, 0.4, size=grid.shape).astype(numpy.float32)
    # A folded axis cannot carry a LOW-face layer: cell 0 there is the mirror
    # plane, and pml._resolve_mirror_faces RAISES on a per-side request that names
    # it. The folded case passes its own table for that reason.
    if thickness is None:
        thickness = tuple(
            (pml_thickness, pml_thickness) if grid.shape[axis] >= 6 else (0, 0)
            for axis in range(3))
    return fields, PML(grid=grid, thickness=thickness)


def _reasons(product, fields, pml, sources=()):
    verdict = product.beta_fused_magnetic_pair_coverage(fields, pml, sources)
    return [r for r in verdict.reasons if "array module" not in r]


# ---------------------------------------------------------------------------
# THE TRANSCRIPTION — the claim this whole product rests on
# ---------------------------------------------------------------------------

def test_removing_the_beta_insert_reproduces_the_shipped_fused_pair(bodies):
    """``this kernel`` minus the three beta lines IS ``fused_curl_constitutive_B``.

    An EXACT statement-list equality, in order. This is the diff a reviewer would
    do by hand, made mechanical: if any other line moved, this fails and names it.
    """
    without_beta = [line for line in bodies["fused"] if line not in BETA_INSERT]
    assert without_beta == bodies["ordinary_fused"], (
        "the beta fused pair is not fused_curl_constitutive_B plus the beta "
        "insert; first difference at index "
        + str(next((i for i, (a, b) in enumerate(
            zip(without_beta, bodies["ordinary_fused"])) if a != b), "length")))


def test_removing_the_weld_reproduces_the_shipped_beta_curl(bodies):
    """``this kernel`` minus the wall clear and the constitutive half IS
    ``special_kz.beta_pml_curl_step``.

    The removed set is DERIVED — it is whatever ``fused_curl_constitutive_B`` has
    that ``pml_curl_step`` does not — so a change to the weld moves this test with
    it rather than requiring the list to be maintained by hand.
    """
    weld = [line for line in bodies["ordinary_fused"]
            if line not in bodies["ordinary_curl"]]
    assert weld, "the weld is empty; this test compares nothing"
    without_weld = [line for line in bodies["fused"] if line not in weld]
    assert without_weld == bodies["beta_curl"], (
        "the beta fused pair minus the weld is not beta_pml_curl_step; first "
        "difference at index "
        + str(next((i for i, (a, b) in enumerate(
            zip(without_weld, bodies["beta_curl"])) if a != b), "length")))


def test_the_beta_insert_is_exactly_the_shipped_curls_delta(bodies):
    """The three lines are ``beta_pml_curl_step``'s own delta over ``pml_curl_step``."""
    delta = [line for line in bodies["beta_curl"]
             if line not in bodies["ordinary_curl"]]
    assert delta == list(BETA_INSERT), delta


def test_every_statement_comes_from_a_shipped_body(bodies):
    """Nothing in this kernel was written here."""
    shipped = set(bodies["beta_curl"]) | set(bodies["ordinary_fused"])
    invented = [line for line in bodies["fused"] if line not in shipped]
    assert invented == [], invented


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
    if product.beta_fused_curl_constitutive_B is not None:
        pytest.skip("triton is installed here; the absence arm is exercised on "
                    "the merge-bar box without it")
    fields, pml = _build()
    assert product.beta_fused_magnetic_pair_coverage(fields, pml, ()).reasons
    with pytest.raises(ImportError, match="triton"):
        product.beta_fused_curl_constitutive_B_kernel()


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

def test_the_corpus_family_is_admitted(product):
    """2-D Cartesian, nonzero beta, active PML, unfolded, electric source only."""
    fields, pml = _build()
    assert _reasons(product, fields, pml, (_Electric(),)) == []


def test_a_walled_run_is_admitted(product):
    """The wall clear is carried inline, so a walled beta run must stay covered.

    z stays PERIODIC: on a 2-D cell z is the INVARIANT axis and ``Grid`` refuses a
    PEC there by name (grid.py:842-877).
    """
    fields, pml = _build(boundaries={"x": "metallic", "y": "metallic",
                                     "z": "periodic"})
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


def test_a_zero_beta_run_is_refused_by_name(product):
    """The inverted clause: a beta = 0 run is the ordinary fused pair's."""
    fields, pml = _build(beta=0.0)
    reasons = _reasons(product, fields, pml, ())
    assert any("grid.beta is zero" in reason for reason in reasons), reasons


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
                         thickness=((2, 2), (0, 2), (0, 0)))
    reasons = _reasons(product, fields, pml, ())
    assert any("mirror plane" in reason for reason in reasons), reasons


def test_bloch_is_refused(product):
    fields, pml = _build(k_point=(0.2, 0.0, 0.0))
    reasons = _reasons(product, fields, pml, ())
    assert any("k_point" in reason for reason in reasons), reasons


# ---------------------------------------------------------------------------
# Disjointness from the ordinary fused pair, in BOTH directions
# ---------------------------------------------------------------------------

def test_this_arm_and_the_ordinary_fused_pair_never_co_admit(product):
    from meep_gpu.triton_kernels import coverage  # noqa: PLC0415

    for beta in (BETA_CORPUS, 0.0):
        fields, pml = _build(beta=beta)
        mine = [r for r in product.beta_fused_magnetic_pair_coverage(
            fields, pml, ()).reasons if "array module" not in r]
        theirs = [r for r in coverage.fused_pair_coverage(
            fields, pml, "B", ()).reasons if "array module" not in r]
        assert mine or theirs, (
            f"both predicates admitted a beta={beta!r} run: the arm table would "
            f"leave the slot unselected")
        if beta == 0.0:
            assert mine, "a beta = 0 run must be refused here"
        else:
            assert mine == [] and theirs, "a beta run is this arm's"


# ---------------------------------------------------------------------------
# The weld is never wider than either half
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs,sources", [
    ({}, ()),
    ({}, (_Electric(),)),
    ({}, (_Magnetic(),)),
    ({}, None),
    ({"beta": 0.0}, ()),
    ({"complex_storage": True}, ()),
    ({"pml_thickness": 0}, ()),
    ({"boundaries": {"x": "metallic", "y": "metallic", "z": "periodic"}}, ()),
    ({"k_point": (0.2, 0.0, 0.0)}, ()),
])
def test_the_weld_is_never_wider_than_the_halves(product, kwargs, sources):
    from meep_gpu.triton_kernels import special_kz  # noqa: PLC0415

    fields, pml = _build(**kwargs)
    weld = product.beta_fused_magnetic_pair_coverage(fields, pml, sources)
    curl = special_kz.beta_pml_curl_coverage(fields, pml, "step_B")
    magnetic = special_kz.beta_run_constitutive_coverage(fields, pml, "H")
    assert (not weld.covered) or (curl.covered and magnetic.covered), (
        weld.reasons, curl.reasons, magnetic.reasons)


def test_the_plan_refuses_rather_than_raises_on_a_numpy_host(product):
    fields, pml = _build()
    assert product.plan_beta_fused_magnetic_pair(fields, pml, ()) is None
    assert product.plan_beta_fused_magnetic_pair(fields, pml, None) is None
