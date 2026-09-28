"""Laptop contracts for the NONLINEAR fused magnetic B/H pair on Triton.

Everything here runs on the NumPy merge-bar machine: the optional-import
contract, the claim that this module adds NO arithmetic, the predicate's
admission and its per-clause refusals on real ``Grid``/``Fields``/``PML``
objects, the disjointness from the ordinary fused pair in BOTH directions, and
the weld-is-never-wider-than-its-halves identity. The device bytes are the
gate's (``parity/meep_gpu/probe_triton_nonlinear_fused_magnetic_pair.py``);
nothing here launches.
"""

from __future__ import annotations

import ast
import importlib
import pathlib
import sys

import numpy
import pytest

from meep_gpu.fields import Fields
from meep_gpu.grid import Grid
from meep_gpu.pml import PML
from meep_gpu.test_triton_kernels import PACKAGE_DIR

MODULE_NAME = "meep_gpu.triton_kernels.nonlinear_fused_magnetic_pair"
MODULE_PATH = PACKAGE_DIR / "nonlinear_fused_magnetic_pair.py"
API_ROOT = pathlib.Path(__file__).resolve().parents[1]
DRIVER_PATH = API_ROOT / "meep_gpu" / "driver.py"

#: 3rd-harm-1d.py's own decade.
CHI3_CORPUS = 0.06


@pytest.fixture(scope="module")
def product():
    return importlib.import_module(MODULE_NAME)


@pytest.fixture(scope="module")
def coverage():
    return importlib.import_module("meep_gpu.triton_kernels.coverage")


class _Magnetic:
    """A source in the B family — the deposit that lands INSIDE this seam."""
    field_type = "B"


class _Electric:
    """A source in the D family — injected in the OTHER seam; admitted here."""
    field_type = "D"


def _build(cell_size=(0.8, 0.8, 0.8), dimensions=3, boundaries=None,
           pml_thickness=2, complex_storage=False, k_point=(0.0, 0.0, 0.0),
           beta=0.0, chi3=CHI3_CORPUS, seed=11, uniform_pml=False, **grid_kwargs):
    """A real nonlinear Grid/Fields/PML triple on NumPy, PML storage enabled.

    Default: the corpus family — real storage, an active PML, a scalar chi3 on
    every component, k = 0, no fold. Every refusal test perturbs exactly one
    clause off this.
    """
    grid = Grid(resolution=10.0, cell_size=cell_size, dimensions=dimensions,
                boundaries=boundaries, courant=0.35, k_point=k_point,
                beta=beta, **grid_kwargs)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.set_background_eps(2.25)
    if chi3 is not None:
        fields.set_nonlinear_volumes(
            {"Ex": 0.0, "Ey": 0.0, "Ez": 0.0},
            {"Ex": chi3, "Ey": chi3, "Ez": chi3})
    fields.enable_pml_storage()
    rng = numpy.random.default_rng(seed)
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
                 "f_w_Hx", "f_w_Hy", "f_w_Hz"):
        getattr(fields, name)[...] = rng.uniform(
            -0.4, 0.4, size=grid.shape).astype(numpy.float32)
    # A folded axis cannot carry a low-face layer (pml.py:_resolve_mirror_faces),
    # so a per-side request that names it RAISES. The scalar shorthand means
    # "every face that can hold one" and is what a folded case must pass.
    thickness = pml_thickness if uniform_pml else tuple(
        (pml_thickness, pml_thickness) if grid.shape[axis] >= 6 else (0, 0)
        for axis in range(3))
    return fields, PML(grid=grid, thickness=thickness)


def _reasons(product, fields, pml, sources=()):
    """The verdict's reasons with the NumPy-host clause discounted.

    The merge bar has no CuPy, so every predicate in the package refuses with
    "array module is 'numpy', not cupy". Filtering it is what lets this file ask
    about every OTHER clause on the machine it runs on.
    """
    verdict = product.nonlinear_fused_magnetic_pair_coverage(fields, pml, sources)
    return [r for r in verdict.reasons if "array module" not in r]


# ---------------------------------------------------------------------------
# The optional dependency, and the claim that this module adds no arithmetic
# ---------------------------------------------------------------------------

def test_the_package_does_not_import_the_module_eagerly():
    for name in list(sys.modules):
        if name == MODULE_NAME:
            del sys.modules[name]
    importlib.import_module("meep_gpu.triton_kernels")
    eagerly_imported = MODULE_NAME in sys.modules
    importlib.import_module(MODULE_NAME)  # restore for the other tests
    assert not eagerly_imported, (
        "the package __init__ must not pull this unwired module in")


def test_the_module_defines_no_kernel_and_no_plan_class():
    """The whole premise: admission only. A ``triton.jit`` here would refute it."""
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    decorated = [node.name for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef)
                 for decorator in node.decorator_list
                 if "jit" in ast.dump(decorator)]
    assert not decorated, (
        f"this module claims to add no arithmetic but decorates {decorated}")
    classes = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
    assert classes == ["_LinearScopeView"], (
        f"the plan class must be launch.FusedPairPlan's; found {classes}")
    text = MODULE_PATH.read_text(encoding="utf-8")
    assert "import triton" not in text, (
        "an admission-only module must not need the optional dependency")


def test_the_declared_kernel_is_the_shipped_one(product):
    """``KERNEL`` must name a real symbol in the shipped kernel module."""
    kernels_source = (PACKAGE_DIR / "kernels.py").read_text(encoding="utf-8")
    module, _, name = product.KERNEL.partition(".")
    assert module == "kernels"
    assert f"def {name}(" in kernels_source


def test_replaces_is_the_drivers_own_call_order(product):
    """The five call sites, in the order ``driver.step`` really makes them."""
    text = DRIVER_PATH.read_text(encoding="utf-8")
    positions = []
    for name in product.REPLACES:
        index = text.find(f"{name}(self.fields")
        assert index > 0, f"driver.py never calls {name}(self.fields...)"
        positions.append(index)
    assert positions == sorted(positions), (
        f"REPLACES is not in driver order: {product.REPLACES}")


# ---------------------------------------------------------------------------
# Admission
# ---------------------------------------------------------------------------

def test_the_corpus_family_is_admitted(product):
    """A nonlinear PML run, unfolded, electric source only — the corpus shape."""
    fields, pml = _build()
    assert _reasons(product, fields, pml, (_Electric(),)) == []


def test_a_metallic_walled_run_is_admitted(product):
    """Both corpus rows are z-metallic; the wall clear is carried inline."""
    fields, pml = _build(boundaries={"x": "periodic", "y": "periodic",
                                     "z": "metallic"})
    assert _reasons(product, fields, pml, ()) == []


# ---------------------------------------------------------------------------
# Refusals, one clause at a time
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sources,needle", [
    (None, "was not declared"),
    ((_Magnetic(),), "is magnetic"),
    ((_Electric(), _Magnetic()), "is magnetic"),
])
def test_the_source_seam_refuses_by_name(product, sources, needle):
    fields, pml = _build()
    reasons = _reasons(product, fields, pml, sources)
    assert any(needle in reason for reason in reasons), reasons


def test_a_linear_run_is_refused_by_name(product):
    """The inverted clause: this arm exists only where the ordinary one refuses."""
    fields, pml = _build(chi3=None)
    reasons = _reasons(product, fields, pml, ())
    assert any("no chi2/chi3 is installed" in reason for reason in reasons), reasons


def test_a_fold_is_refused_by_name(product):
    from meep_gpu.grid import Mirror  # noqa: PLC0415

    fields, pml = _build(symmetry=(Mirror("Y", 1),), uniform_pml=True)
    reasons = _reasons(product, fields, pml, ())
    assert any("mirror plane" in reason for reason in reasons), reasons


def test_complex_storage_is_refused(product):
    fields, pml = _build(complex_storage=True)
    reasons = _reasons(product, fields, pml, ())
    assert any("force_complex_fields" in reason for reason in reasons), reasons


def test_a_beta_run_is_refused(product):
    # beta is MEEP's out-of-plane 2-D wavevector and Grid refuses it anywhere
    # else by name (grid.py:690, fields.cpp:546-547), so the case must be 2-D.
    fields, pml = _build(cell_size=(0.8, 0.8, 0.0), dimensions=2, beta=0.31)
    reasons = _reasons(product, fields, pml, ())
    assert any("beta" in reason for reason in reasons), reasons


def test_an_inactive_absorber_is_refused(product):
    fields, pml = _build(pml_thickness=0)
    reasons = _reasons(product, fields, pml, ())
    assert any("PML" in reason or "pml" in reason for reason in reasons), reasons


def test_a_grid_that_cannot_answer_the_wall_question_is_refused(product):
    """The inline ``zero_metal_B`` carry needs the grid's own declaration."""
    fields, pml = _build()

    class _Blind:
        def __getattr__(self, name):
            if name in ("has_metallic", "is_metallic", "is_mirrored"):
                raise AttributeError(name)
            return getattr(fields.grid, name)

        def __getattribute__(self, name):
            if name in ("has_metallic", "is_metallic", "is_mirrored"):
                return None
            return object.__getattribute__(self, name)

    class _Blinded:
        grid = _Blind()

        def __getattr__(self, name):
            return getattr(fields, name)

    reasons = _reasons(product, _Blinded(), pml, ())
    assert any("zero_metal_B cannot be carried inline" in reason
               for reason in reasons), reasons


# ---------------------------------------------------------------------------
# Disjointness from the ordinary fused pair — in BOTH directions
# ---------------------------------------------------------------------------

def test_this_arm_and_the_ordinary_fused_pair_never_co_admit(product, coverage):
    """Two admitters on one configuration is a dispatcher picking by ordering."""
    for chi3 in (CHI3_CORPUS, None):
        fields, pml = _build(chi3=chi3)
        mine = [r for r in product.nonlinear_fused_magnetic_pair_coverage(
            fields, pml, ()).reasons if "array module" not in r]
        theirs = [r for r in coverage.fused_pair_coverage(
            fields, pml, "B", ()).reasons if "array module" not in r]
        assert mine or theirs, (
            f"both predicates admitted a chi3={chi3!r} run: the arm table would "
            f"leave the slot unselected")
        if chi3 is None:
            assert theirs == [] and mine, "the linear run is the ordinary pair's"
        else:
            assert mine == [] and theirs, "the nonlinear run is this arm's"


# ---------------------------------------------------------------------------
# The identity the module rests on
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs,sources", [
    ({}, ()),
    ({}, (_Electric(),)),
    ({}, (_Magnetic(),)),
    ({}, None),
    ({"chi3": None}, ()),
    ({"complex_storage": True}, ()),
    ({"cell_size": (0.8, 0.8, 0.0), "dimensions": 2, "beta": 0.31}, ()),
    ({"pml_thickness": 0}, ()),
    ({"boundaries": {"x": "periodic", "y": "periodic", "z": "metallic"}}, ()),
    ({"k_point": (0.2, 0.0, 0.0)}, ()),
])
def test_the_weld_is_never_wider_than_the_halves(product, kwargs, sources):
    fields, pml = _build(**kwargs)
    report = product.nonlinear_fused_magnetic_pair_equivalence(fields, pml, sources)
    assert report["weld_never_wider_than_the_halves"], report


def test_the_plan_refuses_rather_than_raises_on_a_numpy_host(product):
    """``None`` is the only refusal; a raise would break the fallback contract."""
    fields, pml = _build()
    assert product.plan_nonlinear_fused_magnetic_pair(fields, pml, ()) is None
    assert product.plan_nonlinear_fused_magnetic_pair(fields, pml, None) is None
