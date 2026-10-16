"""Laptop contracts for the COMPLEX-beta fused magnetic B/H pair on Triton.

Everything here runs on the NumPy merge-bar machine. The load-bearing leg is the
TRANSCRIPTION: this product claims to be
``complex_fused_magnetic_pair.complex_fused_curl_constitutive_B`` plus exactly
the statements ``special_kz.beta_bloch_pml_curl_step`` adds to
``complex_fields.bloch_pml_curl_step``, and that claim is checked here as EXACT
statement-list equalities against the shipped sources rather than by reading the
docstring. The device bytes are the gate's
(``parity/meep_gpu/probe_triton_complex_beta_fused_magnetic_pair.py``); nothing
here launches.
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

MODULE_NAME = "meep_gpu.triton_kernels.complex_beta_fused_magnetic_pair"
API_ROOT = pathlib.Path(__file__).resolve().parents[1]
DRIVER_PATH = API_ROOT / "meep_gpu" / "driver.py"

#: TestSpecialKz.test_special_kz's own beta (census ``configuration.beta``).
BETA_CORPUS = -0.39073112848927377

@pytest.fixture(scope="module")
def product():
    return importlib.import_module(MODULE_NAME)


class _Magnetic:
    field_type = "B"


class _Electric:
    field_type = "D"


def _statements(text: str) -> List[str]:
    """Executable lines: comments and blanks removed, indentation normalised."""
    out: List[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if line.strip():
            out.append(line.strip())
    return out


def _shipped_body(path: pathlib.Path, name: str) -> List[str]:
    """One shipped kernel's BODY statements, docstring and signature removed.

    Read from the FILE rather than imported (the merge bar has no Triton), and
    the text is EXACT — never ``ast.unparse``d: the comparison includes the
    PARENTHESISATION.
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
        "fused": _shipped_body(
            PACKAGE_DIR / "complex_beta_fused_magnetic_pair.py",
            "complex_beta_fused_curl_constitutive_B"),
        "beta_curl": _shipped_body(PACKAGE_DIR / "special_kz.py",
                                   "beta_bloch_pml_curl_step"),
        "twin_fused": _shipped_body(PACKAGE_DIR / "complex_fused_magnetic_pair.py",
                                    "complex_fused_curl_constitutive_B"),
        "bloch_curl": _shipped_body(PACKAGE_DIR / "complex_fields.py",
                                    "bloch_pml_curl_step"),
    }


def _build(cell_size=(2.0, 0.4, 0.0), boundaries=None, complex_storage=True,
           beta=BETA_CORPUS, k_point=(0.9205, 0.0, 0.0), seed=17,
           thickness=None, pml_thickness=2, symmetry=(), **grid_kwargs):
    """A complex beta Grid/Fields/PML triple on NumPy, PML storage enabled.

    Default: the corpus family — 2-D Cartesian (the only place MEEP allows
    beta), complex storage via an in-plane Bloch kx, an active PML, no fold.
    Every refusal test perturbs exactly one clause off this.
    """
    grid = Grid(resolution=10.0, cell_size=cell_size, dimensions=2,
                boundaries=boundaries, courant=0.35, k_point=k_point, beta=beta,
                symmetry=symmetry, **grid_kwargs)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.set_background_eps(2.25)
    fields.enable_pml_storage()
    rng = numpy.random.default_rng(seed)
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez"):
        real = rng.uniform(-0.4, 0.4, size=grid.shape).astype(numpy.float32)
        getattr(fields, name)[...] = real
    if thickness is None:
        thickness = tuple(
            (pml_thickness, pml_thickness) if grid.shape[axis] >= 6 else (0, 0)
            for axis in range(3))
    return fields, PML(grid=grid, thickness=thickness)


def _reasons(product, fields, pml, sources=()):
    verdict = product.complex_beta_fused_magnetic_pair_coverage(
        fields, pml, sources)
    return [r for r in verdict.reasons if "array module" not in r]


# ---------------------------------------------------------------------------
# THE TRANSCRIPTION — the claim this whole product rests on
# ---------------------------------------------------------------------------

def test_removing_the_beta_insert_reproduces_the_shipped_complex_fused_pair(bodies):
    """``this kernel`` minus its ``if HAS_BETA:`` pass IS ``complex_fused_curl_constitutive_B``.

    An EXACT statement-list equality, in order, against the shipped body read from the
    file (``test_triton_special_kz.kernel_body_and_insert``). If any statement of the
    root moved, this fails and names the first difference.
    """
    from meep_gpu.test_triton_special_kz import kernel_body_and_insert

    body, insert = kernel_body_and_insert(PACKAGE_DIR / "complex_beta_fused_magnetic_pair.py", "complex_beta_fused_curl_constitutive_B")
    root, root_insert = kernel_body_and_insert(PACKAGE_DIR / "complex_fused_magnetic_pair.py",
                                               "complex_fused_curl_constitutive_B")
    assert insert and not root_insert, "the beta pass is missing, or the root has one"
    assert body == root, (
        "first difference at index "
        + str(next((i for i, (a, b) in enumerate(zip(body, root)) if a != b),
                   "length")))


def test_the_beta_insert_is_exactly_the_shipped_curls_delta(bodies):
    """The pass is ``beta_bloch_pml_curl_step``'s own, statement for statement, and that curl is
    ``bloch_pml_curl_step`` plus exactly that pass: the lifter adds nothing of its own
    to the beta pass (the increment, its ownership mask and the ladder)."""
    from meep_gpu.test_triton_special_kz import kernel_body_and_insert

    _body, insert = kernel_body_and_insert(PACKAGE_DIR / "complex_beta_fused_magnetic_pair.py", "complex_beta_fused_curl_constitutive_B")
    curl_body, curl_insert = kernel_body_and_insert(PACKAGE_DIR / "special_kz.py",
                                                    "beta_bloch_pml_curl_step")
    root_curl, _ = kernel_body_and_insert(PACKAGE_DIR / "complex_fields.py",
                                          "bloch_pml_curl_step")
    assert curl_insert and insert == curl_insert, (insert[:3], curl_insert[:3])
    assert curl_body == root_curl


def test_removing_the_weld_reproduces_the_shipped_beta_bloch_curl(bodies):
    """``this kernel`` minus the weld (wall clear + constitutive half) IS
    ``special_kz.beta_bloch_pml_curl_step``.

    The removed set is DERIVED — whatever the beta-less twin has that
    ``bloch_pml_curl_step`` does not — so a change to the weld moves this test
    with it rather than requiring a hand-maintained list.
    """
    weld = [line for line in bodies["twin_fused"]
            if line not in bodies["bloch_curl"]]
    assert weld, "the weld is empty; this test compares nothing"
    without_weld = [line for line in bodies["fused"] if line not in weld]
    assert without_weld == bodies["beta_curl"], (
        "the complex beta fused pair minus the weld is not "
        "beta_bloch_pml_curl_step; first difference at index "
        + str(next((i for i, (a, b) in enumerate(
            zip(without_weld, bodies["beta_curl"])) if a != b), "length")))


def test_every_statement_comes_from_a_shipped_body(bodies):
    """Nothing in this kernel was written here."""
    shipped = set(bodies["beta_curl"]) | set(bodies["twin_fused"])
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


def test_backward_matches_the_sub_step_table(product):
    from meep_gpu.triton_kernels import launch  # noqa: PLC0415

    assert product.BACKWARD == int(
        launch.SUB_STEPS[product.CURL_SUB_STEP]["backward"])
    assert product.BACKWARD == 0


def test_replaces_is_the_drivers_own_call_order(product):
    text = DRIVER_PATH.read_text(encoding="utf-8")
    positions = []
    for name in product.REPLACES:
        index = text.find(f"{name}(self.fields")
        assert index > 0, f"driver.py never calls {name}(self.fields...)"
        positions.append(index)
    assert positions == sorted(positions), product.REPLACES


def test_the_kernel_body_calls_only_the_licensed_connection_helper(product):
    """The only complex helper the kernel calls is the Bloch connection; every other
    product, the beta pass's quarter turn included, has a real operand and is formed
    part by part inline."""
    text = (PACKAGE_DIR / "complex_beta_fused_magnetic_pair.py").read_text(
        encoding="utf-8")
    tree = ast.parse(text)
    kernel = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef)
                  and node.name == "complex_beta_fused_curl_constitutive_B")
    called = {node.func.id for node in ast.walk(kernel)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    helpers = {name for name in called if name.startswith(("_mul", "_rotate", "_connect"))}
    assert helpers == set(product.MULTIPLY_HELPERS) == {"_connect"}, helpers


def test_the_beta_words_are_rounded_exactly_as_the_array_path_rounds_them(product):
    """The plan binds ``beta_curl_coefficients(..., magnetic=True,
    complex_storage=True)``: one real word per sign, the coefficient itself (the B
    side's quarter turn by +1j), rounded once -- no complex coefficient."""
    from meep_gpu.triton_kernels import special_kz  # noqa: PLC0415

    words = special_kz.beta_curl_coefficients(BETA_CORPUS, 0.035, True, True)
    expected = tuple(float(numpy.float32(sign * 2.0 * numpy.pi * BETA_CORPUS * 0.035))
                     for sign in (1.0, -1.0))
    assert tuple(words) == expected


# ---------------------------------------------------------------------------
# Admission and refusals
# ---------------------------------------------------------------------------

def test_the_corpus_family_is_admitted(product):
    """2-D Cartesian, nonzero beta, in-plane kx, complex storage, active PML,
    unfolded, electric source only — test_special_kz's own configuration."""
    fields, pml = _build()
    assert _reasons(product, fields, pml, (_Electric(),)) == []


def test_a_walled_run_is_admitted(product):
    """The wall clear is carried inline, so a walled complex beta run must stay
    covered. Metallic axes carry k = 0 (the phase-consistency clause)."""
    fields, pml = _build(boundaries={"x": "metallic", "y": "metallic",
                                     "z": "periodic"}, k_point=(0.0, 0.0, 0.0))
    assert _reasons(product, fields, pml, ()) == []


def test_an_undeclared_source_list_is_a_refusal_and_not_an_empty_set(product):
    fields, pml = _build()
    reasons = _reasons(product, fields, pml, None)
    assert any("was not declared" in reason for reason in reasons), reasons


def test_a_magnetic_source_without_an_index_is_refused_by_name(product):
    fields, pml = _build()
    reasons = _reasons(product, fields, pml, (_Magnetic(),))
    assert any("does not publish the index" in reason for reason in reasons), reasons


def test_a_zero_beta_run_is_refused_by_name(product):
    fields, pml = _build(beta=0.0)
    reasons = _reasons(product, fields, pml, ())
    assert any("beta is zero" in reason for reason in reasons), reasons


def test_real_storage_is_refused_this_is_the_complex_arm(product):
    fields, pml = _build(complex_storage=False, k_point=(0.0, 0.0, 0.0))
    reasons = _reasons(product, fields, pml, ())
    assert any("storage is real float32" in reason for reason in reasons), reasons


def test_a_fold_is_refused_by_this_products_own_clause(product):
    """The folded complex beta cell belongs to the folded product; this weld's
    own seam clause names the two symmetry passes, over and above the halves'."""
    from meep_gpu.grid import Mirror  # noqa: PLC0415

    # A folded axis cannot carry a LOW-face layer (cell 0 is the mirror plane),
    # so this case passes its own thickness table: PML on x only.
    fields, pml = _build(symmetry=(Mirror("Y", 1),), k_point=(0.0, 0.0, 0.0),
                         cell_size=(2.0, 1.0, 0.0),
                         thickness=((2, 2), (0, 0), (0, 0)))
    reasons = _reasons(product, fields, pml, ())
    assert any("mirror plane is active" in reason for reason in reasons), reasons
    assert any("this weld carries neither" in reason for reason in reasons)


def test_the_predicate_reports_both_halves_reasons_with_their_side_named(product):
    fields, pml = _build(beta=0.0)
    reasons = _reasons(product, fields, pml, ())
    assert any(reason.startswith("complex beta curl half: ")
               for reason in reasons), reasons
    assert any(reason.startswith("complex beta constitutive half: ")
               for reason in reasons), reasons


def test_the_builder_refuses_every_configuration_the_predicate_refuses(product):
    """None is the only refusal; the builder may not out-admit the predicate."""
    cases = (
        {"beta": 0.0},
        {"complex_storage": False, "k_point": (0.0, 0.0, 0.0)},
    )
    for keywords in cases:
        fields, pml = _build(**keywords)
        covered = product.complex_beta_fused_magnetic_pair_coverage(
            fields, pml, ()).covered
        plan = product.plan_complex_beta_fused_magnetic_pair(
            fields, pml, ())
        assert not covered
        assert plan is None, keywords


def test_the_plan_refuses_a_cell_count_whose_word_index_overflows_int32(product):
    with pytest.raises(ValueError, match="int32"):
        product.ComplexBetaFusedMagneticPairPlan(
            (2048, 1024, 1024), 0.5, (0, 0, 0), (False, False, False),
            (0, 0, 0), (0.0,) * 6, (0.1, -0.1), 256,
            [], [], [], [], [], [], [], chunk_bounds=None)


def test_the_from_arrays_route_carries_the_identity_arm(product):
    """``has_beta=0`` is the gate's certified-kernel arm; the engine route never
    binds it (the predicate requires beta nonzero)."""
    import inspect  # noqa: PLC0415

    signature = inspect.signature(
        product.plan_complex_beta_fused_magnetic_pair_from_arrays)
    assert signature.parameters["has_beta"].default == 1


def test_the_carry_flag_and_the_seam_clause_move_together(product):
    """The flag is passed into ``seam_source_reasons``; nothing else reads it."""
    text = (PACKAGE_DIR / "complex_beta_fused_magnetic_pair.py").read_text(
        encoding="utf-8")
    assert text.count("CARRIES_DEPOSIT_REPAIR = ") == 1
    assert "carries_repair=CARRIES_DEPOSIT_REPAIR" in text
