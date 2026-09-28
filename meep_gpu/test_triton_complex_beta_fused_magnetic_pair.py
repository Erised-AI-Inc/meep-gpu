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
import json
import pathlib
import sys
from typing import List

import numpy
import pytest

from meep_gpu.expansion_refusal import declaring_run_policy
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid
from meep_gpu.pml import PML
from meep_gpu.test_triton_kernels import PACKAGE_DIR

MODULE_NAME = "meep_gpu.triton_kernels.complex_beta_fused_magnetic_pair"
API_ROOT = pathlib.Path(__file__).resolve().parents[1]
DRIVER_PATH = API_ROOT / "meep_gpu" / "driver.py"

#: TestSpecialKz.test_special_kz's own beta (census ``configuration.beta``).
BETA_CORPUS = -0.39073112848927377

#: The statements the beta term adds, and the ONLY thing that may separate this
#: kernel from ``complex_fused_curl_constitutive_B`` — K2's own delta over
#: ``bloch_pml_curl_step``, derived below from the shipped sources as well.
BETA_INSERT = (
    "if HAS_BETA:",
    "t_re, t_im = _mul_imag_coefficient_left(bp_re, bp_im, b_re, b_im, EXPANSION)",
    "curl0_re = curl0_re - t_re",
    "curl0_im = curl0_im - t_im",
    "t_re, t_im = _mul_imag_coefficient_left(bm_re, bm_im, a_re, a_im, EXPANSION)",
    "curl1_re = curl1_re - t_re",
    "curl1_im = curl1_im - t_im",
)

#: The extended probe artifact this suite reads — the 2026-09-01 keep cut whose
#: base-four table is character-identical to the standing 2026-08-16 one.
PROBE_PATH = (API_ROOT / "parity" / "meep_gpu" / "results" /
              "complex_expansion_beta_extension_2026-09-01" / "probe_keep" /
              "probe.json")


@pytest.fixture(scope="module")
def product():
    return importlib.import_module(MODULE_NAME)


@pytest.fixture(scope="module")
def probe():
    return json.loads(PROBE_PATH.read_text(encoding="utf-8"))


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


def _reasons(product, fields, pml, sources=(), probe=None):
    with declaring_run_policy("keep"):
        verdict = product.complex_beta_fused_magnetic_pair_coverage(
            fields, pml, sources, probe=probe)
    return [r for r in verdict.reasons if "array module" not in r]


# ---------------------------------------------------------------------------
# THE TRANSCRIPTION — the claim this whole product rests on
# ---------------------------------------------------------------------------

def test_removing_the_beta_insert_reproduces_the_shipped_complex_fused_pair(bodies):
    """``this kernel`` minus the insert IS ``complex_fused_curl_constitutive_B``,
    statement for statement, in order."""
    without_beta = [line for line in bodies["fused"] if line not in BETA_INSERT]
    assert without_beta == bodies["twin_fused"], (
        "the complex beta fused pair is not complex_fused_curl_constitutive_B "
        "plus the beta insert; first difference at index "
        + str(next((i for i, (a, b) in enumerate(
            zip(without_beta, bodies["twin_fused"])) if a != b), "length")))


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


def test_the_beta_insert_is_exactly_the_shipped_curls_delta(bodies):
    """The insert is ``beta_bloch_pml_curl_step``'s own delta over
    ``bloch_pml_curl_step`` — this file's constant cannot drift from K2."""
    delta = [line for line in bodies["beta_curl"]
             if line not in bodies["bloch_curl"]]
    assert delta == list(BETA_INSERT), delta


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


def test_the_probe_pattern_claim_is_the_extended_set(product):
    from meep_gpu.triton_kernels import special_kz  # noqa: PLC0415

    assert product.PRODUCT_PROBE_PATTERNS == special_kz.BETA_PROBE_PATTERNS
    assert special_kz.BETA_PROBE_PATTERN in product.PRODUCT_PROBE_PATTERNS


def test_the_kernel_body_calls_only_the_four_licensed_multiply_helpers(product):
    """The fourth is the beta insert's orientation; a fifth would need its own
    probe pattern before it could be licensed."""
    text = (PACKAGE_DIR / "complex_beta_fused_magnetic_pair.py").read_text(
        encoding="utf-8")
    tree = ast.parse(text)
    kernel = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef)
                  and node.name == "complex_beta_fused_curl_constitutive_B")
    called = {node.func.id for node in ast.walk(kernel)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id.startswith("_mul") or
              (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
               and node.func.id.startswith("_rotate"))}
    helpers = {name for name in called if name.startswith(("_mul", "_rotate"))}
    assert helpers == set(product.LICENSED_MULTIPLY_HELPERS), helpers


def test_the_beta_words_are_rounded_exactly_as_the_array_path_rounds_them(product):
    """The plan binds ``beta_curl_coefficients(..., magnetic=True,
    complex_storage=True)`` — the +1j pair with the signed-zero real word."""
    from meep_gpu.triton_kernels import special_kz  # noqa: PLC0415

    words = special_kz.beta_curl_coefficients(BETA_CORPUS, 0.035, True, True)
    expected = []
    for sign in (1.0, -1.0):
        value = numpy.complex64(
            (sign * 2.0 * numpy.pi * BETA_CORPUS * 0.035) * 1j)
        expected.append((float(numpy.float32(value.real)),
                         float(numpy.float32(value.imag))))
    assert [tuple(pair) for pair in words] == expected
    # the real words are SIGNED zeros, passed through
    assert all(pair[0] == 0.0 for pair in words)


# ---------------------------------------------------------------------------
# Admission and refusals
# ---------------------------------------------------------------------------

def test_the_corpus_family_is_admitted(product, probe):
    """2-D Cartesian, nonzero beta, in-plane kx, complex storage, active PML,
    unfolded, electric source only — test_special_kz's own configuration."""
    fields, pml = _build()
    assert _reasons(product, fields, pml, (_Electric(),), probe) == []


def test_a_walled_run_is_admitted(product, probe):
    """The wall clear is carried inline, so a walled complex beta run must stay
    covered. Metallic axes carry k = 0 (the phase-consistency clause)."""
    fields, pml = _build(boundaries={"x": "metallic", "y": "metallic",
                                     "z": "periodic"}, k_point=(0.0, 0.0, 0.0))
    assert _reasons(product, fields, pml, (), probe) == []


def test_an_undeclared_source_list_is_a_refusal_and_not_an_empty_set(product, probe):
    fields, pml = _build()
    reasons = _reasons(product, fields, pml, None, probe)
    assert any("was not declared" in reason for reason in reasons), reasons


def test_a_magnetic_source_without_an_index_is_refused_by_name(product, probe):
    fields, pml = _build()
    reasons = _reasons(product, fields, pml, (_Magnetic(),), probe)
    assert any("does not publish the index" in reason for reason in reasons), reasons


def test_a_zero_beta_run_is_refused_by_name(product, probe):
    fields, pml = _build(beta=0.0)
    reasons = _reasons(product, fields, pml, (), probe)
    assert any("beta is zero" in reason for reason in reasons), reasons


def test_real_storage_is_refused_this_is_the_complex_arm(product, probe):
    fields, pml = _build(complex_storage=False, k_point=(0.0, 0.0, 0.0))
    reasons = _reasons(product, fields, pml, (), probe)
    assert any("storage is real float32" in reason for reason in reasons), reasons


def test_a_fold_is_refused_by_this_products_own_clause(product, probe):
    """The folded complex beta cell belongs to the folded product; this weld's
    own seam clause names the two symmetry passes, over and above the halves'."""
    from meep_gpu.grid import Mirror  # noqa: PLC0415

    # A folded axis cannot carry a LOW-face layer (cell 0 is the mirror plane),
    # so this case passes its own thickness table: PML on x only.
    fields, pml = _build(symmetry=(Mirror("Y", 1),), k_point=(0.0, 0.0, 0.0),
                         cell_size=(2.0, 1.0, 0.0),
                         thickness=((2, 2), (0, 0), (0, 0)))
    reasons = _reasons(product, fields, pml, (), probe)
    assert any("mirror plane is active" in reason for reason in reasons), reasons
    assert any("this weld carries neither" in reason for reason in reasons)


def test_the_base_four_probe_is_refused_on_the_extended_clause(product):
    """The standing 2026-08-16 artifact classifies only the base four; the
    predicate must refuse it by the extended-pattern clause — the exact refusal
    the 2026-09-01 extension artifact discharges."""
    old = json.loads((API_ROOT / "parity" / "meep_gpu" / "results" /
                      "complex_expansion_convention_2026-08-16" / "results" /
                      "probe_keep" / "probe.json").read_text(encoding="utf-8"))
    fields, pml = _build()
    reasons = _reasons(product, fields, pml, (), old)
    assert any("EXTENDED pattern set" in reason for reason in reasons), reasons


def test_a_missing_expansion_probe_is_a_refusal_not_a_guessed_arm(product):
    fields, pml = _build()
    with declaring_run_policy("keep"):
        verdict = product.complex_beta_fused_magnetic_pair_coverage(
            fields, pml, (), probe=None)
    assert not verdict.covered
    assert any("expansion probe artifact" in reason for reason in verdict.reasons)


def test_the_predicate_reports_both_halves_reasons_with_their_side_named(product, probe):
    fields, pml = _build(beta=0.0)
    reasons = _reasons(product, fields, pml, (), probe)
    assert any(reason.startswith("complex beta curl half: ")
               for reason in reasons), reasons
    assert any(reason.startswith("complex beta constitutive half: ")
               for reason in reasons), reasons


def test_the_builder_refuses_every_configuration_the_predicate_refuses(product, probe):
    """None is the only refusal; the builder may not out-admit the predicate."""
    cases = (
        {"beta": 0.0},
        {"complex_storage": False, "k_point": (0.0, 0.0, 0.0)},
    )
    for keywords in cases:
        fields, pml = _build(**keywords)
        with declaring_run_policy("keep"):
            covered = product.complex_beta_fused_magnetic_pair_coverage(
                fields, pml, (), probe=probe).covered
            plan = product.plan_complex_beta_fused_magnetic_pair(
                fields, pml, (), probe=probe)
        assert not covered
        assert plan is None, keywords


def test_the_plan_refuses_a_cell_count_whose_word_index_overflows_int32(product):
    with pytest.raises(ValueError, match="int32"):
        product.ComplexBetaFusedMagneticPairPlan(
            (2048, 1024, 1024), 0.5, (0, 0, 0), (False, False, False),
            (0, 0, 0), (0.0,) * 6, ((0.0, 0.1), (-0.0, -0.1)), 1, 256,
            [], [], [], [], [], [], [])


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
