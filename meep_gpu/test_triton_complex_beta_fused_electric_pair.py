"""Laptop contracts for the COMPLEX-beta fused electric D/E pair on Triton.

The transcription claim: this product is
``complex_fused_electric_pair.complex_fused_curl_constitutive_D`` plus exactly
the statements ``special_kz.beta_bloch_pml_curl_step`` adds to
``complex_fields.bloch_pml_curl_step`` — checked as EXACT statement-list
equalities against the shipped sources. The device bytes are the gate's
(``parity/meep_gpu/probe_triton_complex_beta_fused_electric_pair.py``); nothing
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

MODULE_NAME = "meep_gpu.triton_kernels.complex_beta_fused_electric_pair"
API_ROOT = pathlib.Path(__file__).resolve().parents[1]
DRIVER_PATH = API_ROOT / "meep_gpu" / "driver.py"

#: TestSpecialKz.test_special_kz's own beta (census ``configuration.beta``).
BETA_CORPUS = -0.39073112848927377

#: The statements the beta term adds — K2's own delta over
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
    out: List[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if line.strip():
            out.append(line.strip())
    return out


def _shipped_body(path: pathlib.Path, name: str) -> List[str]:
    """One shipped kernel's BODY statements, docstring and signature removed —
    EXACT text, never ``ast.unparse``d (the comparison includes the
    parenthesisation)."""
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
            PACKAGE_DIR / "complex_beta_fused_electric_pair.py",
            "complex_beta_fused_curl_constitutive_D"),
        "beta_curl": _shipped_body(PACKAGE_DIR / "special_kz.py",
                                   "beta_bloch_pml_curl_step"),
        "twin_fused": _shipped_body(PACKAGE_DIR / "complex_fused_electric_pair.py",
                                    "complex_fused_curl_constitutive_D"),
        "bloch_curl": _shipped_body(PACKAGE_DIR / "complex_fields.py",
                                    "bloch_pml_curl_step"),
    }


def _build(cell_size=(2.0, 0.4, 0.0), boundaries=None, complex_storage=True,
           beta=BETA_CORPUS, k_point=(0.9205, 0.0, 0.0), seed=17,
           thickness=None, pml_thickness=2, symmetry=(), **grid_kwargs):
    """A complex beta Grid/Fields/PML triple on NumPy, PML storage enabled."""
    grid = Grid(resolution=10.0, cell_size=cell_size, dimensions=2,
                boundaries=boundaries, courant=0.35, k_point=k_point, beta=beta,
                symmetry=symmetry, **grid_kwargs)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.set_background_eps(2.25)
    fields.enable_pml_storage()
    rng = numpy.random.default_rng(seed)
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Hx", "Hy", "Hz"):
        real = rng.uniform(-0.4, 0.4, size=grid.shape).astype(numpy.float32)
        getattr(fields, name)[...] = real
    if thickness is None:
        thickness = tuple(
            (pml_thickness, pml_thickness) if grid.shape[axis] >= 6 else (0, 0)
            for axis in range(3))
    return fields, PML(grid=grid, thickness=thickness)


def _reasons(product, fields, pml, sources=(), probe=None):
    with declaring_run_policy("keep"):
        verdict = product.complex_beta_fused_electric_pair_coverage(
            fields, pml, sources, probe=probe)
    return [r for r in verdict.reasons if "array module" not in r]


# ---------------------------------------------------------------------------
# THE TRANSCRIPTION — the claim this whole product rests on
# ---------------------------------------------------------------------------

def test_removing_the_beta_insert_reproduces_the_shipped_complex_fused_pair(bodies):
    without_beta = [line for line in bodies["fused"] if line not in BETA_INSERT]
    assert without_beta == bodies["twin_fused"], (
        "the complex beta fused electric pair is not "
        "complex_fused_curl_constitutive_D plus the beta insert; first "
        "difference at index "
        + str(next((i for i, (a, b) in enumerate(
            zip(without_beta, bodies["twin_fused"])) if a != b), "length")))


def test_removing_the_weld_reproduces_the_shipped_beta_bloch_curl(bodies):
    """The removed set is DERIVED — whatever the beta-less twin has that
    ``bloch_pml_curl_step`` does not (the wall clear, the inv_eps scaling and
    the whole constitutive half)."""
    weld = [line for line in bodies["twin_fused"]
            if line not in bodies["bloch_curl"]]
    assert weld, "the weld is empty; this test compares nothing"
    without_weld = [line for line in bodies["fused"] if line not in weld]
    assert without_weld == bodies["beta_curl"], (
        "the complex beta fused electric pair minus the weld is not "
        "beta_bloch_pml_curl_step; first difference at index "
        + str(next((i for i, (a, b) in enumerate(
            zip(without_weld, bodies["beta_curl"])) if a != b), "length")))


def test_the_beta_insert_is_exactly_the_shipped_curls_delta(bodies):
    delta = [line for line in bodies["beta_curl"]
             if line not in bodies["bloch_curl"]]
    assert delta == list(BETA_INSERT), delta


def test_every_statement_comes_from_a_shipped_body(bodies):
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


def test_backward_and_scale_are_the_D_and_E_sides(product):
    from meep_gpu.triton_kernels import launch  # noqa: PLC0415

    assert product.BACKWARD == int(
        launch.SUB_STEPS[product.CURL_SUB_STEP]["backward"])
    assert product.BACKWARD == 1
    assert product.SCALE == 1
    twin = importlib.import_module(
        "meep_gpu.triton_kernels.complex_fused_electric_pair")
    assert product.SCALE == twin.SCALE


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


def test_the_wall_map_is_the_D_sides_and_derived_from_iyee(product):
    """An x wall clears the TANGENTIAL Dy/Dz — the exact complement of the B
    side's map, re-derived from the shipped IYEE_SHIFTS."""
    from meep_gpu.fields import IYEE_SHIFTS  # noqa: PLC0415

    derived = tuple(
        tuple(index for index, name in enumerate(product.CURL_TARGETS)
              if IYEE_SHIFTS[name][axis] == 0)
        for axis in range(3))
    assert derived == product.WALL_CLEARED_COMPONENTS
    assert product.WALL_CLEARED_COMPONENTS == ((1, 2), (0, 2), (0, 1))


def test_the_beta_words_are_the_electric_sides_minus_i_pair(product):
    """The plan binds ``beta_curl_coefficients(..., magnetic=False,
    complex_storage=True)`` — the -1j pair with the signed-zero real word."""
    from meep_gpu.triton_kernels import special_kz  # noqa: PLC0415

    words = special_kz.beta_curl_coefficients(BETA_CORPUS, 0.035, False, True)
    expected = []
    for sign in (1.0, -1.0):
        value = numpy.complex64(
            (sign * 2.0 * numpy.pi * BETA_CORPUS * 0.035) * -1j)
        expected.append((float(numpy.float32(value.real)),
                         float(numpy.float32(value.imag))))
    assert [tuple(pair) for pair in words] == expected
    assert all(pair[0] == 0.0 for pair in words)


# ---------------------------------------------------------------------------
# Admission and refusals
# ---------------------------------------------------------------------------

def test_an_electric_deposit_is_carried_and_a_magnetic_source_is_irrelevant(
        product, probe):
    """The cell's one corpus row injects electrically in THIS seam; the shipped
    repair is what carries it. A magnetic source belongs to the other seam."""
    from meep_gpu import deposit_repair  # noqa: PLC0415
    from meep_gpu.sources import GaussianEnvelope, VolumeSource  # noqa: PLC0415

    fields, pml = _build()
    source = VolumeSource(grid=fields.grid, component="Ey",
                          center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                          envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2),
                          amplitude=1.0)
    assert deposit_repair._deposit_index(source) is not None
    assert _reasons(product, fields, pml, (source, _Magnetic()), probe) == []


def test_an_electric_source_without_an_index_is_refused_by_name(product, probe):
    fields, pml = _build()
    reasons = _reasons(product, fields, pml, (_Electric(),), probe)
    assert any("does not publish the index" in reason for reason in reasons), reasons


def test_an_undeclared_source_list_is_a_refusal_and_not_an_empty_set(product, probe):
    fields, pml = _build()
    reasons = _reasons(product, fields, pml, None, probe)
    assert any("was not declared" in reason for reason in reasons), reasons


def test_a_zero_beta_run_is_refused_by_name(product, probe):
    fields, pml = _build(beta=0.0)
    reasons = _reasons(product, fields, pml, (), probe)
    assert any("beta is zero" in reason for reason in reasons), reasons


def test_real_storage_is_refused_this_is_the_complex_arm(product, probe):
    fields, pml = _build(complex_storage=False, k_point=(0.0, 0.0, 0.0))
    reasons = _reasons(product, fields, pml, (), probe)
    assert any("storage is real float32" in reason for reason in reasons), reasons


def test_a_fold_is_refused_by_this_products_own_clause(product, probe):
    from meep_gpu.grid import Mirror  # noqa: PLC0415

    fields, pml = _build(symmetry=(Mirror("Y", 1),), k_point=(0.0, 0.0, 0.0),
                         cell_size=(2.0, 1.0, 0.0),
                         thickness=((2, 2), (0, 0), (0, 0)))
    reasons = _reasons(product, fields, pml, (), probe)
    assert any("mirror plane is active" in reason for reason in reasons), reasons


def test_a_registered_susceptibility_is_refused_by_this_products_own_clause(
        product, probe):
    fields, pml = _build()
    fields.polarizations = ("stub",)
    reasons = _reasons(product, fields, pml, (), probe)
    assert any("(D - sum P)" in reason for reason in reasons), reasons


def test_a_fields_that_cannot_hand_over_inv_eps_is_refused_not_crashed(
        product, probe):
    fields, pml = _build()

    class _NoInvEps:
        def __getattr__(self, name):
            if name == "inverse_epsilon_for":
                raise AttributeError(name)
            return getattr(fields, name)

        inverse_epsilon_for = None

    shim = _NoInvEps()
    reasons = _reasons(product, shim, pml, (), probe)
    assert any("inverse_epsilon_for" in reason for reason in reasons), reasons


def test_the_base_four_probe_is_refused_on_the_extended_clause(product):
    old = json.loads((API_ROOT / "parity" / "meep_gpu" / "results" /
                      "complex_expansion_convention_2026-08-16" / "results" /
                      "probe_keep" / "probe.json").read_text(encoding="utf-8"))
    fields, pml = _build()
    reasons = _reasons(product, fields, pml, (), old)
    assert any("EXTENDED pattern set" in reason for reason in reasons), reasons


def test_the_builder_refuses_every_configuration_the_predicate_refuses(product, probe):
    cases = (
        {"beta": 0.0},
        {"complex_storage": False, "k_point": (0.0, 0.0, 0.0)},
    )
    for keywords in cases:
        fields, pml = _build(**keywords)
        with declaring_run_policy("keep"):
            covered = product.complex_beta_fused_electric_pair_coverage(
                fields, pml, (), probe=probe).covered
            plan = product.plan_complex_beta_fused_electric_pair(
                fields, pml, (), probe=probe)
        assert not covered
        assert plan is None, keywords


def test_the_plan_refuses_a_placeholder_inv_eps_binding(product):
    with pytest.raises(ValueError, match="inv_eps"):
        product.ComplexBetaFusedElectricPairPlan(
            (4, 4, 1), 0.5, (0, 0, 0), (False, False, False),
            (0, 0, 0), (0.0,) * 6, ((0.0, 0.1), (-0.0, -0.1)), 1, 256,
            [], [], [], [], [], [], None, [])


def test_the_plan_refuses_a_cell_count_whose_word_index_overflows_int32(product):
    import numpy as np  # noqa: PLC0415

    stub = np.zeros(1, np.float32)
    with pytest.raises(ValueError, match="int32"):
        product.ComplexBetaFusedElectricPairPlan(
            (2048, 1024, 1024), 0.5, (0, 0, 0), (False, False, False),
            (0, 0, 0), (0.0,) * 6, ((0.0, 0.1), (-0.0, -0.1)), 1, 256,
            [], [], [], [], [], [], (stub, stub, stub), [])


def test_the_from_arrays_route_carries_the_identity_arm(product):
    import inspect  # noqa: PLC0415

    signature = inspect.signature(
        product.plan_complex_beta_fused_electric_pair_from_arrays)
    assert signature.parameters["has_beta"].default == 1


def test_the_carry_flag_and_the_seam_clause_move_together(product):
    text = (PACKAGE_DIR / "complex_beta_fused_electric_pair.py").read_text(
        encoding="utf-8")
    assert text.count("CARRIES_DEPOSIT_REPAIR = ") == 1
    assert "carries_repair=CARRIES_DEPOSIT_REPAIR" in text
    assert product.CARRIES_DEPOSIT_REPAIR is True
