"""Laptop contracts for the REAL-beta fused ELECTRIC D/E pair on Triton.

Everything here runs on the NumPy merge-bar machine. The load-bearing ones are two:

* the TRANSCRIPTION leg — this product claims to be
  ``kernels.fused_curl_constitutive_D`` plus exactly the three lines
  ``special_kz.beta_pml_curl_step`` adds to ``kernels.pml_curl_step``, and that
  claim is checked here as two EXACT statement-list equalities against the shipped
  sources rather than by reading the docstring;
* the DEPOSIT FLAG — the single corpus row of this cell declares an ELECTRIC source,
  so :data:`~.beta_fused_electric_pair.CARRIES_DEPOSIT_REPAIR` at False takes the
  product from one seam-instance to zero. That is asserted DIRECTLY, by flipping the
  flag and re-asking the predicate, rather than restated in prose. Its MAGNETIC twin
  declares the flag False and this suite pins that asymmetry too: on the B seam the
  same row is electric-only.

The device bytes are the gate's
(``parity/meep_gpu/probe_triton_beta_fused_electric_pair.py``); nothing here
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

MODULE_NAME = "meep_gpu.triton_kernels.beta_fused_electric_pair"
MODULE_PATH = PACKAGE_DIR / "beta_fused_electric_pair.py"
API_ROOT = pathlib.Path(__file__).resolve().parents[1]
DRIVER_PATH = API_ROOT / "meep_gpu" / "driver.py"
PARITY_DIR = API_ROOT / "parity" / "meep_gpu"
GATE = PARITY_DIR / "probe_triton_beta_fused_electric_pair.py"

#: refl-angular-kz2d.py's own beta — the single row this cell has.
BETA_CORPUS = 0.3321611318837033

#: The three lines the beta term adds, and the ONLY thing that may separate this
#: kernel from ``kernels.fused_curl_constitutive_D``.
BETA_INSERT = (
    "if HAS_BETA:",
    "curl0 = curl0 - (beta_plus * b)",
    "curl1 = curl1 - (beta_minus * a)",
)


@pytest.fixture(scope="module")
def product():
    return importlib.import_module(MODULE_NAME)


class _Magnetic:
    field_type = "B"


class _Electric:
    """An electric source that publishes the index the injection writes.

    BOTH halves matter: the field type decides whether it is in the D seam at all,
    and the deposit index is what ``deposit_repair.save`` needs.
    """

    field_type = "D"

    def __init__(self, index=(1, 1, 0)) -> None:
        self._point_ix, self._point_iy, self._point_iz = index


class _ElectricWithoutIndex:
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
    importing ``kernels.py`` raises and this check would only ever bite on a device
    run.

    The text is EXACT — never ``ast.unparse``d. What is being compared includes the
    PARENTHESISATION, and unparsing re-derives minimal parentheses, which would
    silently equate ``dtdx * ((c_y - c) + (b - b_z))`` with a different float32
    grouping.
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
        "fused": _shipped_body(MODULE_PATH, "beta_fused_curl_constitutive_D"),
        "beta_curl": _shipped_body(PACKAGE_DIR / "special_kz.py",
                                   "beta_pml_curl_step"),
        "ordinary_fused": _shipped_body(PACKAGE_DIR / "kernels.py",
                                        "fused_curl_constitutive_D"),
        "ordinary_curl": _shipped_body(PACKAGE_DIR / "kernels.py",
                                       "pml_curl_step"),
        "magnetic_twin": _shipped_body(
            PACKAGE_DIR / "beta_fused_magnetic_pair.py",
            "beta_fused_curl_constitutive_B"),
    }


def _build(cell_size=(1.2, 1.0, 0.0), dimensions=2, boundaries=None,
           pml_thickness=2, complex_storage=False, beta=BETA_CORPUS,
           k_point=(0.0, 0.0, 0.0), seed=17, thickness=None, **grid_kwargs):
    """A real beta Grid/Fields/PML triple on NumPy, PML storage enabled.

    Default: the corpus family — 2-D Cartesian (which is the ONLY place MEEP allows
    beta, fields.cpp:546-547), real storage, an active PML, k = 0, no fold. Every
    refusal test perturbs exactly one clause off this. Deliberately IDENTICAL in
    shape to the magnetic twin's fixture, so a difference between the two suites is a
    difference between the two products.
    """
    grid = Grid(resolution=10.0, cell_size=cell_size, dimensions=dimensions,
                boundaries=boundaries, courant=0.35, k_point=k_point, beta=beta,
                **grid_kwargs)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.set_background_eps(2.25)
    fields.enable_pml_storage()
    rng = numpy.random.default_rng(seed)
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
                 "f_w_Ex", "f_w_Ey", "f_w_Ez"):
        getattr(fields, name)[...] = rng.uniform(
            -0.4, 0.4, size=grid.shape).astype(numpy.float32)
    if thickness is None:
        thickness = tuple(
            (pml_thickness, pml_thickness) if grid.shape[axis] >= 6 else (0, 0)
            for axis in range(3))
    return fields, PML(grid=grid, thickness=thickness)


def _reasons(product, fields, pml, sources=()):
    verdict = product.beta_fused_electric_pair_coverage(fields, pml, sources)
    return [r for r in verdict.reasons if "array module" not in r]


# ---------------------------------------------------------------------------
# THE TRANSCRIPTION — the claim this whole product rests on
# ---------------------------------------------------------------------------

def test_removing_the_beta_insert_reproduces_the_shipped_fused_pair(bodies):
    """``this kernel`` minus the three beta lines IS ``fused_curl_constitutive_D``.

    An EXACT statement-list equality, in order. This is the diff a reviewer would do
    by hand, made mechanical: if any other line moved, this fails and names it.
    """
    without_beta = [line for line in bodies["fused"] if line not in BETA_INSERT]
    assert without_beta == bodies["ordinary_fused"], (
        "the beta fused electric pair is not fused_curl_constitutive_D plus the "
        "beta insert; first difference at index "
        + str(next((i for i, (a, b) in enumerate(
            zip(without_beta, bodies["ordinary_fused"])) if a != b), "length")))


def test_removing_the_weld_reproduces_the_shipped_beta_curl(bodies):
    """``this kernel`` minus the wall clear and the constitutive half IS
    ``special_kz.beta_pml_curl_step``.

    The removed set is DERIVED — it is whatever ``fused_curl_constitutive_D`` has
    that ``pml_curl_step`` does not — so a change to the weld moves this test with it
    rather than requiring the list to be maintained by hand.
    """
    weld = [line for line in bodies["ordinary_fused"]
            if line not in bodies["ordinary_curl"]]
    assert weld, "the weld is empty; this test compares nothing"
    without_weld = [line for line in bodies["fused"] if line not in weld]
    assert without_weld == bodies["beta_curl"], (
        "the beta fused electric pair minus the weld is not beta_pml_curl_step; "
        "first difference at index "
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


def test_the_electric_weld_is_not_the_magnetic_one(bodies):
    """The two twins DIFFER, and where they differ is the D seam.

    A kernel copied from the magnetic twin would pass every transcription test above
    against the wrong pair of shipped bodies, so the difference is asserted directly:
    this one carries the six-row wall clear and the inverse-epsilon multiply, and the
    twin carries neither.
    """
    assert bodies["fused"] != bodies["magnetic_twin"]
    ours = set(bodies["fused"])
    theirs = set(bodies["magnetic_twin"])
    # The E side's inverse-permittivity multiply, per component.
    for component in range(3):
        line = (f"src{component} = v{component} * tl.load("
                f"ie{component} + idx, mask=live, other=0.0)")
        assert line in ours, line
        assert line not in theirs, line
    # zero_metal_D clears TWO tangential components per axis; zero_metal_B one.
    assert "v1 = tl.where(at_x, 0.0, v1)" in ours
    assert "v2 = tl.where(at_x, 0.0, v2)" in ours
    assert "v2 = tl.where(at_x, 0.0, v2)" not in theirs


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


def test_the_module_answers_coverage_but_the_kernel_fails_clearly_without_triton(
        product):
    if product.beta_fused_curl_constitutive_D is not None:
        pytest.skip("triton is installed here; the absence arm is exercised on the "
                    "merge-bar box without it")
    fields, pml = _build()
    assert product.beta_fused_electric_pair_coverage(fields, pml, ()).reasons
    with pytest.raises(ImportError, match="triton"):
        product.beta_fused_curl_constitutive_D_kernel()


def test_backward_matches_the_sub_step_table(product):
    from meep_gpu.triton_kernels import launch  # noqa: PLC0415

    assert product.BACKWARD == int(
        launch.SUB_STEPS[product.CURL_SUB_STEP]["backward"])
    assert product.BACKWARD == 1
    assert product.CURL_SUB_STEP == "step_D"
    assert product.CONSTITUTIVE_SIDE == "E"


def test_replaces_is_the_drivers_own_call_order(product):
    text = DRIVER_PATH.read_text(encoding="utf-8")
    positions = []
    for name in product.REPLACES:
        index = text.find(f"{name}(self.fields")
        assert index > 0, f"driver.py never calls {name}(self.fields...)"
        positions.append(index)
    assert positions == sorted(positions), product.REPLACES


def test_the_single_launch_site_passes_the_packages_shared_guard_constant():
    text = MODULE_PATH.read_text(encoding="utf-8")
    assert "enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard)" \
        in text
    assert text.count("kernel[self._grid](") == 1


# ---------------------------------------------------------------------------
# Admission and refusals
# ---------------------------------------------------------------------------

def test_the_corpus_family_is_admitted_WITH_its_electric_deposit(product):
    """2-D Cartesian, nonzero beta, active PML, unfolded, ELECTRIC source.

    The corpus row's own shape. The electric source is what the whole product turns
    on, so it is in the ADMITTING case rather than only in a refusal.
    """
    fields, pml = _build()
    assert _reasons(product, fields, pml, (_Electric(),)) == []


def test_a_walled_run_is_admitted(product):
    """The wall clear is carried inline, so a walled beta run must stay covered.

    z stays PERIODIC: on a 2-D cell z is the INVARIANT axis and ``Grid`` refuses a
    PEC there by name (grid.py:842-877).
    """
    fields, pml = _build(boundaries={"x": "metallic", "y": "metallic",
                                     "z": "periodic"})
    assert _reasons(product, fields, pml, (_Electric(),)) == []


def test_a_magnetic_source_never_reaches_this_seam(product):
    fields, pml = _build()
    assert _reasons(product, fields, pml, (_Magnetic(),)) == []


@pytest.mark.parametrize("sources,needle", [
    (None, "was not declared"),
    ((_ElectricWithoutIndex(),), "does not publish the index"),
])
def test_the_source_seam_refuses_by_name(product, sources, needle):
    fields, pml = _build()
    reasons = _reasons(product, fields, pml, sources)
    assert any(needle in reason for reason in reasons), reasons


def test_the_flag_at_False_would_cost_the_whole_cell(product, monkeypatch):
    """The measurement the module's docstring makes, executed rather than restated.

    The single corpus row declares an electric source. With
    ``CARRIES_DEPOSIT_REPAIR`` False the seam clause refuses it outright, so the cell
    is worth ZERO — which is why the flag is declared in the same change as the
    wiring that brackets the launch.
    """
    fields, pml = _build()
    assert product.CARRIES_DEPOSIT_REPAIR is True
    monkeypatch.setattr(product, "CARRIES_DEPOSIT_REPAIR", False)
    reasons = _reasons(product, fields, pml, (_Electric(),))
    assert any("is electric" in reason for reason in reasons), reasons


def test_the_magnetic_twin_keeps_the_flag_False_for_the_mirror_image_reason(product):
    """The asymmetry is the point, so it is pinned rather than described.

    On the B seam the same corpus row is electric-only, so no repair is consulted at
    all and the twin's clause costs it nothing. A twin that silently gained the flag
    would be claiming a bracket its plan does not build.
    """
    twin = importlib.import_module(
        "meep_gpu.triton_kernels.beta_fused_magnetic_pair")
    assert twin.CARRIES_DEPOSIT_REPAIR is False
    assert product.CARRIES_DEPOSIT_REPAIR is True


def test_the_flag_is_passed_to_the_clause_that_reads_it_and_not_merely_declared():
    """A flag nothing forwards is a comment. Read off the source."""
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and getattr(node.func, "attr", "") == "seam_source_reasons"]
    assert len(calls) == 1, calls
    keywords = {keyword.arg: ast.unparse(keyword.value)
                for keyword in calls[0].keywords}
    assert keywords["carries_repair"] == "CARRIES_DEPOSIT_REPAIR", keywords
    # THE SEAM IS 'D'. A D-seam product mis-assigned 'B' refuses the magnetic source
    # it should admit and ADMITS the electric one it must repair.
    assert ast.unparse(calls[0].args[2]) == "'D'", ast.unparse(calls[0].args[2])


def test_a_zero_beta_run_is_refused_by_name(product):
    """The inverted clause: a beta = 0 run is the ordinary fused pair's."""
    fields, pml = _build(beta=0.0)
    reasons = _reasons(product, fields, pml, (_Electric(),))
    assert any("grid.beta is zero" in reason for reason in reasons), reasons


def test_complex_storage_is_refused(product):
    fields, pml = _build(complex_storage=True)
    reasons = _reasons(product, fields, pml, (_Electric(),))
    assert any("force_complex_fields" in reason for reason in reasons), reasons


def test_an_inactive_absorber_is_refused(product):
    fields, pml = _build(pml_thickness=0)
    reasons = _reasons(product, fields, pml, (_Electric(),))
    assert any("PML" in reason for reason in reasons), reasons


def test_a_fold_is_refused_by_name(product):
    from meep_gpu.grid import Mirror  # noqa: PLC0415

    fields, pml = _build(symmetry=(Mirror("Y", 1),),
                         thickness=((2, 2), (0, 2), (0, 0)))
    reasons = _reasons(product, fields, pml, (_Electric(),))
    assert any("mirror plane" in reason for reason in reasons), reasons


def test_bloch_is_refused(product):
    fields, pml = _build(k_point=(0.2, 0.0, 0.0))
    reasons = _reasons(product, fields, pml, (_Electric(),))
    assert any("k_point" in reason for reason in reasons), reasons


# ---------------------------------------------------------------------------
# Disjointness from the ordinary fused pair, in BOTH directions
# ---------------------------------------------------------------------------

def test_this_arm_and_the_ordinary_fused_pair_never_co_admit(product):
    from meep_gpu.triton_kernels import coverage  # noqa: PLC0415

    for beta in (BETA_CORPUS, 0.0):
        fields, pml = _build(beta=beta)
        sources = (_Electric(),)
        mine = [r for r in product.beta_fused_electric_pair_coverage(
            fields, pml, sources).reasons if "array module" not in r]
        theirs = [r for r in coverage.fused_pair_coverage(
            fields, pml, "D", sources).reasons if "array module" not in r]
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
    ({"beta": 0.0}, (_Electric(),)),
    ({"complex_storage": True}, (_Electric(),)),
    ({"pml_thickness": 0}, (_Electric(),)),
    ({"boundaries": {"x": "metallic", "y": "metallic", "z": "periodic"}},
     (_Electric(),)),
    ({"k_point": (0.2, 0.0, 0.0)}, (_Electric(),)),
])
def test_the_weld_is_never_wider_than_the_halves(product, kwargs, sources):
    from meep_gpu.triton_kernels import special_kz  # noqa: PLC0415

    fields, pml = _build(**kwargs)
    weld = product.beta_fused_electric_pair_coverage(fields, pml, sources)
    curl = special_kz.beta_pml_curl_coverage(fields, pml, "step_D")
    electric = special_kz.beta_run_constitutive_coverage(fields, pml, "E")
    assert (not weld.covered) or (curl.covered and electric.covered), (
        weld.reasons, curl.reasons, electric.reasons)


def test_the_plan_refuses_rather_than_raises_on_a_numpy_host(product):
    fields, pml = _build()
    assert product.plan_beta_fused_electric_pair(fields, pml, (_Electric(),)) is None
    assert product.plan_beta_fused_electric_pair(fields, pml, None) is None


def test_the_beta_seam_flag_is_INERT_on_real_storage_and_live_on_complex():
    """MEASURED, and it corrects the thing a reader would assume.

    ``beta_curl_coefficients`` takes ``magnetic=``, and on COMPLEX storage that flag
    decides a multiply by ``+1j`` or ``-1j`` (special_kz.py:277-281). On REAL storage
    it decides NOTHING: the ``+-i`` is MEEP's implicit-i trick and lives only in the
    complex path, so both seams get the same two words. That is exactly what this
    product's kernel docstring says about the beta term's sign, and it is asserted
    HERE rather than assumed, in both directions.

    THE CONSEQUENCE, stated so nobody has to rediscover it: on this REAL-storage
    family a builder that copied its twin's ``magnetic=True`` would be harmless. The
    call is still written ``magnetic=(CURL_SUB_STEP == 'step_B')`` because that is
    what the array path computes per call site — but the gate may NOT claim a
    discrimination it does not have, so no mutation swaps the flag and none pretends
    to catch it. What the gate DOES arm is the two beta WORDS exchanged
    (``h_beta_words_swapped``), which differ in sign on either storage.
    """
    from meep_gpu.triton_kernels import special_kz  # noqa: PLC0415

    fields, _pml = _build()
    grid = fields.grid
    real_magnetic = special_kz.beta_curl_coefficients(
        grid.beta, grid.dt, magnetic=True, complex_storage=False)
    real_electric = special_kz.beta_curl_coefficients(
        grid.beta, grid.dt, magnetic=False, complex_storage=False)
    assert real_magnetic == real_electric, (real_magnetic, real_electric)
    # ...and the two words within one seam DO differ, in sign: that is the pair the
    # host mutation exchanges.
    assert real_electric[0] == -real_electric[1] != 0.0, real_electric

    complex_magnetic = special_kz.beta_curl_coefficients(
        grid.beta, grid.dt, magnetic=True, complex_storage=True)
    complex_electric = special_kz.beta_curl_coefficients(
        grid.beta, grid.dt, magnetic=False, complex_storage=True)
    assert complex_magnetic != complex_electric, (complex_magnetic,
                                                  complex_electric)


# ---------------------------------------------------------------------------
# The deferral, the wiring records and the gate
# ---------------------------------------------------------------------------

def test_the_composer_routes_this_product_and_dispatch_admits_it():
    """ROUTED 2026-09-02 by the installer wave, and RELEASED at dispatch — both halves.

    This test replaces the refusal it used to make. Until 2026-09-02 it asserted
    that ``launch.py`` and ``fastpath.py`` named this module NOWHERE, which was the
    seam that kept a certified-but-unrouted product deferred. The wave routed it:
    ``launch.CERTIFIED_FUSED_PRODUCTS`` holds its row and
    ``_install_certified_fused_products`` builds it through the shared
    ``_install_fused_pair``, taking both slots only when ``_pair_may_absorb`` finds
    the arm table has already given them to the arms this kernel implements.

    The 2026-09-13 phase-B batch RELEASED it, and the second half is that sentence
    made checkable from the PRODUCT's side rather than from the composer's: the
    label this product writes is now in ``fastpath.RELEASED_FUSED_ARMS``, so clause
    (8) admits the whole plan; and it carries an ``ARM_CERTIFICATION`` row (its gate
    has a tracked ledger entry) while having LEFT ``PENDING_DEVICE_GATE_ARMS`` (the
    pending rung no longer refuses it).
    """
    import pathlib as _pathlib  # noqa: PLC0415

    from meep_gpu import fastpath as _fastpath  # noqa: PLC0415
    from meep_gpu.triton_kernels import launch as _launch  # noqa: PLC0415

    source = _pathlib.Path(_launch.__file__).read_text(encoding="utf-8")
    assert "plan_beta_fused_electric_pair" in source
    row = _launch.CERTIFIED_FUSED_PRODUCTS["beta_fused_electric_pair"]
    assert row["module"] == "beta_fused_electric_pair"
    assert row["builder"] == "plan_beta_fused_electric_pair"
    label = row["label"]
    assert label == 'fused pair D (real beta)'
    assert _fastpath.arm_is_fused(label)
    assert label in _fastpath.RELEASED_FUSED_ARMS
    assert label in _fastpath.ARM_CERTIFICATION
    assert label not in _fastpath.PENDING_DEVICE_GATE_ARMS
    assert label in _fastpath.FUSED_ARM_CONSTITUENTS
def test_the_module_is_named_in_the_deposit_repair_wiring_records():
    text = (PACKAGE_DIR.parent
            / "test_fused_pair_deposit_wiring.py").read_text(encoding="utf-8")
    assert f"triton_kernels/{MODULE_PATH.name}" in text


def test_the_module_claims_identity_ONLY_through_the_run_that_measured_it(product):
    import json  # noqa: PLC0415

    fingerprints = json.loads(
        (PACKAGE_DIR / "fingerprints.json").read_text(encoding="utf-8"))
    if "NOT RELEASED" in product.DEVICE_STATUS:
        assert MODULE_PATH.stem not in json.dumps(fingerprints), \
            "an unreleased product may not carry a checked-in fingerprint"
    else:  # pragma: no cover - reached once the gate has released
        assert "gate.json" in product.DEVICE_STATUS


def test_the_gate_exists_and_names_this_product():
    assert GATE.exists(), GATE
    text = GATE.read_text(encoding="utf-8")
    assert "beta_fused_electric_pair" in text
    assert "beta_fused_curl_constitutive_D" in text


def test_the_weld_driver_the_board_and_the_battery_all_know_this_product():
    driver = (PARITY_DIR / "drive_triton_weld_gates.py").read_text(encoding="utf-8")
    board = (PARITY_DIR
             / "build_triton_fusion_matrix.py").read_text(encoding="utf-8")
    battery = (PARITY_DIR / "triton_predicate_battery.py").read_text(encoding="utf-8")
    assert "probe_triton_beta_fused_electric_pair.py" in driver
    assert '"beta_fused_electric_pair"' in board
    assert "beta_fused_electric_pair_coverage" in battery


def test_the_module_is_declared_ROUTED_rather_than_NOT_AN_ARM():
    """It LEFT ``test_triton_planner_composition.NOT_AN_ARM`` on 2026-09-02.

    Both directions, because both failures are wrong in opposite ways: a routed
    module still named in that tuple would tell a reader the planner cannot reach
    it, and a module missing from ``SUPPORT_MODULES`` would break the lazy-import
    seam this package rests on.
    """
    from meep_gpu.test_triton_planner_composition import NOT_AN_ARM  # noqa: PLC0415
    from meep_gpu.triton_kernels import launch as _launch  # noqa: PLC0415

    assert "beta_fused_electric_pair" not in NOT_AN_ARM
    assert "beta_fused_electric_pair" in _launch.SUPPORT_MODULES
def load_gate():
    import importlib.util  # noqa: PLC0415

    spec = importlib.util.spec_from_file_location("_bfep_gate", GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_gates_no_device_legs_all_pass_here():
    gate = load_gate()
    for name in ("transcription_leg", "equivalence_leg", "corpus_admission_leg",
                 "mutation_arming_leg"):
        leg = getattr(gate, name)()
        assert leg.get("passed") is True, (name, leg)


def test_every_mutation_the_gate_declares_is_ARMED_against_the_shipped_source():
    gate = load_gate()
    source = gate.shipped_source()
    for name, _kind, _reason, rewrite in gate.mutation_table():
        mutated, count = rewrite(source)
        assert count >= 1, name
        assert mutated != source, name


def test_every_mutation_is_scored_on_a_case_that_enters_the_branch_it_rewrites():
    gate = load_gate()
    for name, _kind, _reason, _rewrite in gate.mutation_table():
        gate.mutation_case_for(name)   # raises if the pairing is dead


def test_the_gate_binds_the_seam_passes_the_driver_actually_calls(product):
    gate = load_gate()
    assert tuple(gate.SEAM_PASSES) == tuple(product.REPLACES)


def test_the_gate_carries_a_carry_family_and_a_null_control():
    """Every carry case needs a bracket-removed control that MUST diverge."""
    gate = load_gate()
    assert gate.CARRY_CASES, "the deposit carry is this cell's whole value"
    text = GATE.read_text(encoding="utf-8")
    assert "bracket" in text and "null" in text.lower()


def test_the_gate_carries_the_certified_kernel_identity_arm():
    """``HAS_BETA = 0`` must reproduce ``kernels.fused_curl_constitutive_D``.

    That is the arm the ``HAS_BETA`` constexpr exists for; a gate without it would
    carry the constexpr as decoration.
    """
    text = GATE.read_text(encoding="utf-8")
    assert "has_beta" in text
    assert "fused_curl_constitutive_D" in text


def test_the_gate_pins_the_files_this_product_actually_rests_on():
    gate = load_gate()
    names = set(gate.source_hashes())
    for required in ("meep_gpu/triton_kernels/beta_fused_electric_pair.py",
                     "meep_gpu/triton_kernels/special_kz.py",
                     "meep_gpu/triton_kernels/kernels.py",
                     "meep_gpu/deposit_repair.py",
                     "meep_gpu/stepping.py",
                     "meep_gpu/driver.py",
                     "meep_gpu/test_triton_beta_fused_electric_pair.py"):
        assert required in names, required


def test_the_gate_declares_the_cell_the_board_scored():
    gate = load_gate()
    assert gate.CELL == ("D->E", "real beta PML", "real beta run")
