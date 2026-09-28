"""Laptop contracts for the BFAST fused ELECTRIC D/E pair on Triton.

Everything here runs on the NumPy merge-bar machine. The load-bearing ones are three:

* the TRANSCRIPTION leg — this product claims to be
  ``kernels.fused_curl_constitutive_D`` plus exactly the ``if HAS_BFAST:`` block
  ``bfast_curl.bfast_pml_curl_step`` adds to ``kernels.pml_curl_step``, and that
  claim is checked here as two EXACT statement-list equalities against the shipped
  sources rather than by reading the docstring;
* the DEPOSIT FLAG — the single corpus row of this cell declares an ELECTRIC source,
  so :data:`~.bfast_fused_electric_pair.CARRIES_DEPOSIT_REPAIR` at False takes the
  product from one seam-instance to zero. Asserted DIRECTLY, by flipping the flag
  and re-asking the predicate;
* the SEAM FLAG ON THE COEFFICIENTS — unlike the beta family, whose ``magnetic=``
  argument is inert on real storage, BFAST reads a different term table AND negates
  both coefficients on the D side. A builder that copied its twin's ``magnetic=True``
  would hand this product the wrong six words, and the two answers are compared
  directly here.

The device bytes are the gate's
(``parity/meep_gpu/probe_triton_bfast_fused_electric_pair.py``); nothing here
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

MODULE_NAME = "meep_gpu.triton_kernels.bfast_fused_electric_pair"
MODULE_PATH = PACKAGE_DIR / "bfast_fused_electric_pair.py"
API_ROOT = pathlib.Path(__file__).resolve().parents[1]
DRIVER_PATH = API_ROOT / "meep_gpu" / "driver.py"
PARITY_DIR = API_ROOT / "parity" / "meep_gpu"
GATE = PARITY_DIR / "probe_triton_bfast_fused_electric_pair.py"

#: A live three-component shear, the corpus class.
BFAST_K = (0.13, -0.21, 0.07)

#: The block the BFAST tail adds, and the ONLY thing that may separate this kernel
#: from ``kernels.fused_curl_constitutive_D``. DERIVED below rather than listed: it
#: is whatever ``bfast_pml_curl_step`` has that ``pml_curl_step`` does not.


@pytest.fixture(scope="module")
def product():
    return importlib.import_module(MODULE_NAME)


class _Magnetic:
    field_type = "B"


class _Electric:
    """An electric source that publishes the index the injection writes."""

    field_type = "D"

    def __init__(self, index=(1, 1, 1)) -> None:
        self._point_ix, self._point_iy, self._point_iz = index


class _ElectricWithoutIndex:
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
    """One shipped kernel's BODY as a list of top-level STATEMENT sources.

    THE GRAIN IS STATEMENTS, NOT LINES, and that is forced rather than chosen: the
    BFAST tail contains LINES that also appear elsewhere in both bodies — ``if BCY
    == METALLIC:`` guards both the advance mask and the curl mask — so subtracting
    the tail by text leaves orphaned guard headers and the equality fails on a body
    that is correct. Measured 2026-08-31. The tail is exactly ONE ``ast.If`` node.

    Read from the FILE rather than imported: the merge bar has no Triton, so
    importing ``kernels.py`` raises and this check would only ever bite on a device
    run. The text is EXACT — never ``ast.unparse``d, because what is compared
    includes the PARENTHESISATION.
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
        assert segment is not None, name
        lines = segment.splitlines()
        lines = [lines[0]] + [line[margin:] if line[:margin].strip() == ""
                              else line.lstrip() for line in lines[1:]]
        out.append("\n".join(_statements("\n".join(lines))))
    return out


@pytest.fixture(scope="module")
def bodies():
    return {
        "fused": _shipped_body(MODULE_PATH, "bfast_fused_curl_constitutive_D"),
        "bfast_curl": _shipped_body(PACKAGE_DIR / "bfast_curl.py",
                                    "bfast_pml_curl_step"),
        "ordinary_fused": _shipped_body(PACKAGE_DIR / "kernels.py",
                                        "fused_curl_constitutive_D"),
        "ordinary_curl": _shipped_body(PACKAGE_DIR / "kernels.py",
                                       "pml_curl_step"),
        "magnetic_twin": _shipped_body(
            PACKAGE_DIR / "bfast_fused_magnetic_pair.py",
            "bfast_fused_curl_constitutive_B"),
    }


def _build(cell_size=(0.8, 0.8, 0.8), dimensions=3, boundaries=None,
           pml_thickness=2, complex_storage=False, bfast_scaled_k=BFAST_K,
           k_point=(0.0, 0.0, 0.0), seed=23, thickness=None, **grid_kwargs):
    """A real BFAST Grid/Fields/PML triple on NumPy, PML storage enabled.

    Default: real storage, an active PML, a live three-component shear, k = 0, no
    fold. Every refusal test perturbs exactly one clause off this. Deliberately
    IDENTICAL in shape to the magnetic twin's fixture, so a difference between the
    two suites is a difference between the two products.
    """
    grid = Grid(resolution=10.0, cell_size=cell_size, dimensions=dimensions,
                boundaries=boundaries, courant=0.35, k_point=k_point,
                bfast_scaled_k=bfast_scaled_k, **grid_kwargs)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.set_background_eps(2.25)
    fields.enable_pml_storage()
    rng = numpy.random.default_rng(seed)
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
                 "f_w_Ex", "f_w_Ey", "f_w_Ez",
                 "f_bfast_Dx", "f_bfast_Dy", "f_bfast_Dz"):
        array = getattr(fields, name, None)
        if array is not None:
            array[...] = rng.uniform(-0.4, 0.4, size=grid.shape).astype(numpy.float32)
    if thickness is None:
        thickness = tuple(
            (pml_thickness, pml_thickness) if grid.shape[axis] >= 6 else (0, 0)
            for axis in range(3))
    return fields, PML(grid=grid, thickness=thickness)


def _reasons(product, fields, pml, sources=()):
    verdict = product.bfast_fused_electric_pair_coverage(fields, pml, sources)
    return [r for r in verdict.reasons if "array module" not in r]


# ---------------------------------------------------------------------------
# THE TRANSCRIPTION — the claim this whole product rests on
# ---------------------------------------------------------------------------

def bfast_insert(bodies) -> List[str]:
    """``bfast_pml_curl_step``'s own delta over ``pml_curl_step``, DERIVED.

    At statement grain that delta is exactly ONE ``if HAS_BFAST:`` node, which is
    what makes the subtraction below exact.
    """
    return [statement for statement in bodies["bfast_curl"]
            if statement not in bodies["ordinary_curl"]]


def test_the_bfast_delta_is_exactly_one_statement(bodies):
    insert = bfast_insert(bodies)
    assert len(insert) == 1, [s[:60] for s in insert]
    assert insert[0].startswith("if HAS_BFAST:"), insert[0][:60]


def test_removing_the_bfast_tail_reproduces_the_shipped_fused_pair(bodies):
    """``this kernel`` minus the BFAST statement IS ``fused_curl_constitutive_D``.

    An EXACT statement-list equality, in order. This is the diff a reviewer would do
    by hand, made mechanical: if any other statement moved, this fails and names it.
    """
    insert = bfast_insert(bodies)
    assert insert, "the BFAST insert is empty; this test compares nothing"
    without = [statement for statement in bodies["fused"]
               if not statement.startswith("if HAS_BFAST:")]
    assert without == bodies["ordinary_fused"], (
        "the BFAST fused electric pair is not fused_curl_constitutive_D plus the "
        "BFAST insert; first difference at index "
        + str(next((i for i, (a, b) in enumerate(
            zip(without, bodies["ordinary_fused"])) if a != b), "length")))


def test_removing_the_weld_reproduces_the_shipped_bfast_curl(bodies):
    """``this kernel`` minus the wall clear and the constitutive half IS
    ``bfast_curl.bfast_pml_curl_step``.

    The removed set is DERIVED — whatever ``fused_curl_constitutive_D`` has that
    ``pml_curl_step`` does not — so a change to the weld moves this test with it.
    """
    weld = [line for line in bodies["ordinary_fused"]
            if line not in bodies["ordinary_curl"]]
    assert weld, "the weld is empty; this test compares nothing"
    without_weld = [line for line in bodies["fused"] if line not in weld]
    assert without_weld == bodies["bfast_curl"], (
        "the BFAST fused electric pair minus the weld is not bfast_pml_curl_step; "
        "first difference at index "
        + str(next((i for i, (a, b) in enumerate(
            zip(without_weld, bodies["bfast_curl"])) if a != b), "length")))


def test_every_statement_comes_from_a_shipped_body(bodies):
    """Nothing in this kernel was written here."""
    shipped = set(bodies["bfast_curl"]) | set(bodies["ordinary_fused"])
    invented = [statement for statement in bodies["fused"]
                if statement not in shipped]
    assert invented == [], [s[:80] for s in invented]


def test_the_electric_weld_is_not_the_magnetic_one(bodies):
    """The two twins DIFFER, and where they differ is the D seam."""
    assert bodies["fused"] != bodies["magnetic_twin"]
    ours = "\n".join(bodies["fused"])
    theirs = "\n".join(bodies["magnetic_twin"])
    for component in range(3):
        line = (f"src{component} = v{component} * tl.load("
                f"ie{component} + idx, mask=live, other=0.0)")
        assert line in ours, line
        assert line not in theirs, line
    assert "v1 = tl.where(at_x, 0.0, v1)" in ours
    assert "v2 = tl.where(at_x, 0.0, v2)" in ours
    assert "v2 = tl.where(at_x, 0.0, v2)" not in theirs
    # The state arrays are the D side's.
    assert "f_bfast_B" not in ours


def test_the_advance_mask_precedes_the_state_store(bodies):
    """``adv`` is masked BEFORE ``state <- state + adv`` (S:902).

    That ordering is the whole reason ``at_x``/``at_y``/``at_z`` move one statement
    earlier in this body than in ``kernels.fused_curl_constitutive_D``, and it is
    read INSIDE the one tail statement rather than across the body.
    """
    joined = "\n".join(bodies["fused"])
    predicates = joined.index("at_x, at_y, at_z = i == 0, j == 0, k == 0")
    tail = joined.index("if HAS_BFAST:")
    first_mask = joined.index("adv0 = tl.where(at_y, 0.0, adv0)")
    state_store = joined.index("tl.store(s0 + idx, st0 + adv0, mask=live)")
    curl_mask = joined.index("curl0 = tl.where(at_y, 0.0, curl0)")
    assert predicates < tail < first_mask < state_store < curl_mask, (
        predicates, tail, first_mask, state_store, curl_mask)


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
    if product.bfast_fused_curl_constitutive_D is not None:
        pytest.skip("triton is installed here; the absence arm is exercised on the "
                    "merge-bar box without it")
    fields, pml = _build()
    assert product.bfast_fused_electric_pair_coverage(fields, pml, ()).reasons
    with pytest.raises(ImportError, match="triton"):
        product.bfast_fused_curl_constitutive_D_kernel()


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


def test_the_state_arrays_are_the_D_sides(product):
    from meep_gpu.triton_kernels.bfast_curl import BFAST_STATE_NAMES  # noqa: PLC0415

    assert BFAST_STATE_NAMES[product.CURL_SUB_STEP] == (
        "f_bfast_Dx", "f_bfast_Dy", "f_bfast_Dz")
    text = MODULE_PATH.read_text(encoding="utf-8")
    assert "BFAST_STATE_NAMES[CURL_SUB_STEP]" in text
    assert "f_bfast_B" not in text


# ---------------------------------------------------------------------------
# Admission and refusals
# ---------------------------------------------------------------------------

def test_the_corpus_family_is_admitted_WITH_its_electric_deposit(product):
    fields, pml = _build()
    assert _reasons(product, fields, pml, (_Electric(),)) == []


def test_a_walled_run_is_admitted(product):
    fields, pml = _build(boundaries={"x": "metallic", "y": "metallic",
                                     "z": "metallic"})
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
    """The measurement the module's docstring makes, executed rather than restated."""
    fields, pml = _build()
    assert product.CARRIES_DEPOSIT_REPAIR is True
    monkeypatch.setattr(product, "CARRIES_DEPOSIT_REPAIR", False)
    reasons = _reasons(product, fields, pml, (_Electric(),))
    assert any("is electric" in reason for reason in reasons), reasons


def test_the_magnetic_twin_keeps_the_flag_False_for_the_mirror_image_reason(product):
    twin = importlib.import_module(
        "meep_gpu.triton_kernels.bfast_fused_magnetic_pair")
    assert twin.CARRIES_DEPOSIT_REPAIR is False
    assert product.CARRIES_DEPOSIT_REPAIR is True


def test_the_flag_is_passed_to_the_clause_that_reads_it_and_not_merely_declared():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and getattr(node.func, "attr", "") == "seam_source_reasons"]
    assert len(calls) == 1, calls
    keywords = {keyword.arg: ast.unparse(keyword.value)
                for keyword in calls[0].keywords}
    assert keywords["carries_repair"] == "CARRIES_DEPOSIT_REPAIR", keywords
    assert ast.unparse(calls[0].args[2]) == "'D'", ast.unparse(calls[0].args[2])


def test_a_shear_free_run_is_refused_by_name(product):
    """The inverted clause: a BFAST-inactive run is the ordinary fused pair's."""
    fields, pml = _build(bfast_scaled_k=(0.0, 0.0, 0.0))
    reasons = _reasons(product, fields, pml, (_Electric(),))
    assert reasons, "a shear-free run must be refused"


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
                         thickness=((2, 2), (0, 2), (2, 2)))
    reasons = _reasons(product, fields, pml, (_Electric(),))
    assert reasons, "a folded run must be refused"


# ---------------------------------------------------------------------------
# Disjointness from the ordinary fused pair, in BOTH directions
# ---------------------------------------------------------------------------

def test_this_arm_and_the_ordinary_fused_pair_never_co_admit(product):
    from meep_gpu.triton_kernels import coverage  # noqa: PLC0415

    for shear in (BFAST_K, (0.0, 0.0, 0.0)):
        fields, pml = _build(bfast_scaled_k=shear)
        sources = (_Electric(),)
        mine = [r for r in product.bfast_fused_electric_pair_coverage(
            fields, pml, sources).reasons if "array module" not in r]
        theirs = [r for r in coverage.fused_pair_coverage(
            fields, pml, "D", sources).reasons if "array module" not in r]
        assert mine or theirs, (
            f"both predicates admitted a shear={shear!r} run: the arm table would "
            f"leave the slot unselected")
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
    ({"bfast_scaled_k": (0.0, 0.0, 0.0)}, (_Electric(),)),
    ({"complex_storage": True}, (_Electric(),)),
    ({"pml_thickness": 0}, (_Electric(),)),
    ({"boundaries": {"x": "metallic", "y": "metallic", "z": "metallic"}},
     (_Electric(),)),
])
def test_the_weld_is_never_wider_than_the_halves(product, kwargs, sources):
    from meep_gpu.triton_kernels import bfast_curl  # noqa: PLC0415

    fields, pml = _build(**kwargs)
    weld = product.bfast_fused_electric_pair_coverage(fields, pml, sources)
    curl = bfast_curl.bfast_pml_curl_coverage(fields, pml, "step_D")
    electric = bfast_curl.bfast_run_constitutive_coverage(fields, pml, "E")
    assert (not weld.covered) or (curl.covered and electric.covered), (
        weld.reasons, curl.reasons, electric.reasons)


def test_the_plan_refuses_rather_than_raises_on_a_numpy_host(product):
    fields, pml = _build()
    assert product.plan_bfast_fused_electric_pair(fields, pml, (_Electric(),)) is None
    assert product.plan_bfast_fused_electric_pair(fields, pml, None) is None


def test_the_bfast_coefficients_DIFFER_between_the_two_seams():
    """``bfast_curl_coefficients(magnetic=...)`` IS seam-dependent here.

    THE OPPOSITE OF THE BETA FAMILY, and that asymmetry is why this is a test rather
    than a comment. The D side reads ``BFAST_TERMS['step_D']`` and NEGATES both
    coefficients in host float64 (bfast_curl.py:343-351), so a builder that copied
    its twin's ``magnetic=True`` would hand this product six wrong words — where the
    same copy on the beta family would be harmless.
    """
    from meep_gpu.triton_kernels import bfast_curl  # noqa: PLC0415

    invariant = (False, False, False)
    magnetic = bfast_curl.bfast_curl_coefficients(BFAST_K, invariant, magnetic=True)
    electric = bfast_curl.bfast_curl_coefficients(BFAST_K, invariant, magnetic=False)
    assert magnetic != electric, (magnetic, electric)
    beta = importlib.import_module("meep_gpu.triton_kernels.special_kz")
    assert beta.beta_curl_coefficients(0.3, 0.05, magnetic=True,
                                       complex_storage=False) \
        == beta.beta_curl_coefficients(0.3, 0.05, magnetic=False,
                                       complex_storage=False), (
        "the beta family's flag is inert on real storage; if that ever changes the "
        "two products' comments about it both need revisiting")


# ---------------------------------------------------------------------------
# Registration - the product is known to every record that must know it
# ---------------------------------------------------------------------------

def test_the_composer_routes_this_product_and_the_driver_seam_released_it():
    """ROUTED 2026-09-02 by the installer wave, RELEASED 2026-09-13 — both halves.

    This test replaces the deferral it used to make. Until 2026-09-02 it asserted
    that ``launch.py`` and ``fastpath.py`` named this module NOWHERE, which was the
    seam that kept a certified-but-unrouted product deferred. The wave routed it:
    ``launch.CERTIFIED_FUSED_PRODUCTS`` holds its row and
    ``_install_certified_fused_products`` builds it through the shared
    ``_install_fused_pair``, taking both slots only when ``_pair_may_absorb`` finds
    the arm table has already given them to the arms this kernel implements.

    WIRING WAS NOT RELEASING, and this is the file where that sentence ended. It
    was named ``..._and_dispatch_still_refuses_it`` for as long as the label sat
    outside ``fastpath.RELEASED_FUSED_ARMS``; on 2026-09-13 the Phase B batch
    ``dispatch_fused_route_2026-09-13_phaseB`` drove it through the driver's own
    consults on ``bfast_1d`` — the corpus case this product's D seam was selected
    for — so the release names that case and the pending rung no longer holds it.

    THE PARTITION IS STILL EXACTLY ONE OF TWO, in the other direction: the label
    is in ``ARM_CERTIFICATION`` (its ledger entry was cut by ``seed_triton_welds.py``
    from the 2026-09-12 identity fleet) and out of ``PENDING_DEVICE_GATE_ARMS``.
    Both are asserted rather than one, because a label released while still pending
    would be a plan claiming a certification the rung says it lacks.
    """
    import pathlib as _pathlib  # noqa: PLC0415

    from meep_gpu import fastpath as _fastpath  # noqa: PLC0415
    from meep_gpu.triton_kernels import launch as _launch  # noqa: PLC0415

    source = _pathlib.Path(_launch.__file__).read_text(encoding="utf-8")
    assert "plan_bfast_fused_electric_pair" in source
    row = _launch.CERTIFIED_FUSED_PRODUCTS["bfast_fused_electric_pair"]
    assert row["module"] == "bfast_fused_electric_pair"
    assert row["builder"] == "plan_bfast_fused_electric_pair"
    label = row["label"]
    assert label == 'fused pair D (BFAST)'
    assert _fastpath.arm_is_fused(label)
    assert label in _fastpath.RELEASED_FUSED_ARMS
    assert _fastpath.RELEASED_FUSED_ARMS[label] == ("bfast_1d",)
    assert label in _fastpath.ARM_CERTIFICATION
    assert label not in _fastpath.PENDING_DEVICE_GATE_ARMS
    assert label in _fastpath.FUSED_ARM_CONSTITUENTS
def test_the_module_is_named_in_the_deposit_repair_wiring_records():
    text = (PACKAGE_DIR.parent
            / "test_fused_pair_deposit_wiring.py").read_text(encoding="utf-8")
    assert f"triton_kernels/{MODULE_PATH.name}" in text


def test_the_module_is_declared_ROUTED_rather_than_NOT_AN_ARM():
    """It LEFT ``test_triton_planner_composition.NOT_AN_ARM`` on 2026-09-02.

    Both directions, because both failures are wrong in opposite ways: a routed
    module still named in that tuple would tell a reader the planner cannot reach
    it, and a module missing from ``SUPPORT_MODULES`` would break the lazy-import
    seam this package rests on.
    """
    from meep_gpu.test_triton_planner_composition import NOT_AN_ARM  # noqa: PLC0415
    from meep_gpu.triton_kernels import launch as _launch  # noqa: PLC0415

    assert "bfast_fused_electric_pair" not in NOT_AN_ARM
    assert "bfast_fused_electric_pair" in _launch.SUPPORT_MODULES
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
    assert "bfast_fused_electric_pair" in text
    assert "bfast_fused_curl_constitutive_D" in text


def test_the_weld_driver_the_board_and_the_battery_all_know_this_product():
    driver = (PARITY_DIR / "drive_triton_weld_gates.py").read_text(encoding="utf-8")
    board = (PARITY_DIR
             / "build_triton_fusion_matrix.py").read_text(encoding="utf-8")
    battery = (PARITY_DIR / "triton_predicate_battery.py").read_text(encoding="utf-8")
    assert "probe_triton_bfast_fused_electric_pair.py" in driver
    assert '"bfast_fused_electric_pair"' in board
    assert "bfast_fused_electric_pair_coverage" in battery


def load_gate():
    import importlib.util  # noqa: PLC0415

    spec = importlib.util.spec_from_file_location("_bfep2_gate", GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_gates_no_device_legs_all_pass_here():
    gate = load_gate()
    for leg in gate.NO_DEVICE_LEGS:
        row = leg()
        assert row.get("passed") is True, (leg.__name__, row.get("findings"))


def test_every_mutation_the_gate_declares_is_ARMED_against_the_shipped_source():
    gate = load_gate()
    source = gate.shipped_source()
    for name, _kind, _reason, rewrite in gate.mutation_table():
        mutated, count = rewrite(source)
        assert count >= 1, name
        assert mutated != source, name
        ast.parse(mutated)      # a mutant that does not parse measured nothing


def test_every_mutation_is_scored_on_a_case_that_enters_the_branch_it_rewrites():
    gate = load_gate()
    for name, _kind, _reason, _rewrite in gate.mutation_table():
        gate.mutation_case_for(name)   # raises if the pairing is dead


def test_the_gate_binds_the_seam_passes_the_driver_actually_calls(product):
    gate = load_gate()
    assert tuple(gate.SEAM_PASSES) == tuple(product.REPLACES)


def test_the_gate_carries_a_carry_family_and_a_null_control():
    gate = load_gate()
    assert gate.CARRY_CASES, "the deposit carry is this cell's whole value"
    text = GATE.read_text(encoding="utf-8")
    assert "bracket" in text and "null" in text.lower()


def test_the_gate_carries_the_certified_kernel_identity_arm():
    """``HAS_BFAST = 0`` must reproduce ``kernels.fused_curl_constitutive_D``."""
    text = GATE.read_text(encoding="utf-8")
    assert "has_bfast" in text
    assert "fused_curl_constitutive_D" in text


def test_the_gate_pins_the_files_this_product_actually_rests_on():
    gate = load_gate()
    names = set(gate.source_hashes())
    for required in ("meep_gpu/triton_kernels/bfast_fused_electric_pair.py",
                     "meep_gpu/triton_kernels/bfast_curl.py",
                     "meep_gpu/triton_kernels/kernels.py",
                     "meep_gpu/deposit_repair.py",
                     "meep_gpu/stepping.py",
                     "meep_gpu/driver.py",
                     "meep_gpu/test_triton_bfast_fused_electric_pair.py"):
        assert required in names, required


def test_the_gate_declares_the_cell_the_board_scored():
    gate = load_gate()
    assert gate.CELL == ("D->E", "BFAST PML", "BFAST run")
