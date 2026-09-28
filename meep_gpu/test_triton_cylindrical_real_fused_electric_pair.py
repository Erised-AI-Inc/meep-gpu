"""Tests for the Dcyl m = 0 fused ELECTRIC pair — ``step_D`` welded into ``update_E``.

Everything here runs on a laptop: no GPU, no CuPy, no Triton. What needs hardware —
byte identity per COMPLETE driver step against the array path and against the two
separately certified products this launch replaces — lives in
``parity/meep_gpu/probe_triton_cylindrical_real_fused_electric_pair.py``. A green
suite here is not a certification, and the module under test says so too.

What IS pinned here:

* the cell this product occupies, read off the census, and that it is the SAME three
  rows its magnetic twin serves on the other seam;
* the transcription, read off the shipped source: every curl line must be
  ``cylindrical_triton.cyl_pml_curl_step``'s AND the released magnetic twin's, and
  the wall clear and constitutive half must be
  ``kernels.fused_curl_constitutive_D``'s. The two twins differ only in which
  ``BACKWARD`` arm compiles, so a line that is in one and not the other is a drift
  between a gated body and an ungated one;
* the FIVE lines that are compile-time ABSENT at ``BACKWARD = 0`` and are therefore
  executed for the first time by this product;
* every clause of the seam predicate, in both directions, including
  ``CARRIES_DEPOSIT_REPAIR`` — which on this cell is the product rather than one
  clause of it, since all three corpus rows declare an electric source;
* that the two driver passes ``REPLACES`` omits really cannot execute on a grid this
  predicate admits, executed rather than argued;
* that ``ZM_X`` and ``ZM_Y`` are structurally false on every admitted Dcyl grid, so
  the mutation table's choice to arm ``ZM_Z`` alone is licensed;
* the deferral — nothing in ``launch.py`` or ``fastpath.py`` names this module.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import importlib.util
import json
import pathlib
import sys

import pytest

from meep_gpu.fields import Fields, IYEE_SHIFTS
from meep_gpu.grid import Grid
from meep_gpu.pml import PML
from meep_gpu.triton_kernels import cylindrical_real_fused_electric_pair as product
from meep_gpu.triton_kernels import cylindrical_real_fused_magnetic_pair as twin
from meep_gpu.triton_kernels import cylindrical_triton as cyl
from meep_gpu.triton_kernels.coverage import zero_metal_axes

MODULE_NAME = "meep_gpu.triton_kernels.cylindrical_real_fused_electric_pair"
PACKAGE_DIR = pathlib.Path(product.__file__).parent
MODULE_PATH = PACKAGE_DIR / "cylindrical_real_fused_electric_pair.py"
API_ROOT = PACKAGE_DIR.parents[1]
GATE = (API_ROOT / "parity" / "meep_gpu"
        / "probe_triton_cylindrical_real_fused_electric_pair.py")
CENSUS = (API_ROOT / "parity" / "meep_gpu" / "results"
          / "predicate_coverage_triton_2026-08-31_cells")


def build(cell=(2.0, 0.0, 2.0), resolution=10.0, courant=0.5, m=0,
          force_complex_fields=False):
    """A real Dcyl Grid/Fields/PML triple on NumPy, with the PML storage allocated."""
    grid = Grid(resolution=resolution, cell_size=cell, cylindrical=True, m=m,
                boundaries={"z": "metallic"}, courant=courant)
    fields = Fields(grid=grid, force_complex_fields=force_complex_fields)
    fields.enable_pml_storage()
    fields.enable_field_storage()
    return fields, PML(grid=grid, thickness={"x": (0, 0.3), "z": 0.3})


def residual(verdict):
    return [reason for reason in verdict.reasons if "cupy" not in reason]


def electric_deposit(fields):
    """A REAL source that publishes the index the injection writes.

    :class:`Source` below carries a ``field_type`` and nothing else: enough to PLACE
    a source in a seam, and deliberately not enough to CARRY one.
    """
    from meep_gpu import deposit_repair
    from meep_gpu.sources import GaussianEnvelope, VolumeSource

    centre = (0.25 * float(fields.grid.cell_size[0]), 0.0, 0.0)
    source = VolumeSource(grid=fields.grid, component="Ez", center=centre,
                          size=(0.0, 0.0, 0.0),
                          envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2),
                          amplitude=1.0)
    assert source._n_source_points, "the case deposits nothing and measures nothing"
    assert deposit_repair._deposit_index(source) is not None, (
        "the case cannot exercise the repair: this source publishes no deposit index")
    return source


class Source:
    def __init__(self, field_type: str) -> None:
        self.field_type = field_type


def load_gate():
    spec = importlib.util.spec_from_file_location("probe_cyl_real_fused_D", GATE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def census_rows():
    def load(name):
        path = CENSUS / name
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines()
                if line.strip()]

    record = load("examples.jsonl") + load("tests.jsonl")
    matched = {(r.get("leg"), r.get("row")): r
               for r in load("tests_param_matched.jsonl")}
    record = [matched.pop((r.get("leg"), r.get("row")), r) for r in record]
    return [r for r in record if r.get("measured")]


def covered(row, key):
    entry = row["predicates"].get(key, {})
    return bool(entry.get("covered_modulo_backend", entry.get("covered")))


def shipped(path: pathlib.Path, name: str) -> str:
    """One function's source text, docstring and comments removed."""
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    node = next(found for found in ast.walk(tree)
                if isinstance(found, ast.FunctionDef) and found.name == name)
    lines = ast.get_source_segment(text, node).splitlines()
    body = node.body[0]
    if (isinstance(body, ast.Expr) and isinstance(body.value, ast.Constant)
            and isinstance(body.value.value, str)):
        start = body.lineno - node.lineno
        end = (body.end_lineno or body.lineno) - node.lineno
        lines = lines[:start] + lines[end + 1:]
    return "\n".join(line.split("#", 1)[0].strip() for line in lines
                     if line.split("#", 1)[0].strip())


MINE = lambda: shipped(MODULE_PATH, "cyl_real_fused_curl_constitutive_D")
TWIN = lambda: shipped(PACKAGE_DIR / "cylindrical_real_fused_magnetic_pair.py",
                       "cyl_real_fused_curl_constitutive_B")


# ---------------------------------------------------------------------------
# The cell
# ---------------------------------------------------------------------------

def test_the_cell_is_the_three_rows_its_magnetic_twin_serves_on_the_other_seam():
    """The two halves are the SAME two predicates; only the seam moves."""
    rows = census_rows()
    if len(rows) != 186:
        pytest.skip(f"the census at {CENSUS} is absent or truncated ({len(rows)} rows)")
    electric = sorted(f"{r['leg']}:{r['row']}" for r in rows
                      if covered(r, "cylindrical_curl@step_D")
                      and covered(r, "cylindrical_constitutive@update_E"))
    magnetic = sorted(f"{r['leg']}:{r['row']}" for r in rows
                      if covered(r, "cylindrical_curl@step_B")
                      and covered(r, "cylindrical_constitutive@update_H"))
    assert electric == magnetic, (electric, magnetic)
    assert len(electric) == 3, electric


def test_every_row_of_the_cell_carries_an_electric_deposit():
    """Which is why CARRIES_DEPOSIT_REPAIR is the product rather than one clause."""
    rows = census_rows()
    if len(rows) != 186:
        pytest.skip(f"the census at {CENSUS} is absent or truncated ({len(rows)} rows)")
    cell = [r for r in rows if covered(r, "cylindrical_curl@step_D")
            and covered(r, "cylindrical_constitutive@update_E")]
    assert cell
    for row in cell:
        kinds = row["configuration"].get("source_field_types") or []
        assert kinds and all(str(kind) != "B" for kind in kinds), (row["row"], kinds)


# ---------------------------------------------------------------------------
# The transcription: this body is the twin's, on the other BACKWARD arm
# ---------------------------------------------------------------------------

CURL_LINES = (
    "curl0 = dtdx * ((c_p - c) + (b - b_z))",
    "curl1 = dtdx * ((a_z - a) + (c - c_r))",
    "curl2 = dtdx * ((b_r - b) + (a - a_p))",
    "n0 = ((p0 * km_y) - curl0) * si_y",
    "n1 = ((p1 * km_z) - curl1) * si_z",
    "n2 = ((p2 * km_x) - curl2) * si_x",
)

#: The lines that are COMPILE-TIME ABSENT at ``BACKWARD = 0``, and are therefore
#: executed for the first time by this product. Its magnetic twin CARRIES every one
#: of them and compiles none.
BACKWARD_ONE_LINES = (
    "p_here = tl.load(pfx + idx, mask=live, other=0.0)",
    "p_down = tl.load(pfx + o_r, mask=vr, other=0.0)",
    "curl2 = dtdx * ((p_down - p_here) + (a - a_p))",
    "curl1 = tl.where(at_r, 0.0, curl1)",
    "v1 = tl.where(at_r, 0.0, v1)",
)


@pytest.mark.parametrize("line", CURL_LINES + BACKWARD_ONE_LINES)
def test_every_curl_line_is_the_certified_bodys_and_the_released_twins(line):
    """The parenthesisation decides float32 bits; a reformat is a different number.

    Checked against BOTH sources, because the two twins differ only in which arm
    compiles: a line in one and not the other is a drift between a gated body and an
    ungated one.
    """
    assert line in MINE()
    assert line in TWIN()
    assert line in shipped(PACKAGE_DIR / "cylindrical_triton.py",
                           "cyl_pml_curl_step")


def test_the_wall_clear_is_the_D_familys_six_rows_and_not_the_Bs_three():
    """``zero_metal_D`` clears the two TANGENTIAL components on each axis."""
    mine, twin_text = MINE(), TWIN()
    for line in ("v1 = tl.where(at_x, 0.0, v1)", "v2 = tl.where(at_x, 0.0, v2)",
                 "v0 = tl.where(at_y, 0.0, v0)", "v2 = tl.where(at_y, 0.0, v2)",
                 "v0 = tl.where(at_z, 0.0, v0)", "v1 = tl.where(at_z, 0.0, v1)"):
        assert line in mine, line
    # ...and the twin's THREE rows (component m on axis m) must NOT all be here: the
    # B family clears the normal component, which is a different set.
    assert "v1 = tl.where(at_y, 0.0, v1)" in twin_text
    assert "v1 = tl.where(at_y, 0.0, v1)" not in mine


def test_the_constitutive_half_is_the_SCALE_1_arm_with_D_on_the_left():
    """``update_E`` writes ``source * inverse_epsilon`` (stepping.py:1011-1013);
    ``update_H`` writes ``source``. A body that dropped the multiply would be the
    magnetic twin's constitutive welded onto the electric seam."""
    mine = MINE()
    for index in (0, 1, 2):
        assert (f"src{index} = v{index} * "
                f"tl.load(ie{index} + idx, mask=live, other=0.0)") in mine
        assert f"src{index} = v{index}\n" not in mine + "\n"


def test_the_axis_rules_are_the_D_sides_and_both_halves_are_present():
    """``Dz[0] += (4*Courant)*Hp[0]`` then ``Dp[0] = 0`` — and the add is a POST-add
    on the updated field, deliberately not folded into the curl."""
    mine = MINE()
    assert "v1 = tl.where(at_r, 0.0, v1)" in mine
    assert ("v2 = tl.where(at_r, v2 + four_dtdx * "
            "tl.load(hp + idx, mask=live, other=0.0),") in mine
    # The B side's own rule must NOT run here: it is in the other arm.
    assert mine.count("v0 = tl.where(at_r, 0.0, v0)") == 1  # the else arm only


def test_the_body_differs_from_the_RELEASED_twin_by_exactly_four_things():
    """THE WHOLE TRANSCRIPTION CLAIM, as a mechanical diff rather than a review.

    The magnetic twin released 2026-08-20 and carries BOTH ``BACKWARD`` arms. This
    body is that one with (1) the kernel renamed, (2) three inverse-permittivity
    pointers added, (3) the wall clear swapped from the B family's three rows to the
    D family's six, and (4) the three constitutive products gaining the ``SCALE = 1``
    multiply. Anything else in the diff is a drift between a GATED body and an
    ungated one, and a reader should not have to notice it.
    """
    import difflib

    mine, other = MINE().splitlines(), TWIN().splitlines()
    added, removed = [], []
    for line in difflib.unified_diff(other, mine, lineterm="", n=0):
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            continue
        (added if line.startswith("+") else removed).append(line[1:])

    assert removed == [
        "def cyl_real_fused_curl_constitutive_B(",
        "v0 = tl.where(at_x, 0.0, v0)",
        "v1 = tl.where(at_y, 0.0, v1)",
        "v2 = tl.where(at_z, 0.0, v2)",
        "src0 = v0", "src1 = v1", "src2 = v2",
    ], removed
    assert added == [
        "def cyl_real_fused_curl_constitutive_D(",
        "ie0, ie1, ie2,",
        "v1 = tl.where(at_x, 0.0, v1)", "v2 = tl.where(at_x, 0.0, v2)",
        "v0 = tl.where(at_y, 0.0, v0)", "v2 = tl.where(at_y, 0.0, v2)",
        "v0 = tl.where(at_z, 0.0, v0)", "v1 = tl.where(at_z, 0.0, v1)",
        "src0 = v0 * tl.load(ie0 + idx, mask=live, other=0.0)",
        "src1 = v1 * tl.load(ie1 + idx, mask=live, other=0.0)",
        "src2 = v2 * tl.load(ie2 + idx, mask=live, other=0.0)",
    ], added


def test_the_body_carries_no_boundary_constexpr():
    """This kernel serves exactly the Dcyl triple and compiles it in."""
    mine = MINE()
    for needle in ("BCX", "BCY", "BCZ", "PERIODIC", "METALLIC", "MIRROR"):
        assert needle not in mine, needle


def test_the_order_is_axis_rule_then_wall_then_store_then_constitutive_read():
    lines = MINE().splitlines()
    axis = lines.index("v1 = tl.where(at_r, 0.0, v1)")
    clear = lines.index("v0 = tl.where(at_z, 0.0, v0)")
    store = lines.index("tl.store(f0 + idx, v0, mask=live)")
    read = lines.index("src0 = v0 * tl.load(ie0 + idx, mask=live, other=0.0)")
    assert axis < clear < store < read, (axis, clear, store, read)


def test_the_backward_constexpr_is_step_Ds_and_only_step_Ds():
    assert product.BACKWARD == cyl.SUB_STEPS[product.CURL_SUB_STEP]["backward"] == 1
    assert twin.BACKWARD == 0


def test_REPLACES_names_three_passes_and_INERT_PASSES_names_the_other_two():
    assert product.REPLACES == ("step_D", "zero_metal_D", "update_E")
    assert set(product.INERT_PASSES) == {"fill_symmetry_bc_D",
                                         "fill_folded_far_ghosts_D"}
    assert product.CylindricalRealFusedElectricPairPlan.replaces == product.REPLACES
    # ...and the SEAM has five: the gate absorbs all of them, so a fill that ever did
    # work on an admitted grid becomes a byte divergence there.
    assert set(load_gate().SEAM_PASSES) == (set(product.REPLACES)
                                            | set(product.INERT_PASSES))


def test_the_two_omitted_passes_are_inert_on_every_admitted_grid():
    """EXECUTED, not argued — the gate's own leg, which runs here."""
    row = load_gate().inert_passes_leg()
    assert row["passed"], row["findings"]
    assert row["rows"], "the leg scored no grid"
    for entry in row["rows"]:
        assert entry["moved_by_the_inert_passes"] == []
        assert entry["moved_by_zero_metal_D"], entry


def test_ZM_X_and_ZM_Y_are_structurally_false_on_every_admitted_configuration():
    row = load_gate().structural_walls_leg()
    assert row["passed"], row["findings"]
    for entry in row["rows"]:
        assert entry["zero_metal_axes"] == [False, False, True], entry


# ---------------------------------------------------------------------------
# The seam clauses
# ---------------------------------------------------------------------------

def test_a_dcyl_m0_grid_is_admitted_modulo_the_numpy_host():
    fields, pml = build()
    assert residual(product.cylindrical_real_fused_electric_pair_coverage(
        fields, pml, ())) == []


def test_a_magnetic_source_is_admitted_and_an_electric_DEPOSIT_is_carried():
    """The polarity, both ways, and the CARRY — which on this cell IS the product."""
    fields, pml = build()
    assert residual(product.cylindrical_real_fused_electric_pair_coverage(
        fields, pml, (Source("B"),))) == []
    assert residual(product.cylindrical_real_fused_electric_pair_coverage(
        fields, pml, (electric_deposit(fields),))) == []


def test_an_electric_source_that_publishes_no_deposit_index_is_REFUSED():
    """FAIL CLOSED. A source the repair cannot save is refused BY NAME."""
    fields, pml = build()
    reasons = product.cylindrical_real_fused_electric_pair_coverage(
        fields, pml, (Source("D"),)).reasons
    assert any("does not publish the index it writes" in reason
               for reason in reasons), reasons


def test_the_flag_at_False_would_cost_the_whole_cell(monkeypatch):
    """MEASURED, not asserted: with the flag off, an electric deposit is refused."""
    fields, pml = build()
    monkeypatch.setattr(product, "CARRIES_DEPOSIT_REPAIR", False)
    reasons = product.cylindrical_real_fused_electric_pair_coverage(
        fields, pml, (electric_deposit(fields),)).reasons
    assert any("is electric" in reason for reason in reasons), reasons


def test_the_flag_is_declared_and_passed_to_the_clause_that_reads_it():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert product.CARRIES_DEPOSIT_REPAIR is True
    assert "CARRIES_DEPOSIT_REPAIR = True" in source
    assert "carries_repair=CARRIES_DEPOSIT_REPAIR" in source
    # ...and the MAGNETIC twin's is False, because on the same three rows the
    # magnetic seam is empty without any repair.
    assert twin.CARRIES_DEPOSIT_REPAIR is False


def test_an_undeclared_source_list_is_a_refusal_and_not_an_empty_set():
    fields, pml = build()
    reasons = product.cylindrical_real_fused_electric_pair_coverage(
        fields, pml, None).reasons
    assert any("was not declared" in reason for reason in reasons), reasons


def test_m_not_zero_is_refused_by_name():
    fields, pml = build(m=1, force_complex_fields=True)
    reasons = product.cylindrical_real_fused_electric_pair_coverage(
        fields, pml, ()).reasons
    assert any("m = 0" in reason for reason in reasons), reasons


def test_a_cartesian_grid_is_refused_by_name():
    grid = Grid(resolution=10.0, cell_size=(1.6, 1.6, 0.0), dimensions=2,
                boundaries={"x": "metallic", "y": "metallic", "z": "periodic"})
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    reasons = product.cylindrical_real_fused_electric_pair_coverage(
        fields, PML(grid=grid, thickness=0.2), ()).reasons
    assert any("not cylindrical" in reason for reason in reasons), reasons


def test_no_pml_is_refused_because_update_E_is_a_no_op_without_one():
    fields, _pml = build()
    reasons = product.cylindrical_real_fused_electric_pair_coverage(
        fields, None, ()).reasons
    assert any("PML" in reason for reason in reasons), reasons


def test_the_predicate_reports_both_halves_reasons_with_their_side_named():
    grid = Grid(resolution=10.0, cell_size=(1.6, 1.6, 0.0), dimensions=2,
                boundaries={"x": "metallic", "y": "metallic", "z": "periodic"})
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    reasons = product.cylindrical_real_fused_electric_pair_coverage(
        fields, PML(grid=grid, thickness=0.2), ()).reasons
    assert any(r.startswith("cylindrical curl half: ") for r in reasons), reasons
    assert any(r.startswith("cylindrical constitutive half: ")
               for r in reasons), reasons


def test_a_registered_susceptibility_is_refused_by_the_E_half():
    """``update_E``'s source becomes ``(D - sum P)``; that is the ADE kernel's."""
    from meep_gpu.dispersion import PolarizationState, Susceptibility
    import numpy

    fields, pml = build()
    fields.polarizations.append(PolarizationState(
        Susceptibility(frequency=1.0, gamma=0.1),
        {"Ex": 0.3, "Ey": 0.3, "Ez": 0.3}, fields.grid, numpy.float32))
    reasons = product.cylindrical_real_fused_electric_pair_coverage(
        fields, pml, ()).reasons
    assert any("susceptibility is registered" in reason for reason in reasons), reasons


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

def test_the_plan_refuses_a_grid_that_stores_more_than_one_phi_cell():
    import numpy

    with pytest.raises(ValueError, match="one phi cell"):
        product.CylindricalRealFusedElectricPairPlan(
            shape=(6, 3, 6), dtdx=0.3, zero_metal=(False, False, True), block=64,
            targets=[numpy.zeros((6, 3, 6), dtype=numpy.float32)] * 3,
            auxiliaries=[numpy.zeros((6, 3, 6), dtype=numpy.float32)] * 3,
            sources=[numpy.zeros((6, 3, 6), dtype=numpy.float32)] * 3,
            curl_coefficients=[numpy.zeros(6, dtype=numpy.float32)] * 6,
            e_targets=[numpy.zeros((6, 3, 6), dtype=numpy.float32)] * 3,
            e_aux=[numpy.zeros((6, 3, 6), dtype=numpy.float32)] * 3,
            inverse_epsilon=[numpy.zeros((6, 3, 6), dtype=numpy.float32)] * 3,
            e_coefficients=[numpy.zeros(6, dtype=numpy.float32)] * 6,
            xp=numpy)


def test_the_plan_refuses_a_placeholder_inverse_epsilon_binding():
    """The kernel LOADS inv_eps on every launch; a null would be read as a
    coefficient. The magnetic twin needs no such clause — its arm multiplies by
    nothing."""
    import numpy

    with pytest.raises(ValueError, match="inv_eps"):
        product.CylindricalRealFusedElectricPairPlan(
            shape=(6, 1, 6), dtdx=0.3, zero_metal=(False, False, True), block=64,
            targets=[numpy.zeros((6, 1, 6), dtype=numpy.float32)] * 3,
            auxiliaries=[numpy.zeros((6, 1, 6), dtype=numpy.float32)] * 3,
            sources=[numpy.zeros((6, 1, 6), dtype=numpy.float32)] * 3,
            curl_coefficients=[numpy.zeros(6, dtype=numpy.float32)] * 6,
            e_targets=[numpy.zeros((6, 1, 6), dtype=numpy.float32)] * 3,
            e_aux=[numpy.zeros((6, 1, 6), dtype=numpy.float32)] * 3,
            inverse_epsilon=None,
            e_coefficients=[numpy.zeros(6, dtype=numpy.float32)] * 6,
            xp=numpy)


def test_the_builder_refuses_every_configuration_the_predicate_refuses():
    """``None`` is the only refusal: a raise reaches a caller that would otherwise
    have stepped correctly."""
    for kwargs in ({"m": 1, "force_complex_fields": True},):
        fields, pml = build(**kwargs)
        assert not product.cylindrical_real_fused_electric_pair_coverage(
            fields, pml, ()).covered
        assert product.plan_cylindrical_real_fused_electric_pair(
            fields, pml, ()) is None
    fields, pml = build()
    assert product.plan_cylindrical_real_fused_electric_pair(
        fields, None, ()) is None


# ---------------------------------------------------------------------------
# The optional-import contract and the deferral
# ---------------------------------------------------------------------------

def test_the_package_does_not_import_the_module_eagerly():
    for name in list(sys.modules):
        if name == MODULE_NAME:
            del sys.modules[name]
    importlib.import_module("meep_gpu.triton_kernels")
    assert MODULE_NAME not in sys.modules


def test_the_module_imports_and_answers_coverage_with_triton_absent(monkeypatch):
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "triton" or name.startswith("triton."):
            raise ImportError("triton is blocked for this test")
        return real_import(name, *args, **kwargs)

    for name in [n for n in sys.modules if n.startswith("triton")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.delitem(sys.modules, MODULE_NAME, raising=False)
    monkeypatch.setattr(builtins, "__import__", blocked)
    reloaded = importlib.import_module(MODULE_NAME)
    fields, pml = build()
    assert isinstance(reloaded.cylindrical_real_fused_electric_pair_coverage(
        fields, pml, ()).covered, bool)
    assert reloaded.plan_cylindrical_real_fused_electric_pair(fields, pml, ()) is None
    with pytest.raises(ImportError):
        reloaded.cyl_real_fused_curl_constitutive_D_kernel()


def test_the_composer_routes_this_product_and_the_driver_seam_released_it():
    """ROUTED 2026-09-02 by the installer wave, RELEASED 2026-09-11 — both halves.

    This test replaces the deferral it used to make. Until 2026-09-02 it asserted
    that ``launch.py`` and ``fastpath.py`` named this module NOWHERE, which was the
    seam that kept a certified-but-unrouted product deferred. The wave routed it:
    ``launch.CERTIFIED_FUSED_PRODUCTS`` holds its row and
    ``_install_certified_fused_products`` builds it through the shared
    ``_install_fused_pair``, taking both slots only when ``_pair_may_absorb`` finds
    the arm table has already given them to the arms this kernel implements.

    WIRING WAS NOT RELEASING, and this is the file where that sentence ended. It
    was named ``..._and_dispatch_still_refuses_it`` for as long as the label sat
    outside ``fastpath.RELEASED_FUSED_ARMS``; on 2026-09-11
    ``dispatch_fused_route_2026-09-11_realarms`` drove it through the driver's own
    consults on ``cylindrical_m0``, a route-gate case ADDED for this pair and its
    magnetic sibling — until it existed no Dcyl shape had ever reached the driver
    seam, which is why this product waited on a new case rather than on a new
    measurement of an old one. The fourteen sibling products that were NOT driven
    keep the old name and the old assertion, which is what makes the two states
    legible per file.

    THE PARTITION IS STILL EXACTLY ONE OF TWO, in the other direction: the label
    is in ``ARM_CERTIFICATION`` (its ledger entry was cut from this campaign's
    fleet artifact by ``seed_triton_welds.py``) and out of
    ``PENDING_DEVICE_GATE_ARMS``. Both are asserted rather than one, because a
    label released while still pending would be a plan claiming a certification
    the rung says it lacks.
    """
    import pathlib as _pathlib  # noqa: PLC0415

    from meep_gpu import fastpath as _fastpath  # noqa: PLC0415
    from meep_gpu.triton_kernels import launch as _launch  # noqa: PLC0415

    source = _pathlib.Path(_launch.__file__).read_text(encoding="utf-8")
    assert "plan_cylindrical_real_fused_electric_pair" in source
    row = _launch.CERTIFIED_FUSED_PRODUCTS["cylindrical_real_fused_electric_pair"]
    assert row["module"] == "cylindrical_real_fused_electric_pair"
    assert row["builder"] == "plan_cylindrical_real_fused_electric_pair"
    label = row["label"]
    assert label == 'fused pair D (cylindrical)'
    assert _fastpath.arm_is_fused(label)
    assert label in _fastpath.RELEASED_FUSED_ARMS
    assert _fastpath.RELEASED_FUSED_ARMS[label] == ("cylindrical_m0",)
    assert label in _fastpath.ARM_CERTIFICATION
    assert label not in _fastpath.PENDING_DEVICE_GATE_ARMS
    assert label in _fastpath.FUSED_ARM_CONSTITUENTS
def test_the_module_is_declared_ROUTED_rather_than_NOT_AN_ARM():
    """It LEFT ``test_triton_planner_composition.NOT_AN_ARM`` on 2026-09-02.

    Both directions, because both failures are wrong in opposite ways: a routed
    module still named in that tuple would tell a reader the planner cannot reach
    it, and a module missing from ``SUPPORT_MODULES`` would break the lazy-import
    seam this package rests on.
    """
    from meep_gpu.test_triton_planner_composition import NOT_AN_ARM  # noqa: PLC0415
    from meep_gpu.triton_kernels import launch as _launch  # noqa: PLC0415

    assert "cylindrical_real_fused_electric_pair" not in NOT_AN_ARM
    assert "cylindrical_real_fused_electric_pair" in _launch.SUPPORT_MODULES
def test_the_module_is_named_in_the_deposit_repair_wiring_records():
    from meep_gpu.test_fused_pair_deposit_wiring import WIRED_FOR_THE_REPAIR
    key = "triton_kernels/cylindrical_real_fused_electric_pair.py"
    assert key in WIRED_FOR_THE_REPAIR


def test_the_single_launch_site_passes_the_packages_shared_guard_constant():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard)" \
        in source
    assert source.count("kernel[self._grid](") == 1


def test_the_fill_geometry_claim_matches_the_engines_own_Yee_table():
    """The docstring's ownership-mask table: for D, shift 0 on the two axes that are
    NOT the component's own."""
    for index, name in enumerate(("Dx", "Dy", "Dz")):
        zeros = tuple(axis for axis in range(3) if IYEE_SHIFTS[name][axis] == 0)
        assert index not in zeros, (name, zeros)
        assert len(zeros) == 2, (name, zeros)


# ---------------------------------------------------------------------------
# The gate's own no-device legs
# ---------------------------------------------------------------------------

def test_the_gate_exists_and_names_the_product_it_measures():
    assert GATE.exists()
    text = GATE.read_text(encoding="utf-8")
    assert "meep_gpu.triton_kernels.cylindrical_real_fused_electric_pair" in text
    assert "cyl_real_fused_curl_constitutive_D" in text


def test_the_gates_transcription_leg_traces_every_line_to_its_source():
    row = load_gate().transcription_leg()
    assert row["passed"], row["findings"]
    assert row["zero_metal_lines_checked"] == 6


def test_the_gates_predicate_leg_exercises_every_clause_in_both_directions():
    row = load_gate().predicate_leg()
    assert row["passed"], row["findings"]
    admitted = [r for r in row["rows"] if r["expect_admitted"]]
    refused = [r for r in row["rows"] if not r["expect_admitted"]]
    assert len(admitted) >= 3 and len(refused) >= 4, (len(admitted), len(refused))


def test_the_gates_corpus_leg_lands_on_the_cell_the_board_scored():
    row = load_gate().corpus_admission_leg()
    assert row["passed"], row["findings"]
    assert row["seam_instances_gained"] == 3, row["funnel"]
    assert row["board_rows"] == row["funnel"]["cell_rows"]
    assert (row["funnel"]["carrying_an_electric_deposit"]
            == row["funnel"]["both"] == 3)


def test_the_gate_arms_every_mutation_and_none_on_a_dead_branch():
    row = load_gate().dead_branch_leg()
    assert row["passed"], row["findings"]
    assert len(row["mutations"]) >= 13, row["mutations"]
    assert row["dead_constexprs"] == ["ZM_X", "ZM_Y"]


def test_the_material_mutation_is_scored_on_a_grid_whose_material_varies():
    """A shifted material read is a bitwise no-op in a uniform medium, which is the
    measured reason an earlier product's equivalent mutation came back UNCAUGHT. The
    gate BUILDS the grid and compares adjacent entries rather than declaring it."""
    gate = load_gate()
    assert gate.MUTATION_REQUIRES_A_VARYING_MATERIAL
    for name in gate.MUTATION_REQUIRES_A_VARYING_MATERIAL:
        assert name in [m[0] for m in gate.mutation_table()], name
        gate.mutation_case_for(name)          # raises if the material is constant
    # ...and the guard BITES: a constant epsilon must be refused. The grid is the
    # one the case builds; only the material is replaced.
    import numpy
    from meep_gpu.fields import Fields
    from meep_gpu.grid import Grid

    real = Fields.set_isotropic_epsilon_volume

    def uniform(self, epsilon, inverse):
        flat = numpy.full_like(numpy.asarray(epsilon), 2.25)
        return real(self, flat, (1.0 / flat).astype(numpy.float32))

    Fields.set_isotropic_epsilon_volume = uniform
    try:
        with pytest.raises(AssertionError, match="CONSTANT"):
            gate._assert_the_material_actually_varies(
                "probe", (2.0, 0.0, 2.0), 10.0, 0.5)
    finally:
        Fields.set_isotropic_epsilon_volume = real


def test_the_dead_branch_check_CATCHES_a_rewrite_of_a_dead_branch():
    """A green checker that cannot fail is not a check. This drives a rewrite that
    DOES touch the structurally-false ZM_X block and requires it to be reported."""
    gate = load_gate()
    source = gate._shipped_text("cyl_real_fused_curl_constitutive_D")
    blocks = gate._dead_blocks(source)
    assert set(blocks) == {"ZM_X", "ZM_Y"}, blocks
    # The ZM_X block's own first statement, removed.
    tampered, hits = gate._rewrite_block(
        source, ["if ZM_X:", "v1 = tl.where(at_x, 0.0, v1)",
                 "v2 = tl.where(at_x, 0.0, v2)"], ["v1 = v1", "v2 = v2"])
    assert hits == 1
    changed = [line.strip() for line in tampered.splitlines() if line.strip()]
    assert not gate._contains_run(changed, blocks["ZM_X"])
    # ...and a rewrite that only SHIFTS the block must NOT be reported, which is the
    # false positive an index-based comparison produced on m4.
    shifted, hits = gate._rewrite_block(
        source,
        ["v2 = tl.where(at_r, v2 + four_dtdx * tl.load(hp + idx, mask=live, other=0.0),",
         "v2)"], ["v2 = v2"])
    assert hits == 1
    moved_lines = [line.strip() for line in shifted.splitlines() if line.strip()]
    assert gate._contains_run(moved_lines, blocks["ZM_X"])
    assert gate._contains_run(moved_lines, blocks["ZM_Y"])


def test_the_carry_family_and_its_null_control_are_both_present():
    """A bracket that changes nothing is not load-bearing, so every carry case is run
    twice — and the unbracketed twin must diverge."""
    gate = load_gate()
    assert gate.CARRY_CASES
    text = GATE.read_text(encoding="utf-8")
    assert "bracket=False" in text
    assert "require_identical=False" in text
    assert gate.verdict_of({"first_divergence": None, "fused_kernel_launches": 3},
                           require_identical=False, require_launches_at_least=1,
                           require_moved=False)[0] is False
    assert gate.verdict_of({"first_divergence": {"array": "Ex"},
                            "fused_kernel_launches": 0},
                           require_identical=False, require_launches_at_least=1,
                           require_moved=False)[0] is False
    assert gate.verdict_of({"first_divergence": {"array": "Ex"},
                            "fused_kernel_launches": 1},
                           require_identical=False, require_launches_at_least=1,
                           require_moved=False)[0] is True


def test_the_gate_binds_the_seam_passes_the_driver_actually_calls():
    import meep_gpu.driver as driver_module
    gate = load_gate()
    for name in gate.SEAM_PASSES:
        assert hasattr(driver_module, name), name


def test_the_gate_pins_the_files_this_product_actually_rests_on():
    pinned = set(load_gate().source_hashes())
    for name in ("meep_gpu/triton_kernels/cylindrical_real_fused_electric_pair.py",
                 "meep_gpu/triton_kernels/cylindrical_real_fused_magnetic_pair.py",
                 "meep_gpu/triton_kernels/cylindrical_triton.py",
                 "meep_gpu/deposit_repair.py",
                 "meep_gpu/test_triton_cylindrical_real_fused_electric_pair.py"):
        assert name in pinned, name


def test_the_gate_has_not_been_run_or_names_the_run_it_took():
    """DEVICE STATUS, read off the gate rather than believed."""
    header = GATE.read_text(encoding="utf-8").split('"""')[1]
    assert "DEVICE STATUS" in header
    if "UNRUN" in header:
        assert "may cite it as a release" in header
    else:
        assert "results/" in header
