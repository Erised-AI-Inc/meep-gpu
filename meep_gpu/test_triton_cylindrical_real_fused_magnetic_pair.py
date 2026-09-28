"""Tests for the Dcyl m = 0 fused magnetic pair — ``step_B`` welded into ``update_H``.

Everything here runs on a laptop: no GPU, no CuPy, no Triton. What needs hardware —
byte identity against the array path and against the two separately certified
products this launch replaces — lives in
``parity/meep_gpu/probe_triton_cylindrical_real_fused_magnetic_pair.py``. A green
suite here is not a certification, and the module under test says so too.

What IS pinned here:

* the cell this product occupies and the fact that it is DISJOINT from its complex
  sibling's — 3 rows against 16, intersection empty, read off the census;
* that the two driver passes :data:`REPLACES` omits really cannot execute on a grid
  this predicate admits, executed rather than argued;
* that ``ZM_X`` and ``ZM_Y`` are structurally false on every admitted Dcyl grid, so
  the mutation table's choice to arm ``ZM_Z`` alone is licensed;
* every clause of the seam predicate, in both directions;
* the transcription, read off the shipped source, including the ORDER of the axis
  rule, the wall clear, the store and the constitutive read;
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
from meep_gpu.triton_kernels import cylindrical_real_fused_magnetic_pair as product
from meep_gpu.triton_kernels import cylindrical_triton as cyl
from meep_gpu.triton_kernels.coverage import zero_metal_axes

PACKAGE_DIR = pathlib.Path(product.__file__).parent
API_ROOT = PACKAGE_DIR.parents[1]
GATE = (API_ROOT / "parity" / "meep_gpu"
        / "probe_triton_cylindrical_real_fused_magnetic_pair.py")
CENSUS = (API_ROOT / "parity" / "meep_gpu" / "results"
          / "predicate_coverage_2026-08-16_wired_convention")


def build(cell=(2.0, 0.0, 2.0), resolution=10.0, courant=0.5, m=0,
          force_complex_fields=False):
    """A real Dcyl Grid/Fields/PML triple on NumPy, with the PML storage allocated."""
    grid = Grid(resolution=resolution, cell_size=cell, cylindrical=True, m=m,
                boundaries={"z": "metallic"}, courant=courant)
    fields = Fields(grid=grid, force_complex_fields=force_complex_fields)
    fields.enable_pml_storage()
    return fields, PML(grid=grid, thickness={"x": (0, 0.3), "z": 0.3})


def residual(verdict):
    return [reason for reason in verdict.reasons if "cupy" not in reason]


class Source:
    def __init__(self, field_type: str) -> None:
        self.field_type = field_type


def load_gate():
    spec = importlib.util.spec_from_file_location("probe_cyl_real_fused_B", GATE)
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


# ---------------------------------------------------------------------------
# The cell, and that it is not the complex sibling's
# ---------------------------------------------------------------------------

def test_this_cell_is_disjoint_from_the_complex_cylindrical_one():
    """3 rows against 16, intersection empty — measured off the census, not assumed.

    Two products in one package both called "the cylindrical B->H pair" would be a
    coverage question dressed as a naming one. They are different arms: this one
    takes the m = 0 REAL curl, its sibling the |m| >= 1 complex one.
    """
    rows = census_rows()
    if len(rows) != 186:
        pytest.skip(f"the census record is not present ({len(rows)} rows)")
    real = {f"{r['leg']}:{r['row']}" for r in rows
            if covered(r, "cylindrical_curl@step_B")}
    complex_arm = {f"{r['leg']}:{r['row']}" for r in rows
                   if covered(r, "cylindrical_complex_curl@step_B")}
    assert len(real) == 3, sorted(real)
    assert len(complex_arm) == 16
    assert real & complex_arm == set()


def test_the_cell_has_no_attrition_and_the_funnel_says_so():
    """curl -> constitutive -> no magnetic source, all three at 3. Re-derived here."""
    rows = census_rows()
    if len(rows) != 186:
        pytest.skip(f"the census record is not present ({len(rows)} rows)")
    curl = [r for r in rows if covered(r, "cylindrical_curl@step_B")]
    both = [r for r in curl if covered(r, "cylindrical_constitutive@update_H")]
    clear = [r for r in both
             if not any(str(kind) == "B"
                        for kind in (r["configuration"].get("source_field_types")
                                     or []))]
    assert (len(curl), len(both), len(clear)) == (3, 3, 3)
    # Every one declares an ELECTRIC source and a metallic z wall — which is why the
    # source clause costs nothing here and the wall clear is not vacuous.
    for row in clear:
        configuration = row["configuration"]
        assert configuration["source_field_types"] == ["D"], configuration
        assert configuration["metallic"] == [False, False, True], configuration
        assert configuration["mirrored"] == [False, False, False], configuration


# ---------------------------------------------------------------------------
# What the product replaces, and what it does not
# ---------------------------------------------------------------------------

def test_REPLACES_names_three_passes_and_INERT_PASSES_names_the_other_two():
    gate = load_gate()
    assert tuple(product.REPLACES) == ("step_B", "zero_metal_B", "update_H")
    assert set(product.INERT_PASSES) == {"fill_symmetry_bc_B",
                                         "fill_folded_far_ghosts_B"}
    assert set(product.REPLACES) | set(product.INERT_PASSES) == set(gate.SEAM_PASSES)


def test_the_two_omitted_passes_are_inert_on_every_admitted_grid():
    """EXECUTED, not read off another module's guard.

    The gate's own leg runs both passes on a randomised state of an admitted Dcyl
    grid and requires that not one word moves — with ``zero_metal_B`` as the control
    that must move something, so an inert pass cannot be confused with a frozen
    state.
    """
    row = load_gate().inert_passes_leg()
    assert row["findings"] == [], row["findings"]
    assert row["rows"], "the leg measured no configuration at all"
    for case in row["rows"]:
        assert case["moved_by_the_inert_passes"] == [], case
        assert case["moved_by_zero_metal_B"], case


def test_the_wall_clear_is_not_vacuous_on_an_admitted_grid():
    """``ZM_Z`` fires on every admitted Dcyl grid: z is metallic or the curl refuses."""
    fields, _pml = build()
    assert zero_metal_axes(fields.grid) == (False, False, True)
    # The component the clear touches is the one whose z Yee shift is 0.
    assert IYEE_SHIFTS["Bz"][2] == 0
    assert IYEE_SHIFTS["Bx"][2] == 1 and IYEE_SHIFTS["By"][2] == 1


def test_ZM_X_and_ZM_Y_are_structurally_false_on_every_admitted_configuration():
    """Which is what licenses the mutation table to arm ``ZM_Z`` alone."""
    row = load_gate().structural_walls_leg()
    assert row["findings"] == [], row["findings"]
    assert row["admitted"] > 0
    for case in row["rows"]:
        if case["admitted"]:
            assert case["walls"] == [False, False, True], case
    assert row["phi_pec_refused_by_grid"]["refused"] is True


# ---------------------------------------------------------------------------
# The predicate, clause by clause
# ---------------------------------------------------------------------------

def test_a_dcyl_m0_grid_is_admitted_modulo_the_numpy_host():
    fields, pml = build()
    verdict = product.cylindrical_real_fused_magnetic_pair_coverage(fields, pml, ())
    assert residual(verdict) == []
    assert any("cupy" in reason for reason in verdict.reasons)


def test_an_electric_source_is_admitted_and_a_magnetic_one_is_refused_by_name():
    fields, pml = build()
    assert residual(product.cylindrical_real_fused_magnetic_pair_coverage(
        fields, pml, (Source("D"),))) == []
    refused = residual(product.cylindrical_real_fused_magnetic_pair_coverage(
        fields, pml, (Source("B"),)))
    assert any("is magnetic" in reason for reason in refused)
    assert any("driver.py:3283" in reason for reason in refused)


def test_an_undeclared_source_list_is_a_refusal_and_not_an_empty_set():
    fields, pml = build()
    refused = residual(
        product.cylindrical_real_fused_magnetic_pair_coverage(fields, pml, None))
    assert any("was not declared" in reason for reason in refused)
    assert product.plan_cylindrical_real_fused_magnetic_pair(fields, pml, None) is None


def test_m_not_zero_is_refused_by_name():
    fields, pml = build(m=1, force_complex_fields=True)
    refused = residual(
        product.cylindrical_real_fused_magnetic_pair_coverage(fields, pml, ()))
    assert any("carries m = 0" in reason for reason in refused)


def test_a_cartesian_grid_is_refused_by_name():
    # z stays PERIODIC: a 2-D cell is invariant along it and Grid refuses a PEC
    # there by name (grid.py:842-877).
    grid = Grid(resolution=10.0, cell_size=(1.6, 1.6, 0.0), dimensions=2,
                boundaries={"x": "metallic", "y": "metallic", "z": "periodic"})
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    refused = residual(product.cylindrical_real_fused_magnetic_pair_coverage(
        fields, PML(grid=grid, thickness=0.2), ()))
    assert any("not cylindrical" in reason for reason in refused)


def test_no_pml_is_refused_because_update_H_is_a_no_op_without_one():
    fields, _pml = build()
    refused = residual(
        product.cylindrical_real_fused_magnetic_pair_coverage(fields, None, ()))
    assert any("PML" in reason for reason in refused)


def test_the_predicate_reports_both_halves_reasons_with_their_side_named():
    grid = Grid(resolution=10.0, cell_size=(1.6, 1.6, 0.0), dimensions=2,
                boundaries={"x": "metallic", "y": "metallic", "z": "periodic"})
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    reasons = residual(product.cylindrical_real_fused_magnetic_pair_coverage(
        fields, PML(grid=grid, thickness=0.2), ()))
    assert any(reason.startswith("cylindrical curl half: ") for reason in reasons)
    assert any(reason.startswith("cylindrical constitutive half: ")
               for reason in reasons)


# ---------------------------------------------------------------------------
# The kernel's compile-time bindings
# ---------------------------------------------------------------------------

def test_the_backward_constexpr_is_step_Bs_and_only_step_Bs():
    assert product.BACKWARD == cyl.SUB_STEPS[product.CURL_SUB_STEP]["backward"] == 0
    assert product.CURL_SUB_STEP == "step_B"
    assert product.CONSTITUTIVE_SIDE == "H"


def test_the_plan_refuses_a_grid_that_stores_more_than_one_phi_cell():
    """A Dcyl grid stores one phi cell; the plan is the last gate before a compile."""
    kwargs = dict(dtdx=0.5, zero_metal=(False, False, True), block=256,
                  targets=[], auxiliaries=[], sources=[], curl_coefficients=[],
                  h_targets=[], h_aux=[], h_coefficients=[], xp=None)
    with pytest.raises(ValueError, match="one phi cell"):
        product.CylindricalRealFusedMagneticPairPlan(shape=(8, 4, 8), **kwargs)


def test_the_builder_refuses_every_configuration_the_predicate_refuses():
    for kwargs, sources in (({"m": 1, "force_complex_fields": True}, ()),
                            ({}, (Source("B"),)),
                            ({}, None)):
        fields, pml = build(**kwargs)
        assert not product.cylindrical_real_fused_magnetic_pair_coverage(
            fields, pml, sources).covered
        assert product.plan_cylindrical_real_fused_magnetic_pair(
            fields, pml, sources) is None


# ---------------------------------------------------------------------------
# The optional dependency stays optional
# ---------------------------------------------------------------------------

def test_the_module_imports_and_answers_coverage_with_triton_absent(monkeypatch):
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "triton" or name.startswith("triton."):
            raise ImportError("triton is not installed (simulated)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    for name in list(sys.modules):
        if name == "triton" or name.startswith("triton."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.delitem(
        sys.modules, "meep_gpu.triton_kernels.cylindrical_real_fused_magnetic_pair",
        raising=False)
    module = importlib.import_module(
        "meep_gpu.triton_kernels.cylindrical_real_fused_magnetic_pair")
    assert module.triton is None
    assert module.cyl_real_fused_curl_constitutive_B is None
    fields, pml = build()
    assert module.cylindrical_real_fused_magnetic_pair_coverage(fields, pml, ()).reasons
    assert module.plan_cylindrical_real_fused_magnetic_pair(fields, pml, ()) is None
    with pytest.raises(ImportError, match="triton"):
        module.cyl_real_fused_curl_constitutive_B_kernel()


# ---------------------------------------------------------------------------
# File ownership and the deferral
# ---------------------------------------------------------------------------

def code_of(path: pathlib.Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)) and ast.get_docstring(node):
            node.body = node.body[1:]
    return ast.unparse(tree)


def test_this_module_touches_neither_dispatch_nor_the_other_track():
    code = code_of(PACKAGE_DIR / "cylindrical_real_fused_magnetic_pair.py")
    assert "fastpath" not in code
    assert "cuda_kernels" not in code
    assert "options=" not in code


def test_the_single_launch_site_passes_the_packages_shared_guard_constant():
    source = (PACKAGE_DIR
              / "cylindrical_real_fused_magnetic_pair.py").read_text(encoding="utf-8")
    spelling = "enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard)"
    assert source.count("enable_fp_fusion=") == 1
    assert spelling in source


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
    electric sibling — until it existed no Dcyl shape had ever reached the driver
    seam, which is why this product waited on a new case rather than on a new
    measurement of an old one. The fourteen sibling products that were NOT driven
    keep the old name and the old assertion, which is what makes the two states
    legible per file.

    THIS ONE WAS ALREADY ON THE CERTIFIED SIDE OF THE PARTITION, which the other
    three released with it were not: its ledger entry
    (``triton_cylindrical_real_fused_magnetic_pair_device_gate``) was cut on
    2026-08-20, long before any case drove it, so the release is the only half
    that moved here. ``rebind_triton_welds.py`` re-pins that entry after the
    campaign because this very file is one of the sources it names.
    """
    import pathlib as _pathlib  # noqa: PLC0415

    from meep_gpu import fastpath as _fastpath  # noqa: PLC0415
    from meep_gpu.triton_kernels import launch as _launch  # noqa: PLC0415

    source = _pathlib.Path(_launch.__file__).read_text(encoding="utf-8")
    assert "plan_cylindrical_real_fused_magnetic_pair" in source
    row = _launch.CERTIFIED_FUSED_PRODUCTS["cylindrical_real_fused_magnetic_pair"]
    assert row["module"] == "cylindrical_real_fused_magnetic_pair"
    assert row["builder"] == "plan_cylindrical_real_fused_magnetic_pair"
    label = row["label"]
    assert label == 'fused pair B (cylindrical)'
    assert _fastpath.arm_is_fused(label)
    assert label in _fastpath.RELEASED_FUSED_ARMS
    assert _fastpath.RELEASED_FUSED_ARMS[label] == ("cylindrical_m0",)
    assert label in _fastpath.ARM_CERTIFICATION
    assert label not in _fastpath.PENDING_DEVICE_GATE_ARMS
    assert label in _fastpath.FUSED_ARM_CONSTITUENTS
def test_the_gate_exists_names_the_product_and_reports_the_run_it_took():
    source = GATE.read_text(encoding="utf-8")
    assert "plan_cylindrical_real_fused_magnetic_pair" in source
    assert "cyl_real_fused_curl_constitutive_B" in source
    assert "RELEASED 2026-08-20" in source


def test_the_module_claims_identity_ONLY_through_the_weld_that_measured_it():
    """A weld licenses the claim; a docstring does not.

    An identity claim in this module has to be the one a weld records, and the weld
    has to be the one whose digests match THIS file as it sits on disk — otherwise
    the claim describes bytes nobody measured.
    """
    import hashlib

    source = (PACKAGE_DIR
              / "cylindrical_real_fused_magnetic_pair.py").read_text(encoding="utf-8")
    record = json.loads(
        (PACKAGE_DIR / "fingerprints.json").read_text(encoding="utf-8"))
    weld = record["triton_cylindrical_real_fused_magnetic_pair_device_gate"]
    assert weld["status"] == "PASS"
    assert "RELEASED 2026-08-20" in source

    named = {name: digest for name, digest in weld["source_sha256"].items()
             if name.endswith("triton_kernels/cylindrical_real_fused_magnetic_pair.py")}
    assert named, "the weld does not name the module it certifies"
    for name, digest in named.items():
        live = hashlib.sha256(pathlib.Path(API_ROOT / name).read_bytes()).hexdigest()
        assert live == digest, (
            f"{name} has changed since the gate ran; the identity claim in its "
            f"docstring no longer describes the bytes that were measured")
    assert "NOT WIRED" in source, (
        "a released gate is not a wiring; the module must still say so")


def test_the_released_artifact_is_in_the_tree_and_says_it_released():
    artifact = (API_ROOT / "parity" / "meep_gpu" / "results"
                / "triton_cylindrical_real_fused_magnetic_pair_2026-08-20"
                / "gate.json")
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["release"]["released"] is True, payload["release"]
    assert payload["device_status"] == "RUN"
    assert len(payload["device_legs"]) == 6
    assert all(row["passed"] for row in payload["device_legs"])
    caught = {row["mutation"]: row["caught"] for row in payload["mutations"]}
    assert len(caught) == 10
    assert sum(caught.values()) == 9, caught
    assert caught["m10_commuted_multiply"] is False


# ---------------------------------------------------------------------------
# The gate's own no-device legs, executed here
# ---------------------------------------------------------------------------

def test_the_gates_transcription_leg_traces_every_arithmetic_line_to_its_source():
    row = load_gate().transcription_leg()
    assert row["findings"] == [], row["findings"]
    assert row["curl_lines_checked"] >= 13
    assert row["zero_metal_lines_checked"] == 3


def test_the_curls_ownership_mask_is_observable_in_the_gates_own_fixture():
    """A mask whose value is already present measures nothing when it is dropped.

    Measured on the first device run of this gate: with the ELECTRIC constitutive
    history left at zero, ``Ep`` is exactly ``+0.0`` on both masked planes forever and
    ``m5_ownership_mask_dropped`` came back UNCAUGHT. The fixture now seeds it, and
    this contract is what stops a future harness from quietly un-seeding it.
    """
    row = load_gate().mask_observability_leg()
    assert row["findings"] == [], row["findings"]
    seeded = [entry for entry in row["rows"]
              if entry["arm"] == "electric_history_seeded"]
    control = [entry for entry in row["rows"]
               if entry["arm"] == "electric_history_zero"]
    assert seeded and control
    for entry in seeded:
        assert entry["nonzero_words_on_the_r0_row"] > 0, entry
        assert entry["nonzero_words_on_the_z0_plane"] > 0, entry
    for entry in control:
        assert entry["nonzero_words_on_the_r0_row"] == 0, entry
        assert entry["nonzero_words_on_the_z0_plane"] == 0, entry


def test_the_gates_design_sweep_agrees_with_the_array_path_and_catches_its_flips():
    row = load_gate().design_sweep_leg()
    assert row["findings"] == [], row["findings"]
    verdicts = {(entry["case"], entry["knob"]): entry["identical"]
                for entry in row["rows"]}
    for case, _cell, _res, _courant, _steps in load_gate().CASES:
        assert verdicts[(case, "faithful")] is True
        assert verdicts[(case, "drop_the_wall_clear")] is False
        assert verdicts[(case, "constitutive_before_the_wall")] is False


def test_the_gates_predicate_leg_exercises_every_clause_in_both_directions():
    row = load_gate().predicate_leg()
    assert row["findings"] == [], row["findings"]
    directions = {entry["case"]: entry["expect_admitted"] for entry in row["rows"]}
    assert directions["dcyl_m0_electric_source"] is True
    assert directions["dcyl_m0_magnetic_source"] is False


def test_the_gates_corpus_leg_agrees_with_the_metal_censuss_own_row_set():
    row = load_gate().corpus_admission_leg()
    if row.get("seam_instances_gained") is None:
        pytest.skip(f"a census record is absent on this host: {row['findings']}")
    assert row["findings"] == [], row["findings"]
    assert row["triton"]["admitted_rows"] == row["metal"]["admitted_rows"]
    assert row["seam_instances_gained"] == 3


def test_the_gate_arms_every_mutation_it_declares_and_none_on_a_dead_branch():
    """A rewrite that matches nothing reports a defect as uncaught while measuring
    nothing; a rewrite under a false constexpr does the same, more quietly."""
    gate = load_gate()
    text = gate._shipped_text("cyl_real_fused_curl_constitutive_B")
    for name, _why, _expectation, rewrite in gate.mutation_table():
        mutated, hits = rewrite(text)
        assert hits > 0, name
        assert mutated != text, name
    assert gate._assert_no_dead_branch_rewrites(text) == []


def test_the_gate_binds_the_seam_passes_the_driver_actually_calls():
    gate = load_gate()
    driver_source = (PACKAGE_DIR.parent / "driver.py").read_text(encoding="utf-8")
    for name in gate.SEAM_PASSES:
        assert f"{name}(self.fields" in driver_source, name


def test_the_gates_material_names_are_the_ones_the_scan_really_finds():
    gate = load_gate()
    fields, _pml = build()
    found = {name for name, value in vars(fields).items()
             if getattr(value, "shape", None) == tuple(fields.grid.shape)}
    for name in gate.MATERIAL:
        assert name in found, (name, sorted(found))
