"""The host half of the four Metal residue welds' certification.

THE PRODUCTS close what the 2026-09-01 board left: the two CANNOT-BIND cells
(``bfast_fused_electric_pair`` at 33 unpacked pointers, ``conductive_fused_electric_
pair`` at 39 — both rebound under the ceiling by :mod:`.metal_kernels
.coefficient_pack`) and the two seams of the one slot-UNSELECTED row
(``beta_complex_fused_electric_pair`` / ``beta_complex_fused_magnetic_pair``, the
special_kz complex-beta welds whose constitutive arm landed 2026-08-19). The bytes
are the gate's (``parity/meep_gpu/gate_metal_residue_fused_pairs.py``, on this
Mac's MPS); what is here is everything answerable without a device run and
everything that would be a silent widening if it drifted: the declarations, the
binding counts against the emitted text, the pack cross-checks, the census-measured
CARRIES flags, the predicate in both directions, the absorb rows, and the
registration shape.
"""

from __future__ import annotations

import importlib
import json
import pathlib
import re
import sys

import pytest

from . import deposit_repair
from .metal_kernels import launch as metal_launch

HERE = pathlib.Path(__file__).parent
PACKAGE_DIR = HERE / "metal_kernels"
PARITY = HERE.parent / "parity" / "meep_gpu"
GATE = PARITY / "gate_metal_residue_fused_pairs.py"
CENSUS = PARITY / "results" / "metal_coverage_tranche6_2026-08-19"
RECLOSE = (PARITY / "results" / "metal_coverage_special_kz_reclose_2026-08-19"
           / "smoke" / "test_special_kz.json")

FAMILIES = ("bfast_fused_electric_pair", "conductive_fused_electric_pair",
            "beta_complex_fused_electric_pair", "beta_complex_fused_magnetic_pair")

#: family -> (corpus row, seam pair label, in-seam source field types) — the
#: census facts each flag is measured against below.
CELLS = {
    "bfast_fused_electric_pair":
        ("TestReflectanceAngular.test_reflectance_angular_2_35_7", "D", ["D"]),
    "conductive_fused_electric_pair":
        ("TestAdjointSolver.test_damping", "D", ["D", "B"]),
    "beta_complex_fused_electric_pair":
        ("TestSpecialKz.test_special_kz", "D", ["D"]),
    "beta_complex_fused_magnetic_pair":
        ("TestSpecialKz.test_special_kz", "B", ["D"]),
}

_BINDING = re.compile(r"\[\[buffer\((\d+)\)\]\]")
_POINTER = re.compile(r"device\s+(?:const\s+)?[\w:]+\s*\*\s*\w+\s*\[\[buffer")


@pytest.fixture(scope="module", params=FAMILIES)
def product(request):
    return importlib.import_module(f"meep_gpu.metal_kernels.{request.param}")


@pytest.fixture(scope="module")
def matrix():
    if str(PARITY) not in sys.path:
        sys.path.insert(0, str(PARITY))
    module = importlib.import_module("metal_composition_matrix")
    module.prepare_environment()
    return module


def _census_row(name):
    for stem in ("tests_param_matched", "tests"):
        path = CENSUS / f"{stem}.jsonl"
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("row") == name and row.get("measured"):
                return row
    raise AssertionError(f"census carries no measured row {name!r}")


def _emit(product):
    """One representative shipped source per family."""
    name = product.FAMILY
    if name == "bfast_fused_electric_pair":
        return product.bfast_fused_electric_pair_source(
            (0, 0, 1), (False, False, True))
    if name == "conductive_fused_electric_pair":
        return product.conductive_fused_electric_pair_source(
            (0, 0, 1), (True, True, True), (False, False, True))
    if name == "beta_complex_fused_electric_pair":
        return product.beta_complex_fused_electric_pair_source(
            (0, 0, 0), (1, 0, 0), (False, False, False), "FMA_V1")
    return product.beta_complex_fused_magnetic_pair_source(
        (0, 0, 0), (1, 0, 0), (False, False, False), "FMA_V1")


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------

def test_the_slot_and_replaces_are_the_seams_own(product):
    if product.FAMILY == "beta_complex_fused_magnetic_pair":
        assert product.SLOT == "step_B"
        assert product.REPLACES == ("step_B", "zero_metal_B", "update_H")
    else:
        assert product.SLOT == "step_D"
        assert product.REPLACES == ("step_D", "zero_metal_D", "update_E")


def test_the_carries_flag_is_the_census_measurement(product):
    """CARRIES_DEPOSIT_REPAIR measured off the census, per the campaign rule: True
    exactly where the family's own seam holds a deposit on its corpus row."""
    row_name, pair, source_types = CELLS[product.FAMILY]
    census = _census_row(row_name)
    assert census["configuration"]["source_field_types"] == source_types
    seam_has_deposit = (("B" in source_types) if pair == "B"
                        else any(k != "B" for k in source_types))
    assert product.CARRIES_DEPOSIT_REPAIR is seam_has_deposit, (
        f"{product.FAMILY}: the census row {row_name} gives its {pair} seam "
        f"{'a deposit' if seam_has_deposit else 'no deposit'}; the flag must be "
        f"the measurement")


def test_the_flag_reaches_the_seam_clause_by_name(product):
    text = (PACKAGE_DIR / f"{product.FAMILY}.py").read_text(encoding="utf-8")
    assert "carries_repair=CARRIES_DEPOSIT_REPAIR" in text
    # None of the four declares a repair path of its own: three ride the
    # split-field default (their absorbers are ACTIVE) and the fourth carries no
    # deposit at all — a PLAIN_PATH declaration here would be a wrong claim.
    assert "REPAIR_PATHS" not in text


def test_the_absorb_rows_name_the_two_measured_arms():
    assert metal_launch.FUSED_PAIR_ARMS["bfast_fused_electric_pair"] == (
        "BFAST", "BFAST")
    assert metal_launch.FUSED_PAIR_ARMS["conductive_fused_electric_pair"] == (
        "conductive PML curl", "ordinary")
    assert metal_launch.FUSED_PAIR_ARMS["beta_complex_fused_electric_pair"] == (
        "special_kz complex beta", "special_kz complex beta")
    assert metal_launch.FUSED_PAIR_ARMS["beta_complex_fused_magnetic_pair"] == (
        "special_kz complex beta", "special_kz complex beta")


def test_the_products_register_unwired_on_their_slot(product):
    from .metal_kernels import arms

    spec = next(spec for spec in arms.registered(product.SLOT)
                if spec.family == product.FAMILY)
    assert spec.wired is False
    assert spec.is_weld
    assert spec.replaces == product.REPLACES


# ---------------------------------------------------------------------------
# The binding counts, against the emitted text
# ---------------------------------------------------------------------------

def test_the_shipped_signature_counts_are_the_emitted_ones(product):
    source = _emit(product)
    slots = sorted({int(number) for number in _BINDING.findall(source)})
    pointers = len(_POINTER.findall(source))
    assert len(slots) == product.PACKED_BINDINGS, (len(slots), pointers)
    assert slots == list(range(len(slots)))
    if hasattr(product, "PACKED_POINTERS"):
        assert pointers == product.PACKED_POINTERS


def test_the_refuted_signatures_count_what_the_module_claims(product):
    refuted = {"separate": product.refuted_separate_scalar_source()}
    expected = {"separate": product.SEPARATE_SCALAR_BINDINGS}
    if hasattr(product, "refuted_unpacked_pointer_source"):
        refuted["unpacked"] = product.refuted_unpacked_pointer_source()
        expected["unpacked"] = product.UNPACKED_POINTER_BINDINGS
    for label, source in refuted.items():
        slots = sorted({int(number) for number in _BINDING.findall(source)})
        assert len(slots) == expected[label], (label, len(slots))


@pytest.mark.parametrize("family", ["bfast_fused_electric_pair",
                                    "conductive_fused_electric_pair"])
def test_the_pack_declaration_is_arithmetically_sound(family):
    """Packing N vectors into one buffer saves exactly N - 1 pointers — the same
    cross-check the board's calibration applies before adding the savings back.
    Parametrised over the two PACKED families by name: the beta twins carry no
    pack and a skip here would be a silent coverage gap."""
    product = importlib.import_module(f"meep_gpu.metal_kernels.{family}")
    saved = product.UNPACKED_POINTERS - product.PACKED_POINTERS
    assert saved == len(product.PACKED_VECTORS) - 1
    # The pack's members are disjointly named and the struct fields and prologue
    # are generated from the SAME tuple.
    assert len(set(product.PACKED_VECTORS)) == len(product.PACKED_VECTORS)
    source = _emit(product)
    for name in product.PACKED_VECTORS:
        assert f"off_{name};" in source
        assert re.search(rf"device const float\* {name} = ", source), name


@pytest.mark.parametrize("family", ["beta_complex_fused_electric_pair",
                                    "beta_complex_fused_magnetic_pair"])
def test_the_beta_params_record_matches_the_metal_struct(family):
    """The 64-byte host record: five float2 first, then four uints and dtdx —
    offsets stated so the record cannot drift into agreeing by accident.
    Parametrised over the two beta twins by name: the packed families carry
    uint/float structs whose layouts their own plan builders assert."""
    product = importlib.import_module(f"meep_gpu.metal_kernels.{family}")
    dtype = product.params_record_dtype()
    assert dtype.itemsize == product.PARAMS_ITEMSIZE == 64
    offsets = {name: dtype.fields[name][1] for name in dtype.names}
    assert offsets == {"px": 0, "py": 8, "pz": 16, "bp": 24, "bm": 32,
                       "nx": 40, "ny": 44, "nz": 48, "n_elem": 52, "dtdx": 56}


# ---------------------------------------------------------------------------
# The predicate, in both directions, and the composed bracket
# ---------------------------------------------------------------------------

class _Electric:
    field_type = "D"
    component = "Ez"
    is_integrated = False
    import numpy as _np
    _point_ix = _np.array([1])
    _point_iy = _np.array([1])
    _point_iz = _np.array([0])


def _fixture_for(product, matrix):
    if product.FAMILY == "bfast_fused_electric_pair":
        return matrix.cart(bfast_scaled_k=(0.2, 0.0, 0.0))
    if product.FAMILY == "conductive_fused_electric_pair":
        return matrix.conductive(matrix.cart())
    return matrix.flat(beta=0.33, complex_storage=True)


def _coverage(product, fields, pml, sources):
    from .metal_kernels.device import Residency

    return getattr(product, f"metal_{product.FAMILY}_coverage")(
        fields, pml, sources, Residency())


def test_the_predicate_admits_its_own_cell(product, matrix):
    fields, pml = _fixture_for(product, matrix)
    sources = (() if product.FAMILY == "beta_complex_fused_magnetic_pair"
               else (_Electric(),))
    verdict = _coverage(product, fields, pml, sources)
    assert verdict.covered, verdict.reasons[:6]


def test_the_predicate_refuses_a_foreign_cell_with_each_halfs_reason(
        product, matrix):
    fields, pml = matrix.cart()  # plain PML, no bfast, no beta, no sigma
    verdict = _coverage(product, fields, pml, ())
    assert not verdict.covered
    assert any(reason.startswith(("curl half:", "constitutive half:"))
               for reason in verdict.reasons), verdict.reasons[:4]


def test_an_undeclared_source_list_is_refused_not_read_as_empty(product, matrix):
    fields, pml = _fixture_for(product, matrix)
    verdict = _coverage(product, fields, pml, None)
    assert any("was not declared" in reason for reason in verdict.reasons), (
        verdict.reasons[:4])


def test_the_composed_seam_installs_the_pair(product, matrix):
    """``plan_step(..., fuse=True)`` on the family's own fixture must land this
    product in its two slots — bracketed exactly where its seam holds a deposit."""
    from .metal_kernels import device
    from .sources import ContinuousEnvelope, VolumeSource

    fields, pml = _fixture_for(product, matrix)
    if product.FAMILY == "beta_complex_fused_magnetic_pair":
        sources = ()
    else:
        sources = (VolumeSource(
            grid=fields.grid, component="Ez", center=(0.0, 0.0, 0.0),
            size=(0.0, 0.0, 0.0),
            envelope=ContinuousEnvelope(frequency=1.0, is_integrated=False)),)
    plan = metal_launch.plan_step(fields, pml, residency=device.Residency(),
                                  sources=sources, fuse=True)
    reasons = plan.reasons.get(f"fused_pair_{product.FAMILY}", ())
    assert not reasons, reasons
    curl_slot, _wall, const_slot = product.REPLACES
    leading = plan.plans[curl_slot]
    trailing = plan.plans[const_slot]
    if sources:
        assert isinstance(leading, deposit_repair.LeadingRepairPlan)
        assert isinstance(trailing, deposit_repair.TrailingRepairPlan)
        assert tuple(leading.repair_paths) == (deposit_repair.SPLIT_FIELD_PATH,)
    else:
        assert type(trailing).__name__ == "NoopPlan"
    inner = metal_launch.declaring_plan(leading)
    assert inner.replaces_sub_steps == product.REPLACES


def test_the_conductive_pair_consults_the_one_home_rescale_clause(matrix):
    """The lifted clause is IMPORTED from the no-PML conductive module — one home
    — and still refuses the table-less fallback by name on this family too."""
    conductive = importlib.import_module(
        "meep_gpu.metal_kernels.conductive_fused_electric_pair")

    class _TableLess:
        field_type = "D"
        component = "Ez"
        is_integrated = False

    fields, pml = matrix.conductive(matrix.cart())
    found = _coverage(conductive, fields, pml, (_TableLess(),))
    assert any("publishes NO deposit table" in reason for reason in found.reasons)
    text = (PACKAGE_DIR / "conductive_fused_electric_pair.py").read_text("utf-8")
    assert "from .no_pml_conductive_fused_electric_pair import" in text


# ---------------------------------------------------------------------------
# The special_kz cell's stale-census correction, pinned at its source
# ---------------------------------------------------------------------------

def test_the_reclose_record_carries_the_selections_the_board_substitutes():
    """The board's `_apply_reclose` substitutes exactly one row; the artifact it
    reads must keep saying what the substitution assumes."""
    entries = json.loads(RECLOSE.read_text())
    row = next(e for e in entries if e["row"] == "TestSpecialKz.test_special_kz")
    selected = (row.get("plan_step") or {}).get("selected", {})
    assert selected.get("step_B") == "special_kz complex beta"
    assert selected.get("update_H") == "special_kz complex beta"
    assert selected.get("step_D") == "special_kz complex beta"
    assert selected.get("update_E") == "special_kz complex beta"
    for key in ("beta_run_complex_constitutive@update_H",
                "beta_run_complex_constitutive@update_E"):
        assert row["predicates"][key]["covered"] is True, key


# ---------------------------------------------------------------------------
# The gate exists and measures what these tests only declare
# ---------------------------------------------------------------------------

def test_the_device_gate_measures_the_ceilings_and_the_welds():
    source = GATE.read_text(encoding="utf-8")
    for family in FAMILIES:
        assert family in source
    for anchor in ("leg_binding_ceiling", "leg_identity", "leg_mutation",
                   "leg_deposit", "refuted_unpacked_pointer_source",
                   "bisect", "run_current_measurement"):
        assert anchor in source, anchor
