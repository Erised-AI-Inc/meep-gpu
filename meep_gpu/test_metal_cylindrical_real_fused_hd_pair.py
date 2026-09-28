"""The Metal Dcyl m = 0 fused H->D pair, as claims a laptop can check.

WHAT THIS SUITE OWNS AND WHAT IT DELIBERATELY DOES NOT. The BYTES are the device gate's
-- ``parity/meep_gpu/gate_metal_cylindrical_real_fused_hd_pair.py`` walks six engines
through complete driver steps and compares uint32 words. No assertion here duplicates
that. What lives here is everything true about this family WITHOUT a device:

* that launch 1's source is the certified ``update_H`` cell function plus the certified
  radial scan with EXACTLY the two declared edits (the two ``src[...]`` reads become the
  recompute), that it reads nothing it writes, and that launch 2 IS the certified
  cylindrical curl byte for byte;
* that the scan's divide is spelled ``/`` and the field sits LEFT of the weight -- the
  real family's spelling, which the complex family must INVERT;
* that the binding counts the module spells are the counts its emitter produces, and
  that the headroom it declares is the ceiling minus that count;
* that the arm is registered UNWIRED on ``update_H`` replacing both seam slots, holds no
  absorb row and no registry row yet, declares itself uninstallable and declares its
  weld owed with the artifact named -- the partition ``test_metal_weld_contract`` reads;
* that the predicate refuses what it must refuse, by name;
* that its cell is the three corpus rows the census selects ``cylindrical m=0`` on both
  seam slots, measured off the census rather than argued from the names.

THE CORPUS DEMAND: 3 H->D seam-instances (``tests:TestAdjointSolver.test_adjoint_solver_
cyl_n2f_fields_0_0``, ``..._1_0``, ``tests:TestPMLCylindrical.test_pml_cyl_0_0_0``), the
whole ``cylindrical m=0 -> cylindrical m=0`` cell on every board, each with an electric
source whose withdraw is the no-op.
"""

from __future__ import annotations

import ast
import json
import os
import pathlib
import re

import numpy as np
import pytest

# The policy this family is certified under; `setdefault` so a campaign export wins.
# See test_metal_cylindrical_real_fused_electric_pair.py for why this must be set at
# module scope rather than inherited from a sibling's import order.
os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")

PACKAGE_DIR = pathlib.Path(__file__).resolve().parent
REPO = PACKAGE_DIR.parent
GATE = REPO / "parity" / "meep_gpu" / "gate_metal_cylindrical_real_fused_hd_pair.py"
CENSUS = REPO / "parity" / "meep_gpu" / "results" / "metal_coverage_2026-09-04_m0complex"
SEAM_RECORD = (REPO / "parity" / "meep_gpu" / "results" / "h_to_d_seam_2026-09-04"
               / "h_to_d_seam.jsonl")

from meep_gpu import withdraw_hoist  # noqa: E402
from meep_gpu.metal_kernels import (  # noqa: E402
    arms,
    cylindrical_real as cyl,
    cylindrical_real_fused_hd_pair as family,
    fused_hd_pair as cartesian,
    launch as metal_launch,
    registry,
)
from meep_gpu.metal_kernels.device import MAX_BUFFER_BINDINGS, Residency  # noqa: E402

CODES_METALLIC_Z = (cyl.METALLIC, cyl.PERIODIC, cyl.METALLIC)
CODES_PERIODIC_Z = (cyl.METALLIC, cyl.PERIODIC, cyl.PERIODIC)


def emit():
    return family.cylindrical_real_fused_hd_pair_constitutive_source()


def build(z_kind: str = "metallic", shape=(20, 1, 20), courant: float = 0.5,
          m: int = 0, complex_storage: bool = False):
    """A real Dcyl Grid/Fields/PML triple on NumPy, with the PML storage allocated."""
    from meep_gpu.fields import Fields
    from meep_gpu.grid import Grid
    from meep_gpu.pml import PML

    grid = Grid(resolution=1.0, cell_size=(float(shape[0]), 0.0, float(shape[2])),
                cylindrical=True, m=m, boundaries={"z": z_kind},
                courant=float(courant), xp=np)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.enable_pml_storage()
    return fields, PML(grid=grid, thickness={"x": (0, max(2, shape[0] // 4)),
                                             "z": max(2, shape[2] // 4)})


def _source(fields, component: str, integrated: bool):
    from meep_gpu.sources import GaussianEnvelope, VolumeSource

    return VolumeSource(grid=fields.grid, component=component,
                        center=(0.5, 0.0, 0.5), size=(0.0, 0.0, 0.0),
                        envelope=GaussianEnvelope(frequency=1.0, fwidth=0.5,
                                                  is_integrated=integrated))


# ---------------------------------------------------------------------------
# The construction: launch 1 is lifted certified text, launch 2 is the certified curl
# ---------------------------------------------------------------------------

def test_launch_1_carries_the_certified_update_H_cell_function_verbatim():
    assert cartesian.h_cell_function() in emit()


def test_the_scan_is_the_certified_text_with_exactly_the_two_declared_edits():
    certified = cyl.cylindrical_prefix_source("step_D")
    tail = certified.split("    int base = j * nzi + k;\n", 1)[1][: -len("}\n")]
    lifted = family.certified_scan_tail()
    a, b = tail.splitlines(), lifted.splitlines()
    assert len(a) == len(b)
    changed = [(x, y) for x, y in zip(a, b) if x != y]
    assert changed == [(e["line"], e["became"]) for e in family.SCAN_LIFT_EDITS]
    assert len(changed) == 2
    assert lifted in emit()


def test_the_divide_is_spelled_slash_and_the_field_sits_left_of_the_weight():
    """The REAL family's spelling. The complex family must spell the reciprocal
    multiply instead -- the inversion the adjudication measured at ~25% of words."""
    source = emit()
    assert "float inc = (w - prev) / divisor[i - 1];" in source
    # Executable text only: the certified scan's own COMMENT names the refuted
    # spelling, which is how a reader learns it is refuted.
    code = "\n".join(line for line in source.splitlines()
                     if not line.strip().startswith("//"))
    assert "(1.0f / divisor" not in code
    assert f"h_cell(i, j, k, {cartesian.H_CELL_TAIL_ARGS}).a1 * weights[i];" in source
    assert "acc = acc + inc;" in source
    assert "float acc = 0.0f;" in source


def test_launch_1_reads_nothing_it_writes():
    """Every ``ho*``/``wo*`` occurrence is a store of the thread's OWN cell; no ``src``."""
    source = emit()
    for line in source.splitlines():
        if not re.search(r"\b(ho|wo)[012]\[", line):
            continue
        for piece in line.split(";"):
            if piece.strip():
                assert re.match(r"^(ho|wo)[012]\[ii\] = own\.[a-z0-9]+$", piece.strip()), line
    assert "src[" not in source
    assert "ho1[i * nyz" not in source and "hi1[i * nyz" not in source


def test_the_leader_is_the_r_equals_zero_thread_of_each_column():
    source = emit()
    assert "if (i != 0) { return; }" in source
    assert source.index("ho0[ii] = own.a0;") < source.index("if (i != 0) { return; }")


def test_launch_2_is_the_certified_cylindrical_curl_byte_for_byte():
    for codes in (CODES_METALLIC_Z, CODES_PERIODIC_Z):
        assert (family.cylindrical_real_fused_hd_pair_curl_source(codes)
                == cyl.cylindrical_curl_source("step_D", codes))


def test_the_builder_compiles_launch_2_through_the_certified_entry_point():
    text = (PACKAGE_DIR / "metal_kernels" / "cylindrical_real_fused_hd_pair.py").read_text(
        encoding="utf-8")
    body = text.split("def _curl_functions(", 1)[1].split("\ndef ", 1)[0]
    assert "compile_cylindrical_curl(CURL_SUB_STEP, codes, mode)" in body


def test_a_periodic_r_axis_is_refused_by_name_before_any_splice():
    with pytest.raises(ValueError, match="must compile as METALLIC"):
        family.cylindrical_real_fused_hd_pair_curl_source((cyl.PERIODIC, cyl.PERIODIC,
                                                            cyl.METALLIC))


def test_the_ir0_is_the_certified_familys_step_D_value():
    assert family.IR0 == cyl.PREFIX_IR0["step_D"] == 0.5


# ---------------------------------------------------------------------------
# The signature
# ---------------------------------------------------------------------------

def test_the_binding_count_the_module_spells_is_the_count_its_emitter_produces():
    source = emit()
    signature = source.split("kernel void cyl_real_hd_constitutive_prefix(", 1)[1]
    signature = signature.split("uint idx [[thread_position_in_grid]])", 1)[0]
    assert signature.count("[[buffer(") == family.CONSTITUTIVE_BINDINGS == 25
    assert family.shipped_signature_bindings() == family.CONSTITUTIVE_BINDINGS
    pointers = [line for line in signature.splitlines() if "device" in line]
    assert len(pointers) == family.CONSTITUTIVE_POINTERS == 24
    assert family.HEADROOM == MAX_BUFFER_BINDINGS - family.CONSTITUTIVE_BINDINGS == 6


def test_the_refuted_and_fitting_signatures_count_as_declared():
    fitting = family.largest_fitting_source()
    refuted = family.refuted_over_the_ceiling_source()
    def bindings(text):
        signature = text.split("kernel void cyl_real_hd_constitutive_prefix(", 1)[1]
        return signature.split("uint idx [[thread_position_in_grid]])", 1)[0].count("[[buffer(")
    assert bindings(fitting) == MAX_BUFFER_BINDINGS
    assert bindings(refuted) == MAX_BUFFER_BINDINGS + 1


def test_launch_2_binds_the_certified_curls_twenty_two():
    source = family.cylindrical_real_fused_hd_pair_curl_source(CODES_METALLIC_Z)
    signature = source.split("kernel void cyl_pml_curl_step(", 1)[1]
    signature = signature.split("uint idx [[thread_position_in_grid]])", 1)[0]
    assert signature.count("[[buffer(") == family.CURL_BINDINGS == 22


def test_the_rotating_group_leads_the_signature_in_rotated_names_order():
    source = emit()
    names = [line.split("*")[1].split("[[")[0].strip()
             for line in source.splitlines() if "[[buffer(" in line and "device" in line]
    assert names[:6] == ["ho0", "ho1", "ho2", "wo0", "wo1", "wo2"]
    assert names[6:12] == ["hi0", "hi1", "hi2", "wi0", "wi1", "wi2"]
    assert family.ROTATED_NAMES == ("Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz")


def test_the_corpus_digest_pins_launch_1s_emitted_text():
    digest = family.corpus_digest()
    assert isinstance(digest, dict) and len(digest["sha256"]) == 64


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------

def test_a_dcyl_m0_grid_with_a_declared_empty_source_set_is_admitted():
    for z_kind in ("metallic", "periodic"):
        fields, pml = build(z_kind)
        verdict = family.metal_cylindrical_real_fused_hd_pair_coverage(fields, pml, (),
                                                                      Residency())
        assert verdict.covered, (z_kind, verdict.reasons)


def test_an_undeclared_source_list_is_a_refusal_and_not_an_empty_set():
    fields, pml = build()
    verdict = family.metal_cylindrical_real_fused_hd_pair_coverage(fields, pml, None,
                                                                  Residency())
    assert not verdict.covered
    assert any("was not declared" in r for r in verdict.reasons)
    assert family.plan_metal_cylindrical_real_fused_hd_pair(fields, pml, None,
                                                             Residency()) is None


def test_a_plain_electric_source_is_admitted_and_an_integrated_one_is_refused_by_name():
    fields, pml = build()
    plain = _source(fields, "Ez", integrated=False)
    integrated = _source(fields, "Ez", integrated=True)
    assert family.metal_cylindrical_real_fused_hd_pair_coverage(
        fields, pml, (plain,), Residency()).covered
    assert len(withdraw_hoist.standing_withdraws((integrated,))) == 1
    verdict = family.metal_cylindrical_real_fused_hd_pair_coverage(
        fields, pml, (integrated,), Residency())
    assert not verdict.covered
    assert any("standing integrated" in r and "HOISTS_THE_WITHDRAW = False" in r
               for r in verdict.reasons)


def test_m_not_zero_and_a_cartesian_grid_are_both_refused_by_name():
    from meep_gpu.fields import Fields
    from meep_gpu.grid import Grid
    from meep_gpu.pml import PML

    fields, pml = build(m=1, complex_storage=True)
    verdict = family.metal_cylindrical_real_fused_hd_pair_coverage(fields, pml, (),
                                                                  Residency())
    assert not verdict.covered
    assert any("grid.m = 1" in r for r in verdict.reasons)

    grid = Grid(resolution=1.0, cell_size=(6.0, 6.0, 6.0), xp=np)
    cart = Fields(grid=grid)
    cart.enable_pml_storage()
    verdict = family.metal_cylindrical_real_fused_hd_pair_coverage(
        cart, PML(grid=grid, thickness=1.0), (), Residency())
    assert not verdict.covered
    assert any(r.startswith("cylindrical constitutive half:") for r in verdict.reasons)
    assert any(r.startswith("cylindrical curl half:") for r in verdict.reasons)


def test_the_predicate_is_literally_the_two_cylindrical_m0_arm_predicates():
    text = (PACKAGE_DIR / "metal_kernels" / "cylindrical_real_fused_hd_pair.py").read_text(
        encoding="utf-8")
    body = text.split("def metal_cylindrical_real_fused_hd_pair_coverage", 1)[1]
    body = body.split("\ndef ", 1)[0]
    assert "cylindrical_real_constitutive_coverage(fields, pml, CONSTITUTIVE_SIDE," in body
    assert "cylindrical_real_curl_coverage(fields, pml, CURL_SUB_STEP, residency)" in body
    assert "seam_withdraw_reasons(" in body
    assert "deposit_repair" not in body.split('"""')[-1] or "seam_source_reasons" not in body


# ---------------------------------------------------------------------------
# The seam and the declarations
# ---------------------------------------------------------------------------

def test_REPLACES_is_the_seams_two_slots_and_the_seam_is_the_withdraw_hoists():
    assert family.SLOT == "update_H"
    assert family.REPLACES == ("update_H", "step_D")
    assert family.SEAM == withdraw_hoist.SEAM
    assert metal_launch.FUSED_PAIR_SEAMS["update_H"] == ("step_D", withdraw_hoist.SEAM)
    assert family.CARRIES_DEPOSIT_REPAIR is False
    assert family.HOISTS_THE_WITHDRAW is False


def test_the_plan_is_two_launches_and_counts_them_apart():
    assert family.MetalCylindricalRealFusedHdPairPlan.launches_per_run == 2
    slots = family.MetalCylindricalRealFusedHdPairPlan.__slots__
    assert "constitutive_launches" in slots and "curl_launches" in slots


def test_the_arm_is_registered_UNWIRED_and_carries_its_absorb_and_registry_rows():
    """WIRED 2026-09-06: the absorb row lets the seam loop ask it; the registry row
    imports it; ``wired=False`` keeps it out of ``_select_slot``."""
    rows = [row for row in arms.registered() if row.family == family.FAMILY]
    assert len(rows) == 1, rows
    assert rows[0].slot == "update_H"
    assert rows[0].wired is False
    assert rows[0].is_weld and tuple(rows[0].replaces) == family.REPLACES
    assert metal_launch.FUSED_PAIR_ARMS.get(family.FAMILY) == ("cylindrical m=0",
                                                               "cylindrical m=0")
    assert family.FAMILY in registry.FAMILY_MODULES


def test_uninstallable_with_a_measured_reason_and_the_weld_minted():
    assert family.INSTALLABLE is False
    assert "step_D was selected by the released" in family.INSTALLABLE_REASON
    assert "TIE" in family.INSTALLABLE_REASON
    assert family.WELD_OWED == ""
    welded = json.loads((PACKAGE_DIR / "metal_kernels" / "fingerprints.json").read_text())
    assert f"metal_{family.FAMILY}_device_gate" in welded


def test_this_module_touches_neither_dispatch_nor_the_other_tracks():
    tree = ast.parse((PACKAGE_DIR / "metal_kernels"
                      / "cylindrical_real_fused_hd_pair.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)) and ast.get_docstring(node):
            node.body = node.body[1:]
    code = ast.unparse(tree)
    assert "fastpath" not in code
    assert "cuda_kernels" not in code
    assert "triton_kernels.launch" not in code or "SUB_STEPS" in code


# ---------------------------------------------------------------------------
# What the corpus says this is worth
# ---------------------------------------------------------------------------

def census_rows():
    def load(name):
        path = CENSUS / name
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    record = load("examples.jsonl") + load("tests.jsonl") + load("tests_param_matched.jsonl")
    return [r for r in record if r.get("measured")]


def test_the_cell_is_the_three_rows_the_census_selects_cylindrical_m0_on_both_slots():
    rows = census_rows()
    if not rows:
        pytest.skip("the Metal census record is not present")
    cell = sorted(f"{r['leg']}:{r['row']}" for r in rows
                  if ((r.get("plan_step") or {}).get("selected") or {}).get("update_H")
                  == "cylindrical m=0"
                  and ((r.get("plan_step") or {}).get("selected") or {}).get("step_D")
                  == "cylindrical m=0")
    assert cell == [
        "tests:TestAdjointSolver.test_adjoint_solver_cyl_n2f_fields_0_0",
        "tests:TestAdjointSolver.test_adjoint_solver_cyl_n2f_fields_1_0",
        "tests:TestPMLCylindrical.test_pml_cyl_0_0_0",
    ]


def test_no_row_of_the_cell_carries_a_standing_withdraw():
    if not SEAM_RECORD.exists():
        pytest.skip("the H->D seam record is not present")
    blocks = {json.loads(line)["label"]: json.loads(line)["h_to_d_seam"]
              for line in SEAM_RECORD.read_text().splitlines() if line.strip()}
    for label in ("tests:TestAdjointSolver.test_adjoint_solver_cyl_n2f_fields_0_0",
                  "tests:TestAdjointSolver.test_adjoint_solver_cyl_n2f_fields_1_0",
                  "tests:TestPMLCylindrical.test_pml_cyl_0_0_0"):
        assert blocks[label]["withdraw_in_seam"] is False
        assert blocks[label]["n_electric_withdraws_that_do_work"] == 0


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

def test_the_gate_exists_names_the_product_and_declares_the_battery_contract():
    source = GATE.read_text(encoding="utf-8")
    assert "plan_metal_cylindrical_real_fused_hd_pair" in source
    assert "cyl_real_hd_constitutive_prefix" in source
    assert 'SUBJECT_PACKAGE = "metal_kernels"' in source
    assert "def evaluate(driver" in source and "def runtime_reasons(" in source
    assert "refuted_over_the_ceiling_source" in source
    assert "scan_divide_is_a_reciprocal_multiply" in source
    assert "scan_ladder_at_ir0_zero" in source
    assert "scan_reads_the_scratch_hy_corpus_scale" in source
