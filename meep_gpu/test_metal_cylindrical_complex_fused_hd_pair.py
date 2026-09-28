"""Merge-bar tests for the Metal complex Dcyl two-launch H->D pair
(``cylindrical_complex_fused_hd_pair``).

WHAT THIS SUITE OWNS AND WHAT IT DOES NOT. The BYTES are the device gate's —
``parity/meep_gpu/gate_metal_cylindrical_complex_fused_hd_pair.py`` walks the product beside
the array path and the two certified singles for twelve complete steps and compares uint32
words. What lives here is everything true about this product WITHOUT a device: the lift is
the certified body with exactly the declared edits, launch 2 is the certified curl byte for
byte, the lead kernel reads nothing it writes, the tables agree with the driver's seam, the
composition flags are the declared ones — plus the smallest non-vacuous device check and
the gate's measured counts read from its artifact.
"""

from __future__ import annotations

import ast
import hashlib
import json
import pathlib
import re

import numpy as np
import pytest

from meep_gpu import withdraw_hoist
from meep_gpu.metal_kernels import complex_fields
from meep_gpu.metal_kernels import cylindrical_complex as cylc
from meep_gpu.metal_kernels import cylindrical_complex_fused_hd_pair as family
from meep_gpu.metal_kernels import cylindrical_complex_scan as scan
from meep_gpu.metal_kernels import shaders, templates
from meep_gpu.metal_kernels.device import MAX_BUFFER_BINDINGS
from meep_gpu.triton_kernels.launch import SUB_STEPS

PACKAGE_DIR = pathlib.Path(__file__).resolve().parent
REPO = PACKAGE_DIR.parent
MODULE = PACKAGE_DIR / "metal_kernels" / "cylindrical_complex_fused_hd_pair.py"
GATE = REPO / "parity" / "meep_gpu" / "gate_metal_cylindrical_complex_fused_hd_pair.py"
#: The RELEASED cut. ``_cyl`` and ``_cyl2`` beside it are earlier cuts refused on gate
#: harness facts (a 32-row blocked-scan tile that never closes on the 16-row fixture; a
#: host mutation counted by lead launches it deliberately issues none of).
ARTIFACT = (REPO / "parity" / "meep_gpu" / "results"
            / "metal_cylindrical_complex_fused_hd_pair_2026-09-06_cyl3" / "gate.json")
EXPANSIONS = tuple(templates.EXPANSION_ARMS)


def _has_mps() -> bool:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


requires_mps = pytest.mark.skipif(not _has_mps(), reason="no MPS device")


def _code(text: str) -> str:
    return "\n".join(line.split("//", 1)[0] for line in text.splitlines()
                     if line.split("//", 1)[0].strip())


# ---------------------------------------------------------------------------
# 1. The seam, the flags, the counts
# ---------------------------------------------------------------------------

def test_the_product_spans_the_drivers_h_to_d_seam_and_carries_nothing_else():
    assert family.SLOT == "update_H"
    assert tuple(family.REPLACES) == ("update_H", "step_D") == tuple(withdraw_hoist.SEAM_SPAN)
    assert family.SEAM == withdraw_hoist.SEAM == "H_to_D"
    assert family.CARRIES_DEPOSIT_REPAIR is False
    assert family.HOISTS_THE_WITHDRAW is False
    assert family.MetalCylindricalComplexFusedHdPairPlan.launches_per_run == 2
    assert family.MetalCylindricalComplexFusedHdPairPlan.replaces_sub_steps == family.REPLACES


def test_installable_is_false_and_the_reason_names_the_tie_and_the_round_trip():
    assert family.INSTALLABLE is False
    reason = family.INSTALLABLE_REASON
    assert "fused electric D/E pair" in reason
    assert "THREE launches" in reason
    assert "round trip" in reason
    assert "update_H through update_E" in reason


def test_the_binding_counts_are_the_emitted_ones_and_under_the_ceiling():
    for expansion in EXPANSIONS:
        assert family.lead_signature_bindings(expansion) == family.LEAD_BINDINGS == 25
        source = family.lead_source(expansion)
        signature = source.split("kernel void", 1)[1].split("{", 1)[0]
        indices = [int(part.split(")")[0]) for part in signature.split("[[buffer(")[1:]]
        assert indices == list(range(25))
        assert signature.count("device float2*") == 7          # 6 scratch + the prefix
        assert signature.count("device const float2*") == 9    # hi, wi, b
        assert signature.count("device const float*") == 8     # 6 coefficients + 2 rows
    assert family.LEAD_POINTERS == 24
    assert family.CURL_BINDINGS == cylc.CURL_BINDINGS == 27
    assert max(family.LEAD_BINDINGS, family.CURL_BINDINGS) <= MAX_BUFFER_BINDINGS
    assert MAX_BUFFER_BINDINGS - family.LEAD_BINDINGS == 6


@pytest.mark.parametrize("extra", (0, 3, 6, 7))
def test_the_refuted_lead_plus_signature_adds_exactly_the_extra_pointers(extra):
    source = family.refuted_lead_plus_source(extra)
    signature = source.split("kernel void", 1)[1].split("{", 1)[0]
    assert signature.count("[[buffer(") == family.LEAD_BINDINGS + extra


def test_enumerate_sources_is_one_lead_and_six_certified_curls_per_arm():
    for expansion in EXPANSIONS:
        sources = family.enumerate_sources(expansion)
        leads = [label for label in sources if label.startswith(family.LEAD_KERNEL)]
        curls = [label for label in sources if label.startswith(family.CURL_KERNEL)]
        assert len(leads) == 1 and len(curls) == 6
        assert len(set(sources.values())) == 7


# ---------------------------------------------------------------------------
# 2. The lift
# ---------------------------------------------------------------------------

def _arithmetic_lines(text: str):
    return [line.split("//", 1)[0].strip() for line in text.splitlines()
            if "c_mul_coefficient_left" in line.split("//", 1)[0]
            or line.split("//", 1)[0].strip().startswith("float kp_")]


@pytest.mark.parametrize("expansion", EXPANSIONS)
def test_the_lift_is_the_certified_complex_h_body_with_exactly_the_declared_edits(expansion):
    certified = complex_fields.bloch_constitutive_source("H", expansion)
    certified_tail = certified.split(family.DECODE_END, 1)[1][: -len("}\n")]
    tail = family.certified_complex_constitutive_tail(expansion)
    rederived = certified_tail
    for target in range(3):
        for old, new in ((f"    float2 prev{target} = w{target}[ii];\n",
                          f"    float2 prev{target} = wi{target}[ii];\n"),
                         (f"    float2 src{target} = g{target}[ii];\n",
                          f"    float2 src{target} = b{target}[ii];\n"),
                         (f"    w{target}[ii] = src{target};\n", ""),
                         (f"    float2 a{target} = f{target}[ii];\n",
                          f"    float2 a{target} = hi{target}[ii];\n"),
                         (f"    f{target}[ii] = a{target};\n", "")):
            assert rederived.count(old) == 1, old
            rederived = rederived.replace(old, new)
    assert rederived == tail
    assert _arithmetic_lines(certified_tail) == _arithmetic_lines(tail)
    assert len(_arithmetic_lines(tail)) == 9
    for stem in ("f", "w", "g", "e"):
        for target in range(3):
            assert f"{stem}{target}[" not in tail


def test_the_declared_edits_touch_no_arithmetic():
    assert len(family.CONSTITUTIVE_LIFT_EDITS) == 6
    for entry in family.CONSTITUTIVE_LIFT_EDITS:
        assert "c_mul" not in entry["became"]
        assert "+" not in entry["became"].split("=")[-1] or "removed" in entry["became"]


@pytest.mark.parametrize("expansion", EXPANSIONS)
def test_launch_two_is_the_certified_curl_byte_for_byte(expansion):
    for bcz in (templates.PERIODIC, templates.METALLIC):
        for m_arm in cylc.M_ARMS:
            ours = family.curl_source(bcz, m_arm, expansion)
            theirs = cylc.cylindrical_curl_source(bcz, True, m_arm, expansion)
            assert hashlib.sha256(ours.encode()).hexdigest() == hashlib.sha256(
                theirs.encode()).hexdigest()
            assert "kernel void cyl_complex_pml_curl_step(" in ours


# ---------------------------------------------------------------------------
# 3. The lead kernel: the seam, the scan, the purity
# ---------------------------------------------------------------------------

def test_the_leader_scans_the_recomputed_post_update_hy_and_its_own_row_zero():
    code = _code(family.lead_source("FMA_V1"))
    assert "c_mul_field_left(own.a1, weights[0])" in code
    assert f"c_mul_field_left(h_cell(i, j, k, {family.H_CELL_TAIL_ARGS}).a1, weights[i])" in code
    assert "if (i == 0) {" in code
    assert "hi1[i * nyz + base]" not in code, "the leader must never scan the PRE-update Hy"


def test_the_lead_carries_the_scan_modules_loop_and_divide_helper_verbatim():
    source = family.lead_source("FMA_V1")
    assert scan._DIVIDE_HELPER.strip() in source
    assert "return z * (1.0f / d);" in source
    assert "z / d" not in _code(source)
    for line in scan.COLUMN_SCAN_LOOP.splitlines():
        if line.strip() and "__SRC" not in line:
            assert line.strip() in source, line
    assert source.count(shaders.contraction_pragma(shaders.CONTRACT_OFF)) == 1


def test_nothing_the_lead_launch_writes_is_read_by_it():
    source = family.lead_source("FMA_V1")
    body = _code(source.split("kernel void", 1)[1].split("{", 1)[1])
    written = ("ho0", "ho1", "ho2", "wo0", "wo1", "wo2", "out")
    read = ("hi0", "hi1", "hi2", "wi0", "wi1", "wi2", "b0", "b1", "b2",
            "kp0", "km0", "kp1", "km1", "kp2", "km2", "weights", "divisor")
    for name in written:
        stores = len(re.findall(rf"\b{name}\[[^\]]*\]\s*=(?!=)", body))
        loads = len(re.findall(rf"\b{name}\[", body)) - stores
        assert stores >= 1 and loads == 0, (name, stores, loads)
    for name in read:
        assert not re.findall(rf"\b{name}\[[^\]]*\]\s*=(?!=)", body), name
    helper = _code(source.split("static inline h_cell_result h_cell(", 1)[1].split(
        "\nkernel void", 1)[0])
    assert not re.findall(r"\b\w+\[[^\]]*\]\s*=(?!=)", helper), "h_cell must store nothing"


def test_the_rotation_order_is_the_signature_order():
    assert family.ROTATED_NAMES == ("Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz")
    signature = family.lead_source("FMA_V1").split("kernel void", 1)[1].split("{", 1)[0]
    names = re.findall(r"\*\s+(\w+)\s+\[\[buffer\((\d+)\)\]\]", signature)
    assert [n for n, _ in names[:6]] == ["ho0", "ho1", "ho2", "wo0", "wo1", "wo2"]
    assert [n for n, _ in names[6:12]] == ["hi0", "hi1", "hi2", "wi0", "wi1", "wi2"]
    assert tuple(SUB_STEPS["step_D"]["sources"]) == ("Hx", "Hy", "Hz")


def test_the_h_cell_pointer_table_is_the_signature_order_the_call_sites_forward():
    names = [name for _kind, name in family.H_CELL_POINTERS]
    assert names == ["hi0", "hi1", "hi2", "wi0", "wi1", "wi2", "b0", "b1", "b2",
                     "kp0", "km0", "kp1", "km1", "kp2", "km2"]
    assert family.H_CELL_TAIL_ARGS.startswith("nxi, nyi, nzi, hi0")


# ---------------------------------------------------------------------------
# 4. Structural contracts
# ---------------------------------------------------------------------------

def test_the_module_imports_no_torch_at_module_scope():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] != "torch" for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("torch")


def test_registration_and_the_weld_move_together():
    """Before the wiring lands: no arm registered (so ``plan_step`` cannot see it and
    ``FAMILY_MODULES`` is not owed an entry) and WELD_OWED declared. After: both flip
    in one change — this test accepts either consistent state and refuses the mixed ones."""
    text = MODULE.read_text(encoding="utf-8")
    registered = "def register_arms(" in text
    if registered:
        assert family.WELD_OWED == ""
        from meep_gpu.metal_kernels import registry
        assert "cylindrical_complex_fused_hd_pair" in registry.FAMILY_MODULES
    else:
        assert family.WELD_OWED and len(family.WELD_OWED) > 200
        assert "results/" in family.WELD_OWED


def test_the_gate_exists_and_names_the_product():
    text = GATE.read_text(encoding="utf-8")
    assert "cylindrical_complex_fused_hd_pair" in text
    assert "leader_scans_pre_launch_hy" in text


# ---------------------------------------------------------------------------
# 5. The smallest non-vacuous device check
# ---------------------------------------------------------------------------

@requires_mps
def test_two_seams_are_bit_identical_and_the_order_control_bites():
    import sys

    sys.path.insert(0, str(REPO / "parity" / "meep_gpu"))
    import metal_composition_matrix as matrix  # noqa: E402

    from meep_gpu import stepping  # noqa: E402
    from meep_gpu.metal_kernels.device import Residency  # noqa: E402

    matrix.prepare_environment()
    if cylc.expansion_from_probe(cylc.load_expansion_probe()) is None:
        pytest.skip("the cylindrical complex expansion probe is not bound on this host")

    def build():
        fields, pml = matrix.cylindrical(m=1, complex_storage=True, z_kind="metallic")
        rng = np.random.default_rng(9)
        for name in ("fu_Dx", "fu_Dy", "fu_Dz", "f_w_Hx", "f_w_Hy", "f_w_Hz"):
            array = getattr(fields, name)
            array[...] = (rng.uniform(-0.3, 0.3, array.shape)
                          + 1j * rng.uniform(-0.3, 0.3, array.shape)).astype(array.dtype)
        return fields, pml

    def words(a):
        return np.ascontiguousarray(a).view(np.uint32).reshape(-1)

    names = ("Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz", "Dx", "Dy", "Dz",
             "fu_Dx", "fu_Dy", "fu_Dz")
    for order, must_match in (("lead_then_curl", True), ("curl_then_lead", False)):
        ref_fields, ref_pml = build()
        dev_fields, dev_pml = build()
        residency = Residency()
        plan = family.plan_metal_cylindrical_complex_fused_hd_pair(
            dev_fields, dev_pml, sources=(), residency=residency)
        assert plan is not None
        residency.sync_in()
        differing = 0
        for _step in range(2):
            stepping.update_H(ref_fields, ref_pml)
            stepping.step_D(ref_fields, ref_pml)
            if order == "lead_then_curl":
                plan.run()
            else:
                plan.run_curl()
                plan.run_lead()
            residency.sync_out()
            differing += sum(int(np.count_nonzero(words(getattr(dev_fields, n))
                                                  != words(getattr(ref_fields, n))))
                             for n in names)
            assert residency.verify() == {}
            stepping.update_E(ref_fields, ref_pml)
            stepping.step_B(ref_fields, ref_pml)
            stepping.update_E(dev_fields, dev_pml)
            stepping.step_B(dev_fields, dev_pml)
            residency.sync_in()
        assert plan.launches == 4 and plan.lead_launches == 2 and plan.curl_launches == 2
        if must_match:
            assert differing == 0, differing
            moved = sum(int(np.count_nonzero(words(getattr(dev_fields, n))))
                        for n in names)
            assert moved > 64
        else:
            assert differing > 0, "the curl-before-lead control did not bite"


# ---------------------------------------------------------------------------
# 6. The measured counts, read from the released artifact
# ---------------------------------------------------------------------------

def _artifact():
    if not ARTIFACT.exists():
        pytest.skip(f"{ARTIFACT} is not on this host")
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_the_released_artifact_says_it_released_and_bisected_the_ceiling():
    record = _artifact()
    assert record["summary"]["status"] == "passed"
    assert record["release"]["released"] is True
    ceiling = record["legs"]["binding_ceiling"]
    assert ceiling["ceiling_measured"] == 31 == MAX_BUFFER_BINDINGS
    assert all(v["text_bindings"] == 25 and v["compiled"] for v in ceiling["lead"].values())
    assert ceiling["lead_plus"]["31"]["compiled"] is True
    assert ceiling["lead_plus"]["32"]["compiled"] is False
    assert "out of bounds" in ceiling["lead_plus"]["32"]["error"]


def test_the_artifact_walked_six_configurations_against_both_references():
    record = _artifact()
    product = record["legs"]["product"]
    assert len(product) == 6
    assert all(row["bit_identical"] and row["compared_words"] == 184_320 for row in product)
    assert {row["m_arm"] for row in product} == {cylc.M_ZERO, cylc.M_ONE, cylc.M_MANY}
    assert {row["bcz"] for row in product} == {templates.PERIODIC, templates.METALLIC}
    assert all(row["bit_identical"] for row in record["legs"]["value_class"])
    sync = record["legs"]["sync"]
    assert sync["product_seam_syncs_out_total"] == 0 == sync["product_seam_syncs_in_total"]
    assert sync["singles_seam_syncs_out_total"] == sync["steps_observed"] == 72
    assert record["legs"]["launch_structure"]["lead_prefix_vs_standalone_scan_differing"] == 0


def test_the_artifact_caught_every_armed_defect_and_confirmed_the_nulls():
    record = _artifact()
    verdicts = record["summary"]["mutation_verdicts"]
    for armed in ("true_divide_spelling", "leader_scans_pre_launch_hy",
                  "leader_row_zero_pre_launch", "blocked_scan_8", "curl_before_lead",
                  "curl_without_lead", "rotation_undone"):
        assert verdicts[armed] == "CAUGHT 3/3", armed
    for null in ("literal_numpy_divide_loop", "commuted_accumulator",
                 "byte_neutral_store_order"):
        assert verdicts[null] == "CAUGHT 0/3", null
    assert record["legs"]["disarm"]["passed"] is True
    assert record["legs"]["purity"]["passed"] is True
    assert record["legs"]["lift"]["passed"] is True
    assert record["legs"]["refusal"]["integrated_electric_refused_by_name"] is True
