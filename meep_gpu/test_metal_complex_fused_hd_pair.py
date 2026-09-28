"""Merge-bar tests for the Metal COMPLEX/BLOCH H->D pair
(``metal_kernels/complex_fused_hd_pair.py``).

WHAT THIS SUITE OWNS AND WHAT IT DOES NOT. The BYTES are the device gate's --
``parity/meep_gpu/gate_metal_complex_fused_hd_pair.py`` walks the product beside the
array path and the two certified complex singles for twelve complete driver steps and
compares uint32 words, over eight synthetic configurations and all 17 corpus rows of
the cell. What lives here is everything true about this product WITHOUT a device: the
lift is the certified complex body with exactly the declared edits, the welded curl
REVERSES to the certified emitter's own text, the kernel reads nothing it writes, the
tables agree with the driver's seam, the predicate refuses what it must, the
composition flags are the declared ones -- plus the smallest non-vacuous device check
and the gate's measured counts read from its artifact.
"""

from __future__ import annotations

import json
import pathlib
import re

import numpy as np
import pytest

from meep_gpu import withdraw_hoist
from meep_gpu.metal_kernels import complex_fields
from meep_gpu.metal_kernels import complex_fused_hd_pair as family
from meep_gpu.metal_kernels import shaders, templates
from meep_gpu.metal_kernels.device import MAX_BUFFER_BINDINGS
from meep_gpu.triton_kernels.coverage import CONSTITUTIVE_SIDES

PACKAGE_DIR = pathlib.Path(__file__).resolve().parent
REPO = PACKAGE_DIR.parent
MODULE = PACKAGE_DIR / "metal_kernels" / "complex_fused_hd_pair.py"
GATE = REPO / "parity" / "meep_gpu" / "gate_metal_complex_fused_hd_pair.py"
#: The RELEASED cut. REPOINTED 2026-09-07 from `_cx2`, the pre-wire run, to this
#: family's own leg of the post-wire fleet re-cut: the wiring moved bytes that gate
#: imported (`launch.py`, `registry.py`, this module's `WELD_OWED`), so `_cx2` no
#: longer describes the tree and the weld in `fingerprints.json` binds `_wired`.
ARTIFACT = (REPO / "parity" / "meep_gpu" / "results"
            / "metal_complex_fused_hd_pair_2026-09-07_wired" / "gate.json")
EXPANSIONS = tuple(templates.EXPANSION_ARMS)
ARM = "FMA_V1"


def _has_mps() -> bool:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False
    return bool(getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available())


requires_mps = pytest.mark.skipif(not _has_mps(), reason="no MPS device")


def _code(text: str) -> str:
    return "\n".join(line.split("//", 1)[0] for line in text.splitlines()
                     if line.split("//", 1)[0].strip())


def _artifact() -> dict:
    if not ARTIFACT.is_file():
        pytest.skip(f"the released gate artifact {ARTIFACT} is not in this checkout")
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. The seam, the flags, the composition
# ---------------------------------------------------------------------------

def test_the_product_spans_the_drivers_h_to_d_seam_and_carries_nothing_else():
    assert family.SLOT == "update_H"
    assert tuple(family.REPLACES) == ("update_H", "step_D")
    assert tuple(family.REPLACES) == tuple(withdraw_hoist.SEAM_SPAN)
    assert family.SEAM == withdraw_hoist.SEAM == "H_to_D"
    # NOTHING IS INJECTED BETWEEN THE TWO CONSULTS, so there is no deposit to bracket;
    # what IS there is the electric withdraw, which this product does not perform.
    assert family.CARRIES_DEPOSIT_REPAIR is False
    assert family.HOISTS_THE_WITHDRAW is False


def test_the_module_does_not_consult_the_deposit_repair_at_all():
    """A False flag and an unconsulted module are different claims; this pins both.

    Read from SOURCE and by CODE rather than by prose: the docstring and one clause
    comment both name ``deposit_repair`` to say why it is silent on this seam, and a
    check that forbade the name would forbid the explanation.
    """
    code = _code(MODULE.read_text(encoding="utf-8"))
    assert "import deposit_repair" not in code
    assert "deposit_repair." not in code
    assert "seam_source_reasons" not in code


def test_installable_is_false_and_the_reason_carries_the_measured_arbitration():
    assert family.INSTALLABLE is False
    reason = family.INSTALLABLE_REASON
    assert "complex_fused_magnetic_pair" in reason
    assert "complex_fused_electric_pair" in reason
    assert "LOSS on 17" in reason
    assert "GAIN on 0" in reason
    # The four sentences no artifact of this family may omit.
    assert "PREDICATE ADMISSION" in reason
    assert "installs on zero rows" in reason
    assert "executes nowhere outside its own gate" in reason
    assert "no timing claim and no dispatch claim" in reason


def test_the_weld_is_owed_and_says_what_a_release_still_owes():
    """The module ships with the gate RELEASED and the fingerprints entry NOT written.

    ``test_metal_weld_contract.test_every_metal_family_is_welded`` partitions the gate
    fleet against ``fingerprints.json``; a non-empty ``WELD_OWED`` is the only way a
    gated family may stand outside that file, and this pins that the string says which
    artifact and which entry.
    """
    assert family.WELD_OWED == "", (
        "WELD_OWED must be empty once fingerprints.json carries "
        "metal_complex_fused_hd_pair_device_gate")
    from meep_gpu.metal_kernels import launch as metal_launch

    # The weld entries are TOP-LEVEL keys of fingerprints.json, beside the
    # "metal_kernels" digest block.
    assert "metal_complex_fused_hd_pair_device_gate" in (
        metal_launch.load_fingerprints())


def test_the_module_registers_one_unwired_row_on_update_H():
    """ENUMERABLE, NOT SELECTABLE. ``arms.arms_for`` skips an unwired row so
    ``plan_step`` cannot select it; ``arms.registered`` still returns it, which is
    what lets the seam loop ask the product and then refuse it on INSTALLABLE."""
    from meep_gpu.metal_kernels import arms as metal_arms
    from meep_gpu.metal_kernels import launch as metal_launch

    rows = [arm for arm in metal_arms.registered("update_H")
            if arm.family == family.FAMILY]
    assert len(rows) == 1
    assert rows[0].wired is False
    assert tuple(rows[0].replaces) == tuple(family.REPLACES)
    assert metal_launch.FUSED_PAIR_ARMS[family.FAMILY] == ("complex/Bloch",
                                                           "complex/Bloch")
    # `arms_for(slot, context)` is exactly `registered(slot)` filtered on `wired`
    # (arms.py:259-263), so the selectable set is read here WITHOUT building a
    # composition context -- the property is a fact about the registry row, not
    # about any one grid.
    assert family.FAMILY not in {spec.family
                                 for spec in metal_arms.registered("update_H")
                                 if spec.wired}


# ---------------------------------------------------------------------------
# 2. The lift — the certified bodies, with exactly the declared edits
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("expansion", EXPANSIONS)
def test_the_constitutive_lift_is_the_certified_body_under_the_declared_edits(expansion):
    certified = complex_fields.bloch_constitutive_source("H", expansion)
    tail = certified.split(family.DECODE_END, 1)[1][: -len("}\n")]
    for old, new in (("    float kp_0 = kp0[i], km_0 = km0[i];\n",
                      "    float kp_0 = kp0[i], km_0 = kmx[i];\n"),
                     ("    float kp_1 = kp1[j], km_1 = km1[j];\n",
                      "    float kp_1 = kp1[j], km_1 = kmy[j];\n"),
                     ("    float kp_2 = kp2[k], km_2 = km2[k];\n",
                      "    float kp_2 = kp2[k], km_2 = kmz[k];\n")):
        assert tail.count(old) == 1
        tail = tail.replace(old, new)
    for target in range(3):
        for old, new in (
                (f"    float2 prev{target} = w{target}[ii];\n",
                 f"    float2 prev{target} = wi{target}[ii];\n"),
                (f"    float2 src{target} = g{target}[ii];\n",
                 f"    float2 src{target} = b{target}[ii];\n"),
                (f"    w{target}[ii] = src{target};\n", ""),
                (f"    float2 a{target} = f{target}[ii];\n",
                 f"    float2 a{target} = hi{target}[ii];\n"),
                (f"    f{target}[ii] = a{target};\n", "")):
            assert tail.count(old) == 1, old
            tail = tail.replace(old, new)
    assert tail == family.certified_complex_constitutive_tail(expansion)


def test_the_lift_edit_table_names_every_edit_and_touches_no_arithmetic():
    rows = family.CONSTITUTIVE_LIFT_EDITS
    assert len(rows) == 7
    for row in rows:
        assert set(row) == {"line", "became", "why"}
        assert row["why"].strip()
    # NOT ONE right-hand side moves: every entry is a pointer spelling, a store that
    # moves to the caller, or the decode that becomes parameters.
    accumulations = [row for row in rows if "c_mul" in row["line"]]
    assert accumulations == []


@pytest.mark.parametrize("expansion", EXPANSIONS)
def test_the_lifted_constitutive_accumulations_are_character_identical(expansion):
    certified = complex_fields.bloch_constitutive_source(
        "H", expansion).split(family.DECODE_END, 1)[1]
    lifted = family.certified_complex_constitutive_tail(expansion)

    def accumulations(text):
        return [line.split("//", 1)[0].strip() for line in text.splitlines()
                if "c_mul_coefficient_left" in line.split("//", 1)[0]]

    assert accumulations(certified) == accumulations(lifted)
    assert len(accumulations(lifted)) == 6


def test_the_lifted_body_indexes_no_volume_the_fused_signature_renames():
    tail = family.certified_complex_constitutive_tail(ARM)
    for stem in ("f", "w", "g", "e", "km"):
        for target in range(3):
            assert f"{stem}{target}[" not in tail


@pytest.mark.parametrize("codes,phased", [((0, 0, 0), (1, 1, 1)),
                                          ((1, 1, 0), (0, 0, 1)),
                                          ((1, 1, 1), (0, 0, 0)),
                                          ((0, 1, 0), (1, 0, 1))])
def test_the_welded_curl_reverses_to_the_certified_text(codes, phased):
    """THE STRONGEST STATEMENT THIS SUITE CAN MAKE WITHOUT A DEVICE: undo the nine
    declared load redirections and the certified emitter's own body comes back, byte
    for byte -- which says NOTHING ELSE moved."""
    prologue, body = family.welded_curl_tail(codes, phased, ARM)
    restored = body
    for var, target in family.OWN_LOAD_EDITS:
        restored = restored.replace(f"    float2 {var}   = own.a{target};\n",
                                    f"    float2 {var}   = g{target}[ii];\n")
    offsets = family.offset_coordinates(body)
    for _var, target in family.HALO_TAPS:
        for offset, coordinates in offsets.items():
            call = (f"h_cell({', '.join(coordinates)}, "
                    f"{family.H_CELL_TAIL_ARGS}).a{target}")
            restored = restored.replace(call, f"g{target}[{offset}]")
    theirs = complex_fields.bloch_curl_source(codes, True, phased, ARM)
    theirs_body = theirs.split("uint idx [[thread_position_in_grid]])\n{\n",
                               1)[1][: -len("}\n")]
    assert prologue + restored == theirs_body


def test_the_offsets_are_parsed_from_the_certified_index_lines():
    _prologue, body = family.welded_curl_tail((0, 0, 0), (0, 0, 0), ARM)
    offsets = family.offset_coordinates(body)
    assert offsets == {"ox": ("si", "j", "k"), "oy": ("i", "sj", "k"),
                       "oz": ("i", "j", "sk")}


def test_a_curl_whose_index_layout_changed_raises_rather_than_redirecting():
    _prologue, body = family.welded_curl_tail((0, 0, 0), (0, 0, 0), ARM)
    broken = body.replace("int ox = si * nyz + j * nzi + k;",
                          "int ox = si * nyz + j * nzi;")
    with pytest.raises(AssertionError):
        family.offset_coordinates(broken)


def test_the_phase_block_is_left_standing_on_the_register():
    """The Bloch rotation applies to the LOADED REGISTER after the gather, so this weld
    redirects the load and never sees the multiply. A tap that lost its rotation would
    be a smooth, plausible, wrong band structure."""
    source = family.complex_fused_hd_pair_source((0, 0, 0), (1, 1, 1), ARM)
    for axis, operands in (("x", ("b_x", "c_x")), ("y", ("a_y", "c_y")),
                           ("z", ("a_z", "b_z"))):
        for operand in operands:
            assert f"{operand} = w{axis} ? c_mul({operand}, p{axis}) : {operand};" \
                in source


def test_an_unphased_axis_emits_no_multiply_at_all():
    """The SKIP, not a multiply by 1+0j, is what keeps k = 0 bit-identical to the plain
    complex engine (stepping.py:1815-1817)."""
    source = family.complex_fused_hd_pair_source((0, 0, 0), (0, 0, 0), ARM)
    assert "c_mul(" not in _code(source).split("kernel void", 1)[1]


# ---------------------------------------------------------------------------
# 3. The signature and the ceiling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("expansion", EXPANSIONS)
def test_the_shipped_signature_binds_exactly_the_platform_ceiling(expansion):
    assert family.shipped_signature_bindings(expansion) == family.PACKED_BINDINGS
    assert family.PACKED_BINDINGS == MAX_BUFFER_BINDINGS == 31


def test_the_refuted_binding_counts_are_the_declared_ones():
    assert family.ONE_MORE_POINTER_BINDINGS == family.PACKED_BINDINGS + 1
    assert family.UNSHARED_KMS_BINDINGS == family.PACKED_BINDINGS + 3
    assert family.SEPARATE_SCALAR_BINDINGS == 38
    assert family.SPLIT_PLANE_BINDINGS == 62
    # Each builder asserts its own count internally, so calling them IS the check.
    for builder in (family.refuted_one_more_pointer_source,
                    family.refuted_unshared_kms_source,
                    family.refuted_separate_scalar_source,
                    family.split_plane_pair_signature):
        assert builder().count("[[buffer(") >= family.PACKED_BINDINGS


def test_the_two_halves_share_one_yee_sub_lattice_and_that_is_why_thirty_fits():
    """The premise of the 30-pointer signature, read off the SHIPPED tables."""
    assert complex_fields.SUB_STEPS["step_D"]["suffix"] == ""
    assert CONSTITUTIVE_SIDES["H"]["half_integer"] is False


def test_the_signature_binds_twenty_one_float2_volumes_and_nine_float_vectors():
    source = family.complex_fused_hd_pair_source((0, 0, 0), (0, 0, 0), ARM)
    signature = source.split(f"kernel void {family.KERNEL}(", 1)[1].split(
        "uint idx [[thread_position_in_grid]])", 1)[0]
    assert signature.count("float2*") == 21
    assert len(re.findall(r"device const float\*", signature)) == 9
    assert signature.count("constant Params&") == 1


def test_the_params_record_puts_the_phases_first_and_is_forty_eight_bytes():
    dtype = family.params_record_dtype()
    assert dtype.itemsize == family.PARAMS_ITEMSIZE == 48
    assert dtype.names == ("px", "py", "pz", "nx", "ny", "nz", "n_elem", "dtdx")
    assert [dtype.fields[name][1] for name in dtype.names] == [0, 8, 16, 24, 28, 32,
                                                              36, 40]


def test_the_rotating_group_is_the_signatures_first_twelve_bindings():
    source = family.complex_fused_hd_pair_source((0, 0, 0), (0, 0, 0), ARM)
    signature = source.split(f"kernel void {family.KERNEL}(", 1)[1].split(
        "uint idx [[thread_position_in_grid]])", 1)[0]
    names = re.findall(r"\b(\w+)\s+\[\[buffer\(\d+\)\]\]", signature)
    assert names[:12] == ["ho0", "ho1", "ho2", "wo0", "wo1", "wo2",
                          "hi0", "hi1", "hi2", "wi0", "wi1", "wi2"]
    assert family.ROTATED_NAMES == ("Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz")


# ---------------------------------------------------------------------------
# 4. Purity — nothing written is read
# ---------------------------------------------------------------------------

def test_the_launch_reads_nothing_it_writes():
    source = family.complex_fused_hd_pair_source((0, 0, 0), (1, 1, 1), ARM)
    kernel = _code(source.split(f"kernel void {family.KERNEL}(", 1)[1].split("{", 1)[1])
    for name in ("ho0", "ho1", "ho2", "wo0", "wo1", "wo2"):
        stores = len(re.findall(rf"\b{name}\[[^\]]*\]\s*=(?!=)", kernel))
        loads = len(re.findall(rf"\b{name}\[", kernel)) - stores
        assert (stores, loads) == (1, 0), (name, stores, loads)


def test_the_in_place_group_is_touched_only_at_the_threads_own_cell():
    source = family.complex_fused_hd_pair_source((0, 0, 0), (1, 1, 1), ARM)
    kernel = _code(source.split(f"kernel void {family.KERNEL}(", 1)[1].split("{", 1)[1])
    for name in ("f0", "f1", "f2", "u0", "u1", "u2"):
        assert not re.findall(rf"\b{name}\[(?!ii\])", kernel), name


def test_the_cell_function_stores_nothing_and_serves_seven_call_sites():
    source = family.complex_fused_hd_pair_source((0, 0, 0), (1, 1, 1), ARM)
    helper = source.split("static inline h_cell_result h_cell(", 1)[1].split(
        "\nkernel void", 1)[0]
    assert not re.findall(r"\b\w+\[[^\]]*\]\s*=(?!=)", _code(helper))
    kernel = _code(source.split(f"kernel void {family.KERNEL}(", 1)[1].split("{", 1)[1])
    # ONE own cell plus SIX backward taps, all through the same function.
    assert kernel.count("h_cell(") == 7


def test_no_magnetic_pointer_survives_the_weld():
    source = family.complex_fused_hd_pair_source((0, 0, 0), (1, 1, 1), ARM)
    for target in range(3):
        assert f"g{target}[" not in source


def test_the_kernel_performs_no_floating_point_division():
    """Every PML reciprocal is precomputed on the host into ``sinv_a``; the two
    divisions in the emitted text are the INTEGER index decode."""
    source = family.complex_fused_hd_pair_source((0, 0, 0), (1, 1, 1), ARM)
    code = _code(source)
    divides = re.findall(r"[\w\.\)]\s*/\s*[\w\(]", code)
    integer = re.findall(r"\bii\s*/\s*nzi|\bplane\s*/\s*nyi", code)
    assert len(integer) == 2
    assert len(divides) == len(integer)


# ---------------------------------------------------------------------------
# 5. Specialisation and enumeration
# ---------------------------------------------------------------------------

def test_a_phased_metallic_axis_is_refused_by_the_certified_emitter():
    with pytest.raises(ValueError):
        family.complex_fused_hd_pair_source((1, 0, 0), (1, 0, 0), ARM)


def test_a_malformed_triple_is_refused_by_name():
    with pytest.raises(ValueError):
        family.complex_fused_hd_pair_source((0, 0), (0, 0, 0), ARM)
    with pytest.raises(ValueError):
        family.complex_fused_hd_pair_source((0, 0, 0), (0, 0), ARM)


def test_enumerate_sources_covers_only_the_reachable_specialisations():
    sources = family.enumerate_sources(ARM)
    # 8 boundary triples; a metallic axis cannot carry a phase, so the phase count per
    # triple is 2 ** (number of periodic axes).
    expected = sum(2 ** sum(1 for code in codes if code == templates.PERIODIC)
                   for codes in [(cx, cy, cz) for cx in (0, 1) for cy in (0, 1)
                                 for cz in (0, 1)])
    assert len(sources) == expected == 27
    assert len(set(sources.values())) == len(sources)


def test_every_emitted_source_carries_one_contraction_guard_and_the_helpers():
    for arm in EXPANSIONS:
        source = family.complex_fused_hd_pair_source((0, 0, 0), (1, 1, 1), arm)
        assert source.count(shaders.contraction_pragma(shaders.CONTRACT_OFF)) == 1
        assert templates.complex_helpers(arm) in source


# ---------------------------------------------------------------------------
# 6. Coverage — what the predicate refuses, without a device
# ---------------------------------------------------------------------------

class _Grid:
    def __init__(self, **kwargs):
        self.shape = kwargs.pop("shape", (4, 4, 4))
        for key, value in kwargs.items():
            setattr(self, key, value)

    def is_mirrored(self, axis):
        return bool(getattr(self, "mirrored", (False, False, False))[axis])


def test_an_undeclared_source_set_is_refused_rather_than_assumed_empty():
    """IGNORANCE IS NEVER AN EMPTY SET: ``Fields`` does not hold the source list."""
    reasons = withdraw_hoist.seam_withdraw_reasons(
        object(), None, undeclared="the source set was not declared",
        refusal=lambda index, source: "x",
        hoists_the_withdraw=family.HOISTS_THE_WITHDRAW, span=family.REPLACES)
    assert any("not declared" in reason for reason in reasons)


def test_a_fields_object_without_a_grid_is_refused_by_name():
    verdict = family.metal_complex_fused_hd_pair_coverage(object(), object(), ())
    assert not verdict.covered
    assert any("grid" in reason for reason in verdict.reasons)


def test_the_fold_is_refused_by_this_module_by_name():
    """Both halves already refuse a mirror plane; this module restates it, because on
    THIS seam neither symmetry fill runs and a reader would otherwise conclude a folded
    H->D seam is free."""
    text = MODULE.read_text(encoding="utf-8")
    assert "is folded: this product implements the `complex/Bloch`" in text
    assert "folded complex" in text


def test_the_rotation_invariant_is_a_predicate_clause():
    text = MODULE.read_text(encoding="utf-8")
    assert "this weld rotates it against a plan-owned" in text


# ---------------------------------------------------------------------------
# 7. The plan carries its specialisation
# ---------------------------------------------------------------------------

def test_the_plan_class_extends_the_shared_scratch_weld_and_adds_three_slots():
    from meep_gpu.metal_kernels import offdiag_weld_common as weld

    assert issubclass(family.MetalComplexFusedHdPairPlan, weld.ScratchWeldPairPlan)
    assert family.MetalComplexFusedHdPairPlan.__slots__ == ("phased", "phase_values",
                                                            "expansion")
    # `run` and `_resolve` are the BASE's: the rotation is not reimplemented here.
    assert "run" not in vars(family.MetalComplexFusedHdPairPlan)
    assert "_resolve" not in vars(family.MetalComplexFusedHdPairPlan)


# ---------------------------------------------------------------------------
# 8. The device — the smallest non-vacuous check
# ---------------------------------------------------------------------------

@requires_mps
@pytest.mark.parametrize("expansion", EXPANSIONS)
def test_every_reachable_specialisation_compiles_on_this_platform(expansion):
    from meep_gpu.metal_kernels.device import compile_source

    for source in family.enumerate_sources(expansion).values():
        compile_source(source)


@requires_mps
def test_one_complete_driver_step_is_bit_identical_to_the_array_path():
    """The smallest non-vacuous device case, with the control that BITES beside it.

    The heavy walk is the gate's; what this owns is that the product is reachable and
    correct on a fresh checkout with a device, and that a comparison which could not
    tell two states apart would have failed here too.
    """
    import sys

    import torch  # noqa: F401

    sys.path.insert(0, str(REPO / "parity" / "meep_gpu"))
    import metal_composition_matrix as matrix  # noqa: E402

    from meep_gpu import stepping  # noqa: E402
    from meep_gpu.fields import Fields  # noqa: E402
    from meep_gpu.grid import Grid  # noqa: E402
    from meep_gpu.metal_kernels.device import Residency  # noqa: E402
    from meep_gpu.pml import PML  # noqa: E402

    # THE ARM IS A MEASURED PLATFORM FACT read from the checked-in probe artifact, and
    # binding it is what makes this a test rather than a skip. `prepare_environment`
    # is the ONE place the candidate list lives.
    matrix.prepare_environment()
    if complex_fields.expansion_from_probe(
            complex_fields.load_expansion_probe()) is None:
        from conftest import requires_resource_skip  # noqa: PLC0415

        requires_resource_skip(
            "metal_complex_expansion_probe",
            "no complex expansion probe artifact under parity/meep_gpu/results; the "
            "arm is measured, never defaulted, so the product cannot be built")

    state = ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
             "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
             "f_w_Hx", "f_w_Hy", "f_w_Hz", "f_w_Ex", "f_w_Ey", "f_w_Ez")

    def build(seed):
        grid = Grid(resolution=10.0, cell_size=(1.2, 1.0, 0.9), boundaries="periodic",
                    dimensions=3, courant=0.35, k_point=(0.3, 0.0, -0.25), xp=np)
        fields = Fields(grid=grid, force_complex_fields=True)
        fields.enable_pml_storage()
        count = int(np.prod(grid.shape))
        index = np.arange(count, dtype=np.float32).reshape(grid.shape)
        eps = (1.45 + 0.30 * np.sin(index * np.float32(0.037))).astype(np.float32)
        fields.set_isotropic_epsilon_volume(
            eps, (np.float32(1.0) / eps).astype(np.float32))
        pml = PML(grid=grid, thickness=tuple((2, 2) for _ in range(3)))
        rng = np.random.default_rng(seed)
        for name in state:
            array = getattr(fields, name, None)
            if array is None:
                continue
            view = np.empty(grid.shape, dtype=np.complex64).view(np.float32)
            view[...] = rng.standard_normal(view.shape).astype(np.float32) * 0.3
            array[...] = view.view(np.complex64).reshape(grid.shape)
        return fields, pml

    def words(array):
        return np.ascontiguousarray(array).view(np.uint32).reshape(-1)

    def differing(left, right):
        return int(np.count_nonzero(words(left) != words(right)))

    reference, reference_pml = build(20260907)
    device_fields, device_pml = build(20260907)
    residency = Residency()
    plan = family.plan_metal_complex_fused_hd_pair(
        device_fields, device_pml, sources=(), residency=residency)
    assert plan is not None
    residency.sync_in()

    stepping.update_H(reference, reference_pml)
    stepping.step_D(reference, reference_pml)
    residency.sync_in()
    plan.run()
    residency.sync_out()

    assert plan.launches == 1
    total = 0
    for name in state:
        left = getattr(reference, name, None)
        if left is None:
            continue
        total += int(words(left).size)
        assert differing(left, getattr(device_fields, name)) == 0, name
    assert total > 0

    # THE CONTROL THAT BITES: undo the rotation and the H group is the pre-launch one.
    for name in family.ROTATED_NAMES:
        current = getattr(device_fields, name)
        setattr(device_fields, name, plan.rotated[name])
        plan.rotated[name] = current
    assert sum(differing(getattr(reference, name), getattr(device_fields, name))
               for name in family.ROTATED_NAMES) > 0


# ---------------------------------------------------------------------------
# 9. The artifact — the gate's own measured counts
# ---------------------------------------------------------------------------

def test_the_released_artifact_is_this_family_and_released():
    payload = _artifact()
    assert payload["family"] == family.FAMILY
    assert payload["release"]["released"] is True
    assert payload["canonical_verdict"]["released"] is True
    assert payload["summary"]["status"] == "passed"
    assert payload["weld"]["passed"] is True


def test_the_artifact_carries_every_leg_this_family_owes():
    payload = _artifact()
    assert set(payload["summary"]["legs_run"]) == {
        "binding_ceiling", "lift", "driver_order", "refusal", "product",
        "value_class", "null_control", "purity", "divide_spelling",
        "helper_spellings", "corpus_rows", "arbitration", "mutations", "disarm"}


def test_the_artifact_measured_the_ceiling_as_an_equality():
    row = _artifact()["legs"]["binding_ceiling"]
    assert row["ceiling_measured"] == MAX_BUFFER_BINDINGS == 31
    assert row["headroom_measured"] == 0
    assert row["ceiling_measured_equals_declared"] is True
    assert not any(entry["compiled"] for entry in row["refuted"].values())


def test_the_artifact_drove_every_corpus_row_of_the_cell():
    row = _artifact()["legs"]["corpus_rows"]
    assert row["rows_in_cell"] == 17
    assert row["rows_admitted"] == 17
    assert row["rows_driven"] == 17
    assert row["rows_bit_identical"] == 17
    assert row["compared_words"] > 100_000_000
    # The cell carries NO row with a standing in-seam withdraw, which is what separates
    # it from the Cartesian real cell's 2 of 49.
    assert row["facts"]["rows_with_a_standing_withdraw"] == []


def test_the_artifact_measured_the_arbitration_rather_than_asserting_it():
    row = _artifact()["legs"]["arbitration"]
    assert row["verdicts"] == {"loss": 17, "tie": 0, "gain": 0}
    assert row["installable_declared"] is False


def test_the_artifact_caught_every_armed_defect_and_no_null():
    payload = _artifact()
    rows = payload["legs"]["mutations"]
    armed = [row for row in rows if row["must_catch"] is True]
    nulls = [row for row in rows if row["must_catch"] is False]
    recorded = [row for row in rows if row["must_catch"] is None]
    assert len(armed) == 9 and all(row["caught"] == row["ran"] > 0 for row in armed)
    assert len(nulls) == 3 and all(row["caught"] == 0 for row in nulls)
    # The two record-only rows are the sign-of-zero spellings the step-level instrument
    # cannot see; the helper leg is where they are caught, and the pair is the point.
    assert {row["mutation"] for row in recorded} == {"negation_zero_minus_x",
                                                     "folded_zero_cross_term"}
    assert payload["legs"]["disarm"]["diverged"] == 0


def test_the_artifact_re_measured_the_three_load_bearing_complex_spellings():
    rows = {row["spelling"]: row
            for row in _artifact()["legs"]["helper_spellings"]["rows"]}
    assert rows["c_mul/negation_zero_minus_x"]["differing"] == 36
    assert rows["c_mul/negation_zero_minus_x"]["words"] == 512
    assert rows["c_mul_field_left/plane_wise"]["differing"] == 24
    assert rows["c_mul_field_left/plane_wise"]["words"] == 128
    assert rows["c_mul_coefficient_left/folded_zero_cross_term"]["differing"] == 12
    assert rows["c_mul_coefficient_left/field_left_orientation"]["differing"] == 0


def test_the_artifact_re_measured_the_complex_by_real_divide_fact():
    row = _artifact()["legs"]["divide_spelling"]
    assert row["float_divides_in_the_emitted_kernel"] == 0
    measured = row["numpy_complex64_over_float32"]
    # numpy's complex64/float32 IS the reciprocal multiply on random data.
    assert measured["random"]["reciprocal_multiply_differing"] == 0
    assert measured["random"]["componentwise_differing"] > 0


def test_the_gate_declares_itself_a_census_battery():
    text = GATE.read_text(encoding="utf-8")
    assert 'SUBJECT_PACKAGE = "metal_kernels"' in text
    assert "def evaluate(" in text
    assert "def runtime_reasons(" in text
