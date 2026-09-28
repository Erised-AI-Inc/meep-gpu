"""Laptop tests for the REAL BETA Metal H->D weld, BOTH variants.

THE MERGE BAR, NOT THE CERTIFICATION. The byte claim is a device gate that runs
complete driver steps and compares uint32 words
(``parity/meep_gpu/gate_metal_beta_real_fused_hd_pair.py``). These tests are chosen for
what a byte gate would catch late, on a configuration nobody sweeps, or not at all --
and on this family that matters more than usual, because the CELL IS TWO CORPUS ROWS.
The lift can only ever be as strong as its denominator; these tests are where the
levers a two-row cell cannot pull are held.

* **THE VARIANT PARTITION.** One product, two board cells, sound only while no grid is
  admitted by both emitters. ``resolve_variant`` reads the fold off the grid, and the
  partition is asserted total and exclusive;
* **BETA IS A CURL-ONLY TERM**, the premise that lets the certified real ``update_H``
  be welded in unchanged. ``certified_h_cell`` asserts it at build time and the test
  drives that assertion INCLUDING the negative;
* **the packed record is the only UNPADDED one of the four H->D welds** -- seven 4-byte
  members, 28 bytes -- and its offsets are asserted explicitly rather than left to
  numpy's natural layout;
* **the sign of the beta term.** IEEE-754 makes ``curl - (c*g)`` and the array path's
  ``curl + (-(c*g))`` the same bits, so the SIGN is the one thing this line's spelling
  can move; it is a gate mutation and its premise is asserted here;
* **the lift**, reconstructed independently and diffed line-wise on both variants;
* **the predicate's refusals**, both directions on both inverted clauses (beta, and
  storage), every one answerable without a GPU.

Everything that needs a device is guarded and skipped.

WHAT THIS FILE DOES NOT CLAIM. Nothing here is a byte-identity measurement and nothing
here licenses installing the product. The cells' numbers -- 1 row at
``(special_kz real beta)`` and 1 at ``(folded beta real)`` -- come from the Metal
fusion board and are asserted against it.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

_PARITY = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       os.pardir, "parity", "meep_gpu"))
if _PARITY not in sys.path:
    sys.path.insert(0, _PARITY)

import metal_composition_matrix as matrix  # noqa: E402

ENVIRONMENT = matrix.prepare_environment()

from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid, Mirror  # noqa: E402
from meep_gpu.metal_kernels import (  # noqa: E402
    arms,
    beta_complex_fused_hd_pair as complex_beta,
    beta_real_fused_hd_pair as beta,
    folded_beta,
    fused_hd_pair as plain,
    offdiag_weld_common as weld,
    shaders,
    special_kz,
)
from meep_gpu.metal_kernels.device import MAX_BUFFER_BINDINGS  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.triton_kernels.coverage import CONSTITUTIVE_SIDES  # noqa: E402
from meep_gpu.triton_kernels.symmetry import (  # noqa: E402
    CODE_METALLIC as M, CODE_MIRROR_METALLIC as MM, CODE_MIRROR_PERIODIC as MP,
    CODE_PERIODIC as P,
)

API_ROOT = Path(__file__).resolve().parent.parent
#: The 2026-09-10 per-cell board, on which this product declares BOTH its cells and
#: the folded one is credited row by row against this gate's own lift records.
CELLS_BOARD = (API_ROOT / "parity" / "meep_gpu" / "results"
               / "fusion_matrix_metal_2026-09-11_dispatch" / "fusion_matrix.json")
BOARD = (API_ROOT / "parity" / "meep_gpu" / "results"
         / "fusion_matrix_metal_2026-09-07_wired" / "fusion_matrix.json")

SPECIALISATIONS = [
    ("plain", (P, P, P)), ("plain", (M, P, P)), ("plain", (M, M, M)),
    ("folded", (P, MP, P)), ("folded", (P, MM, P)), ("folded", (M, MM, P)),
    ("folded", (MM, MM, MM)),
]


def _torch_mps() -> bool:
    try:
        import torch  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return False
    return bool(getattr(getattr(torch, "backends", None), "mps", None)
                and torch.backends.mps.is_available())


needs_mps = pytest.mark.skipif(not _torch_mps(),
                               reason="no torch.mps device on this host")


class _FakeResidency:
    device = "cpu"

    def mirror(self, name, host, dtype=None, constant=False):  # noqa: ANN001, ARG002
        return host

    def tensor_for_host(self, host):  # noqa: ANN001
        return host


def _pair(*, folds=(), beta_value=0.33, complex_storage=False, boundaries=None):
    size = [2.0, 2.1, 0.0]
    if folds:
        size = [1.6, 2.0, 0.0]
    grid = Grid(resolution=10.0, cell_size=tuple(size), dimensions=2, courant=0.35,
                symmetry=tuple(Mirror(axis, int(phase)) for axis, phase in folds),
                boundaries=boundaries, beta=beta_value, xp=np)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    shape = tuple(grid.shape)
    fields.set_epsilon_volumes(
        {name: np.full(shape, np.float32(value), np.float32)
         for name, value in (("Ex", 2.0), ("Ey", 2.5), ("Ez", 3.0))},
        {name: np.full(shape, np.float32(1.0 / value), np.float32)
         for name, value in (("Ex", 2.0), ("Ey", 2.5), ("Ez", 3.0))})
    fields.enable_pml_storage()
    folded_indices = {"XYZ".index(axis) for axis, _phase in folds}
    thickness = tuple((0, 0) if shape[index] < 6
                      else (0, 2) if index in folded_indices else (2, 2)
                      for index in range(3))
    return fields, PML(grid=grid, thickness=thickness)


# ---------------------------------------------------------------------------
# The declarations
# ---------------------------------------------------------------------------

def test_the_span_is_the_seam_and_both_seam_flags_are_false_with_a_reason():
    from meep_gpu import withdraw_hoist  # noqa: PLC0415

    assert beta.REPLACES == ("update_H", "step_D")
    assert beta.SLOT == "update_H"
    assert beta.SEAM == withdraw_hoist.SEAM
    assert beta.CARRIES_DEPOSIT_REPAIR is False
    assert beta.HOISTS_THE_WITHDRAW is False
    assert beta.INSTALLABLE is False
    assert "MEASURED" in beta.INSTALLABLE_REASON


def test_the_family_is_welded_and_the_debt_it_replaced_is_kept():
    """``WELD_OWED`` empty is a CLAIM, and the ledger is the other half of it.

    ``test_metal_weld_contract.test_every_metal_family_is_welded`` partitions the
    ``gate_metal_*.py`` fleet against ``fingerprints.json`` and requires the declaring
    set and the welded set to be disjoint AND to exhaust it, so the two halves of a
    release move together: emptying this string without the entry claims a gate this
    tree cannot show, and cutting the entry without emptying it leaves a welded family
    still declaring its weld is owed. Both are read off the tree here.

    THE ORDER IS THE POINT. Emptying ``WELD_OWED`` MOVES the module, and the module is
    the first pin of its own weld, so ``mint_metal_weld.py`` refuses an artifact whose
    recorded digests have moved. The gate was therefore re-run AGAINST the emptied
    bytes and minted from that artifact. The debt text is kept in
    ``_RETIRED_WELD_OWED`` rather than deleted, which the third assertion reads.
    """
    from meep_gpu.metal_kernels import launch as metal_launch  # noqa: PLC0415

    assert beta.WELD_OWED == "", (
        "the family holds a released device gate; WELD_OWED must be empty, and the "
        "weld contract's partition is what reads it")
    assert ("metal_beta_real_fused_hd_pair_device_gate"
            in metal_launch.load_fingerprints()), (
        "WELD_OWED is empty and fingerprints.json carries no "
        "metal_beta_real_fused_hd_pair_device_gate: one of the two is false")
    assert "gate_metal_beta_real_fused_hd_pair.py" in beta._RETIRED_WELD_OWED
    assert "fingerprints.json" in beta._RETIRED_WELD_OWED


def test_one_arm_row_serves_two_cells_and_it_is_unwired_and_gated_on_beta():
    rows = [row for row in arms.registered() if row.family == beta.FAMILY]
    assert len(rows) == 1
    row = rows[0]
    assert row.slot == beta.SLOT
    assert row.wired is False
    assert tuple(row.replaces) == beta.REPLACES
    assert row.gate is beta._has_beta
    assert set(beta.VARIANT_CELLS) == set(beta.VARIANTS)


def test_the_pointer_half_of_the_signature_is_the_plain_real_products():
    for name in ("PACKED_BINDINGS", "ONE_MORE_POINTER_BINDINGS",
                 "UNSHARED_KMS_BINDINGS"):
        assert getattr(beta, name) == getattr(plain, name), name
    assert beta.ROTATED_NAMES is plain.ROTATED_NAMES
    # THE TWO BETA WORDS ARE THE ONLY DIFFERENCE, and unpacked they cost exactly two.
    assert beta.SEPARATE_SCALAR_BINDINGS == plain.SEPARATE_SCALAR_BINDINGS + 2 == 37


def test_this_is_the_only_unpadded_params_record_of_the_four_hd_welds():
    """Seven 4-byte members at 4-byte alignment: 28 bytes, no tail padding."""
    dtype = beta.params_record_dtype()
    assert dtype.itemsize == beta.PARAMS_ITEMSIZE == 28
    assert dtype.names == ("nx", "ny", "nz", "n_elem", "dtdx",
                           "beta_plus", "beta_minus")
    assert [dtype.fields[name][1] for name in dtype.names] == [0, 4, 8, 12, 16, 20, 24]
    # The COMPLEX twin's is padded, which is why each states its own itemsize.
    assert complex_beta.PARAMS_ITEMSIZE == 64


def test_the_ceiling_is_an_equality_on_both_variants():
    assert beta.PACKED_BINDINGS == MAX_BUFFER_BINDINGS
    for variant in beta.VARIANTS:
        assert beta.shipped_signature_bindings(variant) == MAX_BUFFER_BINDINGS


@pytest.mark.requires_resource("metal_fusion_board")
def test_the_board_files_one_row_in_each_of_this_products_two_cells():
    if not BOARD.is_file():
        from conftest import requires_resource_skip  # noqa: PLC0415

        requires_resource_skip("metal_fusion_board",
                               f"{BOARD} is not in this checkout")
    payload = json.loads(BOARD.read_text(encoding="utf-8"))
    for variant, (update_h, step_d) in beta.VARIANT_CELLS.items():
        rows = [row for row in payload["h_to_d_seam"]["instances"]
                if row["update_H"] == update_h and row["step_D"] == step_d]
        assert len(rows) == 1, (variant, rows)
        assert rows[0]["bucket"] == "buildable_not_built", rows
        assert not rows[0]["withdraw_in_seam"]



@pytest.mark.requires_resource("metal_fusion_board")
def test_the_cells_board_serves_both_cells_from_the_bound_artifact():
    """On the per-cell board every row of BOTH cells is served by this family.

    The dated board above keeps its own assertion (it filed the folded rows as
    buildable_not_built, and that was true of it); this one reads the board that
    prices per cell. The folded cell must be bound to THIS gate's own artifact, and
    the rows it credits are exactly the rows that artifact drove bit-identical.
    """
    if not CELLS_BOARD.is_file():
        from conftest import requires_resource_skip  # noqa: PLC0415

        requires_resource_skip("metal_fusion_board",
                               f"{CELLS_BOARD} is not in this checkout")
    payload = json.loads(CELLS_BOARD.read_text(encoding="utf-8"))
    expected = {"plain": 1, "folded": 1}
    product = payload["products"][beta.FAMILY]
    for variant, (update_h, step_d) in beta.VARIANT_CELLS.items():
        rows = [row for row in payload["h_to_d_seam"]["instances"]
                if row["update_H"] == update_h and row["step_D"] == step_d]
        assert len(rows) == expected[variant], (variant, rows)
        assert all(row["bucket"] == "served" for row in rows), rows
        assert all(row["served_by"] == beta.FAMILY for row in rows), rows
        assert not [row for row in rows if row["withdraw_in_seam"]]
        cell = product["cells"][f"{update_h} -> {step_d}"]
        assert sorted(cell["rows"]) == sorted(row["row"] for row in rows), cell
    assert set(product["cells"]) == {
        f"{u} -> {d}" for u, d in beta.VARIANT_CELLS.values()}
    folded = product["cells"][" -> ".join(beta.VARIANT_CELLS["folded"])]
    assert folded["bound_by"] == "cell"
    assert folded["binding"][1].startswith("metal_beta_real_fused_hd_pair_"), folded
    assert product["admitted"] == 2

# ---------------------------------------------------------------------------
# The variant partition
# ---------------------------------------------------------------------------

def test_the_variant_partition_is_total_and_exclusive():
    unfolded, _pml = _pair()
    folded_fields, _pml2 = _pair(folds=(("Y", 1),), beta_value=0.3)
    assert beta.resolve_variant(unfolded.grid) == "plain"
    assert beta.resolve_variant(folded_fields.grid) == "folded"
    assert beta.resolve_variant(None) is None
    assert beta.resolve_variant(object()) is None


def test_each_variant_resolves_its_codes_through_its_own_resolver():
    fields, pml = _pair(folds=(("Y", 1),), beta_value=0.3)
    folded_codes = beta.resolve_codes("folded", fields.grid, pml)
    plain_codes = beta.resolve_codes("plain", fields.grid, pml)
    assert folded_codes is not None and plain_codes is not None
    assert folded_codes != plain_codes
    assert folded_codes[1] in (MM, MP)
    assert plain_codes[1] == P


# ---------------------------------------------------------------------------
# Beta is a curl-only term, and the sign is the spelling's one degree of freedom
# ---------------------------------------------------------------------------

def test_the_constitutive_lift_is_beta_free_and_the_check_can_fail(monkeypatch):
    body = beta.certified_h_cell()
    for name in beta.BETA_IDENTIFIERS:
        assert name not in body

    poisoned = body.replace("h_cell_result h_cell(",
                            "h_cell_result h_cell(/* beta_plus */", 1)
    monkeypatch.setattr(plain, "h_cell_function", lambda *a, **k: poisoned)
    with pytest.raises(AssertionError, match="beta"):
        beta.certified_h_cell()


def test_the_beta_term_is_subtracted_which_is_the_array_paths_add_of_a_negation():
    """IEEE-754 defines ``a - b`` as ``a + (-b)``, so the two spellings are one.

    That is why the SIGN is armed as a mutation and the SPELLING is not: a gate that
    armed ``-`` versus ``+ (-x)`` would be measuring a tautology, and one that armed
    neither would leave the only real degree of freedom untested.
    """
    _prologue, tail = beta.welded_curl_tail("plain", (P, P, P))
    assert "curl0 = curl0 - (beta_plus * b);" in tail
    assert "curl1 = curl1 - (beta_minus * a);" in tail
    assert "curl2" not in tail.split("curl1 = curl1 - (beta_minus * a);")[1][:60]


def test_the_beta_coefficient_is_this_seams_electric_convention_and_unwidened():
    """Under REAL storage the two conventions coincide -- and that is MEASURED here.

    ``beta_curl_coefficients`` multiplies by ``+/-1j`` only under complex storage
    (stepping.py:798-799), so a real run's magnetic and electric words are the same
    pair. The gate therefore arms the PAIR REVERSED instead of the convention, and this
    test is what licenses that substitution rather than leaving it a choice.
    """
    electric = special_kz.beta_curl_coefficients(0.33, 0.035, magnetic=False,
                                                 complex_storage=False)
    magnetic = special_kz.beta_curl_coefficients(0.33, 0.035, magnetic=True,
                                                 complex_storage=False)
    assert electric == magnetic
    assert electric[0] == -electric[1]

    source = Path(beta.__file__).read_text(encoding="utf-8")
    call = source.split("special_kz.beta_curl_coefficients(", 1)[1].split(")", 1)[0]
    assert "float(" not in call, call
    assert "grid.beta" in call and "grid.dt" in call
    assert "magnetic=False, complex_storage=False" in source


# ---------------------------------------------------------------------------
# The lift, on both variants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variant,codes", SPECIALISATIONS)
def test_the_curl_lift_moves_exactly_the_nine_magnetic_loads(variant, codes):
    parent = beta.beta_real_curl_source(variant, codes)
    body = parent.split(plain._BODY_ANCHOR, 1)[1][: -len("}\n")]
    prologue, tail = beta.welded_curl_tail(variant, codes)
    fused = set((prologue + tail).splitlines())
    moved = [line for line in body.splitlines() if line.strip() and line not in fused]
    assert len(moved) == 9, moved
    assert sum(1 for line in moved if " ? g" in line) == 6


@pytest.mark.parametrize("variant,codes", SPECIALISATIONS)
def test_no_magnetic_pointer_survives_in_the_curl_half(variant, codes):
    _prologue, tail = beta.welded_curl_tail(variant, codes)
    for target in range(3):
        assert f"g{target}[" not in tail, target
    assert tail.count("h_cell(") == 6


@pytest.mark.parametrize("variant,codes", SPECIALISATIONS)
def test_the_ghost_is_the_exact_metallic_literal(variant, codes):
    _prologue, tail = beta.welded_curl_tail(variant, codes)
    ghosts = [line for line in tail.splitlines() if " ? h_cell(" in line]
    assert len(ghosts) == 6
    for line in ghosts:
        assert line.endswith(f" : {beta._GHOST};"), line


@pytest.mark.parametrize("variant,codes", SPECIALISATIONS)
def test_the_beta_insert_survives_the_lift_character_for_character(variant, codes):
    _prologue, tail = beta.welded_curl_tail(variant, codes)
    for line in special_kz._REAL_BETA_INSERT.splitlines():
        assert line in tail, line


@pytest.mark.parametrize("codes", [(P, MP, P), (M, MM, P), (MM, MM, MM)])
def test_the_folded_variant_is_the_plain_template_plus_the_top_mask(codes):
    derived = folded_beta.folded_beta_curl_template()
    assert derived.count("__TOP_MASK__") == 1
    assert special_kz._BETA_CURL_TEMPLATE in derived.replace(
        folded_beta._TOP_MASK_BLOCK, "")


def test_the_prologue_cut_is_at_the_decode_and_declares_what_h_cell_needs():
    """The real beta template puts ``int nxi`` ABOVE the decode; the complex one below.

    That is why this weld cuts at ``DECODE_END`` and its complex twin cuts at
    ``int nyz``, and the cut is only correct while the prologue still declares every
    name the ``h_cell`` call takes -- which the lift asserts and this test drives.
    """
    prologue, _tail = beta.welded_curl_tail("plain", (P, P, P))
    assert "int nxi = int(nx);" in prologue
    assert "int nyi = int(ny), nzi = int(nz);" in prologue
    assert prologue.rstrip().endswith("int i   = plane / nyi;")


def test_the_two_variants_emit_different_sources_and_the_same_signature():
    plain_source = beta.beta_real_fused_hd_pair_source("plain", (P, P, P))
    folded_source = beta.beta_real_fused_hd_pair_source("folded", (P, MP, P))
    assert plain_source != folded_source
    for source in (plain_source, folded_source):
        signature = source.split(f"kernel void {beta.KERNEL}(", 1)[1]
        signature = signature.split("uint idx [[thread_position_in_grid]])", 1)[0]
        assert signature.count("[[buffer(") == MAX_BUFFER_BINDINGS


def test_both_halves_read_one_sub_lattice_which_is_what_makes_thirty_pointers():
    from meep_gpu.metal_kernels.launch import SUB_STEPS  # noqa: PLC0415

    assert SUB_STEPS["step_D"]["suffix"] == ""
    assert CONSTITUTIVE_SIDES["H"]["half_integer"] is False


def test_the_enumeration_covers_both_variants_and_no_unfolded_folded_label():
    labels = beta.enumerate_sources()
    assert any(label.startswith(f"{beta.KERNEL}/plain/") for label in labels)
    assert any(label.startswith(f"{beta.KERNEL}/folded/") for label in labels)
    for label in labels:
        if "/folded/" not in label:
            continue
        codes = [int(character) for character in label.rsplit("/", 1)[1]]
        assert {MM, MP} & set(codes), label


# ---------------------------------------------------------------------------
# The predicate, both directions on both inverted clauses
# ---------------------------------------------------------------------------

@pytest.fixture()
def flush_policy(monkeypatch):
    monkeypatch.setenv("MEEP_GPU_SUBNORMAL_POLICY", "flush")
    return "flush"


@pytest.mark.parametrize("folds", [(), (("Y", 1),)])
def test_the_predicate_admits_both_of_this_products_cells(flush_policy, folds):
    fields, pml = _pair(folds=folds, beta_value=0.3)
    verdict = beta.metal_beta_real_fused_hd_pair_coverage(
        fields, pml, (), _FakeResidency())
    assert verdict.covered, list(verdict.reasons)


def test_a_beta_free_real_grid_is_refused_naming_beta(flush_policy):
    fields, pml = _pair(beta_value=0.0)
    verdict = beta.metal_beta_real_fused_hd_pair_coverage(
        fields, pml, (), _FakeResidency())
    assert not verdict.covered
    assert any("beta" in reason for reason in verdict.reasons)


def test_the_plain_real_product_admits_what_this_one_refuses(flush_policy):
    """THE OTHER DIRECTION of the beta clause."""
    fields, pml = _pair(beta_value=0.0)
    verdict = plain.metal_fused_hd_pair_coverage(fields, pml, (), _FakeResidency())
    assert verdict.covered, list(verdict.reasons)


def test_complex_storage_is_refused_naming_the_other_product(flush_policy):
    fields, pml = _pair(complex_storage=True)
    verdict = beta.metal_beta_real_fused_hd_pair_coverage(
        fields, pml, (), _FakeResidency())
    assert not verdict.covered
    assert any("complex" in reason or "real" in reason for reason in verdict.reasons)


def test_an_undeclared_source_list_is_refused_by_name(flush_policy):
    fields, pml = _pair()
    verdict = beta.metal_beta_real_fused_hd_pair_coverage(
        fields, pml, None, _FakeResidency())
    assert not verdict.covered
    assert any("not declared" in reason for reason in verdict.reasons)


def test_a_standing_integrated_electric_withdraw_is_refused_by_name(flush_policy):
    class _Source:
        field_type = "electric"
        is_integrated = True
        _n_source_points = 3

        def withdraw(self, fields):  # noqa: ANN001, ARG002
            return None

    fields, pml = _pair()
    verdict = beta.metal_beta_real_fused_hd_pair_coverage(
        fields, pml, (_Source(),), _FakeResidency())
    assert not verdict.covered
    joined = " | ".join(verdict.reasons)
    assert "HOISTS_THE_WITHDRAW = False" in joined
    assert "INSTALLABLE = False" in joined


def test_a_grid_with_no_fold_answer_is_refused_rather_than_guessed(flush_policy):
    class _Fields:
        grid = None

    verdict = beta.metal_beta_real_fused_hd_pair_coverage(
        _Fields(), None, (), _FakeResidency())
    assert not verdict.covered
    assert any("resolves its variant" in reason for reason in verdict.reasons)


# ---------------------------------------------------------------------------
# The rotation, on a host with no device
# ---------------------------------------------------------------------------

def test_the_rotation_refuses_an_aliased_pair_rather_than_launching():
    class _Fields:
        pass

    fields = _Fields()
    twin = {}
    for name in beta.ROTATED_NAMES:
        array = np.zeros(4, dtype=np.float32)
        setattr(fields, name, array)
        twin[name] = array

    plan = weld.ScratchWeldPairPlan(
        beta.FAMILY, _FakeResidency(), fields, twin, {}, beta.ROTATED_NAMES,
        (), (), (2, 2, 1), (0, 0, 0), (0, 0, 0), (), beta.REPLACES)
    with pytest.raises(RuntimeError, match="ONE tensor"):
        plan._resolve()


# ---------------------------------------------------------------------------
# The device-gated smoke check
# ---------------------------------------------------------------------------

@needs_mps
@pytest.mark.parametrize("variant,codes", SPECIALISATIONS)
def test_every_shipped_specialisation_compiles_on_this_host(variant, codes):
    from meep_gpu.metal_kernels.device import compile_source  # noqa: PLC0415

    os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")
    library = compile_source(beta.beta_real_fused_hd_pair_source(variant, codes))
    assert hasattr(library, beta.KERNEL)


@needs_mps
def test_the_unpacked_thirty_seven_binding_signature_does_not_compile():
    from meep_gpu.metal_kernels.device import compile_source  # noqa: PLC0415

    with pytest.raises(Exception):
        compile_source(beta.refuted_separate_scalar_source())
