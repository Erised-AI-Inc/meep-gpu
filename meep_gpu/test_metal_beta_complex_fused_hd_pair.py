"""Laptop tests for the COMPLEX BLOCH-BETA Metal H->D weld, BOTH variants.

THE MERGE BAR, NOT THE CERTIFICATION. The byte claim is a device gate that runs
complete driver steps and compares uint32 words
(``parity/meep_gpu/gate_metal_beta_complex_fused_hd_pair.py``). These tests are what
must stay green on every change, and they are chosen for what a byte gate would catch
late or not at all:

* **THE VARIANT PARTITION.** This is ONE product covering TWO board cells, and that is
  only sound while no grid can be admitted by both emitters. ``resolve_variant`` reads
  the fold off the grid; the tests assert the partition is total, exclusive, and that
  the PLAN and the PREDICATE resolve it the same way -- a predicate that admitted on
  one variant while the plan built the other is the silent failure this shape can make;
* **BETA IS A CURL-ONLY TERM**, which is the premise that lets the certified complex
  ``update_H`` be welded in unchanged. ``certified_h_cell`` asserts it at build time
  and the test drives that assertion, including the negative: a planted beta
  identifier in the constitutive body must RAISE;
* **the beta coefficient's provenance.** ``beta`` and ``dt`` are passed AS GIVEN, with
  no ``float()`` widening, because widening the chain changes the float32 word on 37%
  of random draws; and this seam's curl is ``step_D``, so the ELECTRIC convention
  (``-1j``) is the right one. Both are asserted against ``special_kz``'s own function;
* **the packed record.** Five ``float2`` members then five scalars, ``itemsize`` 64
  declared explicitly because the payload is 60 -- a record four bytes short of what
  the shader may address is a plausible number rather than a crash;
* **the ghost spelling**, which is this family's OWN: the certified beta complex curl
  hoists ``const float2 zero2`` where the non-beta one spells the literal, and the lift
  REQUIRES its own spelling rather than accepting either;
* **the lift**, reconstructed independently and diffed line-wise on both variants;
* **the predicate's refusals**, every one answerable without a GPU, in both directions
  on the two inverted clauses (beta, and storage).

Everything that needs a device is guarded and skipped.

WHAT THIS FILE DOES NOT CLAIM. Nothing here is a byte-identity measurement and nothing
here licenses installing the product. The cells' numbers -- 1 row at
``(special_kz complex beta)`` and 3 at ``(folded beta complex)``, 0 with a standing
in-seam withdraw -- come from the Metal fusion board and are asserted against it.
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
    beta_complex_fused_hd_pair as beta,
    complex_fields,
    complex_fused_hd_pair as plain,
    folded_beta,
    shaders,
    special_kz,
    symmetry,
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

ARM = "NAIVE"

#: ``(variant, codes)`` the emission tests sweep. Both variants, both terminations,
#: a live non-folded wall and the three-axis fold.
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


def _pair(*, folds=(), beta_value=0.33, complex_storage=True, boundaries=None,
          k_point=(0.0, 0.0, 0.0)):
    size = [2.0, 2.1, 0.0]
    for axis, _phase in folds:
        size["XYZ".index(axis)] = 2.0
    if folds:
        size[0] = 1.6
    grid = Grid(resolution=10.0, cell_size=tuple(size), dimensions=2, courant=0.35,
                symmetry=tuple(Mirror(axis, int(phase)) for axis, phase in folds),
                boundaries=boundaries, beta=beta_value, k_point=tuple(k_point),
                xp=np)
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


def _probe():
    record = special_kz.load_expansion_probe()
    if record is None:
        from conftest import requires_resource_skip  # noqa: PLC0415

        requires_resource_skip("metal_beta_expansion_probe",
                               "no beta expansion probe artifact on this host")
    return record


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
    assert ("metal_beta_complex_fused_hd_pair_device_gate"
            in metal_launch.load_fingerprints()), (
        "WELD_OWED is empty and fingerprints.json carries no "
        "metal_beta_complex_fused_hd_pair_device_gate: one of the two is false")
    assert "gate_metal_beta_complex_fused_hd_pair.py" in beta._RETIRED_WELD_OWED
    assert "fingerprints.json" in beta._RETIRED_WELD_OWED


def test_one_arm_row_serves_two_cells_and_it_is_unwired_and_gated_on_beta():
    """ONE ROW FOR TWO CELLS is what "one product" means at the arm table."""
    rows = [row for row in arms.registered() if row.family == beta.FAMILY]
    assert len(rows) == 1
    row = rows[0]
    assert row.slot == beta.SLOT
    assert row.wired is False
    assert tuple(row.replaces) == beta.REPLACES
    assert row.gate is beta._has_beta
    assert set(beta.VARIANT_CELLS) == set(beta.VARIANTS)


def test_the_pointer_half_of_the_signature_is_the_plain_complex_products():
    """IMPORTED where it is the same and RE-DERIVED where it is not.

    The three POINTER-side refuted signatures are the plain complex product's, because
    the pointer half IS its; ``SEPARATE_SCALAR_BINDINGS`` is this product's own, and it
    is 45 rather than 38 -- four for the beta words and three for the phases, which the
    certified beta curl binds as six separate floats rather than three ``float2``.
    """
    for name in ("PACKED_BINDINGS", "ONE_MORE_POINTER_BINDINGS",
                 "UNSHARED_KMS_BINDINGS", "SPLIT_PLANE_BINDINGS"):
        assert getattr(beta, name) == getattr(plain, name), name
    assert beta.SEPARATE_SCALAR_BINDINGS == 45
    assert beta.SEPARATE_SCALAR_BINDINGS == plain.SEPARATE_SCALAR_BINDINGS + 7
    assert beta.ROTATED_NAMES is plain.ROTATED_NAMES
    assert beta.PARAMS_ITEMSIZE == 64


def test_the_ceiling_is_an_equality_on_both_variants():
    assert beta.PACKED_BINDINGS == MAX_BUFFER_BINDINGS
    for variant in beta.VARIANTS:
        assert beta.shipped_signature_bindings(variant, ARM) == MAX_BUFFER_BINDINGS


def test_the_packed_record_declares_sixty_four_bytes_for_sixty_of_payload():
    """The itemsize is DECLARED because the natural record is four bytes short."""
    dtype = beta.params_record_dtype()
    assert dtype.itemsize == beta.PARAMS_ITEMSIZE == 64
    assert dtype.names[:5] == ("px", "py", "pz", "bp", "bm")
    assert [dtype.fields[name][1] for name in dtype.names] == [
        0, 8, 16, 24, 32, 40, 44, 48, 52, 56]


@pytest.mark.requires_resource("metal_fusion_board")
def test_the_board_files_four_rows_across_this_products_two_cells():
    if not BOARD.is_file():
        from conftest import requires_resource_skip  # noqa: PLC0415

        requires_resource_skip("metal_fusion_board",
                               f"{BOARD} is not in this checkout")
    payload = json.loads(BOARD.read_text(encoding="utf-8"))
    per_cell = {}
    for variant, (update_h, step_d) in beta.VARIANT_CELLS.items():
        rows = [row for row in payload["h_to_d_seam"]["instances"]
                if row["update_H"] == update_h and row["step_D"] == step_d]
        per_cell[variant] = rows
        assert all(row["bucket"] == "buildable_not_built" for row in rows), rows
        assert not [row for row in rows if row["withdraw_in_seam"]]
    assert len(per_cell["plain"]) == 1, per_cell["plain"]
    assert len(per_cell["folded"]) == 3, per_cell["folded"]



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
    expected = {"plain": 1, "folded": 3}
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
    assert folded["binding"][1].startswith("metal_beta_complex_fused_hd_pair_"), folded
    assert product["admitted"] == 4

# ---------------------------------------------------------------------------
# The variant partition — what makes this ONE product rather than two
# ---------------------------------------------------------------------------

def test_the_variant_partition_is_total_and_exclusive():
    unfolded, _pml = _pair()
    folded_fields, _pml2 = _pair(folds=(("Y", 1),), beta_value=0.3)
    assert beta.resolve_variant(unfolded.grid) == "plain"
    assert beta.resolve_variant(folded_fields.grid) == "folded"
    # A GRID THAT CANNOT BE ASKED IS A REFUSAL, never a default.
    assert beta.resolve_variant(None) is None
    assert beta.resolve_variant(object()) is None


def test_each_variant_resolves_its_codes_through_its_own_resolver():
    """``folded_axis_kinds`` on a fold, ``_boundary_kinds`` off it -- and they DIFFER.

    Using the plain expression on a folded grid maps ``"mirror"`` to 0 = PERIODIC: the
    backward ghost becomes a WRAP and the cell-0 mask is not widened. Both codes are
    valid, so nothing downstream can catch it -- which is why the two are compared here
    and the mistake is a gate mutation.
    """
    fields, pml = _pair(folds=(("Y", 1),), beta_value=0.3)
    folded_codes = beta.resolve_codes("folded", fields.grid, pml)
    plain_codes = beta.resolve_codes("plain", fields.grid, pml)
    assert folded_codes is not None and plain_codes is not None
    assert folded_codes != plain_codes
    assert folded_codes[1] in (MM, MP)
    assert plain_codes[1] == P


def test_the_predicate_and_the_plan_resolve_the_same_variant(monkeypatch):
    """The silent failure this one-product shape can make, closed by construction."""
    monkeypatch.setenv("MEEP_GPU_SUBNORMAL_POLICY", "flush")
    probe = _probe()
    for folds, expected in (((), "plain"), ((("Y", 1),), "folded")):
        fields, pml = _pair(folds=folds, beta_value=0.3)
        verdict = beta.metal_beta_complex_fused_hd_pair_coverage(
            fields, pml, (), _FakeResidency(), probe)
        assert verdict.covered, list(verdict.reasons)
        assert beta.resolve_variant(fields.grid) == expected
        assert all(expected in reason or True for reason in verdict.reasons)


# ---------------------------------------------------------------------------
# Beta is a curl-only term
# ---------------------------------------------------------------------------

def test_the_constitutive_lift_is_beta_free_and_the_check_can_fail(monkeypatch):
    """The premise, ASSERTED at build time -- and the assertion shown to bite."""
    body = beta.certified_h_cell(ARM)
    for name in beta.BETA_IDENTIFIERS:
        assert name not in body

    # THE NEGATIVE. A certified constitutive that grew a beta dependence must fail the
    # BUILD rather than step a beta run with a beta-free magnetic field.
    poisoned = body.replace("h_cell_result h_cell(",
                            "h_cell_result h_cell(/* bpr */", 1)
    monkeypatch.setattr(plain, "h_cell_function", lambda *a, **k: poisoned)
    with pytest.raises(AssertionError, match="beta"):
        beta.certified_h_cell(ARM)


def test_the_beta_coefficient_is_this_seams_electric_convention():
    """``step_D`` takes ``-1j``; the magnetic convention is a smooth wrong answer."""
    electric = special_kz.beta_curl_coefficients(0.33, 0.035, magnetic=False,
                                                 complex_storage=True)
    magnetic = special_kz.beta_curl_coefficients(0.33, 0.035, magnetic=True,
                                                 complex_storage=True)
    assert electric != magnetic
    # The plan asks with magnetic=False; a test that only asserted "it calls the
    # function" would not see the argument.
    source = Path(beta.__file__).read_text(encoding="utf-8")
    assert "magnetic=False, complex_storage=True" in source


def test_beta_and_dt_reach_the_coefficient_unwidened():
    """No ``float()`` normalisation: widening the chain moves the float32 word.

    ``special_kz`` measured 14,926 of 40,000 random draws differing when the chain is
    widened first, and the path is live -- ``Grid`` coerces ``beta`` but nothing
    coerces ``dt``.
    """
    source = Path(beta.__file__).read_text(encoding="utf-8")
    call = source.split("special_kz.beta_curl_coefficients(", 1)[1].split(")", 1)[0]
    assert "float(" not in call, call
    assert "grid.beta" in call and "grid.dt" in call


# ---------------------------------------------------------------------------
# The lift, on both variants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variant,codes", SPECIALISATIONS)
def test_the_curl_lift_moves_exactly_the_nine_magnetic_loads(variant, codes):
    parent = beta.beta_complex_curl_source(variant, codes, (0, 0, 0), ARM)
    body = parent.split(plain._BODY_ANCHOR, 1)[1][: -len("}\n")]
    prologue, tail = beta.welded_curl_tail(variant, codes, (0, 0, 0), ARM)
    fused = set((prologue + tail).splitlines())
    moved = [line for line in body.splitlines() if line.strip() and line not in fused]
    assert len(moved) == 9, moved
    assert sum(1 for line in moved if " ? g" in line) == 6


@pytest.mark.parametrize("variant,codes", SPECIALISATIONS)
def test_no_magnetic_pointer_survives_in_the_curl_half(variant, codes):
    _prologue, tail = beta.welded_curl_tail(variant, codes, (0, 0, 0), ARM)
    for target in range(3):
        assert f"g{target}[" not in tail, target
    assert tail.count("h_cell(") == 6


@pytest.mark.parametrize("variant,codes", SPECIALISATIONS)
def test_the_ghost_is_this_familys_own_hoisted_spelling(variant, codes):
    """``zero2``, not the literal -- and the lift REQUIRES it rather than accepting both.

    A laxer check would pass on a parent that changed its ghost spelling, leaving six
    taps redirected past a DIFFERENT ghost.
    """
    _prologue, tail = beta.welded_curl_tail(variant, codes, (0, 0, 0), ARM)
    assert beta._GHOST_DECLARATION in tail
    ghosts = [line for line in tail.splitlines() if " ? h_cell(" in line]
    assert len(ghosts) == 6
    for line in ghosts:
        assert line.endswith(f" : {beta._GHOST};"), line


@pytest.mark.parametrize("variant,codes", SPECIALISATIONS)
def test_the_beta_insert_survives_the_lift_character_for_character(variant, codes):
    _prologue, tail = beta.welded_curl_tail(variant, codes, (0, 0, 0), ARM)
    for line in special_kz._COMPLEX_BETA_INSERT.splitlines():
        assert line in tail, line


@pytest.mark.parametrize("codes", [(P, MP, P), (M, MM, P), (MM, MM, MM)])
def test_the_folded_variant_is_the_plain_template_plus_the_top_mask(codes):
    """``folded_beta_bloch_curl_template`` is the parent with ONE slot inserted."""
    derived = folded_beta.folded_beta_bloch_curl_template()
    assert derived.count("__TOP_MASK__") == 1
    assert special_kz._BETA_BLOCH_CURL_TEMPLATE in derived.replace(
        folded_beta._TOP_MASK_BLOCK, "")


def test_the_two_variants_emit_different_sources_and_the_same_signature():
    plain_source = beta.beta_complex_fused_hd_pair_source(
        "plain", (P, P, P), (0, 0, 0), ARM)
    folded_source = beta.beta_complex_fused_hd_pair_source(
        "folded", (P, MP, P), (0, 0, 0), ARM)
    assert plain_source != folded_source
    for source in (plain_source, folded_source):
        signature = source.split(f"kernel void {beta.KERNEL}(", 1)[1]
        signature = signature.split("uint idx [[thread_position_in_grid]])", 1)[0]
        assert signature.count("[[buffer(") == MAX_BUFFER_BINDINGS


def test_both_halves_read_one_sub_lattice_which_is_what_makes_thirty_pointers():
    from meep_gpu.metal_kernels.launch import SUB_STEPS  # noqa: PLC0415

    assert SUB_STEPS["step_D"]["suffix"] == ""
    assert CONSTITUTIVE_SIDES["H"]["half_integer"] is False


def test_the_enumeration_never_offers_a_phase_on_a_non_periodic_axis():
    labels = beta.enumerate_sources(ARM)
    assert labels
    for label in labels:
        codes = [int(character) for character in label.split("/")[2]]
        flags = [int(character) for character in label.split("/")[3][2:]]
        for code, flag in zip(codes, flags):
            assert not (flag and code != P), label


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
    verdict = beta.metal_beta_complex_fused_hd_pair_coverage(
        fields, pml, (), _FakeResidency(), _probe())
    assert verdict.covered, list(verdict.reasons)


def test_a_beta_free_complex_grid_is_refused_naming_beta(flush_policy):
    fields, pml = _pair(beta_value=0.0)
    verdict = beta.metal_beta_complex_fused_hd_pair_coverage(
        fields, pml, (), _FakeResidency(), _probe())
    assert not verdict.covered
    assert any("beta" in reason for reason in verdict.reasons)


def test_the_plain_complex_product_admits_what_this_one_refuses(flush_policy):
    """THE OTHER DIRECTION of the beta clause."""
    fields, pml = _pair(beta_value=0.0)
    verdict = plain.metal_complex_fused_hd_pair_coverage(
        fields, pml, (), _FakeResidency(), complex_fields.load_expansion_probe())
    assert verdict.covered, list(verdict.reasons)


def test_real_storage_is_refused_naming_the_other_product(flush_policy):
    fields, pml = _pair(complex_storage=False)
    verdict = beta.metal_beta_complex_fused_hd_pair_coverage(
        fields, pml, (), _FakeResidency(), _probe())
    assert not verdict.covered
    assert any("complex64" in reason or "real" in reason
               for reason in verdict.reasons)


def test_an_undeclared_source_list_is_refused_by_name(flush_policy):
    fields, pml = _pair()
    verdict = beta.metal_beta_complex_fused_hd_pair_coverage(
        fields, pml, None, _FakeResidency(), _probe())
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
    verdict = beta.metal_beta_complex_fused_hd_pair_coverage(
        fields, pml, (_Source(),), _FakeResidency(), _probe())
    assert not verdict.covered
    joined = " | ".join(verdict.reasons)
    assert "HOISTS_THE_WITHDRAW = False" in joined
    assert "INSTALLABLE = False" in joined


def test_a_grid_with_no_fold_answer_is_refused_rather_than_guessed(flush_policy):
    class _Fields:
        grid = None

    verdict = beta.metal_beta_complex_fused_hd_pair_coverage(
        _Fields(), None, (), _FakeResidency(), _probe())
    assert not verdict.covered
    assert any("resolves its variant" in reason for reason in verdict.reasons)


# ---------------------------------------------------------------------------
# The device-gated smoke check
# ---------------------------------------------------------------------------

@needs_mps
@pytest.mark.parametrize("variant,codes", SPECIALISATIONS)
def test_every_shipped_specialisation_compiles_on_this_host(variant, codes):
    from meep_gpu.metal_kernels.device import compile_source  # noqa: PLC0415

    os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")
    library = compile_source(
        beta.beta_complex_fused_hd_pair_source(variant, codes, (0, 0, 0), ARM))
    assert hasattr(library, beta.KERNEL)


@needs_mps
def test_the_unpacked_forty_five_binding_signature_does_not_compile():
    """45 -> 31 is what makes room for a weld on a curl already at 30 of 31."""
    from meep_gpu.metal_kernels.device import compile_source  # noqa: PLC0415

    with pytest.raises(Exception):
        compile_source(beta.refuted_separate_scalar_source())
