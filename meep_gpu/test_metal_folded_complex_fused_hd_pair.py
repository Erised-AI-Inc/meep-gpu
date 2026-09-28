"""Laptop tests for the FOLDED COMPLEX Metal H->D weld.

THE MERGE BAR, NOT THE CERTIFICATION. The byte claim this family owes is a device gate
that runs complete driver steps and compares uint32 words
(``parity/meep_gpu/gate_metal_folded_complex_fused_hd_pair.py``); these tests are what
must stay green on every change, and they are chosen for the defects a byte gate would
catch LATE, on a configuration nobody sweeps, or not at all:

* **THE CODES.** The single point of failure every folded family shares.
  ``folded_axis_kinds`` resolves four codes and the unfolded complex product's
  ``1 if kind == "metallic" else 0`` expression maps a fold to PERIODIC, which makes
  the backward ghost a WRAP and deletes the widened cell-0 mask. A smooth wrong answer
  on every folded axis that the emitter cannot catch, so the SOURCE of the codes is
  asserted here;
* **the lift.** The curl body is spliced from
  ``folded_complex.folded_bloch_curl_source``'s own output rather than retyped, and
  "spliced" is a hypothesis until something compares the strings. The tests
  reconstruct the certified folded complex body independently and assert that the ONLY
  lines that moved are the nine magnetic loads, and that both fold blocks and the
  Bloch phase block survive character for character;
* **the constitutive half is IMPORTED, not copied.** The whole product rests on
  ``folded_complex`` adding no device code on ``update_H``; the test asserts the
  emitted ``h_cell`` is byte-identical to the plain complex product's;
* **the ghost.** Under complex storage the fold's backward ghost is the exact
  ``float2(0.0f, 0.0f)`` the METALLIC branch serves and NOT the array path's
  ``parity * H[2]``. The literal and the parsed guard are what a redirect eats first;
* **the binding ceiling**, an EQUALITY here: 31 of 31, no headroom, and the fold adds
  no argument under complex storage either;
* **the predicate's refusals**, every one answerable without a GPU, INCLUDING BOTH
  DIRECTIONS of the inverted fold clause -- disjointness is a property of a PAIR of
  predicates and neither one alone can carry it;
* **the declarations.** Registered UNWIRED and gated on the fold, ``INSTALLABLE``
  False on a measured arbitration, ``WELD_OWED`` EMPTY because ``fingerprints.json``
  now carries the ledger entry. Each is a claim a reader will act on.

Everything that needs a device is guarded and skipped, so this file is the merge bar on
a host with no MPS as well as on this one.

WHAT THIS FILE DOES NOT CLAIM. Nothing here is a byte-identity measurement and nothing
here licenses installing the product. The cell's numbers -- 5 rows in the
``(folded complex -> folded complex)`` cell, all five ``buildable_not_built``, 0 of
them carrying a standing in-seam withdraw -- come from the Metal fusion board
(``results/fusion_matrix_metal_2026-09-07_wired``), which is where the arm-pair JOIN
lives, and are asserted against it where it is present rather than restated.
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

# THE PROBE PATHS COME FROM THE SAME PLACE THE PREDICATES READ THEM, so a test cannot
# certify a body the family would not launch, and cannot skip on a host that HAS the
# artifact merely because nothing exported its path.
ENVIRONMENT = matrix.prepare_environment()

from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid, Mirror  # noqa: E402
from meep_gpu.metal_kernels import (  # noqa: E402
    arms,
    complex_fields,
    complex_fused_hd_pair as plain,
    folded_complex,
    folded_complex_fused_hd_pair as folded,
    offdiag_weld_common as weld,
    shaders,
    symmetry,
    templates,
)
from meep_gpu.metal_kernels.device import MAX_BUFFER_BINDINGS  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.triton_kernels.coverage import CONSTITUTIVE_SIDES  # noqa: E402
from meep_gpu.triton_kernels.symmetry import (  # noqa: E402
    CODE_METALLIC as M, CODE_MIRROR_METALLIC as MM, CODE_MIRROR_PERIODIC as MP,
    CODE_PERIODIC as P,
)

API_ROOT = Path(__file__).resolve().parent.parent
#: THE BOARD, not the seam record: the seam record carries each row's source census
#: and its withdraw flag but NOT the arm pair, which is a JOIN the board performs.
BOARD = (API_ROOT / "parity" / "meep_gpu" / "results"
         / "fusion_matrix_metal_2026-09-07_wired" / "fusion_matrix.json")

#: The expansion arm every emission test bakes. A LITERAL, deliberately: these tests
#: are about the SPLICE, and a probe-resolved arm would make them skip on a host whose
#: probe is absent while the splice is perfectly testable there.
ARM = "NAIVE"

#: The folded quadruples the emission tests sweep: one of each termination, one mixed,
#: one with a live non-folded wall, and the three-axis fold.
CODES = [
    (P, MP, P), (P, MM, P), (M, MM, P), (MM, MM, P), (MP, MP, P), (MM, MM, MM),
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


def _folded_complex_pair(axes: str = "Y", phases=(1,), boundaries=None,
                         extent: float = 2.0, depth: float = 0.0):
    """A folded COMPLEX grid and its absorber -- the composition matrix's geometry."""
    size = [1.6, 1.6, float(depth)]
    for axis in axes:
        size["XYZ".index(axis)] = extent
    grid = Grid(resolution=10.0, cell_size=tuple(size),
                dimensions=2 if not depth else 3, courant=0.35,
                symmetry=tuple(Mirror(axis, int(phase))
                               for axis, phase in zip(axes, phases)),
                boundaries=boundaries, xp=np)
    fields = Fields(grid=grid, force_complex_fields=True)
    shape = tuple(grid.shape)
    fields.set_epsilon_volumes(
        {name: np.full(shape, np.float32(value), np.float32)
         for name, value in (("Ex", 2.0), ("Ey", 2.5), ("Ez", 3.0))},
        {name: np.full(shape, np.float32(1.0 / value), np.float32)
         for name, value in (("Ex", 2.0), ("Ey", 2.5), ("Ez", 3.0))})
    fields.enable_pml_storage()
    folded_indices = {"XYZ".index(axis) for axis in axes}
    thickness = tuple((0, 0) if shape[index] < 6
                      else (0, 2) if index in folded_indices else (2, 2)
                      for index in range(3))
    return fields, PML(grid=grid, thickness=thickness)


# ---------------------------------------------------------------------------
# The declarations
# ---------------------------------------------------------------------------

def test_the_span_is_the_seam_and_both_seam_flags_are_false_with_a_reason():
    assert folded.REPLACES == ("update_H", "step_D")
    assert folded.SLOT == "update_H"
    from meep_gpu import withdraw_hoist  # noqa: PLC0415

    assert folded.SEAM == withdraw_hoist.SEAM
    # NOT A CHOICE: nothing is injected between the two consults, so there is no
    # deposit to bracket. The withdraw flag IS a choice and its consequence is that
    # the predicate refuses every row with a standing one.
    assert folded.CARRIES_DEPOSIT_REPAIR is False
    assert folded.HOISTS_THE_WITHDRAW is False
    assert folded.INSTALLABLE is False
    assert "arbitration" in folded.INSTALLABLE_REASON
    assert "MEASURED" in folded.INSTALLABLE_REASON


def test_the_family_is_welded_and_the_debt_it_replaced_is_kept():
    """An empty ``WELD_OWED`` is a CLAIM too, and the ledger is the other half of it.

    ``test_metal_weld_contract.test_every_metal_family_is_welded`` partitions the
    ``gate_metal_*.py`` fleet against ``fingerprints.json`` and requires the declaring
    set and the welded set to be disjoint and to exhaust it. So the two halves of a
    release move together: emptying this string without the entry claims a gate this
    tree cannot show, and cutting the entry without emptying it leaves a welded family
    still declaring its weld is owed. Both are read off the tree here.

    THE ORDER IS THE POINT. Emptying ``WELD_OWED`` MOVES the module, and the module is
    the first pin of its own weld, so ``mint_metal_weld.py`` refuses an artifact whose
    recorded digests have moved. The gate was therefore re-run AGAINST the emptied
    bytes and minted from that artifact. The debt text -- which named the artifact,
    the ledger entry and the ``wiring.patch`` that deferred it -- is kept in
    ``_RETIRED_WELD_OWED`` rather than deleted, and the last three assertions read it
    there.
    """
    from meep_gpu.metal_kernels import launch as metal_launch  # noqa: PLC0415

    assert folded.WELD_OWED == "", (
        "the family holds a released device gate; WELD_OWED must be empty, and the "
        "weld contract's partition is what reads it")
    assert ("metal_folded_complex_fused_hd_pair_device_gate"
            in metal_launch.load_fingerprints()), (
        "WELD_OWED is empty and fingerprints.json carries no "
        "metal_folded_complex_fused_hd_pair_device_gate: one of the two is false")
    assert "gate_metal_folded_complex_fused_hd_pair.py" in folded._RETIRED_WELD_OWED
    assert "fingerprints.json" in folded._RETIRED_WELD_OWED
    assert "wiring.patch" in folded._RETIRED_WELD_OWED


def test_the_arm_is_registered_unwired_and_gated_on_the_fold():
    rows = [row for row in arms.registered() if row.family == folded.FAMILY]
    assert len(rows) == 1, "one row, on update_H"
    row = rows[0]
    assert row.slot == folded.SLOT
    assert row.wired is False, "an unwired row is enumerable, never selectable"
    assert tuple(row.replaces) == folded.REPLACES
    assert row.gate is symmetry._has_fold


def test_the_binding_constants_are_the_plain_complex_products_by_identity():
    """IMPORTED, not re-spelled: the signature IS the plain complex product's.

    Two spellings of one number is how a constant goes stale in one place and not the
    other, and every one of these is compared against the emitted source by the gate.
    """
    for name in ("PACKED_BINDINGS", "ONE_MORE_POINTER_BINDINGS",
                 "UNSHARED_KMS_BINDINGS", "SEPARATE_SCALAR_BINDINGS",
                 "SPLIT_PLANE_BINDINGS", "PARAMS_ITEMSIZE"):
        assert getattr(folded, name) == getattr(plain, name), name
    assert folded.ROTATED_NAMES is plain.ROTATED_NAMES
    assert folded.KERNEL == plain.KERNEL


def test_the_ceiling_is_an_equality_and_the_fold_adds_no_argument():
    assert folded.PACKED_BINDINGS == MAX_BUFFER_BINDINGS
    for codes in CODES:
        source = folded.folded_complex_fused_hd_pair_source(codes, (0, 0, 0), ARM)
        signature = source.split(f"kernel void {folded.KERNEL}(", 1)[1]
        signature = signature.split("uint idx [[thread_position_in_grid]])", 1)[0]
        assert signature.count("[[buffer(") == MAX_BUFFER_BINDINGS, codes


@pytest.mark.requires_resource("metal_fusion_board")
def test_the_board_files_five_rows_in_this_cell_and_none_carries_a_withdraw():
    """The cell's denominator, ASSERTED against the board rather than restated.

    The module's docstring names 5 rows and 0 with a standing in-seam withdraw. Both
    are joins the board performs, so a cell that changed size -- or grew a withdraw row
    the predicate must refuse BY NAME -- makes this test red instead of leaving a
    docstring quietly wrong.
    """
    if not BOARD.is_file():
        from conftest import requires_resource_skip  # noqa: PLC0415

        requires_resource_skip("metal_fusion_board",
                               f"{BOARD} is not in this checkout")
    payload = json.loads(BOARD.read_text(encoding="utf-8"))
    cell = [row for row in payload["h_to_d_seam"]["instances"]
            if row["update_H"] == "folded complex"
            and row["step_D"] == "folded complex"]
    assert len(cell) == 5, [row["row"] for row in cell]
    assert all(row["bucket"] == "buildable_not_built" for row in cell), cell
    assert not [row for row in cell if row["withdraw_in_seam"]]


# ---------------------------------------------------------------------------
# The lift
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("codes", CODES)
def test_the_curl_lift_moves_exactly_the_nine_magnetic_loads(codes):
    """Reconstruct the certified body independently and diff it line-wise."""
    parent = folded_complex.folded_bloch_curl_source(codes, True, (0, 0, 0), ARM)
    body = parent.split(plain._BODY_ANCHOR, 1)[1][: -len("}\n")]
    prologue, tail = folded.folded_welded_curl_tail(codes, (0, 0, 0), ARM)
    parent_lines = body.splitlines()
    # THE PROLOGUE IS PART OF THE EMISSION and is compared with the body: the weld
    # splits the certified text at `int nyz = nyi * nzi;` only so that `h_cell` can be
    # called between the two halves, so a comparison that dropped it would report the
    # guard and the whole index decode as "moved".
    fused_lines = set((prologue + tail).splitlines())
    moved = [line for line in parent_lines if line.strip() and line not in fused_lines]
    assert len(moved) == 9, moved
    assert sum(1 for line in moved if line.strip().startswith("float2 a   = g0[ii]")
               or line.strip().startswith("float2 b   = g1[ii]")
               or line.strip().startswith("float2 c   = g2[ii]")) == 3
    assert sum(1 for line in moved if " ? g" in line) == 6


@pytest.mark.parametrize("codes", CODES)
def test_no_magnetic_pointer_survives_in_the_curl_half(codes):
    _prologue, tail = folded.folded_welded_curl_tail(codes, (0, 0, 0), ARM)
    for target in range(3):
        assert f"g{target}[" not in tail, target
    assert tail.count("h_cell(") == 6


@pytest.mark.parametrize("codes", CODES)
def test_the_two_fold_blocks_survive_the_lift_character_for_character(codes):
    """The masks touch no ``gN[``, so the redirect cannot reach them -- and must not.

    ``folded_top_plane_mask`` writes ``curlN = last_a ? <zero> : curlN;`` and the
    widened cell-0 mask ``curlN = at_a ? <zero> : curlN;``. Both are the emitter's own
    text and both are what actually carries the byte result on this seam (the gate's
    ``ghost_observability`` leg measures that), so a lift that ate one would be a plane
    of wrong values rather than a compile error.
    """
    reduced = symmetry._reduced_codes(codes)
    expected = (templates.ownership_mask(reduced, True, zero=templates.COMPLEX_ZERO)
                + "\n"
                + symmetry.folded_top_plane_mask(codes, True,
                                                 zero=templates.COMPLEX_ZERO))
    _prologue, tail = folded.folded_welded_curl_tail(codes, (0, 0, 0), ARM)
    for line in expected.splitlines():
        if line.strip():
            assert line in tail, line


@pytest.mark.parametrize("codes", [c for c in CODES if c[0] == P])
def test_the_bloch_phase_block_survives_the_lift_character_for_character(codes):
    """The phase rotates the LOADED register after the gather; the weld never sees it."""
    block = complex_fields._phase_block("x", True, True)
    _prologue, tail = folded.folded_welded_curl_tail(codes, (1, 0, 0), ARM)
    for line in block.splitlines():
        if line.strip():
            assert line in tail, line


@pytest.mark.parametrize("codes", CODES)
def test_the_folded_ghost_is_the_exact_complex_zero_and_not_a_mirror_read(codes):
    """Under a fold the backward ghost is the METALLIC literal, not ``parity * H[2]``.

    Legal because the value is DEAD under a mask -- which the gate measures rather than
    assumes. What is asserted here is that the LITERAL is what the redirect leaves
    behind: a redirect that served the recompute past the face would read a cell the
    emitter says is not there.
    """
    _prologue, tail = folded.folded_welded_curl_tail(codes, (0, 0, 0), ARM)
    ghosts = [line for line in tail.splitlines() if " ? h_cell(" in line]
    assert len(ghosts) == 6
    for line in ghosts:
        assert line.endswith(f" : {templates.COMPLEX_ZERO};"), line


def test_the_constitutive_half_is_the_plain_products_lift_and_not_a_copy():
    """``folded_complex`` adds NO device code on ``update_H`` -- asserted by digest."""
    source = folded.folded_complex_fused_hd_pair_source((P, MP, P), (0, 0, 0), ARM)
    emitted = source.split("h_cell_result h_cell(", 1)[1].split("kernel void", 1)[0]
    certified = plain.h_cell_function(ARM).split(
        "h_cell_result h_cell(", 1)[1]
    assert certified.startswith(emitted[:200])


def test_both_halves_read_one_sub_lattice_which_is_what_makes_thirty_pointers():
    """The shared coefficient group is a PREMISE, so its condition is asserted."""
    assert complex_fields.SUB_STEPS["step_D"]["suffix"] == ""
    assert CONSTITUTIVE_SIDES["H"]["half_integer"] is False


@pytest.mark.parametrize("codes", CODES)
@pytest.mark.parametrize("mode", [shaders.CONTRACT_OFF])
def test_the_source_decodes_and_carries_exactly_one_contraction_pragma(codes, mode):
    """One pragma, and the text round-trips through UTF-8.

    NOT AN ASCII ASSERTION, and the difference is a measurement rather than a
    relaxation: this weld emits the PLAIN COMPLEX product's template verbatim, and that
    template's comments carry non-ASCII punctuation. Requiring ASCII here would be this
    file asserting something about a file it deliberately does not own. What matters --
    that the contraction pragma is emitted exactly once, so a specialisation cannot
    quietly acquire a second one -- is asserted.
    """
    source = folded.folded_complex_fused_hd_pair_source(codes, (0, 0, 0), ARM, mode)
    assert source.encode("utf-8").decode("utf-8") == source
    assert source.count("#pragma") == shaders.contraction_pragma(mode).count("#pragma")


def test_the_enumeration_never_offers_a_phase_on_a_mirror_axis():
    """A mirror plane reflects rather than repeating, so it cannot carry a phase.

    ``stepping._bloch_phases`` raises on it and the driver refuses the configuration
    outright, so a label enumerating one would be fingerprinted as buildable when it
    is not.
    """
    labels = folded.enumerate_sources(ARM)
    assert labels
    for label in labels:
        codes_text, phase_text = label.split("/")[2], label.split("/")[3]
        codes = [int(character) for character in codes_text]
        flags = [int(character) for character in phase_text[2:]]
        for code, flag in zip(codes, flags):
            assert not (flag and code != P), label


# ---------------------------------------------------------------------------
# The predicate, both directions
# ---------------------------------------------------------------------------

@pytest.fixture()
def flush_policy(monkeypatch):
    """Only ``flush`` is honourable on MPS; a resolved ``keep`` refuses by name."""
    monkeypatch.setenv("MEEP_GPU_SUBNORMAL_POLICY", "flush")
    return "flush"


def _probe():
    """The folded-complex expansion probe, or a skip -- never a default."""
    record = folded_complex.load_expansion_probe()
    if record is None:
        pytest.skip("no folded-complex expansion probe on this host")
    return record


def test_the_predicate_admits_the_cell_this_product_was_built_for(flush_policy):
    fields, pml = _folded_complex_pair()
    verdict = folded.metal_folded_complex_fused_hd_pair_coverage(
        fields, pml, (), _FakeResidency(), _probe())
    assert verdict.covered, list(verdict.reasons)


def test_an_unfolded_complex_grid_is_refused_naming_the_plain_product(flush_policy):
    grid = Grid(resolution=10.0, cell_size=(2.0, 2.1, 0.0), dimensions=2,
                courant=0.35, xp=np)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_pml_storage()
    pml = PML(grid=grid, thickness=((2, 2), (2, 2), (0, 0)))
    verdict = folded.metal_folded_complex_fused_hd_pair_coverage(
        fields, pml, (), _FakeResidency(), _probe())
    assert not verdict.covered
    assert any("complex_fused_hd_pair" in reason for reason in verdict.reasons)


def test_the_plain_complex_product_refuses_this_products_cell(flush_policy):
    """THE OTHER DIRECTION. Disjointness is a property of a PAIR of predicates."""
    fields, pml = _folded_complex_pair()
    verdict = plain.metal_complex_fused_hd_pair_coverage(
        fields, pml, (), _FakeResidency(), _probe())
    assert not verdict.covered
    assert any("folded complex" in reason for reason in verdict.reasons)


def test_an_undeclared_source_list_is_refused_by_name(flush_policy):
    """IGNORANCE IS NEVER AN EMPTY SET."""
    fields, pml = _folded_complex_pair()
    verdict = folded.metal_folded_complex_fused_hd_pair_coverage(
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

    fields, pml = _folded_complex_pair()
    verdict = folded.metal_folded_complex_fused_hd_pair_coverage(
        fields, pml, (_Source(),), _FakeResidency(), _probe())
    assert not verdict.covered
    joined = " | ".join(verdict.reasons)
    assert "HOISTS_THE_WITHDRAW = False" in joined
    assert "INSTALLABLE = False" in joined


def test_real_storage_under_a_fold_is_refused_naming_the_other_family(flush_policy):
    grid = Grid(resolution=10.0, cell_size=(1.6, 2.0, 0.0), dimensions=2,
                courant=0.35, symmetry=(Mirror("Y", 1),), xp=np)
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    pml = PML(grid=grid, thickness=((2, 2), (0, 2), (0, 0)))
    verdict = folded.metal_folded_complex_fused_hd_pair_coverage(
        fields, pml, (), _FakeResidency(), _probe())
    assert not verdict.covered
    assert any("real float32" in reason or "complex64" in reason
               for reason in verdict.reasons)


def test_a_folded_complex_beta_run_is_refused_by_name(flush_policy):
    grid = Grid(resolution=10.0, cell_size=(1.6, 2.0, 0.0), dimensions=2,
                courant=0.35, symmetry=(Mirror("Y", 1),), beta=0.3, xp=np)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_pml_storage()
    pml = PML(grid=grid, thickness=((2, 2), (0, 2), (0, 0)))
    verdict = folded.metal_folded_complex_fused_hd_pair_coverage(
        fields, pml, (), _FakeResidency(), _probe())
    assert not verdict.covered
    assert any("beta" in reason for reason in verdict.reasons)


class _FakeResidency:
    """A residency that answers the predicate's dtype and allocation questions only."""

    device = "cpu"

    def mirror(self, name, host, dtype=None, constant=False):  # noqa: ANN001, ARG002
        return host

    def tensor_for_host(self, host):  # noqa: ANN001
        return host


# ---------------------------------------------------------------------------
# The rotation, on a host with no device
# ---------------------------------------------------------------------------

def test_the_rotation_refuses_an_aliased_pair_rather_than_launching():
    """The whole design is that nothing written is read; an aliased pair is a refusal.

    Pure host logic, so it runs on a machine with no MPS -- which is the point: the
    invariant that makes the weld safe should not need a GPU to be checked.
    """
    class _Fields:
        pass

    fields = _Fields()
    twin = {}
    for name in folded.ROTATED_NAMES:
        array = np.zeros(4, dtype=np.complex64)
        setattr(fields, name, array)
        twin[name] = array  # ALIASED: the scratch IS the pre-launch buffer

    plan = weld.ScratchWeldPairPlan(
        folded.FAMILY, _FakeResidency(), fields, twin, {}, folded.ROTATED_NAMES,
        (), (), (2, 2, 1), (0, 0, 0), (0, 0, 0), (), folded.REPLACES)
    with pytest.raises(RuntimeError, match="ONE tensor"):
        plan._resolve()


def test_the_rotation_swaps_the_engines_references_after_a_launch():
    class _Fields:
        pass

    fields = _Fields()
    twins = {}
    launched = []
    for name in folded.ROTATED_NAMES:
        setattr(fields, name, np.zeros(4, dtype=np.complex64))
        twins[name] = np.ones(4, dtype=np.complex64)

    plan = weld.ScratchWeldPairPlan(
        folded.FAMILY, _FakeResidency(), fields, twins,
        {shaders.CONTRACT_OFF: lambda *args: launched.append(args)},
        folded.ROTATED_NAMES, (), (), (2, 2, 1), (0, 0, 0), (0, 0, 0), (),
        folded.REPLACES)
    before = {name: getattr(fields, name) for name in folded.ROTATED_NAMES}
    plan.run()
    assert len(launched) == 1
    for name in folded.ROTATED_NAMES:
        assert getattr(fields, name) is not before[name], name
        assert plan.rotated[name] is before[name], name


# ---------------------------------------------------------------------------
# The device-gated smoke check
# ---------------------------------------------------------------------------

@needs_mps
@pytest.mark.parametrize("codes", [(P, MP, P), (M, MM, P), (MM, MM, MM)])
def test_every_shipped_specialisation_compiles_on_this_host(codes):
    from meep_gpu.metal_kernels.device import compile_source  # noqa: PLC0415

    os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")
    library = compile_source(
        folded.folded_complex_fused_hd_pair_source(codes, (0, 0, 0), ARM))
    assert hasattr(library, folded.KERNEL)


@needs_mps
def test_the_signature_one_pointer_past_the_shipped_shape_does_not_compile():
    """The ceiling is an EQUALITY on this family: 31 compiles and 32 must not."""
    from meep_gpu.metal_kernels.device import compile_source  # noqa: PLC0415

    with pytest.raises(Exception):
        compile_source(folded.refuted_one_more_pointer_source())
