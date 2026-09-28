"""Tests for the folded fused ELECTRIC pair — ``step_D`` welded into ``update_E``.

Everything here runs on a laptop: no GPU, no CuPy, no Triton. What needs hardware —
byte identity against the array path and against the three separately certified
products this launch replaces — lives in
``parity/meep_gpu/probe_triton_folded_fused_pair.py``. A green suite here is not a
certification, and the module under test says so too.

What IS pinned here:

* the geometry that makes this product the mirror image of the magnetic one — the
  near fill touches component ``m`` on the two axes that are NOT ``m`` for D and on
  ``m`` alone for B, which is what leaves the constitutive coefficient index FIXED
  and what creates the parity products the magnetic half cannot have;
* every clause of the seam predicate, in both directions;
* the transcription, read off the shipped source: the curl half's parenthesisation,
  the wall clear's six rows and the constitutive half's two-accumulation grouping;
* the deferral — nothing in ``launch.py`` or ``fastpath.py`` names this module;
* the design and the carry geometry, through the gate's own no-device legs, which
  run here;
* that every armed mutation is scored on a grid that ENTERS the branch it rewrites.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import importlib.util
import itertools
import pathlib
import sys

import pytest

from meep_gpu.fields import Fields, IYEE_SHIFTS, mirror_parity
from meep_gpu.grid import Grid, Mirror
from meep_gpu.pml import PML
from meep_gpu.triton_kernels import folded_fused_magnetic_pair as magnetic
from meep_gpu.triton_kernels import folded_fused_pair as product
from meep_gpu.triton_kernels import symmetry
from meep_gpu.triton_kernels.coverage import zero_metal_axes

PACKAGE_DIR = pathlib.Path(product.__file__).parent
API_ROOT = PACKAGE_DIR.parents[1]
GATE = API_ROOT / "parity" / "meep_gpu" / "probe_triton_folded_fused_pair.py"
RUNNER = API_ROOT / "parity" / "meep_gpu" / "run_triton_folded_fused_pair_direct.sh"

D_COMPONENTS = ("Dx", "Dy", "Dz")


def build(cell=(1.6, 3.0, 1.0), boundaries="metallic", mirrors=(("Y", 1),),
          thickness=0.2):
    """A real Grid/Fields/PML triple on NumPy, with the PML storage allocated."""
    grid = Grid(resolution=10.0, cell_size=cell, boundaries=boundaries,
                symmetry=tuple(Mirror(axis, phase) for axis, phase in mirrors))
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    return fields, PML(grid=grid, thickness=thickness)


def residual(verdict):
    """The reasons that are not the NumPy-host backend clause."""
    return [reason for reason in verdict.reasons if "cupy" not in reason]



def electric_deposit(fields):
    """A REAL source that publishes the index the injection writes.

    :class:`Source` below carries a ``field_type`` and nothing else: enough to PLACE
    a source in a seam, and deliberately not enough to CARRY one -- the repair
    refuses it by name for publishing no deposit index. A case about the carry needs
    the engine's own source, and one that deposits nothing would make the admitting
    assertion pass by measuring an empty scatter, so both are checked here rather
    than assumed at each call site.
    """
    from meep_gpu import deposit_repair  # noqa: PLC0415
    from meep_gpu.sources import GaussianEnvelope, VolumeSource  # noqa: PLC0415

    source = VolumeSource(grid=fields.grid, component="Ez",
                          center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
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
    spec = importlib.util.spec_from_file_location("probe_folded_fused_D", GATE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# The geometry: this is the D family and not the B family
# ---------------------------------------------------------------------------

def test_the_near_fill_touches_each_D_component_on_the_two_axes_that_are_not_its_own():
    """``_fill_symmetry_ghost_cells`` fills axis ``a`` where ``iyee[m][a] == 0``.

    For D that is every axis EXCEPT the component's own, which is the whole reason
    this product's carry differs from the magnetic twin's. Derived from
    ``IYEE_SHIFTS`` rather than restated, and compared against the module's own
    table.
    """
    derived = tuple(
        tuple(axis for axis in range(3) if IYEE_SHIFTS[name][axis] == 0)
        for name in D_COMPONENTS)
    assert derived == product.NEAR_FILL_AXES
    assert derived == ((1, 2), (0, 2), (0, 1))
    for target, axes in enumerate(derived):
        assert target not in axes, (
            f"{D_COMPONENTS[target]} is filled on its own axis; that is the B "
            f"family's geometry, not this one's")


def test_the_two_families_fill_geometries_are_complementary():
    """B fills the component's OWN axis; D fills the other two. Measured, not said."""
    for target in range(3):
        b_axes = set(axis for axis in range(3)
                     if target in magnetic.NEAR_FILL_COMPONENTS[axis])
        d_axes = set(product.NEAR_FILL_AXES[target])
        assert b_axes == {target}
        assert d_axes == {0, 1, 2} - {target}
        assert b_axes.isdisjoint(d_axes)


def test_the_constitutive_coefficient_index_does_NOT_move_on_the_electric_half():
    """The property the whole carry rests on, checked rather than asserted.

    ``update_E`` indexes ``kps``/``kms`` on the component's own axis, and the fill
    images along an axis that is never the component's own — so the imaged ghost
    and its source take the SAME coefficient entry, and the kernel reuses the
    registers. That is exactly what the magnetic twin cannot do.
    """
    import meep_gpu.stepping as stepping

    terms = {component: axis_name
             for component, _source, axis_name in stepping.E_CONSTITUTIVE_TERMS}
    for target, name in enumerate(("Ex", "Ey", "Ez")):
        own_axis = "xyz".index(terms[name])
        assert own_axis == target
        assert own_axis not in product.NEAR_FILL_AXES[target], (
            f"{name}'s coefficient axis is also one of {D_COMPONENTS[target]}'s "
            f"fill axes; the carry's one-index reuse would be wrong")


def test_a_D_component_is_a_fill_destination_on_up_to_two_axes():
    """Which is where the parity PRODUCTS come from, and the corner blocks."""
    counts = [len(axes) for axes in product.NEAR_FILL_AXES]
    assert counts == [2, 2, 2]
    subsets = sum(2 ** len(axes) - 1 for axes in product.NEAR_FILL_AXES)
    assert subsets == 9, "three destinations per component when both axes fold"


def test_the_near_fill_parity_is_plus_the_planes_phase_for_every_D_component():
    """``mirror_parity`` reduces to ``+phase`` on a shift-0 component (fields.py:117)."""
    for target, name in enumerate(D_COMPONENTS):
        for axis in product.NEAR_FILL_AXES[target]:
            for phase in (1, -1):
                assert mirror_parity(name, axis, phase) == phase


def test_two_parities_applied_in_order_equal_their_product():
    """Why the corner may be spelled as one product, and why order is unobservable.

    ``+-1.0`` multiplication is a sign-bit flip: exact for every float32 including
    subnormals and both zeros. So ``phase_z * (phase_y * v)`` and
    ``(phase_z * phase_y) * v`` are the same word, and so is the reversed order.
    Measured over the whole float32 exponent range plus both zeros.
    """
    import numpy as np

    values = np.array([0.0, -0.0, 1.4e-45, -1.4e-45, 1.1754944e-38, -3.4028235e38,
                       0.1, -7.5], dtype=np.float32)
    for first, second in itertools.product((1, -1), repeat=2):
        a = (np.float32(second) * (np.float32(first) * values))
        b = (np.float32(second * first) * values)
        c = (np.float32(first) * (np.float32(second) * values))
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
        assert np.array_equal(a.view(np.uint32), c.view(np.uint32))


def test_the_wall_clear_and_the_near_fill_are_disjoint_on_every_folded_axis():
    """``_zero_metal`` skips a folded axis, so the ghost's wall flags are its source's."""
    for mirrors in ((("Y", 1),), (("X", 1),), (("X", 1), ("Y", -1))):
        fields, _pml = build(cell=(3.0, 3.0, 1.0), mirrors=mirrors)
        grid = fields.grid
        walls = zero_metal_axes(grid)
        for axis in range(3):
            if grid.is_mirrored(axis):
                assert not walls[axis], (
                    f"axis {axis} is both folded and walled; the carry's "
                    f"disjointness no longer holds")


# ---------------------------------------------------------------------------
# The predicate, clause by clause
# ---------------------------------------------------------------------------

def test_a_folded_metallic_grid_is_admitted_modulo_the_numpy_host():
    fields, pml = build()
    verdict = product.folded_fused_pair_coverage(fields, pml, ())
    assert residual(verdict) == []
    assert any("cupy" in reason for reason in verdict.reasons)


def test_a_magnetic_source_is_admitted_and_an_electric_DEPOSIT_is_carried(monkeypatch):
    """The clause tested in BOTH directions, or it cannot tell refusal from refusal.

    THE SEAM'S OWN SOURCE IS NOW CARRIED, NOT REFUSED (2026-08-30). What the clause
    decides is no longer "is there a source in my seam" but "can the deposit repair
    put this one back". So the refusing direction is asserted on a source that
    publishes NO deposit index -- the fail-closed rung -- and a third case holds
    :data:`CARRIES_DEPOSIT_REPAIR` down to show the retired refusal is still what
    the False branch returns. Three cases, because two could not tell a carry from
    a clause that quietly went away.
    """
    fields, pml = build()
    # The other seam's source never concerned this pair and still does not.
    assert residual(product.folded_fused_pair_coverage(
        fields, pml, (Source("B"),))) == []

    # (1) CARRIED: a real electric deposit in this pair's own seam.
    assert residual(product.folded_fused_pair_coverage(
        fields, pml, (electric_deposit(fields),))) == []

    # (2) STILL REFUSED BY NAME: a source whose deposit cannot be saved.
    refused = residual(product.folded_fused_pair_coverage(
        fields, pml, (Source("D"),)))
    assert any("does not publish the index it writes" in reason
               for reason in refused), refused

    # (3) THE FLAG IS WHAT DID IT: held False, the original refusal comes back.
    monkeypatch.setattr(product, "CARRIES_DEPOSIT_REPAIR", False)
    held = residual(product.folded_fused_pair_coverage(
        fields, pml, (electric_deposit(fields),)))
    assert any("is electric" in reason for reason in held), held
    assert any("driver.py:3294" in reason for reason in held), held


def test_an_undeclared_source_list_is_a_refusal_and_not_an_empty_set():
    fields, pml = build()
    refused = residual(product.folded_fused_pair_coverage(fields, pml, None))
    assert any("was not declared" in reason for reason in refused)
    assert product.plan_folded_fused_pair(fields, pml, None) is None


def test_a_folded_periodic_axis_is_ADMITTED_now_that_the_carry_is_GATED():
    """The refusal is retired — and this test is what pins the ORDER it went in.

    Until 2026-08-21 a folded PERIODIC axis was refused by name, because
    ``fill_folded_far_ghosts_D`` runs inside this seam and the kernel did not carry
    it. The carry was written first, the clause STOOD while it was unmeasured (so
    nothing could price it), a device gate executed four folded PERIODIC cases
    through it — two of them 3-D — and only then did the clause lift.

    Both halves are asserted here, because either alone is the defect: the predicate
    must ADMIT, and the kernel must CARRY. A retirement without the carry is a
    predicate that admits a configuration the kernel gets wrong; a carry without the
    retirement is dead code.
    """
    fields, pml = build(boundaries="periodic")
    residual_reasons = residual(product.folded_fused_pair_coverage(fields, pml, ()))
    assert not any("folded PERIODIC" in reason for reason in residual_reasons), (
        residual_reasons)

    # ...and the KERNEL carries it. REPLACES and the body must agree, both ways.
    assert "fill_folded_far_ghosts_D" in product.REPLACES
    text = (PACKAGE_DIR / "folded_fused_pair.py").read_text(encoding="utf-8")
    assert "CARRIED INLINE as of 2026-08-21" in text
    assert "RELEASED 2026-08-21" in text
    # The retirement's own evidence has to be named where a reader will find it.
    assert "run_farcarryD5" in text


def test_the_far_carry_row_checks_SURVIVED_the_retirement():
    """Retiring the blanket clause must not retire the conditions it stood in for.

    The carry's ownership move needs the reflect row to be inside the allocation and
    off both the plane it writes and the plane the near fill reads. Those checks
    were written BESIDE the blanket refusal precisely so that lifting it would leave
    them in force; this asserts that it did, and it is the test that would fail if a
    later round "simplified" the predicate by deleting what the refusal used to
    cover.
    """
    fields, pml = build(boundaries="periodic")
    # A stored extent the reflect row cannot sit inside: the row is derived from the
    # real grid and the extent from the shrunk one, so the range check must fire.
    # AXIS 1, because that is the one `build` folds. Shrinking an unfolded axis
    # would fire nothing here and the test would pass by measuring the wrong thing.
    grid = _ShrunkGrid(fields.grid, 1, 3)
    verdict = product.folded_fused_pair_coverage(_FieldsOn(fields, grid), pml, ())
    reasons = residual(verdict)
    assert any("reflect row" in reason for reason in reasons), reasons
    assert not verdict.covered

def test_the_harness_route_ADMITS_a_folded_periodic_grid_so_the_gate_can_reach_it():
    """``plan_..._from_arrays`` bypasses the predicate by design.

    If the PLAN CLASS also refused ``MIRROR_PERIODIC`` there would be no route to
    the ``FAR_a`` blocks at all and the carry could never be gated — a refusal that
    made its own hazard unmeasurable. The widening has landed in exactly one place,
    and it is a harness route: the coverage predicate still refuses.

    Asserted on the plan's own source rather than by building one, because building
    one needs device pointers this host does not have. What the gate will exercise
    is the behaviour; what this pins is that the route exists and that the refusal
    did not move with it.
    """
    import inspect

    body = inspect.getsource(product.FoldedFusedPairPlan.__init__)
    assert "runs inside this seam and is not carried" not in body, (
        "the plan still raises on a folded PERIODIC axis; the gate cannot reach the "
        "far carry through any route")
    assert "self.far = tuple(code == CODE_MIRROR_PERIODIC" in body
    assert "self.near = tuple(code in MIRROR_CODES" in body
    # ...and the reflect row it will index with is VALIDATED, in both directions.
    assert "the far carry would image the" in body
    assert "stepping._far_reflect_rows answers None" in body

    builder = inspect.signature(product.plan_folded_fused_pair_from_arrays)
    assert "reflect" in builder.parameters, builder

    # The coverage predicate is unchanged in its verdict: still None from the
    # engine-object route.
    fields, pml = build(boundaries="periodic")
    assert product.plan_folded_fused_pair(fields, pml, ()) is None

def test_the_folded_periodic_refusal_is_REQUIRED_and_not_inherited_from_a_half():
    """Which clause actually costs this family its two remaining rows.

    A composition product's clause is INHERITED when a half already refuses
    everything it would, and then a gate may retire it. Measured rather than read:
    both halves are asked the same question on the same object, and both ADMIT a
    folded PERIODIC axis. The refusal exists only here, so carrying the pass is the
    only thing that can retire it.

    Worth 2 seam-instances on both 2026-08-20 closed boards — the whole of this
    product's ``ceiling 4, admits 2`` gap.
    """
    fields, pml = build(boundaries="periodic")
    codes, code_reasons = symmetry.folded_axis_kinds(fields.grid, pml)
    assert not code_reasons and symmetry.CODE_MIRROR_PERIODIC in codes, codes

    for half in (symmetry.folded_composition_curl_coverage(
                     fields, pml, product.CURL_SUB_STEP),
                 symmetry.folded_constitutive_coverage(
                     fields, pml, product.CONSTITUTIVE_SIDE)):
        assert residual(half) == [], residual(half)

    # NON-VACUITY, both directions: the same halves on the same fixture must admit
    # the OTHER termination too, and this product must admit it, or the comparison
    # is measuring a broken fixture rather than a clause.
    metallic, metallic_pml = build(boundaries="metallic")
    assert residual(
        product.folded_fused_pair_coverage(metallic, metallic_pml, ())) == []


def test_the_far_carry_brought_the_top_plane_mask_WITH_it():
    """The obligation the far carry does not discharge on its own — now discharged.

    ``symmetry.pml_curl_step_folded``'s second added block masks the LAST plane of
    every target whose Yee shift is 1 on a folded PERIODIC axis, because that slot
    sits past MEEP's owned window and the fill pass writes it. While this family
    REFUSED ``MIRROR_PERIODIC`` the block was unreachable and the kernel carried
    none; carrying ``fill_folded_far_ghosts_D`` (2026-08-21) brings the axis in, so
    the block must come with it. THIS TEST INVERTED ON THAT DATE and the inversion
    is the point: a far carry that arrived without the mask would step ``fu_D``
    from an unmasked curl on the very plane it images.

    Measured as a DIFF against the certified body rather than by re-implementing
    the mask: a test that spelled the mask itself would agree with a kernel that
    had spelled it wrongly. Both sides are asserted, so the test fails if either
    moves.

    THE D ARM IS THREE LINES, NOT SIX. ``Dx (1,0,0) Dy (0,1,0) Dz (0,0,1)`` is
    shift 1 on the component's OWN axis only; the B family's shifts put it on the
    two others. A body carrying six would be masking the magnetic geometry.
    """
    def periodic_guards(module_path: pathlib.Path, kernel: str) -> int:
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == kernel:
                return sum(1 for child in ast.walk(node)
                           if isinstance(child, ast.If)
                           and "MIRROR_PERIODIC" in ast.unparse(child.test))
        raise AssertionError(f"{module_path.name} declares no kernel {kernel!r}")

    certified = periodic_guards(PACKAGE_DIR / "symmetry.py", "pml_curl_step_folded")
    fused = periodic_guards(PACKAGE_DIR / "folded_fused_pair.py",
                            "folded_fused_curl_constitutive_D")
    assert certified > 0, "the certified folded curl no longer carries a top-plane mask"
    assert fused == 3, (
        f"{fused} MIRROR_PERIODIC guards appear in the fused body, expected 3 — the "
        "D family's Yee shifts put shift 1 on the component's own axis only")

    text = (PACKAGE_DIR / "folded_fused_pair.py").read_text(encoding="utf-8")
    for line in ("curl0 = tl.where(last_x, 0.0, curl0)",
                 "curl1 = tl.where(last_y, 0.0, curl1)",
                 "curl2 = tl.where(last_z, 0.0, curl2)"):
        assert line in text, line
        assert line in (PACKAGE_DIR / "symmetry.py").read_text(encoding="utf-8"), (
            f"{line!r} is not in the certified emitter; the transcription source moved")


def test_an_unfolded_grid_is_refused_because_this_is_a_composition_product():
    fields, pml = build(cell=(1.6, 1.6, 1.0), mirrors=())
    refused = residual(product.folded_fused_pair_coverage(fields, pml, ()))
    assert any("no mirror plane is active" in reason for reason in refused)


def test_no_pml_is_refused_because_update_E_is_a_no_op_without_one():
    fields, _pml = build()
    refused = residual(product.folded_fused_pair_coverage(fields, None, ()))
    assert any("PML" in reason for reason in refused)


def test_the_predicate_reports_both_halves_reasons_with_their_side_named():
    fields, pml = build(cell=(1.6, 1.6, 1.0), mirrors=())
    reasons = residual(product.folded_fused_pair_coverage(fields, pml, ()))
    assert any(reason.startswith("folded curl half: ") for reason in reasons)
    assert any(reason.startswith("folded constitutive half: ") for reason in reasons)


class _ShrunkGrid:
    """A grid that answers everything the real one does, with a shorter fold axis."""

    def __init__(self, grid, axis: int, cells: int) -> None:
        self._grid = grid
        shape = list(grid.shape)
        shape[axis] = cells
        self.shape = tuple(shape)

    def __getattr__(self, name):
        return getattr(self._grid, name)


class _FieldsOn:
    def __init__(self, fields, grid) -> None:
        self._fields = fields
        self.grid = grid

    def __getattr__(self, name):
        return getattr(self._fields, name)


def test_a_folded_axis_with_two_or_fewer_stored_cells_is_refused():
    """The near fill images stored cell 2 FROM THAT CELL'S OWN LANE.

    A fold that does not store it is an out-of-range write, not a soft error, so
    the clause is restated in this product rather than left to the curl half.
    """
    fields, pml = build()
    for cells in (1, 2):
        grid = _ShrunkGrid(fields.grid, 1, cells)
        verdict = product.folded_fused_pair_coverage(_FieldsOn(fields, grid), pml, ())
        assert any("images stored cell 2" in reason
                   for reason in residual(verdict)), (cells, verdict.reasons)


# ---------------------------------------------------------------------------
# The kernel's compile-time bindings
# ---------------------------------------------------------------------------

def test_the_backward_constexpr_is_step_Ds_and_only_step_Ds():
    from meep_gpu.triton_kernels.launch import SUB_STEPS

    assert product.BACKWARD == SUB_STEPS[product.CURL_SUB_STEP]["backward"] == 1
    assert product.CURL_SUB_STEP == "step_D"
    assert product.CONSTITUTIVE_SIDE == "E"


def test_the_kernels_literal_source_index_is_the_named_constant():
    """The kernel spells ``2``; ``symmetry.MIRROR_SOURCE_INDEX`` names it.

    A jit body that closed over a module-level Python int is a Triton-version
    question this file cannot measure without a device, so the literal stays and
    the two are pinned equal here instead.
    """
    source = (PACKAGE_DIR / "folded_fused_pair.py").read_text(encoding="utf-8")
    assert symmetry.MIRROR_SOURCE_INDEX == 2
    for coordinate in ("i", "j", "k"):
        assert f"({coordinate} == 2)" in source
    # The near offsets are named locals since the far carry; the literal 2 is in
    # their definitions, which is where the source index now lives.
    for definition in ("dn_x = -2 * nyz", "dn_y = -2 * nz", "dn_z = -2"):
        assert definition in source, definition


def test_every_carry_block_offsets_by_the_stride_of_the_axis_it_images():
    """Read off the shipped text through the gate's own parser, not restated here."""
    gate = load_gate()
    near = {0: "dn_x", 1: "dn_y", 2: "dn_z"}
    far = {0: "df_x", 1: "df_y", 2: "df_z"}
    definitions = {"dn_x": "-2 * nyz", "dn_y": "-2 * nz", "dn_z": "-2",
                   "df_x": "(nx - 1 - rx) * nyz", "df_y": "(ny - 1 - ry) * nz",
                   "df_z": "nz - 1 - rz"}
    seen = 0
    for record in gate.parse_carry_table():
        # THE FAR TERM COMES FIRST in every composite: the offset is written
        # destination-outward, and the order is asserted so a reordering that
        # happened to evaluate the same is still visible.
        want = " ".join(["+ " + far[axis] for axis in record["far_axes"]]
                        + ["+ " + near[axis] for axis in record["near_axes"]])
        assert record["offset"] == want, record
        # ...and each name resolves to the stride of the axis it images, read off
        # the kernel's own definition rather than restated in the assertion.
        for axis in record["near_axes"] + record["far_axes"]:
            name = near.get(axis) if axis in record["near_axes"] else far[axis]
            for candidate in (near[axis], far[axis]):
                if candidate in record["offsets"]:
                    assert record["offsets"][candidate] == definitions[candidate], (
                        candidate, record["offsets"][candidate])
            seen += 1
    assert seen >= 21, seen


def test_the_plan_refuses_a_fold_that_also_carries_a_wall_or_a_bad_phase():
    """The plan re-checks what the predicate checked, at the point of binding.

    A predicate can be bypassed — ``plan_..._from_arrays`` is the gate's route and
    runs none of it — so the invariants the kernel's constexprs encode are asserted
    where the constexprs are chosen. Built directly rather than through the
    ``_from_arrays`` helper because that one imports ``kernels.DEFAULT_BLOCK``,
    which needs Triton; this contract has to bite on a laptop.
    """
    kwargs = dict(shape=(8, 8, 4), dtdx=0.5, block=256,
                  targets=[], auxiliaries=[], sources=[], curl_coefficients=[],
                  e_targets=[], e_aux=[], inverse_epsilon=[], e_coefficients=[])
    folded = symmetry.CODE_MIRROR_METALLIC
    with pytest.raises(ValueError, match="not \\+1 or -1"):
        product.FoldedFusedPairPlan(
            bc=(1, folded, 1), zero_metal=(True, False, True),
            phases=(0, 0, 0), **kwargs)
    with pytest.raises(ValueError, match="fold and a wall clear"):
        product.FoldedFusedPairPlan(
            bc=(1, folded, 1), zero_metal=(True, True, True),
            phases=(0, 1, 0), **kwargs)
    with pytest.raises(ValueError, match="folded PERIODIC"):
        product.FoldedFusedPairPlan(
            bc=(1, symmetry.CODE_MIRROR_PERIODIC, 1), zero_metal=(True, False, True),
            phases=(0, 1, 0), **kwargs)
    with pytest.raises(ValueError, match="no axis is folded"):
        product.FoldedFusedPairPlan(
            bc=(1, 1, 1), zero_metal=(True, False, True),
            phases=(0, 0, 0), **kwargs)


def test_mirror_phases_answers_zero_off_a_folded_axis_and_the_plane_on_one():
    for mirrors, expected in (((("Y", 1),), (0, 1, 0)),
                              ((("Y", -1),), (0, -1, 0)),
                              ((("X", 1), ("Y", -1)), (1, -1, 0))):
        fields, _pml = build(cell=(3.0, 3.0, 1.0), mirrors=mirrors)
        assert product.mirror_phases(fields.grid) == expected


def test_the_builder_refuses_every_configuration_the_predicate_refuses():
    for kwargs, sources in (({"boundaries": "periodic"}, ()),
                            ({"cell": (1.6, 1.6, 1.0), "mirrors": ()}, ()),
                            ({}, (Source("D"),)),
                            ({}, None)):
        fields, pml = build(**kwargs)
        assert not product.folded_fused_pair_coverage(fields, pml, sources).covered
        assert product.plan_folded_fused_pair(fields, pml, sources) is None


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
    monkeypatch.delitem(sys.modules, "meep_gpu.triton_kernels.folded_fused_pair",
                        raising=False)
    module = importlib.import_module("meep_gpu.triton_kernels.folded_fused_pair")
    assert module.triton is None
    assert module.folded_fused_curl_constitutive_D is None
    fields, pml = build()
    assert module.folded_fused_pair_coverage(fields, pml, ()).reasons
    assert module.plan_folded_fused_pair(fields, pml, ()) is None
    with pytest.raises(ImportError, match="triton"):
        module.folded_fused_curl_constitutive_D_kernel()


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
    code = code_of(PACKAGE_DIR / "folded_fused_pair.py")
    assert "fastpath" not in code
    assert "cuda_kernels" not in code
    assert "options=" not in code


def test_the_single_launch_site_passes_the_packages_shared_guard_constant():
    """A site that omits the guard compiles a kernel that is bit-wrong and plausible."""
    source = (PACKAGE_DIR / "folded_fused_pair.py").read_text(encoding="utf-8")
    spelling = "enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard)"
    assert source.count("enable_fp_fusion=") == 1
    assert spelling in source


def test_the_planner_names_this_module_only_through_the_folded_fusion_route():
    """THE DEFERRAL WAS LIFTED 2026-08-27, and what replaced it is narrower.

    This used to assert that NOTHING in ``launch.py`` names this module — the seam
    that kept a deferred product deferred, on the ground that ``plan_step`` assigns
    at most one plan per ``STEP_ORDER`` slot while this product spans five driver
    passes. The two-slot protocol (``launch._install_fused_pair``) is that missing
    composition rule and ``launch.FOLDED_FUSED_PAIR_ARMS`` is the mapping it needs,
    so ``launch.py`` now names this module in exactly two places: the declared lazy
    seam (``SUPPORT_MODULES``) and the import inside
    ``_folded_fused_pair_entries``.

    THE OTHER TWO FILES ARE UNCHANGED, and that is the half of the old claim that
    still holds: ``fastpath.plan_fast_path`` is untouched, so no default run reaches
    this plan, and the package's ``__init__`` exports nothing from here. The KERNEL
    symbol stays private to this module in all three.

    ``fastpath.py`` NOW NAMES THE FAMILY, AND THE CLAIM IS NARROWED AGAIN RATHER
    THAN DROPPED (2026-08-29). The fusion opt-in gave every fused label an
    ``ARM_CERTIFICATION`` row, so the family name appears there — as a STRING in a
    lookup table read only when an artifact is written, which is the opposite of an
    entry point: it exists so a record naming a dispatched arm also names the gate
    that cut it. "The string is absent" was a proxy for "the planner cannot reach
    this module from fastpath", and the proxy stopped fitting. What replaces it is
    the property itself, in BOTH directions — see the same helper's note in
    ``test_triton_folded_fused_magnetic_pair.py``, whose twin this is. THIS FILE'S
    CHECK COVERS THE D ROW ONLY (corrected 2026-09-19): ``folded_fused_pair`` is
    NOT a substring of ``folded_fused_magnetic_pair`` (``magnetic_`` sits between
    ``fused_`` and ``pair``), so the B row is held inside the table by the twin's
    own call in ``test_triton_folded_fused_magnetic_pair.py``, not by this one.
    This note used to say the substring made this check cover both rows.

    OWED, AND NOT DOABLE IN THIS ROUND: the module's own docstring still says
    "Nothing in ``launch.py`` names this module" and "STILL NOT WIRED". Its bytes
    are digest-pinned by ``fingerprints.json``'s
    ``triton_folded_fused_pair_device_gate``, which this round may not edit, so the
    correction has to land in the same edit as the re-weld.
    """
    launch = (PACKAGE_DIR / "launch.py").read_text(encoding="utf-8")
    fastpath = (PACKAGE_DIR.parent / "fastpath.py").read_text(encoding="utf-8")
    exported = (PACKAGE_DIR / "__init__.py").read_text(encoding="utf-8")

    assert "folded_fused_pair" in launch
    assert "from .folded_fused_pair import (" in launch
    assert "folded_fused_curl_constitutive_D" not in launch, (
        "the planner reaches this product through plan_folded_fused_pair; naming "
        "the kernel would be a second, unmeasured entry point")

    # The package's own namespace is unchanged: it exports nothing from here.
    assert "folded_fused_pair" not in exported, "triton_kernels/__init__.py"
    assert "folded_fused_curl_constitutive_D" not in exported, "triton_kernels/__init__.py"

    # THE KERNEL stays private to this module in fastpath too — that half of the
    # old claim is untouched, and it is the half that would make a second entry
    # point.
    assert "folded_fused_curl_constitutive_D" not in fastpath, "fastpath.py"

    _fastpath_names_the_family_only_in_arm_certification(fastpath,
                                                         "folded_fused_pair")

    from meep_gpu import fastpath as fastpath_module  # noqa: PLC0415

    assert fastpath_module.ARM_CERTIFICATION["fused pair D (folded)"] == (
        "folded_fused_pair", "triton_folded_fused_pair_device_gate"), (
        "the arm label this product wins must map to THIS family and to the weld "
        "that measured it; an unmapped fused arm writes an artifact naming no gate")


def _fastpath_names_the_family_only_in_arm_certification(source: str,
                                                         family: str) -> None:
    """Every mention of ``family`` in ``fastpath.py`` falls inside ARM_CERTIFICATION.

    The twin of the helper in ``test_triton_folded_fused_magnetic_pair.py``, and
    duplicated rather than shared on purpose: each of these two files is digest
    pinned by its OWN weld, and a helper in a third module would put one weld's
    assertions behind bytes that weld does not name.

    Read structurally rather than by counting occurrences: the assignment's line
    span comes from the parse tree, so a row added, removed or re-indented inside
    the table does not move the boundary, and a mention that escapes the table —
    an import, a predicate, a refusal string — lands outside it and fails with the
    line number.

    A ``#`` COMMENT IS NOT SCANNED (2026-09-19): a comment cannot be an import, a
    predicate or a refusal string, so it is none of the three things this helper
    guards. The scan reads the token stream and skips ``tokenize.COMMENT`` tokens
    only; every other token still counts — a name, an attribute, a string, an
    f-string part, a docstring — and an escape is reported at its token's first
    line. What forced it: the 2026-09-14 off-diagonal row of
    ``FUSED_RELEASE_ARM_AXES`` (``fastpath.py:2457``, under the "fused pair B
    (folded)" label) says the folded pairs "are built by
    _install_folded_fused_pairs", and that ``launch.py`` function name contains
    ``folded_fused_pair``, so the old line scan failed on prose that names no
    entry point. The twin helper still scans raw lines — it passes because
    ``_install_folded_fused_pairs`` does not contain
    ``folded_fused_magnetic_pair`` — and takes this change at its own re-weld.
    """
    import ast  # noqa: PLC0415
    import io  # noqa: PLC0415
    import tokenize  # noqa: PLC0415

    tree = ast.parse(source)
    span = None
    for node in ast.walk(tree):
        targets = ([node.target] if isinstance(node, ast.AnnAssign)
                   else getattr(node, "targets", []))
        for target in targets:
            if isinstance(target, ast.Name) and target.id == "ARM_CERTIFICATION":
                span = (node.lineno, node.end_lineno)
    assert span is not None, "fastpath.py no longer defines ARM_CERTIFICATION"

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names] + [
                getattr(node, "module", None) or ""]
            assert not any(family in name for name in names), (
                f"fastpath.py imports {family} at line {node.lineno}: that is an "
                "entry point, not a certification row")

    escaped = sorted({token.start[0] for token in
                      tokenize.generate_tokens(io.StringIO(source).readline)
                      if token.type != tokenize.COMMENT and family in token.string
                      and not span[0] <= token.start[0] <= span[1]})
    assert not escaped, (
        f"fastpath.py names {family} outside ARM_CERTIFICATION (lines {escaped}); "
        f"the table spans {span[0]}-{span[1]}")


def test_the_planner_can_actually_reach_this_products_builder():
    """The route, exercised rather than read off the text above."""
    from meep_gpu.triton_kernels import launch as launch_module

    entries = launch_module._folded_fused_pair_entries()
    coverage, builder = entries["D"]
    assert coverage is product.folded_fused_pair_coverage
    assert builder is product.plan_folded_fused_pair
    assert launch_module.FOLDED_FUSED_PAIRS["D"]["curl"] == "step_D"
    assert launch_module.FOLDED_FUSED_PAIRS["D"]["update"] == "update_E"


def test_the_module_claims_identity_ONLY_through_the_weld_that_measured_it():
    """A weld licenses the claim; a docstring does not.

    ``fingerprints.json`` is the register of measured byte-identity claims. An
    identity claim in this module has to be the one a weld records, and the weld
    has to be the one whose digests match THIS file as it sits on disk — otherwise
    the claim describes bytes nobody measured.
    """
    import hashlib
    import json

    source = (PACKAGE_DIR / "folded_fused_pair.py").read_text(encoding="utf-8")
    record = json.loads(
        (PACKAGE_DIR / "fingerprints.json").read_text(encoding="utf-8"))
    weld = record["triton_folded_fused_pair_device_gate"]
    assert weld["status"] == "PASS"
    assert "RELEASED 2026-08-20" in source

    named = {name: digest for name, digest in weld["source_sha256"].items()
             if name.endswith("triton_kernels/folded_fused_pair.py")}
    assert named, "the weld does not name the module it certifies"
    for name, digest in named.items():
        live = hashlib.sha256(pathlib.Path(API_ROOT / name).read_bytes()).hexdigest()
        assert live == digest, (
            f"{name} has changed since the gate ran; the identity claim in its "
            f"docstring no longer describes the bytes that were measured")

    # THE FAR CARRY'S OWN RELEASE, named where a reader will find it. The weld above
    # was re-cut on 2026-08-21 for it: the module drifted from its 2026-08-20 weld
    # the moment the carry was written, and for the hours between that edit and the
    # device run this test failed on purpose — the fingerprint was NOT edited to
    # match, because a digest edited to match the tree is an assertion where a
    # measurement belongs.
    assert "RELEASED 2026-08-21" in source
    assert "run_farcarryD6" in weld["records"] or "run_farcarryD5" in weld["records"]

    # AND THE BOUND THAT DID NOT MOVE: a weld licenses a claim, not a dispatch.
    # THE VOCABULARY MOVED, THE CLAIM DID NOT. Until 2026-08-27 these modules
    # disclaimed dispatch with the words "NOT WIRED". The routing round then gave
    # them an arm -- launch.py:1168 "THE TWO FOLDED PAIRS, routed 2026-08-27 by
    # _install_folded_fused_pairs" -- so "not wired" became false while the thing
    # this test is defending stayed exactly true, and the modules now say
    # "ROUTED BUT STILL NOT DISPATCHED" instead. This assertion tracks that
    # wording and enforces the SAME bound the comment above states: a weld
    # licenses a claim, not a dispatch. It is not weaker -- a module that
    # silently began claiming dispatch still fails here.
    assert "NOT DISPATCHED" in source, (
        "a released gate is not a wiring; the module must still say so")


def test_the_gate_exists_names_the_product_and_reports_the_run_it_took():
    source = GATE.read_text(encoding="utf-8")
    assert "plan_folded_fused_pair" in source
    assert "folded_fused_curl_constitutive_D" in source
    assert "RELEASED 2026-08-20" in source


def test_the_released_artifact_is_in_the_tree_and_says_it_released():
    """The record the docstring points at must exist and must carry the verdict."""
    import json

    artifact = (API_ROOT / "parity" / "meep_gpu" / "results"
                / "triton_folded_fused_pair_2026-08-20" / "gate.json")
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["release"]["released"] is True, payload["release"]
    assert payload["device_status"] == "RUN"
    assert len(payload["device_legs"]) == 7
    assert all(row["passed"] for row in payload["device_legs"])
    assert len(payload["mutations"]) == 11
    caught = {row["mutation"]: row["caught"] for row in payload["mutations"]}
    assert sum(caught.values()) == 10, caught
    assert caught["m11_commuted_multiply"] is False


# ---------------------------------------------------------------------------
# The gate's own no-device legs, executed here
# ---------------------------------------------------------------------------

def test_the_gates_transcription_leg_traces_every_arithmetic_line_to_its_source():
    row = load_gate().transcription_leg()
    assert row["findings"] == []
    assert row["passed"]
    assert row["curl_lines_checked"] == 6
    assert row["zero_metal_lines_checked"] == 6
    assert row["top_plane_lines_checked"] == 3
    assert row["carry_ghost_lines_checked"] == 8


def test_the_gates_carry_table_leg_reproduces_the_array_paths_own_fill():
    """The one leg that executes the kernel's GEOMETRY rather than mirroring it.

    21 blocks as of the far carry, not 9: each component is the source of every
    nonempty subset of its two NEAR axes and its one FAR axis, which is 7 apiece.
    """
    row = load_gate().carry_table_leg()
    assert row["findings"] == [], row["findings"]
    assert len(row["blocks"]) == 21, [
        (entry["target"], entry["near_axes"], entry["far_axes"])
        for entry in row["blocks"]]
    assert sum(1 for entry in row["blocks"] if entry["far_axes"]) == 12, (
        "four of each component's seven destinations involve the far half")
    for case in row["rows"]:
        assert case["destination_writes"] > 0, case
        assert case["collisions"] == [], case
    # NON-VACUITY: the leg must reach a folded PERIODIC axis and a wall, or it
    # agrees with a kernel whose far carry or clear order is wrong.
    assert any(any(case["far"]) for case in row["rows"])
    assert any(any(case["far"]) and any(case["walls"]) for case in row["rows"])
    assert any(sum(case["far"]) == 3 for case in row["rows"]), (
        "no case folds three PERIODIC axes, so the triple composites are shipped "
        "code this leg never executed")


def test_the_gates_carry_table_verdict_FLIPS_on_every_defect_it_claims_to_see():
    """A leg that passes proves nothing; a leg that fails on a wrong kernel does.

    Every mutation the table declares is planted into the shipped text and the
    carry table is re-parsed and re-executed from it. The expectation is DECLARED
    per mutation — a defect the leg says it cannot see must report NOT CAUGHT with
    its reason, and every other must flip the verdict.

    THIS LEG FOUND A DEFECT IN THE CARRY-TABLE LEG ITSELF on 2026-08-21: five
    mutations were planted and the table still passed, because the executor was
    re-deriving ownership from ``NEAR_FILL_AXES`` instead of reading the kernel's
    own lane and mask expressions — a mirrored evaluator inside the leg whose whole
    purpose is not to be one. It now reads them.
    """
    row = load_gate().planted_defect_leg()
    assert row["findings"] == [], row["findings"]
    assert row["passed"]
    assert row["mutations_planted"] >= 17
    assert row["caught_here"] == row["scoreable_here"], row["rows"]
    # The far carry's own mutations are scoreable HERE, not deferred: they are the
    # branches no device leg has executed, so a laptop-side flip is the only
    # evidence that exists for them today.
    scoreable = {entry["mutation"] for entry in row["rows"]
                 if not entry["declared_out_of_reach"]}
    for name in ("m12_far_composite_is_a_raw_product",
                 "m13_far_takes_the_source_coefficient",
                 "m14_far_carry_dropped",
                 "m15_far_parity_sign",
                 "m17_far_reflect_row_is_n_minus_two"):
        assert name in scoreable, name
    # ...and what is deferred is declared, so the deferral cannot grow quietly.
    assert set(row["deferred_to_the_device_gate"]) == {
        "m6_constitutive_association", "m7_history_read_after_write",
        "m8_zero_metal_dropped", "m11_commuted_multiply",
        "m16_top_plane_mask_dropped"}


def test_the_gates_design_sweep_agrees_with_the_array_path_and_catches_its_flips():
    row = load_gate().design_sweep_leg()
    assert row["findings"] == []
    caught = {(entry["case"], entry["knob"]): entry["identical"]
              for entry in row["rows"]}
    for case in ("even_y", "odd_y", "even_x", "two_folds"):
        assert caught[(case, "faithful")] is True
        assert caught[(case, "drop_the_near_fill")] is False
        assert caught[(case, "constitutive_before_the_fill")] is False


def test_the_gates_predicate_leg_exercises_every_clause_in_both_directions():
    row = load_gate().predicate_leg()
    assert row["findings"] == []
    directions = {entry["case"]: entry["expect_admitted"] for entry in row["rows"]}
    assert directions["folded_metallic_magnetic_source"] is True
    # THE THREE ROWS THE 2026-08-30 CARRY ADDED, asserted by NAME and by direction.
    # A set equality alone would let a future round rename the carried row to the
    # refused one and stay green, so the directions are pinned individually: the
    # deposit is ADMITTED, the source that cannot publish its index is REFUSED, and
    # the deposit is REFUSED AGAIN once the flag is held down.
    assert directions["folded_metallic_electric_deposit_is_carried"] is True
    assert directions[
        "folded_metallic_electric_source_without_a_deposit_index"] is False
    assert directions[
        "folded_metallic_electric_deposit_refused_when_the_flag_is_held_False"] is False


def test_the_gates_corpus_leg_agrees_with_the_metal_censuss_own_row_set():
    """Two independently written predicate stacks, one row set. Measured here.

    If this ever fails, the finding is NOT that the product is worth less: it is
    that one of the two backends' predicate ladders is wrong, which matters more.
    """
    row = load_gate().corpus_admission_leg()
    assert row["findings"] == [], row["findings"]
    assert row["triton"]["no_electric_source_rows"] == \
        row["metal"]["no_electric_source_rows"]
    assert row["triton"]["curl"] == row["metal"]["curl"]
    assert row["triton"]["both"] == row["metal"]["both"]
    assert row["seam_instances_gained"] >= 1


def test_the_gate_arms_every_mutation_it_declares():
    """A rewrite that matches nothing reports a defect as uncaught while measuring
    nothing — this project has paid for that once already."""
    gate = load_gate()
    import inspect as _inspect
    import textwrap as _textwrap

    text = _textwrap.dedent(gate._shipped_text("folded_fused_curl_constitutive_D"))
    for name, _why, _expectation, rewrite in gate.mutation_table():
        mutated, hits = rewrite(text)
        assert hits > 0, name
        assert mutated != text, name


def test_every_mutation_is_scored_on_a_grid_that_ENTERS_the_branch_it_rewrites():
    """THE DEAD-BRANCH TRAP, priced.

    Every carry block sits under ``BC? == MIRROR_METALLIC`` on a named axis and
    every corner block under two. A mutation scored on a case that does not fold
    those axes rewrites real lines the case never reaches and comes back "uncaught"
    while measuring nothing.
    """
    gate = load_gate()
    for name, _why, _expectation, _rewrite in gate.mutation_table():
        index, case = gate.mutation_case_for(name)
        folded = {axis.upper() for axis, _phase in case[3]}
        for required in gate.MUTATION_REQUIRES_FOLD.get(name, ()):
            assert required in folded, (name, case[0])


def test_the_corner_mutations_are_scored_on_a_two_fold_case_of_the_RIGHT_kind():
    """The NEAR corner blocks need two folds; they do NOT need them PERIODIC.

    There are two two-fold cases since 2026-08-21 — one METALLIC, one PERIODIC —
    so "the only one" is no longer the test. The near corner sits under two
    ``NEAR_a`` guards, which are true on EITHER mirror code, so either case would
    reach it; the METALLIC one is chosen because it isolates the near corner from
    the far composites that share its cell on the periodic one.
    """
    gate = load_gate()
    two_fold = {index: case for index, case in enumerate(gate.CASES)
                if len(case[3]) >= 2}
    assert len(two_fold) >= 1, two_fold
    for name in ("m3_corner_carry_dropped", "m4_corner_weight_is_one_axis_twice"):
        index = gate.MUTATION_CASE[name]
        assert index in two_fold, (name, index, sorted(two_fold))
        # The two folds must carry DIFFERENT parities, or the corner's product is
        # its own square and a swapped factor is invisible.
        phases = {phase for _axis, phase in two_fold[index][3]}
        assert phases == {1, -1}, (name, two_fold[index])


def test_the_parity_mutations_are_scored_on_an_ODD_plane():
    """On an even plane the multiply is by +1 and a dropped parity is invisible."""
    gate = load_gate()
    index, case = gate.mutation_case_for("m1_parity_dropped")
    assert any(phase == -1 for _axis, phase in case[3]), case


def test_the_gate_binds_the_seam_passes_the_driver_actually_calls():
    """A pass named here but not called there would be substituted into nothing."""
    gate = load_gate()
    driver_source = (PACKAGE_DIR.parent / "driver.py").read_text(encoding="utf-8")
    for name in gate.SEAM_PASSES:
        assert f"{name}(self.fields" in driver_source, name
    assert gate.SEAM_PASSES == ("step_D", "fill_symmetry_bc_D", "zero_metal_D",
                                "fill_folded_far_ghosts_D", "update_E")
    # REPLACES gained `fill_folded_far_ghosts_D` in the SAME EDIT that put the
    # carry in the kernel body. A module that claims a pass it does not carry, or
    # carries one it does not claim, is a defect this family has already paid for
    # once on the Metal board — where a passing test pinned the stale tuple and hid
    # it. Both directions are asserted here so neither can move alone.
    assert tuple(product.REPLACES) == ("step_D", "fill_D", "zero_metal_D",
                                       "fill_folded_far_ghosts_D", "update_E")
    body = gate._shipped_text("folded_fused_curl_constitutive_D")
    carries_far = "FAR_X" in body and "df_x" in body
    assert carries_far == ("fill_folded_far_ghosts_D" in product.REPLACES), (
        "REPLACES and the kernel body disagree about the far fill")


def test_the_gates_material_names_are_the_ones_the_scan_really_finds():
    """A set of names that matches nothing disables the vacuity floor AND the
    material check, silently. Measured against a real Fields object."""
    gate = load_gate()
    fields, _pml = build()
    found = {name for name, value in vars(fields).items()
             if getattr(value, "shape", None) == tuple(fields.grid.shape)}
    for name in gate.MATERIAL:
        assert name in found, (name, sorted(found))


def test_the_case_table_REACHES_the_far_carry_and_the_guard_refuses_a_pairing_that_does_not():
    """Every branch this kernel ships must be named by a case that executes it.

    THE SHIPPED-BUT-UNGATED TRAP, priced. Before 2026-08-21 every case in the gate
    folded over a METALLIC outer declaration, and every ``FAR_a`` block is
    compile-time absent there — so a far carry released on that table would have
    been code no leg ever ran. Four folded PERIODIC cases were added with the carry.

    The guard is shown to be ARMED, not merely present: a far mutation is re-pointed
    at a folded METALLIC case and ``mutation_case_for`` must refuse it. A guard that
    never fires is indistinguishable from no guard.
    """
    gate = load_gate()

    def declaration(case, axis_letter: str) -> str:
        boundaries = case[2]
        if isinstance(boundaries, str):
            return boundaries
        return str(boundaries.get(axis_letter.lower(), "periodic"))

    periodic_folds = [
        case for case in gate.CASES
        if any(declaration(case, axis) == "periodic"
               for axis, _phase in case[3])]
    assert periodic_folds, (
        "no case folds an axis over a PERIODIC outer declaration, so not one FAR_a "
        "block would execute and the far carry would be shipped code no leg ran")

    # ...including one with TWO of them (the composites), one at an ODD full count
    # (the reflect row), and one beside a WALL (the parity/clear order).
    assert any(sum(1 for axis, _p in case[3]
                   if declaration(case, axis) == "periodic") >= 2
               for case in periodic_folds), "no case folds two PERIODIC axes"
    assert any(int(round(12.0 * float(case[1]["XYZ".index(axis.upper())])))% 2 == 1
               for case in periodic_folds for axis, _p in case[3]
               if declaration(case, axis) == "periodic"), (
        "no folded PERIODIC case has an ODD full count; the reflect row is "
        "stored - 2 at an even one and a baked n-2 would read as correct")
    assert any(
        any(declaration(case, axis) == "periodic" for axis, _p in case[3])
        and any(declaration(case, letter) == "metallic"
                and letter.upper() not in {a.upper() for a, _p in case[3]}
                for letter in "xy")
        for case in periodic_folds), (
        "no case carries a folded PERIODIC axis beside a wall; the near ghost's "
        "parity-then-clear and the far ghost's clear-then-parity cannot disagree "
        "without one")

    # THE GUARD IS ARMED. Re-point a far mutation at a folded METALLIC case and the
    # pairing must be refused by name.
    metallic_case = next(index for index, case in enumerate(gate.CASES)
                         if case[0] == "two_folds_metallic")
    saved = gate.MUTATION_CASE["m14_far_carry_dropped"]
    gate.MUTATION_CASE["m14_far_carry_dropped"] = metallic_case
    try:
        with pytest.raises(AssertionError, match="DEAD BRANCH"):
            gate.mutation_case_for("m14_far_carry_dropped")
    finally:
        gate.MUTATION_CASE["m14_far_carry_dropped"] = saved

    # ...and every mutation the table declares must pair with a case that reaches it.
    for name, _why, _expectation, _rewrite in gate.mutation_table():
        gate.mutation_case_for(name)
