"""The hand-CUDA fusion block, and the in-seam deposit it carries.

WHAT THIS FILE SETTLES. Until 2026-08-28 ``covers_fused_magnetic_pair`` had NO laptop
test at all -- its module imported CuPy at scope, so the census recorded it as
``askable: false`` and the merge bar never reached it. The predicate is the part whose
failure mode is a SILENT WRONG ANSWER, so the questions here are the ones that decide
whether the deposit carry is real:

1. **Does the predicate ADMIT what it used to refuse?** A magnetic deposit in the
   seam, and the before/after is measured against the SHIPPED flag rather than
   asserted -- the same case is asked twice, once with the flag held at ``False``.
2. **Does it still REFUSE what the repair cannot reconstruct, by name?** A source that
   does not publish the index it writes; an undeclared source set; and every mirrored
   axis, whose two mirror fills this product does not carry at all.
3. **Does the two-consult protocol reproduce the driver's own order, bit for bit?**
   Driven the way ``FdtdDriver.step`` drives it, against the array path, on NumPy,
   with a null control that must diverge.
4. **Does the block fail closed?** It may not claim a slot the arm table refused or
   gave to another arm; it may not run unless asked; it may not raise.

NOTHING HERE COMPILES OR LAUNCHES ANYTHING. The fused product's own arithmetic is a
DEVICE claim and no test on this host may speak to it: what is measured here is the
protocol around the launch, which is exactly the half the repair lives in.
"""

from __future__ import annotations

import ast
import pathlib
import sys
import types
from unittest import mock

import numpy
import pytest

from .. import deposit_repair, stepping, withdraw_hoist
from ..fields import Fields
from ..grid import Grid, Mirror
from ..pml import PML
from ..sources import GaussianEnvelope, VolumeSource
from ..triton_kernels import launch as launch_module
from ..triton_kernels.coverage import Coverage
from ..triton_kernels.launch import NoopPlan
from . import arms, fused_pairs
from .test_dispersive import build as build_dispersive
from . import fused_magnetic_pair as family
from ..test_deposit_repair import _build, _differing, _reference, _words


class _NumpyWearingCupysName(types.ModuleType):
    """NumPy behind CuPy's ``__name__`` -- this directory's standing stand-in.

    Same object ``test_arms.py`` uses, and the same one the coverage census factors
    its backend clause with. Every CUDA predicate's first question is whether the
    array module is CuPy at all, and that is the one thing about the device library a
    laptop cannot supply; everything else is a real object either way.
    """

    def __init__(self):
        super().__init__("cupy")

    def __getattr__(self, item):
        return getattr(numpy, item)


@pytest.fixture
def xp():
    return _NumpyWearingCupysName()


def build(xp, *, boundaries=("periodic", "periodic", "periodic"), symmetry=(),
          cell=(8.0, 8.0, 8.0)):
    """A frozen ``(fields, pml, grid)`` triple the certified B/H pair admits."""
    grid = Grid(resolution=1.0, cell_size=cell, boundaries=boundaries,
                symmetry=symmetry, xp=xp)
    thickness = tuple((0, 0) if grid.shape[axis] < 6
                      else (0, 2) if grid.is_mirrored(axis)
                      else (2, 2)
                      for axis in range(3))
    layer = PML(grid=grid, thickness=thickness)
    fields = Fields(grid=grid)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    return fields, layer, grid


def magnetic_source(grid, component="Hz"):
    return VolumeSource(grid=grid, component=component, center=(0.0, 0.0, 0.0),
                        size=(0.0, 0.0, 0.0),
                        envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2),
                        amplitude=1.0)


class _ArrayPair:
    """Stands in for the fused kernel: curl + wall clear + constitutive in one object.

    THE THING UNDER TEST IS THE PROTOCOL, NOT THE KERNEL. The shipped product is one
    launch of ``fused_magnetic_pair_pml_real`` and this is three array calls, but to
    the two-consult protocol they are the same THING: an object whose ``run()`` closes
    the seam against whatever the field holds when it is called, against an
    UNINJECTED field. CuPy is not installed on the merge-bar host, and a test that
    needed it would not run here at all -- it would be a skip, which is not evidence.

    The calls are the product's own ``REPLACES``, in its own order, READ FROM THE
    MODULE rather than transcribed -- which is why this stand-in gained the two
    mirror fills on 2026-08-28 without a line changing here. The table is looked up
    on ``stepping`` by the pass's own name for the same reason: a pass added to
    ``REPLACES`` with no ``stepping`` function is a KeyError here rather than a
    silently shorter seam.
    """

    #: Which of ``REPLACES``' passes take the absorber layer. Read off
    #: ``stepping``'s signatures rather than guessed, so a pass that gained or lost
    #: the argument is a TypeError rather than a silently skipped absorber.
    TAKES_PML = frozenset({"step_B", "update_H"})

    def __init__(self, fields, pml):
        self._fields, self._pml = fields, pml
        self.runs = 0

    def run(self, *_args, **_kwargs):
        self.runs += 1
        for name in family.REPLACES:
            pass_ = getattr(stepping, name)
            if name in self.TAKES_PML:
                pass_(self._fields, self._pml)
            else:
                pass_(self._fields)


# --------------------------------------------------------------------------
# 1. THE PREDICATE ADMITS WHAT IT USED TO REFUSE
# --------------------------------------------------------------------------

def test_the_product_declares_the_repair_and_the_block_that_brackets_it_exists():
    """The flag and the wiring change together or not at all.

    Both halves are read rather than assumed, because the failure this pins is a flag
    flipped without a bracket: that product would compute the constitutive half
    against a pre-injection field and report success.
    """
    assert family.CARRIES_DEPOSIT_REPAIR is True
    source = pathlib.Path(family.__file__).read_text(encoding="utf-8")
    assert "carries_repair=CARRIES_DEPOSIT_REPAIR" in source, (
        "the predicate must pass the module's own constant, not a literal")
    block = pathlib.Path(fused_pairs.__file__).read_text(encoding="utf-8")
    assert "LeadingRepairPlan(" in block and "TrailingRepairPlan(" in block


def test_a_magnetic_deposit_is_admitted_now_and_was_refused_before(xp, monkeypatch):
    """THE MOVEMENT, measured on one configuration asked twice.

    The refusing answer is produced by holding the SHIPPED flag at ``False`` -- the
    state this product was in until the block landed -- so the before column is the
    real clause and not a remembered string.
    """
    fields, layer, grid = build(xp)
    sources = [magnetic_source(grid)]

    covered, reason = family.covers_fused_magnetic_pair(fields, layer, grid, sources)
    assert covered, reason

    monkeypatch.setattr(family, "CARRIES_DEPOSIT_REPAIR", False)
    covered_before, reason_before = family.covers_fused_magnetic_pair(
        fields, layer, grid, sources)
    assert not covered_before
    assert "is magnetic" in reason_before and "driver.py:3293" in reason_before


def test_an_electric_source_never_disqualified_this_seam(xp):
    """The complement, so the case above is not measuring "any source at all"."""
    fields, layer, grid = build(xp)

    class _Electric:
        field_type = "D"

    covered, reason = family.covers_fused_magnetic_pair(
        fields, layer, grid, [_Electric()])
    assert covered, reason


# --------------------------------------------------------------------------
# 2. AND STILL REFUSES, BY NAME, WHAT THE REPAIR CANNOT RECONSTRUCT
# --------------------------------------------------------------------------

def test_a_source_that_hides_the_index_it_writes_is_refused_by_name(xp):
    """``deposit_repair.save`` reads ``_point_ix/_point_iy/_point_iz``; a source that
    publishes none cannot be saved and restored, and admitting it would put the
    deposit inside a launch nothing repairs."""
    fields, layer, grid = build(xp)

    class _Opaque:
        field_type = "B"

    covered, reason = family.covers_fused_magnetic_pair(
        fields, layer, grid, [_Opaque()])
    assert not covered
    assert "does not publish the index it writes" in reason
    assert "source 0" in reason


def test_an_undeclared_source_set_is_still_a_refusal(xp):
    """IGNORANCE IS NOT AN EMPTY SET, and the flip does not touch that: a repair that
    was never handed a source saves nothing and restores nothing."""
    fields, layer, grid = build(xp)
    covered, reason = family.covers_fused_magnetic_pair(fields, layer, grid, None)
    assert not covered
    assert "the source set was not declared" in reason


def test_the_repairs_own_state_refusal_is_shadowed_here_and_that_is_not_a_hole(xp):
    """WHERE THE REMAINING ``repairable`` CLAUSES LAND ON THIS SEAM, stated rather
    than implied.

    ``deposit_repair.repairable`` refuses three things. Two are D-side only (an
    off-diagonal chi1inv row and an instantaneous chi2/chi3) and no CUDA D/E fused
    product exists, so nothing on this track can reach them yet. The third -- an
    unallocated ``f_w_<component>`` -- is reachable in principle but is SHADOWED: the
    constitutive half's own certified predicate refuses the same configuration first,
    and this predicate is a conjunction that asks it first. Both answer no, so the
    shadowing is defence in depth and not a gap; pinned here so that a later
    reordering of the conjunction cannot quietly turn it into one.
    """
    fields, layer, grid = build(xp)
    fields.f_w_Hx = None
    assert deposit_repair.repairable(fields, "B") == (
        False, ("f_w_Hx is not allocated; there is no state to save",))
    covered, reason = family.covers_fused_magnetic_pair(
        fields, layer, grid, [magnetic_source(grid)])
    assert not covered
    assert reason.startswith("constitutive half:") and "f_w_Hx" in reason


@pytest.mark.parametrize("axis,name", ((0, "X"), (1, "Y"), (2, "Z")))
def test_a_folded_axis_is_carried_now_and_was_refused_before(xp, axis, name):
    """THE FILLS WERE A SEPARATE QUESTION FROM THE DEPOSIT, AND THEY ARE ANSWERED.

    Until 2026-08-28 a folded seam was refused because it runs
    ``fill_symmetry_bc_B`` and ``fill_folded_far_ghosts_B`` after the injection and
    this product carried neither. The OWNERSHIP INVERSION carries both -- the source
    thread writes the destination from a live register, so no thread reads a word
    another block writes -- and ``REPLACES`` names all five passes.

    BOTH DIRECTIONS ARE MEASURED. The admission is asked of the shipped predicate;
    the plan the launch would take is asked of the shipped builder, and it must
    actually name the fold rather than come back empty (an "admission" whose plan is
    all-zeros carries nothing and would be a silent wrong answer, not a fusion).
    """
    fields, layer, grid = build(xp, symmetry=(Mirror(name, 1),))
    assert grid.is_mirrored(axis), "the case did not actually fold an axis"
    covered, reason = family.covers_fused_magnetic_pair(fields, layer, grid, ())
    assert covered, reason
    assert {"fill_symmetry_bc_B", "fill_folded_far_ghosts_B"} <= set(family.REPLACES)

    plan = family.fused_magnetic_pair_fills(grid)
    assert plan["near"][axis] == 1, plan
    assert plan["phase"][axis] in (1.0, -1.0), plan
    assert sum(plan["near"]) == 1, "only the folded axis carries the near fill"

    # ... AND THE REFUSAL THE CARRY DOES NOT LIFT. A folded axis too short to hold
    # the near fill's source row is still refused BY NAME: the carry images stored
    # row 2 and ``stepping._mirror_source`` raises where it does not exist.
    class _TooShort:
        def __getattr__(self, item):
            return getattr(grid, item)

        def stored_cells(self, other):
            return 2 if other == axis else grid.stored_cells(other)

    short, short_reason = family.covers_fused_magnetic_pair(
        fields, layer, _TooShort(), ())
    assert not short
    assert "does not exist" in short_reason or "stored_cells" in short_reason


# --------------------------------------------------------------------------
# 3. THE TWO-CONSULT PROTOCOL AGAINST THE DRIVER'S OWN ORDER
# --------------------------------------------------------------------------

#: The two grids the numeric leg runs on: the 2-D shape the sibling deposit tests use
#: and a 3-D one, because the deposit is a POINT and a 2-D grid leaves one axis with
#: no interior for an index error to land in.
GRIDS = ((1.2, 1.2, 0.0), (1.0, 1.0, 1.0))


@pytest.mark.parametrize("cell", GRIDS, ids=("2d", "3d"))
def test_the_two_slots_consulted_as_the_driver_consults_them_match_the_array_path(cell):
    """THE ONE THAT MATTERS, and the only numeric claim this file makes.

    Two independently seeded copies of the same state. One is stepped in the DRIVER'S
    order -- curl, inject, fill, clear, far fill, constitutive. The other is stepped
    the way a fused pair owning both consults is: the whole seam closed against an
    UNINJECTED field at the first consult, then the driver's own inject/fill/clear,
    then the repair at the second. Every float32 word of all 24 arrays is compared as
    a uint32, so "close" is not a passing answer.

    MEASURED 2026-08-28 on this host: 0 differing words of 4704 (2-D, 14x14x1) and 0
    of 41472 (3-D, 12x12x12) over 8 steps, with 12 deposit points repaired in each.
    The null control below diverges on 8 words in both -- 4 in ``Hz`` and 4 in
    ``f_w_Hz`` -- which is what makes the zeros above evidence rather than a tautology.
    """
    grid_a, fields_a, pml_a = _build(cell=cell)
    grid_b, fields_b, pml_b = _build(cell=cell)
    src_a = [magnetic_source(grid_a)]
    src_b = [magnetic_source(grid_b)]
    assert src_b[0]._n_source_points, "the case deposits nothing and cannot discriminate"

    plans, selected = {}, {}
    inner = _ArrayPair(fields_b, pml_b)
    fused_pairs._install_fused_pair(plans, selected, fields_b, pml_b, src_b, "B",
                                    "step_B", "update_H", "fused magnetic pair", inner)
    assert isinstance(plans["step_B"], deposit_repair.LeadingRepairPlan)
    assert isinstance(plans["update_H"], deposit_repair.TrailingRepairPlan)

    dt = grid_a.dt
    for step in range(8):
        _reference(fields_a, pml_a, src_a, step * dt, dt, "B")
        plans["step_B"].run()
        for source in src_b:
            source.inject(fields_b, step * dt)
        stepping.fill_symmetry_bc_B(fields_b)
        stepping.zero_metal_B(fields_b)
        stepping.fill_folded_far_ghosts_B(fields_b)
        plans["update_H"].run()

    differing = _differing(_words(fields_a), _words(fields_b))
    assert not differing, differing
    assert inner.runs == 8
    assert plans["step_B"].repairs > 0, "the repair reported touching no point"


@pytest.mark.parametrize("cell", GRIDS, ids=("2d", "3d"))
def test_the_same_protocol_without_the_repair_diverges(cell):
    """The null control. Without it the case above proves only that both routes ran."""
    grid_a, fields_a, pml_a = _build(cell=cell)
    grid_b, fields_b, pml_b = _build(cell=cell)
    src_a = [magnetic_source(grid_a)]
    src_b = [magnetic_source(grid_b)]
    inner = _ArrayPair(fields_b, pml_b)
    dt = grid_a.dt
    for step in range(8):
        _reference(fields_a, pml_a, src_a, step * dt, dt, "B")
        inner.run()
        for source in src_b:
            source.inject(fields_b, step * dt)
        stepping.fill_symmetry_bc_B(fields_b)
        stepping.zero_metal_B(fields_b)
        stepping.fill_folded_far_ghosts_B(fields_b)
    differing = _differing(_words(fields_a), _words(fields_b))
    assert differing, ("the unrepaired protocol matched the reference, so the case "
                       "above proves nothing")
    # WHERE it lands is the discriminating half: the constitutive target the deposit
    # feeds and its PML auxiliary, which are exactly the two arrays `apply` restores.
    assert set(differing) == {"Hz", "f_w_Hz"}, differing


def test_a_clean_seam_keeps_the_cheap_sentinel_in_the_second_slot():
    """No deposit, no repair: the second consult must stay the no-op it always was."""
    _grid, fields, pml = _build()
    plans, selected = {}, {}
    fused_pairs._install_fused_pair(plans, selected, fields, pml, (), "B", "step_B",
                                    "update_H", "fused magnetic pair",
                                    _ArrayPair(fields, pml))
    assert isinstance(plans["update_H"], NoopPlan)
    assert selected["step_B"] == selected["update_H"] == "fused magnetic pair"


def test_the_second_slot_refuses_when_the_first_never_ran():
    """``dispatch`` contracts that True means the whole sub-step ran; a silent no-op
    here would leave the constitutive half undone and report success."""
    grid, fields, pml = _build()
    plans, selected = {}, {}
    fused_pairs._install_fused_pair(plans, selected, fields, pml,
                                    [magnetic_source(grid)], "B", "step_B",
                                    "update_H", "fused magnetic pair",
                                    _ArrayPair(fields, pml))
    with pytest.raises(deposit_repair.DepositNotRepairable):
        plans["update_H"].run()


# --------------------------------------------------------------------------
# 4. THE BLOCK, AND THE WAYS IT MUST FAIL CLOSED
# --------------------------------------------------------------------------

def test_fusion_is_opt_in_and_the_default_composition_is_unchanged(xp):
    """``fuse=False`` is the census's composition, and it must stay exactly that."""
    fields, layer, grid = build(xp)
    plan = arms.plan_step(fields, layer, grid, sources=())
    assert plan.selected["step_B"] == "PML"
    assert plan.selected["update_H"] == "ordinary"
    assert not [key for key in plan.reasons if key.startswith("fused_pair_")]


def test_a_clean_seam_fuses_and_the_pair_reports_the_pass_it_carries(xp):
    fields, layer, grid = build(xp)
    plan = arms.plan_step(fields, layer, grid, sources=(), fuse=True)
    assert plan.selected["step_B"] == plan.selected["update_H"] == "fused magnetic pair"
    assert isinstance(plan.plans["step_B"], fused_pairs.CudaFusedPairPlan)
    assert isinstance(plan.plans["update_H"], NoopPlan)
    # ``zero_metal_B`` is carried in registers and is in NEITHER slot name. A
    # composition that reported only the slots would say it ran on the array path.
    assert "zero_metal_B" in fused_pairs.replaced_sub_steps(plan.plans)
    assert fused_pairs.declaring_plan(plan.plans["update_H"]) is plan.plans["step_B"]


def test_a_seam_carrying_a_deposit_fuses_into_the_two_repair_plans(xp):
    """The composer route, end to end: the predicate admits and the block brackets."""
    fields, layer, grid = build(xp)
    plan = arms.plan_step(fields, layer, grid, sources=[magnetic_source(grid)],
                          fuse=True)
    assert isinstance(plan.plans["step_B"], deposit_repair.LeadingRepairPlan)
    assert isinstance(plan.plans["update_H"], deposit_repair.TrailingRepairPlan)
    assert fused_pairs.declaring_plan(plan.plans["update_H"]) is \
        plan.plans["step_B"].inner
    # AND THE LAUNCHABILITY IS READ THROUGH THE WRAPPERS. Neither repair plan
    # declares `launchable`; read bare, a seam would stop reporting as launchable the
    # moment it started carrying a deposit, which is a claim about the kernel made by
    # a fact about the source list.
    assert "step_B" in plan.launchable and "update_H" in plan.launchable


def test_an_undeclared_source_set_leaves_the_seam_unfused_with_a_named_reason(xp):
    """A refusal is not a silence: ``plan_step`` must say why the seam did not fuse."""
    fields, layer, grid = build(xp)
    plan = arms.plan_step(fields, layer, grid, fuse=True)
    assert "step_B" in plan.plans and not isinstance(
        plan.plans["step_B"], fused_pairs.CudaFusedPairPlan)
    reasons = plan.reasons["fused_pair_cuda_fused_magnetic_pair"]
    assert len(reasons) == 1, reasons
    assert "the source set was not declared" in reasons[0]


def test_a_pair_may_not_claim_a_slot_the_arm_table_gave_to_another_arm(xp):
    """The one place a ``STEP_ORDER`` slot is filled without going through the table.

    Written unconditionally it would be a hole straight through the ambiguity and
    builder-refusal clauses -- the fused product would take a slot the composer
    declined to give anyone. Driven here by handing the block a selection in which
    ``update_H`` went to a different arm.

    THE EXEMPLAR ARM IS ``BFAST``, NOT ``nonlinear``: until 2026-09-01 this test
    handed ``update_H`` to ``nonlinear``, and that pair became a DECLARED
    absorption (``FUSED_PAIR_EXTRA_ARMS``) because the nonlinear family's H arm
    IS the ordinary certified kernel this pair splices. The property pinned here
    is unchanged -- an arm the pair does not implement may not be substituted --
    so the exemplar moved to one that stays another family's own arithmetic.
    """
    fields, layer, grid = build(xp)
    plans = {}
    reasons = {}
    selected = {"step_B": "PML", "update_H": "BFAST"}
    context = arms.StepContext(fields, layer, grid, sources=())
    fused_pairs.install_fused_pairs(plans, reasons, selected, context)
    assert not plans
    reason = reasons["fused_pair_cuda_fused_magnetic_pair"][0]
    assert "was selected by the 'BFAST' arm" in reason
    assert "may not substitute it" in reason


def test_the_declared_nonlinear_extra_arm_absorbs(xp):
    """``FUSED_PAIR_EXTRA_ARMS``'s one row, driven through the installer.

    On a run carrying an instantaneous chi2/chi3 the composer gives ``update_H``
    to the ``cuda_nonlinear`` family's ``nonlinear`` arm, whose kernel there IS
    ``fused_update_H_pml_real`` (``registry._TABLE``'s per-slot dict) -- so the
    pair implements that selection and the extras table says so. The absorb is
    the wiring half of the widening; the bit-identity half is
    ``gate_cuda_fused_magnetic_pair.py``'s nonlinear fixtures on a device.
    """
    fields, layer, grid = build(xp)
    plans = {}
    reasons = {}
    selected = {"step_B": "PML", "update_H": "nonlinear"}
    context = arms.StepContext(fields, layer, grid, sources=())
    fused_pairs.install_fused_pairs(plans, reasons, selected, context)
    assert "step_B" in plans and "update_H" in plans
    assert selected["step_B"] == selected["update_H"] == "fused magnetic pair"
    assert fused_pairs.FUSED_PAIR_EXTRA_ARMS["cuda_fused_magnetic_pair"] == (
        ("PML", "nonlinear"),)
    # The extras table declares only families the products table knows.
    assert set(fused_pairs.FUSED_PAIR_EXTRA_ARMS) <= set(fused_pairs.FUSED_PRODUCTS)


def test_an_unselected_slot_is_not_claimable_either(xp):
    fields, layer, grid = build(xp)
    plans, reasons = {}, {}
    context = arms.StepContext(fields, layer, grid, sources=())
    fused_pairs.install_fused_pairs(plans, reasons, {"step_B": "PML"}, context)
    assert not plans
    assert "is not selected by any arm" in \
        reasons["fused_pair_cuda_fused_magnetic_pair"][0]


def test_a_raising_builder_is_a_named_refusal_and_never_an_exception(xp, monkeypatch):
    """An opt-in optimisation that CRASHES the plan is strictly worse than one that
    refuses it -- and on a host with no CuPy the builder is exactly what raises."""
    product = dict(fused_pairs.FUSED_PRODUCTS["cuda_fused_magnetic_pair"])

    def explode(_context):
        raise ModuleNotFoundError("No module named 'cupy'")

    product["plan"] = explode
    monkeypatch.setitem(fused_pairs.FUSED_PRODUCTS, "cuda_fused_magnetic_pair",
                        product)
    fields, layer, grid = build(xp)
    plan = arms.plan_step(fields, layer, grid, sources=(), fuse=True)
    assert plan.selected["step_B"] == "PML"
    assert "raised ModuleNotFoundError" in \
        plan.reasons["fused_pair_cuda_fused_magnetic_pair"][0]


def test_a_product_with_no_absorb_declaration_is_refused_before_it_is_asked(xp,
                                                                            monkeypatch):
    """A missing absorb row is a fact about the TABLE, true of every configuration, so
    it is reported on every configuration rather than appearing with the grid."""
    monkeypatch.setitem(fused_pairs.FUSED_PRODUCTS, "cuda_invented_pair",
                        {"curl_slot": "step_B", "module": "nothing",
                         "coverage": lambda _c: (_ for _ in ()).throw(
                             AssertionError("the predicate must not be asked")),
                         "plan": lambda _c: None})
    fields, layer, grid = build(xp)
    plan = arms.plan_step(fields, layer, grid, sources=(), fuse=True)
    assert "has no absorb declaration" in \
        plan.reasons["fused_pair_cuda_invented_pair"][0]


def test_two_products_claiming_one_seam_leave_it_unfused_naming_both(xp, monkeypatch):
    """Two admitters is an ambiguity, not a pick -- and the wording deliberately
    avoids ``_select_slot``'s "admit this configuration", which is how a probe
    recognises an ambiguous SLOT."""
    monkeypatch.setitem(fused_pairs.FUSED_PRODUCTS, "cuda_twin_pair",
                        {"curl_slot": "step_B", "module": "nothing",
                         "coverage": lambda _c: fused_pairs.Coverage(True, ("ok",)),
                         "plan": lambda _c: None})
    monkeypatch.setitem(fused_pairs.FUSED_PAIR_ARMS, "cuda_twin_pair",
                        ("PML", "ordinary"))
    fields, layer, grid = build(xp)
    plan = arms.plan_step(fields, layer, grid, sources=(), fuse=True)
    reason = plan.reasons["fused_pair_cuda_twin_pair"][0]
    assert "all claim the step_B/update_H seam" in reason
    assert "admit this configuration" not in reason
    assert not isinstance(plan.plans["step_B"], fused_pairs.CudaFusedPairPlan)


# --------------------------------------------------------------------------
# 5. THE TABLE ITSELF
# --------------------------------------------------------------------------

def test_the_absorb_declaration_matches_the_arms_the_predicate_conjoins():
    """``FUSED_PAIR_ARMS`` is READ FROM THE PREDICATE, not assigned to it.

    ``covers_fused_magnetic_pair`` opens with the ``PML`` curl arm's own predicate and
    the ``ordinary`` constitutive arm's own predicate. If it ever conjoined different
    ones, absorbing those two slots would substitute a numerical product no arm
    admitted -- so the two are pinned together here rather than left to a comment.
    """
    from . import registry  # noqa: PLC0415 - the table, read for its labels

    source = pathlib.Path(family.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    predicate = next(node for node in tree.body
                     if isinstance(node, ast.FunctionDef)
                     and node.name == "covers_fused_magnetic_pair")
    called = {node.func.id for node in ast.walk(predicate)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert {"covers_real_pml_curl", "covers_real_pml_constitutive"} <= called

    by_predicate = {row["name"]: row["label"] for row in registry._TABLE}
    assert fused_pairs.FUSED_PAIR_ARMS["cuda_fused_magnetic_pair"] == (
        by_predicate["covers_real_pml_curl"],
        by_predicate["covers_real_pml_constitutive"])


def test_every_declared_seam_names_a_slot_pair_the_composer_walks():
    """Every seam row names two walked slots, and its seam name is one of three kinds.

    THE THREE KINDS ARE NOT SPELLINGS, they are what the DRIVER puts between the two
    consults, and each row says which:

    * ``'B'`` / ``'D'`` -- an INJECTION. The row names one of
      ``deposit_repair.SEAMS``' lists, because the driver injects inside that seam.
    * ``None`` -- NOTHING. The E->P row, and it MUST stay ``None``: the driver puts
      nothing between ``update_E`` and ``update_P`` (driver.py:3313/:3315), so there
      is no list to select, and any string would make ``_in_seam_indexed`` classify
      sources injected OUTSIDE the seam as inside it.
    * ``withdraw_hoist.SEAM`` -- the electric WITHDRAW, and nothing else. The H->D
      row, 2026-09-04. It is not a field letter for exactly the reason the E->P row
      is not: ``_in_seam_indexed`` reads ``pair != "B"`` as the electric list, so a
      letter here would bracket this seam with a deposit repair for an injection
      that happens in the NEXT seam. It is not ``None`` either, because ``None``
      means the seam is empty and this one is not -- the driver's ``withdraw`` loop
      sits between the consults and a spanning product has to perform it first.

    Exactly one row may say ``None`` and exactly one may say the withdraw seam.
    """
    from ..triton_kernels.launch import STEP_ORDER  # noqa: PLC0415

    kinds = {"injection": [], "empty": [], "withdraw": []}
    for curl_name, (update_name, pair_name) in fused_pairs.FUSED_PAIR_SEAMS.items():
        assert curl_name in STEP_ORDER and update_name in STEP_ORDER
        if pair_name is None:
            kinds["empty"].append(curl_name)
        elif pair_name == withdraw_hoist.SEAM:
            kinds["withdraw"].append(curl_name)
        else:
            assert pair_name in deposit_repair.SEAMS
            kinds["injection"].append(curl_name)
    assert kinds["empty"] == ["update_E"], kinds
    assert kinds["withdraw"] == ["update_H"], kinds
    assert fused_pairs.FUSED_PAIR_SEAMS["update_E"] == ("update_P", None)
    # THE NAME IS THE WITHDRAW MODULE'S OWN, not a fourth copy of the string: the
    # installer routes on it and ``withdraw_hoist`` refuses a span that is not its
    # own, so two spellings would be two things to drift.
    assert fused_pairs.FUSED_PAIR_SEAMS["update_H"] == (
        "step_D", withdraw_hoist.SEAM)
    assert withdraw_hoist.SEAM_SPAN == ("update_H", "step_D")


def test_the_h_to_d_seam_row_is_last_because_placed_first_it_would_displace_a_released_product():
    """The H->D row is APPENDED, and the key order is a correctness property.

    ``install_fused_pairs`` iterates ``FUSED_PAIR_SEAMS.items()`` in dict order and
    reads the LIVE ``selected`` as it goes. The H->D seam's FIRST slot is
    ``update_H``, which is the B->H seam's SECOND slot, and its SECOND slot is
    ``step_D``, which is the D->E seam's FIRST. Offered first, an H->D product would
    find both of those slots still carrying ARM labels -- because neither incumbent
    had run yet -- ``_pair_may_absorb`` would let it take them, and a released,
    gate-passing product would lose its seam on every row that reached it. Offered
    last, it finds a fused label in the slot and is refused by name.

    This is pinned as an ORDER rather than argued in a comment because dict order is
    invisible at the call site: a row moved while tidying the table would be a silent
    behaviour change on every corpus row, with no test to notice.
    """
    order = list(fused_pairs.FUSED_PAIR_SEAMS)
    assert order.index("update_H") > order.index("step_B"), order
    assert order.index("update_H") > order.index("step_D"), order
    assert order[-1] == "update_H", order


def test_the_h_to_d_seam_is_installed_through_the_withdraw_hoist_not_the_deposit_bracket():
    """``_install_fused_pair`` routes the seam NAME, so both tracks read one protocol.

    THE BRANCH IS IN THE INSTALLER AND NOT IN A FAMILY, which is the same argument
    that keeps the deposit bracket out of the families: a family free to decide for
    itself which mechanism its seam carries is a family that can decide wrong, and
    the wrong answer here (the deposit bracket, on a seam with no injection) computes
    and converges.
    """
    class Pair:
        label = "h to d weld"

    plans, selected = {}, {}
    fused_pairs._install_fused_pair(
        plans, selected, fields=None, pml=None, sources=(),
        pair_name=withdraw_hoist.SEAM, curl_name="update_H",
        update_name="step_D", label="h to d weld", pair=Pair())
    assert isinstance(plans["update_H"], withdraw_hoist.LeadingWithdrawPlan)
    assert plans["update_H"].span == ("update_H", "step_D")
    assert plans["update_H"].placement == withdraw_hoist.BEFORE_UPDATE_H
    # NO TRAILING PLAN: nothing is injected in this seam, so the second slot is the
    # no-op sentinel, never ``TrailingRepairPlan``.
    assert isinstance(plans["step_D"], launch_module.NoopPlan)
    assert not isinstance(plans["step_D"], deposit_repair.TrailingRepairPlan)
    assert selected == {"update_H": "h to d weld", "step_D": "h to d weld"}


def test_an_uninstallable_product_is_refused_by_name_and_claims_no_slot():
    """``INSTALLABLE = False`` is honoured by the INSTALLER, on every configuration.

    IT IS A FACT ABOUT THE PRODUCT, NOT ABOUT THE RUN, so it is asked beside the
    absorb declaration -- before the predicate -- and reported on every
    configuration. A product whose own module refuses the composition must not reach
    a slot even where its predicate admits and the arm table would let it absorb.
    """
    module = types.ModuleType("uninstallable_family")
    module.INSTALLABLE = False
    module.INSTALLABLE_REASON = "measured never installable on this slot path"
    sys.modules[f"{fused_pairs.__package__}.uninstallable_family"] = module
    product = {"curl_slot": "step_B", "module": "uninstallable_family",
               "coverage": lambda _context: Coverage(True, ()),
               "plan": lambda _context: None}
    try:
        assert fused_pairs._declared_uninstallable("invented", product) == (
            "invented declares INSTALLABLE = False: measured never installable on "
            "this slot path")
        plans, reasons, selected = {}, {}, {"step_B": "PML", "update_H": "ordinary"}
        with mock.patch.dict(fused_pairs.FUSED_PRODUCTS,
                             {"invented": product}, clear=False), \
             mock.patch.dict(fused_pairs.FUSED_PAIR_ARMS,
                             {"invented": ("PML", "ordinary")}, clear=False):
            fused_pairs.install_fused_pairs(plans, reasons, selected, object())
        assert not plans
        assert reasons["fused_pair_invented"] == (
            "invented declares INSTALLABLE = False: measured never installable on "
            "this slot path",)
        assert selected == {"step_B": "PML", "update_H": "ordinary"}
    finally:
        sys.modules.pop(f"{fused_pairs.__package__}.uninstallable_family", None)


def test_an_uninstallable_product_is_not_a_neighbouring_seam_claimant():
    """``_later_seam_claimant`` still filters on the flag -- and since 2026-09-05 the
    RELEASED B->H pair no longer depends on that filter.

    Read bare, the ``update_H`` row of ``_SLOT_OPENS_A_LATER_SEAM`` makes
    ``_later_seam_claimant`` name any admitting H->D product as the claimant of
    ``update_H``. A product that declares itself uninstallable claims nothing, so it is
    not one; the control is the same product with the flag at ``True``, which MUST be
    named -- otherwise this test would pass with the filter deleted.

    WHAT CHANGED: ``_neighbouring_seam_claimant`` never asks that question of the B->H
    pair at all, in EITHER flag state, because ``step_B`` sits in one row of
    ``FUSED_PAIR_SEAMS`` and the span is an end edge. Until this round the flipped
    flag made this function return the H->D stub as the B->H pair's claimant -- the
    displacement of a released product the key order is pinned to prevent, reached
    from the other direction and asserted here as the OLD behaviour.
    """
    module = types.ModuleType("h_to_d_stub")
    module.INSTALLABLE = False
    sys.modules[f"{fused_pairs.__package__}.h_to_d_stub"] = module
    product = {"curl_slot": "update_H", "module": "h_to_d_stub",
               "coverage": lambda _context: Coverage(True, ()),
               "plan": lambda _context: None}
    try:
        with mock.patch.dict(fused_pairs.FUSED_PRODUCTS,
                             {"h_to_d": product}, clear=False):
            assert fused_pairs._later_seam_claimant(
                {}, {}, {}, object(), "update_H") is None
            assert fused_pairs._neighbouring_seam_claimant(
                {}, {}, {}, object(), "cuda_fused_magnetic_pair",
                "step_B", "update_H") is None
            module.INSTALLABLE = True
            assert fused_pairs._later_seam_claimant(
                {}, {}, {}, object(), "update_H") == "h_to_d"
            assert fused_pairs._neighbouring_seam_claimant(
                {}, {}, {}, object(), "cuda_fused_magnetic_pair",
                "step_B", "update_H") is None
            assert fused_pairs._neighbouring_seam_claimant(
                {}, {}, {}, object(), "cuda_fused_magnetic_pair",
                "step_B", "update_H", ("step_B", "update_H")) is None
    finally:
        sys.modules.pop(f"{fused_pairs.__package__}.h_to_d_stub", None)


def test_slot_degrees_are_read_off_the_table_and_the_end_edges_are_b_to_h_and_e_to_p():
    """THE GUARD IS DERIVED, NOT SPELLED, and on this four-row table the derivation
    reproduces the released 2026-09-02 ruling: ``D_to_E`` is INTERIOR here (``update_E``
    sits in two rows) and yields to ``E_to_P``, where on the three-row Metal and Triton
    tables it is an end edge. This is what fails the day a fifth row is priced without
    thinking about position."""
    degrees = fused_pairs._slot_degrees(fused_pairs.FUSED_PAIR_SEAMS)
    assert degrees == {"step_B": 1, "update_H": 2, "step_D": 2, "update_E": 2,
                       "update_P": 1}, degrees
    spans = {curl: (curl, update)
             for curl, (update, _p) in fused_pairs.FUSED_PAIR_SEAMS.items()}
    end_edges = {curl for curl, span in spans.items()
                 if fused_pairs._is_end_edge_span(span, fused_pairs.FUSED_PAIR_SEAMS)}
    assert end_edges == {"step_B", "update_E"}, end_edges
    for interior in ("update_H", "step_D"):
        assert not fused_pairs._is_end_edge_span(spans[interior],
                                                 fused_pairs.FUSED_PAIR_SEAMS)
    # A three-slot span that reaches update_P is an end edge however it is asked --
    # the three-slot welds make no trade and are never withheld.
    assert fused_pairs._is_end_edge_span(("step_D", "update_E", "update_P"),
                                         fused_pairs.FUSED_PAIR_SEAMS)


def _composition(plan, slots=("step_B", "update_H", "step_D", "update_E", "update_P")):
    return {slot: plan.selected.get(slot) for slot in slots}


def test_a_registered_admitting_uninstallable_h_to_d_stand_in_does_not_cost_the_b_to_h_pair_its_seam():
    """ARRANGEMENT F OF THE TIE MEASUREMENT, ON THIS COMPOSER, FROM BOTH SIDES.

    A stand-in H->D product whose predicate ADMITS and which cannot install (no absorb
    row) leaves every slot's owner exactly as shipped -- the B->H pair keeps
    ``step_B``/``update_H`` -- and is refused on its missing row. Give it a row and it
    is refused instead by ``_pair_may_absorb`` naming the pair that installed first.
    Paired with the negative control (predicate REFUSES), which must also leave the
    composition unchanged, so the test cannot pass by the stand-in being invisible.
    Measured red on the Metal composer on 2026-09-04 (3 launches / 1 seam against the
    shipped 2 / 2); green here only because no H->D product had yet been registered on
    this track when the ``update_H`` row landed.
    """
    fields, layer, grid = build_dispersive(poles=0, pml_on=True)
    shipped = _composition(arms.plan_step(fields, layer, grid, sources=(), fuse=True))
    b_to_h = shipped["step_B"]
    assert b_to_h is not None and shipped["update_H"] == b_to_h, shipped
    assert shipped["step_D"] == shipped["update_E"] not in (None, b_to_h), shipped

    class StandInPlan:
        label = "fused H/D pair (stand-in)"

    name = "h_to_d_stand_in"
    module = types.ModuleType(name)
    sys.modules[f"{fused_pairs.__package__}.{name}"] = module

    def product(admit):
        return {"curl_slot": "update_H", "module": name,
                "coverage": lambda _context: Coverage(
                    admit, () if admit else ("the stand-in refuses by name",)),
                "plan": lambda _context: StandInPlan()}

    try:
        # F: admits, no absorb row.
        with mock.patch.dict(fused_pairs.FUSED_PRODUCTS, {name: product(True)},
                             clear=False):
            plan = arms.plan_step(fields, layer, grid, sources=(), fuse=True)
        assert _composition(plan) == shipped, _composition(plan)
        assert "has no absorb declaration" in plan.reasons[f"fused_pair_{name}"][0]
        # B: admits, WITH the row its two slots' arms would need.
        with mock.patch.dict(fused_pairs.FUSED_PRODUCTS, {name: product(True)},
                             clear=False), \
             mock.patch.dict(fused_pairs.FUSED_PAIR_ARMS, {name: ("ordinary", "PML")},
                             clear=False):
            plan = arms.plan_step(fields, layer, grid, sources=(), fuse=True)
        assert _composition(plan) == shipped, _composition(plan)
        assert plan.reasons[f"fused_pair_{name}"][0].startswith(
            f"update_H was selected by the {b_to_h!r} arm"), plan.reasons[f"fused_pair_{name}"]
        # G: refuses -- the negative control.
        with mock.patch.dict(fused_pairs.FUSED_PRODUCTS, {name: product(False)},
                             clear=False), \
             mock.patch.dict(fused_pairs.FUSED_PAIR_ARMS, {name: ("ordinary", "PML")},
                             clear=False):
            plan = arms.plan_step(fields, layer, grid, sources=(), fuse=True)
        assert _composition(plan) == shipped, _composition(plan)
        assert "refuses by name" in plan.reasons[f"fused_pair_{name}"][0]
    finally:
        sys.modules.pop(f"{fused_pairs.__package__}.{name}", None)
    assert _composition(arms.plan_step(fields, layer, grid, sources=(), fuse=True)) == shipped


def test_the_neighbouring_seam_rule_asks_the_later_seam_only():
    """THE EARLIER-NEIGHBOUR HALF WAS REMOVED 2026-09-04, and this pins why.

    It was redundant where it was meant to help — an H->D span's last slot is
    ``step_D``, which opens the D->E seam, so the later half names a claimant and
    returns first — and where it did fire it reversed a released ruling: the only
    product class whose last slot opens no seam is E->P, so it refused the
    polarization pair while the later half was already refusing the D->E product in
    that pair's favour. Both neighbours annihilated. See
    ``test_the_e_to_p_incumbent_survives_arbitration_on_a_dispersive_run``.
    """
    source = pathlib.Path(fused_pairs.__file__).read_text(encoding="utf-8")
    assert "EARLIER-NEIGHBOUR HALF WAS REMOVED" in source, (
        "the removal's reasoning is the record of a measurement; keep it with the code")
    tree = ast.parse(source)
    function = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "_neighbouring_seam_claimant")
    calls = {node.func.id for node in ast.walk(function)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert "_later_seam_claimant" in calls
    # The earlier half looped FUSED_PRODUCTS looking for the neighbour that would
    # lose a slot; nothing in this function iterates the product table any more.
    assert not [node for node in ast.walk(function)
                if isinstance(node, ast.Name) and node.id == "FUSED_PRODUCTS"]

def test_the_product_table_and_the_absorb_table_name_the_same_families():
    """A product the block can find but may not absorb for is refused rather than
    bracketed; a row in the absorb table with no product is a bound on nothing."""
    assert set(fused_pairs.FUSED_PRODUCTS) == set(fused_pairs.FUSED_PAIR_ARMS)
    # And the key is the MODULE'S OWN family name, not a second spelling of it: the
    # board, the census and `plan.reasons` all key on this string, and three tables
    # agreeing by coincidence is how a family goes missing from one of them.
    assert "cuda_fused_magnetic_pair" == family.FAMILY
    assert fused_pairs.FUSED_PRODUCTS[family.FAMILY]["curl_slot"] == family.SLOT


# --------------------------------------------------------------------------
# 6. THE DEVICE TEXT THIS EDIT DID NOT TOUCH
# --------------------------------------------------------------------------
#
# The deposit carry is a HOST change: two plan objects around a launch. Nothing
# below is new evidence about the kernel's arithmetic -- only a device gate is --
# but the claim "the device text is unchanged" is worth pinning where a laptop can
# check it, because that claim is what says the re-cut owed is a re-cut and not a
# re-certification of different bytes.

def test_the_lift_is_twelve_edits_and_no_silent_thirteenth():
    """``LIFT_EDITS`` is data so a gate can assert the list rather than the prose.

    FIVE UNTIL 2026-08-28, SEVEN SINCE, and the two that arrived are the ownership
    inversion's two halves -- the guard inside ``pml_apply_reg`` and the guarded,
    carry-bearing constitutive statement. TWELVE SINCE THE OWN-CELL HOIST: the
    twelve guarded pre-loads, the two derived ``_pre`` helpers, and the two sets of
    own-cell call rewrites. A thirteenth entry would be a thirteenth
    departure from the certified text and may not arrive quietly; the five that were
    there before must all still be there, which is what stops a rewrite from
    replacing one departure with another and keeping the count.
    """
    assert len(family.LIFT_EDITS) == 12
    lines = {edit["line"] for edit in family.LIFT_EDITS}
    assert {
        "    int own_z = !(...);",
        "__device__ __forceinline__ float pml_apply_reg(\n    ...)",
        "(nothing: appended after _REAL_CONSTITUTIVE_PRELUDE)",
        "        b_x = pml_apply_reg(Bx, fu_Bx, idx, curl, ..., own_x);",
        "        constitutive_apply(Hx, f_w_Hx, idx, b_x, kps_x[i], kms_int_x[i]);",
    } <= lines
    assert {
        "__device__ __forceinline__ void pml_apply(",
        "    f[idx] = (((f[idx] * kms_u) + fu_new) - fprev) * sinv_u;",
        "        pml_apply(Bx, fu_Bx, idx, curl, ...);",
        "    constitutive_apply(Hx, f_w_Hx, idx, Bx[idx], kps_x[i], kms_x[i]);",
        "        if (t < plane) Bx[base] = 0.0f;   (in_seam_passes.zero_metal_B)",
    } <= lines
    carried = [edit for edit in family.LIFT_EDITS
               if "OWNERSHIP INVERSION" in edit["why"]]
    assert len(carried) == 2, [edit["line"] for edit in carried]
    assert any("if (!owned) return 0.0f;" in edit["became"] for edit in carried)
    assert any("if (own_x) {" in edit["became"] for edit in carried)


def test_the_one_re_spelled_pass_still_emits_the_diagonal_and_a_plus_zero():
    """``zero_metal_carry`` needs none of the CuPy-importing siblings, so the ONE
    re-spelled pass in the product is checkable here.

    THE DIAGONAL, and it is the clause that would be silently wrong if the D
    family's table were reused: Bx clears on an x wall, By on y, Bz on z. The stored
    constant is ``+0.0f`` and never ``-0.0f`` (a signed zero is a different word),
    and both the register and the global store are emitted -- clearing only the
    register leaves global B unwiped for the next step's curl, clearing only global
    leaves the constitutive half reading the unwiped register.
    """
    text = family.zero_metal_carry()
    assert text.isascii()
    for component, wall, coordinate in (("Bx", "wall_x", "i"), ("By", "wall_y", "j"),
                                        ("Bz", "wall_z", "k")):
        register = f"b_{component[1]}"
        # THE OWNERSHIP GUARD JOINED THIS LINE ON 2026-08-28 and it is not
        # decoration: on a cell one of the two fills images, the array path's own
        # order overwrites the clear (zero_metal_B at driver.py:3295, the far fill
        # at :3296), so clearing it here would be a SECOND thread writing that word.
        assert (f"if (own_{component[1]} && {wall} && {coordinate} == 0) "
                f"{{ {register} = 0.0f; {component}[idx] = 0.0f; }}") in text
    assert "-0.0f" not in text
    # The off-diagonal (D) table would clear two components per wall; none of its
    # pairings may appear here.
    for wrong in ("wall_x && i == 0) { b_y", "wall_x && i == 0) { b_z"):
        assert wrong not in text
    # ... and the guard must be the component's OWN flag: own_y on the Bx line
    # would leave the x-wall plane cleared by a thread that does not own it.
    for wrong in ("own_y && wall_x", "own_z && wall_x", "own_x && wall_y"):
        assert wrong not in text


def test_the_emitter_refuses_by_name_where_the_certified_text_is_unreachable():
    """The defensive imports buy the predicate a laptop; they must not buy the
    EMITTER a degraded emit. There is nothing to splice, so it raises and says which
    halves are missing."""
    if family.step_curl_kernels is not None:
        pytest.skip("the certified halves import on this host; the refusal is unreachable")
    with pytest.raises(RuntimeError) as excinfo:
        family.fused_magnetic_pair_source()
    message = str(excinfo.value)
    assert "step_curl_kernels" in message and "constitutive_kernels" in message
    assert "covers_fused_magnetic_pair needs none of them" in message


def test_a_launch_is_refused_by_name_rather_than_by_attribute_error():
    """``_get_kernel`` is the one function here that needs the device library."""
    if family.cp is not None:
        pytest.skip("CuPy is importable on this host")
    with pytest.raises(RuntimeError, match="CuPy is not importable"):
        family._get_kernel()


def test_the_replaced_passes_are_the_products_own_declaration(xp):
    """``zero_metal_B`` is in neither slot name and is carried in registers, so the
    pair's ``replaces`` has to come from the module, not from the two slots."""
    fields, layer, grid = build(xp)
    context = arms.StepContext(fields, layer, grid, sources=())
    pair = fused_pairs._fused_magnetic_pair_plan(context)
    assert pair.replaces == tuple(family.REPLACES)
    assert pair.replaces_sub_steps == pair.replaces
    assert pair.launchable is True


def test_the_e_to_p_incumbent_survives_arbitration_on_a_dispersive_run():
    """THE 2026-09-02 TRADE RULING, PINNED FROM BOTH SIDES.

    ``_later_seam_claimant`` withholds the D->E pair because the polarization pair
    claims ``update_E``; nothing pinned the other direction, and on 2026-09-04 an
    earlier-neighbour rule briefly refused the polarization pair because a D->E
    product admits. Both neighbours annihilated and the seam went unserved, silently,
    because every test asserted who was WITHHELD and none asserted who INSTALLS.

    A neighbouring-seam rule that refuses the E->P incumbent has re-litigated a
    released ruling from the other side, and this test is the tripwire.

    EXTENDED 2026-09-05 WITH THE INSTALLED SIDE, driven through the composer: the
    polarization pair holds ``update_E`` AND ``update_P``, two pairs are installed
    over ``step_B..update_P`` rather than one, and the D->E product's recorded reason
    names ``cuda_fused_polarization_pair``. The three-slot welds are taken out of the
    table for that leg because on this fixture they supersede the two-slot trade
    (asserted first, as shipped); the trade is what this test is about.
    """
    fields, layer, grid = build_dispersive(poles=2, pml_on=True)
    context = arms.StepContext(fields, layer, grid, None, None, ())
    # The released direction still holds: the D->E pair loses update_E to the E->P pair.
    assert fused_pairs._later_seam_claimant(
        {}, {}, {}, context, "update_E", ("step_D", "update_E")
    ) == "cuda_fused_polarization_pair"
    # ...and the E->P incumbent is NOT refused in return -- an end edge, never asked.
    assert fused_pairs._neighbouring_seam_claimant(
        {}, {}, {}, context, "cuda_fused_polarization_pair",
        "update_E", "update_P", ("update_E", "update_P")) is None
    # The D->E pair, asked as a two-slot span, IS refused naming the E->P incumbent.
    withheld = fused_pairs._neighbouring_seam_claimant(
        {}, {}, {}, context, "cuda_dispersive_fused_electric_pair",
        "step_D", "update_E", ("step_D", "update_E"))
    assert withheld is not None and withheld[0] == "cuda_fused_polarization_pair"
    # The shape the rule does exist for is still refused, by the later half.
    refused = fused_pairs._neighbouring_seam_claimant(
        {}, {}, {}, context, "a_stub_h_to_d_product",
        "update_H", "step_D", ("update_H", "step_D"))
    assert refused is not None and refused[0] == "cuda_dispersive_fused_electric_pair"

    # WHO INSTALLS, through the composer. As shipped, the three-slot weld strictly
    # contains the D->E product's span and takes all three slots.
    shipped = arms.plan_step(fields, layer, grid, sources=(), fuse=True)
    assert shipped.selected["step_D"] == shipped.selected["update_E"] \
        == shipped.selected["update_P"] == "three-slot dispersive weld", shipped.selected
    assert "strictly contains" in shipped.reasons[
        "fused_pair_cuda_dispersive_fused_electric_pair"][0]
    # With the two-slot products alone, the trade: the polarization pair installs on
    # update_E/update_P and the D->E product is withheld naming it.
    two_slot = {name: product for name, product in fused_pairs.FUSED_PRODUCTS.items()
                if "three_slot" not in name}
    with mock.patch.dict(fused_pairs.FUSED_PRODUCTS, two_slot, clear=True):
        plan = arms.plan_step(fields, layer, grid, sources=(), fuse=True)
    e_to_p = plan.selected["update_P"]
    assert e_to_p is not None and plan.selected["update_E"] == e_to_p, plan.selected
    assert plan.selected["step_B"] == plan.selected["update_H"] is not None, plan.selected
    assert plan.selected["step_D"] not in (e_to_p, plan.selected["step_B"]), plan.selected
    pairs = {label for slot, label in plan.selected.items()
             if slot in ("step_B", "update_H", "step_D", "update_E", "update_P")
             and list(plan.selected.values()).count(label) >= 2}
    assert pairs == {plan.selected["step_B"], e_to_p}, pairs
    reason = plan.reasons["fused_pair_cuda_dispersive_fused_electric_pair"][0]
    assert reason.startswith("cuda_fused_polarization_pair admits this run's "
                             "update_E/update_P seam"), reason
    assert "fused_pair_cuda_fused_polarization_pair" not in plan.reasons
