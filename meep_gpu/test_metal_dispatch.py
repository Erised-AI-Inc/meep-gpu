"""The METAL table's dispatch ladder and its residency seam, on the laptop.

WHAT THIS PINS AND WHAT IT DELIBERATELY DOES NOT. The numbers the Metal kernels
produce are the device gates' business (``metal_kernels/fingerprints.json``, 62
welds) and the shipped composition is ``test_metal_planner_composition``'s. What
neither of those can see is the layer between them: which products the LADDER
admits, what happens to the ones it does not, and whether the residency bracket
that makes a device launch legal on a host-authoritative engine is placed where the
composer says it is. Those are decisions, they are measurable without a byte
comparison, and every one of them is silent when it is wrong — a mis-placed bracket
is a stale mirror, and a stale mirror reads as an arithmetic defect in a kernel that
is in fact correct.

THREE THINGS ARE HELD HERE:

1. **THE BRACKET RULE, PER PLAN SHAPE**, counted on a residency that records what it
   was asked to do. Six shapes reach ``launch.synced``: a bare device plan, a
   wrapper whose ``inner`` is its ``absorbed_by`` (the deposit repair's leading half
   and the withdraw hoist's), a plan with an ``absorbed_by`` and no ``inner`` (the
   sentinel and the trailing repair), a plan that declares
   ``performs_device_work = False``, and a split fill that answers
   ``run_near``/``run_far``. Each gets a different answer and each answer is a
   correctness claim.
2. **THE RELEASE TABLE'S OWN INTEGRITY** — every released label is one the registry
   can write, has an absorb declaration the ladder can see through, names a
   certification that RESOLVES in the ledger, and carries a per-arm axis row. That
   last one is the anti-forgery test: a typed row citing a weld that does not exist
   is invisible in an artifact, because the arm names a gate and the gate names
   nothing.
3. **THE LADDER'S REFUSALS** — that an un-admitted fused product refuses the whole
   plan by name, that a pending family refuses through its CONSTITUENTS as well as
   through its own label, and that the envelope names the axis it declined on.

THE SEAM THIS FILE WAITS ON. The rung-3 branch that routes a NumPy engine with an
MPS device into ``metal_dispatch`` is a separate edit batch on ``fastpath.py``.
Tests below that need a symbol from that batch SKIP WITH THE SYMBOL NAMED rather
than being written around: a test that quietly asserted less until the batch landed
would be a test nobody notices stopped measuring. :data:`_PENDING_FASTPATH` is the
list, and it doubles as the join checklist.
"""

from __future__ import annotations

import ast
import json
import os
import pathlib
import sys

import pytest

_PARITY = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       os.pardir, "parity", "meep_gpu"))
if _PARITY not in sys.path:
    sys.path.insert(0, _PARITY)

import metal_composition_matrix as matrix  # noqa: E402

from conftest import requires_resource_skip  # noqa: E402

from meep_gpu import fastpath, metal_dispatch  # noqa: E402
from meep_gpu.metal_kernels import arms, device, launch  # noqa: E402

PACKAGE_DIR = pathlib.Path(__file__).resolve().parent
ENVIRONMENT = matrix.prepare_environment()

#: The ``fastpath`` symbols this table's ladder is written against and which the
#: dispatch batch on that file has not landed yet. A test that needs one skips by
#: name, so the skip list IS the join checklist and an unskipped test after the
#: batch is a test that was never waiting.
_PENDING_FASTPATH = {
    "_finish": "the factored tail both ladders emit through",
    "candidate_tables": "the rung-3 branch that chooses a table from the hardware",
    "metal_hardware_present": "the MPS availability read rung 3 makes",
    "TABLE_SUBNORMAL_POLICY": "the per-table policy map rung 8bM reads",
    "TABLE_GOVERNED_EXECUTORS": "the per-table executor tuple rung 8bM drives",
}


#: The resource name the sanctioned skip declares. It is a genuinely-absent
#: ARTIFACT in the same sense a missing oracle file is: the symbols below exist only
#: after the dispatch edit batch lands on ``fastpath.py``, and until then the test
#: has nothing to measure. Declaring it by name means the session summary counts
#: these rather than letting them read as coverage.
FASTPATH_BATCH = "fastpath_metal_dispatch_batch"


@pytest.fixture(autouse=True, scope="module")
def _leave_the_process_as_this_file_found_it():
    """Put the FPU and the policy bookkeeping back when this file is done.

    RUNG 8bM INSTALLS, AND AN ATTAINED INSTALL DELIBERATELY STANDS. That is right in
    production — a process that asked to flush and got it should keep flushing — and
    it is a leak in a suite: the tests below drive the real ``_subnormal_gate`` for
    the Metal table, which puts this host's FPU on ``flush`` through the Darwin fenv
    lever and never takes it off.

    MEASURED 2026-09-10, which is why this is here rather than argued: with this file
    running first in the same process, seven ``test_triton_complex_fields`` tests
    fail with "the candidates were asked for under the IEEE-keep policy but this
    host's FPU measurably flushes float32 subnormals", and the same seven pass when
    that file runs alone. A test that leaks process state into its neighbours is the
    one failure a green isolated run cannot see, so the entry state is measured and
    restored here. ``test_subnormal_policy`` carries the per-test twin of this for
    the same reason.
    """
    from meep_gpu import backends, subnormal_policy  # noqa: PLC0415

    flushing_at_entry = backends.subnormals_flushed()
    try:
        yield
    finally:
        subnormal_policy._reset_for_tests()  # noqa: SLF001
        if backends.subnormals_flushed() != flushing_at_entry:
            subnormal_policy._drive_host_fpu(flushing_at_entry)  # noqa: SLF001


def _requires(symbol: str) -> None:
    """Skip, naming the exact ``fastpath`` symbol and what it is for."""
    if not hasattr(fastpath, symbol):
        requires_resource_skip(
            FASTPATH_BATCH,
            f"fastpath.{symbol} has not landed yet "
            f"({_PENDING_FASTPATH.get(symbol, 'the dispatch batch')})")


def _subnormal_gate_takes_a_table() -> bool:
    """Has ``_subnormal_gate`` grown its ``table`` argument?

    AN ARITY RATHER THAN A NAME, which is why this is not in
    :data:`_PENDING_FASTPATH`: the symbol EXISTS today with two parameters and the
    ladder calls it with three, so ``hasattr`` answers True while the call is a
    ``TypeError``. Asking the signature is the only reading that is not wrong.
    """
    import inspect  # noqa: PLC0415

    try:
        return len(inspect.signature(fastpath._subnormal_gate).parameters) >= 3  # noqa: SLF001
    except (TypeError, ValueError):  # pragma: no cover - a builtin would not sign
        return False


# ---------------------------------------------------------------------------
# Scaffolding
# ---------------------------------------------------------------------------

class CountingResidency(device.Residency):
    """A residency that records every bracket, in order, without a device.

    SUBCLASSED RATHER THAN STUBBED so the plans bind mirrors through the real
    registration path and the counts are about the real objects. The two ``sync``
    methods are the whole instrument: ``events`` is the ORDER, which is what
    separates "the bracket is present" from "the bracket is around the right thing".
    """

    def __init__(self) -> None:
        super().__init__()
        self.events: list = []

    def sync_in(self, names=None) -> None:  # noqa: ANN001
        self.events.append("in")
        self.syncs_in += 1

    def sync_out(self, names=None) -> None:  # noqa: ANN001
        self.events.append("out")
        self.syncs_out += 1


class Recorder:
    """A plan-shaped object that records when it ran, relative to the brackets."""

    performs_device_work = True

    def __init__(self, residency: CountingResidency, name: str = "plan") -> None:
        self.residency = residency
        self.name = name
        self.runs = 0
        self.launches = 0

    def run(self, *args, **kwargs) -> None:  # noqa: ANN002,ANN003,ARG002
        self.residency.events.append(f"run:{self.name}")
        self.runs += 1
        self.launches += 1


class SplitRecorder(Recorder):
    """A split fill: two passes the driver runs either side of the wall clear."""

    def run_near(self) -> None:
        self.residency.events.append(f"near:{self.name}")
        self.runs += 1

    def run_far(self) -> None:
        self.residency.events.append(f"far:{self.name}")
        self.runs += 1


class HostOnly:
    """The no-PML null arm's shape: it binds no buffer and stales no mirror."""

    performs_device_work = False

    def __init__(self, residency: CountingResidency) -> None:
        self.residency = residency
        self.runs = 0

    def run(self, *args, **kwargs) -> None:  # noqa: ANN002,ANN003,ARG002
        self.residency.events.append("run:null")
        self.runs += 1


class Absorbed:
    """``launch.NoopPlan``'s shape: an ``absorbed_by`` and no ``inner``."""

    def __init__(self, owner) -> None:  # noqa: ANN001
        self.absorbed_by = owner

    def run(self, *args, **kwargs) -> None:  # noqa: ANN002,ANN003,ARG002
        return None


class LeadingWrapper:
    """The deposit repair's leading half: HOST work first, then the inner launch."""

    def __init__(self, inner, residency: CountingResidency) -> None:  # noqa: ANN001
        self.inner = inner
        self.absorbed_by = inner
        self.residency = residency

    def run(self, *args, **kwargs) -> None:  # noqa: ANN002,ANN003,ARG002
        self.residency.events.append("host-work")
        self.inner.run()


def _probes():
    from meep_gpu.metal_kernels import (  # noqa: PLC0415
        complex_fields, cylindrical_complex, folded_complex, special_kz)

    return {"complex_probe": complex_fields.load_expansion_probe(),
            "beta_probe": special_kz.load_expansion_probe(),
            "folded_complex_probe": folded_complex.load_expansion_probe(),
            "cylindrical_complex_probe":
                cylindrical_complex.load_expansion_probe()}


def _compose(pair, *, fuse=True, fuse_labels=None, sources=()):
    """The ladder's rung 5M, minus the ladder: the two-pass scout and the bracket."""
    fields, pml = pair
    residency = CountingResidency()
    probes = _probes()
    scout = launch.plan_step(fields, pml, residency=device.Residency(),
                             sources=sources, fuse=fuse, fuse_labels=fuse_labels,
                             **probes)
    covered = set()
    for slot, product in scout.plans.items():
        covered.add(slot)
        covered.update(getattr(launch.declaring_plan(product),
                               "replaces_sub_steps", ()) or ())
    live = launch.live_sub_steps(fields, pml, sources) or ()
    synced = tuple(name for name in live if name not in covered)
    plan = launch.plan_step(fields, pml, residency=residency, sources=sources,
                            synced=synced, fuse=fuse, fuse_labels=fuse_labels,
                            **probes)
    launch.wrap_for_residency(plan, residency)
    return plan, residency


def _run_a_step(plan, residency) -> None:
    """Run every slot the composer filled, in the driver's own order.

    The far pass is dispatched separately and only where the plan OWNS it, which is
    what the driver does: ``zero_metal_*`` sits between the near and far fills, so a
    plan may not fuse across it unless it declares that it did.
    """
    from meep_gpu.triton_kernels.launch import STEP_ORDER  # noqa: PLC0415

    for slot in STEP_ORDER:
        entry = plan.plans.get(slot)
        if entry is None:
            continue
        if slot in ("fill_B", "fill_D") and hasattr(entry, "run_near"):
            entry.run_near()
            far = getattr(entry, "run_far", None)
            if far is not None:
                far()
            continue
        entry.run()


# ---------------------------------------------------------------------------
# 0. The precondition
# ---------------------------------------------------------------------------

def test_the_environment_this_suite_needs_is_present():
    """A vacuous suite must fail rather than pass quietly.

    Without ``flush`` every Metal predicate refuses on the policy clause and the
    composer fills nothing, so every bracket count below would be zero and every
    "no ambiguity" assertion would hold over an empty set. The matrix module's own
    ``prepare_environment`` is what sets it, and this asserts the result rather than
    trusting the call.
    """
    assert os.environ.get("MEEP_GPU_SUBNORMAL_POLICY") == "flush"
    from meep_gpu.metal_kernels import subnormal  # noqa: PLC0415

    assert subnormal.mps_policy_reasons() == [], (
        "the MPS executor refuses this process's resolved policy; every predicate "
        "below would refuse and this suite would measure nothing")


# ---------------------------------------------------------------------------
# 1. The bracket rule, per plan shape
# ---------------------------------------------------------------------------

def test_a_bare_device_plan_is_bracketed_and_the_order_is_in_run_out():
    residency = CountingResidency()
    plan = Recorder(residency)
    wrapped = launch.synced(plan, residency)
    wrapped.run()
    assert residency.events == ["in", "run:plan", "out"], residency.events


def test_a_wrapper_that_does_host_work_first_is_bracketed_around_its_INNER():
    """The placement claim, and the one that is silent when it is wrong.

    ``deposit_repair.LeadingRepairPlan`` saves the deposit points and
    ``withdraw_hoist.LeadingWithdrawPlan`` performs the seam's electric withdraw
    BEFORE their launch. A bracket around the whole slot would upload before that
    host work, so the launch would read a ``D`` still holding the previous step's
    standing dipole — the exact defect the hoist exists to prevent, and one that
    produces a plausible wrong answer rather than an error.
    """
    residency = CountingResidency()
    inner = Recorder(residency, "inner")
    wrapper = LeadingWrapper(inner, residency)
    wrapped = launch.synced(wrapper, residency)
    assert wrapped is wrapper, "the wrapper stays in the slot; its inner is wrapped"
    wrapped.run()
    assert residency.events == ["host-work", "in", "run:inner", "out"], residency.events


def test_a_plan_that_names_an_absorber_and_has_no_inner_is_left_alone():
    """The sentinel and the trailing repair: the launch is elsewhere.

    Their host writes are carried by the NEXT launch's ``sync_in``, which is the
    host-authoritative invariant working rather than a gap in it.
    """
    residency = CountingResidency()
    owner = Recorder(residency, "owner")
    absorbed = Absorbed(owner)
    assert launch.synced(absorbed, residency) is absorbed
    absorbed.run()
    assert residency.events == []


def test_a_plan_that_performs_no_device_work_is_left_alone():
    residency = CountingResidency()
    null = HostOnly(residency)
    assert launch.synced(null, residency) is null
    null.run()
    assert residency.events == ["run:null"]


def test_run_near_and_run_far_are_each_bracketed_separately():
    """The split fill, and why the two names go through ``__getattr__``.

    ``FastPathPlan.dispatch`` reaches a split fill through
    ``hasattr(plan, "run_near")`` and then calls it, so plain attribute forwarding
    would launch the near pass UNBRACKETED. Defining the two as real methods would
    be worse: ``hasattr`` would then answer True for ``MirrorGhostFillPlan``, which
    deliberately has none because it fuses both passes, and the dispatch would take
    the split branch and raise.
    """
    residency = CountingResidency()
    wrapped = launch.synced(SplitRecorder(residency, "fill"), residency)
    assert hasattr(wrapped, "run_near")
    wrapped.run_near()
    wrapped.run_far()
    assert residency.events == ["in", "near:fill", "out",
                                "in", "far:fill", "out"], residency.events


def test_a_fused_fill_that_has_no_run_near_is_not_made_to_look_split():
    """The control for the test above: ``hasattr`` must stay False on a fused fill."""
    residency = CountingResidency()
    wrapped = launch.synced(Recorder(residency, "fused-fill"), residency)
    assert not hasattr(wrapped, "run_near")
    with pytest.raises(AttributeError):
        wrapped.run_near()


def test_the_wrapper_declares_the_pair_identity_rather_than_forwarding_it():
    """``fastpath._pair_identity`` must answer the SAME integer for both slots.

    A fused pair is identified by the object that does the work — ``absorbed_by``
    where there is one, the entry itself otherwise. A bare pair plan names nobody,
    so a wrapper that merely forwarded the missing attribute would answer
    ``id(wrapper)`` on the curl slot while the sentinel in the absorbed slot
    answered ``id(pair)``: the two halves of one pair would silently stop being one
    pair, disarming ``_split_pairs`` and the mid-step licence clause.
    """
    residency = CountingResidency()
    pair = Recorder(residency, "pair")
    curl = launch.synced(pair, residency)
    absorbed = Absorbed(pair)
    assert (fastpath._pair_identity(curl)        # noqa: SLF001
            == fastpath._pair_identity(absorbed))  # noqa: SLF001


# ---------------------------------------------------------------------------
# 2. The bracket count on a real composition
# ---------------------------------------------------------------------------

def test_an_ordinary_pml_step_carries_two_brackets_and_no_more():
    """MEASURED, not derived: two pairs, and only the LEADING slot of each launches.

    Reproduced 2026-09-10 through the real driver on ``pml_2d``: 45 mirrors, 768
    ``sync_in`` and 768 ``sync_out`` over 384 complete steps. The count is the whole
    residency claim in one number — one bracket per launch and none around the
    slots a launch absorbed.
    """
    plan, residency = _compose(matrix.cart())
    labels = set(metal_dispatch.fused_labels())
    fused_slots = [slot for slot, arm in (plan.selected or {}).items()
                   if arm in labels]
    assert len(fused_slots) == 4, plan.selected
    _run_a_step(plan, residency)
    assert residency.syncs_in == 2, residency.events
    assert residency.syncs_out == 2, residency.events
    assert residency.names, "an empty mirror registry makes every count above vacuous"


def test_a_live_update_P_adds_exactly_one_bracket():
    """The dispersive step: the same two pairs plus the ADE slot's own launch."""
    plan, residency = _compose(matrix.dispersive(matrix.cart()))
    assert "update_P" in plan.plans, plan.selected
    _run_a_step(plan, residency)
    assert residency.syncs_in == 3, (residency.events, plan.selected)
    assert residency.syncs_out == 3, residency.events


def test_the_update_P_slot_holds_ONE_plan_and_not_a_list():
    """The asymmetry ``FastPathPlan.dispatch`` has to know about, pinned here.

    The Triton slot holds a LIST of per-susceptibility plans, each taking
    ``fields.drive_field``; this table fills it with ONE ``MetalAdeUpdatePPlan``
    whose ``run()`` takes a contract. Handing it a drive field raises ``KeyError``
    on the dispatch path, where the contract says exceptions PROPAGATE — measured on
    ``dispersive_2d`` and ``folded_dispersive_2d`` before the branch learned the
    difference.
    """
    plan, _residency = _compose(matrix.dispersive(matrix.cart()))
    entry = plan.plans["update_P"]
    assert not isinstance(entry, (list, tuple)), entry
    assert hasattr(entry, "run")


def test_wrap_for_residency_returns_the_same_object_it_was_handed():
    """The mutation is the point: ``fastpath`` reads the slots back out of it."""
    plan, residency = _compose(matrix.cart())
    assert launch.wrap_for_residency(plan, residency) is plan


# ---------------------------------------------------------------------------
# 3. The release table's integrity
# ---------------------------------------------------------------------------

def test_every_released_label_is_one_the_registry_can_write():
    labels = set(metal_dispatch.fused_labels())
    for arm in metal_dispatch.METAL_RELEASED_FUSED_ARMS:
        assert arm in labels, (
            f"{arm!r} is released and no registry weld writes it; the composer can "
            f"never put it in a slot, so the row licenses nothing")


def test_every_released_label_has_an_absorb_declaration_the_ladder_can_see_through():
    """A fused product writes ONE label into BOTH slots it absorbs.

    So the arms it implements disappear from ``selected``, and a rung reading the
    label alone would pass over a PENDING one underneath it. The ladder refuses a
    fused label it cannot name the constituents of; a released label that could not
    be named would be refused by that rung on every run.
    """
    constituents = metal_dispatch.fused_arm_constituents()
    for arm in metal_dispatch.METAL_RELEASED_FUSED_ARMS:
        assert arm in constituents, f"{arm!r} has no FUSED_PAIR_ARMS absorb row"
        assert constituents[arm], f"{arm!r} declares an empty constituent set"


def test_every_certification_row_resolves_to_a_readable_ledger_entry():
    """THE ANTI-FORGERY TEST, and it is the one that catches a row nobody can check.

    A typed row citing a weld that does not exist is invisible in an artifact: the
    arm names a gate, the gate names nothing, and only a reader who opens the ledger
    can tell. The twin of the Triton suite's
    ``test_every_recorded_certification_points_at_a_readable_record``.
    """
    assert metal_dispatch.unresolved_certification_rows() == ()
    ledger = metal_dispatch.fingerprints()
    assert ledger, "metal_kernels/fingerprints.json is unreadable here"
    from meep_gpu import metal_runs  # noqa: PLC0415

    for arm, (family, gate) in metal_dispatch.ARM_CERTIFICATION.items():
        assert family and gate, arm
        entry = ledger.get(gate)
        assert isinstance(entry, dict), f"{arm} names {gate}, which the ledger lacks"
        # THE RUN'S FACTS LIVE IN ITS ARCHITECTURE'S RECORD, and only a live one
        # describes the bytes the entry pins.
        runs = metal_runs.live_runs(entry)
        assert runs, f"{arm} names {gate}, which has no live per-architecture run"
        for architecture, run in runs.items():
            assert run.get("recorded_utc"), (
                f"{arm} names {gate}, whose {architecture} run records no time")
            assert run.get("host"), (
                f"{arm} names {gate}, whose {architecture} run records no host")


def test_the_release_rows_are_self_consistent():
    """The module's own check, run as a test so a drifted row is red rather than latent."""
    assert metal_dispatch.release_rows_without_a_weld() == ()


def test_every_released_arm_has_a_per_arm_axis_row_and_the_absence_fails_closed():
    """THE FAIL-CLOSED TWIN, both directions.

    The shared envelope is only the axes EVERY released arm agreed on, so an arm
    missing from the per-arm table has no statement about the axes that separate the
    arms from each other — and admitting on silence is the shape of every over-claim
    the table exists to prevent. The second half drives the absence and requires the
    refusal to name it.
    """
    for arm in metal_dispatch.METAL_RELEASED_FUSED_ARMS:
        assert arm in metal_dispatch.METAL_FUSED_RELEASE_ARM_AXES, arm
    shape = {"dimensions": 2, "pml_active": True, "bfast": False,
             "nonlinearity": False, "off_diagonal_epsilon": False,
             "cylindrical": False, "complex_storage": False, "bloch": False,
             "beta": 0, "conductivity": False, "susceptibilities": 0}
    why = metal_dispatch.fused_release_arm_reasons_metal(shape, "an arm nobody typed")
    assert why and "no per-arm axis row" in why[0], why


def test_an_unreadable_run_shape_refuses_the_release_rather_than_admitting_it():
    """The opposite rule from the device rung's, and deliberately so.

    That rung does not refuse a capability it could not read, because refusing on an
    unread fact ASSERTS one. Here the decision is an ADMISSION, and admitting on an
    unread fact asserts one.
    """
    why = metal_dispatch.fused_release_reasons_metal({"unreadable": "grid.xp"})
    assert why and "grid.xp" in why[0], why
    assert metal_dispatch.released_fused_arms_metal({"unreadable": "grid.xp"}) == ()


def test_the_envelope_names_the_axis_it_declined_on():
    """A predicate that has only ever said yes is not known to be able to say no.

    IT ASKS BOTH HALVES OF THE RELEASE, from 2026-09-12, and that is the point of
    the rewrite rather than an embellishment. ``off_diagonal_epsilon`` moved from
    the SHARED envelope to the PER-ARM axes that day. A test that kept asking only
    the shared table would have gone quiet about the axis instead of going red —
    it would still have found a shared row saying no (there are three others) and
    reported the envelope healthy while the axis it was written for had moved out
    from under it.
    """
    # THE SHARED HALF, which on 2026-09-17 stopped pinning ``pml_active`` (and
    # ``bfast``) the same way it stopped pinning ``off_diagonal_epsilon`` on 09-12:
    # three arms were released whose own axes say pml_active=False / bfast=True,
    # and a shared row consulted first refused them with an EMPTY admitted set
    # while their per-arm reasons read clean. So the shared half now PASSES this
    # shape, and the refusal that used to live here is asked of an arm instead --
    # the same migration this test already follows for the off-diagonal axis
    # below, and the reason this test exists at all.
    no_pml = {"dimensions": 2, "pml_active": False, "bfast": False,
              "nonlinearity": False, "off_diagonal_epsilon": False,
              "cylindrical": False, "complex_storage": False, "bloch": False,
              "beta": 0, "conductivity": False, "susceptibilities": 0}
    assert metal_dispatch.fused_release_reasons_metal(no_pml) == ()
    # Nothing is admitted on this shape even so: the no-PML electric pair, the one
    # arm built for pml_active=False, pins susceptibilities=2 (its case carries an
    # ADE), and every PML arm refuses BY NAME on the axis that moved.
    assert metal_dispatch.released_fused_arms_metal(no_pml) == ()
    pml_why = metal_dispatch.fused_release_arm_reasons_metal(
        no_pml, "fused magnetic B/H pair")
    assert pml_why and "pml_active=False, not True" in pml_why[0], pml_why
    assert any("measured" in reason for reason in pml_why), pml_why

    # THE PER-ARM HALF, on the axis that moved there. The shared table PASSES on an
    # off-diagonal grid now, so anything refused here is refused by an arm's own
    # axes — which is the half that would otherwise go untested.
    offdiag = dict(no_pml, pml_active=True, off_diagonal_epsilon=True,
                   dimensions=3)
    assert metal_dispatch.fused_release_reasons_metal(offdiag) == ()
    admitted = metal_dispatch.released_fused_arms_metal(offdiag)
    assert admitted == ("fused magnetic B/H pair",), admitted

    # And the refusal must be BY NAME: every other arm says the axis out loud —
    # an arm with NO off-diagonal corner by its absence from the corner table, and
    # the one other arm that HAS a corner (the folded magnetic pair) by the corner
    # itself, whose dimensions point is 2 and the sphere is 3-D. That arm is refused
    # on the sphere by its own row as well (no fold — its dimensions row admits 3-D
    # since folded_3d), and both reasons are present rather than one standing in
    # for the other.
    named = [arm for arm in metal_dispatch.METAL_RELEASED_FUSED_ARMS
             if any("off_diagonal_epsilon" in reason for reason in
                    metal_dispatch.fused_release_arm_reasons_metal(offdiag, arm))]
    # The arms that do NOT say the axis out loud are exactly the ones an
    # off-diagonal grid is FOR: the ordinary magnetic pair (its corner admits
    # this sphere) and the three off-diagonal electric pairs released 2026-09-17,
    # which exist only for off-diagonal grids and are refused here on OTHER axes
    # (dimensions, the fold, complex storage) rather than on this one.
    built_for_offdiag = {"fused magnetic B/H pair",
                         "off-diagonal fused electric D/E pair",
                         "folded off-diagonal fused electric D/E pair",
                         "complex no-PML off-diagonal fused electric D/E pair"}
    assert set(named) == (set(metal_dispatch.METAL_RELEASED_FUSED_ARMS)
                          - built_for_offdiag), named
    folded_why = metal_dispatch.fused_release_arm_reasons_metal(
        offdiag, "folded fused B/H pair")
    assert any("folded is absent" in reason for reason in folded_why), folded_why
    assert any("only at dimensions in [2]" in reason for reason in folded_why), folded_why


def test_the_off_diagonal_admission_is_bounded_to_the_corner_its_cases_sat_at():
    """THE REGRESSION. This assertion is why the arm's coverage is not unbounded.

    WHAT HAPPENED, 2026-09-12. ``off_diagonal_epsilon`` moved from the shared
    envelope to the per-arm table, and ``fused magnetic B/H pair`` was given no
    replacement row because it drives BOTH values of the axis — which is the
    table's documented rule and which was, here, the wrong move. The True side was
    driven by exactly ONE case, ``pml_3d``, at ONE corner of the axes the row
    leaves open. Omitting the axis therefore admitted the arm on 1-D and 2-D
    off-diagonal grids that no case drives, and the shipped composer was measured
    INSTALLING a fused pair on a 2-D off-diagonal grid. It was caught by audit, not
    by a test — so this is the test.

    IT ASSERTS THE CORNER AND EVERY FAIL-CLOSED EDGE, because the previous shape of
    this test passed while the defect was live: it only ever asked about
    ``dimensions=3``, which is the one point that was correct. The corner is now
    two-dimensional for the ordinary pair (``offdiag_2d`` joined ``pml_3d``) and
    one point for the folded pair (``folded_offdiag_2d``), and each open axis of
    each corner is tested from outside as well as from inside.
    """
    shape = {"pml_active": True, "bfast": False, "nonlinearity": False,
             "cylindrical": False, "complex_storage": False, "bloch": False,
             "beta": 0, "conductivity": False, "susceptibilities": 0}
    arm = "fused magnetic B/H pair"
    folded_arm = "folded fused B/H pair"

    # 2-D (offdiag_2d, the cylinder) and 3-D (pml_3d, the sphere) off-diagonal are
    # driven, lossless and dispersion-free, and are admitted. Since 2026-09-17 the
    # 2-D point also admits ``off-diagonal fused electric D/E pair`` -- the D-seam
    # product released FOR offdiag_2d, whose corner is empty because its row pins
    # every axis to that one case (dimensions 2 among them, so it stays out of 3-D).
    electric = "off-diagonal fused electric D/E pair"
    inside = dict(shape, dimensions=2, off_diagonal_epsilon=True)
    assert metal_dispatch.released_fused_arms_metal(inside) == (arm, electric)
    inside = dict(shape, dimensions=3, off_diagonal_epsilon=True)
    assert metal_dispatch.released_fused_arms_metal(inside) == (arm,)

    # 1-D off-diagonal is NOT driven and must be refused, by a reason that names
    # the axis and the values that were.
    at_1d = dict(shape, dimensions=1, off_diagonal_epsilon=True)
    assert metal_dispatch.released_fused_arms_metal(at_1d) == ()
    why = metal_dispatch.fused_release_arm_reasons_metal(at_1d, arm)
    assert any("dimensions=1" in reason for reason in why), why
    assert any("only at dimensions in [2, 3]" in reason for reason in why), why

    # The OTHER two open axes of the row are bounded too: the arm runs with a
    # conductivity and over a susceptibility on DIAGONAL grids, and off-diagonal
    # at neither. Each is refused from outside the corner, naming its own axis.
    for axis, outside_value in (("conductivity", True), ("susceptibilities", 1)):
        outside = dict(shape, dimensions=2, off_diagonal_epsilon=True)
        outside[axis] = outside_value
        assert metal_dispatch.released_fused_arms_metal(outside) == (), axis
        why = metal_dispatch.fused_release_arm_reasons_metal(outside, arm)
        assert any(f"{axis}={outside_value!r}" in reason for reason in why), why
        assert any(f"only at {axis} in" in reason for reason in why), why

    # The DIAGONAL grids the arm really was driven on at those corners stay in.
    for dimensions in (1, 2, 3):
        inside = dict(shape, dimensions=dimensions, off_diagonal_epsilon=False)
        assert arm in metal_dispatch.released_fused_arms_metal(inside), dimensions
    for axis, value in (("conductivity", True), ("susceptibilities", 1)):
        inside = dict(shape, dimensions=2, off_diagonal_epsilon=False)
        inside[axis] = value
        assert arm in metal_dispatch.released_fused_arms_metal(inside), axis

    # FAILS CLOSED on an unread axis of the corner: admitting on a fact that did
    # not read would assert the fact, which is this module's rule everywhere else.
    unread = dict(shape, off_diagonal_epsilon=True)
    assert metal_dispatch.released_fused_arms_metal(unread) == ()
    why = metal_dispatch.fused_release_arm_reasons_metal(unread, arm)
    assert any("dimensions did not read" in reason for reason in why), why

    # THE FOLDED PAIR'S CORNER IS ONE POINT: susceptibilities=0. Its row pins every
    # other axis already, so the corner lists nothing else — and its one open axis
    # is exactly the one folded_2d and folded_dispersive_2d drive both values of.
    folded = dict(shape, dimensions=2, folded="mirror plane on Y",
                  off_diagonal_epsilon=True)
    # Since 2026-09-17 this point also admits the folded off-diagonal electric
    # pair released FOR folded_offdiag_2d -- the folded twin of the 2-D case above,
    # with the same empty corner for the same reason. The susceptibility edge below
    # still refuses BOTH, which is the bound this test exists to hold.
    folded_electric = "folded off-diagonal fused electric D/E pair"
    assert metal_dispatch.released_fused_arms_metal(folded) == (folded_arm,
                                                                 folded_electric)
    dispersive = dict(folded, susceptibilities=1)
    assert metal_dispatch.released_fused_arms_metal(dispersive) == ()
    why = metal_dispatch.fused_release_arm_reasons_metal(dispersive, folded_arm)
    assert any("susceptibilities=1" in reason for reason in why), why
    assert any("only at susceptibilities in [0]" in reason for reason in why), why
    # ...and the diagonal dispersive fold stays admitted, which is what makes the
    # corner a bound on off-diagonal coverage rather than a new row on the axis.
    assert folded_arm in metal_dispatch.released_fused_arms_metal(
        dict(dispersive, off_diagonal_epsilon=False))

    # And no OTHER arm has off-diagonal coverage at any corner -- other than the
    # three released 2026-09-17 that exist ONLY for off-diagonal grids. Those are
    # not exempt from being bounded: each is pinned to its one case by its own row
    # (dimensions, the fold, complex storage) and the 1-D / 3-D / folded / dispersive
    # edges above already show them staying out where their cases did not go. They
    # are exempt from THIS clause because "has no off-diagonal coverage" is the
    # opposite of what they are for.
    built_for_offdiag = {electric, folded_electric,
                         "complex no-PML off-diagonal fused electric D/E pair"}
    for other in metal_dispatch.METAL_RELEASED_FUSED_ARMS:
        if other in (arm, folded_arm) or other in built_for_offdiag:
            continue
        for dimensions in (1, 2, 3):
            for fold in (None, "mirror plane on Y"):
                probe = dict(shape, dimensions=dimensions, off_diagonal_epsilon=True)
                if fold:
                    probe["folded"] = fold
                assert other not in metal_dispatch.released_fused_arms_metal(probe)


def test_each_off_diagonal_corner_lists_exactly_the_axes_its_row_leaves_open():
    """The rule the corner table states about itself, checked rather than trusted.

    A corner names every axis the arm's row in ``METAL_FUSED_RELEASE_ARM_AXES``
    leaves OPEN — absent, or a set of more than one value — and no axis the row
    already pins: one fact with two homes is one that can disagree with itself,
    and an open axis MISSING from the corner is the original over-claim in a new
    place (off-diagonal coverage silently extended across a value no off-diagonal
    case carried).
    """
    # The shape axes an arm's row may speak to: everything ``fastpath._run_shape``
    # reads that is not a shared-envelope row, not the off-diagonal axis itself,
    # and not descriptive (grid_shape, k_point, m).
    shared = {row[0] for row in metal_dispatch.METAL_FUSED_RELEASE_ENVELOPE}
    speakable = {"dimensions", "cylindrical", "folded", "complex_storage", "bloch",
                 "beta", "conductivity", "susceptibilities"} - shared
    for arm, corner in metal_dispatch.METAL_OFF_DIAGONAL_CORNERS.items():
        assert arm in metal_dispatch.METAL_RELEASED_FUSED_ARMS, arm
        row = dict((axis, value) for axis, value, _ in
                   metal_dispatch.METAL_FUSED_RELEASE_ARM_AXES[arm])
        pinned = {axis for axis, value in row.items()
                  if not isinstance(value, frozenset) or len(value) == 1}
        open_axes = speakable - pinned
        assert set(corner) == open_axes, (arm, sorted(corner), sorted(open_axes))
        for axis, driven in corner.items():
            assert isinstance(driven, frozenset) and driven, (arm, axis)
            if isinstance(row.get(axis), frozenset):
                assert driven <= row[axis], (arm, axis, driven, row[axis])


def test_the_pending_table_is_empty_and_its_last_two_arms_are_released_on_the_group_key():
    """The pending rung still refuses something REAL when it holds anything, and it
    holds nothing: the two families it held were released on the same group-key
    ruling as their three siblings, and their certification rows resolve.

    A pending entry naming a label no registry writes would be a refusal of
    nothing; a released arm still listed as pending would be both admitted and
    stopped. Neither can be true of an empty table, and the former entries are
    checked to be released with a certification that names the group weld their
    cells are bound by.
    """
    labels = set(metal_dispatch.fused_labels())
    for arm in metal_dispatch.METAL_PENDING_DEVICE_GATE_ARMS:
        assert arm in labels, f"{arm!r} is pending and no registry weld writes it"
        assert arm not in metal_dispatch.METAL_RELEASED_FUSED_ARMS, (
            f"{arm!r} is both released and pending")
    assert metal_dispatch.METAL_PENDING_DEVICE_GATE_ARMS == {}
    for arm in ("complex fused electric D/E pair", "folded complex fused D/E pair"):
        assert arm in labels
        assert arm in metal_dispatch.METAL_RELEASED_FUSED_ARMS, arm
        family, key = metal_dispatch.ARM_CERTIFICATION[arm]
        assert key == "metal_tranche7_fused_pairs_device_gate", (arm, key)


def test_the_fused_label_predicate_is_the_registry_and_not_the_triton_spelling():
    """``arm_is_fused`` answers False for every Metal label, and that is correct.

    It is the TRITON vocabulary — a ``"fused pair"`` prefix plus two names — and
    widening it would make one predicate answer for two vocabularies. Reading the
    registry's own ``is_weld`` flag instead means a new Metal weld is fused the
    moment it registers, with nothing to remember to update.
    """
    metal_labels = metal_dispatch.fused_labels()
    assert metal_labels, "the registry declares no weld at all"
    assert not any(fastpath.arm_is_fused(label) for label in metal_labels), (
        "a Metal label is now recognised by the Triton predicate; one of the two "
        "vocabularies has moved and the ladder reads the wrong one")
    assert fastpath.arm_is_fused("fused pair B"), (
        "the control: the Triton predicate still answers for its own vocabulary")


# ---------------------------------------------------------------------------
# 4. The ladder's refusals
# ---------------------------------------------------------------------------

def _drive(pair, *, admitted=(), sources=(), finish=None):
    """Drive :func:`metal_dispatch.decide` against a composed configuration.

    ``finish`` is passed rather than imported precisely so a probe can drive the
    ladder with its own tail — which is what this is. The default records the call
    and returns a sentinel, so a test can tell "reached the tail" from "refused".
    """
    fields, pml = pair
    grid = getattr(fields, "grid", None)
    # THE RECORD IS THE LADDER'S OWN SHAPE, not a bare dict. ``_refuse`` announces
    # through ``_announce``, which reads the enable and kill-switch blocks
    # ``_base_record`` fills; handing ``decide`` an empty dict would make every
    # refusal a KeyError and this file would be testing its own scaffolding.
    record: dict = fastpath._base_record()  # noqa: SLF001
    seen: dict = {}

    def _tail(**kwargs):
        seen.update(kwargs)
        return "DISPATCHED"

    import types  # noqa: PLC0415

    xp = types.SimpleNamespace(__name__="numpy")
    try:
        result = metal_dispatch.decide(record, fields, pml, grid, sources, xp,
                                       finish=finish or _tail)
    except TypeError as exc:
        # RUNG 8bM'S ARITY, AND NOTHING ELSE. ``_subnormal_gate`` exists today with
        # two parameters and the ladder calls it with three, so a configuration that
        # survives every structural rung dies here — which is exactly the set of
        # tests that must wait for the dispatch batch, and none of the refusal tests
        # above, which return before this rung. Narrowed to that one call so a real
        # TypeError anywhere else in the ladder still fails loudly.
        if "_subnormal_gate" not in str(exc) or _subnormal_gate_takes_a_table():
            raise
        requires_resource_skip(
            FASTPATH_BATCH,
            "rung 8bM calls fastpath._subnormal_gate(record, xp, table) and the "
            "shipped symbol takes two arguments; this configuration reaches that "
            "rung, so it cannot be judged until the dispatch batch lands")
    return record, result, seen


def test_an_unadmitted_fused_label_refuses_the_whole_plan_by_name(monkeypatch):
    """Clause 8M, driven positively even though no composer can reach it.

    The composer is handed the admitted label set, so an un-admitted product is
    never installed and the clause is a BACKSTOP with no live path. "No composer can
    reach it today" is a fact about the composer, not about the clause, and a clause
    nothing exercises is a clause nobody knows still works — so the offer is
    monkeypatched away and the refusal is required to name the label, the gate and
    the switch.
    """
    real_plan_step = launch.plan_step

    def offer_everything(fields, pml, **kwargs):
        kwargs["fuse_labels"] = None      # the pre-offer composer: install anything
        kwargs["fuse"] = True
        return real_plan_step(fields, pml, **kwargs)

    monkeypatch.setattr(launch, "plan_step", offer_everything)
    monkeypatch.setattr(metal_dispatch, "METAL_RELEASED_FUSED_ARMS", {})
    monkeypatch.setattr(metal_dispatch, "METAL_FUSED_RELEASE_ARM_AXES", {})
    record, result, _seen = _drive(matrix.cart())
    assert result is None
    reason = record.get("refused_because") or ""
    assert "a fused cross-sub-step product won" in reason, reason
    assert "has not been driven through the driver seam" in reason, reason
    assert metal_dispatch.METAL_DRIVER_ROUTE_GATE in reason, reason
    assert fastpath.FUSE_ARMS_SWITCH in reason, reason


def test_a_pending_family_is_refused_through_its_constituents(monkeypatch):
    """The see-through, which is the rung a wiring edit could otherwise walk past.

    A fused product writes one label into both slots it absorbs, so the arms it
    implements vanish from ``selected``. A rung reading the label alone would pass
    over a pending constituent — which is how a wiring edit could quietly lift a
    standing refusal by covering the arms it names.
    """
    constituents = metal_dispatch.fused_arm_constituents()
    arm = "fused magnetic B/H pair"
    assert constituents.get(arm), constituents
    monkeypatch.setattr(metal_dispatch, "METAL_PENDING_DEVICE_GATE_ARMS",
                        {constituents[arm][0]: "a constituent this test made pending"})
    record, result, _seen = _drive(matrix.cart())
    assert result is None
    assert "a constituent this test made pending" in (record.get("refused_because") or "")


def test_a_fused_label_with_no_absorb_declaration_refuses_rather_than_dispatching(
        monkeypatch):
    """A product no rung below can judge is exactly the product that must not run."""
    monkeypatch.setattr(metal_dispatch, "fused_arm_constituents", dict)
    record, result, _seen = _drive(matrix.cart())
    assert result is None
    reason = record.get("refused_because") or ""
    assert "cannot name the arms it substitutes" in reason, reason


def test_a_fold_whose_fills_carry_no_kernel_refuses_the_whole_plan(monkeypatch):
    """Rung 6bM, and the wording carries the phrases the route gate's classifier greps for."""
    pair = matrix.folded()
    real_plan_step = launch.plan_step

    def without_fills(fields, pml, **kwargs):
        plan = real_plan_step(fields, pml, **kwargs)
        for slot in ("fill_B", "fill_D"):
            plan.plans.pop(slot, None)
        return plan

    monkeypatch.setattr(launch, "plan_step", without_fills)
    record, result, _seen = _drive(pair)
    assert result is None
    reason = record.get("refused_because") or ""
    assert "mirror-folded" in reason and "fill sub-steps" in reason, reason


def test_the_null_drop_reads_the_plan_rather_than_matching_a_label():
    """``performs_device_work`` is the composer's own declaration.

    Matching the arm's LABEL would make a rename a silent correctness change, and it
    is read THROUGH the wrapper because after a pair is installed two slots hold
    objects that declare nothing of their own.
    """
    if not _subnormal_gate_takes_a_table():
        requires_resource_skip(
            FASTPATH_BATCH,
            "fastpath._subnormal_gate has 2 parameters and rung 8bM passes 3; the "
            "ladder cannot reach the null-drop record until the dispatch batch lands")
    record, _result, _seen = _drive(matrix.real_stored_e_no_pml())
    assert "dropped_null" in json.dumps(record) or record.get("slots"), record


def test_the_record_names_the_table_and_the_route_gate():
    record, _result, _seen = _drive(matrix.cart())
    assert record["table"] == "metal"
    assert (record.get("fusion") or {}).get("driver_route_gate") == \
        metal_dispatch.METAL_DRIVER_ROUTE_GATE


def test_the_record_carries_the_residency_the_composition_installed(monkeypatch):
    """The shipped bracket, asked for BY NAME: ``held`` is the default since 2026-09-27."""
    monkeypatch.setenv(metal_dispatch.RESIDENCY_ENV, metal_dispatch.RESIDENCY_SHIPPED)
    record, result, seen = _drive(matrix.cart())
    assert result == "DISPATCHED", record.get("refused_because")
    residency = record["residency"]
    assert residency["mode"] == metal_dispatch.RESIDENCY_SHIPPED
    assert residency["mirrors"] > 0
    assert residency["invariant"] == "host-authoritative between launches"
    assert residency["held"] == [] and residency["admission"] is None
    assert seen["table"] == "metal"
    assert seen["residency"] is not None


def test_the_default_residency_the_composition_installs_is_held(monkeypatch):
    """With the variable unset the ladder admits and arms a hold, and says so."""
    monkeypatch.delenv(metal_dispatch.RESIDENCY_ENV, raising=False)
    pair = matrix.cart()
    record, result, seen = _drive(pair)
    try:
        assert result == "DISPATCHED", record.get("refused_because")
        residency = record["residency"]
        assert residency["mode"] == metal_dispatch.RESIDENCY_HELD
        assert residency["held"], residency
        assert residency["invariant"].startswith("device-authoritative"), residency
        assert residency["admission"]["refused"] is False, residency["admission"]
        assert residency["released_prior_hold"] is None
    finally:
        # The probe tail builds no plan, so nothing else would ever release this hold.
        metal_dispatch.release_residency_hold(seen["residency"], pair[0])


def test_the_subnormal_block_records_this_tables_policy_and_executors():
    record, _result, _seen = _drive(matrix.cart())
    assert record["subnormal"]["certification_policy"] == "flush"
    assert record["subnormal"]["governed_executors"] == ["host", "mps"]


def test_rung_8bM_refuses_an_installed_keep_by_name():
    """The one-policy rule, at the rung that enforces it for this table.

    MPS flushes float32 denormals natively and exposes no lever, so ``keep`` is
    unattainable on the device half; a process that installed it must be refused
    rather than dispatched under a split.
    """
    if not _subnormal_gate_takes_a_table():
        requires_resource_skip(
            FASTPATH_BATCH,
            "fastpath._subnormal_gate(record, xp, table) has not landed; rung 8bM "
            "calls it with three arguments")
    record: dict = fastpath._base_record()  # noqa: SLF001
    import types  # noqa: PLC0415

    from meep_gpu import subnormal_policy  # noqa: PLC0415

    refusal = fastpath._subnormal_gate(  # noqa: SLF001
        record, types.SimpleNamespace(__name__="numpy"), metal_dispatch.TABLE)
    host = subnormal_policy.executor_report("host")
    if subnormal_policy.get_subnormal_policy() == subnormal_policy.KEEP:
        assert refusal is not None
        assert "subnormal policy" in refusal, refusal
    elif host.get("requested") == subnormal_policy.FLUSH and not host.get("attained"):
        # The third outcome: flush was requested and this host could not install it.
        # A process that has not imported MEEP has one other lever on its FTZ/DAZ
        # bits, Darwin's _FE_DFL_DISABLE_DENORMS_ENV, so on Linux the policy is
        # unattainable and the rung must refuse by name rather than dispatch under a
        # policy the host is not running.
        assert refusal is not None
        assert "UNATTAINABLE" in refusal, refusal
    else:
        # The process installed flush (this suite's own precondition), so the rung
        # must ADMIT rather than refuse: the same rung, the other direction, which
        # is what makes the refusal above a decision instead of a constant.
        assert refusal is None, refusal


def test_the_table_policy_map_and_this_module_agree():
    """The two must not drift: one names the value, the other reads it."""
    _requires("TABLE_SUBNORMAL_POLICY")
    assert fastpath.TABLE_SUBNORMAL_POLICY["metal"] == metal_dispatch.SUBNORMAL_POLICY
    _requires("TABLE_GOVERNED_EXECUTORS")
    assert (tuple(fastpath.TABLE_GOVERNED_EXECUTORS["metal"])
            == metal_dispatch.GOVERNED_EXECUTORS)


def test_a_numpy_engine_with_an_mps_device_takes_the_metal_table():
    _requires("candidate_tables")
    import types  # noqa: PLC0415

    import numpy  # noqa: PLC0415

    grid = types.SimpleNamespace(xp=numpy)
    tables = fastpath.candidate_tables(grid)
    if not fastpath.metal_hardware_present():
        requires_resource_skip(
            "mps_device",
            "no MPS device on this host; the candidate set is empty by hardware and "
            "the refusal that covers it is test_dispatch_contract's")
    assert tables == (fastpath.METAL_TABLE,), tables


# ---------------------------------------------------------------------------
# 5. The package boundary this table must not cross
# ---------------------------------------------------------------------------

def _module_level_imports(path: pathlib.Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots = set()
    for node in tree.body:          # Module level only: function-local stays legal.
        if isinstance(node, ast.Import):
            roots |= {alias.name.partition(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            base = (node.module or "").partition(".")[0]
            if base:
                roots.add(base)
            if node.level:
                roots |= {alias.name.partition(".")[0] for alias in node.names}
    return roots


def test_metal_dispatch_imports_nothing_metal_and_nothing_torch_at_module_level():
    """Importing this module on a host with neither must cost nothing.

    It carries the Metal table and is imported by ``fastpath`` at the backend rung,
    so a module-level ``torch`` or ``metal_kernels`` here would be pulled into every
    process that touches the fast path — including the NumPy array path, which has
    no business compiling shaders. ``test_package_boundary`` measures the same rule
    for the package; this measures it for the one file most likely to break it.
    """
    roots = _module_level_imports(PACKAGE_DIR / "metal_dispatch.py")
    assert "torch" not in roots, roots
    assert "metal_kernels" not in roots, roots


def test_fastpath_reaches_torch_and_metal_kernels_only_function_locally():
    """The same rule, on the file the rung-3 branch lands in.

    Written now rather than with that batch on purpose: it passes today (that file
    names neither) and it is what stops the batch from landing a module-level import
    while every other test still goes green.
    """
    roots = _module_level_imports(PACKAGE_DIR / "fastpath.py")
    assert "torch" not in roots, roots
    assert "metal_kernels" not in roots, roots
