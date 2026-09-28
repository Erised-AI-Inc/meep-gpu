"""What the driver-route gate's ARMED NULL may and may not withhold, and when it
is entitled to decline.

``withheld_control`` forces every ABSORBED consult to answer False so the driver's
array sub-step runs ON TOP of the constitutive half the fused launch already
computed. Two things about that were wrong until 2026-09-14, and both are silent
failures of the kind this tree is built to catch — a control that passes while
measuring nothing, and a control that fails while measuring nothing.

THE FIRST IS WHAT IT WITHHELD. A three-slot weld is ONE product with TWO device
groups (``cuda_kernels.fused_pairs.CudaFusedTriplePlan``), and
``_install_fused_triple`` puts the trailing group — a ``_TripleHalfPlan``, which
LAUNCHES — into ``update_P``. Its ``absorbed_by`` names the TRIPLE and it has no
``inner``, so the "is this the leading repair slot" test read False and the slot was
withheld like a sentinel. Withholding a launcher does not let an array call run on
top of a launch; it DELETES the launch. On the 2026-09-14 CUDA shipped leg the two
cases driving ``cuda:no-absorber three-slot dispersive weld`` were the only two of
twenty-one dispatching cases reporting NULL-DID-NOT-DIVERGE, because with group 2
suppressed and an inactive layer the "fused" leg WAS the array leg. The PML triples
passed the same control with their second group equally suppressed — they diverge
off the constitutive accumulation instead — which is a pass on a composition the
control does not describe.

THE SECOND IS WHERE THE MECHANISM EXISTS AT ALL. ``stepping.update_E`` forks on
``_pml_is_active``: the active arm ACCUMULATES, so a second application is a wrong
answer and the control is armed; the inactive arm is a pure OVERWRITE, so a second
application is the identity and the control has no armed direction whatever the
kernel did. Measured off-device on NumPy lifts, 2026-09-14, applying the array
sub-step a second time in its own slot over 12 driver steps::

    case                      pml_is_active   update_E twice   update_P twice
    absorber_1d                     False        0 of 65          13 of 65
    material_dispersion_0d          False        0 of 29           5 of 29
    dispersive_2d                   True        12 of 41           0 of 41
    pml_1d                          True         8 of 32           0 of 32
    dispersive5_2d                  True        22 of 77          22 of 77

Host-only. Nothing here lifts a simulation or touches a device: what is under test
is the gate helper's classification and its verdict chain, which is the half that
was wrong.
"""

from __future__ import annotations

import pathlib
import sys
import types
from typing import Any, Dict, List

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import gate_dispatch_fused_route as route_gate  # noqa: E402

SLOTS = ("step_D", "update_E", "update_P")


def _triple(launched: List[str]) -> Any:
    """A REAL ``CudaFusedTriplePlan``, built off-device.

    Its two launchers append to ``launched`` instead of touching a GPU; everything
    the classification reads — ``absorbed_by``, the absence of ``inner``, the
    ``launches`` counter and its increment in ``run`` — is the shipped object's.
    """
    from meep_gpu.cuda_kernels import fused_pairs as cuda_fused

    return cuda_fused.CudaFusedTriplePlan(
        family="cuda_three_slot_no_pml_dispersive_weld",
        label="no-absorber three-slot dispersive weld", kernel_label="k",
        replaces=SLOTS, slots=SLOTS,
        context=types.SimpleNamespace(fields=object()),
        resolve_launch_args=lambda context: {},
        launch_leading=lambda fields, arguments: launched.append("leading"),
        launch_trailing=lambda fields, arguments: launched.append("trailing"))


def _plans_for_a_triple_with_a_deposit(triple: Any) -> Dict[str, Any]:
    """The mapping ``_install_fused_triple`` writes when the seam carries a deposit.

    ``step_D`` holds the leading group behind the repair bracket, ``update_E`` the
    repair itself, ``update_P`` the trailing device group. The repair plans are the
    shipped classes, constructed with the fields and layer they never reach here
    because nothing in this file calls ``run``.
    """
    from meep_gpu import deposit_repair

    leading = deposit_repair.LeadingRepairPlan(
        triple.leading, object(), None, (), "D")
    return {"step_D": leading,
            "update_E": deposit_repair.TrailingRepairPlan(
                "update_E", leading, object(), None),
            "update_P": triple.trailing}


def _classify(plans: Dict[str, Any], monkeypatch) -> Dict[str, Any]:
    """Run every slot through the patched ``dispatch`` and return what it recorded.

    ``_withhold_absorbed_consult`` captures ``FastPathPlan.dispatch`` as its
    fall-through, so a recorder is installed FIRST and the patch wraps that: the
    slots the control leaves alone reach the recorder, and the ones it withholds
    never do. That is the distinction under test, read from both directions.
    """
    from meep_gpu import fastpath as fp

    reached: List[str] = []
    monkeypatch.setattr(fp.FastPathPlan, "dispatch",
                        lambda self, slot, fields: reached.append(slot) or True,
                        raising=True)
    restore, state = route_gate._withhold_absorbed_consult()  # noqa: SLF001
    try:
        stub = types.SimpleNamespace(
            slots=tuple(plans), step_plan=types.SimpleNamespace(plans=plans))
        answers = {slot: fp.FastPathPlan.dispatch(stub, slot, object())
                   for slot in plans}
    finally:
        restore()
    return {"state": state, "answers": answers, "reached": reached}


def test_a_triples_trailing_half_is_LEFT_RUNNING_because_it_launches(monkeypatch):
    """The repair: ``update_P`` holds a device group, so it is not withheld.

    ``step_D`` is skipped as the leading repair slot (it names its own launcher),
    ``update_E`` is the genuine sentinel and is withheld, and ``update_P`` — the
    trailing half — is skipped as a LAUNCHING slot and recorded separately, so a
    reader can see the second group was left free to run rather than deduce it.
    """
    triple = _triple([])
    out = _classify(_plans_for_a_triple_with_a_deposit(triple), monkeypatch)
    assert out["state"]["skipped_leading"] == {"step_D": 1}
    assert out["state"]["skipped_launching"] == {"update_P": 1}
    assert out["state"]["slots"] == {"update_E": 1}
    assert out["state"]["withheld"] == 1
    # The two the control leaves alone reach the real dispatch; the withheld one
    # answers False without ever getting there, which is the array path.
    assert out["reached"] == ["step_D", "update_P"]
    assert out["answers"]["update_E"] is False


def test_the_OLD_reading_withholds_the_trailing_group_and_deletes_its_launch(
        monkeypatch):
    """THE MUTATION, and it is the defect this repair is for.

    ``_slot_launches`` is put back to the pre-2026-09-14 answer — "no slot
    launches", which is what the ``absorbed_by is inner`` test alone concluded about
    a ``_TripleHalfPlan``. ``update_P`` is then withheld, the trailing group never
    reaches its launcher, and a leg driving this product compares an array run
    against an array run. The control would report NULL-DID-NOT-DIVERGE and a reader
    would go looking for a defect in the kernel.
    """
    triple = _triple([])
    monkeypatch.setattr(route_gate, "_slot_launches", lambda plan: False)
    out = _classify(_plans_for_a_triple_with_a_deposit(triple), monkeypatch)
    assert out["state"]["skipped_launching"] == {}
    assert sorted(out["state"]["slots"]) == ["update_E", "update_P"]
    assert out["answers"]["update_P"] is False
    assert "update_P" not in out["reached"]


def test_a_seamless_triple_keeps_its_LEADING_group_too(monkeypatch):
    """The same misreading reaches GROUP ONE where the seam carries no deposit.

    ``_install_fused_triple`` puts ``triple.leading`` bare into the curl slot when
    there is no in-seam source, and a bare half has ``absorbed_by`` = the triple and
    no ``inner`` exactly as the trailing one does. Under the old reading the control
    would have withheld the curl slot itself.
    """
    from meep_gpu.triton_kernels.launch import NoopPlan

    triple = _triple([])
    plans = {"step_D": triple.leading,
             "update_E": NoopPlan("update_E", triple.leading),
             "update_P": triple.trailing}
    out = _classify(plans, monkeypatch)
    assert sorted(out["state"]["skipped_launching"]) == ["step_D", "update_P"]
    assert out["state"]["slots"] == {"update_E": 1}


def test_the_sentinels_and_the_repair_wrappers_do_not_read_as_launchers():
    """``_slot_launches`` is the structural test, checked against every shape.

    A name-based test would go stale the next time a product is installed a new way;
    the counter cannot, because a plan that launches keeps one and a wrapper that
    does not, does not. Asserted on the shipped classes rather than on stand-ins.
    """
    from meep_gpu import deposit_repair
    from meep_gpu.triton_kernels.launch import NoopPlan

    triple = _triple([])
    leading = deposit_repair.LeadingRepairPlan(
        triple.leading, object(), None, (), "D")
    assert route_gate._slot_launches(triple.leading) is True  # noqa: SLF001
    assert route_gate._slot_launches(triple.trailing) is True  # noqa: SLF001
    assert route_gate._slot_launches(triple) is True  # noqa: SLF001
    assert route_gate._slot_launches(NoopPlan("update_E", triple)) is False  # noqa: SLF001
    assert route_gate._slot_launches(leading) is False  # noqa: SLF001
    assert route_gate._slot_launches(  # noqa: SLF001
        deposit_repair.TrailingRepairPlan("update_E", leading, object(),
                                          None)) is False


def test_no_plan_that_can_SIT_IN_an_absorbed_slot_carries_a_launch_counter():
    """The discriminator, asserted over the TREE rather than over a hand-listed few.

    ``_slot_launches`` skips any absorbed slot whose plan keeps a ``launches``
    counter. That is right exactly while no sentinel or wrapper keeps one — and a
    sentinel that gained one would SILENTLY stop being withheld, turning an armed
    control into NOT-ARMED with no verdict change to notice. ``withheld_control`` is
    shared with ``gate_dispatch_metal_route``, so the blast radius is both dispatch
    legs.

    So the property is checked by enumeration: every class in the plan modules that
    declares or assigns ``absorbed_by`` must NOT carry ``launches``, except
    ``_TripleHalfPlan``, which is the one that launches. Measured 2026-09-14 over
    ``deposit_repair`` (LeadingRepairPlan, TrailingRepairPlan),
    ``triton_kernels.launch`` (NoopPlan), ``metal_kernels.launch`` (SyncedPlan) and
    ``withdraw_hoist`` (LeadingWithdrawPlan): six classes, one launcher.
    """
    import importlib
    import inspect

    modules = ("meep_gpu.withdraw_hoist", "meep_gpu.deposit_repair",
               "meep_gpu.triton_kernels.launch", "meep_gpu.metal_kernels.launch",
               "meep_gpu.cuda_kernels.fused_pairs", "meep_gpu.cuda_kernels.arms",
               "meep_gpu.fastpath")
    found: Dict[str, bool] = {}
    for name in modules:
        module = importlib.import_module(name)
        for class_name, cls in vars(module).items():
            if not inspect.isclass(cls) or cls.__module__ != name:
                continue
            slots = set()
            for base in cls.__mro__:
                slots |= set(getattr(base, "__slots__", ()) or ())
            try:
                source = inspect.getsource(cls)
            except (OSError, TypeError):  # pragma: no cover - source always present
                source = ""
            if not ("absorbed_by" in slots or "self.absorbed_by" in source):
                continue
            found[class_name] = ("launches" in slots
                                 or "self.launches" in source)
    assert found, "no absorbed-slot plan classes found; the enumeration broke"
    launchers = {name for name, carries in found.items() if carries}
    assert launchers == {"_TripleHalfPlan"}, found
    # Non-vacuity: the wrappers this control must keep withholding are all here.
    assert {"NoopPlan", "TrailingRepairPlan", "LeadingRepairPlan",
            "SyncedPlan"} <= set(found), sorted(found)


def _leg_with(plans: Dict[str, Any]) -> Dict[str, Any]:
    return {"driver": types.SimpleNamespace(
        _fast_path=types.SimpleNamespace(
            step_plan=types.SimpleNamespace(plans=plans)))}


def test_a_trailing_group_that_did_not_launch_is_a_failure_not_a_skip():
    """THE POSITIVE HALF, and the mutation that must redden it.

    Leaving a device group running is only honest while the row can show it ran. The
    group's own counter is the witness: consulted twelve times, it must have launched
    at least twelve times. A group whose counter stood still is precisely the thing
    the withheld comparison used to be asked about and could not answer, and the word
    it produces is NOT in the control pass set.
    """
    triple = _triple([])
    leg = _leg_with({"update_P": triple.trailing})
    # Armed: the group launched on every consult.
    for _ in range(12):
        triple.trailing.run()
    ok = route_gate._launching_slots_ran(leg, {"update_P": 12}, 12)  # noqa: SLF001
    assert ok["failures"] == []
    assert ok["deltas"] == {"update_P": 12}

    # THE MUTATION: the same slot, the same consults, a group that never launched.
    silent = _triple([])
    bad = route_gate._launching_slots_ran(  # noqa: SLF001
        _leg_with({"update_P": silent.trailing}), {"update_P": 12}, 12)
    assert len(bad["failures"]) == 1
    assert "second half did not run" in bad["failures"][0]
    assert "LAUNCHING-SLOT-DID-NOT-LAUNCH" not in route_gate.CONTROL_PASS


@pytest.mark.parametrize("plans,needle", [
    ({}, "carries nothing in that slot"),
    ({"update_P": types.SimpleNamespace(absorbed_by=object())},
     "spells no launch counter"),
])
def test_an_unreadable_launching_slot_is_a_finding_not_an_exception(plans, needle):
    """A control that raises loses the twelve steps it already paid for.

    Both unreadable shapes — an empty slot and a plan with no counter — are recorded
    as failures rather than thrown, so the row says what went wrong and the leg still
    carries its other controls.
    """
    out = route_gate._launching_slots_ran(  # noqa: SLF001
        _leg_with(plans), {"update_P": 12}, 12)
    assert len(out["failures"]) == 1
    assert needle in out["failures"][0]


def test_the_overwrite_branch_is_read_from_the_engines_own_fork():
    """``_constitutive_is_an_overwrite`` asks ``stepping._pml_is_active``, and fails closed.

    A layer that cannot be read is ``None`` — neither "plain" nor "accumulating" —
    so the NOT-APPLICABLE branch in :func:`withheld_control` is never reached on
    ignorance. That is the difference between declining a control whose mechanism
    does not exist and silencing one whose answer is inconvenient.
    """
    class _Raises:
        @property
        def pml(self):
            raise RuntimeError("no layer here")

    assert route_gate._constitutive_is_an_overwrite({}) is None  # noqa: SLF001
    assert route_gate._constitutive_is_an_overwrite(  # noqa: SLF001
        {"driver": _Raises()}) is None
    # An absent layer is the plain branch, which is what an mp.Absorber lifts to.
    assert route_gate._constitutive_is_an_overwrite(  # noqa: SLF001
        {"driver": types.SimpleNamespace(pml=None)}) is True


def test_not_applicable_passes_and_the_two_null_words_do_not():
    """The verdict words, against the set that decides the leg's release.

    ``NOT-APPLICABLE`` is a control pass — the mechanism does not exist on that
    branch — while both failing words stay out of it, so a blind instrument and a
    missing launch cannot be confused in ``summary.json``.
    """
    assert "NOT-APPLICABLE" in route_gate.CONTROL_PASS
    assert "NULL-DID-NOT-DIVERGE" not in route_gate.CONTROL_PASS
    assert "LAUNCHING-SLOT-DID-NOT-LAUNCH" not in route_gate.CONTROL_PASS
