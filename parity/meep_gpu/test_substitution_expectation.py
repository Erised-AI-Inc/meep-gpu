"""What the route gate expects a fused unit to SAVE, and why the shape is declared.

THE QUESTION THIS FILE EXISTS FOR: ``expected_substitution`` decides every dispatch
case's ``EXACT`` verdict -- a measured drop that misses it reads
``DROPPED-BUT-NOT-EXACT`` and the case fails -- and until 2026-09-14 it had NO test
of its own. The only three references to it in the tree were inside the gate that
defines it, so its arithmetic was pinned by nothing but the device runs that
happened to agree with it, and a shape it did not model was indistinguishable from a
weld that had stopped fusing.

That is exactly how ``complex_no_pml_3d`` failed. The function modelled one triple
shape -- a leading group welding ``step_D`` and ``update_E``, a trailing group still
advancing ``update_P`` once per component -- and the complex no-absorber weld is the
other: COMPONENT-MAJOR, welding all three sub-steps per component and launching
nothing in its trailing group. Measured on the GPU host 2026-09-14 over 600 dispatches,
it dropped 2.0 launches a step where this function expected 1, with
``hook_drop_per_step`` agreeing at 2.0 and the slot sets identical -- a correct
result failing on the expectation rather than on the run.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import gate_dispatch_fused_route as route_gate  # noqa: E402


def test_a_two_slot_pair_occupies_two_slots_and_saves_one_launch():
    assert route_gate.expected_substitution({"pairs": 1, "poles": 1}) == {
        "slots": 2, "drop_per_step": 1}


def test_a_separate_tail_triple_occupies_three_slots_and_still_saves_one():
    """The shape the function was written on: only the leading weld removes a
    launch, because the trailing group replaces the array ADE consult one for one."""
    assert route_gate.expected_substitution(
        {"pairs": 1, "poles": 1, "triple": True}) == {
            "slots": 3, "drop_per_step": 1}


def test_a_component_major_triple_saves_one_more_because_its_tail_launches_nothing():
    """3 launches replace 3P + 2, not 4, so the drop is 3P - 1 rather than 3P - 2."""
    assert route_gate.expected_substitution(
        {"pairs": 1, "poles": 1, "triple": True, "tail_in_leading": True}) == {
            "slots": 3, "drop_per_step": 2}


def test_the_declared_term_moves_only_the_rows_that_declare_it():
    """AN ADDITIVE TERM, AND THE POINT IS THAT EVERY OTHER ROW IS UNTOUCHED.

    The drive tables are shared across backends and the formula has no per-row
    override path, so a term that changed an undeclared row's expectation would
    silently re-grade cases nobody was looking at.
    """
    for name, table in (("DRIVE", route_gate.DRIVE),
                        ("DRIVE_CUDA", route_gate.DRIVE_CUDA)):
        for case, spec in table.items():
            if spec.get("tail_in_leading"):
                continue
            pairs = int(spec["pairs"])
            poles = int(spec.get("poles", 1))
            assert (route_gate.expected_substitution(spec)["drop_per_step"]
                    == pairs + 3 * (poles - 1)), f"{name}[{case}]"


def test_tail_in_leading_is_meaningless_without_a_triple():
    """It names the shape of a THREE-SLOT weld's trailing group; a pair has none,
    and a row that sets it on a pair must not quietly gain a launch."""
    assert route_gate.expected_substitution(
        {"pairs": 1, "poles": 1, "tail_in_leading": True})["drop_per_step"] == 1


def test_the_complex_no_pml_row_expects_what_the_device_measured():
    """THE ROW THIS TERM WAS ADDED FOR, pinned against the run rather than the
    reasoning: the GPU host 2026-09-14, unfused 6.0 launches a step against fused 4.0."""
    spec = route_gate.DRIVE_CUDA["complex_no_pml_3d"]
    assert spec.get("triple") and spec.get("tail_in_leading")
    assert route_gate.expected_substitution(spec) == {
        "slots": 3, "drop_per_step": 2}


def test_poles_still_scale_the_drop_on_both_triple_shapes():
    """The ADE runs 3P launches a step unfused and 3 fused, whichever shape the
    weld has, so P moves both by the same 3(P - 1)."""
    for extra, base in ((({}), 1), ({"tail_in_leading": True}, 2)):
        for poles in (1, 2, 5):
            spec = {"pairs": 1, "poles": poles, "triple": True, **extra}
            assert (route_gate.expected_substitution(spec)["drop_per_step"]
                    == base + 3 * (poles - 1)), (poles, extra)


def test_the_pending_rung_classifier_names_every_pending_sub_step_arm():
    """THE RUNG NO ROUTE CASE REACHES ANY MORE, kept from going stale on the host.

    ``FOLD_REFUSAL_RUNGS``'s ``pending-device-gate`` needle was read off the text the
    ladder returned for ``folded_complex_offdiag_2d`` on 2026-09-15. The 2026-09-15
    ruling certified that case's update_E arm, so from that batch no Triton DRIVE
    case refuses on this rung and nothing on a device would notice a pending reason
    reworded past the needle -- a reason the classifier no longer recognises reads
    REFUSAL-CHANGED on the first case that meets it, which is the failure this table
    has already paid for once. So the needle is held to the map's own text: every
    sub-step arm still pending must classify to this rung, and to no earlier one.
    """
    from meep_gpu import fastpath  # noqa: PLC0415

    pending = {arm: reason for arm, reason in fastpath.PENDING_DEVICE_GATE_ARMS.items()
               if not fastpath.arm_is_fused(arm)}
    assert pending, "no sub-step arm is pending; retire the rung rather than this test"
    for arm, reason in pending.items():
        assert route_gate.classify_refusal(reason) == "pending-device-gate", (
            arm, reason, route_gate.classify_refusal(reason))


# ---------------------------------------------------------------------------
# The cuda_alone leg's Triton-less baseline (tritonless_baseline_only)
# ---------------------------------------------------------------------------

def _not_comparable(absorbed, expected, drop, *, lost=(), driven=None,
                    agree=True):
    """A proof as :func:`substitution_proof` leaves it in the declared-slot branch."""
    return {"verdict": "NOT-COMPARABLE", "why": "declared slots differ",
            "absorbed_array_slots": list(absorbed), "lost_slots": list(lost),
            "driven_slots": sorted(driven if driven is not None else absorbed),
            "declared_baseline_array_slots": [],
            "expected_drop_declared": expected, "expected_drop_per_step": expected,
            "launch_drop_per_step": drop, "counts_agree": agree}


def _withheld(monkeypatch):
    monkeypatch.setattr(route_gate, "TRITON_WITHHELD", {"mechanism": "test"})


def test_a_wholly_absorbed_triple_costs_its_own_four_launches_whatever_the_poles():
    """THE ARITHMETIC IS THE ABSORBED-SLOT BRANCH'S, with update_P's real cost.

    Against the drive row's baseline (one launch per curl or constitutive slot, 3P
    for update_P) a separate-tail triple the Triton-less veto leg left WHOLLY on the
    array path lowers the counted drop by 3P + 2, so the row's own 3(P - 1) cancels
    and the counted drop is the other units' saving minus the weld's four launches.
    ``expected_drop - len(absorbed)`` would read -1 on dispersive_2d where the model
    and the composition both say -3.
    """
    for case in ("dispersive_2d", "dispersive5_2d", "dispersive6_2d",
                 "folded_dispersive_2d", "absorber_1d"):
        spec = route_gate.DRIVE_CUDA[case]
        expected = route_gate.expected_substitution(spec)["drop_per_step"]
        absorbed = ("step_D", "update_E", "update_P")
        counted = expected - sum(route_gate.baseline_launches_per_step(slot, spec)
                                 for slot in absorbed)
        assert counted == (int(spec["pairs"]) - 1) - 4, (case, counted)


def test_a_triton_less_proof_that_saved_the_counted_launches_is_rewritten(monkeypatch):
    _withheld(monkeypatch)
    spec = route_gate.DRIVE_CUDA["dispersive_2d"]
    proof = _not_comparable(["step_D", "update_E", "update_P"], 2, -3.0,
                            driven=["step_B", "step_D", "update_E", "update_H",
                                    "update_P"])
    assert route_gate.tritonless_baseline_only(proof, spec)
    assert proof["verdict"] == route_gate.TRITONLESS_SUBSTITUTION_VERDICT
    assert proof["verdict"] not in route_gate.EXACT_SUBSTITUTION_VERDICTS
    assert proof["strict_verdict"] == "NOT-COMPARABLE"
    assert proof["expected_drop_per_step"] == -3


def test_a_pair_row_keeps_one_launch_per_absorbed_slot(monkeypatch):
    """conductive_2d's shape: the conductive D/E pair over a veto leg that left its
    two slots on the array path, so 2 - 2 = 0 launches are counted."""
    _withheld(monkeypatch)
    spec = route_gate.DRIVE_CUDA["conductive_2d"]
    proof = _not_comparable(["step_D", "update_E"], 2, 0.0,
                            driven=["step_B", "step_D", "update_E", "update_H"])
    assert route_gate.tritonless_baseline_only(proof, spec)
    assert proof["expected_drop_per_step"] == 0


def test_a_triton_less_proof_with_the_wrong_drop_still_fails(monkeypatch):
    """The waiver is ``absorbed == declared`` and nothing else: the literal
    one-per-slot count (-1 here) is not what the fused leg must save."""
    _withheld(monkeypatch)
    spec = route_gate.DRIVE_CUDA["dispersive_2d"]
    proof = _not_comparable(["step_D", "update_E", "update_P"], 2, -1.0,
                            driven=["step_B", "step_D", "update_E", "update_H",
                                    "update_P"])
    assert not route_gate.tritonless_baseline_only(proof, spec)
    assert proof["verdict"] == "NOT-COMPARABLE"
    assert proof["expected_drop_tritonless"] == -3
    assert "counted drop is -3" in proof["why_on_this_leg"]


def test_the_waiver_needs_agreeing_witnesses_no_lost_slot_and_a_driving_unit(
        monkeypatch):
    _withheld(monkeypatch)
    spec = route_gate.DRIVE_CUDA["conductive_2d"]
    absorbed = ["step_D", "update_E"]
    driven = ["step_B", "step_D", "update_E", "update_H"]
    for proof in (_not_comparable(absorbed, 2, 0.0, driven=driven, agree=False),
                  _not_comparable(absorbed, 2, 0.0, driven=driven, lost=["update_P"]),
                  _not_comparable(absorbed, 2, 0.0, driven=["step_B", "update_H"]),
                  _not_comparable([], 2, 2.0, driven=driven)):
        assert not route_gate.tritonless_baseline_only(proof, spec)
        assert proof["verdict"] == "NOT-COMPARABLE"


def test_the_waiver_is_the_triton_less_legs_only(monkeypatch):
    monkeypatch.setattr(route_gate, "TRITON_WITHHELD", None)
    proof = _not_comparable(["step_D", "update_E"], 2, 0.0,
                            driven=["step_B", "step_D", "update_E", "update_H"])
    assert not route_gate.tritonless_baseline_only(
        proof, route_gate.DRIVE_CUDA["conductive_2d"])
    assert proof["verdict"] == "NOT-COMPARABLE"
