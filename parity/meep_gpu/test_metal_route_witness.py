"""The Metal route gate's launch witness and its withheld-consult ladder, host-only.

Two defects found on 2026-09-18 in ``dispatch_metal_route_2026-09-17_envelope``, both
in the INSTRUMENT rather than the product:

1. ``MetalLaunchCounter.attach`` wrapped only an owner's ``_functions`` table and
   counted the owner as attached regardless. The real-cylindrical plans keep their
   compiled kernels under ``_prefix_functions``/``_fused_functions`` and friends, so
   the ``cylindrical`` fused leg read 6396 launches against 0 compiled calls and the
   case read VACUOUS-PASS on a product that ran.
2. ``withheld_control`` had no NOT-APPLICABLE rung for an inactive layer, where the
   array constitutive is a pure overwrite and the comparison is array against array
   whatever the kernel did.

And one in the EXPECTATION, found on 2026-09-19 in
``dispatch_metal_route_2026-09-19_witness``: ``substitution_proof`` expected exactly
one launch fewer per pair, and ``complex_no_pml_3d``'s pair folds a stored-E single
that launches once per component into its one launch (8.0 -> 5.0 a step), so a
correct result read DROPPED-BUT-NOT-EXACT. The expectation is now declared per drive
row and cross-checked against the unfused leg's own counter.

These tests pin the repaired instrument on fakes; what a kernel COMPUTES is the
device campaign's business.
"""

from __future__ import annotations

import pathlib
import sys
import types

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import gate_dispatch_metal_route as route  # noqa: E402


def _kernel() -> object:
    """A stand-in leaf whose TYPE NAME is the compiled Metal kernel's."""
    kernel_type = type(route._COMPILED_LEAF_TYPE, (), {"__call__": lambda self, *a: None})  # noqa: SLF001
    return kernel_type()


class _SlottedPair:
    """Shaped like the real-cylindrical pair plans: slotted, NOT a KernelPlan."""

    __slots__ = ("_prefix_functions", "_fused_functions", "launches")

    def __init__(self) -> None:
        self._prefix_functions = {"B": _kernel()}
        self._fused_functions = {"B": _kernel()}
        self.launches = 0

    def run(self) -> None:
        self.launches += 2
        self._prefix_functions["B"]()
        self._fused_functions["B"]()


class _KernelPlanLike:
    def __init__(self) -> None:
        self._functions = {"E": {"x": _kernel()}}   # the nested symmetry shape
        self.launches = 0


class _Blind:
    """Books launches, holds no table the witness can reach."""

    def __init__(self) -> None:
        self.launches = 0
        self._kernel = _kernel()


class _Null:
    performs_device_work = False

    def __init__(self) -> None:
        self.launches = 0


def _holder(**plans):
    return types.SimpleNamespace(plans=plans)


def test_every_function_table_on_a_slotted_owner_is_found_by_its_suffix():
    assert route._function_table_names(_SlottedPair()) == [  # noqa: SLF001
        "_prefix_functions", "_fused_functions"]
    assert route._function_table_names(_KernelPlanLike()) == ["_functions"]  # noqa: SLF001
    assert route._function_table_names(_Blind()) == []  # noqa: SLF001


def test_both_tables_of_a_cylindrical_shaped_owner_are_wrapped_and_counted():
    counter = route.MetalLaunchCounter()
    pair = _SlottedPair()
    seen = counter.attach("fused", _holder(step_B=pair, update_H=types.SimpleNamespace(
        absorbed_by=pair)))
    assert seen["wrapped"] == 1
    assert seen["unwitnessed_owners"] == []
    assert seen["non_kernel_leaves"] == []
    before = counter.snapshot()
    pair.run()
    delta = route.MetalLaunchCounter.delta(before, counter.snapshot())
    assert delta == {"total": 2, "function_calls": 2}


def test_an_owner_the_witness_cannot_see_is_reported_not_counted():
    counter = route.MetalLaunchCounter()
    seen = counter.attach("fused", _holder(step_B=_Blind(), update_H=_Null()))
    assert seen["wrapped"] == 0, "an owner with nothing wrapped is not 'attached'"
    assert seen["unwitnessed_owners"] == ["_Blind"], (
        "a null arm declares performs_device_work False and has nothing to witness")


def test_a_leaf_that_is_not_a_compiled_kernel_is_named_rather_than_counted():
    owner = _KernelPlanLike()
    owner._functions["E"]["host"] = lambda: None
    seen = route.MetalLaunchCounter().attach("fused", _holder(update_E=owner))
    assert seen["non_kernel_leaves"] == ["_KernelPlanLike._functions: function"]


def _leg(chunks, **extra):
    leg = {"active_step_path": "fused", "chunks": chunks,
           "metal_launches": {"total": sum(c["launches"] for c in chunks),
                              "function_calls": sum(c["function_calls"] for c in chunks)},
           "report": {"table": "metal",
                      "fusion": {"driven": {"step_B": "pair", "update_H": "pair"}},
                      "launch_counters": {"step_B": {"dispatches": 10, "plan_launches": 20},
                                          "update_H": {"dispatches": 10}}}}
    leg.update(extra)
    return leg


def _chunk(steps, launches, calls):
    return {"steps": steps, "launches": launches, "function_calls": calls,
            "launches_per_step": launches / steps, "function_calls_per_step": calls / steps}


SPEC = {"pairs": 1}


def _failures(leg):
    return route.fused_leg_is_real(leg, SPEC)["failures"]


def test_agreeing_steady_state_counts_pass_the_new_clauses():
    assert not [f for f in _failures(_leg([_chunk(1, 4, 0), _chunk(10, 20, 20)]))
                if "steady" in f or "HARNESS" in f]


@pytest.mark.parametrize("chunks, needle", [
    ([_chunk(1, 4, 4), _chunk(10, 20, 0)], "in steady state"),     # the cylindrical shape
    ([_chunk(1, 4, 4), _chunk(10, 20, 10)], "disagree"),
    ([_chunk(10, 20, 20)], "single chunk"),                          # no KeyError
])
def test_a_blind_or_disagreeing_witness_is_a_failure(chunks, needle):
    assert any(needle in f for f in _failures(_leg(chunks))), _failures(_leg(chunks))


def test_unwitnessed_owners_on_the_leg_are_a_harness_failure():
    leg = _leg([_chunk(1, 4, 4), _chunk(10, 20, 20)], unwitnessed_owners=["_Blind"])
    assert any(f.startswith("HARNESS: plan owner(s)") for f in _failures(leg))


def test_counts_agree_needs_the_levels_as_well_as_the_drops():
    """2.0 == 2.0 on the drops read EXACT while the fused level stood at 0 calls."""
    fused = _leg([_chunk(1, 6, 0), _chunk(10, 40, 0)])
    unfused = _leg([_chunk(1, 6, 2), _chunk(10, 60, 20)])
    unfused["report"]["fusion"] = {"vetoed": True, "driven": {}}
    out = route.substitution_proof(fused, unfused, {"pairs": 1})
    assert out["launch_drop_per_step"] == out["function_drop_per_step"] == 2.0
    assert out["levels_agree"] is False
    assert out["counts_agree"] is False
    assert out["verdict"] != "EXACT"


# --------------------------------------------------------------------------
# The expected drop: pairs + sum(N - 1) over DECLARED collapsed singles
# --------------------------------------------------------------------------
#
# The shapes below are the ones measured on dispatch_metal_route_2026-09-19_witness,
# per-slot launches a step over 600 steps. COLLAPSE is complex_no_pml_3d: the
# unfused `complex no-PML stored E` single launches once per component on update_E
# and the pair folds all three into its one launch, 8 -> 5. PER-COMPONENT is the
# dispersive rows: the unfused update_E single launches 3 a step and so does the
# pair on step_D, so the pair still saves exactly one.

_SEAM = ("step_B", "step_D", "update_E", "update_P")
_PAIR = "fused complex conductive no-PML curl -> complex stored E"
_DRIVEN = {"step_D": _PAIR, "update_E": _PAIR}
_KEY = route.COLLAPSED_SINGLES_KEY

UNFUSED = {"step_B": 1, "step_D": 1, "update_E": 3, "update_P": 3}        # 8 a step
COLLAPSE = {"step_B": 1, "step_D": 1, "update_E": None, "update_P": 3}    # 5 a step
PER_COMPONENT = {"step_B": 1, "step_D": 3, "update_E": None, "update_P": 3}  # 7


def _sub_leg(per_slot, *, driven, vetoed=None, steps=600):
    """A leg whose chunk totals and per-slot counters both follow ``per_slot``."""
    per_step = sum(n or 0 for n in per_slot.values())
    tail = steps - 1
    return {"active_step_path": "fused",
            # The first chunk carries no compiled calls (the witness attaches after
            # it), exactly as a real leg's does.
            "chunks": [_chunk(1, per_step, 0), _chunk(tail, per_step * tail,
                                                      per_step * tail)],
            "report": {"table": "metal",
                       "fusion": {"driven": dict(driven), "vetoed": vetoed},
                       "slots": {s: {"state": "dispatched"} for s in _SEAM},
                       "launch_counters": {
                           s: {"dispatches": steps,
                               "plan_launches": None if n is None else n * steps}
                           for s, n in per_slot.items()}}}


def _proof(fused_slots, unfused_slots, spec, *, driven=_DRIVEN):
    fused = _sub_leg(fused_slots, driven=driven)
    unfused = _sub_leg(unfused_slots, driven={}, vetoed=True)
    return route.substitution_proof(fused, unfused, spec)


def test_a_declared_collapse_whose_counter_matches_is_exact():
    """The 2026-09-19 measurement: 8 -> 5 is a drop of 3 = 1 pair + (3 - 1)."""
    spec = {"pairs": 1, _KEY: {"update_E": 3}}
    out = _proof(COLLAPSE, UNFUSED, spec)
    assert out["launch_drop_per_step"] == out["function_drop_per_step"] == 3.0
    assert out["expected_drop_per_step"] == 3
    assert out[_KEY] == {"update_E": 3}, "the declaration is recorded beside its number"
    assert out["declaration_check"]["update_E"]["unfused_per_step"] == 3.0
    assert out["verdict"] == "EXACT"


@pytest.mark.parametrize("unfused_update_e, needle", [
    (1, "counter reads 600 over 600 steps"),      # a per-step single, not per-component
    (2, "counter reads 1200 over 600 steps"),
    (None, "counter reads None"),                  # the slot's plan booked nothing
])
def test_a_declaration_the_unfused_counter_contradicts_is_refused(unfused_update_e,
                                                                  needle):
    """The CHUNK totals still read 8 -> 5 (drop 3 == the declared 3); only the unfused
    leg's own per-slot counter disagrees, and that alone must refuse."""
    unfused = _sub_leg(UNFUSED, driven={}, vetoed=True)
    unfused["report"]["launch_counters"]["update_E"]["plan_launches"] = (
        None if unfused_update_e is None else unfused_update_e * 600)
    fused = _sub_leg(COLLAPSE, driven=_DRIVEN)
    out = route.substitution_proof(fused, unfused, {"pairs": 1, _KEY: {"update_E": 3}})
    assert out["launch_drop_per_step"] == 3.0
    assert out["verdict"] == "DECLARATION-MISMATCH"
    assert "update_E is declared at 3 launches a step" in out["why"]
    assert needle in out["why"], out["why"]


def test_a_declared_slot_no_pair_replaced_is_refused_even_when_the_sum_works():
    """Declaring update_P (3 a step unfused, counter agrees) instead of update_E gives
    the same expected 3 as the right declaration, so the drop alone would read EXACT
    on a declaration about a slot the pair never touched."""
    out = _proof(COLLAPSE, UNFUSED, {"pairs": 1, _KEY: {"update_P": 3}})
    assert out["expected_drop_per_step"] == 3 == out["launch_drop_per_step"]
    assert out["verdict"] == "DECLARATION-MISMATCH"
    assert "update_P is declared collapsed" in out["why"]


@pytest.mark.parametrize("fused_slots, drop, why", [
    ({"step_B": 1, "step_D": 2, "update_E": None, "update_P": 3}, 2.0,
     "the pair launched twice a step: it collapsed two of the three, not all"),
    (PER_COMPONENT, 1.0,
     "a per-component single the pair does NOT collapse, declared as though it did"),
])
def test_a_pair_that_did_not_collapse_is_refused_on_its_own_counter(fused_slots, drop,
                                                                    why):
    """The fused side is READ, not inferred from the total: a declared row's pair must
    book one launch a step. Until the 2026-09-19 review these read
    DROPPED-BUT-NOT-EXACT off the drop alone, which a compensating slot could hide
    (the next test); the fused leg's own counter now refuses them first."""
    out = _proof(fused_slots, UNFUSED, {"pairs": 1, _KEY: {"update_E": 3}})
    assert out["launch_drop_per_step"] == drop, why
    assert out["expected_drop_per_step"] == 3
    assert out["verdict"] == "DECLARATION-MISMATCH", why
    assert "the fused leg's pairs book" in out["why"], out["why"]


def test_a_compensating_slot_cannot_make_a_non_collapse_exact():
    """A pair that did not collapse (3 a step on step_D) beside an unrelated slot that
    lost 2 launches a step (update_P 3 -> 1) sums to the declared drop of 3. The chunk
    totals alone read EXACT; the per-slot counters on both legs refuse it."""
    fused = {"step_B": 1, "step_D": 3, "update_E": None, "update_P": 1}      # 5 a step
    out = _proof(fused, UNFUSED, {"pairs": 1, _KEY: {"update_E": 3}})
    assert out["launch_drop_per_step"] == out["expected_drop_per_step"] == 3
    assert out["verdict"] == "DECLARATION-MISMATCH"
    assert "the fused leg's pairs book" in out["why"]
    assert "slots outside the pairs changed their launches" in out["why"]
    assert out["declaration_check"]["undriven_slots_moved"] == {
        "update_P": {"unfused_per_step": 3.0, "fused_per_step": 1.0}}


@pytest.mark.parametrize("n", [0, 1])
def test_a_collapse_of_fewer_than_two_launches_declares_nothing_and_refuses(n):
    """N = 1 declares nothing and N = 0 is false: a declared 0 lowers the expected drop
    below `pairs`, and a step that saved nothing would read EXACT -- "no data" scoring
    as the best value. The row refuses by name whatever its counters say."""
    unfused = dict(UNFUSED, update_E=n)
    fused = dict(COLLAPSE)
    out = _proof(fused, unfused, {"pairs": 1, _KEY: {"update_E": n}})
    assert out["verdict"] == "DECLARATION-MISMATCH"
    assert f"declared collapsed from {n} launches a step" in out["why"], out["why"]


def test_an_undeclared_case_is_unchanged():
    """No declaration: the expectation is `pairs`, no counter check runs, and neither
    a per-component single nor an undeclared collapse is read as a declaration error.

    The per-component row is EXACT although its update_E single launches 3 a step --
    nine rows of the 2026-09-19 campaign have that shape, which is why "an undeclared
    single launches once" is NOT asserted. The undeclared collapse still refuses, on
    its drop, which is how complex_no_pml_3d was found."""
    exact = _proof(PER_COMPONENT, UNFUSED, {"pairs": 1})
    assert exact["expected_drop_per_step"] == 1 == exact["launch_drop_per_step"]
    assert exact["verdict"] == "EXACT"
    assert "declaration_check" not in exact
    assert exact[_KEY] == {}
    collapse = _proof(COLLAPSE, UNFUSED, {"pairs": 1})
    assert collapse["expected_drop_per_step"] == 1
    assert collapse["verdict"] == "DROPPED-BUT-NOT-EXACT"


def test_a_leg_that_fused_nothing_reads_no_drop_not_a_declaration_error():
    """A policy-refused or DID-NOT-FUSE fused leg has no pair to collapse anything."""
    out = _proof(UNFUSED, UNFUSED, {"pairs": 1, _KEY: {"update_E": 3}}, driven={})
    assert "declaration_check" not in out
    assert out["verdict"] == "NO-DROP"


def test_the_declaration_moves_only_the_row_that_declares_it():
    """One row declares a collapse, and every other row's expectation is `pairs`."""
    declaring = {case for case, spec in route.DRIVE.items() if _KEY in spec}
    assert declaring == {"complex_no_pml_3d"}
    assert route.DRIVE["complex_no_pml_3d"][_KEY] == {"update_E": 3}
    for case, spec in route.DRIVE.items():
        expected = route.expected_drop_per_step(spec)
        assert expected == (3 if case == "complex_no_pml_3d" else spec["pairs"]), case


# --------------------------------------------------------------------------
# The withheld-consult ladder: NOT-APPLICABLE only where every guard holds
# --------------------------------------------------------------------------

def _stub_control(monkeypatch, *, overwrite, withheld_slots, driven, identical=True,
                  skipped_launching=None):
    class _Driver:
        def close(self):
            pass

    def lifted(*_a, **_k):
        leg = {"driver": _Driver(), "label": "withheld",
               "report": {"fusion": {"driven": driven}}}
        return leg, dict(leg, label="reference")

    def withhold():
        state = {"withheld": sum(withheld_slots.values()), "slots": dict(withheld_slots),
                 "skipped_leading": {"step_D": 12},
                 "skipped_launching": dict(skipped_launching or {})}
        return (lambda: None), state

    monkeypatch.setattr(route, "_pair_of_lifted", lifted)
    monkeypatch.setattr(route, "step_leg", lambda *a, **k: None)
    monkeypatch.setattr(route.route, "_withhold_absorbed_consult", withhold)
    monkeypatch.setattr(route.route, "_constitutive_is_an_overwrite", lambda leg: overwrite)
    monkeypatch.setattr(route.route, "_launching_slots_ran",
                        lambda leg, skipped, steps: {"slots": sorted(skipped),
                                                     "deltas": {}, "failures": []})
    monkeypatch.setattr(route.e2e, "collect_state", lambda driver: {})
    calls = []

    def compare(_a, _b):
        # The FIRST comparison is the precondition (identical lifts); the second is
        # the control's own reading.
        calls.append(1)
        same = identical if len(calls) > 1 else True
        return {"identical": same, "arrays_differing": 0 if same else 3,
                "arrays": 29, "differences": []}

    monkeypatch.setattr(route.e2e, "compare_state", compare)
    monkeypatch.setattr(route, "say", lambda *_a, **_k: None)


D_PAIR = {"step_D": "no-PML fused electric D/E pair",
          "update_E": "no-PML fused electric D/E pair"}


def test_an_inactive_layer_with_an_armed_in_seam_control_is_not_applicable(monkeypatch):
    _stub_control(monkeypatch, overwrite=True, withheld_slots={"update_E": 12},
                  driven=D_PAIR)
    row = route.withheld_control("absorber_1d", {"seam": ["D"]}, None, None)
    assert row["verdict"] == "NOT-APPLICABLE"
    assert row["verdict"] in route.CONTROL_PASS
    assert row["why"].index("byte comparison") < row["why"].index("in-seam-pass-mutated-D"), (
        "the main comparison carries the claim; the in-seam row only corroborates")


@pytest.mark.parametrize("overwrite, seam, driven, skipped, why", [
    (False, ["D"], D_PAIR, None, "an ACTIVE layer accumulates: the mechanism exists"),
    (None, ["D"], D_PAIR, None, "an unreadable layer fails closed"),
    (True, [], D_PAIR, None, "no in-seam control is scheduled for the withheld side"),
    (True, ["D"], {}, None, "no fused pair spans the withheld side"),
    (True, ["D"], D_PAIR, {"update_P": 12}, "a device group was left running"),
])
def test_every_missing_guard_keeps_the_blind_row_a_failure(monkeypatch, overwrite, seam,
                                                           driven, skipped, why):
    _stub_control(monkeypatch, overwrite=overwrite, withheld_slots={"update_E": 12},
                  driven=driven, skipped_launching=skipped)
    row = route.withheld_control("absorber_1d", {"seam": seam}, None, None)
    assert row["verdict"] == "NULL-DID-NOT-DIVERGE", why
    assert row["verdict"] not in route.CONTROL_PASS


def test_a_withheld_polarization_slot_is_never_waved_through(monkeypatch):
    _stub_control(monkeypatch, overwrite=True,
                  withheld_slots={"update_E": 12, "update_P": 12}, driven=D_PAIR)
    row = route.withheld_control("no_pml_dispersive_2d", {"seam": ["D"]}, None, None)
    assert row["every_withheld_slot_is_constitutive"] is False
    assert row["verdict"] == "NULL-DID-NOT-DIVERGE"


def test_a_divergence_still_reads_diverged_on_either_branch(monkeypatch):
    _stub_control(monkeypatch, overwrite=True, withheld_slots={"update_E": 12},
                  driven=D_PAIR, identical=False)
    row = route.withheld_control("absorber_1d", {"seam": ["D"]}, None, None)
    assert row["verdict"] == "DIVERGED-AS-REQUIRED"
