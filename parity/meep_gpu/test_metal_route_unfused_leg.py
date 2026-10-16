"""The Metal route gate judges BOTH kernel legs, and its instrument does not repair them.

Three defects in the gate, all found on 2026-10-09/10 behind one real defect in the
product (three complex cylindrical plans read the radial source's STALE host copy under
a held residency, lane MCC):

1. THE UNFUSED LEG WAS RECORDED AND NEVER READ. ``run_case`` compared the unfused leg --
   the certified singles -- with the array path at every checkpoint, kept the answer in
   the row, and decided the verdict off the fused leg alone. The 2026-10-05 probe leg
   passed ``cylindrical_m1`` and ``cylindrical_m0_complex`` PASS-FUSED with the unfused
   leg differing on 24 of 32 arrays at step 1600. ``second_consult_site`` had the same
   blind spot.
2. THE SEAM CENSUS REPAIRED THE FUSED LEG. ``_seam_census`` read the fused driver's state
   before every launch through ``collect_state``, whose ``getattr`` goes through a held
   residency's READ BARRIER and copies every device-owned volume to the host -- exactly
   the stale copies the defective plans then read. On the 01..86 tree the fused leg
   passed with the census and diverged without it (22,684 of 358,400 words at step 24).
3. THE ARMED NULL WAS BLIND ON THE SERIES. Once the kernels' constitutive halves are
   bit-identical to the array path's, the absorbed-consult-withheld control diverges
   only through the own-axis accumulation, and twelve steps from the start do not reach
   it on many cases (NULL-DID-NOT-DIVERGE on 3 of 5 probe-leg cases on 01..79). The
   fused route gate's rule (lane FXR) -- place the window where the mechanism exists,
   NOT-APPLICABLE by name where no withheld component accumulates -- is adopted here.

Host-only: every test below drives the gate's pure verdict functions, or a stand-in
driver, and nothing lifts a simulation or touches a device.
"""

from __future__ import annotations

import copy
import dataclasses
import pathlib
import sys
import types
from typing import Any, Dict, List

import numpy
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import gate_dispatch_metal_route as gate  # noqa: E402

LADDER = (1, 2, 4, 8, 12, 24, 48, 192, 768, 1600)


def _checkpoints(unfused_bad_from=None, fused_bad_from=None) -> List[Dict[str, Any]]:
    out = []
    for steps in LADDER:
        unfused_bad = unfused_bad_from is not None and steps >= unfused_bad_from
        fused_bad = fused_bad_from is not None and steps >= fused_bad_from
        out.append({"steps": steps, "arrays": 32, "words": 358400,
                    "fused_vs_array_identical": not fused_bad,
                    "unfused_vs_array_identical": not unfused_bad,
                    "fused_arrays_differing": 24 if fused_bad else 0,
                    "unfused_arrays_differing": 24 if unfused_bad else 0,
                    "differences": []})
    return out


def _passing_row(**over) -> Dict[str, Any]:
    """A dispatch row the verdict chain passes PASS-FUSED, every reading recorded."""
    row = {
        "case": "cylindrical_m1",
        "checkpoints": _checkpoints(),
        "first_divergent_checkpoint": None,
        "final_state_fused_vs_array": {"identical": True},
        "final_state_unfused_vs_array": {"identical": True},
        "observables": {"identical": True},
        "observables_unfused": {"identical": True},
        "comparator_control": {"armed": True, "comparator_saw_it": True,
                               "words_mismatched": 1},
        "evidence": {"driven": {"step_D": "pair"}, "real": True, "failures": []},
        "array_leg_control": {"clean": True},
        "substitution": {"verdict": "EXACT"},
        "second_consult_site": {"verdict": "PASS"},
    }
    row.update(over)
    return row


def _unfused_diverged(row: Dict[str, Any], start: int = 2) -> Dict[str, Any]:
    """The 2026-10-05 probe leg's cylindrical_m1 shape: unfused differs from step 2."""
    row = copy.deepcopy(row)
    row["checkpoints"] = _checkpoints(unfused_bad_from=start)
    row["final_state_unfused_vs_array"] = {"identical": False}
    return row


def _verdict(row, spec=None, *, report=None, fused_path="fused", unfused_path="fused",
             policy=None) -> str:
    spec = spec or {"expect": "dispatch"}
    return gate.case_verdict(row, spec, fused_report=report or {}, fused_path=fused_path,
                             unfused_path=unfused_path, fused_dispatched_slots=["step_D"],
                             policy_installed=policy)


# --------------------------------------------------------------------------
# 1. The unfused leg is read, on every branch
# --------------------------------------------------------------------------

def test_the_unfused_agreement_reads_every_checkpoint_the_final_state_and_the_observables():
    assert gate.unfused_agreement(_passing_row())["identical"] is True
    found = gate.unfused_agreement(_unfused_diverged(_passing_row()))
    assert found["identical"] is False
    assert found["first_divergent_checkpoint"] == 2
    assert found["checkpoints_diverging"] == 9
    assert "9 of 10 checkpoints differed, first at step 2 (24 of 32 arrays)" in found["why"]
    late = _passing_row(observables_unfused={"identical": False})
    assert gate.unfused_agreement(late)["identical"] is False


@pytest.mark.parametrize("missing", ["checkpoints", "final_state_unfused_vs_array",
                                     "observables_unfused"])
def test_a_reading_the_row_does_not_carry_is_not_agreement(missing):
    row = _passing_row()
    row.pop(missing)
    assert gate.unfused_agreement(row)["identical"] is False


def test_a_dispatch_row_whose_singles_diverge_fails_even_when_the_fused_leg_agrees():
    assert _verdict(_passing_row()) == "PASS-FUSED"
    row = _unfused_diverged(_passing_row())
    assert _verdict(row) == "UNFUSED-DIVERGENCE"
    assert "UNFUSED-DIVERGENCE" not in gate.PASS_VERDICTS
    assert row["unfused_agreement"]["first_divergent_checkpoint"] == 2
    assert "certified singles" in row["why"]


def test_the_fused_legs_own_divergence_is_named_first():
    row = _unfused_diverged(_passing_row())
    row["checkpoints"] = _checkpoints(unfused_bad_from=2, fused_bad_from=2)
    row["first_divergent_checkpoint"] = 2
    assert _verdict(row) == "DIVERGENCE"


def test_the_unfused_rule_precedes_every_later_clause():
    row = _unfused_diverged(_passing_row(substitution={"verdict": "NO-DROP"},
                                         second_consult_site={"verdict": "FAIL"}))
    assert _verdict(row) == "UNFUSED-DIVERGENCE"


def test_a_no_fused_arm_row_whose_singles_diverge_fails():
    spec = {"expect": "no-fused-arm"}
    row = _passing_row()
    assert _verdict(row, spec, fused_path="fused") == "PASS-NO-FUSED-ARM"
    assert _verdict(_unfused_diverged(row), spec) == "UNFUSED-DIVERGENCE"


def test_a_policy_refused_row_whose_singles_diverge_fails():
    report = {"refused_because": "the resolved float32 subnormal policy is 'keep'",
              "slots": {}}
    row = _passing_row(evidence={"driven": {}, "real": False, "failures": []})
    assert _verdict(row, report=report, fused_path="array",
                    policy="keep") == "PASS-POLICY-REFUSED"
    row = _unfused_diverged(row)
    assert _verdict(row, report=report, fused_path="array",
                    policy="keep") == "UNFUSED-DIVERGENCE"


# --------------------------------------------------------------------------
# 1b. The second consult site judges both kernel legs
# --------------------------------------------------------------------------

def _site(**over) -> Dict[str, Any]:
    out = {"synchronized_state": {"fused_vs_array": True, "unfused_vs_array": True},
           "restored_fused_vs_array": True, "restored_unfused_vs_array": True,
           "array_leg_launched": 0, "no_pair_was_declined_at_the_second_site": True}
    out.update(over)
    return out


def test_the_second_consult_site_passes_only_when_both_kernel_legs_agree():
    assert gate.second_consult_verdict(_site()) == ("PASS", [])
    verdict, failing = gate.second_consult_verdict(
        _site(synchronized_state={"fused_vs_array": True, "unfused_vs_array": False}))
    assert (verdict, failing) == ("FAIL", ["synchronized unfused == array"])
    verdict, failing = gate.second_consult_verdict(_site(restored_unfused_vs_array=False))
    assert failing == ["restored unfused == array"]
    site = _site()
    site.pop("restored_unfused_vs_array")
    assert gate.second_consult_verdict(site)[0] == "FAIL"


# --------------------------------------------------------------------------
# 2. The census reads without the residency read barrier
# --------------------------------------------------------------------------

def _barriered_fields(counter: Dict[str, int]):
    """A dataclass whose arrays sit behind a generated-subclass property, as a hold does."""

    @dataclasses.dataclass
    class Fields:
        Ex: Any = None
        polarizations: Any = None

    class Barriered(Fields):
        pass

    def barrier(name):
        def fetch(self):
            counter["barrier"] += 1
            return self.__dict__[name]

        def store(self, value):
            self.__dict__[name] = value
        return property(fetch, store)

    Barriered.Ex = barrier("Ex")

    class DictBarrier(dict):
        def __getitem__(self, key):
            counter["dict_barrier"] += 1
            return dict.__getitem__(self, key)

    # Spelt as WORDS, never converted from a float: a host thread left flushing by an
    # earlier test would turn a converted subnormal into zero before the census saw it.
    subnormal, one, two = 0x00000001, 0x3F800000, 0x40000000
    ex = numpy.array([subnormal, one, 0, subnormal], dtype=numpy.uint32).view(numpy.float32)
    pol = DictBarrier({"P": numpy.array([subnormal, two],
                                        dtype=numpy.uint32).view(numpy.float32)})
    return Barriered(Ex=ex, polarizations=pol)


def test_the_census_counts_without_touching_either_barrier():
    counter = {"barrier": 0, "dict_barrier": 0}
    fields = _barriered_fields(counter)
    driver = type("Driver", (), {"fields": fields, "pml": None})()
    counts = gate._census_state(driver)  # noqa: SLF001
    assert counts == {"fields.Ex": 2, "fields.polarizations['P']": 1}
    assert counter == {"barrier": 0, "dict_barrier": 0}, (
        "a barriered fetch moves a held volume to the host, and the census rides the "
        "fused leg before every launch")


def test_the_census_rides_only_the_fused_legs_own_launches(monkeypatch):
    from meep_gpu.metal_kernels import launch  # noqa: PLC0415

    ran: List[str] = []
    monkeypatch.setattr(launch.SyncedPlan, "run", lambda self, *a, **k: ran.append(self.tag))
    counter = {"barrier": 0, "dict_barrier": 0}
    fields = _barriered_fields(counter)
    driver = type("Driver", (), {"fields": fields, "pml": None})()

    class Residency:
        def __init__(self, owned):
            self.owned = owned

        def _by_host_identity(self, array):
            return object() if any(array is item for item in self.owned) else None

    mine = types.SimpleNamespace(residency=Residency([fields.__dict__["Ex"]]), tag="fused")
    other = types.SimpleNamespace(residency=Residency([numpy.zeros(4, numpy.float32)]),
                                  tag="unfused")
    restore, state = gate._seam_census(driver)  # noqa: SLF001
    try:
        for plan in (mine, other, mine):
            launch.SyncedPlan.run(plan)
    finally:
        restore()
    assert ran == ["fused", "unfused", "fused"], "every launch still runs"
    assert (state["launches"], state["other_legs_launches"]) == (2, 1)
    assert state["worst"] == 3
    assert counter == {"barrier": 0, "dict_barrier": 0}


def test_the_unbarriered_walk_finds_what_the_barriered_walk_finds():
    counter = {"barrier": 0, "dict_barrier": 0}
    fields = _barriered_fields(counter)
    driver = type("Driver", (), {"fields": fields, "pml": None})()
    raw: Dict[str, Any] = {}
    gate._walk_unbarriered(fields, "fields", raw, set(), 0)  # noqa: SLF001
    assert sorted(raw) == sorted(gate.e2e.collect_state(driver))


# --------------------------------------------------------------------------
# 3. The armed null: the window is placed, and NOT-APPLICABLE is named
# --------------------------------------------------------------------------

ACCUMULATING = {"update_H": {"Hx": "inside_and_outside", "Hy": "outside_only",
                             "Hz": "absent"}}
NONE_ACCUMULATE = {"update_H": {"Hx": "outside_only", "Hy": "absent", "Hz": "absent"}}


def _withheld(**over):
    kwargs = dict(withheld=24, launching_failures=[], identical=True,
                  overwrite_branch=False, constitutive_only=True, in_seam_carries=True,
                  structure=ACCUMULATING, reach={"first_change_step": 26})
    kwargs.update(over)
    return gate.withheld_verdict(**kwargs)


@pytest.mark.parametrize("over, expected", [
    ({"withheld": 0}, ("NOT-ARMED", "NOTHING-WITHHELD")),
    ({"launching_failures": ["x"]},
     ("LAUNCHING-SLOT-DID-NOT-LAUNCH", "LAUNCHING-SLOT-DID-NOT-LAUNCH")),
    ({"identical": False}, ("DIVERGED-AS-REQUIRED", "DIVERGED")),
    ({"overwrite_branch": True, "structure": None},
     ("NOT-APPLICABLE", "INACTIVE-LAYER")),
    ({"overwrite_branch": True, "structure": None, "in_seam_carries": False},
     ("NULL-DID-NOT-DIVERGE", "ARMED-AND-IDENTICAL")),
    ({"structure": NONE_ACCUMULATE}, ("NOT-APPLICABLE", "NO-OWN-AXIS-LAYER")),
    ({"structure": NONE_ACCUMULATE, "in_seam_carries": False},
     ("NULL-DID-NOT-DIVERGE", "ARMED-AND-IDENTICAL")),
    ({"reach": {"first_change_step": None}}, ("NOT-ARMED", "ACCUMULATION-NOT-REACHED")),
    ({}, ("NULL-DID-NOT-DIVERGE", "ARMED-AND-IDENTICAL")),
    ({"structure": None}, ("NULL-DID-NOT-DIVERGE", "ARMED-AND-IDENTICAL")),
    ({"constitutive_only": False, "structure": NONE_ACCUMULATE},
     ("NULL-DID-NOT-DIVERGE", "ARMED-AND-IDENTICAL")),
])
def test_the_withheld_ladder(over, expected):
    verdict = _withheld(**over)
    assert verdict == expected
    if expected[0] in ("NOT-ARMED", "NULL-DID-NOT-DIVERGE", "LAUNCHING-SLOT-DID-NOT-LAUNCH"):
        assert expected[0] not in gate.CONTROL_PASS


def test_the_accumulating_components_skip_plain_and_absent_ones():
    assert gate._accumulating_components(ACCUMULATING) == ["update_H:Hx"]  # noqa: SLF001
    assert gate._accumulating_components(NONE_ACCUMULATE) == []  # noqa: SLF001


def _stub_window(monkeypatch, *, first_change, compares):
    class _Driver:
        def close(self):
            pass

    driven = {"step_B": "pair", "update_H": "pair"}

    def lifted(*_a, **_k):
        leg = {"driver": _Driver(), "label": "withheld",
               "report": {"fusion": {"driven": driven}}}
        return leg, dict(leg, label="reference")

    def withhold():
        return (lambda: None), {"withheld": 12, "slots": {"update_H": 12},
                                "skipped_leading": {}, "skipped_launching": {}}

    stepped: List[int] = []
    monkeypatch.setattr(gate, "_pair_of_lifted", lifted)
    monkeypatch.setattr(gate, "step_leg",
                        lambda case, leg, counter, n: stepped.append(n))
    monkeypatch.setattr(gate.route, "_withhold_absorbed_consult", withhold)
    monkeypatch.setattr(gate.route, "_constitutive_is_an_overwrite", lambda leg: False)
    monkeypatch.setattr(gate.route, "_launching_slots_ran",
                        lambda leg, skipped, steps: {"slots": [], "deltas": {},
                                                     "failures": []})
    monkeypatch.setattr(gate, "_own_axis_structure", lambda leg, slots: ACCUMULATING)
    monkeypatch.setattr(gate, "reach_leg",
                        lambda case, res, counter, slots: {"first_change_step": first_change,
                                                           "first_change_by_slot": {},
                                                           "steps_stepped": 26, "cap": 200})
    monkeypatch.setattr(gate.e2e, "collect_state", lambda driver: {})
    answers = iter([True] + list(compares))

    def compare(_a, _b):
        same = next(answers)
        return {"identical": same, "arrays_differing": 0 if same else 4, "arrays": 32,
                "differences": []}

    monkeypatch.setattr(gate.e2e, "compare_state", compare)
    monkeypatch.setattr(gate, "say", lambda *_a, **_k: None)
    return stepped


def test_the_window_is_extended_to_control_steps_past_the_first_armed_step(monkeypatch):
    stepped = _stub_window(monkeypatch, first_change=26, compares=[True, False])
    row = gate.withheld_control("cylindrical_m1", {"seam": ["B"]}, None, None)
    assert stepped == [12, 12, 25, 25], "both legs, then both legs again by 26 + 11 - 12"
    assert row["window"] == {"control_steps": 12, "steps": 37, "extended": True,
                             "first_armed_step": 26}
    assert row["consults_withheld"] == 24
    assert (row["verdict"], row["reason"]) == ("DIVERGED-AS-REQUIRED", "DIVERGED")


def test_an_extended_window_that_stays_identical_is_still_a_failure(monkeypatch):
    _stub_window(monkeypatch, first_change=26, compares=[True, True])
    row = gate.withheld_control("cylindrical_m1", {"seam": ["B"]}, None, None)
    assert (row["verdict"], row["reason"]) == ("NULL-DID-NOT-DIVERGE",
                                               "ARMED-AND-IDENTICAL")
    assert "from step 26" in row["why"]


def test_a_reach_leg_that_never_fires_is_not_armed_by_name(monkeypatch):
    stepped = _stub_window(monkeypatch, first_change=None, compares=[True])
    row = gate.withheld_control("cylindrical_m1", {"seam": ["B"]}, None, None)
    assert stepped == [12, 12]
    assert (row["verdict"], row["reason"]) == ("NOT-ARMED", "ACCUMULATION-NOT-REACHED")


# --------------------------------------------------------------------------
# 4. The engine-smoke rows
# --------------------------------------------------------------------------

def test_the_engine_smoke_table_drives_m1_and_m2_held_and_shipped():
    assert {spec["m"] for spec in gate.ENGINE_SMOKE.values()} == {1, 2}
    assert gate.ENGINE_SMOKE_RESIDENCIES == ("held", "shipped")
    for name, spec in gate.ENGINE_SMOKE.items():
        assert name not in gate.DRIVE, "an engine-smoke row is not a route row"
        assert set(spec["arms"]) == {"step_B", "update_H", "step_D", "update_E"}
    held, shipped = gate.engine_smoke_env("held"), gate.engine_smoke_env("shipped")
    assert held["MEEP_GPU_METAL_RESIDENCY"] is None, "held is the default, left unset"
    assert shipped["MEEP_GPU_METAL_RESIDENCY"] == "shipped"
    for env in (held, shipped):
        assert {env[k] for k in ("MEEP_GPU_DISPATCH", "MEEP_GPU_FUSED",
                                 "MEEP_GPU_FUSE_ARMS")} == {None}
    with pytest.raises(ValueError):
        gate.engine_smoke_env("scoped")


ARMS = dict(gate._DCYL_COMPLEX_ARMS)  # noqa: SLF001


def _report(mode="held", steps=64, **over):
    report = {"fusion": {"driven": dict(ARMS)},
              "launch_counters": {slot: {"dispatches": steps} for slot in ARMS},
              "residency": {"mode": mode, "held": ["Ex"] if mode == "held" else [],
                            "admission": {"refused": False}}}
    report.update(over)
    return report


def _engine_row(**legs_over) -> Dict[str, Any]:
    legs = {}
    for label in gate.ENGINE_SMOKE_RESIDENCIES:
        record = gate.engine_leg_record(_report(label), "fused", 64, label, ARMS)
        legs[label] = {"checkpoints": [{"steps": 64, "identical": True}],
                       "first_divergent_checkpoint": None, "words_differing": 0,
                       "words_compared": 1376256, "record": record}
    for label, over in legs_over.items():
        legs[label].update(over)
    return {"precondition": {"identical": True}, "legs": legs,
            "comparator_control": {"armed": True, "comparator_saw_it": True,
                                   "words_mismatched": 1}}


def test_an_engine_row_passes_only_on_its_own_record():
    assert gate.engine_smoke_verdict(_engine_row(), policy_installed=None) == \
        "PASS-ENGINE-SMOKE"
    assert "PASS-ENGINE-SMOKE" in gate.PASS_VERDICTS


def test_a_held_leg_that_diverges_fails_the_engine_row():
    row = _engine_row(held={"first_divergent_checkpoint": 4, "words_differing": 111415})
    assert gate.engine_smoke_verdict(row, policy_installed=None) == "DIVERGENCE"
    assert "held: first at step 4, 111415 of 1376256 words" in row["why"]


@pytest.mark.parametrize("label, report, path, needle", [
    ("held", _report("held"), "array", "the step path is 'array'"),
    ("held", _report("held", fusion={"driven": {}}), "fused", "step_B drove None"),
    ("shipped", _report("shipped", steps=63), "fused", "dispatched 63 times over 64"),
])
def test_a_leg_whose_record_does_not_show_the_pairs_every_step_did_not_fuse(
        label, report, path, needle):
    record = gate.engine_leg_record(report, path, 64, label, ARMS)
    row = _engine_row(**{label: {"record": record}})
    assert gate.engine_smoke_verdict(row, policy_installed=None) == "DID-NOT-FUSE"
    assert needle in row["why"]


@pytest.mark.parametrize("label, report, needle", [
    ("held", _report("shipped"), "the residency mode is 'shipped', not 'held'"),
    ("shipped", _report("held"), "the residency mode is 'held', not 'shipped'"),
    ("held", dict(_report("held"), residency={"mode": "held", "held": [],
                                              "admission": {"refused": False}}),
     "the held set is empty"),
    ("held", dict(_report("held"), residency={"mode": "held", "held": ["Ex"],
                                              "admission": {"refused": False},
                                              "hold_refused": ["Hy"]}),
     "holds refused"),
])
def test_a_leg_that_ran_another_residency_than_it_asked_for_fails(label, report, needle):
    record = gate.engine_leg_record(report, "fused", 64, label, ARMS)
    row = _engine_row(**{label: {"record": record}})
    assert gate.engine_smoke_verdict(row, policy_installed=None) == \
        "RESIDENCY-NOT-AS-ASKED"
    assert needle in row["why"]


def test_an_engine_row_under_an_installed_keep_must_refuse_and_still_agree():
    refused = gate.engine_leg_record(_report("held"), "array", 64, "held", ARMS)
    row = _engine_row(held={"record": refused}, shipped={"record": refused})
    assert gate.engine_smoke_verdict(row, policy_installed="keep") == \
        "PASS-POLICY-REFUSED"
    row = _engine_row()
    assert gate.engine_smoke_verdict(row, policy_installed="keep") == \
        "DISPATCHED-UNDER-AN-UNCERTIFIED-POLICY"


def test_a_blind_comparator_or_unequal_lifts_fail_the_engine_row():
    row = _engine_row()
    row["comparator_control"] = {"armed": True, "comparator_saw_it": False}
    assert gate.engine_smoke_verdict(row, policy_installed=None) == "COMPARATOR-BLIND"
    row = _engine_row()
    row["precondition"] = {"identical": False}
    assert gate.engine_smoke_verdict(row, policy_installed=None) == "HARNESS-FAILURE"
