"""The route gate's absorbed-consult-withheld control: where it can diverge, off-device.

WHY THIS FILE EXISTS. The control lets the ARRAY ``update_H`` / ``update_E`` run on top
of the fused launch's own constitutive half and requires the two legs to part. Since
the kernels' constitutive halves became bit-identical to the array path's (series entry
22-KT1 on the Triton table), the only thing that parts them is the dsigw ACCUMULATION,
and the precert pass of 2026-10-09 read NULL-DID-NOT-DIVERGE on pml_2d, folded_2d,
offdiag_2d and the 1-D, complex and cylindrical cases on every route leg of both NVIDIA
tables -- blocking every release -- while every case itself was byte-identical.

Measured on the NumPy array path of the 01..86 tree, two causes, both pinned here:

* the accumulation lives on a component's OWN-axis layer only, so on pml_1d (Ex and Hy
  over a z layer) no withheld component can ever accumulate: NOT-APPLICABLE by name;
* on pml_2d the magnetic components do accumulate, but a second application first
  changes a word at step 13, one past the control's twelve-step window: the window is
  placed by a reach leg, not fixed.

The off-device half: the verdict table, the structure reader's fail-closed rules, and
(with MEEP installed) the re-application witness on real NumPy lifts -- the step it
fires at, and that the leg it watches steps exactly as an unwatched one.
"""

from __future__ import annotations

import pathlib
import sys
import types

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import gate_dispatch_end_to_end as e2e  # noqa: E402
import gate_dispatch_fused_route as route_gate  # noqa: E402

ACCUMULATING = {"update_H": {"Hx": "split", "Hy": "select", "Hz": "absent"},
                "update_E": {"Ex": "absent", "Ey": "absent", "Ez": "outside_only"}}
PLAIN = {"update_H": {"Hx": "absent", "Hy": "outside_only", "Hz": "absent"},
         "update_E": {"Ex": "outside_only", "Ey": "absent", "Ez": "absent"}}


def _verdict(**overrides):
    arguments = {"withheld": 24, "launching_failures": [], "identical": True,
                 "overwrite_branch": False, "constitutive_only": True,
                 "structure": ACCUMULATING,
                 "reach": {"first_change_step": 13}}
    arguments.update(overrides)
    return route_gate.withheld_verdict(**arguments)


# ---------------------------------------------------------------------------
# The verdict table
# ---------------------------------------------------------------------------

def test_a_divergence_passes_whatever_the_structure_says():
    for structure in (ACCUMULATING, PLAIN, None):
        assert _verdict(identical=False, structure=structure) == (
            "DIVERGED-AS-REQUIRED", "DIVERGED")


def test_armed_and_identical_is_the_null_failure():
    """The must-catch branch: a reach leg that fired and legs that did not part."""
    verdict, reason = _verdict()
    assert verdict == "NULL-DID-NOT-DIVERGE" and reason == "ARMED-AND-IDENTICAL"
    assert verdict not in route_gate.CONTROL_PASS


def test_no_own_axis_layer_is_not_applicable_by_name():
    verdict, reason = _verdict(structure=PLAIN, reach=None)
    assert (verdict, reason) == ("NOT-APPLICABLE", "NO-OWN-AXIS-LAYER")
    assert verdict in route_gate.CONTROL_PASS


def test_the_inactive_layer_ruling_is_unchanged():
    assert _verdict(overwrite_branch=True, structure=None, reach=None) == (
        "NOT-APPLICABLE", "INACTIVE-LAYER")


def test_an_unreached_accumulation_is_a_named_failure():
    verdict, reason = _verdict(reach={"first_change_step": None})
    assert (verdict, reason) == ("NOT-ARMED", "ACCUMULATION-NOT-REACHED")
    assert verdict not in route_gate.CONTROL_PASS


def test_ignorance_grants_nothing():
    """FAIL CLOSED: an unreadable plan, a non-constitutive withheld slot, or an
    unreadable layer leaves an identical comparison a failure."""
    assert _verdict(structure=None, reach=None)[0] == "NULL-DID-NOT-DIVERGE"
    assert _verdict(constitutive_only=False, structure=PLAIN)[0] == (
        "NULL-DID-NOT-DIVERGE")
    assert _verdict(overwrite_branch=None, structure=None, reach=None)[0] == (
        "NULL-DID-NOT-DIVERGE")


def test_nothing_withheld_and_a_silent_group_still_fail_first():
    assert _verdict(withheld=0) == ("NOT-ARMED", "NOTHING-WITHHELD")
    assert _verdict(launching_failures=["update_P did not launch"])[0] == (
        "LAUNCHING-SLOT-DID-NOT-LAUNCH")


def test_the_accumulating_components_are_read_per_slot():
    assert route_gate._accumulating_components(ACCUMULATING) == [  # noqa: SLF001
        "update_H:Hx", "update_H:Hy"]
    assert route_gate._accumulating_components(PLAIN) == []  # noqa: SLF001


# ---------------------------------------------------------------------------
# The structure reader
# ---------------------------------------------------------------------------

class _Plan:
    def __init__(self, pml, modes):
        self.pml = pml
        self._modes = modes

    def spec(self, sub_step, component):
        assert sub_step == "constitutive"
        return types.SimpleNamespace(mode=self._modes[component])


def _leg(modes, *, absent=(), plan_pml=None, pml=None):
    pml = pml if pml is not None else object()
    plan = _Plan(plan_pml if plan_pml is not None else pml, modes)
    fields = types.SimpleNamespace(region_plan=plan, absent_components=tuple(absent))
    return {"driver": types.SimpleNamespace(fields=fields, pml=pml)}


def test_the_structure_is_read_off_the_installed_plan():
    modes = {"Hx": "split", "Hy": "select", "Hz": "outside_only",
             "Ex": "outside_only", "Ey": "outside_only", "Ez": "outside_only"}
    got = route_gate._own_axis_structure(  # noqa: SLF001
        _leg(modes, absent=("Hz", "Ex", "Ey")), {"update_H": 12, "update_E": 12})
    assert got == ACCUMULATING


def test_the_structure_reader_fails_closed():
    modes = {name: "outside_only" for name in ("Hx", "Hy", "Hz", "Ex", "Ey", "Ez")}
    reader = route_gate._own_axis_structure  # noqa: SLF001
    assert reader({"driver": None}, {"update_H": 1}) is None
    assert reader(_leg(modes, plan_pml=object()), {"update_H": 1}) is None
    assert reader(_leg(modes), {"update_P": 1}) is None
    no_plan = _leg(modes)
    no_plan["driver"].fields.region_plan = None
    assert reader(no_plan, {"update_H": 1}) is None


# ---------------------------------------------------------------------------
# The re-application witness, on real NumPy lifts (MEEP required)
# ---------------------------------------------------------------------------

def _numpy_lift(case):
    mp = pytest.importorskip("meep")
    import meep_gpu  # noqa: PLC0415

    try:
        mp.verbosity(0)
    except Exception:  # noqa: BLE001
        pass
    sim, _monitors, _until = route_gate.CASES[case](mp)
    return meep_gpu.lift_simulation(sim, prefer_gpu=False)


def _fire(case, steps):
    driver = _numpy_lift(case)
    restore, state = route_gate._reapplication_witness(  # noqa: SLF001
        driver.fields, ("update_H", "update_E"))
    try:
        for _ in range(steps):
            driver.step()
            if len(state["first_change"]) == 2:
                break
    finally:
        restore()
    structure = route_gate._own_axis_structure(  # noqa: SLF001
        {"driver": driver}, ("update_H", "update_E"))
    driver.close()
    return state, structure


@pytest.mark.parametrize("case, first_h", [("pml_2d", 13), ("pml_3d", 2)])
def test_the_witness_fires_where_the_accumulation_is_reached(monkeypatch, case, first_h):
    monkeypatch.setenv("MEEP_GPU_DISPATCH", "0")
    state, structure = _fire(case, 20)
    assert state["first_change"].get("update_H") == first_h, state["first_change"]
    assert route_gate._accumulating_components(structure)  # noqa: SLF001


def test_pml_1d_has_no_own_axis_cell_and_never_fires(monkeypatch):
    monkeypatch.setenv("MEEP_GPU_DISPATCH", "0")
    state, structure = _fire("pml_1d", 60)
    assert state["first_change"] == {}
    assert structure is not None
    assert route_gate._accumulating_components(structure) == []  # noqa: SLF001


def test_the_watched_leg_steps_exactly_as_an_unwatched_one(monkeypatch):
    """THE WITNESS MUST NOT MOVE THE LEG IT WATCHES. A witness that leaked a write
    would make every control it sits beside read DIVERGED-AS-REQUIRED -- a silent
    pass. Compared over every array on the fields and the PML, past the step the
    witness first fires at."""
    monkeypatch.setenv("MEEP_GPU_DISPATCH", "0")
    watched, plain = _numpy_lift("pml_2d"), _numpy_lift("pml_2d")
    restore, state = route_gate._reapplication_witness(  # noqa: SLF001
        watched.fields, ("update_H", "update_E"))
    try:
        for _ in range(20):
            watched.step()
    finally:
        restore()
    for _ in range(20):
        plain.step()
    assert state["first_change"].get("update_H") == 13
    verdict = e2e.compare_state(e2e.collect_state(watched),
                                e2e.collect_state(plain))
    watched.close()
    plain.close()
    assert verdict["identical"], verdict["differences"][:3]
    assert verdict["words_compared"] > 0
