"""The route gate's controls dry run reads every case's wrappers before a freeze.

WHY THIS FILE EXISTS. The 2026-10-13 round's held Metal route campaign failed 2 of its
4 legs on ``residency-inner-bracket-dropped``: series entry 134 had put
``launch.StartOfHPlan`` into the plans of ``special_kz_2d`` and ``bfast_1d``, the
control armed on it and read NULL-DID-NOT-DIVERGE, and the route smoke before the
freeze had run two OTHER cases, on which the control could not arm. The control's
answer depends on the wrappers a case's frozen plan holds, which only a lift shows.

``--controls-dry-run`` reads that for every drive case on one lift each: the wrapper
chains (refused when a wrapper is not in ``ROUTE_WRAPPER_RULINGS``), each residency
control's state after the gate's own arm schedule and the verdict class it implies,
the moves against a previous campaign leg's rows, and the cases a full gate must
smoke. Host-only: stand-in launches and residencies around the real ``launch`` and
wrapper classes, with the lift and the stepping replaced.
"""

from __future__ import annotations

import json
import pathlib
import sys
import types

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gate_dispatch_metal_route as metal_route  # noqa: E402

from meep_gpu import deposit_repair, withdraw_hoist  # noqa: E402
from meep_gpu.metal_kernels import device as device_module  # noqa: E402
from meep_gpu.metal_kernels import launch  # noqa: E402

PLACEMENT = "residency-inner-bracket-dropped"
SYNC_IN = "residency-sync_in-dropped"
FLOOR = "residency-verify-after-every-sync_out"


class _Launch:
    def __init__(self, name):
        self.name = name

    def run(self, *args, **kwargs):
        pass


class _Wrapper:
    """A wrapper no control was written for: it holds an ``inner`` and does host work."""

    def __init__(self, inner):
        self.inner = inner
        self.absorbed_by = inner

    def run(self, *args, **kwargs):  # pragma: no cover - never stepped here
        self.inner.run(*args, **kwargs)


def _residency_class():
    class _Residency:
        held = frozenset({"Ez"})

        def __init__(self):
            self.names = ("Ez", "Dz")
            self.sparse_scatters = self.sparse_zeros = self.double_subtracts = 0

        def _targets(self, target):
            return []

        def acquire_write(self, target):
            return target

        def sync_in(self, names=None):
            pass

        def sync_out(self, names=None):
            pass

        def verify(self):
            return {}

        def by_state(self):
            return {device_module.DEVICE_OWNED: ()}

    return _Residency


def _driver(residency, *, start_of_h=True, extra=None):
    """``special_kz_2d``'s 2026-10-13 shape: a leading repair on step_D and, with
    ``start_of_h``, a start of H around a bracketed single on update_H."""
    fields = types.SimpleNamespace(h_separated=False)
    repair = deposit_repair.LeadingRepairPlan(_Launch("electric pair"), fields, None, (),
                                              "D")
    plans = {"step_D": launch.synced(repair, residency)}
    if start_of_h:
        plans["update_H"] = launch.synced(
            launch.StartOfHPlan(_Launch("update_H single"), fields, None,
                                after_launch=False), residency)
    if extra is not None:
        plans["update_E"] = extra(launch.SyncedPlan(_Launch("update_E"), residency))
    driver = types.SimpleNamespace(
        fields=fields, closed=False,
        _fast_path=types.SimpleNamespace(step_plan=types.SimpleNamespace(plans=plans)))
    driver.close = lambda: setattr(driver, "closed", True)
    return driver


# ---------------------------------------------------------- the plan's wrappers ----

def test_the_chains_name_every_wrapper_down_to_the_occupant():
    chains = metal_route.plan_chains(_driver(_residency_class()()))
    assert chains == {"step_D": ["LeadingRepairPlan", "SyncedPlan", "_Launch"],
                      "update_H": ["StartOfHPlan", "SyncedPlan", "_Launch"]}, chains
    assert metal_route.unruled_wrappers(chains) == {}


def test_a_wrapper_no_ruling_names_is_reported_by_slot(monkeypatch):
    """MUST CATCH: entry 134's wrapper, unknown to the controls, refuses before a freeze."""
    rulings = dict(metal_route.ROUTE_WRAPPER_RULINGS)
    rulings.pop("StartOfHPlan")
    monkeypatch.setattr(metal_route, "ROUTE_WRAPPER_RULINGS", rulings)
    chains = metal_route.plan_chains(_driver(_residency_class()()))
    assert metal_route.unruled_wrappers(chains) == {"update_H": ["StartOfHPlan"]}


def test_the_rulings_name_every_wrapper_class_the_metal_composer_installs():
    for cls in (launch.SyncedPlan, launch.StartOfHPlan, deposit_repair.LeadingRepairPlan,
                withdraw_hoist.LeadingWithdrawPlan):
        assert cls.__name__ in metal_route.ROUTE_WRAPPER_RULINGS, cls


# ------------------------------------------------------------- the prediction ----

@pytest.mark.parametrize("name,state,expected", [
    (PLACEMENT, {"armed": False, "looked_through": {"update_H": {}}}, "NOT-APPLICABLE"),
    (PLACEMENT, {"armed": True, "slots": ["update_H"]}, "ARMS"),
    (PLACEMENT, {"armed": False, "started_while_looked_through": ["update_H"]},
     "HARNESS-FAILURE"),
    (SYNC_IN, {"armed": True, "mode": metal_route.HELD_SYNC_IN_MODE, "suppressed": 0},
     "NOT-APPLICABLE"),
    (SYNC_IN, {"armed": True, "mode": metal_route.HELD_SYNC_IN_MODE, "suppressed": 1},
     "ARMS"),
    (SYNC_IN, {"armed": True, "steps_with_stale_mirrors": 0}, "ARMS"),
    (FLOOR, {"reads": 0, "mirrors": 45}, "NOT-ARMED"),
    (FLOOR, {"reads": 24, "mirrors": 45}, "READS"),
])
def test_each_state_implies_the_class_the_gate_reads_from_it(name, state, expected):
    assert metal_route.predict_control(name, state) == expected


def test_a_plan_shape_the_reference_never_graded_is_a_move():
    """MUST CATCH: the shape the round's route campaign met, against the leg before it."""
    reference = {("special_kz_2d", PLACEMENT): {
        "verdict": "NOT-APPLICABLE",
        "state": {"armed": False, "slots": [], "read_only_slots": ["step_D"],
                  "wrapper_kinds": {"step_D": "LeadingRepairPlan"}}}}
    driver = _driver(_residency_class()())
    driver.fields.h_separated = True     # one step past the freeze, as the gate arms
    restore, state = metal_route._drop_the_inner_bracket(driver)  # noqa: SLF001
    restore()
    controls = {PLACEMENT: {"state": state,
                            "predicts": metal_route.predict_control(PLACEMENT, state)}}
    moves = metal_route.dry_run_moves("special_kz_2d", controls, reference)
    assert moves == [f"{PLACEMENT}: looked_through {{}} -> {{'update_H': 'SyncedPlan'}}"], \
        moves
    same = {("special_kz_2d", PLACEMENT): {"verdict": "NOT-APPLICABLE", "state": state}}
    assert metal_route.dry_run_moves("special_kz_2d", controls, same) == []


# ----------------------------------------------------------- the whole dry run ----

@pytest.fixture
def dry_run(monkeypatch, tmp_path):
    """Run :func:`controls_dry_run` with the lift and the stepping replaced.

    Each "lift" builds :func:`_driver` with ``shape``'s keywords; a "step" separates H
    as the driver does after every step, sends the doors one write, and runs every
    residency bracket of the plan, so the vacuity floor's patched bracket reads.
    """

    def run(names, shape=None, reference=None):
        cls = _residency_class()
        shapes = shape or {}

        def lift(case, builder, label, env, counter, gpu_id, res):
            residency = cls()
            return {"label": label, "env": env, "residency": residency,
                    "driver": _driver(residency, **shapes.get(case, {}))}

        def step(case, leg, counter, num_steps):
            leg["driver"].fields.h_separated = True
            leg["report"] = {"fusion": {"driven": {"step_D": "fused electric D/E pair",
                                                   "update_E": "fused electric D/E pair"}}}
            for _ in range(num_steps):
                leg["residency"].sparse_scatters += 1
                for synced in metal_route._synced_plans(leg["driver"]):  # noqa: SLF001
                    synced.run()

        monkeypatch.setattr(metal_route.e2e, "run_leg", lift)
        monkeypatch.setattr(metal_route, "step_leg", step)
        monkeypatch.setattr(metal_route, "require_metal_driver", lambda driver, what: None)
        monkeypatch.setattr(metal_route.gate_provenance, "stamp", lambda payload: payload)
        ref = None
        if reference is not None:
            ref = tmp_path / "controls.jsonl"
            ref.write_text("".join(json.dumps(row) + "\n" for row in reference))
        out = tmp_path / f"out{len(list(tmp_path.glob('out*')))}"
        out.mkdir()
        rc = metal_route.controls_dry_run(names, None, str(out), {}, str(ref) if ref else None)
        summary = json.loads((out / "controls_dry_run.json").read_text())
        rows = [json.loads(line) for line in
                (out / "controls_dry_run.jsonl").read_text().splitlines()]
        return rc, summary, rows

    return run


def test_every_case_is_read_on_one_lift_and_written_as_it_lands(dry_run):
    names = ["special_kz_2d", "pml_2d"]
    rc, summary, rows = dry_run(names, shape={"pml_2d": {"start_of_h": False}})
    assert rc == 0, summary["problems"]
    assert [row["case"] for row in rows] == names
    assert summary["cases_read"] == 2 and summary["ok"] is True
    for row in rows:
        assert row["controls"][PLACEMENT]["predicts"] == "NOT-APPLICABLE", row
        assert row["controls"][SYNC_IN]["predicts"] == "NOT-APPLICABLE", row
        assert row["controls"][FLOOR]["predicts"] == "READS", row
        held = row["controls"][SYNC_IN]["state"]
        assert held["door_counts"]["sparse_scatters"] == metal_route.CONTROL_STEPS, held
        assert row["controls"][PLACEMENT]["steps"] == 0
    assert rows[0]["controls"][PLACEMENT]["state"]["looked_through"]["update_H"][
        "occupant"] == "SyncedPlan"
    assert summary["wrapper_slots"] == {"LeadingRepairPlan": 2, "SyncedPlan": 3,
                                        "StartOfHPlan": 1}, summary["wrapper_slots"]
    assert summary["smoke_cases"] == []


def test_a_case_whose_placement_control_arms_is_named_for_the_full_gate(dry_run):
    names = ["special_kz_2d", "pml_2d"]
    shape = {"pml_2d": {"start_of_h": False,
                        "extra": lambda bracketed: withdraw_hoist.LeadingWithdrawPlan(
                            bracketed, types.SimpleNamespace(), ())}}
    rc, summary, rows = dry_run(names, shape=shape)
    assert rows[1]["controls"][PLACEMENT]["predicts"] == "ARMS", rows[1]
    assert summary["smoke_cases"] == ["pml_2d"], summary
    assert rc == 0, summary["problems"]


def test_an_unruled_wrapper_or_a_move_needs_a_ruling(dry_run):
    rc, summary, _rows = dry_run(["pml_2d"], shape={"pml_2d": {"extra": _Wrapper}})
    assert rc == 1
    assert any("_Wrapper" in line and "ROUTE_WRAPPER_RULINGS" in line
               for line in summary["problems"]), summary["problems"]
    reference = [{"case": "special_kz_2d", "control": name, "verdict": verdict,
                  "state": {"armed": False, "wrapper_kinds": {"step_D": "LeadingRepairPlan"},
                            "read_only_slots": ["step_D"]}}
                 for name, verdict in ((PLACEMENT, "NOT-APPLICABLE"),
                                       (SYNC_IN, "NOT-APPLICABLE"),
                                       (FLOOR, "RESIDENCY-CLEAN"))]
    rc, summary, rows = dry_run(["special_kz_2d", "bfast_1d"], reference=reference)
    assert rc == 1
    assert rows[0]["moved"] == [
        f"{PLACEMENT}: looked_through {{}} -> {{'update_H': 'SyncedPlan'}}"], rows[0]
    assert "bfast_1d" in summary["smoke_cases"], summary   # never graded by the reference
    assert all("no row in the reference leg" in move for move in rows[1]["moved"])


def test_a_raising_case_is_a_recorded_row_and_a_problem(dry_run, monkeypatch):
    def boom(case, spec, res, counter):
        raise RuntimeError("lift failed")

    monkeypatch.setattr(metal_route, "controls_dry_run_case", boom)
    rc, summary, rows = dry_run(["pml_2d"])
    assert rc == 1 and rows[0]["error"].startswith("RuntimeError"), rows
    assert summary["cases_read"] == 0


@pytest.mark.parametrize("extra", [["--band-witness"], ["--no-controls"],
                                   ["--install-policy", "flush"]])
def test_the_dry_run_takes_the_shipped_configuration_only(tmp_path, extra):
    assert metal_route.main(["--out", str(tmp_path / "o"), "--controls-dry-run",
                             *extra]) == 2
    assert not (tmp_path / "o").exists()


def test_a_reference_without_the_dry_run_is_refused(tmp_path):
    assert metal_route.main(["--out", str(tmp_path / "o"), "--dry-run-reference",
                             str(tmp_path / "c.jsonl")]) == 2
