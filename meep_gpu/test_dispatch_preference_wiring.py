"""The dispatch preference WIRED: measured cost arbitrating the composer's fused slots.

1.0.1 puts the measured-cost veto in, fast by default: a fused product an
architecture's timing record measures slower than its certified singles gives way to
them, and an architecture with NO record of its own must not keep a slower default for
lack of one (the portable prior). ``test_dispatch_preference.py`` tests the consult and
the record format; this file tests what the ladder does with them
(``fastpath._apply_preference`` at rung 5, ``metal_dispatch.decide`` at rung 5M):

* KNOWN VALUE -- a vetoed product dispatches its certified singles; a keep verdict, an
  unmeasured key and a size outside the measured bands keep the shape rule;
* The no-record rule -- an architecture with no record of its own is arbitrated by the
  portable prior and says so in one NOTE line; the prior keeps a clean seam and a grid
  above the recorded crossover, and with no usable record anywhere it vetoes nothing;
  it is its vendor's, never another's: NVIDIA records never arbitrate the Metal table,
  Apple records never an NVIDIA table, and each verdict names its vendor and records;
* PER ARCHITECTURE AND CARD -- the record is the live card's own when one is shipped,
  else the representative named for the live device's architecture, never another
  architecture's: the dispatch record states the role, the record's card and the live
  card; a card's own record that refuses or answers a barrier is not replaced by the
  representative; an architecture with no record reads the prior of its vendor's
  NEAREST timed architecture (compute capability for NVIDIA, chip generation for Apple),
  and a spelling the rule cannot read is refused by name, not guessed;
* CERTIFIED -- ``MEEP_GPU_HARDWARE=certified`` arbitrates exactly as the default does
  and the preference never moves ``certified``;
* THE TAKER -- where withholding a product hands its slots to another fused product,
  that product is judged by its own table at the run's size: vetoed too, both are
  withheld and must end on certified singles; kept, the veto is dropped; and when the
  wider composition does not stand, the step goes on exactly as without it;
* FAIL-SAFE -- a veto the second composition cannot confirm (the slot would go to the
  array path, or to singles the record did not price) is dropped; a refused record is
  loud and treated as no record; an explicit opt-in is never arbitrated; ``span`` opens
  no record; an unreadable switch value is refused by name;
* THE ROUTE -- the deposit-repair route predicted at plan time is the route the
  bracket's own counters then write, on real NumPy fields and sources.

Nothing here launches a kernel. The composers are stand-ins that install a fused pair
when its label is offered and the single arms when it is not, which is the contract the
real composers keep (``fuse_labels``); the device half is the certification run's.
"""

from __future__ import annotations

import json
import os
import pathlib
import types

import numpy
import pytest

import meep_gpu.fastpath as fastpath
import meep_gpu.subnormal_policy as policy_module
from meep_gpu import deposit_repair
from meep_gpu import dispatch_preference as dp
from meep_gpu.test_dispatch_contract import (CountingPlan, block_cuda,
                                             certified_envelope_fields,
                                             certified_envelope_grid,
                                             certified_envelope_pml, step_plan,
                                             stub_triton)
from meep_gpu.test_dispatch_preference import _entry, _record
from meep_gpu.triton_kernels import launch as launch_module

HERE = os.path.dirname(os.path.abspath(__file__))

#: The architecture the stand-in device reads, and one no record here names.
RECORDED = "8.6"
UNRECORDED = (8, 9)


# ---------------------------------------------------------------------------
# Scaffolding
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_process(monkeypatch, tmp_path):
    policy_module._reset_for_tests()
    fastpath.reset_dispatch_announcements()
    fastpath._POLICY_INSTALLED_BY_DISPATCH = None
    for name in (fastpath.FUSED_KILL_SWITCH, fastpath.SUBNORMAL_INSTALL_SWITCH,
                 fastpath.DISPATCH_LOG, fastpath.FUSE_ARMS_SWITCH,
                 fastpath.BACKEND_PREFERENCE_SWITCH, fastpath.KERNEL_TABLE_SWITCH,
                 fastpath.HARDWARE_SWITCH, fastpath.PREFERENCE_SWITCH,
                 policy_module.POLICY_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(fastpath.DISPATCH_ENABLE, "1")
    # The stand-in plans have nothing to compile; the warm pass is a separate rung.
    monkeypatch.setenv(fastpath.WARM_SWITCH, "0")
    monkeypatch.setenv("CUPY_CACHE_DIR", str(tmp_path / "cupy-cache"))
    monkeypatch.setattr(fastpath, "TIMING_RECORD_ROOT", str(tmp_path / "records"))
    yield
    policy_module._reset_for_tests()
    fastpath.reset_dispatch_announcements()
    fastpath._POLICY_INSTALLED_BY_DISPATCH = None


def _root(tmp_path) -> pathlib.Path:
    return tmp_path / "records"


#: The route every record below was "cut" on: the LIVE deposit-repair digest and the
#: cheapest bracket, which is what the shipped records name.
def _route_facts():
    return {"deposit_repair_sha256": dp.repair_route_sha256(), "index": "linear",
            "repairs_per_bracketed_step": 1.0, "restricted": True, "fell_back": False}


def _barrier(table="triton"):
    live = dp.live_subject_sha256(table)
    return {"sources": sorted(live), "digests": {name: [digest]
                                                 for name, digest in live.items()},
            "rows_not_recording": {name: 0 for name in live}, "allowed_unpinned": []}


def _write(tmp_path, entries, *, table="triton", capability=RECORDED, route=None,
           device="STAND-IN", name=None):
    """A record cut on ``device``, at the architecture's name or at ``name`` beside it."""
    facts = dict(_route_facts(), **(route or {}))
    digest = facts.pop("deposit_repair_sha256")
    record = _record(entries, table=table, route=digest, barrier=_barrier(table),
                     host=[{"device": device, "compute_capability": capability}],
                     **facts)
    path = pathlib.Path(dp.record_path(table, capability, root=str(_root(tmp_path))))
    if name is not None:
        path = path.with_name(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dp.dumps(record), encoding="utf-8")
    return path


def _slow_d(**overrides):
    """``fused pair D`` bracketed, 0.72x its singles, 24,000-2,400,000 cells (the gated band)."""
    spec = dict(product="fused pair D", bracketed=True, lo=24000, hi=2400000,
                fused=0.87, baseline=0.63, displaces=("PML", "ordinary"),
                more_rows=((0.86, 0.62),))
    spec.update(overrides)
    return _entry(**spec)


def _keep_d(**overrides):
    spec = dict(product="fused pair D", bracketed=True, lo=5400000, hi=5400000,
                fused=1.22, baseline=1.27, displaces=("PML", "ordinary"),
                more_rows=((1.23, 1.27),))
    spec.update(overrides)
    return _entry(**spec)


def _clean_b():
    return _entry(product="fused pair B", bracketed=False, lo=24000, hi=5400000,
                  fused=0.62, baseline=0.654, displaces=("PML", "ordinary"),
                  more_rows=((0.61, 0.66),))


def _device_grid(capability=(8, 6), shape=None, name="STAND-IN"):
    grid = certified_envelope_grid()
    if shape is not None:
        grid.shape = shape
    grid.xp.cuda = types.SimpleNamespace(runtime=types.SimpleNamespace(
        getDevice=lambda: 0,
        getDeviceProperties=lambda index: {"name": name.encode(), "major": capability[0],
                                           "minor": capability[1]},
        runtimeGetVersion=lambda: 11080, driverGetVersion=lambda: 12030))
    return grid


class Composer:
    """A Triton composer stand-in that keeps the real one's ``fuse_labels`` contract.

    A pair whose label is offered takes both slots of its seam (with the deposit-repair
    bracket when ``bracketed`` names that seam); otherwise the seam keeps its single
    arms, ``singles`` naming their labels, or stays unfilled when ``singles`` is None.
    Every offer is recorded, so a test reads how many compositions ran and with what.
    """

    def __init__(self, bracketed=("D",), singles=("PML", "ordinary"), pairs=("B", "D")):
        self.bracketed = set(bracketed)
        self.singles = singles
        self.pairs = pairs
        self.offers = []

    def __call__(self, fields, pml, **kwargs):
        offered = set(kwargs.get("fuse_labels") or ())
        self.offers.append(sorted(offered))
        plans, selected = {}, {}
        for seam, curl, update in (("B", "step_B", "update_H"),
                                   ("D", "step_D", "update_E")):
            label = f"fused pair {seam}"
            if seam in self.pairs and label in offered:
                fused = CountingPlan(label)
                if seam in self.bracketed:
                    leading = deposit_repair.LeadingRepairPlan(fused, fields, pml, (),
                                                               seam)
                    plans[curl] = leading
                    plans[update] = deposit_repair.TrailingRepairPlan(update, leading,
                                                                      fields, pml)
                else:
                    plans[curl] = fused
                    plans[update] = launch_module.NoopPlan(update, fused)
                selected[curl] = selected[update] = label
            elif self.singles is not None:
                plans[curl] = CountingPlan(f"{seam} curl")
                plans[update] = CountingPlan(f"{seam} update")
                selected[curl], selected[update] = self.singles
        return step_plan(plans, selected)


def _freeze(monkeypatch, composer, *, capability=(8, 6), shape=None,
            route=None, name="STAND-IN"):
    """One configuration freeze on the released envelope (pml_2d's shape, 24,000 cells).

    The plan-time route prediction is a stand-in here (the stand-in bracket has no
    source to predict from); :func:`test_the_route_predicted_at_plan_time_is_the_one_the_counters_write`
    tests the real one.
    """
    stub_triton(monkeypatch)
    block_cuda(monkeypatch)
    import meep_gpu.triton_kernels as package

    monkeypatch.setattr(package, "plan_step", composer)
    monkeypatch.setattr(fastpath, "_planned_repair_route",
                        lambda leading, digest: dict(_route_facts(), **(route or {})))
    plan = fastpath.plan_fast_path(certified_envelope_fields(), certified_envelope_pml(),
                                   _device_grid(capability, shape, name), sources=())
    return plan, fastpath.last_dispatch_report()


def _arms(record):
    return {slot: entry["arm"] for slot, entry in record["slots"].items()
            if entry.get("state") == "dispatched"}


def _candidate(record, label):
    found = [item for item in record["preference"]["candidates"]
             if item["label"] == label]
    assert len(found) == 1, record["preference"]
    return found[0]


# ---------------------------------------------------------------------------
# Known value: the veto, and the three ways the shape rule stays
# ---------------------------------------------------------------------------


def _mark_merges(monkeypatch):
    """Each merge stamps its ordinal into the arbitration block it was handed."""
    from meep_gpu import fastpath_cuda

    real = fastpath_cuda.merge_tables
    count = []

    def marking(composed, pending_of, record):
        count.append(1)
        record.setdefault("arbitration", {}).setdefault("marks", []).append(len(count))
        return real(composed, pending_of=pending_of, record=record)

    monkeypatch.setattr(fastpath_cuda, "merge_tables", marking)


def test_a_vetoed_product_dispatches_its_certified_singles(monkeypatch, tmp_path):
    _write(tmp_path, [_slow_d(), _clean_b()])
    composer = Composer()
    _mark_merges(monkeypatch)
    plan, record = _freeze(monkeypatch, composer)
    # The record describes the composition that runs: the second merge's writes, made
    # into a copy of what the first one started from, and nothing of the first's.
    assert record["arbitration"]["marks"] == [2]
    assert plan is not None, record["refused_because"]
    assert _arms(record) == {"step_B": "fused pair B", "update_H": "fused pair B",
                             "step_D": "PML", "update_E": "ordinary"}
    item = _candidate(record, "fused pair D")
    assert item["applied"] is True and item["rule"] == "record"
    assert item["verdict"]["reason"] == "slower_beyond_spread"
    assert item["verdict"]["evidence"]["displaces_checked"] is True
    assert item["took_the_slots"] == ["PML", "ordinary"]
    assert record["preference"]["withheld_by_table"] == {"triton": ["fused pair D"]}
    assert record["preference"]["compositions"] == 2
    assert "fused pair D" in composer.offers[0]
    assert "fused pair D" not in composer.offers[1]
    assert "fused pair B" in composer.offers[1]
    assert record["preference"]["records"]["triton"]["status"] == "read"
    assert record["preference"]["records"]["triton"]["path"] == os.path.join(
        "triton_kernels", "timing_cc86.json")
    assert record["preference"]["note"].startswith("meep_gpu: NOTE dispatch preference:")


def test_a_keep_verdict_keeps_the_fused_product(monkeypatch, tmp_path):
    _write(tmp_path, [_keep_d(lo=24000, hi=2400000), _clean_b()])
    composer = Composer()
    _plan, record = _freeze(monkeypatch, composer)
    assert _arms(record)["step_D"] == "fused pair D"
    assert _arms(record)["step_B"] == "fused pair B"
    item = _candidate(record, "fused pair D")
    assert item["applied"] is False
    assert item["why_not_applied"] == "not_slower_beyond_spread"
    assert record["preference"]["compositions"] == 1 and len(composer.offers) == 1
    assert record["preference"]["note"] is None


def test_an_unmeasured_key_falls_to_the_portable_prior(monkeypatch, tmp_path, capsys):
    """A record SILENT on a product leaves the architecture where an unrecorded one is,
    so the prior answers, and both verdicts are kept on the candidate."""
    _write(tmp_path, [_slow_d(status="unmeasured", reason="1 row, minimum 2"),
                      _slow_d(product="fused pair D (other)"),
                      _keep_d(product="fused pair D (other)")])
    _plan, record = _freeze(monkeypatch, Composer())
    assert _arms(record)["step_D"] == "PML"
    item = _candidate(record, "fused pair D")
    assert item["applied"] is True and item["rule"] == "portable_prior"
    assert item["record_silence"] == "key_unmeasured"
    assert item["record_verdict"]["reason"] == "key_unmeasured"
    assert item["verdict"]["reason"] == (
        "prior_bracketed_seam_at_or_below_the_recorded_crossover")
    assert record["preference"]["rule_by_table"] == {"triton": "record"}
    assert record["preference"]["prior_fills_record_silence"] is True
    assert "does not price it here (key_unmeasured)" in record["preference"]["note"]
    assert "cells on 8.6, this device's own architecture (triton 8.6)" in (
        record["preference"]["note"])
    evidence = item["verdict"]["evidence"]
    assert (evidence["prior_architecture"], evidence["basis"], evidence["distance"]) == (
        "8.6", dp.OWN_ARCHITECTURE, 0)
    assert record["preference"]["note"] in capsys.readouterr().err


def test_with_the_silence_switch_off_an_unmeasured_key_keeps_the_shape_rule(
        monkeypatch, tmp_path):
    """'No record, so the shape rule', one constant away."""
    monkeypatch.setattr(fastpath, "PRIOR_FILLS_RECORD_SILENCE", False)
    _write(tmp_path, [_slow_d(status="unmeasured", reason="1 row, minimum 2")])
    _plan, record = _freeze(monkeypatch, Composer())
    assert _arms(record)["step_D"] == "fused pair D"
    item = _candidate(record, "fused pair D")
    assert item["why_not_applied"] == "key_unmeasured" and item["rule"] == "record"
    assert record["preference"]["prior"] is None


def test_a_size_outside_the_measured_bands_falls_to_the_prior(monkeypatch, tmp_path):
    """Between the recorded veto and keep bands the prior has no claim either."""
    _write(tmp_path, [_slow_d(), _keep_d()])
    _plan, record = _freeze(monkeypatch, Composer(), shape=(2000, 1500, 1))
    assert _arms(record)["step_D"] == "fused pair D"
    item = _candidate(record, "fused pair D")
    assert item["record_silence"] == "outside_the_measured_bands"
    assert item["why_not_applied"] == "prior_above_the_recorded_crossover"


def test_a_clean_seam_the_record_does_not_price_is_kept(monkeypatch, tmp_path):
    _write(tmp_path, [_slow_d(), _keep_d()])
    _plan, record = _freeze(monkeypatch, Composer(bracketed=()))
    assert _arms(record)["step_D"] == "fused pair D"
    item = _candidate(record, "fused pair D")
    assert item["record_silence"] == "class_not_in_record"
    assert item["why_not_applied"] == "prior_says_nothing_about_a_clean_seam"


def test_a_measured_keep_wins_over_the_prior(monkeypatch, tmp_path):
    """The prior fills silence only: a key that measured keep is never second-guessed,
    even where the prior (from another record) would veto."""
    _write(tmp_path, [_keep_d(lo=24000, hi=2400000)], capability="8.6")
    _write(tmp_path, [_slow_d(), _keep_d()], capability="8.0")
    _plan, record = _freeze(monkeypatch, Composer())
    assert _arms(record)["step_D"] == "fused pair D"
    item = _candidate(record, "fused pair D")
    assert item["rule"] == "record" and "record_silence" not in item
    assert "record_verdict" not in item
    assert item["why_not_applied"] == "not_slower_beyond_spread"


def test_a_record_that_cannot_be_applied_is_not_silence(monkeypatch, tmp_path):
    """Another route is staleness, not silence: the shape rule, and no prior."""
    _write(tmp_path, [_slow_d(), _keep_d()])
    _plan, record = _freeze(monkeypatch, Composer(),
                            route={"repairs_per_bracketed_step": 3.0, "restricted": False})
    item = _candidate(record, "fused pair D")
    assert item["why_not_applied"] == "record_is_for_another_repair_route"
    assert "record_silence" not in item and "record_verdict" not in item
    assert item["rule"] == "record"


# ---------------------------------------------------------------------------
# The no-record rule: the portable prior
# ---------------------------------------------------------------------------


def test_an_unrecorded_architecture_gets_the_portable_prior_and_its_note(
        monkeypatch, tmp_path, capsys):
    _write(tmp_path, [_slow_d(), _keep_d(), _clean_b()])
    composer = Composer(bracketed=("D",))
    plan, record = _freeze(monkeypatch, composer, capability=UNRECORDED)
    assert plan is not None, record["refused_because"]
    assert _arms(record) == {"step_B": "fused pair B", "update_H": "fused pair B",
                             "step_D": "PML", "update_E": "ordinary"}
    preference = record["preference"]
    assert preference["architecture"] == "8.9"
    assert preference["rule_by_table"] == {"triton": "portable_prior"}
    assert preference["records"]["triton"]["status"] == "absent"
    assert preference["records"]["triton"]["path"] == os.path.join(
        "triton_kernels", "timing_cc89.json")
    nvidia = preference["prior"]["by_vendor"]["nvidia"]
    assert sorted(nvidia["by_architecture"]) == ["8.6"]
    assert nvidia["by_architecture"]["8.6"]["bracketed_veto_up_to_cells"] == 2400000
    assert nvidia["by_architecture"]["8.6"]["lowest_bracketed_keep_cells"] == 5400000
    assert "bracketed_veto_up_to_cells" not in preference["prior"]
    assert "bracketed_veto_up_to_cells" not in nvidia
    item = _candidate(record, "fused pair D")
    assert item["applied"] is True and item["rule"] == "portable_prior"
    assert item["verdict"]["reason"] == (
        "prior_bracketed_seam_at_or_below_the_recorded_crossover")
    evidence = item["verdict"]["evidence"]
    assert evidence["rule"] == dp.PRIOR_RULE
    assert evidence["vendor"] == "nvidia"
    assert (evidence["live_architecture"], evidence["prior_architecture"],
            evidence["basis"], evidence["distance"]) == (
        "8.9", "8.6", "nearest compute capability", 3)
    assert item["bracketed"] is True
    clean = _candidate(record, "fused pair B")
    assert clean["bracketed"] is False
    assert clean["applied"] is False
    assert clean["why_not_applied"] == "prior_says_nothing_about_a_clean_seam"
    note = preference["note"]
    assert "portable prior" in note and "timing_cc89" not in note
    assert "no triton timing record for 8.9" in note
    assert "2,400,000 cells" in note and "triton 8.6" in note
    assert "on 8.6, the nearest compute capability to 8.9 timed" in note
    assert "every recorded architecture" not in note
    assert note in capsys.readouterr().err


def test_the_prior_keeps_a_grid_above_the_recorded_crossover(monkeypatch, tmp_path):
    _write(tmp_path, [_slow_d(), _keep_d()])
    _plan, record = _freeze(monkeypatch, Composer(), capability=UNRECORDED,
                            shape=(2000, 1500, 1))
    assert _arms(record)["step_D"] == "fused pair D"
    assert (_candidate(record, "fused pair D")["why_not_applied"]
            == "prior_above_the_recorded_crossover")


def test_the_prior_extends_below_the_smallest_measured_size(monkeypatch, tmp_path):
    """The bracket is a fixed per-step host cost, so a smaller grid only loses more."""
    _write(tmp_path, [_slow_d(), _keep_d()])
    _plan, record = _freeze(monkeypatch, Composer(), capability=UNRECORDED,
                            shape=(40, 30, 1))
    assert _arms(record)["step_D"] == "PML"
    assert _candidate(record, "fused pair D")["cells"] == 1200


def test_with_no_usable_record_anywhere_the_prior_vetoes_nothing_and_says_why(
        monkeypatch, tmp_path):
    _plan, record = _freeze(monkeypatch, Composer(), capability=UNRECORDED)
    assert _arms(record)["step_D"] == "fused pair D"
    by_vendor = record["preference"]["prior"]["by_vendor"]
    assert {vendor: block["why_none"] for vendor, block in by_vendor.items()} == {
        "apple": "no shipped apple record holds on this tree and measures a key",
        "nvidia": "no shipped nvidia record holds on this tree and measures a key"}
    item = _candidate(record, "fused pair D")
    assert item["why_not_applied"] == "no_prior"
    assert item["verdict"]["evidence"]["why_none"] == (
        "no shipped nvidia record holds on this tree and measures a key")


def test_a_record_cut_on_another_repair_route_prices_no_prior(monkeypatch, tmp_path):
    """A record its own architecture would answer ``no_record`` for prices no other one."""
    _write(tmp_path, [_slow_d(), _keep_d()], route={"deposit_repair_sha256": "e" * 64})
    _plan, record = _freeze(monkeypatch, Composer(), capability=UNRECORDED)
    assert _arms(record)["step_D"] == "fused pair D"
    prior = record["preference"]["prior"]
    assert prior["by_vendor"]["nvidia"]["by_architecture"] == {}
    assert "deposit-repair route" in prior["not_used"][0]["why"]


def test_a_record_whose_emitters_moved_prices_no_prior(monkeypatch, tmp_path):
    _write(tmp_path, [_slow_d(), _keep_d()])
    path = pathlib.Path(dp.record_path("triton", RECORDED, root=str(_root(tmp_path))))
    record = json.loads(path.read_text())
    record["subject_barrier"]["digests"] = {name: ["0" * 64]
                                            for name in record["subject_barrier"]["digests"]}
    path.write_text(dp.dumps(dp.seal(record)))
    _plan, frozen = _freeze(monkeypatch, Composer(), capability=UNRECORDED)
    assert _arms(frozen)["step_D"] == "fused pair D"
    assert "emitters" in frozen["preference"]["prior"]["not_used"][0]["why"]


def test_an_nvidia_table_takes_no_prior_from_an_apple_record(monkeypatch, tmp_path):
    """The prior is per vendor: an Apple record that vetoes at this size prices nothing on
    an NVIDIA architecture with no record of its own, and the shape rule stands."""
    _write(tmp_path, [_slow_d(), _keep_d()], table="metal", capability="applegpu_g13s")
    _plan, record = _freeze(monkeypatch, Composer(), capability=UNRECORDED)
    assert _arms(record)["step_D"] == "fused pair D"
    item = _candidate(record, "fused pair D")
    assert item["rule"] == "portable_prior" and item["why_not_applied"] == "no_prior"
    evidence = item["verdict"]["evidence"]
    assert evidence["vendor"] == "nvidia" and evidence["derived_from"] == []
    assert evidence["why_none"] == (
        "no shipped nvidia record holds on this tree and measures a key")
    prior = record["preference"]["prior"]
    assert prior["by_vendor"]["apple"]["by_architecture"]["applegpu_g13s"][
        "bracketed_veto_up_to_cells"] == 2400000
    assert record["preference"]["note"] is None


def test_the_ceiling_is_the_highest_veto_below_every_keep():
    """A veto band ABOVE a keep says nothing about the sizes between them."""
    record = _record([_slow_d(), _keep_d(),
                      _slow_d(lo=7000000, hi=7000000)], route=dp.repair_route_sha256(),
                     barrier=_barrier(), index="linear", repairs_per_bracketed_step=1.0,
                     restricted=True, fell_back=False)
    prior = dp.portable_prior([("a.json", record)],
                              repair_route_sha256=dp.repair_route_sha256(),
                              subject_sha256_by_table={"triton": dp.live_subject_sha256(
                                  "triton")})
    assert prior["by_vendor"]["nvidia"]["by_architecture"]["8.6"][
        "bracketed_veto_up_to_cells"] == 2400000
    assert prior["derived_from"][0]["veto_bands"] == [[24000, 2400000],
                                                      [7000000, 7000000]]


def test_the_prior_in_span_mode_is_not_consulted():
    """Span mode returns before it reads the prior at all, whatever it was handed."""
    verdict = dp.consult_prior({"bracketed_veto_up_to_cells": 10 ** 9}, table="triton",
                               architecture=RECORDED, candidate="fused pair D",
                               bracketed=True, cells=10, mode="span")
    assert verdict.outcome == "no_record"
    assert verdict.reason == "span_mode_record_not_consulted"
    with pytest.raises(dp.ConsultRefused):
        dp.consult_prior(None, table="triton", architecture=RECORDED, candidate="x",
                         bracketed=True, cells=1, mode="fastest")


#: An Apple GPU family no record below names, and the one the Apple records are cut on.
APPLE_RECORDED = "applegpu_g13s"
APPLE_UNRECORDED = "applegpu_g14s"


def _two_vendor_records():
    """An NVIDIA record whose bracketed crossover is 2,400,000 cells (keep 5,400,000) and
    an Apple record whose crossover is 1,014,000 (keep 2,400,000). Pooled across vendors,
    the Apple keep would pull the NVIDIA ceiling down to 1,014,000."""
    facts = dict(index="linear", repairs_per_bracketed_step=1.0, restricted=True,
                 fell_back=False)
    nvidia = _record([_slow_d(), _keep_d()], table="triton", route=dp.repair_route_sha256(),
                     barrier=_barrier("triton"),
                     host=[{"device": "NVIDIA STAND-IN", "compute_capability": RECORDED}],
                     **facts)
    apple = _record([_slow_d(hi=1014000), _keep_d(lo=2400000, hi=2400000)], table="metal",
                    route=dp.repair_route_sha256(), barrier=_barrier("metal"),
                    host=[{"device": "APPLE STAND-IN", "compute_capability": APPLE_RECORDED}],
                    **facts)
    return nvidia, apple


def _live_subjects():
    return {table: dp.live_subject_sha256(table) for table in dp.TABLES}


def test_the_prior_is_derived_per_vendor_and_never_across_vendors():
    nvidia, apple = _two_vendor_records()
    prior = dp.portable_prior([("nvidia.json", nvidia), ("apple.json", apple)],
                              repair_route_sha256=dp.repair_route_sha256(),
                              subject_sha256_by_table=_live_subjects())
    assert sorted(prior["by_vendor"]) == list(dp.VENDORS) == ["apple", "nvidia"]
    assert "bracketed_veto_up_to_cells" not in prior
    assert "lowest_bracketed_keep_cells" not in prior
    by_vendor = prior["by_vendor"]
    nvidia_86 = by_vendor["nvidia"]["by_architecture"][RECORDED]
    apple_13 = by_vendor["apple"]["by_architecture"][APPLE_RECORDED]
    assert (nvidia_86["bracketed_veto_up_to_cells"],
            nvidia_86["lowest_bracketed_keep_cells"]) == (2400000, 5400000)
    assert (apple_13["bracketed_veto_up_to_cells"],
            apple_13["lowest_bracketed_keep_cells"]) == (1014000, 2400000)
    assert sorted(by_vendor["nvidia"]["by_architecture"]) == [RECORDED]
    assert sorted(by_vendor["apple"]["by_architecture"]) == [APPLE_RECORDED]
    assert by_vendor["nvidia"]["tables"] == ["cuda", "triton"]
    assert by_vendor["apple"]["tables"] == ["metal"]
    assert [item["path"] for item in nvidia_86["derived_from"]] == ["nvidia.json"]
    assert [item["path"] for item in apple_13["derived_from"]] == ["apple.json"]
    assert [(item["path"], item["vendor"]) for item in prior["derived_from"]] == [
        ("nvidia.json", "nvidia"), ("apple.json", "apple")]
    # 1,200,000 cells lies above the Apple crossover and below the NVIDIA one.
    for table, vendor, outcome, ceiling, used, live in (
            ("triton", "nvidia", "vetoed", 2400000, ["nvidia.json"], RECORDED),
            ("cuda", "nvidia", "vetoed", 2400000, ["nvidia.json"], RECORDED),
            ("metal", "apple", "no_record", 1014000, ["apple.json"], APPLE_RECORDED)):
        verdict = dp.consult_prior(prior, table=table, architecture=live,
                                   candidate="a fused pair", bracketed=True,
                                   cells=1200000, mode="measured")
        assert verdict.outcome == outcome, (table, verdict.as_dict())
        assert verdict.evidence["vendor"] == vendor
        assert verdict.evidence["bracketed_veto_up_to_cells"] == ceiling
        assert [item["path"] for item in verdict.evidence["derived_from"]] == used


def test_a_vendor_with_no_record_that_holds_has_no_prior():
    nvidia, _apple = _two_vendor_records()
    prior = dp.portable_prior([("nvidia.json", nvidia)],
                              repair_route_sha256=dp.repair_route_sha256(),
                              subject_sha256_by_table=_live_subjects())
    assert prior["by_vendor"]["apple"] == {
        "tables": ["metal"], "by_architecture": {},
        "why_none": "no shipped apple record holds on this tree and measures a key"}
    verdict = dp.consult_prior(prior, table="metal", architecture=APPLE_RECORDED,
                               candidate="a fused pair", bracketed=True, cells=1000,
                               mode="measured")
    assert (verdict.outcome, verdict.reason) == ("no_record", "no_prior")
    assert verdict.evidence["vendor"] == "apple" and verdict.evidence["derived_from"] == []
    assert verdict.evidence["why_none"] == (
        "no shipped apple record holds on this tree and measures a key")


def test_a_prior_that_names_no_vendor_or_a_table_with_none_is_refused_by_name():
    with pytest.raises(dp.ConsultRefused) as caught:
        dp.consult_prior({"bracketed_veto_up_to_cells": 10 ** 9}, table="metal",
                         architecture=APPLE_RECORDED, candidate="x", bracketed=True,
                         cells=10, mode="measured")
    assert caught.value.code == "unreadable_prior"
    with pytest.raises(dp.ConsultRefused) as caught:
        dp.consult_prior({"by_vendor": {"apple": {"bracketed_veto_up_to_cells": 10 ** 9}}},
                         table="metal", architecture=APPLE_RECORDED, candidate="x",
                         bracketed=True, cells=10, mode="measured")
    assert caught.value.code == "unreadable_prior"
    nvidia, _apple = _two_vendor_records()
    prior = dp.portable_prior([("nvidia.json", nvidia)],
                              repair_route_sha256=dp.repair_route_sha256(),
                              subject_sha256_by_table=_live_subjects())
    with pytest.raises(dp.ConsultRefused) as caught:
        dp.consult_prior(prior, table="opencl", architecture=RECORDED, candidate="x",
                         bracketed=True, cells=10, mode="measured")
    assert caught.value.code == "unknown_table"


def test_the_bracket_is_found_the_way_the_bench_finds_it():
    inner = CountingPlan("fused")
    leading = deposit_repair.LeadingRepairPlan(inner, None, None, (), "D")
    trailing = deposit_repair.TrailingRepairPlan("update_E", leading, None, None)
    wrapper = types.SimpleNamespace(inner=leading)
    assert fastpath._slot_bracket(leading) is leading
    assert fastpath._slot_bracket(trailing) is leading
    assert fastpath._slot_bracket(wrapper) is leading
    assert fastpath._slot_bracket(inner) is None
    assert fastpath._slot_bracket(launch_module.NoopPlan("update_H", inner)) is None


def test_an_unreadable_device_keeps_the_shape_rule_and_says_why(monkeypatch, tmp_path):
    """Which timed architecture is nearest a device whose architecture was not read is
    not known, so it is not guessed: the table answers no prior and the shape rule
    stands, with the reason named."""
    _write(tmp_path, [_slow_d(), _keep_d()])
    stub_triton(monkeypatch)
    block_cuda(monkeypatch)
    import meep_gpu.triton_kernels as package

    monkeypatch.setattr(package, "plan_step", Composer())
    monkeypatch.setattr(fastpath, "_planned_repair_route",
                        lambda leading, digest: _route_facts())
    grid = certified_envelope_grid()          # no cuda.runtime: the device is unread
    fastpath.plan_fast_path(certified_envelope_fields(), certified_envelope_pml(), grid,
                            sources=())
    record = fastpath.last_dispatch_report()
    assert record["preference"]["records"]["triton"]["status"] == "architecture_unread"
    item = _candidate(record, "fused pair D")
    assert item["rule"] == "portable_prior"
    assert item["why_not_applied"] == "live_architecture_unparsed"
    evidence = item["verdict"]["evidence"]
    assert evidence["live_architecture"] is None and evidence["prior_architecture"] is None
    assert "not known" in evidence["why_none"]
    assert _arms(record)["step_D"] == "fused pair D"


# ---------------------------------------------------------------------------
# One record per architecture
# ---------------------------------------------------------------------------


def test_the_record_read_is_the_live_architectures_own(monkeypatch, tmp_path):
    """A record named for 8.9 that says KEEP decides an 8.9 device; the 8.6 record that
    says VETO decides an 8.6 device and is only a prior source for the 8.9 one."""
    _write(tmp_path, [_slow_d(), _keep_d()], capability="8.6")
    _write(tmp_path, [_keep_d(lo=24000, hi=2400000)], capability="8.9")
    _plan, on_86 = _freeze(monkeypatch, Composer(), capability=(8, 6))
    assert _arms(on_86)["step_D"] == "PML"
    assert _candidate(on_86, "fused pair D")["rule"] == "record"
    fastpath.reset_dispatch_announcements()
    _plan, on_89 = _freeze(monkeypatch, Composer(), capability=(8, 9))
    assert _arms(on_89)["step_D"] == "fused pair D"
    item = _candidate(on_89, "fused pair D")
    assert item["rule"] == "record" and item["why_not_applied"] == "not_slower_beyond_spread"
    assert on_89["preference"]["records"]["triton"]["path"] == os.path.join(
        "triton_kernels", "timing_cc89.json")


# ---------------------------------------------------------------------------
# The nearest timed architecture: a device of an architecture with no record
# ---------------------------------------------------------------------------

#: A grid between the two crossovers the records below measure (1,200,000 cells).
BETWEEN = (1200, 1000, 1)


def _two_nvidia_architectures(tmp_path):
    """cc 8.6 records a bracketed crossover of 2,400,000 cells (keep at 5,400,000) and
    cc 9.0 one of 1,014,000 (keep at 2,400,000). Pooled, the 9.0 keep would pull the
    8.6 ceiling down to 1,014,000; read nearest, each device takes one of them."""
    _write(tmp_path, [_slow_d(), _keep_d()], capability="8.6", device="CARD 86")
    _write(tmp_path, [_slow_d(hi=1014000), _keep_d(lo=2400000, hi=2400000)],
           capability="9.0", device="CARD 90")


@pytest.mark.parametrize("live, prior_from, distance, step_d", [
    ((8, 7), "8.6", 1, "PML"),                 # 87 is 1 from 86 and 3 from 90
    ((8, 8), "8.6", 2, "PML"),                 # 2 from each: the lower capability wins
    ((8, 9), "9.0", 1, "fused pair D"),        # 1,200,000 is above 9.0's crossover
    ((7, 5), "8.6", 11, "PML"),
    ((12, 0), "9.0", 30, "fused pair D"),
])
def test_a_device_with_no_record_reads_its_nearest_timed_architecture(
        monkeypatch, tmp_path, live, prior_from, distance, step_d):
    _two_nvidia_architectures(tmp_path)
    _plan, record = _freeze(monkeypatch, Composer(), capability=live, shape=BETWEEN)
    assert record["preference"]["rule_by_table"] == {"triton": "portable_prior"}
    assert _arms(record)["step_D"] == step_d
    evidence = _candidate(record, "fused pair D")["verdict"]["evidence"]
    assert evidence["live_architecture"] == f"{live[0]}.{live[1]}"
    assert evidence["prior_architecture"] == prior_from
    assert evidence["basis"] == "nearest compute capability"
    assert evidence["distance"] == distance
    assert {source["architecture"][0] for source in evidence["derived_from"]} == {prior_from}
    assert evidence["bracketed_veto_up_to_cells"] == {"8.6": 2400000,
                                                      "9.0": 1014000}[prior_from]


def test_a_recorded_architecture_reads_its_own_prior_where_its_record_is_silent(
        monkeypatch, tmp_path):
    """A product the 9.0 record does not price is answered by 9.0's own prior, not by the
    two architectures pooled and not by 8.6's: the bracketed pair keeps its slots at
    1,200,000 cells, which 8.6's crossover would have vetoed."""
    _two_nvidia_architectures(tmp_path)
    _write(tmp_path, [_slow_d(product="fused pair D (other)", hi=1014000),
                      _keep_d(product="fused pair D (other)", lo=2400000, hi=2400000)],
           capability="9.0", device="CARD 90")
    _plan, record = _freeze(monkeypatch, Composer(), capability=(9, 0), shape=BETWEEN)
    item = _candidate(record, "fused pair D")
    assert item["record_silence"] == "class_not_in_record"
    assert item["why_not_applied"] == "prior_above_the_recorded_crossover"
    evidence = item["verdict"]["evidence"]
    assert (evidence["prior_architecture"], evidence["basis"], evidence["distance"]) == (
        "9.0", dp.OWN_ARCHITECTURE, 0)
    assert _arms(record)["step_D"] == "fused pair D"


def test_the_nearest_architecture_without_a_ceiling_is_the_answer(monkeypatch, tmp_path):
    """The nearest timed architecture measured its bracketed product a keep at every size:
    a farther architecture's veto is not asked instead."""
    _write(tmp_path, [_slow_d(), _keep_d()], capability="8.6", device="CARD 86")
    _write(tmp_path, [_keep_d(lo=24000, hi=2400000)], capability="9.0", device="CARD 90")
    _plan, record = _freeze(monkeypatch, Composer(), capability=(8, 9))
    item = _candidate(record, "fused pair D")
    assert item["why_not_applied"] == "no_prior"
    evidence = item["verdict"]["evidence"]
    assert evidence["prior_architecture"] == "9.0"
    assert evidence["why_none"] == (
        "no nvidia record of 9.0 used measures a bracketed veto below its keeps")
    assert _arms(record)["step_D"] == "fused pair D"


def test_an_apple_gpu_reads_the_closest_chip_generation(tmp_path):
    """Two Apple families timed, generations 13 and 15: generation 14 is one from each and
    takes the lower, 16 takes 15, and a family that names no generation takes none."""
    _write(tmp_path, [_slow_d(hi=1014000), _keep_d(lo=2400000, hi=2400000)], table="metal",
           capability="applegpu_g13s", device="APPLE 13")
    _write(tmp_path, [_slow_d(hi=60000), _keep_d(lo=110000, hi=110000)], table="metal",
           capability="applegpu_g15p", device="APPLE 15")
    for live, prior_from, distance in (("applegpu_g14x", "applegpu_g13s", 1),
                                       ("applegpu_g16x", "applegpu_g15p", 1),
                                       ("applegpu_g13g", "applegpu_g13s", 0)):
        record, plan, _chosen, _scratch, _seen = _arbitrate_metal(live)
        evidence = _candidate(record, "fused electric D/E pair")["verdict"]["evidence"]
        assert (evidence["prior_architecture"], evidence["basis"], evidence["distance"]) == (
            prior_from, "closest chip generation", distance), live
    record, plan, chosen, _scratch, _seen = _arbitrate_metal("applegpu_other")
    item = _candidate(record, "fused electric D/E pair")
    assert item["why_not_applied"] == "live_architecture_unparsed"
    assert chosen == METAL_OFFER and plan.selected == METAL_SELECTED


def test_a_timed_spelling_the_rule_cannot_read_refuses_the_prior(tmp_path):
    """An Apple record whose family names no generation cannot be ranked against the
    others, so the table answers no prior, naming it, rather than ranking around it."""
    _write(tmp_path, [_slow_d(), _keep_d()], table="metal", capability="applegpu_g13s",
           device="APPLE 13")
    _write(tmp_path, [_slow_d(), _keep_d()], table="metal", capability="applegpu_test",
           device="APPLE TEST")
    record, plan, chosen, _scratch, _seen = _arbitrate_metal("applegpu_g14x")
    item = _candidate(record, "fused electric D/E pair")
    assert item["why_not_applied"] == "prior_architecture_unparsed"
    assert "applegpu_test" in item["verdict"]["evidence"]["why_none"]
    assert chosen == METAL_OFFER and plan.selected == METAL_SELECTED


# ---------------------------------------------------------------------------
# One record per card: own, representative, absent
# ---------------------------------------------------------------------------

#: The card the records below are cut on, another card of its architecture (a name
#: that differs by a suffix included: a 3090 Ti is not a 3090), and a card of an
#: architecture no record names.
TIMED = "NVIDIA GeForce RTX 3090"
OTHER = "NVIDIA RTX A6000"
SUFFIXED = "NVIDIA GeForce RTX 3090 Ti"
UNTIMED = "NVIDIA GeForce RTX 4080"


def _card_name(card, table="triton", architecture=RECORDED):
    return os.path.basename(dp.record_path(table, architecture, card=card))


def _role(report, tmp_path, table="triton"):
    """Whose record a run read, as the block states it, checked against the record: own
    when the live card (``environment.device.name``) has the slug of the card the record
    the block names was cut on, else representative."""
    entry = report["preference"]["records"][table]
    record = dp.load_record(str(_root(tmp_path) / entry["path"]))
    assert record[dp.DIGEST_FIELD] == entry["record_sha256"]
    live = report["environment"]["device"]["name"]
    assert entry["live_card"] == live
    (cut_on,) = dp.record_devices(record)
    assert entry["record_card"] == cut_on
    derived = "own" if dp.card_slug(live) == dp.card_slug(cut_on) else "representative"
    assert entry["role"] == derived, (entry, derived)
    return entry["role"]


def _verdicts(report):
    return [(item["label"], item["rule"], item["applied"],
             (item.get("verdict") or {}).get("reason"))
            for item in report["preference"]["candidates"]]


def test_a_card_the_record_was_cut_on_reads_it_as_its_own(monkeypatch, tmp_path):
    _write(tmp_path, [_slow_d(), _keep_d(), _clean_b()], device=TIMED)
    _plan, record = _freeze(monkeypatch, Composer(), name=TIMED)
    assert record["environment"]["device"]["name"] == TIMED
    entry = record["preference"]["records"]["triton"]
    assert entry["status"] == "read"
    assert entry["path"] == os.path.join("triton_kernels", "timing_cc86.json")
    assert _arms(record)["step_D"] == "PML"
    assert _candidate(record, "fused pair D")["rule"] == "record"
    assert _role(record, tmp_path) == "own"
    assert "this card's own triton timing record for 8.6" in record["preference"]["note"]


def test_another_card_of_the_architecture_reads_the_representative(monkeypatch, tmp_path):
    """The same record, the same verdicts: only the role differs, the NOTE names the card
    the representative was cut on, and a representative's verdict is no measurement of
    the live card."""
    _write(tmp_path, [_slow_d(), _keep_d(), _clean_b()], device=TIMED)
    _plan, own = _freeze(monkeypatch, Composer(), name=TIMED)
    for card in (OTHER, SUFFIXED):
        fastpath.reset_dispatch_announcements()
        _plan, other = _freeze(monkeypatch, Composer(), name=card)
        assert other["environment"]["device"]["name"] == card
        entries = {table: {name: value for name, value in entry.items()
                           if name not in ("role", "live_card")}
                   for table, entry in other["preference"]["records"].items()}
        assert entries == {table: {name: value for name, value in entry.items()
                                   if name not in ("role", "live_card")}
                           for table, entry in own["preference"]["records"].items()}
        assert other["preference"]["rule_by_table"] == {"triton": "record"}
        assert _verdicts(other) == _verdicts(own)
        assert _arms(other) == _arms(own)
        assert _role(other, tmp_path) == "representative", card
        assert (f"the triton timing record for 8.6, cut on {TIMED} as the architecture's "
                "representative, measured it") in other["preference"]["note"]


def test_an_architecture_with_no_record_reads_the_portable_prior(monkeypatch, tmp_path):
    _write(tmp_path, [_slow_d(), _keep_d()], device=TIMED)
    _plan, record = _freeze(monkeypatch, Composer(), capability=UNRECORDED, name=UNTIMED)
    entry = record["preference"]["records"]["triton"]
    assert entry["status"] == "absent" and "record_sha256" not in entry
    assert "role" not in entry and entry["live_card"] == UNTIMED
    assert entry["path"] == os.path.join("triton_kernels", "timing_cc89.json")
    assert record["preference"]["rule_by_table"] == {"triton": "portable_prior"}
    assert _candidate(record, "fused pair D")["rule"] == "portable_prior"


def test_a_product_the_representative_does_not_price_is_answered_by_the_prior(
        monkeypatch, tmp_path):
    _write(tmp_path, [_slow_d(product="fused pair D (other)"),
                      _keep_d(product="fused pair D (other)")], device=TIMED)
    _plan, record = _freeze(monkeypatch, Composer(), name=OTHER)
    item = _candidate(record, "fused pair D")
    assert item["rule"] == "portable_prior" and item["applied"] is True
    assert item["record_silence"] == "class_not_in_record"
    assert record["preference"]["rule_by_table"] == {"triton": "record"}
    assert _role(record, tmp_path) == "representative"


def test_a_cards_own_record_beside_the_representative_is_read_by_that_card(
        monkeypatch, tmp_path):
    """The live card's own record beside the representative is the one it reads; the
    representative's card and every other card read the representative. Both records
    enter the prior of their architecture."""
    _write(tmp_path, [_clean_b()], device=OTHER)
    _write(tmp_path, [_slow_d(), _keep_d()], device=TIMED, name=_card_name(TIMED))
    own_path = os.path.join("triton_kernels", "timing_cc86-nvidia_geforce_rtx_3090.json")
    bare_path = os.path.join("triton_kernels", "timing_cc86.json")
    _plan, record = _freeze(monkeypatch, Composer(), name=TIMED)
    entry = record["preference"]["records"]["triton"]
    assert entry["path"] == own_path
    assert _role(record, tmp_path) == "own"
    item = _candidate(record, "fused pair D")
    assert item["rule"] == "record" and item["applied"] is True
    assert _arms(record)["step_D"] == "PML"
    for card, role in ((OTHER, "own"), (SUFFIXED, "representative")):
        fastpath.reset_dispatch_announcements()
        _plan, other = _freeze(monkeypatch, Composer(), name=card)
        assert other["preference"]["records"]["triton"]["path"] == bare_path, card
        assert _role(other, tmp_path) == role, card
        item = _candidate(other, "fused pair D")
        assert item["rule"] == "portable_prior"
        assert item["record_silence"] == "class_not_in_record"
        used = sorted(source["path"] for source in item["verdict"]["evidence"]["derived_from"])
        assert used == [own_path, bare_path]
        assert _arms(other)["step_D"] == "PML"


def test_a_cards_own_record_that_answers_a_barrier_is_not_replaced(monkeypatch, tmp_path):
    """A card's own record cut on another deposit-repair route answers that barrier for
    the card, and the representative -- another card's rows, which would veto -- is not
    asked in its place."""
    _write(tmp_path, [_slow_d(), _keep_d()], device=OTHER)
    _write(tmp_path, [_slow_d(), _keep_d()], device=TIMED, name=_card_name(TIMED),
           route={"deposit_repair_sha256": "e" * 64})
    _plan, record = _freeze(monkeypatch, Composer(), name=TIMED)
    entry = record["preference"]["records"]["triton"]
    assert entry["path"].endswith("timing_cc86-nvidia_geforce_rtx_3090.json")
    assert entry["role"] == "own"
    item = _candidate(record, "fused pair D")
    assert item["rule"] == "record"
    assert item["why_not_applied"] == "record_is_for_another_repair_route"
    assert _arms(record)["step_D"] == "fused pair D"


def test_a_card_record_named_for_another_card_is_refused_and_not_replaced(
        monkeypatch, tmp_path, capsys):
    """A file at the live card's own name whose rows ran on another card is refused by
    name; the run arbitrates the table by the prior and does not read the representative
    in its place."""
    _write(tmp_path, [_clean_b()], device=OTHER)
    _write(tmp_path, [_slow_d(), _keep_d()], device=OTHER, name=_card_name(TIMED))
    _plan, record = _freeze(monkeypatch, Composer(), name=TIMED)
    entry = record["preference"]["records"]["triton"]
    assert entry["path"].endswith("timing_cc86-nvidia_geforce_rtx_3090.json")
    assert entry["status"] == "refused" and "role" not in entry
    assert entry["refused_because"].startswith("record_named_for_another_card")
    assert record["preference"]["rule_by_table"] == {"triton": "portable_prior"}
    item = _candidate(record, "fused pair D")
    assert item["rule"] == "portable_prior" and item["why_not_applied"] == "no_prior"
    assert _arms(record)["step_D"] == "fused pair D"
    assert "was refused (record_named_for_another_card" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Certified scope: arbitration only
# ---------------------------------------------------------------------------


def test_certified_scope_arbitrates_as_the_default_and_stays_certified(
        monkeypatch, tmp_path):
    _write(tmp_path, [_slow_d(), _clean_b()])
    monkeypatch.setenv(fastpath.HARDWARE_SWITCH, "certified")
    plan, record = _freeze(monkeypatch, Composer())
    assert plan is not None, record["refused_because"]
    assert _arms(record)["step_D"] == "PML"
    assert record["certified"] is True
    assert record["preference"]["mode"] == "measured"


def test_certified_scope_on_a_device_with_no_record_still_takes_the_prior(
        monkeypatch, tmp_path):
    """The fast-by-default case: a device certified for bit identity that no timing run reached."""
    _write(tmp_path, [_slow_d(), _keep_d()], capability="8.0")
    monkeypatch.setenv(fastpath.HARDWARE_SWITCH, "certified")
    plan, record = _freeze(monkeypatch, Composer(), capability=(8, 6))
    assert plan is not None, record["refused_because"]
    assert record["preference"]["rule_by_table"] == {"triton": "portable_prior"}
    assert _arms(record)["step_D"] == "PML"
    assert record["certified"] is True


# ---------------------------------------------------------------------------
# Fail-safe: a veto that cannot be confirmed, a refused record, the switch
# ---------------------------------------------------------------------------


def test_a_veto_that_would_leave_a_slot_on_the_array_path_is_dropped(monkeypatch, tmp_path):
    _write(tmp_path, [_slow_d()])
    composer = Composer(singles=None, pairs=("D",))
    _plan, record = _freeze(monkeypatch, composer)
    assert _arms(record)["step_D"] == "fused pair D"
    item = _candidate(record, "fused pair D")
    assert item["applied"] is False
    assert item["why_not_applied"].startswith("withholding it left a slot on the array path")
    assert record["preference"]["compositions"] == 2


def test_singles_the_record_did_not_price_drop_the_veto(monkeypatch, tmp_path):
    _write(tmp_path, [_slow_d()])
    _plan, record = _freeze(monkeypatch, Composer(singles=("conductive PML", "ordinary")))
    assert _arms(record)["step_D"] == "fused pair D"
    why = _candidate(record, "fused pair D")["why_not_applied"]
    assert why == ("the record does not price these singles: "
                   "baseline_is_not_what_the_candidate_displaces")


def test_the_prior_drops_a_veto_another_fused_product_would_answer(monkeypatch, tmp_path):
    """Withholding a product must hand its slots to singles, never to another product:
    the clean product that took them was already judged on the shape rule's plan (the
    prior says nothing about a clean seam), so it is kept and the veto is dropped, with
    no composition more."""
    _write(tmp_path, [_slow_d(), _keep_d()])

    class Taker(Composer):
        def __call__(self, fields, pml, **kwargs):
            plan = super().__call__(fields, pml, **kwargs)
            if "fused pair D" not in set(kwargs.get("fuse_labels") or ()):
                fused = CountingPlan("weld")
                plan.plans["step_D"] = fused
                plan.plans["update_E"] = launch_module.NoopPlan("update_E", fused)
                plan.selected["step_D"] = plan.selected["update_E"] = "fused pair B"
            return plan

    _plan, record = _freeze(monkeypatch, Taker(), capability=UNRECORDED)
    item = _candidate(record, "fused pair D")
    assert item["applied"] is False
    assert item["why_not_applied"].startswith(
        "withholding it let another fused product take the slots")
    taker = _candidate(record, "fused pair B")
    assert "taker_of" not in taker and taker["applied"] is False
    assert taker["why_not_applied"] == "prior_says_nothing_about_a_clean_seam"
    assert "withheld_with" not in item
    assert record["preference"]["compositions"] == 2
    assert _arms(record)["step_D"] == "fused pair D"


def test_a_dropped_veto_recomposes_with_the_vetoes_that_stand(monkeypatch, tmp_path):
    """Two prior vetoes; withholding D would leave the array path, so D is dropped and
    the step composes a THIRD time with B alone withheld."""
    _write(tmp_path, [_slow_d(), _keep_d()])

    class OnlyBSingles(Composer):
        def __call__(self, fields, pml, **kwargs):
            plan = super().__call__(fields, pml, **kwargs)
            if "fused pair D" not in set(kwargs.get("fuse_labels") or ()):
                for slot in ("step_D", "update_E"):
                    plan.plans.pop(slot, None)
                    plan.selected.pop(slot, None)
            return plan

    composer = OnlyBSingles(bracketed=("B", "D"))
    _plan, record = _freeze(monkeypatch, composer, capability=UNRECORDED)
    assert _arms(record) == {"step_B": "PML", "update_H": "ordinary",
                             "step_D": "fused pair D", "update_E": "fused pair D"}
    assert record["preference"]["compositions"] == 3
    assert record["preference"]["withheld_by_table"] == {"triton": ["fused pair B"]}
    assert _candidate(record, "fused pair B")["applied"] is True
    assert _candidate(record, "fused pair D")["applied"] is False
    assert composer.offers[2] == sorted(set(composer.offers[0]) - {"fused pair B"})


def test_an_explicit_opt_in_is_not_arbitrated(monkeypatch, tmp_path):
    _write(tmp_path, [_slow_d()])
    monkeypatch.setenv(fastpath.FUSE_ARMS_SWITCH, "fused pair D")
    _plan, record = _freeze(monkeypatch, Composer())
    assert _arms(record)["step_D"] == "fused pair D"
    assert "explicit opt-in" in _candidate(record, "fused pair D")["why_not_applied"]


def test_a_refused_record_is_loud_and_its_table_falls_to_the_prior(
        monkeypatch, tmp_path, capsys):
    path = _write(tmp_path, [_slow_d()], capability="8.6")
    tampered = json.loads(path.read_text())
    tampered["recorded_utc"] = "2026-01-01T00:00:00Z"        # edited after the cut
    path.write_text(dp.dumps(tampered))
    _write(tmp_path, [_slow_d(), _keep_d()], capability="8.0")
    _plan, record = _freeze(monkeypatch, Composer())
    entry = record["preference"]["records"]["triton"]
    assert entry["status"] == "refused"
    assert "record_digest_mismatch" in entry["refused_because"]
    assert record["preference"]["rule_by_table"] == {"triton": "portable_prior"}
    assert _arms(record)["step_D"] == "PML"
    assert "timing record triton_kernels/timing_cc86.json was refused" in (
        capsys.readouterr().err)


def test_a_predicted_route_the_record_was_not_cut_on_keeps_the_shape_rule(
        monkeypatch, tmp_path):
    _write(tmp_path, [_slow_d()])
    _plan, record = _freeze(monkeypatch, Composer(),
                            route={"repairs_per_bracketed_step": 3.0,
                                   "restricted": False})
    assert _arms(record)["step_D"] == "fused pair D"
    assert (_candidate(record, "fused pair D")["why_not_applied"]
            == "record_is_for_another_repair_route")


def test_span_mode_opens_no_record_and_composes_once(monkeypatch, tmp_path):
    _write(tmp_path, [_slow_d()])
    monkeypatch.setenv(fastpath.PREFERENCE_SWITCH, "span")

    def never(*_args, **_kwargs):
        raise AssertionError("span mode opened a record")

    monkeypatch.setattr(dp, "load_record", never)
    composer = Composer()
    _plan, record = _freeze(monkeypatch, composer)
    assert _arms(record)["step_D"] == "fused pair D"
    assert record["preference"]["mode"] == "span"
    assert record["preference"]["candidates"] == []
    assert len(composer.offers) == 1


def test_measured_is_the_default_and_its_spelling_is_accepted(monkeypatch, tmp_path):
    _write(tmp_path, [_slow_d()])
    monkeypatch.setenv(fastpath.PREFERENCE_SWITCH, "measured")
    _plan, record = _freeze(monkeypatch, Composer())
    assert record["preference"]["mode"] == "measured"
    assert _arms(record)["step_D"] == "PML"
    assert fastpath.PREFERENCE_DEFAULT == "measured"


def test_an_unreadable_preference_value_is_refused_by_name(monkeypatch, tmp_path):
    monkeypatch.setenv(fastpath.PREFERENCE_SWITCH, "fastest")
    composer = Composer()
    plan, record = _freeze(monkeypatch, composer)
    assert plan is None
    assert record["refused_because"].startswith("MEEP_GPU_PREFER='fastest' is not one of")
    assert composer.offers == []


def test_every_record_carries_the_preference_block():
    record = fastpath._base_record()
    assert record["preference"]["variable"] == fastpath.PREFERENCE_SWITCH
    assert "not_reached" in record["preference"]


# ---------------------------------------------------------------------------
# The arbiter alone: per-table independence, CUDA namespacing, the Metal table
# ---------------------------------------------------------------------------


class _Plan(types.SimpleNamespace):
    pass


def _merged(selected, bracketed_slots=()):
    plans = {}
    for slot in selected:
        inner = CountingPlan(slot)
        plans[slot] = (deposit_repair.LeadingRepairPlan(inner, None, None, (), "D")
                       if slot in bracketed_slots else inner)
    return _Plan(plans=plans, selected=dict(selected))


def _arbitrate(selected, singles, *, offer, architecture=RECORDED, table_of=None,
               is_fused=None, bracketed_slots=("step_D", "update_E"), opted_in=(),
               grid_shape=(200, 120, 1)):
    record = {"run_shape": {"dimensions": 2, "grid_shape": grid_shape,
                            "complex_storage": False, "susceptibilities": 0}}
    seen = []

    def compose(trial):
        seen.append({name: tuple(labels) for name, labels in trial.items()})
        merged = dict(selected)
        withheld = {label for name in offer for label in offer[name]
                    if label not in trial[name]}
        for slot, label in selected.items():
            if label in withheld:
                merged[slot] = singles[slot]
        return _merged(merged), "scratch"

    plan, chosen, scratch = fastpath._apply_preference(
        record, mode="measured", step_plan=_merged(selected, bracketed_slots),
        admitted_by_table=offer, compose=compose,
        table_of=table_of or (lambda label: "cuda" if str(label).startswith("cuda:")
                              else "triton"),
        is_fused=is_fused or fastpath.arm_is_fused, architecture=architecture,
        opted_in=opted_in)
    return record, plan, chosen, scratch, seen


def test_each_table_is_arbitrated_by_its_own_record(monkeypatch, tmp_path):
    """A Triton product vetoed by the Triton record, a CUDA product kept by the CUDA one."""
    monkeypatch.setattr(fastpath, "_planned_repair_route",
                        lambda leading, digest: _route_facts())
    _write(tmp_path, [_slow_d()], table="triton")
    _write(tmp_path, [_entry(product="cuda:fused magnetic pair", bracketed=False,
                             lo=24000, hi=2400000, fused=0.50, baseline=0.55,
                             displaces=("cuda:PML", "cuda:ordinary"),
                             more_rows=((0.50, 0.56),))], table="cuda")
    selected = {"step_B": "cuda:fused magnetic pair", "update_H": "cuda:fused magnetic pair",
                "step_D": "fused pair D", "update_E": "fused pair D"}
    singles = {"step_B": "cuda:PML", "update_H": "cuda:ordinary",
               "step_D": "PML", "update_E": "ordinary"}
    offer = {"triton": ("fused pair D",), "cuda": ("cuda:fused magnetic pair",)}
    record, plan, chosen, scratch, seen = _arbitrate(selected, singles, offer=offer)
    assert chosen == {"triton": (), "cuda": ("cuda:fused magnetic pair",)}
    assert scratch == "scratch"
    assert plan.selected["step_B"] == "cuda:fused magnetic pair"
    assert plan.selected["step_D"] == "PML"
    preference = record["preference"]
    assert preference["rule_by_table"] == {"cuda": "record", "triton": "record"}
    assert preference["records"]["cuda"]["path"] == os.path.join("cuda_kernels",
                                                                 "timing_cc86.json")


def test_a_cuda_veto_is_keyed_and_confirmed_in_the_namespaced_spelling(monkeypatch,
                                                                         tmp_path):
    monkeypatch.setattr(fastpath, "_planned_repair_route",
                        lambda leading, digest: _route_facts())
    _write(tmp_path, [_entry(product="cuda:fused electric pair", bracketed=True,
                             lo=24000, hi=2400000, fused=0.81, baseline=0.51,
                             displaces=("cuda:PML", "cuda:ordinary"),
                             more_rows=((0.80, 0.50),))], table="cuda")
    selected = {"step_D": "cuda:fused electric pair",
                "update_E": "cuda:fused electric pair"}
    singles = {"step_D": "cuda:PML", "update_E": "cuda:ordinary"}
    record, plan, chosen, _scratch, _seen = _arbitrate(
        selected, singles, offer={"cuda": ("cuda:fused electric pair",)})
    assert chosen == {"cuda": ()}
    assert plan.selected == singles
    item = _candidate(record, "cuda:fused electric pair")
    assert item["applied"] and item["verdict"]["key"].startswith("cuda:fused electric pair|")


def _cascade(selected, chain, singles, *, architecture=RECORDED,
             bracketed_slots=("step_B", "update_H", "step_D", "update_E")):
    """Two tables, one seam each way: a slot goes to the first label of ``chain[slot]``
    still offered, else to its single, as the CUDA-first route composes a seam whose
    CUDA product was withheld (the Triton product of the same seam takes it)."""
    offer: dict = {}
    for labels in chain.values():
        for label in labels:
            table = "cuda" if label.startswith("cuda:") else "triton"
            offer.setdefault(table, ())
            if label not in offer[table]:
                offer[table] += (label,)
    record = {"run_shape": {"dimensions": 2, "grid_shape": (200, 120, 1),
                            "complex_storage": False, "susceptibilities": 0}}
    seen = []

    def compose(trial):
        seen.append({name: tuple(labels) for name, labels in trial.items()})
        offered = {label for labels in trial.values() for label in labels}
        merged = {slot: next((label for label in chain[slot] if label in offered),
                             singles[slot]) for slot in selected}
        return _merged(merged, bracketed_slots), "scratch"

    plan, chosen, _scratch = fastpath._apply_preference(
        record, mode="measured", step_plan=_merged(selected, bracketed_slots),
        admitted_by_table=offer, compose=compose,
        table_of=lambda label: "cuda" if str(label).startswith("cuda:") else "triton",
        is_fused=fastpath.arm_is_fused, architecture=architecture, opted_in=())
    return record, plan, chosen, seen


CUDA_D = "cuda:fused electric pair"
D_CHAIN = {"step_D": (CUDA_D, "fused pair D"), "update_E": (CUDA_D, "fused pair D")}


def test_a_taker_its_own_rule_vetoes_is_withheld_with_the_veto(monkeypatch, tmp_path):
    """CUDA first: the CUDA product is vetoed by the prior (no CUDA record), withholding
    it hands the seam to the Triton product, which the Triton record measured slower
    than its singles at this size; both are withheld, the seam ends on the singles the
    Triton record priced, and the step composed three times in all."""
    monkeypatch.setattr(fastpath, "_planned_repair_route",
                        lambda leading, digest: _route_facts())
    _write(tmp_path, [_slow_d(), _keep_d()], table="triton")
    selected = {"step_D": CUDA_D, "update_E": CUDA_D}
    singles = {"step_D": "PML", "update_E": "ordinary"}
    record, plan, chosen, seen = _cascade(selected, D_CHAIN, singles)
    assert plan.selected == singles
    assert chosen == {"cuda": (), "triton": ()}
    preference = record["preference"]
    assert preference["compositions"] == 3 and len(seen) == 2
    assert preference["withheld_by_table"] == {"cuda": [CUDA_D],
                                               "triton": ["fused pair D"]}
    vetoed, taker = _candidate(record, CUDA_D), _candidate(record, "fused pair D")
    assert vetoed["applied"] and vetoed["rule"] == "portable_prior"
    # One vendor's tables pool: the CUDA product is priced by the Triton record's bands.
    assert vetoed["verdict"]["evidence"]["vendor"] == "nvidia"
    assert [source["table"] for source in vetoed["verdict"]["evidence"]["derived_from"]] == [
        "triton"]
    assert vetoed["withheld_with"] == ["fused pair D"]
    assert taker["applied"] and taker["rule"] == "record"
    assert taker["taker_of"] == [CUDA_D] and taker["slots"] == ["step_D", "update_E"]
    assert taker["took_the_slots"] == ["PML", "ordinary"]
    assert taker["verdict"]["displaces"] == ["PML", "ordinary"]
    assert preference["rule_by_table"] == {"cuda": "portable_prior", "triton": "record"}
    assert CUDA_D in preference["note"] and "fused pair D on step_D" in preference["note"]


def test_a_taker_its_own_rule_keeps_drops_the_veto(monkeypatch, tmp_path):
    monkeypatch.setattr(fastpath, "_planned_repair_route",
                        lambda leading, digest: _route_facts())
    _write(tmp_path, [_entry(product=CUDA_D, bracketed=True, lo=24000, hi=2400000,
                             fused=0.81, baseline=0.51,
                             displaces=("cuda:PML", "cuda:ordinary"),
                             more_rows=((0.80, 0.50),))], table="cuda")
    _write(tmp_path, [_slow_d(fused=0.60, baseline=0.66, more_rows=((0.60, 0.66),))],
           table="triton")
    selected = {"step_D": CUDA_D, "update_E": CUDA_D}
    record, plan, chosen, seen = _cascade(
        selected, D_CHAIN, {"step_D": "cuda:PML", "update_E": "cuda:ordinary"})
    assert plan.selected == selected and chosen == {"cuda": (CUDA_D,),
                                                    "triton": ("fused pair D",)}
    assert record["preference"]["compositions"] == 2
    vetoed, taker = _candidate(record, CUDA_D), _candidate(record, "fused pair D")
    assert vetoed["applied"] is False and "withheld_with" not in vetoed
    assert vetoed["why_not_applied"] == (
        "withholding it let another fused product take the slots: fused pair D")
    assert taker["verdict"]["outcome"] == "not_vetoed" and taker["applied"] is False
    assert taker["taker_of"] == [CUDA_D]


def test_when_withholding_the_taker_does_not_stand_the_step_goes_on_as_without_it(
        monkeypatch, tmp_path):
    """The Triton taker is vetoed by its record, but withholding it hands the seam to the
    CUDA singles, which that record did not price: the wider composition does not stand,
    and the step composes once more with the veto that stood on the first composition
    (the B seam's), exactly as it would have without the taker rule."""
    monkeypatch.setattr(fastpath, "_planned_repair_route",
                        lambda leading, digest: _route_facts())
    _write(tmp_path, [_slow_d(), _keep_d()], table="triton")
    selected = {"step_B": "fused pair B", "update_H": "fused pair B",
                "step_D": CUDA_D, "update_E": CUDA_D}
    chain = dict(D_CHAIN, step_B=("fused pair B",), update_H=("fused pair B",))
    singles = {"step_B": "PML", "update_H": "ordinary", "step_D": "cuda:PML",
               "update_E": "cuda:ordinary"}
    record, plan, chosen, seen = _cascade(selected, chain, singles)
    assert plan.selected == dict(selected, step_B="PML", update_H="ordinary")
    preference = record["preference"]
    assert preference["compositions"] == 4 and len(seen) == 3
    assert seen[2] == {"triton": ("fused pair D",), "cuda": (CUDA_D,)}
    assert preference["withheld_by_table"] == {"triton": ["fused pair B"]}
    assert _candidate(record, "fused pair B")["applied"] is True
    vetoed, taker = _candidate(record, CUDA_D), _candidate(record, "fused pair D")
    assert vetoed["applied"] is False and taker["applied"] is False
    assert taker["why_not_applied"] == ("the record does not price these singles: "
                                        "baseline_is_not_what_the_candidate_displaces")
    assert vetoed["why_not_applied"].startswith(
        "withholding it let another fused product take the slots: fused pair D; "
        "withholding that as well did not stand: fused pair D: the record does not "
        "price these singles")


METAL_SELECTED = {"step_D": "fused electric D/E pair", "update_E": "fused electric D/E pair",
                  "step_B": "fused magnetic B/H pair", "update_H": "fused magnetic B/H pair"}
METAL_SINGLES = {"step_D": "PML", "update_E": "ordinary"}
METAL_OFFER = {"metal": ("fused electric D/E pair", "fused magnetic B/H pair")}


def _arbitrate_metal(architecture, grid_shape=(200, 120, 1)):
    """The Metal table's two pairs, the electric one bracketed, arbitrated at rung 5M's
    call into ``fastpath._apply_preference`` on ``architecture``."""
    fused = set(METAL_OFFER["metal"])
    return _arbitrate(METAL_SELECTED, METAL_SINGLES, offer=METAL_OFFER,
                      architecture=architecture, table_of=lambda label: "metal",
                      is_fused=lambda label: label in fused, grid_shape=grid_shape)


def test_the_metal_table_takes_no_prior_from_the_nvidia_records(tmp_path):
    """No Apple record exists, and the prior is per vendor: the NVIDIA records that would
    veto the bracketed pair at this size price nothing here, so the shape rule stands and
    the verdict says whose prior was asked and why it had none."""
    _write(tmp_path, [_slow_d(), _keep_d()], table="triton")
    record, plan, chosen, _scratch, seen = _arbitrate_metal(APPLE_RECORDED)
    assert chosen == METAL_OFFER and seen == []
    assert plan.selected == METAL_SELECTED
    preference = record["preference"]
    assert preference["records"]["metal"]["path"] == os.path.join(
        "metal_kernels", "timing_applegpu_g13s.json")
    assert preference["records"]["metal"]["status"] == "absent"
    assert preference["rule_by_table"] == {"metal": "portable_prior"}
    assert preference["note"] is None
    item = _candidate(record, "fused electric D/E pair")
    assert item["bracketed"] is True and item["applied"] is False
    assert item["why_not_applied"] == "no_prior"
    evidence = item["verdict"]["evidence"]
    assert evidence["vendor"] == "apple" and evidence["derived_from"] == []
    assert evidence["why_none"] == (
        "no shipped apple record holds on this tree and measures a key")
    by_vendor = preference["prior"]["by_vendor"]
    assert by_vendor["apple"]["by_architecture"] == {}
    assert by_vendor["nvidia"]["by_architecture"][RECORDED][
        "bracketed_veto_up_to_cells"] == 2400000


def test_the_metal_table_takes_the_prior_from_apple_records_alone(tmp_path):
    """An Apple GPU family with no record of its own reads the Apple records' crossover
    (1,014,000 cells here), never the NVIDIA one beside it (2,400,000): the bracketed pair
    gives way at 24,000 cells by the Apple prior, and keeps its slots at 1,200,000 cells,
    which the NVIDIA prior would have vetoed."""
    _write(tmp_path, [_slow_d(hi=1014000), _keep_d(lo=2400000, hi=2400000)], table="metal",
           capability=APPLE_RECORDED)
    _write(tmp_path, [_slow_d(), _keep_d()], table="triton")
    apple_path = os.path.join("metal_kernels", "timing_applegpu_g13s.json")
    record, plan, chosen, _scratch, _seen = _arbitrate_metal(APPLE_UNRECORDED)
    assert chosen == {"metal": ("fused magnetic B/H pair",)}
    assert plan.selected["step_D"] == "PML"
    assert plan.selected["step_B"] == "fused magnetic B/H pair"
    preference = record["preference"]
    assert preference["records"]["metal"]["status"] == "absent"
    assert preference["rule_by_table"] == {"metal": "portable_prior"}
    item = _candidate(record, "fused electric D/E pair")
    assert item["applied"] is True and item["rule"] == "portable_prior"
    evidence = item["verdict"]["evidence"]
    assert evidence["vendor"] == "apple"
    assert evidence["bracketed_veto_up_to_cells"] == 1014000
    assert [source["path"] for source in evidence["derived_from"]] == [apple_path]
    note = preference["note"]
    assert f"no metal timing record for {APPLE_UNRECORDED}" in note
    assert "1,014,000 cells" in note and "metal applegpu_g13s" in note
    assert "triton" not in note and "2,400,000" not in note
    record, plan, chosen, _scratch, _seen = _arbitrate_metal(APPLE_UNRECORDED,
                                                             grid_shape=(1200, 1000, 1))
    assert chosen == METAL_OFFER and plan.selected == METAL_SELECTED
    item = _candidate(record, "fused electric D/E pair")
    assert item["why_not_applied"] == "prior_above_the_recorded_crossover"
    assert item["verdict"]["evidence"]["vendor"] == "apple"


def test_the_metal_ladder_composes_with_the_offer_the_preference_left():
    """Rung 5M hands the arbitrated offer, not the release's, to the composition that
    runs; and it reaches the consult only through ``fastpath._apply_preference``."""
    source = pathlib.Path(HERE, "metal_dispatch.py").read_text(encoding="utf-8")
    assert "fastpath._apply_preference(" in source
    assert "fuse_labels=composed_offer" in source
    assert "fuse=bool(composed_offer)" in source
    assert "dispatch_preference" not in source.replace("fastpath._apply_preference", "")


def test_a_run_with_no_fused_product_composes_once_and_reads_nothing(monkeypatch):
    def never(*_args, **_kwargs):
        raise AssertionError("a record was opened with nothing to arbitrate")

    monkeypatch.setattr(dp, "load_record", never)
    record, plan, chosen, scratch, seen = _arbitrate(
        {"step_D": "PML", "update_E": "ordinary"}, {}, offer={"triton": ()},
        bracketed_slots=())
    assert seen == [] and scratch is None
    assert record["preference"]["why"] == "the shape rule installed no fused product"


# ---------------------------------------------------------------------------
# The route predicted at plan time, against the counters a real bracket writes
# ---------------------------------------------------------------------------


def _counted(leading, steps):
    trailing = deposit_repair.TrailingRepairPlan("slot", leading, leading._fields,
                                                 leading._pml)
    for _ in range(steps):
        leading.run()
        trailing.run()
    p = leading.prepared
    linear = p.linear_saves + p.linear_repairs
    cells = p.cells_saves + p.cells_repairs
    index = ("linear" if linear and not cells else "3-tuple" if cells and not linear
             else "mixed" if linear and cells else "never-ran")
    return {"index": index,
            "repairs_per_bracketed_step": round((p.linear_repairs + p.cells_repairs)
                                                / steps, 3),
            "restricted": p.restricted_sources > 0,
            "fell_back": p.restriction_fallbacks > 0}


def _route_cases():
    from meep_gpu.test_deposit_repair import _build, _make_source, _plain_build
    from meep_gpu.test_deposit_repair_linear_route import _StubSource

    def one(component):
        return lambda fields: [_make_source(fields.grid, "D", component,
                                            (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))]

    def two(fields):
        return one("Ez")(fields) + one("Ex")(fields)

    def fortran(fields):
        fields.Ez = numpy.asfortranarray(fields.Ez)

    split = (deposit_repair.SPLIT_FIELD_PATH,)
    plain = (deposit_repair.PLAIN_PATH,)
    return {
        "split_one_source": (_build, one("Ez"), split, None),
        "split_two_sources": (_build, two, split, None),
        "split_one_array_unflattenable": (_build, two, split, fortran),
        "an_index_the_linear_spelling_refuses": (
            _build, lambda fields: [_StubSource(numpy, -1, 3, 0)], split, None),
        "plain_one_source": (_plain_build, one("Ez"), plain, None),
    }


@pytest.mark.parametrize("case", sorted(_route_cases()))
def test_the_route_predicted_at_plan_time_is_the_one_the_counters_write(case):
    build, sources_of, paths, mutate = _route_cases()[case]
    _grid, fields, pml = build()
    if mutate is not None:
        mutate(fields)
    leading = deposit_repair.LeadingRepairPlan(CountingPlan("fused"), fields, pml,
                                               sources_of(fields), "D", paths)
    planned = fastpath._planned_repair_route(leading, "f" * 64)
    assert "unreadable" not in planned, planned
    counted = _counted(leading, 3)
    assert {name: planned[name] for name in counted} == counted
    assert leading.prepared.linear_saves + leading.prepared.cells_saves + \
        leading.prepared.linear_repairs + leading.prepared.cells_repairs > 0


def test_the_route_prediction_writes_nothing_into_the_plans_own_cache():
    from meep_gpu.test_deposit_repair import _build, _make_source

    _grid, fields, pml = _build()
    source = _make_source(fields.grid, "D", "Ez", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    leading = deposit_repair.LeadingRepairPlan(CountingPlan("fused"), fields, pml,
                                               [source], "D")
    fastpath._planned_repair_route(leading, "f" * 64)
    assert leading.prepared.entries == {}
    assert (leading.prepared.linear_saves, leading.prepared.restricted_sources) == (0, 0)


def test_a_route_that_cannot_be_read_states_nothing():
    planned = fastpath._planned_repair_route(object(), "f" * 64)
    assert set(planned) == {"deposit_repair_sha256", "unreadable"}
    assert dp.route_facts_missing(planned)
