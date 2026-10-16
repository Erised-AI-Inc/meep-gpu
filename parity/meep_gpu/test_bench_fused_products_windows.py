"""Device-free tests for how the fused-product bench restores, sizes and times its windows.

Added for the 1.0 re-certification round, from the CERT-8 harness findings:

* H1 -- ``thaw`` wrote into a ``numpy.float32`` the state walk had frozen (a dispersive
  leg's recurrence constants) and raised ``TypeError`` on 11 of 74 DRIVE rows;
* window sizing -- three 4-step calibrations, each carrying the cost of the restore
  before it, over-read the device step 3.2-4.9x and the 4,000-step cap bound windows
  short of their target (12 s asked, 2.5-3.7 s run on the Mac);
* H4 -- the first timed window of a leg ran 5-7 % slower than the five after it and
  struck the row on the 5 % within-leg gate;
* the GPU-only row (``--legs gpu-only``): one lift, no singles and no array leg;
* F5 -- provenance answered with the HEAD of whatever repository enclosed the tree.

The drivers are host stubs on a simulated clock: ``timed_window`` is replaced by one
that reads the clock the stub advances, so a cost model -- per-step seconds, a fixed
cost paid by the first run after each restore, a slow first full-length run -- is
measured exactly, with no sleep and no device.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_API = os.path.abspath(os.path.join(HERE, "..", ".."))
for _path in (HERE, REPO_API):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import bench_fused_products as bench  # noqa: E402


# ---------------------------------------------------------------------------
# A simulated clock and a host driver whose costs are declared
# ---------------------------------------------------------------------------

class _Clock:
    def __init__(self):
        self.now = 0.0


def _timed_window_on(clock):
    def timed_window(leg, launches, sync):
        sync()
        start = clock.now
        for _ in range(launches):
            leg()
        sync()
        return clock.now - start
    return timed_window


class _Counters:
    """``e2e.LaunchCounters``' shape: per-table totals the stub drivers bump."""

    def __init__(self):
        self.totals = {"triton": 0, "cuda": 0}
        self.triton = SimpleNamespace(install=lambda: None)
        self.cuda = SimpleNamespace(install=lambda: None, wrap_memo_store=lambda: None)

    def snapshot(self):
        return dict(self.totals)

    def delta(self, before, after):
        return {table: {"total": after[table] - before[table],
                        "by_kernel": ({f"{table}_kernel": 1}
                                      if after[table] > before[table] else {})}
                for table in ("triton", "cuda")}


class _Driver:
    """A host driver on the simulated clock.

    ``per_step`` seconds a step; ``after_restore`` seconds paid by the first run after
    each restore (what a held residency's post-restore uploads cost); ``first_long``
    an extra share of a run's time paid ONCE, by the first run lasting at least
    ``long_seconds`` (the shape of H4). ``launches`` kernel launches a step, counted
    on the Triton witness; ``slots`` the plan's slot->arm map.
    """

    def __init__(self, clock, counters, slots, launches, per_step=1e-3,
                 after_restore=0.0, first_long=0.0, long_seconds=float("inf")):
        self.xp = numpy
        self.gpu = "cuda"
        self.clock, self.counters = clock, counters
        self.slots, self.launches = dict(slots), launches
        self.per_step, self.after_restore = per_step, after_restore
        self.first_long, self.long_seconds = first_long, long_seconds
        self._restored = False
        self._step_count = 0
        # THE H1 SHAPE: a dispersive leg's polarization holds its recurrence constants
        # as four ``numpy.float32`` words beside its arrays.
        self.fields = SimpleNamespace(
            Ez=numpy.zeros(32, dtype=numpy.float32),
            polarizations=[SimpleNamespace(
                P=numpy.zeros(32, dtype=numpy.float32),
                constants=tuple(numpy.float32(v) for v in (1.99, 0.998, 5.5e-4, 0.998)))])
        self.pml = None
        self._fast_path = None
        self._dft_monitors, self._flux_monitors = [], []
        self.runs = []

    @property
    def step_count(self):
        return self._step_count

    @step_count.setter
    def step_count(self, value):
        # ``bench.thaw`` restores the clock last: that is the restore this stub sees.
        self._step_count = int(value)
        self._restored = True

    @property
    def active_step_path(self):
        return "fused" if self.slots else "array"

    def run(self, num_steps):
        cost = self.per_step * num_steps
        if self._restored:
            cost += self.after_restore
            self._restored = False
        if self.first_long and cost >= self.long_seconds:
            cost *= 1.0 + self.first_long
            self.first_long = 0.0
        self.clock.now += cost
        self.runs.append(num_steps)
        self.counters.totals["triton"] += self.launches * num_steps
        self.fields.Ez += numpy.float32(1.0)
        self.fields.polarizations[0].P += numpy.float32(0.5)
        self._step_count += num_steps

    def fast_path_report(self):
        if not self.slots:
            return {"decision": "array", "slots": {}, "composition": None}
        return {"decision": "dispatch",
                "slots": {s: {"state": "dispatched", "arm": a} for s, a in self.slots.items()},
                "composition": {"table": "triton", "tables_dispatched": ["triton"]}}

    def close(self):
        pass


FUSED_SLOTS = {"step_B": "fused pair B", "update_H": "fused pair B",
               "step_D": "PML", "update_E": "offdiag"}
SINGLES_SLOTS = {"step_B": "PML", "update_H": "ordinary", "step_D": "PML",
                 "update_E": "offdiag"}
SPEC = {"arms": ["fused pair B"], "pairs": 1, "seam": ["B"], "expect": "dispatch",
        "reached_by": "release", "why": "stub"}
STUB_MODULE = SimpleNamespace(CASES={"stub": None},
                              _fuse_env=lambda arms, reached_by="release": {},
                              unfused_env=lambda: {}, ARRAY_ENV={})


@pytest.fixture
def rig(monkeypatch):
    clock, counters = _Clock(), _Counters()
    monkeypatch.setattr(bench, "timed_window", _timed_window_on(clock))
    monkeypatch.setattr(bench, "sync_for", lambda _driver: (lambda: None))
    monkeypatch.setattr(bench, "box_state", lambda: {"loadavg": [0.0, 0.0, 0.0]})
    monkeypatch.setattr(bench, "WITNESSES", ["triton"])
    return SimpleNamespace(clock=clock, counters=counters,
                           witness=bench.NvidiaWitness(counters))


def _lift(rig, monkeypatch, build):
    lifted = []

    def fake_run_leg(_case, _builder, label, _env, _counter, _gpu, _res):
        lifted.append(label)
        return {"label": label, "driver": build(label), "env": {},
                "grid_shape": [32, 1, 1], "until": 1.0}

    monkeypatch.setattr(bench.e2e, "run_leg", fake_run_leg)
    return lifted


def _legs(rig, **costs):
    def build(label):
        slots, launches = {"fused": (FUSED_SLOTS, 3), "unfused": (SINGLES_SLOTS, 4),
                           "array": ({}, 0)}[label]
        return _Driver(rig.clock, rig.counters, slots, launches, **costs.get(label, {}))
    return build


def _run(rig, legs_mode=None, target=0.5, **kw):
    extra = {} if legs_mode is None else {"legs_mode": legs_mode}
    return bench.run_case("stub", dict(SPEC), STUB_MODULE, "triton", 0, rig.witness,
                          None, repeats=6, target_seconds=target,
                          step_cap=kw.pop("step_cap", bench.build_parser().parse_args(
                              ["--out", "unused"]).step_cap),
                          warm_steps=6, memory_budget=0, **extra, **kw)


# ---------------------------------------------------------------------------
# H1: a NumPy scalar in the walked state
# ---------------------------------------------------------------------------

def test_a_frozen_numpy_scalar_survives_the_restore(rig):
    driver = _Driver(rig.clock, rig.counters, FUSED_SLOTS, 3)
    frozen = bench.freeze(driver)
    driver.run(5)
    assert bench.moved_words(driver, frozen) > 0
    bench.thaw(driver, frozen)          # raised TypeError on a numpy.float32
    assert bench.moved_words(driver, frozen) == 0
    assert driver.step_count == frozen["step_count"]
    assert bench.non_finite(driver) == 0
    assert isinstance(driver.fields.polarizations[0].constants[0], numpy.float32)


def test_a_scalar_a_step_rebound_is_refused_by_name(rig):
    driver = _Driver(rig.clock, rig.counters, FUSED_SLOTS, 3)
    frozen = bench.freeze(driver)
    driver.fields.polarizations[0].constants = (numpy.float32(2.0),) + tuple(
        driver.fields.polarizations[0].constants[1:])
    with pytest.raises(RuntimeError, match=r"constants\[0\]"):
        bench.thaw(driver, frozen)
    assert bench.moved_words(driver, frozen) > 0


# ---------------------------------------------------------------------------
# Window sizing and the step cap
# ---------------------------------------------------------------------------

def test_the_window_is_sized_from_a_probe_after_the_restore(rig):
    """A restore that costs 12 steps' time is not read as the step's cost.

    The old calibration timed three 4-step runs, each paying the restore, and read
    (12 + 4) / 4 = 4x the step: a 2 s window came out at 0.5 s.
    """
    driver = _Driver(rig.clock, rig.counters, FUSED_SLOTS, 3, per_step=1e-3,
                     after_restore=12e-3)
    frozen = bench.freeze(driver)
    sizing = bench.size_chunk({"driver": driver}, frozen, lambda: None, 2.0, 10**9)
    assert sizing["calibration_per_step_seconds"] == pytest.approx(1e-3, rel=1e-9)
    assert sizing["steps_per_window"] == 2000
    assert sizing["probe"]["rounds"][-1]["seconds"] >= 0.2
    # and a window of that many steps, restore included, lands on the target
    bench.thaw(driver, frozen)
    start = rig.clock.now
    driver.run(sizing["steps_per_window"])
    assert rig.clock.now - start == pytest.approx(2.012, rel=1e-9)


def test_one_slow_probe_does_not_size_the_window(rig):
    """Measured on the GPU host: the first leg's first long run after its freeze read
    25 ms/step against 6 in every later run, and one probe sized a 2 s window at 0.5 s.
    The window is sized from the median of the probes at the final step count."""
    driver = _Driver(rig.clock, rig.counters, FUSED_SLOTS, 3, per_step=1e-3,
                     first_long=3.0, long_seconds=0.1)
    sizing = bench.size_chunk({"driver": driver}, bench.freeze(driver), lambda: None,
                              2.0, 10**9)
    final = [p for p in sizing["probe"]["rounds"] if p["steps"] == 250]
    assert [round(p["seconds"], 6) for p in final] == [1.0, 0.25, 0.25]
    assert sizing["steps_per_window"] == 2000


def test_the_default_step_cap_does_not_bind_a_twelve_second_window(rig):
    """110,592 cells on the Mac: 12 s asked, the 4,000-step cap ran 2.6 s."""
    cap = bench.build_parser().parse_args(["--out", "unused"]).step_cap
    driver = _Driver(rig.clock, rig.counters, FUSED_SLOTS, 3, per_step=1e-4)
    sizing = bench.size_chunk({"driver": driver}, bench.freeze(driver), lambda: None,
                              12.0, cap)
    assert sizing["steps_per_window"] == 120_000
    assert not sizing["probe"]["capped"]


def test_a_cap_below_the_target_still_binds_and_says_so(rig):
    driver = _Driver(rig.clock, rig.counters, FUSED_SLOTS, 3, per_step=1e-4)
    sizing = bench.size_chunk({"driver": driver}, bench.freeze(driver), lambda: None,
                              12.0, 500)
    assert sizing["steps_per_window"] == 500 and sizing["probe"]["capped"]


# ---------------------------------------------------------------------------
# One realistic system: run_case over three stub legs, then the GPU-only row
# ---------------------------------------------------------------------------

def test_a_row_restores_sizes_and_times_every_leg_to_its_target(rig, monkeypatch):
    """H1, the sizing and H4 together, as a dispersive 512,000-cell row met them.

    Every leg pays a restore of 12 steps' time; the fused leg's first full-length run
    is 7 % slow. The row must lift, restore through the scalars, size every window to
    the target, discard each leg's first full window, and pass the spread gate.
    """
    lifted = _lift(rig, monkeypatch, _legs(
        rig,
        fused={"per_step": 7e-4, "after_restore": 8.4e-3, "first_long": 0.07,
               "long_seconds": 0.25},
        unfused={"per_step": 7.5e-4, "after_restore": 9e-3},
        array={"per_step": 6e-3, "after_restore": 2.4e-2}))
    row = _run(rig)
    assert lifted == ["fused", "unfused", "array"]
    assert row["verdict"] == "TIMED", row.get("floors")
    assert row["floors"]["spread_within_gate"] is True
    assert row["floors"]["bit_identical"] is True
    assert [w["leg"] for w in row["windows_discarded"]] == ["fused", "unfused", "array"]
    slow = row["windows_discarded"][0]
    assert slow["seconds_per_step"] > 1.06 * row["per_leg"]["fused"]["median_seconds_per_step"]
    for label in ("fused", "unfused", "array"):
        timed = [w["seconds"] for w in row["windows"] if w["leg"] == label]
        assert len(timed) == 6
        assert all(abs(s - 0.5) < 0.05 for s in timed), (label, timed)
    assert row["ratios"]["unfused_over_fused"] == pytest.approx(
        (7.5e-4 * 667 + 9e-3) / 667 / ((7e-4 * 714 + 8.4e-3) / 714), rel=1e-2)


def test_no_discard_leaves_the_first_window_in_and_the_gate_strikes_it(rig, monkeypatch):
    """The control: with ``discard_windows=0`` the same row is struck, as on CERT-8."""
    _lift(rig, monkeypatch, _legs(
        rig, fused={"per_step": 7e-4, "first_long": 0.07, "long_seconds": 0.25}))
    row = _run(rig, discard_windows=0)
    assert row["windows_discarded"] == []
    assert row["floors"]["spread_within_gate"] is False
    assert row["verdict"] == "TIMED-NOT-REPORTABLE"


def test_a_gpu_only_row_lifts_one_leg_and_claims_no_ratio(rig, monkeypatch):
    lifted = _lift(rig, monkeypatch, _legs(rig, fused={"per_step": 7e-4}))
    row = _run(rig, legs_mode="gpu-only")
    assert lifted == ["fused"]
    assert row["legs_mode"] == "gpu-only" and row["legs_timed"] == ["fused"]
    assert row["verdict"] == "TIMED", row.get("floors")
    assert set(row["per_leg"]) == {"fused"}
    assert row["ratios"] is None
    assert row["bit_identity"]["measured"] is False
    assert row["bit_identity"]["fused_vs_array_identical"] is None
    assert row["precondition"]["measured"] is False
    # the comparative floors are ABSENT, not passed
    assert "bit_identical" not in row["floors"]
    assert "singles_are_the_same_table" not in row["floors"]
    sub = row["substitution"]
    assert sub["mode"] == "gpu-only" and sub["comparative"] is False
    assert sub["tables_dispatched"] == ["triton"]
    assert sub["welded_arms"] == {"fused pair B": ["step_B", "update_H"]}
    assert sub["checks"] == {"a_named_arm_holds_more_than_one_slot": True,
                             "fused_products_match_the_rows_pairs": True,
                             "fused_leg_launched": True}
    assert row["moved_words"]["fused"] > 0


def test_a_gpu_only_row_that_fused_nothing_is_not_proved(rig, monkeypatch):
    def build(label):
        return _Driver(rig.clock, rig.counters, SINGLES_SLOTS, 4)
    _lift(rig, monkeypatch, build)
    row = _run(rig, legs_mode="gpu-only")
    assert row["substitution"]["proved"] is False
    assert row["verdict"] == "TIMED-NOT-REPORTABLE"


def test_a_campaign_of_gpu_only_rows_summarises_without_a_ratio(rig, monkeypatch, tmp_path):
    """The summary's fusion medians are over rows that carry a ratio; a GPU-only row
    is reportable and named as such."""
    rows = []
    monkeypatch.setattr(bench, "append_jsonl", lambda _path, row: rows.append(row))
    monkeypatch.setattr(bench, "save", lambda summary, _path: rows.append(summary))
    monkeypatch.setattr(bench, "provenance", lambda _table=None: {
        "sha256": {}, "git": {"head": None, "any_dirty": None}})
    monkeypatch.setattr(bench, "run_case", lambda case, *_a, **_k: {
        "case": case, "verdict": "TIMED", "reportable": True, "ratios": None,
        "floors": {"substitution_proved": True}, "legs_mode": "gpu-only"})
    args = bench.build_parser().parse_args(["--out", str(tmp_path), "--legs", "gpu-only"])
    code = bench._campaign(args, ["stub"], STUB_MODULE, {"stub": SPEC}, rig.witness,
                           str(tmp_path / "rows.jsonl"))
    summary = rows[-1]
    assert code == 0
    assert summary["reportable_gpu_only"] == ["stub"]
    assert summary["fused_vs_singles"] == {} and summary["median_fused_vs_singles"] is None


# ---------------------------------------------------------------------------
# F5: provenance asks git only about a checkout of this tree
# ---------------------------------------------------------------------------

def _git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), "-c", "user.name=test",
                           "-c", "user.email=test@example.invalid", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_a_tree_inside_another_checkout_records_no_head(tmp_path, monkeypatch):
    outer = tmp_path / "outer"
    outer.mkdir()
    _git(outer, "init", "-q")
    (outer / "README").write_text("outer\n")
    _git(outer, "add", "README")
    _git(outer, "commit", "-q", "-m", "outer")
    head = _git(outer, "rev-parse", "HEAD")
    tree = outer / "results" / "ladder" / "tree"
    (tree / "parity" / "meep_gpu").mkdir(parents=True)
    (tree / bench.BENCH_RELATIVE).write_text("# a snapshot's copy\n")
    monkeypatch.setattr(bench, "REPO_API", str(tree))
    stamp = bench.provenance()
    assert stamp["git"]["head"] is None, "recorded the enclosing repository's HEAD"
    assert "no checkout of this tree" in stamp["git"]["error"]
    assert stamp["git"]["any_dirty"] is None
    # the positive control: once the enclosing repository tracks the copy, it is this
    # tree's checkout and its HEAD is recorded
    _git(outer, "add", str(tree / bench.BENCH_RELATIVE))
    _git(outer, "commit", "-q", "-m", "track")
    stamp = bench.provenance()
    assert stamp["git"]["head"] == _git(outer, "rev-parse", "HEAD") != head


def test_the_fused_timing_report_leaves_gpu_only_rows_out_by_name(rig, monkeypatch):
    import build_fused_timing_report as report  # noqa: PLC0415
    _lift(rig, monkeypatch, _legs(rig, fused={"per_step": 7e-4}))
    solo = _run(rig, legs_mode="gpu-only")
    solo.update(provenance={"sha256": {k: "f" * 64 for k in report.PINNED_KEYS}},
                prefer_gpu=True, smoke=False)
    assert "1 GPU-only row(s)" in report.render([solo], "t")
    full = _run(rig)
    full.update(provenance=solo["provenance"], prefer_gpu=True, smoke=False, case="full")
    text = report.render([full, solo], "t")
    assert "1 GPU-only row(s) (`--legs gpu-only`: one leg, no fusion number) are left out: " \
        "stub." in text
    assert "`full`" in text
