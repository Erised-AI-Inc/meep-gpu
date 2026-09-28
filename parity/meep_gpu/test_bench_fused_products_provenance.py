"""Host tests for the fused-timing campaign's provenance and deposit-repair route record.

Added 2026-09-19. ``bench_fused_products.py`` rows carried no source digest, so a row
timed after that day's two overhead fixes (``meep_gpu/deposit_repair.py``'s linear
repair route, ``meep_gpu/cuda_kernels/fused_pairs.py`` holding source/arguments per
plan) could not be told from a 2026-09-17 row timed before them, and nothing in a row
said whether the linear route ran or silently fell back to the 3-tuple one. These tests
pin the three pieces that fix that: the provenance block, the route counters read off a
frozen plan, and the report rendering both new rows and rows written before either field
existed. No digest is hard-coded: the files are edited during a campaign, which is what
the field is for.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from types import SimpleNamespace

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_API = os.path.abspath(os.path.join(HERE, "..", ".."))
for _path in (HERE, REPO_API):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import bench_fused_products as bench  # noqa: E402
import build_fused_timing_report as report  # noqa: E402
from meep_gpu import deposit_repair  # noqa: E402

HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")


class _Counters:
    """The launch-witness bundle's shape with nothing installed.

    The real ``e2e.LaunchCounters`` witnesses patch launch entry points process-wide
    when installed, and ``prime`` wraps the compile memo store; neither belongs in a
    test process shared with the rest of the suite. These tests read the route
    counters, which need no launch witness at all.
    """

    def __init__(self):
        self.triton = SimpleNamespace(install=lambda: None)
        self.cuda = SimpleNamespace(install=lambda: None, wrap_memo_store=lambda: None)

    @staticmethod
    def snapshot():
        return {"triton": {"total": 0, "by_kernel": {}},
                "cuda": {"total": 0, "by_kernel": {}}}

    @staticmethod
    def delta(_before, after):
        return after


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def test_provenance_digests_every_pinned_source_as_it_reads_on_disk():
    stamp = bench.provenance()
    keys = [key for key, _module, _path in bench.PINNED_SOURCES]
    assert keys == list(report.PINNED_KEYS)
    assert sorted(stamp["sha256"]) == sorted(keys)
    for key in keys:
        sha = stamp["sha256"][key]
        assert sha is not None and HEX64.match(sha), (key, sha)
        path = stamp["paths"][key]
        path = path if os.path.isabs(path) else os.path.join(REPO_API, path)
        with open(path, "rb") as handle:
            assert hashlib.sha256(handle.read()).hexdigest() == sha, key
    assert stamp["paths"]["bench"] == "parity/meep_gpu/bench_fused_products.py"
    assert stamp["paths"]["deposit_repair"] == "meep_gpu/deposit_repair.py"
    assert stamp["not_imported"] == {}


def test_provenance_git_block_is_answered_or_says_why_not():
    git = bench.provenance()["git"]
    if git["error"] is None:
        assert git["head"] is not None and HEX40.match(git["head"]), git["head"]
        assert set(git["dirty"]) == set(report.PINNED_KEYS)
        assert all(isinstance(v, bool) or v is None for v in git["dirty"].values())
        assert git["any_dirty"] == any(bool(v) for v in git["dirty"].values())
        assert set(git["porcelain"]) == {k for k, v in git["dirty"].items() if v}
    else:
        assert git["head"] is None and git["any_dirty"] is None


def test_provenance_survives_a_tree_with_no_git(monkeypatch):
    def refuse(*_args):
        raise FileNotFoundError("git: not a repository")
    monkeypatch.setattr(bench, "_git", refuse)
    stamp = bench.provenance()
    assert stamp["git"]["head"] is None
    assert "FileNotFoundError" in stamp["git"]["error"]
    assert all(v is None for v in stamp["git"]["dirty"].values())
    assert all(HEX64.match(v) for v in stamp["sha256"].values())


def test_every_row_main_writes_carries_the_stamp_including_error_rows(
        monkeypatch, tmp_path):
    """The stamp goes on at the one line every row passes through in :func:`main`."""
    calls = []

    def fake_run_case(case, *_args, **_kwargs):
        calls.append(case)
        if case == "boom":
            raise RuntimeError("lift exploded")
        return {"case": case, "verdict": "SKIP-NOT-THIS-LEG"}

    monkeypatch.setattr(bench, "selected_cases", lambda *_a: ["skipped", "boom"])
    monkeypatch.setattr(bench, "drive_rows", lambda _n: {"skipped": {}, "boom": {}})
    monkeypatch.setattr(bench, "run_case", fake_run_case)
    monkeypatch.setattr(bench.e2e, "LaunchCounters", _Counters)
    # Every global ``main`` writes, registered so each is put back after the test.
    monkeypatch.setattr(bench.e2e, "PREFER_GPU", True)
    monkeypatch.setattr(bench.route, "BACKEND", bench.route.BACKEND)
    monkeypatch.setattr(bench, "WITNESSES", [])
    monkeypatch.setattr(bench, "ONLY_ARMS", [])
    assert bench.main(["--out", str(tmp_path), "--smoke"]) == 1
    rows = [json.loads(line) for line in open(tmp_path / "rows.jsonl")]
    assert [r["verdict"] for r in rows] == ["SKIP-NOT-THIS-LEG", "ERROR"]
    for row in rows:
        assert HEX64.match(row["provenance"]["sha256"]["deposit_repair"])
        assert row["prefer_gpu"] is False
    summary = json.load(open(tmp_path / "summary.json"))
    assert summary["provenance"]["sha256"] == rows[0]["provenance"]["sha256"]


# ---------------------------------------------------------------------------
# Route counters off a frozen plan
# ---------------------------------------------------------------------------

class _Wrapper:
    """A slot wrapper that keeps its occupant as ``inner`` (no such wrapper exists
    around a bracket today; the walk must still find one if it appears)."""

    def __init__(self, inner):
        self.inner = inner


def _bracket():
    leading = deposit_repair.LeadingRepairPlan(object(), None, None, (), "D")
    trailing = deposit_repair.TrailingRepairPlan("update_E", leading, None, None)
    return leading, trailing


def _driver(plans, stale=False):
    return SimpleNamespace(_fast_path=SimpleNamespace(
        step_plan=SimpleNamespace(plans=plans)), _fast_path_stale=stale)


def _bump(prepared, ls=0, lr=0, cs=0, cr=0):
    prepared.linear_saves += ls
    prepared.linear_repairs += lr
    prepared.cells_saves += cs
    prepared.cells_repairs += cr


def test_route_delta_counts_each_cache_once_and_orders_slots_by_step():
    d_lead, d_trail = _bracket()
    b_lead, b_trail = _bracket()
    plans = {"update_E": d_trail, "step_D": d_lead, "update_H": b_trail,
             "step_B": _Wrapper(b_lead), "update_P": [object()]}
    driver = _driver(plans)
    before = bench.repair_snapshot(driver)
    assert before["bracketed_slots"] == ["step_B", "update_H", "step_D", "update_E"]
    assert len(before["caches"]) == 2
    _bump(d_lead.prepared, ls=6, lr=6)
    _bump(b_lead.prepared, ls=3, lr=2, cs=1, cr=1)
    delta = bench.repair_delta(before, bench.repair_snapshot(driver))
    assert {k: delta[k] for k in bench.REPAIR_COUNTERS} == {
        "linear_saves": 9, "linear_repairs": 8, "cells_saves": 1, "cells_repairs": 1}
    assert delta["plan_changed_inside_window"] is False


def test_route_delta_after_a_refreeze_counts_the_new_cache_from_zero():
    lead, trail = _bracket()
    driver = _driver({"step_D": lead, "update_E": trail})
    _bump(lead.prepared, ls=100, lr=100)
    before = bench.repair_snapshot(driver)
    fresh, fresh_trail = _bracket()
    _bump(fresh.prepared, ls=3, lr=3)
    driver._fast_path.step_plan.plans = {"step_D": fresh, "update_E": fresh_trail}
    delta = bench.repair_delta(before, bench.repair_snapshot(driver))
    assert delta["linear_saves"] == 3 and delta["linear_repairs"] == 3
    assert delta["plan_changed_inside_window"] is True


def test_route_snapshot_is_empty_without_a_current_plan():
    lead, trail = _bracket()
    for driver in (SimpleNamespace(_fast_path=None, _fast_path_stale=False),
                   _driver({"step_D": lead, "update_E": trail}, stale=True),
                   _driver({"step_B": object(), "update_H": object()}),
                   SimpleNamespace()):
        snap = bench.repair_snapshot(driver)
        assert snap == {"caches": {}, "bracketed_slots": []}
        delta = bench.repair_delta(snap, snap)
        assert delta["bracketed_slots"] == [] and delta["linear_saves"] == 0


def test_route_total_sums_windows_and_unions_slots():
    windows = [
        {"linear_saves": 6, "linear_repairs": 6, "cells_saves": 0, "cells_repairs": 0,
         "bracketed_slots": ["step_D", "update_E"],
         "plan_changed_inside_window": False},
        {"linear_saves": 6, "linear_repairs": 6, "cells_saves": 0, "cells_repairs": 1,
         "bracketed_slots": ["step_D", "update_E"],
         "plan_changed_inside_window": True},
    ]
    total = bench.repair_route_total(windows)
    assert total == {"linear_saves": 12, "linear_repairs": 12, "cells_saves": 0,
                     "cells_repairs": 1, "bracketed_slots": ["step_D", "update_E"],
                     "plan_changed_inside_window": True, "windows": 2}
    assert bench.repair_route_total([])["bracketed_slots"] == []


@pytest.fixture
def _no_leaked_subnormal_policy(monkeypatch):
    """Undo the policy a frozen plan installs, as ``meep_gpu/conftest.py`` does.

    Freezing a real plan installs the table's subnormal policy PROCESS-WIDE (FTZ/DAZ
    on the host), and ``parity/meep_gpu`` has no conftest to take it back out. Measured
    2026-09-19: left installed, every later test that counts subnormals read 0 --
    eight tests in test_pml_gate_harness, test_cuda_dynamic_loop_arity,
    test_gate_cuda_offdiag and test_measure_predicate_coverage failed in the full run
    and passed alone. The environment variable is restored by monkeypatch; the install
    outranks it and needs the uninstall.
    """
    import numpy as _np  # noqa: PLC0415

    def flushes() -> bool:
        return bool((_np.array([1e-40], dtype=_np.float32) * _np.float32(1.0))[0] == 0)

    flushed_before = flushes()
    monkeypatch.delenv("MEEP_GPU_SUBNORMAL_POLICY", raising=False)
    yield
    from meep_gpu import subnormal_policy  # noqa: PLC0415
    if subnormal_policy.policy_is_installed():
        subnormal_policy.uninstall_subnormal_policy()
    # THE FPU TOO. ``uninstall_subnormal_policy`` deliberately leaves the host FPU where
    # it is, and on this Mac the Metal table's flush is driven by ``fesetenv``
    # (``_drive_host_fpu_by_fenv``), so the thread kept flushing after the test: that
    # was the leak, measured -- the probe after it read ``1e-40 * 1 == 0`` with no
    # policy installed. Put back exactly what was found.
    if flushes() and not flushed_before:
        subnormal_policy._drive_host_fpu(False)  # noqa: SLF001
    assert flushes() == flushed_before, "the host FPU was not restored"


def test_route_counters_move_on_a_real_host_plan(monkeypatch, _no_leaked_subnormal_policy):
    """pml_2d lifted on the host, frozen by the bench's own ``prime``, one timed window.

    On a Mac the Metal table serves and brackets the D seam (the source is Ez). A host
    with no kernel table composes no bracket, and the test says so rather than passing.
    """
    pytest.importorskip("meep")
    from meep_gpu import backends  # noqa: PLC0415

    # On a Mac the Metal table is reached by a GPU driver; elsewhere the lift stays
    # the reference, which composes no bracket, and the test says so.
    monkeypatch.setattr(bench.e2e, "PREFER_GPU", backends.available_gpu() == "metal")
    module = bench.table_module("triton")
    spec = bench.drive_rows("triton")["pml_2d"]
    counter = _Counters()
    env = bench.leg_env(module, spec, "fused")
    # ``prime``/``window`` apply the leg's environment to ``os.environ`` directly.
    held = {name: os.environ.get(name) for name in env}
    leg = bench.e2e.run_leg("pml_2d", module.CASES["pml_2d"], "fused", env,
                            counter, 0, None)
    try:
        bench.prime("pml_2d", leg, counter, 2)
        if not bench.repair_snapshot(leg["driver"])["bracketed_slots"]:
            pytest.skip("no kernel table on this host composed a deposit-repair bracket")
        frozen = bench.freeze(leg["driver"])
        entry = bench.window("pml_2d", leg, frozen, counter, 3, 0)
    finally:
        leg["driver"].close()
        for name, value in held.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    route = entry["deposit_repair_route"]
    assert route["bracketed_slots"] == ["step_D", "update_E"]
    assert route["linear_saves"] > 0 and route["linear_repairs"] > 0, route
    assert route["cells_saves"] == 0 and route["cells_repairs"] == 0, route
    assert route["plan_changed_inside_window"] is False


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def _row(case, sha=None, route=None, prefer_gpu=None):
    """A reportable row in the bench's shape; ``sha``/``route`` None = an old row."""
    per_leg = {leg: {"median_seconds_per_step": s}
               for leg, s in (("fused", 0.001), ("unfused", 0.002), ("array", 0.004))}
    if route is not None:
        per_leg["fused"]["deposit_repair_route"] = route
    plan = {"arms": {"step_D": "fused pair D", "update_E": "fused pair D"},
            "composition": {"tables_dispatched": ["triton"]}}
    row = {"case": case, "drive_table": "triton", "utc": "2026-09-19T00:00:00Z",
           "grid_shape": [8, 8, 1], "reportable": True, "verdict": "TIMED",
           "floors": {"substitution_proved": True},
           "ratios": {"unfused_over_fused": 2.0, "array_over_fused": 4.0},
           "per_leg": per_leg, "plans": {"fused": plan, "unfused": plan},
           "substitution": {"tables_dispatched": ["triton"],
                            "fused_and_singles_used_the_same_table": True,
                            "launches_per_step": {"fused": 2, "unfused": 4}}}
    if sha is not None:
        row["provenance"] = {"sha256": {key: sha for key in report.PINNED_KEYS},
                             "git": {"head": "a" * 40, "any_dirty": False}}
    if prefer_gpu is not None:
        row["prefer_gpu"] = prefer_gpu
    return row


def _route(ls=0, lr=0, cs=0, cr=0, slots=("step_D", "update_E")):
    return {"linear_saves": ls, "linear_repairs": lr, "cells_saves": cs,
            "cells_repairs": cr, "bracketed_slots": list(slots),
            "plan_changed_inside_window": False, "windows": 6}


def test_old_rows_render_as_not_recorded():
    text = report.render([_row("pml_2d"), _row("offdiag_2d")], "t")
    assert "Source provenance: **not recorded** on all 2 rows" in text
    assert "not one population" not in text
    table = [line for line in text.splitlines() if line.startswith("| `pml_2d` | ")]
    assert len(table) == 1 and "| not recorded |" in table[0]
    assert "- **not recorded** (2): `offdiag_2d`, `pml_2d`" in text


def test_mixed_digests_are_named_in_the_header_and_per_row():
    rows = [_row("old_a"), _row("new_a", sha="1" * 64, route=_route(6, 6)),
            _row("new_b", sha="2" * 64, route=_route(6, 6))]
    text = report.render(rows, "t")
    assert "combine 3 `meep_gpu/deposit_repair.py` provenances" in text
    assert "not one population" in text
    for spelled in ("not recorded: 1 row", "`111111111111`: 1 row",
                    "`222222222222`: 1 row"):
        assert spelled in text
    header = next(line for line in text.splitlines() if line.startswith("| case |"))
    assert header.endswith(" src |")
    assert "these pinned sources ALSO differ: `fused_pairs` (2 digests)" in text


def test_one_digest_names_head_and_no_src_column():
    rows = [_row("a", sha="3" * 64, route=_route(6, 6)),
            _row("b", sha="3" * 64, route=_route(6, 6))]
    text = report.render(rows, "t")
    assert ("all 2 rows timed `meep_gpu/deposit_repair.py` `333333333333`; git HEAD "
            "`aaaaaaaaaaaa`") in text
    assert "not one population" not in text
    header = next(line for line in text.splitlines() if line.startswith("| case |"))
    assert not header.endswith(" src |")


@pytest.mark.parametrize("route, verdict", [
    (_route(42, 42), "linear"),
    (_route(0, 24), "linear"),                     # plain path: saves nothing
    (_route(0, 0, 6, 6), "3-tuple fallback"),
    (_route(6, 5, 0, 1), "mixed"),
    (_route(0, 0, 0, 0), "never ran"),
    (_route(slots=()), "no bracket"),
    (None, "not recorded"),
])
def test_route_verdicts(route, verdict):
    assert report.route_verdict(route) == verdict


def test_route_section_names_a_fallback_row_and_the_licence_says_why():
    rows = [_row("fast", sha="4" * 64, route=_route(42, 42)),
            _row("slow", sha="4" * 64, route=_route(0, 0, 42, 42))]
    text = report.render(rows, "t")
    assert "## Deposit-repair route on the fused leg" in text
    assert "`fast` — step_D+update_E: linear (42/42 linear, 0/0 3-tuple)" in text
    assert ("`slow` — step_D+update_E: **3-TUPLE FALLBACK** (0/0 linear, 42/42 "
            "3-tuple)") in text
    assert "whose `repair route` does not read `linear`" in text


def test_smoke_rows_are_flagged_and_unwitnessed_groups_do_not_crash():
    """Host rows have no launch witness; the one/two-pair split used to raise on them."""
    one = _row("one", sha="5" * 64, route=_route(6, 6), prefer_gpu=False)
    two = _row("two", sha="5" * 64, route=_route(6, 6), prefer_gpu=False)
    two["plans"]["fused"] = dict(two["plans"]["fused"], arms={
        "step_B": "fused pair B", "update_H": "fused pair B",
        "step_D": "fused pair D", "update_E": "fused pair D"})
    for row in (one, two):
        row["substitution"]["launches_per_step"] = {"fused": 0, "unfused": 0}
    text = report.render([one, two], "t")
    assert "**2 of 2 rows are host smoke rows**" in text
    assert "The two ranges do not meet" not in text


# --------------------------------------------------------------------------
# The headline is computed per program, never over pre- and post-fix rows at once
# --------------------------------------------------------------------------

def _timed_row(case, digest, pairs, fused_ms, singles_ms, launches, prefer_gpu=True):
    import build_fused_timing_report as report  # noqa: PLC0415
    per = {leg: {"median_seconds_per_step": ms / 1e3}
           for leg, ms in (("fused", fused_ms), ("unfused", singles_ms), ("array", 2.0))}
    return {"case": case, "reportable": True, "prefer_gpu": prefer_gpu,
            "provenance": {"sha256": {"deposit_repair": digest}},
            "per_leg": per,
            "ratios": {"unfused_over_fused": singles_ms / fused_ms,
                       "array_over_fused": 2.0 / fused_ms},
            "floors": {"singles_are_the_same_table": True},
            "substitution": {"fused_and_singles_used_the_same_table": True,
                             "launches_per_step": {"fused": launches, "unfused": 4.0},
                             "tables_dispatched": ["triton"]},
            "plans": {"fused": {"arms": {f"s{i}": f"pair {i}" for i in range(pairs)}}},
            "_pairs": pairs}


def test_overlapping_ranges_never_print_do_not_meet_and_smoke_enters_no_headline(
        monkeypatch):
    import build_fused_timing_report as report  # noqa: PLC0415
    monkeypatch.setattr(report, "fused_pairs", lambda r: r["_pairs"])
    rows = [
        _timed_row("a", "c1d557040c6d", 1, 0.40, 0.45, 3.0),   # pre-fix, one pair
        _timed_row("b", "c1d557040c6d", 2, 2.60, 0.52, 2.0),   # pre-fix, two pairs
        _timed_row("c", "57767d76c3e6", 1, 0.40, 0.45, 3.0),   # post-fix, one pair
        _timed_row("d", "57767d76c3e6", 2, 0.20, 0.52, 2.0),   # post-fix: 0.100 ms/launch overlaps 0.133
        _timed_row("e", "57767d76c3e6", 2, 9.99, 0.10, 2.0, prefer_gpu=False),
    ]
    text = report.render(rows, "t")
    assert text.count("### Rows timed on deposit_repair") == 2, text
    post = text[text.index("deposit_repair `57767d76c3e6`"):]
    post = post[:post.index("###")] if "###" in post[3:] else post
    assert "do not meet" not in post.split("## ")[0], post
    assert "overlap" in post
    assert "enter no headline figure" in text
    # the smoke row's 0.01x never enters a median
    assert "over 2 cases" in post


# --------------------------------------------------------------------------
# The size sweep is rendered from its rows, and its sentences are derived
# --------------------------------------------------------------------------

def _sweep_row(cells, fused_ms, singles_ms, array_ms, pairs=2):
    return {"reportable": True, "grid_shape": [cells, 1, 1], "_table": "triton",
            "_case": "pml_2d", "_res": 0, "_pairs": pairs,
            "per_leg": {leg: {"median_seconds_per_step": ms / 1e3}
                        for leg, ms in (("fused", fused_ms), ("unfused", singles_ms),
                                        ("array", array_ms))},
            "substitution": {"fused_and_singles_used_the_same_table": True}}


def test_the_sweep_brackets_the_crossover_and_derives_the_flat_floor(monkeypatch):
    import build_fused_timing_report as report  # noqa: PLC0415
    monkeypatch.setattr(report, "fused_pairs", lambda r: r["_pairs"])
    after = [_sweep_row(24_000, 1.40, 0.50, 2.3), _sweep_row(1_000_000, 1.42, 0.63, 2.4),
             _sweep_row(2_400_000, 1.42, 1.04, 4.6), _sweep_row(5_400_000, 2.11, 2.28, 9.9)]
    before = [_sweep_row(24_000, 2.60, 0.52, 2.4)]
    text = "\n".join(report.sweep_lines(after, before))
    assert "crosses its singles between 2,400,000 and 5,400,000 cells" in text
    assert "flat at 1.40-1.42 ms from 24,000 to 2,400,000 cells" in text
    assert "| 2.600 / 0.200x |" in text, "the before column reads the older sweep's row"
    assert text.count("| — |") == 3, "sizes the older sweep never measured say so"


def test_a_sweep_that_never_wins_says_so_rather_than_inventing_a_crossover(monkeypatch):
    import build_fused_timing_report as report  # noqa: PLC0415
    monkeypatch.setattr(report, "fused_pairs", lambda r: r["_pairs"])
    after = [_sweep_row(24_000, 2.6, 0.5, 2.3), _sweep_row(5_400_000, 2.6, 2.2, 9.9)]
    text = "\n".join(report.sweep_lines(after, []))
    assert "behind its singles at every measured size, up to 5,400,000 cells" in text
    assert "crosses" not in text
