"""The component restriction's two counters, as the fused-timing bench records them.

``deposit_repair._PreparedCells`` counts, per source per save, whether a bracket was
narrowed to the component that source writes (``restricted_sources``) or kept every
target of its seam because the write path could not be established
(``restriction_fallbacks``). A fallback is a sound repair and is three entries where one
was possible, so a timing row that cannot show one is a row whose bracket cost cannot
be read: until 2026-09-20 it appeared only as ``linear_repairs`` per step reading 3,
which one source that fell back and three restricted sources both produce.

Pinned here, because this is the file that knows what the counters mean:

* the bench and the report between them name EVERY counter the cache keeps, so a
  counter added to the cache and not to the record fails here rather than silently;
* a snapshot/delta over a real bracket carries both counters, with a source that
  restricts and a source that falls back in ONE save;
* ABSENT IS NOT ZERO -- a leg summed from windows that never recorded the counters (a
  row written before 2026-09-20) gets no entry, not a zero nothing measured;
* the report says a fallback LOUDLY, says a clean restriction quietly, says nothing for
  an old row, and never changes the route verdict: a full bracket on the linear index
  is still the linear route.

``parity/meep_gpu/test_bench_fused_products_provenance.py`` holds the record's other
tests; its exact-dict expectations for a leg with no restriction counters are what the
third bullet keeps true.
"""

from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

import numpy
import pytest

from . import deposit_repair
from .test_deposit_repair import _build, _make_source

PARITY = str(pathlib.Path(__file__).resolve().parents[1] / "parity" / "meep_gpu")

ORIGIN = (0.0, 0.0, 0.0)


@pytest.fixture(scope="module")
def bench():
    if PARITY not in sys.path:
        sys.path.insert(0, PARITY)
    import bench_fused_products as module  # noqa: PLC0415
    return module


@pytest.fixture(scope="module")
def report():
    if PARITY not in sys.path:
        sys.path.insert(0, PARITY)
    import build_fused_timing_report as module  # noqa: PLC0415
    return module


class _Unread:
    """Publishes an index and a component, and is no engine source class: its write
    path was never read, so its bracket keeps all three targets."""

    field_type = "D"
    component = "Ez"

    def __init__(self):
        self._point_ix, self._point_iy, self._point_iz = (
            numpy.array([3]), numpy.array([4]), numpy.array([0]))


class _Inner:
    def run(self, *_args, **_kwargs) -> None:
        """The launch: nothing, which is all a counter test needs of it."""


def _driver(plans):
    return SimpleNamespace(_fast_path=SimpleNamespace(
        step_plan=SimpleNamespace(plans=plans)), _fast_path_stale=False)


def test_the_record_names_every_counter_the_cache_keeps(bench, report):
    kept = set(deposit_repair._PreparedCells.__slots__) - {"entries"}
    assert set(bench.REPAIR_COUNTERS) | set(bench.RESTRICTION_COUNTERS) == kept
    assert not set(bench.REPAIR_COUNTERS) & set(bench.RESTRICTION_COUNTERS)
    assert tuple(report.ROUTE_COUNTERS) == tuple(bench.REPAIR_COUNTERS)
    assert tuple(report.RESTRICTION_COUNTERS) == tuple(bench.RESTRICTION_COUNTERS)


def test_a_window_records_a_restricted_source_beside_one_that_fell_back(bench):
    """KNOWN VALUE: one engine source (restricted: 1 entry) and one unread source (all
    three targets: 3 entries) in ONE bracket, two steps."""
    _grid, fields, pml = _build()
    sources = [_make_source(fields.grid, "D", "Ex", ORIGIN, ORIGIN), _Unread()]
    leading = deposit_repair.LeadingRepairPlan(_Inner(), fields, pml, sources, "D")
    trailing = deposit_repair.TrailingRepairPlan("update_E", leading, fields, pml)
    driver = _driver({"step_D": leading, "update_E": trailing})
    before = bench.repair_snapshot(driver)
    steps = 2
    for _step in range(steps):
        leading.run()
        trailing.run()
    delta = bench.repair_delta(before, bench.repair_snapshot(driver))
    assert {name: delta[name] for name in bench.RESTRICTION_COUNTERS} == {
        "restricted_sources": steps, "restriction_fallbacks": steps}
    assert {name: delta[name] for name in bench.REPAIR_COUNTERS} == {
        "linear_saves": 4 * steps, "linear_repairs": 4 * steps,
        "cells_saves": 0, "cells_repairs": 0}
    assert delta["bracketed_slots"] == ["step_D", "update_E"]
    # Two such windows, summed into a leg.
    total = bench.repair_route_total([delta, delta])
    assert total["restricted_sources"] == total["restriction_fallbacks"] == 2 * steps
    assert total["linear_repairs"] == 8 * steps and total["windows"] == 2


def test_a_leg_whose_windows_never_recorded_the_counters_gets_no_entry(bench):
    """ABSENT IS NOT ZERO. A window written before the counters existed says nothing
    about fallbacks, and a summed zero would read as "none happened"."""
    old = {"linear_saves": 6, "linear_repairs": 6, "cells_saves": 0, "cells_repairs": 0,
           "bracketed_slots": ["step_D", "update_E"], "plan_changed_inside_window": False}
    total = bench.repair_route_total([old, old])
    assert not set(bench.RESTRICTION_COUNTERS) & set(total), total
    new = dict(old, restricted_sources=2, restriction_fallbacks=0)
    total = bench.repair_route_total([new, new])
    assert (total["restricted_sources"], total["restriction_fallbacks"]) == (4, 0)
    assert not set(bench.RESTRICTION_COUNTERS) & set(bench.repair_route_total([]))


def _row(route):
    return {"case": "pml_2d", "per_leg": {"fused": {"deposit_repair_route": route}}}


def _route(**restriction):
    return dict({"linear_saves": 6, "linear_repairs": 6, "cells_saves": 0,
                 "cells_repairs": 0, "bracketed_slots": ["step_D", "update_E"],
                 "plan_changed_inside_window": False, "windows": 6}, **restriction)


def test_the_report_says_a_fallback_loudly_and_never_changes_the_route_verdict(report):
    old = report.route_cell(_row(_route()))
    clean = report.route_cell(_row(_route(restricted_sources=6, restriction_fallbacks=0)))
    fell = report.route_cell(_row(_route(restricted_sources=4, restriction_fallbacks=2)))
    assert "source-saves" not in old.lower(), old
    assert clean == old.replace(
        ")", "), 6 source-saves restricted to the written component", 1), (old, clean)
    assert "**2 OF 6 SOURCE-SAVES KEPT THE FULL THREE-TARGET BRACKET**" in fell, fell
    for route in (_route(), _route(restricted_sources=6, restriction_fallbacks=0),
                  _route(restricted_sources=0, restriction_fallbacks=6)):
        assert report.route_verdict(route) == "linear", route
    # BOTH ZERO ON A BRACKET THAT RAN is a module that kept neither counter (the bench
    # reads a missing counter as 0): every source of a save is counted one way or the
    # other, so there is nothing to say -- and "0 restricted" would be a false reading.
    unmeasured = report.route_cell(
        _row(_route(restricted_sources=0, restriction_fallbacks=0)))
    assert unmeasured == old, (unmeasured, old)
