"""Host tests for the timing-record cutter (``cut_timing_record.py``).

Written before the cutter. Synthetic ``rows.jsonl`` trees under ``tmp_path`` carry
exactly the fields the cutter reads, in the two row spellings on disk: the fused-product
bench's (one row per case, three legs inside it) and the repair Phase 0 harness's (one
row per LEG). Groups, in the repository's order for a new decision engine:

* KNOWN VALUE -- medians, ratios and spreads of a hand-built tree, and the consult
  vetoing exactly the product the tree measured slower;
* DEGENERATE -- every floor, the same-table control, a verdict that is not a timing,
  a host smoke row, a contended row (joined by device UUID, not by the row's own
  over-reporting field), a row with no digest, duplicate lines, too few rows, nothing
  admitted at all;
* ONE RECORD, ONE PROGRAM -- rows on two deposit-repair digests, or on one digest with
  differing restriction counters, REFUSE to pool; so do two fused-pair emitters;
* SCALING -- bands are runs of measured sizes with one verdict, a verdict change
  between two sizes is an explicit unmeasured gap, and nothing is extrapolated;
* SHAPE / SERIALIZATION -- deterministic bytes, no wall clock, ``--check``;
* REALISTIC -- the shipped records recut byte-for-byte from the rows on disk.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_API = os.path.abspath(os.path.join(HERE, "..", ".."))
for _path in (HERE, REPO_API):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import cut_timing_record as cut  # noqa: E402
from meep_gpu import dispatch_preference as dp  # noqa: E402

SHA_A = "a1" * 32
SHA_B = "b2" * 32
PAIRS_SHA = "c3" * 32
BENCH_SHA = "d4" * 32

PINNED_GPU = "GPU-00000000-aaaa"
OTHER_GPU = "GPU-11111111-bbbb"
GPUS = f"0, {PINNED_GPU}, 3 MiB, 0 %\n1, {OTHER_GPU}, 3 MiB, 0 %"
OWN_PID = 4507

FLOORS = ("substitution_proved", "bit_identical", "moved_words_positive",
          "non_finite_zero", "no_compiles_inside_timed_region", "spread_within_gate",
          "singles_are_the_same_table")

TWO_PAIR = {"fused pair B": ["step_B", "update_H"], "fused pair D": ["step_D", "update_E"]}
ONE_PAIR = {"fused pair B": ["step_B", "update_H"]}
SINGLES = {"step_B": "PML", "update_H": "ordinary", "step_D": "PML",
           "update_E": "ordinary"}


def box(compute_apps=""):
    return {"CUDA_VISIBLE_DEVICES": "0", "gpus": GPUS, "compute_apps": compute_apps,
            "pid": OWN_PID, "utc": "2026-09-19T13:02:20Z",
            # the field the bench fills is index-blind and over-reports; the cutter
            # must not read it
            "foreign_compute_apps_on_pinned_gpu": [f"{OTHER_GPU}, 999, 5000 MiB"]}


def route(bracketed_slots, steps, per_step=3, restricted=None, fallbacks=None):
    block = {"bracketed_slots": list(bracketed_slots), "cells_saves": 0,
             "cells_repairs": 0, "plan_changed_inside_window": False, "windows": 6,
             "linear_saves": per_step * steps if bracketed_slots else 0,
             "linear_repairs": per_step * steps if bracketed_slots else 0}
    if restricted is not None:
        block["restricted_sources"] = restricted
        block["restriction_fallbacks"] = fallbacks or 0
    return block


def bench_row(case="pml_2d", table="triton", grid=(200, 120, 1), fused_ms=1.4,
              base_ms=0.5, welded=None, bracketed_slots=("step_D", "update_E"),
              sha=SHA_A, utc="2026-09-19T13:02:20Z", fused_spread=0.02,
              base_spread=0.01, per_step=3, restricted=None, fallbacks=None,
              complex_storage=False, dimensions=2, susceptibilities=0,
              pairs_sha=PAIRS_SHA, bench_sha=BENCH_SHA, polarization_sha=None,
              unfused=None, tables_dispatched=None):
    welded = dict(TWO_PAIR if welded is None else welded)
    novel = {slot: label for label, slots in welded.items() for slot in slots}
    unfused = dict(SINGLES if unfused is None else unfused)
    steps = 6 * 1000
    digests = {"deposit_repair": sha, "fused_pairs": pairs_sha, "bench": bench_sha,
               "fastpath": "e5" * 32, "driver": "f6" * 32}
    if polarization_sha:
        digests["fused_polarization_pair"] = polarization_sha
    if sha is None:
        digests.pop("deposit_repair")
    shape = {"dimensions": dimensions, "grid_shape": list(grid),
             "complex_storage": complex_storage, "susceptibilities": susceptibilities}
    plan = {"run_shape": shape, "arms": dict(unfused, **novel),
            "environment": {"backend": "cupy", "backend_version": "13.5.1",
                            "triton": "3.1.0",
                            "device": {"name": "EXAMPLE GPU", "compute_capability": "8.6",
                                       "cuda_driver": 12030, "cuda_runtime": 11080}},
            "subnormal": {"gate": {"policy_in_force": "keep"}}}
    return {
        "case": case, "drive_table": table, "utc": utc, "verdict": "TIMED",
        "reportable": True, "prefer_gpu": True, "grid_shape": list(grid),
        "floors": {name: True for name in FLOORS},
        "box_before": box(), "box_after": box(),
        "per_leg": {
            "fused": {"median_seconds_per_step": fused_ms / 1e3, "spread": fused_spread,
                      "steps_per_window": 1000, "windows": 6,
                      "deposit_repair_route": route(bracketed_slots, steps, per_step,
                                                    restricted, fallbacks)},
            "unfused": {"median_seconds_per_step": base_ms / 1e3, "spread": base_spread,
                        "steps_per_window": 1000, "windows": 6,
                        "deposit_repair_route": route((), steps)},
        },
        "plans": {"fused": plan, "unfused": plan},
        "provenance": {"sha256": digests},
        "ratios": {"unfused_over_fused": base_ms / fused_ms},
        "substitution": {"welded_arms": welded, "novel_slots": novel,
                         "unfused_arms": {slot: unfused[slot] for slot in novel},
                         "fused_and_singles_used_the_same_table": True,
                         "tables_dispatched": ([table] if tables_dispatched is None
                                               else list(tables_dispatched)),
                         "proved": True},
    }


def phase0_rows(case="pml_2d", table="triton", grid=(200, 120, 1), today_ms=1.4,
                restricted_ms=0.9, singles_ms=0.5, sha=SHA_A,
                utc="2026-09-19T23:13:06Z"):
    """One Phase 0 point: five rows, one per leg, in the harness's own spelling."""
    out = []
    for leg, ms, verdict, per_step, applied in (
            ("today", today_ms, "TIMED", 3, 0),
            ("ceiling_r0", 0.4, "CEILING-NOT-A-TIMING", 0, 0),
            ("raw_r_a1", 0.7, "CEILING-NOT-A-TIMING", 3, 0),
            ("restricted_g", restricted_ms, "TIMED", 1, 12000),
            ("singles", singles_ms, "TIMED", 0, 0)):
        template = bench_row(case=case, table=table, grid=grid, sha=sha, utc=utc)
        slots = ("step_D", "update_E") if per_step else ()
        row = {key: template[key] for key in (
            "case", "drive_table", "utc", "reportable", "prefer_gpu", "grid_shape",
            "floors", "box_before", "plans", "provenance", "substitution")}
        row["provenance"]["sha256"]["phase0_harness"] = "07" * 32
        row.update({
            "leg": leg, "verdict": verdict, "phase": "0.4", "res": 20,
            "reportable": verdict == "TIMED",
            "cells": grid[0] * grid[1] * grid[2],
            "all_legs_ms_per_step": {"today": today_ms, "singles": singles_ms},
            "all_legs_spread": {"today": 0.02, "singles": 0.01},
            "patch_counters": {"restriction_applied": applied, "restriction_fell_back": 0,
                               "save_calls": 6000, "apply_calls": 6000},
            "per_leg": {"median_seconds_per_step": ms / 1e3, "median_ms_per_step": ms,
                        "spread": 0.02 if leg != "singles" else 0.01,
                        "steps_per_window": 1000, "windows": 6,
                        "deposit_repair_route": route(slots, 6000, per_step or 3)},
        })
        if verdict != "TIMED":
            row["floors"] = dict(row["floors"], bit_identical=False)
        out.append(row)
    return out


def write(tmp_path, relative, rows):
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def cut_tree(tmp_path, roots=("tree",), table="triton", **options):
    return cut.cut(table, [str(tmp_path / root) for root in roots],
                   base=str(tmp_path), **options)


def ask_cells(record, cells, product="fused pair D", bracketed=True,
              displaces=("PML", "ordinary")):
    """The outcome a run of ``cells`` gets from this record, asked as its own tree."""
    return dp.consult(
        record, table=record["table"], candidate=product, displaces=displaces,
        run_shape={"dimensions": 2, "grid_shape": [cells, 1, 1]}, bracketed=bracketed,
        mode="measured", repair_route=record["route"],
        subject_sha256=dp.compared_emitters(record)).outcome


def by_band(record, product=None, bracketed=None):
    """The entries of one class in CELL order; ``entries`` is in key-string order."""
    return sorted(entries(record, product, bracketed),
                  key=lambda entry: entry["band"]["lo_cells"])


def entries(record, product=None, bracketed=None):
    return [entry for _key, entry in sorted(record["keys"].items())
            if (product is None or entry["product"] == product)
            and (bracketed is None or entry["bracketed"] is bracketed)]


def standard_tree(tmp_path):
    """Two rows of a two-pair bracketed case and two of a one-pair clean case."""
    write(tmp_path, "tree/triton/rows.jsonl", [
        bench_row(case="pml_2d", fused_ms=1.40, base_ms=0.50),
        bench_row(case="pml_2d_again", fused_ms=1.60, base_ms=0.52, fused_spread=0.03,
                  utc="2026-09-19T14:00:00Z"),
        bench_row(case="offdiag_2d", welded=ONE_PAIR, bracketed_slots=(),
                  fused_ms=0.45, base_ms=0.49),
        bench_row(case="offdiag_2d_again", welded=ONE_PAIR, bracketed_slots=(),
                  fused_ms=0.46, base_ms=0.49),
    ])


# ---------------------------------------------------------------------------
# Known value
# ---------------------------------------------------------------------------

def test_a_hand_built_tree_cuts_to_its_medians_ratios_and_spreads(tmp_path):
    standard_tree(tmp_path)
    record = cut_tree(tmp_path)
    dp.validate(record, expect_table="triton")
    slow, = entries(record, "fused pair D")
    assert slow["status"] == "measured" and slow["bracketed"] is True
    assert slow["rows"] == 2
    assert slow["band"] == {"lo_cells": 24000, "hi_cells": 24000}
    assert slow["slots"] == ["step_D", "update_E"]
    assert slow["displaces"] == ["PML", "ordinary"]
    assert slow["fused_ms_per_step"] == pytest.approx(1.50)
    assert slow["baseline_ms_per_step"] == pytest.approx(0.51)
    assert slow["ratio_min"] == pytest.approx(0.52 / 1.60, abs=1e-6)
    assert slow["ratio_max"] == pytest.approx(0.50 / 1.40, abs=1e-6)
    assert slow["ratio"] == pytest.approx((0.325 + 0.357143) / 2, abs=1e-5)
    assert slow["fused_spread"] == pytest.approx(0.03)
    assert slow["baseline_spread"] == pytest.approx(0.01)
    assert slow["slower_beyond_spread"] is True
    assert [row["case"] for row in slow["measurements"]] == ["pml_2d", "pml_2d_again"]
    assert slow["measurements"][0]["artifact"] == "tree/triton/rows.jsonl"
    assert slow["measurements"][0]["line"] == 1
    assert slow["measurements"][0]["grid_shape"] == [200, 120, 1]


def test_the_cut_record_vetoes_exactly_the_slower_product(tmp_path):
    standard_tree(tmp_path)
    record = cut_tree(tmp_path)
    shape = {"dimensions": 2, "grid_shape": [200, 120, 1], "complex_storage": False,
             "susceptibilities": 0}
    asked = dict(table="triton", displaces=("PML", "ordinary"), run_shape=shape,
                 mode="measured", repair_route=record["route"],
                 subject_sha256=dp.compared_emitters(record))
    assert dp.consult(record, candidate="fused pair D", bracketed=True, **asked).vetoed
    clean = dp.consult(record, candidate="fused pair B", bracketed=False, **asked)
    assert clean.outcome == "not_vetoed"
    assert not dp.consult(record, candidate="fused pair D", bracketed=True,
                          **dict(asked, mode="span")).vetoed


def test_a_two_pair_row_prices_only_the_product_that_carries_the_bracket(tmp_path):
    """The clean partner of a bracketed pair is not priced by that row.

    A two-pair row's whole-step loss follows the bracket. Attributing it to the clean
    product too would put a 0.36x row and a 1.09x row under one key.
    """
    standard_tree(tmp_path)
    record = cut_tree(tmp_path)
    clean, = entries(record, "fused pair B")
    assert clean["bracketed"] is False
    assert [row["case"] for row in clean["measurements"]] == ["offdiag_2d",
                                                              "offdiag_2d_again"]
    assert all(row["co_fused"] == [] for row in clean["measurements"])
    slow, = entries(record, "fused pair D")
    assert slow["measurements"][0]["co_fused"] == [
        {"product": "fused pair B", "bracketed": False}]
    assert entries(record, "fused pair B", bracketed=True) == []
    assert record["attribution"]["not_attributed"] == 2


def test_a_plan_of_two_clean_products_prices_neither_of_them(tmp_path):
    """One number cannot price two products, and a veto must not be a guess.

    A row times a PLAN. When two fused products sit on it and neither carries the
    bracket, the whole-plan ratio is the two together: writing it into a key for each
    lets a partner that loses veto a product nothing was ever measured to be slower
    than -- the one error a fail-closed veto may not make. So a clean product is
    priced only from rows where it is the plan's SOLE fused product.
    """
    write(tmp_path, "tree/rows.jsonl", [
        # two pairs, NEITHER bracketed, and the plan as a whole loses
        bench_row(case="both_clean", bracketed_slots=(), fused_ms=1.4, base_ms=0.5),
        bench_row(case="both_clean_again", bracketed_slots=(), fused_ms=1.5,
                  base_ms=0.5, utc="2026-09-19T14:00:00Z"),
        # and the same two products, one plan each, are priced as themselves
        bench_row(case="sole_b", welded=ONE_PAIR, bracketed_slots=(), fused_ms=0.45,
                  base_ms=0.49),
        bench_row(case="sole_b_again", welded=ONE_PAIR, bracketed_slots=(),
                  fused_ms=0.46, base_ms=0.49)])
    record = cut_tree(tmp_path)
    clean, = entries(record, "fused pair B")
    assert [row["case"] for row in clean["measurements"]] == ["sole_b", "sole_b_again"]
    assert clean["slower_beyond_spread"] is False
    assert entries(record, "fused pair D") == []
    # two rows times two products, none of them attributed, and the record says why
    assert record["attribution"]["not_attributed"] == 4
    assert "sole fused product" in record["attribution"]["why"]


def test_a_two_product_plan_prices_only_a_lone_bracketed_product(tmp_path):
    """Two bracketed products in one plan price neither: the key has no co-fused axis
    and the bracket cannot be attributed to one of them."""
    both = bench_row(case="both_bracketed",
                     bracketed_slots=("step_B", "update_H", "step_D", "update_E"))
    write(tmp_path, "tree/rows.jsonl", [both, bench_row(case="second")])
    record = cut_tree(tmp_path)
    priced = {row["case"] for entry in record["keys"].values()
              for row in entry["measurements"]}
    assert priced == {"second"}
    assert record["attribution"]["not_attributed"] == 3       # 2 + the clean partner


def test_the_record_carries_route_host_policy_digests_and_artifacts(tmp_path):
    standard_tree(tmp_path)
    record = cut_tree(tmp_path)
    assert record["schema"] == dp.SCHEMA and record["table"] == "triton"
    assert record["route"] == {
        "deposit_repair_sha256": SHA_A, "index": "linear",
        "repairs_per_bracketed_step": 3.0, "restricted": False, "fell_back": False,
        "restriction_counters_recorded_on": "0 of 2 bracketed rows",
        "id": f"{SHA_A[:12]}:linear:3-per-step:full"}
    assert record["policy"] == ["keep"]
    assert record["host"] == [{"backend": "cupy", "backend_version": "13.5.1",
                               "compute_capability": "8.6", "cuda_driver": 12030,
                               "cuda_runtime": 11080, "device": "EXAMPLE GPU",
                               "triton": "3.1.0"}]
    assert record["pinned_sources"]["deposit_repair"] == [SHA_A]
    assert record["pinned_sources"]["fused_pairs"] == [PAIRS_SHA]
    assert record["pinned_sources"]["fused_polarization_pair"] == ["not recorded"]
    assert record["pinned_sources"]["bench"] == [BENCH_SHA]
    files = record["inputs"]["files"]
    assert [item["path"] for item in files] == ["tree/triton/rows.jsonl"]
    assert files[0]["rows"] == 4 and len(files[0]["sha256"]) == 64
    assert record["rows"] == {"read": 4, "duplicates": 0, "baseline_legs": 0,
                              "admitted": 4, "excluded": 0}
    assert record["rules"]["min_rows_per_key"] == 2
    assert record["recorded_utc"] == "2026-09-19T14:00:00Z"   # the newest row's


# ---------------------------------------------------------------------------
# Degenerate: what is refused admission, and named
# ---------------------------------------------------------------------------

def excluded_reasons(record):
    return sorted(item["reason"] for item in record["excluded"])


@pytest.mark.parametrize("floor", FLOORS)
def test_a_row_failing_any_floor_is_not_admitted(tmp_path, floor):
    bad = bench_row(case="bad")
    bad["floors"][floor] = False
    bad["reportable"] = False
    bad["verdict"] = "TIMED-NOT-REPORTABLE"
    standard = [bench_row(), bench_row(case="second")]
    write(tmp_path, "tree/rows.jsonl", standard + [bad])
    record = cut_tree(tmp_path)
    assert record["rows"]["admitted"] == 2
    reason, = excluded_reasons(record)
    assert floor in reason
    assert [item["case"] for item in record["excluded"]] == ["bad"]


def test_a_control_on_another_table_is_refused_even_if_the_row_says_timed(tmp_path):
    """The same-table control is a floor of its own, checked where it is measured."""
    bad = bench_row(case="cross_table")
    bad["substitution"]["fused_and_singles_used_the_same_table"] = False
    write(tmp_path, "tree/rows.jsonl", [bench_row(), bench_row(case="second"), bad])
    record = cut_tree(tmp_path)
    assert "same kernel table" in excluded_reasons(record)[0]


def test_a_floor_that_was_never_recorded_is_not_a_floor_that_passed(tmp_path):
    bad = bench_row(case="old")
    bad["floors"].pop("singles_are_the_same_table")
    write(tmp_path, "tree/rows.jsonl", [bench_row(), bench_row(case="second"), bad])
    assert "not recorded" in excluded_reasons(cut_tree(tmp_path))[0]


@pytest.mark.parametrize("verdict", ["SKIP-NOT-THIS-LEG", "ERROR", "HARNESS-FAILURE",
                                     "SKIP-TOO-LARGE", "CEILING-NOT-A-TIMING"])
def test_a_verdict_that_is_not_a_timing_is_not_admitted(tmp_path, verdict):
    skipped = {"case": "skipped", "drive_table": "triton", "verdict": verdict,
               "utc": "2026-09-19T13:00:00Z", "prefer_gpu": True}
    write(tmp_path, "tree/rows.jsonl", [bench_row(), bench_row(case="second"), skipped])
    record = cut_tree(tmp_path)
    assert excluded_reasons(record) == [f"verdict {verdict}"]


def test_a_host_smoke_row_is_not_admitted(tmp_path):
    smoke = bench_row(case="smoke")
    smoke["prefer_gpu"] = False
    write(tmp_path, "tree/rows.jsonl", [bench_row(), bench_row(case="second"), smoke])
    assert "smoke" in excluded_reasons(cut_tree(tmp_path))[0]


def test_contention_is_joined_by_device_uuid(tmp_path):
    def with_apps(case, before="", after=""):
        row = bench_row(case=case)
        row["box_before"], row["box_after"] = box(before), box(after)
        return row

    rows = [
        bench_row(case="quiet"),
        # a neighbour on ANOTHER device is not contention, whatever the row's own
        # index-blind field says
        with_apps("neighbour_elsewhere", before=f"{OTHER_GPU}, 999, 5000 MiB",
                  after=f"{OTHER_GPU}, 999, 5000 MiB"),
        # the bench's own process on the pinned device is not a neighbour
        with_apps("own_process", after=f"{PINNED_GPU}, {OWN_PID}, 282 MiB"),
        with_apps("neighbour_before", before=f"{PINNED_GPU}, 999, 100 MiB"),
        with_apps("neighbour_after", after=f"{OTHER_GPU}, 5, 1 MiB\n"
                                           f"{PINNED_GPU}, 999, 100 MiB"),
    ]
    write(tmp_path, "tree/rows.jsonl", rows)
    record = cut_tree(tmp_path)
    assert sorted(item["case"] for item in record["excluded"]) == [
        "neighbour_after", "neighbour_before"]
    assert all("contended" in reason for reason in excluded_reasons(record))
    assert cut.contended(rows[0]) is None and cut.contended(rows[1]) is None
    assert cut.contended(rows[2]) is None
    assert "before" in cut.contended(rows[3]) and "after" in cut.contended(rows[4])


def test_a_row_whose_device_cannot_be_identified_is_not_admitted(tmp_path):
    blind = bench_row(case="blind")
    blind["box_before"] = {"pid": OWN_PID}
    write(tmp_path, "tree/rows.jsonl", [bench_row(), bench_row(case="second"), blind])
    assert "contention cannot be read" in excluded_reasons(cut_tree(tmp_path))[0]


def test_a_row_with_no_after_probe_is_counted_and_can_be_refused(tmp_path):
    half = bench_row(case="half_probed")
    half.pop("box_after")
    write(tmp_path, "tree/rows.jsonl", [bench_row(), half])
    record = cut_tree(tmp_path)
    assert record["rows"]["admitted"] == 2
    assert record["contention"]["admitted_without_an_after_probe"] == 1
    strict = cut_tree(tmp_path, require_box_after=True)
    assert strict["rows"]["admitted"] == 1
    assert "after-probe" in excluded_reasons(strict)[0]


def test_a_row_that_cannot_say_which_repair_code_it_timed_is_not_admitted(tmp_path):
    write(tmp_path, "tree/rows.jsonl", [bench_row(), bench_row(case="second"),
                                        bench_row(case="undated", sha=None)])
    assert "deposit_repair digest" in excluded_reasons(cut_tree(tmp_path))[0]


def test_a_row_of_another_table_is_left_out_by_name(tmp_path):
    write(tmp_path, "tree/rows.jsonl", [bench_row(), bench_row(case="second"),
                                        bench_row(case="other", table="cuda")])
    record = cut_tree(tmp_path)
    assert excluded_reasons(record) == ["a row of the 'cuda' table"]
    assert record["rows"]["admitted"] == 2


def test_a_plan_that_dispatched_two_tables_is_left_out_by_name(tmp_path):
    """The drive table is what was ASKED for; ``tables_dispatched`` is what SERVED.

    A plan whose slots were served by two tables is not a measurement of either one's
    plan: a term common to both legs moves the ratio toward 1, so such a row cannot
    manufacture a veto -- but the ms/step and the ratio it records are not this
    table's, and a same-table comparison is exactly what the admission rule claims.
    """
    write(tmp_path, "tree/rows.jsonl", [
        bench_row(), bench_row(case="second"),
        bench_row(case="mixed", tables_dispatched=["cuda", "triton"])])
    record = cut_tree(tmp_path)
    assert excluded_reasons(record) == [
        "the plan's slots were served by ['cuda', 'triton'], not by the 'triton' "
        "table alone"]
    assert record["rows"]["admitted"] == 2


def test_a_row_that_does_not_say_which_tables_served_it_is_not_admitted(tmp_path):
    silent = bench_row(case="silent")
    silent["substitution"].pop("tables_dispatched")
    write(tmp_path, "tree/rows.jsonl", [bench_row(), bench_row(case="second"), silent])
    record = cut_tree(tmp_path)
    assert excluded_reasons(record) == [
        "the row does not record which kernel tables its plans dispatched"]


def test_a_byte_identical_line_in_two_trees_is_one_measurement(tmp_path):
    """A merged tree repeats the lines it kept; the minimum-rows floor must not see two."""
    row = bench_row()
    write(tmp_path, "first/rows.jsonl", [row])
    write(tmp_path, "merged/rows.jsonl", [row, bench_row(case="second")])
    record = cut_tree(tmp_path, roots=("first", "merged"))
    assert record["rows"] == {"read": 3, "duplicates": 1, "baseline_legs": 0,
                              "admitted": 2, "excluded": 0}
    slow, = entries(record, "fused pair D")
    assert slow["rows"] == 2
    assert slow["measurements"][0]["artifact"] == "first/rows.jsonl"


def test_too_few_rows_writes_the_key_as_unmeasured_with_the_reason(tmp_path):
    write(tmp_path, "tree/rows.jsonl", [bench_row()])
    record = cut_tree(tmp_path)
    lonely, = entries(record, "fused pair D")
    assert lonely["status"] == "unmeasured"
    assert lonely["reason"] == ("no size in it carries 2 rows, so none can set a "
                                "band edge: 24,000 (1 row)")
    assert lonely["ratio_max"] == pytest.approx(0.357143)   # the row still travels
    shape = {"dimensions": 2, "grid_shape": [200, 120, 1]}
    verdict = dp.consult(record, table="triton", candidate="fused pair D",
                         displaces=("PML", "ordinary"), run_shape=shape, bracketed=True,
                         mode="measured", repair_route=record["route"],
                         subject_sha256=dp.compared_emitters(record))
    assert verdict.outcome == "no_record" and verdict.reason == "key_unmeasured"
    assert entries(cut_tree(tmp_path, min_rows=1), "fused pair D")[0]["status"] == "measured"


def test_rows_that_disagree_on_what_the_product_displaces_are_not_pooled(tmp_path):
    other = dict(SINGLES, update_E="off-diagonal")
    write(tmp_path, "tree/rows.jsonl", [bench_row(), bench_row(case="second",
                                                               unfused=other)])
    entry, = entries(cut_tree(tmp_path), "fused pair D")
    assert entry["status"] == "unmeasured"
    assert "displaces" in entry["reason"]


def test_nothing_admitted_refuses_by_name(tmp_path):
    bad = bench_row()
    bad["prefer_gpu"] = False
    write(tmp_path, "tree/rows.jsonl", [bad])
    with pytest.raises(cut.CutRefused) as refusal:
        cut_tree(tmp_path)
    assert refusal.value.code == "nothing_admitted"
    (tmp_path / "empty").mkdir()
    with pytest.raises(cut.CutRefused) as refusal:
        cut_tree(tmp_path, roots=("empty",))
    assert refusal.value.code == "no_rows"
    with pytest.raises(cut.CutRefused) as refusal:
        cut_tree(tmp_path, roots=("absent",))
    assert refusal.value.code == "artifacts_absent"


# ---------------------------------------------------------------------------
# One record, one program
# ---------------------------------------------------------------------------

def test_rows_on_two_repair_digests_refuse_to_pool(tmp_path):
    write(tmp_path, "tree/rows.jsonl", [bench_row(), bench_row(case="after", sha=SHA_B)])
    with pytest.raises(cut.CutRefused) as refusal:
        cut_tree(tmp_path)
    assert refusal.value.code == "mixed_repair_route"
    assert SHA_A[:12] in refusal.value.detail and SHA_B[:12] in refusal.value.detail


def test_rows_whose_restriction_counters_differ_refuse_to_pool(tmp_path):
    """One digest, two brackets: three entries a step against one."""
    write(tmp_path, "tree/rows.jsonl", [
        bench_row(), bench_row(case="restricted", per_step=1, restricted=6000)])
    with pytest.raises(cut.CutRefused) as refusal:
        cut_tree(tmp_path)
    assert refusal.value.code == "mixed_repair_route"
    assert "3-per-step:full" in refusal.value.detail
    assert "1-per-step:restricted" in refusal.value.detail


def test_a_fallback_inside_a_restricted_run_is_its_own_program(tmp_path):
    write(tmp_path, "tree/rows.jsonl", [
        bench_row(per_step=1, restricted=6000),
        bench_row(case="fell_back", per_step=1, restricted=5000, fallbacks=1000)])
    with pytest.raises(cut.CutRefused) as refusal:
        cut_tree(tmp_path)
    assert refusal.value.code == "mixed_repair_route"
    assert "fell-back" in refusal.value.detail


def test_one_route_can_be_selected_and_the_rest_is_left_out_by_name(tmp_path):
    write(tmp_path, "tree/rows.jsonl", [
        bench_row(), bench_row(case="second"),
        bench_row(case="restricted", per_step=1, restricted=6000)])
    wanted = f"{SHA_A[:12]}:linear:3-per-step:full"
    record = cut_tree(tmp_path, select_route=wanted)
    assert record["route"]["id"] == wanted and record["rows"]["admitted"] == 2
    assert excluded_reasons(record) == [
        f"another repair route: {SHA_A[:12]}:linear:1-per-step:restricted"]
    other = cut_tree(tmp_path, select_route=f"{SHA_A[:12]}:linear:1-per-step:restricted",
                     min_rows=1)
    assert other["route"]["restricted"] is True
    assert other["route"]["repairs_per_bracketed_step"] == 1.0
    assert other["route"]["restriction_counters_recorded_on"] == "1 of 1 bracketed rows"
    with pytest.raises(cut.CutRefused) as refusal:
        cut_tree(tmp_path, select_route="0123456789ab:linear:3-per-step:full")
    assert refusal.value.code == "nothing_admitted"


def test_an_unbracketed_row_pools_with_any_route_of_its_own_digest(tmp_path):
    write(tmp_path, "tree/rows.jsonl", [
        bench_row(), bench_row(case="second"),
        bench_row(case="clean", welded=ONE_PAIR, bracketed_slots=()),
        bench_row(case="clean_other_digest", welded=ONE_PAIR, bracketed_slots=(),
                  sha=SHA_B)])
    with pytest.raises(cut.CutRefused) as refusal:
        cut_tree(tmp_path)
    assert refusal.value.code == "mixed_repair_route"
    wanted = f"{SHA_A[:12]}:linear:3-per-step:full"
    record = cut_tree(tmp_path, select_route=wanted)
    assert record["rows"]["admitted"] == 3
    assert "another deposit_repair digest" in excluded_reasons(record)[0]


def test_two_fused_pair_emitters_refuse_to_pool_and_two_benches_do_not(tmp_path):
    write(tmp_path, "tree/rows.jsonl", [bench_row(),
                                        bench_row(case="second", bench_sha="99" * 32)])
    record = cut_tree(tmp_path)
    assert record["pinned_sources"]["bench"] == sorted([BENCH_SHA, "99" * 32])
    write(tmp_path, "tree/rows.jsonl", [bench_row(),
                                        bench_row(case="second", pairs_sha="88" * 32)])
    with pytest.raises(cut.CutRefused) as refusal:
        cut_tree(tmp_path)
    assert refusal.value.code == "mixed_subject_digest"
    assert "fused_pairs" in refusal.value.detail


def test_an_emitter_digest_that_was_not_recorded_does_not_pool_with_one_that_was(
        tmp_path):
    """"Not recorded" is a VALUE, not a digest.

    A row that does not pin an emitter cannot say which one it timed, so it is not
    evidence that it timed the one its neighbour pinned. Admitting the straddle is a
    decision, taken by name on the command line and written into the record.
    """
    write(tmp_path, "tree/rows.jsonl", [
        bench_row(table="cuda"),
        bench_row(table="cuda", case="second", polarization_sha="77" * 32)])
    with pytest.raises(cut.CutRefused) as refusal:
        cut_tree(tmp_path, table="cuda")
    assert refusal.value.code == "unpinned_subject_source"
    assert "fused_polarization_pair" in refusal.value.detail
    assert "--allow-unpinned-subject" in refusal.value.detail
    record = cut_tree(tmp_path, table="cuda",
                      allow_unpinned_subject=["fused_polarization_pair"])
    barrier = record["subject_barrier"]
    assert barrier["sources"] == ["fused_pairs", "fused_polarization_pair"]
    assert barrier["digests"]["fused_polarization_pair"] == sorted(
        ["77" * 32, "not recorded"])
    assert barrier["rows_not_recording"]["fused_polarization_pair"] == 1
    assert barrier["allowed_unpinned"] == ["fused_polarization_pair"]
    assert record["cut"]["allow_unpinned_subject"] == ["fused_polarization_pair"]
    # and the consult will not compare what the record could not pin
    assert dp.compared_emitters(record) == {"fused_pairs": PAIRS_SHA}
    # two RECORDED digests still refuse, and the flag does not excuse them
    write(tmp_path, "tree/rows.jsonl", [
        bench_row(table="cuda", polarization_sha="66" * 32),
        bench_row(table="cuda", case="second", polarization_sha="77" * 32)])
    with pytest.raises(cut.CutRefused) as refusal:
        cut_tree(tmp_path, table="cuda",
                 allow_unpinned_subject=["fused_polarization_pair"])
    assert refusal.value.code == "mixed_subject_digest"


def test_the_subject_barrier_is_scoped_to_the_table_that_ran(tmp_path):
    """``fused_polarization_pair`` is a CUDA module: nothing under ``triton_kernels/``
    names it, so a Triton record must not compare a file its rows never ran."""
    assert dp.SUBJECT_SOURCES_BY_TABLE["triton"] == ("fused_pairs",)
    write(tmp_path, "tree/rows.jsonl", [
        bench_row(), bench_row(case="second", polarization_sha="77" * 32)])
    record = cut_tree(tmp_path)                       # the triton table: no refusal
    assert record["subject_barrier"]["sources"] == ["fused_pairs"]
    assert "fused_polarization_pair" not in record["subject_barrier"]["digests"]
    # it is still RECORDED, because what a row pinned is provenance either way
    assert record["pinned_sources"]["fused_polarization_pair"] == sorted(
        ["77" * 32, "not recorded"])


def test_the_record_says_which_emitters_no_row_could_pin(tmp_path):
    """The per-product kernel text is pinned by no row, and the record says so rather
    than leaving a reader to assume the barrier covers it."""
    standard_tree(tmp_path)
    record = cut_tree(tmp_path)
    assert "no timing row pins" in record["subject_barrier"]["cannot_cover"]
    assert "fused_electric_pair" in record["subject_barrier"]["cannot_cover"]


def test_two_policies_or_two_devices_refuse_to_pool(tmp_path):
    flushed = bench_row(case="flushed")
    flushed["plans"]["fused"]["subnormal"]["gate"]["policy_in_force"] = "flush"
    write(tmp_path, "tree/rows.jsonl", [bench_row(), flushed])
    with pytest.raises(cut.CutRefused) as refusal:
        cut_tree(tmp_path)
    assert refusal.value.code == "mixed_policy"
    elsewhere = bench_row(case="elsewhere")
    elsewhere["plans"]["fused"]["environment"]["device"]["name"] = "ANOTHER GPU"
    write(tmp_path, "tree/rows.jsonl", [bench_row(), elsewhere])
    with pytest.raises(cut.CutRefused) as refusal:
        cut_tree(tmp_path)
    assert refusal.value.code == "mixed_device"


def test_the_phase0_spelling_pairs_each_timed_leg_with_its_singles(tmp_path):
    write(tmp_path, "tree/triton/pml_2d_res20/rows.jsonl", phase0_rows())
    with pytest.raises(cut.CutRefused) as refusal:     # today AND restricted_g
        cut_tree(tmp_path, min_rows=1)
    assert refusal.value.code == "mixed_repair_route"
    today = cut_tree(tmp_path, min_rows=1,
                     select_route=f"{SHA_A[:12]}:linear:3-per-step:full")
    entry, = entries(today, "fused pair D")
    assert entry["fused_ms_per_step"] == pytest.approx(1.4)
    assert entry["baseline_ms_per_step"] == pytest.approx(0.5)
    row, = entry["measurements"]
    assert row["source"] == "phase0:today" and row["line"] == 1
    assert row["baseline_line"] == 5
    assert today["route"]["restriction_counters_recorded_on"] == "1 of 1 bracketed rows"
    assert today["contention"]["admitted_without_an_after_probe"] == 1
    assert sorted(excluded_reasons(today)) == [
        f"another repair route: {SHA_A[:12]}:linear:1-per-step:restricted",
        "verdict CEILING-NOT-A-TIMING", "verdict CEILING-NOT-A-TIMING"]
    restricted = cut_tree(tmp_path, min_rows=1,
                          select_route=f"{SHA_A[:12]}:linear:1-per-step:restricted")
    entry, = entries(restricted, "fused pair D")
    assert entry["fused_ms_per_step"] == pytest.approx(0.9)
    assert restricted["route"]["restricted"] is True


def test_a_phase0_leg_is_refused_with_its_baseline(tmp_path):
    rows = phase0_rows()
    rows[-1]["floors"]["spread_within_gate"] = False      # the singles leg
    rows[-1]["reportable"] = False
    rows[-1]["verdict"] = "TIMED-NOT-REPORTABLE"
    write(tmp_path, "tree/triton/pml_2d_res20/rows.jsonl", rows + phase0_rows(
        case="magnetic_seam_2d"))
    record = cut_tree(tmp_path, min_rows=1,
                      select_route=f"{SHA_A[:12]}:linear:3-per-step:full")
    reasons = excluded_reasons(record)
    assert any(reason.startswith("baseline leg: ") for reason in reasons), reasons
    assert [row["case"] for entry in entries(record) for row in entry["measurements"]
            ] == ["magnetic_seam_2d"]


# ---------------------------------------------------------------------------
# Scaling: bands
# ---------------------------------------------------------------------------

def sweep_tree(tmp_path, ratios):
    rows = []
    for index, (cells, ratio) in enumerate(ratios):
        for repeat in range(2):
            rows.append(bench_row(case=f"sweep_{cells}_{repeat}", grid=(cells, 1, 1),
                                  fused_ms=1.0 + 0.001 * repeat, base_ms=ratio,
                                  utc=f"2026-09-19T13:{index:02d}:{repeat:02d}Z"))
    write(tmp_path, "tree/rows.jsonl", rows)


def test_a_run_of_sizes_with_one_verdict_is_one_band(tmp_path):
    sweep_tree(tmp_path, [(24000, 0.35), (216000, 0.43), (1014000, 0.44)])
    entry, = entries(cut_tree(tmp_path), "fused pair D")
    assert entry["band"] == {"lo_cells": 24000, "hi_cells": 1014000}
    assert entry["rows"] == 6 and entry["slower_beyond_spread"] is True
    assert entry["ratio_max"] == pytest.approx(0.44)


def test_a_verdict_change_between_two_sizes_is_an_explicit_unmeasured_gap(tmp_path):
    sweep_tree(tmp_path, [(24000, 0.35), (2400000, 0.73), (5400000, 1.08)])
    slow, gap, fast = entries(cut_tree(tmp_path), "fused pair D")
    assert slow["band"] == {"lo_cells": 24000, "hi_cells": 2400000}
    assert gap["band"] == {"lo_cells": 2400001, "hi_cells": 5399999}
    assert fast["band"] == {"lo_cells": 5400000, "hi_cells": 5400000}
    assert (slow["status"], gap["status"], fast["status"]) == (
        "measured", "unmeasured", "measured")
    assert gap["rows"] == 0 and gap["measurements"] == []
    assert gap["reason"] == ("the verdict changes between 2,400,000 and 5,400,000 "
                             "cells and no row lies between them")
    assert slow["slower_beyond_spread"] and not fast["slower_beyond_spread"]


def test_the_cut_bands_are_looked_up_without_extrapolation(tmp_path):
    sweep_tree(tmp_path, [(24000, 0.35), (2400000, 0.73), (5400000, 1.08)])
    record = cut_tree(tmp_path)

    assert [ask_cells(record, n) for n in (23999, 24000, 1000000, 2400000, 2400001, 5399999,
                             5400000, 5400001)] == [
        "no_record", "vetoed", "vetoed", "vetoed", "no_record", "no_record",
        "not_vetoed", "no_record"]


def test_a_single_size_inside_the_spread_splits_nothing_it_should_not(tmp_path):
    """Faster and slower-within-spread are the same verdict: not vetoed."""
    sweep_tree(tmp_path, [(24000, 1.09), (216000, 0.985), (1014000, 1.05)])
    entry, = entries(cut_tree(tmp_path), "fused pair D")
    assert entry["band"] == {"lo_cells": 24000, "hi_cells": 1014000}
    assert entry["slower_beyond_spread"] is False


def test_one_noisy_row_does_not_clear_a_band_of_slower_rows(tmp_path):
    """Each row against its OWN spreads, never the best ratio against the worst spread.

    Measured on the hand-written table's electric pair: a 0.25x loss at DRIVE size with
    a 4% window spread, and a 0.976x loss at 2.4M cells with half a percent. Pooled
    extremes read 0.976 against 1 - 0.05 and cleared the whole band, DRIVE sizes included.
    """
    write(tmp_path, "tree/rows.jsonl", [
        bench_row(case="small_a", grid=(9760, 1, 1), fused_ms=1.5, base_ms=0.39,
                  fused_spread=0.04, base_spread=0.015),
        bench_row(case="small_b", grid=(24000, 1, 1), fused_ms=1.4, base_ms=0.35),
        bench_row(case="large", grid=(2400000, 1, 1), fused_ms=1.36, base_ms=1.328,
                  fused_spread=0.004, base_spread=0.005)])
    entry, = entries(cut_tree(tmp_path), "fused pair D")
    assert entry["band"] == {"lo_cells": 9760, "hi_cells": 2400000}
    assert entry["ratio_max"] == pytest.approx(0.976471)
    assert entry["fused_spread"] == pytest.approx(0.04)
    assert entry["veto_margin"] == pytest.approx(1 - 0.009 - 0.976471, abs=1e-6)
    assert entry["slower_beyond_spread"] is True


def test_a_band_edge_must_be_a_size_more_than_one_session_measured(tmp_path):
    """A BAND'S EDGE IS ITS REACH, and one session cannot set it.

    ``fused_spread`` and ``baseline_spread`` price the noise INSIDE a window; the
    spread BETWEEN sessions is not priced anywhere, and at neighbouring sizes the
    corpus shows it at a few parts in a thousand -- the size of the thinnest margins
    measured. So a size one row measured may not be a band edge: the band is trimmed
    to corroborated sizes and the remainder is written unmeasured, with its numbers,
    so the next round sees what it owes.
    """
    write(tmp_path, "tree/rows.jsonl", [
        bench_row(case="lone_small", grid=(9760, 1, 1), fused_ms=1.5, base_ms=0.39),
        bench_row(case="mid_a", grid=(24000, 1, 1), fused_ms=1.4, base_ms=0.35),
        bench_row(case="mid_b", grid=(24000, 1, 1), fused_ms=1.5, base_ms=0.36),
        bench_row(case="top_a", grid=(216000, 1, 1), fused_ms=1.4, base_ms=0.4),
        bench_row(case="top_b", grid=(216000, 1, 1), fused_ms=1.4, base_ms=0.41),
        bench_row(case="lone_large", grid=(2400000, 1, 1), fused_ms=1.36,
                  base_ms=1.328, fused_spread=0.004, base_spread=0.005)])
    small, band, large = by_band(cut_tree(tmp_path), "fused pair D")
    assert band["band"] == {"lo_cells": 24000, "hi_cells": 216000}
    assert band["status"] == "measured" and band["slower_beyond_spread"] is True
    assert band["rows"] == 4
    assert small["band"] == {"lo_cells": 9760, "hi_cells": 23999}
    assert large["band"] == {"lo_cells": 216001, "hi_cells": 2400000}
    for edge, size in ((small, "9,760"), (large, "2,400,000")):
        assert edge["status"] == "unmeasured"
        assert "no size in it carries 2 rows" in edge["reason"]
        assert f"{size} (1 row)" in edge["reason"]
        assert edge["rows"] == 1 and edge["veto_margin"] is not None
    # nothing is answered between the corroborated edges' outside and the band
    record = cut_tree(tmp_path)
    dp.validate(record, expect_table="triton")
    assert [ask_cells(record, n) for n in (9759, 9760, 23999, 24000, 216000, 216001,
                                           2400000, 2400001)] == [
        "no_record", "no_record", "no_record", "vetoed", "vetoed", "no_record",
        "no_record", "no_record"]


def test_a_size_measured_once_INSIDE_a_band_is_kept_and_can_only_soften_it(tmp_path):
    """Why an edge must be corroborated and an interior size need not be.

    ``least_margin`` is a MINIMUM over rows, so adding a row can only remove a veto,
    never create one: an uncorroborated interior row cannot make a band veto, while an
    uncorroborated EDGE extends the band's reach over sizes nothing measured. So the
    interior row stays in the key -- and one fast row at that size clears the band.
    """
    def tree(middle_base):
        write(tmp_path, "tree/rows.jsonl", [
            bench_row(case="lo_a", grid=(24000, 1, 1), fused_ms=1.4, base_ms=0.35),
            bench_row(case="lo_b", grid=(24000, 1, 1), fused_ms=1.5, base_ms=0.36),
            bench_row(case="mid", grid=(216000, 1, 1), fused_ms=1.4,
                      base_ms=middle_base),
            bench_row(case="hi_a", grid=(1014000, 1, 1), fused_ms=1.4, base_ms=0.4),
            bench_row(case="hi_b", grid=(1014000, 1, 1), fused_ms=1.4, base_ms=0.41)])
        return cut_tree(tmp_path)

    slow, = by_band(tree(0.4), "fused pair D")
    assert slow["band"] == {"lo_cells": 24000, "hi_cells": 1014000}
    assert slow["status"] == "measured" and slow["slower_beyond_spread"] is True
    assert slow["rows"] == 5 and "mid" in {r["case"] for r in slow["measurements"]}
    # and one FAST row at that same size is never interpolated over: the run splits
    # there, the uncorroborated middle is its own unmeasured key, and no veto spans it
    assert [(entry["band"]["lo_cells"], entry["band"]["hi_cells"], entry["status"],
             entry.get("slower_beyond_spread"))
            for entry in by_band(tree(1.6), "fused pair D")] == [
        (24000, 24000, "measured", True), (24001, 215999, "unmeasured", None),
        (216000, 216000, "unmeasured", False),
        (216001, 1013999, "unmeasured", None), (1014000, 1014000, "measured", True)]


def test_a_trimmed_band_never_overlaps_the_gap_key_beside_it(tmp_path):
    """Sizes 1 / 3 / 2 / 1 / 2 rows with the verdict flipping at the last: the trimmed
    remainders and the verdict-change gap must tile the class without overlapping."""
    rows = []
    for cells, count, base in ((4000, 1, 0.35), (24000, 3, 0.35), (216000, 2, 0.36),
                               (1014000, 1, 0.4), (5400000, 2, 1.4)):
        for index in range(count):
            rows.append(bench_row(case=f"c{cells}_{index}", grid=(cells, 1, 1),
                                  fused_ms=1.3 + 0.01 * index, base_ms=base,
                                  utc=f"2026-09-19T13:{index:02d}:00Z"))
    write(tmp_path, "tree/rows.jsonl", rows)
    record = cut_tree(tmp_path)
    dp.validate(record, expect_table="triton")        # refuses overlapping bands
    bands = [(entry["band"]["lo_cells"], entry["band"]["hi_cells"], entry["status"])
             for entry in by_band(record, "fused pair D")]
    assert bands == [(4000, 23999, "unmeasured"), (24000, 216000, "measured"),
                     (216001, 1014000, "unmeasured"),
                     (1014001, 5399999, "unmeasured"), (5400000, 5400000, "measured")]
    assert [ask_cells(record, n) for n in (24000, 216000, 1014000, 5400000)] == [
        "vetoed", "vetoed", "no_record", "not_vetoed"]


def test_each_class_is_banded_on_its_own_sizes(tmp_path):
    write(tmp_path, "tree/rows.jsonl", [
        bench_row(case="a", grid=(100, 1, 1)), bench_row(case="b", grid=(200, 1, 1)),
        bench_row(case="c", grid=(64, 64, 64), dimensions=3),
        bench_row(case="d", grid=(64, 64, 64), dimensions=3),
        bench_row(case="e", grid=(300, 1, 1), complex_storage=True),
        bench_row(case="f", grid=(300, 1, 1), susceptibilities=1)])
    record = cut_tree(tmp_path)
    assert sorted(record["keys"]) == [
        "fused pair D|complex|2|0|bracketed|300-300",
        "fused pair D|real|2|0|bracketed|100-200",
        "fused pair D|real|2|1|bracketed|300-300",
        "fused pair D|real|3|0|bracketed|262144-262144"]


# ---------------------------------------------------------------------------
# Shape / serialization
# ---------------------------------------------------------------------------

def test_the_bytes_are_deterministic_and_carry_no_wall_clock(tmp_path):
    standard_tree(tmp_path)
    write(tmp_path, "second/rows.jsonl", [bench_row(case="third", grid=(300, 100, 1))])
    first = dp.dumps(cut_tree(tmp_path, roots=("tree", "second")))
    again = dp.dumps(cut_tree(tmp_path, roots=("second", "tree")))
    assert first == again
    record = json.loads(first)
    assert record["recorded_utc"] == "2026-09-19T14:00:00Z"
    assert record["inputs"]["roots"] == ["second", "tree"]
    assert str(tmp_path) not in first, "an absolute path reached the record"
    dp.validate(record)


def test_the_inputs_digest_covers_every_row_read_not_only_the_admitted(tmp_path):
    standard_tree(tmp_path)
    before = cut_tree(tmp_path)
    smoke = bench_row(case="smoke")
    smoke["prefer_gpu"] = False
    with open(tmp_path / "tree/triton/rows.jsonl", "a") as handle:
        handle.write(json.dumps(smoke) + "\n")
    after = cut_tree(tmp_path)
    assert after["inputs"]["sha256"] != before["inputs"]["sha256"]
    assert {k: v["measurements"] for k, v in after["keys"].items()} == {
        k: v["measurements"] for k, v in before["keys"].items()}


def run(tmp_path, *arguments):
    return cut.main(["--base", str(tmp_path), *arguments])


def test_the_command_line_cuts_writes_a_report_and_checks(tmp_path, capsys):
    standard_tree(tmp_path)
    out = tmp_path / "kernels" / "timing.json"
    assert run(tmp_path, "--table", "triton", "--rows", str(tmp_path / "tree"),
               "--out", str(out)) == 0
    printed = capsys.readouterr().out
    report = (tmp_path / "kernels" / "timing.report.txt").read_text()
    assert report in printed
    assert "measured 2" in report and "unmeasured 0" in report
    assert "fused pair D|real|2|0|bracketed|24000-24000" in report
    assert dp.load_record(str(out), expect_table="triton")["rows"]["admitted"] == 4
    assert run(tmp_path, "--check", str(out)) == cut.EXIT_OK
    assert "recomputes byte for byte" in capsys.readouterr().out


def test_check_refuses_on_any_difference(tmp_path, capsys):
    standard_tree(tmp_path)
    out = tmp_path / "timing.json"
    assert run(tmp_path, "--table", "triton", "--rows", str(tmp_path / "tree"),
               "--out", str(out)) == 0
    with open(tmp_path / "tree/triton/rows.jsonl", "a") as handle:
        handle.write(json.dumps(bench_row(case="late", fused_ms=1.2)) + "\n")
    capsys.readouterr()
    assert run(tmp_path, "--check", str(out)) == cut.EXIT_DIFFERS
    said = capsys.readouterr().out
    assert "DIFFERS" in said and "inputs" in said


def test_check_refuses_a_new_file_under_a_root_it_was_cut_from(tmp_path):
    standard_tree(tmp_path)
    out = tmp_path / "timing.json"
    run(tmp_path, "--table", "triton", "--rows", str(tmp_path / "tree"), "--out", str(out))
    write(tmp_path, "tree/late/rows.jsonl", [bench_row(case="late")])
    assert run(tmp_path, "--check", str(out)) == cut.EXIT_DIFFERS


def test_check_says_artifacts_absent_rather_than_pass_or_differ(tmp_path, capsys):
    standard_tree(tmp_path)
    out = tmp_path / "timing.json"
    run(tmp_path, "--table", "triton", "--rows", str(tmp_path / "tree"), "--out", str(out))
    os.remove(tmp_path / "tree/triton/rows.jsonl")
    os.rmdir(tmp_path / "tree/triton")
    os.rmdir(tmp_path / "tree")
    capsys.readouterr()
    assert run(tmp_path, "--check", str(out)) == cut.EXIT_ARTIFACTS_ABSENT
    assert "cannot recompute" in capsys.readouterr().out


def test_check_refuses_a_hand_edited_record_by_name(tmp_path, capsys):
    standard_tree(tmp_path)
    out = tmp_path / "timing.json"
    run(tmp_path, "--table", "triton", "--rows", str(tmp_path / "tree"), "--out", str(out))
    record = json.loads(out.read_text())
    next(iter(record["keys"].values()))["fused_ms_per_step"] = 0.1
    out.write_text(dp.dumps(record))
    capsys.readouterr()
    assert run(tmp_path, "--check", str(out)) == cut.EXIT_REFUSED
    assert "record_digest_mismatch" in capsys.readouterr().out


def test_a_refusal_on_the_command_line_names_itself_and_writes_nothing(tmp_path, capsys):
    write(tmp_path, "tree/rows.jsonl", [bench_row(), bench_row(case="after", sha=SHA_B)])
    out = tmp_path / "timing.json"
    assert run(tmp_path, "--table", "triton", "--rows", str(tmp_path / "tree"),
               "--out", str(out)) == cut.EXIT_REFUSED
    assert "mixed_repair_route" in capsys.readouterr().out
    assert not out.exists()


def test_a_record_the_reader_would_refuse_is_a_named_refusal_not_a_traceback(
        tmp_path, capsys):
    """``cut`` seals and then validates what it built. That refusal is the reader's,
    not the cutter's, and it must leave by the same door as every other refusal."""
    undated = bench_row(case="undated")
    undated["utc"] = None
    write(tmp_path, "tree/rows.jsonl", [bench_row(), undated])
    out = tmp_path / "timing.json"
    assert run(tmp_path, "--table", "triton", "--rows", str(tmp_path / "tree"),
               "--out", str(out)) == cut.EXIT_REFUSED
    assert "REFUSED malformed_record" in capsys.readouterr().out
    assert not out.exists()


def test_the_admitted_rows_say_whether_their_device_was_probed_afterwards(tmp_path):
    """The record counts the rows admitted without an after-probe; each key's rows
    name themselves, so the ones at a band edge can be found."""
    unprobed = bench_row(case="unprobed")
    unprobed.pop("box_after")
    write(tmp_path, "tree/rows.jsonl", [bench_row(), unprobed])
    record = cut_tree(tmp_path)
    assert record["contention"]["admitted_without_an_after_probe"] == 1
    probed = {row["case"]: row["after_probe"]
              for entry in record["keys"].values() for row in entry["measurements"]}
    assert probed == {"pml_2d": True, "unprobed": False}


# ---------------------------------------------------------------------------
# The preview of served_in_dispatch_preferred
# ---------------------------------------------------------------------------

def census_row(name, cells, sources=("D",), leg="examples"):
    return {"row": name, "leg": leg, "measured": True, "grid_cells": cells,
            "grid_shape": [cells, 1, 1],
            "facts": {"dimensions_attr": 2, "cell_size": [1.0, 1.0, 0.0]},
            "configuration": {"shape": [cells, 1, 1], "force_complex_fields": False,
                              "cylindrical": False, "n_polarizations": 0,
                              "source_field_types": list(sources), "beta": 0.0,
                              "k_point": [0.0, 0.0, 0.0], "pml_active": True,
                              "has_bloch": False, "bfast_active": False,
                              "has_conductivity": False, "has_nonlinearity": False,
                              "has_offdiagonal_epsilon": False, "has_symmetry": False,
                              "mirrored": [False, False, False]}}


def preview_fixture(tmp_path):
    sweep_tree(tmp_path, [(24000, 0.35), (2400000, 0.73), (5400000, 1.08)])
    census = tmp_path / "census"
    write(tmp_path, "census/examples.jsonl", [
        census_row("small.py", 30000), census_row("large.py", 5400000),
        census_row("gap.py", 3000000), census_row("magnetic.py", 30000, sources=("B",))])
    write(tmp_path, "census/tests.jsonl", [])

    def instance(row, seam, product, in_seam):
        return {"row": f"examples:{row}", "seam": seam, "product": product,
                "predicate_admits": True, "in_seam_source": in_seam,
                "curl_arm": "PML", "constitutive_arm": "ordinary"}

    board = {
        "backend": "triton", "census": str(census),
        "aggregate": {"denominator": 597, "served_in_dispatch": 8,
                      "served_in_dispatch_measurement": {
                          "arms": {"fused pair B": 4, "fused pair D": 4},
                          "released_arms": ["fused pair B", "fused pair D"],
                          "label_map": {"products": {"fused_pair_B": "fused pair B",
                                                     "fused_pair_D": "fused pair D"}}}},
        "seam_instances": [instance(row, seam, product, in_seam)
                           for row, source in (("small.py", "D"), ("large.py", "D"),
                                               ("gap.py", "D"), ("magnetic.py", "B"))
                           for seam, product, in_seam in (
                               ("B->H", "fused_pair_B", source == "B"),
                               ("D->E", "fused_pair_D", source == "D"))],
    }
    path = tmp_path / "board.json"
    path.write_text(json.dumps(board))
    return path


def test_the_preview_counts_the_instances_that_fall_in_a_vetoed_band(tmp_path):
    board = preview_fixture(tmp_path)
    record = cut_tree(tmp_path)
    preview = cut.preview(record, str(board))
    assert preview["label"].startswith("PREVIEW")
    assert preview["denominator"] == 597
    assert preview["served_in_dispatch"] == 8
    d_pair = preview["products"]["fused pair D"]
    assert d_pair["dispatching_on_the_board"] == 4 and d_pair["joined"] == 4
    assert d_pair["bracketed"] == 3
    assert d_pair["vetoed"] == 1                      # small.py, bracketed, 30,000 cells
    assert d_pair["vetoed_rows"] == ["examples:small.py"]
    assert d_pair["not_vetoed"] == 1                  # large.py at the top of the sweep
    assert d_pair["no_record"] == {"bracketed: key_unmeasured": 1,
                                   "clean: class_not_in_record": 1}
    assert d_pair["join_agrees_with_the_board"] is True
    b_pair = preview["products"]["fused pair B"]
    assert b_pair["vetoed"] == 0 and b_pair["bracketed"] == 1
    assert b_pair["no_record"] == {"bracketed: class_not_in_record": 1,
                                   "clean: class_not_in_record": 3}
    assert preview["joined"] == 8 and preview["bracketed"] == 4
    assert preview["join_agrees_with_the_board"] is True
    assert preview["vetoed"] == 1
    assert preview["served_in_dispatch_preferred_preview"] == 7
    text = cut.render_report(record, preview=preview)
    assert "PREVIEW" in text and "7 / 597" in text and "8 / 597" in text


def test_a_board_without_instances_is_joined_from_the_census_product_blocks(tmp_path):
    """The hand-written table's board lists no per-instance rows; its census does.

    A product spanning three slots serves TWO of the 597 seam-instances, so it joins,
    and would leave, as two -- while it carries one bracket.
    """
    board = preview_fixture(tmp_path)
    document = json.loads(board.read_text())
    document.pop("seam_instances")
    document["aggregate"]["served_in_dispatch_measurement"]["arms"] = {"fused pair D": 4}
    document["aggregate"]["served_in_dispatch_measurement"]["label_map"] = {
        "products": {"three_slot_product": "fused pair D"}}
    document["aggregate"]["served_in_dispatch"] = 4
    board.write_text(json.dumps(document))
    rows = [census_row("small.py", 30000), census_row("large.py", 5400000)]
    for row in rows:
        row["three_slot_product"] = {"covered_modulo_backend": True,
                                     "spans": ["step_D", "update_E", "update_P"]}
    rows.append(census_row("refused.py", 30000))
    rows[-1]["three_slot_product"] = {"covered_modulo_backend": False,
                                      "spans": ["step_D", "update_E", "update_P"]}
    write(tmp_path, "census/examples.jsonl", rows)
    preview = cut.preview(cut_tree(tmp_path), str(board))
    tally = preview["products"]["fused pair D"]
    assert "census" in preview["joined_from"]
    assert tally["joined"] == 4 and tally["bracketed"] == 2
    assert tally["vetoed"] == 2 and tally["not_vetoed"] == 2
    assert tally["join_agrees_with_the_board"] is True
    assert preview["served_in_dispatch_preferred_preview"] == 2


def test_the_census_bracket_rule_is_checked_and_never_guessed(tmp_path):
    """Whether a product carries the bracket decides WHICH KEY it is asked for, and on
    the census path it is read off the first slot's field letter. A span whose first
    slot names no source field (``update_H``) cannot answer it, and answering "clean"
    for it would ask the wrong key of a run the record may veto."""
    board = preview_fixture(tmp_path)
    document = json.loads(board.read_text())
    document.pop("seam_instances")
    document["aggregate"]["served_in_dispatch_measurement"]["arms"] = {"fused pair D": 2}
    document["aggregate"]["served_in_dispatch_measurement"]["label_map"] = {
        "products": {"late_product": "fused pair D"}}
    document["aggregate"]["served_in_dispatch"] = 2
    board.write_text(json.dumps(document))
    rows = [census_row("readable.py", 30000), census_row("unreadable.py", 30000)]
    rows[0]["late_product"] = {"covered_modulo_backend": True,
                               "spans": ["step_D", "update_E"]}
    rows[1]["late_product"] = {"covered_modulo_backend": True,
                               "spans": ["update_H", "step_D"]}
    write(tmp_path, "census/examples.jsonl", rows)
    preview = cut.preview(cut_tree(tmp_path), str(board))
    tally = preview["products"]["fused pair D"]
    assert tally["vetoed"] == 1
    assert tally["no_record"] == {
        "unknown: the census cannot say whether this span carries the bracket: "
        "spans[0] is 'update_H', whose field letter is not one a source can inject": 1}
    assert cut.CENSUS_SOURCE_FIELD_LETTERS == ("B", "D")


def test_the_preview_carries_the_live_repair_route_not_only_the_report(tmp_path):
    """A programmatic reader of the preview must not be able to quote its numbers
    without the fact that the record describes another tree."""
    board = preview_fixture(tmp_path)
    record = cut_tree(tmp_path)
    preview = cut.preview(record, str(board), live_route="ff" * 32)
    assert preview["record_route"] == SHA_A
    assert preview["live_repair_route"] == "ff" * 32
    assert preview["record_is_for_the_live_route"] is False
    same = cut.preview(record, str(board), live_route=SHA_A)
    assert same["record_is_for_the_live_route"] is True
    unknown = cut.preview(record, str(board))
    assert unknown["live_repair_route"] is None
    assert unknown["record_is_for_the_live_route"] is None


def test_the_preview_says_when_the_board_carries_no_instances(tmp_path):
    board = preview_fixture(tmp_path)
    document = json.loads(board.read_text())
    document.pop("seam_instances")
    board.write_text(json.dumps(document))
    preview = cut.preview(cut_tree(tmp_path), str(board))
    assert preview["served_in_dispatch_preferred_preview"] is None
    assert "no per-instance" in preview["not_joined_because"]


# ---------------------------------------------------------------------------
# Realistic: the shipped records, recut from the rows on disk
# ---------------------------------------------------------------------------

def shipped(table):
    return os.path.join(REPO_API, "meep_gpu", f"{table}_kernels", "timing.json")


def _rows_on_disk(table):
    record = dp.load_record(shipped(table), expect_table=table)
    roots = [os.path.join(HERE, root) for root in record["inputs"]["roots"]]
    return all(os.path.exists(root) for root in roots)


@pytest.mark.parametrize("table", ["triton", "cuda"])
def test_the_shipped_record_names_what_it_was_cut_from(table):
    record = dp.load_record(shipped(table), expect_table=table)
    assert record["cut"]["table"] == table
    assert record["cut"]["by"] == "parity/meep_gpu/cut_timing_record.py"
    assert all(root.startswith("results/") for root in record["inputs"]["roots"])
    assert all(row["artifact"].startswith("results/")
               for entry in record["keys"].values() for row in entry["measurements"])
    assert record["rows"]["admitted"] >= 20
    measured = [e for e in record["keys"].values() if e["status"] == "measured"]
    assert len(measured) >= 4


@pytest.mark.requires_resource("fused-timing-rows")
@pytest.mark.parametrize("table", ["triton", "cuda"])
def test_the_shipped_record_recuts_byte_for_byte_from_the_rows_on_disk(table, capsys):
    if not _rows_on_disk(table):
        pytest.skip("results/ is untracked and the timing rows are not on this host")
    assert cut.main(["--check", shipped(table)]) == cut.EXIT_OK
    assert "recomputes byte for byte" in capsys.readouterr().out
