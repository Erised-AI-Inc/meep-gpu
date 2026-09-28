#!/usr/bin/env python
"""Turn ``bench_fused_products`` rows into the benchmark document, and nothing else.

THE ROWS ARE THE EVIDENCE; THIS IS A VIEW OF THEM. Every number here is read out
of ``rows.jsonl`` and none is computed from anything else, so the document cannot
drift from the run that produced it -- the failure mode the sibling benchmark
documents avoid by being generated rather than transcribed.

WHAT IT REFUSES TO PRINT. A row whose floors did not all pass is NOT given a
speedup column. It is listed, by name, under the floor it failed, because a case
that could not be measured is a fact about the campaign and hiding it would make
the table read as complete when it is not. ``reference_census_refuse_unmeasurable
_rows_by_name`` is the same rule one layer up.

TWO POPULATIONS, NEVER ONE MEDIAN (2026-09-20). Rows the METAL table served are a
different kind of measurement from rows the two NVIDIA tables served -- a host
engine with device mirrors and a residency bracket around every launch, on a
laptop, against device arrays on a GPU host -- so they get a section, a table and a
headline of their own (:func:`metal_section`) and enter no NVIDIA figure. A document
rendered from NVIDIA rows alone is byte-for-byte what it was before that section
existed.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple


def load(path: str) -> List[Dict[str, Any]]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


# ---------------------------------------------------------------------------
# Provenance and the deposit-repair route (fields added to the rows 2026-09-19)
# ---------------------------------------------------------------------------

#: What a row written before a field existed shows in its place. Such a row is
#: RENDERED, never dropped: the 2026-09-17 campaign is all rows of that kind.
NOT_RECORDED = "not recorded"

#: The keys ``bench_fused_products.PINNED_SOURCES`` stamps, in its order. The first is
#: the one the header groups on; the rest are named only if they vary.
PINNED_KEYS = ("deposit_repair", "fused_pairs", "fused_polarization_pair", "fastpath",
               "driver", "bench")

#: ``bench_fused_products.TABLE_PINNED_SOURCES["metal"]``, in its order: what a Metal
#: row pins beyond :data:`PINNED_KEYS`. The residency bracket that sets a Metal row's
#: cost lives in the second and third.
METAL_PINNED_KEYS = ("metal_dispatch", "metal_launch", "metal_device",
                     "metal_route_gate")

#: The ``_PreparedCells`` counters the bench sums per leg (``deposit_repair._PreparedCells.__slots__``).
ROUTE_COUNTERS = ("linear_saves", "linear_repairs", "cells_saves", "cells_repairs")

#: ``bench_fused_products.RESTRICTION_COUNTERS``: per source per save, whether the
#: bracket was narrowed to the component that source writes or kept all three targets.
#: A row written before 2026-09-20 carries neither, and is described without them.
RESTRICTION_COUNTERS = ("restricted_sources", "restriction_fallbacks")


def is_smoke(row: Dict[str, Any]) -> bool:
    """Is this a ``--smoke`` row: a check of the harness, never a timing?

    ``smoke`` IS ITS OWN FIELD SINCE 2026-09-20 and is read where a row has it. Before
    then ``prefer_gpu`` False MEANT smoke, because only ``--smoke`` lifted on the host
    array module; a Metal row written before 2026-09-27 was lifted ``prefer_gpu=False``
    under the enable by design and is a real measurement, so the old reading would
    have dropped every such row from every headline. (From 2026-09-27 a Metal row is
    lifted ``prefer_gpu=True`` and reads True; ``prefer_gpu=False`` is the NumPy
    reference, which never dispatches.) A row without the field keeps the old rule.
    """
    if "smoke" in row:
        return bool(row["smoke"])
    return row.get("prefer_gpu") is False


def served_tables(row: Dict[str, Any]) -> List[str]:
    """The kernel table(s) that served this row's FUSED leg, asked of the row.

    The table that served, never the drive table the campaign was started for: the
    first host smoke run read the Triton DRIVE rows and the Metal table served every
    slot. A row with no plan on record (a skip, an error) answers with nothing.
    """
    sub = row.get("substitution") or {}
    fused = (sub.get("tables_by_leg") or {}).get("fused")
    if fused:
        return sorted(fused)
    plan = ((row.get("plans") or {}).get("fused") or {}).get("composition") or {}
    return sorted(plan.get("tables_dispatched") or sub.get("tables_dispatched") or [])


def population(row: Dict[str, Any]) -> str:
    """``metal`` or ``nvidia``: which headline, table and section a row belongs to.

    The two NVIDIA tables stay ONE population, as they have been since the first
    campaign: one host, one GPU, device arrays, no residency seam, and the published
    headline is a median over both. A row that recorded no serving table falls back
    to the drive table it was read from.
    """
    served = served_tables(row)
    if served:
        return "metal" if "metal" in served else "nvidia"
    return "metal" if row.get("drive_table") == "metal" else "nvidia"


def digest(row: Dict[str, Any], key: str = "deposit_repair") -> str:
    """The row's sha256 for one pinned source, or :data:`NOT_RECORDED`."""
    sha = ((row.get("provenance") or {}).get("sha256") or {}).get(key)
    return sha or NOT_RECORDED


def _short(value: str) -> str:
    return value if value == NOT_RECORDED else f"`{value[:12]}`"


def provenance_lines(rows: Sequence[Dict[str, Any]]) -> List[str]:
    """The header's account of which code the rows timed.

    COUNTED PER ``deposit_repair.py`` DIGEST, and a mix is said in bold. The linear
    repair route landed 2026-09-18/19 and moved the per-step host cost of every
    bracketed row; a report that pooled rows timed before it with rows timed after it
    would print one median over two different programs, which is the silent mixing
    the provenance block exists to prevent.
    """
    counts = collections.Counter(digest(r) for r in rows)
    spelled = "; ".join(f"{_short(key)}: {n} row{'s' if n != 1 else ''}"
                        for key, n in counts.most_common())
    out: List[str] = []
    if len(counts) > 1:
        out.append(
            f"**These rows combine {len(counts)} `meep_gpu/deposit_repair.py` "
            f"provenances and are not one population** — {spelled}. Rows timed "
            f"against different repair code measure different programs: compare "
            f"within one digest, never across. A row reading *{NOT_RECORDED}* "
            f"predates the 2026-09-19 provenance block and cannot say which "
            f"`deposit_repair.py` it timed. The `src` column gives each row's "
            f"digest.")
    else:
        only = next(iter(counts))
        if only == NOT_RECORDED:
            out.append(
                f"Source provenance: **{NOT_RECORDED}** on all {len(rows)} rows. "
                f"They predate the 2026-09-19 provenance block, so which "
                f"`meep_gpu/deposit_repair.py` they timed cannot be read off them.")
        else:
            gits = [((r.get("provenance") or {}).get("git") or {}) for r in rows]
            heads = sorted({g.get("head") or "unknown" for g in gits})
            dirty = sorted({str(g.get("any_dirty")) for g in gits})
            out.append(
                f"Source provenance: all {len(rows)} rows timed "
                f"`meep_gpu/deposit_repair.py` {_short(only)}; git HEAD "
                f"{', '.join(_short(h) if h != 'unknown' else h for h in heads)}; "
                f"working tree dirty for a pinned path: {'/'.join(dirty)}.")
    # A ``--smoke`` ROW IS A CHECK OF THE HARNESS, never a GPU timing: it lifts the
    # NumPy reference. :func:`is_smoke` reads the row's own ``smoke`` field, and an
    # explicit ``prefer_gpu`` False only on rows older than that field.
    smoke = sum(1 for r in rows if is_smoke(r))
    if smoke:
        out.append(
            f"**{smoke} of {len(rows)} rows are host smoke rows** (`--smoke`, lifted "
            f"on the host array module): they check the harness and are not GPU "
            f"timings.")
    recorded = [r for r in rows if r.get("provenance")]
    varying = [key for key in PINNED_KEYS[1:]
               if len({digest(r, key) for r in recorded}) > 1]
    if varying:
        out.append(
            "Among the rows that record provenance, these pinned sources ALSO "
            "differ: " + ", ".join(
                f"`{key}` ({len({digest(r, key) for r in recorded})} digests)"
                for key in varying) + ".")
    return out


def repair_route(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The fused leg's summed deposit-repair route counters, or None if unrecorded."""
    route = ((row.get("per_leg") or {}).get("fused") or {}).get("deposit_repair_route")
    return route if isinstance(route, dict) else None


def route_verdict(route: Optional[Dict[str, Any]]) -> str:
    """One word for which route the bracket took over the timed windows.

    ``linear`` only when EVERY counted call took the cached linear index; a single
    3-tuple call makes it ``mixed``. ``never ran`` is a bracket in the plan whose
    counters did not move -- the shape a bracket takes when nothing reached it.
    """
    if route is None:
        return NOT_RECORDED
    if not route.get("bracketed_slots"):
        return "no bracket"
    linear = int(route.get("linear_saves", 0)) + int(route.get("linear_repairs", 0))
    cells = int(route.get("cells_saves", 0)) + int(route.get("cells_repairs", 0))
    if linear and not cells:
        return "linear"
    if cells and not linear:
        return "3-tuple fallback"
    if linear and cells:
        return "mixed"
    return "never ran"


def restriction_note(route: Dict[str, Any]) -> str:
    """What the row recorded about the component restriction, or nothing at all.

    The route verdict stands either way -- a full three-target bracket on the linear
    index IS the linear route -- so this is a note beside it, and a LOUD one only when
    some source fell back: that bracket is sound and is three entries where one was
    possible, which is a fact about the timing beside it.
    """
    restricted = int(route.get("restricted_sources", 0))
    fallbacks = int(route.get("restriction_fallbacks", 0))
    if not restricted and not fallbacks:
        # Not recorded, or recorded off a module that keeps neither counter (the bench
        # reads a missing one as 0): a save counts every source one way or the other.
        return ""
    if not fallbacks:
        return f", {restricted} source-saves restricted to the written component"
    return (f", **{fallbacks} OF {restricted + fallbacks} SOURCE-SAVES KEPT THE FULL "
            f"THREE-TARGET BRACKET**")


def route_cell(row: Dict[str, Any]) -> str:
    route = repair_route(row)
    verdict = route_verdict(route)
    if route is None or verdict == "no bracket":
        return verdict
    counts = (f"{route.get('linear_saves', 0)}/{route.get('linear_repairs', 0)} "
              f"linear, {route.get('cells_saves', 0)}/"
              f"{route.get('cells_repairs', 0)} 3-tuple")
    text = verdict if verdict == "linear" else f"**{verdict.upper()}**"
    text = f"{'+'.join(route['bracketed_slots'])}: {text} ({counts})"
    text += restriction_note(route)
    if route.get("plan_changed_inside_window"):
        text += ", plan re-froze inside a window"
    return text


def route_lines(rows: Sequence[Dict[str, Any]]) -> List[str]:
    """Per row with a timed fused leg: did the bracket take the linear route?

    EVERY TIMED ROW, not only the reportable ones -- a refused row's bracket ran the
    same code, and whether it took the fast route is a fact about that code.
    """
    timed = [r for r in rows if ((r.get("per_leg") or {}).get("fused"))]
    if not timed:
        return []
    tables = {r.get("drive_table") for r in timed}
    def name(row: Dict[str, Any]) -> str:
        return (f"`{row['case']}`" if len(tables) <= 1
                else f"`{row['case']}` ({row.get('drive_table')})")
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in timed:
        groups.setdefault(route_verdict(repair_route(row)), []).append(row)
    out = ["## Deposit-repair route on the fused leg", "",
           "Read off the frozen plan, not inferred: every `LeadingRepairPlan` in the "
           "fused leg's slots owns a `deposit_repair._PreparedCells` whose counters "
           "record, per `(component, source)` per call, whether the save and the "
           "repair took the cached linear index or fell back to the 3-tuple index. "
           "The bench snapshots them before and after each timed window, as it does "
           "the launch witnesses, and sums them per leg. Counts are written "
           "`saves/repairs`; the plain (no-absorber) path saves no state, so its "
           "saves read 0 by construction.", ""]
    order = ("linear", "mixed", "3-tuple fallback", "never ran", "no bracket",
             NOT_RECORDED)
    for verdict in order + tuple(sorted(set(groups) - set(order))):
        members = groups.get(verdict) or []
        if not members:
            continue
        if verdict in ("no bracket", NOT_RECORDED):
            listed = ", ".join(sorted({name(r) for r in members}))
            out.append(f"- **{verdict}** ({len(members)}): {listed}")
            continue
        out.append(f"- **{verdict}** ({len(members)}):")
        for row in sorted(members, key=lambda r: (str(r.get("drive_table")),
                                                   r["case"])):
            out.append(f"  - {name(row)} — {route_cell(row)}")
    out.append("")
    return out


def pinned_device(box: Dict[str, Any]) -> Optional[str]:
    """The UUID of the device this row's process was pinned to, from the row itself.

    ``CUDA_VISIBLE_DEVICES`` may be a UUID, in which case it IS the answer, or an
    index, in which case the row's own ``nvidia-smi --query-gpu=index,uuid,…`` listing
    resolves it. Unset, or naming a device the listing does not carry, answers None:
    the process saw every device on the box and no app on it can be excluded.
    """
    pin = str(box.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if pin.startswith("GPU-"):
        return pin
    for line in str(box.get("gpus") or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2 and parts[0] == pin:
            return parts[1]
    return None


def foreign_on_the_pinned_device(box: Dict[str, Any]) -> Tuple[List[str], str]:
    """The compute apps this row shared its OWN device with, and how sure that is.

    THE PROBE'S FIELD IS NOT FILTERED WHEN THE PIN IS AN INDEX. ``box_state`` drops
    apps on other devices only when ``CUDA_VISIBLE_DEVICES`` starts with ``GPU-``, so
    on an eight-GPU host pinned to ``0`` the field it calls
    ``foreign_compute_apps_on_pinned_gpu`` lists every compute app on the machine. Each
    line is ``gpu_uuid, pid, used_memory``, so the device is right there to compare
    against, and this does the comparison the name promises. Where the pin cannot be
    resolved, every app counts and the caller says so in the mark.
    """
    apps = list(box.get("foreign_compute_apps_on_pinned_gpu") or [])
    device = pinned_device(box)
    if device is None:
        return apps, " on a box whose pinned device the row does not name"
    return [app for app in apps
            if app.split(",")[0].strip() == device], ""


def contended(row: Dict[str, Any]) -> Optional[str]:
    """Whether a neighbour held the device while this row was measured.

    The probes run before AND after every case for this reason: a neighbour that lands
    mid-case shows up in neither alone. Both are read, and the row is called contended
    if either saw a foreign process on the device this row was using.
    """
    marks = []
    # ``box_*`` IS THE GPU HOST'S PROBE (an NVIDIA row): every compute app the driver
    # reported, narrowed here to the device this row had. ``host_*`` IS THE LAPTOP'S
    # (a Metal row): every other process found driving the one Metal device -- a route
    # campaign, a fleet recut, a second bench -- which needs no narrowing.
    for when in ("box_before", "box_after", "host_before", "host_after"):
        box = row.get(when) or {}
        if when.startswith("box_"):
            foreign, scope = foreign_on_the_pinned_device(box)
        else:
            foreign, scope = list(box.get("foreign_processes") or []), ""
        if foreign:
            marks.append(f"{when.split('_')[1]}: {len(foreign)} foreign "
                         f"process(es){scope}")
    return "; ".join(marks) if marks else None


def _launch_cell(row: Dict[str, Any]) -> str:
    f, u = per_launch(row, "fused"), per_launch(row, "unfused")
    if f is None or u is None:
        return "—"
    return f"{f:.3f} : {u:.3f}" + (f" ({f / u:.1f}x)" if u else "")


def same_table(row: Dict[str, Any]) -> bool:
    """Did the fused leg and its control run the SAME kernel table?

    Derived from the row's own per-leg plans rather than read from a flag, so rows
    written before the flag existed answer too -- which matters, because the first
    CUDA rows of the 2026-09-17 campaign are exactly those rows.

    A control on a different table is not a control. Until 2026-09-17 the CUDA table
    had no singles baseline at all -- veto its fused arms and it offered nothing, so
    dispatch fell through to the Triton table's singles and the ratio compared two
    tables rather than measuring what fusing bought. Since then the merge adopts its
    two certified real-PML singles as SEAMS (``fastpath_cuda.SINGLE_ARM_SEAMS``), so
    a plain real-PML row's control is that table's own kernels; every other row's
    control still falls through, and this function says which is which per row.
    """
    flag = (row.get("substitution") or {}).get(
        "fused_and_singles_used_the_same_table")
    if isinstance(flag, bool):
        return flag
    plans = row.get("plans") or {}
    def tables(leg: str) -> List[str]:
        return sorted(((plans.get(leg) or {}).get("composition") or {})
                      .get("tables_dispatched") or [])
    fused = tables("fused")
    return bool(fused) and fused == tables("unfused")


def fused_pairs(row: Dict[str, Any]) -> int:
    """How many distinct FUSED PAIRS this row's fused plan installed.

    THE DISCRIMINATOR, found 2026-09-17 and the reason this file groups at all. A
    plan holding ONE fused pair beside single arms launches at the singles' own cost
    and often beats them; a plan holding TWO costs roughly eight times as much per
    launch and never wins. Reporting one median over both populations averages two
    things that do not overlap and hides the only actionable fact in the table.

    Counted from the arms the plan actually installed, not from the DRIVE row's
    request, because a requested arm that lost arbitration is not in the plan.
    """
    arms = set((((row.get("plans") or {}).get("fused") or {}).get("arms") or {})
               .values())
    return sum(1 for arm in arms if "pair" in arm.lower())


def control_note(row: Dict[str, Any]) -> str:
    plans = row.get("plans") or {}
    def tables(leg: str) -> str:
        got = sorted(((plans.get(leg) or {}).get("composition") or {})
                     .get("tables_dispatched") or [])
        return "+".join(got) or "array"
    return f"control ran {tables('unfused')}, fused ran {tables('fused')}"


def per_launch(row: Dict[str, Any], leg: str) -> Optional[float]:
    """Milliseconds per device launch on this leg, or None if unwitnessed.

    WHERE THE COST IS, which the per-step column cannot say. A fused product that
    is slower per STEP while issuing FEWER launches is not launch-bound -- the
    launches it does issue are individually more expensive. Dividing one measured
    column by the other separates "too many launches" from "each launch costs too
    much", and only the second is a property of the kernel.
    """
    lps = ((row.get("substitution") or {}).get("launches_per_step") or {}).get(leg)
    per = (row.get("per_leg") or {}).get(leg, {}).get("median_seconds_per_step")
    if not lps or per is None:
        return None
    return 1e3 * per / lps


def table(rows: Sequence[Dict[str, Any]], show_src: bool = False) -> List[str]:
    # ``src`` ONLY WHEN THE ROWS MIX PROVENANCES: a column that reads the same on
    # every row carries nothing the header has not said once.
    lines = ["| case | grid | fused ms/step | singles ms/step | array ms/step "
             "| vs singles | vs array | ms/launch fused:singles | pairs "
             "| tables | repair route | contended |"
             + (" src |" if show_src else ""),
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|"
             + ("---|" if show_src else "")]
    for row in rows:
        per = row["per_leg"]
        shape = "x".join(str(v) for v in row.get("grid_shape") or [])
        mark = contended(row)
        lines.append(
            f"| `{row['case']}` | {shape} "
            f"| {1e3 * per['fused']['median_seconds_per_step']:.3f} "
            f"| {1e3 * per['unfused']['median_seconds_per_step']:.3f} "
            f"| {1e3 * per['array']['median_seconds_per_step']:.3f} "
            f"| {('**%.2fx**' % row['ratios']['unfused_over_fused']) if same_table(row) else '— (%s)' % control_note(row)} "
            # THREE PLACES BECAUSE THIS RATIO GOES BELOW ONE. Two places printed
            # the 1-D smoke row's 0.0057 as "0.01x", which reads as a near-tie
            # between paths that differ by a factor of 178.
            f"| {row['ratios']['array_over_fused']:.3f}x "
            f"| {_launch_cell(row)} "
            f"| {fused_pairs(row)} "
            f"| {', '.join(row['substitution']['tables_dispatched']) or '—'} "
            f"| {route_cell(row)} "
            f"| {'yes — ' + mark if mark else 'no'} |"
            + (f" {_short(digest(row))} |" if show_src else ""))
    return lines


def headline_block(fusion: List[Dict[str, Any]]) -> List[str]:
    """The median, the one-pair / two-pair split, and the gap sentence, for ONE program.

    The gap sentence is printed only when the two per-launch ranges really do not
    meet (the cheapest two-pair launch dearer than the dearest one-pair one): it is a
    measured claim, and a set whose ranges overlap must not be told they do not.
    """
    out: List[str] = []
    if not fusion:
        return out
    one = [r for r in fusion if fused_pairs(r) <= 1]
    two = [r for r in fusion if fused_pairs(r) >= 2]
    ratios = [r["ratios"]["unfused_over_fused"] for r in fusion]
    wins = sum(1 for v in ratios if v > 1.0)
    out += [
        f"Median **{statistics.median(ratios):.2f}x** over {len(fusion)} "
        f"cases whose control ran the same table, faster on "
        f"**{wins} of {len(fusion)}**; the one-pair / two-pair split:",
        ""]
    rows_out = ["| fused plan holds | cases | ms/launch | vs singles "
                "| fused wins |", "|---|---:|---|---|---:|"]
    for label, grp in (("**one** fused pair, beside single arms", one),
                       ("**two** fused pairs", two)):
        if not grp:
            continue
        ml = [v for v in (per_launch(r, "fused") for r in grp) if v is not None]
        vs = [r["ratios"]["unfused_over_fused"] for r in grp]
        rows_out.append(
            f"| {label} | {len(grp)} "
            f"| {min(ml):.3f} – {max(ml):.3f} " if ml else
            f"| {label} | {len(grp)} | — ")
        rows_out[-1] += (f"| {min(vs):.2f}x – **{max(vs):.2f}x** "
                         f"| {sum(1 for v in vs if v > 1.0)} of {len(grp)} |")
    out += rows_out + [""]
    # ``default=None``: a row with no launch witness (a host run, where Metal serves
    # and neither NVIDIA counter is present) has no per-launch figure, and a group
    # made only of such rows raised ValueError here -- found 2026-09-19 rendering the
    # first ``--smoke`` rows. No figure, no sentence.
    hi = max((v for v in (per_launch(r, "fused") for r in one) if v is not None),
             default=None)
    lo = min((v for v in (per_launch(r, "fused") for r in two) if v is not None),
             default=None)
    if one and two and hi is not None and lo is not None and lo > hi:
        out += [
            f"The two ranges do not meet: the most expensive one-pair launch "
            f"is {hi:.3f} ms and the cheapest two-pair launch is {lo:.3f} ms, "
            f"a gap of {lo / hi:.1f}x with no row in between. **The split "
            f"is not about how many pairs are fused; it is about whether a "
            f"fused pair owns the seam the case's point source is injected "
            f"on.** Every two-pair plan necessarily holds that seam; every "
            f"one-pair winner holds its pair on the OTHER one, and the one-pair "
            f"rows that lose hold it ON the source's seam; the pair "
            f"on the source's seam carries the deposit-repair bracket, a fixed "
            f"host cost the launch count does not see. The seam-swap ablation "
            f"that proves it is under *Where these sizes sit* below.", ""]
    elif one and two and hi is not None and lo is not None:
        out += [f"The two per-launch ranges overlap (one-pair up to {hi:.3f} ms, "
                f"two-pair from {lo:.3f} ms).", ""]
    return out


def per_step(row: Dict[str, Any], leg: str) -> Optional[float]:
    """Seconds per complete step on one leg (the AB/BA median), or None."""
    return (((row.get("per_leg") or {}).get(leg) or {})
            .get("median_seconds_per_step"))


def load_sweep(directories: Sequence[str]) -> List[Dict[str, Any]]:
    """Sweep rows, one per (table, case, res), read from ``<dir>[/<table>]/<case>_res<R>``.

    The table is the parent directory's name when it is ``triton`` or ``cuda``; a
    sweep written without that level (the 2026-09-17 ones, Triton only) is read from
    the row's own ``drive_table`` and defaults to ``triton``.
    """
    import glob  # noqa: PLC0415
    import re  # noqa: PLC0415

    out: List[Dict[str, Any]] = []
    for directory in directories:
        for path in sorted(glob.glob(os.path.join(directory, "**", "rows.jsonl"),
                                     recursive=True)):
            leaf = os.path.basename(os.path.dirname(path))
            match = re.match(r"(.+)_res(\d+)$", leaf)
            if not match:
                continue
            parent = os.path.basename(os.path.dirname(os.path.dirname(path)))
            for row in load(path):
                table = (parent if parent in ("triton", "cuda")
                         else row.get("drive_table") or "triton")
                out.append(dict(row, _table=table, _case=match.group(1),
                                _res=int(match.group(2))))
    return out


def _cells(row: Dict[str, Any]) -> int:
    total = 1
    for extent in (row.get("grid_shape") or row.get("grid") or []):
        total *= int(extent)
    return total


def sweep_lines(after: List[Dict[str, Any]], before: List[Dict[str, Any]]) -> List[str]:
    """The size sweep, rendered from its rows: per (table, case), every size measured.

    ``before`` rows (a sweep timed on older bytes) are shown beside the ``after`` ones
    at the sizes both measured, so the change is read off two measurements and never
    off a remembered figure. The crossover sentence is DERIVED: the smallest measured
    size at which the fused step beats the same-table singles and the largest at which
    it does not -- a bracket, never an interpolation.
    """
    usable = [r for r in after if r.get("reportable") and same_table(r)
              and per_step(r, "fused") and per_step(r, "unfused") and per_step(r, "array")]
    if not usable:
        return []
    old = {(r["_table"], r["_case"], _cells(r)): r for r in before
           if per_step(r, "fused") and per_step(r, "unfused")}
    out = ["## Size sweep", "",
           "Same case, same composition, only the grid moving (`--res`; one 10 x 6 "
           "um cell, 60 R^2 cells). Every row below passed every floor, its control "
           "ran the same kernel table, and its fused leg's deposit-repair route is "
           "the one recorded in the last column.", ""]
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in usable:
        groups.setdefault((r["_table"], r["_case"]), []).append(r)
    for (table, case) in sorted(groups):
        grp = sorted(groups[(table, case)], key=_cells)
        pairs = fused_pairs(grp[0])
        out += [f"**{table} `{case}`** ({pairs} fused pair{'s' if pairs != 1 else ''})",
                "",
                "| cells | fused ms/step | singles | array | vs singles | vs array "
                "| before: fused / vs singles | repair route |",
                "|---:|---:|---:|---:|---:|---:|---|---|"]
        ratio = {id(r): per_step(r, "unfused") / per_step(r, "fused") for r in grp}
        for r in grp:
            f, u, a = (per_step(r, leg) for leg in ("fused", "unfused", "array"))
            prev = old.get((table, case, _cells(r)))
            was = (f"{1e3 * per_step(prev, 'fused'):.3f} / "
                   f"{per_step(prev, 'unfused') / per_step(prev, 'fused'):.3f}x"
                   if prev is not None else "—")
            vs = ratio[id(r)]
            out.append(f"| {_cells(r):,} | {1e3 * f:.3f} | {1e3 * u:.3f} | "
                       f"{1e3 * a:.3f} | {'**' if vs > 1 else ''}{vs:.3f}x"
                       f"{'**' if vs > 1 else ''} | {a / f:.3f}x | {was} "
                       f"| {route_verdict(repair_route(r))} |")
        wins = [_cells(r) for r in grp if ratio[id(r)] > 1.0]
        loses = [_cells(r) for r in grp if ratio[id(r)] <= 1.0]
        if not wins:
            verdict = f"is behind its singles at every measured size, up to {max(loses):,} cells"
        elif not loses:
            verdict = "is ahead of its singles at every measured size"
        elif max(loses) < min(wins):
            verdict = f"crosses its singles between {max(loses):,} and {min(wins):,} cells"
        elif max(wins) < min(loses):
            verdict = (f"is ahead of its singles up to {max(wins):,} cells and not from "
                       f"{min(loses):,}")
        else:
            verdict = f"alternates around its singles ({len(wins)} of {len(grp)} sizes ahead)"
        # THE FLAT FLOOR, derived: the sizes over which the fused step stays within 10%
        # of its smallest value -- the host-bound regime -- and the range it spans.
        fastest = min(per_step(r, "fused") for r in grp)
        flat = [r for r in grp if per_step(r, "fused") <= 1.10 * fastest]
        floor = ""
        if len(flat) >= 2:
            lo_ms = 1e3 * min(per_step(r, "fused") for r in flat)
            hi_ms = 1e3 * max(per_step(r, "fused") for r in flat)
            floor = (f" Its step is flat at {lo_ms:.2f}-{hi_ms:.2f} ms from "
                     f"{_cells(flat[0]):,} to {_cells(flat[-1]):,} cells, the regime "
                     f"a fixed per-step host cost sets.")
        out += ["", f"The fused step {verdict}.{floor}", ""]
    out += [
        "**Where the fixed cost comes from, and what was removed.** A fused pair that "
        "owns the seam the case's point source is injected on is bracketed by the "
        "deposit repair (`deposit_repair.LeadingRepairPlan.save` before the launch, "
        "`TrailingRepairPlan.apply` after the driver's inject); a seam-swap ablation "
        "proved the cost follows the SOURCE'S seam and neither kernel is slow "
        "(`results/fused_ablation_2026-09-17_seamswap/`, `fused_ablation2_...`). The "
        "2026-09-19 round rewrote the bracket onto a linear index cached per plan "
        "(bit-identical: 0 of 84,449,844 words against the previous arithmetic) and "
        "stopped the CUDA products re-emitting their kernel source on every launch; "
        "the `before` column is the same sweep on the earlier bytes. What remains of "
        "the floor is the bracket's residual small device operations and the "
        "launches themselves: a one-launch repair kernel is the next lever, and it "
        "would be a new certified product.", ""]
    return out


# ---------------------------------------------------------------------------
# The Metal table: its own section, table and headline
# ---------------------------------------------------------------------------

def _residency(row: Dict[str, Any], leg: str = "fused") -> Dict[str, Any]:
    return ((row.get("residency") or {}).get(leg)) or {}


def _span(values: Sequence[float], fmt: str = "{:g}") -> str:
    """``lo–hi`` over what was recorded, one value when they agree, a dash for none."""
    found = sorted(v for v in values if v is not None)
    if not found:
        return "—"
    if found[0] == found[-1]:
        return fmt.format(found[0])
    return f"{fmt.format(found[0])}–{fmt.format(found[-1])}"


def _power_cell(row: Dict[str, Any]) -> str:
    """What the laptop was running on while the case ran, from both host probes."""
    states = [row.get(when) or {} for when in ("host_before", "host_after")]
    on_ac = [s.get("on_ac_power") for s in states if s.get("on_ac_power") is not None]
    if not on_ac:
        return "—"
    text = "AC" if all(on_ac) else "battery" if not any(on_ac) else "AC→battery"
    charge = [s.get("battery_percent") for s in states
              if s.get("battery_percent") is not None]
    if charge and not all(on_ac):
        text += f" {min(charge)}%"
    if any(s.get("low_power_mode") for s in states):
        text += ", low power"
    limits = [s.get("cpu_speed_limit") for s in states
              if s.get("cpu_speed_limit") is not None]
    if limits and min(limits) < 100:
        text += f", CPU limited to {min(limits)}%"
    return text


def _load_cell(row: Dict[str, Any]) -> str:
    loads = [(row.get(when) or {}).get("loadavg") for when in ("host_before",
                                                              "host_after")]
    loads = [load[0] for load in loads if load]
    return " → ".join(f"{v:.1f}" for v in loads) if loads else "—"


def metal_witnessed(row: Dict[str, Any]) -> bool:
    """Was this row measured WITH the Metal table's own launch witness?

    Until 2026-09-27 the Metal table also served a host lift started for another
    drive table (``--drive-table triton --smoke`` on a laptop; such a lift is the
    NumPy reference since, and reaches no table), and such a row has no launch
    witness at all: neither NVIDIA counter sees a Metal launch, so its launches read
    zero on every leg and its substitution rests on the slot map alone. It is a row
    about the Metal table that the Metal instruments never measured, and it is kept
    out of that table's headline and launch columns by this one read.
    """
    return "metal_launch_counter" in ((row.get("substitution") or {})
                                      .get("instruments") or [])


def metal_table(rows: Sequence[Dict[str, Any]]) -> List[str]:
    lines = ["| case | grid | fused ms/step | singles ms/step | array ms/step "
             "| vs singles | vs array | launches/step fused:singles "
             "| ms/launch fused:singles | mirrors | MB in / out a bracket | power "
             "| load 1 min | contended |",
             "|---|---|---:|---:|---:|---:|---:|---|---|---:|---|---|---|---|"]
    for row in rows:
        per = row["per_leg"]
        shape = "x".join(str(v) for v in row.get("grid_shape") or [])
        lps = (row.get("substitution") or {}).get("launches_per_step") or {}
        f, u = per_launch(row, "fused"), per_launch(row, "unfused")
        facts = _residency(row)
        moved = (f"{facts['mirrored_bytes'] / 1e6:.2f} / "
                 f"{(facts.get('mirrored_bytes_out') or 0) / 1e6:.2f}"
                 if facts.get("mirrored_bytes") else "—")
        mark = contended(row)
        seen = metal_witnessed(row)
        lines.append(
            f"| `{row['case']}` | {shape} "
            f"| {1e3 * per['fused']['median_seconds_per_step']:.3f} "
            f"| {1e3 * per['unfused']['median_seconds_per_step']:.3f} "
            f"| {1e3 * per['array']['median_seconds_per_step']:.3f} "
            f"| {('**%.2fx**' % row['ratios']['unfused_over_fused']) if same_table(row) else '— (%s)' % control_note(row)} "
            f"| {row['ratios']['array_over_fused']:.3f}x "
            # AN UNWITNESSED ROW'S ZEROES ARE NOT LAUNCH COUNTS, and are not printed.
            f"| {'%g : %g' % (lps.get('fused') or 0, lps.get('unfused') or 0) if seen else 'no witness'} "
            f"| {'—' if not seen or f is None or u is None else '%.3f : %.3f' % (f, u)} "
            f"| {facts.get('mirrors', '—')} | {moved} "
            f"| {_power_cell(row)} | {_load_cell(row)} "
            f"| {'yes — ' + mark if mark else 'no'} |")
    return lines


def metal_section(rows: Sequence[Dict[str, Any]],
                  good: Sequence[Dict[str, Any]]) -> List[str]:
    """Everything the document says about rows the Metal table served.

    ``rows`` is every Metal-population row, ``good`` those that passed the floors that
    apply to any measurement. EVERY FIGURE AND EVERY FACT BELOW IS READ OFF THE ROWS --
    the residency mode is the string the plan recorded when it froze, the sync call is
    the one the bench recorded making, the power source is what the host probe read --
    so a campaign run under another residency mode, or on mains, renders as that.
    """
    out = ["## The Metal table", ""]
    timed = [r for r in rows if (r.get("per_leg") or {}).get("fused")]
    out += [
        f"{len(rows)} case{'s' if len(rows) != 1 else ''} served by the Metal table, "
        f"{len(timed)} timed, {len(good)} passing every floor. These rows are a "
        "different kind of measurement from the NVIDIA ones and share no headline with "
        "them. The Metal table steps a HOST (NumPy) engine with persistent device "
        "mirrors beside it, and every dispatched plan call sits in a residency bracket: "
        "copy every mirror in, launch, synchronise the device, copy every non-constant "
        "mirror out. The legs are the same three, from one frozen state, in AB/BA "
        "windows with monitors detached; on a row measured with `--drive-table metal` "
        "the substitution is proved by the route "
        "gate's own proof (`gate_dispatch_metal_route.substitution_proof`, which must "
        "read EXACT, with `fused_leg_is_real` and `array_leg_is_clean`), counted twice "
        "over by its `MetalLaunchCounter` — the plans' booked launches and the "
        "compiled calls. Such a row passes the same floors as any other and one more: "
        "every timed window launched at exactly the rate the substitution was proved "
        "at, with the two counts equal.", ""]
    blind = sorted(r["case"] for r in rows
                   if (r.get("per_leg") or {}).get("fused") and not metal_witnessed(r))
    if blind:
        out += [
            f"**{len(blind)} of {len(timed)} timed rows were measured WITHOUT this "
            f"table's launch witness** ({', '.join(f'`{c}`' for c in blind)}): the Metal "
            f"table served a host lift started for another drive table, where neither "
            f"NVIDIA counter can see a Metal launch. Their substitution rests on the "
            f"slot map alone, their launch columns read *no witness*, and they enter "
            f"no headline figure.", ""]
    # THE MEDIAN IS A QUIET-BOX NUMBER OR IT IS NOTHING. This host has ONE device and
    # shares it with whatever else is driving it, so a contended row is a row about a
    # busy machine; averaged in, it moves the headline by an amount nobody can
    # subtract. The two are reported the way the CPU baseline reports its pair --
    # separately, with the difference named as the cost of contention.
    measured = [r for r in good
                if same_table(r) and not is_smoke(r) and metal_witnessed(r)]
    headline = [r for r in measured if not contended(r)]
    loud = [r for r in measured if contended(r)]
    if headline:
        n = len(headline)
        vs = [r["ratios"]["unfused_over_fused"] for r in headline]
        arr = [r["ratios"]["array_over_fused"] for r in headline]
        wins = sum(1 for v in vs if v > 1.0)
        slower = sum(1 for v in arr if v < 1.0)
        out += [
            f"Median **{statistics.median(vs):.2f}x** over {n} case"
            f"{'s' if n != 1 else ''} whose control ran the same table, the fused step "
            f"faster than the certified singles it replaces on **{wins} of {n}**. "
            f"Fusing a pair removes a launch, and on this table a launch is a whole "
            f"bracket.", ""]
        if slower:
            behind = statistics.median([1.0 / v for v in arr if v > 0])
            out += [
                f"Against the array path the fused step is SLOWER than the array path "
                f"on **{slower} of {n}** rows (median `vs array` "
                f"{statistics.median(arr):.3f}x; the array path is a median "
                f"**{behind:.1f}x** faster). That is the measurement, not a defect in "
                f"it: at these sizes the bracket costs more than the step it "
                f"surrounds.", ""]
        else:
            out += [f"Against the array path the fused step is faster on all {n} rows "
                    f"(median `vs array` {statistics.median(arr):.3f}x).", ""]
    if loud:
        vs = [r["ratios"]["unfused_over_fused"] for r in loud]
        arr = [r["ratios"]["array_over_fused"] for r in loud]
        listed = ", ".join(f"`{r['case']}`" for r in loud)
        if headline:
            out += [
                f"**{len(loud)} of {len(measured)} rows were measured while another "
                f"process held the device and are not in that median** ({listed}): "
                f"over those alone the median is **{statistics.median(vs):.2f}x** "
                f"against the singles and {statistics.median(arr):.3f}x against the "
                f"array path. They are reported beside the quiet rows rather than "
                f"dropped, for the reason the CPU baseline keeps a contended dataset "
                f"beside its quiet one: the difference between the two is what "
                f"contention costs on this machine.", ""]
        else:
            out += [
                f"**No row was measured on a quiet device, so there is no quiet-box "
                f"median.** Over the {len(loud)} contended row"
                f"{'s' if len(loud) != 1 else ''} ({listed}) the median is "
                f"**{statistics.median(vs):.2f}x** against the singles and "
                f"{statistics.median(arr):.3f}x against the array path, and neither "
                f"figure may be quoted as a quiet-box number.", ""]
    smoke = sum(1 for r in good if is_smoke(r))
    if smoke:
        out += [f"{smoke} `--smoke` row(s) are tabled below and enter no headline "
                "figure.", ""]
    if good:
        ordered = sorted(good, key=lambda r: -r["ratios"]["array_over_fused"])
        out += metal_table(ordered) + [""]
    else:
        out += ["No Metal row passed every floor; the refusals are listed below and "
                "none of them is a measurement.", ""]

    # THE RESIDENCY CONTEXT, which is what a Metal number is a number ABOUT.
    basis = good or timed
    if basis:
        invariants = sorted({_residency(r).get("invariant") for r in basis
                             if _residency(r).get("invariant")})
        modes = sorted({_residency(r).get("mode") for r in basis
                        if _residency(r).get("mode")})
        mirrors = _span([_residency(r).get("mirrors") for r in basis])
        launch_ms = [v for v in (per_launch(r, leg) for r in good if metal_witnessed(r)
                                 for leg in ("fused", "unfused")) if v is not None]
        brackets = [((r["per_leg"].get("fused") or {}).get("residency_syncs_per_step")
                     or {}).get("out") for r in basis]
        text = (f"**Residency.** Every row ran under the invariant "
                f"*{' / '.join(invariants) or 'not recorded'}*, over {mirrors} mirrors "
                f"a composition")
        if modes:
            text += f" (the plan's own statement of its mode: *{modes[0]}*)"
        text += "."
        if launch_ms:
            text += (f" A launch cost {_span(launch_ms, '{:.3f}')} ms across the fused "
                     f"and singles legs of the rows above")
            sized = sorted((r for r in good if metal_witnessed(r)
                            and per_launch(r, "fused") is not None), key=_cells)
            if len(sized) >= 2 and _cells(sized[0]) != _cells(sized[-1]):
                text += (f" — {per_launch(sized[0], 'fused'):.3f} ms at "
                         f"{_cells(sized[0]):,} cells and "
                         f"{per_launch(sized[-1], 'fused'):.3f} ms at "
                         f"{_cells(sized[-1]):,}")
            text += "."
        if any(b is not None for b in brackets):
            text += (f" The fused leg paid {_span(brackets, '{:g}')} bracket(s) a "
                     f"step, read off the plan's own residency counters.")
        out += [text, ""]
        calls = sorted({(r.get("sync") or {}).get("call") for r in basis
                        if (r.get("sync") or {}).get("call")})
        idle = [(r.get("sync") or {}).get("idle_seconds") for r in basis]
        idle = [v for v in idle if v is not None]
        out += [
            f"**Window edges and compiles.** Every timed window begins and ends on "
            f"`{'` / `'.join(calls) or 'not recorded'}`"
            + (f" (median idle cost {1e6 * statistics.median(idle):.2f} µs, so the "
               f"array leg's two edge calls are not what its window measures)"
               if idle else "")
            + ". Under the per-launch `shipped` bracket that call is redundant by "
            "construction — the bracket synchronises before it copies the mirrors out; "
            "under `held` a fully-held bracket does not drain, and the call is what "
            "ends the window on the device. It is made in both modes so the window does "
            "not depend on the bracket. A compile on "
            "this table is a `torch.mps.compile_shader` call, made at plan build and "
            "memoised by source; the floor requires zero calls to that entry point, "
            "zero growth of the package's compile memo, no re-frozen plan and no "
            "compiled function first launched, inside every timed window.", ""]
        hosts = [r.get("host_before") or {} for r in basis]
        models = sorted({f"{h.get('model')} ({h.get('cpu')})" for h in hosts
                         if h.get("model")})
        known = [h for h in hosts if h.get("on_ac_power") is not None]
        battery = sum(1 for h in known if not h["on_ac_power"])
        low = sum(1 for h in hosts if h.get("low_power_mode"))
        loads = [h["loadavg"][0] for h in hosts if h.get("loadavg")]
        limited = sum(1 for r in basis for when in ("host_before", "host_after")
                      if ((r.get(when) or {}).get("cpu_speed_limit") or 100) < 100)
        out += [
            f"**The host is a laptop: {', '.join(models) or 'model not recorded'}.** "
            f"{battery} of {len(known)} rows on battery"
            f"{'' if len(known) == len(basis) else f' ({len(basis) - len(known)} did not record a power source)'}"
            f", low power mode on {low} of {len(basis)}, one-minute load average "
            f"{_span(loads, '{:.1f}')} before the case, a thermal CPU limit recorded on "
            f"{limited} probe(s), and another process driving the Metal device during "
            f"{sum(1 for r in basis if contended(r))} of {len(basis)}. The same machine "
            f"clocks differently on battery, in low power mode and under a thermal "
            f"limit, so a row is a statement about the state in its `power` column and "
            f"must not be quoted without it.", ""]
    return out


def render(rows: List[Dict[str, Any]], title: str,
           sweep: Sequence[Dict[str, Any]] = (),
           sweep_before: Sequence[Dict[str, Any]] = ()) -> str:
    # TWO POPULATIONS. A row whose control fell through to the other table has NO
    # fusion number -- but its ``vs array`` ratio is untouched by which table the
    # control used, so refusing the whole row would throw away a good measurement to
    # punish a bad one. They are tabled together and the fusion column is blanked
    # with the reason; only ``fusion`` rows enter the headline median.
    every_good = [r for r in rows if r.get("reportable") or (
        r.get("ratios") and r.get("floors")
        and all(v for k, v in r["floors"].items()
                if k not in ("singles_are_the_same_table",)))]
    # AND TWO DEVICE FAMILIES. Rows the Metal table served get their own section
    # (:func:`metal_section`); ``good`` and ``fusion`` below are the NVIDIA rows, so
    # the headline median is over exactly the rows it was over before a Metal row
    # existed.
    metal_rows = [r for r in rows if population(r) == "metal"]
    metal_good = [r for r in every_good if population(r) == "metal"]
    good = [r for r in every_good if population(r) != "metal"]
    fusion = [r for r in good if same_table(r)]
    good.sort(key=lambda r: -r["ratios"]["array_over_fused"])
    out = [f"# {title}", ""]

    if not rows:
        return "\n".join(out + ["No rows.", ""])

    stamps = {r.get("utc", "")[:10] for r in rows if r.get("utc")}
    tables = sorted({t for r in rows for t in
                     (r.get("substitution", {}).get("tables_dispatched") or [])})
    out += [
        f"Generated from `rows.jsonl` — {len(rows)} cases attempted, "
        f"{len(every_good)} timed cleanly, of which "
        f"**{sum(1 for r in every_good if same_table(r))} carry a fusion "
        f"number**; dispatched by {', '.join(tables) or 'no table'}, "
        f"{'/'.join(sorted(stamps))}.",
        "",
        "Every number is a median over the campaign's AB/BA windows, measured from "
        "one frozen state with monitors detached. A row reaches the table below "
        "only if it passed the six floors that apply to any measurement: the "
        "substitution was proved, the fused leg was bit-identical to the array "
        "path, every leg moved bits and produced no non-finite value, no kernel "
        "compiled inside a timed window, and every leg's spread stayed inside the "
        "gate. A SEVENTH floor governs the `vs singles` column alone — the control "
        "must have run the same kernel table as the subject — so a row can be a "
        "sound measurement against the array path and carry no fusion number.",
        "",
    ]
    out += provenance_lines(rows) + [""]
    # OVER THE NVIDIA POPULATION, because this decides the NVIDIA table's shape: how
    # many PROGRAMS its rows were timed on, which splits its headline into per-digest
    # buckets and adds the `src` column. A Metal row pins its own sources and is timed
    # on another machine, so its digest differs as a matter of course; counted here it
    # said two programs were timed where one was, and marked every NVIDIA row with a
    # provenance it does not disagree with.
    mixed = len({digest(r) for r in rows if population(r) != "metal"}) > 1

    if good:
        out += ["## Fused products against the certified singles they replace", ""]
        # THE HEADLINE IS COMPUTED PER PROGRAM, never over two. Rows timed on
        # different deposit_repair bytes are different programs -- the 2026-09-19
        # repair changed the bracket's cost by an order of magnitude at large grids --
        # so a mixed set gets one headline per digest bucket, and a ``--smoke`` row
        # (lifted on the host array module) enters no headline at all.
        headline = [r for r in fusion if not is_smoke(r)]
        smoke_rows = len(fusion) - len(headline)
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for r in headline:
            buckets.setdefault(digest(r), []).append(r)
        for key in sorted(buckets, key=lambda k: (k == "not recorded", k)):
            if mixed:
                out += [f"### Rows timed on deposit_repair `{key}` "
                        f"({len(buckets[key])} of {len(headline)})", ""]
            out += headline_block(buckets[key])
        if smoke_rows:
            out += [f"{smoke_rows} `--smoke` row(s) lifted on the host array module "
                    "are tabled below and enter no headline figure.", ""]
        if len(good) > len(fusion):
            out += [
                f"**{len(good) - len(fusion)} of {len(good)} rows have no fusion "
                f"number**, and their `vs singles` cell says why. Their control leg "
                f"was served by a DIFFERENT kernel table than their fused leg, so "
                f"that ratio would compare two tables rather than measure fusing. "
                f"The cause is a documented property, not a harness fault: outside "
                f"the two certified real-PML single seams the CUDA table adopts "
                f"(`fastpath_cuda.SINGLE_ARM_SEAMS`), veto its fused arms and it "
                f"offers nothing at all, so dispatch falls through to the Triton "
                f"table's singles. Their `vs array` column is unaffected and "
                f"stands.", ""]
        out += table(good, show_src=mixed) + [""]
    elif not metal_rows:
        out += ["## No reportable rows", "",
                "Every case failed at least one floor. The refusals are below; none "
                "of them is a speedup measurement, and none should be read as one.",
                ""]

    if metal_rows:
        out += metal_section(metal_rows, metal_good)

    refused = [r for r in rows if not r.get("reportable")]
    if refused:
        out += ["## Refused, by name and by the floor that refused them", ""]
        by_floor: Dict[str, List[str]] = {}
        # A Metal row is marked only in a document that also holds NVIDIA rows, where
        # one case name can be refused on one device family and measured on the other.
        marked = len(metal_rows) < len(rows)
        for row in refused:
            floors = row.get("floors") or {}
            failed = [k for k, v in floors.items() if not v] or [
                row.get("verdict", "ERROR")]
            for name in failed:
                by_floor.setdefault(name, []).append(
                    f"{row['case']} (metal)" if marked and population(row) == "metal"
                    else row["case"])
        for name in sorted(by_floor):
            cases = ", ".join(f"`{c}`" for c in sorted(set(by_floor[name])))
            out += [f"- **{name}** ({len(set(by_floor[name]))}): {cases}"]
        out += [""]

    out += route_lines(rows)

    out += sweep_lines(list(sweep), list(sweep_before))

    # WHAT THE DOCUMENT LICENSES, stated in the document rather than left to the
    # reader -- the shape every benchmark in this tree carries.
    out += [
        "## What these rows license, and what they do not", "",
        "A row licenses one statement: *on this case, on this host, this fused "
        "product ran the whole step in 1/X of the time the certified single arms "
        "took, with its arithmetic proved identical to the array path*.", "",
        "They do **not** license:", "",
        "- a corpus-wide claim. These are the route gates' DRIVE witnesses, chosen "
        "to exercise distinct compositions, not sampled from the corpus by how "
        "often a shape occurs in it;",
        "- an end-to-end or application speedup. Monitors are detached and the "
        "window is the step loop;",
        "- a GPU-vs-CPU claim, which the design notes (meep-gpu-vs-cpu-throughput) answer and "
        "this campaign does not re-derive;",
        "- anything about a case not in the table, including the cases listed as "
        "refused above;",
        # THE ROUTE IS RECORDED, NOT FLOORED, so what it licenses has to be said here.
        "- a statement about the linear-index deposit repair "
        "(`deposit_repair.py`, 2026-09-18/19) from any row whose `repair route` does "
        "not read `linear`. A row reading *3-tuple fallback* or *mixed* is a sound "
        "measurement of the route that ran, which is not the linear one; a row "
        f"reading *{NOT_RECORDED}* was written before the counters existed "
        "(2026-09-19) and cannot say which route ran, nor — without the provenance "
        "block — which repair code it timed. The deposit-repair bracket figures "
        "quoted below (*Where the two-pair overhead lives*) were measured on "
        "2026-09-17 rows, before the linear route, and describe that code;",
    ]
    # THE NEXT THREE PARAGRAPHS QUOTE THE NVIDIA CAMPAIGN'S OWN FIGURES (its cell
    # counts, its spread at a fixed size, the 2026-08-21 kernel rows), so they are
    # printed where an NVIDIA row is and not over a document of Metal rows alone.
    if len(metal_rows) < len(rows):
        out += [
            "- **any statement about how these ratios move with PROBLEM SIZE.** Each "
            "DRIVE case carries ONE fixed grid, chosen to exercise a composition rather "
            "than to sit on a size ladder, so size is confounded with composition "
            "throughout. The spread AT a fixed cell count is larger than the spread "
            "across cell counts — at 400 cells two cases differ 17x against the array "
            "path, and at 24,000 cells four cases span 0.67x to 2.78x — so no crossover "
            "can be read out of this table, and none is claimed.", "",
            "**Where these sizes sit, and it matters more than any row above.** 40 of "
            "the 44 distinct (case, grid) pairs are at or below 25,000 cells; the largest "
            "is 110,592 -- the regime where a fixed per-step host cost dominates. How the "
            "same composition behaves as ONLY the grid grows is measured separately and "
            "rendered from its own rows under *Size sweep* below.", "",
            "The `vs array` column is comparable to the 2026-08-21 kernel rows and is "
            "**not** the fusion's speedup: the array path is the reference the "
            "arithmetic is proved against, not the baseline the product replaces.", "",
        ]
    else:
        out += [
            "- any statement about how these ratios move with PROBLEM SIZE. Each DRIVE "
            "case carries ONE fixed grid, chosen to exercise a composition, so size is "
            "confounded with composition throughout.", "",
            "The `vs array` column is **not** the fusion's speedup: the array path is "
            "the reference the arithmetic is proved against, not the baseline the "
            "product replaces.", ""]
    if metal_rows:
        out += [
            "**Every Metal row is a row about the residency mode its plan recorded "
            "(stated under *Residency* above), on one laptop.** Under the per-launch "
            "`shipped` bracket it prices the bracket around a launch at least as much "
            "as the kernel inside it; under `held`, the package default since "
            "2026-09-27, it prices the kernels with the whole-registry copies removed. "
            "It licenses no statement about another residency mode, none about "
            "another machine, and none about this machine in a power or thermal state "
            "other than the one the row records. It is not a GPU-vs-CPU claim either: "
            "the array leg is this table's own host engine, stepped in the same "
            "process under the same subnormal policy, and is the reference the fused "
            "arithmetic is proved identical to.", ""]
    # THE CUDA CLAUSE IS NOT OPTIONAL AND IS NOT TYPED IN. Every CUDA row describes
    # a path a user reaches ONLY by setting the preference: under the SHIPPED
    # precedence the release-gated Triton table composes first and holds every slot
    # both admit, which is why the board's own
    # ``by_default_precedence.served_in_dispatch`` reads 0. A CUDA speedup quoted
    # without that sentence is a number about a path nobody is on by default.
    if any((r.get("drive_table") == "cuda")
           or ("cuda" in (r.get("substitution", {}).get("tables_dispatched") or []))
           for r in rows):
        out += [
            "**Every CUDA row requires `MEEP_GPU_BACKEND_PREFERENCE=cuda`.** Under "
            "the shipped precedence the release-gated Triton table composes first "
            "and holds every slot both tables admit, so the hand-CUDA table serves "
            "nothing by default — the board's own `by_default_precedence."
            "served_in_dispatch` reads 0. These rows measure a path that is "
            "reachable and released, but not the one a user is on unless they ask "
            "for it by name.", ""]
    if any(contended(r) for r in rows):
        out += [
            "**Contended rows.** Some rows were measured while another process held "
            "a device, marked in the last column. They are kept rather than dropped, "
            "for the reason the CPU baseline keeps its contended dataset beside its "
            "quiet one: the difference between them is what contention costs. A "
            "contended row is not a quiet-box number and must not be quoted as one.",
            ""]
    return "\n".join(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", required=True, nargs="+",
                        help="one or more rows.jsonl files")
    parser.add_argument("--out", required=True)
    parser.add_argument("--sweep", nargs="*", default=[],
                        help="size-sweep directories (<dir>[/<table>]/<case>_res<R>/rows.jsonl)")
    parser.add_argument("--sweep-before", nargs="*", default=[],
                        help="an older sweep, shown beside the new one at shared sizes")
    parser.add_argument("--title", default="Fused products against the singles "
                                           "they replace")
    args = parser.parse_args(argv)
    rows: List[Dict[str, Any]] = []
    for path in args.rows:
        if not os.path.isfile(path):
            print(f"missing: {path}", file=sys.stderr)
            return 2
        rows.extend(load(path))
    text = render(rows, args.title, load_sweep(args.sweep), load_sweep(args.sweep_before))
    with open(args.out, "w") as handle:
        handle.write(text)
    good = sum(1 for r in rows if r.get("reportable"))
    print(f"{args.out}: {len(rows)} rows, {good} reportable", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
