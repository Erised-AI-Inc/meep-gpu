#!/usr/bin/env python
# SPDX-License-Identifier: GPL-2.0-or-later
"""Table 2, re-cut as MEEP identity: one row per corpus case, words against MEEP.

    python parity/meep_gpu/build_meep_identity_table.py --archive DIR --manifest FILE
        [--context-l2 ROWS.jsonl] [--set mac_keep ...] --out-md FILE [--out-json FILE]
        [--title TEXT] [--provenance TEXT]

WHAT THE TABLE SAYS. Table 2 used to report one relative L2 per corpus row: the lifted
array path against MEEP, stepped on the same simulation. It now reports words: for every
row and every reference (a host and a subnormal mode) the number of compared stored words
(W) and ``get_array`` words (V), and how many of each differ from MEEP's at every compared
step (every step 0 to 20, Table 2's ten checkpoints and its N). A row holds in a reference
when it reads "0 of W / 0 of V" there. The old L2 survives as CONTEXT only: the L2 of the
array path against the certification MEEP of the Mac lock, which fuses multiplies and
adds, measured now (``meep_identity_context_l2.py``) beside the value Table 2 of record
printed for the 0.9 array path.

INPUTS, all records written by tools:

* ``--archive``: the Tier B record (``record_meep_identity_archive.py``,
  ``meep_identity_tier_b.json``) with one set per reference; ``--set`` names the sets
  and their column order (default: every set, sorted).
* ``--manifest``: the corpus manifest (``meep_identity_corpus.py manifest``) for each
  row's kind (example or test), its cells and N, and the L2 of record.
* ``--context-l2``: the rows of ``meep_identity_context_l2.py run``.

L0, MEEP against MEEP. For each pair of sets in the same mode on different hosts, every
row that ran on both is classed by its MEEP digests (``digests.json``: one SHA-256 per
compared step and component, stored and ``get_array``): ``equal`` when every digest is
equal, else the first differing step and component. Where the two MEEPs differ, each
host's MEEP is still the reference its own row is judged against.

Nothing is transcribed: every count comes from the records, and a row a record lacks is
printed as not run, with the record's reason.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SCHEMA = "meep_gpu.meep_identity_table"
VERSION = 1
ARCHIVE_FILE = "meep_identity_tier_b.json"


def _n(value: Any) -> str:
    return "-" if value is None else f"{int(value):,}"


def mode_of(name: str) -> str:
    return name.rsplit("_", 1)[-1]


def host_of(name: str) -> str:
    return name.rsplit("_", 1)[0]


def cell_text(row: Optional[Dict[str, Any]], reference: Optional[Dict[str, Any]] = None) -> str:
    """One row in one set: ``0 of W / 0 of V``, ``S of W / R of V``, or ``not run``.

    With ``reference`` (the same row in the first set), a cell whose W and V equal it is
    shortened to ``S / R``.
    """
    if row is None:
        return "not run: no record"
    if row["status"] == "refused":
        return "not run (reason above)"
    stored, reader = row.get("stored") or {}, row.get("reader") or {}
    s, w = stored.get("word_differing"), stored.get("words")
    r, v = reader.get("word_differing"), reader.get("words")
    same = (reference is not None and (reference.get("stored") or {}).get("words") == w
            and (reference.get("reader") or {}).get("words") == v)
    text = f"{_n(s)} / {_n(r)}" if same else f"{_n(s)} of {_n(w)} / {_n(r)} of {_n(v)}"
    if row["status"] == "error":
        text += " (error: " + short_reason((row.get("problems") or ["?"])[0]) + ")"
    elif row["status"] == "differs":
        first = row.get("first_difference") or {}
        if first:
            text += f" @{first.get('step')}:{first.get('component')}"
    return text


def short_reason(text: str, limit: int = 140) -> str:
    """A refusal's reason on one line, without its leading exception name."""
    text = " ".join(str(text).split())
    for marker in ("MeepSimulationNotLiftable: ", "RuntimeError: "):
        if text.startswith(marker):
            text = text[len(marker):]
    return text if len(text) <= limit else text[: limit - 3] + "..."


def read_digests(archive: Path, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    spec = (row.get("files") or {}).get("digests")
    if not spec:
        return None
    return json.loads((archive / spec["file"]).read_text(encoding="utf-8"))


def read_currents(archive: Path, row: Dict[str, Any]) -> Optional[Any]:
    """The ``sources`` block of a row's recorded currents (its provenance names the host)."""
    spec = (row.get("files") or {}).get("currents")
    if not spec:
        return None
    return json.loads((archive / spec["file"]).read_text(encoding="utf-8")).get("sources")


def l0_class(one: Optional[Dict[str, Any]], two: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """MEEP against MEEP for one row: equal, or the first differing (step, component)."""
    if one is None or two is None:
        return {"class": "not comparable", "why": "a digest record is missing on one host"}
    if [int(s) for s in one["steps"]] != [int(s) for s in two["steps"]]:
        return {"class": "not comparable", "why": "different schedules"}
    pairs = equal = 0
    first = None
    for step in one["steps"]:
        key = str(step)
        for component in one["components"]:
            for kind in ("stored", "reader"):
                pairs += 1
                same = one[kind][key][component] == two[kind][key][component]
                equal += same
                if not same and first is None:
                    first = {"step": int(step), "component": component, "kind": kind}
    if first is None:
        return {"class": "equal", "pairs": pairs, "equal": equal}
    return {"class": "differ", "pairs": pairs, "equal": equal, "first": first}


def source_class(facts: Dict[str, Any]) -> str:
    """Why two MEEPs may step different sources, from the row's fact block."""
    sources = set(facts.get("source_kinds") or [])
    times = set(facts.get("src_time_kinds") or [])
    if sources & {"EigenModeSource", "EigenmodeSource"}:
        return "eigenmode source (each MEEP's own MPB solve)"
    if "CustomSource" in times:
        return "custom source function"
    return "source time functions each MEEP evaluates with its own host's math library"


def build(archive: Path, manifest: Dict[str, Any], set_names: List[str],
          context: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    record = json.loads((archive / ARCHIVE_FILE).read_text(encoding="utf-8"))
    sets = record["sets"]
    names = set_names or sorted(sets)
    missing = [name for name in names if name not in sets]
    if missing:
        raise ValueError(f"the archive holds no set {missing}; it holds {sorted(sets)}")
    rows_meta = {row["label"]: row for row in manifest["rows"]}
    table: List[Dict[str, Any]] = []
    for label, meta in sorted(rows_meta.items(), key=lambda item: (item[1]["kind"] != "example",
                                                                    item[1]["table2"]["grid_cells"]
                                                                    * item[1]["table2"]["driver_steps"],
                                                                    item[0])):
        entry = {"label": label, "kind": meta["kind"], "cells": meta["table2"]["grid_cells"],
                 "n_steps": meta["table2"]["driver_steps"],
                 "l2_of_record": meta["table2"].get("rel_l2"), "sets": {}, "excluded": None}
        for name in names:
            row = sets[name]["rows"].get(label)
            entry["sets"][name] = row
            if row and row.get("excluded"):
                entry["excluded"] = row["excluded"]
        ctx = context.get(label) or {}
        entry["context_l2"] = ctx.get("parity_rel_l2")
        entry["context_error"] = ctx.get("error")
        entry["context_reader"] = ctx.get("reader")
        entry["l0"] = {}
        for i, one in enumerate(names):
            for two in names[i + 1:]:
                if mode_of(one) != mode_of(two) or host_of(one) == host_of(two):
                    continue
                a, b = sets[one]["rows"].get(label), sets[two]["rows"].get(label)
                if not a or not b or a["status"] == "refused" or b["status"] == "refused":
                    entry["l0"][f"{one}|{two}"] = {"class": "not comparable",
                                                   "why": "not run on both hosts"}
                    continue
                value = l0_class(read_digests(archive, a), read_digests(archive, b))
                ca, cb = read_currents(archive, a), read_currents(archive, b)
                value["currents_equal"] = None if ca is None or cb is None else ca == cb
                if value["class"] == "differ":
                    value["source_class"] = source_class(meta.get("facts") or {})
                entry["l0"][f"{one}|{two}"] = value
        table.append(entry)
    summary = {}
    for name in names:
        counts = {"example": {"holds": 0, "of": 0}, "test": {"holds": 0, "of": 0},
                  "differs": [], "error": [], "refused": [], "excluded": [],
                  "stored_words_held": 0, "reader_words_held": 0}
        for entry in table:
            row = entry["sets"][name]
            counts[entry["kind"]]["of"] += 1
            if row is None:
                counts["refused"].append((entry["label"], "no record"))
                continue
            if row.get("excluded"):
                counts["excluded"].append((entry["label"], row["excluded"], row["status"]))
                continue
            if row["status"] == "holds":
                counts[entry["kind"]]["holds"] += 1
                counts["stored_words_held"] += int(row["stored"]["words"])
                counts["reader_words_held"] += int(row["reader"]["words"])
            elif row["status"] == "refused":
                counts["refused"].append((entry["label"], short_reason((row.get("problems") or ["?"])[0], 200)))
            else:
                counts[row["status"]].append((entry["label"], (row.get("problems") or ["?"])[0][:300]))
        summary[name] = counts
    l0 = {}
    for entry in table:
        for pair, value in entry["l0"].items():
            bucket = l0.setdefault(pair, {"equal": 0, "differ": [], "not comparable": 0,
                                          "differ_currents_differ": 0, "equal_currents_differ": 0})
            if value["class"] == "equal":
                bucket["equal"] += 1
                bucket["equal_currents_differ"] += value.get("currents_equal") is False
            elif value["class"] == "differ":
                bucket["differ"].append((entry["label"], value["first"], value.get("source_class")))
                bucket["differ_currents_differ"] += value.get("currents_equal") is False
            else:
                bucket["not comparable"] += 1
    return {"schema": SCHEMA, "version": VERSION, "sets": names, "rows": table, "summary": summary,
            "l0": l0, "set_meta": {name: {k: v for k, v in sets[name].items() if k != "rows"}
                                   for name in names}}


def render(table: Dict[str, Any], title: str, provenance: str) -> str:
    names = table["sets"]
    lines = [f"# {title}", "", provenance, ""]
    lines += ["## Summary", ""]
    lines.append("| reference | examples holding | tests holding | differ | error | not run | excluded | stored words held | reader words held |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for name in names:
        c = table["summary"][name]
        lines.append(f"| {name} | {c['example']['holds']} of {c['example']['of']} | "
                     f"{c['test']['holds']} of {c['test']['of']} | {len(c['differs'])} | {len(c['error'])} | "
                     f"{len(c['refused'])} | {len(c['excluded'])} | {c['stored_words_held']:,} | "
                     f"{c['reader_words_held']:,} |")
    lines.append("")
    lines.append("A row holds in a reference when 0 of its W compared stored words and 0 of its V "
                 "`get_array` words differ from that reference MEEP's at every compared step "
                 "through N, and the planted one-ulp flip is caught. In the rows below a cell after "
                 "the first reference's reads `S / R` (differing stored / reader words) when its W "
                 "and V are the first reference's; the flip column covers every reference where "
                 "the row holds; ghost words (mirror ghosts, which MEEP does not store) are "
                 "reported and never judged.")
    lines.append("")
    excluded = {}
    for name in names:
        for label, reason, status in table["summary"][name]["excluded"]:
            excluded.setdefault((label, reason), []).append(f"{name}: {status}")
    if excluded:
        lines += ["## Named exclusions", ""]
        for (label, reason), where in sorted(excluded.items()):
            lines.append(f"- **{label}**: {reason} ({'; '.join(where)}).")
        lines.append("")
    refusals = {}
    for name in names:
        for label, reason in table["summary"][name]["refused"]:
            refusals.setdefault((label, reason), []).append(name)
    if refusals:
        lines += ["## Not run (the case did not build or lift; the reason by name)", ""]
        for (label, reason), where in sorted(refusals.items()):
            lines.append(f"- **{label}** ({', '.join(where)}): {reason}")
        lines.append("")
    for kind in ("differs", "error"):
        found = {}
        for name in names:
            for label, problem in table["summary"][name][kind]:
                found.setdefault(label, []).append(f"{name}: {short_reason(problem, 160)}")
        if found:
            lines += [f"## Rows that {'differ' if kind == 'differs' else 'did not complete'} (not excluded)", ""]
            for label, where in sorted(found.items()):
                lines.append(f"- **{label}**: " + "; ".join(where))
            lines.append("")
    if table["l0"]:
        lines += ["## L0: the two reference MEEPs against each other", ""]
        for pair, bucket in sorted(table["l0"].items()):
            one, two = pair.split("|")
            lines.append(f"- **{one} against {two}**: every MEEP digest equal on {bucket['equal']} rows; "
                         f"the two MEEPs differ on {len(bucket['differ'])}; {bucket['not comparable']} not "
                         "run on both hosts. On "
                         f"{bucket['differ_currents_differ']} of the {len(bucket['differ'])} rows where they "
                         "differ, the source currents the two lifts recorded from their own MEEP differ "
                         f"too; they also differ on {bucket['equal_currents_differ']} of the "
                         f"{bucket['equal']} equal rows, whose stored and read words still agree. By "
                         "source kind:")
            classes = {}
            for label, first, cls in bucket["differ"]:
                classes.setdefault(cls or "unclassed", []).append(f"{label} (from step {first['step']}, "
                                                              f"{first['component']} {first['kind']})")
            for cls, labels in sorted(classes.items()):
                lines.append(f"  - {cls}: " + ", ".join(sorted(labels)))
        lines.append("")
        lines.append("Where the two MEEPs differ, each row is still judged against its own host's "
                     "MEEP, which is the reference that host's lift reads its sources from. The "
                     "source kinds name what each MEEP evaluates on its own host (an eigenmode "
                     "source's MPB solve, a source time function through the host's math library, a "
                     "random draw); that the kind is the cause is inferred, not measured.")
        lines.append("")
    lines += ["## The rows", ""]
    head = (["#", "row", "kind", "cells", "N", "steps"] + [names[0] + " (stored / reader)"]
            + [name for name in names[1:]] + ["ghost words differing, reported (" + names[0] + ")", "flip",
                                              "L2 vs cert (now)", "L2 vs cert (Table 2 of record)"])
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "---|" * len(head))
    for index, entry in enumerate(table["rows"], 1):
        first = entry["sets"][names[0]]
        cells = [str(index), entry["label"] + (" (excluded)" if entry["excluded"] else ""), entry["kind"],
                 _n(entry["cells"]), _n(entry["n_steps"]),
                 str(first.get("compared_steps")) if first else "-", cell_text(first)]
        cells += [cell_text(entry["sets"][name], first if first and first["status"] != "refused" else None)
                  for name in names[1:]]
        ghost = (first or {}).get("ghost") or {}
        cells.append(f"{_n(ghost.get('word_differing'))} of {_n(ghost.get('words'))}" if ghost.get("words") else "-")
        judged = [entry["sets"][name] for name in names
                  if entry["sets"][name] and entry["sets"][name]["status"] in ("holds", "error")]
        flips = [row.get("planted_flip_caught") for row in judged]
        cells.append("caught" if flips and all(flips) else ("-" if not flips else "MISSED"))
        cells.append(f"{entry['context_l2']:.3e}" if entry["context_l2"] is not None
                     else ("-" if not entry["context_error"] else "not measured"))
        cells.append(f"{entry['l2_of_record']:.3e}" if entry["l2_of_record"] is not None else "-")
        lines.append("| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |")
    lines.append("")
    return "\n".join(lines)


def read_context(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    found = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            found[row["label"]] = row
    return found


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--archive", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--context-l2", dest="context_l2")
    parser.add_argument("--set", dest="sets", action="append", default=[])
    parser.add_argument("--out-md", dest="out_md", required=True)
    parser.add_argument("--out-json", dest="out_json")
    parser.add_argument("--title", default="Table 2: MEEP identity on the corpus")
    parser.add_argument("--provenance", default="")
    arguments = parser.parse_args(argv)
    manifest = json.loads(Path(arguments.manifest).read_text(encoding="utf-8"))
    table = build(Path(arguments.archive), manifest, arguments.sets, read_context(arguments.context_l2))
    Path(arguments.out_md).write_text(render(table, arguments.title, arguments.provenance), encoding="utf-8")
    if arguments.out_json:
        Path(arguments.out_json).write_text(json.dumps(table, indent=1, sort_keys=True, default=str) + "\n",
                                            encoding="utf-8")
    for name in table["sets"]:
        c = table["summary"][name]
        print(f"{name}: examples {c['example']['holds']} of {c['example']['of']}, tests "
              f"{c['test']['holds']} of {c['test']['of']}, differ {len(c['differs'])}, error "
              f"{len(c['error'])}, not run {len(c['refused'])}, excluded {len(c['excluded'])}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
