#!/usr/bin/env python
# SPDX-License-Identifier: GPL-2.0-or-later
"""Mint Tier B of MEEP identity: the corpus's Tier C output, judged and archived by tool.

    python parity/meep_gpu/record_meep_identity_archive.py --archive DIR --set NAME
        --mode keep|flush --source DIR [--source DIR ...] [--exclude LABEL=REASON ...]
        [--replay] [--replay-limit N] [--link]

WHAT TIER B IS. Tier A (``record_meep_identity.py``) mints a few small cases into the
repository, inside a 5 MB budget. Tier B is the same record for the whole corpus, Table 2's
194 rows, kept in the evidence archive (``parity/meep_gpu/results/``, outside the
repository and the wheel): per row and per reference (host and subnormal mode), MEEP's
digests (``digests.json``: one SHA-256 per compared step and component, stored and
``get_array`` words), the plane digests up to step 20 (``planes.npy``), the Tier C result
that measured them (``result.json``) and, where the capsule schema carries the row, its
capsule and recorded currents, so the row replays with no MEEP. MEEP's words themselves are
not archived: the Tier C runner never writes them, and one corpus checkpoint set is about
277 MB raw.

WHAT THIS TOOL DOES, for one set (one reference: a host and a mode):

1. It reads every Tier C output directory under ``--source``
   (``meep_identity_tier_c.py compare --out``; a parent of many such directories is read
   too, as the corpus driver lays them out): each row's ``<mode>/<label>/result.json``, or
   its ``<mode>/<label>.error.json`` when the case was refused before it ran.
2. It JUDGES each row by the Tier C half of ``record_meep_identity.judge_case``, the
   criteria Tier A is minted by: 0 value- and 0 word-differing compared stored words, 0
   differing reader words, every MEEP read exact and equal to MEEP's point read, no
   compared cell without a MEEP word, the FPU in the mode at every compared step, the
   planted flip caught by the word compare, the digest and its plane. A row is
   ``holds``, ``differs`` (compared words differ), ``refused`` (the case did not run: its
   reason by name) or ``error`` (it ran and a claim other than the words failed). A row
   named by ``--exclude`` keeps its measured status and carries the reason; it is
   counted apart.
3. It copies each row's files into ``<archive>/<set>/<label>/``, in the runner's own
   layout (so ``meep_identity_tier_c.py replay --source`` reads the row), and records the
   sha256 and size of every file. ``--link`` hard-links them instead where the source is
   on the same file system (the Tier C output is never written again).
4. With ``--replay``, every holding row with a capsule is replayed from the archive with
   ``meep_identity_tier_c.py replay`` (a process that cannot import MEEP), in order of
   size, ``--replay-limit`` rows at most: every digest must be reproduced and the planted
   flip caught and placed. The counts are written to the row.
5. It writes ``sets[<set>]`` of ``<archive>/meep_identity_tier_b.json`` (other sets are
   kept as they are) and reads it back, checking every file's sha256.

Nothing here judges a row against another reference: L0, MEEP against MEEP across hosts,
is read off two sets' digests by ``build_meep_identity_table.py``.

Progress: one flushed line per row read and per row replayed. Exit status: 0 when the set
was written, 3 when it could not run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

TOOL = "parity/meep_gpu/record_meep_identity_archive.py"
RUNNER = os.path.join(HERE, "meep_identity_tier_c.py")
ARCHIVE_FILE = "meep_identity_tier_b.json"
SCHEMA = "meep_gpu.meep_identity_tier_b"
VERSION = 1
#: The row files archived, by role; the capsule and currents exist only where the capsule
#: schema carried the row.
ROW_FILES = {"result": "result.json", "digests": "digests.json", "planes": "planes.npy",
             "capsule": "capsule.capsule", "currents": "currents.json"}
STATUSES = ("holds", "differs", "error", "refused")


def say(text: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Judging (no MEEP; unit-tested)
# --------------------------------------------------------------------------

def _replay_that_holds(mode: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """A replay record every replay check of ``judge_case`` accepts: only Tier C is judged."""
    digests = len(result.get("steps") or []) * 12
    return {"meep_imported": False, "digests": digests, "stored_equal": digests,
            "reader_equal": digests, "planted_flip": {"caught": True, "placed": True},
            "fpu": {"measured": {"mode": mode}}}


def judge_row(mode: str, result: Dict[str, Any]) -> Tuple[str, List[str]]:
    """``(status, problems)`` of one Tier C result, by ``judge_case``'s Tier C criteria.

    The digest-smoke check is not applied: a corpus row is not one of the smoke's four
    cases, and a run through a cases file records ``0 of 0``.
    """
    from parity.meep_gpu.record_meep_identity import judge_case  # noqa: PLC0415

    if result.get("refused"):
        return "refused", [str(result["refused"])]
    problems = judge_case(mode, result, _replay_that_holds(mode, result))
    if not problems:
        return "holds", []
    stored, reader = result.get("stored") or {}, result.get("reader") or {}
    if stored.get("word_differing") or stored.get("value_differing") or reader.get("word_differing"):
        return "differs", problems
    return "error", problems


def parse_exclusions(items: List[str]) -> Dict[str, str]:
    """``LABEL=REASON`` arguments as a dict; a missing reason is refused."""
    found: Dict[str, str] = {}
    for item in items:
        label, _, reason = item.partition("=")
        if not label.strip() or not reason.strip():
            raise ValueError(f"--exclude {item!r}: give LABEL=REASON")
        found[label.strip()] = reason.strip()
    return found


def count_statuses(rows: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Statuses over the rows, the excluded rows counted apart, with the words held."""
    counts: Dict[str, Any] = {status: 0 for status in STATUSES}
    counts["excluded"] = 0
    counts["rows"] = len(rows)
    counts["stored_words_held"] = counts["reader_words_held"] = 0
    for row in rows.values():
        if row.get("excluded"):
            counts["excluded"] += 1
            continue
        counts[row["status"]] += 1
        if row["status"] == "holds":
            counts["stored_words_held"] += int((row.get("stored") or {}).get("words") or 0)
            counts["reader_words_held"] += int((row.get("reader") or {}).get("words") or 0)
    return counts


# --------------------------------------------------------------------------
# Reading the Tier C output
# --------------------------------------------------------------------------

def find_rows(sources: List[Path], mode: str) -> Iterator[Tuple[str, Path, Path]]:
    """``(label, the runner's --out directory, the row directory or error file)``."""
    seen = set()
    for source in sources:
        outs = [source] if (source / mode).is_dir() else sorted(
            path.parent for path in source.glob(f"*/{mode}") if path.is_dir())
        for out in outs:
            for result in sorted((out / mode).glob("*/result.json")):
                label = result.parent.name
                if label not in seen:
                    seen.add(label)
                    yield label, out, result.parent
            for error in sorted((out / mode).glob("*.error.json")):
                label = error.name[: -len(".error.json")]
                if label not in seen:
                    seen.add(label)
                    yield label, out, error


def read_result(path: Path) -> Dict[str, Any]:
    if path.is_dir():
        return json.loads((path / ROW_FILES["result"]).read_text(encoding="utf-8"))
    error = json.loads(path.read_text(encoding="utf-8"))
    return {"refused": error.get("error") or "refused with no reason", "case": error.get("case")}


def row_record(mode: str, label: str, result: Dict[str, Any], excluded: Optional[str]) -> Dict[str, Any]:
    status, problems = judge_row(mode, result)
    record: Dict[str, Any] = {
        "status": status, "problems": problems[:8],
        "n_steps": result.get("n_steps"), "compared_steps": len(result.get("steps") or []),
        "grid_shape": result.get("grid_shape"), "cylindrical": result.get("cylindrical"),
        "stored": result.get("stored"), "reader": result.get("reader"), "ghost": result.get("ghost"),
        "reads": result.get("reads"), "first_difference": result.get("first_difference"),
        "planted_flip_caught": (result.get("planted_flip") or {}).get("caught"),
        "subnormals": (result.get("subnormals") or {}).get("mode"),
        "omp_num_threads": result.get("omp_num_threads"), "meep_point_read_loss":
            (result.get("meep_point_read_loss") or {}).get("word_differing"),
        "capsule_refused": result.get("capsule_refused"),
        "chunks": (result.get("chunk_division") or {}).get("chunks"),
        "case_facts": result.get("case_facts"), "seconds": result.get("seconds"),
    }
    if excluded:
        record["excluded"] = excluded
    return record


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------

def place(source: Path, target: Path, link: bool) -> None:
    """``target`` as a copy of ``source``, or a hard link to it with ``link``."""
    if link:
        try:
            os.link(source, target)
            return
        except OSError:
            pass
    shutil.copyfile(source, target)


def replay_row(target: Path, mode: str, label: str) -> Dict[str, Any]:
    """The runner's Tier A replay of one archived row (MEEP blocked in its child)."""
    rows = target / "tier_b_replay.jsonl"
    if rows.exists():
        rows.unlink()
    done = subprocess.run([sys.executable, "-u", RUNNER, "replay", "--source", str(target),
                           "--mode", mode, "--cases", label, "--rows", str(rows)],
                          capture_output=True, text=True, check=False)
    found = {}
    if rows.exists():
        lines = [json.loads(line) for line in rows.read_text(encoding="utf-8").splitlines() if line.strip()]
        found = lines[-1] if lines else {}
        rows.unlink()
    keys = ("holds", "stored_equal", "reader_equal", "digests", "first_difference", "planted_flip",
            "meep_imported", "seconds", "error")
    out = {key: found.get(key) for key in keys if key in found}
    out["rc"] = done.returncode
    if not found:
        out["error"] = (done.stdout or "")[-600:] or "no replay row"
    return out


def write_set(archive: Path, name: str, leaf: Dict[str, Any]) -> Dict[str, Any]:
    path = archive / ARCHIVE_FILE
    if path.exists():
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("schema") != SCHEMA or document.get("version") != VERSION:
            raise ValueError(f"{path} is not a {SCHEMA} v{VERSION} record")
    else:
        document = {"schema": SCHEMA, "version": VERSION, "sets": {}}
    document["about"] = ("MEEP identity, Tier B: per reference (host and subnormal mode), every "
                         "corpus row's Tier C verdict, MEEP's digests and plane digests, and the "
                         "capsule and currents where the schema carries the row. Written only by "
                         f"{TOOL}.")
    document["sets"][name] = leaf
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(document, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return document


def check_set(archive: Path, name: str) -> int:
    """Re-read the record and every file's sha256; the number of files checked."""
    document = json.loads((archive / ARCHIVE_FILE).read_text(encoding="utf-8"))
    checked = 0
    for label, row in document["sets"][name]["rows"].items():
        for role, spec in (row.get("files") or {}).items():
            path = archive / spec["file"]
            if sha256_file(path) != spec["sha256"] or path.stat().st_size != spec["bytes"]:
                raise ValueError(f"{name}/{label}: {spec['file']} does not read back")
            checked += 1
    return checked


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--archive", required=True, help="the Tier B directory (in the evidence archive)")
    parser.add_argument("--set", dest="name", required=True,
                        help="the reference's name, e.g. mac_keep or linux_flush")
    parser.add_argument("--mode", choices=("keep", "flush"), required=True)
    parser.add_argument("--source", action="append", required=True,
                        help="a Tier C output directory, or a parent of many")
    parser.add_argument("--exclude", action="append", default=[], help="LABEL=REASON")
    parser.add_argument("--replay", action="store_true", help="replay the holding capsule rows")
    parser.add_argument("--replay-limit", dest="replay_limit", type=int, default=None)
    parser.add_argument("--host", default=None, help="a name for the reference host (recorded)")
    parser.add_argument("--link", action="store_true", help="hard-link the row files where possible")
    arguments = parser.parse_args(argv)
    try:
        exclusions = parse_exclusions(arguments.exclude)
    except ValueError as exc:
        print(f"ARCHIVE NOT RUN: {exc}", flush=True)
        return 3
    archive = Path(arguments.archive).resolve()
    target_root = archive / arguments.name
    staging = archive / f"{arguments.name}.writing"
    if staging.exists():
        print(f"ARCHIVE NOT RUN: {staging} exists (an interrupted run); remove it first", flush=True)
        return 3
    started = time.time()
    tool_sha = sha256_file(Path(__file__))
    rows: Dict[str, Dict[str, Any]] = {}
    images: Dict[str, Any] = {}
    staging.mkdir(parents=True)
    found = list(find_rows([Path(s).resolve() for s in arguments.source], arguments.mode))
    say(f"1/4 {len(found)} rows under {len(arguments.source)} source(s), mode {arguments.mode}")
    for index, (label, out, where) in enumerate(found, 1):
        result = read_result(where)
        record = row_record(arguments.mode, label, result, exclusions.get(label))
        files = {}
        destination = staging / label
        if where.is_dir():
            (destination / arguments.mode / label).mkdir(parents=True)
            for role, filename in ROW_FILES.items():
                source_file = where / filename
                if source_file.exists():
                    copy = destination / arguments.mode / label / filename
                    place(source_file, copy, arguments.link)
                    files[role] = {"file": f"{arguments.name}/{label}/{arguments.mode}/{label}/{filename}",
                                   "sha256": sha256_file(copy), "bytes": copy.stat().st_size}
            ledger = out / "meep_identity.json"
            if ledger.exists():
                place(ledger, destination / "meep_identity.json", arguments.link)
                files["ledger"] = {"file": f"{arguments.name}/{label}/meep_identity.json",
                                   "sha256": sha256_file(destination / "meep_identity.json"),
                                   "bytes": (destination / "meep_identity.json").stat().st_size}
            for key, value in ((result.get("meep") or {}).get("images") or {}).items():
                images.setdefault(key, set()).add(value)
        else:
            destination.mkdir(parents=True)
            copy = destination / where.name
            place(where, copy, arguments.link)
            files["refusal"] = {"file": f"{arguments.name}/{label}/{where.name}",
                                "sha256": sha256_file(copy), "bytes": copy.stat().st_size}
        record["files"] = files
        rows[label] = record
        stored = record.get("stored") or {}
        say(f"    {index}/{len(found)} {label}: {record['status']}"
            + (f" (excluded: {record['excluded']})" if record.get("excluded") else "")
            + (f" stored {stored.get('word_differing')}/{stored.get('words')}" if stored else ""))
    if target_root.exists():
        shutil.rmtree(target_root)
    staging.rename(target_root)
    say(f"2/4 {len(rows)} rows copied into {target_root}")
    replayed = {"rows": 0, "hold": 0, "candidates": 0}
    if arguments.replay:
        candidates = sorted(
            (label for label, row in rows.items()
             if row["status"] == "holds" and "capsule" in row["files"] and not row.get("excluded")),
            key=lambda label: sum(spec["bytes"] for spec in rows[label]["files"].values()))
        replayed["candidates"] = len(candidates)
        if arguments.replay_limit is not None:
            candidates = candidates[: arguments.replay_limit]
        for index, label in enumerate(candidates, 1):
            outcome = replay_row(target_root / label, arguments.mode, label)
            rows[label]["tier_b_replay"] = outcome
            replayed["rows"] += 1
            replayed["hold"] += bool(outcome.get("holds"))
            say(f"    replay {index}/{len(candidates)} {label}: "
                f"{'HOLDS' if outcome.get('holds') else 'DOES NOT HOLD'} stored "
                f"{outcome.get('stored_equal')}/{outcome.get('digests')} reader "
                f"{outcome.get('reader_equal')}/{outcome.get('digests')} ({outcome.get('seconds')} s)"
                + (f" ERROR {str(outcome.get('error'))[:200]}" if outcome.get("error") else ""))
    say(f"3/4 replay: {replayed['hold']} of {replayed['rows']} replayed rows hold "
        f"({replayed['candidates']} holding rows carry a capsule)")
    leaf = {"mode": arguments.mode, "host": arguments.host,
            "meep_images": {key: sorted(values) for key, values in sorted(images.items())},
            "counts": count_statuses(rows), "replay": replayed, "rows": rows,
            "minted": {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tool": TOOL,
                       "tool_sha256": tool_sha, "sources": [str(s) for s in arguments.source],
                       "seconds": round(time.time() - started, 1)}}
    write_set(archive, arguments.name, leaf)
    checked = check_set(archive, arguments.name)
    counts = leaf["counts"]
    say(f"4/4 WROTE {archive / ARCHIVE_FILE} set {arguments.name}: {counts['holds']} hold, "
        f"{counts['differs']} differ, {counts['error']} error, {counts['refused']} refused, "
        f"{counts['excluded']} excluded, of {counts['rows']} rows; {checked} files read back")
    return 0


if __name__ == "__main__":
    sys.exit(main())
