#!/usr/bin/env python
# SPDX-License-Identifier: GPL-2.0-or-later
"""Record development evidence for the bit-parity coverage board -- NOT certification.

    python parity/meep_gpu/record_dev_evidence.py subject --table T --module M [--out FILE]
    python parity/meep_gpu/record_dev_evidence.py record --table T --module M \\
        --backend B --baseline keep|flush --gate NAME --host-class "GPU MODEL" \\
        --subject-before FILE (--words-compared N --words-differing D |
                               --from-artifact JSON --compared-key K --differing-key K) \\
        [--date YYYY-MM-DD] [--recorded-by LABEL] [--note TEXT]
    python parity/meep_gpu/record_dev_evidence.py check

WHO WRITES IT. An integrator, when it re-runs a lane's device legs at integration. A lane
measures on its own base; the integrator measures on the tree that lands, and that is the
only measurement this record holds. Column (d-dev) of ``build_parity_board.py`` reads it;
columns (d) and (e) never do: a certification round re-gates and re-mints the ledgers,
and nothing here stands in for that.

THE SUBJECT. ``subject`` prints what a record binds for one module on the live tree --
the module file's sha256, the sha256 of the device text it emits
(``parity_board_emit.py``: a lifter's compiled text moves with its base even when its own
file does not) and the live array-path oracle. Take it BEFORE the device leg; ``record``
recomputes it after and refuses by name when the two differ ("the evidence subject and
the working tree disagree"), because then the words describe no tree. The board counts a
record only while its subject is the live one, so a later edit drops it with nobody
having to remember to.

WHAT A RECORD HOLDS: table, module, backend, subnormal baseline, words compared, words
differing, the gate or smoke that compared them, the subject, the date and the host
CLASS -- a GPU model or an Apple GPU family, never a host name (one that names this
machine is refused). Optionally the artifact's path and sha256 and the integrator's label.

Every refusal is by name. Exit 0 on success, 3 on a refusal.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import socket
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import build_parity_board as board  # noqa: E402
import parity_board_emit as emit  # noqa: E402

ABOUT = ("DEVELOPMENT EVIDENCE, NOT CERTIFICATION. Integrators record here the device legs "
         "they re-ran at integration (record_dev_evidence.py). Column (d-dev) of the "
         "bit-parity board counts a record only while its subject -- the module's bytes, its "
         "emitted device text and the array path -- is the live one. Columns (d) and (e) "
         "never read this file.")


class EvidenceRefusal(Exception):
    """A named reason a record cannot be written."""


def _row(root: Path, table: str, module: str) -> Dict[str, Any]:
    rows = board.load_rows(root)
    if table not in board.TABLES:
        raise EvidenceRefusal(f"no table {table!r}: one of {board.TABLES}")
    for row in rows["tables"][table]["modules"]:
        if row["module"] == module:
            if row["kind"] not in emit.EMITTED_KINDS:
                raise EvidenceRefusal(f"{table} {module} is a {row['kind']} row: it has no "
                                      f"device form, so no device evidence")
            return row
    raise EvidenceRefusal(f"{table} has no row {module!r}")


def live_subject(root: Path, table: str, module: str,
                 oracle: Optional[Mapping[str, Any]] = None,
                 emitted: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The subject one record binds, on the live tree."""
    row = _row(root, table, module)
    if emitted is None:
        emitted = emit.emitted_witnesses(root, tables=(table,), modules=(module,))[table]
    subject = board.module_subject(root, row, emitted.get(module))
    if oracle is None:
        import contextlib  # noqa: PLC0415

        with contextlib.redirect_stdout(sys.stderr):  # the oracle's progress, off stdout
            oracle = board.live_oracle(root)
    subject["array_path_oracle"] = {"fixture_set": oracle["fixture_set"],
                                    "fingerprint_sha256": oracle["fingerprint_sha256"]}
    subject.update({"table": table, "module": module, "path": row["path"]})
    return subject


def _dotted(document: Any, key: str) -> Any:
    value = document
    for part in key.split("."):
        if isinstance(value, Mapping) and part in value:
            value = value[part]
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            raise EvidenceRefusal(f"the artifact has no {key!r} (missing at {part!r})")
    return value


def _integer(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvidenceRefusal(f"{what} must be an integer, got {value!r}")
    return value


def make_record(root: Path, arguments: Mapping[str, Any], subject_before: Mapping[str, Any],
                live: Mapping[str, Any]) -> Dict[str, Any]:
    """One record, or a named refusal. ``live`` is :func:`live_subject` after the leg."""
    table, module = arguments["table"], arguments["module"]
    bound = {k: subject_before.get(k) for k in ("module_sha256", "emitted_sha256",
                                                 "array_path_oracle")}
    now = {k: live.get(k) for k in bound}
    if (subject_before.get("table"), subject_before.get("module")) != (table, module):
        raise EvidenceRefusal(f"the subject file is {subject_before.get('table')} "
                              f"{subject_before.get('module')}, not {table} {module}")
    if bound != now:
        moved = [k for k in bound if bound[k] != now[k]]
        raise EvidenceRefusal(
            f"the evidence subject and the working tree disagree: {', '.join(moved)} moved "
            f"between the subject taken before the leg and the tree now "
            f"({ {k: str(bound[k])[:16] for k in moved} } -> "
            f"{ {k: str(now[k])[:16] for k in moved} }). The words describe no tree; re-run "
            f"the leg on the tree you record.")
    if arguments["backend"] not in board.TABLE_BACKENDS[table]:
        raise EvidenceRefusal(f"backend {arguments['backend']!r} does not run the {table} "
                              f"table's kernels: one of {board.TABLE_BACKENDS[table]}")
    baseline = arguments["baseline"]
    if baseline not in board.BASELINES[table]:
        raise EvidenceRefusal(f"the {table} table is judged under {board.BASELINES[table]} "
                              f"only (owner ruling R13), not {baseline!r}")
    compared = _integer(arguments["words_compared"], "words compared")
    differing = _integer(arguments["words_differing"], "words differing")
    if compared <= 0:
        raise EvidenceRefusal(f"{compared} words compared: a leg that compared nothing is no "
                              f"evidence")
    if not 0 <= differing <= compared:
        raise EvidenceRefusal(f"{differing} words differing out of {compared}")
    host_class = (arguments.get("host_class") or "").strip()
    if not host_class:
        raise EvidenceRefusal("the host class is required (a GPU model or an Apple GPU "
                              "family)")
    hostname = socket.gethostname().split(".")[0].lower()
    if hostname and hostname in host_class.lower():
        raise EvidenceRefusal(f"the host class {host_class!r} names this machine: record "
                              f"the GPU model or the Apple GPU family, never a host name")
    date = arguments.get("date") or datetime.date.today().isoformat()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        raise EvidenceRefusal(f"the date {date!r} is not YYYY-MM-DD")
    gate = (arguments.get("gate") or "").strip()
    if not gate:
        raise EvidenceRefusal("the gate or smoke name is required")
    record = {"table": table, "module": module, "backend": arguments["backend"],
              "subnormal_baseline": baseline, "words_compared": compared,
              "words_differing": differing, "gate": gate,
              "subject": dict(bound), "date": date, "host_class": host_class}
    for key in ("recorded_by", "note", "artifact", "artifact_sha256"):
        if arguments.get(key):
            record[key] = arguments[key]
    return record


def read_document(root: Path) -> Dict[str, Any]:
    path = root / board.DEV_EVIDENCE_FILE
    if not path.is_file():
        return {"schema": board.DEV_EVIDENCE_SCHEMA, "version": board.DEV_EVIDENCE_VERSION,
                "about": ABOUT, "records": []}
    board.read_dev_evidence(root)  # refuses a malformed file by name
    return json.loads(path.read_text(encoding="utf-8"))


def write_record(root: Path, record: Mapping[str, Any]) -> Path:
    """Add ``record``, replacing one with the same (table, module, backend, baseline,
    gate): a re-run on a later tree supersedes the earlier run of the same leg."""
    document = read_document(root)
    key = tuple(record[k] for k in ("table", "module", "backend", "subnormal_baseline", "gate"))
    kept = [r for r in document["records"]
            if tuple(r[k] for k in ("table", "module", "backend", "subnormal_baseline",
                                    "gate")) != key]
    kept.append(dict(record))
    kept.sort(key=lambda r: (r["table"], r["module"], r["subnormal_baseline"], r["gate"]))
    document.update({"schema": board.DEV_EVIDENCE_SCHEMA,
                     "version": board.DEV_EVIDENCE_VERSION, "about": ABOUT,
                     "records": kept})
    path = root / board.DEV_EVIDENCE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return path


def check(root: Path) -> List[str]:
    """One line per record: whether the board would count it on this tree."""
    records, unavailable = board.read_dev_evidence(root)
    if unavailable:
        return [unavailable]
    import contextlib  # noqa: PLC0415

    with contextlib.redirect_stdout(sys.stderr):
        oracle = board.live_oracle(root)
    tables = sorted({r["table"] for r in records})
    emitted = emit.emitted_witnesses(root, tables=tables,
                                     modules=sorted({r["module"] for r in records}))
    lines = []
    for record in records:
        try:
            live = live_subject(root, record["table"], record["module"], oracle,
                                emitted.get(record["table"], {}))
        except EvidenceRefusal as refusal:
            lines.append(f"REFUSED {record['table']} {record['module']}: {refusal}")
            continue
        bound = record["subject"]
        moved = [k for k in ("module_sha256", "emitted_sha256", "array_path_oracle")
                 if bound.get(k) != live.get(k)]
        state = "current" if not moved else f"stale ({', '.join(moved)} moved)"
        lines.append(f"{state}: {record['table']} {record['module']} {record['backend']} "
                     f"{record['subnormal_baseline']} {record['gate']} "
                     f"{record['words_differing']:,} / {record['words_compared']:,} differ")
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=board.REPO_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    p_subject = sub.add_parser("subject", help="print the live subject of one module")
    p_subject.add_argument("--table", required=True)
    p_subject.add_argument("--module", required=True)
    p_subject.add_argument("--out", type=Path)
    p_record = sub.add_parser("record", help="write one record (refused on a stale subject)")
    for name in ("--table", "--module", "--backend", "--gate", "--host-class"):
        p_record.add_argument(name, required=True)
    p_record.add_argument("--baseline", required=True, choices=("keep", "flush"))
    p_record.add_argument("--subject-before", type=Path, required=True)
    p_record.add_argument("--words-compared", type=int)
    p_record.add_argument("--words-differing", type=int)
    p_record.add_argument("--from-artifact", type=Path)
    p_record.add_argument("--compared-key")
    p_record.add_argument("--differing-key")
    p_record.add_argument("--date")
    p_record.add_argument("--recorded-by")
    p_record.add_argument("--note")
    sub.add_parser("check", help="say which records the board counts on this tree")
    arguments = parser.parse_args(argv)
    root = arguments.root.resolve()
    try:
        if arguments.command == "subject":
            subject = live_subject(root, arguments.table, arguments.module)
            text = json.dumps(subject, indent=1, sort_keys=True) + "\n"
            if arguments.out:
                arguments.out.write_text(text, encoding="utf-8")
            sys.stdout.write(text)
            return 0
        if arguments.command == "record":
            values = vars(arguments).copy()
            if arguments.from_artifact is not None:
                if arguments.words_compared is not None or arguments.words_differing is not None:
                    raise EvidenceRefusal("give the words as numbers or from an artifact, not both")
                if not (arguments.compared_key and arguments.differing_key):
                    raise EvidenceRefusal("--from-artifact needs --compared-key and --differing-key")
                artifact = json.loads(arguments.from_artifact.read_text(encoding="utf-8"))
                values["words_compared"] = _dotted(artifact, arguments.compared_key)
                values["words_differing"] = _dotted(artifact, arguments.differing_key)
                values["artifact_sha256"] = hashlib.sha256(
                    arguments.from_artifact.read_bytes()).hexdigest()
                values["artifact"] = arguments.from_artifact.name
            elif arguments.words_compared is None or arguments.words_differing is None:
                raise EvidenceRefusal("the words compared and differing are required")
            before = json.loads(arguments.subject_before.read_text(encoding="utf-8"))
            live = live_subject(root, arguments.table, arguments.module)
            record = make_record(root, values, before, live)
            path = write_record(root, record)
            print(f"recorded {record['table']} {record['module']} {record['backend']} "
                  f"{record['subnormal_baseline']} {record['gate']}: "
                  f"{record['words_differing']:,} / {record['words_compared']:,} words differ "
                  f"-> {path}", flush=True)
            return 0
        if arguments.command == "check":
            for line in check(root):
                print(line, flush=True)
            return 0
    except (EvidenceRefusal, board.BoardRefusal) as refusal:
        print(f"REFUSED: {refusal}", file=sys.stderr, flush=True)
        return 3
    return 2


if __name__ == "__main__":
    sys.exit(main())
