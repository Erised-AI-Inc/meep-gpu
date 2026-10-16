#!/usr/bin/env python3
"""Re-point the fusion boards' gate-artifact bindings to a round's runs, from a measurement.

Two boards bind their credit to gate artifacts by NAME, and re-hash each named artifact
against the tree on every cut:

* ``build_triton_fusion_matrix.GATE_BOUND`` -- family -> ``results/<root>/<tail>``;
* ``build_fusion_matrix.RELEASE_BINDING`` (and the per-cell ``binding`` rows of its
  ``PRODUCTS``) -- ``("gate", "metal_<gate>_<stamp>")``.

A round re-runs the gates on its frozen tree under its own stamp, and the previous
round's artifacts then record bytes the tree no longer holds, so each board refuses its
first stale binding. Re-pointing them is a RECORDS-TIME step, done once per round on the
host that holds the round's results: this tool measures every binding against the
round's roots and rewrites the table from that measurement, never by hand.

TRITON (``triton``). For every GATE_BOUND entry the candidates are the same artifact
tail under each results root of the same base and the round's stamp
(``<base>_<stamp>`` and ``<base>_<stamp>_<suffix>``; a ``keep_probe/`` tail is also
looked for as ``keep/``, the probe run retired with the expansion licence). Each
candidate is read through the board's own ``gate_bound_check`` (the release-binding
floor's rules: a passed device release under its shape, the credited module recorded)
and re-hashed: it is ADMISSIBLE when the floor raises nothing, 0 recorded files moved,
and it is a keep run (not a ``_flush`` root, and its recorded policy names no flush:
the board binds one policy per process). Exactly one admissible candidate is CHOSEN; two
refuse (the tool never chooses between runs); none leaves the entry UNRESOLVED. An entry
already naming an admissible artifact of the round is UNCHANGED.

* By default (option A1 of the 1.0 records) any unresolved entry refuses the write:
  every credit is re-earned by a run on the frozen bytes, or the board is not cut.
* With ``--withhold-unrun`` (option A2) each unresolved entry keeps its pointer and is
  named in ``CREDIT_WITHHELD_STALE_BINDING`` (which must be empty, or hold only this
  tool's own entries), so the board files its seam-instances buildable_not_built rather
  than crediting bytes no gate ran.

METAL (``metal``). Every ``"gate", "metal_<gate>_<old stamp>"`` reference moves to
``metal_<gate>_<stamp>`` when that run exists, records ``release.released`` true and
re-hashes to the tree with 0 drift; any reference without such a run refuses the write.
``fingerprint`` rows are the ledger's, which the bind rebinds, and are left alone. The
board's own floor re-checks every binding, and each extra cell against the ledger's live
run, when it is cut.

A CHUNKED ROUND (``--stamps S1,S2,...``, the stamps ``chunks.py bind`` selected). The
candidates of an entry are gathered under EVERY stamp given, and the rules above apply
unchanged across them: exactly one admissible candidate is chosen, two refuse, a reference
already naming one of the stamps is unchanged.

ONE DEVICE PER BOARD. A board credits one run per product, so a re-point REPLACES the run it
names; a round on a second device (an H200 beside the RTX A6000, an Apple M3 beside the M1
Max) certifies by the ledger records its bind ADDS, and must not move a board onto its runs.
So a candidate is admissible only on the device the board's binding ran on: ``--device``
(``cc86``; a GPU architecture on Metal), else the device the currently bound run stamps
(an NVIDIA artifact's ``stamped_device``, a Metal run's environment record of its stamp).
A run of another device is refused by name, and so is every run while either device cannot
be read (no ``--device`` and a bound run that records none; a run that records none);
moving a board to another device is said with ``--device``.

Report mode (the default) writes nothing but ``--json``. ``--write`` rewrites the board
module's table, adds one dated note above it saying what moved and why, and re-reads the
written table, restoring the file if it does not hold exactly the measured pointers.
A second ``--write`` on the same results changes nothing.

usage (from parity/meep_gpu on the evidence host):
  python repoint_board_bindings.py triton --stamp S [--withhold-unrun] [--write] [--json F]
  python repoint_board_bindings.py metal  --stamp S [--write] [--json F]
  python repoint_board_bindings.py triton|metal --stamps S1,S2,... [...]    (a chunked round)
  ... [--device cc86|applegpu_g13s]          (the device the board's runs are; default: the
                                              device the bound run stamps)
"""
from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent

#: Where a stamp starts in a results root name: the first ``_YYYY-MM-DD``.
STAMP_START = re.compile(r"_(?=\d{4}-\d\d-\d\d)")
#: The Triton table's anchor, and the withheld table it may fill.
GATE_BOUND_ANCHOR = "GATE_BOUND: Dict[str, str] = {\n"
WITHHELD_ANCHOR = "CREDIT_WITHHELD_STALE_BINDING: Dict[str, str] = {\n"
#: The prefix of the reason this tool writes for a withheld entry, so a rerun can tell
#: its own entries from a person's.
WITHHELD_REASON_LEAD = "RE-POINT PENDING"
#: A Metal gate reference: ``"gate", "metal_<...>"``, on one line or split after the comma.
METAL_GATE_REF = re.compile(r'("gate",\s*)"(metal_[A-Za-z0-9_.-]+)"')
METAL_ANCHOR = "RELEASE_BINDING: Dict[str, Tuple[str, str]] = {\n"


def split_stamp(name: str) -> Tuple[str, Optional[str]]:
    """``<base>_<stamp...>`` -> (base, stamp...), or (name, None) for an undated name."""
    match = STAMP_START.search(name)
    if not match:
        return name, None
    return name[:match.start()], name[match.end():]


def _assigned(text: str, name: str) -> Tuple[Any, ast.AST]:
    """A module-scope assignment's value node (and its literal value when it has one)."""
    for node in ast.parse(text).body:
        target = node.target if isinstance(node, ast.AnnAssign) else (
            node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1
            else None)
        if isinstance(target, ast.Name) and target.id == name:
            return ast.literal_eval(node.value), node.value
    raise SystemExit(f"the board module assigns no {name}")


def _git(repo: Path, *args: str) -> str:
    try:
        done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                              text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return done.stdout.strip() if done.returncode == 0 else ""


def _is_flush_root(root: str, stamp: str) -> bool:
    """A ``_flush`` root of the round (the second policy's twin), by the layout's name."""
    return "flush" in root[len(split_stamp(root)[0]):].replace(stamp, "", 1).split("_")


def triton_device(payload: Any) -> Optional[str]:
    """``cc86`` from an NVIDIA artifact's ``stamped_device.compute_capability``, or None."""
    block = payload.get("stamped_device") if isinstance(payload, dict) else None
    value = block.get("compute_capability") if isinstance(block, dict) else None
    if value is None:
        return None
    text = str(value).strip()
    if text.isdigit() and len(text) >= 2:
        text = f"{text[:-1]}.{text[-1]}"
    major, dot, minor = text.partition(".")
    return f"cc{major}{minor}" if dot and major.isdigit() and minor.isdigit() else None


def metal_device(results: Path, stamp: Optional[str]) -> Optional[str]:
    """The GPU architecture a Metal stamp's environment record names, or None."""
    if not stamp:
        return None
    path = Path(results) / f"metal_environment_{stamp}" / "start.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8")).get("architecture")
    except (OSError, ValueError, AttributeError):
        return None
    return value if isinstance(value, str) and value else None


def _read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _other_device(ran_on: Optional[str], bound: Optional[str]) -> Optional[str]:
    """Why a run may not be credited on a board whose runs are ``bound``'s; None when it
    may. A device that cannot be read, on either side, is a refusal, never a pass."""
    if not bound:
        return ("the device the board's bound run ran on cannot be read (it stamps none, or "
                "its stamp's environment record is absent): name the board's device with "
                "--device")
    if not ran_on:
        return (f"the device it ran on cannot be read, and the board's runs are {bound}'s")
    if ran_on != bound:
        return (f"ran on {ran_on}, and the board's runs are {bound}'s (a board moves to "
                f"another device only with --device {ran_on})")
    return None


def _policy_names_flush(gate: Mapping[str, Any]) -> bool:
    policy = gate.get("subnormal_policy")
    words = (policy.get("policy"), policy.get("resolved")) if isinstance(policy, dict) \
        else (policy,)
    return any(isinstance(word, str) and "flush" in word for word in words)


# ------------------------------------------------------------------------- Triton ----
def measure_triton(board, stamp: str, gate_bound: Optional[Mapping[str, str]] = None,
                   stamps: Optional[Sequence[str]] = None,
                   device: Optional[str] = None) -> Dict[str, Any]:
    """Every GATE_BOUND entry against the round's roots: candidates, the choice, why.

    ``stamps`` (a chunked round's chunk stamps) gathers the candidates under every one;
    ``stamp`` is then only the label the note and the withheld reason carry. ``device``
    (default: the device the bound run stamps) is the only device a candidate may have run
    on."""
    stamps = list(stamps or [stamp])
    gate_bound = dict(board.GATE_BOUND if gate_bound is None else gate_bound)
    results = Path(board.RESULTS)
    roots = sorted(p.name for p in results.iterdir() if p.is_dir()) if results.is_dir() else []
    rows: Dict[str, Any] = {}
    for family, current in sorted(gate_bound.items()):
        parts = current.split("/")
        if len(parts) < 3 or parts[0] != "results":
            raise SystemExit(f"{family}: GATE_BOUND names {current!r}, not results/<root>/<tail>")
        base, _old = split_stamp(parts[1])
        tail = "/".join(parts[2:])
        tails = [tail] + ([tail.replace("keep_probe/", "keep/", 1)]
                          if tail.startswith("keep_probe/") else [])
        mine = [(r, s) for s in stamps for r in roots
                if r == f"{base}_{s}" or r.startswith(f"{base}_{s}_")]
        bound_on = device or triton_device(_read_json(results / current[len("results/"):]))
        candidates = []
        for root, root_stamp in mine:
            for t in tails:
                relative = f"results/{root}/{t}"
                if not (results / root / t).is_file():
                    continue
                check = board.gate_bound_check(family, relative)
                flush = _is_flush_root(root, root_stamp) or (
                    check["gate"] is not None and _policy_names_flush(check["gate"]))
                ran_on = triton_device(check["gate"] if check["gate"] is not None
                                       else _read_json(results / root / t))
                why = (check["refusal"] or (f"{len(check['moved'])} recorded file(s) moved: "
                                            f"{check['moved'][:4]}" if check["moved"] else None)
                       or ("a flush run (the board binds the keep run)" if flush else None)
                       or _other_device(ran_on, bound_on))
                candidates.append({
                    "artifact": relative, "admissible": why is None, "why_not": why,
                    "device": ran_on,
                    "files_checked": len(check["recorded"]), "files_moved": len(check["moved"]),
                    "sha256": hashlib.sha256((results / root / t).read_bytes()).hexdigest()})
        admissible = [c["artifact"] for c in candidates if c["admissible"]]
        if current in admissible:
            state, chosen = "unchanged", current
        elif len(admissible) == 1:
            state, chosen = "repointed", admissible[0]
        elif admissible:
            state, chosen = "ambiguous", None
        else:
            state, chosen = "unresolved", None
        rows[family] = {"current": current, "base": base, "candidates": candidates,
                        "state": state, "chosen": chosen, "device": bound_on}
    counts: Dict[str, int] = {}
    for row in rows.values():
        counts[row["state"]] = counts.get(row["state"], 0) + 1
    return {"board": "triton", "stamp": stamp, "stamps": stamps, "results": str(results),
            "entries": len(rows), "counts": counts, "rows": rows,
            "withheld_before": sorted(getattr(board, "CREDIT_WITHHELD_STALE_BINDING", {}))}


def _literal_pattern(family: str, old: str) -> re.Pattern:
    """The entry's value as written: one literal, or two adjacent literals split at a
    ``/``, after the key and any comment lines between key and value."""
    spellings = ['"' + re.escape(old) + '"'] + [
        '"' + re.escape(old[:k + 1]) + r'"\s*"' + re.escape(old[k + 1:]) + '"'
        for k, ch in enumerate(old) if ch == "/"]
    return re.compile(r'(\n\s+"' + re.escape(family) + r'":(?:\s*#[^\n]*)*\s*)(?:'
                      + "|".join(spellings) + ")")


def _note(lines: Sequence[str]) -> str:
    out = []
    for i, paragraph in enumerate(lines):
        if i:
            out.append("#:\n")
        out += ["#: " + line + "\n" for line in textwrap.wrap(
            paragraph, 86, break_long_words=False, break_on_hyphens=False)]
    return "".join(out)


def _withheld_reason(stamp: str) -> str:
    return (f"{WITHHELD_REASON_LEAD} (round {stamp}): no run of this product's gate on "
            f"the round's bytes was found among the round's results, so its standing "
            f"artifact records bytes the tree no longer holds; withheld until the gate "
            f"runs on these bytes and GATE_BOUND names that run "
            f"(repoint_board_bindings.py).")


def rewrite_triton(text: str, measured: Mapping[str, Any], *, withhold: bool,
                   tree: str = "", today: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    """The builder's text with the measured pointers written in; refuses what it must."""
    stamp, rows = measured["stamp"], measured["rows"]
    ambiguous = sorted(f for f, r in rows.items() if r["state"] == "ambiguous")
    unresolved = sorted(f for f, r in rows.items() if r["state"] == "unresolved")
    if ambiguous:
        raise SystemExit(f"refusing: more than one admissible run for {ambiguous}; the tool "
                         f"never chooses between runs")
    if unresolved and not withhold:
        raise SystemExit(f"refusing: {len(unresolved)} of {len(rows)} entries have no "
                         f"admissible run of the round ({unresolved}); run their gates on "
                         f"these bytes, or pass --withhold-unrun to withhold them by name")
    withheld_now, _node = _assigned(text, "CREDIT_WITHHELD_STALE_BINDING")
    foreign = sorted(f for f, why in withheld_now.items()
                     if not str(why).startswith(WITHHELD_REASON_LEAD))
    if foreign:
        raise SystemExit(f"refusing: CREDIT_WITHHELD_STALE_BINDING holds entries this tool "
                         f"did not write ({foreign}); they are a person's decision")
    want_withheld = sorted(unresolved) if withhold else []
    moves = {f: (r["current"], r["chosen"]) for f, r in rows.items()
             if r["state"] == "repointed"}
    keep_withheld = sorted(withheld_now) == want_withheld
    summary = {"repointed": len(moves), "unchanged": sum(r["state"] == "unchanged"
                                                         for r in rows.values()),
               "withheld": want_withheld, "changed": bool(moves) or not keep_withheld}
    if not summary["changed"]:
        return text, summary
    start = text.index(GATE_BOUND_ANCHOR)
    end = text.index("\n}\n", start) + 3
    table = text[start:end]
    for family, (old, new) in sorted(moves.items()):
        table, n = _literal_pattern(family, old).subn(
            lambda m, new=new: m.group(1) + '"' + new + '"', table)
        if n != 1:
            raise SystemExit(f"{family}: {n} GATE_BOUND matches for {old}")
    text = text[:start] + table + text[end:]
    if not keep_withheld:
        wstart = text.index(WITHHELD_ANCHOR)
        wend = text.index("\n}\n", wstart) + 3
        reason = textwrap.wrap(_withheld_reason(stamp), 80, break_long_words=False,
                               break_on_hyphens=False)
        literal = "".join(f'        "{piece}{" " if i < len(reason) - 1 else ""}"\n'
                          for i, piece in enumerate(reason))
        entries = "".join(f'    "{family}": (\n{literal}    ),\n' for family in want_withheld)
        text = text[:wstart] + WITHHELD_ANCHOR + entries + "}\n" + text[wend:]
    old_roots = sorted({old.split("/")[1] for old, _new in moves.values()})
    new_roots = sorted({new.split("/")[1] for _old, new in moves.values()})
    paragraphs = [
        f"REPOINTED {today or datetime.date.today().isoformat()} TO THE RUNS OF ROUND "
        f"``{stamp}``" + (f" (tree ``{tree[:8]}``)" if tree else "") + f", by "
        f"``repoint_board_bindings.py triton`` from the measured binding table, never by "
        f"hand: {len(moves)} of {len(rows)} entries moved from {len(old_roots)} root(s) to "
        f"{len(new_roots)} root(s) of the round ({', '.join(new_roots)}), each to the one "
        f"run of its gate there that the release-binding floor admits (a passed device "
        f"release of the keep policy, the credited module recorded, 0 recorded files "
        f"moved); {summary['unchanged']} already named such a run.",
    ]
    if want_withheld:
        paragraphs.append(
            f"{len(want_withheld)} entries have no such run and keep their pointers, "
            f"withheld by name in CREDIT_WITHHELD_STALE_BINDING: {', '.join(want_withheld)}.")
    if text.count(GATE_BOUND_ANCHOR) != 1:
        raise SystemExit("GATE_BOUND anchor not unique")
    text = text.replace(GATE_BOUND_ANCHOR, _note(paragraphs) + GATE_BOUND_ANCHOR, 1)
    return text, summary


def verify_triton(text: str, measured: Mapping[str, Any], withheld: Sequence[str]) -> None:
    table, _node = _assigned(text, "GATE_BOUND")
    expect = {f: (r["chosen"] or r["current"]) for f, r in measured["rows"].items()}
    if table != expect:
        wrong = sorted(f for f in set(table) | set(expect) if table.get(f) != expect.get(f))
        raise SystemExit(f"the written GATE_BOUND does not hold the measured pointers: {wrong}")
    now, _node = _assigned(text, "CREDIT_WITHHELD_STALE_BINDING")
    if sorted(now) != sorted(withheld):
        raise SystemExit(f"the written CREDIT_WITHHELD_STALE_BINDING holds {sorted(now)}, "
                         f"not {sorted(withheld)}")


# -------------------------------------------------------------------------- Metal ----
def measure_metal(text: str, results: Path, stamp: str, api: Path,
                  stamps: Optional[Sequence[str]] = None,
                  device: Optional[str] = None) -> Dict[str, Any]:
    """Every ``"gate", "metal_..."`` reference against the round's run of that gate.

    ``stamps`` (a chunked round's chunk stamps): the run is the one stamp holding the gate's
    directory; under two, the reference is unresolved (the tool never chooses between
    runs); ``stamp`` is then only the label the note carries. ``device`` (default: the
    architecture the bound run's stamp records) is the only device the run may have run
    on."""
    stamps = list(stamps or [stamp])
    refs = [m.group(2) for m in METAL_GATE_REF.finditer(text)]
    rows: Dict[str, Any] = {}
    for key in sorted(set(refs)):
        name, artifact_name = (key.rsplit("/", 1) if key.endswith(".json")
                               else (key, "gate.json"))
        base, old = split_stamp(name)
        if old is None:
            rows[key] = {"state": "undated", "chosen": None, "references": refs.count(key),
                         "why_not": "an undated directory name: no stamp to move"}
            continue
        holding = [s for s in stamps if (results / f"{base}_{s}").is_dir()]
        bound_on = device or metal_device(results, old)
        new_name = f"{base}_{holding[0] if len(holding) == 1 else stamps[0]}"
        new_key = f"{new_name}/{artifact_name}" if key.endswith(".json") else new_name
        artifact = results / new_name / artifact_name
        why = None
        moved: List[str] = []
        checked = 0
        if old in stamps:
            state = "unchanged"
        elif len(holding) > 1:
            state, why = "unresolved", (f"{base}_<stamp> exists under {len(holding)} of the "
                                        f"stamps given ({holding}): two runs of one gate")
        elif not artifact.is_file():
            state, why = "unresolved", f"{artifact.relative_to(results.parent)} does not exist"
        else:
            gate = json.loads(artifact.read_text(encoding="utf-8"))
            digests = gate.get("source_sha256") or {}
            checked = len(digests)
            for rel, want in sorted(digests.items()):
                target = Path(rel) if Path(rel).is_absolute() else api / rel
                got = (hashlib.sha256(target.read_bytes()).hexdigest()
                       if target.exists() else "MISSING")
                if got != want:
                    moved.append(rel)
            if not (gate.get("release") or {}).get("released"):
                why = "it does not record release.released"
            elif not digests:
                why = "it records no source digests"
            elif moved:
                why = f"{len(moved)} recorded file(s) moved: {moved[:4]}"
            else:
                why = _other_device(metal_device(results, holding[0]), bound_on)
            state = "unresolved" if why else "repointed"
        rows[key] = {"state": state, "chosen": new_key if state == "repointed" else (
            key if state == "unchanged" else None), "references": refs.count(key),
            "why_not": why, "files_checked": checked, "files_moved": len(moved)}
    counts: Dict[str, int] = {}
    for row in rows.values():
        counts[row["state"]] = counts.get(row["state"], 0) + 1
    return {"board": "metal", "stamp": stamp, "stamps": stamps, "results": str(results),
            "references": len(refs), "directories": len(rows), "counts": counts, "rows": rows}


def rewrite_metal(text: str, measured: Mapping[str, Any], *, tree: str = "",
                  today: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    rows, stamp = measured["rows"], measured["stamp"]
    bad = sorted(k for k, r in rows.items() if r["state"] in ("unresolved", "undated"))
    if bad:
        raise SystemExit(f"refusing: {len(bad)} of {len(rows)} gate directories have no "
                         f"released, zero-drift run of round {stamp}: "
                         + "; ".join(f"{k} ({rows[k]['why_not']})" for k in bad))
    moves = {k: r["chosen"] for k, r in rows.items() if r["state"] == "repointed"}
    summary = {"repointed": len(moves),
               "references_moved": sum(rows[k]["references"] for k in moves),
               "unchanged": sum(r["state"] == "unchanged" for r in rows.values()),
               "changed": bool(moves)}
    if not moves:
        return text, summary
    text, n = METAL_GATE_REF.subn(
        lambda m: m.group(1) + '"' + moves.get(m.group(2), m.group(2)) + '"', text)
    olds = sorted({split_stamp(k)[1] for k in moves})
    paragraph = (
        f"REPOINTED {today or datetime.date.today().isoformat()} from "
        + ", ".join(f"`_{o}`" for o in olds)
        + (f" to `_{stamp}`" if "," not in stamp else
           " to the chunk stamps " + ", ".join(f"`_{s}`" for s in stamp.split(",")))
        + (f" (tree `{tree[:8]}`)" if tree else "")
        + f", {summary['references_moved']} references over {len(moves)} gate directories, "
        f"by `repoint_board_bindings.py metal` from the measured binding table, never by "
        f"hand: each directory of the round exists, records its release true and re-hashes "
        f"to the tree with zero drift.")
    note = "".join("# " + line + "\n" for line in textwrap.wrap(
        paragraph, 86, break_long_words=False, break_on_hyphens=False))
    if text.count(METAL_ANCHOR) != 1:
        raise SystemExit("RELEASE_BINDING anchor not unique")
    text = text.replace(METAL_ANCHOR, note + METAL_ANCHOR, 1)
    return text, summary


def verify_metal(text: str, measured: Mapping[str, Any]) -> None:
    refs = {m.group(2) for m in METAL_GATE_REF.finditer(text)}
    expect = {r["chosen"] for r in measured["rows"].values()}
    if refs != expect:
        raise SystemExit(f"the written references are not the measured ones: "
                         f"{sorted(refs ^ expect)}")


# --------------------------------------------------------------------------- main ----
def _print_triton(measured: Mapping[str, Any]) -> None:
    for family, row in sorted(measured["rows"].items()):
        detail = row["chosen"] if row["chosen"] else "; ".join(
            f"{c['artifact']}: {c['why_not']}" for c in row["candidates"]) or "no candidate"
        print(f"  {row['state']:10s} {family:44s} {detail}", flush=True)
    print(f"TRITON {measured['stamp']}: {measured['entries']} GATE_BOUND entries; "
          + ", ".join(f"{k} {v}" for k, v in sorted(measured["counts"].items())), flush=True)


def _print_metal(measured: Mapping[str, Any]) -> None:
    for key, row in sorted(measured["rows"].items()):
        print(f"  {row['state']:10s} {key:60s} x{row['references']} "
              f"{row['chosen'] or row['why_not']}", flush=True)
    print(f"METAL {measured['stamp']}: {measured['references']} references over "
          f"{measured['directories']} gate directories; "
          + ", ".join(f"{k} {v}" for k, v in sorted(measured["counts"].items())), flush=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("board", choices=("triton", "metal"))
    ap.add_argument("--stamp", default=None, help="the round's stamp, e.g. 2026-10-12_10")
    ap.add_argument("--stamps", default="", metavar="S1,S2,...",
                    help="a chunked round: every chunk stamp (chunks.py bind prints them)")
    ap.add_argument("--device", default=None,
                    help="the device the board's runs are (cc86; applegpu_g13s); default: "
                         "the device the bound run stamps. A run of another device is "
                         "refused")
    ap.add_argument("--withhold-unrun", action="store_true",
                    help="triton only: withhold the entries with no run of the round by "
                         "name instead of refusing (option A2)")
    ap.add_argument("--write", action="store_true", help="rewrite the board's table")
    ap.add_argument("--json", default=None, help="write the measured table here")
    args = ap.parse_args(argv)
    if args.withhold_unrun and args.board != "triton":
        ap.error("--withhold-unrun belongs to the triton board")
    if bool(args.stamp) == bool(args.stamps):
        ap.error("give --stamp (one round) or --stamps (a chunked round), not both")
    stamps = [args.stamp] if args.stamp else \
        [name.strip() for name in args.stamps.split(",") if name.strip()]
    if len(set(stamps)) != len(stamps):
        ap.error(f"--stamps names a stamp twice: {stamps}")
    args.stamp = ",".join(stamps)
    sys.path[:0] = [str(HERE.parents[1]), str(HERE)]
    tree = _git(HERE.parents[1], "rev-parse", "HEAD^{tree}")
    if args.board == "triton":
        import build_triton_fusion_matrix as board  # noqa: PLC0415
        builder = Path(board.__file__).resolve()
        measured = measure_triton(board, args.stamp, stamps=stamps, device=args.device)
        _print_triton(measured)
    else:
        builder = HERE / "build_fusion_matrix.py"
        measured = measure_metal(builder.read_text(encoding="utf-8"), HERE / "results",
                                 args.stamp, HERE.parents[1], stamps=stamps,
                                 device=args.device)
        _print_metal(measured)
    measured["builder"] = str(builder)
    measured["builder_sha256_before"] = hashlib.sha256(builder.read_bytes()).hexdigest()
    before = builder.read_text(encoding="utf-8")
    try:
        if args.board == "triton":
            text, summary = rewrite_triton(before, measured, withhold=args.withhold_unrun,
                                           tree=tree)
        else:
            text, summary = rewrite_metal(before, measured, tree=tree)
    except SystemExit as refused:
        measured["refusal"] = str(refused)
        print(f"{'REFUSED' if args.write else 'A --write would REFUSE'} (nothing written): "
              f"{refused}", flush=True)
        if args.json:
            Path(args.json).write_text(json.dumps(measured, indent=1, sort_keys=True) + "\n",
                                       encoding="utf-8")
        return 1
    measured["summary"] = summary
    if args.write and summary["changed"]:
        builder.write_text(text, encoding="utf-8")
        try:
            after = builder.read_text(encoding="utf-8")
            if args.board == "triton":
                verify_triton(after, measured, summary["withheld"])
            else:
                verify_metal(after, measured)
        except SystemExit:
            builder.write_text(before, encoding="utf-8")
            raise
        print(f"WROTE {builder.name}: {summary}", flush=True)
    elif args.write:
        print(f"UNCHANGED {builder.name}: every binding already names the round's run",
              flush=True)
    else:
        print(f"REPORT ONLY (nothing written): a --write would {summary}", flush=True)
    measured["builder_sha256_after"] = hashlib.sha256(builder.read_bytes()).hexdigest()
    if args.json:
        Path(args.json).write_text(json.dumps(measured, indent=1, sort_keys=True) + "\n",
                                   encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
