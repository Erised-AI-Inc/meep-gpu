#!/usr/bin/env python
# SPDX-License-Identifier: GPL-2.0-or-later
"""Mint the Tier A fixtures of MEEP identity: the only writer of ``meep_identity.json``.

    python parity/meep_gpu/record_meep_identity.py --build-record PATH --work DIR
        [--mode keep|flush] [--cases a,b,c,c_prime] [--cases-file PATH] [--fixtures DIR]
        [--rows FILE]

Run it with the interpreter of an identity-reference MEEP (``tools/check_install.py
--require-identity-reference``). It writes nothing unless the run reproduces every claim
the ledger will make:

1. It hashes its inputs (this tool, the Tier C runner, ``meep_gpu/meep_identity.py``,
   ``meep_gpu/capsule.py``, the array path's modules ``array_path_oracle.
   PROVENANCE_MODULES`` and the digest smoke with its record) before the run and after
   it; any change refuses.
2. It judges the MEEP: ``check_install.identity_check`` in the mode, against the build
   record given; any reason refuses (census, record, FPU mode, OpenMP for flush, libm).
   In keep mode, when a minted case is not one of the digest smoke's four (whose Tier C
   run checks the smoke record in process), ``meep_identity_smoke.py`` must also
   reproduce its record: the MEEP is the identity reference that minted it.
3. Tier C (``meep_identity_tier_c.py compare``) runs into ``--work``, a new directory,
   each case in its own process: lift, pair-order stepping, every stored and reader word
   compared with MEEP's, a planted 1-ulp flip, and the smoke record's digests.
4. Tier A (``meep_identity_tier_c.py replay``) replays ``--work`` in processes that
   cannot import MEEP: every digest must be reproduced from the capsule, and its planted
   flip caught and placed in its plane.
5. Every case is judged (:func:`judge_case`): 0 value-differing and 0 word-differing
   compared words, 0 differing reader words, every read an exact float32, the FPU in the
   mode at every compared step, the planted flip caught by the word compare, the digest
   and its plane, the smoke record reproduced, and the replay reproducing every digest
   with MEEP never imported. Ghost words are reported, not judged.
6. The fixture bytes are measured against ``meep_identity.FIXTURE_BUDGET_BYTES``.
7. It copies the case files into ``<fixtures>/<mode>/`` (replacing that directory, and
   only it) and writes the ledger's ``sets[<mode>]`` leaf; another mode's leaf is kept as
   it is. The copies are checked by sha256 and the ledger is read back with
   ``meep_identity.read_ledger``.

Progress: the runner's lines are passed on as they arrive; one flushed line per stage.
Exit status: 0 when minted; 1 when a claim did not hold (nothing written); 3 when it
could not run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

TOOL = "parity/meep_gpu/record_meep_identity.py"
RUNNER = os.path.join(HERE, "meep_identity_tier_c.py")
INPUTS = ("parity/meep_gpu/record_meep_identity.py", "parity/meep_gpu/meep_identity_tier_c.py",
          "parity/meep_gpu/meep_identity_smoke.py", "parity/meep_gpu/meep_identity_smoke.json",
          "meep_gpu/meep_identity.py", "meep_gpu/capsule.py", "meep_gpu/array_path_oracle.py")


def say(text: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def input_digests() -> Dict[str, str]:
    """sha256 of every input file, the array path's modules included."""
    from meep_gpu.array_path_oracle import PROVENANCE_MODULES  # noqa: PLC0415

    names = sorted(set(INPUTS) | set(PROVENANCE_MODULES))
    return {name: sha256_file(os.path.join(ROOT, name)) for name in names}


def load_tool(name: str) -> Any:
    path = os.path.join(ROOT, "tools", name + ".py")
    spec = importlib.util.spec_from_file_location("meep_gpu_tool_" + name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def portable(path: str) -> str:
    """A path a committed record may carry: from its last ``results/`` component, or its name."""
    parts = os.path.abspath(path).split(os.sep)
    if "results" in parts:
        last = len(parts) - 1 - parts[::-1].index("results")
        return "/".join(parts[last:])
    return parts[-1]


# --------------------------------------------------------------------------
# Judging (no MEEP; unit-tested)
# --------------------------------------------------------------------------

def judge_case(mode: str, tier_c: Dict[str, Any], replay: Dict[str, Any]) -> List[str]:
    """Why one case may not be minted from these results. Empty means it may."""
    problems: List[str] = []
    if tier_c.get("error"):
        return [f"Tier C did not run: {tier_c['error']}"]
    if replay.get("error"):
        return [f"the Tier A replay did not run: {replay['error']}"]
    stored, reader = tier_c.get("stored") or {}, tier_c.get("reader") or {}
    steps = tier_c.get("steps") or []
    if not stored.get("words"):
        problems.append("Tier C compared no stored word")
    if stored.get("value_differing", 1) or stored.get("word_differing", 1):
        problems.append(f"Tier C: {stored.get('value_differing')} value-differing and "
                        f"{stored.get('word_differing')} word-differing of {stored.get('words')} "
                        "compared stored words")
    if not reader.get("words") or reader.get("word_differing", 1):
        problems.append(f"Tier C: {reader.get('word_differing')} of {reader.get('words')} reader "
                        "words differ")
    reads = tier_c.get("reads") or {}
    if reads.get("inexact", 1) or reads.get("imaginary", 1):
        problems.append(f"Tier C: {reads.get('inexact')} MEEP reads were not an exact float32 and "
                        f"{reads.get('imaginary')} had an imaginary part")
    if reads.get("point_read_differing", 1):
        problems.append(f"Tier C: {reads.get('point_read_differing')} exact MEEP words are not "
                        "what MEEP's point read returns for them")
    if reads.get("not_stored_compared", 1):
        problems.append(f"Tier C: {reads.get('not_stored_compared')} compared points hold no "
                        "MEEP stored word")
    if reads.get("held_copies_differing", 1):
        problems.append(f"Tier C: {reads.get('held_copies_differing')} compared points are held "
                        "with different words by two MEEP chunks")
    if reads.get("held_aliased_unexplained", 1) or reads.get("held_aliased_complex", 1):
        problems.append(f"Tier C: {reads.get('held_aliased_unexplained')} compared B points of an "
                        "H==B chunk hold neither B's nor H's point read and "
                        f"{reads.get('held_aliased_complex')} were read through the point read "
                        "on complex storage")
    if reads.get("held_unconnected_unexplained", 1):
        problems.append(f"Tier C: {reads.get('held_unconnected_unexplained')} compared B points of a "
                        "chunk that took an H array of its own hold neither their image's word nor "
                        "the word they held before")
    if (tier_c.get("subnormals") or {}).get("mode") != mode:
        problems.append(f"Tier C ran with the FPU {(tier_c.get('subnormals') or {}).get('mode')}, "
                        f"not {mode}")
    if not steps or tier_c.get("fpu_checks") != len(steps):
        problems.append(f"Tier C measured the FPU at {tier_c.get('fpu_checks')} of {len(steps)} "
                        "compared steps")
    flip = tier_c.get("planted_flip") or {}
    if not (flip.get("caught") and flip.get("words_differing") == 1 and flip.get("digest_caught")
            and flip.get("planes_differing") == [flip.get("plane")]):
        problems.append(f"Tier C's planted flip was not caught by the word compare, the digest "
                        f"and its plane: {flip}")
    smoke = tier_c.get("smoke_record") or {}
    if smoke.get("equal") != smoke.get("of") or smoke.get("lattice_equal") is False:
        problems.append(f"MEEP's words reproduce {smoke.get('equal')} of {smoke.get('of')} digests "
                        f"of the smoke record (lattice equal: {smoke.get('lattice_equal')})")
    if replay.get("meep_imported") is not False:
        problems.append("MEEP was imported in the Tier A replay process")
    digests = replay.get("digests")
    if not digests or replay.get("stored_equal") != digests or replay.get("reader_equal") != digests:
        problems.append(f"the Tier A replay reproduced {replay.get('stored_equal')} stored and "
                        f"{replay.get('reader_equal')} reader digests of {digests}; first "
                        f"difference {replay.get('first_difference')}")
    if digests and digests != len(steps) * 12:
        problems.append(f"the replay judged {digests} digests; the schedule has {len(steps) * 12}")
    replay_flip = replay.get("planted_flip") or {}
    if not (replay_flip.get("caught") and replay_flip.get("placed")):
        problems.append(f"the Tier A replay's planted flip was not caught and placed: {replay_flip}")
    if ((replay.get("fpu") or {}).get("measured") or {}).get("mode") != mode:
        problems.append(f"the Tier A replay ran with the FPU "
                        f"{((replay.get('fpu') or {}).get('measured') or {}).get('mode')}, not {mode}")
    return problems


def ledger_leaf(mode: str, cases: Dict[str, Dict[str, Any]], oracle: Dict[str, Any],
                provenance_code: Dict[str, str], reference: Dict[str, Any],
                minted: Dict[str, Any]) -> Dict[str, Any]:
    """The ``sets[mode]`` leaf of the ledger."""
    return {"cases": cases, "array_path_oracle": oracle,
            "provenance_code_sha256": provenance_code, "reference": reference, "minted": minted,
            "mode": mode}


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------

def run_runner(arguments: List[str]) -> int:
    """The Tier C runner as a child, its lines passed on as they arrive."""
    command = [sys.executable, "-u", RUNNER, *arguments]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print("    " + line.rstrip("\n"), flush=True)
    return process.wait()


def read_rows(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def reference_identity(check: Any, mode: str, build_record: str) -> Dict[str, Any]:
    """``check_install``'s identity facts and verdict for this interpreter's MEEP."""
    namespace = argparse.Namespace(require_identity_reference=mode, identity_build_record=build_record,
                                   gpu_id=0, verbose=False)
    _lines, failure = check.identity_check(namespace)
    payload, _done = check.run_child("identity", namespace, ("--identity-mode", mode))
    payload = payload or {}
    with open(build_record, encoding="utf-8") as handle:
        record = json.load(handle)
    census = record.get("fma_census") or {}
    return {
        "verdict": None if failure is None else failure.reason,
        "meep": {key: (payload.get("meep") or {}).get(key) for key in ("version", "single_precision", "mpi")},
        "images": {os.path.basename(path): digest for path, digest in sorted((payload.get("images") or {}).items())},
        "build_record": {"file": portable(build_record), "sha256": sha256_file(build_record),
                         "fp_contract": record.get("fp_contract"),
                         "census": (census.get("verdict") or {}).get("text")},
        "libm": payload.get("libm"),
        "subnormals": payload.get("subnormals"),
        "omp_num_threads": payload.get("omp_num_threads"),
        "system": payload.get("system"), "python": payload.get("python"), "numpy": payload.get("numpy"),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--build-record", required=True,
                        help="the identity reference's build record (timing_records.py meep-build-record)")
    parser.add_argument("--work", required=True, help="a new directory for the Tier C output")
    parser.add_argument("--mode", choices=("keep", "flush"), default="keep")
    parser.add_argument("--cases", help="a comma-separated subset of the cases (default: all of them)")
    parser.add_argument("--cases-file", dest="cases_file",
                        help="a module with build(mp, label) and CASES (default: the four probe cases)")
    parser.add_argument("--fixtures", help="the fixtures directory (default: meep_identity.fixtures_root())")
    parser.add_argument("--rows", help="append one JSON line per case here")
    arguments = parser.parse_args(argv)

    from meep_gpu import array_path_oracle, meep_identity as identity  # noqa: PLC0415
    from parity.meep_gpu import meep_identity_smoke as smoke  # noqa: PLC0415

    mode = arguments.mode
    work = Path(arguments.work).resolve()
    fixtures = Path(arguments.fixtures).resolve() if arguments.fixtures else identity.fixtures_root()
    if arguments.cases_file:
        from parity.meep_gpu.meep_identity_tier_c import load_cases  # noqa: PLC0415

        default_cases = list(load_cases(arguments.cases_file).CASES)
    else:
        default_cases = list(smoke.CASES)
    cases = [c for c in arguments.cases.split(",") if c] if arguments.cases else default_cases
    if work.exists():
        print(f"MINT NOT RUN: {work} exists; give a new --work directory", flush=True)
        return 3
    started = time.time()
    before = input_digests()
    say(f"1/7 inputs hashed: {len(before)} files")

    check = load_tool("check_install")
    try:
        reference = reference_identity(check, mode, os.path.abspath(arguments.build_record))
    except (OSError, ValueError) as exc:
        print(f"MINT NOT RUN: the identity reference could not be judged: {exc}", flush=True)
        return 3
    if reference["verdict"] is not None:
        print(f"MINT REFUSED: {reference['verdict']}", flush=True)
        return 1
    say(f"2/7 identity reference in {mode}: MEEP {reference['meep']}, census "
        f"{reference['build_record']['census']}")
    if mode == "keep" and (arguments.cases_file or not set(cases) <= set(smoke.CASES)):
        done = subprocess.run([sys.executable, "-u", os.path.join(HERE, "meep_identity_smoke.py"),
                               "--mode", "keep"], capture_output=True, text=True, check=False)
        reference["smoke"] = done.stdout.strip().splitlines()[-1:] if done.stdout else []
        if done.returncode != 0:
            print(f"MINT REFUSED: the digest smoke did not reproduce its record: "
                  f"{reference['smoke']}", flush=True)
            return 1
        say(f"    digest smoke: {reference['smoke']}")

    work.mkdir(parents=True)
    rows = str(work / "tier_c_rows.jsonl")
    command = ["compare", "--out", str(work), "--mode", mode, "--cases", ",".join(cases), "--rows", rows]
    if arguments.cases_file:
        command += ["--cases-file", os.path.abspath(arguments.cases_file)]
    say(f"3/7 Tier C compare: {' '.join(command)}")
    rc_compare = run_runner(command)
    tier_c = {row["case"]: row for row in read_rows(rows)}
    full = {}
    for case in cases:
        path = work / mode / case / "result.json"
        if path.exists():
            with open(path, encoding="utf-8") as handle:
                full[case] = json.load(handle)
    replay_rows = str(work / "tier_a_rows.jsonl")
    if rc_compare in (0, 1) and len(full) == len(cases):
        say("4/7 Tier A replay of the Tier C output, MEEP blocked")
        run_runner(["replay", "--source", str(work), "--mode", mode, "--cases", ",".join(cases),
                    "--rows", replay_rows])
    else:
        say(f"4/7 Tier A replay not run: Tier C exited {rc_compare} with {len(full)} of {len(cases)} cases")
    replays = {row["case"]: row for row in read_rows(replay_rows)}

    problems: Dict[str, List[str]] = {}
    for case in cases:
        found = judge_case(mode, full.get(case) or {"error": (tier_c.get(case) or {}).get("error", "no result")},
                           replays.get(case) or {"error": "no replay result"})
        if found:
            problems[case] = found
    say(f"5/7 judged: {len(cases) - len(problems)} of {len(cases)} case(s) may be minted")
    for case, found in problems.items():
        for problem in found:
            say(f"    {case}: {problem}")

    total = sum(int((full[case].get("files") or {}).get(role, {}).get("bytes", 0))
                for case in full for role in ("digests", "planes"))
    for case in full:
        for role in ("capsule", "currents"):
            total += (work / mode / case / {"capsule": "capsule.capsule", "currents": "currents.json"}[role]).stat().st_size
    other_bytes = 0
    existing = None
    if (fixtures / identity.LEDGER_FILE).exists():
        existing = identity.read_ledger(fixtures)
        for other_mode, leaf in existing["sets"].items():
            if other_mode == mode:
                continue
            other_bytes += sum(int(spec["bytes"]) for entry in leaf["cases"].values()
                               for spec in entry["files"].values())
    say(f"6/7 fixture bytes: {total} for this set, {other_bytes} in other sets, budget "
        f"{identity.FIXTURE_BUDGET_BYTES}")
    if total + other_bytes > identity.FIXTURE_BUDGET_BYTES:
        problems.setdefault("budget", []).append(
            f"the fixtures would take {total + other_bytes} bytes, over the "
            f"{identity.FIXTURE_BUDGET_BYTES}-byte budget")
    after = input_digests()
    if after != before:
        moved = sorted(name for name in before if before[name] != after.get(name))
        problems.setdefault("inputs", []).append(f"inputs changed during the run: {moved}")
    for case in cases:
        if arguments.rows:
            with open(arguments.rows, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({"case": case, "mode": mode, "problems": problems.get(case, []),
                                         "tier_c": (tier_c.get(case) or {}), "tier_a": replays.get(case)},
                                        sort_keys=True, default=str) + "\n")
    if problems:
        print(f"MINT REFUSED: {sum(len(v) for v in problems.values())} problem(s) in "
              f"{sorted(problems)}; nothing was written", flush=True)
        return 1

    run = array_path_oracle.live()
    oracle = {"fixture_set": run.fixture_set, "fingerprint_sha256": run.fingerprint_sha256}
    staged = json.loads((work / identity.LEDGER_FILE).read_text(encoding="utf-8"))
    if staged["sets"][mode]["array_path_oracle"] != oracle:
        print("MINT REFUSED: the array-path oracle moved between Tier C and the mint", flush=True)
        return 1
    provenance_code, _absent = array_path_oracle.provenance_code_sha256(Path(ROOT))
    entries = {}
    staging = fixtures / f"{mode}.minting"
    if staging.exists():
        print(f"MINT NOT RUN: {staging} exists (an interrupted mint); remove it first", flush=True)
        return 3
    for case in cases:
        source = work / mode / case
        target = staging / case
        target.mkdir(parents=True)
        files = {}
        for role, name in (("capsule", "capsule.capsule"), ("currents", "currents.json"),
                           ("digests", "digests.json"), ("planes", "planes.npy")):
            shutil.copyfile(source / name, target / name)
            digest = sha256_file(str(target / name))
            if digest != sha256_file(str(source / name)):
                print(f"MINT NOT RUN: the copy of {source / name} does not read back", flush=True)
                return 3
            files[role] = {"file": f"{mode}/{case}/{name}", "sha256": digest,
                           "bytes": (target / name).stat().st_size}
        result, replay = full[case], replays[case]
        entries[case] = {
            "files": files,
            "tier_c": {key: result.get(key) for key in (
                "n_steps", "steps", "stored", "ghost", "reader", "reads", "fpu_checks",
                "planted_flip", "smoke_record", "subnormals", "omp_num_threads", "chunk_division",
                "seconds")},
            "tier_a": {key: replay.get(key) for key in (
                "stored_equal", "reader_equal", "digests", "planted_flip", "load_report",
                "meep_imported", "seconds")},
        }
    minted = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tool": TOOL,
              "tool_sha256": before[TOOL], "inputs_sha256": before,
              "host": (full[cases[0]].get("host") or {}), "seconds": round(time.time() - started, 1)}
    ledger = existing if existing is not None else {
        "schema": identity.LEDGER_SCHEMA, "version": identity.LEDGER_VERSION, "sets": {}}
    ledger["about"] = (
        "MEEP identity, Tier A: per subnormal mode, the minted cases' capsules, currents and "
        "digests of MEEP's words (meep_gpu/meep_identity.py). Written only by "
        "parity/meep_gpu/record_meep_identity.py.")
    ledger["fixture_budget_bytes"] = identity.FIXTURE_BUDGET_BYTES
    ledger["sets"][mode] = ledger_leaf(mode, entries, oracle, provenance_code, reference, minted)
    ledger["fixture_bytes"] = total + other_bytes
    old = fixtures / mode
    retired = fixtures / f"{mode}.retired"
    if old.exists():
        old.rename(retired)
    staging.rename(old)
    (fixtures / identity.LEDGER_FILE).write_text(json.dumps(ledger, indent=1, sort_keys=True) + "\n",
                                                 encoding="utf-8")
    if retired.exists():
        shutil.rmtree(retired)
    identity.read_ledger(fixtures)
    say(f"7/7 MINTED {fixtures / identity.LEDGER_FILE}: {mode} set of {len(cases)} case(s), "
        f"{sum(full[c]['stored']['words'] for c in cases)} compared stored words and "
        f"{sum(full[c]['reader']['words'] for c in cases)} reader words equal to MEEP's, every "
        f"digest reproduced by the replay, {total + other_bytes} fixture bytes "
        f"({time.time() - started:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
