#!/usr/bin/env python
# SPDX-License-Identifier: GPL-2.0-or-later
"""Table 2's context column: the relative L2 of the array path against a MEEP that fuses.

    MEEP_IDENTITY_CORPUS_MANIFEST=FILE MGPU_SITE_MEEP_SOURCE=DIR \\
        python parity/meep_gpu/meep_identity_context_l2.py run --labels FILE --out DIR \\
            [--cases-file parity/meep_gpu/meep_identity_corpus.py] [--timeout S]

WHAT IT ANSWERS. Table 2 judges the array path by words against the identity reference
(Tier C, ``meep_identity_tier_c.py``). Beside the word counts it keeps one number from
its old form as CONTEXT, never as a verdict: the whole-field relative L2 against the
certification MEEP of the Mac lock (``cert``), a MEEP 1.33.0 single-precision build whose
compiler contracts multiplies and adds into fused operations. Where the array path holds
against the identity reference word for word, this number is MEEP's own build-to-build
spread (the identity reference against ``cert``), and it is what the old relative-L2
bands of Table 2 were measuring all along.

HOW, per row and in a process of its own, with the interpreter of the MEEP to compare
against (``cert``: the Mac lock's environment):

1. The row is built and lifted on the NumPy reference exactly as Tier C builds it
   (``meep_identity_tier_c.load_cases`` / ``build_case``, the cases file's
   ``after_lift`` grid check and its ``steps_planned`` N).
2. The driver steps to N, then MEEP steps the same simulation to N with
   ``sim.run(until=(N - 1/2) dt)``; both step counts are asserted.
3. The number is ``sweep_corpus_lift_parity.compare_fields``, the definition Table 2's
   L2 has always had: every one of Ex, Ey, Ez, Hx, Hy, Hz (Er, Ep, Ez, Hr, Hp, Hz on a
   cylindrical grid) read with ``get_array`` on both sides (``Fields.to_meep_array``,
   the package's MEEP-order reader), stacked as complex128, and
   ``|ours - theirs| / |theirs|`` over the whole volume, with each component's own ratio
   beside it. The words of those reads that differ are counted too.

The subnormal mode is the interpreter's default (``keep`` on Apple silicon), the thread
count is ``OMP_NUM_THREADS=1`` and kernel dispatch is off; each is recorded.

OUTPUT: ``DIR/rows.jsonl``, one line per row appended and flushed as it ends, and
``DIR/rows/<label>.json``; each row's process runs in ``DIR/rows``, where the corpus module
makes its scratch copy of the row's data files. A row already in ``rows.jsonl`` is
skipped, so a resumed run does not count a row twice. Progress: one flushed line per row
start and end.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SCHEMA = "meep_gpu.meep_identity_context_l2"
VERSION = 1
DEFAULT_CASES = os.path.join(HERE, "meep_identity_corpus.py")
DEFAULT_TIMEOUT_S = 14400


def say(text: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def summarize_reads(pairs: Dict[str, Any]) -> Dict[str, int]:
    """Words and differing words of the six ``get_array`` reads (bit patterns; no MEEP)."""
    import numpy  # noqa: PLC0415

    words = differing = 0
    for ours, theirs in pairs.values():
        ours, theirs = numpy.atleast_1d(ours), numpy.atleast_1d(theirs)
        if ours.shape != theirs.shape or ours.dtype != theirs.dtype:
            words += int(theirs.size)
            differing += int(theirs.size)
            continue
        view = numpy.uint32 if ours.dtype.itemsize in (4, 8) else None
        a = ours.view(view) if view is not None else ours
        b = theirs.view(view) if view is not None else theirs
        words += int(a.size)
        differing += int(numpy.count_nonzero(a != b))
    return {"words": words, "word_differing": differing}


def child(label: str, cases_file: str, out_json: str) -> int:
    """One row: lift, step both sides to N, the L2 and the differing words."""
    record: Dict[str, Any] = {"schema": SCHEMA, "version": VERSION, "label": label,
                              "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                              "dispatch": os.environ.get("MEEP_GPU_DISPATCH")}
    started = time.time()
    try:
        import numpy  # noqa: PLC0415
        import meep as mp  # noqa: PLC0415

        mp.verbosity(0)
        import meep_gpu  # noqa: PLC0415
        from meep_gpu import meep_identity as identity  # noqa: PLC0415
        from parity.meep_gpu.meep_identity_tier_c import build_case, load_cases, meep_component  # noqa: PLC0415
        from parity.meep_gpu.sweep_corpus_lift_parity import compare_fields  # noqa: PLC0415

        record["meep"] = {"version": str(mp.__version__), "single_precision": bool(mp.is_single_precision()),
                          "file": os.path.realpath(mp.__file__)}
        record["subnormals"] = identity.subnormal_state()
        cases = load_cases(cases_file)
        sim = build_case(cases, mp, label)
        driver = meep_gpu.lift_simulation(sim, prefer_gpu=False)
        if driver.gpu is not None:
            raise RuntimeError(f"the lift resolved driver.gpu = {driver.gpu!r}, not the NumPy reference")
        if hasattr(cases, "after_lift"):
            record["case_facts"] = cases.after_lift(mp, sim, driver)
        n_steps = int(cases.steps_planned(mp, sim))
        cylindrical = bool(getattr(driver.grid, "cylindrical", False))
        record.update({"n_steps": n_steps, "grid_shape": [int(v) for v in driver.grid.shape],
                       "cylindrical": cylindrical})
        while driver.step_count < n_steps:
            driver.step()
        if n_steps:
            sim.run(until=(n_steps - 0.5) * sim.fields.dt)
        if int(sim.fields.t) != n_steps or driver.step_count != n_steps:
            raise RuntimeError(f"MEEP at {int(sim.fields.t)}, driver at {driver.step_count}, not {n_steps}")
        alias = driver.h_alias_rule()

        class _Reads:
            def get_array(self, name: str) -> Any:
                return driver.fields.to_meep_array(name, alias)

        record.update(compare_fields(mp, numpy, _Reads(), sim, cylindrical))
        pairs = {}
        for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            pairs[name] = (numpy.asarray(driver.fields.to_meep_array(name, alias)),
                           numpy.asarray(sim.get_array(component=meep_component(mp, driver, name))))
        record["reader"] = summarize_reads(pairs)
        record["subnormals_after"] = identity.subnormal_state()
        driver.close()
    except BaseException as exc:  # noqa: BLE001 - a row that cannot be measured is recorded by name
        record["error"] = f"{type(exc).__name__}: {exc}"[:1500]
    record["seconds"] = round(time.time() - started, 3)
    Path(out_json).write_text(json.dumps(record, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return 0 if "error" not in record else 1


def run(labels: List[str], out: Path, cases_file: str, timeout: float) -> int:
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows").mkdir(exist_ok=True)
    rows_path = out / "rows.jsonl"
    done = set()
    if rows_path.exists():
        done = {json.loads(line)["label"] for line in rows_path.read_text(encoding="utf-8").splitlines()
                if line.strip()}
    todo = [label for label in labels if label not in done]
    say(f"START {len(labels)} rows, {len(done)} already recorded, {len(todo)} to run; cases {cases_file}")
    for index, label in enumerate(labels, 1):
        if label in done:
            continue
        target = out / "rows" / f"{label}.json"
        if target.exists():
            target.unlink()
        say(f"BEGIN {index}/{len(labels)} {label}")
        started = time.time()
        log = open(out / "rows" / f"{label}.log", "w", encoding="utf-8")
        try:
            # The child runs in DIR/rows: a corpus row is built in a scratch copy of its
            # data files made under the working directory (survey_meep_examples.scratch_workdir).
            rc = subprocess.run([sys.executable, "-u", os.path.abspath(__file__), "child", "--label", label,
                                 "--cases-file", cases_file, "--out-json", str(target)],
                                stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=False,
                                cwd=str(out / "rows")).returncode
        except subprocess.TimeoutExpired:
            rc = None
        log.close()
        if target.exists():
            row = json.loads(target.read_text(encoding="utf-8"))
        else:
            row = {"schema": SCHEMA, "version": VERSION, "label": label,
                   "error": f"the child exited {rc} with no record" if rc is not None
                   else f"the child ran past {timeout} s"}
        row["wall_s"] = round(time.time() - started, 1)
        with open(rows_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            handle.flush()
        reader = row.get("reader") or {}
        say(f"END {index}/{len(labels)} {label} L2 {row.get('parity_rel_l2')} reader "
            f"{reader.get('word_differing')}/{reader.get('words')} ({row['wall_s']} s)"
            + (f" ERROR {row['error'][:200]}" if row.get("error") else ""))
    say("FINISHED")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command")
    go = sub.add_parser("run", help="every label of FILE, one process each, resumable")
    go.add_argument("--labels", required=True, help="one row label per line")
    go.add_argument("--out", required=True)
    go.add_argument("--cases-file", dest="cases_file", default=DEFAULT_CASES)
    go.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    one = sub.add_parser("child", help=argparse.SUPPRESS)
    one.add_argument("--label", required=True)
    one.add_argument("--cases-file", dest="cases_file", required=True)
    one.add_argument("--out-json", dest="out_json", required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "run":
        labels = [line.strip() for line in Path(arguments.labels).read_text(encoding="utf-8").splitlines()
                  if line.strip() and not line.startswith("#")]
        return run(labels, Path(arguments.out).resolve(), os.path.abspath(arguments.cases_file),
                   arguments.timeout)
    if arguments.command == "child":
        return child(arguments.label, arguments.cases_file, arguments.out_json)
    parser.print_help()
    return 3


if __name__ == "__main__":
    sys.exit(main())
