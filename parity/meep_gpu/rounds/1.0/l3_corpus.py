#!/usr/bin/env python3
"""L3, the direct corpus check: dispatched runs of the MEEP corpus, judged word for word
against the stored digests of the reference MEEP's words (the Tier B record).

One invocation is one LEG: one kernel table (or precedence) and one subnormal baseline,
over the corpus rows that hold in the reference set the leg judges against. Each row runs
in a process of its own (``l3_row.py``, beside this file), which lifts the row on this
host's GPU and steps the driver alone: MEEP is imported, because the lift needs it, and is
never stepped, so the leg runs on a GPU host that does not run MEEP.

    python -u l3_corpus.py --table triton|cuda|metal --baseline keep|flush \\
        --record <Tier B record>/meep_identity_tier_b.json --out <leg directory> \\
        [--repo <checkout>] [--rows LABEL,LABEL | --labels FILE] [--parallel K] [--gpu N]
        [--second-reference SET]

The table. ``triton``: an NVIDIA GPU under the package's default precedence (the Triton
table composes first). ``cuda``: an NVIDIA GPU with ``MEEP_GPU_BACKEND_PREFERENCE=cuda``
(the hand-written CUDA table composes first). ``metal``: the Apple GPU, flush only.

The reference sets. By default an NVIDIA leg judges against ``linux_<baseline>`` and a
Metal leg against ``linux_flush`` (primary), ``mac_flush`` and ``linux_keep``, all from the
same dispatched run (``--sets``, ``--primary``). A set in another mode than the run's is
compared, never relabelled, and must be named in ``--cross-mode-sets`` (default for Metal:
``linux_keep``). The lift must load the MEEP of ``--image-set`` (default: the primary set
on NVIDIA, ``mac_flush`` on Metal), a set the leg judges: its images' sha256 must equal
the record's.

A second reference build. The record may hold a second Linux reference of a mode, minted
from the corpus run of another build of the reference recipe on another CPU. Its words
equal ``linux_<baseline>``'s on every row whose Apple and Linux reference words agree, and
may differ on the record's host-dependent rows (those whose ``mac_<baseline>`` and
``linux_<baseline>`` words differ: the eigenmode solves, whose words follow the BLAS
kernel family the CPU's vector extensions select). An NVIDIA leg on a host whose reference
MEEP is that build names it with ``--second-reference SET``: the leg judges
``linux_<baseline>`` (primary) and ``SET``, the lift loads ``SET``'s MEEP, and the rows
where the two references' words differ in the record are expected to differ against
``linux_<baseline>`` and hold against ``SET``. Each of those rows must be host-dependent,
read off the record's digest files at their sha256 before any row runs; a row on which
the two Linux references differ while the Apple and Linux references agree refuses the
leg by name, so no such difference is ever expected. Every other row must hold against
``linux_<baseline>``. On NVIDIA the images of a set other than the primary are admitted
only this way.

Before a row runs, its ``digests.json``, ``planes.npy`` and ``result.json`` in every judged
set are verified against the record's sha256; a row that does not verify is refused by
name and not run. Rows the primary set does not hold, and rows the record names as
exclusions in the primary set whatever their measured status, are named in the summary and
never run; a row holding in the primary set but not in another judged set refuses the leg.

The environment of every row is built here: every ``MEEP_GPU_*`` and ``TRIDENT_FDTD_*``
variable is removed (dispatch on, residency and hardware scope at their defaults), with
``MEEP_GPU_BACKEND_PREFERENCE=cuda`` added for ``--table cuda`` and
``MEEP_GPU_DISPATCH_LOG`` set per row; ``OMP``, ``MKL``, ``OPENBLAS``, ``NUMEXPR`` and
``VECLIB`` threads 1; ``PYTHONNOUSERSITE=1``; ``PYTHONPATH`` = ``--meep-site`` (when given)
then the checkout. On NVIDIA: ``CUDA_VISIBLE_DEVICES`` from ``--gpu``,
``CUPY_ACCELERATORS`` empty, a CuPy and a Triton cache per leg under ``--cache-root``; a
keep leg's CuPy cache sits under a path carrying ``ftz_stripped`` and is minted by the
package (``subnormal_policy.point_cupy_cache_at_keep_policy``) before the first row; the
Triton launcher's toolchain (``legs.launcher_toolchain``), with or without ``--clean-env``:
``CC``, the calling shell's, else the reference environment's own compiler
(``<prefix>/bin/x86_64-conda-linux-gnu-cc``), and ``TRITON_LIBCUDA_PATH``, also first on
``LD_LIBRARY_PATH``, ``--libcuda-stub``, else the shell's, else the directory the
environment's build recorded (``share/meep-gpu-build/libcuda.json``); a reference
environment without its compiler or its recorded libcuda link refuses the leg by name,
since Triton would compile every launcher with the host's ``gcc``. Everything
else is inherited from the calling shell (the activated environment), or, with
``--clean-env``, nothing is: ``HOME``, ``USER``, ``LANG=C`` and ``PATH`` =
``<interpreter prefix>/bin:/usr/bin:/bin`` (``--path``), the reference rows' ``PATH``.
The corpus manifest is passed by ``--corpus-manifest`` (``MEEP_IDENTITY_CORPUS_MANIFEST``),
or inherited. ``--shipped-default`` (informational,
not a certification leg) runs the rows as a user's run would: the caller installs no
subnormal policy, ``CUPY_ACCELERATORS`` is not set and the CuPy cache is not minted; the
host FPU is recorded at every compared step and not enforced.

The corpus is named, never inherited. The rows rebuild their simulations from the MEEP
corpus: the MEEP 1.35.0 release tarball's ``python/`` tree, holding ``examples/`` and
``tests/``. ``--corpus-root`` names it; without it, the leg takes the tree the reference
environment's build kept, as that environment's build record names it (``--build-record``,
default ``<prefix of --python>/share/meep-gpu-build/reference_build.json``; its ``corpus``
entry: ``root``, relative to the prefix, ``list``, the sha256 of every corpus file, with
the list's own sha256, and ``siblings``, the files outside ``python/`` that corpus rows read,
each by path and sha256, kept beside it). Neither given nor recorded, the leg is refused by
name, and so is a record without ``siblings`` (``legs.corpus_siblings``). When
the record names a corpus, the tree the leg uses, given or recorded, is judged by the
round's one corpus rule, ``legs.verify_corpus`` (the rule ``metal_round.py`` and the gate
stage use): every listed file at its listed sha256, every sibling beside the root at its
sha256, and no unlisted file Python could
import (any suffix of ``importlib.machinery.all_suffixes()``, byte code outside
``__pycache__`` included), or the leg is refused naming the files; any other unlisted
file is recorded, and ``__pycache__`` is not read. Every row gets the root as ``MEEP_GPU_CORPUS_ROOT``,
the variable the round's drivers export, and its parent as ``MGPU_SITE_MEEP_SOURCE``, the
one the corpus module reads (``<source>/python/examples``), so a shell's own value of
either never reaches a row; ``leg.json`` records the root, where it came from and what was
verified.

It takes no GPU lock and activates no environment: on a shared host the caller holds the
card's lock for the leg's duration (``RUNBOOK.md``, "CERT-5: L3, the direct corpus check").

Output, under ``--out`` (AGENTS rule 7: everything is written as it lands):

* ``progress.log`` (and stdout): one flushed line per row (i of n, the verdict and digests
  equal of D per set, step path, table, dispatched slots, dispatches, N, wall) and a
  heartbeat every ``--heartbeat`` seconds naming the rows in flight;
* ``rows.jsonl``: one JSON line per row, appended as it lands; ``rows/<label>.json`` the
  row's full result (with the plan's ``fast_path_report``); ``logs/<label>.log`` the row's
  own per-step lines; ``dispatch_logs/<label>.jsonl`` its dispatch log;
* ``summary.json``, rewritten after every row, with ``import_records``: how many rows
  carry the import record their process wrote (``l3_row.IMPORTED_KEY``), how many
  repository files those records name, and any file named with two digests;
  ``leg.json``, the leg's arguments, environment, launcher toolchain and checkout state
  at the start;
* ``LEG_DONE`` at the end, with the verdict and the declared rows' findings.

The verdict (``summary.json`` ``verdict``, the last progress line and the exit status):
``HOLDS`` (exit 0) when every planned row ran, none errored, none was refused for its
files, and every row holds against the primary set; otherwise ``DIFFERS`` (a row differs
or errored) or ``INCOMPLETE`` (a planned row did not run), exit 1. Exit 3: refused before
any row ran (the record, the sets or the rows named), exit 2: usage. Rows expected to
differ against the primary set count as ``HOLDS`` when they differ against it and hold
against ``--expect-differ-holds`` instead: ``--expect-differ FILE`` names them, and
``--expect-differ references`` derives them from the record before any row runs, as the
rows whose two reference MEEPs' stored or reader digests differ at some compared step (on
Metal, ``linux_flush`` against ``mac_flush``: the rows where the two hosts' MEEPs do not
agree, so no run can equal both); the list is written to ``expected_differ.txt``.
``--second-reference`` derives its own list the same way, from the primary set and the
second reference, after showing every row of it host-dependent (above); ``summary.json``
names the second reference, the Apple set that showed it and the Apple digest files read
(``second_reference``).

Rows declared run-to-run variable. A row whose sources include an ``EigenModeSource`` is
declared run-to-run variable (:data:`VARIABLE_CLASS`), read off the corpus manifest's
``facts.source_kinds`` by the classifier the Table 2 generator uses
(``build_meep_identity_table.source_class``), before any row runs; ``declared_variable.txt``
names them. The lift solves such a row's mode again through MEEP's ``get_eigenmode`` (MPB),
whose eigensolver chooses exact or approximate line minimization from ``clock()`` timings,
so the deposited mode, and MEEP's own, can differ from one process to the next. Such a row
records the digest of the source tables its lift deposited and its own digests at every
compared step (``l3_row.py``). A run of it is as required when it holds against the
primary set (or, for a row expected to differ, differs against it and holds against
``--expect-differ-holds``); a run that is not runs again in a fresh process, up to
:data:`VARIABLE_ATTEMPTS` runs in all that produced words (a run that errored is not one,
and stops the re-runs as it stops any row). The row holds when a run holds and every
earlier run stepped source tables whose digest differs from the holding run's; a run on
the holding run's own tables that gave other words, a run without a table digest before
the holding one, or no holding run in :data:`VARIABLE_ATTEMPTS` is a finding
(``DIFFERS``). Each line of a declared row carries its attribution, stamped as it lands
(:data:`ATTRIBUTION_KEY`: the outcome over every run so far and each run's table digest),
and a holding line that is a finding reads ``l3_verdict`` ``differs``, so the last line
of a label, which counts as always, carries the row's outcome to a resume, a carry or any
other reader; ``summary.json`` reads the stamps (``run_to_run_variable``). Every run is
kept: its line in ``rows.jsonl`` and its row file, log and dispatch log under
``attempts/``. A line a carry placed is never run again: its runs are those its stamp
records. A leg is started again only with the declaration it started with, over the rows
planned at both starts (``declared_first_start.json``, written at the first start and never
rewritten, records it, and a line of a declared row names its label): another is refused by
name, and a resume that narrows or widens the planned rows is not. A row whose runs stopped
before it held or reached the cap (an error, or a stop between runs) is undecided, not a
finding, and a leg stopped with runs due reads ``INCOMPLETE``.
With no corpus manifest named no row is declared, every row is judged by the rule below,
and the leg says why. Every other row is judged exactly as before: a differing row is a
finding, read and not run again.

Plan pass and carry. ``--plan-only`` runs every row as ``l3_row.py``'s plan pass (lifted,
checked against its sets, one step, the plan recorded, nothing judged; verdict
``PLAN-ONLY``), the pass ``carry.py l3`` compares a leg's recorded plans with. A leg
directory seeded by ``carry.py place`` holds the carried rows, each with
``carried_from``; they count as run rows, and ``summary.json`` names them under
``carried``.

Resume. A label already in ``rows.jsonl`` is skipped, never re-appended; with
``--retry-errors`` a label whose last line is an error runs again and its new line is
appended (the last line of a label counts). A declared row whose last line is not as
required, and has had fewer than :data:`VARIABLE_ATTEMPTS` runs, continues its runs.
``<out>/STOP`` stops the leg before its next row. A signal kills the rows in flight, which
are not recorded, so a resume runs them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l3_row  # noqa: E402
import round_status  # noqa: E402  (it loads the record spellings, neutral_ids.py)
import legs  # noqa: E402  (the corpus rule's one home: legs.verify_corpus)

if str(HERE.parents[3]) not in sys.path:  # the checkout root, for the classifier's one home
    sys.path.append(str(HERE.parents[3]))
from parity.meep_gpu.build_meep_identity_table import source_class  # noqa: E402

#: The record spellings (``parity/meep_gpu/neutral_ids.py``): a leg's log, rows and summary
#: go back with the round, so they name no machine, account or local path.
N = round_status.neutral_ids()

#: The run-to-run variable class: rows whose sources include an ``EigenModeSource``. The
#: lift solves their mode again through MPB, whose eigensolver chooses exact or approximate
#: line minimization from ``clock()`` timings, so the deposited mode can differ from one
#: process to the next. The class's name is the classifier's own (the Table 2 generator's
#: ``source_class``), and its rows are read off the corpus manifest, never transcribed.
VARIABLE_CLASS = source_class({"source_kinds": ["EigenModeSource"]})
#: A declared row runs at most this many times (runs that produced words), each in a fresh
#: process, stopping at the first that is as required.
VARIABLE_ATTEMPTS = 4
#: Where each line of a declared row carries its attribution, stamped when the line lands:
#: :func:`attribute` over every run of the row up to and including that one, with each
#: run's table digest (:func:`stamp`). The last line's stamp is the row's outcome; a line
#: that held and is a finding reads ``l3_verdict`` ``differs``.
ATTRIBUTION_KEY = "l3_attribution"
#: The outcomes of :func:`attribute` under which a declared row holds.
HOLDING_OUTCOMES = ("holds", "holds_variable")
#: The outcomes of :func:`attribute` that are not yet a hold or a finding: every run so far
#: errored, or the runs stopped before the row held or reached :data:`VARIABLE_ATTEMPTS`.
UNDECIDED_OUTCOMES = ("error", "undecided")
#: Written once, at a leg's first start, beside ``leg.json`` and never rewritten: the rows the
#: leg declared run-to-run variable then, the rows it planned then, and the corpus manifest
#: it read (:func:`declared_resume_problem`).
FIRST_START = "declared_first_start.json"

ROW_PROGRAM = HERE / "l3_row.py"
#: The checkout this file belongs to (``parity/meep_gpu/rounds/1.0/`` below its root).
DEFAULT_REPO = HERE.parents[3]
CORPUS_MODULE = "parity/meep_gpu/meep_identity_corpus.py"

TABLES = tuple(l3_row.TABLES)
NVIDIA_TABLES = ("triton", "cuda")
#: The default reference sets and image set of a Metal leg (Metal is
#: claimed against the x86 flush reference, and read beside the Apple flush reference
#: and the Linux keep reference from the same dispatched run).
METAL_SETS = ("linux_flush", "mac_flush", "linux_keep")
METAL_IMAGE_SET = "mac_flush"
METAL_CROSS_MODE = ("linux_keep",)
#: The set an NVIDIA leg judges against by default, and the Apple set of the same mode whose
#: words, against it, say which rows are host-dependent (``--second-reference``).
LINUX_SET = "linux_{baseline}"
APPLE_SET = "mac_{baseline}"

#: Removed from every row's environment by prefix, then set per leg and per row.
SCRUB_PREFIXES = ("MEEP_GPU_", "TRIDENT_FDTD_")
SCRUB_NAMES = ("PYTHONPATH", "CUPY_CACHE_DIR", "TRITON_CACHE_DIR")
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
CLEAN_ENV_KEEP = ("HOME", "USER", "LOGNAME")

#: The reference environment's build record, relative to the prefix of the rows'
#: interpreter (written by the reference build scripts' ``build_meep_reference_record.py
#: manifest``).
BUILD_RECORD = Path("share") / "meep-gpu-build" / "reference_build.json"
#: The record's entry for the MEEP corpus the build kept: ``root``, the release tarball's
#: ``python/`` tree (relative to the prefix the record lies in, or absolute); ``list``, a
#: file relative to the record's directory holding one ``<sha256>  <path>`` line per
#: corpus file, the path relative to ``root``; ``list_sha256``, that file's sha256;
#: ``files``, its line count; and ``siblings``, the files outside ``python/`` that corpus
#: rows read (read by ``legs.corpus_siblings``).
CORPUS_RECORD_KEY = "corpus"
#: The variable the round's drivers export the corpus root as, and the one the corpus
#: module reads (``meep_identity_corpus.SOURCE_ENVIRONMENT``: ``<source>/python/...``).
CORPUS_VARIABLE = "MEEP_GPU_CORPUS_ROOT"
SOURCE_VARIABLE = "MGPU_SITE_MEEP_SOURCE"
CORPUS_DIRECTORIES = ("examples", "tests")

EXIT_HOLDS, EXIT_NOT_HOLDS, EXIT_USAGE, EXIT_REFUSED = 0, 1, 2, 3


class Refusal(Exception):
    """A named reason the leg cannot start."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_list(path: str) -> List[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


# ---------------------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------------------

def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="l3_corpus.py",
        description="L3: dispatched runs of the MEEP corpus judged against the stored digests "
                    "of the reference MEEP's words, one leg per invocation.")
    ap.add_argument("--table", choices=TABLES, required=True,
                    help="triton: NVIDIA, default precedence; cuda: NVIDIA with "
                         "MEEP_GPU_BACKEND_PREFERENCE=cuda; metal: the Apple GPU (flush only)")
    ap.add_argument("--baseline", choices=("keep", "flush"), required=True,
                    help="the subnormal baseline: the policy installed, and the mode of the "
                         "reference set judged against")
    ap.add_argument("--record", required=True,
                    help="the Tier B record (meep_identity_tier_b.json)")
    ap.add_argument("--ref-root", default=None,
                    help="the directory the record's row files are relative to "
                         "(default: the record's directory)")
    ap.add_argument("--expect-record-sha256", default=None,
                    help="refuse a record whose sha256 is not this")
    ap.add_argument("--out", required=True, help="the leg's directory")
    ap.add_argument("--leg", default=None,
                    help="the leg's name (default: <table>_<baseline>, _shipped appended)")
    ap.add_argument("--repo", default=str(DEFAULT_REPO),
                    help="the checkout the rows import the package and harness from "
                         "(default: the one this file is in)")
    ap.add_argument("--cases-file", default=None,
                    help=f"the corpus module (default: <repo>/{CORPUS_MODULE})")
    ap.add_argument("--sets", default=None,
                    help="comma list of reference sets judged (default: linux_<baseline> on "
                         "NVIDIA; " + ",".join(METAL_SETS) + " on Metal)")
    ap.add_argument("--primary", default=None,
                    help="the set the verdict is taken against (default: the first of --sets)")
    ap.add_argument("--cross-mode-sets", default=None,
                    help="sets allowed in another mode than the run's, compared and never "
                         "relabelled (default: none on NVIDIA, "
                         + ",".join(METAL_CROSS_MODE) + " on Metal)")
    ap.add_argument("--image-set", default=None,
                    help="the set whose MEEP images the lift must load (default: the primary "
                         "set on NVIDIA, " + METAL_IMAGE_SET + " on Metal)")
    rows = ap.add_mutually_exclusive_group()
    rows.add_argument("--rows", default=None, help="comma list of row labels (default: every "
                                                   "row the primary set holds)")
    rows.add_argument("--labels", default=None, help="a file of row labels, one per line, "
                                                     "'#' comments")
    ap.add_argument("--expect-differ", default=None,
                    help="rows expected to differ against the primary set: a file of labels, "
                         "or 'references' for the rows whose primary and --expect-differ-holds "
                         "reference digests differ in the record")
    ap.add_argument("--expect-differ-holds", default=None,
                    help="the set those rows must hold against instead")
    ap.add_argument("--second-reference", default=None, metavar="SET",
                    help="NVIDIA: this host's reference MEEP is the second Linux reference "
                         "build the record holds as SET: judge linux_<baseline> (primary) and "
                         "SET, load SET's MEEP, and expect the rows where the two references "
                         "differ in the record to differ against linux_<baseline> and hold "
                         "against SET; each must be a host-dependent row (mac_<baseline> "
                         "differs from linux_<baseline> on it), or the leg is refused")
    ap.add_argument("--parallel", type=int, default=1, help="rows at a time (default 1)")
    ap.add_argument("--timeout", type=int, default=5400,
                    help="seconds before a row is killed and recorded as an error")
    ap.add_argument("--heartbeat", type=int, default=120,
                    help="seconds between heartbeat lines (0: none)")
    ap.add_argument("--retry-errors", action="store_true",
                    help="run again a row whose last line in rows.jsonl is an error")
    ap.add_argument("--python", default=sys.executable,
                    help="the interpreter of the rows (default: this one)")
    ap.add_argument("--gpu", default=None,
                    help="NVIDIA: the card, as CUDA_VISIBLE_DEVICES (default: inherited)")
    ap.add_argument("--cache-root", default=None,
                    help="NVIDIA: where the leg's CuPy and Triton caches go "
                         "(default: <out>/caches)")
    ap.add_argument("--libcuda-stub", default=None,
                    help="NVIDIA: a directory holding the libcuda stub Triton links against "
                         "(default: $TRITON_LIBCUDA_PATH, else the one the reference "
                         "environment's build recorded in share/meep-gpu-build/libcuda.json)")
    ap.add_argument("--meep-site", default=None,
                    help="a site-packages directory placed first on PYTHONPATH, for a MEEP "
                         "built outside the interpreter's own")
    ap.add_argument("--corpus-root", default=None,
                    help="the MEEP corpus: the MEEP 1.35.0 release tarball's python/ tree, "
                         "holding examples/ and tests/ (default: the tree the reference "
                         "environment's build record names; exported as "
                         "MEEP_GPU_CORPUS_ROOT, its parent as MGPU_SITE_MEEP_SOURCE)")
    ap.add_argument("--build-record", default=None,
                    help="the reference environment's build record, which names the corpus "
                         "the build kept (default: <prefix of --python>/share/meep-gpu-build/"
                         "reference_build.json)")
    ap.add_argument("--corpus-manifest", default=None,
                    help="the corpus manifest (MEEP_IDENTITY_CORPUS_MANIFEST; default: inherited)")
    ap.add_argument("--clean-env", action="store_true",
                    help="build each row's environment from nothing but HOME, USER, LANG=C, "
                         "TMPDIR and PATH, instead of the calling shell's")
    ap.add_argument("--path", default=None,
                    help="with --clean-env: the rows' PATH (default: "
                         "<interpreter prefix>/bin:/usr/bin:/bin)")
    ap.add_argument("--plan-only", action="store_true",
                    help="not a certification leg: every row is lifted, checked against "
                         "its sets, stepped ONCE and its plan recorded (l3_row.py's "
                         "plan_only); nothing is judged and the verdict is PLAN-ONLY. The "
                         "plan pass carry.py compares a leg's recorded plans with")
    ap.add_argument("--shipped-default", action="store_true",
                    help="informational: install no subnormal policy and leave the CuPy "
                         "environment as a user's run has it; the FPU is recorded, not "
                         "enforced")
    ap.add_argument("--child", default=str(ROW_PROGRAM),
                    help="the per-row program (default: l3_row.py beside this file)")
    return ap


def _comma(text: Optional[str]) -> List[str]:
    return [item.strip() for item in (text or "").split(",") if item.strip()]


def resolve(args: argparse.Namespace) -> argparse.Namespace:
    """Fill the table-dependent defaults; refuse a combination the check cannot run."""
    if args.table == "metal" and args.baseline != "flush":
        raise Refusal("the Metal table is certified under flush only: the Apple GPU flushes "
                      "subnormals and exposes no lever to keep them (--baseline flush)")
    if args.parallel < 1:
        raise Refusal(f"--parallel {args.parallel}: at least one row at a time")
    metal = args.table == "metal"
    if bool(args.expect_differ) != bool(args.expect_differ_holds):
        raise Refusal("--expect-differ and --expect-differ-holds go together")
    if args.second_reference:
        second_reference_arguments(args)
    args.set_list = _comma(args.sets) or (list(METAL_SETS) if metal
                                          else [LINUX_SET.format(baseline=args.baseline)])
    if len(set(args.set_list)) != len(args.set_list):
        raise Refusal(f"--sets names a set twice: {args.set_list}")
    args.primary = args.primary or args.set_list[0]
    if args.primary not in args.set_list:
        raise Refusal(f"the primary set {args.primary} is not among the sets {args.set_list}")
    if args.cross_mode_sets is None:
        args.cross_mode = set(METAL_CROSS_MODE) if metal else set()
    else:
        args.cross_mode = set(_comma(args.cross_mode_sets))
    if args.primary in args.cross_mode:
        raise Refusal(f"the primary set {args.primary} cannot be a cross-mode set")
    args.image_set = args.image_set or (METAL_IMAGE_SET if metal else args.primary)
    if args.image_set not in args.set_list:
        raise Refusal(f"the image set {args.image_set} is not among the sets {args.set_list}: "
                      f"the lift loads the MEEP of a set the leg judges")
    if args.expect_differ_holds and args.expect_differ_holds not in args.set_list:
        raise Refusal(f"--expect-differ-holds {args.expect_differ_holds} is not among the "
                      f"sets {args.set_list}")
    args.leg = args.leg or (f"{args.table}_{args.baseline}"
                            + ("_shipped" if args.shipped_default else ""))
    args.repo = str(Path(args.repo).resolve())
    args.cases_file = str(Path(args.cases_file or Path(args.repo) / CORPUS_MODULE).resolve())
    args.record = str(Path(args.record).resolve())
    args.ref_root = str(Path(args.ref_root or Path(args.record).parent).resolve())
    args.out = str(Path(args.out).resolve())
    args.cache_root = str(Path(args.cache_root or Path(args.out) / "caches").resolve())
    return args


def second_reference_arguments(args: argparse.Namespace) -> None:
    """``--second-reference SET``: the sets, the image set and the set the expected rows
    hold against, filled in; refused on Metal, beside the flags it fills, and when it names
    the primary set itself."""
    if args.table not in NVIDIA_TABLES:
        raise Refusal(f"--second-reference names a second Linux reference build, which an "
                      f"NVIDIA leg loads; the {args.table} table is judged as it is")
    given = [flag for flag, value in (("--sets", args.sets), ("--primary", args.primary),
                                      ("--image-set", args.image_set),
                                      ("--expect-differ", args.expect_differ),
                                      ("--expect-differ-holds", args.expect_differ_holds))
             if value]
    if given:
        raise Refusal(f"--second-reference sets the leg's sets, primary, image set and the rows "
                      f"expected to differ itself; it is not given with {', '.join(given)}")
    primary = LINUX_SET.format(baseline=args.baseline)
    if args.second_reference == primary:
        raise Refusal(f"--second-reference {primary} is the leg's primary set; name the second "
                      f"reference build's own set")
    args.sets = f"{primary},{args.second_reference}"
    args.image_set = args.second_reference
    args.expect_differ_holds = args.second_reference


# ---------------------------------------------------------------------------------------
# The record and the rows
# ---------------------------------------------------------------------------------------

def plan_rows(args: argparse.Namespace, record: Mapping[str, Any]) -> Dict[str, Any]:
    """The rows the leg runs, the rows the record does not hold, and the modes of the sets."""
    sets = record.get("sets") or {}
    for name in [*args.set_list, args.image_set]:
        if name not in sets:
            raise Refusal(f"set {name} is not in the record (it holds {sorted(sets)})")
    modes = {name: sets[name].get("mode") for name in args.set_list}
    for name, mode in modes.items():
        if mode != args.baseline and name not in args.cross_mode:
            raise Refusal(f"set {name} is mode {mode}, not {args.baseline}; a set of another "
                          f"mode is compared only when named in --cross-mode-sets")
    second = getattr(args, "second_reference", None)
    if args.table in NVIDIA_TABLES and args.image_set != args.primary and not second:
        raise Refusal(f"an NVIDIA leg loads the MEEP of its primary set {args.primary}; the "
                      f"images of {args.image_set} are admitted only by --second-reference, "
                      f"which judges that build's host-dependent rows")
    if second and modes.get(second) != args.baseline:
        raise Refusal(f"the second reference {second} is mode {modes.get(second)}, not "
                      f"{args.baseline}")
    rows_primary = sets[args.primary].get("rows") or {}
    # An excluded row keeps its measured status beside its reason in the record, and may hold
    # in one process and differ in the next: it is never planned, whatever it measured.
    holding = [label for label, row in rows_primary.items()
               if row.get("status") == "holds" and not row.get("excluded")]
    planned = set(holding)
    not_run = {label: (f"excluded ({row.get('status')})" if row.get("excluded")
                       else row.get("status"))
               for label, row in rows_primary.items() if label not in planned}
    for name in args.set_list:
        other = sets[name].get("rows") or {}
        missing = [label for label in holding if (other.get(label) or {}).get("status") != "holds"]
        if missing:
            raise Refusal(f"rows holding in {args.primary} but not in {name}: {missing}")
    labels = holding
    wanted = _comma(args.rows) if args.rows else (read_list(args.labels) if args.labels else None)
    if wanted is not None:
        if len(set(wanted)) != len(wanted):
            raise Refusal("a row is named twice")
        missing = [label for label in wanted if label not in holding]
        if missing:
            raise Refusal(f"rows not holding in {args.primary}: {missing}")
        labels = wanted
    reference = None
    if second:
        reference = second_reference_differ(args, record, labels)
        expected = reference["rows_expected_to_differ"]
    elif args.expect_differ == "references":
        expected = references_differ(args, record, labels)
    else:
        expected = read_list(args.expect_differ) if args.expect_differ else []
    stray = [label for label in expected if label not in rows_primary]
    if stray:
        raise Refusal(f"--expect-differ names rows the record does not hold: {stray}")
    return {"labels": labels, "not_run": not_run, "modes": modes,
            "meep_images": sets[args.image_set].get("meep_images") or {},
            "expected_differ": expected, "second_reference": reference}


def references_differ(args: argparse.Namespace, record: Mapping[str, Any],
                      labels: Sequence[str]) -> List[str]:
    """The planned rows whose primary-set and ``--expect-differ-holds``-set MEEP digests
    differ, stored or reader, at some compared step: read off the record's own digest
    files, before any row runs. A row whose digest file is missing is left out (it is then
    refused for its files when its turn comes)."""
    root = Path(args.ref_root)
    other = args.expect_differ_holds
    differ = []
    for label in labels:
        documents = []
        for name in (args.primary, other):
            entry = ((record["sets"][name]["rows"].get(label) or {}).get("files") or {}).get("digests") or {}
            path = root / entry.get("file", "")
            if not entry.get("file") or not path.is_file():
                break
            documents.append(json.loads(path.read_text(encoding="utf-8")))
        if len(documents) == 2 and any(documents[0].get(kind) != documents[1].get(kind)
                                       for kind in ("stored", "reader")):
            differ.append(label)
    return differ


def second_reference_differ(args: argparse.Namespace, record: Mapping[str, Any],
                            labels: Sequence[str]) -> Dict[str, Any]:
    """The planned rows on which the second reference's MEEP digests differ from the primary
    set's, stored or reader, at some compared step, each shown host-dependent: the record's
    Apple set of the mode (:data:`APPLE_SET`) differs from the primary set on it too. Every
    digest file read is first checked against the record's sha256. A row the two Linux
    references differ on while the Apple and Linux references agree refuses the leg by name:
    a difference there is a finding, never an expected one."""
    root = Path(args.ref_root)
    sets = record["sets"]
    primary, second = args.primary, args.second_reference
    apple = APPLE_SET.format(baseline=args.baseline)
    if apple not in sets:
        raise Refusal(f"the record holds no {apple}: without the Apple reference of mode "
                      f"{args.baseline} no row can be shown host-dependent, so the second "
                      f"reference {second} is not judged")

    def digests(name: str, label: str) -> Tuple[Dict[str, Any], str]:
        entry = (((sets[name]["rows"].get(label) or {}).get("files") or {})
                 .get("digests") or {})
        path = root / entry.get("file", "")
        if not entry.get("file") or not entry.get("sha256") or not path.is_file():
            raise Refusal(f"{name} {label}: the record names no digests file there, so "
                          f"whether {second} may differ on it cannot be read")
        if sha256_file(path) != entry["sha256"]:
            raise Refusal(f"{name} {label}: the digests file's sha256 differs from the record")
        return json.loads(path.read_text(encoding="utf-8")), entry["file"]

    def same(one: Mapping[str, Any], other: Mapping[str, Any]) -> bool:
        return all(one.get(kind) == other.get(kind) for kind in ("stored", "reader"))

    expected: List[str] = []
    agreeing: List[str] = []
    apple_read: List[str] = []
    for label in labels:
        old, _ = digests(primary, label)
        new, _ = digests(second, label)
        if same(old, new):
            continue
        words, path = digests(apple, label)
        apple_read.append(path)
        (agreeing if same(words, old) else expected).append(label)
    if agreeing:
        raise Refusal(f"the second reference {second} differs from {primary} on rows where the "
                      f"record's Apple and Linux references agree ({apple} equals {primary}): "
                      f"{agreeing}; a difference there is a finding, never an expected one")
    return {"set": second, "of": primary, "host_dependence": apple,
            "rows_expected_to_differ": expected, "apple_digests_read": apple_read}


def verify_row(args: argparse.Namespace, record: Mapping[str, Any],
               label: str) -> Dict[str, Any]:
    """``{"refs": {set: directory}}``, or ``{"bad": [...]}`` when a file fails the record."""
    bad, refs = [], {}
    root = Path(args.ref_root)
    for name in args.set_list:
        files = (record["sets"][name]["rows"][label] or {}).get("files") or {}
        for role in ("digests", "planes", "result"):
            entry = files.get(role) or {}
            if not entry.get("file") or not entry.get("sha256"):
                bad.append(f"{name} {role} not named by the record")
                continue
            path = root / entry["file"]
            if not path.is_file():
                bad.append(f"{name} {role} missing")
            elif sha256_file(path) != entry["sha256"]:
                bad.append(f"{name} {role} sha256 differs from the record")
        if not bad:
            refs[name] = str((root / files["digests"]["file"]).parent)
    return {"bad": bad} if bad else {"refs": refs}


# ---------------------------------------------------------------------------------------
# Rows declared run-to-run variable
# ---------------------------------------------------------------------------------------

def declared_variable(args: argparse.Namespace, labels: Sequence[str]) -> Dict[str, Any]:
    """The planned rows the leg declares run-to-run variable (:data:`VARIABLE_CLASS`), read
    off the corpus manifest (``--corpus-manifest``, else ``MEEP_IDENTITY_CORPUS_MANIFEST``,
    the one the rows read) before any row runs.

    With no manifest named no row is declared, and the result says why: every row is then
    judged by the strict rule. A manifest that names no ``source_kinds`` for a planned row
    refuses the leg by name, since whether that row is run-to-run variable cannot be read.
    """
    path = getattr(args, "corpus_manifest", None) or os.environ.get("MEEP_IDENTITY_CORPUS_MANIFEST")
    if not path:
        return {"class": VARIABLE_CLASS, "read": False, "rows": [],
                "why": "no corpus manifest named, so no row is declared and every row is "
                       "judged strictly"}
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = {row["label"]: row for row in document["rows"]}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise Refusal(f"the corpus manifest {path} does not read: {type(exc).__name__}: "
                      f"{exc}") from exc
    blind = [label for label in labels
             if "source_kinds" not in ((rows.get(label) or {}).get("facts") or {})]
    if blind:
        raise Refusal(f"the corpus manifest {path} names no source_kinds for {blind}, so "
                      f"whether they are run-to-run variable cannot be read")
    return {"class": VARIABLE_CLASS, "read": True, "attempts": VARIABLE_ATTEMPTS,
            "manifest": path, "manifest_sha256": sha256_file(Path(path)),
            "rows": sorted(label for label in labels
                           if source_class(rows[label]["facts"]) == VARIABLE_CLASS)}


def table_digest(line: Mapping[str, Any]) -> Optional[str]:
    """The sha256 of the source tables a declared row's run deposited, or None."""
    return (((line.get(l3_row.DECLARED_KEY) or {}).get("tables")) or {}).get("sha256")


def attribute(lines: Sequence[Mapping[str, Any]],
              as_required: Callable[[Mapping[str, Any]], bool]) -> Tuple[str, str]:
    """``(outcome, why)`` for every line of one declared row, in the order they ran.

    A line that errored carries no words and no tables, so it decides nothing and is left
    out. Of the rest, the first that is as required is the holding run: ``holds`` when it is
    the first, ``holds_variable`` when every earlier run stepped source tables whose digest
    differs from the holding run's. ``differs`` when no run held, when an earlier run
    stepped the holding run's own tables (the same inputs gave other words: a finding, not
    the eigenmode solve), or when a run carries no table digest; ``error`` when every run
    errored; ``undecided`` when no run held yet and fewer than :data:`VARIABLE_ATTEMPTS`
    runs produced words (more runs are due, or an error stopped them)."""
    judged = [line for line in lines if not line.get("error")]
    if not judged:
        return "error", f"every run errored ({len(lines)})"
    held = [index for index, line in enumerate(judged) if as_required(line)]
    if not held:
        if len(judged) < VARIABLE_ATTEMPTS:
            return "undecided", (f"no run of {len(judged)} held so far; up to "
                                 f"{VARIABLE_ATTEMPTS} runs may run")
        return "differs", f"no run of {len(judged)} held"
    hold = judged[held[0]]
    for line in judged[:held[0]]:
        if not table_digest(line) or not table_digest(hold):
            return "differs", ("a run carries no source-table digest, so the run that held is "
                               "not attributed")
        if table_digest(line) == table_digest(hold):
            return "differs", ("the same source tables gave other words: not the eigenmode "
                               "solve")
    if held[0] == 0:
        return "holds", "held on run 1"
    return "holds_variable", f"held on run {held[0] + 1}, every earlier run on other source tables"


def stamp(lines: Sequence[Mapping[str, Any]],
          as_required: Callable[[Mapping[str, Any]], bool]) -> Dict[str, Any]:
    """The attribution a declared row's line carries (:data:`ATTRIBUTION_KEY`): the
    outcome of :func:`attribute` over ``lines`` (every run of the row so far, this one
    last), the runs that produced words and those that errored, each judged run's table
    digest, and the first difference of each judged run that was not as required."""
    outcome, why = attribute(lines, as_required)
    judged = [line for line in lines if not line.get("error")]
    return {"outcome": outcome, "why": why, "class": VARIABLE_CLASS,
            "runs_with_words": len(judged), "runs_errored": len(lines) - len(judged),
            "tables": [table_digest(line) for line in judged],
            "first_differences": [line.get("first_difference") for line in judged
                                  if not as_required(line)]}


def _declared_lines(out: Path) -> Tuple[set, set]:
    """``(labels with a line, labels with a line of a declared row)`` in ``rows.jsonl``: a
    declared row's line carries ``attempt`` or :data:`ATTRIBUTION_KEY`."""
    rows = out / "rows.jsonl"
    if not rows.is_file():
        return set(), set()
    lines = [json.loads(line) for line in rows.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    return ({str(row.get("label")) for row in lines},
            {str(row.get("label")) for row in lines if "attempt" in row or ATTRIBUTION_KEY in row})


def first_start(out: Path) -> Optional[Dict[str, Any]]:
    """The leg's first-start record (:data:`FIRST_START`), or, for a leg directory that holds
    rows and no such record (a leg written before it existed), the one its rows imply: the
    rows with a line of a declared row were declared, every row with a line was planned."""
    path = out / FIRST_START
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    ran, declared = _declared_lines(out)
    if not ran:
        return None
    return {"rows": sorted(declared), "planned": sorted(ran), "manifest_sha256": None,
            "implied_by_rows": True}


def declared_resume_problem(out: Path, labels: Sequence[str],
                            variable: Mapping[str, Any]) -> Optional[str]:
    """Why a leg directory cannot be started again with this declared set, or None.

    The rows declared at the leg's first start (:data:`FIRST_START`, written once and never
    rewritten) must be the rows this start declares, over the rows planned at both starts:
    a row planned at only one of them is not compared, so a resume that narrows or widens
    ``--rows``, or a leg a carry seeded, is accepted. Of the rows this start plans, every
    one ``rows.jsonl`` holds a declared row's line of (``attempt`` or
    :data:`ATTRIBUTION_KEY`) must be declared. Otherwise a row's runs would be judged by
    another rule than the one they ran under; over the same rows only the corpus manifest
    can change the declaration, so the refusal names it."""
    now = set(variable.get("rows") or ())
    planned = set(labels)
    first = first_start(out)
    if first is not None:
        both = planned & set(first.get("planned") or ())
        was = set(first.get("rows") or ()) & both
        is_now = now & both
        if was != is_now:
            then = first.get("manifest_sha256") or ("not recorded" if first.get("implied_by_rows")
                                                    else "none named")
            return (f"the leg in {out} declared {len(was)} of the {len(both)} rows planned both at "
                    f"its first start and now run-to-run variable (corpus manifest sha256 {then}), "
                    f"and this start declares {len(is_now)} of them "
                    f"({variable.get('manifest_sha256') or variable.get('why')}): declared then "
                    f"and not now {sorted(was - is_now)}, now and not then "
                    f"{sorted(is_now - was)}; start it with the corpus manifest it started with")
    _ran, declared = _declared_lines(out)
    missing = sorted((declared & planned) - now)
    if missing:
        return (f"{out / 'rows.jsonl'} holds runs of rows declared run-to-run variable that this "
                f"start plans and does not declare: {missing}; start it with the corpus "
                f"manifest it started with")
    return None


# ---------------------------------------------------------------------------------------
# The environment
# ---------------------------------------------------------------------------------------

def python_prefix(python: str) -> str:
    path = Path(python)
    return str(path.parent.parent) if path.parent.name == "bin" else str(path.parent)


# ---------------------------------------------------------------------------------------
# The corpus
# ---------------------------------------------------------------------------------------

def record_prefix(record_path: Path) -> Path:
    """The prefix a build record lies in (``<prefix>/share/meep-gpu-build/<record>``)."""
    return record_path.parent.parent.parent


def recorded_corpus(record_path: Path) -> Optional[Dict[str, Any]]:
    """The corpus the build record names (:data:`CORPUS_RECORD_KEY`): ``{"root", "files",
    "siblings", "list", "list_sha256", "record_sha256"}``, or None when there is no record
    or it names no corpus. A record that does not read, an entry without ``root``, ``list``,
    ``list_sha256`` and its file count ``files`` (the contract ``legs.read_corpus_record``
    holds), a list whose sha256 is not the record's, that does not parse or that
    disagrees with the record's file count, or an entry whose ``siblings`` are absent or
    malformed (``legs.corpus_siblings``, the reading ``legs.read_corpus_record`` uses), is
    refused by name."""
    if not record_path.is_file():
        return None
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise Refusal(f"the build record {record_path} does not read: {exc}") from exc
    entry = record.get(CORPUS_RECORD_KEY) if isinstance(record, dict) else None
    if entry is None:
        return None
    if not (isinstance(entry, dict) and all(isinstance(entry.get(key), str) and entry[key]
                                            for key in ("root", "list", "list_sha256"))
            and isinstance(entry.get("files"), int) and not isinstance(entry["files"], bool)):
        raise Refusal(f"the build record {record_path} names a MEEP corpus "
                      f"({CORPUS_RECORD_KEY!r}) without a root, a file list, its sha256 and "
                      f"its file count")
    listing = record_path.parent / entry["list"]
    if not listing.is_file():
        raise Refusal(f"the corpus file list {listing} the build record names does not exist")
    if sha256_file(listing) != entry["list_sha256"]:
        raise Refusal(f"the corpus file list {listing} is not the one the build record "
                      f"{record_path} names (its sha256 differs)")
    files: Dict[str, str] = {}
    for number, line in enumerate(listing.read_text(encoding="utf-8").splitlines(), 1):
        digest, _, rel = line.partition("  ")
        if len(digest) != 64 or not rel:
            raise Refusal(f"{listing}:{number}: not a '<sha256>  <path>' line")
        files[rel] = digest
    if not files or entry["files"] != len(files):
        raise Refusal(f"the corpus file list {listing} holds {len(files)} files; the build "
                      f"record names {entry.get('files')}")
    siblings, why = legs.corpus_siblings(entry, record_path)
    if why:
        raise Refusal(why)
    root = Path(entry["root"]).expanduser()
    if not root.is_absolute():
        root = record_prefix(record_path) / root
    return {"root": root.resolve(), "files": files, "siblings": siblings, "list": str(listing),
            "list_sha256": entry["list_sha256"], "record_sha256": sha256_file(record_path)}


def resolve_corpus(args: argparse.Namespace) -> Dict[str, Any]:
    """The MEEP corpus the leg's rows read, and where it came from: ``--corpus-root``, else
    the reference environment's build record. Refused by name when neither names one, when
    the root is not a ``python/`` tree holding ``examples/`` and ``tests/``, and, when the
    record names a corpus, when the tree differs from the record's file list by the round's
    one corpus rule (``legs.verify_corpus``)."""
    record_path = Path(args.build_record or Path(python_prefix(args.python)) / BUILD_RECORD)
    recorded = recorded_corpus(record_path)
    if args.corpus_root:
        root, source = Path(args.corpus_root).expanduser().resolve(), "--corpus-root"
    elif recorded is not None:
        root, source = recorded["root"], "build record"
    else:
        why = ("does not exist" if not record_path.is_file()
               else f"names none ({CORPUS_RECORD_KEY!r})")
        raise Refusal(f"no MEEP corpus: --corpus-root was not given and the build record "
                      f"{record_path} {why}; the corpus is named, never inherited")
    if root.name != "python":
        raise Refusal(f"the MEEP corpus root {root} is not a python/ tree: the corpus module "
                      f"reads <{SOURCE_VARIABLE}>/python/examples and /tests")
    missing = [name for name in CORPUS_DIRECTORIES if not (root / name).is_dir()]
    if missing:
        raise Refusal(f"the MEEP corpus root {root} holds no {'/'.join(missing)}/")
    corpus: Dict[str, Any] = {
        "root": str(root), "source": source,
        "build_record": str(record_path) if record_path.is_file() else None,
        "build_record_sha256": (recorded or {}).get("record_sha256"),
        "list": (recorded or {}).get("list"), "list_sha256": (recorded or {}).get("list_sha256"),
        "siblings": dict((recorded or {}).get("siblings") or {}), "unlisted": []}
    if recorded is None:
        corpus["verified"] = (f"not verified: {record_path} "
                              + ("does not exist" if not record_path.is_file()
                                 else "names no corpus"))
        return corpus
    problems, notes = legs.verify_corpus(root, recorded["files"], recorded["siblings"])
    if problems:
        raise Refusal(f"the MEEP corpus {root} is not the one the build record {record_path} "
                      f"names: {'; '.join(problems)}")
    corpus["unlisted"] = notes[1:]
    corpus["verified"] = "; ".join(notes)
    return corpus


def leg_environment(args: argparse.Namespace, base: Mapping[str, str]) -> Dict[str, str]:
    """The environment every row of the leg starts from (``MEEP_GPU_DISPATCH_LOG`` is
    added per row)."""
    if args.clean_env:
        env = {name: base[name] for name in CLEAN_ENV_KEEP if name in base}
        env.update({"LANG": "C",
                    "PATH": args.path or f"{python_prefix(args.python)}/bin:/usr/bin:/bin"})
    else:
        env = {name: value for name, value in base.items()
               if not name.startswith(SCRUB_PREFIXES) and name not in SCRUB_NAMES}
    for name in THREAD_VARIABLES:
        env[name] = "1"
    env.update({"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONUNBUFFERED": "1", "KMP_DUPLICATE_LIB_OK": "TRUE", "MPLBACKEND": "Agg",
                "TMPDIR": str(Path(args.out) / "tmp")})
    env["PYTHONPATH"] = os.pathsep.join([p for p in (args.meep_site, args.repo) if p])
    corpus = getattr(args, "corpus", None) or {}
    if corpus.get("root"):
        env[CORPUS_VARIABLE] = corpus["root"]
        env[SOURCE_VARIABLE] = str(Path(corpus["root"]).parent)
    if args.corpus_manifest:
        env["MEEP_IDENTITY_CORPUS_MANIFEST"] = args.corpus_manifest
    preference = l3_row.TABLES[args.table]["preference"]
    if preference:
        env["MEEP_GPU_BACKEND_PREFERENCE"] = preference
    if args.table in NVIDIA_TABLES:
        if args.gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        if args.shipped_default:
            env.pop("CUPY_ACCELERATORS", None)
            cupy = "cupy_shipped"
        else:
            env["CUPY_ACCELERATORS"] = ""
            cupy = "cupy_ftz_stripped" if args.baseline == "keep" else "cupy_plain"
        caches = Path(args.cache_root) / args.leg
        env["CUPY_CACHE_DIR"] = str(caches / cupy)
        env["TRITON_CACHE_DIR"] = str(caches / "triton")
        toolchain = getattr(args, "toolchain", None)
        if toolchain is None:
            toolchain = {"libcuda": args.libcuda_stub}
        legs.launcher_environment(env, toolchain)
    return env


def leg_toolchain(args: argparse.Namespace, environ: Mapping[str, str]) -> Dict[str, Any]:
    """The Triton launcher's toolchain of an NVIDIA leg (``legs.launcher_toolchain``): the
    caller's ``CC`` and ``--libcuda-stub`` (else ``TRITON_LIBCUDA_PATH``) first, else the
    reference environment of ``--python``'s; refused by name when that environment lacks
    its compiler or its recorded libcuda link."""
    toolchain = legs.launcher_toolchain(python_prefix(args.python), dict(environ),
                                        args.libcuda_stub or "")
    if toolchain["problems"]:
        raise Refusal("; ".join(toolchain["problems"]))
    return toolchain


def mint_keep_cache(args: argparse.Namespace, env: Mapping[str, str]) -> str:
    """Point the keep leg's CuPy cache at the keep policy, as the package does it; refuse
    the leg when the package does not mark it."""
    cache = Path(env["CUPY_CACHE_DIR"])
    cache.mkdir(parents=True, exist_ok=True)
    code = ("from meep_gpu import subnormal_policy as sp; "
            "print(sp.point_cupy_cache_at_keep_policy()['provenance']['marked'])")
    done = subprocess.run([args.python, "-s", "-c", code], cwd=args.repo, env=dict(env),
                          capture_output=True, text=True, timeout=600)
    last = (done.stdout.strip().splitlines() or [""])[-1]
    if done.returncode != 0 or last != "True":
        raise Refusal(f"the keep CuPy cache at {cache} was not marked by the package "
                      f"(rc {done.returncode}): {(done.stderr or done.stdout)[-400:]}")
    return last


def repo_state(repo: str) -> Dict[str, Any]:
    """The checkout's HEAD, tree and porcelain status, or why they could not be read."""
    def git(*argv: str) -> str:
        return subprocess.run(["git", "-C", repo, *argv], capture_output=True, text=True,
                              timeout=120, check=True).stdout.strip()
    try:
        return {"head": git("rev-parse", "HEAD"), "tree": git("rev-parse", "HEAD^{tree}"),
                "dirty": sorted(git("status", "--porcelain", "--untracked-files=all").splitlines())}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"unavailable": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------------------
# The leg
# ---------------------------------------------------------------------------------------

class Leg:
    def __init__(self, args: argparse.Namespace, record: Mapping[str, Any],
                 plan: Mapping[str, Any], env: Mapping[str, str]):
        self.args, self.record, self.plan, self.env = args, record, plan, dict(env)
        self.out = Path(args.out)
        self.lock = threading.Lock()
        self.summary_lock = threading.Lock()
        self.live: Dict[str, subprocess.Popen] = {}
        self.started_at: Dict[str, float] = {}
        self.stopping = False
        self.t0 = time.time()
        self.rows: Dict[str, Dict[str, Any]] = {}
        #: Every line of every label, in the order they landed (a declared row's runs).
        self.attempts: Dict[str, List[Dict[str, Any]]] = {}
        self.declared = set((plan.get("declared_variable") or {}).get("rows") or [])
        self.refused_files: Dict[str, List[str]] = {}
        self.record_sha256 = sha256_file(Path(args.record))

    # ------------------------------------------------------------------ output ----
    def say(self, text: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {N.text(text)}"
        with self.lock:
            print(line, flush=True)
            with open(self.out / "progress.log", "a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def declared_text(self) -> str:
        """The rows the leg declares run-to-run variable, or why it declares none."""
        variable = self.plan.get("declared_variable") or {}
        if not variable.get("read"):
            return f"no row declared run-to-run variable ({variable.get('why', 'not read')})"
        return (f"{len(self.declared)} declared run-to-run variable ({VARIABLE_CLASS}, read off "
                f"the corpus manifest sha256 {variable.get('manifest_sha256')}), up to "
                f"{VARIABLE_ATTEMPTS} runs each")

    def load_done(self) -> None:
        path = self.out / "rows.jsonl"
        if not path.exists():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                self.rows[row["label"]] = row
                self.attempts.setdefault(row["label"], []).append(row)

    # ------------------------------------------------- the declared rows' runs ----
    def as_required(self, row: Mapping[str, Any]) -> bool:
        """Whether one run of a row is what the leg requires of it: it holds against the
        primary set, or, for a row expected to differ, differs against it and holds against
        ``--expect-differ-holds``."""
        if row.get("error"):
            return False
        verdicts = row.get("l3_verdicts") or {}
        if row.get("label") in set(self.plan["expected_differ"]):
            return (verdicts.get(self.args.primary) == "differs"
                    and verdicts.get(self.args.expect_differ_holds) == "holds")
        return verdicts.get(self.args.primary) == "holds"

    def rerun_due(self, label: str) -> bool:
        """A declared row whose last run is not as required, did not error, and has had
        fewer than :data:`VARIABLE_ATTEMPTS` runs that produced words, runs again. A line a
        carry placed (``carried_from``) never runs again: its runs are the ones its stamp
        records, in the leg it came from."""
        if label not in self.declared or getattr(self.args, "plan_only", False):
            return False
        lines = self.attempts.get(label) or []
        if not lines or lines[-1].get("error") or self.as_required(lines[-1]):
            return False
        if lines[-1].get("carried_from"):  # a carried line's runs are its stamp's, not this leg's
            return False
        return sum(1 for line in lines if not line.get("error")) < VARIABLE_ATTEMPTS

    def variable_outcome(self) -> Tuple[Dict[str, Any], Dict[str, Tuple[str, str]]]:
        """The ``run_to_run_variable`` block of the summary, and each declared row's
        ``(outcome, why)``, read off the stamp its last line carries
        (:data:`ATTRIBUTION_KEY`), never recomputed from the declaration of this start. A
        declared row whose last line carries no stamp (a line placed by a carry from a leg
        that stamped none) is attributed on the lines this leg holds. A plan pass judges
        nothing."""
        variable = self.plan.get("declared_variable") or {}
        block = {key: variable.get(key) for key in ("class", "read", "why", "manifest_sha256")
                 if key in variable}
        block.update({"attempts": VARIABLE_ATTEMPTS, "rows": sorted(self.declared)})
        if getattr(self.args, "plan_only", False):
            block.update({"judged": False, "why_not_judged": "a plan pass judges nothing",
                          "held_first": [], "held_on_rerun": {}, "runs": {}, "findings": {}})
            return block, {}
        outcomes: Dict[str, Tuple[str, str]] = {}
        stamps: Dict[str, Mapping[str, Any]] = {}
        for label in self.plan["labels"]:
            last = self.rows.get(label)
            if last is None:
                continue
            found = last.get(ATTRIBUTION_KEY)
            if not isinstance(found, Mapping):
                if label not in self.declared:
                    continue
                found = stamp(self.attempts.get(label) or [last], self.as_required)
            stamps[label] = found
            outcomes[label] = (str(found.get("outcome")), str(found.get("why")))
        block.update({
            "judged": True,
            "held_first": sorted(label for label, (kind, _w) in outcomes.items()
                                 if kind == "holds"),
            "held_on_rerun": {label: {"runs": stamps[label].get("runs_with_words"),
                                      "tables": stamps[label].get("tables"),
                                      "first_differences": stamps[label].get("first_differences")}
                              for label, (kind, _w) in outcomes.items()
                              if kind == "holds_variable"},
            "runs": {label: {"with_words": stamps[label].get("runs_with_words"),
                             "errored": stamps[label].get("runs_errored")}
                     for label in outcomes},
            "findings": {label: why for label, (kind, why) in outcomes.items()
                         if kind not in HOLDING_OUTCOMES + UNDECIDED_OUTCOMES},
            "undecided": {label: why for label, (kind, why) in outcomes.items()
                          if kind in UNDECIDED_OUTCOMES}})
        return block, outcomes

    def summary(self) -> Dict[str, Any]:
        args = self.args
        rows = [self.rows[label] for label in self.plan["labels"] if label in self.rows]
        by_set = {}
        for name in args.set_list:
            counts: Dict[str, int] = {}
            stored = reader = digests = flips = 0
            for row in rows:
                verdict = row["l3_verdicts"].get(name, "error")
                counts[verdict] = counts.get(verdict, 0) + 1
                tally = (row.get("per_set") or {}).get(name) or {}
                stored += tally.get("stored_equal") or 0
                reader += tally.get("reader_equal") or 0
                digests += tally.get("digests") or 0
                flip = tally.get("planted_flip") or {}
                flips += bool(flip.get("caught") and flip.get("placed"))
            by_set[name] = {
                "verdicts": counts, "digests_compared": digests,
                "stored_digests_equal": stored, "reader_digests_equal": reader,
                "flips_caught_and_placed": flips,
                "differs": sorted(row["label"] for row in rows
                                  if row["l3_verdicts"].get(name) == "differs")}
        dispatch: Dict[str, int] = {}
        verdicts: Dict[str, int] = {}
        tables: Dict[str, int] = {}
        reasons: Dict[str, List[str]] = {}
        for row in rows:
            dispatch[row["l3_dispatch"]] = dispatch.get(row["l3_dispatch"], 0) + 1
            verdicts[row["l3_verdict"]] = verdicts.get(row["l3_verdict"], 0) + 1
            facts = (row.get("dispatch_at_end") or {}).get("dispatch") or {}
            if row["l3_dispatch"] == "dispatched":
                tables[str(facts.get("table"))] = tables.get(str(facts.get("table")), 0) + 1
            elif row["l3_dispatch"] == "array":
                reasons.setdefault(str(facts.get("refused_because"))[:300], []).append(row["label"])
        digests_of: Dict[str, set] = {}
        for row in rows:
            for path, digest in (row.get(l3_row.IMPORTED_KEY) or {}).items():
                digests_of.setdefault(path, set()).add(digest)
        expected = set(self.plan["expected_differ"])
        primary = args.primary
        unexpected, expected_held, expected_failed = [], [], []
        for row in rows:
            against = row["l3_verdicts"].get(primary)
            if row["label"] in expected:
                other = row["l3_verdicts"].get(args.expect_differ_holds)
                if against == "differs" and other == "holds":
                    expected_held.append(row["label"])
                else:
                    expected_failed.append(row["label"])
            elif against != "holds":
                unexpected.append(row["label"])
        # A declared row holds only as attributed: a holding run whose earlier runs are not
        # shown to be other draws of the solve is a finding (rows_not_holding_against_primary).
        variable, outcomes = self.variable_outcome()
        for label, (kind, _why) in outcomes.items():
            if kind in HOLDING_OUTCOMES or kind in UNDECIDED_OUTCOMES:
                continue
            if label in expected_held:
                expected_held.remove(label)
            if label not in unexpected and label not in expected_failed:
                unexpected.append(label)
        # A declared row with runs still due (a leg stopped between them) is not done: it is
        # neither held nor a finding yet.
        due = sorted(label for label in self.declared if label in self.rows
                     and self.rerun_due(label))
        unexpected = [label for label in unexpected if label not in due]
        expected_failed = [label for label in expected_failed if label not in due]
        variable["runs_due"] = due
        complete = (len(rows) == len(self.plan["labels"]) and not self.refused_files
                    and not due)
        reference = self.plan.get("second_reference")
        if reference:  # the expected rows are listed under expected_differ
            reference = {key: value for key, value in reference.items()
                         if key != "rows_expected_to_differ"}
        if getattr(args, "plan_only", False):
            verdict = ("PLAN-ONLY" if complete and not any(r.get("error") for r in rows)
                       else "INCOMPLETE")
        elif unexpected or expected_failed:
            verdict = "DIFFERS"
        elif not complete:
            verdict = "INCOMPLETE"
        else:
            verdict = "HOLDS"
        return {
            "tool": "parity/meep_gpu/rounds/1.0/l3_corpus.py", "leg": args.leg,
            "table": args.table, "baseline": args.baseline,
            "shipped_default": bool(args.shipped_default), "sets": args.set_list,
            "primary": primary, "image_set": args.image_set,
            "record": args.record, "record_sha256": self.record_sha256,
            "rows_planned": len(self.plan["labels"]), "rows_done": len(rows),
            "by_set": by_set, "verdicts_against_primary": verdicts, "dispatch": dispatch,
            "dispatched_by_table": tables, "array_fallback_reasons": reasons,
            "errors": {row["label"]: row.get("error") for row in rows if row.get("error")},
            "not_run_record": self.plan["not_run"],
            "refused_unverified_files": self.refused_files,
            "expected_differ": {"rows": sorted(expected), "holds_against": args.expect_differ_holds,
                                "as_expected": sorted(expected_held),
                                "not_as_expected": sorted(expected_failed)},
            "second_reference": reference,
            "run_to_run_variable": variable,
            "rows_not_holding_against_primary": sorted(unexpected),
            "plan_only": bool(getattr(args, "plan_only", False)),
            # Rows a carry placed here from an earlier leg (carry.py place), each naming
            # its source and the proof it was carried by; they count as run rows.
            "carried": sorted(row["label"] for row in rows if row.get("carried_from")),
            "import_records": {
                "rows_with_one": sum(1 for row in rows if row.get(l3_row.IMPORTED_KEY)),
                "rows": len(rows), "files": len(digests_of),
                "files_with_two_digests": sorted(path for path, found in digests_of.items()
                                                 if len(found) > 1)},
            "verdict": verdict,
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }

    def write_summary(self, extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        with self.summary_lock:
            summary = self.summary()
            if extra:
                summary.update(extra)
            text = json.dumps(N.value(summary), indent=1, sort_keys=True)
            (self.out / "summary.json.tmp").write_text(text + "\n", encoding="utf-8")
            os.replace(self.out / "summary.json.tmp", self.out / "summary.json")
        return summary

    # ------------------------------------------------------------------- rows ----
    def run_row(self, label: str, refs: Mapping[str, str]) -> Optional[Dict[str, Any]]:
        args = self.args
        work = self.out / "work" / label
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        spec = {"label": label, "table": args.table, "mode": args.baseline,
                "shipped_default": bool(args.shipped_default), "refs": dict(refs),
                "ref_modes": self.plan["modes"], "primary": args.primary,
                "image_set": args.image_set, "meep_images": self.plan["meep_images"],
                "cases_file": args.cases_file, "out_json": str(work / "result.json"),
                l3_row.DECLARED_KEY: label in self.declared}
        declared = label in self.declared
        attempt = len(self.attempts.get(label) or []) + 1
        if declared:
            spec["attempt"] = attempt
        if getattr(args, "plan_only", False):
            spec["plan_only"] = True
        (work / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
        env = dict(self.env)
        env["MEEP_GPU_DISPATCH_LOG"] = str(work / "dispatch_log.jsonl")
        log = self.out / "logs" / f"{label}.log"
        started = time.time()
        with open(log, "w", encoding="utf-8") as handle:
            with self.lock:
                if self.stopping:
                    return None
                process = subprocess.Popen([args.python, "-s", "-u", args.child,
                                            str(work / "spec.json")], cwd=str(work), env=env,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                # The row's output reaches its log written neutrally (N.copy_lines).
                copier = N.copy_lines(process.stdout, handle)
                self.live[label] = process
                self.started_at[label] = started
            try:
                rc: Any = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                rc = f"killed after {args.timeout} s"
            finally:
                copier.join(timeout=120)
                with self.lock:
                    self.live.pop(label, None)
                    self.started_at.pop(label, None)
        if self.stopping:
            return None
        row = None
        if (work / "result.json").exists():
            try:
                row = json.loads((work / "result.json").read_text(encoding="utf-8"))
            except ValueError:
                row = None
        if row is None:
            tail = log.read_text(errors="replace").strip().splitlines()[-6:]
            row = {"label": label, "mode": args.baseline, "table": args.table,
                   "error": f"the row's process exited {rc} before reporting: "
                            f"{' | '.join(tail)}"}
        row["child_rc"] = rc
        row["wall_seconds"] = round(time.time() - started, 3)
        if declared:
            row["attempt"] = attempt
        if (work / "dispatch_log.jsonl").exists():
            shutil.copy(work / "dispatch_log.jsonl", self.out / "dispatch_logs" / f"{label}.jsonl")
        (self.out / "rows" / f"{label}.json").write_text(
            json.dumps(N.value(row), sort_keys=True, default=str) + "\n", encoding="utf-8")
        if declared:  # every run of a declared row is kept; the unsuffixed files are the last
            kept = self.out / "attempts"
            kept.mkdir(exist_ok=True)
            shutil.copy(self.out / "rows" / f"{label}.json", kept / f"{label}.run{attempt}.json")
            shutil.copy(log, kept / f"{label}.run{attempt}.log")
            if (work / "dispatch_log.jsonl").exists():
                shutil.copy(work / "dispatch_log.jsonl",
                            kept / f"{label}.run{attempt}.dispatch.jsonl")
        row.pop("fast_path_report", None)
        row["l3_verdicts"] = {name: l3_row.set_verdict(row, name) for name in args.set_list}
        row["l3_dispatch"] = l3_row.dispatch_state(row)
        row["l3_verdict"] = l3_row.row_verdict(row, args.primary)
        if getattr(args, "plan_only", False):  # a plan pass judges nothing
            judged = "error" if row.get("error") else "plan_only"
            row["l3_verdicts"] = {name: judged for name in args.set_list}
            row["l3_dispatch"] = row["l3_verdict"] = judged
        shutil.rmtree(work, ignore_errors=True)
        return row

    def land(self, row: Dict[str, Any], planned: int) -> None:
        args = self.args
        with self.lock:
            label = row["label"]
            if label in self.declared and not getattr(args, "plan_only", False):
                # The row's attribution travels on its own line, so a resume, a carry or any
                # reader of the last line sees it; a holding run that is a finding reads
                # differs.
                row[ATTRIBUTION_KEY] = stamp((self.attempts.get(label) or []) + [row],
                                             self.as_required)
                if (row[ATTRIBUTION_KEY]["outcome"] not in HOLDING_OUTCOMES
                        and self.as_required(row)):
                    row["l3_verdict"] = "differs"
            with open(self.out / "rows.jsonl", "a", encoding="utf-8") as handle:
                handle.write(json.dumps(N.value(row), sort_keys=True, default=str) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self.rows[row["label"]] = row
            self.attempts.setdefault(row["label"], []).append(row)
        summary = self.write_summary()
        end = row.get("dispatch_at_end") or {}
        facts = end.get("dispatch") or {}
        sums = row.get("launch_sums") or {}
        per = row.get("per_set") or {}
        done = sum(1 for label in self.plan["labels"] if label in self.rows)
        self.say(f"{args.leg} {done}/{planned} {row['label']}: "
                 + " ".join(f"{name}={row['l3_verdicts'].get(name)}"
                            f"({(per.get(name) or {}).get('stored_equal')}/"
                            f"{(per.get(name) or {}).get('reader_equal')} of "
                            f"{(per.get(name) or {}).get('digests')})"
                            for name in args.set_list)
                 + f"; {row['l3_dispatch']} step_path {end.get('step_path')} table "
                   f"{facts.get('table')} slots {len(facts.get('dispatched_slots') or [])}/"
                   f"{facts.get('slots')} dispatches {sums.get('dispatches')}; N "
                   f"{row.get('n_steps')}; {row.get('wall_seconds')} s"
                 + (f"; declared run-to-run variable, run {row.get('attempt')}, source tables "
                    f"{str(table_digest(row))[:16]}" if row["label"] in self.declared else "")
                 + (f"; ERROR {str(row.get('error'))[:300]}" if row.get("error") else "")
                 + f" | {args.primary} {summary['by_set'][args.primary]['verdicts']} "
                   f"dispatch {summary['dispatch']} ({time.time() - self.t0:.0f} s)")

    def heartbeat(self, done: threading.Event) -> None:
        while not done.wait(self.args.heartbeat):
            with self.lock:
                live = {label: time.time() - since for label, since in self.started_at.items()}
            if not live:
                continue
            parts = []
            for label, age in sorted(live.items()):
                log = self.out / "logs" / f"{label}.log"
                try:
                    last = (log.read_text(errors="replace").strip().splitlines() or [""])[-1]
                except OSError:
                    last = ""
                parts.append(f"{label} {age:.0f} s: {last[-160:]}")
            self.say(f"{self.args.leg} heartbeat, {len(live)} in flight: " + " || ".join(parts))

    def stop(self, why: str) -> None:
        with self.lock:
            self.stopping = True
            live = list(self.live.values())
        for process in live:
            try:
                process.kill()
            except OSError:
                pass
        self.say(f"{self.args.leg} stopping ({why}): {len(live)} row(s) in flight killed, "
                 f"not recorded")

    def run(self) -> int:
        args = self.args
        self.load_done()
        retry = {label for label, row in self.rows.items() if args.retry_errors and row.get("error")}
        again = {label for label in self.rows if self.rerun_due(label)}
        todo: List[tuple] = []
        for label in self.plan["labels"]:
            if label in self.rows and label not in retry and label not in again:
                continue
            checked = verify_row(args, self.record, label)
            if "bad" in checked:
                self.refused_files[label] = checked["bad"]
                continue
            todo.append((label, checked["refs"]))
        planned = len(self.plan["labels"])
        self.say(f"START leg {args.leg}: table {args.table}, baseline {args.baseline}"
                 + (" (shipped default: no policy installed)" if args.shipped_default else "")
                 + f"; sets {args.set_list}, primary {args.primary}, image set {args.image_set}"
                 + (f" (second reference {args.second_reference}: "
                    f"{len(self.plan['expected_differ'])} host-dependent rows expected to differ "
                    f"against {args.primary})" if getattr(args, "second_reference", None) else "")
                 + "; "
                   f"{planned} rows planned of {len(self.record['sets'][args.primary]['rows'])} in "
                   f"the primary set ({len(self.plan['not_run'])} not holding in the record: "
                   f"{sorted(self.plan['not_run'])}); "
                 + self.declared_text()
                 + f"; {len(self.rows) - len(retry) - len(again)} already done; "
                   f"{len(retry)} to retry; {len(again)} declared to run again; "
                   f"{len(self.refused_files)} refused (files do not "
                   f"verify): {sorted(self.refused_files)}; {len(todo)} to run, {args.parallel} "
                   f"at a time; record {args.record} sha256 {self.record_sha256}"
                 + (f"; MEEP corpus {args.corpus['root']} ({args.corpus['source']}; "
                    f"{args.corpus['verified']})" if getattr(args, "corpus", None) else ""))
        self.write_summary()
        queue = list(todo)
        done = threading.Event()
        beat = threading.Thread(target=self.heartbeat, args=(done,), daemon=True)
        if args.heartbeat > 0:
            beat.start()

        def worker() -> None:
            while True:
                with self.lock:
                    if self.stopping or not queue:
                        return
                    label, refs = queue.pop(0)
                if (self.out / "STOP").exists():
                    self.say(f"{args.leg}: STOP file present, no new row")
                    with self.lock:
                        queue.clear()
                    return
                try:
                    row = self.run_row(label, refs)
                except Exception as exc:  # noqa: BLE001 - recorded as the row's error
                    row = {"label": label, "error": f"driver: {type(exc).__name__}: {exc}",
                           "l3_verdicts": {name: "error" for name in args.set_list},
                           "l3_dispatch": "error", "l3_verdict": "error"}
                    if label in self.declared:
                        row["attempt"] = len(self.attempts.get(label) or []) + 1
                if row is not None and not self.stopping:
                    self.land(row, planned)
                    if self.rerun_due(label):
                        runs = sum(1 for line in self.attempts.get(label) or []
                                   if not line.get("error"))
                        with self.lock:
                            if not self.stopping:
                                queue.append((label, refs))
                        self.say(f"{args.leg} {label}: declared run-to-run variable and not as "
                                 f"required on run {runs} of {VARIABLE_ATTEMPTS}; it runs again "
                                 f"in a fresh process")

        workers = [threading.Thread(target=worker) for _ in range(min(args.parallel, len(todo)) or 0)]
        for thread in workers:
            thread.start()
        for thread in workers:
            thread.join()
        done.set()
        summary = self.write_summary({"repo_at_end": repo_state(args.repo)})
        self.say(f"END leg {args.leg}: "
                 + "; ".join(f"{name} {summary['by_set'][name]['verdicts']} digests stored "
                             f"{summary['by_set'][name]['stored_digests_equal']} reader "
                             f"{summary['by_set'][name]['reader_digests_equal']} of "
                             f"{summary['by_set'][name]['digests_compared']}, flips "
                             f"{summary['by_set'][name]['flips_caught_and_placed']}"
                             for name in args.set_list)
                 + f"; dispatch {summary['dispatch']} by table {summary['dispatched_by_table']}; "
                   f"errors {sorted(summary['errors'])}; refused files "
                   f"{sorted(summary['refused_unverified_files'])}")
        self.say(f"VERDICT {summary['verdict']} leg {args.leg}: {summary['rows_done']} of "
                 f"{summary['rows_planned']} rows run; against {args.primary} "
                 f"{summary['verdicts_against_primary']}"
                 + (f"; not holding {summary['rows_not_holding_against_primary']}"
                    if summary["rows_not_holding_against_primary"] else "")
                 + (f"; expected to differ, as expected "
                    f"{len(summary['expected_differ']['as_expected'])} of "
                    f"{len(summary['expected_differ']['rows'])}" if self.plan["expected_differ"]
                    else "")
                 + (f"; declared run-to-run variable {len(self.declared)}: held on run 1 "
                    f"{len(summary['run_to_run_variable']['held_first'])}, held on a re-run "
                    f"{len(summary['run_to_run_variable']['held_on_rerun'])} "
                    f"{sorted(summary['run_to_run_variable']['held_on_rerun'])}, findings "
                    f"{summary['run_to_run_variable']['findings']}" if self.declared else ""))
        if not self.stopping:
            (self.out / "LEG_DONE").write_text(
                json.dumps({"verdict": summary["verdict"],
                            "by_set": {name: summary["by_set"][name]["verdicts"]
                                       for name in args.set_list},
                            "run_to_run_findings": summary["run_to_run_variable"]["findings"]})
                + "\n", encoding="utf-8")
        return (EXIT_HOLDS if summary["verdict"] in ("HOLDS", "PLAN-ONLY")
                else EXIT_NOT_HOLDS)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def refuse(why: str) -> int:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] REFUSING: {N.text(why)}"
        print(line, flush=True)
        with open(out / "progress.log", "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        return EXIT_REFUSED

    try:
        args = resolve(args)
        record = json.loads(Path(args.record).read_text(encoding="utf-8"))
        if args.expect_record_sha256 and sha256_file(Path(args.record)) != args.expect_record_sha256:
            raise Refusal(f"the record {args.record} is not sha256 {args.expect_record_sha256}")
        plan = plan_rows(args, record)
        # plan_rows stays the plan round_inputs.py derives each leg's files from (it reads no
        # manifest); the declared class is read here, after it and before any row runs.
        plan["declared_variable"] = declared_variable(args, plan["labels"])
        problem = declared_resume_problem(out, plan["labels"], plan["declared_variable"])
        if problem:
            raise Refusal(problem)
        if not Path(args.cases_file).is_file():
            raise Refusal(f"no corpus module at {args.cases_file}")
        if not Path(args.child).is_file():
            raise Refusal(f"no row program at {args.child}")
        args.corpus = resolve_corpus(args)
        if args.table in NVIDIA_TABLES:
            args.toolchain = leg_toolchain(args, os.environ)
        for sub in ("logs", "rows", "work", "dispatch_logs", "tmp"):
            (out / sub).mkdir(parents=True, exist_ok=True)
        env = leg_environment(args, os.environ)
        minted = None
        if args.table in NVIDIA_TABLES and args.baseline == "keep" and not args.shipped_default:
            minted = mint_keep_cache(args, env)
        for name in ("CUPY_CACHE_DIR", "TRITON_CACHE_DIR"):
            if name in env:
                Path(env[name]).mkdir(parents=True, exist_ok=True)
    except (Refusal, OSError, ValueError, subprocess.SubprocessError) as exc:
        return refuse(str(exc))
    shown = {name: env.get(name) for name in sorted(env)
             if name == "CC" or name.startswith(
                 ("MEEP_GPU_", "CUPY_", "TRITON_", "CUDA_", "OMP_", "PYTHON", "MGPU_",
                  "MEEP_IDENTITY_", "LD_LIBRARY_PATH", "PATH"))}
    (out / "leg.json").write_text(json.dumps(N.value({
        "argv": list(sys.argv if argv is None else argv),
        "args": {key: value for key, value in vars(args).items()
                 if isinstance(value, (str, int, float, bool, list, type(None)))},
        "environment": shown, "keep_cache_marked": minted, "corpus": args.corpus,
        "declared_variable": plan["declared_variable"],
        "launcher_toolchain": getattr(args, "toolchain", None),
        "repo_at_start": repo_state(args.repo), "python": args.python,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}), indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    if not (out / FIRST_START).exists():  # once, at the leg's first start; never rewritten
        variable = plan["declared_variable"]
        (out / FIRST_START).write_text(json.dumps(N.value({
            "rows": variable["rows"], "planned": list(plan["labels"]), "read": variable["read"],
            "manifest_sha256": variable.get("manifest_sha256"), "why": variable.get("why"),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}), indent=1, sort_keys=True) + "\n",
            encoding="utf-8")
    if args.expect_differ or args.second_reference:
        how = args.expect_differ or (
            f"--second-reference: the record's rows where {args.second_reference} and "
            f"{args.primary} differ, each host-dependent")
        # The list goes back with the round: its header names the record as every other
        # record of the leg names a path (N.text).
        (out / "expected_differ.txt").write_text(N.text(
            f"# {len(plan['expected_differ'])} rows expected to differ against {args.primary} and "
            f"to hold against {args.expect_differ_holds} ({how}); record "
            f"{args.record} sha256 {sha256_file(Path(args.record))}") + "\n"
            + "".join(label + "\n" for label in plan["expected_differ"]), encoding="utf-8")
    variable = plan["declared_variable"]
    # The declared rows, or why there are none; it goes back with the round (N.text).
    (out / "declared_variable.txt").write_text(N.text(
        (f"# {len(variable['rows'])} of {len(plan['labels'])} planned rows declared run-to-run "
         f"variable ({variable['class']}), read off the corpus manifest {variable['manifest']} "
         f"sha256 {variable['manifest_sha256']}; each runs up to {VARIABLE_ATTEMPTS} times"
         if variable["read"] else f"# 0 rows declared run-to-run variable: {variable['why']}"))
        + "\n" + "".join(label + "\n" for label in variable["rows"]), encoding="utf-8")
    leg = Leg(args, record, plan, env)

    def on_signal(signum: int, _frame: Any) -> None:
        leg.stop(f"signal {signum}")
        raise SystemExit(128 + signum)

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, on_signal)
    return leg.run()


if __name__ == "__main__":
    sys.exit(main())
