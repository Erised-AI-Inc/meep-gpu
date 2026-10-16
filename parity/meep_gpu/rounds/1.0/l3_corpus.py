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

THE TABLE. ``triton``: an NVIDIA GPU under the package's default precedence (the Triton
table composes first). ``cuda``: an NVIDIA GPU with ``MEEP_GPU_BACKEND_PREFERENCE=cuda``
(the hand-written CUDA table composes first). ``metal``: the Apple GPU, flush only.

THE REFERENCE SETS. By default an NVIDIA leg judges against ``linux_<baseline>`` and a
Metal leg against ``linux_flush`` (primary), ``mac_flush`` and ``linux_keep``, all from the
same dispatched run (``--sets``, ``--primary``). A set in another mode than the run's is
compared, never relabelled, and must be named in ``--cross-mode-sets`` (default for Metal:
``linux_keep``). The lift must load the MEEP of ``--image-set`` (default: the primary set
on NVIDIA, ``mac_flush`` on Metal): its images' sha256 must equal the record's.

BEFORE A ROW RUNS its ``digests.json``, ``planes.npy`` and ``result.json`` in every judged
set are verified against the record's sha256; a row that does not verify is refused by
name and not run. Rows the primary set does not hold (refused or excluded in the record)
are named in the summary and never run; a row holding in the primary set but not in
another judged set refuses the leg.

THE ENVIRONMENT of every row is built here: every ``MEEP_GPU_*`` and ``TRIDENT_FDTD_*``
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

THE CORPUS IS NAMED, NEVER INHERITED. The rows rebuild their simulations from the MEEP
corpus: the MEEP 1.33.0 release tarball's ``python/`` tree, holding ``examples/`` and
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
card's lock for the leg's duration (``RUNBOOK.md``, CERT-5).

OUTPUT, under ``--out`` (AGENTS rule 7: everything is written as it lands):

* ``progress.log`` (and stdout): one flushed line per row (i of n, the verdict and digests
  equal of D per set, step path, table, dispatched slots, dispatches, N, wall) and a
  heartbeat every ``--heartbeat`` seconds naming the rows in flight;
* ``rows.jsonl``: one JSON line per row, appended as it lands; ``rows/<label>.json`` the
  row's full result (with the plan's ``fast_path_report``); ``logs/<label>.log`` the row's
  own per-step lines; ``dispatch_logs/<label>.jsonl`` its dispatch log;
* ``summary.json``, rewritten after every row; ``leg.json``, the leg's arguments,
  environment, launcher toolchain and checkout state at the start;
* ``LEG_DONE`` at the end, with the verdict.

THE VERDICT (``summary.json`` ``verdict``, the last progress line and the exit status):
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

RESUME. A label already in ``rows.jsonl`` is skipped, never re-appended; with
``--retry-errors`` a label whose last line is an error runs again and its new line is
appended (the last line of a label counts). ``<out>/STOP`` stops the leg before its next
row. A signal kills the rows in flight, which are not recorded, so a resume runs them.
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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l3_row  # noqa: E402
import legs  # noqa: E402  (the corpus rule's one home: legs.verify_corpus)

ROW_PROGRAM = HERE / "l3_row.py"
#: The checkout this file belongs to (``parity/meep_gpu/rounds/1.0/`` below its root).
DEFAULT_REPO = HERE.parents[3]
CORPUS_MODULE = "parity/meep_gpu/meep_identity_corpus.py"

TABLES = tuple(l3_row.TABLES)
NVIDIA_TABLES = ("triton", "cuda")
#: The default reference sets and image set of a Metal leg (owner ruling R1: Metal is
#: claimed against the x86 flush reference, and read beside the Apple flush reference
#: and the Linux keep reference from the same dispatched run).
METAL_SETS = ("linux_flush", "mac_flush", "linux_keep")
METAL_IMAGE_SET = "mac_flush"
METAL_CROSS_MODE = ("linux_keep",)

#: Removed from every row's environment by prefix, then set per leg and per row.
SCRUB_PREFIXES = ("MEEP_GPU_", "TRIDENT_FDTD_")
SCRUB_NAMES = ("PYTHONPATH", "CUPY_CACHE_DIR", "TRITON_CACHE_DIR")
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
CLEAN_ENV_KEEP = ("HOME", "USER", "LOGNAME")

#: The reference environment's build record, relative to the prefix of the rows'
#: interpreter (written by the reference build scripts' ``build_meep_133_record.py
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
                    help="the MEEP corpus: the MEEP 1.33.0 release tarball's python/ tree, "
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
    args.set_list = _comma(args.sets) or (list(METAL_SETS) if metal
                                          else [f"linux_{args.baseline}"])
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
    if bool(args.expect_differ) != bool(args.expect_differ_holds):
        raise Refusal("--expect-differ and --expect-differ-holds go together")
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
    rows_primary = sets[args.primary].get("rows") or {}
    holding = [label for label, row in rows_primary.items() if row.get("status") == "holds"]
    not_run = {label: row.get("status") for label, row in rows_primary.items()
               if row.get("status") != "holds"}
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
    if args.expect_differ == "references":
        expected = references_differ(args, record, labels)
    else:
        expected = read_list(args.expect_differ) if args.expect_differ else []
    stray = [label for label in expected if label not in rows_primary]
    if stray:
        raise Refusal(f"--expect-differ names rows the record does not hold: {stray}")
    return {"labels": labels, "not_run": not_run, "modes": modes,
            "meep_images": sets[args.image_set].get("meep_images") or {},
            "expected_differ": expected}


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
        self.refused_files: Dict[str, List[str]] = {}
        self.record_sha256 = sha256_file(Path(args.record))

    # ------------------------------------------------------------------ output ----
    def say(self, text: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {text}"
        with self.lock:
            print(line, flush=True)
            with open(self.out / "progress.log", "a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def load_done(self) -> None:
        path = self.out / "rows.jsonl"
        if not path.exists():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                self.rows[row["label"]] = row

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
        complete = (len(rows) == len(self.plan["labels"]) and not self.refused_files)
        if unexpected or expected_failed:
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
            "rows_not_holding_against_primary": sorted(unexpected),
            "verdict": verdict,
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }

    def write_summary(self, extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        with self.summary_lock:
            summary = self.summary()
            if extra:
                summary.update(extra)
            text = json.dumps(summary, indent=1, sort_keys=True)
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
                "cases_file": args.cases_file, "out_json": str(work / "result.json")}
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
                                           stdout=handle, stderr=subprocess.STDOUT)
                self.live[label] = process
                self.started_at[label] = started
            try:
                rc: Any = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                rc = f"killed after {args.timeout} s"
            finally:
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
        if (work / "dispatch_log.jsonl").exists():
            shutil.copy(work / "dispatch_log.jsonl", self.out / "dispatch_logs" / f"{label}.jsonl")
        (self.out / "rows" / f"{label}.json").write_text(
            json.dumps(row, sort_keys=True, default=str) + "\n", encoding="utf-8")
        row.pop("fast_path_report", None)
        row["l3_verdicts"] = {name: l3_row.set_verdict(row, name) for name in args.set_list}
        row["l3_dispatch"] = l3_row.dispatch_state(row)
        row["l3_verdict"] = l3_row.row_verdict(row, args.primary)
        shutil.rmtree(work, ignore_errors=True)
        return row

    def land(self, row: Dict[str, Any], planned: int) -> None:
        args = self.args
        with self.lock:
            with open(self.out / "rows.jsonl", "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self.rows[row["label"]] = row
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
        todo: List[tuple] = []
        for label in self.plan["labels"]:
            if label in self.rows and label not in retry:
                continue
            checked = verify_row(args, self.record, label)
            if "bad" in checked:
                self.refused_files[label] = checked["bad"]
                continue
            todo.append((label, checked["refs"]))
        planned = len(self.plan["labels"])
        self.say(f"START leg {args.leg}: table {args.table}, baseline {args.baseline}"
                 + (" (shipped default: no policy installed)" if args.shipped_default else "")
                 + f"; sets {args.set_list}, primary {args.primary}, image set {args.image_set}; "
                   f"{planned} rows planned of {len(self.record['sets'][args.primary]['rows'])} in "
                   f"the primary set ({len(self.plan['not_run'])} not holding in the record: "
                   f"{sorted(self.plan['not_run'])}); {len(self.rows) - len(retry)} already done; "
                   f"{len(retry)} to retry; {len(self.refused_files)} refused (files do not "
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
                if row is not None and not self.stopping:
                    self.land(row, planned)

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
                    else ""))
        if not self.stopping:
            (self.out / "LEG_DONE").write_text(
                json.dumps({"verdict": summary["verdict"],
                            "by_set": {name: summary["by_set"][name]["verdicts"]
                                       for name in args.set_list}}) + "\n", encoding="utf-8")
        return EXIT_HOLDS if summary["verdict"] == "HOLDS" else EXIT_NOT_HOLDS


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def refuse(why: str) -> int:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] REFUSING: {why}"
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
    (out / "leg.json").write_text(json.dumps({
        "argv": list(sys.argv if argv is None else argv),
        "args": {key: value for key, value in vars(args).items()
                 if isinstance(value, (str, int, float, bool, list, type(None)))},
        "environment": shown, "keep_cache_marked": minted, "corpus": args.corpus,
        "launcher_toolchain": getattr(args, "toolchain", None),
        "repo_at_start": repo_state(args.repo), "python": args.python,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    if args.expect_differ:
        (out / "expected_differ.txt").write_text(
            f"# {len(plan['expected_differ'])} rows expected to differ against {args.primary} and "
            f"to hold against {args.expect_differ_holds} ({args.expect_differ}); record "
            f"{args.record} sha256 {sha256_file(Path(args.record))}\n"
            + "".join(label + "\n" for label in plan["expected_differ"]), encoding="utf-8")
    leg = Leg(args, record, plan, env)

    def on_signal(signum: int, _frame: Any) -> None:
        leg.stop(f"signal {signum}")
        raise SystemExit(128 + signum)

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, on_signal)
    return leg.run()


if __name__ == "__main__":
    sys.exit(main())
