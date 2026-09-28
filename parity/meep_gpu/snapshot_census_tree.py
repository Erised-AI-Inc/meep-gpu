"""Snapshot the bytes a WORKTREE-cut census is about to call, and stamp its provenance.

WHY A CENSUS NEEDS THIS AT ALL. A census cut at a released commit is dated by that
commit: ``git show <commit>:<module>`` is what the predicates were. This track is not
committed, so a worktree census has no such handle — ``git show HEAD:...`` returns bytes
the census never saw, and a staleness audit run against them reports CHANGED for
predicates that never moved, which reads as "the census is stale" precisely when it is
fresh. So the census carries its own copy of what it called.

WHAT IS SNAPSHOTTED, AND WHY IT IS NOT JUST ``triton_kernels``. Every earlier
worktree census snapshotted the package. From 2026-08-30 the fusion board also reads
``deposit_repair.seam_source_reasons`` and ``deposit_repair.repairable`` — the two
bodies that decide whether an in-seam deposit can be carried, and therefore the bodies
that answer 77 seam-instances since ``CARRIES_DEPOSIT_REPAIR`` flipped. A predicate
whose verdict is read but whose bytes are unaudited is credit resting on nothing, so
the file is snapshotted beside the package, at the relative position the board's
``AUDITED`` rows spell (``../deposit_repair.py``).

TWO ORDERING FACTS THIS SCRIPT EXISTS TO MAKE CHECKABLE, both stamped rather than
asserted in prose:

* ``snapshot_taken_before_any_kernel_edit`` — the board REFUSES to use a snapshot that
  does not claim this, because a snapshot taken afterwards proves nothing.
* ``snapshot_still_matches_the_tree`` — re-derived by ``--verify`` after the legs run.
  A kernel edited DURING the census would leave a snapshot that is honest about the
  start and wrong about the end, and only this second pass can see it.

ROOTS ARE RESOLVED BY NAME, NEVER BY ``parents[N]``. A promoted harness file that
counts directory levels resolves to the wrong root the moment it moves, and the
failure is silent in the worst way: children die on import while the parent writes a
full-length census of ``measured: false`` rows and exits 0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict


def find_api_root(start: Path) -> Path:
    """The ``the repository root`` package root, found BY NAME. Raises rather than guessing."""
    for candidate in (start, *start.parents):
        if (candidate / "meep_gpu" / "triton_kernels").is_dir():
            return candidate
    raise SystemExit(
        f"cannot locate the repository root above {start}: no ancestor holds "
        f"meep_gpu/triton_kernels. Refusing to guess — a wrong root writes a "
        f"full-length snapshot of the wrong bytes and exits 0.")


API = find_api_root(Path(__file__).resolve())


def find_repository(api: Path) -> Path:
    """The version-control root the ``git`` calls below run in: the nearest
    ancestor of ``api`` (``api`` included) that holds ``.git``, else ``api``."""
    for candidate in (api, *api.parents):
        if (candidate / ".git").exists():
            return candidate
    return api


REPO = find_repository(API)
PACKAGE = API / "meep_gpu" / "triton_kernels"
#: Files outside the package whose bodies the board audits. Relative to PACKAGE, so
#: the spelling here is the spelling `AUDITED` uses.
ALSO_SNAPSHOT = ("../deposit_repair.py",)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot_paths() -> Dict[str, Path]:
    """``{relative spelling: source path}`` — every file the snapshot must hold."""
    out: Dict[str, Path] = {p.name: p for p in sorted(PACKAGE.glob("*.py"))}
    for spelling in ALSO_SNAPSHOT:
        resolved = (PACKAGE / spelling).resolve()
        if not resolved.exists():
            raise SystemExit(f"ALSO_SNAPSHOT names {spelling!r}, which does not exist "
                             f"at {resolved}")
        out[spelling] = resolved
    return out


def take(census: Path) -> int:
    root = census / "tree_snapshot" / "triton_kernels"
    if root.exists():
        raise SystemExit(
            f"{root} already exists. A second snapshot would date the census by "
            f"whenever this was re-run, so it is refused rather than overwritten.")
    root.mkdir(parents=True)
    digests: Dict[str, str] = {}
    for spelling, source in snapshot_paths().items():
        destination = (root / spelling).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        digests[spelling] = sha256(source)
        print(f"[snapshot] {spelling}  {digests[spelling][:12]}", flush=True)

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                          capture_output=True, text=True, check=True).stdout.strip()
    newest = max((p.stat().st_mtime for p in PACKAGE.glob("*.py")), default=0.0)
    provenance = {
        "cut_against": "WORKTREE",
        "git_head": head,
        "snapshot_taken_before_any_kernel_edit": True,
        "snapshot_started_epoch": time.time(),
        "snapshot_started_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "newest_triton_kernels_mtime_epoch": newest,
        "newest_triton_kernels_mtime_local":
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(newest)),
        "tree_snapshot_sha256": digests,
        "also_snapshot": list(ALSO_SNAPSHOT),
        # Filled in by --verify, AFTER the legs. Absent means the census never
        # checked, which a reader must be able to tell from "checked and matched".
        "snapshot_still_matches_the_tree": None,
        "snapshot_files_differing_from_the_tree": None,
    }
    (census / "census_provenance.json").write_text(
        json.dumps(provenance, indent=1, sort_keys=True) + "\n")
    print(f"[snapshot] {len(digests)} files -> {root}", flush=True)
    return 0


def verify(census: Path, extra: Dict[str, str]) -> int:
    path = census / "census_provenance.json"
    provenance = json.loads(path.read_text())
    differing = [spelling for spelling, source in snapshot_paths().items()
                 if provenance["tree_snapshot_sha256"].get(spelling) != sha256(source)]
    provenance["snapshot_files_differing_from_the_tree"] = differing
    provenance["snapshot_still_matches_the_tree"] = not differing
    provenance.update(extra)
    path.write_text(json.dumps(provenance, indent=1, sort_keys=True) + "\n")
    if differing:
        print(f"[verify] {len(differing)} file(s) CHANGED DURING THE CENSUS: "
              f"{differing}", flush=True)
        # Non-zero: the census measured a tree that no longer exists, and every
        # verdict it published is about bytes nobody can name.
        return 4
    print(f"[verify] all {len(provenance['tree_snapshot_sha256'])} snapshotted files "
          f"still match the tree", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--census", required=True)
    parser.add_argument("--verify", action="store_true",
                        help="re-derive the digests AFTER the legs and stamp whether "
                             "the tree moved during the run")
    parser.add_argument("--stamp", default=None,
                        help="JSON object merged into the provenance on --verify "
                             "(battery/driver/chain digests, corpus root, policy)")
    args = parser.parse_args()
    census = Path(args.census).resolve()
    census.mkdir(parents=True, exist_ok=True)
    if args.verify:
        return verify(census, json.loads(args.stamp) if args.stamp else {})
    return take(census)


if __name__ == "__main__":
    raise SystemExit(main())
