"""Does a gate artifact's import record still describe the tree being released?

WHY THIS EXISTS. A chunked certification round (``rounds/1.0/chunks.py``) runs each chunk of
gates on the first pushed tip that holds that chunk's final bytes, while other series entries
are still landing. Under per-kernel certification (R17) that is sound exactly where an
artifact's import record (``gate_provenance.IMPORTED_KEY``: every repository module the gate's
process held) is byte-equal in the release tree. An artifact cut early whose record names a
file a later entry moved describes other bytes, and it must be refused by name rather than
bound: the rebind tools would otherwise take the curated digests from the record and leave the
weld pointing at a run of a tree that does not ship.

THE RULE, one home for every tool that applies it (the three rebind tools in their campaign
mode, and ``chunks.py verify`` over every artifact of a round):

* the record is ``imported_source_sha256``; a probe that wrote only its curated
  ``source_sha256`` is read through that map instead (the fallback
  ``rebind_triton_welds.py`` already applies), and an artifact carrying both is checked
  against both;
* a key is checked when it is a repository path of the package or the harness
  (``meep_gpu/`` or ``parity/meep_gpu/``) and its value is a sha256 digest; a gate's own
  outputs under ``parity/meep_gpu/results/`` are not release bytes and are not checked;
* a checked file the tree holds at other bytes, or no longer holds, has MOVED, and one moved
  file refuses the artifact.

Stdlib only, and imported by no gate: a change here voids no artifact.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

#: The record maps, in the order a reader takes them.
RECORD_KEYS = ("imported_source_sha256", "source_sha256")
#: Repository subtrees whose files are release bytes.
CHECKED_PREFIXES = ("meep_gpu/", "parity/meep_gpu/")
#: Under the harness, but a gate's own outputs, never release bytes.
NOT_RELEASE_PREFIXES = ("parity/meep_gpu/results/",)
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def checked(name: str) -> bool:
    """Whether a record key names a release file this rule compares."""
    return (isinstance(name, str) and name.startswith(CHECKED_PREFIXES)
            and not name.startswith(NOT_RELEASE_PREFIXES) and ".." not in name.split("/"))


def records(payload: Mapping[str, Any]) -> Dict[str, Dict[str, str]]:
    """The record maps an artifact carries, each reduced to its checked entries.

    Every map of :data:`RECORD_KEYS` the artifact carries is returned and checked: a probe
    that wrote only its curated ``source_sha256`` is checked through it, and an artifact
    carrying both is checked against both. A value that is not a mapping is no record.
    """
    out: Dict[str, Dict[str, str]] = {}
    for key in RECORD_KEYS:
        value = payload.get(key) if isinstance(payload, Mapping) else None
        if not isinstance(value, Mapping) or not value:
            continue
        kept = {name: digest for name, digest in value.items()
                if checked(name) and isinstance(digest, str) and _DIGEST.match(digest)}
        if kept:
            out[key] = kept
    return out


def _sha256(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def moved_files(record: Mapping[str, str], root: Path,
                cache: Optional[Dict[str, Optional[str]]] = None) -> List[str]:
    """The checked files of ``record`` the tree at ``root`` holds at other bytes, or not at
    all, sorted. ``cache`` (name -> live digest) may be shared across artifacts."""
    root = Path(root)
    cache = {} if cache is None else cache
    moved = []
    for name, digest in record.items():
        if not checked(name):
            continue
        if name not in cache:
            cache[name] = _sha256(root / name)
        if cache[name] != digest:
            moved.append(name)
    return sorted(moved)


def artifact_moved(payload: Mapping[str, Any], root: Path,
                   cache: Optional[Dict[str, Optional[str]]] = None) -> List[str]:
    """Every moved file over every record map the artifact carries, sorted."""
    found = set()
    for record in records(payload).values():
        found.update(moved_files(record, root, cache))
    return sorted(found)


def describe(moved: Iterable[str], root: Path, limit: int = 4) -> str:
    """``a.py (other bytes), b.py (absent) and 3 more``: the named part of a refusal."""
    moved = sorted(moved)
    parts = [f"{name} ({'absent' if not (Path(root) / name).is_file() else 'other bytes'})"
             for name in moved[:limit]]
    more = f" and {len(moved) - limit} more" if len(moved) > limit else ""
    return ", ".join(parts) + more


def refusal(artifact: str, moved: Iterable[str], root: Path) -> str:
    """The one sentence a tool prints when it refuses an artifact by this rule."""
    moved = sorted(moved)
    return (f"{artifact} records {len(moved)} file(s) the release tree does not hold at the "
            f"recorded bytes: {describe(moved, root)}; the run describes another tree, so it "
            f"is refused (re-run it on bytes equal to the release tree)")
