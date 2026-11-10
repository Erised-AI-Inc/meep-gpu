#!/usr/bin/env python3
"""A fusion board's check that its census was cut against the tree the board prices.

A census records, on every measured row, ``subject_manifest_sha256``: one digest over the
kernels package its battery measured and the shared engine files its predicates read facts
from (``measure_predicate_coverage.subject_digest``, computed once before the first row). A
board that reads the census's selections but derives anything from the working tree pairs a
stale selection with a fresh read, and reports the difference as a finding. The CUDA board
refuses that by name (``build_cuda_fusion_matrix.subject_pin``); the Metal and Triton boards
use :func:`subject_pin` the same way.

The manifest is spelled exactly as the census driver spells it: the sorted lines
``"<relative path>  <sha256>\\n"`` of ``meep_gpu/<package>/*.py`` and :data:`SUBJECT_EXTRA`,
hashed once more. ``test_census_subject.py`` generates a manifest with the census driver's own
function over a copy of a tree and requires this module to reproduce it, so the two cannot
drift apart unnoticed (a drift would make every later board refuse a good census).

Rows that carry no stamp (a census cut by a driver that did not write one), rows stamped with
different manifests (not one census) and a manifest that is not this tree's are each refused
by name. Only the rows the board counts are asked: the unmeasured rows of a census carry no
measurement and no stamp.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

#: The engine files every battery's predicates read facts from, beside the kernels package;
#: the census driver's ``SUBJECT_EXTRA``, spelled the same.
SUBJECT_EXTRA = ("meep_gpu/stepping.py", "meep_gpu/fields.py",
                 "meep_gpu/dispersion.py", "meep_gpu/pml.py", "meep_gpu/grid.py")


def live_manifest(api_root: Path, package: str) -> Tuple[str, int]:
    """``(manifest sha256, subject file count)`` of the tree under ``api_root`` now."""
    api_root = Path(api_root)
    entries: Dict[str, str] = {}
    for path in sorted(api_root.glob(f"meep_gpu/{package}/*.py")):
        entries[str(path.relative_to(api_root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in SUBJECT_EXTRA:
        path = api_root / name
        if path.exists():
            entries[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = "".join(f"{name}  {digest}\n" for name, digest in sorted(entries.items()))
    return hashlib.sha256(manifest.encode("utf-8")).hexdigest(), len(entries)


def subject_pin(record: Iterable[dict], *, api_root: Path, package: str,
                census: Any) -> Dict[str, Any]:
    """The census's subject digest, RE-COMPUTED against the tree under ``api_root``.

    Raises SystemExit naming the disagreement; otherwise returns the pin's facts for the
    board's record. ``record`` is the measured rows the board counts; ``census`` names the
    census in the message."""
    rows = list(record)
    live, count = live_manifest(api_root, package)
    stamps = [row.get("subject_manifest_sha256") for row in rows]
    recorded = sorted({stamp for stamp in stamps if stamp}, key=str)
    unstamped = sum(1 for stamp in stamps if not stamp)
    if not rows:
        raise SystemExit(f"{census} has no measured rows, so there is no subject stamp to "
                         f"check against this tree ({live} over {count} subject files)")
    if unstamped:
        raise SystemExit(
            f"{census}: {unstamped} of {len(rows)} measured rows carry no "
            f"subject_manifest_sha256, so the census cannot be tied to a tree (it was cut by "
            f"a driver that did not stamp its rows, or by hand). This tree's "
            f"meep_gpu/{package} subject hashes to {live} over {count} files. Re-cut the "
            f"census with the promoted driver, or name one that carries the stamp.")
    if recorded != [live]:
        raise SystemExit(
            f"the census subject and the working tree disagree. {census} recorded "
            f"{recorded} over its subject files; this tree's meep_gpu/{package} subject "
            f"hashes to {live} over {count} files. The cells come from the recorded "
            f"selection and everything the board reads from this tree is measured from THIS "
            f"tree - pairing them would report the drift as a finding. Re-cut the census, or "
            f"check out the tree it was cut against.")
    return {"manifest_sha256": live, "subject_files": count,
            "scope": [f"meep_gpu/{package}/*.py", *SUBJECT_EXTRA], "rows_checked": len(rows),
            "agrees_with_census": True}
