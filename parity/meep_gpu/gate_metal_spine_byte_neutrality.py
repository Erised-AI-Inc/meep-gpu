"""Did the spine extraction move a single certified kernel byte? Measured, not asserted.

THE REFACTOR THIS ARBITRATES. Tranche 1 shipped the whole Metal host layer in one
``launch.py``: 812 lines of which the two plan classes duplicated each other almost
verbatim and ``plan_step`` spelled its arm table as two literal loops. A
seven-family tree needs a spine, so ``device.py``, ``plans.py``, ``arms.py``,
``templates.py`` and ``preconditions.py`` were split out and ``launch.py`` re-imports
them.

THE RISK, STATED PRECISELY. Moving host code cannot change a kernel source unless
the emitters themselves changed — but "cannot" is an argument, and this package's
whole discipline is that arguments about float behaviour get replaced by
measurements. The certified claim rests on EIGHTEEN specialised source strings whose
sha256s are recorded in ``results/metal_pml_2026-08-14/provenance.json``, frozen at
certification time. If any one of them moved, the 2026-08-14 gate result no longer
describes the code in the tree.

WHAT THIS GATE ASSERTS, in two directions, because only the pair is evidence:

1. **EVERY certified kernel-source sha256 is UNCHANGED.** Compared against the
   frozen artifact, not against a value recomputed today — a baseline computed
   after the change would agree with the change by construction. Any difference
   FAILS.
2. **The HOST module hashes MOVED.** ``compute_fingerprints`` hashes the host
   modules deliberately, because the host chooses the Yee sub-lattice, binds the
   coefficient pointers and decides the specialisation constants — each of those a
   silent wrong answer if it changes, and none visible in a kernel source. So the
   refactor IS a fingerprint event, and a run where the host hashes had NOT moved
   would mean this gate was pointed at a tree where the refactor had not landed,
   which is a vacuous pass rather than a good one.

That pair — kernel hashes equal, host hashes moved — is better evidence that the
refactor was byte-neutral than not refactoring would have produced. It is also the
reason the moves were made BEFORE the first new family rather than after: seven
families are written against the spine instead of migrated onto it, and this
measurement bounds exactly one change instead of an accumulation.

Exit 0 neutral, 1 a kernel byte moved, 75 the baseline artifact is unreadable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
API_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
if API_ROOT not in sys.path:
    sys.path.insert(0, API_ROOT)

from meep_gpu.metal_kernels import launch as metal_launch  # noqa: E402
from meep_gpu.metal_kernels import shaders  # noqa: E402

#: The CERTIFIED artifact. Frozen at tranche 1's certification; the baseline must
#: come from here rather than from anything this round can recompute.
CERTIFIED = os.path.join(HERE, "results", "metal_pml_2026-08-14", "provenance.json")

#: The host modules tranche 1 hashed. Any module ADDED to ``HOST_MODULES`` since is
#: a new hash rather than a moved one, and is reported separately — a spine module
#: appearing for the first time is expected, a certified one changing is the event.
TRANCHE_1_HOST_MODULES: Tuple[str, ...] = ("shaders.py", "coverage.py",
                                           "launch.py", "subnormal.py")

EXIT_CANNOT_CERTIFY = 75


def log(message: str) -> None:
    print(message, flush=True)


def save(payload: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
        _stamp_provenance(payload)  # bytes THIS process imported; see gate_provenance
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def module_sha256(name: str) -> str:
    path = os.path.join(API_ROOT, "meep_gpu", "metal_kernels", name)
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="artifact JSON path")
    arguments = parser.parse_args()
    started = time.time()

    try:
        with open(CERTIFIED, "r", encoding="utf-8") as handle:
            certified = json.load(handle)
    except OSError as exc:
        log(f"CANNOT CERTIFY HERE: the frozen artifact is unreadable ({exc!r}); "
            f"a baseline recomputed today would agree with the change by "
            f"construction and would prove nothing")
        return EXIT_CANNOT_CERTIFY

    baseline_kernels: Dict[str, str] = dict(certified.get("kernel_sources", {}))
    baseline_hosts: Dict[str, str] = dict(certified.get("sources", {}))
    if not baseline_kernels:
        log("CANNOT CERTIFY HERE: the frozen artifact carries no kernel_sources "
            "block, so there is nothing to compare against")
        return EXIT_CANNOT_CERTIFY

    # --- 1. the kernel sources, as the tree emits them TODAY -----------------
    current_kernels = {label: shaders.source_sha256(source)
                       for label, source in shaders.enumerate_sources().items()}

    moved: List[Dict[str, str]] = []
    missing: List[str] = []
    for label, digest in sorted(baseline_kernels.items()):
        now = current_kernels.get(label)
        if now is None:
            missing.append(label)
        elif now != digest:
            moved.append({"label": label, "certified": digest, "now": now})
    added = sorted(set(current_kernels) - set(baseline_kernels))

    log(f"[kernels] certified={len(baseline_kernels)} emitted_now="
        f"{len(current_kernels)} moved={len(moved)} missing={len(missing)} "
        f"added={len(added)}")
    for entry in moved:
        log(f"  MOVED {entry['label']}: {entry['certified'][:16]} -> "
            f"{entry['now'][:16]}")
    for label in missing:
        log(f"  MISSING {label}: the tree no longer emits a source the "
            f"certification covers")

    # --- 2. the host modules, which MUST have moved --------------------------
    host_rows: List[Dict[str, Any]] = []
    host_moved = 0
    for name in TRANCHE_1_HOST_MODULES:
        certified_digest = baseline_hosts.get(f"metal_kernels/{name}")
        now = module_sha256(name)
        changed = certified_digest is not None and now != certified_digest
        host_moved += int(changed)
        host_rows.append({"module": name, "certified": certified_digest,
                          "now": now, "changed": changed})
        log(f"[host] {name:<18} {'MOVED' if changed else 'unchanged'}")

    # "UNCERTIFIED", NOT "NEW", AND THE DISTINCTION IS THE ONE THAT MATTERS HERE.
    # ``HOST_MODULES`` is derived from the package directory while
    # ``TRANCHE_1_HOST_MODULES`` is a four-name literal, so this list contains every
    # module the certified artifact did not hash — including ``__init__.py``, which
    # existed at tranche 1 and was simply outside that literal. Calling those "new
    # spine modules" put a false statement in the artifact, and using their presence
    # as the vacuity guard made the guard DEAD: the list is non-empty for any tree
    # with more than four modules in it, permanently and from now on.
    uncertified = [name for name in metal_launch.HOST_MODULES
                   if name not in TRANCHE_1_HOST_MODULES]
    for name in uncertified:
        host_rows.append({"module": name, "certified": None,
                          "now": module_sha256(name), "changed": None,
                          "note": ("not hashed by the certified artifact; hashed "
                                   "from this round on. Presence here does NOT "
                                   "mean the file is new")})
    log(f"[host] {host_moved}/{len(TRANCHE_1_HOST_MODULES)} certified modules "
        f"moved; {len(uncertified)} modules outside the certified set now hashed: "
        f"{', '.join(uncertified)}")

    neutral = not moved and not missing
    # A run where NO CERTIFIED host module moved is pointed at a tree without the
    # refactor, and this gate has nothing to arbitrate there. The count of modules
    # the certified artifact never hashed cannot stand in for that: it is non-empty
    # by construction and would make the assertion unfailable.
    exercised = host_moved > 0

    payload = {
        "claim": ("the spine extraction is BYTE-NEUTRAL: every certified "
                  "kernel-source sha256 is unchanged, while the host module "
                  "hashes moved — which is a fingerprint event, deliberately"),
        "baseline": CERTIFIED,
        "kernel_sources": {
            "certified_count": len(baseline_kernels),
            "emitted_now_count": len(current_kernels),
            "moved": moved,
            "missing": missing,
            "added": added,
            "unchanged": len(baseline_kernels) - len(moved) - len(missing),
        },
        "host_modules": host_rows,
        "host_modules_moved": host_moved,
        "modules_outside_the_certified_set": uncertified,
        "byte_neutral": neutral,
        "refactor_present": exercised,
        "status": ("passed" if neutral and exercised else
                   "VACUOUS (no CERTIFIED host module moved: the refactor is not "
                   "in this tree)"
                   if neutral else "FAILED (a certified kernel source moved)"),
        "elapsed_s": round(time.time() - started, 1),
    }
    save(payload, arguments.out)

    assert neutral, (
        f"A CERTIFIED KERNEL SOURCE MOVED: {moved or missing}. The 2026-08-14 gate "
        f"result no longer describes the code in the tree, and the refactor must "
        f"be reverted or the family re-certified deliberately")
    assert exercised, (
        f"none of the {len(TRANCHE_1_HOST_MODULES)} CERTIFIED host modules moved, "
        f"so this gate was pointed at a tree where the refactor has not landed — a "
        f"vacuous pass. Modules outside the certified set are hashed for the record "
        f"and cannot substitute for that: that list is non-empty by construction")

    log(f"SPINE BYTE-NEUTRALITY {payload['status']}: "
        f"{payload['kernel_sources']['unchanged']}/{len(baseline_kernels)} "
        f"certified kernel sources UNCHANGED, {host_moved} certified host modules "
        f"moved, {len(uncertified)} modules outside the certified set hashed -> "
        f"{arguments.out}")
    return 0


if __name__ == "__main__":
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))
