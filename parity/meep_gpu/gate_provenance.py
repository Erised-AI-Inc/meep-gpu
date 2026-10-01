"""What bytes did this gate process actually import?

WHY THIS EXISTS. A device gate's verdict is only worth what the record says it
covers, and until now every weld's ``source_sha256`` was transcribed by hand from
a staged tree chosen by eye. Measured 2026-08-17: the GPU host holds 67 staged trees
whose kernel modules differ, the digest for ``no_pml_stored_e.py`` was taken from
the wrong one, and the weld test caught it only because the recorded digest
happened to differ from the local file. Where a wrong tree HAPPENS to match the
local bytes, that error is invisible — the weld test compares recorded against
local, and a mis-transcription that lands on the same value passes forever.

So the transcription step is removed rather than performed more carefully. This
reads ``module.__file__`` for modules the interpreter has ALREADY IMPORTED, which
is the one description of "the bytes that ran" that cannot be inferred wrongly
from a path, a cwd, a directory name, or a run's mtime.

WHAT IT DOES NOT DO. It does not prove the kernel LAUNCHED — that is the launch
counters' job, and the vacuous pass they catch is a different failure. It records
which source produced the artifact, and — since 2026-09-30 — WHICH DEVICE ran it
(:data:`DEVICE_KEY`), nothing more.

WHY THE DEVICE IS STAMPED HERE, in the one place every track's save() already
passes through, rather than in each gate. A weld's claim is about generated code
for an ARCHITECTURE, so a per-capability certification record can only be keyed
off a device the RUN recorded. Measured over the cited CUDA welds: 14 of them are
bound to ``gate_cuda_fused_complex_pairs.py``, which writes no device of any kind
(nor do ``gate_cuda_complex_offdiag_update_e.py`` and
``gate_cuda_folded_offdiag_rowmask.py``), so their capability was not readable
from their own artifacts and was being pooled out of OTHER families' legs by the
campaign recorder's directory-wide survey — a weld inheriting a device it never
named. One stamping point closes every such gate at once, whether or not it grows
an ``environment`` block of its own, and ``rebind_cuda_welds.py`` reads it.

WHAT THAT EDIT COSTS. This file is pinned by 2 Triton welds
(``triton_dispersive_fused_pair_device_gate``, ``triton_fused_ade_state_device_gate``)
and 3 Metal ones (``metal_complex_conductive_fused_pair_device_gate``,
``metal_complex_no_pml_conductive_device_gate``,
``metal_cylindrical_real_fused_magnetic_pair_device_gate``), so every edit here
moves those five welds' bytes and owes them a re-gate. That is why the earlier
Triton-only need went into ``triton_device_identity.py`` instead; this one is
owed a re-gate on all three tracks anyway, which is what makes the shared home
affordable now.

Stdlib only, no NumPy, no CuPy, no Triton: it must answer inside a gate that has
already refused, and on the laptop that is the merge bar. The device read keeps
that rule by going through ``sys.modules`` — see :func:`device`.

Usage, one line before the artifact is written::

    from gate_provenance import provenance
    payload["source_sha256"] = provenance()
"""

from __future__ import annotations

import hashlib
import os
import sys
from typing import Any, Dict, Optional, Sequence

#: Repo subtrees whose modules are worth recording. A gate imports plenty that is
#: not ours (numpy, cupy, triton); those carry their own versions in the policy
#: stamp and are not what a weld binds.
DEFAULT_ROOTS: Sequence[str] = ("meep_gpu", "parity/meep_gpu")

#: Where the imported-module digests land. Deliberately NOT "source_sha256":
#: gates that curate their own binding list already own that name.
IMPORTED_KEY = "imported_source_sha256"

#: Where the live device lands. A THIRD DISTINCT KEY, and both of the names it is
#: not are names a gate already owns:
#:
#: * NOT ``environment``. Most CUDA gates write ``probe.device_info()`` there, and
#:   ``record_cuda_regate._stamps`` pools an eight-key signature of that block over
#:   every payload in a campaign and REFUSES the campaign when two signatures
#:   appear. A partial block written here would be a second signature, so stamping
#:   into ``environment`` would make exactly the mixed campaigns this stamp exists
#:   to serve unrecordable.
#: * NOT ``provenance``. Two tracks already use that name for a curated source
#:   digest map (``metal_gate_kit.provenance``), and the Triton folded gates write
#:   a PATH STRING there, so there is no shape to merge a device into.
#:
#: The leaf names match ``probe_fused_kernel_bit_identity.device_info()`` so that a
#: reader of either home reads the same keys.
DEVICE_KEY = "stamped_device"

#: the repository root — the directory the weld's repo-relative keys are measured from, so
#: the keys here are byte-identical to the ones already in fingerprints.json.
#: THREE levels: this file -> parity/meep_gpu -> parity -> the repository root. Two levels
#: lands on parity/, where the real kernel tree is invisible and this very file
#: keys as "meep_gpu/gate_provenance.py" — a parity file wearing a kernel path.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _repo_relative(path: str) -> Optional[str]:
    """Repo-relative POSIX key for a file inside the repository root, else None.

    Resolved through realpath on BOTH sides: a staged tree reached by a symlink
    would otherwise key differently from the same file reached directly, and the
    weld would compare two spellings of one path.
    """
    try:
        real = os.path.realpath(path)
        root = os.path.realpath(REPO_ROOT)
    except OSError:
        return None
    if not real.startswith(root + os.sep):
        return None
    return os.path.relpath(real, root).replace(os.sep, "/")


def provenance(roots: Sequence[str] = DEFAULT_ROOTS) -> Dict[str, str]:
    """sha256 of every already-imported module under ``roots``, keyed repo-relative.

    Enumerated from ``sys.modules`` rather than from an explicit per-gate list.
    A hand-maintained list is a second place to forget a file, and a weld that
    silently stops covering a module it used to cover is exactly the drift this
    file exists to end. What the process imported IS the list.

    A module whose file cannot be read is OMITTED rather than recorded as an
    error string: a consumer comparing digests must never be handed a value that
    looks like one and is not.
    """
    out: Dict[str, str] = {}
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if not path or not path.endswith(".py"):
            continue
        key = _repo_relative(path)
        if key is None or not any(key.startswith(r + "/") or key == r for r in roots):
            continue
        try:
            with open(os.path.realpath(path), "rb") as handle:
                out[key] = hashlib.sha256(handle.read()).hexdigest()
        except OSError:
            continue
    return out


def device() -> Dict[str, Any]:
    """WHICH GPU this process has open: its name, compute capability and index.

    ``sys.modules.get("cupy")`` RATHER THAN ``import cupy``, for this file's
    standing reason: it is called from inside a gate that may already have refused,
    and on the laptop that is the merge bar. The lookup returns the very module the
    gate has been launching through — so the answer describes the device the run
    actually used — and on a host with no CuPy it returns ``None`` without importing
    anything, allocating anything or touching a device's state.

    BOTH THE INDEX AND THE MASK are recorded because they are different facts:
    ``CUDA_VISIBLE_DEVICES`` names which devices the process can see, and
    ``getDevice()`` is the ordinal INSIDE that mask, so neither alone says which
    card ran. Same pair, same reason, as ``triton_device_identity.device_identity``.

    NEVER RAISES. An unreadable device is recorded under ``unreadable`` and the
    artifact is still written: losing a completed device run to an identity read
    would be the expensive failure, and a tool that cannot read a capability here
    refuses by name (``rebind_cuda_welds``) rather than guessing one.
    """
    cupy = sys.modules.get("cupy")
    if cupy is None:
        return {"unreadable": "cupy is not imported in this process, so no CUDA "
                              "device was open when this payload was written"}
    try:
        index = int(cupy.cuda.runtime.getDevice())
        properties = cupy.cuda.runtime.getDeviceProperties(index)
        name = properties["name"]
        return {
            "device_name": name.decode() if isinstance(name, bytes) else str(name),
            "compute_capability": f"{properties['major']}.{properties['minor']}",
            "device_index": index,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
    except Exception as exc:  # noqa: BLE001 - an unreadable device is recorded, not raised
        return {"unreadable": repr(exc)}


def stamp(payload: Dict[str, Any], roots: Sequence[str] = DEFAULT_ROOTS
          ) -> Dict[str, Any]:
    """Record the imported bytes, the live device and the verdict; hand back payload.

    Three facts, three keys: :data:`IMPORTED_KEY`, :data:`DEVICE_KEY` and
    :data:`VERDICT_KEY`. Each is written where nothing else writes, and each has its
    own merge rule below, because what "re-stamping" means differs per fact.

    A DISTINCT KEY, not ``source_sha256``. Several gates already build a
    ``source_sha256`` of their own — a curated list of the files that gate means
    to bind — and that is a DIFFERENT FACT from "every repo module this process
    imported". Measured 2026-08-19: writing to the shared name made
    gate_triton_complex_no_pml_stored_e and gate_triton_folded_offdiag raise on a
    collision and die before writing any verdict at all, turning a working gate
    into ``release: null``. Two facts, two keys, no collision.

    Re-stamping with an identical value is allowed (a gate may serialise one
    payload more than once); a CHANGED value still raises, because two writers
    disagreeing about one key is how a record starts describing a run that never
    happened.
    """
    fresh = provenance(roots)
    existing = payload.get(IMPORTED_KEY) or {}

    # THE SET GROWS DURING A RUN, AND THAT IS NOT A CONTRADICTION. Gates call
    # save() more than once — folded_offdiag saves inside run_license, mid-run —
    # and Python imports more modules as later legs execute, so a second call
    # legitimately sees MORE files than the first. Measured 2026-08-19: a
    # whole-dict equality check read that growth as two writers disagreeing and
    # raised, killing both gates before they wrote any verdict (release: null on
    # gates that were working).
    #
    # What must never change is a digest for a path already recorded: that means
    # a source file was rewritten WHILE the gate ran, and every measurement taken
    # on either side of the rewrite describes a different program.
    changed = {path: (existing[path], fresh[path]) for path in existing
               if path in fresh and existing[path] != fresh[path]}
    if changed:
        first = sorted(changed)[0]
        raise RuntimeError(
            f"{first} changed DURING this run ({changed[first][0][:16]}... -> "
            f"{changed[first][1][:16]}...): measurements taken on either side of "
            f"a mid-run rewrite describe different programs. {len(changed)} "
            f"file(s) affected.")

    merged = dict(existing)
    merged.update(fresh)
    payload[IMPORTED_KEY] = merged

    # THE DEVICE, read live. The FIRST READABLE answer is kept: a gate saves more
    # than once, and a later save inside a gate that has since refused can fail a
    # read the first one made — so an unreadable record is retried and a readable
    # one is never replaced. A CHANGED device raises for the same reason a changed
    # digest does: the measurements on either side of the move were taken on
    # different hardware, and a weld is a claim about generated code for ONE
    # architecture. Only the card and the capability are compared; the ordinal
    # inside the mask is not a different device.
    recorded = payload.get(DEVICE_KEY)
    fresh_device = device()
    if not isinstance(recorded, dict) or not recorded.get("compute_capability"):
        payload[DEVICE_KEY] = fresh_device
    else:
        moved = {key: (recorded.get(key), fresh_device[key])
                 for key in ("device_name", "compute_capability")
                 if key in fresh_device and fresh_device[key] != recorded.get(key)}
        if moved:
            raise RuntimeError(
                f"the device changed DURING this run ({moved}): every measurement "
                f"taken on either side of the move describes a different "
                f"architecture, and this payload can only be a claim about one.")

    # THE VERDICT, normalised beside the gate's own spelling. Overwritten freely
    # rather than merged: a payload legitimately moves from unreadable to
    # released as the run completes, and gates save progressively, so the LAST
    # write is the one that describes the finished run.
    payload[VERDICT_KEY] = read_verdict(payload)
    return payload

#: Where the one readable verdict lands, beside whatever shape the gate already
#: wrote. Never REPLACES a gate's own key — that key is the gate's authored
#: record and other tooling reads it.
VERDICT_KEY = "canonical_verdict"

#: Every shape a gate has been measured to write its outcome in, most explicit
#: first. Measured across the 23 Triton gates on 2026-08-19: 8 write
#: ``release`` (a dict), 3 write ``passed``, 8 write ``status``, and 4 write
#: none of those — of which fused_electric buries it at ``summary.pass``.
#: FIVE spellings of one fact.
_PASS_WORDS = {"pass", "passed", "released", "ok"}
_FAIL_WORDS = {"fail", "failed", "refused", "rejected"}


def read_verdict(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Did this gate release? One reading, from whichever shape wrote it.

    WHY THIS EXISTS. A gate's verdict was only recoverable by knowing that
    gate's spelling, so "did it release?" could not be answered mechanically
    across the fleet. That is not a cosmetic problem: measured twice on
    2026-08-19, a reader that knew only ``release.released`` reported
    gate_triton_folded_offdiag as ``released=None`` when it had in fact PASSED
    (it writes ``passed``), and a weld count keyed on ``status == "PASS"``
    silently skipped the three fused records that use neither.

    UNREADABLE IS NOT REFUSED. When no shape is present ``released`` is None,
    never False. A gate whose verdict cannot be read has not failed — it has not
    been read — and collapsing those two is how a green fleet acquires a silent
    hole. Same rule the refusal legs already follow.
    """
    def _reasons(source: Any) -> list:
        if isinstance(source, dict):
            got = source.get("reasons")
            if isinstance(got, (list, tuple)):
                return [str(x) for x in got]
        return []

    release = payload.get("release")
    if isinstance(release, dict) and "released" in release:
        return {"released": bool(release["released"]),
                "reasons": _reasons(release), "read_from": "release.released"}

    for key in ("released", "passed"):
        value = payload.get(key)
        if isinstance(value, bool):
            return {"released": value, "reasons": _reasons(payload),
                    "read_from": key}

    status = payload.get("status")
    if isinstance(status, str):
        word = status.strip().lower()
        if word in _PASS_WORDS:
            return {"released": True, "reasons": [], "read_from": "status"}
        if word in _FAIL_WORDS:
            return {"released": False, "reasons": _reasons(payload),
                    "read_from": "status"}

    # NESTED shapes. The Metal gates put the outcome inside ``summary`` under
    # names the top level never uses: measured 2026-08-19, gate_metal_symmetry
    # writes summary.certified=True AND summary.status='passed', while
    # gate_metal_cylindrical_real writes summary.certified only, and the Triton
    # fused_electric gate writes summary.pass. That is three more spellings on
    # top of the four at top level — SEVEN in total for one fact.
    summary = payload.get("summary")
    if isinstance(summary, dict):
        for key in ("pass", "passed", "released", "certified"):
            value = summary.get(key)
            if isinstance(value, bool):
                return {"released": value, "reasons": _reasons(summary),
                        "read_from": "summary." + key}
        nested = summary.get("status")
        if isinstance(nested, str):
            word = nested.strip().lower()
            if word in _PASS_WORDS:
                return {"released": True, "reasons": _reasons(summary),
                        "read_from": "summary.status"}
            if word in _FAIL_WORDS:
                return {"released": False, "reasons": _reasons(summary),
                        "read_from": "summary.status"}

    # AN EXPLICIT TOP-LEVEL CLAIM. The CUDA off-diagonal gate writes
    # ``certifies: True``. Read before the prose rule because it is a bool.
    certifies = payload.get("certifies")
    if isinstance(certifies, bool):
        return {"released": certifies, "reasons": _reasons(payload),
                "read_from": "certifies"}

    # A NESTED VERDICT OBJECT. gate_triton_cylindrical writes ``verdict`` as a
    # DICT carrying ``pass``, not as prose, so the prose rule below skipped it.
    nested_verdict = payload.get("verdict")
    if isinstance(nested_verdict, dict):
        for key in ("pass", "passed", "released", "certified"):
            value = nested_verdict.get(key)
            if isinstance(value, bool):
                return {"released": value, "reasons": _reasons(nested_verdict),
                        "read_from": "verdict." + key}

    # A VALIDATION BLOCK. gate_triton_no_pml writes ``validation.status``.
    validation = payload.get("validation")
    if isinstance(validation, dict):
        word = str(validation.get("status", "")).strip().lower()
        if word in _PASS_WORDS:
            return {"released": True, "reasons": _reasons(validation),
                    "read_from": "validation.status"}
        if word in _FAIL_WORDS:
            return {"released": False, "reasons": _reasons(validation),
                    "read_from": "validation.status"}

    # A TOP-LEVEL AGGREGATE FLAG. gate_triton_no_pml_constitutive writes ``all_ok``.
    for key in ("all_ok", "ok"):
        value = payload.get(key)
        if isinstance(value, bool):
            return {"released": value, "reasons": _reasons(payload),
                    "read_from": key}

    # A PROSE VERDICT. Four Metal verdict.json files put a sentence at
    # ``verdict``: "PASSED", or "CERTIFIED within the gate's declared scope...".
    # ONLY THE FIRST WORD IS READ, matched exactly. Prose is not a schema, and
    # substring matching here would read "NOT CERTIFIED" as certified — so
    # anything whose first word is not in the vocabulary stays UNREADABLE rather
    # than being guessed at.
    prose = payload.get("verdict")
    if isinstance(prose, str) and prose.strip():
        first = prose.strip().split()[0].strip(":,.").lower()
        if first in _PASS_WORDS or first == "certified":
            return {"released": True, "reasons": _reasons(payload),
                    "read_from": "verdict(prose)"}
        if first in _FAIL_WORDS or first in {"not", "uncertified"}:
            return {"released": False, "reasons": [prose.strip()[:200]],
                    "read_from": "verdict(prose)"}

    # A COMPLETED RUN WITH AN EMPTY FAILURE LIST. Used by the subnormal-policy
    # conformance gates. Both halves are required: ``failures == []`` alone could
    # be a run that has not reached anything yet, and this file must not turn an
    # unfinished run into a pass. With ``finished_utc`` present the run declared
    # itself complete, so an empty list is a verdict rather than a silence.
    failures = payload.get("failures")
    if isinstance(failures, list) and payload.get("finished_utc"):
        return {"released": not failures,
                "reasons": [str(x) for x in failures],
                "read_from": "failures+finished_utc"}

    return {"released": None, "reasons": [],
            "read_from": None,
            "unreadable": "no recognised verdict shape in this payload; "
                          "UNREADABLE is not the same claim as refused"}
