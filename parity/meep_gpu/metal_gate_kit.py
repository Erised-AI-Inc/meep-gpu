"""Gate scaffolding shared by the Metal family byte gates.

EXTRACTED BY COPY FROM ``gate_metal_pml.py``, AND THAT GATE IS NOT RE-POINTED AT IT.
The trade is stated rather than assumed:

* **cost** — roughly four hundred duplicated lines that will drift;
* **benefit** — tranche 1's certification is not reopened as a side effect of a
  buildout. ``gate_metal_pml.py`` is a certified artifact whose nine legs ran on
  2026-08-14; rewriting its plumbing to import this module would make the next
  green run a statement about NEW code. A follow-on round may re-point it and
  re-cut, as a deliberate re-certification named as such.

WHAT IS HERE is the part that is genuinely family-independent: the atomic writer,
the flushed log, the uint32 word compare, the state digest, the launch counter, the
mutation harness with its armed / launch-counted / DISARMED / NEEDLE-MISSED
classification, the vacuity floors, the provenance and fingerprint stamp, and the
``--legs`` driver with the exit-75 cannot-certify-here convention.

WHAT IS NOT HERE, deliberately: anything that decides what a family COMPARES. The
compared set, the seeding, the case matrix and the reference transcription are the
family's own and are the substance of its claim; a kit that supplied them would let
a family inherit a claim it never made.

CASE DISCIPLINE THIS KIT ENFORCES RATHER THAN SUGGESTS:

* uint32 word equality, never ``allclose`` — :func:`differing`;
* a mutation that never LAUNCHED is DISARMED and FAILS, and one whose transform
  matched nothing is NEEDLE-MISSED and FAILS — :class:`MutationHarness`;
* a leg that certifies "the kernel reproduced the array path" must first prove the
  step MOVED STATE, because a no-op agreeing with a no-op is trivially identical —
  :func:`assert_moved`;
* a census of zero is VACUOUS, not passed — :func:`assert_census_floor`;
* the top-level verdict reads the OUTCOME, not the count. ``compared`` is how many
  comparisons ran; ``certified`` is whether they agreed. The Triton track shipped a
  round where every compare failed and the artifact still stamped ``passed: true``
  and exited 0, because the verdict read the count. :func:`summarize` reads both and
  refuses to stamp a pass on zero comparisons or on any failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
API_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
if API_ROOT not in sys.path:
    sys.path.insert(0, API_ROOT)

#: The exit code for "this host cannot certify this claim" — a missing GPU, a torch
#: without MPS, a policy the executor cannot honour. DISTINCT FROM 1, because "the
#: gate could not run" and "the gate ran and the bytes differ" are different facts
#: and a CI that conflated them would read an unrunnable gate as a passing one.
EXIT_CANNOT_CERTIFY = 75


# ---------------------------------------------------------------------------
# Artifact plumbing
# ---------------------------------------------------------------------------

def log(message: str) -> None:
    """One flushed line per case (the progress-reporting rule): ``tail`` is the status check."""
    print(message, flush=True)


def save(payload: Dict[str, Any], path: str) -> None:
    """Atomic rewrite: tmp + fsync + os.replace, after EVERY case.

    An interrupted run keeps everything up to the failure, which is the difference
    between a gate that can be resumed and one that has to be re-run from zero.
    """
    # The bytes THIS process imported, recorded into the artifact before it is
    # serialised. ONE helper for both backends: the Triton gates stamp through the
    # same gate_provenance, so an artifact from either track answers "which source
    # produced this?" in the same key and one verifier reads both.
    from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
    _stamp_provenance(payload)

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256_of(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def provenance(out_dir: str, files: Dict[str, str],
               kernel_sources: Optional[Dict[str, str]] = None,
               name: str = "provenance.json") -> Dict[str, str]:
    """sha256 of every source whose bytes decide what this gate certified.

    ``kernel_sources`` is the family's own ``enumerate_*_sources()`` output, hashed
    beside the module files because a kernel source is a GENERATED string: the
    module hash alone would not notice a specialisation whose emitted text changed.

    ``name`` EXISTS BECAUSE THE DEFAULT SILENTLY DESTROYED EVIDENCE. A family's gate
    and its composition probe write into ONE results directory, and both called this
    with the default file name — so the probe, which runs second and passes no
    ``kernel_sources``, replaced the gate's provenance wholesale. Measured on
    ``results/metal_offdiag_2026-08-15/`` (2026-08-15): the surviving file held the
    probe's six module hashes and NO ``kernel_sources`` block at all, so the
    per-specialisation hashes of the sources the gate actually launched — the record
    the family's own docstring says the gate keeps beside the corpus digest — were
    gone, and the loss was invisible because both artifacts still stamped
    ``passed``. Two writers, two names; the caller says which.
    """
    record = {label: sha256_of(path) for label, path in files.items()}
    payload: Dict[str, Any] = {"sources": record}
    if kernel_sources is not None:
        payload["kernel_sources"] = {
            label: hashlib.sha256(source.encode("utf-8")).hexdigest()
            for label, source in kernel_sources.items()}
    save(payload, os.path.join(out_dir, name))
    return record


# ---------------------------------------------------------------------------
# Comparison — words, never tolerances
# ---------------------------------------------------------------------------

def words(array: Any) -> Any:
    """The uint32 word view of a float32 or complex64 volume.

    complex64 is bit-layout (re, im) interleaved, so a complex volume is two words
    per cell and both are compared. A gate that compared magnitudes here would be
    unable to see a signed-zero divergence at all, which is the class these families
    are most likely to get wrong.
    """
    contiguous = np.ascontiguousarray(array)
    if contiguous.dtype == np.complex64:
        return contiguous.reshape(-1).view(np.uint32)
    return contiguous.astype(np.float32, copy=False).reshape(-1).view(np.uint32)


def differing(left: Any, right: Any) -> int:
    """How many words disagree. uint32 equality, NEVER ``allclose``."""
    return int(np.count_nonzero(words(left) != words(right)))


def state_digest(state: Dict[str, Any]) -> str:
    """A stable digest of a whole named-volume state, for artifact rows."""
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(words(state[name]).tobytes())
    return digest.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Vacuity floors
# ---------------------------------------------------------------------------

def assert_moved(moved: int, context: str, floor: int = 1) -> None:
    """Every leg must prove the step MOVED STATE.

    A no-op agreeing with a no-op is trivially identical. This is the assertion that
    turns "the bytes matched" into "the bytes matched and something happened", and
    it is why the constitutive legs need a seeding other than zero init — zero init
    is a FIXED POINT of that sub-step and would move nothing at all.
    """
    assert moved >= floor, (
        f"VACUOUS: {context} moved {moved} words (floor {floor}). A no-op agreeing "
        f"with a no-op is trivially identical and certifies nothing")


def assert_census_floor(count: int, context: str, floor: int = 1) -> None:
    """A census of zero is VACUOUS, not passed: the class was never constructed."""
    assert count >= floor, (
        f"VACUOUS census on {context}: {count} (floor {floor}). A census of zero "
        f"means the case never constructed the class it claims to cover")


# ---------------------------------------------------------------------------
# The mutation harness
# ---------------------------------------------------------------------------

class Counter:
    """A launch-counted wrapper — the DISARMED classification's evidence.

    A mutation leg whose kernel never launched reports its defect as uncaught. The
    count is what makes "the mutant ran and the bytes still matched" a different
    statement from "nothing happened".
    """

    __slots__ = ("function", "launches")

    def __init__(self, function: Any) -> None:
        self.function = function
        self.launches = 0

    def __call__(self, *args: Any) -> Any:
        self.launches += 1
        return self.function(*args)


def needle(source: str, old: str, new: str) -> str:
    """Plant one defect, refusing a transform that matched nothing.

    ``LookupError`` here becomes NEEDLE-MISSED upstream. A transform that silently
    matched nothing would report its defect as UNCAUGHT, which is the single most
    dangerous false pass a mutation leg can produce.
    """
    if old not in source:
        raise LookupError(old)
    return source.replace(old, new)


class MutationHarness:
    """Armed, launch-counted, three-valued mutation accounting.

    ``must_catch``: ``True`` (must be caught) | ``None`` (record only) | ``False``
    (must NOT be caught — the arm that proves a spelling is genuinely equivalent
    rather than merely untested).

    VERDICTS:

    * ``NEEDLE-MISSED`` — the transform matched nothing, so nothing was tested. FAILS;
    * ``DISARMED`` — the mutant compiled but never launched. FAILS: a leg that
      certifies a defect it never ran is a hollow pass;
    * ``CAUGHT c/r`` — ``r`` cases ran and ``c`` diverged from the oracle.
    """

    __slots__ = ("rows", "payload", "out", "key", "started")

    def __init__(self, payload: Dict[str, Any], out: str,
                 key: str = "mutations") -> None:
        self.rows: List[Dict[str, Any]] = []
        self.payload = payload
        self.out = out
        self.key = key
        self.started = time.time()

    @staticmethod
    def verdict(missed: bool, ran: int, launches: int, caught: int) -> str:
        """``missed`` is ANY miss, not only a total one.

        THIS USED TO READ ``missed and ran == 0`` AND THAT WAS A FALSE-PASS PATH.
        A mutation swept over two sub-steps (or two sides) whose needle matched in
        only one of them left ``missed=True`` and ``ran=1``, and the verdict came
        back ``CAUGHT 1/1`` — a pass, with the half that was never planted silently
        dropped. That is exactly the failure NEEDLE-MISSED exists to prevent, only
        harder to see, because the row still carries a plausible count. A partial
        miss is a miss: the defect the label names was not planted everywhere the
        entry claimed, so the entry's own scope is what failed.

        Audited 2026-08-15 across the Metal gates: no entry currently misses
        partially, so this is a strictly stricter guard on a green tree rather than
        a behaviour change to any recorded verdict.
        """
        if missed:
            return "NEEDLE-MISSED"
        if launches == 0:
            return "DISARMED"
        return f"CAUGHT {caught}/{ran}"

    def record(self, label: str, verdict: str, launches: int, caught: int, ran: int,
               must_catch: Optional[bool], why: str,
               extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        row = {"mutation": label, "verdict": verdict, "launches": launches,
               "caught": caught, "ran": ran, "must_catch": must_catch, "why": why}
        row.update(extra or {})
        self.rows.append(row)
        log(f"[mutation] {label:<40} launches={launches} {verdict} "
            f"({time.time() - self.started:.1f}s)")
        self.payload["legs"][self.key] = self.rows
        save(self.payload, self.out)
        assert verdict != "DISARMED", (
            f"{label} never launched: a mutation leg that certifies a defect it did "
            f"not run is a hollow pass")
        assert verdict != "NEEDLE-MISSED", (
            f"{label}: the transform matched nothing in the shipped source, so the "
            f"defect it names was never planted")
        if must_catch is True:
            assert caught == ran and ran > 0, f"{label} was not caught: {row}"
        elif must_catch is False:
            assert caught == 0, f"{label} was caught but must not be: {row}"
        return row


# ---------------------------------------------------------------------------
# Predicted nulls
# ---------------------------------------------------------------------------

def predicted_null(row: Dict[str, Any], reason: str) -> Dict[str, Any]:
    """Record a case that is EXPECTED to measure nothing, WITH its reason.

    A predicted null quietly dropped is indistinguishable from a case that was
    forgotten. Recorded with its reason, it is a claim a reader can check.
    """
    row["predicted_null"] = True
    row["predicted_null_reason"] = reason
    return row


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------

def environment_stamp() -> Dict[str, Any]:
    """Everything about this host that could change the answer."""
    from meep_gpu.metal_kernels import launch as metal_launch  # noqa: PLC0415

    record: Dict[str, Any] = {
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "metal_frontend": metal_launch.metal_frontend_version(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        import torch  # noqa: PLC0415

        record["torch"] = str(torch.__version__)
        record["mps_available"] = bool(torch.backends.mps.is_available())
    except Exception as exc:  # noqa: BLE001 - reported, never guessed
        record["torch"] = None
        record["mps_available"] = False
        record["torch_error"] = repr(exc)
    return record


def cannot_certify(payload: Dict[str, Any], out: str, reasons: Sequence[str]) -> int:
    """Stamp the artifact and return :data:`EXIT_CANNOT_CERTIFY`."""
    payload["summary"] = {"status": "cannot-certify-here", "reasons": list(reasons)}
    save(payload, out)
    for reason in reasons:
        log(f"[cannot-certify] {reason}")
    return EXIT_CANNOT_CERTIFY


def summarize(payload: Dict[str, Any], out: str, claim: str, scope: str,
              stated_weakness: str, started: float,
              legs_run: Sequence[str], compared: int, certified: bool,
              extra: Optional[Dict[str, Any]] = None) -> int:
    """The top-level verdict, which reads the OUTCOME and the COUNT.

    ``compared`` is how many comparisons ran and ``certified`` is whether they all
    agreed. BOTH are required: a gate that stamps ``passed`` from a count alone will
    pass a round in which every comparison failed (the Triton track shipped exactly
    that), and a gate that stamps it from an outcome alone will pass a round in
    which nothing was compared.
    """
    ok = bool(certified) and compared > 0
    payload["summary"] = {
        "status": "passed" if ok else "FAILED",
        "legs_run": list(legs_run),
        "comparisons": compared,
        "certified": bool(certified),
        "claim": claim,
        "scope": scope,
        "stated_weakness": stated_weakness,
        "elapsed_s": round(time.time() - started, 1),
    }
    payload["summary"].update(extra or {})
    save(payload, out)
    log(f"{'PASSED' if ok else 'FAILED'}: {compared} comparisons, "
        f"certified={certified}, {payload['summary']['elapsed_s']}s -> {out}")
    return 0 if ok else 1


def run_legs(legs: Sequence[Tuple[str, Callable[[Dict[str, Any], str], None]]],
             payload: Dict[str, Any], out: str, wanted: Sequence[str]) -> List[str]:
    """Run the requested subset, one banner per leg. Returns the legs that ran."""
    ran: List[str] = []
    for name, leg in legs:
        if wanted and name not in wanted:
            log(f"[skip] leg {name} (not in --legs)")
            continue
        log(f"=== LEG {name} ===")
        leg(payload, out)
        ran.append(name)
    return ran


def argument_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--out", required=True, help="artifact JSON path")
    parser.add_argument("--legs", default="", help="comma-separated leg subset")
    return parser


def wanted_legs(value: str) -> Tuple[str, ...]:
    return tuple(name.strip() for name in value.split(",") if name.strip())
