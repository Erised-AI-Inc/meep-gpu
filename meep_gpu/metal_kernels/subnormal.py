"""The MPS executor column of the float32 subnormal policy.

:mod:`meep_gpu.subnormal_policy` resolves what this process does to float32
subnormals and drives each executor to it. This module adds the arm for the one
executor that has NO LEVER AT ALL:

* **flush is NATIVE and UNCONTROLLABLE.** Metal's compiler defaults to fast math,
  which includes denormal flushing, and there is no source-level or environment
  override. Both spellings that would ask for IEEE denormals are compile errors on
  this toolchain — the ``denormal`` option is rejected ("invalid option
  'denormal'; expected 'contract', 'reassociate' or 'exceptions'") and the
  vendor-prefixed ``denorm`` spelling likewise.
* **keep is a REFUSAL.** Not a warning, not a silent downgrade: this module's
  parent already holds that an explicit request the host cannot deliver is a
  refusal, and the MPS arm has nothing to install.

THE AWKWARD PART, STATED RATHER THAN HIDDEN. The process-wide default on this
arm64 host RESOLVES TO KEEP — ``match_meep`` measures MEEP's own behaviour, and
MEEP's ``set_zero_subnormals`` is a no-op here because it is guarded by ``#if
HAVE_IMMINTRIN_H``. So the default policy on this machine is exactly the one the
MPS executor cannot honour, and an MPS plan must either be run under an explicit
``flush`` request or be refused at plan time. That is why
``coverage._metal_backend_reasons`` consults :func:`mps_policy_reasons`, and why
the gate and the composition probe both request ``flush`` explicitly and stamp
the resolution into every artifact.

IT IS ALSO WHY THE PRECONDITION IS THE REAL BOUND, not this refusal. Under a
CHECKED subnormal-free precondition, flush and keep are indistinguishable: no
operand, result or intermediate is in the band, so no flush ever fires. The
measured cliff behind that claim, from scaling the whole field state and
re-running the same kernel:

    1e+00  physical        in_subnormal=0     -> IDENTICAL (3188 words moved)
    1e-20  small normal    in_subnormal=0     -> IDENTICAL (3195 moved)
    1e-30  nearer          in_subnormal=0     -> IDENTICAL (3195 moved)
    1e-38  subnormal band  in_subnormal=3233  -> DIVERGES, 6480 words
    1e-40  deep            in_subnormal=3240  -> DIVERGES, 6480 words

Three decades of headroom, then total divergence. That is a CLIFF, not a
tolerance, which is why the claim is byte-identity subject to a checked
precondition rather than a stated tolerance.

THE CENSUS MUST COVER RESULTS, NOT ONLY OPERANDS. In the 1e-38 case the oracle's
OUTPUT subnormal count (6432) is nearly double its INPUT count (3233): the curl
PRODUCES subnormals it was not given. :func:`census` is therefore applied to every
operand and every result, and (for the families that multiply field by field by
field) will have to be applied to every intermediate.

WHAT DOES NOT PORT, and must not be: every mechanical part of the CuPy and Triton
arms. The NVRTC ``-ftz`` strip, the policy-suffixed ``CUPY_CACHE_DIR``, the
CUB/``CUPY_ACCELERATORS`` refusal, the ``nvvm-reflect-ftz`` rewrite, the LLVM
denormal attribute injection and the PTX audit all act on a compiler seam that
does not exist here.

NAMED GAP, WITH NO ANALOGUE: THERE IS NO PTX-EQUIVALENT AUDIT. ``subnormal_policy``
can read every generated instruction and REFUSE THE COMPILE on a policy violation,
which upgrades "complete by accident" to "invariant enforced at compile time".
``torch.mps.compile_shader`` exposes no disassembly, so that upgrade is not
available on Metal: the mutation legs and the byte gate are the only arbiters
here. This is a stated weakness of the Metal certification relative to the Triton
one and belongs in any claim made from it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .. import subnormal_policy

#: The executor name this arm registers under, matching the naming of the
#: ``host`` / ``cupy`` / ``triton`` columns in :mod:`meep_gpu.subnormal_policy`.
MPS_EXECUTOR = "mps"

#: Re-exported so a gate can set the request without importing two modules.
POLICY_ENV = subnormal_policy.POLICY_ENV
FLUSH = subnormal_policy.FLUSH
KEEP = subnormal_policy.KEEP
MATCH_MEEP = subnormal_policy.MATCH_MEEP

#: What this executor CAN be driven to. One entry, and that is the whole point.
ATTAINABLE = (FLUSH,)

#: The op-level exceptions this arm records, in the shape
#: ``subnormal_policy.EXECUTOR_OP_EXCEPTIONS`` uses. To be MERGED into that table
#: when the read-only fence over the shipped modules lifts; kept here meanwhile so
#: the facts have a home rather than living in a commit message.
EXECUTOR_OP_EXCEPTIONS: Dict[str, Dict[str, str]] = {
    "mps_flushed_zero_sign": {
        "executor": MPS_EXECUTOR,
        "ops": "add / mul against a subnormal operand",
        "policies": FLUSH,
        "behaviour": "the flushed zero's SIGN is per-op",
        "why": "Measured on this host: `x + zero_rt` loses the sign of a flushed "
               "negative subnormal while `x * two_rt` keeps it. So a family whose "
               "output can carry a signed zero derived from a subnormal cannot "
               "assume one sign convention across ops, and the signed-zero census "
               "has to be taken per op rather than per kernel.",
    },
    "mps_daz_visible_to_predicates": {
        "executor": MPS_EXECUTOR,
        "ops": "comparison against zero (== 0.0f, < 0.0f)",
        "policies": FLUSH,
        "behaviour": "denormals-are-zero is visible to PREDICATES, not only to "
                     "arithmetic",
        "why": "Measured: `x == 0.0f` is TRUE for a subnormal x, and `x < 0.0f` is "
               "FALSE for a NEGATIVE subnormal x. Any select on a FIELD VALUE must "
               "therefore be checked per family. The two kernels in this tranche "
               "are clear: the ownership mask and the ghost rule branch on INTEGER "
               "indices only. A metal/PEC mask that branched on a material flag "
               "would need the check.",
    },
}


def resolve_mps_policy(policy: Optional[str] = None) -> tuple:
    """``(policy, source)`` for the MPS arm, in the order a DISPATCH can answer.

    FOUR SOURCES, MOST AUTHORITATIVE FIRST, and the third is the one this function
    exists for:

    1. an EXPLICIT ARGUMENT — a caller that already knows what it is asking;
    2. the INSTALLED policy — a measurement of this process's arithmetic, which
       outranks any intention;
    3. the DECLARED policy (:func:`..expansion_refusal.declared_run_policy`) — what
       the surrounding dispatch has stated it will install. A
       :class:`ContradictedRunPolicy` is NOT resolved to either of its members and
       falls through to the resolution below, which is the fail-closed answer the
       contradiction rule asks for;
    4. :func:`subnormal_policy.resolve_policy` — the request or this host's own
       default.

    WHY THE DECLARATION HAD TO BE ADDED HERE. ``coverage._metal_backend_reasons``
    consults :func:`mps_policy_reasons` at every predicate, and a dispatch reaches
    the composer FOUR RUNGS BEFORE it installs a policy — so with only source (4)
    the composer refused every arm on this host, whose ``match_meep`` resolution is
    ``keep`` (MEEP's ``set_zero_subnormals`` being the ``#if HAVE_IMMINTRIN_H``
    no-op on arm64). Measured: with no ``MEEP_GPU_SUBNORMAL_POLICY`` set, the
    shipped composer selected NOTHING on all fourteen driver-route cases, every
    reason being this module's. Reading the declaration is the same move
    ``complex_fields.expansion_policy_reasons`` already makes on the Triton track,
    and it is placed in THIS file rather than in
    ``subnormal_policy.policy_resolution`` deliberately: this module is Metal-only
    and re-gated by the Metal fleet anyway, while ``policy_resolution`` is pinned by
    six Triton ledger welds and a CUDA certification block that would all re-drift
    for a change none of them needs.
    """
    if policy is not None:
        return subnormal_policy.resolve_policy(policy)
    if subnormal_policy.policy_is_installed():
        return subnormal_policy.get_subnormal_policy(), "installed policy"
    from ..expansion_refusal import declared_run_policy  # noqa: PLC0415

    declared = declared_run_policy()
    if isinstance(declared, str):
        return declared, "policy declared by the surrounding dispatch"
    return subnormal_policy.resolve_policy(policy)


def mps_policy_reasons(policy: Optional[str] = None) -> List[str]:
    """Why this process's subnormal policy forbids an MPS launch.

    Empty when the policy is :data:`FLUSH` — the one this executor delivers
    natively. A resolved ``keep`` returns one reason naming both the resolution and
    the absent lever, because a silent downgrade here would mean certifying bytes
    under a policy the device was never in.

    Resolved through :func:`resolve_mps_policy`, which is what lets a dispatch that
    has DECLARED ``flush`` and will install it satisfy this clause four rungs before
    the install.
    """
    try:
        resolved, source = resolve_mps_policy(policy)
    except Exception as exc:  # noqa: BLE001 - an unresolvable policy is a refusal
        return [f"the subnormal policy could not be resolved ({exc!r}); an MPS "
                f"launch is refused rather than run under an unknown policy"]
    if resolved in ATTAINABLE:
        return []
    return [f"the resolved float32 subnormal policy is {resolved!r} (from the "
            f"{source}), and the MPS executor cannot honour it: Metal flushes "
            f"denormals natively and exposes no lever — both denormal pragma "
            f"spellings are compile errors. Request {FLUSH!r} explicitly "
            f"(MEEP_GPU_SUBNORMAL_POLICY={FLUSH}) and certify under the checked "
            f"subnormal-free precondition, or stay on the array path"]


def mps_policy_report(policy: Optional[str] = None) -> Dict[str, Any]:
    """The stamp an artifact carries: what was asked, what it resolved to, and why.

    Every gate and probe artifact embeds this beside the torch and Metal-frontend
    versions, because the claim is only as good as the precondition it was
    certified under.
    """
    record = subnormal_policy.policy_resolution(policy)
    reasons = mps_policy_reasons(policy)
    try:
        in_force, answered_by = resolve_mps_policy(policy)
    except Exception as exc:  # noqa: BLE001 - a stamp never raises over a refusal
        in_force, answered_by = None, f"unresolvable ({exc!r})"
    return {
        "executor": MPS_EXECUTOR,
        "requested": record["requested"],
        "source": record["source"],
        "resolved": record["policy"],
        # WHICH OF THE FOUR SOURCES ANSWERED, recorded because the answer can
        # DIFFER from ``resolved`` above: ``policy_resolution`` reads only the request and
        # the host default, while this arm also honours an install and a dispatch's
        # declaration. An artifact that reported one of them would be describing a
        # different question from the one the admission was decided on.
        "in_force": in_force,
        "in_force_source": answered_by,
        "match_meep_resolution": record["resolution"],
        "attainable": list(ATTAINABLE),
        "admitted": not reasons,
        "reasons": reasons,
        "mechanism": "native (Metal fast-math denormal flushing); no lever exists",
        "ptx_equivalent_audit": None,
        "ptx_equivalent_audit_note":
            "compile_shader exposes no disassembly; the mutation legs and the byte "
            "gate are the only arbiters on this executor",
    }


def install_mps_policy(policy: str, *, strict: bool = True) -> Dict[str, Any]:
    """"Drive" the MPS executor to ``policy`` and MEASURE the result — the executor arm.

    THE SHAPE IS ``subnormal_policy.install_host_policy``'S, because
    :func:`subnormal_policy.install_subnormal_policy` reads the report rather than
    the executor: ``attained`` decides whether the install stands, ``installed``
    decides whether a rollback may drop the process's lock, and ``reasons`` is what
    the refusal quotes.

    THERE IS NOTHING TO INSTALL AND THAT IS THE HONEST REPORT. Metal's compiler
    defaults to fast math, denormal flushing included, and both spellings that would
    ask for IEEE denormals are compile errors on this toolchain (see the module
    docstring). So ``flush`` is ATTAINED with no action and ``installed`` is False —
    no seam was wrapped, nothing has to be put back, and a rollback that consulted
    this executor must not be blocked by it. ``keep`` is a REFUSAL by name rather
    than a silent downgrade, which is this package's rule for an explicit request the
    hardware cannot deliver.
    """
    attained = policy in ATTAINABLE
    reasons: List[str] = [] if attained else [
        f"the {policy!r} float32 subnormal policy is unattainable on the MPS "
        f"executor: Metal flushes denormals natively and exposes no lever — both "
        f"denormal pragma spellings are compile errors on this toolchain. The only "
        f"attainable policy here is {FLUSH!r}"]
    report: Dict[str, Any] = {
        "executor": MPS_EXECUTOR,
        "requested": policy,
        "attained": attained,
        # NOT an install: no compiler seam is wrapped and no process state moves, so
        # a refused sibling's rollback stays free to unlock the policy.
        "installed": False,
        "mechanism": "native (Metal fast-math denormal flushing); no lever exists",
        "attainable": list(ATTAINABLE),
        "reasons": reasons,
        "exceptions": [EXECUTOR_OP_EXCEPTIONS["mps_flushed_zero_sign"],
                       EXECUTOR_OP_EXCEPTIONS["mps_daz_visible_to_predicates"]]
        if attained else [],
        "ptx_equivalent_audit": None,
    }
    subnormal_policy.record_executor_report(MPS_EXECUTOR, report, strict=strict)
    return report


def _words(array: Any) -> Any:
    """The uint32 word view of a float32 OR complex64 volume.

    COMPLEX64 IS TWO WORDS PER CELL AND BOTH ARE COUNTED. Casting a complex volume
    with ``dtype=np.float32`` silently DISCARDS the imaginary plane — measured while
    the complex tranche's composition probe was being written, where it made the
    census a half-census that would have reported a clean window over data it had
    never looked at. A census that quietly halves its own denominator is worse than
    no census, because it still reports a number.
    """
    import numpy as np  # noqa: PLC0415

    contiguous = np.ascontiguousarray(array)
    if contiguous.dtype == np.complex64:
        return contiguous.reshape(-1).view(np.uint32)
    if contiguous.dtype == np.complex128:
        raise TypeError(
            "complex128 has no float32 word view; this executor steps complex64 "
            "only, and censusing a double volume would report a band that does not "
            "apply to it")
    return contiguous.astype(np.float32, copy=False).reshape(-1).view(np.uint32)


def census(array: Any) -> int:
    """How many float32 words of ``array`` are in the subnormal band.

    Exponent field zero and mantissa non-zero, read off the BITS: a value
    comparison would be answered by the very flushing this counts. Zeros (both
    signs) are not subnormal and are not counted.
    """
    import numpy as np  # noqa: PLC0415

    words = _words(array)
    return int(np.count_nonzero(((words & np.uint32(0x7F800000)) == 0)
                                & ((words & np.uint32(0x007FFFFF)) != 0)))


def signed_zero_census(array: Any) -> Dict[str, int]:
    """How many exact ``-0.0`` and ``+0.0`` words ``array`` carries.

    A census of ZERO is VACUOUS, not passed: it means the case never constructed
    the class it claims to cover. The gate asserts a floor per seeding.
    """
    import numpy as np  # noqa: PLC0415

    words = _words(array)
    return {"negative_zero": int(np.count_nonzero(words == np.uint32(0x80000000))),
            "positive_zero": int(np.count_nonzero(words == np.uint32(0x00000000)))}
