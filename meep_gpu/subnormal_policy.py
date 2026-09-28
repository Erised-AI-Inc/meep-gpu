"""One process-wide float32 subnormal policy, obeyed by every executor.

WHAT THIS IS. Three independent things in this process decide, separately, whether
a float32 subnormal survives an arithmetic operation: the host FPU (MEEP sets the
x86 FTZ and DAZ bits when it initializes), CuPy's device binaries (CuPy 13.5.1
appends ``-ftz=true`` to every NVRTC compile, ``cupy/cuda/compiler.py:552``), and
Triton's device binaries (Triton 3.1.0 emits plain IEEE ``mul.rn.f32`` and keeps —
except through libdevice, which it routes to the ``.ftz`` variants unconditionally).
Left alone they do not agree, and a half-applied policy is worse than either
policy applied uniformly: measured on the bfast composition, 7/7 cases and 80/80
steps are bit-identical under uniform keep, and diverge at step 1 with 92 differing
floats of 2880 across 22 arrays when only Triton is flipped. This module is the
single place that decides, and it drives all three from that one decision.

THE DEFAULT IS "WHAT MEEP ITSELF DOES ON THIS HOST", AND IT IS MEASURED RATHER
THAN TABULATED. The default request is :data:`MATCH_MEEP`, and
:func:`resolve_match_meep` answers it the same way every other verdict in this
module is reached — by reading the machine. MEEP decides its own subnormal
behaviour at import: ``initialize::initialize()`` calls ``setup()`` calls
``set_zero_subnormals(true)`` (``src/mympi.cpp:188``, call site ``:219``), and
``set_zero_subnormals(false)`` appears nowhere in MEEP. MEEP's own comment says it
is a SPEED choice, for "the tails of exponentially decaying sources" (meep#1708).
That call is compiled under ``#if HAVE_IMMINTRIN_H`` — x86 SSE — so it moves the
FPU on x86 and is a NO-OP elsewhere. Which is exactly why the question needs no
platform table and no mutation to answer: with MEEP imported, this process's FPU
IS what MEEP left, and ``backends.subnormals_flushed()`` reads it. Measured on
this package's own dev laptop (arm64, MEEP 1.33.0): flushing False before
``import meep``, False after, and False again after ``mp.set_zero_subnormals(True)``
— so the default resolves to :data:`KEEP` there, because the host was measured
and not because the machine string was recognized.

A simulation lifted onto this engine is compared against a MEEP run on an x86
cluster, where that same measurement reads flushing — so the default is the
configuration the comparison is against, on either host. MEEP on arm64 and MEEP on
x86 already do different float32 arithmetic in the subnormal range; byte-matching
MEEP across platforms was never one target. That is exactly why this is a FLAG
(:data:`FLUSH` / :data:`KEEP`, settable by argument or
``MEEP_GPU_SUBNORMAL_POLICY``) rather than a hardcoded choice, why the DEFAULT is
a question about this host rather than a constant — a constant ``"flush"`` default
would make this module's own startup a hard refusal on every non-x86 host — and
why an EXPLICIT request this host cannot deliver is a REFUSAL rather than a silent
downgrade.

THE THREE EXECUTORS, AND WHAT EACH POLICY DOES TO THEM:

===========  ==========================================  =============================
executor     ``"flush"``                                 ``"keep"``
===========  ==========================================  =============================
host/NumPy   ``mp.set_zero_subnormals(True)``, then      ``mp.set_zero_subnormals(False)``
             MEASURE that the FPU actually flushes       then measure that it does not
CuPy         native for compiled kernels — CuPy already  strip ``-ftz=true`` at the
             appends ``-ftz=true`` — and REFUSE unless   ``compile_using_nvrtc`` seam,
             ``CUPY_ACCELERATORS=''`` took CUB out of    with a policy-suffixed cache
             the reduction path before CuPy imported     dir; CUB keeps already
Triton       inject the LLVM per-function attribute      set the module flag
             ``denormal-fp-math-f32`` into the ``llir``  ``nvvm-reflect-ftz`` to 0 so
             stage, and AUDIT the resulting PTX          libdevice stops flushing
===========  ==========================================  =============================

NEITHER POLICY IS FREE AT THE EDGES, and the exceptions are recorded rather than
smoothed over — see :data:`EXECUTOR_OP_EXCEPTIONS`. Two are load-bearing:

* Under ``"flush"`` the HOST does not flush ``-x``. x86 FTZ/DAZ are MXCSR bits
  governing SSE arithmetic, and negation lowers to a bitwise sign flip (``xorps``),
  so ``np.negative`` on a float32 subnormal returns it unchanged — measured on
  the GPU host: ``0x00004000 -> 0x80004000`` with the FPU flushing every other op.
  The device's ``neg.ftz.f32`` DOES flush. Nothing here can reconcile that; it is
  named, and the conformance gate expects the host cell to keep.
* CuPy's DEFAULT reduction path is a CUB binary compiled when CuPy was BUILT, so
  neither CuPy's own ``-ftz=true`` nor this module's strip reaches it: measured,
  ``cp.sum`` over one ``2^-135`` among zeros returns ``0x00004000`` with the rest
  of the process flushing. AND IT CANNOT BE SWITCHED OFF FROM INSIDE THE PROCESS.
  ``set_reduction_accelerators([])`` returns cleanly, the getter then reports
  ``[]``, and the dispatch does not move: measured on device in exactly that
  state, ``cp.sum`` / ``cp.max`` / ``cp.add.reduce`` / ``ndarray.sum`` all still
  returned ``0x00004000`` with ZERO NVRTC compiles, while the same buffer sliced
  ``[::2]`` — a shape CUB declines — compiled a kernel and returned
  ``0x00000000``. The conformance gate is what caught it, failing exactly
  ``cupy_reduction/reduce_sum`` and ``reduce_max`` while the install still claimed
  ``attained=True``. So under ``"flush"`` this module REFUSES unless CuPy imported
  with the accelerator list already empty (``CUPY_ACCELERATORS=''`` in the
  environment), and the refusal names the variable. Under ``"keep"`` CUB already
  keeps and is left alone, which also means ``"keep"`` needs no such setup.

NEVER PASS ``-ftz=false``. NVRTC hard-errors on a duplicate ``-ftz`` option, in
both orders; the only alignment direction for CuPy is to remove the option CuPy
adds. And the CuPy cache key is computed ABOVE the ``compile_using_nvrtc`` seam,
so the strip changes binaries without changing keys — hence the mandatory
policy-suffixed ``CUPY_CACHE_DIR``. Triton's key does include ``backend.hash()``,
which this module folds BOTH the policy name AND a digest of the injection
MECHANISM into: the name alone is not enough, measured — four different injected
attribute strings compiled into one ``TRITON_CACHE_DIR`` all returned the first
variant's PTX.

WHY THE PTX AUDIT EXISTS AT ALL. The LLVM attribute is complete on today's kernel
source — every f32 arithmetic instruction the shipped kernels emit lowers through
LLVM and flips (censused: 26 kernels AOT-compiled twice, 11 736 f32 arithmetic
instructions, every one carrying ``.ftz`` under flush and none left behind;
``selp.f32`` and ``ld.param.f32`` are transport and unchanged). It is NOT complete
as a property of the mechanism: a plain ``/`` on floats lowers to ``div.full.f32`` as
opaque inline asm, which LLVM never sees, so that one instruction would silently
keep subnormals while its neighbours flush — manufacturing exactly the internal
divergence this module exists to prevent. The audit turns "complete by accident"
into an invariant enforced at compile time: it reads every generated PTX and
refuses the compile if any audited f32 instruction disagrees with the requested
policy. It reads PREDICATED instructions too (``@%p1 mul.rn.f32`` is an
instruction, not a directive) and the packed ``.f32x2`` opcodes. The shipped
kernels use ``tl.math.fma``, ``tl.math.div_rn``, ``tl.where``, ``!=`` and plain
``+ - *``; none uses a plain ``/``.

WHY INSTALLATION MUST COME FIRST, and what happens when it does not. Both device
executors memoize compiled kernels IN PROCESS, above their disk caches: a CuPy
kernel compiled before the install keeps flushing after a ``"keep"`` install, and
a Triton kernel compiled before a ``"flush"`` install keeps keeping, with every
counter reading zero. Installing therefore DROPS those in-process caches
(``cupy._util.clear_memo``; every live ``JITFunction.cache``) and records how many
entries it dropped — a nonzero count is the fingerprint of a late install.

INSTALLATION IS ADDITIVE. This module never edits :mod:`meep_gpu.backends`; it
composes with it. :func:`install_cupy_policy` calls
``backends.guard_kernel_compilation`` first and then wraps the same entry points
on top, and because both wrappers are built with ``functools.wraps`` (which copies
``__dict__``) each one's idempotence marker propagates outward — so a later
``guard_kernel_compilation`` finds its mark and does not stack a second layer.
That composition is pinned by ``test_subnormal_policy.py``.

Importable with no CuPy, no Triton and no MEEP: nothing outside the standard
library and :mod:`meep_gpu.backends` is imported at module level.
"""

from __future__ import annotations

import ctypes
import functools
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import re
import sys
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import backends

# ---------------------------------------------------------------------------
# The flag
# ---------------------------------------------------------------------------

#: Flush float32 subnormals to zero — what stock MEEP does on a host where its own
#: ``_set_zero_subnormals`` is compiled in.
FLUSH = "flush"

#: Keep float32 subnormals — IEEE-754, and what MEEP itself does on every host
#: where that function is the ``#if HAVE_IMMINTRIN_H`` no-op.
KEEP = "keep"

#: The two policies an executor can be DRIVEN to, and the only two an artifact can
#: be stamped with. :data:`MATCH_MEEP` is deliberately not here: it is a question,
#: and bytes are certified under its answer.
POLICIES: Tuple[str, str] = (FLUSH, KEEP)

#: Ask for whatever MEEP itself does on THIS host — and find out by MEASURING it
#: (:func:`resolve_match_meep`), never by recognizing a machine string. The
#: default, and the reason ``platform.machine()`` decides nothing in this module.
MATCH_MEEP = "match_meep"

#: Everything a caller may ask for, by argument or through
#: ``MEEP_GPU_SUBNORMAL_POLICY``. A superset of :data:`POLICIES` by exactly the
#: one name that resolves to a member of it.
REQUESTABLE: Tuple[str, str, str] = (FLUSH, KEEP, MATCH_MEEP)

#: Where :data:`MATCH_MEEP` lands when the measurement cannot be taken at all —
#: MEEP not imported here, and unimportable by this module (see
#: :func:`_meep_module`). Chosen, and recorded as a fallback rather than as an
#: answer, because IEEE-754 is attainable on EVERY host: a ``"flush"`` fallback
#: would turn an unmeasurable startup into a hard refusal wherever MEEP's own knob
#: is a no-op — the exact failure a constant ``"flush"`` default has, and the
#: reason there is no constant default.
MATCH_MEEP_FALLBACK = KEEP


def default_policy() -> str:
    """What is REQUESTED when nobody asks: :data:`MATCH_MEEP`.

    A question, not an answer. :func:`resolve_match_meep` answers it with a
    measurement and yields a member of :data:`POLICIES`; :func:`get_subnormal_policy`
    hands back that answer, and the stamp carries both.
    """
    return MATCH_MEEP


def _meep_module() -> Any:
    """An ALREADY-IMPORTED MEEP, or None. This module never imports MEEP.

    ``backends.py`` states the boundary: "no module in this package may import
    cupy at module level, and none may import meep at all". Importing it here to
    reach ``set_zero_subnormals`` — or to find out what it does to the FPU — would
    break that rule and, worse, would do it invisibly: ``import meep`` initializes
    MPI, so merely choosing a flag would print ``Using MPI version 3.1, 1 processes``
    and join a communicator. Measured: an earlier revision of this function did
    exactly that.

    The consequence is deliberate and runs in two places. The DEFAULT cannot be
    measured in a process that has not imported MEEP (:func:`resolve_match_meep`
    says so and falls back by name), and on a host whose FPU does not already agree
    with an explicitly requested policy, the host leg is UNATTAINABLE and says so.
    In the process this package is actually wired into, MEEP is imported long
    before any policy is resolved or installed.
    """
    return sys.modules.get("meep")


def _meep_installed() -> Optional[bool]:
    """Is MEEP installed in this environment? ``None`` when that cannot be told.

    ``find_spec`` LOCATES a module without executing it, so this answers "could
    this process have imported MEEP" without initializing MPI. It is used only to
    make the unmeasurable case say WHICH unmeasurable case it is — a standalone
    engine user with no MEEP at all, or a MEEP that simply has not been imported
    yet — and never to decide a policy.
    """
    try:
        return importlib.util.find_spec("meep") is not None
    except Exception:
        # A sentinel ``None`` parked in ``sys.modules`` (what tests use to block an
        # optional dependency) makes ``find_spec`` raise rather than answer.
        return None


def resolve_match_meep() -> Dict[str, Any]:
    """MEASURE what MEEP itself does to float32 subnormals here, and say why.

    THE RULE: the default policy is whatever MEEP does on this host. MEEP settles
    that for itself at import — ``initialize::initialize() -> setup() ->
    set_zero_subnormals(true)``, compiled under ``#if HAVE_IMMINTRIN_H`` — so with
    MEEP imported the FPU is already in the state MEEP put it in and
    ``backends.subnormals_flushed()`` reads it. No platform table (the machine
    string is recorded and never consulted), no mutation (this asks a question; it
    is :func:`install_host_policy` that drives), and no import.

    Measured on this package's own dev laptop (arm64, MEEP 1.33.0): flushing False
    before ``import meep``, False after — and ``mp.set_zero_subnormals(True)``
    leaves it False. The default there resolves to ``"keep"``, BECAUSE THE HOST WAS
    MEASURED. On an x86 host the same read returns True and it resolves to
    ``"flush"``.

    WHEN THE MEASUREMENT CANNOT BE TAKEN this does not quietly pick a side. A
    process that has not imported MEEP has nothing to read, and this module may not
    import it (:func:`_meep_module`) — so the record says ``measured=False``, names
    :data:`MATCH_MEEP_FALLBACK` as a FALLBACK rather than an answer, says why that
    one, and says what to do to get the question answered instead. The whole record
    travels into :func:`policy_stamp`.

    ONE CAVEAT IS RECORDED RATHER THAN GUARDED: once this module has installed a
    policy, the FPU state is partly ITS doing, so the reading is no longer
    attributable to MEEP alone. ``attributable_to_meep`` says so. It does not
    change any resolution in practice — :func:`get_subnormal_policy` returns the
    installed policy without resolving anything — but a stamp that claimed a
    measurement of MEEP when it had measured this module would be a false record.
    """
    module = _meep_module()
    installed = _STATE["policy"]
    record: Dict[str, Any] = {
        "requested": MATCH_MEEP,
        "meep_imported": module is not None,
        "meep_installed": _meep_installed(),
        "machine": platform.machine(),  # RECORDED for the reader; consulted by nothing.
        "policy_installed": installed,
    }
    if module is not None:
        flushing = backends.subnormals_flushed()
        policy = FLUSH if flushing else KEEP
        why = (f"{MATCH_MEEP} (the default request) resolved to {policy!r} "
               f"because MEEP's own initialization left this host "
               f"{'flushing' if flushing else 'keeping'} float32 subnormals — "
               f"measured through backends.subnormals_flushed() with meep already "
               f"in sys.modules, on machine {platform.machine()!r}")
        if installed is not None:
            why += (f"; NOTE this process has already installed the {installed!r} "
                    f"policy, so the FPU state read here is partly this module's "
                    f"doing and is no longer attributable to MEEP alone")
        record.update(
            policy=policy,
            measured=True,
            flushing=flushing,
            mechanism="backends.subnormals_flushed(), read after MEEP's own initialization",
            attributable_to_meep=installed is None,
            fallback=None,
            reasons=[],
            why=why,
        )
        return record

    presence = {
        True: "MEEP is installed in this environment but has not been imported in "
              "this process",
        False: "MEEP is not installed in this environment, so there is no MEEP "
               "behaviour on this host to match",
        None: "whether MEEP is installed here could not be determined without "
              "importing it",
    }[record["meep_installed"]]
    reason = (
        f"{presence}, and this package may not import MEEP to find out — importing "
        f"it initializes MPI, so merely choosing a flag would join a communicator — "
        f"so what MEEP does to float32 subnormals here was not measurable")
    record.update(
        policy=MATCH_MEEP_FALLBACK,
        measured=False,
        flushing=None,
        mechanism="none (MEEP is not imported in this process)",
        attributable_to_meep=False,
        fallback=MATCH_MEEP_FALLBACK,
        reasons=[reason],
        why=(f"{MATCH_MEEP} (the default request) could NOT be measured: {reason}. "
             f"It fell back "
             f"to {MATCH_MEEP_FALLBACK!r} — IEEE-754, which is attainable on every "
             f"host, so an unmeasurable default can never turn a startup into a "
             f"refusal — and this is a FALLBACK, not a measurement of MEEP. Import "
             f"MEEP before resolving the policy to have the question answered, or "
             f"request {FLUSH!r} or {KEEP!r} explicitly."),
    )
    return record



#: Environment override, consulted when no policy is passed explicitly.
POLICY_ENV = "MEEP_GPU_SUBNORMAL_POLICY"

#: Marker set on every wrapper this module installs, so a second install is a
#: no-op. Deliberately DISTINCT from ``backends._GUARD_MARK``: the two wrappers
#: are independent and each must be able to see whether it, specifically, is on.
_POLICY_MARK = "_meep_gpu_subnormal_policy"

#: The policy name a ``"keep"`` artifact carries. Unchanged from the name the nine
#: certified families were cut under, so their stamps stay comparable and
#: ``gate_triton_complex.probe_record_policy_reasons`` keeps accepting them.
CUPY_KEEP_POLICY_NAME = "ieee_keep_ftz_stripped"

#: The policy name a ``"flush"`` artifact carries.
FLUSH_POLICY_NAME = "meep_x86_flush"

#: What a stamp says when NO policy has been installed. Deliberately not a policy
#: name: the resolved policy is a preference until something drives an executor to
#: it, and a stamp is a certification.
UNINSTALLED_POLICY_NAME = "none_installed"

#: Token a ``CUPY_CACHE_DIR`` must carry under ``"keep"`` — and must NOT carry
#: under ``"flush"``. CuPy computes its cache key above the seam the strip
#: installs at, so a shared directory either gets poisoned with stripped binaries
#: or silently serves flushed ones.
CUPY_CACHE_POLICY_TOKEN = "ftz_stripped"

#: The CuPy compiler front ends, in every backend CuPy ships — the same tuple
#: ``backends.guard_kernel_compilation`` wraps, because it is the same seam.
_COMPILER_ENTRY_POINTS = backends._COMPILER_ENTRY_POINTS

#: The option CuPy appends unconditionally, in both spellings.
_CUPY_FTZ_OPTIONS = ("-ftz=true", "--ftz=true")

#: The LLVM per-function attribute that makes f32 arithmetic lower to ``.ftz``
#: PTX. Triton 3.1.0's ``CUDAOptions`` carries no ftz field, so this is the knob.
#: Measured: with this attribute LLVM lowers every f32 arithmetic op in a real
#: kernel to its ``.ftz`` form; the older ``nvptx-f32ftz`` spelling is IGNORED by
#: the LLVM in Triton 3.1.0, so the spelling is part of the mechanism, not a
#: preference — which is why :func:`mechanism_digest` folds it into the cache key.
TRITON_DENORMAL_ATTRIBUTE = '"denormal-fp-math-f32"="preserve-sign,preserve-sign"'

#: Prefix the policy contributes to Triton's cache key via ``CUDABackend.hash``.
TRITON_HASH_PREFIX = "meep-gpu-subnormal-"

#: Bumped whenever the injected text changes shape. Part of the cache key: Triton
#: keys on ``backend.hash()``, so a mechanism change that does not change the key
#: silently serves the previous mechanism's binaries — measured, four different
#: injected attribute strings in one cache directory all returned the first's PTX.
INJECTION_REVISION = "2"

#: The LLVM module flag Triton sets unconditionally (``set_nvvm_reflect_ftz``,
#: ``backends/nvidia/compiler.py:221``). It decides which libdevice variant
#: ``__nvvm_reflect`` selects — ``div.rn.ftz.f32`` vs ``div.rn.f32`` — and is
#: NOT governed by the per-function denormal attribute. ``"keep"`` rewrites it to
#: 0, which is what makes ``"keep"`` actually IEEE-754 for ``tl.math.div_rn``.
TRITON_REFLECT_FTZ_FLAG = "nvvm-reflect-ftz"

#: What each policy wants that flag to be. ``"flush"`` wants Triton's own value
#: (1: libdevice flushes, which is what flushing asks for), ``"keep"`` wants 0.
_REFLECT_FTZ_VALUE: Dict[str, int] = {FLUSH: 1, KEEP: 0}

#: Executor-by-executor exceptions: op classes whose subnormal behaviour is NOT
#: decided by the requested policy. Each entry is measured, and the conformance
#: gate reads this table rather than assuming the policy is uniform everywhere.
EXECUTOR_OP_EXCEPTIONS: Dict[str, Dict[str, str]] = {
    "host_negation": {
        "executor": "host",
        "ops": "neg (np.negative, unary -)",
        "policies": FLUSH,
        "behaviour": "keeps",
        "why": "x86 FTZ/DAZ are MXCSR bits governing SSE ARITHMETIC; negation "
               "lowers to a bitwise sign flip (xorps) and never consults them. "
               "Measured on the GPU host with the FPU flushing: np.negative("
               "0x00004000) -> 0x80004000, while np.multiply / np.subtract / "
               "np.divide on the same operand all returned zero. The device's "
               "neg.ftz.f32 flushes, so host and device differ on -x under flush "
               "and no knob in this module can reconcile them.",
    },
    "cupy_cub_reductions": {
        "executor": "cupy",
        "ops": "sum / max / min / add.reduce over a contiguous array",
        "policies": FLUSH,
        "behaviour": "keeps unless CUPY_ACCELERATORS is empty before CuPy imports",
        "why": "CuPy 13.5.1 defaults to the CUB reduction accelerator, a binary "
               "compiled when CuPy was BUILT, so no compile-time option this "
               "module can reach applies. Measured with the rest of the process "
               "flushing: cp.sum over one 2^-135 among 4095 zeros returned "
               "0x00004000 and cupy_strip_counters() recorded zero NVRTC calls. "
               "set_reduction_accelerators([]) does NOT move the dispatch — "
               "measured on device with the list reported empty, cp.sum, cp.max, "
               "cp.add.reduce and ndarray.sum all still returned 0x00004000 with "
               "zero NVRTC compiles, while the same buffer sliced [::2] (a shape "
               "CUB declines) compiled a kernel and returned 0x00000000. The only "
               "lever is CUPY_ACCELERATORS='' in the environment before the "
               "import, so install_cupy_policy REFUSES flush without it.",
    },
    "triton_libdevice": {
        "executor": "triton",
        "ops": "div_rn / sqrt / rsqrt / ex2 / lg2 / sin / cos / tanh",
        "policies": KEEP,
        "behaviour": "flushes unless the nvvm-reflect-ftz module flag is rewritten",
        "why": "Triton calls set_nvvm_reflect_ftz unconditionally, so libdevice "
               "resolves to its .ftz variants: measured, tl.math.div_rn emits "
               "div.rn.ftz.f32 natively. Rewriting the module flag to 0 in the "
               "llir stage gives div.rn.f32 on the same kernel through the same "
               "seam, which is what install_triton_policy does under keep.",
    },
}


class SubnormalPolicyUnattainable(RuntimeError):
    """The requested policy could not be delivered by one of the executors.

    Raised rather than downgraded: a process that asked to flush and quietly did
    not is a process whose bytes cannot be attributed to either policy.
    """


class SubnormalPolicyLocked(RuntimeError):
    """A different policy was requested after one was already installed.

    Refused rather than swapped. Binaries and cache entries already exist under
    the installed policy — CuPy's cache key in particular is computed above the
    seam the strip installs at, so a swap would serve the previous policy's
    device binaries under the new policy's name.
    """


_STATE: Dict[str, Any] = {
    "policy": None,        # The installed policy, or None before the first install.
    "requested": None,     # What was ASKED for: "flush" | "keep" | "match_meep".
    "resolution": None,    # The match_meep measurement, when that is what was asked.
    "resolved_from": None, # "argument" | "environment" | "default"
    "executors": {},       # executor name -> its install report
}


def policy_resolution(policy: Optional[str] = None) -> Dict[str, Any]:
    """What was asked, where it came from, what it resolves to, and why.

    Order: explicit argument, then ``MEEP_GPU_SUBNORMAL_POLICY``, then
    :func:`default_policy`. The first two may name any member of
    :data:`REQUESTABLE`; the third IS :data:`MATCH_MEEP`, which
    :func:`resolve_match_meep` answers by measuring this host.

    AN EXPLICIT ``"flush"`` OR ``"keep"`` IS AN ANSWER ALREADY and is passed
    straight through — nothing about match_meep may change what an explicit
    request does, in either direction. ``resolution`` is the measurement record for
    a match_meep request and ``None`` for an explicit one, so a reader can always
    tell a measured policy from a demanded one.
    """
    if policy is not None:
        source, chosen = "argument", policy
    else:
        from_env = os.environ.get(POLICY_ENV)
        if from_env is not None and from_env.strip() != "":
            source, chosen = "environment", from_env
        else:
            source, chosen = "default", default_policy()
    requested = str(chosen).strip().lower()
    if requested not in REQUESTABLE:
        raise ValueError(
            f"subnormal policy {chosen!r} (from the {source}) is not one of "
            f"{REQUESTABLE}: {FLUSH!r} flushes float32 subnormals to zero, which is "
            f"what stock MEEP does wherever its own set_zero_subnormals is compiled "
            f"in; {KEEP!r} is IEEE-754; {MATCH_MEEP!r} measures which of those MEEP "
            f"itself does on this host and takes that"
        )
    if requested != MATCH_MEEP:
        return {"requested": requested, "source": source,
                "policy": requested, "resolution": None}
    resolution = resolve_match_meep()
    return {"requested": MATCH_MEEP, "source": source,
            "policy": resolution["policy"], "resolution": resolution}


def resolve_policy(policy: Optional[str] = None) -> Tuple[str, str]:
    """The policy that will actually be installed, and where the request came from.

    ``(policy, source)``, where ``policy`` is always a member of :data:`POLICIES`
    — an executor can be driven to ``"flush"`` or to ``"keep"`` and to nothing
    else. When :data:`MATCH_MEEP` was requested, this is its MEASURED answer; the
    request itself and the measurement behind it are in :func:`policy_resolution`,
    and both travel into :func:`policy_stamp`.
    """
    record = policy_resolution(policy)
    return record["policy"], record["source"]


def get_subnormal_policy() -> str:
    """The RESOLVED policy in force — installed if one has been, else the resolution.

    Always ``"flush"`` or ``"keep"``, never :data:`MATCH_MEEP`: a caller asking
    what this process does to subnormals wants the answer, and the question it
    came from is in the stamp.

    Named ``get_subnormal_policy`` and not ``subnormal_policy`` on purpose: a
    package-level function by the latter name would SHADOW the
    ``meep_gpu.subnormal_policy`` MODULE, so ``import
    meep_gpu.subnormal_policy as sp`` would hand back a function. The same trap
    is already documented for ``harminv`` in the package ``__init__``.
    """
    installed = _STATE["policy"]
    return installed if installed is not None else resolve_policy()[0]


def policy_is_installed() -> bool:  # Has any executor been driven yet?
    return _STATE["policy"] is not None


#: Bumped on every transition of the installed policy — install, rollback,
#: uninstall. ADDED for the dispatch gate, which needs two things a boolean
#: cannot give it: to tell "an earlier freeze of this process installed this" from
#: "a caller did", and to notice mid-run that the policy its kernels were gated on
#: is no longer the one in force. Both are identity questions about a policy
#: INSTANCE, and a name plus a flag answers neither across an uninstall.
_EPOCH = 0


def policy_epoch() -> int:
    """A counter that CHANGES whenever the installed policy does.

    Equal values mean "the same install is still in force"; any install,
    rollback or uninstall in between makes them differ. Cheap enough to consult
    on every dispatch consult (:meth:`fastpath.FastPathPlan.dispatch`), which is
    the point: the configuration freeze's guarantee is about the freeze, and
    ``uninstall_subnormal_policy`` is a public function.
    """
    return _EPOCH


def _bump_epoch() -> int:
    global _EPOCH
    _EPOCH += 1
    return _EPOCH


def executor_report(executor: str) -> Dict[str, Any]:
    """This executor's install record, or an empty dict if it never ran.

    The public read of what :func:`policy_stamp` embeds, so a gate can ask "what
    did the CuPy leg actually do" without reaching into module state.
    """
    return dict(_STATE["executors"].get(executor) or {})


def record_executor_report(executor: str, report: Dict[str, Any], *,
                           strict: bool = True) -> Dict[str, Any]:
    """File one executor arm's install record, and refuse if it did not attain.

    THE WRITE SIDE OF :func:`executor_report`, and it exists so an executor arm
    that lives in ANOTHER MODULE can file its report without reaching into
    ``_STATE`` or into :func:`_announce`. The MPS arm
    (``metal_kernels.subnormal.install_mps_policy``) is the first: its lever is
    native, so keeping its report shape in the Metal package is right, and keeping
    the BOOKKEEPING here is what makes ``policy_stamp``, ``unattained`` and the
    refusal wording one implementation rather than two.

    ``strict`` has the same meaning it has everywhere in this module: raise
    :class:`SubnormalPolicyUnattainable` on an unattained policy, or warn and let
    the caller read ``attained`` off the report.
    """
    _STATE["executors"][executor] = report
    _announce(report, strict=strict)
    return report


# ---------------------------------------------------------------------------
# Host / NumPy: MEEP's own knob, and the measurement that it worked
# ---------------------------------------------------------------------------


def host_policy_reasons(policy: str, module: Any, before: bool, after: bool) -> List[str]:
    """Why the host FPU is not obeying ``policy``, by name. Empty means it is."""
    want_flush = policy == FLUSH
    if after == want_flush:
        return []
    machine = platform.machine()
    symbol = _FENV_SYMBOL[bool(want_flush)]
    if module is None:
        return [
            f"MEEP is not imported in this process, and MEEP's "
            f"``set_zero_subnormals`` is the only exposure of this process's "
            f"FTZ/DAZ bits that this package may use (importing MEEP here would "
            f"break the package's no-MEEP-import boundary and would initialize "
            f"MPI as a side effect); measured host flushing is {before!r} and "
            f"{policy!r} needs {want_flush!r}. The fenv fallback ({symbol}) was "
            f"tried and did not move the FPU either. Import MEEP before installing "
            f"the policy, or request {KEEP!r} on a host that already keeps"
        ]
    return [
        f"mp.set_zero_subnormals({want_flush!r}) did not change this host's FPU: "
        f"flushing measured {before!r} before the call and {after!r} after, on "
        f"machine {machine!r}. MEEP's ``_set_zero_subnormals`` is compiled under "
        f"``#if HAVE_IMMINTRIN_H`` (x86 SSE), so on a non-x86 host it is a no-op — "
        f"which also means MEEP itself keeps subnormals on this machine. The fenv "
        f"fallback below it ({symbol}) was tried and did not move the FPU either "
        f"(its symbol is absent on this libm, or it moved nothing measurable), so "
        f"{policy!r} is unattainable for the host path here"
    ]


#: How the non-x86 host lever is named in a report, one spelling per direction.
#: ``FE_DFL_DISABLE_DENORMS_ENV`` is Darwin's own "default environment, denormals
#: off" object; ``FE_DFL_ENV`` is the standard default, which restores IEEE
#: denormals. A report names the object that was actually installed, because "the
#: fenv lever" is two different objects and a stamp that said only "fenv" could not
#: be checked against the state it claims.
_FENV_SYMBOL = {True: "_FE_DFL_DISABLE_DENORMS_ENV", False: "_FE_DFL_ENV"}
_FENV_MECHANISM = {True: "fenv:FE_DFL_DISABLE_DENORMS_ENV", False: "fenv:FE_DFL_ENV"}

#: The prefix every fenv mechanism string starts with, so a reader — and
#: :func:`install_host_policy`'s ``installed`` field — can recognise one without
#: matching on either direction's spelling.
FENV_MECHANISM_PREFIX = "fenv:"


def _drive_host_fpu_by_fenv(want_flush: bool) -> bool:
    """The non-x86 host lever: install a whole floating-point ENVIRONMENT. Did it work?

    WHY THIS EXISTS AT ALL. MEEP's ``set_zero_subnormals`` is the only FTZ/DAZ knob
    this package may use, and on arm64 it is the ``#if HAVE_IMMINTRIN_H`` no-op — so
    without a second lever the ``flush`` policy is permanently unattainable on Apple
    silicon, and the Metal executor, which can deliver NOTHING BUT flush, is
    permanently refused with it. Darwin exports ``_FE_DFL_DISABLE_DENORMS_ENV``
    exactly for this, and ``fesetenv`` on it flips FTZ/DAZ for the calling thread.

    THE MEASUREMENT IS THE VERDICT AND THE RESTORE IS UNCONDITIONAL, both for the
    same reason ``backends._default_environment`` works that way: a header reading
    cannot settle whether an environment object carries the denormal bits on this
    ABI, and a lever that moved the FPU without attaining the policy would leave the
    process in a third state belonging to nobody. So the previous environment is
    saved, the candidate installed, ``subnormals_flushed`` asked, and the save put
    back the moment the answer is no.

    CHOSEN OVER ``torch.set_flush_denormal``, which was measured to work equally
    well and reversibly on this host, so that this module stays TORCH-FREE: the
    lever has to be available in array-only processes — the dispatch campaign's
    band-witness leg is one — and in a process that has no MPS device at all.

    ``backends`` itself is NOT edited: :func:`backends._math_library` already
    resolves ``fegetenv``/``fesetenv`` with the argtypes set, and that is the whole
    of what is borrowed.
    """
    library = backends._math_library()  # noqa: SLF001 - the resolved libm, argtypes set
    if library is None:
        return False
    symbol = _FENV_SYMBOL[bool(want_flush)]
    try:
        candidate = ctypes.addressof(ctypes.c_char.in_dll(library, symbol))
    except (ValueError, AttributeError):
        return False
    saved = ctypes.create_string_buffer(backends._FENV_BYTES)  # noqa: SLF001
    if library.fegetenv(ctypes.addressof(saved)) != 0:
        return False
    if library.fesetenv(candidate) != 0:
        return False
    if backends.subnormals_flushed() == bool(want_flush):
        return True
    library.fesetenv(ctypes.addressof(saved))
    return False


def _drive_host_fpu(want_flush: bool) -> Tuple[Any, str]:
    """Drive this thread's FTZ/DAZ state. Returns ``(module used, mechanism)``.

    MEEP'S KNOB FIRST, ALWAYS, because on x86 it IS the mechanism the certified
    families were cut under and matching MEEP is the point of the ``match_meep``
    resolution. Only when the measurement says the knob did not move the FPU does
    :func:`_drive_host_fpu_by_fenv` get its turn — so on x86 nothing about this
    function's behaviour changes, and on arm64 the policy becomes attainable
    instead of being refused on a platform fact.

    Split out of :func:`install_host_policy` because the ROLLBACK path needs the
    same operation: a refused install must put the FPU back where it found it, or
    the process keeps running in the refused policy's mode while this module
    reports that no policy is installed. The fenv arm serves the rollback too — it
    is symmetric by construction, ``_FE_DFL_ENV`` being the object that restores
    denormals.
    """
    module = _meep_module()
    mechanism = "none (MEEP is not imported in this process)"
    if module is not None and hasattr(module, "set_zero_subnormals"):
        module.set_zero_subnormals(bool(want_flush))
        mechanism = "meep.set_zero_subnormals"
        if backends.subnormals_flushed() == bool(want_flush):
            return module, mechanism
    lever = _FENV_MECHANISM[bool(want_flush)]
    if _drive_host_fpu_by_fenv(want_flush):
        if module is None:
            return module, lever
        return module, f"{mechanism} (no-op here) + {lever}"
    return module, mechanism


def install_host_policy(policy: str, *, strict: bool = True) -> Dict[str, Any]:
    """Drive the host FPU to ``policy`` through MEEP, and MEASURE the result.

    The measurement is the verdict, never the platform table: this asks the FPU
    whether it is flushing (``backends.subnormals_flushed``) before and after, the
    same way ``backends._default_environment`` decides what ``FE_DFL_ENV`` is.

    ``installed`` NAMES WHAT ACTUALLY DROVE THE FPU, which since the fenv lever
    landed is no longer the same question as "was MEEP imported": on a host where
    MEEP's knob is the no-op, the environment object is what moved the bits, and a
    report saying nothing was installed would leave the process flushing under a
    record that disclaims having done it.
    """
    want_flush = policy == FLUSH
    before = backends.subnormals_flushed()
    module, mechanism = _drive_host_fpu(want_flush)
    after = backends.subnormals_flushed()
    reasons = host_policy_reasons(policy, module, before, after)
    report = {
        "executor": "host",
        "requested": policy,
        "attained": not reasons,
        "installed": module is not None or FENV_MECHANISM_PREFIX in mechanism,
        "mechanism": mechanism,
        "machine": platform.machine(),
        "flushing_before": before,
        "flushing_after": after,
        "reasons": reasons,
        "exceptions": [EXECUTOR_OP_EXCEPTIONS["host_negation"]] if want_flush else [],
    }
    _STATE["executors"]["host"] = report
    _announce(report, strict=strict)
    return report


# ---------------------------------------------------------------------------
# CuPy: the -ftz=true strip, composed onto backends.guard_kernel_compilation
# ---------------------------------------------------------------------------


def cupy_cache_reasons(policy: str, cache_dir: Optional[str]) -> List[str]:
    """Why this ``CUPY_CACHE_DIR`` may not be used under ``policy``, by name.

    CuPy computes its kernel cache key ABOVE the ``compile_using_nvrtc`` seam, so
    stripping the option changes the binaries without changing the key. The
    directory name is therefore the only separator there is, in both directions:
    a keep run needs a private policy-suffixed directory, and a flush run must not
    be handed the keep run's.
    """
    cache_dir = cache_dir or ""
    if policy == KEEP:
        if not cache_dir:
            return [
                "CUPY_CACHE_DIR is unset: the -ftz=true strip changes binaries "
                "without changing CuPy's cache keys (the key is computed above "
                "the compile_using_nvrtc seam), so a private cache directory "
                f"carrying {CUPY_CACHE_POLICY_TOKEN!r} is mandatory under {KEEP!r}"
            ]
        if CUPY_CACHE_POLICY_TOKEN not in cache_dir:
            return [
                f"CUPY_CACHE_DIR {cache_dir!r} does not carry the policy token "
                f"{CUPY_CACHE_POLICY_TOKEN!r}: a cache shared with flush-policy "
                f"runs either gets poisoned with stripped binaries or silently "
                f"serves flushed ones"
            ]
        return []
    if cache_dir and CUPY_CACHE_POLICY_TOKEN in cache_dir:
        return [
            f"CUPY_CACHE_DIR {cache_dir!r} carries the keep-policy token "
            f"{CUPY_CACHE_POLICY_TOKEN!r} but {FLUSH!r} was requested: that "
            f"directory holds binaries compiled with -ftz=true stripped, and "
            f"CuPy would serve them under the flush policy's name"
        ]
    return []


# --- the keep policy's own cache directory ---------------------------------
#
# ADDED for the dispatch policy gate (fastpath.plan_fast_path installs "keep"
# before the warm pass compiles anything). It lives HERE and not in the caller
# because :func:`cupy_cache_reasons` above owns the token rule, and a rule and the
# only way to satisfy it belong in one place. Nothing else in this module changed.


def _cupy_default_cache_dir(cupy: Any = None) -> str:
    """Where CuPy would cache with ``CUPY_CACHE_DIR`` unset. Never raises."""
    try:
        return str(_cupy_compiler(cupy).get_cache_dir())
    except Exception:  # noqa: BLE001 - a CuPy that will not answer gets the documented default
        return os.path.join(os.path.expanduser("~"), ".cupy", "kernel_cache")


def keep_policy_cache_dir(cache_dir: Optional[str] = None, cupy: Any = None) -> str:
    """A ``CUPY_CACHE_DIR`` the ``"keep"`` policy may use, derived from this one.

    A directory that already satisfies :func:`cupy_cache_reasons` is returned
    unchanged; anything else gets :data:`CUPY_CACHE_POLICY_TOKEN` appended, so a
    caller's chosen LOCATION survives and only the policy separation is added.
    With nothing set, the sibling of CuPy's own default is used.

    ONLY ``"keep"`` IS DERIVABLE. Under ``"flush"`` the sole cache refusal is a
    directory carrying the keep token, and the fix there is to stop pointing at
    another policy's cache — a decision about the caller's environment, not a
    string this function may invent. So there is no ``flush_policy_cache_dir``:
    that case stays a named refusal out of :func:`cupy_cache_reasons`.
    """
    current = (os.environ.get("CUPY_CACHE_DIR", "") if cache_dir is None
               else (cache_dir or ""))
    current = current.rstrip(os.sep)
    if current and not cupy_cache_reasons(KEEP, current):
        return current
    base = current or _cupy_default_cache_dir(cupy).rstrip(os.sep)
    return f"{base}-{CUPY_CACHE_POLICY_TOKEN}"


#: The provenance marker's suffix. A SIBLING of the cache directory, never a file
#: inside it: what CuPy 13.5.1 does when it meets an unexpected entry in its own
#: cache directory is not something this package has measured, and a policy
#: mechanism may not rest on a guess about the library it is governing.
KEEP_POLICY_MARKER_SUFFIX = ".meep_gpu_keep_policy"


def keep_policy_cache_marker(cache_dir: str) -> str:
    """Path of the sibling file that says this directory was minted by this policy."""
    return cache_dir.rstrip(os.sep) + KEEP_POLICY_MARKER_SUFFIX


def keep_policy_cache_provenance(cache_dir: str) -> Dict[str, Any]:
    """Whether this keep-policy cache directory is one this package can vouch for.

    THE TOKEN IN THE NAME IS NOT EVIDENCE ABOUT THE BINARIES INSIDE.
    :func:`cupy_cache_reasons` enforces the naming rule and that is all it can do:
    CuPy's cache key is computed ABOVE the ``compile_using_nvrtc`` seam the strip
    installs at, so nothing in a cached ``.cubin`` records which policy compiled
    it, and a hit serves it back with no NVRTC call — no counter moves, and
    ``ftz_removed`` simply stays low. A directory that already holds entries and
    carries no marker is therefore of UNKNOWN origin, and "unknown" on the input
    side of a bit-identity claim is a refusal.

    An EMPTY (or absent) directory is clean whether or not it is marked: there is
    nothing in it to have been compiled under another policy.
    """
    marker = keep_policy_cache_marker(cache_dir)
    marked = os.path.exists(marker)
    try:
        entries = len(os.listdir(cache_dir)) if os.path.isdir(cache_dir) else 0
    except Exception as exc:  # noqa: BLE001 - an unreadable directory is unknown
        return {"marker": marker, "marked": marked, "entries": None,
                "reasons": [f"the directory could not be listed ({exc}), so what it "
                            f"holds cannot be established"]}
    record: Dict[str, Any] = {"marker": marker, "marked": marked, "entries": entries,
                              "reasons": []}
    if entries and not marked:
        record["reasons"] = [
            f"{cache_dir!r} already holds {entries} entries and carries no "
            f"{KEEP_POLICY_MARKER_SUFFIX} marker, so this package did not mint it "
            f"and cannot say which policy compiled what is in it. CuPy's cache key "
            f"is computed above the strip seam, so the directory NAME is the only "
            f"separator there is and a hit is served with no NVRTC call to notice. "
            f"Delete that directory (it is a cache; it costs a recompile) or point "
            f"CUPY_CACHE_DIR somewhere else"]
    return record


def point_cupy_cache_at_keep_policy(cupy: Any = None) -> Dict[str, Any]:
    """Set ``CUPY_CACHE_DIR`` to a keep-policy directory, and report what moved.

    THE LATE SET BITES, and that is the measured precondition for installing this
    policy after CuPy has already compiled something. CuPy 13.5.1 reads the
    variable on EVERY compile — ``get_cache_dir()`` is
    ``os.environ.get('CUPY_CACHE_DIR', _default_cache_dir)``
    (``cupy/cuda/compiler.py:503``) and is called from inside
    ``_compile_with_cache_cuda`` (``:548``), not once at import — so a directory
    chosen now governs every compile from now on. The in-process memo is what the
    install drops separately (:func:`_clear_cupy_kernel_memo`); together they are
    what makes a post-setup install reach the kernels that matter.

    Returns the before/after pair and whether the move was needed, for the
    artifact. Creating the directory is deliberate: an unwritable choice fails
    here, where it is a named refusal, rather than inside the first compile.
    """
    before = os.environ.get("CUPY_CACHE_DIR")
    chosen = keep_policy_cache_dir(cupy=cupy)
    record: Dict[str, Any] = {
        "variable": "CUPY_CACHE_DIR",
        "before": before,
        "after": chosen,
        "changed": (before or "") != chosen,
        "derived_from": "the caller's CUPY_CACHE_DIR" if before else
                        "CuPy's own default cache directory",
    }
    try:
        os.makedirs(chosen, exist_ok=True)
        record["writable"] = os.access(chosen, os.W_OK)
    except Exception as exc:  # noqa: BLE001 - recorded and refused by the caller
        record["writable"] = False
        record["error"] = repr(exc)
        return record
    # PROVENANCE, before the pointer moves. A directory this package did not mint
    # is refused rather than adopted — see :func:`keep_policy_cache_provenance` —
    # and the variable is left exactly where the caller had it, so a refusal
    # cannot leave the process compiling into a cache it was told not to trust.
    provenance = keep_policy_cache_provenance(chosen)
    record["provenance"] = provenance
    if provenance["reasons"]:
        return record
    if not provenance["marked"]:
        try:
            with open(provenance["marker"], "w", encoding="utf-8") as handle:
                json.dump({"policy": KEEP, "token": CUPY_CACHE_POLICY_TOKEN,
                           "directory": chosen, "machine": platform.machine(),
                           "written_by": "meep_gpu.subnormal_policy"}, handle)
            provenance["marked"] = True
        except Exception as exc:  # noqa: BLE001 - not fatal HERE; fatal next process
            provenance["marker_error"] = repr(exc)
    os.environ["CUPY_CACHE_DIR"] = chosen
    return record


# --- end of the dispatch policy gate's addition ----------------------------


def _strip_ftz(options: Any) -> Tuple[Any, ...]:
    """CuPy's option tuple with ``-ftz=true`` removed. Never adds ``-ftz=false``.

    NVRTC hard-errors on a duplicate ``-ftz`` option — measured in both orders —
    so removal is the only alignment direction.
    """
    return tuple(option for option in tuple(options) if option not in _CUPY_FTZ_OPTIONS)


def _wrap_compiler_entry_point(original: Callable, state: Dict[str, Any], name: str) -> Callable:
    """One CuPy compiler front end, with the strip applied and counted.

    ``functools.wraps`` copies ``__dict__``, so if ``original`` already carries
    ``backends._GUARD_MARK`` this wrapper inherits it and a later
    ``guard_kernel_compilation`` correctly sees the seam as already guarded
    instead of stacking another layer on every ``resolve_backend`` call.
    """

    @functools.wraps(original)
    def stripped(source, options=(), *args, **kwargs):
        counters = state["entry_points"].setdefault(name, {"calls": 0, "removed": 0})
        counters["calls"] += 1
        filtered = _strip_ftz(options)
        if len(filtered) != len(tuple(options)):
            counters["removed"] += 1
        if len(state["example_options"]) < 4:
            state["example_options"].append(
                {"entry_point": name, "before": list(options), "after": list(filtered)}
            )
        return original(source, filtered, *args, **kwargs)

    setattr(stripped, _POLICY_MARK, KEEP)
    return stripped


def _cupy_compiler(cupy: Any) -> Any:  # ``cupy.cuda.compiler``, however it is reachable.
    compiler = getattr(getattr(cupy, "cuda", None), "compiler", None)
    if compiler is not None:
        return compiler
    return importlib.import_module("cupy.cuda.compiler")


def _clear_cupy_kernel_memo(cupy: Any) -> Dict[str, Any]:
    """Drop CuPy's IN-PROCESS compiled-kernel memo, so the policy reaches everything.

    CuPy memoizes elementwise, ufunc and raw kernels per device inside the
    process, above the disk cache and above the seam the strip installs at.
    Measured: ``cp.multiply`` compiled BEFORE a ``"keep"`` install kept flushing
    afterwards, with the install reporting ``attained=True`` — while the first
    kernel compiled after the install obeyed the policy. Dropping the memo is what
    makes "installed" mean "in force"; kernels are recompiled on next use.

    ``cupy._util.clear_memo`` is the documented hammer (it is what
    ``cupy.clear_memo`` exposes) and clears every ``@memoize``d cache, which is
    exactly the set of caches that hold compiled kernels.
    """
    for holder, name in ((cupy, "clear_memo"), (getattr(cupy, "_util", None), "clear_memo")):
        function = getattr(holder, name, None)
        if callable(function):
            function()
            return {"cleared": True, "mechanism": f"cupy{'' if holder is cupy else '._util'}.clear_memo()"}
    try:
        util = importlib.import_module("cupy._util")
    except Exception as exc:  # pragma: no cover - a CuPy without its own util module
        return {"cleared": False, "mechanism": f"unavailable: {exc}"}
    function = getattr(util, "clear_memo", None)
    if not callable(function):
        return {"cleared": False, "mechanism": "cupy._util exposes no clear_memo"}
    function()
    return {"cleared": True, "mechanism": "cupy._util.clear_memo()"}


def _cupy_reduction_accelerators(policy: str, cupy: Any = None) -> Dict[str, Any]:
    """Bring CuPy's REDUCTION path under the policy, or say why it is not.

    CuPy 13.5.1 routes ``sum``/``max``/``min``/``*.reduce`` on a contiguous array
    through CUB by default — a binary compiled when CuPy was built, which neither
    CuPy's ``-ftz=true`` nor this module's strip can touch. Measured with the rest
    of the process flushing: ``cp.sum`` over one ``2^-135`` among 4095 zeros
    returned ``0x00004000`` (kept) and zero NVRTC compiles were recorded.

    THE RUNTIME SWITCH-OFF DOES NOT WORK, and this function used to believe it
    did. ``set_reduction_accelerators([])`` returns cleanly and
    ``get_reduction_accelerators()`` then reports ``[]`` — and the very next
    ``cp.sum`` still runs CUB. Measured on device (the GPU host, CuPy 13.5.1, the
    conformance gate's own operand): with the accelerators reported empty,
    ``cp.sum`` / ``cp.max`` / ``cp.add.reduce`` / ``ndarray.sum`` all returned
    ``0x00004000`` with ZERO NVRTC compiles, while the same buffer sliced
    (``[::2]``, a shape CUB will not take) compiled a kernel and returned
    ``0x00000000``. The dispatch reads a list built when CuPy was IMPORTED, so
    the only lever is ``CUPY_ACCELERATORS`` in the environment, before the import.

    ``before`` is therefore the verdict, not ``after``: a non-empty list means
    CuPy imported with CUB enabled and no in-process call can take it away. Under
    ``"flush"`` that is a refusal naming the environment variable. Under ``"keep"``
    CUB already keeps and agrees with the policy, so it is left alone and merely
    recorded. The reach of this in the engine is not hypothetical: ``driver.py``
    calls ``xp.max(xp.abs(...))`` for divergence detection and ``xp.sum`` for
    field energy, ``stepping.py`` calls ``xp.min``/``xp.max``, and ``dft.py`` has
    14 reduction sites behind flux, energy and near2far.
    """
    accelerator = getattr(getattr(cupy, "_core", None), "_accelerator", None)
    try:
        if accelerator is None:
            accelerator = importlib.import_module("cupy._core._accelerator")
        getter = accelerator.get_reduction_accelerators
    except Exception as exc:
        return {"reachable": False, "before": None, "after": None,
                "reasons": [] if policy == KEEP else [
                    f"cupy._core._accelerator is not importable ({exc}), so CuPy's "
                    f"CUB reduction accelerator cannot be switched off; under "
                    f"{FLUSH!r} reductions would keep subnormals while every other "
                    f"CuPy path flushes. Set CUPY_ACCELERATORS='' before importing "
                    f"CuPy and install the policy again"]}
    # The RAW objects are what goes back on rollback; the strings are for the
    # report only. CuPy's getter returns an accelerator ENUM whose ``str`` is its
    # integer value ("1"), and its setter accepts only the NAMES ("cub",
    # "cutensor") — measured: feeding the report's own strings back raised
    # ``ValueError: Unknown accelerator: 1`` from inside the rollback, which turned
    # a clean refusal into a crash with the accelerators left switched off.
    raw_before = list(getter())
    before = [str(name) for name in raw_before]
    if policy == KEEP:
        return {"reachable": True, "before": before, "after": before, "reasons": [],
                "note": "left alone: CUB keeps subnormals, which is what 'keep' asks for"}
    setter = getattr(accelerator, "set_reduction_accelerators", None)
    if not callable(setter):
        return {"reachable": True, "before": before, "after": before, "reasons": [
            "cupy._core._accelerator exposes no set_reduction_accelerators, so the "
            "CUB reduction path cannot be brought under the flush policy in this "
            "process; set CUPY_ACCELERATORS='' before importing CuPy instead"]}
    setter([])
    _STATE["_cupy_accelerators"] = (setter, raw_before)  # so a refusal can put them back
    after = [str(name) for name in getter()]
    reasons: List[str] = []
    if after:
        reasons.append(
            f"CuPy still reports reduction accelerators {after!r} after asking "
            f"for none: the CUB reduction path stays outside the {FLUSH!r} "
            f"policy and cp.sum / cp.max would keep subnormals")
    elif before:
        # The list went empty and the dispatch did not. This is the measured case,
        # not a defensive one — see the docstring for the four reductions that
        # returned 0x00004000 with zero NVRTC compiles in exactly this state.
        reasons.append(
            f"CuPy imported with the reduction accelerators {before!r} active, and "
            f"set_reduction_accelerators([]) does not change where a reduction is "
            f"dispatched in CuPy {getattr(cupy, '__version__', '?')}: measured on "
            f"device with the list reported empty, cp.sum / cp.max / cp.add.reduce "
            f"over one 2^-135 among 4095 zeros all returned 0x00004000 with zero "
            f"NVRTC compiles, because CUB is a binary built with CuPy and the "
            f"dispatch reads a list fixed at import. Set CUPY_ACCELERATORS='' in "
            f"the environment BEFORE CuPy is imported and install the policy again; "
            f"there is no in-process route to {FLUSH!r} for reductions")
    return {"reachable": True, "before": before, "after": after,
            "runtime_setter_changes_dispatch": False,
            "governed_by": "CUPY_ACCELERATORS at CuPy import time",
            "reasons": reasons}


def _cupy_policy_lock(compiler: Any, policy: str) -> None:
    """Refuse a second, DIFFERENT policy at a seam that already carries one.

    The strip is installed under ``"keep"`` only, so a later ``"flush"`` install
    would find nothing of its own to do and report ``native`` — while the keep
    strip is still filtering ``-ftz=true`` out of every compile. That is a process
    whose bytes belong to neither policy.
    """
    for name in _COMPILER_ENTRY_POINTS:
        installed = getattr(getattr(compiler, name, None), _POLICY_MARK, None)
        if installed is not None and installed != policy:
            raise SubnormalPolicyLocked(
                f"cupy.cuda.compiler.{name} already carries the {installed!r} "
                f"subnormal policy and {policy!r} was requested. The strip is a "
                f"process-wide wrapper and its binaries are already in the cache; "
                f"choose the policy before the first compile, in a fresh process."
            )


def install_cupy_policy(policy: str, cupy: Any = None, *, strict: bool = True) -> Dict[str, Any]:
    """Make CuPy's device binaries obey ``policy``.

    Under ``"flush"`` CuPy's COMPILES are left native — it already appends
    ``-ftz=true`` to every NVRTC compile — and the work is elsewhere: the
    cache-directory refusal, switching CUB reductions off so the one CuPy path
    that never compiles cannot ignore the policy, and dropping the in-process
    kernel memo. Under ``"keep"`` the option is stripped at the compiler front
    ends as well.

    Either way this first calls ``backends.guard_kernel_compilation``, which is a
    different concern layered at the same seam: it suspends the HOST FPU's
    flushing for the duration of a compile so NVRTC can parse the subnormal
    literals in CCCL's ``<cuda/std/limits>``. That guard matters MORE under
    ``"flush"``, not less, because the host is then deliberately flushing — so
    under ``"flush"`` a guard that could not be installed is a REFUSAL, not a note.
    """
    if cupy is None:
        if importlib.util.find_spec("cupy") is None:
            report = {"executor": "cupy", "requested": policy, "attained": True,
                      "mechanism": "none (CuPy not installed; the device policy "
                                   "governs device binaries only)",
                      "installed": False, "reasons": []}
            _STATE["executors"]["cupy"] = report
            return report
        cupy = importlib.import_module("cupy")

    cache_dir = os.environ.get("CUPY_CACHE_DIR", "")
    report: Dict[str, Any] = {
        "executor": "cupy",
        "requested": policy,
        "cache_dir": cache_dir,
        "cache_preexisting": bool(cache_dir and os.path.isdir(cache_dir) and os.listdir(cache_dir)),
        "reasons": [],
    }
    cache_problems = cupy_cache_reasons(policy, cache_dir)
    if cache_problems:
        # Routed through ``_announce`` like every other refusal: ``strict=False``
        # is documented to warn and record ``attained=False``, and a refusal that
        # raises regardless would make that contract a lie.
        report.update(attained=False, installed=False,
                      mechanism="refused before touching CuPy",
                      reasons=[f"refusing to install the {policy!r} subnormal "
                               f"policy for CuPy: " + "; ".join(cache_problems)])
        _STATE["executors"]["cupy"] = report
        _announce(report, strict=strict)
        return report

    try:
        compiler = _cupy_compiler(cupy)
    except Exception as exc:  # pragma: no cover - a CuPy without its compiler
        report.update(attained=False, installed=False, mechanism="unavailable",
                      reasons=[f"cupy.cuda.compiler is not importable: {exc}"])
        _STATE["executors"]["cupy"] = report
        _announce(report, strict=strict)
        return report

    _cupy_policy_lock(compiler, policy)
    guarded = backends.guard_kernel_compilation(cupy)
    report["compile_guard_installed"] = bool(guarded)
    memo = _clear_cupy_kernel_memo(cupy)
    report["kernel_memo"] = memo
    reductions = _cupy_reduction_accelerators(policy, cupy)
    report["reduction_accelerators"] = reductions
    reasons: List[str] = list(reductions["reasons"])
    if not memo.get("cleared"):
        # THE DROP IS THE MECHANISM, so a drop that did not happen is a policy
        # that is not in force — and this used to be recorded and then left out of
        # the verdict, so ``attained`` read True. Measured (this module's own
        # note): ``cp.multiply`` compiled BEFORE a "keep" install kept flushing
        # afterwards while the install reported attained=True. The driver compiles
        # many CuPy kernels during setup, before the first step, so an undropped
        # memo means those keep flushing while Triton keeps — the split, with the
        # gate reporting success.
        reasons.append(
            f"CuPy's in-process compiled-kernel memo was NOT dropped "
            f"({memo.get('mechanism')}): kernels compiled before this install stay "
            f"memoized above both the disk cache and the seam the strip installs "
            f"at, so they keep obeying whatever policy they were built under")

    if policy == FLUSH:
        if not guarded:
            reasons.append(
                "backends.guard_kernel_compilation could not guarantee that host "
                "flushing is suspended for the duration of an NVRTC compile "
                "(either this CuPy exposes none of "
                f"{', '.join(_COMPILER_ENTRY_POINTS)} or FE_DFL_ENV is not "
                "addressable here), and 'flush' deliberately turns host flushing "
                "ON — so the first cold kernel including CCCL's <cuda/std/limits> "
                "would die on __FLT_DENORM_MIN__")
        report.update(
            attained=not reasons,
            installed=False,
            mechanism="compiles native — CuPy 13.5.1 appends '-ftz=true' to every "
                      "NVRTC compile (cupy/cuda/compiler.py:552, "
                      "_compile_with_cache_cuda); the flush policy adds no option "
                      "override, switches the CUB reduction accelerator off, and "
                      "drops the in-process kernel memo",
            exceptions=[EXECUTOR_OP_EXCEPTIONS["cupy_cub_reductions"]],
            reasons=reasons,
        )
        _STATE["executors"]["cupy"] = report
        _announce(report, strict=strict)
        return report

    state = _STATE.setdefault("_cupy_strip", {"entry_points": {}, "example_options": []})
    found: List[str] = []
    for name in _COMPILER_ENTRY_POINTS:
        original = getattr(compiler, name, None)
        if original is None:
            continue
        found.append(name)
        if getattr(original, _POLICY_MARK, None) is None:
            _STATE.setdefault("_cupy_originals", {}).setdefault(name, (compiler, original))
            setattr(compiler, name, _wrap_compiler_entry_point(original, state, name))

    if not found:
        reasons.append(
            "this CuPy exposes none of the compiler entry points the strip "
            f"installs at ({', '.join(_COMPILER_ENTRY_POINTS)}): the seam has moved")
    report.update(
        attained=not reasons,
        installed=bool(found),
        wrapped_entry_points=found,
        mechanism="cupy.cuda.compiler.{" + ",".join(found) + "} wrapped; "
                  "'-ftz=true' filtered from every option tuple (never replaced "
                  "by '-ftz=false': NVRTC rejects a duplicate -ftz option); "
                  "policy-suffixed private CUPY_CACHE_DIR because the cache key "
                  "is computed above the seam",
        reasons=reasons,
    )
    _STATE["executors"]["cupy"] = report
    _announce(report, strict=strict)
    return report


def cupy_strip_counters() -> Dict[str, int]:
    """Total NVRTC/NVCC calls seen and option tuples stripped, across entry points."""
    state = _STATE.get("_cupy_strip") or {"entry_points": {}}
    calls = sum(int(c["calls"]) for c in state["entry_points"].values())
    removed = sum(int(c["removed"]) for c in state["entry_points"].values())
    return {"calls": calls, "removed": removed}


# ---------------------------------------------------------------------------
# Triton: the LLVM per-function attribute, and the PTX audit that enforces it
# ---------------------------------------------------------------------------

#: An attribute-group DEFINITION. Deliberately does not anchor at end of line: a
#: line with anything after the closing brace (a trailing ``;`` comment) is still
#: a definition, and failing to recognize it made the injector append a SECOND
#: definition of the same group — invalid IR. The brace is located separately.
_ATTRIBUTE_GROUP_RE = re.compile(r"^attributes\s+#(\d+)\s*=\s*\{")


def _scan_outside_strings(line: str, start: int, stop: int, wanted: str) -> int:
    """Index of the first ``wanted`` character in ``line[start:stop]``, ignoring
    anything inside a double-quoted LLVM string (``section "!x"``, ``@"na#me"``)."""
    index, in_string = start, False
    while index < stop:
        character = line[index]
        if in_string:
            if character == "\\":
                index += 2
                continue
            if character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif character == wanted:
            return index
        index += 1
    return -1


def _define_attribute_slot(line: str) -> Tuple[int, List[int]]:
    """Where a function attribute group belongs on this ``define`` line, and which
    groups it already references.

    THIS IS THE WHOLE POINT OF THE FUNCTION. LLVM's grammar puts function
    attributes BEFORE metadata attachments::

        define <ret> @f(<args>) [unnamed_addr] [#N] [section ..] (!name !M)* {

    Inserting ``#N`` before the body's ``{`` — i.e. AFTER ``!dbg !7`` — produces
    text LLVM refuses with ``expected '{' in function body``, and the refusal is
    ``llvm::report_fatal_error``: an abort inside the compiler process, not a
    Python exception, so nothing can catch it or report it. Triton 3.1.0 emits
    exactly that shape (no attribute group, ``!dbg`` attachment) for every kernel,
    so this is the only path that ever runs on real input.

    Returns ``(insertion index, referenced group numbers)``; the index is the
    first metadata attachment after the parameter list, or the body brace when
    there is none.
    """
    brace = line.rfind("{")
    if brace < 0:
        return -1, []
    opening = _scan_outside_strings(line, 0, brace, "(")
    end_of_params = -1
    if opening >= 0:
        depth, index = 0, opening
        while index < brace:
            character = line[index]
            if character == '"':  # skip a quoted string wholesale
                closing = _scan_outside_strings(line, index + 1, brace, '"')
                index = brace if closing < 0 else closing + 1
                continue
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth == 0:
                    end_of_params = index + 1
                    break
            index += 1
    tail_start = end_of_params if end_of_params >= 0 else 0
    referenced = [int(number) for number in
                  re.findall(r"#(\d+)", line[tail_start:brace])]
    metadata = _scan_outside_strings(line, tail_start, brace, "!")
    return (metadata if metadata >= 0 else brace), referenced


def inject_denormal_attribute(llir: str) -> Tuple[str, Dict[str, int]]:
    """Put ``denormal-fp-math-f32`` on every function defined in this LLVM IR text.

    Text, not the LLVM C API, because Triton's ``llir`` stage hands the next stage
    a STRING and ``make_ptx`` re-parses it — so the text is the artifact.

    Composes rather than replaces: an existing attribute group gains the attribute
    (its other attributes are untouched), and a ``define`` with no group at all
    gets a freshly minted one, placed where LLVM's grammar accepts it (see
    :func:`_define_attribute_slot`). A function that already carries the attribute
    is left alone, so this is idempotent.
    """
    lines = llir.splitlines()
    groups: Dict[int, int] = {}   # group number -> line index of its definition
    highest = -1
    for index, line in enumerate(lines):
        stripped = line.strip()
        match = _ATTRIBUTE_GROUP_RE.match(stripped)
        if match and "}" in stripped:
            number = int(match.group(1))
            groups[number] = index
            highest = max(highest, number)

    referenced: set = set()
    minted: List[Tuple[int, int, int]] = []  # (line index, new group number, slot)
    defines = 0
    for index, line in enumerate(lines):
        if not line.lstrip().startswith("define "):
            continue
        defines += 1
        slot, found = _define_attribute_slot(line)
        if found:
            referenced |= set(found)
            continue
        if slot < 0:  # A define whose body brace is on a later line: nothing to do.
            continue
        highest += 1
        minted.append((index, highest, slot))

    extended = 0
    for number in sorted(referenced):
        index = groups.get(number)
        if index is None:  # A referenced group with no definition: mint it below.
            lines.append(f"attributes #{number} = {{ {TRITON_DENORMAL_ATTRIBUTE} }}")
            extended += 1
            continue
        if "denormal-fp-math-f32" in lines[index]:
            continue
        head, _, tail = lines[index].rpartition("}")
        lines[index] = f"{head.rstrip()} {TRITON_DENORMAL_ATTRIBUTE} }}{tail}"
        extended += 1

    for index, number, slot in minted:
        line = lines[index]
        lines[index] = f"{line[:slot].rstrip()} #{number} {line[slot:]}"
        lines.append(f"attributes #{number} = {{ {TRITON_DENORMAL_ATTRIBUTE} }}")

    counts = {"defines": defines, "groups_extended": extended, "groups_minted": len(minted)}
    return "\n".join(lines) + ("\n" if llir.endswith("\n") else ""), counts


_REFLECT_FTZ_RE = re.compile(
    r"(!\s*\{\s*i32\s+4\s*,\s*!\"" + TRITON_REFLECT_FTZ_FLAG + r"\"\s*,\s*i32\s+)(\d+)(\s*\})")


def set_reflect_ftz(llir: str, value: int) -> Tuple[str, int]:
    """Rewrite the ``nvvm-reflect-ftz`` module flag, and say how many it rewrote.

    This is the ONLY knob over libdevice. Triton calls ``set_nvvm_reflect_ftz``
    unconditionally, so ``__nvvm_reflect("__CUDA_FTZ")`` resolves to 1 and
    ``tl.math.div_rn`` lowers to ``div.rn.ftz.f32`` even under ``"keep"`` —
    measured, and measured again the other way: rewriting this one flag to 0 gives
    ``div.rn.f32`` on the same kernel through the same seam. Both
    ``llvm.nvvm.div.rn.ftz.f`` and ``llvm.nvvm.div.rn.f`` are declared in the IR
    behind the reflect branch, so the choice is live at this point.

    Returns ``(text, rewrites)``. Zero rewrites means the flag was absent, which
    is a fact the caller must record rather than assume away.
    """
    replacement, count = _REFLECT_FTZ_RE.subn(
        lambda match: f"{match.group(1)}{int(value)}{match.group(3)}", llir)
    return replacement, count


#: LLVM IR shaped exactly like what Triton 3.1.0's ``make_llir`` hands the ``ptx``
#: stage — no attribute group, a ``!dbg`` metadata attachment — captured from a
#: real compile. The install-time self-check runs the injector over this and
#: refuses to report success unless the result is well-formed, because the failure
#: mode it guards against is an ABORT inside LLVM that no exception handler sees.
TRITON_LLIR_SHAPE = (
    "; ModuleID = 'LLVMDialectModule'\n"
    "source_filename = \"LLVMDialectModule\"\n"
    "target triple = \"nvptx64-nvidia-cuda\"\n"
    "\n"
    "define void @probe_mul(ptr addrspace(1) %0, ptr addrspace(1) %1, i32 %2) "
    "local_unnamed_addr !dbg !7 {\n"
    "  %4 = fmul float 1.000000e+00, 2.000000e+00\n"
    "  ret void\n"
    "}\n"
    "\n"
    "!llvm.module.flags = !{!0}\n"
    "\n"
    "!0 = !{i32 4, !\"" + TRITON_REFLECT_FTZ_FLAG + "\", i32 1}\n"
)

#: The same grammar slot as :data:`TRITON_LLIR_SHAPE`, reduced to a module a real
#: LLVM parser will accept standalone (``!dbg`` demands a whole debug-info graph;
#: a plain metadata attachment occupies the identical position in the grammar).
_PARSEABLE_SHAPE = (
    "define void @k(float %0) local_unnamed_addr !foo !0 {\n"
    "  ret void\n"
    "}\n"
    "\n"
    "!0 = !{}\n"
)


def _llvm_parser() -> Any:
    """``llvmlite.binding`` if it is importable and usable, else None.

    Optional on purpose: ``llvmlite`` is not a dependency of this package (it
    arrives with numba). Where it is present the verification is definitive
    because LLVM's own parser answers; where it is absent the structural check
    still runs and the report says which one decided.
    """
    if importlib.util.find_spec("llvmlite") is None:
        return None
    try:
        binding = importlib.import_module("llvmlite.binding")
    except Exception:  # pragma: no cover - a broken llvmlite is "not available"
        return None
    initialize = getattr(binding, "initialize", None)
    if callable(initialize):
        try:
            initialize()
        except Exception:
            pass  # llvmlite >= 0.45 initializes LLVM itself and deprecates the call.
    return binding if hasattr(binding, "parse_assembly") else None


def parse_llvm_error(text: str, parser: Any = None) -> Optional[str]:
    """``None`` if LLVM parses this IR, else the parse error. Requires llvmlite."""
    parser = parser or _llvm_parser()
    if parser is None:
        return None
    try:
        module = parser.parse_assembly(text)
    except Exception as exc:
        return str(exc)[:400]
    close = getattr(module, "close", None)
    if callable(close):
        close()
    return None


def verify_reflect_rewrite(llir_shape: str = TRITON_LLIR_SHAPE) -> Dict[str, Any]:
    """Check that the ``"keep"`` policy's llir rewrite fires and stays parseable.

    Keep's verdict is measured for the same reason flush's is: it now EDITS the
    IR, so "the hook is installed" is not "the policy is attained". A rewrite that
    matched nothing would leave libdevice flushing under a policy named ``keep``.
    """
    text, rewrites = set_reflect_ftz(llir_shape, _REFLECT_FTZ_VALUE[KEEP])
    checks = [{"check": f"the {TRITON_REFLECT_FTZ_FLAG} module flag is rewritten",
               "ok": rewrites == 1, "rewrites": rewrites}]
    reasons: List[str] = []
    if rewrites != 1:
        reasons.append(
            f"the {TRITON_REFLECT_FTZ_FLAG} module flag was not found in "
            f"Triton-shaped IR, so {KEEP!r} cannot stop libdevice flushing and "
            f"tl.math.div_rn would keep emitting div.rn.ftz.f32 under a policy "
            f"named 'keep'")
    parser = _llvm_parser()
    verified_by = "rewrite count only (llvmlite not importable)"
    if parser is not None:
        error = parse_llvm_error(text.replace("!dbg !7 ", ""), parser)
        checks.append({"check": "llvmlite parses the rewritten module",
                       "ok": error is None, **({} if error is None else {"error": error})})
        verified_by = "llvmlite parse + rewrite count"
        if error is not None:
            reasons.append(f"LLVM's own parser rejected the rewritten IR: {error}")
    return {"ok": not reasons, "verified_by": verified_by,
            "checks": checks, "counts": {"reflect_rewrites": rewrites}, "reasons": reasons}


def verify_injection(llir_shape: str = TRITON_LLIR_SHAPE) -> Dict[str, Any]:
    """Check that what the injector produces is IR LLVM will actually accept.

    Why this exists as an INSTALL-TIME step rather than a test: LLVM rejects a
    misplaced attribute group with ``report_fatal_error`` — ``LLVM ERROR`` and
    ``abort()``, exit 134 — from inside Triton's compiler. There is no exception
    to catch, no refusal to report and no artifact to inspect afterwards. So the
    mechanism is verified against real-shaped input BEFORE the hook is armed, and
    a failure is a refusal at install time, which is a thing a user can read.

    Two checks. The placement check is structural and always runs: the minted
    ``#N`` must land before the metadata attachment. The parse check runs whenever
    ``llvmlite`` is importable and is definitive — it hands the text to LLVM's own
    parser. ``llvmlite`` is not a dependency of this package; its absence
    downgrades the verification, it does not fail it.
    """
    checks: List[Dict[str, Any]] = []
    reasons: List[str] = []

    injected, counts = inject_denormal_attribute(llir_shape)
    define = next((line for line in injected.splitlines()
                   if line.lstrip().startswith("define ")), "")
    group = define.find("#")
    metadata = _scan_outside_strings(define, 0, len(define), "!")
    placed = bool(counts["groups_minted"] or counts["groups_extended"])
    ordered = group >= 0 and (metadata < 0 or group < metadata)
    checks.append({"check": "attribute group precedes the metadata attachment",
                   "ok": bool(placed and ordered), "define": define.strip()})
    if not (placed and ordered):
        reasons.append(
            f"the injected attribute group is misplaced on a Triton-shaped define: "
            f"{define.strip()!r}. LLVM's grammar puts function attributes BEFORE "
            f"metadata attachments and rejects the reverse with "
            f"\"expected '{{' in function body\" — as report_fatal_error, i.e. an "
            f"abort, so arming the hook would make every Triton compile kill the "
            f"process")

    parser = _llvm_parser()
    if parser is None:
        verified_by = "placement only (llvmlite not importable)"
    else:
        verified_by = "llvmlite parse + placement"
        for name, text in (("baseline", _PARSEABLE_SHAPE),
                           ("injected", inject_denormal_attribute(_PARSEABLE_SHAPE)[0])):
            error = parse_llvm_error(text, parser)
            checks.append({"check": f"llvmlite parses the {name} module",
                           "ok": error is None,
                           **({} if error is None else {"error": error})})
            if error is not None and name == "injected":
                reasons.append(f"LLVM's own parser rejected the injected IR: {error}")
            elif error is not None:
                verified_by = f"placement only (llvmlite could not parse the baseline: {error})"
                del checks[-1]
                break

    return {"ok": not reasons, "verified_by": verified_by,
            "checks": checks, "counts": counts, "reasons": reasons}


def mechanism_digest(policy: str) -> str:
    """A short digest of WHAT the injection does, for Triton's cache key.

    Triton keys its disk cache on ``backend.hash()``. Folding only the policy NAME
    in is not enough: measured, four different injected attribute strings compiled
    into one ``TRITON_CACHE_DIR`` all returned the FIRST variant's PTX, because
    the key never changed. Everything that changes the emitted bytes therefore
    goes into the digest.
    """
    material = "|".join((policy, INJECTION_REVISION, TRITON_DENORMAL_ATTRIBUTE,
                         TRITON_REFLECT_FTZ_FLAG,
                         str(_REFLECT_FTZ_VALUE.get(policy))))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]


#: PTX opcodes whose f32 form is arithmetic this policy governs. ``mov``, ``ld``,
#: ``st`` and ``cvt`` are deliberately absent: they are not arithmetic, and
#: ``cvt``'s flush behaviour is its own modifier, not the denormal attribute's.
AUDITED_OPCODES = frozenset({
    "add", "sub", "mul", "fma", "mad", "neg", "abs", "min", "max", "setp",
    "div", "rcp", "sqrt", "rsqrt", "ex2", "lg2", "sin", "cos", "tanh",
})

#: Opcodes whose ``.ftz`` is decided by ``set_nvvm_reflect_ftz`` — which Triton
#: 3.1.0 calls UNCONDITIONALLY (``backends/nvidia/compiler.py:221``) and which
#: routes libdevice through its FTZ variants — rather than by the per-function
#: denormal attribute. Under ``"keep"`` this module now rewrites that module flag
#: to 0, so these opcodes ARE governed and are audited like any other; the set is
#: the exemption used only when the rewrite could not be applied (the flag was
#: absent from the IR), where refusing would refuse a kernel over a knob that was
#: not there to turn. It is never an exemption under FLUSH, which is what makes an
#: inline-asm ``div.full.f32`` from a future plain ``/`` fail the audit instead of
#: quietly keeping subnormals among flushing neighbours.
LIBDEVICE_REFLECT_OPCODES = frozenset({
    "div", "rcp", "sqrt", "rsqrt", "ex2", "lg2", "sin", "cos", "tanh"})

#: A PTX instruction may be guarded by a predicate — ``@%p1 mul.rn.f32 ...`` or
#: ``@!%p2 ...``. It is an INSTRUCTION, not a directive, and skipping the line
#: because it starts with ``@`` hid exactly the class of instruction the auditor
#: exists to catch: one that keeps subnormals among flushing neighbours.
_PREDICATE_RE = re.compile(r"^@!?%\w+\s+")


def audit_ptx(ptx: str, policy: str, *, reflect_ftz_governed: bool = True) -> Dict[str, Any]:
    """Every audited f32 instruction in this PTX, and which ones defy ``policy``.

    Returns counts plus a ``violations`` list of ``(line number, instruction)``.
    Under ``"flush"`` an audited f32 instruction without ``.ftz`` is a violation;
    under ``"keep"`` one WITH ``.ftz`` is — including the libdevice-routed family,
    unless ``reflect_ftz_governed`` is False, which is how the caller says "the
    ``nvvm-reflect-ftz`` flag was not in this IR, so those opcodes were never mine
    to decide".

    Predicated instructions are audited. So are the packed ``.f32x2`` opcodes: the
    type token is matched as ``f32`` or ``f32``-prefixed, not by exact equality.
    """
    audited = 0
    with_ftz = 0
    violations: List[Tuple[int, str]] = []
    by_opcode: Dict[str, int] = {}
    exempt = frozenset() if (policy == FLUSH or reflect_ftz_governed) else LIBDEVICE_REFLECT_OPCODES
    for number, raw in enumerate(ptx.splitlines(), start=1):
        line = raw.split("//", 1)[0].strip()
        if not line or line.startswith((".", "{", "}")):
            continue
        body = _PREDICATE_RE.sub("", line)
        if body.startswith("@"):  # An unrecognized ``@`` form: not an instruction.
            continue
        token = body.split()[0].rstrip(",;") if body.split() else ""
        parts = token.split(".")
        if len(parts) < 2 or not any(part.startswith("f32") for part in parts):
            continue
        opcode = parts[0]
        if opcode not in AUDITED_OPCODES:
            continue
        audited += 1
        by_opcode[token] = by_opcode.get(token, 0) + 1
        ftz = "ftz" in parts
        with_ftz += int(ftz)
        if policy == FLUSH and not ftz:
            violations.append((number, line))
        elif policy == KEEP and ftz and opcode not in exempt:
            violations.append((number, line))
    return {
        "audited": audited,
        "with_ftz": with_ftz,
        "missing_ftz": audited - with_ftz,
        "by_opcode": by_opcode,
        "violations": violations,
    }


def triton_cache_reasons(policy: str, cache_dir: Optional[str], hash_folded: bool) -> List[str]:
    """Why this ``TRITON_CACHE_DIR`` may not be used under ``policy``, by name.

    Unlike CuPy's, Triton's cache key includes ``backend.hash()``, which this
    module folds the policy AND the mechanism digest into — so when the fold is in
    place the key already separates the policies and the directory needs no
    convention. The refusal exists for the case where the fold could not be
    installed (Triton moved the seam), which is precisely when the directory is
    the only separator left.

    BOTH policies need the fold. ``"keep"`` used to be Triton's native behaviour
    and could share a cache with a pre-policy process; it no longer is, because
    keep now rewrites the ``nvvm-reflect-ftz`` module flag to stop libdevice
    flushing, and that changes the emitted PTX.
    """
    if hash_folded:
        return []
    if not cache_dir:
        return [
            "the policy could not be folded into CUDABackend.hash(), so Triton's "
            "cache key does not carry it, and TRITON_CACHE_DIR is unset: a "
            "private cache directory is the only remaining separator between "
            "flush-policy and keep-policy binaries"
        ]
    if policy not in cache_dir:
        return [
            f"the policy could not be folded into CUDABackend.hash() and "
            f"TRITON_CACHE_DIR {cache_dir!r} does not name the {policy!r} policy: "
            f"binaries compiled under the other policy would be served under it"
        ]
    return []


def _executor_present(name: str) -> bool:
    """Is ``name`` importable, with a RAISE read as PRESENT rather than absent?

    ``importlib.util.find_spec`` RAISES ``ValueError`` on a module object whose
    ``__spec__`` is ``None``. Unguarded, that turned a policy DECISION into an
    exception out of the middle of :func:`install_subnormal_policy` — measured,
    as ``ValueError('triton.__spec__ is None')`` — after the bookkeeping had
    already been written, which is the state finding 8 describes. Reading it as
    "absent" would be worse: a module present enough to be found and interrogated
    is a module that runs, and skipping it is the split this module exists to
    prevent. So: found means present, and unanswerable means present too.
    """
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:  # noqa: BLE001 - see the docstring
        return name in sys.modules


def _cuda_backend_class() -> Any:
    """Triton's NVIDIA ``CUDABackend`` CLASS, reached through the backend registry.

    ``triton.backends.backends["nvidia"].compiler`` is the class the compiler
    actually instantiates. Importing ``triton.backends.nvidia.compiler`` directly
    can bind a DIFFERENT class object, and hooks installed on that one never fire.
    """
    from triton.backends import backends as registry  # noqa: PLC0415

    return registry["nvidia"].compiler


def _wrap_add_stages(backend_class: Any, policy: str, state: Dict[str, Any]) -> bool:
    """Wrap ``CUDABackend.add_stages`` so every compile carries the policy.

    Triton 3.1.0's ``CUDAOptions`` has no ftz field, so ``add_stages`` — where the
    stage callables are installed — is the only interception point. This wraps
    what the original produced rather than replacing it: the ``llir`` stage's
    OUTPUT is rewritten (the denormal attribute under flush, the
    ``nvvm-reflect-ftz`` module flag under keep), and the ``ptx`` stage's output is
    audited under both policies.

    A seam already carrying a DIFFERENT policy is refused, not overwritten: the
    idempotence marker records which policy armed it, and an install that found
    someone else's wrapper and reported success would leave the process running
    one policy's stages under the other policy's name.
    """
    original = getattr(backend_class, "add_stages", None)
    if original is None:
        return False
    installed = getattr(original, _POLICY_MARK, None)
    if installed is not None:
        if installed != policy:
            raise SubnormalPolicyLocked(
                f"triton's CUDABackend.add_stages already carries the "
                f"{installed!r} subnormal policy and {policy!r} was requested. The "
                f"wrapper is a class attribute and its binaries are already in "
                f"Triton's cache; choose the policy before the first compile, in "
                f"a fresh process.")
        return True

    @functools.wraps(original)
    def add_stages(self, stages, options, *args, **kwargs):
        result = original(self, stages, options, *args, **kwargs)
        # Per-COMPILE context: what the llir stage found is what the ptx stage is
        # allowed to assume. ``add_stages`` runs once per compile, so this closure
        # is not shared between kernels.
        context = {"reflect_governed": policy == FLUSH}
        if "llir" in stages:
            inner_llir = stages["llir"]

            def llir_stage(src, metadata):
                text = inner_llir(src, metadata)
                if policy == FLUSH:
                    text, counts = inject_denormal_attribute(text)
                    state["llir_defines"] += counts["defines"]
                    state["llir_groups_extended"] += counts["groups_extended"]
                    state["llir_groups_minted"] += counts["groups_minted"]
                text, rewrites = set_reflect_ftz(text, _REFLECT_FTZ_VALUE[policy])
                context["reflect_governed"] = bool(rewrites) or policy == FLUSH
                state["llir_compiles"] += 1
                state["reflect_rewrites"] += rewrites
                state["reflect_flag_absent"] += int(rewrites == 0)
                return text

            stages["llir"] = llir_stage
        if "ptx" in stages:
            inner_ptx = stages["ptx"]

            def ptx_stage(src, metadata):
                text = inner_ptx(src, metadata)
                audit = audit_ptx(text, policy,
                                  reflect_ftz_governed=context["reflect_governed"])
                state["ptx_compiles"] += 1
                state["ptx_audited"] += audit["audited"]
                state["ptx_with_ftz"] += audit["with_ftz"]
                if audit["violations"]:
                    state["ptx_violations"] += len(audit["violations"])
                    sample = "; ".join(f"line {n}: {t}" for n, t in audit["violations"][:6])
                    raise SubnormalPolicyUnattainable(
                        f"the {policy!r} subnormal policy is not uniform in this "
                        f"kernel's PTX: {len(audit['violations'])} of "
                        f"{audit['audited']} audited f32 arithmetic instructions "
                        f"disagree with it. {sample}. Under {FLUSH!r} the usual "
                        f"cause is an operation LLVM never sees — a plain '/' on "
                        f"floats lowers to 'div.full.f32' as inline asm — which "
                        f"would keep subnormals while every neighbouring "
                        f"instruction flushes. Rewrite it (tl.math.div_rn is the "
                        f"shipped spelling) or extend the policy deliberately."
                    )
                return text

            stages["ptx"] = ptx_stage
        return result

    setattr(add_stages, _POLICY_MARK, policy)
    _STATE.setdefault("_triton_originals", {}).setdefault(
        "add_stages", (backend_class, original))
    backend_class.add_stages = add_stages
    return True


def _wrap_backend_hash(backend_class: Any, policy: str) -> bool:
    """Fold the policy AND the mechanism into Triton's cache key via ``CUDABackend.hash``.

    Triton's key is ``triton_key + source hash + backend.hash() + options.hash()``,
    all computed ABOVE the ``add_stages`` seam. Without this, one policy's binaries
    are served to the other and the divergence is both invisible and irreproducible.

    Folded under BOTH policies. ``"keep"`` was Triton's native behaviour when this
    module only injected under flush; it is not any more, because keep rewrites
    the ``nvvm-reflect-ftz`` module flag and that changes the PTX. And the suffix
    carries :func:`mechanism_digest`, not just the name: measured, four different
    injected attribute strings compiled into one cache directory all returned the
    first variant's PTX, because the policy name had not changed.
    """
    original = getattr(backend_class, "hash", None)
    if original is None:
        return False
    installed = getattr(original, _POLICY_MARK, None)
    if installed is not None:
        if installed != policy:
            raise SubnormalPolicyLocked(
                f"triton's CUDABackend.hash already folds in the {installed!r} "
                f"subnormal policy and {policy!r} was requested; binaries are "
                f"already cached under the first key. Choose the policy before "
                f"the first compile, in a fresh process.")
        return True
    suffix = f"-{TRITON_HASH_PREFIX}{policy}-{mechanism_digest(policy)}"

    @functools.wraps(original)
    def hash_with_policy(self, *args, **kwargs):
        return f"{original(self, *args, **kwargs)}{suffix}"

    setattr(hash_with_policy, _POLICY_MARK, policy)
    _STATE.setdefault("_triton_originals", {}).setdefault("hash", (backend_class, original))
    backend_class.hash = hash_with_policy
    return True


def _clear_triton_jit_caches() -> Dict[str, int]:
    """Drop every live ``JITFunction``'s IN-PROCESS compiled-kernel cache.

    ``JITFunction.run`` keys its own dict on specialization plus options, ABOVE
    the ``compile()`` call where ``backend.hash()`` is consulted — so folding the
    policy into the cache key separates the DISK cache and nothing else. Measured:
    a kernel launched before a ``"flush"`` install produced identical bytes after
    it, with ``triton_counters()`` still reading zero compiles; clearing the
    kernel's cache forced the recompile that revealed the injection was broken.

    The instances are reached through the garbage collector because Triton keeps
    no registry of them; this runs once, at install.
    """
    found = dropped = 0
    try:
        import gc  # noqa: PLC0415 - only needed on this path

        jit_module = importlib.import_module("triton.runtime.jit")
        jit_class = getattr(jit_module, "JITFunction", None)
        if jit_class is None:
            return {"jit_functions": 0, "entries_dropped": 0, "reachable": 0}
        for obj in gc.get_objects():
            if not isinstance(obj, jit_class):
                continue
            found += 1
            cache = getattr(obj, "cache", None)
            if not isinstance(cache, dict):
                continue
            for per_device in list(cache.values()):
                if isinstance(per_device, dict):
                    dropped += len(per_device)
                    per_device.clear()
    except Exception:  # pragma: no cover - a Triton whose jit module moved
        return {"jit_functions": found, "entries_dropped": dropped, "reachable": 0}
    return {"jit_functions": found, "entries_dropped": dropped, "reachable": 1}


def install_triton_policy(policy: str, *, strict: bool = True) -> Dict[str, Any]:
    """Make Triton's device binaries obey ``policy``.

    Under ``"flush"`` the LLVM denormal attribute is injected into the ``llir``
    stage's output and the audit becomes the refusal that keeps the attribute's
    completeness an invariant rather than an accident. Under ``"keep"`` Triton's
    arithmetic is already IEEE (3.1.0 emits ``mul.rn.f32`` with ``ptx_ftz=0``,
    under fusion on and off) but libdevice is NOT — ``set_nvvm_reflect_ftz`` is
    called unconditionally — so keep rewrites that module flag to 0 and audits the
    result. Both policies fold into the cache key and drop the in-process caches.

    THE VERDICT IS MEASURED, NOT STRUCTURAL. "The hook is installed" and "the
    policy is attained" are different claims: an earlier revision reported
    ``attained=True`` for a mechanism that aborted the process on the first
    compile. :func:`verify_injection` runs the rewrite over Triton-shaped IR — and
    hands it to LLVM's own parser when ``llvmlite`` is importable — BEFORE the
    hook is armed, and a failure there is a refusal.
    """
    if not _executor_present("triton"):
        report = {"executor": "triton", "requested": policy, "attained": True,
                  "mechanism": "none (Triton not installed; the device policy "
                               "governs device binaries only)",
                  "installed": False, "reasons": []}
        _STATE["executors"]["triton"] = report
        return report
    try:
        backend_class = _cuda_backend_class()
    except Exception as exc:
        report = {"executor": "triton", "requested": policy, "attained": False,
                  "installed": False, "mechanism": "unavailable",
                  "reasons": [f"triton.backends.backends['nvidia'].compiler is not "
                              f"reachable, so the only interception point this "
                              f"policy has does not exist: {exc}"]}
        _STATE["executors"]["triton"] = report
        _announce(report, strict=strict)
        return report

    state = _STATE.setdefault("_triton", {
        "llir_compiles": 0, "llir_defines": 0, "llir_groups_extended": 0,
        "llir_groups_minted": 0, "reflect_rewrites": 0, "reflect_flag_absent": 0,
        "ptx_compiles": 0, "ptx_audited": 0, "ptx_with_ftz": 0, "ptx_violations": 0,
    })

    verification = verify_injection() if policy == FLUSH else verify_reflect_rewrite()
    reasons: List[str] = list(verification["reasons"])
    if not verification["ok"]:
        report = {"executor": "triton", "requested": policy, "attained": False,
                  "installed": False, "mechanism": "refused before arming the hook",
                  "verification": verification, "reasons": reasons}
        _STATE["executors"]["triton"] = report
        _announce(report, strict=strict)
        return report

    hash_folded = _wrap_backend_hash(backend_class, policy)
    staged = _wrap_add_stages(backend_class, policy, state)
    dropped = _clear_triton_jit_caches()

    cache_dir = os.environ.get("TRITON_CACHE_DIR", "")
    reasons += triton_cache_reasons(policy, cache_dir, hash_folded)
    if not staged:
        reasons.append(
            "triton's CUDABackend exposes no add_stages: the only interception "
            "point this policy has (CUDAOptions carries no ftz field) is gone — "
            "this is a Triton version event, treat it as a correctness event"
        )

    version = getattr(importlib.import_module("triton"), "__version__", "unknown")
    report = {
        "executor": "triton",
        "requested": policy,
        "attained": not reasons,
        "installed": bool(staged),
        "triton_version": version,
        "cache_dir": cache_dir,
        "hash_folded": bool(hash_folded),
        "verification": verification,
        "in_process_caches_dropped": dropped,
        "exceptions": [] if policy == FLUSH else [EXECUTOR_OP_EXCEPTIONS["triton_libdevice"]],
        "mechanism": (
            "CUDABackend.add_stages wrapped; the llir stage's output gains "
            f"{TRITON_DENORMAL_ATTRIBUTE} on every function, and every generated "
            "PTX is audited per instruction and REFUSED if any audited f32 "
            "arithmetic instruction lacks .ftz"
            if policy == FLUSH else
            "CUDABackend.add_stages wrapped; Triton 3.1.0's own arithmetic already "
            "keeps (mul.rn.f32, ptx_ftz=0) but set_nvvm_reflect_ftz is called "
            f"unconditionally, so the llir stage rewrites the {TRITON_REFLECT_FTZ_FLAG} "
            "module flag to 0 and libdevice stops flushing (div.rn.ftz.f32 -> "
            "div.rn.f32); every generated PTX is audited per instruction and "
            "REFUSED if any audited f32 arithmetic instruction carries .ftz"
        ),
        "reasons": reasons,
    }
    _STATE["executors"]["triton"] = report
    _announce(report, strict=strict)
    return report


def triton_counters() -> Dict[str, int]:
    """Injection and audit counters accumulated since install."""
    return dict(_STATE.get("_triton") or {})


# ---------------------------------------------------------------------------
# Installation and the artifact stamp
# ---------------------------------------------------------------------------


def _announce(report: Dict[str, Any], *, strict: bool) -> None:
    """A report that did not attain its policy is a refusal, or a loud declaration."""
    if report.get("attained"):
        return
    message = (
        f"the {report['requested']!r} subnormal policy is UNATTAINABLE for the "
        f"{report['executor']!r} executor: " + "; ".join(report.get("reasons") or ["(no reason recorded)"])
    )
    if strict:
        raise SubnormalPolicyUnattainable(message)
    warnings.warn(message, RuntimeWarning, stacklevel=3)


def install_subnormal_policy(policy: Optional[str] = None, *, cupy: Any = None,
                             strict: bool = True,
                             executors: Tuple[str, ...] = ("host", "cupy", "triton"),
                             ) -> Dict[str, Any]:
    """Resolve a policy and drive every named executor to it.

    With no argument this asks :data:`MATCH_MEEP` and MEASURES the answer
    (:func:`resolve_match_meep`), so the stamp carries three things and not one:
    the question, the policy the executors were actually driven to, and the
    measurement that connects them.

    ``strict`` decides what an unattainable policy does: raise
    :class:`SubnormalPolicyUnattainable` (the default — a process that asked to
    flush and quietly did not is a process whose bytes mean nothing), or emit a
    ``RuntimeWarning`` and record ``attained=False`` in the stamp.

    Idempotent for the same policy. A DIFFERENT policy after one is installed is
    refused with :class:`SubnormalPolicyLocked`: device binaries and cache
    entries already exist under the first one.

    ``executors`` NAMES THE ARMS TO DRIVE and defaults to the three an NVIDIA host
    has. ``"mps"`` is the fourth, asked for explicitly by a Metal dispatch
    (``executors=("host", "mps")``): its lever is native rather than a compiler
    seam, so it attains ``flush`` with no action and REFUSES ``keep`` by name — see
    ``metal_kernels.subnormal.install_mps_policy``. It is not in the default tuple
    because a process with no MPS device has no business importing the Metal
    package to be told an arm it never asked for was skipped.

    THE HOST GOES FIRST, AND ITS FAILURE STOPS THE REST. Under ``strict=False`` an
    earlier revision warned that the host could not flush and then installed the
    device halves anyway, manufacturing the exact configuration this module exists
    to prevent — host keeping, Triton flushing, which is measured to diverge at
    step 1 with 92 differing floats of 2880. An unattained host now aborts the
    device legs and is recorded as such.

    AND A REFUSAL PUTS THE FPU BACK. ``install_host_policy`` mutates the process
    MXCSR through MEEP's knob before anything else runs; a later refusal that
    restored only the bookkeeping left the process flushing with
    ``policy_is_installed()`` False — bytes attributable to no policy at all,
    which is the state the refusal exists to avoid.
    """
    resolution = policy_resolution(policy)
    resolved, source = resolution["policy"], resolution["source"]
    previous = _STATE["policy"]
    previous_state = {key: _STATE[key] for key in
                      ("policy", "requested", "resolution", "resolved_from")}
    host_flushing_at_entry = backends.subnormals_flushed()
    if previous is not None and previous != resolved:
        how = (f"from the {source}" if resolution["requested"] != MATCH_MEEP else
               f"from the {source}, which asked {MATCH_MEEP!r} and measured "
               f"{resolved!r}")
        raise SubnormalPolicyLocked(
            f"the {previous!r} subnormal policy is already installed in this "
            f"process and {resolved!r} was requested ({how}). Kernels "
            f"compiled and cached under {previous!r} are still reachable — CuPy's "
            f"cache key in particular is computed above the seam the strip "
            f"installs at — so a swap would serve one policy's device binaries "
            f"under the other's name. Choose the policy before the first compile, "
            f"in a fresh process."
        )
    _STATE["policy"] = resolved
    _STATE["requested"] = resolution["requested"]
    _STATE["resolution"] = resolution["resolution"]
    _STATE["resolved_from"] = source
    _bump_epoch()

    try:
        if "host" in executors:
            host = install_host_policy(resolved, strict=strict)
            if not host["attained"]:
                # strict=True already raised; this is the strict=False path.
                for name in ("cupy", "triton", "mps"):
                    if name in executors:
                        _STATE["executors"][name] = _skipped_after_host(resolved, name)
                _rollback(previous_state, host_flushing_at_entry)
                return policy_stamp()
        if "cupy" in executors:
            install_cupy_policy(resolved, cupy, strict=strict)
        if "triton" in executors:
            install_triton_policy(resolved, strict=strict)
        if "mps" in executors:
            # THE ONE ARM WHOSE LEVER LIVES IN ANOTHER MODULE, and the import is
            # lazy for the reason the module docstring gives: this file is
            # importable with no CuPy, no Triton and no MEEP, and torch belongs on
            # that list. ``metal_kernels.subnormal`` imports no torch either, but
            # the executor table must not acquire a Metal-package dependency at
            # module scope just to name an arm most processes never ask for.
            from .metal_kernels.subnormal import install_mps_policy  # noqa: PLC0415

            install_mps_policy(resolved, strict=strict)
    except BaseException:
        # ANY escape, not only :class:`SubnormalPolicyUnattainable`. The bookkeeping
        # above is written BEFORE anything is driven, so an install that dies from
        # some other cause used to leave the process LOCKED to a policy no executor
        # had been driven to — ``policy_is_installed()`` True, ``policy_stamp()``
        # reading 'ieee_keep_ftz_stripped' with only ['host'] in ``executors``, a
        # later ``install_subnormal_policy('flush')`` refused as Locked, and the FPU
        # left where the aborted install put it. That is exactly "an artifact said a
        # policy was covered when no executor had been driven to it", reached by a
        # different door. The doors are real and unguarded: ``install_cupy_policy``
        # calls ``os.listdir`` on the cache directory (OSError on a write-but-not-
        # read directory) and ``install_triton_policy`` calls
        # ``importlib.util.find_spec`` unguarded, which raises ValueError on a
        # module whose ``__spec__`` is None.
        #
        # ``_rollback`` is itself conditional — it returns without touching anything
        # if a device seam DID install — so a partial install still keeps its lock,
        # which is the behaviour that was always correct.
        _rollback(previous_state, host_flushing_at_entry)
        raise
    return policy_stamp()


def _skipped_after_host(policy: str, executor: str) -> Dict[str, Any]:
    """The record a device leg gets when the host could not attain the policy.

    Not installed, and not silently "attained because there was nothing to do":
    a half-applied policy is the failure mode, so the leg that never ran says so.
    """
    return {"executor": executor, "requested": policy, "attained": False,
            "installed": False, "mechanism": "not attempted",
            "reasons": ["the host executor did not attain the policy, and a "
                        "device half installed on top of a host that keeps (or "
                        "flushes) differently is the half-applied policy this "
                        "module exists to prevent"]}


def _rollback(previous_state: Dict[str, Any], host_flushing_at_entry: bool) -> None:
    """Undo a refused install: the bookkeeping AND the process FPU.

    A refusal that installed no DEVICE COMPILER seam leaves the process unlocked,
    so the caller can ask for the other policy instead of being told the refused
    one is already in force. The host FPU is re-drivable and does not lock the
    policy — but it must actually be re-driven, which is the half an earlier
    revision skipped.

    ``mps`` IS NAMED IN THE SEAM CHECK AND REPORTS ``installed`` FALSE TODAY, which
    is not a contradiction: Metal's flushing is native, so that arm wraps no
    compiler seam and memoizes no binary under a policy, and there is nothing a
    rollback could fail to undo. It is named anyway because the question the check
    asks is "did any DEVICE executor install something a rollback cannot recall",
    and an arm answering that question by being absent from the tuple would be
    answering it by accident.

    ``previous_state`` is the whole snapshot — policy, request, resolution and
    source — because they are one fact in four keys: leaving the refused install's
    ``requested`` behind would have the stamp report a question that nothing
    answered.
    """
    if any(_STATE["executors"].get(name, {}).get("installed")
           for name in ("cupy", "triton", "mps")):
        return
    _STATE.update(previous_state)
    _bump_epoch()
    _restore_cupy_accelerators()
    if backends.subnormals_flushed() != host_flushing_at_entry:
        _drive_host_fpu(host_flushing_at_entry)
        host = _STATE["executors"].get("host")
        if isinstance(host, dict):
            host["fpu_restored_to"] = backends.subnormals_flushed()


def _restore_cupy_accelerators() -> bool:
    """Put CuPy's reduction accelerators back, if this module switched them off.

    Same rule as the FPU: a refused install must not leave a process-wide setting
    behind under a policy that is not installed. Switching CUB off changes no
    bytes under either policy — the NVRTC route obeys whichever one is in force —
    but it does change performance, and an unexplained residue is a defect
    whatever it costs.
    """
    entry = _STATE.pop("_cupy_accelerators", None)
    if entry is None:
        return False
    setter, before = entry
    try:
        setter(before)
    except Exception as exc:
        # A rollback runs INSIDE the handling of a refusal, so raising here
        # replaces the refusal — the thing the caller needs to read — with a
        # traceback from the cleanup. Measured: it did exactly that on device.
        _STATE["_cupy_accelerator_restore_failed"] = f"{type(exc).__name__}: {exc}"
        warnings.warn(
            f"could not put CuPy's reduction accelerators back to {before!r}: "
            f"{exc}. They are switched OFF in this process; reductions still obey "
            f"whichever policy is in force, but the CUB fast path is gone until "
            f"this process ends.", RuntimeWarning, stacklevel=2)
        return False
    return True


def set_subnormal_policy(policy: Optional[str] = None, *, strict: bool = True) -> Dict[str, Any]:
    """Choose the float32 subnormal policy for this process, and install it.

    ``meep_gpu.set_subnormal_policy("flush")`` — flush float32 subnormals to zero
    on the host, in CuPy and in Triton. This is stock MEEP's behaviour on x86;
    MEEP's own comment calls it a speed choice for the tails of exponentially
    decaying sources (meep#1708).

    ``meep_gpu.set_subnormal_policy("keep")`` — IEEE-754 everywhere. This is what
    byte-matches MEEP on hosts where MEEP itself keeps, which is every host where
    ``_set_zero_subnormals`` is the ``#if HAVE_IMMINTRIN_H`` no-op.

    ``meep_gpu.set_subnormal_policy()`` — :data:`MATCH_MEEP`: whichever of those
    MEEP itself does HERE, MEASURED (:func:`resolve_match_meep`) rather than
    assumed, so the default configuration is the one a MEEP comparison on this
    host is against. The stamp carries both the question and its answer.

    Call it BEFORE the first device kernel compiles: both device executors
    memoize compiled kernels in process, and while this drops those caches, it
    cannot recall bytes a kernel has already written.

    Returns the stamp.
    """
    return install_subnormal_policy(policy, strict=strict)


def policy_stamp() -> Dict[str, Any]:
    """The record every artifact this policy governs must carry.

    Top-level ``policy`` keeps the name the nine certified families were cut under
    when the policy is ``"keep"`` (:data:`CUPY_KEEP_POLICY_NAME`), so existing
    stamps stay comparable and the gates' ``probe_record_policy_reasons`` keeps
    accepting them — but ONLY once a policy has actually been installed. Before
    that, ``policy`` reads :data:`UNINSTALLED_POLICY_NAME`: the resolution is a
    preference, and an artifact stamped from a preference would carry a
    certification-grade label ("this run's bytes were cut under the stripped
    policy") that nothing in the process earned. Measured: with
    ``MEEP_GPU_SUBNORMAL_POLICY=keep`` and no install, the old stamp read
    ``ieee_keep_ftz_stripped`` with zero NVRTC calls and zero options stripped.

    ``requested`` IS THE QUESTION AND ``resolved`` IS THE ANSWER. A default startup
    asks :data:`MATCH_MEEP`, which is not a policy any executor can be driven to,
    so the two differ and both are recorded — with ``match_meep`` carrying the
    measurement that connects them: what was read, on what mechanism, and why it
    landed where it did. An explicit request answers itself, and ``match_meep`` is
    then ``None``, which is how a reader tells a measured policy from a demanded
    one.
    """
    if policy_is_installed():
        policy = _STATE["policy"]
        requested = _STATE["requested"] or policy
        resolution = _STATE["resolution"]
    else:
        record = policy_resolution()
        policy, requested, resolution = (
            record["policy"], record["requested"], record["resolution"])
    counters = cupy_strip_counters()
    installed_name = CUPY_KEEP_POLICY_NAME if policy == KEEP else FLUSH_POLICY_NAME
    stamp: Dict[str, Any] = {
        "policy": installed_name if policy_is_installed() else UNINSTALLED_POLICY_NAME,
        "would_be_policy": installed_name,
        "requested": requested,
        "resolved": policy,
        "match_meep": resolution,
        "resolved_from": _STATE["resolved_from"] or "not installed",
        "installed": policy_is_installed(),
        "machine": platform.machine(),
        "nvrtc_calls": counters["calls"],
        "ftz_removed": counters["removed"],
        "triton": triton_counters(),
        "executors": {name: dict(report) for name, report in _STATE["executors"].items()},
    }
    cupy_report = _STATE["executors"].get("cupy") or {}
    stamp["cache_dir"] = cupy_report.get("cache_dir", os.environ.get("CUPY_CACHE_DIR", ""))
    stamp["cache_preexisting"] = bool(cupy_report.get("cache_preexisting", False))
    stamp["unattained"] = sorted(
        name for name, report in _STATE["executors"].items() if not report.get("attained")
    )
    return stamp


def uninstall_subnormal_policy() -> Dict[str, int]:
    """Put every seam back the way it was found, and forget the policy.

    Forgetting the bookkeeping alone is not enough and was measured to be a trap:
    the Triton wrappers live on the ``CUDABackend`` CLASS, so a reset that cleared
    only ``_STATE`` left the previous policy's ``add_stages`` and ``hash`` in place
    and the NEXT install returned a stamp naming a policy the bytes would not
    obey. The host FPU is deliberately NOT restored here — the caller who wants a
    particular FPU state installs a policy that says so.
    """
    restored = int(_restore_cupy_accelerators())
    for name, (compiler, original) in (_STATE.pop("_cupy_originals", {}) or {}).items():
        setattr(compiler, name, original)
        restored += 1
    for name, (backend_class, original) in (_STATE.pop("_triton_originals", {}) or {}).items():
        setattr(backend_class, name, original)
        restored += 1
    _STATE.update({"policy": None, "requested": None, "resolution": None,
                   "resolved_from": None, "executors": {}})
    _STATE.pop("_cupy_strip", None)
    _STATE.pop("_triton", None)
    _bump_epoch()
    return {"seams_restored": restored}


def _reset_for_tests() -> None:  # Test hook: a full uninstall, seams included.
    uninstall_subnormal_policy()
