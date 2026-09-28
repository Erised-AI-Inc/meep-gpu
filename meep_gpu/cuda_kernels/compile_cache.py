"""The compiled-kernel memo and its key — with no CuPy in sight.

WHAT WAS WRONG. ``step_curl_kernels`` memoized every compiled kernel on
``(name, is_complex)`` and nothing else: not the compile options, not the source
it compiled, and not the float32 subnormal policy in force when it compiled.
``_clear_kernel_cache()`` was called by nothing in the engine and nothing in a
policy install, so a policy installed after a kernel had first compiled was
never seen by that module at all — the second caller was handed the first
caller's binary under the second caller's policy name. The certification the
track actually holds could not even be filtered by policy, because the field
did not exist anywhere (disposition §1.1, §1.2).

THE KEY IS THE FIX, AND IT IS STRONGER THAN A HOOK. The disposition asks for two
things: key the memo on ``(name, is_complex, options, policy)`` and clear it on a
policy install. A hook that clears on install has to be CALLED, and this package
may not edit :mod:`meep_gpu.subnormal_policy` to add the call — so a hook here
would be one more thing that silently does not happen. Folding the policy into
the KEY needs nobody to remember: a lookup taken under a policy the entry was not
compiled under simply misses, and the kernel is rebuilt. :func:`clear_kernel_cache`
is still exported (the bit-identity probe drives it between guard sets), but
nothing's correctness rests on it being called.

THE SOURCE IS IN THE KEY TOO, and it costs nothing. The kernel source strings are
module-level constants, so Python has already cached their hashes and a tuple
lookup compares them by identity — while a mutation harness that rewrites one
(``probe_fused_kernel_bit_identity.py``) can no longer be served the unmutated
binary even if it forgets to drop the memo. That is the same defect class the
Triton round hit three times: a leg reporting a pass for a mutation it never
applied.

WHAT THIS DOES NOT FIX, STATED HERE BECAUSE HALF A FIX READS LIKE A WHOLE ONE.
Keying this memo does NOT separate CuPy's own disk cache. CuPy computes that key
ABOVE the ``compile_using_nvrtc`` seam where ``subnormal_policy`` strips
``-ftz=true`` (``subnormal_policy.py:87-89``), so a rebuild under a new policy
can still be served the OLD policy's binary off disk. The other half is a fresh,
policy-token-carrying ``CUPY_CACHE_DIR`` per leg, which belongs to whoever runs
the gate; :func:`compile_policy_token` folds the directory in so that at least
this memo can tell two of them apart, and ``run_pml_gate_mutations.py`` takes a
private one per leg.

NOTHING HERE IMPORTS CUPY — the same reason ``coverage.py`` does not. The memo's
policy is decision logic whose failure mode is a silent wrong binary, so it is
exercised at the merge bar on a laptop (``test_compile_cache.py``) rather than
only on the one host that can launch a kernel.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

#: Token when no policy has been installed. Deliberately NOT a policy name, and
#: for the same reason ``subnormal_policy.UNINSTALLED_POLICY_NAME`` is not: with
#: nothing installed CuPy compiles natively (it appends ``-ftz=true`` itself),
#: which is ONE state whatever preference the environment expresses. A resolved
#: preference that has driven no executor changes no binary.
UNINSTALLED = "none_installed"

#: Token when :mod:`meep_gpu.subnormal_policy` could not be reached at all — the
#: kernel module loaded by path, outside the package, with ``meep_gpu`` not in
#: ``sys.modules`` (the bit-identity probe's standalone load). Recorded rather
#: than guessed: a token that silently claimed "no policy" would make two
#: genuinely different compiles share one entry.
UNAVAILABLE = "policy_module_unavailable"

#: Environment variable naming CuPy's on-disk kernel cache. Folded into the
#: token because CuPy's cache key is computed above the strip seam, so the
#: directory name is the only separator there is.
CUPY_CACHE_DIR_ENV = "CUPY_CACHE_DIR"

_compiled_kernels: Dict[Tuple[Any, ...], Any] = {}
_compile_log: List[Dict[str, Any]] = []

#: A SUCCESSFUL policy-module lookup, memoized — the module object never goes
#: stale (an install mutates its state, not its identity) and this sits on the
#: launch path. A FAILED lookup is deliberately not memoized: the standalone
#: by-path load can reach the module later, once something else imports
#: ``meep_gpu``, and caching the miss would pin the whole process to
#: :data:`UNAVAILABLE` for a reason that stopped being true.
_POLICY_MODULE: Optional[Any] = None


def _policy_module() -> Optional[Any]:
    """The subnormal-policy module, or ``None`` when it cannot be reached.

    IMPORTED, NEVER REIMPLEMENTED: the policy is one process-wide decision and a
    second reading of it here would be a second answer. Three routes, in falling
    order of directness, and the last two exist because this package's kernel
    module is ALSO loaded by path outside the package (the bit-identity probe),
    where a relative import raises.

    The by-path route deliberately stops short of ``import meep_gpu``: pulling the
    whole package in from inside a compile would break the probe's standalone
    property. If ``meep_gpu`` is already imported — which it is in any process
    that reached the real-PML leg — the submodule import is cheap and safe
    (``subnormal_policy`` imports only the standard library and ``backends``).
    A failure is reported as :data:`UNAVAILABLE` rather than assumed away.
    """
    global _POLICY_MODULE
    if _POLICY_MODULE is not None:
        return _POLICY_MODULE
    module: Optional[Any] = None
    try:
        from .. import subnormal_policy  # noqa: PLC0415 - by design, see above
        module = subnormal_policy
    except Exception:  # noqa: BLE001 - loaded by path: no parent package
        module = sys.modules.get("meep_gpu.subnormal_policy")
        if module is None and "meep_gpu" in sys.modules:
            try:
                module = importlib.import_module("meep_gpu.subnormal_policy")
            except Exception:  # noqa: BLE001 - a truncated tree; the token says so
                module = None
    _POLICY_MODULE = module
    return module


def compile_policy_token() -> str:
    """What, about this process, decides the BYTES a compile produces or serves.

    Two facts, in one string:

    * the installed float32 subnormal policy, and whether the CuPy strip is
      actually installed at the compiler front ends. The second is not implied by
      the first — under ``"flush"`` CuPy is left native and installs no wrapper —
      and it is the one that changes the option tuple NVRTC sees;
    * ``CUPY_CACHE_DIR``, because CuPy's cache key is computed above the seam the
      strip installs at, so two directories are two sets of binaries for the same
      key and one directory shared across policies is a poisoned one.

    ON THE LAUNCH PATH, AND MEASURED THERE RATHER THAN CALLED CHEAP: 0.708 us
    uninstalled and 0.736 us with a policy installed (laptop, 200 000 iterations,
    ``timeit``), of which the ``os.environ`` read is 0.311 us,
    ``executor_report("cupy")`` 0.153 us (it copies a dict), and
    ``policy_is_installed`` plus ``get_subnormal_policy`` 0.090 us together; the
    rest is the f-string. ``_get_kernel``'s other per-call work — the 14-entry
    code map and the key build — is measured in that function's docstring.

    NOT MEMOIZED, DELIBERATELY. Freezing the token per process is the staleness
    this whole key exists to remove: a policy installed after the first launch
    would not move a frozen token, and the second caller would be handed the
    first caller's binary under the second caller's policy name — the exact
    defect of the ``(name, is_complex)`` key. Two calls per timestep against
    ~5.7 ms per timestep at 160³ and the array path's measured 716.7 Mcells/s is
    2.6e-4 of a step.

    It never triggers the ``match_meep`` MEASUREMENT, because an uninstalled
    resolution is a preference and preferences compile nothing.
    """
    cache = os.environ.get(CUPY_CACHE_DIR_ENV, "")
    module = _policy_module()
    if module is None:
        return f"{UNAVAILABLE}|cache={cache}"
    if not module.policy_is_installed():
        return f"{UNINSTALLED}|cache={cache}"
    stripped = bool(module.executor_report("cupy").get("installed"))
    return (f"{module.get_subnormal_policy()}|cupy_ftz_stripped={stripped}"
            f"|cache={cache}")


def kernel_cache_key(name: str, is_complex: bool, options: Sequence[str],
                     source: str, policy_token: Optional[str] = None) -> Tuple[Any, ...]:
    """``(name, is_complex, options, policy, source)`` — every axis that moves bytes.

    ``options`` is what the KERNEL MODULE passes. A harness that substitutes the
    compiler itself and overrides the option tuple from outside (the probe's
    ``CupyShim``) is invisible here by construction, which is why that harness
    drops the memo on every guard change; ``test_pml_gate_harness.py`` pins that
    it does.

    ``policy_token`` is taken from :func:`compile_policy_token` when not given.
    It is a parameter so a test can pin two tokens against one another without
    installing a process-wide policy, and so a caller that already has the token
    does not read the environment twice.
    """
    token = compile_policy_token() if policy_token is None else policy_token
    return (str(name), bool(is_complex), tuple(options), token, source)


def get_or_compile(key: Tuple[Any, ...], factory: Callable[[], Any]) -> Any:
    """The memo. ``factory`` runs exactly once per distinct key, and is logged.

    The log is the armed-mutation accounting the disposition asks for (§1.5): it
    records the sha256 of the source ACTUALLY handed to the compiler, so a gate
    leg can assert that the mutated bytes reached it rather than that the
    mutation matched some text. A leg reporting a pass for a mutation it never
    applied is a defect this project hit three times on the sibling track.

    WHAT ONE ENTRY IS, EXACTLY — a FACTORY INVOCATION, i.e. one ``cp.RawKernel``
    construction. It is an UPPER BOUND on NVRTC calls and not a count of them:
    CuPy's disk-cache key is computed above the seam ``subnormal_policy`` strips
    ``-ftz=true`` at, so a construction can be answered off disk with no compiler
    involved. This project measured exactly that — ``cuda_policy_reach``'s
    ``after_install__cuda_kernel__module_memo_cleared`` leg cleared the memo,
    constructed a fresh kernel, and got the same bytes back. For the ARMED
    mutations this log exists to account for, the two coincide: mutated source is
    a miss in that source-keyed disk cache. Calling it a compile count would have
    an auditor of a mutation leg reading the wrong quantity.
    """
    if key not in _compiled_kernels:
        _compiled_kernels[key] = factory()
        name, is_complex, options, token, source = key
        _compile_log.append({
            "name": name,
            "is_complex": bool(is_complex),
            "options": list(options),
            "policy_token": token,
            "source_sha256": hashlib.sha256(
                str(source).encode("utf-8")).hexdigest(),
            "source_chars": len(str(source)),
        })
    return _compiled_kernels[key]


def clear_kernel_cache() -> int:
    """Drop every memoized kernel; returns how many entries went.

    The COUNT is the point — a nonzero return from a caller that believed the
    cache was cold is the fingerprint of a late clear, the same signal
    ``subnormal_policy`` reports for its own in-process memo drop.
    """
    dropped = len(_compiled_kernels)
    _compiled_kernels.clear()
    return dropped


def cached_keys() -> Tuple[Tuple[Any, ...], ...]:
    """Every live memo key, for a caller that wants to see the partition."""
    return tuple(_compiled_kernels)


def cache_size() -> int:
    return len(_compiled_kernels)


def compile_log() -> Tuple[Dict[str, Any], ...]:
    """One entry per ``cp.RawKernel`` CONSTRUCTION — not per lookup — oldest first.

    See :func:`get_or_compile` for why that is deliberately not called a compile
    count: it is an upper bound on NVRTC calls, exact for mutated sources.
    """
    return tuple(dict(entry) for entry in _compile_log)


def clear_compile_log() -> int:
    dropped = len(_compile_log)
    _compile_log.clear()
    return dropped


def compiled_source_digests() -> Tuple[str, ...]:
    """The distinct source digests the compiler has been handed, in first order."""
    seen: List[str] = []
    for entry in _compile_log:
        if entry["source_sha256"] not in seen:
            seen.append(entry["source_sha256"])
    return tuple(seen)
