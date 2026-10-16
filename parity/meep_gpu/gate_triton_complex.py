"""Bit-identity gate for the complex-field (Bloch / force_complex_fields) Triton kernels.

The module under test is ``meep_gpu/triton_kernels/complex_fields.py``, and it IS
on the dispatch path: ``launch.plan_step`` calls its two coverage predicates and
``fastpath._decide`` composes through it, with ``fastpath.ARM_CERTIFICATION``
carrying the 'complex PML' and 'complex' arms over the step_B/step_D/update_H/update_E
slots. This paragraph said
"NOT wired into production dispatch (``plan_fast_path`` keeps returning None)"
until 2026-08-15, when ``plan_step`` on a complex64 Grid/Fields/PML triple was
measured emitting per-arm reasons prefixed 'complex PML: ' — so what keeps a
configuration out of a kernel is this gate plus the module's refusal enumeration,
NOT the absence of a caller, and a reader was being told otherwise.

This gate is the arbiter the module's docstring promises: every grouping choice
the kernel could not force by construction (tl.math.fma lowering, zero-cross-term
survival, helper inlining, the where-select on the wrapped plane, the curl
grouping) is HELD HERE, empirically, per platform.

WHY SUB-STEP GRANULARITY. The composition probe
(``probe_triton_complex_composition.py``) certifies complete driver steps; this
gate certifies each sub-step against an in-file transcription of the array path,
because a whole-step comparison cannot attribute a defect to a sub-step and
cannot exercise the deliberately wrong configurations (mutations) at all.

SUBNORMAL POLICY — THE STRIP IS THE SHIP CONFIGURATION. CuPy 13.5.1 appends
``-ftz=true`` to every NVRTC compile (cupy/cuda/compiler.py:552), flushing
float32 subnormals on the device; the engine ships IEEE subnormal-KEEP, the only
alignment direction (NVRTC rejects a duplicate ``-ftz`` option and Triton has no
flush knob — settled in ``results/device_subnormal_policy_2026-08-11/``).
This gate therefore installs the demonstrated strip — wrap
``cupy.cuda.compiler.compile_using_nvrtc``, filter ``-ftz=true``, private
policy-suffixed ``CUPY_CACHE_DIR`` (the cache key is computed above the seam) —
BEFORE any CuPy compile, counts the option tuples it stripped, refuses to
license when the strip cannot be confirmed exercised, and stamps every artifact
with the policy it certified under.

Legs, in order:

* ``reference``     — the transcription pinned against ``stepping.py`` itself,
  on real ``Grid``/``Fields``/``PML`` objects, byte for byte. Runs on NumPy
  (the laptop merge bar) and again on CuPy when a device is present; emits a
  per-step state sha256 chain so the two backends' bytes can be compared
  offline across machines.
* ``synthetic``     — the sweep. Shapes (3-D, one-cell z, (1,1,n)) x boundary/
  phase configurations (unphased, general Bloch with negative components, the
  exact Brillouin edge, one-axis k with the None-skip, phased x metallic,
  metallic k=0, transverse-collapsed k) x dtdx (0.5 AND the non-representable
  0.35) x guard on/off x sub-step, against the in-file complex reference.
  Seeds carry imag -0.0 planes and zero-real/negative-imag planes — the only
  rows on which several transcription defects are visible at all.
* ``constitutive``  — update_H / update_E, same treatment.
* ``multi_step``    — B->H->D->E cycles from one state; the auxiliaries are
  state, so a kernel right for one launch and wrong forever after diverges here.
* ``mutations``     — defects planted in the shipped kernel's SOURCE (compiled
  from a real file, launch-counted so a leg whose mutant never ran FAILS rather
  than reporting a hollow pass) plus host-side plan mutations. Each is caught,
  or recorded as the null it was predicted to be — never silently.
* ``engine``        — ``plan_complex_pml_curl``/``plan_complex_constitutive``
  from the engine's own objects against ``stepping.step_B``/``step_D``/
  ``update_H``/``update_E`` on real complex grids (including the exact
  Brillouin-edge phase), plus the phase-table host mutation (phase from k via
  cmath instead of ``grid.bloch_phase``) — asserted at the TABLE seam as
  complex128 bytes, where the Brillouin-edge defect is deterministic; at field
  level the generic-state case is a predicted, recorded null (the edge delta
  is sub-half-ulp of every nonzero float32 word — the m1 constraint) and the
  catch is demanded on a purely imaginary state instead, whose exactly-zero
  real words the mutated phase moves (:data:`ENGINE_PHASE_MUTATION_CASES`).

THE ALIGNED FORMS (bit_parity 0.10, WP KT5). The kernels step MEEP's forms part by
part: every product is a real coefficient times one stored part, the curl is MEEP's
left-to-right ``((g1s - g1) + g2) - g2s`` with the record's partner presence, each
cell takes the split-field recurrence or the dsig-only form by its chunk's membership
(the chunk record, ``gate_chunk_records``), the constitutive half accumulates on its
own layer's cells only, and the Bloch phase is MEEP's connection of the stored window:
plane 0 of the UNSHIFTED operand, ``conj(eikna)`` under B and ``eikna`` under D, a
copy, a negation or the unfused product as MEEP classifies it
(``complex_fields.connection_arguments``). The in-file reference below states those
forms and is pinned against ``stepping`` by the reference leg; the synthetic legs key
both sides on ``two_run_record`` so every boundary plane, a two-run line and a
lattice swap are exercised on every shape. The kernels form every complex product
part by part (one body), so the mutation battery holds them to the forms.

Every case prints one flushed line as it lands; the JSON artifact is rewritten
atomically (tmp + fsync + os.replace) after every case (the progress-reporting rule); the
gate writes its own ``provenance.json`` (source sha256s including the as-read
hash of the untouched ``fingerprints.json``) into the results directory.
Correctness only: this gate makes no throughput or timing claims.

Usage (the GPU host, one clear device; the cache dir MUST be private to the run and
carry the policy token — the gate aborts otherwise)::

    CUDA_VISIBLE_DEVICES=5 \\
    CUPY_CACHE_DIR=$RUN_ROOT/results/cupy_cache_ftz_stripped_$JOB \\
        python -u gate_triton_complex.py \\
        --out results/triton_complex_<date>/gate.json

Laptop (NumPy only; CUDA legs and the strip skip cleanly and say so)::

    python -u gate_triton_complex.py --legs reference \\
        --out /tmp/gate_local.json
"""

from __future__ import annotations

import argparse
import cmath
import hashlib
import importlib.util
import inspect
import json
import math
import os
import re
import sys
import tempfile
import textwrap
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_API = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_API not in sys.path:
    sys.path.insert(0, _REPO_API)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    import cupy as cp
except ImportError:  # Laptop leg: the reference transcription still validates.
    cp = None

try:
    import triton  # noqa: F401

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

# Everything reusable comes from the shared probe; nothing below re-implements
# the byte comparator, the ULP key, the recurrence, or the coefficient tables.
import probe_fused_kernel_bit_identity as probe  # noqa: E402
from gate_chunk_records import layer_record, region_plan_record, two_run_record  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu import subnormal_policy as _policy  # noqa: E402
from meep_gpu.chunk_map import RECORD_PRESENCE_WORD, membership_from_record  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.triton_kernels import complex_fields  # noqa: E402

SEED = probe.SEED
bit_compare = probe.bit_compare
combine = probe.combine
_face = probe._face
PERIODIC = probe.PERIODIC
METALLIC = probe.METALLIC


def _constexpr_value(value: Any) -> Any:
    return getattr(value, "value", value)


#: Boundary string -> the kernel constexpr code the plan binds.
CODE_OF = {PERIODIC: 0, METALLIC: 1}
assert _constexpr_value(complex_fields.PERIODIC) == CODE_OF[PERIODIC]
assert _constexpr_value(complex_fields.METALLIC) == CODE_OF[METALLIC]

FU_NAMES = ("fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz")
FW_NAMES = ("f_w_Ex", "f_w_Ey", "f_w_Ez", "f_w_Hx", "f_w_Hy", "f_w_Hz")
FIELD_12 = ("Bx", "By", "Bz", "Dx", "Dy", "Dz",
            "Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
ALL_STATE = FIELD_12 + FU_NAMES + FW_NAMES


def log(message: str) -> None:
    print(message, flush=True)


def save(results: Dict[str, Any], out_path: str) -> None:
    """Atomic rewrite: tmp + fsync + os.replace, after every case."""
    directory = os.path.dirname(os.path.abspath(out_path)) or "."
    os.makedirs(directory, exist_ok=True)
    temporary = out_path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
        _stamp_provenance(results)  # bytes THIS process imported; see gate_provenance
        json.dump(results, handle, indent=1, sort_keys=True, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, out_path)


def _digest(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def write_provenance(results_dir: str, extra: Optional[Dict[str, Any]] = None) -> str:
    """The gate binds its own provenance — ``fingerprints.json`` is another
    track's file and carries no entry for this module; its AS-READ hash is
    recorded here instead so a concurrent edit is visible in the artifact."""
    import meep_gpu  # noqa: PLC0415

    package_dir = os.path.dirname(os.path.abspath(meep_gpu.__file__))
    kernel_dir = os.path.join(package_dir, "triton_kernels")
    tracked = {
        "meep_gpu/driver.py": os.path.join(package_dir, "driver.py"),
        "meep_gpu/fields.py": os.path.join(package_dir, "fields.py"),
        "meep_gpu/grid.py": os.path.join(package_dir, "grid.py"),
        "meep_gpu/pml.py": os.path.join(package_dir, "pml.py"),
        "meep_gpu/sources.py": os.path.join(package_dir, "sources.py"),
        "meep_gpu/stepping.py": os.path.join(package_dir, "stepping.py"),
        "triton_kernels/__init__.py": os.path.join(kernel_dir, "__init__.py"),
        "triton_kernels/coverage.py": os.path.join(kernel_dir, "coverage.py"),
        "triton_kernels/launch.py": os.path.join(kernel_dir, "launch.py"),
        "triton_kernels/complex_fields.py": os.path.join(kernel_dir, "complex_fields.py"),
        "triton_kernels/fingerprints.json": os.path.join(kernel_dir, "fingerprints.json"),
        "parity/gate_triton_complex.py": os.path.abspath(__file__),
        "parity/probe_fused_kernel_bit_identity.py": os.path.join(
            _HERE, "probe_fused_kernel_bit_identity.py"),
    }
    composition = os.path.join(_HERE, "probe_triton_complex_composition.py")
    if os.path.exists(composition):
        tracked["parity/probe_triton_complex_composition.py"] = composition
    record: Dict[str, Any] = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "cupy": None if cp is None else cp.__version__,
        "triton": triton.__version__ if _TRITON_AVAILABLE else None,
        "seed": SEED,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "device": (probe.device_info() if cp is not None else None),
        "sha256": {name: _digest(path) for name, path in tracked.items()},
    }
    if extra:
        record.update(extra)
    path = os.path.join(results_dir, "provenance.json")
    save(record, path)
    return path


# ---------------------------------------------------------------------------
# The device subnormal policy — the strip is the ship configuration
# ---------------------------------------------------------------------------
#
# Mechanism demonstrated by results/device_subnormal_policy_2026-08-11/scripts/
# run_boundedness_aligned.py: wrapping ``cupy.cuda.compiler.compile_using_nvrtc``
# and filtering the ``-ftz=true`` option flips every CuPy device binary to IEEE
# subnormal-keep. The wrapper composes with ``backends.guard_kernel_compilation``
# (the guard wraps whatever is installed at the seam), and the cache dir must be
# private and policy-suffixed because the cache key is computed ABOVE the seam:
# a shared cache either gets poisoned with stripped binaries or silently serves
# flushed ones.

#: The policy name a KEEP certification carries — the one the nine certified
#: families were cut under. Imported from the policy module rather than re-spelled
#: here so the gate's label and the module's can never drift apart.
STRIPPED_POLICY = _policy.CUPY_KEEP_POLICY_NAME

#: The policy name a FLUSH certification carries. Both names are now reachable
#: from this gate: the default is no longer the constant ``"keep"`` but the
#: REQUEST ``match_meep``, which :func:`meep_gpu.subnormal_policy.resolve_match_meep`
#: answers by measuring what MEEP itself left this host's FPU doing. On x86 with
#: MEEP imported that measurement reads flushing and the answer is ``"flush"``.
FLUSH_POLICY = _policy.FLUSH_POLICY_NAME

#: Token the private CUPY_CACHE_DIR must carry UNDER KEEP, so a cache cut under
#: one policy can never silently serve the other. Under FLUSH the requirement
#: inverts — CuPy's own unconditional ``-ftz=true`` IS the flush mechanism, so a
#: stripped cache would serve keep binaries under the flush name — and
#: :func:`meep_gpu.subnormal_policy.cupy_cache_reasons` is what states both
#: directions.
POLICY_CACHE_TOKEN = _policy.CUPY_CACHE_POLICY_TOKEN

FTZ_STRIP_MECHANISM = (
    "cupy.cuda.compiler.compile_using_nvrtc wrapped; '-ftz=true' (appended "
    "unconditionally by _compile_with_cache_cuda, cupy/cuda/compiler.py:552, "
    "CuPy 13.5.1) filtered from every NVRTC option tuple; private "
    "policy-suffixed CUPY_CACHE_DIR because the cache key is computed above "
    "the seam (results/device_subnormal_policy_2026-08-11/)")

POLICY_DEPENDENCE = (
    "bytes cut under this policy hold UNDER THE STRIPPED IEEE-KEEP POLICY ONLY "
    "— the ship configuration wherever the default request match_meep "
    "resolves to 'keep' (arm64, or any host with no MEEP to measure); on x86 "
    "with MEEP imported it resolves to 'flush' and the flush stamp's "
    "dependence applies instead. Bytes cut under the other policy differ in "
    "the subnormal range.")

FLUSH_MECHANISM = (
    "meep_gpu.subnormal_policy.install_subnormal_policy('flush'): the host FPU "
    "through mp.set_zero_subnormals(True) and MEASURED with "
    "backends.subnormals_flushed(); CuPy natively (13.5.1 appends '-ftz=true' to "
    "every NVRTC compile at cupy/cuda/compiler.py:552) with CUB removed from the "
    "reduction path by CUPY_ACCELERATORS='' before the CuPy import, because CUB "
    "is a prebuilt binary neither -ftz nor any in-process switch reaches; Triton "
    "by injecting denormal-fp-math-f32 into the llir stage and AUDITING every "
    "generated PTX instruction, refusing the compile on any audited f32 "
    "arithmetic instruction that lacks .ftz")

FLUSH_DEPENDENCE = (
    "bytes cut under this policy are the configuration a MEEP comparison ON AN "
    "X86 HOST is against — stock MEEP flushes float32 subnormals there "
    "(initialize() -> setup() -> set_zero_subnormals(true), src/mympi.cpp:188, "
    "compiled under #if HAVE_IMMINTRIN_H). They are NOT comparable in the "
    "subnormal range with the same family's ieee_keep_ftz_stripped record; the "
    "case counts, step counts, mutation splits and refusal sets are, and that is "
    "what a cross-policy comparison checks.")

_FTZ_STRIP: Optional[Dict[str, Any]] = None


def resolved_policy() -> str:
    """``'flush'`` or ``'keep'`` — the policy in force, or the one that would be.

    Never ``'match_meep'``: that is the QUESTION, and this is its answer. Before
    an install the answer is a resolution (a preference); after one it is what the
    executors were actually driven to.
    """
    return _policy.get_subnormal_policy()


def active_policy_name() -> str:
    """The policy NAME an artifact cut in this process must carry.

    The keep-era constant :data:`STRIPPED_POLICY` used to be that name
    unconditionally. It no longer is, and every place that compared an artifact's
    stamp against a hardcoded name had to become this call — otherwise a flush
    run would refuse its own bytes.
    """
    return FLUSH_POLICY if resolved_policy() == _policy.FLUSH else STRIPPED_POLICY


def ensure_meep_imported_for_policy() -> Dict[str, Any]:
    """Import MEEP so the DEFAULT policy is a measurement and not a fallback.

    ``meep_gpu`` itself may not import MEEP (importing it initializes MPI, so
    merely choosing a flag would join a communicator), which is why
    ``resolve_match_meep`` falls back to ``'keep'`` by name in a process that has
    no MEEP in ``sys.modules``. A parity GATE is under no such boundary — its job
    is to certify the bytes the engine ships, and the process the engine ships in
    is one where MEEP is imported long before any policy resolves.

    So the gate imports it, loudly, and records what that did to the host FPU.
    Under ``'flush'`` this is not optional in a second way: ``install_host_policy``
    drives the FPU through ``mp.set_zero_subnormals`` and there is no other
    exposure of the FTZ/DAZ bits this package may use.
    """
    already = "meep" in sys.modules
    before = _policy.backends.subnormals_flushed()
    record: Dict[str, Any] = {
        "already_imported": already,
        "host_flushing_before_import": before,
    }
    try:
        import meep  # noqa: F401, PLC0415
    except Exception as exc:
        record.update(imported=False, error=f"{type(exc).__name__}: {exc}",
                      host_flushing_after_import=before)
        log(f"[policy] MEEP is not importable here ({exc}); the default request "
            f"{_policy.MATCH_MEEP!r} is unmeasurable and falls back to "
            f"{_policy.MATCH_MEEP_FALLBACK!r}")
        return record
    after = _policy.backends.subnormals_flushed()
    record.update(imported=True, host_flushing_after_import=after,
                  version=getattr(sys.modules["meep"], "__version__", "unknown"))
    log(f"[policy] MEEP imported (already_imported={already}); host float32 "
        f"subnormals flushing {before} -> {after}")
    return record


#: A float32 subnormal and its bit pattern. ``x * 1.0`` is exact under IEEE-754
#: and returns ``x``; under FTZ the RESULT is flushed, so this one multiply
#: separates the two policies on any executor that can run it. Same operand class
#: the conformance gate uses (``parity/meep_gpu/gate_subnormal_policy.py``).
_SUBNORMAL_WORD = 0x00004000  # 2**-135
_FLUSHED_WORD = 0x00000000


def _measure_policy_on_executors() -> Dict[str, Any]:
    """MEASURE what the host and CuPy actually do to a float32 subnormal, now.

    The keep policy's evidence is a pair of counters — NVRTC option tuples seen
    and stripped — and the flush policy has no counters at all, because CuPy's own
    unconditional ``-ftz=true`` IS the flush mechanism and nothing is wrapped. A
    flush stamp built only from "the install returned attained=True" would be a
    structural claim where the keep stamp carries a measured one.

    So the gate measures. One multiply per executor, the operand the conformance
    gate uses, and the resulting WORD recorded beside what the policy requires.
    ``agrees`` is what :func:`ftz_strip_license_reasons` reads.
    """
    want = _FLUSHED_WORD if resolved_policy() == _policy.FLUSH else _SUBNORMAL_WORD
    operand = np.array([_SUBNORMAL_WORD], dtype=np.uint32).view(np.float32)
    record: Dict[str, Any] = {"expected_word": hex(want),
                              "operand_word": hex(_SUBNORMAL_WORD),
                              "operation": "float32 x * 1.0"}
    host = int((operand * np.float32(1.0)).view(np.uint32)[0])
    record["host"] = {"word": hex(host), "agrees": host == want}
    if cp is not None:
        device = cp.asarray(operand) * np.float32(1.0)
        word = int(cp.asnumpy(device.view(cp.uint32))[0])
        record["cupy"] = {"word": hex(word), "agrees": word == want}
    return record


def ftz_cache_reasons(cache_dir: Optional[str]) -> List[str]:
    """Why this CUPY_CACHE_DIR may not be used under the policy in force, by name.

    Delegated: CuPy computes its kernel cache key ABOVE the
    ``compile_using_nvrtc`` seam, so the directory name is the only separator
    there is, and the rule runs in BOTH directions — a keep run needs the
    policy-suffixed directory, a flush run must not be handed a keep run's.
    """
    return list(_policy.cupy_cache_reasons(resolved_policy(), cache_dir))


def install_ftz_strip(policy: Optional[str] = None) -> Dict[str, Any]:
    """Install THE SUBNORMAL POLICY on every executor, BEFORE any device compile.

    ``policy`` names the request. Omitting it keeps the historical behaviour —
    :data:`~meep_gpu.subnormal_policy.MATCH_MEEP`, resolved by measuring this
    host — which is what every existing caller relies on and what the default
    preserves. Pass a name when the CALLER, not the host, is the authority.

    WHY THE PARAMETER EXISTS (2026-08-17). A gate that installs a policy of its
    own AND calls this function has TWO authorities, and they disagree silently
    whenever the host's measured answer differs from the arms' certification.
    Measured: the complex stored-E gate installed ``keep`` (its arms then carried
    ``complex_fields.CERTIFIED_UNDER_SUBNORMAL_POLICY = "keep"``, a constant since
    deleted) and then
    reached here, where the no-request path resolved MATCH_MEEP to ``flush`` on
    x86 and refused the whole run. Setting ``MEEP_GPU_SUBNORMAL_POLICY`` made
    the two agree by accident, which is worse than the failure: it makes a
    correct-looking run depend on an environment variable nobody records.
    With this parameter the caller names its policy once, here, and there is
    exactly one authority for the process.

    The name is the keep era's and is kept because seventeen other scripts call
    it; what it does is now policy-aware. It resolves the policy the way every
    other verdict in this campaign is reached — by measuring — and drives the
    host, CuPy and Triton to the answer through
    :func:`meep_gpu.subnormal_policy.install_subnormal_policy`.

    WITH NO REQUEST the resolution is :data:`~meep_gpu.subnormal_policy.MATCH_MEEP`:
    whatever MEEP itself does to float32 subnormals on THIS host, read off the FPU
    MEEP's own initialization left behind. That is why MEEP is imported first here
    — in a process with no MEEP the question is unmeasurable and falls back to
    ``'keep'`` by name, which would silently re-cut the keep-era record on an x86
    host instead of the flush one the engine actually ships there.

    ``MEEP_GPU_SUBNORMAL_POLICY=keep`` (or ``=flush``) still overrides, and an
    explicit request is passed through untouched — which is what makes a
    same-bytes keep control run available beside every flush re-cut.

    Idempotent. Raises when the requested policy is unattainable on this host, or
    when the cache directory would let one policy's binaries serve under the
    other's name — loudly, at startup, before any artifact exists.
    """
    global _FTZ_STRIP
    if _FTZ_STRIP is not None:
        return _FTZ_STRIP
    if cp is None:
        # A HOST WITH NO CUPY GETS A RECORDED NO-OP, and this early return is
        # load-bearing rather than an optimization. There are no device binaries
        # to govern here, and the alternative — importing MEEP and driving the
        # process FPU — is a side effect with nothing to buy it: measured, it made
        # the laptop contract tranches install a process-wide policy and import
        # MEEP, which then broke test_subnormal_policy's own unmeasurable-default
        # tests later in the same pytest session (they assert, correctly, that the
        # suite has not imported MEEP).
        _FTZ_STRIP = {"installed": False,
                      "resolved": resolved_policy(),
                      "reason": "no cupy on this host; the device half of the "
                                "policy governs device binaries only, and this "
                                "gate does not move the host FPU for a run that "
                                "cannot reach a device"}
        return _FTZ_STRIP
    meep_record = ensure_meep_imported_for_policy()
    # ONE RESOLUTION, USED BY EVERY CLAUSE BELOW. ``policy_resolution`` is asked
    # for the CALLER's request, not for a bare re-measurement — otherwise this
    # function carries two authorities of its own: the cache guard on the next
    # line would judge against the host's measured answer while the install two
    # lines further down obeyed the argument. Measured 2026-08-17: the complex
    # stored-E gate passed 'keep' (its arms' certification constant) and was
    # refused for requesting 'flush', because the guard had re-resolved
    # MATCH_MEEP behind its back. The parameter reached the install and not the
    # clause that gates it, which is the same class of defect the parameter was
    # added to remove — one level down.
    resolution = _policy.policy_resolution(policy)
    policy = resolution["policy"]
    cache_dir = os.environ.get("CUPY_CACHE_DIR", "")
    # UNSET UNDER 'keep' IS DERIVABLE, NOT AN ERROR. The token rule exists because
    # CuPy's cache key is computed above the compile_using_nvrtc seam, so the
    # directory NAME is the only separator between stripped and unstripped
    # binaries — and subnormal_policy.keep_policy_cache_dir already computes a
    # directory that satisfies it (fastpath.plan_fast_path has derived one this
    # way since the dispatch policy gate landed). Requiring every caller to spell
    # the token by hand bought nothing and cost plenty: measured across 2026-08-17
    # and -19, EIGHT gate runs were refused or mis-paired for naming a cache dir
    # wrongly, including handing a keep-token directory to a flush gate.
    #
    # A dir that is SET but wrong is still refused: that is a caller conflict, not
    # a gap this code may paper over. And 'flush' stays underivable by design —
    # see keep_policy_cache_dir's own note.
    derived_cache = None
    if cp is not None and policy == _policy.KEEP and not cache_dir:
        derived_cache = _policy.keep_policy_cache_dir()
        os.environ["CUPY_CACHE_DIR"] = derived_cache
        cache_dir = derived_cache
    problems = list(_policy.cupy_cache_reasons(policy, cache_dir))
    if cp is not None and problems:
        raise RuntimeError(f"refusing to install the {policy!r} subnormal "
                           f"policy: " + "; ".join(problems))
    stamp = _policy.install_subnormal_policy(policy)
    state: Dict[str, Any] = {
        "installed": bool(stamp["installed"]),
        "policy": stamp["policy"],
        "resolved": stamp["resolved"],
        "requested": stamp["requested"],
        "resolved_from": stamp["resolved_from"],
        "match_meep": stamp["match_meep"],
        "meep_import": meep_record,
        "calls": int(stamp["nvrtc_calls"]),
        "removed": int(stamp["ftz_removed"]),
        "cache_dir": stamp["cache_dir"],
        "cache_preexisting": bool(stamp["cache_preexisting"]),
        "cache_dir_derived": derived_cache,  # None when the caller supplied one

        "triton": dict(stamp["triton"]),
        "unattained": list(stamp["unattained"]),
        "executors": {name: dict(report)
                      for name, report in stamp["executors"].items()},
    }
    _FTZ_STRIP = state
    log(f"[policy] {stamp['requested']!r} resolved to {stamp['resolved']!r} "
        f"(from the {stamp['resolved_from']}); artifacts carry "
        f"{stamp['policy']!r}. cupy_cache={cache_dir or '(unset)'} "
        f"preexisting={state['cache_preexisting']} "
        f"triton_cache={os.environ.get('TRITON_CACHE_DIR', '') or '(unset)'} "
        f"unattained={state['unattained'] or 'none'}")
    if isinstance(stamp["match_meep"], dict):
        log(f"[policy] {stamp['match_meep']['why']}")
    # AFTER the install, and recorded in the state the stamp copies: what the
    # executors measurably do to a subnormal now. Under keep this also guarantees
    # the strip's counters are non-zero, because the CuPy leg compiles a kernel.
    state["measured"] = _measure_policy_on_executors()
    state["calls"] = int(_policy.cupy_strip_counters()["calls"])
    state["removed"] = int(_policy.cupy_strip_counters()["removed"])
    log(f"[policy] measured: {json.dumps(state['measured'])}")
    return state


def ftz_strip_license_reasons() -> List[str]:
    """Why the bytes measured so far may NOT certify anything, by name.

    Empty means: the strip is installed and every NVRTC option tuple observed
    lost its '-ftz=true' (or a policy-suffixed pre-populated cache explains a
    compile-free run) — the measured bytes are the ship configuration's.
    """
    state = _FTZ_STRIP
    if state is None or not state.get("installed"):
        return [
            "the subnormal policy is not installed: bytes certify only under the "
            "policy this engine ships in, driven onto every executor and measured "
            "there; a run that certified nothing about its own subnormal "
            "behaviour certifies the wrong binaries"]
    resolved = state.get("resolved") or resolved_policy()
    reasons: List[str] = []
    for name in state.get("unattained", ()):
        report = state.get("executors", {}).get(name, {})
        reasons.append(
            f"the {name!r} executor did not attain the {resolved!r} "
            f"policy: " + "; ".join(report.get("reasons")
                                    or ["(no reason recorded)"]))
    if resolved == _policy.KEEP:
        calls, removed = int(state["calls"]), int(state["removed"])
        if calls == 0:
            if state.get("cache_preexisting"):
                state["zero_compiles_explained"] = (
                    "no NVRTC compile ran; the policy-suffixed private cache was "
                    "pre-populated, so the served binaries were cut under the "
                    "strip by this same mechanism")
            else:
                reasons.append(
                    "zero NVRTC compiles observed under a FRESH policy-suffixed "
                    "cache: the strip was never exercised and the measured bytes "
                    "cannot be attributed to the stripped policy")
        elif removed < calls:
            reasons.append(
                f"{calls - removed}/{calls} observed NVRTC option tuples carried "
                f"no -ftz=true to strip: CuPy's policy seam has moved (13.5.1 "
                f"appended it unconditionally, cupy/cuda/compiler.py:552) — "
                f"re-verify the seam before licensing")
    # Under flush there is no strip to count: CuPy's own unconditional -ftz=true
    # IS the mechanism, so a zero counter says nothing. Both policies are held
    # instead by the same two measured things — the executor probe taken at
    # install, and Triton's PTX audit, which reads every generated instruction and
    # refuses a compile whose audited f32 arithmetic disagrees with the policy.
    for executor, probe in (state.get("measured") or {}).items():
        if isinstance(probe, dict) and not probe.get("agrees"):
            reasons.append(
                f"the {executor!r} executor measurably does NOT obey the "
                f"{resolved!r} policy: float32 {probe['word']} came back "
                f"from x * 1.0 on the subnormal "
                f"{(state['measured'] or {}).get('operand_word')}, and the policy "
                f"requires {(state['measured'] or {}).get('expected_word')}")
    triton = _policy.triton_counters()
    compiles = int(triton.get("ptx_compiles", 0))
    audited = int(triton.get("ptx_audited", 0))
    violations = int(triton.get("ptx_violations", 0))
    if compiles and not audited:
        reasons.append(
            f"{compiles} Triton PTX modules were generated and NONE was audited: "
            f"the policy's only enforcement point on Triton did not read them, so "
            f"the measured bytes cannot be attributed to it")
    if violations:
        reasons.append(
            f"{violations} audited PTX f32 arithmetic instructions disagreed with "
            f"the {resolved!r} policy: the injection is incomplete on "
            f"these kernels and the bytes are a HALF-APPLIED policy")
    return reasons


def policy_stamp(backend_name: str) -> Dict[str, Any]:
    """The subnormal-policy record one backend's artifact is stamped with."""
    state = _FTZ_STRIP or {}
    module_stamp = _policy.policy_stamp()
    if backend_name == "numpy":
        # The host is an executor under BOTH policies and is driven under both:
        # on x86 under flush it flushes, which is exactly why a NumPy leg may no
        # longer claim IEEE-keep by platform default.
        host = (state.get("executors") or {}).get("host", {})
        return {"policy": ("host_" + FLUSH_POLICY
                           if state.get("resolved") == _policy.FLUSH
                           else "host_ieee_keep"),
                "resolved": state.get("resolved") or resolved_policy(),
                "installed": bool(state.get("installed")),
                "host_flushing": host.get("flushing_after"),
                "mechanism": host.get("mechanism"),
                "note": "host NumPy: no NVRTC compile is involved, so the "
                        "verdict is the MEASURED FPU state, not a platform "
                        "default"}
    if cp is None:
        return {"policy": "no_device",
                "resolved": state.get("resolved") or resolved_policy(),
                "note": "no cupy on this host; the device half of the policy "
                        "governs device binaries only"}
    if state.get("installed"):
        resolved = state.get("resolved") or resolved_policy()
        flush = resolved == _policy.FLUSH
        stamp = {"policy": state.get("policy") or active_policy_name(),
                 "resolved": resolved,
                 "requested": state.get("requested"),
                 "resolved_from": state.get("resolved_from"),
                 "match_meep": state.get("match_meep"),
                 "mechanism": FLUSH_MECHANISM if flush else FTZ_STRIP_MECHANISM,
                 "dependence": FLUSH_DEPENDENCE if flush else POLICY_DEPENDENCE,
                 "nvrtc_calls": int(state.get("calls",
                                              module_stamp["nvrtc_calls"])),
                 "ftz_removed": int(state.get("removed",
                                              module_stamp["ftz_removed"])),
                 "measured": dict(state.get("measured") or {}),
                 "triton": dict(module_stamp["triton"]),
                 "cache_dir": state.get("cache_dir", ""),
                 "cache_preexisting": bool(state.get("cache_preexisting")),
                 "unattained": list(module_stamp["unattained"])}
        if state.get("zero_compiles_explained"):
            stamp["zero_compiles_explained"] = state["zero_compiles_explained"]
        return stamp
    return {"policy": "cupy_default_ftz_flush",
            "resolved": resolved_policy(),
            "warning": "measured WITHOUT any policy installed: CuPy appends "
                       "-ftz=true to every NVRTC compile "
                       "(cupy/cuda/compiler.py:552) while the host FPU and "
                       "Triton decide separately, and a half-applied policy is "
                       "worse than either applied uniformly; these bytes are "
                       "NOT the ship configuration and license nothing"}


def _interleave_c8(re: np.ndarray, im: np.ndarray) -> np.ndarray:
    """Plane assignment, never ``re + 1j*im``: the ADDITION destroys signed
    zeros (-0.0 + (+0.0) is +0.0), and the signed-zero rows are exactly the
    discriminating vectors."""
    out = np.empty(re.shape, dtype=np.complex64)
    out.real = re
    out.imag = im
    return out


# ---------------------------------------------------------------------------
# The in-file complex reference — the array path transcribed, complex branches in
# ---------------------------------------------------------------------------
#
# The shared probe's real-field transcription is reused where the complex path
# is unchanged (the recurrence, the constitutive accumulation); what is added
# here is only what real fields cannot carry: the Bloch wrap multiply, the
# conjugate on the down-shift, and complex storage throughout. The ``reference``
# leg pins this transcription against stepping.py itself, on both backends,
# BEFORE any kernel is measured against it. Backend-generic: ``xp`` is NumPy or
# CuPy, and every operation is one elementwise kernel, so each product and each
# sum rounds on its own on both.

#: ``part_products.CONNECT_TOLERANCE``, restated: MEEP's phase_isclose window.
CONNECT_TOLERANCE = 1e-13


def _from_parts(xp: Any, real: Any, imag: Any) -> Any:
    """A complex64 volume whose parts are ``real`` and ``imag``, with no product."""
    pair = xp.stack((xp.asarray(real, dtype=xp.float32),
                     xp.asarray(imag, dtype=xp.float32)), axis=-1)
    return xp.ascontiguousarray(xp.ascontiguousarray(pair).view(xp.complex64)[..., 0])


def _scale(xp: Any, coefficient: Any, values: Any) -> Any:
    """A REAL coefficient times complex storage, one float32 product per part
    (``part_products.scale``; MEEP's ``DOCMP`` loops)."""
    c = xp.asarray(coefficient, dtype=xp.float32)
    return _from_parts(xp, values.real * c, values.imag * c)


def complex_shift(xp: Any, field: Any, axis: int, boundary: str,
                  backward: bool) -> Any:
    """``stepping._shift_up`` / ``_shift_down``, PERIODIC and METALLIC: a plain roll,
    phased or not (the stored window makes every read one cell over an owned word);
    a METALLIC axis then serves an exact complex zero past its wall."""
    shifted = xp.roll(field, 1 if backward else -1, axis=axis)
    if boundary == PERIODIC:
        return shifted
    if boundary == METALLIC:
        shifted[_face(axis, 0 if backward else -1)] = 0
        return shifted
    raise ValueError(f"boundary {boundary!r} has no complex reference here")


def complex_connected(xp: Any, field: Any, axis: int, backward: bool,
                      phase: Optional[complex]) -> Any:
    """``stepping._bloch_ghost_operand``: plane 0 of the UNSHIFTED operand, connected.

    B (forward differences) reads the not-owned word BELOW the window,
    ``thephase = 1.0 * conj(eikna)``; D the word ABOVE it, ``1.0 * eikna``, both in
    double. MEEP's ``connect_phase``: a copy within 1e-13 of +1, a negation per part
    within 1e-13 of -1, else ``(a*re) - (b*im)``, ``(a*im) + (b*re)`` with ``(a, b)``
    the phase rounded to float32 and every operation rounded on its own. Returns
    ``field`` itself when nothing is connected.
    """
    if phase is None:
        return field
    eikna = complex(phase)
    fr, fi = (eikna.real, eikna.imag) if backward else (eikna.real, -eikna.imag)
    pr, pi = 1.0 * fr - 0.0 * fi, 1.0 * fi + 0.0 * fr
    if abs(pi) < CONNECT_TOLERANCE and abs(pr - 1.0) < CONNECT_TOLERANCE:
        return field
    out = field.copy()
    face = _face(axis, 0)
    re = xp.ascontiguousarray(field[face].real)
    im = xp.ascontiguousarray(field[face].imag)
    if abs(pi) < CONNECT_TOLERANCE and abs(pr + 1.0) < CONNECT_TOLERANCE:
        out[face] = _from_parts(xp, -re, -im)
        return out
    a, b = np.float32(pr), np.float32(pi)
    out[face] = _from_parts(xp, (re * a) - (im * b), (im * a) + (re * b))
    return out


def _presence(record: Any) -> Tuple[bool, bool, bool]:
    """Partner presence per axis, read from the record's presence word."""
    word = int(np.asarray(record).reshape(-1)[RECORD_PRESENCE_WORD])
    return tuple(bool(word & (1 << axis)) for axis in range(3))


def _membership(xp: Any, record: Any, half_integer: bool, axis: int, shape) -> Any:
    """One axis's membership line broadcast to the grid (the kernels' formula)."""
    line = membership_from_record(np.asarray(record).reshape(-1), half_integer, axis,
                                  int(shape[axis]))
    view = [1, 1, 1]
    view[axis] = int(shape[axis])
    return xp.broadcast_to(xp.asarray(line.reshape(view)), tuple(int(n) for n in shape))


def complex_curl(xp: Any, sources: Dict[str, Any], term, dtdx: float,
                 backward: bool, boundaries: Sequence[str],
                 phases: Sequence[Optional[complex]], record: Any,
                 grouping: str = "array_order") -> Any:
    """``dtdx * curl`` as ``stepping._curl_from_operands`` forms it on complex storage.

    MEEP's left-to-right ``((g1s - g1) + g2) - g2s`` with both partners present;
    ``dtdx * (g1s - g1)`` without g2; ``(-dtdx) * (g2s - g2)`` without g1; an exact
    zero without either (step_generic.cpp:72-76, :104-108). Each unshifted partner is
    read CONNECTED along the axis it is differenced on. ``grouping="pairwise"`` is the
    previous ``(g1s - g1) + (g2 - g2s)``, kept as the control the kernel must not
    reproduce.
    """
    _target, g1, a1, g2, a2, _dsig, _dsigu, _iyee = term
    present = _presence(record)
    g1s = complex_shift(xp, sources[g1], a1, boundaries[a1], backward)
    g2s = complex_shift(xp, sources[g2], a2, boundaries[a2], backward)
    g1c = complex_connected(xp, sources[g1], a1, backward, phases[a1])
    g2c = complex_connected(xp, sources[g2], a2, backward, phases[a2])
    scale = np.float32(dtdx)
    if grouping == "pairwise":
        return _scale(xp, scale, (g1s - g1c) + (g2c - g2s))
    if present[a1] and present[a2]:
        return _scale(xp, scale, ((g1s - g1c) + g2c) - g2s)
    if present[a1]:
        return _scale(xp, scale, g1s - g1c)
    if present[a2]:
        return _scale(xp, -scale, g2s - g2c)
    return xp.zeros_like(sources[g1])


def complex_mask(curl: Any, iyee: Sequence[int], boundaries: Sequence[str]) -> None:
    """stepping._mask_non_owned_cells (:1865-1902), the branches this domain can
    reach: a metallic wall masks cell 0 of every shift-0 component; the folded-
    periodic top-cell clause is out of domain (no folds are admitted here)."""
    for axis in range(3):
        if iyee[axis] == 0 and boundaries[axis] == METALLIC:
            curl[_face(axis, 0)] = 0


def complex_pml_step(xp: Any, arrays: Dict[str, Any], coefficients: Dict[str, Any],
                     dtdx: float, sub_step: str, boundaries: Sequence[str],
                     phases: Sequence[Optional[complex]], record: Any,
                     grouping: str = "array_order") -> None:
    """One complex curl sub-step, in place, both PML forms (``stepping._apply_curl_regions``).

    A target whose chunk carries its dsigu layer -- the record's membership on the
    curl's sub-lattice (half-integer for B, integer for D) along the dsigu axis --
    steps the split-field recurrence and writes ``fu``; every other cell steps the
    dsig-only form ``((f*kms) - curl)*sinv`` and leaves ``fu`` untouched. Every
    product is one per part.
    """
    terms = probe.B_PML_TERMS if sub_step == "step_B" else probe.D_PML_TERMS
    backward = sub_step == "step_D"
    for term in terms:
        target, _g1, _a1, _g2, _a2, dsig, dsigu, iyee = term
        curl = complex_curl(xp, arrays, term, dtdx, backward, boundaries, phases,
                            record, grouping)
        complex_mask(curl, iyee, boundaries)
        f, fu = arrays[target], arrays["fu_" + target]
        kms, sinv = coefficients["kms_" + dsig], coefficients["sinv_" + dsig]
        kms_u, sinv_u = coefficients["kms_" + dsigu], coefficients["sinv_" + dsigu]
        inside = _membership(xp, record, not backward, probe.AXIS_NAMES.index(dsigu),
                             f.shape)
        previous = fu.copy()
        n = _scale(xp, sinv, _scale(xp, kms, previous) - curl)
        recurrence = _scale(xp, sinv_u, (_scale(xp, kms_u, f) + n) - previous)
        dsig_only = _scale(xp, sinv, _scale(xp, kms, f) - curl)
        f[...] = xp.where(inside, recurrence, dsig_only)
        fu[...] = xp.where(inside, n, fu)


def complex_constitutive_step(xp: Any, arrays: Dict[str, Any],
                              coefficients: Dict[str, Any], side: str, record: Any,
                              start_h: bool = False) -> None:
    """``stepping._apply_constitutive_regions`` on each component's own axis, in place.

    On a cell whose chunk carries the component's own (dsigw) layer -- the record's
    membership on the integer lattice for H, the half-integer one for E -- the
    accumulation ``f + ((kps*src) - (kms*prev))`` with ``f_w = src``; everywhere else
    ``f = src`` and ``f_w`` untouched. ``src`` is B, or ``D * inv_eps`` per part.
    ``start_h`` is MEEP's start of H (``stepping.separate_h_from_b``): on the run's
    first ``update_H``, H and ``f_w`` are copies of B on those cells before the
    accumulation.
    """
    for target, source_name, axis in probe.CONSTITUTIVE_TERMS[side]:
        axis_name = probe.AXIS_NAMES[axis]
        kps = coefficients["kps_" + axis_name]
        kms = coefficients["kms_" + axis_name]
        field, fw = arrays[target], arrays["f_w_" + target]
        inside = _membership(xp, record, side == "E", axis, field.shape)
        product = (_scale(xp, arrays["inv_eps_" + target], arrays[source_name])
                   if side == "E" else arrays[source_name].copy())
        if side == "H" and start_h:
            field[...] = xp.where(inside, product, field)
            fw[...] = xp.where(inside, product, fw)
        previous = fw.copy()
        accumulated = field + (_scale(xp, kps, product) - _scale(xp, kms, previous))
        field[...] = xp.where(inside, accumulated, product)
        fw[...] = xp.where(inside, product, fw)


# ---------------------------------------------------------------------------
# Seeding — the targeted planes several defects are only visible on
# ---------------------------------------------------------------------------

def _seed_host_complex(shape: Tuple[int, int, int], rng,
                       scale: float = 1.0) -> np.ndarray:
    host = _interleave_c8(
        (scale * rng.uniform(-1.0, 1.0, size=shape)).astype(np.float32),
        (scale * rng.uniform(-1.0, 1.0, size=shape)).astype(np.float32))
    live_axes = [axis for axis in range(3) if shape[axis] > 1]
    if not live_axes:
        return host
    # Plane of imag -0.0 (catches multiply-by-(1+0j) where the skip belongs).
    last_axis = live_axes[-1]
    host.imag[_face(last_axis, 0)] = -0.0
    # Plane of zero-real (half -0.0) with strictly negative imag: the only rows
    # on which the zero cross terms of the zero-imaginary product are visible.
    first_axis = live_axes[0]
    plane_re = host.real[_face(first_axis, -1)]
    plane_im = host.imag[_face(first_axis, -1)]
    plane_im[...] = -np.abs(plane_im) - 0.25
    plane_re[...] = 0.0
    checker = (np.indices(plane_re.shape).sum(axis=0) % 2).astype(bool)
    plane_re[checker] = -0.0
    return host


def make_complex_state(xp, shape: Tuple[int, int, int], rng,
                       names: Sequence[str], sparse: Optional[float] = None
                       ) -> Dict[str, Any]:
    """The seeded state; ``sparse`` keeps that fraction of words and turns every other
    word into a zero of random sign, so a stencil whose every operand is zero is
    common and the SIGN of a negated connection reaches a stored word."""
    out = {}
    for name in names:
        host = _seed_host_complex(shape, rng)
        if sparse:
            for part in (host.real, host.imag):
                keep = rng.random(size=shape) < sparse
                zeros = np.where(rng.random(size=shape) < 0.5,
                                 np.float32(-0.0), np.float32(0.0))
                part[...] = np.where(keep, part, zeros)
        out[name] = xp.asarray(host)
    return out


def make_inverse_epsilon(xp, shape: Tuple[int, int, int], rng) -> Dict[str, Any]:
    return {"inv_eps_" + name: xp.asarray(np.ascontiguousarray(
        rng.uniform(0.2, 0.9, size=shape).astype(np.float32)))
        for name in ("Ex", "Ey", "Ez")}


# ---------------------------------------------------------------------------
# The synthetic sweep product
# ---------------------------------------------------------------------------

def _phase(k: float) -> complex:
    return cmath.exp(2j * math.pi * k)


SHAPES: Tuple[Tuple[int, int, int], ...] = (
    (7, 6, 5),      # 3-D, odd extents, non-cube (an i<->k swap cannot pass)
    (6, 5, 1),      # 2-D sheet, invariant z
    (1, 1, 12),     # the (1,1,n) collapsed-transverse family
)
#: (name, boundaries, phases, optional shape restriction). A metallic axis never
#: carries a phase (grid.py:992-1001; stepping._bloch_phases raises); the
#: transverse-collapsed configuration is Grid-unreachable TODAY (grid.py:980-991)
#: but the kernel's n==1 wrap+phase mechanics are the task-#9 lift's substrate,
#: so the synthetic leg measures them without a Grid.
CONFIGS: Tuple[Dict[str, Any], ...] = (
    {"name": "unphased_all_periodic", "boundaries": (PERIODIC, PERIODIC, PERIODIC),
     "phases": (None, None, None)},
    {"name": "general_bloch", "boundaries": (PERIODIC, PERIODIC, PERIODIC),
     "phases": (_phase(0.21), _phase(-0.13), _phase(0.4))},
    {"name": "edge_x", "boundaries": (PERIODIC, PERIODIC, PERIODIC),
     "phases": (complex(-1.0, 0.0), _phase(0.37), None)},
    {"name": "one_axis_phase_none", "boundaries": (PERIODIC, PERIODIC, PERIODIC),
     "phases": (_phase(0.29), None, None)},
    {"name": "phased_times_metallic", "boundaries": (PERIODIC, METALLIC, PERIODIC),
     "phases": (_phase(0.33), None, _phase(-0.4))},
    {"name": "metallic_k0", "boundaries": (METALLIC, METALLIC, PERIODIC),
     "phases": (None, None, None)},
    {"name": "transverse_collapsed", "boundaries": (PERIODIC, PERIODIC, PERIODIC),
     "phases": (_phase(0.31), _phase(-0.22), None),
     "only_shapes": ((1, 1, 12),)},
)
DTDX: Tuple[float, ...] = (0.5, 0.35)  # 0.35 is MANDATORY: dtdx=0.5-only gates
#                                        caught about a third of unguarded rows.
CURL_SUB_STEPS = ("step_B", "step_D")
MULTI_STEP_COUNT = 6


def config_viable(shape: Sequence[int], config: Dict[str, Any]) -> Optional[str]:
    only = config.get("only_shapes")
    if only is not None and tuple(shape) not in {tuple(s) for s in only}:
        return f"config {config['name']} is restricted to shapes {only}"
    for axis in range(3):
        if config["boundaries"][axis] == METALLIC and int(shape[axis]) == 1:
            return (f"axis {axis} is collapsed (n=1) and metallic: Grid resolves "
                    f"an invariant axis periodic, so the pairing is unreachable")
        if (config["phases"][axis] is not None
                and config["boundaries"][axis] == METALLIC):
            return f"axis {axis} is metallic and phased — never constructed"
    return None


def _mutate_plan(plan: Any, phases: Sequence[Optional[complex]], sub_step: str,
                 host_mutation: Optional[str]) -> None:
    """Host mutations of a built plan's connection table, in place.

    ``m1_connection_direction_swapped`` hands the kernel the OTHER sub-step's
    factor (``eikna`` under B, ``conj(eikna)`` under D): the classic band-structure
    sign error. ``m5_unit_product_instead_of_copy`` connects every unphased axis by
    the product with ``(1, 0)`` instead of MEEP's copy: recorded, never asserted
    (``1*x - 0*y`` differs from ``x`` only on a signed zero).
    """
    if host_mutation == "m1_connection_direction_swapped":
        _kinds, values = complex_fields.connection_arguments(
            phases, backward=sub_step != "step_D")
        plan.connection_values = tuple(values)
    elif host_mutation == "m5_unit_product_instead_of_copy":
        kinds = list(plan.connections)
        values = list(plan.connection_values)
        for axis, phase in enumerate(phases):
            if phase is None:
                kinds[axis] = complex_fields.CONNECTION_KINDS["phase"]
                values[2 * axis:2 * axis + 2] = [1.0, 0.0]
        plan.connections = tuple(kinds)
        plan.connection_values = tuple(values)


def one_curl_case(shape, config, dtdx, sub_step, guard,
                  kernel=None, host_mutation: Optional[str] = None,
                  sparse: Optional[float] = None) -> Dict[str, Any]:
    case: Dict[str, Any] = {
        "shape": list(shape), "config": config["name"],
        "boundaries": list(config["boundaries"]),
        "phases": [repr(p) for p in config["phases"]],
        "dtdx": repr(dtdx), "sub_step": sub_step, "guard": guard,
        "host_mutation": host_mutation, "record": "two_run", "sparse": sparse,
    }
    skip = config_viable(shape, config)
    if skip:
        case["skipped"] = skip
        return case
    rng = np.random.default_rng(SEED + 3)
    names = FIELD_12 + FU_NAMES
    arrays = make_complex_state(cp, tuple(shape), rng, names, sparse=sparse)
    coefficients = probe.synthetic_coefficients(cp, tuple(shape),
                                                sub_step == "step_B")
    flat = probe.flatten_coefficients(coefficients)
    record = two_run_record(shape)

    reference = {name: array.copy() for name, array in arrays.items()}
    complex_pml_step(cp, reference, coefficients, dtdx, sub_step,
                     config["boundaries"], config["phases"], record)

    codes = [CODE_OF[b] for b in config["boundaries"]]
    plan = complex_fields.plan_complex_pml_curl_from_arrays(
        sub_step, arrays, flat, codes, config["phases"], float(dtdx), kernel=kernel, chunk_bounds=record)
    _mutate_plan(plan, config["phases"], sub_step, host_mutation)
    plan.run(guard=guard)
    cp.cuda.runtime.deviceSynchronize()

    targets = ("Bx", "By", "Bz") if sub_step == "step_B" else ("Dx", "Dy", "Dz")
    names_checked = targets + tuple("fu_" + t for t in targets)
    case["verdict"] = combine({name: bit_compare(arrays[name], reference[name])
                               for name in names_checked})
    return case


def one_constitutive_case(shape, side, guard,
                          kernel=None) -> Dict[str, Any]:
    case: Dict[str, Any] = {"shape": list(shape), "side": side, "guard": guard,
                            "sub_step": "update_" + side, "record": "two_run"}
    rng = np.random.default_rng(SEED + 5)
    targets, auxiliaries, sources = probe.constitutive_names(side)
    arrays = make_complex_state(cp, tuple(shape), rng,
                                targets + auxiliaries + sources)
    if side == "E":
        arrays.update(make_inverse_epsilon(cp, tuple(shape), rng))
    coefficients = probe.synthetic_constitutive_coefficients(
        cp, tuple(shape), half_integer=(side == "E"))
    flat = probe.flatten_coefficients(coefficients)

    record = two_run_record(shape)
    reference = {name: (array.copy() if str(array.dtype) == "complex64" else array)
                 for name, array in arrays.items()}
    complex_constitutive_step(cp, reference, coefficients, side, record)

    plan = complex_fields.plan_complex_constitutive_from_arrays(
        side, arrays, flat, kernel=kernel, chunk_bounds=record)
    plan.run(guard=guard)
    cp.cuda.runtime.deviceSynchronize()

    case["verdict"] = combine({name: bit_compare(arrays[name], reference[name])
                               for name in targets + auxiliaries})
    return case


def multi_step_case(shape, config, dtdx) -> Dict[str, Any]:
    """B -> H -> D -> E cycles from one seeded state — the auxiliaries are state,
    so a kernel right for one launch and wrong after is caught here."""
    case: Dict[str, Any] = {"shape": list(shape), "config": config["name"],
                            "dtdx": repr(dtdx), "cycles": MULTI_STEP_COUNT,
                            "record": "two_run"}
    skip = config_viable(shape, config)
    if skip:
        case["skipped"] = skip
        return case
    rng = np.random.default_rng(SEED + 9)
    names = FIELD_12 + FU_NAMES + FW_NAMES
    arrays = make_complex_state(cp, tuple(shape), rng, names)
    arrays.update(make_inverse_epsilon(cp, tuple(shape), rng))
    reference = {name: (array.copy() if str(array.dtype) == "complex64" else array)
                 for name, array in arrays.items()}
    curl_tables = {sub: probe.synthetic_coefficients(cp, tuple(shape),
                                                     sub == "step_B")
                   for sub in CURL_SUB_STEPS}
    curl_flat = {sub: probe.flatten_coefficients(curl_tables[sub])
                 for sub in CURL_SUB_STEPS}
    constitutive_tables = {side: probe.synthetic_constitutive_coefficients(
        cp, tuple(shape), half_integer=(side == "E")) for side in ("H", "E")}
    constitutive_flat = {side: probe.flatten_coefficients(constitutive_tables[side])
                         for side in ("H", "E")}
    codes = [CODE_OF[b] for b in config["boundaries"]]
    record = two_run_record(shape)

    plans = {
        "step_B": complex_fields.plan_complex_pml_curl_from_arrays(
            "step_B", arrays, curl_flat["step_B"], codes, config["phases"],
            float(dtdx), chunk_bounds=record),
        "update_H": complex_fields.plan_complex_constitutive_from_arrays(
            "H", arrays, constitutive_flat["H"], chunk_bounds=record),
        "step_D": complex_fields.plan_complex_pml_curl_from_arrays(
            "step_D", arrays, curl_flat["step_D"], codes, config["phases"],
            float(dtdx), chunk_bounds=record),
        "update_E": complex_fields.plan_complex_constitutive_from_arrays(
            "E", arrays, constitutive_flat["E"], chunk_bounds=record),
    }
    for _ in range(MULTI_STEP_COUNT):
        plans["step_B"].run()
        complex_pml_step(cp, reference, curl_tables["step_B"], dtdx, "step_B",
                         config["boundaries"], config["phases"], record)
        plans["update_H"].run()
        complex_constitutive_step(cp, reference, constitutive_tables["H"], "H", record)
        plans["step_D"].run()
        complex_pml_step(cp, reference, curl_tables["step_D"], dtdx, "step_D",
                         config["boundaries"], config["phases"], record)
        plans["update_E"].run()
        complex_constitutive_step(cp, reference, constitutive_tables["E"], "E", record)
    cp.cuda.runtime.deviceSynchronize()
    case["verdict"] = combine({name: bit_compare(arrays[name], reference[name])
                               for name in names})
    return case


def summarize(cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    guarded = [c for c in cases if c.get("guard") is False and not c.get("skipped")]
    ran = [c for c in cases if not c.get("skipped")]
    return {
        "ran": len(ran),
        "identical": sum(int(c["verdict"]["bit_identical"]) for c in ran),
        "skipped": len(cases) - len(ran),
        "guarded_ran": len(guarded),
        "guarded_identical": sum(int(c["verdict"]["bit_identical"])
                                 for c in guarded),
        "guarded_pass": bool(guarded) and all(c["verdict"]["bit_identical"]
                                              for c in guarded),
    }


def run_synthetic(results: Dict[str, Any], out_path: str,
                  kernel=None, host_mutation: Optional[str] = None,
                  label: str = "gate", guards=(False, True),
                  shapes=SHAPES, configs=CONFIGS,
                  sub_steps=CURL_SUB_STEPS, dtdx_values=DTDX,
                  sparse: Optional[float] = None) -> Dict[str, Any]:
    cases: List[Dict[str, Any]] = []
    combos = [(shape, config, dtdx, sub_step)
              for shape in shapes for config in configs
              for dtdx in dtdx_values for sub_step in sub_steps]
    total = len(guards) * len(combos)
    index = 0
    for guard in guards:
        for shape, config, dtdx, sub_step in combos:
            index += 1
            started = time.time()
            case = one_curl_case(shape, config, dtdx, sub_step, guard,
                                 kernel=kernel, host_mutation=host_mutation,
                                 sparse=sparse)
            case["seconds"] = round(time.time() - started, 3)
            cases.append(case)
            if case.get("skipped"):
                log(f"[{label}] {index}/{total} SKIPPED {case['skipped'][:70]}")
            else:
                verdict = case["verdict"]
                log(f"[{label}] {index}/{total} guard={guard} {sub_step} "
                    f"{'x'.join(str(n) for n in shape)} {config['name']} "
                    f"dtdx={dtdx!r}: identical={verdict['bit_identical']} "
                    f"(differing={verdict['differing_floats']}/"
                    f"{verdict['total_floats']}, "
                    f"maxulp={verdict.get('max_ulp', 0)}) ({case['seconds']} s)")
            summary = summarize(cases)
            summary["cases"] = cases
            results.setdefault("synthetic", {})[label] = summary
            save(results, out_path)
    return results["synthetic"][label]


def run_constitutive(results: Dict[str, Any], out_path: str,
                     kernel=None, label: str = "gate",
                     guards=(False, True), shapes=SHAPES) -> Dict[str, Any]:
    cases: List[Dict[str, Any]] = []
    combos = [(shape, side) for shape in shapes for side in ("H", "E")]
    total = len(guards) * len(combos)
    index = 0
    for guard in guards:
        for shape, side in combos:
            index += 1
            started = time.time()
            case = one_constitutive_case(shape, side, guard,
                                         kernel=kernel)
            case["seconds"] = round(time.time() - started, 3)
            cases.append(case)
            verdict = case["verdict"]
            log(f"[{label}:constitutive] {index}/{total} guard={guard} "
                f"update_{side} {'x'.join(str(n) for n in shape)}: "
                f"identical={verdict['bit_identical']} "
                f"(differing={verdict['differing_floats']}/"
                f"{verdict['total_floats']}) ({case['seconds']} s)")
            summary = summarize(cases)
            summary["cases"] = cases
            results.setdefault("constitutive", {})[label] = summary
            save(results, out_path)
    return results["constitutive"][label]


# ---------------------------------------------------------------------------
# Mutations — source-level (compiled, launch-counted) and host-level
# ---------------------------------------------------------------------------

class CountingKernel:
    """Proof the mutant actually launched. Three prior harness-disarm defects
    make this mandatory: a leg whose mutated kernel never ran reports a hollow
    pass, which is worse than no leg at all."""

    def __init__(self, kernel: Any) -> None:
        self.kernel = kernel
        self.launches = 0

    def __getitem__(self, grid):
        launcher = self.kernel[grid]

        def launch(*args, **kwargs):
            self.launches += 1
            return launcher(*args, **kwargs)

        return launch


_TEMPORARY: List[str] = []


def compile_mutated(source: str, kernel_name: str):
    """Compile a mutated copy of the shipped kernels from a real file on disk
    (Triton reads source via inspect, so exec'd text raises at first launch).
    The constexpr codes are re-declared in the header because a @triton.jit body
    may not read a plain module global, and the shared forms and record offsets are
    imported, as the module imports them.

    THE MODULE NAME MUST NOT START WITH ``triton``. Triton's dependency finder treats
    every callee whose ``__module__`` starts with its own package name as a builtin
    and leaves it out of the kernel's cache key, so under a ``triton_*`` name a
    mutation that edits only a helper (``_connect``) keys like every other
    helper-only mutant of the same specialization and the cache serves the first
    one's binary to the rest (measured on the off-diagonal gate, 2026-10-08). The
    mutation leg also refuses a mutant whose cache key equals the shipped kernel's or
    an earlier mutant's (:func:`run_mutations`)."""
    header = (
        "import triton\nimport triton.language as tl\n"
        "from meep_gpu.triton_kernels.forms import (axis_present, constitutive_form,\n"
        "                                           curl_form, curl_stencil, record_in)\n"
        f"PERIODIC = tl.constexpr({CODE_OF[PERIODIC]})\n"
        f"METALLIC = tl.constexpr({CODE_OF[METALLIC]})\n"
        f"CONNECT_NONE = tl.constexpr({complex_fields.CONNECTION_KINDS['none']})\n"
        f"CONNECT_PHASE = tl.constexpr({complex_fields.CONNECTION_KINDS['phase']})\n"
        f"CONNECT_NEGATE = tl.constexpr({complex_fields.CONNECTION_KINDS['negate']})\n"
        f"RECORD_HALF_INTEGER = tl.constexpr({RECORD_HALF_INTEGER_WORD})\n"
        f"RECORD_PRESENCE = tl.constexpr({RECORD_PRESENCE_WORD})\n\n")
    handle = tempfile.NamedTemporaryFile("w", suffix="_mutated_complex.py",
                                         delete=False, encoding="utf-8")
    handle.write(header + source)
    handle.close()
    _TEMPORARY.append(handle.name)
    spec = importlib.util.spec_from_file_location(
        "complex_mutant_" + str(len(_TEMPORARY)), handle.name)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = module  # type: ignore[union-attr]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return getattr(module, kernel_name)


#: The record word where the half-integer lines start
#: (``chunk_map.record_index(True, 0, 0)``), restated for the mutant header.
RECORD_HALF_INTEGER_WORD = 12


def shipped_source() -> str:
    """The connection helper first (it may be the mutation target), then both kernels."""
    functions = (complex_fields._connect,
                 complex_fields.bloch_pml_curl_step,
                 complex_fields.bloch_constitutive_step)
    return "\n\n".join(textwrap.dedent(inspect.getsource(f.fn))
                       for f in functions)


_CONNECTED_COPY = re.compile(
    r"(\w+_(?:re|im)) = tl\.where\(at_[xyz], (r_re|r_im), \w+_(?:re|im)\)")
_OLD_WRAP_PLANE = (("at_x", "i == nx - 1"), ("at_y", "j == ny - 1"),
                   ("at_z", "k == nz - 1"))


def _needles(*pairs: Tuple[str, str]) -> Callable[[str], Tuple[str, int]]:
    """A transform that rewrites every pair; its site count is the pairs that matched
    (each needle must match, or the count falls short and the leg reports the miss)."""
    def transform(source: str) -> Tuple[str, int]:
        hits = 0
        for needle, replacement in pairs:
            if needle in source:
                hits += 1
                source = source.replace(needle, replacement)
        return source, hits if hits == len(pairs) else 0
    return transform


def mutate_connection_on_every_lane(source: str) -> Tuple[str, int]:
    """The connected word replaces the operand on EVERY lane, not only plane 0."""
    return _CONNECTED_COPY.sub(r"\1 = \2", source), len(_CONNECTED_COPY.findall(source))


def mutate_connection_on_the_old_wrapped_plane(source: str) -> Tuple[str, int]:
    """INVERSION (the previous design): the connection lands on the last plane, where
    the rolled operand used to carry the phase, instead of the unshifted operand's
    plane 0."""
    hits = 0
    for predicate, old_plane in _OLD_WRAP_PLANE:
        pattern = re.compile(rf"= tl\.where\({predicate}, (r_re|r_im), ")
        source, n = pattern.subn(rf"= tl.where({old_plane}, \1, ", source)
        hits += n
    return source, hits


def _swap_lattice(source: str) -> Tuple[str, int]:
    """The two sub-lattices' membership lines exchanged (a half-cell plane error)."""
    head = "    if BACKWARD:\n        lattice = 0\n    else:\n        lattice = RECORD_HALF_INTEGER\n"
    if head not in source:
        return source, 0
    return source.replace(head, "    if BACKWARD:\n        lattice = RECORD_HALF_INTEGER\n"
                                "    else:\n        lattice = 0\n"), 1


#: name -> (transform, kernel family it is measured through, verdict, state).
#: Verdict: True must be caught, False is a declared null. ``state`` "sparse" seeds
#: most words as zeros of random sign (the negation's sign only reaches a stored word
#: through a stencil whose every operand is zero).
SOURCE_MUTATIONS: Dict[str, Tuple[Callable[[str], Tuple[str, int]], str, bool,
                                  Optional[str]]] = {
    "m2_connection_on_every_lane": (mutate_connection_on_every_lane, "curl", True, None),
    "m3_connection_dropped": (
        _needles(("if CNX:", "if 0:"), ("if CNY:", "if 0:"), ("if CNZ:", "if 0:")),
        "curl", True, None),
    "m4_connection_product_sign_error": (
        _needles(("out_im = (c_re * v_im) + (c_im * v_re)",
                  "out_im = (c_re * v_im) - (c_im * v_re)")), "curl", True, None),
    "m6_connection_on_the_old_wrapped_plane": (
        mutate_connection_on_the_old_wrapped_plane, "curl", True, None),
    "m7_negation_as_unary_minus": (
        _needles(("out_re = v_re * -1.0", "out_re = 0.0 - v_re"),
                 ("out_im = v_im * -1.0", "out_im = 0.0 - v_im")), "curl", True,
        "sparse"),
    "a1_pairwise_stencil": (
        _needles(("curl0_re = curl_stencil(c_y_re, c0_re, b0_re, b_z_re, dtdx, py, pz)",
                  "curl0_re = dtdx * ((c_y_re - c0_re) + (b0_re - b_z_re))")),
        "curl", True, None),
    "a2_whole_grid_recurrence": (
        _needles(("du0 = record_in(cb, lattice + 8, k)", "du0 = live"),
                 ("du1 = record_in(cb, lattice + 0, i)", "du1 = live"),
                 ("du2 = record_in(cb, lattice + 4, j)", "du2 = live")),
        "curl", True, None),
    "a3_constitutive_old_grouping": (
        _needles(("constitutive_form(a_re, prev_re, src_re, kp_0, km_0, dw0)",
                  "tl.where(dw0, (a_re + (kp_0 * src_re)) - (km_0 * prev_re), src_re)")),
        "constitutive", True, None),
    "b1_membership_one_row_off": (
        _needles(("du1 = record_in(cb, lattice + 0, i)",
                  "du1 = record_in(cb, lattice + 0, i + 1)")), "curl", True, None),
    "b2_lattice_swapped": (_swap_lattice, "curl", True, None),
    "b3_fu_store_unmasked": (
        _needles(("tl.store(u0 + 2 * idx, n0_re, mask=live & du0)",
                  "tl.store(u0 + 2 * idx, n0_re, mask=live)")), "curl", True, None),
    "b4_presence_ignored": (
        _needles(("presence = tl.load(cb + RECORD_PRESENCE)", "presence = 7")),
        "curl", True, None),
    "b5_constitutive_select_removed": (
        _needles(("dw0 = record_in(cb, lattice + 0, i)", "dw0 = live")),
        "constitutive", True, None),
    "b6_fw_store_unmasked": (
        _needles(("tl.store(w0 + 2 * idx, src_re, mask=live & dw0)",
                  "tl.store(w0 + 2 * idx, src_re, mask=live)")), "constitutive", True,
        None),
    "b7_inv_eps_on_the_word_index": (
        _needles(("ie = tl.load(e0 + idx, mask=live, other=0.0)",
                  "ie = tl.load(e0 + idx // 2, mask=live, other=0.0)")),
        "constitutive", True, None),
    # A declared null: the fu read outside the recurrence's cells feeds no form there.
    "n1_fu_load_unmasked": (
        _needles(("p_re = tl.load(u0 + 2 * idx, mask=live & du0, other=0.0)",
                  "p_re = tl.load(u0 + 2 * idx, mask=live, other=0.0)")),
        "curl", False, None),
}

#: host mutation -> must_catch (None = record only).
HOST_MUTATIONS: Dict[str, Optional[bool]] = {
    "m1_connection_direction_swapped": True,
    "m5_unit_product_instead_of_copy": None,
}

#: The m1 (conjugation) legs run on these. CONSTRAINT (measured 2026-08-11):
#: a phase-conjugation defect is byte-INVISIBLE on an axis whose phase is
#: exactly or nearly real — the Brillouin edge -1+0j differs from its conjugate
#: only in a signed zero, and a k*L a hair off integer-plus-half leaves an imag
#: ~6e-16 that rounds away against O(1) operands in float32 (a planted
#: un-conjugated wrap passed on brillouin-edge-only and near-real-phase grids,
#: and was caught at the first mutated sub-step on every general-phase grid).
#: Every config in this tuple therefore carries at least one GENERAL phase
#: (general_bloch 0.21/-0.13/0.4, edge_x's 0.37 on y, phased_times_metallic
#: 0.33/-0.4, one_axis_phase_none 0.29); an edit reducing the tuple to
#: edge-only or unphased configs silently disarms m1.
_PHASED_CONFIGS = tuple(c for c in CONFIGS
                        if c["name"] in ("general_bloch", "edge_x",
                                         "phased_times_metallic",
                                         "one_axis_phase_none"))
_METALLIC_CONFIGS = tuple(c for c in CONFIGS
                          if c["name"] in ("phased_times_metallic",
                                           "metallic_k0"))
#: The negation's leg: the Brillouin edge (an exact -1, MEEP's negation) on x.
_EDGE_CONFIGS = tuple(c for c in CONFIGS if c["name"] == "edge_x")
_UNPHASED_AXIS_CONFIGS = tuple(c for c in CONFIGS
                               if c["name"] in ("unphased_all_periodic",
                                                "one_axis_phase_none"))


def run_mutations(results: Dict[str, Any], out_path: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    shipped = shipped_source()
    # The cache-key tripwire (see compile_mutated): every mutant's kernel must key
    # apart from the shipped kernel and from every mutant before it. Hosts without
    # Triton's JIT object (no ``cache_key``) record nothing here.
    seen_keys: Dict[str, str] = {}
    for kernel in (complex_fields.bloch_pml_curl_step,
                   complex_fields.bloch_constitutive_step):
        key = getattr(kernel, "cache_key", None)
        if isinstance(key, str):
            seen_keys[key] = "shipped"
    for name, (transform, family, must_catch, state) in SOURCE_MUTATIONS.items():
        mutated, hits = transform(shipped)
        if hits == 0:
            out[name] = {"error": "the mutation matched nothing; the needle "
                                  "has drifted away from the kernel"}
            log(f"[mut] {name}: NEEDLE MISSED — nothing exercised")
            results["mutations"] = out
            save(results, out_path)
            continue
        entry: Dict[str, Any] = {"sites": hits, "family": family,
                                 "must_catch": must_catch, "state": state or "seeded"}
        kernel_name = ("bloch_pml_curl_step" if family == "curl"
                       else "bloch_constitutive_step")
        counter = CountingKernel(compile_mutated(mutated, kernel_name))
        key = getattr(counter.kernel, "cache_key", None)
        collided = seen_keys.get(key) if isinstance(key, str) else None
        if isinstance(key, str) and collided is None:
            seen_keys[key] = name
        if family == "curl":
            summary = run_synthetic(
                results, out_path, kernel=counter,
                label="mutation:" + name, guards=(False,), shapes=SHAPES[:2],
                configs=(_EDGE_CONFIGS if state == "sparse"
                         else _PHASED_CONFIGS + _METALLIC_CONFIGS[1:]),
                dtdx_values=(0.35,), sparse=0.04 if state == "sparse" else None)
        else:
            summary = run_constitutive(
                results, out_path, kernel=counter,
                label="mutation:" + name, guards=(False,), shapes=SHAPES[:2])
        caught = summary["identical"] < summary["ran"]
        entry.update({"ran": summary["ran"], "identical": summary["identical"],
                      "launches": counter.launches, "caught": caught,
                      "as_expected": caught == must_catch})
        if counter.launches == 0:
            entry["error"] = ("DISARMED: the mutated kernel never launched — "
                              "the leg measured nothing and FAILS")
        if collided is not None:
            entry["error"] = (f"CACHE KEY COLLISION with {collided}: the mutant's "
                              "kernel keys like a binary already compiled, so the "
                              "cache may serve that binary and the verdict measures "
                              "it, not this mutation")
        log(f"[mut] {name}: sites={hits} launches={counter.launches} "
            f"CAUGHT={caught} (must_catch={must_catch})")
        out[name] = entry
        results["mutations"] = out
        save(results, out_path)

    for name, must_catch in HOST_MUTATIONS.items():
        configs = {"m1_connection_direction_swapped": _PHASED_CONFIGS,
                   "m5_unit_product_instead_of_copy": _UNPHASED_AXIS_CONFIGS}[name]
        summary = run_synthetic(
            results, out_path, host_mutation=name,
            label="mutation:" + name, guards=(False,), shapes=SHAPES[:2],
            configs=configs, dtdx_values=(0.35,))
        caught = summary["identical"] < summary["ran"]
        entry = {"host": True, "ran": summary["ran"],
                 "identical": summary["identical"], "caught": caught,
                 "must_catch": must_catch}
        if summary["ran"] == 0:
            entry["error"] = "DISARMED: no case ran"
        elif must_catch is None:
            entry["as_expected"] = True  # recorded, never asserted (m5)
        else:
            entry["as_expected"] = caught == must_catch
        log(f"[mut] {name}: identical={summary['identical']}/{summary['ran']} "
            f"CAUGHT={caught} (must_catch={must_catch})")
        out[name] = entry
        results["mutations"] = out
        save(results, out_path)

    # A connection on a wall is a plan the curl plan refuses by name (the array path
    # connects only a wrapping axis): asserted as a refusal, not stepped.
    shape = SHAPES[0]
    arrays = make_complex_state(cp, tuple(shape), np.random.default_rng(SEED),
                                FIELD_12 + FU_NAMES)
    flat = probe.flatten_coefficients(probe.synthetic_coefficients(cp, tuple(shape),
                                                                   True))
    try:
        plan = complex_fields.plan_complex_pml_curl_from_arrays(
            "step_B", arrays, flat, [CODE_OF[PERIODIC], CODE_OF[METALLIC],
                                     CODE_OF[PERIODIC]],
            (None, None, None), 0.35,
            chunk_bounds=two_run_record(shape))
        plan.connections = (0, complex_fields.CONNECTION_KINDS["phase"], 0)
        complex_fields.ComplexPmlCurlPlan(
            "step_B", shape, 0.35, plan.bc, plan.connections,
            (1.0, 0.0, 0.8, 0.6, 1.0, 0.0), plan.block,
            [arrays[n] for n in ("Bx", "By", "Bz")],
            [arrays["fu_" + n] for n in ("Bx", "By", "Bz")],
            [arrays[n] for n in ("Ex", "Ey", "Ez")],
            [flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
            chunk_bounds=two_run_record(shape))
        refusal = None
    except ValueError as exc:
        refusal = str(exc)
    out["connection_on_a_wall_refused"] = {
        "host": True, "refused": refusal is not None, "reason": refusal,
        "as_expected": refusal is not None}
    if refusal is None:
        out["connection_on_a_wall_refused"]["error"] = (
            "the curl plan accepted a connection on a metallic axis")
    log(f"[mut] connection_on_a_wall_refused: refused={refusal is not None}")
    results["mutations"] = out
    save(results, out_path)
    return out


# ---------------------------------------------------------------------------
# The reference-validation leg — the transcription vs stepping.py itself
# ---------------------------------------------------------------------------

REFERENCE_GRIDS: Tuple[Dict[str, Any], ...] = (
    {"name": "complex_k0_metallic_xy", "cell": (2.0, 1.6, 0.0), "dimensions": 2,
     "boundaries": ("metallic", "metallic", "periodic"), "k": (0.0, 0.0, 0.0)},
    {"name": "bloch_inplane", "cell": (2.0, 1.6, 0.0), "dimensions": 2,
     "boundaries": "periodic", "k": (0.8753, 1.2181, 0.0)},
    {"name": "bloch_one_axis", "cell": (2.0, 1.6, 0.0), "dimensions": 2,
     "boundaries": "periodic", "k": (0.37, 0.0, 0.0)},
    # resolution 10, Lx=2.0 -> nx_full=20; k=0.25 makes k*n == 0.5*res exactly:
    # the Brillouin-edge branch, phase EXACTLY -1+0j (grid.py:1011-1017).
    {"name": "brillouin_edge_x", "cell": (2.0, 1.6, 0.0), "dimensions": 2,
     "boundaries": "periodic", "k": (0.25, 0.0, 0.0)},
    # refl-angular's class: k on the axis that carries the absorber (admitted;
    # measured 2.82e-07 in the array path's own validation).
    {"name": "k_on_absorbing_axis", "cell": (0.8, 0.8, 2.0), "dimensions": 3,
     "boundaries": "periodic", "k": (0.0, 0.0, 1.25)},
)
REFERENCE_STEPS = 4
REFERENCE_RESOLUTION = 10.0
REFERENCE_COURANT = 0.35  # non-power-of-two on the laptop leg as well


def _build_reference_fields(xp, spec: Dict[str, Any]):
    grid = Grid(resolution=REFERENCE_RESOLUTION, cell_size=spec["cell"],
                boundaries=spec["boundaries"], dimensions=spec["dimensions"],
                courant=REFERENCE_COURANT, k_point=tuple(spec["k"]), xp=xp)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_pml_storage()
    index = np.arange(int(np.prod(grid.shape)),
                      dtype=np.float32).reshape(grid.shape)
    epsilon = (1.45 + 0.30 * np.sin(index * np.float32(0.037))).astype(np.float32)
    inverse = (np.float32(1.0) / epsilon).astype(np.float32)
    fields.set_isotropic_epsilon_volume(xp.asarray(np.ascontiguousarray(epsilon)),
                                        xp.asarray(np.ascontiguousarray(inverse)))
    # Per-axis thickness: an invariant axis (n = 1) carries no layer at all,
    # exactly as the shared probe's real_pml_layer builds it.
    thickness = tuple((2, 2) if grid.shape[axis] >= 6 else (0, 0)
                      for axis in range(3))
    pml = PML(grid=grid, thickness=thickness)
    rng = np.random.default_rng(SEED + 13)
    for name in ALL_STATE:
        getattr(fields, name)[...] = xp.asarray(_seed_host_complex(
            tuple(grid.shape), rng))
    return grid, fields, pml


def _state_sha256(xp, arrays: Dict[str, Any]) -> str:
    hasher = hashlib.sha256()
    for name in sorted(arrays):
        hasher.update(np.ascontiguousarray(probe.to_host(arrays[name])).tobytes())
    return hasher.hexdigest()


def run_reference_validation(results: Dict[str, Any], out_path: str, xp,
                             backend_name: str) -> Dict[str, Any]:
    """The transcription against stepping.step_B/update_H/step_D/update_E on
    real objects — what makes the synthetic leg's reference a statement about
    the CONTRACT rather than about this file. Emits a per-step state sha256
    chain so the NumPy and CuPy backends' bytes can be compared offline."""
    cases: List[Dict[str, Any]] = []
    for spec in REFERENCE_GRIDS:
        started = time.time()
        grid, fields, pml = _build_reference_fields(xp, spec)
        kinds = stepping._boundary_kinds(grid, pml)
        phases = tuple(grid.bloch_phase(axis) for axis in range(3))
        dtdx = grid.dt / grid.dx
        arrays = {name: getattr(fields, name).copy() for name in ALL_STATE}
        arrays.update({"inv_eps_" + name: fields.inverse_epsilon_for(name)
                       for name in ("Ex", "Ey", "Ez")})
        curl_tables = {sub: probe.layer_coefficients(pml, sub == "step_B")
                       for sub in CURL_SUB_STEPS}
        constitutive_tables = {
            side: probe.layer_constitutive_coefficients(pml, side == "E")
            for side in ("H", "E")}

        case: Dict[str, Any] = {
            "spec": dict(spec), "backend": backend_name,
            "shape": list(grid.shape), "kinds": list(kinds),
            "phases": [repr(p) for p in phases],
            "dtdx": repr(dtdx), "steps": REFERENCE_STEPS,
            "state_sha256_per_step": [],
        }
        parts: Dict[str, Any] = {}
        case["per_step_identical"] = []
        case["first_divergence"] = None
        record = None
        for step in range(1, REFERENCE_STEPS + 1):
            for slot, engine_call, tables in (
                    ("step_B", stepping.step_B, None),
                    ("update_H", stepping.update_H, "H"),
                    ("step_D", stepping.step_D, None),
                    ("update_E", stepping.update_E, "E")):
                start_h = slot == "update_H" and not fields.h_separated
                engine_call(fields, pml)
                # The oracle's own division: its first PML sub-step installs it.
                record = region_plan_record(fields) if record is None else record
                if slot in CURL_SUB_STEPS:
                    complex_pml_step(xp, arrays, curl_tables[slot], dtdx, slot,
                                     kinds, phases, record)
                else:
                    complex_constitutive_step(xp, arrays,
                                              constitutive_tables[tables], tables,
                                              record, start_h=start_h)
                sub_parts = {name: bit_compare(arrays[name],
                                               getattr(fields, name))
                             for name in ALL_STATE}
                identical = all(p["bit_identical"] for p in sub_parts.values())
                if not identical and case["first_divergence"] is None:
                    case["first_divergence"] = f"step {step} after {slot}"
                parts = sub_parts
            case["per_step_identical"].append(
                all(p["bit_identical"] for p in parts.values())
                and case["first_divergence"] is None)
            case["state_sha256_per_step"].append(
                _state_sha256(xp, {name: arrays[name] for name in ALL_STATE}))
        case["verdict"] = combine(parts)
        if case["first_divergence"] is not None:
            case["verdict"]["bit_identical"] = False
        case["seconds"] = round(time.time() - started, 3)
        cases.append(case)
        log(f"[reference:{backend_name}] {spec['name']} shape={tuple(grid.shape)} "
            f"kinds={kinds} identical={case['verdict']['bit_identical']} "
            f"(differing={case['verdict']['differing_floats']}/"
            f"{case['verdict']['total_floats']}) ({case['seconds']} s)")
        results.setdefault("reference", {})[backend_name] = {
            "ran": len(cases),
            "identical": sum(int(c["verdict"]["bit_identical"]) for c in cases),
            "steps_per_case": REFERENCE_STEPS,
            "cases": cases}
        save(results, out_path)
    return results["reference"][backend_name]


# ---------------------------------------------------------------------------
# The engine leg — the predicate + engine-route plans against stepping, on CuPy
# ---------------------------------------------------------------------------

ENGINE_STEPS = 4

#: The engine phase-table host mutation (the table built from k via cmath
#: instead of ``grid.bloch_phase``), asserted at the layer where each case is
#: byte-VISIBLE. The field-level constraint is the m1 note above: at the exact
#: Brillouin edge the two tables differ by 1.2246e-16j — sub-half-ulp of every
#: nonzero float32 word — so on a generically seeded state ZERO field words
#: can move (measured: 0/7680 on the NumPy transcription, 0/15360 on device)
#: and a field-level must-catch there contradicts this gate's own
#: design. The seam (the engine consumes the TABLE, not a recomputation from
#: k) is instead proven where a defect is visible:
#:
#: * at the table itself, as complex128 BYTES: the mutated table must differ
#:   on the edge grid (deterministic — grid.py:1015-1016 returns EXACTLY
#:   -1+0j there), and must be byte-IDENTICAL on a generic-k grid, where
#:   grid.py:1017 IS the cmath route — which is what makes each field-level
#:   null below a PREDICTION, recorded like m5 / null_phase_on_metallic_axis,
#:   never a silent excuse;
#: * at field level on ``imag_only``: every state word's real component
#:   zeroed, so wrapped words carry EXACTLY-zero real parts — the one operand
#:   class the near-real delta moves (to normal-range float32 magnitudes,
#:   ~|imag|*1.22e-16, nowhere near underflow). Measured on the NumPy
#:   transcription: 124/7680 words across By/Bz/Dy/Dz + fu (the two curls
#:   that read the x-wrapped plane); pinned by test_triton_complex_fields.py.
ENGINE_PHASE_MUTATION_CASES: Tuple[Dict[str, Any], ...] = (
    {"grid": "brillouin_edge_x", "state": "seeded",
     "table_must_differ": True, "must_catch": False,
     "null_reason": (
         "predicted field-level null: the naive phase differs from the table "
         "by 1.2246e-16j, sub-half-ulp of every nonzero float32 word (the m1 "
         "constraint note), so no generically seeded word can move; the catch "
         "for this defect is the table-byte assertion plus the imag_only "
         "field-level case")},
    {"grid": "brillouin_edge_x", "state": "imag_only",
     "table_must_differ": True, "must_catch": False,
     "null_reason": (
         "predicted field-level null under MEEP's connection: the naive phase "
         "-1 + 1.2246e-16j is within CONNECT_TOLERANCE (1e-13) of -1, so "
         "connection_arguments classifies it as the same negation as the exact "
         "table and the kernel never multiplies by it, whatever the state (MEEP "
         "classifies it the same way, boundaries.cpp:359-366); the catch for this "
         "defect is the table-byte assertion")},
    {"grid": "bloch_one_axis", "state": "seeded",
     "table_must_differ": False, "must_catch": False,
     "null_reason": (
         "predicted null at every layer: at generic k grid.py:1017 computes "
         "the phase VIA cmath, so the mutated table is byte-identical and the "
         "mutation is inert — asserted at the table seam")},
)


def naive_phase_table(grid) -> Tuple[Optional[complex], ...]:
    """The mutated table: phase from k via cmath, skipping ``grid.bloch_phase``
    and with it the exact Brillouin-edge branch (grid.py:1011-1017)."""
    lengths = (grid.Lx, grid.Ly, grid.Lz)
    return tuple(
        None if float(k) == 0.0 else cmath.exp(2j * math.pi * float(k) * L)
        for k, L in zip(grid.k_point, lengths))


def phase_tables_bytes_differ(mutated: Sequence[Optional[complex]],
                              table: Sequence[Optional[complex]]) -> bool:
    """The table-seam comparator: per axis, as complex128 BYTES — the width
    the table is built at. A complex64 or field-level comparison is exactly
    what the m1 constraint rules out at the Brillouin edge."""
    for mutated_phase, table_phase in zip(mutated, table):
        if (mutated_phase is None) != (table_phase is None):
            return True
        if mutated_phase is not None and (
                np.complex128(mutated_phase).tobytes()
                != np.complex128(table_phase).tobytes()):
            return True
    return False


def build_phase_mutation_case(xp, case: Dict[str, Any]):
    """Grid + two identical field states for one ENGINE_PHASE_MUTATION_CASES
    entry — backend-generic, so the laptop test measures the SAME construction
    on the NumPy transcription that the device leg runs through the plan."""
    spec = next(s for s in REFERENCE_GRIDS if s["name"] == case["grid"])
    grid, fields, pml = _build_reference_fields(xp, spec)
    if case["state"] == "imag_only":
        # Zero every real component: wrapped words then carry EXACTLY-zero
        # real parts, the one operand class the near-real phase delta moves.
        for name in ALL_STATE:
            array = getattr(fields, name)
            array[...] = array.imag * array.dtype.type(1j)
    reference = Fields(grid=grid, force_complex_fields=True)
    reference.enable_pml_storage()
    reference.set_isotropic_epsilon_volume(fields.epsilon_for("Ex"),
                                           fields.inverse_epsilon_for("Ex"))
    for name in ALL_STATE:
        getattr(reference, name)[...] = getattr(fields, name)
    return grid, fields, reference, pml


def run_engine(results: Dict[str, Any], out_path: str) -> Dict[str, Any]:
    cases: List[Dict[str, Any]] = []
    for spec in REFERENCE_GRIDS:
        started = time.time()
        grid, fields, pml = _build_reference_fields(cp, spec)
        reference = Fields(grid=grid, force_complex_fields=True)
        reference.enable_pml_storage()
        reference.set_isotropic_epsilon_volume(fields.epsilon_for("Ex"),
                                               fields.inverse_epsilon_for("Ex"))
        for name in ALL_STATE:
            getattr(reference, name)[...] = getattr(fields, name)

        case: Dict[str, Any] = {"spec": dict(spec), "shape": list(grid.shape),
                                "steps": ENGINE_STEPS}
        coverage = {
            "step_B": complex_fields.complex_pml_curl_coverage(
                fields, pml, "step_B"),
            "update_H": complex_fields.complex_constitutive_coverage(
                fields, pml, "H"),
            "step_D": complex_fields.complex_pml_curl_coverage(
                fields, pml, "step_D"),
            "update_E": complex_fields.complex_constitutive_coverage(
                fields, pml, "E"),
        }
        refused = {slot: list(verdict.reasons)
                   for slot, verdict in coverage.items() if not verdict.covered}
        if refused:
            case["error"] = ("the predicate refused a configuration the gate "
                             "built: " + repr(refused))
            cases.append(case)
            log(f"[engine] {spec['name']} REFUSED: {refused}")
            results["engine"] = {"ran": len(cases), "identical": 0, "cases": cases}
            save(results, out_path)
            continue
        plans = {
            "step_B": complex_fields.plan_complex_pml_curl(
                fields, pml, "step_B"),
            "update_H": complex_fields.plan_complex_constitutive(
                fields, pml, "H"),
            "step_D": complex_fields.plan_complex_pml_curl(
                fields, pml, "step_D"),
            "update_E": complex_fields.plan_complex_constitutive(
                fields, pml, "E"),
        }
        case["connections"] = list(plans["step_B"].connections)
        parts: Dict[str, Any] = {}
        for _step in range(ENGINE_STEPS):
            for slot, engine_call in (("step_B", stepping.step_B),
                                      ("update_H", stepping.update_H),
                                      ("step_D", stepping.step_D),
                                      ("update_E", stepping.update_E)):
                plans[slot].run()
                engine_call(reference, pml)
                cp.cuda.runtime.deviceSynchronize()
                for name in ALL_STATE:
                    parts[name] = bit_compare(getattr(fields, name),
                                              getattr(reference, name))
        case["verdict"] = combine(parts)
        case["seconds"] = round(time.time() - started, 3)
        cases.append(case)
        log(f"[engine] {spec['name']} shape={tuple(grid.shape)} "
            f"connections={case['connections']} "
            f"identical={case['verdict']['bit_identical']} "
            f"(differing={case['verdict']['differing_floats']}/"
            f"{case['verdict']['total_floats']}) ({case['seconds']} s)")
        results["engine"] = {
            "ran": len(cases),
            "identical": sum(int(c.get("verdict", {}).get("bit_identical", False))
                             for c in cases),
            "cases": cases}
        save(results, out_path)

    # Host mutation: the phase table built from k via cmath instead of
    # grid.bloch_phase — asserted per ENGINE_PHASE_MUTATION_CASES at the layer
    # where each case is byte-visible (the table seam as complex128 bytes; the
    # fields only on the imag_only construction), with every generic-state
    # field null recorded as the prediction it is, never silently expected.
    mutation_cases: List[Dict[str, Any]] = []
    for case in ENGINE_PHASE_MUTATION_CASES:
        grid, fields, reference, pml = build_phase_mutation_case(cp, case)
        kinds = stepping._boundary_kinds(grid, pml)
        naive_phases = naive_phase_table(grid)
        grid_phases = tuple(grid.bloch_phase(axis) for axis in range(3))
        table_differs = phase_tables_bytes_differ(naive_phases, grid_phases)
        parts = {}
        ran = 0
        for sub_step in CURL_SUB_STEPS:
            arrays = {name: getattr(fields, name)
                      for name in FIELD_12 + FU_NAMES}
            flat = probe.flatten_coefficients(
                probe.layer_coefficients(pml, sub_step == "step_B"))
            plan = complex_fields.plan_complex_pml_curl_from_arrays(
                sub_step, arrays, flat, [CODE_OF[k] for k in kinds],
                naive_phases, float(grid.dt / grid.dx),
                chunk_bounds=layer_record(grid, pml))
            plan.run()
            ran += 1
            getattr(stepping, sub_step)(reference, pml)
            cp.cuda.runtime.deviceSynchronize()
            targets = ("Bx", "By", "Bz") if sub_step == "step_B" else ("Dx", "Dy", "Dz")
            for name in targets + tuple("fu_" + t for t in targets):
                parts[sub_step + ":" + name] = bit_compare(
                    getattr(fields, name), getattr(reference, name))
        verdict = combine(parts)
        caught = not verdict["bit_identical"]
        entry = {"grid": case["grid"], "state": case["state"],
                 "table_bytes_differ": table_differs,
                 "table_must_differ": case["table_must_differ"],
                 "caught": caught, "must_catch": case["must_catch"],
                 "as_expected": (table_differs == case["table_must_differ"]
                                 and caught == case["must_catch"]),
                 "ran_sub_steps": ran,
                 "differing_floats": verdict["differing_floats"],
                 "total_floats": verdict["total_floats"],
                 "naive_phases": [repr(p) for p in naive_phases],
                 "grid_phases": [repr(p) for p in grid_phases]}
        if "null_reason" in case:
            entry["null_reason"] = case["null_reason"]
        mutation_cases.append(entry)
        log(f"[engine-mut] phase_from_k_directly on {case['grid']}/"
            f"{case['state']}: table_bytes_differ={table_differs} "
            f"(must={case['table_must_differ']}) field caught={caught} "
            f"(must_catch={case['must_catch']}, "
            f"differing={verdict['differing_floats']}/"
            f"{verdict['total_floats']})")
    results["engine_host_mutation"] = {"phase_from_k_directly": mutation_cases}
    save(results, out_path)
    return results["engine"]


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------

DEFAULT_LEGS = "reference,synthetic,constitutive,multi_step,mutations,engine"
DEVICE_LEGS = ("synthetic", "constitutive", "multi_step", "mutations", "engine")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--legs", default=DEFAULT_LEGS)
    args = parser.parse_args(argv)
    legs = tuple(name.strip() for name in args.legs.split(",") if name.strip())
    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."

    # The ship policy, installed BEFORE any CuPy compile (a clean no-op without
    # CuPy; raises at startup on a cache dir that could mix policies). Every
    # licensed byte below this line is an IEEE-keep byte or a refusal.
    install_ftz_strip()

    device_available = cp is not None and _TRITON_AVAILABLE
    results: Dict[str, Any] = {
        "seed": SEED,
        "granularity": "sub-step (the composition probe holds the whole-step claim)",
        "correctness_only": "no throughput or timing claims in this artifact",
        "step_budgets": {"reference": REFERENCE_STEPS, "engine": ENGINE_STEPS,
                         "multi_step_cycles": MULTI_STEP_COUNT,
                         "note": "bit-identity is claimed for exactly these "
                                 "stated budgets and no further"},
        "sweep": {"shapes": [list(s) for s in SHAPES],
                  "configs": [{"name": c["name"],
                               "boundaries": list(c["boundaries"]),
                               "phases": [repr(p) for p in c["phases"]]}
                              for c in CONFIGS],
                  "dtdx": [repr(v) for v in DTDX]},
        "host": {"cupy": None if cp is None else cp.__version__,
                 "triton": triton.__version__ if _TRITON_AVAILABLE else None,
                 "numpy": np.__version__,
                 "python": sys.version.split()[0]},
        "subnormal_policy": policy_stamp("cupy"),  # refreshed with final counters
        "skipped_legs": {},
    }
    os.makedirs(out_dir, exist_ok=True)
    write_provenance(out_dir)
    save(results, args.out)

    failures: List[str] = []

    if "reference" in legs:
        summary = run_reference_validation(results, args.out, np, "numpy")
        if summary["identical"] != summary["ran"]:
            failures.append(f"reference transcription differs from stepping.py "
                            f"on numpy: {summary['identical']}/{summary['ran']}")
        if cp is not None:
            summary = run_reference_validation(results, args.out, cp, "cupy")
            if summary["identical"] != summary["ran"]:
                failures.append(f"reference transcription differs from "
                                f"stepping.py on cupy: "
                                f"{summary['identical']}/{summary['ran']}")
        else:
            log("[reference] cupy leg SKIPPED cleanly: no cupy on this host")

    for leg in DEVICE_LEGS:
        if leg not in legs:
            continue
        if not device_available:
            reason = ("no CUDA/triton on this host — this leg runs on the "
                      "measurement machine; nothing was measured here")
            results["skipped_legs"][leg] = reason
            log(f"[{leg}] SKIPPED cleanly: {reason}")
            save(results, args.out)
            continue
        if leg == "synthetic":
            summary = run_synthetic(results, args.out)
            if not summary["guarded_pass"]:
                failures.append(f"synthetic curl sweep: "
                                f"{summary['guarded_identical']}/"
                                f"{summary['guarded_ran']} guarded identical")
            log(f"[SUMMARY] synthetic curl guarded "
                f"{summary['guarded_identical']}/{summary['guarded_ran']}")
        elif leg == "constitutive":
            summary = run_constitutive(results, args.out)
            if not summary["guarded_pass"]:
                failures.append(f"constitutive sweep: "
                                f"{summary['guarded_identical']}/"
                                f"{summary['guarded_ran']} guarded identical")
        elif leg == "multi_step":
            cases = [multi_step_case(shape, config, 0.35)
                     for shape in SHAPES[:2]
                     for config in CONFIGS
                     if config.get("only_shapes") is None]
            ran = [c for c in cases if not c.get("skipped")]
            identical = sum(int(c["verdict"]["bit_identical"]) for c in ran)
            results["multi_step"] = {"ran": len(ran), "identical": identical,
                                     "cycles": MULTI_STEP_COUNT, "cases": cases}
            if identical != len(ran):
                failures.append(f"multi-step: {identical}/{len(ran)} identical")
            log(f"[SUMMARY] multi-step {identical}/{len(ran)} identical over "
                f"{MULTI_STEP_COUNT} cycles")
            save(results, args.out)
        elif leg == "mutations":
            out = run_mutations(results, args.out)
            for name, entry in out.items():
                if entry.get("error"):
                    failures.append(f"mutation {name}: {entry['error']}")
                elif entry.get("as_expected") is False:
                    failures.append(f"mutation {name}: caught="
                                    f"{entry.get('caught')} but must_catch="
                                    f"{entry.get('must_catch')}")
        elif leg == "engine":
            summary = run_engine(results, args.out)
            if summary["identical"] != summary["ran"]:
                failures.append(f"engine leg: {summary['identical']}/"
                                f"{summary['ran']} identical")
            for entry in results.get("engine_host_mutation", {}).get(
                    "phase_from_k_directly", []):
                if not entry["as_expected"]:
                    failures.append(
                        f"engine host mutation on {entry['grid']}/"
                        f"{entry['state']}: table_bytes_differ="
                        f"{entry['table_bytes_differ']} (expected "
                        f"{entry['table_must_differ']}), field caught="
                        f"{entry['caught']} (expected {entry['must_catch']})")

    # The policy the whole artifact certifies under, with FINAL strip counters
    # (every leg's compiles included). Any residual problem is a failure even
    # if every leg passed: bytes of unconfirmed policy certify nothing.
    if cp is not None:
        for reason in ftz_strip_license_reasons():
            message = "subnormal policy: " + reason
            if message not in failures and not any(
                    reason in existing for existing in failures):
                failures.append(message)
    results["subnormal_policy"] = policy_stamp("cupy")

    results["summary"] = {
        "status": "passed" if not failures else "FAILED",
        "failures": failures,
        "device_legs_ran": device_available,
        "certified_under_subnormal_policy":
            results["subnormal_policy"].get("policy"),
    }
    save(results, args.out)
    log(f"[done] {args.out} status={results['summary']['status']} "
        f"failures={failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
