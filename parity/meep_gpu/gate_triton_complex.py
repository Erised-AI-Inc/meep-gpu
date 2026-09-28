"""Bit-identity gate for the complex-field (Bloch / force_complex_fields) Triton kernels.

The module under test is ``meep_gpu/triton_kernels/complex_fields.py``, and it IS
on the dispatch path: ``launch.plan_step`` calls its two coverage predicates and
``fastpath._decide`` loads the expansion artifact and forwards it into that
composer, with ``fastpath.ARM_CERTIFICATION`` carrying the 'complex PML' and
'complex' arms over the step_B/step_D/update_H/update_E slots. This paragraph said
"NOT wired into production dispatch (``plan_fast_path`` keeps returning None)"
until 2026-08-15, when ``plan_step`` on a complex64 Grid/Fields/PML triple was
measured emitting per-arm reasons prefixed 'complex PML: ' — so what keeps an
unlicensed arm out of a kernel is this gate plus the module's refusal enumeration,
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

WHY THE EXPANSION PROBE COMES FIRST. The reference implementation's complex
multiply is a platform fact (NumPy on the dev laptop measured FMA_V1: rounded
inner products, fused outer add; a different dispatch is NAIVE; a plane-wise
mixed-dtype loop would be neither). ``complex_fields.EXPANSION`` binds only to a
measured probe artifact — this file writes that artifact from its own preflight
(``measure_expansion_record``) and the coverage predicate refuses without one.
A platform whose DISCRIMINATING patterns disagree, or whose bytes match neither
arm, is a refusal BY NAME, recorded with example words so the next arm can be
designed from the artifact.

WHAT THE PROBE MEASURES IS POLICY-CONDITIONAL, and the candidates are cut under
the policy in force. Cutting them under the OTHER policy makes the platform match
NOTHING — the 54/48/125-word NEITHER of 2026-08-11, which was never a true reading
of a flushing platform, only a comparison across a policy boundary.

WHERE THE DISCRIMINATING POWER LIVES IS A PROPERTY OF THE VECTORS, not of the
arithmetic, and getting that backwards is what cost this probe its power under
flush. Until 2026-08-15 every row that separated FMA_V1 from NAIVE on the
mixed-dtype orientations fed the device a SUBNORMAL OPERAND (:data:`_SMALL`),
which the ship policy destroys on first use: matched candidates then answered
FMA_V1 uniquely on ``c8_mul_c8`` and AMBIGUOUS_BOTH on all three mixed patterns
(0 of 4560 words each; results/complex_expansion_flush_coincidence_2026-08-15/),
so the licence for the two coefficient-multiply helpers was inherited from a
different orientation. :data:`_NORMAL_UNDERFLOWING_ZR` — all-normal operands
whose PRODUCT underflows — has no subnormal operand to destroy and separates the
arms under BOTH policies. With it every pattern discriminates, measured on the
probe's own vectors, candidate against candidate: ``c8_mul_f4_field_left`` and
``f4_mul_c8_coefficient_left`` 0 -> 128 words apart under flush and 6 -> 74 under
keep, ``python_float_left`` 0 -> 96-128 under flush and 6 under keep either way
(its coefficient is a broadcast scalar, so only some scalars underflow the added
rows), ``c8_mul_c8`` 1088 under both and never blind. ``measure_expansion_record``
REFUSES to emit a record whose operand classes are measurably blind under the
policy in force.
A pattern that cannot discriminate is therefore excluded from the agreement test
rather than vetoing it, and when NOTHING discriminates the arm falls back to
``complex_fields.ENVIRONMENT_DEFAULTS`` — the arm already measured for this
execution environment — recorded as defaulted, never as measured. Two blockers
sit before the rule and are fixed here too: the probe's vectors are built from
WORDS (a flushing process cannot regenerate a subnormal through a float
conversion) and ``fma32``/``mul32`` are integer-exact (``_self_check_fma32``'s
"-0 on underflow" identity is unsatisfiable on a flushing FPU otherwise, and it
raised AssertionError in the ship configuration).

SUBNORMAL POLICY — THE STRIP IS THE SHIP CONFIGURATION. CuPy 13.5.1 appends
``-ftz=true`` to every NVRTC compile (cupy/cuda/compiler.py:552), flushing
float32 subnormals on the device; the engine ships IEEE subnormal-KEEP, the only
alignment direction (NVRTC rejects a duplicate ``-ftz`` option and Triton has no
flush knob — settled in ``results/device_subnormal_policy_2026-08-11/``, jobs
2324-2326). Job 2327 measured the mixed-dtype orientations through that flush
lens and matched NO licensed arm (54/48/125 mismatch words); the 2026-08-11
diagnosis reproduced all 21 mismatch counts lane-exactly from ftz semantics
alone, and under the strip the platform is exact FMA_V1 on every orientation
(job 2328, byte-identical to the host IEEE-keep emulation on all 2280 vectors).
WHAT THAT REPRODUCTION DOES AND DOES NOT CERTIFY, because it was once read as
more than it is: it establishes that the mismatches are FTZ semantics — operands
and results flushed — and nothing else. It does NOT certify WHAT THE TININESS
TEST IS APPLIED TO. Measured 2026-08-15 and again 2026-08-16: on that
2280-vector set all three candidate rules — the exact value, the value rounded
to 24 significant bits with unbounded exponent, and the delivered word — produce
IDENTICAL candidate words on every pattern and every arm, so the control is
blind to the question by construction. The class that separates the delivered
word from the other two arrived with :data:`_NORMAL_UNDERFLOWING_ZR`; the class
that separates the exact value from the rounded one is narrower still and is
reached by no vector this gate cuts. The rule is settled against hardware below,
not here.
This gate therefore installs the demonstrated strip — wrap
``cupy.cuda.compiler.compile_using_nvrtc``, filter ``-ftz=true``, private
policy-suffixed ``CUPY_CACHE_DIR`` (the cache key is computed above the seam) —
BEFORE any CuPy compile, counts the option tuples it stripped, refuses to
license when the strip cannot be confirmed exercised, and stamps every artifact
with the policy it certified under: **an expansion licence holds under the
policy it was cut under and does not transfer**. With
:data:`_NORMAL_UNDERFLOWING_ZR` in the vectors every orientation discriminates
under BOTH policies, so the arm is measured rather than inherited on either; the
sentence here used to say the mixed orientations "cannot discriminate at all"
under flush, which described the vector set as it stood before 2026-08-15 and is
measurably false of the one this gate now cuts.

Legs, in order:

* ``expansion``     — the preflight probe. Classifies the platform's complex
  multiply per call-site orientation (:data:`complex_fields.PROBE_PATTERNS`)
  against float32 candidates cut UNDER THE POLICY IN FORCE (fma emulated with
  integer arithmetic, so no double rounding; under flush, PTX .ftz semantics on
  top), on random + signed-zero + underflow vectors whose words are built
  without the FPU. Writes the probe artifact JSON — patterns, per-pattern
  arm-vs-arm disagreement, candidate policy, environment and vector census —
  for the other legs and for the composition probe.
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

    python -u gate_triton_complex.py --legs expansion,reference \\
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
import platform
import re
import sys
import tempfile
import textwrap
import time
from fractions import Fraction
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

from meep_gpu import stepping  # noqa: E402
from meep_gpu import subnormal_policy as _policy  # noqa: E402
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
# run_boundedness_aligned.py (jobs 2324-2326) and certified for this gate's
# expansion by the 2026-08-11 diagnosis (results/complex_expansion_diagnosis_
# 2026-08-11/, job 2328): wrapping ``cupy.cuda.compiler.compile_using_nvrtc``
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
    "the seam (results/device_subnormal_policy_2026-08-11/, jobs 2324-2326; "
    "certified for this gate's expansion by job 2328)")

POLICY_DEPENDENCE = (
    "the FMA_V1 classification and every license derived from it hold UNDER "
    "THE STRIPPED IEEE-KEEP POLICY ONLY — the ship configuration wherever the "
    "default request match_meep resolves to 'keep' (arm64, or any host with no "
    "MEEP to measure); on x86 with MEEP imported it resolves to 'flush' and the "
    "flush stamp's dependence applies instead. Bytes cut under the other policy "
    "differ in the subnormal range. Scoring a flushed device against these KEPT "
    "candidates is what produced the 54/48/125-word NEITHER of job 2327 — a "
    "broken comparison, not a platform that matches no arm "
    "(results/complex_expansion_flush_coincidence_2026-08-15/). Every "
    "orientation in this record DISCRIMINATES under the policy it was cut "
    "under, measured candidate-against-candidate on these vectors: the two "
    "licensable arms are 1088 words apart on c8_mul_c8 under both policies, "
    "128 (flush) / 74 (keep) apart on c8_mul_f4_field_left and "
    "f4_mul_c8_coefficient_left, and 128-96 (flush) / 6 (keep) apart on "
    "python_float_left across its four scalars; measure_expansion_record "
    "refuses to emit a record whose operand classes are blind. Before "
    "2026-08-15 all three zero-imaginary orientations were 0 of 4560 words "
    "apart under flush — every row that separated the arms there fed the "
    "device a subnormal OPERAND, which flush destroys — and this sentence "
    "recorded that as a property of the platform rather than of the vectors "
    "(_NORMAL_UNDERFLOWING_ZR is the class that repaired it).")

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
    "what a cross-policy comparison checks. THE EXPANSION EVIDENCE THIS POLICY "
    "CARRIES, because flush is where it was thinnest: every orientation "
    "discriminates here — measured candidate-against-candidate on this record's "
    "own vectors, the two licensable arms are 1088 words apart on c8_mul_c8, "
    "128 apart on c8_mul_f4_field_left and f4_mul_c8_coefficient_left, and "
    "128-96 apart on python_float_left across its four scalars. That last "
    "range read 128-64 until the FTZ candidate emulation stopped testing "
    "tininess on the DELIVERED float32 word and started testing it on the "
    "result rounded to 24 significant bits with UNBOUNDED exponent, which is "
    "what IEEE 754 §7.5, the Intel SDM (Vol. 1 §4.9.1.5 with §10.2.3.3) and the "
    "measured hardware all say: at scalar 0.5 the tie rows used to be rounded "
    "up onto the smallest normal by both arms, and once they underflow as the "
    "hardware underflows them the arms split on the sign of the resulting zero "
    "on 32 further rows (all zr=0x80FFFFFF). The correction "
    "RAISED this probe's discriminating power; it did not lower it. The "
    "intermediate rule installed on 2026-08-15, which tested the EXACT value "
    "before rounding, produces these same numbers — measured, it moves 0 of "
    "75,308 candidate words across every pattern and scalar cut here — so this "
    "range does not distinguish the two, and the hardware measurement does. "
    "Before "
    "2026-08-15 all three zero-imaginary orientations were 0 of 4560 words "
    "apart under this policy, because every row that separated the arms there "
    "fed the device a subnormal OPERAND and flush destroys those on first use; "
    "_NORMAL_UNDERFLOWING_ZR (all-normal operands, underflowing PRODUCT) is the "
    "class that survives flush and separates them, and "
    "measure_expansion_record refuses to emit a record whose operand classes "
    "are measurably blind.")

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
    Measured: the complex stored-E gate installed ``keep`` (its arms carry
    ``complex_fields.CERTIFIED_UNDER_SUBNORMAL_POLICY = "keep"``) and then
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
    """Why the bytes measured so far may NOT license an expansion, by name.

    Empty means: the strip is installed and every NVRTC option tuple observed
    lost its '-ftz=true' (or a policy-suffixed pre-populated cache explains a
    compile-free run) — the measured bytes are the ship configuration's.
    """
    state = _FTZ_STRIP
    if state is None or not state.get("installed"):
        return [
            "the subnormal policy is not installed: the expansion may only be "
            "licensed under the policy this engine ships in, driven onto every "
            "executor and measured there; a run that certified nothing about "
            "its own subnormal behaviour certifies the wrong binaries, and it "
            "cannot even build honest candidates — they are cut under the "
            "policy in force, and with none in force there is nothing to cut "
            "them under"]
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


def probe_record_policy_reasons(record: Any) -> List[str]:
    """A probe artifact may only license bytes cut under the stripped policy.

    ``None`` (missing artifact) passes through: absence is already a refusal
    by name downstream; this check is for a PRESENT artifact whose bytes were
    cut under the wrong policy, or before artifacts stated their policy.
    """
    if record is None:
        return []
    stamp = record.get("subnormal_policy") if isinstance(record, dict) else None
    policy = stamp.get("policy") if isinstance(stamp, dict) else None
    expected = active_policy_name()
    if policy != expected:
        return [
            f"the probe artifact does not certify the policy this run is cut "
            f"under ({expected!r}); it carries {policy!r} — bytes cut under the "
            f"OTHER subnormal policy differ in the subnormal range, and an "
            f"artifact from before artifacts stated their policy states nothing"]
    if resolved_policy() == _policy.FLUSH:
        # Nothing is stripped under flush, so the strip counters are structurally
        # zero and checking them would refuse every honest flush artifact.
        return []
    calls = stamp.get("nvrtc_calls")
    removed = stamp.get("ftz_removed")
    if not isinstance(calls, int) or not isinstance(removed, int):
        return ["the probe artifact's policy stamp carries no strip counters; "
                "an unverifiable stamp is not a certification"]
    if calls == 0 and not stamp.get("cache_preexisting"):
        return ["the probe artifact's stamp records zero NVRTC compiles under "
                "a fresh cache: the strip was never exercised when the record "
                "was cut, so its bytes cannot be attributed to the stripped "
                "policy"]
    if calls > 0 and removed < calls:
        return [f"the probe artifact's stamp records {calls - removed}/{calls} "
                f"NVRTC option tuples that carried no -ftz=true to strip: the "
                f"policy seam had moved when the record was cut"]
    return []


# ---------------------------------------------------------------------------
# Exact float32 fma — integer arithmetic, no double rounding
# ---------------------------------------------------------------------------
#
# ``math.fma`` (3.13+) is a float64 fma: rounding its result to float32 double-
# rounds, exactly the hazard the FMA_V1 candidate must not carry. The emulation
# below is exact by construction: the product of two float32s and the float32
# addend are exact rationals, their sum is an exact rational, and the single
# rounding to float32 (round-to-nearest-even, subnormals included, signed zeros
# per IEEE 754 addition rules) happens once, in integer arithmetic.

# WHY THE EMULATION IS WORD-BASED, not float-based. Every FPU-visible step below
# used to be a host float operation (``np.float32(x)`` on a python float,
# ``np.ldexp`` on the rounded significand). Both are CONVERSIONS, and a host
# running the shipped flush policy answers them with zero for anything in the
# subnormal range: the "exact IEEE" emulation silently became policy-dependent,
# and ``_self_check_fma32``'s "-0 on underflow" identity became unsatisfiable —
# measured, as the AssertionError that stopped the shipped probe from running in
# a flushing process at all (results/complex_expansion_flush_coincidence_
# 2026-08-15/device_flush.json.shipped_gate_in_this_process). Words and integer
# arithmetic have no policy, so fma32/mul32 below mean the same thing under both.

_F32_SIGN = 0x80000000
_F32_MIN_NORMAL_WORD = 0x00800000
_F32_MIN_NORMAL = 2.0 ** -126  # exact as a python float; NOT a float32 conversion


def _float32_from_word(word: int) -> np.float32:
    """The float32 with this bit pattern. A reinterpretation, never a conversion."""
    return np.array([word], dtype=np.uint32).view(np.float32)[0]


def _word_of_float32(value: np.float32) -> int:
    """The bit pattern of a float32 SCALAR, without touching the FPU."""
    return int(np.asarray(value, dtype=np.float32).view(np.uint32))


def _binade_exponent(magnitude: Fraction) -> int:
    """The unique integer ``e`` with ``2**e <= magnitude < 2**(e + 1)``.

    Exact integer arithmetic on numerator and denominator, so it is right for
    any positive rational and never touches the FPU. ``bit_length`` overshoots
    by one whenever the quotient has not yet reached ``2**e``, which the
    comparison corrects.

    Shared by the two roundings below — the float32 one, whose exponent range is
    BOUNDED at the subnormal grid, and the unbounded-exponent one the underflow
    condition is tested on. Duplicating it would be duplicating the one step of
    either that is easy to get subtly wrong.
    """
    p, q = magnitude.numerator, magnitude.denominator
    exponent = p.bit_length() - q.bit_length()
    if (p << max(0, -exponent)) < (q << max(0, exponent)):
        exponent -= 1
    return exponent


def _round_24_significant_bits(magnitude: Fraction) -> Fraction:
    """RN(magnitude) to 24 SIGNIFICANT BITS WITH UNBOUNDED EXPONENT, ties to even.

    ``magnitude`` is a positive exact rational; the return value is exact.

    THIS IS NOT :func:`_round_float32_word` WITH THE ENCODING STRIPPED, and the
    difference is the whole of the tininess question. That one floors its grid
    at 2**-149 because float32 has a smallest exponent; this one has no floor at
    all, so inside the binade [2**-127, 2**-126) it rounds on a grid of 2**-150
    — half the spacing — and a value a quarter of a subnormal ulp below 2**-126
    can therefore round UP onto 2**-126 here while the float32 rounding of the
    same value lands on the same word for a different reason.

    It is the "result computed as though the exponent range were unbounded"
    that IEEE 754 and the Intel SDM both phrase the underflow condition in terms
    of. See :func:`_ftz_word_from_exact`.
    """
    exponent = _binade_exponent(magnitude)
    shift = 23 - exponent
    p, q = magnitude.numerator, magnitude.denominator
    if shift >= 0:
        numerator, denominator = p << shift, q
    else:
        numerator, denominator = p, q << (-shift)
    quotient, remainder = divmod(numerator, denominator)
    twice = 2 * remainder
    if twice > denominator or (twice == denominator and (quotient & 1)):
        quotient += 1
    # A carry to 2**24 needs no renormalization: the VALUE is what is wanted,
    # not an encoding, and quotient * 2**-shift is that value either way.
    return (Fraction(quotient, 1 << shift) if shift >= 0
            else Fraction(quotient << (-shift), 1))


def _round_float32_word(value: Fraction, sign_when_zero: float) -> int:
    """round_f32(exact rational) as a WORD — round-to-nearest-even, sign kept.

    Subnormals and the round-up into the smallest normal fall out of the integer
    encoding: at the subnormal scale the quotient IS the word, and a quotient
    that carries to 2**23 is exactly ``_F32_MIN_NORMAL_WORD``.

    The exponent range is BOUNDED here — that is what makes this the delivered
    float32 result rather than the value the underflow condition is tested on.
    """
    if value == 0:
        return _F32_SIGN if math.copysign(1.0, sign_when_zero) < 0 else 0
    negative = value < 0
    magnitude = -value if negative else value
    p, q = magnitude.numerator, magnitude.denominator
    exponent = _binade_exponent(magnitude)
    shift = 23 - exponent
    if exponent < -126:
        shift = 149  # subnormal scale
    if shift >= 0:
        numerator, denominator = p << shift, q
    else:
        numerator, denominator = p, q << (-shift)
    quotient, remainder = divmod(numerator, denominator)
    twice = 2 * remainder
    if twice > denominator or (twice == denominator and (quotient & 1)):
        quotient += 1
    if exponent < -126:
        word = quotient  # subnormal encoding; 2**23 is the smallest normal
    else:
        if quotient == (1 << 24):  # rounding carried into the next binade
            quotient = 1 << 23
            exponent += 1
        biased = exponent + 127
        if biased >= 0xFF:
            raise OverflowError(f"probe vector overflowed float32: {value!r}")
        word = (biased << 23) | (quotient - (1 << 23))
    return word | (_F32_SIGN if negative else 0)


def _round_float32(value: Fraction, sign_when_zero: float) -> np.float32:
    return _float32_from_word(_round_float32_word(value, sign_when_zero))


def _float32_word(value) -> int:
    """The float32 word of any finite scalar, round-to-nearest-even, FPU-free.

    A float32 input is REINTERPRETED (a conversion would flush it on a flushing
    host); anything wider is rounded in integer arithmetic.
    """
    if isinstance(value, np.float32):
        return _word_of_float32(value)
    wide = float(value)  # python/float64: exact, no float32 rounding happens here
    if wide == 0.0:
        return _F32_SIGN if math.copysign(1.0, wide) < 0 else 0
    return _round_float32_word(Fraction(wide), 1.0 if wide > 0 else -1.0)


def _fraction_of_word(word: int) -> Fraction:
    """The exact value of a float32 word. Raises on inf/NaN — probe vectors are
    finite by construction and a silent inf would poison every comparison."""
    biased = (word >> 23) & 0xFF
    mantissa = word & 0x7FFFFF
    if biased == 0xFF:
        raise ValueError(f"non-finite float32 word 0x{word:08x} in a probe vector")
    if biased == 0:
        magnitude = Fraction(mantissa, 1 << 149)
    else:
        significand = (1 << 23) | mantissa
        exponent = biased - 127 - 23
        magnitude = (Fraction(significand << exponent) if exponent >= 0
                     else Fraction(significand, 1 << -exponent))
    return -magnitude if word & _F32_SIGN else magnitude


def fma32(a, b, c) -> np.float32:
    """round_f32(exact a*b + c) for finite float32 operands, IEEE-correct.

    Independent of the host's subnormal policy: the operands are decoded from
    their words and the sum is rounded once, in integer arithmetic.
    """
    wa, wb, wc = _float32_word(a), _float32_word(b), _float32_word(c)
    fa, fb, fc = _fraction_of_word(wa), _fraction_of_word(wb), _fraction_of_word(wc)
    product_negative = bool((wa ^ wb) & _F32_SIGN)
    if fa == 0 or fb == 0:
        if fc == 0:
            # IEEE addition of zeros, round-to-nearest: like signs keep the
            # sign, unlike signs give +0.
            if product_negative and (wc & _F32_SIGN):
                return _float32_from_word(_F32_SIGN)
            return _float32_from_word(0)
        return _float32_from_word(wc)
    exact = fa * fb + fc
    if exact == 0:
        return _float32_from_word(0)  # exact cancellation of nonzeros -> +0 (RNE)
    return _round_float32(exact, 1.0 if exact > 0 else -1.0)


def mul32(a, b) -> np.float32:
    """round_f32(exact a*b), IEEE-correct and policy-independent (see fma32)."""
    wa, wb = _float32_word(a), _float32_word(b)
    fa, fb = _fraction_of_word(wa), _fraction_of_word(wb)
    negative = bool((wa ^ wb) & _F32_SIGN)
    if fa == 0 or fb == 0:
        return _float32_from_word(_F32_SIGN if negative else 0)
    return _round_float32(fa * fb, -1.0 if negative else 1.0)


# ---------------------------------------------------------------------------
# The candidate arms are computed UNDER THE POLICY IN FORCE
# ---------------------------------------------------------------------------
#
# THE DEFECT THIS CLOSES. The probe scores the DEVICE's bytes against candidate
# arms computed on the HOST. When the device flushes float32 subnormals and the
# candidates keep them, the comparison is not a classification: the 48 kept-
# subnormal-product lanes and 6 zero-sign-flip lanes that separate FMA_V1 from
# NAIVE on the mixed-dtype patterns are exactly the lanes flush destroys, so the
# platform matches NEITHER arm (54/48/125 words of 2280, measured 2026-08-11 and
# reproduced 2026-08-15) and the gate refused a platform it had classified
# perfectly well under the other policy. Candidates are therefore cut under the
# policy in force, and the probe artifact says which.
#
# THE FTZ SEMANTICS emulated here are the flushing hardware's, and WHAT THE
# TININESS TEST IS APPLIED TO is the whole of the difference. There are THREE
# candidate rules, not two, and the middle one is the measured one:
#
#   (a) THE EXACT VALUE, before any rounding: nonzero and |exact| < 2**-126.
#   (b) THE VALUE ROUNDED TO 24 SIGNIFICANT BITS WITH UNBOUNDED EXPONENT:
#       nonzero and below 2**-126 after that rounding. This is IEEE 754's
#       "after rounding" option and it is THE RULE THIS FILE IMPLEMENTS.
#   (c) THE DELIVERED float32 WORD still being subnormal. Not an IEEE option at
#       all — it is (b) with the rounding floored at the 2**-149 subnormal grid,
#       which is exactly the floor that destroys the information.
#
# IEEE 754-2019 §7.5 offers (a) and (b) and requires only that one be used
# consistently. (a) and (b) part company on the class where the exact value is
# below 2**-126 but its 24-bit unbounded-exponent rounding ties UP onto 2**-126:
# |exact| in [2**-126 - 2**-151, 2**-126). (b) and (c) part company on the class
# where the exact value is below 2**-126 by more than that but the float32
# rounding, on its coarser subnormal grid, still lands on 2**-126: |exact| in
# [2**-126 - 2**-150, 2**-126 - 2**-151). Neither class is hypothetical here.
#
# WHAT THE DOCUMENTS SAY — AND THE MANUAL AND THE SILICON AGREE.
#   * x86 SSE. The SDM defines FTZ as the masked response to the underflow
#     condition (Vol. 1 §10.2.3.3, "Flush-To-Zero") and defines that condition
#     on the result of rounding with an unbounded exponent (Vol. 1 §4.9.1.5,
#     "Numeric Underflow Exception (#U)": "the result of rounding with unbounded
#     exponent ... is non-zero and tiny"). That is rule (b), stated plainly, and
#     it is what the hardware was measured to do.
#   * PTX. The ISA says .ftz "flushes subnormal inputs and results to
#     sign-preserving zero" (§9.7.3, and per-instruction for mul/fma) and never
#     says what "results" is tested on. IT SETTLES NOTHING, in either direction:
#     it is consistent with (a), (b) and (c) alike, and must not be quoted for
#     or against any of them.
#   * THE MEASUREMENT, which is what actually decides. 37,439 vectors whose
#     classes were derived by factoring exact rationals — constructed
#     independently of this gate and of its tests — run on two executors: x86
#     mulss / vfmadd213ss with MXCSR set once per pass over all four FTZ|DAZ
#     combinations, and PTX mul.rn.ftz.f32 / fma.rn.ftz.f32 through inline asm
#     in a CuPy RawKernel on an RTX A6000. The two executors agree with each
#     other on 37439 of 37439. Scored against the three rules: (a) differs on
#     10, (b) differs on 0, (c) differs on 18. All ten of (a)'s disagreements
#     are (a) returning a sign-preserving zero where the hardware returns
#     0x00800000. (The claim is about the VALUE the mode returns; the #U flag
#     was not read per row.)
#   * THE SEPARATING EXAMPLE, which any future edit here must reproduce:
#     0x24042108 * 0x1BF80000, both operands NORMAL so DAZ cannot reach it.
#     The exact product is (2**25 - 1) * 2**-151 = 2**-126 - 2**-151. Its
#     magnitude is below 2**-126, so (a) calls it tiny and returns zero; its
#     24-bit unbounded-exponent rounding is a half-even tie that lands ON
#     2**-126, so (b) calls it normal and delivers 0x00800000. Both executors
#     return 0x00800000.
#   * AND THE TIE, which is what rules (c) out: 0x00FFFFFF * 0.5 is exactly
#     2**-126 - 2**-150. Its unbounded-exponent rounding is itself, below
#     2**-126, so (b) flushes it; the delivered float32 word is the normal
#     0x00800000, so (c) keeps it. The hardware returns 0x00000000, and
#     0x80FFFFFF * 0.5 returns 0x80000000 (four executors, 2026-08-15; see
#     results/complex_expansion_policy_2026-08-15/results/tininess_before_rounding.json
#     for the same tie measured through this repository's own policy install).
#     A second mechanism reaches that tie with three NORMAL operands and no
#     subnormal product to blame: fma(2**-75, -2**-75, 2**-126) sums exactly to
#     2**-126 - 2**-150, and fma.rn.ftz.f32 returns zero where fma.rn.f32
#     returns 0x00800000 — so the test is applied to the exact SUM, and the
#     intermediate product is not flushed on its own.
#
# WHAT THIS FILE GOT WRONG, TWICE, AND WHY EACH HID.
#   Until 2026-08-15 it implemented (c): a result was flushed only if the
#   DELIVERED float32 word was still subnormal, so on the tie class the
#   candidates kept a smallest-normal word the device had zeroed. Under the
#   flush policy that is 128 of FMA_V1's words on ``python_float_left`` at
#   scalar 0.5, every one of them in the ``_NORMAL_UNDERFLOWING_ZR`` block and
#   every one carrying zr in {0x00FFFFFF, 0x80FFFFFF}; the pattern classified
#   NEITHER, the four scalars disagreed, and the gate refused a platform for the
#   arbiter's defect. It hid because 0.5 is the only coefficient in the probe's
#   set that produces the tie, and because the tie needs an operand at the TOP
#   of a binade whose halving lands half an ulp below 2**-126 — a class the
#   vector set did not carry at all before ``_NORMAL_UNDERFLOWING_ZR``.
#
#   The correction installed on 2026-08-15 overshot to (a), and its own
#   discriminating value could not have caught the overshoot: 2**-126 - 2**-150
#   is (2 - 2**-23) x 2**-127, EXACTLY 24 significant bits, so no rounding
#   occurs and it is tiny under both (a) and (b). "Rounds onto the smallest
#   normal" was true of the SUBNORMAL grid, not of IEEE's unbounded-exponent
#   rounding, and the hardware zeroing it is consistent with either. The control
#   sweep that was meant to bound the change never reached the separating class
#   either — its scalars are powers of two plus 0.75 and 1.5, and a product of
#   those with a float32 cannot land in a window a quarter of a subnormal ulp
#   wide. (a) also contradicted the SDM, and the file said so, asserting that
#   the manual "reads as the other one, so this is transcribed from the silicon,
#   not from the manual". That assertion was false: the manual says (b), the
#   silicon does (b), and only this file dissented.
#
#   WHAT THE OVERSHOOT COST IN CANDIDATE WORDS: nothing, measured. Restoring (b)
#   moves 0 of 75,308 candidate words across every pattern and scalar this gate
#   cuts under flush, and 0 of the probe's own invocations land in the
#   separating class. So the 2026-08-16 FMA_V1 flush licence stands unchanged.
#   That is a fact about how narrow the window is — relative measure ~2**-25 —
#   and NOT a reason to leave a wrong rule in a certification path.
#
# AND THE SECOND DEFECT AT THE SAME SITE: the vectorized flush arms were
# ``flush_subnormals(host_float32_op(...))``, so the middle step borrowed
# whatever FPU the process was running on. On the flushing x86 host that was
# accidentally the hardware's own rule (the hardware did the flush) while the
# scalar arms implemented (c); on this arm64 laptop all three arms were (c).
# One "flush" request meant three different things. Every candidate operation
# below is now word-and-rational arithmetic with no FPU step at all, so it means
# the same thing on every host.
#
# ADD AND SUB CANNOT TELL THE THREE RULES APART, which bounds what this
# correction can move. Every finite float32 is an integer multiple of 2**-149,
# so the exact sum or difference of two of them is too; a result of magnitude
# below 2**-126 is then an integer multiple of 2**-149 with |k| < 2**23, i.e.
# exactly representable as a subnormal, and no rounding occurs at all — so all
# three rules read the same value and give the same verdict. Only multiply, fma,
# divide and sqrt — whose exact result can be finer than 2**-149 — can
# discriminate, and the candidate arms use the first two.

FTZ_CANDIDATE_SEMANTICS = (
    "hardware .ftz: subnormal float32 INPUTS flushed to sign-preserving zero "
    "(the DAZ half), then the exact result formed in unbounded precision and "
    "ROUNDED TO 24 SIGNIFICANT BITS WITH UNBOUNDED EXPONENT, round-to-nearest-"
    "even; if THAT value is nonzero and below 2**-126 the result is a "
    "sign-preserving zero, otherwise the ordinary float32 rounding is delivered "
    "— IEEE 754-2019 §7.5's AFTER-rounding tininess, and note that the "
    "unbounded exponent is load-bearing: this is not a rounding onto the "
    "subnormal grid, and it is not a test on the delivered word. The manual and "
    "the silicon agree on this rule. Intel SDM Vol. 1 §10.2.3.3 makes FTZ the "
    "masked response to the underflow condition and §4.9.1.5 defines that "
    "condition on 'the result of rounding with unbounded exponent ... non-zero "
    "and tiny'; measured, 37,439 independently constructed vectors on two "
    "executors (x86 mulss/vfmadd213ss under MXCSR, and PTX mul.rn.ftz.f32/"
    "fma.rn.ftz.f32 on an RTX A6000) agree with each other 37439/37439 and with "
    "this rule on all 37439, against 10 disagreements for a test on the exact "
    "value and 18 for a test on the delivered word. The PTX ISA is SILENT on "
    "what the test is applied to and settles nothing either way")

IEEE_CANDIDATE_SEMANTICS = (
    "IEEE-754 float32 with subnormals kept: exact integer-arithmetic fma32/mul32 "
    "for the fused arm, host float32 arithmetic for the vectorized arms")


def flush_subnormals(x) -> np.ndarray:
    """Sign-preserving flush of every subnormal float32 word in ``x`` (PTX .ftz)."""
    words = np.ascontiguousarray(np.asarray(x, dtype=np.float32)).view(np.uint32).copy()
    subnormal = (((words >> 23) & 0xFF) == 0) & ((words & 0x7FFFFF) != 0)
    words[subnormal] &= np.uint32(_F32_SIGN)
    return words.view(np.float32)


def subnormal_words(x) -> int:
    """How many subnormal float32 words ``x`` carries — the census that catches a
    process which destroyed the probe's discriminating operands."""
    words = np.ascontiguousarray(np.asarray(x, dtype=np.float32)).view(np.uint32)
    return int(np.count_nonzero((((words >> 23) & 0xFF) == 0)
                                & ((words & 0x7FFFFF) != 0)))


# ---------------------------------------------------------------------------
# THE FLUSH RULE, STATED ONCE
# ---------------------------------------------------------------------------

#: 2**-126 as an EXACT rational — the tininess threshold. A float comparison
#: would be exact too (``Fraction`` compares against a float by converting it
#: exactly), but the threshold is the one number in this file that may not be
#: read as approximate, so it is written as what it is.
_F32_MIN_NORMAL_EXACT = Fraction(1, 1 << 126)


def _ftz_operand_word(value) -> int:
    """The operand half of ``.ftz`` (equivalently DAZ): this value's float32
    word, with a subnormal replaced by a zero of the same sign.

    Word arithmetic, never a conversion — a flushing host would answer a
    float64 -> float32 conversion of a subnormal with zero on its own, and then
    the emulation would be reporting its host rather than the policy it claims.
    """
    word = _float32_word(value)
    if ((word >> 23) & 0xFF) == 0 and (word & 0x7FFFFF) != 0:
        return word & _F32_SIGN
    return word


def _ftz_word_from_exact(exact: Fraction, negative_zero: bool) -> int:
    """THE RULE. round_ftz(exact) as a float32 word — the ONE place the flush
    rule is stated, reused by every candidate operation below.

    ``exact`` is the infinitely precise result of the operation, as an exact
    rational, formed AFTER the operands have been through
    :func:`_ftz_operand_word`. ``negative_zero`` is the sign IEEE gives a result
    that is exactly zero (the operation's own rule: like-signed zeros keep the
    sign, everything else including exact cancellation gives +0).

    Four clauses, in this order:

    1. An exactly zero result is not "tiny"; it keeps its IEEE sign.
    2. Round the exact result to 24 SIGNIFICANT BITS WITH UNBOUNDED EXPONENT,
       round-to-nearest-even (:func:`_round_24_significant_bits`). This is the
       underflow condition's own subject, and the unbounded exponent is the
       part that is easy to lose: it is NOT a rounding onto the subnormal grid.
    3. If that rounded value is nonzero and its magnitude is STRICTLY below
       2**-126, return a zero of the exact result's sign. Strictly: a rounded
       value OF 2**-126 is the smallest normal, so the boundary belongs to the
       normal side.
    4. Otherwise deliver the ordinary float32 rounding of the exact result.

    WHERE CLAUSE 2 EARNS ITS KEEP — the class that separates this rule from a
    test on the exact value. Take the exact product 2**-126 - 2**-151, a quarter
    of a subnormal ulp below the smallest normal (reachable with two NORMAL
    operands, so DAZ never sees it: 0x24042108 * 0x1BF80000). Its magnitude is
    below 2**-126, so a test on the EXACT value calls it tiny and flushes. But
    24-bit unbounded-exponent rounding works on a 2**-150 grid there, the value
    sits exactly halfway between (2**24 - 1) * 2**-150 and 2**24 * 2**-150, and
    round-half-to-even takes the even neighbour — which is 2**-126 itself. Not
    tiny, so the result is delivered: 0x00800000, which is what the hardware
    returns on both executors it was measured on. Testing the exact value
    answers 0x00000000 and is wrong.

    AND WHERE CLAUSE 2 IS NOT MERELY THE DELIVERED WORD. Take 2**-126 - 2**-150
    instead (0x00FFFFFF * 0.5), half a subnormal ulp below. Clause 2's grid is
    2**-150 and the value is exactly on it, so it rounds to itself, below
    2**-126: tiny, flushed to zero — and again that is what the hardware
    returns. Inspecting the DELIVERED word cannot see this: the float32 rounding
    is floored at the 2**-149 subnormal grid, the value is a tie there, and
    half-to-even carries it onto the normal 0x00800000, indistinguishable from a
    word that never underflowed. That delivered-word rule is what this file
    carried until 2026-08-15, and it is why no test on the delivered word
    survives anywhere here.

    The delivered result of clause 4 can never be subnormal: whenever clause 2
    says "not tiny" the exact magnitude is at least 2**-126 - 2**-151, and the
    nearest multiple of 2**-149 to that is 2**-126.
    """
    if exact == 0:
        return _F32_SIGN if negative_zero else 0
    negative = exact < 0
    magnitude = -exact if negative else exact
    # Clause 3's "nonzero" is discharged by clause 1 and needs no test of its
    # own: with an unbounded exponent there is nothing for a nonzero value to
    # underflow to, so _round_24_significant_bits of a positive rational is
    # positive. Clause 1 is the only route to a zero result.
    if _round_24_significant_bits(magnitude) < _F32_MIN_NORMAL_EXACT:
        return _F32_SIGN if negative else 0
    return _round_float32_word(exact, -1.0 if negative else 1.0)


def _ftz_mul_word(a, b) -> int:
    """round_ftz(exact a*b) as a word. Operands flushed first, product formed
    exactly, :func:`_ftz_word_from_exact` decides."""
    wa, wb = _ftz_operand_word(a), _ftz_operand_word(b)
    negative = bool((wa ^ wb) & _F32_SIGN)
    return _ftz_word_from_exact(_fraction_of_word(wa) * _fraction_of_word(wb),
                                negative)


def _ftz_fma_word(a, b, c) -> int:
    """round_ftz(exact a*b + c) as a word — ONE rounding, and the tininess test
    applied to the exact SUM.

    The intermediate product is NOT flushed on its own: measured, an fma whose
    exact product is far below the subnormal band but whose exact sum is a
    normal number returns that normal, and one whose exact sum is half an ulp
    below 2**-126 returns zero. Both fall out of testing the sum.
    """
    wa, wb, wc = _ftz_operand_word(a), _ftz_operand_word(b), _ftz_operand_word(c)
    fa, fb, fc = (_fraction_of_word(wa), _fraction_of_word(wb),
                  _fraction_of_word(wc))
    # IEEE zero sign: a zero product plus a zero addend keeps the sign only when
    # both are negative; anything else that lands on zero (exact cancellation of
    # nonzeros included) is +0 under round-to-nearest.
    negative_zero = (fa == 0 or fb == 0) and fc == 0 and \
        bool((wa ^ wb) & _F32_SIGN) and bool(wc & _F32_SIGN)
    return _ftz_word_from_exact(fa * fb + fc, negative_zero)


def _ftz_add_word(a, b) -> int:
    """round_ftz(exact a + b) as a word.

    Routed through the same rule as the multiply even though addition provably
    cannot tell the three tininess rules apart (see the derivation above):
    the point is that the flush candidate set contains no host FPU operation at
    all, so it means one thing on every host rather than borrowing the process's
    own convention.
    """
    wa, wb = _ftz_operand_word(a), _ftz_operand_word(b)
    fa, fb = _fraction_of_word(wa), _fraction_of_word(wb)
    negative_zero = fa == 0 and fb == 0 and bool(wa & wb & _F32_SIGN)
    return _ftz_word_from_exact(fa + fb, negative_zero)


def _ftz_sub_word(a, b) -> int:
    """round_ftz(exact a - b) as a word — ``a + (-b)``, negation being a sign
    flip so a subnormal operand is not routed through the FPU to be negated."""
    return _ftz_add_word(a, _float32_from_word(_float32_word(b) ^ _F32_SIGN))


def _ftz_elementwise(rule, *operands) -> np.ndarray:
    """Apply a word rule to broadcast float32 operands, producing WORDS.

    The vectorized flush arms used to be ``flush_subnormals(host_op(...))``,
    which put a host float32 operation between two word flushes and therefore
    inherited the host's own tininess convention. They are the same rule as the
    scalar arms now, applied per element, so a "flush" candidate is one thing.
    """
    arrays = np.broadcast_arrays(*[np.asarray(operand, dtype=np.float32)
                                   for operand in operands])
    flat = [np.ascontiguousarray(array).ravel() for array in arrays]
    words = np.empty(flat[0].size, dtype=np.uint32)
    for index in range(words.size):
        words[index] = np.uint32(rule(*(column[index] for column in flat)))
    return words.view(np.float32).reshape(arrays[0].shape)


def _ftz_mul(a, b) -> np.ndarray:
    return _ftz_elementwise(_ftz_mul_word, a, b)


def _ftz_add(a, b) -> np.ndarray:
    return _ftz_elementwise(_ftz_add_word, a, b)


def _ftz_sub(a, b) -> np.ndarray:
    return _ftz_elementwise(_ftz_sub_word, a, b)


def _plain_mul(a, b) -> np.ndarray:
    return np.asarray(a, np.float32) * np.asarray(b, np.float32)


def _plain_add(a, b) -> np.ndarray:
    return np.asarray(a, np.float32) + np.asarray(b, np.float32)


def _plain_sub(a, b) -> np.ndarray:
    return np.asarray(a, np.float32) - np.asarray(b, np.float32)


def _ftz_fma_scalar(a, b, c) -> np.float32:
    return _float32_from_word(_ftz_fma_word(a, b, c))


def _ftz_mul_scalar(a, b) -> np.float32:
    return _float32_from_word(_ftz_mul_word(a, b))


def candidate_arithmetic(flush: Optional[bool] = None) -> Dict[str, Any]:
    """The five operations the candidate arms are built from, under one policy.

    ``flush=None`` means THE POLICY IN FORCE, which is what every call site
    wants: the candidates must be cut under the same policy as the bytes they
    will be scored against.
    """
    if flush is None:
        flush = resolved_policy() == _policy.FLUSH
    if flush:
        return {"flush": True, "policy": _policy.FLUSH,
                "semantics": FTZ_CANDIDATE_SEMANTICS,
                "mul": _ftz_mul, "add": _ftz_add, "sub": _ftz_sub,
                "fma_scalar": _ftz_fma_scalar, "mul_scalar": _ftz_mul_scalar}
    return {"flush": False, "policy": _policy.KEEP,
            "semantics": IEEE_CANDIDATE_SEMANTICS,
            "mul": _plain_mul, "add": _plain_add, "sub": _plain_sub,
            "fma_scalar": fma32, "mul_scalar": mul32}


def _self_check_fma32() -> None:
    """Pins the emulation against identities computable without an fma.

    RUNS UNDER BOTH SUBNORMAL POLICIES. It did not: ``tiny = np.float32(1e-45)``
    is a conversion, a flushing host answers it with +0.0, and the "-0 on
    underflow" identity below then fails — an AssertionError at this function
    that stopped the shipped probe dead in the ship configuration (measured,
    results/complex_expansion_flush_coincidence_2026-08-15/device_flush.json,
    key ``shipped_gate_in_this_process``). The subnormal is built from its WORD
    instead, and the identity is a fact about fma32, which is now integer-exact
    and therefore policy-independent — so the assertion holds either way rather
    than being weakened.
    """
    rng = np.random.default_rng(SEED)
    xs = rng.uniform(-3.0, 3.0, size=64).astype(np.float32)
    ys = rng.uniform(-3.0, 3.0, size=64).astype(np.float32)
    for x, y in zip(xs, ys):
        assert fma32(x, y, 0.0).tobytes() == mul32(x, y).tobytes() or (x * y) == 0.0
        assert fma32(1.0, x, y).tobytes() == np.float32(np.float32(x) + np.float32(y)).tobytes()
    assert fma32(-0.0, 2.0, -0.0).tobytes() == np.float32(-0.0).tobytes()
    assert fma32(0.0, 2.0, -0.0).tobytes() == np.float32(0.0).tobytes()
    tiny = _float32_from_word(0x00000001)  # 2**-149, built without the FPU
    assert _word_of_float32(tiny) == 0x00000001, (
        "a float32 subnormal did not survive being read from its own word: the "
        "probe cannot be run in this process at all")
    assert float(fma32(-tiny, 0.25, 0.0)) == 0.0
    assert math.copysign(1.0, float(fma32(-tiny, 0.25, 0.0))) < 0  # -0 on underflow
    # And the FTZ emulation the flush candidates are built from: subnormal in,
    # sign-preserving zero out; normal words untouched.
    flushed = flush_subnormals(_float32_from_word(0x80000001))
    assert int(np.asarray(flushed).view(np.uint32)[0]) == 0x80000000
    assert _ftz_operand_word(_float32_from_word(0x80000001)) == 0x80000000
    assert _ftz_operand_word(np.float32(1.5)) == _word_of_float32(np.float32(1.5))
    assert subnormal_words(np.array([tiny, np.float32(1.0)], dtype=np.float32)) == 1
    # THE TININESS RULE, pinned at BOTH boundaries — the one that rules out a
    # test on the delivered word, and the one that rules out a test on the exact
    # value. Either alone leaves a wrong rule passing.
    #
    # (i) 0x00FFFFFF * 0.5 is exactly 2**-126 - 2**-150. Rounded to 24
    # significant bits with an unbounded exponent it is ITSELF (that value has
    # exactly 24 significant bits), which is below 2**-126, so it flushes — and
    # the flushing hardware returns zero for it (four executors, 2026-08-15).
    # The DELIVERED float32 word is the normal 0x00800000, because the float32
    # rounding is floored at the 2**-149 grid and half-to-even carries the tie
    # up; an emulation that tested that word answers 0x00800000 and refuses the
    # platform.
    top = _float32_from_word(0x00FFFFFF)
    assert _ftz_mul_word(top, 0.5) == 0x00000000
    assert _ftz_mul_word(_float32_from_word(0x80FFFFFF), 0.5) == 0x80000000
    assert _ftz_fma_word(top, 0.5, 0.0) == 0x00000000
    assert mul32(top, 0.5).tobytes() == _float32_from_word(0x00800000).tobytes(), (
        "the IEEE-keep arm must still round the tie UP onto the smallest "
        "normal, or the two policies have stopped differing where they must")
    # (ii) THE SEPARATING EXAMPLE. 0x24042108 * 0x1BF80000 is exactly
    # (2**25 - 1) * 2**-151 = 2**-126 - 2**-151, and BOTH operands are normal so
    # DAZ cannot reach the row. |exact| < 2**-126, so a test on the exact value
    # flushes it to zero; its 24-bit unbounded-exponent rounding is a half-even
    # tie that lands ON 2**-126, so the correct rule delivers it. Hardware:
    # 0x00800000 on x86 and on PTX. This is the assertion that a re-installed
    # before-rounding test fails.
    assert _ftz_mul_word(_float32_from_word(0x24042108),
                         _float32_from_word(0x1BF80000)) == 0x00800000
    assert _ftz_mul_word(_float32_from_word(0xA4042108),
                         _float32_from_word(0x1BF80000)) == 0x80800000
    assert _ftz_fma_word(_float32_from_word(0x24042108),
                         _float32_from_word(0x1BF80000), 0.0) == 0x00800000
    # ... and the rounding it turns on is the UNBOUNDED-exponent one, not the
    # float32 one: on this value the two agree by coincidence, so the rule is
    # pinned on a value where they do not (2**-126 - 2**-150 above).
    assert _round_24_significant_bits(
        _fraction_of_word(0x24042108) * _fraction_of_word(0x1BF80000)
    ) == _F32_MIN_NORMAL_EXACT
    assert _round_24_significant_bits(
        _F32_MIN_NORMAL_EXACT - Fraction(1, 1 << 150)
    ) == _F32_MIN_NORMAL_EXACT - Fraction(1, 1 << 150)
    # The boundary itself belongs to the NORMAL side: an exact product OF
    # 2**-126 is the smallest normal, not a tiny result.
    assert _ftz_mul_word(_float32_from_word(0x01000000), 0.5) == 0x00800000
    # ... and the tie is reached by an ADDITION too, with three normal operands
    # and an intermediate product far below the band: the rule is applied to the
    # exact SUM, and the intermediate is not flushed on its own.
    assert _ftz_fma_word(_float32_from_word(0x1A000000),
                         _float32_from_word(0x9A000000),
                         _float32_from_word(0x00800000)) == 0x00000000
    assert _ftz_fma_word(_float32_from_word(0x1A000000),
                         _float32_from_word(0x1A000000),
                         _float32_from_word(0x00800000)) == 0x00800000
    # The vectorized arms are the SAME rule, not a second transcription of it.
    probe_words = (0x00FFFFFF, 0x80FFFFFF, 0x01000000, 0x3F800000, 0x00800000)
    left = np.array([_float32_from_word(word) for word in probe_words],
                    dtype=np.float32)
    right = np.full(left.shape, np.float32(0.5))
    assert _ftz_mul(left, right).tobytes() == np.array(
        [_float32_from_word(_ftz_mul_word(a, b))
         for a, b in zip(left, right)], dtype=np.float32).tobytes()


# ---------------------------------------------------------------------------
# The expansion preflight probe
# ---------------------------------------------------------------------------

RANDOM_VECTORS = 2048
_ZEROS = (0.0, -0.0)
_SMALL = (1e-45, -1e-45, 3e-44, -3e-44, 1e-40, -1e-40)
_PLAIN = (1.0, -1.0, 0.75, -0.5, 1.5, -2.25)

#: ALL-NORMAL operands whose PRODUCT underflows — the class the probe was blind
#: to, and the only one that separates the arms on the mixed-dtype orientations
#: under the flush policy.
#:
#: MEASURED 2026-08-15, and this is why the class had to be added. Every other
#: discriminating row the probe carries feeds a SUBNORMAL OPERAND to the device
#: (:data:`_SMALL`, 1e-45…1e-40). Under the ship policy those are flushed on
#: first use, so on the shipped x86 platform the two mixed patterns came out
#: 0 of 4560 words apart — AMBIGUOUS_BOTH — and the licence for
#: ``_mul_field_left`` / ``_mul_coefficient_left``, i.e. EVERY real-coefficient
#: multiply in both kernels, was inherited from ``c8_mul_c8``, a different
#: compiled orientation, with zero evidence for the call sites themselves.
#:
#: These rows have no subnormal operand at all. ``zr`` is a min-normal-binade
#: float32 and ``c`` is a small power of two, so ``zr * c`` lands in or below the
#: subnormal band while both inputs stay normal and survive any flush. The arms
#: then split on the sign of the underflowed zero: FMA_V1 computes
#: ``fma(zr, c, +0.0)``, whose single rounding keeps the exact product's sign and
#: gives -0.0; NAIVE rounds ``zr*c`` to a (flushed) -0.0 and subtracts the -0.0
#: cross term, giving +0.0. Reproduced in hardware with no emulation:
#: zr = 0x00800000, c = 0xb3000000, zi = -1.0 gives NAIVE re = 0x00000000 against
#: FMA_V1 re = 0x80000000. Over a sampled binade the split is 100% of rows.
#:
#: It also discriminates under KEEP (measured 24 of 288 words on a 144-row
#: all-normal set), which retires the recorded premise that the subnormal-INPUT
#: rows are the only rows separating the arms under keep. They are not; they are
#: only the rows the probe happened to carry.
_NORMAL_UNDERFLOWING_ZR = (0x00800000, 0x80800000, 0x00800001, 0x80C00000,
                           0x00FFFFFF, 0x80FFFFFF, 0x01000000, 0x81000001)
_NORMAL_UNDERFLOWING_C = (0x33000000, 0xB3000000, 0x33800000, 0xB3800000,
                          0x34000000, 0xB4000000, 0x20000000, 0xA0000000)
_NORMAL_UNDERFLOWING_ZI = (1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.0, -0.0)


def _asarray_float32(values: Sequence[float]) -> np.ndarray:
    """float32 array whose WORDS do not depend on the host's subnormal policy.

    ``np.asarray([1e-45, ...], dtype=np.float32)`` is a float64 -> float32
    CONVERSION, and a host running the ship policy answers it with zero: every
    row of :data:`_SMALL` — the underflow rows that are half the probe's
    discriminating power — is destroyed before the device is asked anything.
    Measured: a first device attempt under flush lost 72 ``zr`` words this way
    (results/complex_expansion_flush_coincidence_2026-08-15/). Values below the
    float32 min normal are therefore encoded in integer arithmetic and written
    as words; the rest go through the ordinary conversion, which no policy
    touches.
    """
    values = [float(v) for v in values]
    out = np.ascontiguousarray(np.asarray(values, dtype=np.float32))
    words = out.view(np.uint32).copy()
    for index, value in enumerate(values):
        if value != 0.0 and abs(value) < _F32_MIN_NORMAL:
            words[index] = np.uint32(_round_float32_word(
                Fraction(value), 1.0 if value > 0 else -1.0))
    return words.view(np.float32)


def _negate32(value) -> np.float32:
    """IEEE negation as a sign-bit flip — ``-float(x)`` would widen a subnormal
    through the FPU, which a flushing host answers with zero."""
    return _float32_from_word(_float32_word(value) ^ _F32_SIGN)


def _zero_imag_operands(rng) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(z_re, z_im, c) rows: random + the signed-zero and underflow rows that
    discriminate the full zero-imag product from the plane-wise one, and its
    FMA_V1 arm from its NAIVE one.

    THE LAST BLOCK IS WHAT MAKES THIS PROBE DISCRIMINATE UNDER FLUSH. Every
    other discriminating row here hands the device a subnormal OPERAND, which
    the ship policy destroys on first use; those rows measured 0 of 4560 words
    apart under flush, and the mixed orientations therefore licensed nothing.
    :data:`_NORMAL_UNDERFLOWING_ZR` rows have no subnormal operand at all — only
    a subnormal PRODUCT — so they survive both policies. See that constant."""
    zr = list(rng.uniform(-2.0, 2.0, RANDOM_VECTORS))
    zi = list(rng.uniform(-2.0, 2.0, RANDOM_VECTORS))
    c = list(rng.uniform(-1.5, 1.5, RANDOM_VECTORS))
    for zero in _ZEROS:
        for other in _PLAIN + _ZEROS:
            for coefficient in (0.5, -0.5, 1.25, 0.0, -0.0):
                zr.append(zero)
                zi.append(other)
                c.append(coefficient)
                zr.append(other)
                zi.append(zero)
                c.append(coefficient)
    for small in _SMALL:  # products that underflow to zero under one rounding
        for other in (1.0, -1.0, 0.0, -0.0):
            for coefficient in (0.25, -0.25, 0.5):
                zr.append(small)
                zi.append(other)
                c.append(coefficient)
    out_zr = _asarray_float32(zr)
    out_zi = _asarray_float32(zi)
    out_c = _asarray_float32(c)
    # All-normal operands, underflowing product. Appended as WORDS, never as
    # Python floats: these are exact float32 bit patterns and must not be routed
    # back through a float64 -> float32 conversion.
    n_zr, n_zi, n_c = _normal_underflow_rows()
    return (np.concatenate([out_zr, n_zr]),
            np.concatenate([out_zi, n_zi]),
            np.concatenate([out_c, n_c]))


def _normal_underflow_rows() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(z_re, z_im, c) rows built from WORDS: all operands normal, product
    subnormal or below. See :data:`_NORMAL_UNDERFLOWING_ZR` for the measurement
    that put them here and the sign mechanism that makes them discriminate."""
    zr: List[np.float32] = []
    zi: List[np.float32] = []
    c: List[np.float32] = []
    for zr_word in _NORMAL_UNDERFLOWING_ZR:
        for c_word in _NORMAL_UNDERFLOWING_C:
            for imag in _NORMAL_UNDERFLOWING_ZI:
                zr.append(_float32_from_word(zr_word))
                c.append(_float32_from_word(c_word))
                zi.append(np.float32(imag))  # exact, never subnormal
    return (np.ascontiguousarray(np.array(zr, dtype=np.float32)),
            np.ascontiguousarray(np.array(zi, dtype=np.float32)),
            np.ascontiguousarray(np.array(c, dtype=np.float32)))


def _c8_operands(rng) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ar = list(rng.uniform(-2.0, 2.0, RANDOM_VECTORS))
    ai = list(rng.uniform(-2.0, 2.0, RANDOM_VECTORS))
    br = list(rng.uniform(-2.0, 2.0, RANDOM_VECTORS))
    bi = list(rng.uniform(-2.0, 2.0, RANDOM_VECTORS))
    for zero in _ZEROS:  # signed zeros through the cross terms
        for other in _PLAIN:
            ar.append(zero); ai.append(other); br.append(0.8); bi.append(0.6)
            ar.append(other); ai.append(zero); br.append(-0.8); bi.append(0.6)
    for scale in (1.0, 1.0 + 2 ** -20, 1.0 - 2 ** -21):  # near-cancellation
        ar.append(1.3); ai.append(1.1)
        br.append(1.1 * scale); bi.append(1.3)
    return (_asarray_float32(ar), _asarray_float32(ai),
            _asarray_float32(br), _asarray_float32(bi))


def _candidates_c8(ar, ai, br, bi, flush: Optional[bool] = None
                   ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Field-LEFT full complex multiply (the S:1862 phase-rotation orientation).

    ``flush=None`` builds the arms under the policy in force — see
    :func:`candidate_arithmetic` for why that is not optional."""
    ops = candidate_arithmetic(flush)
    mul, add, sub = ops["mul"], ops["add"], ops["sub"]
    fma, mul1 = ops["fma_scalar"], ops["mul_scalar"]
    naive_re = sub(mul(ar, br), mul(ai, bi))
    naive_im = add(mul(ar, bi), mul(ai, br))
    n = ar.size
    v1_re = np.empty(n, np.float32)
    v1_im = np.empty(n, np.float32)
    v2_re = np.empty(n, np.float32)
    v2_im = np.empty(n, np.float32)
    for i in range(n):
        v1_re[i] = fma(ar[i], br[i], _negate32(mul1(ai[i], bi[i])))
        v1_im[i] = fma(ar[i], bi[i], mul1(ai[i], br[i]))
        # Diagnostic only: the OTHER product fused. Never licenses a constexpr.
        v2_re[i] = fma(_negate32(ai[i]), bi[i], mul1(ar[i], br[i]))
        v2_im[i] = fma(ai[i], br[i], mul1(ar[i], bi[i]))
    return {"NAIVE": (naive_re.astype(np.float32), naive_im.astype(np.float32)),
            "FMA_V1": (v1_re, v1_im),
            "FMA_V2_diagnostic": (v2_re, v2_im)}


def _candidates_zero_imag(zr, zi, c, field_left: bool, flush: Optional[bool] = None
                          ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """The zero-imaginary product in the kernel's two arms plus the plane-wise
    diagnostic. ``field_left`` selects _mul_field_left's cross-term signs,
    otherwise _mul_coefficient_left's; ``flush=None`` is the policy in force."""
    ops = candidate_arithmetic(flush)
    mul, add, sub = ops["mul"], ops["add"], ops["sub"]
    fma, mul1 = ops["fma_scalar"], ops["mul_scalar"]
    zero = np.zeros_like(np.asarray(zr, dtype=np.float32))
    if field_left:
        naive_re = sub(mul(zr, c), mul(zi, zero))
        naive_im = add(mul(zr, zero), mul(zi, c))
    else:
        naive_re = sub(mul(c, zr), mul(zero, zi))
        naive_im = add(mul(c, zi), mul(zero, zr))
    n = zr.size
    v1_re = np.empty(n, np.float32)
    v1_im = np.empty(n, np.float32)
    for i in range(n):
        if field_left:
            v1_re[i] = fma(zr[i], c[i], _negate32(mul1(zi[i], 0.0)))
            v1_im[i] = fma(zr[i], 0.0, mul1(zi[i], c[i]))
        else:
            v1_re[i] = fma(c[i], zr[i], _negate32(mul1(0.0, zi[i])))
            v1_im[i] = fma(c[i], zi[i], mul1(0.0, zr[i]))
    plane_re = mul(zr, c) if field_left else mul(c, zr)
    plane_im = mul(zi, c) if field_left else mul(c, zi)
    return {"NAIVE": (naive_re.astype(np.float32), naive_im.astype(np.float32)),
            "FMA_V1": (v1_re, v1_im),
            "PLANEWISE_diagnostic": (plane_re.astype(np.float32),
                                     plane_im.astype(np.float32))}


def _interleave_c8(re: np.ndarray, im: np.ndarray) -> np.ndarray:
    """Plane assignment, never ``re + 1j*im``: the ADDITION destroys signed
    zeros (-0.0 + (+0.0) is +0.0), and the signed-zero rows are exactly the
    discriminating vectors."""
    out = np.empty(re.shape, dtype=np.complex64)
    out.real = re
    out.imag = im
    return out


def _platform_product(xp, kind: str, operands, scalar: Optional[float] = None
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """The backend's own bytes for one pattern, brought to the host."""
    if kind == "c8":
        ar, ai, br, bi = operands
        z1 = xp.asarray(_interleave_c8(ar, ai))
        z2 = xp.asarray(_interleave_c8(br, bi))
        product = xp.multiply(z1, z2)
    else:
        zr, zi, c = operands
        z = xp.asarray(_interleave_c8(zr, zi))
        if kind == "field_left":
            product = xp.multiply(z, xp.asarray(c))
        elif kind == "coefficient_left":
            product = xp.multiply(xp.asarray(c), z)
        elif kind == "python_float_left":
            product = xp.multiply(float(scalar), z)
        else:
            raise ValueError(kind)
    host = probe.to_host(product).astype(np.complex64)
    return np.ascontiguousarray(host.real), np.ascontiguousarray(host.imag)


def _disagreement_words(left: Tuple[np.ndarray, np.ndarray],
                        right: Tuple[np.ndarray, np.ndarray]) -> int:
    """How many float32 WORDS two (re, im) plane pairs differ in."""
    return sum(int(np.count_nonzero(np.ascontiguousarray(a).view(np.uint32)
                                    != np.ascontiguousarray(b).view(np.uint32)))
               for a, b in zip(left, right))


def _arms_apart(candidates: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> int:
    """How many WORDS the two LICENSABLE arms differ in on these vectors.

    Zero is the statement that this operand class has no power to classify
    anything — whatever the device returns, both arms "match" it. Measured
    candidate-against-candidate, so it never depends on the device answering."""
    return _disagreement_words(candidates["FMA_V1"], candidates["NAIVE"])


def _classify(measured: Tuple[np.ndarray, np.ndarray],
              candidates: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
    """One pattern's verdict, with the evidence for AMBIGUOUS_BOTH beside it.

    ``licensable_arms_disagreement_words`` is measured candidate-against-
    candidate, so a reader never has to take AMBIGUOUS_BOTH on trust: zero there
    IS the statement that the arms are bit-identical on these vectors under the
    policy the candidates were cut under, and a pattern that cannot tell the arms
    apart must not veto a licence it has no information about.
    """
    p_re, p_im = measured
    detail: Dict[str, Any] = {"matches": {}, "mismatch_words": {}}
    for arm, (c_re, c_im) in candidates.items():
        mism = (int(np.count_nonzero(p_re.view(np.uint32) != c_re.view(np.uint32)))
                + int(np.count_nonzero(p_im.view(np.uint32) != c_im.view(np.uint32))))
        detail["matches"][arm] = mism == 0
        detail["mismatch_words"][arm] = mism
    if "FMA_V1" in candidates and "NAIVE" in candidates:
        apart = _disagreement_words(candidates["FMA_V1"], candidates["NAIVE"])
        detail["licensable_arms_disagreement_words"] = apart
        detail["discriminates"] = apart > 0
    licensed = [arm for arm in ("FMA_V1", "NAIVE") if detail["matches"][arm]]
    if len(licensed) == 1:
        detail["classified"] = licensed[0]
    elif len(licensed) == 2:
        detail["classified"] = complex_fields.AMBIGUOUS_BOTH
    else:
        diagnostic = [arm for arm, hit in detail["matches"].items() if hit]
        detail["classified"] = diagnostic[0] if diagnostic else complex_fields.NEITHER
    return detail


def probe_environment(backend_name: str) -> Dict[str, Any]:
    """The execution environment this record was cut in.

    The first three fields are what :data:`complex_fields.ENVIRONMENT_DEFAULT_KEYS`
    keys the environment-default table on, so a record carries its own eligibility
    for that table and a record read on another machine is judged by where it was
    MEASURED, not by where it is read. The rest is for the reader: recorded so a
    future divergence can be attributed, never keyed, because nothing has measured
    the arm to depend on it and keying on it would refuse a known platform after
    an unrelated driver bump.
    """
    environment: Dict[str, Any] = {
        "backend": backend_name,
        "machine": platform.machine(),
        "cupy_version": getattr(cp, "__version__", None) if cp is not None else None,
        "keyed_on": list(complex_fields.ENVIRONMENT_DEFAULT_KEYS),
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
    }
    if cp is not None:
        try:
            environment["cuda_runtime_version"] = int(
                cp.cuda.runtime.runtimeGetVersion())
            props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
            name = props.get("name")
            environment["device_name"] = (name.decode() if isinstance(name, bytes)
                                          else str(name))
            environment["compute_capability"] = f"{props['major']}.{props['minor']}"
        except Exception as exc:  # noqa: BLE001 - a device query is not the gate
            environment["device_query_error"] = f"{type(exc).__name__}: {exc}"
    return environment


def _vector_census(named: Dict[str, np.ndarray]) -> Dict[str, int]:
    return {name: subnormal_words(array) for name, array in named.items()}


def measure_expansion_record(xp, backend_name: str) -> Dict[str, Any]:
    """One backend's probe record, in the schema ``expansion_license`` reads.

    The four required patterns are :data:`complex_fields.PROBE_PATTERNS`; the
    scalar-broadcast rotation is recorded as an extra diagnostic key (the
    module's contract ignores it; the sweep is the arbiter if it disagrees).

    THREE THINGS THE RECORD NOW STATES, because the licence depends on all
    three and a licence that cannot be audited is not one:

    * ``candidates`` — which subnormal policy the candidate arms were built
      under. It must be the policy the platform's bytes were cut under; scoring
      a flushed device against kept candidates produces the 54/48/125-word
      NEITHER of 2026-08-11, which is a broken comparison, not a platform fact.
    * ``environment`` — where the record was measured, which is what the
      environment-default table is keyed on.
    * ``vector_census`` — how many subnormal words the probe's own operands
      still carry. This is the operand-destruction blocker made visible: a
      flushing process that regenerated the vectors through a float conversion
      would arrive here with zero, having thrown away the lanes the probe is
      built on, and would then classify a strictly weaker question in silence.
    """
    _self_check_fma32()
    ops = candidate_arithmetic()
    rng = np.random.default_rng(SEED + 71)
    record: Dict[str, Any] = {"backend": backend_name, "patterns": {},
                              "detail": {}, "vectors": {},
                              "environment": probe_environment(backend_name),
                              "candidates": {
                                  "policy": ops["policy"],
                                  "semantics": ops["semantics"],
                                  "host_flushing": bool(
                                      _policy.backends.subnormals_flushed()),
                                  "requirement": (
                                      "the candidate arms must be cut under the "
                                      "same subnormal policy as the platform "
                                      "bytes they are scored against"),
                              }}
    if not ops["flush"] and record["candidates"]["host_flushing"]:
        raise RuntimeError(
            "the candidates were asked for under the IEEE-keep policy but this "
            "host's FPU measurably flushes float32 subnormals: the vectorized "
            "arms would be built under a policy the record does not claim")

    c8 = _c8_operands(rng)
    detail = _classify(_platform_product(xp, "c8", c8), _candidates_c8(*c8))
    record["patterns"]["c8_mul_c8"] = detail["classified"]
    record["detail"]["c8_mul_c8"] = detail
    record["vectors"]["c8_mul_c8"] = int(c8[0].size)

    zero_imag = _zero_imag_operands(rng)
    record["vector_census"] = {
        "subnormal_words": _vector_census(
            {"c8_ar": c8[0], "c8_ai": c8[1], "c8_br": c8[2], "c8_bi": c8[3],
             "zr": zero_imag[0], "zi": zero_imag[1], "c": zero_imag[2]}),
        "interleave_differing_words": _disagreement_words(
            (zero_imag[0], zero_imag[1]),
            (np.ascontiguousarray(_interleave_c8(zero_imag[0], zero_imag[1]).real),
             np.ascontiguousarray(_interleave_c8(zero_imag[0], zero_imag[1]).imag))),
    }
    if record["vector_census"]["subnormal_words"]["zr"] == 0:
        raise RuntimeError(
            "the probe's underflow rows carry no subnormal words: this process "
            "destroyed the operands that separate the arms before measuring "
            "anything, and the classification below would be of a weaker "
            "question than the one the artifact claims")

    # THE COUNT THAT MATTERS, and the one the census above does NOT give. The
    # subnormal-word census counts operand ENCODING: those 72 words are written
    # as integers and survive being written, then are flushed the instant the
    # device touches them. Under the ship policy the census therefore read 72
    # while the mixed patterns had exactly zero power to tell the arms apart.
    # This measures the power directly — candidate against candidate, under the
    # policy in force — and refuses to emit a record whose comparison could not
    # have discriminated anything. See :data:`_NORMAL_UNDERFLOWING_ZR`.
    record["vector_census"]["licensable_arms_apart"] = {
        "c8_mul_c8": _arms_apart(_candidates_c8(*c8)),
        "zero_imag_field_left": _arms_apart(
            _candidates_zero_imag(*zero_imag, field_left=True)),
        "zero_imag_coefficient_left": _arms_apart(
            _candidates_zero_imag(*zero_imag, field_left=False)),
    }
    blind = sorted(name for name, apart
                   in record["vector_census"]["licensable_arms_apart"].items()
                   if apart == 0)
    if blind:
        raise RuntimeError(
            f"the probe's operand classes {blind} cannot tell FMA_V1 from NAIVE "
            f"under the policy in force ({resolved_policy()!r}): the candidate "
            f"arms are bit-identical on every one of those vectors, so scoring "
            f"the device against them measures nothing and any verdict — an arm "
            f"or an AMBIGUOUS_BOTH — would be an artifact of the vector set. "
            f"Add operands that discriminate under THIS policy; all-normal "
            f"operands with an underflowing product are the class that does "
            f"(see _NORMAL_UNDERFLOWING_ZR)")

    for name, kind, field_left in (
            ("c8_mul_f4_field_left", "field_left", True),
            ("f4_mul_c8_coefficient_left", "coefficient_left", False)):
        detail = _classify(_platform_product(xp, kind, zero_imag),
                           _candidates_zero_imag(*zero_imag, field_left=field_left))
        record["patterns"][name] = detail["classified"]
        record["detail"][name] = detail
        record["vectors"][name] = int(zero_imag[0].size)

    # python-float scalar (dtdx = courant): candidates use the f32-rounded
    # scalar, coefficient-left orientation (stepping.py:1682, scalar LEFT).
    scalar_details = []
    for scalar in (0.35, 0.5, -0.35, 0.1):
        zr, zi, _ = zero_imag
        c = np.full(zr.shape, np.float32(scalar), dtype=np.float32)
        detail = _classify(
            _platform_product(xp, "python_float_left", zero_imag, scalar=scalar),
            _candidates_zero_imag(zr, zi, c, field_left=False))
        scalar_details.append({"scalar": scalar, **detail})
    classes = {entry["classified"] for entry in scalar_details}
    record["patterns"]["python_float_left"] = (classes.pop() if len(classes) == 1
                                               else "DISAGREES_ACROSS_SCALARS")
    record["detail"]["python_float_left"] = scalar_details
    # The scalar leg records its detail as a LIST (one entry per scalar) and used
    # to write no ``vectors`` entry at all — the only pattern of the four missing
    # one. Harmless while nothing read it; not harmless now that
    # ``expansion_license`` requires a positive vector count behind any
    # AMBIGUOUS_BOTH, which would refuse an HONEST record on a bookkeeping gap
    # rather than on a measurement. Written from the same operands the leg used.
    record["vectors"]["python_float_left"] = int(zero_imag[0].size)

    # Diagnostic: scalar-broadcast complex rotation (shifted[plane] *= c8 scalar).
    ar, ai, br, bi = c8
    phase = np.complex64(complex(float(br[0]), float(bi[0])))
    z1 = xp.asarray(_interleave_c8(ar, ai))
    product = probe.to_host(xp.multiply(z1, phase)).astype(np.complex64)
    scalar_c8 = _classify(
        (np.ascontiguousarray(product.real), np.ascontiguousarray(product.imag)),
        _candidates_c8(ar, ai, np.full(ar.shape, phase.real, np.float32),
                       np.full(ar.shape, phase.imag, np.float32)))
    record["detail"]["c8_scalar_broadcast_diagnostic"] = scalar_c8

    # Stamped AFTER the measurement, so the strip counters in the stamp cover
    # the compiles this record's own products triggered — a record that claims
    # the stripped policy with zero exercised compiles is refusable by name
    # (probe_record_policy_reasons checks the numbers, not just the name).
    record["subnormal_policy"] = policy_stamp(backend_name)
    return record


def run_expansion(results: Dict[str, Any], out_path: str,
                  probe_artifact_path: str) -> Optional[Dict[str, Any]]:
    """Both backends measured; the CUPY record is what the kernels bind to.

    Returns None on a CuPy host when the measurement may not license anything —
    either the strip could not be confirmed exercised (``leg['refusal']`` says
    so) or the patterns license no single constexpr. The policy check runs
    FIRST: a classification cut under the wrong subnormal policy is not a
    classification of the ship configuration at all (job 2327's lesson).
    """
    leg: Dict[str, Any] = {}
    leg["numpy"] = measure_expansion_record(np, "numpy")
    log(f"[expansion] numpy patterns: {leg['numpy']['patterns']}")
    cupy_record = None
    if cp is not None:
        cupy_record = measure_expansion_record(cp, "cupy")
        log(f"[expansion] cupy patterns: {cupy_record['patterns']}")
        save(cupy_record, probe_artifact_path)
        leg["cupy"] = cupy_record
        leg["artifact"] = probe_artifact_path
        policy_problems = ftz_strip_license_reasons()
        leg["subnormal_policy"] = policy_stamp("cupy")
        if policy_problems:
            leg["refusal"] = "subnormal policy: " + "; ".join(policy_problems)
            leg["licensed_expansion"] = None
            log(f"[expansion] POLICY REFUSAL: {leg['refusal']}")
            cupy_record = None  # nothing may license against these bytes
        else:
            log(f"[expansion] policy: {active_policy_name()} confirmed "
                f"exercised (nvrtc calls="
                f"{leg['subnormal_policy']['nvrtc_calls']} removed="
                f"{leg['subnormal_policy']['ftz_removed']} ptx_audited="
                f"{leg['subnormal_policy'].get('triton', {}).get('ptx_audited')})")
            verdict = complex_fields.expansion_license(cupy_record)
            expansion = verdict["expansion"]
            leg["license"] = verdict
            leg["licensed_expansion"] = verdict["arm"]
            log(f"[expansion] licence: arm={verdict['arm']} "
                f"basis={verdict['basis']} discriminating="
                f"{sorted(verdict['discriminating'])} non_discriminating="
                f"{verdict['non_discriminating']} policy={verdict['policy']!r}")
            if verdict["basis"] == "environment_default":
                log(f"[expansion] DEFAULTED FROM ENVIRONMENT (not measured this "
                    f"run): {verdict['why_arbitrary']}")
            if expansion is None:
                leg["refusal"] = (
                    "the cupy record does not license a single EXPANSION "
                    "constexpr; refused by name: "
                    + "; ".join(verdict["refusals"]))
                log(f"[expansion] REFUSAL: {leg['refusal']}")
    else:
        leg["cupy"] = "skipped: cupy is not importable on this host"
        log("[expansion] cupy leg SKIPPED cleanly: no cupy on this host")
    results["expansion"] = leg
    save(results, out_path)
    return cupy_record


# ---------------------------------------------------------------------------
# The in-file complex reference — the array path transcribed, complex branches in
# ---------------------------------------------------------------------------
#
# The shared probe's real-field transcription is reused where the complex path
# is unchanged (the recurrence, the constitutive accumulation); what is added
# here is only what real fields cannot carry: the Bloch wrap multiply, the
# conjugate on the down-shift, and complex storage throughout. The ``reference``
# leg pins this transcription against stepping.py itself, on both backends,
# BEFORE any kernel is measured against it.

def complex_shift(xp: Any, field: Any, axis: int, boundary: str, backward: bool,
                  phase: Optional[complex]) -> Any:
    """stepping._shift_up (:1723) / _shift_down (:1787), PERIODIC and METALLIC
    branches, with _apply_bloch_phase (:1846-1862) applied to the single wrapped
    plane: up-shift far plane ``*= phase``, down-shift plane 0 ``*= conj(phase)``
    — and SKIPPED entirely when the phase is None (:1762-1765), never done
    against 1+0j."""
    shifted = xp.roll(field, 1 if backward else -1, axis=axis)
    index = 0 if backward else -1
    if boundary == PERIODIC:
        if phase is not None:
            factor = phase.conjugate() if backward else phase
            shifted[_face(axis, index)] *= shifted.dtype.type(factor)
        return shifted
    if boundary == METALLIC:
        shifted[_face(axis, index)] = 0
        return shifted
    raise ValueError(f"boundary {boundary!r} has no complex reference here")


def complex_curl(xp: Any, sources: Dict[str, Any], term, dtdx: float,
                 backward: bool, boundaries: Sequence[str],
                 phases: Sequence[Optional[complex]]) -> Any:
    """stepping._curl_from_operands (:1601-1636), expression form — pinned
    bit-identical to the pooled form by ``test_step_scratch_is_bit_identical``."""
    _target, g1, a1, g2, a2, _dsig, _dsigu, _iyee = term
    shifted_first = complex_shift(xp, sources[g1], a1, boundaries[a1], backward,
                                  phases[a1])
    shifted_second = complex_shift(xp, sources[g2], a2, boundaries[a2], backward,
                                   phases[a2])
    return dtdx * ((shifted_first - sources[g1])
                   + (sources[g2] - shifted_second))


def complex_mask(curl: Any, iyee: Sequence[int], boundaries: Sequence[str]) -> None:
    """stepping._mask_non_owned_cells (:1865-1902), the branches this domain can
    reach: a metallic wall masks cell 0 of every shift-0 component; the folded-
    periodic top-cell clause is out of domain (no folds are admitted here)."""
    for axis in range(3):
        if iyee[axis] == 0 and boundaries[axis] == METALLIC:
            curl[_face(axis, 0)] = 0


def complex_pml_step(xp: Any, arrays: Dict[str, Any], coefficients: Dict[str, Any],
                     dtdx: float, sub_step: str, boundaries: Sequence[str],
                     phases: Sequence[Optional[complex]]) -> None:
    """One complex split-field PML curl sub-step, in place. The recurrence is the
    shared probe's transcription of stepping._apply_pml_update, unchanged: a
    complex run changes the ghost rule and the storage dtype, nothing else."""
    terms = probe.B_PML_TERMS if sub_step == "step_B" else probe.D_PML_TERMS
    backward = sub_step == "step_D"
    for term in terms:
        target, _g1, _a1, _g2, _a2, dsig, dsigu, iyee = term
        curl = complex_curl(xp, arrays, term, dtdx, backward, boundaries, phases)
        complex_mask(curl, iyee, boundaries)
        probe.reference_recurrence(
            arrays[target], arrays["fu_" + target], curl,
            coefficients["kms_" + dsig], coefficients["sinv_" + dsig],
            coefficients["kms_" + dsigu], coefficients["sinv_" + dsigu])


def complex_constitutive_step(arrays: Dict[str, Any], coefficients: Dict[str, Any],
                              side: str) -> None:
    """update_H (S:916-923) / update_E's no-polarization branch (S:980-989),
    through the shared probe's transcription of _apply_constitutive_pml —
    generic over dtype, so the complex path reuses it verbatim."""
    probe.reference_constitutive_step(side, arrays, arrays, coefficients)


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
                       names: Sequence[str]) -> Dict[str, Any]:
    return {name: xp.asarray(_seed_host_complex(shape, rng)) for name in names}


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


def _mutated_plan_phases(phases: Sequence[Optional[complex]], sub_step: str,
                         boundaries: Sequence[str],
                         host_mutation: Optional[str]
                         ) -> Tuple[Optional[complex], ...]:
    if host_mutation == "m1_unconjugated_down_wrap":
        # The classic sign error: the down-wrap takes the raw phase instead of
        # the conjugate. The plan conjugates for step_D itself, so handing it
        # pre-conjugated phases makes the kernel run the RAW phase there.
        if sub_step == "step_D":
            return tuple(None if p is None else p.conjugate() for p in phases)
        return tuple(phases)
    if host_mutation == "m5_one_times_instead_of_skip":
        return tuple(complex(1.0, 0.0) if p is None else p for p in phases)
    if host_mutation == "null_phase_on_metallic_axis":
        # Positive phase components keep every rotated ghost zero exactly +0.0,
        # so this branch is dead code by IEEE zero rules — the leg asserts it.
        return tuple(complex(0.8, 0.6) if (p is None and b == METALLIC) else p
                     for p, b in zip(phases, boundaries))
    return tuple(phases)


def one_curl_case(shape, config, dtdx, sub_step, guard, expansion,
                  kernel=None, host_mutation: Optional[str] = None) -> Dict[str, Any]:
    case: Dict[str, Any] = {
        "shape": list(shape), "config": config["name"],
        "boundaries": list(config["boundaries"]),
        "phases": [repr(p) for p in config["phases"]],
        "dtdx": repr(dtdx), "sub_step": sub_step, "guard": guard,
        "host_mutation": host_mutation,
    }
    skip = config_viable(shape, config)
    if skip:
        case["skipped"] = skip
        return case
    rng = np.random.default_rng(SEED + 3)
    names = FIELD_12 + FU_NAMES
    arrays = make_complex_state(cp, tuple(shape), rng, names)
    coefficients = probe.synthetic_coefficients(cp, tuple(shape),
                                                sub_step == "step_B")
    flat = probe.flatten_coefficients(coefficients)

    reference = {name: array.copy() for name, array in arrays.items()}
    complex_pml_step(cp, reference, coefficients, dtdx, sub_step,
                     config["boundaries"], config["phases"])

    codes = [CODE_OF[b] for b in config["boundaries"]]
    plan_phases = _mutated_plan_phases(config["phases"], sub_step,
                                       config["boundaries"], host_mutation)
    plan = complex_fields.plan_complex_pml_curl_from_arrays(
        sub_step, arrays, flat, codes, plan_phases, float(dtdx),
        int(expansion), kernel=kernel)
    plan.run(guard=guard)
    cp.cuda.runtime.deviceSynchronize()

    targets = ("Bx", "By", "Bz") if sub_step == "step_B" else ("Dx", "Dy", "Dz")
    names_checked = targets + tuple("fu_" + t for t in targets)
    case["verdict"] = combine({name: bit_compare(arrays[name], reference[name])
                               for name in names_checked})
    return case


def one_constitutive_case(shape, side, guard, expansion,
                          kernel=None) -> Dict[str, Any]:
    case: Dict[str, Any] = {"shape": list(shape), "side": side, "guard": guard,
                            "sub_step": "update_" + side}
    rng = np.random.default_rng(SEED + 5)
    targets, auxiliaries, sources = probe.constitutive_names(side)
    arrays = make_complex_state(cp, tuple(shape), rng,
                                targets + auxiliaries + sources)
    if side == "E":
        arrays.update(make_inverse_epsilon(cp, tuple(shape), rng))
    coefficients = probe.synthetic_constitutive_coefficients(
        cp, tuple(shape), half_integer=(side == "E"))
    flat = probe.flatten_coefficients(coefficients)

    reference = {name: (array.copy() if str(array.dtype) == "complex64" else array)
                 for name, array in arrays.items()}
    complex_constitutive_step(reference, coefficients, side)

    plan = complex_fields.plan_complex_constitutive_from_arrays(
        side, arrays, flat, int(expansion), kernel=kernel)
    plan.run(guard=guard)
    cp.cuda.runtime.deviceSynchronize()

    case["verdict"] = combine({name: bit_compare(arrays[name], reference[name])
                               for name in targets + auxiliaries})
    return case


def multi_step_case(shape, config, dtdx, expansion) -> Dict[str, Any]:
    """B -> H -> D -> E cycles from one seeded state — the auxiliaries are state,
    so a kernel right for one launch and wrong after is caught here."""
    case: Dict[str, Any] = {"shape": list(shape), "config": config["name"],
                            "dtdx": repr(dtdx), "cycles": MULTI_STEP_COUNT}
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

    plans = {
        "step_B": complex_fields.plan_complex_pml_curl_from_arrays(
            "step_B", arrays, curl_flat["step_B"], codes, config["phases"],
            float(dtdx), int(expansion)),
        "update_H": complex_fields.plan_complex_constitutive_from_arrays(
            "H", arrays, constitutive_flat["H"], int(expansion)),
        "step_D": complex_fields.plan_complex_pml_curl_from_arrays(
            "step_D", arrays, curl_flat["step_D"], codes, config["phases"],
            float(dtdx), int(expansion)),
        "update_E": complex_fields.plan_complex_constitutive_from_arrays(
            "E", arrays, constitutive_flat["E"], int(expansion)),
    }
    for _ in range(MULTI_STEP_COUNT):
        plans["step_B"].run()
        complex_pml_step(cp, reference, curl_tables["step_B"], dtdx, "step_B",
                         config["boundaries"], config["phases"])
        plans["update_H"].run()
        complex_constitutive_step(reference, constitutive_tables["H"], "H")
        plans["step_D"].run()
        complex_pml_step(cp, reference, curl_tables["step_D"], dtdx, "step_D",
                         config["boundaries"], config["phases"])
        plans["update_E"].run()
        complex_constitutive_step(reference, constitutive_tables["E"], "E")
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


def run_synthetic(results: Dict[str, Any], out_path: str, expansion: int,
                  kernel=None, host_mutation: Optional[str] = None,
                  label: str = "gate", guards=(False, True),
                  shapes=SHAPES, configs=CONFIGS,
                  sub_steps=CURL_SUB_STEPS, dtdx_values=DTDX) -> Dict[str, Any]:
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
            case = one_curl_case(shape, config, dtdx, sub_step, guard, expansion,
                                 kernel=kernel, host_mutation=host_mutation)
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


def run_constitutive(results: Dict[str, Any], out_path: str, expansion: int,
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
            case = one_constitutive_case(shape, side, guard, expansion,
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
    may not read a plain module global."""
    header = (
        "import triton\nimport triton.language as tl\n"
        f"PERIODIC = tl.constexpr({CODE_OF[PERIODIC]})\n"
        f"METALLIC = tl.constexpr({CODE_OF[METALLIC]})\n"
        f"NAIVE = tl.constexpr({_constexpr_value(complex_fields.NAIVE)})\n"
        f"FMA_V1 = tl.constexpr({_constexpr_value(complex_fields.FMA_V1)})\n\n")
    handle = tempfile.NamedTemporaryFile("w", suffix="_mutated_complex.py",
                                         delete=False, encoding="utf-8")
    handle.write(header + source)
    handle.close()
    _TEMPORARY.append(handle.name)
    spec = importlib.util.spec_from_file_location(
        "triton_mutated_complex_" + str(len(_TEMPORARY)), handle.name)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = module  # type: ignore[union-attr]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return getattr(module, kernel_name)


def shipped_source() -> str:
    """Helpers first (they may be the mutation target), then both kernels."""
    functions = (complex_fields._rotate_field_left,
                 complex_fields._mul_field_left,
                 complex_fields._mul_coefficient_left,
                 complex_fields.bloch_pml_curl_step,
                 complex_fields.bloch_constitutive_step)
    return "\n\n".join(textwrap.dedent(inspect.getsource(f.fn))
                       for f in functions)


_M2 = re.compile(r"= tl\.where\(w[xyz], (rot_re|rot_im), \w+_(?:re|im)\)")
_M3 = (
    (re.compile(r"tl\.math\.fma\(z_re, c, \(z_im \* 0\.0\) \* -1\.0\)"), "z_re * c"),
    (re.compile(r"tl\.math\.fma\(z_re, 0\.0, z_im \* c\)"), "z_im * c"),
    (re.compile(r"\(z_re \* c\) - \(z_im \* 0\.0\)"), "z_re * c"),
    (re.compile(r"\(z_re \* 0\.0\) \+ \(z_im \* c\)"), "z_im * c"),
    (re.compile(r"tl\.math\.fma\(c, z_re, \(0\.0 \* z_im\) \* -1\.0\)"), "c * z_re"),
    (re.compile(r"tl\.math\.fma\(c, z_im, 0\.0 \* z_re\)"), "c * z_im"),
    (re.compile(r"\(c \* z_re\) - \(0\.0 \* z_im\)"), "c * z_re"),
    (re.compile(r"\(c \* z_im\) \+ \(0\.0 \* z_re\)"), "c * z_im"),
)
_M4_FMA_RE = re.compile(r"tl\.math\.fma\(g_re, p_re, \(g_im \* p_im\) \* -1\.0\)")
_M4_FMA_IM = re.compile(r"tl\.math\.fma\(g_re, p_im, g_im \* p_re\)")
_M4_NAIVE_RE = re.compile(r"\(g_re \* p_re\) - \(g_im \* p_im\)")
_M4_NAIVE_IM = re.compile(r"\(g_re \* p_im\) \+ \(g_im \* p_re\)")
_M6 = re.compile(
    r"if BACKWARD:\n(\s+)wx, wy, wz = i == 0, j == 0, k == 0\n"
    r"(\s+)else:\n(\s+)wx, wy, wz = i == nx - 1, j == ny - 1, k == nz - 1")


def mutate_whole_array_phase(source: str) -> Tuple[str, int]:
    """m2: the rotation lands on EVERY lane instead of only the wrapped plane."""
    return _M2.sub(r"= \1", source), len(_M2.findall(source))


def mutate_zero_cross_terms_folded(source: str) -> Tuple[str, int]:
    """m3: the zero-imaginary product collapses to plane-wise {re*c, im*c} —
    exactly the constant-fold the compiler must not perform. Byte-wrong only on
    signed zeros (measured 4/8 targeted patterns), which the seeds carry."""
    hits = 0
    for needle, replacement in _M3:
        source, n = needle.subn(replacement, source)
        hits += n
    return source, hits


def mutate_expansion_arm_regrouped(source: str) -> Tuple[str, int]:
    """m4: the phase product's arms swapped (each arm computes the OTHER's
    grouping), so the mutation is live under EITHER probe binding."""
    hits = 0
    source, n = _M4_FMA_RE.subn("__M4_TMP_RE__", source); hits += n
    source, n = _M4_FMA_IM.subn("__M4_TMP_IM__", source); hits += n
    source, n = _M4_NAIVE_RE.subn("tl.math.fma(g_re, p_re, (g_im * p_im) * -1.0)",
                                  source); hits += n
    source, n = _M4_NAIVE_IM.subn("tl.math.fma(g_re, p_im, g_im * p_re)",
                                  source); hits += n
    source = source.replace("__M4_TMP_RE__", "(g_re * p_re) - (g_im * p_im)")
    source = source.replace("__M4_TMP_IM__", "(g_re * p_im) + (g_im * p_re)")
    return source, hits


def mutate_wrap_plane_swapped(source: str) -> Tuple[str, int]:
    """m6: the wrapped-plane predicate swapped between the two shift directions
    (face -1 vs 0) — the rotation lands on the wrong plane."""
    replaced = _M6.sub(
        "if BACKWARD:\n\\1wx, wy, wz = i == nx - 1, j == ny - 1, k == nz - 1\n"
        "\\2else:\n\\3wx, wy, wz = i == 0, j == 0, k == 0", source)
    return replaced, len(_M6.findall(source))


#: name -> (transform, kernels it must be measured through). ``both`` runs the
#: curl sweep AND the constitutive sweep — m3's plane-wise defect degrades into
#: nonzero sums inside the curl recurrence, and is guaranteed visible only at
#: the constitutive kernel's direct ``f_w`` store of the scaled source.
SOURCE_MUTATIONS: Dict[str, Tuple[Callable[[str], Tuple[str, int]], str]] = {
    "m2_phase_applied_to_whole_array": (mutate_whole_array_phase, "curl"),
    "m3_zero_cross_terms_folded": (mutate_zero_cross_terms_folded, "both"),
    "m4_expansion_arm_regrouped": (mutate_expansion_arm_regrouped, "curl"),
    "m6_wrap_plane_swapped": (mutate_wrap_plane_swapped, "curl"),
}

#: host mutation -> must_catch (None = record only, False = must NOT be caught).
HOST_MUTATIONS: Dict[str, Optional[bool]] = {
    "m1_unconjugated_down_wrap": True,
    "m5_one_times_instead_of_skip": None,
    "null_phase_on_metallic_axis": False,
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
_UNPHASED_AXIS_CONFIGS = tuple(c for c in CONFIGS
                               if c["name"] in ("unphased_all_periodic",
                                                "one_axis_phase_none"))


def run_mutations(results: Dict[str, Any], out_path: str,
                  expansion: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    shipped = shipped_source()
    for name, (transform, families) in SOURCE_MUTATIONS.items():
        mutated, hits = transform(shipped)
        if hits == 0:
            out[name] = {"error": "the mutation matched nothing; the needle "
                                  "has drifted away from the kernel"}
            log(f"[mut] {name}: NEEDLE MISSED — nothing exercised")
            results["mutations"] = out
            save(results, out_path)
            continue
        entry: Dict[str, Any] = {"sites": hits, "families": families}
        caught = False
        launches = 0
        if families in ("curl", "both"):
            counter = CountingKernel(compile_mutated(mutated,
                                                     "bloch_pml_curl_step"))
            summary = run_synthetic(
                results, out_path, expansion, kernel=counter,
                label="mutation:" + name, guards=(False,), shapes=SHAPES[:2],
                configs=_PHASED_CONFIGS, dtdx_values=(0.35,))
            entry["curl"] = {"ran": summary["ran"],
                             "identical": summary["identical"]}
            caught = caught or summary["identical"] < summary["ran"]
            launches += counter.launches
        if families in ("constitutive", "both"):
            counter = CountingKernel(compile_mutated(mutated,
                                                     "bloch_constitutive_step"))
            summary = run_constitutive(
                results, out_path, expansion, kernel=counter,
                label="mutation:" + name, guards=(False,), shapes=SHAPES[:2])
            entry["constitutive"] = {"ran": summary["ran"],
                                     "identical": summary["identical"]}
            caught = caught or summary["identical"] < summary["ran"]
            launches += counter.launches
        entry["launches"] = launches
        entry["caught"] = caught
        if launches == 0:
            entry["error"] = ("DISARMED: the mutated kernel never launched — "
                              "the leg measured nothing and FAILS")
        log(f"[mut] {name}: sites={hits} launches={launches} CAUGHT={caught}")
        out[name] = entry
        results["mutations"] = out
        save(results, out_path)

    # m4's guard-on leg is measured, never asserted: with fp fusion enabled the
    # compiler may contract either grouping, so catch there is information.
    mutated, hits = mutate_expansion_arm_regrouped(shipped)
    if hits:
        counter = CountingKernel(compile_mutated(mutated, "bloch_pml_curl_step"))
        summary = run_synthetic(
            results, out_path, expansion, kernel=counter,
            label="mutation:m4_guard_on", guards=(True,), shapes=SHAPES[:1],
            configs=_PHASED_CONFIGS[:2], dtdx_values=(0.35,))
        out["m4_guard_on_measured"] = {
            "ran": summary["ran"], "identical": summary["identical"],
            "caught": summary["identical"] < summary["ran"],
            "launches": counter.launches, "asserted": False}
        results["mutations"] = out
        save(results, out_path)

    for name, must_catch in HOST_MUTATIONS.items():
        configs = {"m1_unconjugated_down_wrap": _PHASED_CONFIGS,
                   "m5_one_times_instead_of_skip": _UNPHASED_AXIS_CONFIGS,
                   "null_phase_on_metallic_axis": _METALLIC_CONFIGS}[name]
        sub_steps = (("step_D",) if name == "m1_unconjugated_down_wrap"
                     else CURL_SUB_STEPS)
        summary = run_synthetic(
            results, out_path, expansion, host_mutation=name,
            label="mutation:" + name, guards=(False,), shapes=SHAPES[:2],
            configs=configs, sub_steps=sub_steps, dtdx_values=(0.35,))
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
        for step in range(1, REFERENCE_STEPS + 1):
            for slot, engine_call, tables in (
                    ("step_B", stepping.step_B, None),
                    ("update_H", stepping.update_H, "H"),
                    ("step_D", stepping.step_D, None),
                    ("update_E", stepping.update_E, "E")):
                engine_call(fields, pml)
                if slot in CURL_SUB_STEPS:
                    complex_pml_step(xp, arrays, curl_tables[slot], dtdx, slot,
                                     kinds, phases)
                else:
                    complex_constitutive_step(arrays,
                                              constitutive_tables[tables], tables)
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
#: can move (measured: 0/7680 on the NumPy transcription, 0/15360 on device,
#: job 2329) and a field-level must-catch there contradicts this gate's own
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
     "table_must_differ": True, "must_catch": True},
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


def run_engine(results: Dict[str, Any], out_path: str,
               record: Dict[str, Any]) -> Dict[str, Any]:
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
                fields, pml, "step_B", probe=record),
            "update_H": complex_fields.complex_constitutive_coverage(
                fields, pml, "H", probe=record),
            "step_D": complex_fields.complex_pml_curl_coverage(
                fields, pml, "step_D", probe=record),
            "update_E": complex_fields.complex_constitutive_coverage(
                fields, pml, "E", probe=record),
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
                fields, pml, "step_B", probe=record),
            "update_H": complex_fields.plan_complex_constitutive(
                fields, pml, "H", probe=record),
            "step_D": complex_fields.plan_complex_pml_curl(
                fields, pml, "step_D", probe=record),
            "update_E": complex_fields.plan_complex_constitutive(
                fields, pml, "E", probe=record),
        }
        case["phased"] = list(plans["step_B"].phased)
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
            f"phased={case['phased']} identical={case['verdict']['bit_identical']} "
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
    expansion = complex_fields.expansion_from_probe(record)
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
                naive_phases, float(grid.dt / grid.dx), int(expansion))
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

DEFAULT_LEGS = "expansion,reference,synthetic,constitutive,multi_step,mutations,engine"
DEVICE_LEGS = ("synthetic", "constitutive", "multi_step", "mutations", "engine")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--legs", default=DEFAULT_LEGS)
    parser.add_argument("--probe-artifact", default=None,
                        help="where to write the expansion probe JSON "
                             "(default: probe.json beside --out)")
    args = parser.parse_args(argv)
    legs = tuple(name.strip() for name in args.legs.split(",") if name.strip())
    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    probe_artifact = args.probe_artifact or os.path.join(out_dir, "probe.json")

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
    record: Optional[Dict[str, Any]] = None
    expansion: Optional[int] = None

    if "expansion" in legs:
        record = run_expansion(results, args.out, probe_artifact)
        if cp is not None and record is None:
            # A CuPy host that measured but may not license: policy refusal
            # (the strip unexercised) — leg["refusal"] carries the reason.
            failures.append("expansion leg refused: "
                            + str(results["expansion"].get("refusal")))
        elif record is not None:
            verdict = complex_fields.expansion_license(record)
            expansion = verdict["expansion"]
            if expansion is None:
                failures.append("expansion probe refused: "
                                + "; ".join(verdict["refusals"]))
    elif device_available:
        record = complex_fields.load_expansion_probe(probe_artifact)
        policy_problems = probe_record_policy_reasons(record)
        if policy_problems:
            failures.append("probe artifact policy: "
                            + "; ".join(policy_problems))
            record = None
        expansion = complex_fields.expansion_from_probe(record)

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
        if expansion is None:
            reason = ("the expansion probe licensed no constexpr; the kernel "
                      "may not be launched against an unmeasured platform")
            results["skipped_legs"][leg] = reason
            log(f"[{leg}] REFUSED: {reason}")
            save(results, args.out)
            continue
        if leg == "synthetic":
            summary = run_synthetic(results, args.out, expansion)
            if not summary["guarded_pass"]:
                failures.append(f"synthetic curl sweep: "
                                f"{summary['guarded_identical']}/"
                                f"{summary['guarded_ran']} guarded identical")
            log(f"[SUMMARY] synthetic curl guarded "
                f"{summary['guarded_identical']}/{summary['guarded_ran']}")
        elif leg == "constitutive":
            summary = run_constitutive(results, args.out, expansion)
            if not summary["guarded_pass"]:
                failures.append(f"constitutive sweep: "
                                f"{summary['guarded_identical']}/"
                                f"{summary['guarded_ran']} guarded identical")
        elif leg == "multi_step":
            cases = [multi_step_case(shape, config, 0.35, expansion)
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
            out = run_mutations(results, args.out, expansion)
            for name, entry in out.items():
                if entry.get("error"):
                    failures.append(f"mutation {name}: {entry['error']}")
                elif name in SOURCE_MUTATIONS and not entry.get("caught"):
                    failures.append(f"mutation {name} was NOT caught")
                elif name in HOST_MUTATIONS and entry.get("as_expected") is False:
                    failures.append(f"host mutation {name}: caught="
                                    f"{entry.get('caught')} but must_catch="
                                    f"{HOST_MUTATIONS[name]}")
        elif leg == "engine":
            summary = run_engine(results, args.out, record)
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

    licence = (complex_fields.expansion_license(record) if record is not None
               else None)
    results["summary"] = {
        "status": "passed" if not failures else "FAILED",
        "failures": failures,
        "device_legs_ran": device_available and expansion is not None,
        "certified_under_subnormal_policy":
            results["subnormal_policy"].get("policy"),
        "expansion_licence": None if licence is None else {
            "arm": licence["arm"],
            "basis": licence["basis"],
            "discriminating_patterns": licence["discriminating"],
            "non_discriminating_patterns": licence["non_discriminating"],
            "arms_coincide_on_every_pattern":
                licence["arms_coincide_on_every_pattern"],
            "environment_default": licence["environment_default"],
            "candidate_policy": licence["candidate_policy"],
            "policy_conditional": licence["policy_conditional"],
        },
    }
    save(results, args.out)
    log(f"[done] {args.out} status={results['summary']['status']} "
        f"failures={failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
