"""Complex-field (force_complex_fields / Bloch k_point) PML curl + constitutive tranche.

WIRED, as of the plan_step integration. ``launch.plan_step`` calls
:func:`complex_pml_curl_coverage` and :func:`complex_constitutive_coverage`
through the launch-side forwarders (launch.py:232-257, :2045, :2147, :2223) and
``fastpath._decide`` loads the probe artifact and forwards it into that composer
(fastpath.py:1701, :1722), with ``fastpath.ARM_CERTIFICATION`` carrying the
'complex PML' and 'complex' arms. The four complex slots are dispatchable, and
what keeps an unlicensed arm out of a kernel is the refusal enumeration below —
not the absence of a caller. THIS PARAGRAPH SAID "NOT WIRED" UNTIL 2026-08-15,
which was measured false: ``plan_step`` on a complex64 Grid/Fields/PML triple
already emitted per-arm reasons prefixed 'complex PML: ' and 'complex: ', so a
reader was being told the licensing hole was unreachable from production while
it sat on the dispatch path.

``fingerprints.json`` still carries no entry for this file — the byte gate binds
its own provenance record inside its results directory instead — so an edit here
invalidates no recorded certification through that mechanism, and the gate is the
only thing that can re-establish one.

WHAT THIS IS FOR. Seven corpus scripts demand it, and only seven: five Bloch-phase
runs (refl-angular.py, antenna_pec_ground_plane_1D.py, binary_grating_oblique.py,
mode_coeff_phase.py, oblique-planewave.py) and two complex-storage-at-k=0 runs
(wvg-src.py, solve-cw.py — only its complex-storage class; the mirror stays with
``symmetry.py``). None of the nine carries a susceptibility, a conductivity, BFAST
or chi2/chi3, which is why every one of those is refused outright rather than
carried. ``special_kz`` scripts (refl-angular-kz2d.py, parallel-wvgs-force.py) are
``grid.beta``, NOT ``bloch_phase``, and are refused by name.

TWO KERNELS, one per sub-step family, complex64 stepped as float32 WORD PAIRS
(complex64 is bit-layout (re, im) interleaved; a C-contiguous complex volume is a
C-contiguous float32 volume of shape ``(nx, ny, nz, 2)``, so complex cell ``w``
is words ``2*w`` and ``2*w + 1``):

* :func:`bloch_pml_curl_step` — complex ``stepping.step_B`` / ``step_D`` under
  split-field PML, with the Bloch wrap phase applied to the wrapped lane of each
  shifted operand;
* :func:`bloch_constitutive_step` — complex ``update_H`` / ``update_E`` (dsigw).
  The constitutive sub-step never sees the wrap, so it takes NO phase.

WHY THE PLANE TRANSCRIPTION IS LEGAL (from source): every recurrence on the
complex step path has real coefficients — curl (stepping.py:1648-1683),
split-field PML (:1952-1982), constitutive dsigw (:2112-2143) — and the imaginary
plane enters ONLY through the Bloch wrap multiply (:1767-1771 up / :1818-1822
down, both via ``_apply_bloch_phase`` :1846-1862) and through a complex source
amplitude, which is a ``sources``-side fact outside these kernels.
stepping.py:41-50 states this and adds the load-bearing invariant: ``inv_eps``
and every PML coefficient are float32 in BOTH storage modes (fields.py:571-573
against fields.py:1203-1204).

WHY IT IS NOT BYTE-TRIVIAL (measured, not provable from source):

1. NumPy's complex64 multiply is FUSED on the reference laptop (FMA_V1:
   ``re = fma(a_re, b_re, -fl(a_im*b_im))``, ``im = fma(a_re, b_im, +fl(a_im*b_re))``;
   the naive separately-rounded form differs in ~25% of words — measured 24.7%
   on 10^6 random pairs, NumPy 2.4.3). This is a
   compiled-dispatch fact per platform, so the kernel's ``EXPANSION`` constexpr is
   BOUND FROM A PROBE ARTIFACT (see the probe contract below) and the byte gate is
   the arbiter — a missing or ambiguous probe is a coverage refusal by name.
2. Every real-coefficient multiply on the array path is a FULL complex multiply
   with a zero-imaginary operand (np.multiply carries only 'FF->F' complex loops),
   so the transcription carries the zero cross terms LITERALLY: with the field on
   the left, ``re' = fma(f_re, c, (f_im*0.0) * -1.0); im' = fma(f_re, 0.0,
   +(f_im*c))``. Plane-wise ``{re*c, im*c}`` is byte-wrong on signed zeros
   (measured 4/8 targeted patterns), and a random-data sweep provably cannot
   catch the difference — the gate seeds signed-zero/subnormal rows for exactly
   this. The negated addend is spelled ``* -1.0``, never unary ``-``: Triton's
   ``0.0 - x`` unary-minus lowering canonicalizes ±0 addends to +0
   (semantic.py:386-391), and the zero-init composition census proved the
   canonicalization reachable at stored bytes (see :func:`_mul_field_left`).
3. Operand order changes bytes under FMA_V1. The array path's orientations are
   therefore normative and transcribed literally: ``dtdx`` scalar LEFT
   (stepping.py:1682), ``fu *= kms`` etc. field LEFT (:1976-1982), ``kps*fw`` /
   ``kms*fw_previous`` coefficient LEFT (:2133-2134, :2140-2142), ``D * inv_eps``
   D LEFT (:982-984), phase multiply plane LEFT with the phase pre-rounded to
   complex64 (:1862).

WHAT IS TRANSCRIBED, and from where (stepping.py unless noted):

* term table            ``B_CURL_TERMS`` / ``D_CURL_TERMS`` (:213-223)
* ghost rule            ``_shift_up`` (:1723) / ``_shift_down`` (:1787),
                        PERIODIC and METALLIC branches only
* Bloch wrap multiply   ``_apply_bloch_phase`` (:1846-1862): up-shift wrapped
                        plane ``*= phase`` (:1767-1771), down-shift plane 0
                        ``*= conj(phase)`` (:1818-1822); SKIPPED entirely when the
                        phase is None (:1768-1770 up / :1819-1821 down), which is
                        what keeps k = 0 bit-identical to the plain complex engine
* curl grouping         ``_curl_from_operands`` (:1601-1636)
* ownership mask        ``_mask_non_owned_cells`` (:1865-1902)
* PML recurrence        ``_apply_pml_update`` (:1905-1935)
* constitutive dsigw    ``_apply_constitutive_pml`` (:2065-2096)
* coefficient pairing   ``_curl_coefficients`` (:2418) / ``_constitutive_coefficients``
                        (:2428); host-bound, never chosen in-kernel
* phase legality        ``_bloch_phases`` (:2313-2367): phased axis must resolve
                        PERIODIC; PML on a phased axis is ADMITTED (measured
                        2.82e-07); the module docstring at :119-125 contradicts
                        this and is stale — nothing here transcribes from it
* phase value           ``grid.bloch_phase`` (grid.py:1011-1017): exp(2*pi*i*k*L),
                        Brillouin edge EXACTLY -1+0j

GROUPING CHOICES THE GATE MUST HOLD (Triton gives no way to force them):

* ``tl.math.fma`` is the explicit-FMA arm of ``EXPANSION``; whether it lowers to
  a single ``fma.rn.f32`` under ``enable_fp_fusion=False`` (rather than being
  split or re-fused) is a gate question, not a construction guarantee.
* The ``0.0 * t`` / ``t * 0.0`` zero cross terms must SURVIVE the MLIR pipeline
  — constant-folding them to a literal ``+0.0`` changes signed-zero bytes. A
  dedicated gate mutation replaces them with literal ``0.0`` and must be caught
  (or recorded as the pipeline having folded them, which fails the gate).
* The FMA arms' negated addends are spelled ``(a * b) * -1.0``, never unary
  ``-(a * b)``: unary minus lowers as ``0.0 - x`` and turns every ±0 addend
  into +0 before the fma, which the composition probe's zero-init rows measure
  as a stored-byte divergence from the array path. The composition probe's
  armed negation-mutation leg compiles the unary spelling back in and must see
  it diverge.
* The three ``@triton.jit`` helpers (:func:`_rotate_field_left`,
  :func:`_mul_field_left`, :func:`_mul_coefficient_left`) are inlined by Triton;
  the transcription assumes inlining does not reassociate their bodies.
* The phase rotation is computed on every lane and SELECTED onto the wrapped lane
  with ``tl.where`` (Triton has no per-lane scalar branch); ``where`` is a
  bitwise select, so unwrapped lanes keep their loaded bits by construction, but
  any compiler transform merging the two paths is the gate's to catch.
* The curl's plane-wise grouping ``((c_y - c) + (b - b_z))`` is held by the gate
  empirically, exactly as kernels.py:48-54 declares for the real-field kernel; a
  Triton version bump is a correctness event for this file too.
* Word addressing computes ``2*idx`` in int32; the coverage predicates halve the
  int32 element bound to ``2*ncells < 2**31`` to compensate.
* Within one complex cell the two word stores (re then im) replace the array
  path's single complex store — non-semantic (no aliasing between planes).

PROBE CONTRACT. The reference implementation's complex-multiply expansion is a
platform fact, so ``EXPANSION`` binds to a measured artifact, not a guess: a JSON
file (path in the ``MEEP_GPU_COMPLEX_EXPANSION_PROBE`` environment variable, or
passed as ``probe=``) of the form ``{"backend": "cupy", "patterns": {<name>:
"FMA_V1" | "NAIVE" | "AMBIGUOUS_BOTH", ...}}`` with every pattern of
:data:`PROBE_PATTERNS` present. The gate writes the artifact from its own
preflight probe and substitutes into CuPy runs, which is why the binding is to
the CuPy result.

THE LICENCE IS POLICY-CONDITIONAL, and a pattern that cannot discriminate is not
a veto. A pattern classifies :data:`AMBIGUOUS_BOTH` when EVERY licensable arm
reproduced the platform's bytes on its vectors — the arms are bit-identical
there and the pattern constrains nothing, so it is excluded from the agreement
test. This is the shipped x86 flush configuration's normal state on the three
mixed-dtype patterns: flush destroys exactly the subnormal lanes that separate
FMA_V1 from NAIVE, on the device AND in the candidates. What still refuses is
:data:`NEITHER` — no arm reproduced the bytes — and a disagreement between two
patterns that DO discriminate. When nothing discriminates at all, the arm falls
back to :data:`ENVIRONMENT_DEFAULTS`, the arm already measured for this
execution environment under a subnormal-KEEPING policy, and the licence records
that it was defaulted rather than measured this run; an environment with no row
refuses by name. :func:`expansion_license` is the whole rule and its verdict is
what artifacts should record; :func:`expansion_from_probe` is its constexpr.

ONE RULE, SEVERAL PATTERN SETS. The sibling tranches launch operand orientations
this file's four do not cover and each requires its own pattern on top of them —
``special_kz.BETA_PROBE_PATTERNS`` (imaginary coefficient LEFT) and
``folded_complex.PARITY_PROBE_PATTERNS`` (unit-real scalar LEFT). They pass their
superset to :func:`expansion_license` as ``probe_patterns`` rather than
reimplementing the rule. Until 2026-08-15 they did reimplement it, and the copies
had drifted: both accepted ``AMBIGUOUS_BOTH`` for their OWN pattern only and let
any base pattern that could not discriminate VETO the licence — the pre-clause
behaviour, which under the flush policy refuses a platform that has answered the
question on the patterns that can answer it. Two spellings of an arbiter is one
too many; there is now one.

Import contract: this module is importable WITHOUT Triton — the predicates and
plan builders (to ``None``) must answer on the laptop that is the merge bar.
Triton is imported at module scope inside a guard, the kernels degrade to
:class:`_UnavailableKernel`, and ``ENABLE_FP_FUSION`` is imported from
:mod:`kernels` only inside ``run()`` so the guard keeps its single spelling.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple


class _UnavailableKernel:
    """Fail at the launch expression while leaving host predicates importable."""

    __slots__ = ("name", "error")

    def __init__(self, name: str, error: BaseException) -> None:
        self.name = name
        self.error = error

    def __getitem__(self, grid):
        raise ImportError(
            f"the optional triton package is required to launch {self.name}; "
            f"host coverage remains available without it: {self.error}") from self.error


try:  # The host predicate and public refusal path must work without Triton.
    import triton
    import triton.language as tl
    _TRITON_IMPORT_ERROR: Optional[BaseException] = None
except ImportError as _exc:  # pragma: no cover - exercised by the absence test
    _TRITON_IMPORT_ERROR = _exc

    class _MissingTriton:
        @staticmethod
        def jit(function):
            return _UnavailableKernel(function.__name__, _TRITON_IMPORT_ERROR)

    class _MissingLanguage:
        @staticmethod
        def constexpr(value):
            return value

    triton = _MissingTriton()  # type: ignore[assignment]
    tl = _MissingLanguage()  # type: ignore[assignment]

from .coverage import (  # READ-ONLY imports; nothing here mutates coverage.py
    CONSTITUTIVE_SIDES,
    COVERED_BOUNDARIES,
    CURL_SUB_STEPS,
    CURL_TARGETS,
    Coverage,
    _boundary_kinds,
    _call,
    _coefficient_reasons,
    _inverse_epsilon_reasons,
    _susceptibility_reasons,
)
from .launch import SUB_STEPS, CupyPointer, _flat

# Same constexpr codes as ``kernels.PERIODIC``/``kernels.METALLIC``. Importing
# kernels.py here would import Triton unconditionally and defeat this module's
# host-only coverage route (the same reason no_pml.py restates them).
PERIODIC = tl.constexpr(0)
METALLIC = tl.constexpr(1)

#: The two transcription arms of the reference complex multiply. ``FMA_V1`` is
#: the fused form measured on the reference laptop's NumPy (rounded inner
#: products, fused outer add — the fmaddsub/fcmla shape); ``NAIVE`` is the
#: separately-rounded scalar-loop form a non-SIMD dispatch takes. The host binds
#: the constexpr from the probe artifact, never from a guess.
NAIVE = tl.constexpr(0)
FMA_V1 = tl.constexpr(1)

#: Host-side names for the constexpr arms, as the probe artifact spells them.
EXPANSIONS: Dict[str, int] = {"NAIVE": 0, "FMA_V1": 1}

#: Environment variable naming the probe artifact JSON (see the module docstring).
PROBE_PATH_ENVIRONMENT = "MEEP_GPU_COMPLEX_EXPANSION_PROBE"

#: The call-site patterns the probe must classify, one per operand orientation
#: the transcription carries (module docstring point 3). Every pattern that
#: DISCRIMINATES must agree for one ``EXPANSION`` constexpr to represent the
#: platform; disagreement between two discriminating patterns refuses.
PROBE_PATTERNS: Tuple[str, ...] = (
    "c8_mul_c8",                      # the phase rotation (S:1862, plane LEFT)
    "c8_mul_f4_field_left",           # fu *= kms family (S:1929-1935); D * inv_eps (S:982-984)
    "f4_mul_c8_coefficient_left",     # kps*fw / kms*fw_previous (S:2086-2095)
    "python_float_left",              # xp.multiply(dtdx, total) (S:1635)
)

#: A pattern verdict meaning EVERY licensable arm reproduced the platform's
#: bytes on that pattern's vectors. The arms are therefore bit-identical there,
#: the pattern CANNOT discriminate between them, and it constrains nothing — it
#: is excluded from the agreement test rather than vetoing it.
#:
#: THE VERDICT IS ONLY AS GOOD AS THE VECTORS, and on the shipped x86
#: configuration the vectors were the whole story. Under the flush policy the
#: mixed-dtype patterns came out AMBIGUOUS_BOTH (0 of 4560 words per pattern;
#: results/complex_expansion_flush_coincidence_2026-08-15/) — but that was a
#: property of the probe's operand set, not of the platform. Every row the probe
#: carried that could separate FMA_V1 from NAIVE fed the device a SUBNORMAL
#: OPERAND, which the ship policy destroys on first use. AUDIT 2026-08-15 found
#: a reachable class the probe never sampled — all-normal operands whose PRODUCT
#: underflows — on which the arms differ in 100% of rows under flush, and which
#: also separates them under keep. The probe now carries it
#: (``gate_triton_complex._NORMAL_UNDERFLOWING_ZR``), every pattern discriminates
#: under both policies, and the gate REFUSES to emit a record whose operand
#: classes have no power to tell the arms apart. Measured on the probe's own
#: vectors, candidate against candidate: ``c8_mul_f4_field_left`` and
#: ``f4_mul_c8_coefficient_left`` went 0 -> 128 words apart under flush and
#: 6 -> 74 under keep; ``python_float_left`` went 0 -> 96-128 under flush and
#: stays at 6 under keep (its coefficient is a broadcast scalar, so only some
#: scalars underflow the added rows); ``c8_mul_c8`` was never blind — 1088 words
#: apart under both, which is why it alone named the arm for a stretch.
#:
#: So an AMBIGUOUS_BOTH surviving today is a much stronger claim than one from
#: before that date, and :func:`expansion_license` will not accept it on the word
#: alone: the record must carry the vector count and the measured
#: candidate-against-candidate disagreement that back it.
AMBIGUOUS_BOTH = "AMBIGUOUS_BOTH"

#: A pattern verdict meaning NO arm reproduced the platform's bytes. This is the
#: refusal the probe exists for and it is never licensable: binding a constexpr
#: to a platform whose bytes no transcription reproduces yields a kernel that
#: differs from the array path. A NEITHER platform earns a new arm, not a shrug.
NEITHER = "NEITHER"

#: What an expansion licence is and is not portable across.
POLICY_CONDITIONAL_LICENCE = (
    "POLICY-CONDITIONAL: the arm was classified by scoring the platform's bytes "
    "against candidate arms computed under ONE subnormal policy, and it "
    "reproduces those bytes under that policy only. A licence obtained under "
    "'flush' does not transfer to 'keep' or the reverse. Scoring a flushed "
    "device against kept candidates is not a classification at all — it is the "
    "measured 54/48/125-word NEITHER of 2026-08-11, which the same platform "
    "answered as FMA_V1 (c8) and AMBIGUOUS_BOTH (mixed) once the candidates were "
    "cut under the policy in force — AMBIGUOUS_BOTH on that round's vector set, "
    "which carried no operand class able to separate the arms under flush; the "
    "probe now carries one and every orientation discriminates "
    "(results/complex_expansion_flush_coincidence_2026-08-15/). ENFORCED, not "
    "merely stated: expansion_policy_reasons() refuses a record whose policy is "
    "not the one the consuming run is cut under, _expansion_reasons() applies it "
    "whenever a policy is installed, and the dispatch seam applies it against "
    "the policy every dispatchable family was certified under, at the rung that "
    "loads the artifact. Until 2026-08-15 this sentence was the only thing "
    "standing between a flush-cut licence and a keep run, and a sentence is not "
    "a check."
)

#: The fields an :data:`ENVIRONMENT_DEFAULTS` row is KEYED on. A row matches only
#: when the probe record's own ``environment`` block carries every one of these
#: values exactly. Keep this list short and causal: the arm is a property of the
#: compiled dispatch, so the library that generates it and the machine it was
#: measured on are the key; everything else the record observes (CUDA runtime,
#: device name, driver) is carried in the artifact for the reader but is NOT a
#: key, because no measurement has shown the arm to depend on it. Widening the
#: key is the safe direction — an unmatched environment refuses.
ENVIRONMENT_DEFAULT_KEYS: Tuple[str, ...] = ("backend", "machine", "cupy_version")

#: The float32 subnormal policy EVERY COMPLEX ARM IN THIS PACKAGE WAS CERTIFIED
#: UNDER. Not "the policy this run uses" and not "the policy the artifact was cut
#: under" — the policy the gates that certified these kernels actually ran them
#: in. See ``fingerprints.json``'s family records: every complex family's bytes
#: were cut with this policy installed.
#:
#: WHY IT IS A SECOND CONSTANT AND NOT ``fastpath.CERTIFICATION_SUBNORMAL_POLICY``.
#: The dependency runs one way — ``fastpath`` is the DISPATCHER and imports this
#: library, so the library may not import it back — and that direction is not
#: merely a mechanical constraint, it is the point. A caller that reaches
#: ``launch.plan_step`` WITHOUT going through ``fastpath`` (it is public API, and
#: the sibling gates and probes use it) would get no certification check at all
#: if the fact lived only in the dispatcher. The arms know what they were
#: certified under; they should not have to ask the dispatcher.
#:
#: The two constants must agree — dispatch installs what the arms were certified
#: under, which is exactly why ``fastpath`` refuses a process that installed
#: anything else — and ``test_dispatch_expansion_refusal`` welds them so a change
#: to either is a test failure rather than a silence.
CERTIFIED_UNDER_SUBNORMAL_POLICY = "keep"

#: The arm ALREADY MEASURED for an execution environment, consulted ONLY when
#: this run's probe cannot discriminate at all (every pattern AMBIGUOUS_BOTH).
#:
#: This is not a fallback for a probe that failed — a NEITHER pattern still
#: refuses, and a probe that discriminates always wins and never reaches here.
#: It is the answer to a narrow question: when the policy in force has erased the
#: lanes that tell the arms apart, is the arm unknown? On a platform that has
#: been measured under a policy that KEEPS those lanes, no: the platform fact was
#: established there and a policy that erases the evidence does not unmake it.
#: Each row therefore carries the artifact that measured it, and a licence taken
#: from a row is stamped ``basis='environment_default'``, never 'measured'.
#:
#: THE LIMIT OF THAT ARGUMENT, measured 2026-08-15. It assumed the flush run and
#: the keep artifact disagree only about lanes flush erases. They did not: on the
#: class that separates the arms on the mixed orientations under FLUSH — all-normal
#: operands with an underflowing product — the keep artifact is SILENT, because
#: under keep that class was not in the probe either. So for a stretch the row was
#: repairing a gap with a measurement that had never covered it. The repair was to
#: close the gap at the source rather than lean harder on the table: the probe now
#: samples the class under both policies and the mixed patterns discriminate, so on
#: this platform a licence is MEASURED and this table is not reached. Keep the row
#: for a genuinely coincident future platform, and read it knowing that
#: 'environment_default' means "no evidence from this run", nothing stronger.
ENVIRONMENT_DEFAULTS: Tuple[Dict[str, str], ...] = (
    {
        "backend": "cupy",
        "machine": "x86_64",
        "cupy_version": "13.5.1",
        "expansion": "FMA_V1",
        "measured_under_policy": "ieee_keep_ftz_stripped",
        "artifact": ("apps/api/parity/meep_gpu/results/"
                     "complex_expansion_diagnosis_2026-08-11/gate_stripped.json"),
        "evidence": (
            "job 2328, stripped leg: the platform is EXACT FMA_V1 on all four "
            "probe patterns — 0 mismatch words over 2075-2280 vectors per "
            "pattern — measured under the ftz-stripped IEEE-keep policy, which "
            "keeps the subnormal lanes that separate the arms. The default leg "
            "of the same job, scored against kept candidates while the device "
            "flushed, is the 54/48/125 NEITHER this table exists to not repeat."),
    },
)

#: Elements per program — complex CELLS, not words. Restated from
#: ``kernels.DEFAULT_BLOCK`` for the same import reason as the boundary codes;
#: no autotune, because these kernels write their own inputs in place and the
#: tuner would silently apply the update dozens of times (kernels.py:56-59).
DEFAULT_BLOCK = 256

__all__ = [
    "AMBIGUOUS_BOTH",
    "CERTIFIED_UNDER_SUBNORMAL_POLICY",
    "DIAGNOSTIC_ARMS",
    "ENVIRONMENT_DEFAULTS",
    "ENVIRONMENT_DEFAULT_KEYS",
    "EXPANSIONS",
    "NEITHER",
    "POLICY_CONDITIONAL_LICENCE",
    "PROBE_PATH_ENVIRONMENT",
    "PROBE_PATTERNS",
    "environment_default",
    "expansion_certification_reasons",
    "expansion_license",
    "expansion_policy_reasons",
    "bloch_pml_curl_step",
    "bloch_constitutive_step",
    "bloch_phase_table",
    "complex_pml_curl_coverage",
    "complex_constitutive_coverage",
    "expansion_from_probe",
    "load_expansion_probe",
    "plan_complex_pml_curl",
    "plan_complex_pml_curl_from_arrays",
    "plan_complex_constitutive",
    "plan_complex_constitutive_from_arrays",
    "ComplexPmlCurlPlan",
    "ComplexConstitutivePlan",
]


# ---------------------------------------------------------------------------
# The kernels
# ---------------------------------------------------------------------------

@triton.jit
def _rotate_field_left(g_re, g_im, p_re, p_im, EXPANSION: tl.constexpr):
    """One full complex multiply, FIELD on the left — the S:1862 orientation.

    ``shifted[plane] *= shifted.dtype.type(phase)``: the wrapped lane times the
    complex64-rounded Bloch factor. FMA_V1 is the measured NumPy form (inner
    products separately rounded, outer add fused); NAIVE rounds every product.
    The FMA addend's negation is ``* -1.0`` — see :func:`_mul_field_left` for
    the lowering fact that makes the spelling load-bearing.
    """
    if EXPANSION == FMA_V1:
        out_re = tl.math.fma(g_re, p_re, (g_im * p_im) * -1.0)
        out_im = tl.math.fma(g_re, p_im, g_im * p_re)
    else:
        out_re = (g_re * p_re) - (g_im * p_im)
        out_im = (g_re * p_im) + (g_im * p_re)
    return out_re, out_im


@triton.jit
def _mul_field_left(z_re, z_im, c, EXPANSION: tl.constexpr):
    """(z) * (c + 0j) with the FIELD on the left — ``fu *= kms`` (S:1929-1935),
    ``field *= kms_u/sinv_u`` (S:1932-1935), ``D * inv_eps`` with D left (S:982-984).

    The ``z_im * 0.0`` / ``z_re * 0.0`` cross terms are the array path's zero
    cross terms and MUST NOT be folded to a literal 0.0: they carry the sign of
    the field's words into the result exactly as the full complex multiply does
    (measured byte-wrong on signed zeros without them). Their survival through
    the compiler is a gate question with a dedicated mutation.

    NEGATION IS ``* -1.0``, NEVER unary ``-``: Triton lowers ``-x`` as
    ``0.0 - x`` (triton 3.1.0, language/semantic.py:386-391), and under
    round-to-nearest ``0.0 - (+0.0)`` is ``+0.0`` — the addend's zero SIGN is
    lost — while the array path's complex multiply negates the rounded cross
    product sign-exactly (a NumPy/CuPy sign flip). ``* -1.0`` is the IEEE-exact
    negation (LLVM folds it to neg.f32; measured on device by the special_kz
    tranche's m9 product-layer pin, job 2332). The unary spelling carried here
    until 2026-08-12 canonicalized every ±0 addend to +0 before a store, and
    the zero-init composition census showed the class is REACHED: from all-+0.0
    state the array path stores ``re = -0.0`` words wherever a thin absorber's
    deepest ``kms = kappa - sigma`` goes negative (negative coefficient times a
    quiet +0.0 word), which the canonicalized kernel cannot reproduce
    (``probe_triton_complex_composition.py`` zero_init_quiet rows: 224/226
    words per odd step across fu/B/D/f_w volumes, measured on NumPy
    2026-08-12).
    """
    if EXPANSION == FMA_V1:
        out_re = tl.math.fma(z_re, c, (z_im * 0.0) * -1.0)
        out_im = tl.math.fma(z_re, 0.0, z_im * c)
    else:
        out_re = (z_re * c) - (z_im * 0.0)
        out_im = (z_re * 0.0) + (z_im * c)
    return out_re, out_im


@triton.jit
def _mul_coefficient_left(c, z_re, z_im, EXPANSION: tl.constexpr):
    """(c + 0j) * (z) with the COEFFICIENT on the left — ``kps * fw`` and
    ``kms * fw_previous`` (S:2086-2087, S:2093-2095), and the scalar-left
    ``xp.multiply(dtdx, total)`` (S:1635).

    Note the zero-sign asymmetry against :func:`_mul_field_left`: here the re
    cross term is ``0.0 * z_im`` (sign of the field's imag through the LEFT
    zero), and the FMA orientation fuses the coefficient product. Which factor
    is fused differs between the two forms, which is why the call-site
    orientation table is normative and both are probed. The FMA addend's
    negation is ``* -1.0`` — see :func:`_mul_field_left` for why.
    """
    if EXPANSION == FMA_V1:
        out_re = tl.math.fma(c, z_re, (0.0 * z_im) * -1.0)
        out_im = tl.math.fma(c, z_im, 0.0 * z_re)
    else:
        out_re = (c * z_re) - (0.0 * z_im)
        out_im = (c * z_im) + (0.0 * z_re)
    return out_re, out_im


@triton.jit
def bloch_pml_curl_step(
    f0, f1, f2,                       # targets: Bx,By,Bz or Dx,Dy,Dz (complex64 as words)
    u0, u1, u2,                       # auxiliaries: fu_B* or fu_D* (complex64 as words)
    g0, g1, g2,                       # sources: Ex,Ey,Ez or Hx,Hy,Hz (complex64 as words)
    kmx, sinvx, kmy, sinvy, kmz, sinvz,   # kms/sinv per axis, float32, one Yee sub-lattice
    nx, ny, nz, n_elem, dtdx,         # n_elem = COMPLEX cells; dtdx pre-rounded to f32
    pxr, pxi, pyr, pyi, pzr, pzi,     # per-axis complex64-rounded phase (conj for BACKWARD)
    BACKWARD: tl.constexpr,           # 0 = B (forward differences), 1 = D
    BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
    PHX: tl.constexpr, PHY: tl.constexpr, PHZ: tl.constexpr,
    EXPANSION: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One complex curl sub-step of all three components, split-field PML included.

    The real-field body (kernels.pml_curl_step) with three changes and only
    three: every field access is a float32 WORD PAIR at ``2*idx``/``2*idx+1``;
    the wrapped lane of each phased axis's shifted operand is rotated by the
    Bloch factor BEFORE the difference (the slot between ``_rolled`` and the
    subtract, S:1767-1771 / S:1818-1822); and every real-coefficient multiply is
    the zero-imaginary complex product per ``EXPANSION`` rather than a bare
    float multiply. ``PH* = 0`` compiles the rotation away entirely, mirroring
    the ``phase is None`` skip that keeps k = 0 bit-identical (S:1768-1770 /
    S:1819-1821).
    """
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = idx < n_elem
    nyz = ny * nz
    k = idx % nz
    plane = idx // nz
    j = plane % ny
    i = plane // ny

    # --- the ghost rule, per axis (stepping._shift_up / _shift_down) ------------
    # PERIODIC wraps; METALLIC serves an exact 0.0 past the wall — a (+0.0, +0.0)
    # word pair from `other=` IS the complex metallic ghost (S:1781-1783, S:1827-1829).
    if BACKWARD:
        si, sj, sk = i - 1, j - 1, k - 1
    else:
        si, sj, sk = i + 1, j + 1, k + 1
    vx, vy, vz = live, live, live
    if BCX == METALLIC:
        vx = live & (si >= 0) & (si < nx)
    else:
        si = tl.where(si < 0, nx - 1, tl.where(si == nx, 0, si))
    if BCY == METALLIC:
        vy = live & (sj >= 0) & (sj < ny)
    else:
        sj = tl.where(sj < 0, ny - 1, tl.where(sj == ny, 0, sj))
    if BCZ == METALLIC:
        vz = live & (sk >= 0) & (sk < nz)
    else:
        sk = tl.where(sk < 0, nz - 1, tl.where(sk == nz, 0, sk))

    ox = si * nyz + j * nz + k
    oy = i * nyz + sj * nz + k
    oz = i * nyz + j * nz + sk

    # --- wrapped-lane predicates, one plane per axis ----------------------------
    # Up-shift wraps where the unwrapped neighbour index was n (i == n-1); the
    # down-shift wraps where it was -1 (i == 0). One plane per phased axis,
    # matching the single-plane multiply of S:1862. On a collapsed (n = 1) axis
    # every lane is the wrap lane, which is exactly xp.roll's behaviour there.
    if BACKWARD:
        wx, wy, wz = i == 0, j == 0, k == 0
    else:
        wx, wy, wz = i == nx - 1, j == ny - 1, k == nz - 1

    # --- loads: two words per operand ------------------------------------------
    a_re = tl.load(g0 + 2 * idx, mask=live, other=0.0)
    a_im = tl.load(g0 + 2 * idx + 1, mask=live, other=0.0)
    b_re = tl.load(g1 + 2 * idx, mask=live, other=0.0)
    b_im = tl.load(g1 + 2 * idx + 1, mask=live, other=0.0)
    c_re = tl.load(g2 + 2 * idx, mask=live, other=0.0)
    c_im = tl.load(g2 + 2 * idx + 1, mask=live, other=0.0)
    a_y_re = tl.load(g0 + 2 * oy, mask=vy, other=0.0)
    a_y_im = tl.load(g0 + 2 * oy + 1, mask=vy, other=0.0)
    a_z_re = tl.load(g0 + 2 * oz, mask=vz, other=0.0)
    a_z_im = tl.load(g0 + 2 * oz + 1, mask=vz, other=0.0)
    b_x_re = tl.load(g1 + 2 * ox, mask=vx, other=0.0)
    b_x_im = tl.load(g1 + 2 * ox + 1, mask=vx, other=0.0)
    b_z_re = tl.load(g1 + 2 * oz, mask=vz, other=0.0)
    b_z_im = tl.load(g1 + 2 * oz + 1, mask=vz, other=0.0)
    c_x_re = tl.load(g2 + 2 * ox, mask=vx, other=0.0)
    c_x_im = tl.load(g2 + 2 * ox + 1, mask=vx, other=0.0)
    c_y_re = tl.load(g2 + 2 * oy, mask=vy, other=0.0)
    c_y_im = tl.load(g2 + 2 * oy + 1, mask=vy, other=0.0)

    # --- Bloch phase on the wrapped lane, BEFORE the difference -----------------
    # Which operand crossed which face follows the stencil: b_x/c_x crossed x,
    # a_y/c_y crossed y, a_z/b_z crossed z. Field LEFT (S:1862); the host passed
    # the CONJUGATE for BACKWARD (S:1818-1822). `tl.where` is a bitwise select,
    # so unwrapped lanes keep the loaded words untouched.
    if PHX:
        rot_re, rot_im = _rotate_field_left(b_x_re, b_x_im, pxr, pxi, EXPANSION)
        b_x_re = tl.where(wx, rot_re, b_x_re)
        b_x_im = tl.where(wx, rot_im, b_x_im)
        rot_re, rot_im = _rotate_field_left(c_x_re, c_x_im, pxr, pxi, EXPANSION)
        c_x_re = tl.where(wx, rot_re, c_x_re)
        c_x_im = tl.where(wx, rot_im, c_x_im)
    if PHY:
        rot_re, rot_im = _rotate_field_left(a_y_re, a_y_im, pyr, pyi, EXPANSION)
        a_y_re = tl.where(wy, rot_re, a_y_re)
        a_y_im = tl.where(wy, rot_im, a_y_im)
        rot_re, rot_im = _rotate_field_left(c_y_re, c_y_im, pyr, pyi, EXPANSION)
        c_y_re = tl.where(wy, rot_re, c_y_re)
        c_y_im = tl.where(wy, rot_im, c_y_im)
    if PHZ:
        rot_re, rot_im = _rotate_field_left(a_z_re, a_z_im, pzr, pzi, EXPANSION)
        a_z_re = tl.where(wz, rot_re, a_z_re)
        a_z_im = tl.where(wz, rot_im, a_z_im)
        rot_re, rot_im = _rotate_field_left(b_z_re, b_z_im, pzr, pzi, EXPANSION)
        b_z_re = tl.where(wz, rot_re, b_z_re)
        b_z_im = tl.where(wz, rot_im, b_z_im)

    # --- the curl (stepping._curl_from_operands): DO NOT flatten these parens ---
    # Complex add/sub is component-wise (measured 0/2^17 word mismatches), so the
    # grouping is kernels.py's per plane; the dtdx multiply is the zero-imag
    # complex product with the SCALAR on the left (S:1635).
    t0_re = ((c_y_re - c_re) + (b_re - b_z_re))
    t0_im = ((c_y_im - c_im) + (b_im - b_z_im))
    t1_re = ((a_z_re - a_re) + (c_re - c_x_re))
    t1_im = ((a_z_im - a_im) + (c_im - c_x_im))
    t2_re = ((b_x_re - b_re) + (a_re - a_y_re))
    t2_im = ((b_x_im - b_im) + (a_im - a_y_im))
    curl0_re, curl0_im = _mul_coefficient_left(dtdx, t0_re, t0_im, EXPANSION)
    curl1_re, curl1_im = _mul_coefficient_left(dtdx, t1_re, t1_im, EXPANSION)
    curl2_re, curl2_im = _mul_coefficient_left(dtdx, t2_re, t2_im, EXPANSION)

    # --- ownership mask (stepping._mask_non_owned_cells) -----------------------
    # Writes +0.0 to BOTH planes — the array path assigns complex zero (S:1896, S:1902).
    at_x, at_y, at_z = i == 0, j == 0, k == 0
    if BACKWARD:
        if BCY == METALLIC:
            curl0_re = tl.where(at_y, 0.0, curl0_re)
            curl0_im = tl.where(at_y, 0.0, curl0_im)
        if BCZ == METALLIC:
            curl0_re = tl.where(at_z, 0.0, curl0_re)
            curl0_im = tl.where(at_z, 0.0, curl0_im)
        if BCX == METALLIC:
            curl1_re = tl.where(at_x, 0.0, curl1_re)
            curl1_im = tl.where(at_x, 0.0, curl1_im)
        if BCZ == METALLIC:
            curl1_re = tl.where(at_z, 0.0, curl1_re)
            curl1_im = tl.where(at_z, 0.0, curl1_im)
        if BCX == METALLIC:
            curl2_re = tl.where(at_x, 0.0, curl2_re)
            curl2_im = tl.where(at_x, 0.0, curl2_im)
        if BCY == METALLIC:
            curl2_re = tl.where(at_y, 0.0, curl2_re)
            curl2_im = tl.where(at_y, 0.0, curl2_im)
    else:
        if BCX == METALLIC:
            curl0_re = tl.where(at_x, 0.0, curl0_re)
            curl0_im = tl.where(at_x, 0.0, curl0_im)
        if BCY == METALLIC:
            curl1_re = tl.where(at_y, 0.0, curl1_re)
            curl1_im = tl.where(at_y, 0.0, curl1_im)
        if BCZ == METALLIC:
            curl2_re = tl.where(at_z, 0.0, curl2_re)
            curl2_im = tl.where(at_z, 0.0, curl2_im)

    # --- split-field recurrence (stepping._apply_pml_update) -------------------
    # dsig/dsigu cycle: target 0 -> (y, z), 1 -> (z, x), 2 -> (x, y). Every
    # multiply is the zero-imag product with the FIELD on the left (S:1929-1935);
    # the subtraction of the curl and of fprev is plane-wise between them. The
    # loaded p registers ARE S:1928's fprev copy.
    km_x = tl.load(kmx + i, mask=live, other=0.0)
    si_x = tl.load(sinvx + i, mask=live, other=0.0)
    km_y = tl.load(kmy + j, mask=live, other=0.0)
    si_y = tl.load(sinvy + j, mask=live, other=0.0)
    km_z = tl.load(kmz + k, mask=live, other=0.0)
    si_z = tl.load(sinvz + k, mask=live, other=0.0)

    p0_re = tl.load(u0 + 2 * idx, mask=live, other=0.0)
    p0_im = tl.load(u0 + 2 * idx + 1, mask=live, other=0.0)
    q_re, q_im = _mul_field_left(p0_re, p0_im, km_y, EXPANSION)
    q_re = q_re - curl0_re
    q_im = q_im - curl0_im
    n0_re, n0_im = _mul_field_left(q_re, q_im, si_y, EXPANSION)
    e_re = tl.load(f0 + 2 * idx, mask=live, other=0.0)
    e_im = tl.load(f0 + 2 * idx + 1, mask=live, other=0.0)
    r_re, r_im = _mul_field_left(e_re, e_im, km_z, EXPANSION)
    r_re = (r_re + n0_re) - p0_re
    r_im = (r_im + n0_im) - p0_im
    v0_re, v0_im = _mul_field_left(r_re, r_im, si_z, EXPANSION)

    p1_re = tl.load(u1 + 2 * idx, mask=live, other=0.0)
    p1_im = tl.load(u1 + 2 * idx + 1, mask=live, other=0.0)
    q_re, q_im = _mul_field_left(p1_re, p1_im, km_z, EXPANSION)
    q_re = q_re - curl1_re
    q_im = q_im - curl1_im
    n1_re, n1_im = _mul_field_left(q_re, q_im, si_z, EXPANSION)
    e_re = tl.load(f1 + 2 * idx, mask=live, other=0.0)
    e_im = tl.load(f1 + 2 * idx + 1, mask=live, other=0.0)
    r_re, r_im = _mul_field_left(e_re, e_im, km_x, EXPANSION)
    r_re = (r_re + n1_re) - p1_re
    r_im = (r_im + n1_im) - p1_im
    v1_re, v1_im = _mul_field_left(r_re, r_im, si_x, EXPANSION)

    p2_re = tl.load(u2 + 2 * idx, mask=live, other=0.0)
    p2_im = tl.load(u2 + 2 * idx + 1, mask=live, other=0.0)
    q_re, q_im = _mul_field_left(p2_re, p2_im, km_x, EXPANSION)
    q_re = q_re - curl2_re
    q_im = q_im - curl2_im
    n2_re, n2_im = _mul_field_left(q_re, q_im, si_x, EXPANSION)
    e_re = tl.load(f2 + 2 * idx, mask=live, other=0.0)
    e_im = tl.load(f2 + 2 * idx + 1, mask=live, other=0.0)
    r_re, r_im = _mul_field_left(e_re, e_im, km_y, EXPANSION)
    r_re = (r_re + n2_re) - p2_re
    r_im = (r_im + n2_im) - p2_im
    v2_re, v2_im = _mul_field_left(r_re, r_im, si_y, EXPANSION)

    # --- stores: u then f (kernels.py:193-198 order), both planes ---------------
    tl.store(u0 + 2 * idx, n0_re, mask=live)
    tl.store(u0 + 2 * idx + 1, n0_im, mask=live)
    tl.store(u1 + 2 * idx, n1_re, mask=live)
    tl.store(u1 + 2 * idx + 1, n1_im, mask=live)
    tl.store(u2 + 2 * idx, n2_re, mask=live)
    tl.store(u2 + 2 * idx + 1, n2_im, mask=live)
    tl.store(f0 + 2 * idx, v0_re, mask=live)
    tl.store(f0 + 2 * idx + 1, v0_im, mask=live)
    tl.store(f1 + 2 * idx, v1_re, mask=live)
    tl.store(f1 + 2 * idx + 1, v1_im, mask=live)
    tl.store(f2 + 2 * idx, v2_re, mask=live)
    tl.store(f2 + 2 * idx + 1, v2_im, mask=live)


@triton.jit
def bloch_constitutive_step(
    f0, f1, f2,                       # targets: Hx,Hy,Hz or Ex,Ey,Ez (complex64 as words)
    w0, w1, w2,                       # auxiliaries: f_w_H* or f_w_E* (complex64 as words)
    g0, g1, g2,                       # sources: Bx,By,Bz or Dx,Dy,Dz (complex64 as words)
    e0, e1, e2,                       # inverse epsilon, float32 VOLUMES, E side only
    kp0, km0, kp1, km1, kp2, km2,     # kps/kms on each component's OWN axis, float32
    nx, ny, nz, n_elem,               # n_elem = COMPLEX cells
    SCALE: tl.constexpr,              # 0 = H (source is B), 1 = E (source is D*inv_eps)
    EXPANSION: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One complex constitutive PML sub-step (``dsigw``), all three components.

    The mirror of kernels.constitutive_step with word-pair addressing. No
    neighbour reads, no ghost rule, no ownership mask and NO PHASE — the
    constitutive sub-step never sees the wrap (S:907-993). ``prev`` is read
    BEFORE the ``w`` store (S:2083-2085); the wrong order looks like a slightly
    worse absorber inside the PML only. The two accumulations stay separate and
    left-to-right — never flattened to ``f + (kp*src - km*prev)`` — and each is
    the coefficient-LEFT zero-imag complex product (S:2086-2087, S:2093-2095).
    ``inv_eps`` is a float32 volume indexed by the COMPLEX cell index (one real
    coefficient per cell, both planes), never word-doubled.
    """
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = idx < n_elem
    k = idx % nz
    plane = idx // nz
    j = plane % ny
    i = plane // ny

    # Component 0 takes its coefficient from axis x, 1 from y, 2 from z (dsigw).
    kp_0 = tl.load(kp0 + i, mask=live, other=0.0)
    km_0 = tl.load(km0 + i, mask=live, other=0.0)
    kp_1 = tl.load(kp1 + j, mask=live, other=0.0)
    km_1 = tl.load(km1 + j, mask=live, other=0.0)
    kp_2 = tl.load(kp2 + k, mask=live, other=0.0)
    km_2 = tl.load(km2 + k, mask=live, other=0.0)

    # --- component 0 -----------------------------------------------------------
    prev_re = tl.load(w0 + 2 * idx, mask=live, other=0.0)   # BEFORE the store.
    prev_im = tl.load(w0 + 2 * idx + 1, mask=live, other=0.0)
    src_re = tl.load(g0 + 2 * idx, mask=live, other=0.0)
    src_im = tl.load(g0 + 2 * idx + 1, mask=live, other=0.0)
    if SCALE:
        ie = tl.load(e0 + idx, mask=live, other=0.0)
        src_re, src_im = _mul_field_left(src_re, src_im, ie, EXPANSION)  # D LEFT (S:982-984)
    tl.store(w0 + 2 * idx, src_re, mask=live)
    tl.store(w0 + 2 * idx + 1, src_im, mask=live)
    a_re = tl.load(f0 + 2 * idx, mask=live, other=0.0)
    a_im = tl.load(f0 + 2 * idx + 1, mask=live, other=0.0)
    t_re, t_im = _mul_coefficient_left(kp_0, src_re, src_im, EXPANSION)
    a_re = a_re + t_re
    a_im = a_im + t_im
    t_re, t_im = _mul_coefficient_left(km_0, prev_re, prev_im, EXPANSION)
    a_re = a_re - t_re
    a_im = a_im - t_im
    tl.store(f0 + 2 * idx, a_re, mask=live)
    tl.store(f0 + 2 * idx + 1, a_im, mask=live)

    # --- component 1 -----------------------------------------------------------
    prev_re = tl.load(w1 + 2 * idx, mask=live, other=0.0)
    prev_im = tl.load(w1 + 2 * idx + 1, mask=live, other=0.0)
    src_re = tl.load(g1 + 2 * idx, mask=live, other=0.0)
    src_im = tl.load(g1 + 2 * idx + 1, mask=live, other=0.0)
    if SCALE:
        ie = tl.load(e1 + idx, mask=live, other=0.0)
        src_re, src_im = _mul_field_left(src_re, src_im, ie, EXPANSION)
    tl.store(w1 + 2 * idx, src_re, mask=live)
    tl.store(w1 + 2 * idx + 1, src_im, mask=live)
    a_re = tl.load(f1 + 2 * idx, mask=live, other=0.0)
    a_im = tl.load(f1 + 2 * idx + 1, mask=live, other=0.0)
    t_re, t_im = _mul_coefficient_left(kp_1, src_re, src_im, EXPANSION)
    a_re = a_re + t_re
    a_im = a_im + t_im
    t_re, t_im = _mul_coefficient_left(km_1, prev_re, prev_im, EXPANSION)
    a_re = a_re - t_re
    a_im = a_im - t_im
    tl.store(f1 + 2 * idx, a_re, mask=live)
    tl.store(f1 + 2 * idx + 1, a_im, mask=live)

    # --- component 2 -----------------------------------------------------------
    prev_re = tl.load(w2 + 2 * idx, mask=live, other=0.0)
    prev_im = tl.load(w2 + 2 * idx + 1, mask=live, other=0.0)
    src_re = tl.load(g2 + 2 * idx, mask=live, other=0.0)
    src_im = tl.load(g2 + 2 * idx + 1, mask=live, other=0.0)
    if SCALE:
        ie = tl.load(e2 + idx, mask=live, other=0.0)
        src_re, src_im = _mul_field_left(src_re, src_im, ie, EXPANSION)
    tl.store(w2 + 2 * idx, src_re, mask=live)
    tl.store(w2 + 2 * idx + 1, src_im, mask=live)
    a_re = tl.load(f2 + 2 * idx, mask=live, other=0.0)
    a_im = tl.load(f2 + 2 * idx + 1, mask=live, other=0.0)
    t_re, t_im = _mul_coefficient_left(kp_2, src_re, src_im, EXPANSION)
    a_re = a_re + t_re
    a_im = a_im + t_im
    t_re, t_im = _mul_coefficient_left(km_2, prev_re, prev_im, EXPANSION)
    a_re = a_re - t_re
    a_im = a_im - t_im
    tl.store(f2 + 2 * idx, a_re, mask=live)
    tl.store(f2 + 2 * idx + 1, a_im, mask=live)


# ---------------------------------------------------------------------------
# The probe artifact — how EXPANSION is bound
# ---------------------------------------------------------------------------

def load_expansion_probe(path: Optional[str] = None) -> Any:
    """Read the probe artifact JSON, or None when it is absent or unreadable.

    ``path`` overrides the :data:`PROBE_PATH_ENVIRONMENT` environment variable.
    Unreadable is treated exactly like missing: both are refusals downstream,
    never a silent default expansion.

    THE ONE THING IT WILL NOT DO IS HAND BACK AN ARTIFACT THIS DISPATCH ALREADY
    REFUSED. When a refusal of :data:`PROBE_PATH_ENVIRONMENT` is standing, this
    returns that refusal instead of the record — see
    :mod:`meep_gpu.expansion_refusal`. Without this, refusing an artifact was
    advisory: a caller could drop a policy-mismatched record and the very next
    rung, holding nothing, would open the same file and license the arm anyway.
    That is the ``xferflush`` defect, and the environment variable is the route
    it took.

    The check reads MODULE STATE from inside this body rather than being
    installed over this name, because ``folded_complex`` and ``special_kz`` bind
    this function with ``from .complex_fields import load_expansion_probe`` at
    import time — a rebound module attribute would never reach them.

    An explicit ``path`` is NOT silenced by a standing refusal: naming a file is
    naming a different artifact, and what disqualifies that one is the policy
    clause at the coverage seam, not a refusal recorded against a variable it did
    not come from.
    """
    import json  # noqa: PLC0415
    import os  # noqa: PLC0415
    if path is None:
        from ..expansion_refusal import standing_refusal  # noqa: PLC0415
        refused = standing_refusal(PROBE_PATH_ENVIRONMENT)
        if refused is not None:
            return refused
    candidate = path if path is not None else os.environ.get(PROBE_PATH_ENVIRONMENT)
    if not candidate:
        return None
    try:
        with open(candidate, "r", encoding="utf-8") as handle:
            record = json.load(handle)
    except Exception:  # noqa: BLE001 - unreadable probe == missing probe
        return None
    return record if isinstance(record, dict) else None


def environment_default(environment: Any) -> Optional[Dict[str, str]]:
    """The :data:`ENVIRONMENT_DEFAULTS` row this environment matches, or None.

    Every key of :data:`ENVIRONMENT_DEFAULT_KEYS` must be present and equal. A
    partial match is no match: the table is a record of what was measured where,
    never a blanket default.
    """
    if not isinstance(environment, dict):
        return None
    for row in ENVIRONMENT_DEFAULTS:
        if all(environment.get(key) == row[key] for key in ENVIRONMENT_DEFAULT_KEYS):
            return dict(row)
    return None


#: Candidate arms a probe scores the platform against that may NEVER license a
#: constexpr — they exist to be excluded. If one of these reproduces the
#: platform's bytes on a pattern the record also calls :data:`AMBIGUOUS_BOTH`,
#: that pattern separated nothing at all: the known-wrong transcription passed
#: the same test the licensable arms passed.
DIAGNOSTIC_ARMS: Tuple[str, ...] = ("PLANEWISE_diagnostic", "FMA_V2_diagnostic")


def _ambiguity_evidence_reasons(record: Dict[str, Any], name: str) -> List[str]:
    """Why this :data:`AMBIGUOUS_BOTH` claim may not be taken at face value.

    AMBIGUOUS_BOTH is a claim about a MEASUREMENT — that the licensable arms were
    computed on real vectors and came out bit-identical. Read as a bare string it
    is indistinguishable from a comparison that had no vectors: the gate scores an
    arm by ``count_nonzero(platform != candidate)``, which is 0 over an empty
    array, so every arm "matches" an empty probe and every pattern classifies
    AMBIGUOUS_BOTH. Driven 2026-08-15: an empty-vector record was licensed
    FMA_V1 with zero refusals, from a comparison of nothing.

    So the evidence must be IN the record and must be positive:

    * ``vectors[name] > 0`` — the comparison had operands. The gate already
      records this per pattern.
    * ``detail[name]['licensable_arms_disagreement_words'] == 0`` — the arms were
      measured candidate-against-candidate and found equal. ``_classify`` writes
      exactly this field for this purpose; a record that omits it is asserting
      coincidence it never measured.
    * no arm of :data:`DIAGNOSTIC_ARMS` matched. Driven 2026-08-15: a record whose
      mixed patterns showed ``{NAIVE: 0, FMA_V1: 0, PLANEWISE_diagnostic: 0}``
      mismatch words was licensed 'measured' with no flag — but plane-wise is the
      transcription the zero cross terms exist to rule out, and a comparison it
      passes cannot discriminate anything.
    """
    out: List[str] = []
    vectors = record.get("vectors")
    count = vectors.get(name) if isinstance(vectors, dict) else None
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        out.append(
            f"pattern {name!r} claims {AMBIGUOUS_BOTH} but the record carries no "
            f"positive vector count for it (vectors[{name!r}]={count!r}): a "
            f"comparison over zero vectors makes EVERY arm match, so the claim "
            f"is indistinguishable from a comparison of nothing")
    detail = record.get("detail")
    entry = detail.get(name) if isinstance(detail, dict) else None
    if isinstance(entry, list):  # python_float_left is a list of per-scalar details
        entries = [item for item in entry if isinstance(item, dict)]
    elif isinstance(entry, dict):
        entries = [entry]
    else:
        entries = []
    if not entries:
        out.append(
            f"pattern {name!r} claims {AMBIGUOUS_BOTH} but the record carries no "
            f"detail block for it: the claim that the licensable arms coincide "
            f"is a measurement, and an unrecorded measurement is an assertion")
        return out
    for item in entries:
        apart = item.get("licensable_arms_disagreement_words")
        if not isinstance(apart, int) or isinstance(apart, bool):
            out.append(
                f"pattern {name!r} claims {AMBIGUOUS_BOTH} but its detail block "
                f"records no measured 'licensable_arms_disagreement_words': the "
                f"arms were never compared to each other, so nothing establishes "
                f"that they coincide rather than that the comparison was blind")
        elif apart != 0:
            out.append(
                f"pattern {name!r} claims {AMBIGUOUS_BOTH} while its own detail "
                f"block measures the licensable arms {apart} words APART on those "
                f"vectors: both cannot have reproduced the platform's bytes, so "
                f"the record contradicts itself")
        matches = item.get("matches")
        if isinstance(matches, dict):
            hit = [arm for arm in DIAGNOSTIC_ARMS if matches.get(arm)]
            if hit:
                out.append(
                    f"pattern {name!r} claims {AMBIGUOUS_BOTH} and the diagnostic "
                    f"arm(s) {hit} reproduced the platform's bytes too: a "
                    f"comparison the known-wrong transcription passes has no "
                    f"power to license anything")
        mismatch = item.get("mismatch_words")
        if isinstance(mismatch, dict):
            hit = [arm for arm in DIAGNOSTIC_ARMS if mismatch.get(arm) == 0]
            if hit:
                out.append(
                    f"pattern {name!r} claims {AMBIGUOUS_BOTH} and the diagnostic "
                    f"arm(s) {hit} matched at 0 mismatch words: a comparison the "
                    f"known-wrong transcription passes has no power to license "
                    f"anything")
    # Dedupe while keeping order — the per-scalar list can repeat one reason.
    seen: Dict[str, None] = {}
    for reason in out:
        seen.setdefault(reason, None)
    return list(seen)


class _UnreadablePolicy:
    """The type of :data:`UNREADABLE_POLICY`. One value, compared by identity."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return "UNREADABLE_POLICY"


#: What a caller passes for ``policy`` when it TRIED to determine the subnormal
#: policy this run will use and could not. Distinct from ``None`` on purpose:
#: ``None`` is "I am not asking", this is "I asked and got no answer", and the
#: two deserve opposite outcomes — see :func:`expansion_policy_reasons`. A
#: singleton rather than a string so it can never collide with a policy name.
UNREADABLE_POLICY = _UnreadablePolicy()


def expansion_policy_reasons(record: Any, policy: Any) -> List[str]:
    """Why this record's licence may not be consumed by a run under ``policy``.

    THE CHECK :func:`expansion_license` STRUCTURALLY CANNOT MAKE.
    :data:`POLICY_CONDITIONAL_LICENCE` has always SAID a licence obtained under
    one subnormal policy does not transfer to the other. Nothing enforced it:
    the verdict is computed from the record alone, so the same artifact was
    licensed identically whichever policy the reading process ran under
    (driven 2026-08-15 across ``keep``/``flush``/``match_meep``: arm FMA_V1,
    basis 'measured', zero refusals, every time).

    That is not academic on the shipped x86 path. The artifact this platform
    cuts under the ship policy is stamped ``'flush'``, and
    ``fastpath.CERTIFICATION_SUBNORMAL_POLICY`` is ``'keep'`` — dispatch REFUSES
    any process that installed anything else and installs 'keep' itself. So the
    natural artifact and the only permitted run policy disagree by construction,
    and under 'keep' the arms are measured ~24.7% of words apart. Consuming the
    flush-cut licence there is precisely the transfer the constant forbids.

    ``policy`` IS FOUR-VALUED. The third is the 2026-08-16 correction and the
    fourth is 2026-08-16's.

    * a policy name (``'keep'`` / ``'flush'``) — compare, and refuse a mismatch.
    * ``None`` — THE CALLER IS NOT ASKING. Artifact-reading tooling (a gate
      script, a licence audit block) reads a record as a DOCUMENT; it binds no
      arm and steps no field, so there is no run whose policy could disagree
      with it. Abstaining is the right answer to a question nobody asked.
    * :data:`UNREADABLE_POLICY` — THE CALLER ASKED AND COULD NOT READ THE
      ANSWER. That refuses.
    * :class:`~meep_gpu.expansion_refusal.ContradictedRunPolicy` — THE CALLER
      ASKED AND GOT TWO ANSWERS. Two declarations are open at once and they
      disagree about a fact this process has only one of. That refuses, and for
      a sharper reason than the unreadable case: it is the shape the laundering
      attempt takes. Declaring the policy an artifact was cut under, from inside
      a dispatch that declared something else, used to bind the arm the outer
      declaration would have refused (measured 2026-08-16 across all five gating
      predicates). Anything that is neither a policy name nor one of the three
      values above refuses here too, because a policy this clause cannot read is
      not a licence whatever it is.

    WHY THE THIRD VALUE EXISTS, AND WHY IT FAILS CLOSED. Until 2026-08-16 both
    of the last two were spelled ``None`` and both abstained, on the argument
    that refusing on an unread fact asserts one. That argument is wrong here and
    the asymmetry is the reason: consuming the licence asserts something too —
    it asserts that the licence TRANSFERS — and :data:`POLICY_CONDITIONAL_LICENCE`
    says it does not. Between two assertions, the house default takes the one
    that cannot emit a wrong word. "I could not read which policy is in force"
    is not evidence that a flush-cut licence holds under keep; it is the absence
    of the evidence that would be needed to consume it.

    That was not an edge case either. ``_policy_in_force`` could read nothing at
    EVERY rung of EVERY dispatch, because ``fastpath`` consumes the probe four
    rungs before it installs a policy — so the abstention was not a fallback,
    it was the whole of the behaviour on the shipped path.

    Failing closed is only affordable because the same change gives the clause
    something to judge with: dispatch DECLARES the policy it requires and will
    install (``expansion_refusal.declaring_run_policy``), so an honest run is
    judged rather than refused, and what fails closed is the run that genuinely
    cannot say which arithmetic it will use.

    A record with no policy stamp always refuses.
    """
    if record is None:
        return []
    if policy is UNREADABLE_POLICY:
        return ["the subnormal policy this run will use could not be read and "
                "was not declared, so the expansion probe artifact cannot be "
                "shown to have been cut under it: "
                f"{POLICY_CONDITIONAL_LICENCE}"]
    if policy is None:
        return []
    if not isinstance(policy, str):
        # Not a name, not None, not UNREADABLE_POLICY. The one value that
        # reaches here in practice is a contradiction between open declarations;
        # it is named so the audit trail says what happened, and anything else
        # refuses under the same clause rather than falling through to a
        # comparison against a value nothing can compare.
        try:
            from ..expansion_refusal import (  # noqa: PLC0415
                is_contradicted_run_policy)
            contradicted = is_contradicted_run_policy(policy)
        except Exception:  # noqa: BLE001 - an unreadable rule is not a licence
            contradicted = False
        if contradicted:
            return [f"the subnormal policy this run will use was declared "
                    f"{list(policy.policies)!r} by declarations open at the "
                    f"same time, which disagree about a fact this process has "
                    f"only one of, so no artifact can be shown to have been cut "
                    f"under the policy in force: {POLICY_CONDITIONAL_LICENCE}"]
        return [f"the subnormal policy this run will use resolved to "
                f"{policy!r}, which is not a policy name, so the expansion "
                f"probe artifact cannot be shown to have been cut under it: "
                f"{POLICY_CONDITIONAL_LICENCE}"]
    stamp = record.get("subnormal_policy") if isinstance(record, dict) else None
    resolved = stamp.get("resolved") if isinstance(stamp, dict) else None
    if resolved is None:
        return ["the expansion probe artifact states no resolved subnormal "
                "policy, so it cannot be shown to have been cut under the "
                f"{policy!r} policy this run requires; an artifact from before "
                "artifacts stated their policy certifies nothing"]
    if resolved != policy:
        return [f"the expansion probe artifact was cut under the {resolved!r} "
                f"float32 subnormal policy and this run requires {policy!r}: "
                f"{POLICY_CONDITIONAL_LICENCE}"]
    return []


def expansion_certification_reasons(policy: Any) -> List[str]:
    """Why an arm of THIS package may not be bound while ``policy`` is in force.

    A DIFFERENT QUESTION FROM :func:`expansion_policy_reasons`, and the whole
    defect was that only one of them was being asked.

    * :func:`expansion_policy_reasons` asks ARTIFACT vs RUN: does this
      measurement describe the arithmetic this run will use? It compares the
      record's stamp against the policy in force.
    * this asks KERNEL vs RUN: was this kernel ever validated for the arithmetic
      this run will use? It compares :data:`CERTIFIED_UNDER_SUBNORMAL_POLICY`
      against the policy in force, and the artifact does not enter into it.

    Both must hold, and neither implies the other. A flush-cut artifact under a
    flush run passes the first and fails the second here, because these kernels
    were certified under ``'keep'`` only.

    THE HOLE THAT MAKES THIS NECESSARY. Declarations are process-wide and
    deliberately NOT segregated by backend — a subnormal policy is one
    process-wide fact and two backends cannot legitimately be under different
    ones (see :data:`~meep_gpu.expansion_refusal._DECLARATIONS`). But their
    KERNELS are certified under different ones, and must be: the MPS executor
    flushes float32 subnormals natively and exposes no lever, so Metal HAS to
    declare ``'flush'``, while every complex family here was certified under
    ``'keep'``. With only the artifact-vs-run comparison in place, Metal's
    honest, correct declaration licensed these arms: MEASURED 2026-08-16, a
    top-level ``declaring_run_policy('flush')`` with a flush-cut artifact —
    arriving explicitly OR through the reader off the environment variable —
    scored ZERO reasons at all five gating seams and bound FMA_V1. The mirror
    direction (a keep-cut record under a declared flush) was already refused, by
    the artifact clause; nothing asked the certification question in either.

    THE FIX MAY NOT BE TO FORBID FLUSH. A backend whose device cannot keep
    subnormals must be able to say so, and refusing that would be refusing the
    truth. What is forbidden is binding an arm certified under one policy while
    another is in force, which is a statement about THIS package's kernels and
    leaves every other backend's funnel to make its own.

    ``policy=None`` abstains, for the same reason it does in
    :func:`expansion_policy_reasons`: artifact-reading tooling that binds no arm
    is asking nothing. Everything else that is not exactly
    :data:`CERTIFIED_UNDER_SUBNORMAL_POLICY` refuses — including an unreadable or
    contradicted policy, so this clause is self-sufficient and does not rely on
    being called beside another one that happens to catch those.
    """
    if policy is None:
        return []
    if policy == CERTIFIED_UNDER_SUBNORMAL_POLICY:
        return []
    return [f"every complex arm in this package was CERTIFIED under the "
            f"{CERTIFIED_UNDER_SUBNORMAL_POLICY!r} float32 subnormal policy and "
            f"the policy in force is {policy!r}; a kernel's bytes were only ever "
            f"compared against the array path under the policy its gate ran in, "
            f"so nothing establishes what this arm emits under a different one. "
            f"This is not a question about the probe artifact — a record cut "
            f"under {policy!r} would not answer it either. A backend whose "
            f"device requires {policy!r} may declare it and bind ITS OWN "
            f"certified arms; these are not those"]


def _policy_in_force() -> Any:
    """The subnormal policy this run will use, or :data:`UNREADABLE_POLICY`.

    TWO SOURCES, IN THIS ORDER, AND THE ORDER IS THE ARGUMENT.

    1. AN INSTALLED POLICY. A measurement of this process's actual arithmetic.
       Unambiguous, and it outranks anything anyone merely intends.
    2. A DECLARED POLICY (``expansion_refusal.declaring_run_policy``). What the
       surrounding dispatch has committed to install and refuses to run without.
       It exists because of an ordering that is not going to change: ``fastpath``
       consumes the probe four rungs before ``_subnormal_gate`` installs
       anything, so source 1 answers nothing at any rung that binds an arm.

    THE ORDER IS NOT A TIE-BREAK, and it is pinned by a test rather than only
    argued here (``test_dispatch_expansion_refusal`` ::
    ``test_an_installed_policy_outranks_a_declaration_that_disagrees_with_it``).
    Inverting it — reading the declaration first — hands a licence question to an
    intent while a measurement of this process's arithmetic is sitting right
    there, which is how a run that installed ``flush`` consumes a ``flush``-cut
    licence while some rung above it still says ``keep``.

    WHAT PROTECTS THE SHIPPED PATH WHEN THE TWO SOURCES DISAGREE — corrected
    2026-08-16, because the sentence that used to be here was wrong about the
    ladder. It said ``fastpath`` refuses any process that installed a policy
    other than the one it requires, "so the two sources disagreeing there is
    already a refusal at a HIGHER rung". Read the ladder: the composition window
    and ``plan_step`` are rung (5) (``fastpath.py``, the ``plan_step`` call);
    ``_subnormal_gate`` is rung (8b), roughly sixty lines LATER in the same
    function. Later is not higher, and the order is the whole point — 8b is
    deliberately last so that a policy is never installed for a configuration
    that was never going to dispatch.

    So the install/declaration disagreement is NOT prevented at the rung that
    binds. MEASURED 2026-08-16 with the install source doubled (this arm64 host
    cannot install ``flush``): installed ``flush`` + declared ``keep`` gives
    ``in_force == 'flush'``, and a flush-cut record used to score zero reasons at
    all five gating seams and bind FMA_V1 there. Rung 8b then refused the process
    and ``plan_fast_path`` returned None — but that DISCARDS a plan already
    built, and a discard is not a prevention: it protects only because nothing
    downstream consumes a discarded plan, which is a property of this one caller
    rather than of the licence rule. Driven end to end, rung (5) resolved a
    licence with zero reasons and 8b threw the plan away afterwards.

    What protects the shipped path AT THE RUNG THAT BINDS is
    :func:`expansion_certification_reasons`: the policy in force must be the one
    these arms were CERTIFIED under, whatever the artifact says and whichever
    source answered. Under it the doubled-install leg above refuses at rung (5),
    which is where the refusal belonged all along. 8b remains the process-wide
    guarantee that the arithmetic a dispatched kernel runs in is the one it was
    certified in; it is a backstop, not the thing that keeps a licence honest.

    A DECLARATION THAT CONTRADICTS ANOTHER DECLARATION is a different matter and
    is returned as it is found: ``declared_run_policy`` answers with a
    :class:`~meep_gpu.expansion_refusal.ContradictedRunPolicy`, this function
    passes it through, and :func:`expansion_policy_reasons` refuses on it by
    name. Collapsing it to :data:`UNREADABLE_POLICY` here would refuse too, but
    would report "could not be read" about a policy that was declared twice —
    the audit trail has to say which.

    Otherwise :data:`UNREADABLE_POLICY` — NOT ``None``. A caller that cannot say
    which arithmetic it will use has not earned a policy-conditional licence, and
    ``None`` here would be read downstream as "not asking" and abstain. This
    function always asks.
    """
    try:
        from .. import subnormal_policy as _sp  # noqa: PLC0415
    except Exception:  # noqa: BLE001 - a package that cannot answer does not answer
        _sp = None
    if _sp is not None:
        try:
            if _sp.policy_is_installed():
                return _sp.get_subnormal_policy()
        except Exception:  # noqa: BLE001
            return UNREADABLE_POLICY
    try:
        from ..expansion_refusal import declared_run_policy  # noqa: PLC0415
        declared = declared_run_policy()
    except Exception:  # noqa: BLE001
        declared = None
    if declared is not None:
        return declared
    return UNREADABLE_POLICY


def expansion_license(record: Any,
                      probe_patterns: Optional[Sequence[str]] = None
                      ) -> Dict[str, Any]:
    """The full licensing verdict a probe record earns — arm, basis, refusals.

    THE RULE, in order. A pattern is DISCRIMINATING when it names one arm and
    NON-DISCRIMINATING when it classifies :data:`AMBIGUOUS_BOTH` (every
    licensable arm reproduced the platform's bytes there, so the pattern cannot
    prefer one).

    ``probe_patterns`` is the set the record must classify, defaulting to
    :data:`PROBE_PATTERNS`. The sibling tranches pass a SUPERSET — each adds the
    operand orientation its kernel launches and nothing else — so that one
    implementation of the rule serves every complex family. It is a parameter
    rather than a copy for the reason the development notes gives for not keeping two
    spellings of anything: a second implementation is a second place for the
    exclusion clause to be subtly wrong, and this clause is the arbiter that
    licenses a constexpr into a kernel.

    1. The record must name the CuPy backend — the kernels bind to CuPy's bytes.
    2. Every pattern of ``probe_patterns`` must be classified. Anything that
       is neither an arm of :data:`EXPANSIONS` nor :data:`AMBIGUOUS_BOTH` —
       :data:`NEITHER`, a diagnostic arm, a disagreement across scalars, a
       missing key, a non-string JSON value — REFUSES BY NAME. This is the whole
       point of the probe and no other clause may soften it.
    3. The record must declare that its CANDIDATES were computed under the same
       subnormal policy as the bytes they were scored against — ON EVERY PATH,
       not only when something is ambiguous. A comparison cut across a policy
       boundary is broken whatever verdict it produced (2026-08-11's NEITHER is
       the same defect wearing a different answer), and a record that states no
       policy at all states nothing. AUDIT 2026-08-15: this clause used to be
       guarded by ``if verdict['non_discriminating']``, so a record whose four
       patterns all named an arm was licensed with ``basis='measured'`` while
       openly declaring ``resolved='keep'`` against ``candidates.policy='flush'``
       — and a record carrying no policy block at all was licensed too.
    4. Every :data:`AMBIGUOUS_BOTH` pattern must carry the EVIDENCE that its
       comparison had vectors and had no discriminating power — ``vectors[name]``
       above zero and a ``detail`` block whose measured
       ``licensable_arms_disagreement_words`` is zero. AMBIGUOUS_BOTH asserted
       without that evidence is indistinguishable from a comparison of nothing:
       ``count_nonzero`` over an empty array is zero, so an empty probe makes
       every arm "match". A pattern whose ``detail`` also shows a DIAGNOSTIC arm
       (plane-wise, FMA_V2) reproducing the bytes refuses — a comparison the
       known-wrong transcription passes has no power to license anything.
    5. If any pattern discriminates, the discriminating patterns decide: they
       must agree on one arm, and the non-discriminating ones are excluded from
       the agreement test rather than vetoing it. Basis ``'measured'``.

       EXCLUDED AND DISAGREEING ARE OPPOSITE SITUATIONS and clause 4 is the only
       thing keeping them apart, so read them together. A pattern is excluded
       when it CANNOT tell the arms apart — its own detail measures them zero
       words apart on a positive vector count. A pattern that CAN tell them
       apart and names a different arm from another such pattern lands in
       ``discriminating`` and REFUSES here; a pattern that claims
       :data:`AMBIGUOUS_BOTH` while its own detail measures the arms some words
       apart is a record contradicting itself and refuses in clause 4. Neither
       is ever excluded. Conflating the two would license an arm the platform
       measurably does not implement, which is the whole failure this mechanism
       exists to prevent, so ``non_discriminating`` carries only patterns whose
       blindness was MEASURED, and a rejected ambiguity claim is listed
       separately under ``ambiguity_refused``.
    6. If NOTHING discriminates, the arm is not measurable in this run. Consult
       :data:`ENVIRONMENT_DEFAULTS` with the record's ``environment`` block: a
       matching row licenses ITS arm with basis ``'environment_default'`` and the
       artifact that measured it; no matching row REFUSES BY NAME.

    Case 6 also records ``arms_coincide_on_every_pattern``: on those vectors,
    under that policy, both arms produce identical bits, so the arm taken from
    the table cannot change the emitted words there. That is a statement about
    the probe's vectors, not a proof over all inputs — and
    :data:`ENVIRONMENT_DEFAULTS`' own caveat says what that is worth.

    WHAT THIS FUNCTION STILL DOES NOT DO, deliberately: it never compares the
    record's policy against the policy THIS PROCESS will run under. It cannot —
    a record is a document, and which policy is in force is a property of the
    caller. :func:`expansion_policy_reasons` is that check, and
    :func:`_expansion_reasons` applies it at the coverage seam.
    """
    patterns_required: Tuple[str, ...] = tuple(
        PROBE_PATTERNS if probe_patterns is None else probe_patterns)
    verdict: Dict[str, Any] = {
        "expansion": None,
        "arm": None,
        "basis": None,
        "refusals": [],
        "probe_patterns": list(patterns_required),
        "discriminating": {},
        "non_discriminating": [],
        "ambiguity_refused": [],
        "arms_coincide_on_every_pattern": False,
        "policy": None,
        "policy_resolved": None,
        "candidate_policy": None,
        "environment": None,
        "environment_default": None,
        "refused_by_policy": None,
        "policy_conditional": POLICY_CONDITIONAL_LICENCE,
    }
    refuse = verdict["refusals"].append

    # CLAUSE 0 — THE INVARIANT, ENFORCED WHERE EVERY ARM IS BOUND.
    #
    # If any rung of this dispatch refused this artifact, NO later rung may bind
    # an arm from it: not by re-reading the environment variable, not through a
    # re-export, not through ENVIRONMENT_DEFAULTS. This function is where that
    # is enforceable rather than merely stated, because it is the ONE funnel
    # every EXPANSION constexpr on this backend passes through —
    # ``expansion_from_probe``, ``special_kz.beta_expansion_license`` and
    # ``folded_complex.parity_expansion_license`` all end here, and clause 6's
    # environment-default route is INSIDE it. A refusal honoured anywhere else
    # would be a refusal something could route around.
    #
    # It is first, ahead of the ``isinstance(record, dict)`` guard below, because
    # a refusal must refuse BY ITS OWN REASONS. Falling through to "no probe
    # record" would report the artifact as absent — which is the exact conflation
    # this value exists to end, and which would make the audit trail say the
    # opposite of what happened.
    from ..expansion_refusal import RefusedExpansionProbe  # noqa: PLC0415
    if isinstance(record, RefusedExpansionProbe):
        verdict["refused_by_policy"] = list(record.reasons)
        refuse(f"the expansion probe artifact offered through {record.key} was "
               f"REFUSED by this dispatch and may not license an arm at any "
               f"later rung: " + "; ".join(record.reasons))
        return verdict

    if not isinstance(record, dict):
        refuse("no probe record: the EXPANSION constexpr is a measured platform "
               "fact and may not be guessed")
        return verdict
    if record.get("backend") != "cupy":
        refuse(f"the probe record names backend {record.get('backend')!r}, not "
               f"'cupy': the kernels bind to CuPy's bytes and a host record "
               f"licenses nothing about them")
        return verdict

    stamp = record.get("subnormal_policy")
    if isinstance(stamp, dict):
        verdict["policy"] = stamp.get("policy")
        verdict["policy_resolved"] = stamp.get("resolved")
    candidates = record.get("candidates")
    if isinstance(candidates, dict):
        verdict["candidate_policy"] = candidates.get("policy")
    if isinstance(record.get("environment"), dict):
        verdict["environment"] = dict(record["environment"])

    patterns = record.get("patterns")
    if not isinstance(patterns, dict):
        refuse("the probe record carries no 'patterns' map, so no pattern was "
               "classified at all")
        return verdict

    codes: List[int] = []
    for name in patterns_required:
        value = patterns.get(name)
        # ``isinstance(value, str)`` FIRST: ``value in EXPANSIONS`` hashes the
        # JSON value, and a list or an object raises TypeError out of a function
        # two written contracts promise never raises into a caller
        # (``load_expansion_probe``'s "unreadable is treated exactly like
        # missing" and ``plan_complex_pml_curl``'s "None is the only refusal").
        # Measured 2026-08-15: ``patterns['c8_mul_c8'] = ['FMA_V1']`` took the
        # gate's engine leg down with an unhandled TypeError.
        if isinstance(value, str) and value in EXPANSIONS:
            # DISCRIMINATED. It named an arm; it votes, and clause 5 refuses if
            # the votes disagree. Never excluded.
            verdict["discriminating"][name] = value
            codes.append(EXPANSIONS[value])
        elif isinstance(value, str) and value == AMBIGUOUS_BOTH:
            # CLAIMS IT COULD NOT DISCRIMINATE. Only a claim whose evidence
            # holds is excluded; one whose own detail measures the arms apart
            # (or has no vectors, or that a diagnostic arm also passed) is a
            # contradiction, and it refuses instead of being quietly dropped.
            ambiguity_reasons = _ambiguity_evidence_reasons(record, name)
            if ambiguity_reasons:
                verdict["ambiguity_refused"].append(name)
                for reason in ambiguity_reasons:
                    refuse(reason)
            else:
                verdict["non_discriminating"].append(name)
        elif value is None:
            refuse(f"pattern {name!r} is not classified; every pattern of "
                   f"{tuple(patterns_required)} must be measured")
        else:
            refuse(f"pattern {name!r} classified {value!r} ({type(value).__name__}): "
                   f"the platform's bytes are reproduced by NO licensable arm "
                   f"there, and a constexpr bound to such a platform yields a "
                   f"kernel that differs from the array path — this platform "
                   f"earns a new arm, not a shrug")
    if verdict["refusals"]:
        return verdict

    verdict["arms_coincide_on_every_pattern"] = not verdict["discriminating"]

    # Clause 3, on EVERY path. A record whose candidates were cut under one
    # policy and whose bytes were cut under another is a broken comparison
    # whatever it concluded; so is a record that names no policy at all.
    resolved = verdict["policy_resolved"]
    candidate_policy = verdict["candidate_policy"]
    if not resolved:
        refuse("the record states no resolved subnormal policy: every verdict "
               "in it — an arm as much as an ambiguity — is only readable "
               "against the policy the bytes were cut under, and an artifact "
               "from before artifacts stated their policy states nothing")
    elif candidate_policy != resolved:
        refuse(f"the candidates were computed under {candidate_policy!r} while "
               f"the platform's bytes were cut under {resolved!r}: that is a "
               f"broken comparison whatever verdict it produced — an arm as "
               f"much as an ambiguity — and the 54/48/125-word NEITHER of "
               f"2026-08-11 is this same defect wearing a different answer")
    if verdict["refusals"]:
        return verdict

    if codes:
        if len(set(codes)) != 1:
            refuse(f"the discriminating patterns disagree ({verdict['discriminating']}): "
                   f"one constexpr cannot represent a platform whose orientations "
                   f"take different arms, and such a platform earns a new arm")
            return verdict
        verdict["expansion"] = codes[0]
        verdict["arm"] = next(k for k, v in EXPANSIONS.items() if v == codes[0])
        verdict["basis"] = "measured"
        return verdict

    row = environment_default(verdict["environment"])
    if row is None:
        refuse(f"no pattern discriminates under {verdict['policy_resolved']!r} "
               f"(all of {tuple(patterns_required)} classified {AMBIGUOUS_BOTH}) and this "
               f"execution environment matches no measured row of "
               f"ENVIRONMENT_DEFAULTS on {ENVIRONMENT_DEFAULT_KEYS} "
               f"(environment={verdict['environment']!r}): an unmeasured "
               f"platform is exactly where a default would be a guess — measure "
               f"the arm under a subnormal-KEEPING policy and add the row with "
               f"its artifact")
        return verdict
    verdict["environment_default"] = row
    verdict["expansion"] = EXPANSIONS[row["expansion"]]
    verdict["arm"] = row["expansion"]
    verdict["basis"] = "environment_default"
    verdict["why_arbitrary"] = (
        f"every pattern classified {AMBIGUOUS_BOTH} under "
        f"{verdict['policy_resolved']!r}: this run did NOT discriminate the arm. "
        f"{row['expansion']} is the arm already measured for this execution "
        f"environment ({row['measured_under_policy']}, {row['artifact']}), and "
        f"because both licensable arms reproduce the platform's bytes on every "
        f"probe vector here, the choice cannot change the words emitted ON THOSE "
        f"VECTORS. It is a default from a prior measurement, not a measurement "
        f"of this run — and 'on those vectors' is the whole of its scope: it "
        f"says nothing about an input class the probe does not sample.")
    return verdict


def expansion_from_probe(record: Any,
                         probe_patterns: Optional[Sequence[str]] = None
                         ) -> Optional[int]:
    """The single ``EXPANSION`` constexpr a probe record licenses, or None.

    The narrow answer :func:`expansion_license` computes; see there for the rule.
    Kept as the callers' entry point because a plan builder wants the constexpr
    or a refusal, not a verdict document. ``probe_patterns`` is forwarded, so a
    tranche that adds an operand orientation asks the same question over its own
    superset.
    """
    return expansion_license(record, probe_patterns)["expansion"]


def _resolve_expansion(probe: Any = None) -> Optional[int]:
    """Probe record (or the environment's) -> constexpr code, None on refusal.

    Refuses whenever :func:`_expansion_reasons` does, so the constexpr a plan
    binds and the reasons a coverage report prints can never disagree — a
    licence refused at the coverage seam but still bound at the plan seam is
    exactly how an unlicensed arm reaches a kernel.
    """
    if _expansion_reasons(probe):
        return None
    record = probe if probe is not None else load_expansion_probe()
    if record is None:
        return None
    return expansion_from_probe(record)


def _expansion_reasons(probe: Any = None) -> List[str]:
    """Clause 13: the EXPANSION binding must come from a measured artifact CUT
    UNDER THE POLICY IN FORCE.

    The policy half is not decoration. :data:`POLICY_CONDITIONAL_LICENCE` has
    always stated that a licence obtained under one subnormal policy does not
    transfer to the other; until 2026-08-15 nothing enforced it anywhere in the
    library or on the dispatch path, and the same artifact was licensed
    identically whichever policy the reading process ran under. See
    :func:`expansion_policy_reasons`.

    THE THREE STATES ``probe`` CAN BE IN, and they are three and not two:

    * a record — judge it;
    * ``None`` — NOTHING WAS OFFERED. Reading the environment is legitimate here,
      and if that finds nothing this refuses by name for absence;
    * a :class:`~meep_gpu.expansion_refusal.RefusedExpansionProbe` — SOMETHING WAS
      OFFERED AND REFUSED. It refuses by ITS OWN reasons, and the environment is
      never consulted, because the artifact there is the one that was refused.

    Reporting a refused artifact as an absent one would be the original defect
    told backwards: the audit would say no artifact was available on a run where
    one was available, judged and rejected.

    The policy half is applied against :func:`_policy_in_force`, which fails
    closed. ``fastpath`` also passes its own required policy to
    :func:`expansion_policy_reasons` explicitly at its own rung, four rungs before
    it installs anything, and declares it for the rungs below.
    """
    record = probe if probe is not None else load_expansion_probe()
    from ..expansion_refusal import RefusedExpansionProbe  # noqa: PLC0415
    if isinstance(record, RefusedExpansionProbe):
        return [f"the expansion probe artifact offered through {record.key} was "
                f"REFUSED by this dispatch and may not license an arm at any "
                f"later rung: " + "; ".join(record.reasons)]
    if record is None:
        return [
            "no complex-multiply expansion probe artifact is available for this "
            f"backend (set {PROBE_PATH_ENVIRONMENT} or pass probe=); the EXPANSION "
            "constexpr is a measured platform fact and may not be guessed"]
    verdict = expansion_license(record)
    if verdict["expansion"] is None:
        return [
            "the expansion probe artifact is missing, ambiguous or not for the "
            f"cupy backend (needs backend='cupy' and agreeing {PROBE_PATTERNS}); "
            "refused by name: " + "; ".join(verdict["refusals"])]
    # BOTH policy questions, resolved against ONE reading of what is in force.
    # Artifact-vs-run first so its reason stays the leading one where a caller
    # reports the head of this list; kernel-vs-run after — see
    # expansion_certification_reasons for why one does not imply the other.
    policy = _policy_in_force()
    return (expansion_policy_reasons(record, policy)
            + expansion_certification_reasons(policy))


# ---------------------------------------------------------------------------
# Coverage — positive refusal enumeration
# ---------------------------------------------------------------------------
#
# The clause numbering deliberately mirrors ``coverage._grid_reasons`` so the two
# can be diffed. Clause 2 is INVERTED (this product REQUIRES complex64 storage,
# where every shipped kernel requires real) and clause 7 (k = 0) is REPLACED by
# the per-axis phase-consistency clause: has_bloch is ADMITTED, and with no phase
# anywhere the kernel must reduce to the plain complex path (PH* = 0), which the
# gate's unphased rows measure. ``coverage._grid_reasons``/``pml_curl_coverage``/
# ``constitutive_coverage`` are NOT reusable here — their clauses 2 and 7 refuse
# this module's whole domain — and ``_layout_reasons``/``_volume_reasons`` pin
# float32, so the complex twins below exist rather than a widened shared check.


def _complex_volume_reasons(label: str, array: Any, shape: Sequence[int]) -> List[str]:
    """Shape, dtype and contiguity of one COMPLEX volume, reported by name."""
    out: List[str] = []
    if tuple(getattr(array, "shape", ())) != tuple(shape):
        out.append(f"{label} shape {tuple(getattr(array, 'shape', ()))!r} != grid shape "
                   f"{tuple(shape)!r}")
    if str(getattr(array, "dtype", None)) != "complex64":
        out.append(f"{label} dtype {getattr(array, 'dtype', None)} is not complex64 "
                   f"(this product steps complex storage as float32 word pairs)")
    flags = getattr(array, "flags", None)
    if not bool(getattr(flags, "c_contiguous", False)):
        out.append(f"{label} is not C-contiguous")
    return out


def _complex_layout_reasons(fields: Any, shape: Sequence[int],
                            names: Sequence[str]) -> List[str]:
    """Every named field volume must be complex64, C-contiguous, ``grid.shape``.

    The int32 bound is HALVED against the real path's (coverage.py:622-626):
    word offsets are ``2*idx`` in int32, so ``2*ncells`` is what must stay below
    ``2**31``, not ``ncells``.
    """
    out: List[str] = []
    if len(shape) != 3:
        return [f"grid shape {tuple(shape)!r} is not three-dimensional"]
    total = int(shape[0]) * int(shape[1]) * int(shape[2])
    if 2 * total >= 2 ** 31:
        out.append(f"{total} cells is {2 * total} float32 words, which exceeds the "
                   f"kernel's int32 word-offset range (2*ncells must stay below 2**31)")
    for name in names:
        array = getattr(fields, name, None)
        if array is None:
            continue  # Already reported by the allocation clause.
        out.extend(_complex_volume_reasons(name, array, shape))
    return out


def _complex_grid_reasons(fields: Any, pml: Any, grid: Any,
                          probe: Any = None, *,
                          require_active_pml: bool = True) -> List[str]:
    """The clauses the complex predicates share.

    ``require_active_pml`` selects which SIDE of clause 3 this caller wants, and
    that clause alone: True demands an absorber that absorbs (the split-field
    product in this module), False demands its absence (the degenerate
    no-absorber product in :mod:`complex_no_pml_curl`). The two are a PARTITION
    of ``stepping._pml_is_active``, which is the same test the array path
    branches on at stepping.py:508 — so no configuration can be admitted by both
    and none can fall between them, and the loss mode a second admitter would
    cause (an UNSELECTED slot falling silently back to the array path) cannot
    arise. Every other clause is shared verbatim, deliberately: a duplicated
    copy of clause 5's phase logic is two things that must be kept in step
    forever, and the 2026-08-11 coverage audit is what that costs.
    """
    reasons: List[str] = []

    # 1. CuPy backend. The kernels launch against device pointers.
    xp = getattr(grid, "xp", None)
    if getattr(xp, "__name__", "") != "cupy":
        reasons.append(f"array module is {getattr(xp, '__name__', xp)!r}, not cupy")

    # 2 (INVERTED). Complex64 storage REQUIRED. A real run belongs to the shipped
    #    real-field kernels; real storage read as float32 word pairs of a complex
    #    array is the wrong-stride wrong answer in the other direction. Either
    #    force_complex_fields or a nonzero k_point puts the run here (the array
    #    path itself raises on real storage with a phase, stepping.py:1893-1908);
    #    the storage dtype itself is verified by the layout clause.
    if not (getattr(fields, "force_complex_fields", False)
            or getattr(grid, "has_bloch", False)):
        reasons.append(
            "storage is real float32 (neither force_complex_fields nor a nonzero "
            "k_point): a real run belongs to the shipped real-field kernels")

    # 3. An absorber that actually absorbs: this is the split-field product only.
    #    A plain complex curl (pml=None) is a separate product and is refused.
    #    UNDER ``require_active_pml=False`` this clause is INVERTED and nothing
    #    else changes — see the docstring for why that is a partition.
    active = pml is not None and bool(getattr(pml, "is_active", False))
    if require_active_pml:
        if not active:
            reasons.append("no active PML layer (this product implements the complex "
                           "split-field path only)")
    elif active:
        reasons.append("an active PML layer is installed (that is the complex "
                       "split-field product's path, not this one)")

    # 4. Only the two ghost rules the curl kernel writes; a fold or the
    #    cylindrical axis changes the stored extent for the constitutive side too.
    kinds = _boundary_kinds(grid, pml if (pml is not None
                                          and getattr(pml, "is_active", False))
                            else None)
    if kinds is None:
        reasons.append("boundary kinds could not be resolved for this grid")
    else:
        for axis, kind in enumerate(kinds):
            if kind not in COVERED_BOUNDARIES:
                reasons.append(f"axis {axis} boundary {kind!r} is outside {COVERED_BOUNDARIES}")
    if _call(grid, "has_symmetry", default=False):
        reasons.append("a mirror plane is active (symmetry folding is not carried)")
    for axis in range(3):
        if _call(grid, "is_mirrored", axis, default=False):
            reasons.append(f"axis {axis} is folded by a mirror plane")
    if getattr(grid, "cylindrical", False):
        reasons.append("cylindrical (Dcyl) coordinates are not carried")
    for axis in range(3):
        if _call(grid, "is_axis", axis, default=False):
            reasons.append(f"axis {axis} is the cylindrical r = 0 axis")

    # 5. Per-axis phase consistency. A phased axis must resolve PERIODIC
    #    (stepping._bloch_phases raises there, stepping.py:2401-2415; the grid
    #    guarantees it, grid.py:973-1001 — re-checked because a mis-bound PH
    #    constexpr is a plane of wrong values, not a crash), and a metallic axis
    #    must carry a k component of exactly 0. has_bloch FALSE IS ADMITTED: with
    #    PH*=0 the kernel must reduce to the plain complex path, and the gate's
    #    unphased rows are that reduction measured. The phase is read
    #    DEFENSIVELY, not through ``_call``: an unreadable phase is not an
    #    unphased one (the rule _susceptibility_reasons applies to driven(),
    #    coverage.py:224-227), and admitting one would let ``bloch_phase_table``
    #    raise into the plan builder, violating its None-only refusal contract
    #    (demonstrated by the 2026-08-11 coverage audit).
    k_point = tuple(getattr(grid, "k_point", (0.0, 0.0, 0.0)))
    if kinds is not None:
        phase_reader = getattr(grid, "bloch_phase", None)
        if not callable(phase_reader):
            reasons.append(
                "grid.bloch_phase is missing or not callable; an unreadable "
                "phase table is not an unphased one")
        for axis, kind in enumerate(kinds):
            if callable(phase_reader):
                try:
                    phase = phase_reader(axis)
                except Exception as exc:  # noqa: BLE001 - unreadable is refused
                    reasons.append(
                        f"grid.bloch_phase({axis}) raised {exc!r}; an unreadable "
                        f"phase is not an unphased one")
                else:
                    if phase is not None and kind != "periodic":
                        reasons.append(
                            f"axis {axis} carries Bloch phase {phase!r} but "
                            f"resolved to {kind!r}; only a periodic wrap can "
                            f"carry a phase")
            if kind == "metallic" and float(k_point[axis]) != 0.0:
                reasons.append(
                    f"axis {axis} is metallic with k component {k_point[axis]!r}; "
                    f"a PEC wall gives the axis no lattice vector for the phase "
                    f"(stepping._bloch_phases raises on the same pairing)")

    # 6. No conductivity anywhere on the curl targets: conductive complex is a
    #    separate product (stepping.py:2001-2109), with no corpus demand. A
    #    missing or non-callable reader is refused OUTRIGHT — deliberately
    #    stricter than the shipped predicate's magnetic-flag fallback
    #    (coverage.py:275-291): inferring "no conductivity" from the ABSENCE of
    #    condfac_for is admission by attribute absence, the exact reasoning
    #    coverage exists to refuse, and an electric conductivity has no fallback
    #    flag at all (demonstrated by the 2026-08-11 coverage audit; the real
    #    Fields always defines the reader, fields.py:832, so no engine-built
    #    configuration changes verdict).
    reader = getattr(fields, "condfac_for", None)
    if not callable(reader):
        reasons.append(
            "fields does not expose condfac_for; an unreadable conductivity "
            "table is not an absent one")
    else:
        for target in CURL_TARGETS:
            try:
                conductive = reader(target) is not None
            except Exception as exc:  # noqa: BLE001 - unreadable means not covered
                reasons.append(f"condfac_for({target!r}) raised {exc!r}")
                continue
            if conductive:
                reasons.append(
                    f"a conductivity is installed on {target}; conductive complex "
                    f"stepping is a separate, unbuilt product")

    # 7. No dispersion, refused OUTRIGHT: none of the nine demand scripts carries
    #    a susceptibility, and complex ADE belongs to a future tranche. The shape
    #    clauses still run so an unreadable susceptibility is named, not shrugged at.
    if getattr(fields, "has_polarizations", False) or (
            getattr(fields, "polarizations", ()) or ()):
        reasons.append(
            "a susceptibility is registered: complex-storage ADE is a future "
            "tranche, and no corpus script combines dispersion with complex fields")
    reasons.extend(_susceptibility_reasons(fields))

    # 8. No instantaneous nonlinearity (Pade factor + the DOCMP split,
    #    stepping.py:999-1000, :1080-1116), no BFAST (:392-396, complex cast
    #    :923-924), no special_kz beta (:754-811 — refl-angular-kz2d and
    #    parallel-wvgs-force are grid.beta, NOT bloch_phase, grid.py:378-383).
    if getattr(fields, "has_nonlinearity", False):
        reasons.append("chi2/chi3 is installed (the Pade constitutive factor is not carried)")
    if getattr(grid, "bfast_active", False):
        reasons.append("BFAST is active (the second additive curl term is not carried)")
    if float(getattr(grid, "beta", 0.0)) != 0.0:
        reasons.append(f"special_kz beta={getattr(grid, 'beta', 0.0)!r} is nonzero "
                       f"(that is grid.beta, not a Bloch phase)")

    # 9. Stored E. Under an active PML always true (fields.py:678-700), but the
    #    clause is the REASON: an edit that makes stores_E optional under PML
    #    must be caught here, exactly as coverage.py:296-304 argues. Under
    #    ``require_active_pml=False`` it is NOT automatic and does real work: with
    #    E unstored, stepping._read_component derives D * inv_eps into a scratch
    #    buffer (stepping.py:2440-2449) instead of differencing storage, which is
    #    no_pml.py's DERIVE product and is not transcribed here.
    if not getattr(fields, "stores_E", False):
        reasons.append("E is recomputed from D rather than stored")

    # 13. EXPANSION binding requires a platform probe artifact (module docstring).
    reasons.extend(_expansion_reasons(probe))

    return reasons


def complex_pml_curl_coverage(fields: Any, pml: Any, sub_step: str,
                              probe: Any = None) -> Coverage:
    """May the complex split-field PML curl kernel step this (fields, pml, sub_step)?

    Positive clauses only; a failing clause appends its reason and the scan
    continues. Off-diagonal epsilon is ADMITTED here (constitutive-only, its
    whole effect is inside ``update_E``, stepping.py:1001-1008) and refused by the
    E-side constitutive predicate — the same per-sub-step split coverage.py makes.
    """
    if sub_step not in CURL_SUB_STEPS:
        raise ValueError(
            f"sub_step must be one of {tuple(CURL_SUB_STEPS)}, got {sub_step!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    reasons = _complex_grid_reasons(fields, pml, grid, probe)

    spec = SUB_STEPS[sub_step]
    names = (tuple(spec["targets"])
             + tuple("fu_" + name for name in spec["targets"])
             + tuple(spec["sources"]))
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")

    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(_complex_layout_reasons(fields, shape, names))
    if pml is not None and getattr(pml, "is_active", False) and len(shape) == 3:
        # Both sub-lattices, as pml_curl_coverage checks (coverage.py:316-317):
        # the suffix the plan binds is the sub-step's own, and a swap is the
        # half-cell mutation the gate carries.
        reasons.extend(_coefficient_reasons(pml, shape, ("kms", "sinv"), ("", "_h")))

    return Coverage(not reasons, tuple(reasons))


def complex_constitutive_coverage(fields: Any, pml: Any, side: str,
                                  probe: Any = None) -> Coverage:
    """May the complex constitutive kernel step ``update_H`` ('H') / ``update_E`` ('E')?

    The E side additionally refuses everything that changes what ``source`` is:
    a registered polarization is already refused for the whole module (clause 7),
    and an off-diagonal chi1inv row makes the sub-step non-element-wise
    (stepping.py:1001-1008), so it is refused HERE and admitted by the curl.
    With no polarization, ``displacement_minus_polarization`` returns the D
    array itself (fields.py:1096-1098), so binding D directly IS the array
    path's source.
    """
    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, got {side!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    spec = CONSTITUTIVE_SIDES[side]
    reasons = _complex_grid_reasons(fields, pml, grid, probe)

    if side == "E" and getattr(fields, "has_offdiagonal_epsilon", False):
        reasons.append(
            "an off-diagonal chi1inv row is installed (the row product reads "
            "neighbours; this sub-step is element-wise)")

    names = tuple(spec["targets"]) + tuple(spec["aux"]) + tuple(spec["sources"])
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")

    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(_complex_layout_reasons(fields, shape, names))

    if side == "E" and len(shape) == 3:
        # inv_eps stays float32 under complex storage (stepping.py:37-38,
        # fields.py:1203-1204), so coverage.py's own float32 pin is exactly right.
        reasons.extend(_inverse_epsilon_reasons(fields, shape))

    if pml is not None and getattr(pml, "is_active", False) and len(shape) == 3:
        suffix = ("_h",) if spec["half_integer"] else ("",)
        reasons.extend(_coefficient_reasons(pml, shape, ("kps", "kms"), suffix))

    return Coverage(not reasons, tuple(reasons))


# ---------------------------------------------------------------------------
# The phase table and its argument encoding
# ---------------------------------------------------------------------------

def bloch_phase_table(grid: Any, kinds: Sequence[str]) -> Tuple[Optional[complex],
                                                                Optional[complex],
                                                                Optional[complex]]:
    """Per-axis Bloch wrap factor, None on an unphased axis — never ``1+0j``.

    Resolved through ``grid.bloch_phase`` exactly as ``stepping._bloch_phases``
    resolves it (stepping.py:2398-2416): the grid value is exp(2*pi*i*k*L) with
    the Brillouin edge EXACTLY -1+0j (grid.py:1011-1017), and None means the
    multiply is SKIPPED — the constexpr skip is the bit-identity of k = 0.
    The non-periodic check mirrors the array path's raise and is unreachable
    behind an admitting predicate; it is kept so a caller who skips the
    predicate is refused loudly rather than handed a mis-bound PH constexpr.
    """
    if not getattr(grid, "has_bloch", False):
        return (None, None, None)
    phases: List[Optional[complex]] = []
    for axis in range(3):
        phase = grid.bloch_phase(axis)
        if phase is not None and kinds[axis] != "periodic":
            raise ValueError(
                f"axis {axis} carries Bloch phase {phase!r} but resolved to "
                f"{kinds[axis]!r}; only a periodic wrap can carry a phase "
                f"(stepping._bloch_phases raises on the same configuration)")
        phases.append(phase)
    return (phases[0], phases[1], phases[2])


def _phase_arguments(phases: Sequence[Optional[complex]],
                     backward: bool) -> Tuple[Tuple[int, int, int],
                                              Tuple[float, ...]]:
    """Encode the phase table as (PH flags, six float32 components).

    The phase is rounded to complex64 BEFORE splitting — the array path's
    ``shifted.dtype.type(phase)`` (stepping.py:1909) — and for the backward
    sub-step the imaginary part is NEGATED, which is the conjugate of
    ``_shift_down`` (stepping.py:1865-1869); negation is exact, so the order of
    conjugation and rounding is immaterial. An unphased axis passes (1.0, 0.0)
    under PH = 0, and the kernel emits no multiply for it at all.
    """
    import numpy  # noqa: PLC0415
    flags: List[int] = []
    values: List[float] = []
    for phase in phases:
        if phase is None:
            flags.append(0)
            values.extend((1.0, 0.0))
            continue
        rounded = numpy.complex64(phase)
        real = float(numpy.float32(rounded.real))
        imag = float(numpy.float32(rounded.imag))
        if backward:
            imag = -imag
        flags.append(1)
        values.extend((real, imag))
    return (flags[0], flags[1], flags[2]), tuple(values)


def _word_view(array: Any) -> Any:
    """The float32 word view of one complex64 volume — the pointer the kernel gets.

    ``complex64`` is bit-layout (re, im) interleaved, so a C-contiguous complex
    volume viewed as float32 is the SAME allocation with a doubled last axis;
    no copy, no move, and the engine's own references stay live. Anything that
    is not a C-contiguous complex64 volume is refused at plan time — a strided
    view would be read in the wrong order by the flat word index.
    """
    import numpy  # noqa: PLC0415
    if str(getattr(array, "dtype", None)) != "complex64":
        raise ValueError(f"expected a complex64 volume, got dtype "
                         f"{getattr(array, 'dtype', None)!r}")
    if not bool(getattr(getattr(array, "flags", None), "c_contiguous", False)):
        raise ValueError("complex volume is not C-contiguous; its float32 word "
                         "view would not be either")
    return array.view(numpy.float32)


# ---------------------------------------------------------------------------
# The plans
# ---------------------------------------------------------------------------

class ComplexPmlCurlPlan:
    """A launchable, allocation-free COMPLEX split-field PML curl sub-step.

    Same two construction routes and one launch route as ``launch.PmlCurlPlan``:
    :func:`plan_complex_pml_curl` from the engine's objects through the
    predicate, :func:`plan_complex_pml_curl_from_arrays` from bare device arrays
    for the gate (including the deliberately wrong ones), both launched through
    :meth:`run` so the bytes the gate certifies are the bytes the engine would
    launch. The complex volumes are bound as float32 WORD VIEWS here, once, at
    plan time; ``n_elem`` stays the COMPLEX cell count.
    """

    __slots__ = ("sub_step", "shape", "n_elem", "dtdx", "backward", "bc",
                 "phased", "phase_values", "expansion", "block", "num_warps",
                 "_targets", "_aux", "_sources", "_coefficients", "_grid",
                 "_kernel")

    def __init__(self, sub_step: str, shape, dtdx: float, bc, phased,
                 phase_values, expansion: int, block: int,
                 targets, auxiliaries, sources, coefficients, kernel=None,
                 num_warps: Optional[int] = None) -> None:
        if sub_step not in SUB_STEPS:
            raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
        self.sub_step = sub_step
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        # float(): the array path consumes the Python scalar at complex64
        # precision, i.e. the float32-rounded scalar on each plane (measured:
        # python-float * c8 == plane-wise with the f32-rounded scalar); Triton
        # types a Python float argument as fp32, so the two are the same bits.
        self.dtdx = float(dtdx)
        self.backward = int(SUB_STEPS[sub_step]["backward"])
        self.bc = tuple(int(code) for code in bc)
        self.phased = tuple(int(flag) for flag in phased)
        self.phase_values = tuple(float(value) for value in phase_values)
        self.expansion = int(expansion)
        self.block = int(block)
        self.num_warps = None if num_warps is None else int(num_warps)
        self._targets = tuple(CupyPointer(_word_view(a)) for a in targets)
        self._aux = tuple(CupyPointer(_word_view(a)) for a in auxiliaries)
        self._sources = tuple(CupyPointer(_word_view(a)) for a in sources)
        self._coefficients = tuple(CupyPointer(_flat(a)) for a in coefficients)
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        # The override exists for exactly one caller: the gate's mutation legs,
        # which compile deliberately broken copies of the shipped kernel.
        self._kernel = kernel

    def run(self, guard: Optional[bool] = None) -> None:
        """Launch the sub-step, in place. ``guard`` is the gate's, not a caller's."""
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        nx, ny, nz = self.shape
        kernel = self._kernel if self._kernel is not None else bloch_pml_curl_step
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        kernel[self._grid](
            *self._targets, *self._aux, *self._sources, *self._coefficients,
            nx, ny, nz, self.n_elem, self.dtdx,
            *self.phase_values,
            BACKWARD=self.backward,
            BCX=self.bc[0], BCY=self.bc[1], BCZ=self.bc[2],
            PHX=self.phased[0], PHY=self.phased[1], PHZ=self.phased[2],
            EXPANSION=self.expansion,
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )

    def __repr__(self) -> str:
        return (f"ComplexPmlCurlPlan({self.sub_step}, shape={self.shape}, "
                f"bc={self.bc}, phased={self.phased}, expansion={self.expansion}, "
                f"block={self.block}, num_warps={self.num_warps})")


def plan_complex_pml_curl(fields: Any, pml: Any, sub_step: str,
                          block: Optional[int] = None,
                          num_warps: Optional[int] = None,
                          probe: Any = None) -> Optional[ComplexPmlCurlPlan]:
    """Build a plan from the engine's own objects, or None when out of coverage.

    None is the only refusal (Y-style: never raise into a caller that would
    otherwise have stepped correctly). ``probe`` overrides the environment's
    probe artifact; without a usable one the predicate has already refused.
    """
    if sub_step not in SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
    if not complex_pml_curl_coverage(fields, pml, sub_step, probe=probe).covered:
        return None
    expansion = _resolve_expansion(probe)
    if expansion is None:  # pragma: no cover - the predicate already refused
        return None
    from ..stepping import _boundary_kinds as resolve  # noqa: PLC0415

    spec = SUB_STEPS[sub_step]
    grid = fields.grid
    kinds = resolve(grid, pml)
    phases = bloch_phase_table(grid, kinds)
    phased, values = _phase_arguments(phases, backward=bool(spec["backward"]))
    return ComplexPmlCurlPlan(
        sub_step, grid.shape, grid.dt / grid.dx,
        [1 if kind == "metallic" else 0 for kind in kinds],
        phased, values, expansion,
        DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in spec["targets"]],
        [getattr(fields, "fu_" + name) for name in spec["targets"]],
        [getattr(fields, name) for name in spec["sources"]],
        [getattr(pml, f"{stem}_{axis}{spec['suffix']}")
         for axis in "xyz" for stem in ("kms", "sinv")],
        num_warps=num_warps,
    )


def plan_complex_pml_curl_from_arrays(sub_step: str, arrays: Dict[str, Any],
                                      flat: Dict[str, Any], codes,
                                      phases: Sequence[Optional[complex]],
                                      dtdx: float, expansion: int,
                                      block: Optional[int] = None,
                                      kernel: Any = None,
                                      num_warps: Optional[int] = None,
                                      ) -> ComplexPmlCurlPlan:
    """Build a plan from bare device arrays — the gate's and benchmark's route.

    ``arrays`` is keyed by component name (complex64 volumes), ``flat`` by
    ``kms_x``/``sinv_x``/... on the sub-lattice the caller already selected,
    ``codes`` is the 0/1 periodic/metallic triple, ``phases`` the per-axis
    Optional[complex] table (None = unphased; the conjugation for step_D is
    applied HERE, per sub-step, exactly as the engine route applies it), and
    ``expansion`` the probe-measured constexpr. No predicate runs: the caller is
    a harness that constructed the configuration deliberately, including the
    deliberately wrong ones, and ``kernel=`` carries the mutation override.
    """
    spec = SUB_STEPS[sub_step]
    shape = tuple(int(n) for n in arrays[spec["targets"][0]].shape)
    phased, values = _phase_arguments(tuple(phases), backward=bool(spec["backward"]))
    return ComplexPmlCurlPlan(
        sub_step, shape, dtdx, codes, phased, values, int(expansion),
        DEFAULT_BLOCK if block is None else block,
        [arrays[name] for name in spec["targets"]],
        [arrays["fu_" + name] for name in spec["targets"]],
        [arrays[name] for name in spec["sources"]],
        [flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
        kernel=kernel, num_warps=num_warps,
    )


class ComplexConstitutivePlan:
    """A launchable, allocation-free COMPLEX ``dsigw`` constitutive sub-step.

    The Yee sub-lattice is chosen at build time and nowhere else — integer
    (``kps_a``/``kms_a``) for H, half-integer (``_h``) for E (stepping.py:948
    against :1015) — and the kernel never asks which lattice its six pointers
    came from, so the swap stays the gate's half-cell mutation. The H side binds
    the source word views as inverse-epsilon placeholders, the same device as
    ``launch.ConstitutivePlan`` (launch.py:438-442): the loads sit behind the
    ``SCALE`` constexpr and are compiled away, but a pointer argument still has
    to type.
    """

    __slots__ = ("side", "shape", "n_elem", "scale", "expansion", "block",
                 "num_warps", "_targets", "_aux", "_sources", "_inv_eps",
                 "_coefficients", "_grid", "_kernel")

    def __init__(self, side: str, shape, expansion: int, block: int,
                 targets, auxiliaries, sources, inverse_epsilon, coefficients,
                 kernel=None, num_warps: Optional[int] = None) -> None:
        if side not in CONSTITUTIVE_SIDES:
            raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, got {side!r}")
        self.side = side
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.scale = 1 if side == "E" else 0
        self.expansion = int(expansion)
        self.block = int(block)
        self.num_warps = None if num_warps is None else int(num_warps)
        self._targets = tuple(CupyPointer(_word_view(a)) for a in targets)
        self._aux = tuple(CupyPointer(_word_view(a)) for a in auxiliaries)
        self._sources = tuple(CupyPointer(_word_view(a)) for a in sources)
        # inv_eps volumes stay float32 (fields.py:1203-1204) and are indexed by
        # the COMPLEX cell index in-kernel — bound directly, never word-viewed.
        self._inv_eps = (tuple(CupyPointer(a) for a in inverse_epsilon)
                         if inverse_epsilon is not None else self._sources)
        self._coefficients = tuple(CupyPointer(_flat(a)) for a in coefficients)
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        self._kernel = kernel

    def run(self, guard: Optional[bool] = None) -> None:
        """Launch the sub-step, in place. Same ``guard`` contract as the curl plan."""
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        nx, ny, nz = self.shape
        kernel = self._kernel if self._kernel is not None else bloch_constitutive_step
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        kernel[self._grid](
            *self._targets, *self._aux, *self._sources, *self._inv_eps,
            *self._coefficients,
            nx, ny, nz, self.n_elem,
            SCALE=self.scale,
            EXPANSION=self.expansion,
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )

    def __repr__(self) -> str:
        return (f"ComplexConstitutivePlan({self.side}, shape={self.shape}, "
                f"expansion={self.expansion}, block={self.block}, "
                f"num_warps={self.num_warps})")


def plan_complex_constitutive(fields: Any, pml: Any, side: str,
                              block: Optional[int] = None,
                              num_warps: Optional[int] = None,
                              probe: Any = None) -> Optional[ComplexConstitutivePlan]:
    """Build a complex constitutive plan from the engine's objects, or None.

    With no polarization registered (clause 7 refused any), the E source IS the
    D array — ``displacement_minus_polarization`` returns it unchanged
    (fields.py:1096-1098) — so the binding below is the array path's source, not
    an optimisation of it.
    """
    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, got {side!r}")
    if not complex_constitutive_coverage(fields, pml, side, probe=probe).covered:
        return None
    expansion = _resolve_expansion(probe)
    if expansion is None:  # pragma: no cover - the predicate already refused
        return None

    spec = CONSTITUTIVE_SIDES[side]
    suffix = "_h" if spec["half_integer"] else ""
    return ComplexConstitutivePlan(
        side, fields.grid.shape, expansion,
        DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in spec["targets"]],
        [getattr(fields, name) for name in spec["aux"]],
        [getattr(fields, name) for name in spec["sources"]],
        ([fields.inverse_epsilon_for(name) for name in spec["targets"]]
         if side == "E" else None),
        [getattr(pml, f"{stem}_{axis}{suffix}")
         for axis in "xyz" for stem in ("kps", "kms")],
        num_warps=num_warps,
    )


def plan_complex_constitutive_from_arrays(side: str, arrays: Dict[str, Any],
                                          flat: Dict[str, Any], expansion: int,
                                          block: Optional[int] = None,
                                          kernel: Any = None,
                                          num_warps: Optional[int] = None,
                                          ) -> ComplexConstitutivePlan:
    """Build a complex constitutive plan from bare device arrays — the gate's route.

    ``arrays`` is keyed by component name (plus ``inv_eps_Ex``... on the E side,
    float32), ``flat`` by ``kps_x``/``kms_x``/... on the sub-lattice THE CALLER
    selected, and ``expansion`` is the probe-measured constexpr. ``kernel=``
    carries the mutation override; dropping it silently disarms every mutation
    leg (launch.py:516-522's measured lesson).
    """
    spec = CONSTITUTIVE_SIDES[side]
    shape = tuple(int(n) for n in arrays[spec["targets"][0]].shape)
    return ComplexConstitutivePlan(
        side, shape, int(expansion),
        DEFAULT_BLOCK if block is None else block,
        [arrays[name] for name in spec["targets"]],
        [arrays[name] for name in spec["aux"]],
        [arrays[name] for name in spec["sources"]],
        ([arrays["inv_eps_" + name] for name in spec["targets"]]
         if side == "E" else None),
        [flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kps", "kms")],
        kernel=kernel, num_warps=num_warps,
    )


# ---------------------------------------------------------------------------
# WIRING — live, and what still holds the line
# ---------------------------------------------------------------------------
#
# ``launch.plan_step`` consults these predicates through its own forwarders
# (launch.py:232-257 import them lazily; the four complex arms are selected at
# :2045 step_B/step_D, :2147 update_H, :2223 update_E), and ``fastpath._decide``
# resolves the probe artifact at fastpath.py:1701 and forwards it into that
# composer at :1722. ``fastpath.ARM_CERTIFICATION`` carries 'complex PML' and
# 'complex', so these are dispatchable slots.
#
# THIS BLOCK SAID "WIRING — none, deliberately" UNTIL 2026-08-15 and was
# measurably stale (``plan_step`` on a complex64 triple emits 'complex PML: '
# reasons, i.e. it called the predicate). The old text also carried the argument
# that ``coverage._grid_reasons`` clauses 2 and 7 refuse every complex/Bloch run
# so an admitted-overlap ambiguity could not arise; that argument retired with
# the wiring, and what prevents an unlicensed arm from reaching a kernel now is
# the refusal enumeration above — clause 13 (a measured artifact, cut under the
# policy in force) and the licence rule in :func:`expansion_license`.
#
# Still true, and load-bearing for a Triton-less host: this module is NOT
# imported by the package ``__init__`` (which eagerly imports only ``coverage``),
# and launch's forwarders import it lazily inside the function body, so the
# import cost and the Triton dependency are paid only by a run that actually has
# complex storage. Tests and the gate import it directly.
