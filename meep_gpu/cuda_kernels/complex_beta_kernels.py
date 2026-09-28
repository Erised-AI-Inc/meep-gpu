"""``grid.beta`` (special_kz) under complex64 storage -- the hand-CUDA curl pair.

WHAT THIS FAMILY IS. A 2-D run carries ``exp(i*2*pi*beta*z)`` analytically, so
``d/dz`` on the invariant axis is the EXACT factor ``i*2*pi*beta`` -- an
``i*beta*zhat x`` cross product folded into the curl (MEEP step_db.cpp:148-176).
Under COMPLEX storage that ``i`` is paid explicitly, in the ``+-1j`` branch at
``stepping.py:798-799``. Both shipped beta predicates refuse complex storage BY
NAME for exactly that branch -- ``special_kz_curl._beta_reasons``: "the +-1j
coefficient branch (stepping.py:798-799) is the complex family's, not this pair's"
-- and the certified complex family refuses beta by name in return. This module is
the family both refusals name.

WHAT IT IS WORTH, MEASURED, AND WHY IT IS BUILT SECOND
=============================================================================

On the 2026-08-20 union census
(``parity/meep_gpu/results/cuda_predicate_coverage_2026-08-20_closeout/``) the
beta clause is the FIRST refusal on 16 unserved slots over 4 complex-storage rows.
TWELVE OF THOSE SIXTEEN ARE ALSO FOLDED, so a beta kernel with no fold branch
would buy 4 slots and not 16:

* ``tests_param/TestSpecialKz.test_special_kz`` -- 500x1x1, all-periodic, kx =
  0.9205, beta = -0.3907, NO fold. This is the row a fold-free beta kernel serves.
* ``tests_param/TestSpecialKz.test_eigsrc_kz__idx0`` -- 420x212x1, Mirror(Y) with
  a PERIODIC termination, beta = 0.2, k = 0.
* ``tests_param/TestEigCoeffs.test_binary_grating_special_kz__idx0`` and
  ``__idx1`` -- 135x92x1, Mirror(Y) periodic with an IN-PLANE kx on the PML'd X
  axis, beta = -0.685 / -0.912.

THE ORDER WAS CHOSEN FROM THAT ARITHMETIC AND IS STATED SO IT CAN BE CHECKED. The
fold was built FIRST (``complex_folded_kernels``), because its clause is first
refusal on 32 slots against beta's 16, and because 12 of beta's 16 need the fold
anyway. This family's device source is therefore the FOLDED complex curl with the
beta insert -- one pair of kernels covering folded and unfolded beta runs alike,
since the fold branches are runtime boundary codes and are simply dead on an
unfolded grid. Building beta first and folding second would have produced two
kernels that each served part of one row set and a composition nobody had
measured.

THE INSERT, TRANSCRIBED
=============================================================================

* CALL SITES -- ``step_B`` (stepping.py:384-391): Bx adds the Ey-CENTRE term at
  sign +1 (:389), By the Ex-CENTRE term at sign -1 (:391), Bz nothing; ``step_D``
  (:438-445): Dx <- Hy at +1 (:443), Dy <- Hx at -1 (:445), Dz nothing. ``cc``
  runs over ``d_c`` in {X, Y} only, which is why the third component gets none.
* COEFFICIENT (stepping.py:797-799) -- ``sign * 2*pi * grid.beta * grid.dt`` in
  float64, then multiplied by ``+1j`` on the MAGNETIC side and ``-1j`` on the
  electric one THROUGH PYTHON'S OWN COMPLEX ARITHMETIC. NO ``dtdx``: this is an
  analytic derivative, not a finite difference.
* ROUNDING (stepping.py:811) -- ``partner_values.dtype.type(coefficient)``, ONE
  rounding to complex64 on the host, then ``return -(c * partner)``.
* POSITION -- added to ``curl`` AFTER ``_curl_from_operands`` (:342 / :429) and
  BEFORE ``_mask_non_owned_cells`` (:369 / :450). Under a fold that mask has TWO
  arms and both of them reach a beta target, so an insert below the masks leaves a
  live increment on the mirror plane and on the far-ghost plane -- invisible in the
  interior. That is the gate's ``beta_after_mask`` needle.
* PARTNER -- the SAME-CELL component snapshot (:293 electric, :408 magnetic), which
  in every one of the six curl terms is a CENTRE operand the stencil already
  loaded: target 0 takes ``f_2`` and target 1 takes ``f_1``. No new pointer, no
  ghost rule, no fold parity, no wrap ever touches it.

THE SIGNED ZERO IN THE REAL WORD IS MEASURED, NOT SYNTHESISED
=============================================================================

``coefficient * 1j`` in Python is ``complex(coefficient, 0.0) * complex(0.0, 1.0)``
= ``(coefficient*0.0 - 0.0*1.0, coefficient*1.0 + 0.0*0.0)``, so the REAL word is
``coefficient * 0.0 - 0.0`` -- which is ``-0.0`` exactly when ``coefficient`` is
negative and the ``+1j`` (magnetic) branch is taken, and ``+0.0`` in every other
combination. MEASURED on this interpreter over the five corpus betas at two
courants (2026-08-20, ``numpy`` 2.x): of the 40 (sign, side) combinations, the ten
with ``coefficient < 0`` on the magnetic side carry ``0x80000000`` in the real
word and the other thirty carry ``0x00000000``. That asymmetry is why
:func:`beta_curl_coefficients` returns the word Python produced instead of
writing ``0.0f`` into the kernel: with the field's other word exactly zero the
cross term ``c_re * z`` carries the zero's SIGN into an addend, and
``fma(x, y, -0.0f)`` differs from ``fma(x, y, +0.0f)`` wherever ``x*y`` is exactly
zero -- which a thin absorber's deepest ``kms`` reaches from all-zero state. The
Triton sibling ``special_kz.beta_curl_coefficients`` passes the word through for
the same reason and a gate mutation builds it in-kernel as a literal.

THE PRODUCT IS COEFFICIENT-LEFT, AND THAT IS A NAME RATHER THAN AN ARITHMETIC
=============================================================================

``stepping.py:811`` spells ``partner_values.dtype.type(coefficient) *
partner_values`` -- the SCALAR on the left, so NumPy's and CuPy's inner loop takes
the coefficient as ``in1``. Under ``FMA_V1`` that decides which product is fused
and which is separately rounded, so the orientation is normative. This family
therefore emits ``mul_imag_coefficient_left`` as a one-line call into
``complex_emitter``'s own ``rotate_field_left``, whose body already fuses the FIRST
operand's product: "left" means "first operand" in every one of this family's
orientations, so the two are the same arithmetic and a second body would be a
second spelling of one rule -- with two arms to keep in step by hand.
``test_complex_beta.py`` pins the emitted helper to that one line so a future edit
cannot quietly grow a body.

THE TWO ARMS GO BLIND ON THIS PRODUCT, AND THAT IS A MEASUREMENT TOO. With the
coefficient's real word exactly ``+-0.0`` the fused and the separately-rounded
expansions agree bit for bit (``fma(+-0, z, p)`` rounds ``p`` once and adds a
signed zero, which is what the naive form computes), so the arm cannot be
discriminated HERE. It is still bound, because every OTHER multiply in the kernel
-- the phase rotation, the ``dtdx`` scalar product, the four PML products -- is the
certified complex family's and does discriminate. The predicate below inherits that
licence check rather than relaxing it.

WHAT IS NOT HERE. ``update_H`` / ``update_E`` read nothing beta-dependent: beta
enters the step loop at stepping.py:384-391 and :467-474 and nowhere else. So
:func:`covers_complex_beta_constitutive` is an ADMISSION over the CERTIFIED
complex constitutive pair, and ``gate_cuda_complex_beta.py`` carries a device arm
for it rather than letting "nothing was found to read beta" stand as evidence.

NO DISPATCH. Nothing in ``meep_gpu`` imports ``cuda_kernels``, so a True from the
predicates below licenses a MEASUREMENT and not a production step.

NOTHING HERE IMPORTS CUPY OR NUMPY AT MODULE SCOPE.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Sequence, Tuple

from . import complex_emitter
from . import complex_folded_kernels as _folded
from . import coverage as _coverage

__all__ = (
    "BETA_COMPLEX_ADMISSION",
    "BETA_DELTAS",
    "BETA_KERNELS",
    "CERTIFIED_KERNELS",
    "UNCERTIFIED_KERNELS",
    "beta_curl_coefficients",
    "beta_source",
    "corpus_digest",
    "covers_complex_beta_constitutive",
    "covers_complex_beta_curl",
    "kernel_source",
    "set_kernel_source",
)

#: The two kernels this family emits, keyed by the sub-step they replace. The
#: constitutive sides are NOT here: they are the certified complex pair's,
#: unchanged.
BETA_KERNELS: Dict[str, str] = {
    "step_B": "step_B_pml_complex_beta",
    "step_D": "step_D_pml_complex_beta",
}

#: Byte-identical to the array path with a DEVICE verdict behind it, 2026-09-01:
#: :data:`BETA_COMPLEX_ADMISSION` carries ``released`` True with both policy
#: artifacts, and ``certification.json``'s ``cuda_complex_beta_2026-09-01`` block
#: names the same two. A name here without a record block is the failure
#: ``test_kernel_partition.py`` exists to catch.
CERTIFIED_KERNELS = (
    "step_B_pml_complex_beta",
    "step_D_pml_complex_beta",
)

#: EMPTY since 2026-09-01, and what emptied it was a RUN rather than an argument:
#: gate_cuda_complex_beta.py on the GPU host (one RTX A6000, cc 8.6, CuPy 13.5.1,
#: NVRTC 11.6), 48/48 curl cases byte-identical per sub-step at one launch and at
#: 60, folded and unfolded, AND the certified complex constitutive pair 24/24 per
#: side on the same beta runs, both float32 subnormal policies; 16 mutations
#: CAUGHT, 3 NULLS CONFIRMED and 1 REPORTED platform finding
#: (``synthesise_the_zero_real_word``: the emitted arithmetic reads the beta
#: coefficient's imaginary word alone, so the passed-through real word is
#: defensive) per policy. Spelled as a dict WITHOUT a type annotation for the
#: reason ``constitutive_kernels.py`` records: the partition readers walk the
#: syntax tree so they run where there is no CuPy, and an annotated assignment is
#: an ``ast.AnnAssign`` the plain-assignment readers do not match -- annotating it
#: makes the name invisible and the partition unenforced.
UNCERTIFIED_KERNELS = {}

CURL_SUB_STEPS: Tuple[str, ...] = ("step_B", "step_D")

#: Which side of the step each sub-step is, in ``stepping._special_kz_beta_term``'s
#: sense: the MAGNETIC side takes ``+1j`` and the electric side ``-1j``
#: (stepping.py:798-799). ``step_B`` is called with ``magnetic=True`` (:389, :391)
#: and ``step_D`` with ``magnetic=False`` (:472, :474).
MAGNETIC: Dict[str, bool] = {"step_B": True, "step_D": False}

#: Which stencil register each beta target's partner already sits in. Target 0
#: (Bx / Dx) takes the SECOND source's centre -- Ey on the B side, Hy on the D side
#: -- which the emitted template names ``f_2``; target 1 (By / Dy) takes the FIRST
#: source's centre, ``f_1``. Target 2 takes none. Transcribed from
#: ``stepping.B_CURL_TERMS`` / ``D_CURL_TERMS`` (stepping.py:214-223) against the
#: call sites (:389/:391, :472/:474), and pinned against them by
#: ``test_complex_beta.py``.
#:
#: THE REGISTER MUST BE THE UNSHIFTED CENTRE, never an ``sf``/``ss`` shifted
#: operand: ``_special_kz_beta_term`` takes the component SNAPSHOT (:293, :408) and
#: no shift helper is anywhere near it.
BETA_PARTNER: Dict[int, Tuple[str, str]] = {
    0: ("f_2", "bp"),
    1: ("f_1", "bm"),
}


def beta_curl_coefficients(beta: float, dt: float, magnetic: bool):
    """``((plus_re, plus_im), (minus_re, minus_im))`` -- stepping.py:797-811, twice.

    Transcribed line for line, with the ``+-1j`` branch TAKEN because this family
    is complex storage only::

        coefficient = sign * 2.0 * math.pi * grid.beta * grid.dt   # :770, float64
        coefficient = coefficient * (1j if magnetic else -1j)      # :771-772
        partner_values.dtype.type(coefficient)                     # :784, ONE round

    TWO COEFFICIENTS, ONE PER SIGN, rather than one negated in the kernel. Complex
    negation is exact, so the two spellings cannot differ in value -- but binding
    both keeps the transcription literal per call site instead of resting on that
    identity, and it is what the gate's ``swap_beta_signs`` host mutation corrupts.

    THE REAL WORD IS WHATEVER PYTHON PRODUCED and is passed through: see the module
    docstring for the measurement. ``float()`` of a ``numpy.float32`` preserves the
    zero's sign, so the words the kernel receives are the array path's bits.

    ``numpy`` is imported in the BODY so this module stays stdlib-only at import
    and can be read on the census host; the rounding is spelled with the same
    object ``stepping`` uses (``dtype.type``, i.e. ``numpy.complex64``) rather than
    with ``struct``, so the two cannot round differently.
    """
    import numpy  # noqa: PLC0415

    out = []
    for sign in (1.0, -1.0):
        coefficient = sign * 2.0 * math.pi * float(beta) * float(dt)  # :770
        coefficient = coefficient * (1j if magnetic else -1j)         # :771-772
        rounded = numpy.complex64(coefficient)                        # :784
        out.append((float(numpy.float32(rounded.real)),
                    float(numpy.float32(rounded.imag))))
    return tuple(out)


#: The coefficient-left complex product, as a NAME over ``complex_emitter``'s own
#: arm-dependent body. See the module docstring: "left" is "first operand" in every
#: one of this family's orientations, so this is not a second arithmetic and it has
#: no arm branch of its own.
_MUL_IMAG_HELPER = '''
// (c_re + i*c_im) * z with the COEFFICIENT on the LEFT -- stepping.py:784's
// partner_values.dtype.type(coefficient) * partner_values, whose left operand is
// the scalar. THIS IS rotate_field_left WITH THE COEFFICIENT AS THE FIRST
// OPERAND, which is the whole of it: that helper fuses the first operand's
// product under FMA_V1 and rounds both separately under NAIVE, and "left" means
// "first operand" in every orientation this family carries. A second body here
// would be a second spelling of one rule, with two arms to keep in step by hand.
__device__ __forceinline__ cf mul_imag_coefficient_left(cf c, cf z) {
    return rotate_field_left(c, z);
}
'''  # stepping.py live lines for the frozen device-text citation(s) in this string: 784->811


def _beta_line(register: str, coefficient: str) -> str:
    """The one statement the beta insert adds to one target."""
    return (
        f"        // beta (stepping.py:770-784) at this term's own sign, added to\n"  # stepping.py live lines for the frozen device-text citation(s) in this string: 770-784->797-811
        f"        // the curl BEFORE the masks -- the array path's order (:361/:363\n"  # stepping.py live lines for the frozen device-text citation(s) in this string: 361->389, 363->391
        f"        // after :342 and before :369; :443/:445 after :429, before :450).\n"
        f"        // curl + (-(c * partner)) IS curl - (c * partner): IEEE defines\n"
        f"        // subtraction as addition of the negation, complex negation\n"
        f"        // negates both words and complex add is plane-wise, so the single\n"
        f"        // subtract is the same bits on every input, signed zeros included.\n"
        f"        curl = cf_sub(curl, mul_imag_coefficient_left({coefficient}, {register}));\n")


def _deltas_for(sub_step: str) -> Tuple[Tuple[str, str, str, int], ...]:
    """Every text delta this family applies to the FOLDED complex source, in order.

    ``(label, old, new, expected sites)``. A SITE COUNT IS A WELD: if a sibling
    moves an anchor the transform must FAIL rather than emit a kernel with one beta
    term missing, which compiles, runs, and is wrong on one component.

    NO COMMENT GOES INSIDE THE PARAMETER LIST, and that is a hard constraint rather
    than a style choice: ``gate_cuda_complex.parse_signature`` reads the signature
    with ``[^)]*``, so a ``)`` anywhere between the kernel's parentheses -- in a
    source citation, say -- truncates the parsed parameter list and the
    host-compiled backend builds a driver with the wrong arity. Measured
    2026-08-20: a four-line citation in the signature cost one parameter, 34 of 35,
    and the host leg failed to link. The explanation lives in the body instead.
    """
    backward = complex_emitter.KERNELS[sub_step][1]
    out = [
        (
            "coefficient_left_helper",
            "\n// stepping._shift_up (stepping.py:1725-1786), PERIODIC and METALLIC",  # stepping.py live lines for the frozen device-text citation(s) in this string: 1725-1786->1772-1833
            _MUL_IMAG_HELPER
            + "\n// stepping._shift_up (stepping.py:1725-1786), PERIODIC and METALLIC",  # stepping.py live lines for the frozen device-text citation(s) in this string: 1725-1786->1772-1833
            1,
        ),
        (
            "beta_coefficient_arguments",
            "    float pxr, float pxi, float pyr, float pyi, float pzr, float pzi\n) {",
            "    float pxr, float pxi, float pyr, float pyi, float pzr, float pzi,\n"
            "    float bp_re, float bp_im, float bm_re, float bm_im\n) {",
            1,
        ),
        (
            "beta_coefficient_registers",
            "    cf pz; pz.re = pzr; pz.im = pzi;\n",
            "    cf pz; pz.re = pzr; pz.im = pzi;\n"
            "    // The two complex64-rounded beta coefficients, one per call-site\n"
            "    // sign, host-rounded ONCE on the host at stepping.py:784. The real\n"  # stepping.py live lines for the frozen device-text citation(s) in this string: 784->811
            "    // word is a SIGNED zero left there by Python's own complex\n"
            "    // multiply and is passed through, never synthesised here.\n"
            "    cf bp; bp.re = bp_re; bp.im = bp_im;\n"
            "    cf bm; bm.re = bm_re; bm.im = bm_im;\n",
            1,
        ),
    ]
    for target, (register, coefficient) in sorted(BETA_PARTNER.items()):
        mask = _folded.fold_mask_lines(
            complex_emitter._MASK_AXES[backward][target], target)
        out.append((f"beta_term_target_{target}",
                    mask, _beta_line(register, coefficient) + mask, 1))
    out.append((
        "rename",
        f'extern "C" __global__ void {_folded.FOLDED_KERNELS[sub_step]}(',
        f'extern "C" __global__ void {BETA_KERNELS[sub_step]}(',
        1,
    ))
    return tuple(out)


#: The delta labels, in application order -- what a reader checks a diff against.
BETA_DELTAS: Tuple[str, ...] = tuple(
    label for label, _o, _n, _s in _deltas_for("step_B"))


def beta_source(sub_step: str, expansion) -> str:
    """The device source for one complex-beta curl sub-step under one arm.

    Derived from :func:`complex_folded_kernels.folded_source`, which is itself
    derived from ``complex_emitter``: the fold branches ride along and are simply
    dead on an unfolded grid, which is what lets ONE pair of kernels serve all four
    beta corpus rows. ``expansion`` is REQUIRED and never defaulted.
    """
    if sub_step not in BETA_KERNELS:
        raise ValueError(
            f"sub_step must be one of {sorted(BETA_KERNELS)}, got {sub_step!r}")
    source = _folded.folded_source(sub_step, expansion)
    for label, old, new, expected in _deltas_for(sub_step):
        found = source.count(old)
        if found != expected:
            raise RuntimeError(
                f"the complex-beta delta {label!r} for {sub_step} matched {found} "
                f"sites in the folded-complex source, not {expected}; a sibling has "
                f"moved an anchor and this transform must not emit a kernel with a "
                f"beta term missing")
        source = source.replace(old, new)
    return source


#: The mutable copy the launcher compiles, keyed by (sub-step, arm). The gate
#: mutates through :func:`set_kernel_source`; ONE seam, no second copy to forget.
_SOURCES: Dict[Tuple[str, int], str] = {}


def kernel_source(sub_step: str, expansion) -> str:
    """The device text this family would compile for one sub-step and arm."""
    arm = complex_emitter.normalized_expansion(expansion)
    key = (sub_step, arm)
    if key not in _SOURCES:
        _SOURCES[key] = beta_source(sub_step, arm)
    return _SOURCES[key]


def set_kernel_source(sub_step: str, expansion, source: str) -> None:
    """Replace one kernel's device text -- the gate's mutation seam, and only that."""
    if sub_step not in BETA_KERNELS:
        raise ValueError(
            f"sub_step must be one of {sorted(BETA_KERNELS)}, got {sub_step!r}")
    _SOURCES[(sub_step, complex_emitter.normalized_expansion(expansion))] = source


def reset_kernel_sources() -> int:
    """Drop every mutated body; returns how many entries went. The gate's undo."""
    count = len(_SOURCES)
    _SOURCES.clear()
    return count


def corpus_digest() -> str:
    """One sha256 over every source this family can emit, canonically ordered."""
    import hashlib  # noqa: PLC0415

    digest = hashlib.sha256()
    for sub_step in sorted(BETA_KERNELS):
        for name in sorted(complex_emitter.EXPANSIONS):
            digest.update(f"{sub_step}|{name}".encode("ascii"))
            digest.update(beta_source(sub_step, name).encode("utf-8"))
    return digest.hexdigest()


#: WHAT HAS BEEN MEASURED ON A DEVICE FOR THIS FAMILY, AND WHAT HAS NOT.
#: Empty of a verdict until ``gate_cuda_complex_beta.py`` fills it from an
#: artifact; ``released`` False is the honest default and not a placeholder.
BETA_COMPLEX_ADMISSION: Dict[str, Any] = {
    "gate": "parity/meep_gpu/gate_cuda_complex_beta.py",
    # RELEASED ON DEVICE, 2026-09-01: the GPU host, one RTX A6000 (cc 8.6, CuPy
    # 13.5.1, NVRTC 11.6), under BOTH float32 subnormal policies, the arm bound
    # from the per-policy expansion probe (FMA_V1 both). The ``device_leg`` block
    # below carries the numbers, read off the two artifacts; the ``host_leg``
    # block is kept as the 2026-08-20 transcription record it always was.
    "released": True,
    "artifacts": (
        "parity/meep_gpu/results/cuda_complex_beta_2026-09-01/keep/gate.json",
        "parity/meep_gpu/results/cuda_complex_beta_2026-09-01/flush/gate.json",
    ),
    "device_leg": {
        "recorded_utc": "2026-09-01T12:35:56Z",
        "host": "the GPU host",
        "device": "NVIDIA RTX A6000",
        "compute_capability": "8.6",
        "cupy_version": "13.5.1",
        "nvrtc_version": "11.6",
        "policies": ("ieee_keep_ftz_stripped", "meep_x86_flush"),
        "arm": "FMA_V1",
        # 6 fixtures x 2 sub-steps x 2 courants x 2 value classes, PER POLICY.
        "curl_cases_scored": 48,
        "curl_single_launch_identical": 48,
        "curl_multi_step_identical": 48,
        "folded_cases": 32,
        "unfolded_cases": 16,
        # The CERTIFIED complex constitutive pair on the same beta runs, scored
        # per side -- the admission's own device leg, not an inference.
        "constitutive_cases_scored": {"H": 24, "E": 24},
        "constitutive_identical": {"H": 24, "E": 24},
        "differing_words": 0,
        "beta_term_is_live_floor": ("asserted per case, INCLUDING the component "
                                    "pattern: targets 0 and 1 must move and "
                                    "target 2 must not, measured against "
                                    "stepping"),
        # Per policy: 16 must-catch CAUGHT, 3 nulls confirmed, 1 REPORTED.
        "mutation_legs": 20,
        "mutation_legs_caught": 16,
        "mutation_legs_null_confirmed": 3,
        "mutation_legs_reported": 1,
        "synthesise_the_zero_real_word": (
            "REPORTED, 0/12 both policies -- the leg's own contract calls an "
            "all-legs-uncaught outcome a platform finding: the emitted "
            "arithmetic reads the beta coefficient through "
            "mul_imag_coefficient_left, which consumes the imaginary word "
            "alone, so the real word this mutation zeroes is defensive "
            "pass-through on this platform"),
    },
    "host_leg": {
        "artifact": ("parity/meep_gpu/results/"
                     "cuda_complex_beta_2026-08-20b_host/gate.json"),
        "backend": "host-compiled (clang++), NumPy state; certifies no NVRTC",
        "recorded_utc": "2026-08-20T09:40:00Z",
        "courants": (0.5, 0.35),
        "value_classes": ("uniform", "subnormal_band"),
        "multi_step_budget": 60,
        # 6 fixtures x 2 sub-steps x 2 courants x 2 value classes.
        "curl_cases_scored": 48,
        "curl_single_launch_identical": 48,
        "curl_multi_step_identical": 48,
        "folded_cases": 32,
        "unfolded_cases": 16,
        "constitutive_cases_scored": {"H": 24, "E": 24},
        "constitutive_identical": {"H": 24, "E": 24},
        "differing_words": 0,
        "beta_term_is_live_floor": ("asserted per case, INCLUDING the component "
                                    "pattern: targets 0 and 1 must move and target "
                                    "2 must not, measured against stepping"),
        "mutation_legs": 17,
        "mutation_legs_caught": 13,
        "mutation_legs_null_confirmed": 3,
        "mutation_legs_uncaught": 1,
        "drop_the_beta_term": "CAUGHT 12/12",
        "swap_the_beta_signs": "CAUGHT 12/12",
        "beta_added_not_subtracted": "CAUGHT 12/12",
        "scale_beta_by_dtdx": "CAUGHT 12/12",
        "beta_on_the_third_component": "CAUGHT 12/12",
        # THE COMPOSITION NEEDLE. The insert moved BELOW the masks, which a FOLD
        # makes reach twice as far: measured 0 differing words on an all-periodic
        # unfolded grid, 32 per sub-step with one wall, 64 per sub-step on a folded
        # PERIODIC axis. Scored only on fixtures that carry a mask on a beta target.
        "beta_after_mask": "CAUGHT 10/10 (scoped to fixtures with a wall or a fold)",
        "beta_subtract_as_add_negative": "NULL CONFIRMED",
        # THE REAL FAMILY'S MEASURED NULL, RE-MEASURED HERE rather than inherited:
        # every beta grid has nz == 1, so both partners' only shifted companion is
        # the neighbour along Z and the two are the same word.
        "beta_partner_shifted": "NULL CONFIRMED 0/12",
        # A MEASURED REFUSAL, AND THE ONE OPEN FINDING OF THIS FAMILY. Writing the
        # coefficient's real word as a literal +0.0f instead of passing the SIGNED
        # zero Python's complex multiply produced changed NO output word on 12
        # legs. So on this platform, at the `uniform` value class, the pass-through
        # is belt and braces and NOT load-bearing -- which is the opposite of what
        # the Triton sibling's reasoning predicts, and is recorded as a null rather
        # than argued away. WHAT WOULD DISCRIMINATE IT and is not in this gate: a
        # `zero_init` or `signed_zero` value class, where the field's other word is
        # exactly zero and the cross term's sign is the only thing that can move a
        # word. Until that leg runs, "the signed zero matters" is UNMEASURED here.
        "synthesise_the_zero_real_word": "UNCAUGHT 0/12 -- see the note above",
    },
    #: SLOTS, recomputed from the census record and re-measured by the union
    #: analyzer over the patched copy: 16 on 4 rows, all four sub-steps, because
    #: the certified complex constitutive pair still refuses beta by name.
    "slots": 16,
    "rows": 4,
    "sub_steps": ("step_B", "step_D", "update_H", "update_E"),
    "denominator": 759,
    "what_it_does_not_license": (
        "any dispatch: nothing in meep_gpu imports cuda_kernels.",
        "REAL storage with beta: that is special_kz_curl's, certified separately, "
        "and this family refuses it by name.",
        "a z-thick grid: Grid._resolve_beta refuses beta outside an effective 2-D "
        "Cartesian grid, so nz == 1 on every grid this family can be handed and "
        "the z ghost rule is exercised at one cell only.",
        "cylindrical (Dcyl) coordinates with beta: MEEP aborts (fields.cpp:"
        "546-547) and Grid._resolve_beta refuses the pairing at construction.",
        "a folded axis carrying a Bloch phase: inherited refusal from "
        "complex_folded_kernels._fold_reasons; there is no such run.",
        "a conductivity, BFAST or three simultaneous fold planes: refused by "
        "inherited clauses, untested here.",
        "any throughput claim: this is a correctness family and times nothing.",
        "that the coefficient's SIGNED ZERO real word is load-bearing: the device "
        "leg measured the opposite and the record says so as a REPORTED finding "
        "-- synthesise_the_zero_real_word came back uncaught 0/12 on BOTH "
        "policies over uniform and subnormal-band operands, because the emitted "
        "arithmetic reads the coefficient through mul_imag_coefficient_left, "
        "which consumes the imaginary word alone. The pass-through stays in the "
        "kernel as defence in depth; nothing may cite it as measured "
        "load-bearing.",
        "a FUSED product on this family's cell: both halves of the B_to_H "
        "complex-beta cell are certified singles as of 2026-09-01, and the weld "
        "that serves its 3 reachable rows is the separate product "
        "complex_beta_fused_magnetic_pair, licensed by its own record "
        "(cuda_complex_beta_fused_magnetic_pair_2026-09-02) and never by this "
        "one.",
    ),
}


# ---------------------------------------------------------------------------
# THE PREDICATES
# ---------------------------------------------------------------------------

class _BetaFreeGrid:
    """The run's grid, answering ``beta = 0.0`` and forwarding everything else.

    The same technique, and the same reason, as
    ``special_kz_curl._BetaFreeGrid``: the shipped predicates SHORT-CIRCUIT, so
    the one clause this family inverts has to be satisfied at the INPUT rather
    than filtered out of a refusal string afterwards. ``grid.beta`` is read in
    exactly one place in the certified complex predicate --
    ``coverage._grid_facts`` (:670), consumed at :2373 -- so this changes the
    answer to that clause and to nothing else.
    """

    __slots__ = ("_grid",)

    def __init__(self, grid: Any) -> None:
        object.__setattr__(self, "_grid", grid)

    @property
    def beta(self) -> float:
        return 0.0

    def __getattr__(self, item: str) -> Any:
        return getattr(object.__getattribute__(self, "_grid"), item)


class _BetaFreeFoldedGrid(_folded._FoldFreeGrid):
    """Fold-free AND beta-free, for the folded arm's delegation into the base.

    Two proxies rather than a proxy of a proxy, because ``_FoldFreeGrid``
    forwards through ``__getattr__`` and a ``beta`` property on a subclass is
    resolved before that forwarding -- which is exactly the composition wanted and
    is spelled as one class so a reader can see both overrides in one place.
    """

    __slots__ = ()

    @property
    def beta(self) -> float:
        return 0.0


def _beta_reasons(fields: Any, grid: Any) -> Any:
    """The clauses THIS family owns, in place of the one it inverts. None if clear.

    Five, each a wrong answer rather than a crash if it is dropped:

    1. **beta must be NONZERO.** A beta = 0 complex run belongs to the certified
       complex pair (unfolded) or to ``complex_folded_kernels`` (folded); admitting
       one here would put two families on one slot, which the union census reports
       as a FINDING rather than as extra coverage.
    2. **COMPLEX storage.** A real-storage beta run does NOT take the ``+-1j``
       branch (``partner_values.dtype.kind`` is ``'f'``) and belongs to
       ``special_kz_curl``, which is certified. The refusal names that family
       rather than describing the storage.
    3. **NOT cylindrical.** MEEP aborts (fields.cpp:546-547) and
       ``Grid._resolve_beta`` refuses the pairing at construction (grid.py:668-697)
       -- RESTATED here rather than inherited, because a predicate that reasons
       "the constructor would have refused it" admits by argument from absence.
    4. **dimensions == 2**, for the same reason and from the same lines.
    5. **THE FOLD, when there is one, must satisfy every clause
       ``complex_folded_kernels`` owns** -- the termination cross-check and the
       no-phase-on-a-folded-axis rule. This family's kernels ARE the folded ones
       with an insert, so a fold it could not serve is a fold this cannot serve
       either, and asking the sibling is what stops the two from drifting apart.
    """
    try:
        beta = float(getattr(grid, "beta", 0.0))
    except Exception as exc:  # noqa: BLE001 - a raise is not a refusal unless caught
        return f"the grid could not be asked for beta: {type(exc).__name__}: {exc}"
    if beta == 0.0:
        return ("grid.beta is zero: this pair exists only for special_kz runs, and "
                "a beta = 0 complex run belongs to coverage."
                "covers_real_pml_complex_curl or to complex_folded_kernels")
    complex_storage = bool(getattr(fields, "force_complex_fields", False)) or bool(
        getattr(grid, "has_bloch", False))
    if not complex_storage:
        return ("real float32 storage with beta: stepping's +-1j branch "
                "(stepping.py:798-799) is not taken at all under real storage, and "
                "that configuration is special_kz_curl's certified pair's")
    if bool(getattr(grid, "cylindrical", False)):
        return ("cylindrical (Dcyl) coordinates with beta: MEEP aborts "
                "(fields.cpp:546-547) and Grid._resolve_beta refuses the pairing "
                "at construction (grid.py:668-697)")
    try:
        dimensions = int(getattr(grid, "dimensions", 0))
    except Exception as exc:  # noqa: BLE001
        return (f"the grid could not be asked for dimensions: "
                f"{type(exc).__name__}: {exc}")
    if dimensions != 2:
        return (f"grid dimensions={dimensions} is not the effective-2-D grid beta "
                f"requires (grid.py:668-697; MEEP fields.cpp:546-547, \"Nonzero "
                f"beta unsupported in dimensions other than 2\")")
    try:
        folded = bool(grid.has_symmetry()) or any(
            bool(grid.is_mirrored(axis)) for axis in range(3))
    except Exception as exc:  # noqa: BLE001
        return (f"the grid could not be asked whether it is folded: "
                f"{type(exc).__name__}: {exc}")
    if folded:
        return _folded._fold_reasons(fields, grid)
    return None


def _proxy_for(grid: Any) -> Any:
    """The grid to delegate with: beta-free always, fold-free when there is a fold."""
    try:
        folded = bool(grid.has_symmetry()) or any(
            bool(grid.is_mirrored(axis)) for axis in range(3))
    except Exception:  # noqa: BLE001 - _beta_reasons already refused this grid
        folded = False
    return _BetaFreeFoldedGrid(grid) if folded else _BetaFreeGrid(grid)


def covers_complex_beta_curl(fields: Any, pml: Any, grid: Any, sub_step: str,
                             license: Any = None,
                             subnormal_policy: Any = None) -> tuple:
    """Whether ``step_{B,D}_pml_complex_beta`` may serve this run.

    ``sub_step`` is ``"step_B"`` or ``"step_D"``. Returns ``(covered, reason)``.

    DELEGATION, NOT A SECOND CLAUSE SET. Every question except beta and the fold is
    the CERTIFIED complex curl predicate's, asked through it on a proxy grid whose
    beta reads zero and -- when the grid is folded -- which also reports no fold, so
    this family cannot drift to a weaker standard than the one it borrows its
    arithmetic and its masks from. The conductivity split, the halved int32 bound,
    the volume and coefficient-vector tail and the EXPANSION licence are all
    inherited rather than restated.

    THE SUB-STEP ARGUMENT IS REQUIRED and carries more here than elsewhere: as well
    as the per-term conductivity split (``stepping._apply_curl`` reads
    ``fields.condfac_for(term.target)`` per TERM, stepping.py:508), the two
    sub-steps take DIFFERENT beta coefficients -- ``+1j`` on the magnetic side and
    ``-1j`` on the electric one (:771-772) -- so one predicate answering for both
    would be answering about two different kernels.
    """
    if sub_step not in CURL_SUB_STEPS:
        raise ValueError(
            f"sub_step must be 'step_B' or 'step_D', got {sub_step!r}")
    own = _beta_reasons(fields, grid)
    if own is not None:
        return False, own
    for name, predicate in (
            ("coverage.covers_real_pml_complex_curl",
             _coverage.covers_real_pml_complex_curl),
            ("complex_folded_kernels.covers_complex_folded_curl",
             _folded.covers_complex_folded_curl)):
        served, _ = predicate(fields, pml, grid, sub_step, license=license,
                              subnormal_policy=subnormal_policy)
        taken = _folded._already_served(served, name)
        if taken is not None:
            return False, taken
    return _coverage.covers_real_pml_complex_curl(
        fields, pml, _proxy_for(grid), sub_step,
        license=license, subnormal_policy=subnormal_policy)


def covers_complex_beta_constitutive(fields: Any, pml: Any, grid: Any, side: str,
                                     license: Any = None,
                                     subnormal_policy: Any = None) -> tuple:
    """Whether the CERTIFIED complex constitutive pair may serve a BETA run.

    ``side`` is ``"H"`` or ``"E"``; returns ``(covered, reason)``.

    THIS FAMILY BUILDS NO CONSTITUTIVE KERNEL, and that is the finding rather than
    an omission: ``stepping.update_H`` (:907-925) and ``update_E`` (:926-995) reach
    ``_apply_constitutive_pml`` (:2065) and neither they nor it read ``grid.beta``
    or any array the beta term writes. So the admission is the certified
    predicate's with the beta clause -- and, on a folded grid, the fold clause --
    inverted, and the ARITHMETIC is ``complex_emitter``'s own, untouched.

    THAT READING IS NOT THE EVIDENCE. ``gate_cuda_complex_beta.py`` carries a
    constitutive arm that runs the shipped complex pair against
    ``stepping.update_H`` / ``update_E`` on beta grids whose two curls have already
    moved the state. A sub-step admitted because nothing was found to read beta,
    with no device leg, is admitted by argument from absence -- which is what the
    Metal track's tranche-6 split exists to warn about.
    """
    if side not in ("H", "E"):
        raise ValueError(f"side must be 'H' or 'E', got {side!r}")
    own = _beta_reasons(fields, grid)
    if own is not None:
        return False, own
    for name, predicate in (
            ("coverage.covers_real_pml_complex_constitutive",
             _coverage.covers_real_pml_complex_constitutive),
            ("complex_folded_kernels.covers_complex_folded_constitutive",
             _folded.covers_complex_folded_constitutive)):
        served, _ = predicate(fields, pml, grid, side, license=license,
                              subnormal_policy=subnormal_policy)
        taken = _folded._already_served(served, name)
        if taken is not None:
            return False, taken
    return _coverage.covers_real_pml_complex_constitutive(
        fields, pml, _proxy_for(grid), side,
        license=license, subnormal_policy=subnormal_policy)


# ---------------------------------------------------------------------------
# COMPILATION AND LAUNCH
# ---------------------------------------------------------------------------

#: ``--fmad=false`` is CORRECTNESS here and not tuning, and the beta insert adds a
#: FOURTH contraction candidate to the three ``complex_emitter``'s note 5 names:
#: ``cf_sub(curl, c*g)`` is a multiply feeding a subtract, exactly the shape a
#: compiler fuses, where the array path rounds the product (stepping.py:811) and
#: the addition (:389) separately. The gate substitutes the empty tuple as a
#: control and REPORTS whether the guard changed an answer, rather than asserting
#: that it must.
_COMPILE_OPTIONS: Tuple[str, ...] = ('--fmad=false',)

_COMPLEX_THREADS = 256


def _get_kernel(sub_step: str, expansion):
    """Compile one sub-step under one arm, memoized on (name, options, policy, source).

    The source is read through :func:`kernel_source` PER CALL so a body rewritten
    through :func:`set_kernel_source` is a memo MISS and reaches NVRTC.
    """
    import cupy as cp  # noqa: PLC0415

    from .compile_cache import get_or_compile, kernel_cache_key  # noqa: PLC0415

    arm = complex_emitter.normalized_expansion(expansion)
    name = BETA_KERNELS[sub_step]
    code = kernel_source(sub_step, arm)
    key = kernel_cache_key(f"{name}_arm{arm}", True, _COMPILE_OPTIONS, code)
    return get_or_compile(
        key, lambda: cp.RawKernel(code, name, options=_COMPILE_OPTIONS))


def _clear_kernel_cache() -> int:
    """Drop every memoized kernel; returns how many entries went."""
    from .compile_cache import clear_kernel_cache  # noqa: PLC0415

    return clear_kernel_cache()


def step_complex_beta(sub_step: str, fields: Any, expansion,
                      grid: Any = None, pml: Any = None, *,
                      dtdx: float = None,
                      tables: Dict[str, Any] = None,
                      boundary_codes: Sequence[Any] = None,
                      phase_flags: Sequence[Any] = None,
                      phase_values: Sequence[Any] = None,
                      beta_coefficients=None) -> None:
    """One complex-beta curl sub-step, in place, in one launch.

    The argument list is ``complex_folded_kernels.step_folded_complex``'s with the
    four beta words APPENDED, which is what lets a reader diff the two launches.
    ``beta_coefficients`` is :func:`beta_curl_coefficients`' ``((plus_re, plus_im),
    (minus_re, minus_im))``; supplied explicitly it is the gate's door, and derived
    from ``grid`` it comes from that grid's own ``beta`` and ``dt`` with the
    sub-step's own ``magnetic`` flag.
    """
    import numpy as np  # noqa: PLC0415

    from .complex_pml_kernels import (bloch_phase_arguments,  # noqa: PLC0415
                                      complex_curl_tables, word_view)

    if sub_step not in BETA_KERNELS:
        raise ValueError(
            f"sub_step must be one of {sorted(BETA_KERNELS)}, got {sub_step!r}")
    backward = complex_emitter.KERNELS[sub_step][1]
    if (pml is None) == (tables is None):
        raise ValueError(
            "pass exactly one of pml (the tables are derived from the sub-step's "
            "own sub-lattice) or tables (the gate supplies its own)")
    if tables is None:
        tables = complex_curl_tables(pml, complex_emitter.HALF_INTEGER[sub_step])
    if (grid is None) == (boundary_codes is None):
        raise ValueError(
            "pass exactly one of grid (the fold-aware boundary codes, the phase "
            "table and the beta coefficients are resolved from it) or "
            "boundary_codes (the gate's own)")
    if grid is not None:
        codes, refusal = _folded.folded_complex_boundary_codes(grid)
        if refusal is not None:
            raise ValueError(
                f"this grid has no folded-complex boundary codes: {refusal}")
        boundary_codes = codes
        derived_flags, derived_values = bloch_phase_arguments(grid, backward)
        if phase_flags is None:
            phase_flags = derived_flags
        if phase_values is None:
            phase_values = derived_values
        if dtdx is None:
            dtdx = grid.dt / grid.dx  # stepping.py:314, :431.
        if beta_coefficients is None:
            beta_coefficients = beta_curl_coefficients(
                grid.beta, grid.dt, MAGNETIC[sub_step])
    if (phase_flags is None or phase_values is None or dtdx is None
            or beta_coefficients is None):
        raise ValueError(
            "without a grid the caller must supply phase_flags, phase_values, "
            "dtdx and beta_coefficients; there is nothing here to derive them from")

    targets = ("Bx", "By", "Bz") if sub_step == "step_B" else ("Dx", "Dy", "Dz")
    sources = ("Ex", "Ey", "Ez") if sub_step == "step_B" else ("Hx", "Hy", "Hz")
    shape = getattr(fields, targets[0]).shape
    arguments = [word_view(getattr(fields, name)) for name in targets]
    arguments += [word_view(getattr(fields, "fu_" + name)) for name in targets]
    arguments += [word_view(getattr(fields, name)) for name in sources]
    arguments += [np.int32(shape[0]), np.int32(shape[1]), np.int32(shape[2]),
                  np.float32(dtdx)]
    for axis in ("x", "y", "z"):
        arguments += [tables[f"kms_{axis}"], tables[f"sinv_{axis}"]]
    arguments += [np.int32(code) for code in boundary_codes]
    arguments += [np.int32(flag) for flag in phase_flags]
    arguments += [np.float32(value) for value in phase_values]
    (plus_re, plus_im), (minus_re, minus_im) = beta_coefficients
    arguments += [np.float32(plus_re), np.float32(plus_im),
                  np.float32(minus_re), np.float32(minus_im)]
    cells = int(shape[0]) * int(shape[1]) * int(shape[2])
    blocks = (cells + _COMPLEX_THREADS - 1) // _COMPLEX_THREADS
    _get_kernel(sub_step, expansion)(
        (blocks,), (_COMPLEX_THREADS,), tuple(arguments))
