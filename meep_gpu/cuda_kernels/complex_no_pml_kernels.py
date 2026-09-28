"""COMPLEX storage with NO ABSORBER: six hand-CUDA kernels and three predicates.

WHAT THE CORPUS ACTUALLY DEMANDS HERE, MEASURED BEFORE ANYTHING WAS WRITTEN.
The 2026-08-20 closeout census leaves 22 unserved slots on SIX rows whose common
shape is "complex64 storage, no active absorber" -- ``cuda_complex`` refuses them
for "no active PML layer" and ``cuda_no_pml_curl`` for "Bx/Dx is complex64, not
float32". The brief for this family read those 22 as 12 curl + 6 update_E +
4 update_P and projected THREE kernels. Lifting the six rows and asking the real
``Grid``/``Fields`` objects says otherwise, and the split is the finding:
``parity/meep_gpu/results/cuda_complex_no_pml_recon_2026-08-20/facts/``.

=======================================  ==========  ==========  ==============
row (4 x test_dump_load, 2 x matgrid)    condfac     poles       offdiag eps
=======================================  ==========  ==========  ==============
``TestLoadDump.*_3d`` (4 rows)           ALL SIX     1 lorentz   no
``TestMaterialGrid.test_matgrid_3d``     none        none        YES
``TestMaterialGrid.test_subpixel_smoothing``  none   none        YES
=======================================  ==========  ==========  ==============

All six store E (``fields.stores_E`` True, ``Ex`` a live complex64 volume), all
six have ``pml is None``, no ``fu_*`` and no ``f_w_*``, ``get_H('Hx') is
fields.Bx``, and all six carry a Bloch phase on all three periodic axes.

TWO CONSEQUENCES, BOTH OF WHICH CHANGED THE DESIGN:

1. **THE TRAP THAT COST THE REAL no-PML FAMILY A ROUND DOES NOT FIRE HERE, AND
   A DIFFERENT ONE DOES.** ``no_pml_curl.py`` needed a third kernel because
   ``fields.stores_E`` was False on five of eight rows and ``step_B`` differenced
   a DERIVED ``D * inv_eps`` volume. Measured on all six rows here,
   ``stores_E`` is True and ``fields.Ex`` is allocated, so ``step_B`` differences
   the stored E and NO derived arm is needed. What fires instead is the
   CONDUCTIVITY: ``fields.condfac_for`` returns a float32 volume for every one of
   Bx..Dz on the four ``test_dump_load`` rows, so ``stepping._apply_curl`` routes
   them to ``_apply_conductive_update`` (stepping.py:536-537, :1985-1998) -- a
   different tail from ``target -= curl``. EIGHT of the twelve curl slots are
   conductive; four are plain. One kernel cannot serve both, so there are two
   curl pairs here and not one.

2. **THE SIX update_E SLOTS ARE TWO DIFFERENT SUB-STEPS.** The four dispersive
   rows want ``E = ((D - P0) - P1 ...) * inv_eps`` (stepping.py:1013-1022 through
   ``Fields.displacement_minus_polarization``, fields.py:1079-1105). The two
   material-grid rows carry ``has_offdiagonal_epsilon`` True, so their
   ``update_E`` is the TENSOR ROW PRODUCT (stepping.py:1001-1008) -- a four-point
   transverse average with partner-axis shifts, Bloch phases on the shifted
   operands and wall masks. THIS FILE DOES NOT BUILD THAT and refuses it by name;
   it is the two slots this family leaves on the table. See
   :data:`WHAT_IS_NOT_BUILT`.

So: six kernels, three predicates, TWENTY of the twenty-two slots.

============================================================================
WHAT COLLAPSES, AND WHAT DOES NOT
============================================================================

``stepping._apply_curl`` (stepping.py:490-539) has four tails. With
``_pml_is_active(pml)`` False, two remain::

    elif condfac is not None:
        _apply_conductive_update(target, curl, condfac, fields.condinv_for(...))
    else:
        target -= curl                                     # stepping.py:539

and ``_apply_conductive_update`` (stepping.py:1985-1998) is three in-place
statements, MEEP ``step_generic.cpp:87-97``::

    field *= condfac
    field -= curl
    field *= condinv

THE CURL ITSELF IS UNCHANGED -- the term table, the ghost rule with its Bloch
rotation on the wrapped lane, the plane-wise grouping, the ownership masks. That
is not a claim about this file, it is a property of how it is BUILT:
:func:`_curl_source` takes ``complex_emitter._CURL_TEMPLATE`` -- the certified
complex PML curl's own string -- replaces the SIGNATURE and the three tail calls,
and then ASSERTS that the three stencil lines survived unchanged and that no
``pml_apply``, ``kms_``, ``sinv_`` or auxiliary token remains. A reader can diff
the emitted text against ``complex_emitter``'s and see that nothing in the
arithmetic moved; :func:`_curl_source` is what makes that true by construction
rather than by care.

THE PRELUDE IS THE SIBLING'S, VERBATIM AND WHOLE, including ``pml_apply`` and
``constitutive_apply``, which are DEAD CODE in every kernel below. Forking it to
drop them would be the silent divergence ``no_pml_curl._sibling_prelude`` exists
to prevent; carrying them turns the liability into evidence, because the gate
arms every mutation of both as a MUST-BE-UNCAUGHT null and their silence is the
measurement that this family really does not route through those recurrences.
``complex_emitter`` imports nothing at all (stdlib typing only), so it is
imported here rather than parsed out of the source the way ``no_pml_curl`` has to
parse ``step_curl_kernels`` -- one set of bytes either way.

============================================================================
THE ARM IS AN ARGUMENT
============================================================================

Same package boundary as ``complex_pml_kernels``: this module derives no
expansion arm, opens no probe artifact, and re-implements no part of
``triton_kernels.complex_fields.expansion_license``. The gate reads the artifact,
runs the licence, and hands the arm in; the predicates take the VERDICT and check
its shape through :func:`coverage.complex_expansion_refusal`.

``update_P`` NEEDS A PROBE THE SHIPPED FOUR-PATTERN RECORD DOES NOT CARRY, and
that is a refusal rather than a widening. ``xp.multiply(P, c_now, out=scratch)``
(dispersion.py:686) multiplies a complex64 ARRAY by a PYTHON FLOAT, which is
neither ``c8_mul_f4_field_left`` (array coefficient) nor ``python_float_left``
(scalar on the left). The Triton sibling names the missing orientation
``c8_mul_python_float_field_left`` and refuses without it
(``triton_kernels/complex_ade.py:14-24``); :data:`ADE_PROBE_PATTERN` is that name
and :func:`covers_complex_no_pml_ade_update_p` refuses a licence whose record
does not classify it. The four-pattern ``expansion_probe_2026-08-17`` record does
not; the seven-pattern ``triton_welds_2026-08-19/unified_expansion/gate.json``
does, under the ``keep`` policy.

============================================================================
FIVE THINGS DECIDE BIT-IDENTITY HERE
============================================================================

Four of them are ``complex_emitter``'s and hold unchanged because the prelude and
the stencil ARE its bytes: the zero cross terms carried literally, the normative
operand orientations, ``* -1.0f`` rather than unary minus, and the plane-wise
curl grouping ``((sf - f1) + (f2 - ss))``. Two more are this file's own, and both
were MEASURED on NumPy before they were written down here rather than reasoned
about (``test_complex_no_pml.py`` re-measures each and prints the count):

**THE CONDUCTIVE TAIL'S ORDER IS LOAD-BEARING; ITS C SPELLING IS NOT.**
``field *= condfac; field -= curl; field *= condinv`` against the same three
operations reassociated as ``((field - curl) * condfac) * condinv`` differs on
400 000 of 400 000 float32 words on a random fixture. What is NOT a hazard, and
is recorded here because the opposite is the natural guess: writing the three as
one nested C expression is the SAME expression tree with the same three
roundings, and differs on 0 of 400 000 words. So :func:`conductive_apply` uses a
named temporary for readability and the gate arms the REORDERING, not the
inlining.

**THE POLES MAY NOT BE PRE-SUMMED.** ``((D - P0) - P1)`` against
``D - (P0 + P1)`` differs on 50 279 of 400 000 words at two poles. The array path
subtracts one contributor at a time in registration order (fields.py:1102-1104),
so :func:`minus_poles` does, and the gate arms the pre-sum.

The zero-imaginary product is the sibling's fact and is re-measured here for the
two new coefficient sites (``condfac``/``condinv``): the full complex multiply
against a plane-wise ``{re*c, im*c}`` differs on 3 of 8 words over the four
signed-zero patterns, which is why both conductive multiplies call
``mul_field_left`` rather than scaling the planes.

============================================================================
PLATFORM REQUIREMENTS AND NOT-DISPATCHED
============================================================================

* PURE ASCII device strings, no division, no unary minus on a float path -- the
  three requirements ``complex_emitter`` and ``ade_kernels`` adjudicated, and the
  bodies here are within them by inspection and by test.
* ``cupy`` is imported INSIDE the launchers, never at module scope, because the
  predicates are consumed by the coverage census and the census runs on a laptop.
* NOTHING IN ``meep_gpu`` IMPORTS ``cuda_kernels``. A predicate returning True
  here licenses a MEASUREMENT, not a production step. What has and has not been
  measured on a device is :data:`COMPLEX_NO_PML_ADMISSION`; nothing in this file
  may be read as a device verdict while its ``host`` field is None.

============================================================================
WHAT THE DEVICE SAID, AND THE ONE THING IT DID NOT
============================================================================

Measured 2026-08-20 on the GPU host (RTX A6000, index 5 verified empty by UUID,
CuPy 13.5.1, NVRTC 11.6) under the ``ieee_keep_ftz_stripped`` policy: 480 of 480
scored cases bit-identical to ``stepping`` at one launch AND at the 60-launch
budget, over all seven arms, both courants, three value classes, an absent
absorber and an inert layer, 3-D / 2-D / 1-D shapes and zero / one / two poles;
47 armed mutations, 19 CAUGHT and 28 NULL CONFIRMED, none escaped and none
unaccounted; all three planted defects flipped the release verdict. The census
delta is +20 slots (653 -> 673 of 759) and +4 rows covered at every sub-step
(136 -> 140), recomputed as a CONTROLLED comparison over one record.

THE FLUSH POLICY IS OWED AND IS NOT A CLAUSE THIS FILE CAN WIDEN. The gate
REFUSED under ``meep_x86_flush``, by name and before any measurement:
``update_P``'s fifth orientation is classified only by the seven-pattern
``triton_welds_2026-08-19/unified_expansion/gate.json`` record, which was cut
under ``keep``, and an expansion licence is POLICY-CONDITIONAL. Every certified
sibling on this track spans both policies; this family spans one, and closing
that needs a flush-policy probe carrying ``c8_mul_python_float_field_left``.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Sequence, Tuple

from . import complex_emitter
from . import coverage as _coverage

__all__ = (
    "ADE_PROBE_PATTERN",
    "CERTIFIED_KERNELS",
    "COMPLEX_NO_PML_ADMISSION",
    "CURL_KERNELS",
    "KERNEL_KEYS",
    "MAX_POLES",
    "UNCERTIFIED_KERNELS",
    "WHAT_IS_NOT_BUILT",
    "clear_kernel_cache",
    "complex_no_pml_curl_arm",
    "corpus_digest",
    "covers_complex_no_pml_ade_update_p",
    "covers_complex_no_pml_curl",
    "covers_complex_no_pml_stored_e",
    "kernel_name",
    "kernel_source",
    "poles_per_component",
    "set_kernel_source",
    "shipped_kernel_names",
    "step_complex_no_pml_curl",
    "update_E_complex_no_pml_stored",
    "update_P_complex_no_pml",
)

# ---------------------------------------------------------------------------
# The tables
# ---------------------------------------------------------------------------

#: The kernel this family emits for each ``(sub_step, arm)`` key. The keys are
#: what :func:`kernel_source` and the mutation seam take; the values are the
#: ``extern "C"`` names NVRTC is asked for.
KERNEL_KEYS: Dict[str, str] = {
    "step_B": "step_B_no_pml_complex",
    "step_D": "step_D_no_pml_complex",
    "step_B_conductive": "step_B_no_pml_complex_conductive",
    "step_D_conductive": "step_D_no_pml_complex_conductive",
    "update_E": "update_E_no_pml_complex_stored",
    "update_P": "update_P_no_pml_complex",
    "update_P_uniform": "update_P_no_pml_complex_uniform",
}

#: Which keys are curls, and which of the two tails each takes. ``step_D`` has no
#: storage arm because it never reads E: its sources are the B arrays whatever E
#: storage the run has (``Fields.get_H`` returns them without an absorber,
#: fields.py:1164-1187 -- measured ``get_H('Hx') is fields.Bx`` on all six rows).
CURL_KERNELS: Dict[str, Tuple[str, bool]] = {
    "step_B": ("step_B", False),
    "step_D": ("step_D", False),
    "step_B_conductive": ("step_B", True),
    "step_D_conductive": ("step_D", True),
}

#: Compiled pole slots per E component in :func:`_stored_e_source`. Equal to the
#: Triton sibling's ``complex_no_pml_stored_e.MAX_POLES`` so a corpus row with
#: more poles is refused by the same number on both tracks. The four dispersive
#: corpus rows drive ONE pole per component; the other seven slots are bound to
#: the source pointer and never read, exactly as the sibling plan binds them.
MAX_POLES = 8

#: The expansion-probe orientation ``update_P`` needs and the other five kernels
#: do not. See the module header. Spelled identically to
#: ``triton_kernels.complex_ade.COMPLEX_ADE_PROBE_PATTERN`` and pinned equal to it
#: by ``test_complex_no_pml.py``.
ADE_PROBE_PATTERN = "c8_mul_python_float_field_left"

#: The three E components, in ``stepping.E_CONSTITUTIVE_TERMS`` order.
_ELECTRIC = ("Ex", "Ey", "Ez")

#: This sub-step's displacement source per E component (``fields.py:1095``).
_DISPLACEMENT = {"Ex": "Dx", "Ey": "Dy", "Ez": "Dz"}

#: RECORDED 2026-08-27 as ``cuda_complex_no_pml_2026-08-27`` in ``certification.json``; the
#: two halves landed together as the partition test requires. Both legs RELEASED
#: under BOTH float32 subnormal policies, and the SUBJECT module was checked
#: unchanged since those runs before the record was landed.
CERTIFIED_KERNELS = (
    "step_B_no_pml_complex",
    "step_B_no_pml_complex_conductive",
    "step_D_no_pml_complex",
    "step_D_no_pml_complex_conductive",
    "update_E_no_pml_complex_stored",
    "update_P_no_pml_complex",
    "update_P_no_pml_complex_uniform",
)
#: EMPTY SINCE 2026-08-27, when both of the things that had kept the seven above
#: out of ``CERTIFIED_KERNELS`` were closed in one round: the FLUSH leg (its
#: refusal was an ARTIFACT gap -- the seven-pattern expansion record ``update_P``
#: needs existed only under ``keep`` -- closed by cutting that record under
#: ``flush`` with no code change) and the RECORD, landed as
#: ``cuda_complex_no_pml_2026-08-27`` in ``certification.json``, whose
#: ``_why_the_flush_leg_took_until_now`` carries the closure story. The
#: partition test requires every shipped kernel to sit in exactly one tuple, so
#: the emptiness here is a claim the suite checks, not a leftover.
UNCERTIFIED_KERNELS = {}
#: ``--fmad=false`` is CORRECTNESS, not tuning, and identical to every sibling
#: module's tuple. Spelled here rather than imported so that loading this file by
#: path -- which a bit-identity probe does -- cannot pick up a different tuple
#: than the one a gate compiled.
_COMPILE_OPTIONS: Tuple[str, ...] = ("--fmad=false",)

#: Lanes per block, one COMPLEX CELL per lane. ``complex_pml_kernels`` and every
#: certified sibling landed on 256 independently. Every kernel here guards on the
#: element bound and touches only its own cell's output, so a different value
#: changes throughput and no bit.
_THREADS = 256


# ---------------------------------------------------------------------------
# The device source
# ---------------------------------------------------------------------------

#: The four helpers this family adds to the sibling's prelude. Each is one
#: transcribed array-path statement sequence and nothing else.
_NO_PML_HELPERS = r'''
// stepping._apply_curl's FOURTH tail (stepping.py:510) -- the whole sub-step
// tail when the layer is inert and the component carries no conductivity:
//
//     target -= curl
//
// No auxiliary, no coefficient, no recurrence. The certified split-field body
// reaches the same value through fu and a pair of unit coefficient columns; that
// binding is NOT used here, because it is not exact -- with target = -0.0 and
// curl = +0.0 it yields +0.0 where the array path yields -0.0
// (triton_kernels/complex_no_pml_curl.py's measured caveat). Writing the
// subtraction directly has no such pattern.
__device__ __forceinline__ void no_pml_apply(float* f, int idx, cf curl) {
    cf_store(f, idx, cf_sub(cf_load(f, idx), curl));
}

// stepping._apply_conductive_update (stepping.py:1938-1951), MEEP
// step_generic.cpp:87-97, IN ITS IN-PLACE ORDER:
//
//     field *= condfac
//     field -= curl
//     field *= condinv
//
// THE ORDER IS THE ARITHMETIC AND THE SPELLING IS NOT. Reassociating the same
// three operations to ((field - curl) * condfac) * condinv differs on 400000 of
// 400000 float32 words (measured, test_complex_no_pml.py). Writing them as one
// nested expression instead of three statements is the SAME expression tree with
// the same three roundings and differs on 0 of 400000 -- the named temporary
// below is for a reader, not for a rounding.
//
// BOTH MULTIPLIES ARE THE ZERO-IMAGINARY COMPLEX PRODUCT WITH THE FIELD ON THE
// LEFT, never a plane-wise {re*c, im*c}: over the four signed-zero patterns the
// two differ on 3 of 8 words, because the cross term carries the sign of the
// field's OTHER word into an addend that is exactly zero. condfac and condinv
// are float32 volumes (fields.condfac_for / condinv_for, fields.py:832-836) and
// stay float32 under complex storage, exactly as inv_eps does.
__device__ __forceinline__ void conductive_apply(
    float* f, int idx, cf curl, float condfac, float condinv
) {
    cf t = mul_field_left(cf_load(f, idx), condfac);
    t = cf_sub(t, curl);
    cf_store(f, idx, mul_field_left(t, condinv));
}

// Fields.displacement_minus_polarization (fields.py:1079-1105): MEEP's f_minus_p.
// D MINUS EVERY DRIVING P, IN REGISTRATION ORDER, LEFT TO RIGHT -- the array path
// copies D into a scratch and calls state.subtract_into once per contributor
// (fields.py:1102-1104, dispersion.py:693-695), so the poles may NOT be pre-summed.
// MEASURED: ((D - P0) - P1) against D - (P0 + P1) differs on 50279 of 400000
// float32 words on a random two-pole fixture (test_complex_no_pml.py).
//
// np IS A RUNTIME COUNT AND THE UNUSED SLOTS ARE BOUND TO THE SOURCE POINTER,
// never dereferenced. The branch is uniform across the whole launch (np is a
// kernel scalar), so it costs no divergence and moves no bit.
__device__ __forceinline__ cf minus_poles(
    const float* g,
    const float* p0, const float* p1, const float* p2, const float* p3,
    const float* p4, const float* p5, const float* p6, const float* p7,
    int idx, int np
) {
    cf s = cf_load(g, idx);
    if (np > 0) s = cf_sub(s, cf_load(p0, idx));
    if (np > 1) s = cf_sub(s, cf_load(p1, idx));
    if (np > 2) s = cf_sub(s, cf_load(p2, idx));
    if (np > 3) s = cf_sub(s, cf_load(p3, idx));
    if (np > 4) s = cf_sub(s, cf_load(p4, idx));
    if (np > 5) s = cf_sub(s, cf_load(p5, idx));
    if (np > 6) s = cf_sub(s, cf_load(p6, idx));
    if (np > 7) s = cf_sub(s, cf_load(p7, idx));
    return s;
}

// dispersion.PolarizationState.update (dispersion.py:683-688), MEEP
// susceptibility.cpp:251-258, whose array form is three passes over one scratch:
//
//     xp.multiply(p, c_now, out=scratch)   // scratch = P^n * c_now
//     scratch += c_prev * p_prev           //         + c_prev * P^(n-1)
//     scratch += c_drive * (sigma * w)     //         + c_drive * (sigma * W^n)
//
// GROUPING: ((p*c_now) + (c_prev*q)) + (c_drive*(s*w)) -- left-associated across
// the two `+=`, with the inner `s*w` formed before c_drive multiplies it.
// ORIENTATION: p*c_now is FIELD-left (the array is the left operand of
// xp.multiply); c_prev*q, s*w and c_drive*(...) are all COEFFICIENT-left.
// Character for character the Triton complex sibling's expression
// (triton_kernels/complex_ade.py:127-140).
__device__ __forceinline__ cf ade_step(cf p, cf q, cf w, float s,
                                       float c_now, float c_prev, float c_drive) {
    cf out = cf_add(mul_field_left(p, c_now), mul_coefficient_left(c_prev, q));
    return cf_add(out, mul_coefficient_left(c_drive, mul_coefficient_left(s, w)));
}
'''  # stepping.py live lines for the frozen device-text citation(s) in this string: 510->539, 1938-1951->1985-1998

#: The plain-tail curl signature. ``__EXTRA__`` is empty here and carries the two
#: conductivity volume triples on the conductive arm.
_CURL_SIGNATURE = r'''
extern "C" __global__ void __NAME__(
    float* __restrict__ f0, float* __restrict__ f1, float* __restrict__ f2,
    const float* __restrict__ g0, const float* __restrict__ g1,
    const float* __restrict__ g2,
__EXTRA__    int nx, int ny, int nz, float dtdx,
    int bc_x, int bc_y, int bc_z,
    int ph_x, int ph_y, int ph_z,
    float pxr, float pxi, float pyr, float pyi, float pzr, float pzi
) {'''

#: The conductive arm's extra parameters. NOT ``__restrict__``, deliberately and
#: for the reason ``complex_emitter``'s inverse-epsilon parameters are not: MEEP
#: allocates a conductivity per (component, direction) and a run whose three
#: targets share one profile would hand the same device pointer three times, and
#: ``restrict`` on mutually aliasing arguments is a promise the caller cannot
#: keep. They are read-only, so nothing is lost but the promise. They are float32
#: VOLUMES indexed by the COMPLEX CELL index and NEVER word-doubled.
_CURL_CONDUCTIVE_PARAMS = (
    "    const float* condfac_0, const float* condfac_1,\n"
    "    const float* condfac_2,\n"
    "    const float* condinv_0, const float* condinv_1,\n"
    "    const float* condinv_2,\n")

#: The two tail spellings, per target index.
_TAIL_PLAIN = "        no_pml_apply(f{0}, idx, curl);"
_TAIL_CONDUCTIVE = ("        conductive_apply(f{0}, idx, curl, "
                    "condfac_{0}[idx], condinv_{0}[idx]);")

#: The one stencil line each of the three per-target blocks must still carry after
#: the substitution. Its survival is what makes "the curl itself is unchanged" a
#: property of the build rather than a claim in a docstring.
_STENCIL_LINE = ("        cf curl = mul_coefficient_left(dtdx, "
                 "cf_add(cf_sub(sf, f_1), cf_sub(f_2, ss)));")

#: Patterns that must NOT survive into a no-absorber curl. The split-field call,
#: the two coefficient stems and the three auxiliary pointers ARE that recurrence;
#: finding one means the tail substitution missed a site. Spelled as regular
#: expressions with an explicit left boundary because ``no_pml_apply(`` contains
#: ``pml_apply(`` as a substring -- a plain ``in`` test fires on this family's own
#: correct tail, which is how this check first read as a failure.
_FORBIDDEN_IN_CURL_BODY = (r"(?<![A-Za-z0-9_])pml_apply\(",
                           r"(?<![A-Za-z0-9_])kms_",
                           r"(?<![A-Za-z0-9_])sinv_",
                           r"(?<![A-Za-z0-9_])u[012](?![A-Za-z0-9_])")

_STORED_E_TEMPLATE = r'''
extern "C" __global__ void update_E_no_pml_complex_stored(
    float* __restrict__ f0, float* __restrict__ f1, float* __restrict__ f2,
    const float* __restrict__ g0, const float* __restrict__ g1,
    const float* __restrict__ g2,
    const float* inv_eps_0, const float* inv_eps_1, const float* inv_eps_2,
__POLE_PARAMS__    int np0, int np1, int np2, int n_elem
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elem) return;

    // stepping.update_E's no-PML store (stepping.py:984-993):
    //     source = fields.displacement_minus_polarization(component)
    //     E[...] = source * fields.inverse_epsilon_for(component)
    // SOURCE ON THE LEFT (stepping.py:984-985, `source * ...`), which is the
    // FIELD-left zero-imaginary product; inv_eps is a float32 volume indexed by
    // the COMPLEX CELL index and never word-doubled (stepping.py:41-50 against
    // fields.py:1203-1204).
    //
    // NO ghost rule, NO ownership mask, NO PHASE and NO f_w, all four transcribed
    // rather than forgotten: this branch writes every cell of the volume, reads no
    // neighbour, and without an absorber there is no auxiliary to keep. Anyone
    // porting the split-field constitutive kernel into this one will reach for the
    // last of them. It does not belong here -- the stored E IS the constitutive
    // product without a layer (fields.py:1150).
    cf_store(f0, idx, mul_field_left(
        minus_poles(g0, a0, a1, a2, a3, a4, a5, a6, a7, idx, np0),
        inv_eps_0[idx]));
    cf_store(f1, idx, mul_field_left(
        minus_poles(g1, b0, b1, b2, b3, b4, b5, b6, b7, idx, np1),
        inv_eps_1[idx]));
    cf_store(f2, idx, mul_field_left(
        minus_poles(g2, c0, c1, c2, c3, c4, c5, c6, c7, idx, np2),
        inv_eps_2[idx]));
}
'''  # stepping.py live lines for the frozen device-text citation(s) in this string: 984-993->1013-1022, 984-985->1013-1014

_UPDATE_P_TEMPLATE = r'''
extern "C" __global__ void __NAME__(
    float* __restrict__ p_out,
    const float* __restrict__ p_now,
    const float* __restrict__ p_prev,
    __SIGMA_PARAM__,
    const float* __restrict__ drive,
    float c_now, float c_prev, float c_drive,
    int n_elem
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elem) return;

    // w IS THE DRIVE FIELD. Fields.drive_field (fields.py:1140-1163) returns
    // f_w_<c> under an active layer and the STORED E without one; this family is
    // the second case, measured (`drive_field('Ex') is fields.Ex` on all four
    // dispersive rows). The predicate pins that identity rather than pinning an
    // array's existence, because the two arrays agree exactly outside an absorber
    // and binding the wrong one is invisible in every no-PML comparison.
    //
    // p_out IS A THIRD BUFFER. The launcher rotates it into the P slot after the
    // launch (dispersion.py:689-691); writing p_now in place would destroy P^n
    // before the next step reads it as P^(n-1) -- a second-order recurrence
    // silently reduced to first order, which is stable, smooth and wrong.
    cf_store(p_out, idx, ade_step(
        cf_load(p_now, idx), cf_load(p_prev, idx), cf_load(drive, idx),
        __SIGMA_READ__, c_now, c_prev, c_drive));
}
'''


def _pole_parameters() -> str:
    """The three ``MAX_POLES``-wide pointer banks, in component order."""
    lines = []
    for stem in ("a", "b", "c"):
        names = [f"const float* {stem}{slot}" for slot in range(MAX_POLES)]
        for start in range(0, MAX_POLES, 4):
            lines.append("    " + ", ".join(names[start:start + 4]) + ",")
    return "\n".join(lines) + "\n"


def _prelude(arm: int) -> str:
    """The sibling's whole prelude plus this family's four helpers.

    ``complex_emitter._HEAD + _ARM_SOURCE[arm] + _TAIL`` is taken WHOLE, including
    ``pml_apply`` and ``constitutive_apply``, which nothing here calls. See the
    module header for why the dead code is carried and how the gate turns it into
    evidence.
    """
    return (complex_emitter._HEAD + complex_emitter._ARM_SOURCE[arm]
            + complex_emitter._TAIL + _NO_PML_HELPERS)


def _curl_source(sub_step: str, conductive: bool, arm: int) -> str:
    """One no-absorber complex curl, built FROM the certified complex PML curl.

    The three edits, and nothing else: the signature is replaced (no auxiliary,
    no coefficient vectors, optionally two conductivity triples), and each of the
    three ``pml_apply`` calls becomes this family's tail. Every stencil line, every
    ghost call, every phase argument and every ownership mask is
    ``complex_emitter._CURL_TEMPLATE``'s own text.

    The assertions are the point. A template rename upstream, a fourth target
    block, or a tail site this substitution missed all raise HERE, at import, with
    the site named -- rather than compiling into a kernel that silently steps the
    split-field recurrence with unit coefficients.
    """
    name = KERNEL_KEYS[sub_step + ("_conductive" if conductive else "")]
    backward = complex_emitter.KERNELS[sub_step][1]
    template = complex_emitter._CURL_TEMPLATE
    marker = "\n) {"
    if template.count(marker) != 1:
        raise RuntimeError(
            "complex_emitter._CURL_TEMPLATE no longer has exactly one signature "
            "terminator; the no-absorber curl is built by replacing its signature "
            "and cannot be built without finding it")
    body = template.split(marker, 1)[1]

    signature = _CURL_SIGNATURE.replace(
        "__EXTRA__", _CURL_CONDUCTIVE_PARAMS if conductive else "")
    tail = _TAIL_CONDUCTIVE if conductive else _TAIL_PLAIN
    substituted, count = re.subn(
        r"        pml_apply\(f(\d), u\1, idx, curl,[^;]*\);",
        lambda match: tail.format(match.group(1)), body)
    if count != 3:
        raise RuntimeError(
            f"expected three pml_apply tail sites in the shared complex curl "
            f"template, replaced {count}; a missed site would step the "
            f"split-field recurrence")
    source = signature + substituted
    source = source.replace("__NAME__", name)
    source = source.replace("__SHIFT__", "cshift_dn" if backward else "cshift_up")
    for target, axes in enumerate(complex_emitter._MASK_AXES[backward]):
        source = source.replace(f"__AXES{target}__", "+".join(axes))
        source = source.replace(
            f"__MASK{target}__", complex_emitter._mask_lines(axes))

    if source.count(_STENCIL_LINE) != 3:
        raise RuntimeError(
            "the three curl stencil lines did not survive the tail substitution; "
            "this family exists to reuse them unchanged")
    for pattern in _FORBIDDEN_IN_CURL_BODY:
        found = re.search(pattern, substituted)
        if found is not None:
            raise RuntimeError(
                f"{found.group(0)!r} survived into a no-absorber curl body; the "
                f"split-field recurrence has no place here")
    if "__" in source.replace("__restrict__", "").replace("__global__", "") \
            .replace("__device__", "").replace("__forceinline__", "") \
            .replace("__fmaf_rn", ""):
        raise RuntimeError("an unsubstituted placeholder survived the curl build")
    return _prelude(arm) + source


def _stored_e_source(arm: int) -> str:
    """``update_E`` with the stored E, the poles subtracted and inv_eps applied."""
    return _prelude(arm) + _STORED_E_TEMPLATE.replace(
        "__POLE_PARAMS__", _pole_parameters())


def _update_p_source(sigma_is_volume: bool, arm: int) -> str:
    """``update_P``, in its volume-sigma and uniform-sigma specializations.

    The two strings differ in ONE parameter declaration and ONE read, exactly as
    ``ade_kernels``' pair does, and for the same reason the choice is compile-time:
    a scalar has no array to bind, so a runtime branch would need a pointer the
    caller does not have. ``coverage.ade_sigma_is_volume`` is the single decider
    the predicate and the launcher both call.
    """
    key = "update_P" if sigma_is_volume else "update_P_uniform"
    body = _UPDATE_P_TEMPLATE.replace("__NAME__", KERNEL_KEYS[key])
    if sigma_is_volume:
        body = body.replace("__SIGMA_PARAM__",
                            "const float* __restrict__ sigma")
        body = body.replace("__SIGMA_READ__", "sigma[idx]")
    else:
        body = body.replace("__SIGMA_PARAM__", "float sigma")
        body = body.replace("__SIGMA_READ__", "sigma")
    return _prelude(arm) + body


def _build(key: str, arm: int) -> str:
    """The device text for one key under one arm; no memo, built per call."""
    if key in CURL_KERNELS:
        sub_step, conductive = CURL_KERNELS[key]
        return _curl_source(sub_step, conductive, arm)
    if key == "update_E":
        return _stored_e_source(arm)
    if key in ("update_P", "update_P_uniform"):
        return _update_p_source(key == "update_P", arm)
    raise ValueError(f"no such kernel key {key!r}; have {sorted(KERNEL_KEYS)}")


#: The gate's MUTATION SEAM: ``{(key, arm): source}`` overrides, consulted by
#: :func:`kernel_source` before it builds. A MUTABLE MODULE GLOBAL on purpose --
#: the source-mutation battery rewrites entries here and re-launches, and
#: :func:`_get_kernel` re-reads it on every call so a rewritten string is seen. A
#: map memoized at first call would hand back the pre-mutation bytes forever,
#: which is "a leg reporting a pass for a mutation it never applied" and which the
#: sibling track hit three times.
_OVERRIDES: Dict[Tuple[str, int], str] = {}


def kernel_name(key: str) -> str:
    """The ``extern "C"`` symbol NVRTC is asked for, by key."""
    if key not in KERNEL_KEYS:
        raise ValueError(f"no such kernel key {key!r}; have {sorted(KERNEL_KEYS)}")
    return KERNEL_KEYS[key]


def kernel_source(key: str, expansion: Any) -> str:
    """The device text for one kernel under one expansion arm.

    ``expansion`` is an arm name or its code and is REQUIRED, never defaulted: a
    wrong arm is a wrong answer rather than a crash -- both arms compile, both
    run, and they differ in the last bits of about a quarter of the words.
    ``complex_emitter.normalized_expansion`` is the one refusal.
    """
    arm = complex_emitter.normalized_expansion(expansion)
    if key not in KERNEL_KEYS:
        raise ValueError(f"no such kernel key {key!r}; have {sorted(KERNEL_KEYS)}")
    override = _OVERRIDES.get((key, arm))
    return override if override is not None else _build(key, arm)


def set_kernel_source(key: str, expansion: Any, source: Optional[str]) -> None:
    """Install (or with ``source=None`` remove) one mutated device string.

    Paired with :func:`clear_kernel_cache`: the compile memo keys on the SOURCE,
    so a mutated body is a miss and reaches NVRTC without the clear, but the clear
    is what makes the accounting -- how many constructions came from the mutated
    bytes -- exact.
    """
    arm = complex_emitter.normalized_expansion(expansion)
    if key not in KERNEL_KEYS:
        raise ValueError(f"no such kernel key {key!r}; have {sorted(KERNEL_KEYS)}")
    if source is None:
        _OVERRIDES.pop((key, arm), None)
    else:
        _OVERRIDES[(key, arm)] = source


def shipped_kernel_names(source: Optional[str] = None):
    """Every kernel NVRTC could be asked to compile, read from the emitted text."""
    if source is not None:
        return set(re.findall(r'extern "C" __global__ void (\w+)\(', source))
    found = set()
    for key in KERNEL_KEYS:
        for arm in complex_emitter.EXPANSIONS.values():
            found |= set(re.findall(r'extern "C" __global__ void (\w+)\(',
                                    _build(key, arm)))
    return found


def corpus_digest() -> str:
    """One sha256 over every source this family can emit, canonically ordered.

    Seven kernels times two arms is fourteen sources; a single changed character
    anywhere in this module or in ``complex_emitter``'s shared prelude moves this
    value. The per-source digests of the arm a gate actually launches belong in
    that gate's artifact.
    """
    import hashlib  # noqa: PLC0415 - stdlib, imported at the one call site

    digest = hashlib.sha256()
    for key in sorted(KERNEL_KEYS):
        for name in sorted(complex_emitter.EXPANSIONS):
            digest.update(f"{key}|{name}".encode("ascii"))
            digest.update(kernel_source(key, name).encode("utf-8"))
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# The predicates
# ---------------------------------------------------------------------------

class _InertLayerProxy:
    """The run's layer, answering ``is_active`` True so the sibling's clauses run.

    Every clause of ``_complex_grid_refusal`` except the absorber one is about the
    grid, the storage and the fields, and none is changed by the absorber's
    absence. Satisfying that one clause AT THE INPUT is what
    ``no_pml_curl._ActiveLayerProxy`` does and for the reason it gives: these
    predicates SHORT-CIRCUIT, so there is no accumulated refusal list a filter
    could subtract from, and forgiving a refusal after the fact would silently
    skip every clause behind it.

    A genuinely absent layer (``pml is None``) still cannot answer a coefficient
    question -- but this family reads no coefficient vector, and neither does the
    shared clause set, so nothing here ever asks.
    """

    __slots__ = ("_layer",)

    def __init__(self, layer: Any) -> None:
        self._layer = layer

    @property
    def is_active(self) -> bool:
        return True

    def __getattr__(self, item: str) -> Any:
        if self._layer is None:
            raise AttributeError(
                f"no absorber object exists on this run, so {item!r} cannot be "
                f"answered; the complex no-absorber predicates ask the shared "
                f"clause set only about the grid, the storage and the fields")
        return getattr(self._layer, item)


class _FeatureMaskedFields:
    """The run's fields with the two facts THIS file answers itself masked off.

    ``_complex_grid_refusal`` refuses a registered susceptibility outright, on the
    grounds that "complex-storage ADE is a future tranche" -- true when it was
    written and false now, since this file builds exactly that. It also carries no
    conductivity clause of its own (the CURL predicate does, per target). Masking
    those two facts at the input lets every OTHER clause -- the inverted storage
    clause, the fold, Dcyl, the boundary kinds, the per-axis phase consistency,
    BFAST, beta, chi2/chi3, the stored-E requirement, the halved int32 bound --
    run verbatim rather than being restated here and drifting.

    THE MASKED FACTS ARE THEN CHECKED POSITIVELY, each in the predicate that owns
    it: the curl predicate reads the conductivity PER TARGET and routes it to the
    arm that implements it, and :func:`covers_complex_no_pml_stored_e` and
    :func:`covers_complex_no_pml_ade_update_p` check every pole buffer they bind.
    The same technique, and the same accounting, as the Triton sibling's
    ``complex_no_pml_conductive._CurlScopeView``.

    Everything else forwards, so a clause added to the shared refusal is inherited
    rather than skipped.
    """

    __slots__ = ("_fields",)

    def __init__(self, fields: Any) -> None:
        object.__setattr__(self, "_fields", fields)

    @property
    def polarizations(self) -> tuple:
        return ()

    @property
    def has_polarizations(self) -> bool:
        return False

    def condfac_for(self, component: str) -> Any:  # noqa: ARG002 - masked by design
        return None

    def __getattr__(self, item: str) -> Any:
        return getattr(object.__getattribute__(self, "_fields"), item)


def _shared_refusal(fields: Any, pml: Any, grid: Any) -> Any:
    """The complex family's shared clause set, with this family's two inversions.

    Inversion 1: the layer must be INERT, checked here rather than inherited --
    the shared clause requires an ACTIVE one and the proxy satisfies it.
    Inversion 2: ``Fields`` must not be in PML STORAGE mode. That is a one-way
    switch (``Fields.enable_pml_storage``, fields.py:678-700) which also changes
    what ``get_H`` returns; a ``Fields`` whose storage was switched on while the
    layer handed to the stepper is inert reads H from the STORED array, which
    ``update_H`` declines to write (stepping.py:944-945), so a curl would
    difference a frozen H. The array path has the same hazard; this refuses the
    configuration rather than reproducing it, exactly as ``no_pml.py:434-437``
    does for the real twin.
    """
    if pml is not None and bool(getattr(pml, "is_active", False)):
        return ("an active absorber is installed: the split-field recurrence is "
                "fused_*_pml_complex_bloch's and this family does not implement it")
    if bool(getattr(fields, "_pml_active", False)):
        return ("Fields is in PML storage mode while the layer is inert: get_H "
                "would serve a stored H that update_H never writes, and this "
                "family writes no f_w")
    return _coverage._complex_grid_refusal(
        _FeatureMaskedFields(fields), _InertLayerProxy(pml), grid)


def complex_no_pml_curl_arm(fields: Any, sub_step: str) -> str:
    """Which curl arm a run takes at this sub-step. A classification, never a refusal.

    ``stepping._apply_curl`` reads ``fields.condfac_for(term.target)`` PER TARGET
    (stepping.py:508), so the arm is a property of the sub-step and not of the run.
    A sub-step whose three targets DISAGREE takes two different tails within one
    launch, which one kernel cannot serve -- that is a refusal and it belongs to
    the predicate, so this raises rather than picking one.
    """
    if sub_step not in _coverage.CURL_SUB_STEPS:
        raise ValueError(
            f"sub_step must be one of {sorted(_coverage.CURL_SUB_STEPS)}, "
            f"got {sub_step!r}")
    reader = getattr(fields, "condfac_for", None)
    if not callable(reader):
        raise ValueError(
            "fields does not expose condfac_for; an unreadable conductivity "
            "table is not an absent one and the arm cannot be classified")
    flags = {reader(component) is not None
             for component in _coverage.CURL_SUB_STEPS[sub_step]}
    if len(flags) != 1:
        raise ValueError(
            f"{sub_step}'s three targets disagree about conductivity; "
            f"_apply_curl would take two different tails in one sub-step. Ask "
            f"covers_complex_no_pml_curl, which refuses this by name.")
    return "conductive" if flags.pop() else "plain"


def _curl_volume_refusal(fields: Any, xp: Any, shape: Tuple,
                         sub_step: str, conductive: bool) -> Any:
    """Every pointer :func:`step_complex_no_pml_curl` binds, checked as bound.

    NOT a copy of the certified complex predicate's array loop. That loop asks for
    the eighteen volumes a PML run has; a no-absorber run has neither the stored H
    nor any ``fu``, and asking for them is the refusal this family exists to stop
    inheriting. The list below is exactly what the launcher passes as a pointer.
    """
    targets = _coverage.CURL_SUB_STEPS[sub_step]
    # THE SOURCE OF step_D IS SPELLED AS B, because that is the array the kernel
    # receives: Fields.get_H returns the B array itself without an absorber
    # (fields.py:1164-1187), and stepping.step_D differences exactly it.
    sources = ("Ex", "Ey", "Ez") if sub_step == "step_B" else ("Bx", "By", "Bz")
    for name in tuple(targets) + tuple(sources):
        problem = _coverage._complex_volume_problem(
            name, getattr(fields, name, None), xp, shape)
        if problem is not None:
            return problem
    # THE AUXILIARY MUST BE ABSENT, not merely unused. This family writes no
    # ``fu``; an ALLOCATED one means something else built it and this step would
    # leave it stale -- a wrong answer on the NEXT sub-step rather than this one.
    for name in targets:
        if getattr(fields, "fu_" + name, None) is not None:
            return (f"fu_{name} is allocated while the layer is inert: this "
                    f"family writes no auxiliary, so something else built it and "
                    f"this step would leave it stale")
    if not conductive:
        return None
    for index, name in enumerate(targets):
        for stem, reader_name in (("condfac", "condfac_for"),
                                  ("condinv", "condinv_for")):
            reader = getattr(fields, reader_name, None)
            if not callable(reader):
                return f"fields does not expose {reader_name}"
            try:
                volume = reader(name)
            except Exception as exc:  # noqa: BLE001 - a raise is not a refusal unless caught
                return f"{reader_name}({name!r}) raised {type(exc).__name__}: {exc}"
            # THE DTYPE DECIDES THE ARITHMETIC AND NOT JUST THE LAUNCH: the array
            # path's ``field *= condfac`` on a float64 table would promote the
            # whole recurrence, so this float32 kernel would be a different
            # computation rather than a different rounding.
            problem = _coverage._array_problem(
                f"{stem} for {name} (slot {index})", volume, xp, shape)
            if problem is not None:
                return problem
    return None


def covers_complex_no_pml_curl(fields: Any, pml: Any, grid: Any, sub_step: str,
                               license: Any = None,
                               subnormal_policy: Any = None) -> tuple:
    """Whether ``step_{B,D}_no_pml_complex[_conductive]`` may serve this run.

    ``sub_step`` is ``"step_B"`` or ``"step_D"``. Returns ``(covered, reason)``
    naming the FIRST refusal, this package's convention.

    THE SUB-STEP ARGUMENT IS REQUIRED, for the reason every curl predicate here
    requires it and one more: the conductivity is read PER TARGET COMPONENT, so a
    D-side conductivity routes ``step_D`` to the conductive tail and leaves
    ``step_B`` the plain one. It ALSO selects which of the two kernels is asked
    about, which is why :func:`complex_no_pml_curl_arm` is a classification rather
    than a clause.

    DISPERSION IS ADMITTED HERE and refused by the certified complex twin. That is
    not a widening of the twin: the twin's own reason is that "complex-storage ADE
    is a future tranche ... a curl admitted into a step whose update_E and update_P
    are both refused buys a sub-step nothing can compose with". This file builds
    both of those, so the premise is gone. The curl reads no ``P``: it differences
    the stored E that ``update_E`` wrote (stepping.py:2438-2453), and dispersion
    changes the values in it and not the recurrence -- which is exactly why the
    REAL curl predicate admits dispersion too.

    A MIXED sub-step -- one lossy target and two lossless -- IS REFUSED, because
    ``_apply_curl`` would take two tails inside one sub-step and neither kernel
    implements the pair.
    """
    if sub_step not in _coverage.CURL_SUB_STEPS:
        raise ValueError(
            f"sub_step must be one of {sorted(_coverage.CURL_SUB_STEPS)}, "
            f"got {sub_step!r}")
    refusal = _coverage.complex_expansion_refusal(license, subnormal_policy)
    if refusal is not None:
        return False, refusal
    refusal = _shared_refusal(fields, pml, grid)
    if refusal is not None:
        return False, refusal
    reader = getattr(fields, "condfac_for", None)
    if not callable(reader):
        return False, ("fields does not expose condfac_for; an unreadable "
                       "conductivity table is not an absent one")
    flags = []
    for component in _coverage.CURL_SUB_STEPS[sub_step]:
        try:
            flags.append(reader(component) is not None)
        except Exception as exc:  # noqa: BLE001 - a raise is not a refusal unless caught
            return False, (f"condfac_for({component!r}) raised "
                           f"{type(exc).__name__}: {exc}")
    if len(set(flags)) != 1:
        return False, (f"{sub_step}'s three targets disagree about conductivity "
                       f"({dict(zip(_coverage.CURL_SUB_STEPS[sub_step], flags))}); "
                       f"_apply_curl takes two different tails in one sub-step and "
                       f"neither kernel implements the pair")
    conductive = flags[0]
    xp, _ = _coverage._backend(grid)
    facts, unreadable = _coverage._grid_facts(grid)
    if unreadable is not None:
        return False, unreadable
    problem = _curl_volume_refusal(fields, xp, facts["shape"], sub_step, conductive)
    if problem is not None:
        return False, problem
    return True, "covered"


def poles_per_component(fields: Any) -> Dict[str, Tuple[Any, ...]]:
    """The states driving each E component, in ``fields.polarizations`` order.

    REGISTRATION ORDER IS BIT-LOAD-BEARING: the array path subtracts contributors
    one at a time in list order (fields.py:1096-1104), and float32 subtraction is
    not commutative across three or more operands.
    """
    states = tuple(getattr(fields, "polarizations", ()) or ())
    order: Dict[str, Tuple[Any, ...]] = {}
    for component in _ELECTRIC:
        driving = []
        for state in states:
            drives = getattr(state, "drives", None)
            if callable(drives) and drives(component):
                driving.append(state)
        order[component] = tuple(driving)
    return order


def covers_complex_no_pml_stored_e(fields: Any, pml: Any, grid: Any,
                                   license: Any = None,
                                   subnormal_policy: Any = None) -> tuple:
    """Whether ``update_E_no_pml_complex_stored`` may serve ``update_E``.

    THE OFF-DIAGONAL ROW IS REFUSED BY NAME and it is the largest thing this
    family did not build: two of the six corpus rows carry
    ``has_offdiagonal_epsilon``, which turns ``update_E`` into MEEP's tensor row
    product (stepping.py:1001-1008) -- a four-point transverse average with
    partner-axis shifts, a Bloch rotation on each shifted operand and wall masks.
    That is a different kernel, not a different coefficient.

    A CONDUCTIVITY IS ADMITTED, on the certified twin's own measured grounds:
    ``fields.condfac_for`` is read in ``stepping._apply_curl`` (stepping.py:508)
    and nowhere else in the module. ``update_E`` never mentions it, so refusing it
    here would be a refusal with no line behind it.

    A RUN WITH NO POLE IS ADMITTED and is not a no-op: with ``stores_E`` True the
    array path still writes ``E[...] = D * inv_eps`` (stepping.py:1022), which is
    this kernel with every ``np`` zero. What IS refused is ``stores_E`` False,
    where ``update_E`` returns at stepping.py:983-984 -- that is
    ``no_pml_constitutive``'s null arm, and admitting it here would put a launch
    where the array path does nothing.
    """
    refusal = _coverage.complex_expansion_refusal(license, subnormal_policy)
    if refusal is not None:
        return False, refusal
    refusal = _shared_refusal(fields, pml, grid)
    if refusal is not None:
        return False, refusal
    if getattr(fields, "has_offdiagonal_epsilon", False):
        return False, ("off-diagonal chi1inv: update_E is MEEP's tensor row "
                       "product, which reads the other components' volumes "
                       "through a four-point transverse average; this family is "
                       "element-wise and does not build it")
    for name in _ELECTRIC:
        if getattr(fields, "f_w_" + name, None) is not None:
            return False, (f"f_w_{name} is allocated while the layer is inert: "
                           f"this family writes no auxiliary and the stored E IS "
                           f"the constitutive product without a layer "
                           f"(fields.py:1150), so an allocated one means "
                           f"something else built it and this step would leave "
                           f"it stale")
    xp, _ = _coverage._backend(grid)
    facts, unreadable = _coverage._grid_facts(grid)
    if unreadable is not None:
        return False, unreadable
    shape = facts["shape"]
    for name in tuple(_ELECTRIC) + tuple(_DISPLACEMENT[c] for c in _ELECTRIC):
        problem = _coverage._complex_volume_problem(
            name, getattr(fields, name, None), xp, shape)
        if problem is not None:
            return False, problem
    reader = getattr(fields, "inverse_epsilon_for", None)
    if not callable(reader):
        return False, "fields does not expose inverse_epsilon_for"
    for component in _ELECTRIC:
        try:
            volume = reader(component)
        except Exception as exc:  # noqa: BLE001 - a raise is not a refusal unless caught
            return False, f"inverse_epsilon_for({component!r}) raised {exc!r}"
        if volume is None:
            return False, f"inverse_epsilon_for({component!r}) is None"
        if not getattr(volume, "shape", ()):
            return False, (f"inverse_epsilon_for({component!r}) is a scalar, not "
                           f"a volume")
        problem = _coverage._array_problem(
            f"inverse_epsilon_for({component!r})", volume, xp, shape)
        if problem is not None:
            return False, problem
    order = poles_per_component(fields)
    for component, states in order.items():
        if len(states) > MAX_POLES:
            return False, (f"{component} is driven by {len(states)} poles, more "
                           f"than the kernel's MAX_POLES={MAX_POLES} slots")
        for index, state in enumerate(states):
            kind = getattr(getattr(state, "susceptibility", None), "kind", None)
            if kind not in _coverage.COVERED_SUSCEPTIBILITY_KINDS:
                return False, (f"{component} pole {index}: kind {kind!r} is "
                               f"outside {_coverage.COVERED_SUSCEPTIBILITY_KINDS}")
            array = (getattr(state, "P", {}) or {}).get(component)
            problem = _coverage._complex_volume_problem(
                f"{component} pole {index} P", array, xp, shape)
            if problem is not None:
                return False, problem
    return True, "covered"


def covers_complex_no_pml_ade_update_p(fields: Any, pml: Any, grid: Any,
                                       license: Any = None,
                                       subnormal_policy: Any = None) -> tuple:
    """Whether the complex ADE kernels may serve the WHOLE ``update_P`` sub-step.

    ALL OR NOTHING OVER EVERY STATE AND EVERY DRIVEN COMPONENT, and that is a
    property of the sub-step rather than a conservatism.
    ``PolarizationState.update`` rotates ONE SHARED SCRATCH buffer from component
    to component inside a single call (dispersion.py:689-691), so covering two of
    three components and leaving the third to the array path would interleave two
    rotations over one buffer set. Sub-steps compose; halves of one do not.

    THE PROBE CLAUSE IS THIS FAMILY'S OWN. ``xp.multiply(P, c_now, out=scratch)``
    is a complex64 array times a PYTHON FLOAT, an orientation the four-pattern
    complex probe does not classify. See :data:`ADE_PROBE_PATTERN`.
    """
    refusal = _coverage.complex_expansion_refusal(license, subnormal_policy)
    if refusal is not None:
        return False, refusal
    refusal = _ade_probe_refusal(license)
    if refusal is not None:
        return False, refusal
    refusal = _shared_refusal(fields, pml, grid)
    if refusal is not None:
        return False, refusal
    xp, _ = _coverage._backend(grid)
    facts, unreadable = _coverage._grid_facts(grid)
    if unreadable is not None:
        return False, unreadable
    shape = facts["shape"]
    try:
        states = tuple(getattr(fields, "polarizations", ()) or ())
    except Exception as exc:  # noqa: BLE001 - unreadable is not empty
        return False, f"the polarization list is unreadable ({exc!r})"
    if not states:
        return False, "no polarization is registered; update_P is a no-op"
    driven_total = 0
    for index, state in enumerate(states):
        driven = getattr(state, "driven", None)
        try:
            components = tuple(driven()) if callable(driven) else ()
        except Exception as exc:  # noqa: BLE001 - malformed is a refusal
            return False, f"polarization {index}: driven() raised {exc!r}"
        driven_total += len(components)
        for component in components:
            covered, reason = _ade_component(fields, grid, xp, shape,
                                             state, component)
            if not covered:
                return False, f"polarization {index} {component}: {reason}"
    if driven_total == 0:
        return False, "no driven component exists; update_P is a no-op"
    return True, "covered"


def _ade_probe_refusal(license: Any) -> Any:
    """Why this licence may not bind an arm for ``update_P``; None if it may.

    The shared shape check is :func:`coverage.complex_expansion_refusal`'s and has
    already run. What is added here is a question about the RECORD behind the
    verdict: it must classify the fifth orientation. A verdict that carries no
    readable pattern table is refused rather than assumed complete -- an
    unmeasured orientation is not a measured one.
    """
    if not isinstance(license, dict):
        return None  # the shared clause already refused a non-verdict
    patterns = license.get("patterns")
    if patterns is None:
        record = license.get("record")
        patterns = record.get("patterns") if isinstance(record, dict) else None
    if not isinstance(patterns, dict):
        return (f"the expansion licence carries no readable pattern table, so "
                f"{ADE_PROBE_PATTERN!r} cannot be shown to have been measured; "
                f"update_P multiplies a complex64 array by a Python float and "
                f"that orientation is neither of the two the four-pattern probe "
                f"classifies")
    verdict = patterns.get(ADE_PROBE_PATTERN)
    if verdict is None:
        return (f"the expansion probe does not classify {ADE_PROBE_PATTERN!r}; "
                f"xp.multiply(P, c_now) is a complex64 array times a Python "
                f"float (dispersion.py:686) and an unmeasured orientation may "
                f"not be inferred from a measured one")
    if verdict not in complex_emitter.EXPANSIONS:
        return (f"the expansion probe classifies {ADE_PROBE_PATTERN!r} as "
                f"{verdict!r}, which names no emittable arm")
    if verdict != license.get("arm"):
        return (f"the expansion probe classifies {ADE_PROBE_PATTERN!r} as "
                f"{verdict!r} while the licence binds {license.get('arm')!r}; "
                f"one kernel cannot be compiled under two arms")
    return None


def _ade_component(fields: Any, grid: Any, xp: Any, shape: Tuple,
                   state: Any, component: str) -> tuple:
    """Whether the complex ADE kernels may advance ONE ``(state, component)``."""
    if component not in _ELECTRIC:
        return False, f"component {component!r} is outside {_ELECTRIC}"
    drives = getattr(state, "drives", None)
    try:
        if not (callable(drives) and drives(component)):
            return False, f"this susceptibility does not drive {component}"
    except Exception as exc:  # noqa: BLE001 - a raise is not a refusal unless caught
        return False, f"drives({component!r}) raised {exc!r}"
    kind = getattr(getattr(state, "susceptibility", None), "kind", None)
    if kind not in _coverage.COVERED_SUSCEPTIBILITY_KINDS:
        return False, (f"kind {kind!r} is outside "
                       f"{_coverage.COVERED_SUSCEPTIBILITY_KINDS}")
    problem = _coverage._ade_coefficient_problem(state)
    if problem is not None:
        return False, problem

    # The three rotating buffers, complex64 here rather than float32.
    buffers = {
        f"P[{component!r}]": (getattr(state, "P", {}) or {}).get(component),
        f"P_prev[{component!r}]": (getattr(state, "P_prev", {}) or {}).get(component),
        "_scratch": getattr(state, "_scratch", None),
    }
    for label, array in buffers.items():
        problem = _coverage._complex_volume_problem(label, array, xp, shape)
        if problem is not None:
            return False, problem
    addresses = {}
    for label, array in buffers.items():
        address = _coverage._base_address(array)
        if address is None:
            return False, (f"{label} exposes no readable base address; the "
                           f"rotation cannot be shown to be alias-free")
        if address in addresses:
            return False, (f"{label} aliases {addresses[address]}; the recurrence "
                           f"reads P and P_prev while writing the scratch, and the "
                           f"rotation that follows would advance one buffer twice")
        addresses[address] = label

    # THE DRIVE. Without an absorber ``Fields.drive_field`` returns the STORED E
    # (fields.py:1140-1163), which is the constitutive product there. This is an
    # IDENTITY check and not an existence check, for the reason
    # ``coverage._ade_drive_problem`` gives from the other side: the two candidate
    # arrays agree exactly outside a layer, so binding the wrong one is invisible
    # in every no-PML comparison and wrong only where it matters.
    reader = getattr(fields, "drive_field", None)
    if not callable(reader):
        return False, "fields does not expose drive_field()"
    try:
        drive = reader(component)
    except Exception as exc:  # noqa: BLE001 - a raise is not a refusal unless caught
        return False, f"drive_field({component!r}) raised {exc!r}"
    stored = getattr(fields, component, None)
    if drive is None or stored is None or drive is not stored:
        return False, (f"drive_field({component!r}) is not the stored {component}; "
                       f"without an active layer it must be, and a family that "
                       f"bound f_w here would be binding an array update_E never "
                       f"wrote")
    problem = _coverage._complex_volume_problem(
        f"drive ({component})", drive, xp, shape)
    if problem is not None:
        return False, problem
    drive_address = _coverage._base_address(drive)
    if drive_address is not None and drive_address in addresses:
        return False, (f"the drive for {component} aliases "
                       f"{addresses[drive_address]}; it is read while the scratch "
                       f"is written")

    sigma = (getattr(state, "sigma", {}) or {}).get(component)
    if sigma is None:
        return False, f"sigma[{component!r}] is missing"
    if _coverage.ade_sigma_is_volume(state, component):
        # FLOAT32 UNDER COMPLEX STORAGE, like inv_eps and the conductivity: sigma
        # is a material coefficient, not a field, and the kernel indexes it by the
        # COMPLEX CELL index without word doubling.
        problem = _coverage._array_problem(f"sigma[{component!r}]", sigma, xp, shape)
        if problem is not None:
            return False, problem
        sigma_address = _coverage._base_address(sigma)
        if sigma_address is not None and sigma_address in addresses:
            return False, (f"sigma[{component!r}] aliases {addresses[sigma_address]}")
    else:
        try:
            number = float(sigma)
        except Exception:  # noqa: BLE001 - neither a scalar nor a volume
            return False, (f"sigma[{component!r}]={sigma!r} is neither a scalar "
                           f"nor a volume")
        if number != number or number in (float("inf"), float("-inf")):
            return False, f"sigma[{component!r}]={number!r} is not finite"
    return True, "covered"


# ---------------------------------------------------------------------------
# The launchers -- cupy is imported HERE, never at module scope
# ---------------------------------------------------------------------------

def clear_kernel_cache() -> int:
    """Drop every memoized kernel; returns how many entries went."""
    from .compile_cache import clear_kernel_cache as _clear  # noqa: PLC0415

    return _clear()


def _word_view(array: Any) -> Any:
    """The float32 word view of one complex64 volume -- the pointer a kernel gets.

    Refuses anything that is not a C-contiguous complex64 volume. A strided view
    would be read in the wrong order by the flat word index: a scrambled volume,
    not a launch failure, which is why this raises rather than reshaping.
    """
    import cupy as cp  # noqa: PLC0415 - a device-only import

    if array is None:
        raise ValueError("expected a complex64 volume, got None")
    if array.dtype != cp.complex64:
        raise ValueError(f"expected a complex64 volume, got dtype {array.dtype}")
    if not array.flags.c_contiguous:
        raise ValueError(
            "complex volume is not C-contiguous; its float32 word view would not "
            "be either, and the kernel indexes it as a flat word array")
    return array.view(cp.float32)


def _get_kernel(key: str, expansion: Any):
    """Compile (or fetch) one kernel, memoized on (key, arm, options, policy, source)."""
    import cupy as cp  # noqa: PLC0415 - a device-only import

    from .compile_cache import (get_or_compile,  # noqa: PLC0415
                                kernel_cache_key)

    arm = complex_emitter.normalized_expansion(expansion)
    code = kernel_source(key, arm)
    name = kernel_name(key)
    cache_key = kernel_cache_key(f"{key}_arm{arm}", True, _COMPILE_OPTIONS, code)
    return get_or_compile(
        cache_key, lambda: cp.RawKernel(code, name, options=_COMPILE_OPTIONS))


def _blocks(count: int) -> int:
    return (int(count) + _THREADS - 1) // _THREADS


def step_complex_no_pml_curl(fields: Any, sub_step: str, expansion: Any,
                             grid: Any = None, *,
                             arm: Optional[str] = None,
                             dtdx: float = None,
                             boundary_codes: Sequence[Any] = None,
                             phase_flags: Sequence[Any] = None,
                             phase_values: Sequence[Any] = None,
                             conductivity: Optional[Tuple[Any, ...]] = None) -> str:
    """Launch one no-absorber complex curl. Returns the kernel name launched.

    SUPPLY ``grid`` and everything derivable is derived HERE, by the same
    functions the predicate asks, which is what makes "the launcher and the
    predicate cannot disagree about which boundary code or which phase this
    sub-step reads" an enforced property rather than a convention. The overrides
    are the GATE'S DOOR and stay, keyword-only: a gate feeds deliberately wrong
    boundary codes and unconjugated backward phases, and a launcher that could not
    be handed its own could not arm those mutations.

    ``conductivity`` is likewise a gate seam: the six volumes normally come from
    ``fields.condfac_for``/``condinv_for``, and a host-mutation leg substitutes
    corrupted ones here so the defect it plants is on the POINTERS a launch binds
    rather than on the fields object every other leg shares.
    """
    import numpy as np  # noqa: PLC0415 - only the scalar types are needed

    from .complex_pml_kernels import (bloch_phase_arguments,  # noqa: PLC0415
                                      complex_boundary_codes)

    if sub_step not in _coverage.CURL_SUB_STEPS:
        raise ValueError(
            f"sub_step must be one of {sorted(_coverage.CURL_SUB_STEPS)}, "
            f"got {sub_step!r}")
    if getattr(fields, "_pml_active", False):
        # THE REFUSAL COMES BEFORE THE COMPILE, so a configuration this family does
        # not serve costs an exception rather than an NVRTC call.
        raise ValueError(
            "Fields is in PML storage mode, so get_H would return the stored H "
            "rather than the B array this family differences; ask "
            "covers_complex_no_pml_curl first -- it refuses this configuration "
            "rather than raising")
    arm = arm or complex_no_pml_curl_arm(fields, sub_step)
    if arm not in ("plain", "conductive"):
        raise ValueError(f"arm must be 'plain' or 'conductive', got {arm!r}")
    key = sub_step + ("_conductive" if arm == "conductive" else "")
    backward = complex_emitter.KERNELS[sub_step][1]

    if grid is not None:
        if boundary_codes is None:
            boundary_codes = complex_boundary_codes(grid)
        derived_flags, derived_values = bloch_phase_arguments(grid, backward)
        if phase_flags is None:
            phase_flags = derived_flags
        if phase_values is None:
            phase_values = derived_values
        if dtdx is None:
            dtdx = grid.dt / grid.dx  # stepping.py:314, :431.
    if boundary_codes is None or phase_flags is None or phase_values is None \
            or dtdx is None:
        raise ValueError(
            "without a grid the caller must supply boundary_codes, phase_flags, "
            "phase_values and dtdx; there is nothing here to derive them from")

    targets = _coverage.CURL_SUB_STEPS[sub_step]
    sources = (("Ex", "Ey", "Ez") if sub_step == "step_B"
               else ("Bx", "By", "Bz"))
    shape = getattr(fields, targets[0]).shape
    arguments = [_word_view(getattr(fields, name)) for name in targets]
    arguments += [_word_view(getattr(fields, name)) for name in sources]
    if arm == "conductive":
        if conductivity is None:
            conductivity = tuple(
                [fields.condfac_for(name) for name in targets]
                + [fields.condinv_for(name) for name in targets])
        if len(conductivity) != 6:
            raise ValueError(
                f"the conductive arm binds three condfac and three condinv "
                f"volumes, got {len(conductivity)}")
        arguments += list(conductivity)
    arguments += [np.int32(shape[0]), np.int32(shape[1]), np.int32(shape[2]),
                  np.float32(dtdx)]
    arguments += [np.int32(code) for code in boundary_codes]
    arguments += [np.int32(flag) for flag in phase_flags]
    arguments += [np.float32(value) for value in phase_values]
    kernel = _get_kernel(key, expansion)
    cells = int(shape[0]) * int(shape[1]) * int(shape[2])
    kernel((_blocks(cells),), (_THREADS,), tuple(arguments))
    return kernel_name(key)


def update_E_complex_no_pml_stored(fields: Any, expansion: Any, *,
                                   poles: Optional[Dict[str, Sequence[Any]]] = None
                                   ) -> str:
    """``stepping.update_E``'s no-PML store for complex storage. Returns the name.

    THE POLE POINTERS ARE RESOLVED HERE, at launch, never cached by a plan.
    ``PolarizationState.update`` rotates ``P``/``P_prev``/``_scratch`` every
    ``update_P``, so a launcher that snapshotted ``state.P[component]`` would be
    reading a retired history from the second step onward -- stale in a way that
    still computes.

    ``poles`` is the gate's door: a mutation leg binds a deliberately reordered or
    truncated bank so the registration-order clause has something to be measured
    against.
    """
    import numpy as np  # noqa: PLC0415 - only the scalar types are needed

    # ``poles`` carries P ARRAYS per component; the default resolves them from the
    # live states, HERE at launch and never from a cached plan, because
    # ``PolarizationState.update`` rotates P/P_prev/scratch on every ``update_P``.
    if poles is None:
        banks_of_arrays = {component: tuple(state.P[component]
                                            for state in states)
                           for component, states in
                           poles_per_component(fields).items()}
    else:
        banks_of_arrays = {component: tuple(poles[component])
                           for component in _ELECTRIC}
    targets = [_word_view(getattr(fields, name)) for name in _ELECTRIC]
    sources = [_word_view(getattr(fields, _DISPLACEMENT[name]))
               for name in _ELECTRIC]
    inverse = [fields.inverse_epsilon_for(name) for name in _ELECTRIC]
    counts = []
    banks = []
    for index, component in enumerate(_ELECTRIC):
        entries = banks_of_arrays[component]
        if len(entries) > MAX_POLES:
            raise ValueError(
                f"{component} is driven by {len(entries)} poles; MAX_POLES="
                f"{MAX_POLES}")
        views = [_word_view(array) for array in entries]
        counts.append(len(views))
        # The unused slots are bound to the SOURCE pointer and never read: the
        # runtime np guard is false for them. Binding None would be a null pointer
        # in the signature; binding a live array keeps every argument valid.
        banks.append(views + [sources[index]] * (MAX_POLES - len(views)))
    shape = getattr(fields, _ELECTRIC[0]).shape
    cells = int(shape[0]) * int(shape[1]) * int(shape[2])
    arguments = targets + sources + inverse
    for bank in banks:
        arguments += bank
    arguments += [np.int32(count) for count in counts] + [np.int32(cells)]
    kernel = _get_kernel("update_E", expansion)
    kernel((_blocks(cells),), (_THREADS,), tuple(arguments))
    return kernel_name("update_E")


def update_P_complex_no_pml(fields: Any, expansion: Any, *, drive=None) -> int:
    """``stepping.update_P`` for complex storage, one launch per driven component.

    THE ROTATION IS THE POINT and it is transcribed from
    ``dispersion.PolarizationState.update`` (dispersion.py:689-691) rather than
    reimplemented, and it happens ONLY AFTER the launch returns, so a raised
    launch leaves the state exactly as it found it rather than half-advanced.

    ``drive`` is the gate's door. It defaults to ``fields.drive_field``, the bound
    method, exactly as ``stepping.update_P`` passes it (stepping.py:1428); a
    control leg binds something else, which must DIVERGE.
    """
    import numpy as np  # noqa: PLC0415 - only the scalar types are needed

    if drive is None:
        drive = getattr(fields, "drive_field", None)
    if not callable(drive):
        raise ValueError(
            "drive must be callable: it is Fields.drive_field, which returns the "
            "stored E without an active layer. This launcher never reaches for a "
            "field itself.")
    launches = 0
    for state in tuple(getattr(fields, "polarizations", ()) or ()):
        components = tuple(state.driven())
        if not components:
            continue
        coefficients = tuple(state._coefficients)
        if len(coefficients) != 3:
            raise ValueError(
                f"a polarization carries {len(coefficients)} recurrence "
                f"coefficients, not the (c_now, c_prev, c_drive) triple this "
                f"kernel bakes as scalars")
        for component in components:
            # Re-read every semantic slot HERE, after the previous component's
            # rotation, never before the loop.
            w = drive(component)
            p = state.P[component]
            p_prev = state.P_prev[component]
            scratch = state._scratch
            sigma = state.sigma[component]
            volume = bool(_coverage.ade_sigma_is_volume(state, component))
            _require_no_aliasing(scratch, p, p_prev, w,
                                 sigma if volume else None, component)
            kernel = _get_kernel("update_P" if volume else "update_P_uniform",
                                 expansion)
            cells = int(scratch.size)
            kernel((_blocks(cells),), (_THREADS,), (
                _word_view(scratch), _word_view(p), _word_view(p_prev),
                sigma if volume else np.float32(sigma), _word_view(w),
                np.float32(coefficients[0]), np.float32(coefficients[1]),
                np.float32(coefficients[2]), np.int32(cells),
            ))
            launches += 1
            # dispersion.py:689-691, verbatim, and only once the launch returned.
            state.P[component] = scratch
            state.P_prev[component] = p
            state._scratch = p_prev
    return launches


def _require_no_aliasing(p_out, p_now, p_prev, drive, sigma, component: str) -> None:
    """The output distinct from every input this launch reads.

    THE PREDICATE ALREADY CHECKED THIS and it is checked again at the launch,
    because the two answer about different moments: the predicate answers about
    the state as it stood when a planner asked, and the rotation moves three names
    between every launch.
    """
    named = [("p_out", p_out), ("p_now", p_now), ("p_prev", p_prev),
             ("drive", drive)]
    if sigma is not None:
        named.append(("sigma", sigma))
    seen = {}
    for label, array in named:
        address = _coverage._base_address(array)
        if address is None:
            raise ValueError(
                f"{label} for {component} exposes no readable base address; an "
                f"unverifiable buffer is not accepted")
        if address in seen:
            raise ValueError(
                f"{label} aliases {seen[address]} for {component}; every field "
                f"pointer in this kernel's signature is __restrict__, and the "
                f"recurrence reads P, P_prev and the drive while writing the "
                f"scratch")
        seen[address] = label


# ---------------------------------------------------------------------------
# WHAT HAS AND HAS NOT BEEN MEASURED
# ---------------------------------------------------------------------------

#: The two slots this family leaves unserved and why, so a later round does not
#: have to rediscover it. Read off the recon artifact, not recalled.
WHAT_IS_NOT_BUILT: dict = {
    "slots": 2,
    "sub_step": "update_E",
    "rows": ("tests/TestMaterialGrid.test_matgrid_3d",
             "tests/TestMaterialGrid.test_subpixel_smoothing"),
    "why": ("both rows carry has_offdiagonal_epsilon=True (measured, "
            "parity/meep_gpu/results/cuda_complex_no_pml_recon_2026-08-20/), so "
            "update_E is MEEP's tensor row product (stepping.py:1001-1008): a "
            "four-point transverse average with partner-axis shifts, a Bloch "
            "rotation on each shifted operand and wall masks. Composing the "
            "complex word-pair addressing with the off-diagonal row is a kernel "
            "neither cuda_kernels/offdiag_emitter.py nor this file has; the "
            "Triton track built it as complex_offdiag_update_e."
            "complex_no_pml_offdiag_update_e_coverage and it is the obvious next "
            "transcription target."),
    "what_it_costs": ("2 slots of 759, and it is why those two rows reach 3 of 4 "
                      "sub-steps rather than 4."),
}

#: WHAT HAS BEEN MEASURED ON A DEVICE FOR THIS FAMILY, AND WHAT HAS NOT.
#:
#: ``host`` is None until a device leg has run. Nothing in this file may be read
#: as a device verdict while it is None --
#: ``test_complex_no_pml.test_admission_record_is_all_or_nothing`` is what keeps
#: the two states from blurring, and it is the same rule
#: ``no_pml_curl.NO_PML_CURL_ADMISSION`` and
#: ``no_pml_constitutive.NULL_CONSTITUTIVE_CONFIRMATION`` carry.
COMPLEX_NO_PML_ADMISSION: dict = {
    "gate": "parity/meep_gpu/gate_cuda_complex_no_pml.py",
    "recon": ("parity/meep_gpu/results/"
              "cuda_complex_no_pml_recon_2026-08-20"),
    "artifacts": "parity/meep_gpu/results/cuda_complex_no_pml_2026-08-21_rename",
    "artifact_sha256": {
        "keep": "7f5d4d5e130050ff63d8983eafd79b63ed089277a4e8b3c5848f8ea4ad733364",
        "flush": "cc969f6c13983b98205bebd24facb074cf3b755e83b0e00c6b17b57cfd320b6f",
    },
    "recorded_utc": "2026-08-21T23:12:19Z",
    "host": "the GPU host",
    "device": ("NVIDIA RTX A6000 (index 5, verified empty by UUID), CuPy 13.5.1, "
               "NVRTC 11.6, driver 12030"),
    "cases_scored": 480,
    "cases_skipped": 0,
    "single_launch_identical": 480,
    "multi_step_identical": 480,
    "multi_step_budget": 60,
    "arms_swept": ("step_B/plain", "step_D/plain", "step_B/conductive",
                   "step_D/conductive", "update_E", "update_P/volume_sigma",
                   "update_P/uniform_sigma"),
    "arm_case_counts": {"step_B/plain": 48, "step_D/plain": 48,
                        "step_B/conductive": 48, "step_D/conductive": 48,
                        "update_E": 144, "update_P/volume_sigma": 96,
                        "update_P/uniform_sigma": 48},
    "courants": (0.5, 0.35),
    "value_classes": ("uniform", "subnormal_band", "signed_zero"),
    "absorber_shapes_swept": ("none", "inert_layer"),
    "shapes_swept": ("3-D", "2-D", "1-D"),
    "walls_swept": ("periodic phased", "periodic unphased", "metallic", "mixed"),
    "poles_swept": (0, 1, 2),
    "expansion_arm": "FMA_V1",
    # THE NON-VACUITY, BY COUNT rather than by class name. A leg whose operands
    # cannot contain the class it claims to test measures nothing and reports a
    # pass while doing it.
    "operand_census": {"words_scored": 7391520, "subnormals": 1154272,
                       "negative_zeros": 329961},
    "cases_with_a_live_bloch_phase": 300,
    "source_mutations_armed": 47,
    "source_mutations_caught": 19,
    "source_mutations_null_confirmed": 28,
    "source_mutations_escaped": 0,
    "source_mutations_unaccounted": 0,
    "host_mutations_caught": ("reverse_conductivity_volumes",
                              "alias_condinv_to_condfac", "reverse_pole_order",
                              "bind_the_stored_E_twice"),
    "host_mutations_null_confirmed": ("identity_rebind",),
    # THE FOUR DEAD-PRELUDE NULLS, and what their silence measures. The shared
    # ``complex_emitter`` prelude compiles ``pml_apply`` and
    # ``constitutive_apply`` into all seven kernels and NONE of them calls
    # either, so these arm at one site each on every kernel and came back
    # UNCAUGHT on every leg. That is the measurement that this family's tails
    # really are ``target -= curl``, the conductive triple, the pole subtraction
    # and the ADE recurrence -- and not the split-field or dsigw recurrences,
    # which is otherwise a claim resting on reading the source.
    "dead_prelude_nulls_confirmed": ("dead_drop_fu_store",
                                     "dead_swap_pml_coefficients",
                                     "dead_constitutive_order",
                                     "dead_constitutive_flatten"),
    "verdict_flips_against_planted_defect": True,
    "planted_defects": ("a_scored_case_diverges",
                        "a_must_be_caught_source_mutation_escapes",
                        "a_must_be_caught_host_mutation_escapes"),
    # THE CONTRACTION GUARD IS CARRIED AND ITS EFFECT IS NOT CLAIMED. The
    # unguarded build was bit-identical to the guarded one on all 240 comparable
    # cases at the inexact courant. ``--fmad=false`` stays because the hazard is
    # real in principle; this run is evidence that NVRTC 11.6 on an A6000 did not
    # take it, and evidence of nothing else.
    "contraction_guard": {"flag": "--fmad=false", "comparable_cases": 240,
                          "unguarded_diverged": 0,
                          "reading": "MEASURED DECORATIVE on this compiler and "
                                     "these bodies"},
    "kernel_source_sha256": {
        "step_B_no_pml_complex":
            "ec2d0d515ce49ed8344dbaac92148655cd1f82eb3cfb3a4539aabe5b154f69f4",
        "step_D_no_pml_complex":
            "ba14579dc827b304a647411089088ffa2bba93d92fb2afba6e04021592523bb9",
        "step_B_no_pml_complex_conductive":
            "d1e1e5ce2187b135981b68bbd52a85707c3862593461b8888462793c1b136418",
        "step_D_no_pml_complex_conductive":
            "0d3f7269863304f1cc84e976f972c429d5dba647bf810f197ffc5b56698dc7dd",
        "update_E_no_pml_complex_stored":
            "8a08eb543387f8e48803b7d423a566134aa8eb285e902ed7260e1be9a1b641dd",
        "update_P_no_pml_complex":
            "e35d178e9f5e70060f4d1e36a98c618c4723092822be296c161efc2e0a278a24",
        "update_P_no_pml_complex_uniform":
            "019f44d6c6add86c06865caac448fa67d2020de02dcd39be7e6c6b2664147dec",
    },
    "corpus_digest":
        "d6ad68452085b2c7afe2bff88150775eecbfa87f0f392dbda6fde9e2048b10cc",
    # THE SLOT DELTA, RECOMPUTED AS A CONTROLLED COMPARISON rather than as a
    # difference of two rounds: the analyzer is run TWICE over THIS round's
    # record, once with ``cuda_complex_no_pml`` in ``UNION_FAMILIES`` and once
    # without, so the delta is one predicate's and nothing else's. The
    # without-column reproduces the closeout round's 653/136 exactly, which is
    # what makes the two commensurable.
    "census": ("parity/meep_gpu/results/"
               "cuda_predicate_coverage_2026-08-20_complex_no_pml"),
    "slots_before": 653,
    "slots_after": 673,
    "slots_gained": 20,
    "rows_covered_at_every_sub_step_before": 136,
    "rows_covered_at_every_sub_step_after": 140,
    "rows_served": ("TestLoadDump.test_load_dump_structure_3d",
                    "TestLoadDump.test_load_dump_structure_sharded_3d",
                    "TestLoadDump.test_load_dump_chunk_layout_sim_3d",
                    "TestLoadDump.test_load_dump_chunk_layout_file_3d",
                    "TestMaterialGrid.test_matgrid_3d (curls only)",
                    "TestMaterialGrid.test_subpixel_smoothing (curls only)"),
    "conductive_share_of_curl_slots": "8 of 12",
    "what_it_does_not_license": (
        "THE FLUSH POLICY -- BY THIS ROUND'S GATE. That gate ran under 'keep' "
        "and REFUSED under 'flush', by name and before any measurement: update_P "
        "needs the seven-pattern expansion record and the only one then on disk "
        "(triton_welds_2026-08-19/unified_expansion/gate.json) was cut under "
        "'keep'. An expansion licence is POLICY-CONDITIONAL and does not "
        "transfer; the refusal is the artifact "
        "results/cuda_complex_no_pml_2026-08-20/flush/. CLOSED 2026-08-27: the "
        "gap was the artifact, not the platform -- the record was cut under "
        "'flush' (carrying c8_mul_python_float_field_left, with a keep control "
        "reproducing the 2026-08-19 pattern table) and both legs RELEASED as "
        "cuda_complex_no_pml_2026-08-27, which is the record that licenses the "
        "flush policy. This round's artifact still refuses it, and this clause "
        "keeps saying so.",
        "the off-diagonal complex update_E -- see WHAT_IS_NOT_BUILT. 2 slots.",
        "a mirror fold, a cylindrical grid, BFAST, special_kz and chi2/chi3, each "
        "refused by a clause inherited from coverage._complex_grid_refusal and "
        "untested here.",
        "a sub-step whose three targets disagree about conductivity: _apply_curl "
        "would take two tails in one sub-step and neither kernel implements the "
        "pair.",
        "stores_E False. All six corpus rows in this family store E (measured), "
        "so the derived-E arm no_pml_curl.py needed has no demand here and is not "
        "built; update_E with stores_E False is no_pml_constitutive's null arm.",
        "any throughput claim: the gate is a correctness gate and times nothing.",
        "any dispatch. Nothing in meep_gpu imports cuda_kernels, so a True verdict "
        "licenses a MEASUREMENT and not a production step.",
    ),
}
