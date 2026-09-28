"""Raw-CUDA Lorentz/Drude polarization recurrence -- the ``update_P`` sub-step.

THE ONLY STRUCTURAL ZERO ON THE BOARD, AND WHY IT WAS ONE. The 2026-08-16
predicate census measured the hand-CUDA track at 180 of 759 slots across three
certified families, and its per-sub-step table reads::

    step_B       46 / 186  admitted
    step_D       45 / 186  admitted
    update_H     46 / 186  admitted
    update_E     27 / 186  admitted
    update_P      0 /  15  admitted   <- NO CUDA KERNEL, NO PREDICATE

(``parity/meep_gpu/results/cuda_predicate_coverage_2026-08-16_offdiag/
analysis.txt``.) Every other zero in that table is a refusal with a clause behind
it. This one was absence: no kernel to admit anything, and -- worse for a
fail-closed track -- no written refusal for the configurations a kernel would
silently mis-serve. This module and ``coverage.covers_real_pml_ade_update_p``
close both halves.

WHERE THAT LEFT THE TABLE, re-derived from the same record on 2026-08-19 by
running the shipped predicates over its per-row ``configuration`` blocks:
``update_P`` now admits 8 of those 15 rows -- 5 of them on the mirror clause
(:data:`coverage.ADE_MIRROR_ADMISSION`), which is a PROJECTION and not part of
what this family's gate ran. In the same round the constitutive pair's fold
refusal was measured and withdrawn, taking ``update_H`` from 46 to 122 and
``update_E`` from 27 to 79. The track reads 308 of 759 slots over the three
byte-gated families, or 316 counting this one.

=============================================================================
WHAT IS TRANSCRIBED, AND FROM WHERE
=============================================================================

The source of truth is ``dispersion.PolarizationState.update``
(dispersion.py:658-691), reached from ``stepping.update_P`` (stepping.py:1389)
and from nowhere else. Its array form is three passes over one scratch buffer::

    xp.multiply(p, c_now, out=scratch)      # scratch = P^n * c_now
    scratch += c_prev * p_prev              #         + c_prev * P^(n-1)
    scratch += c_drive * (sigma * w)        #         + c_drive * (sigma * W^n)

TWO CERTIFIED SIBLINGS ALREADY WRITE THAT AS ONE EXPRESSION, and this file
copies theirs rather than deriving a third:

* Triton ``kernels.ade_update_p`` (``triton_kernels/kernels.py:693-727``), one of
  the five kernels pinned by ``triton_kernels/fingerprints.json``::

      tl.store(p_out + idx,
               ((p * c_now) + (c_prev * q)) + (c_drive * (s * w)), mask=live)

* Metal ``ade_source``'s float32 body (``metal_kernels/ade_update_p.py:117-121``)::

      p_out[idx] = ((p * c_now) + (c_prev * q)) + (c_drive * (s * w));

The two agree character for character on the arithmetic, which is the strongest
evidence available that the grouping is the array path's and not one platform's
accident. The CUDA body below is the same string again.

=============================================================================
FOUR THINGS DECIDE BIT-IDENTITY HERE
=============================================================================

1. THE GROUPING IS LEFT-ASSOCIATED ACROSS THE TWO ``+=``.
   ``((p*c_now) + (c_prev*q)) + (c_drive*(s*w))`` -- NOT
   ``(p*c_now) + ((c_prev*q) + (c_drive*(s*w)))``, and not one fused sum. The
   array path's two ``+=`` statements fix the association, and float32 addition
   is not associative.

2. THE DRIVE PRODUCT IS ``sigma * w`` FIRST, THEN ``c_drive *`` THAT.
   ``scratch += c_drive * (sigma * w)`` (dispersion.py:686) builds the inner
   product first. Distributing ``c_drive`` over it, or reassociating to
   ``(c_drive * sigma) * w``, is a different float32 number wherever the
   intermediate rounds.

3. NO FMA. ``--fmad=false`` is CORRECTNESS here, not tuning, and there are three
   contraction candidates in one line: both additions can absorb the product to
   their left. The option tuple is SPELLED IN THIS FILE rather than imported, for
   the reason both siblings spell theirs -- a bit-identity probe loads these
   modules BY PATH, outside the package, where an import could pick up a
   different tuple than the one a gate compiled.

4. ``p_out`` IS A THIRD BUFFER, NEVER ``p_now``. The array path writes the
   scratch and rotates the names afterwards (dispersion.py:689-691); it never
   updates P in place. Writing in place would destroy ``P^n`` before the NEXT
   step reads it as ``P^(n-1)`` -- a second-order recurrence silently reduced to
   first order, which is stable, smooth and wrong. The rotation is transcribed in
   :func:`update_P_fused_pml_real`, not reimplemented.

=============================================================================
THE TWO SPECIALIZATIONS, AND WHY THE AXIS IS COMPILE-TIME
=============================================================================

``sigma`` is a uniform scalar in most runs and a full volume in a graded one.
Both siblings make that a compile-time axis and so does this file: a scalar has
no array to bind, so a runtime branch would need a pointer to something the
caller does not have. Two device strings, two kernel names, one launcher.

The choice is made by ``coverage.ade_sigma_is_volume`` -- the SAME function the
predicate checks it with, one call site each. Getting the two out of step binds a
scalar where the signature declares a pointer, or the reverse: a wrong answer
rather than a crash, and the sibling gate's mutation m13 is exactly it.

=============================================================================
WHAT THE PREDICATE ADMITS THAT ITS SIBLINGS REFUSE
=============================================================================

A MIRROR FOLD, worth 5 of the corpus's 15 ``update_P`` rows. This sub-step reads
no neighbour and indexes no per-axis coefficient vector, so the stored extent --
the thing a fold changes and the thing the curl and constitutive predicates
refuse it for -- reaches nothing here. ``stepping.update_P`` says so in its own
docstring (stepping.py:1398-1403): "Needs no boundary pass of any kind ... Under
mirror symmetry it therefore runs over the WHOLE stored array with no ownership
mask". ``coverage.ADE_MIRROR_ADMISSION`` carries the count and names the rows.

A CYLINDRICAL AXIS IS STILL REFUSED, on the same line, and the asymmetry is
deliberate: the mirror clause has 5 corpus rows behind it and Dcyl has 0, so
admitting the fold moves a measured number and admitting Dcyl would rest on the
argument alone.

=============================================================================
PLATFORM FACTS THIS FILE DEPENDS ON (CUDA path, adjudicated 2026-08-15/16)
=============================================================================

* SIGNED ZERO: CUDA lowers ``-x`` to ``neg.f32`` and does NOT canonicalize signed
  zeros. No unary minus appears on any float path below, and no zero-padding fold
  is used to collapse the two sigma variants into one -- ``x + 0.0f`` turns
  ``-0.0f`` into ``+0.0f``, which is the hole the off-diagonal module measured.
* DIVISION: none. The one PTX exception on record -- ptxas expanding
  ``div.rn.f32`` into a sequence whose range checks carry ``.FTZ`` in SASS
  regardless of the PTX modifier -- cannot arise in a divide-free family.
* THE DEVICE STRINGS ARE PURE ASCII, a compile requirement rather than a style
  rule: ``cupy.cuda.compiler.compile_using_nvrtc`` writes the source through a
  bare ``open(..., 'w')`` (compiler.py:368), so the bytes go through the
  interpreter's LOCALE encoding -- ASCII under C/POSIX, which is what a
  non-interactive shell on the validation host gets. Two em-dashes in a comment
  killed a sibling kernel at its first launch on 2026-08-15.

=============================================================================
CERTIFIED 2026-08-19. STILL NOT DISPATCHED.
=============================================================================

Both kernels are in ``CERTIFIED_KERNELS`` and ``UNCERTIFIED_KERNELS`` IS EMPTY.
The verdict is ``certification.json`` block ``ade_2026-08-19``, cut by
``parity/meep_gpu/gate_cuda_ade.py`` on the GPU host (NVIDIA RTX A6000, cc 8.6, CuPy
13.5.1, NVRTC 11.6) under BOTH float32 subnormal policies. Per policy:

* 242/242 single-launch cases bit-identical to
  ``dispersion.PolarizationState.update``, split 121/121 volume sigma and
  121/121 uniform sigma -- so BOTH specializations are certified, not one and a
  projection;
* 32/32 at the 60-launch multi-step budget, which is this family's load-bearing
  leg: the ROTATION is what a single launch cannot exercise, and a plan that
  advanced the same buffer twice would be identical on launch one and wrong from
  launch two onward. ``h3_stale_pointers`` is armed for exactly that and comes
  back ``caught_only_multi_step`` (2/4), which is the leg proving the budget is
  the instrument and not decoration;
* the UNGUARDED CONTROL DIVERGES on 0/242 identical -- ``--fmad=false`` is
  load-bearing here, measured rather than assumed;
* 19/19 mutation legs as required, none unmeasurable, every leg disarmed
  afterwards. Among them the WRONG-DRIVE CONTROL (``h1_wrong_drive_stored_E``)
  CAUGHT on 0/4, with its own null (``h2``) UNCAUGHT on 4/4 where the stored E
  and ``f_w`` genuinely coincide -- binding the stored E instead of ``f_w`` is
  "the single most likely silent wrong answer in dispersion"
  (fields.py:1149-1156), and this is the one leg a no-PML test can never supply;
* three mutations that MUST BE UNCAUGHT (``n1``-``n3``: the two commutations and
  the load reorder) came back uncaught, so the battery is not a comparator that
  has degenerated into failing everything;
* the subnormal band is non-vacuous by count: 1 427 880 subnormal operands of
  3 046 432 under the band value class.

WHAT THE VERDICT DOES NOT COVER, counted off the payloads rather than recalled.
THE MIRROR ADMISSION IS ONLY THINLY MEASURED. Of the 242 guarded cases per
policy, exactly TWO are folded (``foldX_16x16x8``: one X plane, periodic
termination, stored 10x16x8, lorentzian, two states driving three components,
uniform value class, Courant 0.5, one per sigma form). Both are bit-identical
under both policies, and the two unguarded folded controls diverge on 2469 and
2432 words of 17920. What carries NO folded case at all:

* the 60-launch multi-step leg -- 0 of 32, so the ROTATION has never been
  advanced on a folded extent;
* the mutation battery -- 0 of its cases, so no defect has been shown to be
  catchable on a fold;
* the subnormal band value class, the inexact Courant, a METALLIC folded
  termination, a Y or Z plane, and more than one plane at once.

So :data:`coverage.ADE_MIRROR_ADMISSION` -- worth 5 of the corpus's 15
``update_P`` rows -- stays a PROJECTION: it is not contradicted by anything that
ran, and it is not carried by it either. Two single-launch cases on one plane
are a smoke test, not the sweep the sibling family got, and the corpus's five
folded rows include 2-D ``TestLoadDump`` runs this shape does not resemble. What
would close it is a folded leg in the shape ``gate_cuda_folded_constitutive.py``
took: every axis folded somewhere, both terminations, both parities, an odd full
count, and the fold reaching the multi-step and mutation legs.

Nothing dispatches this. No module in ``meep_gpu/`` imports ``cuda_kernels`` at
all (``test_package_boundary.py`` pins the absence in both directions); the only
importers are this directory's tests and the parity probes. Certification is a
statement about bytes, not about wiring.
"""

import cupy as cp
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..fields import Fields
    from ..pml import PML


def _sibling(module_name: str):
    """Load a same-directory sibling by path -- the bit-identity probe's route.

    The probe loads these modules OUTSIDE the package, where a relative import
    has no anchor. Every sibling is fetched through this one helper so the
    fallback cannot be written three slightly different ways.
    """
    import importlib.util  # noqa: PLC0415
    import os  # noqa: PLC0415

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        f"{module_name}.py")
    spec = importlib.util.spec_from_file_location(
        f"cuda_kernels_{module_name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The predicate and the compile memo live in CUPY-FREE siblings so they stay
# importable, exercisable and mutable on a machine with no GPU. Imported back
# here because this module is the family's public face.
try:
    from . import compile_cache
    from . import coverage as _coverage
except ImportError:  # loaded by path, outside the package
    compile_cache = _sibling("compile_cache")
    _coverage = _sibling("coverage")

ADE_ELECTRIC_COMPONENTS = _coverage.ADE_ELECTRIC_COMPONENTS
ade_sigma_is_volume = _coverage.ade_sigma_is_volume
covers_real_pml_ade_component = _coverage.covers_real_pml_ade_component
covers_real_pml_ade_update_p = _coverage.covers_real_pml_ade_update_p

#: Byte-identical to the array path with a DEVICE gate verdict behind it. Both
#: names arrived here on 2026-08-19 with ``certification.json`` block
#: ``ade_2026-08-19``; the partition below is what makes that a visible edit
#: rather than a quiet one, and the record test is what makes the block a
#: precondition rather than a courtesy.
CERTIFIED_KERNELS = ("update_P_pml_real", "update_P_pml_real_uniform")

#: Shipped but not gated, naming the gate owed and the certified family that
#: covers the same corpus surface today. Being on this list means: not certified
#: against the array path, not to be wired, and not to be trusted by a reader who
#: found the kernel before the docstring. EMPTY since 2026-08-19.
#:
#: Spelled as a plain dict WITHOUT a type annotation for the reason both siblings
#: record: a partition test reads it off the syntax tree without importing the
#: module -- the only way to read it on a host with no CuPy -- and an annotated
#: assignment is an ``ast.AnnAssign``, which that reader does not match.
UNCERTIFIED_KERNELS = {}

# =============================================================================
# THE DEVICE CODE
# =============================================================================
#
# The two strings differ in ONE parameter declaration and ONE load. They are
# written out rather than emitted from a template because the difference is two
# tokens: a template would hide the only thing a reader needs to compare, and the
# arithmetic line -- the part that must be identical in both -- is what a
# transcription test pins.

_ADE_UPDATE_P_PROLOGUE = r'''
// dispersion.PolarizationState.update (dispersion.py:686-691), MEEP
// susceptibility.cpp:251-258, whose array form is three passes over one scratch:
//
//     xp.multiply(p, c_now, out=scratch)   // scratch = P^n * c_now
//     scratch += c_prev * p_prev           //         + c_prev * P^(n-1)
//     scratch += c_drive * (sigma * w)     //         + c_drive * (sigma * W^n)
//
// so the grouping is ((p*c_now) + (c_prev*q)) + (c_drive*(s*w)): left-associated
// across the two `+=`, with the inner `s*w` formed before c_drive multiplies it.
// Character for character the Triton kernel's expression
// (triton_kernels/kernels.py:728) and the Metal one's
// (metal_kernels/ade_update_p.py:121).
//
// `w` IS THE DRIVE FIELD, NOT THE STORED E. Fields.drive_field (fields.py:1160-1162)
// returns f_w_<c> under an active layer and the stored E without one; they agree
// exactly OUTSIDE the absorber, so binding the wrong one is invisible in every
// no-PML test. The predicate pins the identity of what drive_field returns and
// the launcher binds only what it hands back.
//
// NO OWNERSHIP MASK AND NO GHOST RULE, transcribed rather than forgotten:
// stepping.update_P (stepping.py:1369-1374) reads no neighbour and writes every
// cell of the stored array, folded extent included. Anyone porting the curl
// kernel's shape into this one will reach for both. Neither belongs here.
//
// p_out IS A THIRD BUFFER. The caller rotates it into the P slot after the
// launch; writing p_now in place would destroy P^n before the next step reads it
// as P^(n-1), turning a second-order recurrence into a first-order one that
// still converges.
'''  # stepping.py live lines for the frozen device-text citation(s) in this string: 1369-1374->1398-1403

# SIGMA_IS_VOLUME=1: a graded susceptibility, one coefficient per cell.
_update_P_pml_real_kernel_code = _ADE_UPDATE_P_PROLOGUE + r'''
extern "C" __global__ void update_P_pml_real(
    float* __restrict__ p_out,
    const float* __restrict__ p_now,
    const float* __restrict__ p_prev,
    const float* __restrict__ sigma,
    const float* __restrict__ drive,
    float c_now, float c_prev, float c_drive,
    int n_elem
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elem) return;

    float p = p_now[idx];
    float q = p_prev[idx];
    float w = drive[idx];
    float s = sigma[idx];
    p_out[idx] = ((p * c_now) + (c_prev * q)) + (c_drive * (s * w));
}
'''

# SIGMA_IS_VOLUME=0: a uniform susceptibility. The scalar is passed by value, so
# nothing is loaded per cell; every other token is the variant above.
_update_P_pml_real_uniform_kernel_code = _ADE_UPDATE_P_PROLOGUE + r'''
extern "C" __global__ void update_P_pml_real_uniform(
    float* __restrict__ p_out,
    const float* __restrict__ p_now,
    const float* __restrict__ p_prev,
    float sigma,
    const float* __restrict__ drive,
    float c_now, float c_prev, float c_drive,
    int n_elem
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elem) return;

    float p = p_now[idx];
    float q = p_prev[idx];
    float w = drive[idx];
    float s = sigma;
    p_out[idx] = ((p * c_now) + (c_prev * q)) + (c_drive * (s * w));
}
'''

# NVRTC compile options -- CORRECTNESS, not performance. Identical to all three
# sibling modules' and spelled here rather than imported so that loading this
# file by path (which the probe does) cannot pick up a different tuple than the
# one a gate compiled.
_COMPILE_OPTIONS = ('--fmad=false',)

#: Lanes per block. One element per lane, flat 1-D grid -- the geometry all three
#: certified hand-CUDA families use, and the one ``triton_kernels`` arrived at
#: independently as ``DEFAULT_BLOCK``. The sub-step is element-wise, so there is
#: no tile to shape and no reuse to exploit.
_ADE_THREADS = 256


def _clear_kernel_cache() -> int:
    """Drop every memoized kernel; returns how many entries went.

    Shared memo with all three sibling modules -- the cache is keyed on the
    source string, so neither two modules nor two sigma variants can collide.
    """
    return compile_cache.clear_kernel_cache()


def _get_kernel(sigma_is_volume: bool):
    """Compile on first use, memoized on (name, options, policy, source).

    THE CODE MAP IS REBUILT PER CALL AND THAT IS LOAD-BEARING, for the reason
    every sibling rebuilds its own: the source is part of the memo key, so the
    map has to be read before there is a key to miss on, and the bit-identity
    probe mutates kernels by assigning over the module-level source string. A map
    memoized at first call would hand back the pre-mutation string forever -- a
    leg reporting a pass for a mutation it never applied.
    """
    code_map = {
        True: (_update_P_pml_real_kernel_code,
               'update_P_pml_real'),
        False: (_update_P_pml_real_uniform_kernel_code,
                'update_P_pml_real_uniform'),
    }
    code, func_name = code_map[bool(sigma_is_volume)]
    key = compile_cache.kernel_cache_key(
        func_name, False, _COMPILE_OPTIONS, code)
    return compile_cache.get_or_compile(
        key, lambda: cp.RawKernel(code, func_name, options=_COMPILE_OPTIONS))


def _launch(kernel, p_out, p_now, p_prev, sigma, drive,
            coefficients, sigma_is_volume: bool) -> None:
    """One component, one launch. ``sigma`` is a device volume or a Python float."""
    n_elem = int(p_out.size)
    blocks = (n_elem + _ADE_THREADS - 1) // _ADE_THREADS
    c_now, c_prev, c_drive = coefficients
    sigma_argument = sigma if sigma_is_volume else np.float32(sigma)
    kernel((blocks,), (_ADE_THREADS,), (
        p_out, p_now, p_prev, sigma_argument, drive,
        np.float32(c_now), np.float32(c_prev), np.float32(c_drive),
        np.int32(n_elem),
    ))


def update_P_fused_pml_real(fields: 'Fields', pml: 'PML' = None, *,
                            drive=None) -> int:
    """``stepping.update_P``, one launch per driven component. Returns the count.

    THE ROTATION IS THE POINT OF THIS FUNCTION, and it is transcribed from
    ``dispersion.PolarizationState.update`` (dispersion.py:689-691) rather than
    reimplemented::

        state.P[component]      = scratch     # this step's result
        state.P_prev[component] = p           # what P held
        state._scratch          = p_prev      # the retired history

    POINTERS ARE RESOLVED IMMEDIATELY BEFORE EACH LAUNCH, deliberately. The three
    buffers rotate PER COMPONENT inside one call, and the retired history becomes
    the NEXT component's scratch. A launcher that read ``P``/``P_prev``/
    ``_scratch`` once and reused the views -- the way the certified constitutive
    launcher may cache its coefficient tables, because those never move -- would
    be stale from the second component of the first step, and stale in a way that
    still computes: it would advance one buffer twice and freeze another. A
    single-launch comparison cannot see that. The sibling plan carries the same
    hazard in as many words (``triton_kernels/launch.py:1768-1782``).

    THE ROTATION HAPPENS ONLY AFTER THE LAUNCH RETURNS, so a raised launch leaves
    the state exactly as it found it rather than half-advanced.

    ``pml`` IS ACCEPTED AND UNUSED, which is not an oversight but the array
    path's own signature: ``stepping.update_P`` takes it and deletes it
    (stepping.py:1424), because "the absorber reaches the polarization through W
    and nowhere else". It is taken here so every launcher on this track has the
    same shape, and so the predicate a caller must consult -- which DOES need the
    layer, to know that ``drive_field`` will hand back ``f_w`` -- takes the same
    arguments.

    ``drive`` IS THE GATE'S DOOR and stays, keyword-only. It defaults to
    ``fields.drive_field``, the bound method, exactly as
    ``stepping.update_P`` passes it (stepping.py:1428). A byte gate's control leg
    binds the STORED E instead, which must DIVERGE; a launcher that could not
    take its own drive could not arm that control at all.
    """
    if drive is None:
        drive = getattr(fields, "drive_field", None)
    if not callable(drive):
        raise ValueError(
            "drive must be callable: it is Fields.drive_field, which returns "
            "f_w_<c> under an active layer and the stored E without one. This "
            "launcher never reaches for a field itself.")

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
            volume = bool(ade_sigma_is_volume(state, component))
            _require_no_aliasing(scratch, p, p_prev, w,
                                 sigma if volume else None, component)
            _launch(_get_kernel(volume), scratch, p, p_prev, sigma, w,
                    coefficients, volume)
            launches += 1
            # dispersion.py:689-691, verbatim, and only once the launch returned.
            # Every buffer is owned by exactly one slot at a time, so no two names
            # ever alias.
            state.P[component] = scratch
            state.P_prev[component] = p
            state._scratch = p_prev
    return launches


def _require_no_aliasing(p_out, p_now, p_prev, drive, sigma, component: str) -> None:
    """The output distinct from every input this launch reads.

    THE PREDICATE ALREADY CHECKED THIS and it is checked again at the launch,
    because the two are answering about different moments. The predicate answers
    about the state as it stood when a planner asked; the rotation moves three
    names between every launch, so what it certified is one permutation and this
    is the one being run. The certified off-diagonal launcher carries the same
    belt-and-braces for the same reason.

    Equality of BASE addresses only: two overlapping views with different bases
    pass unseen, the accepted limitation every launcher on this track shares.
    Every pointer in the device signature carries ``__restrict__``, so an alias
    is not merely wrong -- it is undefined behaviour the compiler is licensed to
    exploit.
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
                f"{label} aliases {seen[address]} for {component}; every pointer "
                f"in this kernel's signature is __restrict__, and the recurrence "
                f"reads P, P_prev and the drive while writing the scratch")
        seen[address] = label
