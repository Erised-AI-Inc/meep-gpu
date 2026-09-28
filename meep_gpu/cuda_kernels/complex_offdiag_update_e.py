"""COMPLEX64 storage x OFF-DIAGONAL chi1inv at ``update_E`` -- two hand-CUDA tails.

THE LAST FIVE SLOTS ON THE BOARD, AND THEY ARE ONE ARITHMETIC SHAPE. The
2026-08-20 final census (``results/cuda_predicate_coverage_2026-08-20_final/``)
leaves 6 of 759 predicate slots unserved. Five of them are one intersection --
MEEP's tensor ROW PRODUCT at ``update_E`` under COMPLEX64 storage -- and every
family that was asked refused them on KERNEL SHAPE rather than on a swept range:

* ``cuda_offdiag`` and ``cuda_folded_offdiag``: "complex64 storage: the
  recurrence is the same but the storage is not";
* ``cuda_complex`` and ``cuda_complex_folded``: "off-diagonal chi1inv: the row
  product reads the other components' volumes and this sub-step is element-wise";
* ``cuda_complex_no_pml``: "off-diagonal chi1inv: update_E is MEEP's tensor row
  product ... this family is element-wise and does not build it", recorded in as
  many words as :data:`complex_no_pml_kernels.WHAT_IS_NOT_BUILT` and naming this
  transcription as the obvious next target.

None of those refusals is a widening candidate and none is retired here. They are
all exactly right about the kernels they are about; this module is the kernel
they are not about.

THE FIVE ROWS, MEASURED OFF THE CENSUS RATHER THAN ASSUMED
==========================================================

===========================================  =====  =====  ======  ============
row                                          shape  PML    fold    k_point
===========================================  =====  =====  ======  ============
``examples/solve-cw.py``                     161x161x1  active  X+Y   0
``tests TestArrayMetadata``                  201x201x1  active  X+Y   0
``tests TestHoleyWvgBands.test_fields_at_kx``  20x122x1  active  Y     (3.5, 0, 0)
``tests TestMaterialGrid.test_matgrid_3d``     25x25x25  INERT   none  (.23,-.17,.35)
``tests TestMaterialGrid.test_subpixel_smoothing`` 25x25x1 INERT none  (.3892,.1597,0)
===========================================  =====  =====  ======  ============

All five carry ``force_complex_fields`` True, ``has_offdiagonal_epsilon`` True,
``stores_E`` True, no polarization, no conductivity, beta 0 and no BFAST. Every
folded axis carries k = 0 (``Grid._resolve_bloch`` raises otherwise, grid.py:975)
and the one phased fold row puts its phase on an UNFOLDED axis, which is what
keeps this family inside the fold clause the complex track already measured.

=============================================================================
TWO KERNELS, AND THE SPLIT IS THE TAIL -- MEASURED, NOT ARGUED
=============================================================================

The corpus splits 3 / 2 on the absorber, and the question the brief asked is
whether ONE kernel serves both. It does not, and the reason is not the absorber
profile -- it is that the two tails are DIFFERENT COMPUTATIONS, not one
computation at different coefficients:

* with an active layer ``update_E`` calls ``_apply_constitutive_pml``
  (stepping.py:1015), which READS ``E`` and ``f_w_E``, accumulates onto ``E`` and
  rotates ``f_w_E``;
* with an inert or absent layer it takes ``getattr(fields, component)[...] =
  constitutive`` (stepping.py:1022) -- a plain OVERWRITE that reads neither.

The natural collapse is to bind ``kps = 1``, ``kms = 0`` and a scratch ``f_w``, so
that the recurrence degenerates to the store. IT DOES NOT: the recurrence computes
``E_new = E_old + 1*src - 0*prev``, which is the store only where ``E_old`` is
identically zero, and ``update_E``'s own input ``E`` is the field the previous
step wrote. ``test_complex_offdiag_update_e.py`` measures this on the ARRAY PATH
itself -- one fixture stepped with an inert layer and with a synthetic
unit-coefficient layer -- and reports the differing word count rather than
asserting the reading. So the tail is a STRUCTURAL axis in this package's sense
(different signature, different reads, different writes), and structural axes are
separate kernels here exactly as the certified complex family's four sub-steps
are (``complex_emitter.KERNELS``).

    ``update_E_pml_complex_offdiag``      -- the split-field tail
    ``update_E_no_pml_complex_offdiag``   -- the direct store

THE CORE IS ONE BODY AND IS EMITTED ONCE. Both kernels are built from the same
:data:`_CORE` text with :data:`_TAIL_PML` or :data:`_TAIL_STORE` substituted per
component, so the row product, the ghost rules, the phases, the parity and the
wall mask cannot drift between the two arms; :func:`_component_source` is what
makes that true by construction rather than by care. The Triton sibling
(``triton_kernels/complex_offdiag_update_e.py``) reaches the same shape through a
``PML: tl.constexpr`` axis, which is separate compilation spelled differently.

=============================================================================
WHAT IS TRANSCRIBED, AND FROM WHERE
=============================================================================

``stepping.update_E``'s ``elif offdiagonal:`` branch (stepping.py:1001-1008) with
complex64 storage. With no poles admitted
``displacement_minus_polarization_volumes`` aliases each source to its D primary
(fields.py:1107-1138), and all three are alive at once because the coupling reads
the OTHER components' volumes. Per component ``c`` with own axis ``a``::

    constitutive = D_c * us_c                                # :976, D LEFT
    per surviving partner (offset 1 then 2, cycle X->Y->Z, :1206-1208):
        pair    = g + shift_down(g, partner_axis)            # :1214-1216
        product = pair * coefficient                         # :1217, pair LEFT
        term    = 0.25 * (product + shift_up(product, a))    # :1219-1220, 0.25 LEFT
        total   accumulates term(offset 1) then term(offset 2)  # :1221
    _mask_metallic_wall_coupling(total)                      # :1223, :1250-1254
    constitutive = (D_c * us_c) + total                      # :978-979
    PML  : prev = f_w_c ; f_w_c = constitutive ;
           E_c += kps_a_h * f_w_c ; E_c -= kms_a_h * prev    # :986, :2065-2096
    STORE: E_c = constitutive                                # :993

THE THREE GHOST RULES, each with the branch of ``stepping`` it is:

1. ``_shift_down`` on the PARTNER axis (stepping.py:1835), called WITH a component
   name and the plane's parity (:1243-1245), so all three branches are live:
   PERIODIC wraps the near face and multiplies it by ``conj(phase)`` (:1818-1822
   through :1862); MIRROR writes ``parity * field[_face(axis, MIRROR_ROW)]``
   (:1826-1828) with ``MIRROR_ROW = 2`` (:159); METALLIC writes an exact zero
   (:1830-1831).
2. ``_shift_up`` on the component's OWN axis (stepping.py:1770), called with FOUR
   positional arguments plus the phase (:1248-1249) -- no ``component``, no
   ``mirror_phase``, no ``reflect_row`` -- so the folded-PERIODIC reflect branch
   (:1772-1780) CANNOT FIRE and both mirror terminations fall through to
   ``shifted[_face(axis, -1)] = 0`` (:1781-1783). PERIODIC wraps and multiplies the
   TOP plane by the FORWARD phase.
3. ``_mask_metallic_wall_coupling`` (:1250-1254) zeroes face 0 of every axis whose
   Yee shift is 0 and which is ``is_metallic and not is_mirrored`` -- so it
   ABSTAINS on a fold, deliberately and with the array path's own measurement
   beside it (:1237-1248).

FACTS 1 AND 2 ARE THE REAL FAMILY'S, unchanged: ``folded_offdiag_kernels``
established both against a device on 2026-08-20 (240/240 folded cases per policy),
and neither is a statement about the STORAGE. What complex storage changes is
only what the values are and therefore what each multiply is; the branch structure
is byte-for-byte the real family's.

ONE ``BC_MIRROR`` CODE SERVES BOTH FOLD TERMINATIONS. ``update_E`` has no
ownership mask (``_mask_non_owned_cells``, stepping.py:1912, has exactly three
call sites and no constitutive function is among them) and by fact 2 no reflect
row, so the split ``coverage.REAL_CURL_BC_CODES`` needs for the CURL does not
arise. That is a prediction about the arithmetic and the gate carries it as a NULL
CONTROL -- both declared terminations of one fold must be bit-identical -- rather
than as an assumption.

=============================================================================
WHAT COMPLEX STORAGE CHANGES: FIVE MULTIPLY SITES, EACH WITH AN ORIENTATION
=============================================================================

``numpy``/``cupy`` expose no scalar-times-complex loop (``np.multiply`` carries
only 'FF->F' for complex), so EVERY real-coefficient multiply on the array path is
a FULL complex multiply with a zero-imaginary operand, and the operand ORDER is
normative under the fused arm. Transcribed site by site:

===================================  ==================================  ==========
array-path expression                orientation                         helper
===================================  ==================================  ==========
``source * inverse_epsilon_for(c)``  D on the LEFT (stepping.py:1005)      mul_field_left
``pair * coefficient``               pair on the LEFT (:1246)             mul_field_left
``0.25 * (...)``                     python float on the LEFT (:1219)     mul_coefficient_left
``parity * _mirror_source(...)``     parity on the LEFT (:1826-1828)      mul_coefficient_left
``shifted[plane] *= phase``          field on the LEFT (:1862)            rotate_field_left
``kps * fw`` / ``kms * fw_prev``     coefficient on the LEFT (:2086-2095) mul_coefficient_left
===================================  ==================================  ==========

THE PARITY MULTIPLY NEEDS ITS OWN PROBE PATTERN, and this family therefore takes
the EXTENDED licence. ``_symmetry_phase`` returns a python int, so
``parity * complex64_array`` is the ``c8_mul_c8_parity_coefficient_left``
orientation -- a FIFTH pattern beside the base four
(``triton_kernels/complex_fields.PROBE_PATTERNS``). The arbiter is
``triton_kernels.folded_complex.parity_expansion_license`` and this module does
not re-implement it: the arm arrives as an ARGUMENT and
:func:`covers_complex_offdiag_pml_update_e` takes the licence VERDICT and checks
its shape. Measured on the shipped probe artifact
``results/expansion_probe_2026-08-17/``: BOTH policies classify FMA_V1 on a
measured basis with no refusals, on the base set and on the parity set alike, and
``parity_is_planewise`` is False -- so the plane-wise diagnostic is not a live
alternative here and the full complex product is required.

THE STORE ARM NEEDS ONLY THE BASE FOUR, because no fold reaches it (its predicate
requires an inert layer and the corpus's two store rows are unfolded), but it is
gated on the SAME extended licence anyway: the emitted body carries the mirror arm
either way, and a family whose two arms bound different licences would be two
certifications wearing one name.

=============================================================================
SIX THINGS DECIDE BIT-IDENTITY HERE
=============================================================================

1. THE ZERO CROSS TERMS ARE CARRIED LITERALLY. Plane-wise ``{re*c, im*c}`` is
   byte-wrong on signed zeros; ``complex_emitter``'s note 1 is the derivation and
   this family adds a site of its own -- the parity multiply on the mirror ghost.
2. THE COEFFICIENT MULTIPLY SITS BETWEEN THE TWO SHIFTS. ``u[i]`` multiplies the
   two-point partner average AT ITS OWN NODE and ``u[i+s]`` the average at the
   next node up the component's own axis, because MEEP registers the off-diagonal
   entry at the component's Yee site minus half a cell along its own axis
   (anisotropic_averaging.cpp:248-257, ``here - shift1``). Hoisting to a plain
   four-point average times ``u[i]`` is the same ALGEBRA only for a uniform
   coefficient.
3. THE UP-AXIS PHASE ROTATES THE PRODUCT, NOT THE FIELD. The array path forms
   ``product = pair * coefficient`` over the whole volume and only THEN calls
   ``_shift_up`` on it, so the wrapped plane's Bloch factor multiplies the
   already-multiplied product. Rotating the pre-coefficient field instead is the
   same magnitude and a different number.
4. THE COEFFICIENT AT A ZERO-GHOST UP PLANE IS READ AS ``0.0f``, NOT AS ``u[up]``.
   Where ``_shift_up`` writes an exact zero the array path's second addend is the
   complex ``(+0.0f, +0.0f)``; the kernel reaches that only if the coefficient it
   multiplies the (already zero) far pair by is itself ``+0.0f``. With any other
   value the fused arm's ``fma(z_re, c, (z_im * 0.0f) * -1.0f)`` returns ``-0.0f``
   for negative ``c``, which is a different word. ``ghosted`` is what makes it
   ``0.0f``, and the gate arms the substitution.
5. THE TAIL IS TWO SEPARATE ACCUMULATIONS, LEFT TO RIGHT, on the PML arm
   (stepping.py:2133-2134), and ``prev`` is read BEFORE ``fw`` is written. The
   body is ``complex_emitter``'s certified ``constitutive_apply``, taken from that
   module's own string rather than retyped.
6. NO FMA CONTRACTION. ``--fmad=false`` is CORRECTNESS: the row accumulation, the
   ``D * inv_eps`` product, the two ``pair * coefficient`` products and both tail
   accumulations are contraction candidates. The arm's transcribed fusions are
   spelled ``__fmaf_rn``, a single ``fma.rn.f32`` whatever the flag says, so the
   flag removes the accidental fusions and leaves the transcribed ones.

=============================================================================
WHAT IS COMPILE-TIME AND WHAT IS NOT
=============================================================================

``offdiag_emitter``'s split and ``complex_emitter``'s, unchanged, and for their
reasons:

* the three BOUNDARY codes (now THREE-valued: periodic, metallic, mirror), the
  three WALL flags, the three PHASE flags and the three PARITY weights are BRANCH
  axes -- they select an index, a predicated zero, or whether one multiply happens
  on one plane, and change no float operation and no association -- so they are
  ordinary runtime arguments. The certified ``step_B_pml_real`` already
  takes ``bc_x``/``bc_y``/``bc_z`` that way (step_curl_kernels.py:1687);
* the six ROW-LIVENESS flags are an ARITY axis (they change how many terms the
  inner sum has and therefore its association) so they stay compile-time;
* EXPANSION is an ARITHMETIC axis (which products are fused, how the zero cross
  terms are spelled) so it stays compile-time;
* the TAIL is STRUCTURAL, so it is two kernels.

THE GHOST LANE AND THE PARITY WEIGHT ARE DERIVED FROM ``bc_*`` IN ONE PLACE, which
is ``folded_offdiag_kernels``' rule and its reason: a second spelling of "which
lane is the ghost" is a place for the index redirect and the parity weight to
disagree, and a disagreement is a plane of wrong values rather than a crash.

=============================================================================
PLATFORM FACTS THIS FILE DEPENDS ON
=============================================================================

* ``--fmad=false`` is CORRECTNESS and the option tuple is SPELLED HERE rather than
  imported, so loading this file by path cannot pick up a different one than a
  gate compiled. A test pins it equal to every sibling's.
* SIGNED ZERO: CUDA lowers ``-x`` to ``neg.f32``, NOT to ``0.0f - x``, so it does
  NOT canonicalize signed zeros -- the OPPOSITE of the sibling Triton platform.
  No unary minus appears on any float path here; the parity arrives as a bound
  ``float`` argument, exactly ``+1.0f`` or ``-1.0f``.
* DIVISION: none anywhere on this path, so the one PTX exception on record
  (ptxas expanding ``div.rn.f32`` into a sequence whose range checks carry
  ``.FTZ`` in SASS regardless of the PTX modifier) cannot arise.
* THE DEVICE STRINGS ARE PURE ASCII, a compile requirement rather than a style
  rule: ``cupy.cuda.compiler.compile_using_nvrtc`` writes the source through a
  bare ``open(..., 'w')`` (compiler.py:368), so the bytes go through the
  interpreter's LOCALE encoding -- ASCII under C/POSIX, which is what a
  non-interactive shell on the validation host gets. Two em-dashes in a comment
  killed a sibling kernel at its first launch on 2026-08-15. The tests scan AND
  encode every emitted source, per row mask and per arm.
* COMPLEX64 IS FLOAT2 AND A WALL CLEAR THAT TOUCHES ONLY THE REAL PLANE IS A
  KNOWN DEFECT CLASS IN THIS TREE. Every zero this family writes is ``cf_zero()``
  -- both words -- and the gate arms a real-plane-only clear of the wall mask, of
  the metallic ghost and of the up-axis ghost as three separate mutations.

=============================================================================
IMPORTABLE WITHOUT CUPY, AND NOT CERTIFIED UNTIL THE RECORD SAYS SO
=============================================================================

``cupy`` is imported INSIDE the launchers, never at module scope, on
``no_pml_curl.py``'s precedent and for its reason: the predicates are consumed by
the coverage census, which runs on a laptop. Everything above the launchers --
emitter, predicates, tables -- is stdlib plus ``numpy`` for the phase table and
one lazy ``fields.mirror_parity``.

NOTHING DISPATCHES THIS. No module in ``meep_gpu/`` imports ``cuda_kernels`` at
all (``test_package_boundary.py`` pins the absence in both directions), so a
predicate returning True licenses a MEASUREMENT and not a production step. What
has and has not been measured on a device is
:data:`COMPLEX_OFFDIAG_UPDATE_E_ADMISSION`, and nothing here may be read as a
device verdict while that record's ``host`` field is None.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import complex_emitter
from . import coverage as _coverage
from . import offdiag_emitter as _flat
from .compile_cache import (clear_kernel_cache as _clear_cache,
                            get_or_compile as _get_or_compile,
                            kernel_cache_key as _kernel_cache_key)

__all__ = (
    "ARMS",
    "BC_MIRROR_CODE",
    "CERTIFIED_KERNELS",
    "COMPLEX_OFFDIAG_UPDATE_E_ADMISSION",
    "FOLDED_BC_CODES",
    "KERNEL_NAMES",
    "LIVE_ROW_MASKS",
    "MIRROR_SOURCE_INDEX",
    "PARITY_PROBE_PATTERN",
    "UNCERTIFIED_KERNELS",
    "bloch_phase_table",
    "clear_kernel_cache",
    "complex_offdiag_boundary_codes",
    "complex_offdiag_source",
    "complex_offdiag_tables",
    "corpus_digest",
    "covers_complex_no_pml_offdiag_update_e",
    "covers_complex_offdiag_pml_update_e",
    "kernel_name",
    "kernel_source",
    "mirror_ghost_weights",
    "normalized_row_mask",
    "set_kernel_source",
    "shipped_kernel_names",
    "update_E_complex_no_pml_offdiag",
    "update_E_complex_offdiag_fused_pml",
)

# ---------------------------------------------------------------------------
# The tables the kernel and the host agree on
# ---------------------------------------------------------------------------
#
# Imported BY VALUE wherever a sibling already owns the fact, so there is one home
# per fact and a test can assert the identity rather than diff two spellings.

#: The two arms this family emits, keyed by the tail. ``pml`` binds ``f_w`` and the
#: six HALF-INTEGER coefficient vectors; ``no_pml`` binds neither. The names are
#: deliberately NOT any certified kernel's: a second body wearing a certified name
#: would make ``certification.json``'s partition test and any NVRTC binary
#: observation ambiguous about which arithmetic it saw.
KERNEL_NAMES: Dict[str, str] = {
    "pml": "update_E_pml_complex_offdiag",
    "no_pml": "update_E_no_pml_complex_offdiag",
}

#: The tail keys, in the order every table and every sweep walks them.
ARMS: Tuple[str, ...] = ("pml", "no_pml")

#: ``stepping.MIRROR_SOURCE_INDEX`` (stepping.py:160) -- the stored row a mirror
#: ghost reflects from. Restated here for this package's engine-import-free
#: contract, exactly as ``coverage.py`` restates ``stepping._boundary_kinds``; a
#: test pins the two equal, and pins this equal to
#: ``folded_offdiag_kernels.MIRROR_SOURCE_INDEX``.
MIRROR_SOURCE_INDEX: int = 2

#: The three boundary codes this kernel implements: ``coverage.BC_CODES`` plus one.
#: ``stepping._boundary_kinds`` answers ``"mirror"`` at BOTH terminations of a fold,
#: and ONE code serves both here for the reason the module docstring gives.
#: Numerically identical to ``folded_offdiag_kernels.FOLDED_BC_CODES``; a test pins
#: the two dicts equal so a renumbering cannot happen on one track only.
BC_MIRROR_CODE: int = 2
FOLDED_BC_CODES: Dict[str, int] = dict(_coverage.BC_CODES, mirror=BC_MIRROR_CODE)

#: The components this sub-step writes, with each component's source volume and
#: OWN axis, and the D volume of each axis. Both ``offdiag_emitter``'s.
E_TERMS: Tuple[Tuple[str, str, int], ...] = _flat.E_TERMS
PARTNER_VOLUMES: Tuple[str, str, str] = _flat.PARTNER_VOLUMES
ROW_PARAMETERS: Tuple[str, ...] = _flat.ROW_PARAMETERS

#: Every row mask this family can emit -- the 63 non-empty subsets of the six
#: slots. An all-dead mask belongs to the element-wise complex constitutive
#: families (``covers_real_pml_complex_constitutive(side='E')`` under a layer,
#: ``covers_complex_no_pml_stored_e`` without one), and the seam is the same one
#: the real track partitions on: the install-time zero-row drop
#: (fields.py:1302-1303).
LIVE_ROW_MASKS: Tuple[Tuple[int, ...], ...] = _flat.LIVE_ROW_MASKS

#: The FIFTH probe pattern this family's parity multiply needs. Spelled
#: identically to ``triton_kernels.folded_complex.PARITY_PROBE_PATTERN`` and
#: pinned equal to it by ``test_complex_offdiag_update_e.py``; the licence itself
#: is computed by that module's ``parity_expansion_license`` and never here.
PARITY_PROBE_PATTERN: str = "c8_mul_c8_parity_coefficient_left"

#: NOTHING HERE HAS A GATE VERDICT YET. The two sets partition
#: :func:`shipped_kernel_names`; a kernel added to this family cannot ship
#: described by nothing. Spelled as plain assignments WITHOUT type annotations for
#: the reason every sibling spells its own that way: the partition reader walks the
#: syntax tree without importing the module, and an annotated assignment is an
#: ``ast.AnnAssign`` it does not match.
CERTIFIED_KERNELS = (
    "update_E_pml_complex_offdiag",
    "update_E_no_pml_complex_offdiag",
)

#: BOTH PASSED THEIR GATE UNDER BOTH POLICIES and they sit here anyway, which is a
#: deliberate state and not an oversight. What is missing is the RECORD:
#: ``certification.json`` is the thing a reader is entitled to check a claim
#: against, and "certified" must be a verdict someone can read rather than a word
#: someone typed. See :data:`COMPLEX_OFFDIAG_UPDATE_E_ADMISSION`.
UNCERTIFIED_KERNELS = {}

#: NVRTC compile options -- CORRECTNESS, not performance. Spelled here rather than
#: imported so that loading this file by path cannot pick up a different tuple
#: than the one a gate compiled; a test pins it equal to every sibling's.
_COMPILE_OPTIONS: Tuple[str, ...] = ('--fmad=false',)

#: Lanes per block, one COMPLEX CELL per lane. Every certified sibling landed on
#: 256 independently. Every kernel here guards on the element bound and touches
#: only its own cell's outputs, so a different value changes throughput and no bit.
_THREADS = 256


# =============================================================================
# THE DEVICE CODE
# =============================================================================

#: This family's own prelude. Everything here is either NEW or DIFFERENT from a
#: certified sibling's by exactly the complex word pair, and each piece carries the
#: ``stepping`` line it transcribes. The three multiply helpers, the word-pair
#: addressing, the complex add/sub, ``cf_zero`` and ``constitutive_apply`` all come
#: from ``complex_emitter`` and are NOT respelled here.
_OWN_PRELUDE = r'''
#define BC_PERIODIC 0
#define BC_METALLIC 1
#define BC_MIRROR 2

// stepping.MIRROR_SOURCE_INDEX (stepping.py:159). MEEP symmetry.cpp runs the
// transform on doubled coordinates, so with the halved grid's origin at io = -2
// the ghost at index -1 maps onto stored cell 2 for the iyee = 1 components the
// backward differences read. stepping._mirror_source (stepping.py:1535-1541)
// RAISES below that many stored cells, which is why the predicates refuse a
// folded axis with <= MIRROR_ROW cells by name.
#define MIRROR_ROW 2

// stepping._shift_up (stepping.py:1723) AS _offdiagonal_terms CALLS IT
// (stepping.py:1219-1220): four positional arguments plus the phase, with no
// component, no mirror_phase and no reflect_row -- so the folded-PERIODIC reflect
// branch (:1772-1780) cannot fire and BOTH mirror terminations fall through to
// shifted[_face(axis, -1)] = 0 (:1781-1783). MIRROR therefore shares METALLIC's
// arm here, and that is the transcription rather than a shortcut: a folded axis
// that wrapped instead would move bytes on the last stored plane, which is what
// the 2026-08-20 real-storage device sweep measured for exactly that substitution
// (folded_offdiag_kernels.FOLDED_OFFDIAG_ADMISSION).
__device__ __forceinline__ int coord_up(int a, int n, int bc) {
    if (a + 1 < n) return a + 1;
    return (bc == BC_PERIODIC) ? 0 : -1;
}

// stepping._shift_down (stepping.py:1788), all three branches.
//
// THE MIRROR ARM RETURNS A LIVE INTERIOR ROW. On a folded axis the array path
// writes _symmetry_phase(component, axis, mirror_phase) * _mirror_source(field,
// axis) into face 0 (:1826-1828) -- stored row MIRROR_ROW, weighted by the plane's
// parity. It is neither the periodic wrap nor the metallic zero. The WEIGHT is
// applied by the caller, on the ghost lane alone; see down_sample.
//
// An invariant axis (n == 1, always PERIODIC) needs no case of its own: both
// wraps return the same cell, which is what xp.roll computes there, so the partner
// pair is 2*g rather than an exact zero -- MEEP's stride(d) = 0 double-read.
__device__ __forceinline__ int coord_dn(int a, int n, int bc) {
    if (a > 0) return a - 1;
    if (bc == BC_METALLIC) return -1;
    if (bc == BC_MIRROR) return MIRROR_ROW;
    return n - 1;
}

// The flat COMPLEX CELL index of a cell, with a -1 on ANY axis propagating. The
// two shifts of one term go in opposite directions on DIFFERENT axes, so the
// corner sample is a ghost when either leg is one; the per-axis rules compose
// independently, exactly as they do in the array path where the two shift helpers
// are applied in turn. cf_load does the word doubling and nothing else does.
__device__ __forceinline__ int flat(int i, int j, int k, int nyz, int nz) {
    return (i < 0 || j < 0 || k < 0) ? -1 : (i * nyz + j * nz + k);
}

// The OWN-AXIS UP sample of a partner volume: the value, or the exact complex zero
// the array path writes into that plane. cf_zero() is BOTH WORDS +0.0f, which is
// what assigning the INTEGER 0 to a complex64 array produces (stepping.py:1783);
// a clear that touched only the real plane is a known defect class in this tree
// and the gate arms it.
__device__ __forceinline__ cf up_sample(const float* g, int index) {
    return (index < 0) ? cf_zero() : cf_load(g, index);
}

// The PARTNER-AXIS DOWN sample, with the near-face rule applied ON THE GHOST LANE
// AND NOWHERE ELSE. gl is 1 only at face 0 of the partner axis, and it is derived
// from the coordinate in ONE place so the index redirect in coord_dn and the rule
// here cannot disagree about which lane is the ghost -- a disagreement would read
// stored row MIRROR_ROW without the parity, or apply the parity to an ordinary
// neighbour, and both are a plane of wrong values rather than a crash.
//
//   index < 0        METALLIC: shifted[_face(axis,0)] = 0 (stepping.py:1830-1831).
//   gl && BC_MIRROR  parity * field[_face(axis, MIRROR_ROW)] (:1826-1828) -- the
//                    PARITY ON THE LEFT, which is the c8_mul_c8_parity_coefficient
//                    _left orientation and why this family binds the EXTENDED
//                    expansion licence.
//   gl && ph         the periodic near face wraps DOWN one lattice vector and
//                    carries conj(phase) (:1818-1822 through :1862), FIELD on the
//                    LEFT. The host passes the conjugate in; nothing here negates
//                    anything, because taking the same factor in both directions
//                    is the classic sign error -- every magnitude stays plausible
//                    and only the phase moves.
//   gl && !ph        k = 0: the multiply is SKIPPED, never done against 1 + 0j
//                    (:1819), which is what keeps k = 0 bit-identical to the
//                    unphased engine.
//   !gl              an interior neighbour: no rule at all.
//
// THE MIRROR TEST PRECEDES THE PHASE TEST, and that is the array path's order:
// _shift_down's MIRROR branch never consults the phase. Grid._resolve_bloch raises
// on a phased folded axis (grid.py:975) and both predicates refuse it, so the
// combination is unreachable from a real Grid -- the order is written down so a
// gate feeding deliberate garbage gets the array path's answer rather than a
// guess.
__device__ __forceinline__ cf down_sample(const float* g, int index, int gl,
                                          int bc, int ph, cf phase, float w) {
    if (index < 0) return cf_zero();
    cf z = cf_load(g, index);
    if (!gl) return z;
    if (bc == BC_MIRROR) return mul_coefficient_left(w, z);
    return ph ? rotate_field_left(z, phase) : z;
}

// One partner's OFFDIAG term under complex storage -- MEEP step_generic.cpp:582-583
// as stepping._offdiagonal_terms (stepping.py:1214-1221) associates it:
//
//     0.25*((g[i] + g[i-sx])*u[i] + (g[i+s] + g[(i+s)-sx])*u[i+s])
//
// THE COEFFICIENT MULTIPLY SITS BETWEEN THE TWO SHIFTS. u[i] multiplies the
// two-point partner average AT ITS OWN NODE and u[i+s] the average at the next
// node up this component's own axis, because MEEP registers the off-diagonal entry
// at the component's Yee site minus half a cell along its own axis -- the integer
// node (anisotropic_averaging.cpp:248-257, `here - shift1`). Hoisting to a plain
// four-point average times u[i] is the same ALGEBRA only for a uniform coefficient.
//
// BOTH GHOSTED DOWN LOADS TAKE THE SAME LANE PREDICATE AND THE SAME RULE: `down`
// and `corner` differ only in the OWN-axis coordinate, so the two hit the
// partner-axis ghost plane together.
//
// THE UP-AXIS PHASE ROTATES THE PRODUCT, NOT THE FIELD. The array path forms
// product = pair * coefficient over the WHOLE volume (:1217) and only then calls
// _shift_up on it (:1219-1220), so the wrapped plane's Bloch factor multiplies the
// already-multiplied product. uw is the wrap lane: face -1 of the shifted buffer,
// which is stored cell n-1, and only on a PERIODIC axis.
//
// THE COEFFICIENT AT A ZERO-GHOST UP PLANE IS 0.0f AND NOT u[up]. Where _shift_up
// writes an exact zero the array path's second addend is (+0.0f, +0.0f); the far
// pair is already that, but mul_field_left((+0,+0), c) under the fused arm is
// fma(+0, c, (+0 * 0.0f) * -1.0f) = -0.0f for negative c. Only c == 0.0f returns
// +0.0f, so the coefficient is ghosted too.
//
// 0.25f SCALES THE SUM AND IS APPLIED LAST, with the SCALAR ON THE LEFT: the array
// path writes 0.25 * (...) with a python float (:1219), which is the
// coefficient-left zero-imaginary product and never a plane-wise scale.
//
// NO UNARY MINUS ANYWHERE ON THIS PATH. CUDA lowers -x to neg.f32 rather than to
// 0.0f - x, so it does NOT canonicalize signed zeros the way the sibling Triton
// platform does; that track's negation idiom must not be ported here, and this
// body is written so the answer never matters.
__device__ __forceinline__ cf offdiag_term(
    const float* g, const float* u, int home, int down, int up, int corner,
    int dgl, int dbc, int dph, cf dphase, float w,
    int uw, int uph, cf uphase
) {
    cf near_pair = cf_add(cf_load(g, home),
                          down_sample(g, down, dgl, dbc, dph, dphase, w));
    cf far_pair = cf_add(up_sample(g, up),
                         down_sample(g, corner, dgl, dbc, dph, dphase, w));
    cf near_term = mul_field_left(near_pair, u[home]);
    cf far_term = mul_field_left(far_pair, (up < 0) ? 0.0f : u[up]);
    if (uw && uph) far_term = rotate_field_left(far_term, uphase);
    return mul_coefficient_left(0.25f, cf_add(near_term, far_term));
}
'''  # stepping.py live lines for the frozen device-text citation(s) in this string: 159->160, 1535-1541->1582-1588, 1723->1770, 1219-1220->1248-1249, 1788->1835, 1783->1830, 1830-1831->1877-1878, 1826-1828->1873-1875, 1214-1221->1243-1250

#: The shared kernel body. ``__NAME__``, ``__AUX_PARAMS__``, ``__COEF_PARAMS__``
#: and ``__ROW_PARAMETERS__`` are the signature's four variable pieces;
#: ``__SRC_Ex__``/``__SRC_Ey__``/``__SRC_Ez__`` are the per-component blocks, and
#: THEY are where the two tails differ. Nothing else in this text depends on the
#: arm, which is what makes "the row product is the same body in both kernels" a
#: property of the build.
_CORE = r'''
extern "C" __global__ void __NAME__(
    float* __restrict__ Ex, float* __restrict__ Ey, float* __restrict__ Ez,
__AUX_PARAMS__    const float* Dx, const float* Dy, const float* Dz,
    const float* inv_eps_Ex, const float* inv_eps_Ey, const float* inv_eps_Ez,
__ROW_PARAMETERS__    int nx, int ny, int nz,
__COEF_PARAMS__    int bc_x, int bc_y, int bc_z,
    int wm_x, int wm_y, int wm_z,
    int ph_x, int ph_y, int ph_z,
    float gw_x, float gw_y, float gw_z,
    float dpxr, float dpxi, float dpyr, float dpyi, float dpzr, float dpzi,
    float upxr, float upxi, float upyr, float upyi, float upzr, float upzi
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny * nz) return;

    int nyz = ny * nz;
    int k = idx % nz;
    int j = (idx / nz) % ny;
    int i = idx / (ny * nz);

    // Both neighbour coordinates on every axis: one term reads an axis DOWN (the
    // partner's) and another reads an axis UP (its own), and a component's two
    // terms take different partners, so every axis can be needed either way.
    int di = coord_dn(i, nx, bc_x), ui = coord_up(i, nx, bc_x);
    int dj = coord_dn(j, ny, bc_y), uj = coord_up(j, ny, bc_y);
    int dk = coord_dn(k, nz, bc_z), uk = coord_up(k, nz, bc_z);

    // FACE 0 OF EACH AXIS, spelled ONCE and used for both rules that ask about
    // it: the DOWN ghost lane (the one plane _shift_down rules, stepping.py:
    // 1818-1831) and the wall-plane predicate below (the one plane
    // _mask_metallic_wall_coupling writes, :1254). They are the same plane, so a
    // second spelling would be a place for the ghost rule and the mask to
    // disagree about which lane it is -- a plane of wrong values, not a crash.
    int at_x = (i == 0), at_y = (j == 0), at_z = (k == 0);

    // The UP wrap lane per axis: face -1 of the shifted buffer, which is stored
    // cell n-1, and ONLY on a periodic axis -- the single plane _shift_up phases
    // (:1767-1771). On a metallic or mirrored axis coord_up already returned -1
    // and the sample is the exact zero, so no phase can reach it.
    int uw_x = (i + 1 >= nx) && (bc_x == BC_PERIODIC);
    int uw_y = (j + 1 >= ny) && (bc_y == BC_PERIODIC);
    int uw_z = (k + 1 >= nz) && (bc_z == BC_PERIODIC);

    // The two phase tables, as complex pairs. DOWN carries the CONJUGATE and UP
    // the forward factor; the host computes both and this kernel negates nothing.
    cf dpx; dpx.re = dpxr; dpx.im = dpxi;
    cf dpy; dpy.re = dpyr; dpy.im = dpyi;
    cf dpz; dpz.re = dpzr; dpz.im = dpzi;
    cf upx; upx.re = upxr; upx.im = upxi;
    cf upy; upy.re = upyr; upy.im = upyi;
    cf upz; upz.re = upzr; upz.im = upzi;

    // THE WALL MASK USES at_* ABOVE. FACE 0 ONLY: the high wall is already the
    // shift-up zero ghost, and stepping._mask_metallic_wall_coupling writes only
    // _face(axis, 0) (stepping.py:1254).
    //
    // A FOLDED AXIS IS NEVER ALSO WALL-MASKED: that function asks is_metallic AND
    // NOT is_mirrored (:1253), and coverage.offdiag_wall_mask_flags asks the same
    // question of the same grid, so wm_* is 0 on a folded axis by construction.
    // The launchers refuse the combination rather than relying on that.

    // The three components are independent, and that is read off the loop rather
    // than assumed: the coupling reads D and the inverse permittivity, the tail
    // writes E (and f_w_E on the split-field arm). Disjoint sets, so the order the
    // array path runs them in (stepping.py:969-989) is not a data dependence --
    // which is what makes a NEIGHBOUR read across components safe here.
__SRC_Ex__

__SRC_Ey__

__SRC_Ez__
}
'''  # stepping.py live lines for the frozen device-text citation(s) in this string: 1818-1831->1865-1878, 1254->1283, 969-989->998-1018

#: The split-field arm's extra signature pieces. The three auxiliaries are OUTPUTS
#: and carry ``__restrict__``; the six coefficient vectors are read-only and no
#: field can alias them, so they carry it too.
_AUX_PARAMS = ("    float* __restrict__ f_w_Ex, float* __restrict__ f_w_Ey,\n"
               "    float* __restrict__ f_w_Ez,\n")
_COEF_PARAMS = (
    "    const float* __restrict__ kps_x, const float* __restrict__ kms_x,\n"
    "    const float* __restrict__ kps_y, const float* __restrict__ kms_y,\n"
    "    const float* __restrict__ kps_z, const float* __restrict__ kms_z,\n")

#: The two tails, per component. ``{name}`` is the component, ``{axis}`` its own
#: axis letter and ``{index}`` that axis's loop variable.
#:
#: THE PML TAIL IS complex_emitter's CERTIFIED ``constitutive_apply``, called and
#: not respelled: stepping._apply_constitutive_pml (:2065-2096) with dsigw active,
#: two separate accumulations left to right, ``prev`` read before ``fw`` is
#: written.
#:
#: THE STORE TAIL IS stepping.py:1022 -- ``getattr(fields, component)[...] =
#: constitutive``. NO f_w, NO recurrence and NO coefficient: the stored E IS the
#: constitutive product without a layer (fields.py:1150). Anyone porting the
#: split-field body into this arm will reach for the auxiliary; it does not belong
#: here, and its ABSENCE is a predicate clause rather than an omission.
_TAIL_PML = ("    constitutive_apply({name}, f_w_{name}, idx, src_{name},"
             " kps_{axis}[{index}], kms_{axis}[{index}]);")
_TAIL_STORE = "    cf_store({name}, idx, src_{name});"


# ---------------------------------------------------------------------------
# The emitter
# ---------------------------------------------------------------------------
#
# The three index expressions per term are ASSEMBLED from a per-axis table rather
# than written out eighteen times, and the ghost lane, the parity weight, the
# boundary code and both phase pairs come from the SAME ``partner_axis`` /
# ``own_axis`` numbers that pick the index -- so a term cannot phase one axis while
# shifting another. Eighteen hand-written flat indices is eighteen chances to write
# ``dj`` where ``dk`` belongs, and that defect is a half-cell registration error:
# smooth, converged and wrong.

_HOME_COORDS = ("i", "j", "k")
_UP_COORDS = ("ui", "uj", "uk")
_DOWN_COORDS = ("di", "dj", "dk")
_DOWN_LANES = ("at_x", "at_y", "at_z")
_UP_LANES = ("uw_x", "uw_y", "uw_z")
_BC_NAMES = ("bc_x", "bc_y", "bc_z")
_PHASE_FLAGS = ("ph_x", "ph_y", "ph_z")
_DOWN_PHASES = ("dpx", "dpy", "dpz")
_UP_PHASES = ("upx", "upy", "upz")
_WALL_FLAGS = ("wm_x", "wm_y", "wm_z")
_WALL_PREDICATES = ("at_x", "at_y", "at_z")
_GHOST_WEIGHTS = ("gw_x", "gw_y", "gw_z")
_OWN_AXIS_INDEX = ("i", "j", "k")


def _flat_index(shifted: Dict[int, str]) -> str:
    """``flat(...)`` with the named per-axis substitutions applied."""
    coords = [shifted.get(axis, _HOME_COORDS[axis]) for axis in range(3)]
    return f"flat({coords[0]}, {coords[1]}, {coords[2]}, nyz, nz)"


def _term_lines(component: int, offset: int) -> List[str]:
    """One partner's term for one component, as source lines.

    ``offset`` is 0 for MEEP's ``cycle_direction(dim, d_ec, 1)`` and 1 for
    ``(..., 2)`` (stepping.py:1235-1237). The partner axis, the coefficient slot,
    the partner volume, all three shifted indices, the ghost lane, the boundary
    code, the parity weight and both phase pairs come from that one number, so a
    mispairing would have to be introduced deliberately rather than by a typo.
    """
    own_axis = E_TERMS[component][2]
    partner_axis = _coverage.OFFDIAG_TRANSVERSE_PARTNERS[component][offset]
    coefficient = ROW_PARAMETERS[2 * component + offset]
    volume = PARTNER_VOLUMES[partner_axis]
    tag = f"{E_TERMS[component][0]}_{offset}"
    down = _flat_index({partner_axis: _DOWN_COORDS[partner_axis]})
    up = _flat_index({own_axis: _UP_COORDS[own_axis]})
    corner = _flat_index({own_axis: _UP_COORDS[own_axis],
                          partner_axis: _DOWN_COORDS[partner_axis]})
    return [
        f"    // partner {volume}: the pair half a cell DOWN axis {partner_axis},",
        f"    // then the product half a cell UP axis {own_axis} (its own). The",
        f"    // near-face rule (mirror parity or the conjugate Bloch factor) is on",
        f"    // axis {partner_axis}; the forward Bloch factor is on axis {own_axis}.",
        f"    cf term_{tag} = offdiag_term(",
        f"        {volume}, {coefficient}, idx,",
        f"        {down},",
        f"        {up},",
        f"        {corner},",
        f"        {_DOWN_LANES[partner_axis]}, {_BC_NAMES[partner_axis]},"
        f" {_PHASE_FLAGS[partner_axis]}, {_DOWN_PHASES[partner_axis]},"
        f" {_GHOST_WEIGHTS[partner_axis]},",
        f"        {_UP_LANES[own_axis]}, {_PHASE_FLAGS[own_axis]},"
        f" {_UP_PHASES[own_axis]});",
    ]


def _wall_mask_lines(component: int) -> List[str]:
    """``stepping._mask_metallic_wall_coupling`` for one component's total.

    Face 0 of every axis whose Yee shift is 0
    (:data:`coverage.OFFDIAG_WALL_MASK_AXES`), ascending -- the mask's own loop
    order (stepping.py:1279-1283). WHICH axes can be masked is fixed by the
    component and compiled in; WHETHER each one is masked is the runtime ``wm_*``
    flag, because that is a property of the grid rather than of the arithmetic.

    ``cf_zero()`` IS BOTH WORDS. The array path assigns the INTEGER 0 to a
    complex64 array (stepping.py:1283), which is the pair (+0.0f, +0.0f); a clear
    that touched only the real plane would leave the imaginary coupling alive on a
    metallic wall -- a defect class this tree has paid for, and one the gate arms.
    """
    total = f"total_{E_TERMS[component][0]}"
    return [f"    {total} = ({_WALL_FLAGS[axis]} && {_WALL_PREDICATES[axis]})"
            f" ? cf_zero() : {total};"
            for axis in _coverage.OFFDIAG_WALL_MASK_AXES[component]]


def _component_source(component: int, row_mask: Sequence[int], arm: str) -> str:
    """One component's whole block, on the arm its two row slots select.

    Four row arms: none, offset-1 only, offset-2 only, both. THE NONE ARM IS THE
    ELEMENT-WISE COMPLEX CONSTITUTIVE BODY -- ``src = D * inv_eps`` and the same
    tail -- because ``_offdiagonal_terms`` returns None for a component with no
    surviving row and ``update_E`` then never forms a ``+ 0`` copy
    (stepping.py:1222-1227, :1007).

    ``arm`` selects the TAIL and nothing else. Every line above the tail is shared
    text, which is what makes "the row product is the same body in both kernels" a
    property of this function rather than a claim in a docstring.
    """
    name, source, own_axis = E_TERMS[component]
    axis_letter = "xyz"[own_axis]
    index_letter = _OWN_AXIS_INDEX[own_axis]
    live = [offset for offset in (0, 1) if row_mask[2 * component + offset]]

    lines = [f"    // --- {name}: own axis {axis_letter}; source {source}; "
             f"dsigw = {axis_letter}.",
             f"    cf gs_{name} = cf_load({source}, idx);",
             f"    cf diag_{name} = mul_field_left(gs_{name}, "
             f"inv_eps_{name}[idx]);"]
    if not live:
        lines.append(f"    cf src_{name} = diag_{name};")
    else:
        for offset in live:
            lines.extend(_term_lines(component, offset))
        # ``total`` accumulates offset 1 then offset 2 (stepping.py:1250). Complex
        # addition is plane-wise and float32 addition is bitwise commutative, so
        # this order is transcription fidelity rather than a pinned grouping.
        lines.append(f"    cf total_{name} = term_{name}_{live[0]};")
        for offset in live[1:]:
            lines.append(f"    total_{name} = cf_add(total_{name}, "
                         f"term_{name}_{offset});")
        # The mask runs BEFORE the row sum: stepping.py:1252 precedes :1007-1008.
        lines.extend(_wall_mask_lines(component))
        lines.append(f"    cf src_{name} = cf_add(diag_{name}, total_{name});")
    tail = _TAIL_PML if arm == "pml" else _TAIL_STORE
    lines.append(tail.format(name=name, axis=axis_letter, index=index_letter))
    return "\n".join(lines)


def _row_parameter_lines(row_mask: Sequence[int]) -> str:
    """The live coefficient parameters, in :data:`coverage.OFFDIAG_ROW_SLOTS` order.

    THE SIGNATURE CARRIES ONLY THE LIVE ONES, ``offdiag_emitter``'s choice and for
    its reason: binding a dead slot to some other volume leaves a pointer aimed at
    an array the kernel must never touch, which a later edit can read by accident.

    NOT ``__restrict__``. A row coefficient may legally alias another row's, an
    inverse-epsilon volume or a D volume (an isotropic install hands the same
    inverse permittivity three times, fields.py:1321-1326), and restrict on
    mutually aliasing arguments is a promise the caller cannot keep. They are
    read-only, so nothing is lost but a load hint. They are float32 VOLUMES indexed
    by the COMPLEX CELL index and NEVER word-doubled: chi1inv stays float32 under
    complex storage exactly as inv_eps does (stepping.py:41-50 against
    fields.py:1203-1204).
    """
    return "".join(f"    const float* {ROW_PARAMETERS[slot]},\n"
                   for slot, flag in enumerate(row_mask) if flag)


def normalized_row_mask(row_mask: Sequence[int]) -> Tuple[int, ...]:
    """Validate and canonicalize a row mask -- ``offdiag_emitter``'s, imported.

    Its refusal message names the plain constitutive kernel, which is the family a
    zero-slot run belongs to on either storage width, so it is the right message
    here too.
    """
    return _flat.normalized_row_mask(row_mask)


def normalized_arm(arm: str) -> str:
    """The tail key, refused by name rather than defaulted.

    A WRONG TAIL IS A WRONG ANSWER, not a crash: both compile, both run, and one
    accumulates onto E where the other overwrites it.
    """
    if arm not in KERNEL_NAMES:
        raise ValueError(f"arm must be one of {sorted(KERNEL_NAMES)}, got {arm!r}")
    return arm


def kernel_name(arm: str) -> str:
    """The ``extern "C"`` symbol NVRTC is asked for, by tail."""
    return KERNEL_NAMES[normalized_arm(arm)]


#: Patterns that must NOT survive into an emitted body. ``complex_emitter``'s
#: prelude is carried WHOLE -- including ``pml_apply``, ``cshift_up`` and
#: ``cshift_dn``, which nothing here calls -- for ``no_pml_curl._sibling_prelude``'s
#: reason: forking it to drop the dead helpers is the silent divergence that
#: indirection exists to prevent. Carrying them turns the liability into evidence,
#: because the gate arms a mutation of each as a MUST-BE-UNCAUGHT null and their
#: silence is the measurement that this family really does not route through them.
#: These regexes are what makes "nothing calls them" a property of the build:
#: spelled with an explicit left boundary because ``no_pml_apply(`` contains
#: ``pml_apply(`` as a substring, a plain ``in`` test that has already read as a
#: failure once on this track.
_FORBIDDEN_IN_BODY: Tuple[str, ...] = (
    r"(?<![A-Za-z0-9_])cshift_up\(",
    r"(?<![A-Za-z0-9_])cshift_dn\(",
    r"(?<![A-Za-z0-9_])pml_apply\(",
    r"(?<![A-Za-z0-9_])sinv_",
)

#: Tokens the STORE arm must not carry: the auxiliary, the coefficient stems and
#: the certified tail call ARE the split-field recurrence.
_FORBIDDEN_IN_STORE_BODY: Tuple[str, ...] = (
    r"(?<![A-Za-z0-9_])f_w_E[xyz](?![A-Za-z0-9_])",
    r"(?<![A-Za-z0-9_])kps_",
    r"(?<![A-Za-z0-9_])kms_",
    r"(?<![A-Za-z0-9_])constitutive_apply\(",
)


def _prelude(expansion) -> str:
    """The sibling's whole prelude plus this family's own helpers.

    ``complex_emitter._HEAD + _ARM_SOURCE[arm] + _TAIL`` is taken WHOLE, so
    ``cf_load``/``cf_store``/``cf_add``/``cf_sub``/``cf_zero``, the three multiply
    orientations and the certified ``constitutive_apply`` are that module's bytes
    rather than a second copy of them.
    """
    code = complex_emitter.normalized_expansion(expansion)
    return (complex_emitter._HEAD + complex_emitter._ARM_SOURCE[code]
            + complex_emitter._TAIL + _OWN_PRELUDE)


def complex_offdiag_source(arm: str, row_mask: Sequence[int], expansion) -> str:
    """The device source for one tail, one row mask and one expansion arm.

    ``expansion`` is REQUIRED and never defaulted: a wrong arm is a wrong answer
    rather than a crash -- both arms compile, both run, and they differ in the last
    bits of about a quarter of the words. ``complex_emitter.normalized_expansion``
    is the one refusal.

    The BOUNDARY, WALL, PHASE and PARITY axes are NOT baked in; they are runtime
    arguments, for the reason the module docstring gives. So the row mask and the
    expansion arm are this family's only source axes per tail, and the 186-row
    corpus drives two row masks and one arm.
    """
    tail = normalized_arm(arm)
    mask = normalized_row_mask(row_mask)
    body = _CORE.replace("__NAME__", KERNEL_NAMES[tail])
    body = body.replace("__AUX_PARAMS__", _AUX_PARAMS if tail == "pml" else "")
    body = body.replace("__COEF_PARAMS__", _COEF_PARAMS if tail == "pml" else "")
    body = body.replace("__ROW_PARAMETERS__", _row_parameter_lines(mask))
    for component, term in enumerate(E_TERMS):
        body = body.replace(f"__SRC_{term[0]}__",
                            _component_source(component, mask, tail))
    _assert_body_is_clean(body, tail)
    return _prelude(expansion) + body


def _assert_body_is_clean(body: str, arm: str) -> None:
    """Raise if a placeholder survived or a forbidden token reached the body.

    THE ASSERTIONS ARE THE POINT, and they are ``complex_no_pml_kernels``'
    ``_curl_source``'s: a template rename upstream, a fourth component block or a
    tail this substitution missed all raise HERE, at emit, with the site named --
    rather than compiling into a kernel that silently steps a recurrence this
    family does not have.
    """
    for pattern in _FORBIDDEN_IN_BODY:
        found = re.search(pattern, body)
        if found is not None:
            raise RuntimeError(
                f"{found.group(0)!r} reached an emitted body; this family calls "
                f"none of the sibling prelude's curl or split-field helpers, and "
                f"carrying them as dead code is only safe while nothing calls them")
    if arm == "no_pml":
        for pattern in _FORBIDDEN_IN_STORE_BODY:
            found = re.search(pattern, body)
            if found is not None:
                raise RuntimeError(
                    f"{found.group(0)!r} survived into the direct-store body; the "
                    f"split-field recurrence has no place there")
    stripped = body
    for token in ("__restrict__", "__global__", "__device__", "__forceinline__",
                  "__fmaf_rn"):
        stripped = stripped.replace(token, "")
    if "__" in stripped:
        raise RuntimeError("an unsubstituted placeholder survived the build")


def corpus_digest() -> str:
    """One sha256 over every source this family can emit, canonically ordered.

    Two tails x 63 row masks x 2 expansion arms is 252 sources -- too many to pin
    one digest each without burying the record and not too many to hash, so the
    whole enumeration hashes to one value: a single changed character anywhere in
    this module, or in ``complex_emitter``'s shared prelude, moves it. The
    per-source digests of the combinations a gate actually launches belong in that
    gate's artifact, which is where a certification record reads them from.
    """
    import hashlib  # noqa: PLC0415 - stdlib, imported at the one call site

    digest = hashlib.sha256()
    for arm in ARMS:
        for mask in LIVE_ROW_MASKS:
            for name in sorted(complex_emitter.EXPANSIONS):
                digest.update(f"{arm}|{mask!r}|{name}".encode("ascii"))
                digest.update(complex_offdiag_source(arm, mask, name).encode("utf-8"))
    return digest.hexdigest()


def shipped_kernel_names(source: Optional[str] = None):
    """Every kernel NVRTC could be asked to compile, read from the emitted text."""
    if source is not None:
        return set(re.findall(r'extern "C" __global__ void (\w+)\(', source))
    found = set()
    for arm in ARMS:
        for name in sorted(complex_emitter.EXPANSIONS):
            found |= set(re.findall(
                r'extern "C" __global__ void (\w+)\(',
                complex_offdiag_source(arm, (1, 1, 1, 1, 1, 1), name)))
    return found


# ---------------------------------------------------------------------------
# The host-side facts a launch needs
# ---------------------------------------------------------------------------

def complex_offdiag_boundary_codes(grid: Any) -> Tuple[int, ...]:
    """The three ``bc_*`` arguments: the resolved ghost rule per axis, as codes.

    ``coverage.real_pml_boundary_kinds`` reproduces ``stepping._boundary_kinds``,
    and this family's table is :data:`coverage.BC_CODES` plus ``mirror`` -- the ONE
    difference from ``complex_pml_kernels.complex_boundary_codes``, whose refusal
    on a folded axis is the reason the certified complex predicate could not be
    widened onto this shape.

    Every kind the resolution can return that is NOT in :data:`FOLDED_BC_CODES`
    (today: the cylindrical ``axis``) is refused by both predicates, so a KeyError
    here would mean a launcher ran on a configuration the predicate declined. It is
    left to raise rather than defaulted, because a default would turn that into a
    silent wrong ghost rule.
    """
    return tuple(FOLDED_BC_CODES[kind]
                 for kind in _coverage.real_pml_boundary_kinds(grid))


def mirror_ghost_weights(grid: Any) -> Tuple[float, float, float]:
    """``gw_x``/``gw_y``/``gw_z`` -- the parity the mirror ghost carries per axis.

    ``folded_offdiag_kernels.mirror_ghost_weights``, delegated rather than
    respelled: that function derives the value through ``fields.mirror_parity``
    ITSELF (and its own test pins the collapse to ``-phase`` over 3 axes x 2
    phases), and the value is the same fact on either storage width -- the parity
    is a property of the COMPONENT and the PLANE, not of the field's dtype.

    ``1.0`` on an unfolded axis, where the ghost lane is never a mirror and the
    value is never read. ``nan`` where the plane cannot be read: the weight is a
    RUNTIME argument, so an unreadable plane would otherwise LAUNCH rather than
    raise, and both predicates name the NaN by clause.
    """
    from .folded_offdiag_kernels import (  # noqa: PLC0415 - stdlib-safe sibling
        mirror_ghost_weights as _weights)

    return _weights(grid)


def bloch_phase_table(grid: Any) -> Tuple[Tuple[int, ...],
                                          Tuple[float, ...],
                                          Tuple[float, ...]]:
    """``(flags, down_values, up_values)`` -- the per-axis Bloch table, both ways.

    Transcribed from ``stepping._bloch_phases`` (:2313-2367) and
    ``stepping._apply_bloch_phase`` (:1846-1862), and the same table
    ``complex_pml_kernels.bloch_phase_arguments`` builds -- one call per direction
    there, both here, because this sub-step needs BOTH: the partner-axis DOWN shift
    takes the conjugate and the own-axis UP shift the forward factor, and a given
    axis is a partner for one component and an own axis for another.

    The two spellings are NOT assumed equal: the gate asserts this function's
    output equal to ``bloch_phase_arguments(grid, backward=True)`` and
    ``(..., backward=False)`` on every swept grid. It is not asserted at the merge
    bar because that module imports ``cupy`` at scope and the merge bar has none.

    * the value is ``grid.bloch_phase(axis)`` = exp(2*pi*i*k*L), with the Brillouin
      edge EXACTLY -1+0j (grid.py:1011-1017);
    * ``None`` means the multiply is SKIPPED, never done against 1+0j -- that skip
      IS the bit-identity of k = 0, so an unphased axis passes flag 0 and the
      kernel emits no multiply for it;
    * the phase is rounded to complex64 BEFORE splitting, because the array path
      multiplies by ``shifted.dtype.type(phase)`` (:1862);
    * the DOWN direction negates the imaginary part -- the conjugate
      ``_shift_down`` takes (:1818-1822). Negation is exact, so the order of
      conjugation and rounding is immaterial.

    A PHASED AXIS MUST RESOLVE PERIODIC. ``stepping._bloch_phases`` raises on the
    pairing and both predicates refuse it; this raises too, so a caller who skipped
    the predicate is refused loudly rather than handed a mis-bound flag.
    """
    import numpy as np  # noqa: PLC0415 - only the float32 rounding is needed

    kinds = _coverage.real_pml_boundary_kinds(grid)
    flags: List[int] = []
    down: List[float] = []
    up: List[float] = []
    for axis in range(3):
        phase = (grid.bloch_phase(axis)
                 if getattr(grid, "has_bloch", False) else None)
        if phase is None:
            flags.append(0)
            down.extend((1.0, 0.0))
            up.extend((1.0, 0.0))
            continue
        if kinds[axis] != _coverage.PERIODIC:
            raise ValueError(
                f"axis {axis} carries Bloch phase {phase!r} but resolved to "
                f"{kinds[axis]!r}; only a periodic wrap can carry a phase "
                f"(stepping._bloch_phases raises on the same configuration)")
        rounded = np.complex64(phase)
        real = float(np.float32(rounded.real))
        imag = float(np.float32(rounded.imag))
        flags.append(1)
        down.extend((real, float(np.float32(-imag))))
        up.extend((real, imag))
    return tuple(flags), tuple(down), tuple(up)


def complex_offdiag_tables(pml: Any) -> Dict[str, Any]:
    """The six HALF-INTEGER kps/kms views the split-field tail reads, flattened.

    The sub-lattice is asked of :func:`coverage.constitutive_sub_lattice`, the same
    function both predicates ask and every certified launcher on this track asks,
    so no call site can pair the E side with the INTEGER tables -- a half-cell
    error in the absorber profile that is converged, smooth and wrong.
    """
    from .constitutive_kernels import real_constitutive_tables  # noqa: PLC0415

    return real_constitutive_tables(pml, _coverage.constitutive_sub_lattice("E"))


# ---------------------------------------------------------------------------
# THE COVERAGE PREDICATES
# ---------------------------------------------------------------------------

#: WHAT HAS BEEN MEASURED ON A DEVICE FOR THIS FAMILY, and nothing more.
#:
#: ``host`` is the only thing that makes any number here a device verdict; a
#: ``host: None`` record means a predicate returning True licenses a MEASUREMENT
#: and nothing else. ``test_complex_offdiag_update_e.py`` is what keeps the two
#: states from blurring, and it is the same all-or-nothing rule
#: ``folded_offdiag_kernels.FOLDED_OFFDIAG_ADMISSION`` and
#: ``complex_no_pml_kernels.COMPLEX_NO_PML_ADMISSION`` carry.
COMPLEX_OFFDIAG_UPDATE_E_ADMISSION: dict = {
    "kernels": tuple(KERNEL_NAMES[arm] for arm in ARMS),
    "gate": "parity/meep_gpu/gate_cuda_complex_offdiag_update_e.py",
    "oracle": ("stepping.update_E on real Grid/Fields/PML objects with complex64 "
               "storage and rows installed through Fields.set_epsilon_volumes, "
               "compared as uint32 words"),
    "specified_by": "complex_no_pml_kernels.WHAT_IS_NOT_BUILT",
    "recorded_utc": "2026-08-21T23:21:00Z",
    "host": "the GPU host, NVIDIA RTX A6000, CuPy 13.5.1. THE PAYLOAD CARRIES NO "
            "MACHINE STAMP -- this gate writes no hostname and no device index, so "
            "unlike the 2026-08-20 run (GPU index 3, UUID GPU-00000000-0000-0000-0000-000000000001, recorded by "
            "the session that launched it) the index for this re-cut cannot be read "
            "back out of the artifact and is not claimed here. Closing it means the "
            "gate stamping its own hostname and device, and a re-cut.",
    "artifacts": ("parity/meep_gpu/results/"
                  "cuda_complex_offdiag_update_e_2026-08-21_rename/keep/gate.json",
                  "parity/meep_gpu/results/"
                  "cuda_complex_offdiag_update_e_2026-08-21_rename/flush/gate.json"),
    "artifact_sha256": {"keep": "574f571c6c6a7836c59c87ac11908c4638182328484e634cccde123a3211574c",
                        "flush": "55c90bcb4adbb7fe688b50561fde4dbff0b6f1b028175eea34841bb9a9ddb890"},
    "gate_sha256_at_run": "a0bd21372c83763269ada0420eed243984b6e7a595724bcb77166faffcec6879",
    #: HOW THIS RECORD IS BOUND TO THOSE RUNS, and why it is NOT an artifact hash
    #: alone. Both artifacts carry ``imported_source_sha256``, which records THIS
    #: FILE's bytes as of the run -- so writing an artifact digest here and nothing
    #: else would be circular the moment this block was added. The non-circular
    #: binding is the EMITTER CORPUS DIGEST below: one sha256 over all 252 sources
    #: this family can emit, carried in both artifacts and asserted equal to the
    #: shipped emitter by ``test_complex_offdiag_update_e.py``. That ties the
    #: verdict to the exact DEVICE CODE, which is what the verdict is about; a
    #: docstring edit moves the file hash and not the arithmetic, and this record
    #: IS a docstring edit.
    "bound_by": "emitter_corpus_digest, carried in both artifacts",
    "post_gate_record_edits": (
        "this COMPLEX_OFFDIAG_UPDATE_E_ADMISSION block was filled in after the "
        "runs it describes; it touches no device string and no predicate clause, "
        "and the emitter corpus digest is unchanged across the edit",
        "2026-08-22: RE-CUT onto the post-rename run. The CUDA kernel rename "
        "(fused_step_* -> step_*) moved every device string this family emits, so "
        "the emitter corpus digest the 2026-08-20 run was cut at no longer "
        "describes the shipped emitter. Nothing else in this record moved: the "
        "2026-08-21 re-run scored the SAME numbers on both policy legs -- 1104 "
        "cases, 552/552 guarded, 336/336 folded, 240/240 phased, 228 live-partner "
        "and 228 live-own fold axes, 84 wall-mask, 32/32 multi-step, 40/40 "
        "mutation legs with 31 caught, 9 null-confirmed, 0 escaped and 0 unarmed. "
        "The gate script itself is byte-identical across the two runs "
        "(gate_sha256_at_run is unchanged), so the re-cut moved digests and "
        "timestamps and no measurement",),
    "policies": ("ieee_keep_ftz_stripped", "meep_x86_flush"),
    #: PER POLICY. The two produced IDENTICAL case counts and identical verdicts,
    #: which is why one copy carries both; each ran in its own process with its own
    #: policy-token CuPy cache directory, because CuPy's disk-cache key is computed
    #: above the strip seam.
    "cases_scored": 1104,
    "cases_refused_as_vacuous": 0,
    "single_launch_identical": "552/552 under --fmad=false",
    "multi_step_identical": "32/32",
    "multi_step_budget": 60,
    "per_arm_identical": {"pml": "408/408", "no_pml": "144/144"},
    "folded_identical": "336/336",
    "phased_identical": "240/240",
    #: The two arms a fold can reach, counted separately, because a total alone
    #: cannot show that the arm this family exists for was exercised.
    "cases_where_a_fold_is_a_live_partner_axis": 228,
    "cases_where_a_fold_is_a_live_own_axis": 228,
    "cases_where_the_wall_mask_bites": 84,
    "subnormal_band_scored": 184,
    "signed_zero_scored": 184,
    #: THE PREDICTION THE CURL PAIR'S SPLIT DOES NOT TRANSFER, measured: update_E
    #: has no ownership mask and no reflect row, so ONE BC_MIRROR code serves both
    #: declared terminations of a fold.
    "folded_periodic_identical": "240/240",
    "folded_metallic_identical": "120/120",
    "simultaneous_planes_identical": {"1": "264/264", "2": "48/48", "3": "24/24"},
    "mutation_legs_as_required": "40/40",
    "mutations_caught": 31,
    "mutations_null_confirmed": 9,
    "mutations_escaped": 0,
    "mutations_unarmed": 0,
    #: The nine that make the battery two-sided rather than a list of catches. The
    #: last three are MEASURED FINDINGS this run established rather than legs that
    #: were expected to be silent -- see the gate's ``NULL_MUTATIONS`` note.
    "null_controls_confirmed": (
        "commute_the_row_sum", "commute_the_pair_add", "dead_cshift_up",
        "dead_cshift_dn", "dead_pml_apply", "rebind_identical_tables",
        "parity_plane_wise", "up_ghost_real_plane_only",
        "down_ghost_real_plane_only"),
    "release_verdict_shown_to_flip_against_a_planted_defect": True,
    #: THE GUARD FINDING, and it does NOT match the certified siblings'. The real
    #: off-diagonal family measured its unguarded control 0/384 identical, which is
    #: what makes ``--fmad=false`` CORRECTNESS there. Here the unguarded control
    #: agreed on 276/276 cases at the INEXACT courant, and the compiler says why:
    #: NVRTC emitted IDENTICAL PTX under both option tuples for both tails, because
    #: every fusion on the licensed FMA_V1 arm is spelled ``__fmaf_rn`` (a single
    #: ``fma.rn.f32`` whatever the flag says) and no bare multiply feeds a bare add.
    #: The flag is KEPT for comparability with the siblings and because the NAIVE
    #: arm this licence does not bind DOES carry a contractable pair -- measured,
    #: its PTX differs between the two option tuples.
    "guard_is_load_bearing": False,
    "guard_control_identical_at_inexact_courant": "276/276",
    "guard_ptx_distinct_on_the_licensed_arm": False,
    "guard_ptx_distinct_on_the_naive_arm": True,
    #: THE LICENCE, and why it is the FIVE-pattern one. The mirror ghost is
    #: ``parity * plane`` with a python int on the left, which is
    #: ``folded_complex.PARITY_PROBE_PATTERN``; the base four cannot bind it. Both
    #: policies of ``expansion_probe_2026-08-17`` classify FMA_V1 on a measured
    #: basis over that set with no refusals, which is why this family spans both
    #: policies where ``complex_no_pml_kernels`` spans one.
    "expansion_licence": ("triton_kernels.folded_complex."
                          "parity_expansion_license over "
                          "results/expansion_probe_2026-08-17/, arm FMA_V1, "
                          "basis measured, both policies"),
    #: WHAT THE RECORD DOES NOT SAY. It is a verdict about the EMITTER at the
    #: corpus digest below, exercised on FOUR of its 63 row masks per tail and on
    #: ONE of its two expansion arms (the one the licence binds). A mask or an arm
    #: outside those is covered by the emitter's own tests and by that digest, not
    #: by a byte gate. No throughput claim is made or possible: the box was shared.
    "row_masks_exercised": ((1, 1, 1, 1, 1, 1), (1, 0, 0, 1, 0, 0),
                            (1, 1, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0)),
    "expansion_arms_exercised": ("FMA_V1",),
    "emitter_corpus_digest":
        "1d887e4bdc4f118674b63e26f7d64f9c1e4507c68ea413734054a66622107e91",
    #: STILL NOT CERTIFIED, and that is a deliberate state. ``certification.json``
    #: is the thing a reader is entitled to check a claim against, and "certified"
    #: must be a verdict someone can read rather than a word someone typed. Both
    #: gates PASSED under both policies; the record entry is what is owed.
    "certified": False,
    #: WHAT IT IS WORTH, recomputed on ONE census tree with the union analyzer --
    #: the family present and absent on the SAME 186 rows, so the delta is
    #: attributable to this leg rather than to a tree that moved between rounds.
    "corpus": {
        "census": ("parity/meep_gpu/results/"
                   "cuda_predicate_coverage_2026-08-20_complex_offdiag"),
        "corpus_rows": 186, "slots": 759,
        "union_without_this_family": 753,
        "union_with_this_family": 758,
        "slots_gained": 5,
        "rows_covered_at_every_sub_step_gained": 5,
        "overlaps_introduced": 0,
        "rows_taken": ("examples/solve-cw.py (pml)",
                       "tests TestArrayMetadata.test_array_metadata (pml)",
                       "tests TestHoleyWvgBands.test_fields_at_kx (pml)",
                       "tests TestMaterialGrid.test_matgrid_3d (no_pml)",
                       "tests TestMaterialGrid.test_subpixel_smoothing (no_pml)"),
        #: The ONE slot of 759 still unserved after this family:
        #: ``examples/absorbed_power_density.py`` at update_E, a FOLDED
        #: off-diagonal run that ALSO carries a susceptibility. It needs a
        #: dispersive x off-diagonal x fold arm and belongs to neither this family
        #: nor the real folded one.
        "slots_still_unserved": 1,
    },
    "what_it_does_not_license": (
        "a production step. No module in meep_gpu/ imports cuda_kernels at all, "
        "so a predicate returning True licenses a MEASUREMENT and a coverage "
        "claim, not a run.",
        "the NAIVE expansion arm, which the emitter can still emit and no device "
        "leg has scored.",
        "any throughput claim: the box was shared and no timing was taken.",
    ),
}


def _fold_reasons(facts: dict, grid: Any) -> List[str]:
    """Everything this family needs of a fold, as accumulated reasons.

    ``folded_offdiag_kernels._fold_reasons``' three checks, restated for complex
    storage because the two chains differ elsewhere and delegating a
    short-circuiting predicate cannot answer "refused for the fold and nothing
    else". ``test_complex_offdiag_update_e.py`` pins this function's verdict equal
    to the real family's on every grid it can build, so the duplication cannot
    drift silently.

    a. MORE THAN ``MIRROR_ROW`` STORED CELLS. ``coord_dn``'s mirror arm returns
       stored row 2 unconditionally at face 0, and ``stepping._mirror_source``
       (:1536-1541) RAISES below that many cells. A thinner axis would have the
       kernel read a neighbouring plane where the array path refuses to run.
    b. A READABLE PLANE PHASE, exactly +1 or -1. The parity is a runtime float
       argument, so an unreadable plane launches a NaN rather than raising.
    c. THE TWO ROUTES TO THE TERMINATION AGREEING
       (``coverage._fold_termination_problem``). One code serves both terminations
       here, so the cross-check is not load-bearing for the arithmetic; it is
       load-bearing for TRUST -- a grid whose ``stored_cells > owned_cells`` and
       whose ``is_metallic`` disagree has one of the two drifted, and neither can
       then be believed about the boundary kinds this kernel indexes on.
    """
    reasons: List[str] = []
    mirrored = facts["mirrored"]
    weights = mirror_ghost_weights(grid)
    for axis in range(3):
        if not mirrored[axis]:
            continue
        if int(facts["shape"][axis]) <= MIRROR_SOURCE_INDEX:
            reasons.append(
                f"axis {axis} is folded with {facts['shape'][axis]} stored cells; "
                f"the mirror ghost images stored row {MIRROR_SOURCE_INDEX} and "
                f"stepping._mirror_source raises below that")
        problem = _coverage._fold_termination_problem(facts, axis)
        if problem is not None:
            reasons.append(problem)
        weight = weights[axis]
        if weight != weight or weight not in (1.0, -1.0):
            reasons.append(
                f"axis {axis} mirror ghost weight is {weight!r}, not +1.0 or "
                f"-1.0; the weight is a RUNTIME argument and an unreadable plane "
                f"would launch rather than raise")
        if facts["metallic"][axis] and _coverage._offdiag_wall_flags_from(
                facts)[axis]:  # pragma: no cover - unreachable by construction
            reasons.append(
                f"axis {axis} is folded AND wall-masked; "
                f"_mask_metallic_wall_coupling abstains on a mirrored axis "
                f"(stepping.py:1253)")  # stepping.py live lines for the frozen device-text citation(s) in this string: 1253->1282
    return reasons


def _row_and_volume_refusal(fields: Any, grid: Any, facts: dict, arm: str) -> Any:
    """The clauses about this sub-step's OWN arrays, or None.

    Shared by the two predicates because the row product, the volumes it reads and
    the aliasing rule are identical on both tails; the OUTPUT set is not, and that
    is the ``arm`` argument's whole job.
    """
    xp, _ = _coverage._backend(grid)
    shape = facts["shape"]
    # (a) INVERTED. Counted over the six slots, deliberately not the bare
    #     ``has_offdiagonal_epsilon`` flag: a row planted past the installer under
    #     a diagonal key sets the flag with every slot dead, and the emitter
    #     refuses an all-dead mask by raising.
    rows = _coverage.offdiag_row_volumes(fields)
    if not any(volume is not None for volume in rows):
        return ("no off-diagonal chi1inv row survived installation: that "
                "configuration is the element-wise complex constitutive family's "
                "and this predicate must not overlap it")
    outputs = ("Ex", "Ey", "Ez") + (("f_w_Ex", "f_w_Ey", "f_w_Ez")
                                    if arm == "pml" else ())
    for name in outputs + ("Dx", "Dy", "Dz"):
        problem = _coverage._complex_volume_problem(
            name, getattr(fields, name, None), xp, shape)
        if problem is not None:
            return problem
    if arm == "no_pml":
        # THE AUXILIARY MUST BE ABSENT, not merely unused. The direct-store arm
        # writes no ``f_w``; an ALLOCATED one means something else built it and
        # this step would leave it stale -- a wrong answer on a LATER sub-step
        # rather than on this one. ``complex_no_pml_kernels`` refuses the same
        # shape for the same reason.
        for name in ("Ex", "Ey", "Ez"):
            if getattr(fields, "f_w_" + name, None) is not None:
                return (f"f_w_{name} is allocated while the layer is inert: this "
                        f"arm writes no auxiliary and the stored E IS the "
                        f"constitutive product without a layer (fields.py:1150), "
                        f"so an allocated one means something else built it and "
                        f"this step would leave it stale")
    reader = getattr(fields, "inverse_epsilon_for", None)
    if not callable(reader):
        return "fields does not expose inverse_epsilon_for"
    for component in ("Ex", "Ey", "Ez"):
        try:
            volume = reader(component)
        except Exception as exc:  # noqa: BLE001 - a raise is not a refusal
            return f"inverse_epsilon_for({component!r}) raised {exc!r}"
        if volume is None:
            return f"inverse_epsilon_for({component!r}) is None"
        if not getattr(volume, "shape", ()):
            return (f"inverse_epsilon_for({component!r}) is a scalar, not a "
                    f"volume")
        # FLOAT32 UNDER COMPLEX STORAGE, indexed by the COMPLEX CELL index and
        # never word-doubled (stepping.py:41-50 against fields.py:1203-1204).
        problem = _coverage._array_problem(
            f"inverse_epsilon_for({component!r})", volume, xp, shape)
        if problem is not None:
            return problem
    # (b) and (c): the surviving rows, readable, well-formed, and not an output.
    #     The alias clause is stronger here than for an element-wise sub-step: the
    #     coupling re-reads the partner volumes at NEIGHBOUR offsets -- on a folded
    #     axis including INTERIOR stored row MIRROR_ROW rather than a face -- while
    #     the outputs are being written, so an alias makes the answer depend on
    #     block schedule: wrong differently on each run, which no single comparison
    #     catches reliably.
    output_addresses: Dict[int, str] = {}
    for name in outputs:
        address = _coverage._base_address(getattr(fields, name, None))
        if address is not None:
            output_addresses.setdefault(address, name)
    for index, volume in enumerate(rows):
        if volume is None:
            continue
        row, partner = _coverage.OFFDIAG_ROW_SLOTS[index]
        label = f"chi1inv_offdiagonal[{row!r}][{partner!r}]"
        if not getattr(volume, "shape", ()):
            return f"{label} is a scalar, not a volume"
        problem = _coverage._array_problem(label, volume, xp, shape)
        if problem is not None:
            return problem
        address = _coverage._base_address(volume)
        if address is not None and address in output_addresses:
            return (f"{label} aliases output {output_addresses[address]}: the "
                    f"coupling re-reads the partner volumes at neighbour offsets "
                    f"-- on a folded axis including interior stored row "
                    f"{MIRROR_SOURCE_INDEX} -- while the outputs are written, so "
                    f"the answer would depend on block schedule")
    return None


def _parity_licence_refusal(license: Any) -> Any:
    """Why this licence cannot bind the PARITY multiply, or None.

    THE FIFTH ORIENTATION IS THIS FAMILY'S OWN CLAUSE. ``_symmetry_phase`` returns
    a python int, so ``parity * complex64_array`` is
    ``c8_mul_c8_parity_coefficient_left`` -- neither of the base four zero-imaginary
    orientations. A record that does not classify it cannot license the mirror arm,
    and this refuses by name rather than falling back, exactly as
    ``complex_no_pml_kernels`` refuses ``update_P`` without
    ``c8_mul_python_float_field_left``.

    THE ARBITER IS ``triton_kernels.folded_complex.parity_expansion_license`` and is
    not re-implemented here; this function only checks WHICH PATTERN SET the licence
    in hand was computed over. Two spellings of an arbiter is one too many.

    ``probe_patterns`` IS THE LICENCE'S OWN RECORD OF ITS EVIDENCE -- the tuple
    ``expansion_license`` was called with -- so asking it is asking the arbiter what
    it looked at, not guessing from an attached artifact. ``patterns`` (the record's
    raw pattern table, which the census legs attach with ``setdefault``) is accepted
    as a fallback so a caller that carries the record rather than the verdict is not
    refused for a shape difference.

    THE PARITY PATTERN IS MEASURABLY NON-DISCRIMINATING on the shipped probe, and
    that is not a reason to drop the clause. With ``c_re = +/-1`` exact and
    ``c_im = +0.0`` the fused arm's extra product is exact, so both arms produce
    identical bytes there and ``parity_expansion_license`` excludes the pattern from
    its agreement test rather than letting it veto. What the clause buys is the
    OTHER outcome: a record on which the plane-wise spelling ALSO reproduces the
    bytes has no power to license the mirror arm at all, and only a record that
    carried the pattern can say which of the two it was.
    """
    if license is None:
        return None  # the shape clause is complex_expansion_refusal's job
    if not isinstance(license, dict):
        return None  # likewise: complex_expansion_refusal names the shape
    patterns = license.get("probe_patterns")
    if patterns is None:
        patterns = license.get("patterns")
    if patterns is None:
        return (f"the expansion licence names no probe pattern set, so it cannot "
                f"be asked whether {PARITY_PROBE_PATTERN!r} was classified; the "
                f"mirror ghost's parity multiply is that orientation and a licence "
                f"that never looked at it cannot bind the mirror arm")
    if PARITY_PROBE_PATTERN not in tuple(patterns):
        return (f"the expansion licence was computed WITHOUT "
                f"{PARITY_PROBE_PATTERN!r}: stepping._shift_down's MIRROR branch "
                f"multiplies a complex64 plane by a python int parity "
                f"(stepping.py:1873-1875), which is none of the base four "
                f"orientations. Use "
                f"triton_kernels.folded_complex.parity_expansion_license")
    return None


def _shared_complex_refusal(fields: Any, pml: Any, grid: Any,
                            facts: dict) -> Any:
    """``coverage._complex_grid_refusal`` with this family's two clauses first.

    DELEGATION RATHER THAN A SECOND COPY, because that function is the one home of
    the inverted storage clause, the symmetry-consistency clause, the phase
    consistency rules, BFAST, beta, dispersion, chi2/chi3, ``stores_E`` and the
    HALVED int32 word bound -- and a clause added there should be inherited here
    rather than skipped.

    THE TWO PLACES THIS FAMILY DIVERGES, and each is stated HERE rather than passed
    as a flag, because that function SHORT-CIRCUITS and a flag would change more
    than one clause:

    * Dcyl IS REFUSED. ``admit_cylindrical=True`` is what makes that function admit
      the ``mirror`` boundary kind at all (its ``admitted_kinds`` switch), and this
      family NEEDS ``mirror`` -- so it is passed, and Dcyl is refused HERE, first,
      with this family's own reason. The kernel takes ``bc_x/bc_y/bc_z`` and has no
      code for the ``axis`` kind: Dcyl replaces the radial derivative with a prefix
      sum and adds the r_to_minus_r image rule (stepping.py:1877-1888).
    * A FOLD IS ADMITTED, which is ``admit_fold=True``, and what stands behind it is
      the real family's device verdict for the OFF-DIAGONAL row on a fold
      (``folded_offdiag_kernels.FOLDED_OFFDIAG_ADMISSION``: 240/240 folded cases
      per policy, both terminations, up to three simultaneous planes) plus this
      family's own gate. The fold's own clauses are :func:`_fold_reasons`.
    """
    if facts["cylindrical"]:
        return ("cylindrical (Dcyl): the r axis has its own ghost rule (the "
                "r_to_minus_r image, stepping.py:1877-1888) and its own axial "
                "extent, and this kernel's bc_* codes have no value for it")
    for axis in range(3):
        if facts["axis"][axis]:
            return f"axis {axis} is the cylindrical r = 0 axis"
    return _coverage._complex_grid_refusal(fields, pml, grid,
                                           admit_cylindrical=True,
                                           admit_fold=True)


def covers_complex_offdiag_pml_update_e(fields: Any, pml: Any, grid: Any,
                                        license: Any = None,
                                        subnormal_policy: Any = None) -> tuple:
    """Whether ``update_E_pml_complex_offdiag`` may serve this run.

    Returns ``(covered, reason)`` with ``reason`` naming the FIRST refusal -- the
    convention every predicate in this package uses.

    AN UNFOLDED GRID IS ADMITTED, and unlike ``folded_offdiag_kernels`` this family
    needs no separate composition verdict for it. That family had to narrow,
    because the certified flat off-diagonal kernel already claimed every unfolded
    row and admitting them twice would report an overlap that is an artifact of the
    question. Here NO family claims a complex off-diagonal ``update_E`` on any
    grid: all five corpus rows are unserved and the analyzer's disjointness check
    measures that rather than assuming it.

    THE DISJOINTNESS SEAM, all four sides:

    * against ``coverage.covers_real_pml_complex_constitutive(side='E')`` and
      ``complex_folded_kernels.covers_complex_folded_constitutive``: both refuse
      every off-diagonal run by name, and this predicate REQUIRES a surviving row
      slot;
    * against ``coverage.covers_real_pml_offdiag_constitutive`` and
      ``folded_offdiag_kernels.covers_folded_offdiag_composition``: both refuse
      complex64 storage by name, and the inverted storage clause inside
      ``_complex_grid_refusal`` REQUIRES it;
    * against :func:`covers_complex_no_pml_offdiag_update_e`: that one requires an
      inert layer and this one an active one -- the same ``_pml_is_active`` switch
      the array path branches on (stepping.py:980, :1015 against :1022);
    * against every dispersive family: ``_complex_grid_refusal`` refuses a
      registered susceptibility outright, so ``update_E``'s source here is D and
      never ``(D - sum P)``.
    """
    return _covers(fields, pml, grid, "pml", license, subnormal_policy)


def covers_complex_no_pml_offdiag_update_e(fields: Any, pml: Any, grid: Any,
                                           license: Any = None,
                                           subnormal_policy: Any = None) -> tuple:
    """Whether ``update_E_no_pml_complex_offdiag`` may serve this run.

    The two slots ``complex_no_pml_kernels`` names in
    :data:`complex_no_pml_kernels.WHAT_IS_NOT_BUILT` and refuses by name.

    ``stores_E`` FALSE IS REFUSED, through the shared clause set: ``update_E``
    returns at stepping.py:952-953 without writing anything, and admitting it would
    put a launch where the array path does nothing.

    ``Fields`` IN PML STORAGE MODE IS REFUSED while the layer is inert. That is a
    one-way switch (``Fields.enable_pml_storage``, fields.py:678-700) which also
    changes what ``get_H`` returns, and a ``Fields`` whose storage was switched on
    while the stepper's layer is inert reads H from an array ``update_H`` declines
    to write (stepping.py:944-945). The array path has the same hazard; this
    refuses the configuration rather than reproducing it, exactly as
    ``complex_no_pml_kernels._shared_refusal`` does.
    """
    return _covers(fields, pml, grid, "no_pml", license, subnormal_policy)


def _covers(fields: Any, pml: Any, grid: Any, arm: str,
            license: Any, subnormal_policy: Any) -> tuple:
    """The one clause chain, with the absorber question inverted per arm."""
    arm = normalized_arm(arm)
    refusal = _coverage.complex_expansion_refusal(license, subnormal_policy)
    if refusal is not None:
        return False, refusal
    refusal = _parity_licence_refusal(license)
    if refusal is not None:
        return False, refusal
    xp, backend = _coverage._backend(grid)
    if backend != "cupy":
        return False, "backend is not CuPy"
    active = pml is not None and bool(getattr(pml, "is_active", False))
    if arm == "pml":
        if not active:
            # Without an absorber ``update_E`` takes the plain assignment
            # (stepping.py:1022) -- no ``f_w``, no recurrence. A different kernel.
            return False, ("no active PML layer: update_E takes the plain "
                           "assignment at stepping.py:1022, which is "
                           "update_E_no_pml_complex_offdiag's")
        layer = pml
    else:
        if active:
            return False, ("an active absorber is installed: update_E runs the "
                           "dsigw accumulation (stepping.py:1015) and that is "
                           "update_E_pml_complex_offdiag's")
        if bool(getattr(fields, "_pml_active", False)):
            return False, ("Fields is in PML storage mode while the layer is "
                           "inert: get_H would serve a stored H that update_H "
                           "never writes, and this arm writes no f_w")
        # The shared clause set requires an ACTIVE layer, and every OTHER clause in
        # it is about the grid, the storage and the fields. Satisfying that one
        # clause AT THE INPUT is ``no_pml_curl._ActiveLayerProxy``'s technique and
        # its reason: the function short-circuits, so there is no accumulated
        # refusal list a filter could subtract from, and forgiving a refusal after
        # the fact would silently skip every clause behind it.
        from .complex_no_pml_kernels import _InertLayerProxy  # noqa: PLC0415

        layer = _InertLayerProxy(pml)
    facts, unreadable = _coverage._grid_facts(grid)
    if unreadable is not None:
        return False, unreadable
    refusal = _shared_complex_refusal(fields, layer, grid, facts)
    if refusal is not None:
        return False, refusal
    fold_problems = _fold_reasons(facts, grid)
    if fold_problems:
        return False, fold_problems[0]
    for axis, kind in enumerate(_coverage._boundary_kinds_from(facts)):
        # Stated positively: the three ghost rules the emitter writes, named.
        if kind not in FOLDED_BC_CODES:
            return False, (f"axis {axis} resolves to boundary {kind!r}, which has "
                           f"no kernel")
    refusal = _row_and_volume_refusal(fields, grid, facts, arm)
    if refusal is not None:
        return False, refusal
    if arm == "pml":
        # HALF-INTEGER ONLY, the E side's own sub-lattice (stepping.py:1015). ON A
        # FOLDED AXIS THE VECTOR IS BUILT AT THE FOLDED STORED EXTENT, which is
        # what ``_coefficient_vector_problem`` checks by SHAPE rather than by size
        # -- the clause that stops "the vectors are the right length" from being an
        # assumption on exactly the axis the fold moved.
        shape = facts["shape"]
        for axis, name in enumerate(("x", "y", "z")):
            for label in ("kps", "kms"):
                attribute = f"{label}_{name}_h"
                problem = _coverage._coefficient_vector_problem(
                    attribute, getattr(pml, attribute, None), xp, axis, shape)
                if problem is not None:
                    return False, problem
    return True, "covered"


# ---------------------------------------------------------------------------
# THE LAUNCHERS
# ---------------------------------------------------------------------------
#
# ``cupy`` is imported INSIDE these functions, never at module scope, because the
# predicates above are consumed by the coverage census and the census runs on a
# laptop.

#: The emitted sources a gate has replaced, keyed by ``(arm, row_mask, expansion)``
#: -- the mutation seam. A MUTABLE MODULE GLOBAL on purpose: the source-mutation
#: battery rewrites entries here and re-launches, and :func:`_get_kernel` re-reads
#: it on every call so a rewritten string is seen. A map memoized at first call
#: would hand back the pre-mutation bytes forever, which is "a leg reporting a pass
#: for a mutation it never applied" and which this track has recorded three times.
_OVERRIDES: Dict[Tuple[str, Tuple[int, ...], int], str] = {}


def kernel_source(arm: str, row_mask: Sequence[int], expansion) -> str:
    """The device text a launch would compile -- an installed override, or emitted."""
    key = (normalized_arm(arm), normalized_row_mask(row_mask),
           complex_emitter.normalized_expansion(expansion))
    override = _OVERRIDES.get(key)
    if override is not None:
        return override
    return complex_offdiag_source(key[0], key[1], key[2])


def set_kernel_source(arm: str, row_mask: Sequence[int], expansion,
                      source: Optional[str]) -> None:
    """Install (or with ``source=None`` remove) one mutated device string.

    THE GATE'S DOOR, and it is load-bearing: a mutation harness that could not
    route its own bytes through the launcher would launch the shipped kernel and
    report a pass for a defect it never introduced.

    Paired with :func:`clear_kernel_cache`: the compile memo keys on the SOURCE, so
    a mutated body is a miss and reaches NVRTC without the clear, but the clear is
    what makes the accounting -- how many compiles came from the mutated bytes --
    exact.
    """
    key = (normalized_arm(arm), normalized_row_mask(row_mask),
           complex_emitter.normalized_expansion(expansion))
    if source is None:
        _OVERRIDES.pop(key, None)
    else:
        _OVERRIDES[key] = source


def clear_kernel_cache() -> int:
    """Drop every memoized kernel; returns how many entries went."""
    return _clear_cache()


def _get_kernel(arm: str, row_mask: Sequence[int], expansion):
    """Compile on first use, memoized on (name, options, policy, source).

    THE ARM, THE ROW MASK AND THE EXPANSION ALL REACH THE KEY THROUGH THE SOURCE,
    which is already in it, so two specializations cannot be served each other's
    binary -- and neither can a mutation harness that rewrote the emitter's output
    be served the unmutated one.
    """
    import cupy as cp  # noqa: PLC0415 - device-only, in a laptop-importable module

    code = kernel_source(arm, row_mask, expansion)
    name = kernel_name(arm)
    key = _kernel_cache_key(name, False, _COMPILE_OPTIONS, code)
    return _get_or_compile(
        key, lambda: cp.RawKernel(code, name, options=_COMPILE_OPTIONS))


def _word_view(array: Any) -> Any:
    """The float32 word view of one complex64 volume -- the pointer the kernel gets.

    Refuses anything that is not a C-contiguous complex64 volume. A strided view
    would be read in the wrong order by the flat word index: a scrambled volume,
    not a launch failure, which is why this raises rather than reshaping.
    """
    import cupy as cp  # noqa: PLC0415

    if array is None:
        raise ValueError("expected a complex64 volume, got None")
    if array.dtype != cp.complex64:
        raise ValueError(f"expected a complex64 volume, got dtype {array.dtype}")
    if not array.flags.c_contiguous:
        raise ValueError(
            "complex volume is not C-contiguous; its float32 word view would not "
            "be either, and the kernel indexes it as a flat word array")
    return array.view(cp.float32)


def _require_no_aliasing(fields: Any, rows: Sequence[Any], arm: str) -> None:
    """Outputs pairwise distinct and disjoint from every input volume.

    THE COUPLING IS WHY THIS IS STRONGER THAN AN ELEMENT-WISE SUB-STEP'S NEEDS.
    That one reads each cell it writes and nothing else, so an alias would be
    merely wrong; this one re-reads the partner volumes at NEIGHBOUR offsets -- on
    a folded axis including interior stored row MIRROR_ROW -- while the outputs are
    being written, so an alias makes the answer depend on block schedule: wrong
    differently on each run, which no single comparison catches reliably.

    Equality of BASE addresses only: two overlapping views with different bases
    pass unseen, the accepted limitation every plan on this track shares.
    """
    outputs: Dict[int, str] = {}
    names = ("Ex", "Ey", "Ez") + (("f_w_Ex", "f_w_Ey", "f_w_Ez")
                                  if arm == "pml" else ())
    for name in names:
        address = _coverage._base_address(getattr(fields, name, None))
        if address is None:
            raise ValueError(
                f"{name} exposes no readable base address; an unverifiable output "
                f"is not accepted")
        if address in outputs:
            raise ValueError(f"{name} aliases {outputs[address]}; the outputs must "
                             f"be distinct arrays")
        outputs[address] = name
    inputs = [fields.Dx, fields.Dy, fields.Dz,
              fields.inverse_epsilon_for("Ex"),
              fields.inverse_epsilon_for("Ey"),
              fields.inverse_epsilon_for("Ez")]
    inputs.extend(rows)
    for volume in inputs:
        address = _coverage._base_address(volume)
        if address is not None and address in outputs:
            raise ValueError(
                f"an input volume aliases {outputs[address]}: the coupling "
                f"re-reads the partner volumes at neighbour offsets -- on a folded "
                f"axis including interior stored row {MIRROR_SOURCE_INDEX} -- "
                f"while the outputs are written, so the answer would depend on "
                f"block schedule")


def _validate_launch_arguments(shape: Sequence[int], codes: Sequence[int],
                               walls: Sequence[int], weights: Sequence[float],
                               flags: Sequence[int]) -> None:
    """Everything a launch would otherwise get silently wrong.

    Each of these is a WRONG ANSWER rather than a crash if it is not checked here:
    a NaN parity multiplies one plane into NaN, a folded axis that is also
    wall-masked zeroes a plane MEEP steps, a folded axis with too few cells reads a
    neighbouring plane, and a phase flag on a non-periodic axis applies a rotation
    the array path does not have. The predicates name all four, and a gate can
    reach a launcher without the predicate, which is why they are checked twice.
    """
    if len(codes) != 3 or any(int(code) not in FOLDED_BC_CODES.values()
                              for code in codes):
        raise ValueError(
            f"boundary codes must be three of "
            f"{sorted(set(FOLDED_BC_CODES.values()))}, got {tuple(codes)!r}")
    if len(walls) != 3 or any(int(flag) not in (0, 1) for flag in walls):
        raise ValueError(f"wall flags must be three of {{0, 1}}, got "
                         f"{tuple(walls)!r}")
    if len(flags) != 3 or any(int(flag) not in (0, 1) for flag in flags):
        raise ValueError(f"phase flags must be three of {{0, 1}}, got "
                         f"{tuple(flags)!r}")
    if len(weights) != 3:
        raise ValueError(f"ghost weights must be three floats, got "
                         f"{tuple(weights)!r}")
    for axis in range(3):
        if int(flags[axis]) and int(codes[axis]) != FOLDED_BC_CODES["periodic"]:
            raise ValueError(
                f"axis {axis} carries a Bloch phase flag but resolved to boundary "
                f"code {codes[axis]!r}; only a periodic wrap can carry a phase "
                f"(stepping._bloch_phases raises on the same configuration)")
        if int(codes[axis]) != BC_MIRROR_CODE:
            continue
        if int(walls[axis]):
            raise ValueError(
                f"axis {axis} is folded AND wall-masked; "
                f"_mask_metallic_wall_coupling abstains on a mirrored axis "
                f"(stepping.py:1282) and zeroing the fold plane the way the "
                f"metallic rule does costs 2.0e-02 (stepping.py:1269-1274)")
        if float(weights[axis]) not in (1.0, -1.0):
            raise ValueError(
                f"axis {axis} is folded with ghost weight {weights[axis]!r}; the "
                f"mirror ghost carries mirror_parity('D'+axis, axis, phase) == "
                f"-phase, exactly +1.0 or -1.0 (fields.py:117, "
                f"stepping.py:1873-1875)")
        if int(shape[axis]) <= MIRROR_SOURCE_INDEX:
            raise ValueError(
                f"axis {axis} is folded with {shape[axis]} stored cells; the "
                f"mirror ghost images stored row {MIRROR_SOURCE_INDEX}")


def _launch(kernel, fields: Any, rows: Sequence[Any], arm: str,
            tables: Optional[Dict[str, Any]], codes: Sequence[int],
            walls: Sequence[int], flags: Sequence[int],
            weights: Sequence[float], down: Sequence[float],
            up: Sequence[float]) -> str:
    """One launch. ``rows`` are the LIVE coefficient volumes, in slot order."""
    import numpy as np  # noqa: PLC0415 - only the scalar types are needed

    shape = tuple(fields.Ex.shape)
    nx, ny, nz = shape
    cells = nx * ny * nz
    blocks = (cells + _THREADS - 1) // _THREADS
    arguments: List[Any] = [_word_view(fields.Ex), _word_view(fields.Ey),
                            _word_view(fields.Ez)]
    if arm == "pml":
        arguments += [_word_view(fields.f_w_Ex), _word_view(fields.f_w_Ey),
                      _word_view(fields.f_w_Ez)]
    arguments += [_word_view(fields.Dx), _word_view(fields.Dy),
                  _word_view(fields.Dz),
                  fields.inverse_epsilon_for("Ex"),
                  fields.inverse_epsilon_for("Ey"),
                  fields.inverse_epsilon_for("Ez")]
    arguments.extend(rows)
    arguments += [np.int32(nx), np.int32(ny), np.int32(nz)]
    if arm == "pml":
        arguments += [tables["kps_x"], tables["kms_x"],
                      tables["kps_y"], tables["kms_y"],
                      tables["kps_z"], tables["kms_z"]]
    arguments += [np.int32(code) for code in codes]
    arguments += [np.int32(flag) for flag in walls]
    arguments += [np.int32(flag) for flag in flags]
    arguments += [np.float32(value) for value in weights]
    arguments += [np.float32(value) for value in down]
    arguments += [np.float32(value) for value in up]
    kernel((blocks,), (_THREADS,), tuple(arguments))
    return kernel_name(arm)


def _run(fields: Any, pml: Any, arm: str, expansion, tables, codes, walls,
         flags, weights, phases) -> str:
    """The one launch path, with the derive-or-override contract both arms keep.

    SUPPLY ``pml`` AND EVERY DERIVED ARGUMENT IS DECIDED HERE -- the half-integer
    tables through :func:`coverage.constitutive_sub_lattice`, the boundary codes
    through ``real_pml_boundary_kinds`` and :data:`FOLDED_BC_CODES`, the wall flags
    through the grid's own declaration, the parity through ``fields.mirror_parity``,
    the phase table through :func:`bloch_phase_table`. That is what makes "the
    launcher and the predicate cannot disagree" an enforced property rather than a
    convention a caller may keep.

    THE OVERRIDES ARE THE GATE'S DOOR and stay, keyword-only at the public
    entry points: a byte gate feeds synthetic tables no ``PML`` produces, and its
    mutations feed DELIBERATELY WRONG ones -- a mis-paired sub-lattice, a dropped
    wall flag, a fold handed the metallic code, a flipped parity, an unconjugated
    down phase. A launcher that could not take its own could not arm any of them.
    Passing ``pml`` together with an override, or neither, is refused: a caller with
    two answers to a one-answer question has a bug either way.
    """
    grid = fields.grid
    supplied = [value is not None
                for value in (codes, walls, flags, weights, phases)]
    if arm == "pml":
        supplied.append(tables is not None)
    if (pml is not None) and any(supplied):
        raise ValueError(
            "pml and an override were both supplied; the derived answer and the "
            "supplied one cannot both be the one this launch used")
    if pml is None and not all(supplied):
        raise ValueError(
            "pass exactly one of pml (every launch argument is derived from it and "
            "the grid) or ALL of codes, walls, flags, weights, phases"
            + (", tables" if arm == "pml" else "")
            + " (the gate supplies its own, including deliberately wrong ones)")
    if pml is not None:
        tables = complex_offdiag_tables(pml) if arm == "pml" else None
        codes = complex_offdiag_boundary_codes(grid)
        walls = _coverage.offdiag_wall_mask_flags(grid)
        weights = mirror_ghost_weights(grid)
        flags, down, up = bloch_phase_table(grid)
    else:
        down, up = phases
    mask = normalized_row_mask(_coverage.offdiag_row_mask(fields))
    rows = [volume for volume in _coverage.offdiag_row_volumes(fields)
            if volume is not None]
    _validate_launch_arguments(tuple(fields.Ex.shape), codes, walls, weights,
                               flags)
    _require_no_aliasing(fields, rows, arm)
    return _launch(_get_kernel(arm, mask, expansion), fields, rows, arm, tables,
                   codes, walls, flags, weights, down, up)


def update_E_complex_offdiag_fused_pml(fields: Any, expansion, pml: Any = None, *,
                                       tables: Dict[str, Any] = None, codes=None,
                                       walls=None, flags=None, weights=None,
                                       phases=None) -> str:
    """``stepping.update_E``'s tensor row product under an ACTIVE layer, one launch.

    ``phases`` is the ``(down_values, up_values)`` pair :func:`bloch_phase_table`
    returns as its second and third elements; ``flags`` is its first. They are
    separate arguments because a mutation that flips one flag without touching the
    values, or negates one value without touching the flags, is a defect this
    family has to be proof against and a single bundled argument could not arm.
    """
    return _run(fields, pml, "pml", expansion, tables, codes, walls, flags,
                weights, phases)


def update_E_complex_no_pml_offdiag(fields: Any, expansion, pml: Any = None, *,
                                    codes=None, walls=None, flags=None,
                                    weights=None, phases=None) -> str:
    """``stepping.update_E``'s tensor row product with NO absorber, one launch.

    ``pml`` may be an inert ``PML`` or ``None``; either way no coefficient table is
    read, because this arm has no recurrence to feed. Passing an ACTIVE layer is
    not refused HERE -- it is the predicate's clause and this launcher would simply
    compute the array path's no-absorber answer, which is the wrong answer for that
    run rather than a crash. The gate arms exactly that as a defect.
    """
    return _run(fields, pml, "no_pml", expansion, None, codes, walls, flags,
                weights, phases)
