"""COMPLEX cylindrical (Dcyl) curl sub-steps, in hand CUDA -- complex64 storage, any m.

WHAT IT REPLACES. ``stepping.step_B`` (:261) and ``stepping.step_D`` (:379) on a Dcyl
grid with complex64 storage -- ``abs(grid.m) >= 1`` since 2026-08-27, and ``m = 0``
since 2026-09-04 (the section THE m = 0 ARM below) -- and an active split-field PML --
everything except the radial prefix scan, which stays on the array path DELIBERATELY
and for the reason ``cylindrical_kernels``' docstring gives at length (the oracle is
``cupy.cumsum``'s float32 summation order and no other scan reproduces it).

THIS IS A THIRD CURL PAIR, NOT A WIDENING OF EITHER EXISTING ONE, and both existing
refusals are CORRECT and stay exactly where they are:

* ``cylindrical_coverage.covers_real_pml_cylindrical_curl`` refuses
  ``force_complex_fields`` by name -- right, because ``cyl_step_B_pml_real``
  reads a float32 stride and has no i*m/r pointer;
* ``coverage.covers_real_pml_complex_curl`` refuses Dcyl by name -- right, because
  ``step_B_pml_complex_bloch`` has no prefix pointer, no i*m/r row, no axis
  increment and no near-axis zeroing.

The sibling Triton tranche MEASURED that second refusal rather than arguing it: the
certified complex Cartesian body, run on a Dcyl |m| >= 1 grid with every cylindrical
clause stripped, produced 1,281,955 differing uint32 words against the array path, 56
of 80 rows differing (``triton_kernels/cylindrical_complex.py``, "THE CURL HALF DOES
NEED A NEW KERNEL"). This gate carries the same control (``stripped_control``) so the
number is this backend's rather than a citation.

=============================================================================
WHAT |m| >= 1 REQUIRES THAT NEITHER EXISTING PAIR HAS
=============================================================================

FIVE things. Everything else -- the ghost rule, the curl grouping on targets 0 and 1,
the ownership mask, the split-field recurrence, the word-pair addressing and the three
complex multiply orientations -- is the CERTIFIED COMPLEX Cartesian body, reached
through the SAME ``complex_emitter`` strings (``_HEAD``, ``_ARM_SOURCE``, ``_TAIL``),
IMPORTED and never copied, so the two cannot drift and the shared mutation battery's
needles arm here unchanged.

1. **THE RADIAL PREFIX SUM**, exactly as the m = 0 pair carries it and for the same
   reason (``cylindrical_prefix``, which is imported here too).
   * B side (stepping.py:327-375): Ep = Ey extended by ONE ZERO WALL ROW to
     (nr + 1, 1, nz), prefixed at ``ir0 = 0.0``; ``Bz``'s WHOLE curl becomes
     ``dtdx * (prefix_ext[1:] - prefix_ext[:-1])`` -- ONE subtract, ONE complex
     multiply, NOT the four-operand grouping, which is a different float32 number.
   * D side (:413-426): Hp = Hy prefixed at ``ir0 = 0.5``, substituted for the ``Dz``
     term's ``first`` source ONLY; ``Dx``'s Hy operands stay RAW.
   The prefix is a COMPLEX volume here where the m = 0 pair's was real; the scan is
   dtype-agnostic (it reads ``f_p.real.dtype`` for its weights, stepping.py:1308).

2. **THE i*m/r COUPLING** (``stepping._cylindrical_imr_term``, :674-724; call sites
   :348-355 on B and :430-437 on D). Per real/imaginary part MEEP adds
   ``the_m / r * g[i]`` with ``g`` the OTHER part of the partner, and the part swap IS
   multiplication by -i, so as one complex update
   ``delta = -i * s * 2m * Courant / r2 * g`` with ``r2 = 2*ir + iyee_shift_r(target)``.
   FOUR call sites and only four: ``Bx <- Ez`` at s = +1, ``Bz <- Ex`` at s = -1,
   ``Dx <- Hz`` at s = -1, ``Dz <- Hx`` at s = +1. Target 1 (``By``/``Dy``) gets
   nothing. THE PARTNERS ARE CENTER LOADS THE CERTIFIED BODY ALREADY MAKES -- ``Ez``/
   ``Hz`` is target 0's ``f_1`` and ``Ex``/``Hx`` is target 2's ``f_2`` -- so the term
   costs no new pointer and no extra traffic. The row vector itself is HOST-BUILT and
   bound as a device array; see :func:`imr_coefficient_row` for why it may not be
   recomputed in the kernel.
   The term is added AFTER the dtdx curl and BEFORE the ownership mask, which is the
   array path's own order (:348 sits between :347's curl and :370's mask).

3. **THE |m| = 1 AXIS-ROW INCREMENTS** (:625-645 B, :524-557 D), which REPLACE curl
   row 0 rather than adding to it, and do so AFTER the ownership mask (:370-372,
   :451-453 assign ``curl[_face(0, 0)] = -increment``):
   * B: ``Bx`` row 0 takes ``-((-dtdx)*(Ep - Ep_zup) - (1j*m*dtdx)*Ez[row 1])``, where
     ``Ez[row 1]`` is the FIRST OFF-AXIS r row (MEEP's ``f[Ez][1-cmp] + (nz+1)``);
   * D: ``Dy`` row 0 takes ``-(dtdx*(Hr - Hr_zdn - 2.0*Hz))``, the factor 2 being the
     doubled-ivec compensation.
   FOLDED INTO THE CURL, not post-added, and that was MEASURED by the array path
   rather than chosen: stepping.py:553-572 records a plain post-add as Ep 4.6e-01 /
   Hr 2.39 wrong under r+z PML at |m| = 1, because Dp's ladders run along Z and R and
   both carry nonzero sigma on the axis row.

4. **THE PER-|m| AXIS RULES**, applied to the STORED value after the recurrence,
   exactly where the array path applies them (after the whole term loop, :375-376 and
   :456-457 calling ``_cylindrical_axis_zero_B`` / ``_D``):
   * |m| = 1 -- B side: NOTHING (``_cylindrical_axis_zero_B`` :648-671 branches on
     ``m == 0`` and ``abs(m) > 1`` only). D side: ``Dz[r=0] = 0``, the FIELD only,
     never ``fu_Dz``.
   * |m| >= 2 (:560-598, :648-671, rows from :601-622) -- ALL THREE components of the
     family AND their ``fu_`` auxiliaries held at zero on rows ``[0:ZERO_ROWS]``,
     where ZERO_ROWS is ``|m|`` (MEEP's default stability hack) or 1 (the
     ``accurate_fields_near_cylorigin`` branch). The component set is the POST-#3164
     one (upstream 593a4b42, first released in 1.33.0); the 1.29-era code zeroed only
     Dp/Dz and Br and the two references genuinely differ (:573-578).
   The array path assigns the INTEGER 0 to a complex64 array, which is the word pair
   (+0.0f, +0.0f) -- ``cf_zero()``, never a sign-carrying zero.

6. **THE m = 0 ARM** (2026-09-04), the third ``m_class`` value. A complex-storage
   m = 0 run is constructible -- ``force_complex_fields`` is an independent switch --
   and the corpus carries one (``examples:dipole_in_vacuum_cyl_off_axis.py``, grid
   (150, 1, 300), m = 0, PML, two electric sources). Until this arm landed the row
   fell BETWEEN the two cylindrical products: the real pair refused the storage and
   this one refused the order, so no product could serve either of its seams. What
   m = 0 requires is the REAL pair's rules (``cylindrical_kernels``) in cf arithmetic,
   read off ``stepping`` rather than off that kernel:
   * NO i*m/r term (:348, :430 -- ``if cylindrical and grid.m != 0``). The four call
     sites are GUARDED on ``m_class != 0`` rather than fed a zero coefficient row,
     because ``curl - (0 * f)`` is NOT the identity on a ``-0.0`` curl
     (``-0.0 - (-0.0)`` is ``+0.0``); the branch is on a kernel scalar, so it is
     uniform across the launch and moves no bit at |m| >= 1.
   * NO curl-row fold (``_cylindrical_axis_increment_B`` :636 and ``_D`` :540 both
     return ``None`` at m = 0).
   * B side, after the recurrence (:657-659): ``Bx[r=0] = 0``, the FIELD only.
   * D side, after the recurrence (:583-587): ``Dz[r=0] += (4.0*(dt/dx)) * Hy[r=0]``
     -- a weak Python float meeting a complex64 volume, i.e. ``mul_coefficient_left``
     with the host-rounded ``float32(4.0*dtdx)`` (:func:`axis_coefficient`), then the
     complex add with ``Dz`` on the left -- followed by ``Dy[r=0] = 0``. The post-add
     is NOT folded into the curl; the array path MEASURED the fold breaking m = 0
     under PML (:545-551). ``Hy`` is the raw stored Hp this kernel already binds.
   The two |m| >= 1 branches are untouched: the m = 0 tail is a THIRD guarded block
   after them, and the D signature gains ONE scalar (``axis_coef``), bound at every m
   and read only under ``m_class == 0``.

5. **NO BLOCH PHASE AT ALL.** ``Grid`` refuses a k on r or phi (grid.py:654-658) and
   the predicate refuses one on z, so every ``ph_*`` this kernel passes ``cshift_*`` is
   0 and the rotation never happens. The flags stay in the SIGNATURE rather than being
   compiled out, because they are the certified helper's own parameters and dropping
   them would mean a second copy of ``cshift_up``. Admitting a z Dcyl phase later is a
   predicate widening plus a gate row, not a kernel change.

WHAT NEEDED NO NEW CODE AT ALL, RE-ESTABLISHED FOR COMPLEX STORAGE:

* **The r axis compiles as METALLIC.** ``stepping._shift_up``'s ``CYL_AXIS`` arm IS
  the ``METALLIC`` arm, character for character (:1781-1783) -- a hard zero at the far
  r face, the PEC wall a Dcyl cell carries at ``r_max``. ``_shift_down``'s ``CYL_AXIS``
  arm is NOT (:1830-1846): it images stored row 0 with a direction sign and the
  ``(-1)^m`` phase. THE NEAR GHOST IS NOT IMPLEMENTED and that is a decision with a
  proof behind it, re-established at every |m| this family carries: the only terms
  taking a shift-down along r are ``Dy`` (partner Hz, second operand) and ``Dz``
  (partner Hp, first operand), and BOTH have r-Yee shift 0, so
  ``_mask_non_owned_cells`` (:1898-1902) zeroes their curl at row 0 -- the only row
  the near ghost writes. At |m| = 1 that row is then replaced outright by the axis
  increment; at |m| >= 2 the field AND its ``fu`` are zeroed there. NO BYTE GATE CAN
  CERTIFY THAT CHOICE from inside the kernel, so it is stated here and the gate carries
  BOTH halves as ARMED NULLS (the direction sign and the ``(-1)^m`` factor). If a run
  ever reports either CAUGHT, the ownership mask has broken, not the ghost.
* **The ownership mask is the certified body's.** With r compiled METALLIC,
  ``if (bc_x == BC_METALLIC && i == 0) curl = cf_zero();`` masks exactly the cells the
  ``is_axis`` clause masks.
* **The constitutive sub-steps need nothing.** ``stepping.update_H`` (:907-924) and
  ``update_E`` (:926-993) carry NO cylindrical branch -- grep them: "cylindrical",
  "is_axis" and "m" do not appear -- so the certified
  ``update_H_pml_complex_bloch`` / ``_E_`` already compute them. THAT IS NOT
  THIS MODULE'S CLAIM TO MAKE: it ships no constitutive predicate and admits no
  constitutive slot. Whoever widens ``coverage.covers_real_pml_complex_constitutive``
  to admit Dcyl owes the identity leg that licenses it.

=============================================================================
THE ARM IS AN ARGUMENT -- the same package boundary the complex family draws
=============================================================================

Every entry point here takes ``expansion``. It has no default, this module derives it
from nothing and never opens a probe artifact: the arbiter is
``triton_kernels.complex_fields.expansion_license`` and a second spelling of it is one
too many. A WRONG ARM IS A WRONG ANSWER rather than a crash -- both arms compile, both
run, and they differ in the last bits of about a quarter of the words -- so
:func:`complex_emitter.normalized_expansion` refuses rather than defaulting.

THE ARM IS DEGENERATE AT THIS FAMILY'S TWO NEW CALL SITES, and that is MEASURED here
rather than assumed. Both new coefficients carry a REAL WORD THAT IS A ZERO:

* the i*m/r row's real word is exactly +0.0 for every m -- measured on this repo's own
  builder at m in {1, -1, 2, 3, 5} and both signs, real word 0x00000000 throughout --
  because ``stepping`` spells the numerator ``(-1j) * (sign*2*m*dtdx)`` and CPython's
  complex product then computes ``re = (-0.0)*X - (-1.0)*0.0``, whose cross-term
  subtraction LAUNDERS the sign to +0.0 for both signs of X;
* the |m| = 1 B-side scalar ``complex64(1j*(m*dtdx))`` carries -0.0 for m < 0
  (measured: m = -1 at dtdx = 0.35 and 0.5 both give real word 0x80000000), because
  ``1j * X`` computes ``re = 0.0*X - 1.0*0.0``, which PRESERVES the sign.

With ``c_re = +-0.0`` the product ``c_re * z_re`` is exact, so
``fma(c_re, z_re, (c_im*z_im) * -1.0f)`` and ``(c_re*z_re) - (c_im*z_im)`` are the same
single-rounding operation. BOTH ARMS ARE STILL EMITTED, because the arm is a measured
platform constexpr bound once for the whole family and a helper honouring one arm
would stop honouring the contract the moment a call site with a nonzero real
coefficient is added -- and because the recurrence's own multiplies (``mul_field_left``,
``mul_coefficient_left``) are NOT degenerate and come from the certified strings.

THE ZERO CROSS TERMS ARE KEPT, and here that is a transcription rather than a
byte-visible choice: ``curl - m`` can carry the difference only at a -0.0 minuend, and
every i*m/r minuend in this kernel is provably never -0.0 (target 0's curl leads with
the phi self-difference, an exact +0.0 on a one-cell axis, and ``+0.0 + y`` is +0.0
even at ``y = -0.0``; target 2's is a prefix difference led the same way). The gate
runs a MINUEND CENSUS over every case and records the count, so the premise is measured
per run rather than inherited.

=============================================================================
WHAT IS COMPILE-TIME AND WHAT IS NOT
=============================================================================

Split exactly as ``complex_emitter`` splits its own axes, by what each does to the
ARITHMETIC:

* ``bc_x`` / ``bc_y`` / ``bc_z``, ``ph_*``, ``m_class`` and ``zero_rows`` are BRANCH
  axes -- they select an index, a predicated zero, or whether a statement runs -- so
  they are ordinary runtime ``int`` arguments. That is also what keeps the shared
  ``drop_metallic_mask`` mutation armed on this source unchanged, and what lets the
  gate hand this pair a deliberately wrong boundary code.
* The SIDE (B vs D) is STRUCTURAL -- different stencils, different masks, a different
  coefficient sub-lattice, a different cylindrical substitution and a different axis
  increment -- so it is two kernels, exactly as every sibling pair is two.
* ``EXPANSION`` is an ARITHMETIC axis and stays the emitter's only parameter.

=============================================================================
CUDA-SPECIFIC, AND WHY THIS MODULE IS CUPY-FREE AT IMPORT
=============================================================================

* ``_COMPILE_OPTIONS``' ``--fmad=false`` is CORRECTNESS, not tuning: ``(fu*kms) - curl``
  and ``(f*kms_u) + fu_new`` are both contraction candidates the array path rounds
  twice, and the FMA_V1 arm's own fusions are spelled ``__fmaf_rn``, a single
  ``fma.rn.f32`` whatever the flag says. The gate scores an unguarded control.
* THE DEVICE STRINGS ARE PURE ASCII. ``cupy.cuda.compiler.compile_using_nvrtc`` writes
  the source through a bare ``open(..., 'w')`` (compiler.py:368), so the bytes go
  through the interpreter's LOCALE encoding -- ASCII under C/POSIX, which is what a
  non-interactive shell on the validation host gets. Two em-dashes killed a sibling
  kernel at its first launch on 2026-08-15. :func:`cylindrical_complex_source` asserts
  it.
* NO FLOATING-POINT DIVISION anywhere: the i*m/r row is bound as an array precisely so
  the kernel performs none. (Integer ``/`` and ``%`` on the lane index are exact and
  are not what that rule is about.)
* ``import cupy`` IS LAZY, inside the compile and launch helpers, and that is
  deliberate. The source is a FUNCTION of the arm, so it cannot be read out of this
  file with ``ast`` the way ``test_cylindrical_pml_real.py`` reads the m = 0 strings --
  it has to be IMPORTED to be checked, and the merge bar is a machine with no GPU. The
  emitter and the launcher stay in ONE file so the kernel signature and the launch
  tuple sit side by side; ``RawKernel`` does not check them against each other, and a
  swapped pair is a wrong answer at a wrong pointer.

=============================================================================
THE ONE fma-OUTPUT NEGATION, AND THE HOST-LEG OPTIMIZATION LEVEL IT FORCES
=============================================================================

``curl[row 0] = -increment`` (stepping.py:399, :481 -- a NumPy unary minus on a
complex64 row, i.e. both sign bits flipped) is the ONLY place in this package that
negates the OUTPUT of a floating-point operation. Everywhere else ``* -1.0f``
appears it is on a plain product used as an fma ADDEND, which is why no sibling has
had to say anything about it. The spelling here is the family's standard ``* -1.0f``
per word, which is IEEE-exact including on signed zeros.

WHAT IS NOT EXACT IS THE HOST C++ COMPILER, and it was MEASURED rather than
suspected. Compiling the emitted source with Apple clang 21 at ``-O1`` and driving
it on NumPy (the gate's ``host`` leg), a zero-initialised |m| = 1 Dcyl grid under a
THIN absorber comes out ONE uint32 word away from the array path -- ``fu_Dy`` at
r = 0, k = 0, real plane, ``-0.0`` where the array path holds ``+0.0`` -- because
the optimizer rewrites ``-(fma(a, b, c))`` as ``fma(-a, b, -c)``, an identity that
is exact for every finite value and WRONG on signed zeros (it negates the addend
before the sum instead of after it). The sweep that settles it, 96 cases per cell
over two sub-steps x four seedings x two absorber depths x six grids:

    -O0   `* -1.0f` 0/96 diverged      sign-bit-xor negation 0/96 diverged
    -O1   `* -1.0f` 3/96 (1 word each) sign-bit-xor negation 3/96 (1 word each)
    -O2   `* -1.0f` 0/96 diverged      sign-bit-xor negation 3/96 (1 word each)

Every divergence is the same cell of the same class: ``step_D``, ``zero_init``,
thin absorber, |m| = 1. NO SPELLING SURVIVES EVERY LEVEL -- an explicit sign-bit
flip does no better and is wrong at ``-O2`` where ``* -1.0f`` is right -- so the
optimization level is the variable, not the source, and the transcription is
established by the level at which the compiler executes it literally.

THE HOST LEG THEREFORE COMPILES AT ``-O0`` and the gate records why. That leg
"compiles nothing and certifies nothing" in any case; what it measures is the
TRANSCRIPTION, and ``-O0`` is the setting under which the host compiler runs the
transcription rather than an algebraic rewrite of it. NVRTC IS A DIFFERENT COMPILER, and it was ASKED rather
than assumed: the device leg's ``zero_init`` cases at |m| = 1 -- the exact cell the
host compiler loses -- came back BIT-IDENTICAL, 129/129 cases under BOTH float32
subnormal policies on an RTX A6000
(``parity/meep_gpu/results/cuda_cylindrical_complex_2026-08-20_v2/``). So NVRTC does
NOT share the fold and ``* -1.0f`` is the right spelling on the target. The finding
stands anyway, because the host leg is the merge bar and a future NVRTC that DID fold
would show up as exactly this word; if the device leg ever reports it, the fold has
become real on CUDA and this family is not releasable until the spelling is made
opaque to it.

=============================================================================
NOT CERTIFIED, NOT WIRED, NOT DISPATCHED
=============================================================================

Neither kernel is in ``certification.json`` and neither may be counted as coverage
until ``parity/meep_gpu/gate_cuda_cylindrical_complex.py`` releases against it on
hardware. ``fastpath.plan_fast_path`` is untouched and still returns ``None`` on every
branch; nothing in ``meep_gpu/`` imports this package at all. :data:`UNCERTIFIED_KERNELS`
is this file's half of the partition ``step_curl_kernels.py`` maintains.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

# THE CERTIFIED COMPLEX STRINGS ARE IMPORTED, NOT COPIED. ``_HEAD`` carries the
# word-pair addressing and the complex add/sub/zero, ``_ARM_SOURCE`` the three multiply
# orientations per arm, and ``_TAIL`` the two ghost helpers, the split-field recurrence
# and the constitutive one. A second copy here would be a second thing to keep in step
# AND would silently un-arm the shared mutation battery, whose needles are those exact
# strings.
from .complex_emitter import (
    EXPANSION_NAMES,
    EXPANSIONS,
    _ARM_SOURCE,
    _HEAD,
    _TAIL,
    normalized_expansion,
)
# The prefix and the predicate live in CuPy-free siblings and are imported back here,
# exactly as the m = 0 pair imports them: pure array work and pure decision logic, both
# of which have to be exercisable at the merge bar.
from .cylindrical_prefix import cylindrical_prefix


#: RECORDED 2026-08-27. The gate was re-run on the GPU host and RELEASED under both
#: float32 subnormal policies; the block is ``cuda_cylindrical_complex_2026-08-27`` in
#: ``certification.json``, and the two halves landed together as the partition
#: test requires. The subject module was checked UNCHANGED since that run before
#: the record was landed, so this names the bytes that ship.
CERTIFIED_KERNELS = (
    "cyl_step_B_pml_complex",
    "cyl_step_D_pml_complex",
)

#: NOT CERTIFIED. Neither kernel in this file has a gate verdict in
#: ``certification.json``; both are here to BE gated. This tuple is the file's half of
#: the partition ``step_curl_kernels.py`` maintains between certified and uncertified
#: kernels, restated so a reader who arrives at this file first is told the status
#: before reading the sources.
UNCERTIFIED_KERNELS: Tuple[str, ...] = ()
#: The compile options this module passes NVRTC. ``--fmad=false`` is correctness on
#: this sub-step; the gate substitutes the empty tuple as a control and reports whether
#: the guard changed an answer.
_COMPILE_OPTIONS: Tuple[str, ...] = ('--fmad=false',)

#: Lanes per block, one COMPLEX CELL per lane -- the same 256 every complex sibling
#: landed on.
_COMPLEX_THREADS = 256

#: The two curl sub-steps, their kernel entry points and whether the side differences
#: DOWN (D) or UP (B).
KERNELS: Dict[str, Tuple[str, bool]] = {
    "step_B": ("cyl_step_B_pml_complex", False),
    "step_D": ("cyl_step_D_pml_complex", True),
}

CURL_SUB_STEPS: Tuple[str, ...] = ("step_B", "step_D")

#: Which Yee sub-lattice each sub-step's coefficient vectors come from. The B curl
#: reads the HALF-INTEGER positions (stepping.py:404) and the D curl the INTEGER ones
#: (:486). Swapped, it is a half-cell error in the absorber profile -- converged,
#: smooth and wrong -- which is why the gate carries it as a host mutation.
HALF_INTEGER: Dict[str, bool] = {"step_B": True, "step_D": False}

#: Which volumes each sub-step writes and reads, in the kernel's argument order
#: (``stepping.B_CURL_TERMS`` / ``D_CURL_TERMS``, :213-223).
CURL_ARRAYS: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {
    "step_B": (("Bx", "By", "Bz"), ("Ex", "Ey", "Ez")),
    "step_D": (("Dx", "Dy", "Dz"), ("Hx", "Hy", "Hz")),
}

#: The i*m/r call sites -- ``(target index, target name, sign)`` per sub-step, read off
#: ``stepping.step_B``:348-355 and ``step_D``:430-437. Target 1 gets no term on either
#: side; the ROW is indexed by the TARGET's own radial Yee shift, so the name matters
#: as much as the index.
IMR_TERMS: Dict[str, Tuple[Tuple[int, str, float], ...]] = {
    "step_B": ((0, "Bx", +1.0), (2, "Bz", -1.0)),
    "step_D": ((0, "Dx", -1.0), (2, "Dz", +1.0)),
}

#: Which target each sub-step's |m| = 1 axis-row increment REPLACES at r = 0:
#: ``Bx`` on the B side (stepping.py:674 returns ``"Bx"``) and ``Dy`` on the D side
#: (:585 returns ``"Dy"``).
AXIS_INCREMENT_TARGET: Dict[str, int] = {"step_B": 0, "step_D": 1}

#: ``m_class`` codes, matched by the device text. THREE ARMS SINCE 2026-09-04. The
#: m = 0 arm was DELIBERATELY absent until then -- m = 0 was ``cylindrical_kernels``'
#: real-storage pair's arithmetic and the predicate refused it by name -- but a
#: complex-storage m = 0 run is constructible (``force_complex_fields`` is an
#: independent switch, and the corpus row ``dipole_in_vacuum_cyl_off_axis.py`` IS
#: one: grid (150, 1, 300), complex64, m = 0, PML), and it fell BETWEEN the two
#: cylindrical products: the real pair refused the storage and this one refused
#: the order. The m = 0 arithmetic under complex storage is the REAL pair's rules
#: in cf arithmetic -- no i*m/r term (the guard is a uniform kernel-scalar branch, so
#: a zero coefficient row is never multiplied in: ``curl - (0 * f)`` is NOT the
#: identity on a ``-0.0`` curl), no curl-row fold, ``Bx[r=0] = 0`` after the B
#: recurrence, and after the D recurrence the on-axis ``Dz`` POST-ADD of
#: ``4*Courant*Hp`` (``stepping._cylindrical_axis_zero_D``:583-585, the coefficient
#: host-rounded exactly as ``cylindrical_kernels.axis_coefficient`` rounds it) and
#: ``Dy[r=0] = 0``. See THE m = 0 ARM in the module docstring.
M_ZERO = 0    # m = 0: Bx[0] = 0 on the B side; Dz[0] += 4*Courant*Hp[0], Dy[0] = 0 on D
M_ONE = 1     # |m| = 1: axis-row increments on Bx and Dy, plus Dz[0] = 0
M_MANY = 2    # |m| >= 2: near-axis zeroing of all six volumes


# ---------------------------------------------------------------------------
# Host-side constants that must be computed the array path's way
# ---------------------------------------------------------------------------

def m_class(m: int) -> int:
    """``m_class`` for one azimuthal order -- ``M_ZERO``, ``M_ONE`` or ``M_MANY``.

    m = 0 is an arm of this family since 2026-09-04 (complex64 storage at m = 0 has
    no other product: the real pair reads a float32 stride). ``stepping``'s own
    branches are ``m == 0`` / ``abs(m) == 1`` / else, and this is that partition.
    """
    if int(m) == 0:
        return M_ZERO
    return M_ONE if abs(int(m)) == 1 else M_MANY


def axis_coefficient(dtdx: float) -> Any:
    """The m = 0 on-axis ``Dz`` multiplicand, rounded ON THE HOST.

    ``stepping._cylindrical_axis_zero_D``:585 writes ``Dz[axis] += (4.0 * (dt/dx)) *
    Hy[axis]``: the coefficient is formed in float64 and meets the complex64 volume
    as a WEAK Python float, so NEP-50 casts that one scalar to complex64 -- the real
    word ``float32(4.0 * dtdx)``, the imaginary word +0.0 -- before the FULL complex
    product with the coefficient on the left (``mul_coefficient_left``). Binding the
    already-rounded word is the literal transcription and the same rounding the real
    pair's ``cylindrical_kernels.axis_coefficient`` performs; recomputing ``4.0f *
    dtdx`` in-kernel from an fp32 ``dtdx`` happens to agree because scaling by four
    is exact, which is a fact about this coefficient and not a licence to recompute.
    """
    import numpy  # noqa: PLC0415 - stdlib-adjacent, imported at the one call site

    return numpy.float32(4.0 * float(dtdx))


def zero_rows(m: int, accurate_fields_near_cylorigin: bool) -> int:
    """How many near-axis rows the |m| >= 2 rule holds at zero.

    ``stepping._cylindrical_axis_rows`` (:601-622): ``slice(0, 1)`` on the ACCURATE
    branch (an ordinary axis boundary condition and nothing else, stable only below
    Courant ~1/(|m| + 0.5), which ``Grid`` refuses above) and ``slice(0, |m|)`` on
    MEEP's default stability hack. Zero at |m| = 1, where neither branch applies.
    """
    if abs(int(m)) < 2:
        return 0
    return 1 if bool(accurate_fields_near_cylorigin) else abs(int(m))


def imr_coefficient_row(xp: Any, target: str, sign: float, m: int, dtdx: float,
                        rows: int, dtype: Any) -> Any:
    """The per-r i*m/r coefficient, built the array path's way and ONLY that way.

    ``stepping._cylindrical_imr_term._build`` (:713-718), transcribed step for step.
    Every step of it is load-bearing:

    * the ``arange`` is FLOAT64 and the doubled coordinate is ``2*ir + iyee_r`` with
      ``iyee_r`` the TARGET's own radial Yee shift (``fields.IYEE_SHIFTS``). Using 0
      for every target is a defect the gate arms;
    * the clamp is ``maximum(.., 1.0)``, a DOMAIN GUARD rather than arithmetic: it
      bites only where the doubled coordinate is 0, i.e. row 0 of a shift-0 target
      (``Bx``, ``Dz``), whose curl the ownership mask zeroes anyway;
    * the numerator is spelled ``(-1j) * (sign * 2.0 * m * dtdx)``, whose real word
      launders to +0.0 for both signs of the real factor (module docstring);
    * the DIVISION is float64 on the host and the result is rounded ONCE by
      ``.astype(dtype)``. Recomputing it in-kernel from an fp32 ``dtdx`` is a
      different number, and the division would additionally have to be correctly
      rounded (an f32 ``/`` lowers to ``div.rn.f32``, whose ptxas expansion carries
      ``.FTZ`` range checks in SASS regardless of the PTX modifier). Binding the row
      makes the whole question disappear: THE KERNEL PERFORMS NO FLOATING-POINT
      DIVISION AT ALL.

    Returned as a length-``rows`` VECTOR, not the (rows, 1, 1) block the array path
    reshapes to -- same words, and the kernel indexes it by ``i``.
    """
    from ..fields import IYEE_SHIFTS  # noqa: PLC0415 - a CuPy-free sibling, imported late

    iyee_r = IYEE_SHIFTS[target][0]
    r_doubled = 2 * xp.arange(int(rows), dtype=xp.float64) + iyee_r
    divisor = xp.maximum(r_doubled, 1.0)
    row = ((-1j) * (sign * 2.0 * int(m) * float(dtdx))) / divisor
    return xp.ascontiguousarray(row.astype(dtype))


def axis_increment_scalars(m: int, dtdx: float) -> Tuple[float, Tuple[float, float]]:
    """The two |m| = 1 B-side host scalars, rounded as the array path rounds them.

    ``stepping._cylindrical_axis_increment_B`` (:643-644) writes

        (-dtdx) * (ep[0] - ep_above[0]) - 1j * (m * dtdx) * ez_off_axis

    where both scalars meet a complex64 array as WEAK Python scalars, so each is
    converted to complex64 once and the multiply is a FULL complex product with the
    zero cross terms (``np.multiply`` carries only ``FF->F`` for complex).

    Returns ``(minus_dtdx, (re, im))``. The first is a PYTHON FLOAT and stays one --
    the ``mul_coefficient_left`` orientation. The second is ``1j * (m * dtdx)``, whose
    REAL WORD IS A SIGNED ZERO: ``re = 0.0*X - 1.0*0.0`` is -0.0 for X < 0, the
    OPPOSITE laundering from the i*m/r row's ``(-1j) * X``. Both are host-rounded and
    passed through; neither is synthesized in-kernel.
    """
    import numpy  # noqa: PLC0415 - stdlib-adjacent, imported at the one call site

    second = numpy.complex64(1j * (int(m) * float(dtdx)))
    return (float(numpy.float32(-float(dtdx))),
            (float(numpy.float32(second.real)), float(numpy.float32(second.imag))))


# ---------------------------------------------------------------------------
# The device source
# ---------------------------------------------------------------------------

#: The ONE helper this family adds to the certified strings, and it is an ALIAS rather
#: than a new spelling. ``rotate_field_left`` in ``complex_emitter``'s arm block is the
#: arm's FULL complex product with the LEFT operand first; the certified family names
#: it for its only call site (the Bloch phase), and here the left operand is a per-r
#: COEFFICIENT instead. Same operation, same arm, same bytes -- so it is aliased,
#: never re-spelled, and a change to either arm reaches both call sites at once.
_CYLINDRICAL_HELPERS = r'''
// (c_re + i*c_im) * z with the COEFFICIENT on the LEFT -- stepping.py:717's
// `factor * partner_values` and :644's `scalar * ez_off_axis`. See the note above.
__device__ __forceinline__ cf mul_complex_left(cf c, cf z) {
    return rotate_field_left(c, z);
}

'''  # stepping.py live lines for the frozen device-text citation(s) in this string: 717->744, 644->673

#: The B side. Bx and By keep the certified Cartesian stencil; Bz's whole curl is the
#: radial forward difference of the WALL-EXTENDED prefix; the i*m/r term rides on
#: targets 0 and 2; the |m| = 1 increment replaces Bx's curl row 0; and the |m| >= 2
#: rule zeroes all six volumes near the axis after the recurrence.
_CYL_STEP_B_TEMPLATE = r'''
extern "C" __global__ void cyl_step_B_pml_complex(
    float* __restrict__ Bx, float* __restrict__ By, float* __restrict__ Bz,
    float* __restrict__ fu_Bx, float* __restrict__ fu_By, float* __restrict__ fu_Bz,
    const float* __restrict__ Ex, const float* __restrict__ Ey,
    const float* __restrict__ Ez,
    const float* __restrict__ pfx,
    const float* __restrict__ imr0, const float* __restrict__ imr2,
    int nx, int ny, int nz, float dtdx,
    float minus_dtdx, float inc_re, float inc_im,
    const float* __restrict__ kms_x, const float* __restrict__ sinv_x,
    const float* __restrict__ kms_y, const float* __restrict__ sinv_y,
    const float* __restrict__ kms_z, const float* __restrict__ sinv_z,
    int bc_x, int bc_y, int bc_z,
    int ph_x, int ph_y, int ph_z,
    int m_class, int zero_rows
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny * nz) return;

    // Strides in COMPLEX CELLS. cf_load does the word doubling.
    const int sx = ny * nz;
    const int sy = nz;
    const int sz = 1;

    int k = idx % nz;
    int j = (idx / nz) % ny;
    int i = idx / (ny * nz);

    // This family carries NO Bloch phase (module docstring, point 5): Grid refuses a
    // k on r or phi and the predicate refuses one on z, so every ph_ flag is 0 and
    // cshift_* never multiplies. The pair is passed anyway because it is the
    // certified helper's own signature.
    cf noph = cf_zero();

    // Bx: curl_x = dEz/dy - dEy/dz; dsig=y, dsigu=z; iyee=(0,1,1) -> mask on x.
    // f_1 IS the i*m/r partner Ez and f_2/ss ARE the |m|=1 increment's Ep pair, so
    // neither addition costs a load.
    {
        cf f_1 = cf_load(Ez, idx);
        cf sf = cshift_up(Ez, idx, j, ny, sy, bc_y, ph_y, noph);
        cf f_2 = cf_load(Ey, idx);
        cf ss = cshift_up(Ey, idx, k, nz, sz, bc_z, ph_z, noph);
        cf curl = mul_coefficient_left(dtdx, cf_add(cf_sub(sf, f_1), cf_sub(f_2, ss)));
        // stepping.py:349-350 -- Bx <- +1 * (i*m/r) * Ez, BEFORE the mask. The array
        // path adds `-(factor * partner)`; `curl - (factor * partner)` is the same
        // bits by IEEE-754 (subtraction IS addition of the negation).
        // GUARDED AT m = 0 (stepping.py:348 `if cylindrical and grid.m != 0`): the
        // array path adds NO term there, and `curl - (0 * f)` is not the identity
        // on a -0.0 curl (-0.0 - (-0.0) is +0.0), so the row is skipped, never
        // multiplied in. m_class is a kernel scalar: the branch is uniform.
        if (m_class != 0) {
            curl = cf_sub(curl, mul_complex_left(cf_load(imr0, i), f_1));
        }
        if (bc_x == BC_METALLIC && i == 0) curl = cf_zero();
        // stepping.py:370-372 -- the |m|=1 axis row REPLACES the masked row, never
        // accumulates: after the mask it is exactly +0.0, and `+0.0 + x == x` for
        // every x EXCEPT -0.0. Ez[row 1] is the FIRST OFF-AXIS r row (:642's
        // xp.take(Ez, 1, axis=0)); the predicate requires nx >= 2 so this load is in
        // bounds. The negation is `* -1.0f` PER WORD -- the family's spelling,
        // IEEE-exact including on signed zeros. THIS IS THIS PACKAGE'S ONLY
        // NEGATION OF AN fma OUTPUT and the gate measures it as such; see the
        // module docstring's HOST-LEG OPTIMIZATION LEVEL block.
        if (m_class == 1 && i == 0) {
            cf e1 = cf_load(Ez, sx + j * sy + k);
            cf p = mul_coefficient_left(minus_dtdx, cf_sub(f_2, ss));
            cf q; q.re = inc_re; q.im = inc_im;
            cf inc = cf_sub(p, mul_complex_left(q, e1));
            curl.re = inc.re * -1.0f;
            curl.im = inc.im * -1.0f;
        }
        pml_apply(Bx, fu_Bx, idx, curl, kms_y[j], sinv_y[j], kms_z[k], sinv_z[k]);
    }

    // By: curl_y = dEx/dz - dEz/dx; dsig=z, dsigu=x; iyee=(1,0,1) -> mask on y.
    // NO cylindrical term of any kind: target 1 takes no i*m/r coupling on either
    // side (stepping.py:348-355 names Bx and Bz only) and no axis increment on the B
    // side.
    {
        cf f_1 = cf_load(Ex, idx);
        cf sf = cshift_up(Ex, idx, k, nz, sz, bc_z, ph_z, noph);
        cf f_2 = cf_load(Ez, idx);
        cf ss = cshift_up(Ez, idx, i, nx, sx, bc_x, ph_x, noph);
        cf curl = mul_coefficient_left(dtdx, cf_add(cf_sub(sf, f_1), cf_sub(f_2, ss)));
        if (bc_y == BC_METALLIC && j == 0) curl = cf_zero();
        pml_apply(By, fu_By, idx, curl, kms_z[k], sinv_z[k], kms_x[i], sinv_x[i]);
    }

    // Bz: CYLINDRICAL SUBSTITUTION (stepping.py:343-347). The phi-derivative partner
    // is the invariant-axis exact zero, so the whole curl IS the radial forward
    // difference of the WALL-EXTENDED prefix: ONE subtract, ONE complex multiply. NOT
    // the four-operand grouping and NOT `dtdx*pfx_up - dtdx*pfx_here`, both of which
    // are different float32 numbers per plane. `pfx` has nx + 1 rows at the same
    // (phi, z) stride, so row i + 1 is idx + sx and the read is in bounds at
    // i = nx - 1. iyee=(1,1,0) -> mask on z, unchanged from the Cartesian body.
    // The i*m/r partner here is Ex, which the Cartesian grouping would have loaded as
    // f_2; the substitution removes that load, so it is made explicitly.
    {
        cf pfx_here = cf_load(pfx, idx);
        cf pfx_up = cf_load(pfx, idx + sx);
        cf curl = mul_coefficient_left(dtdx, cf_sub(pfx_up, pfx_here));
        // stepping.py:351-352 -- Bz <- -1 * (i*m/r) * Ex, BEFORE the mask. Guarded
        // at m = 0 for the Bx block's reason.
        if (m_class != 0) {
            curl = cf_sub(curl, mul_complex_left(cf_load(imr2, i), cf_load(Ex, idx)));
        }
        if (bc_z == BC_METALLIC && k == 0) curl = cf_zero();
        pml_apply(Bz, fu_Bz, idx, curl, kms_x[i], sinv_x[i], kms_y[j], sinv_y[j]);
    }

    // stepping._cylindrical_axis_zero_B (:648-671), AFTER the recurrence exactly as
    // the array path runs it after the term loop (:375-376).
    //   |m| = 1: NOTHING. The branch is `m == 0` or `abs(m) > 1`; |m| = 1 falls
    //            through both, which is a fact READ OFF the array path rather than
    //            inferred from the D side being nearby.
    //   |m| >= 2: all three components AND their fu, on rows [0:zero_rows]. The array
    //            path assigns the INTEGER 0, which is the word pair (+0.0f, +0.0f).
    if (m_class == 2 && i < zero_rows) {
        cf_store(Bx, idx, cf_zero());
        cf_store(By, idx, cf_zero());
        cf_store(Bz, idx, cf_zero());
        cf_store(fu_Bx, idx, cf_zero());
        cf_store(fu_By, idx, cf_zero());
        cf_store(fu_Bz, idx, cf_zero());
    }
    //   m = 0 (:657-659): Br on the axis row is identically zero -- THE FIELD ONLY,
    //            never fu_Bx. The array path assigns the INTEGER 0, which is the
    //            word pair (+0.0f, +0.0f).
    if (m_class == 0 && i == 0) {
        cf_store(Bx, idx, cf_zero());
    }
}
'''  # stepping.py live lines for the frozen device-text citation(s) in this string: 349-350->377-378, 348->376, 370-372->398-400, 348-355->376-383, 343-347->371-375, 351-352->379-380

#: The D side. ``axis_coef`` is the m = 0 on-axis Dz multiplicand -- the host-rounded
#: ``float32(4.0 * dtdx)`` of :func:`axis_coefficient`, bound at every m and read by
#: the ``m_class == 0`` tail only; it sits in the signature WITHOUT a comment because
#: the gates' signature parsers split the parameter list on commas and stop at the
#: first closing parenthesis. Dx and Dy keep the certified Cartesian stencil -- including Dx's RAW Hy
#: operands, which is why this kernel binds both `Hy` and `pfx`; Dz's `first` operand
#: pair comes from the prefix; the i*m/r term rides on targets 0 and 2; the |m| = 1
#: increment replaces Dy's curl row 0; and the axis tail carries Dz[0] = 0 at |m| = 1
#: and the six-volume zeroing at |m| >= 2.
_CYL_STEP_D_TEMPLATE = r'''
extern "C" __global__ void cyl_step_D_pml_complex(
    float* __restrict__ Dx, float* __restrict__ Dy, float* __restrict__ Dz,
    float* __restrict__ fu_Dx, float* __restrict__ fu_Dy, float* __restrict__ fu_Dz,
    const float* __restrict__ Hx, const float* __restrict__ Hy,
    const float* __restrict__ Hz,
    const float* __restrict__ pfx,
    const float* __restrict__ imr0, const float* __restrict__ imr2,
    int nx, int ny, int nz, float dtdx,
    float minus_dtdx, float inc_re, float inc_im,
    float axis_coef,
    const float* __restrict__ kms_x, const float* __restrict__ sinv_x,
    const float* __restrict__ kms_y, const float* __restrict__ sinv_y,
    const float* __restrict__ kms_z, const float* __restrict__ sinv_z,
    int bc_x, int bc_y, int bc_z,
    int ph_x, int ph_y, int ph_z,
    int m_class, int zero_rows
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny * nz) return;

    const int sx = ny * nz;
    const int sy = nz;
    const int sz = 1;

    int k = idx % nz;
    int j = (idx / nz) % ny;
    int i = idx / (ny * nz);

    cf noph = cf_zero();

    // Dx: curl_x = dHz/dy - dHy/dz; dsig=y, dsigu=z; iyee=(1,0,0) -> mask y, z.
    // Hy IS RAW HERE. Only Dz's term takes the prefix (stepping.py:424-426); a prefix
    // leaking into Dx is a defect the gate arms as `prefix_into_dx`.
    {
        cf f_1 = cf_load(Hz, idx);
        cf sf = cshift_dn(Hz, idx, j, ny, sy, bc_y, ph_y, noph);
        cf f_2 = cf_load(Hy, idx);
        cf ss = cshift_dn(Hy, idx, k, nz, sz, bc_z, ph_z, noph);
        cf curl = mul_coefficient_left(dtdx, cf_add(cf_sub(sf, f_1), cf_sub(f_2, ss)));
        // stepping.py:431-432 -- Dx <- -1 * (i*m/r) * Hz, BEFORE the mask. Guarded
        // at m = 0 (stepping.py:430 `if cylindrical and grid.m != 0`): the array
        // path adds no term and `curl - (0 * f)` is not the identity on -0.0.
        if (m_class != 0) {
            curl = cf_sub(curl, mul_complex_left(cf_load(imr0, i), f_1));
        }
        if (bc_y == BC_METALLIC && j == 0) curl = cf_zero();
        if (bc_z == BC_METALLIC && k == 0) curl = cf_zero();
        pml_apply(Dx, fu_Dx, idx, curl, kms_y[j], sinv_y[j], kms_z[k], sinv_z[k]);
    }

    // Dy: curl_y = dHx/dz - dHz/dx; dsig=z, dsigu=x; iyee=(0,1,0) -> mask x, z.
    // NO i*m/r term (target 1 takes none). The |m|=1 axis increment REPLACES row 0
    // here, and its three operands are exactly this block's f_1, sf and f_2.
    {
        cf f_1 = cf_load(Hx, idx);
        cf sf = cshift_dn(Hx, idx, k, nz, sz, bc_z, ph_z, noph);
        cf f_2 = cf_load(Hz, idx);
        cf ss = cshift_dn(Hz, idx, i, nx, sx, bc_x, ph_x, noph);
        cf curl = mul_coefficient_left(dtdx, cf_add(cf_sub(sf, f_1), cf_sub(f_2, ss)));
        if (bc_x == BC_METALLIC && i == 0) curl = cf_zero();
        if (bc_z == BC_METALLIC && k == 0) curl = cf_zero();
        // stepping.py:451-453 with :554-556 -- dtdx * (Hr - Hr[z-1] - 2.0*Hz) at
        // r = 0, negated into curl sign convention. The array path groups it
        // ((Hr - Hr_dn) - 2.0*Hz) left to right, and `2.0 * Hz` is a weak Python float
        // meeting complex64, i.e. the zero-imaginary product with the COEFFICIENT on
        // the left. REPLACES, never accumulates. The negation is `* -1.0f` per
        // word -- see the Bx block and the module docstring's HOST-LEG block.
        if (m_class == 1 && i == 0) {
            cf two_c = mul_coefficient_left(2.0f, f_2);
            cf s = cf_sub(cf_sub(f_1, sf), two_c);
            cf inc = mul_coefficient_left(dtdx, s);
            curl.re = inc.re * -1.0f;
            curl.im = inc.im * -1.0f;
        }
        pml_apply(Dy, fu_Dy, idx, curl, kms_z[k], sinv_z[k], kms_x[i], sinv_x[i]);
    }

    // Dz: CYLINDRICAL SUBSTITUTION (stepping.py:413-426). The `first` source is the
    // prefix of Hp at ir0 = 0.5 and the UNMODIFIED backward machinery then produces
    // the cylindrical derivative; the second operand pair (Hx along phi) is untouched.
    // iyee=(0,0,1) -> mask x, y, unchanged from the Cartesian body. The prefix's own
    // r near ghost is unreachable for the same reason every other one is: Dz's r-Yee
    // shift is 0, so its curl row 0 is masked.
    {
        cf f_1 = cf_load(pfx, idx);
        cf sf = cshift_dn(pfx, idx, i, nx, sx, bc_x, ph_x, noph);
        cf f_2 = cf_load(Hx, idx);
        cf ss = cshift_dn(Hx, idx, j, ny, sy, bc_y, ph_y, noph);
        cf curl = mul_coefficient_left(dtdx, cf_add(cf_sub(sf, f_1), cf_sub(f_2, ss)));
        // stepping.py:435-436 -- Dz <- +1 * (i*m/r) * Hx, the RAW Hx and never the
        // prefixed operand, BEFORE the mask. Guarded at m = 0 for the Dx block's
        // reason.
        if (m_class != 0) {
            curl = cf_sub(curl, mul_complex_left(cf_load(imr2, i), f_2));
        }
        if (bc_x == BC_METALLIC && i == 0) curl = cf_zero();
        if (bc_y == BC_METALLIC && j == 0) curl = cf_zero();
        pml_apply(Dz, fu_Dz, idx, curl, kms_x[i], sinv_x[i], kms_y[j], sinv_y[j]);
    }

    // stepping._cylindrical_axis_zero_D (:560-598), AFTER the recurrence exactly as
    // the array path runs it after the term loop (:456-457).
    //   |m| = 1: Dz on the axis row is zeroed -- the FIELD ONLY, never fu_Dz (:588).
    //   |m| >= 2: all three components AND their fu, on rows [0:zero_rows] (:565-567).
    if (m_class == 1 && i == 0) {
        cf_store(Dz, idx, cf_zero());
    }
    if (m_class == 2 && i < zero_rows) {
        cf_store(Dx, idx, cf_zero());
        cf_store(Dy, idx, cf_zero());
        cf_store(Dz, idx, cf_zero());
        cf_store(fu_Dx, idx, cf_zero());
        cf_store(fu_Dy, idx, cf_zero());
        cf_store(fu_Dz, idx, cf_zero());
    }
    //   m = 0 (:583-587), in the array path's own order: the on-axis Dz POST-ADD
    //            `Dz[0] += (4.0*(dt/dx)) * Hy[0]` -- a weak Python float meeting a
    //            complex64 volume, i.e. the zero-imaginary product with the
    //            COEFFICIENT on the left, then the complex add with Dz on the left
    //            -- and Dp = 0 on the axis. NOT folded into the curl: the fold was
    //            MEASURED to break m = 0 under PML (Er 2.7e-01 / Hp 4.5e-01,
    //            stepping.py:545-551); Dz's dsig is R, whose sigma is zero on the
    //            axis row, so the plain post-add is the exact one. `Hy` is the raw
    //            stored Hp this kernel already binds (fields.get_H("Hy") under an
    //            active PML, update_H not yet run this sub-step), read at the SAME
    //            cell by the SAME lane that stored Dz above: no cross-lane hazard.
    if (m_class == 0 && i == 0) {
        cf_store(Dz, idx, cf_add(cf_load(Dz, idx),
                                 mul_coefficient_left(axis_coef, cf_load(Hy, idx))));
        cf_store(Dy, idx, cf_zero());
    }
}
'''  # stepping.py live lines for the frozen device-text citation(s) in this string: 424-426->453-455, 431-432->460-461, 430->459, 451-453->480-482, 554-556->583-585, 413-426->442-455, 435-436->464-465, 545-551->574-580

_TEMPLATES: Dict[str, str] = {"step_B": _CYL_STEP_B_TEMPLATE,
                              "step_D": _CYL_STEP_D_TEMPLATE}


def cylindrical_complex_source(sub_step: str, expansion) -> str:
    """The device source for one sub-step under one expansion arm.

    ``sub_step`` is ``"step_B"`` or ``"step_D"``; ``expansion`` is an arm name or its
    integer code -- REQUIRED, never defaulted, because a wrong arm is a wrong answer
    rather than a crash.

    The prelude is the CERTIFIED complex family's, imported: ``_HEAD`` (word-pair
    addressing, complex add/sub/zero), ``_ARM_SOURCE[arm]`` (the three multiply
    orientations) and ``_TAIL`` (``cshift_up``/``cshift_dn``, ``pml_apply``,
    ``constitutive_apply``). Only :data:`_CYLINDRICAL_HELPERS` and the per-side
    template are this family's own text.
    """
    if sub_step not in _TEMPLATES:
        raise ValueError(
            f"sub_step must be one of {sorted(_TEMPLATES)}, got {sub_step!r}")
    arm = normalized_expansion(expansion)
    source = (_HEAD + _ARM_SOURCE[arm] + _TAIL + _CYLINDRICAL_HELPERS
              + _TEMPLATES[sub_step])
    # PURE ASCII IS A COMPILE REQUIREMENT, not a style rule -- see the module
    # docstring. Checked here rather than in a test so that a mutation leg which
    # inserts a non-ASCII character is refused at emission with the reason, instead of
    # at NVRTC with a UnicodeEncodeError three frames away.
    source.encode("ascii")
    return source


def corpus_digest() -> str:
    """One sha256 over every source this family can emit, canonically ordered.

    Two sub-steps times two arms is four sources; a single changed character anywhere
    in this module (or in the certified strings it imports) moves this value.
    """
    import hashlib  # noqa: PLC0415 - stdlib, imported at the one call site

    digest = hashlib.sha256()
    for sub_step in sorted(_TEMPLATES):
        for name in sorted(EXPANSIONS):
            digest.update(f"{sub_step}|{name}".encode("ascii"))
            digest.update(cylindrical_complex_source(sub_step, name).encode("ascii"))
    return digest.hexdigest()


def shipped_kernel_names(source: str):
    """Every kernel NVRTC could be asked to compile in ``source``, from the text."""
    import re  # noqa: PLC0415 - stdlib, imported at the one call site

    return set(re.findall(r'extern "C" __global__ void (\w+)\(', source))


# ---------------------------------------------------------------------------
# Compilation -- the CuPy half
# ---------------------------------------------------------------------------

def _clear_kernel_cache() -> int:
    """Drop every memoized kernel; returns how many entries went."""
    from .compile_cache import clear_kernel_cache  # noqa: PLC0415

    return clear_kernel_cache()


def _get_kernel(sub_step: str, expansion):
    """Compile one sub-step under one arm, memoized on (name, options, policy, source).

    THE SOURCE IS EMITTED PER CALL AND THAT IS LOAD-BEARING, for the reason every
    sibling rebuilds its code map per call: the source is part of the memo key, so it
    has to be built before a key exists to miss on, and a gate mutates this family by
    monkeypatching :func:`cylindrical_complex_source`. A source memoized at first call
    would hand back the pre-mutation string forever -- a leg reporting a pass for a
    mutation it never applied.
    """
    import cupy as cp  # noqa: PLC0415 - see the module docstring on the lazy import

    from .compile_cache import get_or_compile, kernel_cache_key  # noqa: PLC0415

    arm = normalized_expansion(expansion)
    name = KERNELS[sub_step][0]
    code = cylindrical_complex_source(sub_step, arm)
    key = kernel_cache_key(f"cyl_complex_{sub_step}_arm{arm}", True,
                           _COMPILE_OPTIONS, code)
    return get_or_compile(
        key, lambda: cp.RawKernel(code, name, options=_COMPILE_OPTIONS))


# ---------------------------------------------------------------------------
# The bindings
# ---------------------------------------------------------------------------

def word_view(array: Any) -> Any:
    """The float32 word view of one complex64 volume -- the pointer the kernel gets.

    Refuses anything that is not a C-contiguous complex64 array. A strided view would
    be read in the wrong order by the flat word index: a scrambled volume, not a launch
    failure, which is why this raises rather than reshaping.
    """
    import cupy as cp  # noqa: PLC0415

    if array is None:
        raise ValueError("expected a complex64 array, got None")
    if array.dtype != cp.complex64:
        raise ValueError(f"expected a complex64 array, got dtype {array.dtype}")
    if not array.flags.c_contiguous:
        raise ValueError(
            "complex array is not C-contiguous; its float32 word view would not be "
            "either, and the kernel indexes it as a flat word array")
    return array.view(cp.float32)


def cylindrical_complex_curl_tables(pml: 'PML', half_integer: bool) -> Dict[str, Any]:
    """One sub-step's six kms/sinv coefficient vectors, as cached device views.

    Called ONCE per frozen configuration, not per launch -- the same discipline every
    sibling applies, and for the same reason (port-reference defect 5.14: six device
    allocations per sub-step for tables that never change).

    THE COEFFICIENTS STAY FLOAT32 UNDER COMPLEX STORAGE. ``PML._reshape_for_broadcast``
    stores float32 in BOTH storage modes, and the zero-imaginary complex product is
    what carries them into complex arithmetic; a complex table here would be a
    different sub-step.
    """
    import cupy as cp  # noqa: PLC0415

    suffix = "_h" if half_integer else ""
    tables: Dict[str, Any] = {}
    for axis in ("x", "y", "z"):
        for stem in ("kms", "sinv"):
            attribute = f"{stem}_{axis}{suffix}"
            vector = getattr(pml, attribute, None)
            if vector is None:
                raise ValueError(f"pml.{attribute} is missing")
            flat = vector.reshape(-1)
            if flat.dtype != cp.float32:
                raise ValueError(
                    f"{attribute} is {flat.dtype}; the complex cylindrical PML "
                    f"kernels index float32 coefficient vectors and carry them into "
                    f"complex arithmetic through the zero-imaginary product")
            if not flat.flags.c_contiguous:
                raise ValueError(
                    f"{attribute} did not flatten to a contiguous view; the kernel "
                    f"indexes it as a bare vector")
            tables[f"{stem}_{axis}"] = flat
    return tables


def cylindrical_complex_boundary_codes(grid) -> Tuple[int, int, int]:
    """This grid's three kernel boundary codes.

    ``coverage.real_pml_boundary_kinds`` resolves the axes the way
    ``stepping._boundary_kinds`` does; ``cylindrical_coverage.cylindrical_boundary_codes``
    maps ``'axis'`` onto ``BC_METALLIC`` on the r axis and REFUSES it anywhere else.
    Shared with the m = 0 pair on purpose: the r-axis mapping has exactly one
    defensible answer and two spellings of it is one too many.
    """
    from .coverage import real_pml_boundary_kinds  # noqa: PLC0415
    from .cylindrical_coverage import cylindrical_boundary_codes  # noqa: PLC0415

    return tuple(int(code) for code in
                 cylindrical_boundary_codes(tuple(real_pml_boundary_kinds(grid))))


def imr_rows_for(sub_step: str, xp: Any, m: int, dtdx: float, rows: int,
                 dtype: Any) -> Tuple[Any, Any]:
    """The two bound i*m/r rows for one sub-step, in kernel argument order.

    GRID INVARIANTS -- Yee shift, radial extent, m, Courant -- so a caller builds them
    ONCE per frozen configuration, exactly as ``stepping``'s own ``scratch.constant``
    cache builds them once per run (:719-723), and never per launch.
    """
    return tuple(imr_coefficient_row(xp, target, sign, m, dtdx, rows, dtype)
                 for _index, target, sign in IMR_TERMS[sub_step])


# ---------------------------------------------------------------------------
# The launch
# ---------------------------------------------------------------------------

def step_cylindrical_complex(sub_step: str, fields: 'Fields', expansion,
                             grid=None, pml: 'PML' = None, *,
                             dtdx: float = None,
                             tables: Dict[str, Any] = None,
                             boundary_codes: Sequence[Any] = None,
                             imr_rows: Sequence[Any] = None,
                             prefix=None,
                             m_class_code: int = None,
                             zero_rows_count: int = None,
                             increment_scalars: Sequence[Any] = None,
                             axis_coef: Any = None,
                             scratch: Any = None) -> None:
    """``stepping.step_B`` / ``step_D`` for a complex Dcyl grid, in one launch.

    Any m: |m| >= 1 (2026-08-27) and m = 0 (2026-09-04). ``axis_coef`` is the m = 0
    D-side on-axis multiplicand; it is bound on ``step_D`` whatever the m class (the
    kernel reads it only under ``m_class == 0``) and derived from ``dtdx`` through
    :func:`axis_coefficient` unless the gate supplies its own.

    SUPPLY ``grid`` AND ``pml`` and everything derivable is derived HERE, by the same
    functions the predicate asks -- which is what makes "the launcher and the predicate
    cannot disagree about which sub-lattice, which boundary code, which m class or
    which row vector this sub-step reads" an enforced property rather than a
    convention.

    THE OVERRIDES ARE THE GATE'S DOOR and stay, keyword-only and named for what they
    are: a gate feeds DELIBERATELY WRONG tables (the swapped sub-lattice), wrong
    boundary codes, a wrong prefix and wrong i*m/r rows, and a launcher that could not
    be handed its own could not arm any of those mutations. Passing a layer AND an
    override for the same quantity is refused: two answers to a question with one.

    NOTHING CALLS THIS YET. ``fastpath.plan_fast_path`` is untouched and these kernels
    are not certified; the callers are this directory's tests and the parity gate.
    """
    import numpy as np  # noqa: PLC0415

    if sub_step not in KERNELS:
        raise ValueError(f"sub_step must be one of {sorted(KERNELS)}, got {sub_step!r}")
    if (pml is None) == (tables is None):
        raise ValueError(
            "pass exactly one of pml (the tables are derived from the sub-step's own "
            "sub-lattice) or tables (the gate supplies its own, including mis-paired "
            "ones)")
    if tables is None:
        tables = cylindrical_complex_curl_tables(pml, HALF_INTEGER[sub_step])
    if (grid is None) == (boundary_codes is None):
        raise ValueError(
            "pass exactly one of grid (the boundary codes, the m class, the row "
            "vectors and dtdx are resolved from it) or boundary_codes (the gate "
            "supplies its own)")

    targets, sources = CURL_ARRAYS[sub_step]
    shape = tuple(int(n) for n in getattr(fields, targets[0]).shape)
    if grid is not None:
        boundary_codes = cylindrical_complex_boundary_codes(grid)
        if dtdx is None:
            dtdx = float(grid.dt / grid.dx)  # stepping.py:314, :431.
        if m_class_code is None:
            m_class_code = m_class(int(grid.m))
        if zero_rows_count is None:
            zero_rows_count = zero_rows(
                int(grid.m), bool(grid.accurate_fields_near_cylorigin))
        if imr_rows is None:
            imr_rows = imr_rows_for(sub_step, grid.xp, int(grid.m), dtdx, shape[0],
                                    getattr(fields, targets[0]).dtype)
        if increment_scalars is None:
            increment_scalars = axis_increment_scalars(int(grid.m), dtdx)
    else:
        missing = [name for name, value in
                   (("dtdx", dtdx), ("m_class_code", m_class_code),
                    ("zero_rows_count", zero_rows_count), ("imr_rows", imr_rows),
                    ("increment_scalars", increment_scalars))
                   if value is None]
    if axis_coef is None and dtdx is not None:
        # Derived from the SAME dtdx the kernel is handed, so the two cannot disagree
        # about the Courant; the gridless route needs dtdx anyway and refuses without.
        axis_coef = axis_coefficient(dtdx)
    if grid is None:
        if axis_coef is None:
            missing.append("axis_coef")
        if missing:
            # The gridless route is the gate's, and it must state EVERY derived
            # quantity rather than letting one default: a defaulted m class or a
            # defaulted increment is a silently different sub-step, which is the
            # class of defect this family exists to refuse.
            raise ValueError(
                f"without a grid the caller must supply {missing}; there is nothing "
                f"here to derive them from")

    if prefix is None:
        prefix = cylindrical_prefix(fields, sub_step, scratch=scratch)

    minus_dtdx, (inc_re, inc_im) = increment_scalars
    arguments = [word_view(getattr(fields, name)) for name in targets]
    arguments += [word_view(getattr(fields, "fu_" + name)) for name in targets]
    arguments += [word_view(getattr(fields, name)) for name in sources]
    arguments += [word_view(prefix)]
    arguments += [word_view(row) for row in imr_rows]
    arguments += [np.int32(shape[0]), np.int32(shape[1]), np.int32(shape[2]),
                  np.float32(dtdx),
                  np.float32(minus_dtdx), np.float32(inc_re), np.float32(inc_im)]
    if sub_step == "step_D":
        # THE m = 0 ON-AXIS Dz MULTIPLICAND, D side only: the B side's m = 0 rule is
        # a zero store and takes no scalar. Bound at every m -- the signature is
        # fixed -- and read by the kernel only under m_class == 0.
        arguments += [np.float32(axis_coef)]
    for axis in ("x", "y", "z"):
        arguments += [tables[f"kms_{axis}"], tables[f"sinv_{axis}"]]
    arguments += [np.int32(code) for code in boundary_codes]
    # NO BLOCH PHASE: this family refuses one, so all three flags are 0 and the
    # certified helper never reaches its rotation.
    arguments += [np.int32(0), np.int32(0), np.int32(0)]
    arguments += [np.int32(m_class_code), np.int32(zero_rows_count)]

    cells = shape[0] * shape[1] * shape[2]
    blocks = (cells + _COMPLEX_THREADS - 1) // _COMPLEX_THREADS
    kernel = _get_kernel(sub_step, expansion)
    kernel((blocks,), (_COMPLEX_THREADS,), tuple(arguments))
