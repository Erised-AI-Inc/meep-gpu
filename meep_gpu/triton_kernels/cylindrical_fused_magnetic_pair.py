"""The CYLINDRICAL COMPLEX magnetic seam in one launch: Dcyl ``step_B`` welded into
``update_H``.

The complex-storage Dcyl arm's B->H product (|m| >= 1 since 2026-08-20; m = 0 under
complex64 storage since 2026-09-04, the ``M_ZERO`` arm carried through from the
curl half), and the only large cell on the fusion matrix
with NO source-seam attrition at all. The two halves are the certified tranches'
own kernels — :func:`.cylindrical_complex.cyl_complex_pml_curl_step` on ``step_B``
and :func:`.complex_fields.bloch_constitutive_step`'s ``SCALE = 0`` arm on the H
side — and the weld is the one substitution
:func:`.kernels.fused_curl_constitutive_B` makes on the real path, done on both
float32 word planes.

EVERY ARITHMETIC LINE BELOW IS A VERBATIM COPY, with a comment naming the source it
came from. The curl half is ``cyl_complex_pml_curl_step``'s body character for
character, BACKWARD arms and all; the constitutive half is
:mod:`.complex_fused_magnetic_pair`'s, which is itself
``bloch_constitutive_step``'s. A reader can diff this body against those two and
see that nothing moved, and the gate's transcription leg is what turns that from a
claim into a measurement. If you find yourself deriving here, you have taken a
wrong turn.

===========================================================================
WHY THIS CELL — measured from the fusion matrix, and it is not a guess
===========================================================================

``parity/meep_gpu/results/fusion_matrix_triton_2026-08-20`` ranks the 24 (curl,
constitutive) cells the 186-row corpus drives that no fused product covers, by the
CEILING at that cell — instances with no in-seam source — not by rows reaching it:

    ceiling 16  (rows 16)  B->H  cylindrical complex PML -> cylindrical complex

RANK 1, and the only cell in the ranking whose ceiling equals its row count.
Re-measured on the census this file prices against
(``parity/meep_gpu/results/predicate_coverage_2026-08-16_wired_convention``, read
on ``covered_modulo_backend`` because that lift ran on NumPy):

    admit the cylindrical complex curl at step_B              16 rows
    admit the cylindrical complex constitutive at update_H    16 rows
    admit BOTH halves                                         16 rows
    ... and declare NO MAGNETIC source                        16 rows

All sixteen declare electric sources only — ``('D',)`` on fourteen, ``('D', 'D')``
on ``cylinder_cross_section.py`` and ``zone_plate.py``. The source clause that
costs ``fused_pair_B`` 22 of 46 rows and ``complex_fused_magnetic_pair`` 4 of 16
costs this family NOTHING. It is REFUSED BY NAME anyway: a corpus is not a
contract, and a Dcyl script driving a magnetic current would be admitted by both
halves and stepped wrong through this weld.

The number is an UPPER bound for the same reason every census-priced number in this
tree is: several clauses of the shipped predicate are not evaluable from a recorded
configuration block, and each can only remove rows.

===========================================================================
THE FIVE PASSES IN THE SEAM
===========================================================================

The driver runs five passes between the two halves (driver.py:3281-3288)::

    step_B -> magnetic sources -> fill_symmetry_bc_B -> zero_metal_B
           -> fill_folded_far_ghosts_B -> update_H

* **the magnetic sources — REFUSED BY NAME** (driver.py:3283-3284). Measured cost
  on the corpus: ZERO rows. Checked anyway; ignorance is never an empty set, so an
  undeclared ``sources`` is a REFUSAL and not an assumed ``()``.
* **``fill_symmetry_bc_B`` — PROVABLY A NO-OP HERE**, and on this family more
  strongly than on the Cartesian twins: ``stepping._fill_symmetry_ghost_cells``
  returns at its first line unless ``grid.has_symmetry()`` (stepping.py:1481-1482),
  and ``Grid`` REFUSES a mirror plane on a Dcyl cell outright (grid.py:648-653), so
  no admitted configuration can even carry one. THE PREDICATE RE-CHECKS IT ANYWAY
  rather than reading its own coverage off another module's guard.
* **``zero_metal_B`` — CARRIED INLINE** (driver.py:3286; ``stepping._zero_metal``
  :2206-2247), through :func:`.coverage.zero_metal_axes`, IMPORTED rather than
  re-spelled. It writes complex zero — ``array[face] = 0`` on a complex64 volume —
  so the register clear is applied to BOTH word planes.
* **``fill_folded_far_ghosts_B`` — PROVABLY A NO-OP HERE**, by the same
  ``has_symmetry`` guard one level up (stepping.py:1565-1566).

**THE WALL TABLE ON A Dcyl GRID IS ``(False, False, z_metallic)``, AND THAT IS A
POSITIVE REQUIREMENT.** ``Grid`` builds ``metallic_axes`` as ``pair[1] == METALLIC
and pair[0] != AXIS`` (grid.py:512-514) and the r axis's pair is
``(AXIS, METALLIC)``, so r is NOT a walled axis for ``_zero_metal``'s purposes even
though its far face is a metallic wall: the r = 0 row belongs to the per-|m| rules,
not to a PEC clear. phi is periodic. So ``ZM_X`` and ``ZM_Y`` are False on every
configuration this product admits, and
:func:`cylindrical_fused_magnetic_pair_coverage` REFUSES a grid whose table says
otherwise rather than compiling a clear the array path does not perform.

THAT IS A GATE-DESIGN FACT, NOT A DECORATION. ``ZM_X`` and ``ZM_Y`` are dead
constexpr branches on this family, so a mutation armed on either would rewrite a
line the scored case never reaches and report UNCAUGHT while measuring nothing —
the dead-branch class this project has already paid for. The gate arms its wall
mutation on ``ZM_Z``.

===========================================================================
THE CYLINDRICAL HALF IS THE CURL HALF, AND ONLY THE CURL HALF
===========================================================================

``stepping.update_H`` (:907-925) and ``update_E`` (:926-995) contain ZERO
occurrences of ``cylindrical``, ``is_axis``, ``m`` or ``axis_zero``, and the two
axis-zero passes (``_cylindrical_axis_zero_B`` :376, ``_D`` :457) are called from
the CURL only. The cylindrical tranche MEASURED that rather than argued it —
480/480 rows, 0 differing uint32 words against a plain complex elementwise
``dsigw`` reference on real Dcyl grids
(:func:`.cylindrical_complex.cylindrical_complex_constitutive_coverage`). So the
constitutive half of this pair is the ORDINARY complex one and this file lifts it
verbatim from :mod:`.complex_fused_magnetic_pair`, whose own weld it is.

CONSEQUENCE FOR THE SEAM: the curl half's ``v0``/``v1``/``v2`` registers already
carry the per-|m| axis rules (``M_ONE``'s ``Dz`` clear is a D-side rule and does
nothing here; ``M_MANY``'s near-axis hold zeroes all three ``v`` AND all three
``n``), so the register the constitutive half consumes is exactly the ``B`` word
the array path would have stored. The wall clear is applied AFTER those rules and
before both the store and the constitutive read — the driver's order — and both
write an exact ``+0.0``, so the two are order-independent in bits and the
transcription follows the driver anyway.

===========================================================================
WHAT IS NOT NEEDED HERE — the ownership restructure, and why
===========================================================================

The folded arm's fused pair has to move ownership: its near mirror fill reads a
cell the same launch writes, Triton has no device-wide barrier, and the naive carry
races on ``B``. NONE OF THAT ARISES ON THIS ARM. ``Grid`` refuses a fold on a Dcyl
cell, both fills are dead, every store is at ``2*idx``/``2*idx+1``, and no lane
reads a word another lane writes. The one r-direction neighbour read this kernel
makes that the Cartesian twins do not — the ``M_ONE`` B-side increment's
``Ez[r + 1]`` at ``off = nyz + j*nz + k`` — is a read of a SOURCE volume (``g2``),
which this launch never writes.

===========================================================================
THE PREFIX STAYS ON THE ARRAY PATH, AND THE PAIR INHERITS THAT
===========================================================================

The radial prefix is a SEQUENTIAL float32 scan whose summation order DEFINES the
answer, ``cupy.cumsum``'s order is the oracle (cylindrical_triton.py:32-56), and
``tl.cumsum`` matches neither. So it cannot move into this kernel any more than it
could move into the certified curl: :meth:`CylindricalFusedMagneticPairPlan.prefix`
calls :func:`.cylindrical_complex.cylindrical_complex_prefix`, which calls
``stepping.cylindrical_rderiv_prefix``. This plan is therefore NOT allocation-free,
exactly as :class:`.cylindrical_complex.CylindricalComplexCurlPlan` is not.

WHAT THE FUSION REMOVES IS ONE LAUNCH, and on a walled run one host wall pass — not
the prefix. No throughput claim is made anywhere in this file.

===========================================================================
THE EXPANSION ARM, AND THE PATTERN SET THIS PRODUCT ASKS OVER
===========================================================================

Every real-coefficient multiply on the complex path is a FULL complex product with
a zero-imaginary operand, and which of the two licensable arms the platform
implements is a MEASURED platform fact bound from a probe artifact — never guessed.
This module does not re-implement that rule and does not re-spell the multiply: it
imports :func:`.complex_fields._mul_field_left`,
:func:`.complex_fields._mul_coefficient_left` and
:func:`.cylindrical_complex._mul_general_coefficient_left` as ``triton.jit`` device
functions and resolves the constexpr through
:func:`.complex_fields._resolve_expansion`.

**THE PATTERN SET IS THE BASE FOUR, UNEXTENDED, and that is the cylindrical
tranche's own measured claim rather than a default.** Both cylindrical-only
orientations are DEGENERATE between the arms because their coefficient's real word
is a zero — the i*m/r row is exactly ``+0.0`` for every (m, sign, r) measured, the
``M_ONE`` scalar ``1j*(m*dtdx)`` is ``-0.0`` at m < 0 — so ``fma(±0.0, z_re,
-(c_im*z_im))`` and ``(±0.0*z_re) - (c_im*z_im)`` are the same single-rounding
operation (``cylindrical_complex._mul_general_coefficient_left``, measured
2026-08-13, ``AMBIGUOUS_BOTH`` in all three orientations). The H-side constitutive
adds nothing: its ``kps``/``kms`` products are ``f4_mul_c8_coefficient_left``,
already in the base four, and the ``SCALE`` arm that would bring in ``inv_eps``
belongs to the E side and is not carried. The seam adds a ``tl.where`` to ``0.0``,
which is a select and not a multiply.

So :data:`~.complex_fields.PROBE_PATTERNS` is exactly the set, and the laptop test
pins that by scanning the shipped body for any multiply helper outside the three
named in :data:`LICENSED_MULTIPLY_HELPERS`.

THE LICENCE IS POLICY-CONDITIONAL. It is cut under ``keep``, which is
:data:`~.complex_fields.CERTIFIED_UNDER_SUBNORMAL_POLICY`; a keep-cut record read
by a flush process is refused by
:func:`.complex_fields.expansion_policy_reasons` and is a broken comparison, not a
platform verdict. The gate installs ``keep`` with ``strict=True`` before its first
device compile for exactly that reason.

NOT WIRED, and not an arm. ``launch.plan_step`` assigns at most one plan per
``STEP_ORDER`` slot and this product spans FIVE driver call sites; there is no slot
it can claim without a composition rule nothing has measured. Nothing in
``launch.py`` names this module, ``fastpath.plan_fast_path`` is unchanged, and
dispatch stays disabled — the same deferral
:mod:`.complex_fused_magnetic_pair`, :mod:`.folded_fused_magnetic_pair` and
:mod:`.fused_dispersive_chain` ship under.

DEVICE STATUS: **RELEASED 2026-08-20**, the GPU host RTX A6000 GPU 4, Triton 3.1.0,
    CuPy 13.5.1. 76/76 gate rows passed
    (``parity/meep_gpu/results/triton_cylindrical_fused_magnetic_pair_2026-08-20/``).
    44 product rows — eleven cases x two launch budgets (1 and 60) x BOTH float32
    subnormal policies — were BIT-IDENTICAL, on the uint32 word view, to the CuPy
    array path AND to the two separately certified Triton products stepping the same
    seam. 22 armed source defects each diverged, 3 host-binding defects each
    diverged, 1 predicted null was confirmed, and the two expansion-arm overrides
    were confirmed inert with their measured reason. The release verdict was shown
    to FLIP: with the ``ZM_Z`` wall clear removed from the shipped module the gate
    reports FAIL and both the product legs and the needle-reachability leg go red.

    STILL NOT WIRED. A weld licenses the claim, not a dispatch. No
    ``fingerprints.json`` entry, deliberately: this module's licensed bytes are a
    function of a PROBE-BOUND arm, so a checked-in hash would record a choice rather
    than a measurement.

    THE VALUE-CLASS FINDING, and it is the one worth carrying forward. The
    ``|m| = 1`` increment's REPLACE-versus-ACCUMULATE choice differs only at a
    ``-0.0`` operand. Measured on the GPU host: NULL under random seeding, NULL under
    zero-init with a thin negative-``kms`` absorber (Bx's split-field dsig axis is
    PHI, whose ``kms`` is 1.0, so an all-``+0.0`` state cannot manufacture the
    class), and CAUGHT on a **+-0 LATTICE**, where ``fu`` itself carries ``-0.0``
    words. A gate that swept uniform seeds alone would have reported that defect
    uncaught and called this kernel certified; the ``signed_zero`` value class is
    what stops it.

Import contract: importable WITHOUT Triton — the predicates and the plan builder
(to ``None``) must answer on the laptop that is the merge bar.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .complex_fields import (
    DEFAULT_BLOCK,
    PROBE_PATTERNS,
    _mul_coefficient_left,
    _mul_field_left,
    _resolve_expansion,
    _word_view,
)
from .coverage import (
    CONSTITUTIVE_SIDES,
    Coverage,
    MAGNETIC_FIELD_TYPE,
    _call,
    zero_metal_axes,
)
from .cylindrical_complex import (
    IMR_TERMS,
    axis_increment_scalars,
    four_dtdx_scalar,
    cylindrical_complex_constitutive_coverage,
    cylindrical_complex_curl_coverage,
    cylindrical_complex_prefix,
    imr_coefficient_row,
    m_class,
    zero_rows,
)
from .launch import SUB_STEPS, CupyPointer, _flat
from .. import deposit_repair as _deposit_repair

#: Does this product bracket its fused launch with the deposit repair? While False the
#: source-presence clause below refuses every in-seam deposit, exactly as it did when the
#: clause was written out here by hand. Flipping it is a claim about the PLAN this module
#: builds -- that the leading slot saves and the trailing slot restores -- and is only
#: ever changed in the same edit as that wiring. See ``deposit_repair`` for why.
CARRIES_DEPOSIT_REPAIR = False

try:  # pragma: no cover - CUDA host only
    import triton
    import triton.language as tl
    _TRITON_IMPORT_ERROR: Optional[BaseException] = None
except ImportError as _exc:  # pragma: no cover - laptop path
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_IMPORT_ERROR = _exc


#: The curl sub-step this product starts at, and the constitutive side it ends at.
CURL_SUB_STEP = "step_B"
CONSTITUTIVE_SIDE = "H"

#: The FIVE driver call sites ONE launch of this plan performs, in driver order
#: (driver.py:3281-3288). Declared, never inferred from the slot name. Only THREE
#: of them do work on an admitted configuration — the two symmetry fills cannot even
#: exist on a Dcyl cell — and the inert two are listed anyway: what a launch
#: REPLACES is what the driver would otherwise have called.
REPLACES: Tuple[str, ...] = ("step_B", "fill_symmetry_bc_B", "zero_metal_B",
                             "fill_folded_far_ghosts_B", "update_H")

#: ``SUB_STEPS['step_B']['backward']``, restated so the kernel's one legal binding
#: is visible without reading :mod:`launch`; the test suite pins the two equal.
BACKWARD = 0

#: The probe patterns this product's operand orientations require — the BASE set,
#: unextended. See the module docstring: the two cylindrical-only orientations are
#: arm-degenerate and no multiply here is outside the four.
PRODUCT_PROBE_PATTERNS: Tuple[str, ...] = tuple(PROBE_PATTERNS)

#: The complex-multiply device functions this kernel is allowed to call, and the
#: only ones. A fourth would be a fourth operand orientation, which would need its
#: own probe pattern before it could be licensed; the test scans for exactly this.
LICENSED_MULTIPLY_HELPERS: Tuple[str, ...] = (
    "_mul_field_left", "_mul_coefficient_left", "_mul_general_coefficient_left")

#: The axes ``zero_metal_axes`` may NEVER report walled on an admitted Dcyl grid.
#: See the module docstring — a consequence of ``Grid``'s ``metallic_axes``
#: construction, checked rather than assumed, because ``ZM_X`` would compile a clear
#: on the r = 0 row that the per-|m| axis rules own.
FORBIDDEN_WALL_AXES: Tuple[int, ...] = (0, 1)

__all__ = [
    "BACKWARD", "CONSTITUTIVE_SIDE", "CURL_SUB_STEP", "FORBIDDEN_WALL_AXES",
    "LICENSED_MULTIPLY_HELPERS", "PRODUCT_PROBE_PATTERNS", "REPLACES",
    "CylindricalFusedMagneticPairPlan",
    "cyl_complex_fused_curl_constitutive_B",
    "cyl_complex_fused_curl_constitutive_B_kernel",
    "cylindrical_fused_magnetic_pair_coverage",
    "plan_cylindrical_fused_magnetic_pair",
    "plan_cylindrical_fused_magnetic_pair_from_arrays",
]


if triton is not None:

    # The device helper the cylindrical curl adds over the certified complex base,
    # imported as a ``triton.jit`` function rather than re-spelled: a second
    # spelling of a complex product is a second place for it to be subtly wrong.
    from .cylindrical_complex import _mul_general_coefficient_left  # noqa: PLC0415

    # The same constexpr codes as ``kernels.PERIODIC``/``kernels.METALLIC``,
    # restated for the reason complex_fields.py:192-194 gives — importing
    # kernels.py pulls Triton in unconditionally and defeats the host-only route.
    PERIODIC = tl.constexpr(0)
    METALLIC = tl.constexpr(1)

    #: ``M_CLASS`` arms, restated from :mod:`.cylindrical_complex` for that reason.
    #: ``M_ZERO`` is the m = 0 arm under COMPLEX storage (2026-09-04): no i*m/r
    #: block, no increment, the m = 0 axis pair. The real-storage m = 0 run is
    #: still ``cylindrical_triton``'s and the split is by ``force_complex_fields``.
    M_ZERO = tl.constexpr(0)
    M_ONE = tl.constexpr(1)
    M_MANY = tl.constexpr(2)

    @triton.jit
    def cyl_complex_fused_curl_constitutive_B(
        f0, f1, f2,                   # curl targets:     Bx,By,Bz (c8 as words)
        u0, u1, u2,                   # curl auxiliaries: fu_Bx,fu_By,fu_Bz
        g0, g1, g2,                   # curl sources:     Ex,Ey,Ez
        pfx,                          # the ARRAY-PATH prefix (c8 as words)
        c0, c2,                       # i*m/r rows for targets 0 and 2 (nr long)
        kmx, sinvx, kmy, sinvy, kmz, sinvz,   # curl kms/sinv, HALF-INTEGER lattice
        h0, h1, h2,                   # constitutive targets: Hx,Hy,Hz
        w0, w1, w2,                   # constitutive aux: f_w_Hx,f_w_Hy,f_w_Hz
        kp0, km0, kp1, km1, kp2, km2, # constitutive kps/kms, INTEGER lattice
        nx, ny, nz, n_elem, dtdx,     # n_elem = COMPLEX cells; dtdx pre-rounded
        minus_dtdx,                   # |m|=1 B: python float -dtdx
        inc_b_re, inc_b_im,           # |m|=1 B: complex64(1j*m*dtdx), SIGNED-ZERO re
        four_dtdx,                    # m=0 D scalar; UNUSED on B, carried for shape
        BACKWARD: tl.constexpr,       # bound to 0 by every builder; see below
        BCZ: tl.constexpr,            # PERIODIC or METALLIC; r METALLIC, phi PERIODIC
        M_CLASS: tl.constexpr,        # M_ZERO (m = 0), M_ONE (|m| = 1) or M_MANY (|m| >= 2)
        ZERO_ROWS: tl.constexpr,      # |m| or 1 under M_MANY; unused under M_ZERO/M_ONE
        ZM_X: tl.constexpr, ZM_Y: tl.constexpr, ZM_Z: tl.constexpr,
        EXPANSION: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Dcyl ``step_B`` + ``zero_metal_B`` + complex ``update_H``, one launch.

        ``BACKWARD`` IS CARRIED AND MUST BE 0. It keeps the curl half a verbatim
        copy of :func:`.cylindrical_complex.cyl_complex_pml_curl_step` rather than
        a hand-specialised one, which is the whole reason the transcription risk
        here is low. It cannot be 1: the D half's seam carries the ELECTRIC
        injection and an ``inv_eps`` scaling this body does not have, and 157 of
        186 corpus rows block that seam outright.

        ``SCALE`` IS NOT CARRIED. This is the H side, where
        :func:`.complex_fields.bloch_constitutive_step`'s ``SCALE = 0`` arm reads
        ``B`` directly; the ``SCALE = 1`` arm's ``inv_eps`` multiply belongs to the
        E side and would add a fourth operand orientation to a kernel whose
        probe-pattern claim is that it adds none.

        ``ZM_X``/``ZM_Y``/``ZM_Z`` are ``coverage.zero_metal_axes``, the question
        ``stepping._zero_metal`` asks. On a Dcyl grid the first two are ALWAYS
        False (module docstring) and the predicate refuses a grid that says
        otherwise; they are carried so the transcription of ``_zero_metal``'s
        diagonal table is complete and legible rather than half-written.
        """
        # ===================================================================
        # THE CURL HALF — verbatim from cylindrical_complex.cyl_complex_pml_curl_
        # step. The ONLY edit below its stores is the wall clear applied to the
        # v registers; nothing else moved.
        # ===================================================================
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = idx < n_elem
        nyz = ny * nz
        k = idx % nz
        plane = idx // nz
        j = plane % ny
        i = plane // ny

        # --- the ghost rule, per axis (stepping._shift_up / _shift_down) -------
        # r is CYL_AXIS: the far face is the metallic zero (S:1781-1783) and the
        # near ghost (S:1830-1842) is DELIBERATELY NOT IMPLEMENTED — unobservable,
        # see cylindrical_complex's module docstring. phi is PERIODIC on a length-1
        # axis: the wrap returns the SAME element, which is what makes the
        # difference an exact +0.0. Computed, never elided (grouping choice 8).
        if BACKWARD:
            si, sj, sk = i - 1, j - 1, k - 1
        else:
            si, sj, sk = i + 1, j + 1, k + 1
        vr = live & (si >= 0) & (si < nx)
        sj = tl.where(sj < 0, ny - 1, tl.where(sj == ny, 0, sj))
        vp = live
        if BCZ == METALLIC:
            vz = live & (sk >= 0) & (sk < nz)
        else:
            sk = tl.where(sk < 0, nz - 1, tl.where(sk == nz, 0, sk))
            vz = live

        o_r = si * nyz + j * nz + k
        o_p = i * nyz + sj * nz + k
        o_z = i * nyz + j * nz + sk

        # --- loads: two words per operand --------------------------------------
        a_re = tl.load(g0 + 2 * idx, mask=live, other=0.0)
        a_im = tl.load(g0 + 2 * idx + 1, mask=live, other=0.0)
        b_re = tl.load(g1 + 2 * idx, mask=live, other=0.0)
        b_im = tl.load(g1 + 2 * idx + 1, mask=live, other=0.0)
        c_re = tl.load(g2 + 2 * idx, mask=live, other=0.0)
        c_im = tl.load(g2 + 2 * idx + 1, mask=live, other=0.0)
        a_p_re = tl.load(g0 + 2 * o_p, mask=vp, other=0.0)
        a_p_im = tl.load(g0 + 2 * o_p + 1, mask=vp, other=0.0)
        a_z_re = tl.load(g0 + 2 * o_z, mask=vz, other=0.0)
        a_z_im = tl.load(g0 + 2 * o_z + 1, mask=vz, other=0.0)
        b_r_re = tl.load(g1 + 2 * o_r, mask=vr, other=0.0)
        b_r_im = tl.load(g1 + 2 * o_r + 1, mask=vr, other=0.0)
        b_z_re = tl.load(g1 + 2 * o_z, mask=vz, other=0.0)
        b_z_im = tl.load(g1 + 2 * o_z + 1, mask=vz, other=0.0)
        c_r_re = tl.load(g2 + 2 * o_r, mask=vr, other=0.0)
        c_r_im = tl.load(g2 + 2 * o_r + 1, mask=vr, other=0.0)
        c_p_re = tl.load(g2 + 2 * o_p, mask=vp, other=0.0)
        c_p_im = tl.load(g2 + 2 * o_p + 1, mask=vp, other=0.0)

        # --- the curl (stepping._curl_from_operands): DO NOT flatten these parens
        t0_re = ((c_p_re - c_re) + (b_re - b_z_re))
        t0_im = ((c_p_im - c_im) + (b_im - b_z_im))
        t1_re = ((a_z_re - a_re) + (c_re - c_r_re))
        t1_im = ((a_z_im - a_im) + (c_im - c_r_im))
        t2_re = ((b_r_re - b_re) + (a_re - a_p_re))
        t2_im = ((b_r_im - b_im) + (a_im - a_p_im))
        curl0_re, curl0_im = _mul_coefficient_left(dtdx, t0_re, t0_im, EXPANSION)
        curl1_re, curl1_im = _mul_coefficient_left(dtdx, t1_re, t1_im, EXPANSION)
        curl2_re, curl2_im = _mul_coefficient_left(dtdx, t2_re, t2_im, EXPANSION)

        # --- the cylindrical substitution on target 2 --------------------------
        if BACKWARD:
            # step_D :425-426 — Dz's `first` source is the prefix.
            p_here_re = tl.load(pfx + 2 * idx, mask=live, other=0.0)
            p_here_im = tl.load(pfx + 2 * idx + 1, mask=live, other=0.0)
            p_down_re = tl.load(pfx + 2 * o_r, mask=vr, other=0.0)
            p_down_im = tl.load(pfx + 2 * o_r + 1, mask=vr, other=0.0)
            t2_re = ((p_down_re - p_here_re) + (a_re - a_p_re))
            t2_im = ((p_down_im - p_here_im) + (a_im - a_p_im))
            curl2_re, curl2_im = _mul_coefficient_left(dtdx, t2_re, t2_im, EXPANSION)
        else:
            # step_B :343-347 — Bz's WHOLE curl is the forward difference of the
            # EXTENDED prefix: one subtract, one multiply. The four-operand
            # grouping above is a different float32 number per plane.
            pu_re = tl.load(pfx + 2 * (idx + nyz), mask=live, other=0.0)
            pu_im = tl.load(pfx + 2 * (idx + nyz) + 1, mask=live, other=0.0)
            pd_re = tl.load(pfx + 2 * idx, mask=live, other=0.0)
            pd_im = tl.load(pfx + 2 * idx + 1, mask=live, other=0.0)
            curl2_re, curl2_im = _mul_coefficient_left(dtdx, pu_re - pd_re,
                                                       pu_im - pd_im, EXPANSION)

        # --- the i*m/r coupling (stepping :674-724, sites :348-355 / :430-437) --
        # AFTER the dtdx curl, BEFORE the ownership mask — the array path's order.
        # The partners are the CENTER registers already loaded: target 0 takes `c`
        # (g2 = Ez/Hz), target 2 takes `a` (g0 = Ex/Hx). The coefficient row is
        # host-built and bound; its real word is +0.0 and MUST survive as a literal
        # cross-term operand. `curl - (c*g)` carries the array path's
        # `curl + (-(c*g))` by IEEE-754.
        # COMPILED OUT at m = 0, as the array path never forms the term there
        # (`if cylindrical and grid.m != 0`); a zero row would move a -0.0 curl
        # word and propagate a non-finite partner. Same guard as the certified
        # curl's.
        if M_CLASS != M_ZERO:
            q0_re = tl.load(c0 + 2 * i, mask=live, other=0.0)
            q0_im = tl.load(c0 + 2 * i + 1, mask=live, other=0.0)
            q2_re = tl.load(c2 + 2 * i, mask=live, other=0.0)
            q2_im = tl.load(c2 + 2 * i + 1, mask=live, other=0.0)
            m0_re, m0_im = _mul_general_coefficient_left(q0_re, q0_im, c_re, c_im,
                                                         EXPANSION)
            curl0_re = curl0_re - m0_re
            curl0_im = curl0_im - m0_im
            m2_re, m2_im = _mul_general_coefficient_left(q2_re, q2_im, a_re, a_im,
                                                         EXPANSION)
            curl2_re = curl2_re - m2_re
            curl2_im = curl2_im - m2_im

        # --- ownership mask (stepping._mask_non_owned_cells, is_axis + metallic) -
        #   B: Bx(0,1,1) -> r ; By(1,0,1) -> phi only, periodic: nothing ;
        #      Bz(1,1,0) -> z
        #   D: Dx(1,0,0) -> phi (nothing) + z ; Dy(0,1,0) -> r + z ; Dz(0,0,1) -> r
        # The z clauses fire only when z is METALLIC; a periodic z masks nothing.
        at_r, at_z = i == 0, k == 0
        if BACKWARD:
            if BCZ == METALLIC:
                curl0_re = tl.where(at_z, 0.0, curl0_re)
                curl0_im = tl.where(at_z, 0.0, curl0_im)
                curl1_re = tl.where(at_z, 0.0, curl1_re)
                curl1_im = tl.where(at_z, 0.0, curl1_im)
            curl1_re = tl.where(at_r, 0.0, curl1_re)
            curl1_im = tl.where(at_r, 0.0, curl1_im)
            curl2_re = tl.where(at_r, 0.0, curl2_re)
            curl2_im = tl.where(at_r, 0.0, curl2_im)
        else:
            curl0_re = tl.where(at_r, 0.0, curl0_re)
            curl0_im = tl.where(at_r, 0.0, curl0_im)
            if BCZ == METALLIC:
                curl2_re = tl.where(at_z, 0.0, curl2_re)
                curl2_im = tl.where(at_z, 0.0, curl2_im)

        # --- the |m| = 1 axis-row increment, AFTER the mask, REPLACING the row --
        # stepping :370-372 (B) / :451-453 (D). The row is exactly +0.0 here, and
        # `+0.0 + x == x` for every x EXCEPT -0.0 — so REPLACE, never accumulate
        # (grouping choice 4). The negation is `* -1.0`, never unary minus.
        if M_CLASS == M_ONE:
            if BACKWARD:
                two_c_re, two_c_im = _mul_coefficient_left(2.0, c_re, c_im,
                                                           EXPANSION)
                s_re = (a_re - a_z_re) - two_c_re
                s_im = (a_im - a_z_im) - two_c_im
                inc_re, inc_im = _mul_coefficient_left(dtdx, s_re, s_im, EXPANSION)
                curl1_re = tl.where(at_r, inc_re * -1.0, curl1_re)
                curl1_im = tl.where(at_r, inc_im * -1.0, curl1_im)
            else:
                # (-dtdx)*(Ep[i] - Ep[i, z+1]) - (1j*m*dtdx)*Ez[r+1] at r = 0.
                # Ez[r+1] is the FIRST OFF-AXIS row of g2, read at the row-1 offset
                # of this lane's column. The predicate refuses nr < 2, which is
                # what keeps that load in bounds.
                off = nyz + j * nz + k
                e1_re = tl.load(g2 + 2 * off, mask=live, other=0.0)
                e1_im = tl.load(g2 + 2 * off + 1, mask=live, other=0.0)
                d_re = b_re - b_z_re
                d_im = b_im - b_z_im
                p_re, p_im = _mul_coefficient_left(minus_dtdx, d_re, d_im, EXPANSION)
                q_re, q_im = _mul_general_coefficient_left(inc_b_re, inc_b_im,
                                                           e1_re, e1_im, EXPANSION)
                inc_re = p_re - q_re
                inc_im = p_im - q_im
                curl0_re = tl.where(at_r, inc_re * -1.0, curl0_re)
                curl0_im = tl.where(at_r, inc_im * -1.0, curl0_im)

        # --- split-field recurrence (stepping._apply_pml_update) ---------------
        # dsig/dsigu cycle (vec.hpp cycle_direction): target 0 -> (y, z),
        # 1 -> (z, x), 2 -> (x, y), the same triple on both sides.
        km_x = tl.load(kmx + i, mask=live, other=0.0)
        si_x = tl.load(sinvx + i, mask=live, other=0.0)
        km_y = tl.load(kmy + j, mask=live, other=0.0)
        si_y = tl.load(sinvy + j, mask=live, other=0.0)
        km_z = tl.load(kmz + k, mask=live, other=0.0)
        si_z = tl.load(sinvz + k, mask=live, other=0.0)

        p0_re = tl.load(u0 + 2 * idx, mask=live, other=0.0)
        p0_im = tl.load(u0 + 2 * idx + 1, mask=live, other=0.0)
        x_re, x_im = _mul_field_left(p0_re, p0_im, km_y, EXPANSION)
        x_re = x_re - curl0_re
        x_im = x_im - curl0_im
        n0_re, n0_im = _mul_field_left(x_re, x_im, si_y, EXPANSION)
        e_re = tl.load(f0 + 2 * idx, mask=live, other=0.0)
        e_im = tl.load(f0 + 2 * idx + 1, mask=live, other=0.0)
        r_re, r_im = _mul_field_left(e_re, e_im, km_z, EXPANSION)
        r_re = (r_re + n0_re) - p0_re
        r_im = (r_im + n0_im) - p0_im
        v0_re, v0_im = _mul_field_left(r_re, r_im, si_z, EXPANSION)

        p1_re = tl.load(u1 + 2 * idx, mask=live, other=0.0)
        p1_im = tl.load(u1 + 2 * idx + 1, mask=live, other=0.0)
        x_re, x_im = _mul_field_left(p1_re, p1_im, km_z, EXPANSION)
        x_re = x_re - curl1_re
        x_im = x_im - curl1_im
        n1_re, n1_im = _mul_field_left(x_re, x_im, si_z, EXPANSION)
        e_re = tl.load(f1 + 2 * idx, mask=live, other=0.0)
        e_im = tl.load(f1 + 2 * idx + 1, mask=live, other=0.0)
        r_re, r_im = _mul_field_left(e_re, e_im, km_x, EXPANSION)
        r_re = (r_re + n1_re) - p1_re
        r_im = (r_im + n1_im) - p1_im
        v1_re, v1_im = _mul_field_left(r_re, r_im, si_x, EXPANSION)

        p2_re = tl.load(u2 + 2 * idx, mask=live, other=0.0)
        p2_im = tl.load(u2 + 2 * idx + 1, mask=live, other=0.0)
        x_re, x_im = _mul_field_left(p2_re, p2_im, km_x, EXPANSION)
        x_re = x_re - curl2_re
        x_im = x_im - curl2_im
        n2_re, n2_im = _mul_field_left(x_re, x_im, si_x, EXPANSION)
        e_re = tl.load(f2 + 2 * idx, mask=live, other=0.0)
        e_im = tl.load(f2 + 2 * idx + 1, mask=live, other=0.0)
        r_re, r_im = _mul_field_left(e_re, e_im, km_y, EXPANSION)
        r_re = (r_re + n2_re) - p2_re
        r_im = (r_im + n2_im) - p2_im
        v2_re, v2_im = _mul_field_left(r_re, r_im, si_y, EXPANSION)

        # --- the per-|m| axis rules, folded into the STORED value ---------------
        # m = 0 (stepping :581-585 D / :661-662 B), complex storage, 2026-09-04:
        #   the B side zeroes Bx on the axis row — the value BELOW this block is
        #   what update_H reads; the D arm is dead here (BACKWARD is 0) and is
        #   carried verbatim from the certified curl.
        # |m| = 1 (stepping :588-589): the D side zeroes Dz on the axis row — the
        #   FIELD ONLY, never fu_Dz — and the B side does NOTHING (:661-663).
        # |m| >= 2 (:565-567, :663-671): all three components AND their fu on every
        #   row within ZERO_ROWS of the axis. The component set is POST-#3164.
        if M_CLASS == M_ZERO:
            if BACKWARD:
                hp_re, hp_im = _mul_coefficient_left(four_dtdx, b_re, b_im, EXPANSION)
                v2_re = tl.where(at_r, v2_re + hp_re, v2_re)
                v2_im = tl.where(at_r, v2_im + hp_im, v2_im)
                v1_re = tl.where(at_r, 0.0, v1_re)
                v1_im = tl.where(at_r, 0.0, v1_im)
            else:
                v0_re = tl.where(at_r, 0.0, v0_re)
                v0_im = tl.where(at_r, 0.0, v0_im)
        if M_CLASS == M_ONE:
            if BACKWARD:
                v2_re = tl.where(at_r, 0.0, v2_re)
                v2_im = tl.where(at_r, 0.0, v2_im)
        if M_CLASS == M_MANY:
            near = i < ZERO_ROWS
            v0_re = tl.where(near, 0.0, v0_re)
            v0_im = tl.where(near, 0.0, v0_im)
            v1_re = tl.where(near, 0.0, v1_re)
            v1_im = tl.where(near, 0.0, v1_im)
            v2_re = tl.where(near, 0.0, v2_re)
            v2_im = tl.where(near, 0.0, v2_im)
            n0_re = tl.where(near, 0.0, n0_re)
            n0_im = tl.where(near, 0.0, n0_im)
            n1_re = tl.where(near, 0.0, n1_re)
            n1_im = tl.where(near, 0.0, n1_im)
            n2_re = tl.where(near, 0.0, n2_re)
            n2_im = tl.where(near, 0.0, n2_im)

        # --- the seam: stepping.zero_metal_B (driver.py:3286) ------------------
        # Applied to the REGISTERS, before both the store and the constitutive
        # read, so the two consumers see the one value the array path leaves in B.
        # BOTH planes: `array[_face(axis, 0)] = 0` on a complex64 volume writes
        # complex zero (stepping._zero_metal :2246). ON A Dcyl GRID ONLY ZM_Z CAN
        # BE TRUE (module docstring) and the predicate refuses a grid that reports
        # otherwise; the r and phi rows are written so the diagonal table is
        # legible, not because either can fire.
        if ZM_X:
            v0_re = tl.where(at_r, 0.0, v0_re)
            v0_im = tl.where(at_r, 0.0, v0_im)
        if ZM_Y:
            at_p = j == 0
            v1_re = tl.where(at_p, 0.0, v1_re)
            v1_im = tl.where(at_p, 0.0, v1_im)
        if ZM_Z:
            v2_re = tl.where(at_z, 0.0, v2_re)
            v2_im = tl.where(at_z, 0.0, v2_im)

        # --- stores: u then f (kernels.py:193-198 order), both planes ----------
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

        # ==================== the constitutive half ===========================
        # VERBATIM from complex_fused_magnetic_pair's constitutive half, which is
        # itself complex_fields.bloch_constitutive_step's SCALE=0 arm with
        # `src = tl.load(g + 2*idx)` replaced by the register pair the curl half
        # just produced. `prev` is read BEFORE the `w` store (S:2083-2085); the two
        # accumulations stay separate and left-to-right, each a coefficient-LEFT
        # zero-imaginary complex product (S:2086-2087, S:2093-2095). update_H
        # carries NO cylindrical branch, which the cylindrical tranche measured
        # (480/480 rows, 0 differing words) rather than assumed.
        kp_0 = tl.load(kp0 + i, mask=live, other=0.0)
        km_0 = tl.load(km0 + i, mask=live, other=0.0)
        kp_1 = tl.load(kp1 + j, mask=live, other=0.0)
        km_1 = tl.load(km1 + j, mask=live, other=0.0)
        kp_2 = tl.load(kp2 + k, mask=live, other=0.0)
        km_2 = tl.load(km2 + k, mask=live, other=0.0)

        # --- component 0 -------------------------------------------------------
        prev_re = tl.load(w0 + 2 * idx, mask=live, other=0.0)   # BEFORE the store.
        prev_im = tl.load(w0 + 2 * idx + 1, mask=live, other=0.0)
        src_re = v0_re
        src_im = v0_im
        tl.store(w0 + 2 * idx, src_re, mask=live)
        tl.store(w0 + 2 * idx + 1, src_im, mask=live)
        a_re = tl.load(h0 + 2 * idx, mask=live, other=0.0)
        a_im = tl.load(h0 + 2 * idx + 1, mask=live, other=0.0)
        t_re, t_im = _mul_coefficient_left(kp_0, src_re, src_im, EXPANSION)
        a_re = a_re + t_re
        a_im = a_im + t_im
        t_re, t_im = _mul_coefficient_left(km_0, prev_re, prev_im, EXPANSION)
        a_re = a_re - t_re
        a_im = a_im - t_im
        tl.store(h0 + 2 * idx, a_re, mask=live)
        tl.store(h0 + 2 * idx + 1, a_im, mask=live)

        # --- component 1 -------------------------------------------------------
        prev_re = tl.load(w1 + 2 * idx, mask=live, other=0.0)
        prev_im = tl.load(w1 + 2 * idx + 1, mask=live, other=0.0)
        src_re = v1_re
        src_im = v1_im
        tl.store(w1 + 2 * idx, src_re, mask=live)
        tl.store(w1 + 2 * idx + 1, src_im, mask=live)
        a_re = tl.load(h1 + 2 * idx, mask=live, other=0.0)
        a_im = tl.load(h1 + 2 * idx + 1, mask=live, other=0.0)
        t_re, t_im = _mul_coefficient_left(kp_1, src_re, src_im, EXPANSION)
        a_re = a_re + t_re
        a_im = a_im + t_im
        t_re, t_im = _mul_coefficient_left(km_1, prev_re, prev_im, EXPANSION)
        a_re = a_re - t_re
        a_im = a_im - t_im
        tl.store(h1 + 2 * idx, a_re, mask=live)
        tl.store(h1 + 2 * idx + 1, a_im, mask=live)

        # --- component 2 -------------------------------------------------------
        prev_re = tl.load(w2 + 2 * idx, mask=live, other=0.0)
        prev_im = tl.load(w2 + 2 * idx + 1, mask=live, other=0.0)
        src_re = v2_re
        src_im = v2_im
        tl.store(w2 + 2 * idx, src_re, mask=live)
        tl.store(w2 + 2 * idx + 1, src_im, mask=live)
        a_re = tl.load(h2 + 2 * idx, mask=live, other=0.0)
        a_im = tl.load(h2 + 2 * idx + 1, mask=live, other=0.0)
        t_re, t_im = _mul_coefficient_left(kp_2, src_re, src_im, EXPANSION)
        a_re = a_re + t_re
        a_im = a_im + t_im
        t_re, t_im = _mul_coefficient_left(km_2, prev_re, prev_im, EXPANSION)
        a_re = a_re - t_re
        a_im = a_im - t_im
        tl.store(h2 + 2 * idx, a_re, mask=live)
        tl.store(h2 + 2 * idx + 1, a_im, mask=live)

else:  # pragma: no cover - laptop path
    cyl_complex_fused_curl_constitutive_B = None  # type: ignore[assignment]


def cyl_complex_fused_curl_constitutive_B_kernel() -> Any:
    """The JIT kernel, or a named ImportError on a host without Triton."""
    if cyl_complex_fused_curl_constitutive_B is None:
        raise ImportError(
            "the cylindrical complex fused magnetic B/H kernel needs the optional "
            f"`triton` package (pip install triton). Original error: "
            f"{_TRITON_IMPORT_ERROR}")
    return cyl_complex_fused_curl_constitutive_B


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def cylindrical_fused_magnetic_pair_coverage(fields: Any, pml: Any,
                                             sources: Any = None,
                                             probe: Any = None) -> Coverage:
    """May ONE launch span Dcyl ``step_B`` -> wall -> complex ``update_H``?

    A conjunction of the two halves' own predicates plus the seam clauses. Nothing
    is weakened: a configuration either half refuses is refused here with that
    half's reasons, prefixed so a reader can tell which side said it — the same
    construction :func:`.complex_fused_magnetic_pair.complex_fused_magnetic_pair_coverage`
    and :func:`.coverage.fused_pair_coverage` use. The Dcyl geometry clauses, the
    storage clause (complex64 at every m; the real m = 0 run is cylindrical_triton's),
    the nr >= 2 clause and the EXPANSION licence are all
    INHERITED from the halves and not restated.
    """
    reasons: List[str] = []

    curl = cylindrical_complex_curl_coverage(fields, pml, CURL_SUB_STEP, probe=probe)
    if not curl.covered:
        reasons.extend(f"cylindrical curl half: {reason}" for reason in curl.reasons)
    constitutive = cylindrical_complex_constitutive_coverage(
        fields, pml, CONSTITUTIVE_SIDE, probe=probe)
    if not constitutive.covered:
        reasons.extend(f"constitutive half: {reason}"
                       for reason in constitutive.reasons)

    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False,
                        tuple(dict.fromkeys(reasons + ["fields carries no grid"])))

    # THE SOURCE SEAM. A magnetic source is injected BETWEEN the two halves
    # (driver.py:3283-3284), so a fused pair would consume a pre-injection B. An
    # ELECTRIC source is injected in the D/E seam and does not disqualify this
    # pair. MEASURED COST ON THE CORPUS: ZERO — all sixteen cylindrical rows
    # declare electric sources only, which is what makes this the only large cell
    # with no source attrition. IGNORANCE IS NOT AN EMPTY SET: `Fields` does not
    # hold the source list, so a predicate that inferred "no magnetic source" from
    # not being told would be the over-covering refusal this clause prevents.
    reasons.extend(_deposit_repair.seam_source_reasons(
        fields, sources, 'B',
        undeclared=(
            "the source set was not declared: this predicate cannot infer an empty "
            "magnetic source seam from Fields"
        ),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) is magnetic: the "
            f"driver injects it BETWEEN step_B and update_H "
            f"(driver.py:3283), which is work inside the seam this kernel "
            f"closes"),
        carries_repair=CARRIES_DEPOSIT_REPAIR))

    # THE TWO SYMMETRY PASSES. `fill_symmetry_bc_B` (driver.py:3285) and
    # `fill_folded_far_ghosts_B` (:3287) both return at their first line unless
    # `grid.has_symmetry()` (stepping.py:1481-1482, :1565-1566); `Grid` refuses a
    # mirror on a Dcyl cell outright (grid.py:648-653) and both halves restate
    # that. RE-CHECKED HERE ANYWAY, not inferred: this module's coverage may not be
    # read off another module's guard, and the cost is one attribute read.
    if _call(grid, "has_symmetry", default=False):
        reasons.append(
            "a mirror plane is active: stepping.fill_symmetry_bc_B (driver.py:3285) "
            "and fill_folded_far_ghosts_B (:3287) run inside this seam and this "
            "weld carries neither")
    for axis in range(3):
        if _call(grid, "is_mirrored", axis, default=False):
            reasons.append(
                f"axis {axis} is folded by a mirror plane: the two symmetry passes "
                f"in this seam are no longer no-ops and this weld carries neither")

    # THE WALL CLEAR is carried inline, so the grid must be able to answer which
    # axes are walled; an unanswerable one compiles to ZM=False and silently skips
    # a plane the array path clears.
    answerable = True
    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            answerable = False
            reasons.append(
                f"grid does not expose {name}; zero_metal_B cannot be carried inline")

    # THE WALL TABLE MUST BE THE Dcyl ONE. `Grid` cannot report r or phi walled on
    # a cylindrical cell (module docstring), and ZM_X would compile a clear on the
    # r = 0 row that the per-|m| axis rules own — a zero the array path never
    # writes there. Refused by name rather than compiled.
    if answerable:
        walls = zero_metal_axes(grid)
        for axis in FORBIDDEN_WALL_AXES:
            if walls[axis]:
                reasons.append(
                    f"zero_metal_axes reports axis {axis} walled on a grid this "
                    f"family admits; on a Dcyl cell neither r nor phi can be "
                    f"(grid.py:512-514), and the r = 0 clear ZM_X would compile is "
                    f"a row the per-|m| axis rules own")

    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class CylindricalFusedMagneticPairPlan:
    """One launch for five of the driver's magnetic call sites, plus the array-path
    prefix the cylindrical family cannot avoid.

    The complex volumes are bound as float32 WORD VIEWS at plan time, once, and
    ``n_elem`` stays the COMPLEX cell count — the same split
    :class:`.complex_fields.ComplexPmlCurlPlan` makes, for the same reason.

    IT IS NOT ALLOCATION-FREE, and cannot be: the prefix is recomputed every launch
    from the current sources, exactly as
    :class:`.cylindrical_complex.CylindricalComplexCurlPlan` recomputes it. With
    the engine's ``StepScratch`` threaded through it allocates nothing beyond what
    the array path already pools.

    The two i*m/r coefficient rows and the axis-increment scalars are GRID
    INVARIANTS — Yee shift, radial extent, m, Courant — so they are built ONCE at
    plan time, exactly as ``stepping``'s own ``scratch.constant`` cache builds them
    once per run (:713-723), and never rebuilt per launch.
    """

    __slots__ = (
        "shape", "n_elem", "dtdx", "backward", "bcz", "m_class", "zero_rows",
        "zero_metal", "expansion", "block", "num_warps", "increment_scalars",
        "four_dtdx", "xp", "scratch", "_targets", "_aux", "_sources",
        "_source_map",
        "_curl_coefficients", "_imr_rows", "_h_targets", "_h_aux",
        "_h_coefficients", "_grid", "_kernel",
    )

    #: The five driver call sites one launch performs. Declared, so a composition
    #: can be inspected rather than inferred from the curl slot's name.
    replaces = REPLACES

    def __init__(self, shape, dtdx: float, m: int,
                 accurate_fields_near_cylorigin: bool, bcz: int, zero_metal,
                 expansion: int, block: int,
                 targets, auxiliaries, sources, curl_coefficients,
                 h_targets, h_aux, h_coefficients, xp: Any, scratch: Any = None,
                 kernel: Any = None, num_warps: Optional[int] = 1) -> None:
        spec = SUB_STEPS[CURL_SUB_STEP]
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        # float(): the array path consumes the Python scalar at complex64 precision
        # and Triton types a Python float argument as fp32, so the two are the same
        # bits (ComplexPmlCurlPlan's measured note).
        self.dtdx = float(dtdx)
        self.backward = BACKWARD
        self.bcz = int(bcz)
        self.m_class = m_class(m)
        self.zero_rows = zero_rows(m, accurate_fields_near_cylorigin)
        self.zero_metal = tuple(bool(value) for value in zero_metal)
        self.expansion = int(expansion)
        self.block = int(block)
        self.num_warps = None if num_warps is None else int(num_warps)
        self.increment_scalars = axis_increment_scalars(m, dtdx)
        self.four_dtdx = four_dtdx_scalar(dtdx)
        self.xp = xp
        self.scratch = scratch
        self._targets = tuple(CupyPointer(_word_view(a)) for a in targets)
        self._aux = tuple(CupyPointer(_word_view(a)) for a in auxiliaries)
        self._sources = tuple(CupyPointer(_word_view(a)) for a in sources)
        # The raw arrays, by component name: the prefix is computed from them per
        # launch, so the plan holds the arrays and not only their addresses.
        self._source_map = {name: array
                            for name, array in zip(spec["sources"], sources)}
        self._curl_coefficients = tuple(
            CupyPointer(_flat(a)) for a in curl_coefficients)
        dtype = targets[0].dtype
        self._imr_rows = tuple(
            CupyPointer(_word_view(xp.ascontiguousarray(
                imr_coefficient_row(xp, spec["targets"][index], sign, m, dtdx,
                                    self.shape[0], dtype).reshape(-1))))
            for index, _register, sign in IMR_TERMS[CURL_SUB_STEP])
        self._h_targets = tuple(CupyPointer(_word_view(a)) for a in h_targets)
        self._h_aux = tuple(CupyPointer(_word_view(a)) for a in h_aux)
        self._h_coefficients = tuple(
            CupyPointer(_flat(a)) for a in h_coefficients)
        # THE INT32 WORD BOUND. The kernel addresses words as ``2 * idx`` in int32,
        # so the complex cell count must leave room for the doubling — the same
        # halving the complex predicates apply. Refused here as well because the
        # from-arrays route runs no predicate at all.
        if 2 * self.n_elem >= 2 ** 31:
            raise ValueError(
                f"{self.n_elem} complex cells needs {2 * self.n_elem} int32 word "
                f"indices, which overflows the kernel's 2 * idx addressing")
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        # An override for exactly one caller: the gate's mutation legs, which
        # compile deliberately broken copies of this kernel. Dropping it is not a
        # slowdown, it is a DISARMING — every mutation leg would then launch the
        # shipped kernel and report the defect as uncaught.
        self._kernel = kernel

    def prefix(self) -> Any:
        """This launch's radial prefix, from the shipped array-path scan."""
        return cylindrical_complex_prefix(self.xp, CURL_SUB_STEP, self._source_map,
                                          scratch=self.scratch)

    def run(self, guard: Optional[bool] = None) -> None:
        """Launch the fused pair, in place. Same ``guard`` contract as every plan."""
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        kernel = (self._kernel if self._kernel is not None
                  else cyl_complex_fused_curl_constitutive_B_kernel())
        (minus_dtdx, inc_b) = self.increment_scalars
        nx, ny, nz = self.shape
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        kernel[self._grid](
            *self._targets, *self._aux, *self._sources,
            CupyPointer(_word_view(self.prefix())), *self._imr_rows,
            *self._curl_coefficients,
            *self._h_targets, *self._h_aux, *self._h_coefficients,
            nx, ny, nz, self.n_elem, self.dtdx,
            minus_dtdx, inc_b[0], inc_b[1], self.four_dtdx,
            BACKWARD=self.backward,
            BCZ=self.bcz,
            M_CLASS=self.m_class,
            ZERO_ROWS=self.zero_rows,
            ZM_X=self.zero_metal[0], ZM_Y=self.zero_metal[1],
            ZM_Z=self.zero_metal[2],
            EXPANSION=self.expansion,
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )

    def __repr__(self) -> str:
        return (f"CylindricalFusedMagneticPairPlan(shape={self.shape}, "
                f"bcz={self.bcz}, m_class={self.m_class}, "
                f"zero_rows={self.zero_rows}, zero_metal={self.zero_metal}, "
                f"expansion={self.expansion}, block={self.block}, "
                f"num_warps={self.num_warps})")


def plan_cylindrical_fused_magnetic_pair(
        fields: Any, pml: Any, sources: Any = None, block: Optional[int] = None,
        num_warps: Optional[int] = 1, kernel: Any = None, probe: Any = None,
        ) -> Optional[CylindricalFusedMagneticPairPlan]:
    """Build the plan from the engine's own objects, or ``None`` when refused.

    ``None`` is the only refusal: a configuration this kernel does not carry must
    fall back to the array path, never raise into a caller that would otherwise
    have stepped correctly.
    """
    if not cylindrical_fused_magnetic_pair_coverage(fields, pml, sources,
                                                    probe=probe).covered:
        return None
    expansion = _resolve_expansion(probe)
    if expansion is None:  # pragma: no cover - the predicate already refused
        return None
    from ..stepping import _boundary_kinds as resolve  # noqa: PLC0415

    curl_spec = SUB_STEPS[CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    grid = fields.grid
    kinds = resolve(grid, pml)
    return CylindricalFusedMagneticPairPlan(
        grid.shape, grid.dt / grid.dx, int(grid.m),
        bool(grid.accurate_fields_near_cylorigin),
        1 if kinds[2] == "metallic" else 0,
        zero_metal_axes(grid), expansion,
        DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in curl_spec["targets"]],
        [getattr(fields, "fu_" + name) for name in curl_spec["targets"]],
        [getattr(fields, name) for name in curl_spec["sources"]],
        [getattr(pml, f"{stem}_{axis}{curl_spec['suffix']}")
         for axis in "xyz" for stem in ("kms", "sinv")],
        [getattr(fields, name) for name in side_spec["targets"]],
        [getattr(fields, name) for name in side_spec["aux"]],
        # The curl takes the HALF-INTEGER lattice on step_B and the constitutive
        # the INTEGER one on H (stepping.py:948 against the B curl's
        # half_integer=True). The kernel takes both and never asks which is which,
        # so a swap here is a silent half-cell error in the absorber profile; the
        # gate carries a mutation for exactly it.
        [getattr(pml, f"{stem}_{axis}") for axis in "xyz"
         for stem in ("kps", "kms")],
        grid.xp, scratch=getattr(fields, "scratch", None),
        kernel=kernel, num_warps=num_warps,
    )


def plan_cylindrical_fused_magnetic_pair_from_arrays(
        arrays: Dict[str, Any], curl_flat: Dict[str, Any],
        constitutive_flat: Dict[str, Any], dtdx: float, m: int,
        accurate_fields_near_cylorigin: bool, bcz: int, zero_metal,
        expansion: int, xp: Any, scratch: Any = None,
        block: Optional[int] = None, kernel: Any = None,
        num_warps: Optional[int] = 1) -> CylindricalFusedMagneticPairPlan:
    """Build from bare device arrays — the gate's route.

    No coverage predicate runs: the caller is a harness that constructed the
    configuration deliberately, including the deliberately wrong ones. ``curl_flat``
    is keyed on the sub-lattice the caller ALREADY SELECTED, so the gate can hand
    over a swapped pair and measure that the swap is caught, and ``kernel=`` carries
    the mutation override — dropping it silently disarms every mutation leg.
    """
    curl_spec = SUB_STEPS[CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    shape = tuple(int(n) for n in arrays[curl_spec["targets"][0]].shape)
    return CylindricalFusedMagneticPairPlan(
        shape, dtdx, m, accurate_fields_near_cylorigin, bcz, zero_metal,
        int(expansion), DEFAULT_BLOCK if block is None else block,
        [arrays[name] for name in curl_spec["targets"]],
        [arrays["fu_" + name] for name in curl_spec["targets"]],
        [arrays[name] for name in curl_spec["sources"]],
        [curl_flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
        [arrays[name] for name in side_spec["targets"]],
        [arrays[name] for name in side_spec["aux"]],
        [constitutive_flat[f"{stem}_{axis}"]
         for axis in "xyz" for stem in ("kps", "kms")],
        xp, scratch=scratch, kernel=kernel, num_warps=num_warps,
    )
