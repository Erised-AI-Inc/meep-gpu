"""The CYLINDRICAL (Dcyl) curl sub-steps at m = 0 — one kernel, two predicates, two plans.

WHAT IT REPLACES. ``stepping.step_B`` and ``stepping.step_D`` on a Dcyl grid with
``grid.m == 0`` and an active split-field PML — everything except the radial prefix
scan, which stays on the array path DELIBERATELY and by measurement (see "THE SCAN"
below). One ``@triton.jit`` body serves both sub-steps, as
:func:`kernels.pml_curl_step` does for the Cartesian pair.

WHY THIS FILE EXISTS AT ALL — the case is LAUNCH-BOUND, not bandwidth-bound.
Measured on the GPU host (RTX A6000, GPU 7, cupy 13.5.1 / triton 3.1.0, 2026-08-10;
raw JSON ``<staging-root>/cyl_recon_2026-08-10/``): step time is FLAT across a
4x cell increase — 2.95 ms at (320, 1, 320) = 102,400 cells against 3.04 ms at
(640, 1, 640) = 409,600, i.e. 34.7 Mcells/s versus 135 Mcells/s for the same wall
clock. One elementwise CuPy launch on this shape costs 9.54 us; ``step_B`` costs
954 us (100 launch-equivalents) and a whole driver step 2916 us (306). Fusing the
three curls, the ghosts, the ownership mask, the PML recurrence and the m = 0 axis
rules into ONE launch per sub-step is what this kernel does, on a case the benchmark
suite currently refuses outright at every tier.

**MEASURED** (same box, GPU 7, ``results/triton_cylindrical_2026-08-10/gate.json``,
bench leg): the curl pair falls from **1.993 ms to 0.550 ms at (320, 1, 320) — 3.62x**
— and from 2.065 ms to 0.439 ms at (640, 1, 640) — **4.70x**, i.e. 51.4 -> 186.1 and
198.4 -> 932.0 Mcells/s. The recon PROJECTED 21x on the pair and that projection is
**not met**: it assumed the array-path prefix would cost about 8 launches, and the
remaining 0.550 ms is still ~58 launch-equivalents, most of it the prefix's own dozen
launches plus Triton's Python-side launch cost. 3.62x on the pair is roughly 2.0x on
a whole step by arithmetic (2916 - 2016 + 550 us), which is the same order as the
4.27x §14 measured on ``2d_pml`` — but it is arithmetic, not a whole-step measurement,
and the optional prefix-fusion follow-on at the bottom of this file is where the rest
of the projected headroom actually lives.

THE SCAN STAYS ON THE ARRAY PATH, and this is the round's decisive measurement, not
a convenience:

* ``cupy.cumsum(axis=0)`` in float32 is DETERMINISTIC (repeat runs and ``out=``
  byte-equal) but is NOT a sequential float32 accumulation — 87,270 of 102,400
  elements differ at (320, 1, 320); 241,819 of 262,144 at (1024, 256).
* ``numpy.cumsum`` IS byte-equal to a sequential accumulation, at every row
  measured. So the array path's own two backends already disagree bitwise on this
  prefix, and the ORACLE for a GPU kernel is CuPy's scan order — not MEEP's loop
  and not exact arithmetic.
* ``tl.cumsum`` in a single tile (BLOCK_R 64/512/1024, one program per column,
  ``enable_fp_fusion=False``) matches NEITHER: 57,172 of 102,400 elements differ
  from CuPy at (320, 320). Tiling is not the blocker — nr <= BLOCK_R fits one tile
  per column with no cross-tile carry, verified at nr = 1024. Summation ORDER is,
  and it is a CuPy-version-dependent one.
* It is not worth much anyway: the two prefix computations are 7.2 % of the whole
  step at (320, 1, 320) and 8.7 % at (640, 1, 640); the two cumsum launches alone
  are 2.0 % / 6.7 %. And the scan's only parallelism is the nz columns (nphi is
  always 1), i.e. 320-640 programs on an 84-SM device.

A PARTIAL COVER, therefore, and its ceiling is real: with the prefix on the array
path the curl pair cannot go below ~8 launches of prefix work, so the 21x above is
bounded to ~11-13x. The prefix ALSO CANNOT BE ALGEBRAICALLY ELIMINATED —
``prefix[i+1] - prefix[i] != increment[i+1]`` in float32; the ``bz_flat_grouping``
mutation is exactly that error and it moved 211 ``Bz`` floats on step 0.

WHAT THE GATE MEASURED, AND THE STEP BUDGET THAT MAY BE CLAIMED.
``parity/meep_gpu/gate_triton_cylindrical.py``, the GPU host, RTX A6000 GPU 7 (3 MiB
resident when taken), cupy 13.5.1 / numpy 2.2.6 / triton 3.1.0,
``results/triton_cylindrical_2026-08-10/``:

* **single launch: 32/32 bytewise identical, guarded** — 4 shapes x 4 Courant
  numbers (three non-power-of-two) x both sub-steps, every target and every
  auxiliary compared as uint32.
* **0/32 identical unguarded** (``enable_fp_fusion=True``), including 0/24 at the
  non-power-of-two Courant numbers. The control bites.
* **9/9 mutations as predicted**, with a 16/16 clean control and no false positive.
  Eight are caught; ``axis_ghost_sign`` — putting the r_to_minus_r near ghost BACK —
  is **16/16 identical, i.e. NOT caught**, which is the prediction below confirmed on
  hardware rather than argued.
* **consecutive steps: first divergence at step 23-72**, one float, always an
  ``fu_*`` auxiliary, always a subnormal against a flushed zero (e.g. array path
  ``0x80000000`` = -0.0 against Triton ``0x8039955d`` = -5.29e-39). **This is plan
  §16's amplifying disagreement, not a defect here**, and the controls in the same
  run say so: the array path against ITSELF is identical 4/4 over 1000 steps, and the
  array path against itself **with one mantissa bit flipped diverges at step 1 in
  4/4**. CuPy flushes subnormals and this kernel does not; §16.6 lists the three ways
  out and this tranche takes none of them.

  **So the claimable budget is: bit-identical PER LAUNCH, everywhere in the sweep;
  and over consecutive steps, to step 23 at the worst row measured.** Not 400, which
  is the NumPy transcription's budget, and not 1000, which is only what the gate ran.
* the EXISTING ``kernels.constitutive_step`` **is** bytewise identical to
  ``stepping.update_H``/``update_E`` on a Dcyl grid, 4/4 — the claim under
  :func:`cylindrical_constitutive_coverage` is measured, not inferred.

WHAT IS TRANSCRIBED, and from where (``meep_gpu/stepping.py``, this tree):

* term tables            ``B_CURL_TERMS`` / ``D_CURL_TERMS``      :213-223
* Yee shifts             ``fields.IYEE_SHIFTS``                   fields.py:214-219
* ghost rules            ``_shift_up`` :1723 / ``_shift_down`` :1787 (CYL_AXIS
                         :1782 far face, :1836-1846 near ghost)
* curl grouping          ``_curl_from_operands``                  :1601
* ownership mask         ``_mask_non_owned_cells``                :1865 (is_axis
                         clause :1899-1904)
* PML recurrence         ``_apply_pml_update``                    :1905
* coefficient pairing    ``_curl_coefficients``                   :2418
* B-side prefix + wall   ``step_B`` cylindrical branch            :295-346
* D-side prefix          ``step_D`` cylindrical branch            :410-426
* the scan itself        ``cylindrical_rderiv_prefix``            :1257-1305
* m = 0 axis rules       ``_cylindrical_axis_zero_B`` :648 / ``_D`` :560

THE m = 0 CYLINDRICAL ADDITIONS, in the order the kernel applies them:

1. **Bz's whole curl is replaced.** ``step_B`` :343-346 extends Ep (= Ey) by ONE
   ZERO WALL ROW to (nr+1, 1, nz), prefixes it at ``ir0 = 0.0``, and computes
   ``dtdx * (prefix_ext[1:] - prefix_ext[:-1])`` — ONE subtract then ONE multiply,
   NOT the four-operand grouping. The wall row is not decoration: without it the
   forward difference at the last row becomes minus the whole accumulated sum
   (the ``no_wall_row`` mutation moves 39 ``Bz`` floats, all in the last rows).
2. **Dz's ``first`` source is the prefix.** ``step_D`` :416-426 prefixes Hp (= Hy)
   at ``ir0 = 0.5`` and substitutes it for the Dz term ONLY; the generic backward
   machinery then produces the derivative. ``Dx``'s Hy operands stay raw.
3. **The is_axis ownership mask** zeroes curl row 0 for every target whose r-shift
   is 0 — ``Bx`` on the B side, ``Dy`` and ``Dz`` on the D side.
4. **The axis rules**, m = 0 only: ``Br[0] = 0``; ``Dz[0] += (4*Courant)*Hp[0]``
   (a POST-add, deliberately not folded into the curl — measured, the fold breaks
   m = 0 under PML, stepping.py:552-558); ``Dp[0] = 0``.

THE r_to_minus_r NEAR GHOST IS **NOT IMPLEMENTED**, AND THAT IS A DECISION WITH A
PROOF BEHIND IT. ``_shift_down``'s CYL_AXIS branch (stepping.py:1836-1846) writes a
sign-flipped image of stored row 0 into the near ghost. It is UNOBSERVABLE: the only
terms taking a shift-down along r are ``Dp`` (partner Hz, second operand) and ``Dz``
(partner Hp, first operand), and both have Yee r-shift 0, so
``_mask_non_owned_cells`` zeroes their curl at row 0 — the only row the near ghost
writes. Measured rather than argued, twice: inverting the sign of that whole ghost
plane for EVERY ``shift_down`` on the axis, at m = 0, 1 AND 2, over 20 full
step_B/step_D pairs, changed ``components_changed: []`` in all three; and on hardware
the gate's ``axis_ghost_sign`` mutation — which puts a faithful ghost BACK into this
kernel — came back **16/16 bytewise identical, uncaught**, alongside eight mutations
that were caught. The kernel therefore serves an exact 0.0 there (the ``other=`` of a
masked load) and says so HERE, because **no byte gate can certify that choice**: if a
future run ever reports that mutation CAUGHT, the ownership mask has broken, not the
ghost. This is the same shape as the two worst defects this project has had, which
were silent wrong answers rather than crashes.

THE SIGNED-ZERO TRAP, which only a bytewise gate finds. The phi axis has n = 1, so
its rolled operand equals the original and ``(second - shifted_second)`` is exactly
``+0.0``. A kernel that "optimizes away" the invariant-axis half of the curl returns
``dtdx*(a-b)`` where the array path returns ``dtdx*((a-b) + 0.0f)`` — identical
except when ``(a-b)`` is ``-0.0``, where the array path yields ``+0.0``
(``0x00000000``) and the shortcut yields ``-0.0`` (``0x80000000``). Both differences
are computed honestly below; do not fold the invariant axis out.

WHAT IS NOT COVERED, and must not be attempted as a flag on this kernel: **m != 0**.
It forces complex64 storage, which no kernel in this package carries and for which
Triton has no dtype (it would mean an interleaved float32 view with stride 2 threaded
through every load, store and the recurrence), plus the ``i*m/r`` coupling, the
|m| = 1 axis increments folded into curl row 0 on BOTH sides, and the |m| >= 2
six-component zeroing of rows ``[0:|m|]`` with auxiliaries. That is a second kernel.
:func:`cylindrical_curl_coverage` refuses ``m != 0`` BY NAME rather than by the
absence of an ``i*m/r`` term.

A CUPY VERSION IS PART OF THIS KERNEL'S CORRECTNESS CONTRACT, in a way it is not for
any other kernel here: everything downstream of the scan is bit-identical only for a
given ``cupy.cumsum`` summation order. cupy 13.5.1 was measured deterministic across
repeats and across ``out=``; a CuPy bump is a correctness event for this plan exactly
as a Triton bump already is for ``kernels.py``, and a SILENT one — the failure mode is
a smooth plausible field and the corpus parity gates are tolerance-based. The gate
records the CuPy version; re-run it on any bump.

THE CENTRAL COMPOSER IS WIRED, but production dispatch is not. ``launch.plan_step``
selects these B/D curl plans and the separately measured pointwise H/E plans only
for the strict predicate below. Cartesian coverage remains unchanged and all
overlaps fail closed. ``fastpath.py`` remains untouched: this is a research plan
selected only by an explicit caller, not an active driver path.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import coverage as _coverage
from .coverage import Coverage

try:  # pragma: no cover - the absent branch is exercised by the absence test
    import triton
    import triton.language as tl
    _TRITON_IMPORT_ERROR: Optional[BaseException] = None
except ImportError as _exc:  # pragma: no cover - the laptop path
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_IMPORT_ERROR = _exc


# ---------------------------------------------------------------------------
# The configuration this kernel is compiled for, spelled as data
# ---------------------------------------------------------------------------

#: The ghost rule per axis on a Dcyl grid, as ``stepping._boundary_kinds`` resolves
#: it: r = 0 is the cylindrical axis, phi is the one-cell invariant axis (which
#: resolves to PERIODIC and whose wrap is an exact zero difference), z is walled.
#: CONSTANT for this kernel, so it is spelled here and compiled in rather than
#: passed — a Dcyl grid that resolves to anything else is refused by the predicate,
#: not stepped with different constexprs.
CYLINDRICAL_BOUNDARY_KINDS: Tuple[str, str, str] = ("axis", "periodic", "metallic")

#: ``grid.is_axis`` on the three axes. Refused unless it is exactly this.
CYLINDRICAL_AXIS_FLAGS: Tuple[bool, bool, bool] = (True, False, False)

#: The one azimuthal order this kernel carries. See the module docstring.
COVERED_M: int = 0

#: Targets, auxiliaries and sources per sub-step, in target order — the same table
#: ``launch.SUB_STEPS`` holds, restated here rather than imported so this module
#: does not depend on a file another track owns. ``suffix`` selects the Yee
#: sub-lattice the curl's kms/sinv come from: HALF-INTEGER for B, integer for D
#: (``stepping._curl_coefficients``); swapped, it is a silent half-cell error in
#: the absorber profile, which is why the gate carries ``coeff_lattice_swap``.
SUB_STEPS: Dict[str, Dict[str, Any]] = {
    "step_B": {
        "targets": ("Bx", "By", "Bz"),
        "sources": ("Ex", "Ey", "Ez"),
        "backward": 0,
        "suffix": "_h",
        # step_B :322-333 — Ep = Ey, extended by its zero wall row, ir0 = 0.0.
        "prefix_component": "Ey",
        "prefix_ir0": 0.0,
        "extend_wall_row": True,
    },
    "step_D": {
        "targets": ("Dx", "Dy", "Dz"),
        "sources": ("Hx", "Hy", "Hz"),
        "backward": 1,
        "suffix": "",
        # step_D :416-419 — Hp = Hy, no wall row (the backward difference reads
        # rows i and i-1), ir0 = 0.5 = half of Hp's r-shift.
        "prefix_component": "Hy",
        "prefix_ir0": 0.5,
        "extend_wall_row": False,
    },
}

#: The six curl targets and their auxiliaries, plus every source volume the two
#: sub-steps read. Enumerated so the predicate can check allocation and layout of
#: exactly what the kernel dereferences.
CURL_TARGETS: Tuple[str, ...] = ("Bx", "By", "Bz", "Dx", "Dy", "Dz")
CURL_SOURCES: Tuple[str, ...] = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")

#: Which axis pair the split-field recurrence reads for target 0, 1 and 2 —
#: vec.hpp's cycle_direction, the same triple on both sides.
DSIG_AXES: Tuple[Tuple[int, int], ...] = ((1, 2), (2, 0), (0, 1))

#: The axis letters the kernel's mask spells, per axis index. phi never appears:
#: it is periodic and masks nothing.
MASK_AXIS_LETTERS: Dict[int, str] = {0: "r", 2: "z"}


def ownership_mask_axes(sub_step: str) -> Tuple[Tuple[int, ...], ...]:
    """Which axes each target's curl is zeroed at cell 0 on — DERIVED, not restated.

    ``stepping._mask_non_owned_cells`` (:1865) drops cell 0 of every axis whose
    boundary is ``mirror``/``metallic``/``axis`` and on which the target's Yee shift
    is 0. Both inputs are the engine's own — ``fields.IYEE_SHIFTS`` and
    :data:`CYLINDRICAL_BOUNDARY_KINDS` — so this function computes what the kernel
    hard-codes, and ``test_triton_cylindrical`` compares the two. Hard-coding it in
    the kernel is right (it is a compile-time fact for this configuration); leaving
    it uncheckable would not be.

    The far-face clause of that function (``_stored_past_owned``, :1723) is not
    reproduced: it fires only on a FOLDED periodic axis, which the predicate refuses.
    """
    from ..fields import IYEE_SHIFTS  # noqa: PLC0415

    masking = {axis for axis, kind in enumerate(CYLINDRICAL_BOUNDARY_KINDS)
               if kind in ("mirror", "metallic", "axis")}
    return tuple(
        tuple(axis for axis in range(3)
              if IYEE_SHIFTS[target][axis] == 0 and axis in masking)
        for target in SUB_STEPS[sub_step]["targets"])

#: Elements per program. The same value and the same reason as
#: ``kernels.DEFAULT_BLOCK`` — not autotuned, because the kernel writes its own
#: inputs in place and ``triton.autotune`` re-runs a kernel to time it (measured on
#: the Cartesian curl: every float wrong, max_abs 1.4e4, no error raised).
#:
#: It is RESTATED rather than imported because ``kernels.py`` imports Triton at
#: module scope and a plan must be buildable on a Triton-less host. The two values
#: are pinned equal by ``test_triton_cylindrical``, which reads the constant off
#: that file's SOURCE.
DEFAULT_BLOCK: int = 256


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------

if triton is not None:

    # NO BOUNDARY CONSTEXPRS. ``kernels.pml_curl_step`` takes BCX/BCY/BCZ because it
    # serves several declarations; this one serves exactly one — the Dcyl triple in
    # :data:`CYLINDRICAL_BOUNDARY_KINDS` — so the rules are compiled in directly and
    # the predicate refuses anything else rather than a constexpr selecting it. That
    # also keeps this file from adding a third code to ``kernels.PERIODIC``/
    # ``METALLIC``, which belongs to another track.

    @triton.jit
    def cyl_pml_curl_step(
        f0, f1, f2,                   # targets:    Bx,By,Bz  or  Dx,Dy,Dz   (in/out)
        u0, u1, u2,                   # auxiliaries: fu_B*    or  fu_D*      (in/out)
        g0, g1, g2,                   # sources:    Ex,Ey,Ez  or  Hx,Hy,Hz   (in)
        pfx,                          # the ARRAY-PATH prefix; see the module docstring
        hp,                           # stored Hp for the m=0 axis add; unread on B
        kmx, sinvx, kmy, sinvy, kmz, sinvz,   # kms/sinv per axis, ONE sub-lattice
        nx, ny, nz, n_elem, dtdx, four_dtdx,
        BACKWARD: tl.constexpr,       # 0 = B (forward differences), 1 = D
        BLOCK: tl.constexpr,
    ):
        """One cylindrical curl sub-step, m = 0: three curls, PML recurrence, axis rules.

        The index arithmetic is the general 3-D one (``i*ny*nz + j*nz + k``) even
        though ``ny`` is always 1 on a Dcyl grid, so a future nphi > 1 is a shape
        change rather than a rewrite. ``pfx`` is (nr+1, ny, nz) on the B side and
        (nr, ny, nz) on the D side; row stride is ``ny*nz`` in both.

        ``four_dtdx`` is a SEPARATE argument rather than ``4.0 * dtdx`` computed
        here, and that is not tidiness: the array path forms ``(4.0 * (dt/dx))`` as
        a Python float and NEP-50 casts it WEAKLY onto float32 storage
        (stepping.py:588), so the multiplicand is the float32 rounding of the
        float64 product. Recomputing it in-kernel from an fp32 ``dtdx`` is a
        different number.
        """
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = idx < n_elem
        nyz = ny * nz
        k = idx % nz
        plane = idx // nz
        j = plane % ny
        i = plane // ny

        # --- the ghost rule, per axis -----------------------------------------
        # r (axis 0) is CYL_AXIS. Forward: ``_shift_up``'s CYL_AXIS branch shares
        # the METALLIC one — a hard zero at the FAR r face, the PEC wall
        # (stepping.py:1782). Backward: the near ghost is r_to_minus_r and is
        # DELIBERATELY NOT IMPLEMENTED — it is unobservable, see the module
        # docstring. An exact 0.0 is served instead, which is what the masked load
        # delivers without dereferencing anything.
        # phi (axis 1) is PERIODIC on a length-1 axis: the wrap returns the SAME
        # element, which is what makes the difference an exact +0.0. Computed, not
        # elided — see the signed-zero trap in the module docstring.
        # z (axis 2) is METALLIC: zero past the wall on both faces.
        if BACKWARD:
            si, sj, sk = i - 1, j - 1, k - 1
        else:
            si, sj, sk = i + 1, j + 1, k + 1
        vr = live & (si >= 0) & (si < nx)
        vz = live & (sk >= 0) & (sk < nz)
        sj = tl.where(sj < 0, ny - 1, tl.where(sj == ny, 0, sj))
        vp = live

        o_r = si * nyz + j * nz + k
        o_p = i * nyz + sj * nz + k
        o_z = i * nyz + j * nz + sk

        a = tl.load(g0 + idx, mask=live, other=0.0)
        b = tl.load(g1 + idx, mask=live, other=0.0)
        c = tl.load(g2 + idx, mask=live, other=0.0)
        a_p = tl.load(g0 + o_p, mask=vp, other=0.0)
        a_z = tl.load(g0 + o_z, mask=vz, other=0.0)
        b_r = tl.load(g1 + o_r, mask=vr, other=0.0)
        b_z = tl.load(g1 + o_z, mask=vz, other=0.0)
        c_r = tl.load(g2 + o_r, mask=vr, other=0.0)
        c_p = tl.load(g2 + o_p, mask=vp, other=0.0)

        # --- the curl (stepping._curl_from_operands): DO NOT flatten these parens -
        curl0 = dtdx * ((c_p - c) + (b - b_z))
        curl1 = dtdx * ((a_z - a) + (c - c_r))
        curl2 = dtdx * ((b_r - b) + (a - a_p))

        # --- the cylindrical substitution on target 2 --------------------------
        if BACKWARD:
            # step_D :425-426 — the Dz term's `first` source is the prefix, and the
            # backward machinery is otherwise unchanged. Its row-0 ghost is
            # unreachable for the same reason every other r near ghost is (Dz's
            # r-shift is 0, so the mask below drops row 0 outright).
            p_here = tl.load(pfx + idx, mask=live, other=0.0)
            p_down = tl.load(pfx + o_r, mask=vr, other=0.0)
            curl2 = dtdx * ((p_down - p_here) + (a - a_p))
        else:
            # step_B :343-346 — Bz's WHOLE curl is the forward difference of the
            # EXTENDED prefix: one subtract, one multiply. The four-operand grouping
            # above is a different float32 number (mutation `bz_flat_grouping`
            # measures the sibling error at 211 floats on step 0).
            curl2 = dtdx * (tl.load(pfx + idx + nyz, mask=live, other=0.0)
                            - tl.load(pfx + idx, mask=live, other=0.0))

        # --- ownership mask (stepping._mask_non_owned_cells, is_axis + metallic) --
        # Per axis, for every target whose Yee shift there is 0 (fields.IYEE_SHIFTS):
        #   B: Bx(0,1,1) -> r ;  By(1,0,1) -> phi only, which is periodic: nothing ;
        #      Bz(1,1,0) -> z
        #   D: Dx(1,0,0) -> phi (nothing) + z ;  Dy(0,1,0) -> r + z ;  Dz(0,0,1) -> r
        at_r, at_z = i == 0, k == 0
        if BACKWARD:
            curl0 = tl.where(at_z, 0.0, curl0)
            curl1 = tl.where(at_r, 0.0, curl1)
            curl1 = tl.where(at_z, 0.0, curl1)
            curl2 = tl.where(at_r, 0.0, curl2)
        else:
            curl0 = tl.where(at_r, 0.0, curl0)
            curl2 = tl.where(at_z, 0.0, curl2)

        # --- split-field recurrence (stepping._apply_pml_update) ---------------
        # dsig/dsigu follow vec.hpp's cycle_direction: target 0 takes (y, z),
        # target 1 (z, x), target 2 (x, y) — the same triple on both sides.
        km_x = tl.load(kmx + i, mask=live, other=0.0)
        si_x = tl.load(sinvx + i, mask=live, other=0.0)
        km_y = tl.load(kmy + j, mask=live, other=0.0)
        si_y = tl.load(sinvy + j, mask=live, other=0.0)
        km_z = tl.load(kmz + k, mask=live, other=0.0)
        si_z = tl.load(sinvz + k, mask=live, other=0.0)

        p0 = tl.load(u0 + idx, mask=live, other=0.0)
        n0 = ((p0 * km_y) - curl0) * si_y
        v0 = (((tl.load(f0 + idx, mask=live, other=0.0) * km_z) + n0) - p0) * si_z

        p1 = tl.load(u1 + idx, mask=live, other=0.0)
        n1 = ((p1 * km_z) - curl1) * si_z
        v1 = (((tl.load(f1 + idx, mask=live, other=0.0) * km_x) + n1) - p1) * si_x

        p2 = tl.load(u2 + idx, mask=live, other=0.0)
        n2 = ((p2 * km_x) - curl2) * si_x
        v2 = (((tl.load(f2 + idx, mask=live, other=0.0) * km_y) + n2) - p2) * si_y

        # --- the m = 0 axis rules, folded into the STORED value ----------------
        # B side (_cylindrical_axis_zero_B, stepping.py:648): Br[0] = 0.
        # D side (_cylindrical_axis_zero_D, :560): Dz[0] += (4*Courant)*Hp[0] — a
        # POST-add, applied to the updated field and NOT folded into the curl
        # (measured: the fold breaks m = 0 under PML) — then Dp[0] = 0. The two
        # touch different components, so their order is immaterial.
        if BACKWARD:
            v1 = tl.where(at_r, 0.0, v1)
            v2 = tl.where(at_r, v2 + four_dtdx * tl.load(hp + idx, mask=live, other=0.0),
                          v2)
        else:
            v0 = tl.where(at_r, 0.0, v0)

        tl.store(u0 + idx, n0, mask=live)
        tl.store(u1 + idx, n1, mask=live)
        tl.store(u2 + idx, n2, mask=live)
        tl.store(f0 + idx, v0, mask=live)
        tl.store(f1 + idx, v1, mask=live)
        tl.store(f2 + idx, v2, mask=live)

else:  # pragma: no cover - the laptop path
    cyl_pml_curl_step = None  # type: ignore[assignment]


def cylindrical_curl_kernel() -> Any:
    """The shipped ``@triton.jit`` kernel, or a diagnosable ImportError.

    The accessor exists so a caller that needs the kernel gets an explanation rather
    than a ``None`` that fails later as a ``TypeError`` far from its cause.
    """
    if cyl_pml_curl_step is None:
        raise ImportError(
            "the cylindrical curl kernel needs the optional `triton` package "
            "(pip install triton). The engine runs without it; only this fast path "
            f"is unavailable. Original error: {_TRITON_IMPORT_ERROR}")
    return cyl_pml_curl_step


# ---------------------------------------------------------------------------
# The prefix — the part that stays on the array path
# ---------------------------------------------------------------------------

def cylindrical_prefix(xp: Any, sub_step: str, sources: Dict[str, Any],
                       scratch: Any = None) -> Any:
    """The radial prefix the kernel consumes, computed by the SHIPPED array path.

    ``stepping.cylindrical_rderiv_prefix`` is called, not re-derived: it is the one
    function whose float32 summation order defines the answer (see the module
    docstring), and the same ``StepScratch`` is threaded through so its two cached
    invariant row vectors are shared with the array path rather than rebuilt.

    On the B side the source is Ep (= Ey) EXTENDED BY ONE ZERO WALL ROW to
    (nr+1, ...) — ``stepping.step_B`` :322-333, transcribed line for line including
    the pooled-versus-fresh allocation branch, because the two writes below are what
    ``xp.concatenate`` used to do and the pool is what stopped it allocating the whole
    volume again every step. On the D side the source is Hp (= Hy), unextended: the
    backward difference reads rows ``i`` and ``i-1`` and never needs the wall.
    """
    from ..stepping import _face, _span, cylindrical_rderiv_prefix  # noqa: PLC0415

    spec = SUB_STEPS[sub_step]
    source = sources[spec["prefix_component"]]
    if not spec["extend_wall_row"]:
        return cylindrical_rderiv_prefix(xp, source, spec["prefix_ir0"], scratch=scratch)

    rows = source.shape[0]
    extended_shape = (rows + 1,) + source.shape[1:]
    extended = (xp.empty(extended_shape, dtype=source.dtype) if scratch is None
                else scratch.take("cyl_extended", extended_shape, source.dtype))
    extended[_span(0, 0, rows)] = source
    extended[_face(0, rows)] = 0
    return cylindrical_rderiv_prefix(xp, extended, spec["prefix_ir0"], scratch=scratch)


# ---------------------------------------------------------------------------
# The coverage predicates
# ---------------------------------------------------------------------------
#
# WRITTEN POSITIVELY AND ENUMERATING. Every requirement is named and checked and
# coverage is never inferred from the absence of a known blocker — the two worst
# defects in this project were silent wrong answers, not crashes.
#
# THEY DO NOT CALL ``coverage._grid_reasons``. That function's clause 6
# (coverage.py:172-178) is a BLANKET cylindrical refusal shared by every Cartesian
# predicate in that file; composing it here would make these predicates vacuously
# False. Its clauses are restated below, individually, with the cylindrical ones
# replaced by the positive Dcyl requirements. That duplication is deliberate and it
# is the reason the shared clause must NOT be widened to admit Dcyl: it is now
# load-bearing in two directions at once, and whoever owns ``coverage.py`` should be
# told so.

#: The shared helpers these predicates DO reuse, named as data so a test can assert
#: every one of them still exists and a rename in the shared file fails at the merge
#: bar instead of silently dropping a clause.
SHARED_CLAUSES: Tuple[str, ...] = (
    "_layout_reasons", "_volume_reasons", "_coefficient_reasons",
    "_susceptibility_reasons", "_call")


def cylindrical_axis_policy(grid: Any) -> Optional[bool]:
    """``grid.accurate_fields_near_cylorigin``, READ rather than assumed.

    It is IRRELEVANT AT m = 0 — it selects how many near-axis rows the |m| >= 2
    stability hack holds at zero (``stepping._cylindrical_axis_rows``, :601-624), and
    at m = 0 there is no such hack. It is read and reported anyway, and a grid that
    cannot answer is REFUSED, so that a later widening to |m| >= 2 cannot inherit
    this module's silence about it.
    """
    value = getattr(grid, "accurate_fields_near_cylorigin", None)
    return None if value is None else bool(value)


def _shared_reasons(fields: Any, pml: Any, grid: Any) -> List[str]:
    """The non-cylindrical half of ``coverage._grid_reasons``, restated.

    Clause numbering follows that function so the two can be diffed by eye. Clauses
    5 and 6 (fold / Cartesian-only) are NOT here — they are replaced by the positive
    Dcyl requirements in :func:`cylindrical_curl_coverage`.
    """
    reasons: List[str] = []

    # 1. CuPy backend. The kernel launches against device pointers, and — see the
    #    module docstring — the prefix's summation order is CuPy's, not NumPy's.
    xp = getattr(grid, "xp", None)
    if getattr(xp, "__name__", "") != "cupy":
        reasons.append(f"array module is {getattr(xp, '__name__', xp)!r}, not cupy")

    # 2. Real storage. At m = 0 Dcyl storage IS real float32 (measured on a lifted
    #    grid: Ez float32 at m = 0, complex64 at m = 1 and 2), which is the whole
    #    reason this kernel needs no complex arithmetic.
    if getattr(fields, "force_complex_fields", False):
        reasons.append("force_complex_fields=True (complex64 storage is not carried)")

    # 3. An active split-field absorber. Without one the plain path is the
    #    bit-identical one and this recurrence is the wrong sub-step entirely.
    if pml is None or not getattr(pml, "is_active", False):
        reasons.append("no active PML layer (this kernel implements the split-field "
                       "path only)")

    # 7. k = 0. A Bloch phase needs complex storage and multiplies one wrapped plane.
    if getattr(grid, "has_bloch", False):
        reasons.append(f"nonzero k_point {getattr(grid, 'k_point', None)!r}")
    k_point = getattr(grid, "k_point", (0.0, 0.0, 0.0))
    if any(float(component) != 0.0 for component in k_point):
        reasons.append(f"k_point {tuple(k_point)!r} is not exactly zero")

    # 8. No conductivity on any curl target: that routes to the three-history
    #    conductive-PML recurrence, not to this one.
    reader = getattr(fields, "condfac_for", None)
    if callable(reader):
        for target in CURL_TARGETS:
            if reader(target) is not None:
                reasons.append(f"a conductivity is installed on {target} (mp.Absorber path)")
    if getattr(fields, "has_magnetic_conductivity", False):
        reasons.append("a magnetic (B) conductivity is installed")

    # 10. No instantaneous nonlinearity: the Pade factor REPLACES the constitutive
    #     product and composes with dispersion.
    if getattr(fields, "has_nonlinearity", False):
        reasons.append("chi2/chi3 is installed (the Pade constitutive factor is not carried)")

    # 11/12. BFAST adds a second additive term to every curl; beta adds out-of-plane
    #        couplings. Both are silent additions, not errors.
    if getattr(grid, "bfast_active", False):
        reasons.append("BFAST is active (the second additive curl term is not carried)")
    if float(getattr(grid, "beta", 0.0)) != 0.0:
        reasons.append(f"special_kz beta={getattr(grid, 'beta', 0.0)!r} is nonzero")

    # 9a/9b. A registered susceptibility must be one this package understands. Not a
    #        refusal of dispersion: it is constitutive-only and the curl differences
    #        the STORED E and H arrays. The clause is vacuous against the engine as
    #        it stands and is written anyway.
    reasons.extend(_coverage._susceptibility_reasons(fields))

    return reasons


def _cylindrical_geometry_reasons(fields: Any, pml: Any, grid: Any) -> List[str]:
    """The POSITIVE Dcyl requirements — what replaces the blanket clauses 5 and 6."""
    reasons: List[str] = []

    if not getattr(grid, "cylindrical", False):
        reasons.append("grid is not cylindrical (this kernel steps Dcyl grids only)")

    # m = 0 EXACTLY, refused by name. Everything else needs complex storage, the
    # i*m/r coupling and per-|m| axis rules — a second kernel, not a flag.
    m = getattr(grid, "m", None)
    if m is None:
        reasons.append("grid does not report m")
    elif int(m) != COVERED_M:
        reasons.append(
            f"grid.m = {m!r}; this kernel carries m = {COVERED_M} ONLY. |m| >= 1 forces "
            "complex64 storage, the i*m/r coupling and the per-|m| axis rules "
            "(stepping.py:674-733, :524-559, :625-647) — a separate kernel")

    flags = tuple(bool(_coverage._call(grid, "is_axis", axis, default=False))
                  for axis in range(3))
    if flags != CYLINDRICAL_AXIS_FLAGS:
        reasons.append(f"grid.is_axis {flags!r} is not {CYLINDRICAL_AXIS_FLAGS!r} "
                       "(r must be the leading axis)")

    kinds = _coverage._boundary_kinds(
        grid, pml if (pml is not None and getattr(pml, "is_active", False)) else None)
    if kinds is None:
        reasons.append("boundary kinds could not be resolved for this grid")
    elif tuple(kinds) != CYLINDRICAL_BOUNDARY_KINDS:
        reasons.append(f"boundary kinds {tuple(kinds)!r} are not "
                       f"{CYLINDRICAL_BOUNDARY_KINDS!r}")

    # No fold anywhere. Grid already refuses a mirror on a Dcyl cell; the clause is
    # written because "another module already guards it" is exactly the reasoning
    # this file exists to refuse.
    if _coverage._call(grid, "has_symmetry", default=False):
        reasons.append("a mirror plane is active (symmetry folding is not carried)")
    for axis in range(3):
        if _coverage._call(grid, "is_mirrored", axis, default=False):
            reasons.append(f"axis {axis} is folded by a mirror plane")

    shape = tuple(getattr(grid, "shape", ()))
    if len(shape) != 3:
        reasons.append(f"grid shape {shape!r} is not three-dimensional")
    elif int(shape[1]) != 1:
        reasons.append(f"phi extent is {shape[1]} cells, not 1 (Dcyl is 2.5-D: the "
                       "exp(i*m*phi) dependence is analytic)")

    if cylindrical_axis_policy(grid) is None:
        reasons.append("grid does not report accurate_fields_near_cylorigin; the flag "
                       "is irrelevant at m = 0 but must be readable before any |m| >= 2 "
                       "admission inherits this predicate's silence about it")

    # A configuration the array path itself refuses is not one a kernel may step.
    try:
        from ..stepping import _require_cylindrical_steppable  # noqa: PLC0415

        _require_cylindrical_steppable(grid)
    except Exception as exc:  # noqa: BLE001 - an unsteppable grid is refused, not crashed on
        reasons.append(f"stepping refuses this cylindrical grid: {exc!r}")

    return reasons


def cylindrical_curl_coverage(fields: Any, pml: Any) -> Coverage:
    """May the Triton cylindrical curl kernel step this (fields, pml) pair?

    Every clause names its own requirement and appends its own reason; the scan
    continues after a failure so a refusal reports everything that disqualified the
    run rather than the first thing.
    """
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    reasons = _shared_reasons(fields, pml, grid)
    reasons.extend(_cylindrical_geometry_reasons(fields, pml, grid))

    # STORED E — the invariant behind differencing E while a pole is live, and the
    # one an edit that ever makes stores_E optional under PML must be caught by.
    if not getattr(fields, "stores_E", False):
        reasons.append("E is recomputed from D rather than stored")

    # Every volume the kernel dereferences: six targets, six auxiliaries, six
    # sources. The D side additionally reads the stored Hp for the axis add, which
    # `Fields.get_H` returns as `fields.Hy` under an active PML (fields.py:1164) —
    # the same array already checked here.
    names = (CURL_TARGETS + tuple("fu_" + target for target in CURL_TARGETS)
             + CURL_SOURCES)
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")

    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(_coverage._layout_reasons(fields, shape, names))

    # The curl plans EITHER sub-step, so both Yee sub-lattices must be present.
    if pml is not None and getattr(pml, "is_active", False) and len(shape) == 3:
        reasons.extend(_coverage._coefficient_reasons(pml, shape, ("kms", "sinv"),
                                                      ("", "_h")))

    return Coverage(not reasons, tuple(reasons))


def cylindrical_constitutive_coverage(fields: Any, pml: Any, side: str) -> Coverage:
    """May the EXISTING ``kernels.constitutive_step`` run on a Dcyl grid?

    IT NEEDS NO NEW KERNEL, and that is a finding rather than an assumption:
    ``stepping.update_H`` and ``stepping.update_E`` carry NO cylindrical branch at
    all. They are element-wise with per-axis coefficient tables indexed on the
    component's OWN axis, and cylindrical changes nothing there except that axis 1
    has n = 1. What refuses them today is ``coverage._grid_reasons`` clause 6, the
    blanket cylindrical refusal — which is shared, is right for the Cartesian
    predicates, and must stay.

    The enumeration is :func:`cylindrical_curl_coverage`'s minus the curl-specific
    clauses (no ghost rule, no ownership mask, no prefix), plus the E side's own
    refusals: a registered polarization (the source becomes ``D - sum P`` and that
    configuration belongs to the ADE kernel), an off-diagonal chi1inv row (the row
    product reads neighbours and the sub-step stops being element-wise), and
    ``stores_E`` false.

    MEASURED, not left as an argument from absence — this project's rules refuse to
    ship the latter. The gate's ``constitutive`` leg runs the shipped
    ``ConstitutivePlan`` against ``stepping.update_H``/``update_E`` on a real Dcyl
    grid and reports **4/4 bytewise identical** (two shapes x both sides,
    ``parity/meep_gpu/results/triton_cylindrical_2026-08-10/gate.json``).
    """
    if side not in _coverage.CONSTITUTIVE_SIDES:
        raise ValueError(
            f"side must be one of {tuple(_coverage.CONSTITUTIVE_SIDES)}, got {side!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    spec = _coverage.CONSTITUTIVE_SIDES[side]
    reasons = _shared_reasons(fields, pml, grid)
    reasons.extend(_cylindrical_geometry_reasons(fields, pml, grid))

    if side == "E":
        if getattr(fields, "has_polarizations", False) or (
                getattr(fields, "polarizations", ()) or ()):
            reasons.append(
                "a susceptibility is registered: update_E's source is (D - sum P), not D "
                "(fields.py:1096-1105) — that configuration belongs to the ADE kernel")
        if getattr(fields, "has_offdiagonal_epsilon", False):
            reasons.append(
                "an off-diagonal chi1inv row is installed (the row product reads "
                "neighbours; this sub-step is element-wise)")
        if not getattr(fields, "stores_E", False):
            reasons.append("E is recomputed from D rather than stored")

    names = tuple(spec["targets"]) + tuple(spec["aux"]) + tuple(spec["sources"])
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")

    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(_coverage._layout_reasons(fields, shape, names))

    if side == "E" and len(shape) == 3:
        reasons.extend(_coverage._inverse_epsilon_reasons(fields, shape))

    if pml is not None and getattr(pml, "is_active", False) and len(shape) == 3:
        suffix = ("_h",) if spec["half_integer"] else ("",)
        reasons.extend(_coverage._coefficient_reasons(pml, shape, ("kps", "kms"), suffix))

    return Coverage(not reasons, tuple(reasons))


def plan_cylindrical_constitutive(fields: Any, pml: Any, side: str,
                                  block: Optional[int] = None) -> Any:
    """Build the existing elementwise constitutive plan for the admitted Dcyl slice.

    The compiled arithmetic is the ordinary ``constitutive_step``; cylindrical m=0
    changes neither its pointwise expression nor its coefficient layout.  This
    wrapper owns the distinct predicate, so the Cartesian builder cannot silently
    broaden its blanket Dcyl refusal.
    """
    if side not in _coverage.CONSTITUTIVE_SIDES:
        raise ValueError(
            f"side must be one of {tuple(_coverage.CONSTITUTIVE_SIDES)}, got {side!r}")
    if not cylindrical_constitutive_coverage(fields, pml, side).covered:
        return None

    from .launch import plan_constitutive_from_arrays  # noqa: PLC0415

    spec = _coverage.CONSTITUTIVE_SIDES[side]
    suffix = "_h" if spec["half_integer"] else ""
    arrays = {
        name: getattr(fields, name)
        for name in tuple(spec["targets"]) + tuple(spec["aux"]) + tuple(spec["sources"])
    }
    if side == "E":
        arrays.update({"inv_eps_" + name: fields.inverse_epsilon_for(name)
                       for name in spec["targets"]})
    flat = {
        f"{stem}_{axis}": getattr(pml, f"{stem}_{axis}{suffix}")
        for axis in "xyz" for stem in ("kps", "kms")
    }
    return plan_constitutive_from_arrays(side, arrays, flat, block=block)


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class CylindricalCurlPlan:
    """A launchable cylindrical curl sub-step: array-path prefix, then ONE kernel.

    Built two ways and launched ONE way, exactly as :class:`launch.PmlCurlPlan` is:
    :func:`plan_cylindrical_curl` is the engine route (reads a real ``Fields``/``PML``
    and passes through the predicate), :func:`plan_cylindrical_curl_from_arrays` is
    the gate's route (bare device arrays, no predicate). Both produce this object and
    both go through :meth:`run`, so the bytes the gate certifies are the bytes the
    engine would launch.

    IT IS NOT ALLOCATION-FREE, and cannot be: the prefix is recomputed every launch
    from the current sources. With the engine's ``StepScratch`` threaded through it
    allocates nothing beyond what the array path already pools; with ``scratch=None``
    it allocates the same temporaries the array path's unpooled branch does.
    """

    __slots__ = ("sub_step", "shape", "n_elem", "dtdx", "four_dtdx", "backward",
                 "block", "xp", "scratch", "_targets", "_aux", "_sources",
                 "_source_map", "_coefficients", "_grid", "_kernel", "_pointer")

    def __init__(self, sub_step: str, shape, dtdx: float, block: int,
                 targets, auxiliaries, sources, coefficients, xp: Any,
                 scratch: Any = None, kernel: Any = None) -> None:
        from .launch import CupyPointer, _flat  # noqa: PLC0415

        if sub_step not in SUB_STEPS:
            raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
        spec = SUB_STEPS[sub_step]
        self.sub_step = sub_step
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        # float(): the array path multiplies a float32 volume by a Python float,
        # which NumPy/CuPy cast to float32 before the multiply, and Triton types a
        # Python float argument as fp32 — so the two scalars are the same bits.
        self.dtdx = float(dtdx)
        # The 4*Courant of the m = 0 on-axis Dz add, ROUNDED TO FLOAT32 HERE because
        # the array path forms it as a float64 product and NEP-50 casts it weakly
        # (stepping.py:588). Computing 4.0*dtdx inside the kernel from an fp32 dtdx
        # is a different number.
        self.four_dtdx = float(_float32(4.0 * float(dtdx)))
        self.backward = int(spec["backward"])
        self.block = int(block)
        self.xp = xp
        self.scratch = scratch
        self._pointer = CupyPointer
        self._targets = tuple(CupyPointer(a) for a in targets)
        self._aux = tuple(CupyPointer(a) for a in auxiliaries)
        self._sources = tuple(CupyPointer(a) for a in sources)
        # The raw arrays, by component name: the prefix is computed from them per
        # launch, so the plan holds the arrays and not only their addresses.
        self._source_map = {name: array
                            for name, array in zip(spec["sources"], sources)}
        self._coefficients = tuple(CupyPointer(_flat(a)) for a in coefficients)
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        # An override exists for exactly one caller: the gate's source-mutation leg,
        # which compiles a deliberately broken copy of the shipped kernel.
        self._kernel = kernel

    def prefix(self) -> Any:
        """This launch's radial prefix, from the shipped array-path scan."""
        return cylindrical_prefix(self.xp, self.sub_step, self._source_map,
                                  scratch=self.scratch)

    def run(self, guard: Optional[bool] = None) -> None:
        """Launch the sub-step. In place; the driver's references stay valid.

        ``guard`` is not for callers: it exists so the gate can MEASURE the
        contraction guard's effect (identical with, non-identical without) rather
        than assert it. Everything else takes the module constant.
        """
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        spec = SUB_STEPS[self.sub_step]
        kernel = self._kernel if self._kernel is not None else cylindrical_curl_kernel()
        # Hp for the m = 0 on-axis Dz add. On the B side the kernel never reads it,
        # but the argument still has to type, so it is bound to a real float32
        # volume of the right shape rather than to a null (a null makes the
        # launcher's failure a TypeError far from its cause).
        hp = self._pointer(self._source_map[spec["prefix_component"]])
        nx, ny, nz = self.shape
        kernel[self._grid](
            *self._targets, *self._aux, *self._sources,
            self._pointer(self.prefix()), hp,
            *self._coefficients,
            nx, ny, nz, self.n_elem, self.dtdx, self.four_dtdx,
            BACKWARD=self.backward,
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
        )

    def __repr__(self) -> str:
        return (f"CylindricalCurlPlan({self.sub_step}, shape={self.shape}, m=0, "
                f"block={self.block})")


def _float32(value: float) -> Any:
    """``numpy.float32(value)`` without importing NumPy at module scope."""
    import numpy as np  # noqa: PLC0415

    return np.float32(value)


def plan_cylindrical_curl(fields: Any, pml: Any, sub_step: str,
                          block: Optional[int] = None
                          ) -> Optional[CylindricalCurlPlan]:
    """Build a plan from the engine's own objects, or None when out of coverage.

    None is the only refusal: a configuration this kernel does not carry must fall
    back to the array path, never raise into a caller that would otherwise have
    stepped correctly. The Triton import stays BELOW the predicate, so a NumPy host
    can plan (to ``None``) without the optional dependency being importable at all.
    """
    if sub_step not in SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
    if not cylindrical_curl_coverage(fields, pml).covered:
        return None

    spec = SUB_STEPS[sub_step]
    grid = fields.grid
    return CylindricalCurlPlan(
        sub_step, grid.shape, grid.dt / grid.dx,
        DEFAULT_BLOCK if block is None else block,
        [getattr(fields, n) for n in spec["targets"]],
        [getattr(fields, "fu_" + n) for n in spec["targets"]],
        [getattr(fields, n) for n in spec["sources"]],
        [getattr(pml, f"{stem}_{axis}{spec['suffix']}")
         for axis in "xyz" for stem in ("kms", "sinv")],
        grid.xp,
        scratch=getattr(fields, "scratch", None),
    )


def plan_cylindrical_curl_from_arrays(sub_step: str, arrays: Dict[str, Any],
                                      flat: Dict[str, Any], dtdx: float, xp: Any,
                                      scratch: Any = None,
                                      block: Optional[int] = None,
                                      kernel: Any = None) -> CylindricalCurlPlan:
    """Build a plan from bare device arrays — the gate's and benchmark's route.

    ``arrays`` is keyed by component name, ``flat`` by ``kms_x``/``sinv_x``/... on
    the sub-lattice the caller ALREADY SELECTED (half-integer for B, integer for D),
    so the gate can hand over a swapped pair and measure that the swap is caught. No
    predicate runs here: the caller is a harness that constructed the configuration
    deliberately, and refusing it would defeat the point of a mutation leg.
    """
    spec = SUB_STEPS[sub_step]
    shape = tuple(int(n) for n in arrays[spec["targets"][0]].shape)
    return CylindricalCurlPlan(
        sub_step, shape, dtdx, DEFAULT_BLOCK if block is None else block,
        [arrays[n] for n in spec["targets"]],
        [arrays["fu_" + n] for n in spec["targets"]],
        [arrays[n] for n in spec["sources"]],
        [flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
        xp, scratch=scratch, kernel=kernel,
    )


# ---------------------------------------------------------------------------
# CENTRAL-COMPOSER INTEGRATION — completed 2026-08-11
# ---------------------------------------------------------------------------
#
# ``triton_kernels.__init__`` exposes the cylindrical predicates and builders
# lazily. ``launch.plan_step`` selects both cylindrical curls plus ordinary
# elementwise H/E plans only when the specialized predicates admit them; Cartesian
# predicates retain their blanket Dcyl refusal.  The driver order is unchanged, and
# `fuse=True` leaves all four plans separate with named cylindrical refusals because
# the CuPy prefix and the axis rules must be carried explicitly by a new pair kernel.
# No code here touches `fastpath.py` or production dispatch.
#
# An OPTIONAL follow-on, measured and deliberately not taken: the pre-cumsum stage of
# the prefix (weighted multiply, the row-0 zero fill, the subtract, the divide — 7
# launches on the B side including the extension copy, 5 on the D side) is pure
# elementwise work and CAN be fused into one small kernel per side bit-identically,
# taking the prefix from 12 launches to 4. That is ~85 us of a 2916 us step, about
# 3 %. Do it second, or not at all.
