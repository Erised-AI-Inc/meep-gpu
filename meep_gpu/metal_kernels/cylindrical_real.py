"""CYLINDRICAL (Dcyl) curl sub-steps at m = 0 on Metal — real float32 storage.

The Metal port of ``triton_kernels/cylindrical_triton.py``. Source of truth for the
ARITHMETIC is ``stepping.py``; source of truth for the STRUCTURE — the predicate
clauses, the plan shape, the refusals — is the Triton module, which is certified.
The module TEMPLATE followed for the predicate list, the constitutive companion and
the wiring block is :mod:`.bfast_curl`, which is the closest shipped shape: one
inverted clause, a curl kernel of its own, and a constitutive pair it re-admits
without writing a second kernel.

WHAT IT REPLACES. ``stepping.step_B`` (:261) and ``stepping.step_D`` (:379) on a
Dcyl grid with ``grid.m == 0``, real storage and an active split-field PML —
INCLUDING the radial prefix scan, which is the one place this port deliberately
diverges from the Triton one. See "THE SCAN" below.

THE m = 0 CYLINDRICAL ADDITIONS, in the order the kernel applies them, and there are
only three. Everything else — the ghost rule, the curl grouping, the ownership mask
and the split-field recurrence — is the CERTIFIED Cartesian body, unchanged:

1. **Bz's whole curl is REPLACED** (stepping.py:371-375). Ep (= Ey) is extended by
   ONE ZERO WALL ROW to (nr + 1, nphi, nz) (:359-360), prefixed at ``ir0 = 0.0``,
   and the curl is ``dtdx * (prefix_ext[1:] - prefix_ext[:-1])`` — ONE subtract and
   ONE multiply, NOT the four-operand grouping, which is a different float32 number.
   The wall row is not decoration: without it the forward difference at the last row
   becomes minus the whole accumulated sum, and the comment at :303-314 records the
   symptom that found it (the m = 1 run's first divergence born at the LAST TWO ROWS
   at step 3, rows 0-17 exactly zero, growing inward).
2. **Dz's ``first`` SOURCE is swapped** (stepping.py:447-455). Hp (= Hy) is prefixed
   at ``ir0 = 0.5`` — half of Hp's r-Yee shift — and substituted for the Dz term
   ONLY; the unmodified backward-difference machinery then produces the cylindrical
   derivative. ``Dx``'s Hy operands stay RAW, which is why this kernel binds both the
   prefix and the untouched ``g1``. No wall row is needed on this side: a backward
   difference reads rows i and i-1 (:313-314).
3. **The m = 0 axis rules**, applied AFTER the recurrence:
   ``Bx[r=0] = 0`` (:661-662); ``Dz[r=0] += (4*Courant)*Hp[r=0]`` then
   ``Dy[r=0] = 0`` (:586-587). The D-side increment is a POST-ADD and NOT folded
   into the curl, and that was MEASURED rather than chosen: the fold breaks m = 0
   under PML (Er 2.7e-01 / Hp 4.5e-01 against 3.6e-07 for the post-add,
   stepping.py:575-580), because the split-field routing is not MEEP's axis ladder
   for this rule while the plain post-add is exact — Dz's dsig is R, whose sigma is
   zero on the axis row.

WHAT NEEDED NO NEW CODE AT ALL, and both were MEASURED on this host rather than
inherited from the Triton complex family's 80-row result (which is |m| >= 1 and
complex, a different question):

* **The r axis compiles as METALLIC.** ``_shift_up``'s CYL_AXIS arm IS the METALLIC
  arm, character for character (stepping.py:1828-1830). ``_shift_down``'s CYL_AXIS
  arm is NOT (:1877-1889) — it images stored row 0 with a direction sign and the
  ``(-1)^m`` phase — but it is UNOBSERVABLE here: the only terms taking a shift-down
  along r are ``Dy`` (partner Hz) and ``Dz`` (partner Hp), and both have r-Yee shift
  0. ``probe_metal_cylindrical_real.py`` leg ``axis_ghost`` MEASURES it on the ARRAY
  PATH — two engines stepped four whole steps apart, one with the CYL_AXIS near ghost
  forced to METALLIC's zero: **0 differing words of 33,012 over four cases, with 32
  ghost substitutions actually fired** and 2,036 to 14,176 words moved per case. No
  byte gate can certify that choice from inside the kernel, so it is stated HERE.

  THE MECHANISM IS DIFFERENT ON THE TWO TARGETS, and this paragraph said otherwise
  for a round. ``gate_metal_cylindrical_real.py``'s mutation leg armed the ownership
  mask on BOTH as must-catch, on the reading that the mask is what zeroes the row the
  ghost writes; on **Dy that is right** (dropping the mask diverges by 11-40 words per
  case, and the ghost restored WITHOUT the mask is CAUGHT), and on **Dz it is wrong**.
  Dropping the r mask on Dz changes NOTHING — 0/8 — because that curl's r = 0 value
  is ALREADY an exact ``+0.0`` before the mask runs: its radial operand pair is
  ``(prefix[0], the metallic zero ghost)`` and prefix row 0 is an exact ``+0.0`` by
  construction (stepping.py:1314 ``zeros_like``), while its phi pair is a
  self-difference on the one-cell invariant axis. Restoring the near ghost there and
  dropping the mask is *also* 0/8: ``-0.0 - (+0.0)`` is ``-0.0``, which the phi
  self-difference turns back into ``+0.0`` at the add. So on Dz the prefix's own zero
  row absorbs the ghost and the mask is redundant. The CONCLUSION — the near ghost is
  unobservable at m = 0 — is unchanged and separately measured; only the reason for
  half of it was wrong. If a mutation that puts a faithful ghost back is ever reported
  CAUGHT, read the paired mask-dropped twin beside it before concluding anything.
* **The ownership mask is the certified emitter's.** With the r axis compiled
  METALLIC, ``shaders.ownership_mask`` masks exactly the cells
  ``_mask_non_owned_cells`` masks on a real cylindrical grid — the ``is_axis`` clause
  and the ``is_metallic`` clause land on the same rule. Leg ``emitters`` checks the
  emitter's output against the FUNCTION rather than against a reading of it: agreed
  on both sub-steps, 60 masked cells on the B side and 119 on the D side.

THE SCAN RUNS ON THE DEVICE HERE, AND THE TRITON MODULE'S REASON FOR REFUSING IT IS
INVERTED ON THIS BACKEND RATHER THAN IGNORED. ``triton_kernels/cylindrical_triton.py``
:32-56 calls the array-path scan its round's decisive measurement, and it is: on CuPy
the ORACLE ITSELF is unreproducible, because ``cupy.cumsum`` in float32 is not a
sequential accumulation (87,270 of 102,400 elements differ) and ``tl.cumsum`` matches
neither it nor the sequential sum. **The premise does not hold here.** This backend's
engine holds NumPy (``coverage.MIRRORED_HOST_MODULE``), and the same Triton paragraph
records that ``numpy.cumsum`` IS byte-equal to a sequential accumulation — so the
oracle a Metal kernel must reproduce is the sequential one, which a column-serial scan
reproduces by construction (:data:`templates.COLUMN_SERIAL_SCAN` exists for exactly
this). Two facts were measured before the kernel was written, and either could have
refused the design:

* **Metal's float32 ``/`` is NumPy's divide.** 266,436 words over four operand
  classes including the real ``(ir + ir0) - 0.5`` divisor ladder: **0 disagreements
  that are not the platform's subnormal flush** (16 flushed words, all with the
  reference subnormal and the device zero of the same sign). The control BITES:
  ``fast::divide`` misses 83,904 words, **625 of 2,046 on the divisor ladder itself**,
  so choosing ``/`` is measured rather than stylistic.
* **The column-serial scan is the array path's answer.** 48 shipped cases —
  4 shapes x both ``ir0`` values x both wall arms x three value classes
  (``random_band``, ``wide_dynamic`` geometric ladder, ``cancelling`` partial sums
  through zero) — against ``stepping.cylindrical_rderiv_prefix`` ITSELF, called and
  never re-derived: **0 differing words of 257,712**, every case subnormal-free.

  What this buys, stated as a cost rather than a boast: the alternative was a host
  round trip of the source volume and the prefix on EVERY curl sub-step, which hands
  back a large part of the 12x-64x the residency layer exists to buy
  (``device.py:167-172``). What it costs is a second launch per curl sub-step, and a
  scan kernel whose dispatch is sized from its output volume so ``nr - 1`` threads in
  every ``nr`` return immediately. This family's THROUGHPUT claim must therefore be
  measured separately and must NOT be inherited from the certified families'
  benchmark; nothing here is a throughput claim.

WHAT IS METAL-SPECIFIC, and all that is:

* **Source specialisation, not runtime branches.** ``torch.mps.compile_shader`` takes
  a source string and nothing else, so the direction, the boundary triple, the Bz/Dz
  substitution and the axis tail are all baked in, as Triton bakes its constexprs.
* **The contraction pragma is mandatory** and is spelled once, in :mod:`.shaders`,
  reached through :mod:`.templates`. STATED HONESTLY: the probe measured the
  ``fast`` arm of the PREFIX kernel identical to the shipped one, so that leg says
  nothing about the guard — ``w`` is live across the loop iteration and cannot be
  folded away. The guard is still spelled on both kernels, because the CURL body
  below is the certified one whose grouping the guard demonstrably decides, and one
  spelling per package is the rule.
* **Negation is ``-x``, re-measured on THIS backend for THIS family** rather than
  inherited: over an exhaustive signed-zero/subnormal table plus 8,192 random words,
  ``-x`` and ``x * -1.0f`` are both 0/8,462 while ``0.0f - x`` misses 257/8,462
  (256 flushed subnormals and the ``+0.0 -> -0.0`` word). This family puts signed
  zeros live on the axis row — ``Dy[0] = 0`` writes one and the mask writes three
  more — so the spelling is a measurement here, not a citation. The kernel uses no
  unary minus at all; the rule is recorded because the refuted spelling is a gate
  mutation rather than a comment.
* **Twenty-two bindings**, under the measured 31 ceiling (:data:`.device.MAX_BUFFER_BINDINGS`).

WHAT THE WHOLE STEP MEASURED, and it is the only leg that arbitrates the object the
engine would actually run. Six green per-sub-step comparisons say nothing about a
composed step, so the family was walked through the driver's REAL pass list
(driver.py:3279-3304) — ten passes, with ``zero_metal_*`` sitting BETWEEN the two
fill passes — against a second engine stepped entirely by the array path, comparing
every volume after every pass:

* **6,730,560 word comparisons over 5 cases x 12 steps x 10 passes, 0 differing**,
  at four Courant numbers (two of them not powers of two), both z boundary kinds and
  five shapes; ``Residency.verify()`` empty after every one of the 600 passes, so no
  mirror went stale across the array-path seam;
* per-slot launch counters ASSERTED, not reported: each curl slot owes exactly one
  scan and one curl per cycle and delivered (24, 12, 12) over twelve steps, so no
  slot passed by not executing;
* the composer selected all four arithmetic slots on every case, and the composition
  sweep records ``cylindrical m=0`` winning 8 rows with NO new admitted overlap
  beyond the two pre-existing planted ones.

CAN THAT COMPARISON FAIL? Fourteen planted defects say yes, and one of them found a
CASE defect rather than a kernel one:

* **eleven caught**, including the dropped wall row (543 words), the reciprocal
  multiply (6,373), both weight/divisor off-by-ones (18,604 each), the deleted Bz
  substitution (9,124), all three axis rules (120 / 3,118 / 117), the prefix leaking
  into ``Dx`` (9,480), and — the one worth naming — folding the on-axis Dz increment
  INTO the curl instead of post-adding it (3,118), which is the exact error
  stepping.py:575-580 records as measured wrong for m = 0 under PML;
* **three not caught, by design**: the commuted accumulator (float addition IS
  commutative — the control showing the leg does not flag any edit at all), and BOTH
  near-ghost restorations. Those last two are the arm that proves the METALLIC
  substitution is equivalent rather than merely untested, and they agree with the
  array-path probe leg and with the Triton track's own uncaught ``axis_ghost_sign``.
  TWO CORRECTIONS FROM THE GATE, both recorded there with numbers: the restoration
  was planted on ``b_x``, which NO curl on EITHER sub-step consumes, so that null was
  a dead-store artifact and the gate re-arms it on the two LIVE shift-down-along-r
  operands (``c_x`` and ``pb_x``) with paired mask-dropped twins; and the Dz half of
  the null is not a masking fact at all (see the r-axis paragraph above);
* **``curl_bz_flat_grouping`` was NOT caught at Courant 0.5 and WAS at 0.314159**
  (4,390 words). Distributing ``dtdx`` over the prefix difference is exact whenever
  ``dtdx`` is a power of two, so a case matrix without a non-power-of-two Courant
  cannot see grouping errors in this kernel at all. That is a property of the CASES,
  not of the kernel, and it is why the grouping is ALSO pinned as source text in
  ``test_metal_cylindrical_real``.

SUBNORMALS FLUSH ON THIS BACKEND AND CANNOT BE UNFLUSHED, so ``keep`` is not
offerable and every claim above rides on a CHECKED subnormal-free precondition
(:mod:`.subnormal`, consulted through ``coverage._metal_backend_reasons``). The
radial prefix is where a run reaches the band first — it accumulates a decaying field
down the whole r extent — so the family's gate must census the band as a WINDOW
(first step, last step) and not as a scalar.

THERE IS NO GENERATED-CODE AUDIT ON THIS BACKEND. ``compile_shader`` exposes no AIR,
no GPU ISA and no optimisation report (``device.py:46-62``), so every claim in this
module is BEHAVIOURAL: it catches a wrong answer, not a wrong instruction. That is
this port's standing gap against the Triton and CUDA tracks and it stays stated.

WIRED. This module registers two curl arms and the certified constitutive pair in
:mod:`.arms`, and ``plan_step`` composes them; ``fastpath.plan_fast_path`` still
returns ``None`` on every branch, which is the separate decision. What separates
these arms from every other family is clause 6 INVERTED — ``coverage._grid_reasons``
:176-182 refuses ``cylindrical`` and ``is_axis`` BY NAME, and every one of the nine
carried families either uses that list or restates it, so the separation is total and
stated once per family. The internal split against a future |m| >= 1 family is
clause 14, which this family answers ``m == 0`` and that one will answer
``abs(m) >= 1``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..triton_kernels.coverage import (
    CONSTITUTIVE_SIDES,
    CURL_SUB_STEPS,
    CURL_TARGETS,
    Coverage,
    _boundary_kinds,
    _call,
    _coefficient_reasons,
    _inverse_epsilon_reasons,
    _layout_reasons,
    _susceptibility_reasons,
)
# The SAME table `metal_kernels.launch` binds, imported from the Triton package for
# the reason `bfast_curl` imports it: the two ports cannot then disagree about which
# targets, sources, direction or Yee suffix a sub-step has, and this module acquires
# no import edge into the Metal composer.
from ..triton_kernels.launch import SUB_STEPS
from . import shaders, templates
from .coverage import _metal_backend_reasons, _residency_declaration_reasons
from .device import Residency, compile_source

# `.plans.KernelPlan` is DELIBERATELY NOT IMPORTED. This family's curl plan is a
# SIBLING of that base rather than a subclass — see `CylindricalRealCurlPlan` — so an
# import here would be dead weight that reads as an inheritance the class does not
# have. The reason it cannot subclass is stated at the class, not hidden here.

#: The family name the arm table carries. One spelling, so a refusal message, a
#: registry row and an artifact column cannot drift apart.
FAMILY = "cylindrical_real"

__all__ = [
    "FAMILY",
    "CylindricalRealCurlPlan",
    "compile_cylindrical_curl",
    "compile_cylindrical_prefix",
    "cylindrical_real_constitutive_coverage",
    "cylindrical_real_curl_coverage",
    "cylindrical_curl_source",
    "cylindrical_prefix_source",
    "enumerate_cylindrical_real_sources",
    "plan_cylindrical_real_constitutive",
    "plan_cylindrical_real_curl",
    "plan_cylindrical_real_curl_from_arrays",
    "prefix_row_vectors",
    "register_arms",
    "scan_rows",
]

PERIODIC = shaders.PERIODIC
METALLIC = shaders.METALLIC

#: ``ir0 = origin_r*a + 0.5*iyee_shift(Fp).in_direction(R)`` per sub-step
#: (stepping.py:1297-1299). The r origin IS the axis here, so the whole value is the
#: half-shift: Ep sits at the node (r-shift 0) and Hp half a cell out (r-shift 1).
#: Getting these two backwards is a HALF-CELL error in the radial weights and is
#: silent — a smooth, plausible, slightly wrong field — which is why they are a table
#: with the sub-step as its key rather than a literal at each call site.
PREFIX_IR0: Dict[str, float] = {"step_B": 0.0, "step_D": 0.5}

#: Which sub-step's scan carries the ZERO WALL ROW. Only the B side: its Bz curl is a
#: forward difference of the prefix and would otherwise read minus the whole
#: accumulated sum at the last row (stepping.py:331-342). The D side's backward
#: difference reads rows i and i-1 and never looks past the top.
PREFIX_WALL_ROW: Dict[str, bool] = {"step_B": True, "step_D": False}


def scan_rows(sub_step: str, radial_rows: int) -> int:
    """How many rows this sub-step's radial scan covers — ``nr`` or ``nr + 1``."""
    if sub_step not in SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
    return int(radial_rows) + (1 if PREFIX_WALL_ROW[sub_step] else 0)


# ---------------------------------------------------------------------------
# The radial prefix scan
# ---------------------------------------------------------------------------

#: ``cylindrical_rderiv_prefix`` (stepping.py:1286-1334) as one thread per (phi, z)
#: column, summed STRICTLY SERIALLY from r = 0 upward — the shape
#: :data:`templates.COLUMN_SERIAL_SCAN` carries, with the weighting, the divide and
#: the wall row filled in.
#:
#: THE SERIAL SUM IS A LICENCE, NOT A PREFERENCE. float32 addition is not
#: associative, so a blocked or parallel scan is a different number; the Triton
#: tranche measured ``torch.cumsum`` against ``numpy.cumsum`` at 2691/4096, 283/555
#: and 3210/4096 differing words while a column-serial reference sat at 0/4096.
#:
#: THE DIVIDE IS ``/`` AND NOT ``fast::divide``, measured on this host rather than
#: assumed: ``/`` is NumPy's divide over 266,436 words modulo the platform's
#: subnormal flush, while ``fast::divide`` misses 625 of the 2,046 words of the
#: divisor ladder this kernel actually walks.
_PREFIX_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

kernel void cyl_rderiv_prefix(
    device float*       out     [[buffer(0)]],
    device const float* src     [[buffer(1)]],
    device const float* weights [[buffer(2)]],
    device const float* divisor [[buffer(3)]],
    constant uint&      nx      [[buffer(4)]],
    constant uint&      ny      [[buffer(5)]],
    constant uint&      nz      [[buffer(6)]],
    constant uint&      n_cols  [[buffer(7)]],
    uint idx [[thread_position_in_grid]])
{
    // One thread per (phi, z) column. The dispatch is sized from `out`, which is
    // nx times wider, so most threads return here; that is a stated throughput cost
    // of this family and not a correctness one.
    if (idx >= n_cols) { return; }
    int nxi = int(nx), nyi = int(ny), nzi = int(nz);
    int k = int(idx) % nzi;
    int j = int(idx) / nzi;
    if (j >= nyi) { return; }
    int nyz = nyi * nzi;
    int base = j * nzi + k;

    // increment[0] is an exact +0.0 (stepping.py:1285 zeros_like / :1300), and the
    // cumsum's first output IS that zero. Written rather than assumed: -0.0 here
    // would be a different word, and this family compares words.
    float acc = 0.0f;
    out[base] = acc;
    float prev = __SRC0__ * weights[0];
    for (int i = 1; i < nxi; ++i) {
        int o = i * nyz + base;
        // weighted = f_p * weights, field LEFT, exactly as stepping.py:1284/:1298
        // spells it.
        float w = __SRCI__ * weights[i];
        // stepping.py:1286 — the difference of two ALREADY-WEIGHTED rows over the
        // row-constant divisor. NOT `* (1.0f / divisor)`, which is a different
        // float32 number.
        float inc = (w - prev) / divisor[i - 1];
        // STRICTLY SERIAL: numpy.cumsum is out[i] = out[i-1] + in[i].
        acc = acc + inc;
        out[o] = acc;
        prev = w;
    }
}
"""  # stepping.py live lines for the frozen device-text citation(s) in this string: 1285->1314, 1300->1329, 1284->1313, 1298->1327, 1286->1315


def cylindrical_prefix_source(sub_step: str,
                              contract: str = shaders.CONTRACT_OFF) -> str:
    """The radial-scan source for one sub-step.

    The B side's ZERO WALL ROW is a SOURCE specialisation: the scan covers ``nr + 1``
    rows over an ``nr``-row source and the top row reads the literal ``0.0f`` the
    array path assigns (stepping.py:360). Spelling it as a shortened loop instead
    would drop the wall row's own prefix entry, which is the value Bz's last-row
    forward difference reads.
    """
    if sub_step not in SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
    if PREFIX_WALL_ROW[sub_step]:
        src0 = "(0 < nxi - 1 ? src[base] : 0.0f)"
        srci = "(i < nxi - 1 ? src[i * nyz + base] : 0.0f)"
    else:
        src0 = "src[base]"
        srci = "src[i * nyz + base]"
    return templates.substitute(_PREFIX_TEMPLATE, {
        "__CONTRACT__": templates.contraction_pragma(contract),
        "__SRC0__": src0,
        "__SRCI__": srci,
    })


def compile_cylindrical_prefix(sub_step: str,
                               contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised radial-scan entry point for one sub-step."""
    return compile_source(cylindrical_prefix_source(sub_step, contract)).cyl_rderiv_prefix


def prefix_row_vectors(sub_step: str, radial_rows: int, dtype: Any = None
                       ) -> Tuple[Any, Any]:
    """The two row vectors the scan binds — ``(weights, divisor)``, FLAT.

    NOT RE-DERIVED. ``stepping._cylindrical_rderiv_weights`` is the one definition of
    ``(ir + ir0)`` and ``(ir + ir0) - 0.5``, and CALLING it rather than restating it
    is what stops the two drifting. That is the whole reason, and it is enough.

    THE OTHER REASON THIS DOCSTRING USED TO GIVE IS NOT REACHABLE, measured by
    ``gate_metal_cylindrical_real.py`` leg ``prefix_row_vector_rounding``. Both
    ladders are built in float64 and rounded to float32 exactly ONCE
    (stepping.py:1339-1341), and recomputing them in float32 throughout was described
    here as a double rounding on the divisor and a silent half-ulp error down the
    whole radial ladder. It is not: ``ir0`` is 0.0 or 0.5, so the counts are integers
    and the divisor half-integers, both EXACT in float32. The two spellings are
    bit-identical at every radial extent in the gate's matrix and first diverge around
    2^23 rows — four orders of magnitude past any constructible grid. The mutation
    that plants the float32 recomputation is therefore carried as a MEASURED NULL
    rather than quietly dropped.
    """
    import numpy as np  # noqa: PLC0415
    from ..stepping import _cylindrical_rderiv_weights  # noqa: PLC0415

    resolved = np.float32 if dtype is None else np.dtype(dtype).type
    weights, divisor = _cylindrical_rderiv_weights(
        np, scan_rows(sub_step, radial_rows), PREFIX_IR0[sub_step], resolved)
    return (np.ascontiguousarray(weights).reshape(-1),
            np.ascontiguousarray(divisor).reshape(-1))


# ---------------------------------------------------------------------------
# The curl
# ---------------------------------------------------------------------------

#: The CERTIFIED ``shaders._CURL_TEMPLATE`` with a prefix pointer, one host-rounded
#: axis scalar and three substitution slots, and NO other change: the ghost rule, the
#: curl grouping for targets 0 and 1, the ownership mask and the split-field
#: recurrence are character for character the certified body's.
#:
#: 22 BINDINGS: 10 volumes + 6 coefficient vectors + 4 extents + dtdx + the axis
#: scalar. The ceiling is 31 (``device.MAX_BUFFER_BINDINGS``), so this family fits
#: with nine to spare — and the sibling test asserts the count rather than leaving a
#: later edit to discover the ceiling at compile time.
_CYL_CURL_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

kernel void cyl_pml_curl_step(
    device float*       f0        [[buffer(0)]],
    device float*       f1        [[buffer(1)]],
    device float*       f2        [[buffer(2)]],
    device float*       u0        [[buffer(3)]],
    device float*       u1        [[buffer(4)]],
    device float*       u2        [[buffer(5)]],
    device const float* g0        [[buffer(6)]],
    device const float* g1        [[buffer(7)]],
    device const float* g2        [[buffer(8)]],
    device const float* pfx       [[buffer(9)]],
    device const float* kmx       [[buffer(10)]],
    device const float* sinvx     [[buffer(11)]],
    device const float* kmy       [[buffer(12)]],
    device const float* sinvy     [[buffer(13)]],
    device const float* kmz       [[buffer(14)]],
    device const float* sinvz     [[buffer(15)]],
    constant uint&      nx        [[buffer(16)]],
    constant uint&      ny        [[buffer(17)]],
    constant uint&      nz        [[buffer(18)]],
    constant uint&      n_elem    [[buffer(19)]],
    constant float&     dtdx      [[buffer(20)]],
    constant float&     axis_coef [[buffer(21)]],
    uint idx [[thread_position_in_grid]])
{
    // The dispatch is sized from the first tensor argument's element count, so a
    // volume wider than n_elem would step cells the array path does not own.
    if (idx >= n_elem) { return; }

    int nxi = int(nx), nyi = int(ny), nzi = int(nz);
    int nyz = nyi * nzi;
    int ii  = int(idx);
    int k   = ii % nzi;
    int plane = ii / nzi;
    int j   = plane % nyi;
    int i   = plane / nyi;

    // --- the ghost rule, per axis (stepping._shift_up:1723 / _shift_down:1787) --
    // THE r AXIS IS COMPILED METALLIC. _shift_up's CYL_AXIS arm IS the METALLIC arm
    // (:1781-1783). _shift_down's is not (:1830-1842) but is unobservable at m = 0:
    // only Dy and Dz take a shift-down along r, both have r-Yee shift 0, and the
    // ownership mask zeroes their curl at exactly the row that ghost writes.
    // MEASURED on the array path, 0/33,012 words over four whole steps.
    int si = i __SHIFT__, sj = j __SHIFT__, sk = k __SHIFT__;
    bool vx = true, vy = true, vz = true;
__GHOST_X__
__GHOST_Y__
__GHOST_Z__

    int ox = si * nyz + j * nzi + k;
    int oy = i * nyz + sj * nzi + k;
    int oz = i * nyz + j * nzi + sk;

    // METALLIC serves an exact 0.0 past the wall -- Triton's `other=0.0`. The
    // ternary short-circuits, so the out-of-range offset is never dereferenced.
    float a   = g0[ii];
    float b   = g1[ii];
    float c   = g2[ii];
    float a_y = vy ? g0[oy] : 0.0f;
    float a_z = vz ? g0[oz] : 0.0f;
    float b_x = vx ? g1[ox] : 0.0f;
    float b_z = vz ? g1[oz] : 0.0f;
    float c_x = vx ? g2[ox] : 0.0f;
    float c_y = vy ? g2[oy] : 0.0f;

__PREFIX_LOADS__

    // --- the curl (stepping._curl_from_operands:1601): DO NOT flatten these parens
    // Targets 0 and 1 are the certified Cartesian expressions unchanged; target 2 is
    // the cylindrical substitution.
    float curl0 = dtdx * ((c_y - c) + (b - b_z));
    float curl1 = dtdx * ((a_z - a) + (c - c_x));
__CURL2__

    // --- ownership mask (stepping._mask_non_owned_cells:1865) -------------------
    // With the r axis compiled METALLIC this is the CERTIFIED emitter's output, and
    // it masks exactly the cells the is_axis clause (:1898-1902) masks. Checked
    // against the function itself, not against a reading of it.
    bool at_x = (i == 0), at_y = (j == 0), at_z = (k == 0);
__MASK__

    // --- split-field recurrence (stepping._apply_pml_update:1905) ---------------
    // dsig/dsigu follow vec.hpp's cycle_direction and are the same triple on both
    // sides: target 0 takes (y, z), target 1 (z, x), target 2 (x, y).
    float km_x = kmx[i], si_x = sinvx[i];
    float km_y = kmy[j], si_y = sinvy[j];
    float km_z = kmz[k], si_z = sinvz[k];

    float p0 = u0[ii];
    float n0 = ((p0 * km_y) - curl0) * si_y;
    float v0 = (((f0[ii] * km_z) + n0) - p0) * si_z;

    float p1 = u1[ii];
    float n1 = ((p1 * km_z) - curl1) * si_z;
    float v1 = (((f1[ii] * km_x) + n1) - p1) * si_x;

    float p2 = u2[ii];
    float n2 = ((p2 * km_x) - curl2) * si_x;
    float v2 = (((f2[ii] * km_y) + n2) - p2) * si_y;

__AXIS_TAIL__

    u0[ii] = n0; u1[ii] = n1; u2[ii] = n2;
    f0[ii] = v0; f1[ii] = v1; f2[ii] = v2;
}
"""

#: The B side's prefix use: Bz's WHOLE curl is the radial forward difference of the
#: EXTENDED prefix (stepping.py:371-375). The extended volume has nr + 1 rows at the
#: same (phi, z) stride, so row i + 1 sits at ``ii + nyz``.
_B_PREFIX_LOADS = ("    // Bz's curl is REPLACED, so `b_x` above is dead on this side.\n"
                   "    float pfx_here = pfx[ii];\n"
                   "    float pfx_up   = pfx[ii + nyz];")
_B_CURL2 = ("    // ONE subtract, ONE multiply -- NOT the four-operand grouping,\n"
            "    // which is a different float32 number (stepping.py:347).\n"  # stepping.py live lines for the frozen device-text citation(s) in this string: 347->375
            "    float curl2 = dtdx * (pfx_up - pfx_here);")

#: The D side's prefix use: Dz's ``first`` operand and its shifted neighbour come from
#: the prefix while ``Dx``'s Hy operands stay RAW (stepping.py:447-455), which is why
#: both ``g1`` and ``pfx`` are bound. The ghost rule on the prefix is the SAME
#: METALLIC rule the raw operand takes.
_D_PREFIX_LOADS = ("    float pb   = pfx[ii];\n"
                   "    float pb_x = vx ? pfx[ox] : 0.0f;")
_D_CURL2 = "    float curl2 = dtdx * ((pb_x - pb) + (a - a_y));"

#: The B-side axis rule: Br on the axis is identically zero — a radial vector at
#: r = 0 points nowhere (stepping.py:661-662). Applied AFTER the recurrence, exactly
#: as ``_cylindrical_axis_zero_B`` runs after the term loop (:403-404).
_B_AXIS_TAIL = (
    "    // stepping._cylindrical_axis_zero_B:661-662 -- Bx[r = 0] = 0, after the\n"
    "    // recurrence. A ternary, not a branch: the certified ownership mask uses\n"
    "    // the same idiom and the same `at_x` predicate.\n"
    "    v0 = at_x ? 0.0f : v0;")

#: The D-side axis rules, in the array path's own order (stepping.py:586-587): the
#: on-axis Dz POST-ADD first, then Dp forced to zero.
#:
#: ``axis_coef`` IS HOST-ROUNDED and is not ``4.0f * dtdx``. The array path forms
#: ``4.0 * (grid.dt / grid.dx)`` in float64 and NumPy then casts that one scalar to
#: float32 before the multiply; binding the already-rounded word is the literal
#: transcription. (The two happen to agree because scaling by four is exact, and the
#: sibling test pins that they do rather than leaving it to luck.)
#:
#: ``g1`` IS THE RAW Hp, re-read from storage exactly as ``fields.get_H("Hy")`` does:
#: under an active PML ``update_H`` has not run yet this sub-step, so the stored H is
#: the same H^{n+1/2} the curls consumed.
_D_AXIS_TAIL = (
    "    // stepping._cylindrical_axis_zero_D:586-587. The Dz increment is a\n"
    "    // POST-ADD and deliberately NOT folded into the curl: the fold was\n"
    "    // MEASURED to break m = 0 under PML (Er 2.7e-01 / Hp 4.5e-01 against\n"
    "    // 3.6e-07), because Dz's dsig is R, whose sigma is zero on the axis row,\n"
    "    // so the plain post-add is the exact one (:546-551).\n"
    "    // Two separate roundings, as `A += s * B` performs them.\n"
    "    v2 = at_x ? (v2 + (axis_coef * g1[ii])) : v2;\n"
    "    v1 = at_x ? 0.0f : v1;")

_PREFIX_LOADS: Dict[str, str] = {"step_B": _B_PREFIX_LOADS, "step_D": _D_PREFIX_LOADS}
_CURL2: Dict[str, str] = {"step_B": _B_CURL2, "step_D": _D_CURL2}
_AXIS_TAIL: Dict[str, str] = {"step_B": _B_AXIS_TAIL, "step_D": _D_AXIS_TAIL}


def cylindrical_curl_source(sub_step: str, codes: Sequence[int],
                            contract: str = shaders.CONTRACT_OFF) -> str:
    """The specialised ``cyl_pml_curl_step`` source for one configuration.

    ``codes`` is the per-axis 0/1 PERIODIC/METALLIC triple, with the r axis pinned
    METALLIC — the caller resolves it from ``_boundary_kinds`` and maps ``'axis'``
    there, and this function REFUSES a periodic r axis by name rather than emitting a
    kernel whose far ghost wraps around the cylinder.
    """
    if sub_step not in SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
    codes = tuple(int(code) for code in codes)
    if len(codes) != 3:
        raise ValueError(f"codes must be a per-axis triple, got {codes!r}")
    if codes[0] != METALLIC:
        raise ValueError(
            f"the r axis must compile as METALLIC (code {METALLIC}), got {codes[0]!r}: "
            f"CYL_AXIS shares _shift_up's metallic zero ghost exactly "
            f"(stepping.py:1828-1830), and a PERIODIC r axis would wrap the far face "
            f"onto the axis row")
    backward = bool(SUB_STEPS[sub_step]["backward"])
    return templates.substitute(_CYL_CURL_TEMPLATE, {
        "__CONTRACT__": templates.contraction_pragma(contract),
        "__SHIFT__": "- 1" if backward else "+ 1",
        "__GHOST_X__": templates.ghost("x", codes[0], backward),
        "__GHOST_Y__": templates.ghost("y", codes[1], backward),
        "__GHOST_Z__": templates.ghost("z", codes[2], backward),
        "__PREFIX_LOADS__": _PREFIX_LOADS[sub_step],
        "__CURL2__": _CURL2[sub_step],
        "__MASK__": templates.ownership_mask(codes, backward),
        "__AXIS_TAIL__": _AXIS_TAIL[sub_step],
    })


def compile_cylindrical_curl(sub_step: str, codes: Sequence[int],
                             contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised cylindrical curl entry point for one configuration."""
    return compile_source(
        cylindrical_curl_source(sub_step, codes, contract)).cyl_pml_curl_step


def enumerate_cylindrical_real_sources(contract: str = shaders.CONTRACT_OFF
                                       ) -> Dict[str, str]:
    """Every shipped specialisation, keyed by a stable label.

    TEN sources: two scans (one per sub-step, differing only in the wall row) and
    eight curls. The curl enumeration pins ``codes[0] = METALLIC`` because the
    predicate admits no other r axis — enumerating the unreachable half would put
    strings in ``fingerprints.json`` that no configuration can launch, which reads as
    a claim that they are shipped.
    """
    out: Dict[str, str] = {}
    for sub_step in ("step_B", "step_D"):
        out[f"cyl_rderiv_prefix/{sub_step}"] = cylindrical_prefix_source(
            sub_step, contract)
        for cy in (PERIODIC, METALLIC):
            for cz in (PERIODIC, METALLIC):
                label = f"cyl_pml_curl_step/{sub_step}/{METALLIC}{cy}{cz}"
                out[label] = cylindrical_curl_source(
                    sub_step, (METALLIC, cy, cz), contract)
    return out


# ---------------------------------------------------------------------------
# Coverage — positive refusal enumeration
# ---------------------------------------------------------------------------
#
# The clause numbering mirrors ``coverage._grid_reasons`` so the two can be diffed.
# Clause 6 (CARTESIAN ONLY) is INVERTED: this product REQUIRES a Dcyl grid with the
# r axis declared, where every shipped kernel refuses both BY NAME. Clause 14 is
# NEW and is this family's internal split against a future |m| >= 1 product. Clause 4
# is RESTATED to admit 'axis' on the r axis and nowhere else. Everything else is
# KEPT.

#: Which axis of the engine's (r, phi, z) storage is the cylindrical axis. Named
#: rather than spelled ``0`` at five call sites, because "the radial axis is the
#: leading one" is a storage convention (stepping.py:1300-1301) and a reader should
#: be able to find every place this family depends on it.
RADIAL_AXIS = 0


def _cylindrical_real_grid_reasons(fields: Any, pml: Any, grid: Any) -> List[str]:
    """The clauses every Metal cylindrical m = 0 predicate shares."""
    reasons: List[str] = []

    # 1. The Metal backend, and the host array module the mirrors copy from.
    #    IMPORTED from this package's own coverage module: one backend clause for the
    #    whole package, so a family cannot quietly admit a run on a host whose
    #    subnormal policy the MPS executor refuses.
    reasons.extend(_metal_backend_reasons(grid))

    # 2. Real storage REQUIRED, and KEPT UN-INVERTED. This is THE clause that
    #    separates this family from the cylindrical COMPLEX one, in the SAME
    #    direction the shipped real families are separated from `complex_fields`.
    #    It is not implied by clause 14: a complex-storage m = 0 run is perfectly
    #    constructible (`force_complex_fields` is an independent switch), this kernel
    #    would read the wrong stride on it, and since 2026-09-04 the complex family
    #    carries it (its `M_ZERO` arm) — so the two families partition Dcyl by
    #    storage alone.
    if getattr(fields, "force_complex_fields", False):
        reasons.append("force_complex_fields=True: complex-storage cylindrical is "
                       "the cylindrical COMPLEX family's kernel at every m (its "
                       "M_ZERO arm carries m = 0), not this one")

    # 3. An absorber that actually absorbs (split-field family only).
    if pml is None or not getattr(pml, "is_active", False):
        reasons.append("no active PML layer (this product implements the "
                       "split-field path only)")

    # 4. RESTATED, and this is a WIDENING that must be spelled PER AXIS. On a Dcyl
    #    grid `_boundary_kinds` returns 'axis' for r (stepping.py:2192), which is
    #    outside COVERED_BOUNDARIES, so the shared clause would refuse every
    #    configuration this family exists for. 'axis' is admitted on the RADIAL axis
    #    only: an axis-kind phi or z would mean the grid disagrees with this family
    #    about which direction is radial, and admitting it globally would compile the
    #    wrong ghost onto two axes silently.
    kinds = _boundary_kinds(grid, pml if (pml is not None
                                          and getattr(pml, "is_active", False))
                            else None)
    if kinds is None:
        reasons.append("boundary kinds could not be resolved for this grid")
    else:
        for axis, kind in enumerate(kinds):
            allowed = (("periodic", "metallic", "axis") if axis == RADIAL_AXIS
                       else ("periodic", "metallic"))
            if kind not in allowed:
                reasons.append(f"axis {axis} boundary {kind!r} is outside {allowed}")
        if kinds[RADIAL_AXIS] != "axis":
            reasons.append(
                f"axis {RADIAL_AXIS} resolved to {kinds[RADIAL_AXIS]!r}, not 'axis': "
                f"this product's Bz/Dz substitution IS the cylindrical radial "
                f"derivative and is a wrong answer on a Cartesian leading axis")

    # 5. No mirror plane anywhere, KEPT — and it must be asked TWICE, which is not
    #    tidiness. `_boundary_kinds` puts is_axis AHEAD of is_mirrored
    #    (stepping.py:2191-2195), so a folded r axis reports CYL_AXIS and THE FOLD
    #    VANISHES SILENTLY from the boundary triple; and `_mirror_phases`
    #    (:2291-2308) puts (-1)**grid.m into the SAME SLOT the mirror phase occupies.
    #    The boundary kinds alone therefore cannot answer this question.
    if _call(grid, "has_symmetry", default=False):
        reasons.append("a mirror plane is active (symmetry folding is not carried "
                       "by this family, and a folded r axis is INVISIBLE in the "
                       "boundary kinds — is_axis outranks is_mirrored)")
    for axis in range(3):
        if _call(grid, "is_mirrored", axis, default=False):
            reasons.append(f"axis {axis} is folded by a mirror plane")

    # 6 (INVERTED). Cylindrical REQUIRED, where `coverage._grid_reasons` :176-182
    #    refuses both `cylindrical` and `is_axis` by name. This single inversion is
    #    what separates this family from ALL NINE carried ones, because every one of
    #    them either uses that clause list or restates it.
    if not getattr(grid, "cylindrical", False):
        reasons.append("grid.cylindrical is False: this product exists only for "
                       "Dcyl runs; a Cartesian run belongs to the certified plain "
                       "kernels")
    for axis in range(3):
        declared = bool(_call(grid, "is_axis", axis, default=False))
        if axis == RADIAL_AXIS and not declared:
            reasons.append(f"axis {RADIAL_AXIS} is not the cylindrical r = 0 axis")
        if axis != RADIAL_AXIS and declared:
            reasons.append(f"axis {axis} is declared the cylindrical r = 0 axis; "
                           f"this product's radial substitution is on axis "
                           f"{RADIAL_AXIS}")

    # 6a (THIS FAMILY'S OWN). The phi axis carries ONE cell on a Dcyl grid, which is
    #    what makes its finite difference an exact zero rather than a boundary. The
    #    clause is spelled rather than inferred from `Grid`'s own guard: this
    #    package's doctrine forbids reading a refusal off another module.
    shape = tuple(getattr(grid, "shape", ()))
    if len(shape) == 3 and int(shape[1]) != 1:
        reasons.append(f"the phi axis carries {shape[1]} cells; a Dcyl grid has one, "
                       f"and this kernel's transverse operands assume the invariant "
                       f"wrap that gives")

    # 7. k = 0, KEPT. `Grid` refuses a k on r or phi, and a z Bloch phase would need
    #    complex storage, which clause 2 already refuses — but the pairing is named
    #    rather than left to be implied by another clause.
    if getattr(grid, "has_bloch", False):
        reasons.append(f"nonzero k_point {getattr(grid, 'k_point', None)!r}")
    k_point = getattr(grid, "k_point", (0.0, 0.0, 0.0))
    if any(float(component) != 0.0 for component in k_point):
        reasons.append(f"k_point {tuple(k_point)!r} is not exactly zero")

    # 10. No instantaneous nonlinearity.
    if getattr(fields, "has_nonlinearity", False):
        reasons.append("chi2/chi3 is installed (the Pade constitutive factor is "
                       "not carried; that family is a separate tranche)")

    # 11/12. BFAST and beta, KEPT. Cylindrical + BFAST is refused by `Grid` itself
    #        (grid.py:706-742) and the clause is kept anyway, for the reason
    #        `bfast_curl` keeps its own cylindrical clause: a refusal inferred from
    #        another module's guard is not this predicate's answer.
    if getattr(grid, "bfast_active", False):
        reasons.append("BFAST is active (the second additive curl term is not "
                       "carried; the grid itself refuses bfast on Dcyl, "
                       "grid.py:706-742)")
    if float(getattr(grid, "beta", 0.0)) != 0.0:
        reasons.append(f"special_kz beta={getattr(grid, 'beta', 0.0)!r} is nonzero")

    # 14 (NEW — this family's internal split). m == 0 REQUIRED, refused by name above
    #    it. Clause 2 alone is NOT sufficient in both directions: `stepping`
    #    :704-709 makes complex storage a hard REQUIREMENT of the i*m/r term, so an
    #    |m| >= 1 real run cannot exist on the engine — but "cannot exist" is an
    #    ENGINE COUPLING, not an inverted clause, and this project has already
    #    recorded that residuals separated only by a coupling are PLANTED and
    #    measured, never assumed. So the m clause is spelled, and the |m| >= 1 family
    #    will spell its inversion.
    m = getattr(grid, "m", 0)
    try:
        nonzero = int(m) != 0
    except Exception as exc:  # noqa: BLE001 - an unreadable m is not a zero one
        reasons.append(f"grid.m could not be read as an integer: {exc!r}")
    else:
        if nonzero:
            reasons.append(
                f"grid.m = {m!r}: |m| >= 1 adds the i*m/r coupling on two targets, "
                f"the |m| = 1 axis-row curl REPLACEMENT on both sides and the "
                f"|m| >= 2 six-component near-axis zeroing, and it forces complex64 "
                f"storage (stepping.py:731-736) — that is the |m| >= 1 family's "
                f"kernel and this one refuses it by name rather than by the absence "
                f"of an i*m/r term")

    # 9c. Stored E — the invariant behind admitting dispersion for the curl.
    if not getattr(fields, "stores_E", False):
        reasons.append("E is recomputed from D rather than stored")

    return reasons


def _curl_conductivity_reasons(fields: Any) -> List[str]:
    """Conductivity refused on ALL six curl targets, BOTH directions.

    This kernel transcribes ``_apply_pml_update`` (stepping.py:1952), the PLAIN
    split-field recurrence; a conductivity routes ``_apply_curl`` to the three-stage
    ``f_cond`` recurrence instead (:487-505). THIS PACKAGE SHIPS NO CONDUCTIVE
    PRODUCT AT ALL, so the refusal is unconditional rather than a hand-off.

    A missing or non-callable reader is refused OUTRIGHT: inferring "no conductivity"
    from the absence of ``condfac_for`` is admission by attribute absence.
    """
    reasons: List[str] = []
    reader = getattr(fields, "condfac_for", None)
    if not callable(reader):
        reasons.append("fields does not expose condfac_for; an unreadable "
                       "conductivity table is not an absent one")
        return reasons
    for target in CURL_TARGETS:
        try:
            conductive = reader(target) is not None
        except Exception as exc:  # noqa: BLE001 - unreadable means not covered
            reasons.append(f"condfac_for({target!r}) raised {exc!r}")
            continue
        if conductive:
            reasons.append(
                f"a conductivity is installed on {target}: this kernel transcribes "
                f"the plain split-field recurrence only, and this package ships no "
                f"conductive product")
    return reasons


def cylindrical_real_curl_coverage(fields: Any, pml: Any, sub_step: str,
                                   residency: Any = None) -> Coverage:
    """May the Metal cylindrical m = 0 curl kernel step this (fields, pml, sub_step)?"""
    if sub_step not in CURL_SUB_STEPS:
        raise ValueError(
            f"sub_step must be one of {tuple(CURL_SUB_STEPS)}, got {sub_step!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    reasons = _cylindrical_real_grid_reasons(fields, pml, grid)
    reasons.extend(_residency_declaration_reasons(residency))
    reasons.extend(_curl_conductivity_reasons(fields))

    # 9a/9b. A registered susceptibility must be one this package understands.
    reasons.extend(_susceptibility_reasons(fields))

    # 13. The named sub-step's targets, auxiliaries and sources.
    spec = SUB_STEPS[sub_step]
    names = (tuple(spec["targets"])
             + tuple("fu_" + name for name in spec["targets"])
             + tuple(spec["sources"]))
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")

    # 14b. The D side's axis POST-ADD re-reads Hp from storage exactly as
    #      `fields.get_H("Hy")` does (stepping.py:586). Under an active PML that IS
    #      the stored array, and clause 3 already requires one — but the read is
    #      named here so a future no-PML arm cannot inherit the admission silently.
    if sub_step == "step_D" and getattr(fields, "Hy", None) is None:
        reasons.append("Hy is not stored: the m = 0 axis rule adds "
                       "(4*Courant)*Hp[r=0] to Dz and reads the STORED H "
                       "(stepping.py:586)")

    # 15. Layout: float32, C-contiguous, grid.shape.
    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(_layout_reasons(fields, shape, names))
    if pml is not None and getattr(pml, "is_active", False) and len(shape) == 3:
        reasons.extend(_coefficient_reasons(pml, shape, ("kms", "sinv"), ("", "_h")))

    return Coverage(not reasons, tuple(reasons))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class CylindricalRealCurlPlan:
    """One cylindrical m = 0 curl sub-step, as TWO launches: the scan, then the curl.

    DELIBERATELY A SIBLING OF :class:`.plans.KernelPlan` RATHER THAN A SUBCLASS, for
    the reason :class:`.symmetry.MirrorGhostFillPlan` is one: that base's whole
    contract is that ``run`` unpacks ONE argument tuple and calls ONE function, and
    one implementation of that is what keeps the launch path uniform across the
    family tree. This plan cannot honour it — the curl READS the scan's output, so
    the two are ordered and cannot be one dispatch — and the divergence is stated
    here rather than hidden behind an overridden ``run``.

    THE PREFIX MIRROR IS A PLAN-OWNED SCRATCH VOLUME, not an engine one, and this
    family is the first here to own one at all. Two consequences, both measured
    rather than assumed:

    * it is registered NON-CONSTANT even though nothing on the host reads it,
      because ``constant`` means "the device never writes this" and
      ``Residency.verify`` — the residency invariant's own check — would then report
      it as permanently stale. A mirror the verifier cannot interpret is worse than
      one extra sync;
    * it is registered through :func:`scratch_mirror`, which REUSES the array
      already bound under the name. Allocating a fresh one in the builder made a
      second ``plan_cylindrical_real_curl`` for the same sub-step on one residency
      raise ``mirror ... is already bound to a different host array`` — a refusal
      that is right for engine volumes and wrong here, since the prefix is fully
      overwritten by the scan before the curl reads it and the two row vectors are
      grid invariants.

    ``launches`` COUNTS KERNEL LAUNCHES, NOT ``run`` CALLS, and it is two per cycle.
    A gate asserting the exact count per cycle is what stops a slot passing by not
    executing; :attr:`prefix_launches` and :attr:`curl_launches` are carried
    separately so "the scan ran and the curl did not" is distinguishable from "the
    plan ran once".
    """

    __slots__ = ("sub_step", "shape", "n_elem", "scan_shape", "n_cols", "dtdx",
                 "axis_coef", "backward", "bc", "residency", "volumes",
                 "prefix_host", "launches", "prefix_launches", "curl_launches",
                 "_prefix_functions", "_prefix_args",
                 "_curl_functions", "_curl_args")

    family = "cylindrical m = 0 PML curl"

    #: This plan launches kernels; the composer's planned/null split reads it.
    performs_device_work = True

    #: TWO, and this is the one plan in the tree that is not one. ``run`` dispatches
    #: the radial scan and then the curl that reads it, so a whole-step gate that
    #: asserted one launch per cycle on every slot would fail a correct composition
    #: here. Declared with the same name :class:`.plans.KernelPlan` declares it under
    #: — this class is deliberately not a subclass of that base (see the docstring),
    #: so the attribute is restated rather than inherited.
    launches_per_run = 2

    REPR_FIELDS = ("sub_step", "shape", "bc", "scan_shape")

    def __init__(self, sub_step: str, shape, scan_shape, dtdx: float,
                 axis_coef: float, bc, residency: Residency,
                 prefix_host: Any, prefix_args: Sequence[Any],
                 curl_args: Sequence[Any],
                 prefix_functions: Dict[str, Any],
                 curl_functions: Dict[str, Any],
                 volumes: Sequence[str]) -> None:
        if sub_step not in SUB_STEPS:
            raise ValueError(
                f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
        self.sub_step = sub_step
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.scan_shape = tuple(int(n) for n in scan_shape)
        self.n_cols = self.shape[1] * self.shape[2]
        # float(): the array path multiplies a float32 volume by a Python float,
        # which NumPy casts to float32 before the multiply; Metal binds a Python
        # float into `constant float&` the same way, so the two scalars carry the
        # same bits.
        self.dtdx = float(dtdx)
        self.axis_coef = float(axis_coef)
        self.backward = bool(SUB_STEPS[sub_step]["backward"])
        self.bc = tuple(int(code) for code in bc)
        self.residency = residency
        self.volumes = tuple(volumes)
        self.prefix_host = prefix_host
        self._prefix_functions = dict(prefix_functions)
        self._curl_functions = dict(curl_functions)
        self._prefix_args = tuple(prefix_args)
        self._curl_args = tuple(curl_args)
        self.launches = 0
        self.prefix_launches = 0
        self.curl_launches = 0

    @property
    def variants(self) -> Tuple[str, ...]:
        """The contraction modes BOTH kernels were built with.

        The intersection rather than either one alone: a plan that held a ``fast``
        scan and no ``fast`` curl would report a variant it cannot launch, and the
        gate's guard leg would then measure the shipped curl while believing it had
        removed the guard.
        """
        return tuple(sorted(set(self._prefix_functions) & set(self._curl_functions)))

    def _function(self, table: Dict[str, Any], mode: str, which: str) -> Any:
        function = table.get(mode)
        if function is None:
            raise KeyError(
                f"this plan holds no {mode!r} {which} variant (it was built with "
                f"{self.variants}); build it with contract_variants={(mode,)} "
                f"rather than launching the pinned one")
        return function

    def run_prefix(self, contract: Optional[str] = None) -> None:
        """The radial scan alone — ``cylindrical_rderiv_prefix`` on the device."""
        mode = shaders.CONTRACT_OFF if contract is None else contract
        function = self._function(self._prefix_functions, mode, "scan")
        self.launches += 1
        self.prefix_launches += 1
        function(*self._prefix_args)

    def run_curl(self, contract: Optional[str] = None) -> None:
        """The curl alone. Reads the scan's output, so it is never run first."""
        mode = shaders.CONTRACT_OFF if contract is None else contract
        function = self._function(self._curl_functions, mode, "curl")
        self.launches += 1
        self.curl_launches += 1
        function(*self._curl_args)

    def run(self, contract: Optional[str] = None) -> None:
        """The sub-step: scan, then curl. In place.

        THE ORDER IS THE CONTRACT. The curl reads the prefix the scan writes, and the
        prefix is a function of THIS sub-step's source volume — so a scan hoisted out
        of the loop, or run after the curl, feeds the curl the PREVIOUS timestep's
        radial derivative. That is a smooth, plausible, entirely wrong field rather
        than an error, which is why the two launches live behind one ``run`` and the
        separate entry points exist only for a gate that wants to measure one of them.
        """
        self.run_prefix(contract)
        self.run_curl(contract)

    def describe(self) -> str:
        parts = ", ".join(f"{name}={getattr(self, name)!r}"
                          for name in self.REPR_FIELDS)
        return f"CylindricalRealCurlPlan({parts}, variants={self.variants})"

    def __repr__(self) -> str:
        return self.describe()


def _prefix_functions(sub_step: str, contract_variants: Sequence[str]
                      ) -> Dict[str, Any]:
    return {mode: compile_cylindrical_prefix(sub_step, mode)
            for mode in contract_variants}


def _curl_functions(sub_step: str, codes, contract_variants: Sequence[str]
                    ) -> Dict[str, Any]:
    return {mode: compile_cylindrical_curl(sub_step, codes, mode)
            for mode in contract_variants}


def scratch_mirror(residency: Residency, name: str, build: Any,
                   constant: bool = False) -> Tuple[Any, Any]:
    """Register a PLAN-OWNED scratch volume, reusing the array already bound.

    Every volume the earlier families mirror belongs to the ENGINE, so the builder
    always has the array in hand and one name always means one array. This family
    ALLOCATES its three — the radial prefix and the two row vectors — inside the
    builder, so building the same sub-step's plan twice against one residency handed
    ``mirror`` two different arrays under one name and it refused, correctly by its
    own contract and wrongly for this case: the prefix is fully overwritten by the
    scan before the curl reads it, and the row vectors are grid invariants, so two
    plans SHOULD share them.

    ``build`` is called ONLY on the first registration, so the second build neither
    allocates nor re-derives the ladder.
    """
    existing = residency.host(name)
    host = build() if existing is None else existing
    return host, residency.mirror(name, host, constant=constant)


def boundary_codes(kinds: Sequence[str]) -> Tuple[int, int, int]:
    """``_boundary_kinds``' strings as this family's specialisation triple.

    'axis' MAPS TO METALLIC, and that mapping is the whole reason this family needs
    no ghost code of its own — see the module docstring for the measurement. It is a
    named function rather than an inline comprehension so both plan builders and the
    sibling tests read the SAME mapping.
    """
    return tuple(METALLIC if kind in ("metallic", "axis") else PERIODIC
                 for kind in kinds)  # type: ignore[return-value]


def plan_cylindrical_real_curl(fields: Any, pml: Any, sub_step: str,
                               residency: Optional[Residency] = None,
                               contract_variants: Sequence[str] = (
                                   shaders.CONTRACT_OFF,),
                               ) -> Optional[CylindricalRealCurlPlan]:
    """Build a cylindrical m = 0 curl plan from the engine's own objects, or None.

    None is the only refusal: a configuration this kernel does not carry must fall
    back to the array path, never raise into a caller that would otherwise have
    stepped correctly.
    """
    if sub_step not in SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
    if not cylindrical_real_curl_coverage(fields, pml, sub_step, residency).covered:
        return None
    import numpy as np  # noqa: PLC0415
    from ..stepping import _boundary_kinds as resolve  # noqa: PLC0415

    spec = SUB_STEPS[sub_step]
    grid = fields.grid
    codes = boundary_codes(resolve(grid, pml))
    shape = tuple(int(n) for n in grid.shape)
    scan_shape = (scan_rows(sub_step, shape[0]), shape[1], shape[2])

    targets = [residency.mirror(n, getattr(fields, n)) for n in spec["targets"]]
    auxiliaries = [residency.mirror("fu_" + n, getattr(fields, "fu_" + n))
                   for n in spec["targets"]]
    sources = [residency.mirror(n, getattr(fields, n)) for n in spec["sources"]]
    coefficients = [
        residency.mirror(f"pml:{stem}_{axis}{spec['suffix']}",
                         getattr(pml, f"{stem}_{axis}{spec['suffix']}"),
                         constant=True)
        for axis in "xyz" for stem in ("kms", "sinv")]

    # The scan reads the sub-step's PHI source -- Ep on the B side and Hp on the D
    # side, both of which are `sources[1]`. Named through the term table rather than
    # by index so a table change cannot silently repoint it.
    scan_source_name = spec["sources"][1]
    scan_source = residency.mirror(scan_source_name,
                                   getattr(fields, scan_source_name))

    prefix_host, prefix = scratch_mirror(
        residency, f"cyl_pfx:{sub_step}",
        lambda: np.zeros(scan_shape, dtype=np.float32))
    _, weight_vector = scratch_mirror(
        residency, f"cyl_weights:{sub_step}",
        lambda: prefix_row_vectors(sub_step, shape[0])[0], constant=True)
    _, divisor_vector = scratch_mirror(
        residency, f"cyl_divisor:{sub_step}",
        lambda: prefix_row_vectors(sub_step, shape[0])[1], constant=True)

    dtdx = grid.dt / grid.dx
    # stepping.py:586 forms `4.0 * (grid.dt / grid.dx)` in float64 and NumPy rounds
    # that ONE scalar to float32 before the multiply. Zero on the B side, which binds
    # the slot without reading it -- one binding layout for both sub-steps, the same
    # discipline `constitutive_source` follows for its inverse-epsilon buffers.
    axis_coef = (4.0 * dtdx) if sub_step == "step_D" else 0.0
    n_elem = shape[0] * shape[1] * shape[2]
    volumes = (tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
               + tuple(spec["sources"]))
    return CylindricalRealCurlPlan(
        sub_step, shape, scan_shape, dtdx, axis_coef, codes, residency,
        prefix_host,
        (prefix, scan_source, weight_vector, divisor_vector,
         scan_shape[0], shape[1], shape[2], shape[1] * shape[2]),
        tuple(targets) + tuple(auxiliaries) + tuple(sources) + (prefix,)
        + tuple(coefficients)
        + (shape[0], shape[1], shape[2], n_elem, float(dtdx), float(axis_coef)),
        _prefix_functions(sub_step, contract_variants),
        _curl_functions(sub_step, codes, contract_variants),
        volumes)


def plan_cylindrical_real_curl_from_arrays(
        sub_step: str, arrays: Dict[str, Any], flat: Dict[str, Any], codes,
        dtdx: float, residency: Residency,
        prefix_function: Optional[Dict[str, Any]] = None,
        curl_function: Optional[Dict[str, Any]] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        axis_coef: Optional[float] = None) -> CylindricalRealCurlPlan:
    """Build a plan from bare host arrays — the gate's route.

    No predicate runs: the caller is a harness that constructed the configuration
    deliberately, including the deliberately wrong ones. ``prefix_function`` and
    ``curl_function`` are the mutation seams, and they are SEPARATE so a leg can
    plant a defect in one kernel while the other stays shipped — dropping them is not
    a silent slowdown but a silent DISARMING, since every mutation leg would then
    launch the shipped kernels and report the defect as uncaught.
    """
    import numpy as np  # noqa: PLC0415

    spec = SUB_STEPS[sub_step]
    shape = tuple(int(n) for n in arrays[spec["targets"][0]].shape)
    scan_shape = (scan_rows(sub_step, shape[0]), shape[1], shape[2])

    targets = [residency.mirror(n, arrays[n]) for n in spec["targets"]]
    auxiliaries = [residency.mirror("fu_" + n, arrays["fu_" + n])
                   for n in spec["targets"]]
    sources = [residency.mirror(n, arrays[n]) for n in spec["sources"]]
    coefficients = [residency.mirror(f"pml:{stem}_{axis}:{sub_step}",
                                     flat[f"{stem}_{axis}"], constant=True)
                    for axis in "xyz" for stem in ("kms", "sinv")]

    scan_source_name = spec["sources"][1]
    prefix_host, prefix = scratch_mirror(
        residency, f"cyl_pfx:{sub_step}",
        lambda: np.zeros(scan_shape, dtype=np.float32))
    _, weight_vector = scratch_mirror(
        residency, f"cyl_weights:{sub_step}",
        lambda: prefix_row_vectors(sub_step, shape[0])[0], constant=True)
    _, divisor_vector = scratch_mirror(
        residency, f"cyl_divisor:{sub_step}",
        lambda: prefix_row_vectors(sub_step, shape[0])[1], constant=True)
    resolved_coef = ((4.0 * float(dtdx)) if sub_step == "step_D" else 0.0
                     ) if axis_coef is None else float(axis_coef)
    n_elem = shape[0] * shape[1] * shape[2]
    volumes = (tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
               + tuple(spec["sources"]))
    return CylindricalRealCurlPlan(
        sub_step, shape, scan_shape, dtdx, resolved_coef, codes, residency,
        prefix_host,
        (prefix, residency.mirror(scan_source_name, arrays[scan_source_name]),
         weight_vector, divisor_vector,
         scan_shape[0], shape[1], shape[2], shape[1] * shape[2]),
        tuple(targets) + tuple(auxiliaries) + tuple(sources) + (prefix,)
        + tuple(coefficients)
        + (shape[0], shape[1], shape[2], n_elem, float(dtdx), resolved_coef),
        prefix_function if prefix_function is not None
        else _prefix_functions(sub_step, contract_variants),
        curl_function if curl_function is not None
        else _curl_functions(sub_step, codes, contract_variants),
        volumes)


# ---------------------------------------------------------------------------
# The constitutive companion — no new kernel, a restated predicate
# ---------------------------------------------------------------------------
#
# WHY THIS IS SOUND, and it is a READ of stepping.py rather than an analogy.
# `update_H` (S:907) and `update_E` (S:926) contain NO CYLINDRICAL BRANCH AT ALL:
# measured on this tree by reading their own source, the words "cylindrical",
# "is_axis" and "grid.m" do not appear in either. They are element-wise with per-axis
# coefficient tables indexed on the component's OWN axis, and Dcyl changes nothing
# about that except that axis 1 has n = 1. So the ARITHMETIC of the constitutive pair
# on a Dcyl run is byte-for-byte the arithmetic the certified constitutive kernel
# already reproduces, and only the ADMISSION has to change.
#
# THE TRITON TRACK MEASURED THE SAME CLAIM ON ITS OWN KERNEL — the existing
# `kernels.constitutive_step` bytewise identical to `stepping.update_H`/`update_E` on
# a Dcyl grid, 4/4 (`cylindrical_triton.py:84-86`) — which is corroboration on
# another backend and NOT this port's evidence. This one's evidence is its own gate.
#
# THIS IS THE SAME MOVE `bfast_curl.bfast_run_constitutive_coverage` MAKES, for the
# same reason, and it is the cheapest coverage on either track: no kernel, no new
# specialisation. What it buys is WHOLE-STEP composition — without it a Dcyl run
# takes its curls on the device and its constitutive pair on the array path, which
# forces a sync per sub-step and makes the residency verdict refuse. Six of this
# family's twelve slots are this predicate.

def cylindrical_real_constitutive_coverage(fields: Any, pml: Any, side: str,
                                           residency: Any = None) -> Coverage:
    """May the CERTIFIED Metal constitutive kernel step a Dcyl m = 0 run?

    :func:`_cylindrical_real_grid_reasons`' clause set with nothing added and nothing
    removed — clause 6 is already inverted there — plus the constitutive side's own
    clauses, exactly as the certified predicate spells them. The conductivity clause
    is deliberately NOT applied: a conductivity changes the CURL recurrence
    (``_apply_curl`` reads ``condfac_for`` and nothing else does), so refusing it here
    would refuse a configuration this sub-step steps correctly.
    """
    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, "
                         f"got {side!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    spec = CONSTITUTIVE_SIDES[side]
    reasons = _cylindrical_real_grid_reasons(fields, pml, grid)
    reasons.extend(_residency_declaration_reasons(residency))
    reasons.extend(_susceptibility_reasons(fields))

    if side == "E":
        if getattr(fields, "has_polarizations", False) or (
                getattr(fields, "polarizations", ()) or ()):
            reasons.append(
                "a susceptibility is registered: update_E's source is (D - sum P), "
                "not D (fields.py:1096-1105) — that configuration belongs to the "
                "ADE kernel")
        if getattr(fields, "has_offdiagonal_epsilon", False):
            reasons.append(
                "an off-diagonal chi1inv row is installed (the row product reads "
                "neighbours; this sub-step is element-wise)")

    names = tuple(spec["targets"]) + tuple(spec["aux"]) + tuple(spec["sources"])
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")

    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(_layout_reasons(fields, shape, names))
    if side == "E" and len(shape) == 3:
        reasons.extend(_inverse_epsilon_reasons(fields, shape))
    if pml is not None and getattr(pml, "is_active", False) and len(shape) == 3:
        suffix = ("_h",) if spec["half_integer"] else ("",)
        reasons.extend(_coefficient_reasons(pml, shape, ("kps", "kms"), suffix))

    return Coverage(not reasons, tuple(reasons))


def plan_cylindrical_real_constitutive(fields: Any, pml: Any, side: str,
                                       residency: Any = None,
                                       contract_variants: Sequence[str] = (
                                           templates.CONTRACT_OFF,)) -> Any:
    """A CERTIFIED constitutive plan for a Dcyl m = 0 run, or None.

    The arithmetic, the source and the plan class are the certified ones; only the
    ADMISSION is this module's. Building the plan through the certified builder —
    rather than re-deriving one here — is what makes "same kernel" a fact instead of
    a claim: a divergence would have to come from the predicate, which is the only
    thing this family contributes.
    """
    from .launch import (  # noqa: PLC0415 - avoids a circular import at module load
        ConstitutivePlan, _constitutive_functions,
    )

    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, "
                         f"got {side!r}")
    if not cylindrical_real_constitutive_coverage(fields, pml, side,
                                                  residency).covered:
        return None
    spec = CONSTITUTIVE_SIDES[side]
    suffix = "_h" if spec["half_integer"] else ""
    targets = [residency.mirror(n, getattr(fields, n)) for n in spec["targets"]]
    auxiliaries = [residency.mirror(n, getattr(fields, n)) for n in spec["aux"]]
    sources = [residency.mirror(n, getattr(fields, n)) for n in spec["sources"]]
    inverse_epsilon = (
        [residency.mirror("inv_eps_" + n, fields.inverse_epsilon_for(n),
                          constant=True) for n in spec["targets"]]
        if side == "E" else None)
    coefficients = [
        residency.mirror(f"pml:{stem}_{axis}{suffix}",
                         getattr(pml, f"{stem}_{axis}{suffix}"), constant=True)
        for axis in "xyz" for stem in ("kps", "kms")]
    volumes = tuple(spec["targets"]) + tuple(spec["aux"]) + tuple(spec["sources"])
    return ConstitutivePlan(
        side, fields.grid.shape, residency, targets, auxiliaries, sources,
        inverse_epsilon, coefficients,
        _constitutive_functions(side, contract_variants), volumes)


# ---------------------------------------------------------------------------
# WIRING — two curl arms and the certified constitutive pair
# ---------------------------------------------------------------------------
#
# WHAT SEPARATES THEM, stated because a new family must state its inversion rather
# than inherit disjointness. `coverage._grid_reasons` clause 6 refuses `cylindrical`
# AND `is_axis` BY NAME (:176-182), and every carried family either uses that list
# (`pml_curl`, `constitutive`, `no_pml_constitutive`, `offdiag_update_e`) or restates
# it verbatim (`complex_fields`, `special_kz`, `bfast_curl`, `symmetry`,
# `folded_complex`, `folded_offdiag_update_e`). Clause 6 here is INVERTED — a Dcyl
# grid with the r axis declared is REQUIRED — so the separation from all nine is
# total, and it is one clause rather than a coincidence of several.
#
# THE ONE FAMILY THIS DOES NOT SEPARATE FROM BY CLAUSE 6 IS THE CYLINDRICAL COMPLEX
# ONE, and clause 2 (real storage) is what does: this family REQUIRES float32 and
# that one REQUIRES complex64, inverted in both directions. Clause 14 (m == 0 here)
# was a SECOND separation while the complex family carried |m| >= 1 only; since
# 2026-09-04 that family's M_ZERO arm carries complex64 at m = 0 too, so clause 14
# now separates this family from the ENGINE's own refusal (a float32 run at |m| >= 1
# cannot exist, stepping.py:731-736) rather than from a sibling product, and the two
# families partition Dcyl by storage alone.
#
# ON `update_H`/`update_E` THE ADMISSION IS NOT AN AMBIGUITY EITHER, and the
# reasoning is the same clause: `constitutive` and `no_pml_constitutive` are the only
# other arms there, the first refuses a Dcyl run through the shared clause 6, and the
# second requires an INACTIVE absorber where clause 3 here requires an active one —
# two independent separations, not one. The composition sweep MEASURES that per
# registered arm rather than assuming it.
#
# `update_P` still carries no Metal arm at all and this family does not change that:
# a Dcyl run with a registered susceptibility is refused on the E side by name, so
# nothing here reaches the pole recurrence.


def _curl_arm_coverage(context: Any, slot: str) -> Coverage:
    return cylindrical_real_curl_coverage(context.fields, context.pml, slot,
                                          context.residency)


def _curl_arm_plan(context: Any, slot: str) -> Optional[CylindricalRealCurlPlan]:
    return plan_cylindrical_real_curl(context.fields, context.pml, slot,
                                      context.residency, context.contract_variants)


#: Which constitutive side each slot names, so one pair of callables serves both.
_CONSTITUTIVE_SLOT_SIDES: Dict[str, str] = {"update_H": "H", "update_E": "E"}


def _constitutive_arm_coverage(context: Any, slot: str) -> Coverage:
    return cylindrical_real_constitutive_coverage(
        context.fields, context.pml, _CONSTITUTIVE_SLOT_SIDES[slot],
        context.residency)


def _constitutive_arm_plan(context: Any, slot: str) -> Any:
    return plan_cylindrical_real_constitutive(
        context.fields, context.pml, _CONSTITUTIVE_SLOT_SIDES[slot],
        context.residency, context.contract_variants)


def register_arms() -> Tuple[Any, ...]:
    """This family's four arms: two curls and the certified constitutive pair."""
    from . import arms  # noqa: PLC0415 - deferred: `arms` imports nothing of ours

    registered = [
        arms.register(family=FAMILY, slot=slot, label="cylindrical m=0",
                      coverage=_curl_arm_coverage, plan=_curl_arm_plan,
                      prefix="cylindrical m=0: ",
                      noun="cylindrical m = 0 PML curl", wired=True)
        for slot in ("step_B", "step_D")]
    registered.extend(
        arms.register(family=FAMILY, slot=slot, label="cylindrical m=0",
                      coverage=_constitutive_arm_coverage,
                      plan=_constitutive_arm_plan,
                      prefix="cylindrical m=0: ",
                      noun="cylindrical m = 0 constitutive", wired=True)
        for slot in _CONSTITUTIVE_SLOT_SIDES)
    return tuple(registered)


#: Registered ON IMPORT, once — the registry refuses a duplicate by design.
ARMS: Tuple[Any, ...] = register_arms()
