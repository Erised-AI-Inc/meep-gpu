"""The CYLINDRICAL COMPLEX electric seam in one launch: Dcyl ``step_D`` welded into
``update_E``.

The complex-storage Dcyl arm's D->E product (|m| >= 1 since 2026-08-20; m = 0 under
complex64 storage since 2026-09-04, when the curl half grew its ``M_ZERO`` arm and
this weld carried it — the corpus row is examples:dipole_in_vacuum_cyl_off_axis.py),
and the twin of
:mod:`.cylindrical_fused_magnetic_pair`. The two halves are the certified tranches'
own kernels — :func:`.cylindrical_complex.cyl_complex_pml_curl_step` on ``step_D``
and :func:`.complex_fields.bloch_constitutive_step`'s ``SCALE = 1`` arm on the E
side — and the weld is the one substitution
:func:`.kernels.fused_curl_constitutive_D` makes on the real path, done on both
float32 word planes.

EVERY ARITHMETIC LINE BELOW IS A VERBATIM COPY, with a comment naming the source it
came from. The curl half is ``cyl_complex_pml_curl_step``'s body character for
character, BACKWARD arms and all; the constitutive half is
:mod:`.complex_fused_electric_pair`'s, which is itself
``bloch_constitutive_step``'s ``SCALE = 1`` arm. A reader can diff this body
against those two and see that nothing moved, and the gate's transcription leg is
what turns that from a claim into a measurement. If you find yourself deriving
here, you have taken a wrong turn.

===========================================================================
WHY THIS CELL — measured from the fusion matrix, and it is not a guess
===========================================================================

``parity/meep_gpu/results/fusion_matrix_triton_2026-08-30_electric`` ranks the
fourteen (curl, constitutive) cells the 186-row corpus drives that no fused
product covers, by rows reaching them::

    D->E  16  ceiling 16   cylindrical complex PML -> cylindrical complex   NOT BUILT

RANK 1 on the board after :mod:`.complex_fused_electric_pair` closed the tie, and
the LARGEST remaining cell on any backend's board. It is also a cell only this
backend can take: the Metal board scores the same conjunction under "cannot be
bound at any sharing" — its argument-table ceiling is 31 bindings and this pair
needs more — while Triton has no binding ceiling at all.

Re-measured on the census this file prices against
(``predicate_coverage_triton_2026-08-28``, read on ``covered_modulo_backend``
because that lift ran on NumPy):

    admit the cylindrical complex curl at step_D              16 rows
    admit the cylindrical complex constitutive at update_E    16 rows
    admit BOTH halves                                         16 rows
    ... and declare NO ELECTRIC source                         0 rows

ZERO. That last line is the whole difference between this product and its magnetic
twin, and it is why the twin could ship with ``CARRIES_DEPOSIT_REPAIR = False`` and
this one cannot: all sixteen Dcyl rows declare electric sources — ``('D',)`` on
fourteen, ``('D', 'D')`` on ``cylinder_cross_section.py`` and ``zone_plate.py``.
Under the source clause as the twin states it this arm would admit NOTHING AT ALL.
The board's ``ceiling_no_in_seam_source`` of 16 is the count with the deposit
repair carried, which is what :data:`CARRIES_DEPOSIT_REPAIR` below declares and
what ``deposit_repair.repairable`` decides per run.

The number is an UPPER bound for the same reason every census-priced number in this
tree is: several clauses of the shipped predicate are not evaluable from a recorded
configuration block, and each can only remove rows.

===========================================================================
THE FIVE PASSES IN THE SEAM
===========================================================================

The driver runs five passes between the two halves (driver.py:3292-3304)::

    step_D -> electric sources -> fill_symmetry_bc_D -> zero_metal_D
           -> fill_folded_far_ghosts_D -> update_E

* **the electric sources — CARRIED, through the deposit repair.** ``driver.step``
  injects them between the two halves (driver.py:3296-3299), so a bare fused pair
  would consume a pre-injection ``D``. The pair does not run bare: the plan is
  bracketed by :class:`~..deposit_repair.LeadingRepairPlan` /
  :class:`~..deposit_repair.TrailingRepairPlan`, and the clause below passes
  ``carries_repair=CARRIES_DEPOSIT_REPAIR`` into
  :func:`~..deposit_repair.seam_source_reasons` so that
  :func:`~..deposit_repair.repairable` decides — refusing BY NAME every seam it
  cannot invert. Ignorance is still never an empty set: ``Fields`` does not hold
  the source list, so an undeclared ``sources`` remains a REFUSAL and not an
  assumed ``()``.
* **``fill_symmetry_bc_D`` — PROVABLY A NO-OP HERE**, and on this family more
  strongly than on the Cartesian twins: ``stepping._fill_symmetry_ghost_cells``
  returns at its first line unless ``grid.has_symmetry()`` (stepping.py:1481-1482),
  and ``Grid`` REFUSES a mirror plane on a Dcyl cell outright (grid.py:648-653), so
  no admitted configuration can even carry one. THE PREDICATE RE-CHECKS IT ANYWAY
  rather than reading its own coverage off another module's guard.
* **``zero_metal_D`` — CARRIED INLINE** (driver.py:3301; ``stepping._zero_metal``
  :2206-2247), through :func:`.coverage.zero_metal_axes`, IMPORTED rather than
  re-spelled. It writes complex zero — ``array[face] = 0`` on a complex64 volume —
  so the register clear is applied to BOTH word planes.
* **``fill_folded_far_ghosts_D`` — PROVABLY A NO-OP HERE**, by the same
  ``has_symmetry`` guard one level up (stepping.py:1565-1566).

**THE WALL TABLE ON A Dcyl GRID IS ``(False, False, z_metallic)``, AND THAT IS A
POSITIVE REQUIREMENT.** ``Grid`` builds ``metallic_axes`` as ``pair[1] == METALLIC
and pair[0] != AXIS`` (grid.py:512-514) and the r axis's pair is
``(AXIS, METALLIC)``, so r is NOT a walled axis for ``_zero_metal``'s purposes even
though its far face is a metallic wall: the r = 0 row belongs to the per-|m| rules,
not to a PEC clear. phi is periodic. So ``ZM_X`` and ``ZM_Y`` are False on every
configuration this product admits, and
:func:`cylindrical_fused_electric_pair_coverage` REFUSES a grid whose table says
otherwise rather than compiling a clear the array path does not perform. The gate
arms its wall mutation on ``ZM_Z`` for the dead-branch reason the twin records.

**THE WALL CLEAR'S COMPONENT MAP IS THE COMPLEMENT OF THE TWIN'S, AND IT IS THE ONE
PLACE A TRANSCRIBER WILL GET THIS SEAM WRONG.** ``stepping._zero_metal`` clears
every component whose Yee shift on the walled axis is ZERO (stepping.py:2243-2247).
On the B side that is the component on its own axis, so a z wall clears ``Bz``
alone; on the D side it is the two TANGENTIAL ones, so a z wall clears ``Dx`` and
``Dy`` and leaves ``Dz`` untouched. That is MEEP's ``on_metal_boundary``: a perfect
conductor shorts the tangential displacement and the normal component does not sit
on the wall. The map is baked as :data:`WALL_CLEARED_COMPONENTS` and re-derived
from the shipped ``IYEE_SHIFTS`` by the predicate, which refuses on a disagreement
— the clause that caught exactly this slip on :mod:`.complex_fused_electric_pair`'s
first device run, where the body carried the magnetic twin's map and was wrong in
both directions at once.

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
verbatim from :mod:`.complex_fused_electric_pair`, whose own weld it is.

CONSEQUENCE FOR THE SEAM, AND IT IS THE ONE ORDERING FACT THIS PRODUCT TURNS ON.
``_cylindrical_axis_zero_D`` RUNS AFTER THE PML RECURRENCE, not before it — in the
certified curl body the per-|m| rules are applied to the ``v``/``n`` registers
between the recurrence and the store (cylindrical_complex.py:960-985). So the value
this weld must hand the constitutive half is the one AT THE CERTIFIED STORE, after
those rules, not the recurrence's output: a weld that read ``v`` before them would
give ``update_E`` a displacement the array path never leaves in ``D``. The
transcription below keeps the certified order — recurrence, per-|m| rules, wall
clear, store, constitutive read — and the ``|m| >= 2`` case is where the difference
is observable, because that arm zeroes all three components AND their ``fu`` on
every row within ``ZERO_ROWS`` of the axis.

The ``|m| = 1`` D-side rule zeroes ``Dz`` on the axis row — the FIELD ONLY, never
``fu_Dz`` — and the ``|m| = 1`` axis-row INCREMENT replaces target 1 (``Dy``), not
target 0: ``AXIS_INCREMENT_TARGET['step_D'] == 1`` against ``step_B``'s 0. Both are
the certified body's own arms, taken by ``BACKWARD``.

===========================================================================
WHAT THIS BODY CARRIES THAT THE MAGNETIC TWIN DOES NOT
===========================================================================

Two constexprs and one component map, and they are the whole difference between the
two kernels:

* **``BACKWARD`` IS 1, NOT 0.** ``SUB_STEPS['step_D']['backward']`` is 1
  (launch.py:1205), and the certified curl body is already parametric on it — the
  ghost shift, the prefix substitution on target 2, the ownership mask, the axis
  increment and the per-|m| store rule each take their ``if BACKWARD:`` arm. The
  body below is therefore the SAME transcription as the twin's, not a
  hand-specialised one, which is what keeps the transcription risk confined to the
  two certified bodies.
* **``SCALE`` IS 1, NOT 0.** This is the E side, where
  :func:`.complex_fields.bloch_constitutive_step`'s ``SCALE = 1`` arm multiplies
  the source by ``inv_eps`` before the ``w`` store (complex_fields.py:782-784,
  :803, :824) — ``src = D * inv_eps``, field LEFT, the ONE multiply the H side
  compiles away. It adds three float32 pointer arguments (``e0``/``e1``/``e2``) and
  NO fourth operand orientation: the multiply is
  :func:`.complex_fields._mul_field_left`, which the PML recurrence in the curl
  half already calls six times.
* **THE COEFFICIENT LATTICES ARE THE MIRROR IMAGE OF THE TWIN'S**, and the swap is
  silent. The curl takes the INTEGER lattice on ``step_D``
  (``SUB_STEPS['step_D']['suffix'] == ""``) and the constitutive takes the
  HALF-INTEGER one on E (``CONSTITUTIVE_SIDES['E']['half_integer'] is True``, so
  ``_h``) — exactly the opposite pair from the magnetic weld. The kernel takes both
  and never asks which is which, so getting them the wrong way round is a half-cell
  error in the absorber profile that converges to a slightly worse absorber and
  nothing else. The builder is the single place that chooses, and the gate carries
  a mutation for exactly it.

``inv_eps`` is a float32 VOLUME indexed by the COMPLEX cell index — one real
coefficient per cell, applied to both planes, never word-doubled
(``fields.py:1203-1204``; the same binding
:class:`.complex_fields.ComplexConstitutivePlan` makes). Word-doubling it would
read the imaginary neighbour's coefficient into the real plane, which is smooth,
converged and wrong.

===========================================================================
WHAT IS NOT NEEDED HERE — the ownership restructure, and why
===========================================================================

The folded arm's fused pairs have to move ownership: their near mirror fill reads a
cell the same launch writes, Triton has no device-wide barrier, and the naive carry
races on ``D``. NONE OF THAT ARISES ON THIS ARM. ``Grid`` refuses a fold on a Dcyl
cell, both fills are dead, every store is at ``2*idx``/``2*idx+1``, and no lane
reads a word another lane writes. The r-direction neighbour reads this kernel makes
— the prefix's ``o_r`` load on target 2 and the ownership-masked source loads — are
reads of volumes this launch never writes (``pfx`` is built on the host before the
launch; ``g0``/``g1``/``g2`` are the H volumes).

===========================================================================
THE PREFIX STAYS ON THE ARRAY PATH, AND THE PAIR INHERITS THAT
===========================================================================

The radial prefix is a SEQUENTIAL float32 scan whose summation order DEFINES the
answer, ``cupy.cumsum``'s order is the oracle (cylindrical_triton.py:32-56), and
``tl.cumsum`` matches neither. So it cannot move into this kernel any more than it
could move into the certified curl: :meth:`CylindricalFusedElectricPairPlan.prefix`
calls :func:`.cylindrical_complex.cylindrical_complex_prefix`, which calls
``stepping.cylindrical_rderiv_prefix``. On the D side that prefix is
``(nr, ny, nz)`` and takes ``Hy`` at ``ir0 = 0.5`` with NO extended wall row
(``cylindrical_complex.PREFIX['step_D']``), which is the second place a
transcriber can silently take the B side's shape. This plan is therefore NOT
allocation-free, exactly as :class:`.cylindrical_complex.CylindricalComplexCurlPlan`
is not.

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

**THE PATTERN SET IS THE BASE FOUR, UNEXTENDED**, and for the union of the two
tranches' own measured claims. The cylindrical-only orientations are DEGENERATE
between the arms because their coefficient's real word is a zero
(``cylindrical_complex._mul_general_coefficient_left``, measured 2026-08-13,
``AMBIGUOUS_BOTH`` in all three orientations). The E-side constitutive adds the
``inv_eps`` multiply, which is ``_mul_field_left`` — already in the base four and
already called six times by the recurrence above it — and its ``kps``/``kms``
products are ``f4_mul_c8_coefficient_left``, also in the base four. The seam adds a
``tl.where`` to ``0.0``, which is a select and not a multiply.

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
dispatch stays disabled — the same deferral :mod:`.cylindrical_fused_magnetic_pair`,
:mod:`.complex_fused_electric_pair` and :mod:`.fused_dispersive_chain` ship under.

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

#: Does this product bracket its fused launch with the deposit repair? TRUE, and
#: unlike the magnetic twin it MUST be: every one of the sixteen Dcyl corpus rows
#: declares an electric source, so with the repair off this arm admits zero rows and
#: the product would exist without serving anything.
#:
#: THE PRECONDITION THIS PRODUCT MEETS is that its constitutive half is POINTWISE, on
#: both sides of the step: the diagonal ``update_E`` is ``(D - sum P) * inv_eps`` cell
#: by cell (``stepping.py:1012-1015``) — and with no polarization registered, which the
#: clause below requires, ``displacement_minus_polarization`` returns ``D`` unchanged
#: (``fields.py:1096-1098``) — while the kernel's own half reads its coefficients at
#: ``+ i``/``+ j``/``+ k`` and its state at ``+ idx``. The CURL half is not pointwise,
#: but it does not have to be: the deposit lands in ``D``, which the curl half WRITES
#: rather than reads, so the repair's reconstruction is of the curl's output and the
#: neighbour reads are all of ``H`` and of the host-built prefix. No lane's answer
#: depends on a deposit point, so a whole-grid launch against an uninjected field is
#: wrong ONLY at the deposit points and nowhere else. The configurations that are NOT
#: pointwise on the constitutive side — an off-diagonal chi1inv row, an instantaneous
#: chi2/chi3 — are refused per run by ``deposit_repair.repairable``, which the clause
#: below CONSULTS rather than assumes.
#:
#: IT IS NOT A HINT. A product that passes True without those wrappers computes the
#: constitutive half against a pre-injection field and reports success, which is the
#: exact failure ``deposit_repair`` exists to prevent. Flag and wiring move together.
CARRIES_DEPOSIT_REPAIR = True

try:  # pragma: no cover - CUDA host only
    import triton
    import triton.language as tl
    _TRITON_IMPORT_ERROR: Optional[BaseException] = None
except ImportError as _exc:  # pragma: no cover - laptop path
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_IMPORT_ERROR = _exc


#: The curl sub-step this product starts at, and the constitutive side it ends at.
CURL_SUB_STEP = "step_D"
CONSTITUTIVE_SIDE = "E"

#: The three volumes the curl half writes, in kernel argument order. READ from
#: :data:`.launch.SUB_STEPS` rather than spelled, because the wall clause below
#: indexes ``IYEE_SHIFTS`` by these names and a second spelling is a second place
#: for the component order to be wrong.
CURL_TARGETS: Tuple[str, ...] = tuple(SUB_STEPS[CURL_SUB_STEP]["targets"])

#: The FIVE driver call sites ONE launch of this plan performs, in driver order
#: (driver.py:3292-3304). Declared, never inferred from the slot name. Only THREE
#: of them do work on an admitted configuration — the two symmetry fills cannot even
#: exist on a Dcyl cell — and the inert two are listed anyway: what a launch
#: REPLACES is what the driver would otherwise have called.
REPLACES: Tuple[str, ...] = ("step_D", "fill_symmetry_bc_D", "zero_metal_D",
                             "fill_folded_far_ghosts_D", "update_E")

#: ``SUB_STEPS['step_D']['backward']``, restated so the kernel's one legal binding
#: is visible without reading :mod:`launch`; the test suite pins the two equal.
BACKWARD = 1

#: ``ComplexConstitutivePlan``'s ``scale`` for side E, restated for the same reason
#: and pinned the same way. It is the constexpr that switches on the ``inv_eps``
#: multiply, and it is the ONE arithmetic difference between this body's
#: constitutive half and the magnetic twin's.
SCALE = 1

#: WHICH CURL TARGETS ``zero_metal_D`` CLEARS, PER WALLED AXIS — the map the kernel
#: BAKES, written here so the predicate can compare it to the engine's own.
#:
#: ``stepping._zero_metal`` clears every component whose Yee shift on the walled axis
#: is ZERO (stepping.py:2243-2247). On the D side that is the TWO components whose
#: axis is not the walled one — MEEP's ``on_metal_boundary``: a perfect conductor
#: shorts the TANGENTIAL displacement, and the normal component does not sit on the
#: wall. It is the exact COMPLEMENT of the magnetic twin's map, where a z wall clears
#: Bz and nothing else.
#:
#: ON A Dcyl GRID ONLY THE THIRD ROW CAN FIRE (see :data:`FORBIDDEN_WALL_AXES`), so
#: the live entry is ``(0, 1)`` — a z wall clears ``Dx`` and ``Dy``. The other two
#: rows are written so the diagonal table is legible and so the derivation below
#: checks all three, not because either can be reached.
#:
#: THIS CONSTANT IS A CLAIM ABOUT THE KERNEL, NOT A DERIVATION. It is what the
#: ``if ZM_*:`` blocks were written to do; :func:`cylindrical_fused_electric_pair_coverage`
#: recomputes the same map from ``fields.IYEE_SHIFTS`` and refuses when they differ,
#: which is the only thing standing between a transcription slip here and a plane of
#: silently wrong values. The slip is not hypothetical: the Cartesian D->E twin
#: shipped the magnetic map to its first device run and was wrong in both directions
#: at once.
WALL_CLEARED_COMPONENTS: Tuple[Tuple[int, ...], ...] = ((1, 2), (0, 2), (0, 1))

#: The probe patterns this product's operand orientations require — the BASE set,
#: unextended. See the module docstring: the cylindrical orientations are
#: arm-degenerate and the E-side ``inv_eps`` multiply is ``_mul_field_left``, which
#: the recurrence above it already calls.
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
    "BACKWARD", "CARRIES_DEPOSIT_REPAIR", "CONSTITUTIVE_SIDE", "CURL_SUB_STEP",
    "CURL_TARGETS", "FORBIDDEN_WALL_AXES", "LICENSED_MULTIPLY_HELPERS",
    "PRODUCT_PROBE_PATTERNS", "REPLACES", "SCALE", "WALL_CLEARED_COMPONENTS",
    "CylindricalFusedElectricPairPlan",
    "cyl_complex_fused_curl_constitutive_D",
    "cyl_complex_fused_curl_constitutive_D_kernel",
    "cylindrical_fused_electric_pair_coverage",
    "plan_cylindrical_fused_electric_pair",
    "plan_cylindrical_fused_electric_pair_from_arrays",
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
    def cyl_complex_fused_curl_constitutive_D(
        f0, f1, f2,                   # curl targets:     Dx,Dy,Dz (c8 as words)
        u0, u1, u2,                   # curl auxiliaries: fu_D*    (c8 as words)
        g0, g1, g2,                   # curl sources:     Hx,Hy,Hz (c8 as words)
        pfx,                          # the ARRAY-PATH prefix (c8 as words)
        c0, c2,                       # i*m/r rows for targets 0 and 2 (c8, nr long)
        kmx, sinvx, kmy, sinvy, kmz, sinvz,   # curl kms/sinv, INTEGER lattice
        h0, h1, h2,                   # constitutive targets: Ex,Ey,Ez
        w0, w1, w2,                   # constitutive aux: f_w_Ex,f_w_Ey,f_w_Ez
        e0, e1, e2,                   # inverse epsilon, float32 VOLUMES
        kp0, km0, kp1, km1, kp2, km2, # constitutive kps/kms, HALF-INTEGER lattice
        nx, ny, nz, n_elem, dtdx,     # n_elem = COMPLEX cells; dtdx pre-rounded
        minus_dtdx,                   # |m|=1 B scalar; UNUSED on D, carried for shape
        inc_b_re, inc_b_im,           # |m|=1 B scalars; UNUSED on D
        four_dtdx,                    # m=0 D: f32(4*dtdx) for the on-axis Dz add
        BACKWARD: tl.constexpr,       # bound to 1 by every builder; see below
        SCALE: tl.constexpr,          # bound to 1 by every builder; see below
        BCZ: tl.constexpr,            # PERIODIC or METALLIC; r is CYL_AXIS, phi periodic
        M_CLASS: tl.constexpr,        # M_ZERO (m = 0), M_ONE (|m| = 1) or M_MANY (|m| >= 2)
        ZERO_ROWS: tl.constexpr,      # |m| or 1 under M_MANY; unused under M_ZERO/M_ONE
        ZM_X: tl.constexpr, ZM_Y: tl.constexpr, ZM_Z: tl.constexpr,
        EXPANSION: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Dcyl ``step_D`` + ``zero_metal_D`` + complex ``update_E``, one launch.

        ``BACKWARD`` IS CARRIED AND MUST BE 1. It keeps the curl half a verbatim
        copy of :func:`.cylindrical_complex.cyl_complex_pml_curl_step` rather than
        a hand-specialised one, which is the whole reason the transcription risk
        here is low. It cannot be 0: the B half's seam carries the MAGNETIC
        injection and has no ``inv_eps`` scaling, and that product already exists
        (:mod:`.cylindrical_fused_magnetic_pair`).

        ``SCALE`` IS CARRIED AND MUST BE 1. This is the E side, where
        :func:`.complex_fields.bloch_constitutive_step`'s ``SCALE = 1`` arm
        multiplies the source by ``inv_eps`` BEFORE the ``w`` store; the
        ``SCALE = 0`` arm reads ``B`` directly and belongs to the H side.

        ``minus_dtdx``/``inc_b_re``/``inc_b_im`` are the B-side ``|m| = 1``
        scalars. They are DEAD on this sub-step and are carried anyway so this
        signature is the certified curl's plus the constitutive's, argument for
        argument — a signature that dropped them would stop being a transcription
        and the gate's argument-order leg would have nothing to compare against.

        ``BCZ`` takes ``PERIODIC`` or ``METALLIC``; r is the CYL_AXIS identity and
        phi is periodic, both compiled in by the certified body.

        ``ZM_X``/``ZM_Y``/``ZM_Z`` are ``coverage.zero_metal_axes``. Only ``ZM_Z``
        can be true on an admitted Dcyl grid; the predicate refuses the other two
        rather than compiling a clear the array path does not perform.
        """
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = idx < n_elem
        nyz = ny * nz
        k = idx % nz
        plane = idx // nz
        j = plane % ny
        i = plane // ny

        # ======================= the curl half ================================
        # Verbatim from cylindrical_complex.cyl_complex_pml_curl_step; the only
        # edit below its stores is the wall clear applied to the v registers.

        # --- the ghost rule, per axis (stepping._shift_up / _shift_down) -------
        # r is CYL_AXIS: the far face is the metallic zero (S:1781-1783) and the
        # near ghost (S:1830-1842) is DELIBERATELY NOT IMPLEMENTED — unobservable.
        # phi is PERIODIC on a length-1 axis: the wrap returns the SAME element,
        # which is what makes the difference an exact +0.0. Computed, never elided.
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
            # step_D :425-426 — Dz's `first` source is the prefix; the backward
            # machinery is otherwise unchanged. Its row-0 ghost is unreachable for
            # the same reason every other r near ghost is (Dz's r-shift is 0).
            p_here_re = tl.load(pfx + 2 * idx, mask=live, other=0.0)
            p_here_im = tl.load(pfx + 2 * idx + 1, mask=live, other=0.0)
            p_down_re = tl.load(pfx + 2 * o_r, mask=vr, other=0.0)
            p_down_im = tl.load(pfx + 2 * o_r + 1, mask=vr, other=0.0)
            t2_re = ((p_down_re - p_here_re) + (a_re - a_p_re))
            t2_im = ((p_down_im - p_here_im) + (a_im - a_p_im))
            curl2_re, curl2_im = _mul_coefficient_left(dtdx, t2_re, t2_im, EXPANSION)
        else:
            # step_B :343-347 — Bz's WHOLE curl is the forward difference of the
            # EXTENDED prefix: one subtract, one multiply.
            pu_re = tl.load(pfx + 2 * (idx + nyz), mask=live, other=0.0)
            pu_im = tl.load(pfx + 2 * (idx + nyz) + 1, mask=live, other=0.0)
            pd_re = tl.load(pfx + 2 * idx, mask=live, other=0.0)
            pd_im = tl.load(pfx + 2 * idx + 1, mask=live, other=0.0)
            curl2_re, curl2_im = _mul_coefficient_left(dtdx, pu_re - pd_re,
                                                       pu_im - pd_im, EXPANSION)

        # --- the i*m/r coupling (stepping :674-724, sites :348-355 / :430-437) --
        # AFTER the dtdx curl, BEFORE the ownership mask — the array path's order.
        # The partners are the CENTER registers already loaded: target 0 takes `c`
        # (g2 = Hz), target 2 takes `a` (g0 = Hx). The coefficient row is
        # host-built and bound; its real word is +0.0 and MUST survive as a literal
        # cross-term operand (a plane-wise shortcut is byte-wrong on signed zeros).
        # `curl - (c*g)` carries the array path's `curl + (-(c*g))` by IEEE-754.
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

        # --- ownership mask (stepping._mask_non_owned_cells, is_axis + metallic)
        # Per axis, for every target whose Yee shift there is 0 (fields.IYEE_SHIFTS):
        #   B: Bx(0,1,1) -> r ; By(1,0,1) -> phi only, periodic: nothing ; Bz(1,1,0) -> z
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
        # ON THIS SUB-STEP THE TARGET IS 1 (Dy), not 0: AXIS_INCREMENT_TARGET.
        if M_CLASS == M_ONE:
            if BACKWARD:
                # dtdx * (Hr[i] - Hr[i, z-1] - 2.0*Hz[i]) at r = 0, on the whole row.
                # Hr = g0 center `a`, its z-down neighbour, and Hz = g2 center `c`.
                two_c_re, two_c_im = _mul_coefficient_left(2.0, c_re, c_im, EXPANSION)
                s_re = (a_re - a_z_re) - two_c_re
                s_im = (a_im - a_z_im) - two_c_im
                inc_re, inc_im = _mul_coefficient_left(dtdx, s_re, s_im, EXPANSION)
                curl1_re = tl.where(at_r, inc_re * -1.0, curl1_re)
                curl1_im = tl.where(at_r, inc_im * -1.0, curl1_im)
            else:
                # (-dtdx)*(Ep[i] - Ep[i, z+1]) - (1j*m*dtdx)*Ez[r+1] at r = 0.
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
        # dsig/dsigu cycle (vec.hpp cycle_direction): target 0 -> (y, z), 1 -> (z, x),
        # 2 -> (x, y), the same triple on both sides. Every multiply is the
        # zero-imaginary product with the FIELD on the left (S:1929-1935).
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

        # --- the per-|m| axis rules, folded into the STORED value --------------
        # THIS IS THE ORDERING FACT THE MODULE DOCSTRING NAMES. The array path runs
        # `_cylindrical_axis_zero_D` AFTER `_apply_pml_update`, so the value the
        # constitutive half must consume is the one BELOW this block, not the
        # recurrence's output. A weld that read `v` above it would hand update_E a
        # displacement the array path never leaves in D.
        # m = 0 (stepping :581-585 D / :661-662 B), complex storage, 2026-09-04:
        #   the D side POST-adds (4*Courant)*Hp[0] to the UPDATED Dz on the axis
        #   row — Hp is the g1 CENTER already loaded (`b`) — then zeroes Dy there;
        #   the B side zeroes Bx. The value BELOW this block is what update_E reads.
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

        # --- the seam: stepping.zero_metal_D (driver.py:3301) ------------------
        # Applied to the REGISTERS, before both the store and the constitutive
        # read, so the two consumers see the one value the array path leaves in D.
        # BOTH planes: `array[_face(axis, 0)] = 0` on a complex64 volume writes
        # complex zero (stepping._zero_metal :2246). ON A Dcyl GRID ONLY ZM_Z CAN
        # BE TRUE (module docstring) and the predicate refuses a grid that reports
        # otherwise; the r and phi rows are written so the diagonal table is
        # legible, not because either can fire.
        #
        # THE COMPONENT MAP IS THE COMPLEMENT OF THE MAGNETIC TWIN'S. `_zero_metal`
        # clears every component whose Yee shift on the WALLED axis is ZERO
        # (stepping.py:2243-2247). For B that is the component on its own axis --
        # IYEE_SHIFTS['Bz'] = (1, 1, 0), so a z wall clears Bz and nothing else.
        # For D it is the OTHER TWO -- IYEE_SHIFTS['Dz'] = (0, 0, 1), so a z wall
        # clears Dx and Dy and LEAVES Dz ALONE.
        if ZM_X:
            v1_re = tl.where(at_r, 0.0, v1_re)
            v1_im = tl.where(at_r, 0.0, v1_im)
            v2_re = tl.where(at_r, 0.0, v2_re)
            v2_im = tl.where(at_r, 0.0, v2_im)
        if ZM_Y:
            at_p = j == 0
            v0_re = tl.where(at_p, 0.0, v0_re)
            v0_im = tl.where(at_p, 0.0, v0_im)
            v2_re = tl.where(at_p, 0.0, v2_re)
            v2_im = tl.where(at_p, 0.0, v2_im)
        if ZM_Z:
            v0_re = tl.where(at_z, 0.0, v0_re)
            v0_im = tl.where(at_z, 0.0, v0_im)
            v1_re = tl.where(at_z, 0.0, v1_re)
            v1_im = tl.where(at_z, 0.0, v1_im)

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
        # VERBATIM from complex_fused_electric_pair's constitutive half, which is
        # itself complex_fields.bloch_constitutive_step's SCALE=1 arm with
        # `src = tl.load(g + 2*idx)` replaced by the register pair the curl half
        # just produced. `prev` is read BEFORE the `w` store (S:2083-2085); the
        # inv_eps multiply happens BETWEEN the source read and that store
        # (complex_fields.py:783-786), so `f_w_E*` holds `D * inv_eps` and not `D`;
        # the two accumulations stay separate and left-to-right, each a
        # coefficient-LEFT zero-imaginary complex product (S:2086-2087, S:2093-2095).
        # update_E carries NO cylindrical branch, which the cylindrical tranche
        # measured (480/480 rows, 0 differing words) rather than assumed.
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
        if SCALE:
            # inv_eps is a float32 volume at the COMPLEX cell index — `+ idx`, not
            # `+ 2 * idx`. D LEFT (S:982-984).
            ie = tl.load(e0 + idx, mask=live, other=0.0)
            src_re, src_im = _mul_field_left(src_re, src_im, ie, EXPANSION)
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
        if SCALE:
            ie = tl.load(e1 + idx, mask=live, other=0.0)
            src_re, src_im = _mul_field_left(src_re, src_im, ie, EXPANSION)
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
        if SCALE:
            ie = tl.load(e2 + idx, mask=live, other=0.0)
            src_re, src_im = _mul_field_left(src_re, src_im, ie, EXPANSION)
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
    cyl_complex_fused_curl_constitutive_D = None  # type: ignore[assignment]


def cyl_complex_fused_curl_constitutive_D_kernel() -> Any:
    """The JIT kernel, or a named ImportError on a host without Triton."""
    if cyl_complex_fused_curl_constitutive_D is None:
        raise ImportError(
            "the cylindrical complex fused electric D/E kernel needs the optional "
            f"`triton` package (pip install triton). Original error: "
            f"{_TRITON_IMPORT_ERROR}")
    return cyl_complex_fused_curl_constitutive_D


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def cylindrical_fused_electric_pair_coverage(fields: Any, pml: Any,
                                             sources: Any = None,
                                             probe: Any = None) -> Coverage:
    """May ONE launch span Dcyl ``step_D`` -> wall -> complex ``update_E``?

    A conjunction of the two halves' own predicates plus the seam clauses. Nothing
    is weakened: a configuration either half refuses is refused here with that
    half's reasons, prefixed so a reader can tell which side said it — the same
    construction
    :func:`.cylindrical_fused_magnetic_pair.cylindrical_fused_magnetic_pair_coverage`,
    :func:`.complex_fused_electric_pair.complex_fused_electric_pair_coverage` and
    :func:`.coverage.fused_pair_coverage` use. The Dcyl geometry clauses, the
    storage clause (complex64 at every m; the real m = 0 run is cylindrical_triton's),
    the nr >= 2 clause and the EXPANSION licence are all
    INHERITED from the halves and not restated.

    ``probe`` is forwarded to both halves unchanged, so the EXPANSION licence is
    decided ONCE over :data:`PRODUCT_PROBE_PATTERNS` — the base set, because this
    product launches no operand orientation its halves do not.
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

    # THE SOURCE SEAM. An electric source is injected BETWEEN the two halves
    # (driver.py:3296-3299), so a bare fused pair would consume a pre-injection D.
    # A MAGNETIC source is injected in the B/H seam and does not disqualify this
    # pair. IGNORANCE IS NOT AN EMPTY SET: `Fields` does not hold the source list,
    # so a predicate that inferred "no electric source" from not being told would
    # be the over-covering refusal this clause exists to prevent.
    #
    # THIS IS THE CLAUSE THE WHOLE PRODUCT TURNS ON, and more completely than on
    # the Cartesian D->E twin: ALL SIXTEEN Dcyl rows declare an electric source, so
    # with `carries_repair=False` this arm admits ZERO rows. The flag is passed,
    # not assumed: `repairable` still refuses by name every seam it cannot invert.
    reasons.extend(_deposit_repair.seam_source_reasons(
        fields, sources, 'D',
        undeclared=(
            "the source set was not declared: this predicate cannot infer an empty "
            "electric source seam from Fields"
        ),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) is electric: the "
            f"driver injects it BETWEEN step_D and update_E "
            f"(driver.py:3296), which is work inside the seam this kernel "
            f"closes"),
        carries_repair=CARRIES_DEPOSIT_REPAIR))

    # THE TWO SYMMETRY PASSES. `fill_symmetry_bc_D` (driver.py:3300) and
    # `fill_folded_far_ghosts_D` (:3302) both return at their first line unless
    # `grid.has_symmetry()` (stepping.py:1481-1482, :1565-1566); `Grid` refuses a
    # mirror on a Dcyl cell outright (grid.py:648-653) and both halves restate
    # that. RE-CHECKED HERE ANYWAY, not inferred: this module's coverage may not be
    # read off another module's guard, and the cost is one attribute read.
    if _call(grid, "has_symmetry", default=False):
        reasons.append(
            "a mirror plane is active: stepping.fill_symmetry_bc_D (driver.py:3300) "
            "and fill_folded_far_ghosts_D (:3302) run inside this seam and this "
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
                f"grid does not expose {name}; zero_metal_D cannot be carried inline")

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

    # ...AND THE COMPONENT MAP THE CLEAR BAKES MUST BE THE ENGINE'S OWN. Recomputed
    # from the shipped ``IYEE_SHIFTS`` by exactly ``_zero_metal``'s rule — shift 0
    # on the walled axis — rather than compared against a second transcription of
    # it. This is the clause that would have refused the map the Cartesian D->E
    # twin shipped to its first device run, where it carried the magnetic twin's
    # `axis d -> component d` and was wrong on every metallic grid in both
    # directions at once.
    from ..fields import IYEE_SHIFTS  # noqa: PLC0415

    derived = tuple(
        tuple(index for index, name in enumerate(CURL_TARGETS)
              if IYEE_SHIFTS[name][axis] == 0)
        for axis in range(3))
    if derived != WALL_CLEARED_COMPONENTS:
        reasons.append(
            f"stepping._zero_metal clears {derived} per walled axis on "
            f"{list(CURL_TARGETS)}, but this kernel bakes "
            f"{WALL_CLEARED_COMPONENTS}; the inline wall clear would zero one plane "
            f"and leave another live")

    # A susceptibility makes the constitutive source (D - sum P) rather than D
    # (fields.py:1096-1105). `cylindrical_complex_constitutive_coverage` already
    # refuses it on the E side; restated because THIS kernel bakes the plain product
    # and a reader should not have to chase the other predicate to learn that. This
    # is a D->E-only clause: it has no counterpart on the magnetic twin, whose
    # source is B whatever is registered.
    if tuple(getattr(fields, "polarizations", ()) or ()):
        reasons.append(
            "a susceptibility is registered: this kernel bakes the plain "
            "constitutive product, whose source is D and not (D - sum P)")

    # THE INVERSE-EPSILON VOLUMES ARE READ BY THE KERNEL, so `Fields` must be able
    # to hand one over per target component. The magnetic twin has no counterpart
    # to this clause: its SCALE=0 arm binds the source word views as placeholders
    # and never loads them. An absent accessor here would be a TypeError inside the
    # builder rather than a refusal, which is the wrong way for an uncovered
    # configuration to fail.
    if getattr(fields, "inverse_epsilon_for", None) is None:
        reasons.append(
            "fields does not expose inverse_epsilon_for; the E-side constitutive "
            "half cannot bind the inv_eps volumes its SCALE=1 arm loads")

    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class CylindricalFusedElectricPairPlan:
    """One launch for five of the driver's electric call sites, plus the array-path
    prefix the cylindrical family cannot avoid.

    The complex volumes are bound as float32 WORD VIEWS at plan time, once, and
    ``n_elem`` stays the COMPLEX cell count — the same split
    :class:`.complex_fields.ComplexPmlCurlPlan` makes, for the same reason. The
    ``inv_eps`` volumes are the exception and are bound RAW: they are float32 with
    one real coefficient per complex cell, and the kernel indexes them at ``+ idx``
    (``fields.py:1203-1204``).

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
        "shape", "n_elem", "dtdx", "backward", "scale", "bcz", "m_class",
        "zero_rows", "zero_metal", "expansion", "block", "num_warps",
        "increment_scalars", "four_dtdx", "xp", "scratch", "_targets", "_aux",
        "_sources",
        "_source_map", "_curl_coefficients", "_imr_rows", "_e_targets", "_e_aux",
        "_inv_eps", "_e_coefficients", "_grid", "_kernel",
    )

    #: The five driver call sites one launch performs. Declared, so a composition
    #: can be inspected rather than inferred from the curl slot's name.
    replaces = REPLACES

    def __init__(self, shape, dtdx: float, m: int,
                 accurate_fields_near_cylorigin: bool, bcz: int, zero_metal,
                 expansion: int, block: int,
                 targets, auxiliaries, sources, curl_coefficients,
                 e_targets, e_aux, inverse_epsilon, e_coefficients, xp: Any,
                 scratch: Any = None, kernel: Any = None,
                 num_warps: Optional[int] = 1) -> None:
        spec = SUB_STEPS[CURL_SUB_STEP]
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        # float(): the array path consumes the Python scalar at complex64 precision
        # and Triton types a Python float argument as fp32, so the two are the same
        # bits (ComplexPmlCurlPlan's measured note).
        self.dtdx = float(dtdx)
        self.backward = BACKWARD
        self.scale = SCALE
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
        self._e_targets = tuple(CupyPointer(_word_view(a)) for a in e_targets)
        self._e_aux = tuple(CupyPointer(_word_view(a)) for a in e_aux)
        # NOT word-viewed, and not optional. Unlike ComplexConstitutivePlan, which
        # binds the source views as placeholders on the H side because its SCALE
        # constexpr compiles the loads away, THIS product is E-only: SCALE is 1 on
        # every launch, so a placeholder here would be read as an epsilon.
        if inverse_epsilon is None:
            raise ValueError(
                "the E-side fused pair loads inv_eps on every launch (SCALE=1); "
                "a placeholder binding would be read as a coefficient")
        self._inv_eps = tuple(CupyPointer(a) for a in inverse_epsilon)
        self._e_coefficients = tuple(
            CupyPointer(_flat(a)) for a in e_coefficients)
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
        """This launch's radial prefix, from the shipped array-path scan.

        THE D-SIDE PREFIX IS NOT THE B-SIDE ONE: ``(nr, ny, nz)`` from ``Hy`` at
        ``ir0 = 0.5`` with no extended wall row (``cylindrical_complex.PREFIX``).
        The sub-step name is what selects it, and it is a constant of this module.
        """
        return cylindrical_complex_prefix(self.xp, CURL_SUB_STEP, self._source_map,
                                          scratch=self.scratch)

    def run(self, guard: Optional[bool] = None) -> None:
        """Launch the fused pair, in place. Same ``guard`` contract as every plan."""
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        kernel = (self._kernel if self._kernel is not None
                  else cyl_complex_fused_curl_constitutive_D_kernel())
        (minus_dtdx, inc_b) = self.increment_scalars
        nx, ny, nz = self.shape
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        kernel[self._grid](
            *self._targets, *self._aux, *self._sources,
            CupyPointer(_word_view(self.prefix())), *self._imr_rows,
            *self._curl_coefficients,
            *self._e_targets, *self._e_aux, *self._inv_eps, *self._e_coefficients,
            nx, ny, nz, self.n_elem, self.dtdx,
            minus_dtdx, inc_b[0], inc_b[1], self.four_dtdx,
            BACKWARD=self.backward,
            SCALE=self.scale,
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
        return (f"CylindricalFusedElectricPairPlan(shape={self.shape}, "
                f"bcz={self.bcz}, m_class={self.m_class}, "
                f"zero_rows={self.zero_rows}, zero_metal={self.zero_metal}, "
                f"expansion={self.expansion}, block={self.block}, "
                f"num_warps={self.num_warps})")


def plan_cylindrical_fused_electric_pair(
        fields: Any, pml: Any, sources: Any = None, block: Optional[int] = None,
        num_warps: Optional[int] = 1, kernel: Any = None, probe: Any = None,
        ) -> Optional[CylindricalFusedElectricPairPlan]:
    """Build the plan from the engine's own objects, or ``None`` when refused.

    ``None`` is the only refusal: a configuration this kernel does not carry must
    fall back to the array path, never raise into a caller that would otherwise
    have stepped correctly.
    """
    if not cylindrical_fused_electric_pair_coverage(fields, pml, sources,
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
    return CylindricalFusedElectricPairPlan(
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
        [fields.inverse_epsilon_for(name) for name in side_spec["targets"]],
        # THE MIRROR IMAGE OF THE MAGNETIC TWIN'S CHOICE. The curl takes the
        # INTEGER lattice on step_D (SUB_STEPS['step_D']['suffix'] == "") and the
        # constitutive the HALF-INTEGER one on E (stepping.py:1015 against the D
        # curl's half_integer=False). The kernel takes both and never asks which is
        # which, so a swap here is a silent half-cell error in the absorber
        # profile; the gate carries a mutation for exactly it.
        [getattr(pml, f"{stem}_{axis}_h") for axis in "xyz"
         for stem in ("kps", "kms")],
        grid.xp, scratch=getattr(fields, "scratch", None),
        kernel=kernel, num_warps=num_warps,
    )


def plan_cylindrical_fused_electric_pair_from_arrays(
        arrays: Dict[str, Any], curl_flat: Dict[str, Any],
        constitutive_flat: Dict[str, Any], dtdx: float, m: int,
        accurate_fields_near_cylorigin: bool, bcz: int, zero_metal,
        expansion: int, xp: Any, scratch: Any = None,
        block: Optional[int] = None, kernel: Any = None,
        num_warps: Optional[int] = 1) -> CylindricalFusedElectricPairPlan:
    """Build from bare device arrays — the gate's route.

    No coverage predicate runs: the caller is a harness that constructed the
    configuration deliberately, including the deliberately wrong ones. ``arrays``
    is keyed by component name plus ``fu_D*``, ``f_w_E*`` and ``inv_eps_Ex``...
    (float32); ``curl_flat`` is keyed on the sub-lattice the caller ALREADY
    SELECTED, so the gate can hand over a swapped pair and measure that the swap is
    caught, and ``kernel=`` carries the mutation override — dropping it silently
    disarms every mutation leg.
    """
    curl_spec = SUB_STEPS[CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    shape = tuple(int(n) for n in arrays[curl_spec["targets"][0]].shape)
    return CylindricalFusedElectricPairPlan(
        shape, dtdx, m, accurate_fields_near_cylorigin, bcz, zero_metal,
        int(expansion), DEFAULT_BLOCK if block is None else block,
        [arrays[name] for name in curl_spec["targets"]],
        [arrays["fu_" + name] for name in curl_spec["targets"]],
        [arrays[name] for name in curl_spec["sources"]],
        [curl_flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
        [arrays[name] for name in side_spec["targets"]],
        [arrays[name] for name in side_spec["aux"]],
        [arrays["inv_eps_" + name] for name in side_spec["targets"]],
        [constitutive_flat[f"{stem}_{axis}"]
         for axis in "xyz" for stem in ("kps", "kms")],
        xp, scratch=scratch, kernel=kernel, num_warps=num_warps,
    )
