"""The COMPLEX electric seam in one launch: complex ``step_D`` welded into ``update_E``.

The complex/Bloch arm's D->E product, and the twin of
:mod:`.complex_fused_magnetic_pair`. The two halves are the certified complex
tranche's own kernels — :func:`.complex_fields.bloch_pml_curl_step` on ``step_D``
and :func:`.complex_fields.bloch_constitutive_step` on the E side — and the weld
is the one substitution :func:`.kernels.fused_curl_constitutive_D` makes on the
real path, done on both float32 word planes.

WHY THIS HALF EXISTS AT ALL, AND WHY IT DID NOT ON 2026-08-19. The B->H twin's
module docstring records the census funnel that decided the order the two arms
were built in::

    complex D->E   16 halves ->  1 no-electric-source
    complex B->H   16 halves -> 12 no-magnetic-source

FIFTEEN of the sixteen complex rows declare an electric source, so under the
source clause as it stood this product would have admitted ONE row and the
magnetic twin twelve — which is why the magnetic one was built first and this one
was not built at all. THAT PREMISE IS NOW FALSE. ``deposit_repair`` carries a
deposit across the seam rather than refusing it (see :data:`CARRIES_DEPOSIT_REPAIR`
below), and the cell's ceiling is measured at SIXTEEN of sixteen in
``parity/meep_gpu/results/fusion_matrix_triton_2026-08-30_before_electric``::

    D->E   16  ceiling 16   complex PML -> complex        NO FUSED PRODUCT

That line is the whole reason this module exists: it was the largest single
buildable gap on the Triton board, tied with the cylindrical complex D->E cell,
and the board's own bucket table scored it under "reachable; NO PRODUCT OCCUPIES
THIS CELL — not built" rather than under any refusal.

===========================================================================
THE FIVE PASSES IN THE SEAM
===========================================================================

The driver runs five passes between the two halves (driver.py:3292-3304)::

    step_D -> electric sources -> fill_symmetry_bc_D -> zero_metal_D
           -> fill_folded_far_ghosts_D -> update_E

Each is answered by name, and the FIRST is answered differently here than on the
magnetic twin — which is the whole of what is new about the seam:

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
* **``fill_symmetry_bc_D`` — PROVABLY A NO-OP HERE, not carried and not refused.**
  ``stepping._fill_symmetry_ghost_cells`` returns at its first line unless
  ``grid.has_symmetry()`` (stepping.py:1481-1482), and BOTH halves' own predicate
  refuses ``has_symmetry`` and every ``is_mirrored`` axis outright
  (``complex_fields._complex_grid_reasons`` — "symmetry folding is not carried").
  So on every configuration this product admits the pass cannot write a word. THE
  PREDICATE RE-CHECKS IT ANYWAY rather than reading its own coverage off another
  module's guard: if the complex tranche ever admits a fold, this weld would
  silently swallow a pass that had started doing work.
* **``zero_metal_D`` — CARRIED INLINE** (driver.py:3301;
  ``stepping._zero_metal`` :2206-2247), through :func:`.coverage.zero_metal_axes`,
  which is IMPORTED rather than re-spelled. It writes complex zero — ``array[face]
  = 0`` on a complex64 volume — so the register clear is applied to BOTH word
  planes.
* **``fill_folded_far_ghosts_D`` — PROVABLY A NO-OP HERE**, by the same
  ``has_symmetry`` guard one level up (``stepping._fill_folded_far_ghosts``,
  stepping.py:1565-1566), and re-checked by the same clause.

===========================================================================
WHAT THIS BODY CARRIES THAT THE MAGNETIC TWIN DOES NOT
===========================================================================

Two constexprs and one component map, and they are the whole difference between
the two kernels:

* **``BACKWARD`` IS 1, NOT 0.** ``SUB_STEPS['step_D']['backward']`` is 1
  (launch.py:1202), and the certified curl body is already parametric on it — the
  ghost shift, the wrapped-lane predicate and the ownership mask each take their
  ``if BACKWARD:`` arm. The body below is therefore the SAME transcription as the
  twin's, not a hand-specialised one, which is what keeps the transcription risk
  confined to the two certified bodies.
* **``SCALE`` IS 1, NOT 0.** This is the E side, where
  :func:`.complex_fields.bloch_constitutive_step`'s ``SCALE = 1`` arm multiplies
  the source by ``inv_eps`` before the ``w`` store (complex_fields.py:782-784,
  :803, :824) — ``src = D * inv_eps``, field LEFT, the ONE multiply the H side
  compiles away. It adds three float32 pointer arguments (``e0``/``e1``/``e2``)
  and NO fourth operand orientation: the multiply is
  :func:`.complex_fields._mul_field_left`, which the PML recurrence in the curl
  half already calls six times. That is why :data:`PRODUCT_PROBE_PATTERNS` is the
  BASE set here exactly as it is on the twin, and why
  :data:`LICENSED_MULTIPLY_HELPERS` is the same three names.

* **THE WALL CLEAR CLEARS THE OTHER TWO COMPONENTS.** ``stepping._zero_metal``
  clears every component whose Yee shift on the walled axis is ZERO
  (stepping.py:2243-2247). On the B side that is the component on its own axis, so
  an x wall clears ``Bx`` alone; on the D side it is the two TANGENTIAL ones, so an
  x wall clears ``Dy`` and ``Dz`` and leaves ``Dx`` untouched. The map is the exact
  complement of the twin's and is the one place a transcriber will get this seam
  wrong — this body did, on its first device run, and came back wrong in BOTH
  directions at once on both metallic cases while every periodic case agreed word
  for word over ten steps. It is now baked as
  :data:`WALL_CLEARED_COMPONENTS` and re-derived from the shipped ``IYEE_SHIFTS``
  by the predicate, which refuses on a disagreement.

``inv_eps`` is a float32 VOLUME indexed by the COMPLEX cell index — one real
coefficient per cell, applied to both planes, never word-doubled
(``fields.py:1203-1204``; the same binding :class:`.complex_fields.ComplexConstitutivePlan`
makes). Word-doubling it would read the imaginary neighbour's coefficient into the
real plane, which is smooth, converged and wrong.

THE COEFFICIENT LATTICES ARE THE MIRROR IMAGE OF THE TWIN'S, and the swap is
silent. The curl takes the INTEGER lattice on ``step_D`` (``SUB_STEPS['step_D']
['suffix'] == ""``) and the constitutive takes the HALF-INTEGER one on E
(``CONSTITUTIVE_SIDES['E']['half_integer'] is True``, so ``_h``) — which is
exactly the opposite pair from the magnetic weld. The kernel takes both and never
asks which is which, so getting them the wrong way round is a half-cell error in
the absorber profile that converges to a slightly worse absorber and nothing else.
The builder is the single place that chooses, and the gate carries a mutation for
exactly it.

===========================================================================
WHAT IS NOT NEEDED HERE — the ownership restructure, and why
===========================================================================

The folded arm's fused pairs have to move ownership: their near mirror fill reads
a cell the same launch writes, Triton has no device-wide barrier, and the naive
carry races on ``D``. NONE OF THAT ARISES ON THIS ARM. Both fills are refused out
of existence by the halves' own symmetry clause, so no lane reads a word another
lane writes, every store is at ``2*idx``/``2*idx+1``, and the weld is a straight
one: the constitutive half's ``src`` load is replaced by the register pair the
curl half just produced, and nothing else moves. Saying so is not a shrug — it is
the reason this product's transcription risk is confined to the two certified
bodies plus one register clear, and the gate's transcription leg checks exactly
that by tracing every arithmetic line back to the shipped source it came from.

THE OTHER THING THAT IS NOT NEEDED: an off-diagonal or dispersive constitutive.
Both change ``update_E``'s source from ``D`` to something else — a stencil over
the partner volumes, or ``(D - sum P)`` — and both are refused, the first by
``complex_constitutive_coverage`` itself and the second by the restated clause in
:func:`complex_fused_electric_pair_coverage`. Neither refusal is inherited: this
module's rule is that its coverage may not be read off another module's guard.

===========================================================================
DEVICE STATUS
===========================================================================

**NOT RELEASED.** No byte-identity claim is made anywhere in this file and this
module has no ``fingerprints.json`` entry until its gate,
``parity/meep_gpu/probe_triton_complex_fused_electric_pair.py``, has run on a
device and ``rebind_triton_welds.py`` has bound the weld to the bytes that
executed. Until then this is a product with a predicate and a plan, which is not
the same thing as a product with an identity.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .complex_fields import (
    DEFAULT_BLOCK,
    PROBE_PATTERNS,
    _mul_coefficient_left,
    _mul_field_left,
    _phase_arguments,
    _resolve_expansion,
    _rotate_field_left,
    _word_view,
    bloch_phase_table,
    complex_constitutive_coverage,
    complex_pml_curl_coverage,
)
from .coverage import (
    CONSTITUTIVE_SIDES,
    Coverage,
    _call,
    zero_metal_axes,
)
from .launch import SUB_STEPS, CupyPointer, _flat
from .. import deposit_repair as _deposit_repair

#: Does this product bracket its fused launch with the deposit repair? YES, and the
#: wiring this claims is ``launch._install_fused_pair`` (launch.py:1648): when
#: ``deposit_repair.in_seam_sources`` is non-empty it puts a ``LeadingRepairPlan`` in
#: this pair's FIRST slot (``step_D``), which saves the field and ``f_w`` at every
#: deposit point immediately before the launch, and a ``TrailingRepairPlan`` in the
#: SECOND (``update_E``), which recomputes them after the driver has injected, filled
#: symmetry and cleared walls.
#:
#: THIS FLAG IS THE WHOLE REASON THIS PRODUCT IS WORTH BUILDING. Fifteen of the
#: sixteen complex rows declare an electric source; with the flag False the source
#: clause below refuses every one of them and the product admits ONE row. The census
#: funnel in the module docstring is that measurement, and it is why the magnetic twin
#: shipped in August and this one did not.
#:
#: THE PRECONDITION THIS PRODUCT MEETS is that its constitutive half is POINTWISE, on
#: both sides of the step: the diagonal ``update_E`` is ``(D - sum P) * inv_eps`` cell
#: by cell (``stepping.py:1012-1015``) — and with no polarization registered, which the
#: clause below requires, ``displacement_minus_polarization`` returns ``D`` unchanged
#: (``fields.py:1096-1098``) — while the kernel's own half reads six values at
#: ``+ idx`` and its coefficients at ``+ i``/``+ j``/``+ k``. No neighbour read on
#: either side, so a whole-grid launch against an uninjected field is wrong ONLY at
#: the deposit points and nowhere else. The two configurations that are NOT pointwise
#: — an off-diagonal chi1inv row, whose ``update_E`` is a stencil over the curl's own
#: in-place ``D`` output, and an instantaneous chi2/chi3 — are refused per run by
#: ``deposit_repair.repairable``, which the clause below CONSULTS rather than assumes.
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
#: :data:`.launch.SUB_STEPS` rather than spelled, because the wall-clause below
#: indexes ``IYEE_SHIFTS`` by these names and a second spelling is a second place
#: for the component order to be wrong.
CURL_TARGETS: Tuple[str, ...] = tuple(SUB_STEPS[CURL_SUB_STEP]["targets"])

#: The FIVE driver call sites ONE launch of this plan performs, in driver order
#: (driver.py:3292-3304). Declared, never inferred from the slot name. Only THREE
#: of them do work on an admitted configuration — the two symmetry fills return at
#: their first line without a symmetry — and the inert two are listed anyway: what
#: a launch REPLACES is what the driver would otherwise have called, and a pass
#: that is inert today is a pass whose guard the predicate has to keep re-checking.
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
#: ``stepping._zero_metal`` clears every component whose Yee shift on the walled
#: axis is ZERO (stepping.py:2243-2247). On the D side that is the TWO components
#: whose axis is not the walled one — MEEP's ``on_metal_boundary``: a perfect
#: conductor shorts the TANGENTIAL displacement, and the normal component does not
#: sit on the wall. It is the exact COMPLEMENT of the magnetic twin's map, where an
#: x wall clears Bx and nothing else.
#:
#: INDEPENDENTLY CONFIRMED BY THE ONE OTHER D->E PRODUCT IN THE TREE.
#: :mod:`.folded_fused_pair`, released 2026-08-20 and routed into the planner since
#: 2026-08-27, carries exactly this map in its own inline clear — ``ZM_X`` clears
#: ``v1`` and ``v2``, ``ZM_Y`` clears ``v0`` and ``v2``, ``ZM_Z`` clears ``v0`` and
#: ``v1`` (folded_fused_pair.py:708-716). So the map below is not only what
#: ``IYEE_SHIFTS`` derives: it is what a separately gated D->E weld already ships.
#:
#: THIS CONSTANT IS A CLAIM ABOUT THE KERNEL, NOT A DERIVATION. It is what the
#: ``if ZM_*:`` blocks were written to do; :func:`complex_fused_electric_pair_coverage`
#: recomputes the same map from ``fields.IYEE_SHIFTS`` and refuses when they differ,
#: which is the only thing standing between a transcription slip here and a plane of
#: silently wrong values. The slip is not hypothetical: this body shipped the twin's
#: map to its first device run and was wrong in both directions at once.
WALL_CLEARED_COMPONENTS: Tuple[Tuple[int, ...], ...] = ((1, 2), (0, 2), (0, 1))

#: The probe patterns this product's operand orientations require — the BASE set,
#: unextended. See the module docstring: the ``inv_eps`` multiply this body adds
#: over the twin's is ``_mul_field_left``, which the curl half already calls.
#: Named rather than passed implicitly so the claim is inspectable and testable.
PRODUCT_PROBE_PATTERNS: Tuple[str, ...] = tuple(PROBE_PATTERNS)

#: The complex-multiply device functions this kernel is allowed to call, and the
#: only ones. A fourth would be a fourth operand orientation, which would need its
#: own probe pattern before it could be licensed; the test scans for exactly this.
LICENSED_MULTIPLY_HELPERS: Tuple[str, ...] = (
    "_rotate_field_left", "_mul_field_left", "_mul_coefficient_left")

__all__ = [
    "BACKWARD", "CARRIES_DEPOSIT_REPAIR", "CONSTITUTIVE_SIDE", "CURL_SUB_STEP",
    "CURL_TARGETS", "LICENSED_MULTIPLY_HELPERS", "PRODUCT_PROBE_PATTERNS",
    "REPLACES", "SCALE", "WALL_CLEARED_COMPONENTS",
    "ComplexFusedElectricPairPlan",
    "complex_fused_curl_constitutive_D",
    "complex_fused_curl_constitutive_D_kernel",
    "complex_fused_electric_pair_coverage",
    "plan_complex_fused_electric_pair",
    "plan_complex_fused_electric_pair_from_arrays",
]


if triton is not None:

    # The same constexpr code as ``complex_fields.METALLIC``, restated for the
    # reason that module restates ``kernels``' — importing a ``tl.constexpr``
    # wrapper and re-wrapping it is not the same object, and the comparison in the
    # kernel body is against a literal code. ONLY METALLIC IS NAMED: the certified
    # complex curl body branches on ``== METALLIC`` and takes periodic as the else,
    # so a ``PERIODIC`` constant here would be a name nothing reads.
    METALLIC = tl.constexpr(1)

    @triton.jit
    def complex_fused_curl_constitutive_D(
        f0, f1, f2,                       # curl targets: Dx,Dy,Dz (complex64 as words)
        u0, u1, u2,                       # curl auxiliaries: fu_Dx,fu_Dy,fu_Dz
        g0, g1, g2,                       # curl sources: Hx,Hy,Hz
        kmx, sinvx, kmy, sinvy, kmz, sinvz,   # curl kms/sinv, INTEGER lattice
        h0, h1, h2,                       # constitutive targets: Ex,Ey,Ez
        w0, w1, w2,                       # constitutive aux: f_w_Ex,f_w_Ey,f_w_Ez
        e0, e1, e2,                       # inverse epsilon, float32 VOLUMES
        kp0, km0, kp1, km1, kp2, km2,     # constitutive kps/kms, HALF-INTEGER lattice
        nx, ny, nz, n_elem, dtdx,         # n_elem = COMPLEX cells; dtdx pre-rounded
        pxr, pxi, pyr, pyi, pzr, pzi,     # per-axis complex64-rounded phase
        BACKWARD: tl.constexpr,           # bound to 1 by every builder; see below
        SCALE: tl.constexpr,              # bound to 1 by every builder; see below
        BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
        ZM_X: tl.constexpr, ZM_Y: tl.constexpr, ZM_Z: tl.constexpr,
        PHX: tl.constexpr, PHY: tl.constexpr, PHZ: tl.constexpr,
        EXPANSION: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Complex ``step_D`` + ``zero_metal_D`` + complex ``update_E``, one launch.

        ``BACKWARD`` IS CARRIED AND MUST BE 1. It keeps the curl half a verbatim
        copy of :func:`.complex_fields.bloch_pml_curl_step` rather than a hand-
        specialised one, which is the whole reason the transcription risk here is
        low. It cannot be 0: the B half's seam carries the MAGNETIC injection and
        has no ``inv_eps`` scaling, and that product already exists
        (:mod:`.complex_fused_magnetic_pair`).

        ``SCALE`` IS CARRIED AND MUST BE 1. This is the E side, where
        :func:`.complex_fields.bloch_constitutive_step`'s ``SCALE = 1`` arm
        multiplies the source by ``inv_eps`` BEFORE the ``w`` store; the
        ``SCALE = 0`` arm reads ``B`` directly and belongs to the H side. Carried
        as a constexpr rather than inlined for the same reason ``BACKWARD`` is:
        the body stays a transcription of one certified function, and the gate's
        transcription leg can compare statement lists rather than paraphrases.

        ``BCX``/``BCY``/``BCZ`` take the two codes the complex tranche covers,
        ``PERIODIC`` and ``METALLIC`` (``coverage.COVERED_BOUNDARIES``); a fold is
        refused by both halves' own predicate, so no mirror code reaches here.

        ``ZM_X``/``ZM_Y``/``ZM_Z`` are ``coverage.zero_metal_axes``, the question
        ``stepping._zero_metal`` asks — the grid's own declaration, NOT the
        resolved ghost rule, which an invariant axis softens to periodic while the
        wall clear still fires there.
        """
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = idx < n_elem
        nyz = ny * nz
        k = idx % nz
        plane = idx // nz
        j = plane % ny
        i = plane // ny

        # ======================= the curl half ================================
        # Verbatim from complex_fields.bloch_pml_curl_step; the only edit below
        # its stores is the wall clear applied to the v registers.

        # --- the ghost rule, per axis (stepping._shift_up / _shift_down) -------
        # PERIODIC wraps; METALLIC serves an exact 0.0 past the wall — a
        # (+0.0, +0.0) word pair from `other=` IS the complex metallic ghost
        # (S:1781-1783, S:1827-1829).
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

        # --- wrapped-lane predicates, one plane per axis ------------------------
        if BACKWARD:
            wx, wy, wz = i == 0, j == 0, k == 0
        else:
            wx, wy, wz = i == nx - 1, j == ny - 1, k == nz - 1

        # --- loads: two words per operand --------------------------------------
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

        # --- Bloch phase on the wrapped lane, BEFORE the difference -------------
        # Field LEFT (S:1862); the host passed the CONJUGATE for BACKWARD
        # (S:1818-1822). `tl.where` is a bitwise select, so unwrapped lanes keep
        # the loaded words untouched.
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

        # --- the curl (stepping._curl_from_operands): DO NOT flatten these parens
        t0_re = ((c_y_re - c_re) + (b_re - b_z_re))
        t0_im = ((c_y_im - c_im) + (b_im - b_z_im))
        t1_re = ((a_z_re - a_re) + (c_re - c_x_re))
        t1_im = ((a_z_im - a_im) + (c_im - c_x_im))
        t2_re = ((b_x_re - b_re) + (a_re - a_y_re))
        t2_im = ((b_x_im - b_im) + (a_im - a_y_im))
        curl0_re, curl0_im = _mul_coefficient_left(dtdx, t0_re, t0_im, EXPANSION)
        curl1_re, curl1_im = _mul_coefficient_left(dtdx, t1_re, t1_im, EXPANSION)
        curl2_re, curl2_im = _mul_coefficient_left(dtdx, t2_re, t2_im, EXPANSION)

        # --- ownership mask (stepping._mask_non_owned_cells) -------------------
        # Writes +0.0 to BOTH planes — the array path assigns complex zero
        # (S:1896, S:1902). THIS IS NOT THE WALL CLEAR: it zeroes the CURL at a
        # non-owned cell, which the PML recurrence then folds into a v that is
        # generally nonzero. `zero_metal_D` further down zeroes the RESULT.
        # ON THIS SUB-STEP THE BACKWARD ARM IS THE LIVE ONE, and it masks TWO
        # components per metallic axis rather than one: a backward difference at
        # stored cell 0 reads the cell below the wall for both components whose
        # curl crosses that axis.
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

        # --- split-field recurrence (stepping._apply_pml_update) ---------------
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

        # --- the seam: stepping.zero_metal_D (driver.py:3301) ------------------
        # Applied to the REGISTERS, before both the store and the constitutive
        # read, so the two consumers see the one value the array path leaves in D.
        # BOTH planes: `array[_face(axis, 0)] = 0` on a complex64 volume writes
        # complex zero (stepping._zero_metal :2246). A folded axis is refused by
        # both halves, which is also why `_zero_metal`'s own fold skip
        # (:2237-2239) has nothing to skip here.
        #
        # THE COMPONENT MAP IS THE COMPLEMENT OF THE MAGNETIC TWIN'S, AND IT IS THE
        # ONE PLACE A TRANSCRIBER WILL GET THIS SEAM WRONG. `_zero_metal` clears
        # every component whose Yee shift on the WALLED axis is ZERO
        # (stepping.py:2243-2247). For B that is the component on its own axis --
        # IYEE_SHIFTS['Bx'] = (0, 1, 1), so an x wall clears Bx and nothing else.
        # For D it is the OTHER TWO -- IYEE_SHIFTS['Dx'] = (1, 0, 0), so an x wall
        # clears Dy and Dz and LEAVES Dx ALONE. That is MEEP's `on_metal_boundary`:
        # a perfect conductor shorts the TANGENTIAL displacement and the normal
        # component is not on the wall at all.
        #
        # MEASURED, NOT REASONED. This body carried the twin's `ZM_X -> v0` map on
        # its first device run and came back wrong in BOTH directions at once on
        # both metallic cases -- Dx differing by 140 words with the fused route
        # zeroing a plane the array path kept, and by 72 words the other way -- while
        # every periodic case agreed word for word over ten steps
        # (results/run_elecpair1). :func:`complex_fused_electric_pair_coverage`
        # re-derives this map from the shipped ``IYEE_SHIFTS`` and REFUSES if the
        # two disagree, so the constant below cannot drift away from the array path
        # in silence a second time.
        if ZM_X:
            v1_re = tl.where(at_x, 0.0, v1_re)
            v1_im = tl.where(at_x, 0.0, v1_im)
            v2_re = tl.where(at_x, 0.0, v2_re)
            v2_im = tl.where(at_x, 0.0, v2_im)
        if ZM_Y:
            v0_re = tl.where(at_y, 0.0, v0_re)
            v0_im = tl.where(at_y, 0.0, v0_im)
            v2_re = tl.where(at_y, 0.0, v2_re)
            v2_im = tl.where(at_y, 0.0, v2_im)
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
        # Verbatim from complex_fields.bloch_constitutive_step's SCALE=1 arm, with
        # `src = tl.load(g + 2*idx)` replaced by the register pair the curl half
        # just produced — the same substitution kernels.fused_curl_constitutive_D
        # makes on the real path, done on both planes. `prev` is read BEFORE the
        # `w` store (S:2083-2085); the inv_eps multiply happens BETWEEN the source
        # read and that store (complex_fields.py:783-786), so `f_w_E*` holds
        # `D * inv_eps` and not `D`; the two accumulations stay separate and
        # left-to-right, each a coefficient-LEFT zero-imaginary complex product
        # (S:2086-2087, S:2093-2095).
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
    complex_fused_curl_constitutive_D = None  # type: ignore[assignment]


def complex_fused_curl_constitutive_D_kernel() -> Any:
    """The JIT kernel, or a named ImportError on a host without Triton."""
    if complex_fused_curl_constitutive_D is None:
        raise ImportError(
            "the complex fused electric D/E kernel needs the optional `triton` "
            f"package (pip install triton). Original error: {_TRITON_IMPORT_ERROR}")
    return complex_fused_curl_constitutive_D


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def complex_fused_electric_pair_coverage(fields: Any, pml: Any,
                                         sources: Any = None,
                                         probe: Any = None) -> Coverage:
    """May ONE launch span complex ``step_D`` -> wall -> complex ``update_E``?

    A conjunction of the two halves' own predicates plus the seam clauses.
    Nothing is weakened: a configuration either half refuses is refused here with
    that half's reasons, prefixed so a reader can tell which side said it — the
    same construction
    :func:`.complex_fused_magnetic_pair.complex_fused_magnetic_pair_coverage`,
    :func:`.folded_fused_pair.folded_fused_pair_coverage` and
    :func:`.coverage.fused_pair_coverage` use.

    ``probe`` is forwarded to both halves unchanged, so the EXPANSION licence is
    decided ONCE, by :func:`.complex_fields.expansion_license`, over
    :data:`PRODUCT_PROBE_PATTERNS` — the base set, because this product launches
    no operand orientation its halves do not (see the module docstring). It is
    not re-implemented here and no clause below softens it.
    """
    reasons: List[str] = []

    curl = complex_pml_curl_coverage(fields, pml, CURL_SUB_STEP, probe=probe)
    if not curl.covered:
        reasons.extend(f"complex curl half: {reason}" for reason in curl.reasons)
    constitutive = complex_constitutive_coverage(fields, pml, CONSTITUTIVE_SIDE,
                                                 probe=probe)
    if not constitutive.covered:
        reasons.extend(f"complex constitutive half: {reason}"
                       for reason in constitutive.reasons)

    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, tuple(dict.fromkeys(reasons + ["fields carries no grid"])))

    # THE SOURCE SEAM. An electric source is injected BETWEEN the two halves
    # (driver.py:3296-3299), so a bare fused pair would consume a pre-injection D.
    # A MAGNETIC source is injected in the B/H seam and does not disqualify this
    # pair. IGNORANCE IS NOT AN EMPTY SET: `Fields` does not hold the source list,
    # so a predicate that inferred "no electric source" from not being told would
    # be the over-covering refusal this clause exists to prevent.
    #
    # THIS IS THE CLAUSE THE WHOLE PRODUCT TURNS ON. 15 of the 16 complex rows
    # declare an electric source, so with `carries_repair=False` this arm admits
    # ONE row; with the repair it admits what `deposit_repair.repairable` allows,
    # which the board measures at 16 of 16 for this cell. The flag is passed, not
    # assumed: `repairable` still refuses by name every seam it cannot invert.
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
    # `grid.has_symmetry()` (stepping.py:1481-1482, :1565-1566), and BOTH halves
    # already refuse a symmetry. RE-CHECKED HERE ANYWAY, not inferred: this
    # module's coverage may not be read off another module's guard, and if the
    # complex tranche ever admits a fold, this weld would silently swallow two
    # passes that had started doing work. The clause costs one attribute read and
    # is the difference between "no pass runs" and "no pass ran when I last
    # looked".
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
    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            reasons.append(
                f"grid does not expose {name}; zero_metal_D cannot be carried inline")

    # ...AND THE COMPONENT MAP IT BAKES MUST BE THE ENGINE'S OWN. Recomputed from
    # the shipped ``IYEE_SHIFTS`` by exactly ``_zero_metal``'s rule — shift 0 on the
    # walled axis — rather than compared against a second transcription of it. This
    # is the clause that would have refused the map this module shipped to its first
    # device run, where it carried the magnetic twin's `axis d -> component d` and
    # was wrong on every metallic grid in both directions at once.
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
    # (fields.py:1096-1105). `complex_constitutive_coverage` already refuses it on
    # the E side; restated because THIS kernel bakes the plain product and a reader
    # should not have to chase the other predicate to learn that — the same reason
    # `folded_fused_pair_coverage` restates it. This is a D->E-only clause: it has
    # no counterpart on the magnetic twin, whose source is B whatever is registered.
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

class ComplexFusedElectricPairPlan:
    """One allocation-free launch for five of the driver's electric call sites.

    The complex volumes are bound as float32 WORD VIEWS at plan time, once, and
    ``n_elem`` stays the COMPLEX cell count — the same split
    :class:`.complex_fields.ComplexPmlCurlPlan` makes, for the same reason. The
    ``inv_eps`` volumes are the exception and are bound RAW: they are float32 with
    one real coefficient per complex cell, and the kernel indexes them at ``+ idx``
    (``fields.py:1203-1204``; the same binding
    :class:`.complex_fields.ComplexConstitutivePlan` makes).
    """

    __slots__ = (
        "shape", "n_elem", "dtdx", "backward", "scale", "bc", "zero_metal",
        "phased", "phase_values", "expansion", "block", "num_warps",
        "_targets", "_aux", "_sources", "_curl_coefficients",
        "_e_targets", "_e_aux", "_inv_eps", "_e_coefficients", "_grid", "_kernel",
    )

    #: The five driver call sites one launch performs. Declared, so a composition
    #: can be inspected rather than inferred from the curl slot's name.
    replaces = REPLACES

    def __init__(self, shape, dtdx: float, bc, zero_metal, phased, phase_values,
                 expansion: int, block: int,
                 targets, auxiliaries, sources, curl_coefficients,
                 e_targets, e_aux, inverse_epsilon, e_coefficients,
                 kernel: Any = None, num_warps: Optional[int] = 1) -> None:
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        # float(): the array path consumes the Python scalar at complex64
        # precision, and Triton types a Python float argument as fp32, so the two
        # are the same bits (complex_fields.ComplexPmlCurlPlan's measured note).
        self.dtdx = float(dtdx)
        self.backward = BACKWARD
        self.scale = SCALE
        self.bc = tuple(int(code) for code in bc)
        self.zero_metal = tuple(bool(value) for value in zero_metal)
        self.phased = tuple(int(flag) for flag in phased)
        self.phase_values = tuple(float(value) for value in phase_values)
        self.expansion = int(expansion)
        self.block = int(block)
        self.num_warps = None if num_warps is None else int(num_warps)
        self._targets = tuple(CupyPointer(_word_view(a)) for a in targets)
        self._aux = tuple(CupyPointer(_word_view(a)) for a in auxiliaries)
        self._sources = tuple(CupyPointer(_word_view(a)) for a in sources)
        self._curl_coefficients = tuple(
            CupyPointer(_flat(a)) for a in curl_coefficients)
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
        # THE INT32 WORD BOUND. The kernel addresses words as ``2 * idx`` in
        # int32, so the complex cell count must leave room for the doubling —
        # the same halving the complex predicates apply (module docstring of
        # complex_fields, "Word addressing"). Refused here as well because the
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

    def run(self, guard: Optional[bool] = None) -> None:
        """Launch the fused pair, in place. Same ``guard`` contract as every plan."""
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        kernel = (self._kernel if self._kernel is not None
                  else complex_fused_curl_constitutive_D_kernel())
        nx, ny, nz = self.shape
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        kernel[self._grid](
            *self._targets, *self._aux, *self._sources, *self._curl_coefficients,
            *self._e_targets, *self._e_aux, *self._inv_eps, *self._e_coefficients,
            nx, ny, nz, self.n_elem, self.dtdx,
            *self.phase_values,
            BACKWARD=self.backward,
            SCALE=self.scale,
            BCX=self.bc[0], BCY=self.bc[1], BCZ=self.bc[2],
            ZM_X=self.zero_metal[0], ZM_Y=self.zero_metal[1],
            ZM_Z=self.zero_metal[2],
            PHX=self.phased[0], PHY=self.phased[1], PHZ=self.phased[2],
            EXPANSION=self.expansion,
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )

    def __repr__(self) -> str:
        return (f"ComplexFusedElectricPairPlan(shape={self.shape}, bc={self.bc}, "
                f"zero_metal={self.zero_metal}, phased={self.phased}, "
                f"expansion={self.expansion}, block={self.block}, "
                f"num_warps={self.num_warps})")


def plan_complex_fused_electric_pair(fields: Any, pml: Any, sources: Any = None,
                                     block: Optional[int] = None,
                                     num_warps: Optional[int] = 1,
                                     kernel: Any = None,
                                     probe: Any = None,
                                     ) -> Optional[ComplexFusedElectricPairPlan]:
    """Build the plan from the engine's own objects, or ``None`` when refused.

    ``None`` is the only refusal: a configuration this kernel does not carry must
    fall back to the array path, never raise into a caller that would otherwise
    have stepped correctly.
    """
    if not complex_fused_electric_pair_coverage(fields, pml, sources,
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
    phased, values = _phase_arguments(bloch_phase_table(grid, kinds),
                                      backward=bool(curl_spec["backward"]))
    return ComplexFusedElectricPairPlan(
        grid.shape, grid.dt / grid.dx,
        [1 if kind == "metallic" else 0 for kind in kinds],
        zero_metal_axes(grid), phased, values, expansion,
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
        kernel=kernel, num_warps=num_warps,
    )


def plan_complex_fused_electric_pair_from_arrays(
        arrays: Dict[str, Any], curl_flat: Dict[str, Any],
        constitutive_flat: Dict[str, Any], codes, zero_metal,
        phases: Sequence[Optional[complex]], dtdx: float, expansion: int,
        block: Optional[int] = None, kernel: Any = None,
        num_warps: Optional[int] = 1) -> ComplexFusedElectricPairPlan:
    """Build from bare device arrays — the gate's route.

    No coverage predicate runs: the caller is a harness that constructed the
    configuration deliberately, including the deliberately wrong ones. ``arrays``
    is keyed by component name plus ``fu_D*``, ``f_w_E*`` and ``inv_eps_Ex``...
    (float32); ``phases`` is the per-axis ``Optional[complex]`` table (None =
    unphased) and the conjugation for a backward sub-step is applied HERE, exactly
    as the engine route applies it — this product is backward-only, so it always
    is.
    """
    curl_spec = SUB_STEPS[CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    shape = tuple(int(n) for n in arrays[curl_spec["targets"][0]].shape)
    phased, values = _phase_arguments(tuple(phases),
                                      backward=bool(curl_spec["backward"]))
    return ComplexFusedElectricPairPlan(
        shape, dtdx, codes, zero_metal, phased, values, int(expansion),
        DEFAULT_BLOCK if block is None else block,
        [arrays[name] for name in curl_spec["targets"]],
        [arrays["fu_" + name] for name in curl_spec["targets"]],
        [arrays[name] for name in curl_spec["sources"]],
        [curl_flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
        [arrays[name] for name in side_spec["targets"]],
        [arrays[name] for name in side_spec["aux"]],
        [arrays["inv_eps_" + name] for name in side_spec["targets"]],
        [constitutive_flat[f"{stem}_{axis}"]
         for axis in "xyz" for stem in ("kps", "kms")],
        kernel=kernel, num_warps=num_warps,
    )
