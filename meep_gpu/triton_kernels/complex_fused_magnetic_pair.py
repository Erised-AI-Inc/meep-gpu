"""The COMPLEX magnetic seam in one launch: complex ``step_B`` welded into ``update_H``.

The complex/Bloch arm's B->H product. The two halves are the certified complex
tranche's own kernels — :func:`.complex_fields.bloch_pml_curl_step` on ``step_B``
and :func:`.complex_fields.bloch_constitutive_step` on the H side — and the weld
is the one substitution :func:`.kernels.fused_curl_constitutive_B` makes on the
real path, done on both float32 word planes.

WHY THE B/H HALF AND NOT THE D/E HALF — measured, from the 186-row census
``parity/meep_gpu/results/predicate_coverage_2026-08-16_wired_convention``, read on
``covered_modulo_backend`` because that lift ran on NumPy. The seam funnels:

    complex D->E   16 halves ->  1 no-electric-source
    complex B->H   16 halves -> 12 no-magnetic-source

THE SOURCE CLAUSE IS THE BINDING ONE ON BOTH HALVES, and on this arm it is far
kinder to the magnetic side than on the folded one: of the 16 complex/Bloch rows,
FOUR declare a magnetic source (``oblique-planewave.py``, ``wvg-src.py``,
``TestModeDecomposition.test_phase_1``, ``TestWvgSrc.test_wvg_src``) and FIFTEEN
declare an electric one. So this product admits 12 where the electric twin admits
1. The derivation is
``parity/meep_gpu/results/triton_complex_fused_magnetic_pair_census_2026-08-19T2355/count_corpus_admission.py``
and the number is an UPPER bound — four clauses of the shipped predicate are not
evaluable from the census's configuration block, and each can only remove rows.

===========================================================================
THE FIVE PASSES IN THE SEAM
===========================================================================

The driver runs five passes between the two halves (driver.py:3281-3288)::

    step_B -> magnetic sources -> fill_symmetry_bc_B -> zero_metal_B
           -> fill_folded_far_ghosts_B -> update_H

Each is answered by name, and two of them are answered DIFFERENTLY here than on
the folded arm — which is the whole of what is new:

* **the magnetic sources — REFUSED BY NAME.** ``driver.step`` injects them between
  the two halves (driver.py:3283-3284), so a fused pair would consume a
  pre-injection ``B``. Ignorance is never an empty set: ``Fields`` does not hold
  the source list, so an undeclared ``sources`` is a REFUSAL and not an assumed
  ``()``. This is the clause that caps the family, and it is 4 of the 16 rows.
* **``fill_symmetry_bc_B`` — PROVABLY A NO-OP HERE, not carried and not refused.**
  ``stepping._fill_symmetry_ghost_cells`` returns at its first line unless
  ``grid.has_symmetry()`` (stepping.py:1481-1482), and BOTH halves' own predicate
  refuses ``has_symmetry`` and every ``is_mirrored`` axis outright
  (``complex_fields._complex_grid_reasons``, complex_fields.py:1715-1719 — "symmetry
  folding is not carried"). So on every configuration this product admits the pass
  cannot write a word. THE PREDICATE RE-CHECKS IT ANYWAY rather than reading its
  own coverage off another module's guard: if the complex tranche ever admits a
  fold, this weld would silently swallow a pass that had started doing work.
* **``zero_metal_B`` — CARRIED INLINE** (driver.py:3286;
  ``stepping._zero_metal`` :2206-2247), through :func:`.coverage.zero_metal_axes`,
  which is IMPORTED rather than re-spelled. It writes complex zero — ``array[face]
  = 0`` on a complex64 volume — so the register clear is applied to BOTH word
  planes.
* **``fill_folded_far_ghosts_B`` — PROVABLY A NO-OP HERE**, by the same
  ``has_symmetry`` guard one level up (``stepping._fill_folded_far_ghosts``,
  stepping.py:1565-1566), and re-checked by the same clause.

===========================================================================
WHAT IS NOT NEEDED HERE — the ownership restructure, and why
===========================================================================

The folded arm's fused pair has to move ownership: its near mirror fill reads a
cell the same launch writes, Triton has no device-wide barrier, and the naive
carry races on ``B``. NONE OF THAT ARISES ON THIS ARM. Both fills are refused out
of existence by the halves' own symmetry clause, so no lane reads a word another
lane writes, every store is at ``2*idx``/``2*idx+1``, and the weld is a straight
one: the constitutive half's ``src`` load is replaced by the register pair the
curl half just produced, and nothing else moves. Saying so is not a shrug — it is
the reason this product's transcription risk is confined to the two certified
bodies plus one register clear, and the gate's transcription leg checks exactly
that by tracing every arithmetic line back to the shipped source it came from.

===========================================================================
THE COMPLEX PRODUCTS
===========================================================================

Every real-coefficient product on the complex path is the real scalar times each
stored part (``part_products.scale``), and the curl's only complex operation is the
Bloch connection, which this module imports (:func:`.complex_fields._connect`)
rather than re-spells, with :mod:`.forms`' curl, recurrence and constitutive forms.
The seam adds a ``tl.where`` to ``0.0``, which is a select, not a multiply.

NOT WIRED, and not an arm. ``launch.plan_step`` assigns at most one plan per
``STEP_ORDER`` slot and this product spans FIVE driver call sites; there is no slot
it can claim without a composition rule nothing has measured. Nothing in
``launch.py`` names this module, ``fastpath.plan_fast_path`` is unchanged, and
dispatch stays disabled — the same deferral :mod:`.folded_fused_magnetic_pair` and
:mod:`.fused_dispersive_chain` ship under.

DEVICE STATUS: **RELEASED 2026-08-20**, the GPU host RTX A6000 GPU 7, Triton 3.1.0,
    under the 'keep' float32 subnormal policy. 8/8 device legs bit-identical to the
    array path and 9/9 mutations matching their declared expectation
    (``parity/meep_gpu/results/triton_fused_magnetic_pairs_2026-08-20/``). The weld
    is ``triton_complex_fused_magnetic_pair_device_gate`` in ``fingerprints.json``.

    STILL NOT WIRED. A weld licenses the claim, not a dispatch: nothing in a
    default run reaches this plan. The gate is
``parity/meep_gpu/probe_triton_complex_fused_magnetic_pair.py`` and no leg of it
marked ``device`` has ever executed — this stream was built and laptop-tested
under a standing instruction not to run a device gate. No byte-identity claim is
made anywhere in this file, and this module has no ``fingerprints.json`` entry for
exactly that reason.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .complex_fields import (
    DEFAULT_BLOCK,
    _word_view,
    bloch_phase_table,
    connection_arguments,
    complex_constitutive_coverage,
    complex_pml_curl_coverage,
)
from .coverage import (
    CONSTITUTIVE_SIDES,
    Coverage,
    MAGNETIC_FIELD_TYPE,
    _call,
    zero_metal_axes,
)
from .launch import SUB_STEPS, CupyPointer, _flat
from .. import deposit_repair as _deposit_repair

#: Does this product bracket its fused launch with the deposit repair? Flipping it is a
#: claim about the PLAN this module builds -- that the leading slot saves and the trailing
#: slot restores -- and is only ever changed in the same edit as that wiring. See
#: ``deposit_repair`` for why.
#:
#: BECAME ``True`` ON 2026-08-30, and the thing that changed is not this module. While it
#: was False the source-presence clause below refused every in-seam deposit, because
#: ``deposit_repair.apply`` wrote only the deposit index itself and the driver's
#: post-injection ``fill_symmetry_bc_*`` / ``fill_folded_far_ghosts_*`` write that cell's
#: MIRROR IMAGES somewhere else -- so a repaired seam would have left the images holding a
#: pre-injection field. That premise is now false: ``deposit_repair._fill_image_rules``
#: (deposit_repair.py:217) and ``repair_cells`` (:256) extend the saved and restored set to
#: exactly the cells those two fills image a deposit into, which is the INVERSE of the
#: forward carry this module's kernel already performs. Metal made the identical flip on
#: 2026-08-28 for the identical reason.
#:
#: THE FLAG IS NOT THE WHOLE CLAIM. It only reaches ``deposit_repair.repairable``, which
#: goes on refusing BY NAME every seam the repair cannot invert -- an off-diagonal
#: constitutive (a stencil over the curl's own in-place output), a nonlinear one, a
#: cylindrical r = 0 axis, a fold too short to hold the near fill's source row, and an
#: absorber whose split-field recurrence never ran. Flipping this widens the predicate to
#: what the repair can carry and to nothing else.
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
CURL_SUB_STEP = "step_B"
CONSTITUTIVE_SIDE = "H"

#: The FIVE driver call sites ONE launch of this plan performs, in driver order
#: (driver.py:3281-3288). Declared, never inferred from the slot name. Only THREE
#: of them do work on an admitted configuration — the two symmetry fills return at
#: their first line without a symmetry — and the inert two are listed anyway: what
#: a launch REPLACES is what the driver would otherwise have called, and a pass
#: that is inert today is a pass whose guard the predicate has to keep re-checking.
REPLACES: Tuple[str, ...] = ("step_B", "fill_symmetry_bc_B", "zero_metal_B",
                             "fill_folded_far_ghosts_B", "update_H")

#: ``SUB_STEPS['step_B']['backward']``, restated so the kernel's one legal binding
#: is visible without reading :mod:`launch`; the test suite pins the two equal.
BACKWARD = 0

#: The complex-multiply device functions this kernel is allowed to call, and the
#: only ones: the partner connection. Every other product has a real operand and is
#: formed part by part, inline, as the certified halves form it; the test scans the
#: shipped body for any other helper.
MULTIPLY_HELPERS: Tuple[str, ...] = ("_connect",)

__all__ = [
    "BACKWARD", "CONSTITUTIVE_SIDE", "CURL_SUB_STEP", "MULTIPLY_HELPERS", "REPLACES",
    "ComplexFusedMagneticPairPlan",
    "complex_fused_curl_constitutive_B",
    "complex_fused_curl_constitutive_B_kernel",
    "complex_fused_magnetic_pair_coverage",
    "plan_complex_fused_magnetic_pair",
    "plan_complex_fused_magnetic_pair_from_arrays",
]


if triton is not None:

    # The same constexpr code as ``complex_fields.METALLIC``, restated for the
    # reason that module restates ``kernels``' — importing a ``tl.constexpr``
    # wrapper and re-wrapping it is not the same object, and the comparison in the
    # kernel body is against a literal code. ONLY METALLIC IS NAMED: the certified
    # complex curl body branches on ``== METALLIC`` and takes periodic as the else,
    # so a ``PERIODIC`` constant here would be a name nothing reads.
    METALLIC = tl.constexpr(1)

    # The forms, the record layout and the connection, read by the kernel below.
    from .complex_fields import _connect  # noqa: E402
    from .forms import (axis_present, constitutive_form, curl_form,  # noqa: E402
                        curl_stencil, record_in)
    from .kernels import RECORD_HALF_INTEGER, RECORD_PRESENCE  # noqa: E402

    @triton.jit(do_not_specialize=["start_h"])
    def complex_fused_curl_constitutive_B(
        f0, f1, f2,                       # curl targets: Bx,By,Bz (complex64 as words)
        u0, u1, u2,                       # curl auxiliaries: fu_Bx,fu_By,fu_Bz
        g0, g1, g2,                       # curl sources: Ex,Ey,Ez
        kmx, sinvx, kmy, sinvy, kmz, sinvz,   # curl kms/sinv, HALF-INTEGER lattice
        h0, h1, h2,                       # constitutive targets: Hx,Hy,Hz
        w0, w1, w2,                       # constitutive aux: f_w_Hx,f_w_Hy,f_w_Hz
        kp0, km0, kp1, km1, kp2, km2,     # constitutive kps/kms, INTEGER lattice
        cb,                               # chunk record: the forms' membership and presence
        nx, ny, nz, n_elem, dtdx, start_h,   # n_elem = COMPLEX cells; start_h: start of H
        cxr, cxi, cyr, cyi, czr, czi,     # per-axis connection factor, float32 parts
        BACKWARD: tl.constexpr,           # bound to 0 by every builder; see below
        BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
        ZM_X: tl.constexpr, ZM_Y: tl.constexpr, ZM_Z: tl.constexpr,
        CNX: tl.constexpr, CNY: tl.constexpr, CNZ: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Complex ``step_B`` + ``zero_metal_B`` + complex ``update_H``, one launch.

        The two halves are :func:`.complex_fields.bloch_pml_curl_step` and
        :func:`.complex_fields.bloch_constitutive_step`'s H arm, per part, with the
        one substitution :func:`.kernels.fused_curl_constitutive_B` makes on the
        real path: the constitutive half's ``src`` load becomes the register pair the
        curl half just produced. Both halves read the chunk record ``cb`` (the curl's
        dsigu lines on the half-integer lattice, the constitutive's dsigw lines on the
        integer one) and the curl's partner connection is the curl body's.

        ``BACKWARD`` IS CARRIED AND MUST BE 0: it keeps the curl half a verbatim copy
        of the certified body. ``start_h`` is MEEP's start of H, as
        :func:`.kernels.constitutive_step` carries it. ``ZM_X``/``ZM_Y``/``ZM_Z`` are
        ``coverage.zero_metal_axes``, the question ``stepping._zero_metal`` asks --
        the grid's own declaration, NOT the resolved ghost rule.
        """
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = idx < n_elem
        nyz = ny * nz
        k = idx % nz
        plane = idx // nz
        j = plane % ny
        i = plane // ny

        # ======================= the curl half ================================
        # complex_fields.bloch_pml_curl_step's body, statement for statement; the
        # only edit below its last form is the wall clear on the v registers.
        # --- the ghost rule, per axis (stepping._shift_up / _shift_down) ------------
        # PERIODIC wraps (a plain roll, phased or not); METALLIC serves an exact 0.0 past
        # the wall -- a (+0.0, +0.0) word pair from `other=` IS the complex metallic ghost.
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

        # --- loads: two words per operand ------------------------------------------
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

        # --- the connection on the UNSHIFTED partners, plane 0 of each partner's axis
        # Target 0 reads c along y and b along z, target 1 a along z and c along x,
        # target 2 b along x and a along y (stepping's term table). `tl.where` is a
        # bitwise select, so every cell off plane 0 keeps its loaded words.
        at_x, at_y, at_z = i == 0, j == 0, k == 0
        c0_re, c0_im = c_re, c_im          # target 0's g1, connected on y
        b0_re, b0_im = b_re, b_im          # target 0's g2, connected on z
        a1_re, a1_im = a_re, a_im          # target 1's g1, connected on z
        c1_re, c1_im = c_re, c_im          # target 1's g2, connected on x
        b2_re, b2_im = b_re, b_im          # target 2's g1, connected on x
        a2_re, a2_im = a_re, a_im          # target 2's g2, connected on y
        if CNX:
            r_re, r_im = _connect(c_re, c_im, cxr, cxi, CNX)
            c1_re = tl.where(at_x, r_re, c_re)
            c1_im = tl.where(at_x, r_im, c_im)
            r_re, r_im = _connect(b_re, b_im, cxr, cxi, CNX)
            b2_re = tl.where(at_x, r_re, b_re)
            b2_im = tl.where(at_x, r_im, b_im)
        if CNY:
            r_re, r_im = _connect(c_re, c_im, cyr, cyi, CNY)
            c0_re = tl.where(at_y, r_re, c_re)
            c0_im = tl.where(at_y, r_im, c_im)
            r_re, r_im = _connect(a_re, a_im, cyr, cyi, CNY)
            a2_re = tl.where(at_y, r_re, a_re)
            a2_im = tl.where(at_y, r_im, a_im)
        if CNZ:
            r_re, r_im = _connect(b_re, b_im, czr, czi, CNZ)
            b0_re = tl.where(at_z, r_re, b_re)
            b0_im = tl.where(at_z, r_im, b_im)
            r_re, r_im = _connect(a_re, a_im, czr, czi, CNZ)
            a1_re = tl.where(at_z, r_re, a_re)
            a1_im = tl.where(at_z, r_im, a_im)

        # --- the curl (stepping._curl_from_operands), per part ---------------------
        # kernels.pml_curl_step's three calls on each part: MEEP's left-to-right
        # association with both partners, the one-partner form where the record says the
        # other partner's derivative axis is invariant, an exact +0 with neither. The
        # dtdx scale is the real scalar times each part (part_products.scale).
        presence = tl.load(cb + RECORD_PRESENCE)
        px = axis_present(presence, 0)
        py = axis_present(presence, 1)
        pz = axis_present(presence, 2)
        curl0_re = curl_stencil(c_y_re, c0_re, b0_re, b_z_re, dtdx, py, pz)
        curl0_im = curl_stencil(c_y_im, c0_im, b0_im, b_z_im, dtdx, py, pz)
        curl1_re = curl_stencil(a_z_re, a1_re, c1_re, c_x_re, dtdx, pz, px)
        curl1_im = curl_stencil(a_z_im, a1_im, c1_im, c_x_im, dtdx, pz, px)
        curl2_re = curl_stencil(b_x_re, b2_re, a2_re, a_y_re, dtdx, px, py)
        curl2_im = curl_stencil(b_x_im, b2_im, a2_im, a_y_im, dtdx, px, py)

        # --- ownership mask (stepping._mask_non_owned_cells) -----------------------
        # Writes +0.0 to BOTH parts -- the array path assigns complex zero.
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

        # --- the PML forms (stepping._apply_curl_regions), per part ----------------
        # dsig/dsigu cycle: target 0 -> (y, z), 1 -> (z, x), 2 -> (x, y). One record
        # select per target serves both parts; fu is read and written on the
        # recurrence's cells only (the auxiliary contract), both words.
        km_x = tl.load(kmx + i, mask=live, other=0.0)
        si_x = tl.load(sinvx + i, mask=live, other=0.0)
        km_y = tl.load(kmy + j, mask=live, other=0.0)
        si_y = tl.load(sinvy + j, mask=live, other=0.0)
        km_z = tl.load(kmz + k, mask=live, other=0.0)
        si_z = tl.load(sinvz + k, mask=live, other=0.0)

        if BACKWARD:
            lattice = 0
        else:
            lattice = RECORD_HALF_INTEGER
        du0 = record_in(cb, lattice + 8, k)
        du1 = record_in(cb, lattice + 0, i)
        du2 = record_in(cb, lattice + 4, j)

        p_re = tl.load(u0 + 2 * idx, mask=live & du0, other=0.0)
        p_im = tl.load(u0 + 2 * idx + 1, mask=live & du0, other=0.0)
        v0_re, n0_re = curl_form(tl.load(f0 + 2 * idx, mask=live, other=0.0), p_re, curl0_re,
                                 km_y, si_y, km_z, si_z, du0)
        v0_im, n0_im = curl_form(tl.load(f0 + 2 * idx + 1, mask=live, other=0.0), p_im,
                                 curl0_im, km_y, si_y, km_z, si_z, du0)

        p_re = tl.load(u1 + 2 * idx, mask=live & du1, other=0.0)
        p_im = tl.load(u1 + 2 * idx + 1, mask=live & du1, other=0.0)
        v1_re, n1_re = curl_form(tl.load(f1 + 2 * idx, mask=live, other=0.0), p_re, curl1_re,
                                 km_z, si_z, km_x, si_x, du1)
        v1_im, n1_im = curl_form(tl.load(f1 + 2 * idx + 1, mask=live, other=0.0), p_im,
                                 curl1_im, km_z, si_z, km_x, si_x, du1)

        p_re = tl.load(u2 + 2 * idx, mask=live & du2, other=0.0)
        p_im = tl.load(u2 + 2 * idx + 1, mask=live & du2, other=0.0)
        v2_re, n2_re = curl_form(tl.load(f2 + 2 * idx, mask=live, other=0.0), p_re, curl2_re,
                                 km_x, si_x, km_y, si_y, du2)
        v2_im, n2_im = curl_form(tl.load(f2 + 2 * idx + 1, mask=live, other=0.0), p_im,
                                 curl2_im, km_x, si_x, km_y, si_y, du2)

        # --- the seam: stepping.zero_metal_B, between the two sub-steps ---------
        # Applied to the REGISTERS, before both the store and the constitutive read,
        # so the two consumers see the one value the array path leaves in B. BOTH
        # parts: `array[_face(axis, 0)] = 0` on a complex64 volume writes complex
        # zero. On the B side the component on the walled axis is cleared (its Yee
        # shift there is 0).
        if ZM_X:
            v0_re = tl.where(at_x, 0.0, v0_re)
            v0_im = tl.where(at_x, 0.0, v0_im)
        if ZM_Y:
            v1_re = tl.where(at_y, 0.0, v1_re)
            v1_im = tl.where(at_y, 0.0, v1_im)
        if ZM_Z:
            v2_re = tl.where(at_z, 0.0, v2_re)
            v2_im = tl.where(at_z, 0.0, v2_im)

        # --- stores: fu (recurrence cells only), then the fields, both parts ----
        tl.store(u0 + 2 * idx, n0_re, mask=live & du0)
        tl.store(u0 + 2 * idx + 1, n0_im, mask=live & du0)
        tl.store(u1 + 2 * idx, n1_re, mask=live & du1)
        tl.store(u1 + 2 * idx + 1, n1_im, mask=live & du1)
        tl.store(u2 + 2 * idx, n2_re, mask=live & du2)
        tl.store(u2 + 2 * idx + 1, n2_im, mask=live & du2)
        tl.store(f0 + 2 * idx, v0_re, mask=live)
        tl.store(f0 + 2 * idx + 1, v0_im, mask=live)
        tl.store(f1 + 2 * idx, v1_re, mask=live)
        tl.store(f1 + 2 * idx + 1, v1_im, mask=live)
        tl.store(f2 + 2 * idx, v2_re, mask=live)
        tl.store(f2 + 2 * idx + 1, v2_im, mask=live)

        # ==================== the constitutive half ===========================
        # complex_fields.bloch_constitutive_step's SCALE=0 arm, per part, with
        # `src = tl.load(g + 2*idx)` replaced by the register pair the curl half just
        # produced -- the substitution kernels.fused_curl_constitutive_B
        # makes on the real path. The record's dsigw line on the integer lattice
        # picks the form; f_w is read and written on the accumulation's cells only,
        # `prev` before the store.
        kp_0 = tl.load(kp0 + i, mask=live, other=0.0)
        km_0 = tl.load(km0 + i, mask=live, other=0.0)
        kp_1 = tl.load(kp1 + j, mask=live, other=0.0)
        km_1 = tl.load(km1 + j, mask=live, other=0.0)
        kp_2 = tl.load(kp2 + k, mask=live, other=0.0)
        km_2 = tl.load(km2 + k, mask=live, other=0.0)
        dw0 = record_in(cb, 0 + 0, i)
        dw1 = record_in(cb, 0 + 4, j)
        dw2 = record_in(cb, 0 + 8, k)
        # MEEP's start of H: the registers v0..v2 ARE the B words MEEP's first
        # update_eh copies from.
        start = start_h != 0

        # --- component 0 -------------------------------------------------------
        prev_re = tl.load(w0 + 2 * idx, mask=live & dw0, other=0.0)   # BEFORE the store.
        prev_im = tl.load(w0 + 2 * idx + 1, mask=live & dw0, other=0.0)
        src_re = v0_re
        src_im = v0_im
        tl.store(w0 + 2 * idx, src_re, mask=live & dw0)
        tl.store(w0 + 2 * idx + 1, src_im, mask=live & dw0)
        a_re = tl.load(h0 + 2 * idx, mask=live & dw0, other=0.0)
        a_im = tl.load(h0 + 2 * idx + 1, mask=live & dw0, other=0.0)
        prev_re = tl.where(start, src_re, prev_re)    # MEEP's start of H
        prev_im = tl.where(start, src_im, prev_im)
        a_re = tl.where(start, src_re, a_re)
        a_im = tl.where(start, src_im, a_im)
        tl.store(h0 + 2 * idx, constitutive_form(a_re, prev_re, src_re, kp_0, km_0, dw0),
                 mask=live)
        tl.store(h0 + 2 * idx + 1,
                 constitutive_form(a_im, prev_im, src_im, kp_0, km_0, dw0), mask=live)

        # --- component 1 -------------------------------------------------------
        prev_re = tl.load(w1 + 2 * idx, mask=live & dw1, other=0.0)   # BEFORE the store.
        prev_im = tl.load(w1 + 2 * idx + 1, mask=live & dw1, other=0.0)
        src_re = v1_re
        src_im = v1_im
        tl.store(w1 + 2 * idx, src_re, mask=live & dw1)
        tl.store(w1 + 2 * idx + 1, src_im, mask=live & dw1)
        a_re = tl.load(h1 + 2 * idx, mask=live & dw1, other=0.0)
        a_im = tl.load(h1 + 2 * idx + 1, mask=live & dw1, other=0.0)
        prev_re = tl.where(start, src_re, prev_re)    # MEEP's start of H
        prev_im = tl.where(start, src_im, prev_im)
        a_re = tl.where(start, src_re, a_re)
        a_im = tl.where(start, src_im, a_im)
        tl.store(h1 + 2 * idx, constitutive_form(a_re, prev_re, src_re, kp_1, km_1, dw1),
                 mask=live)
        tl.store(h1 + 2 * idx + 1,
                 constitutive_form(a_im, prev_im, src_im, kp_1, km_1, dw1), mask=live)

        # --- component 2 -------------------------------------------------------
        prev_re = tl.load(w2 + 2 * idx, mask=live & dw2, other=0.0)   # BEFORE the store.
        prev_im = tl.load(w2 + 2 * idx + 1, mask=live & dw2, other=0.0)
        src_re = v2_re
        src_im = v2_im
        tl.store(w2 + 2 * idx, src_re, mask=live & dw2)
        tl.store(w2 + 2 * idx + 1, src_im, mask=live & dw2)
        a_re = tl.load(h2 + 2 * idx, mask=live & dw2, other=0.0)
        a_im = tl.load(h2 + 2 * idx + 1, mask=live & dw2, other=0.0)
        prev_re = tl.where(start, src_re, prev_re)    # MEEP's start of H
        prev_im = tl.where(start, src_im, prev_im)
        a_re = tl.where(start, src_re, a_re)
        a_im = tl.where(start, src_im, a_im)
        tl.store(h2 + 2 * idx, constitutive_form(a_re, prev_re, src_re, kp_2, km_2, dw2),
                 mask=live)
        tl.store(h2 + 2 * idx + 1,
                 constitutive_form(a_im, prev_im, src_im, kp_2, km_2, dw2), mask=live)

else:  # pragma: no cover - laptop path
    complex_fused_curl_constitutive_B = None  # type: ignore[assignment]


def complex_fused_curl_constitutive_B_kernel() -> Any:
    """The JIT kernel, or a named ImportError on a host without Triton."""
    if complex_fused_curl_constitutive_B is None:
        raise ImportError(
            "the complex fused magnetic B/H kernel needs the optional `triton` "
            f"package (pip install triton). Original error: {_TRITON_IMPORT_ERROR}")
    return complex_fused_curl_constitutive_B


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def complex_fused_magnetic_pair_coverage(fields: Any, pml: Any,
                                         sources: Any = None) -> Coverage:
    """May ONE launch span complex ``step_B`` -> wall -> complex ``update_H``?

    A conjunction of the two halves' own predicates plus the seam clauses.
    Nothing is weakened: a configuration either half refuses is refused here with
    that half's reasons, prefixed so a reader can tell which side said it — the
    same construction :func:`.folded_fused_magnetic_pair.folded_fused_magnetic_pair_coverage`
    and :func:`.coverage.fused_pair_coverage` use.

    """
    reasons: List[str] = []

    curl = complex_pml_curl_coverage(fields, pml, CURL_SUB_STEP)
    if not curl.covered:
        reasons.extend(f"complex curl half: {reason}" for reason in curl.reasons)
    constitutive = complex_constitutive_coverage(fields, pml, CONSTITUTIVE_SIDE)
    if not constitutive.covered:
        reasons.extend(f"complex constitutive half: {reason}"
                       for reason in constitutive.reasons)

    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, tuple(dict.fromkeys(reasons + ["fields carries no grid"])))

    # THE SOURCE SEAM. A magnetic source is injected BETWEEN the two halves
    # (driver.py:3283-3284), so a fused pair would consume a pre-injection B. An
    # ELECTRIC source is injected in the D/E seam and does not disqualify this
    # pair. IGNORANCE IS NOT AN EMPTY SET: `Fields` does not hold the source list,
    # so a predicate that inferred "no magnetic source" from not being told would
    # be the over-covering refusal this clause exists to prevent. Measured: this
    # clause is what takes the arm from 16 corpus rows to 12.
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
    # `grid.has_symmetry()` (stepping.py:1481-1482, :1565-1566), and BOTH halves
    # already refuse a symmetry. RE-CHECKED HERE ANYWAY, not inferred: this
    # module's coverage may not be read off another module's guard, and if the
    # complex tranche ever admits a fold, this weld would silently swallow two
    # passes that had started doing work. The clause costs one attribute read and
    # is the difference between "no pass runs" and "no pass ran when I last
    # looked".
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
    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            reasons.append(
                f"grid does not expose {name}; zero_metal_B cannot be carried inline")

    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class ComplexFusedMagneticPairPlan:
    """One allocation-free launch for five of the driver's magnetic call sites.

    The complex volumes are bound as float32 WORD VIEWS at plan time, once, and
    ``n_elem`` stays the COMPLEX cell count — the same split
    :class:`.complex_fields.ComplexPmlCurlPlan` makes, for the same reason.
    """

    __slots__ = (
        "shape", "n_elem", "dtdx", "backward", "bc", "zero_metal", "connections",
        "connection_values", "block", "num_warps", "chunk_bounds",
        "_targets", "_aux", "_sources", "_curl_coefficients",
        "_h_targets", "_h_aux", "_h_coefficients", "_cb", "_grid", "_kernel",
        "_h_state",
    )

    #: The five driver call sites one launch performs. Declared, so a composition
    #: can be inspected rather than inferred from the curl slot's name.
    replaces = REPLACES

    def __init__(self, shape, dtdx: float, bc, zero_metal, connections,
                 connection_values, block: int,
                 targets, auxiliaries, sources, curl_coefficients,
                 h_targets, h_aux, h_coefficients,
                 kernel: Any = None, num_warps: Optional[int] = 1, *,
                 chunk_bounds: Any, h_state: Any = None) -> None:
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        # float(): the array path consumes the Python scalar at complex64
        # precision, and Triton types a Python float argument as fp32, so the two
        # are the same bits (complex_fields.ComplexPmlCurlPlan's measured note).
        self.dtdx = float(dtdx)
        self.backward = BACKWARD
        self.bc = tuple(int(code) for code in bc)
        self.zero_metal = tuple(bool(value) for value in zero_metal)
        self.connections = tuple(int(kind) for kind in connections)
        self.connection_values = tuple(float(value) for value in connection_values)
        self.block = int(block)
        self.num_warps = None if num_warps is None else int(num_warps)
        self._targets = tuple(CupyPointer(_word_view(a)) for a in targets)
        self._aux = tuple(CupyPointer(_word_view(a)) for a in auxiliaries)
        self._sources = tuple(CupyPointer(_word_view(a)) for a in sources)
        self._curl_coefficients = tuple(
            CupyPointer(_flat(a)) for a in curl_coefficients)
        self._h_targets = tuple(CupyPointer(_word_view(a)) for a in h_targets)
        self._h_aux = tuple(CupyPointer(_word_view(a)) for a in h_aux)
        self._h_coefficients = tuple(
            CupyPointer(_flat(a)) for a in h_coefficients)
        # THE INT32 WORD BOUND. The kernel addresses words as ``2 * idx`` in
        # int32, so the complex cell count must leave room for the doubling —
        # the same halving the complex predicates apply (module docstring of
        # complex_fields, "Word addressing"). Refused here as well because the
        # from-arrays route runs no predicate at all.
        if 2 * self.n_elem >= 2 ** 31:
            raise ValueError(
                f"{self.n_elem} complex cells needs {2 * self.n_elem} int32 word "
                f"indices, which overflows the kernel's 2 * idx addressing")
        from .launch import bind_chunk_bounds  # noqa: PLC0415

        self.chunk_bounds, self._cb = bind_chunk_bounds(
            chunk_bounds, self.shape, targets[0], "ComplexFusedMagneticPairPlan")
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        # An override for exactly one caller: the gate's mutation legs, which
        # compile deliberately broken copies of this kernel. Dropping it is not a
        # slowdown, it is a DISARMING — every mutation leg would then launch the
        # shipped kernel and report the defect as uncaught.
        self._kernel = kernel
        # The Fields the H half starts from (MEEP's start of H), or None: the
        # contract of launch.ConstitutivePlan's h_state.
        self._h_state = h_state

    def start_h(self) -> int:
        """1 on the one launch that starts H from B, 0 otherwise."""
        state = self._h_state
        return 0 if state is None or state.h_separated else 1

    def run(self, guard: Optional[bool] = None) -> None:
        """Launch the fused pair, in place. Same ``guard`` contract as every plan."""
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        kernel = (self._kernel if self._kernel is not None
                  else complex_fused_curl_constitutive_B_kernel())
        nx, ny, nz = self.shape
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        start = self.start_h()
        kernel[self._grid](
            *self._targets, *self._aux, *self._sources, *self._curl_coefficients,
            *self._h_targets, *self._h_aux, *self._h_coefficients, self._cb,
            nx, ny, nz, self.n_elem, self.dtdx, start,
            *self.connection_values,
            BACKWARD=self.backward,
            BCX=self.bc[0], BCY=self.bc[1], BCZ=self.bc[2],
            ZM_X=self.zero_metal[0], ZM_Y=self.zero_metal[1],
            ZM_Z=self.zero_metal[2],
            CNX=self.connections[0], CNY=self.connections[1],
            CNZ=self.connections[2],
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )
        if start and self._grid[0]:  # an empty grid (the warm pass) starts nothing
            self._h_state.h_separated = True

    def __repr__(self) -> str:
        return (f"ComplexFusedMagneticPairPlan(shape={self.shape}, bc={self.bc}, "
                f"zero_metal={self.zero_metal}, connections={self.connections}, "
                f"block={self.block}, num_warps={self.num_warps}, "
                f"chunk_bounds={self.chunk_bounds.tolist()})")


def plan_complex_fused_magnetic_pair(fields: Any, pml: Any, sources: Any = None,
                                     block: Optional[int] = None,
                                     num_warps: Optional[int] = 1,
                                     kernel: Any = None,
                                     chunk_layout: Any = None,
                                     ) -> Optional[ComplexFusedMagneticPairPlan]:
    """Build the plan from the engine's own objects, or ``None`` when refused.

    ``None`` is the only refusal: a configuration this kernel does not carry must
    fall back to the array path, never raise into a caller that would otherwise
    have stepped correctly. ``chunk_layout`` is the run's chunk division, ``None`` for
    the one-rank default (``launch.chunk_bounds_for``).
    """
    if not complex_fused_magnetic_pair_coverage(fields, pml, sources).covered:
        return None
    from ..stepping import _boundary_kinds as resolve  # noqa: PLC0415
    from .launch import chunk_bounds_for  # noqa: PLC0415

    curl_spec = SUB_STEPS[CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    grid = fields.grid
    kinds = resolve(grid, pml)
    connections, values = connection_arguments(bloch_phase_table(grid, kinds),
                                               backward=bool(curl_spec["backward"]))
    return ComplexFusedMagneticPairPlan(
        grid.shape, grid.dt / grid.dx,
        [1 if kind == "metallic" else 0 for kind in kinds],
        zero_metal_axes(grid), connections, values,
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
        kernel=kernel, num_warps=num_warps,
        chunk_bounds=chunk_bounds_for(fields, pml, chunk_layout), h_state=fields,
    )


def plan_complex_fused_magnetic_pair_from_arrays(
        arrays: Dict[str, Any], curl_flat: Dict[str, Any],
        constitutive_flat: Dict[str, Any], codes, zero_metal,
        phases: Sequence[Optional[complex]], dtdx: float,
        block: Optional[int] = None, kernel: Any = None,
        num_warps: Optional[int] = 1, *, chunk_bounds: Any) -> ComplexFusedMagneticPairPlan:
    """Build from bare device arrays — the gate's route.

    No coverage predicate runs: the caller is a harness that constructed the
    configuration deliberately, including the deliberately wrong ones. ``phases``
    is the per-axis ``Optional[complex]`` table (None = unphased) and the
    conjugation for a backward sub-step is applied HERE, exactly as the engine
    route applies it — this product is forward-only, so it never is.
    """
    curl_spec = SUB_STEPS[CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    shape = tuple(int(n) for n in arrays[curl_spec["targets"][0]].shape)
    connections, values = connection_arguments(tuple(phases),
                                               backward=bool(curl_spec["backward"]))
    return ComplexFusedMagneticPairPlan(
        shape, dtdx, codes, zero_metal, connections, values,
        DEFAULT_BLOCK if block is None else block,
        [arrays[name] for name in curl_spec["targets"]],
        [arrays["fu_" + name] for name in curl_spec["targets"]],
        [arrays[name] for name in curl_spec["sources"]],
        [curl_flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
        [arrays[name] for name in side_spec["targets"]],
        [arrays[name] for name in side_spec["aux"]],
        [constitutive_flat[f"{stem}_{axis}"]
         for axis in "xyz" for stem in ("kps", "kms")],
        kernel=kernel, num_warps=num_warps, chunk_bounds=chunk_bounds,
    )
