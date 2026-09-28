"""The BFAST ELECTRIC seam in one launch: BFAST ``step_D`` welded into ``update_E``.

The ``BFAST PML`` -> ``BFAST run`` cell's D->E product, and the ELECTRIC TWIN of
:mod:`.bfast_fused_magnetic_pair`. Both halves are already certified and neither is
re-derived here:

* the curl half is :func:`.bfast_curl.bfast_pml_curl_step` — itself
  ``kernels.pml_curl_step``'s certified body plus ONE constexpr-gated tail — taken
  through its ``BACKWARD == 1`` arm, which that body already carries and which its
  magnetic twin binds to 0 and never compiles;
* the constitutive half is ``kernels.constitutive_step``'s ``SCALE = 1`` arm,
  admitted on a BFAST run by the shipped
  :func:`.bfast_curl.bfast_run_constitutive_coverage` ("the constitutive sub-steps
  read nothing bfast-dependent — fields.py:227-230, the pass is on step_db only");
* and the WELD is the one substitution :func:`.kernels.fused_curl_constitutive_D`
  already makes between exactly those two bodies on a BFAST-inactive run.

SO THIS KERNEL IS ``fused_curl_constitutive_D``, STATEMENT FOR STATEMENT, PLUS THE
SAME ``if HAS_BFAST:`` BLOCK ``bfast_pml_curl_step`` ADDS TO ``pml_curl_step``. A
reader must be able to diff this body against those two and see nothing else moved;
the gate's transcription leg asserts exactly that, as two statement-list equalities
against both shipped sources.

===========================================================================
WHY THIS CELL WAS UNREACHABLE AND IS NOW WORTH ONE
===========================================================================

The cell's single corpus row is
``tests:TestReflectanceAngular.test_reflectance_angular_2_35_7`` — shape
(1, 1, 1800), all-periodic, ``has_metallic == false``, ``bfast_active == true`` — and
it declares ``source_field_types == ['D']``. That injection lands BETWEEN the two
halves (driver.py:3294-3299), which is exactly what made the D->E cell unreachable
before :mod:`..deposit_repair` went live on 2026-08-28. Measured on the 2026-08-31
board (``results/fusion_matrix_triton_2026-08-31_folded_disp``)::

    ceiling  1  (rows 1)  D->E  BFAST PML -> BFAST run   NO FUSED PRODUCT

with ``in_seam_source: true`` and ``in_seam_source_blocks: false``. So
:data:`CARRIES_DEPOSIT_REPAIR` is not one clause of this product — it IS the
product: at False the source clause refuses the only row the cell has and the arm
admits ZERO. The MAGNETIC twin keeps the flag at False for the mirror-image reason:
on ITS seam that same row is electric-only.

===========================================================================
THE TAIL, AND WHERE IT GOES
===========================================================================

``stepping._bfast_term`` (S:896-904) adds MEEP's BFAST second additive pass to the
curl. The array path folds it in AFTER the ``dtdx`` curl and BEFORE the ownership
mask (stepping.py:475-478 after :458, before :479 on the D side), and that is where
it sits here. Transcribed from :func:`.bfast_curl.bfast_pml_curl_step`, not
re-derived:

* **the operands are the loads the curl already made.** BFAST SUMS the same
  shifted/center pairs the curl DIFFERENCES (stepping.py:1594-1598 — the shared
  gather is the engine's deliberate advantage over MEEP's two loops);
* per target, ``total = k1*(g1s + g1c) - k2*(g2s + g2c)`` and
  ``adv = total - 2.0*state``;
* ``adv`` is masked by the SAME owned-cell predicate the curl is masked by, and
  masked BEFORE the state store (S:902) — which is why ``at_x``/``at_y``/``at_z``
  are computed one statement earlier here than in
  ``kernels.fused_curl_constitutive_D``, exactly as ``bfast_pml_curl_step``
  computes them one statement earlier than ``pml_curl_step`` does;
* ``state <- state + adv`` in place, then ``curl <- curl - adv`` — the
  caller-subtracts sign convention (S:904), spelled as IEEE subtraction, never as a
  Triton unary minus;
* NO ``dtdx`` anywhere in the tail (S:836-837);
* ALL THREE targets always run the tail (grid.py:744-753). With seeded state the
  zero-k advance ``-2*state`` is byte-visible, so no arm skips it.

**THE D SIDE'S ``k1``/``k2`` ARE THE NEGATED PAIR, and that IS load-bearing here.**
:func:`.bfast_curl.bfast_curl_coefficients` negates both in host float64 on the
electric side (``bfast_curl.py:350-351``, MEEP's
``if (ft == D_stuff) { k1 = -k1; k2 = -k2; }`` at S:886-887) AND reads a different
term table (``BFAST_TERMS['step_D']``), so a builder that copied its twin's
``magnetic=True`` would hand this product the wrong six words. That is the reverse
of the beta family, where the flag is inert on real storage, and the host suite
pins the difference by measurement rather than by assertion.

THE THREE STATE POINTERS ARE THE ENGINE'S OWN ``f_bfast_D*`` ARRAYS, bound
pointer-identically. That is not an optimisation: the driver's flux backup/restore
mutates them in place (driver.py:4126-4135), so a copy would desynchronise the run.

``HAS_BFAST`` IS CARRIED AND MUST BE 1 on any configuration this product's predicate
admits — the BFAST family's clause 11 is INVERTED inside ``bfast_pml_curl_coverage``
(``grid.bfast_active`` required), so a BFAST-inactive run is the ordinary
:func:`.coverage.fused_pair_coverage`'s and is refused here by name. It is carried
rather than hard-coded for ``bfast_pml_curl_step``'s own reason: it keeps the curl
half a verbatim copy, and it gives the gate a certified-kernel arm
(``HAS_BFAST = 0`` must reproduce ``kernels.fused_curl_constitutive_D`` bit for
bit).

===========================================================================
THE FIVE PASSES IN THE SEAM
===========================================================================

``driver.step`` runs five passes between the two halves (driver.py:3292-3304)::

    step_D -> ELECTRIC SOURCES -> fill_symmetry_bc_D -> zero_metal_D
           -> fill_folded_far_ghosts_D -> update_E

* **the electric sources — CARRIED, through the SHIPPED deposit repair.**
  :class:`.deposit_repair.LeadingRepairPlan` saves the two stateful arrays at the
  deposit points before the launch and :class:`.deposit_repair.TrailingRepairPlan`
  recomputes them after, in the driver's own order. IGNORANCE IS STILL NEVER AN
  EMPTY SET: an undeclared ``sources`` is a REFUSAL, and a source that cannot
  publish the index it writes is refused by name;
* **``fill_symmetry_bc_D`` and ``fill_folded_far_ghosts_D`` — PROVABLY NO-OPS
  HERE**, not carried and not refused. Both return at their first line unless
  ``grid.has_symmetry()`` (stepping.py:1481-1482, :1565-1566), and the BFAST family
  refuses every fold. THE PREDICATE RE-CHECKS IT ANYWAY rather than reading its own
  coverage off another module's guard;
* **``zero_metal_D`` — CARRIED INLINE** (driver.py:3301), through
  :func:`.coverage.zero_metal_axes`. SIX ROWS on this side against the magnetic
  twin's three — each axis clears the two TANGENTIAL D components — which is
  ``fused_curl_constitutive_D``'s own block and is byte-copied from it.

The corpus row is all-periodic with ``has_metallic == false``, so the inline
``zero_metal_D`` carry compiles to three ``False`` flags ON THAT ROW; the gate
scores every wall-clear mutation on a WALLED case instead.

NOT WIRED, and not an arm. ``launch.plan_step`` assigns at most one plan per
``STEP_ORDER`` slot and this product spans FIVE driver call sites. Wired, it would
also contend for ``step_D`` with the ``BFAST PML`` arm, which IS wired and admits
the same configurations — and ``_select_slot`` (launch.py:1983-1996) leaves a slot
with two admitters UNSELECTED, taking the certified curl off the device with it.
Nothing in ``launch.py`` names this module; ``fastpath.plan_fast_path`` is
unchanged.

Import contract: importable WITHOUT Triton — the predicate and the plan builder (to
``None``) must answer on the laptop that is the merge bar.

DEVICE STATUS: see :data:`DEVICE_STATUS`. A weld licenses a claim, not a dispatch.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .bfast_curl import (
    BFAST_STATE_NAMES,
    bfast_curl_coefficients,
    bfast_pml_curl_coverage,
    bfast_run_constitutive_coverage,
)
from .coverage import (
    CONSTITUTIVE_SIDES,
    Coverage,
    _call,
    zero_metal_axes,
)
from .. import deposit_repair as _deposit_repair

#: Does this product bracket its fused launch with the deposit repair? TRUE, and on
#: THIS cell it is the difference between one seam-instance and zero: the single
#: corpus row (``tests:TestReflectanceAngular.test_reflectance_angular_2_35_7``)
#: declares an ELECTRIC source, which the driver injects between the two halves
#: (driver.py:3294-3299). The flag is a claim about the PLAN this module builds --
#: that the leading slot saves and the trailing slot restores -- and is only ever
#: changed in the same edit as that wiring. Its MAGNETIC twin keeps the flag at
#: False for the mirror-image reason: on the B seam that same row is electric-only.
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

#: ``SUB_STEPS['step_D']['backward']``, restated so the kernel's one legal binding
#: is visible without reading :mod:`launch`; the test suite pins the two equal.
BACKWARD = 1

#: The FIVE driver call sites ONE launch of this plan performs, in driver order
#: (driver.py:3292-3304). Declared, never inferred from the slot name. Only TWO of
#: them do work on an admitted configuration — the two symmetry fills return at
#: their first line without a mirror plane, which this family refuses outright — and
#: the inert two are listed anyway: what a launch REPLACES is what the driver would
#: otherwise have called. THE INJECTION IS NOT ONE OF THE FIVE: it is carried by the
#: deposit repair bracket around the launch, not by the launch.
REPLACES: Tuple[str, ...] = ("step_D", "fill_symmetry_bc_D", "zero_metal_D",
                             "fill_folded_far_ghosts_D", "update_E")

#: What has and has not been executed on a device. Edited only by a released gate.
DEVICE_STATUS: str = (
    "RELEASED 2026-08-31 on the GPU host GPU 6 (verified physically empty before the "
    "run), RE-RUN in place after this paragraph was written -- the gate pins this "
    "module in source_sha256, so the run that certifies it has to be the one that "
    "saw the paragraph. Record: "
    "parity/meep_gpu/results/triton_bfast_electric_2026-08-31e/"
    "bfast_fused_electric_pair/gate.json, subnormal policy "
    "'ieee_keep_ftz_stripped', sm_86, CuPy 13.5.1. "
    "WHAT IT MEASURED: 4 no-device legs (transcription at STATEMENT grain, "
    "equivalence, corpus admission, mutation pairing) and 25 device legs -- 12 quiet, "
    "6 carry, 2 null control, 3 reduction, 2 armed -- each comparing the uint32 view "
    "of EVERY allocated volume after a COMPLETE driver.step(), against BOTH the CuPy "
    "array path AND the separately certified products this weld replaces. No "
    "`allclose` appears anywhere in that gate. A CountingKernel proves the fused "
    "kernel is what executed (launch counts recorded per leg), and every carry leg "
    "has a NULL CONTROL with the deposit bracket removed that MUST diverge and does. "
    "19 of 19 kernel mutations and 4 of 4 host mutations were caught; 5 refusals "
    "hold by name. "
    "THE REDUCTION LEG IS THE TRANSCRIPTION CLAIM ON SILICON: at HAS_BFAST = 0 this "
    "kernel is bit-identical to the shipped kernels.fused_curl_constitutive_D over "
    "the same volumes, while BOTH differ from the array path -- so the flag is read "
    "and the agreement is not a tautology. Its floor is two-sided (the BFAST state "
    "must be INERT here and MOVING on the array path); an earlier absolute "
    "'every array moved' floor asserted the opposite of the leg's own claim and "
    "failed run c on a correct kernel. "
    "WHAT IT DOES NOT LICENSE: nothing here is timed, and no throughput, dispatch or "
    "end-to-end claim follows from it. A weld licenses a claim, not a dispatch.")

__all__ = [
    "BACKWARD", "CARRIES_DEPOSIT_REPAIR", "CONSTITUTIVE_SIDE", "CURL_SUB_STEP",
    "DEVICE_STATUS", "REPLACES",
    "BfastFusedElectricPairPlan",
    "bfast_fused_curl_constitutive_D",
    "bfast_fused_curl_constitutive_D_kernel",
    "bfast_fused_electric_pair_coverage",
    "plan_bfast_fused_electric_pair",
    "plan_bfast_fused_electric_pair_from_arrays",
]


if triton is not None:

    # The same constexpr code as ``kernels.METALLIC``, restated for the reason every
    # sibling restates it — importing a ``tl.constexpr`` wrapper and re-wrapping it
    # is not the same object, and the comparison in the kernel body is against a
    # literal code.
    METALLIC = tl.constexpr(1)

    @triton.jit
    def bfast_fused_curl_constitutive_D(
        f0, f1, f2,                       # curl targets: Dx,Dy,Dz
        u0, u1, u2,                       # curl auxiliaries: fu_Dx,fu_Dy,fu_Dz
        g0, g1, g2,                       # curl sources: Hx,Hy,Hz
        s0, s1, s2,                       # f_bfast_D* IIR states, one per target
        kmx, sinvx, kmy, sinvy, kmz, sinvz,   # curl kms/sinv, INTEGER lattice
        e0, e1, e2,                       # constitutive targets: Ex,Ey,Ez
        w0, w1, w2,                       # constitutive aux: f_w_Ex,f_w_Ey,f_w_Ez
        ie0, ie1, ie2,                    # diagonal inverse epsilon volumes
        kp0, km0, kp1, km1, kp2, km2,     # constitutive kps/kms, HALF-INTEGER
        nx, ny, nz, n_elem, dtdx,
        k1_0, k2_0, k1_1, k2_1, k1_2, k2_2,   # f32 (k1,k2) per target, host-rounded
        BACKWARD: tl.constexpr,           # bound to 1 by every builder; see below
        BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
        ZM_X: tl.constexpr, ZM_Y: tl.constexpr, ZM_Z: tl.constexpr,
        HAS_BFAST: tl.constexpr,          # 1 on every admitted configuration
        BLOCK: tl.constexpr,
    ):
        """BFAST ``step_D`` + ``zero_metal_D`` + ``update_E``, one launch.

        ``kernels.fused_curl_constitutive_D``'s body with
        ``bfast_curl.bfast_pml_curl_step``'s ``HAS_BFAST`` tail in the one place the
        array path folds it (S:446-449 after :429, before :450). Nothing else moves.

        ``BACKWARD`` IS CARRIED AND MUST BE 1. It keeps the curl half a verbatim copy
        rather than a hand-specialised one. It cannot be 0: the wall clear below is
        the D family's six rows, the constitutive half is the E side with its
        inverse-permittivity multiply, and the ``k1``/``k2`` this side is passed are
        the NEGATED pair off a different term table.

        ``HAS_BFAST`` MUST BE 1 on any admitted configuration — the BFAST family's
        clause 11 is inverted and requires ``grid.bfast_active``. It is carried
        rather than hard-coded so the gate has a certified-kernel arm: at
        ``HAS_BFAST = 0`` this body must reproduce
        :func:`.kernels.fused_curl_constitutive_D` bit for bit.

        ``ZM_X``/``ZM_Y``/``ZM_Z`` are ``coverage.zero_metal_axes`` — the question
        ``stepping._zero_metal`` asks (the grid's own declaration), NOT the resolved
        ghost rule.
        """
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = idx < n_elem
        nyz = ny * nz
        k = idx % nz
        plane = idx // nz
        j = plane % ny
        i = plane // ny

        # ======================= the curl half ================================
        # Verbatim from bfast_curl.bfast_pml_curl_step, which is itself
        # kernels.pml_curl_step plus the HAS_BFAST tail.
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

        a = tl.load(g0 + idx, mask=live, other=0.0)
        b = tl.load(g1 + idx, mask=live, other=0.0)
        c = tl.load(g2 + idx, mask=live, other=0.0)
        a_y = tl.load(g0 + oy, mask=vy, other=0.0)
        a_z = tl.load(g0 + oz, mask=vz, other=0.0)
        b_x = tl.load(g1 + ox, mask=vx, other=0.0)
        b_z = tl.load(g1 + oz, mask=vz, other=0.0)
        c_x = tl.load(g2 + ox, mask=vx, other=0.0)
        c_y = tl.load(g2 + oy, mask=vy, other=0.0)

        curl0 = dtdx * ((c_y - c) + (b - b_z))
        curl1 = dtdx * ((a_z - a) + (c - c_x))
        curl2 = dtdx * ((b_x - b) + (a - a_y))

        # The ownership predicates, shared by the BFAST advance mask and the curl
        # mask below (stepping._mask_non_owned_cells — the SAME predicate, applied
        # to the advance at :902 and to the summed curl at :369/:450). COMPUTED ONE
        # STATEMENT EARLIER than kernels.fused_curl_constitutive_D computes it, which
        # is exactly how bfast_pml_curl_step differs from pml_curl_step.
        at_x, at_y, at_z = i == 0, j == 0, k == 0

        # --- the BFAST tail (stepping._bfast_term :896-904), SHARED operands ----
        # AFTER the dtdx curl, BEFORE the ownership mask — the array path's fold
        # order. The sums pair each SHIFTED load with its center in the operand
        # order of :896-897; no dtdx (:836-837); no unary minus anywhere.
        if HAS_BFAST:
            st0 = tl.load(s0 + idx, mask=live, other=0.0)
            st1 = tl.load(s1 + idx, mask=live, other=0.0)
            st2 = tl.load(s2 + idx, mask=live, other=0.0)
            total0 = (k1_0 * (c_y + c)) - (k2_0 * (b_z + b))
            total1 = (k1_1 * (a_z + a)) - (k2_1 * (c_x + c))
            total2 = (k1_2 * (b_x + b)) - (k2_2 * (a_y + a))
            adv0 = total0 - (2.0 * st0)
            adv1 = total1 - (2.0 * st1)
            adv2 = total2 - (2.0 * st2)
            # --- advance ownership mask, BEFORE the state store (S:902) ---------
            if BACKWARD:
                if BCY == METALLIC:
                    adv0 = tl.where(at_y, 0.0, adv0)
                if BCZ == METALLIC:
                    adv0 = tl.where(at_z, 0.0, adv0)
                if BCX == METALLIC:
                    adv1 = tl.where(at_x, 0.0, adv1)
                if BCZ == METALLIC:
                    adv1 = tl.where(at_z, 0.0, adv1)
                if BCX == METALLIC:
                    adv2 = tl.where(at_x, 0.0, adv2)
                if BCY == METALLIC:
                    adv2 = tl.where(at_y, 0.0, adv2)
            else:
                if BCX == METALLIC:
                    adv0 = tl.where(at_x, 0.0, adv0)
                if BCY == METALLIC:
                    adv1 = tl.where(at_y, 0.0, adv1)
                if BCZ == METALLIC:
                    adv2 = tl.where(at_z, 0.0, adv2)
            tl.store(s0 + idx, st0 + adv0, mask=live)
            tl.store(s1 + idx, st1 + adv1, mask=live)
            tl.store(s2 + idx, st2 + adv2, mask=live)
            curl0 = curl0 - adv0
            curl1 = curl1 - adv1
            curl2 = curl2 - adv2

        # --- ownership mask (stepping._mask_non_owned_cells) --------------------
        if BACKWARD:
            if BCY == METALLIC:
                curl0 = tl.where(at_y, 0.0, curl0)
            if BCZ == METALLIC:
                curl0 = tl.where(at_z, 0.0, curl0)
            if BCX == METALLIC:
                curl1 = tl.where(at_x, 0.0, curl1)
            if BCZ == METALLIC:
                curl1 = tl.where(at_z, 0.0, curl1)
            if BCX == METALLIC:
                curl2 = tl.where(at_x, 0.0, curl2)
            if BCY == METALLIC:
                curl2 = tl.where(at_y, 0.0, curl2)
        else:
            if BCX == METALLIC:
                curl0 = tl.where(at_x, 0.0, curl0)
            if BCY == METALLIC:
                curl1 = tl.where(at_y, 0.0, curl1)
            if BCZ == METALLIC:
                curl2 = tl.where(at_z, 0.0, curl2)

        # --- split-field recurrence (stepping._apply_pml_update) ---------------
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

        # --- the seam: stepping.zero_metal_D (driver.py:3301) ------------------
        # SIX rows against the magnetic twin's three: each axis clears the two
        # TANGENTIAL D components. Applied to the REGISTER, before both the store
        # and the constitutive read. Byte-copied from
        # kernels.fused_curl_constitutive_D.
        if ZM_X:
            v1 = tl.where(at_x, 0.0, v1)
            v2 = tl.where(at_x, 0.0, v2)
        if ZM_Y:
            v0 = tl.where(at_y, 0.0, v0)
            v2 = tl.where(at_y, 0.0, v2)
        if ZM_Z:
            v0 = tl.where(at_z, 0.0, v0)
            v1 = tl.where(at_z, 0.0, v1)

        tl.store(u0 + idx, n0, mask=live)
        tl.store(u1 + idx, n1, mask=live)
        tl.store(u2 + idx, n2, mask=live)
        tl.store(f0 + idx, v0, mask=live)
        tl.store(f1 + idx, v1, mask=live)
        tl.store(f2 + idx, v2, mask=live)

        # ==================== the constitutive half ===========================
        # Verbatim from kernels.fused_curl_constitutive_D. Component 0 takes its
        # coefficient from axis x, 1 from y, 2 from z (stepping.E_CONSTITUTIVE_TERMS
        # :227) — the component's OWN axis. D IS ON THE LEFT of the inverse-epsilon
        # multiply because the array path writes
        # `source * inverse_epsilon_for(component)` (stepping.py:1011-1013).
        kp_0 = tl.load(kp0 + i, mask=live, other=0.0)
        km_0 = tl.load(km0 + i, mask=live, other=0.0)
        kp_1 = tl.load(kp1 + j, mask=live, other=0.0)
        km_1 = tl.load(km1 + j, mask=live, other=0.0)
        kp_2 = tl.load(kp2 + k, mask=live, other=0.0)
        km_2 = tl.load(km2 + k, mask=live, other=0.0)

        prev0 = tl.load(w0 + idx, mask=live, other=0.0)
        src0 = v0 * tl.load(ie0 + idx, mask=live, other=0.0)
        tl.store(w0 + idx, src0, mask=live)
        a0 = tl.load(e0 + idx, mask=live, other=0.0)
        a0 = a0 + kp_0 * src0
        a0 = a0 - km_0 * prev0
        tl.store(e0 + idx, a0, mask=live)

        prev1 = tl.load(w1 + idx, mask=live, other=0.0)
        src1 = v1 * tl.load(ie1 + idx, mask=live, other=0.0)
        tl.store(w1 + idx, src1, mask=live)
        a1 = tl.load(e1 + idx, mask=live, other=0.0)
        a1 = a1 + kp_1 * src1
        a1 = a1 - km_1 * prev1
        tl.store(e1 + idx, a1, mask=live)

        prev2 = tl.load(w2 + idx, mask=live, other=0.0)
        src2 = v2 * tl.load(ie2 + idx, mask=live, other=0.0)
        tl.store(w2 + idx, src2, mask=live)
        a2 = tl.load(e2 + idx, mask=live, other=0.0)
        a2 = a2 + kp_2 * src2
        a2 = a2 - km_2 * prev2
        tl.store(e2 + idx, a2, mask=live)

else:  # pragma: no cover - laptop path
    bfast_fused_curl_constitutive_D = None  # type: ignore[assignment]


def bfast_fused_curl_constitutive_D_kernel() -> Any:
    """The JIT kernel, or a named ImportError on a host without Triton."""
    if bfast_fused_curl_constitutive_D is None:
        raise ImportError(
            "the BFAST fused electric D/E kernel needs the optional `triton` "
            f"package (pip install triton). Original error: {_TRITON_IMPORT_ERROR}")
    return bfast_fused_curl_constitutive_D


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def bfast_fused_electric_pair_coverage(fields: Any, pml: Any,
                                       sources: Any = None) -> Coverage:
    """May ONE launch span BFAST ``step_D`` -> wall -> ``update_E``?

    A conjunction of the two halves' own SHIPPED predicates plus the seam clauses.
    Nothing is weakened: a configuration either half refuses is refused here with
    that half's reasons, prefixed so a reader can tell which side said it.

    The two halves are the arms ``launch.plan_step`` really selects on a BFAST row
    today (``BFAST PML`` on ``step_D``, ``BFAST run`` on ``update_E``), so this
    predicate is narrower than the pair the planner already puts on the device,
    never wider. The BFAST family's clause 11 is INVERTED inside
    ``bfast_pml_curl_coverage`` itself, so it needs no rung here: a BFAST-inactive
    row fails both halves together.
    """
    reasons: List[str] = []

    curl = bfast_pml_curl_coverage(fields, pml, CURL_SUB_STEP)
    if not curl.covered:
        reasons.extend(f"BFAST curl half: {reason}" for reason in curl.reasons)
    constitutive = bfast_run_constitutive_coverage(fields, pml, CONSTITUTIVE_SIDE)
    if not constitutive.covered:
        reasons.extend(f"BFAST constitutive half: {reason}"
                       for reason in constitutive.reasons)

    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False,
                        tuple(dict.fromkeys(reasons + ["fields carries no grid"])))

    # THE SOURCE SEAM, and on this cell it is the clause that decides everything. An
    # ELECTRIC source is injected BETWEEN the two halves (driver.py:3294-3299), so a
    # fused pair would consume a pre-injection D. The one corpus row of this cell
    # declares exactly that, so with CARRIES_DEPOSIT_REPAIR False the arm would admit
    # NOTHING; it is True, and the plan below brackets its launch with the repair.
    # IGNORANCE IS NEVER AN EMPTY SET. A MAGNETIC source is injected in the B/H half
    # and does not disqualify this pair.
    reasons.extend(_deposit_repair.seam_source_reasons(
        fields, sources, 'D',
        undeclared=(
            "the source set was not declared: this predicate cannot infer an empty "
            "electric source seam from Fields"
        ),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) is electric: the "
            f"driver injects it BETWEEN step_D and update_E "
            f"(driver.py:3294), which is work inside the seam this kernel "
            f"closes"),
        carries_repair=CARRIES_DEPOSIT_REPAIR))

    # THE TWO SYMMETRY PASSES, re-checked rather than read off another module's
    # guard: if the BFAST tranche ever admits a fold this weld would silently swallow
    # two passes that had started doing work.
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

    # THE WALL CLEAR is carried inline, so the grid must be able to answer which axes
    # are walled; an unanswerable one compiles to ZM=False and silently skips a plane
    # the array path clears.
    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            reasons.append(
                f"grid does not expose {name}; zero_metal_D cannot be carried inline")

    # THE INVARIANCE TABLE. The six k1/k2 words are gated on ``grid.is_invariant``
    # (bfast_curl_coefficients, grid.py:1165-1183 — the grid's OWN declared
    # dimensionality, never a shape test), so a grid that cannot answer it would
    # silently be given the un-gated coefficients.
    if getattr(grid, "is_invariant", None) is None:
        reasons.append(
            "grid does not expose is_invariant; the BFAST k1/k2 gating "
            "(bfast_curl_coefficients, grid.py:1165-1183) cannot be resolved")

    # THE INVERSE-EPSILON VOLUMES ARE READ BY THE KERNEL, so `Fields` must be able to
    # hand one over per E component. An absent accessor would be a TypeError inside
    # the builder rather than a refusal. The MAGNETIC twin needs no such clause:
    # update_H is `H = B` with mu = 1 baked in and binds no inverse volume at all.
    if getattr(fields, "inverse_epsilon_for", None) is None:
        reasons.append(
            "fields does not expose inverse_epsilon_for; the constitutive half "
            "cannot bind the inv_eps volumes it loads")

    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class BfastFusedElectricPairPlan:
    """One allocation-free launch for five of the driver's electric call sites.

    :class:`.launch.FusedPairPlan`'s ``pair='D'`` bindings plus the three state
    pointers, the six host-rounded scalars and the ``HAS_BFAST`` constexpr — which
    is exactly the delta :class:`.bfast_curl.BfastPmlCurlPlan` carries over
    :class:`.launch.PmlCurlPlan`. The inverse-epsilon volumes are the E side's and
    are bound here; the magnetic twin has none.
    """

    __slots__ = ("shape", "n_elem", "dtdx", "ks", "has_bfast", "backward", "bc",
                 "zero_metal", "block", "num_warps",
                 "_targets", "_aux", "_sources", "_states", "_curl_coefficients",
                 "_e_targets", "_e_aux", "_inverse_epsilon", "_e_coefficients",
                 "_grid", "_kernel")

    #: The five driver call sites one launch performs. Declared, so a composition can
    #: be inspected rather than inferred from the curl slot's name.
    replaces = REPLACES

    def __init__(self, shape, dtdx: float, bc, zero_metal, ks, block: int,
                 targets, auxiliaries, sources, states, curl_coefficients,
                 e_targets, e_aux, inverse_epsilon, e_coefficients,
                 kernel: Any = None, num_warps: Optional[int] = 1,
                 has_bfast: int = 1) -> None:
        from .launch import CupyPointer, _flat  # noqa: PLC0415

        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.dtdx = float(dtdx)
        # Already f32-rounded by the host (bfast_curl.bfast_curl_coefficients);
        # float() keeps the bits, and Triton types a Python float argument as fp32.
        self.ks = tuple(float(value) for value in ks)
        if len(self.ks) != 6:
            raise ValueError(f"ks must be the six (k1,k2) scalars, got {ks!r}")
        self.has_bfast = int(has_bfast)
        self.backward = BACKWARD
        self.bc = tuple(int(code) for code in bc)
        self.zero_metal = tuple(1 if flag else 0 for flag in zero_metal)
        self.block = int(block)
        # ONE WARP, the measured default for a fused pair
        # (launch.FUSED_DEFAULT_NUM_WARPS). Explicit None remains "take Triton's
        # default".
        self.num_warps = None if num_warps is None else int(num_warps)
        self._targets = tuple(CupyPointer(a) for a in targets)
        self._aux = tuple(CupyPointer(a) for a in auxiliaries)
        self._sources = tuple(CupyPointer(a) for a in sources)
        self._states = tuple(CupyPointer(a) for a in states)
        self._curl_coefficients = tuple(
            CupyPointer(_flat(a)) for a in curl_coefficients)
        self._e_targets = tuple(CupyPointer(a) for a in e_targets)
        self._e_aux = tuple(CupyPointer(a) for a in e_aux)
        if inverse_epsilon is None:
            raise ValueError(
                "the fused pair loads inv_eps on every launch; a placeholder "
                "binding would be read as a coefficient")
        self._inverse_epsilon = tuple(CupyPointer(a) for a in inverse_epsilon)
        self._e_coefficients = tuple(CupyPointer(_flat(a)) for a in e_coefficients)
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        # An override for exactly one caller: the gate's mutation legs.
        self._kernel = kernel

    def run(self, guard: Optional[bool] = None) -> None:
        """Launch the fused pair, in place. Same ``guard`` contract as every plan."""
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        kernel = (self._kernel if self._kernel is not None
                  else bfast_fused_curl_constitutive_D_kernel())
        nx, ny, nz = self.shape
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        kernel[self._grid](
            *self._targets, *self._aux, *self._sources, *self._states,
            *self._curl_coefficients,
            *self._e_targets, *self._e_aux, *self._inverse_epsilon,
            *self._e_coefficients,
            nx, ny, nz, self.n_elem, self.dtdx,
            *self.ks,
            BACKWARD=self.backward,
            BCX=self.bc[0], BCY=self.bc[1], BCZ=self.bc[2],
            ZM_X=self.zero_metal[0], ZM_Y=self.zero_metal[1],
            ZM_Z=self.zero_metal[2],
            HAS_BFAST=self.has_bfast,
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )

    def __repr__(self) -> str:
        return (f"BfastFusedElectricPairPlan(shape={self.shape}, bc={self.bc}, "
                f"zero_metal={self.zero_metal}, ks={self.ks!r}, "
                f"has_bfast={self.has_bfast}, block={self.block}, "
                f"num_warps={self.num_warps})")


def plan_bfast_fused_electric_pair(fields: Any, pml: Any, sources: Any = None,
                                   block: Optional[int] = None,
                                   num_warps: Optional[int] = 1,
                                   kernel: Any = None,
                                   ) -> Optional[BfastFusedElectricPairPlan]:
    """Build the plan from the engine's own objects, or ``None`` when refused.

    ``None`` is the only refusal: a configuration this kernel does not carry must
    fall back to the array path, never raise into a caller that would otherwise have
    stepped correctly.
    """
    if not bfast_fused_electric_pair_coverage(fields, pml, sources).covered:
        return None
    from ..stepping import _boundary_kinds as resolve  # noqa: PLC0415
    from .kernels import DEFAULT_BLOCK  # noqa: PLC0415
    from .launch import SUB_STEPS  # noqa: PLC0415

    curl_spec = SUB_STEPS[CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    grid = fields.grid
    kinds = resolve(grid, pml)
    # THE SAME FUNCTION THE CURL PLAN CALLS, with the grid's OWN declared invariance
    # flags (grid.py:1165-1183 — never a shape test) and the sub-step's side.
    # `magnetic=False` HERE IS LOAD-BEARING, unlike the beta family's flag: the D
    # side reads BFAST_TERMS['step_D'] and NEGATES both coefficients in host float64
    # (bfast_curl.py:343-351), so a copied `magnetic=True` would hand this product
    # the twin's six words. The host suite measures that the two answers differ.
    invariant = tuple(grid.is_invariant(axis) for axis in range(3))
    ks = bfast_curl_coefficients(grid.bfast_scaled_k, invariant,
                                 magnetic=(CURL_SUB_STEP == "step_B"))
    return BfastFusedElectricPairPlan(
        grid.shape, grid.dt / grid.dx,
        [1 if kind == "metallic" else 0 for kind in kinds],
        zero_metal_axes(grid), ks,
        DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in curl_spec["targets"]],
        [getattr(fields, "fu_" + name) for name in curl_spec["targets"]],
        [getattr(fields, name) for name in curl_spec["sources"]],
        # The engine's OWN state arrays, pointer-identical: the driver's flux
        # backup/restore mutates them in place (driver.py:4126-4135).
        [getattr(fields, name) for name in BFAST_STATE_NAMES[CURL_SUB_STEP]],
        [getattr(pml, f"{stem}_{axis}{curl_spec['suffix']}")
         for axis in "xyz" for stem in ("kms", "sinv")],
        [getattr(fields, name) for name in side_spec["targets"]],
        [getattr(fields, name) for name in side_spec["aux"]],
        [fields.inverse_epsilon_for(name) for name in side_spec["targets"]],
        # The curl takes the INTEGER lattice on step_D and the constitutive the
        # HALF-INTEGER one on E (stepping.py:948 vs :1015). The kernel takes both and
        # never asks which is which, so a swap here is a silent half-cell error in
        # the absorber profile; the gate carries a mutation for exactly it.
        [getattr(pml, f"{stem}_{axis}"
                      f"{'_h' if side_spec['half_integer'] else ''}")
         for axis in "xyz" for stem in ("kps", "kms")],
        kernel=kernel, num_warps=num_warps,
    )


def plan_bfast_fused_electric_pair_from_arrays(
        arrays: Dict[str, Any], curl_flat: Dict[str, Any],
        constitutive_flat: Dict[str, Any], codes, zero_metal, ks,
        dtdx: float, block: Optional[int] = None, kernel: Any = None,
        num_warps: Optional[int] = 1, has_bfast: int = 1,
        ) -> BfastFusedElectricPairPlan:
    """Build from bare device arrays — the gate's route.

    No coverage predicate runs: the caller is a harness that constructed the
    configuration deliberately, including the deliberately wrong ones. ``arrays``
    additionally carries ``inv_eps_Ex``...; ``kernel=`` carries the mutation override
    and ``has_bfast=0`` the identity leg's certified-kernel arm, which must reproduce
    ``kernels.fused_curl_constitutive_D`` bit for bit.
    """
    from .kernels import DEFAULT_BLOCK  # noqa: PLC0415
    from .launch import SUB_STEPS  # noqa: PLC0415

    curl_spec = SUB_STEPS[CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    shape = tuple(int(n) for n in arrays[curl_spec["targets"][0]].shape)
    return BfastFusedElectricPairPlan(
        shape, dtdx, codes, zero_metal, ks,
        DEFAULT_BLOCK if block is None else block,
        [arrays[name] for name in curl_spec["targets"]],
        [arrays["fu_" + name] for name in curl_spec["targets"]],
        [arrays[name] for name in curl_spec["sources"]],
        [arrays[name] for name in BFAST_STATE_NAMES[CURL_SUB_STEP]],
        [curl_flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
        [arrays[name] for name in side_spec["targets"]],
        [arrays[name] for name in side_spec["aux"]],
        [arrays["inv_eps_" + name] for name in side_spec["targets"]],
        [constitutive_flat[f"{stem}_{axis}"]
         for axis in "xyz" for stem in ("kps", "kms")],
        kernel=kernel, num_warps=num_warps, has_bfast=has_bfast,
    )
