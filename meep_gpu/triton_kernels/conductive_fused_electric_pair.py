"""The conductive PML ``step_D`` welded into the ordinary ``update_E``.

ONE LAUNCH FOR ALL THREE COMPONENTS, and the displacement never leaves a register
between the wall clear and the constitutive multiply. The separate composition for
this configuration is TWO launches -- one whole-grid conductive curl plus one
whole-grid constitutive -- and this is one.

It is built against the LAST unbuilt non-folded D->E cell on the Triton fusion
board -- ``(conductive PML, ordinary)``, one corpus row,
``tests:TestAdjointSolver.test_damping``, shape (150, 150, 1), metallic in x and y,
measured at ``D->E 1, ceiling 1`` in
``parity/meep_gpu/results/fusion_matrix_triton_2026-08-31_electrics`` -- and IT DOES
NOT SERVE THAT ROW. Read the clause in
:func:`conductive_fused_electric_pair_coverage` that says why before reading
anything else here: the driver applies a conductive run's electric injection as
three WHOLE-VOLUME passes (driver.py:3363-3370) rather than at the deposit points,
which rewrites ``-0.0`` to ``+0.0`` across the whole target component, and a repair
defined over the deposit points cannot reconstruct that. The row's source is not
integrated, so the row is refused BY NAME and this product's corpus count is ZERO.

WHY IT SHIPS ANYWAY. It is complete, gated and correct on everything it admits --
conductive PML runs with a magnetic source, and conductive PML runs with an
INTEGRATED electric source, which driver.py:3355-3356 injects point-wise and which
three device legs measure as carried. The day driver.py:3363-3370 scales at the
deposit indices the sources already publish, the clause comes out and the cell
closes with no other change. The alternative -- deleting a measured, gated weld
because a file this module does not own is whole-volume -- would lose the
measurement along with the product.

EVERYTHING HERE IS TRANSCRIBED, and every arithmetic line carries the function it
came from:

* the index arithmetic, the ghost rule, the curl and the ownership mask --
  :func:`.conductivity.conductive_pml_curl_step`, which is itself
  :func:`.kernels.pml_curl_step`'s body character for character, because
  conductivity enters ``_apply_curl`` only AFTER the curl is formed
  (stepping.py:508-534);
* the recurrence -- :func:`.conductivity._conductive_component`, called through
  :func:`_conductive_component_registers` below, which is that function with ONE
  statement moved and nothing else;
* the wall clear and the constitutive half --
  :func:`.kernels.fused_curl_constitutive_D`, whose six ``zero_metal_D`` rows and
  three-line ``update_E`` accumulation transcribe stepping.py:2206-2247 and
  :1010-1022.

A reader can diff this body against those three and see that nothing moved. If you
find yourself deriving here, you have taken a wrong turn.

===========================================================================
WHY THE RECURRENCE IS CALLED AND NOT COPIED -- the one structural decision
===========================================================================

:func:`.conductivity._conductive_component` STORES its result. A fused pair cannot
use it unchanged, because ``zero_metal_D`` has to reach the stepped displacement
BEFORE ``update_E`` reads it (driver.py:3301-3304) and a value already in memory is
a value this launch would have to read back.

Two ways out were available and the cheaper-looking one was rejected:

* CALL THE SHIPPED HELPER AND RE-LOAD ``f`` -- three extra loads, and a
  store-then-load of the same pointer inside one program. Bit-exact, and it makes
  the transcription claim trivial. REJECTED anyway: it makes the kernel's
  correctness depend on Triton honouring a store->load dependency through a raw
  pointer, which is a property of the compiler rather than of this file, and no
  sibling in this package leans on it.
* MOVE ONE STATEMENT. :func:`_conductive_component_registers` is
  ``_conductive_component`` with the single ``tl.store(f_ptr + idx, v, mask=live)``
  removed from each of its two constexpr branches and ``return v`` added. The
  ``u_ptr`` and ``c_ptr`` stores stay exactly where they are, MASKS INCLUDED --
  ``live & dsigu`` and ``live & dsig`` are the "unchanged" column of the case table
  and are not this weld's to touch.

The second is what is here, and the gate does not take the claim on trust: its
transcription leg parses both functions and asserts the statement lists are equal
after removing exactly that one store, so a second edit to either body turns the
gate red rather than drifting quietly.

THE STORE ORDER CHANGES AND THAT IS NOT AN ARITHMETIC CHANGE. The shipped helper
writes ``f``, then ``u``, then ``c``; here ``u`` and ``c`` are written inside the
helper and ``f`` is written by the caller after the wall clear. The three are
DISTINCT VOLUMES on every admitted configuration -- ``D*``, ``fu_D*`` and
``f_cond_D*`` are separate allocations -- so no lane's read of one can see another's
write, and the values stored are bit-for-bit the ones the shipped helper stores.
What did change is which value lands in ``D*`` on a walled plane, and that change is
the POINT: it is ``zero_metal_D``, which the array path runs between the two halves.

===========================================================================
THE SEAM -- four driver passes, and what happens to each
===========================================================================

The driver runs four passes between ``step_D`` and ``update_E``
(driver.py:3292-3304):

* **the electric injection (:3294-3299) -- CARRIED for an INTEGRATED source,
  REFUSED for a scaled one.** :data:`CARRIES_DEPOSIT_REPAIR` is True and the plan
  brackets its launch with ``deposit_repair``'s leading and trailing slots;
  ``deposit_repair.repairable(fields, 'D', pml)`` returns ``(True, ())`` on a
  conductive PML run, measured rather than argued. THE CONDUCTIVE INJECTION IS NOT
  THE SAME CLAUSE, and that is the correction this product paid for: for a
  non-integrated source ``FdtdDriver`` takes the ``condinv`` path
  (driver.py:3295-3296), which rescales the WHOLE VOLUME by difference
  (:3363-3370) rather than the deposit, and the repair cannot follow it. The
  predicate refuses that case by name;
* **``fill_symmetry_bc_D`` (:3300) -- INERT, because a mirror plane is refused.**
  Without ``grid.has_symmetry()`` the array path's fill returns at its first line
  (stepping.py:1481-1482). The refusal is restated in this module's own predicate
  rather than inherited, for the reason every sibling restates it;
* **``zero_metal_D`` (:3301) -- CARRIED INLINE**, six rows, transcribed from
  :func:`.kernels.fused_curl_constitutive_D`. This is not an optional nicety on this
  cell: its one row is metallic in x and y;
* **``fill_folded_far_ghosts_D`` (:3302) -- INERT for the same reason as the first
  fill** (stepping.py:1565-1566).

===========================================================================
WHAT THIS MODULE IS NOT
===========================================================================

It is NOT the no-PML conductive weld. :mod:`.complex_conductive_fused_pair` serves
``(complex conductive no-PML curl, complex no-PML stored E)`` and carries
``_apply_conductive_update`` -- case D alone, no ``f_cond``, no ``fu``. This one
carries the PML SPLIT-FIELD path, which is four cases selected by two exact float
comparisons, and refuses a run without an active layer by name. The two are
different recurrences and neither body may be read for the other.

DEVICE STATUS: see :data:`DEVICE_STATUS`. A weld licenses a claim, not a dispatch.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .conductivity import (
    CONDUCTIVE_SUB_STEPS,
    conductive_pml_curl_coverage,
    conductive_targets,
)
from .coverage import (
    CONSTITUTIVE_SIDES,
    Coverage,
    _call,
    constitutive_coverage,
    zero_metal_axes,
)
from .. import deposit_repair as _deposit_repair

#: Does this product bracket its fused launch with the deposit repair? TRUE, and it
#: is MEASURED rather than defaulted -- three device legs run an INTEGRATED electric
#: source through the shipped bracket and find the engine bit-identical, and two null
#: controls with the bracket removed diverge as they must.
#:
#: IT IS NOT WHAT MAKES THIS CELL REACHABLE, and that is the one place this module
#: differs from its 08-31 siblings. The cell's only corpus row declares a
#: NON-INTEGRATED electric source, which the coverage clause below refuses for a
#: reason the repair cannot fix: the driver's condinv rescale is whole-volume
#: (driver.py:3363-3370). The flag is still load-bearing for every conductive run
#: whose electric source IS integrated, which is what the carry legs measure.
#:
#: THE FLAG IS NOT THE WHOLE CLAIM. It only reaches ``deposit_repair.repairable``,
#: which goes on refusing BY NAME every seam the repair cannot invert -- an
#: off-diagonal constitutive, a nonlinear one, a fold whose fill map it cannot read,
#: and an absorber whose split-field recurrence never ran. This family refuses a fold
#: outright and REQUIRES an active layer through its curl half, so the clause that
#: sank the no-PML conductive cells ("f_w_Ex is not allocated; there is no state to
#: save") cannot fire here: an active layer is what allocates them.
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

#: ``SUB_STEPS['step_D']['backward']``, restated so the kernel's one legal binding is
#: visible without reading :mod:`launch`; the test suite pins the two equal.
BACKWARD = 1

#: The FIVE driver call sites ONE launch of this plan performs, in driver order
#: (driver.py:3292-3304). Declared, never inferred from the slot name. Only THREE of
#: them do work on an admitted configuration -- the two symmetry fills return at
#: their first line without a mirror plane, which this family refuses outright -- and
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
    "parity/meep_gpu/results/triton_conductive_electric_2026-08-31f/"
    "conductive_fused_electric_pair/gate.json, subnormal policy "
    "'ieee_keep_ftz_stripped', sm_86, CuPy 13.5.1. "
    "WHAT IT MEASURED: 6 no-device legs (transcription at STATEMENT grain over FOUR "
    "shipped bodies, equivalence, corpus admission, mutation pairing, the coefficient "
    "predicates behind this gate's two nulls, and the driver's whole-volume rescale) "
    "and 35 device legs -- 18 quiet, 9 carry, 3 null control, 3 reduction, 2 armed -- "
    "897 complete driver steps, 858 fused launches, each comparing the uint32 view of "
    "EVERY allocated volume after a COMPLETE driver.step() against BOTH the CuPy "
    "array path AND the two separately certified products this weld replaces "
    "(conductivity's conductive PML curl on step_D, launch's ordinary constitutive on "
    "update_E). No `allclose` appears anywhere in that gate. A CountingKernel proves "
    "the fused kernel is what executed, and every carry leg has a NULL CONTROL with "
    "the deposit bracket removed that MUST diverge and does -- all three did. "
    "21 of 23 kernel mutations caught and 5 of 5 host mutations; the other two are "
    "CONFIRMED NULLS with their evidence in MUTATION_EVIDENCE and re-measured every "
    "run by the coefficient-predicates leg (`kms != 1` and `sinv != 1` select the "
    "same set on every axis of every case grid, and no `kms` exceeds 1, so neither a "
    "tolerance nor an AND can change which case a cell takes). 7 of 7 refusals hold "
    "by name, one of them this product's own. "
    "THE REDUCTION LEG IS THE TRANSCRIPTION CLAIM ON SILICON: at COND=(0,0,0) this "
    "kernel is bit-identical to the shipped kernels.fused_curl_constitutive_D over "
    "the same volumes while BOTH differ from the array path, on all three value "
    "classes. "
    "WHAT IT DOES NOT LICENSE: nothing here is timed, no throughput, dispatch or "
    "end-to-end claim follows, and -- the one that matters for the board -- this "
    "product serves NONE of its cell's one corpus row. The row's electric source is "
    "not integrated, and on a conductive run the driver rescales such an injection "
    "over the WHOLE volume (driver.py:3363-3370) rather than at the deposit; the "
    "predicate refuses it by name and the whole_volume_rescale leg carries the "
    "arithmetic. A weld licenses a claim, not a dispatch.")

__all__ = [
    "BACKWARD", "CARRIES_DEPOSIT_REPAIR", "CONSTITUTIVE_SIDE", "CURL_SUB_STEP",
    "DEVICE_STATUS", "REPLACES",
    "ConductiveFusedElectricPairPlan",
    "conductive_fused_curl_constitutive_D",
    "conductive_fused_curl_constitutive_D_kernel",
    "conductive_fused_electric_pair_coverage",
    "plan_conductive_fused_electric_pair",
    "plan_conductive_fused_electric_pair_from_arrays",
]


if triton is not None:

    # The same constexpr codes as ``kernels``' own, restated for the reason every
    # sibling restates them -- importing a ``tl.constexpr`` wrapper and re-wrapping
    # it is not the same object, and the comparison in the kernel body is against a
    # literal code. ONLY METALLIC IS NAMED: both certified bodies branch on
    # ``== METALLIC`` and take periodic as the else.
    METALLIC = tl.constexpr(1)

    @triton.jit
    def _conductive_component_registers(f_ptr, u_ptr, c_ptr, cf_ptr, ci_ptr,
                                        idx, live, curl, km1, si1, km2, si2,
                                        COND: tl.constexpr):
        """:func:`.conductivity._conductive_component`, returning ``v`` instead of
        storing it.

        ONE STATEMENT MOVED. Both constexpr branches are that function's, character
        for character, minus ``tl.store(f_ptr + idx, v, mask=live)``; the ``u_ptr``
        and ``c_ptr`` stores keep their masks, which are the "unchanged" column of
        the four-case table and not this weld's to touch. ``f_ptr`` is still taken
        because the ``f`` LOAD stays here -- only the write moved.

        The gate's transcription leg parses both bodies and asserts the statement
        lists are equal after removing exactly that store, so an edit to either one
        turns it red.
        """
        f = tl.load(f_ptr + idx, mask=live, other=0.0)
        u = tl.load(u_ptr + idx, mask=live, other=0.0)
        if COND == 0:
            # stepping._apply_pml_update (:1905), verbatim from kernels.pml_curl_step.
            n = ((u * km1) - curl) * si1
            v = (((f * km2) + n) - u) * si2
            tl.store(u_ptr + idx, n, mask=live)
        else:
            c = tl.load(c_ptr + idx, mask=live, other=0.0)
            cf = tl.load(cf_ptr + idx, mask=live, other=0.0)
            ci = tl.load(ci_ptr + idx, mask=live, other=0.0)

            # stepping.py:2055-2058 -- an EXACT float comparison, not a tolerance.
            dsig = (km1 != 1.0) | (si1 != 1.0)
            dsigu = (km2 != 1.0) | (si2 != 1.0)

            # The five branch expressions. DO NOT FLATTEN THESE PARENTHESES.
            c_new = ((c * cf) - curl) * ci                     # cases A and C
            u_cond = ((u * cf) - curl) * ci                    # case B
            u_split = (((u * km1) + c_new) - c) * si1          # case A
            u_new = tl.where(dsig, u_split, u_cond)
            f_split = (((f * km2) + u_new) - u) * si2          # cases A and B
            f_first = (((f * km1) + c_new) - c) * si1          # case C
            f_direct = ((f * cf) - curl) * ci                  # case D
            v = tl.where(dsigu, f_split, tl.where(dsig, f_first, f_direct))

            tl.store(u_ptr + idx, u_new, mask=live & dsigu)
            tl.store(c_ptr + idx, c_new, mask=live & dsig)
        return v

    @triton.jit
    def conductive_fused_curl_constitutive_D(
        f0, f1, f2,                       # curl targets: Dx,Dy,Dz
        u0, u1, u2,                       # curl auxiliaries: fu_Dx,fu_Dy,fu_Dz
        c0, c1, c2,                       # f_cond_D*  (a placeholder where COND == 0)
        cf0, cf1, cf2,                    # condfac    (a placeholder where COND == 0)
        ci0, ci1, ci2,                    # condinv    (a placeholder where COND == 0)
        g0, g1, g2,                       # curl sources: Hx,Hy,Hz
        kmx, sinvx, kmy, sinvy, kmz, sinvz,   # curl kms/sinv, INTEGER lattice
        e0, e1, e2,                       # constitutive targets: Ex,Ey,Ez
        w0, w1, w2,                       # constitutive aux: f_w_Ex,f_w_Ey,f_w_Ez
        ie0, ie1, ie2,                    # diagonal inverse epsilon volumes
        kp0, km0, kp1, km1, kp2, km2,     # constitutive kps/kms, HALF-INTEGER
        nx, ny, nz, n_elem, dtdx,
        BACKWARD: tl.constexpr,           # bound to 1 by every builder; see below
        BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
        ZM_X: tl.constexpr, ZM_Y: tl.constexpr, ZM_Z: tl.constexpr,
        COND0: tl.constexpr, COND1: tl.constexpr, COND2: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Conductive ``step_D`` + ``zero_metal_D`` + ``update_E``, one launch.

        :func:`.conductivity.conductive_pml_curl_step`'s body down to and including
        the six coefficient loads, then the three recurrences through
        :func:`_conductive_component_registers`, then
        :func:`.kernels.fused_curl_constitutive_D`'s wall clear, D store and
        constitutive half. Nothing else is here.

        ``BACKWARD`` IS CARRIED AND MUST BE 1. It keeps the curl half a verbatim copy
        of the certified body rather than a hand-specialised one. It cannot be 0: the
        wall clear below is the D family's SIX rows -- a metallic axis clears the two
        TANGENTIAL D components, where the magnetic family clears the one normal B --
        and the constitutive half is the E side with its inverse-permittivity
        multiply.

        ``COND0``/``COND1``/``COND2`` are :func:`.conductivity.conductive_targets`'s
        three answers and at least one is 1 on any admitted configuration; the
        predicate refuses an all-lossless grid by name, because that configuration is
        :func:`.kernels.fused_curl_constitutive_D`'s own. They are carried rather
        than hard-coded so the gate has a certified-kernel arm: at
        ``COND0 = COND1 = COND2 = 0`` this body must reproduce that kernel bit for
        bit, which is a leg rather than an argument.
        """
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = idx < n_elem
        nyz = ny * nz
        k = idx % nz
        plane = idx // nz
        j = plane % ny
        i = plane // ny

        # --- the ghost rule, per axis (stepping._shift_up / _shift_down) --------
        # PERIODIC wraps; METALLIC serves an exact 0.0 past the wall, which
        # `tl.load`'s `other=` delivers without dereferencing anything.
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

        # --- the curl (stepping._curl_from_operands): DO NOT flatten these ------
        curl0 = dtdx * ((c_y - c) + (b - b_z))
        curl1 = dtdx * ((a_z - a) + (c - c_x))
        curl2 = dtdx * ((b_x - b) + (a - a_y))

        # --- ownership mask (stepping._mask_non_owned_cells) -------------------
        at_x, at_y, at_z = i == 0, j == 0, k == 0
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

        # --- the recurrence, per component -------------------------------------
        # dsig/dsigu follow vec.hpp's cycle_direction and are the same triple on
        # both sides: target 0 takes (y, z), target 1 (z, x), target 2 (x, y).
        km_x = tl.load(kmx + i, mask=live, other=0.0)
        si_x = tl.load(sinvx + i, mask=live, other=0.0)
        km_y = tl.load(kmy + j, mask=live, other=0.0)
        si_y = tl.load(sinvy + j, mask=live, other=0.0)
        km_z = tl.load(kmz + k, mask=live, other=0.0)
        si_z = tl.load(sinvz + k, mask=live, other=0.0)

        v0 = _conductive_component_registers(f0, u0, c0, cf0, ci0, idx, live, curl0,
                                             km_y, si_y, km_z, si_z, COND0)
        v1 = _conductive_component_registers(f1, u1, c1, cf1, ci1, idx, live, curl1,
                                             km_z, si_z, km_x, si_x, COND1)
        v2 = _conductive_component_registers(f2, u2, c2, cf2, ci2, idx, live, curl2,
                                             km_x, si_x, km_y, si_y, COND2)

        # zero_metal_D: each axis clears the two tangential D components.
        if ZM_X:
            v1 = tl.where(at_x, 0.0, v1)
            v2 = tl.where(at_x, 0.0, v2)
        if ZM_Y:
            v0 = tl.where(at_y, 0.0, v0)
            v2 = tl.where(at_y, 0.0, v2)
        if ZM_Z:
            v0 = tl.where(at_z, 0.0, v0)
            v1 = tl.where(at_z, 0.0, v1)

        # THE D STORE IS KEPT, and it is the store the helper no longer performs.
        # The array path leaves the stepped displacement in the D volume and later
        # sub-steps read it; dropping it would fuse away a value the run still needs,
        # which is a different defect from a wrong arithmetic line and one no per-step
        # E comparison would see.
        tl.store(f0 + idx, v0, mask=live)
        tl.store(f1 + idx, v1, mask=live)
        tl.store(f2 + idx, v2, mask=live)

        # ==================== the constitutive half ===========================
        # Verbatim from kernels.fused_curl_constitutive_D, which is itself
        # constitutive_step's SCALE=1 arm with `tl.load(g + idx)` replaced by the
        # register the curl half computed. Component 0 takes its coefficient from
        # axis x, 1 from y, 2 from z (stepping.E_CONSTITUTIVE_TERMS :227) -- the
        # component's OWN axis. D IS ON THE LEFT of the inverse-epsilon multiply
        # because the array path writes `source * inverse_epsilon_for(component)`
        # (stepping.py:1011-1013).
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
    conductive_fused_curl_constitutive_D = None  # type: ignore[assignment]
    _conductive_component_registers = None  # type: ignore[assignment]


def conductive_fused_curl_constitutive_D_kernel() -> Any:
    """The JIT kernel, or a named ImportError on a host without Triton."""
    if conductive_fused_curl_constitutive_D is None:
        raise ImportError(
            "the conductive fused electric D/E kernel needs the optional `triton` "
            f"package (pip install triton). Original error: {_TRITON_IMPORT_ERROR}")
    return conductive_fused_curl_constitutive_D


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def conductive_fused_electric_pair_coverage(fields: Any, pml: Any,
                                            sources: Any = None) -> Coverage:
    """May ONE launch span conductive ``step_D`` -> wall -> ``update_E``?

    A conjunction of the two halves' own SHIPPED predicates plus the seam clauses.
    Nothing is weakened: a configuration either half refuses is refused here with
    that half's reasons, prefixed so a reader can tell which side said it -- the
    construction :func:`.coverage.fused_pair_coverage` and every fused sibling use.

    The two halves are the arms ``launch.plan_step`` really selects on a conductive
    PML row today (``conductive PML`` on ``step_D``, ``ordinary`` on ``update_E``),
    so this predicate is narrower than the pair the planner already puts on the
    device, never wider.
    """
    reasons: List[str] = []

    curl = conductive_pml_curl_coverage(fields, pml, CURL_SUB_STEP)
    if not curl.covered:
        reasons.extend(f"conductive curl half: {reason}" for reason in curl.reasons)
    constitutive = constitutive_coverage(fields, pml, CONSTITUTIVE_SIDE)
    if not constitutive.covered:
        reasons.extend(f"constitutive half: {reason}"
                       for reason in constitutive.reasons)

    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False,
                        tuple(dict.fromkeys(reasons + ["fields carries no grid"])))

    # AT LEAST ONE CONDUCTIVE COMPONENT, refused BY NAME rather than left to the
    # plan's ValueError. An all-lossless grid is kernels.fused_curl_constitutive_D's
    # own configuration and that product already serves it; admitting it here would
    # put two products on one cell, which the board scores as an ambiguity rather
    # than as coverage.
    if not any(conductive_targets(fields, CURL_SUB_STEP)):
        reasons.append(
            "no D component carries a conductivity: this configuration is "
            "kernels.fused_curl_constitutive_D's, and coverage.fused_pair_coverage "
            "already claims it")

    # THE SOURCE SEAM, and on this cell it is the clause that decides everything. An
    # ELECTRIC source is injected BETWEEN the two halves (driver.py:3294-3299), so a
    # fused pair would consume a pre-injection D. The one corpus row of this cell
    # declares exactly that, so with CARRIES_DEPOSIT_REPAIR False the arm would admit
    # NOTHING; it is True, and the plan below brackets its launch with the repair.
    # IGNORANCE IS NEVER AN EMPTY SET: `Fields` does not hold the source list, so an
    # undeclared `sources` is itself a REFUSAL. A MAGNETIC source is injected in the
    # B/H half and does not disqualify this pair.
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

    # A NON-INTEGRATED ELECTRIC SOURCE ON A CONDUCTIVE RUN — the clause that used
    # to refuse it BY NAME was LIFTED ON A MEASUREMENT (2026-09-01), and both the
    # retired measurement and the lift are kept here so a later round need not
    # re-derive either.
    #
    # THE RETIRED COMPOSITION. The driver used to snapshot the target D component,
    # inject, and rescale BY DIFFERENCE over the WHOLE VOLUME (`array -= before;
    # array *= condinv; array += before`). Exact for every finite value, but it
    # rewrote `-0.0` to `+0.0` at EVERY cell of the component — MEASURED on
    # the GPU host GPU 6, 2026-08-31, on this product's own wall_xy grid under the
    # signed-zero value class, one complete driver step, three routes:
    #
    #     is_integrated=True   identical, 3 steps, 3 deposits repaired
    #     is_integrated=False  Ez differs in 14 words and f_w_Ez in 74, every one
    #                          of them -0.0 against +0.0, while D itself AGREES
    #
    # THE DRIVER NO LONGER DOES THAT. `_inject_electric_through_conductivity` now
    # snapshots the target at the deposit cells the sources publish
    # (`deposit_repair._deposit_index` reads the same tables) and replays the
    # rescale per deposit cell in the retired passes' exact operand order — exact
    # at every deposit cell, untouched everywhere else, and closer to stock MEEP
    # (step.cpp:294-317 scales only the injected current). RE-MEASURED with the
    # sparse replay in place, same protocol, same signed-zero seed class: the
    # bracketed fused route is byte-identical to the array path at EVERY step,
    # and the retired whole-volume passes replayed as a control still diverge —
    # the `lifted_refusal` leg of probe_triton_conductive_fused_electric_pair.py
    # carries both measurements. An INTEGRATED source was always carried
    # (driver injects it point-wise and skips the rescale, MEEP step.cpp:300).
    #
    # WHAT SURVIVES is the driver's own fallback: a scaled source publishing NO
    # deposit table (no `_point_ix`) still takes the whole-volume passes, because
    # an unnameable deposit must still be scaled. No in-tree electric source
    # class is one; the clause stays for the duck-typed stranger.
    #
    # WHAT THE LIFT SERVES: the cell's only corpus row
    # (``tests:TestAdjointSolver.test_damping``), whose non-integrated electric
    # source the pre-lift clause refused by name.
    if _call(grid, "has_conductivity", default=None) is not False:
        for index, source in enumerate(_deposit_repair.in_seam_sources(
                tuple(sources) if sources is not None else (), 'D')):
            if (not getattr(source, "is_integrated", False)
                    and not hasattr(source, "_point_ix")):
                reasons.append(
                    f"source {index} ({type(source).__name__}) is a NON-INTEGRATED "
                    f"electric source on a conductive run and publishes NO deposit "
                    f"table (no _point_ix): the driver's fallback applies its "
                    f"condinv scaling as three WHOLE-VOLUME passes "
                    f"(_inject_electric_through_conductivity's dense branch), which "
                    f"rewrites -0.0 to +0.0 at every cell of "
                    f"D{source.component[1] if getattr(source, 'component', None) else '*'}"
                    f" and not only at the deposit, and a point repair cannot "
                    f"reconstruct that")

    # THE TWO SYMMETRY PASSES. Both return at their first line without
    # `grid.has_symmetry()` (stepping.py:1481-1482, :1565-1566). RE-CHECKED HERE
    # rather than inferred from either half's guard: this module's coverage may not
    # be read off another module's, and if the conductive tranche ever admits a fold
    # this weld would silently swallow two passes that had started doing work.
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
    # axes are walled; an unanswerable one compiles to ZM=False and silently skips a
    # plane the array path clears.
    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            reasons.append(
                f"grid does not expose {name}; zero_metal_D cannot be carried inline")

    # THE INVERSE-EPSILON VOLUMES ARE READ BY THE KERNEL, so `Fields` must be able to
    # hand one over per E component. An absent accessor would be a TypeError inside
    # the builder rather than a refusal, which is the wrong way for an uncovered
    # configuration to fail.
    if getattr(fields, "inverse_epsilon_for", None) is None:
        reasons.append(
            "fields does not expose inverse_epsilon_for; the constitutive half "
            "cannot bind the inv_eps volumes it loads")

    # THE THREE CONDUCTIVE VOLUMES ARE READ BY THE KERNEL for every component whose
    # COND is 1. The curl half's own predicate asks the same question and this is a
    # restatement, not a second opinion: an absent reader here is a TypeError inside
    # the builder, and a refusal is the right shape for it.
    for name in ("condfac_for", "condinv_for"):
        if getattr(fields, name, None) is None:
            reasons.append(
                f"fields does not expose {name}; the conductive recurrence cannot "
                f"bind the volumes it loads")

    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class ConductiveFusedElectricPairPlan:
    """One allocation-free launch for five of the driver's electric call sites.

    :class:`.launch.FusedPairPlan`'s ``pair='D'`` bindings plus the nine conductive
    pointers and three ``COND`` constexprs :class:`.conductivity.
    ConductivePmlCurlPlan` carries over :class:`.launch.PmlCurlPlan`. The
    inverse-epsilon volumes are the E side's and are bound here.

    A LOSSLESS COMPONENT BINDS ITS OWN TARGET AS THE PLACEHOLDER for the three
    conductive slots, never a null -- the same rule and the same reason as the curl
    plan: the loads sit behind a ``tl.constexpr`` branch and compile away, but a
    pointer argument still has to type, and ``None`` makes the launcher's failure a
    ``TypeError`` far from its cause.
    """

    __slots__ = ("shape", "n_elem", "dtdx", "cond", "backward", "bc", "zero_metal",
                 "block", "num_warps",
                 "_targets", "_aux", "_history", "_condfac", "_condinv",
                 "_sources", "_curl_coefficients",
                 "_e_targets", "_e_aux", "_inverse_epsilon", "_e_coefficients",
                 "_grid", "_kernel")

    #: The five driver call sites one launch performs. Declared, so a composition can
    #: be inspected rather than inferred from the curl slot's name.
    replaces = REPLACES

    def __init__(self, shape, dtdx: float, bc, zero_metal, cond, block: int,
                 targets, auxiliaries, history, condfac, condinv, sources,
                 curl_coefficients, e_targets, e_aux, inverse_epsilon,
                 e_coefficients, kernel: Any = None,
                 num_warps: Optional[int] = 1) -> None:
        from .launch import CupyPointer, _flat  # noqa: PLC0415

        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        # float(): the array path multiplies a float32 volume by a Python float,
        # which NumPy/CuPy cast to float32 before the multiply. Triton types a
        # Python float argument as fp32, so the two scalars are the same bits.
        self.dtdx = float(dtdx)
        self.backward = BACKWARD
        self.bc = tuple(int(code) for code in bc)
        self.zero_metal = tuple(1 if flag else 0 for flag in zero_metal)
        self.cond = tuple(1 if flag else 0 for flag in cond)
        self.block = int(block)
        # ONE WARP, the measured default for a fused pair
        # (launch.FUSED_DEFAULT_NUM_WARPS). Explicit None remains "take Triton's
        # default".
        self.num_warps = None if num_warps is None else int(num_warps)
        self._targets = tuple(CupyPointer(a) for a in targets)
        self._aux = tuple(CupyPointer(a) for a in auxiliaries)
        # Placeholders for the lossless components, resolved HERE so the launch
        # path never branches (see the class docstring).
        self._history = tuple(CupyPointer(a if a is not None else b)
                              for a, b in zip(history, targets))
        self._condfac = tuple(CupyPointer(a if a is not None else b)
                              for a, b in zip(condfac, targets))
        self._condinv = tuple(CupyPointer(a if a is not None else b)
                              for a, b in zip(condinv, targets))
        self._sources = tuple(CupyPointer(a) for a in sources)
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
        # An override for exactly one caller: the gate's mutation legs, which compile
        # deliberately broken copies of this kernel. Dropping it is not a slowdown,
        # it is a DISARMING.
        self._kernel = kernel

    def run(self, guard: Optional[bool] = None) -> None:
        """Launch the fused pair, in place. Same ``guard`` contract as every plan."""
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        kernel = (self._kernel if self._kernel is not None
                  else conductive_fused_curl_constitutive_D_kernel())
        nx, ny, nz = self.shape
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        kernel[self._grid](
            *self._targets, *self._aux, *self._history, *self._condfac,
            *self._condinv, *self._sources, *self._curl_coefficients,
            *self._e_targets, *self._e_aux, *self._inverse_epsilon,
            *self._e_coefficients,
            nx, ny, nz, self.n_elem, self.dtdx,
            BACKWARD=self.backward,
            BCX=self.bc[0], BCY=self.bc[1], BCZ=self.bc[2],
            ZM_X=self.zero_metal[0], ZM_Y=self.zero_metal[1],
            ZM_Z=self.zero_metal[2],
            COND0=self.cond[0], COND1=self.cond[1], COND2=self.cond[2],
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )

    def __repr__(self) -> str:
        return (f"ConductiveFusedElectricPairPlan(shape={self.shape}, bc={self.bc}, "
                f"zero_metal={self.zero_metal}, cond={self.cond}, "
                f"block={self.block}, num_warps={self.num_warps})")


def plan_conductive_fused_electric_pair(
        fields: Any, pml: Any, sources: Any = None,
        block: Optional[int] = None, num_warps: Optional[int] = 1,
        kernel: Any = None) -> Optional[ConductiveFusedElectricPairPlan]:
    """Build the plan from the engine's own objects, or ``None`` when refused.

    ``None`` is the only refusal: a configuration this kernel does not carry must
    fall back to the array path, never raise into a caller that would otherwise have
    stepped correctly.
    """
    if not conductive_fused_electric_pair_coverage(fields, pml, sources).covered:
        return None
    from ..stepping import _boundary_kinds as resolve  # noqa: PLC0415
    from .kernels import DEFAULT_BLOCK  # noqa: PLC0415
    from .launch import SUB_STEPS  # noqa: PLC0415

    curl_spec = SUB_STEPS[CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    targets = CONDUCTIVE_SUB_STEPS[CURL_SUB_STEP]
    grid = fields.grid
    kinds = resolve(grid, pml)
    return ConductiveFusedElectricPairPlan(
        grid.shape, grid.dt / grid.dx,
        [1 if kind == "metallic" else 0 for kind in kinds],
        zero_metal_axes(grid),
        conductive_targets(fields, CURL_SUB_STEP),
        DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in targets],
        [getattr(fields, "fu_" + name) for name in targets],
        [getattr(fields, "f_cond_" + name) for name in targets],
        [fields.condfac_for(name) for name in targets],
        [fields.condinv_for(name) for name in targets],
        [getattr(fields, name) for name in curl_spec["sources"]],
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


def plan_conductive_fused_electric_pair_from_arrays(
        arrays: Dict[str, Any], curl_flat: Dict[str, Any],
        constitutive_flat: Dict[str, Any], codes, zero_metal, cond,
        dtdx: float, block: Optional[int] = None, kernel: Any = None,
        num_warps: Optional[int] = 1) -> ConductiveFusedElectricPairPlan:
    """Build from bare device arrays -- the gate's route.

    No coverage predicate runs: the caller is a harness that constructed the
    configuration deliberately, including the deliberately wrong ones. ``arrays``
    additionally carries ``f_cond_Dx``..., ``condfac_Dx``..., ``condinv_Dx``... and
    ``inv_eps_Ex``...; ``kernel=`` carries the mutation override and
    ``cond=(0, 0, 0)`` the identity leg's certified-kernel arm, which must reproduce
    ``kernels.fused_curl_constitutive_D`` bit for bit.
    """
    from .kernels import DEFAULT_BLOCK  # noqa: PLC0415
    from .launch import SUB_STEPS  # noqa: PLC0415

    curl_spec = SUB_STEPS[CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    targets = CONDUCTIVE_SUB_STEPS[CURL_SUB_STEP]
    shape = tuple(int(n) for n in arrays[targets[0]].shape)
    return ConductiveFusedElectricPairPlan(
        shape, dtdx, codes, zero_metal, cond,
        DEFAULT_BLOCK if block is None else block,
        [arrays[name] for name in targets],
        [arrays["fu_" + name] for name in targets],
        [arrays.get("f_cond_" + name) for name in targets],
        [arrays.get("condfac_" + name) for name in targets],
        [arrays.get("condinv_" + name) for name in targets],
        [arrays[name] for name in curl_spec["sources"]],
        [curl_flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
        [arrays[name] for name in side_spec["targets"]],
        [arrays[name] for name in side_spec["aux"]],
        [arrays["inv_eps_" + name] for name in side_spec["targets"]],
        [constitutive_flat[f"{stem}_{axis}"]
         for axis in "xyz" for stem in ("kps", "kms")],
        kernel=kernel, num_warps=num_warps,
    )
