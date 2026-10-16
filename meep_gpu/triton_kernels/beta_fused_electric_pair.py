"""The special_kz REAL-beta ELECTRIC seam in one launch: beta ``step_D`` welded
into ``update_E``.

The ``real beta PML`` -> ``real beta run`` cell's D->E product, and the ELECTRIC
TWIN of :mod:`.beta_fused_magnetic_pair`. Both halves are already certified and
neither is re-derived here:

* the curl half is :func:`.special_kz.beta_pml_curl_step` — itself
  ``kernels.pml_curl_step``'s certified body plus ONE constexpr-gated insert — taken
  through its ``BACKWARD == 1`` arm, which that body already carries and which its
  magnetic twin binds to 0 and never compiles;
* the constitutive half is ``kernels.constitutive_step``'s ``SCALE = 1`` arm,
  admitted on a beta run by the shipped
  :func:`.special_kz.beta_run_constitutive_coverage` ("the constitutive sub-steps
  read nothing beta-dependent");
* and the WELD is the one substitution :func:`.kernels.fused_curl_constitutive_D`
  already makes between exactly those two bodies on a beta = 0 run.

SO THIS KERNEL IS ``fused_curl_constitutive_D``, CHARACTER FOR CHARACTER, PLUS THE
PASS ``beta_pml_curl_step`` ADDS TO ``pml_curl_step``. A reader must be
able to diff this body against those two and see nothing else moved; the gate's
transcription leg asserts exactly that, statement by statement, against both
shipped sources.

===========================================================================
WHY THIS CELL WAS WORTH ZERO AND IS NOW WORTH ONE
===========================================================================

:mod:`.beta_fused_magnetic_pair`'s docstring says, of this very cell:

    The D/E partner cell is worth ZERO on this corpus (that row injects
    electrically between ``step_D`` and ``update_E``) and is not built.

THAT SENTENCE WAS TRUE WHEN IT WAS WRITTEN AND IS NOT TRUE NOW, and the difference
is :mod:`..deposit_repair`, which went live on 2026-08-28. The row is
``examples:refl-angular-kz2d.py`` — shape (1200, 1, 1), beta = 0.3321611318837033,
all-periodic, ``has_metallic == false`` — and it declares
``source_field_types == ['D']``. That injection is exactly what made the cell
unreachable; the repair is what carries it. Measured on the 2026-08-31 board
(``results/fusion_matrix_triton_2026-08-31_folded_disp``)::

    ceiling  1  (rows 1)  D->E  real beta PML -> real beta run   NO FUSED PRODUCT

with ``in_seam_source: true`` and ``in_seam_source_blocks: false``. So
:data:`CARRIES_DEPOSIT_REPAIR` is not one clause of this product — it IS the
product: at False the source clause refuses the only row the cell has and the arm
admits ZERO. The MAGNETIC twin keeps the flag at False for the mirror-image reason:
on ITS seam the same row is electric-only, so no repair is consulted at all.

THE SIBLING ON THE OTHER BACKEND is ``metal_kernels/beta_fused_electric_pair.py``,
which declares the flag True for the same row and the same reason. This module is
its Triton twin: the same two certified halves, the same seam, each transcribed
from ITS OWN backend's body.

===========================================================================
THE PASS, AND WHERE IT GOES
===========================================================================

``stepping._special_kz_beta_term`` adds an analytic ``d/dz -> i*2*pi*beta`` term.
MEEP runs it as a pass of its own after every component's curl
(step_db.cpp:161-175), and so does the array path (``stepping._apply_folded_increment``):
after the three curl forms and before the stores, which is where it sits here. Transcribed from
:func:`.special_kz.beta_pml_curl_step`, not re-derived:

* the beta partners are the CENTER loads the curl already made — target 0 takes the
  second source's center ``b`` at sign +1 (Dx <- Hy) and target 1 the first
  source's center ``a`` at sign -1 (Dy <- Hx); target 2 gets nothing, because
  ``step_db.cpp:148-176`` runs ``cc`` over ``d_c`` in {X, Y} only;
* NO ``dtdx`` multiplies the term (it is an analytic derivative, S:733-735);
* the increment is ``(c * g) * -1.0``, the array path's ``-(c * g)`` (the negation a
  multiply, never Triton's unary minus), an exact ``+0`` on every cell the curl's
  ownership mask drops, and the ladder subtracts it (:func:`.forms.fold_ladder`):
  ``f - (sinv * q)`` outside the target's dsigu chunk, ``fu - (sinv * q)`` and
  ``f - (sinv_u * (sinv * q))`` inside it;
* real storage is MEEP's implicit-i trick, so BOTH sub-steps take the same sign
  (the ±i lives only in complex storage). MEASURED, and worth saying plainly because
  a reader will assume otherwise: :func:`.special_kz.beta_curl_coefficients` takes
  ``magnetic=``, this builder passes ``False`` and the twin passes ``True``, and ON
  REAL STORAGE THE TWO ANSWERS ARE IDENTICAL — the flag only decides the ``±1j``
  quarter turn on the complex path (:func:`.special_kz.beta_curl_coefficients`). So a builder here that
  copied its twin's ``magnetic=True`` would be harmless, and this product's gate
  arms NO mutation pretending to catch it; what it arms is the two WORDS exchanged
  (``h_beta_words_swapped``), which differ in sign on either storage. The host suite
  pins both halves of that.

``beta_plus``/``beta_minus`` are host-rounded ONCE by
:func:`.special_kz.beta_curl_coefficients` and passed as float32 words, exactly as
:class:`.special_kz.BetaPmlCurlPlan` passes them.

``HAS_BETA`` IS CARRIED AND MUST BE 1 on any configuration this product's predicate
admits — clause 12 of the beta family is INVERTED (``grid.beta`` must be nonzero),
so a beta = 0 run is the ordinary :func:`.coverage.fused_pair_coverage`'s and is
refused here by name. It is carried rather than hard-coded for
``beta_pml_curl_step``'s own reason: it keeps the curl half a verbatim copy rather
than a hand-specialised one, and it gives the gate a certified-kernel arm
(``HAS_BETA = 0`` must reproduce ``fused_curl_constitutive_D`` bit for bit).

===========================================================================
THE FIVE PASSES IN THE SEAM
===========================================================================

``driver.step`` runs five passes between the two halves (driver.py:3292-3304)::

    step_D -> ELECTRIC SOURCES -> fill_symmetry_bc_D -> zero_metal_D
           -> fill_folded_far_ghosts_D -> update_E

* **the electric sources — CARRIED, through the SHIPPED deposit repair.** The
  injection lands between the two halves, so the fused launch consumes a
  pre-injection ``D``; :class:`.deposit_repair.LeadingRepairPlan` saves the two
  stateful arrays at the deposit points before the launch and
  :class:`.deposit_repair.TrailingRepairPlan` recomputes them after, in the
  driver's own order. IGNORANCE IS STILL NEVER AN EMPTY SET: ``Fields`` does not
  hold the source list, so an undeclared ``sources`` is a REFUSAL, and a source
  that cannot publish the index it writes is refused by name;
* **``fill_symmetry_bc_D`` and ``fill_folded_far_ghosts_D`` — PROVABLY NO-OPS
  HERE**, not carried and not refused. Both return at their first line unless
  ``grid.has_symmetry()`` (stepping.py:1481-1482, :1565-1566), and clause 5 of
  ``_beta_real_grid_reasons`` refuses every fold outright ("the folded real beta
  curl is Phase B"). THE PREDICATE RE-CHECKS IT ANYWAY rather than reading its own
  coverage off another module's guard;
* **``zero_metal_D`` — CARRIED INLINE** (driver.py:3301; ``stepping._zero_metal``
  :2206-2247), through :func:`.coverage.zero_metal_axes`, which is IMPORTED rather
  than re-spelled so the predicate and the compile-time choice cannot disagree. SIX
  ROWS on this side against the magnetic twin's three — each axis clears the two
  TANGENTIAL D components — which is ``fused_curl_constitutive_D``'s own block and
  is byte-copied from it.

The corpus row is all-periodic with ``has_metallic == false``, so the inline
``zero_metal_D`` carry compiles to three ``False`` flags ON THAT ROW — which is why
the gate scores every wall-clear mutation on a WALLED case instead, where the flags
are real.

NOT WIRED, and not an arm. ``launch.plan_step`` assigns at most one plan per
``STEP_ORDER`` slot and this product spans FIVE driver call sites; there is no slot
it can claim without a composition rule nothing has measured. Wired, it would also
contend for ``step_D`` with the ``real beta PML`` arm, which IS wired and admits the
same configurations — and ``_select_slot`` (launch.py:1983-1996) leaves a slot with
two admitters UNSELECTED, taking the certified curl off the device with it. Nothing
in ``launch.py`` names this module; ``fastpath.plan_fast_path`` is unchanged.

Import contract: importable WITHOUT Triton — the predicate and the plan builder (to
``None``) must answer on the laptop that is the merge bar.

DEVICE STATUS: see :data:`DEVICE_STATUS`. A weld licenses a claim, not a dispatch.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .coverage import (
    CONSTITUTIVE_SIDES,
    Coverage,
    _call,
    zero_metal_axes,
)
from .special_kz import (
    beta_curl_coefficients,
    beta_pml_curl_coverage,
    beta_run_constitutive_coverage,
)
from .. import deposit_repair as _deposit_repair

#: Does this product bracket its fused launch with the deposit repair? TRUE, and on
#: THIS cell it is the difference between one seam-instance and zero: the single
#: corpus row (``examples:refl-angular-kz2d.py``) declares an ELECTRIC source, which
#: the driver injects between the two halves (driver.py:3294-3299). The flag is a
#: claim about the PLAN this module builds -- that the leading slot saves and the
#: trailing slot restores -- and is only ever changed in the same edit as that
#: wiring. Its MAGNETIC twin keeps the flag at False for the mirror-image reason:
#: on the B seam that same row is electric-only.
#:
#: THE FLAG IS NOT THE WHOLE CLAIM. It only reaches ``deposit_repair.repairable``,
#: which goes on refusing BY NAME every seam the repair cannot invert -- an
#: off-diagonal constitutive, a nonlinear one, a fold whose fill map it cannot read,
#: and an absorber whose split-field recurrence never ran. This family refuses a fold
#: outright and requires an active layer through both halves, so the clauses that
#: remain live here are the D-side ones.
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
#: (driver.py:3292-3304). Declared, never inferred from the slot name. Only TWO of
#: them do work on an admitted configuration — the two symmetry fills return at their
#: first line without a mirror plane, which this family refuses outright — and the
#: inert two are listed anyway: what a launch REPLACES is what the driver would
#: otherwise have called. THE INJECTION IS NOT ONE OF THE FIVE: it is carried by the
#: deposit repair bracket around the launch, not by the launch.
REPLACES: Tuple[str, ...] = ("step_D", "fill_symmetry_bc_D", "zero_metal_D",
                             "fill_folded_far_ghosts_D", "update_E")

#: What has and has not been executed on a device. Edited only by a released gate.
DEVICE_STATUS: str = (
    "RELEASED 2026-08-31 on the GPU host GPU 7 (verified physically empty before the "
    "run), RE-RUN in place after this paragraph was written -- the gate pins this "
    "module in source_sha256, so the run that certifies it has to be the one that "
    "saw the paragraph. "
    "run: 3 MiB, no compute process), NVIDIA RTX A6000, cc 8.6, Triton 3.1.0 / CuPy "
    "13.5.1, under the 'keep' float32 subnormal policy (ieee_keep_ftz_stripped) "
    "through an ftz-stripped cache: "
    "parity/meep_gpu/results/triton_beta_electric_2026-08-31d/"
    "beta_fused_electric_pair/gate.json, release.released = true. "
    "25 device legs, 611 COMPLETE driver steps, 593 fused launches, every one of 26 "
    "allocated volumes BIT-IDENTICAL on the uint32 view against BOTH oracles -- the "
    "CuPy array path and the two separately certified Triton products this launch "
    "replaces (the 'real beta PML' curl on step_D and the 'real beta run' "
    "constitutive on update_E). "
    "6 CARRY legs, each repairing 3 in-seam electric deposits through the SHIPPED "
    "deposit_repair bracket, and 2 NULL CONTROLS with the bracket removed that BOTH "
    "diverge as they must. 3 REDUCTION legs, one per value class, all DISCRIMINATING: "
    "HAS_BETA = 0 reproduces kernels.fused_curl_constitutive_D bit for bit while "
    "differing from the array path, so the flag is measured load-bearing rather than "
    "assumed. 14/14 kernel mutations caught, 3/3 host mutations caught (both "
    "coefficient-lattice swaps and the beta word exchange), 5/5 refusals, 4/4 "
    "no-device legs. "
    "|| WHAT THE RUN DID NOT ESTABLISH: nothing is TIMED here and no throughput "
    "claim is admissible; and the product is NOT WIRED, so nothing in a default run "
    "reaches this plan. This module has no fingerprints.json entry -- the gate "
    "artifact's own source_sha256 map is the binding, and the fusion matrix reads "
    "it.")

__all__ = [
    "BACKWARD", "CARRIES_DEPOSIT_REPAIR", "CONSTITUTIVE_SIDE", "CURL_SUB_STEP",
    "DEVICE_STATUS", "REPLACES",
    "BetaFusedElectricPairPlan",
    "beta_fused_curl_constitutive_D",
    "beta_fused_curl_constitutive_D_kernel",
    "beta_fused_electric_pair_coverage",
    "plan_beta_fused_electric_pair",
    "plan_beta_fused_electric_pair_from_arrays",
]


if triton is not None:

    # The same constexpr code as ``kernels.METALLIC``, restated for the reason every
    # sibling restates it — importing a ``tl.constexpr`` wrapper and re-wrapping it
    # is not the same object, and the comparison in the kernel body is against a
    # literal code. ONLY METALLIC IS NAMED: both certified bodies branch on
    # ``== METALLIC`` and take periodic as the else.
    METALLIC = tl.constexpr(1)

    # The forms, the fold ladder and the chunk record's word offsets, under the names
    # kernels.py binds them: the lifted body reads them exactly as the root pair does.
    from .forms import (axis_present, constitutive_form, curl_form,  # noqa: E402
                        curl_stencil, fold_ladder, record_in)
    from .kernels import RECORD_HALF_INTEGER, RECORD_PRESENCE  # noqa: E402

    @triton.jit
    def beta_fused_curl_constitutive_D(
        f0, f1, f2,                       # curl targets: Dx,Dy,Dz
        u0, u1, u2,                       # curl auxiliaries: fu_Dx,fu_Dy,fu_Dz
        g0, g1, g2,                       # curl sources: Hx,Hy,Hz
        kmx, sinvx, kmy, sinvy, kmz, sinvz,   # curl kms/sinv, INTEGER lattice
        e0, e1, e2,                       # constitutive targets: Ex,Ey,Ez
        w0, w1, w2,                       # constitutive aux: f_w_Ex,f_w_Ey,f_w_Ez
        ie0, ie1, ie2,                    # diagonal inverse epsilon volumes
        kp0, km0, kp1, km1, kp2, km2,     # constitutive kps/kms, HALF-INTEGER lattice
        cb,                               # chunk record: both halves' forms
        nx, ny, nz, n_elem, dtdx,
        beta_plus, beta_minus,            # f32(sign*2*pi*beta*dt), host-rounded once
        BACKWARD: tl.constexpr,           # bound to 1 by every builder
        BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
        ZM_X: tl.constexpr, ZM_Y: tl.constexpr, ZM_Z: tl.constexpr,
        HAS_BETA: tl.constexpr,           # 1 on every admitted configuration
        BLOCK: tl.constexpr,
    ):
        """Beta ``step_D`` + ``zero_metal_D`` + ``update_E``, one launch.

        ``kernels.fused_curl_constitutive_D``'s body, character for character, plus
        ``special_kz.beta_pml_curl_step``'s beta pass after the three curl forms and
        before the wall clear -- the array path's order (step_D's curl, its beta pass,
        then the seam, then update_E). The registers the pass leaves are the D words
        both the store and the E half read. ``HAS_BETA = 0`` builds
        ``kernels.fused_curl_constitutive_D`` exactly.
        """
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = idx < n_elem
        nyz = ny * nz
        k = idx % nz
        plane = idx // nz
        j = plane % ny
        i = plane // ny

        # ======================= the curl half ====================================
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

        presence = tl.load(cb + RECORD_PRESENCE)
        px = axis_present(presence, 0)
        py = axis_present(presence, 1)
        pz = axis_present(presence, 2)
        curl0 = curl_stencil(c_y, c, b, b_z, dtdx, py, pz)
        curl1 = curl_stencil(a_z, a, c, c_x, dtdx, pz, px)
        curl2 = curl_stencil(b_x, b, a, a_y, dtdx, px, py)

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

        km_x = tl.load(kmx + i, mask=live, other=0.0)
        si_x = tl.load(sinvx + i, mask=live, other=0.0)
        km_y = tl.load(kmy + j, mask=live, other=0.0)
        si_y = tl.load(sinvy + j, mask=live, other=0.0)
        km_z = tl.load(kmz + k, mask=live, other=0.0)
        si_z = tl.load(sinvz + k, mask=live, other=0.0)

        # Which cells take the recurrence: the component's dsigu layer, on the component's
        # own sub-lattice. The dsigu axis is z for target 0, x for 1 and y for 2 (vec.hpp
        # cycle_direction); the sub-lattice is half-integer for every B target and integer
        # for every D one (the iyee table), so each target's record line is a compile-time
        # word index, chunk_map.record_index(half_integer, axis, 0).
        if BACKWARD:
            lattice = 0
        else:
            lattice = RECORD_HALF_INTEGER
        du0 = record_in(cb, lattice + 8, k)
        du1 = record_in(cb, lattice + 0, i)
        du2 = record_in(cb, lattice + 4, j)

        p0 = tl.load(u0 + idx, mask=live & du0, other=0.0)
        v0, n0 = curl_form(tl.load(f0 + idx, mask=live, other=0.0), p0, curl0,
                           km_y, si_y, km_z, si_z, du0)

        p1 = tl.load(u1 + idx, mask=live & du1, other=0.0)
        v1, n1 = curl_form(tl.load(f1 + idx, mask=live, other=0.0), p1, curl1,
                           km_z, si_z, km_x, si_x, du1)

        p2 = tl.load(u2 + idx, mask=live & du2, other=0.0)
        v2, n2 = curl_form(tl.load(f2 + idx, mask=live, other=0.0), p2, curl2,
                           km_x, si_x, km_y, si_y, du2)

        # --- the special-kz beta pass (stepping._special_kz_beta_term, then
        # stepping._apply_folded_increment) ----------------------------------------
        # MEEP's pass of its own after every component's curl (step_db.cpp:161-175),
        # applied to the registers the forms returned. The partners are the CENTER
        # loads: target 0 takes the second source at sign +1 (Bx <- Ey, Dx <- Hy),
        # target 1 the first at sign -1 (By <- Ex, Dy <- Hx); target 2 gets nothing.
        # The increment is the NEGATED product (no dtdx: an analytic derivative),
        # masked to +0 on the cells the curl's own mask drops, and the beta ladder
        # subtracts it (forms.fold_ladder).
        if HAS_BETA:
            q0 = (beta_plus * b) * -1.0
            q1 = (beta_minus * a) * -1.0
            if BACKWARD:
                if BCY == METALLIC:
                    q0 = tl.where(at_y, 0.0, q0)
                if BCZ == METALLIC:
                    q0 = tl.where(at_z, 0.0, q0)
                if BCX == METALLIC:
                    q1 = tl.where(at_x, 0.0, q1)
                if BCZ == METALLIC:
                    q1 = tl.where(at_z, 0.0, q1)
            else:
                if BCX == METALLIC:
                    q0 = tl.where(at_x, 0.0, q0)
                if BCY == METALLIC:
                    q1 = tl.where(at_y, 0.0, q1)
            v0, n0 = fold_ladder(v0, n0, q0, si_y, si_z, du0)
            v1, n1 = fold_ladder(v1, n1, q1, si_z, si_x, du1)

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

        # fu is written on the recurrence's cells ONLY: outside them the dsig-only form
        # reads and writes the field alone and fu keeps its word (the auxiliary contract).
        tl.store(u0 + idx, n0, mask=live & du0)
        tl.store(u1 + idx, n1, mask=live & du1)
        tl.store(u2 + idx, n2, mask=live & du2)
        tl.store(f0 + idx, v0, mask=live)
        tl.store(f1 + idx, v1, mask=live)
        tl.store(f2 + idx, v2, mask=live)

        # ==================== the constitutive half ================================
        kp_0 = tl.load(kp0 + i, mask=live, other=0.0)
        km_0 = tl.load(km0 + i, mask=live, other=0.0)
        kp_1 = tl.load(kp1 + j, mask=live, other=0.0)
        km_1 = tl.load(km1 + j, mask=live, other=0.0)
        kp_2 = tl.load(kp2 + k, mask=live, other=0.0)
        km_2 = tl.load(km2 + k, mask=live, other=0.0)

        # Which cells take the accumulation: the component's own (dsigw) layer on the
        # component's own sub-lattice. The own axis is x for target 0, y for 1, z for 2
        # (the constitutive term tables), so each target's record line is again a
        # compile-time word index, chunk_map.record_index(half_integer, axis, 0).
        dw0 = record_in(cb, RECORD_HALF_INTEGER + 0, i)   # E: the half-integer sub-lattice
        dw1 = record_in(cb, RECORD_HALF_INTEGER + 4, j)
        dw2 = record_in(cb, RECORD_HALF_INTEGER + 8, k)
        prev0 = tl.load(w0 + idx, mask=live & dw0, other=0.0)
        src0 = v0 * tl.load(ie0 + idx, mask=live, other=0.0)
        tl.store(w0 + idx, src0, mask=live & dw0)
        a0 = constitutive_form(tl.load(e0 + idx, mask=live & dw0, other=0.0),
                                 prev0, src0, kp_0, km_0, dw0)
        tl.store(e0 + idx, a0, mask=live)

        prev1 = tl.load(w1 + idx, mask=live & dw1, other=0.0)
        src1 = v1 * tl.load(ie1 + idx, mask=live, other=0.0)
        tl.store(w1 + idx, src1, mask=live & dw1)
        a1 = constitutive_form(tl.load(e1 + idx, mask=live & dw1, other=0.0),
                                 prev1, src1, kp_1, km_1, dw1)
        tl.store(e1 + idx, a1, mask=live)

        prev2 = tl.load(w2 + idx, mask=live & dw2, other=0.0)
        src2 = v2 * tl.load(ie2 + idx, mask=live, other=0.0)
        tl.store(w2 + idx, src2, mask=live & dw2)
        a2 = constitutive_form(tl.load(e2 + idx, mask=live & dw2, other=0.0),
                                 prev2, src2, kp_2, km_2, dw2)
        tl.store(e2 + idx, a2, mask=live)

else:  # pragma: no cover - laptop path
    beta_fused_curl_constitutive_D = None  # type: ignore[assignment]


def beta_fused_curl_constitutive_D_kernel() -> Any:
    """The JIT kernel, or a named ImportError on a host without Triton."""
    if beta_fused_curl_constitutive_D is None:
        raise ImportError(
            "the beta fused electric D/E kernel needs the optional `triton` "
            f"package (pip install triton). Original error: {_TRITON_IMPORT_ERROR}")
    return beta_fused_curl_constitutive_D


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def beta_fused_electric_pair_coverage(fields: Any, pml: Any,
                                      sources: Any = None) -> Coverage:
    """May ONE launch span beta ``step_D`` -> wall -> ``update_E``?

    A conjunction of the two halves' own SHIPPED predicates plus the seam clauses.
    Nothing is weakened: a configuration either half refuses is refused here with
    that half's reasons, prefixed so a reader can tell which side said it — the
    construction :func:`.beta_fused_magnetic_pair.beta_fused_magnetic_pair_coverage`
    and :func:`.coverage.fused_pair_coverage` both use.

    The two halves are the arms ``launch.plan_step`` really selects on a beta row
    today (``real beta PML`` on ``step_D``, ``real beta run`` on ``update_E``), so
    this predicate is narrower than the pair the planner already puts on the device,
    never wider.
    """
    reasons: List[str] = []

    curl = beta_pml_curl_coverage(fields, pml, CURL_SUB_STEP)
    if not curl.covered:
        reasons.extend(f"beta curl half: {reason}" for reason in curl.reasons)
    constitutive = beta_run_constitutive_coverage(fields, pml, CONSTITUTIVE_SIDE)
    if not constitutive.covered:
        reasons.extend(f"beta constitutive half: {reason}"
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

    # THE TWO SYMMETRY PASSES. Both return at their first line without
    # `grid.has_symmetry()` (stepping.py:1481-1482, :1565-1566), and clause 5 of
    # `_beta_real_grid_reasons` already refuses every fold. RE-CHECKED HERE ANYWAY,
    # not inferred: this module's coverage may not be read off another module's
    # guard, and if the beta tranche ever admits a fold this weld would silently
    # swallow two passes that had started doing work.
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
    # configuration to fail. The MAGNETIC twin needs no such clause: update_H is
    # `H = B` with mu = 1 baked in and binds no inverse volume at all.
    if getattr(fields, "inverse_epsilon_for", None) is None:
        reasons.append(
            "fields does not expose inverse_epsilon_for; the constitutive half "
            "cannot bind the inv_eps volumes it loads")

    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class BetaFusedElectricPairPlan:
    """One allocation-free launch for five of the driver's electric call sites.

    :class:`.launch.FusedPairPlan`'s ``pair='D'`` bindings plus the two host-rounded
    beta scalars and the ``HAS_BETA`` constexpr, which is exactly the delta
    :class:`.special_kz.BetaPmlCurlPlan` carries over :class:`.launch.PmlCurlPlan`.
    The inverse-epsilon volumes are the E side's and are bound here; the magnetic
    twin has none.
    """

    __slots__ = ("chunk_bounds", "_cb", "shape", "n_elem", "dtdx", "beta_plus", "beta_minus", "has_beta",
                 "backward", "bc", "zero_metal", "block", "num_warps",
                 "_targets", "_aux", "_sources", "_curl_coefficients",
                 "_e_targets", "_e_aux", "_inverse_epsilon", "_e_coefficients",
                 "_grid", "_kernel")

    #: The five driver call sites one launch performs. Declared, so a composition can
    #: be inspected rather than inferred from the curl slot's name.
    replaces = REPLACES

    def __init__(self, shape, dtdx: float, bc, zero_metal,
                 beta_plus: float, beta_minus: float, block: int,
                 targets, auxiliaries, sources, curl_coefficients,
                 e_targets, e_aux, inverse_epsilon, e_coefficients,
                 kernel: Any = None, num_warps: Optional[int] = 1,
                 has_beta: int = 1, *, chunk_bounds: Any,
                 ) -> None:
        from .launch import CupyPointer, _flat, bind_chunk_bounds  # noqa: PLC0415

        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.chunk_bounds, self._cb = bind_chunk_bounds(
            chunk_bounds, self.shape, targets[0], "BetaFusedElectricPairPlan")
        self.dtdx = float(dtdx)
        # Already f32-rounded by the host (special_kz.beta_curl_coefficients);
        # float() keeps the bits, and Triton types a Python float argument as fp32.
        self.beta_plus = float(beta_plus)
        self.beta_minus = float(beta_minus)
        self.has_beta = int(has_beta)
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
                  else beta_fused_curl_constitutive_D_kernel())
        nx, ny, nz = self.shape
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        kernel[self._grid](
            *self._targets, *self._aux, *self._sources, *self._curl_coefficients,
            *self._e_targets, *self._e_aux, *self._inverse_epsilon,
            *self._e_coefficients, self._cb,
            nx, ny, nz, self.n_elem, self.dtdx,
            self.beta_plus, self.beta_minus,
            BACKWARD=self.backward,
            BCX=self.bc[0], BCY=self.bc[1], BCZ=self.bc[2],
            ZM_X=self.zero_metal[0], ZM_Y=self.zero_metal[1],
            ZM_Z=self.zero_metal[2],
            HAS_BETA=self.has_beta,
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )

    def __repr__(self) -> str:
        return (f"BetaFusedElectricPairPlan(shape={self.shape}, bc={self.bc}, "
                f"zero_metal={self.zero_metal}, beta_plus={self.beta_plus!r}, "
                f"has_beta={self.has_beta}, block={self.block}, "
                f"num_warps={self.num_warps})")


def plan_beta_fused_electric_pair(fields: Any, pml: Any, sources: Any = None,
                                  block: Optional[int] = None,
                                  num_warps: Optional[int] = 1,
                                  kernel: Any = None,
                                  chunk_layout: Any = None,
                                  ) -> Optional[BetaFusedElectricPairPlan]:
    """Build the plan from the engine's own objects, or ``None`` when refused.

    ``None`` is the only refusal: a configuration this kernel does not carry must
    fall back to the array path, never raise into a caller that would otherwise have
    stepped correctly.
    """
    if not beta_fused_electric_pair_coverage(fields, pml, sources).covered:
        return None
    from ..stepping import _boundary_kinds as resolve  # noqa: PLC0415
    from .kernels import DEFAULT_BLOCK  # noqa: PLC0415
    from .launch import SUB_STEPS, chunk_bounds_for  # noqa: PLC0415

    curl_spec = SUB_STEPS[CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    grid = fields.grid
    kinds = resolve(grid, pml)
    # THE SAME FUNCTION THE CURL PLAN CALLS, with `magnetic` from the sub-step —
    # identical arithmetic to the array path's per-call-site computation, and not a
    # second spelling of it. `magnetic=False` here and True in the twin, and on REAL
    # storage the two answers are MEASURED IDENTICAL (the ±i is the implicit-i trick
    # and lives only on the complex path). It is written this way because that is
    # what the array path computes per call site, not because the flag decides a
    # byte here.
    plus, minus = beta_curl_coefficients(grid.beta, grid.dt,
                                         magnetic=(CURL_SUB_STEP == "step_B"),
                                         complex_storage=False)
    return BetaFusedElectricPairPlan(
        grid.shape, grid.dt / grid.dx,
        [1 if kind == "metallic" else 0 for kind in kinds],
        zero_metal_axes(grid), plus, minus,
        DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in curl_spec["targets"]],
        [getattr(fields, "fu_" + name) for name in curl_spec["targets"]],
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
        chunk_bounds=chunk_bounds_for(fields, pml, chunk_layout),
    )


def plan_beta_fused_electric_pair_from_arrays(
        arrays: Dict[str, Any], curl_flat: Dict[str, Any],
        constitutive_flat: Dict[str, Any], codes, zero_metal,
        dtdx: float, beta_plus: float, beta_minus: float,
        block: Optional[int] = None, kernel: Any = None,
        num_warps: Optional[int] = 1, has_beta: int = 1, *,
        chunk_bounds: Any,
        ) -> BetaFusedElectricPairPlan:
    """Build from bare device arrays — the gate's route.

    No coverage predicate runs: the caller is a harness that constructed the
    configuration deliberately, including the deliberately wrong ones. ``arrays``
    additionally carries ``inv_eps_Ex``...; ``kernel=`` carries the mutation override
    and ``has_beta=0`` the identity leg's certified-kernel arm, which must reproduce
    ``kernels.fused_curl_constitutive_D`` bit for bit.
    """
    from .kernels import DEFAULT_BLOCK  # noqa: PLC0415
    from .launch import SUB_STEPS  # noqa: PLC0415

    curl_spec = SUB_STEPS[CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    shape = tuple(int(n) for n in arrays[curl_spec["targets"][0]].shape)
    return BetaFusedElectricPairPlan(
        shape, dtdx, codes, zero_metal, beta_plus, beta_minus,
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
        kernel=kernel, num_warps=num_warps, has_beta=has_beta,
        chunk_bounds=chunk_bounds,
    )
