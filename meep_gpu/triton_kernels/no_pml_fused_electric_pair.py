"""The real no-absorber ``step_D`` welded into stored-E ``update_E``, deposit carried.

ONE LAUNCH FOR ALL THREE COMPONENTS, and the stepped displacement never leaves a
register. The separate composition for this configuration is TWO launches plus an
array-path wall pass — one whole-grid curl, ``stepping.zero_metal_D``, one pointwise
stored-E — and this is one.

It occupies the last two BUILDABLE D->E cells on the Triton fusion board that are not a
stencil, not an arm that launches nothing and not worth zero:

    (conductive no-PML, no-PML stored E)   2 seam-instances
    (no-PML,            no-PML stored E)   1 seam-instance

measured at ``ceiling 2`` and ``ceiling 1`` in
``parity/meep_gpu/results/fusion_matrix_triton_2026-08-31_final/``, whose "BUILDABLE
GAPS WITH A MEASURED RECIPE" list holds five seam-instances in four cells; these are
three of them, and the other two are the folded-beta pair, a different recipe entirely.
The three corpus rows are ``examples:absorber-1d.py``,
``tests:TestAbsorber.test_absorber`` (both conductive, 5 poles per component, METALLIC
on z) and ``examples:material-dispersion.py`` (lossless, 2 poles per component,
periodic on all three).

IT SERVES ALL THREE AS OF 2026-09-01. Until that day it served one: the blocker on
the two conductive rows was NOT the deposit -- that is what
:data:`..deposit_repair.PLAIN_PATH` carries -- but a SECOND pass in the same seam:
``FdtdDriver._inject_electric_through_conductivity`` rescaled a WHOLE D volume for a
NON-INTEGRATED source on a conductive run, at cells no deposit closure can name
(measured: 81 D words moved, 66 E words divergent on step 1 of a signed-zero seed).
THE DRIVER NO LONGER DOES THAT — it replays the condinv rescale sparsely at the
deposit cells the sources publish, identity everywhere else — and
:func:`_scaled_conductive_injection_reasons` carries both the retired measurement
and the lift; the clause survives only for a duck-typed source publishing no
deposit table, which no in-tree source class is. The lossless row never took the
rescale at all (without a conductivity the driver takes the plain per-source
injection loop).

So the honest arithmetic for this module is **+3 of 387 seam-instances** (+1 at the
first release, +2 at the 2026-09-01 lift). The kernel itself implements the
conductive curl -- its ``COND`` constexprs are live, the gate drives them -- so
what was priced pre-lift was the driver's seam pass, not the arithmetic.

===========================================================================
WHY THIS CELL WAS UNBUILDABLE UNTIL TODAY, AND WHAT CHANGED
===========================================================================

ALL THREE ROWS DECLARE AN IN-SEAM ELECTRIC SOURCE. The driver injects it between
``step_D`` and ``update_E`` (``driver.py:3294-3299``, and on the two conductive rows
through ``_inject_electric_through_conductivity`` at ``:3305``), so a fused pair
consumes a pre-injection D and must carry the deposit across its own launch or serve
nothing. That is what ``deposit_repair`` is for — and until 2026-08-31 it could not do
it here. ``repairable(fields, "D", pml=None)`` returned False on a plain-path engine
with two named refusals, because the ONLY repair that existed inverted the SPLIT-FIELD
recurrence ``(E + kps*fresh) - kms*fw_prev``, which is not the recurrence this
configuration runs.

:data:`..deposit_repair.PLAIN_PATH` is the second repair, and it inverts the branch
that DOES run here: ``update_E`` with an inactive layer writes
``E[...] = (D - sum P) * inv_eps`` straight to storage (``stepping.py:1019-1022``), a
PURE OVERWRITE with no ``f_w``, no coefficient and no previous value — so the repair is
one recomputation per component at the deposit cells and there is no state to save at
all. The arithmetic, what makes it exact, and what was measured before it was written
are in that module's docstring; the numbers behind this product's own use of it are in
``parity/meep_gpu/probe_plain_deposit_repair.py`` and this family's device gate.

:data:`CARRIES_DEPOSIT_REPAIR` is therefore True here, and :data:`REPAIR_PATHS` names
WHICH repair the two slots install. Both travel together and are only ever changed in
the same edit as the wiring: a family that declared the flag without the bracket would
compute its constitutive half against a pre-injection field and report success.

===========================================================================
THE SEAM, PASS BY PASS — what is carried, what is refused, what is inert
===========================================================================

The driver runs four passes between ``step_D`` and ``update_E``
(``driver.py:3300-3311``). This family's disposition of each is a clause, not a
comment:

* **the electric injection — CARRIED, through the deposit repair.** The pair owns both
  slots; ``launch._install_fused_pair`` puts a
  :class:`..deposit_repair.LeadingRepairPlan` in ``step_D`` and a
  :class:`..deposit_repair.TrailingRepairPlan` in ``update_E``, so the launch runs
  against an uninjected D and the repair rewrites E at the deposit cells once the
  injection, the fills and the wall clear are final. IGNORANCE IS STILL A REFUSAL:
  ``Fields`` does not hold the source list, so an undeclared ``sources`` is refused
  rather than read as an empty seam;
* **``zero_metal_D`` — CARRIED INLINE, and it had to be.** It writes zero into stored
  cell 0 of the two TANGENTIAL D components of every metallic axis
  (``stepping.py:2206-2247``) and ``update_E`` then reads them. Two of this cell's
  three rows are METALLIC on z, so declining the wall the way
  :mod:`.complex_conductive_fused_pair` declines it — priced there at zero rows — would
  cost two thirds of this cell. The three ``ZM_*`` constexprs below are
  :func:`.kernels.fused_curl_constitutive_D`'s own, transcribed with their comment, and
  they are applied to the REGISTER before both the store and the constitutive read so
  the two consumers see the one value the array path leaves in D;
* **``fill_symmetry_bc_D`` — INERT, because a mirror plane is refused.** Both certified
  halves refuse a live mirror already; the clause is restated by name here because a
  reader should not have to chase a shared grid predicate to learn which driver pass
  that refusal empties. Without a mirror the array path's fill returns at its first line
  (``stepping.py:1481-1482``);
* **``fill_folded_far_ghosts_D`` — INERT for the same reason**
  (``stepping.py:1565-1566``).

===========================================================================
WHY THERE IS NO STENCIL PROBLEM HERE, and why that is not general
===========================================================================

The board refuses EIGHTEEN D->E seam-instances as structurally unfusable:
``stepping._offdiagonal_terms`` (``stepping.py:1235-1253``) reads each partner
component's D volume at FOUR indices while ``step_D`` writes those volumes in place in
the first half of the same launch, and no grid-wide barrier exists inside one launch.
THIS CONSTITUTIVE IS NOT THAT ONE: the certified stored-E body reads its OWN component
at its OWN index — the word the same program just computed — so the value stays in a
register and no lane reads a word another lane writes. The E half refuses an
off-diagonal ``chi1inv`` row already; it is named again below because THIS is the fact
that makes the cell buildable at all.

THE POLE ARRAYS ARE READ, NEVER WRITTEN, by either half. ``update_P`` runs after
``update_E`` (``driver.py:3314``) and is not in this seam, so the ``P`` volumes this
kernel loads are the ones the previous timestep left — exactly as the certified
stored-E body loads them, and exactly what the repair's own recomputation reads.

===========================================================================
EVERYTHING HERE IS TRANSCRIBED, and from where
===========================================================================

* the curl half — :func:`.no_pml_conductive.conductive_plain_curl_step`, character for
  character through its ghost rule, its curl grouping, its ownership mask and its
  per-target conductive tail. That body is itself byte-copied from
  :func:`.no_pml.plain_curl_step` above the tail, and with ``COND0 == COND1 == COND2 ==
  0`` it compiles to that kernel's own lines — which is what lets ONE product occupy
  BOTH board cells rather than two products occupying one each. The lossless cell is
  not a widening: it is the same kernel with three constexprs at zero, and
  ``test_triton_no_pml_conductive`` already diffs the two curl bodies textually so they
  cannot drift. WHICH of the two arms answers for a given run is decided by the arms'
  own partition function (:func:`_curl_half`), never by asking both;
* the wall — :func:`.kernels.fused_curl_constitutive_D`'s ``ZM_*`` block, the D-side
  form (each axis clears the two TANGENTIAL components, not the normal one);
* the constitutive half — :func:`.no_pml_stored_e.stored_e_constitutive_step`, whose
  three unrolled subtraction chains and ``s * inv_eps`` store transcribe
  ``stepping.update_E:981-984`` then ``:993`` through
  ``Fields.displacement_minus_polarization``.

WHAT MOVES IS ONE LINE PER COMPONENT. The certified stored-E body opens each component
with ``s0 = tl.load(g0 + idx, ...)`` — the stepped D. The curl half has already left
that value in a register, so ``s0 = v0`` is the whole weld, the same substitution
:func:`.kernels.fused_curl_constitutive_D` makes on the PML path. THE D STORE IS KEPT:
the array path leaves the stepped displacement in the D volume, the next timestep's
curl and ``update_P`` read it, and a weld that dropped the store would be fusing away a
value the run still needs.

``DERIVE`` IS CARRIED AND BOUND TO ZERO. The certified curl body takes it because
``step_B`` may have to form ``D * inv_eps`` itself when E is not stored
(``stepping._read_component``, ``:2383-2396``); ``step_D`` differences B directly and
never takes that arm. It is kept in the signature rather than deleted so the
transcription is the certified body and not an edit of it, and the plan binds 0.

A reader can diff this body against those three and see that nothing moved. If you find
yourself deriving here, you have taken a wrong turn.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .coverage import Coverage, _boundary_kinds, _call, zero_metal_axes
from .no_pml import plain_curl_coverage
from .no_pml_conductive import (
    SUB_STEPS as _NO_PML_SUB_STEPS,
    conductive_no_pml_targets,
    conductive_plain_curl_coverage,
)
from .no_pml_stored_e import (
    E_TERMS,
    MAX_POLES,
    LivePoleBinding,
    StaticPoleBinding,
    poles_per_component,
    stored_e_constitutive_coverage,
)
from .. import deposit_repair as _deposit_repair

#: The family name, as the board, the battery and the weld record spell it.
FAMILY = "no_pml_fused_electric_pair"

#: Does this product bracket its fused launch with the deposit repair? TRUE, and here
#: that is the difference between three seam-instances and zero: every one of this
#: cell's three corpus rows declares an ELECTRIC source, which the driver injects
#: INSIDE this seam. The flag is a claim about the PLAN this module builds — that the
#: leading slot saves and the trailing slot restores — and is only ever changed in the
#: same edit as that wiring.
CARRIES_DEPOSIT_REPAIR = True

#: WHICH repair the two slots install. :data:`..deposit_repair.PLAIN_PATH` alone: this
#: family refuses an active absorber in BOTH halves, so the recurrence its seam has to
#: invert is always ``update_E``'s plain overwrite and never the split-field one. The
#: declaration is not a preference — ``deposit_repair.repairable`` refuses by name any
#: configuration whose recurrence is not among the paths its caller declares, so a
#: product that named the wrong one would be refused rather than silently mis-repaired.
REPAIR_PATHS: Tuple[str, ...] = (_deposit_repair.PLAIN_PATH,)

#: The curl sub-step this product starts at. There is no ``CONSTITUTIVE_SIDE``
#: constant: the stored-E half is not one of ``coverage.CONSTITUTIVE_SIDES`` — it is the
#: no-PML branch of ``update_E``, which writes E directly from ``(D - sum P) * inv_eps``
#: and touches no ``f_w`` or PML lattice at all.
CURL_SUB_STEP = "step_D"

#: The certified curl arm's own row for that sub-step.
_CURL_SPEC = _NO_PML_SUB_STEPS[CURL_SUB_STEP]

#: The three volumes the curl half writes, and the three it differences -- READ from
#: the certified curl arm's OWN table, never spelled.
#:
#: THE TABLE MATTERS AND THE TWO IN THIS PACKAGE DISAGREE ON PURPOSE.
#: ``launch.SUB_STEPS['step_D']['sources']`` is ``('Hx','Hy','Hz')``, which is right
#: under an absorber where H is stored; ``no_pml_conductive.SUB_STEPS`` says
#: ``('Bx','By','Bz')`` and says why (``no_pml.py:178-181``): with no absorber
#: ``Fields.enable_field_storage`` deliberately does not allocate H at all and
#: ``get_H`` returns the B array itself, so the source pointers ARE the B pointers.
#: Spelling either here would be a restatement that can go stale against the arm this
#: kernel's curl half IS, so both are read from that arm.
CURL_TARGETS: Tuple[str, ...] = tuple(_CURL_SPEC["targets"])
CURL_SOURCES: Tuple[str, ...] = tuple(_CURL_SPEC["sources"])

#: The driver call sites ONE launch of this plan performs, in driver order. THREE, not
#: five: the two symmetry fills are REFUSED by the fold clauses rather than carried, and
#: a launch may only claim to replace what it actually performs. ``update_E`` is listed
#: because the launch computes it; the deposit repair that occupies the SAME slot when
#: the seam carries a source is a correction to that computation, not a fourth call
#: site.
REPLACES: Tuple[str, ...] = ("step_D", "zero_metal_D", "update_E")

#: ``SUB_STEPS['step_D']['backward']``, read from the same arm table above so the
#: kernel's one legal binding cannot drift from the body it was transcribed out of.
BACKWARD = int(_CURL_SPEC["backward"])

#: ``DERIVE`` is a ``step_B`` question and this product is ``step_D`` only. Restated as
#: a constant so the binding is a declaration a test can pin rather than a literal
#: buried in the builder.
DERIVE = 0

__all__ = [
    "BACKWARD", "CARRIES_DEPOSIT_REPAIR", "CURL_SOURCES", "CURL_SUB_STEP",
    "CURL_TARGETS", "DERIVE", "FAMILY", "REPAIR_PATHS", "REPLACES",
    "NoPmlFusedElectricPairPlan",
    "no_pml_fused_curl_constitutive_D",
    "no_pml_fused_curl_constitutive_D_kernel",
    "no_pml_fused_electric_pair_coverage",
    "plan_no_pml_fused_electric_pair",
    "plan_no_pml_fused_electric_pair_from_arrays",
]


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------
#
# Defined at MODULE scope behind a conditional import, not inside a builder:
# `@triton.jit` resolves a body's names — including the `tl.constexpr` annotations —
# through the defining module's `__globals__`, so a kernel defined inside a function
# compiles to `NameError('tl is not defined')` at first launch (dispersive_update_e.py
# :138-143 records the run that showed it, 320 of 320 cases).

try:  # pragma: no cover - the absent branch is exercised by the absence test
    import triton
    import triton.language as tl
    _TRITON_IMPORT_ERROR: Optional[BaseException] = None
except ImportError as _exc:  # pragma: no cover - the laptop path
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_IMPORT_ERROR = _exc


if triton is not None:  # pragma: no cover - device code, certified by the byte gate

    #: Boundary codes, identical to ``kernels``', ``no_pml``'s and
    #: ``no_pml_conductive``'s own. A plan built here and a plan built there index the
    #: same table; a test pins them equal.
    PERIODIC = tl.constexpr(0)
    METALLIC = tl.constexpr(1)

    @triton.jit
    def no_pml_fused_curl_constitutive_D(
        f0, f1, f2,                       # curl targets: Dx, Dy, Dz          (in/out)
        g0, g1, g2,                       # curl sources: Bx, By, Bz          (in)
        e0, e1, e2,                       # inverse epsilon, per E component  (in)
        cf0, cf1, cf2,                    # condfac (a placeholder where COND == 0)
        ci0, ci1, ci2,                    # condinv (a placeholder where COND == 0)
        h0, h1, h2,                       # constitutive targets: Ex, Ey, Ez  (out)
        a0, a1, a2, a3, a4, a5, a6, a7,   # component 0's poles, IN ORDER
        b0, b1, b2, b3, b4, b5, b6, b7,   # component 1's poles, IN ORDER
        c0, c1, c2, c3, c4, c5, c6, c7,   # component 2's poles, IN ORDER
        nx, ny, nz, n_elem, dtdx,
        BACKWARD: tl.constexpr,           # 1 = D (backward differences)
        DERIVE: tl.constexpr,             # 0 here; step_D differences B directly
        BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
        ZM_X: tl.constexpr, ZM_Y: tl.constexpr, ZM_Z: tl.constexpr,
        COND0: tl.constexpr, COND1: tl.constexpr, COND2: tl.constexpr,
        NP0: tl.constexpr, NP1: tl.constexpr, NP2: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """``step_D`` + ``zero_metal_D`` + stored-E ``update_E``, in one launch.

        The curl half is :func:`.no_pml_conductive.conductive_plain_curl_step`, the
        wall is :func:`.kernels.fused_curl_constitutive_D`'s ``ZM_*`` block, and the
        constitutive half is
        :func:`.no_pml_stored_e.stored_e_constitutive_step` with each component's
        opening ``tl.load(D + idx)`` replaced by the register the curl left. Nothing
        else moves; the module docstring says where each block came from.

        ``ENABLE_FP_FUSION`` applies and matters MORE on the conductive tail than on
        the lossless one. ``((f * cf) - curl) * ci`` is a multiply feeding a subtract
        feeding a multiply, with ``curl`` itself a multiply — three separate
        contraction sites where the plain tail has one. DO NOT FLATTEN THE
        PARENTHESES: the array path performs three SEQUENTIAL in-place passes
        (``stepping.py:1996``, ``:1950``, ``:1951``), each rounding to float32 before
        the next reads it.
        """
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = idx < n_elem
        nyz = ny * nz
        k = idx % nz
        plane = idx // nz
        j = plane % ny
        i = plane // ny

        # --- the ghost rule, per axis (stepping._shift_up / _shift_down) --------
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

        # --- the source operands ------------------------------------------------
        # DERIVE forms stepping._read_component's D * inv_eps here rather than
        # reading a scratch volume the host wrote. D ON THE LEFT — the array path's
        # operand order (stepping.py:2450), kept literally. Bound to 0 by this
        # family's plan; see DERIVE's own note above.
        if DERIVE:
            a = tl.load(g0 + idx, mask=live, other=0.0) * tl.load(e0 + idx, mask=live, other=0.0)
            b = tl.load(g1 + idx, mask=live, other=0.0) * tl.load(e1 + idx, mask=live, other=0.0)
            c = tl.load(g2 + idx, mask=live, other=0.0) * tl.load(e2 + idx, mask=live, other=0.0)
            a_y = tl.load(g0 + oy, mask=vy, other=0.0) * tl.load(e0 + oy, mask=vy, other=0.0)
            a_z = tl.load(g0 + oz, mask=vz, other=0.0) * tl.load(e0 + oz, mask=vz, other=0.0)
            b_x = tl.load(g1 + ox, mask=vx, other=0.0) * tl.load(e1 + ox, mask=vx, other=0.0)
            b_z = tl.load(g1 + oz, mask=vz, other=0.0) * tl.load(e1 + oz, mask=vz, other=0.0)
            c_x = tl.load(g2 + ox, mask=vx, other=0.0) * tl.load(e2 + ox, mask=vx, other=0.0)
            c_y = tl.load(g2 + oy, mask=vy, other=0.0) * tl.load(e2 + oy, mask=vy, other=0.0)
        else:
            a = tl.load(g0 + idx, mask=live, other=0.0)
            b = tl.load(g1 + idx, mask=live, other=0.0)
            c = tl.load(g2 + idx, mask=live, other=0.0)
            a_y = tl.load(g0 + oy, mask=vy, other=0.0)
            a_z = tl.load(g0 + oz, mask=vz, other=0.0)
            b_x = tl.load(g1 + ox, mask=vx, other=0.0)
            b_z = tl.load(g1 + oz, mask=vz, other=0.0)
            c_x = tl.load(g2 + ox, mask=vx, other=0.0)
            c_y = tl.load(g2 + oy, mask=vy, other=0.0)

        # --- the curl (stepping._curl_from_operands): DO NOT flatten these parens -
        curl0 = dtdx * ((c_y - c) + (b - b_z))
        curl1 = dtdx * ((a_z - a) + (c - c_x))
        curl2 = dtdx * ((b_x - b) + (a - a_y))

        # --- ownership mask (stepping._mask_non_owned_cells) --------------------
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

        # --- the tail, PER COMPONENT (stepping.py:536-539) ----------------------
        # COND == 0 emits no condfac load, no condinv load and no multiply: the
        # component takes ``no_pml.plain_curl_step``'s own line, which is what makes
        # a mixed launch identical on its lossless components by construction.
        v0 = tl.load(f0 + idx, mask=live, other=0.0)
        if COND0:
            v0 = ((v0 * tl.load(cf0 + idx, mask=live, other=0.0)) - curl0) \
                * tl.load(ci0 + idx, mask=live, other=0.0)
        else:
            v0 = v0 - curl0

        v1 = tl.load(f1 + idx, mask=live, other=0.0)
        if COND1:
            v1 = ((v1 * tl.load(cf1 + idx, mask=live, other=0.0)) - curl1) \
                * tl.load(ci1 + idx, mask=live, other=0.0)
        else:
            v1 = v1 - curl1

        v2 = tl.load(f2 + idx, mask=live, other=0.0)
        if COND2:
            v2 = ((v2 * tl.load(cf2 + idx, mask=live, other=0.0)) - curl2) \
                * tl.load(ci2 + idx, mask=live, other=0.0)
        else:
            v2 = v2 - curl2

        # --- the seam: stepping.zero_metal_D, between the two sub-steps ---------
        # Applied to the REGISTER, before both the store and the constitutive read, so
        # the two consumers see the one value the array path leaves in D.
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

        tl.store(f0 + idx, v0, mask=live)
        tl.store(f1 + idx, v1, mask=live)
        tl.store(f2 + idx, v2, mask=live)

        # ==================== the constitutive half ============================
        # no_pml_stored_e.stored_e_constitutive_step, with each component's opening
        # `s = tl.load(D + idx)` replaced by the register the curl left. THE POLES
        # ARE NOT PRE-SUMMED and are NOT REORDERED: `(D - P0) - P1` is a different
        # float32 number from `D - (P0 + P1)`, caught 20/20 at two poles.

        # --- component 0 -------------------------------------------------------
        s0 = v0
        if NP0 > 0:
            s0 = s0 - tl.load(a0 + idx, mask=live, other=0.0)
        if NP0 > 1:
            s0 = s0 - tl.load(a1 + idx, mask=live, other=0.0)
        if NP0 > 2:
            s0 = s0 - tl.load(a2 + idx, mask=live, other=0.0)
        if NP0 > 3:
            s0 = s0 - tl.load(a3 + idx, mask=live, other=0.0)
        if NP0 > 4:
            s0 = s0 - tl.load(a4 + idx, mask=live, other=0.0)
        if NP0 > 5:
            s0 = s0 - tl.load(a5 + idx, mask=live, other=0.0)
        if NP0 > 6:
            s0 = s0 - tl.load(a6 + idx, mask=live, other=0.0)
        if NP0 > 7:
            s0 = s0 - tl.load(a7 + idx, mask=live, other=0.0)
        tl.store(h0 + idx, s0 * tl.load(e0 + idx, mask=live, other=0.0), mask=live)

        # --- component 1 -------------------------------------------------------
        s1 = v1
        if NP1 > 0:
            s1 = s1 - tl.load(b0 + idx, mask=live, other=0.0)
        if NP1 > 1:
            s1 = s1 - tl.load(b1 + idx, mask=live, other=0.0)
        if NP1 > 2:
            s1 = s1 - tl.load(b2 + idx, mask=live, other=0.0)
        if NP1 > 3:
            s1 = s1 - tl.load(b3 + idx, mask=live, other=0.0)
        if NP1 > 4:
            s1 = s1 - tl.load(b4 + idx, mask=live, other=0.0)
        if NP1 > 5:
            s1 = s1 - tl.load(b5 + idx, mask=live, other=0.0)
        if NP1 > 6:
            s1 = s1 - tl.load(b6 + idx, mask=live, other=0.0)
        if NP1 > 7:
            s1 = s1 - tl.load(b7 + idx, mask=live, other=0.0)
        tl.store(h1 + idx, s1 * tl.load(e1 + idx, mask=live, other=0.0), mask=live)

        # --- component 2 -------------------------------------------------------
        s2 = v2
        if NP2 > 0:
            s2 = s2 - tl.load(c0 + idx, mask=live, other=0.0)
        if NP2 > 1:
            s2 = s2 - tl.load(c1 + idx, mask=live, other=0.0)
        if NP2 > 2:
            s2 = s2 - tl.load(c2 + idx, mask=live, other=0.0)
        if NP2 > 3:
            s2 = s2 - tl.load(c3 + idx, mask=live, other=0.0)
        if NP2 > 4:
            s2 = s2 - tl.load(c4 + idx, mask=live, other=0.0)
        if NP2 > 5:
            s2 = s2 - tl.load(c5 + idx, mask=live, other=0.0)
        if NP2 > 6:
            s2 = s2 - tl.load(c6 + idx, mask=live, other=0.0)
        if NP2 > 7:
            s2 = s2 - tl.load(c7 + idx, mask=live, other=0.0)
        tl.store(h2 + idx, s2 * tl.load(e2 + idx, mask=live, other=0.0), mask=live)

else:  # pragma: no cover - the laptop path
    no_pml_fused_curl_constitutive_D = None  # type: ignore[assignment]


def no_pml_fused_curl_constitutive_D_kernel() -> Any:
    """The shipped ``@triton.jit`` kernel, or a diagnosable ImportError.

    The accessor exists so a caller that needs the kernel gets an explanation rather
    than a ``None`` that fails later as a ``TypeError`` far from its cause.
    """
    if no_pml_fused_curl_constitutive_D is None:
        raise ImportError(
            "the no-PML fused electric pair needs the optional `triton` package and "
            "a CUDA device; import failed with: "
            f"{_TRITON_IMPORT_ERROR!r}")
    return no_pml_fused_curl_constitutive_D


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------


def _curl_half(fields: Any, pml: Any) -> Tuple[bool, Tuple[str, ...], str]:
    """The curl verdict, from THE ONE certified no-absorber arm that speaks for this run.

    THE KERNEL IS BOTH ARMS, and that is a property of the body above rather than a
    claim about them: it is ``conductive_plain_curl_step``, whose per-component ``COND``
    constexpr compiles to ``no_pml.plain_curl_step``'s exact line when the component
    carries no sigma, so with all three at zero the compiled curl IS the lossless
    kernel. That is what lets ONE product occupy BOTH board cells.

    BUT ONLY ONE ARM IS ASKED, and the question that chooses it is the arms' OWN
    partition function. ``no_pml_conductive._any_curl_conductivity`` is the exact
    predicate the conductive arm refuses on by name ("no curl target carries a
    conductivity") and the lossless arm's own sigma clauses are its complement, so
    reading it here asks the arm that would have won rather than asking both and
    taking either.

    ASKING BOTH WAS THE FIRST SHAPE OF THIS FUNCTION AND IT WAS WRONG, in a way only a
    census cut showed. On a host with no CuPy NEITHER arm admits, so both reason lists
    came back — and the LOSING arm's clauses ("a conductivity is installed on Dx", or
    "no curl target carries a conductivity") are refusals of an arm this run was never
    going to take. Any reader that strips the backend clause and asks whether anything
    is left — which is what the predicate census's ``covered_modulo_backend`` does —
    then read every row as refused, and the board would have priced this product's
    cells at zero for a reason that is not a property of the run. Reading the partition
    first removes the ambiguity instead of papering over it.

    Returns ``(covered, reasons, which)``; ``reasons`` are the chosen arm's, labelled
    with which arm gave them.
    """
    from .no_pml_conductive import _any_curl_conductivity  # noqa: PLC0415

    try:
        conductive = bool(_any_curl_conductivity(fields))
    except Exception as error:  # noqa: BLE001 - unreadable is a refusal
        return False, (f"curl half: which no-absorber arm serves this run could not be "
                       f"established ({error!r}); "
                       f"no_pml_conductive._any_curl_conductivity is the partition the "
                       f"two arms refuse on, and a family that cannot read it has not "
                       f"established which body its COND constexprs compile to",), ""
    which = "conductive no-PML" if conductive else "no-PML"
    verdict = (conductive_plain_curl_coverage(fields, pml, CURL_SUB_STEP) if conductive
               else plain_curl_coverage(fields, pml, CURL_SUB_STEP))
    if verdict.covered:
        return True, (), which
    return False, tuple(f"curl half ({which} arm): {reason}"
                        for reason in verdict.reasons), which


def _scaled_conductive_injection_reasons(fields: Any, sources: Any) -> Tuple[str, ...]:
    """Why a scaled electric source with NO deposit table is still refused BY NAME.

    THE CLAUSE THAT USED TO PRICE THIS CELL AT 1 OF 3, LIFTED ON A MEASUREMENT
    (2026-09-01). ``FdtdDriver._inject_electric_through_conductivity`` injects an
    INTEGRATED source point-wise; for a SCALED (non-integrated) one it used to
    rescale the WHOLE target volume by difference::

        before = D.copy() ; inject ; D -= before ; D *= condinv ; D += before

    At a cell the injection did not touch that is the identity for every finite
    value EXCEPT ``-0.0`` (``x - x`` is ``+0.0``), so the pass silently
    canonicalised every negative zero in the target D component at cells no
    deposit closure can name — MEASURED on this product's own protocol (2-D
    1.2x1.2 at resolution 12, metallic x/y, sigma_D = 0.4, two Lorentz poles,
    signed-zero-lattice seed) as **81 Dz words moved** by the driver's own pass
    on step 1 and **66 Ez words divergent** from the array path at the end of
    that same step, every one a ``+0.0``/``-0.0`` pair — and identical from step
    2 on, the wash-out that made per-step comparison load-bearing.

    THE DRIVER NO LONGER DOES THAT. It snapshots the target at the deposit cells
    the sources publish (``deposit_repair._deposit_index`` reads the same
    tables), injects, and replays the rescale per deposit cell in the
    whole-volume passes' exact operand order — exact at every deposit cell,
    UNTOUCHED everywhere else, and closer to stock MEEP (step.cpp:294-317 scales
    only the injected current). RE-MEASURED with the sparse replay in place,
    same protocol, same signed-zero seed class: the bracketed fused route is
    byte-identical to the array path at EVERY step, and the retired whole-volume
    passes replayed as a control still diverge — the ``lifted_refusal`` leg of
    ``probe_triton_no_pml_fused_electric_pair.py`` carries both measurements.

    WHAT SURVIVES is the driver's own fallback: a scaled source that publishes NO
    deposit table at all (no ``_point_ix`` attribute) still takes the
    whole-volume difference passes, because an unnameable deposit must still be
    scaled. No in-tree electric source class is one — every source publishes
    ``_point_ix/_point_iy/_point_iz`` at setup — so on this corpus the clause
    refuses nothing; it stays because a duck-typed source without the table
    would otherwise reintroduce the divergence silently.

    THE ROWS THIS LIFT SERVES: ``examples:absorber-1d.py`` and
    ``tests:TestAbsorber.test_absorber`` — the same configuration twice (an
    ``mp.Absorber``, ``meep.materials.Al``, one non-integrated ``mp.Ex``
    source), each of which the pre-lift clause refused by name.

    FAIL CLOSED: a source whose ``is_integrated`` cannot be read, and a ``Fields``
    that cannot answer ``condinv_for``, are both refusals -- the pass runs on a
    property of the run, and a family that cannot establish it has not
    established it.
    """
    if sources is None:
        return ()  # the undeclared-source refusal is the seam clause's, not this one
    if not bool(getattr(fields, "has_conductivity", False)):
        # THE DRIVER'S OWN GUARD, restated: `if electric and fields.has_conductivity`
        # (driver.py:3304). Without a conductivity the scaled branch cannot run at all
        # and every electric source takes the plain per-source inject loop (:3307).
        return ()
    reader = getattr(fields, "condinv_for", None)
    if not callable(reader):
        return ("fields does not expose condinv_for, so whether "
                "FdtdDriver._inject_electric_through_conductivity would rescale a "
                "whole D volume inside this seam (driver.py:3363-3370) cannot be "
                "established",)
    reasons: List[str] = []
    for index, source in _deposit_repair._in_seam_indexed(sources, "D"):
        try:
            integrated = bool(source.is_integrated)
        except Exception as error:  # noqa: BLE001 - unreadable is a refusal
            reasons.append(
                f"source {index} ({type(source).__name__}) could not answer "
                f"is_integrated ({error!r}), so whether the driver would rescale a "
                f"whole D volume inside this seam cannot be established")
            continue
        if integrated:
            continue
        component = getattr(source, "component", None)
        try:
            target = "D" + str(component)[1]
            scaled = reader(target) is not None
        except Exception as error:  # noqa: BLE001 - unreadable is a refusal
            reasons.append(
                f"source {index} drives {component!r}, whose conductivity could not "
                f"be read ({error!r})")
            continue
        if not scaled:
            continue
        # THE DRIVER'S OWN PARTITION: a published deposit table takes the sparse
        # per-cell replay (identity away from the deposit); only the table-less
        # source falls back to the whole-volume passes.
        if not hasattr(source, "_point_ix"):
            reasons.append(
                f"source {index} ({type(source).__name__}) is electric, NOT "
                f"integrated, on a run whose {target} carries a conductivity, and "
                f"publishes NO deposit table (no _point_ix): the driver's "
                f"fallback rescales the WHOLE {target} volume inside this seam "
                f"(_inject_electric_through_conductivity's dense branch), which "
                f"canonicalises every -0.0 at cells the deposit closure cannot "
                f"name; the fused launch has already computed E from the "
                f"pre-rescale values there")
    return tuple(reasons)


def no_pml_fused_electric_pair_coverage(fields: Any, pml: Any,
                                        sources: Any = None) -> Coverage:
    """May ONE launch span no-absorber ``step_D`` -> ``zero_metal_D`` -> ``update_E``?

    A conjunction of the two halves' OWN certified predicates plus the seam clauses.
    Nothing is weakened: a configuration either half refuses is refused here with that
    half's reasons, prefixed so a reader can tell which side said it — the construction
    :func:`.complex_conductive_fused_pair.complex_conductive_fused_pair_coverage` and
    :func:`.coverage.fused_pair_coverage` use.

    THE ONE PLACE THIS DIFFERS FROM ITS SIBLINGS is that the curl half is served by TWO
    certified arms rather than one, and :func:`_curl_half` reads the arms' own partition
    function to ask the one that serves this run — so exactly one arm's verdict is this
    predicate's verdict, and exactly one arm's reasons are its reasons.
    """
    reasons: List[str] = []

    covered, curl_reasons, _which = _curl_half(fields, pml)
    if not covered:
        reasons.extend(curl_reasons)
    electric = stored_e_constitutive_coverage(fields, pml)
    if not electric.covered:
        reasons.extend(f"E half: {reason}" for reason in electric.reasons)

    # THE SOURCE SEAM, CARRIED. An ELECTRIC source is injected BETWEEN the two halves
    # (driver.py:3294-3299, and through _inject_electric_through_conductivity at :3305
    # on a conductive row), so the pair consumes a pre-injection D and the deposit has
    # to be repaired at the second consult. `carries_repair` is this family's own
    # declaration that its plan installs that bracket, and `repair_paths` says WHICH
    # repair — the plain one, because both halves refuse an active absorber, so the
    # recurrence this seam inverts is always update_E's overwrite.
    #
    # `pml` IS FORWARDED, unlike every predicate written before this one. It has to be:
    # deposit_repair.repairable refuses a plain-only declaration with no layer BY NAME,
    # since which recurrence ran cannot be established without one. What that buys is
    # that an active layer reaching here is refused at the seam clause as well as at
    # both halves, rather than only at both halves.
    #
    # IGNORANCE IS NEVER AN EMPTY SET: `Fields` does not hold the source list, so an
    # undeclared `sources` is itself a REFUSAL. A MAGNETIC source is injected in the
    # B/H half and does not reach this seam.
    reasons.extend(_deposit_repair.seam_source_reasons(
        fields, sources, 'D',
        undeclared=(
            "the source set was not declared: this predicate cannot infer an empty "
            "electric source seam from Fields"
        ),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) is electric: the driver "
            f"injects it BETWEEN step_D and update_E (driver.py:3294-3299), and on a "
            f"conductive row through _inject_electric_through_conductivity "
            f"(driver.py:3305)"),
        carries_repair=CARRIES_DEPOSIT_REPAIR,
        pml=pml,
        repair_paths=REPAIR_PATHS))

    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False,
                        tuple(dict.fromkeys(reasons + ["fields carries no grid"])))

    # THE TWO FILLS. Both halves already refuse a mirror plane; this restates by NAME
    # what that refusal buys on THIS seam — a reader should not have to chase a shared
    # grid clause to learn that two driver passes sit between the halves on a folded
    # grid. The deposit repair's own image closure would carry a deposit ACROSS a fold,
    # but the fills also rebuild rows this launch never wrote, which is a different
    # problem and one this family does not solve: MEASURED at 256 words per array of
    # divergence on a mirrored 16x16x10 grid WITH NO SOURCE AT ALL
    # (parity/meep_gpu/probe_plain_deposit_repair.py, leg `fold_control`), so the fold
    # refusal is about the fills and not about the deposit.
    if _call(grid, "has_symmetry", default=False):
        reasons.append(
            "a mirror plane is active: stepping.fill_symmetry_bc_D (driver.py:3309) "
            "and fill_folded_far_ghosts_D (:3311) both run inside this seam and "
            "neither is carried by this family")
    for axis in range(3):
        if _call(grid, "is_mirrored", axis, default=False):
            reasons.append(
                f"axis {axis} is folded: stepping.fill_symmetry_bc_D and "
                f"stepping.fill_folded_far_ghosts_D both run inside this seam "
                f"(driver.py:3309-3311) and neither is carried by this family")

    # THE WALL IS CARRIED, so what is refused here is an UNREADABLE wall rule rather
    # than a metallic axis. `zero_metal_axes` is the same function the plan compiles
    # ZM_X/ZM_Y/ZM_Z from, so the predicate and the constexpr cannot disagree; a grid
    # that cannot answer `has_metallic`/`is_metallic` answers False there, which would
    # compile the wall away on a run that needs it. Asking the SAME questions here and
    # refusing an unreadable answer is what closes that.
    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if not hasattr(grid, name):
            reasons.append(
                f"the grid does not expose {name}, so coverage.zero_metal_axes cannot "
                f"establish which axes stepping.zero_metal_D clears (stepping.py:2253) "
                f"and this launch carries that pass inline")
    kinds = _boundary_kinds(grid, None)
    if kinds is None:
        # UNREADABLE IS A REFUSAL, NEVER A CRASH. `_boundary_kinds` returns None for a
        # grid the array path will not resolve, and the plan indexes it for BCX/BCY/BCZ.
        reasons.append(
            "the per-axis boundary rule is unreadable, so the ghost rule this launch "
            "compiles (BCX/BCY/BCZ) cannot be established")

    # THE CONDUCTIVE WHOLE-VOLUME RESCALE, AND IT IS THE CLAUSE THAT PRICES THIS CELL.
    # See :func:`_scaled_conductive_injection_reasons` for the measurement.
    reasons.extend(_scaled_conductive_injection_reasons(fields, sources))

    # THE STENCIL, restated. The eighteen structurally unfusable D->E seam-instances on
    # this board are the OFF-DIAGONAL constitutive reading partner D volumes at four
    # indices (stepping.py:1235-1253) while step_D writes them in place. The E half
    # refuses that row already; it is named again because THIS is the fact that makes
    # the present cell buildable at all.
    if bool(getattr(fields, "has_offdiagonal_epsilon", False)):
        reasons.append(
            "an off-diagonal chi1inv row makes update_E a STENCIL over the D volumes "
            "step_D writes in place (stepping.py:1235-1253), and no grid-wide barrier "
            "exists inside one launch")

    # THE POLE COUNTS MUST FIT the kernel's compiled slots. The E half already refuses
    # more than MAX_POLES; evaluated here as well so an over-ceiling configuration is a
    # refusal BY NAME rather than a ValueError at plan time, and so the from-arrays
    # route's ceiling and this one cannot drift apart.
    try:
        order = poles_per_component(fields)
    except Exception as exc:  # noqa: BLE001 - unreadable is a refusal
        return Coverage(False, tuple(dict.fromkeys(
            reasons + [f"the pole partition is unreadable ({exc!r})"])))
    for component, _displacement in E_TERMS:
        count = len(tuple(order.get(component, ())))
        if count > MAX_POLES:
            reasons.append(
                f"{component} is driven by {count} poles, more than the kernel's "
                f"MAX_POLES={MAX_POLES} compiled slots")

    # THE INVERSE-EPSILON VOLUMES ARE READ BY THE KERNEL, so `Fields` must be able to
    # hand one over per E component. An absent accessor would be a TypeError inside the
    # builder rather than a refusal, which is the wrong way for an uncovered
    # configuration to fail.
    if getattr(fields, "inverse_epsilon_for", None) is None:
        reasons.append(
            "fields does not expose inverse_epsilon_for; the constitutive half cannot "
            "bind the inv_eps volumes it loads")

    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


def explain_no_pml_fused_electric_pair(fields: Any, pml: Any,
                                       sources: Any = None) -> Coverage:
    """The verdict with its reasons, for reports. Needs no Triton."""
    return no_pml_fused_electric_pair_coverage(fields, pml, sources)


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------


class NoPmlFusedElectricPairPlan:
    """One allocation-free launch for ``step_D``, ``zero_metal_D`` and ``update_E``.

    THE POLE POINTERS ARE RESOLVED PER LAUNCH, never cached, exactly as
    :class:`.no_pml_stored_e.StoredEConstitutivePlan` resolves them:
    ``PolarizationState.update`` rotates ``P``/``P_prev``/scratch on every step, so a
    pointer captured at plan time would be one timestep stale — and stale in a way that
    still computes, giving a smooth wrong field. What IS cached is pole ORDER, which is
    bit-load-bearing and which :class:`.no_pml_stored_e.LivePoleBinding` refuses to let
    change after planning.

    NO ``f_w`` SLOTS AND NO COEFFICIENT SLOTS, and the absence is structural rather than
    defaulted: a plan that carried them would let a caller bind an active layer's tables
    to a kernel that ignores them, which is a silent half-step.
    """

    __slots__ = ("shape", "n_elem", "dtdx", "backward", "derive", "bc", "zero_metal",
                 "cond", "counts", "block", "num_warps", "_targets", "_sources",
                 "_condfac", "_condinv", "_e_targets", "_inv_eps", "_poles", "_grid",
                 "_kernel", "_pointer")

    #: The driver call sites one launch performs. Declared, so a composition can be
    #: inspected rather than inferred from the curl slot's name.
    replaces = REPLACES

    def __init__(self, shape: Sequence[int], dtdx: float, codes: Sequence[int],
                 zero_metal: Sequence[bool], cond: Sequence[bool], block: int,
                 targets: Sequence[Any], sources: Sequence[Any],
                 condfac: Sequence[Any], condinv: Sequence[Any],
                 e_targets: Sequence[Any], inverse_epsilon: Sequence[Any],
                 poles: Any, kernel: Any = None,
                 num_warps: Optional[int] = 1) -> None:
        from .launch import CupyPointer, _flat  # noqa: PLC0415

        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        # float(): the array path consumes the Python scalar at float32 precision and
        # Triton types a Python float argument as fp32, so the two are the same bits.
        self.dtdx = float(dtdx)
        self.backward = BACKWARD
        self.derive = DERIVE
        self.bc = tuple(int(code) for code in codes)
        self.zero_metal = tuple(1 if flag else 0 for flag in zero_metal)
        self.cond = tuple(int(bool(flag)) for flag in cond)
        if len(self.bc) != 3 or len(self.zero_metal) != 3 or len(self.cond) != 3:
            raise ValueError(
                "this plan needs three boundary codes, three wall flags and three "
                "per-target conductivity flags")
        self._pointer = CupyPointer
        target_arrays = tuple(targets)
        self._targets = tuple(CupyPointer(a) for a in target_arrays)
        self._sources = tuple(CupyPointer(a) for a in sources)
        # A dead coefficient slot binds its target's own array. The pointer is correctly
        # typed and COND=0 compiles every load away — the certified conductive curl
        # plan's own convention, kept rather than re-invented.
        self._condfac = tuple(
            CupyPointer(_flat(array)) if array is not None
            else CupyPointer(target_arrays[index])
            for index, array in enumerate(condfac))
        self._condinv = tuple(
            CupyPointer(_flat(array)) if array is not None
            else CupyPointer(target_arrays[index])
            for index, array in enumerate(condinv))
        self._e_targets = tuple(CupyPointer(a) for a in e_targets)
        if inverse_epsilon is None:
            raise ValueError(
                "the fused pair loads inv_eps on every launch; a placeholder binding "
                "would be read as a coefficient")
        self._inv_eps = tuple(CupyPointer(a) for a in inverse_epsilon)
        self._poles = poles
        self.counts = tuple(int(value) for value in poles.counts)
        if len(self.counts) != len(E_TERMS):
            raise ValueError("a plan requires one pole count per E component")
        for (component, _displacement), count in zip(E_TERMS, self.counts):
            if count > MAX_POLES:
                raise ValueError(
                    f"{component} has {count} poles; MAX_POLES={MAX_POLES}")
        self.block = int(block)
        self.num_warps = None if num_warps is None else int(num_warps)
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        # An override for exactly one caller: the gate's mutation legs, which compile
        # deliberately broken copies of this kernel. Dropping it is not a slowdown, it
        # is a DISARMING — every mutation leg would then launch the shipped kernel and
        # report the defect as uncaught.
        self._kernel = kernel

    def run(self, guard: Optional[bool] = None) -> None:
        """Resolve current P pointers and launch the fused pair, in place."""
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        groups = self._poles.arrays()
        slots: List[Any] = []
        for index, group in enumerate(groups):
            if len(group) != self.counts[index]:
                raise RuntimeError(
                    f"component {index} resolved {len(group)} poles, not the planned "
                    f"{self.counts[index]}")
            slots.extend(self._pointer(array) for array in group)
            # The dead slots bind this component's D pointer: correctly typed, and NP
            # compiles every load away. StoredEConstitutivePlan's own convention.
            slots.extend([self._targets[index]] * (MAX_POLES - len(group)))
        kernel = (self._kernel if self._kernel is not None
                  else no_pml_fused_curl_constitutive_D_kernel())
        nx, ny, nz = self.shape
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        kernel[self._grid](
            *self._targets, *self._sources, *self._inv_eps,
            *self._condfac, *self._condinv, *self._e_targets, *slots,
            nx, ny, nz, self.n_elem, self.dtdx,
            BACKWARD=self.backward,
            DERIVE=self.derive,
            BCX=self.bc[0], BCY=self.bc[1], BCZ=self.bc[2],
            ZM_X=self.zero_metal[0], ZM_Y=self.zero_metal[1], ZM_Z=self.zero_metal[2],
            COND0=self.cond[0], COND1=self.cond[1], COND2=self.cond[2],
            NP0=self.counts[0], NP1=self.counts[1], NP2=self.counts[2],
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )

    def __repr__(self) -> str:
        return (f"NoPmlFusedElectricPairPlan(shape={self.shape}, bc={self.bc}, "
                f"zero_metal={self.zero_metal}, cond={self.cond}, "
                f"poles={self.counts}, block={self.block}, "
                f"num_warps={self.num_warps})")


def plan_no_pml_fused_electric_pair(
        fields: Any, pml: Any, sources: Any = None, block: Optional[int] = None,
        num_warps: Optional[int] = 1, kernel: Any = None,
        ) -> Optional[NoPmlFusedElectricPairPlan]:
    """Build the plan from the engine's own objects, or ``None`` when refused.

    ``None`` is the only refusal: a configuration this kernel does not carry must fall
    back to the array path, never raise into a caller that would otherwise have stepped
    correctly.
    """
    if not no_pml_fused_electric_pair_coverage(fields, pml, sources).covered:
        return None
    from .kernels import DEFAULT_BLOCK  # noqa: PLC0415

    grid = fields.grid
    # `None` for the PML argument, not `pml`: this family's own predicate has already
    # refused an active absorber, and `_boundary_kinds` consulted WITH one would report
    # the absorber's softening rather than the grid's declaration.
    kinds = _boundary_kinds(grid, None)
    flags = conductive_no_pml_targets(fields, CURL_SUB_STEP)
    order = poles_per_component(fields)
    return NoPmlFusedElectricPairPlan(
        grid.shape, grid.dt / grid.dx,
        [1 if kind == "metallic" else 0 for kind in kinds],
        zero_metal_axes(grid),
        flags,
        DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in CURL_TARGETS],
        [getattr(fields, name) for name in CURL_SOURCES],
        [fields.condfac_for(name) if flags[index] else None
         for index, name in enumerate(CURL_TARGETS)],
        [fields.condinv_for(name) if flags[index] else None
         for index, name in enumerate(CURL_TARGETS)],
        [getattr(fields, component) for component, _ in E_TERMS],
        [fields.inverse_epsilon_for(component) for component, _ in E_TERMS],
        LivePoleBinding(fields, order),
        kernel=kernel, num_warps=num_warps,
    )


def plan_no_pml_fused_electric_pair_from_arrays(
        arrays: Dict[str, Any], poles: Dict[str, Sequence[Any]],
        dtdx: float, codes: Sequence[int], zero_metal: Sequence[bool],
        cond: Sequence[bool] = (True, True, True),
        block: Optional[int] = None, kernel: Any = None,
        num_warps: Optional[int] = 1) -> NoPmlFusedElectricPairPlan:
    """Build from bare device arrays — the gate's route.

    No coverage predicate runs: the caller is a harness that constructed the
    configuration deliberately, including the deliberately wrong ones. ``arrays`` is
    keyed by component name plus ``condfac_D*``/``condinv_D*`` and ``inv_eps_Ex``...;
    ``poles`` is keyed by E component and holds the P volumes in registration order,
    and ``kernel=`` carries the mutation override — dropping it silently disarms every
    mutation leg.
    """
    from .kernels import DEFAULT_BLOCK  # noqa: PLC0415

    flags = tuple(bool(flag) for flag in cond)
    shape = tuple(int(n) for n in arrays[CURL_TARGETS[0]].shape)
    return NoPmlFusedElectricPairPlan(
        shape, dtdx, codes, zero_metal, flags,
        DEFAULT_BLOCK if block is None else block,
        [arrays[name] for name in CURL_TARGETS],
        [arrays[name] for name in CURL_SOURCES],
        [arrays[f"condfac_{name}"] if flags[index] else None
         for index, name in enumerate(CURL_TARGETS)],
        [arrays[f"condinv_{name}"] if flags[index] else None
         for index, name in enumerate(CURL_TARGETS)],
        [arrays[component] for component, _ in E_TERMS],
        [arrays["inv_eps_" + component] for component, _ in E_TERMS],
        StaticPoleBinding([poles[component] for component, _ in E_TERMS]),
        kernel=kernel, num_warps=num_warps,
    )
