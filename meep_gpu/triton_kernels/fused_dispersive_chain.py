"""One launch for the whole dispersive electric chain: D curl, E constitutive, ADE.

``dispersive_fused_pair`` closes ``step_D`` into ``update_E`` and stops at the
sub-step boundary; ``fused_ade_state`` closes one susceptibility's per-component
``update_P`` launches into one.  The two products are still TWO launches, and the
dependency between them — ``update_P`` consumes exactly the ``f_w_E`` that
``update_E`` wrote one launch earlier — is the seam this module closes.

What the fusion actually eliminates, per driven component and per cell:

* the store/load round trip on ``f_w_E``: the drive ``W`` is already in a register
  when the ADE recurrence needs it.  The ``f_w_E`` STORE REMAINS — the array is a
  live engine output that DFT monitors, the next step's PML history term and
  ``Fields.drive_field`` all read — so this is a load eliminated, not a buffer;
* the reload of ``P`` for the recurrence: the pole-aware ``D - P`` subtraction in
  the constitutive half has already loaded exactly that volume.

Neither elimination changes a rounded value: a float32 store followed by a float32
load of the same address is the identity, and the register reused is the same
register the eliminated load would have produced.  That is an argument for why
byte identity is EXPECTED, not evidence that it holds — the measured record
(``special_kz`` 24/96, ``nonlinear`` 0/108, ``offdiag`` 0/28, ``bfast`` 0/52
identical under fusion) says identity is a per-product measurement, and this
product's own device gate is what decides it.

SCOPE, and why each bound is here rather than in a follow-up:

* **exactly one susceptibility**, at most one pole per component.  Multi-pole ADE
  fusion has no measured argument behind it: the sub-step-share probe put
  ``update_P`` at 51.9 % of the step at K=6 with a 9.5-28.3 pp within-run spread
  and a non-monotonic size dependence, so the precondition that would justify it
  is bracketed, not cleared.  A K>1 configuration is REFUSED BY NAME here;
* **an active PML**, inherited from the constitutive half — without one
  ``update_E`` is a different sub-step (stepping.py:1022) and, decisively for this
  product, ``Fields.drive_field`` (fields.py:1160-1162) returns the stored ``E``
  rather than ``f_w``, so the register this kernel hands the ADE would be the
  wrong field;
* **no electric source**, inherited from the pair: the driver injects electric
  sources between ``step_D`` and ``update_E``, inside this seam.

THE ALIASING DISCIPLINE, which is a correctness argument and not a style note.
``PolarizationState.update`` rotates one shared scratch buffer through the
components (dispersion.py:686-691), so within ONE launch the destination of the
Ey arm IS the ``P_prev`` volume the Ex arm reads, and the Ez arm's destination is
Ey's ``P_prev``.  Rather than rely on the compiler preserving a store-after-load
order on unannotated (never ``noalias``) pointer arguments, this kernel HOISTS
EVERY ADE HISTORY LOAD ABOVE EVERY ADE STORE.  With all three ``P_prev`` reads
retired before the first destination write, the arms are order-independent and the
rotation cannot be observed by a later arm however the scheduler arranges them.

Composition into ``launch.plan_step`` is deferred, exactly as each family tranche
was: nothing imports this module at engine startup and dispatch stays disabled.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import coverage as _coverage
from .. import deposit_repair as _deposit_repair
from .coverage import ELECTRIC_COMPONENTS, Coverage, sigma_is_volume
from .dispersive_fused_pair import dispersive_fused_pair_coverage
from .dispersive_update_e import (
    E_TERMS,
    MAX_POLES,
    LivePoleBinding,
    poles_per_component,
)
from .fused_ade_state import fused_ade_state_coverage

try:  # pragma: no cover - CUDA host only
    import triton
    import triton.language as tl
    _TRITON_IMPORT_ERROR: Optional[BaseException] = None
except ImportError as _exc:  # pragma: no cover - laptop path
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_IMPORT_ERROR = _exc


#: This product carries at most one pole per component by construction (one
#: susceptibility), so the constitutive half's eight compiled slots are declared
#: but only slot 0 can ever be live.  The bound is enforced host-side and refused
#: by name rather than truncated.
CHAIN_MAX_POLES = 1

#: Does this product bracket its fused launch with the deposit repair? NO. The repair is
#: a TWO-consult protocol -- launch at the first slot, repair at the second -- and this
#: chain owns three (``step_D``, ``update_E``, ``update_P``); the repair reconstructs E
#: and ``f_w_E`` at the deposit points and never re-advances P, which this product has
#: already stepped from the drive its pre-injection launch wrote. Nothing installs it
#: either: ``launch._install_fused_pair`` is the only builder of the two repair plans and
#: it never sees this product. The pair half it inherits DOES declare the repair, which
#: is exactly why the seam clause is asked again in this module rather than inherited.
CARRIES_DEPOSIT_REPAIR = False

#: The settled launch configuration.  ONE WARP is the shipped policy for every
#: cross-sub-step fused product (measured: 1.0739x median over 12 shape/boundary
#: cases against an independently tuned four-kernel control, the 7.3 % regression
#: having been an artifact of Triton's default warp selection).  ``enable_fp_fusion``
#: is off engine-wide (``kernels.ENABLE_FP_FUSION``) and is what keeps the new seam
#: from contracting a multiply and an add into an FMA.  KEEP.
POLICY = {
    "num_warps": 1,
    "block": "kernels.DEFAULT_BLOCK",
    "enable_fp_fusion": "kernels.ENABLE_FP_FUSION (False)",
    "status": "keep",
}


if triton is not None:
    METALLIC = tl.constexpr(1)

    @triton.jit
    def fused_curl_dispersive_E_ade(
        f0, f1, f2,
        u0, u1, u2,
        g0, g1, g2,
        kmx, sinvx, kmy, sinvy, kmz, sinvz,
        e0, e1, e2,
        w0, w1, w2,
        ie0, ie1, ie2,
        a0, a1, a2, a3, a4, a5, a6, a7,
        b0, b1, b2, b3, b4, b5, b6, b7,
        c0, c1, c2, c3, c4, c5, c6, c7,
        kp0, km0, kp1, km1, kp2, km2,
        o0, q0p, sg0,
        o1, q1p, sg1,
        o2, q2p, sg2,
        c_now, c_prev, c_drive,
        nx, ny, nz, n_elem, dtdx,
        NP0: tl.constexpr, NP1: tl.constexpr, NP2: tl.constexpr,
        BACKWARD: tl.constexpr,
        BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
        ZM_X: tl.constexpr, ZM_Y: tl.constexpr, ZM_Z: tl.constexpr,
        ADE0: tl.constexpr, ADE1: tl.constexpr, ADE2: tl.constexpr,
        SIGMA_IS_VOLUME0: tl.constexpr,
        SIGMA_IS_VOLUME1: tl.constexpr,
        SIGMA_IS_VOLUME2: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """D curl, pole-aware E update and the ADE recurrence in one grid."""
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = idx < n_elem
        nyz = ny * nz
        k = idx % nz
        plane = idx // nz
        j = plane % ny
        i = plane // ny

        # D curl: verbatim from fused_curl_dispersive_E.
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

        p0 = tl.load(u0 + idx, mask=live, other=0.0)
        n0 = ((p0 * km_y) - curl0) * si_y
        v0 = (((tl.load(f0 + idx, mask=live, other=0.0) * km_z) + n0) - p0) * si_z

        p1 = tl.load(u1 + idx, mask=live, other=0.0)
        n1 = ((p1 * km_z) - curl1) * si_z
        v1 = (((tl.load(f1 + idx, mask=live, other=0.0) * km_x) + n1) - p1) * si_x

        p2 = tl.load(u2 + idx, mask=live, other=0.0)
        n2 = ((p2 * km_x) - curl2) * si_x
        v2 = (((tl.load(f2 + idx, mask=live, other=0.0) * km_y) + n2) - p2) * si_y

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

        # Pole-aware E update.  The pole-0 load is NAMED so the ADE half can reuse
        # the register instead of re-reading the same volume; every other slot is
        # spelled exactly as the separately gated pair spells it.
        kp_0 = tl.load(kp0 + i, mask=live, other=0.0)
        km_0 = tl.load(km0 + i, mask=live, other=0.0)
        kp_1 = tl.load(kp1 + j, mask=live, other=0.0)
        km_1 = tl.load(km1 + j, mask=live, other=0.0)
        kp_2 = tl.load(kp2 + k, mask=live, other=0.0)
        km_2 = tl.load(km2 + k, mask=live, other=0.0)

        prev0 = tl.load(w0 + idx, mask=live, other=0.0)
        s0 = v0
        if NP0 > 0:
            pa0 = tl.load(a0 + idx, mask=live, other=0.0)
            s0 = s0 - pa0
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
        src0 = s0 * tl.load(ie0 + idx, mask=live, other=0.0)
        tl.store(w0 + idx, src0, mask=live)
        a0v = tl.load(e0 + idx, mask=live, other=0.0)
        a0v = a0v + kp_0 * src0
        a0v = a0v - km_0 * prev0
        tl.store(e0 + idx, a0v, mask=live)

        prev1 = tl.load(w1 + idx, mask=live, other=0.0)
        s1 = v1
        if NP1 > 0:
            pa1 = tl.load(b0 + idx, mask=live, other=0.0)
            s1 = s1 - pa1
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
        src1 = s1 * tl.load(ie1 + idx, mask=live, other=0.0)
        tl.store(w1 + idx, src1, mask=live)
        a1v = tl.load(e1 + idx, mask=live, other=0.0)
        a1v = a1v + kp_1 * src1
        a1v = a1v - km_1 * prev1
        tl.store(e1 + idx, a1v, mask=live)

        prev2 = tl.load(w2 + idx, mask=live, other=0.0)
        s2 = v2
        if NP2 > 0:
            pa2 = tl.load(c0 + idx, mask=live, other=0.0)
            s2 = s2 - pa2
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
        src2 = s2 * tl.load(ie2 + idx, mask=live, other=0.0)
        tl.store(w2 + idx, src2, mask=live)
        a2v = tl.load(e2 + idx, mask=live, other=0.0)
        a2v = a2v + kp_2 * src2
        a2v = a2v - km_2 * prev2
        tl.store(e2 + idx, a2v, mask=live)

        # ADE recurrence, spelled exactly as fused_ade_state spells it, with the
        # drive taken from the register that was just stored to f_w and the P
        # history taken from the register the subtraction already loaded.
        #
        # EVERY HISTORY LOAD FIRST.  o1 IS the volume q0p names and o2 IS q1p's,
        # so a store placed between two of these loads would make the result depend
        # on the scheduler's freedom over aliasing pointers.
        if ADE0:
            h0 = tl.load(q0p + idx, mask=live, other=0.0)
            if SIGMA_IS_VOLUME0:
                sv0 = tl.load(sg0 + idx, mask=live, other=0.0)
            else:
                sv0 = sg0
        if ADE1:
            h1 = tl.load(q1p + idx, mask=live, other=0.0)
            if SIGMA_IS_VOLUME1:
                sv1 = tl.load(sg1 + idx, mask=live, other=0.0)
            else:
                sv1 = sg1
        if ADE2:
            h2 = tl.load(q2p + idx, mask=live, other=0.0)
            if SIGMA_IS_VOLUME2:
                sv2 = tl.load(sg2 + idx, mask=live, other=0.0)
            else:
                sv2 = sg2

        if ADE0:
            tl.store(o0 + idx,
                     ((pa0 * c_now) + (c_prev * h0)) +
                     (c_drive * (sv0 * src0)), mask=live)
        if ADE1:
            tl.store(o1 + idx,
                     ((pa1 * c_now) + (c_prev * h1)) +
                     (c_drive * (sv1 * src1)), mask=live)
        if ADE2:
            tl.store(o2 + idx,
                     ((pa2 * c_now) + (c_prev * h2)) +
                     (c_drive * (sv2 * src2)), mask=live)

else:  # pragma: no cover - laptop path
    fused_curl_dispersive_E_ade = None  # type: ignore[assignment]


def fused_curl_dispersive_E_ade_kernel() -> Any:
    if fused_curl_dispersive_E_ade is None:
        raise ImportError(
            "the fused dispersive D/E/ADE chain kernel needs the optional `triton` "
            f"package (pip install triton). Original error: {_TRITON_IMPORT_ERROR}")
    return fused_curl_dispersive_E_ade


def _single_state(fields: Any) -> Tuple[Optional[Any], List[str]]:
    """The one susceptibility this product carries, or the reason there isn't one."""
    try:
        states = tuple(getattr(fields, "polarizations", ()) or ())
    except Exception as exc:  # noqa: BLE001 - an unreadable list is not an empty one
        return None, [f"the polarization list could not be read ({exc!r})"]
    if len(states) != 1:
        return None, [
            f"{len(states)} susceptibilities are registered: this product fuses the "
            f"ADE of exactly one, and multi-pole ADE fusion has no measured argument "
            f"behind it (the sub-step-share probe left the precondition bracketed, "
            f"not cleared)"]
    return states[0], []


def fused_dispersive_chain_coverage(fields: Any, pml: Any,
                                    sources: Any = None) -> Coverage:
    """May one launch span step_D, update_E and update_P for this configuration?

    Every clause of both halves is asked through the halves' own predicates — this
    product introduces no numerical rule of its own, so it must not introduce a
    second opinion about theirs either.  What IS new is the seam:

    * the drive the ADE consumes must be the ``f_w`` this kernel writes, which is
      what ``Fields.drive_field`` returns only while the PML is active;
    * the P volume the recurrence advances must be the SAME array the constitutive
      half subtracts, so the reused register is the pole it claims to be;
    * one susceptibility, at most one pole per component.
    """
    reasons: List[str] = []

    pair = dispersive_fused_pair_coverage(fields, pml, sources)
    if not pair.covered:
        reasons.extend(f"pair half: {reason}" for reason in pair.reasons)

    # THE SOURCE SEAM IS ASKED HERE, NOT INHERITED. The pair half now declares
    # CARRIES_DEPOSIT_REPAIR, so what it returns for an in-seam electric source is an
    # ADMISSION -- and this chain may not inherit it. The deposit repair is a
    # TWO-CONSULT protocol: the launch runs at the first slot and the repair spends the
    # second. This product owns THREE slots (step_D, update_E, update_P), and the
    # repair reconstructs E and f_w_E only -- it never re-advances P, which this chain
    # has already stepped from the drive the pre-injection launch wrote. Nor does
    # anything install it: ``launch._install_fused_pair`` is the only builder of the two
    # repair plans and it never sees this product. So the seam stays refused, with this
    # module's own declaration, which is False.
    reasons.extend(f"pair half: {reason}" for reason in
                   _deposit_repair.seam_source_reasons(
                       fields, sources, "D",
                       undeclared=(
                           "the source set was not declared: this predicate cannot "
                           "infer an empty electric source seam from Fields"),
                       refusal=lambda index, source: (
                           f"source {index} ({type(source).__name__}) is electric: the "
                           "driver injects it BETWEEN step_D and update_E"),
                       carries_repair=CARRIES_DEPOSIT_REPAIR))

    state, state_reasons = _single_state(fields)
    reasons.extend(state_reasons)

    if state is not None:
        ade = fused_ade_state_coverage(fields, state)
        if not ade.covered:
            reasons.extend(f"ADE half: {reason}" for reason in ade.reasons)

        order = poles_per_component(fields)
        for component, _, _ in E_TERMS:
            count = len(order.get(component, ()))
            if count > CHAIN_MAX_POLES:
                reasons.append(
                    f"{component} is driven by {count} poles; this product compiles "
                    f"{CHAIN_MAX_POLES} live slot per component")
            driven = bool(_coverage._call(state, "drives", component, default=False))
            if driven != (count == 1):
                reasons.append(
                    f"{component}: the susceptibility reports drives={driven} while "
                    f"the pole order carries {count} states; the ADE arm and the "
                    f"subtraction slot must be the same pole")
            if not driven:
                continue
            pole = (getattr(state, "P", {}) or {}).get(component)
            bound = order[component][0] if order.get(component) else None
            if pole is None or bound is None or (getattr(bound, "P", {}) or {}).get(
                    component) is not pole:
                reasons.append(
                    f"{component}: the subtracted pole volume is not the state's own "
                    f"P; the register the ADE reuses would be a different array")

    # The seam's own condition, named rather than inherited: without an active PML
    # drive_field hands back the stored E (fields.py:1160-1162) and the register
    # this kernel passes the recurrence would be the wrong field.
    if not getattr(fields, "_pml_active", False):
        reasons.append(
            "the PML is not active: Fields.drive_field then returns the stored E "
            "rather than f_w, and the ADE drive this kernel keeps in a register is "
            "the constitutive product, not that field")

    return Coverage(not reasons, tuple(reasons))


class FusedDispersiveChainPlan:
    """One allocation-free launch for the whole electric chain of one state.

    The host-side buffer rotation is performed only AFTER a successful launch, the
    same ordering :class:`fused_ade_state.FusedAdeStatePlan` uses and for the same
    reason: a plan that rotated first and then raised would leave the engine
    holding a P history that never happened.
    """

    __slots__ = (
        "shape", "n_elem", "dtdx", "bc", "zero_metal", "block", "num_warps",
        "counts", "state", "components", "_targets", "_aux", "_sources",
        "_curl_coefficients", "_e_targets", "_e_aux", "_inverse_epsilon",
        "_e_coefficients", "_poles", "_kernel", "_pointer", "_sigma_is_volume",
    )

    def __init__(self, shape, dtdx: float, bc, zero_metal, block: int,
                 targets, auxiliaries, sources, curl_coefficients,
                 e_targets, e_aux, inverse_epsilon, e_coefficients, poles,
                 state: Any, components: Sequence[str],
                 kernel: Any = None, num_warps: Optional[int] = 1,
                 pointer: Any = None) -> None:
        from .launch import CupyPointer, _flat  # noqa: PLC0415

        if pointer is None:
            pointer = CupyPointer
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.dtdx = float(dtdx)
        self.bc = tuple(int(value) for value in bc)
        self.zero_metal = tuple(bool(value) for value in zero_metal)
        self.block = int(block)
        self.num_warps = num_warps
        self._pointer = pointer
        self._targets = tuple(pointer(array) for array in targets)
        self._aux = tuple(pointer(array) for array in auxiliaries)
        self._sources = tuple(pointer(array) for array in sources)
        self._curl_coefficients = tuple(
            pointer(_flat(array)) for array in curl_coefficients)
        self._e_targets = tuple(pointer(array) for array in e_targets)
        self._e_aux = tuple(pointer(array) for array in e_aux)
        self._inverse_epsilon = tuple(pointer(array) for array in inverse_epsilon)
        self._e_coefficients = tuple(
            pointer(_flat(array)) for array in e_coefficients)
        self._poles = poles
        self.counts = tuple(int(value) for value in poles.counts)
        if any(value > CHAIN_MAX_POLES for value in self.counts):
            raise ValueError(
                f"pole counts {self.counts} exceed CHAIN_MAX_POLES={CHAIN_MAX_POLES}")
        self.state = state
        self.components = tuple(components)
        for index, (component, _, _) in enumerate(E_TERMS):
            live = component in self.components
            if live != (self.counts[index] == 1):
                raise ValueError(
                    f"{component}: ADE arm live={live} disagrees with pole count "
                    f"{self.counts[index]}; the arm reuses the subtraction's register "
                    f"and the two must name one pole")
        self._sigma_is_volume = {
            component: bool(sigma_is_volume(state, component))
            for component in self.components
        }
        self._kernel = kernel

    @property
    def grid(self) -> Tuple[int, ...]:
        return ((self.n_elem + self.block - 1) // self.block,)

    def _entries(self) -> Dict[str, Tuple[Any, Any, Any, Any]]:
        """The destination chain, resolved before launch exactly as the array path does."""
        state = self.state
        scratch = state._scratch
        entries: Dict[str, Tuple[Any, Any, Any, Any]] = {}
        for component in self.components:
            p = state.P[component]
            p_prev = state.P_prev[component]
            entries[component] = (scratch, p, p_prev, state.sigma[component])
            scratch = p_prev
        entries["__tail__"] = (scratch, None, None, None)
        return entries

    def run(self, guard: Optional[bool] = None) -> None:
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        kernel = (self._kernel if self._kernel is not None
                  else fused_curl_dispersive_E_ade_kernel())
        pointer = self._pointer
        groups = self._poles.arrays()
        slots: List[Any] = []
        for index, group in enumerate(groups):
            if len(group) != self.counts[index]:
                raise RuntimeError(
                    f"component {index} resolved {len(group)} poles, not "
                    f"the compiled count {self.counts[index]}")
            slots.extend(pointer(array) for array in group)
            slots.extend([self._targets[index]] * (MAX_POLES - len(group)))

        entries = self._entries()
        dummy = self.state._scratch
        ade_arguments: List[Any] = []
        ade_flags: List[int] = []
        volume_flags: List[int] = []
        for index, (component, _, _) in enumerate(E_TERMS):
            entry = entries.get(component)
            if entry is None:
                ade_arguments.extend((pointer(dummy), pointer(dummy), 0.0))
                ade_flags.append(0)
                volume_flags.append(0)
                continue
            output, pole, p_prev, sigma = entry
            if pole is not groups[index][0]:
                raise RuntimeError(
                    f"{component}: the ADE arm's pole is not the array the "
                    f"subtraction slot resolved; the reused register would be a "
                    f"different volume")
            volume = self._sigma_is_volume[component]
            ade_arguments.extend((
                pointer(output), pointer(p_prev),
                pointer(sigma) if volume else float(sigma),
            ))
            ade_flags.append(1)
            volume_flags.append(1 if volume else 0)

        c_now, c_prev, c_drive = self.state._coefficients
        nx, ny, nz = self.shape
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        kernel[self.grid](
            *self._targets, *self._aux, *self._sources, *self._curl_coefficients,
            *self._e_targets, *self._e_aux, *self._inverse_epsilon,
            *slots, *self._e_coefficients, *ade_arguments,
            float(c_now), float(c_prev), float(c_drive),
            nx, ny, nz, self.n_elem, self.dtdx,
            NP0=self.counts[0], NP1=self.counts[1], NP2=self.counts[2],
            BACKWARD=1,
            BCX=self.bc[0], BCY=self.bc[1], BCZ=self.bc[2],
            ZM_X=self.zero_metal[0], ZM_Y=self.zero_metal[1],
            ZM_Z=self.zero_metal[2],
            ADE0=ade_flags[0], ADE1=ade_flags[1], ADE2=ade_flags[2],
            SIGMA_IS_VOLUME0=volume_flags[0],
            SIGMA_IS_VOLUME1=volume_flags[1],
            SIGMA_IS_VOLUME2=volume_flags[2], BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )

        state = self.state
        for component in self.components:
            output, pole, _p_prev, _sigma = entries[component]
            state.P[component] = output
            state.P_prev[component] = pole
        state._scratch = entries["__tail__"][0]

    def __repr__(self) -> str:
        return (f"FusedDispersiveChainPlan(shape={self.shape}, "
                f"components={self.components}, poles={self.counts}, "
                f"bc={self.bc}, block={self.block}, num_warps={self.num_warps})")


def plan_fused_dispersive_chain(fields: Any, pml: Any, sources: Any = None,
                                block: Optional[int] = None,
                                num_warps: Optional[int] = 1,
                                kernel: Any = None,
                                ) -> Optional[FusedDispersiveChainPlan]:
    """Build the whole-chain plan for the live engine, or ``None`` when refused."""
    if not fused_dispersive_chain_coverage(fields, pml, sources).covered:
        return None
    from ..stepping import _boundary_kinds  # noqa: PLC0415
    from .kernels import DEFAULT_BLOCK  # noqa: PLC0415

    grid = fields.grid
    state = tuple(fields.polarizations)[0]
    order = poles_per_component(fields)
    return FusedDispersiveChainPlan(
        grid.shape, grid.dt / grid.dx,
        [1 if kind == "metallic" else 0 for kind in _boundary_kinds(grid, pml)],
        _coverage.zero_metal_axes(grid),
        DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in ("Dx", "Dy", "Dz")],
        [getattr(fields, name) for name in ("fu_Dx", "fu_Dy", "fu_Dz")],
        [getattr(fields, name) for name in ("Hx", "Hy", "Hz")],
        [getattr(pml, f"{stem}_{axis}")
         for axis in "xyz" for stem in ("kms", "sinv")],
        [getattr(fields, name) for name in ("Ex", "Ey", "Ez")],
        [getattr(fields, name) for name in ("f_w_Ex", "f_w_Ey", "f_w_Ez")],
        [fields.inverse_epsilon_for(name) for name in ("Ex", "Ey", "Ez")],
        [getattr(pml, f"{stem}_{axis}_h")
         for axis in "xyz" for stem in ("kps", "kms")],
        LivePoleBinding(fields, order), state, tuple(state.driven()),
        kernel=kernel, num_warps=num_warps,
    )
