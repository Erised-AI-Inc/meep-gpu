"""The NONLINEAR electric constitutive sub-step — ``update_E`` under chi2/chi3.

One kernel, one coverage predicate, one plan, in one file, on the pattern
:mod:`dispersive_update_e` set — because this family's whole difference from the
certified constitutive kernel is a pointwise Pade scalar and one four-point
transverse average, and none of it belongs in the shared modules yet.

NOT WIRED. Production dispatch is untouched: ``plan_fast_path`` still returns
``None`` on every branch and nothing here changes that, ``plan_step`` composition
is DEFERRED (``coverage._grid_reasons`` clause 10 — coverage.py:195-196 — already
refuses every chi2/chi3 run at every sub-step, so this predicate cannot create an
admitted-overlap ambiguity until a later coordinated change narrows that clause),
``coverage.py`` itself is another session's file and is imported, never edited,
and ``fingerprints.json`` carries no entry for this module — the byte gate binds
its own provenance record inside its results directory. Callers are the gate
(``parity/meep_gpu/gate_triton_nonlinear.py``), the composition probe
(``probe_triton_nonlinear_composition.py``) and the laptop tests, nothing else.

WHAT IT REPLACES. ``stepping.update_E`` (stepping.py:954) with an active PML, an
instantaneous chi2/chi3 installed (``fields.has_nonlinearity``, stepping.py:989;
fields.py:966-967), diagonal epsilon and NO registered susceptibility. The
nonlinearity enters HERE and only here (stepping.py:975-979): the curls,
``update_H`` and ``update_P`` are untouched arithmetic, which is why the current
package-wide every-sub-step refusal is broader than the recurrence requires —
the composition probe's partial-nonlinearity case is the MEASUREMENT that
licenses narrowing it, the way plan §13.1 narrowed the curl's dispersion clause.

Per component ``c`` with own axis ``a`` the array path runs, with no poles
registered (``displacement_minus_polarization_volumes`` then aliases each
component to its D array, fields.py:1107-1138)::

    gs   = D_c                                            # aliased volume
    us   = fields.inverse_epsilon_for(c)
    g1s  = (g1 + shift_down(g1, p1)) + shift_up(g1 + shift_down(g1, p1), a)
    g2s  = likewise for the second partner                # stepping.py:1150-1193
    Dsqr = gs*gs + 0.0625*(g1s*g1s + g2s*g2s)             # stepping.py:1131-1147
    c2   = gs   * chi2 * (us*us)                          # stepping.py:1054
    c3   = Dsqr * chi3 * (us*us*us)                       # stepping.py:1055
    u    = (1 + c2 + 2*c3) / (1 + 2*c2 + 3*c3)            # stepping.py:1056
    src  = (gs * us) * u                                  # stepping.py:1107-1109
    prev = f_w_c ; f_w_c = src                            # stepping.py:2112-2143
    E_c += kps_a_h * src ;  E_c -= kms_a_h * prev         # half-integer, :1015

THE THINGS THAT DECIDE BIT-IDENTITY, each held by the gate rather than assumed:

1. **THE FOUR-CORNER ASSOCIATION IS THE SHIFTED-PAIR ONE, NOT MEEP C'S.**
   ``stepping._nonlinear_transverse_sums`` (:1158-1162) forms
   ``(g1[i] + g1[i-s1]) + (g1[i+s] + g1[i+s-s1])`` — pair first, then the pair
   shifted — while MEEP C sums left to right (step_generic.cpp:646-648). The
   byte arbiter is stepping.py, so the pair association is what the kernel
   carries, and the gate's left-to-right mutation is what proves the difference
   is byte-visible rather than argued.
2. **THE TWO SHIFTS GO IN OPPOSITE DIRECTIONS** — half a cell DOWN the
   partner's axis, half a cell UP the component's own axis (stepping.py:
   1160-1171). Taking both the same way is the half-cell registration error
   that survives every scalar test; the gate carries a mutation for exactly it.
3. **SCALAR COEFFICIENT POWERS ARE PYTHON-DOUBLE POWERS, ROUNDED ONCE.** With a
   PYTHON-FLOAT ``us``, NumPy evaluates ``(chi1inv * chi1inv)`` in double and
   rounds ONCE when the product meets the float32 array; ``f32(us)*f32(us)`` in
   float32 is a different word for most values. The plan therefore ships THREE
   host-rounded scalars per scalar-epsilon component — ``f32(us)``,
   ``f32(us*us)``, ``f32((us*us)*us)`` (:func:`scalar_inverse_epsilon_arm`) —
   and never rebuilds the powers on the device. A VOLUME ``us`` takes the
   in-kernel float32 powers, exactly as NumPy's elementwise arm does. Scalars
   ride the fp32 kernel ABI (the measured platform fact: a ``tl.float64``
   annotation on a scalar is DEAD, re-rounded to fp32).
4. **THE PADE QUOTIENT'S GROUPINGS ARE THE TRANSCRIBED ONES**:
   ``(gs*chi2)*(us*us)``, ``(Dsqr*chi3)*((us*us)*us)``,
   ``((1 + c2) + 2*c3) / ((1 + 2*c2) + 3*c3)``, then ``(gs*us)*u`` — the
   explicit power groupings and the row-then-scale order of stepping.py:
   1054-1056 and :1107-1109. The division is spelled ``tl.math.div_rn``, NOT
   the plain ``/`` — a MEASURED platform fact, not style: Triton 3.1.0's
   ``semantic.truediv`` emits a bare LLVM ``fdiv``, which the NVPTX backend
   lowers to ``div.full.f32`` (~2 ulp), and job 2336 measured every nonlinear
   sweep case divergent while the linear arm stayed byte-identical;
   ``div_rn`` compiles to ``div.rn.ftz.f32`` and matched IEEE round-to-nearest
   division on 3 x 2^20 operand-class vectors, 0 differing words (divprobe job
   2338). The ``.ftz`` suffix is inert here: both quotient operands are sums
   anchored at 1.0 of f32-grid addends (smallest nonzero magnitude ~2^-25)
   and u is bounded by the pole guard's admitted domain, so no subnormal can
   reach the divide. No certified kernel in this package divides; the gate's
   NEAR-POLE amplitude class (u far from 1) is the leg that makes any
   divergence in the quotient byte-visible in f32 — job 2329's
   assertion-layer lesson applied to division.
5. **A LINEAR COMPONENT IN A PARTLY NONLINEAR RUN COMPILES TO THE CERTIFIED
   PLAIN BODY.** ``NL0``/``NL1``/``NL2`` are ``tl.constexpr``; the 0 arm is
   verbatim :func:`kernels.constitutive_step`'s SCALE=1 component body (MEEP's
   ``else if (u)`` branch, stepping.py:1100-1102), so the per-component split
   the array path performs is a compile-time specialization here — and the
   gate's identity leg pins the NL=(0,0,0) build byte-identical to the
   certified kernel, which is what makes the seam a measurement.
6. **``prev`` IS READ BEFORE ``f_w`` IS WRITTEN**, and the tail keeps the two
   separate accumulations of ``_apply_constitutive_pml`` (stepping.py:
   2112-2143) — byte-for-byte the certified constitutive family's recurrence,
   mutation included. ``ENABLE_FP_FUSION = False`` is the certified
   configuration for this family's multiply-subtract tail (the beta gate
   measured fusion-on changing bytes on 72/96 cases of the same tail shape).

GHOSTS. Phase A carries exactly the two plain rules of ``_shift_up``
(stepping.py:1770) / ``_shift_down`` (:1834): the periodic wrap and the metallic
zero ghost, per axis, composed independently for the doubly-shifted corner. No
mirror parity (the fold changes the stored extent and is refused), no Bloch
phase (k = 0 only), no far reflect-row (the folded-periodic rule; stepping
itself terminates the nonlinear sums with the zero face and the driver refuses
that pairing, stepping.py:1785-1791).

WHAT IS REFUSED, BY NAME, in :func:`nonlinear_constitutive_coverage` — every
``_grid_reasons`` clause restated verbatim EXCEPT the chi2/chi3 one, which is
INVERTED (this product exists only for nonlinear runs; a zero-chi run belongs to
``coverage.constitutive_coverage(side="E")`` and MEEP's install-time trivial-pair
drop, fields.py:853-857/:879-880, is what makes the two predicates disjoint by
construction): the non-CuPy backend, complex storage (the DOCMP split,
stepping.py:1084-1091, is a Phase B leg composing with the certified complex
tranche), no active PML (without one ``update_E`` is a DIFFERENT sub-step —
``field[...] = constitutive``, stepping.py:1019-1022), boundary kinds outside
{periodic, metallic}, any fold, cylindrical coordinates, nonzero ``k_point``,
BFAST, nonzero ``beta``, any registered polarization (the source becomes
``D - sum P`` per-component scratch buffers — a later fused leg with the
dispersive family), any off-diagonal chi1inv row (MEEP's most-general case
scales the whole ROW product, stepping.py:1095-1099 — the offdiag family's),
un-stored E, and the layout/coefficient clauses of the shared helpers.

The pole guard is NOT a predicate clause: ``nonlinear_margin`` (stepping.py:
1344-1386) is sampled OUTSIDE the step loop by the driver's
``_NonlinearityGuard`` (driver.py:684-720) and is unchanged by this kernel. The
gate's amplitude classes respect the derived ``expansion < 1/3`` bound, and one
class sits NEAR it so the large-u arithmetic is byte-exercised.

Import contract: this module is importable WITHOUT Triton — the predicate and
the plan builders (to ``None``) must answer on the laptop that is the merge bar.
"""

from __future__ import annotations

import struct
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import coverage as _coverage

#: The components this sub-step writes, in ``stepping.E_CONSTITUTIVE_TERMS``
#: order (stepping.py:228), with each component's OWN axis — the axis whose
#: HALF-INTEGER coefficient pair the tail reads (MEEP's ``dsigw``).
E_TERMS: Tuple[Tuple[str, str, int], ...] = (
    ("Ex", "Dx", 0), ("Ey", "Dy", 1), ("Ez", "Dz", 2))

#: The transverse partners of each component, in MEEP's ``cycle_direction``
#: order X -> Y -> Z (vec.hpp:586; stepping.py:1183-1185): own axis + 1 first,
#: own axis + 2 second. Ez therefore takes Dx then Dy. The order is pinned by a
#: test; getting it wrong swaps g1s and g2s, which ``Dsqr``'s commutative sum
#: forgives — the tuple is kept literal anyway so the transcription is checkable
#: without that reasoning.
TRANSVERSE_PARTNERS: Tuple[Tuple[int, int], ...] = ((1, 2), (2, 0), (0, 1))

#: The Yee sub-lattice this side reads: half-integer, ``kps_a_h``/``kms_a_h``
#: (stepping.py:1015 via ``_constitutive_coefficients(..., half_integer=True)``).
HALF_INTEGER = True

#: Elements per program — restated from ``kernels.DEFAULT_BLOCK`` (which needs
#: Triton to import); a test pins the two against the source.
DEFAULT_BLOCK = 256

#: The shared clause builders the predicate composes from, named as data so the
#: laptop test can assert every one still exists in the other session's file.
SHARED_CLAUSES: Tuple[str, ...] = (
    "_boundary_kinds", "_layout_reasons", "_inverse_epsilon_reasons",
    "_coefficient_reasons", "_volume_reasons", "_call")


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------
#
# Triton is imported conditionally and the kernel is defined conditionally, for
# the same measured reason dispersive_update_e.py records: a kernel defined
# inside a lazy builder resolves its names through the defining module's
# __globals__ and dies at first launch.

try:  # pragma: no cover - the absent branch is exercised by the absence test
    import triton
    import triton.language as tl
    _TRITON_IMPORT_ERROR: Optional[BaseException] = None
except ImportError as _exc:  # pragma: no cover - the laptop path
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_IMPORT_ERROR = _exc

if triton is not None:
    #: Boundary codes matching ``stepping``'s string kinds, the package's own
    #: values (kernels.py:77-78). Wrapped in tl.constexpr because a @triton.jit
    #: body may not read a plain module global.
    PERIODIC = tl.constexpr(0)
    METALLIC = tl.constexpr(1)
else:
    PERIODIC = 0  # type: ignore[assignment]
    METALLIC = 1  # type: ignore[assignment]


if triton is not None:

    @triton.jit
    def _four_point_sum(g, o_c, o_d, o_u, o_ud, v_c, v_d, v_u, v_ud):
        """MEEP's ``g1s``: one partner's four-corner sum onto this Yee point.

        ``stepping._nonlinear_transverse_sums`` (:1158-1162) forms
        ``pair = g + shift_down(g, partner)`` then ``pair + shift_up(pair, own)``
        — so the bytes are ``(g[i] + g[i-s1]) + (g[i+s] + g[i+s-s1])``, the pair
        association, NOT MEEP C's left-to-right sum. The far pair is the near
        pair's elementwise sum moved up one cell, so loading the two far corners
        and adding them is the same bits as shifting the formed pair: a shift is
        data movement. Ghost rules compose per axis: a periodic index wraps, a
        metallic out-of-range mask delivers ``other=0.0``, and the corner takes
        BOTH axes' rules (mask AND, wrap independent) exactly as shifting the
        already-shifted pair applies them in sequence.
        """
        near = (tl.load(g + o_c, mask=v_c, other=0.0)
                + tl.load(g + o_d, mask=v_d, other=0.0))
        far = (tl.load(g + o_u, mask=v_u, other=0.0)
               + tl.load(g + o_ud, mask=v_ud, other=0.0))
        return near + far

    @triton.jit
    def _pade_u(gs, dsqr, chi2, chi3, us_sq, us_cu):
        """``stepping.calc_nonlinear_u`` (:996-1027), transcribed term for term.

        ``us_sq``/``us_cu`` arrive FORMED — in-kernel float32 powers for a
        volume epsilon, host-rounded double powers for a scalar one (module
        docstring, point 3) — so this body is one association for both arms::

            c2 = (gs   * chi2) * us_sq
            c3 = (dsqr * chi3) * us_cu
            u  = ((1 + c2) + 2*c3) / ((1 + 2*c2) + 3*c3)

        The division is the transcription's only quotient, and it is spelled
        ``tl.math.div_rn(num, den)`` because the plain ``/`` is NOT IEEE
        division on this platform (``div.full.f32``, ~2 ulp — measured, jobs
        2336/2338; docstring point 4). The near-pole gate class is what
        certifies it.
        """
        c2 = (gs * chi2) * us_sq
        c3 = (dsqr * chi3) * us_cu
        num = (1.0 + c2) + 2.0 * c3
        den = (1.0 + 2.0 * c2) + 3.0 * c3
        return tl.math.div_rn(num, den)

    @triton.jit
    def nonlinear_constitutive_step(
        f0, f1, f2,                     # targets:      Ex, Ey, Ez            (in/out)
        w0, w1, w2,                     # auxiliaries:  f_w_Ex, f_w_Ey, f_w_Ez (in/out)
        g0, g1, g2,                     # sources:      Dx, Dy, Dz            (in, read-only)
        e0, e1, e2,                     # inverse-epsilon volumes (own D pointer when scalar)
        q20, q21, q22,                  # chi2 volumes (own D pointer when scalar/linear)
        q30, q31, q32,                  # chi3 volumes (same rule)
        kp0, km0, kp1, km1, kp2, km2,   # kps/kms on each component's OWN axis, half-integer
        ue0, ue0_2, ue0_3,              # scalar inverse-epsilon arm, component 0
        ue1, ue1_2, ue1_3,              #   (f32(us), f32(us^2), f32(us^3) — double powers,
        ue2, ue2_2, ue2_3,              #    rounded once on the host; docstring point 3)
        s20, s21, s22,                  # scalar chi2 per component
        s30, s31, s32,                  # scalar chi3 per component
        nx, ny, nz, n_elem,
        NL0: tl.constexpr, NL1: tl.constexpr, NL2: tl.constexpr,
        EV0: tl.constexpr, EV1: tl.constexpr, EV2: tl.constexpr,
        QV20: tl.constexpr, QV21: tl.constexpr, QV22: tl.constexpr,
        QV30: tl.constexpr, QV31: tl.constexpr, QV32: tl.constexpr,
        BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """``stepping.update_E`` under PML with chi2/chi3 installed, all three components.

        The body is :func:`kernels.constitutive_step`'s E arm with ONE
        substitution per nonlinear component: ``src`` is no longer ``D*inv_eps``
        but ``(D*inv_eps) * u`` with ``u`` the Pade factor built from the
        four-point transverse average of the two partner D volumes. Everything
        else — the prev-before-store ordering, the two separate accumulations,
        the own-axis half-integer coefficient index — is that kernel's,
        unchanged, because the array path's is unchanged.

        ``NL*`` select the nonlinear body per component at compile time; the 0
        arm is the certified plain body verbatim (docstring point 5). ``EV*`` /
        ``QV2*`` / ``QV3*`` select volume vs fp32-scalar operands; unused
        pointer slots are bound to the component's own D pointer by the plan
        and never read. ``BC*`` carry the per-axis ghost rule for the sums'
        neighbour loads — the only stencil this sub-step has.
        """
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = idx < n_elem
        nyz = ny * nz
        k = idx % nz
        plane = idx // nz
        j = plane % ny
        i = plane // ny

        # --- per-axis neighbour indices, both directions, with the ghost rule
        # (stepping._shift_down / _shift_up, plain PERIODIC/METALLIC branches).
        di, dj, dk = i - 1, j - 1, k - 1
        ui, uj, uk = i + 1, j + 1, k + 1
        dvx, dvy, dvz = live, live, live
        uvx, uvy, uvz = live, live, live
        if BCX == METALLIC:
            dvx = live & (di >= 0)
            uvx = live & (ui < nx)
        else:
            di = tl.where(di < 0, nx - 1, di)
            ui = tl.where(ui == nx, 0, ui)
        if BCY == METALLIC:
            dvy = live & (dj >= 0)
            uvy = live & (uj < ny)
        else:
            dj = tl.where(dj < 0, ny - 1, dj)
            uj = tl.where(uj == ny, 0, uj)
        if BCZ == METALLIC:
            dvz = live & (dk >= 0)
            uvz = live & (uk < nz)
        else:
            dk = tl.where(dk < 0, nz - 1, dk)
            uk = tl.where(uk == nz, 0, uk)

        # Component 0 takes its coefficient from axis x, 1 from y, 2 from z.
        kp_0 = tl.load(kp0 + i, mask=live, other=0.0)
        km_0 = tl.load(km0 + i, mask=live, other=0.0)
        kp_1 = tl.load(kp1 + j, mask=live, other=0.0)
        km_1 = tl.load(km1 + j, mask=live, other=0.0)
        kp_2 = tl.load(kp2 + k, mask=live, other=0.0)
        km_2 = tl.load(km2 + k, mask=live, other=0.0)

        # --- component 0: Ex — own axis x; partners Dy (down y) then Dz (down z)
        prev0 = tl.load(w0 + idx, mask=live, other=0.0)   # BEFORE the store.
        gs0 = tl.load(g0 + idx, mask=live, other=0.0)
        if EV0:
            us0 = tl.load(e0 + idx, mask=live, other=0.0)
        else:
            us0 = ue0
        if NL0:
            g1s = _four_point_sum(
                g1, idx,
                i * nyz + dj * nz + k,
                ui * nyz + j * nz + k,
                ui * nyz + dj * nz + k,
                live, dvy, uvx, uvx & dvy)
            g2s = _four_point_sum(
                g2, idx,
                i * nyz + j * nz + dk,
                ui * nyz + j * nz + k,
                ui * nyz + j * nz + dk,
                live, dvz, uvx, uvx & dvz)
            dsqr0 = gs0 * gs0 + 0.0625 * (g1s * g1s + g2s * g2s)
            if QV20:
                chi2_0 = tl.load(q20 + idx, mask=live, other=0.0)
            else:
                chi2_0 = s20
            if QV30:
                chi3_0 = tl.load(q30 + idx, mask=live, other=0.0)
            else:
                chi3_0 = s30
            if EV0:
                us0_sq = us0 * us0
                us0_cu = us0_sq * us0
            else:
                us0_sq = ue0_2
                us0_cu = ue0_3
            u0 = _pade_u(gs0, dsqr0, chi2_0, chi3_0, us0_sq, us0_cu)
            src0 = (gs0 * us0) * u0
        else:
            src0 = gs0 * us0
        tl.store(w0 + idx, src0, mask=live)
        a0 = tl.load(f0 + idx, mask=live, other=0.0)
        a0 = a0 + kp_0 * src0
        a0 = a0 - km_0 * prev0
        tl.store(f0 + idx, a0, mask=live)

        # --- component 1: Ey — own axis y; partners Dz (down z) then Dx (down x)
        prev1 = tl.load(w1 + idx, mask=live, other=0.0)
        gs1 = tl.load(g1 + idx, mask=live, other=0.0)
        if EV1:
            us1 = tl.load(e1 + idx, mask=live, other=0.0)
        else:
            us1 = ue1
        if NL1:
            g1s = _four_point_sum(
                g2, idx,
                i * nyz + j * nz + dk,
                i * nyz + uj * nz + k,
                i * nyz + uj * nz + dk,
                live, dvz, uvy, uvy & dvz)
            g2s = _four_point_sum(
                g0, idx,
                di * nyz + j * nz + k,
                i * nyz + uj * nz + k,
                di * nyz + uj * nz + k,
                live, dvx, uvy, uvy & dvx)
            dsqr1 = gs1 * gs1 + 0.0625 * (g1s * g1s + g2s * g2s)
            if QV21:
                chi2_1 = tl.load(q21 + idx, mask=live, other=0.0)
            else:
                chi2_1 = s21
            if QV31:
                chi3_1 = tl.load(q31 + idx, mask=live, other=0.0)
            else:
                chi3_1 = s31
            if EV1:
                us1_sq = us1 * us1
                us1_cu = us1_sq * us1
            else:
                us1_sq = ue1_2
                us1_cu = ue1_3
            u1 = _pade_u(gs1, dsqr1, chi2_1, chi3_1, us1_sq, us1_cu)
            src1 = (gs1 * us1) * u1
        else:
            src1 = gs1 * us1
        tl.store(w1 + idx, src1, mask=live)
        a1 = tl.load(f1 + idx, mask=live, other=0.0)
        a1 = a1 + kp_1 * src1
        a1 = a1 - km_1 * prev1
        tl.store(f1 + idx, a1, mask=live)

        # --- component 2: Ez — own axis z; partners Dx (down x) then Dy (down y)
        prev2 = tl.load(w2 + idx, mask=live, other=0.0)
        gs2 = tl.load(g2 + idx, mask=live, other=0.0)
        if EV2:
            us2 = tl.load(e2 + idx, mask=live, other=0.0)
        else:
            us2 = ue2
        if NL2:
            g1s = _four_point_sum(
                g0, idx,
                di * nyz + j * nz + k,
                i * nyz + j * nz + uk,
                di * nyz + j * nz + uk,
                live, dvx, uvz, uvz & dvx)
            g2s = _four_point_sum(
                g1, idx,
                i * nyz + dj * nz + k,
                i * nyz + j * nz + uk,
                i * nyz + dj * nz + uk,
                live, dvy, uvz, uvz & dvy)
            dsqr2 = gs2 * gs2 + 0.0625 * (g1s * g1s + g2s * g2s)
            if QV22:
                chi2_2 = tl.load(q22 + idx, mask=live, other=0.0)
            else:
                chi2_2 = s22
            if QV32:
                chi3_2 = tl.load(q32 + idx, mask=live, other=0.0)
            else:
                chi3_2 = s32
            if EV2:
                us2_sq = us2 * us2
                us2_cu = us2_sq * us2
            else:
                us2_sq = ue2_2
                us2_cu = ue2_3
            u2 = _pade_u(gs2, dsqr2, chi2_2, chi3_2, us2_sq, us2_cu)
            src2 = (gs2 * us2) * u2
        else:
            src2 = gs2 * us2
        tl.store(w2 + idx, src2, mask=live)
        a2 = tl.load(f2 + idx, mask=live, other=0.0)
        a2 = a2 + kp_2 * src2
        a2 = a2 - km_2 * prev2
        tl.store(f2 + idx, a2, mask=live)

else:  # pragma: no cover - the laptop path
    nonlinear_constitutive_step = None  # type: ignore[assignment]


def nonlinear_constitutive_step_kernel() -> Any:
    """The shipped ``@triton.jit`` kernel, or a diagnosable ImportError."""
    if nonlinear_constitutive_step is None:
        raise ImportError(
            "the nonlinear update_E kernel needs the optional `triton` package "
            "(pip install triton). The engine runs without it; only this fast "
            f"path is unavailable. Original error: {_TRITON_IMPORT_ERROR}")
    return nonlinear_constitutive_step


# ---------------------------------------------------------------------------
# Host-side scalar arms
# ---------------------------------------------------------------------------

def _f32(value: float) -> float:
    """One round of a Python float to float32, without importing NumPy."""
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def scalar_inverse_epsilon_arm(us: float) -> Tuple[float, float, float]:
    """The three fp32 scalars a SCALAR inverse epsilon ships to the kernel.

    ``(f32(us), f32(us*us), f32((us*us)*us))`` with the powers taken in PYTHON
    DOUBLE and rounded once each — because that is what NumPy's weak-scalar
    semantics do inside ``calc_nonlinear_u`` when ``chi1inv`` is a Python float:
    ``chi1inv * chi1inv`` is a double, rounded once at the multiply with the
    float32 array. ``f32(f32(us) * f32(us))`` is a DIFFERENT word for most
    values (double-rounding), which is why this helper exists as the single
    place the arm is built; a test pins it against NumPy's own bytes.
    """
    us_d = float(us)
    return (_f32(us_d), _f32(us_d * us_d), _f32((us_d * us_d) * us_d))


def scalar_chi_arm(chi: float) -> float:
    """The fp32 scalar a SCALAR chi2/chi3 ships to the kernel.

    Chi appears LINEARLY (once) in its factor, so NumPy's weak-scalar rule is a
    single cast: ``di * chi`` is ``di ⊗_f32 f32(chi)``. One rounding, here.
    """
    return _f32(chi)


def _is_volume(value: Any) -> bool:
    return bool(getattr(value, "shape", None))


def _base_address(array: Any) -> Optional[int]:
    """The device or host base address of a volume, or None when unreadable."""
    data = getattr(array, "data", None)
    pointer = getattr(data, "ptr", None)
    if pointer is not None:  # CuPy
        return int(pointer)
    interface = getattr(array, "__array_interface__", None)
    if isinstance(interface, dict):  # NumPy
        return int(interface["data"][0])
    return None


# ---------------------------------------------------------------------------
# The coverage predicate
# ---------------------------------------------------------------------------

def _nonlinear_grid_reasons(fields: Any, pml: Any, grid: Any) -> List[str]:
    """``coverage._grid_reasons``' clauses, RESTATED, with the chi clause inverted.

    Restated rather than called because the shared file's clause 10
    (coverage.py:195-196) refuses every nonlinear run and that file is another
    session's to narrow — the same discipline ``special_kz._beta_real_grid_reasons``
    followed for the beta clause. Every other clause is kept verbatim in force.
    """
    reasons: List[str] = []

    # 1. CuPy backend. The kernel launches against device pointers.
    xp = getattr(grid, "xp", None)
    if getattr(xp, "__name__", "") != "cupy":
        reasons.append(f"array module is {getattr(xp, '__name__', xp)!r}, not cupy")

    # 2. Real storage. MEEP nonlinearizes the two parts of a complex field
    #    INDEPENDENTLY (the DOCMP split, stepping.py:1084-1091/:1110-1116, with
    #    the parts taken AFTER the sums are formed because a Bloch wrap mixes
    #    them, :1109-1112); that arm is a Phase B leg composing with the
    #    certified complex tranche, not this kernel.
    if getattr(fields, "force_complex_fields", False):
        reasons.append("force_complex_fields=True (the DOCMP nonlinear split is "
                       "a Phase B leg with the complex tranche)")

    # 3. An active absorber. Without one update_E writes
    #    ``field[...] = constitutive`` (stepping.py:1019-1022) — a DIFFERENT
    #    sub-step, with no f_w and no kps/kms, refused with its own name.
    if pml is None or not getattr(pml, "is_active", False):
        reasons.append("no active PML: without one the nonlinear update_E is "
                       "field[...] = constitutive (stepping.py:1019-1022), a "
                       "different sub-step this kernel does not carry")

    # 4. Only the two ghost rules the transverse sums transcribe.
    kinds = _coverage._boundary_kinds(
        grid, pml if (pml is not None and getattr(pml, "is_active", False)) else None)
    if kinds is None:
        reasons.append("boundary kinds could not be resolved for this grid")
    else:
        for axis, kind in enumerate(kinds):
            if kind not in _coverage.COVERED_BOUNDARIES:
                reasons.append(f"axis {axis} boundary {kind!r} is outside "
                               f"{_coverage.COVERED_BOUNDARIES}")

    # 5. No mirror plane anywhere: a fold changes the ghost rule, the stored
    #    extent, and — specifically here — the nonlinear sums' far face on a
    #    folded periodic axis, which stepping terminates with the zero ghost
    #    and the DRIVER refuses to serve (stepping.py:1785-1791).
    if _coverage._call(grid, "has_symmetry", default=False):
        reasons.append("a mirror plane is active (symmetry folding is not carried)")
    for axis in range(3):
        if _coverage._call(grid, "is_mirrored", axis, default=False):
            reasons.append(f"axis {axis} is folded by a mirror plane")

    # 6. Cartesian only.
    if getattr(grid, "cylindrical", False):
        reasons.append("cylindrical (Dcyl) coordinates are not carried")
    for axis in range(3):
        if _coverage._call(grid, "is_axis", axis, default=False):
            reasons.append(f"axis {axis} is the cylindrical r = 0 axis")

    # 7. k = 0. A Bloch phase needs complex storage and phases the wrapped
    #    plane of every shifted operand — including the sums'.
    if getattr(grid, "has_bloch", False):
        reasons.append(f"nonzero k_point {getattr(grid, 'k_point', None)!r}")
    k_point = getattr(grid, "k_point", (0.0, 0.0, 0.0))
    if any(float(component) != 0.0 for component in k_point):
        reasons.append(f"k_point {tuple(k_point)!r} is not exactly zero")

    # 10 (INVERTED). chi2/chi3 must be INSTALLED: this product exists only for
    #     nonlinear runs. A trivial pair never installs (MEEP deletes it,
    #     fields.py:853-857/:879-880), so the zero-chi run is bit-identically
    #     the linear engine's and belongs to constitutive_coverage(side='E') —
    #     the two predicates are disjoint by construction and a test pins it.
    if not getattr(fields, "has_nonlinearity", False):
        reasons.append("no chi2/chi3 is installed: that configuration is "
                       "constitutive_coverage(side='E')'s and this predicate "
                       "must not overlap it")

    # 11/12. BFAST and beta, restated in force.
    if getattr(grid, "bfast_active", False):
        reasons.append("BFAST is active (the second additive curl term is a "
                       "separate family)")
    if float(getattr(grid, "beta", 0.0)) != 0.0:
        reasons.append(f"special_kz beta={getattr(grid, 'beta', 0.0)!r} is nonzero")

    return reasons


def nonlinear_constitutive_coverage(fields: Any, pml: Any) -> "_coverage.Coverage":
    """May the Triton nonlinear kernel step ``update_E`` for this (fields, pml) pair?

    POSITIVE CLAUSES ONLY; a failing clause appends its reason and the scan
    continues. The grid clauses are :func:`_nonlinear_grid_reasons`' (the shared
    set with the chi clause inverted); added here, each a silent wrong answer if
    missing:

    a. **no registered polarization.** With poles the source is ``D - sum P``
       formed in PER-COMPONENT scratch buffers (fields.py:1107-1138) and the
       sums read THOSE, not D — a fused dispersive+nonlinear kernel is a later
       leg, and this predicate must not silently cover its configuration.
    b. **no off-diagonal chi1inv.** MEEP's most-general case scales the whole
       ROW product by the Pade factor (stepping.py:1095-1099); the row product
       belongs to the offdiag family. Refused so ``coupling is None`` is an
       invariant of the launched kernel, not an assumption.
    c. **stored E** — update_E writes an array here, it does not serve
       ``D * inv_eps`` on demand.
    d. every nonlinear component's chi2 AND chi3 answer as a finite float or a
       float32 C-contiguous grid-shape volume (both-or-neither is the
       installer's invariant, fields.py:848-852, and is re-checked).
    e. volume inverse epsilon per component (the shared clause,
       :func:`coverage._inverse_epsilon_reasons`). The kernel CARRIES a scalar
       arm — the harness route exercises it — but the engine installs volumes
       and the engine-route predicate refuses a scalar exactly as the certified
       constitutive predicate does, so the two stay interchangeable there.
    f. layout and the half-integer coefficient tables, via the shared helpers.

    Conductivity is NOT a clause, deliberately: it changes the CURL sub-steps
    only, never the constitutive one (``coverage.constitutive_coverage`` has no
    conductivity clause either, coverage.py:128-130).
    """
    grid = getattr(fields, "grid", None)
    if grid is None:
        return _coverage.Coverage(False, ("fields carries no grid",))

    reasons: List[str] = _nonlinear_grid_reasons(fields, pml, grid)

    # (a) No poles: the D volumes must be the aliased primaries.
    states = tuple(getattr(fields, "polarizations", ()) or ())
    if states or getattr(fields, "has_polarizations", False):
        reasons.append(
            "a susceptibility is registered: the nonlinear source becomes "
            "D - sum P in per-component scratch buffers (fields.py:1107-1138) "
            "— the fused dispersive+nonlinear kernel is a later leg")

    # (b) Diagonal epsilon only.
    if getattr(fields, "has_offdiagonal_epsilon", False):
        reasons.append(
            "an off-diagonal chi1inv row is installed: MEEP's most-general "
            "case scales the whole row product (stepping.py:1095-1099), which "
            "belongs to the offdiag family")

    # (c) Stored E.
    if not getattr(fields, "stores_E", False):
        reasons.append("E is recomputed from D rather than stored")

    # (d) The chi pair of every nonlinear component, readable and well-formed.
    shape = tuple(getattr(grid, "shape", ()))
    nonlinear = tuple(getattr(fields, "nonlinear_components", ()) or ())
    for component in nonlinear:
        if component not in _coverage.ELECTRIC_COMPONENTS:
            reasons.append(f"nonlinear component {component!r} is outside "
                           f"{_coverage.ELECTRIC_COMPONENTS}")
            continue
        for label, reader in (("chi2", "chi2_for"), ("chi3", "chi3_for")):
            try:
                value = getattr(fields, reader)(component)
            except Exception as exc:  # noqa: BLE001 - unreadable is not covered
                reasons.append(f"{reader}({component!r}) raised {exc!r}")
                continue
            if _is_volume(value):
                if len(shape) == 3:
                    reasons.extend(_coverage._volume_reasons(
                        f"{label}[{component}]", value, shape))
            else:
                try:
                    number = float(value)
                except Exception:  # noqa: BLE001 - neither scalar nor volume
                    reasons.append(f"{label}[{component}]={value!r} is neither "
                                   f"a scalar nor a volume")
                    continue
                if number != number or number in (float("inf"), float("-inf")):
                    reasons.append(f"{label}[{component}]={number!r} is not finite")

    # (e)/(f) The volumes this sub-step reads and writes, and their layout.
    names = tuple(term[0] for term in E_TERMS)
    names += tuple("f_w_" + term[0] for term in E_TERMS)
    names += tuple(term[1] for term in E_TERMS)
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")
    reasons.extend(_coverage._layout_reasons(fields, shape, names))
    if len(shape) == 3:
        reasons.extend(_coverage._inverse_epsilon_reasons(fields, shape))
        if pml is not None and getattr(pml, "is_active", False):
            reasons.extend(_coverage._coefficient_reasons(
                pml, shape, ("kps", "kms"), ("_h",) if HALF_INTEGER else ("",)))

    return _coverage.Coverage(not reasons, tuple(reasons))


def chi_pair_for(fields: Any, component: str) -> Optional[Tuple[Any, Any]]:
    """This component's installed (chi2, chi3), or None where it steps linear.

    The single place the per-component split is derived, so the predicate, the
    plan and the gate cannot disagree about which components take the Pade arm
    — the same discipline :func:`dispersive_update_e.poles_per_component`
    exists for. ``Fields.is_nonlinear`` (fields.py:973-974) is the arbiter.
    """
    if not _coverage._call(fields, "is_nonlinear", component, default=False):
        return None
    return (fields.chi2_for(component), fields.chi3_for(component))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class NonlinearConstitutivePlan:
    """A launchable, allocation-free nonlinear ``update_E``.

    The Yee sub-lattice is chosen at the BUILDERS and nowhere else:
    ``kps_a_h``/``kms_a_h``, the half-integer tables (stepping.py:1015). The
    kernel takes six coefficient pointers and never asks which lattice they
    came from — a swap is the silent half-cell absorber error the gate carries
    a mutation for.

    Everything is resolved ahead of the loop: the D volumes are true primaries
    here (no poles are admitted), so unlike the dispersive plan there is no
    per-launch pointer rotation to chase — the pole-free configuration is
    exactly what clause (a) of the predicate guarantees.

    ``__init__`` REFUSES ALIASING between the outputs (E, f_w) and any input
    (D, inverse epsilon, chi volumes, the six coefficient vectors), and among
    the outputs themselves: the
    sums re-read the D volumes at neighbour offsets while E and f_w are being
    written, so an aliased pair would make the result depend on block schedule
    — a wrong answer that varies run to run.
    """

    __slots__ = ("shape", "n_elem", "block", "nonlinear", "boundary_codes",
                 "epsilon_is_volume", "chi2_is_volume", "chi3_is_volume",
                 "epsilon_scalars", "chi2_scalars", "chi3_scalars",
                 "_targets", "_aux", "_sources", "_inv_eps", "_chi2", "_chi3",
                 "_coefficients", "_grid", "_kernel")

    def __init__(self, shape, block: int, targets, auxiliaries, sources,
                 inverse_epsilon, chi2, chi3, coefficients, boundary_codes,
                 kernel: Any = None) -> None:
        from .launch import CupyPointer, _flat  # noqa: PLC0415

        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.block = int(block)
        self.boundary_codes = tuple(int(code) for code in boundary_codes)
        if len(self.boundary_codes) != 3 or any(
                code not in (0, 1) for code in self.boundary_codes):
            raise ValueError(f"boundary codes must be three of {{0, 1}}, got "
                             f"{boundary_codes!r}")

        targets = tuple(targets)
        auxiliaries = tuple(auxiliaries)
        sources = tuple(sources)
        if not (len(targets) == len(auxiliaries) == len(sources) == 3):
            raise ValueError("a plan carries exactly three components")
        chi2 = tuple(chi2)
        chi3 = tuple(chi3)
        inverse_epsilon = tuple(inverse_epsilon)
        for index in range(3):
            if (chi2[index] is None) != (chi3[index] is None):
                raise ValueError(
                    f"component {index}: chi2 and chi3 travel together — both "
                    f"or neither (fields.py:848-852); got chi2={chi2[index]!r} "
                    f"chi3={chi3[index]!r}")
        self.nonlinear = tuple(int(chi2[index] is not None) for index in range(3))
        if not any(self.nonlinear):
            raise ValueError(
                "no component is nonlinear: that configuration belongs to the "
                "certified plain constitutive kernel, and building this plan "
                "for it would overlap the two")

        self._require_no_aliasing(targets, auxiliaries, sources,
                                  inverse_epsilon, chi2, chi3, coefficients)

        self._targets = tuple(CupyPointer(a) for a in targets)
        self._aux = tuple(CupyPointer(a) for a in auxiliaries)
        self._sources = tuple(CupyPointer(a) for a in sources)
        self._coefficients = tuple(CupyPointer(_flat(a)) for a in coefficients)

        # Volume-vs-scalar arms. Unused pointer slots take the component's own
        # D pointer (never read; the constexpr arm is what stops the read, the
        # pointer still has to type). Scalar slots are host-rounded HERE, once.
        self.epsilon_is_volume = tuple(int(_is_volume(v)) for v in inverse_epsilon)
        self.chi2_is_volume = tuple(
            int(value is not None and _is_volume(value)) for value in chi2)
        self.chi3_is_volume = tuple(
            int(value is not None and _is_volume(value)) for value in chi3)
        self._inv_eps = tuple(
            CupyPointer(value) if _is_volume(value) else self._sources[index]
            for index, value in enumerate(inverse_epsilon))
        self._chi2 = tuple(
            CupyPointer(value) if (value is not None and _is_volume(value))
            else self._sources[index]
            for index, value in enumerate(chi2))
        self._chi3 = tuple(
            CupyPointer(value) if (value is not None and _is_volume(value))
            else self._sources[index]
            for index, value in enumerate(chi3))
        self.epsilon_scalars = tuple(
            (0.0, 0.0, 0.0) if _is_volume(value)
            else scalar_inverse_epsilon_arm(float(value))
            for value in inverse_epsilon)
        self.chi2_scalars = tuple(
            0.0 if (value is None or _is_volume(value)) else scalar_chi_arm(value)
            for value in chi2)
        self.chi3_scalars = tuple(
            0.0 if (value is None or _is_volume(value)) else scalar_chi_arm(value)
            for value in chi3)

        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        self._kernel = kernel

    @staticmethod
    def _require_no_aliasing(targets, auxiliaries, sources, inverse_epsilon,
                             chi2, chi3, coefficients) -> None:
        """Base-address disjointness of outputs vs outputs and inputs vs
        outputs — the six coefficient vectors included in the input inventory
        (they are re-read per element while E is written, so a coefficient
        aliasing an output is the same schedule-dependent wrong answer).
        Equality of BASE addresses only: two overlapping views with different
        bases pass unseen, the certified plans' shared (accepted) limitation.
        """
        outputs = {}
        for name_group, group in (("target", targets), ("aux", auxiliaries)):
            for index, array in enumerate(group):
                address = _base_address(array)
                if address is None:
                    raise ValueError(
                        f"{name_group} {index} exposes no readable base address; "
                        f"an unverifiable output is not accepted")
                if address in outputs:
                    raise ValueError(
                        f"{name_group} {index} aliases {outputs[address]}; the "
                        f"outputs must be distinct arrays")
                outputs[address] = f"{name_group} {index}"
        inputs = list(sources) + [v for v in inverse_epsilon if _is_volume(v)]
        inputs += [v for v in tuple(chi2) + tuple(chi3)
                   if v is not None and _is_volume(v)]
        inputs += [v for v in coefficients if _is_volume(v)]
        for array in inputs:
            address = _base_address(array)
            if address is not None and address in outputs:
                raise ValueError(
                    f"an input volume aliases {outputs[address]}: the "
                    f"transverse sums re-read the D volumes at neighbour "
                    f"offsets while the outputs are written, so an alias makes "
                    f"the answer depend on block schedule")

    def run(self, guard: Optional[bool] = None) -> None:
        """Launch the sub-step, in place. Same ``guard`` contract as every plan."""
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        kernel = (self._kernel if self._kernel is not None
                  else nonlinear_constitutive_step_kernel())
        nx, ny, nz = self.shape
        scalars: List[float] = []
        for triple in self.epsilon_scalars:
            scalars.extend(triple)
        kernel[self._grid](
            *self._targets, *self._aux, *self._sources,
            *self._inv_eps, *self._chi2, *self._chi3,
            *self._coefficients,
            *scalars, *self.chi2_scalars, *self.chi3_scalars,
            nx, ny, nz, self.n_elem,
            NL0=self.nonlinear[0], NL1=self.nonlinear[1], NL2=self.nonlinear[2],
            EV0=self.epsilon_is_volume[0], EV1=self.epsilon_is_volume[1],
            EV2=self.epsilon_is_volume[2],
            QV20=self.chi2_is_volume[0], QV21=self.chi2_is_volume[1],
            QV22=self.chi2_is_volume[2],
            QV30=self.chi3_is_volume[0], QV31=self.chi3_is_volume[1],
            QV32=self.chi3_is_volume[2],
            BCX=self.boundary_codes[0], BCY=self.boundary_codes[1],
            BCZ=self.boundary_codes[2],
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
        )

    def __repr__(self) -> str:
        return (f"NonlinearConstitutivePlan(shape={self.shape}, "
                f"nonlinear={self.nonlinear}, bc={self.boundary_codes}, "
                f"block={self.block})")


#: Boundary kind string -> the kernel's constexpr code. The strings are
#: ``stepping``'s own; the codes are the package's (kernels.py:77-78).
BOUNDARY_CODES = {"periodic": 0, "metallic": 1}


def plan_nonlinear_constitutive(fields: Any, pml: Any,
                                block: Optional[int] = None
                                ) -> Optional[NonlinearConstitutivePlan]:
    """Build the nonlinear ``update_E`` plan from the engine's own objects, or None.

    None means REFUSED, and the reasons are available from
    :func:`nonlinear_constitutive_coverage`. The Triton import stays below the
    predicate: a NumPy host must be able to plan (to ``None``) without the
    optional dependency being importable at all.
    """
    if not nonlinear_constitutive_coverage(fields, pml).covered:
        return None
    kinds = _coverage._boundary_kinds(fields.grid, pml)
    codes = tuple(BOUNDARY_CODES[kind] for kind in kinds)
    targets = tuple(term[0] for term in E_TERMS)
    pairs = [chi_pair_for(fields, name) for name in targets]
    return NonlinearConstitutivePlan(
        fields.grid.shape, DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in targets],
        [getattr(fields, "f_w_" + name) for name in targets],
        [getattr(fields, term[1]) for term in E_TERMS],
        [fields.inverse_epsilon_for(name) for name in targets],
        [None if pair is None else pair[0] for pair in pairs],
        [None if pair is None else pair[1] for pair in pairs],
        [getattr(pml, f"{stem}_{axis}_h")
         for axis in "xyz" for stem in ("kps", "kms")],
        codes,
    )


def plan_nonlinear_constitutive_from_arrays(arrays: Dict[str, Any],
                                            flat: Dict[str, Any],
                                            chi2: Dict[str, Any],
                                            chi3: Dict[str, Any],
                                            boundary_codes: Sequence[int],
                                            block: Optional[int] = None,
                                            kernel: Any = None
                                            ) -> NonlinearConstitutivePlan:
    """Build it from bare arrays — the gate's and benchmark's route.

    ``arrays`` is keyed by component name (``Ex``, ``f_w_Ex``, ``Dx``,
    ``inv_eps_Ex``; the inverse-epsilon entries may be Python floats, which is
    how the harness exercises the scalar arm the engine route never reaches),
    ``flat`` by ``kps_x``/``kms_x``/... on the sub-lattice THE CALLER already
    selected, and ``chi2``/``chi3`` map each component to a float, a volume, or
    None for a linear component. No coverage predicate runs here: the caller is
    a harness that constructed the configuration on purpose.

    ``kernel=`` IS LOAD-BEARING: the mutation legs route a mutated kernel
    through it, and a builder that drops it launches the shipped kernel and
    reports a pass for a defect it never introduced (plan §13.4's measured
    harness-disarm failure).
    """
    targets = tuple(term[0] for term in E_TERMS)
    shape = tuple(int(n) for n in arrays[targets[0]].shape)
    return NonlinearConstitutivePlan(
        shape, DEFAULT_BLOCK if block is None else block,
        [arrays[name] for name in targets],
        [arrays["f_w_" + name] for name in targets],
        [arrays[term[1]] for term in E_TERMS],
        [arrays["inv_eps_" + name] for name in targets],
        [chi2.get(name) for name in targets],
        [chi3.get(name) for name in targets],
        [flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kps", "kms")],
        boundary_codes,
        kernel=kernel,
    )


# ---------------------------------------------------------------------------
# The OTHER sub-steps of a nonlinear run — predicate scope only, no new kernel
# ---------------------------------------------------------------------------
#
# MEASURED, not argued. `coverage._grid_reasons` clause 10 (coverage.py:192-196)
# refuses chi2/chi3 on the WHOLE GRID and says so in its own comment ("refused on
# every sub-step rather than only on update_E"). That is broader than the engine:
#
#   * `stepping.step_B` (S:261-376), `stepping.step_D` (S:379-457),
#     `_apply_curl` (S:460-510) and `stepping.update_H` (S:907-923) mention chi2,
#     chi3, `has_nonlinearity`, `calc_nonlinear_u` and `_nonlinear_displacement`
#     NOWHERE — a grep over those ranges returns nothing;
#   * `update_E`'s own docstring says where it does enter: "An instantaneous
#     chi2/chi3 enters HERE and only here" (S:947-951).
#
# The device leg ran the CERTIFIED bodies against `stepping.py` on a chi3 run,
# EIGHT complete cycles each, uint32 over the whole stored inventory. `moved` is
# the MINIMUM over cycles of what the sub-step itself wrote, bracketed around that
# one call on BOTH sides, and `minus0` the minimum live +-0 census going into it:
#
#   A_chi_step_B            IDENTICAL   0 / 12288 words differing  (>=2938 moved)
#   A_chi_step_D            IDENTICAL   0 / 12288                  (>=2922 moved)
#   A_chi_update_H          IDENTICAL   0 / 12288                  (>=3072 moved)
#   A_chi_update_E_CONTROL  DIVERGENT   1536 / 12288  <- and it had to diverge
#
# (parity/meep_gpu/results/residual_closure_2026-08-15/device/bodies/bodies.json,
# minus0 >= 384 on every one.) So the clause is over-broad for the curls and for
# update_H, and load-bearing for update_E — where `nonlinear_constitutive_coverage`
# above already carries the run.
#
# AND THE ADMISSION ITSELF WAS MEASURED, not just the body. The triage leg above
# reaches the certified body by patching the INCUMBENT predicate to admit, which
# licenses a claim about the kernel and not about the two predicates below. The
# closure round's second leg builds through `plan_nonlinear_run_pml_curl` and
# `plan_nonlinear_run_constitutive` with their OWN predicates unpatched — a plan of
# None would be a failed leg — and measures the same identity:
#
#   NEW_nonlinear_run_step_B    IDENTICAL  0 / 12288  (>=2938 moved, minus0 >= 384)
#   NEW_nonlinear_run_step_D    IDENTICAL  0 / 12288  (>=2922 moved, minus0 >= 384)
#   NEW_nonlinear_run_update_H  IDENTICAL  0 / 12288  (>=3072 moved, minus0 >= 384)
#
# (results/residual_closure_2026-08-15/device/newpred/new_predicates.json, whose
# first leg is a 34-check predicate battery with 0 unexpected verdicts.)
#
# The two predicates below are therefore ADMISSIONS, not arithmetic: what they
# plan is `launch.PmlCurlPlan` and `launch.ConstitutivePlan`, the certified
# kernels, untouched. They restate the shared clause set with clause 10 inverted
# — chi2/chi3 must be INSTALLED — so they are disjoint from the incumbents by
# construction rather than by dispatch order, which is the property that stops
# `plan_step` choosing between two admitting predicates at random.


def _nonlinear_run_grid_reasons(fields: Any, pml: Any, grid: Any) -> List[str]:
    """``coverage._grid_reasons``, RESTATED for the sub-steps chi2/chi3 does NOT enter.

    Restated and not called, and not shared with :func:`_nonlinear_grid_reasons`
    either. The shared file's clause 10 refuses every nonlinear run and that file
    is another session's to narrow; ``_nonlinear_grid_reasons`` inverts the same
    clause but spells its absorber and storage refusals in ``update_E``'s words
    ("without one the nonlinear update_E is field[...] = constitutive"), which
    would be a TRUE VERDICT WITH A FALSE REASON on a curl. ``no_pml.py``'s clause
    3b comment records that failure mode and why it is worth a second
    transcription. Every clause other than 10 is kept verbatim in force.
    """
    reasons: List[str] = []

    # 1. CuPy backend. The certified kernels launch against device pointers.
    xp = getattr(grid, "xp", None)
    if getattr(xp, "__name__", "") != "cupy":
        reasons.append(f"array module is {getattr(xp, '__name__', xp)!r}, not cupy")

    # 2. Real storage. complex64 changes the STORAGE, not the recurrence, and the
    #    complex tranche has its own certified curl and constitutive bodies.
    if getattr(fields, "force_complex_fields", False):
        reasons.append("force_complex_fields=True (complex64 storage is not carried; "
                       "that is complex_fields.py's certified pair)")

    # 3. An absorber that actually absorbs — the split-field recurrence is the only
    #    curl these kernels write, and update_H returns at stepping.py:944-945
    #    without one.
    if pml is None or not getattr(pml, "is_active", False):
        reasons.append("no active PML layer (these kernels implement the split-field "
                       "path only; without one update_H returns at stepping.py:944-945)")

    # 4. Only the two ghost rules the curl kernel writes.
    kinds = _coverage._boundary_kinds(
        grid, pml if (pml is not None and getattr(pml, "is_active", False)) else None)
    if kinds is None:
        reasons.append("boundary kinds could not be resolved for this grid")
    else:
        for axis, kind in enumerate(kinds):
            if kind not in _coverage.COVERED_BOUNDARIES:
                reasons.append(f"axis {axis} boundary {kind!r} is outside "
                               f"{_coverage.COVERED_BOUNDARIES}")

    # 5. No mirror plane: a fold changes the ghost rule, the parity mask and the
    #    STORED EXTENT, and therefore every cell's coefficient index.
    if _coverage._call(grid, "has_symmetry", default=False):
        reasons.append("a mirror plane is active (symmetry folding is not carried)")
    for axis in range(3):
        if _coverage._call(grid, "is_mirrored", axis, default=False):
            reasons.append(f"axis {axis} is folded by a mirror plane")

    # 6. Cartesian only.
    if getattr(grid, "cylindrical", False):
        reasons.append("cylindrical (Dcyl) coordinates are not carried")
    for axis in range(3):
        if _coverage._call(grid, "is_axis", axis, default=False):
            reasons.append(f"axis {axis} is the cylindrical r = 0 axis")

    # 7. k = 0.
    if getattr(grid, "has_bloch", False):
        reasons.append(f"nonzero k_point {getattr(grid, 'k_point', None)!r}")
    k_point = getattr(grid, "k_point", (0.0, 0.0, 0.0))
    if any(float(component) != 0.0 for component in k_point):
        reasons.append(f"k_point {tuple(k_point)!r} is not exactly zero")

    # 10 (INVERTED). chi2/chi3 must be INSTALLED. A linear run is the incumbent
    #     predicates' — `coverage.pml_curl_coverage` and
    #     `coverage.constitutive_coverage` — and two predicates admitting one
    #     configuration means the dispatcher picks by ordering. Disjoint by
    #     construction; a test pins it on both sub-step families.
    if not getattr(fields, "has_nonlinearity", False):
        reasons.append("no chi2/chi3 is installed: that configuration is the "
                       "certified curl/constitutive predicates' and this one "
                       "must not overlap them")

    # 11/12. BFAST and beta, restated in force.
    if getattr(grid, "bfast_active", False):
        reasons.append("BFAST is active (the second additive curl term is not carried)")
    if float(getattr(grid, "beta", 0.0)) != 0.0:
        reasons.append(f"special_kz beta={getattr(grid, 'beta', 0.0)!r} is nonzero")

    return reasons


def nonlinear_run_pml_curl_coverage(fields: Any, pml: Any,
                                    sub_step: str) -> "_coverage.Coverage":
    """May the CERTIFIED real-field PML curl step a NONLINEAR run's ``step_B``/``step_D``?

    The kernel is ``kernels.pml_curl_step``, untouched; only the ADMISSION is
    this module's — the same shape ``folded_complex.plan_folded_beta_run_constitutive``
    has ("the arithmetic and the plan class are launch's; only the ADMISSION is
    this module's").

    Every clause of :func:`coverage.pml_curl_coverage` is restated in force
    except clause 10, which is inverted. In particular this predicate STILL
    REFUSES, by name:

    * conductivity on this sub-step's own targets — the conductive curl is a
      different three-factor recurrence (``stepping._apply_conductive_update``,
      S:1938-1951) and belongs to ``conductivity.py``;
    * a registered susceptibility outside electric lorentzian/drude — the shared
      susceptibility clause, unchanged;
    * off-diagonal chi1inv is ADMITTED here exactly as the incumbent curl
      predicate admits it (its whole effect is inside ``update_E``), and is NOT
      admitted by this module's ``update_E`` predicate;
    * a fold, cylindrical coordinates, Bloch, BFAST, beta, complex storage.

    ``sub_step`` is mandatory. The aggregate verdict the incumbent offers on
    ``None`` is a reporting convenience; a builder that plans one sub-step must
    ask about that sub-step, because the conductivity clause is per sub-step.

    ONE CLAUSE IS NARROWER THAN THE INCUMBENT'S, and it is stated as such rather
    than inherited: an unallocated CURL TARGET (``Bx``..``Dz``) is refused here.
    ``coverage.pml_curl_coverage`` admits it — its allocation clause (:307-315)
    lists ``fu_* + B_SOURCES + D_SOURCES`` and stops, while ``_layout_reasons``
    skips a ``None`` array as "Already reported by the allocation clause"
    (coverage.py:630), which for the targets nothing does. Measured: with
    ``fields.Bx = None`` and the backend clause cleared, the incumbent returns
    ``covered=True`` and ``launch.plan_pml_curl`` builds a ``PmlCurlPlan`` over
    the ``None``. Narrowing is always safe (a refusal is the array path); the
    hole belongs to the shared clause and is reported to the round that owns it.
    """
    if sub_step not in _coverage.CURL_SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(_coverage.CURL_SUB_STEPS)}, "
                         f"got {sub_step!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return _coverage.Coverage(False, ("fields carries no grid",))

    reasons = _nonlinear_run_grid_reasons(fields, pml, grid)

    # 8. Conductivity, per sub-step, exactly as coverage.pml_curl_coverage asks it.
    reader = getattr(fields, "condfac_for", None)
    if callable(reader):
        for target in _coverage.CURL_SUB_STEPS[sub_step]:
            try:
                conductive = reader(target) is not None
            except Exception as exc:  # noqa: BLE001 - unreadable means not covered
                reasons.append(f"condfac_for({target!r}) raised {exc!r}")
                continue
            if conductive:
                reasons.append(
                    f"a conductivity is installed on {target}; this curl belongs "
                    f"to the conductive PML product")
    if (sub_step == "step_B" and getattr(fields, "has_magnetic_conductivity", False)
            and not callable(reader)):
        reasons.append(
            "a magnetic (B) conductivity is installed but condfac_for is unavailable")

    # 9a/9b. A registered susceptibility must be one this package understands.
    reasons.extend(_coverage._susceptibility_reasons(fields))

    # 9c. STORED E — the invariant that lets the B curl difference E at all.
    if not getattr(fields, "stores_E", False):
        reasons.append("E is recomputed from D rather than stored")

    # 13/14. The six CURL TARGETS, the six auxiliaries, the six sources, and their
    #        layout. The TARGETS are named here deliberately, and this is the ONE
    #        place this predicate is narrower than the incumbent it restates.
    #
    #        `coverage.pml_curl_coverage` (:307-315) checks allocation for
    #        `fu_* + B_SOURCES + D_SOURCES` only, then hands the full volume list
    #        — TARGETS included — to `_layout_reasons`, which SKIPS a None array
    #        because "Already reported by the allocation clause" (coverage.py:630).
    #        For the targets it is not: nothing reports them, so an unallocated
    #        `Bx` reaches `plan_pml_curl` and a `PmlCurlPlan` is built binding
    #        None. That is the shared clause's hole, this round may not edit
    #        coverage.py, and a restatement that merely reproduced it would ship a
    #        builder that raises into a caller the contract promises to fall back
    #        for. Narrowing is always safe (a refusal is the array path), so the
    #        clause is stated in force HERE and the hole is reported upward for the
    #        round that owns coverage.py.
    volumes = (tuple(_coverage.CURL_TARGETS)
               + tuple("fu_" + t for t in _coverage.CURL_TARGETS)
               + _coverage.B_SOURCES + _coverage.D_SOURCES)
    for name in volumes:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")
    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(_coverage._layout_reasons(fields, shape, volumes))
    if pml is not None and getattr(pml, "is_active", False) and len(shape) == 3:
        reasons.extend(_coverage._coefficient_reasons(
            pml, shape, ("kms", "sinv"), ("", "_h")))

    return _coverage.Coverage(not reasons, tuple(reasons))


def nonlinear_run_constitutive_coverage(fields: Any, pml: Any,
                                        side: str) -> "_coverage.Coverage":
    """May the CERTIFIED constitutive kernel step a NONLINEAR run's ``update_H``?

    ``side='H'`` ONLY, and ``side='E'`` is REFUSED BY NAME rather than raised:
    'E' is a real side of a real sub-step, it is simply the one the Pade factor
    replaces (``stepping.py:999-1000``), and the CONTROL leg measured 1536 of
    12288 words differing there. That configuration is
    :func:`nonlinear_constitutive_coverage`'s, and it is already covered.

    The magnetic side reads ``B``, writes ``H`` and ``f_w_H*`` with the INTEGER
    coefficient pair, and never touches chi2, chi3 or inverse epsilon — which is
    why the whole nonlinear apparatus is irrelevant to it and why the byte
    comparison found 0 of 12288 words differing over four cycles.
    """
    if side not in _coverage.CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(_coverage.CONSTITUTIVE_SIDES)}, "
                         f"got {side!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return _coverage.Coverage(False, ("fields carries no grid",))

    spec = _coverage.CONSTITUTIVE_SIDES[side]
    reasons = _nonlinear_run_grid_reasons(fields, pml, grid)
    reasons.extend(_coverage._susceptibility_reasons(fields))

    if side == "E":
        reasons.append(
            "side='E' is where the Pade factor REPLACES the constitutive product "
            "(stepping.py:999-1000; measured 1536/12288 words differing against "
            "the certified body): that sub-step is "
            "nonlinear_constitutive_coverage's, not this admission's")

    # A polarization changes update_H not at all (it drives E components only, and
    # the shared susceptibility clause above pins that), so there is no pole clause
    # here — deliberately, and the same way coverage.constitutive_coverage(side='H')
    # carries none.

    names = tuple(spec["targets"]) + tuple(spec["aux"]) + tuple(spec["sources"])
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")
    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(_coverage._layout_reasons(fields, shape, names))
    if pml is not None and getattr(pml, "is_active", False) and len(shape) == 3:
        suffix = ("_h",) if spec["half_integer"] else ("",)
        reasons.extend(_coverage._coefficient_reasons(pml, shape, ("kps", "kms"), suffix))

    return _coverage.Coverage(not reasons, tuple(reasons))


def plan_nonlinear_run_pml_curl(fields: Any, pml: Any, sub_step: str,
                                block: Optional[int] = None,
                                num_warps: Optional[int] = None) -> Any:
    """A CERTIFIED PML curl plan for a nonlinear run, or None.

    The kernel and the plan class are ``launch.PmlCurlPlan``'s, untouched; only
    the admission is this module's. The ``launch`` import stays BELOW the
    predicate so a NumPy host can plan (to ``None``) without Triton.
    """
    if sub_step not in _coverage.CURL_SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(_coverage.CURL_SUB_STEPS)}, "
                         f"got {sub_step!r}")
    if not nonlinear_run_pml_curl_coverage(fields, pml, sub_step).covered:
        return None
    from ..stepping import _boundary_kinds as resolve  # noqa: PLC0415
    from .launch import SUB_STEPS, PmlCurlPlan  # noqa: PLC0415

    spec = SUB_STEPS[sub_step]
    grid = fields.grid
    kinds = resolve(grid, pml)
    return PmlCurlPlan(
        sub_step, grid.shape, grid.dt / grid.dx,
        [1 if kind == "metallic" else 0 for kind in kinds],
        DEFAULT_BLOCK if block is None else block,
        [getattr(fields, n) for n in spec["targets"]],
        [getattr(fields, "fu_" + n) for n in spec["targets"]],
        [getattr(fields, n) for n in spec["sources"]],
        [getattr(pml, f"{stem}_{axis}{spec['suffix']}")
         for axis in "xyz" for stem in ("kms", "sinv")],
        num_warps=num_warps,
    )


def plan_nonlinear_run_constitutive(fields: Any, pml: Any, side: str,
                                    block: Optional[int] = None,
                                    num_warps: Optional[int] = None) -> Any:
    """A CERTIFIED constitutive plan for a nonlinear run's ``update_H``, or None."""
    if side not in _coverage.CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(_coverage.CONSTITUTIVE_SIDES)}, "
                         f"got {side!r}")
    if not nonlinear_run_constitutive_coverage(fields, pml, side).covered:
        return None
    from .launch import ConstitutivePlan  # noqa: PLC0415

    spec = _coverage.CONSTITUTIVE_SIDES[side]
    suffix = "_h" if spec["half_integer"] else ""
    return ConstitutivePlan(
        side, fields.grid.shape, DEFAULT_BLOCK if block is None else block,
        [getattr(fields, n) for n in spec["targets"]],
        [getattr(fields, n) for n in spec["aux"]],
        [getattr(fields, n) for n in spec["sources"]],
        ([fields.inverse_epsilon_for(n) for n in spec["targets"]]
         if side == "E" else None),
        [getattr(pml, f"{stem}_{axis}{suffix}")
         for axis in "xyz" for stem in ("kps", "kms")],
        num_warps=num_warps,
    )


# ---------------------------------------------------------------------------
# INTEGRATION — what landed, and the one thing that deliberately did not
# ---------------------------------------------------------------------------
#
# ``launch.plan_step`` consults all three of this module's admissions:
# ``nonlinear_constitutive_coverage`` on ``update_E`` (the ``nonlinear`` arm,
# wired 2026-08-13) and, from the residual-group closure,
# ``nonlinear_run_pml_curl_coverage`` on ``step_B``/``step_D`` (the
# ``nonlinear run PML`` arm) and ``nonlinear_run_constitutive_coverage`` on
# ``update_H`` (the ``nonlinear run`` arm). Every one is gated on
# ``launch.nonlinear_active`` — this module's own inverted clause, transcribed —
# and the package ``__init__`` exports all three behind the lazy seam.
#
# Those last two close the WHOLE STEP for ``3rd-harm-1d.py`` and
# ``Test3rdHarm1d.test_3rd_harm_1d``: 6 slots, and the only residual group that
# takes real-physics rows (no ``TestLoadDump``) to full whole-step coverage.
#
# THE FUSED D-PAIR DOES NOT INHERIT THE ADMISSION, and must not. That pair
# COMPOSES ``update_E``, where the Pade factor lives, so a fused D-pair on a
# nonlinear run is exactly the configuration the CONTROL leg measured at
# 1536/12288 words differing. ``plan_step`` keeps it out structurally rather than
# by a clause: ``specialized_family_owns_the_grid`` includes ``has_nonlinearity``,
# so opt-in fusion is refused by name on every nonlinear grid, both pairs at once.
#
# ``coverage.py`` CLAUSE 10 WAS NOT NARROWED, deliberately. Rewriting it from
# "refused on every sub-step" to "refused only where the nonlinearity changes the
# arithmetic" would make ``pml_curl_coverage`` and this module's curl predicate
# admit the SAME configuration — and ``_select_slot`` fails closed on a double
# admission, so all six slots would go UNSELECTED and fall to the array path. That
# is a silent coverage LOSS, not an error. The scope change is made by ADDING an
# arm; the incumbent clause stays exactly as it is.
#
# STILL OPEN, in a file this module does not own: ``pml_curl_coverage``'s clause
# 13 (:307-315) checks `fu_* + B_SOURCES + D_SOURCES` and stops, while
# `_layout_reasons` skips a None array as "Already reported by the allocation
# clause" (:630) — which, for the six CURL TARGETS, no clause does. Measured by
# construction: with `fields.Bx = None` and the backend clause cleared, the
# incumbent returns covered=True and `plan_pml_curl` builds a PmlCurlPlan binding
# the None, against the builder contract that an uncarried configuration falls
# back to the array path rather than raising into the caller.
# `nonlinear_run_pml_curl_coverage` states that clause in force locally (a
# narrowing is always safe), and
# `test_an_unallocated_curl_target_is_refused_and_the_builder_answers_none`
# asserts the incumbent's hole, so it turns red the day this is fixed and the two
# can be brought back into step.
#
# ``fingerprints.json`` claims no entry for this module: provenance for a
# specialized module lives with its own gate trio, and ``host_sha256`` names bytes
# that ran on a GPU. What this module plans is a CERTIFIED kernel.
#
# Dispatch is a separate question and is unchanged by any of this: ``fastpath``
# reaches the composer from the driver, and what holds it shut is
# ``fastpath.DISPATCH_BY_DEFAULT`` being False.
