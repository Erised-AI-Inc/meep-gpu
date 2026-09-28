"""The TENSOR (off-diagonal epsilon) electric constitutive sub-step — ``update_E``
with off-diagonal chi1inv rows installed.

One kernel, one coverage predicate, one plan, in one file, on the pattern
:mod:`nonlinear_update_e` set — because this family's whole difference from the
certified constitutive kernel is the OFFDIAG row coupling (a coefficient
multiply sitting BETWEEN two shifts) and the metallic wall-coupling mask, and
none of it belongs in the shared modules yet.

NOT WIRED. Production dispatch is untouched: the engine's fast-path hook keeps
returning ``None`` on every branch and nothing here changes that. ``plan_step``
composition is DEFERRED — ``coverage.constitutive_coverage(side="E")`` already
refuses every off-diagonal run at exactly this sub-step (coverage.py:367-370),
and DELIBERATELY only this sub-step: the curl predicates admit off-diagonal
runs (coverage.py:26-31, :343-346), so this kernel is the LAST uncovered
sub-step of an offdiag run, not the first. ``coverage.py`` is another session's
file and is imported, never edited; ``fingerprints.json`` carries no entry for
this module — the byte gate binds its own provenance record inside its results
directory. Callers are the gate (``parity/meep_gpu/gate_triton_offdiag.py``),
the composition probe (``probe_triton_offdiag_composition.py``) and the laptop
tests, nothing else.

WHAT IT REPLACES. ``stepping.update_E`` (stepping.py:954) with an active PML,
``fields.has_offdiagonal_epsilon`` True (fields.py:1312-1315) and NOT
``fields.has_nonlinearity`` — the ``elif offdiagonal:`` branch (stepping.py:
1001-1008). With no poles admitted, ``displacement_minus_polarization_volumes``
aliases each source to its D primary (fields.py:1107-1138), and all three are
alive at once because the coupling reads the OTHER components' volumes
(stepping.py:991-997). Per component ``c`` with own axis ``a``::

    constitutive = D_c * us_c                               # stepping.py:1005
    per surviving partner (offset 1 then 2, cycle X->Y->Z, :1235-1237):
        pair    = g + shift_down(g, partner_axis)           # :1214-1216
        product = pair * coefficient                        # :1217
        term    = 0.25 * (product + shift_up(product, a))   # :1219-1220
        total   accumulates term(offset1) then term(offset2)  # :1221
    _mask_metallic_wall_coupling(total)                     # :1223, :1227-1254
    constitutive = (D_c * us_c) + total                     # :978-979
    prev = f_w_c ; f_w_c = constitutive                     # :2065-2096
    E_c += kps_a_h * f_w_c ;  E_c -= kms_a_h * prev         # half-integer, :986

THE THINGS THAT DECIDE BIT-IDENTITY, each held by the gate rather than assumed:

1. **THE COEFFICIENT MULTIPLY SITS BETWEEN THE TWO SHIFTS** (stepping.py:
   1243-1249): ``u[i]`` multiplies the two-point partner average AT ITS OWN
   NODE and ``u[i+s]`` the average at the next node up — MEEP registers the
   off-diagonal entry at the component's Yee site minus half a cell along its
   own axis, the integer node (anisotropic_averaging.cpp:248-257,
   ``here - shift1``). A plain four-point-average-times-``u[i]`` hoist is the
   same ALGEBRA only for a uniform coefficient — and MEASURED bitwise-
   different even there, because distributivity (``a*u + b*u`` vs
   ``(a+b)*u``) is not a bitwise identity in f32 (the laptop test pins the
   separation). The gate's spatially varying coefficient cases are what make
   the REGISTRATION — not merely the rounding — byte-visible, and its m1
   mutation is the hoist; the uniform-coefficient row is recorded as an
   outcome, refining the plan's predicted null.
2. **THE TWO SHIFTS GO IN OPPOSITE DIRECTIONS** — half a cell DOWN the
   partner's axis on the raw partner volume (``g[i-sx]``), then the already-
   multiplied product half a cell UP the component's own axis (``(pair*u)[i+s]``,
   which is where the corner ``g[(i+s)-sx]`` arises). Per-axis ghost rules
   compose independently on the corner. Taking both shifts the same way is the
   half-cell registration error; the gate's m2 mutation is exactly it.
3. **0.25 SCALES THE SUM, APPLIED LAST** (stepping.py:1248-1249). 0.25 is
   exact in f32 and the association is the transcribed one — but it is NOT a
   byte-observable choice: an exact power-of-two factor commutes with
   round-to-nearest away from underflow, so the distributed form is bitwise
   identical wherever nothing is subnormal (gate job 2340 measured the
   planted distribution NOT caught on the cancellation class with a
   PTX-verified-different binary; laptop, 0/2M mismatches). The gate carries
   the distribution as a recorded NULL control; the transcribed association
   is fidelity, not a pinned grouping. Contrast the coefficient hoist (m1):
   ARBITRARY-coefficient distributivity rounds differently and is caught.
4. **THE METALLIC WALL-COUPLING MASK** (``_mask_metallic_wall_coupling``,
   stepping.py:1256-1283): the coupling total is zeroed at FACE 0 of every
   metallic non-mirrored axis on which the component's Yee shift is 0, BEFORE
   the row sum — the wall's tangential E is never stepped, the diagonal
   product is already zero there via ``zero_metal_D``, but the coupling reads
   live PARTNER volumes beside the wall (measured 2.6e-02 unmasked against a
   2.0e-07 floor masked). Only the low wall is stored; the high wall is the
   shift-up zero ghost. The gate's m6 (dropped) and m7 (over-applied)
   mutations hold both directions.
5. **THE ROW SUM IS DIAGONAL-FIRST**: ``(D_c * us_c) + total`` (stepping.py:
   1007-1008) — and the accumulation order inside ``total`` is offset 1 then
   offset 2 (:1221). Both additions are bitwise commutative in f32, which is
   why the gate carries the commuted row sum as a NULL control (must NOT be
   caught), not as a mutation.
6. **A COMPONENT WITH NO SURVIVING ROW KEEPS THE PURE DIAGONAL ARITHMETIC** —
   ``_offdiagonal_terms`` returns ``None`` and the caller never forms a
   ``+ 0`` copy (stepping.py:1222-1227). The row-mask constexprs compile the
   'none' arm to :func:`kernels.constitutive_step`'s SCALE=1 component body
   verbatim, and the gate's identity leg pins the rows-all-dropped build
   byte-identical to the certified kernel.
7. **``prev`` IS READ BEFORE ``f_w`` IS WRITTEN**, and the tail keeps the two
   separate accumulations of ``_apply_constitutive_pml`` (stepping.py:
   2112-2143) — byte-for-byte the certified constitutive family's recurrence.
   ``ENABLE_FP_FUSION = False`` is the certified configuration for this
   multiply-subtract tail shape (the beta gate measured fusion-on changing
   bytes on 72/96 cases of it). No unary minus appears on any coefficient
   path (the platform's ``semantic.minus`` lowers ``-x`` as ``0.0 - x`` and
   canonicalizes signed-zero addends).

NO DIVISION ANYWHERE. This family returns to the divide-free profile of the
previously certified families — no Pade quotient, no ``div_rn`` concern, and
no host-rounded scalar powers: the coefficients and the inverse epsilon are
VOLUMES (a scalar inverse epsilon is refused by name, the certified
predicate's own clause, coverage.py:649-674).

GHOSTS. Phase A carries exactly the two plain rules of ``_shift_up``
(stepping.py:1770) / ``_shift_down`` (:1834): the periodic wrap and the
metallic zero ghost, per axis, composed independently for the corner. One of
them is transcription fidelity rather than a byte-observable choice: the
partner-axis metallic NEAR (down) zero ghost's entire support is the
partner-axis face-0 plane, which the wall-coupling mask zeroes before the row
sum — gate job 2340 measured the wrapped-ghost mutant (PTX-verified-different)
byte-identical on the all-metallic sweep, and the gate records it as a NULL
control with that derivation (mirrors, where the mask abstains, are refused
by this family's predicate). The
COEFFICIENT is never ghosted anywhere in the stencil (stepping.py:1216-1220):
the partner-axis shift touches the field before the multiply, and the own-axis
shift moves the PRODUCT — under Bloch the wrap factor on the shifted product
would be the FIELD's phase (the coefficient is periodic and phase-free), out
of scope at k = 0 but the invariant shapes this kernel: coefficient loads take
plainly-indexed slots only, never a phased or parity-weighted ghost. On a
reduced (n = 1) axis the wrap returns the same plane and the partner pair is
``2*g`` — NOT the curl's exact zero — matching MEEP's stride(d)=0 double-read;
a sweep case pins it.

STALE DOCSTRING HAZARD (code wins, and the refusals leg measures it):
stepping.py:1219-1220 claims "a folded axis never reaches this function:
``Fields.set_epsilon_volumes`` refuses the combination at install", and
fields.py:1237-1241 (with driver.py:1202-1204) repeats the claim — but
``_validated_offdiagonal_rows`` (fields.py:1262-1310) installs rows UNCHANGED
on folded grids, with fold-equivalence measured 8.3e-13..4.7e-12 and pinned by
``test_tensor_epsilon.py::test_tensor_fold_equivalence_is_exact``. This Triton
family refuses folds anyway (the shared extent/coefficient-index clause), so
the refusal is THIS KERNEL FAMILY'S, not the engine's: the gate's refusals leg
shows the predicate refusing a folded run that stepping demonstrably steps.
The doc fix belongs to the sessions that own those files.

WHAT IS REFUSED, BY NAME, in :func:`offdiag_constitutive_coverage`: every
``coverage._grid_reasons`` clause IN FORCE and called directly (no clause is
inverted — the chi2/chi3 clause stays: MEEP's most-general row-product-times-
Pade case, stepping.py:1095-1116, is a later fused leg with the nonlinear
family; a nonzero beta is additionally refused by the ENGINE itself for
real-storage offdiag runs, stepping.py:800-810, MEEP fields.cpp:548-549 — the
predicate refusal and the engine ValueError are distinct facts the refusals
leg asserts separately). Complex64 storage is Phase B: the pure-offdiag
``elif`` branch does run unchanged on complex arrays with a real f32
coefficient, but which family carries that arm is a Phase B decision. Added
E-side clauses, each a silent wrong answer if missing: any registered
polarization (the source becomes ``D - sum P`` scratch — the ADE/fused
families'); the INVERTED offdiag clause (at least one surviving row REQUIRED,
counted over the six :data:`ROW_SLOTS` rather than the bare
``has_offdiagonal_epsilon`` flag — a row planted past the installer under a
diagonal key sets the flag with every slot dead; zero-row runs belong to
``constitutive_coverage(side="E")``, disjoint by the install-time drop at
fields.py:1302-1303); ``stores_E`` (forced True by any surviving row,
fields.py:1254-1255 — belt and braces with a reason); volume inverse epsilon;
row volumes real/f32/C-contiguous/grid-shape and ALIASING NO E/f_w OUTPUT
(reachable through the public installer, which keeps the caller's array
without copying — fields.py:1296); and the shared layout/coefficient clauses.
Wherever this predicate answers covered the plan builder must return a plan,
never raise: the builder's own raises are backstops for harness callers, and
each has a predicate clause in front of it on the engine route.

Import contract: this module is importable WITHOUT Triton — the predicate and
the plan builders (to ``None``) must answer on the laptop that is the merge
bar.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import coverage as _coverage

#: The components this sub-step writes, in ``stepping.E_CONSTITUTIVE_TERMS``
#: order (stepping.py:228), with each component's OWN axis — the axis whose
#: HALF-INTEGER coefficient pair the tail reads (MEEP's ``dsigw``).
E_TERMS: Tuple[Tuple[str, str, int], ...] = (
    ("Ex", "Dx", 0), ("Ey", "Dy", 1), ("Ez", "Dz", 2))

#: The coupling partners of each component, in MEEP's ``cycle_direction``
#: order X -> Y -> Z (vec.hpp:586; stepping.py:1235-1237): own axis + 1 first,
#: own axis + 2 second. Ez therefore takes the Ex-partner volume then the
#: Ey-partner volume. Unlike the nonlinear family's commutative ``Dsqr``, here
#: the order BINDS COEFFICIENTS TO PARTNERS — the plan fills the six row slots
#: by this table, and the gate's m3 mutation mispairs them.
TRANSVERSE_PARTNERS: Tuple[Tuple[int, int], ...] = ((1, 2), (2, 0), (0, 1))

#: The six coefficient slots, in plan/kernel argument order: for each row
#: component (E_TERMS order), the offset-1 partner then the offset-2 partner.
#: Keys are ``fields._chi1inv_offdiagonal``'s own (row, partner) pairs
#: (fields.py:544-547).
ROW_SLOTS: Tuple[Tuple[str, str], ...] = (
    ("Ex", "Ey"), ("Ex", "Ez"),
    ("Ey", "Ez"), ("Ey", "Ex"),
    ("Ez", "Ex"), ("Ez", "Ey"))

#: Per component, the axes on which its Yee shift is 0 — the axes whose
#: metallic wall plane the coupling mask zeroes (``_mask_metallic_wall_coupling``
#: loops axes ascending, stepping.py:1279-1283; ``IYEE_SHIFTS``, fields.py:
#: 214-219: Ex (1,0,0), Ey (0,1,0), Ez (0,0,1)). A test pins this table
#: against ``fields.IYEE_SHIFTS`` so the two cannot drift.
WALL_MASK_AXES: Tuple[Tuple[int, int], ...] = ((1, 2), (0, 2), (0, 1))

#: The Yee sub-lattice this side reads: half-integer, ``kps_a_h``/``kms_a_h``
#: (stepping.py:1015 via ``_constitutive_coefficients(..., half_integer=True)``).
HALF_INTEGER = True

#: Elements per program — restated from ``kernels.DEFAULT_BLOCK`` (which needs
#: Triton to import); a test pins the two against the source.
DEFAULT_BLOCK = 256

#: The shared clause builders the predicate composes from, named as data so the
#: laptop test can assert every one still exists in the other session's file.
SHARED_CLAUSES: Tuple[str, ...] = (
    "_grid_reasons", "_susceptibility_reasons", "_boundary_kinds",
    "_layout_reasons", "_inverse_epsilon_reasons", "_coefficient_reasons",
    "_volume_reasons", "_call")


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
    def _offdiag_term(g, u, o_c, o_d, o_u, o_ud, v_c, v_d, v_u, v_ud):
        """One partner's OFFDIAG term — MEEP step_generic.cpp:582-583 as
        ``stepping._offdiagonal_terms`` (:1214-1220) associates it::

            0.25*((g[i] + g[i-sx])*u[i] + (g[i+s] + g[(i+s)-sx])*u[i+s])

        The coefficient multiply sits BETWEEN the shifts: the near pair takes
        ``u`` at its own node, the far pair (the shifted PRODUCT's value) takes
        ``u`` at the next node up the component's own axis — loading ``u`` at
        the composed up index is the same bits as shifting the formed product,
        because a shift is data movement and the coefficient wraps plainly
        (phase-free) on a periodic axis while a metallic far face zeroes the
        whole product through the masks. 0.25 scales the sum, applied last.
        No unary minus (platform fact: ``semantic.minus`` canonicalizes
        signed-zero addends).
        """
        near = (tl.load(g + o_c, mask=v_c, other=0.0)
                + tl.load(g + o_d, mask=v_d, other=0.0))
        far = (tl.load(g + o_u, mask=v_u, other=0.0)
               + tl.load(g + o_ud, mask=v_ud, other=0.0))
        return 0.25 * ((near * tl.load(u + o_c, mask=v_c, other=0.0))
                       + (far * tl.load(u + o_u, mask=v_u, other=0.0)))

    @triton.jit
    def _masked_row_sum(diag, total, at_a, at_b,
                        WMA: tl.constexpr, WMB: tl.constexpr):
        """Wall-mask the coupling, THEN form the row sum — the array path's
        order (``_mask_metallic_wall_coupling`` at stepping.py:1252 runs before
        the ``constitutive + coupling`` add at :1007-1008). ``at_a``/``at_b`` are
        the face-0 predicates of the component's two Yee-shift-0 axes, in
        ascending axis order (the mask's own loop order); only face 0 is
        zeroed — the high wall is the shift-up zero ghost."""
        if WMA:
            total = tl.where(at_a, 0.0, total)
        if WMB:
            total = tl.where(at_b, 0.0, total)
        return diag + total

    @triton.jit
    def offdiag_constitutive_step(
        f0, f1, f2,                     # targets:      Ex, Ey, Ez             (in/out)
        w0, w1, w2,                     # auxiliaries:  f_w_Ex, f_w_Ey, f_w_Ez (in/out)
        g0, g1, g2,                     # sources:      Dx, Dy, Dz             (read-only)
        e0, e1, e2,                     # inverse-epsilon VOLUMES (three distinct or aliased)
        u01, u02,                       # row Ex: partner Ey (down y), partner Ez (down z)
        u11, u12,                       # row Ey: partner Ez (down z), partner Ex (down x)
        u21, u22,                       # row Ez: partner Ex (down x), partner Ey (down y)
        kp0, km0, kp1, km1, kp2, km2,   # kps/kms on each component's OWN axis, half-integer
        nx, ny, nz, n_elem,
        R01: tl.constexpr, R02: tl.constexpr,
        R11: tl.constexpr, R12: tl.constexpr,
        R21: tl.constexpr, R22: tl.constexpr,
        BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
        WM_X: tl.constexpr, WM_Y: tl.constexpr, WM_Z: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """``stepping.update_E`` under PML with off-diagonal chi1inv rows.

        The body is :func:`kernels.constitutive_step`'s E arm with ONE
        substitution per coupled component: ``src`` is no longer ``D*inv_eps``
        but ``(D*inv_eps) + total`` with ``total`` the wall-masked OFFDIAG
        accumulation. Everything else — the prev-before-store ordering, the two
        separate accumulations, the own-axis half-integer coefficient index —
        is that kernel's, unchanged, because the array path's is unchanged.

        ``R*`` are the per-component ROW MASK constexprs (offset-1 and offset-2
        surviving-partner flags), giving each component the four arms
        none/first/second/both; the 'none' arm is the certified plain body
        verbatim (docstring point 6). ``BC*`` carry the per-axis ghost rule for
        the coupling's neighbour loads. ``WM_*`` carry the metallic wall-mask
        declaration — the SAME question ``_mask_metallic_wall_coupling`` asks
        (``is_metallic and not is_mirrored``), deliberately NOT the ``BC*``
        ghost codes, exactly as the fused pairs keep ``ZM_*`` separate from
        ``BC*``. Coefficient pointers of dead slots are bound by the plan to
        the component's own D pointer and never read.
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

        # Wall-plane predicates for the coupling mask (face 0 only).
        at_x, at_y, at_z = i == 0, j == 0, k == 0

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
        us0 = tl.load(e0 + idx, mask=live, other=0.0)
        if R01:
            total0 = _offdiag_term(
                g1, u01, idx,
                i * nyz + dj * nz + k,
                ui * nyz + j * nz + k,
                ui * nyz + dj * nz + k,
                live, dvy, uvx, uvx & dvy)
            if R02:
                total0 = total0 + _offdiag_term(
                    g2, u02, idx,
                    i * nyz + j * nz + dk,
                    ui * nyz + j * nz + k,
                    ui * nyz + j * nz + dk,
                    live, dvz, uvx, uvx & dvz)
            src0 = _masked_row_sum(gs0 * us0, total0, at_y, at_z, WM_Y, WM_Z)
        else:
            if R02:
                total0 = _offdiag_term(
                    g2, u02, idx,
                    i * nyz + j * nz + dk,
                    ui * nyz + j * nz + k,
                    ui * nyz + j * nz + dk,
                    live, dvz, uvx, uvx & dvz)
                src0 = _masked_row_sum(gs0 * us0, total0, at_y, at_z,
                                       WM_Y, WM_Z)
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
        us1 = tl.load(e1 + idx, mask=live, other=0.0)
        if R11:
            total1 = _offdiag_term(
                g2, u11, idx,
                i * nyz + j * nz + dk,
                i * nyz + uj * nz + k,
                i * nyz + uj * nz + dk,
                live, dvz, uvy, uvy & dvz)
            if R12:
                total1 = total1 + _offdiag_term(
                    g0, u12, idx,
                    di * nyz + j * nz + k,
                    i * nyz + uj * nz + k,
                    di * nyz + uj * nz + k,
                    live, dvx, uvy, uvy & dvx)
            src1 = _masked_row_sum(gs1 * us1, total1, at_x, at_z, WM_X, WM_Z)
        else:
            if R12:
                total1 = _offdiag_term(
                    g0, u12, idx,
                    di * nyz + j * nz + k,
                    i * nyz + uj * nz + k,
                    di * nyz + uj * nz + k,
                    live, dvx, uvy, uvy & dvx)
                src1 = _masked_row_sum(gs1 * us1, total1, at_x, at_z,
                                       WM_X, WM_Z)
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
        us2 = tl.load(e2 + idx, mask=live, other=0.0)
        if R21:
            total2 = _offdiag_term(
                g0, u21, idx,
                di * nyz + j * nz + k,
                i * nyz + j * nz + uk,
                di * nyz + j * nz + uk,
                live, dvx, uvz, uvz & dvx)
            if R22:
                total2 = total2 + _offdiag_term(
                    g1, u22, idx,
                    i * nyz + dj * nz + k,
                    i * nyz + j * nz + uk,
                    i * nyz + dj * nz + uk,
                    live, dvy, uvz, uvz & dvy)
            src2 = _masked_row_sum(gs2 * us2, total2, at_x, at_y, WM_X, WM_Y)
        else:
            if R22:
                total2 = _offdiag_term(
                    g1, u22, idx,
                    i * nyz + dj * nz + k,
                    i * nyz + j * nz + uk,
                    i * nyz + dj * nz + uk,
                    live, dvy, uvz, uvz & dvy)
                src2 = _masked_row_sum(gs2 * us2, total2, at_x, at_y,
                                       WM_X, WM_Y)
            else:
                src2 = gs2 * us2
        tl.store(w2 + idx, src2, mask=live)
        a2 = tl.load(f2 + idx, mask=live, other=0.0)
        a2 = a2 + kp_2 * src2
        a2 = a2 - km_2 * prev2
        tl.store(f2 + idx, a2, mask=live)

else:  # pragma: no cover - the laptop path
    offdiag_constitutive_step = None  # type: ignore[assignment]


def offdiag_constitutive_step_kernel() -> Any:
    """The shipped ``@triton.jit`` kernel, or a diagnosable ImportError."""
    if offdiag_constitutive_step is None:
        raise ImportError(
            "the off-diagonal update_E kernel needs the optional `triton` "
            "package (pip install triton). The engine runs without it; only "
            f"this fast path is unavailable. Original error: "
            f"{_TRITON_IMPORT_ERROR}")
    return offdiag_constitutive_step


# ---------------------------------------------------------------------------
# Small host helpers
# ---------------------------------------------------------------------------

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


def wall_mask_axes(grid: Any) -> Tuple[int, int, int]:
    """The ``WM_X``/``WM_Y``/``WM_Z`` constexprs, decided the way the mask decides.

    ``_mask_metallic_wall_coupling`` (stepping.py:1279-1283) asks, per axis,
    ``grid.is_metallic(axis) and not grid.is_mirrored(axis)`` — the grid's own
    DECLARATION, deliberately not ``_boundary_kinds``' resolved ghost rule
    (the same split the fused pairs keep between ``ZM_*`` and ``BC*``). One
    function so the predicate-side reasoning and the compile-time choice
    cannot disagree; folds are refused by the predicate so the mirrored arm is
    belt and braces here.
    """
    return tuple(  # type: ignore[return-value]
        int(bool(_coverage._call(grid, "is_metallic", axis, default=False))
            and not bool(_coverage._call(grid, "is_mirrored", axis,
                                         default=False)))
        for axis in range(3))


def row_volumes_for(fields: Any) -> Tuple[Optional[Any], ...]:
    """The six coefficient slots in :data:`ROW_SLOTS` order, ``None`` where dead.

    The single place the slot binding is derived from the engine's own rows
    (``Fields.chi1inv_offdiagonal_for``, fields.py:1317-1319), so the
    predicate, the plan and the gate cannot disagree about which coefficient
    volume pairs with which partner term — the mispairing is the gate's m3
    mutation.
    """
    return tuple(
        _coverage._call(fields, "chi1inv_offdiagonal_for", row,
                        default={}).get(partner)
        for row, partner in ROW_SLOTS)


# ---------------------------------------------------------------------------
# The coverage predicate
# ---------------------------------------------------------------------------

def _row_reasons(fields: Any, shape: Sequence[int]) -> List[str]:
    """Every surviving row volume must be a real f32 C-contiguous grid volume
    that aliases none of this sub-step's outputs.

    The installer validates the FORM of all of this (fields.py:1262-1310), and
    the clauses are written anyway: this package's own rule forbids inferring
    coverage from another module's guard, and a row planted past the installer
    must be refused by name, not stepped. The OUTPUT-ALIAS clause is beyond
    even the installer's reach: ``set_epsilon_volumes`` keeps the caller's
    array without copying (``asarray`` + ``astype(copy=False)``,
    fields.py:1296), so a row that IS one of the E/f_w outputs arrives here
    legally installed — and the plan builder refuses that alias with a raise,
    so the predicate must refuse it FIRST to keep the builder's
    None-means-refused contract (a covered verdict must never meet a
    ValueError)."""
    out: List[str] = []
    outputs: Dict[int, str] = {}
    for term in E_TERMS:
        for name in (term[0], "f_w_" + term[0]):
            address = _base_address(getattr(fields, name, None))
            if address is not None:
                outputs.setdefault(address, name)
    for row, partner in ROW_SLOTS:
        entries = _coverage._call(fields, "chi1inv_offdiagonal_for", row,
                                  default={}) or {}
        value = entries.get(partner)
        if value is None:
            continue
        label = f"chi1inv_offdiag[{row}][{partner}]"
        if not _is_volume(value):
            out.append(f"{label} is not a volume")
            continue
        if str(getattr(value, "dtype", ""))[0] == "c":
            out.append(f"{label} is complex; the inverse-permittivity tensor "
                       f"of a lossless medium is real")
            continue
        address = _base_address(value)
        if address is not None and address in outputs:
            out.append(f"{label} aliases output {outputs[address]}: the "
                       f"coupling would read the volume while the sub-step "
                       f"writes it, and the plan builder refuses the alias — "
                       f"the predicate must refuse it first")
        if len(shape) == 3:
            out.extend(_coverage._volume_reasons(label, value, shape))
    return out


def offdiag_constitutive_coverage(fields: Any, pml: Any) -> "_coverage.Coverage":
    """May the Triton off-diagonal kernel step ``update_E`` for this pair?

    POSITIVE CLAUSES ONLY; a failing clause appends its reason and the scan
    continues. The grid clauses are ``coverage._grid_reasons``' — called
    DIRECTLY, not restated, because unlike the nonlinear and beta families
    this predicate inverts NO shared clause: chi2/chi3 stays refused (the
    row-product-times-Pade most-general case is a later fused leg), beta stays
    refused (the engine itself additionally raises for real+offdiag+beta,
    stepping.py:800-810), and everything else is wanted verbatim. The offdiag
    INVERSION lives in the E-side clauses below, exactly where the certified
    predicate's refusal lives (coverage.py:367-370):

    a. **no registered polarization.** With poles the source is ``D - sum P``
       formed in per-component scratch buffers (fields.py:1107-1138) and the
       coupling would read THOSE, not D — that configuration belongs to a
       later fused leg with the dispersive family.
    b. **at least one surviving off-diagonal row (INVERTED), counted over the
       six ROW_SLOTS** — deliberately not the bare ``has_offdiagonal_epsilon``
       flag, which a row planted past the installer under a diagonal key sets
       while every slot is dead (the builder would raise where this predicate
       had admitted). A zero-slot run is bit-identically the diagonal
       engine's (the install-time drop, fields.py:1302-1303) and belongs to
       ``coverage.constitutive_coverage(side="E")`` — the two predicates are
       disjoint by construction and a test pins it.
    c. **stored E** — forced True by any surviving row (fields.py:1254-1255);
       belt and braces with a reason.
    d. every surviving row volume real, f32, C-contiguous, grid-shape, and
       aliasing no E/f_w output (the public installer keeps the caller's
       array, so the alias is reachable without planting).
    e. volume inverse epsilon per component (the shared clause,
       :func:`coverage._inverse_epsilon_reasons`) — this kernel carries NO
       scalar arm at all.
    f. layout and the half-integer coefficient tables, via the shared helpers.

    Conductivity is NOT a clause, deliberately: it changes the CURL sub-steps
    only, never the constitutive one (coverage.py:128-130).
    """
    grid = getattr(fields, "grid", None)
    if grid is None:
        return _coverage.Coverage(False, ("fields carries no grid",))

    reasons: List[str] = _coverage._grid_reasons(fields, pml, grid)
    reasons.extend(_coverage._susceptibility_reasons(fields))

    # (a) No poles: the coupling must read the aliased D primaries.
    states = tuple(getattr(fields, "polarizations", ()) or ())
    if states or getattr(fields, "has_polarizations", False):
        reasons.append(
            "a susceptibility is registered: the offdiag source becomes "
            "D - sum P in per-component scratch buffers (fields.py:1107-1138) "
            "— the fused dispersive+offdiag kernel is a later leg")

    # (b) INVERTED: at least one surviving row SLOT, or the run is the
    #     certified plain kernel's. Counted over ROW_SLOTS, deliberately not
    #     the bare has_offdiagonal_epsilon flag: a row planted past the
    #     installer under a diagonal key sets the flag with every slot dead,
    #     and the builder would raise ("no row slot survives") on a run this
    #     clause had admitted — None-means-refused requires the predicate to
    #     refuse first.
    if not any(value is not None for value in row_volumes_for(fields)):
        reasons.append(
            "no off-diagonal chi1inv row survived installation: that "
            "configuration is constitutive_coverage(side='E')'s and this "
            "predicate must not overlap it")

    # (c) Stored E.
    if not getattr(fields, "stores_E", False):
        reasons.append("E is recomputed from D rather than stored")

    # (d) The surviving rows, readable and well-formed.
    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(_row_reasons(fields, shape))

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


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class OffdiagConstitutivePlan:
    """A launchable, allocation-free off-diagonal ``update_E``.

    The Yee sub-lattice is chosen at the BUILDERS and nowhere else:
    ``kps_a_h``/``kms_a_h``, the half-integer tables (stepping.py:1015). The
    kernel takes six coefficient pointers and never asks which lattice they
    came from — a swap is the silent half-cell absorber error the certified
    family's gate carries a mutation for.

    The six ROW SLOTS are bound in :data:`ROW_SLOTS` order; a dead slot's
    pointer is bound to the component's own D pointer (never read — the
    constexpr arm is what stops the read, the pointer still has to type). The
    WALL AXES triple is the mask's own question (:func:`wall_mask_axes`),
    carried separately from the boundary codes exactly as the fused pairs
    carry ``ZM_*`` beside ``BC*``.

    ``__init__`` REFUSES ALIASING between the outputs (E, f_w) and any input
    (D, inverse epsilon, row volumes, the six coefficient vectors), and among
    the outputs themselves: the coupling re-reads the partner D volumes at
    neighbour offsets while E and f_w are being written, so an aliased pair
    would make the result depend on block schedule — a wrong answer that
    varies run to run. An all-dead row mask is refused toward the certified
    plain kernel (the install-time drop makes the two families disjoint).
    """

    __slots__ = ("shape", "n_elem", "block", "row_mask", "boundary_codes",
                 "wall_axes", "_targets", "_aux", "_sources", "_inv_eps",
                 "_rows", "_coefficients", "_grid", "_kernel")

    def __init__(self, shape, block: int, targets, auxiliaries, sources,
                 inverse_epsilon, rows, coefficients, boundary_codes,
                 wall_axes, kernel: Any = None) -> None:
        from .launch import CupyPointer, _flat  # noqa: PLC0415

        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.block = int(block)
        self.boundary_codes = tuple(int(code) for code in boundary_codes)
        if len(self.boundary_codes) != 3 or any(
                code not in (0, 1) for code in self.boundary_codes):
            raise ValueError(f"boundary codes must be three of {{0, 1}}, got "
                             f"{boundary_codes!r}")
        self.wall_axes = tuple(int(flag) for flag in wall_axes)
        if len(self.wall_axes) != 3 or any(
                flag not in (0, 1) for flag in self.wall_axes):
            raise ValueError(f"wall axes must be three of {{0, 1}}, got "
                             f"{wall_axes!r}")

        targets = tuple(targets)
        auxiliaries = tuple(auxiliaries)
        sources = tuple(sources)
        inverse_epsilon = tuple(inverse_epsilon)
        rows = tuple(rows)
        if not (len(targets) == len(auxiliaries) == len(sources)
                == len(inverse_epsilon) == 3):
            raise ValueError("a plan carries exactly three components")
        if len(rows) != len(ROW_SLOTS):
            raise ValueError(f"rows must fill the {len(ROW_SLOTS)} slots of "
                             f"ROW_SLOTS (None where dead), got {len(rows)}")
        for index, value in enumerate(inverse_epsilon):
            if not _is_volume(value):
                raise ValueError(
                    f"inverse epsilon {index} is not a volume: this family "
                    f"carries no scalar arm (the certified predicate's own "
                    f"clause, coverage.py:649-674)")
        self.row_mask = tuple(int(value is not None) for value in rows)
        if not any(self.row_mask):
            raise ValueError(
                "no row slot survives: that configuration belongs to the "
                "certified plain constitutive kernel, and building this plan "
                "for it would overlap the two")

        self._require_no_aliasing(targets, auxiliaries, sources,
                                  inverse_epsilon, rows, coefficients)

        self._targets = tuple(CupyPointer(a) for a in targets)
        self._aux = tuple(CupyPointer(a) for a in auxiliaries)
        self._sources = tuple(CupyPointer(a) for a in sources)
        self._inv_eps = tuple(CupyPointer(a) for a in inverse_epsilon)
        # Dead slots bind the row component's own D pointer; the constexpr arm
        # stops the read but the pointer argument still has to type.
        component_of_slot = tuple(
            next(index for index, term in enumerate(E_TERMS)
                 if term[0] == row)
            for row, _partner in ROW_SLOTS)
        self._rows = tuple(
            CupyPointer(value) if value is not None
            else self._sources[component_of_slot[index]]
            for index, value in enumerate(rows))
        self._coefficients = tuple(CupyPointer(_flat(a)) for a in coefficients)
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        self._kernel = kernel

    @staticmethod
    def _require_no_aliasing(targets, auxiliaries, sources, inverse_epsilon,
                             rows, coefficients) -> None:
        """Base-address disjointness of outputs vs outputs and inputs vs
        outputs — the six coefficient vectors included in the input inventory.
        Equality of BASE addresses only: two overlapping views with different
        bases pass unseen, the certified plans' shared (accepted) limitation.
        Inputs may alias EACH OTHER freely (three aliased inverse-epsilon
        volumes are the isotropic install)."""
        outputs = {}
        for name_group, group in (("target", targets), ("aux", auxiliaries)):
            for index, array in enumerate(group):
                address = _base_address(array)
                if address is None:
                    raise ValueError(
                        f"{name_group} {index} exposes no readable base "
                        f"address; an unverifiable output is not accepted")
                if address in outputs:
                    raise ValueError(
                        f"{name_group} {index} aliases {outputs[address]}; "
                        f"the outputs must be distinct arrays")
                outputs[address] = f"{name_group} {index}"
        inputs = list(sources) + list(inverse_epsilon)
        inputs += [value for value in rows if value is not None]
        inputs += [value for value in coefficients if _is_volume(value)]
        for array in inputs:
            address = _base_address(array)
            if address is not None and address in outputs:
                raise ValueError(
                    f"an input volume aliases {outputs[address]}: the "
                    f"coupling re-reads the partner D volumes at neighbour "
                    f"offsets while the outputs are written, so an alias "
                    f"makes the answer depend on block schedule")

    def run(self, guard: Optional[bool] = None) -> None:
        """Launch the sub-step, in place. Same ``guard`` contract as every plan."""
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        kernel = (self._kernel if self._kernel is not None
                  else offdiag_constitutive_step_kernel())
        nx, ny, nz = self.shape
        kernel[self._grid](
            *self._targets, *self._aux, *self._sources, *self._inv_eps,
            *self._rows, *self._coefficients,
            nx, ny, nz, self.n_elem,
            R01=self.row_mask[0], R02=self.row_mask[1],
            R11=self.row_mask[2], R12=self.row_mask[3],
            R21=self.row_mask[4], R22=self.row_mask[5],
            BCX=self.boundary_codes[0], BCY=self.boundary_codes[1],
            BCZ=self.boundary_codes[2],
            WM_X=self.wall_axes[0], WM_Y=self.wall_axes[1],
            WM_Z=self.wall_axes[2],
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
        )

    def __repr__(self) -> str:
        return (f"OffdiagConstitutivePlan(shape={self.shape}, "
                f"rows={self.row_mask}, bc={self.boundary_codes}, "
                f"walls={self.wall_axes}, block={self.block})")


#: Boundary kind string -> the kernel's constexpr code. The strings are
#: ``stepping``'s own; the codes are the package's (kernels.py:77-78).
BOUNDARY_CODES = {"periodic": 0, "metallic": 1}


def plan_offdiagonal_constitutive(fields: Any, pml: Any,
                                  block: Optional[int] = None
                                  ) -> Optional[OffdiagConstitutivePlan]:
    """Build the off-diagonal ``update_E`` plan from the engine's own objects.

    None means REFUSED, and the reasons are available from
    :func:`offdiag_constitutive_coverage`. The sources are taken through
    ``displacement_minus_polarization_volumes`` — the accessor the array path
    itself reads (stepping.py:996, :1004) — which with no poles admitted hands
    back the aliased D primaries (fields.py:1107-1138). The Triton import
    stays below the predicate: a NumPy host must be able to plan (to ``None``)
    without the optional dependency being importable at all.
    """
    if not offdiag_constitutive_coverage(fields, pml).covered:
        return None
    kinds = _coverage._boundary_kinds(fields.grid, pml)
    codes = tuple(BOUNDARY_CODES[kind] for kind in kinds)
    targets = tuple(term[0] for term in E_TERMS)
    volumes = fields.displacement_minus_polarization_volumes()
    return OffdiagConstitutivePlan(
        fields.grid.shape, DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in targets],
        [getattr(fields, "f_w_" + name) for name in targets],
        [volumes[name] for name in targets],
        [fields.inverse_epsilon_for(name) for name in targets],
        row_volumes_for(fields),
        [getattr(pml, f"{stem}_{axis}_h")
         for axis in "xyz" for stem in ("kps", "kms")],
        codes,
        wall_mask_axes(fields.grid),
    )


def plan_offdiagonal_constitutive_from_arrays(
        arrays: Dict[str, Any], flat: Dict[str, Any],
        rows: Dict[str, Dict[str, Any]],
        boundary_codes: Sequence[int], wall_axes: Sequence[int],
        block: Optional[int] = None, kernel: Any = None
        ) -> OffdiagConstitutivePlan:
    """Build it from bare arrays — the gate's and probe's route.

    ``arrays`` is keyed by component name (``Ex``, ``f_w_Ex``, ``Dx``,
    ``inv_eps_Ex``), ``flat`` by ``kps_x``/``kms_x``/... on the sub-lattice
    THE CALLER already selected, and ``rows`` maps row component -> partner
    component -> coefficient volume, with absent entries marking dead slots
    (the caller performs the install-time zero-row drop itself; an explicitly
    present all-zero volume is treated as LIVE, which is deliberate — the
    array path never sees one, so a harness that wants the degenerate
    byte-equality case must drop it the way the installer does). No coverage
    predicate runs here: the caller is a harness that constructed the
    configuration on purpose.

    ``kernel=`` IS LOAD-BEARING: the mutation legs route a mutated kernel
    through it, and a builder that drops it launches the shipped kernel and
    reports a pass for a defect it never introduced (the measured
    harness-disarm failure the certified families document).
    """
    targets = tuple(term[0] for term in E_TERMS)
    shape = tuple(int(n) for n in arrays[targets[0]].shape)
    return OffdiagConstitutivePlan(
        shape, DEFAULT_BLOCK if block is None else block,
        [arrays[name] for name in targets],
        [arrays["f_w_" + name] for name in targets],
        [arrays[term[1]] for term in E_TERMS],
        [arrays["inv_eps_" + name] for name in targets],
        [(rows.get(row) or {}).get(partner) for row, partner in ROW_SLOTS],
        [flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kps", "kms")],
        boundary_codes,
        wall_axes,
        kernel=kernel,
    )


# ---------------------------------------------------------------------------
# INTEGRATION — what the later wiring round has to do, and nothing more
# ---------------------------------------------------------------------------
#
# Additive edits, once the shared files are free (owned by the concurrent
# session and NOT touched from here):
#
#   coverage.py  constitutive_coverage(side="E")'s offdiag clause
#                (coverage.py:367-370) stays — it is the DISJOINTNESS seam,
#                not a stale refusal. plan_step gains one branch: where that
#                clause is the ONLY E-side refusal and
#                offdiag_constitutive_coverage admits, install
#                plan_offdiagonal_constitutive in the update_E slot. The
#                composition probe's whole-step byte identity — certified
#                curls and update_H composing UNCHANGED over the new
#                update_E — is the measurement that licenses the branch.
#   __init__.py  export the kernel accessor, the predicate and the builders.
#   fingerprints.json  add this module and kernel, re-cut against the bytes
#                the gate certified.
#   stepping.py / fields.py / driver.py  fix the stale fold-refusal
#                docstrings (stepping.py:1219-1220, fields.py:1237-1241,
#                driver.py:1202-1204) — the install does NOT refuse folded
#                rows; the code and the fold-equivalence test are the truth.
#
# Dispatch stays disabled throughout: the engine's fast-path hook returns None
# on every branch and nothing in this tranche changes that.
