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
sub-step of an offdiag run, not the first. ``coverage.py`` is a shared module
and is imported, never edited; ``fingerprints.json`` carries no entry for
this module — the byte gate binds its own provenance record inside its results
directory. Callers are the gate (``parity/meep_gpu/gate_triton_offdiag.py``),
the composition probe (``probe_triton_offdiag_composition.py``) and the laptop
tests, nothing else.

WHAT IT REPLACES. ``stepping.update_E`` with an active PML,
``fields.has_offdiagonal_epsilon`` True and NOT ``fields.has_nonlinearity`` -- the
``elif offdiagonal:`` branch. With no poles admitted,
``displacement_minus_polarization_volumes`` aliases each source to its D primary,
and all three are alive at once because the coupling reads the OTHER components'
volumes. Per component ``c`` with own axis ``a`` (``stepping._offdiagonal_terms``,
``_rounded_row``, ``_apply_constitutive_regions``; MEEP step_generic.cpp:568-611)::

    row = double(D_c * us_c)                                # one float32 product, widened
    per surviving partner (offset 1 then 2, cycle X->Y->Z):
        pair    = g + shift_down(g, partner_axis)
        product = pair * coefficient
        summed  = product + shift_up(product, a)            # float32
        summed  = 0 at face 0 of each metallic, non-mirrored axis with iyee 0
        row     = row + 0.25 * double(summed)               # only where the slot is present
    src = float(row)                                        # rounded once
    on the cells whose chunk carries the a-layer (half-integer membership):
        E_c = E_c + ((kps_a_h * src) - (kms_a_h * f_w_c));  f_w_c = src
    everywhere else:
        E_c = src;                                          f_w_c keeps its word

THE THINGS THAT DECIDE BIT-IDENTITY, each held by the gate rather than assumed:

1. **THE COEFFICIENT MULTIPLY SITS BETWEEN THE TWO SHIFTS**: ``u[i]`` multiplies
   the two-point partner average AT ITS OWN NODE and ``u[i+s]`` the average at the
   next node up -- MEEP registers the off-diagonal entry at the component's Yee
   site minus half a cell along its own axis, the integer node
   (anisotropic_averaging.cpp:248-257, ``here - shift1``). A plain
   four-point-average-times-``u[i]`` hoist is the same ALGEBRA only for a uniform
   coefficient -- and MEASURED bitwise-different even there, because
   distributivity (``a*u + b*u`` vs ``(a+b)*u``) is not a bitwise identity in f32
   (the laptop test pins the separation). The gate's spatially varying
   coefficient cases are what make the REGISTRATION -- not merely the rounding --
   byte-visible, and its m1 mutation is the hoist.
2. **THE TWO SHIFTS GO IN OPPOSITE DIRECTIONS** -- half a cell DOWN the partner's
   axis on the raw partner volume (``g[i-sx]``), then the already-multiplied
   product half a cell UP the component's own axis (``(pair*u)[i+s]``, which is
   where the corner ``g[(i+s)-sx]`` arises). Per-axis ghost rules compose
   independently on the corner. Taking both shifts the same way is the half-cell
   registration error; the gate's m2 mutation is exactly it.
3. **THE ROW IS MEEP'S DOUBLE ROW, ROUNDED ONCE.** ``0.25`` is a double literal in
   MEEP's ``OFFDIAG`` macro, so each term is the float32 sum widened and scaled in
   double (exact: a power of two), the terms are added to the widened ``gs*us`` in
   double, left to right (offset 1 then 2), and the store rounds once
   (``tl.cast`` to float64 and back; the narrowing is round-to-nearest-even). The
   previous float32 row (``0.25`` in float32, ``diag + total`` in float32) is the
   gate's b3 mutation.
4. **THE METALLIC WALL-COUPLING MASK** (``_mask_metallic_wall_coupling``): each
   term's float32 sum is zeroed at FACE 0 of every metallic non-mirrored axis on
   which the component's Yee shift is 0, BEFORE it is scaled and added -- the
   wall's tangential E is never stepped, the diagonal product is already zero
   there via ``zero_metal_D``, but the coupling reads live PARTNER volumes beside
   the wall. Only the low wall is stored; the high wall is the shift-up zero
   ghost. The gate's m6 (dropped) and m7 (over-applied) mutations hold both
   directions.
5. **PER-CHUNK PRESENCE.** MEEP deletes an off-diagonal ``chi1inv`` array in every
   chunk where it is identically zero over the chunk's samples and passes NULL for
   it there, so the term is dropped rather than a zero added
   (``Fields.offdiag_presence``; step_generic.cpp:568-583). That differs from
   adding one only where the row so far is ``-0``. The kernel reads the slot's
   presence per chunk of the ``{low, interior, high}`` product
   (:func:`presence_record`), from per-axis segment lines read off the division's
   own owner volume -- not from the membership record's runs, which are the
   layer's and need not be the chunk cuts. A slot no chunk drops compiles to no
   read at all (the ``P*`` constexprs).
6. **A COMPONENT WITH NO SURVIVING ROW KEEPS THE PLAIN PRODUCT** -- ``_offdiagonal_terms``
   returns ``None`` and the caller never forms a ``+ 0`` copy. The row-mask
   constexprs compile the 'none' arm to ``gs*us`` itself, and the gate's identity
   leg pins the rows-all-dropped build byte-identical to the root constitutive
   kernel.
7. **THE CONSTITUTIVE FORM IS PER CHUNK** (:func:`.forms.constitutive_form`, read
   from the record's half-integer line of the component's own axis): MEEP's
   accumulation ``E + ((kps*src) - (kms*f_w))`` with ``f_w = src`` on the cells
   whose chunk carries the layer, ``E = src`` elsewhere. ``f_w`` is read and
   written on the accumulation's cells only (the auxiliary contract), ``prev`` is
   read before ``f_w`` is written, and the plain form is an output select, so a
   ``-0`` source stays ``-0``. ``ENABLE_FP_FUSION = False`` keeps every product
   unfused. No unary minus appears on any coefficient path (the platform's
   ``semantic.minus`` lowers ``-x`` as ``0.0 - x`` and canonicalizes signed-zero
   addends).

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
sum — the gate measured the wrapped-ghost mutant (PTX-verified-different)
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
The doc fix belongs in those files.

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

from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

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

#: The presence vector (:func:`presence_record`) starts with one ``int32`` mask per
#: row slot, in :data:`ROW_SLOTS` order: bit ``c`` set when chunk ``c`` of the
#: ``{low, interior, high}`` product keeps that slot's ``chi1inv`` array. The
#: per-axis segment lines follow these words.
PRESENCE_MASK_WORDS = 6

#: The shared clause builders the predicate composes from, named as data so the
#: laptop test can assert every one still exists in the shared ``coverage.py``.
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
    #: Where the segment lines start in the presence vector (:data:`PRESENCE_MASK_WORDS`).
    PRESENCE_LINES = tl.constexpr(PRESENCE_MASK_WORDS)
    from .forms import constitutive_form, record_in  # noqa: E402
    from .kernels import RECORD_HALF_INTEGER  # noqa: E402
else:
    PERIODIC = 0  # type: ignore[assignment]
    METALLIC = 1  # type: ignore[assignment]


if triton is not None:

    @triton.jit
    def _offdiag_term(g, u, o_c, o_d, o_u, o_ud, v_c, v_d, v_u, v_ud):
        """One partner's OFFDIAG sum before its scale, in float32.

        MEEP step_generic.cpp:580-583 as ``stepping._offdiagonal_terms`` forms it::

            (g[i] + g[i-sx])*u[i] + (g[i+s] + g[(i+s)-sx])*u[i+s]

        The pair sums, the two products and their sum are ``realnum`` operations,
        each rounded to float32. The coefficient multiply sits BETWEEN the shifts:
        the near pair takes ``u`` at its own node, the far pair (the shifted
        PRODUCT's value) takes ``u`` at the next node up the component's own axis --
        loading ``u`` at the composed up index is the same bits as shifting the
        formed product, because a shift is data movement and the coefficient wraps
        plainly (phase-free) on a periodic axis while a metallic far face zeroes the
        whole product through the masks. The ``0.25`` is MEEP's double literal and
        is applied by :func:`_row_add`, in double. No unary minus (platform fact:
        ``semantic.minus`` canonicalizes signed-zero addends).
        """
        near = (tl.load(g + o_c, mask=v_c, other=0.0)
                + tl.load(g + o_d, mask=v_d, other=0.0))
        far = (tl.load(g + o_u, mask=v_u, other=0.0)
               + tl.load(g + o_ud, mask=v_ud, other=0.0))
        return ((near * tl.load(u + o_c, mask=v_c, other=0.0))
                + (far * tl.load(u + o_u, mask=v_u, other=0.0)))

    @triton.jit
    def _row_add(row, summed, at_a, at_b, op, chunk, SLOT: tl.constexpr,
                 WMA: tl.constexpr, WMB: tl.constexpr, PRESENT: tl.constexpr):
        """Add one OFFDIAG term to the double row, as ``stepping._offdiagonal_terms`` does.

        The wall mask zeroes the term's float32 sum first, at face 0 of the
        component's Yee-shift-0 axes that are metallic and not mirrored
        (``_mask_metallic_wall_coupling``; ``at_a``/``at_b`` in ascending axis order,
        the mask's own loop order; the high wall is the shift-up zero ghost). The
        sum is then widened exactly and scaled by the double ``0.25`` (exact: a power
        of two on a widened float32), and the term is added in double. ``PRESENT``
        set means the term is added only on the cells whose chunk keeps row slot
        ``SLOT``'s array: MEEP passes NULL for it in every chunk that deleted it and
        drops the term there rather than adding a zero (step_generic.cpp:568-583).
        Word ``SLOT`` of the presence vector holds one bit per chunk of the
        ``{low, interior, high}`` product, and ``chunk`` (0..26) is the cell's.
        """
        if WMA:
            summed = tl.where(at_a, 0.0, summed)
        if WMB:
            summed = tl.where(at_b, 0.0, summed)
        added = row + 0.25 * tl.cast(summed, tl.float64)
        if PRESENT:
            present = ((tl.load(op + SLOT) >> chunk) & 1) != 0
            added = tl.where(present, added, row)
        return added

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
        cb,                             # chunk record: each component's dsigw membership
        op,                             # off-diagonal presence: slot masks, segment lines
        nx, ny, nz, n_elem,
        R01: tl.constexpr, R02: tl.constexpr,
        R11: tl.constexpr, R12: tl.constexpr,
        R21: tl.constexpr, R22: tl.constexpr,
        P01: tl.constexpr, P02: tl.constexpr,
        P11: tl.constexpr, P12: tl.constexpr,
        P21: tl.constexpr, P22: tl.constexpr,
        PRESENCE: tl.constexpr,
        BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
        WM_X: tl.constexpr, WM_Y: tl.constexpr, WM_Z: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """``stepping.update_E`` under PML with off-diagonal chi1inv rows.

        The body is :func:`kernels.constitutive_step`'s E arm with ONE
        substitution per coupled component: ``src`` is no longer ``D*inv_eps`` but
        MEEP's row, ``gs*us`` widened to double plus each surviving OFFDIAG term in
        double (:func:`_row_add`), rounded once to float32 (``stepping._rounded_row``).
        The tail is that kernel's: the component's own (dsigw) membership line on
        the half-integer sub-lattice picks MEEP's accumulation or ``E = src``
        (:func:`.forms.constitutive_form`), and ``f_w`` is read and written on the
        accumulation's cells only.

        ``R*`` are the per-component ROW MASK constexprs (offset-1 and offset-2
        surviving-partner flags), giving each component the four arms
        none/first/second/both; the 'none' arm is the plain ``gs*us``, which is the
        array path's own product when no row survives. ``P*`` say which surviving
        slots are dropped by some chunk: such a slot's term is added where its
        presence bit is set (:func:`presence_record`), every other slot's
        everywhere; ``PRESENCE`` is any of them, and only then are the segment
        lines read. ``BC*`` carry the per-axis ghost rule for the coupling's
        neighbour loads. ``WM_*`` carry the metallic wall-mask declaration -- the
        SAME question ``_mask_metallic_wall_coupling`` asks (``is_metallic and not
        is_mirrored``), deliberately NOT the ``BC*`` ghost codes, exactly as the
        fused pairs keep ``ZM_*`` separate from ``BC*``. Coefficient pointers of
        dead slots are bound by the plan to the component's own D pointer and
        never read.
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

        # Which chunk of the {low, interior, high} product owns the cell, per
        # component lattice (presence_record's segment lines: integer x, y, z,
        # then half-integer x, y, z). Read only when some slot's presence varies.
        c0, c1, c2 = i * 0, i * 0, i * 0
        if PRESENCE:
            sxi = tl.load(op + PRESENCE_LINES + i, mask=live, other=0)
            syi = tl.load(op + PRESENCE_LINES + nx + j, mask=live, other=0)
            szi = tl.load(op + PRESENCE_LINES + nx + ny + k, mask=live, other=0)
            half = PRESENCE_LINES + nx + ny + nz
            sxh = tl.load(op + half + i, mask=live, other=0)
            syh = tl.load(op + half + nx + j, mask=live, other=0)
            szh = tl.load(op + half + nx + ny + k, mask=live, other=0)
            c0 = (sxh * 9 + syi * 3) + szi     # Ex: half-integer on x
            c1 = (sxi * 9 + syh * 3) + szi     # Ey: half-integer on y
            c2 = (sxi * 9 + syi * 3) + szh     # Ez: half-integer on z

        # Which cells take the accumulation: each component's own (dsigw) layer on
        # the half-integer sub-lattice (chunk_map.record_index(True, axis, 0)).
        dw0 = record_in(cb, RECORD_HALF_INTEGER + 0, i)
        dw1 = record_in(cb, RECORD_HALF_INTEGER + 4, j)
        dw2 = record_in(cb, RECORD_HALF_INTEGER + 8, k)

        # Component 0 takes its coefficient from axis x, 1 from y, 2 from z.
        kp_0 = tl.load(kp0 + i, mask=live, other=0.0)
        km_0 = tl.load(km0 + i, mask=live, other=0.0)
        kp_1 = tl.load(kp1 + j, mask=live, other=0.0)
        km_1 = tl.load(km1 + j, mask=live, other=0.0)
        kp_2 = tl.load(kp2 + k, mask=live, other=0.0)
        km_2 = tl.load(km2 + k, mask=live, other=0.0)

        # --- component 0: Ex — own axis x; partners Dy (down y) then Dz (down z)
        prev0 = tl.load(w0 + idx, mask=live & dw0, other=0.0)   # BEFORE the store.
        gs0 = tl.load(g0 + idx, mask=live, other=0.0)
        us0 = tl.load(e0 + idx, mask=live, other=0.0)
        if R01:
            row0 = tl.cast(gs0 * us0, tl.float64)
            row0 = _row_add(row0, _offdiag_term(
                g1, u01, idx,
                i * nyz + dj * nz + k,
                ui * nyz + j * nz + k,
                ui * nyz + dj * nz + k,
                live, dvy, uvx, uvx & dvy),
                at_y, at_z, op, c0, 0, WM_Y, WM_Z, P01)
            if R02:
                row0 = _row_add(row0, _offdiag_term(
                    g2, u02, idx,
                    i * nyz + j * nz + dk,
                    ui * nyz + j * nz + k,
                    ui * nyz + j * nz + dk,
                    live, dvz, uvx, uvx & dvz),
                    at_y, at_z, op, c0, 1, WM_Y, WM_Z, P02)
            src0 = tl.cast(row0, tl.float32)
        else:
            if R02:
                row0 = tl.cast(gs0 * us0, tl.float64)
                row0 = _row_add(row0, _offdiag_term(
                    g2, u02, idx,
                    i * nyz + j * nz + dk,
                    ui * nyz + j * nz + k,
                    ui * nyz + j * nz + dk,
                    live, dvz, uvx, uvx & dvz),
                    at_y, at_z, op, c0, 1, WM_Y, WM_Z, P02)
                src0 = tl.cast(row0, tl.float32)
            else:
                src0 = gs0 * us0
        tl.store(w0 + idx, src0, mask=live & dw0)
        a0 = tl.load(f0 + idx, mask=live & dw0, other=0.0)
        a0 = constitutive_form(a0, prev0, src0, kp_0, km_0, dw0)
        tl.store(f0 + idx, a0, mask=live)

        # --- component 1: Ey — own axis y; partners Dz (down z) then Dx (down x)
        prev1 = tl.load(w1 + idx, mask=live & dw1, other=0.0)
        gs1 = tl.load(g1 + idx, mask=live, other=0.0)
        us1 = tl.load(e1 + idx, mask=live, other=0.0)
        if R11:
            row1 = tl.cast(gs1 * us1, tl.float64)
            row1 = _row_add(row1, _offdiag_term(
                g2, u11, idx,
                i * nyz + j * nz + dk,
                i * nyz + uj * nz + k,
                i * nyz + uj * nz + dk,
                live, dvz, uvy, uvy & dvz),
                at_x, at_z, op, c1, 2, WM_X, WM_Z, P11)
            if R12:
                row1 = _row_add(row1, _offdiag_term(
                    g0, u12, idx,
                    di * nyz + j * nz + k,
                    i * nyz + uj * nz + k,
                    di * nyz + uj * nz + k,
                    live, dvx, uvy, uvy & dvx),
                    at_x, at_z, op, c1, 3, WM_X, WM_Z, P12)
            src1 = tl.cast(row1, tl.float32)
        else:
            if R12:
                row1 = tl.cast(gs1 * us1, tl.float64)
                row1 = _row_add(row1, _offdiag_term(
                    g0, u12, idx,
                    di * nyz + j * nz + k,
                    i * nyz + uj * nz + k,
                    di * nyz + uj * nz + k,
                    live, dvx, uvy, uvy & dvx),
                    at_x, at_z, op, c1, 3, WM_X, WM_Z, P12)
                src1 = tl.cast(row1, tl.float32)
            else:
                src1 = gs1 * us1
        tl.store(w1 + idx, src1, mask=live & dw1)
        a1 = tl.load(f1 + idx, mask=live & dw1, other=0.0)
        a1 = constitutive_form(a1, prev1, src1, kp_1, km_1, dw1)
        tl.store(f1 + idx, a1, mask=live)

        # --- component 2: Ez — own axis z; partners Dx (down x) then Dy (down y)
        prev2 = tl.load(w2 + idx, mask=live & dw2, other=0.0)
        gs2 = tl.load(g2 + idx, mask=live, other=0.0)
        us2 = tl.load(e2 + idx, mask=live, other=0.0)
        if R21:
            row2 = tl.cast(gs2 * us2, tl.float64)
            row2 = _row_add(row2, _offdiag_term(
                g0, u21, idx,
                di * nyz + j * nz + k,
                i * nyz + j * nz + uk,
                di * nyz + j * nz + uk,
                live, dvx, uvz, uvz & dvx),
                at_x, at_y, op, c2, 4, WM_X, WM_Y, P21)
            if R22:
                row2 = _row_add(row2, _offdiag_term(
                    g1, u22, idx,
                    i * nyz + dj * nz + k,
                    i * nyz + j * nz + uk,
                    i * nyz + dj * nz + uk,
                    live, dvy, uvz, uvz & dvy),
                    at_x, at_y, op, c2, 5, WM_X, WM_Y, P22)
            src2 = tl.cast(row2, tl.float32)
        else:
            if R22:
                row2 = tl.cast(gs2 * us2, tl.float64)
                row2 = _row_add(row2, _offdiag_term(
                    g1, u22, idx,
                    i * nyz + dj * nz + k,
                    i * nyz + j * nz + uk,
                    i * nyz + dj * nz + uk,
                    live, dvy, uvz, uvz & dvy),
                    at_x, at_y, op, c2, 5, WM_X, WM_Y, P22)
                src2 = tl.cast(row2, tl.float32)
            else:
                src2 = gs2 * us2
        tl.store(w2 + idx, src2, mask=live & dw2)
        a2 = tl.load(f2 + idx, mask=live & dw2, other=0.0)
        a2 = constitutive_form(a2, prev2, src2, kp_2, km_2, dw2)
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
# Off-diagonal presence: which chunks keep each row slot's array
# ---------------------------------------------------------------------------

class PresenceRecord(NamedTuple):
    """What the kernel reads to drop a term where MEEP dropped its array.

    ``vector`` is a host ``int32`` vector: :data:`PRESENCE_MASK_WORDS` slot masks
    (bit ``c`` set when chunk ``c`` of the ``{low, interior, high}`` product keeps
    the slot's array), then one segment line per (sub-lattice, axis) over the stored
    rows, integer x, y, z then half-integer x, y, z, each entry the segment (0, 1
    or 2) of the chunk that owns the row along that axis. ``varies`` holds, per
    slot, 1 when some chunk drops the slot's array (the kernel tests the bit) and 0
    when every chunk keeps it (the term is added everywhere, no read).
    """

    vector: Any
    varies: Tuple[int, ...]


#: Every slot present everywhere: the presence of a one-chunk division.
ALL_PRESENT = PresenceRecord(None, (0,) * 6)


def segment_lines(layout: Any) -> Dict[Tuple[int, int], Any]:
    """Per (sub-lattice, axis): the segment of the chunk that owns each stored row.

    The division must be a ``{low, interior, high}`` product along every resolved
    axis: each chunk cut spans the cross-section (``ChunkLayout.boundaries`` is not
    ``None``) and an axis has at most three segments. A chunk's segment along an
    axis is the position of its low face among the cuts. Each line is read off the
    division's own owner volume (``ChunkLayout.owner_volume``, the same per-cell
    owner the array path reads presence through) and the owner volume of every E
    lattice must be the product of its three lines; anything else is refused by
    name (:class:`~meep_gpu.chunk_map.ChunkLayoutRefusal`). The membership record's
    runs are not used: they are the layer's, and a division may cut an axis that
    carries no layer.
    """
    import bisect  # noqa: PLC0415

    import numpy  # noqa: PLC0415

    from ..chunk_map import AXIS_NAMES, ChunkLayoutRefusal  # noqa: PLC0415
    from ..fields import IYEE_SHIFTS  # noqa: PLC0415

    cuts = []
    for axis in range(3):
        cut = layout.boundaries(axis)
        if cut is None:
            raise ChunkLayoutRefusal(
                f"a chunk cut along {AXIS_NAMES[axis]} does not span the cross-section, so "
                f"the off-diagonal presence has no per-axis segment to index")
        if len(cut) - 1 > 3:
            raise ChunkLayoutRefusal(
                f"the division cuts {AXIS_NAMES[axis]} into {len(cut) - 1} segments; the "
                f"off-diagonal presence indexes at most three")
        cuts.append(tuple(int(v) for v in cut))
    triples = numpy.asarray(
        [[max(bisect.bisect_right(cuts[axis], int(lo[axis])) - 1, 0) for axis in range(3)]
         for lo, _hi in layout.boxes], dtype=numpy.int32)
    lines: Dict[Tuple[int, int], Any] = {}
    for _row, _source, own in E_TERMS:
        iyee = IYEE_SHIFTS["E" + "xyz"[own]]
        owned = triples[numpy.asarray(layout.owner_volume(iyee))]
        for axis in range(3):
            index = [0, 0, 0]
            index[axis] = slice(None)
            line = numpy.ascontiguousarray(owned[tuple(index) + (axis,)], dtype=numpy.int32)
            shape = [1, 1, 1]
            shape[axis] = line.size
            if not numpy.array_equal(owned[..., axis],
                                     numpy.broadcast_to(line.reshape(shape),
                                                        owned.shape[:3])):
                raise ChunkLayoutRefusal(
                    f"the owning chunk's {AXIS_NAMES[axis]} segment of an E{'xyz'[own]} cell "
                    f"depends on more than its {AXIS_NAMES[axis]} index")
            key = (int(iyee[axis]), axis)
            if key in lines and not numpy.array_equal(lines[key], line):
                raise ChunkLayoutRefusal(
                    f"two E components read different {AXIS_NAMES[axis]} segment lines on "
                    f"one sub-lattice")
            lines[key] = line
    return lines


def presence_record(layout: Any, volumes: Sequence[Any],
                    shape: Sequence[int]) -> PresenceRecord:
    """Pack per-slot presence volumes for the kernel, or refuse by name.

    ``volumes`` holds one entry per :data:`ROW_SLOTS` slot: ``None`` when the term
    is added on every cell, otherwise a host bool volume of ``shape``, the array
    path's own (``Fields.offdiag_presence``, read from ``layout``). Each slot's
    mask bit ``c`` is the presence on the cells whose owning chunk is ``c`` of the
    ``{low, interior, high}`` product (:func:`segment_lines`); presence that is not
    one value over such a cell class is refused. The kernel's decode then equals
    every given volume cell for cell, which this function checks before returning.
    """
    import numpy  # noqa: PLC0415

    from ..chunk_map import ChunkLayoutRefusal  # noqa: PLC0415

    extent = tuple(int(n) for n in shape)
    volumes = tuple(volumes)
    if len(volumes) != len(ROW_SLOTS):
        raise ValueError(f"presence needs {len(ROW_SLOTS)} slots (None where the term is "
                         f"added everywhere), got {len(volumes)}")
    if all(volume is None for volume in volumes):
        return ALL_PRESENT
    lines = segment_lines(layout)
    for (half, axis), line in lines.items():
        if line.size != extent[axis]:
            raise ChunkLayoutRefusal(
                f"the division's {'xyz'[axis]} segment line has {line.size} rows, the plan "
                f"stores {extent[axis]}")
    words = numpy.zeros(PRESENCE_MASK_WORDS, dtype=numpy.int64)
    varies = []
    component_of_slot = {row: own for row, _source, own in E_TERMS}
    for slot, ((row, partner), volume) in enumerate(zip(ROW_SLOTS, volumes)):
        if volume is None:
            varies.append(0)
            continue
        present = numpy.asarray(volume, dtype=bool)
        if present.shape != extent:
            raise ValueError(f"the presence of {row}<-{partner} has shape {present.shape}, "
                             f"the plan stores {extent}")
        own = component_of_slot[row]
        x, y, z = (lines[(int(axis == own), axis)] for axis in range(3))
        chunk = (x[:, None, None] * 9 + y[None, :, None] * 3) + z[None, None, :]
        word = 0
        for value in numpy.unique(chunk):
            cells = present[chunk == value]
            if cells.all():
                word |= 1 << int(value)
            elif cells.any():
                raise ChunkLayoutRefusal(
                    f"the presence of {row}<-{partner} is not one value over the cells of "
                    f"chunk {int(value)} of the low/interior/high product")
        decoded = ((word >> chunk) & 1).astype(bool)
        if not numpy.array_equal(decoded, present):
            raise ChunkLayoutRefusal(
                f"the packed presence of {row}<-{partner} does not reproduce its volume")
        words[slot] = word
        varies.append(1)
    vector = numpy.concatenate(
        [words.astype(numpy.int32)]
        + [lines[(half, axis)] for half in (0, 1) for axis in range(3)]).astype(numpy.int32)
    vector.setflags(write=False)
    return PresenceRecord(vector, tuple(varies))


def presence_for(fields: Any, pml: Any) -> PresenceRecord:
    """The presence ``stepping.update_E`` adds each row slot's term with, packed.

    Read as the array path reads it: the division is ``stepping._chunk_layout``
    (the lifted one when the driver installed it, the one-rank division under a
    layer otherwise), and each slot's volume is ``ChunkLayout.offdiag_presence`` of
    the coefficient the plan binds (:func:`row_volumes_for`) on it -- what
    ``Fields.offdiag_presence`` returns for a Cartesian grid (its one addition, the
    cylindrical axis row, belongs to a run this family refuses). Refused by name
    where the division is not a ``{low, interior, high}`` product the kernel can
    index (:func:`presence_record`).
    """
    import numpy  # noqa: PLC0415

    from .. import stepping  # noqa: PLC0415
    from ..fields import IYEE_SHIFTS  # noqa: PLC0415

    layout = stepping._chunk_layout(fields, pml)
    if layout is None:
        return ALL_PRESENT
    volumes = []
    for (row, _partner), coefficient in zip(ROW_SLOTS, row_volumes_for(fields)):
        mask = (None if coefficient is None
                else layout.offdiag_presence(coefficient, IYEE_SHIFTS[row]))
        volumes.append(None if mask is None else numpy.asarray(mask, dtype=bool))
    return presence_record(layout, volumes, tuple(fields.grid.shape))


def bind_presence(presence: Any, shape: Sequence[int], row_mask: Sequence[int],
                  like: Any, owner: str) -> Tuple[Tuple[int, ...], Any, Any]:
    """Check a :class:`PresenceRecord` against a plan and upload its vector once.

    Returns ``(varies, host vector, device pointer)``. ``presence`` is required:
    :data:`ALL_PRESENT` states that every term is added on every cell. Refused by
    name: anything but a record, a varying flag on a dead slot, and a vector that
    does not carry the plan's segment lines (``PRESENCE_MASK_WORDS + 2*(nx+ny+nz)``
    words with entries in 0..2) when a flag is set.
    """
    import numpy  # noqa: PLC0415

    from .launch import CupyPointer  # noqa: PLC0415

    # Read by its two fields rather than by class: a module reloaded in one process
    # (a test that re-imports it) holds a second PresenceRecord class.
    if presence is None or not (hasattr(presence, "vector") and hasattr(presence, "varies")):
        raise ValueError(
            f"{owner} was given no off-diagonal presence (PresenceRecord): pass "
            f"presence_for(fields, pml) or presence_record(...), or ALL_PRESENT for a "
            f"one-chunk division")
    varies = tuple(int(bool(flag)) for flag in presence.varies)
    if len(varies) != len(ROW_SLOTS):
        raise ValueError(f"{owner}: presence carries {len(varies)} slot flags")
    for slot, (flag, live) in enumerate(zip(varies, row_mask)):
        if flag and not live:
            raise ValueError(f"{owner}: presence varies on dead slot {ROW_SLOTS[slot]}")
    extent = tuple(int(n) for n in shape)
    if any(varies):
        host = numpy.asarray(presence.vector, dtype=numpy.int32)
        expected = PRESENCE_MASK_WORDS + 2 * sum(extent)
        if host.shape != (expected,):
            raise ValueError(f"{owner}: presence vector of shape {host.shape}, expected "
                             f"({expected},) for the plan's extent {extent}")
        lines = host[PRESENCE_MASK_WORDS:]
        if lines.size and (lines.min() < 0 or lines.max() > 2):
            raise ValueError(f"{owner}: a segment line leaves 0..2")
    else:
        host = numpy.zeros(PRESENCE_MASK_WORDS, dtype=numpy.int32)
    host = host.copy()
    host.setflags(write=False)
    if type(like).__module__.split(".")[0] == "cupy":  # pragma: no cover - device path
        import cupy  # noqa: PLC0415

        device = cupy.asarray(host)
    else:
        device = host
    return varies, host, CupyPointer(device)


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

    # (g) The per-chunk presence of every surviving row slot, packed for the kernel.
    if not reasons:
        reasons.extend(_presence_reasons(fields, pml))

    return _coverage.Coverage(not reasons, tuple(reasons))


def _presence_reasons(fields: Any, pml: Any) -> List[str]:
    """Why the per-chunk off-diagonal presence cannot be packed for the kernel, if so.

    MEEP drops a row slot's term in every chunk that deleted the slot's array
    (``Fields.offdiag_presence``). The kernel indexes that per chunk of a
    ``{low, interior, high}`` product (:func:`presence_for`); a division it cannot
    index is refused here, so the builder never raises on an admitted run.
    """
    try:
        presence_for(fields, pml)
    except Exception as exc:  # noqa: BLE001 - an unpackable presence is a refusal
        return [f"the off-diagonal presence cannot be packed for the kernel ({exc})"]
    return []


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
                 "wall_axes", "chunk_bounds", "presence", "presence_varies",
                 "_targets", "_aux", "_sources", "_inv_eps", "_rows",
                 "_coefficients", "_cb", "_op", "_grid", "_kernel")

    def __init__(self, shape, block: int, targets, auxiliaries, sources,
                 inverse_epsilon, rows, coefficients, boundary_codes,
                 wall_axes, kernel: Any = None, *, chunk_bounds: Any,
                 presence: Any) -> None:
        from .launch import CupyPointer, _flat, bind_chunk_bounds  # noqa: PLC0415

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
        self.chunk_bounds, self._cb = bind_chunk_bounds(
            chunk_bounds, self.shape, targets[0], "the off-diagonal update_E plan")
        self.presence_varies, self.presence, self._op = bind_presence(
            presence, self.shape, self.row_mask, targets[0],
            "the off-diagonal update_E plan")

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
        varies = self.presence_varies
        kernel[self._grid](
            *self._targets, *self._aux, *self._sources, *self._inv_eps,
            *self._rows, *self._coefficients, self._cb, self._op,
            nx, ny, nz, self.n_elem,
            R01=self.row_mask[0], R02=self.row_mask[1],
            R11=self.row_mask[2], R12=self.row_mask[3],
            R21=self.row_mask[4], R22=self.row_mask[5],
            P01=varies[0], P02=varies[1], P11=varies[2], P12=varies[3],
            P21=varies[4], P22=varies[5], PRESENCE=int(any(varies)),
            BCX=self.boundary_codes[0], BCY=self.boundary_codes[1],
            BCZ=self.boundary_codes[2],
            WM_X=self.wall_axes[0], WM_Y=self.wall_axes[1],
            WM_Z=self.wall_axes[2],
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
        )

    def __repr__(self) -> str:
        return (f"OffdiagConstitutivePlan(shape={self.shape}, "
                f"rows={self.row_mask}, presence={self.presence_varies}, "
                f"bc={self.boundary_codes}, "
                f"walls={self.wall_axes}, block={self.block})")


#: Boundary kind string -> the kernel's constexpr code. The strings are
#: ``stepping``'s own; the codes are the package's (kernels.py:77-78).
BOUNDARY_CODES = {"periodic": 0, "metallic": 1}


def plan_offdiagonal_constitutive(fields: Any, pml: Any,
                                  block: Optional[int] = None,
                                  chunk_layout: Any = None
                                  ) -> Optional[OffdiagConstitutivePlan]:
    """Build the off-diagonal ``update_E`` plan from the engine's own objects.

    None means REFUSED, and the reasons are available from
    :func:`offdiag_constitutive_coverage`. The sources are taken through
    ``displacement_minus_polarization_volumes`` — the accessor the array path
    itself reads (stepping.py:996, :1004) — which with no poles admitted hands
    back the aliased D primaries (fields.py:1107-1138). The Triton import
    stays below the predicate: a NumPy host must be able to plan (to ``None``)
    without the optional dependency being importable at all.

    ``chunk_layout`` is the run's chunk division (``FdtdDriver.chunk_layout``),
    ``None`` for the one-rank default; the plan binds its record
    (``launch.chunk_bounds_for``). The presence of each row slot is the array
    path's own (:func:`presence_for`).
    """
    if not offdiag_constitutive_coverage(fields, pml).covered:
        return None
    from .launch import chunk_bounds_for  # noqa: PLC0415

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
        chunk_bounds=chunk_bounds_for(fields, pml, chunk_layout),
        presence=presence_for(fields, pml),
    )


def plan_offdiagonal_constitutive_from_arrays(
        arrays: Dict[str, Any], flat: Dict[str, Any],
        rows: Dict[str, Dict[str, Any]],
        boundary_codes: Sequence[int], wall_axes: Sequence[int],
        block: Optional[int] = None, kernel: Any = None, *,
        chunk_bounds: Any, presence: Any
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

    ``chunk_bounds`` (``chunk_map.chunk_bounds_record``) and ``presence``
    (:func:`presence_record`, or :data:`ALL_PRESENT`) have no default: a gate whose
    plan fell back to some default division would stop measuring the forms it
    exists to measure.
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
        chunk_bounds=chunk_bounds,
        presence=presence,
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
