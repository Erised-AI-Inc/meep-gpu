"""Triton kernels for the real-field PML step path.

Five kernels live here. Three replace one array-path sub-step each:

* :func:`pml_curl_step` — ``stepping.step_B`` and ``stepping.step_D``;
* :func:`constitutive_step` — ``stepping.update_H`` and ``stepping.update_E``;
* :func:`ade_update_p` — one Lorentz/Drude pole in ``dispersion.PolarizationState.update``,
  MEEP's expression on MEEP's four float32 constants.

The other two replace one curl/constitutive pair each, in one launch:

* :func:`fused_curl_constitutive_B` — ``step_B`` + ``zero_metal_B`` + ``update_H``.
* :func:`fused_curl_constitutive_D` — ``step_D`` + ``zero_metal_D`` + ``update_E``.

It is an ADDITION, not a replacement: the two separate kernels stay, they stay
launchable, and the composed-but-unfused path is the control the fused one is
measured against. Nothing here is dispatched — see :mod:`launch`.

Every one of them takes ``ENABLE_FP_FUSION`` and every one of them has its grouping
pinned by the shared byte gate; see the two numbered notes below, which apply to all
five and not only to the curl.

THE FORMS COME FROM :mod:`.forms`, NOT FROM THIS FILE. MEEP picks its PML form per
structure chunk (step_db.cpp:56-59) and its curl expression per partner presence
(fields.cpp:428-453), and so does the array path. The four kernels below read the
run's chunk record -- the ``int32[25]`` of ``chunk_map.chunk_bounds_record``, bound
as ``cb`` by every plan -- and call the shared ``@triton.jit`` helpers of
:mod:`.forms` for the selects. Those helpers are inlined, so a form boundary is not
a rounding boundary; what each one transcribes is stated there, once.

ONE kernel serves both ``stepping.step_B`` and ``stepping.step_D``. The two differ
only in which direction the stencil shifts (forward for B, MEEP's negated strides
for D), which Yee sub-lattice the PML coefficients come from (half-integer for B,
integer for D), and which cell the metallic ownership mask drops — the curl
expression itself, the term-to-axis table and the split-field recurrence are
identical, exactly as ``step_generic.cpp``'s single ``step_curl`` is one function
for both field types.

WHAT IS TRANSCRIBED, and from where (re-checked against the tree, not the plan):

* term table          ``stepping.B_CURL_TERMS`` / ``D_CURL_TERMS`` (stepping.py:213-223)
* ghost rule          ``stepping._shift_up`` (:1723) / ``_shift_down`` (:1787),
                      PERIODIC and METALLIC branches only
* curl grouping       ``stepping._curl_from_operands``, every presence arm
* partner presence    ``stepping.curl_presence`` (the record's presence word)
* ownership mask      ``stepping._mask_non_owned_cells`` (:1865)
* PML forms           ``stepping._apply_curl_regions``: the recurrence
                      (``_apply_pml_update``) on the component's dsigu cells and the
                      dsig-only form (``_apply_dsig_only_update``) elsewhere, with
                      the auxiliary contract on ``fu``
* chunk membership    ``chunk_map.chunk_bounds_record`` / ``chunk_map.record_index``
* coefficient pairing ``stepping._curl_coefficients`` (:2418)

THE TWO THINGS THAT DECIDE BIT-IDENTITY, both measured rather than assumed:

1. ``ENABLE_FP_FUSION = False``. Triton's ``enable_fp_fusion`` gates BOTH the
   LLVM contraction pass and ptxas's ``--fmad=false``; with it left at its True
   default, LLVM itself contracts ``B - dtdx*curl`` into ``fma.rn.f32`` and the
   result diverges by up to 113238 ulp. It is spelled ONCE, here, because it is a
   per-launch keyword and a launch site that omits it produces a kernel that is
   bit-wrong but numerically plausible.
2. The stencil grouping ``dtdx * (((g1s - g1) + g2) - g2s)`` — MEEP's C associates
   ``g1[i+s1] - g1[i] + g2[i] - g2[i+s2]`` left to right (step_generic.cpp:104), and
   so does ``stepping._curl_from_operands``. The pairwise ``((g1s - g1) + (g2 -
   g2s))`` is the same sum in exact arithmetic and a different float32 number; so is
   the full bracket over an absent partner, which is why
   :func:`.forms.curl_stencil` selects the one-partner forms instead of letting an
   exactly-zero difference ride through. Note that Triton's MLIR pipeline DOES
   rewrite the expression it is given (it was measured swapping the two ``add``
   operands, which is harmless because float addition is bitwise commutative) — so
   this grouping is held by the gate empirically, not guaranteed by construction,
   and a Triton version bump is a correctness event for this file and for
   :mod:`.forms`.

The kernel writes its own inputs in place, which is why ``triton.autotune`` is
deliberately absent: the tuner re-runs a kernel to time it, and on this step path
that silently applies the update dozens of times (measured: every float wrong,
max_abs 1.4e4, no error raised).
"""

from __future__ import annotations

import triton
import triton.language as tl

from .forms import (axis_present, constitutive_form, curl_form, curl_stencil,
                    record_in)

#: Triton's contraction guard, spelled exactly once in this package. Every launch
#: passes this constant; ``test_triton_kernels`` fails the build if a second
#: spelling appears anywhere under ``triton_kernels/``.
ENABLE_FP_FUSION = False

#: Boundary codes matching ``stepping``'s string kinds. Wrapped in
#: ``tl.constexpr`` because a ``@triton.jit`` body may not read a plain module
#: global: Triton raises a NameError naming the variable rather than capturing a
#: stale value. Note the wrapper is required — a bare ``PERIODIC: tl.constexpr = 0``
#: ANNOTATION leaves an ``int`` behind and is refused just the same.
PERIODIC = tl.constexpr(0)
METALLIC = tl.constexpr(1)

#: Elements per program. Not autotuned (see the module docstring); 256 was the
#: median of the configurations the feasibility recon measured, all within noise.
DEFAULT_BLOCK = 256

#: Where the chunk record's half-integer lines start (``chunk_map.record_index(True,
#: 0, 0)``); the integer lines start at word 0. Line ``(lattice, axis)`` is then
#: ``lattice + 4 * axis``. ``test_triton_chunk_bounds`` pins both this and
#: :data:`RECORD_PRESENCE` against ``chunk_map``'s own numbers.
RECORD_HALF_INTEGER = tl.constexpr(12)
#: The record's partner-presence word (``chunk_map.RECORD_PRESENCE_WORD``).
RECORD_PRESENCE = tl.constexpr(24)


@triton.jit
def pml_curl_step(
    f0, f1, f2,                       # targets: Bx,By,Bz  or  Dx,Dy,Dz
    u0, u1, u2,                       # auxiliaries: fu_B*  or  fu_D*
    g0, g1, g2,                       # sources: Ex,Ey,Ez  or  Hx,Hy,Hz
    kmx, sinvx, kmy, sinvy, kmz, sinvz,   # kms/sinv per axis, one Yee sub-lattice
    cb,                               # chunk record: the forms' membership and presence
    nx, ny, nz, n_elem, dtdx,
    BACKWARD: tl.constexpr,           # 0 = B (forward differences), 1 = D
    BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One curl sub-step of all three components, both PML forms included.

    The record decides two things here and nothing else does. Its presence word picks
    the curl expression per target (:func:`.forms.curl_stencil`), and its membership
    line for the target's dsigu axis on the target's own sub-lattice picks the form
    (:func:`.forms.curl_form`): the split-field recurrence inside, the dsig-only form
    outside, with ``fu`` read and written inside only.
    """
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = idx < n_elem
    nyz = ny * nz
    k = idx % nz
    plane = idx // nz
    j = plane % ny
    i = plane // ny

    # --- the ghost rule, per axis (stepping._shift_up / _shift_down) ------------
    # PERIODIC wraps; METALLIC serves an exact 0.0 past the wall, which `tl.load`'s
    # `other=` delivers without dereferencing anything.
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

    # --- the curl (stepping._curl_from_operands): DO NOT flatten these parens ---
    # One call per target: MEEP's left-to-right association with both partners, the
    # one-partner form where the record says the other partner's derivative axis is
    # invariant, and an exact +0 with neither. The presence word is the record's
    # (stepping.curl_presence, MEEP's have_plus_deriv / have_minus_deriv).
    presence = tl.load(cb + RECORD_PRESENCE)
    px = axis_present(presence, 0)
    py = axis_present(presence, 1)
    pz = axis_present(presence, 2)
    curl0 = curl_stencil(c_y, c, b, b_z, dtdx, py, pz)
    curl1 = curl_stencil(a_z, a, c, c_x, dtdx, pz, px)
    curl2 = curl_stencil(b_x, b, a, a_y, dtdx, px, py)

    # --- ownership mask (stepping._mask_non_owned_cells) -----------------------
    # Cell 0 of a metallic axis, for every target whose Yee shift there is 0. The
    # B targets have a single zero shift each (Bx:x, By:y, Bz:z); the D targets
    # have two (Dx:y,z  Dy:x,z  Dz:x,y). Masked, then the recurrence runs on zero.
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

    # --- split-field recurrence (stepping._apply_pml_update) -------------------
    # dsig/dsigu follow vec.hpp's cycle_direction and are the same triple on both
    # sides: target 0 takes (y, z), target 1 (z, x), target 2 (x, y).
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

    # fu is written on the recurrence's cells ONLY: outside them the dsig-only form
    # reads and writes the field alone and fu keeps its word (the auxiliary contract).
    tl.store(u0 + idx, n0, mask=live & du0)
    tl.store(u1 + idx, n1, mask=live & du1)
    tl.store(u2 + idx, n2, mask=live & du2)
    tl.store(f0 + idx, v0, mask=live)
    tl.store(f1 + idx, v1, mask=live)
    tl.store(f2 + idx, v2, mask=live)


@triton.jit(do_not_specialize=["start_h"])
def constitutive_step(
    f0, f1, f2,                       # targets: Hx,Hy,Hz  or  Ex,Ey,Ez
    w0, w1, w2,                       # auxiliaries: f_w_H*  or  f_w_E*
    g0, g1, g2,                       # sources: Bx,By,Bz  or  Dx,Dy,Dz
    e0, e1, e2,                       # inverse epsilon, E side only (see SCALE)
    kp0, km0, kp1, km1, kp2, km2,     # kps/kms on each component's OWN axis
    cb,                               # chunk record: the forms' dsigw membership
    nx, ny, nz, n_elem, start_h,      # start_h: MEEP's start of H (see below)
    SCALE: tl.constexpr,              # 0 = H (source is B), 1 = E (source is D*inv_eps)
    BLOCK: tl.constexpr,
):
    """One constitutive PML sub-step of all three components, both forms included.

    The record's membership line for the component's own (dsigw) axis, on the
    component's own sub-lattice, picks the form (:func:`.forms.constitutive_form`):
    MEEP's ``dsigw`` accumulation inside, ``f = src`` outside, with ``f_w`` written
    inside only.

    ``stepping._apply_constitutive_pml`` (stepping.py:2065), which is MEEP's
    ``step_update_EDHB`` with ``dsigw`` active::

        realnum fwprev = fw[i], kapwkw = kapw[kw], sigwkw = sigw[kw];
        fw[i] = g[i] * u[i];                   // B for H, D*inv_eps for E
        f[i] += (kapwkw + sigwkw) * fw[i] - (kapwkw - sigwkw) * fwprev;

    ONE body serves both sides, as ``pml_curl_step`` does for B/D. ``SCALE`` picks
    ``source`` — the H side reads B directly (mu = 1 is baked into the array path
    too: ``stepping.update_H`` passes ``fields.Bx`` as the source), the E side reads
    ``D * inv_eps`` with **D on the left**, because the array path writes
    ``source * fields.inverse_epsilon_for(component)`` (stepping.py:982-984) and
    float multiplication is bitwise commutative but the transcription is not a place
    to rely on that. ``HALF``-integer coefficient selection is NOT a constexpr here:
    the host binds ``kps_a``/``kms_a`` for H and ``kps_a_h``/``kms_a_h`` for E
    (stepping.py:920 vs :986), so the kernel never chooses and cannot choose wrong —
    but the host can, which is why the gate carries a swap mutation for it.

    THREE THINGS DECIDE BIT-IDENTITY HERE.

    1. **The grouping.** MEEP's ``f[i] += X - Y;`` forms the right-hand side first,
       so the word is ``f + ((kps*src) - (kms*prev))``: the two products, their
       difference, then one add into the field — which is what
       ``stepping._apply_constitutive_pml`` spells and what
       :func:`.forms.constitutive_form` carries. The two-accumulation
       ``((f + kps*src) - kms*prev)`` is a different float32 number, and neither
       multiply may contract into an FMA — ``ENABLE_FP_FUSION`` is what
       stops the second. The PML coefficients are the same family of
       non-representable multiplicands §12.2 measured on the curl, so the guard
       matters at every Courant number, not only at a non-power-of-two one.
    2. **The coefficient index is on the component's OWN axis.** ``kps_x``/``kms_x``
       for target 0, ``_y`` for 1, ``_z`` for 2 (``H_CONSTITUTIVE_TERMS`` /
       ``E_CONSTITUTIVE_TERMS``, stepping.py:226-227) — that is MEEP's ``dsigw``,
       the absorption a component accumulates along the direction it points in. It
       is NOT the ``dsig``/``dsigu`` cycle the curl recurrence uses.
    3. **``prev`` is read before ``fw`` is written.** The array path copies ``fw``
       first (``fw_previous = fw.copy()``) for exactly this reason. Writing the store
       before the load produces a wrong answer only where ``kms != 0`` — i.e. INSIDE
       THE PML ONLY — which looks like a slightly worse absorber, not like a bug.

    ``start_h`` IS MEEP'S START OF H (``stepping.separate_h_from_b``). MEEP allocates no
    H array at first — ``alloc_f`` points H at B's storage (fields.cpp:485-494) — and
    the run's first ``update_eh(H_stuff)`` in a chunk that carries H's own layer starts
    H as a COPY of B and ``f_w`` as a copy of that H (update_eh.cpp:165-177). A nonzero
    ``start_h`` is that first update: on the accumulation's cells the field and the
    previous auxiliary are ``src`` rather than the stored words. The plan passes it
    from ``Fields.h_separated`` and sets that flag afterwards, so it is 1 at most once
    per run; it is 0 on the E side always. It is a runtime value Triton must not
    specialize on (``do_not_specialize``), so one compiled program serves both.

    The sub-step reads no neighbour, so it needs no ghost rule, no ownership mask and
    no boundary constexpr. That does not make it boundary-agnostic: a folded or
    cylindrical axis changes the stored extent and therefore ``n_a``, and the
    coverage predicate refuses both.
    """
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = idx < n_elem
    nyz = ny * nz
    k = idx % nz
    plane = idx // nz
    j = plane % ny
    i = plane // ny

    # Which cells take the accumulation: the component's own (dsigw) layer on the
    # component's own sub-lattice. The own axis is x for target 0, y for 1, z for 2
    # (the constitutive term tables), so each target's record line is again a
    # compile-time word index, chunk_map.record_index(half_integer, axis, 0).
    if SCALE:
        lattice = RECORD_HALF_INTEGER
    else:
        lattice = 0
    dw0 = record_in(cb, lattice + 0, i)
    dw1 = record_in(cb, lattice + 4, j)
    dw2 = record_in(cb, lattice + 8, k)
    # MEEP's start of H (see the docstring); always 0 on the E side.
    start = start_h != 0

    # Component 0 takes its coefficient from axis x, 1 from y, 2 from z.
    kp_0 = tl.load(kp0 + i, mask=live, other=0.0)
    km_0 = tl.load(km0 + i, mask=live, other=0.0)
    kp_1 = tl.load(kp1 + j, mask=live, other=0.0)
    km_1 = tl.load(km1 + j, mask=live, other=0.0)
    kp_2 = tl.load(kp2 + k, mask=live, other=0.0)
    km_2 = tl.load(km2 + k, mask=live, other=0.0)

    # --- component 0 -----------------------------------------------------------
    prev0 = tl.load(w0 + idx, mask=live & dw0, other=0.0)   # BEFORE the store; note 3.
    if SCALE:
        src0 = tl.load(g0 + idx, mask=live, other=0.0) * tl.load(e0 + idx, mask=live,
                                                                 other=0.0)
    else:
        src0 = tl.load(g0 + idx, mask=live, other=0.0)
    tl.store(w0 + idx, src0, mask=live & dw0)
    a0 = tl.load(f0 + idx, mask=live & dw0, other=0.0)
    if SCALE == 0:                    # H starts as a copy of B on its own cells
        prev0 = tl.where(start, src0, prev0)
        a0 = tl.where(start, src0, a0)
    a0 = constitutive_form(a0, prev0, src0, kp_0, km_0, dw0)
    tl.store(f0 + idx, a0, mask=live)

    # --- component 1 -----------------------------------------------------------
    prev1 = tl.load(w1 + idx, mask=live & dw1, other=0.0)
    if SCALE:
        src1 = tl.load(g1 + idx, mask=live, other=0.0) * tl.load(e1 + idx, mask=live,
                                                                 other=0.0)
    else:
        src1 = tl.load(g1 + idx, mask=live, other=0.0)
    tl.store(w1 + idx, src1, mask=live & dw1)
    a1 = tl.load(f1 + idx, mask=live & dw1, other=0.0)
    if SCALE == 0:                    # H starts as a copy of B on its own cells
        prev1 = tl.where(start, src1, prev1)
        a1 = tl.where(start, src1, a1)
    a1 = constitutive_form(a1, prev1, src1, kp_1, km_1, dw1)
    tl.store(f1 + idx, a1, mask=live)

    # --- component 2 -----------------------------------------------------------
    prev2 = tl.load(w2 + idx, mask=live & dw2, other=0.0)
    if SCALE:
        src2 = tl.load(g2 + idx, mask=live, other=0.0) * tl.load(e2 + idx, mask=live,
                                                                 other=0.0)
    else:
        src2 = tl.load(g2 + idx, mask=live, other=0.0)
    tl.store(w2 + idx, src2, mask=live & dw2)
    a2 = tl.load(f2 + idx, mask=live & dw2, other=0.0)
    if SCALE == 0:                    # H starts as a copy of B on its own cells
        prev2 = tl.where(start, src2, prev2)
        a2 = tl.where(start, src2, a2)
    a2 = constitutive_form(a2, prev2, src2, kp_2, km_2, dw2)
    tl.store(f2 + idx, a2, mask=live)


@triton.jit(do_not_specialize=["start_h"])
def fused_curl_constitutive_B(
    f0, f1, f2,                       # curl targets: Bx,By,Bz
    u0, u1, u2,                       # curl auxiliaries: fu_Bx,fu_By,fu_Bz
    g0, g1, g2,                       # curl sources: Ex,Ey,Ez
    kmx, sinvx, kmy, sinvy, kmz, sinvz,   # curl kms/sinv, HALF-INTEGER sub-lattice
    h0, h1, h2,                       # constitutive targets: Hx,Hy,Hz
    w0, w1, w2,                       # constitutive auxiliaries: f_w_Hx,f_w_Hy,f_w_Hz
    kp0, km0, kp1, km1, kp2, km2,     # constitutive kps/kms, INTEGER sub-lattice
    cb,                               # chunk record: the forms' membership and presence
    nx, ny, nz, n_elem, dtdx, start_h,  # start_h: constitutive_step's start of H
    BACKWARD: tl.constexpr,           # bound to 0 by every planner; see below
    BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
    ZM_X: tl.constexpr, ZM_Y: tl.constexpr, ZM_Z: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """``step_B`` and ``update_H`` in ONE launch — the cross-sub-step fusion (F1).

    THE POINT IS THE MEMORY ROUND TRIP, NOT THE ARITHMETIC. :func:`pml_curl_step`
    stores ``v0`` into ``Bx`` and :func:`constitutive_step` immediately loads
    ``Bx`` back at the SAME index; both kernels are bandwidth-bound, so that store
    and that load are attacking the binding constraint and nothing else. This body
    is the two bodies concatenated with one substitution: the constitutive half's
    ``src0 = tl.load(g0 + idx)`` becomes the register ``v0`` the curl half just
    computed. Every parenthesisation, every mask, every coefficient index is the
    original's.

    ``start_h`` is :func:`constitutive_step`'s, for the H half: the registers
    ``v0``..``v2`` the curl just computed ARE the B words MEEP's first ``update_eh``
    would copy from.

    IS THE ROUND TRIP LOAD-BEARING FOR BIT-IDENTITY? No, and it is measured rather
    than argued. A Triton fp32 value lives in a 32-bit register on NVIDIA — there
    is no x87-style extended-precision accumulator for it to be truncated FROM —
    so ``store(f32 reg); load(f32)`` is the identity function on the bits. The gate
    carries ``reload_B_from_memory`` (put the load back, keep the fusion) precisely
    to measure that: it must be identical, and its shifted sibling
    ``reload_B_from_memory_but_shift`` must not be, which is what makes the first
    result a measurement instead of a tautology.

    ``BACKWARD`` IS CARRIED AND BOUND TO 0. Keeping it makes the curl half a
    verbatim copy rather than a hand-specialised one, which is the whole reason the
    transcription risk here is low. It must be 0: the D-side pair is Phase B, its
    ownership mask covers two axes per component instead of one, and — decisively —
    an electric source deposits into D between ``step_D`` and ``update_E``, which
    this kernel has no way to carry. ``plan_fused_pair`` refuses anything else.

    ``ZM_X``/``ZM_Y``/``ZM_Z`` CARRY ``stepping.zero_metal_B`` INLINE, and they are
    not decoration. On a walled run the driver clears stored cell 0 of every B
    component whose Yee shift on that axis is 0 — ``Bx`` on an x wall, ``By`` on y,
    ``Bz`` on z (``IYEE_SHIFTS``, fields.py:214-219) — and it does so BETWEEN the
    two sub-steps (driver.py:3167 vs :3169), so the constitutive half must read the
    ZEROED ``B``. Refusing walls instead would silently narrow this kernel's
    coverage relative to ``pml_curl_step``'s own predicate, and no benchmark case
    is walled, so the bug would never surface in a timing row.

    The flags are NOT the ``BC*`` codes. ``_boundary_kinds`` resolves the ghost
    rule; ``_zero_metal`` asks ``grid.is_metallic(axis) and not
    grid.is_mirrored(axis)`` and returns early unless ``grid.has_metallic``. The
    host binds them from that second question, and the gate carries
    ``drop_zero_metal_inline`` and ``zero_metal_after_constitutive`` for it.

    WHAT IS NOT CARRIED, and therefore refused by ``fused_pair_coverage``: a
    magnetic source. ``FdtdDriver.step`` injects magnetic currents between
    ``step_B`` and ``update_H`` (driver.py:3164-3165), so a run that has one has
    real work in the seam and the fusion is not a code motion. Phase A stays free
    of the source question entirely.
    """
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = idx < n_elem
    nyz = ny * nz
    k = idx % nz
    plane = idx // nz
    j = plane % ny
    i = plane // ny

    # ======================= the curl half ====================================
    # Verbatim from pml_curl_step; the only edit below its stores is that nothing
    # reads f0/f1/f2 back.

    # --- the ghost rule, per axis (stepping._shift_up / _shift_down) ------------
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

    # --- the curl (stepping._curl_from_operands): DO NOT flatten these parens ---
    # One call per target: MEEP's left-to-right association with both partners, the
    # one-partner form where the record says the other partner's derivative axis is
    # invariant, and an exact +0 with neither. The presence word is the record's
    # (stepping.curl_presence, MEEP's have_plus_deriv / have_minus_deriv).
    presence = tl.load(cb + RECORD_PRESENCE)
    px = axis_present(presence, 0)
    py = axis_present(presence, 1)
    pz = axis_present(presence, 2)
    curl0 = curl_stencil(c_y, c, b, b_z, dtdx, py, pz)
    curl1 = curl_stencil(a_z, a, c, c_x, dtdx, pz, px)
    curl2 = curl_stencil(b_x, b, a, a_y, dtdx, px, py)

    # --- ownership mask (stepping._mask_non_owned_cells) -----------------------
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

    # --- split-field recurrence (stepping._apply_pml_update) -------------------
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

    # --- the seam: stepping.zero_metal_B, between the two sub-steps ------------
    # Applied to the REGISTER, before both the store and the constitutive read, so
    # the two consumers see the one value the array path leaves in B.
    if ZM_X:
        v0 = tl.where(at_x, 0.0, v0)
    if ZM_Y:
        v1 = tl.where(at_y, 0.0, v1)
    if ZM_Z:
        v2 = tl.where(at_z, 0.0, v2)

    # fu is written on the recurrence's cells ONLY: outside them the dsig-only form
    # reads and writes the field alone and fu keeps its word (the auxiliary contract).
    tl.store(u0 + idx, n0, mask=live & du0)
    tl.store(u1 + idx, n1, mask=live & du1)
    tl.store(u2 + idx, n2, mask=live & du2)
    tl.store(f0 + idx, v0, mask=live)
    tl.store(f1 + idx, v1, mask=live)
    tl.store(f2 + idx, v2, mask=live)

    # ==================== the constitutive half ================================
    # Verbatim from constitutive_step's SCALE=0 arm, with `src0 = tl.load(g0+idx)`
    # replaced by the register v0 (and v1/v2). The three source pointers and the
    # three inv_eps placeholders are gone with it.
    kp_0 = tl.load(kp0 + i, mask=live, other=0.0)
    km_0 = tl.load(km0 + i, mask=live, other=0.0)
    kp_1 = tl.load(kp1 + j, mask=live, other=0.0)
    km_1 = tl.load(km1 + j, mask=live, other=0.0)
    kp_2 = tl.load(kp2 + k, mask=live, other=0.0)
    km_2 = tl.load(km2 + k, mask=live, other=0.0)

    # --- component 0 -----------------------------------------------------------
    # Which cells take the accumulation: the component's own (dsigw) layer on the
    # component's own sub-lattice. The own axis is x for target 0, y for 1, z for 2
    # (the constitutive term tables), so each target's record line is again a
    # compile-time word index, chunk_map.record_index(half_integer, axis, 0).
    dw0 = record_in(cb, 0, i)         # H: the integer sub-lattice
    dw1 = record_in(cb, 4, j)
    dw2 = record_in(cb, 8, k)
    # MEEP's start of H (see the docstring).
    start = start_h != 0
    prev0 = tl.load(w0 + idx, mask=live & dw0, other=0.0)   # BEFORE the store.
    src0 = v0
    tl.store(w0 + idx, src0, mask=live & dw0)
    a0 = tl.load(h0 + idx, mask=live & dw0, other=0.0)
    prev0 = tl.where(start, src0, prev0)    # MEEP's start of H
    a0 = tl.where(start, src0, a0)
    a0 = constitutive_form(a0, prev0, src0, kp_0, km_0, dw0)
    tl.store(h0 + idx, a0, mask=live)

    # --- component 1 -----------------------------------------------------------
    prev1 = tl.load(w1 + idx, mask=live & dw1, other=0.0)
    src1 = v1
    tl.store(w1 + idx, src1, mask=live & dw1)
    a1 = tl.load(h1 + idx, mask=live & dw1, other=0.0)
    prev1 = tl.where(start, src1, prev1)    # MEEP's start of H
    a1 = tl.where(start, src1, a1)
    a1 = constitutive_form(a1, prev1, src1, kp_1, km_1, dw1)
    tl.store(h1 + idx, a1, mask=live)

    # --- component 2 -----------------------------------------------------------
    prev2 = tl.load(w2 + idx, mask=live & dw2, other=0.0)
    src2 = v2
    tl.store(w2 + idx, src2, mask=live & dw2)
    a2 = tl.load(h2 + idx, mask=live & dw2, other=0.0)
    prev2 = tl.where(start, src2, prev2)    # MEEP's start of H
    a2 = tl.where(start, src2, a2)
    a2 = constitutive_form(a2, prev2, src2, kp_2, km_2, dw2)
    tl.store(h2 + idx, a2, mask=live)


@triton.jit
def fused_curl_constitutive_D(
    f0, f1, f2,                       # curl targets: Dx,Dy,Dz
    u0, u1, u2,                       # curl auxiliaries: fu_Dx,fu_Dy,fu_Dz
    g0, g1, g2,                       # curl sources: Hx,Hy,Hz
    kmx, sinvx, kmy, sinvy, kmz, sinvz,   # curl kms/sinv, INTEGER sub-lattice
    e0, e1, e2,                       # constitutive targets: Ex,Ey,Ez
    w0, w1, w2,                       # constitutive auxiliaries: f_w_Ex,f_w_Ey,f_w_Ez
    ie0, ie1, ie2,                    # diagonal inverse epsilon volumes
    kp0, km0, kp1, km1, kp2, km2,     # constitutive kps/kms, HALF-INTEGER sub-lattice
    cb,                               # chunk record: the forms' membership and presence
    nx, ny, nz, n_elem, dtdx,
    BACKWARD: tl.constexpr,           # bound to 1 by the D planner
    BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
    ZM_X: tl.constexpr, ZM_Y: tl.constexpr, ZM_Z: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """``step_D`` and ``update_E`` in one launch — cross-sub-step fusion F2.

    This is the electric counterpart of :func:`fused_curl_constitutive_B`, kept a
    separate kernel because its seam is materially different. ``step_D`` uses the
    backward stencil and integer curl coefficients; ``update_E`` uses half-integer
    constitutive coefficients and multiplies each D register by its own diagonal
    inverse-epsilon volume. A metallic wall clears TWO tangential D components per
    axis, not the one normal B component the magnetic kernel clears.

    Electric sources are not carried. The predicate refuses every run with one,
    because the driver injects it after ``step_D`` and before ``zero_metal_D`` and
    ``update_E``. Dispersion is refused by ``constitutive_coverage`` because the
    consumer would be ``(D - sum(P)) * inv_eps``, not the register-only expression
    below. Under that explicit envelope, the only seam work is ``zero_metal_D``,
    transcribed before both the D store and the E read.
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


@triton.jit
def ade_update_p(
    p_out, p_now, p_prev, sigma, drive,
    two_minus_od, gamma1, omega0dtsqr, gamma1inv, n_elem,
    SIGMA_IS_VOLUME: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One Lorentz/Drude pole's ADE recurrence for one component: MEEP's expression.

    ``dispersion.PolarizationState.update``, which transcribes MEEP's isotropic
    ``update_P`` (susceptibility.cpp:251-256, contraction off)::

        p[i] = gamma1inv *
               (pcur * (2 - omega0dtsqr_denom) - gamma1 * pp[i] + omega0dtsqr * (s[i] * w[i]));

    that is ``gamma1inv * (((p * two_minus_od) - (gamma1 * pp)) + (omega0dtsqr * (s * w)))``,
    every operation one float32 rounding. The four scalars are MEEP's ``realnum`` words,
    ``PolarizationState.constants`` (``Susceptibility.realnum_constants``), formed on the
    host where MEEP forms them; ``two_minus_od`` is exactly 2 for a Drude term. The middle
    term is a SUBTRACTION of ``gamma1 * pp``: folding ``gamma1inv`` into three
    coefficients, ``((p*c_now) + (c_prev*pp)) + (c_drive*(s*w))``, is the same recurrence
    in exact arithmetic and a different float32 number on most words (the array path's
    ``dispersive_update_folded_three_term_form`` mutation). Multiplication operand order
    is free (IEEE-754 multiplication is commutative on every word); the grouping of the
    sum is not. ``sigma`` is a uniform scalar in most runs and a full volume in a graded
    one; both are carried, chosen at compile time so the uniform case pays no load. A
    uniform sigma is MEEP's ``realnum`` too: the plan passes the Python float, which
    Triton types as float32, the word ``numpy.float32(sigma)`` the array path forms.

    ``drive`` is MEEP's ``w`` (update_pols.cpp:44, ``w = f_w ? f_w : f`` per chunk), which
    the plan reads through ``Fields.drive_field``: ``f_w`` on the cells whose chunk
    carries the component's own layer, the stored E everywhere else.

    Writes to ``p_out``, which the caller rotates into the P slot — the array path
    never updates P in place, and reproducing that is what keeps P_prev valid.
    """
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = idx < n_elem
    p = tl.load(p_now + idx, mask=live, other=0.0)
    q = tl.load(p_prev + idx, mask=live, other=0.0)
    w = tl.load(drive + idx, mask=live, other=0.0)
    if SIGMA_IS_VOLUME:
        s = tl.load(sigma + idx, mask=live, other=0.0)
    else:
        s = sigma
    tl.store(p_out + idx,
             gamma1inv * (((p * two_minus_od) - (gamma1 * q)) + (omega0dtsqr * (s * w))),
             mask=live)
