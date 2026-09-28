"""Mirror-symmetry (folded-grid) Triton composition and its two special kernels.

The folded curl and mirror-fill kernels are additions.  The constitutive portion
reuses :func:`kernels.constitutive_step`, because it is element-wise and its PML
coefficient vectors already use the folded grid's stored extent.  The experimental
``plan_step`` composer selects all three pieces for the nondispersive folded slice;
production dispatch remains disabled.

* :func:`pml_curl_step_folded` — ``stepping.step_B`` / ``stepping.step_D`` on a
  grid one or more of whose axes is HALVED by an ``mp.Mirror`` plane. It is
  :func:`kernels.pml_curl_step` with a widened boundary constexpr set and ONE
  added mask; every line of the PML recurrence, the curl grouping, the
  coefficient pairing and the dsig/dsigu cycle is byte-copied from it.
* :func:`mirror_ghost_fill` — ``stepping.fill_symmetry_bc_{B,D}`` plus
  ``stepping.fill_folded_far_ghosts_{B,D}`` for ONE family on ONE axis, in one
  launch instead of up to six strided whole-plane assignments.

WHY THE CURL NEEDS ALMOST NOTHING (measured, not argued). A fold changes the
ghost rule on the folded axis: the near face becomes ``parity * field[2]``
(``stepping._shift_down`` :1787, source index ``MIRROR_SOURCE_INDEX = 2``) and
the far face becomes ``parity * field[reflect_row]`` on a folded PERIODIC axis
(``stepping._shift_up`` :1723) or an exact zero on a folded METALLIC one. BOTH
GHOST VALUES ARE DEAD IN THE CURL: their only consumer is the plane the
ownership mask (``stepping._mask_non_owned_cells`` :1865) zeroes.

    * ``_shift_down`` is used by ``step_D`` only (``stepping._curl_operands``
      :1560) and its ghost lands at stored cell 0, whose target has Yee shift 0
      on that axis — the cell the mirrored/metallic branch of the mask drops.
    * ``_shift_up`` is used by ``step_B`` only and its ghost lands at the LAST
      stored slot, whose target has Yee shift 1 there — the cell the fold's OWN
      mask drops, but only on a folded PERIODIC axis, where the stored array
      carries one slot past MEEP's owned window (``_stored_past_owned`` :1454).
      On a folded METALLIC axis that top plane IS stepped, and the ghost there
      must be exactly ``0.0`` — which is what the inherited METALLIC branch's
      ``other=0.0`` already delivers.

So the whole kernel delta is (a) admit the fold in the predicate, and (b) add the
Yee-shift-1 top-plane mask on a folded PERIODIC axis. The parity arithmetic never
enters the curl at all. That equivalence was measured at sub-step granularity, on
both terminations, with the far face driven live, before this file was written;
the gate re-measures it (``parity/meep_gpu/gate_triton_symmetry.py``).

THE MASKS ARE INVISIBLE AT WHOLE-STEP GRANULARITY. The driver's fill passes
(``driver.py:3163-3183``) overwrite exactly the planes the two masks protect, so
a kernel carrying NEITHER mask is bytewise-identical after a complete step. A
whole-step in-session check therefore certifies a mask-less symmetry kernel as
correct; the gate for this file has to be — and is — the sub-step leg.

COMPOSITION BOUNDARY.  The composer places ``fill_B`` after magnetic source
injection and before ``update_H``, and ``fill_D`` after electric source injection
and before ``update_E``.  It does not fuse across either seam.  Folded dispersion,
conductivity, nonlinear media, complex/Bloch storage, BFAST and special-kz remain
outside this composed slice.  Dispatch stays disabled regardless: the subnormal
question that blocks the shipped kernels blocks this one too, and a folded run
puts a deep-PML plane on the far face, which is exactly where tiny magnitudes live.

WHAT THE GATE MEASURED (``parity/meep_gpu/gate_triton_symmetry.py``, one clear
RTX A6000, Triton 3.1.0 / CuPy 13.5.1 / NumPy 2.2.6, artifact
``parity/meep_gpu/results/triton_symmetry_2026-08-10/``):

* synthetic sub-step sweep — **512/512 bit-identical guarded, 0/512 unguarded**
  (4 shapes x 9 boundary sets x {0.5, 0.35} x phase {+1, -1} x reflect-row parity
  x {step_B, step_D}, comparing the target AND its ``fu_*`` auxiliary);
  multi-step 48/48;
* ``stepping.step_B``/``step_D`` substituted on real folded CuPy grids with real
  PML tables — **7/7 bit-identical** over four steps each;
* ghost fills against ``fill_symmetry_bc_*`` + ``fill_folded_far_ghosts_*`` —
  **14/14 bit-identical**;
* **14 mutations caught (9 on the curl, 5 on the fill), 1 measured null.**
  ``drop_top_plane_mask`` is caught ONLY on ``MIRROR_PERIODIC`` cases and
  ``fold_ghost_wraps`` ONLY on ``MIRROR_METALLIC`` ones, and
  ``reflect_row_n_minus_two`` ONLY at an odd full count — a sweep carrying one
  termination, or one count parity, proves nothing about the other;
* the fills cost 130-410 us/step on the array path against 45-52 us for the
  kernel (2.6x-9x), essentially flat across a 64x area range on both sides,
  which is what launch-bound rather than bandwidth-bound looks like.

The reference the sweep compares against is itself pinned against ``stepping.py``
on NumPy, 8/8 bit-identical, by
``parity/meep_gpu/validate_fold_reference_vs_stepping.py`` — which runs on a
laptop and is what stops "bit-identical" from meaning "the kernel reproduces
whatever the harness wrote twice".
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .coverage import (
    CURL_TARGETS,
    B_SOURCES,
    CONSTITUTIVE_SIDES,
    D_SOURCES,
    Coverage,
    _boundary_kinds,
    _call,
    _coefficient_reasons,
    _inverse_epsilon_reasons,
    _layout_reasons,
    _susceptibility_reasons,
)

# ---------------------------------------------------------------------------
# The constants the kernel and the host agree on
# ---------------------------------------------------------------------------

#: The four ghost rules this kernel writes, as plain ints for the host side.
#: ``PERIODIC``/``METALLIC`` MUST keep :mod:`kernels`' values — a plan built here
#: and a plan built there index the same table — and ``test_triton_symmetry``
#: reads both spellings off the two sources and pins them equal.
CODE_PERIODIC = 0
CODE_METALLIC = 1
CODE_MIRROR_METALLIC = 2
CODE_MIRROR_PERIODIC = 3

#: ``stepping.MIRROR_SOURCE_INDEX`` (stepping.py:160) — MEEP's ``io = -2`` halved
#: origin, so the near ghost images stored cell 2. Re-stated rather than imported
#: so this module stays engine-import-free at module scope; the test pins them.
MIRROR_SOURCE_INDEX = 2

#: ``stepping._boundary_kinds``' own strings, in the order this file codes them.
BOUNDARY_KIND_NAMES: Tuple[str, ...] = ("periodic", "metallic", "mirror")

#: Yee shifts of the six curl targets (``fields.IYEE_SHIFTS``, fields.py:214-219),
#: re-stated for the same reason and pinned by the same test. The B family's
#: shifts are 1 on the two axes that are NOT its own; the D family's are 1 on its
#: own axis only — which is why the two sub-steps mask different planes.
TARGET_IYEE: Dict[str, Tuple[int, int, int]] = {
    "Bx": (0, 1, 1), "By": (1, 0, 1), "Bz": (1, 1, 0),
    "Dx": (1, 0, 0), "Dy": (0, 1, 0), "Dz": (0, 0, 1),
}

#: Which family each fill launch covers. ``targets`` are in kernel argument order.
GHOST_FILL_FAMILIES: Dict[str, Dict[str, Any]] = {
    "B": {"targets": ("Bx", "By", "Bz")},
    "D": {"targets": ("Dx", "Dy", "Dz")},
}


def _kernel_module():
    """Import :mod:`kernels` lazily — it imports Triton, this module must not.

    Same discipline as :mod:`launch`: a NumPy host with no optional dependency
    must be able to import this file, read its predicate and get a refusal.
    """
    from . import kernels  # noqa: PLC0415

    return kernels


# ---------------------------------------------------------------------------
# The kernels
# ---------------------------------------------------------------------------
# Deferred behind a flag so the module imports with Triton absent. Everything
# below the guard is device code; everything above it is the predicate and the
# plan, which run anywhere.

try:  # pragma: no cover - exercised by the Triton-absent test
    import triton
    import triton.language as tl
except Exception:  # noqa: BLE001 - a missing optional dependency is not an error
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]


if triton is not None:  # pragma: no cover - device code, gated by the shared probe

    PERIODIC = tl.constexpr(CODE_PERIODIC)
    METALLIC = tl.constexpr(CODE_METALLIC)
    MIRROR_METALLIC = tl.constexpr(CODE_MIRROR_METALLIC)
    MIRROR_PERIODIC = tl.constexpr(CODE_MIRROR_PERIODIC)

    @triton.jit
    def pml_curl_step_folded(
        f0, f1, f2,                       # targets: Bx,By,Bz  or  Dx,Dy,Dz
        u0, u1, u2,                       # auxiliaries: fu_B*  or  fu_D*
        g0, g1, g2,                       # sources: Ex,Ey,Ez  or  Hx,Hy,Hz
        kmx, sinvx, kmy, sinvy, kmz, sinvz,   # kms/sinv per axis, one sub-lattice
        nx, ny, nz, n_elem, dtdx,
        BACKWARD: tl.constexpr,           # 0 = B (forward differences), 1 = D
        BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """One curl sub-step on a FOLDED grid; otherwise ``kernels.pml_curl_step``.

        The boundary constexprs take four values here instead of two:
        ``PERIODIC``, ``METALLIC``, ``MIRROR_METALLIC`` (a folded axis whose outer
        declaration is metallic) and ``MIRROR_PERIODIC`` (a folded axis whose
        outer declaration is periodic — MEEP's ``use_bloch`` at k = 0). The
        classification comes from :func:`folded_axis_kinds`, through
        ``stepping._stored_past_owned``, and it is the single point of failure in
        this file: backwards on one axis is a plane of wrong values, not a crash.

        TWO DIFFERENCES FROM THE SHIPPED KERNEL, both named:

        1. Both mirror codes take the METALLIC GHOST BRANCH — mask the
           out-of-range neighbour, serve ``other=0.0``. That is not the array
           path's ghost VALUE on a fold (``_shift_down`` :1787 serves
           ``parity * field[2]``; ``_shift_up`` :1723 serves
           ``parity * field[reflect_row]`` on a folded periodic axis), and it does
           not have to be: the only cell that reads either ghost is the cell one
           of the two masks below zeroes. Measured bytewise, on both
           terminations, with the far face live.
        2. A folded PERIODIC axis masks its LAST plane for every target whose Yee
           shift is 1 there. ``_mask_non_owned_cells`` (:1865) does this because
           that slot sits past MEEP's owned window (``owns``, vec.cpp:445-462) and
           the fill pass, not the curl, is what writes it.

        The cell-0 mask widens from "metallic" to "not periodic":
        ``_mask_non_owned_cells`` asks ``is_mirrored or is_metallic or is_axis``,
        and ``_boundary_kinds`` resolves exactly those three to a non-periodic
        kind (the cylindrical one is refused by the predicate).
        """
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = idx < n_elem
        nyz = ny * nz
        k = idx % nz
        plane = idx // nz
        j = plane % ny
        i = plane // ny

        # --- the ghost rule, per axis (stepping._shift_up / _shift_down) --------
        # PERIODIC wraps; every other rule here serves an exact 0.0 past the face,
        # which `tl.load`'s `other=` delivers without dereferencing anything. On a
        # folded axis that zero stands in for a value nothing reads (see 1 above).
        if BACKWARD:
            si, sj, sk = i - 1, j - 1, k - 1
        else:
            si, sj, sk = i + 1, j + 1, k + 1
        vx, vy, vz = live, live, live
        if BCX == PERIODIC:
            si = tl.where(si < 0, nx - 1, tl.where(si == nx, 0, si))
        else:
            vx = live & (si >= 0) & (si < nx)
        if BCY == PERIODIC:
            sj = tl.where(sj < 0, ny - 1, tl.where(sj == ny, 0, sj))
        else:
            vy = live & (sj >= 0) & (sj < ny)
        if BCZ == PERIODIC:
            sk = tl.where(sk < 0, nz - 1, tl.where(sk == nz, 0, sk))
        else:
            vz = live & (sk >= 0) & (sk < nz)

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

        # --- the curl (stepping._curl_from_operands): DO NOT flatten these parens
        curl0 = dtdx * ((c_y - c) + (b - b_z))
        curl1 = dtdx * ((a_z - a) + (c - c_x))
        curl2 = dtdx * ((b_x - b) + (a - a_y))

        # --- ownership mask, cell 0 (stepping._mask_non_owned_cells) ------------
        # Every target whose Yee shift is 0 on a non-periodic axis. Byte-copied
        # from the shipped kernel with `== METALLIC` widened to `!= PERIODIC`.
        at_x, at_y, at_z = i == 0, j == 0, k == 0
        if BACKWARD:
            if BCY != PERIODIC:
                curl0 = tl.where(at_y, 0.0, curl0)
            if BCZ != PERIODIC:
                curl0 = tl.where(at_z, 0.0, curl0)
            if BCX != PERIODIC:
                curl1 = tl.where(at_x, 0.0, curl1)
            if BCZ != PERIODIC:
                curl1 = tl.where(at_z, 0.0, curl1)
            if BCX != PERIODIC:
                curl2 = tl.where(at_x, 0.0, curl2)
            if BCY != PERIODIC:
                curl2 = tl.where(at_y, 0.0, curl2)
        else:
            if BCX != PERIODIC:
                curl0 = tl.where(at_x, 0.0, curl0)
            if BCY != PERIODIC:
                curl1 = tl.where(at_y, 0.0, curl1)
            if BCZ != PERIODIC:
                curl2 = tl.where(at_z, 0.0, curl2)

        # --- ownership mask, the TOP plane of a folded PERIODIC axis ------------
        # The complement of the block above: every target whose Yee shift is 1
        # there. Written out per (side, target, axis) exactly as that block is —
        # no loop, no derived predicate — so a reader checks it against
        # `_mask_non_owned_cells`'s `if iyee[axis] != 0` arm by eye.
        last_x, last_y, last_z = i == nx - 1, j == ny - 1, k == nz - 1
        if BACKWARD:
            # Dx:(1,0,0)  Dy:(0,1,0)  Dz:(0,0,1) — shift 1 on its OWN axis only.
            if BCX == MIRROR_PERIODIC:
                curl0 = tl.where(last_x, 0.0, curl0)
            if BCY == MIRROR_PERIODIC:
                curl1 = tl.where(last_y, 0.0, curl1)
            if BCZ == MIRROR_PERIODIC:
                curl2 = tl.where(last_z, 0.0, curl2)
        else:
            # Bx:(0,1,1)  By:(1,0,1)  Bz:(1,1,0) — shift 1 on the two OTHER axes.
            if BCY == MIRROR_PERIODIC:
                curl0 = tl.where(last_y, 0.0, curl0)
            if BCZ == MIRROR_PERIODIC:
                curl0 = tl.where(last_z, 0.0, curl0)
            if BCX == MIRROR_PERIODIC:
                curl1 = tl.where(last_x, 0.0, curl1)
            if BCZ == MIRROR_PERIODIC:
                curl1 = tl.where(last_z, 0.0, curl1)
            if BCX == MIRROR_PERIODIC:
                curl2 = tl.where(last_x, 0.0, curl2)
            if BCY == MIRROR_PERIODIC:
                curl2 = tl.where(last_y, 0.0, curl2)

        # --- split-field recurrence (stepping._apply_pml_update) ----------------
        # Byte-copied. The PML coefficient vectors are built at the STORED extent
        # on a folded axis (measured: kms_y.shape == (1, 22, 1) on a grid storing
        # 22), so the `n_a` indexing needs no fold-aware change.
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

        tl.store(u0 + idx, n0, mask=live)
        tl.store(u1 + idx, n1, mask=live)
        tl.store(u2 + idx, n2, mask=live)
        tl.store(f0 + idx, v0, mask=live)
        tl.store(f1 + idx, v1, mask=live)
        tl.store(f2 + idx, v2, mask=live)

    @triton.jit
    def mirror_ghost_fill(
        f0, f1, f2,                       # one family: Bx,By,Bz  or  Dx,Dy,Dz
        nx, ny, nz, n_plane,
        reflect_row,                      # RUNTIME: stepping._far_reflect_rows
        AXIS: tl.constexpr,               # 0 = x, 1 = y, 2 = z
        PHASE: tl.constexpr,              # the plane's declared phase, +1 or -1
        FAR: tl.constexpr,                # 1 on a folded PERIODIC axis only
        S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,   # Yee shift on AXIS
        BLOCK: tl.constexpr,
    ):
        """The two mirror ghost planes of ONE family on ONE axis, in one launch.

        Replaces, for that (family, axis) pair, the whole-plane assignments in
        ``stepping._fill_symmetry_ghost_cells`` (:1426) and
        ``stepping._fill_folded_far_ghosts`` (:1489)::

            near:  field[0]  = +phase * field[2]              # Yee shift 0 here
            far:   field[-1] = -phase * field[reflect_row]    # Yee shift 1 here,
                                                              # folded PERIODIC only

        THE TWO PARITIES ARE ``+PHASE`` AND ``-PHASE`` AND NOTHING ELSE.
        ``fields.mirror_parity(c, axis, phase) == phase * (1 - 2 * iyee[c][axis])``
        (fields.py:117), verified exhaustively over 12 components x 3 axes x 2
        phases; the near fill only ever touches shift-0 components and the far
        fill only shift-1 ones, so one signed constexpr per folded axis is the
        entire parity input. The multiply is by exactly +/-1.0, which is exact in
        float32 for every input including subnormals and signed zeros.

        ``reflect_row`` IS A RUNTIME SCALAR AND MUST COME FROM
        ``stepping._far_reflect_rows`` (:1661), ``n_full - stored + 2``. It is
        ``stored - 2`` at an even full count and ``stored - 3`` at an odd one;
        baking ``n - 2`` reflects about the window top instead of about the second
        mirror, which is a whole cell wrong on every odd-count run.

        ONE LAUNCH PER AXIS. The array path applies the axes in X, Y, Z order so
        a corner unowned on two planes carries the product of both parities
        (``_fill_symmetry_ghost_cells``' docstring), and ordering inside a launch
        is not something a grid of programs can promise — so the plan holds a list
        of launches and walks it in that order.

        WHETHER THE ORDER CHANGES THE BYTES WAS MEASURED, NOT ASSUMED, and the
        answer is no: every fill is a multiply by exactly +/-1 from a plane no
        other axis's fill writes, so two axes' fills COMMUTE bitwise and the
        doubly-unowned corner comes out ``phase_x * phase_y * f[2, 2]`` either way
        (gate leg ``reverse_axis_order``: 14/14 identical, recorded as a null).
        The order is kept because matching the array path is what makes that a
        measurement about this configuration rather than a property to lean on.

        Within one axis there is no hazard either: every write plane is distinct
        from every read plane (0 vs 2, and -1 vs ``reflect_row`` <= stored - 2),
        which the predicate checks rather than assumes.
        """
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = idx < n_plane

        # Flatten one plane of the C-contiguous (nx, ny, nz) volume: the offset of
        # index `row` along AXIS, plus the in-plane offset the program owns.
        if AXIS == 0:
            stride = ny * nz
            base = idx
            last = nx - 1
        elif AXIS == 1:
            stride = nz
            base = (idx // nz) * (ny * nz) + (idx % nz)
            last = ny - 1
        else:
            stride = 1
            base = (idx // ny) * (ny * nz) + (idx % ny) * nz
            last = nz - 1

        # --- near face: cell 0 = +PHASE * cell MIRROR_SOURCE_INDEX --------------
        # Only the components with Yee shift 0 on this axis have an unowned cell 0
        # (MEEP's little_owned_corner0(c) = little_corner + 2 - iyee_shift(c)).
        if S0 == 0:
            tl.store(f0 + base, PHASE * tl.load(f0 + base + 2 * stride, mask=live,
                                                other=0.0), mask=live)
        if S1 == 0:
            tl.store(f1 + base, PHASE * tl.load(f1 + base + 2 * stride, mask=live,
                                                other=0.0), mask=live)
        if S2 == 0:
            tl.store(f2 + base, PHASE * tl.load(f2 + base + 2 * stride, mask=live,
                                                other=0.0), mask=live)

        # --- far face: last cell = -PHASE * cell reflect_row --------------------
        # Only on a folded PERIODIC axis, and only for shift-1 components: a
        # shift-0 component's top slot IS MEEP's big_corner, owned and stepped.
        if FAR:
            if S0 == 1:
                tl.store(f0 + base + last * stride,
                         -PHASE * tl.load(f0 + base + reflect_row * stride,
                                          mask=live, other=0.0), mask=live)
            if S1 == 1:
                tl.store(f1 + base + last * stride,
                         -PHASE * tl.load(f1 + base + reflect_row * stride,
                                          mask=live, other=0.0), mask=live)
            if S2 == 1:
                tl.store(f2 + base + last * stride,
                         -PHASE * tl.load(f2 + base + reflect_row * stride,
                                          mask=live, other=0.0), mask=live)


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def folded_axis_kinds(grid: Any, pml: Any) -> Tuple[Optional[Tuple[int, int, int]],
                                                    Tuple[str, ...]]:
    """The per-axis boundary constexprs, or ``None`` plus the reasons.

    THE CLASSIFICATION IS THE SINGLE POINT OF FAILURE IN THIS FILE, so it is
    derived from the engine's own functions and cross-checked against a second
    one rather than re-derived here:

    * the ghost rule comes from ``stepping._boundary_kinds`` — the fold OUTRANKS
      the declaration there, and a configuration it refuses to resolve (a PML on
      the plane face, a folded axis the layer thinks wraps) raises, which is a
      refusal;
    * a folded axis is ``MIRROR_PERIODIC`` iff ``stepping._stored_past_owned``
      says the stored array carries the slot past MEEP's owned window, and
      ``MIRROR_METALLIC`` otherwise. Same discipline ``coverage.zero_metal_axes``
      follows, and for the same reason: the constexpr and the array path must not
      be able to disagree.
    * ``grid.is_metallic`` must AGREE with that split. It is a second, independent
      route to the same fact (``Grid.stored_cells`` adds its extra slot exactly
      when the axis is mirrored and not metallic), so a disagreement means one of
      them has drifted and neither may be trusted.

    Returned as ``(codes, reasons)`` rather than raising: a plan builder turns a
    refusal into ``None`` and the array path steps the run.
    """
    reasons: List[str] = []
    kinds = _boundary_kinds(grid, pml)
    if kinds is None:
        return None, ("boundary kinds could not be resolved for this grid",)

    owned = getattr(grid, "owned_cells", None)
    stored_past_owned = _stored_past_owned_reader()
    if stored_past_owned is None:
        return None, ("stepping._stored_past_owned is not importable; the "
                      "MIRROR_METALLIC / MIRROR_PERIODIC split cannot be derived",)

    codes: List[int] = []
    for axis, kind in enumerate(kinds):
        if kind == "periodic":
            codes.append(CODE_PERIODIC)
            continue
        if kind == "metallic":
            codes.append(CODE_METALLIC)
            continue
        if kind != "mirror":
            codes.append(CODE_PERIODIC)
            reasons.append(f"axis {axis} boundary {kind!r} is outside "
                           f"{BOUNDARY_KIND_NAMES}")
            continue

        # A folded axis. Everything below is a requirement, not a description.
        if not callable(owned):
            reasons.append(
                f"axis {axis} is folded but the grid exposes no owned_cells(); "
                f"_stored_past_owned would answer False for a folded PERIODIC "
                f"axis and the kernel would drop its top-plane mask")
            codes.append(CODE_PERIODIC)
            continue
        if _call(grid, "mirror_phase", axis, default=None) not in (1, -1):
            reasons.append(
                f"axis {axis} is folded but its mirror phase is "
                f"{_call(grid, 'mirror_phase', axis, default=None)!r}, not +1 or -1")
        stored = _call(grid, "stored_cells", axis, default=None)
        if stored is None or int(stored) <= MIRROR_SOURCE_INDEX:
            reasons.append(
                f"axis {axis} is folded with {stored!r} stored cells; the near "
                f"ghost images stored cell {MIRROR_SOURCE_INDEX}")
        try:
            past_owned = bool(stored_past_owned(grid, axis))
        except Exception as exc:  # noqa: BLE001 - an unanswerable axis is refused
            reasons.append(f"axis {axis}: _stored_past_owned raised {exc!r}")
            codes.append(CODE_PERIODIC)
            continue
        declared_metallic = bool(_call(grid, "is_metallic", axis, default=False))
        if past_owned == declared_metallic:
            reasons.append(
                f"axis {axis}: _stored_past_owned={past_owned} and "
                f"is_metallic={declared_metallic} disagree about the fold's "
                f"termination; the two routes to the same fact have drifted")
        codes.append(CODE_MIRROR_PERIODIC if past_owned else CODE_MIRROR_METALLIC)

    if reasons:
        return None, tuple(reasons)
    return (codes[0], codes[1], codes[2]), ()


def _stored_past_owned_reader():
    """``stepping._stored_past_owned``, or None when ``stepping`` will not import."""
    try:
        from ..stepping import _stored_past_owned  # noqa: PLC0415
    except Exception:  # noqa: BLE001 - no stepping, no coverage
        return None
    return _stored_past_owned


def _shared_grid_reasons(fields: Any, pml: Any, grid: Any) -> List[str]:
    """Everything :func:`coverage._grid_reasons` refuses EXCEPT the fold.

    RE-STATED, not imported-and-subtracted. Subtracting a reason string from
    another predicate's output would make this file's coverage a function of that
    file's phrasing: a clause renamed there would silently widen coverage here.
    Every clause below therefore names its own condition, in the same order and
    for the same reason as the original, with clause 5 (the fold) replaced by
    :func:`folded_axis_kinds` in the caller.
    """
    reasons: List[str] = []

    # 1. CuPy backend. The kernels launch against device pointers.
    xp = getattr(grid, "xp", None)
    if getattr(xp, "__name__", "") != "cupy":
        reasons.append(f"array module is {getattr(xp, '__name__', xp)!r}, not cupy")

    # 2. Real storage. A complex run reads the wrong stride as float32.
    if getattr(fields, "force_complex_fields", False):
        reasons.append("force_complex_fields=True (complex64 storage is not carried)")

    # 3. An absorber that actually absorbs; without one the plain path is the
    #    bit-identical one (stepping._pml_is_active).
    if pml is None or not getattr(pml, "is_active", False):
        reasons.append("no active PML layer (this kernel implements the split-field "
                       "path only)")

    # 4/5. The ghost rules, INCLUDING the two folded ones, are resolved by the
    #      caller through folded_axis_kinds; nothing is decided here.

    # 6. Cartesian only. The cylindrical r axis has its own ghost rule (the
    #    r_to_minus_r image) and its own ownership rule (the axis row belongs to
    #    the per-m rules, not the curl), and a folded r is not a thing this kernel
    #    understands even though `_boundary_kinds` would report "axis".
    if getattr(grid, "cylindrical", False):
        reasons.append("cylindrical (Dcyl) coordinates are not carried")
    for axis in range(3):
        if _call(grid, "is_axis", axis, default=False):
            reasons.append(f"axis {axis} is the cylindrical r = 0 axis")

    # 7. k = 0 on EVERY axis. A Bloch phase needs complex storage and multiplies
    #    one wrapped plane; on a folded axis the engine refuses a nonzero k
    #    outright (driver._require_bloch_is_representable), and the zone-edge
    #    pairing MEEP supports is refused there too — a kernel cannot lift what
    #    the array path will not run.
    if getattr(grid, "has_bloch", False):
        reasons.append(f"nonzero k_point {getattr(grid, 'k_point', None)!r}")
    k_point = getattr(grid, "k_point", (0.0, 0.0, 0.0))
    if any(float(component) != 0.0 for component in k_point):
        reasons.append(f"k_point {tuple(k_point)!r} is not exactly zero")

    # 8. No conductivity on any curl target (that routes to the three-history
    #    conductive-PML recurrence), and none on the magnetic side either.
    reader = getattr(fields, "condfac_for", None)
    if callable(reader):
        for target in CURL_TARGETS:
            if reader(target) is not None:
                reasons.append(f"a conductivity is installed on {target} "
                               f"(mp.Absorber path)")
    if getattr(fields, "has_magnetic_conductivity", False):
        reasons.append("a magnetic (B) conductivity is installed")

    # 10. No instantaneous nonlinearity. Beyond the Pade factor replacing the
    #     constitutive product, a nonlinear folded run with a live far face is
    #     refused by the driver itself (`_require_folded_far_face_is_quiet`,
    #     driver.py:2936): the transverse sums shift component PRODUCTS, which
    #     carry no single parity. Refused here regardless of the far face.
    if getattr(fields, "has_nonlinearity", False):
        reasons.append("chi2/chi3 is installed (the Pade constitutive factor is "
                       "not carried, and a nonlinear fold has no single parity)")

    # 11/12. BFAST adds a second additive curl term; beta adds out-of-plane
    #        couplings. Both are silent additions, not errors.
    if getattr(grid, "bfast_active", False):
        reasons.append("BFAST is active (the second additive curl term is not carried)")
    if float(getattr(grid, "beta", 0.0)) != 0.0:
        reasons.append(f"special_kz beta={getattr(grid, 'beta', 0.0)!r} is nonzero")

    return reasons


def folded_pml_curl_coverage(fields: Any, pml: Any) -> Coverage:
    """May :func:`pml_curl_step_folded` step this (fields, pml) pair?

    Written POSITIVELY: what is admitted is enumerated, and everything else is
    refused. The addition over :func:`coverage.pml_curl_coverage` is exactly one
    clause — a mirror plane on one, two or three axes, each of which must resolve
    to ``MIRROR`` through ``stepping._boundary_kinds`` and classify cleanly
    through :func:`folded_axis_kinds`. Everything the shipped predicate refuses is
    refused here too, re-stated rather than inherited.

    WHAT THIS ADMITS, in one sentence: a real-field Cartesian CuPy run under an
    active split-field PML, at k = 0 on every axis, with no conductivity, no
    nonlinearity, no BFAST, no special_kz, no complex storage, whose susceptibilities
    (if any) are electric lorentzian/drude poles, and with up to three axes folded
    by an even or odd mirror plane over a periodic or metallic outer declaration.

    WHAT IT STILL REFUSES BY NAME, beyond the shipped predicate's list: a
    cylindrical grid (folded or not), a nonzero ``k_point`` on ANY axis
    (``parallel-wvgs-force`` in the corpus), a folded axis with two or fewer
    stored cells, and a folded axis whose two routes to "periodic or metallic"
    disagree. The run-level refusals stay where they are and are inherited, not
    duplicated: an odd source about an even plane is refused at setup by
    ``sources._validate_symmetry_parity``, and monitors in the discarded half are
    handled by the readback path.

    ZERO FOLDED AXES IS ADMITTED, and that is not an oversight. With every axis
    resolving to ``PERIODIC``/``METALLIC`` the kernel's constexpr branches reduce
    to the shipped kernel's exactly, which is a property the gate measures rather
    than a claim: the unfolded rows of the sweep are the shipped kernel's own gate
    re-run through this file. Which of the two a composed plan launches is an
    integration decision, not a coverage one.
    """
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    reasons = _shared_grid_reasons(fields, pml, grid)

    codes, fold_reasons = folded_axis_kinds(
        grid, pml if (pml is not None and getattr(pml, "is_active", False)) else None)
    reasons.extend(fold_reasons)

    # A susceptibility must be one this package understands. Dispersion itself is
    # ADMITTED for the curl — it changes the VALUES update_E writes into the array
    # the curl differences, not an operation this kernel performs.
    reasons.extend(_susceptibility_reasons(fields))

    # Stored E: the invariant behind admitting dispersion at all.
    if not getattr(fields, "stores_E", False):
        reasons.append("E is recomputed from D rather than stored")

    # The six PML auxiliaries and the six curl sources.
    for name in tuple("fu_" + target for target in CURL_TARGETS) + B_SOURCES + D_SOURCES:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")

    # Layout: the flat index arithmetic assumes float32, C-contiguous, grid.shape.
    shape = tuple(getattr(grid, "shape", ()))
    volumes = (tuple(CURL_TARGETS) + tuple("fu_" + t for t in CURL_TARGETS)
               + B_SOURCES + D_SOURCES)
    reasons.extend(_layout_reasons(fields, shape, volumes))
    if pml is not None and getattr(pml, "is_active", False) and len(shape) == 3:
        # The coefficient vectors are built at the STORED extent on a folded axis,
        # which is what makes the kernel's n_a indexing fold-agnostic; this clause
        # is where that stops being an assumption.
        reasons.extend(_coefficient_reasons(pml, shape, ("kms", "sinv"), ("", "_h")))

    if codes is None and not reasons:  # pragma: no cover - belt and braces
        reasons.append("the per-axis boundary codes could not be resolved")
    return Coverage(not reasons, tuple(reasons))


def _has_real_fold(grid: Any) -> bool:
    """Whether this grid really owns a mirror fold, not merely fold-capable rules."""
    return bool(_call(grid, "has_symmetry", default=False)) and any(
        bool(_call(grid, "is_mirrored", axis, default=False)) for axis in range(3))


def folded_grid_active(fields: Any) -> bool:
    """Whether ``fields`` belongs to the composer's folded-grid product family."""
    grid = getattr(fields, "grid", None)
    return grid is not None and _has_real_fold(grid)


def folded_composition_curl_coverage(fields: Any, pml: Any,
                                     sub_step: str) -> Coverage:
    """Folded-curl verdict for composition, where an actual fold is mandatory.

    :func:`folded_pml_curl_coverage` deliberately admits an unfolded grid so its
    standalone gate can prove reduction to the ordinary kernel.  That is not a
    routing rule: selecting between two valid products by branch order would make
    the numerical method depend on composer order.  This narrower verdict is the
    one ``plan_step`` uses.
    """
    if sub_step not in ("step_B", "step_D"):
        raise ValueError("sub_step must be 'step_B' or 'step_D', "
                         f"got {sub_step!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))
    base = folded_pml_curl_coverage(fields, pml)
    reasons = list(base.reasons)
    if not _has_real_fold(grid):
        reasons.append("no mirror plane is active: the folded curl is an "
                       "equivalence product here, not a composition candidate")
    return Coverage(not reasons, tuple(reasons))


def _folded_constitutive_grid_reasons(fields: Any, pml: Any,
                                      grid: Any) -> List[str]:
    """Grid clauses for the element-wise constitutive product on a real fold.

    This is intentionally not :func:`_shared_grid_reasons`: that helper is the
    folded CURL's contract and refuses conductivity because conductivity changes
    the curl recurrence.  Conductivity does not change ``update_H``/``update_E``.
    Re-stating the element-wise contract prevents a curl-only refusal from silently
    narrowing an independent sub-step.
    """
    reasons: List[str] = []
    xp = getattr(grid, "xp", None)
    if getattr(xp, "__name__", "") != "cupy":
        reasons.append(f"array module is {getattr(xp, '__name__', xp)!r}, not cupy")
    if getattr(fields, "force_complex_fields", False):
        reasons.append("force_complex_fields=True (complex64 storage is not carried)")
    if pml is None or not getattr(pml, "is_active", False):
        reasons.append("no active PML layer (this constitutive product implements "
                       "the dsigw path only)")

    codes, fold_reasons = folded_axis_kinds(
        grid, pml if (pml is not None and getattr(pml, "is_active", False)) else None)
    reasons.extend(fold_reasons)
    if not _has_real_fold(grid):
        reasons.append("no mirror plane is active: the folded constitutive product "
                       "has no array-path work to specialize")
    elif codes is not None and not any(
            code in (CODE_MIRROR_METALLIC, CODE_MIRROR_PERIODIC) for code in codes):
        reasons.append("no axis resolves to a mirror boundary")

    if getattr(grid, "cylindrical", False):
        reasons.append("cylindrical (Dcyl) coordinates are not carried")
    for axis in range(3):
        if _call(grid, "is_axis", axis, default=False):
            reasons.append(f"axis {axis} is the cylindrical r = 0 axis")

    if getattr(grid, "has_bloch", False):
        reasons.append(f"nonzero k_point {getattr(grid, 'k_point', None)!r}")
    k_point = getattr(grid, "k_point", (0.0, 0.0, 0.0))
    if any(float(component) != 0.0 for component in k_point):
        reasons.append(f"k_point {tuple(k_point)!r} is not exactly zero")

    if getattr(fields, "has_nonlinearity", False):
        reasons.append("chi2/chi3 is installed (the Pade constitutive factor is "
                       "not carried, and a nonlinear fold has no single parity)")
    if getattr(grid, "bfast_active", False):
        reasons.append("BFAST is active (the second additive curl term is not carried)")
    if float(getattr(grid, "beta", 0.0)) != 0.0:
        reasons.append(f"special_kz beta={getattr(grid, 'beta', 0.0)!r} is nonzero")
    return reasons


def folded_constitutive_coverage(fields: Any, pml: Any, side: str) -> Coverage:
    """Coverage for the ordinary constitutive kernel on a folded stored extent.

    The arithmetic is unchanged from :func:`coverage.constitutive_coverage`; the
    specialized contract exists because that general predicate correctly refuses
    folded grids for the ordinary curl/constitutive product family.  This verdict
    proves the folded extent and coefficient lengths explicitly instead of
    weakening that existing family globally.
    """
    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, "
                         f"got {side!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    spec = CONSTITUTIVE_SIDES[side]
    reasons = _folded_constitutive_grid_reasons(fields, pml, grid)
    reasons.extend(_susceptibility_reasons(fields))
    if side == "E":
        if getattr(fields, "has_polarizations", False) or (
                getattr(fields, "polarizations", ()) or ()):
            reasons.append(
                "a susceptibility is registered: folded dispersive update_E is "
                "not yet composed; its source is (D - sum P), not D")
        if getattr(fields, "has_offdiagonal_epsilon", False):
            reasons.append(
                "an off-diagonal chi1inv row is installed (the row product reads "
                "neighbours; this sub-step is element-wise)")
        if not getattr(fields, "stores_E", False):
            reasons.append("E is recomputed from D rather than stored")

    names = tuple(spec["targets"]) + tuple(spec["aux"]) + tuple(spec["sources"])
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")
    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(_layout_reasons(fields, shape, names))
    if side == "E" and len(shape) == 3:
        reasons.extend(_inverse_epsilon_reasons(fields, shape))
    if pml is not None and getattr(pml, "is_active", False) and len(shape) == 3:
        suffix = ("_h",) if spec["half_integer"] else ("",)
        reasons.extend(_coefficient_reasons(pml, shape, ("kps", "kms"), suffix))
    return Coverage(not reasons, tuple(reasons))


def mirror_ghost_fill_coverage(fields: Any, family: str) -> Coverage:
    """May :func:`mirror_ghost_fill` write this family's mirror ghosts?

    Narrower than the curl's predicate and independent of it: this sub-step reads
    no coefficient, no source and no PML, so it has no Courant number, no
    sub-lattice and no absorber clause. What it does need is a real fold, a
    readable phase, a stored extent that can hold the two planes, and — on a
    folded PERIODIC axis — a reflect row from ``stepping._far_reflect_rows``.

    It is refused entirely on a run with no mirror plane: with nothing folded the
    array path's fill passes return immediately and there is nothing to replace.
    """
    if family not in GHOST_FILL_FAMILIES:
        raise ValueError(f"family must be one of {tuple(GHOST_FILL_FAMILIES)}, "
                         f"got {family!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    reasons: List[str] = []
    xp = getattr(grid, "xp", None)
    if getattr(xp, "__name__", "") != "cupy":
        reasons.append(f"array module is {getattr(xp, '__name__', xp)!r}, not cupy")
    if getattr(fields, "force_complex_fields", False):
        reasons.append("force_complex_fields=True (complex64 storage is not carried)")
    if getattr(grid, "cylindrical", False):
        reasons.append("cylindrical (Dcyl) coordinates are not carried")
    if not bool(_call(grid, "has_symmetry", default=False)):
        reasons.append("no mirror plane is active: the array path's fill passes "
                       "return immediately and there is nothing to replace")

    codes, fold_reasons = folded_axis_kinds(grid, None)
    reasons.extend(fold_reasons)
    if codes is not None and not any(
            code in (CODE_MIRROR_METALLIC, CODE_MIRROR_PERIODIC) for code in codes):
        reasons.append("no axis resolves to a mirror boundary")

    rows = _far_reflect_rows(grid)
    shape = tuple(getattr(grid, "shape", ()))
    if codes is not None:
        for axis, code in enumerate(codes):
            if code != CODE_MIRROR_PERIODIC:
                continue
            row = rows[axis] if rows is not None else None
            if row is None:
                reasons.append(f"axis {axis} is a folded PERIODIC axis with no "
                               f"reflect row from stepping._far_reflect_rows")
            elif len(shape) == 3 and not (0 <= int(row) < int(shape[axis]) - 1):
                reasons.append(
                    f"axis {axis} reflect row {row!r} is outside [0, "
                    f"{int(shape[axis]) - 1}); the far ghost would image the "
                    f"plane it writes, or read out of the allocation")

    names = tuple(GHOST_FILL_FAMILIES[family]["targets"])
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")
    reasons.extend(_layout_reasons(fields, shape, names))
    return Coverage(not reasons, tuple(reasons))


def _far_reflect_rows(grid: Any) -> Optional[Tuple[Optional[int], ...]]:
    """``stepping._far_reflect_rows`` (:1661), or None when it cannot be read.

    Read from the engine, never computed here: the row is ``n_full - stored + 2``,
    which is ``stored - 2`` at an even full count and ``stored - 3`` at an odd one,
    and a second implementation of that is a second place to get it wrong.
    """
    try:
        from ..stepping import _far_reflect_rows as reader  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return None
    try:
        return tuple(reader(grid))
    except Exception:  # noqa: BLE001 - an unanswerable grid is refused, not crashed on
        return None


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------

class FoldedPmlCurlPlan:
    """A launchable, allocation-free curl sub-step on a folded grid.

    Deliberately a sibling of :class:`launch.PmlCurlPlan` rather than a subclass:
    the two hold the same bindings but launch different kernels, and a subclass
    that inherited ``run`` would launch the shipped one. Built two ways —
    :func:`plan_folded_pml_curl` from the engine's own objects, and
    :func:`plan_folded_from_arrays` from bare device arrays for the gate — and
    launched ONE way, so the bytes the gate certifies are the bytes the engine
    would launch.
    """

    __slots__ = ("sub_step", "shape", "n_elem", "dtdx", "backward", "bc", "block",
                 "_targets", "_aux", "_sources", "_coefficients", "_grid", "_kernel")

    def __init__(self, sub_step: str, shape, dtdx: float, bc, block: int,
                 targets, auxiliaries, sources, coefficients, kernel=None) -> None:
        from .launch import SUB_STEPS, CupyPointer, _flat  # noqa: PLC0415

        if sub_step not in SUB_STEPS:
            raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, "
                             f"got {sub_step!r}")
        self.sub_step = sub_step
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.dtdx = float(dtdx)
        self.backward = int(SUB_STEPS[sub_step]["backward"])
        self.bc = tuple(int(code) for code in bc)
        self.block = int(block)
        self._targets = tuple(CupyPointer(a) for a in targets)
        self._aux = tuple(CupyPointer(a) for a in auxiliaries)
        self._sources = tuple(CupyPointer(a) for a in sources)
        self._coefficients = tuple(CupyPointer(_flat(a)) for a in coefficients)
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        # An override for exactly one caller: the gate's source-mutation leg,
        # which compiles a deliberately broken copy of this kernel. Dropping it is
        # not a slowdown, it is a DISARMING — every mutation leg would then launch
        # the shipped kernel and report the defect as uncaught.
        self._kernel = kernel

    def run(self, guard: Optional[bool] = None) -> None:
        """Launch the sub-step, in place. Same ``guard`` contract as every plan here."""
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        nx, ny, nz = self.shape
        kernel = self._kernel if self._kernel is not None else pml_curl_step_folded
        kernel[self._grid](
            *self._targets, *self._aux, *self._sources, *self._coefficients,
            nx, ny, nz, self.n_elem, self.dtdx,
            BACKWARD=self.backward,
            BCX=self.bc[0], BCY=self.bc[1], BCZ=self.bc[2],
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
        )

    def __repr__(self) -> str:
        return (f"FoldedPmlCurlPlan({self.sub_step}, shape={self.shape}, "
                f"bc={self.bc}, block={self.block})")


def plan_folded_pml_curl(fields: Any, pml: Any, sub_step: str,
                         block: Optional[int] = None) -> Optional[FoldedPmlCurlPlan]:
    """Build a folded curl plan from the engine's own objects, or None when refused.

    None is the only refusal: a configuration this kernel does not carry must fall
    back to the array path, never raise into a caller that would have stepped
    correctly.
    """
    from .launch import SUB_STEPS  # noqa: PLC0415

    if sub_step not in SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
    if not folded_pml_curl_coverage(fields, pml).covered:
        return None
    codes, _ = folded_axis_kinds(fields.grid, pml)
    if codes is None:  # pragma: no cover - the predicate already refused
        return None
    kernels = _kernel_module()

    spec = SUB_STEPS[sub_step]
    grid = fields.grid
    return FoldedPmlCurlPlan(
        sub_step, grid.shape, grid.dt / grid.dx, codes,
        kernels.DEFAULT_BLOCK if block is None else block,
        [getattr(fields, n) for n in spec["targets"]],
        [getattr(fields, "fu_" + n) for n in spec["targets"]],
        [getattr(fields, n) for n in spec["sources"]],
        [getattr(pml, f"{stem}_{axis}{spec['suffix']}")
         for axis in "xyz" for stem in ("kms", "sinv")],
    )


def plan_folded_constitutive(fields: Any, pml: Any, side: str,
                             block: Optional[int] = None) -> Optional[Any]:
    """Build the existing constitutive kernel against a validated folded extent.

    The central :func:`launch.plan_constitutive` must keep its ordinary predicate,
    so this builder binds :class:`launch.ConstitutivePlan` directly after the
    fold-specific predicate succeeds.  It does not introduce a second kernel.
    """
    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, "
                         f"got {side!r}")
    if not folded_constitutive_coverage(fields, pml, side).covered:
        return None
    from .kernels import DEFAULT_BLOCK  # noqa: PLC0415
    from .launch import ConstitutivePlan  # noqa: PLC0415

    spec = CONSTITUTIVE_SIDES[side]
    suffix = "_h" if spec["half_integer"] else ""
    return ConstitutivePlan(
        side, fields.grid.shape, DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in spec["targets"]],
        [getattr(fields, name) for name in spec["aux"]],
        [getattr(fields, name) for name in spec["sources"]],
        ([fields.inverse_epsilon_for(name) for name in spec["targets"]]
         if side == "E" else None),
        [getattr(pml, f"{stem}_{axis}{suffix}")
         for axis in "xyz" for stem in ("kps", "kms")],
    )


def plan_folded_from_arrays(sub_step: str, arrays: Dict[str, Any],
                            flat: Dict[str, Any], codes, dtdx: float,
                            block: Optional[int] = None,
                            kernel: Any = None) -> FoldedPmlCurlPlan:
    """Build a folded curl plan from bare device arrays — the gate's route.

    No coverage predicate runs: the caller is a harness that constructed the
    configuration deliberately, including the deliberately wrong ones.
    """
    from .launch import SUB_STEPS  # noqa: PLC0415

    kernels = _kernel_module()
    spec = SUB_STEPS[sub_step]
    shape = tuple(int(n) for n in arrays[spec["targets"][0]].shape)
    return FoldedPmlCurlPlan(
        sub_step, shape, dtdx, codes,
        kernels.DEFAULT_BLOCK if block is None else block,
        [arrays[n] for n in spec["targets"]],
        [arrays["fu_" + n] for n in spec["targets"]],
        [arrays[n] for n in spec["sources"]],
        [flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
        kernel=kernel,
    )


class MirrorGhostFillPlan:
    """One family's mirror ghost fills, as one launch PER FOLDED AXIS.

    Holds a launch per axis in X, Y, Z order — the order
    ``_fill_symmetry_ghost_cells`` applies them in, so a corner unowned on two
    planes carries the product of both parities. ``run`` walks that list; there
    is no single-launch form, deliberately (see :func:`mirror_ghost_fill`).
    """

    __slots__ = ("family", "shape", "block", "axes", "_targets", "_kernel")

    def __init__(self, family: str, shape, block: int, targets,
                 axes: Sequence[Dict[str, Any]], kernel=None) -> None:
        from .launch import CupyPointer  # noqa: PLC0415

        if family not in GHOST_FILL_FAMILIES:
            raise ValueError(f"family must be one of {tuple(GHOST_FILL_FAMILIES)}, "
                             f"got {family!r}")
        self.family = family
        self.shape = tuple(int(n) for n in shape)
        self.block = int(block)
        self._targets = tuple(CupyPointer(a) for a in targets)
        self.axes = tuple(dict(entry) for entry in axes)
        self._kernel = kernel

    def run(self, guard: Optional[bool] = None) -> None:
        """Write every folded axis's ghost planes, in X, Y, Z order, in place."""
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        nx, ny, nz = self.shape
        kernel = self._kernel if self._kernel is not None else mirror_ghost_fill
        for entry in self.axes:
            axis = int(entry["axis"])
            n_plane = (ny * nz, nx * nz, nx * ny)[axis]
            grid = ((n_plane + self.block - 1) // self.block,)
            kernel[grid](
                *self._targets,
                nx, ny, nz, n_plane, int(entry["reflect_row"]),
                AXIS=axis,
                PHASE=int(entry["phase"]),
                FAR=1 if entry["far"] else 0,
                S0=int(entry["shifts"][0]), S1=int(entry["shifts"][1]),
                S2=int(entry["shifts"][2]),
                BLOCK=self.block,
                enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            )

    def __repr__(self) -> str:
        return (f"MirrorGhostFillPlan({self.family}, shape={self.shape}, "
                f"axes={[e['axis'] for e in self.axes]}, block={self.block})")


def ghost_fill_axis_entries(grid: Any, family: str) -> Tuple[Dict[str, Any], ...]:
    """One entry per folded axis, in X, Y, Z order — the kernel's constexprs.

    Public and separate from the plan builder so it can be checked on a machine
    with no Triton and no GPU, which is where the two things that silently go
    wrong here live: the X, Y, Z ORDER (a doubly-unowned corner must end up
    carrying the product of both parities) and the REFLECT ROW (``stored - 2``
    at an even full count, ``stored - 3`` at an odd one).
    """
    codes, reasons = folded_axis_kinds(grid, None)
    if codes is None:
        raise ValueError("this grid's folded axes cannot be classified: "
                         + "; ".join(reasons))
    rows = _far_reflect_rows(grid) or (None, None, None)
    targets = tuple(GHOST_FILL_FAMILIES[family]["targets"])
    entries: List[Dict[str, Any]] = []
    for axis, code in enumerate(codes):  # X, Y, Z — the order the fills apply in.
        if code not in (CODE_MIRROR_METALLIC, CODE_MIRROR_PERIODIC):
            continue
        far = code == CODE_MIRROR_PERIODIC
        entries.append({
            "axis": axis,
            "phase": int(grid.mirror_phase(axis)),
            "far": far,
            # -1 is never read when FAR is 0; it is passed rather than left unset
            # so the kernel signature is one shape.
            "reflect_row": int(rows[axis]) if far and rows[axis] is not None else -1,
            "shifts": tuple(TARGET_IYEE[name][axis] for name in targets),
        })
    return tuple(entries)


def plan_mirror_ghost_fill(fields: Any, family: str,
                           block: Optional[int] = None) -> Optional[MirrorGhostFillPlan]:
    """Build one family's ghost-fill plan from the engine's own objects, or None.

    The per-axis entries carry the phase, the reflect row and the three targets'
    Yee shifts on that axis — everything the kernel needs as a constexpr — read
    from the grid and from ``stepping._far_reflect_rows``, never derived here.
    """
    if not mirror_ghost_fill_coverage(fields, family).covered:
        return None
    kernels = _kernel_module()

    grid = fields.grid
    targets = tuple(GHOST_FILL_FAMILIES[family]["targets"])
    axes = ghost_fill_axis_entries(grid, family)
    if not axes:  # pragma: no cover - the predicate already refused
        return None
    return MirrorGhostFillPlan(
        family, grid.shape, kernels.DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in targets], axes)


def plan_mirror_ghost_fill_from_arrays(family: str, arrays: Dict[str, Any],
                                       axes: Sequence[Dict[str, Any]],
                                       block: Optional[int] = None,
                                       kernel: Any = None) -> MirrorGhostFillPlan:
    """Build a ghost-fill plan from bare device arrays — the gate's route.

    ``axes`` is the per-axis entry list :func:`plan_mirror_ghost_fill` builds; the
    gate supplies it directly so it can hand over a deliberately wrong reflect row
    and watch that be caught.
    """
    kernels = _kernel_module()
    targets = tuple(GHOST_FILL_FAMILIES[family]["targets"])
    shape = tuple(int(n) for n in arrays[targets[0]].shape)
    return MirrorGhostFillPlan(
        family, shape, kernels.DEFAULT_BLOCK if block is None else block,
        [arrays[name] for name in targets], axes, kernel=kernel)


def explain_folded(fields: Any, pml: Any) -> Coverage:
    """The folded-curl coverage verdict with its reasons. Needs no Triton."""
    return folded_pml_curl_coverage(fields, pml)
