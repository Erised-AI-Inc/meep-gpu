"""The FOLDED COMPLEX BETA magnetic seam in one launch: beta ``step_B`` into ``update_H``.

:mod:`.folded_complex_fused_magnetic_pair` WITH THE K3b CURL, and the swap is the
whole module: the certified folded complex magnetic weld — both fills, the ownership
restructure, the deposit-repair bracket — takes
:func:`.folded_complex.folded_beta_bloch_pml_curl_step`'s beta pass exactly where
K3b takes it over K1: two host-rounded scalar words (the ``+i`` magnetic quarter turn
rides in the words), the ``HAS_BETA`` constexpr, and the pass after the curl forms and
before the stores. Nothing else moves, and the suite's
transcription leg asserts that statement by statement against BOTH shipped sources.

It claims the cell ``(folded complex beta PML, folded complex)`` at ``B->H``: the
beta-less twin refuses ``grid.beta`` through K1's clause set, the REAL-storage
:mod:`.folded_beta_fused_magnetic_pair` refuses complex64, and
:mod:`.complex_beta_fused_magnetic_pair` refuses a mirror plane. THE SIBLING IS
:mod:`..metal_kernels.folded_beta_complex_fused_magnetic_pair`, serving the same rows
on Metal today.

===========================================================================
WHAT IT IS WORTH: +3 SEAM-INSTANCES OF 387
===========================================================================

MEASURED, from ``results/fusion_matrix_triton_2026-09-01_foldedbeta``: three
reachable B->H seam-instances in this cell, all scored no-admitting-arm with
``curl_admitters: []``::

    tests:TestEigCoeffs.test_binary_grating_special_kz_0_13_2   fill_B + far ghosts, quiet seam
    tests:TestEigCoeffs.test_binary_grating_special_kz_1_17_7   fill_B + far ghosts, quiet seam
    tests:TestSpecialKz.test_eigsrc_kz_0_complex                fill_B + far ghosts, TWO in-seam magnetic sources

The two binary gratings hold no magnetic source, so on them the source clause is
quiet either way; ``test_eigsrc_kz_0_complex`` declares two in-seam ``'B'`` deposits,
so :data:`CARRIES_DEPOSIT_REPAIR` at False would serve 2 of 3 — the flag buys the
third row, through the SAME shipped bracket the beta-less twin flipped on
2026-08-31.

Everything else — the fill geometry, the parity words, the far-carry ownership move,
the reasons ``update_H`` binds no inverse epsilon — is the beta-less twin's, carried
unchanged; the beta partners are CENTER word pairs no fill touches. ONE admission
where that twin carries two: no beta variant of the off-diagonal arms is shipped, so
an off-diagonal beta row (the corpus has none) is refused by name rather than claimed.

NOT WIRED, and not an arm; importable WITHOUT Triton; DEVICE STATUS below. A weld
licenses a claim, not a dispatch.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .complex_fields import (
    DEFAULT_BLOCK,
    _word_view,
    bloch_phase_table,
    connection_arguments,
)
from .coverage import (
    CONSTITUTIVE_SIDES,
    Coverage,
    MAGNETIC_FIELD_TYPE,
    _call,
    zero_metal_axes,
)
from .folded_complex import (
    CODE_MIRROR_METALLIC,
    CODE_MIRROR_PERIODIC,
    CODE_PERIODIC,
    MIRROR_SOURCE_INDEX,
    TARGET_IYEE,
    _far_reflect_rows,
    _stored_past_owned_reader,
    folded_axis_kinds,
    folded_beta_bloch_pml_curl_coverage,
    folded_complex_constitutive_coverage,
    folded_mirror_ghost_fill_complex_coverage,
    mirror_parity_coefficients,
)
from .launch import SUB_STEPS, CupyPointer, _flat
from .special_kz import (
    beta_curl_coefficients,
)
from .. import deposit_repair as _deposit_repair

#: Does this product bracket its fused launch with the deposit repair? TRUE from the
#: first cut, the beta-less twin's measured position inherited deliberately rather
#: than re-derived: the flag reaches only ``deposit_repair.repairable``, which goes
#: on refusing BY NAME every seam the repair cannot invert, and on a folded seam the
#: repair extends its saved/restored set to every cell the two fills image
#: (``repair_cells``' fold rules, gated by ``probe_triton_folded_deposit_closure``).
#: On THIS cell the flag buys one of three rows: ``test_eigsrc_kz_0_complex``
#: declares TWO in-seam magnetic deposits (the board's ``in_seam_source: true`` at
#: B->H), while the two binary-grating rows hold no magnetic source and the clause
#: is quiet on them either way. The gate's carry legs are what measure the bracket
#: doing work; the flag is only ever changed in the same edit as that wiring.
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
CURL_SUB_STEP = "step_B"
CONSTITUTIVE_SIDE = "H"

#: The FIVE driver call sites ONE launch of this plan performs, in driver order
#: (driver.py:3281-3289) and in ``metal_kernels.coverage.RESIDENCY_ORDER``'s
#: spelling. Declared, never inferred from the slot name. All five do work on an
#: admitted configuration of the two rows that fold a PERIODIC axis; on the other two
#: ``fill_folded_far_ghosts_B`` is inert and is listed anyway, because what a launch
#: REPLACES is what the driver would otherwise have called.
REPLACES: Tuple[str, ...] = ("step_B", "fill_B", "zero_metal_B",
                             "fill_folded_far_ghosts_B", "update_H")

#: ``SUB_STEPS['step_B']['backward']``, restated so the kernel's one legal binding is
#: visible without importing :mod:`launch`; the test suite pins the two equal.
BACKWARD = 0

#: The two codes a folded axis can take. The NEAR fill runs on either; the FAR fill
#: only on ``MIRROR_PERIODIC`` (``stepping._stored_past_owned``:1454-1469), which is
#: the whole of what separates the two carries.
MIRROR_CODES: Tuple[int, int] = (CODE_MIRROR_METALLIC, CODE_MIRROR_PERIODIC)

#: The :data:`.folded_complex.GHOST_FILL_FAMILIES` key this product's fill half is
#: asked about. Spelled out rather than reusing :data:`.coverage.MAGNETIC_FIELD_TYPE`,
#: which happens to be the same string for an unrelated reason (it is a SOURCE's
#: declared field type, not a fill family). ``test_triton_folded_complex_fused_
#: magnetic_pair`` pins this against ``GHOST_FILL_FAMILIES``' own keys.
FILL_FAMILY: str = "B"

#: The complex-multiply device functions this kernel is allowed to call, and the only
#: ones: the partner connection. Every other product -- the beta pass's quarter turn
#: and the parity carries included -- has a real operand and is formed part by part,
#: inline; the test scans the shipped body for any other helper.
MULTIPLY_HELPERS: Tuple[str, ...] = ("_connect",)

#: Which component the NEAR fill images on which axis, for THIS family. Derived from
#: ``TARGET_IYEE`` at import rather than written out, and pinned by test against
#: ``fields.IYEE_SHIFTS``: axis ``a`` fills exactly the components whose Yee shift
#: there is 0, which for B is the single component ``a``.
NEAR_FILL_COMPONENTS: Tuple[Tuple[int, ...], ...] = tuple(
    tuple(index for index, name in enumerate(("Bx", "By", "Bz"))
          if TARGET_IYEE[name][axis] == 0)
    for axis in range(3)
)

#: Which component the FAR fill images on which axis — the exact complement
#: (``stepping._fill_folded_far_ghosts`` :1516-1534 writes axis ``a`` for every
#: component whose Yee shift there is ONE), derived from the same table.
FAR_FILL_COMPONENTS: Tuple[Tuple[int, ...], ...] = tuple(
    tuple(index for index, name in enumerate(("Bx", "By", "Bz"))
          if TARGET_IYEE[name][axis] == 1)
    for axis in range(3)
)

#: What has and has not been executed on a device. Edited only by a released gate.
DEVICE_STATUS: str = (
    "NOT YET RUN. This module was created 2026-09-02 (the fusion-residue round, "
    "audit §1.4) and no device has executed it; its gate is "
    "parity/meep_gpu/probe_triton_folded_beta_complex_fused_magnetic_pair.py, and only that "
    "gate's released artifact may edit this constant. Until it runs, this module "
    "licenses NOTHING: the predicate answers on the laptop, the plan builder "
    "answers None without Triton, and no byte-identity claim exists anywhere in "
    "this file.")

__all__ = [
    "BACKWARD", "CONSTITUTIVE_SIDE", "CURL_SUB_STEP", "DEVICE_STATUS",
    "FAR_FILL_COMPONENTS", "FILL_FAMILY",
    "MULTIPLY_HELPERS", "MIRROR_CODES",
    "NEAR_FILL_COMPONENTS", "REPLACES",
    "FoldedBetaComplexFusedMagneticPairPlan",
    "carried_destinations",
    "folded_beta_complex_fused_curl_constitutive_B",
    "folded_beta_complex_fused_curl_constitutive_B_kernel",
    "folded_beta_complex_fused_magnetic_pair_coverage",
    "parity_chain",
    "parity_coefficient_words",
    "plan_folded_beta_complex_fused_magnetic_pair",
    "plan_folded_beta_complex_fused_magnetic_pair_from_arrays",
]


# ---------------------------------------------------------------------------
# The ownership rule and the parity chain — host side, so they can be READ
# ---------------------------------------------------------------------------

def carried_destinations(near: Sequence[int], far: Sequence[int]
                         ) -> Tuple[Tuple[Tuple[int, ...], bool], ...]:
    """Every ghost cell ONE source lane owns, as ``(far axes at top, near?)``.

    THE OWNERSHIP RULE. The driver runs the near fill, the wall clear and the far fill
    in that order (driver.py:3285-3287) and the far fill is applied axis by axis, so a
    cell the fills leave at the top of SEVERAL folded periodic axes is written more
    than once. Back-substituting every pass gives ONE source cell per ghost — the cell
    at stored :data:`.folded_complex.MIRROR_SOURCE_INDEX` on the near axis and at the
    reflect row on each far axis — so one lane computes the displacement every one of
    those ghosts carries and writes them all.

    Identical in SHAPE to :func:`.folded_fused_magnetic_pair`'s enumeration and
    deliberately NOT identical in what it returns: that family composes the parities
    into one compile-time sign, which complex storage does not permit. This returns
    the CELLS only; :func:`parity_chain` returns the ORDERED applications separately.

    Returned smallest-subset-first for a stable emission order; the order the blocks
    are emitted in is unobservable (the cells are distinct) and a stable one keeps a
    source diff readable.
    """
    far = tuple(int(axis) for axis in far)
    combos: List[Tuple[Tuple[int, ...], bool]] = []
    for size in range(len(far) + 1):
        for subset in itertools.combinations(far, size):
            for carries_near in ((False, True) if near else (False,)):
                if not subset and not carries_near:
                    continue  # the lane's OWN cell, not a ghost
                combos.append((subset, carries_near))
    return tuple(sorted(combos, key=lambda item: (len(item[0]), item[0], item[1])))


def parity_chain(near: Sequence[int], subset: Sequence[int], carries_near: bool
                 ) -> Tuple[Tuple[int, str], ...]:
    """The ORDERED ``(axis, pass)`` applications one ghost's value carries.

    THE ORDER IS THE DRIVER'S AND IS TRANSCRIBED, NEVER CHOSEN — see the module
    docstring. Each pass reads the plane the previous one wrote, so back-substituting
    a corner gives ``c_far[a2] (x) ( c_far[a1] (x) ( c_near[n] (x) v ) )`` with
    ``a1 < a2``: the near application INNERMOST, then the far axes ascending.

    A single-entry chain is the common case and the only one the two corpus rows this
    carry admits ever reach — ``special_kz_2_21_2`` and ``triangular_lattice_oblique``
    each fold ONE axis. The longer chains are reachable configuration space all the
    same (two folded periodic axes give a two-entry chain and three give a
    three-entry one) and the gate carries those cases rather than leaving the
    ordering untested because the corpus does not press on it.
    """
    chain: List[Tuple[int, str]] = []
    if carries_near:
        if not near:
            raise AssertionError(
                "a destination that carries the near fill was enumerated for a "
                "component with no near axis; carried_destinations and the near-fill "
                "axis table have drifted")
        chain.append((int(near[0]), "near"))
    previous = -1
    for axis in subset:
        axis = int(axis)
        if axis <= previous:
            raise AssertionError(
                f"the far axis subset {tuple(subset)!r} is not strictly ascending; "
                f"stepping._fill_folded_far_ghosts applies its axes in ascending "
                f"order (:1518) and this chain transcribes that order")
        previous = axis
        chain.append((axis, "far"))
    return tuple(chain)


def parity_coefficient_words(grid: Any) -> Tuple[Tuple[float, float], ...]:
    """The six ``(re, im)`` word pairs the kernel takes — near x/y/z then far x/y/z.

    Every pair is :func:`.folded_complex.mirror_parity_coefficients`' output for that
    axis's declared phase, HOST-ROUNDED through ``numpy.complex64`` exactly once, and
    ``(+0.0, +0.0)`` on an unfolded axis, where no block that reads it is emitted.

    A zero pair on a FOLDED axis would be a whole plane of exact zeros — loud —
    rather than a plausible field, which is the same choice
    :func:`.folded_fused_magnetic_pair.mirror_phases` makes for its constexpr.
    """
    near: List[Tuple[float, float]] = []
    far: List[Tuple[float, float]] = []
    for axis in range(3):
        if not bool(_call(grid, "is_mirrored", axis, default=False)):
            near.append((0.0, 0.0))
            far.append((0.0, 0.0))
            continue
        phase = _call(grid, "mirror_phase", axis, default=None)
        if phase not in (1, -1):
            near.append((0.0, 0.0))
            far.append((0.0, 0.0))
            continue
        near_words, far_words = mirror_parity_coefficients(int(phase))
        near.append(near_words)
        far.append(far_words)
    return tuple(near) + tuple(far)


if triton is not None:

    # The forms, the fold ladder and the chunk record's word offsets, under the names
    # the folded pair binds them; its ghost carries are its own device functions.
    from .complex_fields import _connect  # noqa: E402
    from .forms import (axis_present, constitutive_form, curl_form,  # noqa: E402
                        curl_stencil, fold_ladder, record_in)
    from .kernels import RECORD_HALF_INTEGER, RECORD_PRESENCE  # noqa: E402
    from .folded_complex_fused_magnetic_pair import _carry_ghost_complex  # noqa: E402 - shared

    PERIODIC = tl.constexpr(CODE_PERIODIC)
    MIRROR_METALLIC = tl.constexpr(CODE_MIRROR_METALLIC)
    MIRROR_PERIODIC = tl.constexpr(CODE_MIRROR_PERIODIC)


    @triton.jit(do_not_specialize=["start_h"])
    def folded_beta_complex_fused_curl_constitutive_B(
        f0, f1, f2,                       # curl targets: Bx,By,Bz (complex64 as words)
        u0, u1, u2,                       # curl auxiliaries: fu_Bx,fu_By,fu_Bz
        g0, g1, g2,                       # curl sources: Ex,Ey,Ez
        kmx, sinvx, kmy, sinvy, kmz, sinvz,   # curl kms/sinv, HALF-INTEGER lattice
        h0, h1, h2,                       # constitutive targets: Hx,Hy,Hz
        w0, w1, w2,                       # constitutive aux: f_w_Hx,f_w_Hy,f_w_Hz
        kp0, km0, kp1, km1, kp2, km2,     # constitutive kps/kms, INTEGER lattice
        cb,                               # chunk record: both halves' forms
        nx, ny, nz, n_elem, dtdx,         # n_elem = COMPLEX cells; dtdx pre-rounded
        cxr, cxi, cyr, cyi, czr, czi,     # per-axis connection factor, float32 parts
        rx, ry, rz,                       # RUNTIME: stepping._far_reflect_rows
        n0r, n1r, n2r,                    # NEAR parity words, per axis (+phase)
        d0r, d1r, d2r,                    # FAR parity words, per axis (-phase)
        start_h,                          # RUNTIME: MEEP's start of H
        beta_plus, beta_minus,            # f32 quarter-turn words per sign (B: +c, D: -c)
        BACKWARD: tl.constexpr,           # bound to 0 by every builder
        BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
        NEAR_X: tl.constexpr, NEAR_Y: tl.constexpr, NEAR_Z: tl.constexpr,
        FAR_X: tl.constexpr, FAR_Y: tl.constexpr, FAR_Z: tl.constexpr,
        ZM_X: tl.constexpr, ZM_Y: tl.constexpr, ZM_Z: tl.constexpr,
        CNX: tl.constexpr, CNY: tl.constexpr, CNZ: tl.constexpr,
        HAS_BETA: tl.constexpr,           # 1 on every admitted configuration
        BLOCK: tl.constexpr,
    ):
        """Folded complex beta ``step_B`` + the fills + ``zero_metal_B`` + ``update_H``, one launch.

        ``folded_complex_fused_magnetic_pair.folded_complex_fused_curl_constitutive_B``'s body, character for character, plus the
        per-part beta pass of ``special_kz.beta_bloch_pml_curl_step``
        after the curl forms and before the wall clear and the ghost carries -- the array
        path's order (the curl, its beta pass, then the seam passes, then update_H). The
        pass's increment carries the folded curl's two ownership blocks, so a cell the
        curl leaves to a fill keeps its word through the pass, and every carried ghost
        images the registers the pass leaves. The carries are the folded pair's own
        device functions. ``HAS_BETA = 0`` builds ``folded_complex_fused_curl_constitutive_B`` exactly.
        """
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = idx < n_elem
        nyz = ny * nz
        k = idx % nz
        plane = idx // nz
        j = plane % ny
        i = plane // ny

        # --- DELTA 1: the ghost rule, per axis (stepping._shift_up / _shift_down) ---
        # PERIODIC wraps (a plain roll, phased or not); EVERY other rule here serves an
        # exact (+0.0, +0.0) past the face, which `tl.load`'s `other=` delivers on both
        # words. On a folded axis that zero stands in for a value nothing reads (the two
        # masks below zero every cell that reads it).
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

        # --- loads: two words per operand ------------------------------------------
        a_re = tl.load(g0 + 2 * idx, mask=live, other=0.0)
        a_im = tl.load(g0 + 2 * idx + 1, mask=live, other=0.0)
        b_re = tl.load(g1 + 2 * idx, mask=live, other=0.0)
        b_im = tl.load(g1 + 2 * idx + 1, mask=live, other=0.0)
        c_re = tl.load(g2 + 2 * idx, mask=live, other=0.0)
        c_im = tl.load(g2 + 2 * idx + 1, mask=live, other=0.0)
        a_y_re = tl.load(g0 + 2 * oy, mask=vy, other=0.0)
        a_y_im = tl.load(g0 + 2 * oy + 1, mask=vy, other=0.0)
        a_z_re = tl.load(g0 + 2 * oz, mask=vz, other=0.0)
        a_z_im = tl.load(g0 + 2 * oz + 1, mask=vz, other=0.0)
        b_x_re = tl.load(g1 + 2 * ox, mask=vx, other=0.0)
        b_x_im = tl.load(g1 + 2 * ox + 1, mask=vx, other=0.0)
        b_z_re = tl.load(g1 + 2 * oz, mask=vz, other=0.0)
        b_z_im = tl.load(g1 + 2 * oz + 1, mask=vz, other=0.0)
        c_x_re = tl.load(g2 + 2 * ox, mask=vx, other=0.0)
        c_x_im = tl.load(g2 + 2 * ox + 1, mask=vx, other=0.0)
        c_y_re = tl.load(g2 + 2 * oy, mask=vy, other=0.0)
        c_y_im = tl.load(g2 + 2 * oy + 1, mask=vy, other=0.0)

        # --- the connection on the UNSHIFTED partners, plane 0 of each partner's axis
        # Target 0 reads c along y and b along z, target 1 a along z and c along x,
        # target 2 b along x and a along y (stepping's term table). `tl.where` is a
        # bitwise select, so every cell off plane 0 keeps its loaded words.
        at_x, at_y, at_z = i == 0, j == 0, k == 0
        c0_re, c0_im = c_re, c_im          # target 0's g1, connected on y
        b0_re, b0_im = b_re, b_im          # target 0's g2, connected on z
        a1_re, a1_im = a_re, a_im          # target 1's g1, connected on z
        c1_re, c1_im = c_re, c_im          # target 1's g2, connected on x
        b2_re, b2_im = b_re, b_im          # target 2's g1, connected on x
        a2_re, a2_im = a_re, a_im          # target 2's g2, connected on y
        if CNX:
            r_re, r_im = _connect(c_re, c_im, cxr, cxi, CNX)
            c1_re = tl.where(at_x, r_re, c_re)
            c1_im = tl.where(at_x, r_im, c_im)
            r_re, r_im = _connect(b_re, b_im, cxr, cxi, CNX)
            b2_re = tl.where(at_x, r_re, b_re)
            b2_im = tl.where(at_x, r_im, b_im)
        if CNY:
            r_re, r_im = _connect(c_re, c_im, cyr, cyi, CNY)
            c0_re = tl.where(at_y, r_re, c_re)
            c0_im = tl.where(at_y, r_im, c_im)
            r_re, r_im = _connect(a_re, a_im, cyr, cyi, CNY)
            a2_re = tl.where(at_y, r_re, a_re)
            a2_im = tl.where(at_y, r_im, a_im)
        if CNZ:
            r_re, r_im = _connect(b_re, b_im, czr, czi, CNZ)
            b0_re = tl.where(at_z, r_re, b_re)
            b0_im = tl.where(at_z, r_im, b_im)
            r_re, r_im = _connect(a_re, a_im, czr, czi, CNZ)
            a1_re = tl.where(at_z, r_re, a_re)
            a1_im = tl.where(at_z, r_im, a_im)

        # --- the curl (stepping._curl_from_operands), per part ---------------------
        # kernels.pml_curl_step's three calls on each part: MEEP's left-to-right
        # association with both partners, the one-partner form where the record says the
        # other partner's derivative axis is invariant, an exact +0 with neither. The
        # dtdx scale is the real scalar times each part (part_products.scale).
        presence = tl.load(cb + RECORD_PRESENCE)
        px = axis_present(presence, 0)
        py = axis_present(presence, 1)
        pz = axis_present(presence, 2)
        curl0_re = curl_stencil(c_y_re, c0_re, b0_re, b_z_re, dtdx, py, pz)
        curl0_im = curl_stencil(c_y_im, c0_im, b0_im, b_z_im, dtdx, py, pz)
        curl1_re = curl_stencil(a_z_re, a1_re, c1_re, c_x_re, dtdx, pz, px)
        curl1_im = curl_stencil(a_z_im, a1_im, c1_im, c_x_im, dtdx, pz, px)
        curl2_re = curl_stencil(b_x_re, b2_re, a2_re, a_y_re, dtdx, px, py)
        curl2_im = curl_stencil(b_x_im, b2_im, a2_im, a_y_im, dtdx, px, py)

        # --- DELTA 2: ownership mask, cell 0 (stepping._mask_non_owned_cells) -------
        # Every target whose Yee shift is 0 on a non-periodic axis (``is_mirrored or
        # is_metallic``): the unfolded body's ``== METALLIC`` widened to ``!= PERIODIC``.
        # Writes +0.0 to BOTH parts -- the array path assigns complex zero.
        if BACKWARD:
            if BCY != PERIODIC:
                curl0_re = tl.where(at_y, 0.0, curl0_re)
                curl0_im = tl.where(at_y, 0.0, curl0_im)
            if BCZ != PERIODIC:
                curl0_re = tl.where(at_z, 0.0, curl0_re)
                curl0_im = tl.where(at_z, 0.0, curl0_im)
            if BCX != PERIODIC:
                curl1_re = tl.where(at_x, 0.0, curl1_re)
                curl1_im = tl.where(at_x, 0.0, curl1_im)
            if BCZ != PERIODIC:
                curl1_re = tl.where(at_z, 0.0, curl1_re)
                curl1_im = tl.where(at_z, 0.0, curl1_im)
            if BCX != PERIODIC:
                curl2_re = tl.where(at_x, 0.0, curl2_re)
                curl2_im = tl.where(at_x, 0.0, curl2_im)
            if BCY != PERIODIC:
                curl2_re = tl.where(at_y, 0.0, curl2_re)
                curl2_im = tl.where(at_y, 0.0, curl2_im)
        else:
            if BCX != PERIODIC:
                curl0_re = tl.where(at_x, 0.0, curl0_re)
                curl0_im = tl.where(at_x, 0.0, curl0_im)
            if BCY != PERIODIC:
                curl1_re = tl.where(at_y, 0.0, curl1_re)
                curl1_im = tl.where(at_y, 0.0, curl1_im)
            if BCZ != PERIODIC:
                curl2_re = tl.where(at_z, 0.0, curl2_re)
                curl2_im = tl.where(at_z, 0.0, curl2_im)

        # --- DELTA 3: ownership mask, the TOP plane of a folded PERIODIC axis --------
        # Every target whose Yee shift is 1 there: that slot sits past MEEP's owned
        # window and the far fill, not the curl, writes it (symmetry.py's block, both
        # parts).
        last_x, last_y, last_z = i == nx - 1, j == ny - 1, k == nz - 1
        if BACKWARD:
            # Dx:(1,0,0)  Dy:(0,1,0)  Dz:(0,0,1) -- shift 1 on its OWN axis only.
            if BCX == MIRROR_PERIODIC:
                curl0_re = tl.where(last_x, 0.0, curl0_re)
                curl0_im = tl.where(last_x, 0.0, curl0_im)
            if BCY == MIRROR_PERIODIC:
                curl1_re = tl.where(last_y, 0.0, curl1_re)
                curl1_im = tl.where(last_y, 0.0, curl1_im)
            if BCZ == MIRROR_PERIODIC:
                curl2_re = tl.where(last_z, 0.0, curl2_re)
                curl2_im = tl.where(last_z, 0.0, curl2_im)
        else:
            # Bx:(0,1,1)  By:(1,0,1)  Bz:(1,1,0) -- shift 1 on the two OTHER axes.
            if BCY == MIRROR_PERIODIC:
                curl0_re = tl.where(last_y, 0.0, curl0_re)
                curl0_im = tl.where(last_y, 0.0, curl0_im)
            if BCZ == MIRROR_PERIODIC:
                curl0_re = tl.where(last_z, 0.0, curl0_re)
                curl0_im = tl.where(last_z, 0.0, curl0_im)
            if BCX == MIRROR_PERIODIC:
                curl1_re = tl.where(last_x, 0.0, curl1_re)
                curl1_im = tl.where(last_x, 0.0, curl1_im)
            if BCZ == MIRROR_PERIODIC:
                curl1_re = tl.where(last_z, 0.0, curl1_re)
                curl1_im = tl.where(last_z, 0.0, curl1_im)
            if BCX == MIRROR_PERIODIC:
                curl2_re = tl.where(last_x, 0.0, curl2_re)
                curl2_im = tl.where(last_x, 0.0, curl2_im)
            if BCY == MIRROR_PERIODIC:
                curl2_re = tl.where(last_y, 0.0, curl2_re)
                curl2_im = tl.where(last_y, 0.0, curl2_im)

        # --- ownership of the fill destinations (folded_fused_magnetic_pair's) -----
        # Component m is a NEAR destination at stored cell 0 of axis m and a FAR
        # destination at the top plane of each OTHER folded periodic axis (Bx (0,1,1),
        # By (1,0,1), Bz (1,1,0)). Those lanes do NOT load or store B, H or f_w_H, either
        # part: the source lane writes all three for them, so no lane reads a word
        # another lane writes.
        own0, own1, own2 = live, live, live
        if NEAR_X:
            own0 = own0 & (i != 0)
        if FAR_Y:
            own0 = own0 & (j != ny - 1)
        if FAR_Z:
            own0 = own0 & (k != nz - 1)
        if NEAR_Y:
            own1 = own1 & (j != 0)
        if FAR_X:
            own1 = own1 & (i != nx - 1)
        if FAR_Z:
            own1 = own1 & (k != nz - 1)
        if NEAR_Z:
            own2 = own2 & (k != 0)
        if FAR_X:
            own2 = own2 & (i != nx - 1)
        if FAR_Y:
            own2 = own2 & (j != ny - 1)

        # --- the PML forms (stepping._apply_curl_regions), per part ----------------
        # dsig/dsigu cycle: target 0 -> (y, z), 1 -> (z, x), 2 -> (x, y). One record
        # select per target serves both parts; fu is read and written on the
        # recurrence's cells only (the auxiliary contract), both words.
        km_x = tl.load(kmx + i, mask=live, other=0.0)
        si_x = tl.load(sinvx + i, mask=live, other=0.0)
        km_y = tl.load(kmy + j, mask=live, other=0.0)
        si_y = tl.load(sinvy + j, mask=live, other=0.0)
        km_z = tl.load(kmz + k, mask=live, other=0.0)
        si_z = tl.load(sinvz + k, mask=live, other=0.0)

        if BACKWARD:
            lattice = 0
        else:
            lattice = RECORD_HALF_INTEGER
        du0 = record_in(cb, lattice + 8, k)
        du1 = record_in(cb, lattice + 0, i)
        du2 = record_in(cb, lattice + 4, j)

        p_re = tl.load(u0 + 2 * idx, mask=live & du0, other=0.0)
        p_im = tl.load(u0 + 2 * idx + 1, mask=live & du0, other=0.0)
        v0_re, n0_re = curl_form(tl.load(f0 + 2 * idx, mask=own0, other=0.0), p_re, curl0_re,
                                 km_y, si_y, km_z, si_z, du0)
        v0_im, n0_im = curl_form(tl.load(f0 + 2 * idx + 1, mask=own0, other=0.0), p_im,
                                 curl0_im, km_y, si_y, km_z, si_z, du0)

        p_re = tl.load(u1 + 2 * idx, mask=live & du1, other=0.0)
        p_im = tl.load(u1 + 2 * idx + 1, mask=live & du1, other=0.0)
        v1_re, n1_re = curl_form(tl.load(f1 + 2 * idx, mask=own1, other=0.0), p_re, curl1_re,
                                 km_z, si_z, km_x, si_x, du1)
        v1_im, n1_im = curl_form(tl.load(f1 + 2 * idx + 1, mask=own1, other=0.0), p_im,
                                 curl1_im, km_z, si_z, km_x, si_x, du1)

        p_re = tl.load(u2 + 2 * idx, mask=live & du2, other=0.0)
        p_im = tl.load(u2 + 2 * idx + 1, mask=live & du2, other=0.0)
        v2_re, n2_re = curl_form(tl.load(f2 + 2 * idx, mask=own2, other=0.0), p_re, curl2_re,
                                 km_x, si_x, km_y, si_y, du2)
        v2_im, n2_im = curl_form(tl.load(f2 + 2 * idx + 1, mask=own2, other=0.0), p_im,
                                 curl2_im, km_x, si_x, km_y, si_y, du2)

        # --- the special-kz beta pass (stepping._special_kz_beta_term, then
        # stepping._apply_folded_increment), per part ------------------------------
        # MEEP's pass of its own after every component's curl (step_db.cpp:161-175),
        # applied to the registers the forms returned. The partners are the CENTER
        # words as loaded, never connected: target 0 takes the second source at sign
        # +1, target 1 the first at sign -1; target 2 gets nothing. The increment is
        # the negated quarter turn (part_products.quarter_turn): i*c*(x + iy) is
        # (-(c*y), c*x), so its negation is (c*y, -(c*x)), the two negations of the
        # real part cancelling exactly. The words beta_plus/beta_minus already carry
        # the side's i or -i (special_kz.beta_curl_coefficients). Masked to (+0, +0)
        # on the cells the curl's own mask drops; each part takes the beta ladder.
        if HAS_BETA:
            q0_re = beta_plus * b_im
            q0_im = (beta_plus * b_re) * -1.0
            q1_re = beta_minus * a_im
            q1_im = (beta_minus * a_re) * -1.0
            if BACKWARD:
                if BCY != PERIODIC:
                    q0_re = tl.where(at_y, 0.0, q0_re)
                    q0_im = tl.where(at_y, 0.0, q0_im)
                if BCZ != PERIODIC:
                    q0_re = tl.where(at_z, 0.0, q0_re)
                    q0_im = tl.where(at_z, 0.0, q0_im)
                if BCX != PERIODIC:
                    q1_re = tl.where(at_x, 0.0, q1_re)
                    q1_im = tl.where(at_x, 0.0, q1_im)
                if BCZ != PERIODIC:
                    q1_re = tl.where(at_z, 0.0, q1_re)
                    q1_im = tl.where(at_z, 0.0, q1_im)
                if BCX == MIRROR_PERIODIC:
                    q0_re = tl.where(last_x, 0.0, q0_re)
                    q0_im = tl.where(last_x, 0.0, q0_im)
                if BCY == MIRROR_PERIODIC:
                    q1_re = tl.where(last_y, 0.0, q1_re)
                    q1_im = tl.where(last_y, 0.0, q1_im)
            else:
                if BCX != PERIODIC:
                    q0_re = tl.where(at_x, 0.0, q0_re)
                    q0_im = tl.where(at_x, 0.0, q0_im)
                if BCY != PERIODIC:
                    q1_re = tl.where(at_y, 0.0, q1_re)
                    q1_im = tl.where(at_y, 0.0, q1_im)
                if BCY == MIRROR_PERIODIC:
                    q0_re = tl.where(last_y, 0.0, q0_re)
                    q0_im = tl.where(last_y, 0.0, q0_im)
                if BCZ == MIRROR_PERIODIC:
                    q0_re = tl.where(last_z, 0.0, q0_re)
                    q0_im = tl.where(last_z, 0.0, q0_im)
                if BCX == MIRROR_PERIODIC:
                    q1_re = tl.where(last_x, 0.0, q1_re)
                    q1_im = tl.where(last_x, 0.0, q1_im)
                if BCZ == MIRROR_PERIODIC:
                    q1_re = tl.where(last_z, 0.0, q1_re)
                    q1_im = tl.where(last_z, 0.0, q1_im)
            v0_re, n0_re = fold_ladder(v0_re, n0_re, q0_re, si_y, si_z, du0)
            v0_im, n0_im = fold_ladder(v0_im, n0_im, q0_im, si_y, si_z, du0)
            v1_re, n1_re = fold_ladder(v1_re, n1_re, q1_re, si_z, si_x, du1)
            v1_im, n1_im = fold_ladder(v1_im, n1_im, q1_im, si_z, si_x, du1)

        # --- the seam: stepping.zero_metal_B, on the registers -----------------------
        # Before both the store and the constitutive read. A walled axis is never a folded
        # axis, so these lines cannot touch a fill destination.
        if ZM_X:
            v0_re = tl.where(at_x, 0.0, v0_re)
            v0_im = tl.where(at_x, 0.0, v0_im)
        if ZM_Y:
            v1_re = tl.where(at_y, 0.0, v1_re)
            v1_im = tl.where(at_y, 0.0, v1_im)
        if ZM_Z:
            v2_re = tl.where(at_z, 0.0, v2_re)
            v2_im = tl.where(at_z, 0.0, v2_im)

        # --- stores: fu on the recurrence's cells, the fields on owned cells -------
        tl.store(u0 + 2 * idx, n0_re, mask=live & du0)
        tl.store(u0 + 2 * idx + 1, n0_im, mask=live & du0)
        tl.store(u1 + 2 * idx, n1_re, mask=live & du1)
        tl.store(u1 + 2 * idx + 1, n1_im, mask=live & du1)
        tl.store(u2 + 2 * idx, n2_re, mask=live & du2)
        tl.store(u2 + 2 * idx + 1, n2_im, mask=live & du2)
        tl.store(f0 + 2 * idx, v0_re, mask=own0)
        tl.store(f0 + 2 * idx + 1, v0_im, mask=own0)
        tl.store(f1 + 2 * idx, v1_re, mask=own1)
        tl.store(f1 + 2 * idx + 1, v1_im, mask=own1)
        tl.store(f2 + 2 * idx, v2_re, mask=own2)
        tl.store(f2 + 2 * idx + 1, v2_im, mask=own2)

        # ==================== the constitutive half (update_H) ======================
        # bloch_constitutive_step's H arm per part, `src` the register pair the curl half
        # produced and `live` the component's ownership; dsigw lines on the integer
        # lattice; MEEP's start of H.
        kp_0 = tl.load(kp0 + i, mask=live, other=0.0)
        km_0 = tl.load(km0 + i, mask=live, other=0.0)
        kp_1 = tl.load(kp1 + j, mask=live, other=0.0)
        km_1 = tl.load(km1 + j, mask=live, other=0.0)
        kp_2 = tl.load(kp2 + k, mask=live, other=0.0)
        km_2 = tl.load(km2 + k, mask=live, other=0.0)
        dw0 = record_in(cb, 0, i)
        dw1 = record_in(cb, 4, j)
        dw2 = record_in(cb, 8, k)
        start = start_h != 0

        # --- component 0 ---------------------------------------------------------
        prev_re = tl.load(w0 + 2 * idx, mask=own0 & dw0, other=0.0)   # BEFORE the store.
        prev_im = tl.load(w0 + 2 * idx + 1, mask=own0 & dw0, other=0.0)
        tl.store(w0 + 2 * idx, v0_re, mask=own0 & dw0)
        tl.store(w0 + 2 * idx + 1, v0_im, mask=own0 & dw0)
        a_re = tl.load(h0 + 2 * idx, mask=own0 & dw0, other=0.0)
        a_im = tl.load(h0 + 2 * idx + 1, mask=own0 & dw0, other=0.0)
        prev_re = tl.where(start, v0_re, prev_re)    # H starts as B (start_h)
        prev_im = tl.where(start, v0_im, prev_im)
        a_re = tl.where(start, v0_re, a_re)
        a_im = tl.where(start, v0_im, a_im)
        tl.store(h0 + 2 * idx, constitutive_form(a_re, prev_re, v0_re, kp_0, km_0, dw0),
                 mask=own0)
        tl.store(h0 + 2 * idx + 1,
                 constitutive_form(a_im, prev_im, v0_im, kp_0, km_0, dw0), mask=own0)

        # --- component 1 ---------------------------------------------------------
        prev_re = tl.load(w1 + 2 * idx, mask=own1 & dw1, other=0.0)   # BEFORE the store.
        prev_im = tl.load(w1 + 2 * idx + 1, mask=own1 & dw1, other=0.0)
        tl.store(w1 + 2 * idx, v1_re, mask=own1 & dw1)
        tl.store(w1 + 2 * idx + 1, v1_im, mask=own1 & dw1)
        a_re = tl.load(h1 + 2 * idx, mask=own1 & dw1, other=0.0)
        a_im = tl.load(h1 + 2 * idx + 1, mask=own1 & dw1, other=0.0)
        prev_re = tl.where(start, v1_re, prev_re)    # H starts as B (start_h)
        prev_im = tl.where(start, v1_im, prev_im)
        a_re = tl.where(start, v1_re, a_re)
        a_im = tl.where(start, v1_im, a_im)
        tl.store(h1 + 2 * idx, constitutive_form(a_re, prev_re, v1_re, kp_1, km_1, dw1),
                 mask=own1)
        tl.store(h1 + 2 * idx + 1,
                 constitutive_form(a_im, prev_im, v1_im, kp_1, km_1, dw1), mask=own1)

        # --- component 2 ---------------------------------------------------------
        prev_re = tl.load(w2 + 2 * idx, mask=own2 & dw2, other=0.0)   # BEFORE the store.
        prev_im = tl.load(w2 + 2 * idx + 1, mask=own2 & dw2, other=0.0)
        tl.store(w2 + 2 * idx, v2_re, mask=own2 & dw2)
        tl.store(w2 + 2 * idx + 1, v2_im, mask=own2 & dw2)
        a_re = tl.load(h2 + 2 * idx, mask=own2 & dw2, other=0.0)
        a_im = tl.load(h2 + 2 * idx + 1, mask=own2 & dw2, other=0.0)
        prev_re = tl.where(start, v2_re, prev_re)    # H starts as B (start_h)
        prev_im = tl.where(start, v2_im, prev_im)
        a_re = tl.where(start, v2_re, a_re)
        a_im = tl.where(start, v2_im, a_im)
        tl.store(h2 + 2 * idx, constitutive_form(a_re, prev_re, v2_re, kp_2, km_2, dw2),
                 mask=own2)
        tl.store(h2 + 2 * idx + 1,
                 constitutive_form(a_im, prev_im, v2_im, kp_2, km_2, dw2), mask=own2)

        # ========= the mirror fills, carried by the SOURCE lane =====================
        # folded_fused_magnetic_pair's carries, per part, with the fills' parity words
        # (host-rounded +-1.0, near ``n*r`` = +phase, far ``d*r`` = -phase). A fill is a
        # real scale per part (part_products.scale), so each application is ``word *
        # part``; a composed destination applies them in the driver's order -- the near
        # fill innermost, then the far axes ascending (:func:`parity_chain`) -- and every
        # application is exact. The near destination reads its coefficient pair and
        # membership at row 0 of the component's own axis; a far-only destination keeps
        # the source lane's.
        origin = idx * 0
        kp_d0 = tl.load(kp0 + origin, mask=live, other=0.0)
        km_d0 = tl.load(km0 + origin, mask=live, other=0.0)
        kp_d1 = tl.load(kp1 + origin, mask=live, other=0.0)
        km_d1 = tl.load(km1 + origin, mask=live, other=0.0)
        kp_d2 = tl.load(kp2 + origin, mask=live, other=0.0)
        km_d2 = tl.load(km2 + origin, mask=live, other=0.0)
        dw_d0 = record_in(cb, 0, origin)
        dw_d1 = record_in(cb, 4, origin)
        dw_d2 = record_in(cb, 8, origin)
        near_i = live & (i == 2)
        near_j = live & (j == 2)
        near_k = live & (k == 2)
        far_i = live & (i == rx)
        far_j = live & (j == ry)
        far_k = live & (k == rz)
        dn_x = -2 * nyz
        dn_y = -2 * nz
        dn_z = -2
        df_x = (nx - 1 - rx) * nyz
        df_y = (ny - 1 - ry) * nz
        df_z = (nz - 1 - rz)

        # --- component 0 (Bx): near on x, far on y and z ---
        if FAR_Y:
            _carry_ghost_complex(f0, w0, h0, idx + df_y,
                                 d1r * v0_re,
                                 d1r * v0_im,
                                 kp_0, km_0, dw0, start, own0 & far_j)
        if FAR_Z:
            _carry_ghost_complex(f0, w0, h0, idx + df_z,
                                 d2r * v0_re,
                                 d2r * v0_im,
                                 kp_0, km_0, dw0, start, own0 & far_k)
        if FAR_Y and FAR_Z:
            _carry_ghost_complex(f0, w0, h0, idx + df_y + df_z,
                                 d2r * (d1r * v0_re),
                                 d2r * (d1r * v0_im),
                                 kp_0, km_0, dw0, start, own0 & far_j & far_k)
        if NEAR_X:
            _carry_ghost_complex(f0, w0, h0, idx + dn_x,
                                 n0r * v0_re,
                                 n0r * v0_im,
                                 kp_d0, km_d0, dw_d0, start, own0 & near_i)
        if NEAR_X and FAR_Y:
            _carry_ghost_complex(f0, w0, h0, idx + dn_x + df_y,
                                 d1r * (n0r * v0_re),
                                 d1r * (n0r * v0_im),
                                 kp_d0, km_d0, dw_d0, start, own0 & near_i & far_j)
        if NEAR_X and FAR_Z:
            _carry_ghost_complex(f0, w0, h0, idx + dn_x + df_z,
                                 d2r * (n0r * v0_re),
                                 d2r * (n0r * v0_im),
                                 kp_d0, km_d0, dw_d0, start, own0 & near_i & far_k)
        if NEAR_X and (FAR_Y and FAR_Z):

            _carry_ghost_complex(f0, w0, h0, idx + dn_x + df_y + df_z,
                                 d2r * (d1r * (n0r * v0_re)),
                                 d2r * (d1r * (n0r * v0_im)),
                                 kp_d0, km_d0, dw_d0, start, own0 & near_i & far_j & far_k)

        # --- component 1 (By): near on y, far on x and z ---
        if FAR_X:
            _carry_ghost_complex(f1, w1, h1, idx + df_x,
                                 d0r * v1_re,
                                 d0r * v1_im,
                                 kp_1, km_1, dw1, start, own1 & far_i)
        if FAR_Z:
            _carry_ghost_complex(f1, w1, h1, idx + df_z,
                                 d2r * v1_re,
                                 d2r * v1_im,
                                 kp_1, km_1, dw1, start, own1 & far_k)
        if FAR_X and FAR_Z:
            _carry_ghost_complex(f1, w1, h1, idx + df_x + df_z,
                                 d2r * (d0r * v1_re),
                                 d2r * (d0r * v1_im),
                                 kp_1, km_1, dw1, start, own1 & far_i & far_k)
        if NEAR_Y:
            _carry_ghost_complex(f1, w1, h1, idx + dn_y,
                                 n1r * v1_re,
                                 n1r * v1_im,
                                 kp_d1, km_d1, dw_d1, start, own1 & near_j)
        if NEAR_Y and FAR_X:
            _carry_ghost_complex(f1, w1, h1, idx + dn_y + df_x,
                                 d0r * (n1r * v1_re),
                                 d0r * (n1r * v1_im),
                                 kp_d1, km_d1, dw_d1, start, own1 & near_j & far_i)
        if NEAR_Y and FAR_Z:
            _carry_ghost_complex(f1, w1, h1, idx + dn_y + df_z,
                                 d2r * (n1r * v1_re),
                                 d2r * (n1r * v1_im),
                                 kp_d1, km_d1, dw_d1, start, own1 & near_j & far_k)
        if NEAR_Y and (FAR_X and FAR_Z):

            _carry_ghost_complex(f1, w1, h1, idx + dn_y + df_x + df_z,
                                 d2r * (d0r * (n1r * v1_re)),
                                 d2r * (d0r * (n1r * v1_im)),
                                 kp_d1, km_d1, dw_d1, start, own1 & near_j & far_i & far_k)

        # --- component 2 (Bz): near on z, far on x and y ---
        if FAR_X:
            _carry_ghost_complex(f2, w2, h2, idx + df_x,
                                 d0r * v2_re,
                                 d0r * v2_im,
                                 kp_2, km_2, dw2, start, own2 & far_i)
        if FAR_Y:
            _carry_ghost_complex(f2, w2, h2, idx + df_y,
                                 d1r * v2_re,
                                 d1r * v2_im,
                                 kp_2, km_2, dw2, start, own2 & far_j)
        if FAR_X and FAR_Y:
            _carry_ghost_complex(f2, w2, h2, idx + df_x + df_y,
                                 d1r * (d0r * v2_re),
                                 d1r * (d0r * v2_im),
                                 kp_2, km_2, dw2, start, own2 & far_i & far_j)
        if NEAR_Z:
            _carry_ghost_complex(f2, w2, h2, idx + dn_z,
                                 n2r * v2_re,
                                 n2r * v2_im,
                                 kp_d2, km_d2, dw_d2, start, own2 & near_k)
        if NEAR_Z and FAR_X:
            _carry_ghost_complex(f2, w2, h2, idx + dn_z + df_x,
                                 d0r * (n2r * v2_re),
                                 d0r * (n2r * v2_im),
                                 kp_d2, km_d2, dw_d2, start, own2 & near_k & far_i)
        if NEAR_Z and FAR_Y:
            _carry_ghost_complex(f2, w2, h2, idx + dn_z + df_y,
                                 d1r * (n2r * v2_re),
                                 d1r * (n2r * v2_im),
                                 kp_d2, km_d2, dw_d2, start, own2 & near_k & far_j)
        if NEAR_Z and (FAR_X and FAR_Y):

            _carry_ghost_complex(f2, w2, h2, idx + dn_z + df_x + df_y,
                                 d1r * (d0r * (n2r * v2_re)),
                                 d1r * (d0r * (n2r * v2_im)),
                                 kp_d2, km_d2, dw_d2, start, own2 & near_k & far_i & far_j)

else:  # pragma: no cover - laptop path
    folded_beta_complex_fused_curl_constitutive_B = None  # type: ignore[assignment]


def folded_beta_complex_fused_curl_constitutive_B_kernel() -> Any:
    """The JIT kernel, or a named ImportError on a host without Triton."""
    if folded_beta_complex_fused_curl_constitutive_B is None:
        raise ImportError(
            "the folded complex fused magnetic B/H kernel needs the optional "
            f"`triton` package (pip install triton). Original error: "
            f"{_TRITON_IMPORT_ERROR}")
    return folded_beta_complex_fused_curl_constitutive_B


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def folded_beta_complex_fused_magnetic_pair_coverage(fields: Any, pml: Any,
                                                sources: Any = None) -> Coverage:
    """May ONE launch span folded complex ``step_B`` -> both fills -> ``update_H``?

    A conjunction of the THREE halves' own predicates plus the seam clauses.
    Nothing is weakened: a configuration any half refuses is refused here with that
    half's reasons, prefixed so a reader can tell which side said it — the same
    construction :func:`.folded_fused_magnetic_pair.folded_fused_magnetic_pair_coverage`
    and :func:`.complex_fused_magnetic_pair.complex_fused_magnetic_pair_coverage` use.
    """
    reasons: List[str] = []

    # ONE ADMISSION, where the beta-less twin carries two. That twin ALSO admits
    # the off-diagonal labels because on B->H they are the same kernel and plan
    # class; no BETA variant of those arms is shipped, K3b's own media clause
    # refuses an off-diagonal row by name (``_media_reasons``), and a predicate may
    # not be wider than the kernels a gate has run. An off-diagonal beta row -- of
    # which the corpus has none -- lands in no-admitting-arm, which is the honest
    # bucket until such an arm exists.
    curl = folded_beta_bloch_pml_curl_coverage(
        fields, pml, CURL_SUB_STEP)
    constitutive = folded_complex_constitutive_coverage(
        fields, pml, CONSTITUTIVE_SIDE)
    if not curl.covered:
        reasons.extend(f"folded complex beta curl half: {reason}"
                       for reason in curl.reasons)
    if not constitutive.covered:
        reasons.extend(f"folded complex constitutive half: {reason}"
                       for reason in constitutive.reasons)

    # THE HAZARD THE OFF-DIAGONAL ADMISSION RESTS ON, CHECKED. This launch may not
    # bind an inverse-epsilon volume; if the H side ever grew one, the clause the
    # E-side predicate raises would apply here too.
    if "inverse" in CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]:
        reasons.append(
            "coverage.CONSTITUTIVE_SIDES['H'] now binds an inverse epsilon; the "
            "off-diagonal admission above rests on update_H reading none, and that "
            "is no longer true")

    fill = folded_mirror_ghost_fill_complex_coverage(fields, FILL_FAMILY)
    if not fill.covered:
        reasons.extend(f"folded complex fill half: {reason}"
                       for reason in fill.reasons)

    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False,
                        tuple(dict.fromkeys(reasons + ["fields carries no grid"])))

    # THE SOURCE SEAM. A magnetic source is injected BETWEEN the two halves
    # (driver.py:3283-3284), so a fused pair would consume a pre-injection B. An
    # ELECTRIC source is injected in the D/E half and does not disqualify this pair.
    # IGNORANCE IS NOT AN EMPTY SET: `Fields` does not hold the source list, so a
    # predicate that inferred "no magnetic source" from not being told would be the
    # over-covering refusal this clause exists to prevent.
    reasons.extend(_deposit_repair.seam_source_reasons(
        fields, sources, 'B',
        undeclared=(
            "the source set was not declared: this predicate cannot infer an empty "
            "magnetic source seam from Fields"
        ),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) is magnetic: the "
            f"driver injects it BETWEEN step_B and update_H "
            f"(driver.py:3283), which is work inside the seam this kernel "
            f"closes"),
        carries_repair=CARRIES_DEPOSIT_REPAIR))

    codes, code_reasons = folded_axis_kinds(
        grid, pml if (pml is not None and getattr(pml, "is_active", False)) else None)
    reasons.extend(code_reasons)

    if codes is not None:
        # THE FAR CARRY'S OWNERSHIP MOVE. `fill_folded_far_ghosts_B` (driver.py:3287)
        # images the top stored slot from stepping._far_reflect_rows' row, and the
        # lane that owns that destination is the one AT the reflect row — so a row
        # outside the allocation is an out-of-range write from a lane that owns
        # neither cell, not a soft error on a whole-plane assignment. The three row
        # checks are symmetry.mirror_ghost_fill_coverage's own (symmetry.py:842-895),
        # restated here for the same reason folded_fused_magnetic_pair restates them.
        shape = tuple(getattr(grid, "shape", ()))
        rows = _far_reflect_rows(grid)
        for axis, code in enumerate(codes):
            periodic = int(code) == CODE_MIRROR_PERIODIC
            reader = _stored_past_owned_reader()
            if reader is None:  # pragma: no cover - folded_axis_kinds refused first
                reasons.append(
                    "stepping._stored_past_owned is not importable; the far carry "
                    "cannot be checked against the array path's own predicate")
                break
            past_owned = bool(reader(grid, axis))
            if periodic != past_owned:
                reasons.append(
                    f"axis {axis} code {int(code)} and "
                    f"stepping._stored_past_owned {past_owned} disagree; the kernel "
                    f"would mask the top plane and image the far ghost on different "
                    f"axes than the array path")
            if not periodic or len(shape) != 3:
                continue
            row = rows[axis] if rows is not None else None
            if row is None:
                reasons.append(
                    f"axis {axis} is a folded PERIODIC axis with no reflect row "
                    f"from stepping._far_reflect_rows; the far carry has nothing "
                    f"to image")
                continue
            row, extent = int(row), int(shape[axis])
            if not (0 <= row < extent - 1):
                reasons.append(
                    f"axis {axis} reflect row {row} is outside [0, {extent - 1}); "
                    f"the far carry would image the plane it writes, or read "
                    f"outside the allocation")
            if row == 0:
                reasons.append(
                    f"axis {axis} reflect row is 0, the plane the NEAR fill writes: "
                    f"the far carry would image a ghost rather than an owned cell")
            if extent - 1 == MIRROR_SOURCE_INDEX:
                reasons.append(
                    f"axis {axis} stores {extent} cells, so the far carry's write "
                    f"plane IS the near fill's read plane "
                    f"(cell {MIRROR_SOURCE_INDEX})")

        # THE NEAR FILL IMAGES STORED CELL 2 FROM THAT CELL'S OWN LANE. The folded
        # complex curl predicate already refuses a folded axis storing two cells or
        # fewer; restated because THIS family writes the destination from a
        # different lane and a missing source plane is an out-of-range write, not a
        # soft error. BOTH mirror codes: the near fill runs on either.
        if len(shape) == 3:
            for axis, code in enumerate(codes):
                if (int(code) in MIRROR_CODES
                        and int(shape[axis]) <= MIRROR_SOURCE_INDEX):
                    reasons.append(
                        f"axis {axis} is folded with {int(shape[axis])} stored "
                        f"cells; the near fill images stored cell "
                        f"{MIRROR_SOURCE_INDEX} and this kernel images it from that "
                        f"cell's own lane")

        # THE WALL CLEAR AND THE FILL MUST NOT MEET. On this family both act on
        # component m at stored cell 0 of axis m, and `_zero_metal` skips a folded
        # axis (:2237-2239) so they are disjoint by construction. CHECKED rather
        # than inferred: an overlap would be a plane of wrong values.
        walls = zero_metal_axes(grid)
        for axis, code in enumerate(codes):
            if int(code) in MIRROR_CODES and bool(walls[axis]):
                reasons.append(
                    f"axis {axis} is folded AND zero_metal_axes reports a wall "
                    f"there; stepping._zero_metal skips a folded axis, so the two "
                    f"passes have drifted and the carry's disjointness no longer "
                    f"holds")

        # THE PARITY WORDS THE KERNEL WILL BE PASSED must be readable on every
        # folded axis. `folded_axis_kinds` already refuses a folded axis whose phase
        # is not +-1; this re-derives the WORDS through the builder's own function
        # so a pair that came back (0.0, 0.0) on a folded axis is caught here and
        # not on the device, where it would write an exact zero plane.
        words = parity_coefficient_words(grid)
        for axis, code in enumerate(codes):
            if int(code) not in MIRROR_CODES:
                continue
            if words[axis] == (0.0, 0.0) or words[axis + 3] == (0.0, 0.0):
                reasons.append(
                    f"axis {axis} is folded but its parity coefficient words "
                    f"resolve to near={words[axis]!r} far={words[axis + 3]!r}; the "
                    f"carry would image an exact zero plane")

    # The wall clear is carried inline, so the grid must be able to answer which
    # axes are walled; an unanswerable one compiles to ZM=False and silently skips
    # a plane the array path clears.
    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            reasons.append(
                f"grid does not expose {name}; zero_metal_B cannot be carried inline")

    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class FoldedBetaComplexFusedMagneticPairPlan:
    """One allocation-free launch for five of the driver's magnetic call sites.

    The complex volumes are bound as float32 WORD VIEWS at plan time, once, and
    ``n_elem`` stays the COMPLEX cell count — the same split
    :class:`.complex_fields.ComplexPmlCurlPlan` makes, for the same reason.
    """

    __slots__ = (
        "beta_words", "has_beta",
        "shape", "n_elem", "dtdx", "backward", "bc", "zero_metal", "connections",
        "connection_values", "parity", "reflect", "near", "far", "chunk_bounds", "_cb",
        "block", "num_warps", "_targets", "_aux", "_sources", "_curl_coefficients",
        "_h_targets", "_h_aux", "_h_coefficients", "_grid", "_kernel", "_h_state",
    )

    #: The five driver call sites one launch performs. Declared, so a composition
    #: can be inspected rather than inferred from the curl slot's name.
    replaces = REPLACES

    def __init__(self, shape, dtdx: float, bc, zero_metal, connections,
                 connection_values, parity, block: int,
                 targets, auxiliaries, sources, curl_coefficients,
                 h_targets, h_aux, h_coefficients,
                 reflect: Any = (None, None, None),
                 beta_words: Any = None, has_beta: int = 1,
                 kernel: Any = None, num_warps: Optional[int] = 1, *,
                 chunk_bounds: Any, h_state: Any = None) -> None:
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        # float(): the array path consumes the Python scalar at complex64 precision
        # and Triton types a Python float argument as fp32, so the two are the same
        # bits (complex_fields.ComplexPmlCurlPlan's measured note).
        self.dtdx = float(dtdx)
        self.backward = BACKWARD
        self.bc = tuple(int(code) for code in bc)
        self.zero_metal = tuple(bool(value) for value in zero_metal)
        self.connections = tuple(int(kind) for kind in connections)
        self.connection_values = tuple(float(value) for value in connection_values)
        # SIX word pairs, near x/y/z then far x/y/z, host-rounded once.
        self.parity = tuple((float(pair[0]), float(pair[1])) for pair in parity)
        if len(self.parity) != 6:
            raise ValueError(
                f"the parity table is six (re, im) pairs — near x/y/z then far "
                f"x/y/z — got {len(self.parity)}")
        # DERIVED FROM `bc`, NEVER TAKEN AS AN ARGUMENT. The kernel branches its
        # curl half on the four codes and its two carries on these booleans; if
        # they could be passed independently they could disagree, and a kernel that
        # masked the top plane on one axis set and imaged the far ghost on another
        # is a plane of wrong values rather than a crash.
        self.near = tuple(code in MIRROR_CODES for code in self.bc)
        self.far = tuple(code == CODE_MIRROR_PERIODIC for code in self.bc)
        # -1 where no far ghost exists. A sentinel rather than 0: no lane reads it
        # (the guard that would is emitted only under FAR_a), and an out-of-range
        # index is loud where imaging row 0 would be plausible and wrong.
        self.reflect = tuple(-1 if value is None else int(value)
                             for value in reflect)
        for axis, code in enumerate(self.bc):
            folded = code in MIRROR_CODES
            if folded and self.parity[axis] == (0.0, 0.0):
                raise ValueError(
                    f"axis {axis} is folded but its NEAR parity words are "
                    f"(0.0, 0.0); the carry would image an exact zero plane")
            if folded and self.parity[axis + 3] == (0.0, 0.0):
                raise ValueError(
                    f"axis {axis} is folded but its FAR parity words are "
                    f"(0.0, 0.0); the carry would image an exact zero plane")
            if folded and self.connections[axis]:
                raise ValueError(
                    f"axis {axis} carries both a fold and a Bloch phase; "
                    f"driver._require_bloch_is_representable refuses that "
                    f"configuration outright and a kernel cannot lift what the "
                    f"array path will not run")
            if folded and self.zero_metal[axis]:
                raise ValueError(
                    f"axis {axis} carries both a fold and a wall clear; "
                    f"stepping._zero_metal skips a folded axis")
            if code == CODE_MIRROR_PERIODIC and not (
                    0 <= self.reflect[axis] < int(self.shape[axis]) - 1):
                raise ValueError(
                    f"axis {axis} is folded PERIODIC with reflect row "
                    f"{self.reflect[axis]}, which is outside "
                    f"[0, {int(self.shape[axis]) - 1}); the far carry would image "
                    f"the plane it writes, or read outside the allocation")
            if code != CODE_MIRROR_PERIODIC and self.reflect[axis] != -1:
                raise ValueError(
                    f"axis {axis} is not folded PERIODIC but carries reflect row "
                    f"{self.reflect[axis]}; stepping._far_reflect_rows answers None "
                    f"there and the kernel would read a row nothing wrote")
        # (plus, minus): the two host-rounded beta words (the side's i or -i already
        # applied, special_kz.beta_curl_coefficients); float() keeps the bits.
        words = tuple(beta_words) if beta_words is not None else ()
        if len(words) != 2 or any(isinstance(word, (tuple, list)) for word in words):
            raise ValueError(f"beta_words must be the two real beta words of "
                             f"special_kz.beta_curl_coefficients, got {beta_words!r}")
        self.beta_words = tuple(float(word) for word in words)
        self.has_beta = int(has_beta)
        self.block = int(block)
        self.num_warps = None if num_warps is None else int(num_warps)
        self._targets = tuple(CupyPointer(_word_view(a)) for a in targets)
        self._aux = tuple(CupyPointer(_word_view(a)) for a in auxiliaries)
        self._sources = tuple(CupyPointer(_word_view(a)) for a in sources)
        self._curl_coefficients = tuple(
            CupyPointer(_flat(a)) for a in curl_coefficients)
        self._h_targets = tuple(CupyPointer(_word_view(a)) for a in h_targets)
        self._h_aux = tuple(CupyPointer(_word_view(a)) for a in h_aux)
        self._h_coefficients = tuple(
            CupyPointer(_flat(a)) for a in h_coefficients)
        # THE INT32 WORD BOUND. The kernel addresses words as ``2 * idx`` in int32,
        # so the complex cell count must leave room for the doubling — the same
        # halving the complex predicates apply. Refused here as well because the
        # from-arrays route runs no predicate at all.
        if 2 * self.n_elem >= 2 ** 31:
            raise ValueError(
                f"{self.n_elem} complex cells needs {2 * self.n_elem} int32 word "
                f"indices, which overflows the kernel's 2 * idx addressing")
        from .launch import bind_chunk_bounds  # noqa: PLC0415

        self.chunk_bounds, self._cb = bind_chunk_bounds(
            chunk_bounds, self.shape, targets[0], "FoldedBetaComplexFusedMagneticPairPlan")
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        # An override for exactly one caller: the gate's mutation legs, which
        # compile deliberately broken copies of this kernel. Dropping it is not a
        # slowdown, it is a DISARMING — every mutation leg would then launch the
        # shipped kernel and report the defect as uncaught.
        self._kernel = kernel
        # The Fields the H half starts from (MEEP's start of H), or None: the
        # contract of launch.ConstitutivePlan's h_state.
        self._h_state = h_state

    def start_h(self) -> int:
        """1 on the one launch that starts H from B, 0 otherwise."""
        state = self._h_state
        return 0 if state is None or state.h_separated else 1

    def run(self, guard: Optional[bool] = None) -> None:
        """Launch the fused pair, in place. Same ``guard`` contract as every plan."""
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        kernel = (self._kernel if self._kernel is not None
                  else folded_beta_complex_fused_curl_constitutive_B_kernel())
        nx, ny, nz = self.shape
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        # The parity is a real scale per part: each pair's real word (+-1.0); the
        # imaginary word is +0 and is not read.
        parity_words = tuple(pair[0] for pair in self.parity)
        start = self.start_h()
        kernel[self._grid](
            *self._targets, *self._aux, *self._sources, *self._curl_coefficients,
            *self._h_targets, *self._h_aux, *self._h_coefficients, self._cb,
            nx, ny, nz, self.n_elem, self.dtdx,
            *self.connection_values,
            self.reflect[0], self.reflect[1], self.reflect[2],
            *parity_words, start,
            *self.beta_words,
            BACKWARD=self.backward,
            BCX=self.bc[0], BCY=self.bc[1], BCZ=self.bc[2],
            NEAR_X=self.near[0], NEAR_Y=self.near[1], NEAR_Z=self.near[2],
            FAR_X=self.far[0], FAR_Y=self.far[1], FAR_Z=self.far[2],
            ZM_X=self.zero_metal[0], ZM_Y=self.zero_metal[1],
            ZM_Z=self.zero_metal[2],
            CNX=self.connections[0], CNY=self.connections[1],
            CNZ=self.connections[2],
            HAS_BETA=self.has_beta,
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )
        if start and self._grid[0]:  # an empty grid (the warm pass) starts nothing
            self._h_state.h_separated = True

    def __repr__(self) -> str:
        return (f"FoldedBetaComplexFusedMagneticPairPlan(shape={self.shape}, "
                f"bc={self.bc}, zero_metal={self.zero_metal}, "
                f"connections={self.connections}, near={self.near}, far={self.far}, "
                f"reflect={self.reflect}, parity={self.parity}, block={self.block}, "
                f"num_warps={self.num_warps}, chunk_bounds={self.chunk_bounds.tolist()})")


def plan_folded_beta_complex_fused_magnetic_pair(
        fields: Any, pml: Any, sources: Any = None,
        block: Optional[int] = None, num_warps: Optional[int] = 1,
        kernel: Any = None, chunk_layout: Any = None,
) -> Optional[FoldedBetaComplexFusedMagneticPairPlan]:
    """Build the plan from the engine's own objects, or ``None`` when refused.

    ``None`` is the only refusal: a configuration this kernel does not carry must
    fall back to the array path, never raise into a caller that would otherwise have
    stepped correctly.
    ``chunk_layout`` is the run's chunk division, ``None`` for the one-rank default
    (``launch.chunk_bounds_for``).
    """
    if not folded_beta_complex_fused_magnetic_pair_coverage(fields, pml, sources).covered:
        return None
    from ..stepping import _boundary_kinds as resolve  # noqa: PLC0415
    from .launch import chunk_bounds_for  # noqa: PLC0415

    codes, _ = folded_axis_kinds(fields.grid, pml)
    if codes is None:  # pragma: no cover - the predicate already refused
        return None

    curl_spec = SUB_STEPS[CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    grid = fields.grid
    kinds = resolve(grid, pml)
    connections, values = connection_arguments(bloch_phase_table(grid, kinds),
                                               backward=bool(curl_spec["backward"]))
    words = beta_curl_coefficients(fields.grid.beta, fields.grid.dt,
                                   magnetic=True, complex_storage=True)
    return FoldedBetaComplexFusedMagneticPairPlan(
        grid.shape, grid.dt / grid.dx, codes, zero_metal_axes(grid),
        connections, values, parity_coefficient_words(grid),
        DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in curl_spec["targets"]],
        [getattr(fields, "fu_" + name) for name in curl_spec["targets"]],
        [getattr(fields, name) for name in curl_spec["sources"]],
        [getattr(pml, f"{stem}_{axis}{curl_spec['suffix']}")
         for axis in "xyz" for stem in ("kms", "sinv")],
        [getattr(fields, name) for name in side_spec["targets"]],
        [getattr(fields, name) for name in side_spec["aux"]],
        # The curl takes the HALF-INTEGER lattice on step_B and the constitutive
        # the INTEGER one on H (stepping.py:948 against the B curl's
        # half_integer=True). The kernel takes both and never asks which is which,
        # so a swap here is a silent half-cell error in the absorber profile; the
        # gate carries a mutation for exactly it.
        [getattr(pml, f"{stem}_{axis}") for axis in "xyz"
         for stem in ("kps", "kms")],
        # The far carry's image rows, from the engine's own function rather than
        # recomputed: `n_full - stored + 2` is `stored - 2` at an even full count
        # and `stored - 3` at an odd one (stepping._far_reflect_rows:1661).
        reflect=_far_reflect_rows(grid) or (None, None, None),
        beta_words=words,
        kernel=kernel, num_warps=num_warps,
        chunk_bounds=chunk_bounds_for(fields, pml, chunk_layout), h_state=fields,
    )


def plan_folded_beta_complex_fused_magnetic_pair_from_arrays(
        arrays: Dict[str, Any], curl_flat: Dict[str, Any],
        constitutive_flat: Dict[str, Any], codes, zero_metal,
        phases: Sequence[Optional[complex]], parity, dtdx: float,
        block: Optional[int] = None, kernel: Any = None,
        num_warps: Optional[int] = 1,
        reflect: Any = (None, None, None), beta_words: Any = None, has_beta: int = 1, *, chunk_bounds: Any) -> FoldedBetaComplexFusedMagneticPairPlan:
    """Build from bare device arrays — the gate's route.

    ``beta_words`` is ``(plus, minus)``, normally
    :func:`.special_kz.beta_curl_coefficients`'s output or a deliberately wrong pair
    on a mutation leg; ``has_beta=0`` is the identity leg's folded-pair arm.

    No coverage predicate runs: the caller is a harness that constructed the
    configuration deliberately, including the deliberately wrong ones. ``phases`` is
    the per-axis ``Optional[complex]`` Bloch table (None = unphased) and ``parity``
    the six ``(re, im)`` word pairs, near x/y/z then far x/y/z. The conjugation for a
    backward sub-step is applied HERE, exactly as the engine route applies it — this
    product is forward-only, so it never is.
    """
    curl_spec = SUB_STEPS[CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    shape = tuple(int(n) for n in arrays[curl_spec["targets"][0]].shape)
    connections, values = connection_arguments(tuple(phases),
                                               backward=bool(curl_spec["backward"]))
    return FoldedBetaComplexFusedMagneticPairPlan(
        shape, dtdx, codes, zero_metal, connections, values, parity,
        DEFAULT_BLOCK if block is None else block,
        [arrays[name] for name in curl_spec["targets"]],
        [arrays["fu_" + name] for name in curl_spec["targets"]],
        [arrays[name] for name in curl_spec["sources"]],
        [curl_flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
        [arrays[name] for name in side_spec["targets"]],
        [arrays[name] for name in side_spec["aux"]],
        [constitutive_flat[f"{stem}_{axis}"]
         for axis in "xyz" for stem in ("kps", "kms")],
        reflect=reflect, beta_words=beta_words, has_beta=has_beta,
        kernel=kernel, num_warps=num_warps, chunk_bounds=chunk_bounds,
    )
