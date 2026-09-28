"""The FOLDED COMPLEX BETA electric seam in one launch: beta ``step_D`` into ``update_E``.

:mod:`.folded_complex_fused_pair` WITH THE K3b CURL, and the swap is the whole module:
the certified folded complex weld — both fills, the wall clear inside the parity
chain, the ownership restructure, the deposit-repair bracket — takes
:func:`.folded_complex.folded_beta_bloch_pml_curl_step`'s beta insert exactly where
K3b takes it over K1: four host-rounded scalar words, the ``HAS_BETA`` constexpr, and
the seven-statement insert between the dtdx curl and the ownership masks. Nothing
else moves, and the suite's transcription leg asserts that statement by statement
against BOTH shipped sources (the beta-less fused kernel and K3b).

It claims the cell ``(folded complex beta PML, folded complex)`` at ``D->E``, which no
shipped Triton weld can stand in for: :mod:`.folded_complex_fused_pair` refuses
``grid.beta`` through its curl half (K1's clause set requires beta = 0),
:mod:`.folded_beta_fused_electric_pair` is the REAL-storage beta fold and refuses
complex64 by name, and :mod:`.complex_beta_fused_electric_pair` refuses a mirror
plane. THE SIBLING IS :mod:`..metal_kernels.folded_beta_complex_fused_pair`, which
serves the same rows on Metal today — the proof the weld shape is portable — and each
region here is transcribed from THIS backend's certified bodies, never from the
sibling's MSL.

===========================================================================
WHAT IT IS WORTH: +3 SEAM-INSTANCES OF 387, AND THE PROBE THAT UNLOCKS THEM
===========================================================================

MEASURED, from ``results/fusion_matrix_triton_2026-09-01_foldedbeta``: three
reachable D->E seam-instances sit in this cell, scored "reachable; a slot has NO
admitting arm (array path)" with ``curl_admitters: []`` — the K3b arm refusing on the
then-unmeasured fifth probe pattern (``c8_mul_c8_imaginary_coefficient_left``,
special_kz.py:223)::

    tests:TestEigCoeffs.test_binary_grating_special_kz_0_13_2   fill_D + far ghosts, electric in-seam source
    tests:TestEigCoeffs.test_binary_grating_special_kz_1_17_7   fill_D + far ghosts, electric in-seam source
    tests:TestSpecialKz.test_eigsrc_kz_0_complex                fill_D + far ghosts, electric in-seam sources

All three fold axis y, keep complex64 storage, and declare beta != 0; the two binary
gratings additionally ride an in-plane Bloch kx on the unfolded x axis. ALL THREE
DECLARE AN ELECTRIC SOURCE (``source_field_types`` carries ``'D'``), so under the
source clause with :data:`CARRIES_DEPOSIT_REPAIR` at False this arm would admit
NOTHING — the flag is the product, exactly as it was for the beta-less twin's two
rows. THE REFUSAL WAS DISCHARGED BY MEASUREMENT, not by edit:
``results/complex_expansion_beta_extension_2026-09-01/probe_keep`` classifies the
fifth pattern on the device (AMBIGUOUS_BOTH on all 16 corpus coefficients, licensable
arms 0 words apart per entry) with its base-four table character-for-character
identical to the standing 2026-08-16 keep artifact, and the six-pattern
``results/expansion_probe_2026-08-17`` record the released folded gates consume
licenses BOTH extended sets (beta and parity) as FMA_V1.

===========================================================================
EVERYTHING ELSE IS THE BETA-LESS TWIN'S, BY CONSTRUCTION
===========================================================================

The five passes in the seam, the D-side fill geometry and its exact inverse relation
to the B twin's, the ordered parity chain with ``zero_metal_D`` inside it
(:data:`CLEAR_AFTER_NEAR`), the ownership restructure with every carry mask ANDed
with the component's own ownership mask, the one-``_mul_imag_coefficient_left``-per-
pass composition law, and the reasons the off-diagonal admission is deliberately
absent on this seam are all :mod:`.folded_complex_fused_pair`'s, documented there and
carried here unchanged — the K3b insert touches none of them (the beta partners are
the CENTER word pairs, which no fill and no phase rotation ever touches).

What the beta insert ADDS to the licence question: the kernel launches SIX operand
orientations — the beta-less twin's five plus the imaginary-coefficient-left product
— so :data:`PRODUCT_PROBE_PATTERNS` is the base four PLUS the parity pattern PLUS
:data:`.special_kz.BETA_PROBE_PATTERN`, and the plan builder re-asks BOTH extended
licences (:func:`.folded_complex.parity_expansion_from_probe` and
:func:`.folded_complex.folded_beta_expansion_from_probe`) and requires them to agree
on ONE arm.

NOT WIRED, and not an arm. ``launch.plan_step`` assigns at most one plan per
``STEP_ORDER`` slot and this product spans FIVE driver call sites; nothing in
``launch.py`` names this module and dispatch stays disabled — the same deferral every
sibling fused pair ships under.

Import contract: importable WITHOUT Triton — the predicate and the plan builder (to
``None``) must answer on the laptop that is the merge bar.

DEVICE STATUS: see :data:`DEVICE_STATUS`. A weld licenses a claim, not a dispatch.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .complex_fields import (
    DEFAULT_BLOCK,
    _mul_coefficient_left,
    _mul_field_left,
    _phase_arguments,
    _rotate_field_left,
    _word_view,
    bloch_phase_table,
)
from .coverage import (
    CONSTITUTIVE_SIDES,
    Coverage,
    _call,
    zero_metal_axes,
)
from .folded_complex import (
    CODE_MIRROR_METALLIC,
    CODE_MIRROR_PERIODIC,
    CODE_PERIODIC,
    MIRROR_SOURCE_INDEX,
    PARITY_PROBE_PATTERNS,
    TARGET_IYEE,
    _far_reflect_rows,
    _stored_past_owned_reader,
    folded_axis_kinds,
    folded_beta_bloch_pml_curl_coverage,
    folded_beta_expansion_from_probe,
    folded_complex_constitutive_coverage,
    folded_mirror_ghost_fill_complex_coverage,
    mirror_parity_coefficients,
    parity_expansion_from_probe,
)
from .launch import SUB_STEPS, CupyPointer, _flat
from .special_kz import (
    BETA_PROBE_PATTERN,
    _mul_imag_coefficient_left,
    beta_curl_coefficients,
)
from .. import deposit_repair as _deposit_repair

#: Does this product bracket its fused launch with the deposit repair? TRUE, and on
#: THIS cell it is the difference between three seam-instances and zero: ALL THREE
#: corpus rows (the two binary gratings and ``test_eigsrc_kz_0_complex``) declare an
#: ELECTRIC source, which the driver injects between the two halves
#: (driver.py:3294-3299). The flag is a claim about the PLAN this module builds --
#: that the leading slot saves and the trailing slot restores -- and is only ever
#: changed in the same edit as that wiring. The Metal sibling
#: (``folded_beta_complex_fused_pair``) declares it True on the same rows, which is
#: the independent check.
#:
#: THE FLAG IS NOT THE WHOLE CLAIM. It only reaches ``deposit_repair.repairable``,
#: which goes on refusing BY NAME every seam the repair cannot invert -- an
#: off-diagonal constitutive (a stencil over the curl's own in-place output), a
#: nonlinear one, a cylindrical r = 0 axis, a fold too short to hold the near fill's
#: source row, and an absorber whose split-field recurrence never ran. On a FOLDED
#: seam the repair also extends its saved and restored cell set to every image the two
#: fills write (``deposit_repair.repair_cells``), which is the clause that makes the
#: flag legitimate on this family rather than merely convenient.
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

#: The FIVE driver call sites ONE launch of this plan performs, in driver order
#: (driver.py:3292-3304). Declared, never inferred from the slot name. All five do
#: work on both corpus rows of this cell, which fold a PERIODIC axis. THE INJECTION
#: IS NOT ONE OF THE FIVE: it is carried by the deposit repair bracket around the
#: launch, not by the launch.
REPLACES: Tuple[str, ...] = ("step_D", "fill_D", "zero_metal_D",
                             "fill_folded_far_ghosts_D", "update_E")

#: ``SUB_STEPS['step_D']['backward']``, restated so the kernel's one legal binding is
#: visible without importing :mod:`launch`; the test suite pins the two equal.
BACKWARD = 1

#: The two codes a folded axis can take. The NEAR fill runs on either
#: (``stepping._fill_symmetry_ghost_cells`` gates on the mirror phases alone); the FAR
#: fill only on ``MIRROR_PERIODIC`` (``stepping._stored_past_owned``:1454-1469), which
#: is the whole of what separates the two carries.
MIRROR_CODES: Tuple[int, int] = (CODE_MIRROR_METALLIC, CODE_MIRROR_PERIODIC)

#: The :data:`.folded_complex.GHOST_FILL_FAMILIES` key this product's fill half is
#: asked about. Spelled out rather than derived from ``CURL_SUB_STEP``, and pinned by
#: test against ``GHOST_FILL_FAMILIES``' own keys.
FILL_FAMILY: str = "D"

#: Which folded axes the NEAR fill images for each D component. Derived from
#: ``TARGET_IYEE`` at import rather than written out, and pinned by test against
#: ``fields.IYEE_SHIFTS``: axis ``a`` fills exactly the components whose Yee shift
#: there is 0, which for D is every component EXCEPT ``a``.
NEAR_FILL_AXES: Tuple[Tuple[int, ...], ...] = tuple(
    tuple(axis for axis in range(3) if TARGET_IYEE[name][axis] == 0)
    for name in ("Dx", "Dy", "Dz")
)

#: Which folded PERIODIC axes the FAR fill images for each D component. The exact
#: complement of :data:`NEAR_FILL_AXES` -- ``stepping._fill_folded_far_ghosts``
#: (:1489-1532) writes axis ``a`` for every component whose Yee shift there is ONE --
#: and derived from the same table for the same reason. For D that is the component's
#: OWN axis and only it, so the two sets never overlap.
FAR_FILL_AXES: Tuple[Tuple[int, ...], ...] = tuple(
    tuple(axis for axis in range(3) if TARGET_IYEE[name][axis] == 1)
    for name in ("Dx", "Dy", "Dz")
)

#: Does the wall clear land AFTER the near parities and BEFORE the far one? TRUE on
#: this family, and FALSE on the magnetic twin, which is why that module has no such
#: constant. The driver runs fill_symmetry_bc_D (:3300), zero_metal_D (:3301) and
#: fill_folded_far_ghosts_D (:3302) in that order, and on the D family the clear's
#: rows ARE the near fill's axes (both are the component's shift-0 axes), so a near
#: ghost takes the clear after its parity and any far parity lands after the clear.
#: Stated as data so the host suite can assert the kernel's emission order against it
#: rather than reading the order out of the same body it is checking.
CLEAR_AFTER_NEAR: bool = True

#: The probe patterns this product's operand orientations require: the base four,
#: the PARITY one the fills add, AND the beta tranche's fifth
#: (``c8_mul_c8_imaginary_coefficient_left``) that the K3b insert launches --
#: two extended licences over one record, and both halves' probe clauses ask
#: them. Named rather than passed implicitly so the claim is inspectable and
#: testable.
PRODUCT_PROBE_PATTERNS: Tuple[str, ...] = (tuple(PARITY_PROBE_PATTERNS)
                                           + (BETA_PROBE_PATTERN,))

#: The complex-multiply device functions this kernel is allowed to call, and the only
#: ones. A fifth would be a fifth operand orientation, which would need its own probe
#: pattern before it could be licensed; the test scans for exactly this.
LICENSED_MULTIPLY_HELPERS: Tuple[str, ...] = (
    "_rotate_field_left", "_mul_field_left", "_mul_coefficient_left",
    "_mul_imag_coefficient_left")

#: What has and has not been executed on a device. Edited only by a released gate.
DEVICE_STATUS: str = (
    "NOT YET RUN. This module was created 2026-09-02 (the fusion-residue round, "
    "audit §1.4) and no device has executed it; its gate is "
    "parity/meep_gpu/probe_triton_folded_beta_complex_fused_pair.py, and only that "
    "gate's released artifact may edit this constant. Until it runs, this module "
    "licenses NOTHING: the predicate answers on the laptop, the plan builder "
    "answers None without Triton, and no byte-identity claim exists anywhere in "
    "this file.")

__all__ = [
    "BACKWARD", "CARRIES_DEPOSIT_REPAIR", "CLEAR_AFTER_NEAR", "CONSTITUTIVE_SIDE",
    "CURL_SUB_STEP", "DEVICE_STATUS", "FAR_FILL_AXES", "FILL_FAMILY",
    "LICENSED_MULTIPLY_HELPERS", "MIRROR_CODES", "NEAR_FILL_AXES",
    "PRODUCT_PROBE_PATTERNS", "REPLACES",
    "FoldedBetaComplexFusedPairPlan",
    "carried_destinations",
    "folded_beta_complex_fused_curl_constitutive_D",
    "folded_beta_complex_fused_curl_constitutive_D_kernel",
    "folded_beta_complex_fused_pair_coverage",
    "parity_chain",
    "parity_coefficient_words",
    "plan_folded_beta_complex_fused_pair",
    "plan_folded_beta_complex_fused_pair_from_arrays",
]


# ---------------------------------------------------------------------------
# The ownership rule and the parity chain — host side, so they can be READ
# ---------------------------------------------------------------------------

def carried_destinations(near: Sequence[int], far: Sequence[int]
                         ) -> Tuple[Tuple[Tuple[int, ...], Tuple[int, ...]], ...]:
    """Every ghost cell ONE source lane owns, as ``(near axes, far axes)``.

    THE OWNERSHIP RULE. The driver runs the near fill, the wall clear and the far fill
    in that order (driver.py:3300-3302), and the near fill loops its axes in ascending
    order, so a cell the fills leave at stored 0 of SEVERAL near axes and at the top of
    the far one is written more than once. Back-substituting every pass gives ONE
    source cell per ghost — the cell at stored :data:`.folded_complex.MIRROR_SOURCE_INDEX`
    on each near axis and at the reflect row on the far one — so one lane computes the
    displacement every one of those ghosts carries and writes them all.

    THE D SHAPE IS THE INVERSE OF THE B ONE: up to TWO near axes and at most ONE far
    axis, against the magnetic twin's one near and up to two far. With both near axes
    and the far axis folded a component owns SEVEN ghosts.

    Returned smallest-subset-first for a stable emission order; the order the blocks
    are emitted in is unobservable (the cells are distinct) and a stable one keeps a
    source diff readable.
    """
    near = tuple(int(axis) for axis in near)
    far = tuple(int(axis) for axis in far)
    combos: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []
    for near_size in range(len(near) + 1):
        for near_subset in itertools.combinations(near, near_size):
            for far_size in range(len(far) + 1):
                for far_subset in itertools.combinations(far, far_size):
                    if not near_subset and not far_subset:
                        continue  # the lane's OWN cell, not a ghost
                    combos.append((near_subset, far_subset))
    return tuple(sorted(
        combos, key=lambda item: (len(item[0]) + len(item[1]), item[0], item[1])))


def parity_chain(near_subset: Sequence[int], far_subset: Sequence[int]
                 ) -> Tuple[Tuple[int, str], ...]:
    """The ORDERED ``(axis, pass)`` parity applications one ghost's value carries.

    THE ORDER IS THE DRIVER'S AND IS TRANSCRIBED, NEVER CHOSEN. Each pass reads the
    plane the previous one wrote, so back-substituting a corner gives

        c_far[f] (x) ( CLEAR ( c_near[a2] (x) ( c_near[a1] (x) v ) ) ),  a1 < a2

    — the near applications INNERMOST and ASCENDING (``_fill_symmetry_ghost_cells``
    loops its axes in x, y, z order, stepping.py:1441-1447), then the wall clear
    (:3301, and see :data:`CLEAR_AFTER_NEAR`), then the far axis.

    THE CLEAR IS NOT IN THE RETURNED CHAIN, because it is not a parity: it is a
    ``where`` on the value between the two groups. :data:`CLEAR_AFTER_NEAR` states
    where it lands and the host suite asserts the kernel emits it there. A FAR-ONLY
    ghost takes no clear of its own -- the value it images is the owned register,
    which has already taken the clear -- and a NEAR ghost takes one after its parity.

    A single-entry chain is the common case and the only one the two corpus rows this
    carry admits ever reach — ``special_kz_2_21_2`` and ``triangular_lattice_oblique``
    each fold ONE axis, which gives two components a near ghost and one a far ghost.
    The longer chains are reachable configuration space all the same and the gate
    carries those cases rather than leaving the ordering untested because the corpus
    does not press on it.
    """
    chain: List[Tuple[int, str]] = []
    previous = -1
    for axis in near_subset:
        axis = int(axis)
        if axis <= previous:
            raise AssertionError(
                f"the near axis subset {tuple(near_subset)!r} is not strictly "
                f"ascending; stepping._fill_symmetry_ghost_cells applies its axes in "
                f"ascending order (:1441-1447) and this chain transcribes that order")
        previous = axis
        chain.append((axis, "near"))
    previous = -1
    for axis in far_subset:
        axis = int(axis)
        if axis <= previous:
            raise AssertionError(
                f"the far axis subset {tuple(far_subset)!r} is not strictly "
                f"ascending; stepping._fill_folded_far_ghosts applies its axes in "
                f"ascending order (:1518) and this chain transcribes that order")
        previous = axis
        chain.append((axis, "far"))
    return tuple(chain)


def parity_coefficient_words(grid: Any) -> Tuple[Tuple[float, float], ...]:
    """The six ``(re, im)`` word pairs the kernel takes — near x/y/z then far x/y/z.

    Every pair is :func:`.folded_complex.mirror_parity_coefficients`' output for that
    axis's declared phase, HOST-ROUNDED through ``numpy.complex64`` exactly once, and
    ``(+0.0, +0.0)`` on an unfolded axis, where no block that reads it is emitted.

    ``mirror_parity_coefficients`` returns ``(+phase, -phase)`` — the shift-0 parity
    and the shift-1 one (``fields.mirror_parity`` :117 is ``phase * (1 - 2 * iyee)``).
    On the D family the NEAR fill is the shift-0 pass and the FAR fill the shift-1
    one, exactly as on the B family, so the same pair serves both sides and the
    near/far split is about WHICH AXIS, never about which word.

    A zero pair on a FOLDED axis would be a whole plane of exact zeros — loud — rather
    than a plausible field, which is the same choice
    :func:`.folded_fused_pair.mirror_phases` makes for its constexpr.
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

    PERIODIC = tl.constexpr(CODE_PERIODIC)
    MIRROR_METALLIC = tl.constexpr(CODE_MIRROR_METALLIC)
    MIRROR_PERIODIC = tl.constexpr(CODE_MIRROR_PERIODIC)

    @triton.jit
    def _carry_ghost_complex_E(f, w, e, ie, dst, ghost_re, ghost_im,
                               kp_d, km_d, mask, EXPANSION: tl.constexpr):
        """Write ONE imaged ghost cell of ``D`` and run complex ``update_E`` there.

        THE STATEMENTS ARE THE CERTIFIED ONES, MOVED AND NOT REWRITTEN: they are the
        displacement store followed by
        :func:`.complex_fields.bloch_constitutive_step`'s ``SCALE = 1`` arm — reached
        through :mod:`.complex_fused_electric_pair`'s transcription of it — with
        ``src`` bound to the carried word pair instead of a load, which is exactly
        what the owned cell above does. They are a device function because a D
        component can be the source of up to SEVEN ghost cells and seven inlined
        copies of the same eleven statements are seven places for one of them to
        drift; the real twin's :func:`.folded_fused_pair._carry_ghost_E` made the same
        move for the same reason.

        ``ghost`` is ALREADY the composed value — every parity applied, and the wall
        clear applied or not according to which fill wrote it. This function decides
        nothing about either; it stores what it is handed.

        The ORDER is the array path's: the displacement store, then the workspace
        read, the ``inv_eps`` multiply, the workspace write, and the ``E``
        accumulation. ``prev`` is read BEFORE the write, which is the one ordering the
        constitutive cannot survive being wrong about (S:2083-2085). The inv_eps
        multiply happens BETWEEN the source read and that store
        (complex_fields.py:783-786), so ``f_w_E*`` holds ``D * inv_eps`` and not
        ``D``. The two accumulations stay separate and left-to-right, each a
        coefficient-LEFT zero-imaginary complex product (S:2086-2087, S:2093-2095).

        ``inv_eps`` is a float32 volume at the COMPLEX cell index — ``ie + dst``, not
        ``ie + 2 * dst``. D IS ON THE LEFT of that multiply because the array path
        writes ``source * fields.inverse_epsilon_for(component)``
        (stepping.py:1011-1013).
        """
        tl.store(f + 2 * dst, ghost_re, mask=mask)
        tl.store(f + 2 * dst + 1, ghost_im, mask=mask)
        prev_re = tl.load(w + 2 * dst, mask=mask, other=0.0)
        prev_im = tl.load(w + 2 * dst + 1, mask=mask, other=0.0)
        src_re, src_im = _mul_field_left(
            ghost_re, ghost_im, tl.load(ie + dst, mask=mask, other=0.0), EXPANSION)
        tl.store(w + 2 * dst, src_re, mask=mask)
        tl.store(w + 2 * dst + 1, src_im, mask=mask)
        acc_re = tl.load(e + 2 * dst, mask=mask, other=0.0)
        acc_im = tl.load(e + 2 * dst + 1, mask=mask, other=0.0)
        t_re, t_im = _mul_coefficient_left(kp_d, src_re, src_im, EXPANSION)
        acc_re = acc_re + t_re
        acc_im = acc_im + t_im
        t_re, t_im = _mul_coefficient_left(km_d, prev_re, prev_im, EXPANSION)
        acc_re = acc_re - t_re
        acc_im = acc_im - t_im
        tl.store(e + 2 * dst, acc_re, mask=mask)
        tl.store(e + 2 * dst + 1, acc_im, mask=mask)

    @triton.jit
    def folded_beta_complex_fused_curl_constitutive_D(
        f0, f1, f2,                       # curl targets: Dx,Dy,Dz (complex64 words)
        u0, u1, u2,                       # curl auxiliaries: fu_Dx,fu_Dy,fu_Dz
        g0, g1, g2,                       # curl sources: Hx,Hy,Hz
        kmx, sinvx, kmy, sinvy, kmz, sinvz,   # curl kms/sinv, INTEGER lattice
        h0, h1, h2,                       # constitutive targets: Ex,Ey,Ez
        w0, w1, w2,                       # constitutive aux: f_w_Ex,f_w_Ey,f_w_Ez
        ie0, ie1, ie2,                    # inverse epsilon, float32 VOLUMES
        kp0, km0, kp1, km1, kp2, km2,     # constitutive kps/kms, HALF-INTEGER
        nx, ny, nz, n_elem, dtdx,         # n_elem = COMPLEX cells; dtdx pre-rounded
        pxr, pxi, pyr, pyi, pzr, pzi,     # per-axis complex64-rounded Bloch phase
        bp_re, bp_im, bm_re, bm_im,       # complex64-rounded +-sign beta coefficients (S:770-784)
        rx, ry, rz,                       # RUNTIME: stepping._far_reflect_rows
        n0r, n0i, n1r, n1i, n2r, n2i,     # NEAR parity words, per axis (+phase)
        d0r, d0i, d1r, d1i, d2r, d2i,     # FAR parity words, per axis (-phase)
        BACKWARD: tl.constexpr,           # bound to 1 by every builder; see below
        BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
        NEAR_X: tl.constexpr, NEAR_Y: tl.constexpr, NEAR_Z: tl.constexpr,
        FAR_X: tl.constexpr, FAR_Y: tl.constexpr, FAR_Z: tl.constexpr,
        ZM_X: tl.constexpr, ZM_Y: tl.constexpr, ZM_Z: tl.constexpr,
        PHX: tl.constexpr, PHY: tl.constexpr, PHZ: tl.constexpr,
        HAS_BETA: tl.constexpr,           # compiled-in only for beta != 0 runs
        EXPANSION: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Folded complex ``step_D`` + both fills + ``zero_metal_D`` + ``update_E``.

        ``BACKWARD`` IS CARRIED AND MUST BE 1. It keeps the curl half a verbatim copy
        of :func:`.folded_complex.folded_bloch_pml_curl_step` rather than a
        hand-specialised one, which is the whole reason the transcription risk here is
        low. It cannot be 0: the B family's Yee shifts swap the two fills' roles
        exactly, its wall clear is three rows and never meets a fill, its constitutive
        half reads no ``inv_eps``, and its census funnel is a different number.

        ``BCX``/``BCY``/``BCZ`` take all four of :mod:`.folded_complex`'s codes,
        ``MIRROR_PERIODIC`` included: this kernel carries
        ``fill_folded_far_ghosts_D`` (driver.py:3302), and the folded curl's top-plane
        mask comes with it.

        ``NEAR_a`` / ``FAR_a`` say which fill runs on axis ``a``: NEAR on either
        mirror code (``stepping._fill_symmetry_ghost_cells`` gates on the mirror
        phases alone) and FAR only on ``MIRROR_PERIODIC``
        (``stepping._stored_past_owned``:1454-1469). They are DERIVED FROM ``bc`` by
        the plan and are not independent inputs, so they cannot disagree with the
        codes the curl half branches on.

        ``rx``/``ry``/``rz`` are ``stepping._far_reflect_rows``' answer per axis,
        RUNTIME as they are in
        :func:`.folded_complex.folded_mirror_ghost_fill_complex` and for its reason:
        the row is ``n_full - stored + 2``, which is ``stored - 2`` at an even full
        count and ``stored - 3`` at an odd one, so baking ``n - 2`` reflects about the
        window top instead of about the second mirror and is a whole cell wrong on
        every odd-count run. ``-1`` on an axis with no far ghost, where no lane reads
        it.

        ``n?r``/``n?i``/``d?r``/``d?i`` are the folded axes' PARITY COEFFICIENT WORDS,
        host-rounded through ``numpy.complex64`` by
        :func:`.folded_complex.mirror_parity_coefficients` and PASSED, never
        synthesised in-kernel (special_kz.py:255-264's rule). ``(+0.0, +0.0)`` on an
        unfolded axis, where no block that reads them is emitted: if a carry block
        ever fired on an axis the host did not classify as folded it would write an
        exact zero plane — loud — rather than a plausible field.

        ``PHX``/``PHY``/``PHZ`` are the BLOCH phase flags, not the mirror phases: the
        mirror phase rides entirely in the words above. A folded axis may carry no
        Bloch phase at all and the predicate refuses one that does.

        ``ZM_X``/``ZM_Y``/``ZM_Z`` are ``coverage.zero_metal_axes``, the question
        ``stepping._zero_metal`` asks — the grid's own declaration, NOT the resolved
        ghost rule, which an invariant axis softens to periodic while the wall clear
        still fires there.
        """
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = idx < n_elem
        nyz = ny * nz
        k = idx % nz
        plane = idx // nz
        j = plane % ny
        i = plane // ny

        # ======================= the curl half ================================
        # Verbatim from folded_complex.folded_bloch_pml_curl_step; the only edits
        # below its stores are the ownership masks the two carries need.

        # --- the ghost rule, per axis (stepping._shift_up / _shift_down) -------
        # PERIODIC wraps; EVERY other rule here serves an exact 0.0 past the face,
        # which `tl.load`'s `other=` delivers on BOTH words without dereferencing
        # anything. On a folded axis that zero stands in for a value nothing reads.
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

        # --- wrapped-lane predicates, one plane per axis ------------------------
        if BACKWARD:
            wx, wy, wz = i == 0, j == 0, k == 0
        else:
            wx, wy, wz = i == nx - 1, j == ny - 1, k == nz - 1

        # --- loads: two words per operand --------------------------------------
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

        # --- Bloch phase on the wrapped lane, BEFORE the difference -------------
        # SHIFTED operands only; PH* is 0 on every folded axis by the predicate.
        # Field LEFT (S:1862); `tl.where` is a bitwise select, so unwrapped lanes
        # keep the loaded words untouched.
        if PHX:
            rot_re, rot_im = _rotate_field_left(b_x_re, b_x_im, pxr, pxi, EXPANSION)
            b_x_re = tl.where(wx, rot_re, b_x_re)
            b_x_im = tl.where(wx, rot_im, b_x_im)
            rot_re, rot_im = _rotate_field_left(c_x_re, c_x_im, pxr, pxi, EXPANSION)
            c_x_re = tl.where(wx, rot_re, c_x_re)
            c_x_im = tl.where(wx, rot_im, c_x_im)
        if PHY:
            rot_re, rot_im = _rotate_field_left(a_y_re, a_y_im, pyr, pyi, EXPANSION)
            a_y_re = tl.where(wy, rot_re, a_y_re)
            a_y_im = tl.where(wy, rot_im, a_y_im)
            rot_re, rot_im = _rotate_field_left(c_y_re, c_y_im, pyr, pyi, EXPANSION)
            c_y_re = tl.where(wy, rot_re, c_y_re)
            c_y_im = tl.where(wy, rot_im, c_y_im)
        if PHZ:
            rot_re, rot_im = _rotate_field_left(a_z_re, a_z_im, pzr, pzi, EXPANSION)
            a_z_re = tl.where(wz, rot_re, a_z_re)
            a_z_im = tl.where(wz, rot_im, a_z_im)
            rot_re, rot_im = _rotate_field_left(b_z_re, b_z_im, pzr, pzi, EXPANSION)
            b_z_re = tl.where(wz, rot_re, b_z_re)
            b_z_im = tl.where(wz, rot_im, b_z_im)

        # --- the curl (stepping._curl_from_operands): DO NOT flatten these parens
        t0_re = ((c_y_re - c_re) + (b_re - b_z_re))
        t0_im = ((c_y_im - c_im) + (b_im - b_z_im))
        t1_re = ((a_z_re - a_re) + (c_re - c_x_re))
        t1_im = ((a_z_im - a_im) + (c_im - c_x_im))
        t2_re = ((b_x_re - b_re) + (a_re - a_y_re))
        t2_im = ((b_x_im - b_im) + (a_im - a_y_im))
        curl0_re, curl0_im = _mul_coefficient_left(dtdx, t0_re, t0_im, EXPANSION)
        curl1_re, curl1_im = _mul_coefficient_left(dtdx, t1_re, t1_im, EXPANSION)
        curl2_re, curl2_im = _mul_coefficient_left(dtdx, t2_re, t2_im, EXPANSION)

        # --- the beta term (stepping._special_kz_beta_term), CENTER partners only ---
        # Verbatim from folded_complex.folded_beta_bloch_pml_curl_step (K3b): AFTER
        # the dtdx curl, BEFORE BOTH ownership masks -- the array path's order
        # (S:356-363 against S:369). Coefficient LEFT (S:784); the partners are the
        # CENTER word pairs loaded above, and the Bloch section never rotates them.
        if HAS_BETA:
            t_re, t_im = _mul_imag_coefficient_left(bp_re, bp_im, b_re, b_im, EXPANSION)
            curl0_re = curl0_re - t_re
            curl0_im = curl0_im - t_im
            t_re, t_im = _mul_imag_coefficient_left(bm_re, bm_im, a_re, a_im, EXPANSION)
            curl1_re = curl1_re - t_re
            curl1_im = curl1_im - t_im

        # --- ownership mask, cell 0 (stepping._mask_non_owned_cells) -----------
        # `!= PERIODIC` rather than `== METALLIC`: _mask_non_owned_cells asks
        # `is_mirrored or is_metallic or is_axis`. Writes +0.0 to BOTH planes
        # (S:1896, S:1902).
        at_x, at_y, at_z = i == 0, j == 0, k == 0
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

        # --- ownership mask, the TOP plane of a folded PERIODIC axis -----------
        # folded_bloch_pml_curl_step's DELTA 3, byte-copied from its BACKWARD == 1
        # arm. Dx:(1,0,0) Dy:(0,1,0) Dz:(0,0,1) — shift 1 on its OWN axis only,
        # which is exactly the set the far carry writes, and THREE lines against the
        # B arm's six.
        last_x, last_y, last_z = i == nx - 1, j == ny - 1, k == nz - 1
        if BACKWARD:
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

        # --- ownership of the fill destinations --------------------------------
        # Component m is a NEAR destination at stored cell 0 of every folded axis
        # that is NOT its own, and a FAR destination at the top plane of its own axis
        # when that axis is folded PERIODIC (IYEE_SHIFTS: Dx (1,0,0), Dy (0,1,0),
        # Dz (0,0,1)). Those lanes do NOT load or store D, E or f_w_E: the source
        # lane writes all three for them. The masks are what make the carry race-free
        # — a lane that merely discarded the value would still have READ a word
        # another lane writes.
        own0, own1, own2 = live, live, live
        if NEAR_Y:
            own0 = own0 & (j != 0)
            own2 = own2 & (j != 0)
        if NEAR_Z:
            own0 = own0 & (k != 0)
            own1 = own1 & (k != 0)
        if NEAR_X:
            own1 = own1 & (i != 0)
            own2 = own2 & (i != 0)
        if FAR_X:
            own0 = own0 & (i != nx - 1)
        if FAR_Y:
            own1 = own1 & (j != ny - 1)
        if FAR_Z:
            own2 = own2 & (k != nz - 1)

        # --- split-field recurrence (stepping._apply_pml_update) ---------------
        # Byte-copied from folded_bloch_pml_curl_step. The PML coefficient vectors
        # are built at the STORED extent on a folded axis, so the per-axis indexing
        # needs no fold-aware change. The only edit is the `f?` load's mask: a
        # destination lane never reads D.
        km_x = tl.load(kmx + i, mask=live, other=0.0)
        si_x = tl.load(sinvx + i, mask=live, other=0.0)
        km_y = tl.load(kmy + j, mask=live, other=0.0)
        si_y = tl.load(sinvy + j, mask=live, other=0.0)
        km_z = tl.load(kmz + k, mask=live, other=0.0)
        si_z = tl.load(sinvz + k, mask=live, other=0.0)

        p0_re = tl.load(u0 + 2 * idx, mask=live, other=0.0)
        p0_im = tl.load(u0 + 2 * idx + 1, mask=live, other=0.0)
        q_re, q_im = _mul_field_left(p0_re, p0_im, km_y, EXPANSION)
        q_re = q_re - curl0_re
        q_im = q_im - curl0_im
        n0_re, n0_im = _mul_field_left(q_re, q_im, si_y, EXPANSION)
        e_re = tl.load(f0 + 2 * idx, mask=own0, other=0.0)
        e_im = tl.load(f0 + 2 * idx + 1, mask=own0, other=0.0)
        r_re, r_im = _mul_field_left(e_re, e_im, km_z, EXPANSION)
        r_re = (r_re + n0_re) - p0_re
        r_im = (r_im + n0_im) - p0_im
        v0_re, v0_im = _mul_field_left(r_re, r_im, si_z, EXPANSION)

        p1_re = tl.load(u1 + 2 * idx, mask=live, other=0.0)
        p1_im = tl.load(u1 + 2 * idx + 1, mask=live, other=0.0)
        q_re, q_im = _mul_field_left(p1_re, p1_im, km_z, EXPANSION)
        q_re = q_re - curl1_re
        q_im = q_im - curl1_im
        n1_re, n1_im = _mul_field_left(q_re, q_im, si_z, EXPANSION)
        e_re = tl.load(f1 + 2 * idx, mask=own1, other=0.0)
        e_im = tl.load(f1 + 2 * idx + 1, mask=own1, other=0.0)
        r_re, r_im = _mul_field_left(e_re, e_im, km_x, EXPANSION)
        r_re = (r_re + n1_re) - p1_re
        r_im = (r_im + n1_im) - p1_im
        v1_re, v1_im = _mul_field_left(r_re, r_im, si_x, EXPANSION)

        p2_re = tl.load(u2 + 2 * idx, mask=live, other=0.0)
        p2_im = tl.load(u2 + 2 * idx + 1, mask=live, other=0.0)
        q_re, q_im = _mul_field_left(p2_re, p2_im, km_x, EXPANSION)
        q_re = q_re - curl2_re
        q_im = q_im - curl2_im
        n2_re, n2_im = _mul_field_left(q_re, q_im, si_x, EXPANSION)
        e_re = tl.load(f2 + 2 * idx, mask=own2, other=0.0)
        e_im = tl.load(f2 + 2 * idx + 1, mask=own2, other=0.0)
        r_re, r_im = _mul_field_left(e_re, e_im, km_y, EXPANSION)
        r_re = (r_re + n2_re) - p2_re
        r_im = (r_im + n2_im) - p2_im
        v2_re, v2_im = _mul_field_left(r_re, r_im, si_y, EXPANSION)

        # --- the seam: stepping.zero_metal_D (driver.py:3301) ------------------
        # THE SIX ROWS `_zero_metal` writes for the D family (Dy/Dz on an x wall,
        # Dx/Dz on y, Dx/Dy on z), against the magnetic twin's three. Applied to the
        # REGISTERS, before the store, before the constitutive read and before every
        # carry, so each consumer sees the one value the array path leaves in D.
        # BOTH planes: `array[_face(axis, 0)] = 0` on a complex64 volume writes
        # complex zero (stepping._zero_metal :2246). A walled axis is never a folded
        # axis (:2237-2239), so these lines cannot touch a fill destination; the
        # predicate CHECKS that rather than leaning on it. It does NOT follow that
        # the clear and the near fill are unrelated on this family — a component
        # folded on y is still cleared on a z wall — and that is why each near-ghost
        # block below RE-APPLIES these lines after its parity, in the driver's order.
        if ZM_X:
            v1_re = tl.where(at_x, 0.0, v1_re)
            v1_im = tl.where(at_x, 0.0, v1_im)
            v2_re = tl.where(at_x, 0.0, v2_re)
            v2_im = tl.where(at_x, 0.0, v2_im)
        if ZM_Y:
            v0_re = tl.where(at_y, 0.0, v0_re)
            v0_im = tl.where(at_y, 0.0, v0_im)
            v2_re = tl.where(at_y, 0.0, v2_re)
            v2_im = tl.where(at_y, 0.0, v2_im)
        if ZM_Z:
            v0_re = tl.where(at_z, 0.0, v0_re)
            v0_im = tl.where(at_z, 0.0, v0_im)
            v1_re = tl.where(at_z, 0.0, v1_re)
            v1_im = tl.where(at_z, 0.0, v1_im)

        # --- stores: u then f (kernels.py:193-198 order), both planes ----------
        # `fu` is written at EVERY cell, destinations included: the array path's
        # step_D writes it everywhere and neither fill touches it
        # (stepping.py:1497-1498 writes `field`, never `fu_field`).
        tl.store(u0 + 2 * idx, n0_re, mask=live)
        tl.store(u0 + 2 * idx + 1, n0_im, mask=live)
        tl.store(u1 + 2 * idx, n1_re, mask=live)
        tl.store(u1 + 2 * idx + 1, n1_im, mask=live)
        tl.store(u2 + 2 * idx, n2_re, mask=live)
        tl.store(u2 + 2 * idx + 1, n2_im, mask=live)
        tl.store(f0 + 2 * idx, v0_re, mask=own0)
        tl.store(f0 + 2 * idx + 1, v0_im, mask=own0)
        tl.store(f1 + 2 * idx, v1_re, mask=own1)
        tl.store(f1 + 2 * idx + 1, v1_im, mask=own1)
        tl.store(f2 + 2 * idx, v2_re, mask=own2)
        tl.store(f2 + 2 * idx + 1, v2_im, mask=own2)

        # ==================== the constitutive half ===========================
        # Verbatim from complex_fields.bloch_constitutive_step's SCALE=1 arm, through
        # complex_fused_electric_pair's transcription of it, with
        # `src = tl.load(g + 2*idx)` replaced by the register pair the curl half just
        # produced. Component 0 takes its coefficient from axis x, 1 from y, 2 from z
        # (stepping.E_CONSTITUTIVE_TERMS :227) — the component's OWN axis, which is
        # exactly the axis the NEAR fill never images along and exactly the one the
        # FAR fill does.
        kp_0 = tl.load(kp0 + i, mask=live, other=0.0)
        km_0 = tl.load(km0 + i, mask=live, other=0.0)
        kp_1 = tl.load(kp1 + j, mask=live, other=0.0)
        km_1 = tl.load(km1 + j, mask=live, other=0.0)
        kp_2 = tl.load(kp2 + k, mask=live, other=0.0)
        km_2 = tl.load(km2 + k, mask=live, other=0.0)

        # --- component 0 -------------------------------------------------------
        prev_re = tl.load(w0 + 2 * idx, mask=own0, other=0.0)   # BEFORE the store.
        prev_im = tl.load(w0 + 2 * idx + 1, mask=own0, other=0.0)
        # inv_eps is a float32 volume at the COMPLEX cell index — `+ idx`, not
        # `+ 2 * idx`. D LEFT (S:982-984).
        src_re, src_im = _mul_field_left(
            v0_re, v0_im, tl.load(ie0 + idx, mask=own0, other=0.0), EXPANSION)
        tl.store(w0 + 2 * idx, src_re, mask=own0)
        tl.store(w0 + 2 * idx + 1, src_im, mask=own0)
        a_re = tl.load(h0 + 2 * idx, mask=own0, other=0.0)
        a_im = tl.load(h0 + 2 * idx + 1, mask=own0, other=0.0)
        t_re, t_im = _mul_coefficient_left(kp_0, src_re, src_im, EXPANSION)
        a_re = a_re + t_re
        a_im = a_im + t_im
        t_re, t_im = _mul_coefficient_left(km_0, prev_re, prev_im, EXPANSION)
        a_re = a_re - t_re
        a_im = a_im - t_im
        tl.store(h0 + 2 * idx, a_re, mask=own0)
        tl.store(h0 + 2 * idx + 1, a_im, mask=own0)

        # --- component 1 -------------------------------------------------------
        prev_re = tl.load(w1 + 2 * idx, mask=own1, other=0.0)
        prev_im = tl.load(w1 + 2 * idx + 1, mask=own1, other=0.0)
        src_re, src_im = _mul_field_left(
            v1_re, v1_im, tl.load(ie1 + idx, mask=own1, other=0.0), EXPANSION)
        tl.store(w1 + 2 * idx, src_re, mask=own1)
        tl.store(w1 + 2 * idx + 1, src_im, mask=own1)
        a_re = tl.load(h1 + 2 * idx, mask=own1, other=0.0)
        a_im = tl.load(h1 + 2 * idx + 1, mask=own1, other=0.0)
        t_re, t_im = _mul_coefficient_left(kp_1, src_re, src_im, EXPANSION)
        a_re = a_re + t_re
        a_im = a_im + t_im
        t_re, t_im = _mul_coefficient_left(km_1, prev_re, prev_im, EXPANSION)
        a_re = a_re - t_re
        a_im = a_im - t_im
        tl.store(h1 + 2 * idx, a_re, mask=own1)
        tl.store(h1 + 2 * idx + 1, a_im, mask=own1)

        # --- component 2 -------------------------------------------------------
        prev_re = tl.load(w2 + 2 * idx, mask=own2, other=0.0)
        prev_im = tl.load(w2 + 2 * idx + 1, mask=own2, other=0.0)
        src_re, src_im = _mul_field_left(
            v2_re, v2_im, tl.load(ie2 + idx, mask=own2, other=0.0), EXPANSION)
        tl.store(w2 + 2 * idx, src_re, mask=own2)
        tl.store(w2 + 2 * idx + 1, src_im, mask=own2)
        a_re = tl.load(h2 + 2 * idx, mask=own2, other=0.0)
        a_im = tl.load(h2 + 2 * idx + 1, mask=own2, other=0.0)
        t_re, t_im = _mul_coefficient_left(kp_2, src_re, src_im, EXPANSION)
        a_re = a_re + t_re
        a_im = a_im + t_im
        t_re, t_im = _mul_coefficient_left(km_2, prev_re, prev_im, EXPANSION)
        a_re = a_re - t_re
        a_im = a_im - t_im
        tl.store(h2 + 2 * idx, a_re, mask=own2)
        tl.store(h2 + 2 * idx + 1, a_im, mask=own2)

        # ========= the mirror fills, carried by the SOURCE lane ================
        # stepping._write_mirror_ghost (:1451):     D[..0..]  = (+phase) (x) D[..2..]
        # stepping._fill_folded_far_ghosts (:1529): D[..-1..] = (-phase) (x) D[..r..]
        # and every COMPOSITION of the two, applied ONE
        # `_mul_imag_coefficient_left` PER PASS in the DRIVER'S OWN ORDER (near axes
        # ascending, then the WALL CLEAR, then the far axis) — see parity_chain,
        # CLEAR_AFTER_NEAR and the module docstring. NOT folded into one coefficient
        # and NOT re-ordered: under complex storage both moves change bytes, measured.
        #
        # THE COEFFICIENT INDEX MOVES ONLY FOR THE FAR HALF. update_E indexes
        # component m on axis m (stepping.E_CONSTITUTIVE_TERMS :227); the near fill
        # images along an axis that is NOT m, so its destination sits at the SAME
        # coefficient index as the lane that owns it and reuses that lane's pair — a
        # reload there would reload the identical word. The far fill images along
        # exactly axis m, so its destination reads kps/kms at the TOP row rather than
        # reusing the source lane's at the reflect row.
        #
        # THE NEAR SOURCE INDEX IS THE LITERAL 2, as it is in
        # folded_complex.folded_mirror_ghost_fill_complex (`base + 2 * stride`),
        # rather than the named MIRROR_SOURCE_INDEX: a jit body that closes over a
        # module-level Python int is a Triton-version question this file has no way to
        # measure without a device. The literal is pinned to the name by test.
        #
        # A scalar `nx - 1` would not broadcast against a masked vector load, so the
        # far destination's coefficient row is built as a per-lane tensor the same way
        # the real twin builds its `top_x`.
        top_x = idx * 0 + (nx - 1)
        top_y = idx * 0 + (ny - 1)
        top_z = idx * 0 + (nz - 1)
        kp_f0 = tl.load(kp0 + top_x, mask=live, other=0.0)
        km_f0 = tl.load(km0 + top_x, mask=live, other=0.0)
        kp_f1 = tl.load(kp1 + top_y, mask=live, other=0.0)
        km_f1 = tl.load(km1 + top_y, mask=live, other=0.0)
        kp_f2 = tl.load(kp2 + top_z, mask=live, other=0.0)
        km_f2 = tl.load(km2 + top_z, mask=live, other=0.0)

        # The per-axis source lane and destination offset, once. EVERY CARRY MASK IS
        # ANDED WITH THE COMPONENT'S OWN OWNERSHIP MASK — a DEFECT THE REAL BOARD'S
        # DEVICE GATE FOUND rather than a precaution (results/run_farcarry2/,
        # 2026-08-20): with two fills a lane can be the source of one and the
        # DESTINATION of the other, and it would then write the composite cell from a
        # `v` built on a load its own ownership mask zeroed.
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

        # THE THREE-TERM GUARDS ARE PARENTHESISED, AND THAT IS A MEASURED PLATFORM
        # FACT rather than a style choice: Triton 3.1.0's frontend refuses a chained
        # boolean outright — "chained boolean operators (A or B or C) are not
        # supported; use parentheses to split the chain" — and it refuses at COMPILE
        # time on the device, so `if NEAR_Y and NEAR_Z and FAR_X:` is a kernel that
        # does not build rather than one that builds wrong.

        # --- component 0 (Dx): near on y and z, far on x -----------------------
        # EVERY BLOCK IS SELF-CONTAINED: the chain is recomposed from `v0` inside the
        # block rather than reusing a register a sibling constexpr branch defined.
        # That is the complex magnetic twin's convention and it is deliberate — the
        # value is the same word either way, and a block that carries its whole chain
        # can be PARSED, which is what lets the suite and the gate read the emitted
        # order off the shipped body instead of re-deriving it. The ZM lines inside
        # each near chain are zero_metal_D's rows for THIS component (Dx clears on a y
        # wall and on a z wall), applied AFTER the near parities and BEFORE any far
        # one — the driver's order (:3300 near, :3301 clear, :3302 far).
        if FAR_X:
            gc_re, gc_im = _mul_imag_coefficient_left(d0r, d0i, v0_re, v0_im,
                                                      EXPANSION)
            _carry_ghost_complex_E(f0, w0, h0, ie0,
                                   idx + df_x,
                                   gc_re, gc_im, kp_f0, km_f0,
                                   own0 & far_i, EXPANSION)
        if NEAR_Y:
            gc_re, gc_im = _mul_imag_coefficient_left(n1r, n1i, v0_re, v0_im,
                                                      EXPANSION)
            if ZM_Y:
                gc_re = tl.where(at_y, 0.0, gc_re)
                gc_im = tl.where(at_y, 0.0, gc_im)
            if ZM_Z:
                gc_re = tl.where(at_z, 0.0, gc_re)
                gc_im = tl.where(at_z, 0.0, gc_im)
            _carry_ghost_complex_E(f0, w0, h0, ie0,
                                   idx + dn_y,
                                   gc_re, gc_im, kp_0, km_0,
                                   own0 & near_j, EXPANSION)
        if NEAR_Z:
            gc_re, gc_im = _mul_imag_coefficient_left(n2r, n2i, v0_re, v0_im,
                                                      EXPANSION)
            if ZM_Y:
                gc_re = tl.where(at_y, 0.0, gc_re)
                gc_im = tl.where(at_y, 0.0, gc_im)
            if ZM_Z:
                gc_re = tl.where(at_z, 0.0, gc_re)
                gc_im = tl.where(at_z, 0.0, gc_im)
            _carry_ghost_complex_E(f0, w0, h0, ie0,
                                   idx + dn_z,
                                   gc_re, gc_im, kp_0, km_0,
                                   own0 & near_k, EXPANSION)
        if NEAR_Y and FAR_X:
            gc_re, gc_im = _mul_imag_coefficient_left(n1r, n1i, v0_re, v0_im,
                                                      EXPANSION)
            if ZM_Y:
                gc_re = tl.where(at_y, 0.0, gc_re)
                gc_im = tl.where(at_y, 0.0, gc_im)
            if ZM_Z:
                gc_re = tl.where(at_z, 0.0, gc_re)
                gc_im = tl.where(at_z, 0.0, gc_im)
            gc_re, gc_im = _mul_imag_coefficient_left(d0r, d0i, gc_re, gc_im,
                                                      EXPANSION)
            _carry_ghost_complex_E(f0, w0, h0, ie0,
                                   idx + df_x + dn_y,
                                   gc_re, gc_im, kp_f0, km_f0,
                                   own0 & far_i & near_j, EXPANSION)
        if NEAR_Y and NEAR_Z:
            gc_re, gc_im = _mul_imag_coefficient_left(n1r, n1i, v0_re, v0_im,
                                                      EXPANSION)
            gc_re, gc_im = _mul_imag_coefficient_left(n2r, n2i, gc_re, gc_im,
                                                      EXPANSION)
            if ZM_Y:
                gc_re = tl.where(at_y, 0.0, gc_re)
                gc_im = tl.where(at_y, 0.0, gc_im)
            if ZM_Z:
                gc_re = tl.where(at_z, 0.0, gc_re)
                gc_im = tl.where(at_z, 0.0, gc_im)
            _carry_ghost_complex_E(f0, w0, h0, ie0,
                                   idx + dn_y + dn_z,
                                   gc_re, gc_im, kp_0, km_0,
                                   own0 & near_j & near_k, EXPANSION)
        if NEAR_Z and FAR_X:
            gc_re, gc_im = _mul_imag_coefficient_left(n2r, n2i, v0_re, v0_im,
                                                      EXPANSION)
            if ZM_Y:
                gc_re = tl.where(at_y, 0.0, gc_re)
                gc_im = tl.where(at_y, 0.0, gc_im)
            if ZM_Z:
                gc_re = tl.where(at_z, 0.0, gc_re)
                gc_im = tl.where(at_z, 0.0, gc_im)
            gc_re, gc_im = _mul_imag_coefficient_left(d0r, d0i, gc_re, gc_im,
                                                      EXPANSION)
            _carry_ghost_complex_E(f0, w0, h0, ie0,
                                   idx + df_x + dn_z,
                                   gc_re, gc_im, kp_f0, km_f0,
                                   own0 & far_i & near_k, EXPANSION)
        if NEAR_Y and (NEAR_Z and FAR_X):
            gc_re, gc_im = _mul_imag_coefficient_left(n1r, n1i, v0_re, v0_im,
                                                      EXPANSION)
            gc_re, gc_im = _mul_imag_coefficient_left(n2r, n2i, gc_re, gc_im,
                                                      EXPANSION)
            if ZM_Y:
                gc_re = tl.where(at_y, 0.0, gc_re)
                gc_im = tl.where(at_y, 0.0, gc_im)
            if ZM_Z:
                gc_re = tl.where(at_z, 0.0, gc_re)
                gc_im = tl.where(at_z, 0.0, gc_im)
            gc_re, gc_im = _mul_imag_coefficient_left(d0r, d0i, gc_re, gc_im,
                                                      EXPANSION)
            _carry_ghost_complex_E(f0, w0, h0, ie0,
                                   idx + df_x + dn_y + dn_z,
                                   gc_re, gc_im, kp_f0, km_f0,
                                   own0 & far_i & near_j & near_k, EXPANSION)

        # --- component 1 (Dy): near on x and z, far on y -----------------------
        if FAR_Y:
            gc_re, gc_im = _mul_imag_coefficient_left(d1r, d1i, v1_re, v1_im,
                                                      EXPANSION)
            _carry_ghost_complex_E(f1, w1, h1, ie1,
                                   idx + df_y,
                                   gc_re, gc_im, kp_f1, km_f1,
                                   own1 & far_j, EXPANSION)
        if NEAR_X:
            gc_re, gc_im = _mul_imag_coefficient_left(n0r, n0i, v1_re, v1_im,
                                                      EXPANSION)
            if ZM_X:
                gc_re = tl.where(at_x, 0.0, gc_re)
                gc_im = tl.where(at_x, 0.0, gc_im)
            if ZM_Z:
                gc_re = tl.where(at_z, 0.0, gc_re)
                gc_im = tl.where(at_z, 0.0, gc_im)
            _carry_ghost_complex_E(f1, w1, h1, ie1,
                                   idx + dn_x,
                                   gc_re, gc_im, kp_1, km_1,
                                   own1 & near_i, EXPANSION)
        if NEAR_Z:
            gc_re, gc_im = _mul_imag_coefficient_left(n2r, n2i, v1_re, v1_im,
                                                      EXPANSION)
            if ZM_X:
                gc_re = tl.where(at_x, 0.0, gc_re)
                gc_im = tl.where(at_x, 0.0, gc_im)
            if ZM_Z:
                gc_re = tl.where(at_z, 0.0, gc_re)
                gc_im = tl.where(at_z, 0.0, gc_im)
            _carry_ghost_complex_E(f1, w1, h1, ie1,
                                   idx + dn_z,
                                   gc_re, gc_im, kp_1, km_1,
                                   own1 & near_k, EXPANSION)
        if NEAR_X and FAR_Y:
            gc_re, gc_im = _mul_imag_coefficient_left(n0r, n0i, v1_re, v1_im,
                                                      EXPANSION)
            if ZM_X:
                gc_re = tl.where(at_x, 0.0, gc_re)
                gc_im = tl.where(at_x, 0.0, gc_im)
            if ZM_Z:
                gc_re = tl.where(at_z, 0.0, gc_re)
                gc_im = tl.where(at_z, 0.0, gc_im)
            gc_re, gc_im = _mul_imag_coefficient_left(d1r, d1i, gc_re, gc_im,
                                                      EXPANSION)
            _carry_ghost_complex_E(f1, w1, h1, ie1,
                                   idx + df_y + dn_x,
                                   gc_re, gc_im, kp_f1, km_f1,
                                   own1 & far_j & near_i, EXPANSION)
        if NEAR_X and NEAR_Z:
            gc_re, gc_im = _mul_imag_coefficient_left(n0r, n0i, v1_re, v1_im,
                                                      EXPANSION)
            gc_re, gc_im = _mul_imag_coefficient_left(n2r, n2i, gc_re, gc_im,
                                                      EXPANSION)
            if ZM_X:
                gc_re = tl.where(at_x, 0.0, gc_re)
                gc_im = tl.where(at_x, 0.0, gc_im)
            if ZM_Z:
                gc_re = tl.where(at_z, 0.0, gc_re)
                gc_im = tl.where(at_z, 0.0, gc_im)
            _carry_ghost_complex_E(f1, w1, h1, ie1,
                                   idx + dn_x + dn_z,
                                   gc_re, gc_im, kp_1, km_1,
                                   own1 & near_i & near_k, EXPANSION)
        if NEAR_Z and FAR_Y:
            gc_re, gc_im = _mul_imag_coefficient_left(n2r, n2i, v1_re, v1_im,
                                                      EXPANSION)
            if ZM_X:
                gc_re = tl.where(at_x, 0.0, gc_re)
                gc_im = tl.where(at_x, 0.0, gc_im)
            if ZM_Z:
                gc_re = tl.where(at_z, 0.0, gc_re)
                gc_im = tl.where(at_z, 0.0, gc_im)
            gc_re, gc_im = _mul_imag_coefficient_left(d1r, d1i, gc_re, gc_im,
                                                      EXPANSION)
            _carry_ghost_complex_E(f1, w1, h1, ie1,
                                   idx + df_y + dn_z,
                                   gc_re, gc_im, kp_f1, km_f1,
                                   own1 & far_j & near_k, EXPANSION)
        if NEAR_X and (NEAR_Z and FAR_Y):
            gc_re, gc_im = _mul_imag_coefficient_left(n0r, n0i, v1_re, v1_im,
                                                      EXPANSION)
            gc_re, gc_im = _mul_imag_coefficient_left(n2r, n2i, gc_re, gc_im,
                                                      EXPANSION)
            if ZM_X:
                gc_re = tl.where(at_x, 0.0, gc_re)
                gc_im = tl.where(at_x, 0.0, gc_im)
            if ZM_Z:
                gc_re = tl.where(at_z, 0.0, gc_re)
                gc_im = tl.where(at_z, 0.0, gc_im)
            gc_re, gc_im = _mul_imag_coefficient_left(d1r, d1i, gc_re, gc_im,
                                                      EXPANSION)
            _carry_ghost_complex_E(f1, w1, h1, ie1,
                                   idx + df_y + dn_x + dn_z,
                                   gc_re, gc_im, kp_f1, km_f1,
                                   own1 & far_j & near_i & near_k, EXPANSION)

        # --- component 2 (Dz): near on x and y, far on z -----------------------
        if FAR_Z:
            gc_re, gc_im = _mul_imag_coefficient_left(d2r, d2i, v2_re, v2_im,
                                                      EXPANSION)
            _carry_ghost_complex_E(f2, w2, h2, ie2,
                                   idx + df_z,
                                   gc_re, gc_im, kp_f2, km_f2,
                                   own2 & far_k, EXPANSION)
        if NEAR_X:
            gc_re, gc_im = _mul_imag_coefficient_left(n0r, n0i, v2_re, v2_im,
                                                      EXPANSION)
            if ZM_X:
                gc_re = tl.where(at_x, 0.0, gc_re)
                gc_im = tl.where(at_x, 0.0, gc_im)
            if ZM_Y:
                gc_re = tl.where(at_y, 0.0, gc_re)
                gc_im = tl.where(at_y, 0.0, gc_im)
            _carry_ghost_complex_E(f2, w2, h2, ie2,
                                   idx + dn_x,
                                   gc_re, gc_im, kp_2, km_2,
                                   own2 & near_i, EXPANSION)
        if NEAR_Y:
            gc_re, gc_im = _mul_imag_coefficient_left(n1r, n1i, v2_re, v2_im,
                                                      EXPANSION)
            if ZM_X:
                gc_re = tl.where(at_x, 0.0, gc_re)
                gc_im = tl.where(at_x, 0.0, gc_im)
            if ZM_Y:
                gc_re = tl.where(at_y, 0.0, gc_re)
                gc_im = tl.where(at_y, 0.0, gc_im)
            _carry_ghost_complex_E(f2, w2, h2, ie2,
                                   idx + dn_y,
                                   gc_re, gc_im, kp_2, km_2,
                                   own2 & near_j, EXPANSION)
        if NEAR_X and FAR_Z:
            gc_re, gc_im = _mul_imag_coefficient_left(n0r, n0i, v2_re, v2_im,
                                                      EXPANSION)
            if ZM_X:
                gc_re = tl.where(at_x, 0.0, gc_re)
                gc_im = tl.where(at_x, 0.0, gc_im)
            if ZM_Y:
                gc_re = tl.where(at_y, 0.0, gc_re)
                gc_im = tl.where(at_y, 0.0, gc_im)
            gc_re, gc_im = _mul_imag_coefficient_left(d2r, d2i, gc_re, gc_im,
                                                      EXPANSION)
            _carry_ghost_complex_E(f2, w2, h2, ie2,
                                   idx + df_z + dn_x,
                                   gc_re, gc_im, kp_f2, km_f2,
                                   own2 & far_k & near_i, EXPANSION)
        if NEAR_X and NEAR_Y:
            gc_re, gc_im = _mul_imag_coefficient_left(n0r, n0i, v2_re, v2_im,
                                                      EXPANSION)
            gc_re, gc_im = _mul_imag_coefficient_left(n1r, n1i, gc_re, gc_im,
                                                      EXPANSION)
            if ZM_X:
                gc_re = tl.where(at_x, 0.0, gc_re)
                gc_im = tl.where(at_x, 0.0, gc_im)
            if ZM_Y:
                gc_re = tl.where(at_y, 0.0, gc_re)
                gc_im = tl.where(at_y, 0.0, gc_im)
            _carry_ghost_complex_E(f2, w2, h2, ie2,
                                   idx + dn_x + dn_y,
                                   gc_re, gc_im, kp_2, km_2,
                                   own2 & near_i & near_j, EXPANSION)
        if NEAR_Y and FAR_Z:
            gc_re, gc_im = _mul_imag_coefficient_left(n1r, n1i, v2_re, v2_im,
                                                      EXPANSION)
            if ZM_X:
                gc_re = tl.where(at_x, 0.0, gc_re)
                gc_im = tl.where(at_x, 0.0, gc_im)
            if ZM_Y:
                gc_re = tl.where(at_y, 0.0, gc_re)
                gc_im = tl.where(at_y, 0.0, gc_im)
            gc_re, gc_im = _mul_imag_coefficient_left(d2r, d2i, gc_re, gc_im,
                                                      EXPANSION)
            _carry_ghost_complex_E(f2, w2, h2, ie2,
                                   idx + df_z + dn_y,
                                   gc_re, gc_im, kp_f2, km_f2,
                                   own2 & far_k & near_j, EXPANSION)
        if NEAR_X and (NEAR_Y and FAR_Z):
            gc_re, gc_im = _mul_imag_coefficient_left(n0r, n0i, v2_re, v2_im,
                                                      EXPANSION)
            gc_re, gc_im = _mul_imag_coefficient_left(n1r, n1i, gc_re, gc_im,
                                                      EXPANSION)
            if ZM_X:
                gc_re = tl.where(at_x, 0.0, gc_re)
                gc_im = tl.where(at_x, 0.0, gc_im)
            if ZM_Y:
                gc_re = tl.where(at_y, 0.0, gc_re)
                gc_im = tl.where(at_y, 0.0, gc_im)
            gc_re, gc_im = _mul_imag_coefficient_left(d2r, d2i, gc_re, gc_im,
                                                      EXPANSION)
            _carry_ghost_complex_E(f2, w2, h2, ie2,
                                   idx + df_z + dn_x + dn_y,
                                   gc_re, gc_im, kp_f2, km_f2,
                                   own2 & far_k & near_i & near_j, EXPANSION)
else:  # pragma: no cover - laptop path
    folded_beta_complex_fused_curl_constitutive_D = None  # type: ignore[assignment]


def folded_beta_complex_fused_curl_constitutive_D_kernel() -> Any:
    """The JIT kernel, or a named ImportError on a host without Triton."""
    if folded_beta_complex_fused_curl_constitutive_D is None:
        raise ImportError(
            "the folded complex fused electric D/E kernel needs the optional "
            f"`triton` package (pip install triton). Original error: "
            f"{_TRITON_IMPORT_ERROR}")
    return folded_beta_complex_fused_curl_constitutive_D


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def folded_beta_complex_fused_pair_coverage(fields: Any, pml: Any,
                                       sources: Any = None,
                                       probe: Any = None) -> Coverage:
    """May ONE launch span folded complex ``step_D`` -> both fills -> ``update_E``?

    A conjunction of the THREE halves' own certified predicates plus the seam
    clauses. Nothing is weakened: a configuration any half refuses is refused here
    with that half's reasons, prefixed so a reader can tell which side said it — the
    same construction
    :func:`.folded_complex_fused_magnetic_pair.folded_complex_fused_magnetic_pair_coverage`
    and :func:`.folded_fused_pair.folded_fused_pair_coverage` use.

    THE FILL PREDICATE IS ONE OF THE THREE, and that is what binds the EXTENDED
    pattern set: :func:`.folded_complex.folded_mirror_ghost_fill_complex_coverage`
    carries ``_parity_expansion_reasons``, so a probe artifact that licenses the base
    four but not :data:`.folded_complex.PARITY_PROBE_PATTERN` refuses this product —
    which is correct, because both carries call
    :func:`.special_kz._mul_imag_coefficient_left`.

    ONE ADMISSION ONLY, unlike the magnetic twin. That family admits the off-diagonal
    label as well, because on B->H the two arms are the same kernel; on THIS seam the
    off-diagonal ``update_E`` is a stencil over the D volumes ``step_D`` writes in
    place and belongs to a different product entirely, so the row is refused by name.
    """
    reasons: List[str] = []

    curl = folded_beta_bloch_pml_curl_coverage(
        fields, pml, CURL_SUB_STEP, probe=probe)
    if not curl.covered:
        reasons.extend(f"folded complex beta curl half: {reason}"
                       for reason in curl.reasons)
    constitutive = folded_complex_constitutive_coverage(
        fields, pml, CONSTITUTIVE_SIDE, probe=probe)
    if not constitutive.covered:
        reasons.extend(f"folded complex constitutive half: {reason}"
                       for reason in constitutive.reasons)
    fill = folded_mirror_ghost_fill_complex_coverage(fields, FILL_FAMILY,
                                                     probe=probe)
    if not fill.covered:
        reasons.extend(f"folded complex fill half: {reason}"
                       for reason in fill.reasons)

    # THE STENCIL, RESTATED BY NAME. The eighteen structurally unfusable D->E
    # seam-instances on the Triton board are the OFF-DIAGONAL constitutive reading
    # partner D volumes at four indices (stepping.py:1235-1253) while step_D writes
    # them in place, with no grid-wide barrier inside one launch. The E half refuses
    # that row already; it is named again because THIS is the fact that makes the
    # present cell buildable at all, and because this module's rule is that its
    # coverage may not be read off another module's guard.
    if bool(getattr(fields, "has_offdiagonal_epsilon", False)):
        reasons.append(
            "an off-diagonal chi1inv row makes update_E a STENCIL over the D "
            "volumes step_D writes in place (stepping.py:1235-1253), and no "
            "grid-wide barrier exists inside one launch; that row belongs to the "
            "complex folded off-diagonal product and this weld may not claim it")

    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False,
                        tuple(dict.fromkeys(reasons + ["fields carries no grid"])))

    # THE SOURCE SEAM, and on this cell it is the clause that decides everything. An
    # ELECTRIC source is injected BETWEEN the two halves (driver.py:3294-3299), so a
    # fused pair would consume a pre-injection D. BOTH corpus rows of this cell
    # declare exactly that, so with CARRIES_DEPOSIT_REPAIR False the arm would admit
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

    codes, code_reasons = folded_axis_kinds(
        grid, pml if (pml is not None and getattr(pml, "is_active", False)) else None)
    reasons.extend(code_reasons)

    if codes is not None:
        # THE FAR CARRY'S OWNERSHIP MOVE. `fill_folded_far_ghosts_D` (driver.py:3302)
        # images the top stored slot from stepping._far_reflect_rows' row, and the
        # lane that owns that destination is the one AT the reflect row — so a row
        # outside the allocation is an out-of-range write from a lane that owns
        # neither cell, not a soft error on a whole-plane assignment. The three row
        # checks are symmetry.mirror_ghost_fill_coverage's own (symmetry.py:842-895),
        # restated here for the reason both fused folded families restate them.
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
        # fewer; restated because THIS family writes the destination from a different
        # lane and a missing source plane is an out-of-range write, not a soft error.
        # BOTH mirror codes: the near fill runs on either.
        if len(shape) == 3:
            for axis, code in enumerate(codes):
                if (int(code) in MIRROR_CODES
                        and int(shape[axis]) <= MIRROR_SOURCE_INDEX):
                    reasons.append(
                        f"axis {axis} is folded with {int(shape[axis])} stored "
                        f"cells; the near fill images stored cell "
                        f"{MIRROR_SOURCE_INDEX} and this kernel images it from that "
                        f"cell's own lane")

        # THE WALL CLEAR AND THE FILLS MUST NOT MEET. On this family the clear's rows
        # ARE the near fill's axes — both are the component's shift-0 axes — and
        # `_zero_metal` skips a folded axis (:2237-2239), which is the ONLY thing
        # keeping them apart. CHECKED rather than inferred: an overlap would be a
        # plane of wrong values, and on the D side there is no second reason it could
        # not happen.
        walls = zero_metal_axes(grid)
        for axis, code in enumerate(codes):
            if int(code) in MIRROR_CODES and bool(walls[axis]):
                reasons.append(
                    f"axis {axis} is folded AND zero_metal_axes reports a wall "
                    f"there; stepping._zero_metal skips a folded axis, so the two "
                    f"passes have drifted and the carry's disjointness no longer "
                    f"holds")

        # THE PARITY WORDS THE KERNEL WILL BE PASSED must be readable on every folded
        # axis. `folded_axis_kinds` already refuses a folded axis whose phase is not
        # +-1; this re-derives the WORDS through the builder's own function so a pair
        # that came back (0.0, 0.0) on a folded axis is caught here and not on the
        # device, where it would write an exact zero plane.
        words = parity_coefficient_words(grid)
        for axis, code in enumerate(codes):
            if int(code) not in MIRROR_CODES:
                continue
            if words[axis] == (0.0, 0.0) or words[axis + 3] == (0.0, 0.0):
                reasons.append(
                    f"axis {axis} is folded but its parity coefficient words "
                    f"resolve to near={words[axis]!r} far={words[axis + 3]!r}; the "
                    f"carry would image an exact zero plane")

    # The wall clear is carried inline, so the grid must be able to answer which axes
    # are walled; an unanswerable one compiles to ZM=False and silently skips a plane
    # the array path clears.
    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            reasons.append(
                f"grid does not expose {name}; zero_metal_D cannot be carried inline")

    # THE INVERSE-EPSILON VOLUMES ARE READ BY THE KERNEL at every owned cell AND at
    # every ghost, so `Fields` must be able to hand one over per E component. An
    # absent accessor would be a TypeError inside the builder rather than a refusal,
    # which is the wrong way for an uncovered configuration to fail. The MAGNETIC
    # twin needs no such clause: update_H is `H = B` with mu = 1 baked in and binds
    # no inverse volume at all.
    if getattr(fields, "inverse_epsilon_for", None) is None:
        reasons.append(
            "fields does not expose inverse_epsilon_for; the constitutive half "
            "cannot bind the inv_eps volumes it loads")

    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class FoldedBetaComplexFusedPairPlan:
    """One allocation-free launch for five of the driver's electric call sites.

    The complex volumes are bound as float32 WORD VIEWS at plan time, once, and
    ``n_elem`` stays the COMPLEX cell count — the same split
    :class:`.complex_fields.ComplexPmlCurlPlan` makes, for the same reason. The
    ``inv_eps`` volumes are the exception and are bound RAW: each is float32 with one
    real coefficient per complex cell, indexed at ``+ idx``.
    """

    __slots__ = (
        "shape", "n_elem", "dtdx", "backward", "bc", "zero_metal", "phased",
        "phase_values", "parity", "reflect", "near", "far", "beta_words",
        "has_beta", "expansion",
        "block", "num_warps", "_targets", "_aux", "_sources", "_curl_coefficients",
        "_e_targets", "_e_aux", "_inverse_epsilon", "_e_coefficients",
        "_grid", "_kernel",
    )

    #: The five driver call sites one launch performs. Declared, so a composition can
    #: be inspected rather than inferred from the curl slot's name.
    replaces = REPLACES

    def __init__(self, shape, dtdx: float, bc, zero_metal, phased, phase_values,
                 parity, beta_words, expansion: int, block: int,
                 targets, auxiliaries, sources, curl_coefficients,
                 e_targets, e_aux, inverse_epsilon, e_coefficients,
                 reflect: Any = (None, None, None),
                 kernel: Any = None, num_warps: Optional[int] = 1,
                 has_beta: int = 1) -> None:
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        # float(): the array path consumes the Python scalar at complex64 precision
        # and Triton types a Python float argument as fp32, so the two are the same
        # bits (complex_fields.ComplexPmlCurlPlan's measured note).
        self.dtdx = float(dtdx)
        self.backward = BACKWARD
        self.bc = tuple(int(code) for code in bc)
        self.zero_metal = tuple(bool(value) for value in zero_metal)
        self.phased = tuple(int(flag) for flag in phased)
        self.phase_values = tuple(float(value) for value in phase_values)
        # SIX word pairs, near x/y/z then far x/y/z, host-rounded once.
        self.parity = tuple((float(pair[0]), float(pair[1])) for pair in parity)
        if len(self.parity) != 6:
            raise ValueError(
                f"the parity table is six (re, im) pairs — near x/y/z then far "
                f"x/y/z — got {len(self.parity)}")
        # ((bp_re, bp_im), (bm_re, bm_im)) -- already complex64-rounded words
        # with the signed-zero real parts passed through (float() keeps them);
        # special_kz.beta_curl_coefficients' complex_storage=True output.
        self.beta_words = tuple(tuple(float(word) for word in pair)
                                for pair in beta_words)
        if len(self.beta_words) != 2:
            raise ValueError(
                f"beta_words is two (re, im) pairs -- the +sign and -sign beta "
                f"coefficients -- got {len(self.beta_words)}")
        self.has_beta = int(has_beta)
        # DERIVED FROM `bc`, NEVER TAKEN AS AN ARGUMENT. The kernel branches its curl
        # half on the four codes and its two carries on these booleans; if they could
        # be passed independently they could disagree, and a kernel that masked the
        # top plane on one axis set and imaged the far ghost on another is a plane of
        # wrong values rather than a crash.
        self.near = tuple(code in MIRROR_CODES for code in self.bc)
        self.far = tuple(code == CODE_MIRROR_PERIODIC for code in self.bc)
        # -1 where no far ghost exists. A sentinel rather than 0: no lane reads it
        # (the guard that would is emitted only under FAR_a), and an out-of-range
        # index is loud where imaging row 0 would be plausible and wrong.
        self.reflect = tuple(-1 if value is None else int(value)
                             for value in reflect)
        self.expansion = int(expansion)
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
            if folded and self.phased[axis]:
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
        self.block = int(block)
        self.num_warps = None if num_warps is None else int(num_warps)
        self._targets = tuple(CupyPointer(_word_view(a)) for a in targets)
        self._aux = tuple(CupyPointer(_word_view(a)) for a in auxiliaries)
        self._sources = tuple(CupyPointer(_word_view(a)) for a in sources)
        self._curl_coefficients = tuple(
            CupyPointer(_flat(a)) for a in curl_coefficients)
        self._e_targets = tuple(CupyPointer(_word_view(a)) for a in e_targets)
        self._e_aux = tuple(CupyPointer(_word_view(a)) for a in e_aux)
        if inverse_epsilon is None:
            raise ValueError(
                "the fused pair loads inv_eps on every launch; a placeholder "
                "binding would be read as a coefficient")
        # RAW, not word views: inv_eps stays float32 under complex storage
        # (stepping.py:37-38, fields.py:1203-1204) with one entry per COMPLEX cell.
        self._inverse_epsilon = tuple(CupyPointer(a) for a in inverse_epsilon)
        self._e_coefficients = tuple(
            CupyPointer(_flat(a)) for a in e_coefficients)
        # THE INT32 WORD BOUND. The kernel addresses words as ``2 * idx`` in int32,
        # so the complex cell count must leave room for the doubling — the same
        # halving the complex predicates apply. Refused here as well because the
        # from-arrays route runs no predicate at all.
        if 2 * self.n_elem >= 2 ** 31:
            raise ValueError(
                f"{self.n_elem} complex cells needs {2 * self.n_elem} int32 word "
                f"indices, which overflows the kernel's 2 * idx addressing")
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        # An override for exactly one caller: the gate's mutation legs, which compile
        # deliberately broken copies of this kernel. Dropping it is not a slowdown,
        # it is a DISARMING — every mutation leg would then launch the shipped kernel
        # and report the defect as uncaught.
        self._kernel = kernel

    def run(self, guard: Optional[bool] = None) -> None:
        """Launch the fused pair, in place. Same ``guard`` contract as every plan."""
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        kernel = (self._kernel if self._kernel is not None
                  else folded_beta_complex_fused_curl_constitutive_D_kernel())
        nx, ny, nz = self.shape
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        parity_words = tuple(word for pair in self.parity for word in pair)
        (bp_re, bp_im), (bm_re, bm_im) = self.beta_words
        kernel[self._grid](
            *self._targets, *self._aux, *self._sources, *self._curl_coefficients,
            *self._e_targets, *self._e_aux, *self._inverse_epsilon,
            *self._e_coefficients,
            nx, ny, nz, self.n_elem, self.dtdx,
            *self.phase_values,
            bp_re, bp_im, bm_re, bm_im,
            self.reflect[0], self.reflect[1], self.reflect[2],
            *parity_words,
            BACKWARD=self.backward,
            BCX=self.bc[0], BCY=self.bc[1], BCZ=self.bc[2],
            NEAR_X=self.near[0], NEAR_Y=self.near[1], NEAR_Z=self.near[2],
            FAR_X=self.far[0], FAR_Y=self.far[1], FAR_Z=self.far[2],
            ZM_X=self.zero_metal[0], ZM_Y=self.zero_metal[1],
            ZM_Z=self.zero_metal[2],
            PHX=self.phased[0], PHY=self.phased[1], PHZ=self.phased[2],
            HAS_BETA=self.has_beta,
            EXPANSION=self.expansion,
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )

    def __repr__(self) -> str:
        return (f"FoldedBetaComplexFusedPairPlan(shape={self.shape}, "
                f"bc={self.bc}, zero_metal={self.zero_metal}, "
                f"phased={self.phased}, near={self.near}, far={self.far}, "
                f"reflect={self.reflect}, parity={self.parity}, "
                f"beta_words={self.beta_words!r}, has_beta={self.has_beta}, "
                f"expansion={self.expansion}, block={self.block}, "
                f"num_warps={self.num_warps})")


def plan_folded_beta_complex_fused_pair(
        fields: Any, pml: Any, sources: Any = None,
        block: Optional[int] = None, num_warps: Optional[int] = 1,
        kernel: Any = None, probe: Any = None,
) -> Optional[FoldedBetaComplexFusedPairPlan]:
    """Build the plan from the engine's own objects, or ``None`` when refused.

    ``None`` is the only refusal: a configuration this kernel does not carry must
    fall back to the array path, never raise into a caller that would otherwise have
    stepped correctly.
    """
    if not folded_beta_complex_fused_pair_coverage(fields, pml, sources,
                                              probe=probe).covered:
        return None
    from ..stepping import _boundary_kinds as resolve  # noqa: PLC0415

    record = probe
    if record is None:
        from .folded_complex import load_expansion_probe  # noqa: PLC0415

        record = load_expansion_probe()
    expansion = parity_expansion_from_probe(record)
    beta_expansion = folded_beta_expansion_from_probe(record)
    if expansion is None or beta_expansion != expansion:
        # TWO extended licences answer here -- the fill's parity set and K3b's
        # beta set -- and the kernel binds ONE constexpr, so both must license
        # and agree; the predicate already said so through the two halves' own
        # probe clauses, and this is the builder's own re-ask.
        return None
    codes, _ = folded_axis_kinds(fields.grid, pml)
    if codes is None:  # pragma: no cover - the predicate already refused
        return None

    curl_spec = SUB_STEPS[CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    grid = fields.grid
    kinds = resolve(grid, pml)
    phased, values = _phase_arguments(bloch_phase_table(grid, kinds),
                                      backward=bool(curl_spec["backward"]))
    return FoldedBetaComplexFusedPairPlan(
        grid.shape, grid.dt / grid.dx, codes, zero_metal_axes(grid),
        phased, values, parity_coefficient_words(grid),
        # stepping._special_kz_beta_term's own host rounding (S:770-784):
        # the -1j electric sign rides in the words, not in the kernel.
        beta_curl_coefficients(grid.beta, grid.dt, magnetic=False,
                               complex_storage=True), expansion,
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
        # The far carry's image rows, from the engine's own function rather than
        # recomputed: `n_full - stored + 2` is `stored - 2` at an even full count and
        # `stored - 3` at an odd one (stepping._far_reflect_rows:1661).
        reflect=_far_reflect_rows(grid) or (None, None, None),
        kernel=kernel, num_warps=num_warps,
    )


def plan_folded_beta_complex_fused_pair_from_arrays(
        arrays: Dict[str, Any], curl_flat: Dict[str, Any],
        constitutive_flat: Dict[str, Any], codes, zero_metal,
        phases: Sequence[Optional[complex]], parity, beta_words, dtdx: float,
        expansion: int,
        block: Optional[int] = None, kernel: Any = None,
        num_warps: Optional[int] = 1,
        reflect: Any = (None, None, None),
        has_beta: int = 1) -> FoldedBetaComplexFusedPairPlan:
    """Build from bare device arrays — the gate's route.

    No coverage predicate runs: the caller is a harness that constructed the
    configuration deliberately, including the deliberately wrong ones. ``phases`` is
    the per-axis ``Optional[complex]`` Bloch table (None = unphased) and ``parity``
    the six ``(re, im)`` word pairs, near x/y/z then far x/y/z; ``arrays``
    additionally carries ``inv_eps_Ex``... as float32 volumes. The conjugation for a
    backward sub-step is applied HERE, exactly as the engine route applies it — this
    product is backward-only, so it always is.
    """
    curl_spec = SUB_STEPS[CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    shape = tuple(int(n) for n in arrays[curl_spec["targets"][0]].shape)
    phased, values = _phase_arguments(tuple(phases),
                                      backward=bool(curl_spec["backward"]))
    return FoldedBetaComplexFusedPairPlan(
        shape, dtdx, codes, zero_metal, phased, values, parity, beta_words,
        int(expansion),
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
        reflect=reflect, kernel=kernel, num_warps=num_warps,
        has_beta=has_beta,
    )
