"""The H->D weld on the two REAL beta cells -- ``k_point`` out of the simulated plane.

ONE PRODUCT, TWO VARIANTS, TWO BOARD CELLS, and the two differ in the CURL HALF AND
NOTHING ELSE::

    H_to_D  (real beta run  -> real beta PML)         1 instance
        examples:refl-angular-kz2d.py
    H_to_D  (folded beta run -> folded real beta PML)  1 instance
        tests:TestSpecialKz.test_eigsrc_kz_1_real_imag

Read off ``results/fusion_matrix_triton_2026-09-07_cyl/fusion_matrix.json``
(``aggregate.h_to_d_seam.instances``, ``one_per_row`` true, denominator 597); both
carry ``withdraw_in_seam: false``, ``integrated_electric_sources: 0`` and
``served_by: null``.

=============================================================================
WHY BOTH CELLS' ``update_H`` HALF IS THE CERTIFIED REAL BODY
=============================================================================

Read off the two arms' own predicates rather than off the board's cell labels:

* :func:`.special_kz.beta_run_constitutive_coverage` -- "the constitutive sub-steps
  read nothing beta-dependent (``_apply_constitutive_pml``, stepping.py:2112-2143), so
  admission delegates the ARITHMETIC to the certified ``kernels.constitutive_step``
  unchanged -- no new kernel, no new sub-step";
* :func:`.folded_complex.folded_beta_run_constitutive_coverage` -- "the folded real
  element-wise contract with ONLY the beta clause dropped ... The constitutive
  sub-steps read nothing beta-dependent and nothing fold-dependent beyond the stored
  extent".

Both builders return a ``launch.ConstitutivePlan`` around
:func:`.kernels.constitutive_step`. So :func:`.fused_hd_pair._h_cell` and
:func:`.fused_hd_pair._h_tap` -- that body lifted to an arbitrary cell with exactly
:data:`.fused_hd_pair.CONSTITUTIVE_LIFT_EDITS` -- are ALREADY both variants'
constitutive arithmetic, character for character. They are IMPORTED rather than
copied, for the reason :mod:`.folded_fused_hd_pair` and
:mod:`.cylindrical_real_fused_hd_pair` give: a second copy is a second place for the
lift to drift.

=============================================================================
WHAT BETA CHANGES IN THE CURL, AND WHAT THE FOLD CHANGES ON TOP OF IT
=============================================================================

**Beta.** ``k_point`` out of the plane adds the two real scalars ``beta_plus`` and
``beta_minus`` (:func:`.special_kz.beta_curl_coefficients`, from ``grid.beta`` and
``grid.dt`` with ``magnetic`` off on this side) and a ``HAS_BETA`` constexpr. It adds
NO neighbour: the beta term is a multiple of the field at the program's own cell.

**The fold, on top.** :func:`.folded_complex.folded_beta_pml_curl_step` is the real
beta curl with the three deltas :mod:`.symmetry` names: the ghost branch written the
other way round (``if BC == PERIODIC`` wraps, every other code masks and serves an
exact ``0.0``), the cell-0 ownership mask widened from ``== METALLIC`` to
``!= PERIODIC``, and a SECOND mask block zeroing the LAST plane of a
``MIRROR_PERIODIC`` axis. None of that moves a tap.

**SO THE FOREIGN-TAP SET IS THE PLAIN PRODUCT'S ON BOTH VARIANTS, AND THAT IS
MEASURED RATHER THAN ASSERTED.** :func:`.fused_hd_pair.offset_coordinates` and
:func:`.fused_hd_pair.halo_taps` PARSE each certified body's own offset lines and its
own load lines, and :data:`.fused_hd_pair.HALO_TAPS` declares the six shifted operands
the parse must produce or RAISE. Both bodies pass that parse unchanged: three own-cell
loads, six shifted taps at ``(i-1,j,k)``, ``(i,j-1,k)``, ``(i,j,k-1)``, no halo beyond
them, no mirror-image read.

=============================================================================
THE CODES ARE THE VARIANT'S OWN, AND GETTING THAT WRONG IS SILENT
=============================================================================

The plain variant's triple is ``[1 if kind == "metallic" else 0 for kind in
_boundary_kinds(grid, pml)]`` -- the 0/1 mapping every unfolded plan uses. The FOLDED
variant's is :func:`.folded_complex.folded_axis_kinds`', which resolves FOUR codes.
Handing the folded kernel the 0/1 mapping compiles and launches: ``"mirror"`` maps to
0 = ``PERIODIC``, the ghost WRAPS to the far plane and neither mask block is emitted
at all -- a smooth, converged, entirely wrong answer on every folded axis rather than
a crash. Nothing in the kernel can catch it, because 0 is a valid code.
:func:`plan_beta_real_fused_hd_pair` therefore takes each variant's codes from that
variant's own resolver, and the gate arms the wrong mapping as a mutation that must be
caught on every folded fixture.

=============================================================================
THE SHAPE: SCRATCH OUTPUT AND FOREIGN-CELL RECOMPUTE
=============================================================================

:mod:`.fused_hd_pair`'s, unchanged, and its docstring states the hazard this removes
(``results/triton_fused_offdiag_electric_2026-08-20``: 42 of 60 subject cases
divergent on the mirror-image seam, differing words 4384 -> 0 as ``BLOCK`` went
64 -> 1024). One dispatch:

* the constitutive half writes ``H_new``/``f_w_H_new`` to LAUNCH-LOCAL WRITE-ONLY
  SCRATCH, so ``H``, ``f_w_H`` and ``B`` are pre-launch state for the whole dispatch.
  On this side the newly written ``f_w_H`` IS ``B`` exactly, so an in-place write
  would hand a racing neighbour ``B`` where it needs ``B_prev`` -- the easily missed
  half of the hazard;
* every foreign tap is a RECOMPUTE through :func:`.fused_hd_pair._h_cell`, not a load
  of another program's output;
* ``D``/``fu_D`` step IN PLACE, safe by construction: the curl reads and writes them
  at the program's OWN cell only;
* :class:`.offdiag_scratch_weld.ScratchWeldPairPlan` ROTATES the six ``H``/``f_w_H``
  references after the launch returns.

ONE SUB-LATTICE, ONE COEFFICIENT GROUP, ASSERTED rather than assumed:
``SUB_STEPS['step_D']['suffix']`` is ``''`` and
``coverage.CONSTITUTIVE_SIDES['H']['half_integer']`` is False, so the curl's ``kms_*``
and the constitutive's are the same three volumes and the signature binds them once.
Binding the half-integer set instead compiles, launches, converges, and is a half-cell
error in the absorber profile; the gate arms it.

=============================================================================
WHAT SITS IN THE SEAM, AND WHAT THE FOLD DOES NOT ADD TO IT
=============================================================================

Exactly one statement stands between the two consults and it is the electric
integrated-source withdraw (driver.py:3313-3314), so :data:`CARRIES_DEPOSIT_REPAIR` is
False as a FACT about the driver and :mod:`..withdraw_hoist` is the module that owns
the seam. ON A FOLD that is still true: every B-side fill closes before :3311
(``fill_B`` :3305, ``zero_metal_B`` :3307, ``fill_folded_far_ghosts_B`` :3309) and
every D-side one opens after :3315. So the ``complex_fill_carry`` machinery the folded
D->E and B->H pairs need is NOT needed here and is deliberately not imported -- a
carried fill on this seam would be a pass the driver runs again.

:data:`HOISTS_THE_WITHDRAW` is False in this round because :data:`INSTALLABLE` is
False, which makes ``launch._install_fused_pair``'s withdraw-hoist branch unreachable
for this product. On these two cells that costs nothing: both instances carry no
integrated electric source on the board. The clause is still evaluated on every row --
ignorance is never an empty set, so an undeclared source list is a refusal.

=============================================================================
IT IS NOT INSTALLED, AND THE ARBITRATION IS MEASURED
=============================================================================

:data:`INSTALLABLE` is False; :data:`INSTALLABLE_REASON` carries the label half and
the arbitration half, and the arbitration is a MEASUREMENT per row rather than an
assertion. SERVED ON A FUSION BOARD IS PREDICATE ADMISSION, by that board's own
definition: a released product credits its admitted seam-instances while executing
NOWHERE. This product is in no ``fastpath.RELEASED_FUSED_ARMS`` envelope and
``fastpath`` never plans it. **No timing exists for this shape and none is licensed
here.**

Import contract: importable WITHOUT Triton -- the predicate, the lift checks and the
plan builders (to ``None``) must answer on the laptop that is the merge bar.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import coverage as _coverage
from . import folded_complex as _folded
from . import fused_hd_pair as _plain
from . import special_kz as _kz
from .. import withdraw_hoist as _withdraw_hoist
from .offdiag_scratch_weld import ScratchWeldPairPlan, twin_table

#: The family name, as the board, the battery and the weld record spell it.
FAMILY = "beta_real_fused_hd_pair"

#: The sub-step slot this product STARTS at, in the driver's own order.
SLOT = "update_H"

#: The driver call sites ONE launch performs, in driver order (driver.py:3311, :3315).
REPLACES: Tuple[str, ...] = ("update_H", "step_D")

#: The seam name ``launch.CERTIFIED_FUSED_PAIR_SEAMS`` files this row under.
SEAM: str = _withdraw_hoist.SEAM

#: The constitutive side and the curl sub-step, in driver order.
CONSTITUTIVE_SIDE = "H"
CURL_SUB_STEP = "step_D"

#: The two variants this family emits, in the order the board lists their cells.
VARIANTS: Tuple[str, str] = ("beta", "folded_beta")

#: The PRIMARY arm pair, read off the predicates the ``beta`` variant conjoins.
ARMS: Tuple[str, str] = ("real beta run", "real beta PML")

#: The ADDITIONAL arm pair this family absorbs, in the shape
#: ``launch.CERTIFIED_FUSED_PAIR_EXTRA_ARMS`` takes.
EXTRA_ARMS: Tuple[Tuple[str, str], ...] = (
    ("folded beta run", "folded real beta PML"),)

#: ``variant -> (update_H arm, step_D arm)``. One table, so the two above and the
#: resolver cannot disagree about which cell a variant serves.
VARIANT_ARMS: Dict[str, Tuple[str, str]] = {
    "beta": ARMS,
    "folded_beta": EXTRA_ARMS[0],
}

#: The six volumes one launch ROTATES, in the launch's argument order.
ROTATED: Tuple[str, ...] = _plain.ROTATED

#: The volumes the curl half steps IN PLACE, safe by construction.
IN_PLACE: Tuple[str, ...] = _plain.IN_PLACE

#: The constitutive half's source volumes: B, read const for the whole dispatch.
CONSTITUTIVE_SOURCES: Tuple[str, ...] = _plain.CONSTITUTIVE_SOURCES

#: ``SUB_STEPS['step_D']['backward']``, restated as a declaration a test pins.
BACKWARD = _plain.BACKWARD

#: Elements per program -- the plain product's own restated copy.
DEFAULT_BLOCK = _plain.DEFAULT_BLOCK

#: Does this product bracket its launch with the DEPOSIT repair? NO -- a fact.
CARRIES_DEPOSIT_REPAIR = False

#: Which repair the two slots would install. EMPTY: they install none.
REPAIR_PATHS: Tuple[str, ...] = ()

#: Does this product perform the seam's electric withdraw before its launch? NO.
HOISTS_THE_WITHDRAW = False

#: May the composer install this product? NO, on every configuration.
INSTALLABLE = False

INSTALLABLE_REASON = (
    "THE COMPOSER IS NOT OFFERED THIS PRODUCT AT ALL, and that is the first half of "
    "the reason: routing it through launch._install_certified_fused_products costs a "
    "label, and on this backend a label plan_step can write must also be declared in "
    "the DISPATCHER'S own tables -- outside this package by the one-way rule it "
    "keeps, and outside the round that built this product. "
    "parity/meep_gpu/dispatch_reachability.CERTIFIED_BUT_NOT_INSTALLED names them "
    "from the right side of that wall. The flag is what keeps a later edit that "
    "lands those lines from installing the product before a composition gate has "
    "driven it end to end. "
    "THE SECOND HALF IS THE ARBITRATION, and it is MEASURED PER ROW rather than "
    "asserted: over the driver's step_B - update_H - step_D - update_E slot path "
    "launches are 4 - (installed pairs) and a TWO-SLOT H->D product takes one slot "
    "from EACH neighbour, so installing it is a LOSS where both neighbouring pairs "
    "install, a TIE where exactly one does, and a GAIN where NEITHER does. Both of "
    "this family's cells have a released B->H neighbour (beta_fused_magnetic_pair "
    "and folded_beta_fused_magnetic_pair) and a released D->E neighbour "
    "(beta_fused_electric_pair and folded_beta_fused_electric_pair), so the "
    "PREDICATE join says LOSS on both rows; the gate's arbitration leg re-derives "
    "that through the shipped composer on real fixtures and its lift leg re-derives "
    "it per driven corpus row under `arbitration_over_the_driven_rows`, and those "
    "measurements are the record rather than this sentence. THEY AGREE: both driven "
    "corpus rows measured LOSS -- 2 of 2, no TIE, no GAIN -- in "
    "results/triton_beta_real_fused_hd_pair_2026-09-07/keep/lift/, each a row where "
    "both neighbouring pairs install. Installing this product would RAISE the "
    "per-step launch count on every row it claims, which is why INSTALLABLE is False "
    "and would remain False even with the label wall down. The only span that is "
    "strictly additive on every row is a four-slot "
    "step_B -> update_H -> step_D -> update_E weld, which is not built")

__all__ = [
    "ARMS", "BACKWARD", "CARRIES_DEPOSIT_REPAIR", "CONSTITUTIVE_SIDE",
    "CONSTITUTIVE_SOURCES", "CURL_BODY_ANCHOR", "CURL_FUNCTIONS", "CURL_PATHS",
    "CURL_SUB_STEP", "DECODE_END", "DEFAULT_BLOCK", "EXTRA_ARMS", "FAMILY",
    "FUSED_FUNCTIONS", "HOISTS_THE_WITHDRAW", "INSTALLABLE", "INSTALLABLE_REASON",
    "IN_PLACE", "REPAIR_PATHS", "REPLACES", "ROTATED", "SEAM", "SLOT", "VARIANTS",
    "VARIANT_ARMS",
    "BetaRealFusedHdPairPlan",
    "beta_real_fused_hd_pair_coverage", "certified_curl_tail", "curl_lift_edits",
    "explain_beta_real_fused_hd_pair", "fused_kernel_for", "lifted_curl_tail",
    "plan_beta_real_fused_hd_pair", "plan_beta_real_fused_hd_pair_from_arrays",
    "raw_curl_tail", "resolve_variant",
]


# ---------------------------------------------------------------------------
# The machine-checked lift -- host text, no Triton
# ---------------------------------------------------------------------------

#: Which file each variant's certified curl lives in, spelled once so the lift, the
#: tests and the gate cannot cut from different files.
CURL_PATHS: Dict[str, Path] = {
    "beta": Path(__file__).with_name("special_kz.py"),
    "folded_beta": Path(__file__).with_name("folded_complex.py"),
}

#: The certified curl function each variant lifts, by name.
CURL_FUNCTIONS: Dict[str, str] = {
    "beta": "beta_pml_curl_step",
    "folded_beta": "folded_beta_pml_curl_step",
}

#: Where each certified body ends its index decode. THE ANCHOR CARRIES AN INDENT and
#: the indent is not uniform across this package: both bodies here are declared at
#: module scope inside their module's ``if triton is not None:`` guard but at
#: ``col_offset`` 0, so both anchors carry FOUR spaces, the same as
#: ``kernels.pml_curl_step``'s -- while ``conductivity.conductive_pml_curl_step`` is
#: declared one level deeper and carries eight. Cutting at the wrong indent finds
#: nothing and :func:`.fused_hd_pair._cut` raises, which is the intended failure; the
#: indent is READ from the shipped body by this family's laptop test rather than
#: trusted to a reader's memory of which module nests its kernels.
DECODE_END: Dict[str, str] = {
    "beta": "    i = plane // ny\n",
    "folded_beta": "    i = plane // ny\n",
}

#: Where each fused kernel's lifted CURL body begins in THIS file.
CURL_BODY_ANCHOR: Dict[str, str] = {
    "beta":
        "        # === the certified real beta step_D curl body begins here ===\n",
    "folded_beta":
        "        # === the certified folded real beta step_D curl body begins here "
        "===\n",
}

#: Each variant's fused kernel function name in THIS module.
FUSED_FUNCTIONS: Dict[str, str] = {
    "beta": "fused_constitutive_curl_H_to_D_beta",
    "folded_beta": "fused_constitutive_curl_H_to_D_folded_beta",
}


def _check_variant(variant: str) -> str:
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    return variant


def raw_curl_tail(variant: str) -> str:
    """One variant's certified curl body below its decode, UNEDITED and dedented."""
    _check_variant(variant)
    return _plain._cut(  # noqa: SLF001 - the shared cutter, by design
        _plain._source_of(CURL_FUNCTIONS[variant],  # noqa: SLF001
                          CURL_PATHS[variant]),
        DECODE_END[variant])


def curl_lift_edits(variant: str) -> Tuple[Tuple[str, str], ...]:
    """The nine ``(old, new)`` replacements one variant's curl half takes, GENERATED.

    THE SAME NINE THE PLAIN PRODUCT TAKES, on both bodies, and that is a MEASUREMENT:
    the coordinates and the mask are PARSED out of the certified body by
    :func:`.fused_hd_pair.offset_coordinates` and :func:`.fused_hd_pair.halo_taps`, so
    a beta- or fold-specific change to how either is spelled RAISES here instead of
    quietly redirecting a tap. The ghost BRANCH is untouched -- a tap the certified
    rule masked off still serves an exact ``+0.0``, because ``_h_tap`` closes on
    ``tl.where(valid, value, 0.0)``, which is what ``other=0.0`` delivered.
    """
    tail = raw_curl_tail(variant)
    offsets = _plain.offset_coordinates(tail)
    taps = _plain.halo_taps(tail)
    edits: List[Tuple[str, str]] = list(_plain.OWN_LOAD_EDITS)
    for register, component in _plain.HALO_TAPS:
        _component, offset, mask = taps[register]
        coordinates = ", ".join(offsets[offset])
        edits.append((
            f"{register} = tl.load(g{component} + {offset}, mask={mask}, "
            f"other=0.0)\n",
            f"{register} = _h_tap({component}, {coordinates}, {mask}, "
            f"{_plain.H_CELL_TAIL_ARGS})\n"))
    return tuple(edits)


def certified_curl_tail(variant: str) -> str:
    """One variant's certified curl body with every magnetic read redirected."""
    tail = raw_curl_tail(variant)
    for old, new in curl_lift_edits(variant):
        tail = _plain.needle(tail, old, new)
    for target in range(3):
        if f"g{target} +" in tail:
            raise AssertionError(
                f"the welded {variant} curl half still reads g{target}; in this "
                f"signature that pointer does not exist and every magnetic read must "
                f"be the register or a recompute")
    return tail


def lifted_curl_tail(variant: str) -> str:
    """This module's own fused kernel's curl body, from its marker to the end."""
    _check_variant(variant)
    source = _plain._source_of(  # noqa: SLF001
        FUSED_FUNCTIONS[variant], Path(__file__))
    return _plain._cut(source, CURL_BODY_ANCHOR[variant])  # noqa: SLF001


# ---------------------------------------------------------------------------
# The kernels
# ---------------------------------------------------------------------------
#
# Defined at MODULE scope behind a conditional import, not inside a builder:
# `@triton.jit` resolves a body's names -- including the `tl.constexpr` annotations --
# through the defining module's `__globals__`.

try:  # pragma: no cover - the absent branch is exercised by the absence test
    import triton
    import triton.language as tl
    _TRITON_IMPORT_ERROR: Optional[BaseException] = None
except ImportError as _exc:  # pragma: no cover - the laptop path
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_IMPORT_ERROR = _exc


if triton is not None:  # pragma: no cover - device code, certified by the gate

    #: The FOUR ghost codes. The first two are :mod:`.kernels`' and
    #: :mod:`.special_kz`' own 0/1 and serve the plain beta kernel; all four are
    #: :mod:`.folded_complex`' and serve the folded one. They are read from that
    #: module's integers rather than spelled, and this family's laptop test pins
    #: them equal to both tables -- a divergence in these four integers is a plane
    #: of wrong values rather than a crash.
    PERIODIC = tl.constexpr(_folded.CODE_PERIODIC)
    METALLIC = tl.constexpr(_folded.CODE_METALLIC)
    MIRROR_METALLIC = tl.constexpr(_folded.CODE_MIRROR_METALLIC)
    MIRROR_PERIODIC = tl.constexpr(_folded.CODE_MIRROR_PERIODIC)

    # THE CERTIFIED CONSTITUTIVE, imported as device functions rather than copied.
    # Both cells' `update_H` arm delegates to `kernels.constitutive_step` unchanged
    # (the two predicates quoted in the module docstring say so), which is what makes
    # ONE constitutive half serve two cells.
    from .fused_hd_pair import _h_cell, _h_tap  # noqa: PLC0415

    @triton.jit
    def fused_constitutive_curl_H_to_D_beta(
        ho0, ho1, ho2,                  # SCRATCH out: the stepped Hx, Hy, Hz
        wo0, wo1, wo2,                  # SCRATCH out: the stepped f_w_Hx/y/z
        hi0, hi1, hi2,                  # PRE-LAUNCH Hx, Hy, Hz            (read-only)
        wi0, wi1, wi2,                  # PRE-LAUNCH f_w_Hx/y/z            (read-only)
        b0, b1, b2,                     # Bx, By, Bz                       (read-only)
        f0, f1, f2,                     # curl targets: Dx, Dy, Dz             (in/out)
        u0, u1, u2,                     # curl auxiliaries: fu_Dx/y/z          (in/out)
        kmx, sinvx, kmy, sinvy, kmz, sinvz,   # kms/sinv per axis, INTEGER lattice
        kp0, kp1, kp2,                  # kps on each component's OWN axis, INTEGER
        nx, ny, nz, n_elem, dtdx,
        beta_plus, beta_minus,          # special_kz.beta_curl_coefficients' two
        BACKWARD: tl.constexpr,
        BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
        HAS_BETA: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """``update_H`` + the REAL BETA ``step_D``, one launch, scratch output.

        ``hi``/``wi``/``b`` are const for the whole dispatch and ``ho``/``wo`` are
        write-only scratch, so nothing written by the constitutive half is read by
        this launch; ``f``/``u`` step IN PLACE at the program's OWN cell only.
        ``BCX``/``BCY``/``BCZ`` take the 0/1 codes of an UNFOLDED grid here.
        """
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = idx < n_elem
        nyz = ny * nz
        k = idx % nz
        plane = idx // nz
        j = plane % ny
        i = plane // ny

        # ============ THE WELD: update_H, computed into registers, stored to SCRATCH
        # Nothing stored here is read by this launch. The curl half below takes its
        # OWN cell's magnetic field from these registers and recomputes every foreign
        # tap from PRE-LAUNCH state through the same `_h_cell`; the plan rotates
        # H/f_w_H after the launch returns.
        own0, own1, own2, src0, src1, src2 = _h_cell(
            i, j, k, live, hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
            kp0, kp1, kp2, kmx, kmy, kmz, ny, nz)
        tl.store(ho0 + idx, own0, mask=live)
        tl.store(ho1 + idx, own1, mask=live)
        tl.store(ho2 + idx, own2, mask=live)
        tl.store(wo0 + idx, src0, mask=live)
        tl.store(wo1 + idx, src1, mask=live)
        tl.store(wo2 + idx, src2, mask=live)

        # === the certified real beta step_D curl body begins here ===

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

        a = own0
        b = own1
        c = own2
        a_y = _h_tap(0, i, sj, k, vy, hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                     kp0, kp1, kp2, kmx, kmy, kmz, ny, nz)
        a_z = _h_tap(0, i, j, sk, vz, hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                     kp0, kp1, kp2, kmx, kmy, kmz, ny, nz)
        b_x = _h_tap(1, si, j, k, vx, hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                     kp0, kp1, kp2, kmx, kmy, kmz, ny, nz)
        b_z = _h_tap(1, i, j, sk, vz, hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                     kp0, kp1, kp2, kmx, kmy, kmz, ny, nz)
        c_x = _h_tap(2, si, j, k, vx, hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                     kp0, kp1, kp2, kmx, kmy, kmz, ny, nz)
        c_y = _h_tap(2, i, sj, k, vy, hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                     kp0, kp1, kp2, kmx, kmy, kmz, ny, nz)

        # --- the curl (stepping._curl_from_operands): DO NOT flatten these parens ---
        curl0 = dtdx * ((c_y - c) + (b - b_z))
        curl1 = dtdx * ((a_z - a) + (c - c_x))
        curl2 = dtdx * ((b_x - b) + (a - a_y))

        # --- the beta term (stepping._special_kz_beta_term), CENTER partners only ---
        # AFTER the dtdx curl, BEFORE the ownership mask — the array path's order.
        # No dtdx on the term (analytic derivative, S:733-735); the subtraction IS
        # the array path's `curl + (-(c*g))` (module docstring, grouping choice 2).
        if HAS_BETA:
            curl0 = curl0 - (beta_plus * b)
            curl1 = curl1 - (beta_minus * a)

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
    def fused_constitutive_curl_H_to_D_folded_beta(
        ho0, ho1, ho2,                  # SCRATCH out: the stepped Hx, Hy, Hz
        wo0, wo1, wo2,                  # SCRATCH out: the stepped f_w_Hx/y/z
        hi0, hi1, hi2,                  # PRE-LAUNCH Hx, Hy, Hz            (read-only)
        wi0, wi1, wi2,                  # PRE-LAUNCH f_w_Hx/y/z            (read-only)
        b0, b1, b2,                     # Bx, By, Bz                       (read-only)
        f0, f1, f2,                     # curl targets: Dx, Dy, Dz             (in/out)
        u0, u1, u2,                     # curl auxiliaries: fu_Dx/y/z          (in/out)
        kmx, sinvx, kmy, sinvy, kmz, sinvz,   # kms/sinv per axis, INTEGER lattice
        kp0, kp1, kp2,                  # kps on each component's OWN axis, INTEGER
        nx, ny, nz, n_elem, dtdx,
        beta_plus, beta_minus,          # special_kz.beta_curl_coefficients' two
        BACKWARD: tl.constexpr,
        BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
        HAS_BETA: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """``update_H`` + the FOLDED REAL BETA ``step_D``, one launch, scratch out.

        ``BCX``/``BCY``/``BCZ`` take the FOUR values
        :func:`.folded_complex.folded_axis_kinds` resolves, and that classification is
        the single point of failure in this kernel exactly as it is in
        :mod:`.folded_complex`: the 0/1 mapping instead is a plane of wrong values on
        every folded axis, not a crash. The fold adds NO argument -- the stored extent
        and the four constexpr codes carry it.
        """
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = idx < n_elem
        nyz = ny * nz
        k = idx % nz
        plane = idx // nz
        j = plane % ny
        i = plane // ny

        # ============ THE WELD: update_H, computed into registers, stored to SCRATCH
        # ON A FOLD this is unchanged: `update_H` runs over the whole STORED extent,
        # ghost planes included, and the B those planes hold was written by the
        # driver's near and far fills BEFORE this seam opened -- const input here.
        own0, own1, own2, src0, src1, src2 = _h_cell(
            i, j, k, live, hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
            kp0, kp1, kp2, kmx, kmy, kmz, ny, nz)
        tl.store(ho0 + idx, own0, mask=live)
        tl.store(ho1 + idx, own1, mask=live)
        tl.store(ho2 + idx, own2, mask=live)
        tl.store(wo0 + idx, src0, mask=live)
        tl.store(wo1 + idx, src1, mask=live)
        tl.store(wo2 + idx, src2, mask=live)

        # === the certified folded real beta step_D curl body begins here ===

        # --- DELTA 1: the ghost rule, per axis --------------------------------------
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

        a = own0
        b = own1
        c = own2
        a_y = _h_tap(0, i, sj, k, vy, hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                     kp0, kp1, kp2, kmx, kmy, kmz, ny, nz)
        a_z = _h_tap(0, i, j, sk, vz, hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                     kp0, kp1, kp2, kmx, kmy, kmz, ny, nz)
        b_x = _h_tap(1, si, j, k, vx, hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                     kp0, kp1, kp2, kmx, kmy, kmz, ny, nz)
        b_z = _h_tap(1, i, j, sk, vz, hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                     kp0, kp1, kp2, kmx, kmy, kmz, ny, nz)
        c_x = _h_tap(2, si, j, k, vx, hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                     kp0, kp1, kp2, kmx, kmy, kmz, ny, nz)
        c_y = _h_tap(2, i, sj, k, vy, hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                     kp0, kp1, kp2, kmx, kmy, kmz, ny, nz)

        # --- the curl (stepping._curl_from_operands): DO NOT flatten these parens ---
        curl0 = dtdx * ((c_y - c) + (b - b_z))
        curl1 = dtdx * ((a_z - a) + (c - c_x))
        curl2 = dtdx * ((b_x - b) + (a - a_y))

        # --- the beta term (stepping._special_kz_beta_term), CENTER partners only ---
        # AFTER the dtdx curl, BEFORE BOTH ownership masks. No dtdx on the term
        # (analytic derivative, S:733-735). Real storage is MEEP's implicit-i trick:
        # the SAME sign convention for both sub-steps.
        if HAS_BETA:
            curl0 = curl0 - (beta_plus * b)
            curl1 = curl1 - (beta_minus * a)

        # --- DELTA 2: ownership mask, cell 0 ---------------------------------------
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

        # --- DELTA 3: ownership mask, TOP plane of a folded PERIODIC axis ----------
        last_x, last_y, last_z = i == nx - 1, j == ny - 1, k == nz - 1
        if BACKWARD:
            if BCX == MIRROR_PERIODIC:
                curl0 = tl.where(last_x, 0.0, curl0)
            if BCY == MIRROR_PERIODIC:
                curl1 = tl.where(last_y, 0.0, curl1)
            if BCZ == MIRROR_PERIODIC:
                curl2 = tl.where(last_z, 0.0, curl2)
        else:
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

        # --- split-field recurrence (stepping._apply_pml_update) -------------------
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


def fused_kernel_for(variant: str) -> Any:
    """One variant's shipped kernel object, or a refusal naming the missing import."""
    _check_variant(variant)
    if triton is None:  # pragma: no cover - the laptop path
        raise ImportError(
            "the real beta H->D pair needs Triton to launch; the predicate, the lift "
            f"checks and the plan builder answer without it ({_TRITON_IMPORT_ERROR})")
    return globals()[FUSED_FUNCTIONS[variant]]


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def resolve_variant(fields: Any, pml: Any) -> Tuple[Optional[str], Tuple[str, ...]]:
    """Which variant a run resolves to, or ``(None, reasons)``.

    THE RUN DECIDES, NOT THE CALLER, and the two certified curl predicates are
    DISJOINT by their own fold clauses -- :func:`.special_kz.beta_pml_curl_coverage`
    refuses a mirror plane and
    :func:`.folded_complex.folded_beta_pml_curl_coverage` requires one. Both admitting
    is therefore impossible on a correct pair of predicates, which is exactly why it
    is refused BY NAME here rather than resolved by branch order: were it ever to
    happen, the branch order would silently pick one.
    """
    plain = _kz.beta_pml_curl_coverage(fields, pml, CURL_SUB_STEP)
    folded = _folded.folded_beta_pml_curl_coverage(fields, pml, CURL_SUB_STEP)
    if plain.covered and folded.covered:
        return None, (
            "both the plain and the folded real beta step_D predicates admit this "
            "run: the two are meant to be disjoint on the fold clause, and which "
            "certified curl the array path would have launched is not a fact this "
            "product may decide by branch order",)
    if plain.covered:
        return "beta", ()
    if folded.covered:
        return "folded_beta", ()
    return None, tuple(
        [f"real beta curl half: {reason}" for reason in plain.reasons]
        + [f"folded real beta curl half: {reason}" for reason in folded.reasons])


def beta_real_fused_hd_pair_coverage(
        fields: Any, pml: Any, sources: Any = None,
        variant: Optional[str] = None) -> "_coverage.Coverage":
    """May ONE launch span ``update_H`` -> the electric withdraw -> ``step_D``?

    A conjunction of the two halves' OWN certified predicates plus the seam clauses.
    Nothing is weakened: a configuration either half refuses is refused here with that
    half's reasons, prefixed so a reader can tell which side said it. THE ORDER IS THE
    DRIVER'S -- the constitutive half runs first (driver.py:3311) and is asked first.
    """
    grid = getattr(fields, "grid", None)
    if grid is None:
        return _coverage.Coverage(False, ("fields carries no grid",))

    reasons: List[str] = []
    if variant is None:
        variant, why = resolve_variant(fields, pml)
        if variant is None:
            return _coverage.Coverage(False, tuple(dict.fromkeys(why)))
    else:
        _check_variant(variant)

    if variant == "folded_beta":
        constitutive = _folded.folded_beta_run_constitutive_coverage(
            fields, pml, CONSTITUTIVE_SIDE)
        curl = _folded.folded_beta_pml_curl_coverage(fields, pml, CURL_SUB_STEP)
    else:
        constitutive = _kz.beta_run_constitutive_coverage(
            fields, pml, CONSTITUTIVE_SIDE)
        curl = _kz.beta_pml_curl_coverage(fields, pml, CURL_SUB_STEP)
    if not constitutive.covered:
        reasons.extend(f"constitutive half: {reason}"
                       for reason in constitutive.reasons)
    if not curl.covered:
        reasons.extend(f"curl half: {reason}" for reason in curl.reasons)

    # THE SEAM'S ONE PASS -- the electric integrated-source withdraw
    # (driver.py:3313-3314). IGNORANCE IS NEVER AN EMPTY SET.
    reasons.extend(_withdraw_hoist.seam_withdraw_reasons(
        fields, sources,
        undeclared=(
            "the source set was not declared: this predicate cannot infer from Fields "
            "that no electric withdraw stands between update_H and step_D"),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) has a standing integrated "
            f"electric withdraw, which the driver runs BETWEEN the update_H and "
            f"step_D consults (driver.py:3313-3314); this product declares "
            f"HOISTS_THE_WITHDRAW = False because it declares INSTALLABLE = False, "
            f"so _install_fused_pair's withdraw-hoist branch is unreachable for it "
            f"and nothing would perform the withdraw before the launch"),
        hoists_the_withdraw=HOISTS_THE_WITHDRAW,
        span=REPLACES))

    # THE FOLD IS THE VARIANT SELECTOR HERE, not a refusal: the folded variant
    # REQUIRES a mirror plane and the plain one refuses it, both through the certified
    # curl predicates above. What is checked here is only that the resolved variant
    # and the grid agree, so a caller-forced variant cannot bind the wrong codes.
    mirrored = getattr(grid, "is_mirrored", None)
    folded_axes = [axis for axis in range(3)
                   if callable(mirrored) and bool(mirrored(axis))]
    if variant == "beta" and folded_axes:
        reasons.append(
            f"axes {folded_axes} are folded but the plain real beta variant was "
            f"selected: its 0/1 boundary codes map a mirror to PERIODIC, which wraps "
            f"the ghost and emits neither ownership mask -- a converged, smooth, "
            f"entirely wrong answer rather than a crash")
    if variant == "folded_beta" and not folded_axes:
        reasons.append(
            "the folded real beta variant was selected on a grid with no mirror "
            "plane; folded_axis_kinds resolves no MIRROR code and the folded arm has "
            "no array-path work to specialize")

    # THE ROTATION IS THE PRODUCT'S OWN INVARIANT.
    for name in ROTATED:
        if getattr(fields, name, None) is None:
            reasons.append(
                f"{name} is not allocated; this weld rotates it against a plan-owned "
                f"scratch twin after every launch")
    for name in IN_PLACE + CONSTITUTIVE_SOURCES:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")
    return _coverage.Coverage(not reasons, tuple(dict.fromkeys(reasons)))


def explain_beta_real_fused_hd_pair(fields: Any, pml: Any,
                                    sources: Any = None) -> Dict[str, Any]:
    """The verdict under the name a report reads, WITH the variant it resolved to."""
    variant, why = resolve_variant(fields, pml)
    verdict = beta_real_fused_hd_pair_coverage(fields, pml, sources, variant=variant)
    return {
        "variant": variant,
        "arms": VARIANT_ARMS.get(variant or "", None),
        "covered": bool(verdict.covered),
        "reasons": tuple(verdict.reasons) or tuple(why),
    }


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class BetaRealFusedHdPairPlan(ScratchWeldPairPlan):
    """ONE allocation-free launch for ``update_H`` and one real beta ``step_D``.

    THE SIX ROTATING VOLUMES ARE RESOLVED PER LAUNCH, never cached: the plan owns a
    twin of each, reads the engine's CURRENT attribute immediately before the launch
    to decide which of the pair is live, and moves the engine's references only after
    the launch returns. A pointer captured at plan time would be one rotation stale --
    and stale in a way that still computes.

    ``__init__`` REFUSES ALIASING between anything this launch writes (the six twins,
    ``D`` and ``fu_D``) and anything it reads (``B``, the coefficient vectors), and
    among the written volumes themselves.
    """

    __slots__ = ("variant", "dtdx", "backward", "bc", "beta_plus", "beta_minus",
                 "has_beta", "_b", "_targets", "_aux", "_curl_coeff", "_kps",
                 "_kernel", "_pointer")

    replaces = REPLACES

    def __init__(self, variant: str, shape: Sequence[int], dtdx: float,
                 codes: Sequence[int], beta_plus: float, beta_minus: float,
                 block: int, fields: Any, twins: Dict[str, Any],
                 flux: Sequence[Any], targets: Sequence[Any],
                 auxiliaries: Sequence[Any], curl_coefficients: Sequence[Any],
                 constitutive_coefficients: Sequence[Any], has_beta: int = 1,
                 kernel: Any = None, num_warps: Optional[int] = 1) -> None:
        from .launch import CupyPointer, _flat  # noqa: PLC0415

        super().__init__(FAMILY, fields, twins, ROTATED, shape, block, REPLACES,
                         num_warps=num_warps)
        self.variant = _check_variant(variant)
        self.dtdx = float(dtdx)
        self.backward = BACKWARD
        self.bc = tuple(int(code) for code in codes)
        if len(self.bc) != 3:
            raise ValueError("this plan needs three boundary codes")
        # THE CODE DOMAIN IS PER VARIANT, and it is checked rather than trusted: the
        # plain kernel's ghost branch tests `== METALLIC`, so a 2 or a 3 would take
        # the PERIODIC arm and wrap.
        if self.variant == "beta" and not all(
                code in (_folded.CODE_PERIODIC, _folded.CODE_METALLIC)
                for code in self.bc):
            raise ValueError(
                f"the plain real beta weld takes the 0/1 boundary codes, got "
                f"{self.bc}; a mirror code here takes the PERIODIC arm and wraps")
        if self.variant == "folded_beta" and not any(
                code in (_folded.CODE_MIRROR_METALLIC, _folded.CODE_MIRROR_PERIODIC)
                for code in self.bc):
            raise ValueError(
                f"the folded real beta weld takes folded_axis_kinds' four codes and "
                f"needs at least one mirror axis, got {self.bc}; the 0/1 mapping "
                f"here is a converged, smooth, entirely wrong answer")
        # float(): the array path multiplies a float32 volume by a Python float,
        # which NumPy/CuPy cast to float32 before the multiply. Triton types a Python
        # float argument as fp32, so the two scalars are the same bits.
        self.beta_plus = float(beta_plus)
        self.beta_minus = float(beta_minus)
        self.has_beta = int(has_beta)
        self._pointer = CupyPointer
        self._b = tuple(CupyPointer(a) for a in flux)
        self._targets = tuple(CupyPointer(a) for a in targets)
        self._aux = tuple(CupyPointer(a) for a in auxiliaries)
        self._curl_coeff = tuple(CupyPointer(_flat(a)) for a in curl_coefficients)
        self._kps = tuple(CupyPointer(_flat(a)) for a in constitutive_coefficients)
        if len(self._curl_coeff) != 6 or len(self._kps) != 3:
            raise ValueError(
                "this plan binds six curl coefficient vectors (kms/sinv per axis) and "
                "three constitutive kps; the constitutive kms are the curl's own "
                "three, shared because both halves sit on the integer sub-lattice")
        self._check_aliases(twins, flux, targets, auxiliaries,
                            curl_coefficients, constitutive_coefficients)
        self._kernel = kernel

    def _check_aliases(self, twins, flux, targets, auxiliaries,
                       curl_coefficients, constitutive_coefficients) -> None:
        """No written volume may share an allocation with a read one, or another."""
        def address(array: Any) -> Optional[int]:
            data = getattr(array, "data", None)
            pointer = getattr(data, "ptr", None)
            if pointer is not None:
                return int(pointer)
            interface = getattr(array, "__array_interface__", None)
            if isinstance(interface, dict):
                return int(interface["data"][0])
            return None

        outputs: Dict[int, str] = {}
        named = [(name, twins[name]) for name in ROTATED]
        named += [(IN_PLACE[index], array) for index, array in enumerate(targets)]
        named += [(IN_PLACE[3 + index], array)
                  for index, array in enumerate(auxiliaries)]
        for label, array in named:
            key = address(array)
            if key is None:
                continue
            if key in outputs:
                raise ValueError(
                    f"outputs {outputs[key]} and {label} are the same allocation; "
                    f"one launch would write both")
            outputs[key] = label
        inputs: List[Tuple[str, Any]] = [
            (CONSTITUTIVE_SOURCES[index], array)
            for index, array in enumerate(flux)]
        inputs += [(f"curl_coefficient{index}", array)
                   for index, array in enumerate(curl_coefficients)]
        inputs += [(f"kps{index}", array)
                   for index, array in enumerate(constitutive_coefficients)]
        for label, array in inputs:
            key = address(array)
            if key is not None and key in outputs:
                raise ValueError(
                    f"input {label} aliases output {outputs[key]}: the launch would "
                    f"read a volume it is writing")

    def _launch(self, writes: Sequence[Any], reads: Sequence[Any],
                guard: Optional[bool]) -> None:
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        kernel = (self._kernel if self._kernel is not None
                  else fused_kernel_for(self.variant))
        nx, ny, nz = self.shape
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        scratch = [self._pointer(array) for array in writes]
        prior = [self._pointer(array) for array in reads]
        kernel[self._grid](
            *scratch, *prior, *self._b, *self._targets, *self._aux,
            *self._curl_coeff, *self._kps,
            nx, ny, nz, self.n_elem, self.dtdx,
            self.beta_plus, self.beta_minus,
            BACKWARD=self.backward,
            BCX=self.bc[0], BCY=self.bc[1], BCZ=self.bc[2],
            HAS_BETA=self.has_beta,
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"BetaRealFusedHdPairPlan({self.variant}, shape={self.shape}, "
                f"bc={self.bc}, block={self.block}, num_warps={self.num_warps})")


def _sub_lattice_suffixes() -> Tuple[str, str]:
    """``(curl suffix, constitutive suffix)``, READ from the shipped tables."""
    from .launch import SUB_STEPS  # noqa: PLC0415

    curl = SUB_STEPS[CURL_SUB_STEP]["suffix"]
    constitutive = ("_h" if _coverage.CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
                    ["half_integer"] else "")
    if curl != constitutive:
        raise AssertionError(
            f"step_D reads the {curl or 'integer'!r} PML sub-lattice and update_H "
            f"the {constitutive or 'integer'!r} one; this weld binds ONE kms group "
            f"for both halves and that is only correct while they agree")
    if int(SUB_STEPS[CURL_SUB_STEP]["backward"]) != BACKWARD:
        raise AssertionError(
            f"launch.SUB_STEPS says step_D differences "
            f"{SUB_STEPS[CURL_SUB_STEP]['backward']}, and this product declares "
            f"BACKWARD = {BACKWARD}")
    return curl, constitutive


def plan_beta_real_fused_hd_pair(
        fields: Any, pml: Any, sources: Any = None, block: Optional[int] = None,
        num_warps: Optional[int] = 1, kernel: Any = None
) -> Optional[BetaRealFusedHdPairPlan]:
    """Build the plan from the engine's own objects, or ``None`` when refused.

    ``None`` is the only refusal, for the reason :func:`.launch.plan_pml_curl` gives.
    ``kernel`` is the mutation seam: the gate compiles a deliberately broken copy of
    the shipped kernel and hands it here, and dropping the argument is not a slowdown
    but a silent DISARMING.

    THE BOUNDARY CODES COME FROM THE VARIANT'S OWN RESOLVER. The folded variant takes
    :func:`.folded_complex.folded_axis_kinds`' four; the plain one takes the 0/1
    mapping. Crossing them compiles and is wrong on every folded axis.
    """
    variant, _why = resolve_variant(fields, pml)
    if variant is None:
        return None
    if not beta_real_fused_hd_pair_coverage(
            fields, pml, sources, variant=variant).covered:
        return None
    suffix, constitutive_suffix = _sub_lattice_suffixes()
    grid = fields.grid
    if variant == "folded_beta":
        codes, _reasons = _folded.folded_axis_kinds(grid, pml)
        if codes is None:  # pragma: no cover - the predicate already refused
            return None
    else:
        kinds = _coverage._boundary_kinds(grid, pml)  # noqa: SLF001
        codes = [1 if kind == "metallic" else 0 for kind in kinds]
    plus, minus = _kz.beta_curl_coefficients(
        grid.beta, grid.dt, magnetic=(CURL_SUB_STEP == "step_B"),
        complex_storage=False)
    return BetaRealFusedHdPairPlan(
        variant, grid.shape, grid.dt / grid.dx, codes, plus, minus,
        DEFAULT_BLOCK if block is None else block,
        fields, twin_table(fields, ROTATED),
        [getattr(fields, name) for name in CONSTITUTIVE_SOURCES],
        [getattr(fields, name) for name in IN_PLACE[:3]],
        [getattr(fields, name) for name in IN_PLACE[3:]],
        [getattr(pml, f"{stem}_{axis}{suffix}")
         for axis in "xyz" for stem in ("kms", "sinv")],
        [getattr(pml, f"kps_{axis}{constitutive_suffix}") for axis in "xyz"],
        kernel=kernel, num_warps=num_warps,
    )


def plan_beta_real_fused_hd_pair_from_arrays(
        variant: str, arrays: Dict[str, Any], flat: Dict[str, Any], dtdx: float,
        codes: Sequence[int], beta_plus: float, beta_minus: float, fields: Any,
        block: Optional[int] = None, kernel: Any = None,
        num_warps: Optional[int] = 1, has_beta: int = 1
) -> BetaRealFusedHdPairPlan:
    """Build from bare device arrays -- the gate's route.

    No coverage predicate runs: the caller is a harness that constructed the
    configuration deliberately, including the deliberately wrong ones. ``fields`` is
    the object whose attributes the ROTATION swaps; ``arrays`` supplies the twins
    under ``scratch_Hx`` ...
    """
    _check_variant(variant)
    shape = tuple(int(n) for n in arrays[IN_PLACE[0]].shape)
    twins = {name: arrays["scratch_" + name] for name in ROTATED}
    return BetaRealFusedHdPairPlan(
        variant, shape, dtdx, codes, beta_plus, beta_minus,
        DEFAULT_BLOCK if block is None else block, fields, twins,
        [arrays[name] for name in CONSTITUTIVE_SOURCES],
        [arrays[name] for name in IN_PLACE[:3]],
        [arrays[name] for name in IN_PLACE[3:]],
        [flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
        [flat[f"kps_{axis}"] for axis in "xyz"],
        has_beta=has_beta, kernel=kernel, num_warps=num_warps,
    )
