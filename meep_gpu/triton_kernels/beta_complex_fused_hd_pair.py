"""The H->D weld on the two COMPLEX beta cells -- ``k_point`` out of plane, complex.

ONE PRODUCT, TWO VARIANTS, TWO BOARD CELLS, FOUR SEAM-INSTANCES::

    H_to_D  (folded complex  -> folded complex beta PML)  3
        tests:TestEigCoeffs.test_binary_grating_special_kz_0_13_2
        tests:TestEigCoeffs.test_binary_grating_special_kz_1_17_7
        tests:TestSpecialKz.test_eigsrc_kz_0_complex
    H_to_D  (complex beta run -> complex beta PML)         1
        tests:TestSpecialKz.test_special_kz

Read off ``results/fusion_matrix_triton_2026-09-07_cyl/fusion_matrix.json``
(``aggregate.h_to_d_seam.instances``, ``one_per_row`` true, denominator 597); all four
carry ``withdraw_in_seam: false``, ``integrated_electric_sources: 0`` and
``served_by: null``.

=============================================================================
THE ASYMMETRY IN THE TWO CELL LABELS IS REAL AND IT IS THE POINT
=============================================================================

The folded cell's ``update_H`` arm is ``folded complex`` -- NOT "folded complex beta".
The unfolded cell's is ``complex beta run``. That is not a board typo, and this module
does not paper over it: it is what the two shipped constitutive predicates are.

* On a FOLD the constitutive arm is
  :func:`.folded_complex.folded_complex_constitutive_coverage`, and the same pairing
  is already in ``launch.CERTIFIED_FUSED_PAIR_ARMS`` for this cell's B->H and D->E
  siblings: ``folded_beta_complex_fused_magnetic_pair: ("folded complex beta PML",
  "folded complex")`` and ``folded_beta_complex_fused_pair`` the same. That predicate
  carries no beta clause because the constitutive sub-steps read nothing
  beta-dependent, and the fold clause is what makes it a distinct arm from the
  unfolded one.
* UNFOLDED the arm is :func:`.special_kz.beta_run_complex_constitutive_coverage` --
  the complex element-wise contract with the beta clause INVERTED -- and the table
  carries ``complex_beta_fused_magnetic_pair: ("complex beta PML", "complex beta
  run")``.

Both builders return a :class:`.complex_fields.ComplexConstitutivePlan` around
:func:`.complex_fields.bloch_constitutive_step`. So the CONSTITUTIVE HALF IS ONE BODY
across both cells, and :func:`.complex_fused_hd_pair._h_cell_complex` /
:func:`.complex_fused_hd_pair._h_tap_complex` -- that body lifted to an arbitrary cell
with exactly :data:`.complex_fused_hd_pair.CONSTITUTIVE_LIFT_EDITS` -- are already
both variants' constitutive arithmetic, character for character. They are IMPORTED
rather than copied.

The two cells therefore differ in the CURL HALF AND NOTHING ELSE, which is what
licenses one family with two emitted kernels:
:func:`.folded_complex.folded_beta_bloch_pml_curl_step` and
:func:`.special_kz.beta_bloch_pml_curl_step`. Their SIGNATURES are identical -- the
same chunk record, the same six connection words, the same two beta words, the same
``CNX``/``CNY``/``CNZ`` and ``HAS_BETA`` constexprs -- so one plan class binds both
and the variant selects the kernel and the CODE DOMAIN.

=============================================================================
WHAT BETA ADDS: MEEP'S SEPARATE PASS, A QUARTER TURN BY ONE REAL WORD
=============================================================================

Both certified curls run MEEP's special-kz beta pass after every component's curl
form, through the additive ladder (``forms.fold_ladder``) keyed on the chunk record.
Under complex storage the increment is a quarter turn of the partner's parts by ONE
real word: :func:`.special_kz.beta_curl_coefficients` rounds ``sign * 2*pi*beta*dt``
once to float32 (negated on the D side, whose quarter turn is by ``-i``), and the
kernel forms ``(c * im, (c * re) * -1.0)`` inline -- no complex product. This weld
touches none of that: it is inside the lifted curl text, in the emitter's own order.

Beta adds NO neighbour: the term is a multiple of the field at the program's own cell.
The fold, on top, adds the three deltas :mod:`.symmetry` names (ghost branch inverted,
cell-0 mask widened to ``!= PERIODIC``, and the ``MIRROR_PERIODIC`` top-plane mask),
none of which moves a tap.

**SO THE FOREIGN-TAP SET IS THE UNFOLDED COMPLEX PRODUCT'S ON BOTH VARIANTS, AND THAT
IS MEASURED.** :func:`.complex_fused_hd_pair.offset_coordinates` and
:func:`.complex_fused_hd_pair.halo_taps` PARSE each certified body's own offset lines
and its own load lines; the second additionally requires the two word planes of every
shifted operand to agree on component, cell AND guard, and
:data:`.complex_fused_hd_pair.HALO_TAPS` declares the six operands the parse must
produce or RAISE. Both bodies pass unchanged: six own-cell word loads, six shifted
word PAIRS at ``(i-1,j,k)``, ``(i,j-1,k)``, ``(i,j,k-1)``, and no halo beyond them.

=============================================================================
THE CODES ARE PER VARIANT, AND CROSSING THEM IS SILENT
=============================================================================

The unfolded variant's triple is the 0/1 ``metallic``/``periodic`` mapping; the folded
variant's is :func:`.folded_complex.folded_axis_kinds`' four codes. Handing the folded
kernel the 0/1 mapping compiles and launches -- ``"mirror"`` maps to 0 = ``PERIODIC``,
the ghost WRAPS and neither ownership mask is emitted -- a smooth, converged, entirely
wrong answer on every folded axis rather than a crash. The plan CHECKS the domain it
was handed against the variant, and the gate arms the wrong mapping as a mutation.

=============================================================================
THE SHAPE, THE SEAM, AND WHAT IS NOT INSTALLED
=============================================================================

The weld shape is :mod:`.fused_hd_pair`'s, unchanged: the constitutive half writes
``H_new``/``f_w_H_new`` to LAUNCH-LOCAL WRITE-ONLY SCRATCH so ``H``, ``f_w_H`` and
``B`` are pre-launch state for the whole dispatch; every foreign tap is a RECOMPUTE
through :func:`.complex_fused_hd_pair._h_cell_complex`; ``D``/``fu_D`` step IN PLACE
at the program's own cell only; and
:class:`.offdiag_scratch_weld.ScratchWeldPairPlan` ROTATES the six ``H``/``f_w_H``
references after the launch returns. On this side the newly written ``f_w_H`` IS ``B``
exactly, so an in-place write would hand a racing neighbour ``B`` where it needs
``B_prev`` -- the easily missed half of the hazard.

ONE SUB-LATTICE, ONE COEFFICIENT GROUP, ASSERTED rather than assumed in
:func:`plan_beta_complex_fused_hd_pair`.

Exactly one statement stands between the two consults -- the electric integrated-
source withdraw (driver.py:3313-3314) -- so :data:`CARRIES_DEPOSIT_REPAIR` is False as
a FACT and :mod:`..withdraw_hoist` owns the seam. On a FOLD that is still true: every
B-side fill closes before :3311 and every D-side one opens after :3315, so the
``complex_fill_carry`` machinery the folded D->E and B->H pairs need is NOT needed
here and is deliberately not imported.

:data:`INSTALLABLE` is False. SERVED ON A FUSION BOARD IS PREDICATE ADMISSION, by that
board's own definition: a released product credits its admitted seam-instances while
executing NOWHERE. **No timing exists for this shape and none is licensed here.**

Import contract: importable WITHOUT Triton -- the predicate, the lift checks and the
plan builders (to ``None``) must answer on the laptop that is the merge bar.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import complex_fields as _cf
from . import complex_fused_hd_pair as _complex
from . import coverage as _coverage
from . import folded_complex as _folded
from . import fused_hd_pair as _plain
from . import special_kz as _kz
from .. import withdraw_hoist as _withdraw_hoist
from .complex_fields import _word_view
from .offdiag_scratch_weld import ScratchWeldPairPlan, twin_table

#: The family name, as the board, the battery and the weld record spell it.
FAMILY = "beta_complex_fused_hd_pair"

#: The sub-step slot this product STARTS at, in the driver's own order.
SLOT = "update_H"

#: The driver call sites ONE launch performs, in driver order (driver.py:3311, :3315).
REPLACES: Tuple[str, ...] = ("update_H", "step_D")

#: The seam name ``launch.CERTIFIED_FUSED_PAIR_SEAMS`` files this row under.
SEAM: str = _withdraw_hoist.SEAM

#: The constitutive side and the curl sub-step, in driver order.
CONSTITUTIVE_SIDE = "H"
CURL_SUB_STEP = "step_D"

#: The two variants this family emits, in the order the board lists their cells
#: (the folded cell is three instances and the unfolded one).
VARIANTS: Tuple[str, str] = ("folded_beta_complex", "beta_complex")

#: The PRIMARY arm pair, read off the two predicates the folded variant conjoins.
#: NOTE the ``update_H`` half is ``folded complex`` and not "folded complex beta" --
#: see the module docstring; the same pairing is already in
#: ``launch.CERTIFIED_FUSED_PAIR_ARMS`` for this cell's B->H and D->E siblings.
ARMS: Tuple[str, str] = ("folded complex", "folded complex beta PML")

#: The ADDITIONAL arm pair this family absorbs, in the shape
#: ``launch.CERTIFIED_FUSED_PAIR_EXTRA_ARMS`` takes.
EXTRA_ARMS: Tuple[Tuple[str, str], ...] = (
    ("complex beta run", "complex beta PML"),)

#: ``variant -> (update_H arm, step_D arm)``.
VARIANT_ARMS: Dict[str, Tuple[str, str]] = {
    "folded_beta_complex": ARMS,
    "beta_complex": EXTRA_ARMS[0],
}

#: The six volumes one launch ROTATES, in the launch's argument order.
ROTATED: Tuple[str, ...] = _plain.ROTATED

#: The volumes the curl half steps IN PLACE, safe by construction.
IN_PLACE: Tuple[str, ...] = _plain.IN_PLACE

#: The constitutive half's source volumes: B, read const for the whole dispatch.
CONSTITUTIVE_SOURCES: Tuple[str, ...] = _plain.CONSTITUTIVE_SOURCES

#: ``SUB_STEPS['step_D']['backward']``, restated as a declaration a test pins.
BACKWARD = _complex.BACKWARD

#: Elements per program -- the complex product's own restated copy, PINNED by a test.
DEFAULT_BLOCK = _complex.DEFAULT_BLOCK

#: One launch per run, and the plan counts its own.
LAUNCHES_PER_RUN = 1

#: Does this product bracket its launch with the DEPOSIT repair? NO -- a fact.
CARRIES_DEPOSIT_REPAIR = False

#: Which repair the two slots would install. EMPTY: they install none.
REPAIR_PATHS: Tuple[str, ...] = ()

#: Does this product perform the seam's electric withdraw before its launch? NO in
#: this round. On these two cells that costs nothing -- all four instances carry
#: ``integrated_electric_sources: 0`` -- but the clause is EVALUATED, never assumed.
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
    "from the right side of that wall. "
    "THE SECOND HALF IS THE ARBITRATION, and it is MEASURED PER ROW rather than "
    "asserted: over the driver's step_B - update_H - step_D - update_E slot path "
    "launches are 4 - (installed pairs) and a TWO-SLOT H->D product takes one slot "
    "from EACH neighbour, so installing it is a LOSS where both neighbouring pairs "
    "install, a TIE where exactly one does, and a GAIN where NEITHER does. Which of "
    "the three each of this family's four corpus rows is comes from that row's OWN "
    "composer slot table and is recorded, row by row, in the lift leg of "
    "parity/meep_gpu/gate_triton_beta_complex_fused_hd_pair.py under "
    "`arbitration_over_the_driven_rows`. WHAT IT MEASURED, stated here rather than "
    "left to be discovered by opening an artifact: ALL FOUR driven corpus rows are a "
    "LOSS -- 4 of 4, no TIE, no GAIN -- in "
    "results/triton_beta_complex_fused_hd_pair_2026-09-07/keep/lift/, every one of "
    "them a row where BOTH neighbouring pairs install. Installing this product would "
    "RAISE the per-step launch count on every row it claims, which is why INSTALLABLE "
    "is False and would remain False even with the label wall down. "
    "A THIRD FACT IS SPECIFIC TO THIS ARM and is measured with the "
    "others rather than assumed: the complex arms are certified under the 'keep' "
    "float32 subnormal policy only, so under 'flush' the composer selects nothing "
    "complex on these rows at all and the arbitration question does not arise. The "
    "only span that is strictly additive everywhere is a four-slot "
    "step_B -> update_H -> step_D -> update_E weld, which is not built")

#: The helpers these kernels may reach, and NO OTHER: the Bloch connection alone, as
#: on the plain complex product. The beta pass is a quarter turn of the partner's
#: parts by one real word, spelled inline by the certified bodies, so no complex
#: product remains. A laptop test scans the shipped kernel text for
#: ``_mul_``/``_rotate_``/``_div_``/``_connect`` call sites and fails on anything
#: outside this tuple.
MULTIPLY_HELPERS: Tuple[str, ...] = _complex.MULTIPLY_HELPERS

__all__ = [
    "ARMS", "BACKWARD", "CARRIES_DEPOSIT_REPAIR",
    "CONSTITUTIVE_SIDE", "CONSTITUTIVE_SOURCES", "CURL_BODY_ANCHOR", "CURL_FUNCTIONS",
    "CURL_PATHS", "CURL_SUB_STEP", "DECODE_END", "DEFAULT_BLOCK", "EXTRA_ARMS",
    "FAMILY", "FUSED_FUNCTIONS", "HOISTS_THE_WITHDRAW", "INSTALLABLE",
    "INSTALLABLE_REASON", "IN_PLACE", "LAUNCHES_PER_RUN", "MULTIPLY_HELPERS",
    "REPAIR_PATHS", "REPLACES", "ROTATED", "SEAM", "SLOT", "VARIANTS",
    "VARIANT_ARMS",
    "BetaComplexFusedHdPairPlan",
    "beta_complex_fused_hd_pair_coverage", "certified_curl_tail", "curl_lift_edits",
    "explain_beta_complex_fused_hd_pair", "fused_kernel_for", "lifted_curl_tail",
    "plan_beta_complex_fused_hd_pair",
    "plan_beta_complex_fused_hd_pair_from_arrays", "raw_curl_tail",
    "resolve_variant",
]


# ---------------------------------------------------------------------------
# The machine-checked lift -- host text, no Triton
# ---------------------------------------------------------------------------

#: Which file each variant's certified curl lives in, spelled once so the lift, the
#: tests and the gate cannot cut from different files.
CURL_PATHS: Dict[str, Path] = {
    "folded_beta_complex": Path(__file__).with_name("folded_complex.py"),
    "beta_complex": Path(__file__).with_name("special_kz.py"),
}

#: The certified curl function each variant lifts, by name.
CURL_FUNCTIONS: Dict[str, str] = {
    "folded_beta_complex": "folded_beta_bloch_pml_curl_step",
    "beta_complex": "beta_bloch_pml_curl_step",
}

#: Where each certified body ends its index decode. Both are declared at module scope
#: (``col_offset`` 0) inside their module's Triton guard, so both anchors carry FOUR
#: spaces. Cutting at the wrong indent finds nothing and
#: :func:`.fused_hd_pair._cut` raises; the laptop test READS the indent off the
#: shipped body rather than trusting it.
DECODE_END: Dict[str, str] = {
    "folded_beta_complex": _complex.DECODE_END,
    "beta_complex": _complex.DECODE_END,
}

#: Where each fused kernel's lifted CURL body begins in THIS file.
CURL_BODY_ANCHOR: Dict[str, str] = {
    "folded_beta_complex":
        "        # === the certified folded complex beta step_D curl body begins "
        "here ===\n",
    "beta_complex":
        "        # === the certified complex beta step_D curl body begins here ===\n",
}

#: Each variant's fused kernel function name in THIS module.
FUSED_FUNCTIONS: Dict[str, str] = {
    "folded_beta_complex":
        "fused_folded_beta_complex_constitutive_curl_H_to_D",
    "beta_complex": "fused_beta_complex_constitutive_curl_H_to_D",
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
    """The twelve ``(old, new)`` replacements one variant's curl half takes.

    THE SAME TWELVE THE UNFOLDED COMPLEX PRODUCT TAKES, on both bodies, and that is a
    MEASUREMENT: the component, the coordinates and the mask all come from
    :func:`.complex_fused_hd_pair.offset_coordinates` and
    :func:`.complex_fused_hd_pair.halo_taps` parsing the certified body itself. The
    BLOCH CONNECTION is not touched -- the emitter applies it to the UNSHIFTED
    partners' plane 0 after the loads, which is why
    :func:`.complex_fused_hd_pair._h_tap_complex` carries none -- and neither is the
    BETA pass, a quarter turn of the own cell's partner after the curl forms.
    """
    tail = raw_curl_tail(variant)
    offsets = _complex.offset_coordinates(tail)
    taps = _complex.halo_taps(tail)
    edits: List[Tuple[str, str]] = list(_complex.OWN_LOAD_EDITS)
    for stem, component in _complex.HALO_TAPS:
        _component, offset, mask = taps[stem]["re"]
        coordinates = ", ".join(offsets[offset])
        edits.append((
            f"{stem}_re = tl.load(g{component} + 2 * {offset}, mask={mask}, "
            f"other=0.0)\n"
            f"{stem}_im = tl.load(g{component} + 2 * {offset} + 1, mask={mask}, "
            f"other=0.0)\n",
            f"{stem}_re, {stem}_im = _h_tap_complex({component}, {coordinates}, "
            f"{mask},\n"
            f"                     {_complex.H_CELL_TAIL_ARGS})\n"))
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
                f"be the register pair or a recompute")
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

try:  # pragma: no cover - the absent branch is exercised by the absence test
    import triton
    import triton.language as tl
    _TRITON_IMPORT_ERROR: Optional[BaseException] = None
except ImportError as _exc:  # pragma: no cover - the laptop path
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_IMPORT_ERROR = _exc


if triton is not None:  # pragma: no cover - device code, certified by the gate

    # The names the two certified complex beta curls read -- the Bloch connection, the
    # shared forms with the fold ladder, and the record's word offsets -- the very
    # objects special_kz.beta_bloch_pml_curl_step and
    # folded_complex.folded_beta_bloch_pml_curl_step call, so each lift is the same
    # text resolving to the same functions.
    from .complex_fields import _connect  # noqa: PLC0415
    from .forms import (axis_present, curl_form, curl_stencil,  # noqa: PLC0415
                        fold_ladder, record_in)
    from .kernels import RECORD_HALF_INTEGER, RECORD_PRESENCE  # noqa: PLC0415

    #: The FOUR ghost codes, read from :mod:`.folded_complex`' own integers and pinned
    #: equal to them (and to :mod:`.special_kz`' 0/1) by this family's laptop test.
    PERIODIC = tl.constexpr(_folded.CODE_PERIODIC)
    METALLIC = tl.constexpr(_folded.CODE_METALLIC)
    MIRROR_METALLIC = tl.constexpr(_folded.CODE_MIRROR_METALLIC)
    MIRROR_PERIODIC = tl.constexpr(_folded.CODE_MIRROR_PERIODIC)

    # THE CERTIFIED COMPLEX CONSTITUTIVE, imported as device functions rather than
    # copied. Both cells' `update_H` arm builds the certified
    # `ComplexConstitutivePlan` around `complex_fields.bloch_constitutive_step`, which
    # is what makes ONE constitutive half serve two cells.
    from .complex_fused_hd_pair import _h_cell_complex, _h_tap_complex  # noqa: PLC0415

    @triton.jit(do_not_specialize=["start_h"])
    def fused_folded_beta_complex_constitutive_curl_H_to_D(
        ho0, ho1, ho2,                  # SCRATCH out: stepped Hx,Hy,Hz  (c8 as words)
        wo0, wo1, wo2,                  # SCRATCH out: stepped f_w_H*    (c8 as words)
        hi0, hi1, hi2,                  # PRE-LAUNCH Hx, Hy, Hz            (read-only)
        wi0, wi1, wi2,                  # PRE-LAUNCH f_w_Hx/y/z            (read-only)
        b0, b1, b2,                     # Bx, By, Bz                       (read-only)
        f0, f1, f2,                     # curl targets: Dx, Dy, Dz             (in/out)
        u0, u1, u2,                     # curl auxiliaries: fu_Dx/y/z          (in/out)
        kmx, sinvx, kmy, sinvy, kmz, sinvz,   # kms/sinv per axis, INTEGER lattice
        kp0, kp1, kp2,                  # kps on each component's OWN axis, INTEGER
        cb,                             # chunk record: both halves' forms and presence
        nx, ny, nz, n_elem, dtdx, start_h,   # n_elem = COMPLEX cells; MEEP's start of H
        cxr, cxi, cyr, cyi, czr, czi,   # per-axis connection factor, float32 parts
        beta_plus, beta_minus,          # f32 quarter-turn words per sign (D: -c)
        BACKWARD: tl.constexpr,
        BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
        CNX: tl.constexpr, CNY: tl.constexpr, CNZ: tl.constexpr,   # connection kinds
        HAS_BETA: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """folded complex ``update_H`` + folded complex BETA ``step_D``, one launch.

        ``BCX``/``BCY``/``BCZ`` take the FOUR values
        :func:`.folded_complex.folded_axis_kinds` resolves; the 0/1 mapping here wraps
        the ghost and emits neither ownership mask, which is a plane of wrong values
        rather than a crash. ``hi``/``wi``/``b`` are const for the whole dispatch and
        ``ho``/``wo`` are write-only scratch; ``f``/``u`` step IN PLACE at the
        program's OWN cell only.
        """
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = idx < n_elem
        nyz = ny * nz
        k = idx % nz
        plane = idx // nz
        j = plane % ny
        i = plane // ny

        # ============ THE WELD: update_H, computed into registers, stored to SCRATCH
        # Nothing stored here is read by this launch; every foreign tap below is a
        # recompute from PRE-LAUNCH state through the same `_h_cell_complex`. ON A
        # FOLD this is unchanged: `update_H` runs over the whole STORED extent and the
        # B those ghost planes hold was written by the driver's fills BEFORE the seam.
        (own0_re, own0_im, own1_re, own1_im, own2_re, own2_im,
         src0_re, src0_im, src1_re, src1_im, src2_re, src2_im,
         dw0, dw1, dw2) = _h_cell_complex(
            i, j, k, live, hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
            kp0, kp1, kp2, kmx, kmy, kmz, cb, start_h, ny, nz)
        tl.store(ho0 + 2 * idx, own0_re, mask=live)
        tl.store(ho0 + 2 * idx + 1, own0_im, mask=live)
        tl.store(ho1 + 2 * idx, own1_re, mask=live)
        tl.store(ho1 + 2 * idx + 1, own1_im, mask=live)
        tl.store(ho2 + 2 * idx, own2_re, mask=live)
        tl.store(ho2 + 2 * idx + 1, own2_im, mask=live)
        # The history takes B on the accumulation's cells only (the auxiliary
        # contract); every other cell of the twin takes the pre-launch word pair, so
        # the rotation hands the engine an f_w_H that kept its words there.
        tl.store(wo0 + 2 * idx, tl.where(dw0, src0_re, tl.load(
            wi0 + 2 * idx, mask=live & (dw0 == 0), other=0.0)), mask=live)
        tl.store(wo0 + 2 * idx + 1, tl.where(dw0, src0_im, tl.load(
            wi0 + 2 * idx + 1, mask=live & (dw0 == 0), other=0.0)), mask=live)
        tl.store(wo1 + 2 * idx, tl.where(dw1, src1_re, tl.load(
            wi1 + 2 * idx, mask=live & (dw1 == 0), other=0.0)), mask=live)
        tl.store(wo1 + 2 * idx + 1, tl.where(dw1, src1_im, tl.load(
            wi1 + 2 * idx + 1, mask=live & (dw1 == 0), other=0.0)), mask=live)
        tl.store(wo2 + 2 * idx, tl.where(dw2, src2_re, tl.load(
            wi2 + 2 * idx, mask=live & (dw2 == 0), other=0.0)), mask=live)
        tl.store(wo2 + 2 * idx + 1, tl.where(dw2, src2_im, tl.load(
            wi2 + 2 * idx + 1, mask=live & (dw2 == 0), other=0.0)), mask=live)

        # === the certified folded complex beta step_D curl body begins here ===

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
        a_re = own0_re
        a_im = own0_im
        b_re = own1_re
        b_im = own1_im
        c_re = own2_re
        c_im = own2_im
        a_y_re, a_y_im = _h_tap_complex(0, i, sj, k, vy,
                             hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                             kp0, kp1, kp2, kmx, kmy, kmz, cb, start_h, ny, nz)
        a_z_re, a_z_im = _h_tap_complex(0, i, j, sk, vz,
                             hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                             kp0, kp1, kp2, kmx, kmy, kmz, cb, start_h, ny, nz)
        b_x_re, b_x_im = _h_tap_complex(1, si, j, k, vx,
                             hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                             kp0, kp1, kp2, kmx, kmy, kmz, cb, start_h, ny, nz)
        b_z_re, b_z_im = _h_tap_complex(1, i, j, sk, vz,
                             hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                             kp0, kp1, kp2, kmx, kmy, kmz, cb, start_h, ny, nz)
        c_x_re, c_x_im = _h_tap_complex(2, si, j, k, vx,
                             hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                             kp0, kp1, kp2, kmx, kmy, kmz, cb, start_h, ny, nz)
        c_y_re, c_y_im = _h_tap_complex(2, i, sj, k, vy,
                             hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                             kp0, kp1, kp2, kmx, kmy, kmz, cb, start_h, ny, nz)

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
        v0_re, n0_re = curl_form(tl.load(f0 + 2 * idx, mask=live, other=0.0), p_re, curl0_re,
                                 km_y, si_y, km_z, si_z, du0)
        v0_im, n0_im = curl_form(tl.load(f0 + 2 * idx + 1, mask=live, other=0.0), p_im,
                                 curl0_im, km_y, si_y, km_z, si_z, du0)

        p_re = tl.load(u1 + 2 * idx, mask=live & du1, other=0.0)
        p_im = tl.load(u1 + 2 * idx + 1, mask=live & du1, other=0.0)
        v1_re, n1_re = curl_form(tl.load(f1 + 2 * idx, mask=live, other=0.0), p_re, curl1_re,
                                 km_z, si_z, km_x, si_x, du1)
        v1_im, n1_im = curl_form(tl.load(f1 + 2 * idx + 1, mask=live, other=0.0), p_im,
                                 curl1_im, km_z, si_z, km_x, si_x, du1)

        p_re = tl.load(u2 + 2 * idx, mask=live & du2, other=0.0)
        p_im = tl.load(u2 + 2 * idx + 1, mask=live & du2, other=0.0)
        v2_re, n2_re = curl_form(tl.load(f2 + 2 * idx, mask=live, other=0.0), p_re, curl2_re,
                                 km_x, si_x, km_y, si_y, du2)
        v2_im, n2_im = curl_form(tl.load(f2 + 2 * idx + 1, mask=live, other=0.0), p_im,
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

        # --- stores: fu (recurrence cells only), then the fields, both parts --------
        tl.store(u0 + 2 * idx, n0_re, mask=live & du0)
        tl.store(u0 + 2 * idx + 1, n0_im, mask=live & du0)
        tl.store(u1 + 2 * idx, n1_re, mask=live & du1)
        tl.store(u1 + 2 * idx + 1, n1_im, mask=live & du1)
        tl.store(u2 + 2 * idx, n2_re, mask=live & du2)
        tl.store(u2 + 2 * idx + 1, n2_im, mask=live & du2)
        tl.store(f0 + 2 * idx, v0_re, mask=live)
        tl.store(f0 + 2 * idx + 1, v0_im, mask=live)
        tl.store(f1 + 2 * idx, v1_re, mask=live)
        tl.store(f1 + 2 * idx + 1, v1_im, mask=live)
        tl.store(f2 + 2 * idx, v2_re, mask=live)
        tl.store(f2 + 2 * idx + 1, v2_im, mask=live)


    @triton.jit(do_not_specialize=["start_h"])
    def fused_beta_complex_constitutive_curl_H_to_D(
        ho0, ho1, ho2,                  # SCRATCH out: stepped Hx,Hy,Hz  (c8 as words)
        wo0, wo1, wo2,                  # SCRATCH out: stepped f_w_H*    (c8 as words)
        hi0, hi1, hi2,                  # PRE-LAUNCH Hx, Hy, Hz            (read-only)
        wi0, wi1, wi2,                  # PRE-LAUNCH f_w_Hx/y/z            (read-only)
        b0, b1, b2,                     # Bx, By, Bz                       (read-only)
        f0, f1, f2,                     # curl targets: Dx, Dy, Dz             (in/out)
        u0, u1, u2,                     # curl auxiliaries: fu_Dx/y/z          (in/out)
        kmx, sinvx, kmy, sinvy, kmz, sinvz,   # kms/sinv per axis, INTEGER lattice
        kp0, kp1, kp2,                  # kps on each component's OWN axis, INTEGER
        cb,                             # chunk record: both halves' forms and presence
        nx, ny, nz, n_elem, dtdx, start_h,   # n_elem = COMPLEX cells; MEEP's start of H
        cxr, cxi, cyr, cyi, czr, czi,   # per-axis connection factor, float32 parts
        beta_plus, beta_minus,          # f32 quarter-turn words per sign (D: -c)
        BACKWARD: tl.constexpr,
        BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
        CNX: tl.constexpr, CNY: tl.constexpr, CNZ: tl.constexpr,   # connection kinds
        HAS_BETA: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """complex beta ``update_H`` + complex beta ``step_D``, one launch.

        Identical contract to the folded kernel above; ``BCX``/``BCY``/``BCZ`` take
        the 0/1 codes of an UNFOLDED grid here.
        """
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = idx < n_elem
        nyz = ny * nz
        k = idx % nz
        plane = idx // nz
        j = plane % ny
        i = plane // ny

        # ============ THE WELD: update_H, computed into registers, stored to SCRATCH
        (own0_re, own0_im, own1_re, own1_im, own2_re, own2_im,
         src0_re, src0_im, src1_re, src1_im, src2_re, src2_im,
         dw0, dw1, dw2) = _h_cell_complex(
            i, j, k, live, hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
            kp0, kp1, kp2, kmx, kmy, kmz, cb, start_h, ny, nz)
        tl.store(ho0 + 2 * idx, own0_re, mask=live)
        tl.store(ho0 + 2 * idx + 1, own0_im, mask=live)
        tl.store(ho1 + 2 * idx, own1_re, mask=live)
        tl.store(ho1 + 2 * idx + 1, own1_im, mask=live)
        tl.store(ho2 + 2 * idx, own2_re, mask=live)
        tl.store(ho2 + 2 * idx + 1, own2_im, mask=live)
        # The history takes B on the accumulation's cells only (the auxiliary
        # contract); every other cell of the twin takes the pre-launch word pair, so
        # the rotation hands the engine an f_w_H that kept its words there.
        tl.store(wo0 + 2 * idx, tl.where(dw0, src0_re, tl.load(
            wi0 + 2 * idx, mask=live & (dw0 == 0), other=0.0)), mask=live)
        tl.store(wo0 + 2 * idx + 1, tl.where(dw0, src0_im, tl.load(
            wi0 + 2 * idx + 1, mask=live & (dw0 == 0), other=0.0)), mask=live)
        tl.store(wo1 + 2 * idx, tl.where(dw1, src1_re, tl.load(
            wi1 + 2 * idx, mask=live & (dw1 == 0), other=0.0)), mask=live)
        tl.store(wo1 + 2 * idx + 1, tl.where(dw1, src1_im, tl.load(
            wi1 + 2 * idx + 1, mask=live & (dw1 == 0), other=0.0)), mask=live)
        tl.store(wo2 + 2 * idx, tl.where(dw2, src2_re, tl.load(
            wi2 + 2 * idx, mask=live & (dw2 == 0), other=0.0)), mask=live)
        tl.store(wo2 + 2 * idx + 1, tl.where(dw2, src2_im, tl.load(
            wi2 + 2 * idx + 1, mask=live & (dw2 == 0), other=0.0)), mask=live)

        # === the certified complex beta step_D curl body begins here ===

        # --- the ghost rule, per axis (stepping._shift_up / _shift_down) ------------
        # PERIODIC wraps (a plain roll, phased or not); METALLIC serves an exact 0.0 past
        # the wall -- a (+0.0, +0.0) word pair from `other=` IS the complex metallic ghost.
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

        # --- loads: two words per operand ------------------------------------------
        a_re = own0_re
        a_im = own0_im
        b_re = own1_re
        b_im = own1_im
        c_re = own2_re
        c_im = own2_im
        a_y_re, a_y_im = _h_tap_complex(0, i, sj, k, vy,
                             hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                             kp0, kp1, kp2, kmx, kmy, kmz, cb, start_h, ny, nz)
        a_z_re, a_z_im = _h_tap_complex(0, i, j, sk, vz,
                             hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                             kp0, kp1, kp2, kmx, kmy, kmz, cb, start_h, ny, nz)
        b_x_re, b_x_im = _h_tap_complex(1, si, j, k, vx,
                             hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                             kp0, kp1, kp2, kmx, kmy, kmz, cb, start_h, ny, nz)
        b_z_re, b_z_im = _h_tap_complex(1, i, j, sk, vz,
                             hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                             kp0, kp1, kp2, kmx, kmy, kmz, cb, start_h, ny, nz)
        c_x_re, c_x_im = _h_tap_complex(2, si, j, k, vx,
                             hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                             kp0, kp1, kp2, kmx, kmy, kmz, cb, start_h, ny, nz)
        c_y_re, c_y_im = _h_tap_complex(2, i, sj, k, vy,
                             hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                             kp0, kp1, kp2, kmx, kmy, kmz, cb, start_h, ny, nz)

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

        # --- ownership mask (stepping._mask_non_owned_cells) -----------------------
        # Writes +0.0 to BOTH parts -- the array path assigns complex zero.
        if BACKWARD:
            if BCY == METALLIC:
                curl0_re = tl.where(at_y, 0.0, curl0_re)
                curl0_im = tl.where(at_y, 0.0, curl0_im)
            if BCZ == METALLIC:
                curl0_re = tl.where(at_z, 0.0, curl0_re)
                curl0_im = tl.where(at_z, 0.0, curl0_im)
            if BCX == METALLIC:
                curl1_re = tl.where(at_x, 0.0, curl1_re)
                curl1_im = tl.where(at_x, 0.0, curl1_im)
            if BCZ == METALLIC:
                curl1_re = tl.where(at_z, 0.0, curl1_re)
                curl1_im = tl.where(at_z, 0.0, curl1_im)
            if BCX == METALLIC:
                curl2_re = tl.where(at_x, 0.0, curl2_re)
                curl2_im = tl.where(at_x, 0.0, curl2_im)
            if BCY == METALLIC:
                curl2_re = tl.where(at_y, 0.0, curl2_re)
                curl2_im = tl.where(at_y, 0.0, curl2_im)
        else:
            if BCX == METALLIC:
                curl0_re = tl.where(at_x, 0.0, curl0_re)
                curl0_im = tl.where(at_x, 0.0, curl0_im)
            if BCY == METALLIC:
                curl1_re = tl.where(at_y, 0.0, curl1_re)
                curl1_im = tl.where(at_y, 0.0, curl1_im)
            if BCZ == METALLIC:
                curl2_re = tl.where(at_z, 0.0, curl2_re)
                curl2_im = tl.where(at_z, 0.0, curl2_im)

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
        v0_re, n0_re = curl_form(tl.load(f0 + 2 * idx, mask=live, other=0.0), p_re, curl0_re,
                                 km_y, si_y, km_z, si_z, du0)
        v0_im, n0_im = curl_form(tl.load(f0 + 2 * idx + 1, mask=live, other=0.0), p_im,
                                 curl0_im, km_y, si_y, km_z, si_z, du0)

        p_re = tl.load(u1 + 2 * idx, mask=live & du1, other=0.0)
        p_im = tl.load(u1 + 2 * idx + 1, mask=live & du1, other=0.0)
        v1_re, n1_re = curl_form(tl.load(f1 + 2 * idx, mask=live, other=0.0), p_re, curl1_re,
                                 km_z, si_z, km_x, si_x, du1)
        v1_im, n1_im = curl_form(tl.load(f1 + 2 * idx + 1, mask=live, other=0.0), p_im,
                                 curl1_im, km_z, si_z, km_x, si_x, du1)

        p_re = tl.load(u2 + 2 * idx, mask=live & du2, other=0.0)
        p_im = tl.load(u2 + 2 * idx + 1, mask=live & du2, other=0.0)
        v2_re, n2_re = curl_form(tl.load(f2 + 2 * idx, mask=live, other=0.0), p_re, curl2_re,
                                 km_x, si_x, km_y, si_y, du2)
        v2_im, n2_im = curl_form(tl.load(f2 + 2 * idx + 1, mask=live, other=0.0), p_im,
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
                if BCY == METALLIC:
                    q0_re = tl.where(at_y, 0.0, q0_re)
                    q0_im = tl.where(at_y, 0.0, q0_im)
                if BCZ == METALLIC:
                    q0_re = tl.where(at_z, 0.0, q0_re)
                    q0_im = tl.where(at_z, 0.0, q0_im)
                if BCX == METALLIC:
                    q1_re = tl.where(at_x, 0.0, q1_re)
                    q1_im = tl.where(at_x, 0.0, q1_im)
                if BCZ == METALLIC:
                    q1_re = tl.where(at_z, 0.0, q1_re)
                    q1_im = tl.where(at_z, 0.0, q1_im)
            else:
                if BCX == METALLIC:
                    q0_re = tl.where(at_x, 0.0, q0_re)
                    q0_im = tl.where(at_x, 0.0, q0_im)
                if BCY == METALLIC:
                    q1_re = tl.where(at_y, 0.0, q1_re)
                    q1_im = tl.where(at_y, 0.0, q1_im)
            v0_re, n0_re = fold_ladder(v0_re, n0_re, q0_re, si_y, si_z, du0)
            v0_im, n0_im = fold_ladder(v0_im, n0_im, q0_im, si_y, si_z, du0)
            v1_re, n1_re = fold_ladder(v1_re, n1_re, q1_re, si_z, si_x, du1)
            v1_im, n1_im = fold_ladder(v1_im, n1_im, q1_im, si_z, si_x, du1)

        # --- stores: fu (recurrence cells only), then the fields, both parts --------
        tl.store(u0 + 2 * idx, n0_re, mask=live & du0)
        tl.store(u0 + 2 * idx + 1, n0_im, mask=live & du0)
        tl.store(u1 + 2 * idx, n1_re, mask=live & du1)
        tl.store(u1 + 2 * idx + 1, n1_im, mask=live & du1)
        tl.store(u2 + 2 * idx, n2_re, mask=live & du2)
        tl.store(u2 + 2 * idx + 1, n2_im, mask=live & du2)
        tl.store(f0 + 2 * idx, v0_re, mask=live)
        tl.store(f0 + 2 * idx + 1, v0_im, mask=live)
        tl.store(f1 + 2 * idx, v1_re, mask=live)
        tl.store(f1 + 2 * idx + 1, v1_im, mask=live)
        tl.store(f2 + 2 * idx, v2_re, mask=live)
        tl.store(f2 + 2 * idx + 1, v2_im, mask=live)


def fused_kernel_for(variant: str) -> Any:
    """One variant's shipped kernel object, or a refusal naming the missing import."""
    _check_variant(variant)
    if triton is None:  # pragma: no cover - the laptop path
        raise ImportError(
            "the complex beta H->D pair needs Triton to launch; the predicate, the "
            "lift checks and the plan builder answer without it "
            f"({_TRITON_IMPORT_ERROR})")
    return globals()[FUSED_FUNCTIONS[variant]]


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def resolve_variant(fields: Any, pml: Any) -> Tuple[Optional[str], Tuple[str, ...]]:
    """Which variant a run resolves to, or ``(None, reasons)``.

    THE RUN DECIDES, NOT THE CALLER. The two certified curl predicates are DISJOINT by
    their own fold clauses -- the unfolded one refuses a mirror plane and the folded
    one requires one -- so both admitting is impossible on a correct pair, which is
    exactly why it is refused BY NAME here rather than resolved by branch order.
    """
    folded = _folded.folded_beta_bloch_pml_curl_coverage(
        fields, pml, CURL_SUB_STEP)
    plain = _kz.beta_bloch_pml_curl_coverage(fields, pml, CURL_SUB_STEP)
    if folded.covered and plain.covered:
        return None, (
            "both the folded and the unfolded complex beta step_D predicates admit "
            "this run: the two are meant to be disjoint on the fold clause, and which "
            "certified curl the array path would have launched is not a fact this "
            "product may decide by branch order",)
    if folded.covered:
        return "folded_beta_complex", ()
    if plain.covered:
        return "beta_complex", ()
    return None, tuple(
        [f"folded complex beta curl half: {reason}" for reason in folded.reasons]
        + [f"complex beta curl half: {reason}" for reason in plain.reasons])


def beta_complex_fused_hd_pair_coverage(
        fields: Any, pml: Any, sources: Any = None,
        variant: Optional[str] = None) -> "_coverage.Coverage":
    """May ONE launch span ``update_H`` -> the electric withdraw -> ``step_D``?

    A conjunction of the variant's two halves' OWN certified predicates plus the seam
    clauses. Nothing is weakened -- that carries the complex layout checks without this
    file restating any of them. THE ORDER IS THE DRIVER'S: the constitutive half runs first
    (driver.py:3311) and is asked first.
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

    if variant == "folded_beta_complex":
        constitutive = _folded.folded_complex_constitutive_coverage(
            fields, pml, CONSTITUTIVE_SIDE)
        curl = _folded.folded_beta_bloch_pml_curl_coverage(
            fields, pml, CURL_SUB_STEP)
    else:
        constitutive = _kz.beta_run_complex_constitutive_coverage(
            fields, pml, CONSTITUTIVE_SIDE)
        curl = _kz.beta_bloch_pml_curl_coverage(
            fields, pml, CURL_SUB_STEP)
    if not constitutive.covered:
        reasons.extend(f"complex constitutive half: {reason}"
                       for reason in constitutive.reasons)
    if not curl.covered:
        reasons.extend(f"complex beta curl half: {reason}" for reason in curl.reasons)

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

    # THE FOLD IS THE VARIANT SELECTOR, not a refusal. What is checked here is only
    # that the resolved variant and the grid agree, so a caller-forced variant cannot
    # bind the wrong code domain.
    mirrored = getattr(grid, "is_mirrored", None)
    folded_axes = [axis for axis in range(3)
                   if callable(mirrored) and bool(mirrored(axis))]
    if variant == "beta_complex" and folded_axes:
        reasons.append(
            f"axes {folded_axes} are folded but the unfolded complex beta variant was "
            f"selected: its 0/1 boundary codes map a mirror to PERIODIC, which wraps "
            f"the ghost and emits neither ownership mask -- a converged, smooth, "
            f"entirely wrong answer rather than a crash")
    if variant == "folded_beta_complex" and not folded_axes:
        reasons.append(
            "the folded complex beta variant was selected on a grid with no mirror "
            "plane; folded_axis_kinds resolves no MIRROR code there")

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


def explain_beta_complex_fused_hd_pair(fields: Any, pml: Any, sources: Any = None
                                      ) -> Dict[str, Any]:
    """The verdict under the name a report reads, WITH the variant it resolved to."""
    variant, why = resolve_variant(fields, pml)
    verdict = beta_complex_fused_hd_pair_coverage(
        fields, pml, sources, variant=variant)
    return {
        "variant": variant,
        "arms": VARIANT_ARMS.get(variant or "", None),
        "covered": bool(verdict.covered),
        "reasons": tuple(verdict.reasons) or tuple(why),
    }


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class BetaComplexFusedHdPairPlan(_plain.StartsH, ScratchWeldPairPlan):
    """ONE allocation-free launch for complex ``update_H`` and a complex beta curl.

    THE SIX ROTATING VOLUMES ARE RESOLVED PER LAUNCH, never cached, and word-viewed
    at launch time rather than plan time: the rotation moves which allocation is live.
    ``__init__`` REFUSES ALIASING between anything this launch writes and anything it
    reads, and among the written volumes themselves.
    """

    __slots__ = ("variant", "dtdx", "backward", "bc", "connections",
                 "connection_values", "beta_words", "has_beta", "_b",
                 "_targets", "_aux", "_curl_coeff", "_kps", "_kernel", "_pointer",
                 "_dtype", "_h_state")

    replaces = REPLACES
    launches_per_run = LAUNCHES_PER_RUN

    def __init__(self, variant: str, shape: Sequence[int], dtdx: float,
                 codes: Sequence[int], connections: Sequence[int],
                 connection_values: Sequence[float], beta_words: Any,
                 block: int, fields: Any, twins: Dict[str, Any],
                 flux: Sequence[Any], targets: Sequence[Any],
                 auxiliaries: Sequence[Any], curl_coefficients: Sequence[Any],
                 constitutive_coefficients: Sequence[Any], has_beta: int = 1,
                 kernel: Any = None, num_warps: Optional[int] = 1, *,
                 chunk_bounds: Any, h_state: Any = None) -> None:
        from .launch import CupyPointer, _flat  # noqa: PLC0415

        super().__init__(FAMILY, fields, twins, ROTATED, shape, block, REPLACES,
                         num_warps=num_warps, chunk_bounds=chunk_bounds)
        self.variant = _check_variant(variant)
        self.dtdx = float(dtdx)
        self.backward = BACKWARD
        self.bc = tuple(int(code) for code in codes)
        if len(self.bc) != 3:
            raise ValueError("this plan needs three boundary codes")
        # THE CODE DOMAIN IS PER VARIANT, and it is checked rather than trusted.
        if self.variant == "beta_complex" and not all(
                code in (_folded.CODE_PERIODIC, _folded.CODE_METALLIC)
                for code in self.bc):
            raise ValueError(
                f"the unfolded complex beta weld takes the 0/1 boundary codes, got "
                f"{self.bc}; a mirror code here takes the PERIODIC arm and wraps")
        if self.variant == "folded_beta_complex" and not any(
                code in (_folded.CODE_MIRROR_METALLIC, _folded.CODE_MIRROR_PERIODIC)
                for code in self.bc):
            raise ValueError(
                f"the folded complex beta weld takes folded_axis_kinds' four codes "
                f"and needs at least one mirror axis, got {self.bc}")
        # THE CONNECTION IS THE CERTIFIED CURL PLANS': three kinds and six float32
        # parts from complex_fields.connection_arguments for step_D's direction, and
        # a connection only on a PERIODIC axis, as both certified beta plans check.
        self.connections = tuple(int(kind) for kind in connections)
        self.connection_values = tuple(float(value) for value in connection_values)
        if len(self.connections) != 3 or len(self.connection_values) != 6 or any(
                kind not in _cf.CONNECTION_KINDS.values() for kind in self.connections):
            raise ValueError(
                "this plan binds three connection kinds and six float32 parts (re, im "
                "per axis), exactly as the certified curl plans do")
        for axis, (kind, code) in enumerate(zip(self.connections, self.bc)):
            if kind and code != _folded.CODE_PERIODIC:
                raise ValueError(f"axis {axis} carries a connection but its code "
                                 f"{code} does not wrap")
        # THE TWO BETA WORDS ARE PASSED THROUGH, not rebuilt: they are the real
        # quarter-turn words special_kz.beta_curl_coefficients(..., complex_storage=
        # True) rounds once on the host, the words the certified beta plans bind.
        words = tuple(beta_words)
        if len(words) != 2 or any(isinstance(word, (tuple, list)) for word in words):
            raise ValueError(f"beta_words must be the two real quarter-turn words of "
                             f"special_kz.beta_curl_coefficients(..., "
                             f"complex_storage=True), got {beta_words!r}")
        self.beta_words = tuple(float(word) for word in words)
        self.has_beta = int(has_beta)
        self._dtype = getattr(targets[0], "dtype", None)
        if str(self._dtype) != "complex64":
            raise ValueError(
                f"this plan steps complex64 word pairs, not {self._dtype}")
        if 2 * self.n_elem >= 2 ** 31:
            raise ValueError(
                f"{self.n_elem} complex cells needs {2 * self.n_elem} int32 word "
                f"indices, which overflows this kernel's 2 * idx addressing")
        self._pointer = CupyPointer
        self._b = tuple(CupyPointer(_word_view(a)) for a in flux)
        self._targets = tuple(CupyPointer(_word_view(a)) for a in targets)
        self._aux = tuple(CupyPointer(_word_view(a)) for a in auxiliaries)
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
        # MEEP's start of H, on the real product's terms (fused_hd_pair.StartsH).
        self._h_state = h_state

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
        scratch = [self._pointer(_word_view(array)) for array in writes]
        prior = [self._pointer(_word_view(array)) for array in reads]
        kernel[self._grid](
            *scratch, *prior, *self._b, *self._targets, *self._aux,
            *self._curl_coeff, *self._kps, self._cb,
            nx, ny, nz, self.n_elem, self.dtdx, self.start_h(),
            *self.connection_values,
            *self.beta_words,
            BACKWARD=self.backward,
            BCX=self.bc[0], BCY=self.bc[1], BCZ=self.bc[2],
            CNX=self.connections[0], CNY=self.connections[1],
            CNZ=self.connections[2],
            HAS_BETA=self.has_beta,
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"BetaComplexFusedHdPairPlan({self.variant}, shape={self.shape}, "
                f"bc={self.bc}, connections={self.connections}, "
                f"block={self.block}, num_warps={self.num_warps})")


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


def plan_beta_complex_fused_hd_pair(
        fields: Any, pml: Any, sources: Any = None, block: Optional[int] = None,
        num_warps: Optional[int] = 1, kernel: Any = None
) -> Optional[BetaComplexFusedHdPairPlan]:
    """Build the plan from the engine's own objects, or ``None`` when refused.

    THE BOUNDARY CODES are the variant's own; crossing them compiles and is wrong on
    every folded axis.
    """
    from .launch import chunk_bounds_for  # noqa: PLC0415
    variant, _why = resolve_variant(fields, pml)
    if variant is None:
        return None
    if not beta_complex_fused_hd_pair_coverage(
            fields, pml, sources, variant=variant).covered:
        return None
    suffix, constitutive_suffix = _sub_lattice_suffixes()
    from ..stepping import _boundary_kinds as resolve  # noqa: PLC0415

    grid = fields.grid
    kinds = resolve(grid, pml)
    if variant == "folded_beta_complex":
        codes, _reasons = _folded.folded_axis_kinds(grid, pml)
        if codes is None:  # pragma: no cover - the predicate already refused
            return None
    else:
        codes = [1 if kind == "metallic" else 0 for kind in kinds]
    # THE CONNECTION AND THE BETA WORDS ARE THE CERTIFIED MODULES': step_D's
    # not-owned word per axis (connection_arguments, backward) and the two real
    # quarter-turn words (beta_curl_coefficients, complex storage).
    phases = _cf.bloch_phase_table(grid, kinds)
    connections, values = _cf.connection_arguments(phases, backward=bool(BACKWARD))
    beta_words = _kz.beta_curl_coefficients(
        grid.beta, grid.dt, magnetic=(CURL_SUB_STEP == "step_B"),
        complex_storage=True)
    return BetaComplexFusedHdPairPlan(
        variant, grid.shape, grid.dt / grid.dx, codes, connections, values,
        beta_words,
        DEFAULT_BLOCK if block is None else block,
        fields, twin_table(fields, ROTATED),
        [getattr(fields, name) for name in CONSTITUTIVE_SOURCES],
        [getattr(fields, name) for name in IN_PLACE[:3]],
        [getattr(fields, name) for name in IN_PLACE[3:]],
        [getattr(pml, f"{stem}_{axis}{suffix}")
         for axis in "xyz" for stem in ("kms", "sinv")],
        [getattr(pml, f"kps_{axis}{constitutive_suffix}") for axis in "xyz"],
        kernel=kernel, num_warps=num_warps,
        chunk_bounds=chunk_bounds_for(fields, pml, None),
        h_state=fields,
    )


def plan_beta_complex_fused_hd_pair_from_arrays(
        variant: str, arrays: Dict[str, Any], flat: Dict[str, Any], dtdx: float,
        codes: Sequence[int], phases: Sequence[Optional[complex]], beta_words: Any,
        fields: Any, block: Optional[int] = None,
        kernel: Any = None, num_warps: Optional[int] = 1, has_beta: int = 1, *,
        chunk_bounds: Any
) -> BetaComplexFusedHdPairPlan:
    """Build from bare device arrays -- the gate's route.

    No coverage predicate runs: the caller is a harness that constructed the
    configuration deliberately, including the deliberately wrong ones. ``phases`` is
    the per-axis ``Optional[complex]`` table and step_D's connection is classified
    HERE by the certified module's own encoder; ``beta_words`` arrives as the two
    real quarter-turn words :func:`.special_kz.beta_curl_coefficients` returns. No
    start of H: a harness state has already started.
    """
    _check_variant(variant)
    shape = tuple(int(n) for n in arrays[IN_PLACE[0]].shape)
    twins = {name: arrays["scratch_" + name] for name in ROTATED}
    connections, values = _cf.connection_arguments(tuple(phases),
                                                   backward=bool(BACKWARD))
    return BetaComplexFusedHdPairPlan(
        variant, shape, dtdx, codes, connections, values, beta_words,
        DEFAULT_BLOCK if block is None else block, fields, twins,
        [arrays[name] for name in CONSTITUTIVE_SOURCES],
        [arrays[name] for name in IN_PLACE[:3]],
        [arrays[name] for name in IN_PLACE[3:]],
        [flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
        [flat[f"kps_{axis}"] for axis in "xyz"],
        has_beta=has_beta, kernel=kernel, num_warps=num_warps, chunk_bounds=chunk_bounds,
    )
