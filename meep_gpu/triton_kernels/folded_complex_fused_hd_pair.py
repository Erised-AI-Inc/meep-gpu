"""The H->D weld on a FOLDED COMPLEX grid -- both of that cell's arm pairs, one launch.

ONE PRODUCT, ONE KERNEL, TWO BOARD CELLS, FIVE SEAM-INSTANCES::

    H_to_D  (folded complex             -> folded complex PML)             2
        tests:TestEigCoeffs.test_binary_grating_special_kz_2_21_2
        tests:TestModeDecomposition.test_triangular_lattice_oblique
    H_to_D  (folded complex off-diagonal -> folded complex off-diagonal PML) 3
        examples:solve-cw.py
        tests:TestArrayMetadata.test_array_metadata
        tests:TestHoleyWvgBands.test_fields_at_kx

Read off ``results/fusion_matrix_triton_2026-09-07_cyl/fusion_matrix.json``
(``aggregate.h_to_d_seam.instances``, ``one_per_row`` true, denominator 597); all five
carry ``withdraw_in_seam: false``, ``integrated_electric_sources: 0`` and
``served_by: null``. :mod:`.complex_fused_hd_pair` names these two cells in its own
docstring as needing "their own product and their own ghost-map measurement"; this is
that product, and the ghost-map question is answered below from the folded kernel's
own text and re-measured by this family's gate.

=============================================================================
WHY THE TWO CELLS ARE ONE LAUNCH -- THE STRONGEST FORM OF THE EXTRA-ARM CLAIM
=============================================================================

The off-diagonal cell is not a second kernel and not a second plan class. It is the
SAME two certified plans under a WIDER admission, and the shipped module says so in
its own words: :func:`.folded_complex.plan_folded_complex_offdiag_pml_curl` --
"The kernel and the plan class are K1's, untouched; only the ADMISSION is new" --
returns a :class:`.folded_complex.FoldedComplexPmlCurlPlan` around
:func:`.folded_complex.folded_bloch_pml_curl_step`, and
:func:`.folded_complex.plan_folded_complex_offdiag_constitutive` returns a
:class:`.complex_fields.ComplexConstitutivePlan` around
:func:`.complex_fields.bloch_constitutive_step`. Those are exactly the two plans the
PLAIN folded-complex arms build.

So ONE launch of this product serves both arm pairs, which is the shape
``launch.CERTIFIED_FUSED_PAIR_EXTRA_ARMS`` exists for -- and the precedent is this
cell's own B->H sibling: that table already carries
``folded_complex_fused_magnetic_pair: (("folded complex off-diagonal PML", "folded
complex off-diagonal"),)`` for exactly this pair of cells. :data:`ARMS` and
:data:`EXTRA_ARMS` here are the H->D transpose of that row.

WHAT DIFFERS BETWEEN THE TWO ADMISSIONS, and it is only admission:

* the off-diagonal predicate REQUIRES a real fold (the plain one admits zero folded
  axes so its own gate can measure the reduction to the unfolded kernel);
* the off-diagonal predicate refuses ``beta != 0`` by name -- beta WITH an
  off-diagonal epsilon is the pairing MEEP itself refuses (fields.cpp:548-549);
* the off-diagonal predicate's ``side='E'`` is refused by name, which does not reach
  this product: this seam's constitutive side is ``H``.

:func:`resolve_admission` reports which of the two admitted, so "which arm pair did
this product absorb" stays answerable from the record rather than from branch order.

=============================================================================
WHAT THE FOLD CHANGES IN EACH HALF
=============================================================================

**The constitutive half: nothing at all.** ``update_H`` is pointwise -- it reads
``H``, ``f_w_H`` and ``B`` at ONE cell and writes the same cell (stepping.py:907-923,
:2065-2096) -- and it runs over the whole STORED extent, ghost planes included.
:func:`.folded_complex.folded_complex_constitutive_coverage` says exactly that: "the
fold contributes only its stored extent -- which the PML coefficient vectors already
carry. The kernel launched is ``complex_fields.bloch_constitutive_step``". So
:func:`.complex_fused_hd_pair._h_cell_complex` and
:func:`.complex_fused_hd_pair._h_tap_complex` -- that body lifted to an arbitrary cell
with exactly :data:`.complex_fused_hd_pair.CONSTITUTIVE_LIFT_EDITS` -- are ALREADY
this product's constitutive arithmetic, character for character. They are IMPORTED
rather than copied, for the reason every sibling gives: a second copy is a second
place for the lift to drift.

**The curl half: three deltas, and no new neighbour.**
:func:`.folded_complex.folded_bloch_pml_curl_step` is
:func:`.complex_fields.bloch_pml_curl_step` with

* the ghost branch written the other way round -- ``if BC == PERIODIC`` wraps and
  EVERY other code masks and serves an exact ``(+0.0, +0.0)`` word pair;
* the cell-0 ownership mask widened from ``== METALLIC`` to ``!= PERIODIC``, which is
  ``stepping._mask_non_owned_cells``' own ``is_mirrored or is_metallic or is_axis``;
* a SECOND mask block zeroing the LAST plane of a ``MIRROR_PERIODIC`` axis for every
  target whose Yee shift is 1 there -- the slot past MEEP's owned window that the
  fill, not the curl, writes.

**NONE OF THAT MOVES A TAP, AND THAT IS MEASURED RATHER THAN READ OFF THE HISTORY.**
:func:`.complex_fused_hd_pair.offset_coordinates` and
:func:`.complex_fused_hd_pair.halo_taps` PARSE the folded body's own offset lines and
its own load lines; the second additionally requires the two word planes of each
shifted operand to agree on component, cell AND guard, and
:data:`.complex_fused_hd_pair.HALO_TAPS` declares the six operands the parse must
produce or RAISE. The folded complex body passes that parse unchanged: six own-cell
word loads, six shifted word PAIRS at ``(i-1,j,k)``, ``(i,j-1,k)``, ``(i,j,k-1)``, no
halo beyond them and no mirror-image read.

WHY THE FOLDED GHOST NEEDS NO MIRROR READ. The array path's folded ghost is
``parity * field[2]`` at the near face and ``parity * field[reflect_row]`` at the far
face of a folded PERIODIC axis; the folded kernel serves an exact zero for both
instead. That is legal because BOTH GHOST VALUES ARE DEAD IN THE CURL -- their only
consumer is a plane one of the two ownership masks zeroes. So the weld's foreign-tap
set is the plain product's, and the fold adds NO kernel argument at all: the codes are
``tl.constexpr`` and the stored extent carries the fold.

WHICH ``H`` CELLS THE TAPS LAND ON, ON A FOLD. Every one is an ``update_H`` OUTPUT,
never a filled ``H`` ghost: there is no H fill anywhere in the engine (the fills exist
for B and D only, driver.py:3305-3310 and :3325-3330). Stored cell 0 of a folded axis
is a real stored cell whose ``B`` the NEAR fill wrote BEFORE the seam opened; the last
stored slot of a folded PERIODIC axis is a real stored cell whose ``B`` the FAR fill
wrote, also before the seam. Both are const inputs to this launch, so the recompute at
either is a pure function of unwritten memory exactly as it is in the interior.

=============================================================================
THE SEAM IS FILL-FREE, ON A FOLD AS MUCH AS OFF ONE
=============================================================================

``driver.step`` consults ``fill_B`` (:3305), runs ``zero_metal_B`` behind no consult
(:3307), consults ``fill_folded_far_ghosts_B`` (:3309) and only THEN consults
``update_H`` (:3311); the D-side triple opens at :3325, after ``step_D`` (:3315).
Exactly one statement stands between this product's two consults and it is the
electric integrated-source withdraw (:3313-3314). So :mod:`..deposit_repair` has
nothing to say here, :mod:`..withdraw_hoist` is the module that does, and the
``complex_fill_carry`` / carried-destination machinery the folded D->E and B->H pairs
need is NOT needed and must not be imported -- a carried fill on this seam would be a
pass the driver runs again.

=============================================================================
THE CODES ARE ``folded_axis_kinds``', NEVER A 0/1 MAPPING
=============================================================================

On a folded grid ``_boundary_kinds`` reports ``"mirror"``, which the unfolded
products' ``[1 if kind == "metallic" else 0]`` maps to 0 = ``PERIODIC``: the ghost
would WRAP to the far plane and neither ownership mask would be emitted at all -- a
smooth, converged, entirely wrong answer on every folded axis rather than a crash.
Nothing in the kernel can catch it, because 0 is a valid code.
:func:`plan_folded_complex_fused_hd_pair` therefore takes its codes from
:func:`.folded_complex.folded_axis_kinds`, and the gate arms the wrong mapping as a
mutation that must be caught on every folded fixture.

=============================================================================
WHAT THE TWO MASKS COST, AND WHY THIS GATE COMPARES ``fu_D``
=============================================================================

:mod:`.symmetry`'s own docstring says the masks are INVISIBLE at whole-step
granularity, because the driver's fill passes overwrite exactly the planes they
protect. That statement is about the TARGET. It is not true of the AUXILIARY: ``curlN``
feeds both ``nN`` (stored to ``fu_D``) and ``vN`` (stored to ``D``), and the driver's
near fill, far fill and wall clear rewrite ``D`` at those planes and rewrite nothing of
``fu_D``. The Metal sibling measured it -- dropping the top-plane mask moved 264 words
and dropping the cell-0 mask 540, after COMPLETE driver steps, every one of them in
``fu_D`` and none in ``D``. This family's gate therefore compares every stored volume
the engine allocates, arms both mask drops, and asserts the ATTRIBUTION.

=============================================================================
IT IS NOT INSTALLED, AND THE ARBITRATION IS MEASURED
=============================================================================

:data:`INSTALLABLE` is False; :data:`INSTALLABLE_REASON` carries the label half and
the arbitration half, and the arbitration is a MEASUREMENT per row rather than an
assertion. SERVED ON A FUSION BOARD IS PREDICATE ADMISSION, by that board's own
definition: a released product credits its admitted seam-instances while executing
NOWHERE. This product is in no ``fastpath.RELEASED_FUSED_ARMS`` envelope and
``fastpath`` never plans it. **No timing exists for this shape and none is licensed
here.** The fused route does MORE memory traffic than the two singles for the same
step-level launch count; whether the saved launch and the saved ``H`` write-then-read
round trip pay for three extra pointwise COMPLEX constitutive evaluations per
component per program is a hypothesis.

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
from .. import withdraw_hoist as _withdraw_hoist
from .complex_fields import _word_view
from .offdiag_scratch_weld import ScratchWeldPairPlan, twin_table

#: The family name, as the board, the battery and the weld record spell it.
FAMILY = "folded_complex_fused_hd_pair"

#: The sub-step slot this product STARTS at, in the driver's own order.
SLOT = "update_H"

#: The driver call sites ONE launch performs, in driver order (driver.py:3311, :3315).
REPLACES: Tuple[str, ...] = ("update_H", "step_D")

#: The seam name ``launch.CERTIFIED_FUSED_PAIR_SEAMS`` files this row under.
SEAM: str = _withdraw_hoist.SEAM

#: The constitutive side and the curl sub-step, in driver order.
CONSTITUTIVE_SIDE = "H"
CURL_SUB_STEP = "step_D"

#: The two ADMISSIONS this one launch serves, in the order the board lists their
#: cells. NOT variants: there is ONE kernel and ONE plan class here, and these name
#: which predicate pair admitted the run.
ADMISSIONS: Tuple[str, str] = ("plain", "offdiag")

#: The PRIMARY arm pair, read off the two predicates the plain admission conjoins.
ARMS: Tuple[str, str] = ("folded complex", "folded complex PML")

#: The ADDITIONAL arm pair this ONE LAUNCH absorbs, in the shape
#: ``launch.CERTIFIED_FUSED_PAIR_EXTRA_ARMS`` takes -- the H->D transpose of the row
#: that table already carries for ``folded_complex_fused_magnetic_pair``.
EXTRA_ARMS: Tuple[Tuple[str, str], ...] = (
    ("folded complex off-diagonal", "folded complex off-diagonal PML"),)

#: ``admission -> (update_H arm, step_D arm)``.
ADMISSION_ARMS: Dict[str, Tuple[str, str]] = {
    "plain": ARMS,
    "offdiag": EXTRA_ARMS[0],
}

#: The six volumes one launch ROTATES, in the launch's argument order.
ROTATED: Tuple[str, ...] = _plain.ROTATED

#: The volumes the curl half steps IN PLACE, safe by construction.
IN_PLACE: Tuple[str, ...] = _plain.IN_PLACE

#: The constitutive half's source volumes: B, read const for the whole dispatch.
CONSTITUTIVE_SOURCES: Tuple[str, ...] = _plain.CONSTITUTIVE_SOURCES

#: ``SUB_STEPS['step_D']['backward']``, restated as a declaration a test pins.
BACKWARD = _complex.BACKWARD

#: Elements per program -- the complex product's own restated copy of
#: :data:`.complex_fields.DEFAULT_BLOCK`, PINNED against it by a laptop test.
DEFAULT_BLOCK = _complex.DEFAULT_BLOCK

#: One launch per run, and the plan counts its own. A second launch appearing here
#: without a gate is the arbitration argument silently changing.
LAUNCHES_PER_RUN = 1

#: Does this product bracket its launch with the DEPOSIT repair? NO -- a fact.
CARRIES_DEPOSIT_REPAIR = False

#: Which repair the two slots would install. EMPTY: they install none.
REPAIR_PATHS: Tuple[str, ...] = ()

#: Does this product perform the seam's electric withdraw before its launch? NO in
#: this round. On these two cells that costs nothing -- all five instances carry
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
    "from the right side of that wall. The flag is what keeps a later edit that "
    "lands those lines from installing the product before a composition gate has "
    "driven it end to end; while it stands, the gate's reference is the ARRAY PATH "
    "and the composition-installed reference AWAITS WIRING. "
    "THE SECOND HALF IS THE ARBITRATION, and it is MEASURED PER ROW rather than "
    "asserted: over the driver's step_B - update_H - step_D - update_E slot path "
    "launches are 4 - (installed pairs) and a TWO-SLOT H->D product takes one slot "
    "from EACH neighbour, so installing it is a LOSS where both neighbouring pairs "
    "install, a TIE where exactly one does, and a GAIN where NEITHER does. Which of "
    "the three each of this family's five corpus rows is comes from that row's OWN "
    "composer slot table and is recorded, row by row, in the lift leg of "
    "parity/meep_gpu/gate_triton_folded_complex_fused_hd_pair.py under "
    "`arbitration_over_the_driven_rows`. WHAT IT MEASURED, stated here rather than "
    "left to be discovered by opening an artifact, and it is NOT uniform across this "
    "family's five driven corpus rows: 2 are a LOSS (both neighbouring pairs "
    "install) and 3 are a TIE (exactly one does), with no GAIN, re-derived from the "
    "composer slot tables recorded in "
    "results/triton_folded_complex_fused_hd_pair_2026-09-07/keep/lift/. Installing "
    "this product RAISES the per-step launch count on the two and leaves it "
    "unchanged on the three; it lowers it on none, which is why INSTALLABLE is False "
    "and would remain False even with the label wall down. THE ARTIFACTS THEMSELVES "
    "SAY LOSS ON ALL FIVE and are wrong on three: the kit scored the verdict off the "
    "count of SLOTS carrying a fused label rather than the count of PAIRS, and a "
    "pair holds two slots (fixed 2026-09-08 in "
    "parity/meep_gpu/triton_hd_tail_gate_kit.py; the records are not edited and a "
    "re-run is what re-earns them). "
    "A THIRD FACT IS SPECIFIC TO THIS ARM and is measured with the "
    "others rather than assumed: the complex arms are certified under the 'keep' "
    "float32 subnormal policy only, so under 'flush' the composer selects nothing "
    "complex on these rows at all and the arbitration question does not arise. The "
    "only span that is strictly additive everywhere is a four-slot "
    "step_B -> update_H -> step_D -> update_E weld, which is not built")

#: The one helper this kernel may reach, and NO OTHER: the Bloch connection. Every
#: other product on complex storage is formed part by part on real words. A laptop
#: test scans the shipped kernel text for ``_mul_``/``_rotate_``/``_div_``/``_connect``
#: call sites and fails on anything outside this tuple.
MULTIPLY_HELPERS: Tuple[str, ...] = _complex.MULTIPLY_HELPERS

__all__ = [
    "ADMISSIONS", "ADMISSION_ARMS", "ARMS", "BACKWARD", "CARRIES_DEPOSIT_REPAIR",
    "CONSTITUTIVE_SIDE", "CONSTITUTIVE_SOURCES",
    "CURL_BODY_ANCHOR", "CURL_FUNCTION", "CURL_PATH", "CURL_SUB_STEP", "DECODE_END",
    "DEFAULT_BLOCK", "EXTRA_ARMS", "FAMILY", "HOISTS_THE_WITHDRAW", "INSTALLABLE",
    "INSTALLABLE_REASON", "IN_PLACE", "LAUNCHES_PER_RUN", "MULTIPLY_HELPERS",
    "REPAIR_PATHS", "REPLACES", "ROTATED", "SEAM", "SLOT",
    "FoldedComplexFusedHdPairPlan",
    "certified_curl_tail", "curl_lift_edits",
    "explain_folded_complex_fused_hd_pair", "folded_complex_fused_hd_pair_coverage",
    "fused_folded_complex_constitutive_curl_H_to_D_kernel", "lifted_curl_tail",
    "plan_folded_complex_fused_hd_pair",
    "plan_folded_complex_fused_hd_pair_from_arrays", "raw_curl_tail",
    "resolve_admission",
]


# ---------------------------------------------------------------------------
# The machine-checked lift -- host text, no Triton
# ---------------------------------------------------------------------------

#: The file the certified FOLDED COMPLEX curl lives in, spelled once so the lift, the
#: tests and the gate cannot cut from different files.
CURL_PATH = Path(__file__).with_name("folded_complex.py")

#: The certified curl function this product lifts, by name. ONE, for both cells.
CURL_FUNCTION = "folded_bloch_pml_curl_step"

#: Where that body ends its index decode. FOUR spaces: the function is declared at
#: module scope (``col_offset`` 0) inside :mod:`.folded_complex`'s Triton guard, the
#: same nesting :func:`.complex_fields.bloch_pml_curl_step` has. Cutting at the wrong
#: indent finds nothing and :func:`.fused_hd_pair._cut` raises, which is the intended
#: failure; the laptop test READS the indent off the shipped body rather than
#: trusting a reader's memory of which module nests its kernels.
DECODE_END = _complex.DECODE_END

#: Where this module's fused kernel's lifted CURL body begins. A marker rather than
#: the decode anchor, because the weld block sits between the two.
CURL_BODY_ANCHOR = (
    "        # === the certified folded complex step_D curl body begins here ===\n")

#: This module's fused kernel function name.
FUSED_FUNCTION = "fused_folded_complex_constitutive_curl_H_to_D"


def raw_curl_tail() -> str:
    """The certified folded complex curl body below the decode, UNEDITED, dedented.

    Kept as its own function because three callers need it -- the lift, the parsed tap
    table and the gate's transcription leg -- and a fourth spelling of the cut is a
    fourth place for an anchor to drift.
    """
    return _plain._cut(  # noqa: SLF001 - the shared cutter, by design
        _plain._source_of(CURL_FUNCTION, CURL_PATH), DECODE_END)  # noqa: SLF001


def curl_lift_edits() -> Tuple[Tuple[str, str], ...]:
    """The twelve ``(old, new)`` replacements the curl half takes, GENERATED.

    THE SAME TWELVE THE UNFOLDED COMPLEX PRODUCT TAKES, and that is a MEASUREMENT
    rather than a convenience: six own-cell word loads become the registers this
    launch has already computed, and six shifted PAIRS become a single recompute at
    the cell THE FOLDED EMITTER'S own index line composed, under THE FOLDED EMITTER'S
    own guard, returning both planes. The component, the coordinates and the mask all
    come from :func:`.complex_fused_hd_pair.offset_coordinates` and
    :func:`.complex_fused_hd_pair.halo_taps` parsing the folded body itself, so a
    fold-specific change to how either is spelled RAISES instead of quietly
    redirecting a tap to the unfolded kernel's cell.

    NOTHING IN THE RETURNED PAIRS IS TRANSCRIBED except the call's shape. In
    particular the BLOCH CONNECTION is not touched: the emitter applies it to the
    UNSHIFTED partners' plane 0 after the loads, in the emitter's own order -- which
    is why :func:`.complex_fused_hd_pair._h_tap_complex` deliberately carries none.
    """
    tail = raw_curl_tail()
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


def certified_curl_tail() -> str:
    """The certified folded complex curl body with every magnetic read redirected."""
    tail = raw_curl_tail()
    for old, new in curl_lift_edits():
        tail = _plain.needle(tail, old, new)
    for target in range(3):
        if f"g{target} +" in tail:
            raise AssertionError(
                f"the welded folded complex curl half still reads g{target}; in this "
                f"signature that pointer does not exist and every magnetic read must "
                f"be the register pair or a recompute")
    return tail


def lifted_curl_tail() -> str:
    """This module's own fused kernel's curl body, from its marker to the end."""
    source = _plain._source_of(FUSED_FUNCTION, Path(__file__))  # noqa: SLF001
    return _plain._cut(source, CURL_BODY_ANCHOR)  # noqa: SLF001


# ---------------------------------------------------------------------------
# The kernel
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

    # The names the certified folded complex curl reads -- the Bloch connection, the
    # shared forms and the record's word offsets -- the very objects
    # folded_complex.folded_bloch_pml_curl_step calls, so the lift is the same text
    # resolving to the same functions.
    from .complex_fields import _connect  # noqa: PLC0415
    from .forms import (axis_present, curl_form, curl_stencil,  # noqa: PLC0415
                        record_in)
    from .kernels import RECORD_HALF_INTEGER, RECORD_PRESENCE  # noqa: PLC0415

    #: The FOUR ghost codes, read from :mod:`.folded_complex`' own integers rather
    #: than spelled, and pinned equal to them by this family's laptop test. A plan
    #: built here and a plan built there index the same table, so a divergence in
    #: these four is a plane of wrong values rather than a crash.
    PERIODIC = tl.constexpr(_folded.CODE_PERIODIC)
    METALLIC = tl.constexpr(_folded.CODE_METALLIC)
    MIRROR_METALLIC = tl.constexpr(_folded.CODE_MIRROR_METALLIC)
    MIRROR_PERIODIC = tl.constexpr(_folded.CODE_MIRROR_PERIODIC)

    # THE CERTIFIED COMPLEX CONSTITUTIVE, imported as device functions rather than
    # copied. `_h_cell_complex` is `complex_fields.bloch_constitutive_step`'s side-H
    # body as a function of a cell (its lift is machine-checked in
    # `complex_fused_hd_pair` and re-checked by this family's gate), and
    # `_h_tap_complex` is one component of it under the caller's validity flag. THE
    # FOLD CHANGES NEITHER: this backend's folded complex `update_H` arm builds the
    # certified `ComplexConstitutivePlan` around the certified kernel and adds
    # nothing.
    from .complex_fused_hd_pair import _h_cell_complex, _h_tap_complex  # noqa: PLC0415

    @triton.jit(do_not_specialize=["start_h"])
    def fused_folded_complex_constitutive_curl_H_to_D(
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
        BACKWARD: tl.constexpr,
        BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
        CNX: tl.constexpr, CNY: tl.constexpr, CNZ: tl.constexpr,   # connection kinds
        BLOCK: tl.constexpr,
    ):
        """folded complex ``update_H`` + folded complex ``step_D``, one launch.

        ``BCX``/``BCY``/``BCZ`` take the FOUR values
        :func:`.folded_complex.folded_axis_kinds` resolves -- ``PERIODIC``,
        ``METALLIC``, ``MIRROR_METALLIC``, ``MIRROR_PERIODIC`` -- and that
        classification is the single point of failure in this file exactly as it is
        in :mod:`.folded_complex`: backwards on one axis is a plane of wrong values,
        not a crash.

        ``hi``/``wi``/``b`` are const for the whole dispatch and ``ho``/``wo`` are
        write-only scratch, so nothing written by the constitutive half is read by
        this launch; ``f``/``u`` step IN PLACE and that is safe by construction,
        because the curl reads and writes them at the program's OWN cell only. The
        fold adds NO argument: the stored extent and the four constexpr codes carry
        it. Both halves read the chunk record ``cb`` (the recompute at every cell it
        is handed, the curl at the program's own cell); ``start_h`` is the launch's
        one start-of-H scalar and is not specialized on.
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
        # tap from PRE-LAUNCH state through the same `_h_cell_complex`, so no program
        # observes another program's store; the plan rotates H/f_w_H after the launch
        # returns. ON A FOLD this is unchanged: `update_H` runs over the whole STORED
        # extent, ghost planes included, and the B those planes hold was written by
        # the driver's near and far fills BEFORE this seam opened.
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
        # the rotation hands the engine an f_w_H that kept its words there and the
        # twin is written at every cell.
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

        # === the certified folded complex step_D curl body begins here ===

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


def fused_folded_complex_constitutive_curl_H_to_D_kernel() -> Any:
    """The shipped kernel object, or a refusal naming the missing import."""
    if triton is None:  # pragma: no cover - the laptop path
        raise ImportError(
            "the folded complex H->D pair needs Triton to launch; the predicate, the "
            "lift checks and the plan builder answer without it "
            f"({_TRITON_IMPORT_ERROR})")
    return fused_folded_complex_constitutive_curl_H_to_D


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def resolve_admission(fields: Any, pml: Any) -> Tuple[Optional[str], Tuple[str, ...]]:
    """Which admission a run resolves to, or ``(None, reasons)``.

    THE RUN DECIDES, NOT THE CALLER, and unlike a two-KERNEL family this resolution
    changes NOTHING about the launch: both admissions build the same plan around the
    same kernel. What it decides is which arm pair the record credits, which is why
    the off-diagonal admission is tested FIRST: its predicate is the strictly narrower
    of the two (it requires a real fold AND an off-diagonal row AND ``beta == 0``), so
    a run it admits is unambiguously an off-diagonal row, while the plain predicate
    would also admit it if the shipped off-diagonal clause ever stopped being an
    inversion. A run BOTH admit is refused BY NAME rather than resolved by branch
    order.
    """
    offdiag = _folded.folded_complex_offdiag_pml_curl_coverage(
        fields, pml, CURL_SUB_STEP)
    plain = _folded.folded_complex_pml_curl_coverage(
        fields, pml, CURL_SUB_STEP)
    if offdiag.covered and plain.covered:
        return None, (
            "both the plain and the off-diagonal folded complex step_D predicates "
            "admit this run: the shipped off-diagonal predicate is K1's clause set "
            "with the off-diagonal clause INVERTED, so the two are meant to be "
            "disjoint, and which arm pair the record credits is not a fact this "
            "product may decide by branch order",)
    if offdiag.covered:
        return "offdiag", ()
    if plain.covered:
        return "plain", ()
    return None, tuple(
        [f"folded complex curl half: {reason}" for reason in plain.reasons]
        + [f"folded complex off-diagonal curl half: {reason}"
           for reason in offdiag.reasons])


def folded_complex_fused_hd_pair_coverage(
        fields: Any, pml: Any, sources: Any = None,
        admission: Optional[str] = None) -> "_coverage.Coverage":
    """May ONE launch span ``update_H`` -> the electric withdraw -> ``step_D``?

    A conjunction of the admitted pair's OWN certified predicates plus the seam
    clauses. Nothing is weakened: a configuration either half refuses is refused here
    with that half's reasons, prefixed so a reader can tell which side said it -- and
    that carries the complex layout checks without this file restating any of them.

    THE ORDER IS THE DRIVER'S. The constitutive half is asked FIRST because it runs
    first (driver.py:3311).
    """
    grid = getattr(fields, "grid", None)
    if grid is None:
        return _coverage.Coverage(False, ("fields carries no grid",))

    reasons: List[str] = []
    if admission is None:
        admission, why = resolve_admission(fields, pml)
        if admission is None:
            return _coverage.Coverage(False, tuple(dict.fromkeys(why)))
    elif admission not in ADMISSIONS:
        raise ValueError(f"admission must be one of {ADMISSIONS}, got {admission!r}")

    if admission == "offdiag":
        constitutive = _folded.folded_complex_offdiag_constitutive_coverage(
            fields, pml, CONSTITUTIVE_SIDE)
        curl = _folded.folded_complex_offdiag_pml_curl_coverage(
            fields, pml, CURL_SUB_STEP)
    else:
        constitutive = _folded.folded_complex_constitutive_coverage(
            fields, pml, CONSTITUTIVE_SIDE)
        curl = _folded.folded_complex_pml_curl_coverage(
            fields, pml, CURL_SUB_STEP)
    if not constitutive.covered:
        reasons.extend(f"folded complex constitutive half: {reason}"
                       for reason in constitutive.reasons)
    if not curl.covered:
        reasons.extend(f"folded complex curl half: {reason}"
                       for reason in curl.reasons)

    # THE SEAM'S ONE PASS -- the electric integrated-source withdraw
    # (driver.py:3313-3314). Nothing is INJECTED between the two consults, so
    # `deposit_repair` is not consulted at all. IGNORANCE IS NEVER AN EMPTY SET.
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

    # THE FOLD IS REQUIRED HERE rather than refused, and it is the ONE clause this
    # product adds to the two halves' own. The PLAIN folded-complex curl predicate
    # deliberately admits ZERO folded axes so its own gate can measure the reduction
    # to the unfolded kernel; that admission is right for a curl and wrong for THIS
    # weld, whose plan takes `folded_axis_kinds`' four codes. An unfolded grid is
    # `complex_fused_hd_pair`'s row, and it is refused BY NAME so the two products
    # are disjoint by construction rather than by branch order.
    mirrored = getattr(grid, "is_mirrored", None)
    folded_axes = [axis for axis in range(3)
                   if callable(mirrored) and bool(mirrored(axis))]
    if not folded_axes:
        reasons.append(
            "no axis is folded: an unfolded complex run at this seam is "
            "complex_fused_hd_pair's row (the `complex` / `complex PML` arm pair), "
            "and this product's plan binds folded_axis_kinds' four boundary codes, "
            "which resolve no mirror on such a grid")

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


def explain_folded_complex_fused_hd_pair(
        fields: Any, pml: Any, sources: Any = None) -> Dict[str, Any]:
    """The verdict under the name a report reads, WITH the admission it resolved to.

    One home for "which arm pair did this product absorb", which is the question a
    two-cell family must keep answerable from its own record.
    """
    admission, why = resolve_admission(fields, pml)
    verdict = folded_complex_fused_hd_pair_coverage(
        fields, pml, sources, admission=admission)
    return {
        "admission": admission,
        "arms": ADMISSION_ARMS.get(admission or "", None),
        "covered": bool(verdict.covered),
        "reasons": tuple(verdict.reasons) or tuple(why),
    }


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class FoldedComplexFusedHdPairPlan(_plain.StartsH, ScratchWeldPairPlan):
    """ONE allocation-free launch for folded complex ``update_H`` and ``step_D``.

    THE SIX ROTATING VOLUMES ARE RESOLVED PER LAUNCH, never cached: the plan owns a
    twin of each, reads the engine's CURRENT attribute immediately before the launch
    to decide which of the pair is live, word-views it THERE, and moves the engine's
    references only after the launch returns. A pointer captured at plan time would be
    one rotation stale -- and stale in a way that still computes.

    ``__init__`` REFUSES ALIASING between anything this launch writes (the six twins,
    ``D`` and ``fu_D``) and anything it reads (``B``, the coefficient vectors), and
    among the written volumes themselves.
    """

    __slots__ = ("dtdx", "backward", "bc", "connections", "connection_values",
                 "_b", "_targets", "_aux", "_curl_coeff", "_kps", "_kernel",
                 "_pointer", "_dtype", "_h_state")

    replaces = REPLACES
    launches_per_run = LAUNCHES_PER_RUN

    def __init__(self, shape: Sequence[int], dtdx: float, codes: Sequence[int],
                 connections: Sequence[int], connection_values: Sequence[float],
                 block: int, fields: Any, twins: Dict[str, Any],
                 flux: Sequence[Any], targets: Sequence[Any],
                 auxiliaries: Sequence[Any], curl_coefficients: Sequence[Any],
                 constitutive_coefficients: Sequence[Any], kernel: Any = None,
                 num_warps: Optional[int] = 1, *, chunk_bounds: Any,
                 h_state: Any = None) -> None:
        from .launch import CupyPointer, _flat  # noqa: PLC0415

        super().__init__(FAMILY, fields, twins, ROTATED, shape, block, REPLACES,
                         num_warps=num_warps, chunk_bounds=chunk_bounds)
        self.dtdx = float(dtdx)
        self.backward = BACKWARD
        self.bc = tuple(int(code) for code in codes)
        if len(self.bc) != 3:
            raise ValueError("this plan needs three boundary codes")
        # THE CODE DOMAIN IS CHECKED rather than trusted: this kernel's ghost branch
        # tests `== PERIODIC` and its second mask block tests `== MIRROR_PERIODIC`, so
        # a triple with no mirror code at all is the unfolded 0/1 mapping arriving by
        # mistake -- a converged, smooth, entirely wrong answer rather than a crash.
        if not any(code in (_folded.CODE_MIRROR_METALLIC,
                            _folded.CODE_MIRROR_PERIODIC) for code in self.bc):
            raise ValueError(
                f"this weld takes folded_axis_kinds' four codes and needs at least "
                f"one mirror axis, got {self.bc}; the unfolded 0/1 mapping here wraps "
                f"the ghost and emits neither ownership mask")
        # THE CONNECTION IS THE CERTIFIED CURL PLAN'S: three kinds and six float32
        # parts from complex_fields.connection_arguments for step_D's direction, and
        # a connection only on a PERIODIC axis (a folded axis carries no phase), as
        # FoldedComplexPmlCurlPlan checks.
        self.connections = tuple(int(kind) for kind in connections)
        self.connection_values = tuple(float(value) for value in connection_values)
        if len(self.connections) != 3 or len(self.connection_values) != 6 or any(
                kind not in _cf.CONNECTION_KINDS.values() for kind in self.connections):
            raise ValueError(
                "this plan binds three connection kinds and six float32 parts (re, im "
                "per axis), exactly as FoldedComplexPmlCurlPlan does")
        for axis, (kind, code) in enumerate(zip(self.connections, self.bc)):
            if kind and code != _folded.CODE_PERIODIC:
                raise ValueError(f"axis {axis} carries a connection but its code "
                                 f"{code} does not wrap")
        self._dtype = getattr(targets[0], "dtype", None)
        if str(self._dtype) != "complex64":
            raise ValueError(
                f"this plan steps complex64 word pairs, not {self._dtype}")
        # THE WORD INDEX IS int32 IN THE KERNEL. A grid whose doubled cell count
        # overflows it would address the wrong words rather than fail.
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
                  else fused_folded_complex_constitutive_curl_H_to_D_kernel())
        nx, ny, nz = self.shape
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        # WORD-VIEWED HERE, not at plan time: the rotation moves which allocation is
        # live, so the view has to be taken of whatever the engine names now.
        scratch = [self._pointer(_word_view(array)) for array in writes]
        prior = [self._pointer(_word_view(array)) for array in reads]
        kernel[self._grid](
            *scratch, *prior, *self._b, *self._targets, *self._aux,
            *self._curl_coeff, *self._kps, self._cb,
            nx, ny, nz, self.n_elem, self.dtdx, self.start_h(),
            *self.connection_values,
            BACKWARD=self.backward,
            BCX=self.bc[0], BCY=self.bc[1], BCZ=self.bc[2],
            CNX=self.connections[0], CNY=self.connections[1],
            CNZ=self.connections[2],
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"FoldedComplexFusedHdPairPlan(shape={self.shape}, bc={self.bc}, "
                f"connections={self.connections}, "
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


def plan_folded_complex_fused_hd_pair(
        fields: Any, pml: Any, sources: Any = None, block: Optional[int] = None,
        num_warps: Optional[int] = 1, kernel: Any = None
) -> Optional[FoldedComplexFusedHdPairPlan]:
    """Build the plan from the engine's own objects, or ``None`` when refused.

    ``None`` is the only refusal, for the reason :func:`.launch.plan_pml_curl` gives.
    ``kernel`` is the mutation seam: dropping it is a silent DISARMING, not a
    slowdown.

    THE BOUNDARY CODES ARE :func:`.folded_complex.folded_axis_kinds`', never the 0/1
    mapping; the CONNECTION is the certified module's, for step_D's direction.
    """
    from .launch import chunk_bounds_for  # noqa: PLC0415
    admission, _why = resolve_admission(fields, pml)
    if admission is None:
        return None
    if not folded_complex_fused_hd_pair_coverage(
            fields, pml, sources, admission=admission).covered:
        return None
    suffix, constitutive_suffix = _sub_lattice_suffixes()
    from ..stepping import _boundary_kinds as resolve  # noqa: PLC0415

    grid = fields.grid
    codes, _reasons = _folded.folded_axis_kinds(grid, pml)
    if codes is None:  # pragma: no cover - the predicate already refused
        return None
    kinds = resolve(grid, pml)
    # THE CONNECTION IS THE CERTIFIED MODULE'S, not this one's:
    # `connection_arguments` classifies each axis's not-owned word for step_D's
    # direction (eikna above the window under its backward differences) and rounds a
    # product's parts to float32, as the certified folded curl plan binds it.
    phases = _cf.bloch_phase_table(grid, kinds)
    connections, values = _cf.connection_arguments(phases, backward=bool(BACKWARD))
    return FoldedComplexFusedHdPairPlan(
        grid.shape, grid.dt / grid.dx, codes, connections, values,
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


def plan_folded_complex_fused_hd_pair_from_arrays(
        arrays: Dict[str, Any], flat: Dict[str, Any], dtdx: float,
        codes: Sequence[int], phases: Sequence[Optional[complex]],
        fields: Any, block: Optional[int] = None, kernel: Any = None,
        num_warps: Optional[int] = 1, *, chunk_bounds: Any) -> FoldedComplexFusedHdPairPlan:
    """Build from bare device arrays -- the gate's route.

    No coverage predicate runs: the caller is a harness that constructed the
    configuration deliberately, including the deliberately wrong ones. ``fields`` is
    the object whose attributes the ROTATION swaps; ``arrays`` supplies the twins
    under ``scratch_Hx`` ...; ``phases`` is the per-axis ``Optional[complex]`` table
    and step_D's connection is classified HERE by the certified module's own
    encoder. No start of H: a harness state has already started.
    """
    shape = tuple(int(n) for n in arrays[IN_PLACE[0]].shape)
    twins = {name: arrays["scratch_" + name] for name in ROTATED}
    connections, values = _cf.connection_arguments(tuple(phases),
                                                   backward=bool(BACKWARD))
    return FoldedComplexFusedHdPairPlan(
        shape, dtdx, codes, connections, values,
        DEFAULT_BLOCK if block is None else block,
        fields, twins,
        [arrays[name] for name in CONSTITUTIVE_SOURCES],
        [arrays[name] for name in IN_PLACE[:3]],
        [arrays[name] for name in IN_PLACE[3:]],
        [flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
        [flat[f"kps_{axis}"] for axis in "xyz"],
        kernel=kernel, num_warps=num_warps, chunk_bounds=chunk_bounds,
    )
