"""The H->D weld on a FOLDED grid: folded ``update_H`` welded into folded ``step_D``.

THE SECOND PRODUCT ON THE FOURTH SEAM, and the largest single cell this backend's
fusion board still carries unbuilt: ``(update_H folded -> step_D folded PML)`` is
**78 seam-instances** of a 597 denominator on
``parity/meep_gpu/results/fusion_matrix_triton_2026-09-07_cyl``, of which 75 read
``buildable_not_built`` and 3 read ``withdraw_seam``. :mod:`.fused_hd_pair` names this
cell in its own docstring as "a SECOND PRODUCT rather than a widening of this one";
this is that product, and the question it left open -- "whether a folded H->D halo
needs cells beyond the three backward neighbours" -- is answered NO from the folded
kernel's own text, below, and re-measured by this family's gate.

=============================================================================
WHAT THE FOLD CHANGES, HALF BY HALF
=============================================================================

**The constitutive half: nothing at all.** ``stepping.update_H`` is pointwise --
three ``_apply_constitutive_pml`` calls that read ``H``, ``f_w_H`` and ``B`` at ONE
cell and write the same cell (stepping.py:907-923, :2112-2143) -- and it runs over
the whole STORED extent, ghost planes included. This backend's folded ``update_H``
arm is consequently the CERTIFIED kernel: :func:`.symmetry.plan_folded_constitutive`
builds ``launch.ConstitutivePlan`` around ``kernels.constitutive_step`` and adds no
kernel of its own (symmetry.py:1009-1038). The fold reaches that sub-step "through
exactly one thing: the stored extent". So :func:`.fused_hd_pair._h_cell` and
:func:`.fused_hd_pair._h_tap` -- the certified constitutive body lifted to an
arbitrary cell, with exactly ``fused_hd_pair.CONSTITUTIVE_LIFT_EDITS`` -- are ALREADY
this product's constitutive arithmetic, character for character. They are IMPORTED
rather than copied, for the reason :mod:`.cylindrical_real_fused_hd_pair` gives when
it imports the same two: a second copy is a second place for the lift to drift.

**The curl half: one widened branch, one added mask block, and no new neighbour.**
:func:`.symmetry.pml_curl_step_folded` is ``kernels.pml_curl_step`` with

* the ghost branch written the other way round -- ``if BC == PERIODIC`` wraps and
  EVERY other code masks and serves an exact ``0.0`` -- so both mirror codes take
  the metallic ghost path (symmetry.py:222-242);
* the cell-0 ownership mask widened from ``== METALLIC`` to ``!= PERIODIC``
  (symmetry.py:263-286), which is ``stepping._mask_non_owned_cells``' own
  ``is_mirrored or is_metallic or is_axis``;
* a SECOND mask block that zeroes the LAST plane of a ``MIRROR_PERIODIC`` axis for
  every target whose Yee shift is 1 there (symmetry.py:288-315) -- the slot past
  MEEP's owned window that the fill, not the curl, writes.

None of that moves a tap. The kernel still reads its source at the program's own cell
and at its three BACKWARD neighbours ``(i-1,j,k)``, ``(i,j-1,k)``, ``(i,j,k-1)``, and
the offset lines and load lines are the shipped kernel's own spellings -- which is why
``fused_hd_pair``'s ``OWN_LOAD_EDITS`` and ``HALO_TAPS`` apply here unchanged and the
tap table this file parses out is identical to the plain product's.

WHY THE FOLDED GHOST NEEDS NO MIRROR READ. The array path's folded ghost is
``parity * field[2]`` at the near face (``stepping._shift_down``) and
``parity * field[reflect_row]`` at the far face of a folded PERIODIC axis
(``_shift_up``); the folded kernel serves an exact ``0.0`` for both instead. That is
legal because BOTH GHOST VALUES ARE DEAD IN THE CURL -- their only consumer is a
plane one of the two ownership masks zeroes (symmetry.py:18-35, measured at sub-step
granularity by ``parity/meep_gpu/gate_triton_symmetry.py`` before that file was
written). So the weld's foreign-tap set is exactly the plain product's: three backward
neighbours, no halo beyond them, no mirror-image recompute, and **no fold-dependent
kernel argument at all** -- the codes are ``tl.constexpr`` and the stored extent is
what carries the fold.

WHICH ``H`` CELLS THE TAPS LAND ON, ON A FOLD. Every one of them is an ``update_H``
OUTPUT, never a filled ``H`` ghost: there is no H fill anywhere in the engine (the
fills exist for B and D only, driver.py:3305-3310 and :3325-3330). Stored cell 0 of a
folded axis is a real stored cell whose ``B`` the driver's NEAR fill wrote BEFORE the
seam opened; the last stored slot of a folded PERIODIC axis is a real stored cell
whose ``B`` the FAR fill wrote, also before the seam. Both are const inputs to this
launch, so the recompute at either is a pure function of unwritten memory exactly as
it is in the interior.

=============================================================================
THE SEAM IS FILL-FREE, ON A FOLD AS MUCH AS OFF ONE
=============================================================================

``driver.step`` consults ``fill_B`` (:3305), runs ``zero_metal_B`` behind no consult
(:3307), consults ``fill_folded_far_ghosts_B`` (:3309) and only THEN consults
``update_H`` (:3311); the D-side triple opens at :3325, after ``step_D`` (:3315) and
the electric injection. Exactly one statement stands between this product's two
consults, and it is the electric integrated-source withdraw (:3313-3314). So
:mod:`..deposit_repair` has nothing to say about this seam, :mod:`..withdraw_hoist`
is the module that does, and the ``complex_fill_carry`` / carried-destination
machinery the folded D->E and B->H pairs need is NOT needed here and must not be
imported -- a carried fill on this seam would be a pass the driver runs again.

=============================================================================
THE SHAPE: SCRATCH OUTPUT AND FOREIGN-CELL RECOMPUTE
=============================================================================

Identical to :mod:`.fused_hd_pair`'s, and it is the plain product's docstring that
states the hazard this removes (``results/triton_fused_offdiag_electric_2026-08-20``:
42 of 60 subject cases divergent on the mirror-image seam, differing-word counts
moving 4384 -> 0 as ``BLOCK`` went 64 -> 1024). One dispatch:

* the constitutive half writes ``H_new``/``f_w_H_new`` to LAUNCH-LOCAL SCRATCH, so
  ``H``, ``f_w_H`` and ``B`` are pre-launch state for the whole dispatch and no
  program can observe another program's store. On this side the newly written
  ``f_w_H`` IS ``B`` exactly, so an in-place write would hand a racing neighbour
  ``B`` where it needs ``B_prev`` -- the easily missed half of the hazard;
* every foreign tap is a RECOMPUTE through :func:`.fused_hd_pair._h_cell`, not a load
  of another program's output;
* ``D``/``fu_D`` step IN PLACE, which is safe by construction: the curl reads and
  writes them at the program's OWN cell only;
* :class:`.offdiag_scratch_weld.ScratchWeldPairPlan` ROTATES the six ``H``/``f_w_H``
  references after the launch returns.

ONE SUB-LATTICE, ONE COEFFICIENT GROUP, and the premise is ASSERTED rather than
assumed: ``SUB_STEPS['step_D']['suffix']`` is ``''`` and
``CONSTITUTIVE_SIDES['H']['half_integer']`` is False, so the curl's ``kms_*`` and the
constitutive's are the same three volumes and the signature binds them once. Binding
the half-integer set instead compiles, launches, converges, and is a half-cell error
in the absorber profile -- which is why it is a gate mutation.

THE CODES ARE :func:`.symmetry.folded_axis_kinds`', NEVER THE PLAIN PRODUCT'S 0/1
MAPPING. ``plan_fused_hd_pair`` builds its triple as
``[1 if kind == "metallic" else 0 for kind in _boundary_kinds(grid, pml)]``
(fused_hd_pair.py:1152-1155). On a folded grid ``_boundary_kinds`` reports
``"mirror"``, which that expression maps to 0 = PERIODIC: the ghost would WRAP to the
far plane and the cell-0 mask would not be emitted at all -- a smooth, converged,
entirely wrong answer on every folded axis rather than a crash. Nothing in the kernel
can catch it, because 0 is a valid code. :func:`plan_folded_fused_hd_pair` therefore
takes its codes from ``folded_axis_kinds``, which derives the MIRROR_METALLIC /
MIRROR_PERIODIC split from ``stepping._stored_past_owned`` and CROSS-CHECKS it against
``grid.is_metallic``, and the gate arms the wrong mapping as a mutation that must be
caught on every folded fixture.

=============================================================================
WHAT THE TWO MASKS COST, AND WHY THIS PRODUCT'S GATE COMPARES ``fu_D``
=============================================================================

:mod:`.symmetry`'s own docstring says the masks are INVISIBLE at whole-step
granularity, because the driver's fill passes overwrite exactly the planes they
protect. That statement is about the TARGET. It is not true of the auxiliary, and the
difference is measured rather than reasoned: on the Metal sibling
(``metal_kernels/folded_fused_hd_pair.py``, gated 2026-09-07) dropping the top-plane
mask moved **264** words and dropping the cell-0 mask **540**, after COMPLETE driver
steps, EVERY one of them in ``fu_D`` and none in ``D``. The mechanism is in the
recurrence: ``curlN`` feeds both ``nN`` (stored to ``fu_D``) and ``vN`` (stored to
``D``), and the driver's near fill, far fill and wall clear rewrite ``D`` at those
planes and rewrite nothing of ``fu_D``.

So a whole-step comparison that looked only at the primaries would certify a
MASK-LESS folded curl as correct. This family's gate compares every stored volume the
engine allocates, arms both mask drops, and asserts the ATTRIBUTION -- each must move
``fu_D`` and must NOT move ``D``.

=============================================================================
WHAT THIS FAMILY DOES NOT CARRY
=============================================================================

ONE ARM PAIR ONLY: ``("folded", "folded PML")``. A grid with NO mirror plane is
refused BY NAME and the refusal names :mod:`.fused_hd_pair`, which is the product for
it -- the exact inverse of that module's own folded clause, so the two products are
disjoint by construction rather than by branch order. Folded COMPLEX storage, folded
BFAST, folded beta, a folded off-diagonal permittivity, a folded conductivity on a
``step_D`` target, a folded dispersive ``update_H``, a cylindrical grid, an inactive
absorber and a nonzero ``k_point`` are each refused through the two halves' own
certified predicates, which this one conjoins without weakening.

Import contract: importable WITHOUT Triton -- the predicate, the lift checks and the
plan builder (to ``None``) must answer on the laptop that is the merge bar.

=============================================================================
IT IS NOT INSTALLED, AND THE ARBITRATION IS MEASURED
=============================================================================

:data:`INSTALLABLE` is False. The first half of the reason is the plain product's and
is about wiring: routing a product through ``launch._install_certified_fused_products``
costs a LABEL, and on this backend a label ``plan_step`` can write must also be
declared in the dispatcher's own tables -- outside this package by the one-way rule it
keeps. The second half is the arbitration, and it is a MEASUREMENT over this cell's
own rows rather than an assertion: see :data:`INSTALLABLE_REASON`.

SERVED ON A FUSION BOARD IS PREDICATE ADMISSION, by that board's own definition. A
released product credits its admitted seam-instances while executing NOWHERE, and
nothing in this module implies otherwise: it is in no ``fastpath.RELEASED_FUSED_ARMS``
envelope and ``fastpath`` never plans it. **No timing exists for this shape and none
is licensed here.** The fused route does MORE memory traffic than the two singles for
the same step-level launch count; whether the saved launch and the saved ``H``
write-then-read round trip pay for three extra pointwise constitutive evaluations per
component per program is a hypothesis, not a claim.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import coverage as _coverage
from . import fused_hd_pair as _plain
from . import symmetry as _symmetry
from .. import withdraw_hoist as _withdraw_hoist
from .offdiag_scratch_weld import ScratchWeldPairPlan, twin_table

#: The family name, as the board, the battery and the weld record spell it.
FAMILY = "folded_fused_hd_pair"

#: The sub-step slot this product STARTS at -- the first half of the seam, in the
#: driver's own order, so a refusal is named on the slot the driver reaches first.
SLOT = "update_H"

#: The driver call sites ONE launch of this plan performs, in driver order
#: (driver.py:3311, :3315). NOTHING BETWEEN THEM IS CARRIED, and on a folded grid as
#: much as off one there is only the electric withdraw there to carry: every B-side
#: fill closes before :3311 and every D-side one opens after :3315.
REPLACES: Tuple[str, ...] = ("update_H", "step_D")

#: The seam name ``launch.CERTIFIED_FUSED_PAIR_SEAMS`` files this row under, and the
#: module that owns its in-seam pass. Spelled through :mod:`..withdraw_hoist` rather
#: than as a literal so the two cannot drift.
SEAM: str = _withdraw_hoist.SEAM

#: The constitutive side and the curl sub-step, in driver order.
CONSTITUTIVE_SIDE = "H"
CURL_SUB_STEP = "step_D"

#: The two arms this product implements, on ``(update_H, step_D)`` -- read off the two
#: predicates this module's own coverage conjoins, and the pair
#: ``launch.CERTIFIED_FUSED_PAIR_ARMS`` would bind. These are the labels
#: ``launch._select_slot`` writes for the folded arms (launch.py:3590, :3735).
ARMS: Tuple[str, str] = ("folded", "folded PML")

#: The six volumes one launch ROTATES. Order is the launch's argument order and is
#: not negotiable -- the plan binds every scratch then every pre-launch buffer by
#: exactly this sequence.
ROTATED: Tuple[str, ...] = _plain.ROTATED

#: The volumes the curl half steps IN PLACE, safe by construction: the curl reads and
#: writes them at the PROGRAM'S OWN CELL only.
IN_PLACE: Tuple[str, ...] = _plain.IN_PLACE

#: The constitutive half's source volumes: B, read const for the whole dispatch.
CONSTITUTIVE_SOURCES: Tuple[str, ...] = _plain.CONSTITUTIVE_SOURCES

#: ``SUB_STEPS['step_D']['backward']``, restated as a declaration a test pins against
#: ``launch.SUB_STEPS`` rather than a literal buried in the builder.
BACKWARD = _plain.BACKWARD

#: Elements per program -- the plain product's own restated copy of
#: ``kernels.DEFAULT_BLOCK``, which needs Triton to import.
DEFAULT_BLOCK = _plain.DEFAULT_BLOCK

#: Does this product bracket its launch with the DEPOSIT repair? NO, and it is a fact
#: about the driver rather than a choice: nothing is injected between the ``update_H``
#: consult and the ``step_D`` consult, so there is no deposit in this seam to carry.
CARRIES_DEPOSIT_REPAIR = False

#: Which repair the two slots would install. EMPTY: they install none.
REPAIR_PATHS: Tuple[str, ...] = ()

#: Does this product perform the seam's electric withdraw before its launch? NO in
#: this round, for the plain product's reason: the only wiring that performs the hoist
#: is ``launch._install_fused_pair``'s ``withdraw_hoist.SEAM`` branch, which is
#: unreachable while :data:`INSTALLABLE` is False. The predicate consequently refuses
#: every row with a standing in-seam withdraw BY NAME -- 3 of this cell's 78.
HOISTS_THE_WITHDRAW = False

#: May the composer install this product? NO, on every configuration.
#: ``launch._declared_uninstallable`` reads this and reports
#: :data:`INSTALLABLE_REASON` on every run.
INSTALLABLE = False

INSTALLABLE_REASON = (
    "THE COMPOSER IS NOT OFFERED THIS PRODUCT AT ALL, and that is the first half of "
    "the reason: routing it through launch._install_certified_fused_products costs a "
    "label, and on this backend a label plan_step can write must also be declared in "
    "the DISPATCHER'S own tables -- the see-through the pending-gate rung reads, and "
    "the pending list an unreleased label sits in until a ledger entry exists. Those "
    "tables are outside this package by the one-way rule this package keeps, and "
    "outside the round that built this product; "
    "parity/meep_gpu/dispatch_reachability.CERTIFIED_BUT_NOT_INSTALLED names them "
    "from the right side of that wall. The flag is what keeps a later edit that lands "
    "those lines from installing the product before a composition gate has driven it "
    "end to end. "
    "THE SECOND HALF IS THE ARBITRATION, AND IT IS MEASURED OVER THIS CELL'S OWN "
    "ROWS. Over the driver's step_B - update_H - step_D - update_E slot path launches "
    "are 4 - (installed pairs), and a TWO-SLOT H->D product takes one slot from EACH "
    "neighbour: a LOSS where both neighbouring pairs install, a TIE where exactly one "
    "does, a GAIN where NEITHER does. Joined over the 78 seam-instances of the "
    "(folded -> folded PML) cell on "
    "parity/meep_gpu/results/fusion_matrix_triton_2026-09-07_cyl: the B->H seam is "
    "admitted by folded_fused_magnetic_pair on 78 of 78, and the D->E seam is "
    "admitted on 67 (folded_fused_pair 54, folded_offdiag_fused_electric_pair 9, "
    "folded_dispersive_fused_pair 4) and unserved on 11 (10 where the folded "
    "off-diagonal electric pair's predicate refuses, 1 with no product at all). So "
    "installing this product would be a LOSS on 67 rows, a TIE on 11, and a GAIN on "
    "ZERO -- 'neither neighbour serves' is empty over this cell, which is a stronger "
    "statement than the plain product can make about its own. Unlike the plain "
    "product's cell there is no off-diagonal-permittivity row here that leaves both "
    "neighbours uninstalled. That join is a PREDICATE join off a standing board; the "
    "gate's arbitration leg re-derives it on real fixtures through the shipped "
    "composer, and its lift leg re-derives it per driven corpus row off that row's "
    "OWN composer slot table under `arbitration_over_the_driven_rows`. IT DID, AND "
    "THE MEASURED SPLIT IS NOT THE PREDICATE JOIN'S: over the 74 rows the gate "
    "actually drove it is 55 LOSS, 19 TIE and 0 GAIN, identical on both policy legs, "
    "recorded in results/triton_folded_fused_hd_pair_2026-09-07/{keep,flush}/gate.json "
    "-- against 67 / 11 / 0 over the board's 78 instances above. The two disagree "
    "because they have different denominators and different sources (a standing "
    "board's predicate join against the shipped composer on the rows that lifted), "
    "and BOTH are kept rather than one being retired: what neither shows is a single "
    "GAIN row. Installing this product raises the per-step launch count on 55 rows "
    "and leaves it unchanged on 19, which is why INSTALLABLE is False and would "
    "remain False even with the label wall down. The only span "
    "that is strictly additive on every row is a four-slot "
    "step_B -> update_H -> step_D -> update_E weld, which is not built")

__all__ = [
    "ARMS", "BACKWARD", "CARRIES_DEPOSIT_REPAIR", "CONSTITUTIVE_SIDE",
    "CONSTITUTIVE_SOURCES", "CURL_BODY_ANCHOR", "CURL_SUB_STEP", "DEFAULT_BLOCK",
    "FAMILY", "FOLDED_DECODE_END", "HOISTS_THE_WITHDRAW", "INSTALLABLE",
    "INSTALLABLE_REASON", "IN_PLACE", "REPAIR_PATHS", "REPLACES", "ROTATED", "SEAM",
    "SLOT", "SYMMETRY_PATH",
    "FoldedFusedHdPairPlan",
    "certified_curl_tail", "explain_folded_fused_hd_pair",
    "folded_curl_lift_edits", "folded_fused_hd_pair_coverage",
    "fused_constitutive_curl_H_to_D_folded_kernel", "lifted_curl_tail",
    "plan_folded_fused_hd_pair", "plan_folded_fused_hd_pair_from_arrays",
    "raw_folded_curl_tail",
]


# ---------------------------------------------------------------------------
# The machine-checked lift -- host text, no Triton
# ---------------------------------------------------------------------------

#: The file the certified FOLDED curl lives in. The plain product cuts its curl out of
#: ``kernels.py``; this one cuts it out of :mod:`.symmetry`, and the path is spelled
#: once here so the lift, the tests and the gate cannot cut from different files.
SYMMETRY_PATH = Path(__file__).with_name("symmetry.py")

#: Where the certified FOLDED curl ends its index decode. EIGHT SPACES, not the plain
#: product's four: ``pml_curl_step_folded`` is defined INSIDE ``symmetry``'s
#: ``if triton is not None:`` guard, so its body sits one level deeper than
#: ``kernels.pml_curl_step``'s. Cutting at the four-space anchor finds nothing and
#: :func:`.fused_hd_pair._cut` raises -- which is the intended failure, but the reason
#: it would raise belongs here rather than in a reader's head.
FOLDED_DECODE_END = "        i = plane // ny\n"

#: Where this module's fused kernel's lifted CURL body begins. A marker rather than
#: the shared decode anchor, because the weld block sits between the decode and the
#: curl.
CURL_BODY_ANCHOR = "        # === the certified folded step_D curl body begins here ===\n"


def raw_folded_curl_tail() -> str:
    """:func:`.symmetry.pml_curl_step_folded`'s body below the decode, dedented.

    The UNEDITED certified text. Kept as its own function because three callers need
    it -- the lift, the parsed tap table, and the gate's transcription leg -- and a
    fourth spelling of the cut is a fourth place for an anchor to drift.
    """
    return _plain._cut(  # noqa: SLF001 - the shared cutter, by design
        _plain._source_of("pml_curl_step_folded", SYMMETRY_PATH),  # noqa: SLF001
        FOLDED_DECODE_END)


def folded_curl_lift_edits() -> Tuple[Tuple[str, str], ...]:
    """The nine ``(old, new)`` replacements the folded curl half takes, GENERATED.

    THE SAME NINE THE PLAIN PRODUCT TAKES, and that is a measurement rather than a
    convenience: three own-cell magnetic loads become the registers this launch has
    already computed, and six shifted loads become a recompute at the cell the FOLDED
    emitter's own index line composed, under the FOLDED emitter's own guard. Both the
    coordinates and the mask are PARSED out of ``pml_curl_step_folded``'s own lines by
    :func:`.fused_hd_pair.offset_coordinates` and :func:`.fused_hd_pair.halo_taps`, so
    a fold-specific change to how either is spelled RAISES here instead of quietly
    redirecting a tap to the plain kernel's cell.

    NOTHING IN THE RETURNED PAIRS IS TRANSCRIBED except the call's shape. In
    particular the ghost BRANCH is not touched: a folded axis's mask
    (``vx = live & (si >= 0) & (si < nx)``) is the emitter's own, so a tap the folded
    ghost rule masked off still serves an exact ``+0.0`` -- ``_h_tap`` closes on
    ``tl.where(valid, value, 0.0)``, which is what ``other=0.0`` delivered.
    """
    tail = raw_folded_curl_tail()
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


def certified_curl_tail() -> str:
    """``pml_curl_step_folded``'s own body with every magnetic read redirected."""
    tail = raw_folded_curl_tail()
    for old, new in folded_curl_lift_edits():
        tail = _plain.needle(tail, old, new)
    for target in range(3):
        if f"g{target} +" in tail:
            raise AssertionError(
                f"the welded folded curl half still reads g{target}; in this "
                f"signature that pointer does not exist and every magnetic read must "
                f"be the register or a recompute")
    return tail


def lifted_curl_tail() -> str:
    """This module's own fused kernel's curl body, from its marker to the end."""
    source = _plain._source_of(  # noqa: SLF001
        "fused_constitutive_curl_H_to_D_folded", Path(__file__))
    return _plain._cut(source, CURL_BODY_ANCHOR)  # noqa: SLF001


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------
#
# Defined at MODULE scope behind a conditional import, not inside a builder:
# `@triton.jit` resolves a body's names -- including the `tl.constexpr` annotations --
# through the defining module's `__globals__`, so a kernel defined inside a function
# compiles to `NameError('tl is not defined')` at first launch.

try:  # pragma: no cover - the absent branch is exercised by the absence test
    import triton
    import triton.language as tl
    _TRITON_IMPORT_ERROR: Optional[BaseException] = None
except ImportError as _exc:  # pragma: no cover - the laptop path
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_IMPORT_ERROR = _exc


if triton is not None:  # pragma: no cover - device code, certified by the gate

    #: The FOUR ghost rules, identical to :mod:`.symmetry`'s own and pinned equal to
    #: them by this family's laptop test. A plan built here and a plan built there
    #: index the same table, so a divergence in these four integers is a plane of
    #: wrong values rather than a crash.
    PERIODIC = tl.constexpr(_symmetry.CODE_PERIODIC)
    METALLIC = tl.constexpr(_symmetry.CODE_METALLIC)
    MIRROR_METALLIC = tl.constexpr(_symmetry.CODE_MIRROR_METALLIC)
    MIRROR_PERIODIC = tl.constexpr(_symmetry.CODE_MIRROR_PERIODIC)

    # THE CERTIFIED CONSTITUTIVE, imported as device functions rather than copied.
    # `_h_cell` is `kernels.constitutive_step`'s side-H body as a function of a cell
    # (its lift is machine-checked in `fused_hd_pair` and re-checked by this family's
    # gate), and `_h_tap` is one component of it under the caller's validity flag.
    # The fold changes NEITHER: this backend's folded `update_H` arm builds the
    # certified `ConstitutivePlan` around the certified kernel and adds nothing
    # (symmetry.py:1009-1038). A second copy would be a second place to drift, which
    # is the reason `cylindrical_real_fused_hd_pair` imports the same two.
    from .fused_hd_pair import _h_cell, _h_tap  # noqa: PLC0415

    @triton.jit
    def fused_constitutive_curl_H_to_D_folded(
        ho0, ho1, ho2,                  # SCRATCH out: the stepped Hx, Hy, Hz
        wo0, wo1, wo2,                  # SCRATCH out: the stepped f_w_Hx, f_w_Hy, f_w_Hz
        hi0, hi1, hi2,                  # PRE-LAUNCH Hx, Hy, Hz            (read-only)
        wi0, wi1, wi2,                  # PRE-LAUNCH f_w_Hx, f_w_Hy, f_w_Hz (read-only)
        b0, b1, b2,                     # Bx, By, Bz                       (read-only)
        f0, f1, f2,                     # curl targets: Dx, Dy, Dz             (in/out)
        u0, u1, u2,                     # curl auxiliaries: fu_Dx, fu_Dy, fu_Dz (in/out)
        kmx, sinvx, kmy, sinvy, kmz, sinvz,   # kms/sinv per axis, INTEGER sub-lattice
        kp0, kp1, kp2,                  # kps on each component's OWN axis, INTEGER
        nx, ny, nz, n_elem, dtdx,
        BACKWARD: tl.constexpr,
        BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Folded ``update_H`` + folded ``step_D``, one launch, scratch output.

        ``BCX``/``BCY``/``BCZ`` take the FOUR values :func:`.symmetry.folded_axis_kinds`
        resolves -- ``PERIODIC``, ``METALLIC``, ``MIRROR_METALLIC``, ``MIRROR_PERIODIC``
        -- and that classification is the single point of failure in this file, exactly
        as it is in :mod:`.symmetry`: backwards on one axis is a plane of wrong values,
        not a crash.

        ``hi``/``wi``/``b`` are const for the whole dispatch and ``ho``/``wo`` are
        write-only scratch, so nothing written by the constitutive half is read by
        this launch; ``f``/``u`` step IN PLACE and that is safe by construction,
        because the curl reads and writes them at the program's OWN cell only. The
        fold adds NO argument: the stored extent and the four constexpr codes carry
        it.
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
        # tap from PRE-LAUNCH state through the same `_h_cell`, so no program observes
        # another program's store; the plan rotates H/f_w_H after the launch returns.
        # ON A FOLD this is unchanged: `update_H` runs over the whole STORED extent,
        # ghost planes included, and the B those planes hold was written by the
        # driver's near and far fills BEFORE this seam opened.
        own0, own1, own2, src0, src1, src2 = _h_cell(
            i, j, k, live, hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
            kp0, kp1, kp2, kmx, kmy, kmz, ny, nz)
        tl.store(ho0 + idx, own0, mask=live)
        tl.store(ho1 + idx, own1, mask=live)
        tl.store(ho2 + idx, own2, mask=live)
        tl.store(wo0 + idx, src0, mask=live)
        tl.store(wo1 + idx, src1, mask=live)
        tl.store(wo2 + idx, src2, mask=live)

        # === the certified folded step_D curl body begins here ===

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

else:  # pragma: no cover - the laptop path
    fused_constitutive_curl_H_to_D_folded = None  # type: ignore[assignment]


def fused_constitutive_curl_H_to_D_folded_kernel() -> Any:
    """The shipped kernel object, or a refusal naming the missing import."""
    if triton is None:  # pragma: no cover - the laptop path
        raise ImportError(
            "the folded fused H->D pair needs Triton to launch; the predicate, the "
            f"lift checks and the plan builder answer without it "
            f"({_TRITON_IMPORT_ERROR})")
    return fused_constitutive_curl_H_to_D_folded


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def folded_fused_hd_pair_coverage(fields: Any, pml: Any,
                                  sources: Any = None) -> "_coverage.Coverage":
    """May ONE launch span folded ``update_H`` -> the electric withdraw -> folded ``step_D``?

    A conjunction of the two FOLDED halves' own certified predicates plus the seam
    clauses. Nothing is weakened: a configuration either half refuses is refused here
    with that half's reasons, prefixed so a reader can tell which side said it.

    THE ORDER IS THE DRIVER'S. The constitutive half is asked FIRST because it runs
    first (driver.py:3311), so the first refusal a reader sees names the half the
    driver would have reached first -- and on this seam that matters, since the null
    ``update_H`` under an inactive absorber is the single largest non-fusion reason on
    the board and belongs to the constitutive side.

    WHAT EACH HALF CARRIES BY NAME. The constitutive half
    (:func:`.symmetry.folded_constitutive_coverage`) refuses a non-CuPy array module,
    complex storage, an inactive absorber, an unfolded grid, a cylindrical grid or an
    r = 0 axis, a nonzero ``k_point``, a nonlinearity, BFAST, a nonzero beta, an
    unreadable mirror phase, a folded axis with two or fewer stored cells, a folded
    axis whose two routes to "periodic or metallic" disagree, and any unallocated
    volume or coefficient length at the stored extent. The curl half
    (:func:`.symmetry.folded_composition_curl_coverage`) adds a conductivity on the
    ``step_D`` targets -- which the constitutive half deliberately does NOT refuse,
    because conductivity does not change ``update_H`` -- a recomputed rather than
    stored E, and the six ``fu_*`` plus six curl sources.

    WHAT IS DELIBERATELY NOT CONSULTED. ``deposit_repair`` (nothing is injected in
    this seam), the mirror-phase clauses (this kernel bakes no parity: its folded
    ghost is an exact ``0.0``), the far-carry clauses (the far image is a B fill
    before the seam and a D fill after it, both outside ``REPLACES``), and the wall
    seam (no fill runs between the two consults).
    """
    grid = getattr(fields, "grid", None)
    if grid is None:
        return _coverage.Coverage(False, ("fields carries no grid",))

    reasons: List[str] = []
    constitutive = _symmetry.folded_constitutive_coverage(
        fields, pml, CONSTITUTIVE_SIDE)
    if not constitutive.covered:
        reasons.extend(f"folded constitutive half: {reason}"
                       for reason in constitutive.reasons)
    curl = _symmetry.folded_composition_curl_coverage(fields, pml, CURL_SUB_STEP)
    if not curl.covered:
        reasons.extend(f"folded curl half: {reason}" for reason in curl.reasons)

    # THE SEAM'S ONE PASS. Nothing is INJECTED between the two consults, so
    # `deposit_repair` is not consulted at all and a source of either polarity is not
    # this seam's business -- the magnetic injection is one seam earlier and the
    # electric one is one seam later. What IS between them is the electric integrated-
    # source withdraw (driver.py:3313-3314), and `withdraw_hoist` owns it. IGNORANCE
    # IS NEVER AN EMPTY SET: `Fields` does not hold the source list, so a predicate
    # that inferred "no withdraw stands" from not being told would be the over-covering
    # refusal this clause exists to prevent.
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

    # THE FOLD, REQUIRED BY NAME -- the exact inverse of `fused_hd_pair`'s clause,
    # which refuses a folded grid and names THIS product. Both halves already require
    # a real fold; the restatement is so the reason reads AT THE SEAM rather than only
    # at the arms, and so the two products can be shown disjoint by construction
    # instead of by the order a composer happens to ask them in.
    mirrored = getattr(grid, "is_mirrored", None)
    folded_axes = [axis for axis in range(3)
                   if callable(mirrored) and bool(mirrored(axis))]
    if not folded_axes:
        reasons.append(
            "no axis is folded: this product implements the `folded` update_H and "
            "`folded PML` step_D arms, and an unfolded run selects the `ordinary` "
            "and `PML` arms on both slots; triton_kernels.fused_hd_pair is the "
            "product for that cell")

    # THE CODES MUST RESOLVE. `folded_axis_kinds` is the single point of failure in
    # the folded family, and a plan built on an unresolved triple would silently take
    # the PERIODIC branch on a folded axis. Asked here, with the reasons carried, so
    # the refusal is named at the predicate rather than discovered at the builder.
    codes, fold_reasons = _symmetry.folded_axis_kinds(
        grid, pml if (pml is not None and getattr(pml, "is_active", False)) else None)
    if codes is None:
        reasons.extend(f"folded axis codes: {reason}" for reason in fold_reasons)

    # THE ROTATION IS THE PRODUCT'S OWN INVARIANT. The plan swaps the ENGINE's
    # references for the six volumes in ROTATED after each launch, so those attributes
    # must be allocated.
    for name in ROTATED:
        if getattr(fields, name, None) is None:
            reasons.append(
                f"{name} is not allocated; this weld rotates it against a plan-owned "
                f"scratch twin after every launch")
    for name in IN_PLACE + CONSTITUTIVE_SOURCES:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")
    return _coverage.Coverage(not reasons, tuple(dict.fromkeys(reasons)))


def explain_folded_fused_hd_pair(fields: Any, pml: Any,
                                 sources: Any = None) -> "_coverage.Coverage":
    """The predicate under the name a report reads. One home for the verdict."""
    return folded_fused_hd_pair_coverage(fields, pml, sources)


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class FoldedFusedHdPairPlan(ScratchWeldPairPlan):
    """ONE allocation-free launch for folded ``update_H`` and folded ``step_D``.

    DELIBERATELY A SIBLING OF :class:`.fused_hd_pair.FusedHdPairPlan` RATHER THAN A
    SUBCLASS, and for the reason :class:`.symmetry.FoldedPmlCurlPlan` gives about
    itself: the two hold the same bindings but launch DIFFERENT kernels, and a
    subclass that inherited ``_launch`` would launch the unfolded one -- on a folded
    grid, with the plain kernel's two-valued codes, which is a plane of wrong values
    and not an exception.

    THE SIX ROTATING VOLUMES ARE RESOLVED PER LAUNCH, never cached: the plan owns a
    twin of each, reads the engine's CURRENT attribute immediately before the launch
    to decide which of the pair is live, and moves the engine's references only after
    the launch returns. A pointer captured at plan time would be one rotation stale --
    and stale in a way that still computes.

    ``__init__`` REFUSES ALIASING between anything this launch writes (the six twins,
    ``D`` and ``fu_D``) and anything it reads (``B``, the coefficient vectors), and
    among the written volumes themselves. The base class additionally refuses a
    scratch buffer that IS its own live volume, which would be the in-place weld this
    design exists to avoid, wearing this class's name.

    ``bc`` CARRIES FOUR-VALUED CODES, not the plain product's two, and the constructor
    refuses anything outside :data:`.symmetry.CODE_PERIODIC` ..
    :data:`.symmetry.CODE_MIRROR_PERIODIC`. It cannot refuse a triple that is merely
    WRONG -- 0 is a valid code, and mapping a folded axis to 0 is exactly the defect
    this family's gate arms as a mutation -- so the constructor's check is a floor and
    :func:`plan_folded_fused_hd_pair`'s use of ``folded_axis_kinds`` is the rule.
    """

    __slots__ = ("dtdx", "backward", "bc", "_b", "_targets", "_aux", "_curl_coeff",
                 "_kps", "_kernel", "_pointer")

    replaces = REPLACES

    def __init__(self, shape: Sequence[int], dtdx: float, codes: Sequence[int],
                 block: int, fields: Any, twins: Dict[str, Any],
                 flux: Sequence[Any], targets: Sequence[Any],
                 auxiliaries: Sequence[Any], curl_coefficients: Sequence[Any],
                 constitutive_coefficients: Sequence[Any], kernel: Any = None,
                 num_warps: Optional[int] = 1) -> None:
        from .launch import CupyPointer, _flat  # noqa: PLC0415

        super().__init__(FAMILY, fields, twins, ROTATED, shape, block, REPLACES,
                         num_warps=num_warps)
        self.dtdx = float(dtdx)
        self.backward = BACKWARD
        self.bc = tuple(int(code) for code in codes)
        if len(self.bc) != 3:
            raise ValueError("this plan needs three boundary codes")
        known = (_symmetry.CODE_PERIODIC, _symmetry.CODE_METALLIC,
                 _symmetry.CODE_MIRROR_METALLIC, _symmetry.CODE_MIRROR_PERIODIC)
        outside = [code for code in self.bc if code not in known]
        if outside:
            raise ValueError(
                f"boundary codes {outside} are outside the four folded ghost rules "
                f"{known}; the kernel's constexpr branches would take the PERIODIC "
                f"arm by default and wrap where the fold masks")
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
        """No written volume may share an allocation with a read one, or another.

        The whole design is that nothing written by the constitutive half is read; an
        aliased pair would put the schedule back into the answer, and would do it
        through a BINDING rather than through the kernel text -- which no amount of
        reading the kernel would catch.
        """
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
                  else fused_constitutive_curl_H_to_D_folded_kernel())
        nx, ny, nz = self.shape
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        scratch = [self._pointer(array) for array in writes]
        prior = [self._pointer(array) for array in reads]
        kernel[self._grid](
            *scratch, *prior, *self._b, *self._targets, *self._aux,
            *self._curl_coeff, *self._kps,
            nx, ny, nz, self.n_elem, self.dtdx,
            BACKWARD=self.backward,
            BCX=self.bc[0], BCY=self.bc[1], BCZ=self.bc[2],
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"FoldedFusedHdPairPlan(shape={self.shape}, bc={self.bc}, "
                f"block={self.block}, num_warps={self.num_warps})")


def plan_folded_fused_hd_pair(fields: Any, pml: Any, sources: Any = None,
                              block: Optional[int] = None,
                              num_warps: Optional[int] = 1,
                              kernel: Any = None
                              ) -> Optional[FoldedFusedHdPairPlan]:
    """Build the plan from the engine's own objects, or ``None`` when refused.

    ``None`` is the only refusal, for the reason :func:`.launch.plan_pml_curl` gives:
    a configuration this kernel does not carry must fall back to the array path, never
    raise into a caller that would otherwise have stepped correctly.

    THE CODES COME FROM :func:`.symmetry.folded_axis_kinds` and from nowhere else --
    see the module docstring for what the plain product's 0/1 mapping would do here.

    ``kernel`` is the mutation seam. The gate compiles a deliberately broken copy of
    the shipped kernel and hands it here; dropping the argument is not a silent
    slowdown, it is a silent DISARMING -- every mutation leg would then launch the
    shipped kernel and report the defect as uncaught.
    """
    if not folded_fused_hd_pair_coverage(fields, pml, sources).covered:
        return None
    suffix, constitutive_suffix = _plain._sub_lattice_suffixes()  # noqa: SLF001
    grid = fields.grid
    codes, _reasons = _symmetry.folded_axis_kinds(grid, pml)
    if codes is None:  # pragma: no cover - the predicate already refused
        return None
    return FoldedFusedHdPairPlan(
        grid.shape, grid.dt / grid.dx, codes,
        DEFAULT_BLOCK if block is None else block,
        fields,
        twin_table(fields, ROTATED),
        [getattr(fields, name) for name in CONSTITUTIVE_SOURCES],
        [getattr(fields, name) for name in IN_PLACE[:3]],
        [getattr(fields, name) for name in IN_PLACE[3:]],
        [getattr(pml, f"{stem}_{axis}{suffix}")
         for axis in "xyz" for stem in ("kms", "sinv")],
        [getattr(pml, f"kps_{axis}{constitutive_suffix}") for axis in "xyz"],
        kernel=kernel, num_warps=num_warps,
    )


def plan_folded_fused_hd_pair_from_arrays(
        arrays: Dict[str, Any], flat: Dict[str, Any], dtdx: float,
        codes: Sequence[int], fields: Any, block: Optional[int] = None,
        kernel: Any = None, num_warps: Optional[int] = 1
) -> FoldedFusedHdPairPlan:
    """Build from bare device arrays -- the gate's route.

    No coverage predicate runs: the caller is a harness that constructed the
    configuration deliberately, including the deliberately wrong ones. ``fields`` is
    the object whose attributes the ROTATION swaps, so a gate hands a small namespace
    holding the six rotating volumes under the engine's own names and exercises the
    rotation the engine would get. ``arrays`` supplies the twins under
    ``scratch_Hx`` ...
    """
    shape = tuple(int(n) for n in arrays[IN_PLACE[0]].shape)
    twins = {name: arrays["scratch_" + name] for name in ROTATED}
    return FoldedFusedHdPairPlan(
        shape, dtdx, codes, DEFAULT_BLOCK if block is None else block,
        fields, twins,
        [arrays[name] for name in CONSTITUTIVE_SOURCES],
        [arrays[name] for name in IN_PLACE[:3]],
        [arrays[name] for name in IN_PLACE[3:]],
        [flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
        [flat[f"kps_{axis}"] for axis in "xyz"],
        kernel=kernel, num_warps=num_warps,
    )
