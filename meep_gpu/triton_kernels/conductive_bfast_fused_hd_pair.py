"""The H->D weld on the two real curl tails the Cartesian product does not reach.

ONE PRODUCT, TWO VARIANTS, TWO BOARD CELLS. :mod:`.fused_hd_pair` welds the certified
``update_H`` into the certified LOSSLESS real-PML ``step_D``. Two seam-instances on
the Triton fusion board sit on the SAME seam with the SAME ``update_H`` half and a
DIFFERENT ``step_D`` tail, and they are two of the board's last fifteen
``buildable_not_built`` H->D instances::

    H_to_D  (ordinary  -> conductive PML)   1 instance
        tests:TestAdjointSolver.test_damping
    H_to_D  (BFAST run -> BFAST PML)        1 instance
        tests:TestReflectanceAngular.test_reflectance_angular_2_35_7

Both read off ``results/fusion_matrix_triton_2026-09-07_cyl/fusion_matrix.json``
(``aggregate.h_to_d_seam.instances``, ``one_per_row`` true, denominator 597), each
with ``withdraw_in_seam: false``, ``integrated_electric_sources: 0`` and
``served_by: null``.

=============================================================================
WHY THE TWO CELLS ARE ONE PRODUCT, AND IT IS A MEASUREMENT
=============================================================================

The ``update_H`` half is the SAME CERTIFIED BODY on both rows, and that is read off
the two arms' own predicates rather than off the board's cell label:

* the ``ordinary`` arm at ``update_H`` is ``coverage.constitutive_coverage(..., 'H')``
  around :func:`.kernels.constitutive_step` -- what :mod:`.fused_hd_pair` already
  lifts;
* the ``BFAST run`` arm at ``update_H`` is
  :func:`.bfast_curl.bfast_run_constitutive_coverage`, which is that same clause set
  with the BFAST clause INVERTED and nothing else changed, and whose own docstring
  says so: "the constitutive sub-steps read nothing bfast-dependent ... so admission
  delegates the ARITHMETIC to the certified ``kernels.constitutive_step`` unchanged --
  no new kernel, no new sub-step". Its builder,
  :func:`.bfast_curl.plan_bfast_run_constitutive`, returns a
  ``launch.ConstitutivePlan``.

So the two cells differ in the CURL HALF AND NOTHING ELSE. That is the shape
``cuda_kernels.conductive_bfast_fused_hd_pair`` states for the same two cells on the
CUDA board, and this module takes it: ONE family, ONE predicate that resolves the
variant off the run, TWO emitted kernels, and :data:`ARMS` plus :data:`EXTRA_ARMS`
rather than a widened arm value -- the shape
``launch.CERTIFIED_FUSED_PAIR_EXTRA_ARMS`` already carries for
``folded_complex_fused_magnetic_pair`` and ``no_pml_fused_electric_pair``.

**THE ONE PLACE THIS DIFFERS FROM THOSE TWO EXTRA-ARM ROWS IS NAMED HERE RATHER THAN
GLOSSED.** Both of those serve their second arm pair with the SAME LAUNCH. This
product serves its second arm pair with a DIFFERENT KERNEL of the same family, chosen
by :func:`resolve_variant` off the run's own predicates. One label, two absorb rows,
two device strings -- and :func:`explain_conductive_bfast_fused_hd_pair` reports which
variant a configuration resolved to, so "which arm pair did this product absorb" stays
answerable from the record.

=============================================================================
THE SHAPE: SCRATCH OUTPUT AND FOREIGN-CELL RECOMPUTE, UNCHANGED
=============================================================================

Both certified ``step_D`` tails read ``H`` at the thread's own cell AND at its three
BACKWARD neighbours, and ``update_H`` writes ``H``. Welded in place a program would
read ``H`` at a neighbour at a moment the block schedule decides -- the hazard this
backend measured on the mirror-image D->E seam
(``results/triton_fused_offdiag_electric_2026-08-20``: 42 of 60 subject cases
divergent, differing words moving 4384 -> 0 as ``BLOCK`` went 64 -> 1024) -- and it
would read ``f_w_H`` the same way, which is the easily missed half: on this side the
newly written ``f_w_H`` IS ``B`` exactly, so an in-place write hands a racing
neighbour ``B`` where it needs ``B_prev``.

So, exactly as in :mod:`.fused_hd_pair`:

* **the constitutive half writes nothing in place.** ``H_new`` and ``f_w_H_new`` go to
  LAUNCH-LOCAL WRITE-ONLY SCRATCH, so ``H``, ``f_w_H`` and ``B`` are pre-launch state
  for the whole dispatch and no program observes another program's store;
* **every foreign tap is a RECOMPUTE** through :func:`.fused_hd_pair._h_cell` -- the
  certified ``constitutive_step`` body for side ``H``, whole, evaluated at an
  arbitrary cell. A pure function of unwritten memory has no schedule to depend on;
* ``D``, ``fu_D`` and (on the conductive variant) ``f_cond_D`` step IN PLACE, and that
  is safe by construction: every one of those is read and written at the program's OWN
  cell only (``f0 + idx``, ``u0 + idx``, ``c0 + idx``);
* :class:`.offdiag_scratch_weld.ScratchWeldPairPlan` ROTATES the six ``H``/``f_w_H``
  references after the launch returns.

``_h_cell`` and ``_h_tap`` are IMPORTED from :mod:`.fused_hd_pair` rather than copied,
for the reason :mod:`.cylindrical_real_fused_hd_pair` and :mod:`.folded_fused_hd_pair`
give when they import the same two: a second copy is a second place for the lift to
drift.

**THAT THE TWO TAILS' MAGNETIC ACCESS SHAPE IS THE PLAIN CURL'S IS MEASURED, NOT
ASSUMED.** :func:`.fused_hd_pair.offset_coordinates` and
:func:`.fused_hd_pair.halo_taps` PARSE each certified body's own offset lines and its
own load lines; :data:`.fused_hd_pair.HALO_TAPS` declares the six shifted operands and
the parse must produce exactly them or it RAISES. Both tails pass that parse
unchanged, which is what licenses one weld shape over two kernels.

=============================================================================
ONE SUB-LATTICE, ONE COEFFICIENT GROUP
=============================================================================

``SUB_STEPS['step_D']['suffix']`` is ``''`` and
``coverage.CONSTITUTIVE_SIDES['H']['half_integer']`` is False, so the curl's
``kms_x/_y/_z`` and the constitutive's ARE THE SAME THREE VOLUMES and the signature
binds them once. Both suffixes are READ from the shipped tables and their equality is
ASSERTED in :func:`plan_conductive_bfast_fused_hd_pair` rather than assumed: binding
the half-integer set instead compiles, launches, converges, and is a half-cell error
in the absorber profile. The gate arms it as a mutation.

=============================================================================
WHAT SITS IN THE SEAM
=============================================================================

:data:`CARRIES_DEPOSIT_REPAIR` is False and it is a FACT rather than a decision:
nothing is injected between the ``update_H`` consult (driver.py:3311) and the
``step_D`` consult (:3315). The one pass that IS there is the electric integrated-
source withdraw (:3313-3314), and :mod:`..withdraw_hoist` owns it -- one slot, not
two; a hoist, not a bracket. :data:`HOISTS_THE_WITHDRAW` is False in this round
because :data:`INSTALLABLE` is False, which makes ``launch._install_fused_pair``'s
withdraw-hoist branch unreachable for this product; the predicate therefore refuses BY
NAME every row whose electric withdraw does work.

WHAT THAT COSTS ON THESE TWO CELLS, MEASURED: **nothing**. Both instances carry
``integrated_electric_sources: 0`` and ``withdraw_in_seam: false`` on the board. The
clause is still evaluated on every row rather than assumed away -- ignorance is never
an empty set, so an undeclared source list is a refusal.

=============================================================================
THE CONDUCTIVE VARIANT'S OWN THREE FACTS
=============================================================================

1. ``f_cond_D*``, ``condfac_*`` and ``condinv_*`` are bound POSITIONALLY, and a
   component the run leaves lossless binds the PLACEHOLDER
   :class:`.conductivity.ConductivePmlCurlPlan` binds for it -- the target itself,
   never dereferenced because ``COND{n}`` is 0 for that component. This module does
   not re-derive that policy; it calls the same resolution.
2. ``COND0/1/2`` come from :func:`.conductivity.conductive_targets`, which reads the
   run's own per-component conductivity. A plan with NO conductive component is
   ``kernels.pml_curl_step``'s configuration, and :mod:`.fused_hd_pair` is the product
   for it -- refused here BY NAME rather than launched with three zeros.
3. The conductive history is IN-PLACE state read and written at the own cell, so it
   needs no scratch and takes no part in the rotation. :data:`IN_PLACE` says so.

THE BFAST VARIANT'S: ``f_bfast_D*`` are the run's own state arrays, bound
pointer-identically (the driver-sync requirement
:func:`.bfast_curl.plan_bfast_pml_curl` states), and the six ``k1``/``k2`` scalars come
from :func:`.bfast_curl.bfast_curl_coefficients` with the GRID'S OWN declared
invariance flags -- never a shape test -- exactly as the certified plan computes them.

=============================================================================
IT IS NOT INSTALLED, AND THE REASON HAS TWO HALVES
=============================================================================

:data:`INSTALLABLE` is False and :data:`INSTALLABLE_REASON` carries both halves: the
label the composer would need, and the ARBITRATION, which is MEASURED PER ROW rather
than asserted. Over the driver's ``step_B - update_H - step_D - update_E`` slot path
launches are ``4 - (installed pairs)`` and a two-slot H->D product takes one slot from
EACH neighbour, so installing it is a LOSS where both neighbouring pairs install, a
TIE where exactly one does, and a GAIN where NEITHER does.

SERVED ON A FUSION BOARD IS PREDICATE ADMISSION, by that board's own definition. A
released product credits its admitted seam-instances while executing NOWHERE, and this
module says exactly that: it is in no ``fastpath.RELEASED_FUSED_ARMS`` envelope and
``fastpath`` never plans it. **No timing exists for this shape and none is licensed by
anything here.** The fused route does MORE memory traffic than the two singles for the
same step-level launch count; whether the saved launch and the saved ``H``
write-then-read round trip pay for three extra pointwise constitutive evaluations per
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

from . import bfast_curl as _bfast
from . import conductivity as _conductivity
from . import coverage as _coverage
from . import fused_hd_pair as _plain
from .. import withdraw_hoist as _withdraw_hoist
from .offdiag_scratch_weld import ScratchWeldPairPlan, twin_table

#: The family name, as the board, the battery and the weld record spell it.
FAMILY = "conductive_bfast_fused_hd_pair"

#: The sub-step slot this product STARTS at -- the first half of the seam, in the
#: driver's own order, so a refusal is named on the slot the driver reaches first.
SLOT = "update_H"

#: The driver call sites ONE launch of this plan performs, in driver order
#: (driver.py:3311, :3315). NOTHING BETWEEN THEM IS CARRIED.
REPLACES: Tuple[str, ...] = ("update_H", "step_D")

#: The seam name ``launch.CERTIFIED_FUSED_PAIR_SEAMS`` files this row under, and the
#: module that owns its in-seam pass. Spelled through :mod:`..withdraw_hoist` rather
#: than as a literal so the two cannot drift.
SEAM: str = _withdraw_hoist.SEAM

#: The constitutive side and the curl sub-step, in driver order.
CONSTITUTIVE_SIDE = "H"
CURL_SUB_STEP = "step_D"

#: The two variants this family emits, in the order the board lists their cells.
VARIANTS: Tuple[str, str] = ("conductive", "bfast")

#: The PRIMARY arm pair, on ``(update_H, step_D)``, read off the two predicates the
#: conductive variant's coverage conjoins.
ARMS: Tuple[str, str] = ("ordinary", "conductive PML")

#: The ADDITIONAL arm pair this family absorbs -- the BFAST cell -- in the shape
#: ``launch.CERTIFIED_FUSED_PAIR_EXTRA_ARMS`` takes. A SECOND ROW rather than a
#: widened value, so every existing pin on a primary row stays a pin.
EXTRA_ARMS: Tuple[Tuple[str, str], ...] = (("BFAST run", "BFAST PML"),)

#: ``variant -> (update_H arm, step_D arm)``. One table, so the two above and the
#: resolver cannot disagree about which cell a variant serves.
VARIANT_ARMS: Dict[str, Tuple[str, str]] = {
    "conductive": ARMS,
    "bfast": EXTRA_ARMS[0],
}

#: The six volumes one launch ROTATES: the stored magnetic field and its split-field
#: history. Order is the launch's argument order and is not negotiable.
ROTATED: Tuple[str, ...] = _plain.ROTATED

#: The volumes the curl half steps IN PLACE, per variant, and it is safe by
#: construction for every one: the curl reads and writes them at the PROGRAM'S OWN
#: CELL only (``f0 + idx``, ``u0 + idx``, ``c0 + idx``, ``s0 + idx``).
IN_PLACE_BY_VARIANT: Dict[str, Tuple[str, ...]] = {
    "conductive": _plain.IN_PLACE + ("f_cond_Dx", "f_cond_Dy", "f_cond_Dz"),
    "bfast": _plain.IN_PLACE + ("f_bfast_Dx", "f_bfast_Dy", "f_bfast_Dz"),
}

#: The targets and auxiliaries both variants share -- the plain product's own.
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
#: about the driver rather than a choice: nothing is injected in this seam.
CARRIES_DEPOSIT_REPAIR = False

#: Which repair the two slots would install. EMPTY: they install none.
REPAIR_PATHS: Tuple[str, ...] = ()

#: Does this product perform the seam's electric withdraw before its launch? NO in
#: this round -- see the module docstring. On these two cells that costs nothing:
#: both instances carry no integrated electric source on the board.
HOISTS_THE_WITHDRAW = False

#: May the composer install this product? NO, on every configuration.
#: ``launch._declared_uninstallable`` reads this and reports
#: :data:`INSTALLABLE_REASON` on every run.
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
    "install, a TIE where exactly one does, and a GAIN where NEITHER does. Which of "
    "the three each of this family's two corpus rows is comes from that row's OWN "
    "composer slot table and is recorded, row by row, in the lift leg of "
    "parity/meep_gpu/gate_triton_conductive_bfast_fused_hd_pair.py under "
    "`arbitration_over_the_driven_rows`. WHAT IT MEASURED, stated here rather than "
    "left to be discovered by opening an artifact: BOTH of this family's two driven "
    "corpus rows are a LOSS -- 2 of 2, no TIE, no GAIN -- in "
    "results/triton_conductive_bfast_fused_hd_pair_2026-09-07/keep/lift/. Installing "
    "this product would RAISE the per-step launch count on every row it claims, "
    "which is why INSTALLABLE is False and would remain False even with the label "
    "wall down. The verdict is the PAIR count, not the count of slots carrying a "
    "fused label: the recorded runs scored those rows through a slot count and a "
    "one-pair row read LOSS where the rule makes it a TIE (kit fixed 2026-09-08, "
    "artifacts left alone, a re-run owed). Both of these rows are two-pair rows and "
    "LOSS either way. The only span that is strictly additive on every row is a "
    "four-slot step_B -> update_H -> step_D -> update_E weld, which is not built")

__all__ = [
    "ARMS", "BACKWARD", "BFAST_STATES", "CARRIES_DEPOSIT_REPAIR",
    "CONDUCTIVE_HISTORY", "CONSTITUTIVE_SIDE", "CONSTITUTIVE_SOURCES",
    "CURL_BODY_ANCHOR", "CURL_FUNCTIONS", "CURL_PATHS", "CURL_SUB_STEP",
    "DEFAULT_BLOCK", "EXTRA_ARMS", "FAMILY", "HOISTS_THE_WITHDRAW", "INSTALLABLE",
    "INSTALLABLE_REASON", "IN_PLACE", "IN_PLACE_BY_VARIANT", "REPAIR_PATHS",
    "REPLACES", "ROTATED", "SEAM", "SLOT", "VARIANTS", "VARIANT_ARMS",
    "ConductiveBfastFusedHdPairPlan",
    "certified_curl_tail", "conductive_bfast_fused_hd_pair_coverage",
    "curl_lift_edits", "explain_conductive_bfast_fused_hd_pair",
    "fused_kernel_for", "lifted_curl_tail", "plan_conductive_bfast_fused_hd_pair",
    "plan_conductive_bfast_fused_hd_pair_from_arrays", "raw_curl_tail",
    "resolve_variant",
]

#: The conductive variant's three in-place history volumes, in argument order.
CONDUCTIVE_HISTORY: Tuple[str, str, str] = ("f_cond_Dx", "f_cond_Dy", "f_cond_Dz")

#: The BFAST variant's three in-place state volumes, in argument order. READ from
#: the certified table rather than spelled, so a rename there raises here.
BFAST_STATES: Tuple[str, ...] = tuple(_bfast.BFAST_STATE_NAMES[CURL_SUB_STEP])


# ---------------------------------------------------------------------------
# The machine-checked lift -- host text, no Triton
# ---------------------------------------------------------------------------

#: Which file each variant's certified curl lives in, spelled once so the lift, the
#: tests and the gate cannot cut from different files.
CURL_PATHS: Dict[str, Path] = {
    "conductive": Path(__file__).with_name("conductivity.py"),
    "bfast": Path(__file__).with_name("bfast_curl.py"),
}

#: The certified curl function each variant lifts, by name.
CURL_FUNCTIONS: Dict[str, str] = {
    "conductive": "conductive_pml_curl_step",
    "bfast": "bfast_pml_curl_step",
}

#: Where each certified body ends its index decode. The two differ in INDENT and not
#: in text: ``conductive_pml_curl_step`` is defined INSIDE ``conductivity``'s
#: ``if triton is not None:`` guard, so its body sits one level deeper than
#: ``bfast_curl.bfast_pml_curl_step``'s. Cutting at the wrong one finds nothing and
#: :func:`.fused_hd_pair._cut` raises -- which is the intended failure, but the reason
#: belongs here rather than in a reader's head.
DECODE_END: Dict[str, str] = {
    "conductive": "        i = plane // ny\n",
    "bfast": "    i = plane // ny\n",
}

#: Where each of this module's fused kernels' lifted CURL body begins. A marker
#: rather than the decode anchor, because the weld block sits between the two.
CURL_BODY_ANCHOR: Dict[str, str] = {
    "conductive":
        "        # === the certified conductive step_D curl body begins here ===\n",
    "bfast":
        "        # === the certified BFAST step_D curl body begins here ===\n",
}


def _check_variant(variant: str) -> str:
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    return variant


def raw_curl_tail(variant: str) -> str:
    """One variant's certified curl body below its decode, UNEDITED and dedented.

    Kept as its own function because three callers need it -- the lift, the parsed tap
    table and the gate's transcription leg -- and a fourth spelling of the cut is a
    fourth place for an anchor to drift.
    """
    _check_variant(variant)
    return _plain._cut(  # noqa: SLF001 - the shared cutter, by design
        _plain._source_of(CURL_FUNCTIONS[variant],  # noqa: SLF001
                          CURL_PATHS[variant]),
        DECODE_END[variant])


def curl_lift_edits(variant: str) -> Tuple[Tuple[str, str], ...]:
    """The nine ``(old, new)`` replacements one variant's curl half takes, GENERATED.

    THE SAME NINE THE PLAIN PRODUCT TAKES, on both tails, and that is a MEASUREMENT
    rather than a convenience: three own-cell magnetic loads become the registers this
    launch has already computed, and six shifted loads become a recompute at the cell
    THIS EMITTER'S own index line composed, under THIS EMITTER'S own guard. Both the
    coordinates and the mask are PARSED out of the certified body by
    :func:`.fused_hd_pair.offset_coordinates` and :func:`.fused_hd_pair.halo_taps`, so
    a tail-specific change to how either is spelled RAISES here instead of quietly
    redirecting a tap to the plain kernel's cell.

    NOTHING IN THE RETURNED PAIRS IS TRANSCRIBED except the call's shape. The ghost
    BRANCH is not touched: a tap the certified ghost rule masked off still serves an
    exact ``+0.0`` -- ``_h_tap`` closes on ``tl.where(valid, value, 0.0)``, which is
    what ``other=0.0`` delivered.
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


#: Each variant's fused kernel function name in THIS module.
FUSED_FUNCTIONS: Dict[str, str] = {
    "conductive": "fused_constitutive_curl_H_to_D_conductive",
    "bfast": "fused_constitutive_curl_H_to_D_bfast",
}


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

    #: Boundary codes, identical to :mod:`.kernels`' own and pinned equal to them by
    #: this family's laptop test.
    PERIODIC = tl.constexpr(0)
    METALLIC = tl.constexpr(1)

    # THE CERTIFIED CONSTITUTIVE, imported as device functions rather than copied.
    # `_h_cell` is `kernels.constitutive_step`'s side-H body as a function of a cell
    # (its lift is machine-checked in `fused_hd_pair` and re-checked by this family's
    # gate), and `_h_tap` is one component of it under the caller's validity flag.
    # NEITHER TAIL CHANGES IT: both cells' `update_H` arm delegates to the certified
    # kernel unchanged, which is what makes one constitutive half serve two cells.
    from .fused_hd_pair import _h_cell, _h_tap  # noqa: PLC0415

    # THE CONDUCTIVE RECURRENCE, likewise imported rather than copied.
    # `conductivity._conductive_component` is the four-case recurrence the certified
    # conductive curl calls three times; the lifted body below calls it with the
    # certified body's own arguments, so the branch table, the `tl.where` selection
    # and every store mask are the certified module's and not a transcription. Its
    # own docstring is where that case table is stated.
    from .conductivity import _conductive_component  # noqa: PLC0415
    # The names the two lifted curl bodies read: the shared forms and the record's
    # word offsets, the very objects the certified curls call.
    from .forms import (axis_present, bfast_advance, curl_form,  # noqa: PLC0415
                        curl_stencil, fold_ladder, record_in)
    from .kernels import RECORD_HALF_INTEGER, RECORD_PRESENCE  # noqa: PLC0415

    @triton.jit(do_not_specialize=["start_h"])
    def fused_constitutive_curl_H_to_D_conductive(
        ho0, ho1, ho2,                  # SCRATCH out: the stepped Hx, Hy, Hz
        wo0, wo1, wo2,                  # SCRATCH out: the stepped f_w_Hx/y/z
        hi0, hi1, hi2,                  # PRE-LAUNCH Hx, Hy, Hz            (read-only)
        wi0, wi1, wi2,                  # PRE-LAUNCH f_w_Hx/y/z            (read-only)
        b0, b1, b2,                     # Bx, By, Bz                       (read-only)
        f0, f1, f2,                     # curl targets: Dx, Dy, Dz             (in/out)
        u0, u1, u2,                     # curl auxiliaries: fu_Dx/y/z          (in/out)
        c0, c1, c2,                     # conductive history: f_cond_Dx/y/z    (in/out)
        cf0, cf1, cf2,                  # condfac per target               (read-only)
        ci0, ci1, ci2,                  # condinv per target               (read-only)
        m0, m1, m2,                     # conductive chunks (a placeholder where CMASK == 0)
        kmx, sinvx, kmy, sinvy, kmz, sinvz,   # kms/sinv per axis, INTEGER lattice
        kp0, kp1, kp2,                  # kps on each component's OWN axis, INTEGER
        cb,                             # chunk record: both halves' forms and presence
        nx, ny, nz, n_elem, dtdx, start_h,   # start_h: MEEP's start of H
        BACKWARD: tl.constexpr,
        BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
        COND0: tl.constexpr, COND1: tl.constexpr, COND2: tl.constexpr,
        CMASK0: tl.constexpr, CMASK1: tl.constexpr, CMASK2: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """``update_H`` + the CONDUCTIVE ``step_D``, one launch, scratch output.

        ``hi``/``wi``/``b`` are const for the whole dispatch and ``ho``/``wo`` are
        write-only scratch, so nothing written by the constitutive half is read by
        this launch; ``f``/``u``/``c`` step IN PLACE and that is safe by
        construction, because the curl reads and writes each at the program's OWN
        cell only.
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
        own0, own1, own2, src0, src1, src2, dw0, dw1, dw2 = _h_cell(
            i, j, k, live, hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
            kp0, kp1, kp2, kmx, kmy, kmz, cb, start_h, ny, nz)
        tl.store(ho0 + idx, own0, mask=live)
        tl.store(ho1 + idx, own1, mask=live)
        tl.store(ho2 + idx, own2, mask=live)
        # The history takes B on the accumulation's cells only (the auxiliary
        # contract); every other cell of the twin takes the pre-launch word, so the
        # rotation hands the engine an f_w_H that kept its word there and the twin is
        # written at every cell.
        tl.store(wo0 + idx, tl.where(dw0, src0, tl.load(
            wi0 + idx, mask=live & (dw0 == 0), other=0.0)), mask=live)
        tl.store(wo1 + idx, tl.where(dw1, src1, tl.load(
            wi1 + idx, mask=live & (dw1 == 0), other=0.0)), mask=live)
        tl.store(wo2 + idx, tl.where(dw2, src2, tl.load(
            wi2 + idx, mask=live & (dw2 == 0), other=0.0)), mask=live)

        # === the certified conductive step_D curl body begins here ===

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

        a = own0
        b = own1
        c = own2
        a_y = _h_tap(0, i, sj, k, vy, hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                     kp0, kp1, kp2, kmx, kmy, kmz, cb, start_h, ny, nz)
        a_z = _h_tap(0, i, j, sk, vz, hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                     kp0, kp1, kp2, kmx, kmy, kmz, cb, start_h, ny, nz)
        b_x = _h_tap(1, si, j, k, vx, hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                     kp0, kp1, kp2, kmx, kmy, kmz, cb, start_h, ny, nz)
        b_z = _h_tap(1, i, j, sk, vz, hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                     kp0, kp1, kp2, kmx, kmy, kmz, cb, start_h, ny, nz)
        c_x = _h_tap(2, si, j, k, vx, hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                     kp0, kp1, kp2, kmx, kmy, kmz, cb, start_h, ny, nz)
        c_y = _h_tap(2, i, sj, k, vy, hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                     kp0, kp1, kp2, kmx, kmy, kmz, cb, start_h, ny, nz)

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

        # --- the forms, per component ----------------------------------------------
        # dsig/dsigu follow vec.hpp's cycle_direction and are the same triple on
        # both sides: target 0 takes (y, z), target 1 (z, x), target 2 (x, y).
        km_x = tl.load(kmx + i, mask=live, other=0.0)
        si_x = tl.load(sinvx + i, mask=live, other=0.0)
        km_y = tl.load(kmy + j, mask=live, other=0.0)
        si_y = tl.load(sinvy + j, mask=live, other=0.0)
        km_z = tl.load(kmz + k, mask=live, other=0.0)
        si_z = tl.load(sinvz + k, mask=live, other=0.0)

        # The component's dsig and dsigu chunk memberships, on its own sub-lattice
        # (half-integer for every B target, integer for every D one): each record line
        # is a compile-time word index, chunk_map.record_index(half_integer, axis, 0).
        if BACKWARD:
            lattice = 0
        else:
            lattice = RECORD_HALF_INTEGER
        ds0 = record_in(cb, lattice + 4, j)
        du0 = record_in(cb, lattice + 8, k)
        ds1 = record_in(cb, lattice + 8, k)
        du1 = record_in(cb, lattice + 0, i)
        ds2 = record_in(cb, lattice + 0, i)
        du2 = record_in(cb, lattice + 4, j)

        _conductive_component(f0, u0, c0, cf0, ci0, m0, idx, live, curl0,
                              km_y, si_y, km_z, si_z, ds0, du0, COND0, CMASK0)
        _conductive_component(f1, u1, c1, cf1, ci1, m1, idx, live, curl1,
                              km_z, si_z, km_x, si_x, ds1, du1, COND1, CMASK1)
        _conductive_component(f2, u2, c2, cf2, ci2, m2, idx, live, curl2,
                              km_x, si_x, km_y, si_y, ds2, du2, COND2, CMASK2)

    @triton.jit(do_not_specialize=["start_h"])
    def fused_constitutive_curl_H_to_D_bfast(
        ho0, ho1, ho2,                  # SCRATCH out: the stepped Hx, Hy, Hz
        wo0, wo1, wo2,                  # SCRATCH out: the stepped f_w_Hx/y/z
        hi0, hi1, hi2,                  # PRE-LAUNCH Hx, Hy, Hz            (read-only)
        wi0, wi1, wi2,                  # PRE-LAUNCH f_w_Hx/y/z            (read-only)
        b0, b1, b2,                     # Bx, By, Bz                       (read-only)
        f0, f1, f2,                     # curl targets: Dx, Dy, Dz             (in/out)
        u0, u1, u2,                     # curl auxiliaries: fu_Dx/y/z          (in/out)
        s0, s1, s2,                     # BFAST state: f_bfast_Dx/y/z          (in/out)
        kmx, sinvx, kmy, sinvy, kmz, sinvz,   # kms/sinv per axis, INTEGER lattice
        kp0, kp1, kp2,                  # kps on each component's OWN axis, INTEGER
        cb,                             # chunk record: both halves' forms and presence
        nx, ny, nz, n_elem, dtdx, start_h,   # start_h: MEEP's start of H
        k1_0, k2_0, k1_1, k2_1, k1_2, k2_2,   # the six BFAST scalars
        BACKWARD: tl.constexpr,
        BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
        HAS_BFAST: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """``update_H`` + the BFAST ``step_D``, one launch, scratch output.

        Same contract as the conductive kernel above: ``hi``/``wi``/``b`` const,
        ``ho``/``wo`` write-only scratch, ``f``/``u``/``s`` stepped in place at the
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
        own0, own1, own2, src0, src1, src2, dw0, dw1, dw2 = _h_cell(
            i, j, k, live, hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
            kp0, kp1, kp2, kmx, kmy, kmz, cb, start_h, ny, nz)
        tl.store(ho0 + idx, own0, mask=live)
        tl.store(ho1 + idx, own1, mask=live)
        tl.store(ho2 + idx, own2, mask=live)
        # The history takes B on the accumulation's cells only (the auxiliary
        # contract); every other cell of the twin takes the pre-launch word, so the
        # rotation hands the engine an f_w_H that kept its word there and the twin is
        # written at every cell.
        tl.store(wo0 + idx, tl.where(dw0, src0, tl.load(
            wi0 + idx, mask=live & (dw0 == 0), other=0.0)), mask=live)
        tl.store(wo1 + idx, tl.where(dw1, src1, tl.load(
            wi1 + idx, mask=live & (dw1 == 0), other=0.0)), mask=live)
        tl.store(wo2 + idx, tl.where(dw2, src2, tl.load(
            wi2 + idx, mask=live & (dw2 == 0), other=0.0)), mask=live)

        # === the certified BFAST step_D curl body begins here ===

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

        a = own0
        b = own1
        c = own2
        a_y = _h_tap(0, i, sj, k, vy, hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                     kp0, kp1, kp2, kmx, kmy, kmz, cb, start_h, ny, nz)
        a_z = _h_tap(0, i, j, sk, vz, hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                     kp0, kp1, kp2, kmx, kmy, kmz, cb, start_h, ny, nz)
        b_x = _h_tap(1, si, j, k, vx, hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                     kp0, kp1, kp2, kmx, kmy, kmz, cb, start_h, ny, nz)
        b_z = _h_tap(1, i, j, sk, vz, hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                     kp0, kp1, kp2, kmx, kmy, kmz, cb, start_h, ny, nz)
        c_x = _h_tap(2, si, j, k, vx, hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                     kp0, kp1, kp2, kmx, kmy, kmz, cb, start_h, ny, nz)
        c_y = _h_tap(2, i, sj, k, vy, hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                     kp0, kp1, kp2, kmx, kmy, kmz, cb, start_h, ny, nz)

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

        # --- the BFAST pass (stepping._bfast_term, then stepping._apply_folded_increment)
        # MEEP's STEP_BFAST on each component right after its STEP_CURL
        # (step_db.cpp:124-142), on the SAME gathered partners, applied to the registers
        # the forms returned (forms.bfast_advance: the state's two roundings, the
        # one-partner forms by the record's presence bits, MEEP's plain branch on cells
        # in neither the dsig nor the dsigu chunk). The increment is masked to +0, and
        # the state kept, on the cells the curl's own mask drops (MEEP writes F inside
        # its owned loop only); the BFAST ladder subtracts it (forms.fold_ladder). No
        # dtdx anywhere: MEEP passes it to step_bfast and never reads it.
        if HAS_BFAST:
            ds0 = record_in(cb, lattice + 4, j)
            ds1 = record_in(cb, lattice + 8, k)
            ds2 = record_in(cb, lattice + 0, i)
            st0 = tl.load(s0 + idx, mask=live, other=0.0)
            st1 = tl.load(s1 + idx, mask=live, other=0.0)
            st2 = tl.load(s2 + idx, mask=live, other=0.0)
            bf0, q0 = bfast_advance(st0, c_y, c, b_z, b, k1_0, k2_0, py, pz, ds0 | du0)
            bf1, q1 = bfast_advance(st1, a_z, a, c_x, c, k1_1, k2_1, pz, px, ds1 | du1)
            bf2, q2 = bfast_advance(st2, b_x, b, a_y, a, k1_2, k2_2, px, py, ds2 | du2)
            if BACKWARD:
                if BCY == METALLIC:
                    q0 = tl.where(at_y, 0.0, q0)
                    bf0 = tl.where(at_y, st0, bf0)
                if BCZ == METALLIC:
                    q0 = tl.where(at_z, 0.0, q0)
                    bf0 = tl.where(at_z, st0, bf0)
                if BCX == METALLIC:
                    q1 = tl.where(at_x, 0.0, q1)
                    bf1 = tl.where(at_x, st1, bf1)
                if BCZ == METALLIC:
                    q1 = tl.where(at_z, 0.0, q1)
                    bf1 = tl.where(at_z, st1, bf1)
                if BCX == METALLIC:
                    q2 = tl.where(at_x, 0.0, q2)
                    bf2 = tl.where(at_x, st2, bf2)
                if BCY == METALLIC:
                    q2 = tl.where(at_y, 0.0, q2)
                    bf2 = tl.where(at_y, st2, bf2)
            else:
                if BCX == METALLIC:
                    q0 = tl.where(at_x, 0.0, q0)
                    bf0 = tl.where(at_x, st0, bf0)
                if BCY == METALLIC:
                    q1 = tl.where(at_y, 0.0, q1)
                    bf1 = tl.where(at_y, st1, bf1)
                if BCZ == METALLIC:
                    q2 = tl.where(at_z, 0.0, q2)
                    bf2 = tl.where(at_z, st2, bf2)
            tl.store(s0 + idx, bf0, mask=live)
            tl.store(s1 + idx, bf1, mask=live)
            tl.store(s2 + idx, bf2, mask=live)
            v0, n0 = fold_ladder(v0, n0, q0, si_y, si_z, du0)
            v1, n1 = fold_ladder(v1, n1, q1, si_z, si_x, du1)
            v2, n2 = fold_ladder(v2, n2, q2, si_x, si_y, du2)

        # fu is written on the recurrence's cells ONLY: outside them the dsig-only form
        # reads and writes the field alone and fu keeps its word (the auxiliary contract).
        tl.store(u0 + idx, n0, mask=live & du0)
        tl.store(u1 + idx, n1, mask=live & du1)
        tl.store(u2 + idx, n2, mask=live & du2)
        tl.store(f0 + idx, v0, mask=live)
        tl.store(f1 + idx, v1, mask=live)
        tl.store(f2 + idx, v2, mask=live)


def fused_kernel_for(variant: str) -> Any:
    """One variant's shipped kernel object, or a refusal naming the missing import."""
    _check_variant(variant)
    if triton is None:  # pragma: no cover - the laptop path
        raise ImportError(
            "the conductive/BFAST H->D pair needs Triton to launch; the predicate, "
            "the lift checks and the plan builder answer without it "
            f"({_TRITON_IMPORT_ERROR})")
    return globals()[FUSED_FUNCTIONS[variant]]


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def resolve_variant(fields: Any, pml: Any) -> Tuple[Optional[str], Tuple[str, ...]]:
    """Which variant a run resolves to, or ``(None, reasons)``.

    THE RUN DECIDES, NOT THE CALLER. A configuration is BFAST when the BFAST curl
    predicate admits it at ``step_D`` and conductive when the conductive one does;
    both admitting is a configuration neither certified curl claims alone and is
    refused BY NAME rather than resolved by branch order, because which of the two
    the array path would have run is then not a fact this module may assume.
    """
    conductive = _conductivity.conductive_pml_curl_coverage(
        fields, pml, CURL_SUB_STEP)
    bfast = _bfast.bfast_pml_curl_coverage(fields, pml, CURL_SUB_STEP)
    if conductive.covered and bfast.covered:
        return None, (
            "both the conductive and the BFAST step_D predicates admit this run: "
            "which certified curl the array path would have launched is not a fact "
            "this product may decide by branch order, and no corpus row on the "
            "board's two cells is both",)
    if conductive.covered:
        return "conductive", ()
    if bfast.covered:
        return "bfast", ()
    return None, tuple(
        [f"conductive curl half: {reason}" for reason in conductive.reasons]
        + [f"BFAST curl half: {reason}" for reason in bfast.reasons])


def conductive_bfast_fused_hd_pair_coverage(
        fields: Any, pml: Any, sources: Any = None,
        variant: Optional[str] = None) -> "_coverage.Coverage":
    """May ONE launch span ``update_H`` -> the electric withdraw -> ``step_D``?

    A conjunction of the two halves' OWN certified predicates plus the seam clauses.
    Nothing is weakened: a configuration either half refuses is refused here with that
    half's reasons, prefixed so a reader can tell which side said it.

    THE ORDER IS THE DRIVER'S. The constitutive half is asked FIRST because it runs
    first (driver.py:3311), so the first refusal a reader sees names the half the
    driver would have reached first.

    ``variant`` is for a harness that built one deliberately; a real caller passes
    ``None`` and :func:`resolve_variant` reads it off the run.
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

    # -- the constitutive half, in driver order --------------------------------
    if variant == "bfast":
        constitutive = _bfast.bfast_run_constitutive_coverage(
            fields, pml, CONSTITUTIVE_SIDE)
    else:
        constitutive = _coverage.constitutive_coverage(
            fields, pml, CONSTITUTIVE_SIDE)
    if not constitutive.covered:
        reasons.extend(f"constitutive half: {reason}"
                       for reason in constitutive.reasons)

    # -- the curl half ---------------------------------------------------------
    if variant == "bfast":
        curl = _bfast.bfast_pml_curl_coverage(fields, pml, CURL_SUB_STEP)
    else:
        curl = _conductivity.conductive_pml_curl_coverage(
            fields, pml, CURL_SUB_STEP)
    if not curl.covered:
        reasons.extend(f"curl half: {reason}" for reason in curl.reasons)

    # THE SEAM'S ONE PASS. Nothing is INJECTED between the two consults, so
    # `deposit_repair` is not consulted at all. What IS between them is the electric
    # integrated-source withdraw (driver.py:3313-3314), and `withdraw_hoist` owns it.
    # IGNORANCE IS NEVER AN EMPTY SET: `Fields` does not hold the source list, so a
    # predicate that inferred "no withdraw stands" from not being told would be the
    # over-covering refusal this clause exists to prevent.
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

    # THE FOLD, RESTATED BY NAME. Both halves already refuse a mirror plane, and on
    # THIS seam the reason is not the D->E and B->H pairs': neither fill runs between
    # these two consults, so the H->D seam is fill-FREE on a folded grid. What refuses
    # a fold here is that this product implements arms a folded run does not select.
    mirrored = getattr(grid, "is_mirrored", None)
    for axis in range(3):
        if callable(mirrored) and bool(mirrored(axis)):
            reasons.append(
                f"axis {axis} is folded: this product implements the `{VARIANT_ARMS[variant][0]}` "
                f"update_H and `{VARIANT_ARMS[variant][1]}` step_D arms, and a folded "
                f"run selects folded arms on both slots")

    # THE ROTATION IS THE PRODUCT'S OWN INVARIANT.
    for name in ROTATED:
        if getattr(fields, name, None) is None:
            reasons.append(
                f"{name} is not allocated; this weld rotates it against a plan-owned "
                f"scratch twin after every launch")
    for name in IN_PLACE + CONSTITUTIVE_SOURCES:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")
    if variant == "bfast":
        for name in BFAST_STATES:
            if getattr(fields, name, None) is None:
                reasons.append(
                    f"{name} is not allocated; the BFAST curl steps it in place")
    return _coverage.Coverage(not reasons, tuple(dict.fromkeys(reasons)))


def explain_conductive_bfast_fused_hd_pair(
        fields: Any, pml: Any, sources: Any = None) -> Dict[str, Any]:
    """The verdict under the name a report reads, WITH the variant it resolved to.

    One home for "which arm pair did this product absorb", which is the question a
    two-cell family must keep answerable from its own record.
    """
    variant, why = resolve_variant(fields, pml)
    verdict = conductive_bfast_fused_hd_pair_coverage(
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

class ConductiveBfastFusedHdPairPlan(_plain.StartsH, ScratchWeldPairPlan):
    """ONE allocation-free launch for ``update_H`` and one of the two ``step_D`` tails.

    THE SIX ROTATING VOLUMES ARE RESOLVED PER LAUNCH, never cached: the plan owns a
    twin of each, reads the engine's CURRENT attribute immediately before the launch
    to decide which of the pair is live, and moves the engine's references only after
    the launch returns. A pointer captured at plan time would be one rotation stale --
    and stale in a way that still computes.

    ``__init__`` REFUSES ALIASING between anything this launch writes (the six twins,
    ``D``, ``fu_D`` and the variant's in-place state) and anything it reads (``B``,
    the coefficient vectors, the conductive factors), and among the written volumes
    themselves. The base class additionally refuses a scratch buffer that IS its own
    live volume, which would be the in-place weld this design exists to avoid.
    """

    __slots__ = ("variant", "dtdx", "backward", "bc", "cond", "cmask", "has_bfast",
                 "ks", "_b", "_targets", "_aux", "_state", "_condfac", "_condinv",
                 "_chunks", "_curl_coeff", "_kps", "_kernel", "_pointer", "_h_state")

    replaces = REPLACES

    def __init__(self, variant: str, shape: Sequence[int], dtdx: float,
                 codes: Sequence[int], block: int, fields: Any,
                 twins: Dict[str, Any], flux: Sequence[Any],
                 targets: Sequence[Any], auxiliaries: Sequence[Any],
                 state: Sequence[Any], curl_coefficients: Sequence[Any],
                 constitutive_coefficients: Sequence[Any],
                 condfac: Sequence[Any] = (), condinv: Sequence[Any] = (),
                 cond: Sequence[Any] = (0, 0, 0), ks: Sequence[float] = (),
                 has_bfast: int = 1, kernel: Any = None,
                 num_warps: Optional[int] = 1, *, chunk_bounds: Any,
                 conductive_chunks: Sequence[Any] = (None, None, None),
                 h_state: Any = None) -> None:
        from .launch import CupyPointer, _flat  # noqa: PLC0415

        super().__init__(FAMILY, fields, twins, ROTATED, shape, block, REPLACES,
                         num_warps=num_warps, chunk_bounds=chunk_bounds)
        self.variant = _check_variant(variant)
        self.dtdx = float(dtdx)
        self.backward = BACKWARD
        self.bc = tuple(int(code) for code in codes)
        if len(self.bc) != 3:
            raise ValueError("this plan needs three boundary codes")
        self._pointer = CupyPointer
        self._b = tuple(CupyPointer(a) for a in flux)
        self._targets = tuple(CupyPointer(a) for a in targets)
        self._aux = tuple(CupyPointer(a) for a in auxiliaries)
        # The lossless placeholder policy is the CERTIFIED plan's, called rather than
        # re-derived: a component the run leaves lossless binds its own target, never
        # dereferenced because COND{n} is 0 for it.
        self._state = tuple(CupyPointer(a if a is not None else b)
                            for a, b in zip(state, targets))
        self._condfac = tuple(CupyPointer(a if a is not None else b)
                              for a, b in zip(tuple(condfac) or (None,) * 3, targets))
        self._condinv = tuple(CupyPointer(a if a is not None else b)
                              for a, b in zip(tuple(condinv) or (None,) * 3, targets))
        self._curl_coeff = tuple(CupyPointer(_flat(a)) for a in curl_coefficients)
        self._kps = tuple(CupyPointer(_flat(a)) for a in constitutive_coefficients)
        if len(self._curl_coeff) != 6 or len(self._kps) != 3:
            raise ValueError(
                "this plan binds six curl coefficient vectors (kms/sinv per axis) and "
                "three constitutive kps; the constitutive kms are the curl's own "
                "three, shared because both halves sit on the integer sub-lattice")
        self.cond = tuple(1 if flag else 0 for flag in cond)
        # The certified conductive plan's conductive-chunk binding, called rather than
        # re-derived: a conductive target whose chunks do not all keep the
        # conductivity binds its volume as uint8 and compiles CMASK 1; every other
        # target binds its own field as the placeholder, never dereferenced.
        chunks = tuple(conductive_chunks)
        if len(chunks) != 3:
            raise ValueError("conductive_chunks names one volume (or None) per target")
        self.cmask = tuple(1 if (flag and volume is not None) else 0
                           for flag, volume in zip(self.cond, chunks))
        self._chunks = tuple(
            CupyPointer(_conductivity._chunk_volume(  # noqa: SLF001
                volume, self.shape, target) if mask else target)
            for mask, volume, target in zip(self.cmask, chunks, targets))
        if self.variant == "conductive" and not any(self.cond):
            raise ValueError(
                "a conductive weld with no conductive component is fused_hd_pair's "
                "configuration; build that product instead")
        self.ks = tuple(float(value) for value in ks)
        if self.variant == "bfast" and len(self.ks) != 6:
            raise ValueError("the BFAST variant binds six k1/k2 scalars")
        self.has_bfast = int(has_bfast)
        self._check_aliases(twins, flux, targets, auxiliaries, state,
                            curl_coefficients, constitutive_coefficients)
        self._kernel = kernel
        # MEEP's start of H, on the plain product's terms (fused_hd_pair.StartsH).
        self._h_state = h_state

    def _check_aliases(self, twins, flux, targets, auxiliaries, state,
                       curl_coefficients, constitutive_coefficients) -> None:
        """No written volume may share an allocation with a read one, or another.

        The whole design is that nothing written by the constitutive half is read; an
        aliased pair would put the schedule back into the answer, and would do it
        through a binding rather than through the kernel text.
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

        in_place = IN_PLACE_BY_VARIANT[self.variant]
        outputs: Dict[int, str] = {}
        named = [(name, twins[name]) for name in ROTATED]
        named += [(in_place[index], array) for index, array in enumerate(targets)]
        named += [(in_place[3 + index], array)
                  for index, array in enumerate(auxiliaries)]
        for index, array in enumerate(state):
            if array is not None:
                named.append((in_place[6 + index], array))
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
        if self.variant == "conductive":
            kernel[self._grid](
                *scratch, *prior, *self._b, *self._targets, *self._aux,
                *self._state, *self._condfac, *self._condinv, *self._chunks,
                *self._curl_coeff, *self._kps, self._cb,
                nx, ny, nz, self.n_elem, self.dtdx, self.start_h(),
                BACKWARD=self.backward,
                BCX=self.bc[0], BCY=self.bc[1], BCZ=self.bc[2],
                COND0=self.cond[0], COND1=self.cond[1], COND2=self.cond[2],
                CMASK0=self.cmask[0], CMASK1=self.cmask[1], CMASK2=self.cmask[2],
                BLOCK=self.block,
                enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
                **extra,
            )
        else:
            kernel[self._grid](
                *scratch, *prior, *self._b, *self._targets, *self._aux,
                *self._state, *self._curl_coeff, *self._kps, self._cb,
                nx, ny, nz, self.n_elem, self.dtdx, self.start_h(), *self.ks,
                BACKWARD=self.backward,
                BCX=self.bc[0], BCY=self.bc[1], BCZ=self.bc[2],
                HAS_BFAST=self.has_bfast,
                BLOCK=self.block,
                enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
                **extra,
            )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"ConductiveBfastFusedHdPairPlan({self.variant}, shape={self.shape}, "
                f"bc={self.bc}, block={self.block}, num_warps={self.num_warps})")


def _sub_lattice_suffixes() -> Tuple[str, str]:
    """``(curl suffix, constitutive suffix)``, READ from the shipped tables.

    Their equality is the whole premise of the shared ``kms`` group, so it is
    ASSERTED rather than assumed: if either table moved, sharing the group would bind
    one half's coefficients to the other half's lattice -- a converged, smooth,
    half-cell-wrong absorber profile rather than a failure.
    """
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


def plan_conductive_bfast_fused_hd_pair(
        fields: Any, pml: Any, sources: Any = None, block: Optional[int] = None,
        num_warps: Optional[int] = 1, kernel: Any = None
) -> Optional[ConductiveBfastFusedHdPairPlan]:
    """Build the plan from the engine's own objects, or ``None`` when refused.

    ``None`` is the only refusal, for the reason :func:`.launch.plan_pml_curl` gives:
    a configuration this kernel does not carry must fall back to the array path, never
    raise into a caller that would otherwise have stepped correctly.

    ``kernel`` is the mutation seam. The gate compiles a deliberately broken copy of
    the shipped kernel and hands it here; dropping the argument is not a silent
    slowdown, it is a silent DISARMING.
    """
    from .launch import chunk_bounds_for  # noqa: PLC0415
    variant, _why = resolve_variant(fields, pml)
    if variant is None:
        return None
    if not conductive_bfast_fused_hd_pair_coverage(
            fields, pml, sources, variant=variant).covered:
        return None
    suffix, constitutive_suffix = _sub_lattice_suffixes()
    grid = fields.grid
    kinds = _coverage._boundary_kinds(grid, pml)  # noqa: SLF001 - shared resolver
    codes = [1 if kind == "metallic" else 0 for kind in kinds]
    curl_coefficients = [getattr(pml, f"{stem}_{axis}{suffix}")
                         for axis in "xyz" for stem in ("kms", "sinv")]
    kps = [getattr(pml, f"kps_{axis}{constitutive_suffix}") for axis in "xyz"]
    common = dict(
        shape=grid.shape, dtdx=grid.dt / grid.dx, codes=codes,
        block=DEFAULT_BLOCK if block is None else block, fields=fields,
        twins=twin_table(fields, ROTATED),
        flux=[getattr(fields, name) for name in CONSTITUTIVE_SOURCES],
        targets=[getattr(fields, name) for name in IN_PLACE[:3]],
        auxiliaries=[getattr(fields, name) for name in IN_PLACE[3:]],
        curl_coefficients=curl_coefficients, constitutive_coefficients=kps,
        kernel=kernel, num_warps=num_warps,
        chunk_bounds=chunk_bounds_for(fields, pml, None), h_state=fields)
    if variant == "conductive":
        targets = _conductivity.CONDUCTIVE_SUB_STEPS[CURL_SUB_STEP]
        return ConductiveBfastFusedHdPairPlan(
            "conductive",
            state=[getattr(fields, "f_cond_" + name) for name in targets],
            condfac=[fields.condfac_for(name) for name in targets],
            condinv=[fields.condinv_for(name) for name in targets],
            cond=_conductivity.conductive_targets(fields, CURL_SUB_STEP),
            conductive_chunks=_conductivity.conductive_chunk_volumes(
                fields, CURL_SUB_STEP, None, pml),
            **common)
    invariant = tuple(grid.is_invariant(axis) for axis in range(3))
    ks = _bfast.bfast_curl_coefficients(
        grid.bfast_scaled_k, invariant, magnetic=(CURL_SUB_STEP == "step_B"))
    return ConductiveBfastFusedHdPairPlan(
        "bfast",
        state=[getattr(fields, name) for name in BFAST_STATES],
        ks=ks, **common)


def plan_conductive_bfast_fused_hd_pair_from_arrays(
        variant: str, arrays: Dict[str, Any], flat: Dict[str, Any], dtdx: float,
        codes: Sequence[int], fields: Any, block: Optional[int] = None,
        kernel: Any = None, num_warps: Optional[int] = 1,
        cond: Sequence[Any] = (1, 1, 1), ks: Sequence[float] = (),
        has_bfast: int = 1, *, chunk_bounds: Any) -> ConductiveBfastFusedHdPairPlan:
    """Build from bare device arrays -- the gate's route.

    No coverage predicate runs: the caller is a harness that constructed the
    configuration deliberately, including the deliberately wrong ones. ``fields`` is
    the object whose attributes the ROTATION swaps, so a gate hands a small namespace
    holding the six rotating volumes under the engine's own names and exercises the
    rotation the engine would get. ``arrays`` supplies the twins under
    ``scratch_Hx`` ...
    """
    _check_variant(variant)
    shape = tuple(int(n) for n in arrays[IN_PLACE[0]].shape)
    twins = {name: arrays["scratch_" + name] for name in ROTATED}
    common = dict(
        shape=shape, dtdx=dtdx, codes=codes,
        block=DEFAULT_BLOCK if block is None else block, fields=fields, twins=twins,
        flux=[arrays[name] for name in CONSTITUTIVE_SOURCES],
        targets=[arrays[name] for name in IN_PLACE[:3]],
        auxiliaries=[arrays[name] for name in IN_PLACE[3:]],
        curl_coefficients=[flat[f"{stem}_{axis}"]
                           for axis in "xyz" for stem in ("kms", "sinv")],
        constitutive_coefficients=[flat[f"kps_{axis}"] for axis in "xyz"],
        kernel=kernel, num_warps=num_warps, chunk_bounds=chunk_bounds)
    if variant == "conductive":
        targets = _conductivity.CONDUCTIVE_SUB_STEPS[CURL_SUB_STEP]
        return ConductiveBfastFusedHdPairPlan(
            "conductive",
            state=[arrays.get("f_cond_" + name) for name in targets],
            condfac=[arrays.get("condfac_" + name) for name in targets],
            condinv=[arrays.get("condinv_" + name) for name in targets],
            cond=cond,
            conductive_chunks=tuple(arrays.get("conductive_chunks_" + name)
                                    for name in targets),
            **common)
    return ConductiveBfastFusedHdPairPlan(
        "bfast", state=[arrays[name] for name in BFAST_STATES],
        ks=ks, has_bfast=has_bfast, **common)
