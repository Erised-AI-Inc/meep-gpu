"""The H->D weld on this backend: ``update_H`` welded into ``step_D``, one launch.

THE FOURTH FUSION SEAM, and the first on this backend whose two halves sit in the
order constitutive-then-curl. The driver runs

    ``update_H`` (driver.py:3311) -> the electric integrated-source withdraw
    (:3313-3314) -> ``step_D`` (:3315)

and exactly one statement sits between the two consults: ``for source in electric:
getattr(source, "withdraw", _no_withdraw)(self.fields)``. Nothing is INJECTED here --
the electric injection is one seam later (:3317-3322) and the magnetic one is one
seam earlier (:3283-3284) -- so :mod:`..deposit_repair` has nothing to say about this
seam and :mod:`..withdraw_hoist` is the module that does.

=============================================================================
THE SHAPE: SCRATCH OUTPUT AND FOREIGN-CELL RECOMPUTE
=============================================================================

``step_D``'s curl reads ``H`` at the thread's own cell AND at its three BACKWARD
neighbours, and ``update_H`` writes ``H``. Welded in place, a program would read
``H`` at a neighbour cell at a moment decided by the block schedule -- the hazard
this backend's own ``results/triton_fused_offdiag_electric_2026-08-20`` measured on
the MIRROR-IMAGE D->E seam (42 of 60 subject cases divergent, and the differing-word
count moving from 4384 at BLOCK=64 to 0 at BLOCK=1024) -- and it would read
``f_w_H`` the same way, which is easy to miss: on this side the newly written
``f_w_H`` IS ``B`` exactly, so an in-place write hands a racing neighbour ``B``
where it needs ``B_prev``.

Both premises are removed the way :mod:`.offdiag_fused_electric_pair` and
:mod:`.folded_offdiag_fused_electric_pair` remove them on that seam, and the way
``metal_kernels.fused_hd_pair`` and ``cuda_kernels.offdiag_stencil_weld`` remove them
on theirs:

* **the constitutive half writes nothing in place.** ``H_new`` and ``f_w_H_new`` go
  to LAUNCH-LOCAL SCRATCH, so ``H``, ``f_w_H`` and ``B`` are pre-launch state for the
  whole dispatch and no program can observe another program's store;
* **the foreign read is not a read of another program's output.** It is a RECOMPUTE
  from that same unwritten state, through :func:`_h_cell` -- the certified
  ``constitutive_step`` body for side ``H``, whole, evaluated at an arbitrary cell. A
  pure function of unwritten memory has no schedule to depend on.

:class:`FusedHdPairPlan` then ROTATES the ``H``/``f_w_H`` references, which is
:class:`.offdiag_scratch_weld.ScratchWeldPairPlan`'s certified choreography.

WHY THE RECOMPUTE IS EXACT RATHER THAN CLOSE. ``update_H`` is POINTWISE: per
component it reads ``H``, ``f_w_H`` and ``B`` at ONE cell and writes the same cell
(stepping.py:907-923 through ``_apply_constitutive_pml``, stepping.py:2112-2143). So
the stepped value at any cell is a function of state this launch does not write, and
evaluating it twice gives the same bits by construction rather than by a tolerance.
That is CHEAPER than the D->E welds' recompute, whose recomputed half is itself a
curl over six neighbours.

WHY THE COEFFICIENT INDEX NEEDS NO THEOREM. :func:`_h_cell` takes ``(i, j, k)`` and
indexes ``kp*``/``km*`` at THOSE coordinates, exactly as the certified body indexes
them at the program's own -- so the recompute is the certified arithmetic at the
recomputed cell and not an approximation of it.

=============================================================================
ONE SUB-LATTICE, ONE COEFFICIENT GROUP
=============================================================================

Both halves sit on the same Yee sub-lattice -- ``launch.SUB_STEPS['step_D']['suffix']``
is ``''`` and ``coverage.CONSTITUTIVE_SIDES['H']['half_integer']`` is False -- so the
curl's ``kms_x/_y/_z`` and the constitutive's ``kms_x/_y/_z`` ARE THE SAME THREE
VOLUMES and the signature binds them once. Both suffixes are READ from the shipped
tables and their equality is ASSERTED in :func:`plan_fused_hd_pair` rather than
assumed: binding the HALF-INTEGER set instead compiles, launches, converges, and is a
half-cell error in the absorber profile.

=============================================================================
WHAT SITS IN THE SEAM, AND WHY THIS PRODUCT REFUSES THE WITHDRAW ROWS
=============================================================================

:data:`CARRIES_DEPOSIT_REPAIR` is False and it is a FACT rather than a decision:
nothing is injected between the two consults, so there is no deposit to bracket. The
pass that IS there is the electric withdraw, and :mod:`..withdraw_hoist` is its
contract -- one slot, not two; a hoist, not a bracket.

:data:`HOISTS_THE_WITHDRAW` is False IN THIS ROUND, and the flag and the wiring move
together exactly as ``CARRIES_DEPOSIT_REPAIR``'s do. The only wiring that performs
the hoist is ``launch._install_fused_pair``'s ``withdraw_hoist.SEAM`` branch, which
wraps the built plan in a :class:`..withdraw_hoist.LeadingWithdrawPlan`; this product
declares :data:`INSTALLABLE` False, so that branch is unreachable for it and NOTHING
would perform the withdraw before its launch. A product that declared True without
that wrapper would let the fused launch read a ``D`` still holding the previous
step's standing dipole and report success -- the exact failure ``withdraw_hoist``
exists to prevent. So the predicate refuses, BY NAME, every row whose electric
withdraw does work, which is what the three boards already file under
``withdraw_seam``.

WHAT THAT COSTS, MEASURED rather than estimated
(``parity/meep_gpu/results/h_to_d_seam_2026-09-04/h_to_d_seam.jsonl``, joined on the
Triton arms): the ``(update_H ordinary -> step_D PML)`` cell is **49 rows**, of which
**47** are ``buildable_not_built`` and **2** are ``withdraw_seam`` --
``examples:differential_cross_section.py`` and
``tests:TestIntegratedSource.test_integrated_source``. This predicate therefore
reaches **47 of the cell's 49**, and what would raise it to 49 is a package of two
things landed together: :data:`INSTALLABLE` True, and a device gate leg that drives a
complete step on a row with a live INTEGRATED electric source with the hoist in place
and requires both the un-hoisted launch and the ``after_step_D`` placement to diverge.

=============================================================================
IT IS NOT INSTALLED, AND THE REASON HAS TWO HALVES
=============================================================================

:data:`INSTALLABLE` is False and :data:`INSTALLABLE_REASON` carries both halves.

THE FIRST is that the composer is never offered this product at all: routing it
through ``launch._install_certified_fused_products`` costs a LABEL, and on this
backend a label ``plan_step`` can write must also be declared in the dispatcher's own
tables -- outside this package by the one-way rule it keeps, and outside the round
that built this product. The flag is what keeps a later edit that lands those lines
from installing the product before a composition gate has driven it end to end.

THE SECOND is the arbitration, and it is MEASURED PER ROW rather than asserted. Over
the driver's ``step_B - update_H - step_D - update_E`` slot path launches are
``4 - (installed pairs)`` and a product spanning ``update_H``/``step_D`` takes one
slot from EACH neighbour, so installing it is a LOSS where both neighbouring pairs
install, a TIE where exactly one does, and a GAIN where NEITHER does. WHICH OF THE
THREE A ROW IS, IS NOT UNIFORM OVER THIS CELL: the rows carrying an off-diagonal
permittivity leave both neighbours uninstalled, and installing this product there
would save a launch. The gate's lift leg files every driven corpus row under
``arbitration_over_the_driven_rows`` off that row's OWN composer slot table; the
counts live in the artifact and are not copied here.

On the rows where a neighbouring pair DOES hold ``update_H``, what keeps it there is
not this flag either: it is ``launch._neighbouring_seam_claimant``'s end-edge guard --
a B->H span is never asked whether a neighbour claims it, and the degrees are computed
from ``CERTIFIED_FUSED_PAIR_SEAMS`` rather than spelled -- together with
``_pair_may_absorb`` reading the live ``selected`` after those pairs install first
(the H->D row is APPENDED LAST to that table for exactly this reason). Only a
four-slot ``step_B -> update_H -> step_D -> update_E`` weld is additive on EVERY row,
and it is not built here.

What this product IS worth is stated plainly, because no artifact here may imply
otherwise: the priced predicate gap closed under the rule *close all fusion
gaps regardless*, and a certified half of the only span that could ever pay. **No
timing exists for this shape and none is licensed by anything in this module.** The
fused route does MORE memory traffic than the two singles for the same step-level
launch count; whether the saved launch and the saved ``H`` write-then-read round trip
pay for three extra pointwise constitutive evaluations per component per program is a
hypothesis.

SERVED ON A FUSION BOARD IS PREDICATE ADMISSION, by that board's own definition. A
released product credits its admitted seam-instances while executing NOWHERE, and the
record and the board prose say exactly that: this product is in no
``fastpath.RELEASED_FUSED_ARMS`` envelope and ``fastpath`` never plans it.

=============================================================================
WHAT THIS FAMILY DOES NOT CARRY
=============================================================================

ONE ARM PAIR ONLY: ``(ordinary -> PML)``. The ``(folded -> folded PML)`` cell is 78
rows and the largest on this backend, and it is a SECOND PRODUCT rather than a
widening of this one -- this backend expresses the fold as a separate arm family with
its own ghost map in :mod:`.symmetry` and :mod:`.folded_complex`, and whether a folded
H->D halo needs cells beyond the three backward neighbours is unestablished and is a
probe, not a guess. The nonlinear cell (2 rows) is likewise NOT widened into here:
this backend's ``nonlinear run`` arms are their own family and admitting them would
be a text identity nothing in this round measured. Every other configuration --
complex storage, cylindrical, beta, BFAST, conductive, an inactive absorber -- is
refused through the two halves' own certified predicates, which this one conjoins
without weakening.

NOT WIRED INTO DISPATCH. ``fastpath`` plans only the arms ``launch.plan_step``
selects, and this product is offered to the certified-product seam loop where
``_declared_uninstallable`` refuses it by name on every configuration.

Import contract: importable WITHOUT Triton -- the predicate, the lift checks and the
plan builder (to ``None``) must answer on the laptop that is the merge bar.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import ast
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import coverage as _coverage
from .. import withdraw_hoist as _withdraw_hoist
from .offdiag_scratch_weld import ScratchWeldPairPlan, twin_table

#: The family name, as the board, the battery and the weld record spell it.
FAMILY = "fused_hd_pair"

#: The sub-step slot this product STARTS at -- the first half of the seam, in the
#: driver's own order, so a refusal is named on the slot the driver reaches first.
SLOT = "update_H"

#: The driver call sites ONE launch of this plan performs, in driver order
#: (driver.py:3311, :3315). Declared rather than inferred from the slot name.
#:
#: NOTHING BETWEEN THEM IS CARRIED, and there is only one thing there to carry: the
#: electric withdraw (:3313-3314), which :data:`HOISTS_THE_WITHDRAW` declines in this
#: round. It is not in this tuple because the product does not perform it.
REPLACES: Tuple[str, ...] = ("update_H", "step_D")

#: The seam name ``launch.CERTIFIED_FUSED_PAIR_SEAMS`` files this row under, and the
#: module that owns its in-seam pass. Spelled through :mod:`..withdraw_hoist` rather
#: than as a literal so the two cannot drift.
SEAM: str = _withdraw_hoist.SEAM

#: The constitutive side and the curl sub-step, in driver order.
CONSTITUTIVE_SIDE = "H"
CURL_SUB_STEP = "step_D"

#: The two arms this product implements, on ``(update_H, step_D)`` -- the pair
#: ``launch.CERTIFIED_FUSED_PAIR_ARMS`` binds, read off the two predicates this
#: module's own coverage conjoins.
ARMS: Tuple[str, str] = ("ordinary", "PML")

#: The six volumes one launch ROTATES: the stored magnetic field and its split-field
#: history. Order is the launch's argument order and is not negotiable -- the plan
#: binds every scratch then every pre-launch buffer by exactly this sequence.
ROTATED: Tuple[str, ...] = ("Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz")

#: The volumes the curl half steps IN PLACE, and it is safe by construction: the curl
#: reads and writes them at the PROGRAM'S OWN CELL only (``f0 + idx``, ``u0 + idx``),
#: so no program reads a displacement another program wrote.
IN_PLACE: Tuple[str, ...] = ("Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz")

#: The constitutive half's source volumes: B, read const for the whole dispatch.
CONSTITUTIVE_SOURCES: Tuple[str, ...] = ("Bx", "By", "Bz")

#: ``SUB_STEPS['step_D']['backward']``, restated as a declaration a test pins against
#: ``launch.SUB_STEPS`` rather than a literal buried in the builder.
BACKWARD = 1

#: Elements per program -- restated from ``kernels.DEFAULT_BLOCK`` (which needs Triton
#: to import, and the predicate and the builders must answer on the laptop that is
#: this package's merge bar). The precedent is :mod:`.offdiag_update_e`'s own restated
#: copy, and like that one it is PINNED against the certified module's source by a
#: laptop test rather than trusted.
DEFAULT_BLOCK = 256

#: Does this product bracket its launch with the DEPOSIT repair? NO, and it is a fact
#: about the driver rather than a choice: nothing is injected between the ``update_H``
#: consult (driver.py:3311) and the ``step_D`` consult (:3315), so there is no deposit
#: in this seam to carry across the launch. This predicate therefore does NOT consult
#: ``deposit_repair.seam_source_reasons`` at all, and a source of either polarity is
#: not this seam's business.
CARRIES_DEPOSIT_REPAIR = False

#: Which repair the two slots would install. EMPTY: they install none.
REPAIR_PATHS: Tuple[str, ...] = ()

#: Does this product perform the seam's electric withdraw before its launch? NO in
#: this round -- see the module docstring. The predicate consequently refuses every
#: row with a standing in-seam withdraw BY NAME: 2 of the cell's 49.
HOISTS_THE_WITHDRAW = False

#: May the composer install this product? NO, on every configuration, for a MEASURED
#: verdict rather than a policy. ``launch._declared_uninstallable`` reads this and
#: reports :data:`INSTALLABLE_REASON` on every run.
INSTALLABLE = False

INSTALLABLE_REASON = (
    "THE COMPOSER IS NOT OFFERED THIS PRODUCT AT ALL, and that is the first half of "
    "the reason: routing it through launch._install_certified_fused_products costs a "
    "label, and on this backend a label plan_step can write must also be declared in "
    "the DISPATCHER'S own tables -- the see-through the pending-gate rung reads, and "
    "the pending list an unreleased label sits in until a ledger entry exists. Those "
    "tables are outside this package by the one-way rule this package keeps (nothing "
    "here may know a dispatcher exists), and outside the round that built this "
    "product; parity/meep_gpu/dispatch_reachability.CERTIFIED_BUT_NOT_INSTALLED names "
    "them exactly, which is the right side of that wall to name them from. The flag "
    "is what keeps a later edit that lands those lines from installing the product "
    "before a composition gate has driven it end to end. "
    "THE SECOND HALF IS THE ARBITRATION, and it is MEASURED PER ROW rather than "
    "asserted: over the driver's step_B - update_H - step_D - update_E slot path "
    "launches are 4 - (installed pairs) and a TWO-SLOT H->D product takes one slot "
    "from EACH neighbour, so installing it is a LOSS where both neighbouring pairs "
    "install, a TIE where exactly one does, and a GAIN where NEITHER does. Which of "
    "the three each corpus row is comes from that row's OWN composer slot table and "
    "is recorded, row by row, in the lift leg of "
    "parity/meep_gpu/gate_triton_fused_hd_pair.py under "
    "`arbitration_over_the_driven_rows`; the counts are in the artifact and not "
    "copied here. On a GAIN row installing this product would save one launch, which "
    "is exactly why the flag and not the arithmetic is what withholds it: no "
    "composition gate has driven this product inside plan_step's own installation on "
    "any row. The only span that is strictly additive everywhere is a four-slot "
    "step_B -> update_H -> step_D -> update_E weld, which is not built")

__all__ = [
    "ARMS", "BACKWARD", "CARRIES_DEPOSIT_REPAIR", "CONSTITUTIVE_LIFT_EDITS",
    "DEFAULT_BLOCK",
    "CONSTITUTIVE_SIDE", "CONSTITUTIVE_SOURCES", "CURL_SUB_STEP", "FAMILY",
    "HALO_TAPS", "HOISTS_THE_WITHDRAW", "H_CELL_TAIL_ARGS", "INSTALLABLE",
    "INSTALLABLE_REASON", "IN_PLACE", "OWN_LOAD_EDITS", "REPAIR_PATHS", "REPLACES",
    "ROTATED", "SEAM", "SLOT",
    "FusedHdPairPlan",
    "certified_constitutive_tail", "certified_curl_tail", "curl_lift_edits",
    "explain_fused_hd_pair", "fused_hd_pair_coverage",
    "fused_constitutive_curl_H_to_D_kernel", "halo_taps", "lifted_constitutive_tail",
    "lifted_curl_tail", "offset_coordinates", "plan_fused_hd_pair",
    "plan_fused_hd_pair_from_arrays",
]


# ---------------------------------------------------------------------------
# The machine-checked lift -- host text, no Triton
# ---------------------------------------------------------------------------

#: Where BOTH certified bodies end their index decode. One anchor serves both, which
#: is what lets the fused kernel carry one decode prologue.
DECODE_END = "    i = plane // ny\n"

#: Where :func:`_h_cell` re-composes the flat index from its own coordinates -- the
#: helper's own anchor, below which every line is lifted text.
H_CELL_DECODE_END = "        idx = i * nyz + j * nz + k\n"

#: Where :func:`_h_cell` stops being lifted text and returns its six registers.
H_CELL_RETURN_START = "        return a0, a1, a2, src0, src1, src2\n"

#: Where the fused kernel's lifted CURL body begins. A marker rather than the shared
#: decode anchor, because the weld block sits between the decode and the curl.
CURL_BODY_ANCHOR = "        # === the certified step_D curl body begins here ===\n"

#: What one call site forwards after the four leading arguments. Built once so the
#: declaration, the generated lift edits and every call cannot drift apart.
H_CELL_TAIL_ARGS = ("hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,\n"
                    "             kp0, kp1, kp2, kmx, kmy, kmz, ny, nz")


def _source_of(function_name: str, path: Optional[Path] = None) -> str:
    """One function's source TEXT, by name, off the file rather than by import.

    TEXT AND NOT ``inspect``: :mod:`.kernels` imports Triton at module scope, so on
    the laptop that is this package's merge bar it cannot be imported at all -- and
    the lift checks are exactly the checks that must run there.
    """
    path = Path(__file__).with_name("kernels.py") if path is None else path
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            segment = ast.get_source_segment(text, node)
            if segment:
                # `get_source_segment` stops at the last character of the last
                # statement, so a body whose final line is an anchor would have no
                # newline for the anchor to match. Restored, once, here.
                return segment if segment.endswith("\n") else segment + "\n"
    raise AssertionError(
        f"{path.name} declares no function {function_name}; the lift has nothing to "
        f"cut and would compare two empty strings")


def _cut(source: str, anchor: str, *, stop: Optional[str] = None) -> str:
    """The text below ``anchor`` (and above ``stop``), dedented. Refuses a bad cut."""
    if anchor not in source:
        raise AssertionError(
            f"the body no longer carries the anchor {anchor.strip()!r}; the lift has "
            f"nowhere to cut and a partial cut compares nothing")
    tail = source.split(anchor, 1)[1]
    if stop is not None:
        if stop not in tail:
            raise AssertionError(
                f"the body no longer carries the stop anchor {stop.strip()!r}")
        tail = tail.split(stop, 1)[0]
    tail = textwrap.dedent(tail)
    if not tail.strip():
        raise AssertionError(
            "the body between its anchors is empty; the anchors have crossed")
    return tail


def needle(text: str, old: str, new: str) -> str:
    """Replace ``old`` by ``new``, requiring EXACTLY ONE match.

    A needle that matched nothing is a stale read left standing and one that matched
    twice is two edits; both raise here rather than emitting a kernel whose text is
    not the certified text plus the declared edits.
    """
    hits = text.count(old)
    if hits != 1:
        raise AssertionError(
            f"the lift needle {old.strip()[:70]!r} matches {hits} times, not once; "
            f"the certified body's spelling has changed")
    return text.replace(old, new, 1)


# ---------------------------------------------------------------------------
# The constitutive lift: update_H as a function of an arbitrary cell
# ---------------------------------------------------------------------------

#: Every line of certified CONSTITUTIVE text this weld does not lift verbatim, with
#: the reason. DATA, not prose, so the host suite and the gate can assert the list and
#: a line that stopped matching RAISES instead of silently leaving a stale read.
#:
#: THERE IS NO ARITHMETIC IN THIS TABLE. Every entry is a pointer spelling, a
#: constexpr branch resolved to the side this product serves, or a store that moves to
#: the caller; not one right-hand side, operand order or paren is touched.
CONSTITUTIVE_LIFT_EDITS: Tuple[Dict[str, str], ...] = (
    {"line": "km_0 = tl.load(km0 + i, ...)",
     "became": "km_0 = tl.load(kmx + i, ...)",
     "why": "ONE POINTER RENAME, NO ARITHMETIC. Both halves sit on the same Yee "
            "sub-lattice (SUB_STEPS['step_D']['suffix'] == '' and "
            "CONSTITUTIVE_SIDES['H']['half_integer'] is False), so the curl's kms_x "
            "and the constitutive's km0 ARE THE SAME VOLUME and the fused signature "
            "binds it once. Binding the half-integer set instead compiles and is a "
            "half-cell error in the absorber profile, converged and smooth and "
            "wrong; plan_fused_hd_pair ASSERTS the two suffixes agree."},
    {"line": "if SCALE: ... else: srcN = tl.load(gN + idx, mask=live, other=0.0)",
     "became": "srcN = tl.load(bN + idx, mask=live, other=0.0)",
     "why": "the constexpr branch is RESOLVED to the side this product serves. "
            "SCALE is 0 for update_H (the source is B, mu = 1 baked in exactly as "
            "the array path bakes it at stepping.py:907-923), so the SCALE=1 arm is "
            "text no launch of this product could reach and the three inverse-epsilon "
            "pointers leave with it. The surviving expression is the certified "
            "SCALE=0 arm character for character, under the fused signature's own "
            "name for the magnetic flux density -- `gN` in the curl half is the "
            "MAGNETIC FIELD, so the constitutive source cannot keep that name."},
    {"line": "prevN = tl.load(wN + idx, ...)",
     "became": "prevN = tl.load(wiN + idx, ...)",
     "why": "the split-field history is read from the PRE-LAUNCH buffer, which "
            "nothing in this launch writes. On this side the newly written f_w_H IS "
            "B exactly, so an in-place write would hand a racing neighbour B where "
            "it needs B_prev -- the second, easily missed half of the hazard."},
    {"line": "tl.store(wN + idx, srcN, mask=live)",
     "became": "(removed -- the caller stores srcN to the SCRATCH volume)",
     "why": "the weld writes a scratch volume, so the helper must not store. The "
            "value and the expression that produced it are untouched; only the "
            "destination moves, and only for the program's OWN cell (a foreign "
            "recompute stores nothing at all)."},
    {"line": "aN = tl.load(fN + idx, ...)",
     "became": "aN = tl.load(hiN + idx, ...)",
     "why": "the accumulator is read from the PRE-LAUNCH H, which is const for the "
            "whole dispatch. This is what makes the foreign recompute a pure function "
            "of unwritten memory. In the fused signature `fN` is the DISPLACEMENT."},
    {"line": "tl.store(fN + idx, aN, mask=live)",
     "became": "(removed -- the value is RETURNED)",
     "why": "same as the split-field store: the destination moves to the caller, "
            "which addresses the scratch for its own cell and consumes the register "
            "for the curl half. The two accumulations, their order and their parens "
            "are untouched."},
    {"line": "idx = tl.program_id(0) * BLOCK + ... ; i = plane // ny",
     "became": "(removed -- i, j and k are parameters; idx is composed from them)",
     "why": "THE WHOLE POINT OF THE LIFT. The certified body becomes a function "
            "evaluated at an ARBITRARY cell rather than at the program's own, so the "
            "cell arrives as three coordinates and the flat index is COMPOSED by the "
            "same layout the decode inverts. `live` arrives as the caller's per-lane "
            "validity: a tap the certified ghost rule masked off passes False, every "
            "load inside is masked, and no out-of-range coordinate is dereferenced."},
)


def certified_constitutive_tail() -> str:
    """The ``update_H`` body BELOW the decode prologue, stores removed, dedented.

    Not a transcription: this is :func:`.kernels.constitutive_step`'s own text with
    exactly the substitutions :data:`CONSTITUTIVE_LIFT_EDITS` names, each applied
    through :func:`needle`, so a spelling that drifted raises here rather than leaving
    the fused kernel reading the wrong volume.
    """
    tail = _cut(_source_of("constitutive_step"), DECODE_END)
    axes = ("i", "j", "k")
    pointers = ("kmx", "kmy", "kmz")
    for target in range(3):
        tail = needle(
            tail,
            f"km_{target} = tl.load(km{target} + {axes[target]}, "
            f"mask=live, other=0.0)",
            f"km_{target} = tl.load({pointers[target]} + {axes[target]}, "
            f"mask=live, other=0.0)")
    for target in range(3):
        tail = needle(
            tail,
            f"if SCALE:\n"
            f"    src{target} = tl.load(g{target} + idx, mask=live, other=0.0) * "
            f"tl.load(e{target} + idx, mask=live,\n"
            f"                                                             "
            f"other=0.0)\n"
            f"else:\n"
            f"    src{target} = tl.load(g{target} + idx, mask=live, other=0.0)\n",
            f"src{target} = tl.load(b{target} + idx, mask=live, other=0.0)\n")
        tail = needle(tail, f"prev{target} = tl.load(w{target} + idx",
                      f"prev{target} = tl.load(wi{target} + idx")
        tail = needle(tail,
                      f"tl.store(w{target} + idx, src{target}, mask=live)\n", "")
        tail = needle(tail, f"a{target} = tl.load(f{target} + idx",
                      f"a{target} = tl.load(hi{target} + idx")
        tail = needle(
            tail, f"tl.store(f{target} + idx, a{target}, mask=live)\n", "")
    # THE CHECK THAT MATTERS. In the fused signature `f*`, `w*` and `g*` are the
    # DISPLACEMENT, its split-field auxiliary and the magnetic field; one missed
    # rename is a smooth, plausible, entirely wrong answer rather than a compile
    # error, because those names all exist in the enclosing kernel.
    for stem in ("f", "w", "g", "e"):
        for target in range(3):
            if f"{stem}{target} +" in tail:
                raise AssertionError(
                    f"the lifted update_H body still indexes {stem}{target}, which "
                    f"in the fused kernel is not the volume the certified body meant")
    return tail.rstrip("\n") + "\n"


def lifted_constitutive_tail() -> str:
    """:func:`_h_cell`'s own body between its two anchors, dedented."""
    source = _source_of("_h_cell", Path(__file__))
    return _cut(source, H_CELL_DECODE_END, stop=H_CELL_RETURN_START)


# ---------------------------------------------------------------------------
# The curl lift: step_D with every magnetic read redirected
# ---------------------------------------------------------------------------

#: The curl half's three OWN-cell magnetic loads and the register each becomes. The
#: program has already computed its own ``H`` into ``own0``/``own1``/``own2``, and
#: those registers are the post-``update_H`` values the array path would have loaded.
OWN_LOAD_EDITS: Tuple[Tuple[str, str], ...] = tuple(
    (f"{var} = tl.load(g{target} + idx, mask=live, other=0.0)\n",
     f"{var} = own{target}\n")
    for target, var in enumerate(("a", "b", "c")))

#: The curl half's six SHIFTED magnetic loads, as ``(register, component)``. WHICH
#: CELL each one reads is NOT written here: it is parsed out of the certified body's
#: own ``ox = si * nyz + j * nz + k`` lines by :func:`offset_coordinates`, and which
#: MASK guards it is parsed out of the load line itself by :func:`halo_taps`. So the
#: recompute lands on exactly the cell the emitter's index composed, and a change to
#: how that index or that guard is spelled RAISES instead of quietly redirecting a
#: tap.
#:
#: THE BRANCH AND THE SHIFT ARITHMETIC ARE NOT TOUCHED. Only the leaf load moves. The
#: validity flag is the emitter's own, so a metallic ghost that read an exact ``0.0``
#: still reads an exact ``0.0`` (``_h_tap`` closes on ``tl.where(valid, value,
#: 0.0)``), and the periodic wrap is still the emitter's own integer expression.
HALO_TAPS: Tuple[Tuple[str, int], ...] = (
    ("a_y", 0), ("a_z", 0), ("b_x", 1), ("b_z", 1), ("c_x", 2), ("c_y", 2))


def offset_coordinates(tail: str) -> Dict[str, Tuple[str, str, str]]:
    """``{offset name: (i, j, k) expressions}``, read off the certified index lines.

    The emitter writes one line per shifted axis, each of the shape
    ``ox = si * nyz + j * nz + k`` -- the SAME layout :func:`_h_cell` composes ``idx``
    from. Parsing it rather than tabulating it is what makes "the index expressions
    are the emitter's" a property of the construction.
    """
    out: Dict[str, Tuple[str, str, str]] = {}
    for line in tail.splitlines():
        stripped = line.strip()
        if not stripped.startswith("o") or " = " not in stripped:
            continue
        name, expression = stripped.split(" = ", 1)
        if name not in ("ox", "oy", "oz"):
            continue
        parts = [piece.strip() for piece in expression.split(" + ")]
        if len(parts) != 3 or not parts[0].endswith(" * nyz") \
                or not parts[1].endswith(" * nz"):
            raise AssertionError(
                f"the certified curl composes {name} as {expression!r}, which is not "
                f"the `a * nyz + b * nz + c` layout this weld recomposes from "
                f"coordinates")
        out[name] = (parts[0][: -len(" * nyz")], parts[1][: -len(" * nz")], parts[2])
    if set(out) != {"ox", "oy", "oz"}:
        raise AssertionError(
            f"the certified curl composes {sorted(out)} rather than the three shifted "
            f"offsets this weld redirects")
    return out


def halo_taps(tail: str) -> Dict[str, Tuple[int, str, str]]:
    """``{register: (component, offset name, mask name)}``, read off the load lines.

    The emitter writes each shifted magnetic load as
    ``a_y = tl.load(g0 + oy, mask=vy, other=0.0)``. Everything this weld needs to
    redirect it -- which component, which cell, which guard -- is in that line, so it
    is PARSED rather than transcribed and a spelling change raises.
    """
    out: Dict[str, Tuple[int, str, str]] = {}
    for line in tail.splitlines():
        stripped = line.strip()
        if " = tl.load(g" not in stripped or "+ idx" in stripped:
            continue
        register, rest = stripped.split(" = tl.load(g", 1)
        component = int(rest[0])
        body = rest[1:]
        if not body.startswith(" + ") or not body.endswith(", other=0.0)"):
            raise AssertionError(
                f"the certified curl no longer reads {register} as "
                f"`tl.load(gN + offset, mask=..., other=0.0)`; the ghost rule this "
                f"weld preserves has changed")
        offset, mask = body[len(" + "):-len(", other=0.0)")].split(", mask=", 1)
        out[register.strip()] = (component, offset.strip(), mask.strip())
    declared = {name: component for name, component in HALO_TAPS}
    if set(out) != set(declared) or {k: v[0] for k, v in out.items()} != declared:
        raise AssertionError(
            f"the certified curl's shifted magnetic loads are "
            f"{ {k: v[0] for k, v in out.items()} }, not the declared "
            f"{declared}; this weld redirects exactly one load per shifted tap")
    return out


def curl_lift_edits() -> Tuple[Tuple[str, str], ...]:
    """The nine ``(old, new)`` replacements the curl half takes, GENERATED.

    Three own-cell loads become registers, and six shifted loads become a recompute at
    the cell the emitter's own index line composed, under the emitter's own guard.
    Nothing in the returned pairs is transcribed from this file except the call's
    shape: the component, the coordinates and the mask all come from the parse.
    """
    tail = _cut(_source_of("pml_curl_step"), DECODE_END)
    offsets = offset_coordinates(tail)
    taps = halo_taps(tail)
    edits: List[Tuple[str, str]] = list(OWN_LOAD_EDITS)
    for register, component in HALO_TAPS:
        _component, offset, mask = taps[register]
        coordinates = ", ".join(offsets[offset])
        edits.append((
            f"{register} = tl.load(g{component} + {offset}, mask={mask}, "
            f"other=0.0)\n",
            f"{register} = _h_tap({component}, {coordinates}, {mask}, "
            f"{H_CELL_TAIL_ARGS})\n"))
    return tuple(edits)


def certified_curl_tail() -> str:
    """:func:`.kernels.pml_curl_step`'s body below the decode, every H read redirected."""
    tail = _cut(_source_of("pml_curl_step"), DECODE_END)
    for old, new in curl_lift_edits():
        tail = needle(tail, old, new)
    for target in range(3):
        if f"g{target} +" in tail:
            raise AssertionError(
                f"the welded curl half still reads g{target}; in this signature that "
                f"pointer does not exist and every magnetic read must be the register "
                f"or a recompute")
    return tail


def lifted_curl_tail() -> str:
    """The fused kernel's own curl body, from its marker to the end, dedented."""
    source = _source_of("fused_constitutive_curl_H_to_D", Path(__file__))
    return _cut(source, CURL_BODY_ANCHOR)


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

    #: Boundary codes, identical to :mod:`.kernels`' own.
    PERIODIC = tl.constexpr(0)
    METALLIC = tl.constexpr(1)

    @triton.jit
    def _h_cell(i, j, k, live,
                hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
                kp0, kp1, kp2, kmx, kmy, kmz, ny, nz):
        """``kernels.constitutive_step`` for side H at ONE named cell, returning.

        ONE code path serves the program's own cell and every foreign recompute, so
        the two cannot disagree. The body below the index line is the certified
        body's own text with exactly :data:`CONSTITUTIVE_LIFT_EDITS`;
        :func:`certified_constitutive_tail` and :func:`lifted_constitutive_tail` cut
        both at their declared anchors and the probe asserts the two strings equal,
        so this is a LIFT and not a transcription.

        ``live`` is the caller's per-lane validity for THIS cell, not the dispatch's
        ``idx < n_elem``: a foreign tap masked off by the certified ghost rule passes
        ``live=False``, every load inside is masked, and no out-of-range coordinate
        is ever dereferenced.
        """
        nyz = ny * nz
        idx = i * nyz + j * nz + k

        # Component 0 takes its coefficient from axis x, 1 from y, 2 from z.
        kp_0 = tl.load(kp0 + i, mask=live, other=0.0)
        km_0 = tl.load(kmx + i, mask=live, other=0.0)
        kp_1 = tl.load(kp1 + j, mask=live, other=0.0)
        km_1 = tl.load(kmy + j, mask=live, other=0.0)
        kp_2 = tl.load(kp2 + k, mask=live, other=0.0)
        km_2 = tl.load(kmz + k, mask=live, other=0.0)

        # --- component 0 -----------------------------------------------------------
        prev0 = tl.load(wi0 + idx, mask=live, other=0.0)   # BEFORE the store; note 3.
        src0 = tl.load(b0 + idx, mask=live, other=0.0)
        a0 = tl.load(hi0 + idx, mask=live, other=0.0)
        a0 = a0 + kp_0 * src0
        a0 = a0 - km_0 * prev0

        # --- component 1 -----------------------------------------------------------
        prev1 = tl.load(wi1 + idx, mask=live, other=0.0)
        src1 = tl.load(b1 + idx, mask=live, other=0.0)
        a1 = tl.load(hi1 + idx, mask=live, other=0.0)
        a1 = a1 + kp_1 * src1
        a1 = a1 - km_1 * prev1

        # --- component 2 -----------------------------------------------------------
        prev2 = tl.load(wi2 + idx, mask=live, other=0.0)
        src2 = tl.load(b2 + idx, mask=live, other=0.0)
        a2 = tl.load(hi2 + idx, mask=live, other=0.0)
        a2 = a2 + kp_2 * src2
        a2 = a2 - km_2 * prev2
        return a0, a1, a2, src0, src1, src2

    @triton.jit
    def _h_tap(COMP: tl.constexpr, i, j, k, valid,
               hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
               kp0, kp1, kp2, kmx, kmy, kmz, ny, nz):
        """One FOREIGN magnetic field the curl half would have loaded.

        The whole substitution, in one place: step the cell from PRE-LAUNCH state,
        take the component the certified load named, and serve an exact ``+0.0``
        where that load's mask was False -- which is what ``other=0.0`` served.
        """
        a0, a1, a2, s0, s1, s2 = _h_cell(i, j, k, valid, hi0, hi1, hi2,
                                         wi0, wi1, wi2, b0, b1, b2,
                                         kp0, kp1, kp2, kmx, kmy, kmz, ny, nz)
        value = a0
        if COMP == 1:
            value = a1
        if COMP == 2:
            value = a2
        return tl.where(valid, value, 0.0)

    @triton.jit
    def fused_constitutive_curl_H_to_D(
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
        """``update_H`` + ``step_D``, one launch, scratch output.

        ``hi``/``wi``/``b`` are const for the whole dispatch and ``ho``/``wo`` are
        write-only scratch, so nothing written by the constitutive half is read by
        this launch; ``f``/``u`` step IN PLACE and that is safe by construction,
        because the curl reads and writes them at the program's OWN cell only.
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
        own0, own1, own2, src0, src1, src2 = _h_cell(
            i, j, k, live, hi0, hi1, hi2, wi0, wi1, wi2, b0, b1, b2,
            kp0, kp1, kp2, kmx, kmy, kmz, ny, nz)
        tl.store(ho0 + idx, own0, mask=live)
        tl.store(ho1 + idx, own1, mask=live)
        tl.store(ho2 + idx, own2, mask=live)
        tl.store(wo0 + idx, src0, mask=live)
        tl.store(wo1 + idx, src1, mask=live)
        tl.store(wo2 + idx, src2, mask=live)

        # === the certified step_D curl body begins here ===

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


def fused_constitutive_curl_H_to_D_kernel() -> Any:
    """The shipped kernel object, or a refusal naming the missing import."""
    if triton is None:  # pragma: no cover - the laptop path
        raise ImportError(
            "the fused H->D pair needs Triton to launch; the predicate, the lift "
            f"checks and the plan builder answer without it ({_TRITON_IMPORT_ERROR})")
    return fused_constitutive_curl_H_to_D


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def fused_hd_pair_coverage(fields: Any, pml: Any,
                           sources: Any = None) -> "_coverage.Coverage":
    """May ONE launch span ``update_H`` -> the electric withdraw -> ``step_D``?

    A conjunction of the two halves' OWN certified predicates plus the seam clauses.
    Nothing is weakened: a configuration either half refuses is refused here with that
    half's reasons, prefixed so a reader can tell which side said it.

    THE ORDER IS THE DRIVER'S. The constitutive half is asked FIRST because it runs
    first (driver.py:3311), so the first refusal a reader sees names the half the
    driver would have reached first -- and on this seam that matters, since the null
    ``update_H`` (an inactive absorber returns before its first statement) is the
    single largest non-fusion reason on the board and belongs to the constitutive side.
    """
    grid = getattr(fields, "grid", None)
    if grid is None:
        return _coverage.Coverage(False, ("fields carries no grid",))

    reasons: List[str] = []
    constitutive = _coverage.constitutive_coverage(fields, pml, CONSTITUTIVE_SIDE)
    if not constitutive.covered:
        reasons.extend(f"constitutive half: {reason}"
                       for reason in constitutive.reasons)
    curl = _coverage.pml_curl_coverage(fields, pml, CURL_SUB_STEP)
    if not curl.covered:
        reasons.extend(f"curl half: {reason}" for reason in curl.reasons)

    # THE SEAM'S ONE PASS. Nothing is INJECTED between the two consults, so
    # `deposit_repair` is not consulted at all and a source of either polarity is not
    # this seam's business -- the magnetic injection is one seam earlier
    # (driver.py:3283-3284) and the electric one is one seam later (:3317-3322). What
    # IS between them is the electric integrated-source withdraw (:3313-3314), and
    # `withdraw_hoist` owns it. IGNORANCE IS NEVER AN EMPTY SET: `Fields` does not
    # hold the source list, so a predicate that inferred "no withdraw stands" from not
    # being told would be the over-covering refusal this clause exists to prevent.
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

    # THE FOLD, RESTATED BY NAME. Both halves already refuse a mirror plane
    # (`coverage._grid_reasons` clause 5), and on THIS seam the reason is not the one
    # the D->E and B->H pairs give: neither fill runs between these two consults
    # (fill_B closes one seam earlier at driver.py:3304-3309 and fill_D opens one seam
    # later at :3324-3327), so the H->D seam is fill-FREE on a folded grid. What
    # refuses a fold here is that this product implements the `ordinary` and `PML`
    # arms and a folded run selects the `folded` arms on both slots -- the 78-row
    # cell, and a second product.
    mirrored = getattr(grid, "is_mirrored", None)
    for axis in range(3):
        if callable(mirrored) and bool(mirrored(axis)):
            reasons.append(
                f"axis {axis} is folded: this product implements the `ordinary` "
                f"update_H and `PML` step_D arms, and a folded run selects the "
                f"`folded` arms on both slots; the (folded -> folded PML) cell needs "
                f"its own product and its own ghost-map measurement")

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


def explain_fused_hd_pair(fields: Any, pml: Any,
                          sources: Any = None) -> "_coverage.Coverage":
    """The predicate under the name a report reads. One home for the verdict."""
    return fused_hd_pair_coverage(fields, pml, sources)


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class FusedHdPairPlan(ScratchWeldPairPlan):
    """ONE allocation-free launch for ``update_H`` and ``step_D``.

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

        The whole design is that nothing written by the constitutive half is read;
        an aliased pair would put the schedule back into the answer, and would do it
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
                  else fused_constitutive_curl_H_to_D_kernel())
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
        return (f"FusedHdPairPlan(shape={self.shape}, bc={self.bc}, "
                f"block={self.block}, num_warps={self.num_warps})")


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


def plan_fused_hd_pair(fields: Any, pml: Any, sources: Any = None,
                       block: Optional[int] = None, num_warps: Optional[int] = 1,
                       kernel: Any = None) -> Optional[FusedHdPairPlan]:
    """Build the plan from the engine's own objects, or ``None`` when refused.

    ``None`` is the only refusal, for the reason :func:`.launch.plan_pml_curl` gives:
    a configuration this kernel does not carry must fall back to the array path, never
    raise into a caller that would otherwise have stepped correctly.

    ``kernel`` is the mutation seam. The gate compiles a deliberately broken copy of
    the shipped kernel and hands it here; dropping the argument is not a silent
    slowdown, it is a silent DISARMING -- every mutation leg would then launch the
    shipped kernel and report the defect as uncaught.
    """
    if not fused_hd_pair_coverage(fields, pml, sources).covered:
        return None
    suffix, constitutive_suffix = _sub_lattice_suffixes()
    grid = fields.grid
    kinds = _coverage._boundary_kinds(grid, pml)  # noqa: SLF001 - the shared resolver
    return FusedHdPairPlan(
        grid.shape, grid.dt / grid.dx,
        [1 if kind == "metallic" else 0 for kind in kinds],
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


def plan_fused_hd_pair_from_arrays(
        arrays: Dict[str, Any], flat: Dict[str, Any], dtdx: float,
        codes: Sequence[int], fields: Any, block: Optional[int] = None,
        kernel: Any = None, num_warps: Optional[int] = 1) -> FusedHdPairPlan:
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
    return FusedHdPairPlan(
        shape, dtdx, codes, DEFAULT_BLOCK if block is None else block,
        fields, twins,
        [arrays[name] for name in CONSTITUTIVE_SOURCES],
        [arrays[name] for name in IN_PLACE[:3]],
        [arrays[name] for name in IN_PLACE[3:]],
        [flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
        [flat[f"kps_{axis}"] for axis in "xyz"],
        kernel=kernel, num_warps=num_warps,
    )
