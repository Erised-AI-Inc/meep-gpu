"""The H->D weld on a REAL-storage ``grid.beta`` run: ``update_H`` into ``step_D``.

The special_kz sibling of :mod:`.fused_hd_pair`, on the same seam, with the same
shape -- ONE launch, scratch output, foreign-cell recompute, a rotation after --
and the same two flags (:data:`INSTALLABLE` False, :data:`HOISTS_THE_WITHDRAW`
False). ONE device string, as its real sibling has one: the boundary kind is a
runtime argument on this backend, so a folded beta row and an unfolded one compile
to the same text.

=============================================================================
THE CELL
=============================================================================

``parity/meep_gpu/results/fusion_matrix_cuda_2026-09-07_wired/fusion_matrix_cuda.json``,
``h_to_d_seam.instances`` filtered to ``buildable_not_built``:

* ``(update_H cuda_special_kz/real beta -> step_D cuda_special_kz/real beta)`` --
  **2 instances**, ``examples:refl-angular-kz2d.py`` and
  ``tests:TestSpecialKz.test_eigsrc_kz_1_real_imag``.

Those are the only two REAL-storage beta rows in the corpus, and
``special_kz_curl``'s own admission says so and says why (the other four beta rows
are complex storage and belong to :mod:`.complex_beta_fused_hd_pair`, the sibling
product built in the same round). Both instances carry
``integrated_electric_sources = 0`` and neither is in the ``withdraw_seam`` bucket,
so :data:`HOISTS_THE_WITHDRAW` False costs this cell nothing measurable -- which is
a fact read off the board, not a hope, and the clause is still asked.

THE UPDATE_H HALF IS NOT A BETA KERNEL, AND THAT IS THE CELL'S SHAPE RATHER THAN A
SHORTCUT. ``special_kz_curl`` ships no constitutive kernel:
``covers_special_kz_constitutive`` is an ADMISSION over the CERTIFIED
``update_H_pml_real``, because ``stepping.update_H`` (:907-923) through
``_apply_constitutive_pml`` (:2065) reads no ``grid.beta`` and no array the beta term
writes. The census agrees from the other side -- it records
``cuda_special_kz/real beta`` at ``update_H`` on both rows because the family's
PREDICATE answers there, while the kernel it admits is the ordinary certified one.
So this weld's constitutive half is byte-for-byte
:func:`.fused_hd_pair.raw_update_H_cell_source`, reused rather than re-lifted.

=============================================================================
THE SHAPE: SCRATCH OUTPUT AND FOREIGN-CELL RECOMPUTE
=============================================================================

Identical in structure AND in justification to :mod:`.fused_hd_pair`, so the
argument is not restated at length. In brief: ``step_D``'s curl reads ``H`` at the
thread's own cell and at three BACKWARD neighbours (``shift_dn``) while ``update_H``
writes ``H`` and ``f_w_H``, so an in-place weld is a race on both volumes -- and on
``f_w_H`` especially, since the newly stored ``f_w_H`` IS ``B`` exactly, so an
in-place write hands a racing neighbour ``B`` where its recurrence needs ``B_prev``.
The constitutive half therefore writes NOTHING in place (``H_new`` and ``f_w_H_new``
go to launch-local scratch, and ``H``, ``f_w_H`` and ``B`` are ``const`` for the
whole launch); every foreign read is a RECOMPUTE of the certified pointwise
``update_H`` from that same unwritten state; and the launcher ROTATES the
``H``/``f_w_H`` bindings afterwards
(:func:`.fused_hd_pair.rotate_into_fields`, reused rather than re-spelled).

=============================================================================
WHAT THIS SEAM ADDS THAT NO OTHER H->D WELD HAS: THE BETA PARTNER
=============================================================================

``stepping._special_kz_beta_term`` takes the SAME-CELL component snapshot
(stepping.py:437 magnetic, :321 electric), and in the D-side curl that snapshot is
already in a register: Dx's partner is the Hy CENTRE the stencil loaded into ``f2``
and Dy's is the Hx CENTRE in ``f1`` (special_kz_curl.py's own transcription, pinned
there against ``D_CURL_TERMS`` and the call sites :443/:445). The array path
evaluates ``step_D`` AFTER ``update_H``, so that snapshot is the POST-``update_H``
magnetic field.

**Welded, those two registers are the weld's own** ``own_h[1]`` **and**
``own_h[0]``, because this weld redirects every own-cell magnetic load in the body
and the beta term consumes the redirected declaration rather than a load of its own.
That is checked and not assumed: :func:`welded_curl_tail` asserts, per target block,
that each surviving ``beta_*`` line names a register whose declaration it has just
rewritten to ``own_h[...]``. A beta term left reading the PRE-launch ``H`` would be
one sub-step behind on exactly two of the six curl terms -- converged, plausible and
wrong -- and it is the armed mutation ``beta_partner_reads_stale_h``, which no
sibling H->D gate has a site for.

THE ARITHMETIC IS THE CERTIFIED FAMILY'S. Every operation in this kernel comes from
``step_curl_kernels._REAL_PML_PRELUDE`` (``shift_up``, ``shift_dn``, ``pml_apply``,
the three boundary codes), from ``constitutive_kernels``' ``constitutive_apply``, or
from ``special_kz_curl``'s two beta lines. NOTHING is retyped.

TWO CuPy SPELLINGS MEASURED ON THIS BACKEND ARE RESPECTED RATHER THAN REDISCOVERED,
and on THIS family both are recorded as ABSENCES rather than sites:

* ``complex64 * float32`` -- the contracted four-product form -- has NO SITE here.
  This family is REAL float32 storage throughout; there is no ``cf`` in its text and
  :func:`kernel_source` asserts it.
* ``float32 / float32`` is a TRUE divide, and there is no division in this kernel
  either: the reciprocal the recurrence needs is ``sinv``, computed host-side by
  ``PML``. Recorded rather than omitted, because a reader arriving from the
  cylindrical H->D products (where the increment DOES divide) will look for it, and
  because the complex sibling's divide spelling is a DIFFERENT algorithm that must
  never be carried here. The gate asserts the absence and arms it.

What IS live is ``--fmad=false``: ``curl - (beta * f)`` is a multiply feeding a
subtract, exactly the shape a compiler contracts, where the array path rounds the
product (stepping.py:811) and the addition (:472) separately. That guard is the
certified special_kz pair's own and is restated in :data:`_COMPILE_OPTIONS` rather
than imported, so loading this file by path cannot pick up a different tuple than
the one the gate compiled.

=============================================================================
IT IS NOT INSTALLED, AND THE REASON IS THE SIBLING'S MEASURED VERDICT
=============================================================================

:data:`INSTALLABLE` is False. Over the driver's ``step_B - update_H - step_D -
update_E`` slot path launches are ``4 - (installed pairs)``, and a product spanning
``update_H``/``step_D`` takes one slot from EACH neighbour. On THIS cell both
neighbours are RELEASED products -- ``cuda_special_kz_fused_magnetic_pair`` holds
``step_B``/``update_H`` and ``cuda_special_kz_fused_electric_pair`` holds
``step_D``/``update_E`` -- so the span can only TIE (one neighbour installs) or LOSE
(both do). That is the Cartesian algebra ``fused_hd_pair.INSTALLABLE_REASON`` prices
and there is nothing here to re-price: this cell's seam is 2 launches today and this
product takes it to 1, saving exactly what each neighbour saves.

SHIP THE FLAG FALSE AND MEASURE THE ARBITRATION ANYWAY. What keeps the released
B->H pair in its slot is not this flag but ``fused_pairs._neighbouring_seam_claimant``'s
end-edge guard together with ``_spans_may_absorb`` reading the live ``selected``; the
flag is belt and braces in front of them, and the gate's arbitration leg measures
both arrangements THROUGH THE SHIPPED COMPOSER rather than assuming either.

=============================================================================
NOT DISPATCHED, AND NOT YET IN THE COMPOSER'S TABLES
=============================================================================

Nothing under ``meep_gpu/`` imports ``cuda_kernels`` outside tests;
``fastpath.plan_fast_path`` returns ``None`` on every branch. Beyond that, this
family is not in ``fused_pairs.FUSED_PRODUCTS``, ``registry`` or ``arms`` at all:
the wiring is a separate change that a later round applies. The gate PLANS EVERY
ARRANGEMENT FROM ARRAYS and drives the array path itself; the composer building a
composition that includes THIS product is measured in-process with the wiring rows
patched in (the arbitration leg) and awaits the wiring round.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import coverage as _coverage
from . import fused_hd_pair as _hd
from . import special_kz_curl as _kz
from .compile_cache import (clear_kernel_cache as _clear_cache,
                            get_or_compile as _get_or_compile,
                            kernel_cache_key as _kernel_cache_key)
from .coverage import constitutive_sub_lattice

from .. import withdraw_hoist as _withdraw_hoist

FAMILY = "cuda_special_kz_fused_hd_pair"

#: The kernel's entry-point symbol, spelled once and spelled here as a LITERAL in
#: :data:`_SIGNATURE_HEAD` as well: ``test_kernel_partition.py`` reads shipped kernel
#: names off this file's own text with a regex that requires a C identifier after
#: ``void``, and this module derives the rest of its parameter list from the released
#: real sibling rather than retyping it.
KERNEL_NAME = "fused_hd_pair_special_kz_real"

#: The sub-step slot this arm is registered on: the FIRST half of the seam, in the
#: driver's own order, so a refusal is named on the slot the fusion starts at.
SLOT = "update_H"

#: The driver passes ONE launch performs, in driver order (driver.py:3311, :3315).
#: Nothing between them is carried: the electric withdraw (:3313-3314) is declined in
#: this round, which is why it is not in this tuple.
REPLACES: Tuple[str, ...] = ("update_H", "step_D")

#: The seam name ``fused_pairs.FUSED_PAIR_SEAMS`` files this row under, and the module
#: that owns its in-seam pass. Spelled through :mod:`..withdraw_hoist` rather than as
#: a literal so the two cannot drift.
SEAM: str = _withdraw_hoist.SEAM

#: The six volumes the rotation swaps -- ``fused_hd_pair``'s own tuple, reused so the
#: products on this seam cannot disagree about what a rotation covers.
SCRATCH_VOLUMES: Tuple[str, ...] = _hd.SCRATCH_VOLUMES
H_TARGETS: Tuple[str, str, str] = _hd.H_TARGETS

#: How many shifted magnetic taps and own-cell magnetic loads the certified special_kz
#: ``step_D`` makes -- the certified real curl's own counts, reused from the released
#: sibling rather than restated, because the two bodies differ only by the beta lines,
#: two parameters, the kernel name and three comments (measured: see
#: ``test_special_kz_fused_hd_pair.py``). Both are ASSERTED against the lifted text on
#: every emit.
HALO_TAPS: int = _hd.HALO_TAPS
OWN_LOAD_EDITS: int = _hd.OWN_LOAD_EDITS

#: How many beta increments the certified special_kz ``step_D`` carries: one on
#: target 0 and one on target 1, none on target 2 (``cc`` runs over ``d_c`` in
#: {X, Y} only -- MEEP step_db.cpp:148-176, stepping.py:472-474). A site count is a
#: WELD: a body that grew or lost one must FAIL here rather than emit a kernel with a
#: beta term missing, which compiles, runs and is wrong on one component.
BETA_INSERT_LINES = 2

#: NOT CERTIFIED YET. The gate for this family is
#: ``parity/meep_gpu/gate_special_kz_fused_hd_pair.py`` and its artifacts are
#: ``parity/meep_gpu/results/special_kz_fused_hd_pair_2026-09-07/{keep,flush}/gate.json``.
#: A name moves from the dead set to the certified one only once
#: ``certification.json`` carries a block claiming it -- ``test_kernel_partition.py``
#: refuses a certified name no record block claims, and refuses a dead name a record
#: block does claim, so this pair of assignments is the whole partition for this file.
CERTIFIED_KERNELS: Tuple[str, ...] = ()

#: Spelled as a dict WITHOUT a type annotation, for the reason every sibling records:
#: the partition readers walk the syntax tree so they run where there is no CuPy, and
#: an annotated assignment is an ``ast.AnnAssign`` the plain-assignment readers do not
#: match -- annotating it makes the name invisible and the partition unenforced.
UNCERTIFIED_KERNELS = {
    "fused_hd_pair_special_kz_real":
        "gated by parity/meep_gpu/gate_special_kz_fused_hd_pair.py; the certification "
        "block that would move this name into CERTIFIED_KERNELS is cut from that "
        "gate's artifacts by record_cuda_regate.py and is not in this change",
}

#: Nothing is INJECTED between the two consults -- the electric injection is one seam
#: later (driver.py:3317-3322) and the magnetic one is one seam earlier (:3283-3284)
#: -- so there is no deposit in this seam to bracket. A FACT about the driver, not a
#: choice; this predicate consequently does not consult ``deposit_repair`` at all.
CARRIES_DEPOSIT_REPAIR = False

#: Not in this round. The only wiring that performs the hoist is
#: ``fused_pairs._install_fused_pair``'s ``withdraw_hoist.SEAM`` branch, which
#: ``_declared_uninstallable`` makes unreachable for a product declaring INSTALLABLE
#: False. True here would be a claim about wiring that cannot fire, and would let the
#: fused launch read a D still holding the previous step's standing dipole.
HOISTS_THE_WITHDRAW = False

INSTALLABLE = False
INSTALLABLE_REASON = (
    "slot arbitration, and on THIS cell the Cartesian algebra applies unchanged. Over "
    "the driver's step_B - update_H - step_D - update_E slot path launches are "
    "4 - (installed pairs); both of this cell's neighbours are RELEASED products "
    "(cuda_special_kz_fused_magnetic_pair holds step_B/update_H, "
    "cuda_special_kz_fused_electric_pair holds step_D/update_E), and a span taking "
    "one slot from each can only TIE (one neighbour installs) or LOSE (both do). The "
    "cylindrical H->D products re-priced this because their seam is 7 launches and "
    "the span takes it to 3; here the seam is 2 launches and the span takes it to 1, "
    "saving exactly what each neighbour saves. So INSTALLABLE stays False -- credited "
    "on the board as served by predicate admission, as fused_hd_pair is -- and the "
    "only shape that could pay is a four-slot step_B -> update_H -> step_D -> "
    "update_E weld, which is not built here")

WHAT_A_RELEASE_DOES_NOT_LICENSE = (
    "SERVED on the fusion board is PREDICATE ADMISSION by the board's own definition, "
    "so a released product credits its admitted seam-instances while executing "
    "NOWHERE: nothing under meep_gpu/ imports cuda_kernels outside tests, this family "
    "declares INSTALLABLE = False so the composer refuses to install it on every "
    "configuration once wired, it is not in the composer's tables until the wiring "
    "change lands, and NO TIMING of any kind has been taken of this shape. The fused "
    "route does MORE memory traffic than the two singles for the same step-level "
    "launch count -- up to six extra pointwise constitutive evaluations per thread -- "
    "and whether the saved launch and the saved H write-then-read round trip pay for "
    "them is a HYPOTHESIS. Every launch figure in this module is a COUNT.")

_FUSED_THREADS = 256

#: ``--fmad=false`` is CORRECTNESS and not tuning, and it is the certified special_kz
#: pair's own tuple, restated rather than imported so that loading this file by path
#: cannot pick up a different one than the gate compiled. Four contraction candidates:
#: the constitutive's ``a - kms * prev`` and ``f + kps * src``, the curl's
#: ``(fu * kms) - curl`` and ``(f * kms_u) + fu_new``, and the beta insert's
#: ``curl - (beta * f)``.
_COMPILE_OPTIONS: Tuple[str, ...] = ("--fmad=false",)

__all__ = [
    "BETA_INSERT_LINES", "CARRIES_DEPOSIT_REPAIR", "CERTIFIED_KERNELS", "FAMILY",
    "HALO_TAPS", "HOISTS_THE_WITHDRAW", "H_TARGETS", "INSTALLABLE",
    "INSTALLABLE_REASON", "KERNEL_NAME", "OWN_LOAD_EDITS", "REPLACES",
    "SCRATCH_VOLUMES", "SEAM", "SLOT", "UNCERTIFIED_KERNELS",
    "WHAT_A_RELEASE_DOES_NOT_LICENSE", "assert_scratch_is_disjoint",
    "beta_scalars", "covers_special_kz_fused_hd_pair", "device_sources",
    "kernel_source", "launch_special_kz_fused_hd_pair", "rotate_into_fields",
    "run_special_kz_fused_hd_pair", "signature", "source_digest",
    "special_kz_fused_hd_pair_codes", "special_kz_fused_hd_pair_scratch",
    "special_kz_fused_hd_pair_tables", "welded_curl_tail",
]


# ---------------------------------------------------------------------------
# The lift
# ---------------------------------------------------------------------------

#: The certified special_kz ``step_D``'s device text, taken from the module CONSTANT
#: rather than through ``special_kz_curl.kernel_source``. That function serves the
#: gate's MUTABLE copy, and a weld that read it would silently inherit whatever
#: mutation a sibling gate had installed -- the weld and the single must be able to be
#: mutated independently or a mutation leg cannot attribute a divergence to either.
def _certified_curl_source() -> str:
    return _kz._step_D_special_kz_real_kernel_code  # noqa: SLF001


#: The beta parameters the certified special_kz signature appends, and which this
#: weld's signature appends in the same place and the same order.
_BETA_PARAMETERS = ("    float beta_plus, float beta_minus\n")

#: THE DECLARATION, SPELLED AS A LITERAL. Everything after the opening parenthesis is
#: taken from :func:`.fused_hd_pair.signature` so the two products' parameter lists
#: cannot drift; the name is written out here because the partition reader needs to
#: see it in this file's own text.
_SIGNATURE_HEAD = '\nextern "C" __global__ void fused_hd_pair_special_kz_real(\n'


def signature() -> str:
    """This kernel's parameter list: the real sibling's, plus the two beta scalars.

    THE PARAMETER TEXT IS THE RELEASED PRODUCT'S OWN, cut off its signature at the
    opening parenthesis rather than retyped, so a change to what
    :mod:`.fused_hd_pair` binds arrives here as a compile error or an anchor failure
    instead of as a silent divergence between two hand-kept lists.
    """
    released = _hd.signature()
    marker = f"void {_hd.KERNEL_NAME}(\n"
    if released.count(marker) != 1:
        raise AssertionError(
            f"the released real H->D signature declares {released.count(marker)} "
            f"copies of {marker!r}, not one; this weld takes its parameter list from "
            f"that text and has no anchor to cut it on")
    parameters = released.split(marker, 1)[1]
    tail = "\n) {\n"
    if not parameters.endswith(tail):
        raise AssertionError(
            "the released real H->D signature no longer closes on its own line; this "
            "weld appends the beta scalars immediately before that close")
    body = parameters[: -len(tail)]
    return (_SIGNATURE_HEAD + body + ",\n"
            "    // The two complex64-free beta coefficients, one per call-site sign,\n"
            "    // rounded ONCE on the host at stepping.py:784. Appended LAST so this\n"  # stepping.py live lines for the frozen device-text citation(s) in this string: 784->811
            "    // signature is the released real weld's with the certified\n"
            "    // special_kz curl's own two-scalar tail, and a reader can diff them.\n"
            + _BETA_PARAMETERS + ") {\n")


def welded_curl_tail(source: Optional[str] = None) -> Tuple[str, str]:
    """``(head, tail)`` of a certified ``step_D`` body, every H read redirected.

    PARAMETERISED ON THE CURL TEXT, and that parameter is the point rather than a
    convenience: applied to the CERTIFIED real ``step_D`` this function returns
    exactly what :func:`.fused_hd_pair.welded_curl_tail` returns, which
    ``test_special_kz_fused_hd_pair.py`` asserts as a byte equality. The beta weld is
    therefore the released transform plus the two beta assertions below, MEASURED
    rather than claimed.

    ``head`` is the certified preamble, the stride constants and the index decode,
    lifted VERBATIM and kept as the fused kernel's own. ``tail`` is everything below
    it with the six own-cell magnetic loads taken from registers and the six shifted
    ones recomputed.
    """
    text = _certified_curl_source() if source is None else source
    body = _hd._split_body(text, _kz._REAL_PML_PRELUDE,  # noqa: SLF001
                           "the certified step_D source this weld lifts")
    if (body.count(_hd._THREAD_PREAMBLE) != 1  # noqa: SLF001
            or body.count(_hd._INDEX_DECODE) != 1):  # noqa: SLF001
        raise AssertionError(
            "the certified step_D body no longer carries exactly one thread preamble "
            "and one index decode; this weld keeps both verbatim and splices its "
            "constitutive half directly below the decode")
    cut = body.index(_hd._INDEX_DECODE) + len(_hd._INDEX_DECODE)  # noqa: SLF001
    head, tail = body[:cut], body[cut:]

    own = 0
    for component, target in enumerate(H_TARGETS):
        for variable in ("f1", "f2"):
            old = f"        float {variable} = {target}[idx];\n"
            if old not in tail:
                continue
            tail = _hd._needle(  # noqa: SLF001
                tail, old, f"        float {variable} = own_h[{component}];\n",
                f"{target}'s own-cell load into {variable}")
            own += 1
    if own != OWN_LOAD_EDITS:
        raise AssertionError(
            f"the certified step_D body makes {own} own-cell magnetic loads of the "
            f"shape this weld redirects, not {OWN_LOAD_EDITS}; a load left standing "
            f"would read the PRE-launch H -- and on this family one of them is the "
            f"BETA PARTNER, whose staleness is invisible in every magnitude")

    taps = 0
    for line in [line + "\n" for line in tail.splitlines() if "shift_dn(" in line]:
        head_text, _, rest = line.partition("shift_dn(")
        pointer, _, arguments = rest.partition(", ")
        if pointer not in H_TARGETS:
            raise AssertionError(
                f"the certified step_D body reads {pointer!r} through shift_dn; this "
                f"weld resolves exactly the three magnetic components {H_TARGETS}")
        if not arguments.rstrip("\n").endswith(");"):
            raise AssertionError(
                f"the certified shift_dn call {line!r} does not close on its own line; "
                f"the weld appends its argument pack to that closing line")
        component = H_TARGETS.index(pointer)
        carried = arguments.rstrip("\n")[: -len(");")]
        tail = _hd._needle(  # noqa: SLF001
            tail, line,
            f"{head_text}shift_dn_recompute({component}, {carried}, weld);\n",
            f"the shifted magnetic load {line.strip()!r}")
        taps += 1
    if taps != HALO_TAPS:
        raise AssertionError(
            f"the certified step_D body makes {taps} shifted magnetic loads, not "
            f"{HALO_TAPS}; this weld redirects every one of them and a missed tap "
            f"would read a word another block is writing")
    for target in H_TARGETS:
        if f"{target}[" in tail:
            raise AssertionError(
                f"the welded curl half still reads {target}; in this signature that "
                f"pointer is the PRE-LAUNCH magnetic field and every read must be the "
                f"register or a recompute")
    if "shift_dn(" in tail:
        raise AssertionError(
            "the welded curl half still calls the certified shift_dn, which loads a "
            "stored H; every shifted tap must go through shift_dn_recompute")
    _assert_the_beta_partner_is_the_weld_register(tail, source is not None)
    return head, tail


def _assert_the_beta_partner_is_the_weld_register(tail: str, lifted: bool) -> None:
    """Every surviving beta line consumes a register THIS WELD redirected.

    THE ONE CLAUSE THIS FAMILY OWNS AND NO SIBLING H->D WELD HAS. The beta term's
    partner is the SAME-CELL component snapshot (stepping.py:437), which the array
    path evaluates after ``update_H``; welded, that snapshot must be the weld's own
    ``own_h[...]`` register and never a load of the pre-launch volume. The redirect
    above rewrites the DECLARATION and leaves the beta line's spelling alone, so this
    checks the pairing rather than the line: per target block, each ``beta_*`` factor
    names a register declared ``= own_h[...]`` in that same block.

    THE SITE COUNT IS A WELD ON BOTH PATHS, and only its EMPTY case is conditional. A
    lifted text carrying ZERO beta lines is the certified beta-FREE ``step_D`` -- the
    input the released-transform equality check hands in -- and is accepted; anything
    else, lifted or shipped, must carry exactly :data:`BETA_INSERT_LINES`. A body that
    lost ONE of its two compiles, runs and is wrong on one component, and would slip
    through a rule that only checked the shipped path.
    """
    blocks = tail.split("\n    }\n")
    seen = 0
    for block in blocks:
        declared = {}
        for line in block.splitlines():
            stripped = line.strip()
            for variable in ("f1", "f2"):
                if stripped.startswith(f"float {variable} = "):
                    declared[variable] = stripped[len(f"float {variable} = "):]
        for line in block.splitlines():
            stripped = line.strip()
            if not stripped.startswith("curl = curl - (beta_"):
                continue
            seen += 1
            partner = stripped.rstrip(");").rsplit("* ", 1)[-1]
            source_of = declared.get(partner)
            if source_of is None:
                raise AssertionError(
                    f"the beta line {stripped!r} names the register {partner!r}, "
                    f"which its own target block does not declare; this weld cannot "
                    f"vouch that the beta partner is the post-update_H field")
            if not source_of.startswith("own_h["):
                raise AssertionError(
                    f"the beta line {stripped!r} consumes {partner!r}, declared as "
                    f"{source_of!r} rather than from this launch's own_h register; "
                    f"the beta partner would then be the PRE-launch magnetic field, "
                    f"one sub-step behind on exactly two of the six curl terms")
    if seen != BETA_INSERT_LINES and not (lifted and seen == 0):
        raise AssertionError(
            f"the certified special_kz step_D body carries {seen} beta increments, "
            f"not {BETA_INSERT_LINES}; a body that lost one compiles, runs and is "
            f"wrong on one component")


# ---------------------------------------------------------------------------
# The source
# ---------------------------------------------------------------------------

def kernel_source() -> str:
    """The whole device source. ONE string; every boundary kind is a runtime code."""
    head, tail = welded_curl_tail()
    weld = "\n".join([
        "",
        _hd.weld_args_construction().rstrip("\n"),
        "",
        "    // --- THE WELD: update_H, computed into registers and stored to SCRATCH.",
        "    // Nothing written here is read by this launch. The curl half below takes",
        "    // its OWN cell's magnetic field from these registers -- INCLUDING the",
        "    // two beta partners -- and recomputes every foreign tap from PRE-LAUNCH",
        "    // state through the same raw_update_H_cell, so no thread observes",
        "    // another thread's store; the launcher rotates H/f_w_H afterwards.",
        "    float own_h[3];",
        "    float own_w[3];",
        "    raw_update_H_cell(idx, weld, own_h, own_w);",
        "    Hx_out[idx] = own_h[0]; Hy_out[idx] = own_h[1]; Hz_out[idx] = own_h[2];",
        "    f_w_Hx_out[idx] = own_w[0]; f_w_Hy_out[idx] = own_w[1];",
        "    f_w_Hz_out[idx] = own_w[2];",
        "",
        "    // --- step_D (stepping.step_D / _apply_pml_update) with the special_kz",
        "    // beta insert, the certified body from its own stride constants down,",
        "    // with the twelve magnetic reads redirected and NOTHING else touched.",
        "",
    ])
    source = (_hd.curl_prelude() + _hd.constitutive_prelude() + _hd.weld_args_struct()
              + _hd.raw_update_H_cell_source() + _hd.resolution_source()
              + _hd.shift_dn_recompute_source() + signature() + head + weld + tail
              + "}\n")
    _assert_the_measured_absences(source)
    return source


#: The two CuPy spellings this backend has measured, asserted here as ABSENCES. Both
#: are recorded rather than omitted, because a reader arriving from the complex or the
#: cylindrical H->D products will look for a site for each and must be told there is
#: none -- and because carrying one backend's or one storage class's prescription into
#: the other is the defect the divide spellings were measured to be.
#: THE ONLY DIVISIONS ANY TEXT THIS FAMILY EMITS MAY CONTAIN: the certified index
#: decode's two INTEGER lines, which turn a flat cell index into the per-axis
#: coefficient index. Enumerated as exact text rather than filtered by a type
#: heuristic, so a floating-point divide arriving anywhere is a named failure.
_PERMITTED_DIVIDE_LINES: Tuple[str, ...] = (
    "    int j = (idx / nz) % ny;",
    "    int i = idx / (ny * nz);",
)

#: How many copies of the decode a complete source carries: the fused kernel's own
#: preamble and ``raw_update_H_cell``'s lifted one, which is what indexes the absorber
#: profile at a RECOMPUTED cell. Two copies, two divide lines each.
_PERMITTED_DIVIDES = 2 * len(_PERMITTED_DIVIDE_LINES)


def _assert_the_measured_absences(source: str) -> None:
    """Two ABSENCES, asserted rather than argued -- see the module docstring.

    THE COMPLEX TYPE, because this family is real float32 storage throughout and the
    ``complex64 * float32`` contraction question therefore has no site here; and
    FLOATING-POINT DIVISION, because the reciprocal this recurrence needs is ``sinv``,
    computed host-side by ``PML``. The integer index decode is the one exception and
    it is ENUMERATED, not inferred; its count is asserted too, so a dropped decode --
    the ``coefficient_index_is_the_threads_own`` defect -- fails here as well as in
    the mutation battery.
    """
    if "cf " in source or "cf_load" in source or "cuFloatComplex" in source:
        raise AssertionError(
            "this family is REAL float32 storage throughout and its text carries a "
            "complex type; the complex64 * float32 contraction question has no site "
            "here and a text that acquired one would need the complex family's "
            "measured spelling rather than this one's")
    permitted = 0
    for line in source.splitlines():
        stripped = line.split("//", 1)[0].rstrip()
        if "/" not in stripped.replace("/*", "").replace("*/", ""):
            continue
        if stripped in _PERMITTED_DIVIDE_LINES:
            permitted += 1
            continue
        raise AssertionError(
            f"the welded source divides outside the certified integer index decode: "
            f"{line.strip()!r}. This family performs NO floating-point division -- the "
            f"reciprocal its recurrence needs is sinv, computed host-side by PML -- "
            f"and a divide appearing here would have to be measured against float32 "
            f"TRUE division before it could ship")
    if permitted != _PERMITTED_DIVIDES:
        raise AssertionError(
            f"the welded source carries {permitted} integer index-decode divide lines, "
            f"not {_PERMITTED_DIVIDES}; the decode is what indexes the absorber "
            f"profile at the RECOMPUTED cell, and a missing copy is the half-cell "
            f"class of error -- converged, smooth and wrong")


def device_sources() -> Dict[str, str]:
    """``{kernel name: source}`` -- the shape every family on this track publishes."""
    return {KERNEL_NAME: kernel_source()}


def source_digest() -> str:
    """One sha256 over the family's single device string."""
    import hashlib  # noqa: PLC0415 - stdlib, imported at the one call site

    return hashlib.sha256(kernel_source().encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# The compile memo
# ---------------------------------------------------------------------------

def _get_kernel(source: Optional[str] = None,
                options: Optional[Tuple[str, ...]] = None):
    """Compile on first use, memoized in the PACKAGE'S SHARED MEMO.

    THE SHARED MEMO AND NOT A PRIVATE DICT, which is a measurement property rather
    than tidiness: every gate on this track counts launches by wrapping
    ``compile_cache``'s memo, and a family that kept its own dict would be INVISIBLE
    to that counter -- its launches would read as zero and a launch-structure leg
    would pass a weld that never executed.

    THE SOURCE AND THE OPTIONS ARE IN THE KEY, so a mutation harness that handed in a
    rewritten string cannot be served the shipped binary and the contraction-guard
    control cannot be served the guarded build. ``cupy`` is imported HERE.
    """
    import cupy as cp  # noqa: PLC0415 - device-only

    code = kernel_source() if source is None else source
    build = _COMPILE_OPTIONS if options is None else tuple(options)
    key = _kernel_cache_key(KERNEL_NAME, False, build, code)
    return _get_or_compile(key, lambda: cp.RawKernel(code, KERNEL_NAME, options=build))


def _clear_kernel_cache() -> int:
    return _clear_cache()


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------

def covers_special_kz_fused_hd_pair(fields: Any, pml: Any, grid: Any,
                                    sources: Any = None) -> Tuple[bool, str]:
    """May ONE launch span ``update_H`` -> the electric withdraw -> ``step_D`` here?

    A CONJUNCTION OF THE TWO CERTIFIED HALVES' OWN PREDICATES plus this seam's
    withdraw clause, the shared boundary triple and the rotation's six volumes --
    never a re-derivation of either half's clause set, which is what keeps this
    product from admitting a configuration a half refuses:

    * the H side: ``special_kz_curl.covers_special_kz_constitutive(..., "H")``, which
      is the CERTIFIED real constitutive predicate with the beta clause inverted;
    * the D side: ``special_kz_curl.covers_special_kz_curl(..., "step_D")``, the
      certified real curl predicate with the same clause inverted;
    * the seam: ``withdraw_hoist.seam_withdraw_reasons``, refusing BY NAME every row
      carrying a standing integrated electric withdraw;
    * the rotation: the six volumes :data:`SCRATCH_VOLUMES` names.

    THE ORDER IS THE DRIVER'S: the constitutive half is asked FIRST because it runs
    first (driver.py:3311), so the first refusal a reader sees names the half the
    driver would have reached first.

    BOTH HALVES ASK THE SAME BETA CLAUSES, and that is deliberate rather than
    redundant: ``_beta_reasons`` refuses a zero beta, complex storage, an off-diagonal
    epsilon, a cylindrical grid and a non-2-D grid, and asking it twice is what makes
    a future edit to ONE of the two sub-step predicates a visible disagreement here
    rather than a widening.
    """
    covered, reason = _kz.covers_special_kz_constitutive(fields, pml, grid, "H")
    if not covered:
        return False, f"constitutive half: {reason}"
    covered, reason = _kz.covers_special_kz_curl(fields, pml, grid, "step_D")
    if not covered:
        return False, f"curl half: {reason}"

    # THE SEAM'S ONE PASS. Nothing is INJECTED between the two consults, so
    # `deposit_repair` is not consulted at all and a source of either polarity is not
    # this seam's business. What IS between them is the electric integrated-source
    # withdraw (driver.py:3313-3314). IGNORANCE IS NEVER AN EMPTY SET: `Fields` does
    # not hold the source list, so a predicate that inferred "no withdraw stands"
    # from not being told would be the over-covering refusal this clause prevents.
    seam_reasons = _withdraw_hoist.seam_withdraw_reasons(
        fields, sources,
        undeclared=(
            "the source set was not declared: this predicate cannot infer from Fields "
            "that no electric withdraw stands between update_H and step_D"),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) has a standing integrated "
            f"electric withdraw, which the driver runs BETWEEN the update_H and "
            f"step_D consults (driver.py:3313-3314); this product declares "
            f"HOISTS_THE_WITHDRAW = False because it declares INSTALLABLE = False, so "
            f"fused_pairs._install_fused_pair's withdraw-hoist branch is unreachable "
            f"for it and nothing would perform the withdraw before the launch"),
        hoists_the_withdraw=HOISTS_THE_WITHDRAW,
        span=REPLACES)
    if seam_reasons:
        return False, seam_reasons[0]

    # ONE BOUNDARY TRIPLE FOR BOTH HALVES, CHECKED. The curl resolves a fold's
    # termination into BC_MIRROR_PERIODIC or BC_METALLIC and the constitutive half
    # reads no boundary at all, so the triple this launch binds is the CURL's -- but
    # the resolution must succeed, and a grid it refuses is a grid this launch must
    # not bind.
    codes, refusal = _coverage.real_curl_boundary_codes(grid)
    if refusal is not None:
        return False, (f"the curl's own boundary resolution refuses this grid: "
                       f"{refusal}")
    if len(tuple(codes)) != 3:
        return False, (f"the curl's boundary resolution returned {codes!r}, which is "
                       f"not the per-axis triple this launch binds")

    # THE SUB-LATTICE, ASSERTED RATHER THAN ASSUMED. This kernel binds ONE kms group
    # for both halves, which is only correct while step_D's curl and update_H read the
    # SAME Yee sub-lattice. Both read the integer positions today. If either moved,
    # sharing the group would bind one half's coefficients to the other half's lattice
    # -- a converged, smooth, half-cell-wrong absorber profile rather than a failure.
    if constitutive_sub_lattice("H"):
        return False, ("update_H now reads the HALF-INTEGER PML sub-lattice while "
                       "step_D's curl reads the integer one; this weld binds ONE kms "
                       "group for both halves and may only do so while they agree")

    # THE BETA COEFFICIENTS THE LAUNCH BINDS, asked here rather than left to raise at
    # launch time: a grid whose beta and dt cannot be rounded to two float32 scalars
    # is a refusal with a line behind it.
    try:
        beta_scalars(grid)
    except Exception as error:  # noqa: BLE001 - a raise here is a refusal
        return False, (f"the beta coefficients this launch binds cannot be derived "
                       f"from this grid: {type(error).__name__}: {error}")

    for name in SCRATCH_VOLUMES:
        if getattr(fields, name, None) is None:
            return False, (f"{name} is not allocated; this weld rotates it against a "
                           f"launch-local scratch twin after every launch")
    return True, "covered"


# ---------------------------------------------------------------------------
# The launch
# ---------------------------------------------------------------------------

def beta_scalars(grid: Any) -> Tuple[float, float]:
    """``(plus, minus)`` -- ``special_kz_curl.beta_curl_coefficients``, unchanged.

    CALLED, never re-transcribed: stepping.py:797 and :811 are that function's
    transcription and this weld binds the same two words the certified single binds.
    The ``+-1j`` branch at :771-772 is NOT taken under real storage, so both sub-steps
    take the same plain real coefficient -- which is why there is one pair here and
    not one per side.
    """
    return _kz.beta_curl_coefficients(grid.beta, grid.dt)


def special_kz_fused_hd_pair_tables(pml: Any) -> Dict[str, Any]:
    """The ONE coefficient group both halves read -- the released sibling's resolver.

    ``fused_hd_pair.fused_hd_pair_tables`` already asserts the sharing by base
    address, which is what the shared binding in :func:`signature` rests on; calling
    it rather than re-spelling it is what keeps the two products from ever disagreeing
    about which sub-lattice this seam reads. The beta insert adds no table: its two
    coefficients are host scalars.
    """
    return _hd.fused_hd_pair_tables(pml)


def special_kz_fused_hd_pair_codes(grid: Any) -> Tuple[int, ...]:
    """The boundary triple this launch binds, from the CURL's own resolution."""
    return _hd.fused_hd_pair_codes(grid)


def special_kz_fused_hd_pair_scratch(fields: Any) -> Dict[str, Any]:
    """The six launch-local volumes -- the released sibling's allocator, reused."""
    return _hd.fused_hd_pair_scratch(fields)


def assert_scratch_is_disjoint(fields: Any, scratch: Dict[str, Any],
                               tables: Dict[str, Any]) -> int:
    """THE CHECK THE WHOLE DESIGN RESTS ON -- the released sibling's, reused.

    Same bindings, same shared ``kms`` coincidence, same two properties: no scratch
    volume is any bound input (or the launch is the IN-PLACE weld the board refused),
    and no two ``__restrict__`` arguments are one allocation. The beta scalars are
    values, not pointers, and add nothing to inspect.
    """
    return _hd.assert_scratch_is_disjoint(fields, scratch, tables)


def launch_special_kz_fused_hd_pair(fields: Any, scratch: Dict[str, Any],
                                    tables: Dict[str, Any],
                                    boundary_codes: Sequence[int],
                                    beta_coefficients: Tuple[float, float],
                                    dtdx: float,
                                    kernel: Optional[Any] = None,
                                    threads: int = _FUSED_THREADS) -> Dict[str, Any]:
    """Both of :data:`REPLACES` in ONE launch. THE CALLER ROTATES AFTERWARDS.

    The argument list is the released real weld's with the two beta words APPENDED,
    which is what lets a reader diff the two launches.

    ``kernel`` IS THE GATE'S DOOR: a gate compiles a deliberately broken copy of the
    shipped source and hands it here. ``threads`` is the gate's schedule door -- the
    design's whole claim is that the answer does not depend on which block ran first,
    and a block size that never moves cannot expose the opposite.
    """
    nx, ny, nz = (int(n) for n in fields.Dx.shape)
    blocks = (nx * ny * nz + threads - 1) // threads
    arguments: List[Any] = [scratch[name] for name in SCRATCH_VOLUMES]
    arguments += [getattr(fields, name) for name in
                  ("Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz",
                   "Bx", "By", "Bz",
                   "Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz")]
    arguments += [np.int32(nx), np.int32(ny), np.int32(nz), np.float32(dtdx)]
    for axis in ("x", "y", "z"):
        arguments += [tables["kms"][axis], tables["sinv"][axis]]
    arguments += [tables["kps"][axis] for axis in ("x", "y", "z")]
    arguments += [np.int32(int(code)) for code in boundary_codes]
    plus, minus = beta_coefficients
    arguments += [np.float32(plus), np.float32(minus)]
    (kernel or _get_kernel())((blocks,), (threads,), tuple(arguments))
    return {"launched": True, "blocks": blocks, "threads": threads,
            "elements": nx * ny * nz, "replaces": REPLACES,
            "kernel": KERNEL_NAME,
            "boundary_codes": tuple(int(code) for code in boundary_codes),
            "beta_coefficients": (float(plus), float(minus))}


def rotate_into_fields(fields: Any, scratch: Dict[str, Any]) -> Dict[str, Any]:
    """THE ROTATION -- ``fused_hd_pair.rotate_into_fields``, reused verbatim.

    The consumer audit it rests on is that module's, and it is a statement about the
    ENGINE rather than about a storage class or a beta term: every consumer of the
    magnetic field resolves it BY NAME at use time, so a rebinding between steps is
    invisible to all of them. Calling it here rather than re-spelling it is what keeps
    the products on this seam from ever disagreeing about what a rotation covers.
    """
    return _hd.rotate_into_fields(fields, scratch)


def run_special_kz_fused_hd_pair(fields: Any, grid: Any, pml: Any, dtdx: float, *,
                                 scratch: Optional[Dict[str, Any]] = None,
                                 sources: Any = None,
                                 tables: Optional[Dict[str, Any]] = None,
                                 beta_coefficients: Optional[Tuple[float, float]] = None,
                                 kernel: Optional[Any] = None,
                                 threads: int = _FUSED_THREADS,
                                 rotate: bool = True) -> Dict[str, Any]:
    """Gate on the predicate, resolve everything from the grid, launch once, rotate.

    THE PREDICATE IS ASKED FIRST AND A REFUSAL IS RETURNED, NOT RAISED, because the
    caller's correct response to a configuration this product does not carry is the
    array path -- never an exception into a stepper that would otherwise have stepped
    correctly.

    ``scratch``, ``tables``, ``beta_coefficients``, ``kernel``, ``threads`` and
    ``rotate`` are the gate's doors, keyword-only. ``rotate=False`` is how the
    ``rotation_skipped`` mutation is armed, and ``beta_coefficients`` is how
    ``swap_beta_signs`` is.
    """
    covered, reason = covers_special_kz_fused_hd_pair(fields, pml, grid, sources)
    if not covered:
        return {"launched": False, "reason": reason}
    if tables is None:
        tables = special_kz_fused_hd_pair_tables(pml)
    if scratch is None:
        scratch = special_kz_fused_hd_pair_scratch(fields)
    if beta_coefficients is None:
        beta_coefficients = beta_scalars(grid)
    assert_scratch_is_disjoint(fields, scratch, tables)
    record = launch_special_kz_fused_hd_pair(
        fields, scratch, tables, special_kz_fused_hd_pair_codes(grid),
        beta_coefficients, dtdx, kernel, threads)
    record["rotated"] = bool(rotate)
    record["scratch"] = rotate_into_fields(fields, scratch) if rotate else scratch
    return record
