"""The H->D weld on a COMPLEX64 ``grid.beta`` run: ``update_H`` welded into ``step_D``.

The complex-beta sibling of :mod:`.complex_fused_hd_pair`, on the same seam, with the
same shape -- ONE launch, scratch output, foreign-cell recompute, a rotation after --
and the same two flags (:data:`INSTALLABLE` False, :data:`HOISTS_THE_WITHDRAW`
False). Built in the same round as :mod:`.special_kz_fused_hd_pair`, which is the
REAL-storage beta H->D weld; the two are SEPARATE PRODUCTS and the module docstring
says below what was measured to make them so.

=============================================================================
THE CELL, AND WHY IT IS ONE TEXT RATHER THAN TWO VARIANTS
=============================================================================

``parity/meep_gpu/results/fusion_matrix_cuda_2026-09-07_wired/fusion_matrix_cuda.json``,
``h_to_d_seam.instances`` filtered to ``buildable_not_built``:

* ``(update_H cuda_complex_beta/complex beta -> step_D cuda_complex_beta/complex
  beta)`` -- **4 instances**: ``tests:TestSpecialKz.test_special_kz``,
  ``tests:TestSpecialKz.test_eigsrc_kz_0_complex`` and the two
  ``tests:TestEigCoeffs.test_binary_grating_special_kz_*`` legs.

THE COMPLEX SIBLING NEEDED TWO VARIANTS AND THIS ONE NEEDS NONE, because
``complex_beta_kernels`` already made that decision and recorded the arithmetic
behind it: its device source is the FOLDED complex curl with the beta insert, so the
fold branches are RUNTIME boundary codes and are simply dead on an unfolded grid.
Three of this cell's four rows are folded (Mirror(Y) with a periodic termination)
and one is not, and one text serves all four. :data:`KERNEL_NAME` is therefore a
single name and :func:`kernel_source` a single string per arm, not a variant map.

THE UPDATE_H HALF IS THE CERTIFIED COMPLEX CONSTITUTIVE, unchanged.
``complex_beta_kernels`` ships NO constitutive kernel:
``covers_complex_beta_constitutive`` is an ADMISSION over the certified complex pair,
because ``stepping.update_H`` (:907-925) through ``_apply_constitutive_pml`` (:2065)
reads no ``grid.beta`` and no array the beta term writes -- and that admission has
its own device leg in ``gate_cuda_complex_beta.py`` rather than resting on the
reading. The census records ``cuda_complex_beta/complex beta`` at ``update_H`` on all
four rows because that family's PREDICATE answers there; the kernel it admits is the
ordinary certified complex one. So this weld's constitutive half is byte-for-byte
:func:`.complex_fused_hd_pair.raw_update_H_cell_source`, REUSED rather than re-lifted.

=============================================================================
ONE PRODUCT PER STORAGE CLASS, AND THAT WAS MEASURED
=============================================================================

The round that built this module targeted six seam-instances over two cells -- four
here and two on ``cuda_special_kz/real beta``. The complex H->D product's precedent
for one-product-two-cells is that its two cells SHARE their H half exactly and its
two D halves are one text through three anchored deltas. Neither holds across the
storage boundary, and it was checked rather than assumed
(``test_complex_beta_fused_hd_pair.py``): the two cells' certified ``update_H``
bodies are different strings (``update_H_pml_complex_bloch`` against
``update_H_pml_real``), their curl preludes are different strings
(``complex_emitter``'s ``cf`` prelude against ``step_curl_kernels``'
``_REAL_PML_PRELUDE``), and the scalar type differs in every expression. A single
transform over both would have been two transforms wearing one name. TWO PRODUCTS,
one per storage class -- 4 instances here, 2 there.

=============================================================================
THE SHAPE: SCRATCH OUTPUT AND FOREIGN-CELL RECOMPUTE
=============================================================================

Identical in structure AND in justification to :mod:`.complex_fused_hd_pair`, so it
is not restated at length. ``step_D``'s curl reads ``H`` at the thread's own cell and
at three BACKWARD neighbours (``cshift_dn``) while ``update_H`` writes ``H`` and
``f_w_H`` -- and on ``f_w_H`` especially, since the newly stored ``f_w_H`` IS ``B``
exactly, so an in-place write hands a racing neighbour ``B`` where its recurrence
needs ``B_prev``. So the constitutive half writes NOTHING in place (``H_new`` and
``f_w_H_new`` go to launch-local scratch; ``H``, ``f_w_H`` and ``B`` are ``const``
for the whole launch), every foreign read is a RECOMPUTE of the certified pointwise
``update_H`` from that same unwritten state, and the launcher ROTATES the
``H``/``f_w_H`` bindings afterwards (:func:`.complex_fused_hd_pair.rotate_into_fields`,
reused rather than re-spelled).

WHERE THE BLOCH PHASE SITS, AND WHY THE RECOMPUTE DOES NOT CARRY ONE. The phase is
applied by ``cshift_dn`` to the WRAPPED LANE ONLY and AFTER the load;
:func:`cshift_dn_recompute_source` replaces the LOAD and leaves the branch, the wrap
arithmetic, the ``ph`` guard and the ``rotate_field_left`` call exactly where the
certified helper puts them. A recompute that carried a phase of its own, or applied
one to the near-neighbour branch, would be a band structure that converges to the
wrong dispersion. THE METALLIC AND MIRROR GHOSTS ARE NOT RECOMPUTED: ``cshift_dn``
returns ``cf_zero()`` on those branches without touching ``g``, and the resolved
helper keeps that return verbatim.

=============================================================================
WHAT THIS SEAM ADDS THAT NO OTHER H->D WELD HAS: THE BETA PARTNER
=============================================================================

``stepping._special_kz_beta_term`` takes the SAME-CELL component snapshot
(stepping.py:437), and in the D-side curl that snapshot is already in a register: Dx
takes the Hy CENTRE the stencil loaded into ``f_2`` and Dy the Hx CENTRE in ``f_1``
(``complex_beta_kernels.BETA_PARTNER``, pinned there against ``D_CURL_TERMS`` and the
call sites :443/:445). The array path evaluates ``step_D`` AFTER ``update_H``, so
that snapshot is the POST-``update_H`` magnetic field.

**Welded, those two registers are this launch's own** ``own_h[1]`` **and**
``own_h[0]``, because the weld redirects every own-cell magnetic load in the body and
the beta term consumes the redirected declaration rather than a load of its own. That
is CHECKED and not assumed: :func:`curl_pieces` asserts, per target block, that each
surviving ``mul_imag_coefficient_left`` line names a register whose declaration it
has just rewritten to ``own_h[...]``. A beta term left reading the PRE-launch ``H``
would be one sub-step behind on exactly two of the six curl terms -- converged,
plausible and wrong -- and it is the armed mutation ``beta_partner_reads_stale_h``.

=============================================================================
THE ARITHMETIC IS THE CERTIFIED FAMILIES', INCLUDING ITS ARM
=============================================================================

Every complex operation comes from ``complex_emitter``'s prelude
(``cf_load``/``cf_store``, ``cf_add``/``cf_sub``, ``cf_zero``, ``rotate_field_left``,
``mul_field_left``, ``mul_coefficient_left``, ``pml_apply``, ``constitutive_apply``)
or from ``complex_beta_kernels``' one-line ``mul_imag_coefficient_left``, which is
``rotate_field_left`` with the coefficient as the first operand. NOTHING is retyped.
The arm (``NAIVE``/``FMA_V1``) is an ARGUMENT with no default, for the reason
``complex_emitter.normalized_expansion`` refuses rather than defaulting: a wrong arm
is a wrong ANSWER, not a crash.

TWO CuPy SPELLINGS MEASURED ON THIS BACKEND ARE RESPECTED RATHER THAN REDISCOVERED
(``lanes/cyl_round/cupy_probe``, 2026-09-06/07, one RTX A6000, CuPy 13.5.1, both
float32 subnormal policies):

* ``complex64 * float32`` is the four-product form CONTRACTED by NVRTC. The
  UNCONTRACTED transcription differs on the sign of a FLUSHED zero. The spelling that
  is 0 on all twelve probe cases under ``--fmad=false`` is
  ``_ARM_SOURCE[FMA_V1].mul_field_left``, which is what this kernel calls -- through
  the emitter's own text, under either arm's name binding, because the arm block is
  lifted whole.
* ``complex64 / float32`` is CuPy's SCALED complex/complex algorithm with the
  zero-valued terms kept, NOT numpy's reciprocal multiply. **This family performs NO
  DIVISION** -- the reciprocal its recurrence needs is ``sinv``, computed host-side
  by ``PML`` -- so the divide spelling has no site here, and :func:`kernel_source`
  asserts the absence. It is recorded rather than omitted because a reader arriving
  from the cylindrical H->D products (where the increment DOES divide) will look for
  it, and because the REAL beta sibling's divide is a different algorithm again
  (float32 true division) that must never be carried across.

Both facts are armed as gate mutations anyway, on a PLANTED row-0 tiny-normal class:
a random battery does not exercise the class where the wrong multiply shows.

THE BETA COEFFICIENT'S SIGNED ZERO IS PASSED THROUGH AND IS NOT CLAIMED TO MATTER.
``complex_beta_kernels.beta_curl_coefficients`` returns the words Python's own
complex multiply produced, real word included, and the real word is ``-0.0`` exactly
when the coefficient is negative on the magnetic side. THE COMPLEX BETA FAMILY'S OWN
DEVICE LEG MEASURED THE PASS-THROUGH TO BE INERT -- ``synthesise_the_zero_real_word``
came back UNCAUGHT 0/12 under BOTH policies, because ``mul_imag_coefficient_left``
consumes the imaginary word alone -- and this module inherits that finding rather
than re-arguing it. This gate re-arms the same needle on the H->D seam and records
whatever it measures; a null here is a null, not a defence.

=============================================================================
WHAT SITS IN THE SEAM, AND WHY THIS PRODUCT REFUSES THE WITHDRAW ROWS
=============================================================================

One statement sits between the two consults: ``for source in electric: getattr(source,
"withdraw", _no_withdraw)(self.fields)`` (driver.py:3313-3314). Nothing is INJECTED
here, so :data:`CARRIES_DEPOSIT_REPAIR` is False as a FACT and :mod:`..withdraw_hoist`
-- not :mod:`..deposit_repair` -- is this seam's contract.

:data:`HOISTS_THE_WITHDRAW` is False, and the predicate therefore refuses BY NAME
every row whose electric withdraw does work. On this cell's 4 instances that costs
NOTHING measurable: all four carry ``integrated_electric_sources = 0`` on the
standing board and none is in the ``withdraw_seam`` bucket. The clause is still
asked, because a predicate that admitted a configuration it cannot serve would be
wrong on the first row that acquired an integrated source.

=============================================================================
IT IS NOT INSTALLED, AND THE REASON IS THE SIBLING'S MEASURED VERDICT
=============================================================================

:data:`INSTALLABLE` is False. Over the driver's ``step_B - update_H - step_D -
update_E`` slot path launches are ``4 - (installed pairs)``, and a product spanning
``update_H``/``step_D`` takes one slot from EACH neighbour -- and on THIS cell both
neighbours are RELEASED products (``cuda_complex_beta_fused_magnetic_pair`` holds
``step_B``/``update_H``, ``cuda_complex_beta_fused_electric_pair`` holds
``step_D``/``update_E``), so the span can only TIE or LOSE. That is the Cartesian
algebra ``fused_hd_pair.INSTALLABLE_REASON`` prices and there is nothing here to
re-price. What keeps the released B->H pair in its slot is not this flag but
``fused_pairs._neighbouring_seam_claimant``'s end-edge guard together with
``_spans_may_absorb`` reading the live ``selected``; the flag is belt and braces in
front of them, and the gate's arbitration leg measures both arrangements THROUGH THE
SHIPPED COMPOSER rather than assuming either.

=============================================================================
NOT DISPATCHED, AND NOT YET IN THE COMPOSER'S TABLES
=============================================================================

Nothing under ``meep_gpu/`` imports ``cuda_kernels`` outside tests;
``fastpath.plan_fast_path`` returns ``None`` on every branch. Beyond that, this
family is not in ``fused_pairs.FUSED_PRODUCTS``, ``registry`` or ``arms`` at all: the
wiring is a separate change that a later round applies. The gate PLANS EVERY
ARRANGEMENT FROM ARRAYS and drives the array path itself; the composer building a
composition that includes THIS product is measured in-process with the wiring rows
patched in (the arbitration leg) and awaits the wiring round.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# Every import here is CuPy-FREE at module scope, so the predicate, the emitter and
# the whole transform stay callable on a laptop with no device. CuPy is imported
# inside the launchers and the compile memo, at their call sites.
from . import complex_beta_kernels as _beta
from . import complex_emitter
from . import complex_folded_kernels as _folded
from . import complex_fused_hd_pair as _chd
from .compile_cache import (clear_kernel_cache as _clear_cache,
                            get_or_compile as _get_or_compile,
                            kernel_cache_key as _kernel_cache_key)
from .. import withdraw_hoist as _withdraw_hoist

# The complex TABLE builders and the word view import CuPy at scope; taken
# defensively so this module still imports where there is none.
try:
    from . import complex_pml_kernels
except Exception:  # noqa: BLE001 - no CuPy on this host
    complex_pml_kernels = None  # type: ignore[assignment]

FAMILY = "cuda_complex_beta_fused_hd_pair"
SLOT = "update_H"
REPLACES: Tuple[str, ...] = ("update_H", "step_D")
SEAM: str = _withdraw_hoist.SEAM

#: The one kernel this family emits. ONE name and not a variant map: the beta text is
#: the FOLDED complex curl with the insert, so a fold is a runtime boundary code here
#: and the same string serves this cell's three folded rows and its one unfolded one.
#: Spelled as a literal in :data:`_SIGNATURE_HEAD` as well, because
#: ``test_kernel_partition.py`` reads shipped kernel names off this file's own text
#: with a regex that requires a C identifier after ``void``.
KERNEL_NAME = "fused_hd_pair_pml_complex_beta"

#: The six volumes the rotation swaps -- ``fused_hd_pair``'s own tuple, reused through
#: the complex sibling so no product on this seam can disagree about what a rotation
#: covers.
SCRATCH_VOLUMES: Tuple[str, ...] = _chd.SCRATCH_VOLUMES
H_TARGETS: Tuple[str, str, str] = _chd.H_TARGETS

#: The curl's own magnetic source parameters, in the emitted template's spelling.
CURL_SOURCE_NAMES: Tuple[str, str, str] = _chd.CURL_SOURCE_NAMES

#: How many shifted magnetic taps and own-cell magnetic loads the certified complex
#: beta ``step_D`` makes. Taken from the released complex weld rather than restated:
#: the beta text is the folded complex text plus an insert that adds no magnetic read
#: at all -- its partner is a register the stencil already loaded -- so the two counts
#: are the same by construction and the equality is asserted on every emit.
HALO_TAPS: int = _chd.HALO_TAPS
OWN_LOAD_EDITS: int = _chd.OWN_LOAD_EDITS

#: How many beta increments the certified complex-beta ``step_D`` carries: one on
#: target 0 and one on target 1, none on target 2 (MEEP step_db.cpp:148-176 runs
#: ``cc`` over ``d_c`` in {X, Y} only; stepping.py:472-474). A site count is a WELD.
BETA_INSERT_LINES = 2

#: NOT CERTIFIED YET. The gate for this family is
#: ``parity/meep_gpu/gate_complex_beta_fused_hd_pair.py`` and its artifacts are
#: ``parity/meep_gpu/results/complex_beta_fused_hd_pair_2026-09-07/{keep,flush}/gate.json``.
#: A name moves into the certified set only once ``certification.json`` carries a
#: block claiming it.
CERTIFIED_KERNELS: Tuple[str, ...] = ()

#: Spelled as a dict WITHOUT a type annotation, for the reason every sibling records:
#: the partition readers walk the syntax tree so they run where there is no CuPy, and
#: an annotated assignment is an ``ast.AnnAssign`` the plain-assignment readers do not
#: match -- annotating it makes the name invisible and the partition unenforced.
UNCERTIFIED_KERNELS = {
    "fused_hd_pair_pml_complex_beta":
        "gated by parity/meep_gpu/gate_complex_beta_fused_hd_pair.py; the "
        "certification block that would move this name into CERTIFIED_KERNELS is cut "
        "from that gate's artifacts by record_cuda_regate.py and is not in this change",
}

#: Nothing is injected between the two consults, so there is no deposit to bracket.
#: A fact about the driver, not a choice -- see the module docstring.
CARRIES_DEPOSIT_REPAIR = False

#: Not in this round: the only wiring that performs the hoist is
#: ``fused_pairs._install_fused_pair``'s ``withdraw_hoist.SEAM`` branch, which
#: ``_declared_uninstallable`` makes unreachable for a product declaring INSTALLABLE
#: False. True here would be a claim about wiring that cannot fire.
HOISTS_THE_WITHDRAW = False

INSTALLABLE = False
INSTALLABLE_REASON = (
    "slot arbitration, and on THIS cell the Cartesian algebra applies unchanged. Over "
    "the driver's step_B - update_H - step_D - update_E slot path launches are "
    "4 - (installed pairs); both of this cell's neighbours are RELEASED products "
    "(cuda_complex_beta_fused_magnetic_pair holds step_B/update_H, "
    "cuda_complex_beta_fused_electric_pair holds step_D/update_E), and a span taking "
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
    "launch count -- up to six extra pointwise complex constitutive evaluations per "
    "thread -- and whether the saved launch and the saved H write-then-read round trip "
    "pay for them is a HYPOTHESIS. Every launch figure in this module is a COUNT. It "
    "also licenses nothing about the beta coefficient's SIGNED ZERO real word: the "
    "complex beta family's own device leg measured that pass-through INERT and this "
    "product inherits the finding.")

_FUSED_THREADS = 256

#: ``--fmad=false`` is CORRECTNESS and is the certified complex family's own tuple,
#: restated here rather than imported so that loading this file by path cannot pick up
#: a different one than the gate compiled. The constitutive half's two separate
#: accumulations, the curl's ``((sf - f_1) + (f_2 - ss))`` grouping and the beta
#: insert's ``cf_sub(curl, c*g)`` are all contraction candidates; the arm's own
#: fusions are explicit ``__fmaf_rn`` and survive the flag.
_COMPILE_OPTIONS: Tuple[str, ...] = ("--fmad=false",)

__all__ = [
    "BETA_INSERT_LINES", "CARRIES_DEPOSIT_REPAIR", "CERTIFIED_KERNELS",
    "CURL_SOURCE_NAMES", "FAMILY", "HALO_TAPS", "HOISTS_THE_WITHDRAW", "H_TARGETS",
    "INSTALLABLE", "INSTALLABLE_REASON", "KERNEL_NAME", "OWN_LOAD_EDITS", "REPLACES",
    "SCRATCH_VOLUMES", "SEAM", "SLOT", "UNCERTIFIED_KERNELS",
    "WHAT_A_RELEASE_DOES_NOT_LICENSE", "assert_bindings_are_disjoint",
    "beta_coefficients", "constitutive_prelude", "covers_complex_beta_fused_hd_pair",
    "cshift_dn_recompute_source", "curl_pieces", "current_source", "device_sources",
    "kernel_source", "launch_complex_beta_fused_hd_pair", "reset_kernel_sources",
    "resolve", "rotate_into_fields", "run_complex_beta_fused_hd_pair",
    "set_kernel_source", "signature", "source_digest",
]


def _arm(expansion) -> int:
    return complex_emitter.normalized_expansion(expansion)


def _curl_source(expansion) -> str:
    """The certified ``step_D_pml_complex_beta`` text this weld lifts.

    Taken from :func:`complex_beta_kernels.beta_source` -- the pure emitter -- rather
    than from that module's ``kernel_source``, which serves the GATE'S MUTABLE COPY. A
    weld that read the mutable one would silently inherit whatever mutation a sibling
    gate had installed, and the weld and the single must be mutable independently or a
    mutation leg cannot attribute a divergence to either.
    """
    return _beta.beta_source("step_D", _arm(expansion))


# ---------------------------------------------------------------------------
# The device source
# ---------------------------------------------------------------------------

def constitutive_prelude(expansion) -> str:
    """The beta variant's certified prelude with ``constitutive_apply`` made PURE.

    THE PRELUDE IS THE BETA CURL SOURCE'S OWN, which is what carries the folded
    complex prelude's two deltas -- ``#define BC_MIRROR_PERIODIC 2`` and the widened
    ghost -- AND ``complex_beta_kernels``' own ``mul_imag_coefficient_left`` into the
    weld without this module restating any of the three. The four anchored edits on
    top are :data:`.complex_fused_hd_pair.CONSTITUTIVE_LIFT_EDITS`, applied through
    that module's own table so the two products cannot make ``constitutive_apply``
    pure in two different ways; the arithmetic is untouched.
    """
    prelude, _ = _chd._split_at_kernel(  # noqa: SLF001
        _curl_source(expansion), "the certified complex-beta step_D source")
    for old, new in _chd.CONSTITUTIVE_LIFT_EDITS:
        prelude = _chd._needle(prelude, old, new,  # noqa: SLF001
                               "constitutive_apply's lifted text")
    return prelude


def cshift_dn_recompute_source(expansion) -> str:
    """The certified ``cshift_dn`` with its two leaf LOADS RESOLVED.

    THE BRANCHES AND THE INDEX EXPRESSIONS ARE THE CERTIFIED HELPER'S OWN, parsed out
    of the prelude this family emits rather than retyped: the near-neighbour branch,
    the ghost return, the wrap arithmetic, the ``ph`` guard and the
    ``rotate_field_left`` call are carried across character for character, and only
    ``cf_load(g, EXPR)`` becomes ``resolve_H(comp, EXPR, weld)``.

    That the FOLDED widened ghost (``bc == BC_METALLIC || bc == BC_MIRROR_PERIODIC``)
    arrives here automatically is the whole reason this function reads the beta
    prelude rather than ``complex_emitter._TAIL``: a folded ghost that recomputed a
    magnetic field instead of returning ``cf_zero()`` would be wrong on one plane of
    one component and would converge.
    """
    prelude, _ = _chd._split_at_kernel(  # noqa: SLF001
        _curl_source(expansion), "the certified complex-beta step_D source")
    signature_text = _chd._CSHIFT_DN_SIGNATURE  # noqa: SLF001
    if prelude.count(signature_text) != 1:
        raise AssertionError(
            f"the certified complex-beta prelude carries "
            f"{prelude.count(signature_text)} copies of cshift_dn's declaration, not "
            f"one; the recompute has no anchor")
    body = prelude.split(signature_text, 1)[1]
    if not body.startswith(" {\n"):
        raise AssertionError(
            "the certified cshift_dn no longer opens its body on the declaration's "
            "own line; this weld lifts that body and cannot find it")
    body = body[len(" {\n"):]
    end = body.find("}\n")
    if end < 0:
        raise AssertionError("the certified cshift_dn does not close")
    # EVERYTHING PAST THE CLOSING BRACE IS DROPPED, and dropping it is load-bearing
    # rather than tidy: the rest of the prelude is ``pml_apply``,
    # ``constitutive_apply`` and ``mul_imag_coefficient_left``, which
    # :func:`constitutive_prelude` already emits. Carrying them a second time is a
    # duplicate definition NVRTC refuses outright.
    body = body[:end]
    if body.count("cf_load(g,") != 2:
        raise AssertionError(
            f"the certified cshift_dn makes {body.count('cf_load(g,')} loads of g, "
            f"not 2; this weld resolves exactly the near-neighbour load and the "
            f"periodic wrap")
    for _ in range(2):
        head, expression, tail = _chd._balanced_call(body, "cf_load(g,")  # noqa: SLF001
        body = f"{head}resolve_H(comp,{expression}, weld){tail}"
    if "g[" in body or "(g," in body or " g," in body:
        raise AssertionError(
            "the resolved cshift_dn still reads the pointer g; in this helper there "
            "is no such parameter and every magnetic read must be a recompute")
    return ("\n// stepping._shift_down's certified device helper, with the leaf loads\n"
            "// RESOLVED: the value the curl needs at a backward neighbour is not a\n"
            "// stored word but update_H's result there, recomputed from pre-launch\n"
            "// state. The branch, the ghost return, the wrap arithmetic and the Bloch\n"
            "// rotation ON THE WRAPPED LANE ONLY are the certified helper's own -- the\n"
            "// recompute carries no phase of its own and none is applied to the\n"
            "// near-neighbour branch.\n"
            + _chd._CSHIFT_DN_SIGNATURE_RECOMPUTE + " {\n" + body + "}\n")  # noqa: SLF001


def curl_pieces(expansion, source: Optional[str] = None) -> Tuple[str, str]:
    """``(head, tail)`` of a certified complex ``step_D`` body, every H read redirected.

    PARAMETERISED ON THE CURL TEXT, and that parameter is the point rather than a
    convenience: applied to ``complex_emitter.complex_source("step_D", arm)`` this
    function returns exactly what ``complex_fused_hd_pair.curl_pieces("plain", arm)``
    returns, and applied to ``complex_folded_kernels.folded_source("step_D", arm)``
    exactly what that function's ``"folded"`` variant returns. Both equalities are
    asserted BYTE FOR BYTE by ``test_complex_beta_fused_hd_pair.py``, on a host with
    no CuPy, so the beta weld is the RELEASED transform plus the two beta assertions
    below -- measured rather than claimed.

    ``head`` is the certified preamble, the stride constants, the index decode, the
    three phase constants AND the two beta coefficient registers, lifted VERBATIM.
    ``tail`` is the three target blocks with the six own-cell magnetic loads taken
    from registers and the six shifted ones recomputed. The ownership mask -- the
    folded three-arm block, which the beta text inherits -- rides in ``tail``
    untouched: it zeroes ``curl``, never a magnetic read, and the beta increment sits
    ABOVE it exactly where the array path puts it (:443/:445 after :429, before :450).
    """
    text = _curl_source(expansion) if source is None else source
    _, kernel = _chd._split_at_kernel(  # noqa: SLF001
        text, "the certified complex step_D source this weld lifts")
    if kernel.count(_chd._BODY_ANCHOR) != 1:  # noqa: SLF001
        raise AssertionError(
            "the certified step_D signature terminator is not unique")
    body = kernel.split(_chd._BODY_ANCHOR, 1)[1]  # noqa: SLF001
    if not body.endswith("}\n"):
        raise AssertionError("the certified step_D body does not end with a brace")
    body = body[: -len("}\n")]
    for anchor, what in ((_chd._THREAD_PREAMBLE, "thread preamble"),  # noqa: SLF001
                         (_chd._INDEX_DECODE, "index decode"),  # noqa: SLF001
                         (_chd._FIRST_TARGET, "target-0 marker")):  # noqa: SLF001
        if body.count(anchor) != 1:
            raise AssertionError(
                f"the certified step_D body carries {body.count(anchor)} copies of "
                f"the {what}, not one; this weld keeps it verbatim and splices its "
                f"constitutive half directly above the first target block")
    cut = body.index(_chd._FIRST_TARGET)  # noqa: SLF001
    head, tail = body[:cut], body[cut:]

    own = 0
    for component, pointer in enumerate(CURL_SOURCE_NAMES):
        for variable in ("f_1", "f_2"):
            old = f"        cf {variable} = cf_load({pointer}, idx);\n"
            if old not in tail:
                continue
            tail = _chd._needle(  # noqa: SLF001
                tail, old, f"        cf {variable} = own_h[{component}];\n",
                f"{pointer}'s own-cell load into {variable}")
            own += 1
    if own != OWN_LOAD_EDITS:
        raise AssertionError(
            f"the certified step_D body makes {own} own-cell magnetic loads of the "
            f"shape this weld redirects, not {OWN_LOAD_EDITS}; a load left standing "
            f"would read the PRE-launch H -- and on this family one of them is the "
            f"BETA PARTNER, whose staleness is invisible in every magnitude")

    taps = 0
    for line in [line + "\n" for line in tail.splitlines() if "cshift_dn(" in line]:
        line_head, _, rest = line.partition("cshift_dn(")
        pointer, _, arguments = rest.partition(", ")
        if pointer not in CURL_SOURCE_NAMES:
            raise AssertionError(
                f"the certified step_D body reads {pointer!r} through cshift_dn; this "
                f"weld resolves exactly the three magnetic sources "
                f"{CURL_SOURCE_NAMES}")
        if not arguments.rstrip("\n").endswith(");"):
            raise AssertionError(
                f"the certified cshift_dn call {line!r} does not close on its own "
                f"line; the weld appends its argument pack to that closing line")
        component = CURL_SOURCE_NAMES.index(pointer)
        carried = arguments.rstrip("\n")[: -len(");")]
        tail = _chd._needle(  # noqa: SLF001
            tail, line,
            f"{line_head}cshift_dn_recompute({component}, {carried}, weld);\n",
            f"the shifted magnetic load {line.strip()!r}")
        taps += 1
    if taps != HALO_TAPS:
        raise AssertionError(
            f"the certified step_D body makes {taps} shifted magnetic loads, not "
            f"{HALO_TAPS}; this weld redirects every one of them and a missed tap "
            f"would read a word another block is writing")
    for pointer in CURL_SOURCE_NAMES:
        if f"cf_load({pointer}," in tail or f"({pointer}," in tail:
            raise AssertionError(
                f"the welded curl half still reads {pointer}; in this signature that "
                f"pointer is the PRE-LAUNCH magnetic field and every read must be the "
                f"register or a recompute")
    if "cshift_dn(" in tail:
        raise AssertionError(
            "the welded curl half still calls the certified cshift_dn, which loads a "
            "stored H; every shifted tap must go through cshift_dn_recompute")
    _assert_the_beta_partner_is_the_weld_register(tail, source is not None)
    return head, tail


def _assert_the_beta_partner_is_the_weld_register(tail: str, lifted: bool) -> None:
    """Every surviving beta line consumes a register THIS WELD redirected.

    THE ONE CLAUSE THIS FAMILY OWNS AND NO SIBLING H->D WELD HAS. The beta term's
    partner is the SAME-CELL component snapshot (stepping.py:437), which the array
    path evaluates after ``update_H``; welded, that snapshot must be this launch's own
    ``own_h[...]`` register and never a load of the pre-launch volume. The redirect
    above rewrites the DECLARATION and leaves the beta line's spelling alone, so this
    checks the PAIRING rather than the line: per target block, each
    ``mul_imag_coefficient_left`` factor names a register declared ``= own_h[...]`` in
    that same block.

    A body with no beta lines at all is accepted ONLY when the caller handed in a text
    -- which is what the two equality checks against the released complex transform
    do; the shipped path requires :data:`BETA_INSERT_LINES` of them.
    """
    seen = 0
    for block in tail.split("\n    }\n"):
        declared: Dict[str, str] = {}
        for line in block.splitlines():
            stripped = line.strip()
            for variable in ("f_1", "f_2"):
                if stripped.startswith(f"cf {variable} = "):
                    declared[variable] = stripped[len(f"cf {variable} = "):]
        for line in block.splitlines():
            stripped = line.strip()
            if "mul_imag_coefficient_left(" not in stripped:
                continue
            seen += 1
            partner = stripped.split("mul_imag_coefficient_left(", 1)[1]
            partner = partner.split(")", 1)[0].split(",")[-1].strip()
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
            f"the certified complex-beta step_D body carries {seen} beta increments, "
            f"not {BETA_INSERT_LINES}; a body that lost one compiles, runs and is "
            f"wrong on one component")


#: THE DECLARATION, SPELLED AS A LITERAL, for the partition reader's strict regex.
_SIGNATURE_HEAD = '\nextern "C" __global__ void fused_hd_pair_pml_complex_beta(\n'


def _beta_argument_delta() -> Tuple[str, str]:
    """``complex_beta_kernels``' OWN signature delta, read off its own delta table.

    Read rather than retyped so this weld's parameter list gains the four beta words
    in exactly the place, order and spelling the certified single gained them; a
    change there arrives here as a failed anchor rather than as two hand-kept lists
    drifting apart.
    """
    for label, old, new, _expected in _beta._deltas_for("step_D"):  # noqa: SLF001
        if label == "beta_coefficient_arguments":
            return old, new
    raise AssertionError(
        "complex_beta_kernels no longer declares a 'beta_coefficient_arguments' "
        "delta; this weld takes the four beta parameters from that table and has "
        "nothing to take them from")


def signature() -> str:
    """The kernel's parameter list: the released complex weld's, plus the beta words.

    THE PARAMETER TEXT IS THE RELEASED PRODUCT'S OWN
    (:data:`.complex_fused_hd_pair._SIGNATURE_BODY`) with
    :func:`_beta_argument_delta`'s anchored edit applied, so neither half is retyped
    and a change to either arrives as a failed anchor.
    """
    old, new = _beta_argument_delta()
    body = _chd._SIGNATURE_BODY  # noqa: SLF001
    if body.count(old) != 1:
        raise AssertionError(
            f"the released complex H->D signature carries {body.count(old)} copies of "
            f"the phase-constant tail the beta delta anchors on, not one; this weld "
            f"cannot splice the four beta words into a list it cannot find the end of")
    return _SIGNATURE_HEAD + body.replace(old, new, 1)


def kernel_source(expansion) -> str:
    """The whole device source under one arm. ONE string; the fold is a runtime code."""
    head, tail = curl_pieces(expansion)
    weld = "\n".join([
        "",
        _chd.weld_args_construction().rstrip("\n"),
        "",
        "    // --- THE WELD: update_H, computed into registers and stored to SCRATCH.",
        "    // Nothing written here is read by this launch. The curl half below takes",
        "    // its OWN cell's magnetic field from these registers -- INCLUDING the",
        "    // two beta partners -- and recomputes every foreign tap from PRE-LAUNCH",
        "    // state through the same raw_update_H_cell, so no thread observes",
        "    // another thread's store; the launcher rotates H/f_w_H afterwards.",
        "    cf own_h[3];",
        "    cf own_w[3];",
        "    raw_update_H_cell(idx, weld, own_h, own_w);",
        "    cf_store(Hx_out, idx, own_h[0]);",
        "    cf_store(Hy_out, idx, own_h[1]);",
        "    cf_store(Hz_out, idx, own_h[2]);",
        "    cf_store(f_w_Hx_out, idx, own_w[0]);",
        "    cf_store(f_w_Hy_out, idx, own_w[1]);",
        "    cf_store(f_w_Hz_out, idx, own_w[2]);",
        "",
        "    // --- step_D (stepping.step_D / _apply_pml_update) with the special_kz",
        "    // beta insert under complex storage, the certified body from its first",
        "    // target block down, with the twelve magnetic reads redirected and",
        "    // NOTHING else touched.",
        "",
    ])
    source = (constitutive_prelude(expansion) + _chd.weld_args_struct()
              + _chd.raw_update_H_cell_source(expansion) + _chd.resolution_source()
              + cshift_dn_recompute_source(expansion)
              + signature() + head + weld + tail + "}\n")
    _assert_the_measured_absence_of_a_divide(source)
    return source


#: THE ONLY DIVISIONS ANY TEXT THIS FAMILY EMITS MAY CONTAIN: the certified index
#: decode's two INTEGER lines, which turn a flat cell index into the per-axis
#: coefficient index. Enumerated as exact text rather than filtered by a type
#: heuristic, so a floating-point divide arriving anywhere -- including inside the
#: decode's own block -- is a named failure.
_PERMITTED_DIVIDE_LINES: Tuple[str, ...] = (
    "    int j = (idx / nz) % ny;",
    "    int i = idx / (ny * nz);",
)

#: How many copies of the decode a complete source carries: the fused kernel's own
#: preamble and ``raw_update_H_cell``'s lifted one, which is what indexes the absorber
#: profile at a RECOMPUTED cell. Two copies, two divide lines each.
_PERMITTED_DIVIDES = 2 * len(_PERMITTED_DIVIDE_LINES)


def _assert_the_measured_absence_of_a_divide(source: str) -> None:
    """This family performs NO FLOATING-POINT DIVISION, asserted rather than argued.

    ``complex64 / float32`` on this backend is CuPy's SCALED complex/complex algorithm
    with the zero-valued terms kept, NOT numpy's reciprocal multiply -- a spelling
    that is wrong on 5 to 25 per cent of words if the wrong one is used. The
    reciprocal this recurrence needs is ``sinv``, computed host-side by ``PML``, so
    there is no site; a text that acquired one would have to be measured against that
    algorithm before it could ship, and would not be covered by this family's record.

    THE INTEGER DECODE IS THE ONE EXCEPTION AND IT IS ENUMERATED, not inferred: the
    two lines in :data:`_PERMITTED_DIVIDE_LINES`, twice each (the kernel's own
    preamble and ``raw_update_H_cell``'s lifted copy). Both count is ASSERTED, so a
    dropped decode -- the ``coefficient_index_is_the_threads_own`` defect -- fails
    here as well as in the mutation battery.
    """
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
            f"{line.strip()!r}. This family performs NO floating-point division; a "
            f"divide here would need CuPy's scaled complex algorithm measured against "
            f"it before it could ship")
    if permitted != _PERMITTED_DIVIDES:
        raise AssertionError(
            f"the welded source carries {permitted} integer index-decode divide lines, "
            f"not {_PERMITTED_DIVIDES}; the decode is what indexes the absorber "
            f"profile at the RECOMPUTED cell, and a missing copy is the half-cell "
            f"class of error -- converged, smooth and wrong")


def device_sources() -> Dict[str, str]:
    """``{kernel name: source}`` -- the shape every family on this track publishes.

    The FMA_V1 arm, which is the one every corpus row of this cell licenses on this
    hardware; the NAIVE arm's text is reachable through :func:`kernel_source` and the
    gate compiles both.
    """
    return {KERNEL_NAME: kernel_source("FMA_V1")}


def source_digest() -> str:
    """One sha256 over every source this family can emit, canonically ordered.

    Two arms is two sources; a single changed character here, in ``complex_emitter``,
    in ``complex_folded_kernels`` or in ``complex_beta_kernels`` moves this value.
    """
    import hashlib  # noqa: PLC0415 - stdlib, imported at the one call site

    digest = hashlib.sha256()
    for name in sorted(complex_emitter.EXPANSIONS):
        digest.update(name.encode("ascii"))
        digest.update(kernel_source(name).encode("utf-8"))
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# The compile memo
# ---------------------------------------------------------------------------

#: The mutable copy the launcher compiles, keyed by arm. The gate mutates through
#: :func:`set_kernel_source`; ONE seam, so a mutation cannot miss a second copy.
_SOURCES: Dict[int, str] = {}


def current_source(expansion) -> str:
    """The device text this family would compile now -- the mutated one if any."""
    key = _arm(expansion)
    if key not in _SOURCES:
        _SOURCES[key] = kernel_source(key)
    return _SOURCES[key]


def set_kernel_source(expansion, source: str) -> None:
    """Replace this family's device text -- the gate's mutation seam, and only that."""
    _SOURCES[_arm(expansion)] = source


def reset_kernel_sources() -> int:
    """Drop every mutated body; returns how many entries went. The gate's undo."""
    count = len(_SOURCES)
    _SOURCES.clear()
    return count


def _get_kernel(expansion, source: Optional[str] = None,
                options: Optional[Tuple[str, ...]] = None):
    """Compile on first use, memoized in the PACKAGE'S SHARED MEMO.

    THE SHARED MEMO AND NOT A PRIVATE DICT, which is a measurement property rather
    than tidiness: every gate on this track counts launches by wrapping
    ``compile_cache``'s memo, and a family that kept its own dict would be INVISIBLE
    to that counter -- its launches would read as zero and a launch-structure leg
    would pass a weld that never executed.

    THE SOURCE AND THE OPTIONS ARE IN THE KEY, so a mutation harness that handed in a
    rewritten string cannot be served the shipped binary, the contraction-guard
    control cannot be served the guarded build, and a late policy install cannot be
    served an earlier one. ``cupy`` is imported HERE, never at scope.
    """
    import cupy as cp  # noqa: PLC0415 - device-only

    arm = _arm(expansion)
    code = current_source(arm) if source is None else source
    build = _COMPILE_OPTIONS if options is None else tuple(options)
    key = _kernel_cache_key(KERNEL_NAME, True, build, code)
    return _get_or_compile(key, lambda: cp.RawKernel(code, KERNEL_NAME, options=build))


def _clear_kernel_cache() -> int:
    return _clear_cache()


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------

def covers_complex_beta_fused_hd_pair(fields: Any, pml: Any, grid: Any,
                                      sources: Any = None, license: Any = None,
                                      subnormal_policy: Any = None) -> Tuple[bool, str]:
    """May this product span ``update_H`` -> the electric withdraw -> ``step_D`` here?

    A CONJUNCTION OF THE TWO CERTIFIED HALVES' OWN PREDICATES plus this seam's
    withdraw clause, the boundary resolution the launch binds and the rotation's six
    volumes -- never a re-derivation of either half's clause set:

    * the H side: ``complex_beta_kernels.covers_complex_beta_constitutive(..., "H")``,
      asked FIRST because it runs first and because it asks the expansion licence
      first of all;
    * the D side: ``complex_beta_kernels.covers_complex_beta_curl(..., "step_D")``;
    * the seam: ``withdraw_hoist.seam_withdraw_reasons``, which refuses BY NAME every
      row carrying a standing integrated electric withdraw;
    * the rotation: the six volumes :data:`SCRATCH_VOLUMES` names.

    BOTH HALVES ASK THE SAME BETA AND FOLD CLAUSES through
    ``complex_beta_kernels._beta_reasons``, and that is deliberate rather than
    redundant: asking twice is what makes a future edit to ONE of the two sub-step
    predicates a visible disagreement here rather than a silent widening.
    """
    covered, reason = _beta.covers_complex_beta_constitutive(
        fields, pml, grid, "H", license, subnormal_policy)
    if not covered:
        return False, f"constitutive half: {reason}"
    covered, reason = _beta.covers_complex_beta_curl(
        fields, pml, grid, "step_D", license, subnormal_policy)
    if not covered:
        return False, f"curl half: {reason}"
    seam_reasons = _withdraw_hoist.seam_withdraw_reasons(
        fields, sources,
        undeclared=(
            "the source set was not declared: this predicate cannot infer from Fields "
            "that no electric withdraw stands between update_H and step_D"),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) has a standing integrated "
            f"electric withdraw, which the driver runs BETWEEN the update_H and "
            f"step_D consults (driver.py:3313-3314); this product declares "
            f"HOISTS_THE_WITHDRAW = False, so nothing would perform the withdraw "
            f"before its launch"),
        hoists_the_withdraw=HOISTS_THE_WITHDRAW,
        span=REPLACES)
    if seam_reasons:
        return False, seam_reasons[0]
    # THE BOUNDARY RESOLUTION THE LAUNCH BINDS, asked here rather than left to raise
    # at launch time. This family ALWAYS binds the FOLDED complex codes -- its text is
    # the folded one and the fold arms are dead on an unfolded grid -- so the
    # fold-termination cross-check (Grid.stored_cells > Grid.owned_cells against
    # Grid.is_metallic) that decides BC_METALLIC against BC_MIRROR_PERIODIC is asked
    # on every configuration, folded or not. A grid it cannot resolve is a refusal
    # with a line behind it.
    try:
        _codes, refusal = _folded.folded_complex_boundary_codes(grid)
    except Exception as error:  # noqa: BLE001 - a raise here is a refusal
        return False, (f"the curl's boundary resolution refuses this grid: "
                       f"{type(error).__name__}: {error}")
    if refusal is not None:
        return False, f"boundary resolution: {refusal}"
    # THE BETA COEFFICIENTS THIS LAUNCH BINDS, likewise.
    try:
        beta_coefficients(grid)
    except Exception as error:  # noqa: BLE001
        return False, (f"the beta coefficients this launch binds cannot be derived "
                       f"from this grid: {type(error).__name__}: {error}")
    for name in SCRATCH_VOLUMES:
        if getattr(fields, name, None) is None:
            return False, (f"{name} is not allocated; this product rotates it against "
                           f"a launch-local scratch twin after every launch")
    return True, "covered"


# ---------------------------------------------------------------------------
# The launch
# ---------------------------------------------------------------------------

def beta_coefficients(grid: Any):
    """``((plus_re, plus_im), (minus_re, minus_im))`` for the ELECTRIC side.

    CALLED, never re-transcribed: ``complex_beta_kernels.beta_curl_coefficients`` is
    stepping.py:797-811's transcription, and ``MAGNETIC["step_D"]`` is False, so the
    ``-1j`` branch is the one taken here. The real word is whatever Python's own
    complex multiply produced and is passed through -- see the module docstring for
    what that pass-through has and has not been measured to be worth.
    """
    return _beta.beta_curl_coefficients(grid.beta, grid.dt, _beta.MAGNETIC["step_D"])


def _word_view(array: Any) -> Any:
    if complex_pml_kernels is None:  # pragma: no cover - laptop guard
        raise RuntimeError("complex_pml_kernels did not import; there is no device here")
    return complex_pml_kernels.word_view(array)


def resolve(fields: Any, grid: Any, pml: Any, expansion, *,
            tables: Optional[Dict[str, Any]] = None,
            constitutive: Optional[Dict[str, Any]] = None,
            boundary_codes: Optional[Any] = None,
            phase_flags: Optional[Any] = None,
            phase_values: Optional[Any] = None,
            beta: Optional[Any] = None,
            scratch: Optional[Dict[str, Any]] = None,
            dtdx: Optional[float] = None) -> Dict[str, Any]:
    """Everything the launch needs, derived once per frozen configuration.

    DERIVED HERE BY THE SAME FUNCTIONS THE CERTIFIED LAUNCHERS ASK, which is what
    makes "the weld and the singles cannot disagree about which sub-lattice, which
    boundary code, which phase or which beta words this seam reads" an enforced
    property rather than a convention. THE OVERRIDES ARE THE GATE'S DOOR and stay,
    keyword-only and named for what they are: a gate feeds deliberately wrong tables
    (the swapped sub-lattice), wrong boundary codes (the dropped metallic wall), wrong
    phases (the unconjugated backward factor) and wrong beta words (the swapped
    signs).

    THE kms IDENTITY IS ASSERTED, NOT ASSUMED. ``step_D`` and ``update_H`` both read
    the INTEGER sub-lattice, so the curl's ``kms_*`` and the constitutive's are the
    same three device allocations and the signature binds them once. Binding the
    half-integer set instead COMPILES and is a half-cell error in the absorber
    profile, so the two are compared by base address here and a mismatch RAISES --
    except where the caller supplied one of the two deliberately, which is the
    mutation the gate arms.
    """
    import cupy as cp  # noqa: PLC0415 - device-only

    if complex_pml_kernels is None:  # pragma: no cover - laptop guard
        raise RuntimeError("complex_pml_kernels did not import; there is no device here")
    arm = _arm(expansion)
    supplied = tables is not None or constitutive is not None
    if tables is None:
        tables = complex_pml_kernels.complex_curl_tables(
            pml, complex_emitter.HALF_INTEGER["step_D"])
    if constitutive is None:
        constitutive = complex_pml_kernels.complex_constitutive_tables(
            pml, complex_emitter.HALF_INTEGER["H"])
    if not supplied:
        for axis in "xyz":
            if int(tables[f"kms_{axis}"].data.ptr) != int(
                    constitutive[f"kms_{axis}"].data.ptr):
                raise ValueError(
                    f"the curl's kms_{axis} and the constitutive's are different "
                    f"allocations; step_D and update_H both read the INTEGER Yee "
                    f"sub-lattice and this signature binds that group ONCE, so a "
                    f"disagreement here is a half-cell error in the absorber profile")
    if boundary_codes is None:
        # ``folded_complex_boundary_codes`` returns ``(codes, refusal)`` with exactly
        # one of them None. It carries the fold-termination cross-check that decides
        # BC_METALLIC against BC_MIRROR_PERIODIC on a folded axis; getting that wrong
        # is a top-plane mask applied where MEEP steps the plane, so a refusal here is
        # RAISED rather than dropped on the floor.
        boundary_codes, refusal = _folded.folded_complex_boundary_codes(grid)
        if refusal is not None:
            raise ValueError(
                f"the complex-beta curl's boundary resolution refuses this grid: "
                f"{refusal}")
    derived_flags, derived_values = complex_pml_kernels.bloch_phase_arguments(
        grid, complex_emitter.KERNELS["step_D"][1])
    if phase_flags is None:
        phase_flags = derived_flags
    if phase_values is None:
        phase_values = derived_values
    if beta is None:
        beta = beta_coefficients(grid)
    if dtdx is None:
        dtdx = float(grid.dt / grid.dx)  # stepping.py:314, :431.
    if scratch is None:
        scratch = {}
        for name in SCRATCH_VOLUMES:
            source = getattr(fields, name, None)
            if source is None:
                raise ValueError(
                    f"fields.{name} is not allocated; this weld rotates a scratch "
                    f"shaped like it and has nothing to shape one from")
            # UNINITIALIZED IS CORRECT: every cell of all six is written by every
            # launch, so there is no cell whose prior contents a reader could observe.
            scratch[name] = cp.empty_like(source)
    return {"arm": arm, "tables": tables, "constitutive": constitutive,
            "boundary_codes": tuple(boundary_codes),
            "phase_flags": tuple(phase_flags), "phase_values": tuple(phase_values),
            "beta": tuple(tuple(float(word) for word in pair) for pair in beta),
            "dtdx": float(dtdx), "scratch": scratch, "launches": 0}


def assert_bindings_are_disjoint(fields: Any, state: Dict[str, Any]) -> int:
    """Every pointer this launch binds ``__restrict__``, checked by base address.

    THE RELEASED COMPLEX WELD'S OWN CHECK, reused rather than re-spelled: this
    product binds exactly the same pointers, and the four beta words are values rather
    than allocations and add nothing to inspect.
    """
    return _chd.assert_bindings_are_disjoint(fields, state)


def launch_complex_beta_fused_hd_pair(fields: Any, state: Dict[str, Any],
                                      kernel: Optional[Any] = None,
                                      threads: int = _FUSED_THREADS) -> Dict[str, Any]:
    """THE ONE LAUNCH. ``kernel`` and ``threads`` are the gate's doors.

    The argument list is the released complex weld's with the four beta words
    APPENDED, in the order and place ``complex_beta_kernels.step_complex_beta``
    appends them to the certified single's -- so a reader can diff either pair.
    """
    import numpy as np  # noqa: PLC0415 - device-only path

    nx, ny, nz = (int(n) for n in fields.Dx.shape)
    blocks = (nx * ny * nz + threads - 1) // threads
    tables, constitutive = state["tables"], state["constitutive"]
    arguments: List[Any] = [_word_view(state["scratch"][name])
                            for name in SCRATCH_VOLUMES]
    arguments += [_word_view(getattr(fields, name))
                  for name in ("Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz",
                               "Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz",
                               "Bx", "By", "Bz")]
    arguments += [np.int32(nx), np.int32(ny), np.int32(nz),
                  np.float32(state["dtdx"])]
    for axis in "xyz":
        arguments += [tables[f"kms_{axis}"], tables[f"sinv_{axis}"]]
    arguments += [constitutive[f"kps_{axis}"] for axis in "xyz"]
    arguments += [np.int32(code) for code in state["boundary_codes"]]
    arguments += [np.int32(flag) for flag in state["phase_flags"]]
    arguments += [np.float32(value) for value in state["phase_values"]]
    (plus_re, plus_im), (minus_re, minus_im) = state["beta"]
    arguments += [np.float32(plus_re), np.float32(plus_im),
                  np.float32(minus_re), np.float32(minus_im)]
    launcher = kernel or _get_kernel(state["arm"])
    launcher((blocks,), (threads,), tuple(arguments))
    state["launches"] += 1
    return {"launched": True, "blocks": blocks, "threads": threads,
            "elements": nx * ny * nz, "kernel": KERNEL_NAME,
            "arm": complex_emitter.EXPANSION_NAMES[state["arm"]],
            "beta": state["beta"], "replaces": REPLACES}


def rotate_into_fields(fields: Any, scratch: Dict[str, Any]) -> Dict[str, Any]:
    """THE ROTATION -- ``fused_hd_pair.rotate_into_fields``, reused through the
    complex sibling.

    The consumer audit it rests on is that module's, and it is a statement about the
    ENGINE rather than about a storage class or a beta term: every consumer of the
    magnetic field resolves it BY NAME at use time, so a rebinding between steps is
    invisible to all of them.
    """
    return _chd.rotate_into_fields(fields, scratch)


def run_complex_beta_fused_hd_pair(fields: Any, grid: Any, pml: Any, expansion, *,
                                   sources: Any = None, license: Any = None,
                                   subnormal_policy: Any = None,
                                   state: Optional[Dict[str, Any]] = None,
                                   check: bool = True,
                                   rotate: bool = True) -> Dict[str, Any]:
    """The whole product: the predicate, the launch, the rotation.

    Returns the launch record with ``rotated`` and the state the caller should keep.
    """
    if check:
        covered, reason = covers_complex_beta_fused_hd_pair(
            fields, pml, grid, sources, license, subnormal_policy)
        if not covered:
            raise ValueError(f"this product does not cover this configuration: {reason}")
    if state is None:
        state = resolve(fields, grid, pml, expansion)
    assert_bindings_are_disjoint(fields, state)
    record = launch_complex_beta_fused_hd_pair(fields, state)
    if rotate:
        state["scratch"] = rotate_into_fields(fields, state["scratch"])
    record["rotated"] = bool(rotate)
    record["state"] = state
    return record
