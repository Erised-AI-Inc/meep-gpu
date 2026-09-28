"""The COMPLEX BLOCH-BETA H->D weld: complex ``update_H`` into the special-kz complex
``step_D``, in BOTH the unfolded and the folded variant.

ONE PRODUCT, TWO CELLS, and that shape is a MEASUREMENT rather than a preference --
see *ONE PRODUCT OR TWO* below. ``h_to_d_seam.instances`` files **4** rows across the
two Metal arm pairs this module serves, all four ``buildable_not_built`` and not one
carrying a standing in-seam withdraw:

    (special_kz complex beta -> special_kz complex beta)   1 row
        tests:TestSpecialKz.test_special_kz
    (folded beta complex     -> folded beta complex)       3 rows
        tests:TestEigCoeffs.test_binary_grating_special_kz_0_13_2
        tests:TestEigCoeffs.test_binary_grating_special_kz_1_17_7
        tests:TestSpecialKz.test_eigsrc_kz_0_complex

:mod:`.complex_fused_hd_pair` refuses all four BY NAME -- ``complex_fields``' clause 8
refuses ``grid.beta`` -- and :mod:`.folded_complex_fused_hd_pair` refuses the three
folded ones the same way. This module is the product on the beta side of that
inversion.

=============================================================================
BETA IS A CURL-ONLY TERM, AND THAT IS WHAT MAKES THIS SEAM WELDABLE AT ALL
=============================================================================

``stepping._special_kz_beta_term`` (S:727-784) enters ONLY the two curls (S:356-363,
:438-445). The constitutive sub-steps read nothing beta-dependent
(``_apply_constitutive_pml``, S:2065), which :mod:`.special_kz` records as a
MEASUREMENT -- 7,319 words moved by the pair and ZERO differing between a corpus row's
beta and beta = 0 -- and which is why
:func:`.special_kz.plan_beta_run_complex_constitutive` and
:func:`.folded_beta.plan_folded_beta_complex_constitutive` both build the CERTIFIED
``complex_fields.ComplexConstitutivePlan``: "only the ADMISSION is this module's".

So the constitutive half of this weld is character-for-character
:func:`.complex_fused_hd_pair.h_cell_function`'s -- the certified complex ``update_H``
as a function of an arbitrary cell, with the seven declared
:data:`.complex_fused_hd_pair.CONSTITUTIVE_LIFT_EDITS` -- and this module IMPORTS the
lift rather than owning a copy. The beta words never reach it.

**AND THAT IS AN ASSERTION, NOT A COMMENT.** :func:`certified_h_cell` re-derives the
lift and requires that the emitted ``h_cell`` body contain none of the four beta
identifiers; a future beta-dependent constitutive would fail the build here rather
than silently step a beta run with a beta-free ``H``.

=============================================================================
THE SHAPE, AND THE FOUR BETA WORDS COST ZERO BINDINGS
=============================================================================

The weld is the released siblings' exactly: the certified ``update_H`` recomputed at
the thread's own cell AND at every backward neighbour the curl taps, into
LAUNCH-LOCAL WRITE-ONLY scratch, so ``H``, ``f_w_H`` and ``B`` are ``const`` for the
whole dispatch and no thread observes another thread's store; ``D`` and ``fu_D`` step
in place (the curl reads and writes its OWN cell only); the launcher ROTATES the
``H``/``f_w_H`` bindings against the scratch afterwards
(:class:`.offdiag_weld_common.ScratchWeldPairPlan`).

    3 H_out + 3 f_w_H_out          (float2 scratch, written, never read)
  + 3 H_in  + 3 f_w_H_in           (float2, pre-launch, const)
  + 3 B                            (float2, const)
  + 3 D     + 3 fu_D               (float2, in place)
  + 6 curl coefficients (kms/sinv per axis, ONE shared group)
  + 3 constitutive kps                                                        = 30

plus one ``constant Params&`` = **31 = :data:`.device.MAX_BUFFER_BINDINGS`**, no
headroom, so :data:`PACKED_BINDINGS` is pinned as an EQUALITY and the gate compiles
one pointer past the shipped shape and requires the failure.

**THE FOUR BETA WORDS RIDE IN THE PACKED RECORD AND COST NOTHING.** The certified
complex beta curl binds them as four ``constant float&`` at buffers 26-29 and sits at
30 of 31 with no room for a weld; this kernel packs every non-pointer argument into
ONE ``constant Params&``, which is what turns "the beta curl has no headroom" into
"the beta words are free". That is the same packing the three released H->D products
use for their scalars, extended by two ``float2`` members -- and it is the single
reason this cell is weldable rather than over the ceiling.

**THE PACKED RECORD IS 64 BYTES AND THAT WAS MEASURED, NOT COMPUTED.** Five
``float2`` members then five scalars is 60 bytes of payload, and Metal rounds the
struct up to its 8-byte alignment: ``sizeof(Params)`` read back off a launched probe
on this host is **64**. The host record therefore declares ``itemsize`` 64 explicitly
-- a 60-byte buffer is four bytes short of what the shader may address -- and the
``float2`` members come FIRST for the measured reason
:mod:`.complex_fused_magnetic_pair` gives: with the scalars first, the natural host
record puts every phase one word early and reads back as a plausible complex number
rather than as garbage.

**THE COEFFICIENTS STAY float32 UNDER COMPLEX STORAGE** -- nine coefficient buffers,
not eighteen (stepping.py:41-50). Had they widened with the fields this signature
would need 39 pointers.

**THE SHARED COEFFICIENT GROUP IS LOAD-BEARING.** Both halves sit on the same Yee
sub-lattice (``SUB_STEPS['step_D']['suffix'] == ''`` and
``CONSTITUTIVE_SIDES['H']['half_integer'] is False``), so the curl's ``kms_x/_y/_z``
and the constitutive's ``km0/km1/km2`` ARE THE SAME THREE VOLUMES and are bound once.
Unshared the signature is 34 and does not compile. Bind the HALF-INTEGER set by
mistake instead and nothing fails: a smooth, converged, half-cell-wrong absorber
profile -- which is why that is a gate mutation and not a comment.

=============================================================================
WHAT THE TWO VARIANTS CHANGE, AND WHAT THEY DO NOT
=============================================================================

ONE TRANSFORM, TWO DEVICE STRINGS. :func:`beta_complex_curl_source` selects the
emitter and nothing else:

* ``plain``  -> :func:`.special_kz.beta_bloch_curl_source`
* ``folded`` -> :func:`.folded_beta.folded_beta_bloch_curl_source`, which is the SAME
  parent template with one slot inserted (``__TOP_MASK__``) and the ghost/ownership
  emitters driven over ``symmetry._reduced_codes``.

Every table the lift walks is shared between them because the two templates' nine
magnetic load lines are the same lines: the three own-cell loads
(:data:`.complex_fused_hd_pair.OWN_LOAD_EDITS`), the six shifted taps
(:data:`.complex_fused_hd_pair.HALO_TAPS`) and the index composition
(:func:`.complex_fused_hd_pair.offset_coordinates`). :func:`resolve_variant` reads the
fold off the grid, so a row never chooses between them.

THE GHOST SPELLING IS THIS FAMILY'S OWN AND IS PARSED, NOT ASSUMED. The certified beta
complex curl hoists ``const float2 zero2 = float2(0.0f, 0.0f);`` and serves ``zero2``
past a wall, where the plain complex curl spells the literal. The lift requires the
emitted line to end in the family's own spelling and RAISES otherwise, so a parent
that changed it fails the build instead of leaving a tap redirected past a different
ghost.

THE BLOCH PHASE AND THE BETA TERM ARE BOTH OUTSIDE THE REDIRECT, and for two different
reasons. The phase rotates the SHIFTED operands after the gather
(``special_kz._phase_lines``), so redirecting the LOAD leaves it standing character
for character. The beta term multiplies the UNSHIFTED CENTRE registers ``a`` and ``b``
-- "the beta partners are never rotated, which is what keeps the beta term's operand
the array path's unrotated snapshot by construction rather than by care" -- and those
registers are exactly what :data:`.complex_fused_hd_pair.OWN_LOAD_EDITS` replaces with
``own.a0``/``own.a1``. So the beta term consumes the POST-``update_H`` magnetic field
at the thread's own cell, which is what the array path's ``step_D`` consumes, and the
substitution is a rename of the load rather than a change to the term.

=============================================================================
WHERE THE CODES AND THE COEFFICIENTS COME FROM
=============================================================================

``codes`` on the ``folded`` variant MUST come from
``symmetry.folded_axis_kinds(grid, pml)``; on ``plain`` they are
``1 if kind == "metallic" else 0`` over ``stepping._boundary_kinds``, which is what
:func:`.special_kz.plan_beta_bloch_pml_curl` builds. Using the plain expression on a
folded grid maps ``"mirror"`` to 0 = PERIODIC -- the backward ghost becomes a WRAP and
the cell-0 mask is not widened -- and neither the emitter nor the compiler can catch
it. It is a GATE MUTATION (``codes_from_boundary_kinds``).

``beta_words`` come from :func:`.special_kz.beta_curl_coefficients` with
``magnetic=False`` (this seam's curl is ``step_D``) and ``complex_storage=True``, and
``beta`` and ``dt`` are passed AS GIVEN -- no ``float()`` normalisation. That is that
function's own rule and it is live: 14,926 of 40,000 random draws produce a different
float32 word when the chain is widened first. The gate arms the widened spelling as a
mutation.

``phased``/``phase_values`` come from :func:`.special_kz.bloch_phase_words` with
``backward=True`` -- this is the D seam, whose wrapped lane carries the CONJUGATE
phase. Taking the conjugation anywhere else would give one fact two homes.

THE EXPANSION ARM IS THE **BETA** PROBE'S, not ``complex_fields``'. The beta artifact
classifies one more call-site orientation, so it licenses everything the complex
artifact does and one pattern more; :func:`.special_kz.plan_beta_run_complex_constitutive`
and :mod:`.folded_beta` both bind it that way and this module follows. An absent or
non-discriminating probe is a REFUSAL, not a fallback.

=============================================================================
WHAT SITS IN THE SEAM, AND WHAT THIS PRODUCT REFUSES
=============================================================================

:data:`CARRIES_DEPOSIT_REPAIR` is False and it is a FACT about the driver rather than
a choice: nothing is INJECTED between the ``update_H`` consult (driver.py:3311) and
the ``step_D`` consult (:3315). On the folded variant that statement is stronger, not
weaker: every B-side fill, the wall clear and the far pass close BEFORE :3311 and
every D-side one opens AFTER :3315, so this seam is fill-FREE on a folded grid and a
fill carried here would be a pass the driver runs again.

:data:`HOISTS_THE_WITHDRAW` is False. The only wiring that performs the hoist is
``launch._install_fused_pair``'s ``withdraw_hoist.SEAM`` branch; this product declares
:data:`INSTALLABLE` False, so that branch is unreachable for it and nothing would
perform the withdraw before its launch. The predicate refuses BY NAME every row whose
electric withdraw does work -- **0 of these 4** -- and the clause is kept in full
anyway, because ignorance is never an empty set.

REFUSED through the parents' own certified predicates, which this one conjoins without
weakening: ``beta == 0`` (that is :mod:`.complex_fused_hd_pair`'s and
:mod:`.folded_complex_fused_hd_pair`'s cell, and the clause is INVERTED against
theirs), REAL float32 storage (:mod:`.beta_real_fused_hd_pair`'s), BFAST, cylindrical,
conductive, a live susceptibility, an off-diagonal epsilon row, an inactive absorber,
and a Bloch phase on a mirror axis.

=============================================================================
IT IS NOT INSTALLED, AND THE REASON IS ARBITRATION RATHER THAN ARITHMETIC
=============================================================================

:data:`INSTALLABLE` is False, and :data:`INSTALLABLE_REASON` carries the MEASUREMENT
-- driven through the SHIPPED Metal composer on this product's own fixtures rather
than inferred from a board. Over the driver's ``step_B - update_H - step_D -
update_E`` slot path launches are ``4 - (installed pairs)``, and a product spanning
``update_H``/``step_D`` takes one slot from EACH neighbour. Both neighbouring seams
carry Metal products on both variants -- ``beta_complex_fused_magnetic_pair`` /
``folded_beta_complex_fused_magnetic_pair`` on ``step_B``/``update_H`` and
``beta_complex_fused_electric_pair`` / ``folded_beta_complex_fused_pair`` on
``step_D``/``update_E``.

**NO TIMING EXISTS FOR THIS SHAPE AND NONE IS LICENSED BY ANYTHING IN THIS MODULE.**
A second Metal lane was running on this GPU throughout this family's gate campaign, so
no wall-clock number taken in that window would mean anything even if one had been
taken.

WHAT THE FAMILY'S CREDITED INSTANCES MEAN: the board's 4 instances are **served by
predicate admission** and nothing more; the product **installs on zero rows**; it
**executes nowhere** outside its own gate and tests; and it licenses **no timing claim
and no dispatch claim**.

=============================================================================
ONE PRODUCT OR TWO -- WHAT WAS MEASURED
=============================================================================

The two cells are ONE product because everything a product IS is identical across
them and exactly one clause differs:

* the SIGNATURE is the same 30 pointers plus one packed ``Params``, and
  :func:`shipped_signature_bindings` reads 31 off BOTH emissions;
* the ``Params`` LAYOUT, its 64-byte size and its member order are the same record;
* the ``h_cell`` LIFT is byte-identical -- it is imported, once;
* the ROTATION ORDER, the twins, the mirrors and the plan class are the same;
* all four REFUTED signatures are the same signature, so they are imported;
* the two curl emitters are the SAME PARENT TEMPLATE, one with a slot inserted:
  :func:`.folded_beta.folded_beta_bloch_curl_template` is
  ``special_kz._BETA_BLOCH_CURL_TEMPLATE`` with ``__TOP_MASK__`` after the cell-0 mask
  anchor, and nothing else.

What differs is which emitter runs and which routine resolves ``codes`` -- a source
specialisation beside the boundary codes, which is where a fold belongs. That is the
shape the CUDA sibling ``cuda_kernels/complex_fused_hd_pair.py`` already ships for the
plain/folded complex pair: one transform emitting two device strings, with
``resolve()`` reading the fold off the grid.

The split that was NOT taken is the other one. **Real beta is a separate product**
(:mod:`.beta_real_fused_hd_pair`) because storage changes the pointer TYPE on every
one of the 21 field volumes, the ``Params`` record, the ``h_cell`` lift, the helper
block and three of the four refuted signatures -- there is no shared transform left to
parameterise. The tree already splits every other beta seam that way
(``beta_fused_electric_pair`` vs ``beta_complex_fused_electric_pair``,
``folded_beta_real_fused_pair`` vs ``folded_beta_complex_fused_pair``), and this round
did not invent a third rule.

NO TORCH AT MODULE LEVEL, as everywhere in this package.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .. import withdraw_hoist as _withdraw_hoist
from ..triton_kernels.coverage import CONSTITUTIVE_SIDES, Coverage
from . import complex_fused_hd_pair as _plain
from . import folded_beta, special_kz
from . import offdiag_weld_common as _weld
from . import shaders, symmetry, templates
from .device import MAX_BUFFER_BINDINGS, Residency
from .launch import SUB_STEPS

FAMILY = "beta_complex_fused_hd_pair"

#: The sub-step slot this arm is registered on: the FIRST half of the seam, in the
#: driver's own order, so a refusal is named on the slot the fusion starts at.
SLOT = "update_H"

#: The driver passes one launch of this plan performs, in driver order
#: (driver.py:3311, :3315). Declared rather than inferred from the slot name.
REPLACES: Tuple[str, ...] = ("update_H", "step_D")

#: The seam name ``launch.FUSED_PAIR_SEAMS`` files this row under, spelled through
#: :mod:`..withdraw_hoist` rather than as a literal so the two cannot drift.
SEAM: str = _withdraw_hoist.SEAM

#: The two device strings one transform emits, and the two board cells they serve.
#: DATA rather than prose, so the test suite and the gate can assert the partition.
VARIANTS: Tuple[str, ...] = ("plain", "folded")

#: variant -> the board's ``(update_H arm, step_D arm)`` pair for the cell it serves.
#: A TUPLE rather than prose, because it is a JOIN KEY: the gate's lift and this
#: family's tests both use it to find their corpus rows in the board's
#: ``h_to_d_seam.instances``, and a cell re-typed at either site is a cell that
#: silently measures nothing.
VARIANT_CELLS: Dict[str, Tuple[str, str]] = {
    "plain": ("special_kz complex beta", "special_kz complex beta"),
    "folded": ("folded beta complex", "folded beta complex"),
}

#: Nothing is injected between the two consults, so there is no deposit in this seam
#: to carry across the launch and ``deposit_repair`` is not consulted at all. True on
#: BOTH variants: on the folded one every fill sits outside the span as well.
CARRIES_DEPOSIT_REPAIR = False

#: This product does not perform the seam's electric withdraw, so its predicate
#: refuses every row with a standing one BY NAME. On these two cells that is 0 of 4.
HOISTS_THE_WITHDRAW = False

#: May the composer install this product? NO. See :data:`INSTALLABLE_REASON` and the
#: gate's ``arbitration`` leg, which MEASURES the claim through the shipped composer.
INSTALLABLE = False

INSTALLABLE_REASON = (
    "arbitration, not arithmetic, and MEASURED through the SHIPPED Metal composer "
    "(launch.plan_step, fuse=True) on this product's own fixtures rather than "
    "inferred from a board. Over the driver's step_B - update_H - step_D - update_E "
    "slot path launches are 4 - (installed pairs), and a product spanning "
    "update_H/step_D takes one slot from EACH neighbour. Both neighbouring seams "
    "carry Metal products on both variants -- beta_complex_fused_magnetic_pair and "
    "folded_beta_complex_fused_magnetic_pair on step_B/update_H, "
    "beta_complex_fused_electric_pair and folded_beta_complex_fused_pair on "
    "step_D/update_E -- so wherever both install, installing this product is a "
    "strict LOSS (2 pairs -> 1: one more launch and one fewer seam served), and "
    "wherever exactly one installs it is a TIE the released precedent resolves for "
    "the incumbent, because the B->H pair installs first and holds update_H and "
    "_pair_may_absorb then refuses this product by name. The per-fixture verdict is "
    "the gate's `arbitration` leg, row by row, in "
    "parity/meep_gpu/results/metal_beta_complex_fused_hd_pair_2026-09-08_weld/flush/"
    "gate.json. What that leaves: the board's credited instances are served by PREDICATE "
    "ADMISSION only; the product installs on zero rows; it executes nowhere outside "
    "its own gate and tests; and the weld licenses no timing claim and no dispatch "
    "claim")

#: WHAT A RELEASE STILL OWES. Empty means "this family holds a released device gate
#: bound in ``fingerprints.json``"; non-empty means it does not, and the string is
#: the debt.
#:
#: EMPTY SINCE 2026-09-08: ``fingerprints.json`` carries
#: ``metal_beta_complex_fused_hd_pair_device_gate``, minted by
#: ``mint_metal_weld.py --family beta_complex_fused_hd_pair`` from the RELEASED
#: artifact ``parity/meep_gpu/results/
#: metal_beta_complex_fused_hd_pair_2026-09-08_weld/flush/gate.json``. That gate was
#: RE-RUN against these bytes rather than against the ones the retired declaration
#: below names: emptying this string moves the module, the module is the first pin of
#: its own weld, and ``mint_metal_weld.py`` refuses an artifact whose recorded digests
#: have moved -- so the emptying lands FIRST and the gate measures the emptied tree.
WELD_OWED: str = ""

#: WHAT THE GATE MEASURED, KEPT. The pre-weld declaration is not deleted with the
#: weld: it is the one-line record of which artifact this family's release rests on,
#: and a reader who finds an empty ``WELD_OWED`` should still be able to find that
#: without opening the board. VERBATIM, including the ``_2026-09-07`` stamp it names
#: -- that run is the measurement the debt was written against, and the ``_weld``
#: stamp above is the re-run that could be minted.
_RETIRED_WELD_OWED = (
    "the device gate parity/meep_gpu/gate_metal_beta_complex_fused_hd_pair.py has RUN "
    "and RELEASED on this host under BOTH float32 subnormal policies from empty "
    "policy-separated caches, into "
    "parity/meep_gpu/results/metal_beta_complex_fused_hd_pair_2026-09-07/"
    "{keep,flush}/gate.json. What a release still owes is the LEDGER, not a "
    "measurement: the metal_beta_complex_fused_hd_pair_device_gate entry in "
    "metal_kernels/fingerprints.json binding that artifact, minted by "
    "parity/meep_gpu/mint_metal_weld.py, together with the registry.FAMILY_MODULES "
    "row and the launch.FUSED_PAIR_ARMS absorb rows for BOTH variants. The "
    "2026-09-07 unbuilt-H_to_D round writes all of those as a unified diff "
    "(wiring.patch) rather than applying them, because three further gate campaigns "
    "in flight pin the current bytes of every existing module. Empty this string in "
    "the same change that lands the weld entry")

#: The binding shape. The POINTER half is the plain complex product's exactly, so the
#: three pointer-side refuted signatures are its; only the packed record is wider, and
#: a wider record is still ONE binding.
PACKED_BINDINGS = _plain.PACKED_BINDINGS
ONE_MORE_POINTER_BINDINGS = _plain.ONE_MORE_POINTER_BINDINGS
UNSHARED_KMS_BINDINGS = _plain.UNSHARED_KMS_BINDINGS
SPLIT_PLANE_BINDINGS = _plain.SPLIT_PLANE_BINDINGS

#: The same 30 pointers with every non-pointer argument bound SEPARATELY, the way the
#: certified 30-binding beta complex curl binds its own: four extents, ``dtdx``, SIX
#: phase floats and FOUR beta floats. **45**, and the arithmetic is worth writing out
#: because it is the measurement behind "the packing is what makes this cell weldable":
#: the non-beta complex product's unpacked shape is 38 (30 + 4 uints + ``dtdx`` + three
#: ``float2`` phases), and this one is seven worse -- four of those seven are the beta
#: words and three are the phases, which the certified beta curl binds as six separate
#: ``constant float&`` rather than as three ``float2``. Packed, all fifteen cost ONE
#: binding, and 45 -> 31 is what makes room for a weld on top of a curl that already
#: sat at 30 of 31.
SEPARATE_SCALAR_BINDINGS = 45

#: The host record's total size in bytes. MEASURED off a launched probe on this host:
#: five ``float2`` members then five scalars is 60 bytes of payload and Metal rounds
#: ``sizeof(Params)`` up to its 8-byte alignment, so the record is 64 and a 60-byte
#: buffer is four bytes short of what the shader may address.
PARAMS_ITEMSIZE = 64

#: The rotating volumes, in SIGNATURE ORDER. The plain complex product's, imported:
#: this weld has the same six and the same order, and two spellings of one order is
#: how a launch binds the wrong buffer to every argument after the first mismatch.
ROTATED_NAMES: Tuple[str, ...] = _plain.ROTATED_NAMES

#: The kernel entry point.
KERNEL = "beta_complex_fused_hd_pair_step"

#: The exact complex ghost the certified BETA complex curl serves past a wall. It is
#: this family's own spelling -- a hoisted ``const float2`` rather than the literal the
#: non-beta complex curl uses -- and the lift REQUIRES it rather than assuming it.
_GHOST = "zero2"

#: The line the parent template hoists that ghost on. Checked to be present, so a
#: parent that stopped hoisting it fails the build here.
_GHOST_DECLARATION = "    const float2 zero2 = float2(0.0f, 0.0f);\n"

#: The four identifiers the beta term introduces. The constitutive lift is asserted to
#: contain NONE of them: beta is a curl-only term and a future beta-dependent
#: ``update_H`` must fail this build rather than step a beta run with a beta-free H.
BETA_IDENTIFIERS: Tuple[str, ...] = ("bpr", "bpi", "bmr", "bmi")

if PACKED_BINDINGS > MAX_BUFFER_BINDINGS:  # pragma: no cover - a design invariant
    raise RuntimeError(
        f"the complex beta fused H/D pair binds {PACKED_BINDINGS} buffers and Metal's "
        f"ceiling on this toolchain is {MAX_BUFFER_BINDINGS}")

__all__ = [
    "ARMS", "BETA_IDENTIFIERS", "CARRIES_DEPOSIT_REPAIR", "FAMILY",
    "HOISTS_THE_WITHDRAW", "INSTALLABLE", "INSTALLABLE_REASON", "KERNEL",
    "ONE_MORE_POINTER_BINDINGS", "PACKED_BINDINGS", "PARAMS_ITEMSIZE", "REPLACES",
    "ROTATED_NAMES", "SEAM", "SEPARATE_SCALAR_BINDINGS", "SLOT",
    "SPLIT_PLANE_BINDINGS", "UNSHARED_KMS_BINDINGS", "VARIANTS", "VARIANT_CELLS",
    "WELD_OWED", "_PHASE_RESOLVER", "beta_complex_curl_source", "beta_complex_fused_hd_pair_source",
    "certified_h_cell", "compile_beta_complex_fused_hd_pair", "enumerate_sources",
    "metal_beta_complex_fused_hd_pair_coverage",
    "params_record_dtype", "plan_metal_beta_complex_fused_hd_pair",
    "refuted_one_more_pointer_source", "refuted_separate_scalar_source",
    "refuted_unshared_kms_source", "register_arms", "resolve_codes",
    "resolve_variant", "shipped_signature_bindings", "split_plane_pair_signature",
    "welded_curl_tail",
]

#: The three POINTER-side refuted signatures, IMPORTED. They measure the pointer half
#: of the signature, and this product's pointer half is the plain complex one's, so
#: re-spelling them here would be a second place for the same measurement to go stale.
refuted_one_more_pointer_source = _plain.refuted_one_more_pointer_source
refuted_unshared_kms_source = _plain.refuted_unshared_kms_source
split_plane_pair_signature = _plain.split_plane_pair_signature


# ---------------------------------------------------------------------------
# The lift: the certified COMPLEX update_H at an arbitrary cell — imported, checked
# ---------------------------------------------------------------------------

def certified_h_cell(expansion: str, contract: str = shaders.CONTRACT_OFF) -> str:
    """:func:`.complex_fused_hd_pair.h_cell_function`, with the beta-free assertion.

    THE BODY IS IMPORTED AND UNMODIFIED. What this wrapper adds is the check that
    makes "beta is a curl-only term" a build-time property of this product rather than
    a sentence in a docstring: the emitted ``h_cell`` must contain none of
    :data:`BETA_IDENTIFIERS`. A future beta-dependent constitutive would then fail here
    instead of stepping a beta run with a beta-free magnetic field, which is a smooth
    wrong answer.
    """
    body = _plain.h_cell_function(expansion, contract)
    for name in BETA_IDENTIFIERS:
        if name in body:
            raise AssertionError(
                f"the certified complex update_H body now mentions {name!r}; this "
                f"product welds it into a BETA curl on the premise that the "
                f"constitutive sub-step reads nothing beta-dependent "
                f"(stepping._apply_constitutive_pml, S:2065), and that premise has "
                f"changed")
    return body


# ---------------------------------------------------------------------------
# The lift: the certified BETA complex step_D curl with every H read redirected
# ---------------------------------------------------------------------------

def beta_complex_curl_source(variant: str, codes: Sequence[int],
                             phased: Sequence[int], expansion: str,
                             contract: str = shaders.CONTRACT_OFF) -> str:
    """The certified beta complex ``step_D`` curl for one variant.

    ONE TRANSFORM, TWO DEVICE STRINGS, and the selection is the whole of what the
    variant means: ``plain`` is :func:`.special_kz.beta_bloch_curl_source` and
    ``folded`` is :func:`.folded_beta.folded_beta_bloch_curl_source`, which is that
    same parent template with ``__TOP_MASK__`` inserted after the cell-0 mask anchor.

    ``has_beta`` is not a parameter: this product's predicate REQUIRES ``beta != 0``
    on both variants, so a beta-free build would be a specialisation no plan can
    reach. ``backward`` is True and never a parameter either -- this pair is the D
    seam, and ``step_B``'s forward strides and unconjugated phase are a different
    product.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    backward = bool(SUB_STEPS["step_D"]["backward"])
    if variant == "plain":
        return special_kz.beta_bloch_curl_source(codes, backward, phased, expansion,
                                                 True, contract)
    return folded_beta.folded_beta_bloch_curl_source(codes, backward, phased,
                                                     expansion, True, contract)


def welded_curl_tail(variant: str, codes: Sequence[int], phased: Sequence[int],
                     expansion: str, contract: str = shaders.CONTRACT_OFF
                     ) -> Tuple[str, str]:
    """``(prologue, body)`` of the beta complex ``step_D`` curl, H reads redirected.

    THE TABLES ARE THE PLAIN COMPLEX PRODUCT'S, imported: the three own-cell load
    edits, the six shifted taps and the index parse. The certified beta complex
    template's nine magnetic load lines are character-identical to the non-beta
    complex curl's, which is the fact that lets one set of tables serve both -- and a
    needle that stopped matching RAISES here rather than emitting a kernel that reads
    the wrong volume.

    THE GHOST IS THIS FAMILY'S OWN SPELLING AND IS REQUIRED, NOT ASSUMED. The parent
    hoists ``const float2 zero2 = float2(0.0f, 0.0f);`` and serves ``zero2``; the
    literal the non-beta complex curl uses would silently satisfy a laxer check.

    WHAT IS NOT TOUCHED, and each for its own reason: the ghost ternary's GUARD and
    OFFSET are parsed out of the emitter's own line rather than retyped; the Bloch
    PHASE rotates the shifted operands AFTER the gather and is never seen here; the
    BETA term multiplies the unshifted centre registers ``a`` and ``b``, which the
    own-cell edits have already replaced with ``own.a0``/``own.a1``, so the term
    consumes the post-``update_H`` field at the thread's own cell exactly as the array
    path's ``step_D`` does; and both fold blocks (the top-plane mask and the widened
    cell-0 mask) write ``curlN = ... : curlN;`` and touch no ``gN[``.
    """
    source = beta_complex_curl_source(variant, codes, phased, expansion, contract)
    if _plain._BODY_ANCHOR not in source:
        raise AssertionError(
            "the certified beta complex curl no longer carries the body anchor; the "
            "fused kernel would splice a truncated body")
    body = source.split(_plain._BODY_ANCHOR, 1)[1]
    if not body.endswith("}\n"):
        raise AssertionError("the beta complex curl source does not end with '}'")
    body = body[: -len("}\n")]
    if _plain.PROLOGUE_END not in body:
        raise AssertionError(
            "the beta complex curl no longer declares `int nyz = nyi * nzi;`; this "
            "weld cuts its prologue there because its first statement calls h_cell "
            "with nxi/nyi/nzi")
    prologue, tail = body.split(_plain.PROLOGUE_END, 1)
    prologue = prologue + _plain.PROLOGUE_END
    if _GHOST_DECLARATION not in tail:
        raise AssertionError(
            f"the beta complex curl no longer hoists {_GHOST_DECLARATION.strip()!r}; "
            f"this weld requires the family's own ghost spelling so that a tap "
            f"redirected past a DIFFERENT ghost cannot pass unnoticed")
    offsets = _plain.offset_coordinates(tail)
    for var, target in _plain.OWN_LOAD_EDITS:
        tail = _weld.needle(tail, f"    float2 {var}   = g{target}[ii];\n",
                            f"    float2 {var}   = own.a{target};\n")
    for var, target in _plain.HALO_TAPS:
        prefix = f"    float2 {var} = "
        matches = [line + "\n" for line in tail.splitlines()
                   if line.startswith(prefix)]
        if len(matches) != 1:
            raise AssertionError(
                f"the beta complex curl declares {var} {len(matches)} times; this "
                f"weld redirects exactly one magnetic load per shifted tap")
        old = matches[0]
        # THE GUARD AND THE OFFSET ARE BOTH PARSED FROM THE EMITTER'S OWN LINE.
        head, rest = old.split(" ? ", 1)
        expected = f"g{target}["
        if not rest.startswith(expected) or not rest.endswith(f" : {_GHOST};\n"):
            raise AssertionError(
                f"the beta complex curl no longer reads {var} as `{expected}...]` "
                f"served an exact {_GHOST} past the wall; the ghost rule this weld "
                f"preserves has changed")
        offset = rest[len(expected):].split("]", 1)[0]
        coordinates = offsets[offset]
        call = (f"h_cell({', '.join(coordinates)}, "
                f"{_plain.H_CELL_TAIL_ARGS}).a{target}")
        tail = _weld.needle(tail, old, f"{head} ? {call} : {_GHOST};\n")
    for target in range(3):
        if f"g{target}[" in tail:
            raise AssertionError(
                f"the welded beta complex curl half still reads g{target}; in this "
                f"signature that pointer does not exist and every magnetic read must "
                f"be the register or a recompute")
    return prologue, tail


# ---------------------------------------------------------------------------
# The source
# ---------------------------------------------------------------------------

_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

__HELPERS__

// THE FIVE float2 MEMBERS COME FIRST. Metal aligns float2 to 8 bytes, so with the
// scalars first this struct needs internal padding and the natural host record puts
// every phase one word early — it then reads back as a plausible complex number
// rather than as garbage. Phases and beta words first, no internal padding; the
// struct's own size is 64 (measured off a launched probe, not computed).
struct Params {
    float2 px; float2 py; float2 pz; float2 bp; float2 bm;
    uint nx; uint ny; uint nz; uint n_elem; float dtdx;
};

__H_CELL__

kernel void beta_complex_fused_hd_pair_step(
    device float2*       ho0     [[buffer(0)]],
    device float2*       ho1     [[buffer(1)]],
    device float2*       ho2     [[buffer(2)]],
    device float2*       wo0     [[buffer(3)]],
    device float2*       wo1     [[buffer(4)]],
    device float2*       wo2     [[buffer(5)]],
    device const float2* hi0     [[buffer(6)]],
    device const float2* hi1     [[buffer(7)]],
    device const float2* hi2     [[buffer(8)]],
    device const float2* wi0     [[buffer(9)]],
    device const float2* wi1     [[buffer(10)]],
    device const float2* wi2     [[buffer(11)]],
    device const float2* b0      [[buffer(12)]],
    device const float2* b1      [[buffer(13)]],
    device const float2* b2      [[buffer(14)]],
    device float2*       f0      [[buffer(15)]],
    device float2*       f1      [[buffer(16)]],
    device float2*       f2      [[buffer(17)]],
    device float2*       u0      [[buffer(18)]],
    device float2*       u1      [[buffer(19)]],
    device float2*       u2      [[buffer(20)]],
    device const float*  kmx     [[buffer(21)]],
    device const float*  sinvx   [[buffer(22)]],
    device const float*  kmy     [[buffer(23)]],
    device const float*  sinvy   [[buffer(24)]],
    device const float*  kmz     [[buffer(25)]],
    device const float*  sinvz   [[buffer(26)]],
    device const float*  kp0     [[buffer(27)]],
    device const float*  kp1     [[buffer(28)]],
    device const float*  kp2     [[buffer(29)]],
    constant Params&     prm     [[buffer(30)]],
    uint idx [[thread_position_in_grid]])
{
    // THE TWELVE PACKED ARGUMENTS ARE UNPACKED INTO THE CERTIFIED BODIES' OWN NAMES,
    // once, before any lifted text runs. Everything below this line is then
    // character-for-character what `special_kz.beta_bloch_curl_source` (or its folded
    // derivative) and `complex_fields.bloch_constitutive_source` emit, with the
    // declared lift edits and the redirected magnetic reads.
    //
    // THE BETA WORDS ARE READ HERE AND NOWHERE ELSE. The certified curl binds them as
    // four `constant float&`; packed, they cost nothing, which is the only reason
    // this cell fits inside the 31-binding ceiling at all.
    uint nx = prm.nx, ny = prm.ny, nz = prm.nz, n_elem = prm.n_elem;
    float dtdx = prm.dtdx;
    float pxr = prm.px.x, pxi = prm.px.y;
    float pyr = prm.py.x, pyi = prm.py.y;
    float pzr = prm.pz.x, pzi = prm.pz.y;
    float bpr = prm.bp.x, bpi = prm.bp.y;
    float bmr = prm.bm.x, bmi = prm.bm.y;
__PROLOGUE__
    // --- THE WELD: complex update_H, computed into registers and stored to SCRATCH -
    // Nothing written here is read by this launch. The curl half below takes its OWN
    // cell's magnetic field from these registers and recomputes every foreign tap
    // from PRE-LAUNCH state through the same `h_cell`, so no thread observes another
    // thread's store; the launcher rotates H/f_w_H afterwards. The BETA term's two
    // partners are `a` and `b`, which are exactly these registers.
    h_cell_result own = h_cell(i, j, k, __H_CELL_ARGS__);
    ho0[ii] = own.a0; ho1[ii] = own.a1; ho2[ii] = own.a2;
    wo0[ii] = own.src0; wo1[ii] = own.src1; wo2[ii] = own.src2;

    // --- the beta complex step_D curl (stepping.step_D / _apply_pml_update:1905) --
    // D and fu_D update IN PLACE and that is safe by construction: the curl reads and
    // writes them at the THREAD'S OWN CELL only (`f0[ii]`, `u0[ii]`), so no thread
    // reads a displacement another thread wrote.
__CURL__
}
"""


def beta_complex_fused_hd_pair_source(
        variant: str, codes: Sequence[int], phased: Sequence[int], expansion: str,
        contract: str = shaders.CONTRACT_OFF) -> str:
    """One specialised fused source: (variant, boundaries, phases, arm, mode).

    Every constant Triton bakes into a ``tl.constexpr`` is baked into the string here,
    for the reason :mod:`.shaders` gives: ``torch.mps.compile_shader`` takes a source
    and nothing else, so specialisation by substitution is what keeps the emitted
    arithmetic identical to the bodies this lifts.

    THE WALL CLEAR IS NOT CARRIED and must not be: ``zero_metal_B`` runs one seam
    EARLIER and ``zero_metal_D`` one seam LATER, so a mask here would be a pass the
    driver runs again. On the ``folded`` variant NO FILL is carried either, for the
    same reason and with a stronger premise: all six fold passes sit outside the span.
    """
    codes = tuple(int(code) for code in codes)
    phased = tuple(int(flag) for flag in phased)
    if len(codes) != 3:
        raise ValueError(f"codes must be a per-axis triple, got {codes!r}")
    if len(phased) != 3:
        raise ValueError(f"phased must be a per-axis triple, got {phased!r}")
    prologue, curl = welded_curl_tail(variant, codes, phased, expansion, contract)
    return templates.substitute(_TEMPLATE, {
        "__CONTRACT__": shaders.contraction_pragma(contract),
        "__HELPERS__": templates.complex_helpers(expansion),
        "__H_CELL__": certified_h_cell(expansion, contract),
        "__PROLOGUE__": prologue,
        "__H_CELL_ARGS__": _plain.H_CELL_TAIL_ARGS,
        "__CURL__": curl,
    })


def compile_beta_complex_fused_hd_pair(
        variant: str, codes: Sequence[int], phased: Sequence[int], expansion: str,
        contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised entry point for one (variant, boundaries, phases, arm, mode)."""
    from .device import compile_source  # noqa: PLC0415

    return getattr(
        compile_source(beta_complex_fused_hd_pair_source(variant, codes, phased,
                                                         expansion, contract)),
        KERNEL)


def shipped_signature_bindings(variant: str, expansion: str) -> int:
    """How many bindings the SHIPPED kernel declares, counted off its own source.

    Read from the emitted text rather than from :data:`PACKED_BINDINGS`, so the
    constant is a claim the source can falsify -- and read PER VARIANT, so a fold that
    had grown an argument would be caught rather than hidden behind the plain
    emission.
    """
    codes = ((symmetry.CODE_PERIODIC,) * 3 if variant == "plain"
             else (symmetry.CODE_PERIODIC, symmetry.CODE_MIRROR_PERIODIC,
                   symmetry.CODE_PERIODIC))
    source = beta_complex_fused_hd_pair_source(variant, codes, (0, 0, 0), expansion)
    signature = source.split(f"kernel void {KERNEL}(", 1)[1]
    signature = signature.split("uint idx [[thread_position_in_grid]])", 1)[0]
    return signature.count("[[buffer(")


def refuted_separate_scalar_source(contract: str = shaders.CONTRACT_OFF) -> str:
    """The same 30 pointers with the twelve non-pointer arguments bound SEPARATELY.

    45 bindings, and the gate REQUIRES the compile failure. This is not a strawman: it
    is exactly how :func:`.special_kz.beta_bloch_curl_source` binds its own arguments
    at its 30-binding size -- six separate phase floats and four separate beta floats
    -- so "packed" is a measurement on THIS signature rather than a house style. The
    seven bindings over the non-beta product's 38 are the four beta words plus the
    three the phases cost when they stop being ``float2``.
    """
    lines: List[str] = []
    slot = 0
    for label in _plain._COMPLEX_VOLUMES:
        lines.append(f"    device float2*      {label:<8}[[buffer({slot})]],")
        slot += 1
    for label in _plain._REAL_VECTORS:
        lines.append(f"    device const float* {label:<8}[[buffer({slot})]],")
        slot += 1
    for kind, label in (("constant uint&", "nx"), ("constant uint&", "ny"),
                        ("constant uint&", "nz"), ("constant uint&", "n_elem"),
                        ("constant float&", "dtdx")):
        lines.append(f"    {kind:<19} {label:<8}[[buffer({slot})]],")
        slot += 1
    for label in ("pxr", "pxi", "pyr", "pyi", "pzr", "pzi",
                  "bpr", "bpi", "bmr", "bmi"):
        lines.append(f"    constant float&     {label:<8}[[buffer({slot})]],")
        slot += 1
    if slot != SEPARATE_SCALAR_BINDINGS:  # pragma: no cover - a design invariant
        raise AssertionError((slot, SEPARATE_SCALAR_BINDINGS))
    touch = " + ".join(f"{label}[0]" for label in _plain._REAL_VECTORS)
    return "\n".join((
        "#include <metal_stdlib>",
        "using namespace metal;",
        "",
        shaders.contraction_pragma(contract),
        "",
        "kernel void refuted_separate_scalar(",
        *lines,
        "    uint idx [[thread_position_in_grid]])",
        "{",
        "    if (idx >= n_elem) { return; }",
        f"    float touch = {touch};",
        "    float2 acc = float2(touch, pxr + pxi + pyr + pyi + pzr + pzi",
        "                        + bpr + bpi + bmr + bmi + dtdx);",
        *[f"    acc = acc + {label}[idx];" for label in _plain._COMPLEX_VOLUMES],
        f"    {_plain._COMPLEX_VOLUMES[0]}[idx] = acc;",
        "}",
        "",
    ))


def _phase_triples(codes: Sequence[int]) -> Tuple[Tuple[int, int, int], ...]:
    """Every phase triple legal against one boundary triple/quadruple.

    Only a plain PERIODIC axis can carry a Bloch phase: a METALLIC one cannot, and a
    MIRROR one reflects rather than repeating, so ``stepping._bloch_phases`` raises and
    the driver refuses the configuration outright.
    """
    legal: List[Tuple[int, int, int]] = []
    for px in (0, 1):
        for py in (0, 1):
            for pz in (0, 1):
                triple = (px, py, pz)
                if any(triple[axis] and codes[axis] != symmetry.CODE_PERIODIC
                       for axis in range(3)):
                    continue
                legal.append(triple)
    return tuple(legal)


def enumerate_sources(expansion: str,
                      contract: str = shaders.CONTRACT_OFF) -> Dict[str, str]:
    """Every specialisation a shipped plan can emit, keyed by a stable label.

    ``plain`` enumerates the PERIODIC/METALLIC triples; ``folded`` enumerates the four
    folded codes on every axis with at least one MIRROR present, because an
    all-unfolded quadruple is the ``plain`` variant's own territory and this family's
    ``resolve_variant`` would never route it here. Each is crossed only with the phase
    triples it can legally carry, so the count is the REACHABLE set rather than the
    Cartesian product and a label that cannot be built is not fingerprinted as if it
    could.
    """
    out: Dict[str, str] = {}
    for cx in (symmetry.CODE_PERIODIC, symmetry.CODE_METALLIC):
        for cy in (symmetry.CODE_PERIODIC, symmetry.CODE_METALLIC):
            for cz in (symmetry.CODE_PERIODIC, symmetry.CODE_METALLIC):
                codes = (cx, cy, cz)
                for phased in _phase_triples(codes):
                    label = (f"{KERNEL}/plain/{cx}{cy}{cz}/"
                             f"ph{phased[0]}{phased[1]}{phased[2]}/{expansion}")
                    out[label] = beta_complex_fused_hd_pair_source(
                        "plain", codes, phased, expansion, contract)
    codes_axis = (symmetry.CODE_PERIODIC, symmetry.CODE_METALLIC,
                  symmetry.CODE_MIRROR_METALLIC, symmetry.CODE_MIRROR_PERIODIC)
    mirrors = {symmetry.CODE_MIRROR_METALLIC, symmetry.CODE_MIRROR_PERIODIC}
    for cx in codes_axis:
        for cy in codes_axis:
            for cz in codes_axis:
                codes = (cx, cy, cz)
                if not mirrors.intersection(codes):
                    continue
                for phased in _phase_triples(codes):
                    label = (f"{KERNEL}/folded/{cx}{cy}{cz}/"
                             f"ph{phased[0]}{phased[1]}{phased[2]}/{expansion}")
                    out[label] = beta_complex_fused_hd_pair_source(
                        "folded", codes, phased, expansion, contract)
    return out


# ---------------------------------------------------------------------------
# Resolution — one row never chooses between the two variants
# ---------------------------------------------------------------------------

def _PHASE_RESOLVER(grid: Any, kinds: Sequence[str]) -> Tuple[int, ...]:
    """The per-axis Bloch flags THIS product bakes, through its own parent's resolver.

    Exposed under a name a harness can find so a gate never re-derives them: a leg
    that resolved the flags a different way would compile a different specialisation
    from the one the plan launches, which is how a mutation harness reports a defect
    as uncaught. ``backward=True``: this seam's curl is ``step_D``, whose wrapped lane
    carries the CONJUGATE phase, and the conjugation lives in
    :func:`.special_kz.bloch_phase_words` and nowhere else.
    """
    phased, _words = special_kz.bloch_phase_words(grid, kinds, True)
    return tuple(int(flag) for flag in phased)


def resolve_variant(grid: Any) -> Optional[str]:
    """``"folded"`` when any axis carries a mirror plane, ``"plain"`` otherwise.

    THE FOLD IS READ OFF THE GRID, which is what makes this one product rather than
    two overlapping ones: a row cannot be admitted by both variants, so no composer
    order decides which kernel steps it. ``None`` when the grid cannot answer -- a
    predicate that guessed here would be the over-covering refusal the package's
    "ignorance is never an empty set" rule exists to prevent.
    """
    if grid is None:
        return None
    mirrored = getattr(grid, "is_mirrored", None)
    if not callable(mirrored):
        return None
    return "folded" if any(bool(mirrored(axis)) for axis in range(3)) else "plain"


def resolve_codes(variant: str, grid: Any, pml: Any) -> Optional[Tuple[int, ...]]:
    """The boundary codes for one variant, from that variant's OWN resolver.

    ``folded`` goes through ``symmetry.folded_axis_kinds``, which routes the ghost rule
    through ``stepping._boundary_kinds``, splits MIRROR through
    ``stepping._stored_past_owned`` and cross-checks that split against
    ``grid.is_metallic`` -- two independent routes to the same fact, with a
    disagreement a refusal rather than a coin toss. ``plain`` uses
    ``1 if kind == "metallic" else 0``, which is exactly what
    :func:`.special_kz.plan_beta_bloch_pml_curl` builds.

    USING THE PLAIN EXPRESSION ON A FOLDED GRID IS THE ONE THING THIS FAMILY CAN GET
    WRONG SILENTLY: ``_boundary_kinds`` reports ``"mirror"``, which it maps to
    0 = PERIODIC, so the backward ghost becomes a WRAP to the far plane and the cell-0
    mask is not widened. Both 0 and 1 are valid codes, so neither the emitter nor the
    compiler can catch it. It is a gate mutation.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    if variant == "folded":
        codes, _reasons = symmetry.folded_axis_kinds(grid, pml)
        return None if codes is None else tuple(int(code) for code in codes)
    from ..stepping import _boundary_kinds  # noqa: PLC0415

    return tuple(1 if kind == "metallic" else 0
                 for kind in _boundary_kinds(grid, pml))


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def metal_beta_complex_fused_hd_pair_coverage(
        fields: Any, pml: Any, sources: Any = None, residency: Any = None,
        probe: Any = None) -> Coverage:
    """May ONE dispatch span beta complex ``update_H`` -> withdraw -> ``step_D``?

    A conjunction of the RESOLVED VARIANT's two halves' own certified predicates plus
    the seam clauses. Nothing is weakened: a configuration either half refuses is
    refused here with that half's reasons, prefixed so a reader can tell which side
    said it. That construction is what makes the beta clause, the complex-storage
    clause, the expansion-probe clause, the BFAST clause, the off-diagonal clause, the
    polarization clause and the fold clause INHERITED rather than restated.

    THE VARIANT IS RESOLVED FIRST AND THE OTHER ONE IS NEVER ASKED. Asking both and
    taking the disjunction would admit a row on a predicate whose kernel the plan
    would not build; resolving from the grid means the predicate and the plan reach
    the same emitter by construction.

    THE ORDER IS THE DRIVER'S. The constitutive half is asked FIRST because it runs
    first (driver.py:3311), so the first refusal a reader sees names the half the
    driver would have reached first.
    """
    reasons: List[str] = []
    grid = getattr(fields, "grid", None)
    variant = resolve_variant(grid)
    if variant is None:
        return Coverage(False, ("fields carries no grid that can be asked whether an "
                                "axis is folded; this product resolves its variant "
                                "from that answer and refuses rather than guessing",))

    if variant == "plain":
        magnetic = special_kz.beta_run_complex_constitutive_coverage(
            fields, pml, "H", residency, probe)
        curl = special_kz.beta_bloch_pml_curl_coverage(
            fields, pml, "step_D", residency, probe)
    else:
        magnetic = folded_beta.folded_beta_complex_constitutive_coverage(
            fields, pml, "H", residency, probe)
        curl = folded_beta.folded_beta_composition_bloch_curl_coverage(
            fields, pml, "step_D", residency, probe)
    if not magnetic.covered:
        reasons.extend(f"{variant} beta complex constitutive half: {reason}"
                       for reason in magnetic.reasons)
    if not curl.covered:
        reasons.extend(f"{variant} beta complex curl half: {reason}"
                       for reason in curl.reasons)

    # THE SEAM'S ONE PASS. Nothing is INJECTED between the two consults, so
    # `deposit_repair` is not consulted at all. What IS between them is the electric
    # integrated-source withdraw (driver.py:3313-3314), and `withdraw_hoist` owns it.
    # IGNORANCE IS NEVER AN EMPTY SET: `Fields` does not hold the source list.
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

    # THE CODES MUST RESOLVE. On the folded variant `folded_axis_kinds` returns None
    # when its two independent routes to the MIRROR_METALLIC / MIRROR_PERIODIC split
    # disagree, and a plan built on a guessed split is a plane of wrong values.
    if resolve_codes(variant, grid, pml) is None:
        reasons.append(
            "symmetry.folded_axis_kinds could not resolve the per-axis mirror "
            "termination; this weld bakes those codes into the source and a guessed "
            "MIRROR_METALLIC/MIRROR_PERIODIC split is a plane of wrong values rather "
            "than a crash")

    # THE ROTATION IS THE PRODUCT'S OWN INVARIANT. The plan swaps the ENGINE's
    # references for the six volumes in ROTATED_NAMES after each launch.
    for name in ROTATED_NAMES:
        if getattr(fields, name, None) is None:
            reasons.append(
                f"{name} is not allocated; this weld rotates it against a plan-owned "
                f"scratch twin after every launch")
    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class MetalBetaComplexFusedHdPairPlan(_weld.ScratchWeldPairPlan):
    """The shared scratch-weld plan, carrying the facts that specialise it.

    THE ROTATION IS NOT REIMPLEMENTED. ``run`` and ``_resolve`` are the base's. What
    the subclass adds is the specialisation a harness must not re-derive: the variant
    the source was emitted for, the per-axis phase flags, the rounded and conjugated
    phase values, the two rounded beta word pairs, and the PROBE-MEASURED expansion
    arm. Re-deriving any of them at a call site is how a mutation harness ends up
    compiling a different specialisation from the one it launches.
    """

    __slots__ = ("variant", "phased", "phase_values", "beta_words", "expansion")

    def __init__(self, *arguments: Any, variant: str, phased: Sequence[int],
                 phase_values: Sequence[Tuple[float, float]],
                 beta_words: Sequence[Tuple[float, float]],
                 expansion: str) -> None:
        super().__init__(*arguments)
        self.variant = str(variant)
        self.phased = tuple(int(flag) for flag in phased)
        self.phase_values = tuple((float(re), float(im)) for re, im in phase_values)
        self.beta_words = tuple((float(re), float(im)) for re, im in beta_words)
        self.expansion = str(expansion)


def params_record_dtype() -> Any:
    """The host record's dtype -- FIVE ``float2`` MEMBERS FIRST, itemsize 64.

    ONE HOME FOR THE LAYOUT, because getting it wrong is a silent wrong answer rather
    than a crash. The offsets are stated EXPLICITLY even though the float2-first order
    makes them the natural ones, so the record cannot drift into agreeing with Metal
    only by accident; ``itemsize`` is stated because the payload is 60 bytes and
    Metal's struct is 64 -- measured off a launched probe on this host, not computed.
    """
    import numpy as np  # noqa: PLC0415

    return np.dtype({
        "names": ["px", "py", "pz", "bp", "bm",
                  "nx", "ny", "nz", "n_elem", "dtdx"],
        "formats": [("<f4", 2), ("<f4", 2), ("<f4", 2), ("<f4", 2), ("<f4", 2),
                    "<u4", "<u4", "<u4", "<u4", "<f4"],
        "offsets": [0, 8, 16, 24, 32, 40, 44, 48, 52, 56],
        "itemsize": PARAMS_ITEMSIZE,
    })


def _params_tensor(shape: Sequence[int], dtdx: float,
                   phase_values: Sequence[Tuple[float, float]],
                   beta_words: Sequence[Tuple[float, float]], device: str) -> Any:
    """The five scalars, three phases and two beta pairs as one 64-byte record.

    PLAN-OWNED AND OUTSIDE THE RESIDENCY REGISTRY, for the reason every fused pair's
    ``_params_tensor`` gives: :meth:`.Residency.mirror` binds float32 and complex64
    volumes and refuses anything else BY NAME, because a wider element silently
    reinterprets. This record is neither, so it is built here, once, at plan time.
    Nothing on the launch path allocates.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    record = np.zeros(1, dtype=params_record_dtype())
    nx, ny, nz = (int(n) for n in shape)
    record["nx"], record["ny"], record["nz"] = nx, ny, nz
    record["n_elem"] = nx * ny * nz
    record["dtdx"] = np.float32(dtdx)
    for name, (real, imag) in zip(("px", "py", "pz"), tuple(phase_values)):
        record[name] = (np.float32(real), np.float32(imag))
    for name, (real, imag) in zip(("bp", "bm"), tuple(beta_words)):
        record[name] = (np.float32(real), np.float32(imag))
    words = np.frombuffer(record.tobytes(), dtype=np.int32).copy()
    return torch.from_numpy(words).to(torch.device(device))


def plan_metal_beta_complex_fused_hd_pair(
        fields: Any, pml: Any, sources: Any = None,
        residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        functions: Optional[Mapping[str, Any]] = None,
        probe: Any = None,
        codes: Optional[Sequence[int]] = None,
        beta_words: Optional[Sequence[Tuple[float, float]]] = None,
        ) -> Optional["_weld.ScratchWeldPairPlan"]:
    """Build the beta complex fused H/D plan, or ``None`` when the seam is refused.

    ``None`` is the only refusal, for the reason :func:`.launch.plan_pml_curl` gives:
    a configuration this kernel does not carry must fall back to the array path, never
    raise into a caller that would otherwise have stepped correctly.

    THREE MUTATION SEAMS, each named because dropping one silently DISARMS a gate leg
    rather than slowing it: ``functions`` hands in a deliberately broken compiled
    source; ``codes`` hands in the PERIODIC/METALLIC triple on a folded grid, the one
    structural defect this family can make silently; ``beta_words`` hands in a pair
    rounded through a widened chain, which
    :func:`.special_kz.beta_curl_coefficients` measured as a live 37%-of-draws
    difference rather than a hypothetical one.
    """
    if not metal_beta_complex_fused_hd_pair_coverage(
            fields, pml, sources, residency, probe).covered:
        return None
    record = probe if probe is not None else special_kz.load_expansion_probe()
    expansion = special_kz.beta_expansion_from_probe(record)
    if expansion is None:  # pragma: no cover - the predicate already refused
        return None
    from ..stepping import _boundary_kinds  # noqa: PLC0415

    grid = fields.grid
    variant = resolve_variant(grid)
    if variant is None:  # pragma: no cover - the predicate already refused
        return None
    if codes is None:
        codes = resolve_codes(variant, grid, pml)
        if codes is None:  # pragma: no cover - the predicate already refused
            return None
    codes = tuple(int(code) for code in codes)

    # THE PHASE FLAGS COME FROM `_boundary_kinds` ON BOTH VARIANTS, NOT FROM THE
    # FOLDED CODES: a folded axis is refused a phase by the folded family's own
    # clause, so the two agree on every admitted run, and reading the phase off the
    # RESOLVED kinds keeps this builder identical to `special_kz`'s and keeps the
    # conjugation in one place. backward=True: this is the D seam.
    kinds = _boundary_kinds(grid, pml)
    phased, phase_flat = special_kz.bloch_phase_words(grid, kinds, True)
    phase_values = tuple((phase_flat[2 * axis], phase_flat[2 * axis + 1])
                         for axis in range(3))
    if beta_words is None:
        # magnetic=False: this seam's CURL is step_D. `beta` and `dt` are passed AS
        # GIVEN, which is that function's own rule and is what keeps the coefficient
        # word the array path's bits.
        beta_words = special_kz.beta_curl_coefficients(
            grid.beta, grid.dt, magnetic=False, complex_storage=True)
    beta_words = tuple((float(re), float(im)) for re, im in beta_words)

    volumes: List[str] = []

    def bind_complex(name: str, host: Any) -> Any:
        import numpy  # noqa: PLC0415

        volumes.append(name)
        return residency.mirror(name, host, dtype=numpy.complex64)

    def bind_real(name: str, host: Any) -> Any:
        volumes.append(name)
        return residency.mirror(name, host, constant=True)

    # THE ROTATING GROUP FIRST, in ROTATED_NAMES order: every scratch, then every
    # pre-launch buffer. `ScratchWeldPairPlan.run` splices exactly those two groups
    # ahead of `static_args`, so this order IS the signature's. The twin MUST be
    # mirrored as complex64: the mirror's default resolution is float32, which on a
    # complex host array is a silent cast that discards the imaginary part.
    import numpy as _numpy  # noqa: PLC0415

    twins: Dict[str, Any] = {}
    for name in ROTATED_NAMES:
        host = getattr(fields, name)
        bind_complex(name, host)
        twin, _mirror = _weld.scratch_twin(residency, name, host,
                                           dtype=_numpy.complex64)
        twins[name] = twin

    flux = [bind_complex(name, getattr(fields, name))
            for name in SUB_STEPS["step_B"]["targets"]]
    displacement = [bind_complex(name, getattr(fields, name))
                    for name in SUB_STEPS["step_D"]["targets"]]
    auxiliary = [bind_complex("fu_" + name, getattr(fields, "fu_" + name))
                 for name in SUB_STEPS["step_D"]["targets"]]

    # ONE SUB-LATTICE, TWO HALVES. Both suffixes are READ FROM THE SHIPPED TABLES and
    # their equality is ASSERTED rather than assumed: it is the whole premise of the
    # 30-pointer signature, and if either table moved, sharing the group would bind
    # one half's coefficients to the other half's lattice -- a converged, smooth,
    # half-cell-wrong absorber profile rather than a failure.
    suffix = SUB_STEPS["step_D"]["suffix"]
    constitutive_suffix = "_h" if CONSTITUTIVE_SIDES["H"]["half_integer"] else ""
    if suffix != constitutive_suffix:
        raise AssertionError(
            f"step_D reads the {suffix or 'integer'!r} PML sub-lattice and update_H "
            f"the {constitutive_suffix or 'integer'!r} one; this weld binds ONE kms "
            f"group for both halves and that is only correct while they agree")
    curl_coefficients = [
        bind_real(f"pml:{stem}_{axis}{suffix}", getattr(pml, f"{stem}_{axis}{suffix}"))
        for axis in "xyz" for stem in ("kms", "sinv")]
    constitutive_coefficients = [
        bind_real(f"pml:kps_{axis}{constitutive_suffix}",
                  getattr(pml, f"kps_{axis}{constitutive_suffix}"))
        for axis in "xyz"]

    selected = dict(functions or {})
    for mode in contract_variants:
        if mode not in selected:
            selected[mode] = compile_beta_complex_fused_hd_pair(
                variant, codes, phased, expansion, mode)

    dtdx = grid.dt / grid.dx
    static = (flux + displacement + auxiliary
              + curl_coefficients + constitutive_coefficients
              + [_params_tensor(grid.shape, dtdx, phase_values, beta_words,
                                residency.device)])
    assert (2 * len(ROTATED_NAMES) + len(static)) == PACKED_BINDINGS, (
        2 * len(ROTATED_NAMES) + len(static), PACKED_BINDINGS)
    assert PACKED_BINDINGS == MAX_BUFFER_BINDINGS, (PACKED_BINDINGS,
                                                    MAX_BUFFER_BINDINGS)
    return MetalBetaComplexFusedHdPairPlan(
        FAMILY, residency, fields, dict(twins), dict(selected), ROTATED_NAMES,
        tuple(static), tuple(volumes), tuple(grid.shape), tuple(codes),
        (0, 0, 0), (), REPLACES,
        variant=variant, phased=phased, phase_values=phase_values,
        beta_words=beta_words, expansion=expansion)


# ---------------------------------------------------------------------------
# Registration -- wired=False, gated on beta, refused by the composer on INSTALLABLE
# ---------------------------------------------------------------------------

def _has_beta(context: Any) -> bool:
    """The cheap gate that decides whether this family is CONSULTED at all.

    A gated-out arm contributes NO reason, which is right: on a beta-free run
    ``update_H`` carries other arms that will speak, and this one would only add a
    refusal about a beta nobody asked for. The gate is deliberately the cheapest
    possible read of the grid and never the predicate itself.
    """
    grid = getattr(getattr(context, "fields", None), "grid", None)
    return grid is not None and bool(getattr(grid, "beta", 0.0))


def _arm_coverage(context: Any, slot: str) -> Coverage:
    if slot != SLOT:
        return Coverage(False,
                        (f"beta complex fused H/D pair cannot fill {slot}",))
    return metal_beta_complex_fused_hd_pair_coverage(
        context.fields, context.pml, context.sources, context.residency,
        context.extra.get("beta_probe"))


def _arm_plan(context: Any, slot: str) -> Optional["_weld.ScratchWeldPairPlan"]:
    if slot != SLOT:
        return None
    return plan_metal_beta_complex_fused_hd_pair(
        context.fields, context.pml, context.sources, context.residency,
        context.contract_variants, probe=context.extra.get("beta_probe"))


def register_arms() -> Tuple[Any, ...]:
    """One row on ``update_H``, ``wired=False``, GATED on beta.

    ONE ROW FOR TWO CELLS, which is what "one product" means at the arm table: the
    variant is resolved inside the predicate from the grid, so the row's verdict is
    the resolved variant's and no composer order decides which kernel a folded beta
    row gets.

    REGISTERED SO IT IS ENUMERABLE, NOT SO IT IS SELECTABLE. ``arms.arms_for`` skips
    an unwired row so ``plan_step`` cannot select it, while ``arms.registered`` still
    returns it -- which is what lets a composition sweep measure disjointness against
    this predicate instead of assuming it, and what lets
    ``launch._neighbouring_seam_claimant`` see the row when it asks who claims the
    seam ``update_H`` opens.
    """
    from . import arms  # noqa: PLC0415

    return (arms.register(FAMILY, SLOT, "fused H/D pair (complex beta)",
                          _arm_coverage, _arm_plan,
                          prefix="fused H/D pair (complex beta): ",
                          noun="fused complex-beta H/D-curl pair",
                          gate=_has_beta,
                          wired=False, replaces=REPLACES),)


ARMS = register_arms()
