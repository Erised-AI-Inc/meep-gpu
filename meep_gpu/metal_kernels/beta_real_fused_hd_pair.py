"""The REAL BETA H->D weld: the certified ``update_H`` into the special-kz REAL
``step_D``, in BOTH the unfolded and the folded variant.

ONE PRODUCT, TWO CELLS -- the same shape :mod:`.beta_complex_fused_hd_pair` takes, and
for the same measured reasons. ``h_to_d_seam.instances`` files **2** rows across the
two Metal arm pairs this module serves, both ``buildable_not_built`` and neither
carrying a standing in-seam withdraw:

    (special_kz real beta -> special_kz real beta)   1 row
        examples:refl-angular-kz2d.py
    (folded beta real     -> folded beta real)       1 row
        tests:TestSpecialKz.test_eigsrc_kz_1_real_imag

:mod:`.fused_hd_pair` refuses both BY NAME -- ``coverage._grid_reasons`` clause 12
refuses ``beta != 0`` -- and :mod:`.folded_fused_hd_pair` refuses the folded one the
same way. This module is the product on the beta side of that inversion, under REAL
float32 storage.

TWO ROWS IS A SMALL CELL AND THAT IS STATED RATHER THAN DRESSED UP. The owner rule is
*close all fusion gaps regardless*: a priced predicate gap is closed because it is
priced, not because it is large, and the alternative -- leaving two corpus rows
unserved on a seam whose other four cells are served -- is a hole a later reader has
to re-derive. What the small denominator does change is the strength of the LIFT leg,
and that is reported as N of D with D named rather than smoothed over.

=============================================================================
BETA IS A CURL-ONLY TERM, AND THAT IS WHAT MAKES THIS SEAM WELDABLE AT ALL
=============================================================================

``stepping._special_kz_beta_term`` (S:727-784) enters ONLY the two curls (S:356-363,
:438-445). The constitutive sub-steps read nothing beta-dependent
(``_apply_constitutive_pml``, S:2065), which :mod:`.special_kz` records as a
MEASUREMENT -- 7,319 words moved by the pair and ZERO differing between a corpus row's
beta and beta = 0 -- and which is why :func:`.special_kz.plan_beta_run_constitutive`
and :func:`.folded_beta.plan_folded_beta_constitutive` both build the CERTIFIED
``launch.ConstitutivePlan``: "only the ADMISSION is this module's".

So the constitutive half of this weld is character-for-character
:func:`.fused_hd_pair.h_cell_function`'s -- the certified ``update_H`` as a function of
an arbitrary cell, with the seven declared
:data:`.fused_hd_pair.CONSTITUTIVE_LIFT_EDITS` -- and this module IMPORTS the lift
rather than owning a copy. The beta words never reach it.

**AND THAT IS AN ASSERTION, NOT A COMMENT.** :func:`certified_h_cell` re-derives the
lift and requires the emitted ``h_cell`` body to contain neither beta identifier; a
future beta-dependent constitutive would fail the build here rather than silently step
a beta run with a beta-free ``H``.

=============================================================================
THE SHAPE, AND THE TWO BETA WORDS COST ZERO BINDINGS
=============================================================================

The weld is the released siblings' exactly: the certified ``update_H`` recomputed at
the thread's own cell AND at every backward neighbour the curl taps, into
LAUNCH-LOCAL WRITE-ONLY scratch, so ``H``, ``f_w_H`` and ``B`` are ``const`` for the
whole dispatch and no thread observes another thread's store; ``D`` and ``fu_D`` step
in place; the launcher ROTATES the ``H``/``f_w_H`` bindings against the scratch
afterwards (:class:`.offdiag_weld_common.ScratchWeldPairPlan`).

    3 H_out + 3 f_w_H_out          (float32 scratch, written, never read)
  + 3 H_in  + 3 f_w_H_in           (pre-launch, const)
  + 3 B                            (const)
  + 3 D     + 3 fu_D               (in place)
  + 6 curl coefficients (kms/sinv per axis, ONE shared group)
  + 3 constitutive kps                                                        = 30

plus one ``constant Params&`` = **31 = :data:`.device.MAX_BUFFER_BINDINGS`**, no
headroom. :data:`PACKED_BINDINGS` is pinned as an EQUALITY and the gate compiles one
pointer past the shipped shape and requires the failure.

**THE TWO BETA WORDS RIDE IN THE PACKED RECORD AND COST NOTHING.** The certified real
beta curl binds them as two ``constant float&`` at buffers 20-21 on a 22-binding
kernel; this weld packs every non-pointer argument into ONE ``constant Params&``,
which is the same packing the three released H->D products use for their scalars,
extended by two floats. The record is 28 bytes with no padding: seven 4-byte members
at 4-byte alignment.

**THE SHARED COEFFICIENT GROUP IS LOAD-BEARING.** Both halves sit on the same Yee
sub-lattice (``SUB_STEPS['step_D']['suffix'] == ''`` and
``CONSTITUTIVE_SIDES['H']['half_integer'] is False``), so the curl's ``kms_x/_y/_z``
and the constitutive's ``km0/km1/km2`` ARE THE SAME THREE VOLUMES and are bound once.
Unshared the signature is 34 and does not compile. Bind the HALF-INTEGER set instead
and nothing fails -- a converged, smooth, half-cell-wrong absorber profile -- which is
why that is a gate mutation and not a comment.

=============================================================================
WHAT THE TWO VARIANTS CHANGE, AND WHAT THEY DO NOT
=============================================================================

ONE TRANSFORM, TWO DEVICE STRINGS. :func:`beta_real_curl_source` selects the emitter
and nothing else:

* ``plain``  -> :func:`.special_kz.beta_curl_source`
* ``folded`` -> :func:`.folded_beta.folded_beta_curl_source`, which is the SAME parent
  template with ``__TOP_MASK__`` inserted after the cell-0 mask anchor and the
  ghost/ownership emitters driven over ``symmetry._reduced_codes``.

Every table the lift walks is shared between them and is :mod:`.fused_hd_pair`'s,
imported: the three own-cell load edits, the six shifted taps and the index parse. The
certified beta curl's nine magnetic load lines are character-identical to the
certified non-beta curl's, which is the fact that lets one set of tables serve all
four kernels. :func:`resolve_variant` reads the fold off the grid, so a row never
chooses between them.

THE BETA TERM IS OUTSIDE THE REDIRECT AND THAT IS A PROPERTY OF WHAT IT READS. It
multiplies the UNSHIFTED CENTRE registers ``a`` and ``b`` -- "the partners are the
UNSHIFTED CENTER loads the curl already made" -- and those registers are exactly what
:data:`.fused_hd_pair.OWN_LOAD_EDITS` replaces with ``own.a0``/``own.a1``. So the beta
term consumes the post-``update_H`` magnetic field at the thread's own cell, which is
what the array path's ``step_D`` consumes, and the substitution is a rename of the
load rather than a change to the term.

=============================================================================
WHERE THE CODES AND THE COEFFICIENTS COME FROM
=============================================================================

``codes`` on the ``folded`` variant MUST come from
``symmetry.folded_axis_kinds(grid, pml)``; on ``plain`` they are
``1 if kind == "metallic" else 0`` over ``stepping._boundary_kinds``, which is what
:func:`.special_kz.plan_beta_pml_curl` builds. Using the plain expression on a folded
grid maps ``"mirror"`` to 0 = PERIODIC -- the backward ghost becomes a WRAP and the
cell-0 mask is not widened -- and neither the emitter nor the compiler can catch it
(both codes are valid there). It is a GATE MUTATION (``codes_from_boundary_kinds``).

``beta_plus``/``beta_minus`` come from :func:`.special_kz.beta_curl_coefficients` with
``magnetic=False`` (this seam's curl is ``step_D``) and ``complex_storage=False``, and
``beta`` and ``dt`` are passed AS GIVEN -- no ``float()`` normalisation. That is that
function's own rule and it is live rather than hypothetical: 14,926 of 40,000 random
draws produce a different float32 word when the chain is widened first, and
``Grid(courant=numpy.float32(...))`` reaches it. The gate arms the widened spelling.

=============================================================================
WHAT SITS IN THE SEAM, AND WHAT THIS PRODUCT REFUSES
=============================================================================

:data:`CARRIES_DEPOSIT_REPAIR` is False and it is a FACT about the driver: nothing is
INJECTED between the ``update_H`` consult (driver.py:3311) and the ``step_D`` consult
(:3315). On the folded variant every B-side fill, the wall clear and the far pass
close BEFORE :3311 and every D-side one opens AFTER :3315, so this seam is fill-FREE
on a folded grid and a fill carried here would be a pass the driver runs again.

:data:`HOISTS_THE_WITHDRAW` is False, so the predicate refuses BY NAME every row whose
electric withdraw does work -- **0 of these 2** -- and the clause is kept in full
anyway, because ignorance is never an empty set and a row that grows an integrated
source later must be refused rather than silently served.

REFUSED through the parents' own certified predicates, which this one conjoins without
weakening: ``beta == 0`` (that is :mod:`.fused_hd_pair`'s and
:mod:`.folded_fused_hd_pair`'s cell, and the clause is INVERTED against theirs),
COMPLEX storage (:mod:`.beta_complex_fused_hd_pair`'s), BFAST, cylindrical,
conductive, a live susceptibility, an off-diagonal epsilon row and an inactive
absorber.

=============================================================================
IT IS NOT INSTALLED, AND THE REASON IS ARBITRATION RATHER THAN ARITHMETIC
=============================================================================

:data:`INSTALLABLE` is False, and :data:`INSTALLABLE_REASON` carries the MEASUREMENT
-- driven through the SHIPPED Metal composer on this product's own fixtures rather
than inferred from a board.

**NO TIMING EXISTS FOR THIS SHAPE AND NONE IS LICENSED BY ANYTHING IN THIS MODULE.** A
second Metal lane was running on this GPU throughout this family's gate campaign, so
no wall-clock number taken in that window would mean anything even if one had been
taken.

WHAT THE FAMILY'S CREDITED INSTANCES MEAN: the board's 2 instances are **served by
predicate admission** and nothing more; the product **installs on zero rows**; it
**executes nowhere** outside its own gate and tests; and it licenses **no timing claim
and no dispatch claim**.

=============================================================================
WHY THIS IS A SEPARATE PRODUCT FROM THE COMPLEX BETA ONE
=============================================================================

Storage changes the pointer TYPE on all 21 field volumes, the ``Params`` record (28
bytes here, 64 there), the ``h_cell`` lift (:mod:`.fused_hd_pair`'s versus
:mod:`.complex_fused_hd_pair`'s), the helper block (none here, ``complex_helpers``
there) and three of the four refuted signatures -- there is no shared transform left
to parameterise, so a merged module would be two modules sharing a filename. The tree
already splits every other beta seam that way (``beta_fused_electric_pair`` vs
``beta_complex_fused_electric_pair``, ``folded_beta_real_fused_pair`` vs
``folded_beta_complex_fused_pair``), and this round did not invent a third rule.

What IS shared with the complex product is the fold/unfold merge, and there the
argument is the same one, inverted: the signature, the record, the lift, the rotation
and every refuted signature are IDENTICAL across the two variants, and only the curl
emitter differs. That is a source specialisation beside the boundary codes, which is
where a fold belongs.

NO TORCH AT MODULE LEVEL, as everywhere in this package.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .. import withdraw_hoist as _withdraw_hoist
from ..triton_kernels.coverage import CONSTITUTIVE_SIDES, Coverage
from . import folded_beta, special_kz
from . import fused_hd_pair as _plain
from . import offdiag_weld_common as _weld
from . import shaders, symmetry
from .device import MAX_BUFFER_BINDINGS, Residency
from .launch import SUB_STEPS

FAMILY = "beta_real_fused_hd_pair"

#: The sub-step slot this arm is registered on: the FIRST half of the seam, in the
#: driver's own order, so a refusal is named on the slot the fusion starts at.
SLOT = "update_H"

#: The driver passes one launch of this plan performs, in driver order
#: (driver.py:3311, :3315).
REPLACES: Tuple[str, ...] = ("update_H", "step_D")

#: The seam name ``launch.FUSED_PAIR_SEAMS`` files this row under, spelled through
#: :mod:`..withdraw_hoist` rather than as a literal so the two cannot drift.
SEAM: str = _withdraw_hoist.SEAM

#: The two device strings one transform emits, and the two board cells they serve.
VARIANTS: Tuple[str, ...] = ("plain", "folded")

#: variant -> the board's ``(update_H arm, step_D arm)`` pair for the cell it serves.
#: A TUPLE rather than prose, because it is a JOIN KEY the gate's lift and this
#: family's tests both use to find their corpus rows in the board's
#: ``h_to_d_seam.instances``.
VARIANT_CELLS: Dict[str, Tuple[str, str]] = {
    "plain": ("special_kz real beta", "special_kz real beta"),
    "folded": ("folded beta real", "folded beta real"),
}

#: Nothing is injected between the two consults, so there is no deposit in this seam
#: to carry across the launch. True on BOTH variants.
CARRIES_DEPOSIT_REPAIR = False

#: This product does not perform the seam's electric withdraw, so its predicate
#: refuses every row with a standing one BY NAME. On these two cells that is 0 of 2.
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
    "carry Metal products on both variants -- beta_fused_magnetic_pair and "
    "folded_beta_real_fused_magnetic_pair on step_B/update_H, "
    "beta_fused_electric_pair and folded_beta_real_fused_pair on step_D/update_E -- "
    "so wherever both install, installing this product is a strict LOSS (2 pairs -> "
    "1: one more launch and one fewer seam served), and wherever exactly one installs "
    "it is a TIE the released precedent resolves for the incumbent, because the B->H "
    "pair installs first and holds update_H and _pair_may_absorb then refuses this "
    "product by name. The per-fixture verdict is the gate's `arbitration` leg, row by "
    "row, in parity/meep_gpu/results/metal_beta_real_fused_hd_pair_2026-09-08_weld/"
    "flush/gate.json. What that leaves: the board's "
    "credited instances are served by PREDICATE ADMISSION only; the product installs "
    "on zero rows; it executes nowhere outside its own gate and tests; and the weld "
    "licenses no timing claim and no dispatch claim")

#: WHAT A RELEASE STILL OWES. Empty means "this family holds a released device gate
#: bound in ``fingerprints.json``"; non-empty means it does not.
#:
#: EMPTY SINCE 2026-09-08: ``fingerprints.json`` carries
#: ``metal_beta_real_fused_hd_pair_device_gate``, minted by
#: ``mint_metal_weld.py --family beta_real_fused_hd_pair`` from the RELEASED artifact
#: ``parity/meep_gpu/results/metal_beta_real_fused_hd_pair_2026-09-08_weld/flush/
#: gate.json``. That gate was RE-RUN against these bytes rather than against the ones
#: the retired declaration below names: emptying this string moves the module, the
#: module is the first pin of its own weld, and ``mint_metal_weld.py`` refuses an
#: artifact whose recorded digests have moved -- so the emptying lands FIRST and the
#: gate measures the emptied tree.
WELD_OWED: str = ""

#: WHAT THE GATE MEASURED, KEPT. The pre-weld declaration is not deleted with the
#: weld: it is the one-line record of which artifact this family's release rests on,
#: and a reader who finds an empty ``WELD_OWED`` should still be able to find that
#: without opening the board. VERBATIM, including the ``_2026-09-07`` stamp it names
#: -- that run is the measurement the debt was written against, and the ``_weld``
#: stamp above is the re-run that could be minted.
_RETIRED_WELD_OWED = (
    "the device gate parity/meep_gpu/gate_metal_beta_real_fused_hd_pair.py has RUN "
    "and RELEASED on this host under BOTH float32 subnormal policies from empty "
    "policy-separated caches, into "
    "parity/meep_gpu/results/metal_beta_real_fused_hd_pair_2026-09-07/"
    "{keep,flush}/gate.json. What a release still owes is the LEDGER, not a "
    "measurement: the metal_beta_real_fused_hd_pair_device_gate entry in "
    "metal_kernels/fingerprints.json binding that artifact, minted by "
    "parity/meep_gpu/mint_metal_weld.py, together with the registry.FAMILY_MODULES "
    "row and the launch.FUSED_PAIR_ARMS absorb rows for BOTH variants. The "
    "2026-09-07 unbuilt-H_to_D round writes all of those as a unified diff "
    "(wiring.patch) rather than applying them, because three further gate campaigns "
    "in flight pin the current bytes of every existing module. Empty this string in "
    "the same change that lands the weld entry")

#: The binding shape. The POINTER half is the plain real product's exactly, so the two
#: pointer-side refuted signatures are its; only the packed record is wider, and a
#: wider record is still ONE binding.
PACKED_BINDINGS = _plain.PACKED_BINDINGS
ONE_MORE_POINTER_BINDINGS = _plain.ONE_MORE_POINTER_BINDINGS
UNSHARED_KMS_BINDINGS = _plain.UNSHARED_KMS_BINDINGS

#: The same 30 pointers with the SEVEN scalars bound separately, the way the certified
#: 22-binding real beta curl binds its own. 37; over the ceiling by six, and two
#: bindings worse than the non-beta product's 35 -- the measurement behind "the
#: packing is what makes this cell weldable".
SEPARATE_SCALAR_BINDINGS = 37

#: The host record's total size in bytes: seven 4-byte members at 4-byte alignment,
#: no padding. Stated rather than left implicit, because every sibling states it and
#: a reader comparing them should see why this one is the only unpadded record here.
PARAMS_ITEMSIZE = 28

#: The rotating volumes, in SIGNATURE ORDER. The plain real product's, imported.
ROTATED_NAMES: Tuple[str, ...] = _plain.ROTATED_NAMES

#: The kernel entry point.
KERNEL = "beta_real_fused_hd_pair_step"

#: The exact ghost the certified real beta curl serves past a wall.
_GHOST = "0.0f"

#: The two identifiers the beta term introduces. The constitutive lift is asserted to
#: contain NEITHER: beta is a curl-only term, and a future beta-dependent ``update_H``
#: must fail this build rather than step a beta run with a beta-free H.
BETA_IDENTIFIERS: Tuple[str, ...] = ("beta_plus", "beta_minus")

if PACKED_BINDINGS > MAX_BUFFER_BINDINGS:  # pragma: no cover - a design invariant
    raise RuntimeError(
        f"the real beta fused H/D pair binds {PACKED_BINDINGS} buffers and Metal's "
        f"ceiling on this toolchain is {MAX_BUFFER_BINDINGS}")

__all__ = [
    "ARMS", "BETA_IDENTIFIERS", "CARRIES_DEPOSIT_REPAIR", "FAMILY",
    "HOISTS_THE_WITHDRAW", "INSTALLABLE", "INSTALLABLE_REASON", "KERNEL",
    "ONE_MORE_POINTER_BINDINGS", "PACKED_BINDINGS", "PARAMS_ITEMSIZE", "REPLACES",
    "ROTATED_NAMES", "SEAM", "SEPARATE_SCALAR_BINDINGS", "SLOT",
    "UNSHARED_KMS_BINDINGS", "VARIANTS", "VARIANT_CELLS", "WELD_OWED",
    "beta_real_curl_source", "beta_real_fused_hd_pair_source", "certified_h_cell",
    "compile_beta_real_fused_hd_pair", "enumerate_sources",
    "metal_beta_real_fused_hd_pair_coverage", "params_record_dtype",
    "plan_metal_beta_real_fused_hd_pair", "refuted_one_more_pointer_source",
    "refuted_separate_scalar_source", "refuted_unshared_kms_source", "register_arms",
    "resolve_codes", "resolve_variant", "shipped_signature_bindings",
    "welded_curl_tail",
]

#: The two POINTER-side refuted signatures, IMPORTED. They measure the pointer half of
#: the signature, and this product's pointer half is the plain real one's.
refuted_one_more_pointer_source = _plain.refuted_one_more_pointer_source
refuted_unshared_kms_source = _plain.refuted_unshared_kms_source


# ---------------------------------------------------------------------------
# The lift: the certified update_H at an arbitrary cell — imported, checked
# ---------------------------------------------------------------------------

def certified_h_cell(contract: str = shaders.CONTRACT_OFF) -> str:
    """:func:`.fused_hd_pair.h_cell_function`, with the beta-free assertion.

    THE BODY IS IMPORTED AND UNMODIFIED. What this wrapper adds is the check that
    makes "beta is a curl-only term" a build-time property of this product rather than
    a sentence in a docstring.
    """
    body = _plain.h_cell_function(contract)
    for name in BETA_IDENTIFIERS:
        if name in body:
            raise AssertionError(
                f"the certified update_H body now mentions {name!r}; this product "
                f"welds it into a BETA curl on the premise that the constitutive "
                f"sub-step reads nothing beta-dependent "
                f"(stepping._apply_constitutive_pml, S:2065), and that premise has "
                f"changed")
    return body


# ---------------------------------------------------------------------------
# The lift: the certified REAL BETA step_D curl with every H read redirected
# ---------------------------------------------------------------------------

def beta_real_curl_source(variant: str, codes: Sequence[int],
                          contract: str = shaders.CONTRACT_OFF) -> str:
    """The certified real beta ``step_D`` curl for one variant.

    ONE TRANSFORM, TWO DEVICE STRINGS, and the selection is the whole of what the
    variant means: ``plain`` is :func:`.special_kz.beta_curl_source` and ``folded`` is
    :func:`.folded_beta.folded_beta_curl_source`, which is that same parent template
    with ``__TOP_MASK__`` inserted after the cell-0 mask anchor.

    ``has_beta`` is not a parameter: this product's predicate REQUIRES ``beta != 0``
    on both variants. ``backward`` is True and never a parameter either -- this pair is
    the D seam.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    backward = bool(SUB_STEPS["step_D"]["backward"])
    if variant == "plain":
        return special_kz.beta_curl_source(codes, backward, True, contract)
    return folded_beta.folded_beta_curl_source(codes, backward, True, contract)


def welded_curl_tail(variant: str, codes: Sequence[int],
                     contract: str = shaders.CONTRACT_OFF) -> Tuple[str, str]:
    """``(prologue, body)`` of the real beta ``step_D`` curl, H reads redirected.

    THE TABLES ARE THE PLAIN REAL PRODUCT'S, imported: the three own-cell load edits,
    the six shifted taps and the index parse. The certified beta template's nine
    magnetic load lines are character-identical to the certified non-beta curl's,
    which is the fact that lets one set of tables serve both -- and a needle that
    stopped matching RAISES here rather than emitting a kernel that reads the wrong
    volume.

    WHAT IS NOT TOUCHED: the ghost ternary's GUARD and OFFSET are parsed out of the
    emitter's own line rather than retyped, so a metallic ghost that read an exact
    ``0.0f`` still reads one and the periodic wrap is still the emitter's own integer
    expression; the BETA term multiplies the unshifted centre registers ``a`` and
    ``b``, which the own-cell edits have already replaced with ``own.a0``/``own.a1``;
    and on the folded variant both mask blocks write ``curlN = ... : curlN;`` and
    touch no ``gN[``.
    """
    source = beta_real_curl_source(variant, codes, contract)
    if _plain._BODY_ANCHOR not in source:
        raise AssertionError(
            "the certified real beta curl no longer carries the body anchor; the "
            "fused kernel would splice a truncated body")
    body = source.split(_plain._BODY_ANCHOR, 1)[1]
    if not body.endswith("}\n"):
        raise AssertionError("the real beta curl source does not end with '}'")
    body = body[: -len("}\n")]
    if _plain.DECODE_END not in body:
        raise AssertionError(
            "the real beta curl no longer carries the decode anchor; the fused kernel "
            "would splice a truncated body")
    prologue, tail = body.split(_plain.DECODE_END, 1)
    prologue = prologue + _plain.DECODE_END
    # THE CUT IS AT THE DECODE, NOT AT `int nyz`, AND THAT IS THE REAL TEMPLATE'S OWN
    # SHAPE RATHER THAN A CHOICE. `special_kz._BETA_CURL_TEMPLATE` declares
    # `int nxi = int(nx);` ABOVE `__DECODE__` and `int nyz = nyi * nzi;` below it,
    # where the COMPLEX template puts both below -- which is why the complex sibling
    # cuts at `int nyz` and this one cuts here. `h_cell` is called with nxi/nyi/nzi
    # and never with nyz, so the prologue this cut produces declares everything the
    # call needs; the assertion below requires exactly that, because a template that
    # moved `int nxi` under the decode would emit a call to an undeclared name.
    for needed in ("int nxi = int(nx);", "int nyi = int(ny), nzi = int(nz);"):
        if needed not in prologue:
            raise AssertionError(
                f"the real beta curl no longer declares {needed!r} above the decode; "
                f"this weld calls h_cell with nxi/nyi/nzi from the prologue and "
                f"would emit a call to an undeclared name")
    offsets = _plain.offset_coordinates(tail)
    for old, new in _plain.OWN_LOAD_EDITS:
        tail = _weld.needle(tail, old, new)
    for var, target in _plain.HALO_TAPS:
        prefix = f"    float {var} = "
        matches = [line + "\n" for line in tail.splitlines()
                   if line.startswith(prefix)]
        if len(matches) != 1:
            raise AssertionError(
                f"the real beta curl declares {var} {len(matches)} times; this weld "
                f"redirects exactly one magnetic load per shifted tap")
        old_line = matches[0]
        # THE GUARD AND THE OFFSET ARE BOTH PARSED FROM THE EMITTER'S OWN LINE.
        head, rest = old_line.split(" ? ", 1)
        expected = f"g{target}["
        if not rest.startswith(expected) or not rest.endswith(f" : {_GHOST};\n"):
            raise AssertionError(
                f"the real beta curl no longer reads {var} as `{expected}...]` served "
                f"an exact {_GHOST} past the wall; the ghost rule this weld preserves "
                f"has changed")
        offset = rest[len(expected):].split("]", 1)[0]
        coordinates = offsets[offset]
        call = (f"h_cell({', '.join(coordinates)}, "
                f"{_plain.H_CELL_TAIL_ARGS}).a{target}")
        tail = _weld.needle(tail, old_line, f"{head} ? {call} : {_GHOST};\n")
    for target in range(3):
        if f"g{target}[" in tail:
            raise AssertionError(
                f"the welded real beta curl half still reads g{target}; in this "
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

// SEVEN 4-BYTE MEMBERS AT 4-BYTE ALIGNMENT: 28 bytes, no padding, and the only
// unpadded Params record among the four H->D welds. The two beta words are the
// certified curl's buffers 20-21, packed; packed they cost nothing, which is the only
// reason a beta curl fits inside the 31-binding ceiling with a weld on top of it.
struct Params {
    uint nx; uint ny; uint nz; uint n_elem;
    float dtdx; float beta_plus; float beta_minus;
};

__H_CELL__

kernel void beta_real_fused_hd_pair_step(
    device float*       ho0     [[buffer(0)]],
    device float*       ho1     [[buffer(1)]],
    device float*       ho2     [[buffer(2)]],
    device float*       wo0     [[buffer(3)]],
    device float*       wo1     [[buffer(4)]],
    device float*       wo2     [[buffer(5)]],
    device const float* hi0     [[buffer(6)]],
    device const float* hi1     [[buffer(7)]],
    device const float* hi2     [[buffer(8)]],
    device const float* wi0     [[buffer(9)]],
    device const float* wi1     [[buffer(10)]],
    device const float* wi2     [[buffer(11)]],
    device const float* b0      [[buffer(12)]],
    device const float* b1      [[buffer(13)]],
    device const float* b2      [[buffer(14)]],
    device float*       f0      [[buffer(15)]],
    device float*       f1      [[buffer(16)]],
    device float*       f2      [[buffer(17)]],
    device float*       u0      [[buffer(18)]],
    device float*       u1      [[buffer(19)]],
    device float*       u2      [[buffer(20)]],
    device const float* kmx     [[buffer(21)]],
    device const float* sinvx   [[buffer(22)]],
    device const float* kmy     [[buffer(23)]],
    device const float* sinvy   [[buffer(24)]],
    device const float* kmz     [[buffer(25)]],
    device const float* sinvz   [[buffer(26)]],
    device const float* kp0     [[buffer(27)]],
    device const float* kp1     [[buffer(28)]],
    device const float* kp2     [[buffer(29)]],
    constant Params&    prm     [[buffer(30)]],
    uint idx [[thread_position_in_grid]])
{
    // THE SEVEN PACKED ARGUMENTS ARE UNPACKED INTO THE CERTIFIED BODIES' OWN NAMES,
    // once, before any lifted text runs. Everything below this line is then
    // character-for-character what `special_kz.beta_curl_source` (or its folded
    // derivative) and `shaders.constitutive_source('H')` emit, with the declared lift
    // edits and the redirected magnetic reads.
    uint nx = prm.nx, ny = prm.ny, nz = prm.nz, n_elem = prm.n_elem;
    float dtdx = prm.dtdx;
    float beta_plus = prm.beta_plus, beta_minus = prm.beta_minus;
__PROLOGUE__
    // --- THE WELD: update_H, computed into registers and stored to SCRATCH --------
    // Nothing written here is read by this launch. The curl half below takes its OWN
    // cell's magnetic field from these registers and recomputes every foreign tap
    // from PRE-LAUNCH state through the same `h_cell`, so no thread observes another
    // thread's store; the launcher rotates H/f_w_H afterwards. The BETA term's two
    // partners are `a` and `b`, which are exactly these registers.
    h_cell_result own = h_cell(i, j, k, __H_CELL_ARGS__);
    ho0[ii] = own.a0; ho1[ii] = own.a1; ho2[ii] = own.a2;
    wo0[ii] = own.src0; wo1[ii] = own.src1; wo2[ii] = own.src2;

    // --- the real beta step_D curl (stepping.step_D / _apply_pml_update:1905) -----
    // D and fu_D update IN PLACE and that is safe by construction: the curl reads and
    // writes them at the THREAD'S OWN CELL only (`f0[ii]`, `u0[ii]`).
__CURL__
}
"""


def beta_real_fused_hd_pair_source(variant: str, codes: Sequence[int],
                                   contract: str = shaders.CONTRACT_OFF) -> str:
    """One specialised fused source: (variant, boundaries, contraction mode).

    THE WALL CLEAR IS NOT CARRIED and must not be: ``zero_metal_B`` runs one seam
    EARLIER and ``zero_metal_D`` one seam LATER. On the ``folded`` variant NO FILL is
    carried either, for the same reason with a stronger premise: all six fold passes
    sit outside the span.
    """
    codes = tuple(int(code) for code in codes)
    if len(codes) != 3:
        raise ValueError(f"codes must be a per-axis triple, got {codes!r}")
    prologue, curl = welded_curl_tail(variant, codes, contract)
    return shaders.substitute(_TEMPLATE, {
        "__CONTRACT__": shaders.contraction_pragma(contract),
        "__H_CELL__": certified_h_cell(contract),
        "__PROLOGUE__": prologue,
        "__H_CELL_ARGS__": _plain.H_CELL_TAIL_ARGS,
        "__CURL__": curl,
    })


def compile_beta_real_fused_hd_pair(variant: str, codes: Sequence[int],
                                    contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised entry point for one (variant, boundaries, mode)."""
    from .device import compile_source  # noqa: PLC0415

    return getattr(
        compile_source(beta_real_fused_hd_pair_source(variant, codes, contract)),
        KERNEL)


def shipped_signature_bindings(variant: str) -> int:
    """How many bindings the SHIPPED kernel declares, counted off its own source.

    Read from the emitted text rather than from :data:`PACKED_BINDINGS`, and read PER
    VARIANT so a fold that had grown an argument would be caught rather than hidden
    behind the plain emission.
    """
    codes = ((symmetry.CODE_PERIODIC,) * 3 if variant == "plain"
             else (symmetry.CODE_PERIODIC, symmetry.CODE_MIRROR_PERIODIC,
                   symmetry.CODE_PERIODIC))
    source = beta_real_fused_hd_pair_source(variant, codes)
    signature = source.split(f"kernel void {KERNEL}(", 1)[1]
    signature = signature.split("uint idx [[thread_position_in_grid]])", 1)[0]
    return signature.count("[[buffer(")


def refuted_separate_scalar_source(contract: str = shaders.CONTRACT_OFF) -> str:
    """The same 30 pointers with the SEVEN scalars bound separately. 37 bindings.

    The gate REQUIRES the compile failure. Not a strawman: it is how
    :func:`.special_kz.beta_curl_source` binds its own arguments at its 22-binding
    size, so "packed" is a measurement on THIS signature rather than a house style --
    and the two extra bindings over the non-beta product's 35 are the beta words.
    """
    lines: List[str] = []
    slot = 0
    for label in _plain._WRITTEN:
        lines.append(f"    device float*       {label:<8}[[buffer({slot})]],")
        slot += 1
    for label in _plain._READ:
        lines.append(f"    device const float* {label:<8}[[buffer({slot})]],")
        slot += 1
    for kind, label in (("constant uint&", "nx"), ("constant uint&", "ny"),
                        ("constant uint&", "nz"), ("constant uint&", "n_elem"),
                        ("constant float&", "dtdx"),
                        ("constant float&", "beta_plus"),
                        ("constant float&", "beta_minus")):
        lines.append(f"    {kind:<19} {label:<10}[[buffer({slot})]],")
        slot += 1
    if slot != SEPARATE_SCALAR_BINDINGS:  # pragma: no cover - a design invariant
        raise AssertionError((slot, SEPARATE_SCALAR_BINDINGS))
    touch = " + ".join(f"{label}[0]" for label in _plain._READ)
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
        f"    float acc = {touch} + dtdx + beta_plus + beta_minus;",
        *[f"    acc = acc + {label}[idx];" for label in _plain._WRITTEN],
        f"    {_plain._WRITTEN[0]}[idx] = acc;",
        "}",
        "",
    ))


def enumerate_sources(contract: str = shaders.CONTRACT_OFF) -> Dict[str, str]:
    """Every specialisation a shipped plan can emit, keyed by a stable label.

    ``plain`` enumerates the eight PERIODIC/METALLIC triples; ``folded`` enumerates
    the four folded codes on every axis with at least one MIRROR present, because an
    all-unfolded quadruple is the ``plain`` variant's own territory and
    :func:`resolve_variant` would never route it here. The count is the REACHABLE set,
    so a label that cannot be built is not fingerprinted as if it could.
    """
    out: Dict[str, str] = {}
    for cx in (symmetry.CODE_PERIODIC, symmetry.CODE_METALLIC):
        for cy in (symmetry.CODE_PERIODIC, symmetry.CODE_METALLIC):
            for cz in (symmetry.CODE_PERIODIC, symmetry.CODE_METALLIC):
                codes = (cx, cy, cz)
                out[f"{KERNEL}/plain/{cx}{cy}{cz}"] = beta_real_fused_hd_pair_source(
                    "plain", codes, contract)
    codes_axis = (symmetry.CODE_PERIODIC, symmetry.CODE_METALLIC,
                  symmetry.CODE_MIRROR_METALLIC, symmetry.CODE_MIRROR_PERIODIC)
    mirrors = {symmetry.CODE_MIRROR_METALLIC, symmetry.CODE_MIRROR_PERIODIC}
    for cx in codes_axis:
        for cy in codes_axis:
            for cz in codes_axis:
                codes = (cx, cy, cz)
                if not mirrors.intersection(codes):
                    continue
                out[f"{KERNEL}/folded/{cx}{cy}{cz}"] = beta_real_fused_hd_pair_source(
                    "folded", codes, contract)
    return out


# ---------------------------------------------------------------------------
# Resolution — one row never chooses between the two variants
# ---------------------------------------------------------------------------

def resolve_variant(grid: Any) -> Optional[str]:
    """``"folded"`` when any axis carries a mirror plane, ``"plain"`` otherwise.

    THE FOLD IS READ OFF THE GRID, which is what makes this one product rather than
    two overlapping ones: a row cannot be admitted by both variants, so no composer
    order decides which kernel steps it. ``None`` when the grid cannot answer -- a
    predicate that guessed here would be the over-covering refusal this package's
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

    ``folded`` goes through ``symmetry.folded_axis_kinds``; ``plain`` uses
    ``1 if kind == "metallic" else 0``, which is what
    :func:`.special_kz.plan_beta_pml_curl` builds. Using the plain expression on a
    folded grid is the one thing this family can get wrong silently: ``"mirror"`` maps
    to 0 = PERIODIC, the backward ghost becomes a WRAP and the cell-0 mask is not
    widened, and both codes are valid so neither the emitter nor the compiler can
    catch it.
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

def metal_beta_real_fused_hd_pair_coverage(
        fields: Any, pml: Any, sources: Any = None,
        residency: Any = None) -> Coverage:
    """May ONE dispatch span real beta ``update_H`` -> withdraw -> ``step_D``?

    A conjunction of the RESOLVED VARIANT's two halves' own certified predicates plus
    the seam clauses. Nothing is weakened: a configuration either half refuses is
    refused here with that half's reasons, prefixed so a reader can tell which side
    said it. That construction is what makes the beta clause, the real-storage clause,
    the BFAST clause, the conductivity clause and the fold clause INHERITED rather
    than restated.

    THE VARIANT IS RESOLVED FIRST AND THE OTHER ONE IS NEVER ASKED. Asking both and
    taking the disjunction would admit a row on a predicate whose kernel the plan
    would not build.

    THE ORDER IS THE DRIVER'S. The constitutive half is asked FIRST because it runs
    first (driver.py:3311).

    THE TWO HALVES ARE DELIBERATELY DIFFERENT PREDICATES rather than one. The curl's
    contract refuses a conductivity on the D targets, because a conductivity routes to
    a different curl recurrence; the constitutive's deliberately does not, because a
    conductivity does not change ``update_H``. The conjunction is what refuses the
    conductive rows.
    """
    reasons: List[str] = []
    grid = getattr(fields, "grid", None)
    variant = resolve_variant(grid)
    if variant is None:
        return Coverage(False, ("fields carries no grid that can be asked whether an "
                                "axis is folded; this product resolves its variant "
                                "from that answer and refuses rather than guessing",))

    if variant == "plain":
        magnetic = special_kz.beta_run_constitutive_coverage(
            fields, pml, "H", residency)
        curl = special_kz.beta_pml_curl_coverage(fields, pml, "step_D", residency)
    else:
        magnetic = folded_beta.folded_beta_constitutive_coverage(
            fields, pml, "H", residency)
        curl = folded_beta.folded_beta_composition_curl_coverage(
            fields, pml, "step_D", residency)
    if not magnetic.covered:
        reasons.extend(f"{variant} beta constitutive half: {reason}"
                       for reason in magnetic.reasons)
    if not curl.covered:
        reasons.extend(f"{variant} beta curl half: {reason}"
                       for reason in curl.reasons)

    # THE SEAM'S ONE PASS. Nothing is INJECTED between the two consults. What IS
    # between them is the electric integrated-source withdraw (driver.py:3313-3314).
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

    for name in ROTATED_NAMES:
        if getattr(fields, name, None) is None:
            reasons.append(
                f"{name} is not allocated; this weld rotates it against a plan-owned "
                f"scratch twin after every launch")
    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class MetalBetaRealFusedHdPairPlan(_weld.ScratchWeldPairPlan):
    """The shared scratch-weld plan, carrying the facts that specialise it.

    THE ROTATION IS NOT REIMPLEMENTED. ``run`` and ``_resolve`` are the base's. What
    the subclass adds is the specialisation a harness must not re-derive: the variant
    the source was emitted for and the two rounded beta words in the packed record.
    Re-deriving either at a call site is how a mutation harness ends up compiling a
    different specialisation from the one it launches.
    """

    __slots__ = ("variant", "beta_words")

    def __init__(self, *arguments: Any, variant: str,
                 beta_words: Sequence[float]) -> None:
        super().__init__(*arguments)
        self.variant = str(variant)
        self.beta_words = tuple(float(word) for word in beta_words)


def params_record_dtype() -> Any:
    """The host record's dtype -- seven 4-byte members, itemsize 28, no padding.

    ONE HOME FOR THE LAYOUT. The offsets are stated EXPLICITLY even though they are
    the natural ones, so the record cannot drift into agreeing with Metal only by
    accident.
    """
    import numpy as np  # noqa: PLC0415

    return np.dtype({
        "names": ["nx", "ny", "nz", "n_elem", "dtdx", "beta_plus", "beta_minus"],
        "formats": ["<u4", "<u4", "<u4", "<u4", "<f4", "<f4", "<f4"],
        "offsets": [0, 4, 8, 12, 16, 20, 24],
        "itemsize": PARAMS_ITEMSIZE,
    })


def _params_tensor(shape: Sequence[int], dtdx: float, beta_words: Sequence[float],
                   device: str) -> Any:
    """The five scalars and two beta words as one 28-byte device record.

    PLAN-OWNED AND OUTSIDE THE RESIDENCY REGISTRY: :meth:`.Residency.mirror` binds
    float32 and complex64 volumes and refuses anything else BY NAME, because a wider
    element silently reinterprets. This record is neither, so it is built here, once,
    at plan time. Nothing on the launch path allocates.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    record = np.zeros(1, dtype=params_record_dtype())
    nx, ny, nz = (int(n) for n in shape)
    record["nx"], record["ny"], record["nz"] = nx, ny, nz
    record["n_elem"] = nx * ny * nz
    record["dtdx"] = np.float32(dtdx)
    plus, minus = tuple(beta_words)
    record["beta_plus"] = np.float32(plus)
    record["beta_minus"] = np.float32(minus)
    words = np.frombuffer(record.tobytes(), dtype=np.int32).copy()
    return torch.from_numpy(words).to(torch.device(device))


def plan_metal_beta_real_fused_hd_pair(
        fields: Any, pml: Any, sources: Any = None,
        residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        functions: Optional[Mapping[str, Any]] = None,
        codes: Optional[Sequence[int]] = None,
        beta_words: Optional[Sequence[float]] = None,
        ) -> Optional["_weld.ScratchWeldPairPlan"]:
    """Build the real beta fused H/D plan, or ``None`` when the seam is refused.

    ``None`` is the only refusal, for the reason :func:`.launch.plan_pml_curl` gives:
    a configuration this kernel does not carry must fall back to the array path, never
    raise into a caller that would otherwise have stepped correctly.

    THREE MUTATION SEAMS, each named because dropping one silently DISARMS a gate leg
    rather than slowing it: ``functions`` hands in a deliberately broken compiled
    source; ``codes`` hands in the PERIODIC/METALLIC triple on a folded grid;
    ``beta_words`` hands in a pair rounded through a widened chain.
    """
    if not metal_beta_real_fused_hd_pair_coverage(fields, pml, sources,
                                                  residency).covered:
        return None

    grid = fields.grid
    variant = resolve_variant(grid)
    if variant is None:  # pragma: no cover - the predicate already refused
        return None
    if codes is None:
        codes = resolve_codes(variant, grid, pml)
        if codes is None:  # pragma: no cover - the predicate already refused
            return None
    codes = tuple(int(code) for code in codes)

    if beta_words is None:
        # magnetic=False: this seam's CURL is step_D. `beta` and `dt` are passed AS
        # GIVEN, which is that function's own rule and is what keeps the coefficient
        # word the array path's bits.
        beta_words = special_kz.beta_curl_coefficients(
            grid.beta, grid.dt, magnetic=False, complex_storage=False)
    beta_words = tuple(float(word) for word in beta_words)

    volumes: List[str] = []

    def bind(name: str, host: Any, constant: bool = False) -> Any:
        volumes.append(name)
        return residency.mirror(name, host, constant=constant)

    # THE ROTATING GROUP FIRST, in ROTATED_NAMES order: every scratch, then every
    # pre-launch buffer. `ScratchWeldPairPlan.run` splices exactly those two groups
    # ahead of `static_args`, so this order IS the signature's.
    twins: Dict[str, Any] = {}
    for name in ROTATED_NAMES:
        host = getattr(fields, name)
        bind(name, host)
        twin, _mirror = _weld.scratch_twin(residency, name, host)
        twins[name] = twin

    flux = [bind(name, getattr(fields, name))
            for name in SUB_STEPS["step_B"]["targets"]]
    displacement = [bind(name, getattr(fields, name))
                    for name in SUB_STEPS["step_D"]["targets"]]
    auxiliary = [bind("fu_" + name, getattr(fields, "fu_" + name))
                 for name in SUB_STEPS["step_D"]["targets"]]

    # ONE SUB-LATTICE, TWO HALVES. Both suffixes are READ FROM THE SHIPPED TABLES and
    # their equality is ASSERTED rather than assumed: it is the whole premise of the
    # 30-pointer signature.
    suffix = SUB_STEPS["step_D"]["suffix"]
    constitutive_suffix = "_h" if CONSTITUTIVE_SIDES["H"]["half_integer"] else ""
    if suffix != constitutive_suffix:
        raise AssertionError(
            f"step_D reads the {suffix or 'integer'!r} PML sub-lattice and update_H "
            f"the {constitutive_suffix or 'integer'!r} one; this weld binds ONE kms "
            f"group for both halves and that is only correct while they agree")
    curl_coefficients = [
        bind(f"pml:{stem}_{axis}{suffix}", getattr(pml, f"{stem}_{axis}{suffix}"),
             constant=True)
        for axis in "xyz" for stem in ("kms", "sinv")]
    constitutive_coefficients = [
        bind(f"pml:kps_{axis}{constitutive_suffix}",
             getattr(pml, f"kps_{axis}{constitutive_suffix}"), constant=True)
        for axis in "xyz"]

    selected = dict(functions or {})
    for mode in contract_variants:
        if mode not in selected:
            selected[mode] = compile_beta_real_fused_hd_pair(variant, codes, mode)

    dtdx = grid.dt / grid.dx
    static = (flux + displacement + auxiliary
              + curl_coefficients + constitutive_coefficients
              + [_params_tensor(grid.shape, dtdx, beta_words, residency.device)])
    assert (2 * len(ROTATED_NAMES) + len(static)) == PACKED_BINDINGS, (
        2 * len(ROTATED_NAMES) + len(static), PACKED_BINDINGS)
    assert PACKED_BINDINGS == MAX_BUFFER_BINDINGS, (PACKED_BINDINGS,
                                                    MAX_BUFFER_BINDINGS)
    return MetalBetaRealFusedHdPairPlan(
        FAMILY, residency, fields, dict(twins), dict(selected), ROTATED_NAMES,
        tuple(static), tuple(volumes), tuple(grid.shape), tuple(codes),
        (0, 0, 0), (), REPLACES,
        variant=variant, beta_words=beta_words)


# ---------------------------------------------------------------------------
# Registration -- wired=False, gated on beta, refused by the composer on INSTALLABLE
# ---------------------------------------------------------------------------

def _has_beta(context: Any) -> bool:
    """The cheap gate that decides whether this family is CONSULTED at all.

    A gated-out arm contributes NO reason, which is right: on a beta-free run
    ``update_H`` carries other arms that will speak, and this one would only add a
    refusal about a beta nobody asked for.
    """
    grid = getattr(getattr(context, "fields", None), "grid", None)
    return grid is not None and bool(getattr(grid, "beta", 0.0))


def _arm_coverage(context: Any, slot: str) -> Coverage:
    if slot != SLOT:
        return Coverage(False, (f"real beta fused H/D pair cannot fill {slot}",))
    return metal_beta_real_fused_hd_pair_coverage(
        context.fields, context.pml, context.sources, context.residency)


def _arm_plan(context: Any, slot: str) -> Optional["_weld.ScratchWeldPairPlan"]:
    if slot != SLOT:
        return None
    return plan_metal_beta_real_fused_hd_pair(
        context.fields, context.pml, context.sources, context.residency,
        context.contract_variants)


def register_arms() -> Tuple[Any, ...]:
    """One row on ``update_H``, ``wired=False``, GATED on beta.

    ONE ROW FOR TWO CELLS, which is what "one product" means at the arm table: the
    variant is resolved inside the predicate from the grid, so the row's verdict is
    the resolved variant's and no composer order decides which kernel a folded beta
    row gets.

    REGISTERED SO IT IS ENUMERABLE, NOT SO IT IS SELECTABLE. ``arms.arms_for`` skips
    an unwired row so ``plan_step`` cannot select it, while ``arms.registered`` still
    returns it.
    """
    from . import arms  # noqa: PLC0415

    return (arms.register(FAMILY, SLOT, "fused H/D pair (real beta)",
                          _arm_coverage, _arm_plan,
                          prefix="fused H/D pair (real beta): ",
                          noun="fused real-beta H/D-curl pair",
                          gate=_has_beta,
                          wired=False, replaces=REPLACES),)


ARMS = register_arms()
