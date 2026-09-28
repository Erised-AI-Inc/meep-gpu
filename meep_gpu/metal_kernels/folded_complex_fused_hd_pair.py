"""The FOLDED COMPLEX H->D weld: folded complex ``update_H`` into folded complex
``step_D``.

THE FIFTH PRODUCT ON THE FOURTH SEAM, and the cell both of its parents refuse BY
NAME. ``h_to_d_seam.instances`` files **5** rows under the Metal arm pair
``(folded complex -> folded complex)`` and all five are ``buildable_not_built``, with
none carrying a standing in-seam withdraw:

    examples:solve-cw.py
    tests:TestArrayMetadata.test_array_metadata
    tests:TestEigCoeffs.test_binary_grating_special_kz_2_21_2
    tests:TestHoleyWvgBands.test_fields_at_kx
    tests:TestModeDecomposition.test_triangular_lattice_oblique

:mod:`.complex_fused_hd_pair` refuses every one of them because it implements the
``complex/Bloch`` arms and a folded complex run selects the ``folded complex`` arm on
both slots; :mod:`.folded_fused_hd_pair` refuses every one of them because it
implements the REAL ``folded`` arms. This module is the product on the intersection,
and it stands to its two neighbours exactly as :mod:`.folded_complex` stands to
:mod:`.complex_fields` and :mod:`.symmetry`.

=============================================================================
THIS IS THE PLAIN COMPLEX WELD WITH ONE SUBSTITUTION, AND THAT IS A STRUCTURAL
CLAIM RATHER THAN A CONVENIENCE
=============================================================================

Both halves of this seam are the plain complex product's, unchanged:

* **the constitutive half is character-for-character
  :func:`.complex_fused_hd_pair.h_cell_function`'s.** ``folded_complex`` adds NO
  device code on ``update_H``: :func:`.folded_complex.plan_folded_complex_constitutive`
  re-admits ``complex_fields.bloch_constitutive_step`` through
  ``complex_fields.ComplexConstitutivePlan``, and "only the ADMISSION is this
  family's" is that function's own sentence. So the seven declared
  :data:`.complex_fused_hd_pair.CONSTITUTIVE_LIFT_EDITS` are the folded arm's
  arithmetic too, and this module IMPORTS the lift rather than owning a copy;
* **the curl half is the certified complex curl with two blocks re-specialised.**
  :func:`.folded_complex.folded_bloch_curl_source` is ``complex_fields._CURL_TEMPLATE``
  with ONE slot inserted (``__TOP_MASK__``) and the ghost/ownership emitters driven
  over ``symmetry._reduced_codes``. Its nine magnetic load lines are
  character-identical to the certified complex curl's, which is why every table this
  module walks (:data:`.complex_fused_hd_pair.OWN_LOAD_EDITS`,
  :data:`.complex_fused_hd_pair.HALO_TAPS`,
  :func:`.complex_fused_hd_pair.offset_coordinates`) is the plain product's, imported.

So the only text this module can be blamed for is which EMITTER the curl comes from
and which CODES reach it. Both are gate legs (``transcription``, ``codes_source``)
rather than comments.

=============================================================================
THE SEAM IS FILL-FREE ON A FOLDED GRID -- INHERITED, AND RE-STATED BECAUSE THE
STORAGE DOES NOT CHANGE THE DRIVER
=============================================================================

``FdtdDriver.step`` runs ``fill_symmetry_bc_B`` (:3305), ``zero_metal_B`` (:3307) and
``fill_folded_far_ghosts_B`` (:3309) BEFORE the ``update_H`` consult (:3311), and
their D twins (:3325, :3327, :3329) AFTER the ``step_D`` consult (:3315). The only
statement inside the span is the electric integrated-source withdraw (:3313-3314).
That is a fact about the DRIVER, so it holds under complex storage exactly as
:mod:`.folded_fused_hd_pair` establishes it under real storage: **no fill is carried
here and none may be** -- a fill carried across this launch is a pass the driver runs
again. The ``complex_fill_carry`` machinery :mod:`.folded_complex_fused_pair` needs on
the D->E seam is not imported.

WHAT THE FOLDED COMPLEX FILL DOES CHANGE, and why it does not reach this kernel.
:mod:`.folded_complex` records the one arithmetic delta the composition creates: under
complex64 the array path's mirror parity is a FULL complex multiply by ``(+/-1, +0)``
with its zero cross terms, not a sign flip, so the fill gets its own body there. That
body runs in ``fill_B``/``fill_D``, both OUTSIDE this span. This product is therefore
PARITY-BLIND in exactly the sense :mod:`.folded_fused_hd_pair` is -- the emitted source
for a ``+1`` plane and a ``-1`` plane is byte-identical -- and the gate asserts it by
sha256 rather than arguing it.

=============================================================================
THE SIGNATURE IS THE PLAIN COMPLEX PRODUCT'S EXACTLY, AND THE CEILING IS STILL AN
EQUALITY
=============================================================================

    3 H_out + 3 f_w_H_out          (float2 scratch, written, never read)
  + 3 H_in  + 3 f_w_H_in           (float2, pre-launch, const)
  + 3 B                            (float2, const)
  + 3 D     + 3 fu_D               (float2, in place: own cell only)
  + 6 curl coefficients (kms/sinv per axis, ONE shared group)
  + 3 constitutive kps                                                        = 30

plus one ``constant Params&`` = **31 = :data:`.device.MAX_BUFFER_BINDINGS`**, no
headroom. THE FOLD ADDS NO KERNEL ARGUMENT UNDER COMPLEX STORAGE EITHER, for the four
reasons :mod:`.folded_fused_hd_pair` measures on the real cell and one more that is
this composition's own:

* the mirror codes are a COMPILE-TIME source specialisation
  (:func:`.folded_complex.folded_bloch_curl_source`);
* the parity is not needed: this kernel performs no fill and its ghost is the exact
  complex zero, so the product is parity-blind;
* the reflect row is not needed: the far image is a B fill before the seam and a D
  fill after it;
* no ghost-source row is needed: the near ghost's source cell is read only by the
  fill;
* **and the folded complex fill's own coefficient pair is not needed** -- it is the
  one thing this composition adds over its parents, and it lives entirely in the two
  fill kernels, which this seam does not touch.

Because the signature IS the plain complex product's, the four refuted signatures are
IMPORTED rather than re-spelled: 32 one-more-pointer, 34 unshared ``kms``, 38 separate
scalars, 62 split re/im planes. The gate compiles each on THIS host and requires the
failure; it does not inherit the sibling's verdict.

=============================================================================
WHERE THE CODES COME FROM, AND WHY IT IS THE ONE THING THIS MODULE CAN GET WRONG
SILENTLY
=============================================================================

``codes`` MUST come from ``symmetry.folded_axis_kinds(grid, pml)``. The plain complex
product builds its triple as ``1 if kind == "metallic" else 0`` over
``stepping._boundary_kinds`` (complex_fused_hd_pair.py's ``plan_...``); on a folded
grid ``_boundary_kinds`` reports ``"mirror"``, which that expression maps to
0 = PERIODIC -- so the backward ghost becomes a WRAP to the far plane and the cell-0
mask is not widened. Neither :func:`.folded_complex.folded_bloch_curl_source` nor the
compiler can catch it (0 and 1 are valid codes there). It is a GATE MUTATION
(``codes_from_boundary_kinds``), not a comment.

THE PHASE FLAGS COME FROM ``_boundary_kinds``, NOT FROM THE FOLDED CODES, and that is
:func:`.folded_complex.plan_folded_complex_pml_curl`'s own choice restated: a folded
axis is refused a Bloch phase by that family's clause, so the two agree on every
admitted run, and reading the phase off the RESOLVED kinds keeps the conjugation rule
in one place (``complex_fields.phase_arguments``, ``backward=True`` on this seam).

=============================================================================
WHAT SITS IN THE SEAM, AND WHAT THIS PRODUCT REFUSES
=============================================================================

:data:`CARRIES_DEPOSIT_REPAIR` is False and it is a FACT about the driver rather than
a choice: nothing is INJECTED between the two consults, so :mod:`..deposit_repair` has
nothing to say here.

:data:`HOISTS_THE_WITHDRAW` is False. The only wiring that performs the hoist is
``launch._install_fused_pair``'s ``withdraw_hoist.SEAM`` branch; this product declares
:data:`INSTALLABLE` False, so that branch is unreachable for it and nothing would
perform the withdraw before its launch. The predicate therefore refuses BY NAME every
row whose electric withdraw does work. **On this cell that refuses zero rows** -- the
seam record measures 0 of 5 with ``withdraw_in_seam`` -- and the clause is kept in
full anyway: ignorance is never an empty set, and a row that grows an integrated
source later must be refused rather than silently served.

REFUSED, each through a parent predicate this one conjoins without weakening: real
float32 storage (:mod:`.folded_fused_hd_pair`'s cell), an UNFOLDED complex grid
(:mod:`.complex_fused_hd_pair`'s cell), ``beta``/BFAST (:mod:`.folded_beta`'s and
:mod:`.beta_complex_fused_hd_pair`'s), cylindrical, conductive, a live susceptibility,
an off-diagonal epsilon row and an inactive absorber. The fold clause is INVERTED
against :mod:`.complex_fused_hd_pair`'s in both directions, which is what keeps every
folded complex row unambiguous.

=============================================================================
IT IS NOT INSTALLED, AND THE REASON IS ARBITRATION RATHER THAN ARITHMETIC
=============================================================================

:data:`INSTALLABLE` is False, and :data:`INSTALLABLE_REASON` carries the MEASUREMENT
rather than the argument -- driven through the SHIPPED Metal composer on this cell's
own fixtures, not inferred from the board. Over the driver's ``step_B - update_H -
step_D - update_E`` slot path launches are ``4 - (installed pairs)``, and a product
spanning ``update_H``/``step_D`` takes one slot from EACH neighbour.

**NO TIMING EXISTS FOR THIS SHAPE AND NONE IS LICENSED BY ANYTHING IN THIS MODULE.**
A second Metal lane was running on this GPU while this family was gated, so no
wall-clock number taken in that window would mean anything even if one had been taken.

WHAT THE FAMILY'S CREDITED INSTANCES MEAN, in the words a search will find: the
board's 5 instances are **served by predicate admission** and nothing more; the
product **installs on zero rows**; it **executes nowhere** outside its own gate and
tests; and it licenses **no timing claim and no dispatch claim**.

=============================================================================
IS THIS A SECOND PRODUCT, OR A VARIANT OF THE PLAIN COMPLEX ONE?
=============================================================================

**A LATER MERGE INTO ONE TWO-VARIANT PRODUCT IS THE RIGHT END STATE, AND IT IS NOT
WHAT THIS ROUND SHIPS.** The CUDA sibling ``cuda_kernels/complex_fused_hd_pair.py``
already covers both cells as one product -- one transform emitting two device strings,
plain welding the certified complex Bloch curl and folded the certified folded-complex
one, with ``resolve()`` reading the fold off the grid -- and everything that argues for
that shape holds here: the signature, the ``Params`` layout, the ``h_cell`` lift, the
rotation order, the binding count and all four refuted signatures are IDENTICAL, and
the two predicates differ in exactly one clause, inverted. Merging would put the fold
where it belongs, as a source specialisation beside the boundary codes.

Two things stop it in this round and both are mechanical rather than technical: the
plain module's bytes are pinned by gate campaigns in flight, so editing it re-stales
every weld that records it and costs a three-backend re-gate; and a merged product
would need ONE arm row admitting both cells, which changes what
``launch.FUSED_PAIR_ARMS`` files and what the board joins on. Both are wiring, and the
wiring for this round is written as a diff rather than applied. Until then the
relationship is expressed the way the tree already expresses it between
:mod:`.fused_hd_pair` and :mod:`.folded_fused_hd_pair`: a second module that imports
the first's tables and inverts one clause of its predicate.

NO TORCH AT MODULE LEVEL, as everywhere in this package.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .. import withdraw_hoist as _withdraw_hoist
from ..triton_kernels.coverage import CONSTITUTIVE_SIDES, Coverage
from . import complex_fields
from . import complex_fused_hd_pair as _plain
from . import folded_complex
from . import offdiag_weld_common as _weld
from . import shaders, symmetry, templates
from .device import MAX_BUFFER_BINDINGS, Residency

FAMILY = "folded_complex_fused_hd_pair"

#: The sub-step slot this arm is registered on: the FIRST half of the seam, in the
#: driver's own order, so a refusal is named on the slot the fusion starts at.
SLOT = "update_H"

#: The driver passes one launch of this plan performs, in driver order
#: (driver.py:3311, :3315). Declared rather than inferred from the slot name.
#:
#: NOTHING BETWEEN THEM IS CARRIED, and on a folded grid there is still only one
#: thing there to carry: the electric withdraw (:3313-3314), which
#: :data:`HOISTS_THE_WITHDRAW` declines. Both mirror fills, the wall clear and both
#: far passes sit OUTSIDE this span -- three before ``update_H`` and three after
#: ``step_D``.
REPLACES: Tuple[str, ...] = ("update_H", "step_D")

#: The seam name ``launch.FUSED_PAIR_SEAMS`` files this row under, spelled through
#: :mod:`..withdraw_hoist` rather than as a literal so the two cannot drift.
SEAM: str = _withdraw_hoist.SEAM

#: The ONE variant this product has, and the board's ``(update_H arm, step_D arm)``
#: pair for its cell. Spelled the way the two beta H->D products spell theirs so a
#: reader and a harness can join all three the same way; here the map has one entry
#: because a folded complex run has no second emitter to choose between.
VARIANTS: Tuple[str, ...] = ("folded",)
VARIANT_CELLS: Dict[str, Tuple[str, str]] = {
    "folded": ("folded complex", "folded complex")}

#: Nothing is injected between the two consults, so there is no deposit in this seam
#: to carry across the launch and ``deposit_repair`` is not consulted at all.
CARRIES_DEPOSIT_REPAIR = False

#: This product does not perform the seam's electric withdraw, so its predicate
#: refuses every row with a standing one BY NAME. On this cell that is 0 of 5.
HOISTS_THE_WITHDRAW = False

#: May the composer install this product? NO. See :data:`INSTALLABLE_REASON` and the
#: gate's ``arbitration`` leg, which MEASURES the claim through the shipped composer.
INSTALLABLE = False

INSTALLABLE_REASON = (
    "arbitration, not arithmetic, and MEASURED through the SHIPPED Metal composer "
    "(launch.plan_step, fuse=True) on this cell's own fixtures rather than inferred "
    "from a board. Over the driver's step_B - update_H - step_D - update_E slot path "
    "launches are 4 - (installed pairs), and a product spanning update_H/step_D "
    "takes one slot from EACH neighbour. Both neighbouring seams of this cell carry "
    "released Metal products -- folded_complex_fused_magnetic_pair on "
    "step_B/update_H and folded_complex_fused_pair on step_D/update_E -- so "
    "wherever both install, installing this product is a strict LOSS (2 pairs -> 1: "
    "one more launch and one fewer seam served) and wherever exactly one installs it "
    "is a TIE that the released precedent resolves for the incumbent, because the "
    "B->H pair installs first and holds update_H and _pair_may_absorb then refuses "
    "this product by name. The per-fixture verdict is the gate's `arbitration` leg, "
    "row by row, in parity/meep_gpu/results/"
    "metal_folded_complex_fused_hd_pair_2026-09-08_weld/flush/gate.json. What that "
    "leaves: the board's credited instances are served by PREDICATE ADMISSION only; "
    "the product installs on zero rows; it executes nowhere outside its own gate and "
    "tests; and the weld licenses no timing claim and no dispatch claim. The one "
    "shape that could pay -- one product owning step_B through update_E in two "
    "launches -- composes from pieces that all now exist on this backend and is not "
    "built here")

#: WHAT A RELEASE STILL OWES. Empty means "this family holds a released device gate
#: bound in ``fingerprints.json``"; non-empty means it does not, and the string is
#: the debt. ``test_metal_weld_contract.test_every_metal_family_is_welded``
#: partitions the ``gate_metal_*.py`` fleet against that file and requires the
#: declaring set and the welded set to be DISJOINT and to EXHAUST it, so this
#: declaration is the only way a gated family may stand unwelded.
#:
#: EMPTY SINCE 2026-09-08: ``fingerprints.json`` carries
#: ``metal_folded_complex_fused_hd_pair_device_gate``, minted by
#: ``mint_metal_weld.py --family folded_complex_fused_hd_pair`` from the RELEASED
#: artifact ``parity/meep_gpu/results/
#: metal_folded_complex_fused_hd_pair_2026-09-08_weld/flush/gate.json``. That gate was
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
    "the device gate parity/meep_gpu/gate_metal_folded_complex_fused_hd_pair.py has "
    "RUN and RELEASED on this host under BOTH float32 subnormal policies from empty "
    "policy-separated caches, into "
    "parity/meep_gpu/results/metal_folded_complex_fused_hd_pair_2026-09-07/"
    "{keep,flush}/gate.json. What a release still owes is the LEDGER, not a "
    "measurement: the metal_folded_complex_fused_hd_pair_device_gate entry in "
    "metal_kernels/fingerprints.json binding that artifact, minted by "
    "parity/meep_gpu/mint_metal_weld.py, together with the registry.FAMILY_MODULES "
    "row and the launch.FUSED_PAIR_ARMS absorb row. The 2026-09-07 unbuilt-H_to_D "
    "round writes all of those as a unified diff (wiring.patch) rather than applying "
    "them, because three further gate campaigns in flight pin the current bytes of "
    "every existing module and a shared-file edit re-stales each of them. Empty this "
    "string in the same change that lands the weld entry")

#: The binding shape is the plain COMPLEX product's EXACTLY -- imported rather than
#: re-spelled, because it is the same signature (the fold adds no argument on this
#: backend). The gate re-measures each of the four on this host rather than
#: inheriting the number.
PACKED_BINDINGS = _plain.PACKED_BINDINGS
ONE_MORE_POINTER_BINDINGS = _plain.ONE_MORE_POINTER_BINDINGS
UNSHARED_KMS_BINDINGS = _plain.UNSHARED_KMS_BINDINGS
SEPARATE_SCALAR_BINDINGS = _plain.SEPARATE_SCALAR_BINDINGS
SPLIT_PLANE_BINDINGS = _plain.SPLIT_PLANE_BINDINGS

#: The host record's total size in bytes -- the plain complex product's, because the
#: struct is the plain complex product's. Stated because the natural numpy record is
#: 44 bytes and Metal's struct is 48.
PARAMS_ITEMSIZE = _plain.PARAMS_ITEMSIZE

#: The rotating volumes, in SIGNATURE ORDER. The plain complex product's, imported:
#: this weld has the same six and the same order, and two spellings of one order is
#: how a launch binds the wrong buffer to every argument after the first mismatch.
ROTATED_NAMES: Tuple[str, ...] = _plain.ROTATED_NAMES

#: The kernel entry point. The plain complex product's, because the TEMPLATE is the
#: plain complex product's -- only the curl body substituted into it is folded.
#: :func:`compile_folded_complex_fused_hd_pair` is the only caller that names it.
KERNEL = _plain.KERNEL

#: The complex ghost this weld preserves, spelled through the shipped constant.
_GHOST = templates.COMPLEX_ZERO

if PACKED_BINDINGS > MAX_BUFFER_BINDINGS:  # pragma: no cover - a design invariant
    raise RuntimeError(
        f"the folded complex fused H/D pair binds {PACKED_BINDINGS} buffers and "
        f"Metal's ceiling on this toolchain is {MAX_BUFFER_BINDINGS}")

__all__ = [
    "ARMS", "CARRIES_DEPOSIT_REPAIR", "FAMILY", "HOISTS_THE_WITHDRAW",
    "INSTALLABLE", "INSTALLABLE_REASON", "KERNEL", "ONE_MORE_POINTER_BINDINGS",
    "PACKED_BINDINGS", "PARAMS_ITEMSIZE", "REPLACES", "ROTATED_NAMES", "SEAM",
    "SEPARATE_SCALAR_BINDINGS", "SLOT", "SPLIT_PLANE_BINDINGS",
    "UNSHARED_KMS_BINDINGS", "VARIANTS", "VARIANT_CELLS", "WELD_OWED",
    "compile_folded_complex_fused_hd_pair", "enumerate_sources",
    "folded_complex_fused_hd_pair_source", "folded_welded_curl_tail",
    "metal_folded_complex_fused_hd_pair_coverage",
    "plan_metal_folded_complex_fused_hd_pair", "refuted_one_more_pointer_source",
    "refuted_separate_scalar_source", "refuted_unshared_kms_source",
    "register_arms", "shipped_signature_bindings", "split_plane_pair_signature",
]

#: The four refuted signatures, IMPORTED. They measure the SIGNATURE, and this
#: product's signature is the plain complex one's -- 30 pointers plus one packed
#: ``Params`` -- so re-spelling them here would be a second place for the same
#: measurement to go stale. :mod:`.folded_fused_hd_pair` imports its own twin's the
#: same way.
refuted_one_more_pointer_source = _plain.refuted_one_more_pointer_source
refuted_unshared_kms_source = _plain.refuted_unshared_kms_source
refuted_separate_scalar_source = _plain.refuted_separate_scalar_source
split_plane_pair_signature = _plain.split_plane_pair_signature

#: The host record's dtype -- the plain complex product's, imported for the same
#: reason the rotation order is: one home for a layout whose failure mode is a
#: plausible complex number rather than a crash.
params_record_dtype = _plain.params_record_dtype


# ---------------------------------------------------------------------------
# The lift: the FOLDED COMPLEX step_D curl with every H read redirected
# ---------------------------------------------------------------------------

def folded_welded_curl_tail(codes: Sequence[int], phased: Sequence[int],
                            expansion: str,
                            contract: str = shaders.CONTRACT_OFF
                            ) -> Tuple[str, str]:
    """``(prologue, body)`` of the FOLDED complex ``step_D`` curl, H reads redirected.

    THE BODY IS :func:`.complex_fused_hd_pair.welded_curl_tail`'S, WITH ONE
    SUBSTITUTION: the source is :func:`.folded_complex.folded_bloch_curl_source`
    rather than ``complex_fields.bloch_curl_source``. Everything else -- the body
    anchor, the prologue cut, the offset parse, the three own-cell load edits, the six
    halo redirects, the ghost spelling and the final "no ``gN[`` survives" check -- is
    the plain complex module's own tables and helpers, imported and applied here,
    because the folded template's load lines ARE the certified complex template's own
    (``folded_curl_template`` inserts one block and changes nothing else). A needle
    that stopped matching RAISES here rather than emitting a kernel that reads the
    wrong volume.

    ``backward`` is True and never a parameter -- this pair is the D seam, and
    ``step_B``'s forward strides and unconjugated phase are a different product.

    THE TWO FOLD-SPECIFIC BLOCKS SURVIVE THE LIFT UNTOUCHED, and that is a property of
    what they touch rather than an exemption. ``folded_top_plane_mask`` writes
    ``curlN = last_a ? float2(0.0f, 0.0f) : curlN;`` -- no ``gN[`` -- and the widened
    cell-0 mask writes ``curlN = at_a ? float2(0.0f, 0.0f) : curlN;``, likewise. Only
    the six shifted MAGNETIC loads are redirected, and the ghost ternary's guard and
    offset are PARSED OUT of the emitter's own line rather than retyped, so a folded
    axis's exact complex zero stays exact and no recompute is evaluated past the face.

    THE BLOCH PHASE IS NOT IN THE RECOMPUTE AND MUST NOT BE. The folded complex curl
    applies ``complex_fields._phase_block`` to the LOADED REGISTER after the gather,
    on the wrapped lane only; this weld redirects the LOAD and leaves the phase block
    standing character for character.

    ``codes`` is the four-code folded quadruple :func:`.symmetry.folded_axis_kinds`
    resolves, NEVER a hand-built PERIODIC/METALLIC triple -- see the module docstring.
    """
    source = folded_complex.folded_bloch_curl_source(codes, True, phased, expansion,
                                                     contract)
    if _plain._BODY_ANCHOR not in source:
        raise AssertionError(
            "the folded complex curl source no longer carries the body anchor; the "
            "fused kernel would splice a truncated body")
    body = source.split(_plain._BODY_ANCHOR, 1)[1]
    if not body.endswith("}\n"):
        raise AssertionError("the folded complex curl source does not end with '}'")
    body = body[: -len("}\n")]
    if _plain.PROLOGUE_END not in body:
        raise AssertionError(
            "the folded complex curl no longer declares `int nyz = nyi * nzi;`; this "
            "weld cuts its prologue there because its first statement calls h_cell "
            "with nxi/nyi/nzi")
    prologue, tail = body.split(_plain.PROLOGUE_END, 1)
    prologue = prologue + _plain.PROLOGUE_END
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
                f"the folded complex curl declares {var} {len(matches)} times; this "
                f"weld redirects exactly one magnetic load per shifted tap")
        old = matches[0]
        # THE GUARD AND THE OFFSET ARE BOTH PARSED FROM THE EMITTER'S OWN LINE.
        head, rest = old.split(" ? ", 1)
        expected = f"g{target}["
        if not rest.startswith(expected) or not rest.endswith(f" : {_GHOST};\n"):
            raise AssertionError(
                f"the folded complex curl no longer reads {var} as `{expected}...]` "
                f"served an exact {_GHOST} past the face; the ghost rule this weld "
                f"preserves has changed, and on a fold that rule is what makes the "
                f"ghost DEAD")
        offset = rest[len(expected):].split("]", 1)[0]
        coordinates = offsets[offset]
        call = (f"h_cell({', '.join(coordinates)}, "
                f"{_plain.H_CELL_TAIL_ARGS}).a{target}")
        tail = _weld.needle(tail, old, f"{head} ? {call} : {_GHOST};\n")
    for target in range(3):
        if f"g{target}[" in tail:
            raise AssertionError(
                f"the welded folded complex curl half still reads g{target}; in this "
                f"signature that pointer does not exist and every magnetic read must "
                f"be the register or a recompute")
    return prologue, tail


def folded_complex_fused_hd_pair_source(
        codes: Sequence[int], phased: Sequence[int], expansion: str,
        contract: str = shaders.CONTRACT_OFF) -> str:
    """One specialised source: (folded quadruple, phases, expansion arm, mode).

    The plain complex product's :data:`.complex_fused_hd_pair._TEMPLATE` with the
    FOLDED complex curl's prologue and the FOLDED welded tail. The kernel entry point
    keeps the plain product's name because the template is the plain product's.

    NO FILL AND NO WALL CLEAR IS CARRIED, and on a folded grid that is a stronger
    statement than on an unfolded one: ``fill_symmetry_bc_B``, ``zero_metal_B`` and
    ``fill_folded_far_ghosts_B`` all close BEFORE the ``update_H`` consult, and their
    D twins all open AFTER the ``step_D`` consult. A fill carried here would be a pass
    the driver runs again -- and under complex storage that fill is not even the same
    arithmetic (``folded_complex``'s parity multiply), which is a second, independent
    reason not to reach for it.
    """
    codes = tuple(int(code) for code in codes)
    phased = tuple(int(flag) for flag in phased)
    if len(codes) != 3:
        raise ValueError(f"codes must be a per-axis triple, got {codes!r}")
    if len(phased) != 3:
        raise ValueError(f"phased must be a per-axis triple, got {phased!r}")
    prologue, curl = folded_welded_curl_tail(codes, phased, expansion, contract)
    return templates.substitute(_plain._TEMPLATE, {
        "__CONTRACT__": shaders.contraction_pragma(contract),
        "__HELPERS__": templates.complex_helpers(expansion),
        "__H_CELL__": _plain.h_cell_function(expansion, contract),
        "__PROLOGUE__": prologue,
        "__H_CELL_ARGS__": _plain.H_CELL_TAIL_ARGS,
        "__CURL__": curl,
    })


def compile_folded_complex_fused_hd_pair(
        codes: Sequence[int], phased: Sequence[int], expansion: str,
        contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised entry point for one (folded codes, phases, arm, mode)."""
    from .device import compile_source  # noqa: PLC0415

    return getattr(
        compile_source(folded_complex_fused_hd_pair_source(codes, phased, expansion,
                                                           contract)),
        KERNEL)


def shipped_signature_bindings(expansion: str) -> int:
    """How many bindings the SHIPPED folded kernel declares, off its own source.

    Read from the emitted text rather than from :data:`PACKED_BINDINGS`, so the
    constant is a claim the source can falsify -- and read on a FOLDED quadruple, so a
    fold that had grown an argument would be caught here rather than on the degenerate
    unfolded emission.
    """
    source = folded_complex_fused_hd_pair_source(
        (symmetry.CODE_PERIODIC, symmetry.CODE_MIRROR_PERIODIC,
         symmetry.CODE_PERIODIC), (0, 0, 0), expansion)
    signature = source.split(f"kernel void {KERNEL}(", 1)[1]
    signature = signature.split("uint idx [[thread_position_in_grid]])", 1)[0]
    return signature.count("[[buffer(")


def _phase_triples(codes: Sequence[int]) -> Tuple[Tuple[int, int, int], ...]:
    """Every phase triple legal against one folded quadruple.

    A MIRROR axis cannot carry a Bloch phase (a mirror plane reflects rather than
    repeating), and neither can a plain METALLIC one. So the reachable set is the
    product over the PERIODIC axes only, which is what keeps
    :func:`enumerate_sources` from fingerprinting a label that cannot be built.
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

    Enumerated over the FOUR folded codes on every axis with at least one MIRROR
    present -- an all-unfolded quadruple is :mod:`.complex_fused_hd_pair`'s cell and
    this family's predicate refuses it -- and over the phase triples each quadruple
    can legally carry. The count is the REACHABLE set rather than the Cartesian
    product.
    """
    codes_axis = (symmetry.CODE_PERIODIC, symmetry.CODE_METALLIC,
                  symmetry.CODE_MIRROR_METALLIC, symmetry.CODE_MIRROR_PERIODIC)
    mirrors = {symmetry.CODE_MIRROR_METALLIC, symmetry.CODE_MIRROR_PERIODIC}
    out: Dict[str, str] = {}
    for cx in codes_axis:
        for cy in codes_axis:
            for cz in codes_axis:
                codes = (cx, cy, cz)
                if not mirrors.intersection(codes):
                    continue
                for phased in _phase_triples(codes):
                    label = (f"{KERNEL}/folded/{cx}{cy}{cz}/"
                             f"ph{phased[0]}{phased[1]}{phased[2]}/{expansion}")
                    out[label] = folded_complex_fused_hd_pair_source(
                        codes, phased, expansion, contract)
    return out


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def metal_folded_complex_fused_hd_pair_coverage(
        fields: Any, pml: Any, sources: Any = None, residency: Any = None,
        probe: Any = None) -> Coverage:
    """May ONE dispatch span folded complex ``update_H`` -> withdraw -> ``step_D``?

    A conjunction of the two FOLDED COMPLEX halves' own certified predicates plus the
    seam clauses. Nothing is weakened: a configuration either half refuses is refused
    here with that half's reasons, prefixed so a reader can tell which side said it.
    That construction is what makes the expansion-probe clause, the complex-storage
    clause, the beta and BFAST clauses, the off-diagonal clause, the polarization
    clause and the fold clause INHERITED rather than restated.

    THE ORDER IS THE DRIVER'S. The constitutive half is asked FIRST because it runs
    first (driver.py:3311), so the first refusal a reader sees names the half the
    driver would have reached first.

    THE TWO HALVES ARE DELIBERATELY DIFFERENT PREDICATES rather than one. The curl's
    contract refuses a conductivity on the D targets, because a conductivity routes to
    a different curl recurrence; the constitutive's deliberately does not, because a
    conductivity does not change ``update_H``. The conjunction is what refuses the
    conductive rows, and inheriting one half's clause list for the other would
    silently narrow an independent sub-step.
    """
    reasons: List[str] = []

    magnetic = folded_complex.folded_complex_constitutive_coverage(
        fields, pml, "H", residency, probe)
    if not magnetic.covered:
        reasons.extend(f"folded complex constitutive half: {reason}"
                       for reason in magnetic.reasons)
    curl = folded_complex.folded_complex_composition_curl_coverage(
        fields, pml, "step_D", residency, probe)
    if not curl.covered:
        reasons.extend(f"folded complex curl half: {reason}"
                       for reason in curl.reasons)

    # THE SEAM'S ONE PASS. Nothing is INJECTED between the two consults, so
    # `deposit_repair` is not consulted at all and a source of either polarity is not
    # this seam's business -- the magnetic injection is one seam earlier and the
    # electric one is one seam later. What IS between them is the electric
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

    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False,
                        tuple(dict.fromkeys(reasons + ["fields carries no grid"])))

    # THE FOLD, REQUIRED, AND RESTATED AT PRODUCT LEVEL. Both halves already require a
    # mirror plane. The restatement is so the reason reads AT THE SEAM rather than
    # only at the arms, and it is the exact mirror image of `complex_fused_hd_pair`'s
    # clause, which refuses a fold naming this cell: the two predicates can never both
    # admit, and neither can be widened into the other without making every folded
    # complex row ambiguous.
    mirrored = getattr(grid, "is_mirrored", None)
    folded_axes = [axis for axis in range(3)
                   if callable(mirrored) and bool(mirrored(axis))]
    if not folded_axes:
        reasons.append(
            "no axis is folded by a mirror plane: this product implements the "
            "`folded complex` update_H and step_D arms, and an unfolded complex run "
            "selects the `complex/Bloch` arms, which is complex_fused_hd_pair's cell "
            "and its product")

    # THE ROTATION IS THE PRODUCT'S OWN INVARIANT. The plan swaps the ENGINE's
    # references for the six volumes in ROTATED_NAMES after each launch, so those
    # attributes must be settable and must be the arrays the residency mirrored.
    for name in ROTATED_NAMES:
        if getattr(fields, name, None) is None:
            reasons.append(
                f"{name} is not allocated; this weld rotates it against a plan-owned "
                f"scratch twin after every launch")
    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class MetalFoldedComplexFusedHdPairPlan(_weld.ScratchWeldPairPlan):
    """The shared scratch-weld plan, carrying the three facts that specialise it.

    THE ROTATION IS NOT REIMPLEMENTED. ``run`` and ``_resolve`` are the base's, so
    this family's post-launch buffer rotation is the same code the four D->E welds and
    the three other H->D products execute. What the subclass adds is the
    specialisation a harness must not re-derive: the per-axis phase flags the source
    was baked for, the rounded and conjugated phase values in the packed record, and
    the PROBE-MEASURED expansion arm. A base with ``__slots__`` refuses to carry them,
    and re-deriving them at a call site is how a mutation harness ends up compiling a
    different specialisation from the one it launches.
    """

    #: ``family`` is the BASE's slot and is set from the constructor argument; it is
    #: deliberately NOT shadowed by a class attribute here, which on a slotted base
    #: would hide the descriptor and make the base's own assignment raise.
    __slots__ = ("phased", "phase_values", "expansion")

    def __init__(self, *arguments: Any, phased: Sequence[int],
                 phase_values: Sequence[Tuple[float, float]],
                 expansion: str) -> None:
        super().__init__(*arguments)
        self.phased = tuple(int(flag) for flag in phased)
        self.phase_values = tuple((float(re), float(im)) for re, im in phase_values)
        self.expansion = str(expansion)


def plan_metal_folded_complex_fused_hd_pair(
        fields: Any, pml: Any, sources: Any = None,
        residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        functions: Optional[Mapping[str, Any]] = None,
        probe: Any = None,
        codes: Optional[Sequence[int]] = None,
        ) -> Optional["_weld.ScratchWeldPairPlan"]:
    """Build the folded complex fused H/D plan, or ``None`` when the seam is refused.

    ``None`` is the only refusal, for the reason :func:`.launch.plan_pml_curl` gives:
    a configuration this kernel does not carry must fall back to the array path, never
    raise into a caller that would otherwise have stepped correctly.

    ``functions`` is the SHADER mutation seam and ``codes`` the STRUCTURAL one. The
    gate compiles a deliberately broken copy of the shipped source and hands it
    through the first; through the second it hands the PERIODIC/METALLIC triple the
    plain complex product builds from ``_boundary_kinds``, which on a folded grid is a
    smooth wrong answer rather than a crash and is the one structural defect this
    family can make silently. Dropping either argument is not a slowdown, it is a
    silent DISARMING: every mutation leg would then launch the shipped kernel and
    report the defect as uncaught.
    """
    if not metal_folded_complex_fused_hd_pair_coverage(
            fields, pml, sources, residency, probe).covered:
        return None
    expansion = folded_complex.expansion_from_probe(
        probe if probe is not None else folded_complex.load_expansion_probe())
    if expansion is None:  # pragma: no cover - the predicate already refused
        return None
    from ..stepping import _boundary_kinds  # noqa: PLC0415

    grid = fields.grid
    if codes is None:
        # FROM `folded_axis_kinds` AND FROM NOWHERE ELSE. It routes the ghost rule
        # through `stepping._boundary_kinds` (where a fold outranks the outer
        # declaration), splits MIRROR through `stepping._stored_past_owned`, and
        # cross-checks that split against `grid.is_metallic` -- two independent routes
        # to the same fact, with a disagreement a refusal rather than a coin toss.
        codes, _reasons = symmetry.folded_axis_kinds(grid, pml)
        if codes is None:  # pragma: no cover - the predicate already refused
            return None
    codes = tuple(int(code) for code in codes)

    # THE PHASE FLAGS COME FROM `_boundary_kinds`, NOT FROM THE FOLDED CODES, which is
    # `folded_complex.plan_folded_complex_pml_curl`'s own choice: a folded axis is
    # refused a phase by that family's clause, so the two agree on every admitted run,
    # and reading the phase off the RESOLVED kinds keeps the conjugation in one place.
    # backward=True: this is the D seam.
    kinds = _boundary_kinds(grid, pml)
    flags, values = complex_fields.phase_arguments(
        complex_fields.bloch_phase_table(grid, kinds), backward=True)

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
            for name in ("Bx", "By", "Bz")]
    displacement = [bind_complex(name, getattr(fields, name))
                    for name in ("Dx", "Dy", "Dz")]
    auxiliary = [bind_complex("fu_" + name, getattr(fields, "fu_" + name))
                 for name in ("Dx", "Dy", "Dz")]

    # ONE SUB-LATTICE, TWO HALVES -- and the fold does not touch it. The PML
    # coefficient vectors are built at the FOLDED STORED EXTENT already, so the
    # sharing argument is exactly the plain complex product's: both suffixes are READ
    # FROM THE SHIPPED TABLES and their equality is ASSERTED rather than assumed. It
    # is the whole premise of the 30-pointer signature, and if either table moved,
    # sharing the group would bind one half's coefficients to the other half's
    # lattice -- a converged, smooth, half-cell-wrong absorber profile.
    suffix = complex_fields.SUB_STEPS["step_D"]["suffix"]
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
            selected[mode] = compile_folded_complex_fused_hd_pair(
                codes, flags, expansion, mode)

    dtdx = grid.dt / grid.dx
    static = (flux + displacement + auxiliary
              + curl_coefficients + constitutive_coefficients
              + [_plain._params_tensor(grid.shape, dtdx, values, residency.device)])
    assert (2 * len(ROTATED_NAMES) + len(static)) == PACKED_BINDINGS, (
        2 * len(ROTATED_NAMES) + len(static), PACKED_BINDINGS)
    assert PACKED_BINDINGS == MAX_BUFFER_BINDINGS, (PACKED_BINDINGS,
                                                    MAX_BUFFER_BINDINGS)
    # THE ARGUMENT ORDER IS `ScratchWeldPairPlan.__init__`'s, positionally, so this
    # family and the shared plan cannot drift on what each slot means.
    return MetalFoldedComplexFusedHdPairPlan(
        FAMILY, residency, fields, dict(twins), dict(selected), ROTATED_NAMES,
        tuple(static), tuple(volumes), tuple(grid.shape), tuple(codes),
        (0, 0, 0), (), REPLACES,
        phased=flags, phase_values=values, expansion=expansion)


# ---------------------------------------------------------------------------
# Registration -- wired=False, gated on the fold, refused by the composer on
# INSTALLABLE
# ---------------------------------------------------------------------------

def _arm_coverage(context: Any, slot: str) -> Coverage:
    if slot != SLOT:
        return Coverage(
            False, (f"folded complex fused H/D pair cannot fill {slot}",))
    return metal_folded_complex_fused_hd_pair_coverage(
        context.fields, context.pml, context.sources, context.residency,
        context.extra.get("folded_complex_probe"))


def _arm_plan(context: Any, slot: str) -> Optional["_weld.ScratchWeldPairPlan"]:
    if slot != SLOT:
        return None
    return plan_metal_folded_complex_fused_hd_pair(
        context.fields, context.pml, context.sources, context.residency,
        context.contract_variants,
        probe=context.extra.get("folded_complex_probe"))


def register_arms() -> Tuple[Any, ...]:
    """One row on ``update_H``, ``wired=False``, GATED on the fold.

    REGISTERED SO IT IS ENUMERABLE, NOT SO IT IS SELECTABLE. ``arms.arms_for`` skips
    an unwired row so ``plan_step`` cannot select it, while ``arms.registered`` still
    returns it -- which is what lets a composition sweep measure disjointness against
    this predicate instead of assuming it, and what lets
    ``launch._neighbouring_seam_claimant`` see the row when it asks who claims the
    seam ``update_H`` opens.

    THE GATE IS ``symmetry._has_fold``, the folded families' own convention: a
    gated-out arm contributes NO reason, which is right, because on an unfolded run
    ``update_H`` carries other arms that will speak and this one would only add a
    refusal about a fold nobody asked for.
    """
    from . import arms  # noqa: PLC0415

    return (arms.register(FAMILY, SLOT, "fused H/D pair (folded complex)",
                          _arm_coverage, _arm_plan,
                          prefix="fused H/D pair (folded complex): ",
                          noun="fused folded-complex-H/folded-complex-D-curl pair",
                          gate=symmetry._has_fold,
                          wired=False, replaces=REPLACES),)


ARMS = register_arms()
