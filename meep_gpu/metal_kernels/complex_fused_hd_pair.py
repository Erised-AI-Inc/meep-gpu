"""The COMPLEX/BLOCH H->D weld: complex ``update_H`` welded into complex ``step_D``.

THE COMPLEX TWIN OF :mod:`.fused_hd_pair`, on the second largest cell the Metal
fusion board had left unbuilt on this seam: **17 corpus instances** at
``(update_H complex/Bloch -> step_D complex/Bloch)``, all seventeen
``buildable_not_built`` and not one of them carrying a standing in-seam withdraw
(``parity/meep_gpu/results/h_to_d_seam_2026-09-04/h_to_d_seam.jsonl``, joined on the
Metal arms). The predicate below therefore reaches **17 of the cell's 17**, which is
the one thing that separates this cell from the Cartesian real one, where two rows
are refused by name for a live integrated electric withdraw.

The driver runs

    ``update_H`` (driver.py:3311) -> the electric integrated-source withdraw
    (:3313-3314) -> ``step_D`` (:3315)

and exactly one statement sits between the two consults. Nothing is INJECTED here --
the electric injection is one seam later and the magnetic one is one seam earlier --
so :mod:`..deposit_repair` has nothing to say about this seam and
:mod:`..withdraw_hoist` is the module that does.

=============================================================================
THE SHAPE: SCRATCH OUTPUT AND FOREIGN-CELL RECOMPUTE, ON float2
=============================================================================

Complex ``step_D``'s curl reads ``H`` at the thread's own cell AND at its three
BACKWARD neighbours, and complex ``update_H`` writes ``H``. Welded in place, thread
``idx`` would read ``H[idx - stride]`` at a moment decided by the schedule, and it
would read ``f_w_H`` the same way -- on this side ``f_w_H_new[ii] == B[ii]`` exactly,
so an in-place write hands a racing neighbour ``B`` where it needs ``B_prev``.

Both premises are removed exactly as :mod:`.fused_hd_pair` removes them on the real
Cartesian cell and :mod:`.cylindrical_complex_fused_hd_pair` on the Dcyl one:

* **the constitutive half writes nothing in place.** ``H_new`` and ``f_w_H_new`` go
  to LAUNCH-LOCAL SCRATCH, so ``H``, ``f_w_H`` and ``B`` are ``const`` for the whole
  dispatch and no thread can observe another thread's store;
* **the foreign read is not a read of another thread's output.** It is a RECOMPUTE
  from that same unwritten state, through :func:`h_cell_function` -- the certified
  ``bloch_constitutive_step`` body for side ``H``, whole, evaluated at an arbitrary
  cell. A pure function of unwritten memory has no schedule to depend on.

The launcher then ROTATES the ``H``/``f_w_H`` bindings, which is
:class:`.offdiag_weld_common.ScratchWeldPairPlan`'s certified choreography.

WHY THE RECOMPUTE IS EXACT RATHER THAN CLOSE. Complex ``update_H`` is POINTWISE: per
component it reads ``H``, ``f_w_H`` and ``B`` at ONE cell and writes the same cell
(complex_fields.py's ``_CONSTITUTIVE_TEMPLATE``, the transcription of
stepping.py:907-923 and :2130-2143). The stepped value at any cell is therefore a
function of state this launch does not write, and evaluating it twice gives the same
bits by construction rather than by a tolerance.

THE BLOCH PHASE IS NOT IN THE RECOMPUTE AND MUST NOT BE. The certified complex curl
applies :func:`.complex_fields._phase_block` to the LOADED REGISTER after the gather,
on the wrapped lane only; this weld redirects the LOAD and leaves the phase block
standing character for character. So a wrapped neighbour is still
``c_mul(<the neighbour's stepped H>, p_axis)`` in that order, which is the field-left
orientation the array path uses (stepping.py:1909) and the one the expansion arm was
probed for.

=============================================================================
THE SIGNATURE -- 30 POINTERS PLUS ONE PACKED STRUCT, AND THE CEILING IS AN EQUALITY
=============================================================================

    3 H_out + 3 f_w_H_out          (float2 scratch, written, never read)
  + 3 H_in  + 3 f_w_H_in           (float2, pre-launch, const)
  + 3 B                            (float2, const, the constitutive source)
  + 3 D     + 3 fu_D               (float2, in place: the curl reads and writes its
                                    OWN cell only, so nothing here races)
  + 6 curl coefficients (kms/sinv per axis, float32)
  + 3 constitutive kps             (float32)                                  = 30

plus one ``constant Params&`` is **31 of the 31 bindings the platform allows**
(:data:`.device.MAX_BUFFER_BINDINGS`; buffer attribute indices run 0..30 and a 32nd
is a COMPILE ERROR). There is NO headroom, so :data:`PACKED_BINDINGS` is pinned as an
EQUALITY and the gate compiles one pointer past the shipped shape and requires the
failure.

**THE COMPLEX VOLUMES COST NOTHING EXTRA AND THAT IS THE ONLY REASON THIS FITS.** A
``complex64`` volume binds ONE ``device float2*`` (device.py's ``BINDABLE_DTYPES``),
so the twenty-one field volumes are twenty-one pointers here exactly as their real
twins are in :mod:`.fused_hd_pair`. Bound as separate re/im planes the same kernel
needs :data:`SPLIT_PLANE_BINDINGS` = 62 and cannot be built at all;
:func:`split_plane_pair_signature` is that measurement.

**THE COEFFICIENTS STAY float32 UNDER COMPLEX STORAGE**, which is the invariant
stepping.py:41-50 states and :mod:`.complex_fused_magnetic_pair` records: nine
coefficient buffers, not eighteen. Had they widened with the fields this signature
would need 39 pointers.

**WHAT MAKES 30 POSSIBLE IS A SHARED COEFFICIENT GROUP**, and dropping the sharing
does not merely cost performance, it does not compile. Both halves sit on the same
Yee sub-lattice -- ``complex_fields.SUB_STEPS['step_D']['suffix'] == ''`` and
``CONSTITUTIVE_SIDES['H']['half_integer'] is False`` -- so the curl's ``kms_x/_y/_z``
and the constitutive's ``km0/km1/km2`` ARE THE SAME THREE VOLUMES and are bound once.
Unshared the signature is 33 pointers plus ``Params`` = 34, which
:func:`refuted_unshared_kms_source` compiles and the gate requires to FAIL. Bind the
HALF-INTEGER set by mistake instead and nothing fails: you get a smooth, converged,
half-cell-wrong absorber profile, which is why the host mutation for it is a gate leg
and not a comment.

=============================================================================
THE ARITHMETIC FACTS THIS FAMILY INHERITS, AND WHAT IT ADDS
=============================================================================

Three spellings are load-bearing on every complex Metal kernel and all three were
MEASURED rather than reasoned (:mod:`.complex_fields`): the zero cross terms must be
spelled literally (folding them misses 12/128 words on the exhaustive signed-zero
table), the plane-wise ``{re*c, im*c}`` fast path is wrong (24/128), and negation is
``-x`` (``0.0f - x`` misses 36/512). All three ride in through
:func:`.templates.complex_helpers`, unchanged, and the gate arms each as a mutation
on THIS kernel rather than inheriting the other family's verdict.

THE EXPANSION ARM IS A MEASURED PLATFORM FACT, NEVER A DEFAULT. It is read from the
complex expansion probe (:func:`.complex_fields.load_expansion_probe`) and an absent
or non-discriminating probe is a REFUSAL, not a fallback -- the clause is inherited
from both halves' own predicates and restated by :func:`plan_metal_complex_fused_hd_pair`
returning ``None``.

A COMPLEX-BY-REAL DIVIDE DOES NOT OCCUR IN THIS KERNEL, AND THAT IS A MEASUREMENT
RATHER THAN AN OMISSION. The neighbouring Dcyl product had to settle the spelling of
one (numpy's ``complex64 / float32`` is its complex divide loop, whose value on every
word that reaches a prefix is the reciprocal multiply ``z * (1.0f / d)`` and NOT the
componentwise ``/``; :mod:`.cylindrical_complex_scan`). This seam has no division at
all: every PML reciprocal is precomputed on the HOST into ``sinv_a`` (pml.py) and the
kernel MULTIPLIES by it through ``c_mul_field_left``. The gate carries that as a
static count over the emitted text AND as an armed mutation -- ``c_mul_field_left(z,
si)`` respelled as a componentwise divide by ``1/si`` -- so the fact is re-measured on
this kernel instead of being cited from another one.

=============================================================================
WHAT SITS IN THE SEAM, AND WHAT THIS PRODUCT REFUSES
=============================================================================

:data:`CARRIES_DEPOSIT_REPAIR` is False and it is a FACT rather than a decision:
nothing is injected between the two consults, so there is no deposit to bracket.

:data:`HOISTS_THE_WITHDRAW` is False IN THIS ROUND. The only wiring that performs the
hoist is ``launch._install_fused_pair``'s ``withdraw_hoist.SEAM`` branch, which wraps
the built plan in a :class:`..withdraw_hoist.LeadingWithdrawPlan`; this product
declares :data:`INSTALLABLE` False, so that branch is unreachable for it and NOTHING
would perform the withdraw before its launch. The predicate therefore refuses BY NAME
every row whose electric withdraw does work. **On this cell that refuses zero rows**:
the seam record measures 0 of 17 with ``withdraw_in_seam``, unlike the Cartesian real
cell's 2 of 49. The clause is kept in full anyway -- ignorance is never an empty set,
and a row that grows an integrated source later must be refused, not silently served.

THE FOLD IS REFUSED. ``folded complex`` is its own arm on both slots and its own cell
(5 rows on this board); both halves' predicates already refuse a mirror plane
(:func:`.complex_fields._complex_grid_reasons` clause 4) and this module restates it
by name. ``beta``/BFAST, cylindrical, conductive, dispersive, a susceptibility, an
off-diagonal epsilon, real storage and an inactive absorber are all refused through
the two halves' own certified predicates, which this one conjoins without weakening.

=============================================================================
IT IS NOT INSTALLED, AND THE REASON IS ARBITRATION RATHER THAN ARITHMETIC
=============================================================================

:data:`INSTALLABLE` is False. Over the driver's ``step_B - update_H - step_D -
update_E`` slot path launches are ``4 - (installed pairs)``, and a product spanning
``update_H``/``step_D`` takes one slot from EACH neighbour. Both neighbours exist and
are RELEASED on this exact cell -- :mod:`.complex_fused_magnetic_pair` on
``step_B``/``update_H`` and :mod:`.complex_fused_electric_pair` on
``step_D``/``zero_metal_D``/``update_E`` -- so wherever both install, installing this
product is a strict LOSS (2 pairs -> 1). Driven through the shipped composer on all
17 corpus rows of the cell (2026-09-07, the gate's ``arbitration`` leg): both
neighbours install on **17 of 17**, so the four-slot path costs 2 launches today and
3 with this product. A LOSS on 17, a TIE on 0, a GAIN on 0.

What this product IS worth is stated plainly: the priced predicate gap closed under
the project's rule *close all fusion gaps regardless*, and a certified half of the only
span that could ever pay -- the four-slot ``step_B -> update_H -> step_D -> update_E``
weld, which is NOT built here. **No timing exists for this shape and none is licensed
by anything in this module.**

REGISTERED UNWIRED on ``update_H`` (``wired=False``, ``replaces=REPLACES``), so
the seam loop can ask it and ``plan_step`` cannot select it;
``launch.FUSED_PAIR_ARMS`` carries its absorb row and ``registry.FAMILY_MODULES``
imports it. The composer then refuses it on :data:`INSTALLABLE` on every row.

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
from . import offdiag_weld_common as _weld
from . import shaders, templates
from .device import MAX_BUFFER_BINDINGS, Residency, compile_source

FAMILY = "complex_fused_hd_pair"

#: The sub-step slot this arm would hold a row on: the FIRST half of the seam, in the
#: driver's own order, so a refusal is named on the slot the fusion starts at.
SLOT = "update_H"

#: The driver passes one launch of this plan performs, in driver order
#: (driver.py:3311, :3315). Declared rather than inferred from the slot name.
#:
#: NOTHING BETWEEN THEM IS CARRIED, and there is only one thing there to carry: the
#: electric withdraw (:3313-3314), which :data:`HOISTS_THE_WITHDRAW` declines.
REPLACES: Tuple[str, ...] = ("update_H", "step_D")

#: The seam name ``launch.FUSED_PAIR_SEAMS`` files this row under, spelled through
#: :mod:`..withdraw_hoist` rather than as a literal so the two cannot drift.
SEAM: str = _withdraw_hoist.SEAM

#: Nothing is injected between the two consults, so there is no deposit in this seam
#: to carry across the launch and ``deposit_repair`` is not consulted at all.
CARRIES_DEPOSIT_REPAIR = False

#: This product does not perform the seam's electric withdraw, so its predicate
#: refuses every row with a standing one BY NAME. On this cell that is 0 of 17.
HOISTS_THE_WITHDRAW = False

#: May the composer install this product? NO. See :data:`INSTALLABLE_REASON` and the
#: gate's ``arbitration`` leg, which measures the claim on the corpus rows.
INSTALLABLE = False

INSTALLABLE_REASON = (
    "arbitration, not arithmetic, and MEASURED rather than argued. Both neighbouring "
    "seams of this cell are served by RELEASED Metal products -- "
    "complex_fused_magnetic_pair on step_B/update_H and complex_fused_electric_pair "
    "on step_D/zero_metal_D/update_E -- and over the driver's step_B - update_H - "
    "step_D - update_E slot path launches are 4 - (installed pairs), so a product "
    "spanning update_H/step_D takes one slot from EACH neighbour and can only ever "
    "install ONE pair. Driven through the SHIPPED Metal composer (launch.plan_step, "
    "fuse=True) on all 17 corpus rows of the cell, 2026-09-07: BOTH neighbours "
    "install on 17 of 17, so the path costs 2 launches today and would cost 3 with "
    "this product -- a LOSS on 17 rows, a TIE on 0 and a GAIN on 0. That is the "
    "gate's `arbitration` leg, row by row, in "
    "parity/meep_gpu/results/metal_complex_fused_hd_pair_2026-09-07_cx2/gate.json. "
    "What that leaves: the board's credited instances are served by PREDICATE "
    "ADMISSION only; the product installs on zero rows; it executes nowhere outside "
    "its own gate and tests; and the weld licenses no timing claim and no dispatch "
    "claim. The one shape that could pay -- one product owning step_B through "
    "update_E in two launches -- composes from pieces that all now exist on this "
    "backend and is not built here")

#: WELDED. Empty means ``fingerprints.json`` carries
#: ``metal_complex_fused_hd_pair_device_gate`` (minted from the released artifact by
#: ``parity/meep_gpu/mint_metal_weld.py``); see
#: ``test_metal_weld_contract.test_every_metal_family_is_welded``, which partitions the
#: ``gate_metal_*.py`` fleet against that file.
WELD_OWED: str = ""

#: WHAT THE GATE MEASURED, KEPT. The pre-weld declaration is not deleted with the
#: weld: it is the one-line record of which artifact this family's release rests on,
#: and a reader who finds an empty WELD_OWED should still be able to find that
#: without opening the board.
_RETIRED_WELD_OWED = (
    "the device gate parity/meep_gpu/gate_metal_complex_fused_hd_pair.py has RUN and "
    "RELEASED on this host into "
    "parity/meep_gpu/results/metal_complex_fused_hd_pair_2026-09-07_cx2/gate.json "
    "(14 legs, 178 million uint32 words, 17 of 17 corpus rows of the cell driven and "
    "bit-identical, 9 armed defects caught, 3 measured equivalences); what a release "
    "still owes is the metal_complex_fused_hd_pair_device_gate entry in "
    "metal_kernels/fingerprints.json binding that artifact, which the 2026-09-07 "
    "unbuilt H_to_D round writes as a unified diff (wiring.patch) rather than "
    "applying, because three gate campaigns in flight pin the current bytes of every "
    "existing module. Empty this string in the same change that lands the weld entry")

#: How many bindings the shipped shape needs, and it is the platform ceiling EXACTLY.
#: There is no headroom, so the gate pins this as an equality: 31 must COMPILE and 32
#: must FAIL, both on this host.
PACKED_BINDINGS = 31

#: The shipped signature plus ONE more pointer. Over the ceiling by exactly one, which
#: is what turns :data:`PACKED_BINDINGS` from an inequality into an equality.
ONE_MORE_POINTER_BINDINGS = 32

#: The same signature with the three shared ``kms`` volumes bound TWICE, once for each
#: half -- 33 pointers + ``Params``. Over the ceiling by three, which is why the
#: sharing is load-bearing rather than tidy.
UNSHARED_KMS_BINDINGS = 34

#: The same 30 pointers with the five scalars and three phases bound SEPARATELY, the
#: way the certified 23-binding complex curl binds its own. 38; over the ceiling.
SEPARATE_SCALAR_BINDINGS = 38

#: The same kernel with every complex volume bound as SEPARATE re/im planes: 21
#: complex volumes -> 42 planes, + 9 real coefficient pointers + 5 scalars + 6 phase
#: floats. Far over the ceiling, which is the measurement behind "float2 is FORCED,
#: not preferred".
SPLIT_PLANE_BINDINGS = 62

#: The host record's total size in bytes. The three ``float2`` members come FIRST, for
#: the measured reason :mod:`.complex_fused_magnetic_pair` gives: Metal aligns
#: ``float2`` to 8 bytes, so with the scalars first the natural 44-byte host record
#: puts every phase one word early and reads back as a plausible complex number rather
#: than as garbage.
PARAMS_ITEMSIZE = 48

if PACKED_BINDINGS > MAX_BUFFER_BINDINGS:  # pragma: no cover - a design invariant
    raise RuntimeError(
        f"the complex fused H/D pair binds {PACKED_BINDINGS} buffers and Metal's "
        f"ceiling on this toolchain is {MAX_BUFFER_BINDINGS}")

#: The rotating volumes, in SIGNATURE ORDER: the three stored magnetic fields then the
#: three split-field histories. :meth:`.offdiag_weld_common.ScratchWeldPairPlan.run`
#: binds every SCRATCH first and every PRE-LAUNCH buffer second, in this order, then
#: the static arguments -- so this order IS the signature's, and a kernel that
#: interleaved a static pointer between two rotating ones would bind the wrong buffer
#: to every argument after it.
ROTATED_NAMES: Tuple[str, ...] = ("Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz")

#: The kernel entry point.
KERNEL = "complex_fused_hd_pair_step"

__all__ = [
    "CARRIES_DEPOSIT_REPAIR", "CONSTITUTIVE_LIFT_EDITS", "FAMILY", "HALO_TAPS",
    "HOISTS_THE_WITHDRAW", "H_CELL_POINTERS", "H_CELL_TAIL_ARGS", "INSTALLABLE",
    "INSTALLABLE_REASON", "KERNEL", "ONE_MORE_POINTER_BINDINGS", "OWN_LOAD_EDITS",
    "PACKED_BINDINGS", "PARAMS_ITEMSIZE", "REPLACES", "ROTATED_NAMES", "SEAM",
    "SEPARATE_SCALAR_BINDINGS", "SLOT", "SPLIT_PLANE_BINDINGS",
    "UNSHARED_KMS_BINDINGS", "WELD_OWED",
    "certified_complex_constitutive_tail", "compile_complex_fused_hd_pair",
    "complex_fused_hd_pair_source", "enumerate_sources", "h_cell_function",
    "metal_complex_fused_hd_pair_coverage", "offset_coordinates",
    "params_record_dtype", "plan_metal_complex_fused_hd_pair",
    "refuted_one_more_pointer_source", "refuted_separate_scalar_source",
    "refuted_unshared_kms_source", "register_arms",
    "shipped_signature_bindings",
    "split_plane_pair_signature", "welded_curl_tail",
]


# ---------------------------------------------------------------------------
# The lift: the certified complex H constitutive as a function of an arbitrary cell
# ---------------------------------------------------------------------------

#: Where the certified complex constitutive body ends its index decode --
#: ``templates.DECODE_IJK``'s last line, which
#: :data:`.offdiag_weld_common.DECODE_END` names for every family that lifts a body.
DECODE_END = _weld.DECODE_END

#: Where the certified complex CURL's prologue ends. It is NOT :data:`DECODE_END`: the
#: complex curl puts ``int nxi = int(nx);`` and ``int nyz = nyi * nzi;`` BELOW that
#: line, and this weld's first statement calls ``h_cell`` with ``nxi``. Cutting at the
#: decode instead would emit a call to an undeclared name.
PROLOGUE_END = "    int nyz = nyi * nzi;\n"

#: The marker that separates a certified shader's signature from its body. Both
#: complex templates end their parameter list with this exact line, so ONE anchor
#: lifts either body and a template that stopped carrying it raises rather than
#: splicing a truncated kernel.
_BODY_ANCHOR = "uint idx [[thread_position_in_grid]])\n{\n"

#: ``h_cell``'s pointer parameters, under the CERTIFIED CONSTITUTIVE BODY'S OWN NAMES
#: wherever the lifted text indexes them (``kp0``/``kp1``/``kp2``) and under the fused
#: kernel's names for the volumes the lift renames. That pairing is what lets the
#: lifted arithmetic stand character for character.
#:
#: THE COMPLEX VOLUMES ARE ``float2`` AND THE COEFFICIENTS ARE ``float``. That split
#: is the invariant stepping.py:41-50 states, and it is what keeps the fused signature
#: inside the ceiling.
H_CELL_POINTERS: Tuple[Tuple[str, str], ...] = tuple(
    [("device const float2*", name) for name in
     ("hi0", "hi1", "hi2", "wi0", "wi1", "wi2", "b0", "b1", "b2")]
    + [("device const float*", name) for name in
       ("kp0", "kp1", "kp2", "kmx", "kmy", "kmz")])

#: What one call site forwards after the three coordinates. Built from the table above
#: so the declaration and every call cannot drift apart.
H_CELL_TAIL_ARGS = ", ".join(("nxi", "nyi", "nzi")
                             + tuple(name for _kind, name in H_CELL_POINTERS))

#: Every line of certified CONSTITUTIVE text this weld does not lift verbatim, with
#: the reason. DATA, not prose, so the host suite and the gate can assert the list and
#: a line that stopped matching RAISES instead of silently leaving a stale read
#: standing.
#:
#: THERE IS NO ARITHMETIC IN THIS TABLE. Every entry is a pointer spelling or a store
#: that moves to the caller; not one right-hand side, operand order or paren is
#: touched, and the gate re-derives the lift from this table and requires equality.
CONSTITUTIVE_LIFT_EDITS: Tuple[Dict[str, str], ...] = (
    {"line": "    float kp_0 = kp0[i], km_0 = km0[i];",
     "became": "    float kp_0 = kp0[i], km_0 = kmx[i];",
     "why": "ONE POINTER RENAME, NO ARITHMETIC. Both halves sit on the same Yee "
            "sub-lattice (complex_fields.SUB_STEPS['step_D']['suffix'] == '' and "
            "CONSTITUTIVE_SIDES['H']['half_integer'] is False), so the curl's kms_x "
            "and the constitutive's km0 ARE THE SAME VOLUME and the fused signature "
            "binds it once. Unshared the kernel needs 34 bindings and does not "
            "compile; binding the half-integer set instead DOES compile and is a "
            "half-cell error in the absorber profile."},
    {"line": "    float2 prevN = wN[ii];",
     "became": "    float2 prevN = wiN[ii];",
     "why": "the split-field history is read from the PRE-LAUNCH buffer, which "
            "nothing in this launch writes. On this side f_w_H_new[ii] == B[ii] "
            "exactly, so an in-place write would hand a racing neighbour B where it "
            "needs B_prev."},
    {"line": "    float2 srcN = gN[ii];",
     "became": "    float2 srcN = bN[ii];",
     "why": "pointer spelling only: fused, `gN` in the curl half is the MAGNETIC "
            "field, so the constitutive source keeps its own name. The value is the "
            "same flux-density read the certified body makes (mu = 1 is baked into "
            "the array path too: stepping.update_H passes fields.Bx)."},
    {"line": "    wN[ii] = srcN;",
     "became": "(removed -- the caller stores own.srcN to the SCRATCH volume)",
     "why": "the weld writes a scratch volume, so the helper must not store. The "
            "value and the expression that produced it are untouched; only the "
            "destination moves, and only for the thread's OWN cell (a foreign "
            "recompute stores nothing at all)."},
    {"line": "    float2 aN = fN[ii];",
     "became": "    float2 aN = hiN[ii];",
     "why": "the accumulator is read from the PRE-LAUNCH H, which is const for the "
            "whole dispatch. This is what makes the foreign recompute a pure function "
            "of unwritten memory."},
    {"line": "    fN[ii] = aN;",
     "became": "(removed -- the value is RETURNED in the struct)",
     "why": "same as the split-field store: the destination moves to the caller, "
            "which addresses the scratch for its own cell and consumes the register "
            "for the curl half. The two accumulations, their order and their parens "
            "are untouched."},
    {"line": "    int k   = ii % nzi; ... int i   = plane / nyi;",
     "became": "(removed -- i, j and k are parameters; ii is composed from them)",
     "why": "THE WHOLE POINT OF THE LIFT. The certified body becomes a function "
            "evaluated at an ARBITRARY cell rather than at the thread's own, so the "
            "cell arrives as three coordinates and the flat index is COMPOSED by the "
            "same layout the decode inverts. The n_elem guard leaves with the thread "
            "index: every caller is either the kernel's own guarded thread or a tap "
            "the certified ghost rule already proved valid."},
)


def certified_complex_constitutive_tail(
        expansion: str, contract: str = shaders.CONTRACT_OFF) -> str:
    """The complex ``update_H`` body BELOW the decode prologue, stores removed.

    Not a transcription: this is
    :func:`.complex_fields.bloch_constitutive_source`'s own output for side ``H`` with
    the preamble, the decode prologue and the closing brace cut, and with exactly the
    substitutions :data:`CONSTITUTIVE_LIFT_EDITS` names. Each is applied through
    :func:`.offdiag_weld_common.needle`, so a spelling that drifted raises here rather
    than emitting a kernel that reads the wrong volume.
    """
    source = complex_fields.bloch_constitutive_source("H", expansion, contract)
    edits: List[Tuple[str, str]] = [
        ("    float kp_0 = kp0[i], km_0 = km0[i];\n",
         "    float kp_0 = kp0[i], km_0 = kmx[i];\n"),
        ("    float kp_1 = kp1[j], km_1 = km1[j];\n",
         "    float kp_1 = kp1[j], km_1 = kmy[j];\n"),
        ("    float kp_2 = kp2[k], km_2 = km2[k];\n",
         "    float kp_2 = kp2[k], km_2 = kmz[k];\n"),
    ]
    for target in range(3):
        edits.extend((
            (f"    float2 prev{target} = w{target}[ii];\n",
             f"    float2 prev{target} = wi{target}[ii];\n"),
            (f"    float2 src{target} = g{target}[ii];\n",
             f"    float2 src{target} = b{target}[ii];\n"),
            (f"    w{target}[ii] = src{target};\n", ""),
            (f"    float2 a{target} = f{target}[ii];\n",
             f"    float2 a{target} = hi{target}[ii];\n"),
            (f"    f{target}[ii] = a{target};\n", ""),
        ))
    tail = _weld.lift_curl_tail(source, store_edits=tuple(edits),
                               decode_end=DECODE_END)
    # THE CHECK THAT MATTERS. In the fused signature `f*`, `w*` and `g*` are the
    # DISPLACEMENT, its split-field auxiliary and the magnetic field, and `km*` is the
    # curl's own coefficient group under a different name; one missed rename is a
    # smooth, plausible, entirely wrong answer rather than a compile error, because
    # every one of those names exists in the enclosing kernel.
    for stem in ("f", "w", "g", "e", "km"):
        for target in range(3):
            if f"{stem}{target}[" in tail:
                raise AssertionError(
                    f"the lifted complex update_H body still indexes {stem}{target}, "
                    f"which in the fused kernel is not the volume the certified body "
                    f"meant")
    return tail


def h_cell_function(expansion: str, contract: str = shaders.CONTRACT_OFF) -> str:
    """Complex ``update_H`` at an arbitrary cell, as one inline function.

    ONE code path serves the thread's own cell and every foreign recompute, so the two
    cannot disagree -- the whole identity of this product rests on that sentence.
    Emitted through :func:`.offdiag_weld_common.step_cell_function`, the same helper
    the D->E scratch welds and the Dcyl H->D product use, so the index composition
    ``ii = i*nyz + j*nzi + k`` is written once for the whole package.
    """
    return _weld.step_cell_function(
        "h_cell", certified_complex_constitutive_tail(expansion, contract),
        value_type="float2", pointer_parameters=H_CELL_POINTERS,
        scalar_parameters=(), returns=("a0", "a1", "a2", "src0", "src1", "src2"))


# ---------------------------------------------------------------------------
# The lift: the certified complex step_D curl with every H read redirected
# ---------------------------------------------------------------------------

#: The curl half's three OWN-cell magnetic loads and the register each becomes. The
#: thread has already computed its own ``H`` into ``own``, and that register is the
#: post-``update_H`` value the array path would have loaded.
OWN_LOAD_EDITS: Tuple[Tuple[str, int], ...] = (("a", 0), ("b", 1), ("c", 2))

#: The curl half's six SHIFTED magnetic loads, as ``(register, component)``. WHICH
#: CELL each one reads is NOT written here: it is parsed out of the certified body's
#: own ``int ox = si * nyz + j * nzi + k;`` lines by :func:`offset_coordinates`, so the
#: recompute lands on exactly the cell the emitter's index composed and a change to
#: how that index is spelled RAISES instead of quietly redirecting a tap.
#:
#: THE BRANCH, THE SHIFT ARITHMETIC AND THE PHASE ARE NOT TOUCHED. Only the leaf load
#: moves; the validity flag (``vx``/``vy``/``vz``) is PARSED OUT of the certified line
#: rather than retyped, so a metallic ghost that read an exact complex zero still
#: reads one and the periodic wrap is still the emitter's own integer expression; and
#: the Bloch rotation is applied to the register AFTERWARDS by
#: :func:`.complex_fields._phase_block`, which this weld does not see at all. The
#: ternary is also what keeps an out-of-range recompute from being evaluated, which is
#: the same C guarantee the certified load already relies on for ``g0[oy]`` with a
#: negative ``oy``.
HALO_TAPS: Tuple[Tuple[str, int], ...] = (
    ("a_y", 0), ("a_z", 0), ("b_x", 1), ("b_z", 1), ("c_x", 2), ("c_y", 2))

#: The exact complex ghost the certified curl serves past a metallic wall. Spelled
#: through :data:`.templates.COMPLEX_ZERO` rather than as a literal so this family and
#: the emitter cannot drift.
_GHOST = templates.COMPLEX_ZERO


def offset_coordinates(tail: str) -> Dict[str, Tuple[str, str, str]]:
    """``{offset name: (i, j, k) expressions}``, read off the certified index lines.

    The emitter writes one line per shifted axis, each of the shape
    ``int ox = si * nyz + j * nzi + k;`` -- the SAME layout
    :func:`.offdiag_weld_common.step_cell_function` composes ``ii`` from. Parsing it
    rather than tabulating it is what makes "the index expressions are the emitter's"
    a property of the construction.
    """
    out: Dict[str, Tuple[str, str, str]] = {}
    for line in tail.splitlines():
        stripped = line.strip()
        if not stripped.startswith("int o") or " = " not in stripped:
            continue
        name, expression = stripped[len("int "):].rstrip(";").split(" = ", 1)
        parts = [piece.strip() for piece in expression.split(" + ")]
        if len(parts) != 3 or not parts[0].endswith(" * nyz") \
                or not parts[1].endswith(" * nzi"):
            raise AssertionError(
                f"the certified complex curl composes {name} as {expression!r}, which "
                f"is not the `a * nyz + b * nzi + c` layout this weld recomposes from "
                f"coordinates")
        out[name] = (parts[0][: -len(" * nyz")], parts[1][: -len(" * nzi")], parts[2])
    if set(out) != {"ox", "oy", "oz"}:
        raise AssertionError(
            f"the certified complex curl composes {sorted(out)} rather than the three "
            f"shifted offsets this weld redirects")
    return out


def welded_curl_tail(codes: Sequence[int], phased: Sequence[int], expansion: str,
                     contract: str = shaders.CONTRACT_OFF) -> Tuple[str, str]:
    """``(prologue, body)`` of the complex ``step_D`` curl, every H read redirected.

    ``backward`` is True and never a parameter -- this pair is the D seam, and
    ``step_B``'s forward strides and unconjugated phase are a different product.

    The PROLOGUE is everything through ``int nyz = nyi * nzi;`` (the guard, the decode
    and the two integer widths ``h_cell`` is called with); the BODY is everything
    below it, with the three own-cell loads taken from the register and the six
    shifted loads recomputed.
    """
    source = complex_fields.bloch_curl_source(codes, True, phased, expansion, contract)
    if _BODY_ANCHOR not in source:
        raise AssertionError(
            "the certified complex curl source no longer carries the body anchor; the "
            "fused kernel would splice a truncated body")
    body = source.split(_BODY_ANCHOR, 1)[1]
    if not body.endswith("}\n"):
        raise AssertionError("the certified complex curl source does not end with '}'")
    body = body[: -len("}\n")]
    if PROLOGUE_END not in body:
        raise AssertionError(
            "the certified complex curl no longer declares `int nyz = nyi * nzi;`; "
            "this weld cuts its prologue there because its first statement calls "
            "h_cell with nxi/nyi/nzi")
    prologue, tail = body.split(PROLOGUE_END, 1)
    prologue = prologue + PROLOGUE_END
    offsets = offset_coordinates(tail)
    for var, target in OWN_LOAD_EDITS:
        tail = _weld.needle(tail, f"    float2 {var}   = g{target}[ii];\n",
                            f"    float2 {var}   = own.a{target};\n")
    for var, target in HALO_TAPS:
        prefix = f"    float2 {var} = "
        matches = [line + "\n" for line in tail.splitlines()
                   if line.startswith(prefix)]
        if len(matches) != 1:
            raise AssertionError(
                f"the certified complex curl declares {var} {len(matches)} times; this "
                f"weld redirects exactly one magnetic load per shifted tap")
        old = matches[0]
        # THE GUARD AND THE OFFSET ARE BOTH PARSED FROM THE CERTIFIED LINE, never
        # retyped: everything left of " ? " is the emitter's own validity expression,
        # and the offset name inside the brackets is what selects the coordinates.
        head, rest = old.split(" ? ", 1)
        expected = f"g{target}["
        if not rest.startswith(expected) or not rest.endswith(f" : {_GHOST};\n"):
            raise AssertionError(
                f"the certified complex curl no longer reads {var} as "
                f"`{expected}...]` served an exact {_GHOST} past the wall; the ghost "
                f"rule this weld preserves has changed")
        offset = rest[len(expected):].split("]", 1)[0]
        coordinates = offsets[offset]
        call = f"h_cell({', '.join(coordinates)}, {H_CELL_TAIL_ARGS}).a{target}"
        tail = _weld.needle(tail, old, f"{head} ? {call} : {_GHOST};\n")
    for target in range(3):
        if f"g{target}[" in tail:
            raise AssertionError(
                f"the welded complex curl half still reads g{target}; in this "
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

// THE THREE float2 MEMBERS COME FIRST. Metal aligns float2 to 8 bytes, so with the
// scalars first this struct needs internal padding and the natural host record puts
// every phase one word early — measured on this toolchain, and it reads as a
// plausible complex number rather than as garbage. Phases first, no internal padding.
struct Params {
    float2 px; float2 py; float2 pz;
    uint nx; uint ny; uint nz; uint n_elem; float dtdx;
};

__H_CELL__

kernel void complex_fused_hd_pair_step(
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
    // THE EIGHT PACKED ARGUMENTS ARE UNPACKED INTO THE CERTIFIED BODIES' OWN NAMES,
    // once, before any lifted text runs. Everything below this line is then
    // character-for-character what `complex_fields.bloch_curl_source` and
    // `complex_fields.bloch_constitutive_source` emit, with the declared lift edits
    // and the redirected magnetic reads.
    uint nx = prm.nx, ny = prm.ny, nz = prm.nz, n_elem = prm.n_elem;
    float dtdx = prm.dtdx;
    float2 px = prm.px, py = prm.py, pz = prm.pz;
__PROLOGUE__
    // --- THE WELD: complex update_H, computed into registers and stored to SCRATCH -
    // Nothing written here is read by this launch. The curl half below takes its OWN
    // cell's magnetic field from these registers and recomputes every foreign tap
    // from PRE-LAUNCH state through the same `h_cell`, so no thread observes another
    // thread's store; the launcher rotates H/f_w_H afterwards.
    h_cell_result own = h_cell(i, j, k, __H_CELL_ARGS__);
    ho0[ii] = own.a0; ho1[ii] = own.a1; ho2[ii] = own.a2;
    wo0[ii] = own.src0; wo1[ii] = own.src1; wo2[ii] = own.src2;

    // --- complex step_D (stepping.step_D / _apply_pml_update:1905) ----------------
    // D and fu_D update IN PLACE and that is safe by construction: the curl reads and
    // writes them at the THREAD'S OWN CELL only (`f0[ii]`, `u0[ii]`), so no thread
    // reads a displacement another thread wrote.
__CURL__
}
"""


def complex_fused_hd_pair_source(codes: Sequence[int], phased: Sequence[int],
                                 expansion: str,
                                 contract: str = shaders.CONTRACT_OFF) -> str:
    """One specialised fused source: (boundaries, phases, arm, contraction).

    Every constant Triton bakes into a ``tl.constexpr`` is baked into the string here,
    for the reason :mod:`.shaders` gives: ``torch.mps.compile_shader`` takes a source
    and nothing else, so specialisation by substitution is what keeps the emitted
    arithmetic identical to the bodies this lifts.

    THE WALL CLEAR IS NOT CARRIED and must not be. ``zero_metal_B`` runs one seam
    EARLIER (driver.py:3306, before the ``update_H`` consult) and ``zero_metal_D`` one
    seam LATER (:3324, after ``step_D``); neither is inside this seam, so a mask here
    would be a pass the driver runs again. The OWNERSHIP mask -- a different pass --
    belongs to ``step_D`` itself and rides in with the certified curl body.

    A phased METALLIC axis is refused by the certified emitter itself
    (``complex_fields.bloch_curl_source`` raises, mirroring ``stepping._bloch_phases``
    S:2346-2360). It is not re-checked here: one raise, in the emitter that owns the
    rule.
    """
    codes = tuple(int(code) for code in codes)
    phased = tuple(int(flag) for flag in phased)
    if len(codes) != 3:
        raise ValueError(f"codes must be a per-axis triple, got {codes!r}")
    if len(phased) != 3:
        raise ValueError(f"phased must be a per-axis triple, got {phased!r}")
    prologue, curl = welded_curl_tail(codes, phased, expansion, contract)
    return templates.substitute(_TEMPLATE, {
        "__CONTRACT__": shaders.contraction_pragma(contract),
        "__HELPERS__": templates.complex_helpers(expansion),
        "__H_CELL__": h_cell_function(expansion, contract),
        "__PROLOGUE__": prologue,
        "__H_CELL_ARGS__": H_CELL_TAIL_ARGS,
        "__CURL__": curl,
    })


def compile_complex_fused_hd_pair(codes: Sequence[int], phased: Sequence[int],
                                  expansion: str,
                                  contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised entry point for one (boundaries, phases, arm, mode)."""
    return getattr(
        compile_source(complex_fused_hd_pair_source(codes, phased, expansion,
                                                    contract)),
        KERNEL)


def _phase_triples(codes: Sequence[int]) -> Tuple[Tuple[int, int, int], ...]:
    """Every phase triple legal against one boundary triple. A metallic axis cannot."""
    legal: List[Tuple[int, int, int]] = []
    for px in (0, 1):
        for py in (0, 1):
            for pz in (0, 1):
                triple = (px, py, pz)
                if any(triple[axis] and codes[axis] == templates.METALLIC
                       for axis in range(3)):
                    continue
                legal.append(triple)
    return tuple(legal)


def enumerate_sources(expansion: str,
                      contract: str = shaders.CONTRACT_OFF) -> Dict[str, str]:
    """Every specialisation a shipped plan can emit, keyed by a stable label.

    The phase triple is enumerated only over the axes a given boundary triple can
    legally phase -- a metallic axis cannot carry one -- so the count is the REACHABLE
    set rather than the Cartesian product, and a label that cannot be built is not
    fingerprinted as if it could.
    """
    out: Dict[str, str] = {}
    for cx in (templates.PERIODIC, templates.METALLIC):
        for cy in (templates.PERIODIC, templates.METALLIC):
            for cz in (templates.PERIODIC, templates.METALLIC):
                codes = (cx, cy, cz)
                for phased in _phase_triples(codes):
                    label = (f"{KERNEL}/{cx}{cy}{cz}/"
                             f"ph{phased[0]}{phased[1]}{phased[2]}/{expansion}")
                    out[label] = complex_fused_hd_pair_source(codes, phased, expansion,
                                                              contract)
    return out


# ---------------------------------------------------------------------------
# The refuted signatures — measurements, not arguments
# ---------------------------------------------------------------------------

_COMPLEX_VOLUMES: Tuple[str, ...] = (
    "ho0", "ho1", "ho2", "wo0", "wo1", "wo2",
    "hi0", "hi1", "hi2", "wi0", "wi1", "wi2",
    "b0", "b1", "b2", "f0", "f1", "f2", "u0", "u1", "u2")
_REAL_VECTORS: Tuple[str, ...] = (
    "kmx", "sinvx", "kmy", "sinvy", "kmz", "sinvz", "kp0", "kp1", "kp2")


def _touch_kernel(name: str, complex_volumes: Sequence[str],
                  real_vectors: Sequence[str], contract: str,
                  packed: bool = True) -> Tuple[int, str]:
    """A signature with a body that touches every buffer, and nothing else.

    What is being measured is the SIGNATURE; a body the compiler could drop would let
    dead-code elimination decide the answer, which is why every buffer is read and the
    first is stored.

    ``packed`` binds the five scalars and three phases as ONE ``constant Params&``,
    exactly as the shipped kernel does, so the returned binding count is the number
    this family argues from rather than one binding-model away from it.
    """
    lines: List[str] = []
    slot = 0
    for label in complex_volumes:
        lines.append(f"    device float2*      {label:<8}[[buffer({slot})]],")
        slot += 1
    for label in real_vectors:
        lines.append(f"    device const float* {label:<8}[[buffer({slot})]],")
        slot += 1
    if packed:
        lines.append(f"    constant Params&    prm     [[buffer({slot})]],")
        slot += 1
        unpack = ["    uint n_elem = prm.n_elem;", "    float dtdx = prm.dtdx;",
                  "    float2 px = prm.px;"]
        scale = "px.x + dtdx"
    else:
        for kind, label in (("constant uint&", "nx"), ("constant uint&", "ny"),
                            ("constant uint&", "nz"), ("constant uint&", "n_elem"),
                            ("constant float&", "dtdx")):
            lines.append(f"    {kind:<19} {label:<8}[[buffer({slot})]],")
            slot += 1
        for label in ("px", "py", "pz"):
            lines.append(f"    constant float2&    {label:<8}[[buffer({slot})]],")
            slot += 1
        unpack = []
        scale = "px.x + dtdx"
    touch = " + ".join(f"{label}[0]" for label in real_vectors)
    return slot, "\n".join((
        "#include <metal_stdlib>",
        "using namespace metal;",
        "",
        shaders.contraction_pragma(contract),
        "",
        "struct Params {",
        "    float2 px; float2 py; float2 pz;",
        "    uint nx; uint ny; uint nz; uint n_elem; float dtdx;",
        "};",
        "",
        f"kernel void {name}(",
        *lines,
        "    uint idx [[thread_position_in_grid]])",
        "{",
        *unpack,
        "    if (idx >= n_elem) { return; }",
        f"    float touch = {touch};",
        f"    float2 acc = float2(touch, {scale});",
        *[f"    acc = acc + {label}[idx];" for label in complex_volumes],
        f"    {complex_volumes[0]}[idx] = acc;",
        "}",
        "",
    ))


def shipped_signature_bindings(expansion: str) -> int:
    """How many bindings the SHIPPED kernel declares, counted off its own source.

    Read from the emitted text rather than from :data:`PACKED_BINDINGS`, so the
    constant is a claim the source can falsify.
    """
    source = complex_fused_hd_pair_source((0, 0, 0), (0, 0, 0), expansion)
    signature = source.split(f"kernel void {KERNEL}(", 1)[1]
    signature = signature.split("uint idx [[thread_position_in_grid]])", 1)[0]
    return signature.count("[[buffer(")


def refuted_one_more_pointer_source(contract: str = shaders.CONTRACT_OFF) -> str:
    """The shipped signature plus ONE pointer -- 32 bindings, over the ceiling by one.

    A family with no headroom has to pin the ceiling as an EQUALITY, so the gate
    compiles this and requires the failure beside the shipped 31 that must succeed.
    """
    slots, source = _touch_kernel("refuted_one_more_pointer", _COMPLEX_VOLUMES,
                                  _REAL_VECTORS + ("one_too_many",), contract)
    assert slots == ONE_MORE_POINTER_BINDINGS, (slots, ONE_MORE_POINTER_BINDINGS)
    return source


def refuted_unshared_kms_source(contract: str = shaders.CONTRACT_OFF) -> str:
    """The 34-binding signature that binds the three ``kms`` volumes TWICE.

    The measurement that makes the shared coefficient group a NECESSITY rather than a
    tidiness: with the constitutive half given its own ``km0``/``km1``/``km2`` the
    signature is 33 pointers plus one packed ``Params`` and does not compile at all.
    """
    slots, source = _touch_kernel("refuted_unshared_kms", _COMPLEX_VOLUMES,
                                  _REAL_VECTORS + ("km0", "km1", "km2"), contract)
    assert slots == UNSHARED_KMS_BINDINGS, (slots, UNSHARED_KMS_BINDINGS)
    return source


def refuted_separate_scalar_source(contract: str = shaders.CONTRACT_OFF) -> str:
    """The same 30 pointers with the eight non-pointer arguments bound SEPARATELY.

    38 bindings, and the gate REQUIRES the compile failure. This is not a strawman: it
    is the way :func:`.complex_fields.bloch_curl_source` binds its own arguments at its
    23-binding size, so "packed" is a measurement on this signature rather than a house
    style.
    """
    slots, source = _touch_kernel("refuted_separate_scalar", _COMPLEX_VOLUMES,
                                  _REAL_VECTORS, contract, packed=False)
    assert slots == SEPARATE_SCALAR_BINDINGS, (slots, SEPARATE_SCALAR_BINDINGS)
    return source


def split_plane_pair_signature() -> str:
    """The re/im-SPLIT signature, which must FAIL to compile. 62 bindings.

    Evidence that ``float2`` volumes are FORCED by the 31-binding ceiling rather than
    chosen, exactly as :func:`.complex_fields.split_plane_curl_signature` is for the
    certified curl. The gate compiles this and requires the compile error.
    """
    planes = [f"    device float* v{index} [[buffer({index})]]"
              for index in range(2 * len(_COMPLEX_VOLUMES))]
    start = len(planes)
    reals = [f"    device const float* r{index} [[buffer({index})]]"
             for index in range(start, start + len(_REAL_VECTORS))]
    start += len(_REAL_VECTORS)
    scalars = [f"    constant uint& s{index} [[buffer({index})]]"
               for index in range(start, start + 4)]
    start += 4
    scalars.append(f"    constant float& dtdx [[buffer({start})]]")
    start += 1
    phases = [f"    constant float& p{index} [[buffer({index})]]"
              for index in range(start, start + 6)]
    arguments = ",\n".join(planes + reals + scalars + phases)
    assert start + 6 == SPLIT_PLANE_BINDINGS, (start + 6, SPLIT_PLANE_BINDINGS)
    return ("#include <metal_stdlib>\nusing namespace metal;\n"
            "kernel void split_plane_pair(\n" + arguments +
            ",\n    uint idx [[thread_position_in_grid]])\n"
            "{\n    v0[idx] = v1[idx];\n}\n")


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def metal_complex_fused_hd_pair_coverage(fields: Any, pml: Any, sources: Any = None,
                                         residency: Any = None,
                                         probe: Any = None) -> Coverage:
    """May ONE dispatch span complex ``update_H`` -> the withdraw -> complex ``step_D``?

    A conjunction of the two halves' OWN certified predicates plus the seam clauses.
    Nothing is weakened: a configuration either half refuses is refused here with that
    half's reasons, prefixed so a reader can tell which side said it. That construction
    is what makes the expansion-probe clause, the complex-storage clause, the beta and
    BFAST clauses, the off-diagonal clause, the polarization clause and the fold clause
    INHERITED rather than restated.

    THE ORDER IS THE DRIVER'S. The constitutive half is asked FIRST because it runs
    first (driver.py:3311), so the first refusal a reader sees names the half the
    driver would have reached first.
    """
    reasons: List[str] = []

    magnetic = complex_fields.complex_constitutive_coverage(fields, pml, "H",
                                                            residency, probe)
    if not magnetic.covered:
        reasons.extend(f"constitutive half: {reason}" for reason in magnetic.reasons)
    curl = complex_fields.complex_pml_curl_coverage(fields, pml, "step_D", residency,
                                                    probe)
    if not curl.covered:
        reasons.extend(f"curl half: {reason}" for reason in curl.reasons)

    # THE SEAM'S ONE PASS. Nothing is INJECTED between the two consults, so
    # `deposit_repair` is not consulted at all and a source of either polarity is not
    # this seam's business -- the magnetic injection is one seam earlier
    # (driver.py:3283-3284) and the electric one is one seam later (:3317-3322). What
    # IS between them is the electric integrated-source withdraw (:3313-3314), and
    # `withdraw_hoist` owns it. IGNORANCE IS NEVER AN EMPTY SET: `Fields` does not hold
    # the source list, so a predicate that inferred "no withdraw stands" from not being
    # told would be the over-covering refusal this clause exists to prevent.
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

    # THE FOLD, RESTATED BY NAME. Both halves already refuse a mirror plane
    # (complex_fields._complex_grid_reasons clause 4), and on THIS seam the reason is
    # not the one the D->E and B->H pairs give: neither fill runs between these two
    # consults (fill_B closes one seam earlier at driver.py:3306-3309 and fill_D opens
    # one seam later at :3324-3327), so the H->D seam is fill-FREE on a folded grid.
    # What refuses a fold here is that this product implements the `complex/Bloch` arm
    # and a folded complex run selects the `folded complex` arm on both slots -- a
    # 5-row cell of its own, and a second product.
    for axis in range(3):
        mirrored = getattr(grid, "is_mirrored", None)
        if callable(mirrored) and bool(mirrored(axis)):
            reasons.append(
                f"axis {axis} is folded: this product implements the `complex/Bloch` "
                f"update_H and step_D arms, and a folded complex run selects the "
                f"`folded complex` arm on both slots; that cell needs its own product "
                f"and its own ghost-map measurement")

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

class MetalComplexFusedHdPairPlan(_weld.ScratchWeldPairPlan):
    """The shared scratch-weld plan, carrying the three facts that specialise it.

    THE ROTATION IS NOT REIMPLEMENTED. ``run`` and ``_resolve`` are the base's, so this
    family's post-launch buffer rotation is the same code the four D->E welds and the
    real Cartesian H->D product execute. What the subclass adds is the specialisation a
    harness must not re-derive: the per-axis phase flags the source was baked for, the
    rounded and conjugated phase values in the packed record, and the PROBE-MEASURED
    expansion arm. A base with ``__slots__`` refuses to carry them, and re-deriving
    them at a call site is how a mutation harness ends up compiling a different
    specialisation from the one it launches.
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


def params_record_dtype() -> Any:
    """The host record's dtype -- THE THREE ``float2`` MEMBERS FIRST, itemsize 48.

    ONE HOME FOR THE LAYOUT, because getting it wrong is a silent wrong answer rather
    than a crash: see :mod:`.complex_fused_magnetic_pair` for the measurement. The
    offsets are stated EXPLICITLY even though the phases-first order makes them the
    natural ones, so that the record cannot drift into agreeing with Metal only by
    accident; ``itemsize`` is stated because the natural record is 44 bytes and
    Metal's struct is 48.
    """
    import numpy as np  # noqa: PLC0415

    return np.dtype({
        "names": ["px", "py", "pz", "nx", "ny", "nz", "n_elem", "dtdx"],
        "formats": [("<f4", 2), ("<f4", 2), ("<f4", 2),
                    "<u4", "<u4", "<u4", "<u4", "<f4"],
        "offsets": [0, 8, 16, 24, 28, 32, 36, 40],
        "itemsize": PARAMS_ITEMSIZE,
    })


def _params_tensor(shape: Sequence[int], dtdx: float,
                   phase_values: Sequence[Tuple[float, float]], device: str) -> Any:
    """The five scalars and three phases as one 48-byte device record.

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
    words = np.frombuffer(record.tobytes(), dtype=np.int32).copy()
    return torch.from_numpy(words).to(torch.device(device))


def plan_metal_complex_fused_hd_pair(
        fields: Any, pml: Any, sources: Any = None,
        residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        functions: Optional[Mapping[str, Any]] = None,
        probe: Any = None,
        ) -> Optional["_weld.ScratchWeldPairPlan"]:
    """Build the complex fused H/D plan, or ``None`` when the seam is refused.

    ``None`` is the only refusal, for the reason :func:`.launch.plan_pml_curl` gives: a
    configuration this kernel does not carry must fall back to the array path, never
    raise into a caller that would otherwise have stepped correctly.

    ``functions`` is the mutation seam. The gate compiles a deliberately broken copy of
    the shipped source and hands it here; dropping the argument is not a silent
    slowdown, it is a silent DISARMING -- every mutation leg would then launch the
    shipped kernel and report the defect as uncaught.
    """
    if not metal_complex_fused_hd_pair_coverage(fields, pml, sources, residency,
                                                probe).covered:
        return None
    expansion = complex_fields.expansion_from_probe(
        probe if probe is not None else complex_fields.load_expansion_probe())
    if expansion is None:  # pragma: no cover - the predicate already refused
        return None
    from ..stepping import _boundary_kinds  # noqa: PLC0415

    grid = fields.grid
    kinds = _boundary_kinds(grid, pml)
    codes = tuple(1 if kind == "metallic" else 0 for kind in kinds)
    # backward=True: this is the D seam. The phase table is resolved through the
    # certified helper so the conjugation rule lives in one place.
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

    # ONE SUB-LATTICE, TWO HALVES. `complex_fields.SUB_STEPS['step_D']['suffix']` is ''
    # and `CONSTITUTIVE_SIDES['H']['half_integer']` is False, so the curl's kms and the
    # constitutive's kms are THE SAME THREE VOLUMES and are bound once. Both suffixes
    # are READ FROM THE SHIPPED TABLES rather than written as literals, and their
    # equality is ASSERTED rather than assumed: it is the whole premise of the
    # 30-pointer signature, and if either table moved, sharing the group would bind one
    # half's coefficients to the other half's lattice -- a converged, smooth,
    # half-cell-wrong absorber profile rather than a failure.
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
            selected[mode] = compile_complex_fused_hd_pair(codes, flags, expansion,
                                                           mode)

    dtdx = grid.dt / grid.dx
    static = (flux + displacement + auxiliary
              + curl_coefficients + constitutive_coefficients
              + [_params_tensor(grid.shape, dtdx, values, residency.device)])
    assert (2 * len(ROTATED_NAMES) + len(static)) == PACKED_BINDINGS, (
        2 * len(ROTATED_NAMES) + len(static), PACKED_BINDINGS)
    assert PACKED_BINDINGS == MAX_BUFFER_BINDINGS, (PACKED_BINDINGS,
                                                    MAX_BUFFER_BINDINGS)
    # THE ARGUMENT ORDER IS `ScratchWeldPairPlan.__init__`'s, positionally, so this
    # family and the shared plan cannot drift on what each slot means.
    return MetalComplexFusedHdPairPlan(
        FAMILY, residency, fields, dict(twins), dict(selected), ROTATED_NAMES,
        tuple(static), tuple(volumes), tuple(grid.shape), tuple(codes),
        (0, 0, 0), (), REPLACES,
        phased=flags, phase_values=values, expansion=expansion)


# ---------------------------------------------------------------------------
# Registration -- wired=False, refused by the composer on INSTALLABLE
# ---------------------------------------------------------------------------

def _arm_coverage(context: Any, slot: str) -> Coverage:
    if slot != SLOT:
        return Coverage(False, (f"complex fused H/D pair cannot fill {slot}",))
    return metal_complex_fused_hd_pair_coverage(
        context.fields, context.pml, context.sources, context.residency,
        context.extra.get("complex_probe"))


def _arm_plan(context: Any, slot: str) -> Optional[MetalComplexFusedHdPairPlan]:
    if slot != SLOT:
        return None
    return plan_metal_complex_fused_hd_pair(
        context.fields, context.pml, context.sources, context.residency,
        context.contract_variants, probe=context.extra.get("complex_probe"))


def register_arms() -> Tuple[Any, ...]:
    """One row on ``update_H``, ``wired=False``.

    REGISTERED SO IT IS ENUMERABLE, NOT SO IT IS SELECTABLE. ``arms.arms_for`` skips
    an unwired row so ``plan_step`` cannot select it, while ``arms.registered`` still
    returns it -- which is what lets the composition sweep measure disjointness
    against this predicate instead of assuming it, and what lets
    ``launch._install_fused_pairs`` reach the row through its ``FUSED_PAIR_ARMS``
    entry. The seam loop then refuses it on :data:`INSTALLABLE`, and the measurement
    behind that flag is in :data:`INSTALLABLE_REASON`: both neighbouring released
    pairs install on all 17 rows of this cell, so installing this product instead is
    a loss on every one of them.
    """
    from . import arms  # noqa: PLC0415
    return (arms.register(FAMILY, SLOT, "complex fused H/D pair",
                          _arm_coverage, _arm_plan,
                          prefix="complex fused H/D pair: ",
                          noun="complex fused stored-H/PML D-curl pair",
                          wired=False, replaces=REPLACES),)


ARMS = register_arms()
