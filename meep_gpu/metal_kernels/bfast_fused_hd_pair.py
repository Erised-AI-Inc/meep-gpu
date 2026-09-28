"""The BFAST H->D weld: BFAST-run ``update_H`` welded into the BFAST ``step_D``.

THE THIRD ARM PAIR ON THE FOURTH SEAM, and the first one whose CURL half is not a
member of the certified PML curl family at all. The driver runs

    ``update_H`` (driver.py:3311) -> the electric integrated-source withdraw
    (:3313-3314) -> ``step_D`` (:3315)

and this product performs both consults in ONE dispatch. The shape is
:mod:`.fused_hd_pair`'s exactly -- the certified constitutive recompute into
launch-local write-only scratch, the curl stepping ``D``/``fu_D`` in place, and the
launcher rotating the ``H``/``f_w_H`` bindings afterwards -- with two changes and no
others:

* the curl half is :func:`.bfast_curl.bfast_curl_source`'s body rather than
  :func:`.shaders.curl_source`'s, which brings THREE state volumes
  (``f_bfast_Dx/y/z``) and SIX host-rounded ``k1``/``k2`` scalars;
* the nine read-only per-axis coefficient VECTORS ride ONE buffer through the
  shipped :mod:`.coefficient_pack`, because the unpacked signature does not fit.

=============================================================================
THE SIGNATURE, AND WHY THE PACK IS LOAD-BEARING RATHER THAN TIDY
=============================================================================

Unpacked this weld needs::

    3 H_out + 3 f_w_H_out          (scratch, written, never read)
  + 3 H_in  + 3 f_w_H_in           (pre-launch, const)
  + 3 B                            (const, the constitutive source)
  + 3 D     + 3 fu_D               (in place, own cell only)
  + 3 f_bfast_D                    (in place, own cell only -- the Tustin state)
  + 6 curl coefficients (kms/sinv per axis)
  + 3 constitutive kps                                                        = 33

plus one ``constant Params&`` is **34 of the 31 bindings the platform allows**
(:data:`.device.MAX_BUFFER_BINDINGS`) -- over the ceiling by three. This is the same
verdict :mod:`.conductive_fused_electric_pair` met on the D->E seam and closed the
same way: the nine read-only per-axis vectors (the curl's ``kmx``/``sinvx``/... and
the constitutive's ``kp0``/``kp1``/``kp2``) go into ONE packed buffer with nine
element offsets carried in ``Params``, saving eight pointers. 25 pointers plus
``Params`` is :data:`PACKED_BINDINGS` = 26, five under the ceiling.

:func:`refuted_unpacked_pointer_source` is the 34-binding shape, compiled by the gate
and REQUIRED TO FAIL, so "the pack is what makes this weld exist" is a measurement
rather than an argument.

THE CONSTITUTIVE ``km`` GROUP IS SHARED WITH THE CURL'S, exactly as in
:mod:`.fused_hd_pair`: ``launch.SUB_STEPS['step_D']['suffix']`` is ``''`` and
``CONSTITUTIVE_SIDES['H']['half_integer']`` is False, so the two halves read the SAME
three ``kms`` volumes. The plan ASSERTS that equality rather than assuming it --
binding the half-integer set instead compiles and is a smooth, converged,
half-cell-wrong absorber profile.

=============================================================================
WHAT THE BFAST TAIL DOES AND DOES NOT ADD TO THE WELD
=============================================================================

The BFAST curl is ``shaders``' certified curl template with three additions
(``bfast_curl.py:212-220``): three state pointers, six scalars, and the Tustin tail
between the curl and the ownership mask. **It reads no magnetic cell the plain curl
does not.** The tail's operands are the curl's OWN registers -- ``c_y``/``c``,
``b_z``/``b``, ``a_z``/``a``, ``c_x``/``c``, ``b_x``/``b``, ``a_y``/``a`` -- so
redirecting the plain nine loads (three own-cell, six shifted) redirects the tail
with them, and this module reuses :data:`.fused_hd_pair.OWN_LOAD_EDITS`,
:data:`.fused_hd_pair.HALO_TAPS` and :func:`.fused_hd_pair.offset_coordinates`
UNCHANGED. That reuse is legitimate because the BFAST template's load lines ARE the
certified template's own, character for character; a spelling that drifted makes
:func:`.offdiag_weld_common.needle` raise here rather than emit a kernel reading the
wrong volume.

``s0``/``s1``/``s2`` are read and written at the THREAD'S OWN CELL only
(``st0 = s0[ii]`` ... ``s0[ii] = st0 + adv0``), so they are stepped in place and no
thread observes another thread's store -- the same argument ``D`` and ``fu_D`` rest
on. The state is not part of the rotation and must not be: the array path does not
rotate it either, and the driver's flux backup/restore around the magnetic half-step
(driver.py:4126-4135) depends on the mirror wrapping ``fields.f_bfast_*`` itself.

=============================================================================
WHAT SITS IN THE SEAM
=============================================================================

:data:`CARRIES_DEPOSIT_REPAIR` is False and it is a FACT about the driver rather than
a choice: nothing is INJECTED between the two consults. The magnetic injection is one
seam earlier (driver.py:3283-3284) and the electric one is one seam later
(:3317-3322), so :mod:`..deposit_repair` is not consulted at all and a source of
either polarity is not this seam's business.

What IS between them is the electric integrated-source withdraw (:3313-3314), and
:mod:`..withdraw_hoist` owns it. :data:`HOISTS_THE_WITHDRAW` is False, for
:mod:`.fused_hd_pair`'s reason and not a weaker one: the only wiring that performs
the hoist is ``launch._install_fused_pair``'s ``withdraw_hoist.SEAM`` branch, this
product declares :data:`INSTALLABLE` False, so that branch is unreachable for it and
NOTHING would perform the withdraw before its launch. The predicate therefore refuses
BY NAME every row whose electric withdraw does work. The cell's one corpus row
carries none (``integrated_electric_sources: 0``), so the refusal costs zero
instances here and is kept because a flag that is True without its wiring reports
success on a ``D`` still holding the previous step's standing dipole.

=============================================================================
THE CELL, AND WHAT INSTALLING WOULD COST
=============================================================================

The ``(BFAST -> BFAST)`` H->D cell is **1 corpus row** on the standing Metal census:
``tests:TestReflectanceAngular.test_reflectance_angular_2_35_7``. It is one of the
fifteen ``buildable_not_built`` H->D instances the 2026-09-07 board reports and the
only BFAST one.

:data:`INSTALLABLE` is False and :data:`INSTALLABLE_REASON` carries a MEASURED
arbitration rather than a policy -- and the measurement CORRECTED this paragraph's
first reading rather than confirming it. Over the driver's ``step_B - update_H -
step_D - update_E`` slot path launches are ``4 - (installed pairs)``, and a product
spanning ``update_H``/``step_D`` takes one slot from EACH neighbour. It is natural to
assume both neighbours install here, since :mod:`.bfast_fused_magnetic_pair` and
:mod:`.bfast_fused_electric_pair` are both built and gated. **They do not.** Asked on
the fixture set, the shipped composer selects the separate certified ``BFAST`` arms on
``step_B`` and ``update_H`` and the released ``bfast_fused_electric_pair`` on
``step_D``/``update_E``: the magnetic pair carries no ``launch.FUSED_PAIR_ARMS`` row
and is refused before anything about the run is asked. So the composition is 3
launches and 1 seam served today, and installing this product in the D->E pair's place
would be 3 launches and 1 seam served as well -- a **TIE**, which the released
precedent resolves for the incumbent. The gate's ``arbitration`` leg drives the
shipped composer both with and without an in-process absorb row and records the
verdict rather than restating it here.

What this product IS worth is stated plainly, because no artifact here may imply
otherwise: the priced predicate gap closed under the owner rule *close all fusion
gaps regardless*, and a certified half of the four-slot span that is NOT built here.
**No timing exists for this shape and none is licensed by anything in this module.**

=============================================================================
WHAT THIS FAMILY DOES NOT CARRY
=============================================================================

ONE ARM PAIR ONLY: ``(BFAST -> BFAST)``. Complex storage under BFAST is refused by
the curl half's own predicate (``bfast_complex`` selects nothing on this backend at
all); a FOLDED BFAST run selects the ``folded BFAST`` arm on both slots and is a
different cell and a different product; an INACTIVE absorber puts the constitutive on
the null arm, where there is no second kernel to fuse with; cylindrical storage is
refused by both halves. Every one of those refusals is the two halves' OWN certified
predicate, which this module conjoins without weakening.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..triton_kernels.bfast_curl import BFAST_STATE_NAMES, bfast_curl_coefficients
from ..triton_kernels.coverage import CONSTITUTIVE_SIDES, Coverage
from . import bfast_curl as _bfast
from . import coefficient_pack, shaders
from . import fused_hd_pair as _plain
from . import offdiag_weld_common as _weld
from .device import MAX_BUFFER_BINDINGS, Residency, compile_source
from .launch import SUB_STEPS
from .. import withdraw_hoist as _withdraw_hoist

FAMILY = "bfast_fused_hd_pair"

#: The sub-step slot this arm is registered on: the FIRST half of the seam, in the
#: driver's own order, so a refusal is named on the slot the fusion starts at.
SLOT = "update_H"

#: The driver passes one launch of this plan performs, in driver order
#: (driver.py:3311, :3315). Declared rather than inferred from the slot name.
REPLACES: Tuple[str, ...] = ("update_H", "step_D")

#: The seam name ``launch.FUSED_PAIR_SEAMS`` files this row under, spelled through
#: :mod:`..withdraw_hoist` rather than as a literal so the two cannot drift.
SEAM: str = _withdraw_hoist.SEAM

#: Nothing is injected between the two consults, so there is no deposit to bracket.
#: See the module docstring: this is a fact about the driver, not a choice.
CARRIES_DEPOSIT_REPAIR = False

#: False, and the flag and the wiring move together. See the module docstring.
HOISTS_THE_WITHDRAW = False

#: May the composer install this product? NO, on a MEASURED arbitration verdict.
INSTALLABLE = False

INSTALLABLE_REASON = (
    "MEASURED 2026-09-07 through the shipped Metal composer on the (BFAST -> "
    "BFAST) fixture set, and the measurement CORRECTED the first reading of this "
    "cell rather than confirming it. Over the driver's step_B - update_H - step_D "
    "- update_E slot path launches are 4 - (installed pairs), and this product "
    "takes one slot from EACH neighbouring seam. On a BFAST run exactly ONE "
    "neighbour installs: bfast_fused_electric_pair holds step_D/update_E, while "
    "bfast_fused_magnetic_pair carries NO launch.FUSED_PAIR_ARMS row at all and "
    "is therefore refused before anything about the run is asked -- step_B and "
    "update_H are the separate certified BFAST arms on every one of the nine "
    "fixtures. So the composition is 3 launches / 1 seam served today, and "
    "installing this product in the D->E pair's place would also be 3 launches / "
    "1 seam served: a TIE, not the loss the cell's first reading assumed, and a "
    "gain on nothing. The released precedent resolves a tie for the INCUMBENT. "
    "Driven both ways in the gate's arbitration leg: with no absorb row the seam "
    "loop refuses this product by name before asking anything about the run, and "
    "with the absorb row ('BFAST', 'BFAST') applied IN-PROCESS the refusal moves "
    "to this flag and the released D->E pair keeps its two slots on every "
    "fixture. This is slot arbitration and not a verdict about the arithmetic, "
    "which this family's gate certifies on its synthetic fixtures AND over the "
    "corpus cell. What it leaves: the board's credited instance is served by "
    "predicate admission only; the product installs on zero rows; it executes "
    "nowhere outside its own gate and tests; and it licenses no timing claim and "
    "no dispatch claim")

#: WELDED declaration. Empty means "this family holds a released device gate";
#: non-empty means the gate has run and REFUSED, and the string is what a release
#: still owes. ``test_metal_weld_contract.test_every_metal_family_is_welded``
#: partitions the ``gate_metal_*.py`` fleet against ``fingerprints.json``, so this
#: is the only way a gated family may stand unwelded.
#:
#: EMPTY SINCE 2026-09-08: ``fingerprints.json`` carries
#: ``metal_bfast_fused_hd_pair_device_gate``, minted by
#: ``mint_metal_weld.py --family bfast_fused_hd_pair`` from the RELEASED artifact
#: ``parity/meep_gpu/results/metal_bfast_fused_hd_pair_2026-09-08_weld/flush/
#: gate.json`` -- the tool's own output, unedited, which is the whole difference
#: between this entry and the hand-written one the retired text below records. The
#: gate was RE-RUN against these bytes: emptying this string moves the module, the
#: module is the first pin of its own weld, and ``mint_metal_weld.py`` refuses an
#: artifact whose recorded digests have moved -- so the emptying lands FIRST and the
#: gate measures the emptied tree.
WELD_OWED: str = ""

#: WHAT THE GATE MEASURED, KEPT, AND WHY THIS ONE IS WORTH KEEPING TWICE OVER. The
#: pre-weld declaration is not deleted with the weld: it is the one-line record of
#: which artifact the release rests on, and it is also this family's record of the
#: 2026-09-08 forgery -- an entry hand-written to buy a green test, whose tells were a
#: ``code_sha256`` map holding raw sha256 values and a ``recorded_utc`` rounded to the
#: minute. VERBATIM, including the ``_2026-09-07`` stamp it names.
_RETIRED_WELD_OWED = (
    "the device gate parity/meep_gpu/gate_metal_bfast_fused_hd_pair.py has RUN and RELEASED on this host "
    "under both float32 subnormal policies, into "
    "parity/meep_gpu/results/metal_bfast_fused_hd_pair_2026-09-07/{keep,flush}/gate.json. What a release still "
    "owes is the LEDGER, not a measurement: the metal_bfast_fused_hd_pair_device_gate entry in "
    "metal_kernels/fingerprints.json binding that artifact, MINTED BY "
    "parity/meep_gpu/mint_metal_weld.py against the wired bytes. THIS STRING WAS "
    "EMPTIED ON 2026-09-08 BESIDE A HAND-WRITTEN LEDGER ENTRY AND HAS BEEN RESTORED: "
    "that entry carried raw sha256 values in its code_sha256 map, which "
    "code_identity.code_digest_of_path cannot produce, and a recorded_utc rounded to "
    "the minute where the minting tool writes seconds -- so it was not the tool's "
    "output. A drifted verdict is re-earned by re-running the gate, not by editing "
    "the record (meep_gpu/AGENTS.md). The wiring merge moved this module and "
    "metal_kernels/launch.py, so the gate must be RE-RUN against the wired bytes "
    "before any entry is minted. Empty this string in the same change that lands it"
)

#: The read-only per-axis coefficient VECTORS that ride one buffer, IN PACK ORDER:
#: the curl's six, then the constitutive's three. Disjointly named, so one prologue
#: re-creates all nine under their certified names and every lifted line that
#: indexes them stands character for character.
#:
#: THE CONSTITUTIVE'S ``km0``/``km1``/``km2`` ARE NOT MEMBERS. They are the curl's
#: ``kmx``/``kmy``/``kmz`` -- the same three volumes, shared because both halves sit
#: on the same Yee sub-lattice -- and the lift renames the reads rather than binding
#: them twice, exactly as :mod:`.fused_hd_pair` does.
PACKED_VECTORS: Tuple[str, ...] = ("kmx", "sinvx", "kmy", "sinvy", "kmz", "sinvz",
                                   "kp0", "kp1", "kp2")

#: THE THREE BINDING COUNTS, ALL COMPILED BY THE GATE:
#:
#: * :data:`PACKED_BINDINGS` -- the shipped shape, 25 pointers plus ``Params``.
#: * :data:`UNPACKED_POINTER_BINDINGS` -- 33 pointers plus one packed ``Params&``:
#:   the shape without the coefficient pack, REFUSED at 34 against 31.
#: * :data:`SEPARATE_SCALAR_BINDINGS` -- 33 pointers plus 11 separate scalars.
PACKED_BINDINGS = 26
UNPACKED_POINTER_BINDINGS = 34
SEPARATE_SCALAR_BINDINGS = 44

#: The pointer counts, spelled apart from the binding counts.
PACKED_POINTERS = 25
UNPACKED_POINTERS = 33

if PACKED_BINDINGS > MAX_BUFFER_BINDINGS:  # pragma: no cover - a design invariant
    raise RuntimeError(
        f"the BFAST fused H/D pair binds {PACKED_BINDINGS} buffers and Metal's "
        f"ceiling on this toolchain is {MAX_BUFFER_BINDINGS}")

#: The rotating volumes, in SIGNATURE ORDER. :data:`.fused_hd_pair.ROTATED_NAMES`,
#: imported: the rotation is the same rotation and one spelling of it is one place
#: for it to be right.
ROTATED_NAMES: Tuple[str, ...] = _plain.ROTATED_NAMES

#: The six host-rounded Tustin scalars, in the order the BFAST curl's own signature
#: declares them (bfast_curl.py:251-256) and the order
#: :func:`..triton_kernels.bfast_curl.bfast_curl_coefficients` returns.
K_SCALARS: Tuple[str, ...] = ("k1_0", "k2_0", "k1_1", "k2_1", "k1_2", "k2_2")

__all__ = [
    "CARRIES_DEPOSIT_REPAIR", "FAMILY", "HOISTS_THE_WITHDRAW", "INSTALLABLE",
    "INSTALLABLE_REASON", "K_SCALARS", "PACKED_BINDINGS", "PACKED_POINTERS",
    "PACKED_VECTORS", "REPLACES", "ROTATED_NAMES", "SEAM",
    "SEPARATE_SCALAR_BINDINGS", "SLOT", "UNPACKED_POINTERS",
    "UNPACKED_POINTER_BINDINGS", "WELD_OWED",
    "bfast_fused_hd_pair_source", "bfast_welded_curl_tail",
    "compile_bfast_fused_hd_pair", "metal_bfast_fused_hd_pair_coverage",
    "plan_metal_bfast_fused_hd_pair", "refuted_separate_scalar_source",
    "refuted_unpacked_pointer_source", "register_arms",
    "shipped_signature_bindings",
]


# ---------------------------------------------------------------------------
# The lift: the certified BFAST step_D curl with every H read redirected
# ---------------------------------------------------------------------------

def bfast_welded_curl_tail(codes: Sequence[int],
                           contract: str = shaders.CONTRACT_OFF) -> str:
    """The BFAST ``step_D`` curl body below the decode prologue, H reads redirected.

    THE BODY IS :func:`.fused_hd_pair.welded_curl_tail`'S, WITH ONE SUBSTITUTION: the
    source is :func:`.bfast_curl.bfast_curl_source` rather than
    :func:`.shaders.curl_source`. Every table this walks is the plain product's
    (:data:`.fused_hd_pair.OWN_LOAD_EDITS`, :data:`.fused_hd_pair.HALO_TAPS`,
    :func:`.fused_hd_pair.offset_coordinates`,
    :data:`.fused_hd_pair.H_CELL_TAIL_ARGS`), because the BFAST template's decode,
    ghost, load and offset lines ARE the certified template's own -- so an edit to
    how the certified curl spells a neighbour index reaches both welds without a
    second edit, and a needle that stopped matching RAISES here.

    THE BFAST TAIL SURVIVES THE LIFT UNTOUCHED, and that is a property of what it
    touches rather than an exemption: it reads ``s0``/``s1``/``s2`` at the thread's
    own cell and the curl's OWN registers, and contains no ``gN[`` at all. The final
    "no ``gN[`` survives" check is what makes that a measurement.
    """
    source = _bfast.bfast_curl_source(
        codes, bool(SUB_STEPS["step_D"]["backward"]), contract, has_bfast=True)
    if _plain.DECODE_END not in source:
        raise AssertionError(
            "the certified BFAST curl source no longer carries the decode anchor; "
            "the fused kernel would splice a truncated body")
    tail = source.split(_plain.DECODE_END, 1)[1]
    if not tail.endswith("}\n"):
        raise AssertionError("the certified BFAST curl source does not end with '}'")
    tail = tail[: -len("}\n")]
    offsets = _plain.offset_coordinates(tail)
    for old, new in _plain.OWN_LOAD_EDITS:
        tail = _weld.needle(tail, old, new)
    for var, target in _plain.HALO_TAPS:
        prefix = f"    float {var} = "
        matches = [line + "\n" for line in tail.splitlines()
                   if line.startswith(prefix)]
        if len(matches) != 1:
            raise AssertionError(
                f"the certified BFAST curl declares {var} {len(matches)} times; "
                f"this weld redirects exactly one magnetic load per shifted tap")
        old = matches[0]
        # THE GUARD AND THE OFFSET ARE BOTH PARSED FROM THE EMITTER'S OWN LINE,
        # never retyped: everything left of " ? " is the emitter's validity
        # expression and the offset name inside the brackets selects the
        # coordinates, so a metallic ghost that read an exact 0.0f still does.
        head, rest = old.split(" ? ", 1)
        expected = f"g{target}["
        if not rest.startswith(expected) or not rest.endswith(" : 0.0f;\n"):
            raise AssertionError(
                f"the certified BFAST curl no longer reads {var} as "
                f"`{expected}...]` served an exact 0.0f past the wall; the ghost "
                f"rule this weld preserves has changed")
        offset = rest[len(expected):].split("]", 1)[0]
        coordinates = offsets[offset]
        call = (f"h_cell({', '.join(coordinates)}, "
                f"{_plain.H_CELL_TAIL_ARGS}).a{target}")
        tail = _weld.needle(tail, old, f"{head} ? {call} : 0.0f;\n")
    for target in range(3):
        if f"g{target}[" in tail:
            raise AssertionError(
                f"the welded BFAST curl half still reads g{target}; in this "
                f"signature that pointer does not exist and every magnetic read "
                f"must be the register or a recompute")
    return tail


# ---------------------------------------------------------------------------
# The source
# ---------------------------------------------------------------------------

_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

struct Params {
    uint nx; uint ny; uint nz; uint n_elem; float dtdx;
    float k1_0; float k2_0; float k1_1; float k2_1; float k1_2; float k2_2;
__PACK_FIELDS__
};

__H_CELL__

kernel void bfast_fused_hd_pair_step(
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
    device float*       s0      [[buffer(21)]],
    device float*       s1      [[buffer(22)]],
    device float*       s2      [[buffer(23)]],
    device const float* cpack   [[buffer(24)]],
    constant Params&    prm     [[buffer(25)]],
    uint idx [[thread_position_in_grid]])
{
    // THE SCALARS AND THE PACKED VECTORS ARE UNPACKED INTO THE CERTIFIED BODIES'
    // OWN NAMES, once, before any lifted text runs. Everything below this block is
    // then character-for-character what `bfast_curl.bfast_curl_source` and
    // `shaders.constitutive_source('H')` emit, with the declared lift edits and the
    // redirected magnetic reads.
    uint nx = prm.nx, ny = prm.ny, nz = prm.nz, n_elem = prm.n_elem;
    float dtdx = prm.dtdx;
    float k1_0 = prm.k1_0, k2_0 = prm.k2_0;
    float k1_1 = prm.k1_1, k2_1 = prm.k2_1;
    float k1_2 = prm.k1_2, k2_2 = prm.k2_2;
__PACK_PROLOGUE__
__PROLOGUE__
    // --- THE WELD: update_H, computed into registers and stored to SCRATCH ------
    // Nothing written here is read by this launch. The curl half below takes its
    // OWN cell's magnetic field from these registers and recomputes every foreign
    // tap from PRE-LAUNCH state through the same `h_cell`, so no thread observes
    // another thread's store; the launcher rotates H/f_w_H afterwards.
    h_cell_result own = h_cell(i, j, k, __H_CELL_ARGS__);
    ho0[ii] = own.a0; ho1[ii] = own.a1; ho2[ii] = own.a2;
    wo0[ii] = own.src0; wo1[ii] = own.src1; wo2[ii] = own.src2;

    // --- step_D, BFAST (stepping.step_D / _bfast_term:896 / _apply_pml_update) --
    // D, fu_D and the Tustin state f_bfast_D update IN PLACE and that is safe by
    // construction: the curl and the tail read and write them at the THREAD'S OWN
    // CELL only (`f0[ii]`, `u0[ii]`, `s0[ii]`), so no thread reads a value another
    // thread wrote.
__CURL__
}
"""


def bfast_fused_hd_pair_source(codes: Sequence[int],
                               contract: str = shaders.CONTRACT_OFF) -> str:
    """One specialised fused source for a (boundary triple, contraction mode).

    Every constant Triton bakes into a ``tl.constexpr`` is baked into the string
    here, for the reason :mod:`.shaders` gives: ``torch.mps.compile_shader`` takes a
    source and nothing else.

    THE WALL CLEAR IS NOT CARRIED and must not be. ``zero_metal_B`` runs one seam
    EARLIER (driver.py:3306, before the ``update_H`` consult) and ``zero_metal_D``
    one seam LATER (:3324, after ``step_D``); neither is inside this seam, so a mask
    here would be a pass the driver runs again.
    """
    codes = tuple(int(code) for code in codes)
    if len(codes) != 3:
        raise ValueError(f"codes must be a per-axis triple, got {codes!r}")

    curl = _bfast.bfast_curl_source(
        codes, bool(SUB_STEPS["step_D"]["backward"]), contract, has_bfast=True)
    prologue = (curl.split(_plain._BODY_ANCHOR, 1)[1]
                .split(_plain.DECODE_END, 1)[0] + _plain.DECODE_END)
    return shaders.substitute(_TEMPLATE, {
        "__CONTRACT__": shaders.contraction_pragma(contract),
        "__PACK_FIELDS__": coefficient_pack.params_fields(PACKED_VECTORS),
        "__PACK_PROLOGUE__": coefficient_pack.prologue("cpack", PACKED_VECTORS),
        "__H_CELL__": _plain.h_cell_function(contract),
        "__PROLOGUE__": prologue,
        "__H_CELL_ARGS__": _plain.H_CELL_TAIL_ARGS,
        "__CURL__": bfast_welded_curl_tail(codes, contract),
    })


def compile_bfast_fused_hd_pair(codes: Sequence[int],
                                contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised entry point for one (boundaries, mode)."""
    source = bfast_fused_hd_pair_source(codes, contract)
    return compile_source(source).bfast_fused_hd_pair_step


# ---------------------------------------------------------------------------
# The two refuted signatures — measurements, not arguments
# ---------------------------------------------------------------------------

_WRITTEN: Tuple[str, ...] = ("ho0", "ho1", "ho2", "wo0", "wo1", "wo2",
                             "f0", "f1", "f2", "u0", "u1", "u2",
                             "s0", "s1", "s2")
_READ_UNPACKED: Tuple[str, ...] = ("hi0", "hi1", "hi2", "wi0", "wi1", "wi2",
                                   "b0", "b1", "b2",
                                   "kmx", "sinvx", "kmy", "sinvy", "kmz", "sinvz",
                                   "kp0", "kp1", "kp2")

#: The eleven scalars the shipped kernel packs into one ``constant Params&``.
_SCALARS: Tuple[Tuple[str, str], ...] = (
    ("constant uint&", "nx"), ("constant uint&", "ny"), ("constant uint&", "nz"),
    ("constant uint&", "n_elem"), ("constant float&", "dtdx"),
) + tuple(("constant float&", name) for name in K_SCALARS)


def _touch_kernel(name: str, written: Sequence[str], read: Sequence[str],
                  contract: str, packed: bool = True) -> Tuple[int, str]:
    """A signature with a body that touches every buffer, and nothing else.

    What is being measured is the SIGNATURE; a body the compiler could drop would
    let dead-code elimination decide the answer, which is why every buffer is read
    and every written one is stored. ``packed`` binds the eleven scalars as ONE
    ``constant Params&``, exactly as the shipped kernel does.
    """
    lines: List[str] = []
    slot = 0
    for label in written:
        lines.append(f"    device float*       {label:<8}[[buffer({slot})]],")
        slot += 1
    for label in read:
        lines.append(f"    device const float* {label:<8}[[buffer({slot})]],")
        slot += 1
    if packed:
        lines.append(f"    constant Params&    prm     [[buffer({slot})]],")
        slot += 1
        unpack = ["    uint nx = prm.nx, ny = prm.ny, nz = prm.nz, "
                  "n_elem = prm.n_elem;",
                  "    float dtdx = prm.dtdx;"]
    else:
        for kind, label in _SCALARS:
            lines.append(f"    {kind:<19} {label:<8}[[buffer({slot})]],")
            slot += 1
        unpack = []
    touch = " + ".join(f"{label}[0]" for label in read)
    return slot, "\n".join((
        "#include <metal_stdlib>",
        "using namespace metal;",
        "",
        shaders.contraction_pragma(contract),
        "",
        "struct Params { uint nx; uint ny; uint nz; uint n_elem; float dtdx; };",
        "",
        f"kernel void {name}(",
        *lines,
        "    uint idx [[thread_position_in_grid]])",
        "{",
        *unpack,
        "    if (idx >= n_elem) { return; }",
        f"    float touch = {touch};",
        f"    {written[0]}[idx] = {written[0]}[idx] + touch * float(nx + ny + nz);",
        *[f"    {label}[idx] = {label}[idx] * dtdx;" for label in written[1:]],
        "}",
        "",
    ))


def shipped_signature_bindings() -> int:
    """How many bindings the SHIPPED kernel declares, counted off its own source.

    Read from the emitted text rather than from :data:`PACKED_BINDINGS`, so the
    constant is a claim the source can falsify.
    """
    source = bfast_fused_hd_pair_source((0, 0, 0))
    signature = source.split("kernel void bfast_fused_hd_pair_step(", 1)[1]
    signature = signature.split("uint idx [[thread_position_in_grid]])", 1)[0]
    return signature.count("[[buffer(")


def refuted_unpacked_pointer_source(contract: str = shaders.CONTRACT_OFF) -> str:
    """The 34-binding signature with the nine coefficient vectors UNPACKED.

    THE MEASUREMENT THAT MAKES THE PACK A NECESSITY rather than a tidiness: with
    each read-only per-axis vector given its own binding the signature is 33
    pointers plus one packed ``Params`` and does not compile at all. Not shipped and
    not launchable; it exists so the gate can compile it and require the failure.
    """
    slots, source = _touch_kernel("refuted_unpacked_pointer", _WRITTEN,
                                  _READ_UNPACKED, contract)
    assert slots == UNPACKED_POINTER_BINDINGS, (slots, UNPACKED_POINTER_BINDINGS)
    return source


def refuted_separate_scalar_source(contract: str = shaders.CONTRACT_OFF) -> str:
    """The 44-binding unpacked-scalar signature this platform REFUSES."""
    slots, source = _touch_kernel("refuted_separate_scalar", _WRITTEN,
                                  _READ_UNPACKED, contract, packed=False)
    assert slots == SEPARATE_SCALAR_BINDINGS, (slots, SEPARATE_SCALAR_BINDINGS)
    return source


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def metal_bfast_fused_hd_pair_coverage(fields: Any, pml: Any, sources: Any = None,
                                       residency: Any = None) -> Coverage:
    """May ONE dispatch span ``update_H`` -> the electric withdraw -> ``step_D``
    on a BFAST run?

    A conjunction of the two halves' OWN certified predicates plus the seam clauses.
    Nothing is weakened: a configuration either half refuses is refused here with
    that half's reasons, prefixed so a reader can tell which side said it.

    THE ORDER IS THE DRIVER'S. The constitutive half is asked FIRST because it runs
    first (driver.py:3311), so the first refusal a reader sees names the half the
    driver would have reached first.
    """
    reasons: List[str] = []

    magnetic = _bfast.bfast_run_constitutive_coverage(fields, pml, "H", residency)
    if not magnetic.covered:
        reasons.extend(f"constitutive half: {reason}" for reason in magnetic.reasons)
    curl = _bfast.bfast_pml_curl_coverage(fields, pml, "step_D", residency)
    if not curl.covered:
        reasons.extend(f"curl half: {reason}" for reason in curl.reasons)

    # THE SEAM'S ONE PASS. Nothing is INJECTED between the two consults, so
    # `deposit_repair` is not consulted at all and a source of either polarity is
    # not this seam's business. What IS between them is the electric
    # integrated-source withdraw (driver.py:3313-3314), and `withdraw_hoist` owns
    # it. IGNORANCE IS NEVER AN EMPTY SET: `Fields` does not hold the source list,
    # so a predicate that inferred "no withdraw stands" from not being told would be
    # the over-covering refusal this clause exists to prevent.
    reasons.extend(_withdraw_hoist.seam_withdraw_reasons(
        fields, sources,
        undeclared=(
            "the source set was not declared: this predicate cannot infer from "
            "Fields that no electric withdraw stands between update_H and step_D"),
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

    # THE FOLD, RESTATED BY NAME. Both halves already refuse a mirror plane, and on
    # THIS seam the reason is not the one the D->E and B->H pairs give: neither fill
    # runs between these two consults (fill_B closes one seam earlier at
    # driver.py:3306-3309 and fill_D opens one seam later at :3324-3327), so the
    # H->D seam is fill-FREE on a folded grid. What refuses a fold here is that this
    # product implements the unfolded `BFAST` arms on both slots and a folded BFAST
    # run selects the `folded BFAST` arm instead -- a different cell and a different
    # product.
    for axis in range(3):
        mirrored = getattr(grid, "is_mirrored", None)
        if callable(mirrored) and bool(mirrored(axis)):
            reasons.append(
                f"axis {axis} is folded: this product implements the unfolded "
                f"`BFAST` update_H and step_D arms, and a folded BFAST run selects "
                f"the `folded BFAST` arm on both slots; that cell needs its own "
                f"product and its own ghost-map measurement")

    # THE ROTATION IS THE PRODUCT'S OWN INVARIANT. The plan swaps the ENGINE's
    # references for the six volumes in ROTATED_NAMES after each launch, so those
    # attributes must be settable and must be the arrays the residency mirrored.
    for name in ROTATED_NAMES:
        if getattr(fields, name, None) is None:
            reasons.append(
                f"{name} is not allocated; this weld rotates it against a "
                f"plan-owned scratch twin after every launch")
    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

def _params_tensor(shape: Sequence[int], dtdx: float, ks: Sequence[float],
                   offsets: Mapping[str, int], device: str) -> Any:
    """The five grid scalars, the six Tustin scalars and the nine pack offsets as
    one device record.

    PLAN-OWNED AND OUTSIDE THE RESIDENCY REGISTRY, for the reason every fused pair's
    ``_params_tensor`` gives: :meth:`.Residency.mirror` binds float32 and complex64
    volumes and refuses anything else BY NAME. This record is neither.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    fields = ([("nx", "<u4"), ("ny", "<u4"), ("nz", "<u4"), ("n_elem", "<u4"),
               ("dtdx", "<f4")]
              + [(name, "<f4") for name in K_SCALARS]
              + coefficient_pack.record_dtype_fields(PACKED_VECTORS))
    record = np.zeros(1, dtype=np.dtype(fields))
    nx, ny, nz = (int(n) for n in shape)
    record[0] = tuple(
        [nx, ny, nz, nx * ny * nz, np.float32(dtdx)]
        + [np.float32(value) for value in ks]
        + [int(offsets[name]) for name in PACKED_VECTORS])
    words = np.frombuffer(record.tobytes(), dtype=np.int32).copy()
    return torch.from_numpy(words).to(torch.device(device))


def plan_metal_bfast_fused_hd_pair(
        fields: Any, pml: Any, sources: Any = None,
        residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        functions: Optional[Mapping[str, Any]] = None,
        ) -> Optional["_weld.ScratchWeldPairPlan"]:
    """Build the BFAST fused H/D plan, or ``None`` when the seam is refused.

    ``None`` is the only refusal, for the reason :func:`.launch.plan_pml_curl`
    gives: a configuration this kernel does not carry must fall back to the array
    path, never raise into a caller that would otherwise have stepped correctly.

    ``functions`` is the mutation seam. The gate compiles a deliberately broken copy
    of the shipped source and hands it here; dropping the argument is not a silent
    slowdown, it is a silent DISARMING.
    """
    if not metal_bfast_fused_hd_pair_coverage(fields, pml, sources,
                                              residency).covered:
        return None
    import numpy as np  # noqa: PLC0415

    from ..stepping import _boundary_kinds  # noqa: PLC0415

    grid = fields.grid
    codes = tuple(1 if kind == "metallic" else 0
                  for kind in _boundary_kinds(grid, pml))

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
    # THE TUSTIN STATE, IN PLACE. The mirror wraps `fields.f_bfast_D*` itself, which
    # the driver's flux backup/restore around the magnetic half-step depends on
    # (driver.py:4126-4135); because the IIR is marginally stable a missed restore
    # never decays.
    state = [bind(name, getattr(fields, name))
             for name in BFAST_STATE_NAMES["step_D"]]

    # ONE SUB-LATTICE, TWO HALVES. `SUB_STEPS['step_D']['suffix']` is '' and
    # `CONSTITUTIVE_SIDES['H']['half_integer']` is False, so the curl's kms and the
    # constitutive's kms are THE SAME THREE VOLUMES and are packed once. Both
    # suffixes are READ FROM THE SHIPPED TABLES rather than written as literals, and
    # their equality is ASSERTED rather than assumed: it is the whole premise of the
    # nine-member pack, and if either table moved, sharing the group would bind one
    # half's coefficients to the other half's lattice -- a converged, smooth,
    # half-cell-wrong absorber profile rather than a failure.
    suffix = SUB_STEPS["step_D"]["suffix"]
    constitutive_suffix = "_h" if CONSTITUTIVE_SIDES["H"]["half_integer"] else ""
    if suffix != constitutive_suffix:
        raise AssertionError(
            f"step_D reads the {suffix or 'integer'!r} PML sub-lattice and update_H "
            f"the {constitutive_suffix or 'integer'!r} one; this weld packs ONE kms "
            f"group for both halves and that is only correct while they agree")

    vectors = ([getattr(pml, f"{stem}_{axis}{suffix}")
                for axis in "xyz" for stem in ("kms", "sinv")]
               + [getattr(pml, f"kps_{axis}{constitutive_suffix}")
                  for axis in "xyz"])
    pack, layout = coefficient_pack.packed_mirror(
        residency, f"{FAMILY}:cpack", np, PACKED_VECTORS, vectors)
    volumes.append(f"{FAMILY}:cpack")

    invariant = tuple(grid.is_invariant(axis) for axis in range(3))
    ks = bfast_curl_coefficients(grid.bfast_scaled_k, invariant, magnetic=False)

    selected = dict(functions or {})
    for mode in contract_variants:
        if mode not in selected:
            selected[mode] = compile_bfast_fused_hd_pair(codes, mode)

    dtdx = grid.dt / grid.dx
    static = (flux + displacement + auxiliary + state + [pack]
              + [_params_tensor(grid.shape, dtdx, ks,
                                coefficient_pack.offsets_for(layout),
                                residency.device)])
    assert (2 * len(ROTATED_NAMES) + len(static)) == PACKED_BINDINGS, (
        2 * len(ROTATED_NAMES) + len(static), PACKED_BINDINGS)
    assert PACKED_BINDINGS <= MAX_BUFFER_BINDINGS, (
        PACKED_BINDINGS, MAX_BUFFER_BINDINGS)
    return _weld.plan_scratch_weld(
        FAMILY, residency, fields, rotated_names=ROTATED_NAMES,
        static_args=static, functions=selected, volumes=volumes,
        shape=grid.shape, codes=codes, zero_metal=(0, 0, 0), row_mask=(),
        replaces=REPLACES, twins=twins)


# ---------------------------------------------------------------------------
# Registration — wired=False, and refused by the composer twice over
# ---------------------------------------------------------------------------

def _arm_coverage(context: Any, slot: str) -> Coverage:
    if slot != SLOT:
        return Coverage(False, (f"BFAST fused H/D pair cannot fill {slot}",))
    return metal_bfast_fused_hd_pair_coverage(
        context.fields, context.pml, context.sources, context.residency)


def _arm_plan(context: Any, slot: str) -> Optional["_weld.ScratchWeldPairPlan"]:
    if slot != SLOT:
        return None
    return plan_metal_bfast_fused_hd_pair(
        context.fields, context.pml, context.sources, context.residency,
        context.contract_variants)


def register_arms() -> Tuple[Any, ...]:
    """One row on ``update_H``, ``wired=False``.

    REGISTERED SO IT IS ENUMERABLE, NOT SO IT IS SELECTABLE.
    ``arms.arms_for`` skips an unwired row so ``plan_step`` cannot select it, while
    ``arms.registered`` still returns it -- which is what lets the composition sweep
    measure disjointness against this predicate instead of assuming it, and what
    lets ``launch._neighbouring_seam_claimant`` see the row when it asks who claims
    the seam ``update_H`` opens.
    """
    from . import arms  # noqa: PLC0415

    return (arms.register(FAMILY, SLOT, "BFAST fused H/D pair",
                          _arm_coverage, _arm_plan,
                          prefix="BFAST fused H/D pair: ",
                          noun="fused BFAST stored-H/BFAST D-curl pair",
                          wired=False, replaces=REPLACES),)


ARMS = register_arms()
