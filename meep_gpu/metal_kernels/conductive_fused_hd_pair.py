"""The conductive H->D weld: the ORDINARY ``update_H`` into the CONDUCTIVE ``step_D``.

THE FOURTH ARM PAIR ON THE FOURTH SEAM, and the only one on this backend whose two
halves come from DIFFERENT FAMILIES. The driver runs

    ``update_H`` (driver.py:3311) -> the electric integrated-source withdraw
    (:3313-3314) -> ``step_D`` (:3315)

and on a run carrying an ELECTRIC conductivity the composer's own selection is
``update_H = ordinary`` with ``step_D = conductive PML curl``. That is not a board
mislabel and it is not an inference: ``metal_composition_matrix.MATRIX``'s
``cart_pml_conductive_electric`` row pins exactly that pair as the shipped composer's
answer, and the reason is stated one line above it -- a conductivity changes the CURL
recurrence only (``stepping._apply_curl`` reads ``condfac_for`` at S:479 and nothing
else does), so the dedicated conductive family owns the affected curl while both
constitutive slots and the other curl stay on their ordinary products. The
constitutive half of this weld is therefore :func:`.shaders.constitutive_source`'s
side-``H`` body -- the SAME certified body :mod:`.fused_hd_pair` lifts -- and the
curl half is :func:`.conductive_pml.conductive_pml_curl_source`'s.

=============================================================================
THE SHAPE, AND WHY IT IS THE PLAIN WELD'S
=============================================================================

``step_D``'s curl reads ``H`` at the thread's own cell AND at its three BACKWARD
neighbours, and ``update_H`` writes ``H``. Both hazards are removed the way
:mod:`.fused_hd_pair` removes them, and this module reuses that construction rather
than restating it:

* **the constitutive half writes nothing in place.** ``H_new`` and ``f_w_H_new`` go
  to LAUNCH-LOCAL SCRATCH, so ``H``, ``f_w_H`` and ``B`` are ``const`` for the whole
  dispatch and no thread can observe another thread's store;
* **the foreign read is a RECOMPUTE from that same unwritten state**, through
  :func:`.fused_hd_pair.h_cell_function` -- the certified ``constitutive_step`` body
  for side ``H``, whole, evaluated at an arbitrary cell;
* the launcher then ROTATES the ``H``/``f_w_H`` bindings, which is
  :class:`.offdiag_weld_common.ScratchWeldPairPlan`'s certified choreography.

``D``, ``fu_D`` and the conductive history ``f_cond_D`` step IN PLACE, and that is
safe by construction: the conductive tail reads and writes all three at the THREAD'S
OWN CELL only (``f0[ii]``, ``u0[ii]``, ``c0[ii]``), so no thread reads a value
another thread wrote. The conductivity coefficient volumes ``cf``/``ci`` are
READ-ONLY on the device.

=============================================================================
THE SIGNATURE: TWO PACKS, AND THE SECOND ONE IS LOAD-BEARING
=============================================================================

Unpacked this weld needs::

    3 H_out + 3 f_w_H_out          (scratch, written, never read)
  + 3 H_in  + 3 f_w_H_in           (pre-launch, const)
  + 3 B                            (const, the constitutive source)
  + 3 D + 3 fu_D + 3 f_cond_D      (in place, own cell only)
  + 3 condfac + 3 condinv          (const, read-only conductivity VOLUMES)
  + 6 curl coefficients (kms/sinv per axis)
  + 3 constitutive kps                                                        = 39

plus one ``constant Params&`` is **40 against the platform's 31**
(:data:`.device.MAX_BUFFER_BINDINGS`) -- the same "over by nine" the 2026-09-01 board
scored ``UNFUSABLE ON METAL`` for this cell's D->E twin, which
:mod:`.conductive_fused_electric_pair` closed with :mod:`.coefficient_pack`.

**Here the vector pack alone is not enough, and that is a measurement rather than a
preference.** Packing the nine read-only per-axis VECTORS (the curl's
``kmx``/``sinvx``/... and the constitutive's ``kp0``/``kp1``/``kp2``) saves eight
pointers and lands on **31 pointers plus ``Params`` = 32**: over the ceiling by
EXACTLY ONE. :func:`refuted_vectors_only_source` is that shape, and the gate compiles
it and requires the failure.

The one more binding comes from a SECOND pack holding the six read-only conductivity
VOLUMES. :mod:`.conductive_fused_electric_pair` considered and declined exactly this
("folding three-dimensional volumes in beside them would make the offsets a
mixed-geometry table for no binding need -- 29 is already two under the ceiling");
this seam supplies the binding need, and the answer to the mixed-geometry objection
is to use TWO packs rather than one, so each pack's offsets stay single-geometry. 26
pointers plus ``Params`` is :data:`PACKED_BINDINGS` = 27, four under the ceiling.

**A non-conductive target contributes no pack member at all.** The emitted tail for a
lossless target never names ``cf{n}``/``ci{n}`` (``conductive_pml._tail`` with
``conductive=False`` reads only the four PML coefficients), so its offset is written
as 0 and nothing is stored for it. That is checkable rather than hopeful:
:func:`unread_conductivity_members` returns the members the plan omits and the plan
ASSERTS the emitted source names none of them, so a future edit that made a lossless
tail read ``cf0[ii]`` fails at plan build instead of reading past the pack.

THE CONSTITUTIVE ``km`` GROUP IS SHARED WITH THE CURL'S, exactly as in
:mod:`.fused_hd_pair`: ``launch.SUB_STEPS['step_D']['suffix']`` is ``''`` and
``CONSTITUTIVE_SIDES['H']['half_integer']`` is False, so the two halves read the SAME
three ``kms`` volumes. The plan ASSERTS that equality -- binding the half-integer set
instead compiles and is a smooth, converged, half-cell-wrong absorber profile.

=============================================================================
THE LIFT NEEDS ITS OWN EDIT TABLES, AND THE REASON IS SPELLING
=============================================================================

:mod:`.folded_fused_hd_pair` and :mod:`.bfast_fused_hd_pair` reuse
:data:`.fused_hd_pair.OWN_LOAD_EDITS` and :data:`.fused_hd_pair.HALO_TAPS` whole,
because the folded and BFAST curl templates' load lines ARE
:mod:`.shaders`' template's, character for character. **The conductive template's are
not**: it declares three own-cell loads on ONE line
(``float a = g0[ii], b = g1[ii], c = g2[ii];``) and its six shifted loads TWO to a
line, and it ends its index decode with ``int j = plane % nyi, i = plane / nyi;``
rather than the package's :data:`.offdiag_weld_common.DECODE_END`. So this module
carries :data:`DECODE_END` and :func:`redirect_magnetic_reads` of its own -- but
still PARSES the guard and the offset out of the emitter's own text rather than
retyping either, so a metallic ghost that read an exact ``0.0f`` still reads an exact
``0.0f`` and the periodic wrap is still the emitter's integer expression. A spelling
that drifted RAISES here rather than emitting a kernel that reads the wrong volume.

=============================================================================
WHAT SITS IN THE SEAM
=============================================================================

:data:`CARRIES_DEPOSIT_REPAIR` is False and it is a FACT about the driver rather than
a choice: nothing is INJECTED between the two consults. The magnetic injection is one
seam earlier (driver.py:3283-3284) and the electric one is one seam later
(:3317-3322), so :mod:`..deposit_repair` is not consulted at all -- which is the ONE
place this weld is simpler than its D->E twin, whose single corpus row declares an
electric source inside its seam and which therefore sets the flag True.

What IS between them is the electric integrated-source withdraw (:3313-3314), and
:mod:`..withdraw_hoist` owns it. :data:`HOISTS_THE_WITHDRAW` is False, for
:mod:`.fused_hd_pair`'s reason: the only wiring that performs the hoist is
``launch._install_fused_pair``'s ``withdraw_hoist.SEAM`` branch, this product
declares :data:`INSTALLABLE` False, so that branch is unreachable for it and NOTHING
would perform the withdraw before its launch. The cell's one corpus row carries no
integrated electric source, so the refusal costs zero instances here.

=============================================================================
THE CELL, AND WHAT INSTALLING WOULD COST
=============================================================================

The ``(ordinary -> conductive PML curl)`` H->D cell is **1 corpus row** on the
standing Metal census: ``tests:TestAdjointSolver.test_damping`` -- the SAME row whose
D->E seam :mod:`.conductive_fused_electric_pair` serves.

:data:`INSTALLABLE` is False on a MEASURED arbitration: over the driver's ``step_B -
update_H - step_D - update_E`` slot path launches are ``4 - (installed pairs)``, and
this product takes one slot from EACH neighbour. On this configuration the D->E
neighbour is a released, gated Metal product (:mod:`.conductive_fused_electric_pair`,
which holds ``step_D``) and the B->H neighbour is the released
:mod:`.fused_magnetic_pair` -- the magnetic curl on an electric-conductivity run is
the ORDINARY one, which is what ``cart_pml_conductive_magnetic``'s mirror-image row
records. So installing this product is a LOSS on any row where both serve and a TIE
at best, and a gain on none. The gate's ``arbitration`` leg drives the shipped
composer and records the verdict rather than restating it here.

**No timing exists for this shape and none is licensed by anything in this module.**

=============================================================================
WHAT THIS FAMILY DOES NOT CARRY
=============================================================================

ONE ARM PAIR ONLY: ``(ordinary -> conductive PML curl)``, real float32 Cartesian
storage under an ACTIVE split-field PML. A MAGNETIC conductivity puts the conductive
curl on ``step_B`` and leaves ``step_D`` ordinary -- that is
:mod:`.fused_hd_pair`'s cell, refused here by the curl half's own per-sub-step
clause. Complex storage is :mod:`.complex_conductive_pml`'s and is refused by both
halves; an INACTIVE absorber selects the no-PML conductive curl and the null
constitutive, where there is no second kernel to fuse with; a fold, a beta, BFAST, a
nonlinearity and cylindrical storage are each refused by the two halves' own
certified predicates, which this module conjoins without weakening.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..triton_kernels.conductivity import conductive_targets
from ..triton_kernels.coverage import CONSTITUTIVE_SIDES, Coverage
from . import coefficient_pack, conductive_pml, shaders
from . import fused_hd_pair as _plain
from . import offdiag_weld_common as _weld
from .coverage import constitutive_coverage
from .device import MAX_BUFFER_BINDINGS, Residency, compile_source
from .launch import SUB_STEPS
from .. import withdraw_hoist as _withdraw_hoist

FAMILY = "conductive_fused_hd_pair"

#: The sub-step slot this arm is registered on: the FIRST half of the seam, in the
#: driver's own order, so a refusal is named on the slot the fusion starts at.
SLOT = "update_H"

#: The driver passes one launch of this plan performs, in driver order
#: (driver.py:3311, :3315). Declared rather than inferred from the slot name.
REPLACES: Tuple[str, ...] = ("update_H", "step_D")

#: The seam name ``launch.FUSED_PAIR_SEAMS`` files this row under.
SEAM: str = _withdraw_hoist.SEAM

#: Nothing is injected between the two consults, so there is no deposit to bracket.
CARRIES_DEPOSIT_REPAIR = False

#: False, and the flag and the wiring move together. See the module docstring.
HOISTS_THE_WITHDRAW = False

#: May the composer install this product? NO, on a MEASURED arbitration verdict.
INSTALLABLE = False

INSTALLABLE_REASON = (
    "over the driver's step_B - update_H - step_D - update_E slot path launches "
    "are 4 - (installed pairs), and this product takes one slot from EACH "
    "neighbouring seam. On an ELECTRIC-conductivity run the D->E neighbour is the "
    "released, gated conductive_fused_electric_pair (which holds step_D and "
    "serves the same corpus row, tests:TestAdjointSolver.test_damping) and the "
    "B->H neighbour is the released fused_magnetic_pair, because the MAGNETIC "
    "curl on an electric-conductivity run is the ordinary one "
    "(metal_composition_matrix's cart_pml_conductive_electric row). Installing "
    "this product is therefore a LOSS wherever both neighbours serve and a TIE at "
    "best, and a gain on none. The composer refuses it without reading this flag: "
    "the B->H pair installs first and holds update_H, so _pair_may_absorb refuses "
    "this product by name, and the D->E claimant is named by the composer's later "
    "half. This is slot arbitration and not a verdict about the arithmetic, which "
    "this family's gate certifies on its synthetic fixtures AND over the corpus "
    "cell. What it leaves: the board's credited instance is served by predicate "
    "admission only; the product installs on zero rows; it executes nowhere "
    "outside its own gate and tests; and it licenses no timing claim and no "
    "dispatch claim")

#: WELDED declaration -- empty means "this family holds a released device gate".
#:
#: EMPTY SINCE 2026-09-08: ``fingerprints.json`` carries
#: ``metal_conductive_fused_hd_pair_device_gate``, minted by
#: ``mint_metal_weld.py --family conductive_fused_hd_pair`` from the RELEASED artifact
#: ``parity/meep_gpu/results/metal_conductive_fused_hd_pair_2026-09-08_weld/flush/
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
#: minute. VERBATIM, including the ``_2026-09-07`` stamp it names, except that its
#: citation of the re-run rule now names ``docs/development/certification.md``,
#: the file in this repository that states that rule.
_RETIRED_WELD_OWED = (
    "the device gate parity/meep_gpu/gate_metal_conductive_fused_hd_pair.py has RUN and RELEASED on this host "
    "under both float32 subnormal policies, into "
    "parity/meep_gpu/results/metal_conductive_fused_hd_pair_2026-09-07/{keep,flush}/gate.json. What a release still "
    "owes is the LEDGER, not a measurement: the metal_conductive_fused_hd_pair_device_gate entry in "
    "metal_kernels/fingerprints.json binding that artifact, MINTED BY "
    "parity/meep_gpu/mint_metal_weld.py against the wired bytes. THIS STRING WAS "
    "EMPTIED ON 2026-09-08 BESIDE A HAND-WRITTEN LEDGER ENTRY AND HAS BEEN RESTORED: "
    "that entry carried raw sha256 values in its code_sha256 map, which "
    "code_identity.code_digest_of_path cannot produce, and a recorded_utc rounded to "
    "the minute where the minting tool writes seconds -- so it was not the tool's "
    "output. A drifted verdict is re-earned by re-running the gate, not by editing "
    "the record (docs/development/certification.md). The wiring merge moved this module and "
    "metal_kernels/launch.py, so the gate must be RE-RUN against the wired bytes "
    "before any entry is minted. Empty this string in the same change that lands it"
)

#: The read-only per-axis coefficient VECTORS that ride the first pack, IN PACK
#: ORDER: the curl's six, then the constitutive's three. Disjointly named, so one
#: prologue re-creates all nine under their certified names.
#:
#: THE CONSTITUTIVE'S ``km0``/``km1``/``km2`` ARE NOT MEMBERS -- they are the curl's
#: ``kmx``/``kmy``/``kmz``, the same three volumes, and the lift renames the reads
#: rather than binding them twice.
PACKED_VECTORS: Tuple[str, ...] = ("kmx", "sinvx", "kmy", "sinvy", "kmz", "sinvz",
                                   "kp0", "kp1", "kp2")

#: The read-only conductivity VOLUMES that ride the second pack, IN PACK ORDER.
#: Every member is grid-shaped, which is why they get a pack of their own rather
#: than joining the per-axis vectors in one mixed-geometry offset table.
PACKED_VOLUMES: Tuple[str, ...] = ("cf0", "cf1", "cf2", "ci0", "ci1", "ci2")

#: THE FOUR BINDING COUNTS, ALL COMPILED BY THE GATE:
#:
#: * :data:`PACKED_BINDINGS` -- the shipped shape, 26 pointers plus ``Params``.
#: * :data:`VECTORS_ONLY_BINDINGS` -- the nine per-axis vectors packed and the six
#:   conductivity volumes NOT: 31 pointers plus ``Params``, over the ceiling by
#:   EXACTLY ONE. This is what makes the second pack load-bearing.
#: * :data:`UNPACKED_POINTER_BINDINGS` -- 39 pointers plus ``Params``: the shape the
#:   board's ``UNFUSABLE ON METAL`` verdict priced, refused at 40 against 31.
#: * :data:`SEPARATE_SCALAR_BINDINGS` -- 39 pointers plus 5 separate scalars.
PACKED_BINDINGS = 27
VECTORS_ONLY_BINDINGS = 32
UNPACKED_POINTER_BINDINGS = 40
SEPARATE_SCALAR_BINDINGS = 44

#: The pointer counts, spelled apart from the binding counts.
PACKED_POINTERS = 26
UNPACKED_POINTERS = 39

if PACKED_BINDINGS > MAX_BUFFER_BINDINGS:  # pragma: no cover - a design invariant
    raise RuntimeError(
        f"the conductive fused H/D pair binds {PACKED_BINDINGS} buffers and Metal's "
        f"ceiling on this toolchain is {MAX_BUFFER_BINDINGS}")

#: The rotating volumes, in SIGNATURE ORDER. :data:`.fused_hd_pair.ROTATED_NAMES`,
#: imported: the rotation is the same rotation.
ROTATED_NAMES: Tuple[str, ...] = _plain.ROTATED_NAMES

#: Where the CONDUCTIVE curl template ends its index decode. NOT
#: :data:`.offdiag_weld_common.DECODE_END`: this emitter declares ``j`` and ``i`` on
#: one line where the certified curl declares them on two, so the anchor is its own.
DECODE_END = "    int j = plane % nyi, i = plane / nyi;\n"

#: The conductive template's THREE own-cell magnetic loads -- all on ONE line -- and
#: what they become. The thread has already computed its own ``H`` into ``own``, and
#: that register is the post-``update_H`` value the array path would have loaded.
OWN_LOAD_EDIT: Tuple[str, str] = (
    "    float a = g0[ii], b = g1[ii], c = g2[ii];\n",
    "    float a = own.a0, b = own.a1, c = own.a2;\n")

__all__ = [
    "CARRIES_DEPOSIT_REPAIR", "DECODE_END", "FAMILY", "HOISTS_THE_WITHDRAW",
    "INSTALLABLE", "INSTALLABLE_REASON", "OWN_LOAD_EDIT", "PACKED_BINDINGS",
    "PACKED_POINTERS", "PACKED_VECTORS", "PACKED_VOLUMES", "REPLACES",
    "ROTATED_NAMES", "SEAM", "SEPARATE_SCALAR_BINDINGS", "SLOT",
    "UNPACKED_POINTERS", "UNPACKED_POINTER_BINDINGS", "VECTORS_ONLY_BINDINGS",
    "WELD_OWED", "compile_conductive_fused_hd_pair",
    "conductive_fused_hd_pair_source", "conductive_welded_curl_tail",
    "metal_conductive_fused_hd_pair_coverage",
    "plan_metal_conductive_fused_hd_pair", "redirect_magnetic_reads",
    "refuted_separate_scalar_source", "refuted_unpacked_pointer_source",
    "refuted_vectors_only_source", "register_arms", "shipped_signature_bindings",
    "unread_conductivity_members",
]


# ---------------------------------------------------------------------------
# The lift: the certified conductive step_D curl with every H read redirected
# ---------------------------------------------------------------------------

def redirect_magnetic_reads(tail: str) -> str:
    """Replace the conductive curl's nine magnetic loads with register/recompute.

    THE TWO-PER-LINE SPELLING IS WHY THIS IS NOT
    :func:`.fused_hd_pair.welded_curl_tail`. Each shifted tap is found as the
    FRAGMENT ``<var> = <guard> ? g<t>[<offset>] : 0.0f`` inside whatever line it
    shares, and every piece of it except the leaf load is the emitter's own text:
    the guard is taken from left of ``" ? "``, the offset name from inside the
    brackets, and the coordinates from
    :func:`.fused_hd_pair.offset_coordinates`'s parse of the emitter's own
    ``int ox = si * nyz + j * nzi + k;`` lines. Nothing is retyped, so a metallic
    ghost that read an exact ``0.0f`` still reads an exact ``0.0f``.
    """
    offsets = _plain.offset_coordinates(tail)
    old, new = OWN_LOAD_EDIT
    tail = _weld.needle(tail, old, new)
    for var, target in _plain.HALO_TAPS:
        key = f"{var} = "
        if tail.count(key) != 1:
            raise AssertionError(
                f"the certified conductive curl writes `{key}` {tail.count(key)} "
                f"times; this weld redirects exactly one magnetic load per shifted "
                f"tap")
        start = tail.index(key)
        stop = tail.index(" : 0.0f", start) + len(" : 0.0f")
        fragment = tail[start:stop]
        head, rest = fragment.split(" ? ", 1)
        expected = f"g{target}["
        if not rest.startswith(expected) or not rest.endswith(" : 0.0f"):
            raise AssertionError(
                f"the certified conductive curl no longer reads {var} as "
                f"`{expected}...]` served an exact 0.0f past the wall; the ghost "
                f"rule this weld preserves has changed")
        offset = rest[len(expected):].split("]", 1)[0]
        coordinates = offsets[offset]
        call = (f"h_cell({', '.join(coordinates)}, "
                f"{_plain.H_CELL_TAIL_ARGS}).a{target}")
        tail = tail[:start] + f"{head} ? {call} : 0.0f" + tail[stop:]
    for target in range(3):
        if f"g{target}[" in tail:
            raise AssertionError(
                f"the welded conductive curl half still reads g{target}; in this "
                f"signature that pointer does not exist and every magnetic read "
                f"must be the register or a recompute")
    return tail


def conductive_welded_curl_tail(codes: Sequence[int],
                                conductive: Sequence[bool],
                                contract: str = shaders.CONTRACT_OFF) -> str:
    """The conductive ``step_D`` curl body below the decode prologue, H redirected.

    ``codes`` is the per-axis PERIODIC/METALLIC triple and ``conductive`` the
    per-target lossy flags; ``backward`` is read from :data:`.launch.SUB_STEPS`
    rather than written as a literal, so this family and the shipped table cannot
    drift.
    """
    source = conductive_pml.conductive_pml_curl_source(
        codes, bool(SUB_STEPS["step_D"]["backward"]), conductive, contract)
    if DECODE_END not in source:
        raise AssertionError(
            "the certified conductive curl source no longer carries this family's "
            "decode anchor; the fused kernel would splice a truncated body")
    tail = source.split(DECODE_END, 1)[1]
    if not tail.endswith("}\n"):
        raise AssertionError(
            "the certified conductive curl source does not end with '}'")
    return redirect_magnetic_reads(tail[: -len("}\n")])


def unread_conductivity_members(conductive: Sequence[bool]) -> Tuple[str, ...]:
    """The conductivity pack members a LOSSLESS target contributes nothing for.

    ``conductive_pml._tail`` emits ``cf{n}[ii]``/``ci{n}[ii]`` only on the lossy
    branch, so a lossless target's two members are never indexed and the plan writes
    their offsets as 0 rather than storing a copy of a volume nothing reads. The
    plan ASSERTS the emitted source names none of these, which is what turns "never
    indexed" into a measurement.
    """
    flags = tuple(bool(flag) for flag in conductive)
    if len(flags) != 3:
        raise ValueError(f"conductive must be a per-target triple, got {flags!r}")
    return tuple(f"{stem}{index}" for index in range(3) if not flags[index]
                 for stem in ("cf", "ci"))


# ---------------------------------------------------------------------------
# The source
# ---------------------------------------------------------------------------

_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

struct Params {
    uint nx; uint ny; uint nz; uint n_elem; float dtdx;
__VECTOR_FIELDS__
__VOLUME_FIELDS__
};

__H_CELL__

kernel void conductive_fused_hd_pair_step(
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
    device float*       c0      [[buffer(21)]],
    device float*       c1      [[buffer(22)]],
    device float*       c2      [[buffer(23)]],
    device const float* cpack   [[buffer(24)]],
    device const float* vpack   [[buffer(25)]],
    constant Params&    prm     [[buffer(26)]],
    uint idx [[thread_position_in_grid]])
{
    // THE FIVE SCALARS AND BOTH PACKS ARE UNPACKED INTO THE CERTIFIED BODIES' OWN
    // NAMES, once, before any lifted text runs. Everything below this block is then
    // character-for-character what `conductive_pml.conductive_pml_curl_source` and
    // `shaders.constitutive_source('H')` emit, with the declared lift edits and the
    // redirected magnetic reads.
    uint nx = prm.nx, ny = prm.ny, nz = prm.nz, n_elem = prm.n_elem;
    float dtdx = prm.dtdx;
__VECTOR_PROLOGUE__
__VOLUME_PROLOGUE__
__PROLOGUE__
    // --- THE WELD: update_H, computed into registers and stored to SCRATCH ------
    // Nothing written here is read by this launch. The curl half below takes its
    // OWN cell's magnetic field from these registers and recomputes every foreign
    // tap from PRE-LAUNCH state through the same `h_cell`, so no thread observes
    // another thread's store; the launcher rotates H/f_w_H afterwards.
    h_cell_result own = h_cell(i, j, k, __H_CELL_ARGS__);
    ho0[ii] = own.a0; ho1[ii] = own.a1; ho2[ii] = own.a2;
    wo0[ii] = own.src0; wo1[ii] = own.src1; wo2[ii] = own.src2;

    // --- step_D, conductive (stepping.step_D / _apply_curl:479) -----------------
    // D, fu_D and the conductive history f_cond_D update IN PLACE and that is safe
    // by construction: the conductive tail reads and writes all three at the
    // THREAD'S OWN CELL only (`f0[ii]`, `u0[ii]`, `c0[ii]`).
__CURL__
}
"""


def conductive_fused_hd_pair_source(codes: Sequence[int],
                                    conductive: Sequence[bool],
                                    contract: str = shaders.CONTRACT_OFF) -> str:
    """One specialised fused source: (boundary triple, lossy triple, mode).

    THE WALL CLEAR IS NOT CARRIED and must not be. ``zero_metal_B`` runs one seam
    EARLIER (driver.py:3306) and ``zero_metal_D`` one seam LATER (:3324); neither is
    inside this seam, so a mask here would be a pass the driver runs again.
    """
    codes = tuple(int(code) for code in codes)
    conductive = tuple(bool(flag) for flag in conductive)
    if len(codes) != 3 or len(conductive) != 3:
        raise ValueError(
            f"codes and conductive must each be a triple, got {codes!r} and "
            f"{conductive!r}")

    curl = conductive_pml.conductive_pml_curl_source(
        codes, bool(SUB_STEPS["step_D"]["backward"]), conductive, contract)
    prologue = (curl.split(_plain._BODY_ANCHOR, 1)[1]
                .split(DECODE_END, 1)[0] + DECODE_END)
    return shaders.substitute(_TEMPLATE, {
        "__CONTRACT__": shaders.contraction_pragma(contract),
        "__VECTOR_FIELDS__": coefficient_pack.params_fields(PACKED_VECTORS),
        "__VOLUME_FIELDS__": coefficient_pack.params_fields(PACKED_VOLUMES),
        "__VECTOR_PROLOGUE__": coefficient_pack.prologue("cpack", PACKED_VECTORS),
        "__VOLUME_PROLOGUE__": coefficient_pack.prologue("vpack", PACKED_VOLUMES),
        "__H_CELL__": _plain.h_cell_function(contract),
        "__PROLOGUE__": prologue,
        "__H_CELL_ARGS__": _plain.H_CELL_TAIL_ARGS,
        "__CURL__": conductive_welded_curl_tail(codes, conductive, contract),
    })


def compile_conductive_fused_hd_pair(codes: Sequence[int],
                                     conductive: Sequence[bool],
                                     contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised entry point for one (boundaries, lossy triple, mode)."""
    source = conductive_fused_hd_pair_source(codes, conductive, contract)
    return compile_source(source).conductive_fused_hd_pair_step


# ---------------------------------------------------------------------------
# The three refuted signatures — measurements, not arguments
# ---------------------------------------------------------------------------

_WRITTEN: Tuple[str, ...] = ("ho0", "ho1", "ho2", "wo0", "wo1", "wo2",
                             "f0", "f1", "f2", "u0", "u1", "u2",
                             "c0", "c1", "c2")
_READ_FIELDS: Tuple[str, ...] = ("hi0", "hi1", "hi2", "wi0", "wi1", "wi2",
                                 "b0", "b1", "b2")
_READ_VOLUMES: Tuple[str, ...] = PACKED_VOLUMES
_READ_VECTORS: Tuple[str, ...] = PACKED_VECTORS


def _touch_kernel(name: str, written: Sequence[str], read: Sequence[str],
                  contract: str, packed: bool = True) -> Tuple[int, str]:
    """A signature with a body that touches every buffer, and nothing else.

    What is being measured is the SIGNATURE; a body the compiler could drop would
    let dead-code elimination decide the answer, which is why every buffer is read
    and every written one is stored.
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
        for kind, label in (("constant uint&", "nx"), ("constant uint&", "ny"),
                            ("constant uint&", "nz"), ("constant uint&", "n_elem"),
                            ("constant float&", "dtdx")):
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
    """How many bindings the SHIPPED kernel declares, counted off its own source."""
    source = conductive_fused_hd_pair_source((0, 0, 0), (True, True, True))
    signature = source.split("kernel void conductive_fused_hd_pair_step(", 1)[1]
    signature = signature.split("uint idx [[thread_position_in_grid]])", 1)[0]
    return signature.count("[[buffer(")


def refuted_vectors_only_source(contract: str = shaders.CONTRACT_OFF) -> str:
    """The 32-binding shape with ONLY the per-axis vectors packed.

    THE MEASUREMENT THAT MAKES THE SECOND PACK LOAD-BEARING. With the nine read-only
    per-axis vectors in one buffer and the six read-only conductivity VOLUMES each
    given their own binding, the signature is 31 pointers plus one packed ``Params``
    -- over the ceiling by EXACTLY ONE, so the volume pack is a necessity rather
    than a tidiness. Not shipped and not launchable: it exists so the gate can
    compile it and require the failure.
    """
    slots, source = _touch_kernel(
        "refuted_vectors_only", _WRITTEN,
        _READ_FIELDS + _READ_VOLUMES + ("cpack",), contract)
    assert slots == VECTORS_ONLY_BINDINGS, (slots, VECTORS_ONLY_BINDINGS)
    return source


def refuted_unpacked_pointer_source(contract: str = shaders.CONTRACT_OFF) -> str:
    """The 40-binding shape with NOTHING packed — the board's UNFUSABLE verdict."""
    slots, source = _touch_kernel(
        "refuted_unpacked_pointer", _WRITTEN,
        _READ_FIELDS + _READ_VOLUMES + _READ_VECTORS, contract)
    assert slots == UNPACKED_POINTER_BINDINGS, (slots, UNPACKED_POINTER_BINDINGS)
    return source


def refuted_separate_scalar_source(contract: str = shaders.CONTRACT_OFF) -> str:
    """The 44-binding unpacked-scalar signature this platform REFUSES."""
    slots, source = _touch_kernel(
        "refuted_separate_scalar", _WRITTEN,
        _READ_FIELDS + _READ_VOLUMES + _READ_VECTORS, contract, packed=False)
    assert slots == SEPARATE_SCALAR_BINDINGS, (slots, SEPARATE_SCALAR_BINDINGS)
    return source


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def metal_conductive_fused_hd_pair_coverage(fields: Any, pml: Any,
                                            sources: Any = None,
                                            residency: Any = None) -> Coverage:
    """May ONE dispatch span ``update_H`` -> the electric withdraw -> ``step_D`` on
    an electric-conductivity run?

    A conjunction of the two halves' OWN certified predicates plus the seam clauses,
    and the two halves come from DIFFERENT MODULES: the constitutive half is
    :func:`.coverage.constitutive_coverage`'s side-``H`` verdict -- the ``ordinary``
    arm's own body -- and the curl half is
    :func:`.conductive_pml.metal_conductive_pml_curl_coverage`'s on ``step_D``.
    Nothing is weakened: a configuration either half refuses is refused here with
    that half's reasons, prefixed so a reader can tell which side said it.

    THE ORDER IS THE DRIVER'S. The constitutive half is asked FIRST because it runs
    first (driver.py:3311).
    """
    reasons: List[str] = []

    magnetic = constitutive_coverage(fields, pml, "H", residency)
    if not magnetic.covered:
        reasons.extend(f"constitutive half: {reason}" for reason in magnetic.reasons)
    curl = conductive_pml.metal_conductive_pml_curl_coverage(
        fields, pml, "step_D", residency)
    if not curl.covered:
        reasons.extend(f"curl half: {reason}" for reason in curl.reasons)
    lossy = tuple(conductive_targets(fields, "step_D"))

    # THE SEAM'S ONE PASS -- the electric integrated-source withdraw
    # (driver.py:3313-3314). Nothing is INJECTED between the two consults, so
    # `deposit_repair` is not consulted at all. IGNORANCE IS NEVER AN EMPTY SET:
    # `Fields` does not hold the source list, so a predicate that inferred "no
    # withdraw stands" from not being told would be the over-covering refusal this
    # clause exists to prevent.
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
    # THIS seam the usual reason does not apply: neither fill runs between these two
    # consults (fill_B closes one seam earlier at driver.py:3306-3309 and fill_D
    # opens one seam later at :3324-3327), so the H->D seam is fill-FREE on a folded
    # grid. What refuses a fold here is that this product implements the `ordinary`
    # update_H and `conductive PML curl` step_D arms, and a folded run selects the
    # `folded` arm on the constitutive slot.
    for axis in range(3):
        mirrored = getattr(grid, "is_mirrored", None)
        if callable(mirrored) and bool(mirrored(axis)):
            reasons.append(
                f"axis {axis} is folded: this product implements the `ordinary` "
                f"update_H and `conductive PML curl` step_D arms, and a folded run "
                f"selects the `folded` arm on the constitutive slot; a folded "
                f"conductive cell needs its own product and its own ghost-map "
                f"measurement")

    # THE ROTATION IS THE PRODUCT'S OWN INVARIANT.
    for name in ROTATED_NAMES:
        if getattr(fields, name, None) is None:
            reasons.append(
                f"{name} is not allocated; this weld rotates it against a "
                f"plan-owned scratch twin after every launch")

    # THE CONDUCTIVE HISTORY THIS WELD STEPS IN PLACE, FOR THE LOSSY TARGETS ONLY.
    # The engine allocates `f_cond_*` per component, so on a partly-conductive run a
    # LOSSLESS target has none -- measured 2026-09-07 on the mixed fixture, where a
    # clause demanding all three refused a configuration this kernel steps correctly.
    # The signature still binds three pointers; the plan aliases a lossless target's
    # slot to that target's own tensor, which is `conductive_pml`'s own construction
    # (conductive_pml.py:`history`) and is safe for the same reason: the emitted tail
    # for a lossless target names `c{n}` nowhere.
    for index, name in enumerate(SUB_STEPS["step_D"]["targets"]):
        if not lossy[index]:
            continue
        if getattr(fields, "f_cond_" + name, None) is None:
            reasons.append(
                f"f_cond_{name} is not allocated but {name} carries a conductivity; "
                f"this weld steps that history in place")
    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

def _params_tensor(shape: Sequence[int], dtdx: float,
                   vector_offsets: Mapping[str, int],
                   volume_offsets: Mapping[str, int], device: str) -> Any:
    """The five scalars and both packs' offsets as one device record.

    PLAN-OWNED AND OUTSIDE THE RESIDENCY REGISTRY: :meth:`.Residency.mirror` binds
    float32 and complex64 volumes and refuses anything else BY NAME, and this record
    is neither.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    fields = ([("nx", "<u4"), ("ny", "<u4"), ("nz", "<u4"), ("n_elem", "<u4"),
               ("dtdx", "<f4")]
              + coefficient_pack.record_dtype_fields(PACKED_VECTORS)
              + coefficient_pack.record_dtype_fields(PACKED_VOLUMES))
    record = np.zeros(1, dtype=np.dtype(fields))
    nx, ny, nz = (int(n) for n in shape)
    record[0] = tuple(
        [nx, ny, nz, nx * ny * nz, np.float32(dtdx)]
        + [int(vector_offsets[name]) for name in PACKED_VECTORS]
        + [int(volume_offsets.get(name, 0)) for name in PACKED_VOLUMES])
    words = np.frombuffer(record.tobytes(), dtype=np.int32).copy()
    return torch.from_numpy(words).to(torch.device(device))


def plan_metal_conductive_fused_hd_pair(
        fields: Any, pml: Any, sources: Any = None,
        residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        functions: Optional[Mapping[str, Any]] = None,
        ) -> Optional["_weld.ScratchWeldPairPlan"]:
    """Build the conductive fused H/D plan, or ``None`` when the seam is refused.

    ``None`` is the only refusal: a configuration this kernel does not carry must
    fall back to the array path, never raise into a caller that would otherwise have
    stepped correctly.

    ``functions`` is the mutation seam. Dropping it is not a silent slowdown, it is
    a silent DISARMING.
    """
    if not metal_conductive_fused_hd_pair_coverage(fields, pml, sources,
                                                   residency).covered:
        return None
    import numpy as np  # noqa: PLC0415

    from ..stepping import _boundary_kinds  # noqa: PLC0415

    grid = fields.grid
    codes = tuple(1 if kind == "metallic" else 0
                  for kind in _boundary_kinds(grid, pml))
    conductive = tuple(conductive_targets(fields, "step_D"))

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

    targets = tuple(SUB_STEPS["step_D"]["targets"])
    flux = [bind(name, getattr(fields, name))
            for name in SUB_STEPS["step_B"]["targets"]]
    displacement = [bind(name, getattr(fields, name)) for name in targets]
    auxiliary = [bind("fu_" + name, getattr(fields, "fu_" + name))
                 for name in targets]
    # THE CONDUCTIVE HISTORY, IN PLACE, FOR THE LOSSY TARGETS. The signature binds
    # three pointers unconditionally; a LOSSLESS target has no `f_cond_*` volume at
    # all (the engine allocates them per component) and its slot is aliased to that
    # target's own tensor -- `conductive_pml.plan_metal_conductive_pml_curl`'s own
    # construction, safe for the same reason and checked against the emitted text
    # below rather than assumed: a lossless tail names `c{n}` nowhere.
    history = [
        (bind("f_cond_" + name, getattr(fields, "f_cond_" + name))
         if conductive[index] else displacement[index])
        for index, name in enumerate(targets)]

    # ONE SUB-LATTICE, TWO HALVES -- asserted, not assumed. See the module docstring.
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
    cpack, vector_layout = coefficient_pack.packed_mirror(
        residency, f"{FAMILY}:cpack", np, PACKED_VECTORS, vectors)
    volumes.append(f"{FAMILY}:cpack")

    # THE VOLUME PACK holds only the members a LOSSY target's tail actually indexes.
    # A lossless target contributes nothing and its offset is 0; the assertion below
    # is what makes "nothing indexes it" a measurement rather than a memory.
    omitted = unread_conductivity_members(conductive)
    aliased = tuple(f"c{index}" for index in range(3) if not conductive[index])
    members = [name for name in PACKED_VOLUMES if name not in omitted]
    stems = {"cf": fields.condfac_for, "ci": fields.condinv_for}
    vpack, volume_layout = coefficient_pack.packed_mirror(
        residency, f"{FAMILY}:vpack", np, members,
        [stems[name[:2]](targets[int(name[2])]) for name in members])
    volumes.append(f"{FAMILY}:vpack")

    selected = dict(functions or {})
    for mode in contract_variants:
        if mode not in selected:
            selected[mode] = compile_conductive_fused_hd_pair(codes, conductive,
                                                              mode)
    for mode in tuple(selected):
        emitted = conductive_fused_hd_pair_source(codes, conductive, mode)
        for name in omitted:
            if f"{name}[" in emitted:
                raise AssertionError(
                    f"the emitted source indexes {name}, whose target is lossless "
                    f"and which this plan therefore stores nothing for; reading it "
                    f"would run past the conductivity pack")
        for name in aliased:
            if f"{name}[" in emitted:
                raise AssertionError(
                    f"the emitted source indexes {name}, whose target is lossless "
                    f"and whose history slot this plan aliases to that target's own "
                    f"displacement; writing it would corrupt D in place")

    dtdx = grid.dt / grid.dx
    static = (flux + displacement + auxiliary + history + [cpack, vpack]
              + [_params_tensor(grid.shape, dtdx,
                                coefficient_pack.offsets_for(vector_layout),
                                coefficient_pack.offsets_for(volume_layout),
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
        return Coverage(False,
                        (f"conductive fused H/D pair cannot fill {slot}",))
    return metal_conductive_fused_hd_pair_coverage(
        context.fields, context.pml, context.sources, context.residency)


def _arm_plan(context: Any, slot: str) -> Optional["_weld.ScratchWeldPairPlan"]:
    if slot != SLOT:
        return None
    return plan_metal_conductive_fused_hd_pair(
        context.fields, context.pml, context.sources, context.residency,
        context.contract_variants)


def register_arms() -> Tuple[Any, ...]:
    """One row on ``update_H``, ``wired=False``.

    REGISTERED SO IT IS ENUMERABLE, NOT SO IT IS SELECTABLE, for the reason
    :func:`.fused_hd_pair.register_arms` gives: the composition sweep measures
    disjointness against this predicate instead of assuming it, and
    ``launch._neighbouring_seam_claimant`` sees the row when it asks who claims the
    seam ``update_H`` opens.
    """
    from . import arms  # noqa: PLC0415

    return (arms.register(FAMILY, SLOT, "conductive fused H/D pair",
                          _arm_coverage, _arm_plan,
                          prefix="conductive fused H/D pair: ",
                          noun="fused stored-H/conductive D-curl pair",
                          wired=False, replaces=REPLACES),)


ARMS = register_arms()
