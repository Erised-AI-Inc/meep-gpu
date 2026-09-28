"""The FOLDED COMPLEX fused D/E pair: a folded Bloch ``step_D`` welded into ``update_E``.

THE D-SIDE TWIN OF :mod:`.folded_complex_fused_magnetic_pair`, and the mirror image of
what :mod:`.folded_fused_pair` is to :mod:`.folded_fused_magnetic_pair` on the real
board. It is the cell ``(folded complex, folded complex)`` at ``D_to_E``, which no
shipped weld can stand in for: :mod:`.folded_fused_pair` is real-storage (its fold
parity is an exact sign flip, and under complex storage it is a full complex multiply
with zero cross terms), and :mod:`.complex_fused_electric_pair` refuses a mirror plane
by name through both its halves.

EVERYTHING IS LIFTED OR IMPORTED, and the three sources are the three the B-side twin
uses with the seam turned round:

* the curl half — :func:`.folded_complex.folded_bloch_curl_source` with
  ``backward=True``, cut at its split-field mark exactly as
  :func:`.folded_complex_fused_magnetic_pair.certified_curl_head` cuts the B one;
* the two mirror fills — :func:`.folded_complex.folded_mirror_fill_complex_source` on
  the ``"D"`` ghost-fill family, compared line for line under a stated rename by
  :func:`near_fill_transcription` / :func:`far_fill_transcription`;
* the constitutive half — :func:`.complex_fields.bloch_constitutive_source` on side
  ``"E"``, whose seven statements per component are reproduced by
  :func:`constitutive_transcription` and required to match on every build.

THE FILL GEOMETRY IS THE D GEOMETRY AND IT IS THE EXACT INVERSE OF THE B TWIN'S.
``fields.IYEE_SHIFTS`` gives Dx (1,0,0), Dy (0,1,0), Dz (0,0,1), so a D component's
Yee shift is ONE on its own axis and ZERO on the other two. The near fill therefore
images a component on the two axes that are NOT its own (up to TWO destinations, and a
corner where both fire) and the far fill only on its OWN axis (at most ONE) — the
complement of the B family, where the near fill takes one axis and the far up to two.
:func:`.folded_fused_pair.near_fill_axes`, :func:`.folded_fused_pair.far_fill_axes`
and :func:`.folded_fused_pair.carried_destinations` are the D-side answers and are
IMPORTED rather than re-derived here.

THE PARITY CHAIN IS ORDERED AND IS NOT FOLDED INTO ONE WORD, and that is inherited
from the B twin as a MEASUREMENT rather than a preference: under complex storage,
re-ordering a two-parity chain moves 8 of 128 uint32 words on an exhaustive
signed-zero/subnormal table and folding it into one word moves up to 23 of 128, at
every sign combination
(``results/complex_parity_chain_order_2026-08-21/parity_chain_order.json``). The
driver's order is near fill (:3300) -> wall clear (:3301) -> far fill (:3302), and
:func:`parity_chain` returns exactly that: the near axes ASCENDING (the array path
loops them in x, y, z order, stepping.py:1441-1447), then the far one.

**THE WALL CLEAR SITS INSIDE THE CHAIN HERE, WHICH IT NEVER DOES ON THE B SIDE.** The
B family's near destination is stored 0 of a FOLDED axis and its far destinations are
axes ``_ZERO_METAL_ROWS`` never touches, so no ghost there takes a clear line at all.
On the D side ``_ZERO_METAL_ROWS`` is the OFF-DIAGONAL — the same shift-0 axes the
near fill images — and the driver clears BETWEEN the two fills. So a destination that
carries near passes takes the clear AFTER them and any far parity AFTER the clear,
which is why every chain step here lands in a register and the ghost store is a
separate line. (The B twin folds the last multiply into the store; a value computed
into a register and then stored is the same word, and the split is what lets the clear
be inserted at the driver's own point.)

===========================================================================
THE SIGNATURE — 30 pointers plus one packed struct, EXACTLY ON THE CEILING
===========================================================================

    3 D  +  3 fu_D  +  3 H        (float2, the curl half)
  + 3 E  +  3 f_w_E               (float2, the constitutive half)
  + 3 inverse epsilon             (float32 — one real coefficient per cell)
  + 6 curl coefficients  +  6 constitutive coefficients   (float32)
  = 30 pointers, + one ``constant Params&`` = 31 = :data:`.device.MAX_BUFFER_BINDINGS`

THREE MORE POINTERS THAN THE B TWIN'S 27, and the difference is physics rather than
layout: ``update_H`` is ``H = B`` with mu = 1 baked into the array path, so that
family DROPS the certified constitutive's inverse-epsilon volumes; ``update_E`` reads
``source * inverse_epsilon_for(component)`` (stepping.py:1011-1013) and cannot. The B
twin has three binding slots spare and this one has none, so the fifteen scalars — the
three Bloch phases, the three near parities, the three far parities, four extents,
``dtdx`` and the three reflect rows — all ride inside ONE ``constant Params&``. One
more pointer of any kind is a compile error.

NOT WIRED AS AN ARM: ``plan_step`` assigns at most one arm per slot and this product
spans five. It registers ``wired=False`` and reaches its two slots through
``launch.FUSED_PAIR_ARMS``.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..triton_kernels.coverage import Coverage, _call, zero_metal_axes
from ..triton_kernels.symmetry import (
    CODE_MIRROR_PERIODIC,
    MIRROR_SOURCE_INDEX,
    TARGET_IYEE,
    folded_axis_kinds,
)
from . import complex_fields, shaders, templates
from .device import MAX_BUFFER_BINDINGS, Residency, compile_source
from .folded_complex import (
    expansion_from_probe,
    folded_bloch_curl_source,
    folded_complex_composition_curl_coverage,
    folded_complex_constitutive_coverage,
    load_expansion_probe,
)
from .folded_complex_fused_magnetic_pair import far_parity_words, parity_words
from .folded_fused_pair import (
    _far_carry_reasons,
    _mirror_phase_reasons,
    carried_destinations,
    far_fill_axes,
    near_fill_axes,
)
from .fused_dispersive_pair import _ZERO_METAL_ROWS
from .plans import KernelPlan
from .symmetry import MIRROR_CODES
from .. import deposit_repair as _deposit_repair

#: Does this product bracket its fused launch with the deposit repair? While False the
#: source-presence clause below refuses every in-seam deposit. Flipping it is a claim
#: about the PLAN this module builds -- that the leading slot saves and the trailing
#: slot restores -- and is only ever changed in the same edit as that wiring.
#:
#: TRUE FROM THIS FAMILY'S FIRST COMMIT, in the same edit as its
#: ``launch.FUSED_PAIR_ARMS`` row, and it is what makes the cell worth anything: BOTH
#: corpus rows here carry an electric source inside the seam. Nothing about the claim
#: is new -- ``deposit_repair.repair_cells`` (deposit_repair.py:217-258) carries a
#: deposit across a FOLDED seam by restoring the closure of the cells the two fills
#: image it into, which is the inverse of the forward carry :func:`carried_destinations`
#: implements here, and it indexes and accumulates element-wise over one array per
#: component -- under complex storage, one ``complex64`` array (fields.py:573).
#: ``repairable`` still refuses BY NAME the folds whose fill map it cannot read: the
#: cylindrical r = 0 axis, and a folded axis storing no more than
#: ``MIRROR_SOURCE_INDEX`` cells.
CARRIES_DEPOSIT_REPAIR = True

FAMILY = "folded_complex_fused_pair"

#: The sub-step slot this arm holds a row on. It spans five; a refusal is NAMED on the
#: slot the fusion starts at rather than being invisible to the table.
SLOT = "step_D"

#: The curl sub-step this product starts at, and the constitutive side it ends at.
CURL_SUB_STEP = "step_D"
CONSTITUTIVE_SIDE = "E"

#: ``backward`` for the lifted curl. Named rather than passed, so a reader cannot mistake
#: this for a family that serves either seam.
BACKWARD = True

#: The driver passes ONE launch of this plan performs, in driver order
#: (driver.py:3292-3303). Declared, never inferred from the slot name.
REPLACES: Tuple[str, ...] = ("step_D", "fill_D", "zero_metal_D",
                             "fill_folded_far_ghosts_D", "update_E")

#: The shipped binding count: 30 pointers + one packed ``constant Params&``, which is
#: MAX_BUFFER_BINDINGS exactly.
PACKED_BINDINGS = 31

#: What the same 30 pointers need with the fifteen scalars bound SEPARATELY. Far over
#: the ceiling; :func:`refuted_separate_scalar_source` builds it and the gate requires
#: the compile failure, so the packing is a measurement rather than a precaution.
SEPARATE_SCALAR_BINDINGS = 30 + 4 + 1 + 3 + 3 + 3 + 3

#: The stored index the near fill images, re-exported so a reader of this module does
#: not have to chase ``triton_kernels.symmetry`` for the fold's magic 2.
NEAR_SOURCE_INDEX = MIRROR_SOURCE_INDEX

#: The host record's total size in bytes: nine ``float2`` (72) + four ``uint`` (16) +
#: one ``float`` (4) + three ``int`` (12) = 104, already a multiple of the struct's
#: 8-byte alignment.
PARAMS_ITEMSIZE = 104

if PACKED_BINDINGS > MAX_BUFFER_BINDINGS:  # pragma: no cover - a design invariant
    raise RuntimeError(
        f"the folded complex fused D/E pair binds {PACKED_BINDINGS} buffers and "
        f"Metal's ceiling on this toolchain is {MAX_BUFFER_BINDINGS}")

__all__ = [
    "BACKWARD", "CARRIES_DEPOSIT_REPAIR", "CONSTITUTIVE_SIDE", "CURL_SUB_STEP",
    "FAMILY", "NEAR_SOURCE_INDEX", "PACKED_BINDINGS", "PARAMS_ITEMSIZE", "REPLACES",
    "SEPARATE_SCALAR_BINDINGS", "SLOT", "MetalFoldedComplexFusedPairPlan",
    "certified_constitutive_body", "certified_curl_head",
    "certified_curl_statements", "compile_folded_complex_fused_pair",
    "constitutive_transcription", "far_fill_transcription",
    "folded_complex_fused_pair_source",
    "metal_folded_complex_fused_pair_coverage", "near_fill_transcription",
    "params_record_dtype", "parity_chain",
    "plan_metal_folded_complex_fused_pair", "refuted_separate_scalar_source",
    "register_arms", "zero_metal_lines",
]


# ---------------------------------------------------------------------------
# The source
# ---------------------------------------------------------------------------

_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

__HELPERS__

// THE NINE float2 MEMBERS COME FIRST, for the reason `complex_fused_magnetic_pair`
// measured on this toolchain: Metal aligns float2 to 8 bytes, so with the scalars
// first the struct needs internal padding and the natural host record puts every
// phase one word early — which reads as a plausible complex number rather than as
// garbage. Phases and parities first, no internal padding.
//
// `m0`/`m1`/`m2` are the NEAR mirror parity on axes x/y/z and `d0`/`d1`/`d2` the FAR
// one, as complex64 words rounded on the host by
// `folded_complex.mirror_parity_coefficients` and passed. The two are DIFFERENT words
// — `mirror_parity` is `phase * (1 - 2*iyee)` (fields.py:180-182) — and both must be
// present because one kernel carries both fills.
//
// FIFTEEN SCALARS, ONE BINDING, AND THE BINDING COUNT IS 31 EXACTLY. The far parities
// and the three reflect rows (stepping._far_reflect_rows:1661) ride inside this record
// rather than as six more buffers; this family sits ON the platform's 31-binding
// ceiling (device.py:72) and could not have paid for them.
struct Params {
    float2 px; float2 py; float2 pz;
    float2 m0; float2 m1; float2 m2;
    float2 d0; float2 d1; float2 d2;
    uint nx; uint ny; uint nz; uint n_elem; float dtdx;
    int rx; int ry; int rz;
};

kernel void folded_complex_fused_pair_step(
    device float2*       f0      [[buffer(0)]],
    device float2*       f1      [[buffer(1)]],
    device float2*       f2      [[buffer(2)]],
    device float2*       u0      [[buffer(3)]],
    device float2*       u1      [[buffer(4)]],
    device float2*       u2      [[buffer(5)]],
    device const float2* g0      [[buffer(6)]],
    device const float2* g1      [[buffer(7)]],
    device const float2* g2      [[buffer(8)]],
    device float2*       h0      [[buffer(9)]],
    device float2*       h1      [[buffer(10)]],
    device float2*       h2      [[buffer(11)]],
    device float2*       w0      [[buffer(12)]],
    device float2*       w1      [[buffer(13)]],
    device float2*       w2      [[buffer(14)]],
    device const float*  e0      [[buffer(15)]],
    device const float*  e1      [[buffer(16)]],
    device const float*  e2      [[buffer(17)]],
    device const float*  kmx     [[buffer(18)]],
    device const float*  sinvx   [[buffer(19)]],
    device const float*  kmy     [[buffer(20)]],
    device const float*  sinvy   [[buffer(21)]],
    device const float*  kmz     [[buffer(22)]],
    device const float*  sinvz   [[buffer(23)]],
    device const float*  kp0     [[buffer(24)]],
    device const float*  km0     [[buffer(25)]],
    device const float*  kp1     [[buffer(26)]],
    device const float*  km1     [[buffer(27)]],
    device const float*  kp2     [[buffer(28)]],
    device const float*  km2     [[buffer(29)]],
    constant Params&     prm     [[buffer(30)]],
    uint idx [[thread_position_in_grid]])
{
    // THE FIFTEEN PACKED ARGUMENTS ARE UNPACKED INTO THE LIFTED BODIES' OWN NAMES,
    // once, before any of the spliced text runs.
    //
    // The parity registers are named `mp0`/`mp1`/`mp2` (near) and `fp0`/`fp1`/`fp2`
    // (far) and NOT `c`: the lifted curl head already declares `float2 c = g2[ii];`,
    // and the certified fill's own spelling of the coefficient is `c`. `n0`/`n1`/`n2`
    // are NOT available either — the lifted split-field recurrence owns them.
    uint nx = prm.nx, ny = prm.ny, nz = prm.nz, n_elem = prm.n_elem;
    float dtdx = prm.dtdx;
    float2 px = prm.px, py = prm.py, pz = prm.pz;
    float2 mp0 = prm.m0, mp1 = prm.m1, mp2 = prm.m2;
    float2 fp0 = prm.d0, fp1 = prm.d1, fp2 = prm.d2;
    int reflect_x = prm.rx, reflect_y = prm.ry, reflect_z = prm.rz;
    (void)mp0; (void)mp1; (void)mp2;
    (void)fp0; (void)fp1; (void)fp2;
    (void)reflect_x; (void)reflect_y; (void)reflect_z;

__BODY__
}
"""

#: The marker that separates a certified shader's signature from its body.
_BODY_ANCHOR = "uint idx [[thread_position_in_grid]])\n{\n"

#: The comment line the folded Bloch curl body opens its split-field block with — the
#: CUT POINT of the lift. Matched as a PREFIX so the block's trailing dash run is not
#: a second thing to keep in step.
_RECURRENCE_MARK = "    // --- split-field recurrence"

#: The flat-index coordinate name per axis, and the stride expression per axis, in the
#: lifted decode's own spelling. A second decode here would be a second place to get
#: the C-contiguous layout wrong.
_COORDINATE: Tuple[str, str, str] = ("i", "j", "k")
_STRIDE: Tuple[str, str, str] = ("nyz", "nzi", "1")

#: The kernel register holding the NEAR / FAR mirror parity for each axis. Unpacked
#: from :data:`Params`; see the note there for why not ``c``. The far word is a
#: SEPARATE one rather than a negation of the near: ``mirror_parity`` is
#: ``phase * (1 - 2*iyee)`` and :func:`.folded_complex.mirror_parity_coefficients`
#: rounds BOTH through ``numpy.complex64`` on the host and returns them together.
_PARITY_REGISTER: Tuple[str, str, str] = ("mp0", "mp1", "mp2")
_FAR_PARITY_REGISTER: Tuple[str, str, str] = ("fp0", "fp1", "fp2")

#: ``stepping._far_reflect_rows``' answer per axis, as a RUNTIME int rather than a
#: source specialisation: the row is per-axis and integer, so unlike the parity nothing
#: about it can move a bit.
_REFLECT: Tuple[str, str, str] = ("reflect_x", "reflect_y", "reflect_z")

#: The TOP STORED INDEX per axis, and the boolean the lifted curl head already declares
#: for it — reused rather than re-compared against the same extent.
_LAST: Tuple[str, str, str] = ("(nxi - 1)", "(nyi - 1)", "(nzi - 1)")
_LAST_FLAG: Tuple[str, str, str] = ("last_x", "last_y", "last_z")

#: The seven statements ``_apply_constitutive_pml`` runs for one component under
#: COMPLEX storage on the E side, as they appear in the certified complex E body once
#: the targets are renamed ``f`` -> ``h``. Parameterised on the flat index, the source
#: expression, the coefficient pair and a register prefix — the ghost cell takes the
#: identical seven at a different index.
#:
#: NOT A RETYPING: :func:`constitutive_transcription` renders these with the CERTIFIED
#: spellings and compares them statement for statement against the lifted body, and
#: :func:`folded_complex_fused_pair_source` runs that comparison on every build.
_CONSTITUTIVE_STATEMENTS: Tuple[str, ...] = (
    "float2 {prefix}prev = w{target}[{index}];",
    "float2 {prefix}src = {source};",
    "w{target}[{index}] = {prefix}src;",
    "float2 {prefix}acc = h{target}[{index}];",
    "{prefix}acc = {prefix}acc + c_mul_coefficient_left({kp}, {prefix}src);",
    "{prefix}acc = {prefix}acc - c_mul_coefficient_left({km}, {prefix}prev);",
    "h{target}[{index}] = {prefix}acc;",
)

#: How the certified complex E body spells each of the seven for component ``t``.
_CERTIFIED_PREFIX = "@"
_CERTIFIED_NAMES: Dict[str, str] = {
    "@prev": "prev{target}", "@src": "src{target}", "@acc": "a{target}",
}

#: The certified folded complex fill's own near-pass and far-pass lines, as
#: :func:`.folded_complex.folded_mirror_fill_complex_source` emits them for slot ``t``.
_CERTIFIED_FILL_LINE = (
    "    f{slot}[base] = c_mul(c, f{slot}[base + {index} * stride]);")
_CERTIFIED_FAR_FILL_LINE = (
    "    f{slot}[base + last * stride] = "
    "c_mul(c, f{slot}[base + reflect_row * stride]);")

#: The Yee shift a fill pass images: the NEAR pass writes the shift-0 components and
#: the FAR pass the shift-1 ones.
_PASS_SHIFT: Dict[str, int] = {"near": 0, "far": 1}


def _line_starting(body: str, prefix: str, what: str) -> str:
    """The one line of ``body`` starting with ``prefix``, or a named failure.

    The lift's whole safety property. A missing anchor is a certified emitter that has
    changed under this family, and splicing around it would produce a kernel that
    compiles and is quietly not the certified arithmetic; TWO matches would mean the
    anchor no longer identifies a single statement.
    """
    matches = [line for line in body.splitlines() if line.startswith(prefix)]
    if not matches:
        raise AssertionError(
            f"the certified body no longer carries {what} (looked for a line "
            f"starting {prefix!r}); this family LIFTS that line rather than retyping "
            f"it and cannot splice around its absence")
    if len(matches) > 1:
        raise AssertionError(
            f"{prefix!r} matches {len(matches)} lines in the certified body; the lift "
            f"of {what} would take an arbitrary one")
    return matches[0]


def _body_of(source: str, what: str) -> str:
    """One certified shader's body: everything between its signature and its brace."""
    if _BODY_ANCHOR not in source:
        raise AssertionError(
            f"the certified {what} source no longer carries the body anchor; this "
            f"family lifts that body and would otherwise splice a truncated kernel")
    body = source.split(_BODY_ANCHOR, 1)[1]
    if not body.endswith("}\n"):
        raise AssertionError(f"the certified {what} source does not end with '}}'")
    return body[: -len("}\n")]


def certified_curl_head(codes: Sequence[int], phased: Sequence[int], expansion: str,
                        contract: str = shaders.CONTRACT_OFF) -> str:
    """The folded Bloch ``step_D`` curl body DOWN TO the split-field recurrence, verbatim.

    Not a transcription: this is :func:`.folded_complex.folded_bloch_curl_source`'s own
    output with the ``#include``/helpers/signature preamble removed and the cut taken
    at :data:`_RECURRENCE_MARK`. It carries the guard, the index decode, the ghost
    gather, the per-axis Bloch phase block (which SKIPS an unphased axis rather than
    multiplying by ``1+0j`` — the bit-identity of ``k = 0``), the curl grouping, the
    cell-0 ownership mask and the folded top-plane mask.

    ``backward`` is :data:`BACKWARD` and never a parameter: this pair is the D seam,
    and ``step_B``'s forward strides and unconjugated phase are the B twin.
    """
    body = _body_of(
        folded_bloch_curl_source(codes, BACKWARD, phased, expansion, contract),
        "folded complex curl")
    lines = body.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.startswith(_RECURRENCE_MARK):
            head = "".join(lines[:index])
            if not head.strip():
                raise AssertionError("the lifted folded complex curl head is empty")
            return head
    raise AssertionError(
        f"the certified folded complex curl body no longer opens its split-field "
        f"block with {_RECURRENCE_MARK!r}; this family cuts the lift there")


def certified_curl_statements(codes: Sequence[int], phased: Sequence[int],
                              expansion: str,
                              contract: str = shaders.CONTRACT_OFF,
                              ) -> Dict[str, Any]:
    """The split-field recurrence's own lines, pulled out of the certified body.

    Every arithmetic string this family emits below the cut comes from HERE rather than
    from a keyboard: the coefficient loads, the three ``p``/``n``/``v`` statements and
    the two store lines are the certified folded Bloch curl's, matched by an anchor
    that must identify exactly one line each.

    ``stores`` is the ``f0[ii] = v0; f1[ii] = v1; f2[ii] = v2;`` triple SPLIT into its
    three statements, because this family emits them one per component inside that
    component's ownership block rather than as one line.
    """
    body = _body_of(
        folded_bloch_curl_source(codes, BACKWARD, phased, expansion, contract),
        "folded complex curl")
    coefficients = [_line_starting(body, f"    float km_{axis} = ",
                                   f"the {axis} split-field coefficient load")
                    for axis in "xyz"]
    recurrence = []
    for target in range(3):
        recurrence.append((
            _line_starting(body, f"    float2 p{target} = ",
                           f"target {target}'s previous split field"),
            _line_starting(body, f"    float2 n{target} = ",
                           f"target {target}'s split-field recurrence"),
            _line_starting(body, f"    float2 v{target} = ",
                           f"target {target}'s stepped displacement"),
        ))
    aux_store = _line_starting(body, "    u0[ii] = ", "the three fu stores")
    flux_store = _line_starting(body, "    f0[ii] = ", "the three flux stores")
    stores = tuple(f"{part.strip()};" for part in flux_store.strip().split(";")
                   if part.strip())
    if len(stores) != 3:
        raise AssertionError(
            f"the certified folded complex curl no longer stores three targets on one "
            f"line: {flux_store!r}")
    return {"coefficients": tuple(coefficients), "recurrence": tuple(recurrence),
            "aux_store": aux_store, "stores": stores}


def certified_constitutive_body(expansion: str,
                                contract: str = shaders.CONTRACT_OFF) -> str:
    """The complex ``update_E`` constitutive kernel's BODY, lifted and renamed.

    Two edits, and both are POINTER SPELLING rather than arithmetic: the decode
    prologue is dropped (the curl head already emitted it) and the three targets
    ``f0``/``f1``/``f2`` become ``h0``/``h1``/``h2``, because fused, ``f0`` would mean
    D in the curl half and E in the constitutive half in one scope. The inverse
    epsilon is NOT renamed — this family takes ``h`` for the target, so ``e0``/``e1``/
    ``e2`` keep the meaning the certified body gave them.
    """
    source = complex_fields.bloch_constitutive_source(
        CONSTITUTIVE_SIDE, expansion, contract)
    body = _body_of(source, "complex E constitutive")
    prologue = "    int i   = plane / nyi;\n"
    if prologue not in body:
        raise AssertionError(
            "the certified complex constitutive body no longer carries its decode "
            "prologue; the lift would redeclare the curl head's indices")
    body = body.split(prologue, 1)[1]
    for target in range(3):
        old, new = f"f{target}[ii]", f"h{target}[ii]"
        if old not in body:
            raise AssertionError(
                f"the certified complex constitutive body has no {old}")
        body = body.replace(old, new)
    return body


def _constitutive_coefficient_lines(expansion: str,
                                    contract: str = shaders.CONTRACT_OFF
                                    ) -> Tuple[str, ...]:
    """``kp_t``/``km_t`` on the component's OWN axis, lifted from the certified body.

    ``complex_fields``' constitutive template indexes target 0 on ``i``, 1 on ``j`` and
    2 on ``k`` (``stepping.E_CONSTITUTIVE_TERMS``:227) — MEEP's ``dsigw``, NOT the
    dsig/dsigu cycle the curl recurrence uses. On THIS family the NEAR imaged ghost
    reuses these and the FAR one does not, which is the exact inverse of the B twin and
    is what :func:`_ghost_coefficient_lines` carries.
    """
    body = certified_constitutive_body(expansion, contract)
    return tuple(_line_starting(body, f"    float kp_{target} = ",
                                f"target {target}'s constitutive coefficient pair")
                 for target in range(3))


def _render_constitutive(target: int, index: str, source: str, kp: str, km: str,
                         prefix: str) -> Tuple[str, ...]:
    """The seven constitutive statements for one component at one cell, unindented."""
    return tuple(line.format(target=target, index=index, source=source,
                             kp=kp, km=km, prefix=prefix)
                 for line in _CONSTITUTIVE_STATEMENTS)


def _touches(statement: str, target: int) -> bool:
    """Does one certified constitutive statement belong to component ``target``?"""
    if statement.startswith("float kp_"):
        return False
    marks = (f"prev{target}", f"src{target}", f"a{target} ", f"w{target}[",
             f"h{target}[")
    return any(mark in statement for mark in marks)


def constitutive_transcription(expansion: str,
                               contract: str = shaders.CONTRACT_OFF
                               ) -> Dict[str, Any]:
    """Is :data:`_CONSTITUTIVE_STATEMENTS` the certified complex E body, statement for statement?

    THE MEASUREMENT THAT MAKES THE PARAMETERISATION HONEST. This family re-emits the
    certified constitutive arithmetic once per owned cell and once per imaged ghost, so
    it cannot splice that body wholesale the way :mod:`.complex_fused_electric_pair`
    does. What it can do is render its own template with the CERTIFIED spellings and
    require the result to be exactly the lifted body's statements.

    ``identical`` false is a build-time failure, not a diagnostic:
    :func:`folded_complex_fused_pair_source` calls this and raises.
    """
    body = certified_constitutive_body(expansion, contract)
    statements = [line.strip() for line in body.splitlines()
                  if line.strip() and not line.strip().startswith("//")]
    certified: List[List[str]] = []
    emitted: List[List[str]] = []
    for target in range(3):
        rendered = list(_render_constitutive(
            target, "ii", f"c_mul_field_left(g{target}[ii], e{target}[ii])",
            f"kp_{target}", f"km_{target}", _CERTIFIED_PREFIX))
        for token, spelling in _CERTIFIED_NAMES.items():
            rendered = [line.replace(token, spelling.format(target=target))
                        for line in rendered]
        emitted.append(rendered)
        certified.append([line for line in statements if _touches(line, target)])
    return {
        "emitted": [list(rows) for rows in emitted],
        "certified": [list(rows) for rows in certified],
        "identical": all(emitted[t] == certified[t] for t in range(3)),
    }


def parity_chain(subset: Sequence[int], carries_far: bool, far: Sequence[int]
                 ) -> Tuple[Tuple[int, str], ...]:
    """The ORDERED ``(axis, pass)`` applications one ghost's value carries.

    THE ORDER IS THE DRIVER'S AND IS TRANSCRIBED, NEVER CHOSEN. ``driver.step`` runs
    ``fill_symmetry_bc_D`` (:3300) to completion before ``fill_folded_far_ghosts_D``
    (:3302), and each pass loops its axes in ASCENDING order
    (``stepping._fill_symmetry_ghost_cells``:1441-1447,
    ``_fill_folded_far_ghosts``:1518). Each pass reads the plane the previous one
    wrote, so back-substituting a corner gives the near applications first, ascending,
    then the far one.

    THE D SIDE'S CHAIN IS THE B SIDE'S TURNED ROUND: here up to TWO near applications
    and at most ONE far, where the B twin has one near and up to two far. That falls
    out of ``fields.IYEE_SHIFTS`` and is not a choice: :func:`.folded_fused_pair.near_fill_axes`
    returns the axes that are NOT the component's own and
    :func:`.folded_fused_pair.far_fill_axes` returns its own axis alone.

    THE WALL CLEAR IS NOT IN THIS LIST, deliberately: it is not a parity, and it sits
    BETWEEN the two passes (driver.py:3301). :func:`_carry_blocks` splits the emitted
    chain at that point.
    """
    chain: List[Tuple[int, str]] = []
    previous = -1
    for axis in subset:
        axis = int(axis)
        if axis <= previous:
            raise AssertionError(
                f"the near axis subset {tuple(subset)!r} is not strictly ascending; "
                f"stepping._fill_symmetry_ghost_cells applies its axes in ascending "
                f"order (:1441) and this chain transcribes that order")
        previous = axis
        chain.append((axis, "near"))
    if carries_far:
        if not far:
            raise AssertionError(
                "a destination that carries the far fill was enumerated for a "
                "component with no far axis; carried_destinations and far_fill_axes "
                "have drifted")
        chain.append((int(far[0]), "far"))
    return tuple(chain)


def _certified_fill_line(target: int, pass_name: str) -> str:
    """The certified folded complex fill's own line for one component and one pass."""
    if pass_name == "near":
        return _CERTIFIED_FILL_LINE.format(slot=target, index=NEAR_SOURCE_INDEX)
    return _CERTIFIED_FAR_FILL_LINE.format(slot=target)


def _certified_fill_operand(target: int, pass_name: str) -> Tuple[str, str]:
    """That line's ``(destination, source operand)`` expressions, as it spells them."""
    if pass_name == "near":
        return (f"f{target}[base]",
                f"f{target}[base + {NEAR_SOURCE_INDEX} * stride]")
    return (f"f{target}[base + last * stride]",
            f"f{target}[base + reflect_row * stride]")


def _chain_step(target: int, tag: str, axis: int, pass_name: str,
                first: bool) -> str:
    """ONE ``c_mul`` of the parity chain — THE ONE PLACE it is spelled.

    The emitter and both transcription checks call this, so a line that stopped being
    the certified fill's cannot be one in the kernel and another in the measurement.

    EVERY APPLICATION LANDS IN A REGISTER HERE, where the B twin's last one writes
    straight to the ghost cell. The reason is the wall clear: on the D side it sits
    BETWEEN the two fills (driver.py:3301) and its destination set — the shift-0 axes
    of ``_ZERO_METAL_ROWS`` — is exactly the near fill's, so a near-carrying ghost
    takes clear lines after the near multiplies and before any far one. A value
    computed into a register and then stored is the same word, and the split is what
    lets the clear be inserted at the driver's own point rather than moved.
    """
    coefficient = (_PARITY_REGISTER[axis] if pass_name == "near"
                   else _FAR_PARITY_REGISTER[axis])
    source = f"v{target}" if first else f"{tag}_v"
    declaration = "float2 " if first else ""
    return f"{declaration}{tag}_v = c_mul({coefficient}, {source});"


def _fill_transcription(axis: int, pass_name: str, expansion: str,
                        contract: str = shaders.CONTRACT_OFF) -> Dict[str, Any]:
    """Is this family's ghost write the CERTIFIED folded complex D fill's own line?

    THE SECOND MEASUREMENT, covering the one piece of arithmetic that is neither curl
    nor constitutive. :func:`.folded_complex.folded_mirror_fill_complex_source` emits,
    on the ``"D"`` ghost-fill family::

        near:  ``f{slot}[base] = c_mul(c, f{slot}[base + 2 * stride]);``
        far:   ``f{slot}[base + last * stride] =
                     c_mul(c, f{slot}[base + reflect_row * stride]);``

    — the coefficient on the LEFT, the whole complex product, no shortcut for
    ``phase == +1``. This family writes the same call at a different flat index with
    the coefficient register renamed, because the lifted curl head already owns the
    name ``c``, and with the DESTINATION a register rather than the ghost cell (see
    :func:`_chain_step`). So the comparison is made under a STATED rename, and
    ``identical`` false is a build-time failure.
    """
    from .folded_complex import GHOST_FILL_FAMILIES  # noqa: PLC0415
    from .folded_complex import folded_mirror_fill_complex_source  # noqa: PLC0415

    if pass_name not in _PASS_SHIFT:
        raise ValueError(f"pass_name must be near or far, got {pass_name!r}")
    axis = int(axis)
    names = tuple(GHOST_FILL_FAMILIES["D"]["targets"])
    shifts = tuple(TARGET_IYEE[name][axis] for name in names)
    source = folded_mirror_fill_complex_source(axis, pass_name, shifts, expansion,
                                               contract)
    lines = [line for line in source.splitlines()
             if line.strip().startswith("f") and "c_mul(c," in line]
    coefficient = (_PARITY_REGISTER[axis] if pass_name == "near"
                   else _FAR_PARITY_REGISTER[axis])
    rows: List[Dict[str, Any]] = []
    for target, shift in enumerate(shifts):
        if int(shift) != _PASS_SHIFT[pass_name]:
            continue
        certified = _certified_fill_line(target, pass_name)
        if certified not in lines:
            raise AssertionError(
                f"the certified folded complex {pass_name} fill no longer emits "
                f"{certified!r} on axis {axis}; this family carries that line inline "
                f"and cannot rename around its absence. emitted={lines!r}")
        destination, operand = _certified_fill_operand(target, pass_name)
        marker = f"g{target}_t"
        base = certified.replace("c_mul(c,", f"c_mul({coefficient},")
        for role, first in (("first", True), ("continued", False)):
            renamed = base.replace(
                operand, f"v{target}" if first else f"{marker}_v")
            renamed = renamed.replace(
                destination, ("float2 " if first else "") + f"{marker}_v")
            rows.append({"target": target, "role": role,
                         "certified": certified.strip(),
                         "renamed": renamed.strip(),
                         "emitted": _chain_step(target, marker, axis, pass_name,
                                                first)})
    if not rows:
        raise AssertionError(
            f"axis {axis} images no D component on the {pass_name} pass; "
            f"_fill_transcription would measure nothing and an empty comparison "
            f"reports as a pass")
    return {"axis": axis, "pass": pass_name, "rows": rows,
            "identical": all(row["renamed"] == row["emitted"] for row in rows)}


def near_fill_transcription(axis: int, expansion: str,
                            contract: str = shaders.CONTRACT_OFF) -> Dict[str, Any]:
    """:func:`_fill_transcription` for the NEAR pass — ``cell 0 = parity (x) cell 2``."""
    return _fill_transcription(axis, "near", expansion, contract)


def far_fill_transcription(axis: int, expansion: str,
                           contract: str = shaders.CONTRACT_OFF) -> Dict[str, Any]:
    """:func:`_fill_transcription` for the FAR pass — ``last = parity (x) reflect row``."""
    return _fill_transcription(axis, "far", expansion, contract)


def zero_metal_lines(target: int, zero_metal: Sequence[bool], name: str,
                     indent: str = "    ",
                     zero: str = templates.COMPLEX_ZERO) -> List[str]:
    """``stepping.zero_metal_D`` for ONE target, on one named register.

    The rows are :data:`.fused_dispersive_pair._ZERO_METAL_ROWS` — the D OFF-DIAGONAL
    (Dy/Dz on an x wall, Dx/Dz on y, Dx/Dy on z) — IMPORTED rather than re-derived,
    for the reason :func:`.fused_electric_pair.zero_metal_mask` gives: that table is
    already pinned against the array path, and the magnetic twin's is its complement.

    The zero is a COMPLEX zero — BOTH planes — because the array path assigns one
    (stepping.py:1943, :1949); a real ``0.0f`` here would not even compile, but the
    plane-wise question it stands for is live. The select spelling is
    :func:`.templates.ownership_mask`'s, ``flag ? zero : v``, the form measured to
    deliver an exact ``+0.0`` on this platform.
    """
    return [f"{indent}{name} = {flag} ? {zero} : {name};"
            for row_target, axis, flag in _ZERO_METAL_ROWS
            if row_target == target and bool(zero_metal[axis])]


def _ghost_coefficient_lines(target: int, axis: int, tag: str,
                             indent: str) -> List[str]:
    """The FAR destination's constitutive coefficient pair, read at stored ``last``.

    THE INVERSE OF THE B TWIN'S ASYMMETRY, and it falls out of the same two tables.
    ``complex_fields``' constitutive template indexes target ``t`` on axis ``t``
    (``stepping.E_CONSTITUTIVE_TERMS``:227), and on the D side
    :func:`.folded_fused_pair.far_fill_axes` returns axis ``t`` ALONE — so the FAR
    destination sits at a DIFFERENT coordinate on the indexed axis from the thread
    that owns it. Reusing the source thread's ``kp_t``/``km_t`` would apply the
    absorber profile of the reflect row to the top plane: smooth, converged and wrong
    inside the PML, invisible outside it.

    AND UNLIKE THE B TWIN'S NEAR CASE THIS IS NORMALLY OBSERVABLE: a folded axis
    carries an absorber on its HIGH face only (``stepping._require_consistent_pml``),
    so the mirror plane at stored 0 is outside the layer while the top plane the far
    ghost lands on is inside it.

    THE NEAR GHOST NEEDS NO SUCH LINE: :func:`.folded_fused_pair.near_fill_axes`
    returns only axes ``a != t``, so a near destination sits at the SAME coordinate on
    the indexed axis as its owner and takes that thread's pair unchanged. A carry that
    reloaded there would be reloading the identical word. The coefficients are float32
    in BOTH storage modes (stepping.py:41-50), so these two loads are the real fold's
    two loads exactly.
    """
    if axis != target:
        raise AssertionError(
            f"component {target} images its FAR ghost on axis {axis}; for the D "
            f"family the far fill only ever touches a component's OWN axis "
            f"(fields.IYEE_SHIFTS)")
    name = _COORDINATE[axis]
    return [
        f"{indent}// The DESTINATION's own coefficient entry: stored index "
        f"{_LAST[axis]} on {name}, NOT",
        f"{indent}// the source thread's at {_REFLECT[axis]} (complex_fields' "
        f"constitutive template indexes target {target} on {name}).",
        f"{indent}float {tag}_kp = kp{target}[{_LAST[axis]}], "
        f"{tag}_km = km{target}[{_LAST[axis]}];",
    ]


def _constitutive_block(target: int, index: str, source: str, kp: str, km: str,
                        prefix: str, indent: str) -> List[str]:
    """``_apply_constitutive_pml`` for one component at one cell, indented."""
    return [indent + line for line in
            _render_constitutive(target, index, source, kp, km, prefix)]


def _carry_blocks(target: int, near: Sequence[int], far: Sequence[int],
                  zero_metal: Sequence[bool], indent: str) -> List[str]:
    """Every imaged ghost this thread owns, then ``update_E`` at each of them.

    ONE BLOCK PER DESTINATION IN :func:`.folded_fused_pair.carried_destinations` — the
    D-side enumeration, subsets of the NEAR axes crossed with the single far one.
    Each is guarded on the flags that say this thread is the source of that
    destination — stored ``NEAR_SOURCE_INDEX`` on each near axis it images, the runtime
    reflect row on the far axis — and writes the parity CHAIN applied to ``v``,
    followed by the certified constitutive at that index.

    THE CHAIN IS EMITTED ONE ``c_mul`` PER PASS, IN :func:`parity_chain`'S ORDER, and
    it is not folded into a single word. That is inherited from the B twin as a
    MEASUREMENT (``results/complex_parity_chain_order_2026-08-21``) rather than a
    preference.

    THE WALL CLEAR IS EMITTED INSIDE THE CHAIN, which is the whole structural
    difference from the B twin and follows the driver: ``fill_symmetry_bc_D`` (:3300),
    then ``zero_metal_D`` (:3301), then ``fill_folded_far_ghosts_D`` (:3302). A
    destination with near axes therefore takes the clear AFTER its near multiplies;
    one written only by the far fill takes NO clear line, because the far pass runs
    after the clear and its destination moves along the component's OWN axis, which
    ``_ZERO_METAL_ROWS`` never touches — its clear status is its source thread's,
    inherited by reading ``v{target}`` after those lines.

    THE FLAG SET THE CLEAR IS EVALUATED ON IS THE OWNER THREAD'S ``at_x``/``at_y``/
    ``at_z``, and that is exact rather than approximate: a near destination differs
    from its owner only on FOLDED axes, and ``stepping._zero_metal`` skips a folded
    axis (stepping.py:2282-2284), so on every axis that could clear it the destination
    and the owner share a coordinate. The disjointness is ASSERTED below rather than
    trusted.
    """
    walls = tuple(axis for row_target, axis, _ in _ZERO_METAL_ROWS
                  if row_target == target and bool(zero_metal[axis]))
    if set(walls) & {int(axis) for axis in near}:
        raise AssertionError(
            f"component {target} images a NEAR ghost on folded axes {tuple(near)} "
            f"that zero_metal_D also clears on {walls}; stepping._zero_metal skips a "
            f"folded axis (stepping.py:2282-2284), so the two tables have drifted")
    overlap = sorted(set(walls) & {int(axis) for axis in far})
    if overlap:
        raise AssertionError(
            f"component {target} images a FAR ghost along axes {tuple(far)} that "
            f"zero_metal_D also clears on {overlap}; _ZERO_METAL_ROWS is the D "
            f"off-diagonal and far_fill_axes returns the component's OWN axis, so the "
            f"ghost would no longer inherit its source thread's clear")
    inner = indent + "    "
    lines: List[str] = []
    for subset, carries_far in carried_destinations(near, far):
        tag = (f"g{target}_"
               + "".join(_COORDINATE[axis] for axis in subset)
               + ("f" if carries_far else ""))
        chain = parity_chain(subset, carries_far, far)
        flags: List[str] = []
        terms: List[str] = []
        described: List[str] = []
        for axis, pass_name in chain:
            if pass_name == "near":
                flags.append(f"{_COORDINATE[axis]} == {NEAR_SOURCE_INDEX}")
                terms.append(f"- {NEAR_SOURCE_INDEX} * {_STRIDE[axis]}")
                described.append(
                    f"the near fill on {_COORDINATE[axis]}: {_COORDINATE[axis]} = 0 "
                    f"imaged from {NEAR_SOURCE_INDEX}, parity word "
                    f"{_PARITY_REGISTER[axis]} "
                    f"(stepping._fill_symmetry_ghost_cells:1441, "
                    f"_write_mirror_ghost:1449)")
            else:
                flags.append(f"{_COORDINATE[axis]} == {_REFLECT[axis]}")
                terms.append(f"+ ({_LAST[axis]} - {_REFLECT[axis]}) * {_STRIDE[axis]}")
                described.append(
                    f"the far fill on {_COORDINATE[axis]}: {_COORDINATE[axis]} = "
                    f"{_LAST[axis]} imaged from {_REFLECT[axis]}, parity word "
                    f"{_FAR_PARITY_REGISTER[axis]} "
                    f"(stepping._fill_folded_far_ghosts:1516)")
        cleared = (zero_metal_lines(target, zero_metal, f"{tag}_v", inner)
                   if subset else [])
        lines.append(f"{indent}if ({' && '.join(flags)}) {{")
        lines.extend(f"{inner}// {text}" for text in described)
        lines.append(
            f"{inner}// the parity CHAIN, in the driver's own pass order "
            f"(near ascending, then far):")
        lines.append(
            f"{inner}// "
            + " then ".join(f"{pass_name} on {_COORDINATE[axis]}"
                            for axis, pass_name in chain)
            + (", with zero_metal_D between them (driver.py:3301)." if cleared
               and carries_far
               else ", with zero_metal_D after them (driver.py:3301)." if cleared
               else ", and no wall clears this target.")
            + " NOT folded into one word:")
        lines.append(
            f"{inner}// order and grouping both move bytes under complex storage "
            f"(results/complex_parity_chain_order_2026-08-21).")
        lines.append(f"{inner}int {tag}_i = ii {' '.join(terms)};")
        emitted = 0
        for axis, pass_name in chain:
            if pass_name == "far" and cleared:
                # The clear runs BETWEEN the two fills; emit it before the far
                # parity so the far image carries the CLEARED value, which is the
                # driver's own order.
                lines.extend(cleared)
                cleared = []
            lines.append(inner + _chain_step(target, tag, axis, pass_name,
                                             first=emitted == 0))
            emitted += 1
        lines.extend(cleared)
        lines.append(f"{inner}f{target}[{tag}_i] = {tag}_v;")
        if carries_far:
            lines.extend(
                _ghost_coefficient_lines(target, int(far[0]), tag, inner))
            kp, km = f"{tag}_kp", f"{tag}_km"
        else:
            # The near destination shares the indexed axis with its source thread, so
            # the certified pair this thread already loaded IS the destination's.
            kp, km = f"kp_{target}", f"km_{target}"
        lines.extend(_constitutive_block(
            target, f"{tag}_i",
            f"c_mul_field_left({tag}_v, e{target}[{tag}_i])",
            kp, km, f"{tag}_", inner))
        lines.append(f"{indent}}}")
    return lines


def folded_complex_fused_pair_source(
        codes: Sequence[int], phased: Sequence[int], zero_metal: Sequence[bool],
        expansion: str, contract: str = shaders.CONTRACT_OFF) -> str:
    """One specialised fused source: (folded quadruple, Bloch flags, walls, arm, mode).

    ``codes`` is the per-axis PERIODIC / METALLIC / MIRROR_METALLIC / MIRROR_PERIODIC
    quadruple :func:`.symmetry.folded_axis_kinds` resolves — never a hand-built triple,
    because the MIRROR_METALLIC / MIRROR_PERIODIC split is this family's single point of
    failure and getting it backwards on one axis is a plane of wrong values rather than
    a crash.

    ``phased`` is the per-axis Bloch flag :func:`.complex_fields.phase_arguments`
    resolves. THE PARITY IS NOT HERE: like the B twin and unlike the real fold, it is a
    runtime word, so it does not specialise the source and two grids differing only in
    a mirror parity compile to the SAME kernel.
    """
    codes = tuple(int(code) for code in codes)
    if len(codes) != 3:
        raise ValueError(f"codes must be a per-axis triple, got {codes!r}")
    if not any(code in MIRROR_CODES for code in codes):
        raise ValueError(
            "no axis is folded: this product exists to carry the mirror fills inside "
            "the D seam, and an unfolded complex grid belongs to "
            "complex_fused_electric_pair")
    phased = tuple(int(flag) for flag in phased)
    if len(phased) != 3:
        raise ValueError(f"phased must be a per-axis triple, got {phased!r}")
    zero_metal = tuple(bool(value) for value in zero_metal)
    if len(zero_metal) != 3:
        raise ValueError(f"zero_metal must be a per-axis triple, got {zero_metal!r}")
    # THE DISJOINTNESS THE CARRY RESTS ON, asserted rather than trusted:
    # `zero_metal_axes` is `is_metallic and not is_mirrored` (stepping.py:2282-2284),
    # so a folded axis can never be a walled one.
    for axis, code in enumerate(codes):
        if code in MIRROR_CODES and zero_metal[axis]:
            raise ValueError(
                f"axis {axis} is reported both folded and walled; "
                f"stepping._zero_metal excludes a folded axis by construction and "
                f"this kernel's ghost carry relies on the two sets being disjoint")

    transcription = constitutive_transcription(expansion, contract)
    if not transcription["identical"]:
        raise AssertionError(
            "this family's parameterised constitutive statements no longer reproduce "
            "complex_fields.bloch_constitutive_source('E'); the fused arithmetic "
            "would silently stop being the certified arithmetic. emitted="
            f"{transcription['emitted']!r} certified={transcription['certified']!r}")

    statements = certified_curl_statements(codes, phased, expansion, contract)
    body: List[str] = [
        certified_curl_head(codes, phased, expansion, contract).rstrip("\n")]
    periodic = tuple(axis for axis, code in enumerate(codes)
                     if code == CODE_MIRROR_PERIODIC)
    body.extend([
        "",
        ("    // The three top-plane flags the lifted head declares are read below by "
         "the far"
         if periodic else
         "    // The three top-plane flags the lifted head declares are unread here:"),
        ("    // carry's ownership guard, and by folded_top_plane_mask's own lines "
         "above."
         if periodic else
         "    // no folded PERIODIC axis, so folded_top_plane_mask emits no line and "
         "the"),
        ("    // A (void) keeps the unfolded axes' flags from warning."
         if periodic else "    // block above is a comment."),
        "    (void)last_x; (void)last_y; (void)last_z;",
        "",
        "    // --- split-field recurrence (stepping._apply_pml_update:1905) ------",
    ])
    body.extend(statements["coefficients"])
    body.extend([
        "",
        "    // --- the constitutive coefficient, on the component's OWN axis -----",
        "    // Read HERE for the owned cell and reused at the NEAR ghost, which",
        "    // differs from its source only on axes that are NOT the component's",
        "    // own. The FAR ghost does NOT share it: it moves along exactly the",
        "    // axis this is indexed on (_ghost_coefficient_lines).",
    ])
    body.extend(_constitutive_coefficient_lines(expansion, contract))
    body.append("")

    # The three recurrences and the three `fu` stores, in the certified body's own
    # order. `u` IS WRITTEN AT EVERY CELL, ghost cells included: `step_D` updates `fu`
    # everywhere and the fills touch D only (stepping.py:1497-1498 writes `field`,
    # never `fu_field`).
    for target in range(3):
        previous, recurrence, _ = statements["recurrence"][target]
        body.extend([previous, recurrence])
    body.append("")
    body.append(statements["aux_store"])

    for target in range(3):
        near = near_fill_axes(codes, target)
        if len(near) > 2:
            raise AssertionError(
                f"component {target} is a near-fill destination on {near}; for the D "
                f"family that set is at most two axes (the two that are NOT the "
                f"component's own, fields.IYEE_SHIFTS)")
        far = far_fill_axes(codes, target)
        if len(far) > 1 or set(far) & set(near):
            raise AssertionError(
                f"component {target} is a far-fill destination on {far} against near "
                f"axes {near}; for the D family the two sets are COMPLEMENTARY, so "
                f"far holds at most ONE axis and never a near one")
        for axis in near:
            fill = near_fill_transcription(axis, expansion, contract)
            if not fill["identical"]:
                raise AssertionError(
                    "this family's ghost write is no longer the certified folded "
                    "complex NEAR fill's own line under the stated rename: "
                    f"{fill['rows']!r}")
        for axis in far:
            fill = far_fill_transcription(axis, expansion, contract)
            if not fill["identical"]:
                raise AssertionError(
                    "this family's ghost write is no longer the certified folded "
                    "complex FAR fill's own line under the stated rename: "
                    f"{fill['rows']!r}")
        _, _, displacement = statements["recurrence"][target]
        body.append("")
        # A cell any fill writes is OWNED BY ITS SOURCE THREAD. This thread stops after
        # fu there: the array path's fill overwrites the displacement it would store,
        # and forming v would read f at a word another thread writes.
        owned: List[str] = [f"{_COORDINATE[axis]} == 0" for axis in near]
        owned.extend(_LAST_FLAG[axis] for axis in far)
        if owned:
            body.extend([
                f"    // component {target}: the fills image "
                + ", ".join(
                    [f"stored cell 0 on {_COORDINATE[axis]} (near)" for axis in near]
                    + [f"the top plane on {_COORDINATE[axis]} (far)"
                       for axis in far]) + ",",
                "    // so those cells are OWNED BY THEIR SOURCE THREADS. This one "
                "stops after fu:",
                "    // the fill overwrites the displacement it would store, and "
                "forming v here",
                "    // would read a word another thread writes.",
                f"    if (!({' || '.join(owned)})) {{",
            ])
            indent = "        "
        else:
            body.append(f"    // component {target}: no folded axis images a cell "
                        f"of this component")
            indent = "    "
        body.append(indent + displacement.strip())
        # `zero_metal_D` on the cell this thread owns, then the store, then the
        # constitutive read of the SAME register -- the seam this family closes.
        cleared = zero_metal_lines(target, zero_metal, f"v{target}", indent)
        body.extend(cleared or [f"{indent}// no walled axis clears this target"])
        body.append(indent + statements["stores"][target])
        body.extend(_constitutive_block(
            target, "ii", f"c_mul_field_left(v{target}, e{target}[ii])",
            f"kp_{target}", f"km_{target}", f"o{target}_", indent))
        if owned:
            body.append("")
            body.extend(_carry_blocks(target, near, far, zero_metal, indent))
            body.append("    }")

    return templates.substitute(_TEMPLATE, {
        "__CONTRACT__": shaders.contraction_pragma(contract),
        "__HELPERS__": templates.complex_helpers(expansion),
        "__BODY__": "\n".join(body),
    })


def compile_folded_complex_fused_pair(
        codes: Sequence[int], phased: Sequence[int], zero_metal: Sequence[bool],
        expansion: str, contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised entry point for one (codes, Bloch flags, walls, arm, mode)."""
    return compile_source(folded_complex_fused_pair_source(
        codes, phased, zero_metal, expansion,
        contract)).folded_complex_fused_pair_step


def refuted_separate_scalar_source(contract: str = shaders.CONTRACT_OFF) -> str:
    """The same 30 pointers with the fifteen scalars bound SEPARATELY.

    :data:`SEPARATE_SCALAR_BINDINGS` bindings, far over the ceiling, and the gate
    requires the compile failure -- which is what turns the packing from a preference
    into a measurement.
    """
    volumes = [f"    device float2* q{index} [[buffer({index})]]"
               for index in range(15)]
    reals = [f"    device const float* r{index} [[buffer({index})]]"
             for index in range(15, 30)]
    slot = 30
    scalars: List[str] = []
    for name in ("nx", "ny", "nz", "n_elem"):
        scalars.append(f"    constant uint& {name} [[buffer({slot})]]")
        slot += 1
    scalars.append(f"    constant float& dtdx [[buffer({slot})]]")
    slot += 1
    for group in ("p", "m", "d"):
        for axis in range(3):
            scalars.append(
                f"    constant float2& {group}{axis} [[buffer({slot})]]")
            slot += 1
    for name in ("rx", "ry", "rz"):
        scalars.append(f"    constant int& {name} [[buffer({slot})]]")
        slot += 1
    if slot != SEPARATE_SCALAR_BINDINGS:  # pragma: no cover - a design invariant
        raise AssertionError((slot, SEPARATE_SCALAR_BINDINGS))
    arguments = ",\n".join(volumes + reals + scalars)
    return ("#include <metal_stdlib>\nusing namespace metal;\n"
            + shaders.contraction_pragma(contract) + "\n"
            "kernel void separate_scalar_folded_complex_pair(\n" + arguments +
            ",\n    uint idx [[thread_position_in_grid]])\n"
            "{\n    q0[idx] = float2(dtdx * r15[idx] + float(nx + rx), p0.x);\n}\n")


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def metal_folded_complex_fused_pair_coverage(
        fields: Any, pml: Any, sources: Any = None, residency: Any = None,
        probe: Any = None) -> Coverage:
    """May ONE dispatch span folded complex ``step_D`` -> fills -> wall -> ``update_E``?

    A conjunction of the two halves' OWN certified predicates plus the seam clauses.
    Nothing is weakened: a configuration either half refuses is refused here with that
    half's reasons, prefixed so a reader can tell which side said it. The two halves
    are the ``folded complex`` curl arm's routing verdict and its constitutive arm's
    predicate, which is what keeps this product disjoint from
    :mod:`.folded_fused_pair` (real storage) and from
    :mod:`.complex_fused_electric_pair` (no fold) in both directions.

    ``probe`` is the expansion artifact both halves bind through. It is threaded, not
    re-read: a second read would be a second answer to one question.
    """
    reasons: List[str] = []

    curl = folded_complex_composition_curl_coverage(
        fields, pml, CURL_SUB_STEP, residency, probe)
    if not curl.covered:
        reasons.extend(f"folded complex curl half: {reason}"
                       for reason in curl.reasons)
    electric = folded_complex_constitutive_coverage(
        fields, pml, CONSTITUTIVE_SIDE, residency, probe)
    if not electric.covered:
        reasons.extend(f"folded complex constitutive half: {reason}"
                       for reason in electric.reasons)

    # THE SOURCE SEAM. An ELECTRIC source is injected BETWEEN the two halves
    # (driver.py:3294-3299), so a fused pair would consume a pre-injection D unless
    # the deposit repair brackets the launch -- which for this family it does, and
    # which is the only reason the cell is worth anything at all.
    # IGNORANCE IS NEVER AN EMPTY SET: `Fields` does not hold the source list, so a
    # predicate that inferred "no sources" from not being told would be the
    # over-covering refusal this clause exists to prevent.
    reasons.extend(_deposit_repair.seam_source_reasons(
        fields, sources, 'D',
        undeclared=(
            "the source set was not declared: this predicate cannot infer an empty "
            "electric source seam from Fields"
        ),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) is electric: the driver "
            f"injects it BETWEEN step_D and update_E (driver.py:3294-3299), which is "
            f"work inside the seam this kernel closes"),
        carries_repair=CARRIES_DEPOSIT_REPAIR))

    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False,
                        tuple(dict.fromkeys(reasons + ["fields carries no grid"])))

    codes, code_reasons = folded_axis_kinds(
        grid, pml if (pml is not None and getattr(pml, "is_active", False)) else None)
    reasons.extend(code_reasons)

    # THE FAR FILL IS CARRIED, not refused; these are the conditions its OWNERSHIP
    # MOVE rests on, and the mirror parity must be readable on every folded axis.
    reasons.extend(_far_carry_reasons(grid, codes))
    reasons.extend(_mirror_phase_reasons(grid, codes))

    # THE WALL CLEAR IS CARRIED INLINE, so the grid must be able to answer which axes
    # are walled. A grid that cannot would silently be treated as unwalled, which is a
    # plane of wrong values rather than a crash.
    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            reasons.append(
                f"grid does not expose {name}; zero_metal_D cannot be carried inline")

    shape = tuple(getattr(grid, "shape", ()))
    if codes is not None and len(shape) == 3:
        # THE NEAR FILL IMAGES STORED CELL 2, and the source thread must exist. The
        # curl half's predicate already refuses a folded axis storing too few cells;
        # restated because THIS family writes the destination from a DIFFERENT thread
        # and a missing source plane is an out-of-range write, not a soft error.
        for axis, code in enumerate(codes):
            if int(code) in MIRROR_CODES and int(shape[axis]) <= NEAR_SOURCE_INDEX:
                reasons.append(
                    f"axis {axis} is folded with {int(shape[axis])} stored cells; the "
                    f"near fill images stored cell {NEAR_SOURCE_INDEX} and this "
                    f"kernel images it from that cell's own thread")

    # A polarization would make the constitutive source (D - sum P) rather than D. The
    # E-side predicate already refuses it; restated because this family's kernel bakes
    # the plain product.
    if tuple(getattr(fields, "polarizations", ()) or ()):
        reasons.append("a susceptibility is registered: this kernel bakes the plain "
                       "constitutive product, whose source is D and not (D - sum P)")
    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class MetalFoldedComplexFusedPairPlan(KernelPlan):
    """ONE dispatch that performs five driver passes, complex D never leaving a register.

    ``launches_per_run`` is the base's 1 and is left there deliberately: this plan
    unpacks one argument tuple and calls one function, which is the base's whole
    contract, so the whole-step arbiter's per-cycle launch assertion needs no special
    case for it.
    """

    __slots__ = ("residency", "volumes", "codes", "phased", "phase_values",
                 "near_parities", "far_parities", "zero_metal", "shape", "n_elem",
                 "dtdx", "reflect", "expansion", "params")

    family = "folded complex fused PML D-curl/mirror-fill/stored-E pair"

    replaces_sub_steps = REPLACES

    REPR_FIELDS = ("shape", "codes", "phased", "zero_metal", "reflect", "expansion")

    #: This plan launches a kernel; the composer's planned/null split reads it.
    performs_device_work = True

    def __init__(self, residency: Residency, volumes: Sequence[str],
                 codes: Sequence[int], phased: Sequence[int],
                 phase_values: Sequence[Tuple[float, float]],
                 near_parities: Sequence[Tuple[float, float]],
                 far_parities: Sequence[Tuple[float, float]],
                 zero_metal: Sequence[bool], shape: Sequence[int], dtdx: float,
                 reflect: Sequence[Optional[int]], expansion: str, params: Any,
                 pointers: Sequence[Any], functions: Mapping[str, Any]) -> None:
        self.residency = residency
        self.volumes = tuple(dict.fromkeys(volumes))
        self.codes = tuple(int(code) for code in codes)
        self.phased = tuple(int(flag) for flag in phased)
        self.phase_values = tuple((float(re), float(im)) for re, im in phase_values)
        self.near_parities = tuple((float(re), float(im))
                                   for re, im in near_parities)
        self.far_parities = tuple((float(re), float(im)) for re, im in far_parities)
        self.zero_metal = tuple(bool(value) for value in zero_metal)
        self.shape = tuple(int(n) for n in shape)
        # The COMPLEX CELL count, which is also what the dispatch is sized from.
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.dtdx = float(dtdx)
        self.reflect = tuple(None if row is None else int(row) for row in reflect)
        self.expansion = str(expansion)
        self.params = params
        super().__init__(dict(functions), tuple(pointers) + (params,))

    @property
    def runs(self) -> int:
        return self.launches


def params_record_dtype() -> Any:
    """The host record's dtype — THE NINE ``float2`` MEMBERS FIRST, itemsize 104.

    ONE HOME FOR THE LAYOUT, because getting it wrong is a silent wrong answer rather
    than a crash: with the scalars first Metal's 8-byte ``float2`` alignment puts every
    complex word one slot early, and the result reads as a plausible complex number.
    The offsets are stated EXPLICITLY even though the parities-first order makes them
    the natural ones.
    """
    import numpy as np  # noqa: PLC0415

    names = ["px", "py", "pz", "m0", "m1", "m2", "d0", "d1", "d2"]
    formats: List[Any] = [("<f4", 2)] * 9
    offsets = [8 * index for index in range(9)]
    names += ["nx", "ny", "nz", "n_elem", "dtdx", "rx", "ry", "rz"]
    formats += ["<u4", "<u4", "<u4", "<u4", "<f4", "<i4", "<i4", "<i4"]
    offsets += [72, 76, 80, 84, 88, 92, 96, 100]
    return np.dtype({"names": names, "formats": formats, "offsets": offsets,
                     "itemsize": PARAMS_ITEMSIZE})


def _params_tensor(shape: Sequence[int], dtdx: float,
                   phase_values: Sequence[Tuple[float, float]],
                   near_parities: Sequence[Tuple[float, float]],
                   far_parities: Sequence[Tuple[float, float]],
                   reflect: Sequence[Optional[int]], device: str) -> Any:
    """The fifteen scalars as one 104-byte device record, built once at plan time.

    PLAN-OWNED AND OUTSIDE THE RESIDENCY REGISTRY: :meth:`.Residency.mirror` binds
    float32 and complex64 volumes and refuses anything else BY NAME, and this record is
    neither. Nothing on the launch path allocates.

    ``-1`` stands for an axis with no reflect row (unfolded, or folded METALLIC) — the
    sentinel :func:`.folded_fused_pair._params_tensor` uses, and for its reason: the
    guard that would read it is emitted only for an axis :func:`far_fill_axes`
    returned, so a source that read it anyway would index outside the volume and be
    caught rather than silently imaging row 0.
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
    for name, (real, imag) in zip(("m0", "m1", "m2"), tuple(near_parities)):
        record[name] = (np.float32(real), np.float32(imag))
    for name, (real, imag) in zip(("d0", "d1", "d2"), tuple(far_parities)):
        record[name] = (np.float32(real), np.float32(imag))
    rows = tuple(-1 if value is None else int(value) for value in reflect)
    if len(rows) != 3:
        raise ValueError(f"reflect must be a per-axis triple, got {reflect!r}")
    record["rx"], record["ry"], record["rz"] = rows
    words = np.frombuffer(record.tobytes(), dtype=np.int32).copy()
    return torch.from_numpy(words).to(torch.device(device))


def plan_metal_folded_complex_fused_pair(
        fields: Any, pml: Any, sources: Any = None,
        residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        functions: Optional[Mapping[str, Any]] = None,
        probe: Any = None,
        ) -> Optional[MetalFoldedComplexFusedPairPlan]:
    """Build the folded complex fused D/E plan, or ``None`` when the seam is refused.

    ``None`` is the only refusal: a configuration this kernel does not carry must fall
    back to the array path, never raise into a caller that would otherwise have stepped
    correctly.

    ``functions`` is the mutation seam. The gate compiles a deliberately broken copy of
    the shipped source and hands it here; dropping the argument is not a silent
    slowdown, it is a silent DISARMING.
    """
    if not metal_folded_complex_fused_pair_coverage(
            fields, pml, sources, residency, probe).covered:
        return None
    expansion = expansion_from_probe(
        probe if probe is not None else load_expansion_probe())
    if expansion is None:  # pragma: no cover - the predicate already refused
        return None
    from ..stepping import _boundary_kinds, _far_reflect_rows  # noqa: PLC0415

    grid = fields.grid
    codes, _ = folded_axis_kinds(grid, pml)
    if codes is None:  # pragma: no cover - the predicate already refused
        return None
    codes = tuple(int(code) for code in codes)
    walls = zero_metal_axes(grid)
    kinds = _boundary_kinds(grid, pml)
    # backward=True: this is the D seam. The phase table is resolved through the
    # certified helper so the conjugation rule lives in one place.
    flags, values = complex_fields.phase_arguments(
        complex_fields.bloch_phase_table(grid, kinds), backward=BACKWARD)
    # THE TWO PARITY WORDS PER AXIS, rounded ONCE on the host through
    # `folded_complex.mirror_parity_coefficients` and PASSED, never synthesised
    # in-kernel. The two helpers are the B twin's, IMPORTED: the near word admits any
    # folded axis and the far word only a folded PERIODIC one, and neither question is
    # about which seam this is.
    phases = tuple(int(_call(grid, "mirror_phase", axis, default=0) or 0)
                   for axis in range(3))
    near_parities = parity_words(codes, phases)
    far_parities = far_parity_words(codes, phases)
    # `stepping._far_reflect_rows` is the ONE derivation of the row a folded PERIODIC
    # axis's far face images; asked here rather than re-derived, and passed as a
    # runtime int because nothing about an integer row can move a bit.
    reflect = _far_reflect_rows(grid)

    volumes: List[str] = []

    def bind_complex(name: str, host: Any) -> Any:
        import numpy  # noqa: PLC0415

        volumes.append(name)
        return residency.mirror(name, host, dtype=numpy.complex64)

    def bind_real(name: str, host: Any) -> Any:
        volumes.append(name)
        return residency.mirror(name, host, constant=True)

    # THE ORDER HERE IS THE SIGNATURE'S ORDER and nothing else keeps them in step.
    flux = [bind_complex(name, getattr(fields, name)) for name in ("Dx", "Dy", "Dz")]
    auxiliary = [bind_complex("fu_" + name, getattr(fields, "fu_" + name))
                 for name in ("Dx", "Dy", "Dz")]
    magnetic = [bind_complex(name, getattr(fields, name))
                for name in ("Hx", "Hy", "Hz")]
    electric = [bind_complex(name, getattr(fields, name))
                for name in ("Ex", "Ey", "Ez")]
    workspace = [bind_complex("f_w_" + name, getattr(fields, "f_w_" + name))
                 for name in ("Ex", "Ey", "Ez")]
    inverse_epsilon = [
        bind_real("inv_eps_" + name, fields.inverse_epsilon_for(name))
        for name in ("Ex", "Ey", "Ez")]
    # THE D CURL TAKES THE INTEGER LATTICE and the E constitutive the HALF-INTEGER one
    # (stepping.py:948 vs :1015). The kernel takes both and never asks which is which.
    curl_coefficients = [
        bind_real(f"pml:{stem}_{axis}", getattr(pml, f"{stem}_{axis}"))
        for axis in "xyz" for stem in ("kms", "sinv")]
    constitutive_coefficients = [
        bind_real(f"pml:{stem}_{axis}_h", getattr(pml, f"{stem}_{axis}_h"))
        for axis in "xyz" for stem in ("kps", "kms")]

    pointers = (flux + auxiliary + magnetic + electric + workspace + inverse_epsilon
                + curl_coefficients + constitutive_coefficients)
    if len(pointers) + 1 != PACKED_BINDINGS:  # pragma: no cover - an invariant
        raise AssertionError((len(pointers), PACKED_BINDINGS))

    selected = dict(functions or {})
    for mode in contract_variants:
        if mode not in selected:
            selected[mode] = compile_folded_complex_fused_pair(
                codes, flags, walls, expansion, mode)

    dtdx = grid.dt / grid.dx
    return MetalFoldedComplexFusedPairPlan(
        residency, volumes, codes, flags, values, near_parities, far_parities,
        walls, grid.shape, dtdx, reflect, expansion,
        _params_tensor(grid.shape, dtdx, values, near_parities, far_parities,
                       reflect, residency.device),
        pointers, selected)


# ---------------------------------------------------------------------------
# Registration — NOT WIRED
# ---------------------------------------------------------------------------

def _arm_coverage(context: Any, slot: str) -> Coverage:
    if slot != SLOT:
        return Coverage(
            False, (f"folded complex fused D/E pair cannot fill {slot}",))
    return metal_folded_complex_fused_pair_coverage(
        context.fields, context.pml, context.sources, context.residency)


def _arm_plan(context: Any,
              slot: str) -> Optional[MetalFoldedComplexFusedPairPlan]:
    if slot != SLOT:
        return None
    return plan_metal_folded_complex_fused_pair(
        context.fields, context.pml, context.sources, context.residency,
        context.contract_variants)


def register_arms() -> Tuple[Any, ...]:
    """One row on ``step_D``, ``wired=False``.

    THE FLAG IS THE WHOLE COMPOSITION STORY, and it is every other fused pair's.
    ``plan_step`` assigns at most one arm per slot and this product spans five, so
    there is no slot it could claim through the arm table. Registering unwired keeps it
    ENUMERABLE for the disjointness sweep (``arms.registered``) while
    ``arms.arms_for`` skips it, so ``_select_slot`` cannot select it and no existing
    arm's selection changes.

    THE OTHER HALF IS THE ABSORB TABLE:
    ``launch.FUSED_PAIR_ARMS["folded_complex_fused_pair"]`` declares which arm each of
    the two slots implements.
    """
    from . import arms  # noqa: PLC0415
    return (arms.register(FAMILY, SLOT, "folded complex fused D/E pair",
                          _arm_coverage, _arm_plan,
                          prefix="folded complex fused D/E pair: ",
                          noun=("folded complex PML D-curl/mirror-fill/"
                                "stored-E pair"),
                          wired=False, replaces=REPLACES),)


ARMS = register_arms()
