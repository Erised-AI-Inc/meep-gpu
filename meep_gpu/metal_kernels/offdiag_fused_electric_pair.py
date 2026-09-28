"""The PML ``step_D`` welded into the OFF-DIAGONAL ``update_E`` — one dispatch.

THE FIRST OF THE FOUR STENCIL WELDS, and the first product on either backend to
close a cell the boards have called ``STRUCTURALLY UNFUSABLE ON ANY BACKEND``
since the fusion campaign opened. The 2026-09-01 Metal board scores this cell::

    D->E  (PML curl, offdiag)                          16 rows, 8 clearing
        shaders.curl 15p + offdiag_update_e.offdiag 24p, sharing 3
        -> 36 pointers, ceiling 30                     UNFUSABLE ON METAL
        verdict: STRUCTURALLY UNFUSABLE ON ANY BACKEND — the off-diagonal
        constitutive arm is a STENCIL over the curl arm's own in-place D output
        (stepping.py:1235-1253 reads the partner D volume at four indices;
        step_D writes it in place), and no grid-wide barrier exists inside one
        launch.

BOTH HALVES OF THAT VERDICT ARE ANSWERED, and neither by relaxing anything.

**THE STENCIL.** The premise is "the curl writes D IN PLACE", and that is a
property of the composition rather than of the physics. This kernel does not
write D at all: the curl half writes ``D_new`` and ``fu_new`` to LAUNCH-LOCAL
SCRATCH buffers, the pre-launch ``D``/``fu`` stay readable for the whole
dispatch, and the constitutive half RE-DERIVES every foreign displacement tap
from pre-launch state through the same inline ``step_cell`` the thread used for
its own cell. Nothing written is ever read, so no barrier is needed. The
launcher rotates the buffers afterwards — :class:`.offdiag_weld_common.\
ScratchWeldPairPlan`, which is :mod:`.ade_update_p`'s certified choreography
moved to the D seam.

The cost is real and is stated rather than hidden: a live term re-derives up to
three foreign cells, so a thread runs ``step_cell`` up to thirteen times where
the separate composition runs it once. THAT IS NOT WHAT THIS PRODUCT IS FOR.
The campaign's rule is that every buildable fused product is built for coverage
and that timing chooses routes at the end, never what to build; a cell that has
been recorded unbuildable on every board for the whole campaign is worth closing
whether or not the closed form is the fastest route to it.

**THE BINDING CEILING.** 36 pointers is what the two halves need when D is
SHARED between them. Not sharing it costs three more (pre-launch and scratch are
different buffers) and the split field another three, so the honest unpacked
count is FORTY-TWO — six worse than the verdict the board recorded, and the
board was right that no naive layout fits. :data:`PACKED_POINTERS` is 23,
because the shipped :mod:`.coefficient_pack` closes it twice over:

* ``cpack`` — the twelve read-only PER-AXIS coefficient vectors of both halves,
  the curl's ``kmx``/``sinvx``/... and the constitutive's ``kp0``/``km0``/...,
  disjointly named so one prologue re-creates all twelve under their certified
  names. Saves eleven. This is exactly :mod:`.conductive_fused_electric_pair`'s
  pack, member for member and in the same order.
* ``mpack`` — the NINE read-only MATERIAL VOLUMES: three inverse-epsilon and the
  six off-diagonal ``chi1inv`` rows. Saves eight.

THE SECOND PACK IS A NEW STEP AND ITS COST IS STATED. ``conductive_fused_\
electric_pair`` declined to pack volumes beside vectors, on the ground that
mixing grid-sized members into a per-axis offset table is "a mixed-geometry
table for no binding need — 29 is already two under the ceiling". Here there IS
a binding need: with ``cpack`` alone the signature is 31 pointers plus
``Params``, which is 32 bindings against a ceiling of 31 and does not compile
(:data:`CPACK_ONLY_BINDINGS`, which the gate compiles and REQUIRES to fail). So
the material volumes ride their OWN pack rather than joining the vectors' — two
tables of uniform geometry instead of one mixed one — and the price is that
those nine volumes are resident twice on the device, once as the array path's
mirrors and once inside the pack. A run whose material volumes dominate its
footprint pays about nine volumes for this weld; that is the trade, and the
alternative on this platform is not a cheaper weld but no weld.

**WHAT IS LIFTED.** Neither half is retyped:

* the curl — :func:`.shaders.curl_source` with ``backward=True``, its body cut
  at the decode anchor and re-headed as an inline function of ``(i, j, k)`` by
  :func:`.offdiag_weld_common.lift_curl_tail` / ``step_cell_function``. The two
  store lines become a struct return and NOTHING else moves;
* the constitutive — :func:`.offdiag_update_e.offdiag_source`'s own output, with
  every displacement read redirected by exact needles built from that emitter's
  OWN index tables (:func:`.offdiag_weld_common.displacement_needles`). After
  the substitution no ``gN[`` may survive: in this signature those pointers are
  the MAGNETIC field, so a missed read is a smooth wrong answer, and the builder
  raises rather than emitting one;
* the in-seam ``zero_metal_D`` — carried by the closed form
  (:func:`.offdiag_weld_common.d_final_function`), which on an unfolded grid
  degenerates to ``clear ? +0.0f : v``. That is the same wall table
  :mod:`.fused_electric_pair` carries, reached through ``coverage.zero_metal_axes``
  and ``IYEE_SHIFTS`` rather than spelled again.

THE CLOSED FORM IS MEASURED, twice and independently. Host-side against the
array path's own pass order over seventeen fixtures, two complete steps each,
zero differing uint32 words, with the parity spelling and the multi-axis nesting
measured at word level and every reachable null diverging
(``parity/meep_gpu/results/metal_scratch_weld_closed_form_2026-09-01T2``); and
on device by this family's own gate.

THE SEAM'S OTHER PASSES.

* **the electric source slot — REFUSED BY NAME, and the refusal is the cell.**
  :data:`CARRIES_DEPOSIT_REPAIR` is **False** and cannot be anything else:
  ``deposit_repair.repairable`` refuses to reconstruct an OFF-DIAGONAL chi1inv
  constitutive by name (deposit_repair.py:98-108), because a point repair
  recomputes ``E`` at the deposit cell from its own displacement and this
  constitutive reads its neighbours'. That is why the board's demand column for
  this cell reads 8 of 16 rather than 16: the other eight carry an electric
  source inside the seam and are permanently out of reach of any fused product,
  on any backend. Ignorance is never an empty set — an undeclared source list is
  a refusal, not an assumption of no sources.
* **``fill_symmetry_bc_D`` and ``fill_folded_far_ghosts_D`` — INERT, and refused
  anyway.** Both return before touching a cell unless ``grid.has_symmetry()``
  (stepping.py:1481-1483, :1565-1566). A folded grid is a DIFFERENT product
  (:mod:`.folded_offdiag_fused_electric_pair`), and the predicate says so by
  name rather than leaving a reader to chase a shared grid clause.

NOT WIRED AS AN ARM, for the reason every Metal fused pair records: ``plan_step``
assigns at most one arm per slot and this product spans three. It registers
``wired=False`` and reaches its two slots through ``launch.FUSED_PAIR_ARMS``.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..triton_kernels.coverage import Coverage
from . import coefficient_pack, offdiag_update_e as _offdiag, shaders, templates
from .coverage import pml_curl_coverage, zero_metal_axes
from .device import MAX_BUFFER_BINDINGS, Residency, compile_source
from .launch import SUB_STEPS
from . import offdiag_weld_common as _weld
from .. import deposit_repair as _deposit_repair

#: Does this product bracket its fused launch with the deposit repair? NO, and
#: the flag is FORCED rather than chosen. ``deposit_repair.repairable`` refuses an
#: off-diagonal chi1inv constitutive BY NAME: the repair recomputes E at a deposit
#: cell from that cell's own displacement, and this constitutive reads its
#: neighbours'. A product that passed True here would recompute the deposit cells
#: from the wrong stencil and report success.
CARRIES_DEPOSIT_REPAIR = False

FAMILY = "offdiag_fused_electric_pair"

#: The slot this arm holds a row on; the product spans three.
SLOT = "step_D"

#: The driver passes one launch performs, in driver order (driver.py:3292-3302).
#: The two fills are absent because they are INERT on every grid this family
#: admits — the predicate refuses a fold by name.
REPLACES: Tuple[str, ...] = ("step_D", "zero_metal_D", "update_E")

#: BOTH halves' read-only per-axis coefficient vectors, IN PACK ORDER: the curl's
#: six, then the constitutive's six. Disjointly named, so one prologue re-creates
#: all twelve under their certified names. THE SAME MEMBERS AND THE SAME ORDER as
#: :data:`.conductive_fused_electric_pair.PACKED_VECTORS`.
PACKED_VECTORS: Tuple[str, ...] = ("kmx", "sinvx", "kmy", "sinvy", "kmz", "sinvz",
                                   "kp0", "km0", "kp1", "km1", "kp2", "km2")

#: The read-only MATERIAL VOLUMES, in pack order: three inverse-epsilon volumes
#: under the certified constitutive body's own names, then the six off-diagonal
#: ``chi1inv`` rows in :data:`.offdiag_update_e.ROW_SLOTS` order.
PACKED_VOLUMES: Tuple[str, ...] = ("e0", "e1", "e2",
                                   "u01", "u02", "u11", "u12", "u21", "u22")

#: FOUR BINDING COUNTS, ALL COMPILED BY THE GATE:
#:
#: * :data:`SEPARATE_SCALAR_BINDINGS` — 42 pointers + 5 separate scalars.
#: * :data:`UNPACKED_POINTER_BINDINGS` — 42 pointers + one packed ``Params&``:
#:   the honest cost of NOT sharing D, six worse than the 36 the board priced.
#: * :data:`CPACK_ONLY_BINDINGS` — 31 pointers + ``Params``, the shape one pack
#:   leaves. 32 against a ceiling of 31: OVER BY ONE, and it is the measurement
#:   that makes the second pack a necessity rather than a preference.
#: * :data:`PACKED_BINDINGS` — the shipped shape.
SEPARATE_SCALAR_BINDINGS = 47
UNPACKED_POINTER_BINDINGS = 43
CPACK_ONLY_BINDINGS = 32
PACKED_BINDINGS = 24

#: The pointer counts, spelled apart from the binding counts.
UNPACKED_POINTERS = 42
PACKED_POINTERS = 23

if PACKED_BINDINGS > MAX_BUFFER_BINDINGS:  # pragma: no cover - a design invariant
    raise RuntimeError(
        f"the off-diagonal fused electric pair binds {PACKED_BINDINGS} buffers "
        f"and Metal's ceiling on this toolchain is {MAX_BUFFER_BINDINGS}")

__all__ = [
    "CARRIES_DEPOSIT_REPAIR", "CPACK_ONLY_BINDINGS", "FAMILY", "PACKED_BINDINGS",
    "PACKED_POINTERS", "PACKED_VECTORS", "PACKED_VOLUMES", "REPLACES",
    "SEPARATE_SCALAR_BINDINGS", "SLOT", "UNPACKED_POINTERS",
    "UNPACKED_POINTER_BINDINGS", "certified_curl_body",
    "compile_offdiag_fused_electric_pair", "metal_offdiag_fused_electric_pair_coverage",
    "offdiag_fused_electric_pair_source", "params_record_dtype",
    "plan_metal_offdiag_fused_electric_pair", "refuted_cpack_only_source",
    "refuted_separate_scalar_source", "register_arms",
]


# ---------------------------------------------------------------------------
# The source
# ---------------------------------------------------------------------------

#: The curl's two store lines, and what they become. Both computed their value
#: into a register first, so the edit DELETES the store and touches no arithmetic.
CURL_STORE_EDITS: Tuple[Tuple[str, str], ...] = (
    ("    u0[ii] = n0; u1[ii] = n1; u2[ii] = n2;\n", ""),
    ("    f0[ii] = v0; f1[ii] = v1; f2[ii] = v2;\n", ""),
)

#: ``step_cell``'s pointer parameters, under the CERTIFIED CURL BODY'S OWN NAMES —
#: that is what lets the lifted text index ``f0``/``u0``/``g0``/``kmx`` character
#: for character. The kernel's own names for the same buffers are different (see
#: :data:`_ACTUALS`) because ``f0`` in the fused signature is an E target.
STEP_POINTERS: Tuple[Tuple[str, str], ...] = tuple(
    ("device const float*", name) for name in
    ("f0", "f1", "f2", "u0", "u1", "u2", "g0", "g1", "g2")
    + PACKED_VECTORS[:6])

#: Curl parameter name -> the kernel's name for that buffer.
_ACTUALS: Dict[str, str] = {"f0": "pf0", "f1": "pf1", "f2": "pf2",
                            "u0": "pu0", "u1": "pu1", "u2": "pu2"}

#: The call-site actual list, and the declaration list, built from one table.
STEP_ARGUMENTS = ", " + ", ".join(_ACTUALS.get(name, name)
                                  for _kind, name in STEP_POINTERS)
CALL_ARGUMENTS = (", " + ", ".join(f"{kind} {name}"
                                   for kind, name in STEP_POINTERS)
                  + ", float dtdx")

#: The per-axis ``(down, up)`` neighbour-index variable names, NARROWED from the
#: certified emitter's own five-member table. Only this pair is what the
#: displacement needles compose an index from; narrowing here rather than passing
#: the raw table is what lets one substitution serve emitters whose tables have
#: different shapes.
NEIGHBOUR_NAMES: Dict[int, Tuple[str, str]] = {
    axis: (_offdiag._TWO_WAY_NAMES[letter][0], _offdiag._TWO_WAY_NAMES[letter][1])
    for axis, letter in enumerate(_weld.AXES)}

#: What one thread forwards to ``step_cell`` / ``d_final_c``.
FORWARD = ", ".join(_weld.COORDS) + ", nxi, nyi, nzi" + STEP_ARGUMENTS + ", dtdx"

_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

struct Params {
    uint nx; uint ny; uint nz; uint n_elem; float dtdx;
__PACK_FIELDS__
};

__STEP_CELL__

__D_FINAL__

kernel void offdiag_fused_electric_pair_step(
    device float*       d0s     [[buffer(0)]],
    device float*       d1s     [[buffer(1)]],
    device float*       d2s     [[buffer(2)]],
    device float*       n0s     [[buffer(3)]],
    device float*       n1s     [[buffer(4)]],
    device float*       n2s     [[buffer(5)]],
    device const float* pf0     [[buffer(6)]],
    device const float* pf1     [[buffer(7)]],
    device const float* pf2     [[buffer(8)]],
    device const float* pu0     [[buffer(9)]],
    device const float* pu1     [[buffer(10)]],
    device const float* pu2     [[buffer(11)]],
    device float*       f0      [[buffer(12)]],
    device float*       f1      [[buffer(13)]],
    device float*       f2      [[buffer(14)]],
    device float*       w0      [[buffer(15)]],
    device float*       w1      [[buffer(16)]],
    device float*       w2      [[buffer(17)]],
    device const float* g0      [[buffer(18)]],
    device const float* g1      [[buffer(19)]],
    device const float* g2      [[buffer(20)]],
    device const float* cpack   [[buffer(21)]],
    device const float* mpack   [[buffer(22)]],
    constant Params&    prm     [[buffer(23)]],
    uint idx [[thread_position_in_grid]])
{
    // THE SCALARS AND THE TWO PACKS ARE UNPACKED INTO THE CERTIFIED BODIES' OWN
    // NAMES, once, before any lifted text runs. Everything below is then
    // character-for-character what the two certified emitters produce, plus the
    // seam and the redirected displacement reads.
    uint nx = prm.nx, ny = prm.ny, nz = prm.nz, n_elem = prm.n_elem;
    float dtdx = prm.dtdx;
__CPACK_PROLOGUE__
__MPACK_PROLOGUE__
__BODY__
}
"""

#: Where the seam splices into the certified constitutive body: immediately after
#: its decode prologue, which is where every index the seam needs first exists and
#: before any displacement is read.
CONSTITUTIVE_DECODE_END = "    int i   = plane / nyi;\n"


def certified_curl_body(codes: Sequence[int],
                        contract: str = shaders.CONTRACT_OFF) -> str:
    """The ``step_D`` curl's body as an inline function of ``(i, j, k)``.

    ``backward`` is read from :data:`.launch.SUB_STEPS` rather than written as a
    literal, so this family and the shipped table cannot drift.
    """
    source = shaders.curl_source(codes, bool(SUB_STEPS["step_D"]["backward"]),
                                 contract)
    tail = _weld.lift_curl_tail(source, store_edits=CURL_STORE_EDITS)
    return _weld.step_cell_function(
        "step_cell", tail, value_type="float",
        pointer_parameters=STEP_POINTERS,
        scalar_parameters=(("float", "dtdx"),),
        returns=("v0", "v1", "v2", "n0", "n1", "n2"))


def offdiag_fused_electric_pair_source(
        row_mask: Sequence[int], codes: Sequence[int], walls: Sequence[int],
        contract: str = shaders.CONTRACT_OFF) -> str:
    """One specialised fused source: (live rows, boundary triple, walls, mode).

    Every constant Triton bakes into a ``tl.constexpr`` is baked into the string
    here, for the reason :mod:`.shaders` gives: ``torch.mps.compile_shader`` takes
    a source and nothing else.
    """
    row_mask = tuple(int(flag) for flag in row_mask)
    codes = tuple(int(code) for code in codes)
    walls = tuple(int(flag) for flag in walls)
    if len(codes) != 3 or len(walls) != 3:
        raise ValueError(f"codes and walls are per-axis triples, got {codes!r} "
                         f"and {walls!r}")
    if len(row_mask) != len(_offdiag.ROW_SLOTS):
        raise ValueError(f"row_mask must fill the {len(_offdiag.ROW_SLOTS)} slots "
                         f"of ROW_SLOTS, got {row_mask!r}")

    needles = _weld.displacement_needles(
        row_mask, e_terms=_offdiag.E_TERMS,
        transverse_partners=_offdiag.TRANSVERSE_PARTNERS,
        neighbour_names=NEIGHBOUR_NAMES, index=_offdiag._index,
        register="dfin{0}",
        call="d_final_{0}({1}, nxi, nyi, nzi" + STEP_ARGUMENTS + ", dtdx)")
    body = _weld.weld_body(
        _offdiag.offdiag_source(row_mask, codes, walls, contract),
        decode_end=CONSTITUTIVE_DECODE_END,
        seam=_weld.seam_lines(value_type="float", forward=FORWARD,
                              split_field=True),
        needles=needles)
    finals = "\n\n".join(
        _weld.d_final_function(
            component, step_name="step_cell", call_arguments=CALL_ARGUMENTS,
            value_type="float", folded_axes={}, far_axes={},
            zero_metal=walls, complex_storage=False)
        for component in range(3))
    return templates.substitute(_TEMPLATE, {
        "__CONTRACT__": shaders.contraction_pragma(contract),
        "__PACK_FIELDS__": (coefficient_pack.params_fields(PACKED_VECTORS)
                            + "\n" + coefficient_pack.params_fields(PACKED_VOLUMES)),
        "__STEP_CELL__": certified_curl_body(codes, contract),
        "__D_FINAL__": finals,
        "__CPACK_PROLOGUE__": coefficient_pack.prologue("cpack", PACKED_VECTORS),
        "__MPACK_PROLOGUE__": coefficient_pack.prologue("mpack", PACKED_VOLUMES),
        "__BODY__": body,
    })


def compile_offdiag_fused_electric_pair(
        row_mask: Sequence[int], codes: Sequence[int], walls: Sequence[int],
        contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised entry point for one (rows, boundaries, walls, mode)."""
    return compile_source(offdiag_fused_electric_pair_source(
        row_mask, codes, walls, contract)).offdiag_fused_electric_pair_step


# ---------------------------------------------------------------------------
# The two refuted signatures — measurements, not arguments
# ---------------------------------------------------------------------------

_WRITTEN: Tuple[str, ...] = ("d0s", "d1s", "d2s", "n0s", "n1s", "n2s",
                             "f0", "f1", "f2", "w0", "w1", "w2")
_READ_VOLUMES: Tuple[str, ...] = ("pf0", "pf1", "pf2", "pu0", "pu1", "pu2",
                                  "g0", "g1", "g2")


def _touch_kernel(name: str, written: Sequence[str], read: Sequence[str],
                  scalars: Sequence[Tuple[str, str]], contract: str) -> str:
    """A signature with a body that touches every buffer, and nothing else.

    What is being measured is the SIGNATURE; a body the compiler could drop would
    let dead-code elimination decide the answer.
    """
    lines: List[str] = []
    slot = 0
    for label in written:
        lines.append(f"    device float*       {label:<8}[[buffer({slot})]],")
        slot += 1
    for label in read:
        lines.append(f"    device const float* {label:<8}[[buffer({slot})]],")
        slot += 1
    for kind, label in scalars:
        lines.append(f"    {kind:<19} {label:<8}[[buffer({slot})]],")
        slot += 1
    touch = " + ".join(f"{label}[0]" for label in read)
    return slot, "\n".join((
        "#include <metal_stdlib>",
        "using namespace metal;",
        "",
        shaders.contraction_pragma(contract),
        "",
        f"kernel void {name}(",
        *lines,
        "    uint idx [[thread_position_in_grid]])",
        "{",
        "    if (idx >= n_elem) { return; }",
        f"    float touch = {touch};",
        f"    {written[0]}[idx] = {written[0]}[idx] + touch * float(nx + ny + nz);",
        *[f"    {label}[idx] = {label}[idx] * dtdx;" for label in written[1:]],
        "}",
        "",
    ))


def refuted_separate_scalar_source(contract: str = shaders.CONTRACT_OFF) -> str:
    """The 47-binding all-separate signature this platform REFUSES."""
    read = _READ_VOLUMES + PACKED_VECTORS + PACKED_VOLUMES
    slots, source = _touch_kernel(
        "refuted_separate_scalar", _WRITTEN, read,
        (("constant uint&", "nx"), ("constant uint&", "ny"),
         ("constant uint&", "nz"), ("constant uint&", "n_elem"),
         ("constant float&", "dtdx")), contract)
    assert slots == SEPARATE_SCALAR_BINDINGS, (slots, SEPARATE_SCALAR_BINDINGS)
    return source


def refuted_cpack_only_source(contract: str = shaders.CONTRACT_OFF) -> str:
    """The 32-binding ONE-PACK signature — over the ceiling BY ONE.

    This is the measurement that makes ``mpack`` a necessity: with only the
    twelve per-axis vectors packed, the nine material volumes are still nine
    pointers and the signature is 31 of them plus ``Params``.
    """
    read = _READ_VOLUMES + PACKED_VOLUMES + ("cpack",)
    slots, source = _touch_kernel(
        "refuted_cpack_only", _WRITTEN, read,
        (("constant uint&", "nx"), ("constant uint&", "ny"),
         ("constant uint&", "nz"), ("constant uint&", "n_elem"),
         ("constant float&", "dtdx")), contract)
    # Five separate scalars would be five bindings; the shape under test binds
    # them as ONE Params, so the count is pointers + 1.
    pointers = len(_WRITTEN) + len(read)
    assert pointers + 1 == CPACK_ONLY_BINDINGS, (pointers, CPACK_ONLY_BINDINGS)
    return source


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def metal_offdiag_fused_electric_pair_coverage(
        fields: Any, pml: Any, sources: Any = None,
        residency: Any = None) -> Coverage:
    """May ONE dispatch span ``step_D`` -> wall -> off-diagonal ``update_E``?

    A conjunction of the two halves' OWN certified predicates plus the seam
    clauses. Nothing is weakened: a configuration either half refuses is refused
    here with that half's reasons, prefixed so a reader can tell which side said
    it.
    """
    reasons: List[str] = []

    curl = pml_curl_coverage(fields, pml, "step_D", residency)
    if not curl.covered:
        reasons.extend(f"curl half: {reason}" for reason in curl.reasons)
    electric = _offdiag.offdiag_constitutive_coverage(fields, pml, residency)
    if not electric.covered:
        reasons.extend(f"constitutive half: {reason}" for reason in electric.reasons)

    # THE SOURCE SEAM, and here it is a CEILING rather than a clause a wrapper
    # discharges: `deposit_repair.repairable` refuses an off-diagonal chi1inv by
    # name, so a row with an electric source in this seam can never be fused,
    # however the kernel is written.
    reasons.extend(_deposit_repair.seam_source_reasons(
        fields, sources, 'D',
        undeclared=(
            "the source set was not declared: this predicate cannot infer an "
            "empty electric source seam from Fields"),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) is electric: the driver "
            f"injects it BETWEEN step_D and update_E (driver.py:3294-3299) and "
            f"deposit_repair.repairable REFUSES an off-diagonal chi1inv "
            f"constitutive, so no repair can carry it"),
        carries_repair=CARRIES_DEPOSIT_REPAIR))

    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, tuple(dict.fromkeys(
            reasons + ["fields carries no grid"])))

    # THE TWO FILLS. Both halves already refuse a mirror plane; this restates by
    # NAME what that buys on THIS seam, and names the family that does carry them.
    for axis in range(3):
        mirrored = getattr(grid, "is_mirrored", None)
        if callable(mirrored) and bool(mirrored(axis)):
            reasons.append(
                f"axis {axis} is folded: stepping.fill_symmetry_bc_D and "
                f"stepping.fill_folded_far_ghosts_D both run inside this seam "
                f"(driver.py:3300-3302); folded_offdiag_fused_electric_pair is "
                f"the product that carries them")

    # zero_metal_D is CARRIED, so the grid must be able to answer which axes are
    # walled; a grid that cannot would silently be treated as unwalled.
    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            reasons.append(
                f"grid does not expose {name}; zero_metal_D cannot be carried "
                f"inline")

    # A polarization would make the constitutive source (D - sum P) rather than D.
    if tuple(getattr(fields, "polarizations", ()) or ()):
        reasons.append("a susceptibility is registered: this kernel bakes the "
                       "plain constitutive product, whose source is D and not "
                       "(D - sum P)")
    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

PARAMS_ITEMSIZE = 4 * (5 + len(PACKED_VECTORS) + len(PACKED_VOLUMES))


def params_record_dtype() -> Any:
    """The host record's dtype — five scalars then twenty-one pack offsets.

    ONE HOME FOR THE LAYOUT. Every member is four bytes and the struct has no
    member wider than that, so the natural packing IS Metal's and the offsets are
    the running sum; they are stated explicitly anyway, because a record that
    agreed with the device only by accident is one edit from disagreeing.
    """
    import numpy as np  # noqa: PLC0415

    names = ["nx", "ny", "nz", "n_elem", "dtdx"]
    formats = ["<u4", "<u4", "<u4", "<u4", "<f4"]
    for label in PACKED_VECTORS + PACKED_VOLUMES:
        names.append(f"off_{label}")
        formats.append("<u4")
    return np.dtype({
        "names": names, "formats": formats,
        "offsets": [4 * index for index in range(len(names))],
        "itemsize": PARAMS_ITEMSIZE,
    })


def _params_tensor(shape: Sequence[int], dtdx: float,
                   offsets: Mapping[str, int], device: str) -> Any:
    """The scalars and the two packs' offsets as one device record.

    PLAN-OWNED AND OUTSIDE THE RESIDENCY REGISTRY, for the reason every fused
    pair's ``_params_tensor`` gives: :meth:`.Residency.mirror` binds float32 and
    complex64 volumes and refuses anything else by name.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    record = np.zeros(1, dtype=params_record_dtype())
    nx, ny, nz = (int(n) for n in shape)
    record["nx"], record["ny"], record["nz"] = nx, ny, nz
    record["n_elem"] = nx * ny * nz
    record["dtdx"] = np.float32(dtdx)
    for label, offset in offsets.items():
        record[f"off_{label}"] = int(offset)
    words = np.frombuffer(record.tobytes(), dtype=np.int32).copy()
    return torch.from_numpy(words).to(torch.device(device))


#: The rotating volumes, in SIGNATURE ORDER: the three displacements then the
#: three split-field auxiliaries. The plan binds every scratch first and every
#: pre-launch buffer second, in this order.
ROTATED_NAMES: Tuple[str, ...] = ("Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz")


def plan_metal_offdiag_fused_electric_pair(
        fields: Any, pml: Any, sources: Any = None,
        residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        functions: Optional[Mapping[str, Any]] = None,
        ) -> Optional[_weld.ScratchWeldPairPlan]:
    """Build the fused off-diagonal D/E plan, or ``None`` when refused.

    ``None`` is the only refusal: a configuration this kernel does not carry must
    fall back to the array path, never raise into a caller that would otherwise
    have stepped correctly.

    ``functions`` is the mutation seam. The gate compiles a deliberately broken
    copy of the shipped source and hands it here; dropping the argument is not a
    silent slowdown, it is a silent DISARMING.
    """
    if not metal_offdiag_fused_electric_pair_coverage(
            fields, pml, sources, residency).covered:
        return None
    import numpy as np  # noqa: PLC0415

    from ..stepping import _boundary_kinds  # noqa: PLC0415

    grid = fields.grid
    codes = tuple(1 if kind == "metallic" else 0
                  for kind in _boundary_kinds(grid, pml))
    walls = _offdiag.wall_mask_axes(grid)
    rows = _offdiag.row_volumes_for(fields)
    row_mask = tuple(int(value is not None) for value in rows)

    volumes: List[str] = []

    def bind(name: str, host: Any, constant: bool = False) -> Any:
        volumes.append(name)
        return residency.mirror(name, host, constant=constant)

    # THE ROTATING GROUPS FIRST, in ROTATED_NAMES order: every scratch, then every
    # pre-launch buffer. `ScratchWeldPairPlan.run` splices exactly those two
    # groups ahead of `static_args`, so this order IS the signature's.
    twins: Dict[str, Any] = {}
    for name in ROTATED_NAMES:
        host = getattr(fields, name)
        bind(name, host)
        twin, _mirror = _weld.scratch_twin(residency, name, host)
        twins[name] = twin

    electric = ("Ex", "Ey", "Ez")
    stored = [bind(name, getattr(fields, name)) for name in electric]
    workspace = [bind("f_w_" + name, getattr(fields, "f_w_" + name))
                 for name in electric]
    magnetic = [bind(name, getattr(fields, name))
                for name in SUB_STEPS["step_D"]["sources"]]

    # THE D CURL TAKES THE INTEGER LATTICE and the E constitutive the HALF-INTEGER
    # one (SUB_STEPS['step_D']['suffix'] is '' and the off-diagonal side reads
    # `kps_a_h`/`kms_a_h`). The kernel takes one packed buffer and never asks; the
    # gate carries a host mutation for each group.
    suffix = SUB_STEPS["step_D"]["suffix"]
    vectors = ([getattr(pml, f"{stem}_{axis}{suffix}")
                for axis in "xyz" for stem in ("kms", "sinv")]
               + [getattr(pml, f"{stem}_{axis}_h")
                  for axis in "xyz" for stem in ("kps", "kms")])
    cpack, cpack_layout = coefficient_pack.packed_mirror(
        residency, f"{FAMILY}:cpack", np, PACKED_VECTORS, vectors)

    # A DEAD ROW SLOT STILL OCCUPIES ITS PACK MEMBER, because the pack refuses a
    # zero-length member (two offsets would coincide) and the specialisation is
    # what stops the read. The filler is the component's own inverse epsilon: a
    # read-only volume of the right shape that is already resident, so no
    # allocation and no new name.
    inverse = [fields.inverse_epsilon_for(name) for name in electric]
    component_of_slot = tuple(
        next(index for index, term in enumerate(_offdiag.E_TERMS)
             if term[0] == row)
        for row, _partner in _offdiag.ROW_SLOTS)
    row_volumes = [value if value is not None else inverse[component_of_slot[slot]]
                   for slot, value in enumerate(rows)]
    mpack, mpack_layout = coefficient_pack.packed_mirror(
        residency, f"{FAMILY}:mpack", np, PACKED_VOLUMES, inverse + row_volumes)

    offsets = dict(zip(cpack_layout.names, cpack_layout.offsets))
    offsets.update(zip(mpack_layout.names, mpack_layout.offsets))

    selected = dict(functions or {})
    for mode in contract_variants:
        if mode not in selected:
            selected[mode] = compile_offdiag_fused_electric_pair(
                row_mask, codes, walls, mode)

    dtdx = grid.dt / grid.dx
    static = (list(stored) + list(workspace) + list(magnetic)
              + [cpack, mpack,
                 _params_tensor(grid.shape, dtdx, offsets, residency.device)])
    assert (2 * len(ROTATED_NAMES) + len(static)) == PACKED_BINDINGS, (
        2 * len(ROTATED_NAMES) + len(static), PACKED_BINDINGS)
    return _weld.plan_scratch_weld(
        FAMILY, residency, fields, rotated_names=ROTATED_NAMES,
        static_args=static, functions=selected, volumes=volumes,
        shape=grid.shape, codes=codes, zero_metal=walls, row_mask=row_mask,
        replaces=REPLACES, twins=twins)


# ---------------------------------------------------------------------------
# Registration — wired=False (no arm selection), routed by the absorb table
# ---------------------------------------------------------------------------

def _arm_coverage(context: Any, slot: str) -> Coverage:
    if slot != SLOT:
        return Coverage(False, (f"offdiag fused electric pair cannot fill {slot}",))
    return metal_offdiag_fused_electric_pair_coverage(
        context.fields, context.pml, context.sources, context.residency)


def _arm_plan(context: Any, slot: str) -> Optional[_weld.ScratchWeldPairPlan]:
    if slot != SLOT:
        return None
    return plan_metal_offdiag_fused_electric_pair(
        context.fields, context.pml, context.sources, context.residency,
        context.contract_variants)


def register_arms() -> Tuple[Any, ...]:
    """One row on ``step_D``, ``wired=False``.

    ``plan_step`` assigns at most one ARM per slot and this product spans three,
    so there is no slot it could claim through the arm table. Registering unwired
    keeps it ENUMERABLE for the disjointness sweep while ``arms.arms_for`` skips
    it. The other half is the absorb table:
    ``launch.FUSED_PAIR_ARMS["offdiag_fused_electric_pair"]`` declares which arm
    each of the two slots implements.
    """
    from . import arms  # noqa: PLC0415
    return (arms.register(FAMILY, SLOT, "off-diagonal fused electric D/E pair",
                          _arm_coverage, _arm_plan,
                          prefix="off-diagonal fused electric D/E pair: ",
                          noun="fused PML D-curl/off-diagonal-E pair",
                          wired=False, replaces=REPLACES),)


ARMS = register_arms()
