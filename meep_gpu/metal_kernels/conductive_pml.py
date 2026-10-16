"""Real conductive split-field PML curls for PyTorch/MPS.

This family implements the active-PML branch of ``stepping._apply_curl`` for a
conductive target (``stepping._apply_conductive_pml_update``). Unlike no-PML
conductivity it carries the ``f_cond`` history and selects one of MEEP's four
conductive PML forms per structure chunk, as ``step_generic.cpp::step_curl`` does.

THE FORM IS CHOSEN PER STRUCTURE CHUNK. Membership is runtime data, never source
text: the kernel binds the chunk record (:func:`meep_gpu.chunk_map.chunk_bounds_record`)
as ``constant ChunkBounds& cb`` and reads it through the base curl's anchored bounds
block (:func:`.shaders.curl_bounds_block`), evaluated at the cell's own stored index on
the target's own Yee sub-lattice (half-integer for B, integer for D). Per target, with
``X`` the curl, ``cf``/``ci`` the conductivity coefficients, ``kms``/``sinv`` the dsig
axis's and ``kms_u``/``sinv_u`` the dsigu axis's (every operation rounds to float32 in
this order, no contraction):

* a conductive target, on a cell whose chunk carries both layers::

      fcnd' = ((fcnd*cf) - X)*ci;  fu' = ((fu*kms) + (fcnd' - fcnd))*sinv;
      f'    = (((f*kms_u) + fu') - fu)*sinv_u

* dsig only: ``fcnd'`` as above, then ``f' = ((f*kms) + (fcnd' - fcnd))*sinv``;
* dsigu only: ``fu' = ((fu*cf) - X)*ci``, then ``f'`` as in the first form;
* neither: ``f' = ((f*cf) - X)*ci``;
* a LOSSLESS target of the same sub-step: the base curl's two forms -- the split-field
  recurrence where its chunk carries the dsigu layer, the dsig-only form
  ``((f*kms) - X)*sinv`` everywhere else (:mod:`.shaders`).

The increment ``(fcnd' - fcnd)`` is formed FIRST and then added: MEEP's grouping
(step_generic.cpp:167, :209). ``fu`` is read and written only on dsigu cells and
``f_cond`` only on dsig cells (the auxiliary contract), and the curl stencil is MEEP's
left-to-right bracket, lifted from :func:`.shaders.curl_source` rather than re-spelled.

PER-CHUNK CONDUCTIVITY PRESENCE IS BOUND AS WORDS, beside the record. MEEP deletes a
chunk's conductivity array when every sample it visits is zero (structure.cpp:883-900),
and such a chunk steps the lossless forms even where the component is lossy elsewhere
(``ChunkLayout.conductive_chunks``; ``stepping._apply_conductive_pml_update``'s
``conductive`` mask). The kernel binds one 27-bit word per target
(:data:`PRESENCE_STRUCT`, last, at :data:`PRESENCE_BUFFER`) and indexes it with the
cell's chunk code ``lx + 3*ly + 9*lz``, each axis labelled off the chunk record on the
target's own Yee sub-lattice there -- the off-diagonal family's rule and its device
text (:mod:`.offdiag_update_e`, KERNELS.md §2's presence extension). A lossy target's
cell whose chunk keeps no conductivity steps the lossless forms and leaves ``f_cond``
alone. The words are built on the host from the array path's own per-cell presence
(``RegionPlan.conductive_mask``) and checked to expand back to it on every stored
cell, or the sub-step is refused by name (:func:`presence_words`); a target every
chunk keeps is the all-kept word.

The direct native-MPS gate verifies every branch, history-preservation case, and
mutation. The whole-step gate separately proves the two mixed B/H/D/E compositions in
which one conductive PML curl shares persistent mirrors with the ordinary PML curl and
ordinary constitutive pair; the family is therefore wired into the composer.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..triton_kernels.conductivity import (
    CONDUCTIVE_SUB_STEPS,
    conductive_pml_curl_coverage as shared_conductive_pml_coverage,
    conductive_targets,
)
from ..triton_kernels.coverage import Coverage, _boundary_kinds
from . import chunk_bounds as _chunk_bounds
from . import shaders, templates
from .coverage import _metal_backend_reasons, _residency_declaration_reasons
from .device import Residency, compile_source
from .launch import SUB_STEPS
from .no_pml_curl import _FieldsView, _GridView
from .plans import KernelPlan

FAMILY = "conductive_pml"

#: The chunk record's binding index: one past the last scalar (``dtdx`` at 28).
RECORD_BUFFER = 29

#: The conductivity presence words' binding index: last, one past the record. 31 of
#: Metal's 31 bindings (``device.MAX_BUFFER_BINDINGS``).
PRESENCE_BUFFER = 30

#: The presence words' struct: one 27-bit word per target of the sub-step, in
#: ``CONDUCTIVE_SUB_STEPS`` order. Bound by the single as ``constant
#: ConductivePresence& cw``, embedded by the two welds in their ``Params`` and aliased to
#: ``cw``, so the lifted tails read one name.
PRESENCE_STRUCT = "ConductivePresence"
PRESENCE_MEMBERS: Tuple[str, ...] = ("kept0", "kept1", "kept2")
PRESENCE_NAME = "cw"

#: A target whose conductivity every chunk keeps (or a lossless target, whose word is
#: never read): all 27 chunk codes set.
ALL_CHUNKS_KEPT = (1 << 27) - 1
ALL_KEPT: Tuple[int, ...] = (ALL_CHUNKS_KEPT,) * 3

__all__ = [
    "ALL_CHUNKS_KEPT",
    "ALL_KEPT",
    "FAMILY",
    "PRESENCE_BUFFER",
    "PRESENCE_MEMBERS",
    "PRESENCE_NAME",
    "PRESENCE_STRUCT",
    "RECORD_BUFFER",
    "MetalConductivePmlCurlPlan",
    "certified_stencil",
    "compile_conductive_pml_curl",
    "conductive_pml_curl_source",
    "conductive_presence_reasons",
    "membership_flags",
    "metal_conductive_pml_curl_coverage",
    "plan_metal_conductive_pml_curl",
    "presence_binding",
    "presence_block",
    "presence_for",
    "presence_params_alias",
    "presence_params_member",
    "presence_plan_words",
    "presence_reasons",
    "presence_record_dtype",
    "presence_struct_source",
    "presence_tensor",
    "presence_words",
]


_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;
__CONTRACT__

__RECORD_STRUCT__

__PRESENCE_STRUCT__

kernel void conductive_pml_curl_step(
    device float*       f0 [[buffer(0)]], device float*       f1 [[buffer(1)]], device float*       f2 [[buffer(2)]],
    device float*       u0 [[buffer(3)]], device float*       u1 [[buffer(4)]], device float*       u2 [[buffer(5)]],
    device float*       c0 [[buffer(6)]], device float*       c1 [[buffer(7)]], device float*       c2 [[buffer(8)]],
    device const float* cf0 [[buffer(9)]], device const float* cf1 [[buffer(10)]], device const float* cf2 [[buffer(11)]],
    device const float* ci0 [[buffer(12)]], device const float* ci1 [[buffer(13)]], device const float* ci2 [[buffer(14)]],
    device const float* g0 [[buffer(15)]], device const float* g1 [[buffer(16)]], device const float* g2 [[buffer(17)]],
    device const float* kmx [[buffer(18)]], device const float* sinvx [[buffer(19)]],
    device const float* kmy [[buffer(20)]], device const float* sinvy [[buffer(21)]],
    device const float* kmz [[buffer(22)]], device const float* sinvz [[buffer(23)]],
    constant uint& nx [[buffer(24)]], constant uint& ny [[buffer(25)]], constant uint& nz [[buffer(26)]],
    constant uint& n_elem [[buffer(27)]], constant float& dtdx [[buffer(28)]],
    __RECORD_BINDING__,
    __PRESENCE_BINDING__,
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n_elem) { return; }
    int nxi = int(nx), nyi = int(ny), nzi = int(nz);
    int nyz = nyi * nzi;
    int ii = int(idx), k = ii % nzi, plane = ii / nzi;
    int j = plane % nyi, i = plane / nyi;
__BOUNDS__
    int si = i __SHIFT__, sj = j __SHIFT__, sk = k __SHIFT__;
    bool vx = true, vy = true, vz = true;
__GHOST_X__
__GHOST_Y__
__GHOST_Z__
    int ox = si * nyz + j * nzi + k;
    int oy = i * nyz + sj * nzi + k;
    int oz = i * nyz + j * nzi + sk;
    float a = g0[ii], b = g1[ii], c = g2[ii];
    float a_y = vy ? g0[oy] : 0.0f, a_z = vz ? g0[oz] : 0.0f;
    float b_x = vx ? g1[ox] : 0.0f, b_z = vz ? g1[oz] : 0.0f;
    float c_x = vx ? g2[ox] : 0.0f, c_y = vy ? g2[oy] : 0.0f;
__STENCIL__
    bool at_x = (i == 0), at_y = (j == 0), at_z = (k == 0);
__MASK__
    float km_x = kmx[i], si_x = sinvx[i];
    float km_y = kmy[j], si_y = sinvy[j];
    float km_z = kmz[k], si_z = sinvz[k];
__PRESENCE__
__TAIL0__
__TAIL1__
__TAIL2__
}
"""

#: The first and last text of the base curl's stencil statements in
#: :func:`.shaders.curl_source`: MEEP's left-to-right bracket with the partner-presence
#: guard (step_generic.cpp:72-76, :104-108). Lifted, never re-spelled.
STENCIL_BEGIN = "    float curl0 = "
STENCIL_END = "(has_y ? ((-dtdx) * (a_y - a)) : 0.0f);\n"


def certified_stencil(contract: str = shaders.CONTRACT_OFF) -> str:
    """The base curl's three stencil statements, lifted from its own emitter.

    The text does not depend on the boundary codes or the direction (those decide the
    neighbour offsets, which this family computes with the same names), so it is read
    from one specialisation and refused if it is not there exactly once.
    """
    source = shaders.curl_source((shaders.PERIODIC,) * 3, False, contract)
    if source.count(STENCIL_BEGIN) != 1 or source.count(STENCIL_END) != 1:
        raise AssertionError(
            "the certified curl source no longer carries its stencil statements exactly "
            "once; this family lifts them and would otherwise step another grouping")
    start = source.index(STENCIL_BEGIN)
    return source[start:source.index(STENCIL_END, start) + len(STENCIL_END)].rstrip("\n")


def membership_flags(backward: bool) -> Tuple[Tuple[str, str], ...]:
    """Per target, the ``(dsig, dsigu)`` membership locals its forms read.

    vec.hpp's cycle_direction (:data:`.shaders.CURL_CYCLE`): target 0 takes (y, z),
    target 1 (z, x), target 2 (x, y), on the side's lattice
    (:data:`.shaders.CURL_HALF_INTEGER`) -- the lattice of the coefficient tables the
    target reads, and the one the array path's memberships are drawn on
    (``RegionPlan.axis_mask`` and ``RegionPlan.mask``).
    """
    half = shaders.CURL_HALF_INTEGER[bool(backward)]
    return tuple((_chunk_bounds.membership_name(half, dsig),
                  _chunk_bounds.membership_name(half, dsigu))
                 for dsig, dsigu in shaders.CURL_CYCLE)


def _tail(index: int, conductive: bool, flags: Tuple[str, str], km1: str, si1: str,
          km2: str, si2: str) -> str:
    """One target's forms: ``km1``/``si1`` on the dsig axis, ``km2``/``si2`` on dsigu.

    A lossy target reads ``kc{index}`` (:func:`presence_block`): on a cell whose chunk
    keeps no conductivity it steps the lossless target's two forms, character for
    character the lossless tail's arithmetic on the loaded ``f``, and touches neither
    ``f_cond`` nor ``fu`` off the dsigu cells (``stepping._apply_conductive_pml_update``,
    ``conductive`` False there).
    """
    dsig, dsigu = flags
    if not conductive:
        return f"""    const bool du{index} = {dsigu};
    float p{index} = du{index} ? u{index}[ii] : f{index}[ii];
    float n{index} = ((p{index} * {km1}) - curl{index}) * {si1};
    float f{index}_new = du{index} ? ((((f{index}[ii] * {km2}) + n{index}) - p{index}) * {si2}) : n{index};
    if (du{index}) {{ u{index}[ii] = n{index}; }}
    f{index}[ii] = f{index}_new;"""
    return f"""    const bool ds{index} = {dsig}, du{index} = {dsigu};
    float f{index}_previous = f{index}[ii];
    if (!kc{index}) {{
        float p{index} = du{index} ? u{index}[ii] : f{index}_previous;
        float n{index} = ((p{index} * {km1}) - curl{index}) * {si1};
        float f{index}_lossless = du{index} ? ((((f{index}_previous * {km2}) + n{index}) - p{index}) * {si2}) : n{index};
        if (du{index}) {{ u{index}[ii] = n{index}; }}
        f{index}[ii] = f{index}_lossless;
    }} else if (ds{index} && du{index}) {{
        float c{index}_previous = c{index}[ii];
        float u{index}_previous = u{index}[ii];
        float c{index}_new = ((c{index}_previous * cf{index}[ii]) - curl{index}) * ci{index}[ii];
        float c{index}_step = c{index}_new - c{index}_previous;
        float u{index}_new = ((u{index}_previous * {km1}) + c{index}_step) * {si1};
        float f{index}_new = (((f{index}_previous * {km2}) + u{index}_new) - u{index}_previous) * {si2};
        c{index}[ii] = c{index}_new; u{index}[ii] = u{index}_new; f{index}[ii] = f{index}_new;
    }} else if (du{index}) {{
        float u{index}_previous = u{index}[ii];
        float u{index}_new = ((u{index}_previous * cf{index}[ii]) - curl{index}) * ci{index}[ii];
        float f{index}_new = (((f{index}_previous * {km2}) + u{index}_new) - u{index}_previous) * {si2};
        u{index}[ii] = u{index}_new; f{index}[ii] = f{index}_new;
    }} else if (ds{index}) {{
        float c{index}_previous = c{index}[ii];
        float c{index}_new = ((c{index}_previous * cf{index}[ii]) - curl{index}) * ci{index}[ii];
        float c{index}_step = c{index}_new - c{index}_previous;
        float f{index}_new = ((f{index}_previous * {km1}) + c{index}_step) * {si1};
        c{index}[ii] = c{index}_new; f{index}[ii] = f{index}_new;
    }} else {{
        f{index}[ii] = ((f{index}_previous * cf{index}[ii]) - curl{index}) * ci{index}[ii];
    }}"""


def conductive_pml_curl_source(codes: Sequence[int], backward: bool,
                               conductive: Sequence[bool],
                               contract: str = shaders.CONTRACT_OFF) -> str:
    """Emit one PML/conductive specialization, including its target flags.

    The chunk division is NOT in the text: it is the bound record, so one source serves
    every division and the specialisation set does not grow with the grid.
    """
    codes = tuple(int(code) for code in codes)
    flags = tuple(bool(flag) for flag in conductive)
    if len(codes) != 3 or len(flags) != 3:
        raise ValueError("codes and conductive must each be triples")
    membership = membership_flags(backward)
    return templates.substitute(_TEMPLATE, {
        "__CONTRACT__": templates.contraction_pragma(contract),
        "__RECORD_STRUCT__": _chunk_bounds.struct_source(),
        "__RECORD_BINDING__": _chunk_bounds.binding(RECORD_BUFFER),
        "__PRESENCE_STRUCT__": presence_struct_source(),
        "__PRESENCE_BINDING__": presence_binding(PRESENCE_BUFFER),
        "__PRESENCE__": presence_block(backward, codes, flags),
        "__BOUNDS__": shaders.curl_bounds_block(backward),
        "__SHIFT__": "- 1" if backward else "+ 1",
        "__GHOST_X__": templates.ghost("x", codes[0], backward),
        "__GHOST_Y__": templates.ghost("y", codes[1], backward),
        "__GHOST_Z__": templates.ghost("z", codes[2], backward),
        "__STENCIL__": certified_stencil(contract),
        "__MASK__": templates.ownership_mask(codes, backward),
        "__TAIL0__": _tail(0, flags[0], membership[0], "km_y", "si_y", "km_z", "si_z"),
        "__TAIL1__": _tail(1, flags[1], membership[1], "km_z", "si_z", "km_x", "si_x"),
        "__TAIL2__": _tail(2, flags[2], membership[2], "km_x", "si_x", "km_y", "si_y"),
    })


def compile_conductive_pml_curl(codes: Sequence[int], backward: bool,
                                conductive: Sequence[bool],
                                contract: str = shaders.CONTRACT_OFF) -> Any:
    return compile_source(conductive_pml_curl_source(
        codes, backward, conductive, contract)).conductive_pml_curl_step


def conductive_presence_reasons(fields: Any, pml: Any, sub_step: str,
                                chunk_layout: Any = None) -> List[str]:
    """Why some chunk keeps no conductivity for a lossy target of ``sub_step``; empty if none.

    MEEP keeps a chunk's conductivity array for a component only where some sample its
    ``LOOP_OVER_VOL`` visits is nonzero (structure.cpp:883-900), and a chunk without one
    steps the lossless forms (step_generic.cpp:104, :138-141, :182-183, :231-234). The
    chunk record names the layer runs and nothing else, so a sub-step whose lossy target
    is kept by only some chunks is refused here by name and steps on the array path
    (``stepping._apply_conductive_pml_update``'s ``conductive`` mask).

    ``chunk_layout`` is the division the run steps with (the composer's
    ``StepContext.chunk_layout``); ``None`` is the one-rank division of the grid and
    layer, the one a driver built without MEEP steps with.

    THE COMPLEX FAMILY'S CLAUSE (:mod:`.complex_conductive_pml`), whose kernels bind no
    presence words. This family binds them and asks :func:`presence_reasons` instead.
    """
    from ..chunk_map import ChunkLayout  # noqa: PLC0415
    from ..fields import IYEE_SHIFTS  # noqa: PLC0415

    grid = getattr(fields, "grid", None)
    if grid is None:
        return ["fields carries no grid"]
    lossy = conductive_targets(fields, sub_step)
    if not any(lossy):
        return []
    layout = chunk_layout
    if layout is None:
        try:
            layout = ChunkLayout.one_rank(grid, pml)
        except Exception as exc:  # noqa: BLE001 - a division that cannot be built refuses
            return [f"the one-rank chunk division could not be built ({exc!r}); the "
                    f"conductive forms are chosen per chunk"]
    reader = getattr(fields, "conductivity_for", None)
    reasons: List[str] = []
    for index, target in enumerate(CONDUCTIVE_SUB_STEPS[sub_step]):
        if not lossy[index]:
            continue
        sigma = reader(target) if callable(reader) else None
        if sigma is None:
            reasons.append(
                f"{target} carries condfac/condinv but no conductivity volume, so which "
                f"chunks keep its conductivity cannot be read")
            continue
        try:
            kept = tuple(layout.conductive_chunks(sigma, IYEE_SHIFTS[target]))
        except Exception as exc:  # noqa: BLE001 - an unreadable presence refuses
            reasons.append(f"the per-chunk conductivity presence of {target} could not "
                           f"be read ({exc!r})")
            continue
        if not all(kept):
            reasons.append(
                f"{sum(kept)} of {len(kept)} structure chunks keep the conductivity of "
                f"{target}: MEEP steps the lossless forms in the other chunks "
                f"(structure.cpp:883-900), and the chunk record carries no per-chunk "
                f"conductivity presence; the array path steps this sub-step")
    return reasons


# ---------------------------------------------------------------------------
# The conductivity presence: the device rule and its host twin
# ---------------------------------------------------------------------------

def presence_struct_source() -> str:
    """The Metal declaration of :data:`PRESENCE_STRUCT`, one ``uint`` per target."""
    lines = [f"struct {PRESENCE_STRUCT} {{"]
    lines.extend(f"    uint {name};" for name in PRESENCE_MEMBERS)
    lines.append("};")
    return "\n".join(lines)


def presence_binding(index: int) -> str:
    """The single's kernel parameter that binds the presence words as their own buffer."""
    from .device import MAX_BUFFER_BINDINGS  # noqa: PLC0415

    index = int(index)
    if not 0 <= index < MAX_BUFFER_BINDINGS:
        raise ValueError(
            f"buffer index {index} is outside 0..{MAX_BUFFER_BINDINGS - 1}; a Metal "
            f"kernel binds at most {MAX_BUFFER_BINDINGS} buffers")
    return f"constant {PRESENCE_STRUCT}& {PRESENCE_NAME} [[buffer({index})]]"


def presence_params_member(indent: str = "    ") -> str:
    """The ``Params`` member line a weld embeds the presence words with."""
    return f"{indent}{PRESENCE_STRUCT} {PRESENCE_NAME};"


def presence_params_alias(struct: str = "prm", indent: str = "    ") -> str:
    """The line a ``Params`` weld adds before its body, so the lifted tails read ``cw.``."""
    return (f"{indent}constant {PRESENCE_STRUCT}& {PRESENCE_NAME} = "
            f"{struct}.{PRESENCE_NAME};")


def presence_record_dtype() -> Any:
    """The numpy structured dtype of the packed presence words: three ``<u4``."""
    import numpy  # noqa: PLC0415

    return numpy.dtype([(name, "<u4") for name in PRESENCE_MEMBERS])


def presence_block(backward: bool, codes: Sequence[int],
                   conductive: Sequence[bool]) -> str:
    """The kernel text that names, per LOSSY target, whether its chunk keeps a conductivity.

    Per lossy target ``n``, its chunk code ``lx + 3*ly + 9*lz``, each axis labelled off
    the chunk record on the target's own Yee sub-lattice there (``fields.IYEE_SHIFTS``),
    then ``kc{n}``: bit ``code`` of the target's word. The label text and its row rule
    are :mod:`.offdiag_update_e`'s (one device rule for every per-chunk presence, read
    through ``cb.`` -- the name the single binds and the welds alias). A lossless target
    gets no flag: its tail reads none.
    """
    from ..fields import IYEE_SHIFTS  # noqa: PLC0415
    from .offdiag_update_e import _label_expression, _label_row  # noqa: PLC0415

    codes = tuple(int(code) for code in codes)
    targets = CONDUCTIVE_SUB_STEPS["step_D" if backward else "step_B"]
    lines = ["    // --- the conductivity presence (KERNELS.md §2): bit `code` of each lossy",
             "    // target's word, `code` the cell's chunk lx + 3*ly + 9*lz read off the",
             "    // chunk record on the target's own Yee sub-lattice per axis. A chunk",
             "    // that keeps no conductivity steps the lossless forms."]
    for index, (target, lossy) in enumerate(zip(targets, conductive)):
        if not lossy:
            continue
        iyee = IYEE_SHIFTS[target]
        labels = [_label_expression(bool(iyee[axis]), axis,
                                    _label_row(bool(iyee[axis]), axis, codes))
                  for axis in range(3)]
        lines.append(f"    const uint cc_chunk{index} = uint({labels[0]}")
        lines.append(f"        + 3 * {labels[1]}")
        lines.append(f"        + 9 * {labels[2]});")
        lines.append(f"    const bool kc{index} = (({PRESENCE_NAME}."
                     f"{PRESENCE_MEMBERS[index]} >> cc_chunk{index}) & 1u) != 0u;")
    if len(lines) == 4:
        lines.append("    // no lossy target")
    return "\n".join(lines)


def _codes_for(fields: Any, pml: Any) -> Tuple[int, int, int]:
    """The kernel's baked boundary codes: 1 on a metallic axis, 0 elsewhere."""
    return tuple(1 if kind == "metallic" else 0  # type: ignore[return-value]
                 for kind in _boundary_kinds(fields.grid, pml))


def presence_words(fields: Any, pml: Any, sub_step: str, record: Any,
                   codes: Sequence[int]) -> Tuple[int, int, int]:
    """The three words in ``CONDUCTIVE_SUB_STEPS[sub_step]`` order, from the array path.

    Each lossy target's word is built from the per-cell presence the array path steps
    with -- ``stepping._region_plan(fields, pml).conductive_mask``, the volume
    ``_apply_conductive_pml_update`` selects on -- through :mod:`.offdiag_update_e`'s
    host twin (``cell_chunk_codes`` on ``record``, the chunk record the kernel binds,
    and ``presence_word``), which refuses by name a division whose cells of one chunk
    code disagree and a word that does not expand back to the volume on every stored
    cell (``PresenceRefusal``). A lossless target, and a target every chunk keeps, is
    :data:`ALL_CHUNKS_KEPT`.
    """
    import numpy  # noqa: PLC0415

    from .. import stepping  # noqa: PLC0415
    from ..fields import IYEE_SHIFTS  # noqa: PLC0415
    from .offdiag_update_e import (  # noqa: PLC0415
        PresenceRefusal,
        cell_chunk_codes,
        presence_word,
    )

    lossy = conductive_targets(fields, sub_step)
    shape = tuple(int(n) for n in fields.grid.shape)
    plan = None
    words: List[int] = []
    for index, target in enumerate(CONDUCTIVE_SUB_STEPS[sub_step]):
        if not lossy[index]:
            words.append(ALL_CHUNKS_KEPT)
            continue
        reader = getattr(fields, "conductivity_for", None)
        sigma = reader(target) if callable(reader) else None
        if sigma is None:
            raise PresenceRefusal(
                f"{target} carries condfac/condinv but no conductivity volume, so which "
                f"chunks keep its conductivity cannot be read")
        if plan is None:
            plan = stepping._region_plan(fields, pml)  # noqa: SLF001
        if all(plan.conductive_chunks(target, sigma)):
            words.append(ALL_CHUNKS_KEPT)
            continue
        if record is None:
            raise PresenceRefusal(
                f"the conductivity presence of {target} varies per chunk and no chunk "
                f"record names this run's chunks")
        mask = plan.conductive_mask(target, sigma, numpy)
        chunk_codes = cell_chunk_codes(record, IYEE_SHIFTS[target], shape, codes)
        words.append(presence_word(mask, chunk_codes,
                                   f"the conductivity presence of {target}"))
    return (words[0], words[1], words[2])


def _record_of(fields: Any, pml: Any, chunk_layout: Any) -> Any:
    """The record a composition binds: ``chunk_layout``'s, else the region plan's division."""
    from .. import stepping  # noqa: PLC0415

    layout = chunk_layout
    if layout is None:
        layout = stepping._region_plan(fields, pml).layout  # noqa: SLF001
    return _chunk_bounds.for_layout(layout, tuple(int(n) for n in fields.grid.shape))


def presence_for(fields: Any, pml: Any, sub_step: str, chunk_bounds: Any = None,
                 chunk_layout: Any = None) -> Tuple[int, int, int]:
    """The words the array path steps ``sub_step``'s lossy targets with, for this run.

    ``chunk_bounds`` is the record the kernel binds (the composer's); ``None`` derives it
    from ``chunk_layout``, else from the region plan's division. Raises
    ``offdiag_update_e.PresenceRefusal`` by name where the words cannot reproduce the
    array path's presence.
    """
    record = (chunk_bounds if chunk_bounds is not None
              else _record_of(fields, pml, chunk_layout))
    return presence_words(fields, pml, sub_step, record, _codes_for(fields, pml))


def presence_reasons(fields: Any, pml: Any, sub_step: str, chunk_layout: Any = None,
                     chunk_bounds: Any = None) -> List[str]:
    """Why the presence words cannot be built for this run; empty when they can.

    The predicate's clause, so a covered verdict never meets a raising builder: the
    plan builder calls :func:`presence_for` again with the composer's record.
    """
    from .offdiag_update_e import PresenceRefusal  # noqa: PLC0415

    try:
        presence_for(fields, pml, sub_step, chunk_bounds, chunk_layout)
    except PresenceRefusal as exc:
        return [f"the conductivity presence is refused: {exc}"]
    except Exception as exc:  # noqa: BLE001 - unreadable presence is not covered
        return [f"the per-chunk conductivity presence could not be read ({exc!r})"]
    return []


def presence_plan_words(presence: Any, owner: str) -> Tuple[int, int, int]:
    """Validate the words a plan binds: three integers in ``0 .. 2**27 - 1``."""
    if presence is None:
        raise ValueError(
            f"{owner} was given no conductivity presence words: pass "
            f"presence_for(fields, pml, sub_step, record), or ALL_KEPT for a run whose "
            f"conductivity every chunk keeps")
    words = tuple(presence)
    if len(words) != len(PRESENCE_MEMBERS):
        raise ValueError(f"{owner}: presence carries {len(words)} words, the targets "
                         f"are {len(PRESENCE_MEMBERS)}")
    out = []
    for word in words:
        value = int(word)
        if value != word or not 0 <= value <= ALL_CHUNKS_KEPT:
            raise ValueError(f"{owner}: presence word {word!r} is not a 27-bit mask")
        out.append(value)
    return (out[0], out[1], out[2])


def presence_tensor(words: Sequence[int], device: str = "mps") -> Any:
    """The words as a plan-owned ``torch.int32`` tensor of three (outside the registry)."""
    import numpy  # noqa: PLC0415
    import torch  # noqa: PLC0415

    host = numpy.asarray(tuple(int(word) for word in words), dtype=numpy.int32)
    return torch.from_numpy(host.copy()).to(torch.device(device))


def metal_conductive_pml_curl_coverage(fields: Any, pml: Any, sub_step: str,
                                       residency: Any = None, *,
                                       chunk_layout: Any = None) -> Coverage:
    """The established numerical PML-conductivity clauses plus Metal clauses.

    Metal's own clauses: the backend, the residency declaration, and the per-chunk
    conductivity presence words (:func:`presence_reasons`, read on ``chunk_layout``):
    a division whose presence the words cannot carry is refused by name, a partial
    presence they can carry is served.
    """
    if sub_step not in CONDUCTIVE_SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(CONDUCTIVE_SUB_STEPS)}, got {sub_step!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))
    shared = shared_conductive_pml_coverage(
        _FieldsView(fields, _GridView(grid)), pml, sub_step)
    reasons = list(_metal_backend_reasons(grid))
    reasons.extend(_residency_declaration_reasons(residency))
    reasons.extend(shared.reasons)
    if shared.covered:
        reasons.extend(presence_reasons(fields, pml, sub_step, chunk_layout))
    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


class MetalConductivePmlCurlPlan(KernelPlan):
    """Persistent-mirror plan for one active-PML conductive curl sub-step.

    THE CHUNK RECORD BINDS AFTER THE SCALARS (:data:`RECORD_BUFFER`), checked against
    the plan's shape (:func:`.chunk_bounds.plan_words`) and uploaded once as a
    plan-owned ``int32`` vector on the residency's device, outside the residency
    registry: nothing on the device writes it and nothing on the host reads it back.
    The presence words bind last (:data:`PRESENCE_BUFFER`) the same way, validated by
    :func:`presence_plan_words`. ``chunk_bounds`` and ``presence`` keep the host words
    for inspection.
    """

    __slots__ = ("sub_step", "shape", "n_elem", "dtdx", "backward", "bc",
                 "conductive", "residency", "volumes", "chunk_bounds", "presence")
    REPR_FIELDS = ("sub_step", "shape", "bc", "conductive")

    def __init__(self, sub_step, shape, dtdx, codes, conductive, residency,
                 targets, auxiliaries, history, condfac, condinv, sources,
                 coefficients, functions, volumes, *, chunk_bounds,
                 presence) -> None:
        self.sub_step = sub_step
        self.shape = tuple(int(value) for value in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.dtdx = float(dtdx)
        self.backward = bool(SUB_STEPS[sub_step]["backward"])
        self.bc = tuple(int(code) for code in codes)
        self.conductive = tuple(bool(flag) for flag in conductive)
        self.residency = residency
        self.volumes = tuple(volumes)
        self.chunk_bounds = _chunk_bounds.plan_words(
            chunk_bounds, self.shape, f"MetalConductivePmlCurlPlan({sub_step})")
        self.presence = presence_plan_words(
            presence, f"MetalConductivePmlCurlPlan({sub_step})")
        device = getattr(residency, "device", "mps")
        record = _chunk_bounds.tensor(self.chunk_bounds, device)
        words = presence_tensor(self.presence, device)
        super().__init__(functions, tuple(targets) + tuple(auxiliaries) + tuple(history)
                         + tuple(condfac) + tuple(condinv) + tuple(sources)
                         + tuple(coefficients) + (self.shape[0], self.shape[1],
                                                  self.shape[2], self.n_elem, self.dtdx,
                                                  record, words))


def _functions(codes, backward, conductive, variants):
    return {mode: compile_conductive_pml_curl(codes, backward, conductive, mode)
            for mode in variants}


def plan_metal_conductive_pml_curl(
        fields: Any, pml: Any, sub_step: str, residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        functions: Optional[Dict[str, Any]] = None, *,
        chunk_bounds: Any, chunk_layout: Any = None, presence: Any = None,
        ) -> Optional[MetalConductivePmlCurlPlan]:
    """Build an admitted real conductive PML curl plan, otherwise ``None``.

    ``chunk_bounds`` is the record of the division the run steps with (the composer's
    ``StepContext.chunk_bounds``) and has no default, so every caller states which
    division it binds; ``chunk_layout`` is that division, read for the per-chunk
    conductivity presence (``None``: the region plan's division). ``presence`` is the
    gates' HOST mutation seam for the words; ``None`` builds them
    (:func:`presence_for`, on ``chunk_bounds``).
    """
    if not metal_conductive_pml_curl_coverage(fields, pml, sub_step, residency,
                                              chunk_layout=chunk_layout).covered:
        return None
    assert residency is not None
    grid = fields.grid
    spec = SUB_STEPS[sub_step]
    targets = tuple(CONDUCTIVE_SUB_STEPS[sub_step])
    codes = tuple(1 if kind == "metallic" else 0 for kind in _boundary_kinds(grid, pml))
    target_tensors = [residency.mirror(name, getattr(fields, name)) for name in targets]
    auxiliary_tensors = [residency.mirror("fu_" + name, getattr(fields, "fu_" + name))
                         for name in targets]
    conductive = conductive_targets(fields, sub_step)
    history = [
        (residency.mirror("f_cond_" + name, getattr(fields, "f_cond_" + name))
         if conductive[index] else target_tensors[index])
        for index, name in enumerate(targets)]
    condfac = [
        (residency.mirror(f"condfac_{name}:pml", fields.condfac_for(name), constant=True)
         if conductive[index] else target_tensors[index])
        for index, name in enumerate(targets)]
    condinv = [
        (residency.mirror(f"condinv_{name}:pml", fields.condinv_for(name), constant=True)
         if conductive[index] else target_tensors[index])
        for index, name in enumerate(targets)]
    sources = [residency.mirror(name, getattr(fields, name)) for name in spec["sources"]]
    coefficients = [residency.mirror(f"pml:{stem}_{axis}{spec['suffix']}",
                                     getattr(pml, f"{stem}_{axis}{spec['suffix']}"),
                                     constant=True)
                    for axis in "xyz" for stem in ("kms", "sinv")]
    selected = functions if functions is not None else _functions(
        codes, spec["backward"], conductive, contract_variants)
    words = (presence_words(fields, pml, sub_step, chunk_bounds, codes)
             if presence is None else presence)
    return MetalConductivePmlCurlPlan(
        sub_step, grid.shape, grid.dt / grid.dx, codes, conductive, residency,
        target_tensors, auxiliary_tensors, history, condfac, condinv, sources,
        coefficients, selected,
        targets + tuple("fu_" + name for name in targets)
        + tuple("f_cond_" + name for name in targets) + tuple(spec["sources"]),
        chunk_bounds=chunk_bounds, presence=words,
    )


def _arm_coverage(context: Any, slot: str) -> Coverage:
    return metal_conductive_pml_curl_coverage(
        context.fields, context.pml, slot, context.residency,
        chunk_layout=context.chunk_layout)


def _arm_plan(context: Any, slot: str) -> Optional[MetalConductivePmlCurlPlan]:
    return plan_metal_conductive_pml_curl(
        context.fields, context.pml, slot, context.residency, context.contract_variants,
        chunk_bounds=context.chunk_bounds, chunk_layout=context.chunk_layout)


def _arm_gate(context: Any) -> bool:
    return bool(getattr(context.pml, "is_active", False))


def register_arms() -> Tuple[Any, ...]:
    from . import arms  # noqa: PLC0415

    return tuple(arms.register(
        family=FAMILY, slot=slot, label="conductive PML curl",
        coverage=_arm_coverage, plan=_arm_plan, prefix="conductive PML curl: ",
        noun="conductive PML curl", gate=_arm_gate, wired=True,
    ) for slot in CONDUCTIVE_SUB_STEPS)


ARMS = register_arms()
