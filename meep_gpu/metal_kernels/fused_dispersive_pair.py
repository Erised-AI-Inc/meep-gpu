"""The FIRST fused Metal kernel: ``step_D`` welded into pole-aware ``update_E``.

THE METAL ANALOGUE OF ``triton_kernels.dispersive_fused_pair``, and the smallest
real fusion on this backend. Everything here is TRANSCRIBED from two already
certified Metal bodies plus one array-path pass; no arithmetic is re-derived, and
every line carries the file:line it came from:

* the D curl half — :data:`.shaders._CURL_TEMPLATE` (shaders.py:143-192), emitted
  through the CERTIFIED emitters :func:`.templates.ghost` and
  :func:`.templates.ownership_mask` rather than a second copy of their rules;
* the wall clear — ``stepping._zero_metal`` (stepping.py:2232-2247) restricted to
  ``D_COMPONENTS`` (stepping.py:184) through ``IYEE_SHIFTS`` (fields.py:214-219),
  which is the same table Triton's ``ZM_X``/``ZM_Y``/``ZM_Z`` block spells
  (triton_kernels/dispersive_fused_pair.py:150-158);
* the pole-aware E half — :func:`.dispersive_update_e.dispersive_e_source`
  (dispersive_update_e.py:74-82), itself the transcription of
  ``stepping.update_E`` (stepping.py:1012-1016) and
  ``stepping._apply_constitutive_pml`` (stepping.py:2130-2143).

THE SEAM THIS CLOSES, and it is exactly one line. The certified E body opens
``float source = d_in[idx];`` — it RELOADS the displacement the curl just stored.
Fused, that line becomes ``float source = v0;``: the same float32 register, never
round-tripped through device memory. A float32 value stored to a ``device float*``
and reloaded is bit-identical to the register, so the fusion is a byte-neutral
transformation BY CONSTRUCTION — and the gate MEASURES it anyway, per complete
step, because "by construction" is a hypothesis until a comparator agrees.

===========================================================================
WHY THIS IS PER COMPONENT WHEN THE TRITON PAIR IS ONE LAUNCH — MEASURED
===========================================================================

Triton's ``fused_curl_dispersive_E`` takes all three components in ONE launch.
That signature cannot exist on Metal. Counting the scalar-float bindings an
all-three-component fused D/E kernel needs:

    3 D  +  3 fu_D  +  3 H  +  6 curl coefficients (kms/sinv per axis)
  + 3 E  +  3 f_w_E  +  3 inverse epsilon  +  6 constitutive coefficients
  + 5 scalars (nx, ny, nz, n_elem, dtdx)                         =  35

and :data:`.device.MAX_BUFFER_BINDINGS` is 31 — buffer attribute indices must be
0..30 and a 32nd binding is a COMPILE ERROR, the same measured ceiling that FORCES
``float2`` volumes on the complex families (see :mod:`.templates`).

WHAT THAT DOES AND DOES NOT ESTABLISH. Thirty-five SEPARATE BINDINGS exceed the
ceiling, and the gate's ``binding_ceiling`` leg compiles that signature and
requires the failure. But this docstring previously concluded the one-launch shape
was UNBUILDABLE on Metal, and that is FALSE — measured on this host (torch 2.10.0,
metalfe-32023.850.10) on 2026-08-19:

    35 separate bindings                       -> FAILED, 'buffer' attribute
                                                  parameter is out of bounds
    30 pointers + 5 scalars in one Params&     -> COMPILES (31 bindings)
    30 pointers                                -> COMPILES

Scalars do not need a binding each; packing them into a single ``constant Params&``
buys back four slots. So the ceiling constrains the SIGNATURE, not the fusion: an
all-component one-launch shape is reachable by packing, and the reason this family
dispatches per component is the pole budget (up to eight ``q`` buffers per
component) plus that packing work, not a platform refusal. Stated as measured
rather than as a barrier, because "the platform forbids it" would have closed a
door that is open.

Per component the count is 29 (indices 0..28):

    d_out, fu, e_out, fw, g0, g1, g2, inv_e, q0..q7,
    kmx, sinvx, kmy, sinvy, kmz, sinvz, kps, kms,
    nx, ny, nz, n_elem, dtdx

so one dispatch per E component. THAT IS THE FUSION: the D curl arithmetic and the
E constitutive arithmetic execute in one kernel body with D living in a register.
It is not "one dispatch for the seam", and this docstring says so rather than
letting the family name imply a launch count the platform refuses to allow.

WHAT THE FUSION REMOVES, MEASURED rather than argued — the gate's
``separate_control`` leg runs the two ALREADY CERTIFIED products over the same
seam on identical state and records both sides:

    separate: 4 device dispatches per step (1 curl + 3 pointwise E)
              + 1 HOST round trip inside the seam on a walled run, because
                ``zero_metal_D`` stays on the array path there
    fused:    3 device dispatches per step, 0 host passes inside the seam

and all three engines — array path, separate, fused — agree word for word at
every complete step. A correct fusion is byte-neutral by construction, so the
dispatch counts are the only place the difference is visible at all.

===========================================================================
WHAT THIS FAMILY REFUSES, AND WHY EACH REFUSAL IS THE SEAM RATHER THAN TASTE
===========================================================================

The driver runs FIVE passes between ``step_D`` and ``update_E``
(driver.py:3292-3302, mirrored in :data:`.coverage.RESIDENCY_ORDER`):
``step_D`` -> electric sources -> ``fill_symmetry_bc_D`` -> ``zero_metal_D`` ->
``fill_folded_far_ghosts_D`` -> ``update_E``. A kernel that spans the seam must
therefore prove the three passes in the middle either do nothing or are carried
INLINE:

* **electric sources — REFUSED BY NAME.** The injection happens inside this seam,
  so a fused pair would consume a pre-injection D. The source inventory must be
  DECLARED; ignorance is never treated as an empty set, exactly as
  ``dispersive_fused_pair_coverage`` insists (triton_kernels/dispersive_fused_pair.py:262).
  A magnetic source is injected in the B/H half and does NOT disqualify the pair.
* **the mirror fill and the far ghost pass — REFUSED** through the E half's own
  "a mirror plane is active" clause (dispersive_update_e.py:139-141), which is
  what makes both fill slots dead for this configuration.
* **``zero_metal_D`` — CARRIED INLINE**, not refused, because it is the one pass
  whose arithmetic is a transcribable select on the value already in the register.
  ``_zero_metal`` writes zero into stored cell 0 of every component whose Yee
  shift on a walled axis is 0 (stepping.py:2244-2247), which for D is Dy/Dz on an
  x wall, Dx/Dz on a y wall and Dx/Dy on a z wall — the table
  :data:`_ZERO_METAL_ROWS` spells, and the one Triton's ZM block spells.
  ``fu_D`` is deliberately NOT masked: the array-path pass touches
  ``D_COMPONENTS`` only.

NOT WIRED, AND FOR THE REASON THE TRITON CHAIN IS NOT. ``STEP_ORDER`` assigns at
most one arm per slot and this product spans THREE of them (``step_D``,
``zero_metal_D``, ``update_E``). There is no slot it can claim without a
composition rule nothing has measured, so it registers ``wired=False``: the
composer cannot select it, and the disjointness sweep can still enumerate it.
Nothing about any existing arm's behaviour changes.

DISPATCH IS NOT WIRED EITHER. ``meep_gpu.fastpath.plan_fast_path`` still returns
``None`` on every branch; the callers here are the device gate and the tests.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..triton_kernels.coverage import Coverage, MAGNETIC_FIELD_TYPE
from . import shaders, templates
from .ade_update_p import _physical_name
from .coverage import pml_curl_coverage, zero_metal_axes
from .device import Residency, compile_source
from .dispersive_update_e import (
    E_TERMS,
    MAX_POLES,
    _poles,
    metal_dispersive_e_coverage,
)
from .. import deposit_repair as _deposit_repair

#: Does this product bracket its fused launch with the deposit repair? While False the
#: source-presence clause below refuses every in-seam deposit, exactly as it did when the
#: clause was written out here by hand. Flipping it is a claim about the PLAN this module
#: builds -- that the leading slot saves and the trailing slot restores -- and is only
#: ever changed in the same edit as that wiring. See ``deposit_repair`` for why.
#:
#: TRUE SINCE 2026-08-28, in the same edit as the ``launch.FUSED_PAIR_ARMS`` row that
#: lets the seam loop bracket this family (``launch.py:660``, the
#: ``('PML', 'dispersive PML E')`` entry). Two things had to hold and both were
#: measured rather than argued:
#:
#: * THE ARITHMETIC, and this is the half that was genuinely open. The repair's
#:   electric branch recomputes ``displacement_minus_polarization(c) *
#:   inverse_epsilon_for(c)`` (deposit_repair.py:150-152), which is exactly what a
#:   DISPERSIVE ``update_E`` reads — but every gated use of the repair to date has been
#:   against a constitutive half with no polarization state at all.
#:   ``deposit_repair.repairable`` does not refuse a pole (it refuses only an
#:   off-diagonal chi1inv and an instantaneous chi2/chi3, :98-108), and not-refused is
#:   not measured. Driven through the driver's own consult order with a live
#:   Lorentzian pole on ``Ez``, four cases (point, 24-step, line, and a deposit inside
#:   the PML shell) are byte-identical to the array path over 26 arrays — the two extra
#:   arrays being the pole's own ``P``/``P_prev``, without which the case could not tell
#:   the dispersive branch from the plain ``D * inv_eps`` one. Every unrepaired null
#:   control diverges. ``test_deposit_repair.py``'s ``DISPERSIVE_CASES`` is that
#:   measurement, and it asserts the pole is live so the case cannot pass vacuously.
#: * THE FOLD DOES NOT REACH THIS FAMILY EITHER WAY. A folded seam runs
#:   ``fill_symmetry_bc_D``/``fill_folded_far_ghosts_D`` AFTER the injection, so a
#:   deposit landing on a row one of them reads from is imaged into cells a point
#:   repair never visits -- which is what ``deposit_repair.repair_cells`` closed on
#:   2026-08-28, and what let the two FOLDED Metal families flip. This family is
#:   unaffected in both directions, measured on ``matrix.dispersive(matrix.folded())``:
#:   the conjunction refuses with "curl half: axis 1 is folded by a mirror plane" and
#:   "dispersive constitutive half: a mirror plane is active", so no configuration it
#:   admits has an image to miss or to carry.
CARRIES_DEPOSIT_REPAIR = True

FAMILY = "fused_dispersive_pair"

#: The sub-step slot this arm is registered on. It spans three; it holds a row on
#: the first, so a refusal is NAMED on the slot the fusion starts at rather than
#: being invisible to the table.
SLOT = "step_D"

#: The driver passes one launch of this plan performs, in driver order
#: (driver.py:3292-3302). Read by ``launch.plan_step`` and by the whole-step
#: gate's ``covered_passes``; declared rather than inferred from the slot name.
REPLACES: Tuple[str, ...] = ("step_D", "zero_metal_D", "update_E")

#: How many scalar-float buffers an ALL-THREE-COMPONENT fused D/E kernel would
#: need, and the ceiling it breaks. Spelled as data so the gate can compile the
#: refuted signature from the same number this docstring argues from.
ALL_COMPONENT_BINDINGS = 35

__all__ = [
    "ALL_COMPONENT_BINDINGS", "E_TERMS", "FAMILY", "MAX_POLES", "REPLACES", "SLOT",
    "MetalFusedDispersivePairPlan", "compile_fused_dispersive_pair",
    "fused_dispersive_pair_source", "metal_fused_dispersive_pair_coverage",
    "plan_metal_fused_dispersive_pair", "zero_metal_mask",
]


# ---------------------------------------------------------------------------
# The source
# ---------------------------------------------------------------------------

#: THE POLE BUFFERS ARE ``q0..q7`` HERE AND ``p0..p7`` IN THE POINTWISE FAMILY,
#: and the rename is forced rather than stylistic: the certified curl body names
#: its split-field previous value ``p0``/``p1``/``p2`` (shaders.py:180, :184, :188)
#: and the certified E body names its pole buffers ``p0``..``p7``
#: (dispersive_update_e.py:44-51). Fused, ``p0`` would mean two different things
#: in one scope on the x component. Only the POINTER SPELLING moves; every
#: arithmetic line is unchanged, which is what the gate's arithmetic mutations
#: measure.
_POLE_PREFIX = "q"

_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

kernel void fused_dispersive_pair_component(
    device float*       f__T__      [[buffer(0)]],
    device float*       u__T__      [[buffer(1)]],
    device float*       e_out    [[buffer(2)]],
    device float*       fw       [[buffer(3)]],
    device const float* g0       [[buffer(4)]],
    device const float* g1       [[buffer(5)]],
    device const float* g2       [[buffer(6)]],
    device const float* inv_e    [[buffer(7)]],
    device const float* q0       [[buffer(8)]],
    device const float* q1       [[buffer(9)]],
    device const float* q2       [[buffer(10)]],
    device const float* q3       [[buffer(11)]],
    device const float* q4       [[buffer(12)]],
    device const float* q5       [[buffer(13)]],
    device const float* q6       [[buffer(14)]],
    device const float* q7       [[buffer(15)]],
    device const float* kmx      [[buffer(16)]],
    device const float* sinvx    [[buffer(17)]],
    device const float* kmy      [[buffer(18)]],
    device const float* sinvy    [[buffer(19)]],
    device const float* kmz      [[buffer(20)]],
    device const float* sinvz    [[buffer(21)]],
    device const float* kps      [[buffer(22)]],
    device const float* kms      [[buffer(23)]],
    constant uint&      nx       [[buffer(24)]],
    constant uint&      ny       [[buffer(25)]],
    constant uint&      nz       [[buffer(26)]],
    constant uint&      n_elem   [[buffer(27)]],
    constant float&     dtdx     [[buffer(28)]],
    uint idx [[thread_position_in_grid]])
{
__BODY__
}
"""

#: Per target: the two H operands its curl differences, the curl expression, the
#: two coefficient axes its split-field recurrence takes, and the E coefficient
#: coordinate. TRANSCRIBED from shaders.py:154-190 — the curl lines are that
#: block's own text, character for character, and the recurrence pairs are
#: ``vec.hpp``'s cycle_direction as shaders.py:173-175 records it (target 0 takes
#: (y, z), target 1 (z, x), target 2 (x, y)).
_TARGETS: Tuple[Dict[str, Any], ...] = (
    {   # Dx: curl_x = dHz/dy - dHy/dz  (stepping.py:422)
        "loads": ("    float b   = g1[ii];",
                  "    float c   = g2[ii];",
                  "    float b_z = vz ? g1[oz] : 0.0f;",
                  "    float c_y = vy ? g2[oy] : 0.0f;"),
        "curl": "    float curl0 = dtdx * ((c_y - c) + (b - b_z));",
        "first": "y", "second": "z", "coordinate": "i",
    },
    {   # Dy: curl_y = dHx/dz - dHz/dx  (stepping.py:423)
        "loads": ("    float a   = g0[ii];",
                  "    float c   = g2[ii];",
                  "    float a_z = vz ? g0[oz] : 0.0f;",
                  "    float c_x = vx ? g2[ox] : 0.0f;"),
        "curl": "    float curl1 = dtdx * ((a_z - a) + (c - c_x));",
        "first": "z", "second": "x", "coordinate": "j",
    },
    {   # Dz: curl_z = dHy/dx - dHx/dy  (stepping.py:424)
        "loads": ("    float a   = g0[ii];",
                  "    float b   = g1[ii];",
                  "    float b_x = vx ? g1[ox] : 0.0f;",
                  "    float a_y = vy ? g0[oy] : 0.0f;"),
        "curl": "    float curl2 = dtdx * ((b_x - b) + (a - a_y));",
        "first": "x", "second": "y", "coordinate": "k",
    },
)

#: ``_zero_metal`` for the D components, as (target, walled axis) rows.
#:
#: A component is cleared on the wall of axis ``d`` exactly when its Yee shift
#: there is 0 (stepping.py:2256-2260, :2300). ``IYEE_SHIFTS`` (fields.py:216) gives
#: Dx (1,0,0), Dy (0,1,0), Dz (0,0,1), so the rows are the complement of the
#: diagonal — and they are the same six rows Triton's ZM block carries
#: (triton_kernels/dispersive_fused_pair.py:150-158).
_ZERO_METAL_ROWS: Tuple[Tuple[int, int, str], ...] = (
    (0, 1, "at_y"), (0, 2, "at_z"),
    (1, 0, "at_x"), (1, 2, "at_z"),
    (2, 0, "at_x"), (2, 1, "at_y"),
)


def zero_metal_mask(target: int, zero_metal: Sequence[bool],
                    zero: str = "0.0f") -> str:
    """``stepping.zero_metal_D`` for ONE target, carried inline on the register.

    The array path stores D and then overwrites the wall plane
    (stepping.py:2247); this masks the value before it is stored and before the E
    half consumes it, which is the same field and one fewer pass. ``fu_D`` is not
    masked: ``zero_metal_D`` passes ``D_COMPONENTS`` (stepping.py:2235) and the
    split-field auxiliary is not in it.

    The select spelling is :func:`.templates.ownership_mask`'s — ``v ? 0.0f : v``
    — because that is the form already measured to deliver an exact ``0.0`` on
    this platform (shaders.py rule 3).
    """
    lines = [f"    v{target} = {flag} ? {zero} : v{target};"
             for row_target, axis, flag in _ZERO_METAL_ROWS
             if row_target == target and bool(zero_metal[axis])]
    return "\n".join(lines) or "    // no walled axis clears this target"


def fused_dispersive_pair_source(codes: Sequence[int], axis: int, pole_count: int,
                                 zero_metal: Sequence[bool],
                                 contract: str = shaders.CONTRACT_OFF) -> str:
    """One specialised fused source: (boundary triple, target, poles, walls, mode).

    Every constant Triton bakes into a ``tl.constexpr`` is baked into the string
    here, for the reason :mod:`.shaders` gives: ``torch.mps.compile_shader`` takes
    a source and nothing else, so specialisation by substitution is what keeps the
    emitted arithmetic identical to the bodies this transcribes.
    """
    codes = tuple(int(code) for code in codes)
    if len(codes) != 3:
        raise ValueError(f"codes must be a per-axis triple, got {codes!r}")
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2, got {axis!r}")
    if not 0 <= int(pole_count) <= MAX_POLES:
        raise ValueError(f"pole_count must be in [0, {MAX_POLES}], got {pole_count!r}")
    zero_metal = tuple(bool(value) for value in zero_metal)
    if len(zero_metal) != 3:
        raise ValueError(f"zero_metal must be a per-axis triple, got {zero_metal!r}")
    spec = _TARGETS[axis]

    # --- the curl half, shaders.py:136-192 with backward=True (step_D) ----------
    # The decode block, the ghost rules and the ownership mask come from the
    # CERTIFIED emitters, so a change to either rule reaches this family without a
    # second transcription. `backward` is True and never a parameter: this pair is
    # the D seam, and step_B's forward strides are a different product entirely.
    curl = [
        templates.GUARD,
        "",
        "    int nxi = int(nx), nyi = int(ny), nzi = int(nz);",
        "    int nyz = nyi * nzi;",
        "    int ii  = int(idx);",
        "    int k   = ii % nzi;",
        "    int plane = ii / nzi;",
        "    int j   = plane % nyi;",
        "    int i   = plane / nyi;",
        "",
        "    // --- the ghost rule, per axis (stepping._shift_down:1787) ----------",
        "    int si = i - 1, sj = j - 1, sk = k - 1;",
        "    bool vx = true, vy = true, vz = true;",
        templates.ghost("x", codes[0], True),
        templates.ghost("y", codes[1], True),
        templates.ghost("z", codes[2], True),
        "",
        "    int ox = si * nyz + j * nzi + k;",
        "    int oy = i * nyz + sj * nzi + k;",
        "    int oz = i * nyz + j * nzi + sk;",
        "",
        "    // METALLIC serves an exact 0.0 past the wall -- Triton's `other=0.0`.",
        *spec["loads"],
        "",
        "    // --- the curl (stepping._curl_from_operands:1601): DO NOT flatten these parens",
        spec["curl"],
        "",
        "    // --- ownership mask (stepping._mask_non_owned_cells:1865) ----------",
        "    bool at_x = (i == 0), at_y = (j == 0), at_z = (k == 0);",
    ]
    # ONE emitter, filtered to this target. `ownership_mask` returns the rows for
    # all three D targets; taking the ones that name this curl keeps the text the
    # certified emitter's own rather than a second copy of its table.
    mask = [line for line in templates.ownership_mask(codes, True).splitlines()
            if line.strip().startswith(f"curl{axis} ")]
    curl.extend(mask or [f"    // no metallic axis masks curl{axis}"])

    first, second = spec["first"], spec["second"]
    coordinate = ("i", "j", "k")
    axis_coordinate = {"x": "i", "y": "j", "z": "k"}
    curl.extend([
        "",
        "    // --- split-field recurrence (stepping._apply_pml_update:1905) ------",
        f"    float km_{first} = km{first}[{axis_coordinate[first]}], "
        f"si_{first} = sinv{first}[{axis_coordinate[first]}];",
        f"    float km_{second} = km{second}[{axis_coordinate[second]}], "
        f"si_{second} = sinv{second}[{axis_coordinate[second]}];",
        "",
        f"    float p{axis} = u{axis}[ii];",
        f"    float n{axis} = ((p{axis} * km_{first}) - curl{axis}) * si_{first};",
        f"    float v{axis} = (((f{axis}[ii] * km_{second}) + n{axis}) - p{axis})"
        f" * si_{second};",
        "",
        f"    u{axis}[ii] = n{axis};",
        "",
        "    // --- zero_metal_D, carried inline (stepping._zero_metal:2232) ------",
        zero_metal_mask(axis, zero_metal),
        f"    f{axis}[ii] = v{axis};",
    ])

    # --- the pole-aware E half, dispersive_update_e.py:74-82 -------------------
    # ONE line differs from that body and it is THE SEAM: `float source = d_in[idx]`
    # becomes the live register. Everything after it is character for character.
    chain = [f"    source = source - {_POLE_PREFIX}{index}[idx];"
             for index in range(int(pole_count))]
    electric = [
        "",
        "    // --- pole-aware update_E (stepping.update_E:983-987) ---------------",
        "    float prev = fw[idx];",
        "    // THE SEAM: the register step_D just wrote, not a reload of d_in[idx].",
        f"    float source = v{axis};",
        *chain,
        "    float src = source * inv_e[idx];",
        "    fw[idx] = src;",
        "    float value = e_out[idx];",
        f"    value = value + kps[{coordinate[axis]}] * src;",
        f"    value = value - kms[{coordinate[axis]}] * prev;",
        "    e_out[idx] = value;",
    ]

    return templates.substitute(_TEMPLATE, {
        "__CONTRACT__": templates.contraction_pragma(contract),
        "__T__": str(axis),
        "__BODY__": "\n".join(curl + electric),
    })


def compile_fused_dispersive_pair(codes: Sequence[int], axis: int, pole_count: int,
                                  zero_metal: Sequence[bool],
                                  contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised entry point for one (boundaries, target, poles, walls, mode)."""
    return compile_source(fused_dispersive_pair_source(
        codes, axis, pole_count, zero_metal, contract)).fused_dispersive_pair_component


def refuted_all_component_source(contract: str = shaders.CONTRACT_OFF) -> str:
    """The 35-binding all-three-component signature this platform REFUSES.

    Not shipped and not launchable: it exists so the gate can compile it and
    require the failure, which is what turns :data:`ALL_COMPONENT_BINDINGS` from an
    argument into a measurement. The body is a trivial touch of every buffer —
    what is being measured is the SIGNATURE, and a body the compiler could drop
    would let dead-code elimination decide the answer.
    """
    names = (["f0", "f1", "f2", "u0", "u1", "u2"]
             + ["e0", "e1", "e2", "w0", "w1", "w2"])
    reads = (["g0", "g1", "g2", "ie0", "ie1", "ie2",
              "kmx", "sinvx", "kmy", "sinvy", "kmz", "sinvz",
              "kp0", "km0", "kp1", "km1", "kp2", "km2"])
    lines: List[str] = []
    slot = 0
    for name in names:
        lines.append(f"    device float*       {name:<8}[[buffer({slot})]],")
        slot += 1
    for name in reads:
        lines.append(f"    device const float* {name:<8}[[buffer({slot})]],")
        slot += 1
    for name in ("nx", "ny", "nz", "n_elem"):
        lines.append(f"    constant uint&      {name:<8}[[buffer({slot})]],")
        slot += 1
    lines.append(f"    constant float&     dtdx    [[buffer({slot})]],")
    slot += 1
    assert slot == ALL_COMPONENT_BINDINGS, (slot, ALL_COMPONENT_BINDINGS)
    body = "\n".join(f"    {name}[idx] = {name}[idx] * dtdx;" for name in names)
    reads_body = " + ".join(f"{name}[0]" for name in reads)
    return "\n".join((
        "#include <metal_stdlib>",
        "using namespace metal;",
        "",
        templates.contraction_pragma(contract),
        "",
        "kernel void refuted_all_component(",
        *lines,
        "    uint idx [[thread_position_in_grid]])",
        "{",
        "    if (idx >= n_elem) { return; }",
        f"    float touch = {reads_body};",
        "    f0[idx] = f0[idx] + touch * float(nx + ny + nz);",
        body,
        "}",
        "",
    ))


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def metal_fused_dispersive_pair_coverage(fields: Any, pml: Any, sources: Any = None,
                                         residency: Any = None) -> Coverage:
    """May ONE dispatch per component span ``step_D`` -> wall -> ``update_E``?

    A conjunction of the two halves' OWN certified predicates plus the seam
    clauses. Nothing is weakened: a configuration either half refuses is refused
    here with that half's reasons, prefixed so a reader can tell which side said
    it. That is the same construction ``dispersive_fused_pair_coverage`` uses
    (triton_kernels/dispersive_fused_pair.py:249-283) and the same one
    :mod:`.complex_dispersive_spine` uses to conjoin a B/D/H body with its E
    companion.
    """
    reasons: List[str] = []

    curl = pml_curl_coverage(fields, pml, "step_D", residency)
    if not curl.covered:
        reasons.extend(f"curl half: {reason}" for reason in curl.reasons)
    electric = metal_dispersive_e_coverage(fields, pml, residency)
    if not electric.covered:
        reasons.extend(f"dispersive constitutive half: {reason}"
                       for reason in electric.reasons)

    # THE SEAM. An electric source is injected BETWEEN the two halves
    # (driver.py:3292-3296), so a fused pair would consume a pre-injection D.
    # IGNORANCE IS NOT AN EMPTY SET: `Fields` does not hold the source list, so a
    # predicate that inferred "no sources" from not being told would be the
    # over-covering refusal this clause exists to prevent.
    reasons.extend(_deposit_repair.seam_source_reasons(
        fields, sources, 'D',
        undeclared=(
            "the source set was not declared: this predicate cannot infer an empty "
            "electric source seam from Fields"
        ),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) is electric: the "
            f"driver injects it BETWEEN step_D and update_E"),
        carries_repair=CARRIES_DEPOSIT_REPAIR))

    # zero_metal_D is CARRIED, so the grid must be able to answer which axes are
    # walled. `zero_metal_axes` asks `has_metallic`/`is_metallic`/`is_mirrored`;
    # a grid that cannot answer would silently be treated as unwalled, which is a
    # plane of wrong values rather than a crash.
    grid = getattr(fields, "grid", None)
    if grid is None:
        reasons.append("fields carries no grid")
    else:
        for name in ("has_metallic", "is_metallic", "is_mirrored"):
            if getattr(grid, name, None) is None:
                reasons.append(
                    f"grid does not expose {name}; zero_metal_D cannot be carried "
                    f"inline")
    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Entry:
    """One E component's whole fused binding set, resolved once at plan time."""

    component: str
    displacement_name: str
    axis: int
    pole_count: int
    displacement: Any
    auxiliary_curl: Any
    target: Any
    auxiliary_e: Any
    inverse_epsilon: Any
    kps: Any
    kms: Any
    states: Tuple[Any, ...]


class MetalFusedDispersivePairPlan:
    """Three dispatches that perform four sub-step passes, D never leaving a register.

    ``launches_per_run`` is THREE and is declared rather than inherited: the
    whole-step arbiter asserts the exact per-cycle launch count on every filled
    slot, and that assertion is what stops a slot passing by not executing.
    """

    performs_device_work = True
    replaces_sub_steps = REPLACES
    launches_per_run = len(E_TERMS)

    #: The two binding groups every entry shares. Resolved once by the builder
    #: rather than recomputed per launch: nothing on the launch path allocates
    #: (launch.py:12-14).
    __slots__ = ("entries", "residency", "volumes", "codes", "zero_metal",
                 "shape", "dtdx", "_functions", "_tensors", "_magnetic",
                 "_curl_coefficients", "launches", "runs")

    def __init__(self, entries: Sequence[_Entry], residency: Residency,
                 volumes: Sequence[str], codes: Sequence[int],
                 zero_metal: Sequence[bool], shape: Sequence[int], dtdx: float,
                 magnetic: Sequence[Any], curl_coefficients: Sequence[Any],
                 functions: Mapping[Tuple[str, int], Any],
                 tensors: Mapping[int, Any]) -> None:
        self._magnetic = tuple(magnetic)
        self._curl_coefficients = tuple(curl_coefficients)
        self.entries = tuple(entries)
        self.residency = residency
        self.volumes = tuple(dict.fromkeys(volumes))
        self.codes = tuple(int(code) for code in codes)
        self.zero_metal = tuple(bool(value) for value in zero_metal)
        self.shape = tuple(int(n) for n in shape)
        # float(): the array path multiplies a float32 volume by a Python float
        # and Metal binds a Python float into `constant float&` the same way, so
        # the two scalars are the same bits (launch.py:169-174).
        self.dtdx = float(dtdx)
        self._functions = dict(functions)
        self._tensors = dict(tensors)
        self.launches = 0
        self.runs = 0

    @property
    def variants(self) -> Tuple[str, ...]:
        return tuple(sorted({key[0] for key in self._functions}))

    def run(self, contract: Optional[str] = None) -> None:
        """One complete D-half seam, in place, against the mirrors.

        A mode the plan was NOT built with RAISES rather than silently launching
        the pinned one, for the reason :class:`.plans.KernelPlan` gives: a guard
        argument that quietly did nothing would make the gate's most important leg
        vacuous.

        THE POLE POINTERS ARE RESOLVED AT LAUNCH, NOT AT PLAN TIME. ``update_P``
        rotates ``P``/``P_prev``/``_scratch`` after every recurrence, so the
        PHYSICAL array holding a component's current P changes between steps while
        the mirror registry does not. Reading ``state.P[...]`` here and looking the
        mirror up by object identity is the same discipline
        ``MetalDispersiveEPlan.run`` uses (dispersive_update_e.py:212-215).
        """
        mode = shaders.CONTRACT_OFF if contract is None else contract
        for entry in self.entries:
            key = (mode, entry.axis)
            function = self._functions.get(key)
            if function is None:
                raise KeyError(
                    f"this plan holds no {mode!r} variant (it was built with "
                    f"{self.variants}); build it with contract_variants={(mode,)} "
                    f"rather than launching the pinned one")
            poles = tuple(state.P[entry.component] for state in entry.states)
            if len(poles) != entry.pole_count:
                raise RuntimeError(
                    f"{entry.component} resolved {len(poles)} poles, not the "
                    f"compiled count {entry.pole_count}")
            # THE UNUSED POLE SLOTS BIND THE INVERSE EPSILON, NOT THE DISPLACEMENT.
            # The pointwise family pads with `entry.displacement`
            # (dispersive_update_e.py:214) because its kernel only READS D. This
            # one WRITES D, so padding with it would alias a written buffer into a
            # `device const float*` in the same dispatch. Inverse epsilon is a
            # constant mirror nothing writes, and the padded slots are never read
            # by a source specialised to `pole_count`.
            slots = poles + (entry.inverse_epsilon,) * (MAX_POLES - len(poles))
            function(
                self._tensors[id(entry.displacement)],
                self._tensors[id(entry.auxiliary_curl)],
                self._tensors[id(entry.target)],
                self._tensors[id(entry.auxiliary_e)],
                *(self._tensors[id(value)] for value in self._magnetic),
                self._tensors[id(entry.inverse_epsilon)],
                *(self._tensors[id(value)] for value in slots),
                *(self._tensors[id(value)] for value in self._curl_coefficients),
                self._tensors[id(entry.kps)], self._tensors[id(entry.kms)],
                self.shape[0], self.shape[1], self.shape[2],
                int(entry.displacement.size), self.dtdx,
            )
            self.launches += 1
        self.runs += 1

    def describe(self) -> str:
        return (f"MetalFusedDispersivePairPlan(shape={self.shape}, "
                f"poles={tuple(e.pole_count for e in self.entries)}, "
                f"bc={self.codes}, zero_metal={self.zero_metal}, "
                f"variants={self.variants})")

    def __repr__(self) -> str:
        return self.describe()


def plan_metal_fused_dispersive_pair(
        fields: Any, pml: Any, sources: Any = None,
        residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        functions: Optional[Mapping[Tuple[str, int], Any]] = None,
        ) -> Optional[MetalFusedDispersivePairPlan]:
    """Build the fused D/E plan, or ``None`` when the seam is refused.

    ``None`` is the only refusal, for the reason :func:`.launch.plan_pml_curl`
    gives: a configuration this kernel does not carry must fall back to the array
    path, never raise into a caller that would otherwise have stepped correctly.

    ``functions`` is the mutation seam. The gate compiles a deliberately broken
    copy of the shipped source and hands it here; dropping the argument is not a
    silent slowdown, it is a silent DISARMING — every mutation leg would then
    launch the shipped kernel and report the defect as uncaught.
    """
    if not metal_fused_dispersive_pair_coverage(
            fields, pml, sources, residency).covered:
        return None
    from ..stepping import _boundary_kinds  # noqa: PLC0415

    grid = fields.grid
    codes = tuple(1 if kind == "metallic" else 0
                  for kind in _boundary_kinds(grid, pml))
    walls = zero_metal_axes(grid)
    order = _poles(fields)

    tensors: Dict[int, Any] = {}
    volumes: List[str] = []
    entries: List[_Entry] = []
    required: set = set()

    def bind(name: str, host: Any, constant: bool = False) -> Any:
        tensors[id(host)] = residency.mirror(name, host, constant=constant)
        volumes.append(name)
        return host

    # The curl's magnetic operands and integer-lattice coefficients are SHARED by
    # all three dispatches — one mirror each, which is the whole point of the
    # residency layer (device.py:157-164).
    magnetic = tuple(bind(name, getattr(fields, name))
                     for name in ("Hx", "Hy", "Hz"))
    curl_coefficients = tuple(
        bind(f"pml:{stem}_{axis}", getattr(pml, f"{stem}_{axis}"), constant=True)
        for axis in "xyz" for stem in ("kms", "sinv"))

    for component, displacement_name, axis, axis_name in E_TERMS:
        displacement = bind(displacement_name, getattr(fields, displacement_name))
        auxiliary_curl = bind("fu_" + displacement_name,
                              getattr(fields, "fu_" + displacement_name))
        target = bind(component, getattr(fields, component))
        auxiliary_e = bind("f_w_" + component, getattr(fields, "f_w_" + component))
        inverse = bind(f"inv_eps_{component}",
                       fields.inverse_epsilon_for(component), constant=True)
        # The E half takes the HALF-INTEGER lattice (stepping.py:1015) and the curl
        # half the integer one (stepping.py:948 vs :1015). The kernel takes both and
        # never asks which is which, so a swap here is a silent half-cell error in
        # the absorber profile; the gate carries a mutation for exactly it.
        kps = bind(f"pml:kps_{axis_name}_h", getattr(pml, f"kps_{axis_name}_h"),
                   constant=True)
        kms = bind(f"pml:kms_{axis_name}_h", getattr(pml, f"kms_{axis_name}_h"),
                   constant=True)
        states = order[component]
        entries.append(_Entry(component, displacement_name, axis, len(states),
                              displacement, auxiliary_curl, target, auxiliary_e,
                              inverse, kps, kms, states))
        required.update((mode, axis) for mode in contract_variants)

    for state_index, state in enumerate(tuple(getattr(fields, "polarizations", ())
                                              or ())):
        for component in tuple(state.driven()):
            for label, host in ((f"P:{component}", state.P[component]),
                                (f"P_prev:{component}", state.P_prev[component])):
                bind(_physical_name(state_index, label, host), host)
        scratch = state._scratch
        bind(_physical_name(state_index, "scratch", scratch), scratch)

    selected = dict(functions or {})
    by_axis = {entry.axis: entry.pole_count for entry in entries}
    for key in required:
        if key not in selected:
            selected[key] = compile_fused_dispersive_pair(
                codes, key[1], by_axis[key[1]], walls, key[0])

    return MetalFusedDispersivePairPlan(
        entries, residency, volumes, codes, walls, grid.shape,
        grid.dt / grid.dx, magnetic, curl_coefficients, selected, tensors)


# ---------------------------------------------------------------------------
# Registration — NOT WIRED
# ---------------------------------------------------------------------------

def _arm_coverage(context: Any, slot: str) -> Coverage:
    if slot != SLOT:
        return Coverage(False, (f"fused dispersive pair cannot fill {slot}",))
    return metal_fused_dispersive_pair_coverage(
        context.fields, context.pml, context.sources, context.residency)


def _arm_plan(context: Any, slot: str) -> Optional[MetalFusedDispersivePairPlan]:
    if slot != SLOT:
        return None
    return plan_metal_fused_dispersive_pair(
        context.fields, context.pml, context.sources, context.residency,
        context.contract_variants)


def register_arms() -> Tuple[Any, ...]:
    """One row on ``step_D``, ``wired=False``.

    THE FLAG IS THE WHOLE COMPOSITION STORY. ``plan_step`` assigns at most one arm
    per slot and this product spans three, so there is no slot it could claim
    without a composition rule nothing has measured — the same open question
    ``triton_kernels.launch`` records for the fused dispersive chain. Registering
    unwired keeps it ENUMERABLE for the disjointness sweep (``arms.registered``)
    while ``arms.arms_for`` skips it, so ``plan_step`` cannot select it and no
    existing arm's selection changes.
    """
    from . import arms  # noqa: PLC0415
    return (arms.register(FAMILY, SLOT, "fused dispersive D/E pair",
                          _arm_coverage, _arm_plan,
                          prefix="fused dispersive D/E pair: ",
                          noun="fused PML dispersive D-curl/stored-E pair",
                          wired=False, replaces=REPLACES),)


ARMS = register_arms()
