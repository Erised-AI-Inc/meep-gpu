"""The FOLDED dispersive pair: ``step_D`` -> both fills -> wall -> pole-aware ``update_E``.

THE INTERSECTION THE TWO SHIPPED D-SIDE PAIRS LEAVE UNCOVERED, and the largest one
left on the folded side of the Metal fusion board: :mod:`.folded_fused_pair` welds
the folded curl to the ORDINARY E half, and :mod:`.fused_dispersive_pair` welds the
unfolded curl to the POLE-AWARE one. A folded grid carrying live electric poles is
served by neither, and the corpus has four such instances — every one of them a
``TestLoadDump`` row with five poles and one Y mirror.

EVERYTHING HERE IS TRANSCRIBED, and the three parts come from three already
certified places:

* the folded D curl and BOTH mirror fills — :mod:`.folded_fused_pair`'s own
  geometry, IMPORTED rather than re-derived: :func:`.folded_fused_pair.near_fill_axes`,
  :func:`.folded_fused_pair.far_fill_axes`,
  :func:`.folded_fused_pair.carried_destinations`,
  :func:`.folded_fused_pair.zero_metal_lines`,
  and :func:`.folded_fused_pair._component_parity`. Those functions answer WHICH
  cell images which and with what sign; getting one of them independently wrong
  here would be a plane of wrong values that agrees with nothing, and the whole
  point of importing them is that there is one answer;
* the curl arithmetic per target — :data:`.fused_dispersive_pair._TARGETS`,
  IMPORTED, which is ``shaders._CURL_TEMPLATE``'s own text split per component;
* the pole-aware E half — :func:`.dispersive_update_e.dispersive_e_source`'s body
  (dispersive_update_e.py:74-82), the transcription of ``stepping.update_E``
  (stepping.py:1012-1016) and ``stepping._apply_constitutive_pml``
  (stepping.py:2130-2143).

===========================================================================
PER COMPONENT, AND THE POLE BUDGET IS WHY — not a platform refusal
===========================================================================

:mod:`.folded_fused_pair` takes all three components in ONE launch at exactly 31
bindings, the measured ceiling. Adding the pole budget to that shape is
30 + 3 x 8 = 54 pointers, so this family dispatches PER COMPONENT, for the same
reason and with the same honesty :mod:`.fused_dispersive_pair` states: the budget,
not a refusal. Per component the count is 24 pointers plus one packed
``constant Params&``:

    f{t}, u{t}                          the D target and its split-field auxiliary
    g0, g1, g2                          the three H operands the curl differences
    e_out, fw, inv_e                    the E target, its workspace, 1/eps
    q0..q7                              the pole budget
    kmx, sinvx, kmy, sinvy, kmz, sinvz  the curl's INTEGER lattice
    kps, kms                            the E half's HALF-INTEGER pair
  = 24, + Params = :data:`PACKED_BINDINGS` = 25

seven under the ceiling, which is what pays for the three reflect rows riding
inside ``Params`` — the far carry costs no binding here either.

THE THREE H OPERANDS ARE ALL BOUND even though a given target differences only two
of them. One buffer layout for all three specialisations is what lets the plan bind
``Hx``/``Hy``/``Hz`` once and hand the same three mirrors to every dispatch; the
unused one is never read by a source specialised to that target.

===========================================================================
THE SEAM, AND WHAT SITS INSIDE IT — measured from the driver, not assumed
===========================================================================

``step_D`` -> ELECTRIC SOURCES -> ``fill_symmetry_bc_D`` -> ``zero_metal_D`` ->
``fill_folded_far_ghosts_D`` -> ``update_E`` (driver.py:3292-3303).

* **the electric source slot — CARRIED**, through the deposit repair. See
  :data:`CARRIES_DEPOSIT_REPAIR`. All four corpus rows this pair reaches carry an
  electric source, so without the bracket the cell would be worth ONE row at most;
  it is the product, not an optimisation.
* **``fill_symmetry_bc_D`` and ``fill_folded_far_ghosts_D`` — CARRIED INLINE**,
  through the OWNERSHIP INVERSION :mod:`.folded_fused_pair` established and this
  family reuses unchanged: the destination cell's own thread stops after ``fu`` and
  the SOURCE thread writes the destination from its live register. There is no
  device-wide barrier inside one launch, so the carve-out is what makes the carry
  legal at all rather than a race.
* **``zero_metal_D`` — CARRIED INLINE** on the register, through
  :func:`.folded_fused_pair.zero_metal_lines`, at the same point in the driver's
  order: after the near fill (:3300) and before the far one (:3302).

NOT WIRED AS AN ARM: ``plan_step`` assigns at most one arm per slot and this
product spans five. It registers ``wired=False`` and reaches its two slots through
``launch.FUSED_PAIR_ARMS``.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from ..triton_kernels.coverage import Coverage, _call, zero_metal_axes
from ..triton_kernels.symmetry import MIRROR_SOURCE_INDEX, folded_axis_kinds
from . import folded_dispersive_update_e as _folded_e
from . import shaders, templates
from .ade_update_p import _physical_name
from .device import MAX_BUFFER_BINDINGS, Residency, compile_source
from .dispersive_update_e import E_TERMS, MAX_POLES, _poles
from .folded_fused_pair import (
    _COORDINATE,
    _LAST,
    _LAST_FLAG,
    _RECURRENCE_AXES,
    _REFLECT,
    _STRIDE,
    _component_parity,
    _far_carry_reasons,
    _mirror_phase_reasons,
    carried_destinations,
    far_fill_axes,
    near_fill_axes,
    zero_metal_lines,
)
from .fused_dispersive_pair import _TARGETS, _ZERO_METAL_ROWS
from .symmetry import (
    MIRROR_CODES,
    _parity_spelling,
    _reduced_codes,
    folded_composition_curl_coverage,
    folded_top_plane_mask,
)
from .. import deposit_repair as _deposit_repair

#: Does this product bracket its fused launch with the deposit repair? While False the
#: source-presence clause below refuses every in-seam deposit. Flipping it is a claim
#: about the PLAN this module builds -- that the leading slot saves and the trailing
#: slot restores -- and is only ever changed in the same edit as that wiring.
#:
#: TRUE FROM THIS FAMILY'S FIRST COMMIT, in the same edit as its
#: ``launch.FUSED_PAIR_ARMS`` row. Everything the claim needs was built and measured
#: for :mod:`.folded_fused_pair` on 2026-08-28 and none of it is storage- or
#: pole-specific:
#:
#: * THE FOLDED SEAM. ``deposit_repair.repair_cells`` (deposit_repair.py:217-258)
#:   hands ``save``/``apply`` the CLOSURE of the cells ``fill_symmetry_bc_D`` and
#:   ``fill_folded_far_ghosts_D`` image each deposit point into -- the inverse of the
#:   forward carry :func:`carried_destinations` implements here. A POINT repair is
#:   measurably wrong on a folded seam and that measurement is in that module.
#: * THE CONSTITUTIVE SHAPE. ``deposit_repair.apply`` recomputes
#:   ``field + kps*fresh - kms*fw_prev``, which is EXACTLY the pole-aware E half's
#:   accumulation (dispersive_update_e.py:78-82) -- the poles enter through ``fresh``,
#:   as ``(D - sum P) * inv_eps``, and the repair recomputes that from the repaired D
#:   like any other source. Measured rather than argued: ``repairable`` returns True
#:   on ``metal_composition_matrix.dispersive(matrix.folded())`` with the layer
#:   supplied, which is the call that runs the absorber clauses.
#: * WHAT IS STILL REFUSED BY NAME, inherited: a source publishing no deposit index,
#:   the cylindrical r = 0 axis, and a folded axis storing no more than
#:   ``MIRROR_SOURCE_INDEX`` cells.
CARRIES_DEPOSIT_REPAIR = True

FAMILY = "folded_fused_dispersive_pair"

#: The sub-step slot this arm holds a row on. It spans five; a refusal is NAMED on
#: the slot the fusion starts at rather than being invisible to the table.
SLOT = "step_D"

#: The driver passes one RUN of this plan performs, in driver order
#: (driver.py:3292-3303). Declared, never inferred from the slot name.
REPLACES: Tuple[str, ...] = ("step_D", "fill_D", "zero_metal_D",
                             "fill_folded_far_ghosts_D", "update_E")

#: The shipped per-component binding count: 24 pointers + one packed
#: ``constant Params&``.
PACKED_BINDINGS = 25

#: What the same 24 pointers need with the eight scalars bound SEPARATELY: 32, one
#: over the ceiling. :func:`refuted_separate_scalar_source` builds it and the gate
#: requires the compile failure, so the packing is a measurement rather than a
#: precaution -- and it is a NARROW miss, which is exactly the case a comment would
#: get wrong.
SEPARATE_SCALAR_BINDINGS = 32

#: What an ALL-THREE-COMPONENT folded dispersive kernel would need: the folded real
#: pair's 30 pointers plus three components' pole budgets. Far over the ceiling, and
#: the reason this family dispatches per component.
ALL_COMPONENT_POINTERS = 30 + 3 * MAX_POLES

#: The stored index the near fill images, re-exported so a reader does not have to
#: chase ``triton_kernels.symmetry`` for the fold's magic 2.
NEAR_SOURCE_INDEX = MIRROR_SOURCE_INDEX

if PACKED_BINDINGS > MAX_BUFFER_BINDINGS:  # pragma: no cover - a design invariant
    raise RuntimeError(
        f"the folded fused dispersive pair binds {PACKED_BINDINGS} buffers and "
        f"Metal's ceiling on this toolchain is {MAX_BUFFER_BINDINGS}")

#: THE POLE BUFFERS ARE ``q0..q7`` HERE AND ``p0..p7`` IN THE POINTWISE FAMILY, and
#: the rename is forced rather than stylistic, exactly as it is in
#: :mod:`.fused_dispersive_pair`: the curl body names its split-field previous value
#: ``p{t}``, so fused, ``p0`` would mean two different things in one scope on the x
#: component. Only the POINTER SPELLING moves.
_POLE_PREFIX = "q"

__all__ = [
    "ALL_COMPONENT_POINTERS", "CARRIES_DEPOSIT_REPAIR", "FAMILY",
    "NEAR_SOURCE_INDEX", "PACKED_BINDINGS", "REPLACES", "SEPARATE_SCALAR_BINDINGS",
    "SLOT", "MetalFoldedFusedDispersivePairPlan",
    "compile_folded_fused_dispersive_pair", "folded_fused_dispersive_pair_source",
    "metal_folded_fused_dispersive_pair_coverage",
    "plan_metal_folded_fused_dispersive_pair", "refuted_separate_scalar_source",
    "register_arms",
]


# ---------------------------------------------------------------------------
# The source
# ---------------------------------------------------------------------------

_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

// EIGHT SCALARS, ONE BINDING. The three reflect rows the far carry needs
// (stepping._far_reflect_rows:1661) ride inside this record rather than as three
// more buffers, so carrying fill_folded_far_ghosts_D costs nothing against the
// platform's 31-binding ceiling (device.py:72).
struct Params { uint nx; uint ny; uint nz; uint n_elem; float dtdx;
                int rx; int ry; int rz; };

kernel void folded_fused_dispersive_pair_component(
    device float*       f__T__   [[buffer(0)]],
    device float*       u__T__   [[buffer(1)]],
    device float*       e_out    [[buffer(2)]],
    device float*       fw       [[buffer(3)]],
    device const float* g0       [[buffer(4)]],
    device const float* g1       [[buffer(5)]],
    device const float* g2       [[buffer(6)]],
    device const float* ie__T__  [[buffer(7)]],
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
    device const float* kp__T__  [[buffer(22)]],
    device const float* km__T__  [[buffer(23)]],
    constant Params&    prm      [[buffer(24)]],
    uint idx [[thread_position_in_grid]])
{
    // THE EIGHT SCALARS ARE UNPACKED INTO THE TRANSCRIBED BODIES' OWN NAMES, once,
    // before any of the lifted text runs.
    uint nx = prm.nx, ny = prm.ny, nz = prm.nz, n_elem = prm.n_elem;
    float dtdx = prm.dtdx;
    int reflect_x = prm.rx, reflect_y = prm.ry, reflect_z = prm.rz;
    (void)reflect_x; (void)reflect_y; (void)reflect_z;

__BODY__
}
"""


def _constitutive_block(target: int, index: str, value: str, indent: str,
                        prefix: str, pole_count: int,
                        kp: Optional[str] = None,
                        km: Optional[str] = None) -> List[str]:
    """The POLE-AWARE ``update_E`` for one component at one cell.

    TRANSCRIBED from :func:`.dispersive_update_e.dispersive_e_source`'s body
    (dispersive_update_e.py:74-82), which is itself ``stepping.update_E``
    (stepping.py:1012-1016) plus ``_apply_constitutive_pml`` (:2130-2143). ONE line
    differs and it is THE SEAM: ``float source = d_in[idx];`` becomes the live
    register, and every line after it is the pointwise body's own.

    THE POLE SUBTRACTION CHAIN IS LEFT-ASSOCIATED AND BOUNDED, in the pointwise
    family's own order: ``source = source - q0[..]`` then ``- q1[..]``, one
    statement each, because that is where the array path's float32 round points are
    (``fields.displacement_minus_polarization``). Flattening the chain into one
    expression changes results.

    ``kp``/``km`` default to the thread's own pair, which is what a NEAR ghost
    takes: it differs from its source only along an axis that is never the
    component's own, and the coefficient is indexed on the component's own axis. A
    FAR ghost moves ALONG that axis and passes its own pair
    (:func:`.folded_fused_pair._far_coefficient_lines`) -- the same asymmetry the D
    geometry forces on the ordinary folded pair, imported from it rather than
    re-derived.

    THE POLES ARE READ AT THE DESTINATION INDEX, not at the source thread's. The
    array path runs ``update_E`` over every stored cell AFTER the fills have written
    the ghost, so a ghost's E is formed from the ghost's own ``P`` -- and the fills
    write ``field`` alone (stepping.py:1497-1498), never ``P``. Reading ``q`` at the
    owning thread's index would apply the source cell's polarization to the image,
    which on a folded grid with live poles is a smooth, plausible, wrong answer.
    """
    kp = f"kp{target}" if kp is None else kp
    km = f"km{target}" if km is None else km
    coordinate = _COORDINATE[target]
    if index != "ii":
        # The destination's own coordinate on the indexed axis, recovered from its
        # flat index rather than reused from the owner's: see the docstring.
        coordinate = f"{prefix}coord"
    lines = [
        f"{indent}float {prefix}prev = fw[{index}];",
        f"{indent}float {prefix}source = {value};",
    ]
    lines.extend(f"{indent}{prefix}source = {prefix}source - "
                 f"{_POLE_PREFIX}{pole}[{index}];"
                 for pole in range(int(pole_count)))
    lines.extend([
        f"{indent}float {prefix}src = {prefix}source * ie{target}[{index}];",
        f"{indent}fw[{index}] = {prefix}src;",
        f"{indent}float {prefix}acc = e_out[{index}];",
        f"{indent}{prefix}acc = {prefix}acc + {kp}[{coordinate}] * {prefix}src;",
        f"{indent}{prefix}acc = {prefix}acc - {km}[{coordinate}] * {prefix}prev;",
        f"{indent}e_out[{index}] = {prefix}acc;",
    ])
    return lines


def _carry_blocks(target: int, near: Sequence[int], far: Sequence[int],
                  phases: Sequence[int], zero_metal: Sequence[bool],
                  pole_count: int, indent: str) -> List[str]:
    """Every imaged ghost this thread owns, then the pole-aware ``update_E`` at each.

    THE GEOMETRY IS :mod:`.folded_fused_pair`'S, IMPORTED: which destinations exist
    (:func:`.folded_fused_pair.carried_destinations`), the parity of each
    (:func:`.folded_fused_pair._component_parity`), where the wall clear sits
    relative to each fill (:func:`.folded_fused_pair.zero_metal_lines`) and which
    coefficient pair a far destination takes
    (:func:`.folded_fused_pair._far_coefficient_lines`). Only the CONSTITUTIVE
    BLOCK differs, and only by the pole chain.

    THE TWO FILLS TAKE OPPOSITE ORDERS AROUND THE WALL CLEAR, unchanged from the
    ordinary folded pair because the driver's order is unchanged: the NEAR ghost is
    ``mask(parity * v)`` (the fill at :3300 precedes the clear at :3301) and the FAR
    ghost is ``parity * mask(v)`` with no clear line of its own (:3302 follows it),
    inheriting the owning thread's clear by reading ``v`` after those lines.
    """
    walls = tuple(axis for row_target, axis, _ in _ZERO_METAL_ROWS
                  if row_target == target and bool(zero_metal[axis]))
    overlap = sorted(set(walls) & {int(axis) for axis in far})
    if overlap:
        raise AssertionError(
            f"component {target} images a FAR ghost along axes {tuple(far)} that "
            f"zero_metal_D also clears on {overlap}; the ghost would no longer "
            f"inherit its source thread's clear")
    if set(walls) & {int(axis) for axis in near}:
        raise AssertionError(
            f"component {target} images a NEAR ghost on folded axes {tuple(near)} "
            f"that zero_metal_D also clears on {walls}; stepping._zero_metal skips "
            f"a folded axis (stepping.py:2282-2284), so the two tables have drifted")
    inner = indent + "    "
    lines: List[str] = []
    for subset, carries_far in carried_destinations(near, far):
        tag = ("g" + "".join(_COORDINATE[axis] for axis in subset)
               + ("f" if carries_far else ""))
        flags = [f"({_COORDINATE[axis]} == {NEAR_SOURCE_INDEX})" for axis in subset]
        terms = [f"- {NEAR_SOURCE_INDEX} * {_STRIDE[axis]}" for axis in subset]
        weight = 1
        for axis in subset:
            weight *= _component_parity(target, axis, phases[axis])
        described = [
            f"the near fill on {_COORDINATE[axis]}: {_COORDINATE[axis]} = 0 imaged "
            f"from {NEAR_SOURCE_INDEX} (stepping._fill_symmetry_ghost_cells:1441, "
            f"_write_mirror_ghost:1449)" for axis in subset]
        if carries_far:
            axis = int(far[0])
            flags.append(f"({_COORDINATE[axis]} == {_REFLECT[axis]})")
            terms.append(f"+ ({_LAST[axis]} - {_REFLECT[axis]}) * {_STRIDE[axis]}")
            described.append(
                f"the far fill on {_COORDINATE[axis]}: {_COORDINATE[axis]} = "
                f"{_LAST[axis]} imaged from {_REFLECT[axis]} "
                f"(stepping._fill_folded_far_ghosts:1516)")
        far_weight = (_component_parity(target, int(far[0]), phases[int(far[0])])
                      if carries_far else 1)
        cleared = (zero_metal_lines(target, zero_metal, f"{tag}_v", inner)
                   if subset else [])
        lines.append(f"{indent}if ({' && '.join(flags)}) {{")
        lines.extend(f"{inner}// {text}" for text in described)
        lines.append(f"{inner}// near parity {weight:+d}"
                     + (f", far parity {far_weight:+d}" if carries_far else "")
                     + (", clear between them (driver.py:3301 between :3300 and "
                        ":3302)." if cleared and carries_far
                        else ", clear applied AFTER it." if cleared
                        else ", no wall clears this target so the two compose."
                        if carries_far
                        else ", no wall clears this target."))
        lines.append(f"{inner}int {tag}_i = ii {' '.join(terms)};")
        if cleared:
            lines.append(f"{inner}float {tag}_v = "
                         f"{_parity_spelling(weight, f'v{target}')};")
            lines.extend(cleared)
            if far_weight == -1:
                lines.append(
                    f"{inner}// the far parity, applied to the ALREADY cleared "
                    f"value (driver.py:3302 after :3301).")
                lines.append(f"{inner}{tag}_v = -{tag}_v;")
        else:
            lines.append(f"{inner}float {tag}_v = "
                         f"{_parity_spelling(weight * far_weight, f'v{target}')};")
        lines.append(f"{inner}f{target}[{tag}_i] = {tag}_v;")
        # THE DESTINATION'S OWN COORDINATE ON THE INDEXED AXIS, and it is the one
        # asymmetry the D geometry forces -- the exact inverse of the magnetic
        # twin's. ``dispersive_e_source`` indexes target ``t`` on axis ``t``
        # (stepping.E_CONSTITUTIVE_TERMS), :func:`far_fill_axes` returns axis ``t``
        # ALONE and :func:`near_fill_axes` only axes ``a != t``. So a NEAR
        # destination sits at the SAME coordinate on the indexed axis as its owner
        # and takes that coordinate unchanged, while a FAR one has MOVED along that
        # very axis and must take ``last``. Reusing the owner's would apply the
        # reflect row's absorber profile to the top plane -- normally observable,
        # because a folded axis carries its absorber on the HIGH face only
        # (stepping._require_consistent_pml), so the two entries differ. The
        # pointwise family reads its pair from a POINTER at a coordinate rather than
        # loading a value, so what moves here is the COORDINATE; the value
        # :func:`.folded_fused_pair._far_coefficient_lines` loads at ``last`` for the
        # ordinary pair is the identical word.
        coordinate = _LAST[target] if carries_far else _COORDINATE[target]
        lines.append(f"{inner}int {tag}_coord = {coordinate};")
        lines.extend(_constitutive_block(target, f"{tag}_i", f"{tag}_v", inner,
                                         f"{tag}_", pole_count,
                                         f"kp{target}", f"km{target}"))
        lines.append(f"{indent}}}")
    return lines


def folded_fused_dispersive_pair_source(
        codes: Sequence[int], target: int, pole_count: int,
        phases: Sequence[int], zero_metal: Sequence[bool],
        contract: str = shaders.CONTRACT_OFF) -> str:
    """One specialised source: (folded codes, component, poles, parities, walls, mode).

    ``codes`` is the per-axis PERIODIC / METALLIC / MIRROR_METALLIC /
    MIRROR_PERIODIC quadruple :func:`.symmetry.folded_axis_kinds` resolves -- never
    a hand-built triple, because the MIRROR_METALLIC / MIRROR_PERIODIC split is this
    family's single point of failure.

    ``phases`` is ``grid.mirror_phase(axis)`` per axis, +1 or -1 on a folded axis.
    It is a SOURCE specialisation and not a runtime scalar, which is measured rather
    than preferred: a runtime weight flushes every subnormal on this backend at BOTH
    signs (symmetry.py:657-686).
    """
    codes = tuple(int(code) for code in codes)
    if len(codes) != 3:
        raise ValueError(f"codes must be a per-axis triple, got {codes!r}")
    if target not in (0, 1, 2):
        raise ValueError(f"target must be 0, 1, or 2, got {target!r}")
    if not 0 <= int(pole_count) <= MAX_POLES:
        raise ValueError(f"pole_count must be in [0, {MAX_POLES}], got {pole_count!r}")
    if not any(code in MIRROR_CODES for code in codes):
        raise ValueError(
            "no axis is folded: this product exists to carry the mirror fills "
            "inside the D seam, and an unfolded grid belongs to "
            "fused_dispersive_pair")
    zero_metal = tuple(bool(value) for value in zero_metal)
    if len(zero_metal) != 3:
        raise ValueError(f"zero_metal must be a per-axis triple, got {zero_metal!r}")
    phases = tuple(int(value) if value is not None else 0 for value in phases)
    if len(phases) != 3:
        raise ValueError(f"phases must be a per-axis triple, got {phases!r}")
    for axis, code in enumerate(codes):
        if int(code) in MIRROR_CODES and phases[axis] not in (1, -1):
            raise ValueError(
                f"axis {axis} is folded but its mirror phase is {phases[axis]!r}; "
                f"a plane's parity is +1 or -1 and an even-mirror default standing "
                f"in for a plane that declared otherwise is a run wrong by twice the "
                f"field wherever the parity mattered")
        if int(code) in MIRROR_CODES and zero_metal[axis]:
            raise ValueError(
                f"axis {axis} is reported both folded and walled; "
                f"stepping._zero_metal excludes a folded axis by construction "
                f"(stepping.py:2282-2284) and this kernel's ghost carry relies on "
                f"the two sets being disjoint")

    reduced = _reduced_codes(codes)
    spec = _TARGETS[target]
    first, second = _RECURRENCE_AXES[target]
    axis_coordinate = {"x": "i", "y": "j", "z": "k"}

    body: List[str] = [
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
        "    // BOTH MIRROR CODES TAKE THE METALLIC BRANCH through _reduced_codes;",
        "    // the ghost it serves is dead under the cell-0 mask (symmetry.py:36-49).",
        "    int si = i - 1, sj = j - 1, sk = k - 1;",
        "    bool vx = true, vy = true, vz = true;",
        templates.ghost("x", reduced[0], True),
        templates.ghost("y", reduced[1], True),
        templates.ghost("z", reduced[2], True),
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
        "    // --- ownership mask, cell 0 (stepping._mask_non_owned_cells:1865) --",
        "    bool at_x = (i == 0), at_y = (j == 0), at_z = (k == 0);",
        "    (void)at_x; (void)at_y; (void)at_z;",
    ]
    # ONE emitter, filtered to this target -- the same filter
    # `fused_dispersive_pair` applies. Taking the rows that name this curl keeps
    # the text the certified emitter's own rather than a second copy of its table.
    cell_zero = [line for line in templates.ownership_mask(reduced, True).splitlines()
                 if line.strip().startswith(f"curl{target} ")]
    body.extend(cell_zero or [f"    // no metallic axis masks curl{target}"])
    body.extend([
        "",
        "    // --- ownership mask, the TOP plane of a folded PERIODIC axis -------",
        "    bool last_x = (i == nxi - 1), last_y = (j == nyi - 1), "
        "last_z = (k == nzi - 1);",
        "    (void)last_x; (void)last_y; (void)last_z;",
    ])
    top_plane = [line for line in folded_top_plane_mask(codes, True).splitlines()
                 if line.strip().startswith(f"curl{target} ")]
    body.extend(top_plane
                or [f"    // no folded PERIODIC axis masks curl{target}'s top plane"])
    body.extend([
        "",
        "    // --- split-field recurrence (stepping._apply_pml_update:1905) ------",
        f"    float km_{first} = km{first}[{axis_coordinate[first]}], "
        f"si_{first} = sinv{first}[{axis_coordinate[first]}];",
        f"    float km_{second} = km{second}[{axis_coordinate[second]}], "
        f"si_{second} = sinv{second}[{axis_coordinate[second]}];",
        "",
        f"    float p{target} = u{target}[ii];",
        f"    float n{target} = ((p{target} * km_{first}) - curl{target})"
        f" * si_{first};",
        "",
        "    // `u` IS WRITTEN AT EVERY CELL, ghost cells included: step_D updates",
        "    // fu everywhere and the fills touch D only (stepping.py:1449-1450",  # stepping.py live lines for the frozen device-text citation(s) in this string: 1449-1450->1497-1498
        "    // writes `field`, never `fu_field`).",
        f"    u{target}[ii] = n{target};",
        "",
    ])

    near = near_fill_axes(codes, target)
    far = far_fill_axes(codes, target)
    if len(near) > 2:
        raise AssertionError(
            f"component {target} is a near-fill destination on {near}; for the D "
            f"family that set is at most two axes (fields.IYEE_SHIFTS)")
    if len(far) > 1 or set(far) & set(near):
        raise AssertionError(
            f"component {target} is a far-fill destination on {far} against near "
            f"axes {near}; for the D family the two sets are COMPLEMENTARY and far "
            f"holds at most ONE axis")
    owned = ([f"({_COORDINATE[axis]} == 0)" for axis in near]
             + [_LAST_FLAG[axis] for axis in far])
    if owned:
        body.extend([
            f"    // component {target}: the fills image "
            + ", ".join(
                [f"stored cell 0 on {_COORDINATE[axis]} (near)" for axis in near]
                + [f"the top plane on {_COORDINATE[axis]} (far)" for axis in far])
            + ",",
            "    // so those cells are OWNED BY THEIR SOURCE THREADS. This one stops "
            "after fu:",
            "    // the array path's fill overwrites the displacement it would store, "
            "and",
            "    // forming v here would read a word another thread writes.",
            f"    if (!({' || '.join(owned)})) {{",
        ])
        indent = "        "
    else:
        body.append(f"    // component {target}: no folded axis images a cell of "
                    f"this component")
        indent = "    "
    body.append(
        f"{indent}float v{target} = (((f{target}[ii] * km_{second}) + n{target})"
        f" - p{target}) * si_{second};")
    cleared = zero_metal_lines(target, zero_metal, f"v{target}", indent)
    body.extend(cleared or [f"{indent}// no walled axis clears this target"])
    body.append(f"{indent}f{target}[ii] = v{target};")
    body.extend(_constitutive_block(target, "ii", f"v{target}", indent, "o",
                                    pole_count))
    if owned:
        body.append("")
        body.extend(_carry_blocks(target, near, far, phases, zero_metal,
                                  pole_count, indent))
        body.append("    }")

    return templates.substitute(_TEMPLATE, {
        "__CONTRACT__": templates.contraction_pragma(contract),
        "__T__": str(target),
        "__BODY__": "\n".join(body),
    })


def compile_folded_fused_dispersive_pair(
        codes: Sequence[int], target: int, pole_count: int,
        phases: Sequence[int], zero_metal: Sequence[bool],
        contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised entry point for one (codes, component, poles, parities, walls)."""
    return compile_source(folded_fused_dispersive_pair_source(
        codes, target, pole_count, phases, zero_metal,
        contract)).folded_fused_dispersive_pair_component


def refuted_separate_scalar_source(contract: str = shaders.CONTRACT_OFF) -> str:
    """The same 24 pointers with the eight scalars bound SEPARATELY -- 32 bindings.

    One OVER the ceiling, and the gate requires the compile failure. A narrow miss
    is exactly the case a comment gets wrong, which is why this is a compiled
    measurement rather than an assertion about arithmetic on a number.
    """
    pointers = [f"    device float* v{index} [[buffer({index})]]"
                for index in range(4)]
    pointers += [f"    device const float* r{index} [[buffer({index})]]"
                 for index in range(4, 24)]
    scalars = [f"    constant uint& s{index} [[buffer({index})]]"
               for index in range(24, 28)]
    scalars.append("    constant float& dtdx [[buffer(28)]]")
    scalars += [f"    constant int& t{index} [[buffer({index})]]"
                for index in range(29, 32)]
    arguments = ",\n".join(pointers + scalars)
    return ("#include <metal_stdlib>\nusing namespace metal;\n"
            + templates.contraction_pragma(contract) + "\n"
            "kernel void separate_scalar_folded_dispersive(\n" + arguments +
            ",\n    uint idx [[thread_position_in_grid]])\n"
            "{\n    v0[idx] = dtdx * r4[idx] + float(s24 + t29);\n}\n")


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def metal_folded_fused_dispersive_pair_coverage(
        fields: Any, pml: Any, sources: Any = None,
        residency: Any = None) -> Coverage:
    """May ONE dispatch per component span ``step_D`` -> fills -> wall -> ``update_E``?

    A conjunction of the two halves' OWN certified predicates plus the seam clauses.
    Nothing is weakened: a configuration either half refuses is refused here with
    that half's reasons, prefixed so a reader can tell which side said it. The two
    halves are the ``folded`` curl arm's routing verdict and the ``folded dispersive
    PML E`` arm's own predicate -- which is what keeps this product disjoint from
    :mod:`.folded_fused_pair` (whose E half requires NO susceptibility) and from
    :mod:`.fused_dispersive_pair` (whose halves refuse a fold) in both directions.
    """
    reasons: List[str] = []

    curl = folded_composition_curl_coverage(fields, pml, "step_D", residency)
    if not curl.covered:
        reasons.extend(f"folded curl half: {reason}" for reason in curl.reasons)
    electric = _folded_e.folded_dispersive_e_coverage(fields, pml, residency)
    if not electric.covered:
        reasons.extend(f"folded dispersive constitutive half: {reason}"
                       for reason in electric.reasons)

    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False,
                        tuple(dict.fromkeys(reasons + ["fields carries no grid"])))

    # THE SOURCE SEAM. An electric source is injected BETWEEN the two halves
    # (driver.py:3294-3299), so a fused pair would consume a pre-injection D unless
    # the deposit repair brackets the launch -- which for this family it does.
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
            f"injects it BETWEEN step_D and update_E (driver.py:3294-3299)"),
        carries_repair=CARRIES_DEPOSIT_REPAIR))

    codes, code_reasons = folded_axis_kinds(
        grid, pml if (pml is not None and getattr(pml, "is_active", False)) else None)
    reasons.extend(code_reasons)

    # THE FAR FILL IS CARRIED, not refused; these are the conditions its OWNERSHIP
    # MOVE rests on, and the mirror parity must be readable on every folded axis.
    reasons.extend(_far_carry_reasons(grid, codes))
    reasons.extend(_mirror_phase_reasons(grid, codes))

    # THE WALL CLEAR IS CARRIED INLINE, so the grid must be able to answer which
    # axes are walled. A grid that cannot would silently be treated as unwalled,
    # which is a plane of wrong values rather than a crash.
    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            reasons.append(
                f"grid does not expose {name}; zero_metal_D cannot be carried inline")

    shape = tuple(getattr(grid, "shape", ()))
    if codes is not None and len(shape) == 3:
        for axis, code in enumerate(codes):
            if int(code) in MIRROR_CODES and int(shape[axis]) <= NEAR_SOURCE_INDEX:
                reasons.append(
                    f"axis {axis} is folded with {int(shape[axis])} stored cells; "
                    f"the near fill images stored cell {NEAR_SOURCE_INDEX} and this "
                    f"kernel images it from that cell's own thread")

    # A POLE BUDGET THIS SOURCE CANNOT BAKE. The E half already refuses more than
    # MAX_POLES; restated because the count is a SOURCE specialisation here and a
    # component compiled for the wrong count would subtract the wrong number of
    # arrays rather than fail.
    for component, driven in _poles(fields).items():
        if len(driven) > MAX_POLES:
            reasons.append(
                f"{component} is driven by {len(driven)} poles, more than "
                f"MAX_POLES={MAX_POLES}")
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


class MetalFoldedFusedDispersivePairPlan:
    """Three dispatches performing five driver passes, D never leaving a register.

    ``launches_per_run`` is THREE and is declared rather than inherited: the
    whole-step arbiter asserts the exact per-cycle launch count on every filled
    slot, and that assertion is what stops a slot passing by not executing.
    """

    performs_device_work = True
    replaces_sub_steps = REPLACES
    launches_per_run = len(E_TERMS)

    __slots__ = ("entries", "residency", "volumes", "codes", "phases", "zero_metal",
                 "shape", "dtdx", "reflect", "_functions", "_tensors", "_magnetic",
                 "_curl_coefficients", "_params", "launches", "runs")

    def __init__(self, entries: Sequence[_Entry], residency: Residency,
                 volumes: Sequence[str], codes: Sequence[int],
                 phases: Sequence[int], zero_metal: Sequence[bool],
                 shape: Sequence[int], dtdx: float,
                 reflect: Sequence[Optional[int]],
                 magnetic: Sequence[Any], curl_coefficients: Sequence[Any],
                 params: Any, functions: Mapping[Tuple[str, int], Any],
                 tensors: Mapping[int, Any]) -> None:
        self._magnetic = tuple(magnetic)
        self._curl_coefficients = tuple(curl_coefficients)
        self.entries = tuple(entries)
        self.residency = residency
        self.volumes = tuple(dict.fromkeys(volumes))
        self.codes = tuple(int(code) for code in codes)
        self.phases = tuple(int(value) for value in phases)
        self.zero_metal = tuple(bool(value) for value in zero_metal)
        self.shape = tuple(int(n) for n in shape)
        self.dtdx = float(dtdx)
        # ``None`` where an axis has no far ghost, kept as ``None`` in the plan's
        # own record and mapped to the -1 sentinel only inside the device struct.
        self.reflect = tuple(None if row is None else int(row) for row in reflect)
        self._params = params
        self._functions = dict(functions)
        self._tensors = dict(tensors)
        self.launches = 0
        self.runs = 0

    @property
    def variants(self) -> Tuple[str, ...]:
        return tuple(sorted({key[0] for key in self._functions}))

    def run(self, contract: Optional[str] = None) -> None:
        """One complete folded D-half seam, in place, against the mirrors.

        A mode the plan was NOT built with RAISES rather than silently launching the
        pinned one, for the reason :class:`.plans.KernelPlan` gives: a guard argument
        that quietly did nothing would make the gate's most important leg vacuous.

        THE POLE POINTERS ARE RESOLVED AT LAUNCH, NOT AT PLAN TIME. ``update_P``
        rotates ``P``/``P_prev``/``_scratch`` after every recurrence, so the PHYSICAL
        array holding a component's current P changes between steps while the mirror
        registry does not.
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
            # THE UNUSED POLE SLOTS BIND THE INVERSE EPSILON, NOT THE DISPLACEMENT,
            # for `fused_dispersive_pair`'s reason: this kernel WRITES D, so padding
            # with it would alias a written buffer into a `device const float*` in
            # the same dispatch.
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
                self._params,
            )
            self.launches += 1
        self.runs += 1

    def describe(self) -> str:
        return (f"MetalFoldedFusedDispersivePairPlan(shape={self.shape}, "
                f"poles={tuple(e.pole_count for e in self.entries)}, "
                f"codes={self.codes}, phases={self.phases}, "
                f"zero_metal={self.zero_metal}, reflect={self.reflect}, "
                f"variants={self.variants})")

    def __repr__(self) -> str:
        return self.describe()


def params_record_dtype() -> Any:
    """The host record's dtype -- five scalars and the three reflect rows.

    ONE HOME FOR THE LAYOUT. Every member is four bytes and Metal's alignment for
    ``uint``/``int``/``float`` is four, so the natural record IS the struct; the
    offsets are stated anyway so the agreement cannot become accidental.
    """
    import numpy as np  # noqa: PLC0415

    return np.dtype({
        "names": ["nx", "ny", "nz", "n_elem", "dtdx", "rx", "ry", "rz"],
        "formats": ["<u4", "<u4", "<u4", "<u4", "<f4", "<i4", "<i4", "<i4"],
        "offsets": [0, 4, 8, 12, 16, 20, 24, 28],
        "itemsize": 32,
    })


def _params_tensor(shape: Sequence[int], dtdx: float,
                   reflect: Sequence[Optional[int]], device: str) -> Any:
    """The eight scalars as one 32-byte device record, built once at plan time.

    PLAN-OWNED AND OUTSIDE THE RESIDENCY REGISTRY, for the reason every packed pair
    gives: :meth:`.Residency.mirror` binds float32 and complex64 volumes and refuses
    anything else BY NAME. Nothing on the launch path allocates.

    ``-1`` stands for an axis with no reflect row (unfolded, or folded METALLIC,
    where the stored array stops at ``big_corner`` and no far ghost exists) -- the
    sentinel :func:`.folded_fused_pair._params_tensor` uses, and for its reason: it
    is a value the kernel never reads, because the guard that would read it is
    emitted only for an axis :func:`far_fill_axes` returned, and a source that read
    it anyway would index outside the volume and be caught rather than silently
    imaging row 0.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    record = np.zeros(1, dtype=params_record_dtype())
    nx, ny, nz = (int(n) for n in shape)
    record["nx"], record["ny"], record["nz"] = nx, ny, nz
    record["n_elem"] = nx * ny * nz
    record["dtdx"] = np.float32(dtdx)
    rows = tuple(-1 if value is None else int(value) for value in reflect)
    if len(rows) != 3:
        raise ValueError(f"reflect must be a per-axis triple, got {reflect!r}")
    record["rx"], record["ry"], record["rz"] = rows
    words = np.frombuffer(record.tobytes(), dtype=np.int32).copy()
    return torch.from_numpy(words).to(torch.device(device))


def plan_metal_folded_fused_dispersive_pair(
        fields: Any, pml: Any, sources: Any = None,
        residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        functions: Optional[Mapping[Tuple[str, int], Any]] = None,
        ) -> Optional[MetalFoldedFusedDispersivePairPlan]:
    """Build the folded fused dispersive D/E plan, or ``None`` when refused.

    ``None`` is the only refusal, for the reason :func:`.launch.plan_pml_curl` gives:
    a configuration this kernel does not carry must fall back to the array path,
    never raise into a caller that would otherwise have stepped correctly.

    ``functions`` is the mutation seam. The gate compiles a deliberately broken copy
    of the shipped source and hands it here; dropping the argument is not a silent
    slowdown, it is a silent DISARMING.
    """
    if not metal_folded_fused_dispersive_pair_coverage(
            fields, pml, sources, residency).covered:
        return None
    from ..stepping import _far_reflect_rows  # noqa: PLC0415

    grid = fields.grid
    codes, _ = folded_axis_kinds(grid, pml)
    if codes is None:  # pragma: no cover - the predicate already refused
        return None
    codes = tuple(int(code) for code in codes)
    phases = tuple(int(_call(grid, "mirror_phase", axis, default=0) or 0)
                   for axis in range(3))
    walls = zero_metal_axes(grid)
    # The runtime reflect rows the far carry images from — stepping's OWN answer,
    # never a local `n - 2`: the formula is `n_full - stored + 2`, which is
    # `stored - 2` at an even full count and `stored - 3` at an odd one. ``None`` on
    # an axis with no far ghost, and :func:`_params_tensor` maps that to the -1
    # sentinel the kernel never reads.
    reflect = _far_reflect_rows(grid)
    order = _poles(fields)

    tensors: Dict[int, Any] = {}
    volumes: List[str] = []
    entries: List[_Entry] = []
    required: Set[Tuple[str, int]] = set()

    def bind(name: str, host: Any, constant: bool = False) -> Any:
        tensors[id(host)] = residency.mirror(name, host, constant=constant)
        volumes.append(name)
        return host

    # The curl's magnetic operands and integer-lattice coefficients are SHARED by
    # all three dispatches -- one mirror each, which is the whole point of the
    # residency layer.
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
        # The E half takes the HALF-INTEGER lattice and the curl half the INTEGER
        # one (stepping.py:948 vs :1015). The kernel takes both and never asks which
        # is which, so a swap here is a silent half-cell error in the absorber
        # profile.
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
            selected[key] = compile_folded_fused_dispersive_pair(
                codes, key[1], by_axis[key[1]], phases, walls, key[0])

    dtdx = grid.dt / grid.dx
    return MetalFoldedFusedDispersivePairPlan(
        entries, residency, volumes, codes, phases, walls, grid.shape, dtdx,
        reflect, magnetic, curl_coefficients,
        _params_tensor(grid.shape, dtdx, reflect, residency.device),
        selected, tensors)


# ---------------------------------------------------------------------------
# Registration — NOT WIRED
# ---------------------------------------------------------------------------

def _arm_coverage(context: Any, slot: str) -> Coverage:
    if slot != SLOT:
        return Coverage(
            False, (f"folded fused dispersive pair cannot fill {slot}",))
    return metal_folded_fused_dispersive_pair_coverage(
        context.fields, context.pml, context.sources, context.residency)


def _arm_plan(context: Any,
              slot: str) -> Optional[MetalFoldedFusedDispersivePairPlan]:
    if slot != SLOT:
        return None
    return plan_metal_folded_fused_dispersive_pair(
        context.fields, context.pml, context.sources, context.residency,
        context.contract_variants)


def register_arms() -> Tuple[Any, ...]:
    """One row on ``step_D``, ``wired=False``.

    THE FLAG IS THE WHOLE COMPOSITION STORY, as it is on every other pair.
    ``plan_step`` assigns at most one arm per slot and this product spans five, so
    there is no slot it could claim through the arm table. Registering unwired keeps
    it ENUMERABLE for the disjointness sweep (``arms.registered``) while
    ``arms.arms_for`` skips it, so ``_select_slot`` cannot select it and no existing
    arm's selection changes -- including the ``folded`` curl arm and the ``folded
    dispersive PML E`` arm this pair's own predicate is built out of.

    THE OTHER HALF IS THE ABSORB TABLE:
    ``launch.FUSED_PAIR_ARMS["folded_fused_dispersive_pair"]`` declares which arm
    each of the two slots implements.
    """
    from . import arms  # noqa: PLC0415
    return (arms.register(FAMILY, SLOT, "folded fused dispersive D/E pair",
                          _arm_coverage, _arm_plan,
                          prefix="folded fused dispersive D/E pair: ",
                          noun="folded fused PML dispersive D-curl/stored-E pair",
                          wired=False, replaces=REPLACES),)


ARMS = register_arms()
