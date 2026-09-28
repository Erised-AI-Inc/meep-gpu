"""The two mirror fills carried through a COMPLEX fused ELECTRIC seam -- shared emitters.

WHAT THIS IS. :mod:`.fused_electric_pair` carries ``fill_symmetry_bc_D`` and
``fill_folded_far_ghosts_D`` through the OWNERSHIP INVERSION -- the source
thread writes each imaged ghost from a live register, the destination thread is
carved out -- in real float32 arithmetic, with the D family's geometry: the
NEAR set is the TWO shift-0 axes (reading the PRE-clear register, because
``fill_symmetry_bc_D`` at driver.py:3309 runs BEFORE ``zero_metal_D`` at
:3310), the FAR set is the component's own axis (applied last), and the
OFF-DIAGONAL wall clear sits between the two parity products.
:mod:`.complex_fill_carry` is the same inversion under complex64 storage for
the MAGNETIC seam, with the B geometry. This module is the remaining corner of
that 2x2: the D geometry under complex64 storage, spelled once so the two
products that need it (:mod:`.complex_beta_fused_electric_pair` and
:mod:`.complex_folded_fused_electric_pair`) cannot drift apart on it -- their
certified curl sources genuinely share the fold spelling
(``complex_beta_kernels.beta_source`` is ``complex_folded_kernels.
folded_source`` plus the beta insert, which never touches a fill), so one
carry serves both.

NOTHING HERE IMPORTS CUPY OR NUMPY AT MODULE SCOPE: the emitters splice strings
and the clauses read a grid, and both must run on the census laptop.

=============================================================================
WHAT THE TWO PARENT CARRIES ALREADY MEASURED, AND WHAT THIS ONE COMPOSES
=============================================================================

1. **THE PARITY MULTIPLY IS A FULL COMPLEX PRODUCT, NEVER A SCALE** -- the
   B-side complex carry's own measured fact (the GPU host 2026-09-01, 4108 words:
   ``mul_coefficient_left(+-1, z)`` matched CuPy's ``parity * z`` on every word
   including signed zeros; the plain two-word scale did not). Every parity here
   is applied through the certified ``mul_coefficient_left``.

2. **THE CHAIN ORDER IS REAL AND IS THE DRIVER'S** -- also measured there
   ((-1)x((-1)xz) differs from (+1)xz in 4 of 8216 words). Transcribed for the
   D family: ``_fill_symmetry_ghost_cells`` applies its axes IN X, Y, Z ORDER
   (stepping.py's own docstring: "The axes are filled in X, Y, Z order, which
   leaves a corner unowned on two planes carrying the product of both
   parities"), each pass reading the plane the previous one wrote -- so the
   near applications compose ASCENDING, innermost first. ``zero_metal_D``
   (:3310) then clears at the destination, and ``fill_folded_far_ghosts_D``
   (:3311) applies its single D-side axis LAST. The emitted chain is therefore:
   near multiplies ascending over the PRE-clear register, the clear, the far
   multiply -- never a pre-multiplied product.

3. **THE PRE-CLEAR REGISTER AND THE NAMED CLEAR FLAG** -- the real electric
   pair's own measured ordering (``probe_cuda_electric_fill_carry_order.py``:
   0 differing words for this ordering on twelve configurations, 9 and 11 for
   the naive one on the two where the cross term exists). A D component's near
   axes ARE the axes ``zero_metal_D`` clears it on, so it can be folded on one
   and walled on the other at once; the near product is formed from ``pre_*``
   and the clear is applied to the result, which is the driver's own order.
   Under complex storage the cleared value is the word pair ``(+0.0f, +0.0f)``
   (``cf_zero()``) and the far multiply then gives it the sign the complex
   product gives it, exactly as the array path's pass order composes.

4. **THE OWNERSHIP RULE IS UNCHANGED.** Every step of
   ``fused_electric_pair.carried_destinations``' argument is an INDEX fact --
   it mentions no dtype -- and transfers to word pairs unchanged.

WHAT THIS MODULE DOES NOT DECIDE: which configurations a product admits. Each
product's own predicate conjoins its two certified halves' predicates and THEN
asks :func:`complex_electric_fill_reasons`; this module never answers for a
curl.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from .in_seam_coverage import (IYEE_SHIFTS, MIRROR_SOURCE_INDEX,
                                   folded_far_rows, mirror_fill_phases,
                                   stored_past_owned, zero_metal_axes)
except ImportError:  # loaded by path, outside the package: the bit-identity probe
    import importlib.util as _importlib_util
    import os as _os

    _spec = _importlib_util.spec_from_file_location(
        "cuda_kernels_in_seam_coverage",
        _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                      "in_seam_coverage.py"))
    _in_seam_coverage = _importlib_util.module_from_spec(_spec)
    _spec.loader.exec_module(_in_seam_coverage)
    IYEE_SHIFTS = _in_seam_coverage.IYEE_SHIFTS
    MIRROR_SOURCE_INDEX = _in_seam_coverage.MIRROR_SOURCE_INDEX
    folded_far_rows = _in_seam_coverage.folded_far_rows
    mirror_fill_phases = _in_seam_coverage.mirror_fill_phases
    stored_past_owned = _in_seam_coverage.stored_past_owned
    zero_metal_axes = _in_seam_coverage.zero_metal_axes

__all__ = [
    "NEAR_SOURCE_INDEX", "captured_curl_body", "carried_destinations",
    "complex_electric_fill_reasons", "constitutive_body_with_carry",
    "far_fill_axes", "fill_carry_blocks", "fills_plan", "near_fill_axes",
    "ownership_declarations", "pml_apply_reg_prelude", "pre_clear_registers",
    "zero_metal_carry",
]

#: ``stepping.MIRROR_SOURCE_INDEX`` under this seam's own name -- the fold's magic 2.
NEAR_SOURCE_INDEX = MIRROR_SOURCE_INDEX

#: The three D targets by index, and the certified complex template's own
#: spellings for each axis's coordinate, stride and extent (all declared by the
#: curl body). Extents and strides are in COMPLEX CELLS; ``cf_load``/``cf_store``
#: do the word doubling.
_TARGETS: Tuple[str, ...] = ("Dx", "Dy", "Dz")
_AXIS_NAME: Tuple[str, ...] = ("x", "y", "z")
_COORDINATE: Tuple[str, ...] = ("i", "j", "k")
_STRIDE: Tuple[str, ...] = ("sx", "sy", "sz")
_EXTENT: Tuple[str, ...] = ("nx", "ny", "nz")

#: The fill scalars' runtime names, in the fused signatures' own spelling.
_NEAR_FLAG: Tuple[str, ...] = ("near_x", "near_y", "near_z")
_REFLECT: Tuple[str, ...] = ("reflect_x", "reflect_y", "reflect_z")
_FAR_ACTIVE: Tuple[str, ...] = ("reflect_x >= 0", "reflect_y >= 0", "reflect_z >= 0")
_PHASE: Tuple[str, ...] = ("phase_x", "phase_y", "phase_z")

#: The per-component wall-clear flag names -- emitted by :func:`zero_metal_carry`
#: and read by :func:`fill_carry_blocks`, the real electric pair's own device.
_CLEARED: Tuple[str, ...] = ("clr_0", "clr_1", "clr_2")

#: ``stepping._zero_metal``'s D OFF-DIAGONAL as (wall flag, coordinate, the two
#: register indices wiped there). ``IYEE_SHIFTS`` gives Dx (1,0,0), Dy (0,1,0),
#: Dz (0,0,1): a D component has Yee shift 0 on the OTHER TWO axes, so TWO
#: components sit on each wall. The B family's table is the diagonal and
#: reusing it would clear the wrong components on every walled run.
_ZERO_METAL_ROWS: Tuple[Tuple[str, str, Tuple[int, ...]], ...] = (
    ("wall_x", "i", (1, 2)),
    ("wall_y", "j", (0, 2)),
    ("wall_z", "k", (0, 1)),
)

#: The anchors of the certified COMPLEX ``pml_apply``, and the rewrite that
#: turns it into an OWNED value -- byte-identical to
#: :mod:`.complex_fill_carry`'s, restated here so loading this file by path
#: cannot pick up a different guard than the one a gate compiled.
_PML_APPLY_SIGNATURE = "__device__ __forceinline__ void pml_apply(\n"
_PML_APPLY_SIGNATURE_REG = "__device__ __forceinline__ cf pml_apply_reg(\n"
_PML_APPLY_PARAMETERS = "    float kms, float sinv, float kms_u, float sinv_u\n)"
_PML_APPLY_PARAMETERS_OWNED = (
    "    float kms, float sinv, float kms_u, float sinv_u, int owned\n)")
_PML_APPLY_STORE = (
    "    cf_store(f, idx, mul_field_left(cf_sub(cf_add(a, fu_new), fprev), sinv_u));\n")
_PML_APPLY_LOAD = "    cf a = mul_field_left(cf_load(f, idx), kms_u);\n"
_PML_APPLY_GUARDED_TAIL = (
    "    // THE OWNERSHIP GUARD. On a cell a fill images, the SOURCE thread\n"
    "    // writes f -- so this thread must not read f[idx] either: that word is\n"
    "    // written by another block in this same launch and an ordinary CUDA\n"
    "    // launch has no grid-wide barrier at which the load would be defined.\n"
    "    // fu above is UNGUARDED: step_D writes it at every cell and neither\n"
    "    // fill touches it (stepping.py:1451).\n"
    "    if (!owned) return cf_zero();\n"
    "    // THE ONE EDIT BELOW THE GUARD: the right-hand side, its\n"
    "    // parenthesisation and the store are the certified ones; the value is\n"
    "    // additionally NAMED so the constitutive half can read it from a\n"
    "    // register instead of reloading the word pair from global memory.\n"
    "    cf a = mul_field_left(cf_load(f, idx), kms_u);\n"
    "    cf value = mul_field_left(cf_sub(cf_add(a, fu_new), fprev), sinv_u);\n"
    "    cf_store(f, idx, value);\n"
    "    return value;\n")

#: The last line of the certified index decomposition -- where the carried
#: registers and the ownership flags are declared.
_DECODE_END = "    int i = idx / (ny * nz);\n"


def near_fill_axes(target: int) -> Tuple[int, ...]:
    """Which axes ``fill_symmetry_bc_D`` can image for ONE D component.

    ``stepping._fill_symmetry_ghost_cells`` (:1440-1447) writes axis ``a`` for
    component ``m`` exactly when the plane's phase is declared and
    ``iyee[m][a] == 0`` -- for D, the TWO axes that are NOT the component's
    own, THE EXACT COMPLEMENT of the B family's. Read off ``IYEE_SHIFTS``
    rather than hand-typed, because that inversion is the one place a constant
    is plausible and wrong.
    """
    shifts = IYEE_SHIFTS[_TARGETS[target]]
    return tuple(axis for axis in range(3) if shifts[axis] == 0)


def far_fill_axes(target: int) -> Tuple[int, ...]:
    """Which axes ``fill_folded_far_ghosts_D`` can image for ONE D component.

    ``stepping._fill_folded_far_ghosts`` (:1524-1534) writes axis ``a`` for
    component ``m`` exactly when ``_stored_past_owned`` holds there and
    ``iyee[m][a] == 1`` -- for D, the component's OWN axis alone.
    """
    shifts = IYEE_SHIFTS[_TARGETS[target]]
    return tuple(axis for axis in range(3) if shifts[axis] == 1)


def carried_destinations(near: Sequence[int], far: Sequence[int]
                         ) -> Tuple[Tuple[Tuple[int, ...], Tuple[int, ...]], ...]:
    """Every ghost cell ONE source thread owns, as ``(far axes at top, near axes at 0)``.

    THE OWNERSHIP RULE, the real electric pair's own
    (``fused_electric_pair.carried_destinations``), unchanged: every step of its
    argument is an INDEX fact and transfers to word pairs unchanged. Returned
    smallest-subset-first for a stable emission order.
    """
    near = tuple(int(axis) for axis in near)
    far = tuple(int(axis) for axis in far)
    combos: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []
    for far_size in range(len(far) + 1):
        for far_subset in itertools.combinations(far, far_size):
            for near_size in range(len(near) + 1):
                for near_subset in itertools.combinations(near, near_size):
                    if not far_subset and not near_subset:
                        continue  # the thread's OWN cell, not a ghost
                    combos.append((far_subset, near_subset))
    return tuple(sorted(combos, key=lambda item: (len(item[0]) + len(item[1]),
                                                  item[0], item[1])))


def ownership_declarations() -> str:
    """``own_0``/``own_1``/``own_2`` -- is this thread's own cell imaged by a fill?

    THE ONE PLACE THE INVERSION IS DECIDED; every other emitter here reads
    these three flags. D geometry: a component is near-imaged at stored 0 of
    its two shift-0 axes and far-imaged at the top of its own axis.
    """
    lines = [
        "    // --- the ownership inversion (fill_symmetry_bc_D, "
        "fill_folded_far_ghosts_D) ---",
        "    // A cell a fill images is written by its SOURCE thread, from a live",
        "    // register, below. The thread standing ON it stops after fu: forming",
        "    // the displacement would read a D word another block writes in this",
        "    // same launch, and an ordinary CUDA launch has no grid-wide barrier",
        "    // at which that load is defined. The array path discards that",
        "    // displacement too -- the fill overwrites it (driver.py:3309, :3311).",
    ]
    for target, name in enumerate(_TARGETS):
        tests = [f"({_NEAR_FLAG[axis]} && {_COORDINATE[axis]} == 0)"
                 for axis in near_fill_axes(target)]
        tests += [f"({_FAR_ACTIVE[axis]} && "
                  f"{_COORDINATE[axis]} == {_EXTENT[axis]} - 1)"
                  for axis in far_fill_axes(target)]
        if not tests:
            raise AssertionError(
                f"{name} is imaged by neither fill on any axis; IYEE_SHIFTS gives "
                f"every D component one shift-1 axis and two shift-0 axes, so an "
                f"empty test means the Yee table has drifted")
        near = "".join(_AXIS_NAME[axis] for axis in near_fill_axes(target))
        far = "".join(_AXIS_NAME[axis] for axis in far_fill_axes(target))
        lines.append(f"    // {name} {IYEE_SHIFTS[name]}: near fill on {near}, "
                     f"far fill on {far}.")
        lines.append(f"    int own_{target} = !({' || '.join(tests)});")
    return "\n".join(lines) + "\n"


def pre_clear_registers() -> str:
    """``pre0/pre1/pre2`` -- the displacement as ``fill_symmetry_bc_D`` sees it.

    The real electric pair's own device (its ``pre_clear_registers``), under
    complex storage: ``fill_symmetry_bc_D`` (driver.py:3309) runs one pass
    BEFORE ``zero_metal_D`` (:3310), so a near ghost images the UNCLEARED word
    pair and is only then cleared at its own cell. Emitted unconditionally;
    three dead copies on an unfolded run cost nothing the compiler does not
    remove.
    """
    return (
        "\n    // The PRE-CLEAR displacement, for the near fill carry. "
        "fill_symmetry_bc_D\n"
        "    // (driver.py:3309) runs BEFORE zero_metal_D (:3310), so a near "
        "ghost images\n"
        "    // what the curl left, and the clear then applies at the ghost's "
        "own cell.\n"
        + "".join(f"    cf pre{target} = d{target};\n" for target in range(3)))


def zero_metal_carry() -> str:
    """``zero_metal_D`` (driver.py:3310), carried between the two halves.

    The OFF-DIAGONAL table on the word pairs, the flag NAMED because the near
    fill carry reads it (a near ghost of a cleared cell carries the cleared
    word pair, not the parity-weighted one), and the clear GUARDED on ``own_*``
    (a cell a fill images belongs to its source thread; the array path agrees
    -- the far fill at :3311 overwrites the wall pass at :3310 on such a cell,
    or the near fill at :3309 wrote it first).
    """
    lines = [
        "\n    // stepping._zero_metal (:2206-2245) / in_seam_passes.zero_metal_D,\n"
        "    // CARRIED. The OFF-DIAGONAL: two components per walled axis, stored\n"
        "    // cell 0, the word pair (+0.0f, +0.0f). The flag is NAMED because\n"
        "    // the near fill carry below reads it; GUARDED on own_* because a\n"
        "    // cell a fill images is its source thread's.\n"
    ]
    lines.append("    " + " ".join(f"int {name} = 0;" for name in _CLEARED) + "\n")
    for flag, coordinate, targets in _ZERO_METAL_ROWS:
        sets = " ".join(f"{_CLEARED[target]} = 1;" for target in targets)
        lines.append(f"    if ({flag} && {coordinate} == 0) {{ {sets} }}\n")
    for target in range(3):
        lines.append(
            f"    if (own_{target} && {_CLEARED[target]}) "
            f"{{ d{target} = cf_zero(); cf_store(f{target}, idx, d{target}); }}\n")
    return "".join(lines)


def fill_carry_blocks(target: int, indent: str = "        ") -> List[str]:
    """Every imaged ghost this thread owns, then ``update_E`` at each of them.

    ONE BLOCK PER DESTINATION IN :func:`carried_destinations`, guarded on the
    flags that say THIS thread is that destination's source -- stored
    :data:`NEAR_SOURCE_INDEX` on each near axis, the runtime reflect row on the
    far axis. EVERY BLOCK IS EMITTED INSIDE the component's ownership guard by
    :func:`constitutive_body_with_carry`.

    THE VALUE, IN DRIVER ORDER, AND THE ORDER IS THE ARITHMETIC:

    1. the NEAR multiplies, one certified ``mul_coefficient_left`` per near
       axis IN ASCENDING ORDER (stepping's own "filled in X, Y, Z order"),
       innermost first, over ``pre{t}`` -- the PRE-clear word pair;
    2. the WALL CLEAR at the destination -- the source thread's own
       :data:`_CLEARED` flag IS the test (a near ghost only moves FOLDED axes,
       and a folded axis is never walled, so every walled coordinate is
       unchanged between the two cells) -- ``cf_zero()``, the word pair the
       array path's integer-0 assignment leaves;
    3. the FAR multiply, ``mul_coefficient_left((phase * -1.0f), v)``, applied
       LAST because the driver runs it last; a far-written cell is never
       itself cleared afterwards.

    A pure far ghost takes steps 2-as-already-applied and 3 only: its value
    starts from the POST-clear register ``d{t}``.

    THE COEFFICIENT PAIR IS THE DESTINATION'S. ``update_E`` indexes ``E{m}``
    on the component's OWN axis, and the FAR fill is the one that moves that
    axis on this family, so a far ghost takes ``kps[n - 1]`` and a pure near
    ghost this thread's own index unchanged -- the real electric pair's rule.
    """
    near = near_fill_axes(target)
    far = far_fill_axes(target)
    name = _TARGETS[target]
    if far != (target,):
        raise AssertionError(
            f"{name} is a FAR-fill destination on axes {far}; for the D family "
            f"that set is exactly the component's own axis")
    if len(near) != 2 or target in near:
        raise AssertionError(
            f"{name} is a NEAR-fill destination on axes {near}; on the D family "
            f"near holds exactly the two axes that are not the component's own")
    walls = tuple(_COORDINATE.index(coordinate)
                  for _flag, coordinate, targets in _ZERO_METAL_ROWS
                  if target in targets)
    if set(walls) & set(far):
        raise AssertionError(
            f"{name} images a FAR ghost along {far} that zero_metal_D also "
            f"clears on {sorted(set(walls) & set(far))}; the far ghost would no "
            f"longer inherit its source thread's clear")
    if set(walls) != set(near):
        raise AssertionError(
            f"{name} is cleared on axes {sorted(walls)} and near-imaged on "
            f"{sorted(near)}; on the D family those two sets are THE SAME, "
            f"which is the fact that puts the wall clear BETWEEN the two parity "
            f"products")
    cleared = _CLEARED[target]
    inner = indent + "    "
    lines: List[str] = []
    for far_subset, near_subset in carried_destinations(near, far):
        tag = ("g" + str(target) + "_"
               + "".join(_AXIS_NAME[axis] for axis in far_subset)
               + ("n" if near_subset else "")
               + "".join(_AXIS_NAME[axis] for axis in near_subset))
        guards: List[str] = []
        terms: List[str] = []
        described: List[str] = []
        for axis in far_subset:
            guards.append(f"{_FAR_ACTIVE[axis]} && "
                          f"{_COORDINATE[axis]} == {_REFLECT[axis]}")
            terms.append(f"+ ({_EXTENT[axis]} - 1 - {_REFLECT[axis]}) "
                         f"* {_STRIDE[axis]}")
            described.append(
                f"the far fill on {_AXIS_NAME[axis]}: {_COORDINATE[axis]} = "
                f"{_EXTENT[axis]} - 1 imaged from {_REFLECT[axis]}, weight "
                f"-{_PHASE[axis]} (stepping._fill_folded_far_ghosts:1524-1534)")
        for axis in near_subset:
            guards.append(f"{_NEAR_FLAG[axis]} && "
                          f"{_COORDINATE[axis]} == {NEAR_SOURCE_INDEX}")
            terms.append(f"- {NEAR_SOURCE_INDEX} * {_STRIDE[axis]}")
            described.append(
                f"the near fill on {_AXIS_NAME[axis]}: {_COORDINATE[axis]} = 0 "
                f"imaged from {NEAR_SOURCE_INDEX}, weight {_PHASE[axis]} "
                f"(stepping._fill_symmetry_ghost_cells:1440-1447)")
        coefficient = (f"{_EXTENT[target]} - 1" if far_subset
                       else _COORDINATE[target])
        lines.append(f"{indent}if ({' && '.join(guards)}) {{")
        lines.extend(f"{inner}// {text}" for text in described)
        lines.append(f"{inner}int {tag}_i = idx {' '.join(terms)};")
        if near_subset:
            lines.append(
                f"{inner}// THE PARITY CHAIN, one certified complex multiply per "
                f"pass, in the")
            lines.append(
                f"{inner}// driver's own order: near axes ASCENDING over the "
                f"PRE-clear pair")
            lines.append(
                f"{inner}// (fill_symmetry_bc_D at :3309 precedes the clear at "
                f":3310), the")
            lines.append(
                f"{inner}// clear, then the far multiply. Pre-multiplying moves "
                f"bytes under")
            lines.append(f"{inner}// complex storage.")
            ordered = sorted(near_subset)
            first = ordered[0]
            lines.append(f"{inner}cf {tag}_v = mul_coefficient_left("
                         f"{_PHASE[first]}, pre{target});")
            for axis in ordered[1:]:
                lines.append(f"{inner}{tag}_v = mul_coefficient_left("
                             f"{_PHASE[axis]}, {tag}_v);")
            lines.append(
                f"{inner}// zero_metal_D (:3310) at the DESTINATION. Its walled "
                f"coordinates are")
            lines.append(
                f"{inner}// this thread's -- a near ghost only moves a FOLDED "
                f"axis, and a folded")
            lines.append(
                f"{inner}// axis is never walled -- so the source thread's own "
                f"flag IS the test.")
            lines.append(f"{inner}if ({cleared}) {tag}_v = cf_zero();")
        else:
            lines.append(
                f"{inner}// No near fill on this destination: the register is "
                f"already the")
            lines.append(
                f"{inner}// post-clear value, which is what the far fill reads "
                f"at :3311.")
            lines.append(f"{inner}cf {tag}_v = d{target};")
        for axis in far_subset:
            lines.append(
                f"{inner}// fill_folded_far_ghosts_D (:3311), applied LAST "
                f"because the driver")
            lines.append(
                f"{inner}// runs it last; a far-written cell is not cleared "
                f"afterwards.")
            lines.append(f"{inner}{tag}_v = mul_coefficient_left("
                         f"({_PHASE[axis]} * -1.0f), {tag}_v);")
        lines.append(f"{inner}cf_store(f{target}, {tag}_i, {tag}_v);")
        if far_subset:
            lines.extend([
                f"{inner}// The DESTINATION's own coefficient entry: stored "
                f"index {_EXTENT[target]} - 1 on",
                f"{inner}// {_AXIS_NAME[target]}, NOT this thread's at "
                f"{_REFLECT[target]}: update_E indexes this component",
                f"{inner}// on {_COORDINATE[target]} and the far fill images "
                f"along that same axis.",
            ])
        else:
            lines.append(
                f"{inner}// A near ghost does not move {_COORDINATE[target]}, "
                f"the axis update_E indexes")
            lines.append(
                f"{inner}// this component on, so it takes this thread's own "
                f"pair unchanged.")
        lines.append(
            f"{inner}cf {tag}_s = mul_field_left({tag}_v, "
            f"inv_eps_{target}[{tag}_i]);")
        lines.append(
            f"{inner}constitutive_apply(h{target}, w{target}, {tag}_i, {tag}_s, "
            f"kps_{_AXIS_NAME[target]}[{coefficient}], "
            f"kms_half_{_AXIS_NAME[target]}[{coefficient}]);")
        lines.append(f"{indent}}}")
    return lines


def pml_apply_reg_prelude(prelude: str, name: str) -> str:
    """One family's certified complex prelude with ``pml_apply`` OWNED and valued.

    The B-side complex carry's three anchored rewrites, byte for byte -- the
    certified cf ``pml_apply`` is one helper shared by both sub-steps, so the
    ownership rewrite is too. Raises on a moved anchor rather than emitting a
    kernel that is quietly missing the guard.
    """
    for anchor, what in ((_PML_APPLY_SIGNATURE, "pml_apply's signature"),
                         (_PML_APPLY_PARAMETERS, "pml_apply's parameter list"),
                         (_PML_APPLY_LOAD, "pml_apply's field load"),
                         (_PML_APPLY_STORE, "pml_apply's closing store")):
        if prelude.count(anchor) != 1:
            raise AssertionError(
                f"{name}'s certified prelude carries {what} "
                f"{prelude.count(anchor)} times, not once; the ownership rewrite "
                f"has no anchor")
    prelude = prelude.replace(_PML_APPLY_SIGNATURE, _PML_APPLY_SIGNATURE_REG, 1)
    prelude = prelude.replace(_PML_APPLY_PARAMETERS, _PML_APPLY_PARAMETERS_OWNED, 1)
    if prelude.count(_PML_APPLY_LOAD + _PML_APPLY_STORE) != 1:
        raise AssertionError(
            f"{name}'s certified pml_apply no longer closes with the load+store "
            f"pair this rewrite guards as one unit; the arithmetic may have moved")
    prelude = prelude.replace(_PML_APPLY_LOAD + _PML_APPLY_STORE,
                              _PML_APPLY_GUARDED_TAIL, 1)
    return prelude


def captured_curl_body(body: str, name: str) -> str:
    """One family's certified step_D curl body with the registers captured OWNED.

    Hoists the ``cf d0/d1/d2`` declarations, the ownership flags and the
    PRE-CLEAR registers above the certified braced blocks, then rewrites each
    ``pml_apply(f{t}, u{t}, ...)`` call to
    ``d{t} = pml_apply_reg(f{t}, u{t}, ..., own_{t})``. A fold mask, a Bloch
    block or a beta insert between the decode and the calls rides through
    untouched, which is what lets ONE function serve both the folded and the
    beta curl bodies.

    THE PRE-CLEAR REGISTERS ARE DECLARED HERE AND ASSIGNED AFTER THE CAPTURES:
    ``pre{t}`` must hold the post-curl, PRE-clear value, so its assignment is
    appended after the last certified braced block rather than at declaration.
    """
    if _DECODE_END not in body:
        raise AssertionError(
            f"{name}'s certified curl body no longer decodes i on its own line; "
            f"the carried registers have nowhere to be declared")
    head, tail = body.split(_DECODE_END, 1)
    body = "".join((
        head, _DECODE_END,
        "\n    // The three carried registers. Declared HERE because each\n"
        "    // certified component below is a braced scope, and a value declared\n"
        "    // inside one does not outlive it.\n"
        "    cf d0 = cf_zero();\n"
        "    cf d1 = cf_zero();\n"
        "    cf d2 = cf_zero();\n"
        "\n",
        ownership_declarations(),
        tail,
    ))
    for target in range(3):
        old = f"pml_apply(f{target}, u{target}, "
        if body.count(old) != 1:
            raise AssertionError(
                f"{name}'s certified curl body calls {old.strip()!r} "
                f"{body.count(old)} times, not once; the capture has no anchor")
        body = body.replace(
            old, f"d{target} = pml_apply_reg(f{target}, u{target}, ", 1)
        prefix = f"        d{target} = pml_apply_reg(f{target}, u{target}, "
        call = _line_starting(body, prefix, f"target {target}'s captured pml_apply")
        if not call.endswith(");"):
            raise AssertionError(
                f"{name}'s certified curl body no longer closes target {target}'s "
                f"pml_apply on one line ({call!r}); the ownership flag has no "
                f"anchor to be appended at")
        body = body.replace(call, f"{call[:-2]}, own_{target});", 1)
    if "pml_apply(" in body:
        raise AssertionError(
            "a pml_apply call survived the capture rewrite; its value would be "
            "written to global memory and never read into the seam")
    return body + pre_clear_registers()


def constitutive_body_with_carry(tail: str, name: str) -> str:
    """The certified complex ``update_E`` statements, seamed, renamed and guarded.

    ``tail`` is the certified constitutive body BELOW its (dropped) index
    decomposition. Four kinds of edit: each ``cf s{t} = cf_load(g{t}, idx);``
    becomes the carried register (THE SEAM -- and what lets D be bound exactly
    once); each inverse-permittivity product is ASSERTED, not edited; each
    ``constitutive_apply(f{t}, ...)`` is renamed to ``h{t}`` with the
    HALF-INTEGER sub-lattice renamed ``kms_half_*``; and each renamed statement
    is wrapped in its component's ownership guard with that component's ghost
    blocks beside it.
    """
    for target in range(3):
        old = f"    cf s{target} = cf_load(g{target}, idx);\n"
        if tail.count(old) != 1:
            raise AssertionError(
                f"{name}'s certified constitutive body loads s{target} "
                f"{tail.count(old)} times, not once; the seam has no anchor")
        tail = tail.replace(
            old,
            f"    cf s{target} = d{target};   // THE SEAM: the register the curl "
            f"half just stored\n", 1)
    if "cf_load(g" in tail:
        raise AssertionError(
            "a source reload survived the seam rewrite; the constitutive half "
            "would need D bound a second time, which is the aliasing hazard the "
            "fused signature exists to avoid")
    for target in range(3):
        expected = (f"    s{target} = mul_field_left(s{target}, "
                    f"inv_eps_{target}[idx]);")
        line = _line_starting(tail, f"    s{target} = mul_field_left(",
                              f"target {target}'s inverse-permittivity product")
        if line != expected:
            raise AssertionError(
                f"target {target}'s inverse-permittivity product is {line!r}, "
                f"not the certified {expected!r}; this carry binds inv_eps_* on "
                f"the strength of that line and does not edit it")
    for target, axis in zip(range(3), _AXIS_NAME):
        prefix = f"    constitutive_apply(f{target}, w{target}, idx, s{target}, "
        call = _line_starting(tail, prefix,
                              f"target {target}'s constitutive_apply")
        coordinate = _COORDINATE[target]
        expected = f"{prefix}kps_{axis}[{coordinate}], kms_{axis}[{coordinate}]);"
        if call != expected:
            raise AssertionError(
                f"target {target}'s constitutive_apply is {call!r}, not the "
                f"certified {expected!r}; the sub-lattice rename would be applied "
                f"to an argument list this module has not read")
        renamed = (call.replace(f"(f{target},", f"(h{target},", 1)
                       .replace(f"kms_{axis}[", f"kms_half_{axis}[", 1))
        tail = tail.replace(call, "\n".join(
            [f"    if (own_{target}) {{", f"        {renamed.strip()}"]
            + fill_carry_blocks(target)
            + ["    }"]), 1)
    for target in range(3):
        if f"constitutive_apply(f{target}," in tail:
            raise AssertionError(
                f"a constitutive_apply still targets f{target}; it is D in the "
                f"curl body and E here, and the two would collide. (The ghost "
                f"blocks' cf_store(f{target}, ...) lines are D stores and are "
                f"correct.)")
    for target, axis in zip(range(3), _AXIS_NAME):
        if f"kms_{axis}[{_COORDINATE[target]}])" in tail:
            raise AssertionError(
                f"kms_{axis} survived the sub-lattice rename; it is the INTEGER "
                f"vector in the curl body and the HALF-INTEGER one here, and the "
                f"two would collide on the bare name")
    return tail


def _line_starting(body: str, prefix: str, what: str) -> str:
    """The one line of ``body`` starting with ``prefix``, or a named failure."""
    matches = [line for line in body.splitlines() if line.startswith(prefix)]
    if not matches:
        raise AssertionError(
            f"the certified body no longer carries {what} (looked for a line "
            f"starting {prefix!r}); this carry LIFTS that line rather than "
            f"retyping it and cannot splice around its absence")
    if len(matches) > 1:
        raise AssertionError(
            f"{prefix!r} matches {len(matches)} lines in the certified body; the "
            f"lift of {what} would take an arbitrary one")
    return matches[0]


def fills_plan(grid: Any) -> Dict[str, Tuple[Any, ...]]:
    """The two fills' launch plan: near flags, far reflect rows, mirror phases.

    The same three readings ``in_seam_coverage`` gives the stand-alone passes,
    packed as the flat per-axis triples the fused kernels take. RAISES rather
    than returning a plan it cannot stand behind.
    """
    phases = mirror_fill_phases(grid)
    rows = folded_far_rows(grid)
    walls = zero_metal_axes(grid)
    for axis in range(3):
        if phases[axis] is not None and int(phases[axis]) not in (1, -1):
            raise ValueError(
                f"axis {axis} is folded but its mirror phase is {phases[axis]!r}; "
                f"a plane's parity is +1 or -1 and the even-mirror default "
                f"standing in for a plane that declared otherwise is a run wrong "
                f"by twice the field wherever the parity mattered")
        if phases[axis] is not None and bool(walls[axis]):
            raise ValueError(
                f"axis {axis} is reported both folded and walled; "
                f"stepping._zero_metal excludes a folded axis by construction and "
                f"this kernel's ghost carry relies on the two sets being disjoint")
        if rows[axis] is not None and phases[axis] is None:
            raise ValueError(
                f"axis {axis} carries a far reflect row {rows[axis]!r} and no "
                f"mirror phase; stepping._fill_folded_far_ghosts weights that "
                f"image with the plane's parity and cannot run without one")
    return {
        "near": tuple(int(phases[axis] is not None) for axis in range(3)),
        "reflect": tuple(-1 if rows[axis] is None else int(rows[axis])
                         for axis in range(3)),
        "phase": tuple(0.0 if phases[axis] is None else float(phases[axis])
                       for axis in range(3)),
    }


def complex_electric_fill_reasons(fields: Any, grid: Any) -> Optional[str]:
    """Every reason this carry cannot serve the two fills on this run, or None.

    The B-side complex carry's clause set, restated for the D seam -- the one
    difference being which array's shape the launch walks (``Dx``). First
    refusal wins, this directory's convention.
    """
    mirrored_reader = getattr(grid, "is_mirrored", None)
    has_symmetry = getattr(grid, "has_symmetry", None)
    if not callable(mirrored_reader) or not callable(has_symmetry):
        return ("grid does not expose is_mirrored/has_symmetry; this seam cannot "
                "tell whether the two fills run inside it")
    try:
        folded = tuple(bool(mirrored_reader(axis)) for axis in range(3))
        symmetry = bool(has_symmetry())
    except Exception as exc:  # noqa: BLE001 - an unanswerable axis is not a clean one
        return (f"grid could not answer is_mirrored/has_symmetry: "
                f"{type(exc).__name__}: {exc}")
    if symmetry != any(folded):
        return ("the grid reports has_symmetry() and no mirrored axis, or the "
                "reverse; this carry is decided per axis and cannot be read off "
                "a grid that disagrees with itself")
    try:
        if any(bool(grid.is_axis(axis)) for axis in range(3)):
            return ("cylindrical radial axis: stepping._mirror_phases puts "
                    "(-1)**m in the mirror-phase slot for r, whose per-component "
                    "sign rule is r_to_minus_r_symmetry's rather than "
                    "mirror_parity's")
    except Exception as exc:  # noqa: BLE001
        return (f"the grid could not say whether an axis is the cylindrical "
                f"radial one: {type(exc).__name__}: {exc}")
    try:
        phases = mirror_fill_phases(grid)
    except Exception as exc:  # noqa: BLE001
        return (f"the grid could not state its mirror phases: "
                f"{type(exc).__name__}: {exc}")
    for axis, phase in enumerate(phases):
        if phase is not None and int(phase) not in (1, -1):
            return (f"axis {axis} is folded with mirror phase {phase!r}; "
                    f"stepping._symmetry_phase carries +1 or -1 and raises on "
                    f"None rather than folding with the even default")
    try:
        past = stored_past_owned(grid)
        rows = folded_far_rows(grid)
        stored = tuple(int(grid.stored_cells(axis)) for axis in range(3))
        metallic = tuple(bool(grid.is_metallic(axis)) for axis in range(3))
    except Exception as exc:  # noqa: BLE001
        return (f"the grid could not state its fold extents: "
                f"{type(exc).__name__}: {exc}")
    for axis in range(3):
        if not folded[axis]:
            continue
        if past[axis] is None:
            return (f"axis {axis} is folded and this grid cannot say whether it "
                    f"stores the slot past MEEP's owned window; the far ghost "
                    f"cannot be decided from is_metallic alone")
        if bool(past[axis]) == metallic[axis]:
            return (f"axis {axis} is folded with stored_cells > owned_cells "
                    f"{bool(past[axis])} and is_metallic {metallic[axis]}; the "
                    f"two routes to the fold's termination disagree and neither "
                    f"can be trusted")
        row = rows[axis]
        if row is not None and not (0 <= row <= stored[axis] - 2):
            return (f"axis {axis}'s far ghost would image stored row {row} of "
                    f"{stored[axis]}; the image row must be inside the array and "
                    f"below the slot being written")
    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            return (f"grid does not expose {name}; zero_metal_D cannot be "
                    f"carried in registers beside the fills")
    try:
        walls = zero_metal_axes(grid)
    except Exception as exc:  # noqa: BLE001
        return (f"in_seam_coverage.zero_metal_axes raised on this grid: "
                f"{type(exc).__name__}: {exc}")
    for axis in range(3):
        if folded[axis] and bool(walls[axis]):
            return (f"axis {axis} is reported both folded and walled; "
                    f"stepping._zero_metal excludes a folded axis by construction "
                    f"and the near carry writes no wall line at the ghost it "
                    f"images")
    try:
        stored = tuple(int(grid.stored_cells(axis)) for axis in range(3))
        extents = tuple(int(n) for n in fields.Dx.shape)
    except Exception as exc:  # noqa: BLE001
        return (f"grid could not state its stored extents: "
                f"{type(exc).__name__}: {exc}")
    if stored != extents:
        return (f"grid.stored_cells is {stored} but the launch walks Dx.shape "
                f"{extents}; the two fills' rows are derived from the first and "
                f"indexed into the second")
    for axis in range(3):
        if folded[axis] and extents[axis] <= NEAR_SOURCE_INDEX:
            return (f"folded axis {axis} stores {extents[axis]} cells, so the "
                    f"near fill's source row {NEAR_SOURCE_INDEX} does not exist "
                    f"(stepping._mirror_source raises on it)")
    return None
