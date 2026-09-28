"""The two mirror fills carried through a COMPLEX fused magnetic seam -- shared emitters.

WHAT THIS IS. ``fused_magnetic_pair`` carries ``fill_symmetry_bc_B`` and
``fill_folded_far_ghosts_B`` through the OWNERSHIP INVERSION -- the source thread
writes every imaged ghost from a live register, the destination thread is carved
out -- in real float32 arithmetic. This module is that same machinery under
COMPLEX64 STORAGE, spelled once so the two products that need it
(:mod:`.complex_folded_fused_magnetic_pair` and
:mod:`.complex_beta_fused_magnetic_pair`) cannot drift apart on it: their two
certified curl sources genuinely share the fold spelling
(``complex_beta_kernels.beta_source`` is ``complex_folded_kernels.folded_source``
plus the beta insert, which never touches a fill), so one carry serves both.

NOTHING HERE IMPORTS CUPY OR NUMPY AT MODULE SCOPE: the emitters splice strings
and the clauses read a grid, and both must run on the census laptop.

=============================================================================
WHAT CHANGES UNDER COMPLEX STORAGE, AND WHAT WAS MEASURED BEFORE IT WAS SPELLED
=============================================================================

1. **THE PARITY MULTIPLY IS A FULL COMPLEX PRODUCT, NEVER A SCALE.** The array
   path computes ``phase * plane`` with ``phase`` a Python int and ``plane``
   complex64 on the engine backend (stepping.py:1451, :1529-1532); the backend
   promotes the scalar and runs its complex multiply loop, whose zero cross terms
   carry the field's other word's SIGN into an addend. MEASURED on the GPU host
   (2026-09-01, one RTX A6000, CuPy 13.5.1): over 4108 complex words including
   signed zeros and subnormals, ``mul_coefficient_left(+-1, z)`` matched CuPy's
   ``parity * z`` on every word under both arms, while the plain two-word scale
   differed on the signed-zero rows (1 word at +1, 2 at -1). So each parity is
   applied through the certified ``mul_coefficient_left`` -- the coefficient is
   the LEFT operand, exactly as ``np.multiply(phase, plane)`` binds it.

2. **THE CHAIN ORDER IS REAL AND IS THE DRIVER'S.** A corner ghost carries the
   COMPOSITION of the passes that imaged it, one multiply per pass -- never the
   pre-multiplied parity product. MEASURED in the same probe:
   ``(-1) * ((-1) * z)`` differs from ``(+1) * z`` in 4 of 8216 words. The order
   is transcribed from the driver: ``fill_symmetry_bc_B`` (driver.py:3294) runs to
   completion before ``fill_folded_far_ghosts_B`` (:3296), and the far pass loops
   its axes in ASCENDING order (stepping._fill_folded_far_ghosts:1518), each pass
   reading the plane the previous one wrote -- so the near application is
   INNERMOST, then the far axes ascending, which is also what the Metal sibling
   measured and transcribed (``metal_kernels/folded_complex_fused_magnetic_pair.
   parity_chain``, ``results/complex_parity_chain_order_2026-08-21/``).

3. **THE OWNERSHIP RULE IS UNCHANGED.** Every step of
   ``fused_magnetic_pair.carried_destinations``' argument is an INDEX fact -- it
   mentions no dtype -- and the closed form transfers to word pairs unchanged
   (``results/metal_folded_far_carry_2026-08-20/ownership.json`` is where it was
   measured against the array path's own three passes). The near ghost still
   takes the DESTINATION's coefficient entry at stored index 0, because the near
   fill images along the very axis ``update_H`` indexes that component on; a far
   ghost takes the source thread's own entry unchanged.

4. **THE FILL PREDICATES ARE RESTATED MINUS ONE CLAUSE.** ``in_seam_coverage``'s
   ``covers_fill_symmetry`` / ``covers_fill_folded_far`` refuse complex64 storage
   by name, because the certified STAND-ALONE fill kernels index float32. The
   PLANNERS (``mirror_fill_phases``, ``folded_far_rows``, ``zero_metal_axes``,
   ``stored_past_owned``) are grid-only and dtype-blind, so they are read from
   that module -- one spelling -- while :func:`complex_fill_reasons` carries the
   remaining clauses itself, including the three the carry adds that the
   stand-alone passes never had to ask (an axis both folded and walled, a stored
   shape that disagrees with the launched array, a folded axis too short to hold
   the near source row).

WHAT THIS MODULE DOES NOT DECIDE: which configurations a product admits. Each
product's own predicate conjoins its two certified halves' predicates and THEN
asks :func:`complex_fill_reasons`; this module never answers for a curl.
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
    "complex_fill_reasons", "constitutive_body_with_carry", "far_fill_axes",
    "fill_carry_blocks", "fills_plan", "near_fill_axes",
    "ownership_declarations", "parity_chain", "pml_apply_reg_prelude",
    "zero_metal_carry",
]

#: ``stepping.MIRROR_SOURCE_INDEX`` under this seam's own name -- the fold's magic 2.
NEAR_SOURCE_INDEX = MIRROR_SOURCE_INDEX

#: The three B targets by index, and the certified complex template's own spellings
#: for each axis's coordinate, stride and extent (all declared by the curl body).
_TARGETS: Tuple[str, ...] = ("Bx", "By", "Bz")
_AXIS_NAME: Tuple[str, ...] = ("x", "y", "z")
_COORDINATE: Tuple[str, ...] = ("i", "j", "k")
_STRIDE: Tuple[str, ...] = ("sx", "sy", "sz")
_EXTENT: Tuple[str, ...] = ("nx", "ny", "nz")

#: The fill scalars' runtime names, in the fused signatures' own spelling.
_NEAR_FLAG: Tuple[str, ...] = ("near_x", "near_y", "near_z")
_REFLECT: Tuple[str, ...] = ("reflect_x", "reflect_y", "reflect_z")
_FAR_ACTIVE: Tuple[str, ...] = ("reflect_x >= 0", "reflect_y >= 0", "reflect_z >= 0")
_PHASE: Tuple[str, ...] = ("phase_x", "phase_y", "phase_z")

#: ``stepping._zero_metal``'s B DIAGONAL as (register index, wall flag, coordinate).
#: Spelled here rather than imported from anything D-shaped: the D family's table is
#: the off-diagonal complement, and reusing it would clear the wrong two components
#: on every walled run. ``IYEE_SHIFTS`` gives Bx (0,1,1), By (1,0,1), Bz (1,1,0).
_ZERO_METAL_ROWS: Tuple[Tuple[int, str, str], ...] = (
    (0, "wall_x", "i"),
    (1, "wall_y", "j"),
    (2, "wall_z", "k"),
)

#: The anchors of the certified COMPLEX ``pml_apply``, and the rewrite that turns it
#: into an OWNED value. The expression, its parenthesisation and the store below the
#: guard are the certified ones; ``fu`` stays ABOVE the guard because ``step_B``
#: writes it at every cell and neither fill touches it (stepping.py:1451 writes
#: ``field``, never ``fu_field``).
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
    "    // fu above is UNGUARDED: step_B writes it at every cell and neither\n"
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

#: The last line of the certified index decomposition -- where the carried registers
#: and the ownership flags are declared.
_DECODE_END = "    int i = idx / (ny * nz);\n"


def near_fill_axes(target: int) -> Tuple[int, ...]:
    """Which axes ``fill_symmetry_bc_B`` can image for ONE B component.

    ``stepping._fill_symmetry_ghost_cells`` (:1440-1447) writes axis ``a`` for
    component ``m`` exactly when the plane's phase is declared and
    ``iyee[m][a] == 0``. The Yee test is structural and is what this returns; the
    phase is a runtime flag. Read off ``IYEE_SHIFTS`` rather than returned as
    ``(target,)``, because the D family's answer is the exact complement and a
    hand-written constant here is the one place that inversion is plausible.
    """
    shifts = IYEE_SHIFTS[_TARGETS[target]]
    return tuple(axis for axis in range(3) if shifts[axis] == 0)


def far_fill_axes(target: int) -> Tuple[int, ...]:
    """Which axes ``fill_folded_far_ghosts_B`` can image for ONE B component.

    ``stepping._fill_folded_far_ghosts`` (:1524-1532) writes axis ``a`` for
    component ``m`` exactly when ``_stored_past_owned`` holds there and
    ``iyee[m][a] == 1`` -- the exact complement of :func:`near_fill_axes` on the B
    family, which is the whole geometric content of the far carry.
    """
    shifts = IYEE_SHIFTS[_TARGETS[target]]
    return tuple(axis for axis in range(3) if shifts[axis] == 1)


def carried_destinations(near: Sequence[int], far: Sequence[int]
                         ) -> Tuple[Tuple[Tuple[int, ...], bool], ...]:
    """Every ghost cell ONE source thread owns, as ``(far axes at top, near?)``.

    The ownership rule, PORTED rather than invented --
    ``fused_magnetic_pair.carried_destinations`` is the real-arithmetic original
    and ``results/metal_folded_far_carry_2026-08-20/ownership.json`` the
    measurement (9 configurations, 27 rows, zero differing words). Every step of
    the argument is an INDEX fact, so it transfers to complex word pairs
    unchanged. Returned smallest-subset-first for a stable emission order.
    """
    far = tuple(int(axis) for axis in far)
    combos: List[Tuple[Tuple[int, ...], bool]] = []
    for size in range(len(far) + 1):
        for subset in itertools.combinations(far, size):
            for carries_near in ((False, True) if near else (False,)):
                if not subset and not carries_near:
                    continue  # the thread's OWN cell, not a ghost
                combos.append((subset, carries_near))
    return tuple(sorted(combos, key=lambda item: (len(item[0]), item[0], item[1])))


def parity_chain(near: Sequence[int], subset: Sequence[int], carries_near: bool
                 ) -> Tuple[Tuple[int, str], ...]:
    """The ORDERED ``(axis, pass)`` applications one ghost's value carries.

    THE ORDER IS THE DRIVER'S AND IS TRANSCRIBED, NEVER CHOSEN: the near pass runs
    to completion first (driver.py:3294 before :3296) and the far pass loops its
    axes ascending (stepping.py:1567, :1575), each pass reading the plane the one
    before wrote. Back-substituting a corner therefore gives

        c_far[a2] (x) ( c_far[a1] (x) ( c_near[n] (x) v ) ),    a1 < a2

    and the emitter applies one ``mul_coefficient_left`` per entry in exactly this
    sequence. Under complex64 the order MOVES BYTES -- see the module docstring's
    measured chain probe -- so this function is load-bearing, not documentation.
    """
    chain: List[Tuple[int, str]] = []
    if carries_near:
        if not near:
            raise AssertionError(
                "a destination carrying the near fill was enumerated for a "
                "component with no near axis; carried_destinations and "
                "near_fill_axes have drifted")
        chain.append((int(near[0]), "near"))
    previous = -1
    for axis in subset:
        axis = int(axis)
        if axis <= previous:
            raise AssertionError(
                f"the far axis subset {tuple(subset)!r} is not strictly ascending; "
                f"stepping._fill_folded_far_ghosts applies its axes in ascending "
                f"order and this chain transcribes that order")
        previous = axis
        chain.append((axis, "far"))
    return tuple(chain)


def ownership_declarations() -> str:
    """``own_0``/``own_1``/``own_2`` -- is this thread's own cell imaged by a fill?

    THE ONE PLACE THE INVERSION IS DECIDED; every other emitter here reads these
    three flags. A cell either fill writes is OWNED BY ITS SOURCE THREAD: the
    thread standing on it computes the curl and stores ``fu`` (which the fills
    never touch) and then stops -- it never loads or stores B at its own cell and
    never touches H or f_w_H there, so no load it issues can land on a word
    another block writes in this launch.
    """
    lines = [
        "    // --- the ownership inversion (fill_symmetry_bc_B, "
        "fill_folded_far_ghosts_B) ---",
        "    // A cell a fill images is written by its SOURCE thread, from a live",
        "    // register, below. The thread standing ON it stops after fu: forming",
        "    // the displacement would read a B word another block writes in this",
        "    // same launch, and an ordinary CUDA launch has no grid-wide barrier",
        "    // at which that load is defined. The array path discards that",
        "    // displacement too -- the fill overwrites it (driver.py:3294, :3296).",
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
                f"every B component one shift-0 axis and two shift-1 axes, so an "
                f"empty test means the Yee table has drifted")
        near = "".join(_AXIS_NAME[axis] for axis in near_fill_axes(target))
        far = "".join(_AXIS_NAME[axis] for axis in far_fill_axes(target))
        lines.append(f"    // {name} {IYEE_SHIFTS[name]}: near fill on {near}, "
                     f"far fill on {far}.")
        lines.append(f"    int own_{target} = !({' || '.join(tests)});")
    return "\n".join(lines) + "\n"


def zero_metal_carry() -> str:
    """``zero_metal_B`` (driver.py:3295) for all three targets, on the registers.

    The B DIAGONAL, stored cell 0, the word pair ``(+0.0f, +0.0f)`` -- what
    ``stepping._zero_metal`` assigns as the integer 0 into a complex64 array. The
    register is cleared BESIDE the store so the constitutive half and any far
    image read the wiped value; GUARDED ON ``own_*`` because a cell a fill images
    belongs to its source thread (and the array path agrees: the far fill at
    :3296 overwrites the wall pass at :3295 on such a cell).
    """
    lines = [
        "    // --- zero_metal_B, carried in registers ---------------------------",
        "    // stepping._zero_metal (:2206-2245): the B DIAGONAL -- Bx clears on",
        "    // an x wall, By on y, Bz on z, stored cell 0 only. A folded metallic",
        "    // axis carries NO wall (is_metallic and not is_mirrored, :2237-2239),",
        "    // asserted on the host, so the walls and the fills stay disjoint.",
        "    // GUARDED ON own_*: a cell a fill images is its source thread's, and",
        "    // the array path's own order agrees (:3295 is overwritten by :3296).",
    ]
    for target, flag, coordinate in _ZERO_METAL_ROWS:
        lines.append(
            f"    if (own_{target} && {flag} && {coordinate} == 0) "
            f"{{ b{target} = cf_zero(); cf_store(f{target}, idx, b{target}); }}")
    return "\n".join(lines) + "\n"


def fill_carry_blocks(target: int, indent: str = "        ") -> List[str]:
    """Every imaged ghost this thread owns, then ``update_H`` at each of them.

    One block per destination in :func:`carried_destinations`, guarded on the
    flags that say THIS thread is that destination's source -- stored
    :data:`NEAR_SOURCE_INDEX` on the near axis, the runtime reflect row on each
    far axis. EVERY BLOCK IS EMITTED INSIDE the component's ownership guard by
    :func:`constitutive_body_with_carry`, because with two fills live a thread
    can be the source of one and the destination of the other, and unguarded it
    would write a ghost from a register that was never formed.

    THE VALUE IS THE PARITY CHAIN APPLIED TO ``b{target}``, one certified
    ``mul_coefficient_left`` per pass in :func:`parity_chain`'s order -- never the
    pre-multiplied product, which moves bytes under complex storage (the module
    docstring's measured probe). ``b{target}`` is read AFTER the wall clear, so a
    far image of a cleared row carries the cleared word pair with the sign the
    complex multiply gives it, exactly as the array path's pass order composes.
    """
    near = near_fill_axes(target)
    far = far_fill_axes(target)
    name = _TARGETS[target]
    if near != (target,):
        raise AssertionError(
            f"{name} is a NEAR-fill destination on axes {near}; for the B family "
            f"that set is exactly the component's own axis")
    if len(far) != 2 or target in far:
        raise AssertionError(
            f"{name} is a FAR-fill destination on axes {far}; on the B family the "
            f"near and far sets are complementary")
    walls = tuple(index for index, _flag, _coordinate in _ZERO_METAL_ROWS
                  if index == target)
    if set(walls) & set(far):
        raise AssertionError(
            f"{name} images a FAR ghost along an axis zero_metal_B also clears; "
            f"_ZERO_METAL_ROWS is the B diagonal and far_fill_axes excludes the "
            f"component's own axis, so the ghost would no longer inherit its "
            f"source thread's clear")
    inner = indent + "    "
    lines: List[str] = []
    for subset, carries_near in carried_destinations(near, far):
        tag = (f"g{target}_" + "".join(_AXIS_NAME[axis] for axis in subset)
               + ("n" if carries_near else ""))
        guards: List[str] = []
        terms: List[str] = []
        described: List[str] = []
        for axis in subset:
            guards.append(f"{_FAR_ACTIVE[axis]} && "
                          f"{_COORDINATE[axis]} == {_REFLECT[axis]}")
            terms.append(f"+ ({_EXTENT[axis]} - 1 - {_REFLECT[axis]}) "
                         f"* {_STRIDE[axis]}")
            described.append(
                f"the far fill on {_AXIS_NAME[axis]}: {_COORDINATE[axis]} = "
                f"{_EXTENT[axis]} - 1 imaged from {_REFLECT[axis]}, weight "
                f"-{_PHASE[axis]} (stepping._fill_folded_far_ghosts:1524-1532)")
        if carries_near:
            axis = near[0]
            guards.append(f"{_NEAR_FLAG[axis]} && "
                          f"{_COORDINATE[axis]} == {NEAR_SOURCE_INDEX}")
            terms.append(f"- {NEAR_SOURCE_INDEX} * {_STRIDE[axis]}")
            described.append(
                f"the near fill on {_AXIS_NAME[axis]}: {_COORDINATE[axis]} = 0 "
                f"imaged from {NEAR_SOURCE_INDEX}, weight {_PHASE[axis]} "
                f"(stepping._fill_symmetry_ghost_cells:1440-1447)")
        coefficient = "0" if carries_near else _COORDINATE[target]
        chain = parity_chain(near, subset, carries_near)
        lines.append(f"{indent}if ({' && '.join(guards)}) {{")
        lines.extend(f"{inner}// {text}" for text in described)
        lines.append(f"{inner}int {tag}_i = idx {' '.join(terms)};")
        first_axis, first_pass = chain[0]
        first = (_PHASE[first_axis] if first_pass == "near"
                 else f"({_PHASE[first_axis]} * -1.0f)")
        lines.append(f"{inner}// THE PARITY CHAIN, one certified complex multiply "
                     f"per pass, in the")
        lines.append(f"{inner}// driver's own composition order (near innermost, "
                     f"then far ascending);")
        lines.append(f"{inner}// pre-multiplying the parities moves bytes under "
                     f"complex storage.")
        lines.append(f"{inner}cf {tag}_v = mul_coefficient_left({first}, "
                     f"b{target});")
        for axis, pass_name in chain[1:]:
            weight = (_PHASE[axis] if pass_name == "near"
                      else f"({_PHASE[axis]} * -1.0f)")
            lines.append(f"{inner}{tag}_v = mul_coefficient_left({weight}, "
                         f"{tag}_v);")
        lines.append(f"{inner}cf_store(f{target}, {tag}_i, {tag}_v);")
        if carries_near:
            lines.append(
                f"{inner}// The DESTINATION's own coefficient entry: stored index "
                f"0 on {_AXIS_NAME[target]}, NOT this")
            lines.append(
                f"{inner}// thread's at {NEAR_SOURCE_INDEX}: update_H indexes "
                f"{name} on {_COORDINATE[target]} and the near fill images")
            lines.append(
                f"{inner}// along that same axis, so reusing the source's pair "
                f"would apply stored")
            lines.append(
                f"{inner}// cell {NEAR_SOURCE_INDEX}'s absorber profile to stored "
                f"cell 0.")
        else:
            lines.append(
                f"{inner}// A far ghost does not move {_COORDINATE[target]}, the "
                f"axis update_H indexes this")
            lines.append(
                f"{inner}// component on, so it takes this thread's own pair "
                f"unchanged.")
        lines.append(
            f"{inner}constitutive_apply(h{target}, w{target}, {tag}_i, {tag}_v, "
            f"kps_{_AXIS_NAME[target]}[{coefficient}], "
            f"kms_int_{_AXIS_NAME[target]}[{coefficient}]);")
        lines.append(f"{indent}}}")
    return lines


def pml_apply_reg_prelude(prelude: str, name: str) -> str:
    """One family's certified complex prelude with ``pml_apply`` OWNED and valued.

    Three anchored rewrites and no fourth: the signature gains the return type,
    the parameter list gains ``owned``, and the closing load+store is guarded and
    named. ``constitutive_apply`` is emitted UNTOUCHED -- the seam changes where
    its ``src`` comes from, not what it does with it. Raises on a moved anchor
    rather than emitting a kernel that is quietly missing the guard.
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
    """One family's certified curl body with the three registers captured OWNED.

    Hoists the ``cf b0/b1/b2`` declarations and the ownership flags above the
    certified braced blocks (a value declared inside one does not outlive it),
    then rewrites each ``pml_apply(f{t}, u{t}, ...)`` call to
    ``b{t} = pml_apply_reg(f{t}, u{t}, ..., own_{t})``. Every anchor is asserted;
    a fold mask, a Bloch block or a beta insert between the decode and the calls
    rides through untouched, which is what lets ONE function serve both the
    folded and the beta curl bodies.
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
        "    cf b0 = cf_zero();\n"
        "    cf b1 = cf_zero();\n"
        "    cf b2 = cf_zero();\n"
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
            old, f"b{target} = pml_apply_reg(f{target}, u{target}, ", 1)
        prefix = f"        b{target} = pml_apply_reg(f{target}, u{target}, "
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
    return body


def constitutive_body_with_carry(tail: str, name: str) -> str:
    """The certified complex ``update_H`` statements, seamed, renamed and guarded.

    ``tail`` is the certified constitutive body BELOW its (dropped) index
    decomposition. Three kinds of edit: each ``cf s{t} = cf_load(g{t}, idx);``
    becomes the carried register (THE SEAM -- and what lets B be bound exactly
    once); each ``constitutive_apply(f{t}, ...)`` is renamed to ``h{t}`` with the
    INTEGER sub-lattice renamed ``kms_int_*``; and each renamed statement is
    wrapped in its component's ownership guard with that component's ghost blocks
    beside it.
    """
    for target in range(3):
        old = f"    cf s{target} = cf_load(g{target}, idx);\n"
        if tail.count(old) != 1:
            raise AssertionError(
                f"{name}'s certified constitutive body loads s{target} "
                f"{tail.count(old)} times, not once; the seam has no anchor")
        tail = tail.replace(
            old,
            f"    cf s{target} = b{target};   // THE SEAM: the register the curl "
            f"half just stored\n", 1)
    if "cf_load(g" in tail:
        raise AssertionError(
            "a source reload survived the seam rewrite; the constitutive half "
            "would need B bound a second time, which is the aliasing hazard the "
            "fused signature exists to avoid")
    for target, axis in zip(range(3), _AXIS_NAME):
        prefix = f"    constitutive_apply(f{target}, w{target}, idx, s{target}, "
        call = _line_starting(tail, prefix,
                              f"target {target}'s constitutive_apply")
        expected = (f"{prefix}kps_{axis}[{_COORDINATE[target]}], "
                    f"kms_{axis}[{_COORDINATE[target]}]);")
        if call != expected:
            raise AssertionError(
                f"target {target}'s constitutive_apply is {call!r}, not the "
                f"certified {expected!r}; the sub-lattice rename would be applied "
                f"to an argument list this module has not read")
        renamed = (call.replace(f"(f{target},", f"(h{target},", 1)
                       .replace(f"kms_{axis}[", f"kms_int_{axis}[", 1))
        tail = tail.replace(call, "\n".join(
            [f"    if (own_{target}) {{", f"        {renamed.strip()}"]
            + fill_carry_blocks(target)
            + ["    }"]), 1)
    for target in range(3):
        if f"constitutive_apply(f{target}," in tail:
            raise AssertionError(
                f"a constitutive_apply still targets f{target}; it is B in the "
                f"curl body and H here, and the two would collide. (The ghost "
                f"blocks' cf_store(f{target}, ...) lines are B stores and are "
                f"correct.)")
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
    packed as the flat per-axis triples the fused kernels take -- the complex twin
    of ``fused_magnetic_pair.fused_magnetic_pair_fills``, spelled against the same
    planners so the carry and the certified passes cannot disagree about which
    axes each fill visits. RAISES rather than returning a plan it cannot stand
    behind: the failure mode is a plane of wrong values, not an exception.
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


def complex_fill_reasons(fields: Any, grid: Any) -> Optional[str]:
    """Every reason this carry cannot serve the two fills on this run, or None.

    THE COMPLEX RESTATEMENT of ``in_seam_coverage.covers_fill_symmetry`` /
    ``covers_fill_folded_far`` -- their clauses minus the one this seam exists to
    change (those predicates refuse complex64 storage because the STAND-ALONE fill
    kernels index float32; this carry is the complex spelling) -- plus the three
    clauses the carry adds that a stand-alone pass never had to ask. First refusal
    wins, this directory's convention.
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
    # THE THREE CLAUSES THE CARRY ADDS, from the real pair's own list: each is a
    # fact the stand-alone passes never had to ask, because each launched over a
    # plane of its own AFTER the curl had finished.
    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            return (f"grid does not expose {name}; zero_metal_B cannot be "
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
        extents = tuple(int(n) for n in fields.Bx.shape)
    except Exception as exc:  # noqa: BLE001
        return (f"grid could not state its stored extents: "
                f"{type(exc).__name__}: {exc}")
    if stored != extents:
        return (f"grid.stored_cells is {stored} but the launch walks Bx.shape "
                f"{extents}; the two fills' rows are derived from the first and "
                f"indexed into the second")
    for axis in range(3):
        if folded[axis] and extents[axis] <= NEAR_SOURCE_INDEX:
            return (f"folded axis {axis} stores {extents[axis]} cells, so the "
                    f"near fill's source row {NEAR_SOURCE_INDEX} does not exist "
                    f"(stepping._mirror_source raises on it)")
    return None
