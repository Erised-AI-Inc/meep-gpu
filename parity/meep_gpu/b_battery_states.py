"""Constructors for the B battery's gate states: full-volume, planted −0, ±0 lattice.

The B battery is the set of must-catch legs every table's gates carry once the curl
and constitutive sub-steps select their form by chunk membership
(``results/bit_parity_0_10/DESIGN.md`` D13, and KERNELS.md §7.2 beside it). Its
states have three requirements that the gates' current fixtures do not all meet, and
this module builds them once, so a gate on any table and a host test draw the same
words from the same seed:

* **Every cell live from step 1.** A random full-volume state (no zero word, no
  subnormal, both signs) exercises every chunk-boundary plane on the first launch,
  rather than when a wavefront arrives (:func:`full_volume_state`).
* **The two signed-zero cases a random state never draws.** The plain curl form
  ``f − curl`` and the recurrence with its auxiliary read as 0 differ only at
  ``f = −0, curl = +0`` (1 of 36 special pairs; ``formulation_alignment_scope_
  2026-10-05/verify_signed_zero.log``), and the plain constitutive form ``f = src``
  and the accumulation with zeroed inputs, ``(0 + 1·src) − 1·0``, differ on every
  ``src = −0`` (9,979 of 9,979; ``triton_identities.txt`` I4). Each is planted on a
  plain cell by :func:`plant_curl_negative_zero` and
  :func:`plant_constitutive_negative_zero`, which return the word each form must
  store.
* **Signed zeros everywhere at once.** :func:`signed_zero_lattice` fills every cell
  from the six-word special table (±0, ±1, ±the smallest subnormal) and checks that
  every ordered (cell, next cell) pair of table words occurs along every axis.

And one check: :func:`require_form_coverage` refuses, by name, a gate configuration
that leaves a (component × form) class empty, a membership line with no cell inside
or no cell outside its run on either sub-lattice, or no wrapping axis (D13's
form-coverage rule). The per-cell form masks and the membership lines are ARGUMENTS:
the layout that derives them from a lifted run (``chunk_map``) is a later work
package, and this module imports nothing from the package.

Host-only NumPy. ``±min subnormal`` words survive only under a policy that keeps
subnormals; under flush they read as ±0, which the battery's flush legs inherit.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

#: The six float32 words of ``verify_signed_zero.py``'s exhaustive table, as bits:
#: +0, −0, +1, −1, +min subnormal, −min subnormal.
SPECIAL_WORDS: Tuple[int, ...] = (0x00000000, 0x80000000, 0x3F800000, 0xBF800000,
                                  0x00000001, 0x80000001)
NEGATIVE_ZERO = 0x80000000
POSITIVE_ZERO = 0x00000000

#: The two sub-lattices a membership line is given on (integer and half-integer
#: Yee positions along the axis).
LATTICES: Tuple[str, str] = ("integer", "half")


def words(array: np.ndarray) -> np.ndarray:
    """The uint32 words of a float32 array."""
    return np.ascontiguousarray(array, dtype=np.float32).view(np.uint32)


def full_volume_state(names: Sequence[str], shape: Tuple[int, int, int], rng,
                      low: float = 0.25, high: float = 1.0) -> Dict[str, np.ndarray]:
    """One float32 array per name, every word a normal number of random sign.

    Magnitudes are uniform in ``[low, high)``; ``low`` must be a normal float32, so no
    word is zero or subnormal and no sign is lost. Draw order is the order of
    ``names``, so the same seed and names give the same words.
    """
    if not (np.float32(low) >= np.finfo(np.float32).tiny and high > low):
        raise ValueError(f"magnitudes [{low}, {high}) must be normal float32 and "
                         f"non-empty")
    state: Dict[str, np.ndarray] = {}
    for name in names:
        magnitude = rng.uniform(low, high, shape)
        sign = np.where(rng.random(shape) < 0.5, -1.0, 1.0)
        state[name] = (sign * magnitude).astype(np.float32)
    return state


def _neighbour(cell: Tuple[int, int, int], axis: int, shift: int,
               shape: Tuple[int, int, int], boundaries: Sequence[str]
               ) -> Tuple[int, int, int]:
    index = cell[axis] + shift
    if not 0 <= index < shape[axis]:
        if boundaries[axis] != "periodic":
            raise ValueError(f"cell {cell}: the stencil partner along axis {axis} is a "
                             f"ghost on a {boundaries[axis]} face; plant away from it")
        index %= shape[axis]
    moved = list(cell)
    moved[axis] = index
    return tuple(moved)  # type: ignore[return-value]


def plant_curl_negative_zero(state: Dict[str, np.ndarray],
                             term: Tuple[str, str, int, str, int],
                             cell: Tuple[int, int, int], shift: int,
                             boundaries: Sequence[str], dtdx: float = 0.5,
                             plain: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Plant ``f = −0`` with a curl of exactly ``+0`` at ``cell``, IN PLACE.

    ``term`` is ``(target, first, first_axis, second, second_axis)`` in
    ``no_pml_ref.B_TERMS`` / ``D_TERMS`` order; ``shift`` is the stencil's direction
    (+1 for ``step_B``, which reads ``g[i+1]``; −1 for ``step_D``). Each partner's
    shifted word is set equal to its word at ``cell``, so ``sf − f`` and ``s − ss``
    are ``+0`` and the curl is ``+0`` in the package grouping
    ``dtdx·((sf − f) + (s − ss))`` and in MEEP's ``dtdx·(((sf − f) + s) − ss)`` alike.
    Then MEEP's plain ``f − curl`` stores ``−0`` and the recurrence with its auxiliary
    read as 0 stores ``+0``.

    Refused by name: a cell inside a membership run (``plain`` False there; the
    planted case is about the plain form), a cell on a metallic face (its ownership
    mask and ghost decide the curl instead), and a stencil partner that is a ghost.
    """
    target, first, first_axis, second, second_axis = term
    shape = state[target].shape
    if plain is not None and not bool(plain[cell]):
        raise ValueError(f"cell {cell} of {target} is not a plain cell")
    for axis, kind in enumerate(boundaries):
        if kind != "periodic" and shape[axis] > 1 and cell[axis] in (0, shape[axis] - 1):
            raise ValueError(f"cell {cell} lies on the {kind} face of axis {axis}")
    first_partner = _neighbour(cell, first_axis, shift, shape, boundaries)
    second_partner = _neighbour(cell, second_axis, shift, shape, boundaries)
    state[first][first_partner] = state[first][cell]
    state[second][second_partner] = state[second][cell]
    state[target][cell] = np.float32(-0.0)

    f32 = np.float32
    sf, f = f32(state[first][first_partner]), f32(state[first][cell])
    ss, s = f32(state[second][second_partner]), f32(state[second][cell])
    package = f32(f32(dtdx) * f32(f32(sf - f) + f32(s - ss)))
    meep = f32(f32(dtdx) * f32(f32(f32(sf - f) + s) - ss))
    value = f32(state[target][cell])
    plain_form = f32(value - meep)
    # The recurrence with p := 0 and unit coefficients, operation by operation:
    # n = ((p·kms) − curl)·sinv; f = (((f·kms_u) + n) − p)·sinv_u.
    one, zero = f32(1.0), f32(0.0)
    n = f32(f32(f32(zero * one) - meep) * one)
    collapse = f32(f32(f32(f32(value * one) + n) - zero) * one)
    record = {
        "target": target, "cell": tuple(int(i) for i in cell),
        "partners": {first: tuple(int(i) for i in first_partner),
                     second: tuple(int(i) for i in second_partner)},
        "curl_word": {"package": int(words(np.array([package]))[0]),
                      "meep": int(words(np.array([meep]))[0])},
        "expected_word": int(words(np.array([plain_form]))[0]),
        "collapse_word": int(words(np.array([collapse]))[0]),
    }
    if (record["curl_word"] != {"package": POSITIVE_ZERO, "meep": POSITIVE_ZERO}
            or record["expected_word"] != NEGATIVE_ZERO
            or record["collapse_word"] != POSITIVE_ZERO):
        raise AssertionError(f"the planted curl case did not separate the forms: {record}")
    return record


def plant_constitutive_negative_zero(state: Dict[str, np.ndarray], source: str,
                                     cell: Tuple[int, int, int],
                                     inverse: Optional[np.ndarray] = None,
                                     plain: Optional[np.ndarray] = None
                                     ) -> Dict[str, Any]:
    """Plant ``src = −0`` on a plain constitutive cell, IN PLACE.

    ``source`` is the B or D component (H = B; E = D·χ⁻¹). The plain form stores
    ``−0`` (``−0·χ⁻¹`` is ``−0`` for a positive χ⁻¹); the accumulation with zeroed
    inputs, ``(0 + 1·src) − 1·0``, stores ``+0``. Refused by name: a cell inside a
    membership run, and a χ⁻¹ that is not positive there.
    """
    if plain is not None and not bool(plain[cell]):
        raise ValueError(f"cell {cell} of {source} is not a plain cell")
    state[source][cell] = np.float32(-0.0)
    f32 = np.float32
    src = f32(state[source][cell])
    if inverse is not None:
        scale = f32(inverse[cell])
        if not scale > 0:
            raise ValueError(f"chi^-1 at {cell} is {scale!r}, not positive")
        src = f32(src * scale)
    collapse = f32(f32(f32(0.0) + f32(f32(1.0) * src)) - f32(f32(1.0) * f32(0.0)))
    record = {"source": source, "cell": tuple(int(i) for i in cell),
              "expected_word": int(words(np.array([src]))[0]),
              "collapse_word": int(words(np.array([collapse]))[0])}
    if record["expected_word"] != NEGATIVE_ZERO or record["collapse_word"] != POSITIVE_ZERO:
        raise AssertionError(f"the planted constitutive case did not separate the forms: "
                             f"{record}")
    return record


def pair_coverage(array: np.ndarray, axis: int) -> int:
    """How many of the 36 ordered (cell, next cell) special-word pairs occur along ``axis``."""
    raw = words(array)
    table = {word: index for index, word in enumerate(SPECIAL_WORDS)}
    codes = np.vectorize(table.__getitem__)(raw) if raw.size else raw
    lead = np.take(codes, range(0, raw.shape[axis] - 1), axis=axis)
    follow = np.take(codes, range(1, raw.shape[axis]), axis=axis)
    return int(np.unique((lead * len(SPECIAL_WORDS) + follow).ravel()).size)


def signed_zero_lattice(names: Sequence[str], shape: Tuple[int, int, int], rng
                        ) -> Dict[str, np.ndarray]:
    """One float32 array per name, every word drawn from :data:`SPECIAL_WORDS`.

    Every ordered pair of table words occurs as (cell, next cell) along every axis of
    extent 2 or more, in every array: CHECKED, and refused by name when the shape is
    too small for the draw to cover them.
    """
    table = np.array(SPECIAL_WORDS, dtype=np.uint32)
    state: Dict[str, np.ndarray] = {}
    for name in names:
        drawn = table[rng.integers(0, table.size, size=shape)].view(np.float32)
        for axis in range(3):
            if shape[axis] > 1:
                covered = pair_coverage(drawn, axis)
                if covered != table.size ** 2:
                    raise ValueError(f"{name}: {covered} of {table.size ** 2} ordered "
                                     f"special-word pairs along axis {axis} for shape "
                                     f"{shape}; use a larger grid or another seed")
        state[name] = drawn
    return state


def form_counts(forms: Mapping[str, Mapping[str, np.ndarray]]) -> Dict[str, Dict[str, int]]:
    """Cells per (component × form), from per-cell boolean masks that partition the grid.

    ``forms[component][form]`` is True where that component's cell takes that form.
    Refused by name: masks of one component that overlap, or that leave a cell with
    no form.
    """
    counts: Dict[str, Dict[str, int]] = {}
    for component, masks in forms.items():
        stack = np.stack([np.asarray(mask, dtype=bool) for mask in masks.values()])
        per_cell = stack.sum(axis=0)
        if (per_cell > 1).any():
            raise ValueError(f"{component}: {int((per_cell > 1).sum())} cells take more "
                             f"than one form")
        if (per_cell == 0).any():
            raise ValueError(f"{component}: {int((per_cell == 0).sum())} cells take no "
                             f"form")
        counts[component] = {form: int(np.count_nonzero(mask))
                             for form, mask in masks.items()}
    return counts


def coverage_gaps(forms: Mapping[str, Mapping[str, np.ndarray]],
                  membership: Mapping[Tuple[int, str], np.ndarray],
                  periodic: Sequence[bool]) -> List[str]:
    """Every way a gate configuration falls short of D13's form-coverage rule, by name.

    ``membership[(axis, lattice)]`` is the 1-D in-run line of that axis on that
    sub-lattice (``lattice`` in :data:`LATTICES`); ``periodic[axis]`` says whether the
    axis wraps. An axis of extent 1 has no inside and outside to cover, and is not
    judged.
    """
    gaps: List[str] = []
    for component, per_form in form_counts(forms).items():
        gaps += [f"{component}:{form} has no cell"
                 for form, count in per_form.items() if count == 0]
    for axis in range(3):
        for lattice in LATTICES:
            key = (axis, lattice)
            if key not in membership:
                gaps.append(f"axis {axis} {lattice}: no membership line given")
                continue
            line = np.asarray(membership[key], dtype=bool)
            if line.size <= 1:
                continue
            if not line.any():
                gaps.append(f"axis {axis} {lattice}: no cell inside a run")
            if line.all():
                gaps.append(f"axis {axis} {lattice}: no cell outside every run")
    if not any(periodic):
        gaps.append("no wrapping axis")
    return gaps


def require_form_coverage(forms: Mapping[str, Mapping[str, np.ndarray]],
                          membership: Mapping[Tuple[int, str], np.ndarray],
                          periodic: Sequence[bool]) -> Dict[str, Dict[str, int]]:
    """The (component × form) counts, or a ValueError naming every coverage gap."""
    gaps = coverage_gaps(forms, membership, periodic)
    if gaps:
        raise ValueError("form coverage incomplete: " + "; ".join(gaps))
    return form_counts(forms)
