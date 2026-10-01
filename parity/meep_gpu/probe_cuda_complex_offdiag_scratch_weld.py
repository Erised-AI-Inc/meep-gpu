"""The COMPLEX stencil weld's arithmetic, measured on the host before a line of PTX.

WHAT THIS PROBE IS FOR. The CUDA board records two COMPLEX off-diagonal cells as
STENCIL-BLOCKED -- ``D_to_E (cuda_complex_folded/folded complex, cuda_complex_offdiag/
complex off-diagonal PML)`` (3 demand rows, 1 clearing the source injection) and
``D_to_E (cuda_complex_no_pml/complex no-PML curl, cuda_complex_offdiag/complex
off-diagonal no-PML)`` (2 demand rows, 1 clearing) -- with the verdict "the
constitutive half reads the curl half's in-place output at a cell this thread does not
own". That verdict is TRUE OF THE IN-PLACE WELD. It says nothing about the
SCRATCH-OUTPUT shape, in which the curl half writes a LAUNCH-LOCAL allocation, the
constitutive half RECOMPUTES each foreign cell from pre-launch state through the same
``__device__`` function, and the launcher rotates the bindings afterwards.

``probe_cuda_offdiag_scratch_weld.py`` measured that closed form under FLOAT32 storage
over eleven fixtures. This probe is its COMPLEX64 twin, and it exists because the
closed form is not dtype-blind: **every multiply in it changes**, and the change is
exactly the class a float32 probe cannot see.

=============================================================================
THE THREE COMPLEX FACTS THIS PROBE IS BUILT TO CATCH
=============================================================================

1. **THE PARITY IS A FULL COMPLEX PRODUCT, NOT A WORD SCALE.** The array path spells
   a mirror parity ``phase * plane`` with ``phase`` a Python int and ``plane``
   complex64 (stepping.py:1451, :1529-1532); the backend promotes the scalar and runs
   its complex multiply loop, whose zero cross terms carry the OTHER word's sign into
   an addend. A plane-wise ``{re*w, im*w}`` is byte-wrong on signed zeros. Armed here
   as ``parity_as_word_scale``.
2. **AN EVEN MIRROR MAY NOT BE SKIPPED.** ``1 * z`` is not ``z``: the fused arm's real
   word is ``fma(1, z.re, (0*z.im) * -1.0f)``, which turns ``re = -0.0`` with a
   negative imaginary part into ``+0.0``. Armed as ``parity_skipped_at_plus_one``.
3. **THE WALL CLEAR IS BOTH WORDS.** ``stepping._zero_metal`` assigns the INTEGER 0 to
   a complex64 array, which is ``(+0.0f, +0.0f)``; a clear that touched only the real
   plane is a known defect class in this tree. Armed as ``clear_real_word_only``.

None of the three can be armed by a fixture whose values came from ``rng.normal``,
because such a fixture never puts an exact zero where a parity can reach it -- and a
probe that shipped them as silent no-ops would pass VACUOUSLY. That failure has
already happened once on this track and is recorded in ``the test coverage notes``. So this
probe carries a SECOND LEG whose input is an ADVERSARIAL word catalogue: exact +0.0
and -0.0 real words beside negative imaginary words, both signed subnormals, and
normals, tiled over the whole volume so every redirect source, every cleared plane and
every reflect row lands on one. The two legs answer different questions and both are
required:

* **leg 1, THE DRIVER-ORDER IDENTITY.** Two COMPLETE driver steps, byte for byte as
  uint32 over every stored volume, with the D seam replaced by the closed form. This
  is the claim the product makes.
* **leg 2, THE ADVERSARIAL RAW.** The three in-seam passes run by ``stepping`` itself
  on a planted D, against the closed form evaluated on the SAME planted D. Same claim,
  on the input that makes the three facts above decidable.

WHAT IT DOES NOT DO. It is not a gate and certifies nothing: NumPy is not NVRTC, and
the substitution, the launch counts and the device mutation battery belong to
``gate_cuda_complex_offdiag_stencil_welds.py``. What it removes is the possibility of
discovering the ARITHMETIC is wrong from inside a device slot.

Host-only: NumPy, no CuPy. Progress reporting: one flushed line per case, one fsynced JSONL row.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "keep")
HERE = Path(__file__).resolve().parent
API_ROOT = next(parent for parent in HERE.parents
                if (parent / "meep_gpu" / "cuda_kernels").is_dir())
for _path in (str(API_ROOT), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402

import cuda_predicate_battery as battery  # noqa: E402
import metal_composition_matrix as matrix  # noqa: E402
from cuda_predicate_battery import _GridWithCupyBackend  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.cuda_kernels.coverage import (  # noqa: E402
    OFFDIAG_TRANSVERSE_PARTNERS,
)
from meep_gpu.fields import IYEE_SHIFTS, mirror_parity  # noqa: E402

D_COMPONENTS: Tuple[str, str, str] = ("Dx", "Dy", "Dz")
E_COMPONENTS: Tuple[str, str, str] = ("Ex", "Ey", "Ez")

#: Every stored volume either walk can touch. The comparison reads the whole set, so
#: a weld that got E right and fu wrong is a failure rather than a pass.
STATE_NAMES: Tuple[str, ...] = tuple(
    [f"{stem}{axis}" for stem in ("B", "D", "E", "H") for axis in "xyz"]
    + [f"fu_{stem}{axis}" for stem in ("B", "D") for axis in "xyz"]
    + [f"f_w_{stem}{axis}" for stem in ("E", "H") for axis in "xyz"])

#: The driver's pass order for one complete step with no sources and no poles
#: (driver.py:3300-3335). Inert passes no-op identically on both walks.
FULL_STEP: Tuple[Tuple[str, Callable[[Any, Any], None]], ...] = (
    ("step_B", lambda f, p: stepping.step_B(f, p)),
    ("fill_B", lambda f, p: stepping.fill_symmetry_bc_B(f)),
    ("zero_metal_B", lambda f, p: stepping.zero_metal_B(f)),
    ("far_B", lambda f, p: stepping.fill_folded_far_ghosts_B(f)),
    ("update_H", lambda f, p: stepping.update_H(f, p)),
    ("step_D", lambda f, p: stepping.step_D(f, p)),
    ("fill_D", lambda f, p: stepping.fill_symmetry_bc_D(f)),
    ("zero_metal_D", lambda f, p: stepping.zero_metal_D(f)),
    ("far_D", lambda f, p: stepping.fill_folded_far_ghosts_D(f)),
    ("update_E", lambda f, p: stepping.update_E(f, p)),
)

#: The three D-seam passes the weld absorbs, by the name :data:`FULL_STEP` gives them.
CARRIED_PASSES = ("fill_D", "zero_metal_D", "far_D")

#: The same three, as the callables leg 2 drives directly, in the DRIVER'S ORDER.
IN_SEAM_PASSES: Tuple[Tuple[str, Callable[[Any], None]], ...] = (
    ("fill_symmetry_bc_D", stepping.fill_symmetry_bc_D),
    ("zero_metal_D", stepping.zero_metal_D),
    ("fill_folded_far_ghosts_D", stepping.fill_folded_far_ghosts_D),
)


def log(message: str) -> None:
    print(message, flush=True)


def words(array: Any) -> np.ndarray:
    return np.frombuffer(np.ascontiguousarray(array).tobytes(), dtype=np.uint32)


def differing(left: Any, right: Any) -> int:
    a, b = words(left), words(right)
    if a.shape != b.shape:
        return max(a.size, b.size)
    return int(np.count_nonzero(a != b))


def state_of(fields: Any) -> Dict[str, Any]:
    return {name: getattr(fields, name) for name in STATE_NAMES
            if getattr(fields, name, None) is not None}


def frozen(fields: Any) -> Dict[str, np.ndarray]:
    return {name: np.array(value, copy=True)
            for name, value in state_of(fields).items()}


def restore(fields: Any, snapshot: Dict[str, np.ndarray]) -> None:
    for name, value in snapshot.items():
        getattr(fields, name)[...] = value


def compare(left: Dict[str, np.ndarray],
            right: Dict[str, np.ndarray]) -> Dict[str, int]:
    assert set(left) == set(right), sorted(set(left) ^ set(right))
    return {name: n for name in sorted(left)
            if (n := differing(left[name], right[name]))}


# ---------------------------------------------------------------------------
# Seeding: the trap, and the catalogue that springs it
# ---------------------------------------------------------------------------
#
# THE COMPLEX-PARITY TRAP, SPELLED IN CODE. A complex64 plane assembled as
# ``re + 1j*im`` is ITSELF a complex multiply -- ``1j * im`` runs the multiply loop
# and rewrites zero signs -- so a probe that built its fixtures that way would be
# measuring a plane it did not intend to write, and every signed-zero question would
# be answered about the wrong bytes. Every write below goes through ``.real[...]``
# and ``.imag[...]``, which are real-valued VIEWS: the float32 words land exactly as
# given.

#: Word pairs the parity multiply and the wall clear can be told apart on. The first
#: six are the whole point: a real word of either zero sign beside an imaginary word
#: of either sign is where ``phase * z`` and ``{re*w, im*w}`` disagree, and where
#: ``1 * z`` is not ``z``. The subnormals are carried because the two float32
#: subnormal policies are a live axis on this board and a flushed subnormal is a
#: signed zero.
_SMALLEST_SUBNORMAL = float(np.float32(np.uint32(0x00000001).view(np.float32)))
ADVERSARIAL_WORDS: Tuple[Tuple[float, float], ...] = (
    (0.0, -1.5), (-0.0, -1.5), (0.0, 1.5), (-0.0, 2.5),
    (-0.0, -0.0), (0.0, 0.0),
    (_SMALLEST_SUBNORMAL, -0.0), (-_SMALLEST_SUBNORMAL, 0.0),
    (0.37, -0.0), (-0.0, 0.37),
    (1.0, -0.0), (-1.0, 0.0),
    (0.5, -0.25), (-0.75, 0.125),
)


def pack(real: Any, imag: Any) -> np.complex64:
    """A complex64 scalar from two float32 WORDS, with no complex multiply.

    THE TRAP AGAIN, one scalar down: ``real + 1j*imag`` multiplies, and a multiply
    is exactly what rewrites the zero signs this probe exists to watch. numpy
    scalars are immutable, so the words are written into a one-element array's real
    and imaginary VIEWS and the element is read back out.
    """
    out = np.empty(1, dtype=np.complex64)
    out.real[0] = np.float32(real)
    out.imag[0] = np.float32(imag)
    return out[0]


def seed_state(fields: Any, seed: int) -> None:
    """Physical-band values in every stored volume, written WORD BY WORD.

    A zero-filled volume agrees with a zero-filled volume, so the identity would be
    vacuous without this. Complex volumes get two INDEPENDENT real draws written
    into ``.real`` and ``.imag`` rather than one complex draw, for the reason above.
    """
    rng = np.random.default_rng(seed)
    for name in STATE_NAMES:
        array = getattr(fields, name, None)
        if array is None:
            continue
        if np.iscomplexobj(array):
            array.real[...] = rng.normal(0.0, 0.37, size=array.shape).astype(
                np.float32)
            array.imag[...] = rng.normal(0.0, 0.37, size=array.shape).astype(
                np.float32)
        else:
            array[...] = rng.normal(0.0, 0.37, size=array.shape).astype(array.dtype)


def planted_volume(shape: Tuple[int, int, int], offset: int) -> np.ndarray:
    """A complex64 volume tiled from :data:`ADVERSARIAL_WORDS`, words written direct.

    TILED OVER THE FLAT INDEX so every plane the resolution can touch -- stored row
    0, stored row ``MIRROR_ROW``, the reflect row, the top plane and the interior --
    carries several entries of the catalogue rather than one; ``offset`` rotates the
    tiling per component so the three do not agree cell for cell.
    """
    count = int(np.prod(shape))
    index = (np.arange(count) + offset) % len(ADVERSARIAL_WORDS)
    real = np.array([ADVERSARIAL_WORDS[i][0] for i in index], dtype=np.float32)
    imag = np.array([ADVERSARIAL_WORDS[i][1] for i in index], dtype=np.float32)
    out = np.empty(shape, dtype=np.complex64)
    out.real[...] = real.reshape(shape)
    out.imag[...] = imag.reshape(shape)
    return out


# ---------------------------------------------------------------------------
# The grid facts the resolution reads
# ---------------------------------------------------------------------------

def grid_facts(fields: Any) -> Dict[str, Any]:
    """Exactly the runtime plan the kernel takes as ``int``/``float`` arguments.

    Read off the same three engine functions the shipped launcher reads
    (``stepping._stored_past_owned``, ``._far_reflect_rows``, ``grid.mirror_phase``),
    so a divergence between this probe and the product cannot come from a second
    reading of the grid.
    """
    grid = fields.grid
    folded = tuple(axis for axis in range(3) if grid.is_mirrored(axis))
    walled = tuple(axis for axis in range(3)
                   if grid.is_metallic(axis) and not grid.is_mirrored(axis))
    far_axes = tuple(axis for axis in folded
                     if stepping._stored_past_owned(grid, axis))
    return {"folded": folded, "walled": walled, "far_axes": far_axes,
            "reflect": stepping._far_reflect_rows(grid),
            "phases": {axis: int(grid.mirror_phase(axis)) for axis in folded},
            "shape": tuple(int(n) for n in grid.shape)}


# ---------------------------------------------------------------------------
# The closed form, PER CELL -- the kernel's complex ``resolve_D``, in Python
# ---------------------------------------------------------------------------
#
# ONE SCALAR FUNCTION, ONE RAW TAP, and every multiply a FULL COMPLEX PRODUCT with
# the coefficient on the LEFT -- ``np.complex64(weight) * value``, which is what
# ``np.multiply(python_int, complex64)`` runs after promotion and is the array path's
# own spelling. The three pass applications are the DRIVER'S ORDER (near fill, wall
# clear, far fill) and under complex storage that order MOVES BYTES, because
# ``(-1) * (0+0j)`` is ``(-0, +0)`` rather than ``(0, 0)`` -- so the clear REPLACES
# the near arm rather than following it.


def resolve_cell(component: str, cell: Tuple[int, int, int], raw: np.ndarray,
                 facts: Dict[str, Any], *,
                 drop_parity: bool = False,
                 drop_clear: bool = False,
                 parity_as_word_scale: bool = False,
                 parity_skipped_at_plus_one: bool = False,
                 clear_real_word_only: bool = False,
                 near_source_row: int = 2,
                 drop_far: bool = False,
                 flip_clear_order: bool = False,
                 halo_source_unstepped: bool = False,
                 pre_launch: Optional[np.ndarray] = None,
                 ledger: Optional[List[Tuple[int, int, int]]] = None,
                 ) -> Any:
    """The post-pass value of ``component`` at ``cell``, from the RAW stepped array.

    The keyword arms are the NULL CONTROLS; each must move bytes on a fixture where
    the machinery it disables is live, or the identity is vacuous for that arm.
    ``ledger``, when supplied, records every raw cell this call consulted -- the
    kernel reads exactly one, and the caller asserts it.
    """
    iyee = IYEE_SHIFTS[component]
    own_axis = "xyz".index(component[1])
    shape = facts["shape"]
    scalar = raw.dtype.type
    near_axes = tuple(axis for axis in facts["folded"] if iyee[axis] == 0)
    clear_axes = tuple(axis for axis in facts["walled"] if iyee[axis] == 0)
    is_far_axis = (own_axis in facts["far_axes"]) and not drop_far
    reflect_row = facts["reflect"][own_axis] if is_far_axis else None

    def parity(axis: int) -> int:
        if drop_parity:
            return 1
        return int(mirror_parity(component, axis, facts["phases"][axis]))

    def apply_parity(weight: int, value: Any) -> Any:
        """ONE multiply, the array path's own. The two nulls live here.

        ``parity_as_word_scale`` is the plane-wise ``{re*w, im*w}`` -- the spelling a
        real-storage port reaches for, byte-wrong on signed zeros.
        ``parity_skipped_at_plus_one`` is the even-mirror shortcut, which is correct
        on float32 and is not on complex64.
        """
        if parity_skipped_at_plus_one and int(weight) == 1:
            return value
        if parity_as_word_scale:
            return pack(np.float32(value.real) * np.float32(weight),
                        np.float32(value.imag) * np.float32(weight))
        return scalar(weight) * value

    def zero() -> Any:
        """``_zero_metal``'s integer 0 assigned into a complex64 array."""
        if clear_real_word_only:
            return pack(0.0, 0.0)
        return scalar(0)

    def tap(z: Tuple[int, int, int]) -> Any:
        if ledger is not None:
            ledger.append(z)
        source = raw if not halo_source_unstepped or pre_launch is None else pre_launch
        return source[z]

    def near_value(z: Tuple[int, int, int]) -> Any:
        source = list(z)
        weights = []
        for axis in near_axes:                    # X, Y, Z order: the fill's own
            if z[axis] == 0:
                source[axis] = near_source_row
                weights.append(parity(axis))
        value = tap(tuple(source))
        for weight in weights:                    # nested, one multiply per plane
            value = apply_parity(weight, value)
        return value

    def cleared(z: Tuple[int, int, int]) -> bool:
        if drop_clear:
            return False
        return any(z[axis] == 0 for axis in clear_axes)

    def after_clear(z: Tuple[int, int, int]) -> Any:
        if flip_clear_order:
            # NULL: the parity applied AFTER the clear at a near ghost -- the wrong
            # order, which under complex storage turns (+0, +0) into (-0, +0).
            source = list(z)
            weights = []
            for axis in near_axes:
                if z[axis] == 0:
                    source[axis] = near_source_row
                    weights.append(parity(axis))
            base = zero() if cleared(z) else tap(tuple(source))
            for weight in weights:
                base = apply_parity(weight, base)
            return base
        if cleared(z):
            if ledger is not None:
                ledger.append(z)      # the clear still OWNS this cell: one tap
            return zero()
        return near_value(z)

    if reflect_row is not None and cell[own_axis] == shape[own_axis] - 1:
        redirected = list(cell)
        redirected[own_axis] = reflect_row
        value = after_clear(tuple(redirected))
        # mirror_parity(c, own_axis, phase) IS the far fill's -phase for a shift-1
        # component (fields.py:180-182); nothing else multiplies it.
        return apply_parity(parity(own_axis), value)
    return after_clear(cell)


def resolved_volume(component: str, raw: np.ndarray, facts: Dict[str, Any],
                    *, in_place: bool = False, **arms: Any) -> np.ndarray:
    """:func:`resolve_cell` over every cell -- the scratch the kernel writes.

    ``in_place`` is the SCRATCH-ALIASED-TO-STORAGE null: the resolution reads the
    array it is writing, in C-raster order, which is what a weld that skipped the
    scratch allocation would compute.
    """
    out = raw if in_place else np.empty_like(raw)
    source = out if in_place else raw
    nx, ny, nz = facts["shape"]
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                out[i, j, k] = resolve_cell(component, (i, j, k), source, facts,
                                            **arms)
    return out


# ---------------------------------------------------------------------------
# The tap ledger and the race census
# ---------------------------------------------------------------------------

def tap_ledger(component: str, facts: Dict[str, Any], raw: np.ndarray
               ) -> Dict[str, Any]:
    """How many RAW cells each resolved cell consults. The kernel reads one."""
    nx, ny, nz = facts["shape"]
    counts: Dict[int, int] = {}
    redirected = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                seen: List[Tuple[int, int, int]] = []
                resolve_cell(component, (i, j, k), raw, facts, ledger=seen)
                counts[len(set(seen))] = counts.get(len(set(seen)), 0) + 1
                if set(seen) != {(i, j, k)}:
                    redirected += 1
    return {"distinct_raw_cells_per_output_cell":
                {str(k): v for k, v in sorted(counts.items())},
            "cells_whose_source_is_not_themselves": redirected}


def constitutive_taps(component_index: int, rows: Dict[str, Sequence[str]],
                      facts: Dict[str, Any]
                      ) -> List[Tuple[int, Tuple[int, int, int],
                                      Tuple[int, int, int]]]:
    """Every (partner component, own cell, tapped cell) the off-diagonal stencil reads.

    Transcribed from ``stepping._offdiagonal_terms`` (stepping.py:1235-1251) the way
    ``complex_offdiag_update_e._term_lines`` assembles it: the pair half a cell DOWN
    the PARTNER's axis and the same pair half a cell UP the component's OWN axis, so
    the four samples are home, down, up and the corner. Only the three that are not
    ``home`` can be foreign.
    """
    nx, ny, nz = facts["shape"]
    name = E_COMPONENTS[component_index]
    own_axis = "xyz".index(name[1])
    partners = rows.get(name, ())
    taps: List[Tuple[int, Tuple[int, int, int], Tuple[int, int, int]]] = []
    for offset in range(2):
        partner_name = E_COMPONENTS[
            OFFDIAG_TRANSVERSE_PARTNERS[component_index][offset]]
        if partner_name not in partners:
            continue
        partner_axis = "xyz".index(partner_name[1])
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    home = (i, j, k)
                    down = list(home)
                    down[partner_axis] = home[partner_axis] - 1
                    up = list(home)
                    up[own_axis] = home[own_axis] + 1
                    corner = list(up)
                    corner[partner_axis] = home[partner_axis] - 1
                    for sample in (down, up, corner):
                        if all(0 <= sample[a] < facts["shape"][a]
                               for a in range(3)):
                            taps.append((partner_axis, home, tuple(sample)))
    return taps


def race_census(rows: Dict[str, Sequence[str]], facts: Dict[str, Any],
                pre_launch: Dict[str, np.ndarray],
                resolved: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """How many foreign taps the IN-PLACE weld would read at an undefined time.

    A tap is HAZARDOUS when the value the constitutive must see (the resolved one)
    differs from the value that cell held before the launch: in an in-place weld the
    load returns one or the other depending on which block ran first, and the answer
    is a schedule. Zero hazardous taps would mean the scratch design bought nothing
    on this fixture, so the number is reported per fixture rather than asserted once.
    BOTH WORDS are compared -- a hazard that moved only the imaginary word is a
    hazard.
    """
    total = 0
    hazardous = 0
    for component_index in range(3):
        for partner_axis, _home, sample in constitutive_taps(component_index, rows,
                                                             facts):
            name = D_COMPONENTS[partner_axis]
            total += 1
            before = words(np.asarray(pre_launch[name][sample], dtype=np.complex64))
            after = words(np.asarray(resolved[name][sample], dtype=np.complex64))
            if not np.array_equal(before, after):
                hazardous += 1
    return {"foreign_taps": total, "hazardous_taps": hazardous,
            "hazard_fraction": round(hazardous / total, 6) if total else None}


# ---------------------------------------------------------------------------
# The two walks -- leg 1
# ---------------------------------------------------------------------------

def reference_step(fields: Any, pml: Any) -> None:
    for _name, action in FULL_STEP:
        action(fields, pml)


def scheme_step(fields: Any, pml: Any, facts: Dict[str, Any],
                capture: Optional[Dict[str, Any]] = None, **arms: Any) -> None:
    """One complete step with the D seam replaced by the scratch-output weld.

    ``step_D`` supplies the RAW array (in the kernel it is recomputed per tap from
    pre-launch D/fu/H; here it is the same pure function evaluated once, which is
    the identity the device gate re-measures with two independent launch counters).
    The per-cell resolution then maps raw -> final into a SCRATCH array, the
    write-back IS the rotation, and the array path's own ``update_E`` plays the
    certified constitutive half -- consuming exactly the taps the fused kernel
    re-derives.
    """
    skip_rotation = bool(arms.pop("skip_rotation", False))
    for name, action in FULL_STEP:
        if name in CARRIED_PASSES:
            continue                       # carried by the closed form
        if name == "step_D":
            before = {c: np.array(getattr(fields, c), copy=True)
                      for c in D_COMPONENTS}
            action(fields, pml)
            raw = {c: np.array(getattr(fields, c), copy=True)
                   for c in D_COMPONENTS}
            final = {c: resolved_volume(c, raw[c], facts, pre_launch=before[c],
                                        **arms)
                     for c in D_COMPONENTS}
            if capture is not None:
                capture["pre_launch"] = before
                capture["raw"] = raw
                capture["resolved"] = final
            for c in D_COMPONENTS:
                getattr(fields, c)[...] = before[c] if skip_rotation else final[c]
            continue
        action(fields, pml)


# ---------------------------------------------------------------------------
# Leg 2 -- the adversarial raw, against stepping's own three passes
# ---------------------------------------------------------------------------

def adversarial_leg(fields: Any, facts: Dict[str, Any], seed: int
                    ) -> Dict[str, Any]:
    """The three in-seam passes on a PLANTED D, against the closed form on the same.

    THE CLAIM IS THE SAME as leg 1's and the INPUT is what differs: leg 1's D comes
    out of ``step_D`` and its words are ordinary normals, while this one's is tiled
    from :data:`ADVERSARIAL_WORDS`, so a signed zero sits under every redirect
    source, every cleared plane and every reflect row. The three complex nulls are
    armed HERE, because they are exactly the arms an all-normal input cannot tell
    apart from the truth.
    """
    planted = {name: planted_volume(facts["shape"], offset)
               for offset, name in enumerate(D_COMPONENTS)}
    for name in D_COMPONENTS:
        getattr(fields, name)[...] = planted[name]
    for _label, action in IN_SEAM_PASSES:
        action(fields)
    reference = {name: np.array(getattr(fields, name), copy=True)
                 for name in D_COMPONENTS}
    resolved = {name: resolved_volume(name, planted[name], facts)
                for name in D_COMPONENTS}
    identical = {name: differing(reference[name], resolved[name])
                 for name in D_COMPONENTS}
    return {"planted_word_pairs": len(ADVERSARIAL_WORDS),
            "differing": {k: v for k, v in identical.items() if v},
            "planted": planted, "reference": reference}


# ---------------------------------------------------------------------------
# The arms: no fixture may measure a configuration the product refuses
# ---------------------------------------------------------------------------

def arm_verdicts(fields: Any, pml: Any, family: str, licence: Any) -> Dict[str, Any]:
    """The SHIPPED PRODUCT's own predicate, asked modulo the CuPy backend.

    Not the two halves separately: the product's predicate conjoins them and adds
    the seam clauses, so asking it is what guarantees no fixture below measures a
    configuration the shipped launcher would decline.
    """
    from meep_gpu.cuda_kernels import (  # noqa: PLC0415
        complex_no_pml_offdiag_fused_electric_pair as no_pml_pair)
    from meep_gpu.cuda_kernels import (  # noqa: PLC0415
        folded_complex_offdiag_fused_electric_pair as folded_pair)

    proxy = _GridWithCupyBackend(fields.grid)
    predicate = (folded_pair.covers_folded_complex_offdiag_fused_electric_pair
                 if family == "F"
                 else no_pml_pair.covers_complex_no_pml_offdiag_fused_electric_pair)
    covered, reason = predicate(fields, pml, proxy, sources=(), license=licence,
                                subnormal_policy=battery.COMPLEX_POLICY)
    return {"product": {"covered": bool(covered), "reason": str(reason)}}


def run_case(name: str, family: str, build: Callable[[], Tuple[Any, Any]],
             rows: Dict[str, Sequence[str]], seed: int,
             expect: Dict[str, bool], licence: Any) -> Dict[str, Any]:
    started = time.time()
    fields, pml = build()
    facts = grid_facts(fields)
    arms = arm_verdicts(fields, pml, family, licence)
    assert arms["product"]["covered"], (
        name, "the shipped predicate refuses this fixture",
        arms["product"]["reason"])
    assert np.iscomplexobj(fields.Dx), (name, "the fixture is not complex storage")

    seed_state(fields, seed)
    pre = frozen(fields)

    reference_step(fields, pml)
    ref1 = frozen(fields)
    reference_step(fields, pml)
    ref2 = frozen(fields)

    restore(fields, pre)
    capture: Dict[str, Any] = {}
    scheme_step(fields, pml, facts, capture=capture)
    got1 = frozen(fields)
    scheme_step(fields, pml, facts)
    got2 = frozen(fields)

    step1 = compare(ref1, got1)
    step2 = compare(ref2, got2)

    # Vacuity floors: the machinery each case exists for must be LIVE on it.
    live = {
        "near": bool(facts["folded"]),
        "far": bool(facts["far_axes"]),
        "odd": any(phase == -1 for phase in facts["phases"].values()),
        "even": any(phase == 1 for phase in facts["phases"].values()),
        "wall": bool(facts["walled"]),
        "corner": len(facts["folded"]) > 1,
    }
    for key, wanted in expect.items():
        assert live.get(key, False) == wanted, (
            f"{name}: expected {key}={wanted} but the fixture answers "
            f"{live.get(key)} -- the case does not exercise what it claims")

    ledger = {c: tap_ledger(c, facts, capture["raw"][c]) for c in D_COMPONENTS}
    for component, entry in ledger.items():
        assert set(entry["distinct_raw_cells_per_output_cell"]) == {"1"}, (
            f"{name}/{component}: a resolved cell consulted "
            f"{sorted(entry['distinct_raw_cells_per_output_cell'])} raw cells; the "
            f"kernel's resolve_D reads exactly one and this shape would not fit it")
    race = race_census(rows, facts, capture["pre_launch"], capture["resolved"])
    assert race["foreign_taps"] > 0, (
        f"{name}: the off-diagonal stencil reads no foreign cell on this fixture, "
        f"so the case measures nothing about a stencil weld")

    # ----- LEG 2: the adversarial raw, on a fresh copy of the pre-step state -----
    restore(fields, pre)
    adversarial = adversarial_leg(fields, facts, seed)
    assert not adversarial["differing"], (
        name, "the closed form and stepping's own three passes disagree on the "
              "adversarial word catalogue", adversarial["differing"])

    nulls: Dict[str, Any] = {}

    def null(tag: str, reachable: bool, reason: str = "", *,
             adversarial_arm: bool = False, **arm: Any) -> None:
        """One null control. ``adversarial_arm`` runs it on LEG 2's planted input.

        The three complex arms are only decidable there: on leg 1's stepped normals
        the parity spellings agree word for word, and a null that could not fire
        would be reported as armed-and-silent, which is worse than predicted.
        """
        if not reachable:
            nulls[tag] = {"predicted_null": True, "reason": reason}
            return
        if adversarial_arm:
            broken = {c: resolved_volume(c, adversarial["planted"][c], facts, **arm)
                      for c in D_COMPONENTS}
            moved = {c: n for c in D_COMPONENTS
                     if (n := differing(adversarial["reference"][c], broken[c]))}
            nulls[tag] = {"differing": sum(moved.values()), "volumes": len(moved),
                          "leg": "adversarial"}
        else:
            restore(fields, pre)
            scheme_step(fields, pml, facts, **arm)
            moved = compare(ref1, frozen(fields))
            nulls[tag] = {"differing": sum(moved.values()), "volumes": len(moved),
                          "leg": "driver_order"}
        assert moved, (f"{name}/{tag}: the null control did not diverge; the "
                       f"identity above is vacuous for this arm")

    null("rotation_skipped", True, skip_rotation=True)
    null("parity_dropped", live["odd"],
         "every live parity is +1 on this fixture; the drop is the identity",
         drop_parity=True)
    null("near_redirect_row_wrong", live["near"],
         "no folded axis: no near redirect exists", near_source_row=1)
    null("far_redirect_dropped", live["far"],
         "no stored-past-owned axis: no far ghost exists", drop_far=True)
    null("clear_dropped", live["wall"],
         "no walled axis: zero_metal_D clears nothing on this fixture",
         drop_clear=True)
    null("clear_order_flipped", live["odd"] and live["wall"],
         "needs an odd parity meeting a cleared plane; not reachable here",
         flip_clear_order=True)
    null("halo_source_unstepped", live["near"] or live["far"],
         "no fold: the resolution never redirects, so there is no halo source to "
         "take from the wrong array",
         halo_source_unstepped=True)

    # ----- THE THREE COMPLEX ARMS, on the adversarial input -----
    null("parity_as_word_scale", live["near"] or live["far"],
         "no fold: no parity multiply happens at all on this fixture",
         adversarial_arm=True, parity_as_word_scale=True)
    null("parity_skipped_at_plus_one", live["even"],
         "no EVEN mirror plane on this fixture; the shortcut has nothing to skip",
         adversarial_arm=True, parity_skipped_at_plus_one=True)
    # THE WALL CLEAR'S SECOND WORD. `clear_real_word_only` writes +0.0f into BOTH
    # words here, which is what the array path does -- so on its own it is the
    # IDENTITY and would be a null that always predicts. What makes the question
    # decidable is that the DEVICE gate can bind a clear that leaves the imaginary
    # word alone, which no host expression of `stepping._zero_metal` can. Recorded
    # as predicted with that reason rather than shipped as a silent pass.
    nulls["clear_real_word_only"] = {
        "predicted_null": True,
        "reason": "not expressible on a host walk: numpy's assignment of the "
                  "integer 0 into a complex64 array writes both words, so a "
                  "'real-word-only' clear has no host spelling that is not simply "
                  "the truth. Armed on the device gate, where the store is two "
                  "explicit float32 writes",
    }
    # THE SCRATCH ITSELF IS NOT ARMABLE ON THIS WALK, and saying so is the point --
    # see the real-storage probe's note. The deterministic host analogue is the
    # in-place raster resolution, measured and reported.
    nulls["scratch_aliased_to_storage"] = {
        "predicted_null": True,
        "reason": "not expressible on a walk that resolves the whole volume before "
                  "the constitutive reads it; the in-place raster arm is measured "
                  "below and is the identity. Armed on the device gate instead",
        "in_place_raster_differing": {
            component: moved
            for component in D_COMPONENTS
            if (moved := differing(
                resolved_volume(component,
                                np.array(capture["raw"][component], copy=True),
                                facts, in_place=True),
                capture["resolved"][component]))},
    }

    record = {
        "case": name, "family": family,
        "grid": facts["shape"], "rows": {k: list(v) for k, v in rows.items()},
        "folded": facts["folded"], "walled": facts["walled"],
        "far_axes": facts["far_axes"], "phases": facts["phases"],
        "reflect_rows": facts["reflect"],
        "arms": arms,
        "step1_differing": step1, "step2_differing": step2,
        "adversarial_leg": {"planted_word_pairs":
                                adversarial["planted_word_pairs"],
                            "differing": adversarial["differing"]},
        "tap_ledger": ledger, "race_census": race,
        "nulls": nulls,
        "elapsed_s": round(time.time() - started, 2),
    }
    ok = not step1 and not step2 and not adversarial["differing"]
    note = ", ".join("%s:%s" % (key, value.get("differing", "predicted"))
                     for key, value in nulls.items())
    log(f"  {name:<38} {'IDENTICAL' if ok else 'DIVERGED ' + str(step1)}"
        f"  hazardous={race['hazardous_taps']}/{race['foreign_taps']}"
        f"  nulls={{{note}}} ({record['elapsed_s']} s)")
    assert ok, (name, step1, step2, adversarial["differing"])
    return record


# ---------------------------------------------------------------------------
# The fixtures -- every one admitted by a SHIPPED product's own predicate
# ---------------------------------------------------------------------------

_ONE_ROW: Dict[str, Sequence[str]] = {"Ex": ("Ey",)}
_ALL_ROWS: Dict[str, Sequence[str]] = {"Ex": ("Ey", "Ez"), "Ey": ("Ez", "Ex"),
                                       "Ez": ("Ex", "Ey")}

CASES: Tuple[Tuple[str, str, Callable[[], Tuple[Any, Any]],
                   Dict[str, Sequence[str]], Dict[str, bool]], ...] = (
    # --- Family F: folded complex PML curl x complex off-diagonal update_E -------
    ("F_fold_even", "F",
     lambda: matrix.folded(complex_storage=True, rows=_ONE_ROW), _ONE_ROW,
     dict(near=True, far=True, even=True)),
    ("F_fold_odd", "F",
     lambda: matrix.folded(phase=-1, complex_storage=True, rows=_ONE_ROW), _ONE_ROW,
     dict(near=True, far=True, odd=True)),
    ("F_fold_odd_all_rows", "F",
     lambda: matrix.folded(phase=-1, complex_storage=True, rows=_ALL_ROWS),
     _ALL_ROWS, dict(near=True, far=True, odd=True)),
    ("F_fold_odd_count", "F",
     lambda: matrix.folded(complex_storage=True, rows=_ALL_ROWS, extent=2.1),
     _ALL_ROWS, dict(near=True, far=True, even=True)),
    # THE OTHER TERMINATION. A folded METALLIC axis has no far ghost at all
    # (``_stored_past_owned`` is False), so the far machinery is dead and the near
    # fill is the whole of the fold -- a mutation caught on one termination is not
    # caught on the other.
    ("F_fold_metallic_termination", "F",
     lambda: matrix.folded(boundaries={"y": "metallic"}, complex_storage=True,
                           rows=_ALL_ROWS), _ALL_ROWS,
     dict(near=True, far=False, even=True)),
    # THE ONE SHAPE THAT ARMS ``clear_order_flipped``: an ODD parity meeting a
    # CLEARED plane, which needs a wall on an axis the fold does not own.
    ("F_fold_odd_walled", "F",
     lambda: matrix.folded(phase=-1, boundaries={"x": "metallic"},
                           complex_storage=True, rows=_ALL_ROWS), _ALL_ROWS,
     dict(near=True, odd=True, wall=True)),
    ("F_fold_two_axes_mixed_phase", "F",
     lambda: matrix.folded(axis="XY", phase=(1, -1), complex_storage=True,
                           rows=_ALL_ROWS), _ALL_ROWS,
     dict(near=True, corner=True, odd=True, even=True)),
    ("F_fold_3d", "F",
     lambda: matrix.folded(complex_storage=True, rows=_ALL_ROWS, depth=1.2),
     _ALL_ROWS, dict(near=True, far=True, even=True)),
    # --- Family N: complex no-PML curl x complex off-diagonal update_E -----------
    # NO FOLD REACHES THIS FAMILY -- its curl half refuses one by name -- so every
    # fold arm below is a PREDICTED null with that reason, and the wall clear is the
    # whole of the in-seam work.
    ("N_no_pml_one_row", "N",
     lambda: matrix.cart(pml=0, complex_storage=True, storage=False,
                         rows=_ONE_ROW), _ONE_ROW, dict(near=False, wall=False)),
    ("N_no_pml_all_rows", "N",
     lambda: matrix.cart(pml=0, complex_storage=True, storage=False,
                         rows=_ALL_ROWS), _ALL_ROWS, dict(near=False, wall=False)),
    ("N_no_pml_walls_all_rows", "N",
     lambda: matrix.cart(pml=0, complex_storage=True, storage=False,
                         rows=_ALL_ROWS, boundaries="metallic"), _ALL_ROWS,
     dict(near=False, wall=True)),
    ("N_no_pml_2d_all_rows", "N",
     lambda: matrix.flat(pml_cells=0, complex_storage=True, storage=False,
                         rows=_ALL_ROWS), _ALL_ROWS, dict(near=False, wall=False)),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True,
                        help="DIRECTORY to write probe.json and progress.jsonl into")
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    trail = (args.out / "progress.jsonl").open("w")

    matrix.prepare_environment()
    licence = battery._complex_offdiag_license()

    log("=" * 78)
    log("THE HAND-CUDA COMPLEX STENCIL WELDS -- host arithmetic, no device")
    log("=" * 78)
    log(f"policy      : {os.environ['MEEP_GPU_SUBNORMAL_POLICY']}")
    log(f"licence arm : {licence.get('arm')}")
    log(f"cases       : {len(CASES)}")
    started = time.time()
    records: List[Dict[str, Any]] = []
    for index, (name, family, build, rows, expect) in enumerate(CASES, start=1):
        record = run_case(name, family, build, rows, args.seed + index, expect,
                          licence)
        record["case_index"] = f"{index}/{len(CASES)}"
        records.append(record)
        trail.write(json.dumps(record) + "\n")
        trail.flush()
        os.fsync(trail.fileno())
    trail.close()

    payload = {
        "probe": "cuda COMPLEX offdiag scratch-output welds -- host arithmetic",
        "claim": "the per-cell closed-form resolution of the raw stepped complex D "
                 "equals the driver's fill_symmetry_bc_D / zero_metal_D / "
                 "fill_folded_far_ghosts_D pass order byte for byte -- over two "
                 "COMPLETE driver steps on ordinary values, AND over an adversarial "
                 "word catalogue of signed zeros and subnormals run through "
                 "stepping's own three passes -- on every fixture the two shipped "
                 "complex off-diagonal products admit; each resolved cell consults "
                 "exactly ONE raw cell; and the foreign taps the in-place weld would "
                 "race on are counted rather than argued",
        "certifies": "NOTHING. NumPy is not NVRTC. The substitution, the launch "
                     "counts and the mutation battery belong to the device gate",
        "policy": os.environ["MEEP_GPU_SUBNORMAL_POLICY"],
        "expansion_licence_arm": licence.get("arm"),
        "adversarial_word_pairs": [list(pair) for pair in ADVERSARIAL_WORDS],
        "cases": records,
        "totals": {
            "cases": len(records),
            "identical_step1_and_step2": sum(
                1 for r in records
                if not r["step1_differing"] and not r["step2_differing"]),
            "adversarial_identical": sum(
                1 for r in records if not r["adversarial_leg"]["differing"]),
            "foreign_taps": sum(r["race_census"]["foreign_taps"] for r in records),
            "hazardous_taps": sum(r["race_census"]["hazardous_taps"]
                                  for r in records),
            "nulls_armed": sum(1 for r in records for v in r["nulls"].values()
                               if "differing" in v),
            "nulls_predicted_unreachable": sum(
                1 for r in records for v in r["nulls"].values()
                if v.get("predicted_null")),
        },
        "elapsed_s": round(time.time() - started, 2),
    }
    (args.out / "probe.json").write_text(json.dumps(payload, indent=1) + "\n")
    log("-" * 78)
    log(f"identical   : {payload['totals']['identical_step1_and_step2']}"
        f"/{payload['totals']['cases']} driver-order, "
        f"{payload['totals']['adversarial_identical']}"
        f"/{payload['totals']['cases']} adversarial")
    log(f"foreign taps: {payload['totals']['hazardous_taps']} hazardous of "
        f"{payload['totals']['foreign_taps']}")
    log(f"nulls       : {payload['totals']['nulls_armed']} armed, "
        f"{payload['totals']['nulls_predicted_unreachable']} predicted unreachable")
    log(f"wrote       : {args.out / 'probe.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
