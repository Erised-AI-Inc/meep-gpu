"""Run each CERTIFIED kernel body against ``stepping`` on the configuration its clause refuses.

THE QUESTION, one per group: is the blocking clause over-broad (the body would have
been byte-exact) or load-bearing (the body genuinely diverges)? Only measurement
settles it — this project has been wrong in both directions.

HOW A BODY IS PUT ON A CONFIGURATION IT IS REFUSED ON. The shipped ``plan_*``
builder is used, unmodified, with its OWN coverage function temporarily replaced by
one that admits. That is deliberate and it is the only honest route: the builder is
what resolves the target arrays, the auxiliaries, the sources, the inverse epsilon
and — the part a hand-rolled harness gets wrong — WHICH Yee sub-lattice the PML
coefficient tables come from. Rebuilding that by hand would measure a harness, not
the shipped body. Nothing in ``triton_kernels`` is edited; the patch is a context
manager in this file and it is reverted in a ``finally``.

CASE DISCIPLINE
---------------
* **uint32 compare over the full stored inventory**, never ``allclose``; the first
  differing array and word index are recorded with both raw words.
* **non-power-of-two Courant** (0.35 / 0.3) so no coefficient is exactly representable
  by accident.
* **every leg proves the sub-step MOVED STATE, per cycle and on BOTH sides.** A no-op
  agreeing with a no-op is trivially identical. The snapshot is taken IMMEDIATELY
  BEFORE the sub-step and compared IMMEDIATELY AFTER it, once per cycle, for the
  array path AND for the kernel; the reported figure is the MINIMUM over cycles, so
  one moving cycle cannot cover for a frozen one. Measuring across the whole cycle
  loop instead would count ``perturb``'s own writes as sub-step movement — the
  perturbation footprint strictly contains most sub-steps' — and a completely dead
  update would still report thousands of moved words.
* **constitutive legs keep a +-0 lattice LIVE AT EVERY SUB-STEP.** Zero-init is a
  fixed point of the constitutive sub-step, and the signed zero is exactly what
  separates an accumulation from a store (``no_pml_constitutive.STORED_E_ARM``).
  ``perturb`` therefore holds the lattice cells fixed rather than adding to them
  (``-0.0 + x`` is ``x``, so a perturbation that touched them would erase the needle
  before the first sub-step ran), and the census is taken per cycle with the MINIMUM
  reported: a leg whose lattice ever empties is VACUOUS.
* **the sub-step's SOURCES move too, not just its targets.** ``step_B`` differences
  the stored E and ``step_D`` the stored H; a perturbation confined to B/D/P would
  leave the curl legs reading frozen sources across every cycle, which is the exact
  stale-binding hazard ``perturb`` exists to catch. Every stored volume in the
  inventory is moved, identically on both sides.
* **predicted nulls are recorded with their reason**, not left absent.
* every case carries a PREDICTION made before the run, so a surprise is visible as a
  surprise rather than absorbed into the narrative.

Usage (on a CUDA host, from ``the repository root``)::

    PYTHONPATH=. python -u parity/meep_gpu/probe_residual_group_bodies.py \\
        --out <dir> [--only A1,B1]
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
_API = _HERE.parents[1]
if str(_API) not in sys.path:
    sys.path.insert(0, str(_API))

from meep_gpu import stepping  # noqa: E402
from meep_gpu.dispersion import PolarizationState, Susceptibility  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.triton_kernels.coverage import Coverage  # noqa: E402

try:
    import cupy as cp
except Exception:  # noqa: BLE001 - reported as a skip, never a crash
    cp = None

SEED = 20260814
E_NAMES = ("Ex", "Ey", "Ez")
B_NAMES = ("Bx", "By", "Bz")
D_NAMES = ("Dx", "Dy", "Dz")
H_NAMES = ("Hx", "Hy", "Hz")
#: The sub-steps whose arithmetic has zero-init as a FIXED POINT, and which
#: therefore need a live ``+-0`` lattice to be worth measuring at all. A leg on one
#: of these whose census reaches zero is VACUOUS, not IDENTICAL.
CONSTITUTIVE_SUB_STEPS = ("update_E", "update_H")
#: Eight, not four. The vacuity gate is now per cycle and takes the MINIMUM, so
#: more cycles is strictly more evidence: every extra cycle is another chance for a
#: stale binding to stop moving or for a divergence to open.
CYCLES = 8


def log(path: Optional[Path], message: str) -> None:
    print(message, flush=True)
    if path is not None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")
            handle.flush()


# ---------------------------------------------------------------------------
# The comparison primitives — uint32, whole inventory, first differing word
# ---------------------------------------------------------------------------

def to_host(array: Any) -> np.ndarray:
    if cp is not None and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.asarray(array)


def as_words(array: Any) -> np.ndarray:
    """The raw storage words. complex64 is compared as its two float32 halves."""
    host = np.ascontiguousarray(to_host(array))
    if host.dtype == np.complex64:
        return host.view(np.float32).view(np.uint32).ravel()
    if host.dtype == np.float32:
        return host.view(np.uint32).ravel()
    if host.dtype == np.float64:
        return host.view(np.uint64).ravel()
    return host.ravel()


def inventory(fields: Any) -> Dict[str, Any]:
    """Every array a sub-step could have written, by name. Absent names are skipped."""
    names: List[str] = list(B_NAMES + D_NAMES + E_NAMES + H_NAMES)
    names += ["fu_" + n for n in B_NAMES + D_NAMES]
    names += ["f_w_" + n for n in E_NAMES + H_NAMES]
    names += ["f_cond_" + n for n in B_NAMES + D_NAMES]
    out: Dict[str, Any] = {}
    for name in names:
        value = getattr(fields, name, None)
        if value is not None:
            out[name] = value
    for index, state in enumerate(getattr(fields, "polarizations", ()) or ()):
        for slot in ("P", "P_prev"):
            for component, array in (getattr(state, slot, {}) or {}).items():
                out[f"pol{index}.{slot}[{component}]"] = array
    return out


def compare(reference: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    """uint32 over the whole inventory. Returns the count and the FIRST difference."""
    total = 0
    differing = 0
    first: Optional[Dict[str, Any]] = None
    for name in sorted(reference):
        if name not in candidate:
            continue
        left = as_words(reference[name])
        right = as_words(candidate[name])
        if left.shape != right.shape:
            differing += max(left.size, right.size)
            if first is None:
                first = {"array": name, "note": f"shape {left.shape} vs {right.shape}"}
            continue
        total += left.size
        mask = left != right
        count = int(mask.sum())
        differing += count
        if count and first is None:
            index = int(np.argmax(mask))
            first = {"array": name, "word": index,
                     "reference_word": f"0x{int(left[index]):08x}",
                     "candidate_word": f"0x{int(right[index]):08x}"}
    return {"words": total, "differing": differing,
            "identical": differing == 0, "first_difference": first}


def moved_words(before: Dict[str, np.ndarray], after: Dict[str, Any]) -> int:
    """How many words moved between the two states. Zero at a sub-step means VACUOUS.

    The caller decides what "between" spans, and it MATTERS: taken around the whole
    cycle loop this counts ``perturb``'s writes as well, and the perturbation
    footprint contains most sub-steps' target footprints, so a completely dead
    update still reports thousands. :func:`run_case` therefore brackets ONE
    sub-step at a time.
    """
    moved = 0
    for name, snapshot in before.items():
        if name not in after:
            continue
        now = as_words(after[name])
        if snapshot.shape != now.shape:
            moved += max(snapshot.size, now.size)
            continue
        moved += int((snapshot != now).sum())
    return moved


def snapshot(fields: Any) -> Dict[str, np.ndarray]:
    return {name: as_words(array).copy() for name, array in inventory(fields).items()}


# ---------------------------------------------------------------------------
# Putting a certified body on a configuration its own predicate refuses
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def admitting(*targets: Tuple[Any, str]):
    """Temporarily replace named coverage functions with one that admits.

    The BODY under test is untouched — only the gate in front of its builder is.
    Reverted in a ``finally`` so an exception cannot leave the package patched.
    """
    saved: List[Tuple[Any, str, Any]] = []
    try:
        for module, name in targets:
            saved.append((module, name, getattr(module, name)))
            setattr(module, name, lambda *a, **k: Coverage(True, ()))
        yield
    finally:
        for module, name, original in reversed(saved):
            setattr(module, name, original)


# ---------------------------------------------------------------------------
# Fixtures — real Grid / Fields / PML objects, built the way the engine builds them
# ---------------------------------------------------------------------------

def lattice_masks(shape: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """The ``-0.0`` and ``+0.0`` cells of the signed-zero lattice, as boolean masks.

    One definition, used by BOTH the seeding and the perturbation, so the two
    cannot disagree about which cells the needle occupies. They did disagree
    before: ``perturb`` added a delta over the whole D volume and ``-0.0 + x`` is
    ``x``, so the lattice was gone before the first sub-step ran and every
    constitutive leg was seeded-zero-vacuous by this file's own rule while
    reporting a signed-zero census taken before the loop.
    """
    lattice = np.indices(tuple(shape)).sum(axis=0) % 4
    return lattice == 0, lattice == 1


def _with_signed_zeros(values: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    negative, positive = lattice_masks(shape)
    values[negative] = -0.0
    values[positive] = 0.0
    return values


def lattice_volume(xp, shape: Sequence[int], dtype: Any) -> Any:
    """A device volume carrying the seeded ``-0``/``+0`` lattice, zero elsewhere.

    Built on the HOST and transferred whole, which is the only spelling measured
    to preserve the sign of zero on the device — see :func:`perturb`'s PLATFORM
    FACT note.
    """
    negative, positive = lattice_masks(shape)
    host = np.zeros(tuple(shape), dtype=np.float32)
    host[negative] = -0.0
    host[positive] = 0.0
    if np.dtype(dtype) == np.complex64:
        return xp.asarray(_complex_from_halves(host, host.copy()))
    return xp.asarray(host)


def _complex_from_halves(real: np.ndarray, imaginary: np.ndarray) -> np.ndarray:
    """``real + 1j*imaginary`` WITHOUT losing a signed zero in the imaginary half.

    ``(0+1j) * (-0.0+0j)`` is ``(-0.0, +0.0)`` in IEEE arithmetic, so the spelling
    ``values + 1j * imaginary`` silently promotes every ``-0.0`` of the imaginary
    lattice to ``+0.0`` and seeds half the needle the caller reads as whole.
    Assigning the two halves into an empty complex64 array preserves both.
    """
    out = np.empty(real.shape, dtype=np.complex64)
    out.real = real
    out.imag = imaginary
    return out


def seed_state(fields: Any, xp, rng, amplitude: float = 0.4,
               signed_zero_lattice: bool = True) -> None:
    """Fill every stored volume with values a sub-step can move.

    ``signed_zero_lattice`` writes -0.0 and +0.0 on an alternating lattice into the
    D volumes. Zero-init is a FIXED POINT of the constitutive sub-step, so a leg
    seeded only with zeros compares a no-op against a no-op; and the signed zero is
    precisely what separates an accumulation (``(f + kps*src) - kms*prev``) from a
    store, because ``0.0 + (-0.0)`` is ``+0.0``.

    Seeding is only half of it — see :func:`perturb`, which must LEAVE the lattice
    alone for it to still be there when a sub-step runs.
    """
    shape = tuple(fields.grid.shape)
    complex_storage = bool(getattr(fields, "force_complex_fields", False))
    names = [n for n in (B_NAMES + D_NAMES + E_NAMES + H_NAMES)
             if getattr(fields, n, None) is not None]
    names += [n for n in ["fu_" + m for m in B_NAMES + D_NAMES]
              if getattr(fields, n, None) is not None]
    names += [n for n in ["f_w_" + m for m in E_NAMES + H_NAMES]
              if getattr(fields, n, None) is not None]
    names += [n for n in ["f_cond_" + m for m in B_NAMES + D_NAMES]
              if getattr(fields, n, None) is not None]
    for name in names:
        seeded = signed_zero_lattice and name in D_NAMES
        values = rng.uniform(-amplitude, amplitude, size=shape).astype(np.float32)
        if seeded:
            _with_signed_zeros(values, shape)
        if complex_storage:
            imaginary = rng.uniform(-amplitude, amplitude, size=shape).astype(np.float32)
            if seeded:
                _with_signed_zeros(imaginary, shape)
            getattr(fields, name)[...] = xp.asarray(
                _complex_from_halves(values, imaginary))
        else:
            getattr(fields, name)[...] = xp.asarray(values)
    for state in getattr(fields, "polarizations", ()) or ():
        for slot in ("P", "P_prev"):
            for component, array in (getattr(state, slot, {}) or {}).items():
                fill = rng.uniform(-amplitude, amplitude, size=shape).astype(np.float32)
                if array.dtype == np.complex64:
                    other = rng.uniform(-amplitude, amplitude,
                                        size=shape).astype(np.float32)
                    array[...] = xp.asarray((fill + 1j * other).astype(np.complex64))
                else:
                    array[...] = xp.asarray(fill)


def signed_zero_census(fields: Any) -> int:
    """How many -0.0 words the seeded state actually carries. Zero means VACUOUS."""
    count = 0
    for array in inventory(fields).values():
        words = as_words(array)
        if words.dtype == np.uint32:
            count += int((words == 0x80000000).sum())
    return count


def build(xp, *, cell=(0.8, 0.8, 0.8), dimensions=3, boundaries="metallic",
          courant=0.35, symmetry=(), k_point=(0.0, 0.0, 0.0),
          complex_storage=False, pml_thickness=2, eps=2.25,
          nonlinear=False, poles=0, conductivity=None,
          magnetic_conductivity=None, offdiag=False,
          amplitude=0.4, seed_offset=0) -> Tuple[Any, Any, Dict[str, Any]]:
    """One configuration, on real engine objects. Returns (fields, pml, notes)."""
    grid = Grid(resolution=10.0, cell_size=cell, dimensions=dimensions,
                boundaries=boundaries, courant=courant, symmetry=symmetry,
                k_point=k_point, xp=xp)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    shape = tuple(grid.shape)
    rng = np.random.default_rng(SEED + seed_offset)
    notes: Dict[str, Any] = {"shape": [int(n) for n in shape]}

    if offdiag:
        # A full inverse-permittivity ROW, the way subpixel smoothing of a curved
        # interface produces one. Off-diagonal entries are raw tensor entries and are
        # free to be negative — a uniform positive block would be a weaker needle.
        diagonal = {c: xp.asarray(np.full(shape, eps, dtype=np.float32)) for c in E_NAMES}
        inverse = {c: xp.asarray(np.full(shape, 1.0 / eps, dtype=np.float32))
                   for c in E_NAMES}
        rows = {
            "Ex": {"Ey": xp.asarray(rng.uniform(-0.08, 0.08, shape).astype(np.float32))},
            "Ey": {"Ez": xp.asarray(rng.uniform(-0.08, 0.08, shape).astype(np.float32))},
            "Ez": {"Ex": xp.asarray(rng.uniform(-0.08, 0.08, shape).astype(np.float32))},
        }
        fields.set_epsilon_volumes(diagonal, inverse, rows)
        notes["offdiagonal_rows"] = {k: sorted(v) for k, v in rows.items()}
    else:
        fields.set_background_eps(eps)

    if nonlinear:
        fields.set_nonlinear_volumes({c: 0.02 for c in E_NAMES},
                                     {c: 0.05 for c in E_NAMES})
        notes["chi2"] = 0.02
        notes["chi3"] = 0.05

    states: List[Any] = []
    for index in range(poles):
        term = Susceptibility(frequency=1.0 + 0.3 * index, gamma=0.1, kind="lorentzian")
        sigmas = {c: 0.3 + 0.05 * index for c in E_NAMES}
        state = PolarizationState(term, sigmas, grid,
                                  np.complex64 if complex_storage else np.float32)
        fields.polarizations.append(state)
        states.append(state)
    notes["n_poles"] = len(states)

    if magnetic_conductivity is not None:
        volume = np.full(shape, float(magnetic_conductivity), dtype=np.float32)
        volume *= np.linspace(0.6, 1.4, shape[0], dtype=np.float32)[:, None, None]
        fields.set_b_conductivity(xp.asarray(volume))
        notes["b_conductivity"] = float(magnetic_conductivity)
    if conductivity is not None:
        # A VOLUME, not a scalar: ``set_d_conductivity`` takes grid.shape, and a
        # spatially varying sigma is what an ``mp.Absorber`` actually installs.
        volume = np.full(shape, float(conductivity), dtype=np.float32)
        volume *= np.linspace(0.5, 1.5, shape[0], dtype=np.float32)[:, None, None]
        fields.set_d_conductivity(xp.asarray(volume))
        notes["d_conductivity"] = float(conductivity)

    active = pml_thickness > 0
    if active:
        fields.enable_pml_storage()
    else:
        fields.enable_field_storage()
    # A MIRRORED axis takes the HIGH FACE ONLY: cell 0 lies on the mirror plane,
    # which is a boundary condition and not an absorber (pml.py:408 refuses the
    # low face by name). That is the engine's own rule, not a harness convenience.
    mirrored = tuple(bool(getattr(grid, "is_mirrored", lambda a: False)(axis))
                     for axis in range(3))
    request: Dict[str, Any] = {}
    for axis, letter in enumerate("xyz"):
        if shape[axis] < 6:
            continue
        request[letter] = ({"high": pml_thickness} if mirrored[axis]
                           else pml_thickness)
    notes["pml_request"] = dict(request)
    notes["mirrored_axes"] = list(mirrored)
    pml = PML(grid=grid, thickness=(request or 0) if active else 0)
    notes["pml_is_active"] = bool(getattr(pml, "is_active", False))
    notes["stores_E"] = bool(getattr(fields, "stores_E", False))
    notes["has_nonlinearity"] = bool(getattr(fields, "has_nonlinearity", False))
    notes["has_offdiagonal_epsilon"] = bool(getattr(fields, "has_offdiagonal_epsilon", False))
    notes["condfac_Dz"] = fields.condfac_for("Dz") is not None
    notes["condfac_Bz"] = fields.condfac_for("Bz") is not None
    notes["has_magnetic_conductivity"] = bool(
        getattr(fields, "has_magnetic_conductivity", False))
    notes["polarization_states"] = states
    seed_state(fields, xp, rng, amplitude=amplitude)
    return fields, pml, notes


# ---------------------------------------------------------------------------
# One case = one (configuration, sub-step, certified body) triple
# ---------------------------------------------------------------------------

def perturbed_names(fields: Any) -> List[str]:
    """Every stored volume :func:`perturb` moves, in a fixed order.

    EVERY stored volume, not just the targets. ``step_B`` differences the stored E
    and ``step_D`` the stored H (stepping.py:907-923 keeps H stored under PML), so a
    perturbation confined to B, D and P leaves the two curl legs reading byte-frozen
    sources on every cycle — the same stale-source blindness that let group (G) pass
    falsely, one level up. The auxiliaries (``fu_*``, ``f_w_*``, ``f_cond_*``) move
    for the same reason: they are the recurrence's own previous state.
    """
    names = list(B_NAMES + D_NAMES + E_NAMES + H_NAMES)
    names += ["fu_" + n for n in B_NAMES + D_NAMES]
    names += ["f_w_" + n for n in E_NAMES + H_NAMES]
    names += ["f_cond_" + n for n in B_NAMES + D_NAMES]
    return [n for n in names if getattr(fields, n, None) is not None]


def perturb(fields: Any, xp, cycle: int, amplitude: float = 0.15) -> None:
    """Move the sub-step's INPUTS between cycles, identically on both sides.

    Not decoration. ``plan_folded_offdiagonal_constitutive`` (and any builder that
    resolves ``Fields.displacement_minus_polarization_volumes``) caches a pointer to
    the PERSISTENT per-component scratch at BUILD time (fields.py:1130-1137), while
    the array path REFILLS that scratch on every ``update_E``. Repeating one sub-step
    on frozen inputs therefore compares a live computation against a snapshot that
    happens to still be right — a false pass of exactly the shape this campaign keeps
    catching. Moving every stored volume and every P between cycles is what makes a
    stale binding visible as a divergence instead of hiding as agreement.

    THE ONE EXCEPTION IS THE +-0 LATTICE, and it is not a detail. ``-0.0 + x`` is
    ``x``, so adding a delta over a whole D volume ERASES the signed-zero needle
    before the first sub-step runs. The lattice cells are RESTORED after the delta
    and everything else moves; :func:`signed_zero_census` is then taken per cycle
    with the minimum reported, so a leg that loses the needle is VACUOUS rather than
    silently un-needled.

    PLATFORM FACT, measured on an RTX A6000 (cupy 13.5.1), not assumed: **CuPy's
    boolean-mask assignment FROM A SCALAR does not preserve the sign of zero.**
    ``a[mask] = -0.0``, ``a[mask] = np.float32(-0.0)`` and ``a[mask] =
    cp.float32(-0.0)`` all leave ``+0.0`` on the device, while the identical
    spelling on NumPy leaves ``-0.0`` — so the needle is silently gone on exactly
    the host that matters and intact on the one used to develop the harness. Three
    spellings DO preserve it: mask-assignment from an ARRAY, ``cp.where`` and
    ``cp.copyto(..., where=)``. ``cp.where`` against a host-built lattice volume is
    what this uses, because a whole-array transfer is bit-preserving (measured) and
    the same line is correct on NumPy.

    Zeroing the delta at the lattice instead of restoring afterwards does NOT work
    and was measured failing: ``-0.0 + 0.0`` is ``+0.0``.
    """
    shape = tuple(fields.grid.shape)
    rng = np.random.default_rng(SEED + 9000 + cycle)
    complex_storage = bool(getattr(fields, "force_complex_fields", False))
    negative, positive = lattice_masks(shape)
    frozen = xp.asarray(negative | positive)
    lattice_cache: Dict[Any, Any] = {}

    def delta():
        real = rng.uniform(-amplitude, amplitude, size=shape).astype(np.float32)
        if not complex_storage:
            return xp.asarray(real)
        imaginary = rng.uniform(-amplitude, amplitude, size=shape).astype(np.float32)
        return xp.asarray(_complex_from_halves(real, imaginary))

    for name in perturbed_names(fields):
        array = getattr(fields, name)
        array[...] += delta()
        if name in D_NAMES:
            key = str(array.dtype)
            if key not in lattice_cache:
                lattice_cache[key] = lattice_volume(xp, shape, array.dtype)
            array[...] = xp.where(frozen, lattice_cache[key], array)
    for state in getattr(fields, "polarizations", ()) or ():
        for slot in ("P", "P_prev"):
            for _component, array in (getattr(state, slot, {}) or {}).items():
                array[...] += delta()


def run_case(name: str, group: str, sub_step: str, prediction: str,
             clause: str, make, array_path, kernel_plan,
             cycles: int = CYCLES) -> Dict[str, Any]:
    """Build the configuration TWICE, step one each way, compare as uint32."""
    xp = cp
    row: Dict[str, Any] = {"case": name, "group": group, "sub_step": sub_step,
                           "prediction": prediction, "blocking_clause": clause,
                           "cycles": cycles}
    started = time.time()
    reference_fields, reference_pml, notes = make(0)
    candidate_fields, candidate_pml, _ = make(0)
    row["config"] = {k: v for k, v in notes.items() if k != "polarization_states"}

    # The two builds must START identical, or nothing downstream means anything.
    start = compare(inventory(reference_fields), inventory(candidate_fields))
    row["identical_at_start"] = start["identical"]
    # AT SEED, and labelled as such. What licenses a constitutive leg is the census
    # taken immediately before each sub-step (`signed_zero_words_min` below); this
    # one only says the seeding did what it meant to.
    row["signed_zero_words_at_seed"] = signed_zero_census(reference_fields)
    if not start["identical"]:
        row["error"] = "the two builds did not start byte-identical"
        row["start_difference"] = start["first_difference"]
        row["seconds"] = round(time.time() - started, 2)
        return row

    try:
        plan = kernel_plan(candidate_fields, candidate_pml)
    except BaseException as exc:  # noqa: BLE001
        row["error"] = f"plan build raised {type(exc).__name__}: {exc}"[:400]
        row["traceback"] = traceback.format_exc()[-1200:]
        row["seconds"] = round(time.time() - started, 2)
        return row
    if plan is None:
        row["error"] = "the shipped builder returned None even with its predicate admitting"
        row["seconds"] = round(time.time() - started, 2)
        return row
    row["plan"] = repr(plan)[:200]

    verdicts: List[Dict[str, Any]] = []
    substep_moved: List[int] = []
    kernel_moved: List[int] = []
    live_signed_zeros: List[int] = []
    for cycle in range(cycles):
        # The same perturbation on both sides, BEFORE the sub-step, so every cycle
        # presents fresh inputs and a plan holding a stale source snapshot diverges.
        perturb(reference_fields, xp, cycle)
        perturb(candidate_fields, xp, cycle)
        if cycle == 0:
            # Only meaningful on the first cycle: the delta is identical on both
            # sides, so after the sub-step has been allowed to disagree the two
            # legitimately differ and this check would report the sub-step's own
            # divergence as a harness fault.
            drift = compare(inventory(reference_fields), inventory(candidate_fields))
            row["perturbation_preserves_identity"] = drift["identical"]
            if not drift["identical"]:
                row["error"] = "the perturbation desynchronised the two builds"
                row["perturbation_difference"] = drift["first_difference"]
                row["seconds"] = round(time.time() - started, 2)
                return row

        # THE VACUITY GATE, bracketed around the SUB-STEP and nothing else. Taken
        # around the whole loop it would count `perturb`'s own writes, whose
        # footprint contains most sub-steps' targets, and a dead update would still
        # report thousands of moved words. Both sides are measured: an array path
        # that moves while the kernel sits still is a divergence only if the
        # comparison happens to look at the frozen array, and this makes it a
        # verdict in its own right.
        live_signed_zeros.append(signed_zero_census(reference_fields))
        substep_before = snapshot(reference_fields)
        kernel_before = snapshot(candidate_fields)

        array_path(reference_fields, reference_pml)
        try:
            plan.run()
        except BaseException as exc:  # noqa: BLE001
            row["error"] = f"launch raised {type(exc).__name__}: {exc}"[:400]
            row["traceback"] = traceback.format_exc()[-1200:]
            row["seconds"] = round(time.time() - started, 2)
            return row
        if cp is not None:
            cp.cuda.runtime.deviceSynchronize()

        substep_moved.append(moved_words(substep_before,
                                         inventory(reference_fields)))
        kernel_moved.append(moved_words(kernel_before,
                                        inventory(candidate_fields)))
        verdict = compare(inventory(reference_fields), inventory(candidate_fields))
        verdict["cycle"] = cycle + 1
        verdict["substep_moved"] = substep_moved[-1]
        verdict["kernel_moved"] = kernel_moved[-1]
        verdict["signed_zero_words"] = live_signed_zeros[-1]
        verdicts.append(verdict)

    # The MINIMUM, not the total and not the last: one moving cycle must not cover
    # for a frozen one.
    row["substep_moved_min"] = min(substep_moved)
    row["kernel_moved_min"] = min(kernel_moved)
    row["signed_zero_words_min"] = min(live_signed_zeros)
    row["substep_moved_per_cycle"] = substep_moved
    row["kernel_moved_per_cycle"] = kernel_moved
    # THE SIGNED-ZERO FLOOR, enforced rather than reported. A constitutive sub-step
    # has zero-init as a FIXED POINT, so on a leg whose needle has drained the
    # arithmetic under test is `0 = 0` on the lattice cells and the identity claim
    # is worth nothing there. Making it a VACUOUS verdict is what turned the
    # device's silent scalar-assignment canonicalization (see `perturb`) into a
    # visible failure instead of a passing table.
    row["needs_signed_zeros"] = sub_step in CONSTITUTIVE_SUB_STEPS
    row["signed_zero_floor_met"] = (row["signed_zero_words_min"] > 0
                                    or not row["needs_signed_zeros"])
    row["vacuous"] = (row["substep_moved_min"] == 0
                      or row["kernel_moved_min"] == 0
                      or not row["signed_zero_floor_met"])
    row["per_cycle"] = verdicts
    row["identical"] = all(v["identical"] for v in verdicts)
    # The MAXIMUM differing count over cycles, not the last one's: once the two
    # sides disagree they keep evolving apart, and reporting only the final cycle
    # makes the number depend on how many cycles were run.
    row["differing_max"] = max(v["differing"] for v in verdicts)
    row["differing_final"] = verdicts[-1]["differing"]
    row["words_compared"] = verdicts[-1]["words"]
    row["first_difference"] = next(
        (v["first_difference"] for v in verdicts if v["first_difference"]), None)
    row["verdict"] = ("VACUOUS" if row["vacuous"]
                      else "IDENTICAL" if row["identical"] else "DIVERGENT")
    # `prediction == "ERROR"` needs no branch of its own: a case predicted to refuse
    # reaching ANY of the three verdicts above is already a mismatch, because none
    # of VACUOUS / IDENTICAL / DIVERGENT equals "ERROR". The guard that used to sit
    # here set the same False the next line sets unconditionally, so it read as
    # load-bearing while doing nothing.
    row["matches_prediction"] = row["verdict"] == prediction
    row["seconds"] = round(time.time() - started, 2)
    return row


# ---------------------------------------------------------------------------
# The cases
# ---------------------------------------------------------------------------

def build_cases(xp) -> List[Dict[str, Any]]:
    from meep_gpu.triton_kernels import coverage as cov
    from meep_gpu.triton_kernels import dispersive_update_e as disp
    from meep_gpu.triton_kernels import folded_complex as fcx
    from meep_gpu.triton_kernels import folded_offdiag_update_e as foff
    from meep_gpu.triton_kernels import launch as launch_module
    from meep_gpu.triton_kernels import no_pml as nopml

    # The complex families need the certified EXPANSION probe — a MEASURED device
    # fact, not a guess. Loaded off disk and passed in explicitly, because
    # ``load_expansion_probe`` looks under the repo's results tree and a staged
    # source tree does not carry it; without it the builder returns None and the
    # case reports a harness fault as a coverage verdict.
    probe_record = _load_expansion_probe()

    cases: List[Dict[str, Any]] = []

    # ---- (A) chi2/chi3 on the curls and update_H ---------------------------
    def chi_config(offset):
        return build(xp, nonlinear=True, seed_offset=offset)

    for sub_step, fn in (("step_B", stepping.step_B), ("step_D", stepping.step_D)):
        cases.append({
            "name": f"A_chi_{sub_step}", "group": "A", "sub_step": sub_step,
            "prediction": "IDENTICAL",
            "clause": "chi2/chi3 is installed (the Pade constitutive factor is not carried)",
            "make": chi_config, "array_path": fn,
            "kernel_plan": (lambda f, p, s=sub_step: _admit_plan(
                [(cov, "pml_curl_coverage"), (launch_module, "pml_curl_coverage")],
                lambda: launch_module.plan_pml_curl(f, p, s))),
        })
    cases.append({
        "name": "A_chi_update_H", "group": "A", "sub_step": "update_H",
        "prediction": "IDENTICAL",
        "clause": "chi2/chi3 is installed (the Pade constitutive factor is not carried)",
        "make": chi_config, "array_path": stepping.update_H,
        "kernel_plan": (lambda f, p: _admit_plan(
            [(cov, "constitutive_coverage"), (launch_module, "constitutive_coverage")],
            lambda: launch_module.plan_constitutive(f, p, "H"))),
    })
    # The NEGATIVE CONTROL. chi2/chi3 enters update_E and only update_E
    # (stepping.py:975-979, :999-1000). If this comes back IDENTICAL the whole
    # measurement is insensitive and the three legs above mean nothing.
    cases.append({
        "name": "A_chi_update_E_CONTROL", "group": "A", "sub_step": "update_E",
        "prediction": "DIVERGENT",
        "clause": "control: the Pade factor DOES enter update_E",
        "make": chi_config, "array_path": stepping.update_E,
        "kernel_plan": (lambda f, p: _admit_plan(
            [(cov, "constitutive_coverage"), (launch_module, "constitutive_coverage")],
            lambda: launch_module.plan_constitutive(f, p, "E"))),
    })

    # ---- (B) ade_update_p under no PML -------------------------------------
    def no_pml_dispersive(offset):
        return build(xp, poles=1, pml_thickness=0, seed_offset=offset)

    cases.append({
        "name": "B_ade_update_p_no_pml", "group": "B", "sub_step": "update_P",
        "prediction": "IDENTICAL",
        "clause": "f_w_<component> is not allocated (the drive field under PML)",
        "make": no_pml_dispersive, "array_path": stepping.update_P,
        "kernel_plan": (lambda f, p: _ade_plan(launch_module, f)),
    })
    # The CONTROL that makes the drive-field distinction measurable rather than
    # asserted: under an ACTIVE PML the stored E and f_w differ, and driving P from
    # the stored E is "the single most likely silent wrong answer in dispersion"
    # (fields.py:1149-1156).
    def pml_dispersive(offset):
        return build(xp, poles=1, pml_thickness=2, seed_offset=offset)

    cases.append({
        "name": "B_ade_wrong_drive_CONTROL", "group": "B", "sub_step": "update_P",
        "prediction": "DIVERGENT",
        "clause": "control: under PML, stored E is NOT the drive field",
        "make": pml_dispersive, "array_path": stepping.update_P,
        "kernel_plan": (lambda f, p: _ade_plan(launch_module, f, wrong_drive=True)),
    })

    # (H) is (B)'s sub-step on a CONDUCTIVE no-PML row. Measured directly rather
    # than inferred from (B): "conductivity does not reach the ADE recurrence" is
    # exactly the kind of true-sounding claim this campaign has been wrong about.
    def conductive_no_pml_dispersive(offset):
        return build(xp, poles=1, conductivity=0.7, magnetic_conductivity=0.5,
                     pml_thickness=0, seed_offset=offset)

    cases.append({
        "name": "H_ade_update_p_conductive_no_pml", "group": "H", "sub_step": "update_P",
        "prediction": "IDENTICAL",
        "clause": "f_w_<component> is not allocated (the drive field under PML)",
        "make": conductive_no_pml_dispersive, "array_path": stepping.update_P,
        "kernel_plan": (lambda f, p: _ade_plan(launch_module, f)),
    })

    # (I)'s update_P is a DIFFERENT question and this case is what makes it one:
    # complex64 P/P_prev/scratch against a float32 kernel body. Recorded as a
    # PREDICTED ERROR — a type mismatch is not a divergence, and calling it one
    # would misreport a real kernel gap as a rounding difference.
    def complex_no_pml_dispersive(offset):
        return build(xp, poles=1, complex_storage=True, conductivity=0.7,
                     pml_thickness=0, seed_offset=offset)

    cases.append({
        "name": "I_ade_update_p_complex_storage", "group": "I", "sub_step": "update_P",
        "prediction": "ERROR",
        "clause": "P[<c>] dtype complex64 is not float32 (three clauses of it)",
        "make": complex_no_pml_dispersive, "array_path": stepping.update_P,
        "kernel_plan": (lambda f, p: _ade_plan(launch_module, f)),
    })

    # ---- (C) fold + dispersion at update_E ---------------------------------
    def folded_dispersive(offset):
        return build(xp, cell=(1.6, 1.6, 1.6), symmetry=("X",),
                     poles=1, pml_thickness=2, seed_offset=offset)

    cases.append({
        "name": "C_folded_dispersive_update_E", "group": "C", "sub_step": "update_E",
        "prediction": "IDENTICAL",
        "clause": ("a susceptibility is registered: folded dispersive update_E is not "
                   "yet composed; its source is (D - sum P), not D"),
        "make": folded_dispersive, "array_path": stepping.update_E,
        "kernel_plan": (lambda f, p: _admit_plan(
            [(disp, "dispersive_constitutive_coverage")],
            lambda: disp.plan_dispersive_constitutive(f, p))),
    })

    # ROWSHAPE. The fixture above is a 3-D box with one fold; group (C)'s four rows
    # are TestLoadDump 2-D at [250,126,1] with boundary kinds
    # ('metallic','mirror','periodic'). Same admission, a configuration SHAPED like
    # the rows it is claimed for — because "measured on the corpus configuration" is
    # a claim about the fixture, and the 3-D box was not one.
    def folded_dispersive_rowshape(offset):
        return build(xp, cell=(2.4, 2.4, 0.0), dimensions=2,
                     boundaries={"x": "metallic", "y": "metallic", "z": "periodic"},
                     symmetry=("Y",), poles=1, pml_thickness=2, seed_offset=offset)

    cases.append({
        "name": "C_folded_dispersive_update_E_ROWSHAPE_2D", "group": "C",
        "sub_step": "update_E", "prediction": "IDENTICAL",
        "clause": ("a susceptibility is registered: folded dispersive update_E is not "
                   "yet composed; its source is (D - sum P), not D"),
        "make": folded_dispersive_rowshape, "array_path": stepping.update_E,
        "kernel_plan": (lambda f, p: _admit_plan(
            [(disp, "dispersive_constitutive_coverage")],
            lambda: disp.plan_dispersive_constitutive(f, p))),
    })

    # ---- (D) complex + fold + off-diagonal ---------------------------------
    def complex_fold_offdiag(offset):
        return build(xp, cell=(1.6, 1.6, 1.6), symmetry=("X",),
                     complex_storage=True, offdiag=True, pml_thickness=2,
                     seed_offset=offset)

    for sub_step, fn in (("step_B", stepping.step_B), ("step_D", stepping.step_D)):
        cases.append({
            "name": f"D_complex_fold_offdiag_{sub_step}", "group": "D",
            "sub_step": sub_step, "prediction": "IDENTICAL",
            "clause": ("an off-diagonal chi1inv row is installed: the row product "
                       "reads neighbours through a transverse Yee average"),
            "make": complex_fold_offdiag, "array_path": fn,
            "kernel_plan": (lambda f, p, s=sub_step: _admit_plan(
                [(fcx, "folded_complex_pml_curl_coverage"),
                 (fcx, "folded_complex_composition_curl_coverage")],
                lambda: fcx.plan_folded_complex_pml_curl(f, p, s,
                                                         probe=probe_record))),
        })
    cases.append({
        "name": "D_complex_fold_offdiag_update_H", "group": "D",
        "sub_step": "update_H", "prediction": "IDENTICAL",
        "clause": ("an off-diagonal chi1inv row is installed: the row product reads "
                   "neighbours through a transverse Yee average"),
        "make": complex_fold_offdiag, "array_path": stepping.update_H,
        "kernel_plan": (lambda f, p: _admit_plan(
            [(fcx, "folded_complex_constitutive_coverage")],
            lambda: fcx.plan_folded_complex_constitutive(f, p, "H",
                                                         probe=probe_record))),
    })
    cases.append({
        "name": "D_complex_fold_offdiag_update_E", "group": "D",
        "sub_step": "update_E", "prediction": "DIVERGENT",
        "clause": ("an off-diagonal chi1inv row is installed: the row product reads "
                   "neighbours through a transverse Yee average"),
        "make": complex_fold_offdiag, "array_path": stepping.update_E,
        "kernel_plan": (lambda f, p: _admit_plan(
            [(fcx, "folded_complex_constitutive_coverage")],
            lambda: fcx.plan_folded_complex_constitutive(f, p, "E",
                                                         probe=probe_record))),
    })

    # ROWSHAPE for (D). The fixture above is a 3-D box with ONE fold and no phase;
    # the three rows it is claimed for are 2-D and carry more:
    # test_array_metadata and solve-cw.py fold TWO axes at [201,201,1]/[161,161,1],
    # and test_fields_at_kx carries k=(3.5,0,0) on a periodic axis beside its fold
    # at [20,122,1]. Both are measured rather than assumed to follow.
    def complex_fold_offdiag_twofold(offset):
        # ('mirror', 'mirror', 'periodic') — test_array_metadata's and solve-cw's.
        return build(xp, cell=(2.4, 2.4, 0.0), dimensions=2,
                     boundaries={"x": "metallic", "y": "metallic", "z": "periodic"},
                     symmetry=("X", "Y"), complex_storage=True, offdiag=True,
                     pml_thickness=2, seed_offset=offset)

    def complex_fold_offdiag_bloch(offset):
        # ('periodic', 'mirror', 'periodic') with k=(3.5,0,0) — test_fields_at_kx's.
        return build(xp, cell=(2.4, 2.4, 0.0), dimensions=2, symmetry=("Y",),
                     boundaries="periodic", k_point=(3.5, 0.0, 0.0),
                     complex_storage=True, offdiag=True, pml_thickness=2,
                     seed_offset=offset)

    for tag, maker in (("TWOFOLD", complex_fold_offdiag_twofold),
                       ("BLOCH", complex_fold_offdiag_bloch)):
        for sub_step, fn in (("step_B", stepping.step_B), ("step_D", stepping.step_D)):
            cases.append({
                "name": f"D_ROWSHAPE_{tag}_{sub_step}", "group": "D",
                "sub_step": sub_step, "prediction": "IDENTICAL",
                "clause": ("an off-diagonal chi1inv row is installed: the row product "
                           "reads neighbours through a transverse Yee average"),
                "make": maker, "array_path": fn,
                "kernel_plan": (lambda f, p, s=sub_step: _admit_plan(
                    [(fcx, "folded_complex_pml_curl_coverage"),
                     (fcx, "folded_complex_composition_curl_coverage")],
                    lambda: fcx.plan_folded_complex_pml_curl(f, p, s,
                                                             probe=probe_record))),
            })
        cases.append({
            "name": f"D_ROWSHAPE_{tag}_update_H", "group": "D",
            "sub_step": "update_H", "prediction": "IDENTICAL",
            "clause": ("an off-diagonal chi1inv row is installed: the row product "
                       "reads neighbours through a transverse Yee average"),
            "make": maker, "array_path": stepping.update_H,
            "kernel_plan": (lambda f, p: _admit_plan(
                [(fcx, "folded_complex_constitutive_coverage")],
                lambda: fcx.plan_folded_complex_constitutive(f, p, "H",
                                                             probe=probe_record))),
        })

    # ---- (E) conductivity + no PML at the curls ----------------------------
    def conductive_no_pml(offset):
        # BOTH sides, because that is what an ``mp.Absorber`` installs and what the
        # corpus row carries: ``remaining.txt`` names "a magnetic (B) conductivity is
        # installed" as a separate clause on step_B. A D-only fixture would leave
        # step_B on the plain path and measure the wrong thing.
        return build(xp, conductivity=0.7, magnetic_conductivity=0.5,
                     pml_thickness=0, seed_offset=offset)

    for sub_step, fn in (("step_B", stepping.step_B), ("step_D", stepping.step_D)):
        cases.append({
            "name": f"E_conductive_no_pml_{sub_step}", "group": "E",
            "sub_step": sub_step, "prediction": "DIVERGENT",
            "clause": ("a conductivity is installed on <field> "
                       "(mp.Absorber routes to _apply_conductive_update)"),
            "make": conductive_no_pml, "array_path": fn,
            "kernel_plan": (lambda f, p, s=sub_step: _admit_plan(
                [(nopml, "plain_curl_coverage")],
                lambda: nopml.plan_plain_curl(f, p, s))),
        })

    # ---- (F/H/I/J) the STORE arm of update_E -------------------------------
    # ``no_pml_constitutive.STORED_E_ARM`` records this as measured-and-not-built.
    # Re-measured here on the device, because four of the ten groups rest on it.
    def no_pml_stored_e(offset):
        return build(xp, poles=1, pml_thickness=0, seed_offset=offset)

    cases.append({
        "name": "F_stored_e_arm_update_E", "group": "F/H/I/J", "sub_step": "update_E",
        "prediction": "DIVERGENT",
        "clause": ("no active PML: update_E writes E[...] = constitutive "
                   "(stepping.py:1022), not the accumulation the certified body runs"),
        "make": no_pml_stored_e, "array_path": stepping.update_E,
        "kernel_plan": (lambda f, p: _admit_plan(
            [(disp, "dispersive_constitutive_coverage")],
            lambda: _dispersive_unit_coefficient_plan(disp, f, p))),
    })

    # ---- (G) fold + off-diagonal + dispersion at update_E ------------------
    def folded_offdiag_dispersive(offset):
        return build(xp, cell=(1.6, 1.6, 1.6), symmetry=("X",),
                     offdiag=True, poles=1, pml_thickness=2, seed_offset=offset)

    cases.append({
        "name": "G_folded_offdiag_dispersive_update_E", "group": "G",
        "sub_step": "update_E", "prediction": "DIVERGENT",
        "clause": ("a susceptibility is registered: the offdiag source becomes "
                   "D - sum P in per-component scratch buffers"),
        "make": folded_offdiag_dispersive, "array_path": stepping.update_E,
        "kernel_plan": (lambda f, p: _admit_plan(
            [(foff, "folded_offdiag_constitutive_coverage"),
             (foff, "folded_offdiag_composition_coverage")],
            lambda: foff.plan_folded_offdiagonal_constitutive(f, p))),
    })

    return cases


#: The certified complex-multiply EXPANSION probe, the same artifact the coverage
#: battery loads (``predicate_coverage_.../measure_predicate_coverage.py:PROBE``).
EXPANSION_PROBE = ("parity/meep_gpu/results/complex_signed_zero_2026-08-12/"
                   "results/run_2345/probe.json")


def _load_expansion_probe() -> Optional[Dict[str, Any]]:
    for candidate in (_API / EXPANSION_PROBE, _HERE / "expansion_probe.json"):
        if candidate.exists():
            return _normalize_legacy_expansion_probe(
                json.loads(candidate.read_text(encoding="utf-8")), candidate)
    return None


def _normalize_legacy_expansion_probe(record: Dict[str, Any],
                                      source: Any) -> Dict[str, Any]:
    """Bring a pre-2026-08-15 probe artifact up to the schema the licence reads.

    THE ARTIFACT THIS BATTERY LOADS WAS CUT ON 2026-08-12 (job 2345) and predates
    two fields the licence rule now requires, so as it stands it REFUSES:

    * ``subnormal_policy.resolved`` — the artifact names its policy
      (``ieee_keep_ftz_stripped``) but not what that resolved to. Derived here
      from the name via the gate's own constant, which is a lookup rather than a
      guess, and ASSERTED rather than assumed.
    * ``detail[...].licensable_arms_disagreement_words`` behind each
      AMBIGUOUS_BOTH — derived from the record's OWN numbers and only where they
      determine it: if the recorded ``mismatch_words`` are 0 for both FMA_V1 and
      NAIVE then each equals the platform on those vectors, hence they equal each
      other, hence the arms are 0 words apart. That is entailment, not inference.

    ``candidates.policy`` is the one thing the artifact does not record and this
    function will NOT invent — it is set equal to the resolved policy and the
    record is MARKED, because the candidate arms were built host-side and whether
    that host was keeping subnormals is simply not in the file. The mark is the
    point: this battery is a laptop predicate-coverage check that needs a
    licensable probe as an INPUT (the licence rule itself is tested in
    ``meep_gpu/test_triton_complex_fields.py``), and a normalized legacy artifact
    must never be mistaken for a freshly cut one.

    THE REAL FIX IS A RE-CUT ON A DEVICE. gate_triton_complex writes both fields
    now, and also refuses to emit a record whose operand classes cannot tell the
    arms apart under the policy in force — which this artifact was never checked
    for.
    """
    record = dict(record)
    stamp = dict(record.get("subnormal_policy") or {})
    if stamp and not stamp.get("resolved"):
        assert stamp.get("policy") == "ieee_keep_ftz_stripped", (
            f"{source}: unrecognised legacy policy stamp {stamp.get('policy')!r}; "
            f"this function may only normalize the keep-policy artifact it names")
        stamp["resolved"] = "keep"
        record["subnormal_policy"] = stamp
    if record.get("candidates") is None and stamp.get("resolved"):
        record["candidates"] = {
            "policy": stamp["resolved"],
            "SUPPLIED_BY_NORMALIZATION": (
                f"NOT MEASURED. {source} predates the candidate-policy stamp and "
                f"does not record which policy its candidate arms were built "
                f"under; this value is the record's own resolved policy, "
                f"supplied so a laptop predicate battery can run. It is not "
                f"evidence, and this artifact needs re-cutting on a device."),
        }
    detail = record.get("detail")
    if isinstance(detail, dict):
        record["detail"] = {name: _normalize_legacy_detail(entry)
                            for name, entry in detail.items()}
    vectors = dict(record.get("vectors") or {})
    for name, value in (record.get("patterns") or {}).items():
        if name not in vectors:
            entries = record.get("detail", {}).get(name)
            entries = entries if isinstance(entries, list) else [entries]
            sizes = [e.get("vectors") for e in entries
                     if isinstance(e, dict) and isinstance(e.get("vectors"), int)]
            if sizes:
                vectors[name] = max(sizes)
            elif value != "AMBIGUOUS_BOTH":
                continue          # only an ambiguity needs a vector count
            else:                 # the scalar leg: same operands as its siblings
                vectors[name] = max(vectors.values()) if vectors else 0
    record["vectors"] = vectors
    return record


def _normalize_legacy_detail(entry: Any) -> Any:
    """Back-fill ``licensable_arms_disagreement_words`` where the record's own
    ``mismatch_words`` ENTAIL it: both licensable arms at 0 mismatch against the
    platform means both equal it, so they equal each other. Anything else is left
    alone — a missing number stays missing and refuses."""
    if isinstance(entry, list):
        return [_normalize_legacy_detail(item) for item in entry]
    if not isinstance(entry, dict) or "licensable_arms_disagreement_words" in entry:
        return entry
    mismatch = entry.get("mismatch_words")
    if isinstance(mismatch, dict) and mismatch.get("FMA_V1") == 0 \
            and mismatch.get("NAIVE") == 0:
        entry = dict(entry)
        entry["licensable_arms_disagreement_words"] = 0
        entry["discriminates"] = False
        entry["_derived"] = ("both licensable arms matched the platform at 0 "
                            "mismatch words, so they match each other")
    return entry


def _admit_plan(targets, builder):
    with admitting(*targets):
        return builder()


def _ade_plan(launch_module, fields, wrong_drive: bool = False):
    """The certified ADE plan, built directly, launched through the shipped class.

    ``wrong_drive`` binds the STORED E instead of ``Fields.drive_field`` — the control
    that proves this comparison can see the drive-field distinction at all.
    """
    from meep_gpu.triton_kernels.kernels import DEFAULT_BLOCK

    states = tuple(getattr(fields, "polarizations", ()) or ())
    plans = [launch_module.AdeUpdatePPlan(state, tuple(state.driven()),
                                          fields.grid.shape, DEFAULT_BLOCK)
             for state in states]
    drive = ((lambda c: getattr(fields, c)) if wrong_drive else fields.drive_field)

    class _Composite:
        def run(self, guard=None):
            for plan in plans:
                plan.run(drive, guard)

        def __repr__(self):
            return f"AdeComposite({[repr(p) for p in plans]}, wrong_drive={wrong_drive})"

    return _Composite()


def _dispersive_unit_coefficient_plan(disp, fields, pml):
    """The certified dispersive body with kps=1 / kms=0 — the Arm S substitution.

    ``STORED_E_ARM['differs_from_certified_body']`` claims binding identity
    coefficients does NOT recover ``f[i] = src``: ``0.0 + (-0.0)`` is ``+0.0``. This
    builds exactly that substitution so the claim is a device measurement rather
    than a NumPy note.
    """
    import cupy as _cp

    from meep_gpu.triton_kernels.dispersive_update_e import (
        E_TERMS, DispersiveConstitutivePlan, LivePoleBinding, poles_per_component)
    from meep_gpu.triton_kernels.kernels import DEFAULT_BLOCK

    shape = tuple(fields.grid.shape)
    ones = _cp.ones(shape, dtype=_cp.float32)
    zeros = _cp.zeros(shape, dtype=_cp.float32)
    targets = tuple(term[0] for term in E_TERMS)
    # Without PML there is no f_w; the arm must not write one, so a scratch volume
    # stands in and its contents are NOT compared (it is absent from the reference).
    auxiliary = [_cp.zeros(shape, dtype=_cp.float32) for _ in targets]
    return DispersiveConstitutivePlan(
        shape, DEFAULT_BLOCK,
        [getattr(fields, name) for name in targets],
        auxiliary,
        [getattr(fields, term[1]) for term in E_TERMS],
        [fields.inverse_epsilon_for(name) for name in targets],
        [ones if stem == "kps" else zeros
         for _axis in "xyz" for stem in ("kps", "kms")],
        LivePoleBinding(fields, poles_per_component(fields)),
    )


# ---------------------------------------------------------------------------

def main(argv) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--only", default="")
    parser.add_argument("--cycles", type=int, default=CYCLES)
    args = parser.parse_args(argv)

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    progress = out / "bodies.progress.log"
    results_path = out / "bodies.json"

    results: Dict[str, Any] = {"cycles": args.cycles, "cases": []}
    if cp is None:
        results["skipped"] = "cupy is not importable on this host"
        results_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
        log(progress, "SKIPPED: cupy is not importable on this host")
        return 0
    try:
        import triton  # noqa: F401
        results["triton"] = getattr(triton, "__version__", "unknown")
    except Exception as exc:  # noqa: BLE001
        results["skipped"] = f"triton is not importable: {exc}"
        results_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
        log(progress, f"SKIPPED: triton is not importable: {exc}")
        return 0

    results["cupy"] = cp.__version__
    results["device"] = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    wanted = {n.strip() for n in args.only.split(",") if n.strip()}
    cases = build_cases(cp)
    if wanted:
        cases = [c for c in cases if c["name"] in wanted]
    log(progress, f"bodies: {len(cases)} cases on {results['device']} "
                  f"(cupy {results['cupy']}, triton {results['triton']})")

    for index, case in enumerate(cases, start=1):
        try:
            row = run_case(case["name"], case["group"], case["sub_step"],
                           case["prediction"], case["clause"], case["make"],
                           case["array_path"], case["kernel_plan"],
                           cycles=args.cycles)
        except BaseException as exc:  # noqa: BLE001
            row = {"case": case["name"], "group": case["group"],
                   "sub_step": case["sub_step"], "prediction": case["prediction"],
                   "error": f"{type(exc).__name__}: {exc}"[:400],
                   "traceback": traceback.format_exc()[-1500:],
                   "matches_prediction": case["prediction"] == "ERROR"}
        if row.get("error") and "matches_prediction" not in row:
            row["matches_prediction"] = case["prediction"] == "ERROR"
        results["cases"].append(row)
        results_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
        log(progress,
            f"case {index}/{len(cases)} {row['case']:<44} "
            f"verdict={row.get('verdict', 'ERROR')} "
            f"predicted={row.get('prediction')} "
            f"substep_moved>={row.get('substep_moved_min')} "
            f"kernel_moved>={row.get('kernel_moved_min')} "
            f"minus0>={row.get('signed_zero_words_min')} "
            f"maxdiff={row.get('differing_max')}/{row.get('words_compared')} "
            f"({row.get('seconds')} s)"
            + (f"  ERROR: {row['error']}" if row.get("error") else ""))

    surprises = [r for r in results["cases"]
                 if r.get("matches_prediction") is False]
    errors = [r for r in results["cases"] if r.get("error")]
    results["surprises"] = [r["case"] for r in surprises]
    results["errors"] = [r["case"] for r in errors]
    results_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    log(progress, f"--- {len(results['cases'])} cases, {len(surprises)} surprises, "
                  f"{len(errors)} errors ---")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
