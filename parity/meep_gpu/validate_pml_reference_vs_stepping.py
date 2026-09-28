"""Pin the bit-identity probe's PML reference against ``stepping.py`` itself, on NumPy.

The device gate compares a kernel against
``probe_fused_kernel_bit_identity.reference_pml_step`` — a transcription, not an
import. That is the point (two independent readings of the same contract), but
it leaves one hole: if the transcription is wrong the kernel can be
bit-identical to it and still wrong against the engine. THIS closes the hole,
and it closes it on a host with no GPU, before any device time is spent:

    stepping.step_B / step_D   (the array path, the correctness oracle)
        == reference_pml_step  (the probe's transcription)

bytewise, over the same shape x boundary x dtdx product the gate sweeps, with a
REAL ``meep_gpu.pml.PML`` layer and a REAL ``Fields``. The NumPy and CuPy legs of
the array path are already bit-identical to each other (``test_backends``), so an
agreement measured here holds on the device.

Run (no GPU needed)::

    python -u validate_pml_reference_vs_stepping.py \\
        --out results/pml_reference_vs_stepping.json

One flushed line per case, appended to the JSON as each lands (the progress-reporting rule).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_API = os.path.abspath(os.path.join(HERE, "..", ".."))
if REPO_API not in sys.path:
    sys.path.insert(0, REPO_API)


def load_probe():
    """Load the probe by path. Its cupy import is optional, so this works GPU-less."""
    path = os.path.join(HERE, "probe_fused_kernel_bit_identity.py")
    spec = importlib.util.spec_from_file_location("probe_bit_identity", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe = load_probe()


def load_fused_electric_gate():
    """Load the D/E gate's reference without importing CuPy or Triton."""
    path = os.path.join(HERE, "gate_triton_fused_electric.py")
    spec = importlib.util.spec_from_file_location("fused_electric_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fused_electric = load_fused_electric_gate()

from meep_gpu import stepping  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.pml import PML
from meep_gpu.triton_kernels.coverage import zero_metal_axes  # noqa: E402

# The sub-step reads the STORED E (B side) / H (D side) and writes B/fu_B (or
# D/fu_D). Nothing else in the step is in this slice.
SIDE_ARRAYS = {
    "step_B": (("Ex", "Ey", "Ez"), ("Bx", "By", "Bz"),
               ("fu_Bx", "fu_By", "fu_Bz")),
    "step_D": (("Hx", "Hy", "Hz"), ("Dx", "Dy", "Dz"),
               ("fu_Dx", "fu_Dy", "fu_Dz")),
}


def log(message: str) -> None:
    print(message, flush=True)


def build(shape: Tuple[int, int, int], boundaries: Sequence[str], dtdx: float):
    """A real Grid + Fields + PML whose ``dt/dx`` is exactly the requested dtdx.

    ``Grid.courant`` IS ``dt/dx`` (``dt = courant * dx``, MEEP's Courant number),
    so setting it to the sweep's dtdx is what puts the non-power-of-two 0.35 into
    the array path rather than only into the reference.
    """
    grid = Grid(
        resolution=1.0,
        cell_size=(float(shape[0]), float(shape[1]), float(shape[2])),
        courant=dtdx,
        boundaries=tuple(boundaries),
        dimensions=3,
        xp=np,
    )
    if tuple(grid.shape) != tuple(shape):
        raise RuntimeError(f"Grid built {tuple(grid.shape)} for {tuple(shape)}")
    measured = grid.dt / grid.dx
    if float(measured) != float(dtdx):
        raise RuntimeError(f"grid.dt/grid.dx is {measured!r}, not the requested {dtdx!r}")
    thickness = tuple((2, 2) if shape[axis] >= 6 else (0, 0) for axis in range(3))
    layer = PML(grid=grid, thickness=thickness)
    if not layer.is_active:
        raise RuntimeError(f"PML {thickness} on shape {shape} absorbs nowhere")
    fields = Fields(grid=grid)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    return grid, fields, layer


def seed_fields(fields: Fields, shape, rng) -> None:
    """Fill every array the slice touches, auxiliaries included, with seeded noise.

    A zero ``fu`` makes ``fu * kms`` exactly zero whatever ``kms`` is, which would
    hide a coefficient-indexing error on the first sub-step — the one this
    comparison runs.
    """
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz",
                 "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
                 "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz"):
        target = getattr(fields, name)
        if target is None:
            raise RuntimeError(f"{name} was not allocated by enable_pml_storage()")
        target[...] = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)


def snapshot(fields: Fields, names) -> Dict[str, Any]:
    return {name: getattr(fields, name).copy() for name in names}


def one_case(shape, boundaries, dtdx, sub_step) -> Dict[str, Any]:
    case: Dict[str, Any] = {
        "shape": list(shape), "boundaries": list(boundaries),
        "dtdx": repr(dtdx), "sub_step": sub_step,
    }
    walled_invariant = [a for a in range(3)
                        if shape[a] == 1 and boundaries[a] == "metallic"]
    if walled_invariant:
        case["skipped"] = (f"axis {walled_invariant[0]} has one cell and a metallic "
                           f"wall, which Grid refuses")
        return case
    try:
        grid, fields, layer = build(shape, boundaries, dtdx)
    except Exception as exc:  # noqa: BLE001 - a refusal is a result, not a crash
        case["skipped"] = f"{type(exc).__name__}: {exc}"[:400]
        return case

    resolved = stepping._boundary_kinds(grid, layer)
    case["resolved_boundaries"] = list(resolved)
    if tuple(resolved) != tuple(boundaries):
        # The absorber must not change the ghost rule. If it ever does, the whole
        # kernel slice is mis-specified and this is where it surfaces.
        case["skipped"] = (f"stepping resolved {resolved}, not the declared "
                           f"{tuple(boundaries)}")
        return case

    source_names, target_names, aux_names = SIDE_ARRAYS[sub_step]
    rng = np.random.default_rng(probe.SEED + 7)
    seed_fields(fields, shape, rng)
    before = snapshot(fields, source_names + target_names + aux_names)

    # Leg 1: the array path, exactly as the driver calls it.
    (stepping.step_B if sub_step == "step_B" else stepping.step_D)(fields, layer)
    array_path = snapshot(fields, target_names + aux_names)

    # Leg 2: the probe's transcription, from the identical inputs.
    state = {name: before[name].copy() for name in target_names + aux_names}
    coefficients = probe.layer_coefficients(layer, sub_step == "step_B")
    probe.reference_pml_step(
        np,
        {name: before[name] for name in source_names},
        {name: state[name] for name in target_names},
        {name: state[name] for name in aux_names},
        coefficients, np.float32(dtdx), sub_step, tuple(boundaries), "array_order")

    parts = {name: probe.bit_compare(array_path[name], state[name])
             for name in target_names + aux_names}
    case["vs_stepping"] = probe.combine(parts)

    # NEGATIVE CONTROL. An agreement is only evidence if a disagreement was
    # reachable: re-run the transcription with the ONE thing the gate exists to
    # pin changed — the stencil regrouped to C's left-to-right association — and
    # require that it does NOT match. At dtdx = 0.5 the scaling is exact in
    # binary and the two groupings can legitimately coincide, which is precisely
    # why 0.35 is in the sweep; the control is therefore only asserted there.
    control = {name: before[name].copy() for name in target_names + aux_names}
    probe.reference_pml_step(
        np,
        {name: before[name] for name in source_names},
        {name: control[name] for name in target_names},
        {name: control[name] for name in aux_names},
        coefficients, np.float32(dtdx), sub_step, tuple(boundaries), "kernel_order")
    regrouped = probe.combine({name: probe.bit_compare(array_path[name], control[name])
                               for name in target_names + aux_names})
    case["regrouped_control"] = {
        "bit_identical_to_stepping": regrouped["bit_identical"],
        "differing_floats": regrouped["differing_floats"],
        "max_ulp": regrouped.get("max_ulp", 0),
        "asserted": float(np.float32(dtdx)) != float(dtdx),
    }

    # The sub-step must actually have done something: a comparison of two
    # unchanged copies is identical for the wrong reason.
    moved = {name: not np.array_equal(before[name], array_path[name])
             for name in target_names + aux_names}
    case["arrays_changed"] = moved
    case["all_arrays_changed"] = all(moved.values())
    return case


def one_constitutive_case(shape, boundaries, courant, side) -> Dict[str, Any]:
    """``stepping.update_H`` / ``update_E`` against the probe's second transcription.

    Same hole, same closure, one sub-step later: the device gate compares the
    kernel against ``probe.reference_constitutive_step``, and if that
    transcription is wrong the kernel can be bit-identical to it and still wrong
    against the engine. This is the only leg that can catch that, and it runs on a
    laptop before any device time is spent.

    ``update_E`` needs an inverse-epsilon volume that is NOT 1.0, or ``D*inv_eps``
    is bit-equal to ``D`` and a kernel dropping the multiply would agree.
    """
    case: Dict[str, Any] = {
        "shape": list(shape), "boundaries": list(boundaries),
        "courant": repr(courant), "sub_step": "update_" + side,
    }
    walled_invariant = [a for a in range(3)
                        if shape[a] == 1 and boundaries[a] == "metallic"]
    if walled_invariant:
        case["skipped"] = (f"axis {walled_invariant[0]} has one cell and a metallic "
                           f"wall, which Grid refuses")
        return case
    try:
        grid, fields, layer = build(shape, boundaries, courant)
    except Exception as exc:  # noqa: BLE001
        case["skipped"] = f"{type(exc).__name__}: {exc}"[:400]
        return case

    rng = np.random.default_rng(probe.SEED + 13)
    seed_fields(fields, shape, rng)
    targets, auxiliaries, sources = probe.constitutive_names(side)
    for name in auxiliaries:
        getattr(fields, name)[...] = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    inverse: Dict[str, Any] = {}
    if side == "E":
        for target in targets:
            inverse["inv_eps_" + target] = rng.uniform(
                0.2, 0.9, size=shape).astype(np.float32)
        # One aliased pair is what the engine installs for an isotropic material;
        # per-component volumes are what a smoothed one installs. The kernel binds
        # three pointers either way, so the harsher (three distinct) case is used.
        fields.set_epsilon_volumes(
            {t: 1.0 / inverse["inv_eps_" + t] for t in targets},
            {t: inverse["inv_eps_" + t] for t in targets})

    before = {name: getattr(fields, name).copy()
              for name in targets + auxiliaries + sources}
    arrays = dict(before)
    arrays.update(inverse)

    (stepping.update_H if side == "H" else stepping.update_E)(fields, layer)
    array_path = {name: getattr(fields, name).copy() for name in targets + auxiliaries}

    coefficients = probe.layer_constitutive_coefficients(
        layer, probe.CONSTITUTIVE_HALF_INTEGER[side])
    state = {name: before[name].copy() for name in targets + auxiliaries}
    probe.reference_constitutive_step(side, arrays, state, coefficients, "array_order")
    case["vs_stepping"] = probe.combine(
        {name: probe.bit_compare(array_path[name], state[name])
         for name in targets + auxiliaries})

    # NEGATIVE CONTROL: the regrouped association must NOT match. Unlike the curl's
    # control this one is asserted at BOTH Courants, because the non-representable
    # multiplicands here are the PML coefficients themselves (§12.2) and not the
    # Courant scaling — there is no exact-in-binary case to exempt.
    control = {name: before[name].copy() for name in targets + auxiliaries}
    probe.reference_constitutive_step(side, arrays, control, coefficients, "regrouped")
    regrouped = probe.combine({name: probe.bit_compare(array_path[name], control[name])
                               for name in targets + auxiliaries})
    case["regrouped_control"] = {
        "bit_identical_to_stepping": regrouped["bit_identical"],
        "differing_floats": regrouped["differing_floats"],
        "max_ulp": regrouped.get("max_ulp", 0),
        "asserted": True,
    }

    moved = {name: not np.array_equal(before[name], array_path[name])
             for name in targets + auxiliaries}
    case["arrays_changed"] = moved
    case["all_arrays_changed"] = all(moved.values())
    del grid
    return case


#: Pole frequencies for the dispersive leg. ``require_stable`` refuses any term with
#: ``f >= 1/(pi*dt)``; this grid has ``dx = 1`` so ``dt`` is the Courant number
#: itself and the tightest limit in the sweep is 0.637 (at dtdx = 0.5). Six terms at
#: 0.11 + 0.07n top out at 0.46, inside it at BOTH Courants — a refusal here would be
#: the engine being right and would silently empty the leg.
DISPERSIVE_FREQUENCIES = tuple(0.11 + 0.07 * n for n in range(8))


def build_polarizations(grid, fields, counts: Sequence[int]):
    """Register ``max(counts)`` real susceptibilities giving each component its own set.

    Pole ``n`` drives component ``c`` iff ``counts[c] > n``, and it is given a sigma
    of exactly zero on every other component — which is how the ENGINE decides:
    ``PolarizationState._driven`` (dispersion.py:640-642) drops any component whose
    sigma is identically zero via ``sigma_is_trivial`` (:600), and MEEP does the same
    (``trivial_sigma`` feeding ``needs_P``). So a ``(1, 0, 2)`` row is not a harness
    contrivance: it is the ordinary anisotropic-sigma material, and it is the row that
    catches a plan carrying one pole list for all three components.

    Returns the states and, per component, the ordered list of the ones that drive it
    — the same list ``Fields.displacement_minus_polarization`` walks (fields.py:1097).
    """
    from meep_gpu.dispersion import PolarizationState, Susceptibility  # noqa: PLC0415

    components = ("Ex", "Ey", "Ez")
    states = []
    for index in range(max(int(c) for c in counts)):
        term = Susceptibility(frequency=DISPERSIVE_FREQUENCIES[index], gamma=1e-5)
        term.require_stable(grid.dt)
        sigma = {name: (0.4 + 0.1 * index) if int(counts[axis]) > index else 0.0
                 for axis, name in enumerate(components)}
        state = PolarizationState(term, sigma, grid, fields._field_dtype())
        fields.polarizations.append(state)
        states.append(state)
    order = {name: [s for s in states if s.drives(name)] for name in components}
    for axis, name in enumerate(components):
        if len(order[name]) != int(counts[axis]):
            raise RuntimeError(
                f"{name} ended up with {len(order[name])} poles, not {counts[axis]}; "
                f"the sigma-is-trivial rule is not doing what this leg assumes")
    return states, order


def one_dispersive_case(shape, boundaries, courant, counts) -> Dict[str, Any]:
    """``stepping.update_E``'s DISPERSIVE branch against the probe's third transcription.

    Same hole, same closure, one feature later. The device gate compares the kernel
    against ``probe.reference_dispersive_constitutive_step``; if that transcription is
    wrong the kernel can be bit-identical to it and still wrong against the engine,
    and only this leg — on a laptop, before any device time — can catch it.

    THREE CONTROLS, not one. Besides the regrouped accumulation this leg carries the
    two defects that exist only because a pole LIST entered the sub-step:

    * ``summed_poles``   — ``D - (P0 + P1)`` instead of ``(D - P0) - P1``;
    * ``reversed_poles`` — the list walked backwards.

    Both are asserted to differ ONLY on rows where some component carries two or more
    poles. At one pole they are the same arithmetic and coincide, and a leg that
    demanded a difference there would be asserting something false — the artifact
    records ``asserted`` per control so the distinction is in the record rather than
    in a reader's head.

    ``P`` IS ALSO COMPARED, and it must be UNCHANGED: ``update_E`` consumes the
    polarization and ``update_P`` advances it (stepping.py:954 then :1389). A kernel
    or a transcription that advanced P here would be wrong in a way that still
    produces a smooth field.
    """
    case: Dict[str, Any] = {
        "shape": list(shape), "boundaries": list(boundaries),
        "courant": repr(courant), "sub_step": "update_E_dispersive",
        "pole_counts": [int(c) for c in counts],
    }
    walled_invariant = [a for a in range(3)
                        if shape[a] == 1 and boundaries[a] == "metallic"]
    if walled_invariant:
        case["skipped"] = (f"axis {walled_invariant[0]} has one cell and a metallic "
                           f"wall, which Grid refuses")
        return case
    try:
        grid, fields, layer = build(shape, boundaries, courant)
        states, order = build_polarizations(grid, fields, counts)
    except Exception as exc:  # noqa: BLE001 - a refusal is a result, not a crash
        case["skipped"] = f"{type(exc).__name__}: {exc}"[:400]
        return case

    rng = np.random.default_rng(probe.SEED + 31)
    seed_fields(fields, shape, rng)
    targets = tuple(t[0] for t in probe.DISPERSIVE_TERMS)
    auxiliaries = tuple("f_w_" + t for t in targets)
    for name in auxiliaries:
        getattr(fields, name)[...] = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    inverse: Dict[str, Any] = {}
    for target in targets:
        inverse["inv_eps_" + target] = rng.uniform(
            0.2, 0.9, size=shape).astype(np.float32)
    fields.set_epsilon_volumes(
        {t: 1.0 / inverse["inv_eps_" + t] for t in targets},
        {t: inverse["inv_eps_" + t] for t in targets})
    # Seeded, NONZERO P and P_prev. A zero P makes D - P bit-equal to D and hides
    # the whole feature this leg exists to pin.
    for state in states:
        for name in state.driven():
            state.P[name][...] = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
            state.P_prev[name][...] = rng.uniform(-1.0, 1.0,
                                                  size=shape).astype(np.float32)

    # The probe's reference is keyed by array NAME, and the n-th name of a component
    # is the n-th state that drives it, in fields.polarizations order.
    pole_names = probe.dispersive_pole_names(counts)
    arrays: Dict[str, Any] = {}
    for name in targets + auxiliaries + tuple(t[1] for t in probe.DISPERSIVE_TERMS):
        arrays[name] = getattr(fields, name).copy()
    arrays.update(inverse)
    for target in targets:
        for index, label in enumerate(pole_names[target]):
            arrays[label] = order[target][index].P[target].copy()
    before = {name: arrays[name].copy() for name in targets + auxiliaries}
    p_before = [{c: s.P[c].copy() for c in s.driven()} for s in states]

    stepping.update_E(fields, layer)
    array_path = {name: getattr(fields, name).copy() for name in targets + auxiliaries}

    case["P_untouched_by_update_E"] = all(
        p_before[i][c].tobytes() == states[i].P[c].tobytes()
        for i in range(len(states)) for c in states[i].driven())

    coefficients = probe.layer_constitutive_coefficients(layer, True)
    state_arrays = {name: before[name].copy() for name in targets + auxiliaries}
    probe.reference_dispersive_constitutive_step(
        arrays, state_arrays, coefficients, pole_names, "array_order", "sequential")
    case["vs_stepping"] = probe.combine(
        {name: probe.bit_compare(array_path[name], state_arrays[name])
         for name in targets + auxiliaries})

    multi_pole = bool(max(int(c) for c in counts) > 1)
    for label, grouping, pole_grouping, asserted in (
            ("regrouped_control", "regrouped", "sequential", True),
            ("summed_poles_control", "array_order", "summed", multi_pole),
            ("reversed_poles_control", "array_order", "reversed", multi_pole)):
        control = {name: before[name].copy() for name in targets + auxiliaries}
        probe.reference_dispersive_constitutive_step(
            arrays, control, coefficients, pole_names, grouping, pole_grouping)
        verdict = probe.combine({name: probe.bit_compare(array_path[name], control[name])
                                 for name in targets + auxiliaries})
        case[label] = {
            "bit_identical_to_stepping": verdict["bit_identical"],
            "differing_floats": verdict["differing_floats"],
            "max_ulp": verdict.get("max_ulp", 0),
            "asserted": asserted,
        }

    moved = {name: not np.array_equal(before[name], array_path[name])
             for name in targets + auxiliaries}
    case["arrays_changed"] = moved
    case["all_arrays_changed"] = all(moved.values())
    del grid
    return case


def one_fused_pair_case(shape, boundaries, dtdx) -> Dict[str, Any]:
    """The THREE array-path passes the fused kernel replaces, against the probe's copy.

    Same hole, same closure, one COMPOSITION later. The device gate compares the
    fused kernel against ``probe.reference_fused_pair_step``, which is
    ``reference_pml_step`` + ``reference_zero_metal_B`` + ``reference_constitutive_step``
    — and the middle one is a NEW transcription, of ``stepping.zero_metal_B``. If it
    is wrong the kernel can be bit-identical to it and still wrong against the
    engine, and only this leg can catch that. It runs on a laptop, before any device
    time is spent.

    The auxiliaries are seeded nonzero, and on a walled run that is what makes the
    wall wipe observable at all — see ``build_whole_step_driver``'s
    ``seed_auxiliaries`` for the measurement behind that sentence.
    """
    case: Dict[str, Any] = {
        "shape": list(shape), "boundaries": list(boundaries),
        "dtdx": repr(dtdx), "sub_step": "fused_pair_B",
    }
    walled_invariant = [a for a in range(3)
                        if shape[a] == 1 and boundaries[a] == "metallic"]
    if walled_invariant:
        case["skipped"] = (f"axis {walled_invariant[0]} has one cell and a metallic "
                           f"wall, which Grid refuses")
        return case
    try:
        grid, fields, layer = build(shape, boundaries, dtdx)
    except Exception as exc:  # noqa: BLE001
        case["skipped"] = f"{type(exc).__name__}: {exc}"[:400]
        return case

    rng = np.random.default_rng(probe.SEED + 17)
    seed_fields(fields, shape, rng)
    for name in probe.FUSED_PAIR_H_AUX:
        getattr(fields, name)[...] = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)

    names = probe.FUSED_PAIR_STATE
    before = {name: getattr(fields, name).copy() for name in names}
    arrays = {name: getattr(fields, name).copy() for name in probe.FUSED_PAIR_SOURCES}

    # The engine's own three passes, in the driver's order (driver.py:3163/:3167/:3169).
    stepping.step_B(fields, layer)
    stepping.zero_metal_B(fields)
    stepping.update_H(fields, layer)
    array_path = {name: getattr(fields, name).copy() for name in names}

    zero_metal = tuple(bool(grid.is_metallic(axis) and not grid.is_mirrored(axis))
                       for axis in range(3)) if grid.has_metallic else (False, False, False)
    case["zero_metal"] = [bool(f) for f in zero_metal]
    case["zero_metal_matches_coverage"] = list(zero_metal) == list(
        zero_metal_axes(grid))

    state = {name: before[name].copy() for name in names}
    probe.reference_fused_pair_step(
        np, arrays, state, probe.layer_coefficients(layer, True),
        probe.layer_constitutive_coefficients(layer, False),
        np.float32(dtdx), boundaries, zero_metal, "array_order")
    case["vs_stepping"] = probe.combine(
        {name: probe.bit_compare(array_path[name], state[name]) for name in names})

    control = {name: before[name].copy() for name in names}
    probe.reference_fused_pair_step(
        np, arrays, control, probe.layer_coefficients(layer, True),
        probe.layer_constitutive_coefficients(layer, False),
        np.float32(dtdx), boundaries, zero_metal, "regrouped")
    regrouped = probe.combine({name: probe.bit_compare(array_path[name], control[name])
                               for name in names})
    case["regrouped_control"] = {
        "bit_identical_to_stepping": regrouped["bit_identical"],
        "differing_floats": regrouped["differing_floats"],
        "max_ulp": regrouped.get("max_ulp", 0),
        "asserted": True,
    }
    moved = {name: not np.array_equal(before[name], array_path[name]) for name in names}
    case["arrays_changed"] = moved
    case["all_arrays_changed"] = all(moved.values())
    del grid
    return case


def one_fused_electric_case(shape, boundaries, dtdx) -> Dict[str, Any]:
    """Pin the D/wall/E composition used by the new device gate to the engine."""
    case: Dict[str, Any] = {
        "shape": list(shape), "boundaries": list(boundaries),
        "dtdx": repr(dtdx), "sub_step": "fused_pair_D",
    }
    walled_invariant = [a for a in range(3)
                        if shape[a] == 1 and boundaries[a] == "metallic"]
    if walled_invariant:
        case["skipped"] = (f"axis {walled_invariant[0]} has one cell and a metallic "
                           f"wall, which Grid refuses")
        return case
    try:
        grid, fields, layer = build(shape, boundaries, dtdx)
    except Exception as exc:  # noqa: BLE001
        case["skipped"] = f"{type(exc).__name__}: {exc}"[:400]
        return case

    rng = np.random.default_rng(probe.SEED + 71)
    seed_fields(fields, shape, rng)
    for name in fused_electric.E_TARGETS + fused_electric.E_AUX:
        getattr(fields, name)[...] = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    inverse = {
        f"inv_eps_{name}": rng.uniform(0.2, 0.9, size=shape).astype(np.float32)
        for name in fused_electric.E_TARGETS
    }
    fields.set_epsilon_volumes(
        {name: 1.0 / inverse[f"inv_eps_{name}"] for name in fused_electric.E_TARGETS},
        {name: inverse[f"inv_eps_{name}"] for name in fused_electric.E_TARGETS},
    )

    names = fused_electric.STATE_NAMES
    before = {name: getattr(fields, name).copy() for name in names}
    arrays = {name: getattr(fields, name).copy() for name in fused_electric.SOURCES}
    arrays.update(inverse)

    stepping.step_D(fields, layer)
    stepping.zero_metal_D(fields)
    stepping.update_E(fields, layer)
    array_path = {name: getattr(fields, name).copy() for name in names}

    zero_metal = tuple(bool(grid.is_metallic(axis) and not grid.is_mirrored(axis))
                       for axis in range(3)) if grid.has_metallic else (False, False, False)
    case["zero_metal"] = [bool(flag) for flag in zero_metal]
    case["zero_metal_matches_coverage"] = list(zero_metal) == list(zero_metal_axes(grid))

    curl_coefficients = probe.layer_coefficients(layer, False)
    constitutive_coefficients = probe.layer_constitutive_coefficients(layer, True)
    state = {name: before[name].copy() for name in names}
    fused_electric.reference_fused_electric_step(
        np, arrays, state, curl_coefficients, constitutive_coefficients,
        np.float32(dtdx), boundaries, zero_metal, "array_order")
    case["vs_stepping"] = probe.combine(
        {name: probe.bit_compare(array_path[name], state[name]) for name in names})

    control = {name: before[name].copy() for name in names}
    fused_electric.reference_fused_electric_step(
        np, arrays, control, curl_coefficients, constitutive_coefficients,
        np.float32(dtdx), boundaries, zero_metal, "regrouped")
    regrouped = probe.combine({
        name: probe.bit_compare(array_path[name], control[name]) for name in names})
    case["regrouped_control"] = {
        "bit_identical_to_stepping": regrouped["bit_identical"],
        "differing_floats": regrouped["differing_floats"],
        "max_ulp": regrouped.get("max_ulp", 0),
        "asserted": True,
    }
    moved = {name: not np.array_equal(before[name], array_path[name]) for name in names}
    case["arrays_changed"] = moved
    case["all_arrays_changed"] = all(moved.values())
    del grid
    return case


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="JSON artifact path")
    args = parser.parse_args(argv)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    #: One row per pole set, spelled into the sub-step name so the sweep stays a
    #: flat product and the log line names the configuration it measured.
    dispersive_sub_steps = tuple(
        "dispersive:" + "-".join(str(int(c)) for c in counts)
        for counts in probe.DISPERSIVE_POLE_COUNTS)
    combinations = [
        (shape, boundaries, dtdx, sub_step)
        for shape in probe.PML_SHAPES
        for boundaries in probe.PML_BOUNDARY_SETS
        for dtdx in probe.PML_DTDX
        for sub_step in ("step_B", "step_D", "update_H", "update_E",
                         "fused_pair_B", "fused_pair_D") + dispersive_sub_steps
    ]
    results: Dict[str, Any] = {
        "check": "pml_reference_vs_stepping",
        "backend": "numpy",
        "seed": probe.SEED,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "numpy_version": np.__version__,
        "python": sys.version.split()[0],
        "cases": [],
    }
    started = time.time()
    for index, (shape, boundaries, dtdx, sub_step) in enumerate(combinations, 1):
        case_started = time.time()
        if sub_step == "fused_pair_B":
            case = one_fused_pair_case(shape, boundaries, dtdx)
        elif sub_step == "fused_pair_D":
            case = one_fused_electric_case(shape, boundaries, dtdx)
        elif sub_step.startswith("dispersive:"):
            case = one_dispersive_case(
                shape, boundaries, dtdx,
                tuple(int(v) for v in sub_step.split(":", 1)[1].split("-")))
        elif sub_step.startswith("update_"):
            case = one_constitutive_case(shape, boundaries, dtdx,
                                         sub_step.split("_", 1)[1])
        else:
            case = one_case(shape, boundaries, dtdx, sub_step)
        case["seconds"] = round(time.time() - case_started, 3)
        results["cases"].append(case)
        key = (f"{'x'.join(str(v) for v in shape)}|"
               f"{'/'.join(b[0] for b in boundaries)}|dtdx={dtdx!r}|{sub_step}")
        if case.get("skipped"):
            log(f"[ref] case {index}/{len(combinations)} {key}: SKIPPED "
                f"({case['skipped']})")
        else:
            verdict = case["vs_stepping"]
            log(f"[ref] case {index}/{len(combinations)} {key}: "
                f"identical={verdict['bit_identical']} "
                f"(differing={verdict['differing_floats']}/{verdict['total_floats']}, "
                f"maxulp={verdict.get('max_ulp', 0)}) "
                f"all_arrays_changed={case['all_arrays_changed']} "
                f"regrouped_control_differs="
                f"{not case['regrouped_control']['bit_identical_to_stepping']}"
                f"{'' if case['regrouped_control']['asserted'] else ' (not asserted at this dtdx)'} "
                f"({case['seconds']} s)")
        with open(args.out + ".tmp", "w") as handle:
            json.dump(results, handle, indent=2)
        os.replace(args.out + ".tmp", args.out)

    ran = [c for c in results["cases"] if not c.get("skipped")]
    identical = [c for c in ran if c["vs_stepping"]["bit_identical"]]
    inert = [c for c in ran if not c["all_arrays_changed"]]
    asserted_controls = [c for c in ran if c["regrouped_control"]["asserted"]]
    blind_controls = [c for c in asserted_controls
                      if c["regrouped_control"]["bit_identical_to_stepping"]]
    # The dispersive rows carry two more controls, and they are asserted only where
    # they can discriminate: at one pole ``D - (P0)`` and ``D - (P0)`` are the same
    # arithmetic, so demanding a difference there would assert something false.
    dispersive = [c for c in ran if c["sub_step"] == "update_E_dispersive"]
    pole_controls = [(c, name) for c in dispersive
                     for name in ("summed_poles_control", "reversed_poles_control")]
    pole_asserted = [(c, n) for c, n in pole_controls if c[n]["asserted"]]
    pole_blind = [(c, n) for c, n in pole_asserted
                  if c[n]["bit_identical_to_stepping"]]
    results["summary"] = {
        "ran": len(ran),
        "skipped": len(results["cases"]) - len(ran),
        "identical": len(identical),
        "cases_with_an_unchanged_array": len(inert),
        "regrouped_control_asserted": len(asserted_controls),
        "regrouped_control_indistinguishable": len(blind_controls),
        "dispersive_ran": len(dispersive),
        "dispersive_identical": sum(int(c["vs_stepping"]["bit_identical"])
                                    for c in dispersive),
        "pole_controls_asserted": len(pole_asserted),
        "pole_controls_indistinguishable": len(pole_blind),
        "P_untouched_by_update_E": all(c.get("P_untouched_by_update_E")
                                       for c in dispersive),
        "pass": (len(ran) > 0 and len(identical) == len(ran) and not inert
                 and bool(asserted_controls) and not blind_controls
                 and bool(pole_asserted) and not pole_blind
                 and all(c.get("P_untouched_by_update_E") for c in dispersive)),
    }
    results["elapsed_seconds"] = round(time.time() - started, 2)
    with open(args.out + ".tmp", "w") as handle:
        json.dump(results, handle, indent=2)
    os.replace(args.out + ".tmp", args.out)
    log(f"[done] reference == stepping.py on "
        f"{results['summary']['identical']}/{results['summary']['ran']} cases "
        f"({results['summary']['skipped']} skipped); "
        f"pass={results['summary']['pass']} "
        f"({results['elapsed_seconds']} s) -> {args.out}")
    return 0 if results["summary"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
