"""Does the DISPERSIVE D/E pair's closed form reproduce the driver's five in-seam passes?

THE CLAIM THIS PROBE STATES AND MEASURES. One launch per cell can perform

    step_D (driver.py:3302) -> fill_symmetry_bc_D (:3309) -> zero_metal_D (:3310)
    -> fill_folded_far_ghosts_D (:3311) -> pole-aware update_E (:3313)

with the displacement carried in a register, the two mirror fills carried by the
OWNERSHIP INVERSION (the source thread writes each imaged ghost), and the ordered
``D - sum_n P_n`` chain evaluated AT THE CELL EACH ``constitutive_apply`` WRITES --
which for a ghost is the DESTINATION's pole words and the DESTINATION's inverse
epsilon, not the source thread's own.

WHY THAT LAST CLAUSE IS THE WHOLE PROBE. ``fused_electric_pair`` already carries the
three middle passes for the NON-dispersive constitutive half, and
``probe_cuda_electric_fill_carry_order.py`` measured that composition. Adding poles
adds a per-cell volume the E half reads that the fills DO NOT image: the array path's
``update_E`` runs over the whole volume and evaluates
``fields.displacement_minus_polarization`` cell by cell (fields.py:1096-1105), so at a
ghost cell it subtracts THAT cell's ``P``. A carry that re-used the source thread's
pole registers -- the obvious port, and the cheap one -- computes a different field on
every imaged plane. Nothing in either sibling backend measures this, because BOTH
refuse a folded grid on this family:

  * ``metal_kernels/fused_dispersive_pair.py`` refuses through the E half's own
    "a mirror plane is active" clause and ships the fold as a SEPARATE product;
  * ``triton_kernels/dispersive_fused_pair.py`` refuses the same way through
    ``dispersive_update_e.dispersive_constitutive_coverage``.

CUDA's ``covers_real_pml_dispersive_constitutive`` ADMITS a fold (up to
``ADE_FOLD_PLANES_SWEPT``), and 4 of the 7 corpus rows this cell owns are folded, so
the CUDA product carries what neither sibling does and this arithmetic has never been
measured anywhere.

WHAT IS COMPARED. Raw uint32 WORDS of every stored volume the seam writes -- ``Dx/Dy/Dz``,
``fu_Dx/fu_Dy/fu_Dz``, ``Ex/Ey/Ez``, ``f_w_Ex/f_w_Ey/f_w_Ez`` -- against the array
path's own five passes run from identical state, over COMPLETE steps. Never
``allclose``: ``-0.0 == 0.0`` lies, and the sign of zero is exactly the class of
divergence the near-fill ordering produces.

THE POLE VOLUMES ARE SEEDED NON-ZERO, and that is not a detail. Zero is a fixed point
of ``s = s - P``: a fixture with zero poles turns every pole mutation into a null and
would report a green sweep that measured nothing. ``poles_are_live`` is asserted per
case and a case that cannot arm the pole nulls is recorded VACUOUS rather than passed.

THE SIX ARMED MUTATIONS, each a port a reasonable person would write:

  poles_at_source        the ghost chain subtracts the SOURCE thread's poles
  inv_eps_at_source      the ghost source multiplies the SOURCE thread's inverse epsilon
  subtract_before_clear  the own-cell chain consumes the PRE-clear register
  near_post_clear        the near product multiplies the POST-clear register
  pole_order_reversed    the chain is summed right to left
  far_coefficient_source the far ghost takes the SOURCE row's kps/kms pair

A mutation that does not diverge on a configuration is a NULL, and every null is
recorded with the reason it is unreachable there rather than being silently counted as
agreement. A configuration where no mutation diverges is reported vacuous.

Progress is one flushed line per configuration and the artifact is rewritten after each
one, so an interrupted run keeps every row that landed (the progress-reporting rule).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_API = _HERE.parents[1]
if str(_REPO_API) not in sys.path:
    sys.path.insert(0, str(_REPO_API))

from meep_gpu import stepping  # noqa: E402
from meep_gpu.cuda_kernels import fused_electric_pair as real  # noqa: E402
from meep_gpu.dispersion import PolarizationState, Susceptibility  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid, Mirror  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402

#: Every stored volume the seam reads or writes. The auxiliaries start NONZERO for the
#: reason every leg on this track gives: a zero ``fu`` makes ``fu * kms`` exactly zero
#: on the first pass whatever ``kms`` is, which would hide a mis-indexed coefficient.
_SEEDED: Tuple[str, ...] = (
    "Dx", "Dy", "Dz", "Hx", "Hy", "Hz", "Ex", "Ey", "Ez",
    "fu_Dx", "fu_Dy", "fu_Dz", "f_w_Ex", "f_w_Ey", "f_w_Ez")

#: What the comparison covers, per complete step.
_COMPARED: Tuple[str, ...] = (
    "Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz",
    "Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez")

_TARGETS: Tuple[str, ...] = ("Dx", "Dy", "Dz")
_ELECTRIC: Tuple[str, ...] = ("Ex", "Ey", "Ez")
_AXIS_OF = {"x": 0, "y": 1, "z": 2}

#: ``in_seam_passes._zero_metal_D_kernel_code``'s OFF-DIAGONAL, as the axes each
#: component is cleared on. Read from the SHIPPED module's own table rather than typed,
#: so a change there reaches this probe instead of being transcribed twice.
_CLEAR_AXES: Dict[str, Tuple[int, ...]] = {
    axis: tuple(_AXIS_OF[coordinate_axis]
                for _flag, coordinate, axes in real._ZERO_METAL_ROWS
                for coordinate_axis in ("x", "y", "z")
                if axis in axes and coordinate == {"x": "i", "y": "j", "z": "k"}[
                    coordinate_axis])
    for axis in ("x", "y", "z")
}

SEED = 20260902

MUTATIONS: Tuple[str, ...] = (
    "poles_at_source", "inv_eps_at_source", "subtract_before_clear",
    "near_post_clear", "pole_order_reversed", "far_coefficient_source")

#: Why each mutation can be UNREACHABLE on a configuration. Consulted when it does not
#: diverge, so a null is recorded with its reason instead of counted as agreement.
_NULL_REASONS: Dict[str, str] = {
    "poles_at_source": "no ghost is imaged, or no imaged component carries a pole",
    "inv_eps_at_source": "no ghost is imaged",
    "subtract_before_clear":
        "no wall clears a cell of a component that carries a pole",
    "near_post_clear":
        "no component is folded on one near axis and walled on the other, which is the "
        "only shape where the driver's order between :3309 and :3310 is observable",
    "pole_order_reversed": "no component carries two or more poles",
    "far_coefficient_source":
        "no far ghost is imaged, or the absorber profile does not differ between the "
        "reflect row and the top plane",
}

#: Configurations. THE FIRST TWO ARE THE CORPUS SHAPES this cell actually owns, read
#: off the stamped census: the three stochastic_emitter rows are all-periodic with an
#: active layer, and the four TestLoadDump rows are metallic on x, mirror-folded on y
#: and periodic on z. The rest exist to arm mutations those two cannot.
CASES: Tuple[Dict[str, Any], ...] = (
    {"label": "corpus_all_periodic", "cell": (9., 10., 11.),
     "boundaries": ("periodic",) * 3, "symmetry": (), "pml": 2,
     "arity": (2, 2, 2),
     "why": "the stochastic_emitter shape: PML, all periodic, every component driven"},
    {"label": "corpus_wall_x_fold_y", "cell": (9., 10., 11.),
     "boundaries": ("metallic", "metallic", "periodic"),
     "symmetry": (("Y", 1),), "pml": 2, "arity": (2, 1, 2),
     "why": "the TestLoadDump shape: metallic x and y with y ALSO mirror-folded, so "
            "zero_metal_D walls x alone and the near fill runs on y"},
    {"label": "wall_xyz", "cell": (9., 10., 11.), "boundaries": ("metallic",) * 3,
     "symmetry": (), "pml": 2, "arity": (1, 2, 3),
     "why": "every wall row live, no fold: arms subtract_before_clear alone"},
    {"label": "fold_y_periodic", "cell": (9., 10., 11.),
     "boundaries": ("periodic",) * 3, "symmetry": (("Y", 1),), "pml": 2,
     "arity": (2, 2, 2),
     "why": "both fills live on one axis: arms poles_at_source and inv_eps_at_source"},
    {"label": "fold_y_odd_wall_z", "cell": (9., 10., 11.),
     "boundaries": ("periodic", "periodic", "metallic"), "symmetry": (("Y", -1),),
     "pml": 2, "arity": (2, 2, 2),
     "why": "THE CROSS TERM: Dx folded on y at an ODD phase and walled on z, so its "
            "near ghost lands in the cleared plane -- arms near_post_clear"},
    {"label": "fold_x_deep_pml", "cell": (9., 10., 11.),
     "boundaries": ("periodic",) * 3, "symmetry": (("X", 1),),
     "pml": ((0, 5), (2, 2), (2, 2)), "arity": (3, 2, 2),
     "why": "an absorber reaching the far ghost's coefficient row, so the "
            "destination's kps entry differs from the source's"},
    {"label": "fold_xy_mixed_phase", "cell": (9., 10., 11.),
     "boundaries": ("periodic",) * 3, "symmetry": (("X", 1), ("Y", -1)), "pml": 2,
     "arity": (2, 3, 1),
     "why": "two folded axes at mixed phases: a source thread is itself a fill "
            "destination, and the deepest composition the closed form goes"},
    {"label": "degenerate_arity_zero_on_Ey", "cell": (9., 10., 11.),
     "boundaries": ("periodic",) * 3, "symmetry": (("Y", 1),), "pml": 2,
     "arity": (2, 0, 1),
     "why": "one component with NO pole: the emitted line collapses to the certified "
            "non-dispersive one and the chain must not be formed at all"},
)


# ---------------------------------------------------------------------------
# The fixture
# ---------------------------------------------------------------------------

def build(case: Dict[str, Any]):
    """One seeded NumPy engine: ``(fields, grid, pml)`` with LIVE poles."""
    rng = np.random.default_rng(SEED)
    grid = Grid(resolution=1.0, cell_size=tuple(case["cell"]),
                boundaries=tuple(case["boundaries"]),
                symmetry=tuple(Mirror(axis, int(phase))
                               for axis, phase in case["symmetry"]),
                xp=np, courant=0.5)
    fields = Fields(grid=grid)
    fields.enable_pml_storage()
    # A FOLDED AXIS TAKES ITS ABSORBER ON THE HIGH FACE ONLY: cell 0 lies ON the mirror
    # plane, a boundary condition rather than an outer wall (pml.py:399-412).
    folded = {"XYZ".index(axis) for axis, _phase in case["symmetry"]}
    if isinstance(case["pml"], int):
        thickness = tuple((0, case["pml"]) if axis in folded
                          else (case["pml"], case["pml"]) for axis in range(3))
    else:
        thickness = tuple(tuple(int(v) for v in pair) for pair in case["pml"])
    pml = PML(grid=grid, thickness=thickness)
    shape = tuple(int(n) for n in grid.shape)
    for name in _SEEDED:
        array = getattr(fields, name, None)
        if array is None:
            continue
        array[...] = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)

    # THREE INDEPENDENT inverse-epsilon volumes, never one array bound thrice, and
    # drawn AWAY from 1.0 so dropping the multiply is not the identity.
    epsilon, inverse = {}, {}
    for component in _ELECTRIC:
        values = rng.uniform(1.2, 3.4, size=shape).astype(np.float32)
        epsilon[component] = values
        inverse[component] = (np.float32(1.0) / values).astype(np.float32)
    fields.set_epsilon_volumes(epsilon, inverse)

    # REAL PolarizationState objects at the requested arity: a per-component sigma of
    # zero is the engine's own spelling of "this term does not drive that component"
    # (dispersion.py:645-647). A fixture that poked ``_driven`` would measure an object
    # the engine cannot build.
    arity = tuple(int(value) for value in case["arity"])
    kinds = ("lorentzian", "drude")
    for index in range(max(arity)):
        term = Susceptibility(frequency=0.20 + 0.03 * index, gamma=0.008,
                              kind=kinds[index % 2])
        sigma = {name: (0.35 + 0.04 * index if index < arity[axis] else 0.0)
                 for axis, name in enumerate(_ELECTRIC)}
        fields.polarizations.append(
            PolarizationState(term, sigma, grid, fields._field_dtype()))
    # THE POLE VOLUMES ARE SEEDED NON-ZERO. Zero is a fixed point of `s = s - P`, so a
    # zero fixture turns every pole mutation into a null and the sweep would measure
    # nothing at all.
    for state in fields.polarizations:
        for store in (state.P, state.P_prev):
            for component, array in store.items():
                array[...] = rng.uniform(-0.6, 0.6, size=shape).astype(np.float32)
    return fields, grid, pml


def pole_chain(fields: Any) -> Dict[str, List[np.ndarray]]:
    """``fields.polarizations`` filtered per component, IN ORDER -- the array path's own
    reading (fields.py:1099-1104), because the order is the arithmetic here."""
    return {component: [state.P[component] for state in fields.polarizations
                        if state.drives(component)]
            for component in _ELECTRIC}


def words(array: Any) -> np.ndarray:
    """Raw uint32 WORDS. Byte compares, never ``allclose``."""
    return np.ascontiguousarray(array, dtype=np.float32).ravel().view(np.uint32)


# ---------------------------------------------------------------------------
# The closed form
# ---------------------------------------------------------------------------

def closed_form(pre: Dict[str, np.ndarray], state: Dict[str, np.ndarray],
                poles: Dict[str, List[np.ndarray]],
                coefficients: Dict[str, Tuple[np.ndarray, np.ndarray]],
                shape: Tuple[int, int, int], near: Sequence[int],
                reflect: Sequence[int], phase: Sequence[np.float32],
                walls: Sequence[bool], mutation: str = "",
                ) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """The kernel's own composition, evaluated as arrays.

    ONE ARRAY OP PER EMITTED LINE, in the order the emitted blocks run them, and the
    ownership guard is the mask those blocks nest inside. The WRITE COUNT per cell is
    returned beside the volumes: the carry's whole legality argument is that every
    destination word is written by exactly one thread, and a probe that compared only
    values would not notice two threads racing on one word.
    """
    out: Dict[str, np.ndarray] = {}
    writes: Dict[str, int] = {}
    index = np.indices(shape)
    for target, name in enumerate(_TARGETS):
        letter = name[-1]
        component = "E" + letter
        near_axes = real.near_fill_axes(target)
        far_axes = real.far_fill_axes(target)
        chain = list(poles[component])
        if mutation == "pole_order_reversed":
            chain = list(reversed(chain))
        inv_eps = state["inv_eps_" + component]
        kps, kms = coefficients[letter]

        owned = np.ones(shape, dtype=bool)
        for axis in near_axes:
            if near[axis]:
                owned &= index[axis] != 0
        for axis in far_axes:
            if reflect[axis] >= 0:
                owned &= index[axis] != shape[axis] - 1
        cleared = np.zeros(shape, dtype=bool)
        for axis in _CLEAR_AXES[letter]:
            if walls[axis]:
                cleared |= index[axis] == 0

        # --- the curl's register, the pre-clear copy, and zero_metal_D --------------
        volume = np.array(pre[name], copy=True)
        volume[owned & cleared] = np.float32(0.0)

        electric = np.array(state[component], copy=True)
        auxiliary = np.array(state["f_w_" + component], copy=True)
        written = np.zeros(shape, dtype=np.int32)

        # --- update_E at this thread's own cell -----------------------------------
        source = np.array(pre[name] if mutation == "subtract_before_clear" else volume,
                          copy=True, dtype=np.float32)
        for array in chain:
            source -= array
        product = (source * inv_eps).astype(np.float32)
        previous = np.array(auxiliary, copy=True)
        auxiliary = np.where(owned, product, auxiliary).astype(np.float32)
        updated = ((electric + (kps * product).astype(np.float32)).astype(np.float32)
                   - (kms * previous).astype(np.float32)).astype(np.float32)
        electric = np.where(owned, updated, electric).astype(np.float32)
        written[owned] += 1

        # --- every imaged ghost this thread owns, then update_E at each ------------
        for far_subset, near_subset in real.carried_destinations(near_axes, far_axes):
            source_index: List[Any] = [slice(None)] * 3
            destination: List[Any] = [slice(None)] * 3
            live = True
            for axis in far_subset:
                if reflect[axis] < 0:
                    live = False
                    break
                source_index[axis] = int(reflect[axis])
                destination[axis] = shape[axis] - 1
            for axis in near_subset:
                if not live or not near[axis]:
                    live = False
                    break
                source_index[axis] = real.NEAR_SOURCE_INDEX
                destination[axis] = 0
            if not live:
                continue
            source_key = tuple(source_index)
            destination_key = tuple(destination)
            own_source = owned[source_key]
            cleared_source = cleared[source_key]

            # 1. the NEAR product, from the PRE-clear displacement (:3309 before :3310)
            value = np.array(pre[name][source_key], copy=True)
            if near_subset:
                weight = np.float32(1.0)
                for axis in near_subset:
                    weight = np.float32(weight * phase[axis])
                if mutation == "near_post_clear":
                    value = np.float32(weight) * np.where(
                        cleared_source, np.float32(0.0), value)
                else:
                    value = np.float32(weight) * value
                    # 2. the WALL CLEAR at the destination
                    value = np.where(cleared_source, np.float32(0.0), value)
            else:
                value = np.where(cleared_source, np.float32(0.0), value)
            # 3. the FAR product, applied last (:3311 runs after the clear)
            for axis in far_subset:
                value = np.float32(-phase[axis]) * value
            value = value.astype(np.float32)

            volume[destination_key] = np.where(
                own_source, value, volume[destination_key]).astype(np.float32)

            # update_E AT THE GHOST. The pole words and the inverse epsilon are the
            # DESTINATION's -- the array path evaluates displacement_minus_polarization
            # cell by cell over the whole volume, so the ghost cell subtracts its own P.
            pole_key = source_key if mutation == "poles_at_source" else destination_key
            eps_key = source_key if mutation == "inv_eps_at_source" else destination_key
            ghost = np.array(value, copy=True, dtype=np.float32)
            for array in chain:
                ghost -= array[pole_key]
            ghost_product = (ghost * inv_eps[eps_key]).astype(np.float32)
            coefficient_key = (source_key if mutation == "far_coefficient_source"
                               and far_subset else destination_key)
            ghost_kps = np.broadcast_to(kps, shape)[coefficient_key]
            ghost_kms = np.broadcast_to(kms, shape)[coefficient_key]
            ghost_previous = auxiliary[destination_key]
            ghost_updated = (
                (electric[destination_key]
                 + (ghost_kps * ghost_product).astype(np.float32)).astype(np.float32)
                - (ghost_kms * ghost_previous).astype(np.float32)).astype(np.float32)
            electric[destination_key] = np.where(
                own_source, ghost_updated, electric[destination_key]).astype(np.float32)
            auxiliary[destination_key] = np.where(
                own_source, ghost_product,
                auxiliary[destination_key]).astype(np.float32)
            written[destination_key] = (written[destination_key]
                                        + own_source.astype(np.int32))

        out[name] = volume
        out[component] = electric
        out["f_w_" + component] = auxiliary
        writes[name] = int(np.count_nonzero(written != 1))
    return out, writes


# ---------------------------------------------------------------------------
# The measurement
# ---------------------------------------------------------------------------

def measure(case: Dict[str, Any], steps: int) -> Dict[str, Any]:
    """One configuration: the array path's five passes, then the closed form and every
    armed mutation against it, over ``steps`` complete seams."""
    started = time.time()
    fields, grid, pml = build(case)

    plan = real.fused_electric_pair_fills(grid)
    near, reflect = plan["near"], plan["reflect"]
    phase = tuple(np.float32(value) for value in plan["phase"])
    from meep_gpu.cuda_kernels.in_seam_coverage import zero_metal_axes  # noqa: PLC0415
    walls = zero_metal_axes(grid)
    shape = tuple(int(n) for n in grid.shape)
    coefficients = {letter: stepping._constitutive_coefficients(
        pml, letter, half_integer=True) for letter in ("x", "y", "z")}
    poles = pole_chain(fields)
    arity = tuple(len(poles[component]) for component in _ELECTRIC)
    live = bool(any(np.any(array != 0) for chain in poles.values()
                    for array in chain))

    result: Dict[str, Any] = {
        "label": case["label"], "why": case["why"], "shape": list(shape),
        "arity": list(arity), "poles_are_live": live,
        "near": list(near), "reflect": list(reflect),
        "phase": [float(v) for v in plan["phase"]],
        "walls": [bool(v) for v in walls],
        "fills_live": bool(any(near) or any(row >= 0 for row in reflect)),
        "steps": steps,
    }
    if not live:
        # A FIXTURE THAT CANNOT ARM A NULL IS VACUOUS AND FAILS (rule 6).
        result["fixture_error"] = ("the pole volumes are all zero, so every pole "
                                   "mutation is a null and this case measures nothing")
        result["elapsed_s"] = round(time.time() - started, 3)
        return result

    engines = {label: build(case)[0] for label in ("shipped",) + MUTATIONS}
    reference = fields
    differing = {label: 0 for label in engines}
    races = 0
    for _step in range(steps):
        # --- the array path's five passes, in the driver's order -------------------
        stepping.step_D(reference, pml)
        pre = {name: np.array(getattr(reference, name), copy=True)
               for name in _TARGETS}
        stepping.fill_symmetry_bc_D(reference)
        stepping.zero_metal_D(reference)
        stepping.fill_folded_far_ghosts_D(reference)
        stepping.update_E(reference, pml)
        stepping.update_P(reference, pml)

        for label, engine in engines.items():
            # THE CURL IS THE CERTIFIED PASS, not a second transcription: what this
            # probe is about is the composition between step_D and update_E.
            stepping.step_D(engine, pml)
            engine_pre = {name: np.array(getattr(engine, name), copy=True)
                          for name in _TARGETS}
            state = {name: np.array(getattr(engine, name), copy=True)
                     for name in _ELECTRIC + tuple("f_w_" + c for c in _ELECTRIC)}
            for component in _ELECTRIC:
                state["inv_eps_" + component] = engine.inverse_epsilon_for(component)
            model, writes = closed_form(
                engine_pre, state, pole_chain(engine), coefficients, shape,
                near, reflect, phase, walls,
                mutation="" if label == "shipped" else label)
            for name, array in model.items():
                getattr(engine, name)[...] = array
            if label == "shipped":
                races += sum(writes.values())
            stepping.update_P(engine, pml)
            differing[label] += sum(
                int(np.count_nonzero(words(getattr(engine, name))
                                     != words(getattr(reference, name))))
                for name in _COMPARED)

    result["shipped"] = {"differing_words": differing["shipped"],
                         "cells_not_written_exactly_once": races}
    nulls: List[str] = []
    for label in MUTATIONS:
        diverged = differing[label] > 0
        result[label] = {"differing_words": differing[label],
                         "diverged": diverged,
                         "predicted_null": None if diverged else _NULL_REASONS[label]}
        if not diverged:
            nulls.append(label)
    result["mutations_that_diverge"] = len(MUTATIONS) - len(nulls)
    result["nulls"] = nulls
    result["case_is_vacuous"] = len(nulls) == len(MUTATIONS)
    result["elapsed_s"] = round(time.time() - started, 3)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True,
                        help="DIRECTORY to write cuda_dispersive_electric_pair.json")
    parser.add_argument("--steps", type=int, default=3,
                        help="complete seams per configuration (default 3)")
    arguments = parser.parse_args()
    out_dir = arguments.out
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = out_dir / "cuda_dispersive_electric_pair.json"

    print("the dispersive D/E pair's closed form against the driver's own five passes",
          flush=True)
    print(f"  probe   : {Path(__file__).name}", flush=True)
    print(f"  artifact: {artifact}", flush=True)
    print(f"  cases   : {len(CASES)} x {arguments.steps} complete steps", flush=True)

    rows: List[Dict[str, Any]] = []
    failed = 0
    vacuous = 0
    armed = {label: 0 for label in MUTATIONS}
    for number, case in enumerate(CASES, start=1):
        row = measure(case, arguments.steps)
        rows.append(row)
        if "fixture_error" in row:
            failed += 1
            print(f"  case {number}/{len(CASES)} {case['label']}: FIXTURE ERROR "
                  f"{row['fixture_error']}", flush=True)
        else:
            shipped = row["shipped"]["differing_words"]
            race = row["shipped"]["cells_not_written_exactly_once"]
            if shipped or race:
                failed += 1
            if row["case_is_vacuous"]:
                vacuous += 1
            for label in MUTATIONS:
                armed[label] += int(row[label]["diverged"])
            verdict = "OK" if not shipped and not race else "DIFFERS"
            print(f"  case {number}/{len(CASES)} {case['label']}: shipped={shipped} "
                  f"words races={race} arity={row['arity']} -> {verdict}, "
                  f"{row['mutations_that_diverge']}/{len(MUTATIONS)} mutations armed "
                  f"({row['elapsed_s']} s)", flush=True)
        unarmed = [label for label in MUTATIONS if armed[label] == 0]
        artifact.write_text(json.dumps({
            "probe": "cuda_dispersive_electric_pair",
            "subject": "meep_gpu/cuda_kernels/dispersive_fused_electric_pair.py",
            "claim": ("one launch per cell performs step_D, the two mirror fills, "
                      "zero_metal_D and the pole-aware update_E, with the ordered "
                      "D - sum P chain evaluated at the cell each constitutive_apply "
                      "writes -- the DESTINATION's poles and inverse epsilon at a "
                      "ghost, not the source thread's"),
            "passes": ["step_D (driver.py:3302)",
                       "fill_symmetry_bc_D (driver.py:3309)",
                       "zero_metal_D (driver.py:3310)",
                       "fill_folded_far_ghosts_D (driver.py:3311)",
                       "update_E (driver.py:3313)"],
            "compared_volumes": list(_COMPARED),
            "comparison": "raw uint32 words per complete step, never allclose",
            "mutations": list(MUTATIONS),
            "siblings_that_refuse_this_shape": [
                "metal_kernels/fused_dispersive_pair.py refuses a mirror plane "
                "through dispersive_update_e's own clause",
                "triton_kernels/dispersive_fused_pair.py refuses it through "
                "dispersive_update_e.dispersive_constitutive_coverage"],
            "complete": len(rows) == len(CASES),
            "cases": rows,
            "cases_measured": len(rows),
            "cases_differing": failed,
            "cases_vacuous": vacuous,
            "mutations_armed_somewhere": {label: armed[label] for label in MUTATIONS},
            "mutations_never_armed": unarmed,
            "passed": failed == 0 and not unarmed,
        }, indent=1) + "\n", encoding="utf-8")

    unarmed = [label for label in MUTATIONS if armed[label] == 0]
    print("", flush=True)
    print(f"  {len(rows)} configurations, {failed} differing from the array path, "
          f"{vacuous} vacuous", flush=True)
    for label in MUTATIONS:
        print(f"    {label}: diverged on {armed[label]}/{len(rows)} configurations",
              flush=True)
    if unarmed:
        print(f"  NEVER ARMED: {', '.join(unarmed)} -- the fixture set cannot "
              f"discriminate them and this sweep does not establish them", flush=True)
    print(f"  wrote {artifact}", flush=True)
    return 0 if (failed == 0 and not unarmed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
