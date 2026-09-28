"""Can ONE product own ``step_D`` -> ``update_E`` -> ``update_P``, and where must the
deposit bracket sit inside it?

THE CLAIM THIS PROBE STATES AND MEASURES, off device, before a line of kernel is
written. A single product occupying THREE driver slots reproduces the driver's own
pass order byte for byte when, and only when, its work is grouped like this:

    step_D consult    save(deposit cells + state)                 [LeadingRepairPlan]
                      ONE fused launch spanning
                          step_D (driver.py:3316)
                          -> fill_symmetry_bc_D (:3326)
                          -> zero_metal_D (:3327)
                          -> fill_folded_far_ghosts_D (:3330)
                          -> update_E (:3332)
                      on an UNINJECTED displacement
    (the driver)      electric inject (:3319/:3322), then the three passes again
    update_E consult  apply(the repair)                           [TrailingRepairPlan]
    update_P consult  THREE launches, one per component, every driving state of that
                      component fused into it, the HOST rotating P/P_prev/scratch
                      after each launch returns

and that TWO other groupings a reader would reach for first do NOT:

  * advancing P inside the fused launch (the "one mega-launch" design) computes the
    recurrence from a PRE-INJECTION drive;
  * advancing P before the repair -- at any point before the ``update_E`` consult --
    makes ``deposit_repair.apply`` read P^(n+1), because both of its paths recompute
    the constitutive product through ``Fields.displacement_minus_polarization``
    (deposit_repair.py:781-782 and :922-923), which subtracts ``state.P[c]`` as it
    stands (fields.py:1096-1105).

BOTH are silent: they compute, they converge, and they are wrong only at the deposit
cells. That is why they are ARMED HERE rather than reasoned about.

WHAT IS AND IS NOT MODELLED. The fused ``step_D``->``update_E`` launch is modelled by
its OWN certified equivalence -- the four driver passes run back to back on an
uninjected field -- which is what
``cuda_dispersive_fused_electric_pair`` / ``cuda_no_pml_dispersive_fused_electric_pair``
declare in ``REPLACES`` and what their released gates measured
(``certification.json``, both 2026-09-02 blocks). This probe does not re-derive the
ghost carry or the wall carry; it measures the COMPOSITION those two products and
``cuda_(no_pml_)fused_polarization_pair`` are being welded into, which no gate on this
track has asked about, because no product has spanned three slots.

The ``update_P`` half IS modelled line by line: ``dispersion.PolarizationState.update``
(dispersion.py:679-691) restricted to one component, run COMPONENT-MAJOR with every
driving state of a component in one group, against an array path that runs it
STATE-MAJOR (stepping.py:1427-1428). That the two orders agree is
``fused_polarization_pair``'s own argument -- rotations of different states commute,
each state owns its seven buffers, and within a state the x, y, z sequence is
preserved -- and here it is a measurement rather than an argument.

WHAT IS COMPARED. Raw uint32 WORDS of every stored volume the span writes -- D, E and
``f_w_E`` where the arm allocates it -- AND every ``P`` and ``P_prev`` buffer of every
registered susceptibility, over COMPLETE driver steps, against the driver's own order
run from identical state. Never ``allclose``: ``-0.0 == 0.0`` lies.

THE POLE VOLUMES ARE SEEDED NON-ZERO. Zero is a fixed point of both ``s = s - P`` and
the recurrence's subtraction chain, so a zero fixture turns every pole mutation into a
silent null and a green sweep would have measured nothing. ``poles_are_live`` is
asserted per case, and a case that cannot arm its nulls is recorded VACUOUS.

BOTH ARMS ARE SWEPT, with a deposit-carrying configuration on each:

  pml      an ACTIVE layer, ``update_E``'s split-field recurrence, ``f_w_E`` allocated,
           ``REPAIR_PATHS = (SPLIT_FIELD_PATH,)``   -- the 7-row cell
  no_pml   an INACTIVE layer, ``update_E``'s pure overwrite (stepping.py:1019-1022), no
           ``f_w`` at all, ``REPAIR_PATHS = (PLAIN_PATH,)``  -- the 3-row cell

Progress is one flushed line per configuration and the artifact is rewritten after
each, so an interrupted run keeps every row that landed (the progress-reporting rule).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_API = _HERE.parents[1]
if str(_REPO_API) not in sys.path:
    sys.path.insert(0, str(_REPO_API))

from meep_gpu import deposit_repair, stepping  # noqa: E402
from meep_gpu.dispersion import PolarizationState, Susceptibility  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid, Mirror  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.sources import GaussianPulsedSource  # noqa: E402

_TARGETS: Tuple[str, ...] = ("Dx", "Dy", "Dz")
_ELECTRIC: Tuple[str, ...] = ("Ex", "Ey", "Ez")

#: Everything the fixture seeds. ``fu_D*`` and ``f_w_E*`` exist on the PML arm only.
_SEEDED: Tuple[str, ...] = (
    "Bx", "By", "Bz", "Hx", "Hy", "Hz",
    "Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz",
    "Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez")

SEED = 20260902

#: The orderings this probe arms. Every one is a grouping a three-slot weld could
#: plausibly be built with; each must DIVERGE, or be recorded null with its reason.
MUTATIONS: Tuple[str, ...] = (
    "p_inside_the_launch", "repair_after_p", "unbracketed", "repair_before_fill",
    "stale_pole_pointers", "rotate_before_launch", "shared_scratch_all_components",
    "state_order_reversed", "component_order_reversed", "drive_from_stored_E")

#: Why each ordering can be UNREACHABLE on a configuration. Consulted when it does not
#: diverge, so a null carries its reason instead of being counted as agreement.
_NULL_REASONS: Dict[str, str] = {
    "p_inside_the_launch":
        "the source deposits no word into D on this configuration, so the drive the "
        "fused register carries is already the injected one",
    "repair_after_p":
        "no component carries a pole, so displacement_minus_polarization returns D "
        "itself and advancing P first changes nothing the repair reads",
    "unbracketed": "the source deposits no word into D on this configuration",
    "repair_before_fill":
        "no post-injection fill images this deposit: either the grid carries no mirror "
        "plane, or the DEPOSITED component's Yee shift on the folded axis is not the "
        "one that fill takes (the near fill images shift 0, the far ghost shift 1, "
        "stepping.py:1426-1452 and :1489-1533), or the deposit index is not that "
        "fill's source row",
    "stale_pole_pointers":
        "no state drives more than one component, so no rotation happens between the "
        "component launches for a cached view to go stale across",
    "rotate_before_launch": "no state drives any component",
    "shared_scratch_all_components":
        "no state drives more than one component, so one scratch is enough and the "
        "buffer count that refuses a whole-state launch is not reached",
    "state_order_reversed":
        "PREDICTED: each state owns its own P/P_prev/scratch (dispersion.py:645-651) "
        "and the recurrence reads only that state's buffers plus the shared drive, so "
        "the order of states WITHIN one component is not observable in any compared "
        "volume. Recorded rather than skipped: a confirmed null is the measurement "
        "that the state loop may be fused into one launch",
    "component_order_reversed":
        "PREDICTED: the recurrence for one component reads only that component's P and "
        "P_prev plus the shared drive, and writes only that component's slots, so the "
        "component order changes WHICH allocation carries the retired history and "
        "nothing a compared volume holds. Recorded rather than skipped: a confirmed "
        "null is the measurement that the three component launches may be issued in "
        "any order, which x, y, z is then chosen to match rather than required to",
    "drive_from_stored_E":
        "the layer is inactive, so Fields.drive_field returns the stored E itself "
        "(fields.py:1160-1162) and the substitution is the identity",
}


# ---------------------------------------------------------------------------
# The configurations
# ---------------------------------------------------------------------------
#
# THE FIRST FOUR ARE THE CORPUS SHAPES, read off the stamped census
# (results/cuda_predicate_coverage_2026-09-02_stamped): the ten REAL rows the three-slot
# weld exists for, which are exactly the rows where a D->E product and an E->P product
# trade the update_E slot one for one.
CASES: Tuple[Dict[str, Any], ...] = (
    {"label": "corpus_stochastic_emitter", "arm": "pml", "cell": (9., 10., 11.),
     "boundaries": ("periodic",) * 3, "symmetry": (), "pml": 2, "arity": (6, 6, 6),
     "conductivity": None,
     "why": "the three stochastic_emitter* rows: ACTIVE layer, all periodic, no "
            "symmetry, six susceptibilities driving every component, a D source"},
    {"label": "corpus_loaddump_2d", "arm": "pml", "cell": (9., 10., 11.),
     "boundaries": ("metallic", "metallic", "periodic"), "symmetry": (("Y", 1),),
     "pml": 2, "arity": (5, 5, 5), "conductivity": None,
     # ON THE NEAR FILL'S SOURCE ROW: y = 1.0 resolves to stored iy = 2, which IS
     # ``stepping.MIRROR_SOURCE_INDEX`` -- the row ``fill_symmetry_bc_D`` reads from.
     # A deposit anywhere else is invisible to ``repair_before_fill``, which is the
     # ordering that shows why the repair sits AFTER the three passes and not before.
     "source_component": "Ez", "source_center": (0.0, 1.0, 0.0),
     "why": "the four TestLoadDump.*_2d rows: ACTIVE layer, metallic x, mirror-folded "
            "y, periodic z, five susceptibilities -- the wall AND the near fill are "
            "live inside the seam, so the repair's cell closure is exercised"},
    {"label": "corpus_absorber_1d", "arm": "no_pml", "cell": (0., 0., 8.),
     "boundaries": ("periodic", "periodic", "metallic"), "symmetry": (), "pml": 0,
     "arity": (5, 5, 5), "conductivity": 0.4,
     "why": "absorber-1d.py / TestAbsorber.test_absorber: INACTIVE layer, metallic z, "
            "five susceptibilities, conductive on every D component, a D source"},
    {"label": "corpus_material_dispersion", "arm": "no_pml", "cell": (0., 0., 0.1),
     "boundaries": ("periodic",) * 3, "symmetry": (), "pml": 0, "arity": (2, 2, 2),
     "conductivity": None,
     "why": "material-dispersion.py: INACTIVE layer, a SINGLE cell, all periodic, two "
            "susceptibilities, lossless"},
    # The rest exist to arm orderings the corpus shapes cannot.
    {"label": "pml_fold_y_odd_wall_z", "arm": "pml", "cell": (9., 10., 11.),
     "boundaries": ("periodic", "periodic", "metallic"), "symmetry": (("Y", -1),),
     "pml": 2, "arity": (3, 2, 2), "conductivity": None,
     # AN ODD PLANE MAKES Ex, Ez, Hy ODD and leaves Ey EVEN, so the source drives Ey;
     # y = 1.5 resolves to stored iy = 2 on this grid, the near fill's source row.
     "source_component": "Ey", "source_center": (0.0, 1.5, 0.0),
     "why": "an ODD mirror phase with a wall on the third axis: the deepest cell "
            "closure repair_before_fill can be asked for"},
    {"label": "pml_single_state_two_poles", "arm": "pml", "cell": (7., 8., 9.),
     "boundaries": ("periodic",) * 3, "symmetry": (), "pml": 2, "arity": (2, 0, 1),
     "conductivity": None,
     "why": "ONE component undriven and two states of different reach: arms "
            "stale_pole_pointers and shared_scratch_all_components on a state that "
            "drives two components while another drives one"},
    {"label": "pml_zero_arity", "arm": "pml", "cell": (7., 8., 9.),
     "boundaries": ("periodic",) * 3, "symmetry": (), "pml": 2, "arity": (0, 0, 0),
     "conductivity": None,
     "why": "NO susceptibility at all: every pole ordering must go NULL by "
            "construction, and the composition must still be byte-identical -- the "
            "degenerate case where the weld reduces to the shipped D->E pair"},
)


# ---------------------------------------------------------------------------
# The fixture
# ---------------------------------------------------------------------------

def build(case: Dict[str, Any]):
    """One seeded NumPy engine on the case's arm: ``(fields, grid, pml)``.

    ``pml=0`` builds a layer that absorbs on no face, so ``stepping._pml_is_active`` is
    False, ``update_E`` takes the pure overwrite at stepping.py:1019-1022 and no ``f_w``
    is allocated -- the storage the PLAIN repair inverts and the split-field one
    refuses by name.
    """
    rng = np.random.default_rng(SEED)
    cell = tuple(float(v) for v in case["cell"])
    dimensions = max(1, sum(1 for extent in cell if extent > 0.0))
    grid = Grid(resolution=(1.0 if dimensions == 3 else 10.0), cell_size=cell,
                dimensions=dimensions,
                boundaries=tuple(kind if extent > 0.0 else "periodic"
                                 for kind, extent in zip(case["boundaries"], cell)),
                symmetry=tuple(Mirror(axis, int(phase))
                               for axis, phase in case["symmetry"]),
                xp=np, courant=0.5)
    fields = Fields(grid=grid)
    arm = case["arm"]
    if arm == "pml":
        fields.enable_pml_storage()
        folded = {"XYZ".index(axis) for axis, _phase in case["symmetry"]}
        thickness = tuple((0, case["pml"]) if axis in folded
                          else (case["pml"], case["pml"]) for axis in range(3))
        pml = PML(grid=grid, thickness=thickness)
    else:
        pml = PML(grid=grid, thickness=0)

    shape = tuple(int(n) for n in grid.shape)
    epsilon, inverse = {}, {}
    for component in _ELECTRIC:
        values = rng.uniform(1.2, 3.4, size=shape).astype(np.float32)
        epsilon[component] = values
        inverse[component] = (np.float32(1.0) / values).astype(np.float32)
    fields.set_epsilon_volumes(epsilon, inverse)

    arity = tuple(int(value) for value in case["arity"])
    kinds = ("lorentzian", "drude")
    for index in range(max(arity) if arity else 0):
        term = Susceptibility(frequency=0.20 + 0.03 * index, gamma=0.008,
                              kind=kinds[index % 2])
        sigma = {name: (0.35 + 0.04 * index if index < arity[axis] else 0.0)
                 for axis, name in enumerate(_ELECTRIC)}
        fields.polarizations.append(
            PolarizationState(term, sigma, grid, fields._field_dtype()))
    if case["conductivity"] is not None:
        volume = np.full(shape, float(case["conductivity"]), dtype=np.float32)
        volume *= np.linspace(0.5, 1.5, shape[0], dtype=np.float32)[:, None, None]
        fields.set_d_conductivity(volume)
    if arm == "no_pml":
        fields.enable_field_storage()

    for name in _SEEDED:
        array = getattr(fields, name, None)
        if array is None:
            continue
        array[...] = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    # THE POLE VOLUMES ARE SEEDED NON-ZERO -- see the module docstring.
    for state in fields.polarizations:
        for store in (state.P, state.P_prev):
            for _component, array in store.items():
                array[...] = rng.uniform(-0.6, 0.6, size=shape).astype(np.float32)
    return fields, grid, pml


def words(array: Any) -> np.ndarray:
    """Raw uint32 WORDS. Byte compares, never ``allclose``."""
    return np.ascontiguousarray(array, dtype=np.float32).ravel().view(np.uint32)


def compared_volumes(fields: Any) -> List[Tuple[str, Any]]:
    """Every stored volume the span writes, plus every pole buffer, in a stable order."""
    out: List[Tuple[str, Any]] = []
    for name in _TARGETS + _ELECTRIC + ("f_w_Ex", "f_w_Ey", "f_w_Ez"):
        array = getattr(fields, name, None)
        if array is not None:
            out.append((name, array))
    for index, state in enumerate(fields.polarizations):
        for attribute in ("P", "P_prev"):
            store = getattr(state, attribute, None) or {}
            for component in _ELECTRIC:
                array = store.get(component)
                if array is not None:
                    out.append((f"{attribute}[{index}][{component}]", array))
    return out


def differing_words(left: Any, right: Any) -> int:
    total = 0
    for (name, a), (other, b) in zip(compared_volumes(left), compared_volumes(right)):
        if name != other:
            raise AssertionError(
                f"the two engines expose different volumes ({name!r} vs {other!r}); "
                f"the comparison would be reading past each other")
        total += int(np.count_nonzero(words(a) != words(b)))
    return total


# ---------------------------------------------------------------------------
# THE FUSED D->E LAUNCH, by its own certified equivalence
# ---------------------------------------------------------------------------

def fused_step_D_to_update_E(fields: Any, pml: Any) -> None:
    """The five driver passes ONE launch performs, on an UNINJECTED displacement.

    This is ``dispersive_fused_electric_pair.REPLACES`` in driver order -- and, with
    the two fills inert on an unfolded grid, ``no_pml_dispersive_fused_electric_pair``'s
    three. Both are released products whose gates measured this equivalence against a
    device (``certification.json``, the two 2026-09-02 blocks); nothing here re-derives
    it, and what this probe measures is the composition around it.
    """
    stepping.step_D(fields, pml)
    stepping.fill_symmetry_bc_D(fields)
    stepping.zero_metal_D(fields)
    stepping.fill_folded_far_ghosts_D(fields)
    stepping.update_E(fields, pml)


# ---------------------------------------------------------------------------
# THE THIRD SLOT: update_P, component-major, every driving state fused per component
# ---------------------------------------------------------------------------

def driving_states(fields: Any, component: str) -> List[Any]:
    """The states driving one component, in ``fields.polarizations`` order.

    The SAME order ``Fields.displacement_minus_polarization`` subtracts in
    (fields.py:1099-1104) and ``stepping.update_P`` advances in, so the two halves of
    the weld cannot disagree about the bank.
    """
    return [state for state in fields.polarizations if state.drives(component)]


def advance_component(state: Any, component: str, drive: Any, mutation: str) -> None:
    """ONE state's recurrence for ONE component, then the rotation.

    ``dispersion.PolarizationState.update``'s own three lines and its own rotation
    (dispersion.py:679-691), restricted to one component. The grouping
    ``((p * c_now) + (c_prev * q)) + (c_drive * (s * w))`` is the array path's operand
    order, which is the arithmetic: float32 addition is not associative.
    """
    xp = state.grid.xp
    c_now, c_prev, c_drive = state._coefficients
    p = state.P[component]
    p_prev = state.P_prev[component]
    scratch = state._scratch
    if mutation == "rotate_before_launch":
        # The rotation applied BEFORE the result is written: the slots this launch
        # binds are then last step's, and the defect still computes.
        state.P[component] = scratch
        state.P_prev[component] = p
        state._scratch = p_prev
        p, p_prev, scratch = (state.P[component], state.P_prev[component],
                              state._scratch)
    xp.multiply(p, c_now, out=scratch)
    scratch += c_prev * p_prev
    scratch += c_drive * (state.sigma[component] * drive)
    state.P[component] = scratch
    state.P_prev[component] = p
    state._scratch = p_prev


def slot_update_P(fields: Any, mutation: str = "") -> int:
    """The third slot: THREE launches, one per component. Returns the launch count.

    COMPONENT-MAJOR, with every driving state of a component in ONE group -- which is
    what makes the group a single kernel launch on device -- and the HOST rotating
    after each group, exactly as ``ade_kernels.update_P_fused_pml_real`` rotates
    between its own per-(state, component) launches.

    EVERY SLOT IS RE-READ INSIDE THE LOOP, after the previous component's rotation.
    ``stale_pole_pointers`` is what happens to a launcher that does not.
    """
    launches = 0
    cached: Dict[int, Tuple[Any, Any, Any]] = {}
    order = (tuple(reversed(_ELECTRIC)) if mutation == "component_order_reversed"
             else _ELECTRIC)
    for component in order:
        states = driving_states(fields, component)
        if mutation == "state_order_reversed":
            states = list(reversed(states))
        if not states:
            continue
        drive = (getattr(fields, component) if mutation == "drive_from_stored_E"
                 else fields.drive_field(component))
        if mutation == "shared_scratch_all_components":
            # THE BUFFER-COUNT REFUSAL, ARMED: all of a state's components advanced
            # into the ONE scratch it owns without rotating between them. The second
            # component overwrites the first, and the result still computes.
            for state in states:
                xp = state.grid.xp
                c_now, c_prev, c_drive = state._coefficients
                scratch = state._scratch
                xp.multiply(state.P[component], c_now, out=scratch)
                scratch += c_prev * state.P_prev[component]
                scratch += c_drive * (state.sigma[component] * drive)
            for state in states:
                state.P[component] = state.P[component] + np.float32(0.0)
            launches += 1
            continue
        for state in states:
            if mutation == "stale_pole_pointers":
                key = id(state)
                if key not in cached:
                    cached[key] = (dict(state.P), dict(state.P_prev), state._scratch)
                stale_P, stale_prev, stale_scratch = cached[key]
                state.P.update(stale_P)
                state.P_prev.update(stale_prev)
                state._scratch = stale_scratch
            advance_component(state, component, drive, mutation)
        launches += 1
    return launches


# ---------------------------------------------------------------------------
# The two orders
# ---------------------------------------------------------------------------

def driver_step(fields: Any, pml: Any, sources: Sequence[Any], when: float) -> None:
    """The driver's own electric seam, verbatim (driver.py:3315-3334), no fast path."""
    stepping.step_D(fields, pml)
    for source in sources:
        source.inject(fields, when)
    stepping.fill_symmetry_bc_D(fields)
    stepping.zero_metal_D(fields)
    stepping.fill_folded_far_ghosts_D(fields)
    stepping.update_E(fields, pml)
    stepping.update_P(fields, pml)


def weld_step(fields: Any, pml: Any, sources: Sequence[Any], when: float,
              paths: Sequence[str], mutation: str = "") -> Optional[str]:
    """The three-slot composition. Returns a refusal reason, or ``None``.

    The three consults are exactly where the driver puts them, and the ORDER OF THE
    THREE GROUPS IS THE WHOLE CLAIM: the launch runs uninjected, the repair runs after
    the driver's inject/fill/clear, and the polarization advance runs after the repair.
    """
    try:
        saved = deposit_repair.save(fields, tuple(sources), "D", pml, paths=paths)
    except Exception as error:  # a refusal is the measurement, not a failure
        return f"{type(error).__name__}: {error}"

    # --- the step_D consult: ONE fused launch, on an uninjected displacement --------
    fused_step_D_to_update_E(fields, pml)
    if mutation == "p_inside_the_launch":
        # THE MEGA-LAUNCH DESIGN: the recurrence consumes the register the constitutive
        # half just produced -- which is the PRE-INJECTION drive at every deposit cell.
        slot_update_P(fields)

    # --- the driver, between the consults ------------------------------------------
    for source in sources:
        source.inject(fields, when)
    if mutation == "repair_before_fill":
        try:
            deposit_repair.apply(fields, pml, tuple(sources), "D", saved)
        except Exception as error:
            return f"{type(error).__name__}: {error}"
    stepping.fill_symmetry_bc_D(fields)
    stepping.zero_metal_D(fields)
    stepping.fill_folded_far_ghosts_D(fields)

    if mutation == "repair_after_p":
        # THE ORDER THAT MAKES THE REPAIR READ P^(n+1): both repair paths recompute the
        # constitutive product through displacement_minus_polarization.
        slot_update_P(fields)

    # --- the update_E consult: the repair ------------------------------------------
    if mutation not in ("unbracketed", "repair_before_fill"):
        try:
            deposit_repair.apply(fields, pml, tuple(sources), "D", saved)
        except Exception as error:
            return f"{type(error).__name__}: {error}"

    # --- the update_P consult: three launches, host rotation between ---------------
    if mutation not in ("p_inside_the_launch", "repair_after_p"):
        slot_update_P(fields, mutation)
    return None


# ---------------------------------------------------------------------------
# The measurement
# ---------------------------------------------------------------------------

def _source_for(grid: Any, case: Dict[str, Any]) -> Any:
    """One electric point source, at the component and centre this case declares.

    THE CENTRE IS A CASE FACT rather than a default, because on a folded grid WHERE
    the deposit lands decides whether ``repair_before_fill`` can be armed at all: only
    a deposit on stored ``stepping.MIRROR_SOURCE_INDEX`` of the folded axis is imaged
    by ``fill_symmetry_bc_D``, and only an imaged deposit can show that the repair
    belongs after the three passes.
    """
    return GaussianPulsedSource(
        grid=grid, component=case.get("source_component", "Ez"),
        center=tuple(case.get("source_center", (0.0, 0.0, 0.0))),
        size=(0.0, 0.0, 0.0), frequency=0.35, fwidth=0.2, amplitude=1.0)


def _step_times(source: Any, grid: Any, steps: int) -> List[float]:
    """The times the run injects at, CENTRED ON THE SOURCE'S OWN PEAK.

    A driver's absolute clock is irrelevant here and a naive ``(step + 0.5) * dt``
    is actively misleading: on a fine grid it samples the Gaussian's far tail, the
    deposit underflows to nothing, and every deposit ordering below goes null while
    reporting agreement. The times are the source's peak plus the step offsets, so the
    deposit is the largest this source can make and the nulls that remain are
    structural rather than an artefact of when the probe looked.
    """
    dt = float(grid.dt)
    peak = float(getattr(source, "_peak_time", 0.0))
    return [peak + (step - (steps - 1) / 2.0) * dt for step in range(steps)]


def _paths_for(arm: str, wrong: bool = False) -> Tuple[str, ...]:
    correct = ((deposit_repair.SPLIT_FIELD_PATH,) if arm == "pml"
               else (deposit_repair.PLAIN_PATH,))
    other = ((deposit_repair.PLAIN_PATH,) if arm == "pml"
             else (deposit_repair.SPLIT_FIELD_PATH,))
    return other if wrong else correct


def measure(case: Dict[str, Any], steps: int) -> Dict[str, Any]:
    """One configuration: the driver's order, then the weld and every armed ordering."""
    started = time.time()
    arm = case["arm"]
    reference, grid, pml = build(case)
    shape = tuple(int(n) for n in grid.shape)
    live = bool(any(np.any(array != 0) for state in reference.polarizations
                    for array in state.P.values()))
    active = bool(stepping._pml_is_active(pml))

    result: Dict[str, Any] = {
        "label": case["label"], "arm": arm, "why": case["why"],
        "shape": list(shape), "arity": list(case["arity"]),
        "n_states": len(reference.polarizations),
        "poles_are_live": live,
        "pml_is_active": active,
        "has_f_w": getattr(reference, "f_w_Ex", None) is not None,
        "stores_E": bool(reference.stores_E),
        "has_symmetry": bool(grid.has_symmetry()),
        "steps": steps,
        "repair_paths": list(_paths_for(arm)),
    }
    # THE ARM THIS CASE CLAIMS TO BE ON, CHECKED RATHER THAN ASSUMED: an f_w on the
    # plain arm, or none on the split-field one, would measure a different recurrence.
    expected_fw = (arm == "pml")
    if result["has_f_w"] != expected_fw or active != expected_fw:
        result["fixture_error"] = (
            f"case declares arm={arm!r} but pml_is_active={active} and "
            f"has_f_w={result['has_f_w']}")
        result["elapsed_s"] = round(time.time() - started, 3)
        return result

    # DOES THE FIXTURE ACTUALLY DEPOSIT, AT THE TIMES THE RUN USES? Every deposit
    # ordering below is a null without it, so it is measured at the run's OWN times
    # rather than at a convenient one -- a guard that looked at a different time would
    # certify an arming the run never had.
    times = _step_times(_source_for(grid, case), grid, steps)
    result["inject_times"] = [round(t, 6) for t in times]
    probe_fields, probe_grid, _p = build(case)
    probe_source = _source_for(probe_grid, case)
    result["deposit_index"] = [
        None if getattr(probe_source, name, None) is None
        else int(np.ravel(getattr(probe_source, name))[0])
        for name in ("_point_ix", "_point_iy", "_point_iz")]
    result["mirror_source_index"] = int(stepping.MIRROR_SOURCE_INDEX)
    deposited = 0
    for when in times:
        before = {target: np.array(getattr(probe_fields, target), copy=True)
                  for target in _TARGETS}
        probe_source.inject(probe_fields, when)
        deposited = max(deposited, sum(
            int(np.count_nonzero(words(getattr(probe_fields, target))
                                 != words(before[target])))
            for target in _TARGETS))
    result["deposited_words"] = deposited

    engines = {label: build(case)[0] for label in ("shipped",) + MUTATIONS}
    engine_sources = {label: [_source_for(grid, case)] for label in engines}
    reference_sources = [_source_for(grid, case)]
    refusals: Dict[str, str] = {}
    for when in times:
        driver_step(reference, pml, reference_sources, when)
        for label, engine in engines.items():
            refusal = weld_step(engine, pml, engine_sources[label], when,
                                _paths_for(arm),
                                "" if label == "shipped" else label)
            if refusal is not None:
                refusals.setdefault(label, refusal)

    result["shipped"] = {"differing_words": differing_words(engines["shipped"],
                                                            reference),
                         "refused": refusals.get("shipped")}
    nulls: List[str] = []
    for label in MUTATIONS:
        count = differing_words(engines[label], reference)
        diverged = count > 0 or refusals.get(label) is not None
        result[label] = {"differing_words": count,
                         "diverged": diverged,
                         "refused": refusals.get(label),
                         "predicted_null": None if diverged
                         else _NULL_REASONS[label]}
        if not diverged:
            nulls.append(label)
    result["orderings_that_diverge"] = len(MUTATIONS) - len(nulls)
    result["nulls"] = nulls
    result["case_is_vacuous"] = len(nulls) == len(MUTATIONS)

    # THE WRONG REPAIR DECLARATION, which must be REFUSED BY NAME rather than
    # mis-repairing: deposit_repair.repairable refuses any recurrence not among the
    # paths its caller declares (deposit_repair.py:630-716).
    wrong_engine, _g, _p = build(case)
    wrong_refusal = weld_step(wrong_engine, pml, [_source_for(grid, case)], times[0],
                              _paths_for(arm, wrong=True))
    result["wrong_repair_path"] = {"refused": wrong_refusal}

    # THE LAUNCH COUNT THIS COMPOSITION PREDICTS, counted rather than asserted.
    counter, _g, _p = build(case)
    result["update_P_launches_weld"] = slot_update_P(counter)
    result["update_P_launches_certified_singles"] = sum(
        len(state.driven()) for state in counter.polarizations)
    result["elapsed_s"] = round(time.time() - started, 3)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True,
                        help="DIRECTORY to write cuda_three_slot_weld.json")
    parser.add_argument("--steps", type=int, default=3,
                        help="complete driver steps per configuration")
    arguments = parser.parse_args()
    arguments.out.mkdir(parents=True, exist_ok=True)
    artifact = arguments.out / "cuda_three_slot_weld.json"

    rows: List[Dict[str, Any]] = []
    started = time.time()
    for index, case in enumerate(CASES, start=1):
        row = measure(case, arguments.steps)
        rows.append(row)
        print(f"case {index}/{len(CASES)} {row['label']} [{row['arm']}]: "
              f"shipped_diff={row.get('shipped', {}).get('differing_words', 'n/a')} "
              f"diverged={row.get('orderings_that_diverge', 0)}/{len(MUTATIONS)} "
              f"nulls={row.get('nulls', [])} "
              f"wrong_path_refused="
              f"{bool(row.get('wrong_repair_path', {}).get('refused'))} "
              f"P_launches={row.get('update_P_launches_weld')}"
              f"/{row.get('update_P_launches_certified_singles')} "
              f"({row['elapsed_s']} s)", flush=True)
        artifact.write_text(json.dumps({"rows": rows}, indent=1))

    identical = [r for r in rows if r.get("shipped", {}).get("differing_words") == 0
                 and not r.get("shipped", {}).get("refused")]
    vacuous = [r["label"] for r in rows if r.get("case_is_vacuous")]
    errors = [r["label"] for r in rows if r.get("fixture_error")]
    not_live = [r["label"] for r in rows
                if r.get("arity") != [0, 0, 0] and not r.get("poles_are_live")]
    wrong_refused = [r["label"] for r in rows
                     if r.get("wrong_repair_path", {}).get("refused")]
    summary = {
        "cases": len(rows),
        "identical": len(identical),
        "vacuous": vacuous,
        "fixture_errors": errors,
        "cases_without_live_poles": not_live,
        "wrong_repair_path_refused": len(wrong_refused),
        "orderings": list(MUTATIONS),
        "elapsed_s": round(time.time() - started, 3),
    }
    artifact.write_text(json.dumps({"rows": rows, "summary": summary}, indent=1))
    print(json.dumps(summary, indent=1), flush=True)
    ok = (len(identical) == len(rows) and not vacuous and not errors
          and not not_live and len(wrong_refused) == len(rows))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
