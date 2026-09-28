"""Does the NO-PML DISPERSIVE-STORE D/E pair's closed form reproduce the driver's seam?

THE CLAIM THIS PROBE STATES AND MEASURES. One launch per cell can perform

    step_D (driver.py:3306, no absorber, with a PER-COMPONENT conductive tail)
    -> zero_metal_D (:3310)
    -> plain stored-E dispersive update_E (:3313, stepping.py:1019-1022)

with the stepped displacement carried in a REGISTER between the three passes, and the
in-seam electric deposit carried across the launch by
:data:`..deposit_repair.PLAIN_PATH`.

WHY THE REGISTER IS THE WHOLE CLAIM. In the array path ``step_D`` writes D, then
``zero_metal_D`` writes zeros into D at the wall cells, then ``update_E`` reads D. The
fused kernel computes the curl into ``v``, clears ``v``, and then does BOTH the store
and the constitutive read from that one value. The two agree only if the clear
happens between the curl and both consumers -- which is what
``subtract_before_clear`` and ``store_before_clear`` below arm, one per consumer. Two
of this cell's three corpus rows are METALLIC on z, so the wall is live on the corpus
rather than only in a fixture.

WHAT MAKES THIS CELL'S CONSTITUTIVE HALF DIFFERENT FROM EVERY FUSED PAIR BEFORE IT.
The layer is INACTIVE, so ``update_E`` takes the PURE OVERWRITE at stepping.py:1022 --
``E[...] = (D - sum_n P_n) * inv_eps`` -- with no ``f_w``, no coefficient vector and
no previous value. That is not the split-field recurrence every shipped CUDA electric
pair inverts, and it is exactly why this product declares
``REPAIR_PATHS = (deposit_repair.PLAIN_PATH,)``: the split-field repair would write an
answer the step never computed. ``deposit_repair.repairable`` refuses by name any
configuration whose recurrence is not among the paths its caller declares, so the
declaration is load-bearing rather than descriptive, and the ``split_field_declared``
leg below MEASURES that refusal instead of asserting it.

THE TWO CURL ARMS ARE ONE EMITTER. ``conductive_kernels`` bakes the per-component
conductivity into three ``COND`` defines and takes the ``#else`` plain tail where a
component is lossless, so the arity ``(False, False, False)`` build IS the lossless
no-PML curl. The corpus drives both: ``absorber-1d.py`` and
``TestAbsorber.test_absorber`` are conductive on every D component, and
``material-dispersion.py`` is lossless. Both are swept here.

WHAT IS COMPARED. Raw uint32 WORDS of every stored volume the seam writes -- ``Dx/Dy/Dz``
and ``Ex/Ey/Ez`` -- AND every pole buffer (``P`` and ``P_prev`` of every registered
susceptibility), against the array path's own passes run from identical state, over
COMPLETE driver steps. Never ``allclose``: ``-0.0 == 0.0`` lies, and the sign of zero
is precisely the class this campaign has been bitten by.

THERE IS NO ``fu_D`` AND NO ``f_w_E`` ON THIS PATH and their absence is checked rather
than assumed: ``enable_field_storage`` allocates E WITHOUT ``f_w`` (the plain branch's
storage), and ``fu_D`` belongs to PML storage this configuration never enables. A case
that found either allocated would be measuring a different sub-step.

THE POLE VOLUMES ARE SEEDED NON-ZERO, and that is not a detail. Zero is a fixed point
of ``s = s - P``: a fixture with zero poles turns every pole mutation into a null and
would report a green sweep that measured nothing. ``poles_are_live`` is asserted per
case and a case that cannot arm the pole nulls is recorded VACUOUS rather than passed.

THE ARMED MUTATIONS, each a port a reasonable person would write:

  subtract_before_clear  the constitutive consumes the PRE-clear register
  store_before_clear     the store writes the PRE-clear register
  pole_order_reversed    the ordered chain is summed right to left
  poles_dropped          the chain is not formed at all (E = v * inv_eps)
  inv_eps_shared         all three components multiply component x's inverse epsilon
  clear_all_axes         the wall clears every axis rather than the off-diagonal rows

AND THE THREE DEPOSIT LEGS, which are what ``PLAIN_PATH`` is for:

  bracketed              save -> launch on an UNINJECTED D -> inject -> clear -> repair
  unbracketed            the same without the repair -- A NULL CONTROL THAT MUST DIVERGE
  split_field_declared   the same product declaring the WRONG repair -- must be REFUSED
                         by name rather than mis-repairing

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

from meep_gpu import deposit_repair, stepping  # noqa: E402
from meep_gpu.dispersion import PolarizationState, Susceptibility  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.sources import GaussianPulsedSource  # noqa: E402

_TARGETS: Tuple[str, ...] = ("Dx", "Dy", "Dz")
_ELECTRIC: Tuple[str, ...] = ("Ex", "Ey", "Ez")

#: What the comparison covers, per complete step. The pole buffers are added
#: separately because their names depend on how many susceptibilities are registered.
_COMPARED: Tuple[str, ...] = _TARGETS + _ELECTRIC

#: Volumes this configuration must NOT have allocated. Checked per case: their presence
#: would mean the fixture took the split-field branch and the probe measured the wrong
#: sub-step entirely.
_MUST_BE_ABSENT: Tuple[str, ...] = ("fu_Dx", "fu_Dy", "fu_Dz",
                                    "f_w_Ex", "f_w_Ey", "f_w_Ez")

SEED = 20260902

MUTATIONS: Tuple[str, ...] = (
    "subtract_before_clear", "store_before_clear", "pole_order_reversed",
    "poles_dropped", "inv_eps_shared", "clear_all_axes")

#: Why each mutation can be UNREACHABLE on a configuration. Consulted when it does not
#: diverge, so a null is recorded with its reason instead of counted as agreement.
_NULL_REASONS: Dict[str, str] = {
    "subtract_before_clear":
        "no metallic wall clears a cell where (v - sum P) * inv_eps is non-zero",
    "store_before_clear":
        "no metallic wall clears a cell whose pre-clear register is non-zero",
    "pole_order_reversed": "no component carries two or more poles",
    "poles_dropped": "no component carries a pole with a non-zero volume",
    "inv_eps_shared":
        "the three inverse-epsilon volumes are equal, so sharing one is the identity",
    "clear_all_axes":
        "no metallic axis has a DIAGONAL component that the off-diagonal rows spare",
}

#: Configurations. THE FIRST TWO ARE THE CORPUS SHAPES this product's two cells own,
#: read off the stamped census (results/cuda_predicate_coverage_2026-09-02_stamped):
#: absorber-1d.py and TestAbsorber.test_absorber are 1-D on z, METALLIC on z, five
#: poles per component and conductive on every D component; material-dispersion.py is
#: a single cell, all-periodic, two poles per component and LOSSLESS. The rest exist
#: to arm mutations those two cannot.
CASES: Tuple[Dict[str, Any], ...] = (
    {"label": "corpus_absorber_1d", "cell": (0.0, 0.0, 8.0), "resolution": 10.0,
     "boundaries": "metallic", "poles": 5, "conductivity": 0.4,
     "why": "the absorber-1d.py / TestAbsorber.test_absorber shape: 1-D on z, "
            "METALLIC on z, five poles, conductive on every D component"},
    {"label": "corpus_material_dispersion", "cell": (0.0, 0.0, 0.1),
     "resolution": 10.0, "boundaries": "periodic", "poles": 2,
     "conductivity": None,
     "why": "the material-dispersion.py shape: a SINGLE cell (grid.shape 1x1x1), "
            "all periodic, two poles, LOSSLESS -- the (False, False, False) COND "
            "build"},
    {"label": "wall_xyz_conductive", "cell": (1.2, 1.2, 1.2), "resolution": 8.0,
     "boundaries": "metallic", "poles": 3, "conductivity": 0.35,
     "anisotropic": True,
     "why": "every wall row live in 3-D on a conductive run, with THREE DISTINCT "
            "inverse-epsilon volumes: arms both clear mutations on all three axes "
            "and inv_eps_shared at the same time"},
    {"label": "wall_xyz_lossless", "cell": (1.2, 1.2, 1.2), "resolution": 8.0,
     "boundaries": "metallic", "poles": 2, "conductivity": None,
     "anisotropic": True,
     "why": "the same walls on the LOSSLESS curl arm, so a clear mutation cannot "
            "pass by riding the conductive tail"},
    {"label": "periodic_conductive_2d", "cell": (1.2, 1.2, 0.0), "resolution": 8.0,
     "boundaries": "periodic", "poles": 4, "conductivity": 0.5,
     "anisotropic": True,
     "why": "NO wall at all: both clear mutations must go NULL here, which is what "
            "shows they are measuring the wall and not something else"},
    {"label": "single_pole_wall_z", "cell": (0.0, 0.0, 1.2), "resolution": 10.0,
     "boundaries": "metallic", "poles": 1, "conductivity": 0.25,
     "anisotropic": True,
     "why": "ONE pole: pole_order_reversed must go NULL and poles_dropped must not"},
    {"label": "zero_pole_wall_z", "cell": (0.0, 0.0, 1.2), "resolution": 10.0,
     "boundaries": "metallic", "poles": 0, "conductivity": 0.25,
     "anisotropic": True,
     "why": "NO pole at all: recorded VACUOUS on the pole legs by construction, and "
            "kept because the emitted chain must collapse to E = v * inv_eps"},
)


# ---------------------------------------------------------------------------
# The fixture
# ---------------------------------------------------------------------------

def build(case: Dict[str, Any]):
    """One seeded NumPy engine on the PLAIN branch: ``(fields, grid, pml)``.

    ``PML(thickness=0)`` is a layer that absorbs on no face, so
    ``stepping._pml_is_active`` is False and ``update_E`` takes the pure overwrite at
    stepping.py:1022. ``enable_field_storage`` allocates E WITHOUT ``f_w``, which is
    the storage that branch writes into and the absence the split-field repair refuses
    on -- the two facts this product is built around.

    A wall goes only on a RESOLVED axis: ``Grid`` refuses ``metallic`` on a
    translationally invariant axis by name (grid.py:863), because a PEC there is a
    polarization filter rather than a wall.
    """
    rng = np.random.default_rng(SEED)
    cell = tuple(float(v) for v in case["cell"])
    dimensions = max(1, sum(1 for extent in cell if extent > 0.0))
    per_axis = {name: (case["boundaries"] if extent > 0.0 else "periodic")
                for name, extent in zip("xyz", cell)}
    grid = Grid(resolution=float(case["resolution"]), cell_size=cell,
                dimensions=dimensions, boundaries=per_axis, xp=np)
    fields = Fields(grid=grid)
    fields.set_background_eps(2.25)
    if case.get("anisotropic"):
        # THREE INDEPENDENT inverse-epsilon volumes, never one array bound thrice,
        # and drawn AWAY from 1.0 so dropping the multiply is not the identity.
        # ``set_background_eps`` alone hands the SAME volume to all three components,
        # which makes ``inv_eps_shared`` the identity and its null uninformative.
        shape_now = tuple(int(n) for n in grid.shape)
        epsilon, inverse = {}, {}
        for offset, component in enumerate(_ELECTRIC):
            values = rng.uniform(1.2 + 0.4 * offset, 3.4 + 0.4 * offset,
                                 size=shape_now).astype(np.float32)
            epsilon[component] = values
            inverse[component] = (np.float32(1.0) / values).astype(np.float32)
        fields.set_epsilon_volumes(epsilon, inverse)

    # REAL PolarizationState objects, with a DIFFERENT sigma per component so the
    # three see different pole SUBSETS in ``poles_per_component`` order -- the
    # property the emitted NP0/NP1/NP2 chain lengths specialise on, and the one a
    # port that used a single chain for all three would get wrong.
    for index in range(int(case["poles"])):
        sigma = {"Ex": 0.30 + 0.05 * index,
                 "Ey": 0.0 if index else 0.20,
                 "Ez": 0.25 + 0.05 * index}
        fields.polarizations.append(PolarizationState(
            Susceptibility(1.0 + 0.3 * index, 0.1, "lorentzian"), sigma, grid,
            np.float32))
    if case["conductivity"] is not None:
        # A VOLUME, not a scalar: a spatially varying sigma is what an mp.Absorber
        # actually installs, and a constant one would make condfac/condinv
        # indistinguishable from a scalar the emitter could have hoisted.
        volume = np.full(tuple(grid.shape), float(case["conductivity"]),
                         dtype=np.float32)
        volume *= np.linspace(0.5, 1.5, int(grid.shape[0]),
                              dtype=np.float32)[:, None, None]
        fields.set_d_conductivity(volume)
    fields.enable_field_storage()
    pml = PML(grid=grid, thickness=0)

    shape = tuple(int(n) for n in grid.shape)
    for name in _TARGETS + _ELECTRIC + ("Hx", "Hy", "Hz", "Bx", "By", "Bz"):
        array = getattr(fields, name, None)
        if array is None:
            continue
        array[...] = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    # THE POLE VOLUMES ARE SEEDED NON-ZERO. Zero is a fixed point of ``s = s - P``, so
    # a zero fixture turns every pole mutation into a null and the sweep would measure
    # nothing at all.
    for state in fields.polarizations:
        for attribute in ("P", "P_prev"):
            for array in (getattr(state, attribute, None) or {}).values():
                if array is None:
                    continue
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


def pole_volumes(fields: Any) -> List[np.ndarray]:
    """Every pole buffer, in a stable order, so the comparison covers them too."""
    out: List[np.ndarray] = []
    for state in fields.polarizations:
        for attribute in ("P", "P_prev"):
            store = getattr(state, attribute, None) or {}
            for component in _ELECTRIC:
                array = store.get(component)
                if array is not None:
                    out.append(array)
    return out


# ---------------------------------------------------------------------------
# The closed form
# ---------------------------------------------------------------------------

def _clear_mask(fields: Any, target: str, all_axes: bool) -> Any:
    """The cells ``zero_metal_D`` holds at zero for one D component.

    READ FROM THE SHIPPED PASS rather than transcribed: the mask is measured by
    running ``stepping.zero_metal_D`` on a volume of ones and seeing which words it
    zeroed. A transcription here could drift from ``_zero_metal``'s own off-diagonal
    rule -- which spares the component parallel to each wall -- and a probe that
    disagreed with the engine about WHICH cells clear would score the clear mutations
    against the wrong reference.
    """
    probe = Fields(grid=fields.grid)
    for name in _TARGETS:
        getattr(probe, name)[...] = np.float32(1.0)
    stepping.zero_metal_D(probe)
    mask = np.asarray(getattr(probe, target)) == 0.0
    if all_axes:
        # THE MUTATION'S mask: the union over all three components, i.e. a kernel
        # that cleared every metallic axis instead of only the two TANGENTIAL rows.
        for name in _TARGETS:
            mask = mask | (np.asarray(getattr(probe, name)) == 0.0)
    return mask


def closed_form(register: Dict[str, np.ndarray], poles: Dict[str, List[np.ndarray]],
                inv_eps: Dict[str, np.ndarray], masks: Dict[str, Any],
                masks_all: Dict[str, Any], mutation: str = "") -> Dict[str, np.ndarray]:
    """The fused launch, as one dictionary of the volumes it writes.

    ``register`` is the POST-CURL displacement -- the value the kernel holds in ``v``
    after its curl tail. The curl itself is the certified pass and is NOT re-typed
    here; what this function is about is the composition between it and ``update_E``.
    """
    written: Dict[str, np.ndarray] = {}
    for target, component in zip(_TARGETS, _ELECTRIC):
        value = np.array(register[target], copy=True)
        mask = masks_all[target] if mutation == "clear_all_axes" else masks[target]
        cleared = np.array(value, copy=True)
        cleared[mask] = np.float32(0.0)

        # THE STORE. The kernel writes the CLEARED register, because the array path's
        # ``zero_metal_D`` has already written those zeros into D by the time anything
        # reads it again.
        written[target] = value if mutation == "store_before_clear" else cleared

        # THE CONSTITUTIVE READ, from the same cleared register.
        source_register = value if mutation == "subtract_before_clear" else cleared
        chain = list(poles[component])
        if mutation == "pole_order_reversed":
            chain = list(reversed(chain))
        if mutation == "poles_dropped":
            chain = []
        source = np.array(source_register, copy=True)
        for array in chain:
            source = source - array
        volume = (inv_eps[_ELECTRIC[0]] if mutation == "inv_eps_shared"
                  else inv_eps[component])
        written[component] = (source * volume).astype(np.float32)
    return written


# ---------------------------------------------------------------------------
# The measurement
# ---------------------------------------------------------------------------

def measure(case: Dict[str, Any], steps: int) -> Dict[str, Any]:
    """One configuration: the array path's passes, then the closed form and every
    armed mutation against it, over ``steps`` complete seams."""
    started = time.time()
    fields, grid, pml = build(case)
    shape = tuple(int(n) for n in grid.shape)
    poles = pole_chain(fields)
    arity = tuple(len(poles[component]) for component in _ELECTRIC)
    live = bool(any(np.any(array != 0) for chain in poles.values()
                    for array in chain))
    masks = {target: _clear_mask(fields, target, False) for target in _TARGETS}
    masks_all = {target: _clear_mask(fields, target, True) for target in _TARGETS}
    inv_eps = {component: np.asarray(fields.inverse_epsilon_for(component))
               for component in _ELECTRIC}
    distinct_inv_eps = not all(
        np.array_equal(inv_eps[_ELECTRIC[0]], inv_eps[component])
        for component in _ELECTRIC[1:])

    result: Dict[str, Any] = {
        "label": case["label"], "why": case["why"], "shape": list(shape),
        "arity": list(arity), "poles_are_live": live,
        "pml_is_active": bool(stepping._pml_is_active(pml)),
        "stores_E": bool(fields.stores_E),
        "has_conductivity": bool(case["conductivity"] is not None),
        "cleared_cells": {target: int(np.count_nonzero(masks[target]))
                          for target in _TARGETS},
        "distinct_inverse_epsilon": bool(distinct_inv_eps),
        "steps": steps,
    }

    # THE SUB-STEP THIS PROBE CLAIMS TO MEASURE, CHECKED RATHER THAN ASSUMED. An
    # allocated ``f_w_E*`` or ``fu_D*`` would mean the fixture took the split-field
    # branch and every number below would describe a different recurrence.
    allocated = [name for name in _MUST_BE_ABSENT
                 if getattr(fields, name, None) is not None]
    result["unexpectedly_allocated"] = allocated
    if allocated or result["pml_is_active"] or not result["stores_E"]:
        result["fixture_error"] = (
            f"this case is not on the plain stored-E branch: pml_is_active="
            f"{result['pml_is_active']}, stores_E={result['stores_E']}, "
            f"allocated={allocated}")
        result["elapsed_s"] = round(time.time() - started, 3)
        return result

    engines = {label: build(case)[0] for label in ("shipped",) + MUTATIONS}
    reference = fields
    differing = {label: 0 for label in engines}
    for _step in range(steps):
        # --- the array path's three passes, in the driver's order ------------------
        stepping.step_D(reference, pml)
        stepping.zero_metal_D(reference)
        stepping.update_E(reference, pml)
        stepping.update_P(reference, pml)

        for label, engine in engines.items():
            # THE CURL IS THE CERTIFIED PASS, not a second transcription: what this
            # probe is about is the composition between it and ``update_E``.
            stepping.step_D(engine, pml)
            register = {name: np.array(getattr(engine, name), copy=True)
                        for name in _TARGETS}
            model = closed_form(
                register, pole_chain(engine),
                {component: np.asarray(engine.inverse_epsilon_for(component))
                 for component in _ELECTRIC},
                masks, masks_all,
                mutation="" if label == "shipped" else label)
            for name, array in model.items():
                getattr(engine, name)[...] = array
            stepping.update_P(engine, pml)
            differing[label] += sum(
                int(np.count_nonzero(words(getattr(engine, name))
                                     != words(getattr(reference, name))))
                for name in _COMPARED)
            differing[label] += sum(
                int(np.count_nonzero(words(a) != words(b)))
                for a, b in zip(pole_volumes(engine), pole_volumes(reference)))

    result["shipped"] = {"differing_words": differing["shipped"]}
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
    result["deposit"] = deposit_legs(case)
    result["elapsed_s"] = round(time.time() - started, 3)
    return result


# ---------------------------------------------------------------------------
# The deposit legs -- what PLAIN_PATH is for
# ---------------------------------------------------------------------------

def _source_for(grid: Any) -> Any:
    """One electric point source on the resolved axis, at the grid's own centre."""
    return GaussianPulsedSource(grid=grid, component="Ez", center=(0.0, 0.0, 0.0),
                                size=(0.0, 0.0, 0.0), frequency=1.0, fwidth=0.2,
                                amplitude=1.0)


def _driver_order(fields: Any, pml: Any, sources: Sequence[Any], when: float) -> None:
    """The driver's own electric seam, verbatim (driver.py:3306-3313), no fast path."""
    stepping.step_D(fields, pml)
    for source in sources:
        source.inject(fields, when)
    stepping.zero_metal_D(fields)
    stepping.update_E(fields, pml)
    stepping.update_P(fields, pml)


def _fused_order(fields: Any, pml: Any, sources: Sequence[Any], when: float,
                 masks: Dict[str, Any], masks_all: Dict[str, Any],
                 repair: bool, paths: Sequence[str]) -> Dict[str, Any]:
    """The fused route: save, launch on an UNINJECTED D, inject, clear, repair.

    Returns ``{"refused": reason}`` when ``deposit_repair`` declines the configuration,
    which is the outcome the ``split_field_declared`` leg exists to observe.
    """
    try:
        saved = deposit_repair.save(fields, tuple(sources), "D", pml, paths=paths)
    except Exception as error:  # a refusal is the measurement, not a failure
        return {"refused": f"{type(error).__name__}: {error}"}
    stepping.step_D(fields, pml)
    register = {name: np.array(getattr(fields, name), copy=True) for name in _TARGETS}
    model = closed_form(register, pole_chain(fields),
                        {component: np.asarray(fields.inverse_epsilon_for(component))
                         for component in _ELECTRIC},
                        masks, masks_all)
    for name, array in model.items():
        getattr(fields, name)[...] = array
    for source in sources:
        source.inject(fields, when)
    stepping.zero_metal_D(fields)
    repairs = 0
    if repair:
        repairs = deposit_repair.apply(fields, pml, tuple(sources), "D", saved)
    stepping.update_P(fields, pml)
    return {"refused": None, "repairs": int(repairs)}


def deposit_legs(case: Dict[str, Any], steps: int = 3) -> Dict[str, Any]:
    """``bracketed`` / ``unbracketed`` / ``split_field_declared`` on one configuration.

    ``unbracketed`` IS A NULL CONTROL THAT MUST DIVERGE: without the repair the fused
    launch's constitutive half consumed a pre-injection displacement, so E is wrong at
    every deposit cell. A leg that came back identical would mean the fixture's source
    deposited nothing, and the leg is reported ``vacuous`` in that case rather than
    counted as agreement.
    """
    out: Dict[str, Any] = {}
    reference, grid, pml = build(case)
    masks = {target: _clear_mask(reference, target, False) for target in _TARGETS}
    masks_all = {target: _clear_mask(reference, target, True) for target in _TARGETS}
    sources = [_source_for(grid)]
    reference_sources = [_source_for(grid)]

    # DOES THE FIXTURE ACTUALLY DEPOSIT? A source whose index is empty makes every leg
    # below a null, so the question is asked before the legs are scored.
    probe_fields, probe_grid, _p = build(case)
    probe_source = _source_for(probe_grid)
    before = np.array(probe_fields.Dz, copy=True)
    probe_source.inject(probe_fields, 0.5)
    deposited = int(np.count_nonzero(words(probe_fields.Dz) != words(before)))
    out["deposited_words"] = deposited

    dt = float(grid.dt)
    engines = {label: build(case)[0] for label in
               ("bracketed", "unbracketed", "split_field_declared")}
    engine_sources = {label: [_source_for(grid)] for label in engines}
    refusals: Dict[str, Any] = {}
    for step in range(steps):
        when = (step + 0.5) * dt
        _driver_order(reference, pml, reference_sources, when)
        for label, engine in engines.items():
            paths = ((deposit_repair.SPLIT_FIELD_PATH,)
                     if label == "split_field_declared"
                     else (deposit_repair.PLAIN_PATH,))
            verdict = _fused_order(engine, pml, engine_sources[label], when,
                                   masks, masks_all,
                                   repair=(label != "unbracketed"), paths=paths)
            if verdict["refused"] is not None:
                refusals.setdefault(label, verdict["refused"])
    for label, engine in engines.items():
        differing = sum(
            int(np.count_nonzero(words(getattr(engine, name))
                                 != words(getattr(reference, name))))
            for name in _COMPARED)
        differing += sum(int(np.count_nonzero(words(a) != words(b)))
                         for a, b in zip(pole_volumes(engine),
                                         pole_volumes(reference)))
        out[label] = {"differing_words": differing,
                      "refused": refusals.get(label),
                      "vacuous": deposited == 0}
    out["bracketed_is_identical"] = (out["bracketed"]["differing_words"] == 0
                                     and out["bracketed"]["refused"] is None)
    out["unbracketed_diverges"] = (out["unbracketed"]["differing_words"] > 0
                                   or deposited == 0)
    out["split_field_declared_is_refused"] = refusals.get(
        "split_field_declared") is not None
    _ = sources  # the reference list is the one driven; kept for symmetry of build
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True,
                        help="DIRECTORY to write "
                             "cuda_no_pml_dispersive_electric_pair.json")
    parser.add_argument("--steps", type=int, default=3,
                        help="complete driver steps per configuration")
    arguments = parser.parse_args()
    arguments.out.mkdir(parents=True, exist_ok=True)
    artifact = arguments.out / "cuda_no_pml_dispersive_electric_pair.json"

    rows: List[Dict[str, Any]] = []
    started = time.time()
    for index, case in enumerate(CASES, start=1):
        row = measure(case, arguments.steps)
        rows.append(row)
        # RULE 7: one flushed line per case, and the artifact rewritten after each, so
        # an interrupted run keeps everything up to the failure.
        print(f"case {index}/{len(CASES)} {row['label']}: "
              f"shipped_diff={row.get('shipped', {}).get('differing_words', 'n/a')} "
              f"mutations={row.get('mutations_that_diverge', 0)}/{len(MUTATIONS)} "
              f"nulls={row.get('nulls', [])} "
              f"deposit_ok={row.get('deposit', {}).get('bracketed_is_identical')} "
              f"({row['elapsed_s']} s)", flush=True)
        artifact.write_text(json.dumps({"rows": rows}, indent=1))

    identical = [r for r in rows if r.get("shipped", {}).get("differing_words") == 0]
    vacuous = [r["label"] for r in rows if r.get("case_is_vacuous")]
    errors = [r["label"] for r in rows if r.get("fixture_error")]
    bracket_ok = [r for r in rows
                  if r.get("deposit", {}).get("bracketed_is_identical")]
    unbracket_ok = [r for r in rows
                    if r.get("deposit", {}).get("unbracketed_diverges")]
    refused_ok = [r for r in rows
                  if r.get("deposit", {}).get("split_field_declared_is_refused")]
    summary = {
        "cases": len(rows),
        "identical": len(identical),
        "vacuous": vacuous,
        "fixture_errors": errors,
        "bracketed_identical": len(bracket_ok),
        "unbracketed_diverged": len(unbracket_ok),
        "split_field_refused": len(refused_ok),
        "mutations": list(MUTATIONS),
        "elapsed_s": round(time.time() - started, 3),
    }
    artifact.write_text(json.dumps({"rows": rows, "summary": summary}, indent=1))
    print(json.dumps(summary, indent=1), flush=True)
    ok = (len(identical) == len(rows) and not vacuous and not errors
          and len(bracket_ok) == len(rows) and len(unbracket_ok) == len(rows)
          and len(refused_ok) == len(rows))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
