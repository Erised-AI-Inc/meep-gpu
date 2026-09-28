"""Does the CONDUCTIVE x ORDINARY D/E pair's closed form reproduce the driver's seam?

THE CLAIM THIS PROBE STATES AND MEASURES. One launch per cell can perform

    step_D (driver.py:3306, the three-history CONDUCTIVE-PML recurrence)
    -> zero_metal_D (:3310)
    -> ordinary split-field update_E (:3313, stepping.py:1014-1018)

with the stepped displacement carried in a REGISTER between the three passes, and the
in-seam electric deposit carried across the launch by the split-field repair.

THE CELL IS ``D_to_E (cuda_conductive/conductive, cuda_constitutive/ordinary)`` and it
owns exactly one corpus row: ``tests:TestAdjointSolver.test_damping`` -- an ACTIVE
absorber, a conductivity on every D component, no susceptibility, METALLIC on x and y,
and both a ``D`` and a ``B`` source (the stamped census,
``results/cuda_predicate_coverage_2026-09-02_stamped``). It is the row this campaign
has called THE SIGNED-ZERO ROW.

=============================================================================
WHY IT WAS CALLED THAT, AND WHAT THIS PROBE RE-MEASURES RATHER THAN INHERITS
=============================================================================

``FdtdDriver._inject_electric_through_conductivity`` scales an injected current by
``condinv`` (MEEP step.cpp:294-317). It USED to do that as three WHOLE-VOLUME passes --
``array -= before; array *= condinv; array += before`` -- which are the identity at
every finite value EXCEPT that they canonicalise ``-0.0`` to ``+0.0`` at every cell of
the component, and on a device that flushes subnormals also ZERO every non-deposit
subnormal word. Both rewrites happen at cells no deposit closure can name, so a fused
launch that computed E from pre-rewrite values diverged in words the repair could not
reach -- which is why the fused electric pairs on three backends refused conductive
rows by name.

THE DRIVER NO LONGER DOES THAT. It snapshots the target at the deposit cells the
sources publish and replays the rescale per deposit cell in the retired passes' exact
operand order. THAT FACT IS VERIFIED HERE ON THE LIVE TREE rather than taken on trust:
:func:`sparse_rescale_leg` reads the shipped method, drives it, and MEASURES that the
non-deposit words are untouched -- and replays the retired dense composition as a
CONTROL that must diverge.

THE CLASSIFIER ADMITS BOTH CLASSES, and that is deliberate. The published CuPy finding
(RTX A6000, 2026-09-01) was that the dense passes destroyed 54 signed-zero AND 68
subnormal words of 588. A classifier that only knew about signed zeros would score the
subnormal half as "other" and report a divergence it could not explain. On NumPy the
subnormal class is UNREACHABLE -- the host does not flush subnormals -- so it is
recorded ``predicted_null`` WITH that reason rather than silently absent, and the
signed-zero class is the one that arms here.

=============================================================================
WHAT IS COMPARED
=============================================================================

Raw uint32 WORDS of every stored volume the seam writes -- ``Dx/Dy/Dz``,
``fu_Dx/fu_Dy/fu_Dz``, ``f_cond_Dx/f_cond_Dy/f_cond_Dz``, ``Ex/Ey/Ez`` and
``f_w_Ex/f_w_Ey/f_w_Ez`` -- against the array path's own passes run from identical
state, over COMPLETE driver steps. Never ``allclose``: ``-0.0 == 0.0`` lies, and the
sign of zero is the entire subject of this file.

THE AUXILIARIES START NON-ZERO for the reason every leg on this track gives: a zero
``fu`` makes ``fu * kms`` exactly zero on the first pass whatever ``kms`` is, which
would hide a mis-indexed coefficient. ``f_cond`` starts non-zero for the same reason --
it is the third history of the conductive recurrence and zero is its fixed point.

THE ARMED MUTATIONS, each a port a reasonable person would write:

  subtract_before_clear  the constitutive consumes the PRE-clear register
  store_before_clear     the store writes the PRE-clear register
  fw_not_saved           the accumulation uses the FRESH fw as its previous value
  fw_order_swapped       ``+kms*fw_prev - kps*fw`` instead of ``+kps*fw - kms*fw_prev``
  inv_eps_shared         all three components multiply component x's inverse epsilon
  coefficients_integer   the INTEGER sigma sub-lattice instead of the half-integer one

AND THE DEPOSIT AND RESCALE LEGS:

  bracketed              save -> launch on an UNINJECTED D -> inject -> clear -> repair
  unbracketed            the same without the repair -- A NULL CONTROL THAT MUST DIVERGE
  sparse_rescale         the SHIPPED injection: non-deposit words must be UNTOUCHED
  dense_rescale_control  the RETIRED whole-volume passes -- MUST diverge, and every
                         differing word must classify as signed-zero or subnormal

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
from meep_gpu.driver import FdtdDriver  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.sources import GaussianPulsedSource  # noqa: E402

_TARGETS: Tuple[str, ...] = ("Dx", "Dy", "Dz")
_ELECTRIC: Tuple[str, ...] = ("Ex", "Ey", "Ez")
_AXIS_NAME: Tuple[str, ...] = ("x", "y", "z")

#: Every stored volume the seam reads or writes, including BOTH curl histories.
_COMPARED: Tuple[str, ...] = (
    _TARGETS + tuple("fu_" + name for name in _TARGETS)
    + tuple("f_cond_" + name for name in _TARGETS)
    + _ELECTRIC + tuple("f_w_" + name for name in _ELECTRIC))

SEED = 20260902

MUTATIONS: Tuple[str, ...] = (
    "subtract_before_clear", "store_before_clear", "fw_not_saved",
    "fw_order_swapped", "inv_eps_shared", "coefficients_integer")

_NULL_REASONS: Dict[str, str] = {
    "subtract_before_clear":
        "no metallic wall clears a cell where D * inv_eps is non-zero",
    "store_before_clear":
        "no metallic wall clears a cell whose pre-clear register is non-zero",
    "fw_not_saved": "f_w already equals the fresh constitutive product everywhere",
    "fw_order_swapped": "kps and kms are equal everywhere, so the two orders agree",
    "inv_eps_shared":
        "the three inverse-epsilon volumes are equal, so sharing one is the identity",
    "coefficients_integer":
        "the absorber profile does not differ between the integer and half-integer "
        "sigma sub-lattices on any axis this run resolves",
}

#: Configurations. THE FIRST IS THE CORPUS SHAPE this cell owns, read off the stamped
#: census: TestAdjointSolver.test_damping is 2-D, METALLIC on x and y, periodic on z,
#: an ACTIVE absorber, a conductivity and NO susceptibility. The rest arm what it
#: cannot.
CASES: Tuple[Dict[str, Any], ...] = (
    {"label": "corpus_damping_wall_xy", "cell": (1.5, 1.5, 0.0), "resolution": 8.0,
     "boundaries": {"x": "metallic", "y": "metallic", "z": "periodic"},
     # PER AXIS, because the run is 2-D: an absorber cannot fit on a z axis of one
     # cell, and the corpus row's own layer is on x and y for the same reason.
     "pml": ((2, 2), (2, 2), (0, 0)), "conductivity": 0.4,
     "why": "the TestAdjointSolver.test_damping shape: 2-D, METALLIC on x and y, "
            "periodic on z, an ACTIVE absorber and a conductivity on every D "
            "component"},
    {"label": "wall_xyz_deep_pml", "cell": (1.5, 1.5, 1.5), "resolution": 8.0,
     "boundaries": "metallic", "pml": ((3, 3), (2, 2), (1, 1)),
     "conductivity": 0.35,
     "why": "every wall row live in 3-D under an ASYMMETRIC absorber, so the "
            "half-integer and integer sigma sub-lattices differ on every axis"},
    {"label": "periodic_pml_3d", "cell": (1.5, 1.5, 1.5), "resolution": 8.0,
     "boundaries": "periodic", "pml": 2, "conductivity": 0.5,
     "why": "NO wall at all: both clear mutations must go NULL here, which is what "
            "shows they are measuring the wall and not something else"},
    {"label": "wall_z_thin_pml", "cell": (0.0, 0.0, 1.5), "resolution": 10.0,
     "boundaries": {"x": "periodic", "y": "periodic", "z": "metallic"},
     # THE ABSORBER GOES ON THE ONE RESOLVED AXIS. x and y are single cells here and
     # a layer cannot fit on either.
     "pml": ((0, 0), (0, 0), (1, 1)), "conductivity": 0.25,
     "why": "1-D with a wall and a thin absorber: the smallest shape that still "
            "arms the clear mutations and the coefficient pair"},
)


# ---------------------------------------------------------------------------
# The fixture
# ---------------------------------------------------------------------------

def build(case: Dict[str, Any], zero_curl: bool = False,
          signed_zero_class: bool = False):
    """One seeded NumPy engine on the CONDUCTIVE-PML branch: ``(fields, grid, pml)``.

    ``enable_pml_storage`` allocates ``fu_D`` and ``f_w_E``; the conductivity makes
    ``Fields._ensure_conductive_pml_storage`` allocate ``f_cond_D`` as well, which is
    the third history the conductive recurrence advances and the reason this cell's
    curl arm is a family of its own.

    ``zero_curl`` seeds H and B to EXACTLY zero. That is what lets a ``-0.0`` word
    survive ``step_D`` to the injection point: the conductive tail is
    ``((f * cf) - curl) * ci`` and with ``curl`` exactly ``+0.0`` it maps ``-0.0`` to
    ``-0.0``. Used only by the rescale leg, whose subject IS the sign of zero.

    ``signed_zero_class`` seeds D with a deliberate mix of ``-0.0``, ``+0.0``,
    subnormal and normal words, so the classifier has all four to sort.
    """
    rng = np.random.default_rng(SEED)
    cell = tuple(float(v) for v in case["cell"])
    dimensions = max(1, sum(1 for extent in cell if extent > 0.0))
    boundaries = case["boundaries"]
    if isinstance(boundaries, str):
        boundaries = {name: (boundaries if extent > 0.0 else "periodic")
                      for name, extent in zip("xyz", cell)}
    grid = Grid(resolution=float(case["resolution"]), cell_size=cell,
                dimensions=dimensions, boundaries=boundaries, xp=np)
    fields = Fields(grid=grid)
    shape = tuple(int(n) for n in grid.shape)

    # THREE INDEPENDENT inverse-epsilon volumes, drawn AWAY from 1.0 so dropping or
    # sharing the multiply is not the identity.
    epsilon, inverse = {}, {}
    for offset, component in enumerate(_ELECTRIC):
        values = rng.uniform(1.2 + 0.4 * offset, 3.4 + 0.4 * offset,
                             size=shape).astype(np.float32)
        epsilon[component] = values
        inverse[component] = (np.float32(1.0) / values).astype(np.float32)
    fields.set_epsilon_volumes(epsilon, inverse)

    # A VOLUME, not a scalar: a spatially varying sigma is what an mp.Absorber
    # actually installs, and the corpus row's conductivity is one.
    volume = np.full(shape, float(case["conductivity"]), dtype=np.float32)
    volume *= np.linspace(0.5, 1.5, shape[0], dtype=np.float32)[:, None, None]
    fields.set_d_conductivity(volume)
    fields.enable_pml_storage()
    thickness = (case["pml"] if isinstance(case["pml"], int)
                 else tuple(tuple(int(v) for v in pair) for pair in case["pml"]))
    pml = PML(grid=grid, thickness=thickness)

    magnetic = ("Hx", "Hy", "Hz", "Bx", "By", "Bz")
    for name in _COMPARED + magnetic:
        array = getattr(fields, name, None)
        if array is None:
            continue
        if zero_curl and name in magnetic:
            array[...] = np.float32(0.0)
            continue
        array[...] = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    if signed_zero_class:
        for name in _TARGETS:
            array = getattr(fields, name)
            flat = array.reshape(-1)
            # FOUR WORD CLASSES IN ROTATION so every one of them is present at cells
            # the deposit does NOT name: -0.0, +0.0, the smallest subnormal, and a
            # normal value. The classifier has to sort all four.
            classes = np.array([-0.0, 0.0, np.float32(1e-45), 0.37],
                               dtype=np.float32)
            flat[...] = np.resize(classes, flat.size)
    return fields, grid, pml


def words(array: Any) -> np.ndarray:
    """Raw uint32 WORDS. Byte compares, never ``allclose``."""
    return np.ascontiguousarray(array, dtype=np.float32).ravel().view(np.uint32)


def classify(before: np.ndarray, after: np.ndarray) -> Dict[str, int]:
    """Sort every differing word into the classes this campaign has measured.

    ``signed_zero``  a zero whose SIGN BIT changed, either direction. ``-0.0 -> +0.0``
                     is what the retired whole-volume passes produce; ``+0.0 -> -0.0``
                     is admitted too rather than assumed impossible.
    ``subnormal``    a subnormal word that became zero, or a zero that came from one.
                     UNREACHABLE ON NUMPY -- the host does not flush subnormals -- and
                     recorded as such rather than left to look absent.
    ``other``        anything else, which is a divergence this file cannot explain and
                     is what makes a leg fail rather than pass.
    """
    a = np.ascontiguousarray(before, dtype=np.float32).ravel()
    b = np.ascontiguousarray(after, dtype=np.float32).ravel()
    wa, wb = a.view(np.uint32), b.view(np.uint32)
    differing = wa != wb
    counts = {"signed_zero": 0, "subnormal": 0, "other": 0}
    if not np.any(differing):
        return counts
    ia, ib = wa[differing], wb[differing]
    va, vb = a[differing], b[differing]
    zero_a, zero_b = (ia & 0x7FFFFFFF) == 0, (ib & 0x7FFFFFFF) == 0
    sub_a = ((ia & 0x7F800000) == 0) & ~zero_a
    sub_b = ((ib & 0x7F800000) == 0) & ~zero_b
    signed = zero_a & zero_b
    subnormal = (sub_a & zero_b) | (zero_a & sub_b)
    counts["signed_zero"] = int(np.count_nonzero(signed))
    counts["subnormal"] = int(np.count_nonzero(subnormal))
    counts["other"] = int(np.count_nonzero(differing)) - counts["signed_zero"] - \
        counts["subnormal"]
    _ = (va, vb)
    return counts


# ---------------------------------------------------------------------------
# The closed form
# ---------------------------------------------------------------------------

def _clear_mask(fields: Any, target: str) -> Any:
    """The cells ``zero_metal_D`` holds at zero for one D component, READ FROM THE
    SHIPPED PASS rather than transcribed (see the Group B probe's note)."""
    probe = Fields(grid=fields.grid)
    for name in _TARGETS:
        getattr(probe, name)[...] = np.float32(1.0)
    stepping.zero_metal_D(probe)
    return np.asarray(getattr(probe, target)) == 0.0


def closed_form(register: Dict[str, np.ndarray], fw: Dict[str, np.ndarray],
                electric: Dict[str, np.ndarray], inv_eps: Dict[str, np.ndarray],
                coefficients: Dict[str, Tuple[Any, Any]],
                masks: Dict[str, Any], mutation: str = "") -> Dict[str, np.ndarray]:
    """The fused launch, as one dictionary of the volumes it writes.

    ``register`` is the POST-CURL displacement -- the value the kernel holds in ``v``
    after its conductive-PML tail. The curl itself is the certified pass and is NOT
    re-typed here; what this function is about is the composition between it and
    ``update_E``.
    """
    written: Dict[str, np.ndarray] = {}
    for target, component, axis in zip(_TARGETS, _ELECTRIC, _AXIS_NAME):
        value = np.array(register[target], copy=True)
        cleared = np.array(value, copy=True)
        cleared[masks[target]] = np.float32(0.0)
        written[target] = value if mutation == "store_before_clear" else cleared

        source_register = value if mutation == "subtract_before_clear" else cleared
        volume = (inv_eps[_ELECTRIC[0]] if mutation == "inv_eps_shared"
                  else inv_eps[component])
        fresh = (source_register * volume).astype(np.float32)
        previous = (fresh if mutation == "fw_not_saved"
                    else np.array(fw["f_w_" + component], copy=True))
        kps, kms = coefficients[
            "integer" if mutation == "coefficients_integer" else "half"][axis]
        field = np.array(electric[component], copy=True)
        if mutation == "fw_order_swapped":
            field += kms * previous
            field -= kps * fresh
        else:
            field += kps * fresh
            field -= kms * previous
        written[component] = field.astype(np.float32)
        written["f_w_" + component] = fresh
    return written


def _coefficient_table(pml: Any) -> Dict[str, Dict[str, Tuple[Any, Any]]]:
    """Both sigma sub-lattices, so ``coefficients_integer`` is a real alternative
    rather than a fabricated one: ``stepping._constitutive_coefficients``' own two
    answers, at ``half_integer`` True (the E side's, stepping.py:1015) and False."""
    return {
        "half": {axis: stepping._constitutive_coefficients(pml, axis,
                                                           half_integer=True)
                 for axis in _AXIS_NAME},
        "integer": {axis: stepping._constitutive_coefficients(pml, axis,
                                                              half_integer=False)
                    for axis in _AXIS_NAME},
    }


# ---------------------------------------------------------------------------
# The measurement
# ---------------------------------------------------------------------------

def measure(case: Dict[str, Any], steps: int) -> Dict[str, Any]:
    """One configuration: the array path's passes, then the closed form and every
    armed mutation against it, over ``steps`` complete seams."""
    started = time.time()
    fields, grid, pml = build(case)
    shape = tuple(int(n) for n in grid.shape)
    masks = {target: _clear_mask(fields, target) for target in _TARGETS}
    coefficients = _coefficient_table(pml)
    inv_eps = {component: np.asarray(fields.inverse_epsilon_for(component))
               for component in _ELECTRIC}

    result: Dict[str, Any] = {
        "label": case["label"], "why": case["why"], "shape": list(shape),
        "pml_is_active": bool(stepping._pml_is_active(pml)),
        "stores_E": bool(fields.stores_E),
        "polarizations": len(fields.polarizations),
        "cleared_cells": {target: int(np.count_nonzero(masks[target]))
                          for target in _TARGETS},
        "f_cond_allocated": bool(getattr(fields, "f_cond_Dx", None) is not None),
        "steps": steps,
    }
    # THE SUB-STEP THIS PROBE CLAIMS TO MEASURE, CHECKED RATHER THAN ASSUMED. Without
    # an ACTIVE layer, an allocated f_cond and an empty pole chain this is a different
    # cell entirely and every number below would describe the wrong recurrence.
    if (not result["pml_is_active"] or not result["f_cond_allocated"]
            or result["polarizations"]):
        result["fixture_error"] = (
            f"this case is not the conductive x ordinary cell: "
            f"pml_is_active={result['pml_is_active']}, "
            f"f_cond={result['f_cond_allocated']}, "
            f"polarizations={result['polarizations']}")
        result["elapsed_s"] = round(time.time() - started, 3)
        return result

    engines = {label: build(case)[0] for label in ("shipped",) + MUTATIONS}
    reference = fields
    differing = {label: 0 for label in engines}
    for _step in range(steps):
        stepping.step_D(reference, pml)
        stepping.zero_metal_D(reference)
        stepping.update_E(reference, pml)

        for label, engine in engines.items():
            stepping.step_D(engine, pml)
            register = {name: np.array(getattr(engine, name), copy=True)
                        for name in _TARGETS}
            fw = {"f_w_" + c: np.array(getattr(engine, "f_w_" + c), copy=True)
                  for c in _ELECTRIC}
            electric = {c: np.array(getattr(engine, c), copy=True)
                        for c in _ELECTRIC}
            model = closed_form(
                register, fw, electric,
                {c: np.asarray(engine.inverse_epsilon_for(c)) for c in _ELECTRIC},
                coefficients, masks,
                mutation="" if label == "shipped" else label)
            for name, array in model.items():
                getattr(engine, name)[...] = array
            differing[label] += sum(
                int(np.count_nonzero(words(getattr(engine, name))
                                     != words(getattr(reference, name))))
                for name in _COMPARED)

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
    result["deposit"] = deposit_legs(case, masks, coefficients)
    result["rescale"] = rescale_leg(case)
    result["elapsed_s"] = round(time.time() - started, 3)
    _ = inv_eps
    return result


# ---------------------------------------------------------------------------
# The deposit legs
# ---------------------------------------------------------------------------

def _source_for(grid: Any, component: str = "Ez") -> Any:
    return GaussianPulsedSource(grid=grid, component=component,
                                center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                                frequency=1.0, fwidth=0.2, amplitude=1.0)


class _Injector:
    """The SHIPPED ``_inject_electric_through_conductivity``, bound to bare fields.

    The method reads ``self.fields`` and nothing else, so binding it to a two-line
    shim drives the tree's own code rather than a transcription of it -- which is the
    whole point of the rescale leg.
    """

    __slots__ = ("fields",)

    def __init__(self, fields: Any) -> None:
        self.fields = fields

    inject = FdtdDriver._inject_electric_through_conductivity


def _dense_rescale(fields: Any, sources: Sequence[Any], when: float) -> None:
    """THE RETIRED COMPOSITION, kept as a control and never as a code path.

    Three WHOLE-VOLUME passes, transcribed from the docstring of the method that
    replaced them. It is the identity at every finite value and canonicalises ``-0.0``
    to ``+0.0`` at every cell of the component -- which is exactly what this leg is
    for.
    """
    by_target: Dict[str, List[Any]] = {}
    for source in sources:
        if getattr(source, "is_integrated", False):
            continue
        by_target.setdefault("D" + source.component[1], []).append(source)
    before = {}
    for name in sorted(by_target):
        if fields.condinv_for(name) is None:
            continue
        before[name] = np.array(getattr(fields, name), copy=True)
    for source in sources:
        source.inject(fields, when)
    for name, snapshot in before.items():
        array = getattr(fields, name)
        array -= snapshot
        array *= fields.condinv_for(name)
        array += snapshot


def _driver_order(fields: Any, pml: Any, sources: Sequence[Any], when: float,
                  dense: bool = False) -> None:
    """The driver's own electric seam, verbatim (driver.py:3306-3313)."""
    stepping.step_D(fields, pml)
    if dense:
        _dense_rescale(fields, sources, when)
    else:
        _Injector(fields).inject(list(sources), when)
    stepping.zero_metal_D(fields)
    stepping.update_E(fields, pml)


def _fused_order(fields: Any, pml: Any, sources: Sequence[Any], when: float,
                 masks: Dict[str, Any], coefficients: Dict[str, Any],
                 repair: bool, dense: bool = False) -> None:
    """The fused route: save, launch on an UNINJECTED D, inject, clear, repair."""
    saved = deposit_repair.save(fields, tuple(sources), "D", pml)
    stepping.step_D(fields, pml)
    register = {name: np.array(getattr(fields, name), copy=True) for name in _TARGETS}
    fw = {"f_w_" + c: np.array(getattr(fields, "f_w_" + c), copy=True)
          for c in _ELECTRIC}
    electric = {c: np.array(getattr(fields, c), copy=True) for c in _ELECTRIC}
    model = closed_form(register, fw, electric,
                        {c: np.asarray(fields.inverse_epsilon_for(c))
                         for c in _ELECTRIC},
                        coefficients, masks)
    for name, array in model.items():
        getattr(fields, name)[...] = array
    if dense:
        _dense_rescale(fields, sources, when)
    else:
        _Injector(fields).inject(list(sources), when)
    stepping.zero_metal_D(fields)
    if repair:
        deposit_repair.apply(fields, pml, tuple(sources), "D", saved)


def deposit_legs(case: Dict[str, Any], masks: Dict[str, Any],
                 coefficients: Dict[str, Any], steps: int = 3) -> Dict[str, Any]:
    """``bracketed`` / ``unbracketed`` against the driver's own order.

    ``unbracketed`` IS A NULL CONTROL THAT MUST DIVERGE: without the repair the fused
    launch's constitutive half consumed a pre-injection displacement.
    """
    out: Dict[str, Any] = {}
    reference, grid, pml = build(case)
    reference_sources = [_source_for(grid)]

    probe_fields, probe_grid, _p = build(case)
    probe_source = _source_for(probe_grid)
    before = np.array(probe_fields.Dz, copy=True)
    _Injector(probe_fields).inject([probe_source], 0.5)
    out["deposited_words"] = int(
        np.count_nonzero(words(probe_fields.Dz) != words(before)))

    dt = float(grid.dt)
    engines = {label: build(case)[0] for label in ("bracketed", "unbracketed")}
    engine_sources = {label: [_source_for(grid)] for label in engines}
    for step in range(steps):
        when = (step + 0.5) * dt
        _driver_order(reference, pml, reference_sources, when)
        for label, engine in engines.items():
            _fused_order(engine, pml, engine_sources[label], when, masks,
                         coefficients, repair=(label != "unbracketed"))
    for label, engine in engines.items():
        out[label] = {"differing_words": sum(
            int(np.count_nonzero(words(getattr(engine, name))
                                 != words(getattr(reference, name))))
            for name in _COMPARED),
            "vacuous": out["deposited_words"] == 0}
    out["bracketed_is_identical"] = out["bracketed"]["differing_words"] == 0
    out["unbracketed_diverges"] = out["unbracketed"]["differing_words"] > 0
    return out


def rescale_leg(case: Dict[str, Any], steps: int = 1) -> Dict[str, Any]:
    """THE CHARTER'S OWN QUESTION, re-measured on the live tree.

    Two measurements, both on a fixture seeded with all four word classes and an
    exactly-zero curl so the classes survive ``step_D``:

    ``sparse_untouched``  the SHIPPED injection is driven and the non-deposit words
                          of every D component are compared before and after. They
                          must be UNTOUCHED -- that is the fact the charter told this
                          lane to verify rather than assume.
    ``dense_diverges``    the RETIRED whole-volume passes are replayed as a control
                          through the same fused route. It MUST diverge, and every
                          differing word must classify as signed-zero or subnormal.
    """
    out: Dict[str, Any] = {}

    # --- (1) does the shipped injection touch a non-deposit word? -----------------
    fields, grid, _pml = build(case, zero_curl=True, signed_zero_class=True)
    source = _source_for(grid)
    index = deposit_repair._deposit_index(source)
    out["source_publishes_index"] = index is not None
    before = {name: np.array(getattr(fields, name), copy=True) for name in _TARGETS}
    _Injector(fields).inject([source], 0.5)
    touched: Dict[str, int] = {}
    for name in _TARGETS:
        after = np.asarray(getattr(fields, name))
        mask = np.ones(after.shape, dtype=bool)
        if index is not None and name == "D" + source.component[1]:
            mask[index] = False
        touched[name] = int(np.count_nonzero(
            words(np.asarray(before[name])[mask]) != words(after[mask])))
    out["non_deposit_words_touched"] = touched
    out["sparse_untouched"] = all(count == 0 for count in touched.values())
    out["seeded_signed_zeros"] = int(sum(
        np.count_nonzero((words(before[name]) == 0x80000000)) for name in _TARGETS))
    out["seeded_subnormals"] = int(sum(
        np.count_nonzero(((words(before[name]) & 0x7F800000) == 0)
                         & ((words(before[name]) & 0x7FFFFFFF) != 0))
        for name in _TARGETS))

    # --- (2) the retired dense composition, as a control that must diverge --------
    masks = {target: _clear_mask(fields, target) for target in _TARGETS}
    reference, ref_grid, ref_pml = build(case, zero_curl=True,
                                         signed_zero_class=True)
    engine, eng_grid, _e = build(case, zero_curl=True, signed_zero_class=True)
    ref_sources = [_source_for(ref_grid)]
    eng_sources = [_source_for(eng_grid)]
    coefficients = _coefficient_table(ref_pml)
    dt = float(ref_grid.dt)
    for step in range(steps):
        when = (step + 0.5) * dt
        # THE REFERENCE TAKES THE SHIPPED SPARSE INJECTION -- it is the array path as
        # the tree ships it today. The engine replays the RETIRED dense passes on the
        # fused route, which is the composition whose word rewrites this leg is about.
        _driver_order(reference, ref_pml, ref_sources, when, dense=False)
        _fused_order(engine, ref_pml, eng_sources, when, masks, coefficients,
                     repair=True, dense=True)
    counts = {"signed_zero": 0, "subnormal": 0, "other": 0}
    total = 0
    for name in _COMPARED:
        part = classify(getattr(reference, name), getattr(engine, name))
        for key in counts:
            counts[key] += part[key]
        total += sum(part.values())
    out["dense_control_differing_words"] = total
    out["dense_control_classes"] = counts
    out["dense_diverges"] = total > 0
    out["dense_is_fully_classified"] = counts["other"] == 0 and total > 0
    # THE SUBNORMAL CLASS IS UNREACHABLE ON THIS HOST and is recorded as such rather
    # than left to look absent: NumPy does not flush subnormals, so only a device run
    # can arm it. The published CuPy measurement (RTX A6000, 2026-09-01) is 68
    # subnormal beside 54 signed-zero words of 588.
    out["subnormal_predicted_null"] = (
        None if counts["subnormal"] else
        "NumPy does not flush subnormals to zero, so the subnormal half of the "
        "retired passes' damage cannot arm on this host; only a device run can")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True,
                        help="DIRECTORY to write cuda_conductive_electric_pair.json")
    parser.add_argument("--steps", type=int, default=3,
                        help="complete driver steps per configuration")
    arguments = parser.parse_args()
    arguments.out.mkdir(parents=True, exist_ok=True)
    artifact = arguments.out / "cuda_conductive_electric_pair.json"

    rows: List[Dict[str, Any]] = []
    started = time.time()
    for index, case in enumerate(CASES, start=1):
        row = measure(case, arguments.steps)
        rows.append(row)
        print(f"case {index}/{len(CASES)} {row['label']}: "
              f"shipped_diff={row.get('shipped', {}).get('differing_words', 'n/a')} "
              f"mutations={row.get('mutations_that_diverge', 0)}/{len(MUTATIONS)} "
              f"nulls={row.get('nulls', [])} "
              f"deposit_ok={row.get('deposit', {}).get('bracketed_is_identical')} "
              f"sparse_untouched={row.get('rescale', {}).get('sparse_untouched')} "
              f"dense_classes={row.get('rescale', {}).get('dense_control_classes')} "
              f"({row['elapsed_s']} s)", flush=True)
        artifact.write_text(json.dumps({"rows": rows}, indent=1))

    identical = [r for r in rows if r.get("shipped", {}).get("differing_words") == 0]
    vacuous = [r["label"] for r in rows if r.get("case_is_vacuous")]
    errors = [r["label"] for r in rows if r.get("fixture_error")]
    bracket_ok = [r for r in rows
                  if r.get("deposit", {}).get("bracketed_is_identical")]
    unbracket_ok = [r for r in rows
                    if r.get("deposit", {}).get("unbracketed_diverges")]
    sparse_ok = [r for r in rows if r.get("rescale", {}).get("sparse_untouched")]
    dense_ok = [r for r in rows
                if r.get("rescale", {}).get("dense_is_fully_classified")]
    summary = {
        "cases": len(rows),
        "identical": len(identical),
        "vacuous": vacuous,
        "fixture_errors": errors,
        "bracketed_identical": len(bracket_ok),
        "unbracketed_diverged": len(unbracket_ok),
        "sparse_left_non_deposit_words_untouched": len(sparse_ok),
        "dense_control_diverged_and_fully_classified": len(dense_ok),
        "mutations": list(MUTATIONS),
        "elapsed_s": round(time.time() - started, 3),
    }
    artifact.write_text(json.dumps({"rows": rows, "summary": summary}, indent=1))
    print(json.dumps(summary, indent=1), flush=True)
    ok = (len(identical) == len(rows) and not vacuous and not errors
          and len(bracket_ok) == len(rows) and len(unbracket_ok) == len(rows)
          and len(sparse_ok) == len(rows) and len(dense_ok) == len(rows))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
