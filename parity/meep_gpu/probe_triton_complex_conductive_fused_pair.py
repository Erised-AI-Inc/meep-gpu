#!/usr/bin/env python3
"""Byte gate for the COMPLEX CONDUCTIVE fused pair: no-PML conductive ``step_D``
welded into complex stored-E ``update_E``.

THE CLAIM THIS GATE IS ALLOWED TO SUPPORT: for every configuration
:func:`~meep_gpu.triton_kernels.complex_conductive_fused_pair.complex_conductive_fused_pair_coverage`
admits, ONE launch of ``complex_conductive_fused_curl_constitutive_D`` leaves the
engine in a state that is BIT-IDENTICAL, over every dynamic volume, on the uint32
word view, PER COMPLETE DRIVER STEP, to BOTH

  * the CuPy ARRAY PATH (``stepping.step_D`` -> ``stepping.update_E``), and
  * the SEPARATELY CERTIFIED Triton products it replaces — the complex conductive
    no-PML curl (``complex_no_pml_conductive.plan_complex_conductive_no_pml_curl``
    on ``step_D``) and the complex stored-E constitutive
    (``complex_no_pml_stored_e.plan_complex_stored_e``).

``allclose`` appears nowhere.

WHY THIS SEAM IS EMPTY, AND WHY THAT IS THE DESIGN QUESTION RATHER THAN A SHRUG.
The driver runs four passes between the two halves. This family carries NONE of
them and REFUSES each by a named clause instead:

  * the electric injection — refused, and it costs the cell NOTHING: all four
    corpus rows (``TestLoadDump.test_load_dump_*_3d``) declare a MAGNETIC source
    only, which is injected in the B/H seam. That is why
    ``CARRIES_DEPOSIT_REPAIR`` is False here while the two D->E COMPLEX pairs
    beside it are True;
  * ``zero_metal_D`` — refused by name, and the refusal is PRICED: a metallic axis
    buys zero seam-instances on the measured corpus, and a widening that buys
    nothing is declined;
  * both symmetry fills — inert, because a mirror plane is refused.

So ``REPLACES`` is TWO call sites, not five, and this gate asserts that: the
harness installs the fused plan in ``step_D`` and ``update_E`` ONLY, leaves the
other three on the array path, and COUNTS them. A run in which one of the three did
work would show as a counter event rather than as an invisible correction.

WHAT THIS GATE REFUSES TO INFER
===============================

* **Bytes alone cannot prove the fused path ran.** A silent fallback is
  byte-identical to the array path BY CONSTRUCTION. Every launch goes through a
  :class:`CountingKernel`, every seam pass is counted PER ROUTE, and a fused route
  that reached the array path for an absorbed pass FAILS its row.
* **A no-op agreeing with a no-op is trivially identical.** Every row carries a
  non-vacuity floor: the compared state must have MOVED, and no read-only material
  volume may have.
* **A mutation that rewrites an unreached line measures nothing.** ``COND*``,
  ``NP*``, ``PH*``, ``BCX``/``BCY``/``BCZ`` and ``BACKWARD`` are ``tl.constexpr``
  guards that compile DIFFERENT BODIES. Every needle declares the case it is scored
  on and leg ``needle_reachability`` parses the needle's ENCLOSING GUARDS out of the
  shipped text and evaluates them against that case.
* **A pole chain that subtracts nothing cannot discriminate its order.** The
  mutation case carries FOUR poles on every component, so the order-reversal needle
  has somewhere to be caught; a one-pole row would report it as a null.

THE FAMILY IS NOT WIRED. ``launch.plan_step`` assigns at most one plan per slot and
this product spans two of them with no measured composition rule. Nothing in
``launch.py`` names it and ``fastpath`` is unchanged.

Progress reporting: one flushed line per step, every row appended to the artifact as it lands.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
API_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
for _path in (HERE, API_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

SEED = 20260830

_TEMPORARY: List[str] = []

FAMILY_MODULE = "meep_gpu/triton_kernels/complex_conductive_fused_pair.py"

KEEP_PROBES: Tuple[str, ...] = (
    "parity/meep_gpu/results/expansion_probe_2026-08-17/expansion_probe_keep.json",
    "parity/meep_gpu/results/complex_expansion_convention_2026-08-16/results/"
    "probe_keep/probe.json",
)

#: ``(name, poles, kind, conductive, k_point, steps)``.
#:
#: EVERY AXIS IS A COMPILE-TIME ARM, NOT DECORATION:
#:   poles      ``NP0``/``NP1``/``NP2`` compile a DIFFERENT NUMBER of subtraction
#:              arms. A one-pole row cannot discriminate the chain's ORDER at all.
#:   conductive ``COND0``/``COND1``/``COND2`` gate the conductive tail per target;
#:              a row with all three off compiles no tail and would report every
#:              tail mutation as uncaught.
#:   k_point    ``PHX``/``PHY``/``PHZ`` gate the wrap-plane rotation. A zero-k row
#:              leaves all three unfilled.
#:
#: THE FIRST ROW IS THE CORPUS SHAPE: one pole, all three components conductive,
#: which is what ``TestLoadDump.test_load_dump_*_3d`` drives.
CASES: Tuple[Tuple[str, int, str, Tuple[bool, bool, bool],
                   Tuple[float, float, float], int], ...] = (
    ("corpus_one_pole_all_conductive", 1, "lorentzian", (True, True, True),
     (0.0, 0.0, 0.0), 10),
    ("one_pole_bloch_x", 1, "lorentzian", (True, True, True),
     (0.23, 0.0, 0.0), 10),
    ("two_pole_three_axis_bloch", 2, "lorentzian", (True, True, True),
     (0.17, 0.11, 0.29), 10),
    ("four_pole_at_the_ceiling", 4, "mixed", (True, True, True),
     (0.17, 0.11, 0.29), 10),
    ("mixed_conductivity_x_only", 2, "lorentzian", (True, False, False),
     (0.23, 0.0, 0.0), 10),
)

#: THE LOSSLESS ROW IS A REFUSAL CASE, NOT A PRODUCT CASE, and the distinction is
#: the family boundary rather than a gap in the sweep. With no conductivity on any
#: target the curl half refuses BY NAME — "no curl target carries a conductivity;
#: the lossless complex no-PML family owns this slot" — because a DIFFERENT
#: certified product steps that configuration. Measured on the GPU host 2026-08-30,
#: where carrying it as a product row took the campaign down at the plan builder:
#: the refusal was correct and the sweep was wrong to ask.
#:
#: It is run in :func:`leg_refusal` instead, where the refusal is the measurement.
LOSSLESS_REFUSAL_CASE = ("no_conductivity_control", 2, "lorentzian",
                         (False, False, False), (0.23, 0.0, 0.0), 10)

#: The case every mutation is scored on. FOUR poles, ALL THREE components
#: conductive and a fully three-axis Bloch phase, so every armed line is emitted
#: AND executed: a zero-k row would leave the wrap-plane edits unfilled, a
#: non-conductive row would leave the conductive tail's lines unreached, and a
#: one-pole row would make the chain-order needle a null.
MUTATION_CASE = "four_pole_at_the_ceiling"

MUTATION_STEPS = 3

#: The driver call sites this product REPLACES — two, not five. The other three are
#: refused rather than carried, and a launch may only claim to replace what it
#: performs. They are still WATCHED: the harness installs on all five names so the
#: counter records what each route really called.
SEAM_PASSES: Tuple[str, ...] = (
    "step_D", "fill_symmetry_bc_D", "zero_metal_D", "fill_folded_far_ghosts_D",
    "update_E",
)
ABSORBED: Tuple[str, ...] = ("step_D", "update_E")

#: The names that MUST appear in the dynamic state inventory. Scanned rather than
#: listed so a renamed volume cannot silently drop out of the comparison; this
#: tuple is the tripwire for the scan itself shrinking.
#:
#: ``Hx``/``Hy``/``Hz`` ARE DELIBERATELY NOT HERE, and their absence is a fact
#: about this family rather than a gap in the check. This is the NO-PML branch:
#: with no absorber and mu = 1 the engine keeps no separate magnetic volume — the
#: curl's source IS the stored ``B`` — so a run that carried ``Hx`` would not be
#: the configuration this product admits. Requiring them took the first campaign
#: down at the inventory scan on 2026-08-30, which is the check working: it refused
#: to compare a state it could not fully account for.
#:
#: What replaces them is stronger and is asserted in :func:`inventory`: the curl's
#: OWN source volumes, read through the engine's accessor
#: (``complex_no_pml_curl.source_arrays``) rather than spelled here, so the scan is
#: required to contain whatever this sub-step actually reads.
REQUIRED = ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez")

MATERIAL = ("eps", "inv_eps")


def log(message: str) -> None:
    print(message, flush=True)


def sha256_of(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _assert_material_names_are_real(found: Dict[str, Any]) -> None:
    present = [name for name in MATERIAL if name in found]
    if not present:
        raise AssertionError(
            f"none of the declared read-only material names {MATERIAL} match the "
            f"state scan {sorted(found)}; the material floor is inert")


# ===========================================================================
# Fixture
# ===========================================================================

def build_driver(cp, poles, kind, conductive, k_point, seed: int,
                 magnetic: bool = True, electric: bool = False):
    """One complex NO-PML conductive driver, seeded identically for every route.

    NO ``setup_pml``. This family is the no-PML branch on both halves, and its own
    predicate refuses an active absorber; a fixture that installed one would report
    a family failure that was really a fixture failure.
    """
    from meep_gpu.dispersion import Susceptibility  # noqa: PLC0415
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415

    driver = FdtdDriver(
        cell_size=(1.5, 1.5, 1.5), resolution=8.0, dimensions=3,
        force_complex_fields=True, courant=0.35,
        boundaries="periodic", k_point=k_point, prefer_gpu=True, gpu_id=0,
    )
    shape = driver.shape
    index = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    epsilon = np.ascontiguousarray(
        (1.45 + 0.30 * np.sin(index * np.float32(0.037))).astype(np.float32))
    driver.set_epsilon(cp.asarray(epsilon))

    # THE CONDUCTIVITY IS PER COMPONENT, and the mask is the point of the mixed
    # rows: `conductive_no_pml_targets` reads a per-target flag, and a component
    # with none compiles the SPLIT tail instead of the conductive one.
    if any(conductive):
        sigma = np.zeros(shape, dtype=np.float32)
        sigma[...] = np.float32(0.21) + np.float32(0.05) * np.sin(
            index * np.float32(0.013))
        volumes = {name: (cp.asarray(np.ascontiguousarray(sigma))
                          if flag else cp.zeros(shape, dtype=cp.float32))
                   for name, flag in zip(("Dx", "Dy", "Dz"), conductive)}
        driver.set_conductivity(volumes)

    kinds = ({0: "lorentzian", 1: "drude", 2: "lorentzian", 3: "drude"}
             if kind == "mixed" else {n: kind for n in range(poles)})
    for pole in range(poles):
        term = Susceptibility(frequency=0.7 + 0.21 * pole, gamma=0.02 * (pole + 1),
                              kind=kinds[pole])
        driver.add_susceptibility(term, 0.30 / (pole + 1))

    if magnetic:
        # A MAGNETIC source, injected in the B/H seam. It does NOT reach this seam,
        # which is exactly the polarity that makes this cell's four corpus rows
        # reachable with CARRIES_DEPOSIT_REPAIR at False — so the corpus shape is
        # driven rather than a quiet grid.
        driver.add_source({"component": "Hy", "frequency": 0.31,
                           "center": (0.0, 0.0, 0.0), "width": 0.4})
    if electric:
        driver.add_source({"component": "Ez", "frequency": 0.31,
                           "center": (0.0, 0.0, 0.0), "width": 0.4})

    rng = np.random.default_rng(seed)
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz"):
        host = (rng.uniform(-0.25, 0.25, size=shape)
                + 1j * rng.uniform(-0.25, 0.25, size=shape)).astype(np.complex64)
        driver.set_field(name, cp.asarray(np.ascontiguousarray(host)))
    return driver


#: Private ``Fields`` attributes that are HARNESS SCRATCH rather than state, and
#: are excluded from the compared inventory for a MEASURED reason.
#:
#: ``_fmp_scratch`` is allocated LAZILY by the array path's own ``update_E``, so a
#: route that substitutes that pass never creates it — and the two inventories then
#: differ by a NAME rather than by a word. Measured on the GPU host 2026-08-30: this
#: gate's first campaign reported ``{'reason': 'asymmetric', 'names':
#: ['_fmp_scratch']}`` as a byte divergence on all five product rows and broke the
#: step loop at step 1, so every row was scored over ONE step and none of them
#: measured what it exists to measure. THE ASYMMETRY IS STILL RECORDED —
#: ``inventory_asymmetry_vs_array`` carries it per row and ``verdict_of`` fails on a
#: non-empty list — it just no longer masquerades as a divergence.
#:
#: THE RULE IS THE LEADING UNDERSCORE, not this list: a private attribute is not
#: physical state, and every PUBLIC volume stays in the comparison.
PRIVATE_SCRATCH: Tuple[str, ...] = ("_fmp_scratch",)


def inventory(driver) -> Dict[str, Any]:
    fields = driver.fields
    shape = tuple(fields.grid.shape)
    found: Dict[str, Any] = {}
    for name, value in vars(fields).items():
        if name.startswith("_"):
            continue
        if (getattr(value, "shape", None) == shape
                and getattr(value, "dtype", None) is not None):
            found[name] = value
    _assert_material_names_are_real(found)
    missing = [name for name in REQUIRED if name not in found]
    if missing:
        raise AssertionError(
            f"the state scan lost {missing}; the comparison inventory is not "
            f"complete and every row below would pass for that reason")
    # AND THE CURL'S OWN SOURCES, by IDENTITY rather than by name. On this no-PML
    # branch the sub-step's source accessor returns the stored B arrays themselves,
    # so asking for "Hx" would be asking for a volume this configuration does not
    # have — while asking whether the scan CONTAINS the object the plan will read is
    # the question that actually protects the comparison.
    from meep_gpu.triton_kernels.complex_no_pml_curl import (  # noqa: PLC0415
        source_arrays)

    seen = {id(array) for array in found.values()}
    unscanned = [index for index, array in enumerate(source_arrays(fields, "step_D"))
                 if id(array) not in seen]
    if unscanned:
        raise AssertionError(
            f"the curl's source volumes at positions {unscanned} are not in the "
            f"compared state; a kernel that wrote through one would be invisible "
            f"to every row below")
    return found


def words(cp, array) -> np.ndarray:
    return np.ascontiguousarray(cp.asnumpy(array)).view(np.uint32).ravel()


def snapshot(cp, driver) -> Dict[str, np.ndarray]:
    return {name: words(cp, array) for name, array in inventory(driver).items()}


def polarization_snapshot(cp, driver) -> Dict[str, np.ndarray]:
    """The P volumes, which this kernel READS and must never write.

    ``update_P`` runs AFTER ``update_E`` and is not in this seam, so the pole
    arrays this kernel loads are the ones the previous timestep left. A kernel that
    wrote through a pole pointer would be caught here and nowhere else.

    ``PolarizationState`` holds them in the ``P`` and ``P_prev`` DICTS keyed by E
    component, not as attributes named ``Px``... — read here through those dicts,
    because a scan that guessed attribute names would come back EMPTY on every run
    and the read-only check would be inert. Measured on the GPU host 2026-08-30: it
    did, and the floor below is what said so instead of passing quietly.
    """
    out: Dict[str, np.ndarray] = {}
    for index, state in enumerate(getattr(driver.fields, "polarizations", ()) or ()):
        for label in ("P", "P_prev"):
            table = getattr(state, label, None)
            if not isinstance(table, dict):
                continue
            for component, array in table.items():
                if array is not None:
                    out[f"{index}:{label}:{component}"] = words(cp, array)
    return out


def first_divergence(left: Dict[str, np.ndarray],
                     right: Dict[str, np.ndarray]) -> Optional[Dict[str, Any]]:
    for name in sorted(set(left) & set(right)):
        a, b = left[name], right[name]
        if a.shape != b.shape:
            return {"array": name, "reason": "shape",
                    "left": list(a.shape), "right": list(b.shape)}
        if not np.array_equal(a, b):
            where = int(np.flatnonzero(a != b)[0])
            return {"array": name, "index": where,
                    "left_word": int(a[where]), "right_word": int(b[where]),
                    "differing_words": int(np.count_nonzero(a != b))}
    only = sorted(set(left) ^ set(right))
    return ({"array": "<inventory>", "reason": "asymmetric", "names": only}
            if only else None)


def moved(before: Dict[str, np.ndarray], after: Dict[str, np.ndarray]) -> List[str]:
    return [name for name in sorted(set(before) & set(after))
            if not np.array_equal(before[name], after[name])]


class CountingKernel:
    __slots__ = ("jit", "calls", "grids")

    def __init__(self, jit: Any) -> None:
        self.jit = jit
        self.calls = 0
        self.grids: List[Any] = []

    def __getitem__(self, grid):
        launcher = self.jit[grid]

        def run(*args, **kwargs):
            self.calls += 1
            self.grids.append(tuple(int(value) for value in grid))
            return launcher(*args, **kwargs)

        return run


class Route:
    def __init__(self, plans: Dict[str, Any]) -> None:
        self.plans = plans


class _Absorbed:
    __slots__ = ("name", "absorbed_by")

    def __init__(self, name: str, absorbed_by: Any) -> None:
        self.name = name
        self.absorbed_by = absorbed_by

    def run(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def install(driver_module, routes, counter: Dict[str, int], roles: Dict[int, str]):
    """Replace the named passes for the fields objects in ``routes``.

    INSTALLED ON ALL FIVE NAMES, not only the two this product replaces: the point
    of the counter is to record that the other three ran on the ARRAY PATH in the
    fused route too, which is what "the seam is empty" means operationally.
    """
    originals = {name: getattr(driver_module, name) for name in SEAM_PASSES}

    def bump(role: str, kind: str, name: str) -> None:
        key = f"{role}/{kind}:{name}"
        counter[key] = counter.get(key, 0) + 1

    def replacement(name):
        def wrapper(fields, *args):
            role = roles.get(id(fields), "unknown")
            plan = next((candidate for owner, candidate in routes
                         if fields is owner), None)
            if plan is None or name not in plan.plans:
                bump(role, "array_path", name)
                return originals[name](fields, *args)
            bump(role, "substituted", name)
            plan.plans[name].run()
            return None
        return wrapper

    for name in SEAM_PASSES:
        setattr(driver_module, name, replacement(name))
    return lambda: [setattr(driver_module, name, function)
                    for name, function in originals.items()]


def separate_route(driver, probe):
    """The two separately certified Triton products this launch replaces."""
    from meep_gpu.triton_kernels import (  # noqa: PLC0415
        complex_no_pml_conductive as conductive)
    from meep_gpu.triton_kernels import (  # noqa: PLC0415
        complex_no_pml_stored_e as stored)

    curl = conductive.plan_complex_conductive_no_pml_curl(
        driver.fields, driver.pml, "step_D", probe=probe)
    constitutive = stored.plan_complex_stored_e(driver.fields, driver.pml,
                                                probe=probe)
    missing = [name for name, plan in (("conductive curl", curl),
                                       ("complex stored E", constitutive))
               if plan is None]
    if missing:
        raise AssertionError(
            f"the separate oracle is incomplete: {missing} refused this case, so "
            f"this leg could not compare the fused launch against the products it "
            f"replaces")
    return Route({"step_D": curl, "update_E": constitutive})


def fused_route(plan):
    """The fused plan in its TWO slots. No bracket: this family carries no deposit,
    and the three passes it does not replace stay on the array path by NOT being
    named here."""
    return Route({"step_D": plan, "update_E": _Absorbed("update_E", plan)})


# ===========================================================================
# The three-route leg
# ===========================================================================

def run_leg(cp, name: str, case, product, probe, mutant: Any = None,
            steps: Optional[int] = None, electric: bool = False,
            expansion: Optional[int] = None) -> Dict[str, Any]:
    """Three routes in lockstep, per COMPLETE driver step; stop at the FIRST byte
    divergence."""
    import meep_gpu.driver as driver_module  # noqa: PLC0415

    (label, poles, kind, conductive, k_point, default_steps) = case
    budget = int(default_steps if steps is None else steps)
    built = [build_driver(cp, poles, kind, conductive, k_point, SEED,
                          electric=electric) for _ in range(3)]
    reference, separate, fused = built
    undo: Callable[[], Any] = lambda: None
    counter: Dict[str, int] = {}
    kernel = CountingKernel(
        mutant if mutant is not None
        else product.complex_conductive_fused_curl_constitutive_D_kernel())
    row: Dict[str, Any] = {
        "leg": name, "device": True, "case": label, "steps_budget": budget,
        "shape": list(reference.shape), "poles": poles, "kind": kind,
        "conductive": list(conductive), "k_point": list(k_point),
        "electric_source_in_the_seam": bool(electric),
        "expansion_override": expansion,
        "first_divergence": None, "control_divergence": None,
    }
    try:
        plan = product.plan_complex_conductive_fused_pair(
            fused.fields, fused.pml, tuple(fused._sources), num_warps=1,
            kernel=kernel, probe=probe)
        if plan is None:
            verdict = product.complex_conductive_fused_pair_coverage(
                fused.fields, fused.pml, tuple(fused._sources), probe=probe)
            raise AssertionError(f"the product refused the case: {verdict.reasons}")
        if expansion is not None:
            plan.expansion = int(expansion)
        row["plan"] = repr(plan)
        row["plan_replaces"] = list(plan.replaces)
        row["licensed_expansion"] = plan.expansion
        # THE REPLACES CLAIM IS ASSERTED, not trusted: a product that claimed five
        # call sites here while carrying none of the three in between would be
        # licensing an inline pass it does not perform.
        if tuple(plan.replaces) != ABSORBED:
            raise AssertionError(
                f"this product declares replaces={tuple(plan.replaces)}; the gate "
                f"installs and asserts exactly {ABSORBED}, and a claim to more "
                f"would be a claim to carry passes this family refuses")

        route = fused_route(plan)
        separate_plans = separate_route(separate, probe)
        row["slot_classes"] = {key: type(value).__name__
                               for key, value in route.plans.items()}
        row["separate_products"] = {key: type(value).__name__
                                    for key, value in separate_plans.plans.items()}

        roles = {id(reference.fields): "array", id(separate.fields): "separate",
                 id(fused.fields): "fused"}
        routes = [(separate.fields, separate_plans), (fused.fields, route)]
        undo = install(driver_module, routes, counter, roles)

        opening = snapshot(cp, fused)
        opening_poles = polarization_snapshot(cp, fused)
        row["per_step"] = []
        started = time.time()
        for step in range(1, budget + 1):
            before = snapshot(cp, fused)
            reference.step()
            separate.step()
            fused.step()
            cp.cuda.runtime.deviceSynchronize()
            after_reference = snapshot(cp, reference)
            after_separate = snapshot(cp, separate)
            after_fused = snapshot(cp, fused)
            versus_array = first_divergence(after_fused, after_reference)
            versus_separate = first_divergence(after_fused, after_separate)
            control = first_divergence(after_separate, after_reference)
            step_moved = moved(before, after_fused)
            row["per_step"].append({
                "step": step,
                "fused_vs_array": versus_array,
                "fused_vs_separate_certified_products": versus_separate,
                "oracle_control_vs_array": control,
                "arrays_moved": len(step_moved),
                "fused_launches": kernel.calls,
            })
            divergence = versus_array or versus_separate
            log(f"  {name}/{label} step {step}/{budget} "
                f"identical={divergence is None} control={control is None} "
                f"moved={len(step_moved)} launches={kernel.calls} "
                f"({time.time() - started:.1f} s)")
            row["first_divergence"] = divergence
            row["control_divergence"] = control
            if divergence is not None:
                row["diverged_at_step"] = step
                break

        final = snapshot(cp, fused)
        ever_moved = moved(opening, final)
        material = [key for key in final if key in MATERIAL]
        row["arrays_total"] = len(final)
        row["arrays_compared"] = sorted(final)
        row["arrays_ever_moved"] = len(ever_moved)
        row["arrays_never_moved"] = sorted(
            set(final) - set(ever_moved) - set(material))
        row["material_changed"] = sorted(key for key in material
                                         if key in ever_moved)
        # THE ASYMMETRY IS RECORDED, not compared through. See PRIVATE_SCRATCH.
        row["inventory_asymmetry_vs_array"] = sorted(
            set(final).symmetric_difference(snapshot(cp, reference)))
        row["private_scratch_on_the_array_route"] = sorted(
            name for name in vars(reference.fields) if name in PRIVATE_SCRATCH)
        row["private_scratch_on_the_fused_route"] = sorted(
            name for name in vars(fused.fields) if name in PRIVATE_SCRATCH)
        # THE POLE ARRAYS ARE READ, NEVER WRITTEN, BY THIS SEAM. `update_P` runs
        # after `update_E` and is outside it, so the pole volumes MUST have moved
        # (the run is live) and the fused route's must equal the array path's.
        row["polarization_volumes_seen"] = len(opening_poles)
        row["polarization_divergence"] = first_divergence(
            polarization_snapshot(cp, fused), polarization_snapshot(cp, reference))
        row["launches"] = dict(counter)
        row["fused_kernel_launches"] = kernel.calls
        row["launch_grids"] = sorted({grid for grid in kernel.grids})
        row["elapsed_seconds"] = time.time() - started
        return row
    finally:
        undo()
        for target in built:
            try:
                target.close()
            except Exception:  # noqa: BLE001
                pass
        cp.get_default_memory_pool().free_all_blocks()


def verdict_of(row: Dict[str, Any], *, require_identical: bool = True,
               require_launches: Optional[int] = None,
               require_moved: bool = True) -> Tuple[bool, List[str]]:
    failures: List[str] = []
    if require_identical and row.get("first_divergence") is not None:
        failures.append(f"byte divergence: {row['first_divergence']}")
    if row.get("control_divergence") is not None:
        failures.append(
            f"an ORACLE control itself diverged from the array path, so this leg "
            f"could not have measured the fused launch: {row['control_divergence']}")
    if row.get("polarization_divergence") is not None:
        failures.append(
            f"the P volumes diverged from the array path's: this seam READS them "
            f"and must never write one: {row['polarization_divergence']}")
    if require_identical and not row.get("polarization_volumes_seen"):
        failures.append(
            "no polarization volume was found, so the read-only pole check is "
            "inert and the pole-chain arms this row scores may not be compiled")
    if (require_launches is not None
            and row.get("fused_kernel_launches") != require_launches):
        failures.append(
            f"the fused kernel launched {row.get('fused_kernel_launches')} times, "
            f"expected {require_launches}: the fused path is not what executed")
    if require_moved and (row.get("arrays_never_moved") or []):
        failures.append(
            f"VACUOUS: these arrays never moved: {row['arrays_never_moved']}")
    if row.get("material_changed"):
        failures.append(f"a material input changed: {row['material_changed']}")
    if row.get("inventory_asymmetry_vs_array"):
        failures.append(
            f"the compared inventories differ by NAME between the fused route and "
            f"the array path: {row['inventory_asymmetry_vs_array']}. A volume one "
            f"route has and the other does not is not comparable, and every "
            f"per-word verdict above was taken over the intersection")
    fell_back = {key: value for key, value in (row.get("launches") or {}).items()
                 if key.startswith("fused/array_path:")
                 and key.rsplit(":", 1)[1] in ABSORBED}
    if fell_back:
        failures.append(
            f"the fused route reached the array path for an ABSORBED pass: "
            f"{fell_back}")
    # ...and the THREE it does NOT absorb must have run on the array path, on
    # every step. A run in which they did not is a run in which this gate's
    # "the seam is empty" claim was never exercised.
    for name in SEAM_PASSES:
        if name in ABSORBED:
            continue
        if not (row.get("launches") or {}).get(f"fused/array_path:{name}"):
            failures.append(
                f"the un-absorbed pass {name} never ran on the fused route's array "
                f"path; this row did not exercise the seam this family declines to "
                f"carry")
    return (not failures), failures


# ===========================================================================
# Armed source mutations
# ===========================================================================

def shipped_block() -> str:
    """The shipped kernel definitions, read out of the FILE rather than re-typed.

    BOTH the pole helper and the kernel: the helper is the ONE line this weld
    rewrites, so a needle in it is a needle in the seam itself.
    """
    with open(os.path.join(API_ROOT, FAMILY_MODULE), encoding="utf-8") as handle:
        text = handle.read()
    start = text.index("@triton.jit\ndef _subtract_complex_poles_in_registers(")
    end = text.index("def complex_conductive_fused_curl_constitutive_D_kernel(")
    return text[start:end].rstrip() + "\n"


def compile_mutant(source: str, entry: str):
    """Compile one mutated kernel pair from a REAL FILE."""
    header = (
        "import triton\n"
        "import triton.language as tl\n"
        "from meep_gpu.triton_kernels.complex_fields import (\n"
        "    METALLIC, _mul_coefficient_left, _mul_field_left, _rotate_field_left)\n"
        "\n")
    handle = tempfile.NamedTemporaryFile(
        "w", suffix="_mutated_condfused.py", delete=False, encoding="utf-8")
    handle.write(header + source)
    handle.close()
    _TEMPORARY.append(handle.name)
    spec = importlib.util.spec_from_file_location(
        "triton_mutated_condfused_" + str(len(_TEMPORARY)), handle.name)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = module  # type: ignore[union-attr]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return getattr(module, entry)


def _renamed(source: str, suffix: str) -> Tuple[str, str]:
    """Rename BOTH definitions. The kernel calls the helper by name, so renaming
    only the kernel would have the mutant call the SHIPPED helper — which silently
    disarms every needle placed in the pole chain."""
    name = f"complex_conductive_fused_curl_constitutive_D__{suffix}"
    helper = f"_subtract_complex_poles_in_registers__{suffix}"
    renamed = source.replace("_subtract_complex_poles_in_registers", helper)
    renamed = renamed.replace(
        "def complex_conductive_fused_curl_constitutive_D(", f"def {name}(", 1)
    return renamed, name


MUTATIONS: Dict[str, Tuple[str, str, str, str]] = {
    # ---- THE SEAM ITSELF: the one substitution this weld makes -------------
    "seam_reloads_D_instead_of_taking_the_register":
        (MUTATION_CASE,
         "    s0_re, s0_im = _subtract_complex_poles_in_registers(\n"
         "        v0_re, v0_im, a0, a1, a2, a3, a4, a5, a6, a7, word, live, NP0)",
         "    s0_re, s0_im = _subtract_complex_poles_in_registers(\n"
         "        tl.load(f0 + word, mask=live, other=0.0),\n"
         "        tl.load(f0 + word + 1, mask=live, other=0.0),\n"
         "        a0, a1, a2, a3, a4, a5, a6, a7, word, live, NP0)",
         # A RELOAD OF THE VOLUME THE SAME LAUNCH JUST WROTE. On this kernel the
         # store precedes the constitutive half, so the reload reads the same word
         # the register holds — for the lane that wrote it. It is armed as a NULL
         # with that reason, because the interesting fact is that the D STORE IS
         # KEPT: a weld that dropped it would break the next timestep's update_P,
         # and the `displacement_store_dropped` needle below is what measures that.
         "null"),
    "seam_takes_the_wrong_component":
        (MUTATION_CASE,
         "        v1_re, v1_im, b0, b1, b2, b3, b4, b5, b6, b7, word, live, NP1)",
         "        v0_re, v0_im, b0, b1, b2, b3, b4, b5, b6, b7, word, live, NP1)",
         "catch"),
    "seam_takes_the_pre_conductive_curl":
        (MUTATION_CASE,
         "    s2_re, s2_im = _subtract_complex_poles_in_registers(\n"
         "        v2_re, v2_im, q0, q1, q2, q3, q4, q5, q6, q7, word, live, NP2)",
         "    s2_re, s2_im = _subtract_complex_poles_in_registers(\n"
         "        curl2_re, curl2_im, q0, q1, q2, q3, q4, q5, q6, q7, word, live, NP2)",
         "catch"),
    # ---- THE POLE CHAIN: the order is bit-load-bearing ---------------------
    "pole_chain_order_reversed":
        (MUTATION_CASE,
         "    if NP > 0:\n"
         "        real = real - tl.load(p0 + word, mask=live, other=0.0)\n"
         "        imag = imag - tl.load(p0 + word + 1, mask=live, other=0.0)\n"
         "    if NP > 1:\n"
         "        real = real - tl.load(p1 + word, mask=live, other=0.0)\n"
         "        imag = imag - tl.load(p1 + word + 1, mask=live, other=0.0)\n"
         "    if NP > 2:\n"
         "        real = real - tl.load(p2 + word, mask=live, other=0.0)\n"
         "        imag = imag - tl.load(p2 + word + 1, mask=live, other=0.0)\n"
         "    if NP > 3:\n"
         "        real = real - tl.load(p3 + word, mask=live, other=0.0)\n"
         "        imag = imag - tl.load(p3 + word + 1, mask=live, other=0.0)\n",
         "    if NP > 3:\n"
         "        real = real - tl.load(p3 + word, mask=live, other=0.0)\n"
         "        imag = imag - tl.load(p3 + word + 1, mask=live, other=0.0)\n"
         "    if NP > 2:\n"
         "        real = real - tl.load(p2 + word, mask=live, other=0.0)\n"
         "        imag = imag - tl.load(p2 + word + 1, mask=live, other=0.0)\n"
         "    if NP > 1:\n"
         "        real = real - tl.load(p1 + word, mask=live, other=0.0)\n"
         "        imag = imag - tl.load(p1 + word + 1, mask=live, other=0.0)\n"
         "    if NP > 0:\n"
         "        real = real - tl.load(p0 + word, mask=live, other=0.0)\n"
         "        imag = imag - tl.load(p0 + word + 1, mask=live, other=0.0)\n",
         "catch"),
    "pole_chain_drops_the_last_arm":
        (MUTATION_CASE,
         "    if NP > 3:\n"
         "        real = real - tl.load(p3 + word, mask=live, other=0.0)\n"
         "        imag = imag - tl.load(p3 + word + 1, mask=live, other=0.0)\n",
         "", "catch"),
    "pole_chain_subtracts_the_imaginary_from_the_real":
        (MUTATION_CASE,
         "        imag = imag - tl.load(p0 + word + 1, mask=live, other=0.0)\n",
         "        imag = imag - tl.load(p0 + word, mask=live, other=0.0)\n",
         "catch"),
    # ---- THE inv_eps MULTIPLY ---------------------------------------------
    "inv_eps_word_doubled":
        (MUTATION_CASE,
         "        s0_re, s0_im, tl.load(iv0 + idx, mask=live, other=0.0), EXPANSION)",
         "        s0_re, s0_im, tl.load(iv0 + word, mask=live, other=0.0), EXPANSION)",
         "catch"),
    "inv_eps_applied_before_the_poles":
        (MUTATION_CASE,
         "    s1_re, s1_im = _subtract_complex_poles_in_registers(\n"
         "        v1_re, v1_im, b0, b1, b2, b3, b4, b5, b6, b7, word, live, NP1)\n"
         "    o1_re, o1_im = _mul_field_left(\n"
         "        s1_re, s1_im, tl.load(iv1 + idx, mask=live, other=0.0), EXPANSION)\n",
         "    p1_re, p1_im = _mul_field_left(\n"
         "        v1_re, v1_im, tl.load(iv1 + idx, mask=live, other=0.0), EXPANSION)\n"
         "    o1_re, o1_im = _subtract_complex_poles_in_registers(\n"
         "        p1_re, p1_im, b0, b1, b2, b3, b4, b5, b6, b7, word, live, NP1)\n",
         "catch"),
    # ---- THE CONDUCTIVE TAIL ----------------------------------------------
    # THE CONDUCTIVE TAIL is `D *= condfac; D -= curl; D *= condinv`, three in-place
    # complex64 operations that each round before the next begins. Handing the tail
    # its own inverse where the factor belongs collapses the pair.
    "conductive_factor_and_inverse_swapped":
        (MUTATION_CASE,
         "    if COND0:\n"
         "        factor = tl.load(cf0 + idx, mask=live, other=0.0)\n",
         "    if COND0:\n"
         "        factor = tl.load(ci0 + idx, mask=live, other=0.0)\n",
         "catch"),
    # ...and the ORDER of the three is load-bearing: scaling after the subtraction
    # is a different float32 rounding sequence, not an algebraic rearrangement.
    "conductive_factor_applied_after_the_subtraction":
        (MUTATION_CASE,
         "    if COND1:\n"
         "        factor = tl.load(cf1 + idx, mask=live, other=0.0)\n"
         "        v1_re, v1_im = _mul_field_left(v1_re, v1_im, factor, EXPANSION)\n"
         "    v1_re = v1_re - curl1_re\n"
         "    v1_im = v1_im - curl1_im\n",
         "    v1_re = v1_re - curl1_re\n"
         "    v1_im = v1_im - curl1_im\n"
         "    if COND1:\n"
         "        factor = tl.load(cf1 + idx, mask=live, other=0.0)\n"
         "        v1_re, v1_im = _mul_field_left(v1_re, v1_im, factor, EXPANSION)\n",
         "catch"),
    # ---- THE CERTIFIED COMPLEX BASE ----------------------------------------
    "curl_parens_flattened":
        (MUTATION_CASE,
         "    t1_re = ((a_z_re - a_re) + (c_re - c_x_re))",
         "    t1_re = (a_z_re - a_re + c_re - c_x_re)", "catch"),
    "bloch_rotation_applied_to_every_lane":
        (MUTATION_CASE,
         "        b_x_re = tl.where(wx, rot_re, b_x_re)",
         "        b_x_re = rot_re", "catch"),
    "displacement_store_dropped":
        (MUTATION_CASE,
         "    tl.store(f0 + 2 * idx, v0_re, mask=live)\n", "", "catch"),
    "constitutive_store_dropped":
        (MUTATION_CASE,
         "    tl.store(h0 + word, o0_re, mask=live)\n", "", "catch"),
}

NULL_REASONS: Dict[str, str] = {
    "seam_reloads_D_instead_of_taking_the_register":
        "the D STORE IS KEPT by this weld — the array path leaves the stepped "
        "displacement in D and the next timestep's update_P reads it — and the "
        "store precedes the constitutive half, so a reload reads back exactly the "
        "word the register holds. The edit is therefore byte-invisible HERE and is "
        "armed to record that, not to license dropping the register carry: the "
        "`displacement_store_dropped` row is what measures that the store matters.",
}


GUARD_RULES: Dict[str, Callable[[Dict[str, Any]], bool]] = {
    # BACKWARD is bound to 1 by every builder (the product IS the D seam).
    "if BACKWARD:": lambda case: True,
    # The predicate refuses METALLIC on every axis, so these arms are dead on every
    # admitted configuration and no needle may be armed inside one.
    "if BCX == METALLIC:": lambda case: False,
    "if BCY == METALLIC:": lambda case: False,
    "if BCZ == METALLIC:": lambda case: False,
    "if PHX:": lambda case: case["k_point"][0] != 0.0,
    "if PHY:": lambda case: case["k_point"][1] != 0.0,
    "if PHZ:": lambda case: case["k_point"][2] != 0.0,
    "if COND0:": lambda case: case["conductive"][0],
    "if COND1:": lambda case: case["conductive"][1],
    "if COND2:": lambda case: case["conductive"][2],
    "if NP > 0:": lambda case: case["poles"] > 0,
    "if NP > 1:": lambda case: case["poles"] > 1,
    "if NP > 2:": lambda case: case["poles"] > 2,
    "if NP > 3:": lambda case: case["poles"] > 3,
    "if NP > 4:": lambda case: case["poles"] > 4,
    "if NP > 5:": lambda case: case["poles"] > 5,
    "if NP > 6:": lambda case: case["poles"] > 6,
    "if NP > 7:": lambda case: case["poles"] > 7,
}


def _first_line(needle: str) -> str:
    for line in needle.splitlines():
        if line.strip():
            return line
    raise AssertionError(f"empty needle: {needle!r}")


def guard_chain(block: str, needle: str) -> List[str]:
    """Every ``if`` / ``else`` header ENCLOSING the needle, outermost first."""
    head = _first_line(needle)
    position = block.index(needle)
    prefix = block[:position].splitlines()
    want = len(head) - len(head.lstrip())
    chain: List[str] = []
    index = len(prefix) - 1
    while index >= 0:
        line = prefix[index]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            index -= 1
            continue
        indent = len(line) - len(line.lstrip())
        if indent < want:
            if stripped.startswith("else:"):
                partner = index - 1
                while partner >= 0:
                    candidate = prefix[partner]
                    text = candidate.strip()
                    if text and not text.startswith("#"):
                        depth = len(candidate) - len(candidate.lstrip())
                        if depth == indent and text.startswith(("if ", "elif ")):
                            chain.append(f"else of {text}")
                            break
                        if depth < indent:
                            chain.append("else of <unresolved>")
                            break
                    partner -= 1
                else:
                    chain.append("else of <unresolved>")
                want = indent
            elif stripped.startswith(("if ", "elif ")):
                chain.append(stripped)
                want = indent
            else:
                want = indent
        index -= 1
    chain.reverse()
    return [entry for entry in chain if not entry.startswith("def ")]


def guard_chain_holds(chain: Sequence[str],
                      case: Dict[str, Any]) -> Optional[bool]:
    for entry in chain:
        negate = entry.startswith("else of ")
        condition = entry[len("else of "):] if negate else entry
        rule = GUARD_RULES.get(condition)
        if rule is None:
            return None
        value = bool(rule(case))
        if negate:
            value = not value
        if not value:
            return False
    return True


def leg_needle_reachability() -> Dict[str, Any]:
    """Every needle must be PRESENT, and its ENCLOSING GUARDS must hold on its case."""
    block = shipped_block()
    cases = {case[0]: case for case in CASES}
    rows: List[Dict[str, Any]] = []
    for name, (case_name, old, _new, expectation) in MUTATIONS.items():
        case = cases[case_name]
        context = {"poles": case[1], "conductive": case[3], "k_point": case[4]}
        present = old in block
        chain = guard_chain(block, old) if present else []
        holds = guard_chain_holds(chain, context) if present else None
        # A NEEDLE THAT SWALLOWS ITS OWN ``if`` HAS NO ENCLOSING CHAIN, and the
        # reachability question moves INSIDE it. `pole_chain_order_reversed`
        # rewrites four `if NP > n:` arms and `conductive_*` rewrite `if COND*:`
        # ones; scored on a case that does not compile them, each would rewrite
        # unreached lines and report UNCAUGHT. So the guards the needle CONTAINS
        # are evaluated too, by the same fail-closed table.
        contained = [line.strip() for line in old.splitlines()
                     if line.strip().startswith("if ")]
        inner = guard_chain_holds(contained, context) if present else None
        rows.append({
            "mutation": name, "case": case_name, "expectation": expectation,
            "needle_present": present,
            "needle_occurrences": block.count(old) if present else 0,
            "enclosing_guards": chain,
            "guards_inside_the_needle": contained,
            "case_compiles_the_guards_inside_the_needle": inner,
            "case_compiles_every_guard": (
                None if holds is None or inner is None else bool(holds and inner)),
            "case_poles": context["poles"],
            "case_conductive": list(context["conductive"])})
    markers = {condition: (condition in block) for condition in GUARD_RULES}
    ok = all(entry["needle_present"] and entry["case_compiles_every_guard"] is True
             for entry in rows)
    return {"passed": bool(ok and all(markers.values())),
            "rows": rows,
            "guard_conditions_present_in_the_shipped_source": markers,
            "guard_rules_modelled": sorted(GUARD_RULES),
            "note": "the three BC arms are modelled as permanently False: the "
                    "predicate refuses a metallic axis on every one, so a needle "
                    "placed inside one FAILS this leg rather than reporting "
                    "UNCAUGHT downstream."}


def leg_mutants_compile() -> Dict[str, Any]:
    """EVERY MUTANT MUST PARSE, and this is checked on the laptop.

    A needle that deletes the whole body of an ``if`` leaves an empty block, and the
    mutant is then an ``IndentationError`` rather than a defect. MEASURED ON
    THE GPU HOST 2026-08-30: the cylindrical twin's ``m_one_axis_clear_dropped`` did exactly that and killed
    that campaign 146 seconds in, AFTER thirteen mutations had already been scored —
    so the cost of finding it on a device is the whole run, and the cost of finding
    it here is a syntax parse. This family's needles are checked by the same leg
    rather than trusted because none of them happens to delete a block today.

    THIS IS NOT A SUBSTITUTE FOR THE DEVICE LEG. A mutant that parses may still be
    unreachable, byte-invisible or wrong about what it edits; those are what
    ``needle_reachability`` and the mutation rows measure. This leg answers one
    question only, and it answers it before a GPU is asked for.
    """
    import ast  # noqa: PLC0415

    block = shipped_block()
    rows: List[Dict[str, Any]] = []
    for name, (_case, old, new, _expectation) in MUTATIONS.items():
        if old not in block:
            rows.append({"mutation": name, "parsed": False,
                         "error": "needle absent from the shipped source"})
            continue
        mutated, _entry = _renamed(block.replace(old, new, 1), name)
        try:
            ast.parse(mutated)
            rows.append({"mutation": name, "parsed": True})
        except SyntaxError as exc:  # noqa: PERF203 - one row per mutation
            rows.append({"mutation": name, "parsed": False,
                         "error": f"{type(exc).__name__}: {exc}"})
    # AND THE PRISTINE BLOCK ITSELF, so a leg that reported every mutant broken
    # because the SHIPPED text stopped parsing says which it was.
    try:
        ast.parse(block)
        pristine = True
    except SyntaxError as exc:  # noqa: BLE001
        pristine = f"{type(exc).__name__}: {exc}"
    return {"passed": bool(pristine is True
                           and all(row["parsed"] for row in rows)),
            "pristine_block_parses": pristine,
            "mutants": len(rows),
            "rows": [row for row in rows if not row["parsed"]] or rows[:0],
            "note": "one syntax parse per armed mutant; a mutant that does not "
                    "parse arms nothing and takes the device run down with it."}


def leg_transcription() -> Dict[str, Any]:
    """The two halves must be the certified kernels' own lines, VERBATIM."""
    fused = shipped_block()

    def read(relative: str) -> str:
        with open(os.path.join(API_ROOT, relative), encoding="utf-8") as handle:
            return handle.read()

    curl = read("meep_gpu/triton_kernels/complex_no_pml_conductive.py")
    stored = read("meep_gpu/triton_kernels/complex_no_pml_stored_e.py")

    def statements(text: str, start: str, end: str) -> List[str]:
        block = text[text.index(start):text.index(end)]
        out: List[str] = []
        pending = ""
        depth = 0
        for line in block.splitlines():
            stripped = line.strip()
            if not depth and (not stripped or stripped.startswith("#")
                              or stripped.startswith('"')):
                continue
            pending = (pending + " " + stripped).strip() if pending else stripped
            depth += stripped.count("(") - stripped.count(")")
            if depth <= 0:
                out.append(" ".join(pending.split()))
                pending, depth = "", 0
        if pending:
            out.append(" ".join(pending.split()))
        return out

    curl_lines = statements(
        curl, "    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)",
        "def complex_conductive_no_pml_curl_coverage(")
    fused_statements = set(statements(
        fused, "    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)",
        "    # ==================== the constitutive half ===="))
    curl_missing = [line for line in curl_lines if line not in fused_statements]

    # THE POLE HELPER, statement for statement against the certified one. Its two
    # OPENING LOADS are the only lines the weld removes, so they are the only ones
    # allowed to be missing — and they are named rather than tolerated.
    # The END anchor carries the DECORATOR of the next definition, so the certified
    # scan stops before it. Ending at the bare ``def`` would put a stray
    # ``@triton.jit`` in the certified statement list, which is then "missing" from
    # the weld on every run — a leg that fails for a reason of its own.
    helper = statements(
        stored, "def _subtract_complex_poles(",
        "@triton.jit\ndef complex_stored_e_step(")
    weld = set(statements(
        fused, "def _subtract_complex_poles_in_registers(",
        "@triton.jit\ndef complex_conductive_fused_curl_constitutive_D("))
    helper_missing = [line for line in helper if line not in weld]
    allowed = [line for line in helper_missing
               if line.startswith("real = tl.load(")
               or line.startswith("imag = tl.load(")
               or line.startswith("def _subtract_complex_poles(")
               or "source" in line]
    unexpected = [line for line in helper_missing if line not in allowed]
    return {
        "passed": bool(not curl_missing and not unexpected),
        "certified_curl_statements": len(curl_lines),
        "curl_statements_missing_from_the_fused_body": curl_missing,
        "certified_pole_helper_statements": len(helper),
        "pole_helper_lines_the_weld_removes": allowed,
        "pole_helper_lines_missing_for_no_declared_reason": unexpected,
        "note": "parsed out of the shipped files; nothing here re-implements the "
                "kernel's arithmetic, which is what stops a planted defect from "
                "being mirrored instead of executed.",
    }


def leg_refusal(cp, probe) -> Dict[str, Any]:
    """The source seam, the wall refusal and the absorber refusal, BY NAME."""
    from meep_gpu.sources import ContinuousEnvelope, VolumeSource  # noqa: PLC0415
    from meep_gpu.triton_kernels import (  # noqa: PLC0415
        complex_conductive_fused_pair as product)

    case = CASES[0]
    driver = build_driver(cp, case[1], case[2], case[3], case[4], SEED)
    try:
        fields, pml = driver.fields, driver.pml
        undeclared = product.complex_conductive_fused_pair_coverage(
            fields, pml, None, probe=probe)
        magnetic_only = product.complex_conductive_fused_pair_coverage(
            fields, pml, tuple(driver._sources), probe=probe)
        electric = VolumeSource(grid=fields.grid, component="Ez",
                                center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                                envelope=ContinuousEnvelope(frequency=1.0))
        electric_cov = product.complex_conductive_fused_pair_coverage(
            fields, pml, (electric,), probe=probe)
        electric_plan = product.plan_complex_conductive_fused_pair(
            fields, pml, sources=(electric,), probe=probe)
        no_probe = product.complex_conductive_fused_pair_coverage(
            fields, pml, tuple(driver._sources), probe=None)
    finally:
        driver.close()

    # THE FAMILY BOUNDARY. With no conductivity on any target the curl half refuses
    # by name and a DIFFERENT certified product owns the slot. Measured here rather
    # than assumed, because a fused pair that admitted it would be stepping a
    # configuration whose certified arithmetic is not the one it compiles.
    lossless = build_driver(cp, LOSSLESS_REFUSAL_CASE[1], LOSSLESS_REFUSAL_CASE[2],
                            LOSSLESS_REFUSAL_CASE[3], LOSSLESS_REFUSAL_CASE[4],
                            SEED)
    try:
        lossless_cov = product.complex_conductive_fused_pair_coverage(
            lossless.fields, lossless.pml, tuple(lossless._sources), probe=probe)
        lossless_plan = product.plan_complex_conductive_fused_pair(
            lossless.fields, lossless.pml, sources=tuple(lossless._sources),
            probe=probe)
    finally:
        lossless.close()

    return {
        "passed": bool(not undeclared.covered
                       and any("was not declared" in r for r in undeclared.reasons)
                       and magnetic_only.covered
                       and not electric_cov.covered
                       and any("is electric" in r and "driver.py:3294" in r
                               for r in electric_cov.reasons)
                       and electric_plan is None
                       and not no_probe.covered
                       and not lossless_cov.covered
                       and any("no curl target carries a conductivity" in r
                               for r in lossless_cov.reasons)
                       and lossless_plan is None),
        "undeclared_sources_refused": not undeclared.covered,
        "undeclared_named": [r for r in undeclared.reasons
                             if "was not declared" in r],
        "the_corpus_shape_is_admitted": magnetic_only.covered,
        "magnetic_only_reasons": list(magnetic_only.reasons),
        "electric_source_refused": not electric_cov.covered,
        "electric_plan_is_none": electric_plan is None,
        "electric_named": [r for r in electric_cov.reasons if "is electric" in r],
        "missing_expansion_probe_refused": not no_probe.covered,
        "a_lossless_row_is_refused_to_the_other_family": not lossless_cov.covered,
        "lossless_named": [r for r in lossless_cov.reasons
                           if "conductivity" in r][:2],
        "lossless_plan_is_none": lossless_plan is None,
        "note": "the electric refusal costs this cell ZERO corpus rows — all four "
                "declare a MAGNETIC source only — which is why it is measured here "
                "rather than left to the corpus.",
    }


# ===========================================================================
# Main
# ===========================================================================

def probe_record(explicit: Optional[str]) -> Tuple[Optional[Dict[str, Any]],
                                                   List[str], Optional[str]]:
    candidates = ([explicit] if explicit else
                  [os.path.join(API_ROOT, relative) for relative in KEEP_PROBES])
    reasons: List[str] = []
    for path in candidates:
        if not path or not os.path.isfile(path):
            reasons.append(f"probe artifact absent: {path}")
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                record = json.load(handle)
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"probe artifact unreadable ({path}): {exc}")
            continue
        if not isinstance(record, dict):
            reasons.append(f"probe artifact is not a verdict dict: {path}")
            continue
        return record, reasons, path
    return None, reasons, None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--probe-artifact", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--subnormal-policy", default="keep",
                        choices=("keep", "flush"))
    args = parser.parse_args(list(argv) if argv is not None else None)
    out = args.out
    os.makedirs(out, exist_ok=True)
    started = time.time()

    results: Dict[str, Any] = {
        "gate": "triton_complex_conductive_fused_pair",
        "family_module": FAMILY_MODULE,
        "cases": [[case[0], case[1], case[2], list(case[3]), list(case[4]),
                   case[5]] for case in CASES],
        "refusal_case": [LOSSLESS_REFUSAL_CASE[0], LOSSLESS_REFUSAL_CASE[1],
                         LOSSLESS_REFUSAL_CASE[2],
                         list(LOSSLESS_REFUSAL_CASE[3]),
                         list(LOSSLESS_REFUSAL_CASE[4]),
                         LOSSLESS_REFUSAL_CASE[5]],
        "seam_passes": list(SEAM_PASSES),
        "absorbed_passes": list(ABSORBED),
        "subnormal_policy": args.subnormal_policy,
        "rows": [], "skipped": {}, "failures": [],
    }

    def flush() -> None:
        try:
            from gate_provenance import stamp as _stamp  # noqa: PLC0415
            _stamp(results)
        except Exception:  # noqa: BLE001
            pass
        with open(os.path.join(out, "gate.json"), "w", encoding="utf-8") as handle:
            handle.write(json.dumps(results, indent=2, sort_keys=True,
                                    default=str) + "\n")

    results["rows"].append({"leg": "needle_reachability", "device": False,
                            **leg_needle_reachability()})
    log(f"[needle_reachability] passed={results['rows'][-1]['passed']}")
    flush()
    results["rows"].append({"leg": "transcription", "device": False,
                            **leg_transcription()})
    log(f"[transcription] passed={results['rows'][-1]['passed']}")
    flush()
    results["rows"].append({"leg": "mutants_compile", "device": False,
                            **leg_mutants_compile()})
    log(f"[mutants_compile] passed={results['rows'][-1]['passed']}")
    flush()

    try:
        import cupy as cp  # noqa: PLC0415
        import triton  # noqa: F401, PLC0415
        device_available, why = True, None
    except Exception as exc:  # noqa: BLE001
        cp, device_available, why = None, False, f"{type(exc).__name__}: {exc}"

    if not device_available:
        results["skipped"]["device"] = (
            f"{why} — the device legs run on the measurement machine; NOTHING was "
            f"measured here")
        log(f"[device] SKIPPED cleanly: {results['skipped']['device']}")
        results["verdict"] = "SKIPPED_DEVICE"
        # NO DEVICE, NO RELEASE, and all three of the board's conditions fail by
        # construction rather than by omission. A laptop run measured no bytes and
        # must never be readable as one that did.
        results["device_status"] = "NO_DEVICE"
        results["passed"] = False
        results["release"] = {"released": False,
                              "reasons": ["no device: the laptop legs above ran, "
                                          "the byte legs did not"]}
        results["planted_defect"] = False
        results["subnormal_policy"] = {"policy": None,
                                       "stamp": "not installed: no device"}
        results["elapsed_seconds"] = time.time() - started
        flush()
        return 0

    import gate_triton_complex as shared  # noqa: PLC0415
    results["policy_stamp"] = shared.install_ftz_strip(args.subnormal_policy)
    flush()

    record, reasons, probe_path = probe_record(args.probe_artifact)
    if record is None:
        results["failures"].extend(reasons)
        results["verdict"] = "REFUSED"
        results["elapsed_seconds"] = time.time() - started
        flush()
        log("REFUSED: " + "; ".join(reasons))
        return 1
    results["probe_artifact"] = probe_path
    results["probe_sha256"] = sha256_of(probe_path)
    results["probe_policy_reasons"] = list(
        shared.probe_record_policy_reasons(record))

    from meep_gpu.triton_kernels import complex_fields as cx  # noqa: PLC0415
    from meep_gpu.triton_kernels import (  # noqa: PLC0415
        complex_conductive_fused_pair as product)
    expansion = cx._resolve_expansion(record)
    results["licensed_expansion"] = expansion
    if expansion is None:
        results["failures"].append("the expansion licence refused this record")
        results["verdict"] = "REFUSED"
        results["elapsed_seconds"] = time.time() - started
        flush()
        return 1

    for case in CASES:
        row = run_leg(cp, "product", case, product, record, steps=args.steps)
        passed, failures = verdict_of(row, require_launches=row["steps_budget"])
        row["passed"], row["failures"] = passed, failures
        results["rows"].append(row)
        log(f"[product] {case[0]}: passed={passed} "
            f"launches={row.get('fused_kernel_launches')} "
            f"moved={row.get('arrays_ever_moved')}")
        flush()

    cases = {case[0]: case for case in CASES}

    other = 0 if int(expansion) == 1 else 1
    row = run_leg(cp, "expansion_override", cases[MUTATION_CASE], product, record,
                  steps=MUTATION_STEPS, expansion=other)
    row["expectation"] = "catch"
    row["caught"] = row.get("first_divergence") is not None
    row["reason"] = (
        "this family DOES compile a Bloch rotation, and the mutation case carries a "
        "three-axis phase — so unlike the cylindrical arm the two expansion arms "
        "have a nonzero-real-word coefficient to disagree over. The row is armed as "
        "a CATCH: if it comes back null the licence is decorative on this family "
        "and that is a finding, not a pass.")
    row["passed"] = bool(row["caught"])
    results["rows"].append(row)
    log(f"[expansion_override] arm {expansion}->{other}: caught={row['caught']}")
    flush()

    results["rows"].append({"leg": "refusal", "device": True,
                            **leg_refusal(cp, record)})
    log(f"[refusal] passed={results['rows'][-1]['passed']}")
    flush()

    block = shipped_block()
    for name, (case_name, old, new, expectation) in MUTATIONS.items():
        if old not in block:
            results["rows"].append({
                "leg": "mutation", "label": name, "passed": False,
                "reason": "the needle is ABSENT from the shipped source; this "
                          "mutation armed NOTHING"})
            flush()
            continue
        mutated, entry = _renamed(block.replace(old, new, 1), name)
        kernel = compile_mutant(mutated, entry)
        row = run_leg(cp, "mutation", cases[case_name], product, record,
                      mutant=kernel, steps=MUTATION_STEPS)
        diverged = row.get("first_divergence") is not None
        launched = row.get("fused_kernel_launches") == MUTATION_STEPS
        result_row = {
            "leg": "mutation", "label": name, "case": case_name,
            "expectation": expectation,
            "caught": diverged, "null_confirmed": not diverged,
            "launches_observed": row.get("fused_kernel_launches"),
            "first_divergence": row.get("first_divergence"),
            "passed": bool(diverged and row.get("fused_kernel_launches", 0) >= 1
                           if expectation == "catch"
                           else (not diverged and launched)),
        }
        if expectation == "null":
            result_row["reason"] = NULL_REASONS[name]
        results["rows"].append(result_row)
        log(f"[mutation] {name} on {case_name}: expectation={expectation} "
            f"caught={diverged} launches={row.get('fused_kernel_launches')} "
            f"passed={result_row['passed']}")
        flush()

    rows = results["rows"]
    results["verdict"] = ("PASS" if all(row.get("passed") for row in rows)
                          else "FAIL")
    results["failed_rows"] = [row.get("label", row.get("case", row.get("leg")))
                              for row in rows if not row.get("passed")]
    results["elapsed_seconds"] = time.time() - started
    # THE DEVICE IDENTITY, RECORDED WHERE THE LEDGER TOOLS READ IT (2026-09-15).
    # ``seed_triton_welds.py`` cuts a weld only from an artifact whose own
    # ``environment`` carries ``device``, ``compute_capability`` and ``hostname``,
    # and composes the weld's host line through ``rebind_triton_welds._host_line``
    # from those plus ``triton`` and ``cupy``. This block used to record ``numpy``,
    # ``cupy`` and the strip stamp only: read 2026-09-15 across all twelve Triton
    # artifacts of this gate (2026-08-30 to ``triton_fleet_2026-09-14_target_A``),
    # not one carries ``device``, so the eleven keep runs that released could not be
    # seeded. ``drive_triton_weld_gates.py`` does not close the gap -- it copies the
    # environment into ``gates.jsonl`` as ``host_environment``, which the seeder
    # does not read. ``triton_device_identity.record`` fills the identity keys from
    # the ``cupy`` and ``triton`` modules this process already imported, as
    # ``probe_triton_no_pml_fused_electric_pair.py`` does.
    import triton_device_identity  # noqa: PLC0415
    results["environment"] = triton_device_identity.record({
        "argv": list(sys.argv),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "cupy": getattr(cp, "__version__", None),
        "cupy_cache_dir": os.environ.get("CUPY_CACHE_DIR"),
        "subnormal_policy_stamp": results.get("policy_stamp"),
    })

    # ------------------------------------------------------------------
    # THE RELEASE RECORD, in the shape the fleet's readers require.
    #
    # ``gate_provenance.read_verdict`` normalises the outcome for the campaign
    # driver, and ``build_triton_fusion_matrix``'s GATE_BOUND branch requires ALL
    # of ``device_status == "RUN"``, ``release.released is True`` and ``passed is
    # True`` before it will credit a cell to this artifact. A FALSE ``released`` IS
    # THE TRAP THAT BRANCH IS WRITTEN AGAINST: a laptop run that measured no bytes
    # on any device must not be readable as a release, so the no-device path above
    # writes ``device_status = "NO_DEVICE"`` and ``released = False`` and every one
    # of the three conditions fails by construction.
    #
    # ``subnormal_policy`` is a DICT, not the bare string: the board reads
    # ``(gate["subnormal_policy"] or {}).get("policy")`` and a string raises there.
    # ------------------------------------------------------------------
    results["device_status"] = "RUN"
    results["passed"] = results["verdict"] == "PASS"
    results["planted_defect"] = False
    # THE POLICY BLOCK IS THE INSTALLED MODULE'S OWN STAMP (2026-09-15), in the shape
    # the artifacts of the seeded Triton welds carry -- read off
    # ``triton_fleet_2026-09-14_target_B/folded_dispersive_fused_pair/gate.json``:
    # ``policy`` ieee_keep_ftz_stripped, ``requested`` and ``resolved`` keep, and
    # ``cache_dir``. The hand-written dict it replaces carried ``policy`` and the
    # strip record only, and ``seed_triton_welds.policy_line`` reads ``resolved``
    # (or ``requested``) and ``cache_dir``, so it returned None for every artifact
    # of this gate. ``shared.install_ftz_strip`` installed through
    # ``meep_gpu.subnormal_policy``, so ``policy_stamp()`` describes the policy this
    # process ran under; the board reads only ``policy`` from the block.
    from meep_gpu import subnormal_policy as _subnormal_policy  # noqa: PLC0415
    results["subnormal_policy"] = dict(
        _subnormal_policy.policy_stamp(),
        installed_before_the_first_device_compile=True)
    results["release"] = {
        "released": bool(results["passed"]),
        "reasons": list(results.get("failed_rows") or []),
        "host": "the measurement machine; every device row above ran there",
    }
    results["source_sha256"] = {
        FAMILY_MODULE: sha256_of(os.path.join(API_ROOT, FAMILY_MODULE)),
        # KEYED BY ITS REPO-RELATIVE PATH, not by the word "gate". Every reader of
        # this map — ``build_triton_fusion_matrix``'s GATE_BOUND branch and the
        # weld-record walk — resolves each key against the tree and re-hashes it,
        # so a key that is not a path resolves to nothing and is reported as DRIFT
        # on every cut. Measured 2026-08-31: it was, on both of this round's gates.
        "parity/meep_gpu/probe_triton_complex_conductive_fused_pair.py": sha256_of(os.path.abspath(__file__)),
        "meep_gpu/triton_kernels/complex_no_pml_conductive.py": sha256_of(
            os.path.join(API_ROOT,
                         "meep_gpu/triton_kernels/complex_no_pml_conductive.py")),
        "meep_gpu/triton_kernels/complex_no_pml_stored_e.py": sha256_of(
            os.path.join(API_ROOT,
                         "meep_gpu/triton_kernels/complex_no_pml_stored_e.py")),
        "meep_gpu/triton_kernels/complex_fields.py": sha256_of(
            os.path.join(API_ROOT, "meep_gpu/triton_kernels/complex_fields.py")),
    }
    flush()
    log(f"VERDICT {results['verdict']} in {results['elapsed_seconds']:.1f}s; "
        f"artifact {os.path.join(out, 'gate.json')}")
    for path in _TEMPORARY:
        try:
            os.unlink(path)
        except OSError:
            pass
    return 0 if results["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
