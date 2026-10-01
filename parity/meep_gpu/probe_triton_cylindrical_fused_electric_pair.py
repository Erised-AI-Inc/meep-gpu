#!/usr/bin/env python3
"""Byte gate for the CYLINDRICAL COMPLEX fused ELECTRIC pair: Dcyl ``step_D``
welded into ``update_E``.

THE CLAIM THIS GATE IS ALLOWED TO SUPPORT: for every configuration
:func:`~meep_gpu.triton_kernels.cylindrical_fused_electric_pair.cylindrical_fused_electric_pair_coverage`
admits, ONE launch of ``cyl_complex_fused_curl_constitutive_D`` — bracketed by the
SHIPPED deposit repair wherever the seam carries an electric deposit — leaves the
engine in a state that is BIT-IDENTICAL, over every dynamic volume, on the uint32
word view, PER COMPLETE DRIVER STEP, to BOTH

  * the CuPy ARRAY PATH (``stepping.step_D`` -> the driver's own electric
    injection -> ``fill_symmetry_bc_D`` -> ``zero_metal_D`` ->
    ``fill_folded_far_ghosts_D`` -> ``stepping.update_E``), and
  * the SEPARATELY CERTIFIED Triton products it replaces — the cylindrical complex
    curl (``cylindrical_complex.plan_cylindrical_complex_curl`` on ``step_D``) and
    the certified complex constitutive
    (``complex_fields.plan_complex_constitutive`` on side E), with the three
    in-between passes left on the array path because no Triton product owns them.

``allclose`` appears nowhere.

WHY THE COMPLETE DRIVER STEP AND NOT THE SEAM ALONE. The magnetic twin's gate
drives the seam by hand, which is legitimate there: its seam carries nothing. THIS
seam carries the electric injection on sixteen of sixteen corpus rows, and the
thing that makes the launch legal is the ``LeadingRepairPlan`` /
``TrailingRepairPlan`` bracket ``launch._install_fused_pair`` puts around it. A
harness that drove ``step_D`` and ``update_E`` by hand would measure the kernel
and not the composition, so every device row here runs ``FdtdDriver.step`` and
compares the WHOLE state afterwards.

WHAT THIS GATE REFUSES TO INFER
===============================

* **Bytes alone cannot prove the fused path ran.** A silent fallback is
  byte-identical to the array path BY CONSTRUCTION. Every launch goes through a
  :class:`CountingKernel`, every seam pass is counted PER ROUTE, and a fused route
  that reached the array path for an absorbed pass FAILS its row.
* **A carry leg that repaired nothing measured the quiet case under a carry
  name.** Every carry row asserts ``LeadingRepairPlan.repairs`` is non-zero, and
  the ``carry_null_control`` leg builds the SAME shipped plan with the bracket
  REMOVED, injects anyway, and REQUIRES divergence. A carry family whose null
  control agrees is a family whose bracket does nothing.
* **A no-op agreeing with a no-op is trivially identical.** Every row carries a
  non-vacuity floor: the compared state must have MOVED, and no read-only material
  volume may have.
* **A mutation that rewrites an unreached line measures nothing.** ``M_CLASS``,
  ``BCZ``, ``BACKWARD`` and the three ``ZM_*`` flags are ``tl.constexpr`` guards
  that compile DIFFERENT BODIES. Every needle declares the case it is scored on and
  leg ``needle_reachability`` parses the needle's ENCLOSING GUARDS out of the
  shipped text and evaluates them against that case. An unmodelled guard FAILS the
  leg rather than passing.
* **A record cut under one subnormal policy licenses nothing under the other.**
  The device legs install ``keep``, which is
  :data:`~meep_gpu.triton_kernels.complex_fields.CERTIFIED_UNDER_SUBNORMAL_POLICY`
  and the policy the expansion licence is cut under, before the first device
  compile.

WHAT IT DOES NOT CLAIM. Nothing about throughput. The fusion removes ONE LAUNCH
and, on a walled run, one host wall pass; it does NOT remove the radial prefix
scan, which stays on the array path because its float32 summation order defines
the answer.

THE FAMILY IS NOT WIRED. ``launch.plan_step`` assigns at most one plan per slot
and this product spans five driver call sites. Nothing in ``launch.py`` names it,
``fastpath.plan_fast_path`` is unchanged, and no default run can reach it.

Progress reporting: one flushed line per step, every row appended to the artifact as it lands.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import sys
import tempfile
import textwrap
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

#: The product's own module path, hashed into the artifact.
FAMILY_MODULE = "meep_gpu/triton_kernels/cylindrical_fused_electric_pair.py"

#: The keep-cut expansion probe artifacts in this tree, newest first. The device
#: legs consume whichever the environment names; the row records which, so the
#: artifact says what licensed its own bytes.
KEEP_PROBES: Tuple[str, ...] = (
    "parity/meep_gpu/results/expansion_probe_2026-08-17/expansion_probe_keep.json",
    "parity/meep_gpu/results/complex_expansion_convention_2026-08-16/results/"
    "probe_keep/probe.json",
)

#: ``(name, cell, m, accurate, z_metallic, pml, steps, courant)``.
#:
#: THE COURANT IS PER CASE AND IS NOT DECORATION. ``Grid`` refuses
#: ``accurate_fields_near_cylorigin`` above ``1/(|m| + 0.5)`` BY NAME — without the
#: default treatment's zero rows the near-axis update is unstable there, "a smooth
#: growing mode rather than a crash" — so the |m| = 3 accurate row cannot run at the
#: 0.35 the others use. Measured on the GPU host 2026-08-30, where a single shared
#: Courant took the first campaign down at that row's fixture. It is lowered for
#: that row alone rather than for the sweep, because a Courant change moves ``dtdx``
#: and with it every coefficient the other rows compare.
#:
#: EVERY AXIS IS A COMPILE-TIME ARM, NOT DECORATION:
#:   m class   ``M_ONE`` (|m| = 1) emits the axis-row increment and the ``Dz``
#:             axis clear; ``M_MANY`` (|m| >= 2) emits the near-axis hold over
#:             ``ZERO_ROWS`` and NO increment. Different bodies.
#:   m sign    flips the i*m/r coefficient row's imaginary word.
#:   z         METALLIC emits the ownership mask's z clauses AND the only wall
#:             clear this family can emit (``ZM_Z``); PERIODIC emits neither.
#:   accurate  selects ``zero_rows = 1`` instead of |m| at the same m.
#:
#: A run without a metallic z would report every wall-clear mutation as uncaught,
#: and a run without an |m| >= 2 case would do the same for the near-axis hold.
CASES: Tuple[Tuple[str, Tuple[float, float, float], int, bool, bool,
                   Dict[str, Any], int, float], ...] = (
    ("m1_z_periodic", (2.0, 0.0, 2.0), 1, False, False, {"x": (0, 5)}, 10, 0.35),
    ("m1_z_metallic", (2.0, 0.0, 2.0), 1, False, True, {"x": (0, 5), "z": 5}, 10,
     0.35),
    ("m_minus1_z_metallic", (2.0, 0.0, 2.0), -1, False, True,
     {"x": (0, 5), "z": 5}, 10, 0.35),
    ("m3_z_metallic", (2.0, 0.0, 2.0), 3, False, True, {"x": (0, 5), "z": 5}, 10,
     0.35),
    ("m_minus2_z_periodic", (2.0, 0.0, 2.0), -2, False, False, {"x": (0, 5)}, 10,
     0.35),
    # 0.25, not 0.35: `Grid` refuses accurate_fields_near_cylorigin above
    # 1/(|m| + 0.5) = 0.2857 at |m| = 3. This row exists to compile ZERO_ROWS = 1
    # instead of |m| at the same m, which is the only thing that separates it from
    # the row above.
    ("m3_accurate_z_metallic", (2.0, 0.0, 2.0), 3, True, True,
     {"x": (0, 5), "z": 5}, 10, 0.25),
    # THE m = 0 ARM UNDER COMPLEX STORAGE (2026-09-04): no i*m/r block, no
    # increment, the m = 0 axis pair (Dz[0] += 4*Courant*Hp[0]; Dy[0] = 0) applied
    # to complex word pairs. Courant 0.5 is the corpus row's own
    # (examples:dipole_in_vacuum_cyl_off_axis.py), and the third case is that
    # row's SHAPE — (150, 1, 300) at this fixture's resolution 12 — with a 50-cell
    # absorber on r-high and both z faces (the lift record carries no absorber
    # depth; 50 cells is 1.0 length unit at the row's resolution 50).
    ("m0_z_metallic", (2.0, 0.0, 2.0), 0, False, True, {"x": (0, 5), "z": 5}, 10,
     0.5),
    ("m0_z_periodic", (2.0, 0.0, 2.0), 0, False, False, {"x": (0, 5)}, 10, 0.5),
    ("m0_corpus_shape", (12.5, 0.0, 25.0), 0, False, True, {"x": (0, 50), "z": 50},
     4, 0.5),
)

#: The CARRY family: the same grids run with a real electric deposit IN THIS SEAM
#: and the SHIPPED repair bracket installed. THE PRODUCT IS WORTH ZERO CORPUS ROWS
#: WITHOUT THIS FAMILY AND SIXTEEN WITH IT — all sixteen Dcyl rows declare an
#: electric source — so these rows are the product, not an extra.
#:
#: Three rather than six because the question is about the seam and not about the
#: arm sweep: one |m| = 1 periodic, one |m| = 1 walled, one |m| >= 2 walled.
CARRY_CASES: Tuple[int, ...] = (0, 1, 3, 6)

#: Steps per armed mutation. A defect needing more than this to become byte-visible
#: is reported as a NULL with its step evidence, never as a pass.
MUTATION_STEPS = 3

#: The driver call sites this product spans, in driver order (driver.py:3292-3304).
SEAM_PASSES: Tuple[str, ...] = (
    "step_D", "fill_symmetry_bc_D", "zero_metal_D", "fill_folded_far_ghosts_D",
    "update_E",
)

#: The names that MUST appear in the dynamic state inventory. Scanned rather than
#: listed so a renamed volume cannot silently drop out of the comparison; this
#: tuple is the tripwire for the scan itself shrinking.
REQUIRED = (
    "Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
    "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
    "f_w_Ex", "f_w_Ey", "f_w_Ez", "f_w_Hx", "f_w_Hy", "f_w_Hz",
)

#: THE READ-ONLY MATERIAL VOLUMES, under the names ``inventory`` really finds. A
#: leg that changed one of these measured something other than the step. ON THIS
#: SIDE ``inv_eps`` IS ALSO A KERNEL INPUT, read by the SCALE=1 arm — which makes
#: the read-only assertion stronger here, not weaker: a kernel writing through its
#: epsilon pointer is caught by the same check.
MATERIAL = ("eps", "inv_eps")


def log(message: str) -> None:
    print(message, flush=True)


def sha256_of(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _assert_material_names_are_real(found: Dict[str, Any]) -> None:
    """A MATERIAL name that matches nothing is two absent checks, not a weaker one.

    It removes a volume from the vacuity floor's denominator AND makes
    ``material_changed`` permanently empty. Both would pass silently.
    """
    present = [name for name in MATERIAL if name in found]
    if not present:
        raise AssertionError(
            f"none of the declared read-only material names {MATERIAL} match the "
            f"state scan {sorted(found)}; the material floor is inert")


# ===========================================================================
# Fixture
# ===========================================================================

def build_driver(cp, cell, m, accurate, z_metallic, pml_spec, seed: int,
                 electric: bool, courant: float = 0.35):
    """One complex cylindrical PML driver, seeded identically for every route.

    ``electric`` puts a real source IN THIS SEAM, which is the whole carry family.
    Every route gets the same source so the comparison stays a statement about the
    weld.

    THE INVARIANT-AXIS PEC TRAP IS NOT RE-BOUGHT HERE: phi is the invariant axis
    and is left PERIODIC. Declaring it metallic is refused by ``Grid`` by name, and
    a case that tripped that refusal would report a family failure that was really
    a fixture failure.
    """
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415

    boundaries = (("periodic", "periodic", "metallic") if z_metallic
                  else "periodic")
    driver = FdtdDriver(
        cell_size=cell, resolution=12.0, cylindrical=True, m=int(m),
        force_complex_fields=True, courant=float(courant),  # non-power-of-two
        accurate_fields_near_cylorigin=bool(accurate),
        boundaries=boundaries, prefer_gpu=True, gpu_id=0,
    )
    shape = driver.shape
    index = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    epsilon = np.ascontiguousarray(
        (1.45 + 0.30 * np.sin(index * np.float32(0.037))).astype(np.float32))
    driver.set_epsilon(cp.asarray(epsilon))
    driver.setup_pml(dict(pml_spec))
    if electric:
        # OFF THE AXIS DELIBERATELY. A deposit at r = 0 lands on the row the per-|m|
        # rules rewrite, so a bracket that failed to restore it could be masked by
        # the axis clear rather than measured.
        driver.add_source({"component": "Ez", "frequency": 0.31,
                           "center": (float(cell[0]) * 0.4, 0.0, 0.0),
                           "width": 0.4})
    rng = np.random.default_rng(seed)
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz"):
        host = (rng.uniform(-0.25, 0.25, size=shape)
                + 1j * rng.uniform(-0.25, 0.25, size=shape)).astype(np.complex64)
        driver.set_field(name, cp.asarray(np.ascontiguousarray(host)))
    # THE SPLIT-FIELD AUXILIARIES ARE SEEDED TOO, and that is not decoration.
    # MEASURED ON THE GPU HOST 2026-08-30 (parity/meep_gpu/_scratch/diag_cyl_axis_hold.py):
    # with `fu_D*` left at its zero initialisation, the |m| >= 2 near-axis hold on
    # the AUXILIARIES is byte-invisible -- `n0 = (fu*km - curl)*si` is exactly +0.0
    # on those rows because `fu` starts at zero and the B-side hold has already
    # emptied the H operands the curl reads there -- so the needle that removes it
    # measured 0 differing words over three complete steps while the needle that
    # narrows the hold's ROW RANGE was caught. A zero auxiliary makes `fu*kms`
    # exactly zero whatever `kms` is; seeding it is what gives the hold something
    # to clear and the coefficient something to be wrong about.
    for name in ("fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz"):
        array = getattr(driver.fields, name, None)
        if array is not None:
            host = (rng.uniform(-0.1, 0.1, size=shape)
                    + 1j * rng.uniform(-0.1, 0.1, size=shape)).astype(np.complex64)
            array[...] = cp.asarray(np.ascontiguousarray(host))
    # SEEDED ON THE E SIDE: an all-zero f_w_E makes the history read
    # indistinguishable from a zero and disarms the `prev`-ordering mutation on the
    # first step.
    for name in ("f_w_Ex", "f_w_Ey", "f_w_Ez"):
        array = getattr(driver.fields, name, None)
        if array is not None:
            host = (rng.uniform(-0.05, 0.05, size=shape)
                    + 1j * rng.uniform(-0.05, 0.05, size=shape)).astype(np.complex64)
            array[...] = cp.asarray(np.ascontiguousarray(host))
    return driver


#: Private ``Fields`` attributes that are HARNESS SCRATCH rather than state, and
#: are excluded from the compared inventory for a measured reason.
#:
#: ``_fmp_scratch`` is allocated LAZILY by the array path's own ``update_E``, so a
#: route that substitutes that pass never creates it — and the two inventories then
#: differ by a name rather than by a word. Measured on the GPU host 2026-08-30: the
#: conductive twin's first campaign reported ``{'reason': 'asymmetric', 'names':
#: ['_fmp_scratch']}`` as a byte divergence on every case and broke the step loop at
#: step 1, so every row was scored over ONE step and none of them measured what they
#: exist to measure. THE ASYMMETRY IS STILL RECORDED — ``inventory_asymmetry_vs_array``
#: carries it per row — it just no longer masquerades as a divergence.
#:
#: THE RULE IS THE LEADING UNDERSCORE, not this list: a private attribute is not
#: physical state, and every PUBLIC volume stays in the comparison. The list is the
#: tripwire for that rule, so a private name the scan starts dropping is visible.
PRIVATE_SCRATCH: Tuple[str, ...] = ("_fmp_scratch",)


def inventory(driver) -> Dict[str, Any]:
    """Every device volume the step can touch, found by SCANNING rather than listing."""
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
    return found


def words(cp, array) -> np.ndarray:
    return np.ascontiguousarray(cp.asnumpy(array)).view(np.uint32).ravel()


def snapshot(cp, driver) -> Dict[str, np.ndarray]:
    return {name: words(cp, array) for name, array in inventory(driver).items()}


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
    """Owns the JIT kernel and counts every launch through the plan's ``run``."""

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


def kernel_ptx(jit: Any) -> List[str]:
    out: List[str] = []
    for per_device in (getattr(jit, "cache", None) or {}).values():
        for compiled in per_device.values():
            asm = getattr(compiled, "asm", None)
            if asm and "ptx" in asm:
                out.append(asm["ptx"])
    return out


class Route:
    """The plan bundle installed for one driver, keyed by driver call site."""

    def __init__(self, plans: Dict[str, Any]) -> None:
        self.plans = plans


class _Absorbed:
    """The sentinel left where a driver pass was absorbed by the fused launch."""

    __slots__ = ("name", "absorbed_by")

    def __init__(self, name: str, absorbed_by: Any) -> None:
        self.name = name
        self.absorbed_by = absorbed_by

    def run(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def install(driver_module, routes, counter: Dict[str, int], roles: Dict[int, str]):
    """Replace the five seam passes for the fields objects named in ``routes``.

    Counting is PER ROUTE. A single global counter would mix the reference driver's
    legitimate array-path calls with a fallback in the fused route, which is
    precisely the event this instrument exists to see.
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
    """The two separately certified Triton products this launch replaces.

    THE COMPILED CONSTITUTIVE ARITHMETIC IS THE ORDINARY COMPLEX ONE, which is not
    an approximation: ``stepping.update_E`` contains zero occurrences of
    ``cylindrical``, ``is_axis``, ``m`` or ``axis_zero``, and the cylindrical
    tranche MEASURED that (480/480 rows, 0 differing words) rather than arguing it.

    IT IS REACHED THROUGH THE Dcyl WRAPPER, NOT THROUGH ``complex_fields``, and the
    difference is a PREDICATE rather than a kernel:
    :func:`.complex_fields.plan_complex_constitutive` carries the complex tranche's
    BLANKET Dcyl refusal and returns ``None`` on every grid this gate builds —
    measured here on 2026-08-30, where it took down the first campaign at the
    separate-oracle assertion — while
    :func:`.cylindrical_complex.plan_cylindrical_complex_constitutive` owns the
    distinct predicate that admits the Dcyl slice and then builds the SAME
    ``ComplexConstitutivePlan`` over the SAME ``bloch_constitutive_step``. So the
    oracle is the certified complex constitutive either way; only the guard that
    lets it be built is the cylindrical one.
    """
    from meep_gpu.triton_kernels import cylindrical_complex as cyl  # noqa: PLC0415

    curl = cyl.plan_cylindrical_complex_curl(driver.fields, driver.pml, "step_D",
                                             probe=probe)
    constitutive = cyl.plan_cylindrical_complex_constitutive(
        driver.fields, driver.pml, "E", probe=probe)
    missing = [name for name, plan in (("cylindrical curl", curl),
                                       ("complex constitutive", constitutive))
               if plan is None]
    if missing:
        raise AssertionError(
            f"the separate oracle is incomplete: {missing} refused this case, so "
            f"this leg could not compare the fused launch against the products it "
            f"replaces")
    # The three in-between passes stay on the ARRAY PATH here: no Triton product
    # owns the wall clear, and both symmetry fills are dead on a Dcyl grid. All
    # three are counted, so a difference in which of them ran is a counter event
    # rather than an invisible correction. THE INJECTION ALSO STAYS ON THE ARRAY
    # PATH, which is what makes this route the right oracle for the carry family:
    # the two certified kernels with the driver's own deposit between them.
    return Route({"step_D": curl, "update_E": constitutive})


def fused_route(plan, driver=None, sources=()):
    """The fused plan in its two slots, BRACKETED when the seam carries a deposit.

    THE BRACKET IS THE SHIPPED ONE. ``deposit_repair.LeadingRepairPlan`` /
    ``TrailingRepairPlan`` are the exact pair ``launch._install_fused_pair``
    puts in these two slots; a harness that assembled its own could not license the
    one that ships.
    """
    from meep_gpu import deposit_repair  # noqa: PLC0415

    seam = deposit_repair.in_seam_sources(tuple(sources), "D")
    if not seam:
        return Route({name: (plan if name == "step_D" else _Absorbed(name, plan))
                      for name in SEAM_PASSES}), None
    leading = deposit_repair.LeadingRepairPlan(plan, driver.fields, driver.pml,
                                               seam, "D")
    trailing = deposit_repair.TrailingRepairPlan("update_E", leading, driver.fields,
                                                 driver.pml)
    plans = {name: _Absorbed(name, plan) for name in SEAM_PASSES}
    plans["step_D"] = leading
    plans["update_E"] = trailing
    return Route(plans), leading


# ===========================================================================
# The three-route leg
# ===========================================================================

def run_leg(cp, name: str, case, product, probe, mutant: Any = None,
            steps: Optional[int] = None, install_fused: bool = True,
            electric: bool = False, bracket: bool = True,
            expansion: Optional[int] = None) -> Dict[str, Any]:
    """Three routes in lockstep, per COMPLETE driver step; stop at the FIRST
    byte divergence."""
    import meep_gpu.driver as driver_module  # noqa: PLC0415

    (label, cell, m, accurate, z_metallic, pml_spec, default_steps,
     courant) = case
    budget = int(default_steps if steps is None else steps)
    built = [build_driver(cp, cell, m, accurate, z_metallic, pml_spec, SEED,
                          electric, courant) for _ in range(3)]
    reference, separate, fused = built
    undo: Callable[[], Any] = lambda: None
    counter: Dict[str, int] = {}
    kernel = CountingKernel(
        mutant if mutant is not None
        else product.cyl_complex_fused_curl_constitutive_D_kernel())
    row: Dict[str, Any] = {
        "leg": name, "device": True, "case": label, "steps_budget": budget,
        "shape": list(reference.shape), "m": m, "accurate": accurate,
        "z": "metallic" if z_metallic else "periodic", "pml": dict(pml_spec),
        "courant": courant,
        "electric_source_in_the_seam": bool(electric),
        "fused_substituted": bool(install_fused),
        "bracket_installed": bool(bracket),
        "expansion_override": expansion,
        "first_divergence": None, "control_divergence": None,
    }
    try:
        plan = product.plan_cylindrical_fused_electric_pair(
            fused.fields, fused.pml, tuple(fused._sources), num_warps=1,
            kernel=kernel, probe=probe)
        if plan is None:
            verdict = product.cylindrical_fused_electric_pair_coverage(
                fused.fields, fused.pml, tuple(fused._sources), probe=probe)
            raise AssertionError(f"the product refused the case: {verdict.reasons}")
        if expansion is not None:
            # THE ARM, OVERRIDDEN. Not a source rewrite: the constexpr IS the
            # licence, so flipping it here measures whether the licence is
            # load-bearing rather than decorative.
            plan.expansion = int(expansion)
        row["plan"] = repr(plan)
        row["plan_replaces"] = list(plan.replaces)
        row["licensed_expansion"] = plan.expansion
        row["m_class"] = plan.m_class
        row["zero_rows"] = plan.zero_rows
        row["zero_metal"] = list(plan.zero_metal)

        leading = None
        if install_fused:
            if bracket:
                route, leading = fused_route(plan, fused, tuple(fused._sources))
            else:
                # THE NULL CONTROL. The SAME shipped plan with the bracket removed
                # — the NoopPlan the False branch would leave in the trailing slot
                # — injected into anyway. This route MUST diverge.
                route = Route({key: (plan if key == "step_D"
                                     else _Absorbed(key, plan))
                               for key in SEAM_PASSES})
        else:
            route = Route({})
        row["bracketed"] = leading is not None
        row["slot_classes"] = {key: type(value).__name__
                               for key, value in route.plans.items()}
        if electric and install_fused and bracket and leading is None:
            raise AssertionError(
                "the case declares an in-seam electric source but the bracket was "
                "not installed; this leg would measure an unbracketed launch and "
                "report it as the shipped composition")
        separate_plans = separate_route(separate, probe)
        row["separate_products"] = {key: type(value).__name__
                                    for key, value in separate_plans.plans.items()}

        roles = {id(reference.fields): "array", id(separate.fields): "separate",
                 id(fused.fields): "fused"}
        routes = [(separate.fields, separate_plans), (fused.fields, route)]
        undo = install(driver_module, routes, counter, roles)

        opening = snapshot(cp, fused)
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
                "deposit_repairs": (leading.repairs if leading is not None
                                    else None),
            })
            divergence = versus_array or versus_separate
            log(f"  {name}/{label} step {step}/{budget} "
                f"identical={divergence is None} control={control is None} "
                f"moved={len(step_moved)} launches={kernel.calls} "
                f"repairs={leading.repairs if leading is not None else '-'} "
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
        # THE ASYMMETRY IS RECORDED, not compared through. `PRIVATE_SCRATCH`'s
        # docstring says why the private names are out of the comparison; this row
        # is where a PUBLIC one appearing on one route and not the other becomes
        # visible, and `verdict_of` fails on a non-empty list.
        row["inventory_asymmetry_vs_array"] = sorted(
            set(final).symmetric_difference(snapshot(cp, reference)))
        row["private_scratch_on_the_array_route"] = sorted(
            name for name in vars(reference.fields)
            if name in PRIVATE_SCRATCH)
        row["private_scratch_on_the_fused_route"] = sorted(
            name for name in vars(fused.fields) if name in PRIVATE_SCRATCH)
        row["launches"] = dict(counter)
        row["fused_kernel_launches"] = kernel.calls
        row["deposit_repairs"] = leading.repairs if leading is not None else None
        row["launch_grids"] = sorted({grid for grid in kernel.grids})
        row["ptx_specializations"] = len(kernel_ptx(kernel.jit))
        row["elapsed_seconds"] = time.time() - started
        return row
    finally:
        undo()
        for target in built:
            try:
                target.close()
            except Exception:  # noqa: BLE001 - a close failure must not hide a result
                pass
        cp.get_default_memory_pool().free_all_blocks()


def verdict_of(row: Dict[str, Any], *, require_identical: bool = True,
               require_launches: Optional[int] = None,
               require_moved: bool = True,
               require_repairs: bool = False) -> Tuple[bool, List[str]]:
    """The leg's pass conditions, stated rather than implied."""
    failures: List[str] = []
    if require_identical and row.get("first_divergence") is not None:
        failures.append(f"byte divergence: {row['first_divergence']}")
    if not require_identical and row.get("first_divergence") is None:
        failures.append(
            "this leg REQUIRES divergence and found none: the control it exists to "
            "be is inert, and the thing it was meant to prove load-bearing is not")
    if row.get("control_divergence") is not None:
        failures.append(
            f"an ORACLE control itself diverged from the array path, so this leg "
            f"could not have measured the fused launch: {row['control_divergence']}")
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
    if require_repairs and not row.get("deposit_repairs"):
        failures.append(
            "the carry leg repaired NO deposit cell: the bracket ran but the seam "
            "carried nothing, so this leg measured the quiet case under a carry name")
    fell_back = {key: value for key, value in (row.get("launches") or {}).items()
                 if key.startswith("fused/array_path:")}
    if fell_back:
        failures.append(
            f"the fused route reached the array path for an absorbed pass: "
            f"{fell_back}")
    return (not failures), failures


# ===========================================================================
# Armed source mutations
# ===========================================================================

def shipped_block() -> str:
    """The shipped kernel definition, read out of the FILE rather than re-typed.

    RETURNED AT ITS OWN INDENTATION — the kernel is defined inside
    ``if triton is not None:``, so every body line carries eight spaces. Every
    needle in :data:`MUTATIONS` is written against THIS text.

    Parsing the source is what keeps this from being a MIRRORED EVALUATOR: a gate
    that re-implemented the kernel's arithmetic would mirror a planted defect
    instead of executing it.
    """
    with open(os.path.join(API_ROOT, FAMILY_MODULE), encoding="utf-8") as handle:
        text = handle.read()
    start = text.index("    def cyl_complex_fused_curl_constitutive_D(")
    end = text.index("else:  # pragma: no cover - laptop path")
    return text[start:end].rstrip() + "\n"


def dedented(block: str) -> str:
    """One indentation level off, so the block is importable at module scope."""
    return "\n".join(line[4:] if line.startswith("    ") else line
                     for line in block.splitlines()) + "\n"


def compile_mutant(source: str, entry: str):
    """Compile one mutated kernel from a REAL FILE.

    Triton reads source through ``inspect``, so an ``exec``'d body raises at first
    launch. The constexpr codes and the certified multiply helpers are re-declared
    in the header because a ``@triton.jit`` body may not read a plain module global;
    the helpers are IMPORTED rather than copied, so a mutant still exercises the
    shipped multiply.
    """
    header = (
        "import triton\n"
        "import triton.language as tl\n"
        "from meep_gpu.triton_kernels.complex_fields import (\n"
        "    _mul_coefficient_left, _mul_field_left)\n"
        "from meep_gpu.triton_kernels.cylindrical_complex import (\n"
        "    _mul_general_coefficient_left)\n"
        "PERIODIC = tl.constexpr(0)\n"
        "METALLIC = tl.constexpr(1)\n"
        "M_ZERO = tl.constexpr(0)\n"
        "M_ONE = tl.constexpr(1)\n"
        "M_MANY = tl.constexpr(2)\n\n"
        # shipped_block() starts at `def`, so the DECORATOR is supplied here.
        # Without it the mutated module defines a plain Python function,
        # CountingKernel raises "'function' object is not subscriptable", and the
        # whole mutation battery is unarmed.
        "@triton.jit\n")
    handle = tempfile.NamedTemporaryFile(
        "w", suffix="_mutated_cylfusedD.py", delete=False, encoding="utf-8")
    handle.write(header + source)
    handle.close()
    _TEMPORARY.append(handle.name)
    spec = importlib.util.spec_from_file_location(
        "triton_mutated_cylfusedD_" + str(len(_TEMPORARY)), handle.name)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = module  # type: ignore[union-attr]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return getattr(module, entry)


def _renamed(source: str, suffix: str) -> Tuple[str, str]:
    name = f"cyl_complex_fused_curl_constitutive_D__{suffix}"
    return (source.replace("def cyl_complex_fused_curl_constitutive_D(",
                           f"def {name}(", 1), name)


#: Every armed source defect, as DATA: name -> (case, old, new, expectation).
#:
#: THE CASE EACH IS SCORED ON IS DECLARED, because ``M_CLASS``, ``BCZ``,
#: ``BACKWARD`` and the three ``ZM_*`` flags are ``tl.constexpr`` guards: a needle
#: armed on a case whose arm the shipped kernel does not compile would rewrite an
#: unreached line and report UNCAUGHT while measuring nothing.
MUTATIONS: Dict[str, Tuple[str, str, str, str]] = {
    # ---- THE SEAM ITSELF ---------------------------------------------------
    # The whole weld is `src = v`, three times. These are the needles that ask
    # whether the register carried across the seam is the one the array path
    # leaves in D.
    "seam_takes_the_pre_recurrence_curl":
        ("m1_z_metallic", "        src_re = v0_re\n        src_im = v0_im\n",
         "        src_re = curl0_re\n        src_im = curl0_im\n", "catch"),
    "seam_takes_the_wrong_component":
        ("m1_z_metallic", "        src_re = v1_re\n        src_im = v1_im\n",
         "        src_re = v0_re\n        src_im = v0_im\n", "catch"),
    # THE ORDERING FACT THIS PRODUCT'S DOCSTRING NAMES. `_cylindrical_axis_zero_D`
    # runs AFTER `_apply_pml_update`, so a weld that consumed the recurrence's
    # output rather than the post-rule value would hand update_E a displacement the
    # array path never leaves in D. Armed on |m| >= 2, where the hold is live over
    # ZERO_ROWS rows rather than one component of one row.
    "seam_reads_before_the_axis_rules":
        ("m3_z_metallic",
         "            v2_re = tl.where(near, 0.0, v2_re)\n"
         "            v2_im = tl.where(near, 0.0, v2_im)\n", "", "catch"),
    # ---- THE WALL CLEAR (ZM_Z; ZM_X/ZM_Y are dead on every Dcyl grid) -------
    "zero_metal_dropped":
        ("m1_z_metallic",
         "        if ZM_Z:\n"
         "            v0_re = tl.where(at_z, 0.0, v0_re)\n"
         "            v0_im = tl.where(at_z, 0.0, v0_im)\n"
         "            v1_re = tl.where(at_z, 0.0, v1_re)\n"
         "            v1_im = tl.where(at_z, 0.0, v1_im)\n",
         "        if ZM_Z:\n            pass\n", "catch"),
    "zero_metal_clears_only_the_real_plane":
        ("m1_z_metallic", "            v0_im = tl.where(at_z, 0.0, v0_im)\n", "",
         "catch"),
    # THE COMPONENT MAP. A z wall clears Dx and Dy and LEAVES Dz ALONE, which is the
    # exact COMPLEMENT of the magnetic twin's map — and the one place a transcriber
    # gets this seam wrong. This needle installs the twin's map.
    "zero_metal_takes_the_magnetic_twins_component_map":
        ("m1_z_metallic",
         "            v0_re = tl.where(at_z, 0.0, v0_re)\n"
         "            v0_im = tl.where(at_z, 0.0, v0_im)\n"
         "            v1_re = tl.where(at_z, 0.0, v1_re)\n"
         "            v1_im = tl.where(at_z, 0.0, v1_im)\n",
         "            v2_re = tl.where(at_z, 0.0, v2_re)\n"
         "            v2_im = tl.where(at_z, 0.0, v2_im)\n", "catch"),
    # ---- THE CYLINDRICAL CURL ----------------------------------------------
    "imr_partners_swapped":
        ("m1_z_metallic",
         "            m0_re, m0_im = _mul_general_coefficient_left(q0_re, q0_im, c_re, c_im,",
         "            m0_re, m0_im = _mul_general_coefficient_left(q0_re, q0_im, a_re, a_im,",
         "catch"),
    # THE ROW IS INDEXED BY r, AND BOTH WORDS MUST MOVE FOR THE EDIT TO BE ONE.
    # Measured on the GPU host 2026-08-30: rewriting the REAL word alone is byte-
    # invisible over three complete steps, because the i*m/r coefficient is PURELY
    # IMAGINARY — its real word is an exact +0.0 at EVERY row — so a misindexed
    # real word reads another +0.0. That null is kept below as its own scored row
    # rather than discarded, because it is the measurement behind the module's
    # claim that the row's "real word is +0.0 and MUST survive as a literal
    # cross-term operand"; this needle is the one that actually misindexes the
    # coefficient.
    "imr_row_indexed_by_z":
        ("m1_z_metallic",
         "            q0_re = tl.load(c0 + 2 * i, mask=live, other=0.0)\n"
         "            q0_im = tl.load(c0 + 2 * i + 1, mask=live, other=0.0)\n",
         "            q0_re = tl.load(c0 + 2 * k, mask=live, other=0.0)\n"
         "            q0_im = tl.load(c0 + 2 * k + 1, mask=live, other=0.0)\n",
         "catch"),
    "imr_row_real_word_alone_indexed_by_z":
        ("m1_z_metallic",
         "            q0_re = tl.load(c0 + 2 * i, mask=live, other=0.0)\n",
         "            q0_re = tl.load(c0 + 2 * k, mask=live, other=0.0)\n",
         "null"),
    "imr_sign_flipped":
        ("m1_z_metallic", "            curl0_re = curl0_re - m0_re",
         "            curl0_re = curl0_re + m0_re", "catch"),
    # THE D-SIDE PREFIX SUBSTITUTION on target 2, which is the whole cylindrical
    # difference in the backward arm.
    "prefix_substitution_dropped":
        ("m1_z_metallic",
         "            t2_re = ((p_down_re - p_here_re) + (a_re - a_p_re))\n"
         "            t2_im = ((p_down_im - p_here_im) + (a_im - a_p_im))\n",
         "", "catch"),
    "prefix_neighbour_is_the_centre":
        ("m1_z_metallic",
         "            p_down_re = tl.load(pfx + 2 * o_r, mask=vr, other=0.0)",
         "            p_down_re = tl.load(pfx + 2 * idx, mask=vr, other=0.0)",
         "catch"),
    # ---- THE |m| = 1 AXIS-ROW INCREMENT (M_ONE + BACKWARD body only) -------
    # ON THIS SUB-STEP THE TARGET IS 1 (Dy), not 0 — AXIS_INCREMENT_TARGET's whole
    # content. A transcriber carrying the B side's target here writes a silent
    # wrong-component increment.
    "axis_increment_writes_the_B_sides_target":
        ("m1_z_metallic",
         "                curl1_re = tl.where(at_r, inc_re * -1.0, curl1_re)\n"
         "                curl1_im = tl.where(at_r, inc_im * -1.0, curl1_im)",
         "                curl0_re = tl.where(at_r, inc_re * -1.0, curl0_re)\n"
         "                curl0_im = tl.where(at_r, inc_im * -1.0, curl0_im)",
         "catch"),
    "axis_increment_not_negated":
        ("m1_z_metallic",
         "                curl1_re = tl.where(at_r, inc_re * -1.0, curl1_re)",
         "                curl1_re = tl.where(at_r, inc_re, curl1_re)", "catch"),
    "axis_increment_drops_the_two_Hz_term":
        ("m1_z_metallic", "                s_re = (a_re - a_z_re) - two_c_re",
         "                s_re = (a_re - a_z_re)", "catch"),
    # ---- THE |m| = 1 Dz AXIS CLEAR (M_ONE + BACKWARD) ----------------------
    # THE NEEDLE CARRIES ITS OWN ``if`` AND LEAVES A ``pass``. Deleting the two
    # store lines alone leaves ``if BACKWARD:`` with an empty block followed by the
    # ``else:``, which is an IndentationError rather than a mutant — measured on
    # the GPU host 2026-08-30, where it killed the run 146 s in, AFTER thirteen
    # mutations had already been scored. The ``mutants_compile`` leg below now
    # catches that class on the laptop.
    "m_one_axis_clear_dropped":
        ("m1_z_metallic",
         "            if BACKWARD:\n"
         "                v2_re = tl.where(at_r, 0.0, v2_re)\n"
         "                v2_im = tl.where(at_r, 0.0, v2_im)\n",
         "            if BACKWARD:\n                pass\n", "catch"),
    # ---- THE |m| >= 2 NEAR-AXIS HOLD (M_MANY body only) --------------------
    "axis_hold_skips_the_auxiliaries":
        ("m3_z_metallic", "            n0_re = tl.where(near, 0.0, n0_re)\n", "",
         "catch"),
    "axis_hold_covers_only_row_zero":
        ("m3_z_metallic", "            near = i < ZERO_ROWS",
         "            near = i == 0", "catch"),
    # ---- THE CERTIFIED COMPLEX BASE ----------------------------------------
    "curl_parens_flattened":
        ("m1_z_metallic",
         "        t1_re = ((a_z_re - a_re) + (c_re - c_r_re))",
         "        t1_re = (a_z_re - a_re + c_re - c_r_re)", "catch"),
    "recurrence_axis_pair_swapped":
        ("m1_z_metallic",
         "        x_re, x_im = _mul_field_left(p0_re, p0_im, km_y, EXPANSION)",
         "        x_re, x_im = _mul_field_left(p0_re, p0_im, km_z, EXPANSION)",
         "catch"),
    "fu_store_dropped":
        ("m1_z_metallic", "        tl.store(u0 + 2 * idx, n0_re, mask=live)\n", "",
         "catch"),
    "displacement_store_dropped":
        ("m1_z_metallic", "        tl.store(f0 + 2 * idx, v0_re, mask=live)\n", "",
         "catch"),
    "fw_store_dropped":
        ("m1_z_metallic", "        tl.store(w0 + 2 * idx, src_re, mask=live)\n", "",
         "catch"),
    # ---- THE E-SIDE CONSTITUTIVE -------------------------------------------
    "inv_eps_word_doubled":
        ("m1_z_metallic", "            ie = tl.load(e0 + idx, mask=live, other=0.0)",
         "            ie = tl.load(e0 + 2 * idx, mask=live, other=0.0)", "catch"),
    "inv_eps_multiply_moved_after_the_w_store":
        ("m1_z_metallic",
         "        if SCALE:\n"
         "            # inv_eps is a float32 volume at the COMPLEX cell index — `+ idx`, not\n"
         "            # `+ 2 * idx`. D LEFT (S:982-984).\n"
         "            ie = tl.load(e0 + idx, mask=live, other=0.0)\n"
         "            src_re, src_im = _mul_field_left(src_re, src_im, ie, EXPANSION)\n"
         "        tl.store(w0 + 2 * idx, src_re, mask=live)\n"
         "        tl.store(w0 + 2 * idx + 1, src_im, mask=live)\n",
         "        tl.store(w0 + 2 * idx, src_re, mask=live)\n"
         "        tl.store(w0 + 2 * idx + 1, src_im, mask=live)\n"
         "        if SCALE:\n"
         "            ie = tl.load(e0 + idx, mask=live, other=0.0)\n"
         "            src_re, src_im = _mul_field_left(src_re, src_im, ie, EXPANSION)\n",
         "catch"),
    "constitutive_coefficient_index_moved":
        ("m1_z_metallic", "        kp_0 = tl.load(kp0 + i, mask=live, other=0.0)",
         "        kp_0 = tl.load(kp0 + j, mask=live, other=0.0)", "catch"),
    # THE LOAD MUST CROSS THE STORE, or the needle moves nothing. Measured on
    # the GPU host 2026-08-30: an earlier spelling moved `prev` only past `src_re =
    # v0_re` -- still BEFORE `tl.store(w0 ...)` -- and was correctly uncaught,
    # because reading the history before the write is exactly what the shipped
    # body does wherever the load sits. The needle below carries the whole block
    # through the inv_eps multiply and the store, so the mutant reads back the
    # value it has just written.
    "constitutive_prev_read_after_write":
        ("m1_z_metallic",
         "        prev_re = tl.load(w0 + 2 * idx, mask=live, other=0.0)   # BEFORE the store.\n"
         "        prev_im = tl.load(w0 + 2 * idx + 1, mask=live, other=0.0)\n"
         "        src_re = v0_re\n        src_im = v0_im\n"
         "        if SCALE:\n"
         "            # inv_eps is a float32 volume at the COMPLEX cell index — `+ idx`, not\n"
         "            # `+ 2 * idx`. D LEFT (S:982-984).\n"
         "            ie = tl.load(e0 + idx, mask=live, other=0.0)\n"
         "            src_re, src_im = _mul_field_left(src_re, src_im, ie, EXPANSION)\n"
         "        tl.store(w0 + 2 * idx, src_re, mask=live)\n"
         "        tl.store(w0 + 2 * idx + 1, src_im, mask=live)\n",
         "        src_re = v0_re\n        src_im = v0_im\n"
         "        if SCALE:\n"
         "            ie = tl.load(e0 + idx, mask=live, other=0.0)\n"
         "            src_re, src_im = _mul_field_left(src_re, src_im, ie, EXPANSION)\n"
         "        tl.store(w0 + 2 * idx, src_re, mask=live)\n"
         "        tl.store(w0 + 2 * idx + 1, src_im, mask=live)\n"
         "        prev_re = tl.load(w0 + 2 * idx, mask=live, other=0.0)\n"
         "        prev_im = tl.load(w0 + 2 * idx + 1, mask=live, other=0.0)\n",
         "catch"),
    "constitutive_accumulation_order_reversed":
        ("m1_z_metallic",
         "        t_re, t_im = _mul_coefficient_left(kp_0, src_re, src_im, EXPANSION)\n"
         "        a_re = a_re + t_re\n        a_im = a_im + t_im\n"
         "        t_re, t_im = _mul_coefficient_left(km_0, prev_re, prev_im, EXPANSION)\n"
         "        a_re = a_re - t_re\n        a_im = a_im - t_im\n",
         "        t_re, t_im = _mul_coefficient_left(km_0, prev_re, prev_im, EXPANSION)\n"
         "        a_re = a_re - t_re\n        a_im = a_im - t_im\n"
         "        t_re, t_im = _mul_coefficient_left(kp_0, src_re, src_im, EXPANSION)\n"
         "        a_re = a_re + t_re\n        a_im = a_im + t_im\n",
         "catch"),
    # ---- THE m = 0 AXIS PAIR (M_ZERO body only; 2026-09-04) --------------------
    "m0_dz_axis_add_dropped":
        ("m0_z_metallic",
         "                v2_re = tl.where(at_r, v2_re + hp_re, v2_re)\n"
         "                v2_im = tl.where(at_r, v2_im + hp_im, v2_im)\n",
         "                v2_re = v2_re\n                v2_im = v2_im\n", "catch"),
    "m0_dy_axis_zero_dropped":
        ("m0_z_metallic",
         "                v1_re = tl.where(at_r, 0.0, v1_re)\n"
         "                v1_im = tl.where(at_r, 0.0, v1_im)\n",
         "                v1_re = v1_re\n                v1_im = v1_im\n", "catch"),
    "m0_dz_axis_add_uses_dtdx":
        ("m0_z_metallic",
         "                hp_re, hp_im = _mul_coefficient_left(four_dtdx, b_re, b_im, EXPANSION)",
         "                hp_re, hp_im = _mul_coefficient_left(dtdx, b_re, b_im, EXPANSION)",
         "catch"),

}

NULL_REASONS: Dict[str, str] = {
    "imr_row_real_word_alone_indexed_by_z":
        "the i*m/r coefficient row is PURELY IMAGINARY — `imr_coefficient_row` "
        "builds `+-1j * m / r`, whose real word is an exact +0.0 at every row — so "
        "misindexing the REAL word alone reads another +0.0 and no bit can move. "
        "MEASURED on the GPU host 2026-08-30 over three complete driver steps: 0 "
        "differing words here, against a CATCH for the row above, which moves both "
        "words. The pair is the measurement, and it is the evidence for the "
        "module's claim that the row's real word 'MUST survive as a literal "
        "cross-term operand': a plane-wise shortcut that dropped the +0.0 * z "
        "product would be byte-wrong on signed zeros, and this null is what says "
        "the zero is really there rather than merely intended.",
}


#: How each ``tl.constexpr`` guard this kernel carries is evaluated for one case.
#:
#: FAIL-CLOSED BY CONSTRUCTION: a condition that is not in this table makes
#: :func:`guard_chain_holds` return ``None`` and the leg FAIL, rather than letting
#: an unmodelled guard pass silently. The dead-branch class is a mutation reported
#: UNCAUGHT because the case never entered the block, and a guard evaluator that
#: shrugged at an unknown condition would re-buy it.
GUARD_RULES: Dict[str, Callable[[Dict[str, Any]], bool]] = {
    # BACKWARD is bound to 1 by every builder (the product IS the D seam), so the
    # forward arms of the verbatim-copied curl are DEAD on every case.
    "if BACKWARD:": lambda case: True,
    "if BCZ == METALLIC:": lambda case: case["z_metallic"],
    "if M_CLASS == M_ONE:": lambda case: abs(case["m"]) == 1,
    # The m = 0 arm (2026-09-04): the i*m/r block is compiled OUT there, the
    # m = 0 axis pair compiled IN, and the near-axis hold is its own arm now.
    "if M_CLASS != M_ZERO:": lambda case: case["m"] != 0,
    "if M_CLASS == M_ZERO:": lambda case: case["m"] == 0,
    "if M_CLASS == M_MANY:": lambda case: abs(case["m"]) >= 2,
    "if SCALE:": lambda case: True,
    # On a Dcyl grid `Grid` cannot report r or phi walled, so ZM_X and ZM_Y are
    # dead on EVERY admitted configuration — which is why the shipped product
    # refuses a grid whose wall table says otherwise, and why no needle here is
    # armed inside either.
    "if ZM_X:": lambda case: False,
    "if ZM_Y:": lambda case: False,
    "if ZM_Z:": lambda case: case["z_metallic"],
}


def _first_line(needle: str) -> str:
    for line in needle.splitlines():
        if line.strip():
            return line
    raise AssertionError(f"empty needle: {needle!r}")


def guard_chain(block: str, needle: str) -> List[str]:
    """Every ``if`` / ``else`` header ENCLOSING the needle, outermost first.

    Parsed out of the shipped text by indentation rather than declared in a table
    beside it: a hand-kept table of "which guard is this under" is exactly the
    thing that drifts, and when it drifts a needle silently stops being scored on
    the case that can reach it.
    """
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
    """Does this case COMPILE every guard the needle sits under? ``None`` = unknown.

    ``None`` is a refusal, never an assumed True: an unmodelled guard is exactly
    the situation in which a mutation can report UNCAUGHT while measuring nothing.
    """
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
    """Every needle must be PRESENT, and its ENCLOSING GUARDS must hold on its case.

    THE DEAD-BRANCH CHECK, and the reason this leg exists. A mutation can rewrite
    REAL lines the scored case never reaches, because they sit under a
    ``tl.constexpr`` guard that case does not enter; it then reports UNCAUGHT while
    measuring nothing.
    """
    block = shipped_block()
    cases = {case[0]: case for case in CASES}
    rows: List[Dict[str, Any]] = []
    for name, (case_name, old, _new, expectation) in MUTATIONS.items():
        case = cases[case_name]
        context = {"m": case[2], "accurate": case[3], "z_metallic": case[4]}
        present = old in block
        occurrences = block.count(old) if present else 0
        chain = guard_chain(block, old) if present else []
        holds = guard_chain_holds(chain, context) if present else None
        # A NEEDLE THAT SWALLOWS ITS OWN ``if`` HAS NO ENCLOSING CHAIN, and the
        # reachability question moves INSIDE it — the ``zero_metal_*`` needles carry
        # ``if ZM_Z:`` and the axis-rule ones carry ``if M_CLASS == M_ONE:``. Scored
        # on a case that does not compile them, each would rewrite unreached lines
        # and report UNCAUGHT, so the guards the needle CONTAINS are evaluated too,
        # by the same fail-closed table.
        contained = [line.strip() for line in old.splitlines()
                     if line.strip().startswith("if ")]
        inner = guard_chain_holds(contained, context) if present else None
        rows.append({
            "mutation": name, "case": case_name, "expectation": expectation,
            "needle_present": present,
            "needle_occurrences": occurrences,
            "enclosing_guards": chain,
            "guards_inside_the_needle": contained,
            "case_compiles_the_guards_inside_the_needle": inner,
            "case_compiles_every_guard": (
                None if holds is None or inner is None else bool(holds and inner)),
            "case_m": context["m"],
            "case_z": "metallic" if context["z_metallic"] else "periodic"})
    # THE MARKER-NAME FLOOR. A set of names that matches nothing disables the check
    # it feeds SILENTLY, so every condition this evaluator models is asserted to
    # occur in the shipped source.
    markers = {condition: (condition in block) for condition in GUARD_RULES}
    ok = all(entry["needle_present"] and entry["case_compiles_every_guard"] is True
             for entry in rows)
    return {"passed": bool(ok and all(markers.values())),
            "rows": rows,
            "guard_conditions_present_in_the_shipped_source": markers,
            "guard_rules_modelled": sorted(GUARD_RULES),
            "note": "BACKWARD is bound to 1, so the FORWARD arms of the "
                    "verbatim-copied curl are dead on every case; ZM_X and ZM_Y "
                    "are dead on every Dcyl grid. Both are modelled here so a "
                    "needle placed inside either FAILS this leg rather than "
                    "reporting UNCAUGHT downstream."}


def leg_mutants_compile() -> Dict[str, Any]:
    """EVERY MUTANT MUST PARSE, and this is checked on the laptop.

    A needle that deletes the whole body of an ``if`` leaves an empty block, and the
    mutant is then an ``IndentationError`` rather than a defect. MEASURED ON
    THE GPU HOST 2026-08-30: ``m_one_axis_clear_dropped`` did exactly that and killed
    the campaign 146 seconds in, AFTER thirteen mutations had already been scored —
    so the cost of finding it on a device is the whole run, and the cost of finding
    it here is a syntax parse.

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
        mutated, _entry = _renamed(dedented(block.replace(old, new, 1)), name)
        try:
            ast.parse(mutated)
            rows.append({"mutation": name, "parsed": True})
        except SyntaxError as exc:  # noqa: PERF203 - one row per mutation
            rows.append({"mutation": name, "parsed": False,
                         "error": f"{type(exc).__name__}: {exc}"})
    # AND THE PRISTINE BLOCK ITSELF, so a leg that reported every mutant broken
    # because the SHIPPED text stopped parsing says which it was.
    try:
        ast.parse(dedented(block))
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
    """The two halves must be the certified kernels' own lines, VERBATIM.

    PARSED OUT OF THE SHIPPED TEXT, never re-implemented — a test that
    re-implemented the kernel's assembly would MIRROR a planted defect instead of
    executing it.
    """
    fused = dedented(shipped_block())

    def read(relative: str) -> str:
        with open(os.path.join(API_ROOT, relative), encoding="utf-8") as handle:
            return handle.read()

    curl = read("meep_gpu/triton_kernels/cylindrical_complex.py")
    constitutive = read("meep_gpu/triton_kernels/complex_fused_electric_pair.py")

    def statements(text: str, start: str, end: str) -> List[str]:
        """LOGICAL statements, not physical lines.

        Continuation lines are joined by bracket depth and runs of whitespace are
        collapsed, so a REFLOW — which changes no token and no float32 bit — does
        not fire this check, while any changed token, operand order or paren does.
        """
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
        "    # --- stores: u then f (kernels.py:193-198 order), both planes ---")
    constitutive_lines = statements(
        constitutive, "        kp_0 = tl.load(kp0 + i, mask=live, other=0.0)",
        "else:  # pragma: no cover - laptop path")
    # The END anchor is EXCLUSIVE, so the fused body is read to a sentinel PAST its
    # last statement and that statement is added back by name.
    last = "    tl.store(h2 + 2 * idx + 1, a_im, mask=live)"
    if last not in fused:
        return {"passed": False,
                "reason": "the fused body no longer ends at the h2 store; this "
                          "leg's end anchor is stale and it measured nothing"}
    fused_statements = set(statements(
        fused, "    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)", last))
    fused_statements.add(last.strip())

    curl_missing = [line for line in curl_lines if line not in fused_statements]
    constitutive_missing = [line for line in constitutive_lines
                            if line not in fused_statements]
    # The three seam lines the weld ADDS, and the only ones.
    seam_lines = [f"src_re = v{n}_re" for n in range(3)]
    seam_present = [line for line in seam_lines if line in fused_statements]
    # And the three the weld REMOVES: the constitutive reload of D.
    reload_absent = not any(f"src_re = tl.load(g{n}" in fused for n in range(3))
    return {
        "passed": bool(not curl_missing and not constitutive_missing
                       and len(seam_present) == 3 and reload_absent),
        "certified_curl_statements": len(curl_lines),
        "curl_statements_missing_from_the_fused_body": curl_missing,
        "certified_constitutive_statements": len(constitutive_lines),
        "constitutive_statements_missing_from_the_fused_body": constitutive_missing,
        "seam_lines_present": seam_present,
        "constitutive_reload_of_D_absent": reload_absent,
        "note": "parsed out of the shipped files; nothing here re-implements the "
                "kernel's arithmetic, which is what stops a planted defect from "
                "being mirrored instead of executed.",
    }


def leg_refusal(cp, probe) -> Dict[str, Any]:
    """The source seam, the m split and the polarization clause, measured BY NAME.

    THE ELECTRIC SOURCE IS THE WHOLE PRODUCT HERE. With the deposit repair carried
    it is ADMITTED; the clause still consults ``deposit_repair.repairable`` per run,
    and an UNDECLARED source list is still a refusal rather than an assumed ``()``.
    """
    from meep_gpu.sources import ContinuousEnvelope, VolumeSource  # noqa: PLC0415
    from meep_gpu.triton_kernels import (  # noqa: PLC0415
        cylindrical_fused_electric_pair as product)

    case = CASES[1]
    driver = build_driver(cp, case[1], case[2], case[3], case[4], case[5], SEED,
                          False, case[7])
    try:
        fields, pml = driver.fields, driver.pml
        undeclared = product.cylindrical_fused_electric_pair_coverage(
            fields, pml, None, probe=probe)
        quiet = product.cylindrical_fused_electric_pair_coverage(
            fields, pml, (), probe=probe)
        electric = VolumeSource(grid=fields.grid, component="Ez",
                                center=(0.5, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                                envelope=ContinuousEnvelope(frequency=1.0))
        carried = product.cylindrical_fused_electric_pair_coverage(
            fields, pml, (electric,), probe=probe)
        magnetic = VolumeSource(grid=fields.grid, component="Hy",
                                center=(0.5, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                                envelope=ContinuousEnvelope(frequency=1.0))
        magnetic_cov = product.cylindrical_fused_electric_pair_coverage(
            fields, pml, (magnetic,), probe=probe)
        no_probe = product.cylindrical_fused_electric_pair_coverage(
            fields, pml, (), probe=None)
    finally:
        driver.close()

    # m = 0 UNDER COMPLEX STORAGE IS ADMITTED (2026-09-04, the M_ZERO arm); the
    # REAL-storage m = 0 run is still cylindrical_triton's and must be refused BY
    # NAME on the storage clause — the split between the two products moved from
    # m to force_complex_fields, and both directions are measured here.
    zero = build_driver(cp, case[1], 0, case[3], case[4], case[5], SEED, False,
                        case[7])
    try:
        m0 = product.cylindrical_fused_electric_pair_coverage(
            zero.fields, zero.pml, (), probe=probe)
    finally:
        zero.close()
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415

    real = FdtdDriver(cell_size=case[1], resolution=12.0, cylindrical=True, m=0,
                      force_complex_fields=False, courant=float(case[7]),
                      boundaries=(("periodic", "periodic", "metallic")
                                  if case[4] else "periodic"),
                      prefer_gpu=True, gpu_id=0)
    try:
        real.setup_pml(dict(case[5]))
        m0_real = product.cylindrical_fused_electric_pair_coverage(
            real.fields, real.pml, (), probe=probe)
    finally:
        real.close()

    return {
        "passed": bool(not undeclared.covered
                       and any("was not declared" in r for r in undeclared.reasons)
                       and quiet.covered
                       and carried.covered
                       and magnetic_cov.covered
                       and not no_probe.covered
                       and m0.covered
                       and not m0_real.covered
                       and any("force_complex_fields" in r
                               for r in m0_real.reasons)),
        "undeclared_sources_refused": not undeclared.covered,
        "undeclared_named": [r for r in undeclared.reasons
                             if "was not declared" in r],
        "quiet_seam_admitted": quiet.covered,
        "electric_source_CARRIED": carried.covered,
        "electric_reasons": list(carried.reasons),
        "magnetic_source_is_irrelevant_to_this_seam": magnetic_cov.covered,
        "missing_expansion_probe_refused": not no_probe.covered,
        "missing_probe_named": list(no_probe.reasons)[:3],
        "m_zero_complex_admitted": m0.covered,
        "m_zero_complex_reasons": list(m0.reasons)[:3],
        "m_zero_real_storage_refused": not m0_real.covered,
        "m_zero_real_storage_named": [r for r in m0_real.reasons
                                      if "force_complex_fields" in r][:2],
        "note": "the electric clause is the one this product turns on: all sixteen "
                "Dcyl corpus rows declare an electric source, so with the repair "
                "off this arm admits ZERO rows.",
    }


# ===========================================================================
# Host mutations
# ===========================================================================

def run_host_mutations(cp, case, product, probe) -> List[Dict[str, Any]]:
    """Corrupt the TABLES the host binds — defects no source rewrite can reach.

    The kernel takes twelve coefficient pointers, two i*m/r rows, three inv_eps
    volumes and a prefix, and never asks which lattice or which target they came
    from. Each of these is a silent half-cell, wrong-sign or wrong-component error
    rather than a crash.
    """
    import meep_gpu.driver as driver_module  # noqa: PLC0415
    from meep_gpu.triton_kernels.launch import CupyPointer, _flat  # noqa: PLC0415

    (label, cell, m, accurate, z_metallic, pml_spec, _steps, courant) = case
    rows: List[Dict[str, Any]] = []

    def one(name: str, corrupt: Callable[[Any, Any], None]) -> Dict[str, Any]:
        built = [build_driver(cp, cell, m, accurate, z_metallic, pml_spec, SEED,
                              False, courant) for _ in range(2)]
        reference, actual = built
        counter: Dict[str, int] = {}
        undo: Callable[[], Any] = lambda: None
        try:
            kernel = CountingKernel(
                product.cyl_complex_fused_curl_constitutive_D_kernel())
            plan = product.plan_cylindrical_fused_electric_pair(
                actual.fields, actual.pml, tuple(actual._sources), num_warps=1,
                kernel=kernel, probe=probe)
            if plan is None:
                return {"leg": "host_mutation", "label": name, "passed": False,
                        "reason": "the product refused the host-mutation case"}
            # THE DRIVER, not just its PML: one of these needs the fields'
            # own material volumes, and passing the pml alone put `actual`
            # out of scope in the corrupter.
            corrupt(plan, actual)
            route, _ = fused_route(plan, actual, tuple(actual._sources))
            roles = {id(reference.fields): "array", id(actual.fields): "fused"}
            undo = install(driver_module, [(actual.fields, route)], counter, roles)
            for _ in range(MUTATION_STEPS):
                reference.step()
                actual.step()
            cp.cuda.runtime.deviceSynchronize()
            divergence = first_divergence(snapshot(cp, actual),
                                          snapshot(cp, reference))
            caught = divergence is not None
            return {"leg": "host_mutation", "label": name, "case": label,
                    "caught": caught, "null_confirmed": not caught,
                    "first_divergence": divergence,
                    "launches_observed": kernel.calls,
                    "passed": bool(caught and kernel.calls == MUTATION_STEPS)}
        finally:
            undo()
            for target in built:
                try:
                    target.close()
                except Exception:  # noqa: BLE001
                    pass
            cp.get_default_memory_pool().free_all_blocks()

    def half_integer_curl(plan, driver) -> None:
        """``step_D`` reads the INTEGER lattice; hand it the half-integer one.

        THE MIRROR IMAGE OF THE MAGNETIC TWIN'S CHOICE, and a silent half-cell
        error in the absorber profile that converges to a slightly worse absorber
        and nothing else. This is the one the builder is the single place to get
        wrong.
        """
        plan._curl_coefficients = tuple(
            CupyPointer(_flat(getattr(driver.pml, f"{stem}_{axis}_h")))
            for axis in "xyz" for stem in ("kms", "sinv"))

    def integer_constitutive(plan, driver) -> None:
        """``update_E`` takes the HALF-INTEGER kps/kms; hand it the integer ones."""
        plan._e_coefficients = tuple(
            CupyPointer(_flat(getattr(driver.pml, f"{stem}_{axis}")))
            for axis in "xyz" for stem in ("kps", "kms"))

    def swapped_imr(plan, driver) -> None:
        """Bind target 2's i*m/r row to target 0 and vice versa. The two differ by
        the target's OWN radial Yee shift AND by the sign of the term."""
        plan._imr_rows = tuple(reversed(plan._imr_rows))

    def epsilon_for_its_inverse(plan, driver) -> None:
        """Bind component 0's inv_eps to the PERMITTIVITY volume instead.

        THE ROTATION THIS REPLACES WAS INERT, and measuring that is why it is gone
        rather than merely re-armed: with an ISOTROPIC epsilon the engine hands the
        same array back for all three components (``Fields.has_component_epsilon``
        is False), so rotating the three bindings binds each pointer to the volume
        it already had and no word can move. Measured on the GPU host 2026-08-30: 0
        differing words over three complete steps.

        ``eps`` and ``inv_eps`` are two DIFFERENT volumes of the same shape and
        dtype that the engine really holds, so this is the same class of defect —
        a host binding the kernel cannot check — and it is live on every fixture.
        """
        from meep_gpu.triton_kernels.launch import CupyPointer as _P  # noqa: PLC0415

        epsilon = getattr(driver.fields, "eps", None)
        assert epsilon is not None, "the fixture exposes no eps volume to mis-bind"
        plan._inv_eps = (_P(epsilon),) + tuple(plan._inv_eps[1:])

    for name, corrupt in (
            ("curl_takes_the_half_integer_lattice", half_integer_curl),
            ("constitutive_takes_the_integer_lattice", integer_constitutive),
            ("imr_rows_swapped_between_targets", swapped_imr),
            ("inv_eps_bound_to_the_permittivity_volume", epsilon_for_its_inverse)):
        row = one(name, corrupt)
        rows.append(row)
        log(f"[host_mutation] {name}: caught={row.get('caught')} "
            f"passed={row.get('passed')}")
    return rows


# ===========================================================================
# Main
# ===========================================================================

def probe_record(explicit: Optional[str]) -> Tuple[Optional[Dict[str, Any]],
                                                   List[str], Optional[str]]:
    """The keep-cut expansion probe artifact, or a NAMED refusal.

    A PLACEHOLDER IS REFUSED BY NAME: ``expansion_license`` takes the verdict dict,
    not a path, and handing it a string produces a refusal about the harness
    reported as one about the corpus. So the artifact is loaded here and the loaded
    record is what travels.
    """
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
        except Exception as exc:  # noqa: BLE001 - unreadable == missing
            reasons.append(f"probe artifact unreadable ({path}): {exc}")
            continue
        if not isinstance(record, dict):
            reasons.append(f"probe artifact is not a verdict dict: {path}")
            continue
        return record, reasons, path
    return None, reasons, None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True,
                        help="results DIRECTORY; partial rows land here as they do")
    parser.add_argument("--probe-artifact", default=None)
    parser.add_argument("--steps", type=int, default=None,
                        help="override every case's step budget")
    parser.add_argument("--subnormal-policy", default="keep",
                        choices=("keep", "flush"))
    args = parser.parse_args(list(argv) if argv is not None else None)
    out = args.out
    os.makedirs(out, exist_ok=True)
    started = time.time()

    results: Dict[str, Any] = {
        "gate": "triton_cylindrical_fused_electric_pair",
        "family_module": FAMILY_MODULE,
        "cases": [[case[0], list(case[1]), case[2], case[3], case[4],
                   dict(case[5]), case[6], case[7]] for case in CASES],
        "carry_cases": [CASES[index][0] for index in CARRY_CASES],
        "seam_passes": list(SEAM_PASSES),
        "subnormal_policy": args.subnormal_policy,
        "rows": [], "skipped": {}, "failures": [],
    }

    def flush() -> None:
        try:
            from gate_provenance import stamp as _stamp  # noqa: PLC0415
            _stamp(results)
        except Exception:  # noqa: BLE001 - the stamp must not hide a result
            pass
        with open(os.path.join(out, "gate.json"), "w", encoding="utf-8") as handle:
            handle.write(json.dumps(results, indent=2, sort_keys=True,
                                    default=str) + "\n")

    # --- laptop legs, which run everywhere ---------------------------------
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

    # THE POLICY IS INSTALLED BEFORE ANY DEVICE COMPILE. One authority per process.
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
        cylindrical_fused_electric_pair as product)
    expansion = cx._resolve_expansion(record)
    results["licensed_expansion"] = expansion
    if expansion is None:
        results["failures"].append("the expansion licence refused this record")
        results["verdict"] = "REFUSED"
        results["elapsed_seconds"] = time.time() - started
        flush()
        return 1

    # --- the QUIET family: every arm, no deposit in the seam ----------------
    for case in CASES:
        row = run_leg(cp, "product", case, product, record, steps=args.steps)
        passed, failures = verdict_of(
            row, require_launches=row["steps_budget"])
        row["passed"], row["failures"] = passed, failures
        results["rows"].append(row)
        log(f"[product] {case[0]}: passed={passed} "
            f"launches={row.get('fused_kernel_launches')} "
            f"moved={row.get('arrays_ever_moved')}")
        flush()

    # --- the CARRY family: a real electric deposit IN the seam ---------------
    # THE PRODUCT IS WORTH ZERO CORPUS ROWS WITHOUT THESE ROWS.
    for index in CARRY_CASES:
        case = CASES[index]
        row = run_leg(cp, "carry", case, product, record, steps=args.steps,
                      electric=True)
        passed, failures = verdict_of(row, require_launches=row["steps_budget"],
                                      require_repairs=True)
        row["passed"], row["failures"] = passed, failures
        results["rows"].append(row)
        log(f"[carry] {case[0]}: passed={passed} "
            f"repairs={row.get('deposit_repairs')} "
            f"launches={row.get('fused_kernel_launches')}")
        flush()

    # --- the CARRY NULL CONTROL: the same plan, bracket REMOVED -------------
    # It MUST diverge. A carry family whose null control agrees is a family whose
    # bracket does nothing, and every carry row above would be passing for a reason
    # that has nothing to do with the repair.
    for index in CARRY_CASES[:1]:
        case = CASES[index]
        row = run_leg(cp, "carry_null_control", case, product, record,
                      steps=MUTATION_STEPS, electric=True, bracket=False)
        passed, failures = verdict_of(row, require_identical=False,
                                      require_moved=False)
        row["passed"], row["failures"] = passed, failures
        row["expectation"] = "DIVERGENCE REQUIRED"
        results["rows"].append(row)
        log(f"[carry_null_control] {case[0]}: passed={passed} "
            f"diverged={row.get('first_divergence') is not None}")
        flush()

    # --- the ARM OVERRIDE ---------------------------------------------------
    # The constexpr IS the licence; flipping it measures whether the licence is
    # load-bearing on this family. On the cylindrical arm it is EXPECTED TO BE
    # INERT and the row is scored as a NULL WITH that reason, never silently
    # dropped.
    other = 0 if int(expansion) == 1 else 1
    row = run_leg(cp, "expansion_override", CASES[1], product, record,
                  steps=MUTATION_STEPS, expansion=other)
    row["expectation"] = "null"
    row["null_confirmed"] = row.get("first_divergence") is None
    row["reason"] = (
        "the two arms differ only where a complex product's coefficient has a "
        "NONZERO real word; every general complex product here takes one whose "
        "real word is a zero (the i*m/r rows are exactly +0.0) and this family "
        "compiles no Bloch rotation. The binding is INERT here and is still made "
        "from a measured artifact, because a widening would make it live.")
    row["passed"] = bool(row["null_confirmed"]
                         and row.get("fused_kernel_launches") == MUTATION_STEPS)
    results["rows"].append(row)
    log(f"[expansion_override] arm {expansion}->{other}: "
        f"null_confirmed={row['null_confirmed']}")
    flush()

    # --- refusals -----------------------------------------------------------
    results["rows"].append({"leg": "refusal", "device": True,
                            **leg_refusal(cp, record)})
    log(f"[refusal] passed={results['rows'][-1]['passed']}")
    flush()

    # --- armed source mutations --------------------------------------------
    cases = {case[0]: case for case in CASES}
    block = shipped_block()
    for name, (case_name, old, new, expectation) in MUTATIONS.items():
        if old not in block:
            results["rows"].append({
                "leg": "mutation", "label": name, "passed": False,
                "reason": "the needle is ABSENT from the shipped source; this "
                          "mutation armed NOTHING"})
            flush()
            continue
        mutated, entry = _renamed(dedented(block.replace(old, new, 1)), name)
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
            "arrays_moved": row.get("arrays_ever_moved"),
            # A CATCH breaks the step loop at its first divergence, so it owes AT
            # LEAST ONE launch, not the whole budget; a NULL runs the budget out and
            # owes all of it.
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

    # --- host mutations: what no source rewrite can reach -------------------
    results["rows"].extend(run_host_mutations(cp, CASES[1], product, record))
    flush()

    rows = results["rows"]
    results["verdict"] = ("PASS" if all(row.get("passed") for row in rows)
                          else "FAIL")
    results["failed_rows"] = [row.get("label", row.get("case", row.get("leg")))
                              for row in rows if not row.get("passed")]
    results["elapsed_seconds"] = time.time() - started
    import triton_device_identity  # noqa: PLC0415

    results["environment"] = triton_device_identity.record({
        "numpy": np.__version__,
        "cupy": getattr(cp, "__version__", None),
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
    # THE KEYS THE LEDGER TOOLS READ ride at the top of the block (2026-09-13):
    # ``seed_triton_welds.policy_line`` names the policy from ``resolved`` /
    # ``requested`` and the cache directory from ``cache_dir``, and refused this
    # artifact by name while they sat one level down inside ``stamp``.
    _stamp = results.get("policy_stamp") if isinstance(results.get("policy_stamp"), dict) else {}
    results["subnormal_policy"] = {
        "policy": args.subnormal_policy,
        "requested": _stamp.get("requested") or args.subnormal_policy,
        "resolved": _stamp.get("resolved") or args.subnormal_policy,
        "cache_dir": _stamp.get("cache_dir"),
        "stamp": results.get("policy_stamp"),
        "installed_before_the_first_device_compile": True,
    }
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
        "parity/meep_gpu/probe_triton_cylindrical_fused_electric_pair.py": sha256_of(os.path.abspath(__file__)),
        "meep_gpu/triton_kernels/cylindrical_complex.py": sha256_of(
            os.path.join(API_ROOT,
                         "meep_gpu/triton_kernels/cylindrical_complex.py")),
        "meep_gpu/triton_kernels/complex_fields.py": sha256_of(
            os.path.join(API_ROOT, "meep_gpu/triton_kernels/complex_fields.py")),
        "meep_gpu/triton_kernels/complex_fused_electric_pair.py": sha256_of(
            os.path.join(
                API_ROOT,
                "meep_gpu/triton_kernels/complex_fused_electric_pair.py")),
        "meep_gpu/deposit_repair.py": sha256_of(
            os.path.join(API_ROOT, "meep_gpu/deposit_repair.py")),
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
