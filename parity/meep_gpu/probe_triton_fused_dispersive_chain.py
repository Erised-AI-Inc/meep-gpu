"""CUDA byte gate for the fused dispersive D/E/ADE chain.

THE CLAIM THIS GATE IS ALLOWED TO SUPPORT: for every configuration the chain
product admits, one launch of ``fused_curl_dispersive_E_ade`` leaves the engine in
a state that is BIT-IDENTICAL, per complete driver step, to

  * the CuPy array path (``stepping.step_D`` / ``update_E`` / ``update_P``), and
  * the two SEPARATELY CERTIFIED Triton products it replaces
    (``DispersiveFusedPairPlan`` + ``FusedAdeStatePlan``),

over every allocated volume: the primaries, the split-field PML auxiliaries, the
constitutive ``f_w`` history, every pole ``P`` / ``P_prev`` and the shared scratch
buffer.  Comparison is on the uint32 view.  ``allclose`` appears nowhere.

WHAT THE GATE REFUSES TO INFER.

* **Bytes alone cannot prove the fused path ran.**  A silent fallback to the array
  path is byte-identical to the array path by construction, so identity is the
  WEAKEST possible evidence about which code executed.  Every launch is therefore
  counted through a proxy that owns the kernel object, every substitution is
  counted in the installer, and one ARMED HARNESS MUTATION removes the
  substitution to show the counters — not the bytes — are what catches it.
* **A no-op agreeing with a no-op is trivially identical.**  Every compared array
  must MOVE during the leg; a leg where the electric trio is frozen in all three
  routes is armed and must be caught by the moved-state census.
* **Zero-init is a fixed point of the constitutive sub-step.**  The signed-zero leg
  therefore seeds a +-0 LATTICE rather than a zero field, and carries a census with
  a floor: a census of zero is VACUOUS, not passed.

STEP BUDGETS are stated per leg below and totalled in the artifact.  Every leg
runs three drivers in lockstep and compares after each COMPLETE ``driver.step()``
— not after a sub-step — because the seam this product closes is interior to a
step and a sub-step comparison would let an error cancel across the boundary.

POLICY, stamped and unchanged: num_warps=1 (the settled cross-sub-step policy),
BLOCK=kernels.DEFAULT_BLOCK, enable_fp_fusion=kernels.ENABLE_FP_FUSION (False).
This gate does not tune and does not time.

Usage::

    CUDA_VISIBLE_DEVICES=<verified-empty device> python -u \
        probe_triton_fused_dispersive_chain.py --out results/<dir>/chain.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import re
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

SEED = 20260814

#: (name, boundaries, susceptibility kind, sigma form, steps)
CASES: Tuple[Tuple[str, Tuple[str, str, str], str, str, int], ...] = (
    ("scalar_all_periodic", ("periodic", "periodic", "periodic"),
     "lorentzian", "scalar_all", 12),
    ("volume_subset_metallic_x", ("metallic", "periodic", "periodic"),
     "drude", "volume_subset", 12),
    ("volume_all_metallic_xy", ("metallic", "metallic", "periodic"),
     "lorentzian", "volume_all", 12),
)

#: The signed-zero leg: a +-0 lattice, not a zero field.
SIGNED_ZERO_STEPS = 4
#: Absorber thicknesses for the signed-zero rows.  Thin drives the split-field
#: coefficients negative (which is what makes a manufactured -0.0 possible);
#: thick is carried beside it so the classification itself is exercised.
SIGNED_ZERO_PML_CELLS = (2, 6)
#: Steps per armed mutation.  A mutation that needs more than this to become
#: byte-visible is reported as a null WITH its launch and PTX evidence.
MUTATION_STEPS = 3
#: The floor below which the signed-zero leg is VACUOUS rather than passed.
SIGNED_ZERO_FLOOR = 64

_TEMPORARY: List[str] = []


def log(message: str) -> None:
    print(message, flush=True)


def save(payload: Dict[str, Any], path: str) -> None:
    """Serialise the payload, provenance-stamped, atomically.

    THE STAMP IS HERE AND NOT AT THE CALL SITES, which is the whole reason this
    gate could pass while recording nothing about its own bytes. ``main`` calls
    this eleven times as the legs land, and a stamp bolted onto the last call
    would leave every partial artifact — including the one an aborted run leaves
    behind, which is the artifact a failure is read from — unattributable.

    ``gate_provenance.stamp`` writes ``imported_source_sha256`` (every repo module
    THIS PROCESS imported, enumerated from ``sys.modules``) and
    ``canonical_verdict``. It does NOT touch ``source_sha256``: that key is this
    gate's own curated binding list, a different fact, and the two are kept apart
    deliberately — see gate_provenance's module docstring.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    temporary = path + ".tmp"
    from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415

    # THE POLICY STAMP IS RE-READ, NOT CARRIED. Taken once at install time it
    # records ``installed: true`` with ``ftz_removed: 0``, ``nvrtc_calls: 0`` and
    # every Triton counter at zero — because nothing had compiled yet. Measured on
    # the 2026-08-19 08:21Z run of this gate: all ten Triton counters and both CuPy
    # counters read zero in an artifact whose kernels demonstrably compiled and
    # launched 36 times. That record cannot tell an installed policy apart from an
    # inert one, which is the same vacuity the signed-zero census exists to refuse
    # one level down. Re-reading at every checkpoint makes the counters describe
    # the run, including the partial artifact an aborted run leaves behind.
    if "subnormal_policy" in payload:
        from meep_gpu import subnormal_policy as _policy  # noqa: PLC0415

        payload["subnormal_policy"] = _policy.policy_stamp()
    _stamp_provenance(payload)
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def source_hashes() -> Dict[str, str]:
    """The bytes under certification, stamped into the artifact itself.

    The runner echoes the same hashes to its log, but a certification artifact
    that names its own sources is auditable on its own: the one drift this gate
    has seen was a staged planner (``launch.py``) that predated the commit the
    artifact was recorded against, and it was only recoverable because a sibling
    artifact carried the stamp.  Every module here executes inside a certified
    leg: the product, its two oracle routes, the planner that selects them, the
    array path they are compared against, and this probe.
    """
    names = (
        "meep_gpu/backends.py", "meep_gpu/driver.py", "meep_gpu/stepping.py",
        "meep_gpu/dispersion.py", "meep_gpu/fields.py",
        # The two files that decide what CONFIGURATION this run is, as opposed to
        # what arithmetic it performs. Both are consulted before the first device
        # compile and neither was in this list when the record recorded nothing.
        "meep_gpu/subnormal_policy.py",
        "parity/meep_gpu/gate_provenance.py",
        "meep_gpu/triton_kernels/coverage.py",
        "meep_gpu/triton_kernels/launch.py",
        "meep_gpu/triton_kernels/kernels.py",
        "meep_gpu/triton_kernels/dispersive_update_e.py",
        "meep_gpu/triton_kernels/dispersive_fused_pair.py",
        "meep_gpu/triton_kernels/fused_ade_state.py",
        "meep_gpu/triton_kernels/fused_dispersive_chain.py",
        "meep_gpu/test_triton_fused_dispersive_chain.py",
        "parity/meep_gpu/probe_triton_fused_dispersive_chain.py",
    )
    return {name: sha256(os.path.join(API_ROOT, name)) for name in names}


# ---------------------------------------------------------------------------
# Drivers and state inventory
# ---------------------------------------------------------------------------

def build_driver(cp, boundaries, kind: str, sigma_form: str, seed: int,
                 signed_zero: bool = False, pml_cells: int = 6):
    """One dispersive PML driver, seeded identically for every route."""
    from meep_gpu.dispersion import DRUDE, LORENTZIAN, Susceptibility  # noqa: PLC0415
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415

    driver = FdtdDriver(
        cell_size=(4.0, 3.5, 0.0), resolution=16.0, dimensions=2,
        force_complex_fields=False, courant=0.35, boundaries=boundaries,
        prefer_gpu=True, gpu_id=0,
    )
    shape = driver.shape
    count = int(np.prod(shape))
    index = np.arange(count, dtype=np.float32).reshape(shape)
    epsilon = np.ascontiguousarray(
        (1.55 + 0.30 * np.sin(index * np.float32(0.027))).astype(np.float32))
    driver.set_epsilon(cp.asarray(epsilon))
    # The absorber is what makes f_w differ from the stored E, which is the whole
    # reason the drive-field trap is byte-visible.  THICKNESS IS LOAD-BEARING for
    # the signed-zero leg and nowhere else: a thin absorber drives the split-field
    # ``kms`` NEGATIVE, and without a negative coefficient a zero-init lattice
    # cannot manufacture a ``-0.0`` at all (the class is unreachable rather than
    # untested).  The precedent and the measured thickness/census relation are in
    # gate_triton_cylindrical_complex.signed_zero_reach.
    driver.setup_pml({"x": pml_cells, "y": pml_cells})

    if sigma_form == "scalar_all":
        sigma: Any = 0.37
    else:
        base = np.ascontiguousarray(
            (0.22 + 0.31 * (0.5 + 0.5 * np.sin(index * np.float32(0.019))))
            .astype(np.float32))
        volume = cp.asarray(base)
        sigma = ({"Ex": volume, "Ey": 0.0, "Ez": volume * cp.float32(0.73)}
                 if sigma_form == "volume_subset"
                 else {"Ex": volume, "Ey": volume * cp.float32(0.81),
                       "Ez": volume * cp.float32(0.63)})
    driver.add_susceptibility(
        Susceptibility(
            frequency=0.72 if kind == "lorentzian" else 0.61,
            gamma=0.06 if kind == "lorentzian" else 0.04,
            kind=LORENTZIAN if kind == "lorentzian" else DRUDE),
        sigma)

    state = driver.fields.polarizations[0]
    if signed_zero:
        # +-0 LATTICE.  Zero-init is a fixed point of the constitutive sub-step,
        # so a zero field would certify nothing; the sign pattern is what the
        # kernel's sign-of-zero behaviour is visible in.
        flat = np.arange(count, dtype=np.int64).reshape(shape)
        lattice = np.where((flat % 2) == 0, np.float32(0.0), np.float32(-0.0))
        lattice = np.ascontiguousarray(lattice.astype(np.float32))
        alternate = np.ascontiguousarray(
            np.where(((flat // 2) % 2) == 0, np.float32(-0.0), np.float32(0.0))
            .astype(np.float32))
        for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz"):
            driver.set_field(name, cp.asarray(lattice))
        for component in state.driven():
            state.P[component][...] = cp.asarray(lattice)
            state.P_prev[component][...] = cp.asarray(alternate)
        for name in ("f_w_Ex", "f_w_Ey", "f_w_Ez", "Ex", "Ey", "Ez"):
            getattr(driver.fields, name)[...] = cp.asarray(alternate)
        return driver

    rng = np.random.default_rng(seed)
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz"):
        driver.set_field(name, cp.asarray(np.ascontiguousarray(
            rng.uniform(-0.25, 0.25, size=shape).astype(np.float32))))
    for component in state.driven():
        state.P[component][...] = cp.asarray(np.ascontiguousarray(
            rng.uniform(-0.08, 0.08, size=shape).astype(np.float32)))
        state.P_prev[component][...] = cp.asarray(np.ascontiguousarray(
            rng.uniform(-0.08, 0.08, size=shape).astype(np.float32)))
    for name in ("f_w_Ex", "f_w_Ey", "f_w_Ez"):
        getattr(driver.fields, name)[...] = cp.asarray(np.ascontiguousarray(
            rng.uniform(-0.05, 0.05, size=shape).astype(np.float32)))
    return driver


#: The names that MUST appear in the dynamic inventory.  The inventory is scanned
#: rather than listed so a renamed or newly allocated volume cannot silently drop
#: out of the comparison; this set is the tripwire for the scan itself shrinking.
REQUIRED = (
    "Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
    "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
    "f_w_Ex", "f_w_Ey", "f_w_Ez", "f_w_Hx", "f_w_Hy", "f_w_Hz",
)


def inventory(driver) -> Dict[str, Any]:
    """Every device volume the step can touch, found by scanning rather than listing."""
    fields = driver.fields
    shape = tuple(fields.grid.shape)
    found: Dict[str, Any] = {}
    for name, value in vars(fields).items():
        if name.startswith("__"):
            continue
        if getattr(value, "shape", None) == shape and getattr(value, "dtype", None) is not None:
            found[name] = value
    for index, state in enumerate(fields.polarizations):
        for component in state.driven():
            found[f"P{index}_{component}"] = state.P[component]
            found[f"Pprev{index}_{component}"] = state.P_prev[component]
        if getattr(state, "_scratch", None) is not None:
            found[f"Pscratch{index}"] = state._scratch
    missing = [name for name in REQUIRED if name not in found]
    if missing:
        raise AssertionError(
            f"the state scan lost {missing}; the comparison inventory is not complete")
    return found


def words(cp, array) -> np.ndarray:
    return np.ascontiguousarray(cp.asnumpy(array)).view(np.uint32).ravel()


def snapshot(cp, driver) -> Dict[str, np.ndarray]:
    return {name: words(cp, array) for name, array in inventory(driver).items()}


#: The ONLY name a route may hold that another does not.  ``Fields._fmp_scratch``
#: is the array path's lazily allocated temporary for ``D - sum P``
#: (fields.py:1099-1105); it is overwritten wholesale on every use, carries nothing
#: between steps, and every Triton route performs that subtraction inside a kernel
#: and never allocates it.  Any OTHER asymmetry is a real inventory difference and
#: is reported as a divergence rather than waved through.
ALLOWED_ASYMMETRY = frozenset({"_fmp_scratch", "_fmp_scratch_by_component"})

#: Material inputs.  These must be IDENTICAL across routes and UNCHANGED by the
#: step; requiring them to "move" would be exactly backwards.
MATERIAL = ("eps", "inv_eps")


def first_divergence(left: Dict[str, np.ndarray],
                     right: Dict[str, np.ndarray]) -> Optional[Dict[str, Any]]:
    """The FIRST array and element that differ, in a stable order.  None when identical."""
    asymmetry = set(left).symmetric_difference(right)
    if asymmetry - ALLOWED_ASYMMETRY:
        return {"array": "<inventory>", "detail":
                f"unexpected asymmetry {sorted(asymmetry - ALLOWED_ASYMMETRY)}; "
                f"left={sorted(left)} right={sorted(right)}"}
    for name in sorted(set(left) & set(right)):
        a, b = left[name], right[name]
        if a.shape != b.shape:
            return {"array": name, "detail": f"shape {a.shape} versus {b.shape}"}
        mask = a != b
        if not mask.any():
            continue
        where = int(np.argmax(mask))
        return {
            "array": name, "index": where,
            "differing_words": int(np.count_nonzero(mask)),
            "left_word": int(a[where]), "right_word": int(b[where]),
            "left_float": float(a[where:where + 1].view(np.float32)[0]),
            "right_float": float(b[where:where + 1].view(np.float32)[0]),
        }
    return None


def moved(before: Dict[str, np.ndarray], after: Dict[str, np.ndarray]) -> List[str]:
    return sorted(name for name in before
                  if name in after and bool((before[name] != after[name]).any()))


def signed_zero_census(state: Dict[str, np.ndarray]) -> Dict[str, int]:
    """Negative-zero words per array.  A census of zero means the leg proved nothing."""
    return {name: int(np.count_nonzero(value == np.uint32(0x80000000)))
            for name, value in state.items()}


#: The volumes the CHAIN KERNEL ITSELF writes.  A negative zero found here was
#: MANUFACTURED by the arithmetic under gate; one found in P_prev or the retired
#: scratch at step 1 was merely the seeding, rotated.  The distinction is the
#: difference between measuring the kernel and measuring the harness.
KERNEL_OUTPUTS = (
    "Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz",
    "Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez",
    "P0_Ex", "P0_Ey", "P0_Ez",
)


def manufactured_signed_zeros(state: Dict[str, np.ndarray]) -> int:
    """Negative-zero words in the chain kernel's OWN outputs."""
    return sum(int(np.count_nonzero(value == np.uint32(0x80000000)))
               for name, value in state.items() if name in KERNEL_OUTPUTS)


def min_pml_coefficient(cp, pml) -> float:
    """The most negative split-field coefficient this route's kernels multiply by.

    A ``-0.0`` can only be MANUFACTURED where some coefficient is negative: with
    every coefficient positive, an all-zero lattice reaches ``+0.0`` and stays
    there whatever the kernel does, so a silent census would say nothing about the
    kernel.  This number is what separates "unreachable by construction" from
    "reachable and never reached", which is the difference between a classified
    row and a failed one.
    """
    smallest = float("inf")
    for axis in "xyz":
        for stem in ("kms", "sinv", "kps"):
            for suffix in ("", "_h"):
                array = getattr(pml, f"{stem}_{axis}{suffix}", None)
                if array is None:
                    continue
                smallest = min(smallest, float(cp.asnumpy(array).min()))
    return smallest


def signed_zero_reach(peak: int, min_coefficient: float) -> str:
    """Classify a signed-zero leg: live, unreachable by construction, or VACUOUS.

    Transcribed from the same three-way rule the cylindrical-complex gate carries
    (gate_triton_cylindrical_complex.signed_zero_reach), for the same reason: a row
    that could not have reached the class must not be failed for a property of the
    absorber, and a row that could have and did not must not be passed.
    """
    if peak > 0:
        return "live"
    if min_coefficient >= 0.0:
        return "unreachable_no_negative_coefficient"
    return "VACUOUS"


# ---------------------------------------------------------------------------
# Launch counting and substitution counting
# ---------------------------------------------------------------------------

class CountingKernel:
    """Owns the JIT kernel and counts every launch through the plan's ``run``.

    This is the proof-of-execution instrument.  It is NOT a wrapper around the
    plan: the plan calls ``kernel[grid](...)`` and nothing else, so a plan that
    silently did not launch, or a driver that never reached the plan, shows up as
    a count of zero rather than as a passing byte comparison.
    """

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
    """Every compiled specialization's PTX, for the stale-binary tripwire."""
    out: List[str] = []
    for per_device in (getattr(jit, "cache", None) or {}).values():
        for compiled in per_device.values():
            asm = getattr(compiled, "asm", None)
            if asm and "ptx" in asm:
                out.append(asm["ptx"])
    return out


def install(driver_module, routes, counter: Dict[str, int], roles: Dict[int, str]):
    """Replace the driver's sub-steps for the fields objects named in ``routes``.

    Counting is PER ROUTE.  A single global counter would mix the reference
    driver's legitimate array-path calls with a fallback in the fused route, which
    is precisely the event this instrument exists to see.
    """
    names = ("step_B", "update_H", "step_D", "update_E", "update_P")
    originals = {name: getattr(driver_module, name) for name in names}

    def bump(role: str, kind: str, name: str) -> None:
        key = f"{role}/{kind}:{name}"
        counter[key] = counter.get(key, 0) + 1

    def replacement(name):
        def wrapper(fields, pml=None):
            role = roles.get(id(fields), "unknown")
            plan = next((candidate for owner, candidate in routes
                         if fields is owner), None)
            if plan is None or name not in plan.plans:
                bump(role, "array_path", name)
                return originals[name](fields, pml)
            entry = plan.plans[name]
            bump(role, "substituted", name)
            if name == "update_P":
                for pole_plan in entry:
                    pole_plan.run(fields.drive_field)
            else:
                entry.run()
            return None
        return wrapper

    for name in names:
        setattr(driver_module, name, replacement(name))
    return lambda: [setattr(driver_module, name, function)
                    for name, function in originals.items()]


class ChainRoute:
    """The plan bundle installed for the fused-chain driver.

    ``step_D`` carries the whole electric chain; ``update_E`` and ``update_P`` are
    absorbed.  ``update_P`` maps to an EMPTY LIST rather than being left out: the
    installer must see the slot as replaced, so a fallback to the array-path
    ``update_P`` is a counter event rather than an invisible correction.
    """

    def __init__(self, plans: Dict[str, Any]) -> None:
        self.plans = plans


def chain_plans(driver, chain_module, kernel: Any, install_chain: bool = True,
                block: Optional[int] = None):
    """The chain route, or (armed) the same route with the substitution removed."""
    from meep_gpu.triton_kernels import plan_step  # noqa: PLC0415

    reference_plan = plan_step(
        driver.fields, driver.pml, fuse=True, sources=tuple(driver._sources),
        num_warps=1, fuse_ade=True)
    plan = chain_module.plan_fused_dispersive_chain(
        driver.fields, driver.pml, tuple(driver._sources), block=block,
        num_warps=1, kernel=kernel)
    if plan is None:
        verdict = chain_module.fused_dispersive_chain_coverage(
            driver.fields, driver.pml, tuple(driver._sources))
        raise AssertionError(f"the chain product refused the case: {verdict.reasons}")
    plans = {
        "step_B": reference_plan.plans["step_B"],
        "update_H": reference_plan.plans["update_H"],
    }
    if install_chain:
        plans["step_D"] = plan
        plans["update_E"] = _Absorbed("update_E", plan)
        plans["update_P"] = []
    return ChainRoute(plans), plan, reference_plan


class _Absorbed:
    """The sentinel left where a sub-step was absorbed by the chain launch."""

    __slots__ = ("name", "absorbed_by")

    def __init__(self, name: str, absorbed_by: Any) -> None:
        self.name = name
        self.absorbed_by = absorbed_by

    def run(self) -> None:
        return None


# ---------------------------------------------------------------------------
# The three-route leg
# ---------------------------------------------------------------------------

def run_leg(cp, name: str, boundaries, kind: str, sigma_form: str, steps: int,
            chain_module, mutant: Any = None, install_chain: bool = True,
            freeze_electric: bool = False, signed_zero: bool = False,
            pml_cells: int = 6) -> Dict[str, Any]:
    """Four routes in lockstep; stop at the FIRST byte divergence.

    The oracles, all seeded identically:

    * ``array``   - ``stepping``'s CuPy path, no Triton in the electric trio;
    * ``separate``- the certified pair plus the certified one-launch ADE state;
    * ``legacy``  - the same pair plus the certified PER-COMPONENT ADE plans, which
      is a DIFFERENT sequence of launches for the same arithmetic and therefore a
      second, independent statement of what "the existing separate products" mean;
    * ``chain``   - the product under gate.
    """
    from meep_gpu import driver as driver_module  # noqa: PLC0415
    from meep_gpu.triton_kernels import plan_step  # noqa: PLC0415

    reference = build_driver(cp, boundaries, kind, sigma_form, SEED, signed_zero,
                             pml_cells)
    separate = build_driver(cp, boundaries, kind, sigma_form, SEED, signed_zero,
                            pml_cells)
    legacy = build_driver(cp, boundaries, kind, sigma_form, SEED, signed_zero,
                          pml_cells)
    chain = build_driver(cp, boundaries, kind, sigma_form, SEED, signed_zero,
                         pml_cells)
    undo: Callable[[], Any] = lambda: None
    counter: Dict[str, int] = {}
    kernel = CountingKernel(
        mutant if mutant is not None
        else chain_module.fused_curl_dispersive_E_ade_kernel())
    row: Dict[str, Any] = {
        "leg": name, "steps_budget": steps, "shape": list(reference.shape),
        "boundaries": list(boundaries), "kind": kind, "sigma": sigma_form,
        "signed_zero_lattice": bool(signed_zero), "pml_cells": int(pml_cells),
        "chain_substituted": bool(install_chain),
        "electric_frozen": bool(freeze_electric),
        "first_divergence": None, "control_divergence": None,
    }
    try:
        separate_plan = plan_step(
            separate.fields, separate.pml, fuse=True,
            sources=tuple(separate._sources), num_warps=1, fuse_ade=True)
        legacy_plan = plan_step(
            legacy.fields, legacy.pml, fuse=True,
            sources=tuple(legacy._sources), num_warps=1, fuse_ade=False)
        route, plan, _ = chain_plans(chain, chain_module, kernel, install_chain)
        row["driven"] = list(chain.fields.polarizations[0].driven())
        row["pole_counts"] = list(plan.counts)
        row["separate_products"] = {
            key: type(value).__name__ for key, value in separate_plan.plans.items()}
        row["separate_update_P"] = [type(value).__name__
                                    for value in separate_plan.plans["update_P"]]
        row["legacy_update_P"] = [type(value).__name__
                                  for value in legacy_plan.plans["update_P"]]
        row["chain_products"] = {key: type(value).__name__
                                 for key, value in route.plans.items()}
        expected = {"step_B": "FusedPairPlan", "update_H": "NoopPlan",
                    "step_D": "DispersiveFusedPairPlan", "update_E": "NoopPlan",
                    "update_P": "list"}
        if row["separate_products"] != expected:
            raise AssertionError(
                f"{name}: the separate oracle is not the certified product set "
                f"{row['separate_products']}; reasons={separate_plan.reasons}")
        if row["separate_update_P"] != ["FusedAdeStatePlan"]:
            raise AssertionError(
                f"{name}: the separate ADE oracle is {row['separate_update_P']}")
        if row["legacy_update_P"] != ["AdeUpdatePPlan"]:
            raise AssertionError(
                f"{name}: the per-component ADE oracle is {row['legacy_update_P']}; "
                f"reasons={legacy_plan.reasons}")

        roles = {id(reference.fields): "array", id(separate.fields): "separate",
                 id(legacy.fields): "legacy", id(chain.fields): "chain"}
        routes = [(separate.fields, separate_plan), (legacy.fields, legacy_plan),
                  (chain.fields, route)]
        if freeze_electric:
            # ARMED: every route's electric trio is inert.  All four then agree
            # trivially and only the moved-state census can refuse it.
            frozen = {"step_D": _Absorbed("step_D", None),
                      "update_E": _Absorbed("update_E", None), "update_P": []}
            route.plans.update(frozen)
            separate_plan.plans.update(frozen)
            legacy_plan.plans.update(frozen)
            routes.append((reference.fields, ChainRoute(dict(frozen))))
        undo = install(driver_module, routes, counter, roles)

        opening = snapshot(cp, chain)
        row["per_step"] = []
        census_trail: List[int] = []
        manufactured_trail: List[int] = []
        if signed_zero:
            row["min_pml_coefficient"] = min_pml_coefficient(cp, chain.pml)
            row["seeded_signed_zero_words"] = sum(
                signed_zero_census(opening).values())
        started = time.time()
        for step in range(1, steps + 1):
            before = snapshot(cp, chain)
            reference.step()
            separate.step()
            legacy.step()
            chain.step()
            cp.cuda.runtime.deviceSynchronize()
            after_reference = snapshot(cp, reference)
            after_separate = snapshot(cp, separate)
            after_legacy = snapshot(cp, legacy)
            after_chain = snapshot(cp, chain)
            versus_array = first_divergence(after_chain, after_reference)
            versus_separate = first_divergence(after_chain, after_separate)
            versus_legacy = first_divergence(after_chain, after_legacy)
            control = (first_divergence(after_separate, after_reference)
                       or first_divergence(after_legacy, after_reference))
            step_moved = moved(before, after_chain)
            point = {
                "step": step,
                "chain_vs_array": versus_array,
                "chain_vs_separate_fused_ade": versus_separate,
                "chain_vs_separate_per_component_ade": versus_legacy,
                "oracle_control_vs_array": control,
                "arrays_moved": len(step_moved),
                "chain_launches": kernel.calls,
            }
            if signed_zero:
                # The census is taken PER STEP and the PEAK is what the leg
                # carries: the negative zeros a step manufactures are consumed by
                # the next one, so a final-state census reports their absence
                # rather than their absence of effect.
                census = signed_zero_census(after_chain)
                point["signed_zero_words"] = sum(census.values())
                point["signed_zero_manufactured"] = manufactured_signed_zeros(
                    after_chain)
                point["signed_zero_by_array"] = {key: value for key, value
                                                 in census.items() if value}
                census_trail.append(point["signed_zero_words"])
                manufactured_trail.append(point["signed_zero_manufactured"])
            row["per_step"].append(point)
            divergence = versus_array or versus_separate or versus_legacy
            log(f"  {name} step {step}/{steps} identical={divergence is None} "
                f"control={control is None} moved={len(step_moved)} "
                f"launches={kernel.calls} ({time.time() - started:.1f} s)")
            row["first_divergence"] = divergence
            row["control_divergence"] = control
            if divergence is not None:
                row["diverged_at_step"] = step
                break

        # Every array must have moved at least once during the leg, or the leg
        # certified two frozen states against each other.
        final = snapshot(cp, chain)
        ever_moved = moved(opening, final)
        material = [name for name in final if name in MATERIAL]
        row["arrays_total"] = len(final)
        row["arrays_compared"] = sorted(final)
        row["arrays_ever_moved"] = len(ever_moved)
        row["arrays_never_moved"] = sorted(
            set(final) - set(ever_moved) - set(material))
        row["material_arrays"] = sorted(material)
        row["material_changed"] = sorted(name for name in material
                                         if name in ever_moved)
        row["inventory_asymmetry_vs_array"] = sorted(
            set(final).symmetric_difference(snapshot(cp, reference)))
        row["launches"] = dict(counter)
        row["chain_kernel_launches"] = kernel.calls
        row["launch_grids"] = sorted({grid for grid in kernel.grids})
        row["ptx_specializations"] = len(kernel_ptx(kernel.jit))
        if signed_zero:
            row["signed_zero_peak"] = max(census_trail, default=0)
            row["signed_zero_trail"] = census_trail
            row["signed_zero_manufactured_peak"] = max(manufactured_trail, default=0)
            row["signed_zero_manufactured_trail"] = manufactured_trail
            # Reach is decided on the MANUFACTURED census: a carried seed proves
            # the harness can hold a -0.0, not that the kernel can produce one.
            row["signed_zero_reach"] = signed_zero_reach(
                row["signed_zero_manufactured_peak"], row["min_pml_coefficient"])
        return row
    finally:
        undo()
        for target in (reference, separate, legacy, chain):
            try:
                target.close()
            except Exception:  # noqa: BLE001 - a close failure must not hide a result
                pass
        cp.get_default_memory_pool().free_all_blocks()


def verdict_of(row: Dict[str, Any], *, require_identical: bool = True,
               require_launches: Optional[int] = None,
               require_moved: bool = True) -> Tuple[bool, List[str]]:
    """The leg's pass conditions, stated rather than implied."""
    failures: List[str] = []
    identical = row.get("first_divergence") is None
    if require_identical and not identical:
        failures.append(f"byte divergence: {row['first_divergence']}")
    if row.get("control_divergence") is not None:
        failures.append(
            f"an ORACLE control itself diverged from the array path, so this leg "
            f"could not have measured the chain: {row['control_divergence']}")
    if require_launches is not None and row.get("chain_kernel_launches") != require_launches:
        failures.append(
            f"chain kernel launched {row.get('chain_kernel_launches')} times, "
            f"expected {require_launches}: the fused path is not what executed")
    if require_moved:
        never = row.get("arrays_never_moved") or []
        if never:
            failures.append(f"VACUOUS: these arrays never moved: {never}")
    changed_material = row.get("material_changed") or []
    if changed_material:
        failures.append(
            f"a material input changed during the leg: {changed_material}")
    launches = row.get("launches", {}) or {}
    fell_back = {key: value for key, value in launches.items()
                 if key.startswith("chain/array_path:")
                 and key.split(":")[-1] in ("step_D", "update_E", "update_P")}
    if fell_back:
        failures.append(
            f"the chain route reached the array path for an absorbed sub-step: "
            f"{fell_back}")
    return (not failures), failures


# ---------------------------------------------------------------------------
# Armed kernel mutations
# ---------------------------------------------------------------------------

def shipped_source(chain_module) -> str:
    return textwrap.dedent(
        inspect.getsource(chain_module.fused_curl_dispersive_E_ade.fn))


def compile_mutant(source: str, kernel_name: str) -> Any:
    """Compile a renamed mutant.  The rename is what keeps the JIT cache honest."""
    header = ("import triton\nimport triton.language as tl\n"
              "METALLIC = tl.constexpr(1)\n\n")
    handle = tempfile.NamedTemporaryFile(
        "w", suffix="_mutated_chain.py", delete=False, encoding="utf-8")
    handle.write(header + source)
    handle.close()
    _TEMPORARY.append(handle.name)
    spec = importlib.util.spec_from_file_location(
        "triton_mutated_chain_" + str(len(_TEMPORARY)), handle.name)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = module  # type: ignore[union-attr]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return getattr(module, kernel_name)


#: Every mutation is (id, why it is armed, expectation, rewrite).  ``expectation``
#: is what the design predicts; the artifact records what was MEASURED, and a
#: predicted null that turns out visible (or the reverse) is reported either way.
def mutation_table() -> Tuple[Tuple[str, str, str, Callable[[str], Tuple[str, int]]], ...]:

    def m1_drive_reloaded(source: str) -> Tuple[str, int]:
        """PREDICTED NULL: read f_w back instead of keeping the register.

        This is the fusion itself, undone.  A float32 store followed by a load of
        the same address is the identity, so the null is the claim that the
        eliminated round trip changed no value — and it is the ONLY mutation whose
        null is the product's thesis rather than a limitation of the case."""
        needle = "(c_drive * (sv0 * src0))"
        replacement = ("(c_drive * (sv0 * tl.load(w0 + idx, mask=live, other=0.0)))")
        return source.replace(needle, replacement), source.count(needle)

    def m2_drive_is_stored_E(source: str) -> Tuple[str, int]:
        """The drive-field trap: feed the ADE the PML-accumulated stored E.

        MEEP update_pols.cpp:44 takes f_w when it exists; using the stored E agrees
        exactly OUTSIDE the absorber, so this is a smooth, plausible wrong answer
        that only a PML case can see.  Both PML sides are 6 cells thick here."""
        needle = "(c_drive * (sv0 * src0))"
        return source.replace(needle, "(c_drive * (sv0 * a0v))"), source.count(needle)

    def m3_store_hoisted_above_history_load(source: str) -> Tuple[str, int]:
        """The aliasing hazard, armed: arm 1 stores before arm 0's history load.

        ``o1`` IS the volume ``q0p`` names (dispersion.py:686-691's rotation), so
        this is the exact reordering the hoisting discipline exists to forbid."""
        needle = "h0 = tl.load(q0p + idx, mask=live, other=0.0)"
        hits = source.count(needle)
        pattern = re.compile(
            r"(?P<indent>[ ]*)if ADE0:\n(?P=indent)[ ]{4}"
            r"h0 = tl\.load\(q0p \+ idx, mask=live, other=0\.0\)")
        match = pattern.search(source)
        if match is None:
            return source, 0
        indent = match.group("indent")
        early = (
            f"{indent}if ADE1:\n"
            f"{indent}    if SIGMA_IS_VOLUME1:\n"
            f"{indent}        sv1e = tl.load(sg1 + idx, mask=live, other=0.0)\n"
            f"{indent}    else:\n"
            f"{indent}        sv1e = sg1\n"
            f"{indent}    tl.store(o1 + idx,\n"
            f"{indent}             ((pa1 * c_now) + (c_prev * tl.load("
            f"q1p + idx, mask=live, other=0.0))) +\n"
            f"{indent}             (c_drive * (sv1e * src1)), mask=live)\n")
        return source[:match.start()] + early + source[match.start():], hits

    def m4_pole_register_crossed(source: str) -> Tuple[str, int]:
        """Arm 0 advances arm 1's pole: the register reuse pointed at the wrong volume."""
        needle = "((pa0 * c_now) + (c_prev * h0))"
        return source.replace(needle, "((pa1 * c_now) + (c_prev * h0))"), source.count(needle)

    def m5_history_coefficients_swapped(source: str) -> Tuple[str, int]:
        """c_now and c_prev exchanged: the second-order recurrence's two histories."""
        needle = "((pa0 * c_now) + (c_prev * h0))"
        return source.replace(needle, "((pa0 * c_prev) + (c_now * h0))"), source.count(needle)

    def m6_destination_rotated(source: str) -> Tuple[str, int]:
        """Arm 0's result lands in arm 1's destination: the rotation off by one."""
        needle = "tl.store(o0 + idx,"
        return source.replace(needle, "tl.store(o1 + idx,"), source.count(needle)

    def m7_association_changed(source: str) -> Tuple[str, int]:
        """Right-associated recurrence: same algebra, different float32 rounding."""
        pattern = re.compile(
            r"\(\(pa0 \* c_now\) \+ \(c_prev \* h0\)\) \+\s*"
            r"\(c_drive \* \(sv0 \* src0\)\)")
        replacement = "(pa0 * c_now) + ((c_prev * h0) + (c_drive * (sv0 * src0)))"
        mutated, hits = pattern.subn(replacement, source)
        return mutated, hits

    def m8_commuted_multiply(source: str) -> Tuple[str, int]:
        """PREDICTED NULL: IEEE multiplication commutes; only the PTX may move."""
        needle = "(pa0 * c_now)"
        return source.replace(needle, "(c_now * pa0)"), source.count(needle)

    def m10_pole_reloaded(source: str) -> Tuple[str, int]:
        """PREDICTED NULL: reload P instead of reusing the subtraction's register.

        The OTHER half of the fusion, undone.  ``a0`` is read and never written by
        this kernel, so the reload must return the same word the subtraction
        loaded; a divergence here would mean the register reuse is not the
        identity the product claims it is."""
        needle = "((pa0 * c_now)"
        replacement = "((tl.load(a0 + idx, mask=live, other=0.0) * c_now)"
        return source.replace(needle, replacement), source.count(needle)

    def m9_sigma_dropped(source: str) -> Tuple[str, int]:
        """sigma dropped from the drive term — the strength of the whole term."""
        needle = "(c_drive * (sv0 * src0))"
        return source.replace(needle, "(c_drive * src0)"), source.count(needle)

    return (
        ("m1_drive_reloaded", "the fusion undone", "null", m1_drive_reloaded),
        ("m2_drive_is_stored_E", "the drive-field trap", "caught", m2_drive_is_stored_E),
        ("m3_store_hoisted", "the aliasing hazard", "caught",
         m3_store_hoisted_above_history_load),
        ("m4_pole_crossed", "register reuse crossed", "caught", m4_pole_register_crossed),
        ("m5_coefficients_swapped", "history coefficients", "caught",
         m5_history_coefficients_swapped),
        ("m6_destination_rotated", "rotation off by one", "caught", m6_destination_rotated),
        ("m7_association", "float32 association", "caught", m7_association_changed),
        ("m8_commuted_multiply", "commuted multiply", "null", m8_commuted_multiply),
        ("m9_sigma_dropped", "term strength", "caught", m9_sigma_dropped),
        ("m10_pole_reloaded", "the pole register reuse undone", "null",
         m10_pole_reloaded),
    )


def run_mutations(cp, chain_module, pristine_ptx: Sequence[str]) -> List[Dict[str, Any]]:
    source = shipped_source(chain_module)
    rows: List[Dict[str, Any]] = []
    case = CASES[0]
    for identifier, why, expectation, rewrite in mutation_table():
        mutated, hits = rewrite(source)
        entry: Dict[str, Any] = {
            "mutation": identifier, "why": why, "expectation": expectation,
            "needle_hits": hits, "steps_budget": MUTATION_STEPS,
        }
        if hits == 0 or mutated == source:
            entry["status"] = "NEEDLE-MISSED"
            entry["failure"] = "the rewrite matched nothing; this mutation was never armed"
            rows.append(entry)
            log(f"  {identifier}: NEEDLE-MISSED")
            continue
        kernel_name = "mutant_" + identifier
        mutated = mutated.replace("def fused_curl_dispersive_E_ade(",
                                  f"def {kernel_name}(")
        mutant = compile_mutant(mutated, kernel_name)
        try:
            row = run_leg(cp, f"{case[0]}::{identifier}", case[1], case[2], case[3],
                          MUTATION_STEPS, chain_module, mutant=mutant)
        except AssertionError as exc:
            entry["status"] = "HARNESS-REFUSED"
            entry["failure"] = str(exc)
            rows.append(entry)
            continue
        # PTX EVIDENCE.  The mutant carries a different function name, so the raw
        # text always differs and would be worthless as evidence; the name is
        # normalized away first.  What survives is a real instruction difference —
        # or nothing, which is itself a finding (the compiler canonicalized the
        # rewrite) and is recorded as PTX-IDENTICAL rather than as a passing null.
        normalize = lambda text: text.replace(kernel_name, "fused_curl_dispersive_E_ade")
        mutant_ptx = [normalize(text) for text in kernel_ptx(mutant)]
        entry["launches"] = row.get("chain_kernel_launches")
        entry["ptx_specializations"] = len(mutant_ptx)
        entry["ptx_differs_from_shipped"] = bool(
            mutant_ptx and pristine_ptx and
            all(text not in set(pristine_ptx) for text in mutant_ptx))
        entry["ptx_bytes"] = [len(text) for text in mutant_ptx]
        entry["shipped_ptx_bytes"] = [len(text) for text in pristine_ptx]
        entry["first_divergence"] = row.get("first_divergence")
        entry["diverged_at_step"] = row.get("diverged_at_step")
        entry["arrays_never_moved"] = row.get("arrays_never_moved")
        caught = row.get("first_divergence") is not None
        if not entry["launches"]:
            entry["status"] = "DISARMED"
            entry["failure"] = "the mutant kernel was never launched"
        elif not entry["ptx_differs_from_shipped"]:
            entry["status"] = ("STALE-BINARY" if caught is False and expectation == "caught"
                               else "PTX-IDENTICAL")
            entry["note"] = (
                "the mutant compiled to PTX already present for the shipped kernel; "
                "the source change was canonicalized away and nothing was tested by it")
        elif caught:
            entry["status"] = "CAUGHT"
        else:
            entry["status"] = "NULL"
            entry["note"] = (
                "launched, PTX differs, and the gate saw no byte difference: recorded "
                "as a measured null with its evidence")
        entry["agrees_with_expectation"] = (
            (expectation == "caught" and entry["status"] == "CAUGHT")
            or (expectation == "null" and entry["status"] in ("NULL", "PTX-IDENTICAL")))
        rows.append(entry)
        log(f"  {identifier}: {entry['status']} launches={entry['launches']} "
            f"ptx_differs={entry['ptx_differs_from_shipped']} "
            f"expectation={expectation}")
    return rows


# ---------------------------------------------------------------------------
# Refusal legs
# ---------------------------------------------------------------------------

def run_refusals(cp, chain_module) -> List[Dict[str, Any]]:
    """Configurations the product must refuse, on real device arrays, with reasons."""
    from types import SimpleNamespace  # noqa: PLC0415

    from meep_gpu.dispersion import LORENTZIAN, Susceptibility  # noqa: PLC0415

    rows: List[Dict[str, Any]] = []
    name, boundaries, kind, sigma_form, _ = CASES[0]

    driver = build_driver(cp, boundaries, kind, sigma_form, SEED)
    try:
        verdict = chain_module.fused_dispersive_chain_coverage(
            driver.fields, driver.pml, (SimpleNamespace(field_type="D"),))
        rows.append({
            "refusal": "electric_source_in_the_seam",
            "covered": verdict.covered, "reasons": list(verdict.reasons),
            "plan_is_none": chain_module.plan_fused_dispersive_chain(
                driver.fields, driver.pml,
                (SimpleNamespace(field_type="D"),)) is None})
        verdict = chain_module.fused_dispersive_chain_coverage(
            driver.fields, driver.pml, None)
        rows.append({
            "refusal": "source_set_not_declared",
            "covered": verdict.covered, "reasons": list(verdict.reasons),
            "plan_is_none": chain_module.plan_fused_dispersive_chain(
                driver.fields, driver.pml, None) is None})
    finally:
        driver.close()

    driver = build_driver(cp, boundaries, kind, sigma_form, SEED)
    try:
        driver.add_susceptibility(
            Susceptibility(frequency=0.91, gamma=0.05, kind=LORENTZIAN), 0.21)
        verdict = chain_module.fused_dispersive_chain_coverage(
            driver.fields, driver.pml, ())
        rows.append({
            "refusal": "two_susceptibilities_multipole",
            "covered": verdict.covered, "reasons": list(verdict.reasons),
            "plan_is_none": chain_module.plan_fused_dispersive_chain(
                driver.fields, driver.pml, ()) is None})
    finally:
        driver.close()
    cp.get_default_memory_pool().free_all_blocks()
    for row in rows:
        log(f"  refusal {row['refusal']}: covered={row['covered']} "
            f"plan_is_none={row['plan_is_none']}")
    return rows


# ---------------------------------------------------------------------------
# Harness-level armed mutations
# ---------------------------------------------------------------------------

def run_harness_arms(cp, chain_module) -> List[Dict[str, Any]]:
    """Two guards that byte identity cannot provide, each armed and each caught."""
    rows: List[Dict[str, Any]] = []
    name, boundaries, kind, sigma_form, _ = CASES[0]

    # ARM 1: silent fallback.  The chain is not installed; the driver runs the
    # array path.  Bytes are identical BY CONSTRUCTION and only the launch count
    # and the array-path counter can see it.
    row = run_leg(cp, f"{name}::ARM_silent_fallback", boundaries, kind, sigma_form,
                  MUTATION_STEPS, chain_module, install_chain=False)
    passed, failures = verdict_of(row, require_launches=MUTATION_STEPS)
    rows.append({
        "arm": "silent_fallback_to_array_path",
        "bytes_identical": row.get("first_divergence") is None,
        "chain_kernel_launches": row.get("chain_kernel_launches"),
        "array_path_step_D_calls": row.get("launches", {}).get(
            "chain/array_path:step_D"),
        "caught": not passed, "failures": failures,
        "status": "CAUGHT" if not passed else "DISARMED",
    })
    log(f"  ARM silent_fallback: caught={not passed} bytes_identical="
        f"{row.get('first_divergence') is None} failures={failures}")

    # ARM 2: the electric trio is inert in EVERY route.  All three agree, and the
    # moved-state census is the only thing that can refuse it.
    row = run_leg(cp, f"{name}::ARM_inert_electric", boundaries, kind, sigma_form,
                  MUTATION_STEPS, chain_module, freeze_electric=True)
    passed, failures = verdict_of(row, require_launches=0)
    rows.append({
        "arm": "inert_electric_trio_in_every_route",
        "bytes_identical": row.get("first_divergence") is None,
        "arrays_never_moved": row.get("arrays_never_moved"),
        "caught": not passed, "failures": failures,
        "status": "CAUGHT" if not passed else "DISARMED",
    })
    log(f"  ARM inert_electric: caught={not passed} never_moved="
        f"{len(row.get('arrays_never_moved') or [])}")
    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def environment(cp) -> Dict[str, Any]:
    import triton  # noqa: PLC0415

    properties = cp.cuda.runtime.getDeviceProperties(0)
    return {
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename,
        "python": sys.version.split()[0], "numpy": np.__version__,
        "cupy": cp.__version__, "triton": triton.__version__,
        "device": properties["name"].decode(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def policy_stamp(chain_module) -> Dict[str, Any]:
    from meep_gpu.triton_kernels.kernels import DEFAULT_BLOCK, ENABLE_FP_FUSION  # noqa: PLC0415

    return {
        "num_warps": chain_module.POLICY["num_warps"],
        "block": DEFAULT_BLOCK,
        "enable_fp_fusion": bool(ENABLE_FP_FUSION),
        "status": chain_module.POLICY["status"],
        "note": "settled configuration; this gate does not tune and does not time",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--skip-mutations", action="store_true")
    parser.add_argument(
        "--subnormal-policy", default="keep",
        help="the float32 subnormal policy to DRIVE EVERY EXECUTOR TO before the "
             "first device compile. Default 'keep' — the policy every record in "
             "triton_kernels/fingerprints.json is cut under. The 2026-08-14 run "
             "of this gate installed NOTHING, which left CuPy flushing (it "
             "appends -ftz=true unconditionally) beside a natively-keeping "
             "Triton: a mixed configuration attributable to no policy at all.")
    args = parser.parse_args(argv)

    import cupy as cp  # noqa: PLC0415
    from meep_gpu import backends, subnormal_policy  # noqa: PLC0415

    # BEFORE THE FIRST DEVICE COMPILE, and before the chain module is imported.
    # strict=True: a process that asked to keep and quietly did not is a process
    # whose bytes mean nothing, and this refuses at startup rather than writing an
    # artifact whose policy field is a wish. It is also what enforces the
    # CUPY_CACHE_DIR token rule (subnormal_policy.cupy_cache_reasons): CuPy's
    # cache key is computed ABOVE the seam the -ftz strip installs at, so a
    # directory shared with a flush run would serve flushed binaries under this
    # record's name.
    subnormal_policy.install_subnormal_policy(args.subnormal_policy, cupy=cp,
                                              strict=True)
    policy_record = subnormal_policy.policy_stamp()
    log(f"subnormal policy installed: {policy_record.get('policy')!r} "
        f"(requested {args.subnormal_policy!r}, "
        f"CUPY_CACHE_DIR={os.environ.get('CUPY_CACHE_DIR')!r})")

    from meep_gpu.triton_kernels import fused_dispersive_chain as chain_module  # noqa: PLC0415

    backends.guard_kernel_compilation(cp)
    payload: Dict[str, Any] = {
        "environment": environment(cp),
        "source_sha256": source_hashes(),
        "subnormal_policy": policy_record,
        "policy": policy_stamp(chain_module),
        "budgets": {
            "byte_cases": {case[0]: case[4] for case in CASES},
            "signed_zero_steps": SIGNED_ZERO_STEPS,
            "steps_per_mutation": MUTATION_STEPS,
            "signed_zero_pml_cells": list(SIGNED_ZERO_PML_CELLS),
            "total_steps": sum(case[4] for case in CASES)
                           + SIGNED_ZERO_STEPS * len(SIGNED_ZERO_PML_CELLS)
                           + MUTATION_STEPS * (len(mutation_table()) + 2),
        },
        "cases": [], "signed_zero": [], "mutations": [], "harness_arms": [],
        "refusals": [],
    }
    save(payload, args.out)
    failures: List[str] = []

    for case in CASES:
        log(f"case {case[0]}: {case[2]} sigma={case[3]} boundaries={case[1]}")
        row = run_leg(cp, case[0], case[1], case[2], case[3], case[4], chain_module)
        passed, leg_failures = verdict_of(row, require_launches=case[4])
        row["passed"] = passed
        row["failures"] = leg_failures
        payload["cases"].append(row)
        save(payload, args.out)
        if not passed:
            failures.extend(f"{case[0]}: {reason}" for reason in leg_failures)
            payload["summary"] = {"status": "FAILED", "failures": failures}
            save(payload, args.out)
            log(f"STOP: {case[0]} failed: {leg_failures}")
            return 1

    # SIGNED-ZERO ROWS.  Two absorber thicknesses, because thickness is what
    # decides whether the +-0 class is reachable at all: a thin absorber drives the
    # split-field coefficients negative, a thick one does not.  A row that could
    # not have reached the class is CLASSIFIED, not failed; a row that could have
    # and did not is VACUOUS; and the leg as a whole must contain at least one live
    # row or its zero half measured nothing.
    signed_zero_rows: List[Dict[str, Any]] = []
    for cells in SIGNED_ZERO_PML_CELLS:
        log(f"signed-zero lattice leg, absorber {cells} cells "
            f"({SIGNED_ZERO_STEPS} steps)")
        row = run_leg(cp, f"signed_zero_lattice_pml{cells}", CASES[0][1], CASES[0][2],
                      "volume_all", SIGNED_ZERO_STEPS, chain_module,
                      signed_zero=True, pml_cells=cells)
        passed, leg_failures = verdict_of(row, require_launches=SIGNED_ZERO_STEPS,
                                          require_moved=False)
        if row.get("seeded_signed_zero_words", 0) < SIGNED_ZERO_FLOOR:
            passed = False
            leg_failures.append(
                f"the SEEDING itself carried only "
                f"{row.get('seeded_signed_zero_words')} negative-zero words, below "
                f"the floor {SIGNED_ZERO_FLOOR}: this row never presented the class")
        if row.get("signed_zero_reach") == "VACUOUS":
            passed = False
            leg_failures.append(
                f"VACUOUS: peak census 0 with a NEGATIVE coefficient present "
                f"(min={row.get('min_pml_coefficient'):.6g}); the class was "
                f"reachable and was never reached")
        row["passed"] = passed
        row["failures"] = leg_failures
        signed_zero_rows.append(row)
        payload["signed_zero"] = signed_zero_rows
        save(payload, args.out)
        log(f"  reach={row.get('signed_zero_reach')} peak={row.get('signed_zero_peak')} "
            f"min_coefficient={row.get('min_pml_coefficient')} seeded="
            f"{row.get('seeded_signed_zero_words')}")
        if not passed:
            payload["summary"] = {"status": "FAILED", "failures": leg_failures}
            save(payload, args.out)
            log(f"STOP: signed-zero row {cells} failed: {leg_failures}")
            return 1
    live = [row for row in signed_zero_rows
            if row.get("signed_zero_reach") == "live"
            and (row.get("signed_zero_manufactured_peak") or 0) >= SIGNED_ZERO_FLOOR]
    if not live:
        payload["summary"] = {
            "status": "FAILED",
            "failures": [
                "no signed-zero row MANUFACTURED at least "
                f"{SIGNED_ZERO_FLOOR} negative-zero words in the chain kernel's own "
                "outputs: the leg swept the +-0 class without measuring it once"]}
        save(payload, args.out)
        log("STOP: no signed-zero row was live")
        return 1

    payload["refusals"] = run_refusals(cp, chain_module)
    save(payload, args.out)

    log("harness arms")
    payload["harness_arms"] = run_harness_arms(cp, chain_module)
    save(payload, args.out)

    if not args.skip_mutations:
        log("armed kernel mutations")
        pristine = kernel_ptx(chain_module.fused_curl_dispersive_E_ade)
        payload["pristine_ptx_specializations"] = len(pristine)
        payload["mutations"] = run_mutations(cp, chain_module, pristine)
        save(payload, args.out)

    disarmed = [row for row in payload["mutations"]
                if row.get("status") in ("DISARMED", "NEEDLE-MISSED", "STALE-BINARY",
                                         "HARNESS-REFUSED")]
    unexpected = [row for row in payload["mutations"]
                  if not row.get("agrees_with_expectation")
                  and row.get("status") not in ("PTX-IDENTICAL",)]
    arms_missed = [row for row in payload["harness_arms"] if not row.get("caught")]
    refusals_admitted = [row for row in payload["refusals"] if row.get("covered")
                         or not row.get("plan_is_none")]
    status = "passed"
    if disarmed or arms_missed or refusals_admitted:
        status = "FAILED"
    payload["summary"] = {
        "status": status,
        "cases_exact": f"{len(payload['cases'])}/{len(CASES)}",
        "complete_steps_exact": sum(
            len(row.get("per_step", ())) for row in payload["cases"]),
        "oracles_per_step": "CuPy array path AND (DispersiveFusedPairPlan + FusedAdeStatePlan)",
        "arrays_compared": payload["cases"][0]["arrays_total"] if payload["cases"] else 0,
        "mutations_caught": sum(1 for row in payload["mutations"]
                                if row.get("status") == "CAUGHT"),
        "mutations_null": sum(1 for row in payload["mutations"]
                              if row.get("status") in ("NULL", "PTX-IDENTICAL")),
        "mutations_disarmed": [row["mutation"] for row in disarmed],
        "unexpected_outcomes": [row["mutation"] for row in unexpected],
        "harness_arms_caught": sum(1 for row in payload["harness_arms"]
                                   if row.get("caught")),
        "signed_zero_rows": [
            {"leg": row["leg"], "reach": row.get("signed_zero_reach"),
             "peak": row.get("signed_zero_peak"),
             "manufactured_peak": row.get("signed_zero_manufactured_peak"),
             "min_pml_coefficient": row.get("min_pml_coefficient"),
             "identical": row.get("first_divergence") is None}
            for row in (payload["signed_zero"] or ())],
        "scope": ("one susceptibility; scalar and volume sigma; all and subset "
                  "components; Lorentz and Drude; periodic and metallic; PML active"),
    }
    save(payload, args.out)
    log(f"CHAIN GATE {status.upper()}: {json.dumps(payload['summary'], sort_keys=True)}")
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
