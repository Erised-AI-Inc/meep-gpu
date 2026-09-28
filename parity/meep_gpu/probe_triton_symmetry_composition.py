"""Complete-driver CUDA gate for the nondispersive folded Triton composition.

The standalone folded curl gate is necessarily sub-step granular because the
driver's later fills overwrite its ownership-mask planes.  This complementary
gate asks a different question: does the experimental composer select the curl,
the source-adjacent near/far mirror fills, and the constitutive update together,
and does that six-slot route reproduce the complete CuPy driver state exactly
after every step?

SECOND LEG, ADDED 2026-08-27 WITH THE FOLDED PAIR ROUTING (:func:`check_routing`).
The composition above is the UNFUSED one and is now asked for with ``fuse=False``;
until this round the same six plans came back under ``fuse=True`` only because the
composer blanket-refused every folded fusion, so the flag measured nothing and the
gate would have gone on passing whatever the routed answer became. Each case is now
also planned WITH fusion, and the expected answer is not uniform: the three
sourceless cases must route both pairs into both their slots, and the two cases with
an ON-PLANE source must route the pair whose seam the injection does not cross while
refusing BY NAME the pair whose seam it does — neither folded family declares
``CARRIES_DEPOSIT_REPAIR``, so an in-seam deposit has no repaired admission.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
API_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
if API_ROOT not in sys.path:
    sys.path.insert(0, API_ROOT)


def source(component: str) -> Dict[str, Any]:
    """An on-plane source whose component is even under Mirror(Y, +1)."""
    return {
        "component": component,
        "frequency": 0.7,
        "center": (0.0, 0.0, 0.0),
        "size": (0.0, 0.0, 0.0),
        "amplitude": 0.75,
    }


# name, cell, boundaries, ((axis, phase), ...), source declarations, steps
CASES = (
    ("periodic_even", (3.0, 3.0, 0.0),
     ("periodic", "periodic", "periodic"), (("Y", 1),), (), 8),
    ("periodic_even_electric_on_plane", (3.0, 3.0, 0.0),
     ("periodic", "periodic", "periodic"), (("Y", 1),), (source("Ez"),), 10),
    ("periodic_even_magnetic_on_plane", (3.0, 3.0, 0.0),
     ("periodic", "periodic", "periodic"), (("Y", 1),), (source("Hy"),), 10),
    ("periodic_odd_phase_odd_count", (3.0, 3.0833333333333335, 0.0),
     ("periodic", "periodic", "periodic"), (("Y", -1),), (), 8),
    ("metallic_even", (3.0, 3.0, 0.0),
     ("metallic", "metallic", "periodic"), (("Y", 1),), (), 8),
    ("periodic_two_folds", (3.0, 3.0, 0.0),
     ("periodic", "periodic", "periodic"), (("X", 1), ("Y", 1)), (), 8),
)

EXPECTED = {
    "step_B": "FoldedPmlCurlPlan",
    "fill_B": "MirrorGhostFillPlan",
    "update_H": "ConstitutivePlan",
    "step_D": "FoldedPmlCurlPlan",
    "fill_D": "MirrorGhostFillPlan",
    "update_E": "ConstitutivePlan",
}

STATE_NAMES = (
    "Bx", "By", "Bz", "Dx", "Dy", "Dz",
    "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
    "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
    "f_w_Ex", "f_w_Ey", "f_w_Ez", "f_w_Hx", "f_w_Hy", "f_w_Hz",
)
PRIMARY_NAMES = ("Bx", "By", "Bz", "Dx", "Dy", "Dz")


def log(message: str) -> None:
    print(message, flush=True)


def save(payload: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_driver(cp, cell, boundaries, mirror_specs, source_specs, seed: int):
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415
    from meep_gpu.grid import Mirror  # noqa: PLC0415

    mirrors = tuple(Mirror(axis, phase) for axis, phase in mirror_specs)
    driver = FdtdDriver(
        cell_size=cell, resolution=12.0, dimensions=2,
        force_complex_fields=False, courant=0.35,
        boundaries=boundaries, symmetry=mirrors,
        prefer_gpu=True, gpu_id=0,
    )
    shape = driver.shape
    index = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    epsilon = np.ascontiguousarray(
        (1.45 + 0.30 * np.sin(index * np.float32(0.037))).astype(np.float32))
    driver.set_epsilon(cp.asarray(epsilon))
    folded_axes = {axis.lower() for axis, _phase in mirror_specs}
    driver.setup_pml({
        "x": {"high": 5} if "x" in folded_axes else 5,
        "y": {"high": 5} if "y" in folded_axes else 5,
    })
    for declaration in source_specs:
        driver.add_source(dict(declaration))
    rng = np.random.default_rng(seed)
    for name in PRIMARY_NAMES:
        values = np.ascontiguousarray(
            rng.uniform(-0.25, 0.25, size=shape).astype(np.float32))
        driver.set_field(name, cp.asarray(values))
    return driver


def state(driver) -> Dict[str, Any]:
    return {name: getattr(driver.fields, name) for name in STATE_NAMES
            if getattr(driver.fields, name, None) is not None}


def words(cp, array) -> np.ndarray:
    return np.ascontiguousarray(cp.asnumpy(array)).view(np.uint32).ravel()


def compare(cp, left, right) -> Dict[str, Any]:
    a = words(cp, left)
    b = words(cp, right)
    return {
        "bit_identical": bool(np.array_equal(a, b)),
        "differing_floats": int(np.count_nonzero(a != b)),
        "total_floats": int(a.size),
    }


def plan_types(plan) -> Dict[str, str]:
    return {name: type(entry).__name__ for name, entry in plan.plans.items()}


#: Which pair each on-plane source component sits INSIDE the seam of. The driver
#: injects an electric source between ``step_D`` and ``update_E`` (driver.py:3294)
#: and a magnetic one between ``step_B`` and ``update_H`` (driver.py:3283), so a
#: case's source decides which of the two pairs may fuse and which must refuse.
#: Neither folded family declares ``CARRIES_DEPOSIT_REPAIR``, so the refusal is the
#: whole answer for its own seam — there is no repaired admission to check for.
IN_SEAM_OF = {"E": ("D", "is electric"), "H": ("B", "is magnetic")}

#: What the composer must put in a routed seam's two slots, by pair.
ROUTED_PLANS = {"B": ("step_B", "update_H", "FoldedFusedMagneticPairPlan"),
                "D": ("step_D", "update_E", "FoldedFusedPairPlan")}


def _folded_family_modules():
    """The two folded family modules, for the flag-held-False leg. Imported lazily
    for the reason every family import in this tree is: they import Triton at module
    scope and this file must stay importable where there is none."""
    from meep_gpu.triton_kernels import (  # noqa: PLC0415
        folded_fused_magnetic_pair, folded_fused_pair)

    return {"B": folded_fused_magnetic_pair, "D": folded_fused_pair}


def check_routing(plan_step, candidate, name, declarations) -> Dict[str, Any]:
    """``fuse=True`` on this folded grid: routed where clean, refused where in-seam.

    THE HALF THE SIX-SLOT LEG CANNOT SEE. ``EXPECTED`` above pins the unfused route
    and would keep passing if the fused route were wrong in every way, because it is
    no longer the plan this gate installs. So the routed answer is asked for here,
    per case, and it is asked in the one place where the two halves are known to be
    about the SAME grid and the SAME declared source list.

    AND IT IS NOT A UNIFORM ANSWER, which is what makes it worth measuring rather
    than asserting once. Three of these cases declare no source and both pairs must
    take both their slots. Two declare an ON-PLANE source, and it lands INSIDE
    exactly one of the two seams — so that pair and the other are answered
    differently, on the same grid, in the same call.

    WHAT THE IN-SEAM PAIR MUST DO CHANGED ON 2026-08-30, AND THIS ASSERTION WAS
    INVERTED RATHER THAN DROPPED. It used to require the in-seam pair to be refused
    BY NAME and leave both sub-steps on their separate folded plans. Both folded
    families now declare :data:`CARRIES_DEPOSIT_REPAIR`, so the composer BRACKETS
    that pair instead: ``LeadingRepairPlan`` in the curl slot, ``TrailingRepairPlan``
    in the constitutive one, with the deposit and every cell the post-injection
    fills image it into saved before the launch and restored after.

    THE THING THE OLD ASSERTION WAS PROTECTING IS STILL PROTECTED, and by a stricter
    check than "is it refused". A routing that simply ADMITTED the in-seam pair —
    a bare ``FoldedFusedPairPlan`` in the curl slot and a ``NoopPlan`` in the other —
    would run a launch against a field the driver had not injected yet, and no byte
    comparison in this file could detect it, because this file installs the unfused
    plan. That shape is what the bracket assertion below refuses: the two repair
    plan classes are named, the trailing one must name the leading one as the plan
    it repairs, and a bare fused pair in an in-seam slot fails exactly as the old
    unrouted shape did.
    """
    plan = plan_step(candidate.fields, candidate.pml, fuse=True,
                     sources=tuple(candidate._sources), num_warps=1)
    types = plan_types(plan)
    refused = {IN_SEAM_OF[declaration["component"][0]][0]
               for declaration in declarations
               if declaration["component"][0] in IN_SEAM_OF}
    clauses = {declaration["component"][0]: IN_SEAM_OF[declaration["component"][0]][1]
               for declaration in declarations
               if declaration["component"][0] in IN_SEAM_OF}
    out: Dict[str, Any] = {
        "selected_under_fuse": types,
        "refusals_under_fuse": {key: list(value)
                                for key, value in plan.reasons.items()},
        "pairs_expected_refused": sorted(refused),
    }
    for pair, (curl, update, plan_name) in ROUTED_PLANS.items():
        if pair in refused:
            # THE CARRIED SEAM. Both slots must hold the repair plans, in order:
            # anything else here is either the pre-2026-08-30 refusal (which would
            # now be a regression) or, far worse, a bare fused pair consuming a
            # pre-injection field.
            if types.get(curl) != "LeadingRepairPlan" or \
                    types.get(update) != "TrailingRepairPlan":
                raise AssertionError(
                    f"{name}: the {pair} pair carries this case's in-seam source "
                    f"and must be BRACKETED by the deposit repair; got "
                    f"{curl}={types.get(curl)!r} {update}={types.get(update)!r}, "
                    f"reasons={plan.reasons.get(f'fused_pair_{pair}')}")
            leading = plan.plans[curl]
            trailing = plan.plans[update]
            # `TrailingRepairPlan._leading` is the save it restores from, and
            # `.absorbed_by` is the fused pair underneath — deposit_repair.py:713-716.
            if getattr(trailing, "_leading", None) is not leading:
                raise AssertionError(
                    f"{name}: {update}'s repair does not name {curl}'s save, so the "
                    f"restore would read a state nothing captured")
            if getattr(trailing, "absorbed_by", None) is not getattr(
                    leading, "inner", None):
                raise AssertionError(
                    f"{name}: {update} does not name the plan that absorbed it, so "
                    f"the composition would report it as array-path work")
            # And the bracket is around a REAL fused pair, not around nothing: a
            # bracket over a non-pair would report the seam as fused while the
            # sub-steps ran separately underneath it.
            if type(getattr(leading, "inner", None)).__name__ != plan_name:
                raise AssertionError(
                    f"{name}: the {pair} bracket wraps "
                    f"{type(getattr(leading, 'inner', None)).__name__!r}, not "
                    f"{plan_name!r}")
            # THE OTHER DIRECTION, on the same objects: holding the family's flag
            # down must put the by-name refusal straight back. Without this the
            # bracket above could not be told from a clause that went away.
            module = _folded_family_modules()[pair]
            saved = module.CARRIES_DEPOSIT_REPAIR
            module.CARRIES_DEPOSIT_REPAIR = False
            try:
                held = plan_step(candidate.fields, candidate.pml, fuse=True,
                                 sources=tuple(candidate._sources), num_warps=1)
            finally:
                module.CARRIES_DEPOSIT_REPAIR = saved
            held_types = plan_types(held)
            if held_types.get(curl) != "FoldedPmlCurlPlan" or \
                    held_types.get(update) != "ConstitutivePlan":
                raise AssertionError(
                    f"{name}: with {pair}'s CARRIES_DEPOSIT_REPAIR held False the "
                    f"seam must fall back to the separate folded plans; got "
                    f"{curl}={held_types.get(curl)!r} "
                    f"{update}={held_types.get(update)!r}")
            wanted = next(iter(clauses.values()))
            if not any(wanted in reason
                       for reason in held.reasons.get(f"fused_pair_{pair}", ())):
                raise AssertionError(
                    f"{name}: with the flag held False the {pair} pair was not "
                    f"refused BY NAME for its in-seam source; reasons="
                    f"{held.reasons.get(f'fused_pair_{pair}')}")
            out.setdefault("carried_seams", {})[pair] = {
                "bracketed": [types.get(curl), types.get(update)],
                "with_the_flag_held_False": [held_types.get(curl),
                                             held_types.get(update)],
                "refused_by_name_when_held": True,
            }
            continue
        if types.get(curl) != plan_name or types.get(update) != "NoopPlan":
            raise AssertionError(
                f"{name}: the {pair} pair's seam carries no source and must be "
                f"routed; got {curl}={types.get(curl)!r} "
                f"{update}={types.get(update)!r}, reasons="
                f"{plan.reasons.get(f'fused_pair_{pair}')}")
        if getattr(plan.plans[update], "absorbed_by", None) is not plan.plans[curl]:
            raise AssertionError(
                f"{name}: {update} does not name {curl} as the plan that absorbed "
                f"it, so the composition would report it as array-path work")
    # The mirror fills are the driver's own passes either way: a routed pair carries
    # them INSIDE its launch and the driver still runs them on the host afterwards.
    for slot in ("fill_B", "fill_D"):
        if types.get(slot) != "MirrorGhostFillPlan":
            raise AssertionError(f"{name}: {slot}={types.get(slot)!r}")
    # The refusal this round did NOT retire, still there by name.
    if "folded dispersive D/E fusion is not implemented" not in \
            plan.reasons.get("fused_pair_dispersive_D", ()):
        raise AssertionError(
            f"{name}: the folded dispersive D/E refusal is gone; "
            f"reasons={plan.reasons.get('fused_pair_dispersive_D')}")
    return out


def install(driver_module, plan, owner):
    """Install one candidate route without changing production dispatch.

    ``MirrorGhostFillPlan`` combines the near and far passes.  It runs in the
    driver's first fill slot, after source injection; the later far-fill call is
    suppressed for this owner only.  ``zero_metal_*`` remains between those calls
    on the driver path and is not replaced.
    """
    core = ("step_B", "update_H", "step_D", "update_E", "update_P")
    near = {"fill_symmetry_bc_B": "fill_B", "fill_symmetry_bc_D": "fill_D"}
    far = {"fill_folded_far_ghosts_B": "fill_B",
           "fill_folded_far_ghosts_D": "fill_D"}
    names = core + tuple(near) + tuple(far)
    originals = {name: getattr(driver_module, name) for name in names}

    def core_wrapper(name):
        def wrapper(fields, pml=None):
            entry = plan.plans.get(name) if fields is owner else None
            if entry is None:
                return originals[name](fields, pml)
            if name == "update_P":
                for pole_plan in entry:
                    pole_plan.run(fields.drive_field)
            else:
                entry.run()
            return None
        return wrapper

    def near_wrapper(name, slot):
        def wrapper(fields):
            entry = plan.plans.get(slot) if fields is owner else None
            if entry is None:
                return originals[name](fields)
            entry.run()
            return None
        return wrapper

    def far_wrapper(name, slot):
        def wrapper(fields):
            if fields is owner and slot in plan.plans:
                return None
            return originals[name](fields)
        return wrapper

    for name in core:
        setattr(driver_module, name, core_wrapper(name))
    for name, slot in near.items():
        setattr(driver_module, name, near_wrapper(name, slot))
    for name, slot in far.items():
        setattr(driver_module, name, far_wrapper(name, slot))
    return lambda: [setattr(driver_module, name, function)
                    for name, function in originals.items()]


def run_case(cp, name, cell, boundaries, mirrors, declarations, steps):
    from meep_gpu import driver as driver_module  # noqa: PLC0415
    from meep_gpu.triton_kernels import plan_step  # noqa: PLC0415

    reference = build_driver(cp, cell, boundaries, mirrors, declarations, 20260811)
    candidate = build_driver(cp, cell, boundaries, mirrors, declarations, 20260811)
    control = (build_driver(cp, cell, boundaries, mirrors, (), 20260811)
               if declarations else None)
    undo = lambda: None
    try:
        # THE SIX-SLOT ROUTE IS THIS GATE'S SUBJECT, and it is the UNFUSED one:
        # separate folded curl, combined mirror fill, separate constitutive. It is
        # asked for by name (`fuse=False`) rather than obtained as a side effect of
        # a refusal. Until 2026-08-27 this call passed `fuse=True` and got the same
        # six plans because the composer blanket-refused every folded fusion; when
        # the two folded pairs were ROUTED that coincidence ended and this gate
        # started failing on `selected` — the composition it certifies had not
        # moved, only the flag it was reached through. The fused answer is now a
        # measurement of its own, immediately below.
        plan = plan_step(
            candidate.fields, candidate.pml, fuse=False,
            sources=tuple(candidate._sources), num_warps=1)
        selected = plan_types(plan)
        if selected != EXPECTED:
            raise AssertionError(
                f"{name}: selected={selected}, expected={EXPECTED}, "
                f"refusals={plan.reasons}")
        routing = check_routing(plan_step, candidate, name, declarations)
        undo = install(driver_module, plan, candidate.fields)
        row: Dict[str, Any] = {
            "case": name,
            "shape": [int(value) for value in candidate.shape],
            "stored_cells": [candidate.grid.stored_cells(axis) for axis in range(3)],
            "owned_cells": [candidate.grid.owned_cells(axis) for axis in range(3)],
            "boundaries": list(boundaries),
            "mirrors": [list(spec) for spec in mirrors],
            "sources": [declaration["component"] for declaration in declarations],
            "steps": steps,
            "selected": selected,
            "refusals": {key: list(value) for key, value in plan.reasons.items()},
            "routing": routing,
            "per_step": [],
        }
        source_effect_seen = not declarations
        started = time.time()
        for step in range(1, steps + 1):
            reference.step()
            candidate.step()
            if control is not None:
                control.step()
            cp.cuda.runtime.deviceSynchronize()
            ref_state = state(reference)
            got_state = state(candidate)
            if set(ref_state) != set(got_state):
                raise AssertionError(
                    f"{name} step {step}: state inventory differs: "
                    f"reference={sorted(ref_state)}, candidate={sorted(got_state)}")
            parts = {key: compare(cp, got_state[key], ref_state[key])
                     for key in sorted(ref_state)}
            source_state_same = all(
                getattr(left, "_applied_dipole", 0j)
                == getattr(right, "_applied_dipole", 0j)
                for left, right in zip(candidate._sources, reference._sources))
            if control is not None:
                source_effect_seen = source_effect_seen or any(
                    not compare(cp, ref_state[key],
                                getattr(control.fields, key))["bit_identical"]
                    for key in PRIMARY_NAMES)
            point = {
                "step": step,
                "bit_identical": (
                    all(part["bit_identical"] for part in parts.values())
                    and source_state_same),
                "differing_floats": sum(part["differing_floats"]
                                          for part in parts.values()),
                "total_floats": sum(part["total_floats"] for part in parts.values()),
                "differing_arrays": sorted(
                    key for key, part in parts.items() if not part["bit_identical"]),
                "source_state_identical": source_state_same,
                "source_effect_seen": source_effect_seen,
            }
            row["per_step"].append(point)
            log(f"case {name} step {step}/{steps} "
                f"identical={point['bit_identical']} source_effect={source_effect_seen} "
                f"ndiff={point['differing_floats']} ({time.time() - started:.1f} s)")
            if not point["bit_identical"]:
                raise AssertionError(f"{name} diverged at step {step}: {point}")
        if not source_effect_seen:
            raise AssertionError(
                f"{name}: the source never separated the reference from its control")
        row["bit_identical"] = True
        row["source_effect_seen"] = source_effect_seen
        return row
    finally:
        undo()
        reference.close()
        candidate.close()
        if control is not None:
            control.close()
        cp.get_default_memory_pool().free_all_blocks()


def environment(cp) -> Dict[str, Any]:
    import triton  # noqa: PLC0415

    properties = cp.cuda.runtime.getDeviceProperties(0)
    return {
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "cupy": cp.__version__,
        "triton": triton.__version__,
        "device": properties["name"].decode(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    import cupy as cp  # noqa: PLC0415
    from meep_gpu import backends  # noqa: PLC0415

    backends.guard_kernel_compilation(cp)
    payload: Dict[str, Any] = {"environment": environment(cp), "cases": []}
    save(payload, args.out)
    for case in CASES:
        log(f"starting {case[0]}: mirrors={case[3]} sources="
            f"{[item['component'] for item in case[4]]}")
        payload["cases"].append(run_case(cp, *case))
        save(payload, args.out)
    total_steps = sum(row["steps"] for row in payload["cases"])
    payload["summary"] = {
        "status": "passed",
        "cases_exact": f"{len(payload['cases'])}/{len(CASES)}",
        "complete_steps_exact": f"{total_steps}/{total_steps}",
        "source_cases_nonvacuous": "2/2",
        "fold_terminations": "periodic and metallic",
        "mirror_phases": "+1 and -1",
        "fold_count_parity": "even and odd",
        "multi_axis_fold": "1/1",
    }
    save(payload, args.out)
    log(f"SYMMETRY COMPOSITION GATE PASSED: {payload['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
