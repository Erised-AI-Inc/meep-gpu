#!/usr/bin/env python3
"""The device gate for the two no-absorber stored-E Metal welds — the last buildable cells.

WHAT THIS CERTIFIES, AND ON WHAT. Two products landed in one round, both welds
across the D->E seam and both the FIRST Metal consumers of
``deposit_repair.PLAIN_PATH`` (the repair that inverts ``update_E``'s plain
overwrite rather than the split-field recurrence):

    no_pml_fused_electric_pair             no-PML curl x no-PML stored E          (1 corpus row)
    no_pml_conductive_fused_electric_pair  conductive no-PML curl x stored E      (2 corpus rows)

THE COMPARISON IS PER COMPLETE DRIVER STEP, AS uint32 WORDS, over every stored
volume a step can touch plus every polarization buffer — never ``allclose`` and
never a single sub-step. ``update_P`` runs downstream of the repaired E on this
branch (``Fields.drive_field`` returns the STORED E when the layer is inactive), so
a repair that got E right and P wrong would read as byte-identical without the pole
buffers in the comparison.

THE LEGS, each armed (a negative leg that does not diverge is reported and fails):

    identity          the full mechanism per case: 0 differing words at every step,
                      the declared launch count (3 per run), every compared array
                      moved;
    separate_control  the two ALREADY CERTIFIED halves stepping the same seam as
                      separate dispatches with the wall clear on the host between
                      them, three-way byte agreement against the array path and the
                      fused plan, with both sides' dispatch counts recorded;
    mutation          one arithmetic or index line of the shipped source replaced,
                      compiled, and handed through the plan builder's ``functions``
                      seam; each MUST diverge. The declared equivalence — the seam
                      register replaced by a reload of the word the tail just
                      stored — must NOT;
    deposit           the in-seam electric source, bracketed and unbracketed. The
                      bracketed walk (which asserts the composed bracket carries
                      repair_paths == PLAIN_PATH) must be bit-identical with a
                      nonzero repaired-point count; the unbracketed one — the
                      shipped plan built for an empty source list, injected anyway —
                      MUST diverge;
    lifted_refusal    (conductive family) the configuration the clause USED to
                      refuse — a NON-integrated, table-publishing electric source
                      on a conductive run — walked with the bracket on under the
                      +-0-lattice value class against the driver's LIVE sparse
                      injection: it must now be BYTE-IDENTICAL at every step, the
                      table-less source must still be refused by name, and the
                      RETIRED whole-volume rescale replayed on the actual side
                      MUST still diverge (the armed control that proves the walk
                      sees the hazard class the clause priced);
    binding_ceiling   the shipped per-component signatures COMPILE; each family's
                      own signature padded one binding past MAX_BUFFER_BINDINGS is
                      REFUSED; the counts are parsed from the emitted text and must
                      equal each module's BINDINGS_PER_COMPONENT.

Rule 7: one flushed line per case. Runs locally on MPS. Nothing here is dispatch:
``meep_gpu.fastpath.plan_fast_path`` still returns ``None`` on every branch.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")
HERE = Path(__file__).resolve().parent
API_ROOT = next(parent for parent in HERE.parents
                if (parent / "meep_gpu" / "metal_kernels").is_dir())
for _path in (str(API_ROOT), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402

import metal_composition_matrix as matrix  # noqa: E402

from meep_gpu import deposit_repair, stepping  # noqa: E402
from meep_gpu.metal_kernels import (  # noqa: E402
    launch as metal_launch,
    no_pml_conductive_fused_electric_pair as conductive_family,
    no_pml_fused_electric_pair as plain_family,
    shaders, subnormal,
)
from meep_gpu.metal_kernels.device import (  # noqa: E402
    MAX_BUFFER_BINDINGS, Residency, compile_source, metal_frontend_version,
)

STEPS = 12
SEED = 20260831

ARRAY_PATH: Dict[str, Callable[[Any, Any], None]] = {
    "step_B": lambda f, p: stepping.step_B(f, p),
    "update_H": lambda f, p: stepping.update_H(f, p),
    "step_D": lambda f, p: stepping.step_D(f, p),
    "update_E": lambda f, p: stepping.update_E(f, p),
    "update_P": lambda f, p: stepping.update_P(f, p),
    "fill_B": lambda f, p: stepping.fill_symmetry_bc_B(f),
    "fill_D": lambda f, p: stepping.fill_symmetry_bc_D(f),
    "zero_metal_B": lambda f, p: stepping.zero_metal_B(f),
    "zero_metal_D": lambda f, p: stepping.zero_metal_D(f),
    "fill_folded_far_ghosts_B": lambda f, p: stepping.fill_folded_far_ghosts_B(f),
    "fill_folded_far_ghosts_D": lambda f, p: stepping.fill_folded_far_ghosts_D(f),
}

STATE_NAMES: Tuple[str, ...] = tuple(
    [f"{stem}{axis}" for stem in ("B", "D", "E", "H") for axis in "xyz"]
    + [f"fu_{stem}{axis}" for stem in ("B", "D") for axis in "xyz"]
    + [f"f_w_{stem}{axis}" for stem in ("E", "H") for axis in "xyz"])


def log(message: str) -> None:
    print(message, flush=True)


def words(array: Any) -> np.ndarray:
    return np.frombuffer(np.ascontiguousarray(array).tobytes(), dtype=np.uint32)


def differing(left: Any, right: Any) -> int:
    a, b = words(left), words(right)
    if a.shape != b.shape:
        return max(a.size, b.size)
    return int(np.count_nonzero(a != b))


def state_of(fields: Any) -> Dict[str, Any]:
    """Every stored volume PLUS every polarization buffer; see the module docstring."""
    out = {name: getattr(fields, name) for name in STATE_NAMES
           if getattr(fields, name, None) is not None}
    for index, state in enumerate(tuple(getattr(fields, "polarizations", ()) or ())):
        for attribute in ("P", "P_prev"):
            for component, array in (getattr(state, attribute, {}) or {}).items():
                out[f"pol{index}.{attribute}.{component}"] = array
    return out


def frozen(fields: Any) -> Dict[str, np.ndarray]:
    return {name: np.array(value, copy=True)
            for name, value in state_of(fields).items()}


def compare(left: Any, right: Any) -> Dict[str, int]:
    a, b = state_of(left), state_of(right)
    assert set(a) == set(b), sorted(set(a) ^ set(b))
    return {name: n for name in sorted(a) if (n := differing(a[name], b[name]))}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def build_fixture(boundaries: Optional[Dict[str, str]] = None,
                  poles: Mapping[str, float] = (),
                  conductive: Sequence[str] = ()) -> Tuple[Any, Any]:
    """A no-absorber stored-E engine: the two families' own configuration space.

    ``poles`` maps a component set string like ``"ExEyEz"`` is avoided; each entry
    is one PolarizationState driving the named components. ``conductive`` names the
    D components carrying a sigma, so a one-sided conductivity is a case rather
    than a hypothesis.
    """
    fields, pml = matrix.cart(pml=0, storage=False, eps=True,
                              **({"boundaries": dict(boundaries)}
                                 if boundaries else {}))
    fields.enable_field_storage()
    for components in poles:
        driven = tuple(re.findall(r"[EH][xyz]", components))
        fields.polarizations.append(matrix._polarization(fields, driven))
    if conductive:
        shape = tuple(fields.grid.shape)
        fields.set_d_conductivity(
            {name: np.full(shape, np.float32(0.3)) for name in conductive})
    return fields, pml


def seed_state(fields: Any, seed: int, lattice: bool = False) -> Any:
    """Physical-band values everywhere, or the +-0 lattice; inv_eps made per-cell.

    THE INVERSE EPSILON IS MADE NON-UNIFORM, identically on both sides: on a
    uniform-epsilon fixture every inv_eps pointer holds one value, so a kernel that
    read the permittivity at the wrong cell — or a vacuum fixture where every
    inv_eps is an allocation of ones — is bit-identical to the right one, and a
    mutation arming exactly that defect comes back uncaught for a reason that has
    nothing to do with the kernel.
    """
    rng = np.random.default_rng(seed)
    for name in STATE_NAMES:
        array = getattr(fields, name, None)
        if array is None:
            continue
        if lattice:
            array[...] = np.where(rng.random(array.shape) < 0.5,
                                  np.float32(0.0), np.float32(-0.0))
        else:
            array[...] = rng.normal(0.0, 0.37, array.shape).astype(np.float32)
    rng_eps = np.random.default_rng(seed ^ 0x5EED)
    for component in ("Ex", "Ey", "Ez"):
        volume = fields.inverse_epsilon_for(component)
        if volume is not None and getattr(volume, "flags", None) is not None \
                and volume.flags.writeable:
            volume[...] = (0.2 + 0.6 * rng_eps.random(volume.shape)).astype(
                volume.dtype)
    for state in tuple(getattr(fields, "polarizations", ()) or ()):
        for label in ("P", "P_prev"):
            for component, array in (getattr(state, label, {}) or {}).items():
                if lattice:
                    array[...] = np.where(rng.random(array.shape) < 0.5,
                                          np.float32(0.0), np.float32(-0.0))
                else:
                    array[...] = rng.normal(0.0, 0.2, array.shape).astype(array.dtype)
        state._scratch[...] = 0.0
    return fields


def live_passes(fields: Any, pml: Any) -> Tuple[str, ...]:
    live = metal_launch.live_sub_steps(fields, pml, ())
    assert live is not None, (
        "the live pass set is unreadable; the walk would silently step a subset")
    return tuple(live)


# ---------------------------------------------------------------------------
# The families
# ---------------------------------------------------------------------------

def needle(source: str, old: str, new: str, count: int = 1) -> str:
    hits = source.count(old)
    if hits != count:
        raise AssertionError(
            f"the mutation needle {old!r} matches {hits} times, expected {count}; "
            f"an unarmed mutation reports its defect as uncaught")
    return source.replace(old, new)


#: Everything the legs need, per family. ``cases`` carry (poles, conductive)
#: fixtures spanning: no poles / one pole / the corpus row's own pole ladder; no
#: wall / one wall / all walls; and for the conductive family a one-sided sigma.
#: ``mutations`` are (label, axis, old, new) needles on the per-component source.
FAMILIES: Tuple[Dict[str, Any], ...] = (
    {
        "name": "no_pml_fused_electric_pair",
        "module": plain_family,
        "label": "no-PML fused electric D/E pair",
        "plan": plain_family.plan_metal_no_pml_fused_electric_pair,
        "coverage": plain_family.no_pml_fused_electric_pair_coverage,
        "source": lambda plan, mode, axis, poles: (
            plain_family.no_pml_fused_electric_pair_source(
                plan.codes, axis, poles, plan.zero_metal, mode)),
        "entry": "no_pml_fused_electric_pair_component",
        "cases": (
            ("all_periodic_1pole", dict(poles=("Ez",))),
            # The corpus row's own shape: material-dispersion.py is periodic on all
            # three axes with TWO poles driving every component.
            ("corpus_periodic_2poles_all", dict(poles=("ExEyEz", "ExEyEz"))),
            ("wall_z_2poles_all", dict(boundaries={"z": "metallic"},
                                       poles=("ExEyEz", "ExEyEz"))),
            ("wall_xyz_0poles", dict(boundaries={"x": "metallic", "y": "metallic",
                                                 "z": "metallic"})),
            ("wall_xyz_2poles_all", dict(boundaries={"x": "metallic",
                                                     "y": "metallic",
                                                     "z": "metallic"},
                                         poles=("ExEyEz", "ExEyEz"))),
        ),
        "mutation_case": "wall_xyz_2poles_all",
        "deposit_cases": (("corpus_periodic_2poles_all", False),
                          ("wall_z_2poles_all", True)),
        "null_case": "corpus_periodic_2poles_all",
        "conductive": False,
        "mutations": (
            ("seam_takes_the_curl", 2,
             "float source = value2;", "float source = curl2;"),
            ("seam_takes_the_curl_on_the_second_component", 1,
             "float source = value1;", "float source = curl1;"),
            ("constitutive_drops_the_inverse_epsilon", 0,
             "e_out[idx] = source * inv_e[idx];", "e_out[idx] = source;"),
            ("pole_subtraction_dropped", 2,
             "    source = source - p0[idx];\n", ""),
            ("pole_chain_reordered", 2,
             "    source = source - p0[idx];\n    source = source - p1[idx];\n",
             "    source = source - p1[idx];\n    source = source - p0[idx];\n"),
            ("zero_metal_dropped", 0,
             "    value0 = at_y ? 0.0f : value0;\n", ""),
            ("zero_metal_uses_the_b_side_table", 1,
             "    value1 = at_x ? 0.0f : value1;",
             "    value1 = at_y ? 0.0f : value1;"),
            ("curl_direction_reversed", 0,
             "int si = i - 1, sj = j - 1, sk = k - 1;",
             "int si = i + 1, sj = j + 1, sk = k + 1;"),
            ("curl_parens_flattened", 0,
             "dtdx * ((c_y - c) + (b - b_z))", "dtdx * (c_y - c + b - b_z)"),
            ("flux_store_dropped", 0,
             "    f0[ii] = value0;\n", ""),
        ),
        #: DECLARED EQUIVALENCES, run and REQUIRED to be bit-identical — each is a
        #: rewrite the fused body makes harmless BY CONSTRUCTION, measured rather
        #: than believed. The ownership-mask row is the finding of this gate's
        #: first debug cut: on THIS seam the backward mask's rows for a component
        #: (metallic y/z for curl0, x/z for curl1, x/y for curl2) are EXACTLY the
        #: wall-clear rows the kernel carries for the same component (the D
        #: off-diagonal), and the clear lands on the register AFTER the mask —
        #: so a dropped mask row is rewritten to the same zero before either
        #: consumer reads it, on every grid this family admits (a fold, where the
        #: two sets could differ, is refused by name). Measured: 0 differing words
        #: over the full budget on the all-walls case. It is scored as an
        #: equivalence so a future change that made it OBSERVABLE — say the clear
        #: moving off the register — fails this leg instead of silently widening.
        "equivalences": (
            ("seam_reloads_the_flux_it_just_stored", 0,
             "float source = value0;", "float source = f0[ii];"),
            ("ownership_mask_shadowed_by_the_carried_wall_clear", 0,
             "    curl0 = at_y ? 0.0f : curl0;\n", ""),
        ),
        "refuted": (("over_ceiling", plain_family.refuted_over_ceiling_source),),
    },
    {
        "name": "no_pml_conductive_fused_electric_pair",
        "module": conductive_family,
        "label": "conductive no-PML fused electric D/E pair",
        "plan": conductive_family.plan_metal_no_pml_conductive_fused_electric_pair,
        "coverage":
            conductive_family.no_pml_conductive_fused_electric_pair_coverage,
        "source": lambda plan, mode, axis, poles: (
            conductive_family.no_pml_conductive_fused_electric_pair_source(
                plan.codes, axis, poles,
                plan.entries[axis].conductive, plan.zero_metal, mode)),
        "entry": "no_pml_conductive_fused_electric_pair_component",
        "cases": (
            ("all_periodic_1pole", dict(poles=("Ez",),
                                        conductive=("Dx", "Dy", "Dz"))),
            # The corpus rows' own shape: absorber-1d.py / TestAbsorber are
            # metallic on z with FIVE poles driving every component (Al) and a
            # conductivity on every D target (the Absorber).
            ("corpus_wall_z_5poles_all",
             dict(boundaries={"z": "metallic"},
                  poles=("ExEyEz",) * 5, conductive=("Dx", "Dy", "Dz"))),
            ("one_sided_sigma_2poles",
             dict(poles=("ExEyEz", "ExEyEz"), conductive=("Dx",))),
            ("wall_xyz_2poles_all",
             dict(boundaries={"x": "metallic", "y": "metallic", "z": "metallic"},
                  poles=("ExEyEz", "ExEyEz"), conductive=("Dx", "Dy", "Dz"))),
        ),
        "mutation_case": "wall_xyz_2poles_all",
        # THE SCALED (non-integrated) CASE IS COMPOSED, NOT FORCE-BRACKETED, since
        # the lift: the shipped predicate now admits the corpus shape and the
        # composer installs the bracket itself, with the driver's sparse condinv
        # replay in the seam. The lifted_refusal leg holds the +-0-lattice class;
        # this row holds the physical band.
        "deposit_cases": (("corpus_wall_z_5poles_all", True),
                          ("corpus_wall_z_5poles_all", False),
                          ("all_periodic_1pole", True)),
        "null_case": "corpus_wall_z_5poles_all",
        "conductive": True,
        "priced_case": "corpus_wall_z_5poles_all",
        "mutations": (
            ("seam_takes_the_curl", 2,
             "float source = value2;", "float source = curl2;"),
            ("constitutive_drops_the_inverse_epsilon", 0,
             "e_out[idx] = source * inv_e[idx];", "e_out[idx] = source;"),
            ("pole_subtraction_dropped", 2,
             "    source = source - p0[idx];\n", ""),
            ("conductive_tail_drops_the_condfac", 0,
             "    value0 = value0 * cf0[ii];\n", ""),
            ("conductive_tail_swaps_condfac_and_condinv", 1,
             "    value1 = value1 * cf1[ii];", "    value1 = value1 * ci1[ii];"),
            ("zero_metal_dropped", 0,
             "    value0 = at_y ? 0.0f : value0;\n", ""),
            ("curl_direction_reversed", 0,
             "int si = i - 1, sj = j - 1, sk = k - 1;",
             "int si = i + 1, sj = j + 1, sk = k + 1;"),
            ("flux_store_dropped", 0,
             "    f0[ii] = value0;\n", ""),
        ),
        #: The same two measured equivalences as the plain family; see its comment
        #: for the ownership-mask shadowing argument.
        "equivalences": (
            ("seam_reloads_the_flux_it_just_stored", 0,
             "float source = value0;", "float source = f0[ii];"),
            ("ownership_mask_shadowed_by_the_carried_wall_clear", 0,
             "    curl0 = at_y ? 0.0f : curl0;\n", ""),
        ),
        "refuted": (("over_ceiling",
                     conductive_family.refuted_over_ceiling_source),),
    },
)


def build_case(spec: Mapping[str, Any], name: str) -> Tuple[Any, Any]:
    keywords = dict(dict(spec["cases"])[name])
    return build_fixture(**keywords)


# ---------------------------------------------------------------------------
# Leg: identity
# ---------------------------------------------------------------------------

def metal_step(fields: Any, pml: Any, plan: Any, owned: Sequence[str],
               residency: Residency, live: Sequence[str]) -> None:
    skip = set(owned)
    for name in live:
        if name in skip:
            if name == owned[0]:
                plan.run()
            continue
        residency.sync_out()
        ARRAY_PATH[name](fields, pml)
        residency.sync_in()


def run_case(spec: Mapping[str, Any], case: str, steps: int,
             functions: Optional[Mapping[Any, Any]] = None,
             seed: int = SEED) -> Dict[str, Any]:
    reference, reference_pml = build_case(spec, case)
    actual, actual_pml = build_case(spec, case)
    seed_state(reference, seed)
    seed_state(actual, seed)
    drift = compare(reference, actual)
    assert not drift, f"the two builds are not identical: {drift}"

    residency = Residency()
    plan = spec["plan"](actual, actual_pml, sources=(), residency=residency,
                        functions=functions)
    if plan is None:
        return {"passed": False, "reason": "the pair was refused",
                "refusals": list(spec["coverage"](
                    actual, actual_pml, (), residency).reasons)[:6]}

    live = live_passes(actual, actual_pml)
    assert live == live_passes(reference, reference_pml)
    before = frozen(actual)
    residency.sync_in()

    per_step: List[Dict[str, Any]] = []
    for step in range(1, steps + 1):
        for name in live:
            ARRAY_PATH[name](reference, reference_pml)
        metal_step(actual, actual_pml, plan, plan.replaces_sub_steps, residency,
                   live)
        residency.sync_out()
        difference = compare(reference, actual)
        census = sum(subnormal.census(value)
                     for value in state_of(reference).values())
        per_step.append({"step": step,
                         "differing_words": sum(difference.values()),
                         "differing_arrays": dict(sorted(difference.items())),
                         "reference_subnormals": int(census)})
        if difference or census:
            break

    after = state_of(actual)
    moved = {name: differing(before[name], after[name]) for name in before}
    still = sorted(name for name, count in moved.items() if count == 0)
    identical = (len(per_step) == steps
                 and all(row["differing_words"] == 0 for row in per_step))
    clean = all(row["reference_subnormals"] == 0 for row in per_step)
    launches_ok = (plan.runs == len(per_step)
                   and plan.launches == plan.launches_per_run * len(per_step))
    return {
        "passed": bool(identical and clean and launches_ok and not still),
        "bit_identical": identical,
        "subnormal_free": clean,
        "first_divergence": next((row["step"] for row in per_step
                                  if row["differing_words"]), None),
        "steps_compared": len(per_step),
        "differing_words": per_step[-1]["differing_words"],
        "differing_arrays": per_step[-1]["differing_arrays"],
        "arrays_compared": len(before),
        "arrays_that_never_moved": still,
        "moved_words": int(sum(moved.values())),
        "runs": plan.runs, "launches": plan.launches,
        "launches_per_run": plan.launches_per_run,
        "launches_expected": plan.launches_per_run * len(per_step),
        "live_passes": list(live), "replaces": list(plan.replaces_sub_steps),
        "boundary_codes": list(plan.codes), "zero_metal": list(plan.zero_metal),
        "poles": [entry.pole_count for entry in plan.entries],
        "shape": list(plan.shape), "mirrors": len(residency.names),
    }


def leg_identity(spec: Mapping[str, Any], steps: int) -> Dict[str, Any]:
    rows: Dict[str, Any] = {}
    for name, _keywords in spec["cases"]:
        started = time.time()
        rows[name] = run_case(spec, name, steps)
        log(f"    identity  {spec['name']:40s} {name:28s} "
            f"passed={rows[name]['passed']} "
            f"words={rows[name].get('differing_words')} "
            f"launches={rows[name].get('launches')} "
            f"({time.time() - started:5.1f} s)")
    return {"passed": all(row["passed"] for row in rows.values()), "cases": rows}


# ---------------------------------------------------------------------------
# Leg: the separately certified halves, three engines from one seed
# ---------------------------------------------------------------------------

def leg_separate_control(spec: Mapping[str, Any], case: str,
                         steps: int) -> Dict[str, Any]:
    """Array path vs the two CERTIFIED separate plans vs the fused plan.

    The separate side is what ``plan_step`` composes today for this configuration —
    the curl arm this family's row names on ``step_D`` and the ``no-PML stored E``
    arm on ``update_E``, with ``zero_metal_D`` left on the host between them. All
    three engines must agree word for word at every complete step; the dispatch
    counts and the surviving in-seam host passes are the only place the difference
    shows, since a correct fusion is byte-neutral by construction.
    """
    reference, reference_pml = build_case(spec, case)
    separate, separate_pml = build_case(spec, case)
    fused, fused_pml = build_case(spec, case)
    for engine in (reference, separate, fused):
        seed_state(engine, SEED)

    separate_residency = Residency()
    composed = metal_launch.plan_step(separate, separate_pml,
                                      residency=separate_residency, sources=(),
                                      fuse=False)
    curl = composed.plans.get("step_D")
    electric = composed.plans.get("update_E")
    fused_residency = Residency()
    plan = spec["plan"](fused, fused_pml, sources=(), residency=fused_residency)
    if curl is None or electric is None or plan is None:
        return {"passed": False, "reason": "a certified half or the pair refused",
                "selected": dict(composed.selected)}

    live = live_passes(fused, fused_pml)
    separate_residency.sync_in()
    fused_residency.sync_in()
    per_step: List[Dict[str, Any]] = []
    for step in range(1, steps + 1):
        for name in live:
            ARRAY_PATH[name](reference, reference_pml)
        # the SEPARATE composition: certified curl, host wall clear, certified E.
        for name in live:
            if name == "step_D":
                curl.run()
            elif name == "update_E":
                electric.run()
            else:
                separate_residency.sync_out()
                ARRAY_PATH[name](separate, separate_pml)
                separate_residency.sync_in()
        metal_step(fused, fused_pml, plan, plan.replaces_sub_steps,
                   fused_residency, live)
        separate_residency.sync_out()
        fused_residency.sync_out()
        per_step.append({
            "step": step,
            "separate_vs_array": sum(compare(reference, separate).values()),
            "fused_vs_array": sum(compare(reference, fused).values()),
            "fused_vs_separate": sum(compare(separate, fused).values()),
        })
        if any(value for key, value in per_step[-1].items() if key != "step"):
            break
    identical = (len(per_step) == steps
                 and all(row["separate_vs_array"] == row["fused_vs_array"]
                         == row["fused_vs_separate"] == 0 for row in per_step))
    seam_hosts = [name for name in live if name == "zero_metal_D"]
    row = {
        "passed": bool(identical and plan.launches and curl.launches
                       and electric.launches),
        "case": case,
        "per_step": per_step,
        "steps_compared": len(per_step),
        "separate_selected": {slot: composed.selected.get(slot)
                              for slot in ("step_D", "update_E")},
        "separate_dispatches_per_step": (curl.launches_per_run
                                         + electric.launches_per_run),
        "fused_dispatches_per_step": plan.launches_per_run,
        "seam_host_passes_for_separate": seam_hosts,
        "seam_host_passes_for_fused": [],
    }
    log(f"    separate  {spec['name']:40s} {case:28s} passed={row['passed']} "
        f"separate={row['separate_dispatches_per_step']}/step "
        f"fused={row['fused_dispatches_per_step']}/step")
    return row


# ---------------------------------------------------------------------------
# Leg: mutation (and the declared equivalence)
# ---------------------------------------------------------------------------

def _mutated_functions(spec: Mapping[str, Any], case: str, axis: int, old: str,
                       new: str) -> Dict[Any, Any]:
    """The shipped source with ONE line replaced, keyed (mode, axis) as the plan keys.

    Only the mutated component's key is handed back; the plan builder compiles the
    other two itself, which is what makes the leg measure ONE line rather than one
    component.
    """
    fields, pml = build_case(spec, case)
    seed_state(fields, SEED)
    residency = Residency()
    probe = spec["plan"](fields, pml, sources=(), residency=residency)
    assert probe is not None, "the mutation case is refused; nothing can be armed"
    mode = shaders.CONTRACT_OFF
    poles = probe.entries[axis].pole_count
    source = needle(spec["source"](probe, mode, axis, poles), old, new)
    function = getattr(compile_source(source), spec["entry"])
    return {(mode, axis): function}


def leg_mutation(spec: Mapping[str, Any], steps: int) -> Dict[str, Any]:
    case = spec["mutation_case"]
    rows: Dict[str, Any] = {}
    for label, axis, old, new in spec["mutations"]:
        functions = _mutated_functions(spec, case, axis, old, new)
        result = run_case(spec, case, steps, functions=functions)
        caught = not result.get("bit_identical", True)
        rows[label] = {"passed": bool(caught and result.get("launches")),
                       "caught": caught,
                       "first_divergence": result.get("first_divergence"),
                       "differing_words": result.get("differing_words"),
                       "launches": result.get("launches")}
        log(f"    mutation  {spec['name']:40s} {label:42s} caught={caught} "
            f"words={result.get('differing_words')}")
    equivalences: Dict[str, Any] = {}
    for label, axis, old, new in spec.get("equivalences", ()):
        functions = _mutated_functions(spec, case, axis, old, new)
        result = run_case(spec, case, steps, functions=functions)
        identical = bool(result.get("bit_identical"))
        equivalences[label] = {"passed": identical, "bit_identical": identical,
                               "differing_words": result.get("differing_words"),
                               "launches": result.get("launches")}
        log(f"    equivalent{spec['name']:40s} {label:42s} identical={identical}")
    return {"passed": (all(row["passed"] for row in rows.values())
                       and all(row["passed"] for row in equivalences.values())),
            "mutations": rows, "declared_equivalences": equivalences}


# ---------------------------------------------------------------------------
# Leg: the deposit — bracketed, unbracketed, and the driver's own injection
# ---------------------------------------------------------------------------

def _volume_source(fields: Any, component: str, integrated: bool) -> Any:
    from meep_gpu.sources import ContinuousEnvelope, VolumeSource  # noqa: PLC0415

    return VolumeSource(grid=fields.grid, component=component,
                        center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                        envelope=ContinuousEnvelope(frequency=1.0,
                                                    is_integrated=integrated))


def _withdraw(sources: Sequence[Any], fields: Any) -> None:
    for source in sources:
        hook = getattr(source, "withdraw", None)
        if callable(hook):
            hook(fields)


def _inject_like_the_driver(fields: Any, sources: Sequence[Any],
                            when: float) -> None:
    """``FdtdDriver``'s own electric injection, transcribed from the LIVE driver:
    integrated point-wise; scaled by a SPARSE per-deposit-cell condinv replay at
    the indices the sources publish, with the whole-volume difference passes kept
    only as the fallback for a source publishing no deposit table
    (``_inject_electric_through_conductivity``). The gate's ``lifted_refusal`` leg
    replays the RETIRED whole-volume form through
    :func:`_inject_with_whole_volume_rescale` as its armed control."""
    electric = [s for s in sources if s.field_type != "B"]
    if electric and fields.has_conductivity:
        integrated = [s for s in electric if s.is_integrated]
        scaled = [s for s in electric if not s.is_integrated]
        for source in integrated:
            source.inject(fields, when)
        if not scaled:
            return
        xp = fields.grid.xp
        by_target: Dict[str, list] = {}
        for source in scaled:
            by_target.setdefault("D" + source.component[1], []).append(source)
        sparse: Dict[str, Any] = {}
        dense: Dict[str, Any] = {}
        for name, sources_for in sorted(by_target.items()):
            if fields.condinv_for(name) is None:
                continue
            array = getattr(fields, name)
            columns = []
            for source in sources_for:
                index = deposit_repair._deposit_index(source)
                if index is not None:
                    columns.append(index)
                elif not hasattr(source, "_point_ix"):
                    dense[name] = array.copy()
                    break
            else:
                if not columns:
                    continue
                if len(columns) == 1:
                    ix, iy, iz = columns[0]
                else:
                    ix = xp.concatenate([column[0] for column in columns])
                    iy = xp.concatenate([column[1] for column in columns])
                    iz = xp.concatenate([column[2] for column in columns])
                sparse[name] = (ix, iy, iz, array[ix, iy, iz])
        for source in scaled:
            source.inject(fields, when)
        for name, (ix, iy, iz, before) in sparse.items():
            array = getattr(fields, name)
            values = array[ix, iy, iz]
            values -= before
            values *= fields.condinv_for(name)[ix, iy, iz]
            values += before
            array[ix, iy, iz] = values
        for name, before in dense.items():
            array = getattr(fields, name)
            array -= before
            array *= fields.condinv_for(name)
            array += before
    else:
        for source in electric:
            source.inject(fields, when)


def _inject_with_whole_volume_rescale(fields: Any, sources: Sequence[Any],
                                      when: float) -> None:
    """The RETIRED whole-volume driver pass, kept as the lifted leg's armed control.

    This is the transcription the driver shipped until the sparse replay landed —
    ``before = D.copy(); inject; D -= before; D *= condinv; D += before`` — and it
    is what canonicalises every ``-0.0`` in the target component. The
    ``lifted_refusal`` leg walks it against the array path stepped with the LIVE
    injection and REQUIRES the divergence, which is what proves the leg still
    measures the hazard the clause used to price.
    """
    electric = [s for s in sources if s.field_type != "B"]
    if electric and fields.has_conductivity:
        integrated = [s for s in electric if s.is_integrated]
        scaled = [s for s in electric if not s.is_integrated]
        for source in integrated:
            source.inject(fields, when)
        if scaled:
            targets = sorted({"D" + source.component[1] for source in scaled})
            before = {name: getattr(fields, name).copy() for name in targets}
            for source in scaled:
                source.inject(fields, when)
            for name in targets:
                array = getattr(fields, name)
                condinv = fields.condinv_for(name)
                if condinv is None:
                    continue
                array -= before[name]
                array *= condinv
                array += before[name]
    else:
        for source in electric:
            source.inject(fields, when)


def run_deposit_case(spec: Mapping[str, Any], case: str, integrated: bool,
                     steps: int, repair: bool = True,
                     lattice: bool = False,
                     force_bracket: bool = False,
                     actual_injector: Optional[Callable[..., None]] = None,
                     ) -> Dict[str, Any]:
    """Complete driver steps with an electric source IN the seam, byte compared.

    The walk is the driver's: withdraw, curl consult, injection (through the
    driver's own conductive-or-plain branch), host symmetry fill and wall clear
    (unconditional, even though the kernel carries the clear inline — that is what
    lets the repair read a final field), constitutive consult, ``update_P``.

    ``repair=False`` builds the shipped plan for an EMPTY source list — the bare
    pair plus the ``NoopPlan`` — and injects anyway: the no-mechanism control.
    ``force_bracket=True`` builds the bare pair sourceless and wraps it in the two
    repair plans by hand (the route the priced leg used while the predicate still
    refused the configuration; the lifted leg keeps it so the walk is the recorded
    one). ``actual_injector`` swaps the ACTUAL side's injection — the lifted leg's
    armed control hands it the RETIRED whole-volume rescale while the reference
    side keeps the live driver's, and that asymmetry is the divergence the control
    requires.
    """
    reference, reference_pml = build_case(spec, case)
    actual, actual_pml = build_case(spec, case)
    seed_state(reference, SEED, lattice=lattice)
    seed_state(actual, SEED, lattice=lattice)
    drift = compare(reference, actual)
    assert not drift, f"the two builds are not identical: {drift}"

    reference_sources = (_volume_source(reference, "Ez", integrated),)
    actual_sources = (_volume_source(actual, "Ez", integrated),)
    in_seam = deposit_repair.in_seam_sources(actual_sources, "D")
    assert len(in_seam) == 1, "the source is not in the D seam; the walk is vacuous"
    assert deposit_repair._deposit_index(actual_sources[0]) is not None

    residency = Residency()
    if force_bracket:
        bare = spec["plan"](actual, actual_pml, sources=(), residency=residency)
        if bare is None:
            return {"passed": False, "reason": "even the sourceless pair refused"}
        leading = deposit_repair.LeadingRepairPlan(
            bare, actual, actual_pml, actual_sources, "D",
            spec["module"].REPAIR_PATHS)
        trailing = deposit_repair.TrailingRepairPlan(
            "update_E", leading, actual, actual_pml)
        installed = ("LeadingRepairPlan", "TrailingRepairPlan")
        installed_ok = True
        selected = {"step_D": spec["label"], "update_E": spec["label"]}
    else:
        composed = metal_launch.plan_step(
            actual, actual_pml, residency=residency,
            sources=(actual_sources if repair else ()), fuse=True)
        leading = composed.plans.get("step_D")
        trailing = composed.plans.get("update_E")
        if leading is None or trailing is None:
            return {"passed": False, "reason": "the composer did not fuse the seam",
                    "refusals": [r for key, value in composed.reasons.items()
                                 if key.startswith("fused_pair")
                                 for r in value][:6]}
        installed = (type(leading).__name__, type(trailing).__name__)
        if repair:
            installed_ok = installed == ("LeadingRepairPlan", "TrailingRepairPlan")
            # THE PATH ASSERTION: the composed bracket must carry this product's
            # own declaration, not the split-field default.
            installed_ok = installed_ok and (
                tuple(leading.repair_paths) == (deposit_repair.PLAIN_PATH,))
        else:
            installed_ok = (installed[0] != "LeadingRepairPlan"
                            and installed[1] == "NoopPlan")
        if composed.selected.get("step_D") != spec["label"]:
            return {"passed": False,
                    "reason": f"another product won step_D: "
                              f"{composed.selected.get('step_D')!r}"}
        selected = {slot: composed.selected.get(slot)
                    for slot in ("step_D", "update_E")}
    inner = getattr(leading, "absorbed_by", leading)

    live = live_passes(actual, actual_pml)
    before = frozen(actual)
    dt = float(actual.grid.dt)

    per_step: List[Dict[str, Any]] = []
    for step in range(1, steps + 1):
        when = (step - 1) * dt
        _withdraw(reference_sources, reference)
        for name in live:
            ARRAY_PATH[name](reference, reference_pml)
            if name == "step_D":
                _inject_like_the_driver(reference, reference_sources, when)
        _withdraw(actual_sources, actual)
        inject_actual = (actual_injector if actual_injector is not None
                         else _inject_like_the_driver)
        for name in live:
            if name == "step_D":
                residency.sync_in()
                leading.run()
                residency.sync_out()
                inject_actual(actual, actual_sources, when)
            elif name == "update_E":
                trailing.run()
            elif name == "zero_metal_D":
                # UNCONDITIONAL in the driver (driver.py:3301), even though the
                # kernel carries the clear inline; idempotent, and what lets the
                # repair read a final field.
                ARRAY_PATH[name](actual, actual_pml)
            elif name in set(inner.replaces_sub_steps):
                continue
            else:
                ARRAY_PATH[name](actual, actual_pml)
        difference = compare(reference, actual)
        per_step.append({"step": step,
                         "differing_words": sum(difference.values()),
                         "differing_arrays": dict(sorted(difference.items()))})
    identical = all(row["differing_words"] == 0 for row in per_step)
    after = state_of(actual)
    moved = {name: differing(before[name], after[name]) for name in before}
    still = sorted(name for name, count in moved.items() if count == 0)
    repairs = int(getattr(leading, "repairs", 0))
    launches_ok = (inner.runs == len(per_step)
                   and inner.launches == inner.launches_per_run * len(per_step))
    return {
        "passed": bool(identical and launches_ok and installed_ok
                       and (repairs > 0 if (repair or force_bracket)
                            else repairs == 0)
                       and (not still if not lattice else True)),
        "repair_wired": repair, "forced_bracket": force_bracket,
        "installed_plans": list(installed), "installed_as_expected": installed_ok,
        "repair_paths": list(getattr(leading, "repair_paths", ())),
        "deposit_points_repaired": repairs,
        "bit_identical": identical,
        "first_divergence": next((row["step"] for row in per_step
                                  if row["differing_words"]), None),
        "identical_at_the_last_step": per_step[-1]["differing_words"] == 0,
        "per_step": [{"step": row["step"],
                      "differing_words": row["differing_words"]}
                     for row in per_step],
        "differing_arrays_at_first_divergence": next(
            (row["differing_arrays"] for row in per_step
             if row["differing_words"]), {}),
        "source_integrated": integrated, "value_class": (
            "pm_zero_lattice" if lattice else "physical_band"),
        "arrays_that_never_moved": still,
        "runs": inner.runs, "launches": inner.launches,
        "launches_per_run": inner.launches_per_run,
        "selected": selected,
    }


def leg_deposit(spec: Mapping[str, Any], steps: int) -> Dict[str, Any]:
    rows: Dict[str, Any] = {}
    for case, integrated in spec["deposit_cases"]:
        label = f"{case}:{'integrated' if integrated else 'scaled'}"
        rows[label] = run_deposit_case(spec, case, integrated, steps)
        log(f"    deposit   {spec['name']:40s} {label:42s} "
            f"passed={rows[label]['passed']} "
            f"repaired={rows[label].get('deposit_points_repaired')} "
            f"paths={rows[label].get('repair_paths')}")
    control = run_deposit_case(spec, spec["null_case"],
                               integrated=spec["conductive"], steps=steps,
                               repair=False)
    diverged = not control.get("bit_identical", True)
    rows["null_control"] = {
        "passed": bool(diverged and control.get("launches")),
        "diverged": diverged,
        "installed_plans": control.get("installed_plans"),
        "first_divergence": control.get("first_divergence"),
        "differing_words_at_first_divergence": next(
            (row["differing_words"] for row in control.get("per_step", ())
             if row["differing_words"]), None),
    }
    log(f"    deposit   {spec['name']:40s} {'null_control':42s} "
        f"diverged={diverged} (must)")
    return {"passed": all(row["passed"] for row in rows.values()), "cases": rows}


def leg_lifted_refusal(spec: Mapping[str, Any], steps: int) -> Dict[str, Any]:
    """The formerly refused configuration, MEASURED IDENTICAL — with the old hazard
    still armed.

    THE CLAUSE THIS LEG USED TO STAND BEHIND priced the conductive cell at 0 of 2:
    the driver's whole-volume condinv rescale canonicalised every ``-0.0`` in the
    target D at cells no deposit closure can name, and the walk below (the same
    +-0-lattice class, the same PLAIN_PATH bracket) MEASURED the divergence — 2417
    Dz words moved, 1231 Ez words divergent at step 1, identical from step 2 on.
    The driver now replays the rescale sparsely at the deposit cells the sources
    publish, so the SAME walk is re-run here and the lift is taken on what it
    reads. Four requirements, each armed:

    * the shipped predicate NO LONGER refuses a scaled source that publishes its
      deposit table (the corpus shape) — and the two corpus rows' configuration
      is admitted end to end;
    * the shipped predicate STILL refuses, by name, a scaled source publishing NO
      table (the driver's dense fallback — the surviving hazard);
    * the recorded signed-zero walk, re-run against the LIVE driver injection, is
      BYTE-IDENTICAL at every step with a nonzero repaired-point count;
    * the RETIRED whole-volume rescale, replayed on the actual side only, MUST
      still diverge on the same walk — the control that proves this leg can see
      exactly the hazard class the clause used to price.
    """
    fields, pml = build_case(spec, spec["priced_case"])
    scaled = (_volume_source(fields, "Ez", integrated=False),)
    # A DECLARED residency for the ADMITTING direction: None is itself a refusal
    # (coverage._residency_declaration_reasons), so asking with None would report
    # the lift unlicensed for a reason that has nothing to do with the clause.
    # The refusal-direction check below keeps None — a named-substring check
    # holds either way.
    verdict = spec["coverage"](fields, pml, scaled, Residency())
    published_admitted = bool(verdict.covered)

    class _TableLess:
        """A scaled electric source that publishes no deposit table at all."""

        field_type = "D"
        component = "Ez"
        is_integrated = False

    tableless_verdict = spec["coverage"](fields, pml, (_TableLess(),), None)
    tableless_named = [reason for reason in tableless_verdict.reasons
                       if "publishes NO deposit table" in reason
                       or "no _point_ix" in reason]

    walk = run_deposit_case(spec, spec["priced_case"], integrated=False,
                            steps=steps, lattice=True, force_bracket=True)
    identical = bool(walk.get("bit_identical"))

    control = run_deposit_case(
        spec, spec["priced_case"], integrated=False, steps=steps, lattice=True,
        force_bracket=True, actual_injector=_inject_with_whole_volume_rescale)
    control_diverged = not control.get("bit_identical", True)

    row = {
        "passed": bool(published_admitted and tableless_named and identical
                       and walk.get("launches")
                       and (walk.get("deposit_points_repaired") or 0) > 0
                       and control_diverged),
        "published_deposit_source_admitted": published_admitted,
        "refusal_reasons_if_any": list(verdict.reasons)[:4],
        "tableless_source_still_refused_by_name": bool(tableless_named),
        "tableless_refusal_text": tableless_named[:1],
        "bit_identical": identical,
        "first_divergence": walk.get("first_divergence"),
        "deposit_points_repaired": walk.get("deposit_points_repaired"),
        "launches": walk.get("launches"),
        "value_class": walk.get("value_class"),
        "whole_volume_control": {
            "diverged": control_diverged,
            "first_divergence": control.get("first_divergence"),
            "differing_arrays_at_first_divergence":
                control.get("differing_arrays_at_first_divergence"),
            "identical_at_the_LAST_step": control.get("identical_at_the_last_step"),
        },
        "note": ("the sparse per-deposit-cell condinv replay leaves every "
                 "non-deposit word untouched, so the fused launch's E is computed "
                 "from the same bytes the array path reads; the retired "
                 "whole-volume passes, replayed as the control, still "
                 "canonicalise -0.0 and still diverge"
                 if identical else
                 "the walk did NOT read identical: the lift is not licensed and "
                 "the clause must stand"),
    }
    log(f"    lifted    {spec['name']:40s} identical={identical} "
        f"repaired={row['deposit_points_repaired']} "
        f"control_diverged={control_diverged} "
        f"at_step={row['whole_volume_control']['first_divergence']}")
    return row


# ---------------------------------------------------------------------------
# Leg: the binding ceiling, compiled
# ---------------------------------------------------------------------------

_BINDING = re.compile(r"\[\[buffer\((\d+)\)\]\]")


def leg_binding_ceiling() -> Dict[str, Any]:
    """Shipped signatures COMPILE; each padded one past the ceiling is REFUSED.

    The parsed binding counts must equal each module's own
    ``BINDINGS_PER_COMPONENT``, so the constant the board and the docstrings argue
    from is measured against the emitted text and against this toolchain.
    """
    rows: Dict[str, Any] = {}
    shipped = {
        "no_pml_fused_electric_pair": plain_family
        .no_pml_fused_electric_pair_source((1, 1, 1), 0, 8, (True, True, True)),
        "no_pml_conductive_fused_electric_pair": conductive_family
        .no_pml_conductive_fused_electric_pair_source(
            (1, 1, 1), 0, 8, True, (True, True, True)),
    }
    declared = {
        "no_pml_fused_electric_pair": plain_family.BINDINGS_PER_COMPONENT,
        "no_pml_conductive_fused_electric_pair":
            conductive_family.BINDINGS_PER_COMPONENT,
    }
    for name, source in shipped.items():
        slots = sorted({int(number) for number in _BINDING.findall(source)})
        count = len(slots)
        try:
            compile_source(source)
            compiled = True
            error = ""
        except Exception as exc:  # noqa: BLE001
            compiled = False
            error = str(exc).strip().splitlines()[-1][:160]
        rows[f"{name}/shipped"] = {
            "passed": bool(compiled and count == declared[name]
                           and slots == list(range(count))
                           and count <= MAX_BUFFER_BINDINGS),
            "compiled": compiled, "error": error,
            "bindings_parsed": count, "bindings_declared": declared[name],
            "ceiling": MAX_BUFFER_BINDINGS,
        }
        log(f"    ceiling   {name}/shipped bindings={count} compiled={compiled}")
    for spec in FAMILIES:
        for label, builder in spec["refuted"]:
            key = f"{spec['name']}/{label}"
            source = builder()
            slots = sorted({int(number) for number in _BINDING.findall(source)})
            try:
                compile_source(source)
                rows[key] = {"passed": False, "compiled": True,
                             "bindings_parsed": len(slots)}
            except Exception as error:  # noqa: BLE001 - the failure IS the result
                rows[key] = {
                    "passed": bool(len(slots) == MAX_BUFFER_BINDINGS + 1),
                    "compiled": False,
                    "bindings_parsed": len(slots),
                    "error": str(error).strip().splitlines()[-1][:160]}
            log(f"    ceiling   {key} bindings={len(slots)} "
                f"refused={not rows[key]['compiled']}")
    return {"passed": all(row["passed"] for row in rows.values()),
            "signatures": rows}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    global STEPS

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument("--only", default=None)
    parser.add_argument(
        "--legs",
        default="identity,separate_control,mutation,deposit,lifted_refusal,ceiling")
    parser.add_argument("--steps", type=int, default=STEPS)
    arguments = parser.parse_args()
    STEPS = int(arguments.steps)

    matrix.prepare_environment()
    import torch  # noqa: PLC0415
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available; this gate must run on an Apple GPU")

    legs = tuple(part.strip() for part in arguments.legs.split(",") if part.strip())
    started = time.time()
    record: Dict[str, Any] = {
        "gate": "metal_no_pml_fused_electric_pairs",
        "steps": arguments.steps, "seed": SEED,
        "families": {},
        "corpus_rows": {
            "census": "parity/meep_gpu/results/metal_coverage_tranche6_2026-08-19",
            "no_pml_fused_electric_pair": {
                "cell": "D_to_E (no-PML curl, no-PML stored E)",
                "seam_instances": 1,
                "rows": ["examples:material-dispersion.py"]},
            "no_pml_conductive_fused_electric_pair": {
                "cell": "D_to_E (conductive no-PML curl, no-PML stored E)",
                "seam_instances": 2,
                "rows": ["examples:absorber-1d.py",
                         "tests:TestAbsorber.test_absorber"],
                "served": 2,
                "why": "both rows declare a NON-integrated, table-publishing "
                       "electric source on a conductive run; the driver's sparse "
                       "per-deposit-cell condinv replay lifted the whole-volume "
                       "refusal on the lifted_refusal leg's measurement"},
        },
    }
    for spec in FAMILIES:
        if arguments.only and spec["name"] != arguments.only:
            continue
        log(f"  {spec['name']}")
        entry: Dict[str, Any] = {}
        if "identity" in legs:
            entry["identity"] = leg_identity(spec, STEPS)
        if "separate_control" in legs:
            entry["separate_control"] = leg_separate_control(
                spec, spec["mutation_case"], STEPS)
        if "mutation" in legs:
            entry["mutation"] = leg_mutation(spec, STEPS)
        if "deposit" in legs:
            entry["deposit"] = leg_deposit(spec, STEPS)
        if "lifted_refusal" in legs and spec.get("priced_case"):
            entry["lifted_refusal"] = leg_lifted_refusal(spec, STEPS)
        entry["passed"] = all(value.get("passed") for value in entry.values()
                              if isinstance(value, dict))
        record["families"][spec["name"]] = entry

    if "ceiling" in legs:
        record["binding_ceiling"] = leg_binding_ceiling()

    record["torch_version"] = torch.__version__
    record["numpy_version"] = np.__version__
    record["metal_frontend"] = metal_frontend_version()
    record["subnormal_policy"] = os.environ.get("MEEP_GPU_SUBNORMAL_POLICY")
    record["elapsed_s"] = round(time.time() - started, 2)
    record["verdict"] = "PASS" if all(
        value.get("passed") for value in record["families"].values()) and (
        record.get("binding_ceiling", {}).get("passed", True)) else "FAIL"
    from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
    _stamp_provenance(record)  # bytes THIS process imported; see gate_provenance
    text = json.dumps(record, indent=1, sort_keys=True)
    if arguments.out:
        path = Path(arguments.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")
        log(f"  wrote {path}")
    else:
        print(text)
    log(f"  VERDICT {record['verdict']}  ({record['elapsed_s']} s)")
    return 0 if record["verdict"] == "PASS" else 1


if __name__ == "__main__":
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))
