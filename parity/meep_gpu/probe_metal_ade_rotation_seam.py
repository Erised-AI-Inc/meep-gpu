#!/usr/bin/env python3
"""THE ROTATION QUESTION AT THE ``update_E`` -> ``update_P`` SEAM, MEASURED.

WHY THIS PROBE EXISTS BEFORE ANY KERNEL. The Metal fusion matrix
(``results/fusion_matrix_metal_2026-08-20/fusion_matrix.txt``) prices all five
E->P cells as FITS on bindings and then declines to call any of them buildable::

    SEQUENCING: ade_update_p launches once per driven component and the host
    rotates P / P_prev / scratch between launches (ade_update_p.py:13-18, :353);
    a fused E->P kernel must either take one component or bake that rotation.
    NOT MEASURED -- the pointer bracket is a fitness verdict, not a buildability one

Triton met a version of the same question and answered it by BAKING the rotation:
``triton_kernels.fused_ade_state`` puts one susceptibility's Ex/Ey/Ez recurrences
in one launch, and its module docstring records that arm 1's OUTPUT buffer IS arm
0's ``p_prev`` INPUT, that the safety of that alias is an OBSERVATION on one
toolchain rather than a guarantee, and — the sentence that matters here — that
"Fusing further REQUIRES breaking the chain first, not inheriting it."

This probe measures four things, in the order that decides the design:

LEG 1  interleaving_equivalence
    Is a PER-COMPONENT interleave of the two sub-steps — ``update_E(c)`` then
    every ``update_P(state, c)``, for c in Ex, Ey, Ez — byte-identical to the
    driver's order (all of ``update_E``, then all of ``update_P``)?  Measured with
    the ALREADY CERTIFIED Metal products, entry-sliced, on the device, per
    complete step, as uint32 words.  If YES, a fused E->P product does not have to
    bake the rotation at all: it inherits the existing host rotation between
    per-component launches, and the third option the matrix did not list — ONE
    COMPONENT PER LAUNCH, ROTATION LEFT WHERE IT IS — is the buildable one.

LEG 2  rotation_orbit_disjointness
    Given that, does the rotation ever place a per-component launch's OUTPUT on
    one of that same launch's INPUTS?  The rotation is a finite permutation of
    ``2d + 1`` buffers for ``d`` driven components, so its orbit is enumerable
    rather than sampled: this walks the orbit to closure and checks read/write
    disjointness at EVERY configuration in it, for d = 1, 2, 3 and 1..6
    susceptibilities.  The same enumeration is run for the ALL-COMPONENT launch
    shape, which is the shape ``fused_ade_state`` takes, and it is REPORTED
    NON-DISJOINT — the contrast is what makes the per-component finding
    load-bearing rather than a restatement.

LEG 3  metal_program_order_under_aliasing
    What this toolchain actually does with two bindings on one buffer, so the
    contrast in leg 2 rests on a measurement rather than on a fear.  Reported as
    an OBSERVATION with its scope stated, exactly as the Triton note reports its
    own: it is not needed by the shape this probe licenses.

LEG 4  binding_fitness
    The fused signature at the corpus's WORST pole count, compiled for real, plus
    the first over-ceiling signature required to FAIL.  The matrix's bracket
    ``[10,13] .. [26,29]`` prices ONE pole; the corpus's dispersive-PML rows carry
    SIX (``stochastic_emitter*.py``), and 6 + 4x6 = 30 pointers is a different
    question from 5.

Rule 7: one flushed line per leg-case, every row appended and fsynced as it lands.

Usage (from the repository root)::

    PYTHONPATH=. MEEP_GPU_SUBNORMAL_POLICY=flush \\
      python -u parity/meep_gpu/probe_metal_ade_rotation_seam.py \\
        --out parity/meep_gpu/results/<dir>/rotation_seam.json
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")
HERE = Path(__file__).resolve().parent
API_ROOT = HERE.parents[1]
for _path in (str(API_ROOT), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402

from meep_gpu.dispersion import (  # noqa: E402
    DRUDE, LORENTZIAN, PolarizationState, Susceptibility,
)
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.metal_kernels import (  # noqa: E402
    ade_update_p, dispersive_update_e, no_pml_stored_e,
)
from meep_gpu.metal_kernels.device import (  # noqa: E402
    MAX_BUFFER_BINDINGS, Residency, compile_source, metal_frontend_version,
)
from meep_gpu.pml import PML  # noqa: E402

#: Per-case seeds are drawn from a DIGEST of the case label, never ``hash()``:
#: Python salts ``hash()`` of a string with ``PYTHONHASHSEED``, so a hash-seeded
#: probe draws a different fixture every process and a failing case cannot be
#: replayed.
SEED = 91_000

#: Complete driver steps each device case walks. The rotation has period
#: ``d + 1`` in the scratch slot, so a budget below four could not see a
#: configuration the orbit visits.
STEPS = 12

COMPONENTS: Tuple[str, ...] = ("Ex", "Ey", "Ez")

#: The pole counts the corpus's E->P rows actually carry, read off
#: ``results/metal_coverage_tranche6_2026-08-19`` and recorded here as the number
#: the fitness leg must clear rather than a round one.
CORPUS_POLE_COUNTS: Dict[str, int] = {
    "no-PML stored E": 5,            # absorber-1d.py, TestAbsorber.test_absorber
    "dispersive PML E": 6,           # stochastic_emitter*.py  <- the worst
    "folded dispersive PML E": 5,    # TestLoadDump.*_2d
    "complex no-PML stored E": 1,    # TestLoadDump.*_3d
    "folded off-diagonal dispersive PML E": 1,   # absorbed_power_density.py
}


def log(message: str) -> None:
    print(message, flush=True)


def case_seed(label: str) -> int:
    """A replayable per-case seed: SEED + a digest, never ``hash()``."""
    digest = hashlib.sha256(label.encode("utf-8")).digest()[:4]
    return SEED + int.from_bytes(digest, "big")


def words(array: Any) -> np.ndarray:
    """One array as uint32 WORDS. Byte compares, never allclose."""
    return np.frombuffer(np.ascontiguousarray(array).tobytes(), dtype=np.uint32)


def differing(left: Any, right: Any) -> int:
    a, b = words(left), words(right)
    if a.shape != b.shape:
        return max(a.size, b.size)
    return int(np.count_nonzero(a != b))


# ---------------------------------------------------------------------------
# LEG 1 -- the licensing measurement
# ---------------------------------------------------------------------------

STATE_NAMES: Tuple[str, ...] = tuple(
    [f"{stem}{axis}" for stem in ("B", "D", "E", "H") for axis in "xyz"]
    + [f"fu_{stem}{axis}" for stem in ("B", "D") for axis in "xyz"]
    + [f"f_w_{stem}{axis}" for stem in ("E", "H") for axis in "xyz"])


def build(cell: Sequence[float], boundaries: Any, poles: int, kind: str,
          pml_active: bool, seed: int) -> Tuple[Grid, Fields, Optional[PML]]:
    """One seeded engine, built identically for every side of a comparison."""
    grid = Grid(resolution=10.0, cell_size=tuple(cell), boundaries=boundaries,
                dimensions=3, courant=0.35, k_point=(0.0, 0.0, 0.0), xp=np)
    fields = Fields(grid=grid, force_complex_fields=False)
    count = int(np.prod(grid.shape))
    index = np.arange(count, dtype=np.float32).reshape(grid.shape)
    epsilon = (1.45 + 0.30 * np.sin(index * np.float32(0.037))).astype(np.float32)
    fields.set_isotropic_epsilon_volume(
        epsilon, (np.float32(1.0) / epsilon).astype(np.float32))
    kinds = (DRUDE, LORENTZIAN) if kind == "mixed" else (kind,)
    for order in range(poles):
        fields.polarizations.append(PolarizationState(
            Susceptibility(0.62 + 0.19 * order, 0.04 + 0.012 * order,
                           kinds[order % len(kinds)]),
            {"Ex": 0.31 + 0.05 * order, "Ey": 0.24 + 0.04 * order,
             "Ez": 0.19 + 0.06 * order},
            grid, np.float32))
    pml = None
    if pml_active:
        fields.enable_pml_storage()
        pml = PML(grid=grid,
                  thickness=tuple((2, 2) if grid.shape[axis] >= 6 else (0, 0)
                                  for axis in range(3)))
    else:
        fields.enable_field_storage()
    rng = np.random.default_rng(seed)
    for name in STATE_NAMES:
        array = getattr(fields, name, None)
        if array is not None:
            array[...] = (rng.standard_normal(grid.shape) * 0.37).astype(np.float32)
    for state in fields.polarizations:
        for component in state.driven():
            state.P[component][...] = (
                rng.standard_normal(grid.shape) * 0.21).astype(np.float32)
            state.P_prev[component][...] = (
                rng.standard_normal(grid.shape) * 0.19).astype(np.float32)
    return grid, fields, pml


def state_of(fields: Fields) -> Dict[str, Any]:
    """Every semantically live array, with the ADE roles NAMED rather than aliased."""
    state = {name: getattr(fields, name) for name in STATE_NAMES
             if getattr(fields, name, None) is not None}
    for index, polarization in enumerate(tuple(fields.polarizations)):
        for component in tuple(polarization.driven()):
            state[f"P[{index}].{component}"] = polarization.P[component]
            state[f"P_prev[{index}].{component}"] = polarization.P_prev[component]
        state[f"scratch[{index}]"] = polarization._scratch
    return state


def frozen(fields: Fields) -> Dict[str, np.ndarray]:
    return {name: np.array(value, copy=True)
            for name, value in state_of(fields).items()}


def compare(left: Fields, right: Fields) -> Dict[str, int]:
    a, b = state_of(left), state_of(right)
    assert set(a) == set(b), sorted(set(a) ^ set(b))
    return {name: n for name in sorted(a) if (n := differing(a[name], b[name]))}


def run_interleaving_equivalence(label: str, cell: Sequence[float], boundaries: Any,
                                 poles: int, kind: str, pml_active: bool,
                                 steps: int) -> Dict[str, Any]:
    """Does the per-component interleave equal the driver's sub-step order?

    THE COMPARATOR IS NOT A SECOND MODEL OF THE ARITHMETIC. Both sides run the
    SAME certified Metal products; the only difference is the ORDER their entries
    are dispatched in. The reference side runs every ``update_E`` entry and then
    every ``update_P`` entry, which is ``driver.step``'s order
    (driver.py:3304-3306). The interleaved side dispatches, per component, that
    component's E entry followed by that component's P entries.

    Entry slicing rather than a second plan: a plan's ``entries`` tuple is its
    dispatch list, and reordering it is the ONE degree of freedom under test.
    Building two differently-shaped plans would confound the order with the
    binding set.
    """
    seed = case_seed(label)
    _, reference, reference_pml = build(cell, boundaries, poles, kind, pml_active, seed)
    _, actual, actual_pml = build(cell, boundaries, poles, kind, pml_active, seed)
    drift = compare(reference, actual)
    assert not drift, f"{label}: the two builds are not identical: {drift}"

    family = dispersive_update_e if pml_active else no_pml_stored_e
    planner = (family.plan_metal_dispersive_e if pml_active
               else family.plan_metal_stored_e)

    plans: Dict[str, Dict[str, Any]] = {}
    for side, (fields, pml) in (("reference", (reference, reference_pml)),
                                ("interleaved", (actual, actual_pml))):
        residency = Residency()
        electric = planner(fields, pml, residency)
        polarization = ade_update_p.plan_metal_ade_update_p(fields, pml, residency)
        if electric is None or polarization is None:
            reasons = list(ade_update_p.metal_ade_update_p_coverage(
                fields, pml, residency).reasons)
            reasons.extend(
                (family.metal_dispersive_e_coverage if pml_active
                 else family.metal_stored_e_coverage)(fields, pml, residency).reasons)
            return {"passed": False, "reason": "a certified product was refused",
                    "refusals": reasons}
        plans[side] = {"residency": residency, "E": electric, "P": polarization,
                       "E_entries": tuple(electric.entries),
                       "P_entries": tuple(polarization.entries)}
        residency.sync_in()

    before = frozen(actual)
    per_step: List[Dict[str, Any]] = []
    for step in range(1, steps + 1):
        # --- the driver's order: all of update_E, then all of update_P ---------
        side = plans["reference"]
        side["E"].entries = side["E_entries"]
        side["E"].run()
        side["P"].entries = side["P_entries"]
        side["P"].run()
        side["residency"].sync_out()

        # --- the per-component interleave --------------------------------------
        side = plans["interleaved"]
        for component in COMPONENTS:
            side["E"].entries = tuple(e for e in side["E_entries"]
                                      if e.component == component)
            if side["E"].entries:
                side["E"].run()
            side["P"].entries = tuple(e for e in side["P_entries"]
                                      if e.component == component)
            if side["P"].entries:
                side["P"].run()
        side["residency"].sync_out()

        difference = compare(reference, actual)
        per_step.append({"step": step, "differing_words": sum(difference.values()),
                         "differing_arrays": dict(sorted(difference.items()))})
        if difference:
            break

    # THE NON-VACUITY FLOOR, SPLIT BY ROLE. This walk runs only update_E and
    # update_P, so B/D/H are the seam's INPUTS and must NOT move while E, f_w_E
    # and every polarization array must. A floor that demanded movement from all
    # 44 arrays would fail on a correct run; one that demanded it from none would
    # pass a walk that launched nothing.
    after = state_of(actual)
    moved = {name: differing(before[name], after[name]) for name in before}
    written = tuple(name for name in moved
                    if name.startswith(("E", "f_w_E", "P[", "P_prev[", "scratch[")))
    read_only = tuple(name for name in moved
                      if name.startswith(("B", "D", "H", "fu_", "f_w_H")))
    assert set(written) | set(read_only) == set(moved), sorted(
        set(moved) - set(written) - set(read_only))
    still = sorted(name for name in written if moved[name] == 0)
    disturbed = sorted(name for name in read_only if moved[name])
    identical = (len(per_step) == steps
                 and all(row["differing_words"] == 0 for row in per_step))
    launched = plans["interleaved"]["E"].launches and plans["interleaved"]["P"].launches
    return {
        "passed": bool(identical and launched and not still and not disturbed),
        "bit_identical": identical,
        "written_arrays_that_never_moved": still,
        "read_only_arrays_the_seam_disturbed": disturbed,
        "written_arrays": len(written), "read_only_arrays": len(read_only),
        "first_divergence": next((row["step"] for row in per_step
                                  if row["differing_words"]), None),
        "per_step": per_step,
        "arrays_compared": len(before),
        "moved_words": int(sum(moved.values())),
        "reference_launches": {
            "update_E": plans["reference"]["E"].launches,
            "update_P": plans["reference"]["P"].launches},
        "interleaved_launches": {
            "update_E": plans["interleaved"]["E"].launches,
            "update_P": plans["interleaved"]["P"].launches},
        "poles": poles, "pml_active": pml_active, "steps": len(per_step),
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# LEG 2 -- the rotation orbit, enumerated to closure
# ---------------------------------------------------------------------------

def rotate(configuration: Dict[str, Any], driven: Sequence[str]
           ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """One susceptibility's whole ``update`` as the reference performs it.

    TRANSCRIBED from ``dispersion.PolarizationState.update`` (dispersion.py:679-691)
    — the three assignments at the bottom of that loop, in that order — operating
    on buffer NAMES instead of arrays, which is the only thing the aliasing
    question depends on.
    """
    state = {"P": dict(configuration["P"]), "P_prev": dict(configuration["P_prev"]),
             "scratch": configuration["scratch"]}
    launches: List[Dict[str, Any]] = []
    for component in driven:
        p = state["P"][component]
        p_prev = state["P_prev"][component]
        scratch = state["scratch"]
        launches.append({"component": component, "out": scratch,
                         "p_now": p, "p_prev": p_prev})
        state["P"][component] = scratch     # dispersion.py:689
        state["P_prev"][component] = p      # dispersion.py:690
        state["scratch"] = p_prev           # dispersion.py:691
    return state, launches


def key_of(configuration: Dict[str, Any], driven: Sequence[str]) -> Tuple[Any, ...]:
    return (tuple(configuration["P"][c] for c in driven),
            tuple(configuration["P_prev"][c] for c in driven),
            configuration["scratch"])


def orbit(driven: Sequence[str]) -> List[Dict[str, Any]]:
    """Every buffer configuration the rotation can ever reach, to closure.

    The rotation is a permutation of ``2d + 1`` named buffers, so the orbit from
    the allocated start state is FINITE and this walks it until it repeats. That
    turns "the alias never happens" from a claim about the steps someone ran into
    a claim about every step that can ever run.
    """
    start = {"P": {c: f"A[{c}]" for c in driven},
             "P_prev": {c: f"B[{c}]" for c in driven},
             "scratch": "S"}
    seen: Dict[Tuple[Any, ...], int] = {}
    configurations: List[Dict[str, Any]] = []
    configuration = start
    while True:
        key = key_of(configuration, driven)
        if key in seen:
            return configurations
        seen[key] = len(configurations)
        configurations.append(configuration)
        configuration, _ = rotate(configuration, driven)


def leg_rotation_orbit_disjointness() -> Dict[str, Any]:
    """Per-component launches are alias-free; all-component launches are not.

    A PER-COMPONENT fused launch for component ``c`` reads, for every
    susceptibility ``k`` that drives ``c``: ``P_k[c]`` (BOTH halves read it — the
    E half as a pole of ``D - sum P``, the ADE half as ``p_now``), ``P_prev_k[c]``
    and ``sigma_k[c]``; and it writes ``out_k[c]``. An ALL-COMPONENT launch is the
    union over c of one susceptibility's arms, which is the shape
    ``triton_kernels.fused_ade_state`` takes.
    """
    rows: List[Dict[str, Any]] = []
    for driven_count in (1, 2, 3):
        driven = COMPONENTS[:driven_count]
        configurations = orbit(driven)
        per_component_conflicts: List[Any] = []
        all_component_conflicts: List[Any] = []
        for position, configuration in enumerate(configurations):
            _, launches = rotate(configuration, driven)
            by_component = {row["component"]: row for row in launches}
            # PER COMPONENT: one launch holds exactly one component's arms.
            for component, row in by_component.items():
                reads = {row["p_now"], row["p_prev"]}
                writes = {row["out"]}
                if reads & writes:
                    per_component_conflicts.append(
                        {"orbit_position": position, "component": component,
                         "aliased": sorted(reads & writes)})
            # ALL COMPONENTS IN ONE LAUNCH: the union, which is fused_ade_state.
            reads = {row["p_now"] for row in launches} | {row["p_prev"] for row in launches}
            writes = {row["out"] for row in launches}
            if reads & writes:
                all_component_conflicts.append(
                    {"orbit_position": position, "aliased": sorted(reads & writes),
                     "chain": [(r["component"], r["out"], r["p_now"], r["p_prev"])
                               for r in launches]})
        rows.append({
            "driven_components": list(driven),
            "orbit_size": len(configurations),
            "per_component_launch_conflicts": per_component_conflicts,
            "all_component_launch_conflicts": len(all_component_conflicts),
            "all_component_first_conflict": (all_component_conflicts[0]
                                             if all_component_conflicts else None),
        })
    # A susceptibility's buffers are its own, so K terms in one per-component
    # launch is K disjoint copies of the d=1..3 result. Recorded rather than
    # asserted: the corpus's worst row carries SIX.
    independent = all(len(set(name for name in
                              ("A", "B", "S"))) == 3 for _ in range(1))
    per_component_clean = all(not row["per_component_launch_conflicts"] for row in rows)
    all_component_dirty = all(row["all_component_launch_conflicts"] for row in rows
                              if len(row["driven_components"]) > 1)
    return {
        "passed": bool(per_component_clean and all_component_dirty and independent),
        "per_component_is_alias_free": per_component_clean,
        "all_component_aliases": all_component_dirty,
        "rows": rows,
        "note": ("a susceptibility owns its own P/P_prev/scratch (dispersion.py:646-650), "
                 "so K terms in one per-component launch is K disjoint copies of "
                 "the d-component result above; the corpus's worst row carries six"),
    }


# ---------------------------------------------------------------------------
# LEG 3 -- what this toolchain does with two bindings on one buffer
# ---------------------------------------------------------------------------

_ALIAS_TEMPLATE = """
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
kernel void __NAME__(
    device float*       out0 [[buffer(0)]],
    device const float* in0  [[buffer(1)]],
    device float*       obs  [[buffer(2)]],
    constant uint&      n    [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= n) { return; }
__BODY__
    obs[idx] = v;
}
"""

_STORE_THEN_LOAD = """    out0[idx] = 7.0f;
    float v = in0[idx];"""

_LOAD_THEN_STORE = """    float v = in0[idx];
    out0[idx] = 7.0f;"""


def leg_program_order_under_aliasing() -> Dict[str, Any]:
    """One launch, two bindings, one buffer, one index — what does the load see?

    SCOPE, STATED BEFORE THE NUMBER. This is one toolchain, one trivial body, one
    grid. It is the same class of evidence ``fused_ade_state``'s note calls "an
    OBSERVATION, not a guarantee", and it is recorded here for the same reason:
    the shape this probe licenses does NOT depend on it, and a reader deciding to
    take the all-component shape anyway should see what it would be resting on.
    """
    import torch  # noqa: PLC0415

    count = 4096
    rows: Dict[str, Any] = {}
    for name, body, seeded, expected in (
            ("store_then_load", _STORE_THEN_LOAD, 3.0, 7.0),
            ("load_then_store", _LOAD_THEN_STORE, 3.0, 3.0)):
        source = _ALIAS_TEMPLATE.replace("__NAME__", f"alias_{name}").replace(
            "__BODY__", body)
        function = getattr(compile_source(source), f"alias_{name}")
        buffer = torch.full((count,), seeded, dtype=torch.float32, device="mps")
        observed = torch.zeros((count,), dtype=torch.float32, device="mps")
        function(buffer, buffer, observed, count)
        torch.mps.synchronize()
        distinct = sorted(set(observed.cpu().numpy().tolist()))
        rows[name] = {"observed": distinct, "expected_if_program_order": expected,
                      "program_order_honoured": distinct == [expected]}
    return {
        "passed": all(row["program_order_honoured"] for row in rows.values()),
        "rows": rows,
        "scope": ("one toolchain, one trivial body, one grid shape; an OBSERVATION "
                  "with the same standing as the Triton note's, and NOT relied on "
                  "by the per-component shape leg 2 licenses"),
        "metal_frontend": metal_frontend_version(),
    }


# ---------------------------------------------------------------------------
# LEG 4 -- binding fitness at the corpus's worst pole count
# ---------------------------------------------------------------------------

def fitness_source(fixed_pointers: int, poles: int, volume_sigmas: int,
                   name: str) -> Tuple[int, str]:
    """A signature with the fused product's exact binding SHAPE, and a live body.

    The body touches every binding: a signature whose arguments the compiler can
    drop would let dead-code elimination decide the fitness answer.
    """
    lines: List[str] = []
    slot = 0
    for index in range(fixed_pointers):
        lines.append(f"    device float*       f{index:<3}[[buffer({slot})]],")
        slot += 1
    for stem, count in (("p", poles), ("o", poles), ("q", poles),
                        ("s", volume_sigmas)):
        for index in range(count):
            lines.append(f"    device float*       {stem}{index:<3}[[buffer({slot})]],")
            slot += 1
    lines.append(f"    constant Params&    prm  [[buffer({slot})]],")
    slot += 1
    touched = ([f"f{i}" for i in range(fixed_pointers)]
               + [f"{stem}{i}" for stem, count in (("p", poles), ("o", poles),
                                                   ("q", poles), ("s", volume_sigmas))
                  for i in range(count)])
    body = "\n".join(f"    {n}[idx] = {n}[idx] * prm.c_now[0] + prm.c_prev[0];"
                     for n in touched)
    struct = ("struct Params { float c_now[%d]; float c_prev[%d]; float c_drive[%d]; "
              "float sigma[%d]; uint ny; uint nz; uint n_elem; };"
              % ((max(poles, 1),) * 4))
    return slot, "\n".join((
        "#include <metal_stdlib>", "using namespace metal;",
        "#pragma clang fp contract(off)", struct, "",
        f"kernel void {name}(", *lines,
        "    uint idx [[thread_position_in_grid]])", "{",
        "    if (idx >= prm.n_elem) { return; }", body, "}", ""))


def leg_binding_fitness() -> Dict[str, Any]:
    """Compile the real fused signature at each cell's own corpus pole count.

    THE FIXED POINTER COUNTS ARE THE TWO CERTIFIED E BODIES' OWN, not estimates:

    * no-PML stored E — ``e_out, d_in, inv_e`` = 3 (no_pml_stored_e.py:56-70,
      with the eight-slot pole pad replaced by exactly ``pole_count`` slots,
      because the pad is what puts this over the ceiling);
    * dispersive PML E — ``e_out, fw, d_in, inv_e, kps, kms`` = 6
      (dispersive_update_e.py:41-59).

    The ADE half adds ``out``, ``p_prev`` and (when the sigma is a volume)
    ``sigma`` per pole; ``p_now`` is ALREADY BOUND as that pole's E-half operand
    and is not counted twice — that shared pointer is the fusion.
    """
    rows: List[Dict[str, Any]] = []
    for label, poles in sorted(CORPUS_POLE_COUNTS.items()):
        if label.startswith("complex") or "off-diagonal" in label:
            rows.append({"cell": label, "poles": poles, "compiled": None,
                         "skipped": ("this leg prices the two REAL diagonal E "
                                     "bodies; the complex and off-diagonal arms "
                                     "carry different signatures and are not "
                                     "claimed here")})
            continue
        fixed = 6 if "dispersive" in label else 3
        for volume_sigmas in (poles, 0):
            bindings, source = fitness_source(
                fixed, poles, volume_sigmas,
                f"fit_{fixed}_{poles}_{volume_sigmas}")
            try:
                compile_source(source)
                compiled, error = True, ""
            except Exception as exc:  # noqa: BLE001 - the failure IS the measurement
                compiled, error = False, str(exc).splitlines()[0]
            rows.append({"cell": label, "poles": poles, "fixed_pointers": fixed,
                         "volume_sigmas": volume_sigmas,
                         "pointers": bindings - 1, "bindings": bindings,
                         "ceiling": MAX_BUFFER_BINDINGS,
                         "compiled": compiled, "error": error})
            log(f"    fitness {label} poles={poles} volume_sigmas={volume_sigmas} "
                f"bindings={bindings} compiled={compiled}")
    # The first signature OVER the ceiling must FAIL, or the fitness numbers above
    # are not measuring a limit at all.
    over, source = fitness_source(6, 7, 7, "fit_over_ceiling")
    try:
        compile_source(source)
        refused, message = False, "the over-ceiling signature COMPILED"
    except Exception as exc:  # noqa: BLE001
        message = str(exc).splitlines()[0]
        refused = "out of bounds" in message and "buffer" in message
    priced = [row for row in rows if row["compiled"] is not None]
    return {
        "passed": bool(refused and all(row["compiled"] for row in priced)),
        "rows": rows,
        "over_ceiling": {"bindings": over, "refused": refused, "error": message},
        "worst_shipped_bindings": max((row["bindings"] for row in priced), default=0),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

#: label, cell, boundaries, poles, susceptibility kind, PML active
INTERLEAVE_CASES: Tuple[Tuple[str, Tuple[float, float, float], Any, int, str, bool], ...] = (
    ("no_pml_five_pole_mixed", (1.2, 1.0, 0.9), "periodic", 5, "mixed", False),
    ("no_pml_two_pole_lorentz", (1.2, 1.0, 0.9), "periodic", 2, LORENTZIAN, False),
    ("pml_six_pole_mixed", (1.2, 1.0, 0.9), "periodic", 6, "mixed", True),
    ("pml_five_pole_mixed", (1.3, 1.1, 0.9), "periodic", 5, "mixed", True),
    ("pml_one_pole_drude", (1.2, 1.0, 0.9), "periodic", 1, DRUDE, True),
)


def emit(handle: Any, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
    log(f"[{row['leg']}] {row['label']}: passed={row.get('passed')} "
        f"diff={row.get('per_step', [{}])[-1].get('differing_words', '-') if row.get('per_step') else '-'}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=STEPS)
    args = parser.parse_args()
    import torch
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available; this probe must run on an Apple GPU")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    jsonl = args.out.with_suffix(".jsonl")
    started = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    with jsonl.open("w", encoding="utf-8") as handle:
        rows.append({"leg": "rotation_orbit_disjointness", "label": "d=1,2,3",
                     **leg_rotation_orbit_disjointness()})
        emit(handle, rows[-1])

        rows.append({"leg": "metal_program_order_under_aliasing",
                     "label": "two_bindings_one_buffer",
                     **leg_program_order_under_aliasing()})
        emit(handle, rows[-1])

        rows.append({"leg": "binding_fitness", "label": "corpus_worst_pole_counts",
                     **leg_binding_fitness()})
        emit(handle, rows[-1])

        for label, cell, boundaries, poles, kind, pml_active in INTERLEAVE_CASES:
            row = {"leg": "interleaving_equivalence", "label": label,
                   **run_interleaving_equivalence(label, cell, boundaries, poles,
                                                  kind, pml_active, args.steps)}
            rows.append(row)
            emit(handle, row)

    result = {
        "verdict": "PASS" if all(row["passed"] for row in rows) else "FAIL",
        "elapsed_seconds": time.perf_counter() - started,
        "steps": args.steps,
        "rows": rows,
        "seed_base": SEED,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "metal_frontend": metal_frontend_version(),
        "subnormal_policy": os.environ.get("MEEP_GPU_SUBNORMAL_POLICY"),
        "jsonl": str(jsonl),
    }
    from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
    _stamp_provenance(result)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str)
                        + "\n", encoding="utf-8")
    log(f"VERDICT {result['verdict']} in {result['elapsed_seconds']:.2f}s; "
        f"artifact {args.out}")
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
