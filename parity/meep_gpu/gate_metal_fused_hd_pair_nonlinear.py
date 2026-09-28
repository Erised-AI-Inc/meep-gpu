#!/usr/bin/env python3
"""Native MPS byte gate for the H->D weld's NONLINEAR WIDENING — the one leg its
release still owed.

THIS GATE CERTIFIES NO NEW KERNEL AND NO NEW PRODUCT, and saying so first is the point
of the file. ``meep_gpu/metal_kernels/fused_hd_pair.py`` already ADMITS a chi2/chi3 run:
its predicate conjoins each half's ordinary verdict with
``nonlinear_update_e``'s spine verdict, and ``launch.FUSED_PAIR_EXTRA_ARMS`` carries the
matching absorb row. What it did not have was a MEASUREMENT. The board records the gap
in its own words (``build_fusion_matrix.py``, the ``fused_hd_pair`` row's
``not_evaluable`` list): *"the nonlinear widening ... is admitted by the shipped
predicate and is NOT credited by this board: no device gate has driven a complete step
on a chi2/chi3 configuration."* This gate is that drive.

WHY IT IS A WIDENING AND NOT A THIRD KERNEL, stated as a claim this gate MEASURES
rather than repeats. A chi2/chi3 run selects ``nonlinear_update_e``'s two SPINE arms on
this seam's slots -- ``nonlinear PML magnetic`` at ``update_H`` and ``nonlinear PML
curl`` at ``step_D`` -- and both build through the CERTIFIED builders:
``plan_nonlinear_run_constitutive`` calls ``launch.plan_constitutive`` and
``plan_nonlinear_run_pml_curl`` calls ``launch.plan_pml_curl``, each behind a
``_LinearScopeView`` that hides chi2/chi3 and nothing else. The Pade factor enters
``update_E``, one seam LATER, outside this span entirely. Leg ``text_identity`` drives
that: the sources the two spine plans hand ``compile_source`` are compared as bytes
against the ones the LINEAR arms hand it on the same geometry, and against the two
emitters the weld itself lifts.

**A SECOND PRODUCT HERE WOULD BE A DEFECT, not a widening.** Two fused products
admitting one slot is an over-covering composition the arm table refuses by design, and
the shipped predicate already admits this cell -- measured in leg ``no_second_product``,
which asks every registered ``update_H`` weld whether it admits the nonlinear fixture
and requires exactly one to.

THE CLAIM IS PER COMPLETE DRIVER STEP, as uint32 WORDS over every stored volume the
engine allocates (never ``allclose``: ``-0.0 == 0.0`` lies), against FOUR references
from one seed:

  1. the ARRAY PATH -- ``stepping``'s passes under the driver's own loop, with the Pade
     ``update_E`` where the driver runs it;
  2. the CERTIFIED NONLINEAR SINGLES -- ``plan_nonlinear_run_constitutive(..., 'H')`` at
     ``update_H`` and ``plan_nonlinear_run_pml_curl(..., 'step_D')`` at ``step_D``;
  3. the COMPOSITION THE COMPOSER INSTALLS TODAY -- ``plan_step(fuse=True)``;
  4. the same slots DISPATCHED UNFUSED -- ``plan_step(fuse=False)``.

=============================================================================
THE ARBITRATION FINDING THIS CELL EXISTS FOR
=============================================================================

``fused_hd_pair.INSTALLABLE_REASON`` names this cell as the ONE place a H->D product
would be a genuine gain: *"the 2 remaining rows are the nonlinear cell (3rd-harm-1d),
where neither neighbour installs and an H->D product would be a gain of one seam and one
launch."* Leg ``arbitration`` drives the shipped composer on the nonlinear fixtures and
RECORDS what it selects rather than asserting a slot table -- the substantive claims it
does assert are that this product is installed on nothing today, that every slot is
still filled, and that the count of installed pairs is what the gain arithmetic reads.
**The flag is not flipped by this gate and nothing here licenses flipping it**: a gain
is a reason to consider installing, and installing is a composition decision with its
own release.

LEGS
  host
  text_identity     the spine arms' sources are the linear arms' sources, byte for byte,
                    and are the two emitters the weld lifts
  no_second_product exactly one registered update_H weld admits the nonlinear fixture
  refusal           the widening is admitted, and a LINEAR run is still admitted too;
                    the nonlinear-only spine refuses a linear grid by name
  arbitration       the composer, asked, on the nonlinear fixtures
  fixture_shape     the fixtures carry metal_composition_matrix.nonlinear_real's own
                    chi2/chi3 and the corpus cell's 1-D shape
  policy            the resolved subnormal policy, recorded
  device
  product           the four-reference identity on the nonlinear fixtures
  race              H and f_w_H bound IN PLACE must diverge
  rotation          the post-launch rotation dropped must diverge
  launch_structure  launches per step at the seam and over the whole step
  mutation          the shared gate's armed shader defects, on a NONLINEAR fixture
  lift              both corpus rows of the (nonlinear PML magnetic -> nonlinear PML
                    curl) cell, re-lifted in their own interpreter and driven
  byte_neutral      the one armed edit required NOT to diverge
  disarm            the identical harness, shipped bytes, must not diverge

THE KERNEL, THE MUTATIONS AND THE WHOLE HARNESS ARE ``gate_metal_fused_hd_pair``'S,
imported rather than copied: this gate widens that gate's subject to a configuration it
did not drive, and a second copy of the mutation table would be a second place for the
same needles to go stale.

Rule 7: one flushed line per case; every row appended and fsynced as it lands.
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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")
HERE = Path(__file__).resolve().parent
API_ROOT = next(parent for parent in HERE.parents
                if (parent / "meep_gpu" / "metal_kernels").is_dir())
for _path in (str(API_ROOT), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402

import gate_metal_fused_hd_pair as _shared  # noqa: E402

from meep_gpu import withdraw_hoist  # noqa: E402
from meep_gpu.metal_kernels import (  # noqa: E402
    fused_hd_pair as family, launch as metal_launch, nonlinear_update_e as nonlinear,
    shaders, subnormal,
)
from meep_gpu.metal_kernels.device import (  # noqa: E402
    MAX_BUFFER_BINDINGS, Residency, compile_source, metal_frontend_version,
)

ABSORBED = _shared.ABSORBED
AliasedLaunch = _shared.AliasedLaunch
Arrangement = _shared.Arrangement
RotationSkipped = _shared.RotationSkipped
Shim = _shared.Shim
SyncedPlan = _shared.SyncedPlan
declaring = _shared.declaring
differing = _shared.differing
drive = _shared.drive
pin_array_path = _shared.pin_array_path
stored_volumes = _shared.stored_volumes
synced = _shared.synced
words = _shared.words

#: ``measure_predicate_coverage`` digests this package into every lifted row's
#: ``subject_manifest_sha256``. Declared, as every battery declares it.
SUBJECT_PACKAGE = "metal_kernels"

STEPS = 60
LIFT_CLEAN_STEP_FLOOR = 8
SEED_SCALE_BITS = _shared.SEED_SCALE_BITS

#: ``metal_composition_matrix.nonlinear``'s own values, so the material this certifies
#: is the material the nonlinear family was certified on.
CHI2 = 0.1
CHI3 = 0.2
NONLINEAR_COMPONENTS: Tuple[str, ...] = ("Ez",)

#: name -> the fixture's keywords.
#:
#: THE 1-D ROW IS NOT A DECORATION. Both corpus rows of this cell are 1-D
#: (``examples:3rd-harm-1d.py`` is [1, 1, 2500] and
#: ``tests:Test3rdHarm1d.test_3rd_harm_1d`` is [1, 1, 2000]), and a 1-D grid is where a
#: weld's index decode and its ghost rule are least like the 3-D case it was written
#: on: two of the three extents are 1, so every backward neighbour on those axes is the
#: cell itself under a periodic wrap and the ownership mask covers a whole volume.
#: A gate that drove only 3-D fixtures would credit the cell without touching its shape.
#:
#: THE SCALAR-CHI ROW IS THE OTHER COEFFICIENT ARM. ``nonlinear_update_e`` takes chi
#: either as a volume or as a packed scalar, and while the Pade factor lives one seam
#: LATER the arm a run takes decides which predicate answers here.
CASES: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("nonlinear_3d_periodic", {}),
    ("nonlinear_3d_walled",
     {"boundaries": {axis: "metallic" for axis in "xyz"}}),
    ("nonlinear_3d_scalar_chi", {"scalar": True}),
    ("nonlinear_3d_all_components",
     {"components": ("Ex", "Ey", "Ez"),
      "boundaries": {axis: "metallic" for axis in "xyz"}}),
    ("nonlinear_1d_like_the_corpus", {"cell": (0.0, 0.0, 4.0), "pml": 2}),
    ("nonlinear_1d_walled", {"cell": (0.0, 0.0, 4.0), "pml": 2,
                             "boundaries": {"z": "metallic"}}),
)

#: The case the shader mutations are armed on: every axis walled in 3-D, so every line
#: the shipped kernel can emit is present and the metallic ghost rule is live.
MUTATION_CASE = "nonlinear_3d_walled"

#: The all-periodic 3-D case, where the ownership mask does not fire and the periodic
#: wrap is observable -- the shared gate's own second mutation specialisation.
PERIODIC_MUTATION_CASE = "nonlinear_3d_periodic"

RESOLUTION = 10.0
COURANT = 0.35
PML_CELLS = 2
EPSILON: Dict[str, float] = {"Ex": 2.0, "Ey": 2.5, "Ez": 3.0}
CELL: Tuple[float, float, float] = (2.0, 2.1, 1.2)

MODES: Tuple[str, ...] = ("array", "singles", "composition_today", "unfused",
                          "weld", "weld_seam_only")

LEG_GROUPS: Dict[str, Tuple[str, ...]] = {
    "host": ("text_identity", "no_second_product", "refusal", "arbitration",
             "fixture_shape", "policy"),
    "device": ("product", "race", "rotation", "launch_structure", "mutation",
               "lift", "byte_neutral", "disarm"),
}
ALL_LEGS: Tuple[str, ...] = tuple(leg for group in LEG_GROUPS.values() for leg in group)

CENSUS = "metal_coverage_2026-09-04_m0complex"
SEAM_RECORD = "h_to_d_seam_2026-09-04"
CELL_ARMS: Tuple[str, str] = ("nonlinear PML magnetic", "nonlinear PML curl")

#: The block key this gate's battery hook writes and the lift leg reads back.
BLOCK = "metal_fused_hd_pair_nonlinear_gate"

EXIT_INCOMPLETE = 75


def log(message: str) -> None:
    print(message, flush=True)


# ---------------------------------------------------------------------------
# The fixture
# ---------------------------------------------------------------------------

def codes_of(name: str) -> Tuple[int, ...]:
    walls = dict(CASES)[name].get("boundaries") or {}
    return tuple(1 if walls.get(axis) == "metallic" else 0 for axis in "xyz")


MUTATION_CODES: Tuple[int, ...] = codes_of(MUTATION_CASE)
PERIODIC_CODES: Tuple[int, ...] = codes_of(PERIODIC_MUTATION_CASE)


def install_nonlinearity(fields: Any, keywords: Mapping[str, Any]) -> Dict[str, Any]:
    """Install chi2/chi3 on the named components, the composition matrix's own way.

    ``metal_composition_matrix.nonlinear`` is not called directly because this gate
    builds a DRIVER rather than a bare ``Fields`` triple; the values and the two
    coefficient arms are that function's, and leg ``fixture_shape`` MEASURES the
    equality rather than asserting it.
    """
    components = tuple(keywords.get("components", NONLINEAR_COMPONENTS))
    shape = tuple(fields.grid.shape)
    scalar = bool(keywords.get("scalar", False))
    def value(magnitude: float) -> Any:
        return (np.float32(magnitude) if scalar
                else np.full(shape, np.float32(magnitude)))

    fields.set_nonlinear_volumes({name: value(CHI2) for name in components},
                                 {name: value(CHI3) for name in components})
    return {"components": list(components), "scalar": scalar,
            "chi2": CHI2, "chi3": CHI3}


def build_driver(keywords: Mapping[str, Any], seed: int,
                 sources: Sequence[Mapping[str, Any]] = (),
                 scale_bits: int = SEED_SCALE_BITS) -> Any:
    """One seeded ``FdtdDriver`` carrying an instantaneous chi2/chi3."""
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415

    cell = tuple(keywords.get("cell", CELL))
    dimensions = 1 if cell[0] == 0.0 and cell[1] == 0.0 else 3
    driver = FdtdDriver(cell_size=cell, resolution=RESOLUTION, courant=COURANT,
                        force_complex_fields=False,
                        boundaries=dict(keywords.get("boundaries") or {}) or None,
                        dimensions=dimensions)
    # A PER-AXIS THICKNESS ON A REDUCED-DIMENSION GRID. The engine REFUSES a layer
    # on an axis `dimensions=1` makes translationally invariant (driver.py:2207) --
    # correctly: an invariant axis has no face to absorb at. So the 1-D fixtures put
    # the layer on the one axis they resolve, which is what MEEP does too.
    cells = int(keywords.get("pml", PML_CELLS))
    if dimensions == 1:
        driver.setup_pml(((0, 0), (0, 0), (cells, cells)))
    else:
        driver.setup_pml(cells)
    shape = tuple(driver.grid.shape)
    driver.fields.set_epsilon_volumes(
        {name: np.full(shape, np.float32(value), np.float32)
         for name, value in EPSILON.items()},
        {name: np.full(shape, np.float32(1.0 / value), np.float32)
         for name, value in EPSILON.items()})
    install_nonlinearity(driver.fields, keywords)
    for source in sources:
        driver.add_source(dict(source))
    rng = np.random.default_rng(seed)
    scale = np.float32(2.0) ** int(scale_bits)
    for array in stored_volumes(driver.fields).values():
        array[...] = (0.37 * rng.standard_normal(array.shape)).astype(array.dtype)
        array *= scale
    driver.invalidate_fast_path()
    pin_array_path(driver)
    return driver


# ---------------------------------------------------------------------------
# The arrangements
# ---------------------------------------------------------------------------

def arrangement_singles(driver: Any) -> Arrangement:
    """Reference 2: the two certified NONLINEAR SPINE singles at the seam's slots."""
    residency = Residency()
    fields, pml = driver.fields, driver.pml
    constitutive = nonlinear.plan_nonlinear_run_constitutive(fields, pml, "H",
                                                             residency)
    curl = nonlinear.plan_nonlinear_run_pml_curl(fields, pml, "step_D", residency)
    if constitutive is None or curl is None:
        raise RuntimeError(
            f"a certified nonlinear spine single was refused on a fixture this gate "
            f"expects it to admit ({constitutive!r}, {curl!r})")
    shim = Shim(fields, {"update_H": synced(constitutive, residency),
                         "step_D": synced(curl, residency)})
    return Arrangement("singles", shim, residency,
                       selected={"update_H": "nonlinear PML magnetic",
                                 "step_D": "nonlinear PML curl"})


def build_weld(driver: Any, residency: Residency,
               functions: Optional[Mapping[str, Any]] = None) -> Any:
    plan = family.plan_metal_fused_hd_pair(driver.fields, driver.pml, sources=(),
                                           residency=residency, functions=functions)
    if plan is None:
        reasons = family.metal_fused_hd_pair_coverage(driver.fields, driver.pml, (),
                                                      residency).reasons
        raise RuntimeError("the H/D pair was refused on a NONLINEAR fixture, which "
                           "is the widening this gate exists to drive: "
                           + "; ".join(reasons))
    return plan


def arrangement_weld(driver: Any, functions: Optional[Mapping[str, Any]] = None,
                     launcher: Optional[Callable[[Any, Residency, Any], Any]] = None,
                     rest_unfused: bool = True, name: str = "weld") -> Arrangement:
    residency = Residency()
    plan = build_weld(driver, residency, functions=functions)
    occupant: Any = (plan if launcher is None
                     else launcher(plan, residency, driver.pml))
    occupant = SyncedPlan(occupant, residency)
    plans: Dict[str, Any] = {}
    if rest_unfused:
        base, _residency = _shared._composed(driver, fuse=False,  # noqa: SLF001
                                             residency=residency)
        for slot in metal_launch.STEP_ORDER:
            if slot in base.plans and slot not in family.REPLACES:
                plans[slot] = synced(base.plans[slot], residency)
    plans["update_H"] = occupant
    plans["step_D"] = ABSORBED
    return Arrangement(name, Shim(driver.fields, plans), residency,
                       selected={"update_H": "fused H/D pair (forced, nonlinear)",
                                 "step_D": "fused H/D pair (forced, nonlinear)"})


def all_arrangements(driver: Any, functions: Optional[Mapping[str, Any]] = None,
                     launcher: Optional[Callable[[Any, Residency, Any], Any]] = None,
                     ) -> Dict[str, Arrangement]:
    return {
        "array": Arrangement("array", None, None),
        "singles": arrangement_singles(driver),
        "composition_today": _shared.arrangement_composition(driver, fuse=True),
        "unfused": _shared.arrangement_composition(driver, fuse=False),
        "weld": arrangement_weld(driver, functions=functions, launcher=launcher),
        "weld_seam_only": arrangement_weld(driver, functions=functions,
                                           launcher=launcher, rest_unfused=False,
                                           name="weld_seam_only"),
    }


def run_product(driver: Any, steps: int,
                progress: Optional[Callable[[str], None]] = None,
                movement_floor: str = "all",
                functions: Optional[Mapping[str, Any]] = None,
                launcher: Optional[Callable[[Any, Residency, Any], Any]] = None,
                stop_on_divergence: bool = True,
                require_full_budget: bool = True,
                clean_floor: int = 0) -> Dict[str, Any]:
    """The four-reference identity on one nonlinear driver, per complete step."""
    arrangements = all_arrangements(driver, functions=functions, launcher=launcher)
    result = drive(driver, arrangements, steps, progress=progress,
                   secondary="singles", stop_on_divergence=stop_on_divergence)
    legs = result["arrangements"]
    identical = all(legs[name]["identical"] for name in legs)
    compared = int(result["steps_compared"])
    budget_ok = (compared == steps if require_full_budget
                 else compared >= max(clean_floor, 1))
    launched = all(legs[name]["launches"]["plans"] > 0
                   and legs[name]["launches"]["plans"]
                   == legs[name]["launches"]["functions"]
                   for name in legs)
    weld_ok = all(
        legs[name]["dispatched"].get("update_H", 0) == result["steps_stepped"]
        and legs[name]["absorbed"].get("step_D", 0) == result["steps_stepped"]
        for name in ("weld", "weld_seam_only"))
    seam_outputs = tuple(name for name in
                         [f"{stem}{axis}" for stem in ("D", "fu_D", "H", "f_w_H")
                          for axis in "xyz"]
                         if name in result["volumes_compared"])
    seam_moved = {name: int(result["moved_from_seed"].get(name, 0))
                  for name in seam_outputs}
    if movement_floor == "all":
        # A 1-D FIXTURE MOVES FEWER COMPONENTS THAN A 3-D ONE, and that is the grid
        # rather than the kernel: on a [1, 1, n] grid the curl's transverse
        # differences are exact zeros by construction. So the floor here is the
        # SEAM's outputs rather than every stored volume, with the per-volume split
        # recorded so a reader can see which components the fixture carries.
        floor = bool(seam_outputs) and sum(seam_moved.values()) > 0
    else:
        floor = bool(seam_outputs) and sum(seam_moved.values()) > 0
    settled = bool(result["every_arrangement_gave_the_engine_its_volumes_back"])
    census_covered = result["the_intermediate_census_modelled_every_helper_the_step_ran"]
    result.update({
        "passed": bool(identical and launched and weld_ok and floor and budget_ok
                       and settled and result["step_error"] is None
                       and census_covered is not False
                       and (result["precondition_clean"] or not require_full_budget)),
        "bit_identical": identical,
        "budget_met": budget_ok,
        "every_arrangement_launched_and_the_two_counters_agree": launched,
        "weld_launched_once_per_step_and_absorbed_step_D": weld_ok,
        "movement_floor_met": floor,
        "seam_output_words_moved": seam_moved,
        "seam_output_volumes_that_never_moved": sorted(
            name for name, n in seam_moved.items() if not n),
        "weld_agrees_with_the_certified_nonlinear_singles":
            legs["weld"]["identical_vs_secondary"],
        "the_seam_alone_agrees_with_the_array_path":
            legs["weld_seam_only"]["identical"],
        "references_that_disagree_with_the_array_path": sorted(
            name for name in ("singles", "composition_today", "unfused")
            if not legs[name]["identical"]),
        "composition_today_selected": legs["composition_today"]["selected"],
    })
    return result


# ---------------------------------------------------------------------------
# Host legs
# ---------------------------------------------------------------------------

def _indented(text: str, width: int = 4) -> str:
    """``offdiag_weld_common.step_cell_function``'s own indentation of a lifted tail."""
    return "\n".join(" " * width + line if line.strip() else line
                     for line in text.splitlines())


def leg_text_identity() -> Dict[str, Any]:
    """The spine arms' sources ARE the linear arms', byte for byte.

    THE CLAIM THE WIDENING RESTS ON, driven rather than repeated -- and the witness is
    the compiler cache's OWN KEY SET, which is the source text itself.
    ``device.compile_source`` memoises on the SOURCE STRING
    (``device.py:_LIBRARY_CACHE``), so building the LINEAR arm and then the NONLINEAR
    SPINE arm on the same geometry adds a second key if and only if the two emitted
    texts differ by one byte. The leg therefore snapshots the cache, builds the linear
    plan, builds the spine plan, and requires the spine build to have added NOTHING --
    with the key the linear build added compared by sha256 against
    ``shaders.curl_source`` / ``shaders.constitutive_source('H')``, the two emitters
    the WELD itself lifts.

    A CACHE HIT IS A STRONGER WITNESS THAN A DIGEST COMPARISON I COMPUTE MYSELF: it is
    the same dictionary lookup the compile path makes, so it cannot be right about a
    string the compiler never saw.
    """
    from meep_gpu.metal_kernels import device as _device  # noqa: PLC0415

    rows: Dict[str, Any] = {}
    for name, keywords in CASES:
        driver = build_driver(keywords, 4141)
        linear = build_driver({k: v for k, v in keywords.items()
                               if k not in ("scalar", "components")}, 4141)
        linear.fields.set_nonlinear_volumes({}, {})
        from meep_gpu.stepping import _boundary_kinds  # noqa: PLC0415

        codes = tuple(1 if kind == "metallic" else 0
                      for kind in _boundary_kinds(driver.grid, driver.pml))
        emitted_curl = shaders.curl_source(
            codes, bool(metal_launch.SUB_STEPS["step_D"]["backward"]))
        emitted_constitutive = shaders.constitutive_source("H")

        before = set(_device._LIBRARY_CACHE)  # noqa: SLF001
        linear_curl = metal_launch.plan_pml_curl(linear.fields, linear.pml, "step_D",
                                                 Residency())
        linear_constitutive = metal_launch.plan_constitutive(
            linear.fields, linear.pml, "H", Residency())
        after_linear = set(_device._LIBRARY_CACHE)  # noqa: SLF001
        spine_curl = nonlinear.plan_nonlinear_run_pml_curl(
            driver.fields, driver.pml, "step_D", Residency())
        spine_constitutive = nonlinear.plan_nonlinear_run_constitutive(
            driver.fields, driver.pml, "H", Residency())
        after_spine = set(_device._LIBRARY_CACHE)  # noqa: SLF001

        linear_added = after_linear - before
        spine_added = after_spine - after_linear
        rows[name] = {
            "codes": list(codes),
            "the_spine_curl_plan_was_built": spine_curl is not None,
            "the_spine_constitutive_plan_was_built": spine_constitutive is not None,
            "the_linear_curl_plan_was_built": linear_curl is not None,
            "the_linear_constitutive_plan_was_built": linear_constitutive is not None,
            "sources_the_linear_build_compiled": len(linear_added),
            "sources_the_spine_build_compiled_that_the_linear_build_had_not":
                sorted(hashlib.sha256(text.encode()).hexdigest()[:16]
                       for text in spine_added),
            "the_spine_compiled_nothing_new": not spine_added,
            "the_linear_build_compiled_the_certified_curl":
                emitted_curl in linear_added or emitted_curl in before,
            "the_linear_build_compiled_the_certified_constitutive":
                emitted_constitutive in linear_added
                or emitted_constitutive in before,
            "curl_source_sha256": hashlib.sha256(emitted_curl.encode()).hexdigest(),
            "constitutive_source_sha256": hashlib.sha256(
                emitted_constitutive.encode()).hexdigest(),
            # AND THE WELD LIFTS THOSE SAME TWO TEXTS -- the third side of the
            # triangle, checked on the emitted text rather than on a name.
            # THE LIFT INDENTS THE TAIL BY FOUR, and nothing else: `step_cell_function`
            # wraps it in an inline function body. So the comparison indents the same
            # way rather than loosening to a "looks similar" test.
            "the_weld_lifts_the_certified_constitutive_tail":
                _indented(family.certified_constitutive_tail())
                in family.h_cell_function(),
            "the_welds_curl_half_is_this_curls_body":
                family.welded_curl_tail(codes) in family.fused_hd_pair_source(codes),
        }
        log(f"    text_identity {name}: linear compiled "
            f"{rows[name]['sources_the_linear_build_compiled']} sources, spine "
            f"compiled {len(spine_added)} new")
    return {
        "passed": all(
            row["the_spine_curl_plan_was_built"]
            and row["the_spine_constitutive_plan_was_built"]
            and row["the_linear_curl_plan_was_built"]
            and row["the_linear_constitutive_plan_was_built"]
            and row["the_spine_compiled_nothing_new"]
            and row["the_linear_build_compiled_the_certified_curl"]
            and row["the_linear_build_compiled_the_certified_constitutive"]
            and row["the_weld_lifts_the_certified_constitutive_tail"]
            and row["the_welds_curl_half_is_this_curls_body"]
            for row in rows.values()),
        "cases": rows,
        "witness": ("device._LIBRARY_CACHE is keyed on the SOURCE STRING, so a spine "
                    "build that compiles nothing the linear build had not is a "
                    "byte-equality measurement made by the compile path itself"),
        "what_this_licenses": (
            "on every fixture this gate builds, the two arms a nonlinear run selects "
            "on this seam compile the SAME source text the two linear arms compile, "
            "and the weld lifts those same two texts. That is why the widening is a "
            "predicate question and not a third kernel"),
    }


def _library_of(plan: Any) -> Any:
    """The compiled library one plan's default variant came from.

    ``device.compile_source`` caches on the SOURCE STRING, so two plans built from
    identical text hold functions off ONE library object. Reading the function's owner
    is therefore a byte-equality witness that costs no second compile.
    """
    function = plan._functions.get(shaders.CONTRACT_OFF)  # noqa: SLF001
    return getattr(function, "__self__", function)


def leg_no_second_product() -> Dict[str, Any]:
    """EXACTLY ONE registered ``update_H`` weld admits the nonlinear fixture.

    THE REASON THIS CELL IS A WIDENING AND NOT A NEW PRODUCT, measured. Two fused
    products admitting one slot is an over-covering composition; the arm table's own
    ``_select_slot`` leaves such a slot UNSELECTED naming both. So the honest way to
    say "a second H->D product here would be a defect" is to ask every registered weld
    on this slot and require exactly one to answer yes.
    """
    from meep_gpu.metal_kernels import arms  # noqa: PLC0415

    rows: Dict[str, Any] = {}
    for name, keywords in CASES:
        driver = build_driver(keywords, 515)
        residency = Residency()
        admitting: List[str] = []
        for spec in arms.registered("update_H"):
            if spec.replaces == (spec.slot,):
                continue  # a single-slot arm, not a weld
            context = arms.StepContext(driver.fields, driver.pml, residency,
                                       (shaders.CONTRACT_OFF,), sources=())
            try:
                verdict = spec.coverage(context, "update_H")
            except Exception as exc:  # noqa: BLE001 - a raise IS a refusal
                rows.setdefault("raised", {})[spec.family] = str(exc)[:200]
                continue
            if verdict.covered:
                admitting.append(spec.family)
        rows[name] = {"welds_on_update_H_that_admit_this_fixture": sorted(admitting)}
        log(f"    no_second_product {name}: {sorted(admitting)}")
    return {
        "passed": all(row.get("welds_on_update_H_that_admit_this_fixture")
                      == [family.FAMILY]
                      for name, row in rows.items() if name != "raised"),
        "cases": rows,
        "what_this_settles": (
            f"exactly one registered update_H weld -- {family.FAMILY} -- admits every "
            f"nonlinear fixture this gate builds. A NEW product for this cell would "
            f"be a second admitter on one slot, which arms._select_slot leaves "
            f"UNSELECTED naming both; the cell is a widening of a shipped predicate "
            f"and this is the measurement that says so"),
    }


def leg_refusal() -> Dict[str, Any]:
    """The widening is admitted, and the inverted clause is driven in both directions."""
    rows: Dict[str, Any] = {}

    def record(name: str, covered: bool, reasons: Sequence[str], expect: bool,
               needs: Sequence[str] = ()) -> None:
        joined = " | ".join(reasons)
        rows[name] = {"covered": covered, "expected": expect,
                      "reasons": list(reasons)[:6],
                      "ok": covered == expect
                      and all(phrase in joined for phrase in needs)}

    driver = build_driver(dict(CASES)[MUTATION_CASE], 61)
    verdict = family.metal_fused_hd_pair_coverage(driver.fields, driver.pml, (),
                                                  Residency())
    record("the_nonlinear_fixture_is_admitted", verdict.covered, verdict.reasons, True)

    linear = build_driver(dict(CASES)[MUTATION_CASE], 61)
    linear.fields.set_nonlinear_volumes({}, {})
    verdict = family.metal_fused_hd_pair_coverage(linear.fields, linear.pml, (),
                                                  Residency())
    record("a_linear_grid_is_still_admitted", verdict.covered, verdict.reasons, True)

    # THE SPINE ARMS ARE NONLINEAR-ONLY, and that inversion is what keeps the linear
    # and nonlinear arms from co-admitting one slot.
    spine = nonlinear.nonlinear_run_pml_curl_coverage(linear.fields, linear.pml,
                                                      "step_D", Residency())
    record("the_spine_curl_refuses_a_linear_grid", spine.covered, spine.reasons,
           False, ["no chi2/chi3 is installed"])
    spine = nonlinear.nonlinear_run_constitutive_coverage(linear.fields, linear.pml,
                                                          "H", Residency())
    record("the_spine_constitutive_refuses_a_linear_grid", spine.covered,
           spine.reasons, False, ["no chi2/chi3 is installed"])

    # AND THE ORDINARY ARMS REFUSE THE NONLINEAR GRID, which is the other half.
    from meep_gpu.metal_kernels.coverage import (  # noqa: PLC0415
        constitutive_coverage, pml_curl_coverage,
    )

    ordinary = constitutive_coverage(driver.fields, driver.pml, "H", Residency())
    record("the_ordinary_constitutive_refuses_the_nonlinear_grid", ordinary.covered,
           ordinary.reasons, False)
    ordinary = pml_curl_coverage(driver.fields, driver.pml, "step_D", Residency())
    record("the_ordinary_curl_refuses_the_nonlinear_grid", ordinary.covered,
           ordinary.reasons, False)

    # THE SEAM'S ONE PASS, unchanged by the widening.
    verdict = family.metal_fused_hd_pair_coverage(driver.fields, driver.pml, None,
                                                  Residency())
    record("an_undeclared_source_set_is_refused", verdict.covered, verdict.reasons,
           False, ["the source set was not declared"])
    return {"passed": all(row["ok"] for row in rows.values()), "cases": rows}


def leg_arbitration() -> Dict[str, Any]:
    """The composer, asked, on the nonlinear fixtures.

    THIS IS THE CELL ``fused_hd_pair.INSTALLABLE_REASON`` NAMES AS THE ONE GENUINE
    GAIN, so what it selects is RECORDED rather than asserted: the substantive claims
    are that this product is installed on nothing today, that every slot is still
    filled, and that the installed-pair count is what the gain arithmetic reads.
    Flipping the flag is a composition decision with its own release and NOTHING here
    licenses it.
    """
    rows: Dict[str, Any] = {}
    for name, keywords in CASES:
        driver = build_driver(keywords, 717)
        plan = metal_launch.plan_step(driver.fields, driver.pml,
                                      residency=Residency(), sources=(), fuse=True)
        labels = {slot: str(value) for slot, value in plan.selected.items()}
        # A PAIR IS A LABEL THAT OCCUPIES TWO ADJACENT SLOTS. Counted off the table
        # rather than by name, so a renamed product does not change the arithmetic.
        pairs = sum(1 for a, b in (("step_B", "update_H"), ("step_D", "update_E"))
                    if labels.get(a) and labels.get(a) == labels.get(b))
        rows[name] = {
            "selected": labels,
            "installed_pairs": pairs,
            "launches_per_step_over_the_four_slots": 4 - pairs,
            "this_product_installed": any("fused H/D" in value
                                          for value in labels.values()),
            "every_slot_is_still_filled": sorted(labels) == sorted(
                ["step_B", "update_H", "step_D", "update_E"]),
            "installing_this_product_would_give": (
                "a GAIN of one launch and one seam" if pairs == 0 else
                "a TIE" if pairs == 1 else "a LOSS"),
            "reasons_naming_this_product": [
                reason for key, value in plan.reasons.items()
                if family.FAMILY in key for reason in
                (value if isinstance(value, (list, tuple)) else [value])][:2],
        }
        log(f"    arbitration {name}: pairs={pairs} "
            f"({rows[name]['installing_this_product_would_give']}) "
            f"selected={labels}")
    return {
        "passed": all(not row["this_product_installed"]
                      and row["every_slot_is_still_filled"]
                      for row in rows.values()),
        "cases": rows,
        "installable": family.INSTALLABLE,
        "absorb_row": metal_launch.FUSED_PAIR_ARMS.get(family.FAMILY),
        "extra_arms_row": getattr(metal_launch, "FUSED_PAIR_EXTRA_ARMS",
                                  {}).get(family.FAMILY),
        "what_this_does_not_license": (
            "flipping INSTALLABLE. A gain is a reason to CONSIDER installing; "
            "installing is a composition decision with its own release, and this leg "
            "records the arithmetic rather than acting on it"),
    }


def leg_fixture_shape() -> Dict[str, Any]:
    """The fixtures carry the matrix's own chi2/chi3 and the corpus cell's 1-D shape."""
    rows: Dict[str, Any] = {}
    for name, keywords in CASES:
        driver = build_driver(keywords, 99)
        rows[name] = {
            "grid_shape": list(driver.grid.shape),
            "dimensions": int(getattr(driver.grid, "dimensions", 3)),
            "chi2": CHI2, "chi3": CHI3,
            "has_nonlinearity": bool(getattr(driver.fields, "has_nonlinearity",
                                             False)),
            "scalar_chi_arm": bool(keywords.get("scalar", False)),
            "components": list(keywords.get("components", NONLINEAR_COMPONENTS)),
        }
    one_d = [name for name, row in rows.items() if row["grid_shape"][:2] == [1, 1]]
    return {
        "passed": bool(rows
                       and all(row["has_nonlinearity"] for row in rows.values())
                       and one_d
                       and any(row["scalar_chi_arm"] for row in rows.values())),
        "cases": rows,
        "one_dimensional_fixtures": sorted(one_d),
        "chi_values_are_the_composition_matrixs": {"chi2": CHI2, "chi3": CHI3},
        "what_this_says": (
            "every fixture carries a live instantaneous nonlinearity at the "
            "composition matrix's own chi2 = 0.1 / chi3 = 0.2, both coefficient arms "
            "are exercised, and at least one fixture has the corpus cell's own 1-D "
            "shape -- which is where a weld's index decode and ghost rule are least "
            "like the 3-D case it was written on"),
    }


def leg_policy() -> Dict[str, Any]:
    report = subnormal.mps_policy_report()
    driver = build_driver(dict(CASES)[MUTATION_CASE], 11)
    verdict = family.metal_fused_hd_pair_coverage(driver.fields, driver.pml, (),
                                                  Residency())
    return {
        "passed": True,
        "requested_policy": os.environ.get("MEEP_GPU_SUBNORMAL_POLICY"),
        "mps_policy_report": report,
        "predicate_admits_the_fixture_under_this_policy": bool(verdict.covered),
        "predicate_reasons": list(verdict.reasons)[:6],
        "what_this_records": (
            "the policy this run resolved and whether the shipped predicate admits "
            "the nonlinear fixture under it. A resolved `keep` refuses by name "
            "because the MPS executor has no lever for the float32 flush"),
    }


# ---------------------------------------------------------------------------
# Device legs
# ---------------------------------------------------------------------------

def leg_product(steps: int) -> Dict[str, Any]:
    rows: Dict[str, Any] = {}
    for index, (name, keywords) in enumerate(CASES, start=1):
        started = time.perf_counter()
        driver = build_driver(keywords, 1000 + index)
        rows[name] = run_product(driver, steps)
        log(f"    product {index}/{len(CASES)} {name}: passed={rows[name]['passed']} "
            f"words={rows[name]['words_compared']} steps={rows[name]['steps_compared']} "
            f"({time.perf_counter() - started:.1f} s)")
    return {
        "passed": all(row["passed"] for row in rows.values()),
        "cases": len(rows),
        "words_compared": sum(row["words_compared"] for row in rows.values()),
        "complete_driver_steps": sum(row["steps_compared"] for row in rows.values()),
        "references": list(MODES),
        "rows": rows,
    }


def leg_host_defect(steps: int, name: str,
                    launcher: Callable[[Any, Residency, Any], Any], why: str,
                    cases: Sequence[str]) -> Dict[str, Any]:
    rows: Dict[str, Any] = {}
    for index, case in enumerate(cases, start=1):
        driver = build_driver(dict(CASES)[case], 2000 + index)
        result = run_product(driver, steps, launcher=launcher)
        legs = result["arrangements"]
        rows[case] = {"diverged": not legs["weld"]["identical"],
                      "first_divergence": legs["weld"]["first_divergence"],
                      "differing_words_final": legs["weld"]["differing_words_final"]}
        log(f"    {name} {case}: diverged={rows[case]['diverged']} at "
            f"{rows[case]['first_divergence']}")
    return {"passed": all(row["diverged"] for row in rows.values()),
            "defect": name, "why_it_must_fire": why, "cases": rows}


def leg_launch_structure(steps: int) -> Dict[str, Any]:
    driver = build_driver(dict(CASES)[MUTATION_CASE], 4747)
    arrangements = all_arrangements(driver)
    result = drive(driver, arrangements, steps, stop_on_divergence=False)
    rows: Dict[str, Any] = {}
    for name, arrangement in arrangements.items():
        shim = arrangement.shim
        counts = arrangement.launches()
        seam = 0
        if shim is not None:
            for slot in family.REPLACES:
                plan = shim.plans.get(slot)
                if plan is not None and plan is not ABSORBED:
                    seam += int(getattr(declaring(plan), "launches_per_run", 1))
        rows[name] = {
            "launches_per_step_whole_step_plans":
                counts["plans"] / max(result["steps_stepped"], 1),
            "launches_per_step_whole_step_functions":
                counts["functions"] / max(result["steps_stepped"], 1),
            "launches_per_step_at_the_seam": seam,
            "selected": arrangement.selected,
        }
    weld = rows["weld"]
    singles = rows["singles"]
    today = rows["composition_today"]
    saving = (today["launches_per_step_whole_step_plans"]
              - weld["launches_per_step_whole_step_plans"])
    return {
        "passed": bool(weld["launches_per_step_at_the_seam"] == 1
                       and singles["launches_per_step_at_the_seam"] == 2
                       and all(row["launches_per_step_whole_step_plans"]
                               == row["launches_per_step_whole_step_functions"]
                               for row in rows.values())
                       and result["step_error"] is None),
        "arrangements": rows,
        "whole_step_reduction_against_the_composition_installed_today": saving,
        "what_this_says": (
            f"on the NONLINEAR cell the weld saves one launch at the seam against the "
            f"two singles and {saving:+g} over the whole step against the composition "
            f"the composer installs today -- which is the cell "
            f"fused_hd_pair.INSTALLABLE_REASON names as the one place a H->D product "
            f"is a gain. LAUNCH COUNTS ARE NOT TIME and no timing exists for this "
            f"shape"),
    }


def leg_mutation(steps: int, codes: Sequence[int],
                 periodic_codes: Sequence[int]) -> Dict[str, Any]:
    """The SHARED gate's armed shader defects, driven on a NONLINEAR fixture.

    The kernel is the shared gate's, so its mutation table is imported rather than
    re-spelled: this leg widens WHERE those defects are driven, not WHICH.
    """
    caught: Dict[str, Any] = {}

    def run_one(case: str, seed: int, source: str) -> Dict[str, Any]:
        driver = build_driver(dict(CASES)[case], seed)
        function = compile_source(source).fused_hd_pair_step
        result = run_product(driver, steps,
                             functions={shaders.CONTRACT_OFF: function})
        legs = result["arrangements"]
        return {"diverged": not legs["weld"]["identical"],
                "first_divergence": legs["weld"]["first_divergence"],
                "differing_words_final": legs["weld"]["differing_words_final"]}

    armed: List[Tuple[str, str, str, str]] = []
    for name, (source, why) in sorted(_shared.shader_mutations(codes).items()):
        armed.append((name, source, why, MUTATION_CASE))
    for name, (source, why) in sorted(
            _shared.periodic_wrap_mutations(periodic_codes).items()):
        armed.append((name, source, why, PERIODIC_MUTATION_CASE))
    for index, (name, source, why, case) in enumerate(armed, start=1):
        row = run_one(case, 6000 + index, source)
        row["why_it_must_fire"] = why
        row["case"] = case
        caught[name] = row
        log(f"    mutation {index}/{len(armed)} {name} on {case}: "
            f"diverged={row['diverged']} at {row['first_divergence']}")

    predicted: Dict[str, Any] = {}
    for name, (source, why, control) in sorted(
            _shared.inert_mutations(codes).items()):
        row = run_one(MUTATION_CASE, 7000 + len(predicted), source)
        row["why_it_cannot_fire"] = why
        row["earned_by_control"] = control
        row["control_caught"] = bool(caught.get(control, {}).get("diverged"))
        predicted[name] = row
        log(f"    predicted-null {name}: diverged={row['diverged']} "
            f"(control {control} caught={row['control_caught']})")

    host: Dict[str, Any] = {}
    for name, (launcher, why) in HOST_MUTATIONS.items():
        driver = build_driver(dict(CASES)[MUTATION_CASE], 8000 + len(host))
        result = run_product(driver, steps, launcher=launcher)
        legs = result["arrangements"]
        host[name] = {"diverged": not legs["weld"]["identical"],
                      "first_divergence": legs["weld"]["first_divergence"],
                      "why_it_must_fire": why}
        log(f"    host mutation {name}: diverged={host[name]['diverged']}")
    return {
        "passed": bool(caught and all(row["diverged"] for row in caught.values())
                       and all(not row["diverged"] and row["control_caught"]
                               for row in predicted.values())
                       and all(row["diverged"] for row in host.values())),
        "armed": len(caught),
        "caught": sum(1 for row in caught.values() if row["diverged"]),
        "shader_mutations": caught,
        "predicted_null": predicted,
        "host_mutations": host,
        "mutation_table": "gate_metal_fused_hd_pair.shader_mutations, imported",
    }


def leg_byte_neutral(steps: int, codes: Sequence[int]) -> Dict[str, Any]:
    driver = build_driver(dict(CASES)[MUTATION_CASE], 9191)
    function = compile_source(
        _shared.byte_neutral_source(codes)).fused_hd_pair_step
    result = run_product(driver, steps, functions={shaders.CONTRACT_OFF: function})
    legs = result["arrangements"]
    return {"passed": bool(legs["weld"]["identical"] and result["passed"]),
            "identical": legs["weld"]["identical"],
            "words_compared": result["words_compared"],
            "what_this_shows": ("the harness does not diverge on everything: an edit "
                                "that changes no arithmetic changes no word")}


def leg_disarm(steps: int) -> Dict[str, Any]:
    driver = build_driver(dict(CASES)[MUTATION_CASE], 9292)
    result = run_product(driver, steps)
    legs = result["arrangements"]
    return {"passed": bool(legs["weld"]["identical"] and result["passed"]),
            "identical": legs["weld"]["identical"],
            "words_compared": result["words_compared"],
            "what_this_shows": ("every mutation reported caught above was caught "
                                "because of the mutation")}


HOST_MUTATIONS: Dict[str, Tuple[Callable[[Any, Residency, Any], Any], str]] = {
    "H_and_f_w_H_written_in_place": (
        lambda plan, _residency, _pml: AliasedLaunch(plan, ("Hx", "Hy", "Hz",
                                                            "f_w_Hx", "f_w_Hy",
                                                            "f_w_Hz")),
        "the scratch bound to the live buffer: every foreign recompute then reads a "
        "cell another thread may already have stored"),
    "rotation_skipped": (
        lambda plan, _residency, _pml: RotationSkipped(plan),
        "the engine keeps naming the PRE-launch buffers, so the constitutive half's "
        "output is invisible under the engine's own name"),
}


# ---------------------------------------------------------------------------
# The census battery hook and the lift leg
# ---------------------------------------------------------------------------

def runtime_reasons() -> List[str]:
    return _shared.runtime_reasons()


def evaluate(driver: Any, probe: Any) -> Dict[str, Any]:  # noqa: ARG001
    """The census driver's battery hook: the identity on ONE lifted nonlinear row."""
    steps = int(os.environ.get("MEEP_GPU_NL_HD_GATE_STEPS", 12))
    max_cells = os.environ.get("MEEP_GPU_NL_HD_GATE_MAX_CELLS")
    sources = tuple(getattr(driver, "_sources", ()) or ())
    block: Dict[str, Any] = {
        "family": family.FAMILY, "widening": "nonlinear", "steps_requested": steps,
        "n_sources": len(sources),
        "sources": [{"type": type(s).__name__,
                     "field_type": str(getattr(s, "field_type", "")),
                     "is_integrated": bool(getattr(s, "is_integrated", False)),
                     "withdraw_does_work": bool(
                         withdraw_hoist._withdraw_does_work(s))}  # noqa: SLF001
                    for s in sources]}
    pin_array_path(driver)
    verdict = family.metal_fused_hd_pair_coverage(driver.fields, driver.pml, sources,
                                                  Residency())
    block["predicate_admits"] = bool(verdict.covered)
    block["predicate_reasons"] = list(verdict.reasons)
    block["has_nonlinearity"] = bool(getattr(driver.fields, "has_nonlinearity", False))
    composed, _residency = _shared._composed(driver, fuse=True)  # noqa: SLF001
    block["composer_selected"] = dict(composed.selected)
    cells = int(np.prod(driver.grid.shape))
    block["grid_cells"] = cells
    if not verdict.covered:
        block["driven"] = False
        block["why_not_driven"] = "the predicate refused this row"
        return {BLOCK: block}
    if max_cells and cells > int(max_cells):
        block["driven"] = False
        block["why_not_driven"] = f"{cells} cells exceeds the cap"
        return {BLOCK: block}
    determinism = _shared.waveform_determinism(driver, steps)
    block["waveform_determinism"] = determinism
    if not determinism["deterministic"]:
        block["driven"] = False
        block["undefined_by_construction"] = True
        block["why_not_driven"] = determinism["reason"]
        return {BLOCK: block}
    block["driven"] = True
    block.update(run_product(driver, steps, progress=_shared._child_progress,  # noqa: SLF001
                             movement_floor="seam", require_full_budget=False,
                             clean_floor=1))
    return {BLOCK: block}


def _load_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def lift_basis(results: Path) -> Tuple[List[dict], Dict[str, Any]]:
    census = results / CENSUS
    seam = results / SEAM_RECORD
    if not census.is_dir() or not seam.is_dir():
        raise SystemExit(f"the lift leg needs {census} and {seam}")
    seam_rows = {row["label"]: row for row in _load_jsonl(seam / "h_to_d_seam.jsonl")}
    rows: List[dict] = []
    for leg in ("examples", "tests", "tests_param_matched"):
        path = census / f"{leg}.jsonl"
        if not path.is_file():
            continue
        for record in _load_jsonl(path):
            selected = ((record.get("plan_step") or {}).get("selected") or {})
            if (selected.get("update_H"), selected.get("step_D")) != CELL_ARMS:
                continue
            label = f"{record.get('leg', leg)}:{record['row']}"
            seam_row = seam_rows.get(label, {})
            rows.append({
                "label": label, "leg": record.get("leg", leg), "row": record["row"],
                # See the sibling gates: a shim-expanded module must be asked for by
                # the shim's own generated case name, which the census records.
                "case": record.get("case") or record["row"],
                "module": record.get("module") or seam_row.get("module"),
                "interpreter": (record.get("interpreter") or record.get("python")
                                or seam_row.get("interpreter") or sys.executable),
                "grid_cells": record.get("grid_cells"),
                "grid_shape": record.get("grid_shape"),
                "withdraw_in_seam": bool((seam_row.get("h_to_d_seam") or {})
                                         .get("withdraw_in_seam")),
                "census_selected": selected,
            })
    unique = {row["label"]: row for row in rows}
    facts = {"census": CENSUS, "seam_record": SEAM_RECORD,
             "cell_arms": list(CELL_ARMS), "rows_in_cell": len(unique),
             "rows_with_a_standing_withdraw": sorted(
                 label for label, row in unique.items() if row["withdraw_in_seam"])}
    return list(unique.values()), facts


def leg_lift(out_dir: Path, steps: int, max_cells: Optional[int], timeout: float,
             resume: bool, only: Optional[Sequence[str]]) -> Dict[str, Any]:
    """Both corpus rows of the nonlinear cell, re-lifted and driven."""
    import contextlib  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    import gate_provenance  # noqa: PLC0415
    import measure_predicate_coverage as census  # noqa: PLC0415

    out_dir = Path(out_dir).resolve()
    rows, facts = lift_basis(HERE / "results")
    if only:
        rows = [row for row in rows if row["label"] in set(only)]
    lift_dir = out_dir / "lift"
    per_row = lift_dir / "per_row"
    per_row.mkdir(parents=True, exist_ok=True)
    progress_log = lift_dir / "steps.progress.log"
    examples_dir = Path(census.EXAMPLES_DIR)
    tests_dir = Path(census.TESTS_DIR)
    if not examples_dir.is_dir() or not tests_dir.is_dir():
        return {"passed": False, "facts": facts,
                "reason": f"the MEEP corpus is not at {examples_dir} / {tests_dir}"}
    probe_path = API_ROOT / census.PROBE
    environment = dict(os.environ)
    environment.update({"KMP_DUPLICATE_LIB_OK": "TRUE", "MPLBACKEND": "Agg",
                        "PYTHONPATH": str(API_ROOT),
                        "MEEP_GPU_NL_HD_GATE_STEPS": str(steps),
                        "MEEP_GPU_NL_HD_GATE_PROGRESS": str(progress_log)})
    if max_cells is not None:
        environment["MEEP_GPU_NL_HD_GATE_MAX_CELLS"] = str(max_cells)
    work = lift_dir / "workdir"
    work.mkdir(exist_ok=True)
    for entry in examples_dir.iterdir():
        if entry.suffix in (".py", ".ipynb"):
            continue
        link = work / entry.name
        if not link.exists():
            with contextlib.suppress(OSError):
                link.symlink_to(entry)
    shim_path = Path(census.__file__).resolve().parent / "shim"
    needs_shim = set()
    for row in rows:
        module_name = row.get("module")
        if row["leg"] == "examples" or not module_name:
            continue
        module_file = tests_dir / module_name
        if module_file.is_file() and "import parameterized" in module_file.read_text(
                encoding="utf-8", errors="replace"):
            needs_shim.add(module_name)

    measured: List[dict] = []
    for index, row in enumerate(rows, start=1):
        record_path = per_row / (
            row["label"].replace(":", "__").replace(".", "_") + ".json")
        environment["PYTHONPATH"] = (f"{shim_path}{os.pathsep}{API_ROOT}"
                                     if row.get("module") in needs_shim
                                     else str(API_ROOT))
        started = time.time()
        if not (resume and record_path.exists()):
            if row["leg"] == "examples":
                command = [row["interpreter"], "-u",
                           str(Path(census.__file__).resolve()),
                           "--leg", "examples", "--child-script",
                           str(examples_dir / row["row"]),
                           "--out-json", str(record_path), "--probe", str(probe_path),
                           "--progress-log", str(progress_log),
                           "--battery", Path(__file__).stem]
            elif not row["module"]:
                measured.append({**row, "measured": False,
                                 "note": "no module recorded"})
                continue
            else:
                command = [row["interpreter"], "-u",
                           str(Path(census.__file__).resolve()),
                           "--leg", "tests", "--child-module",
                           str(tests_dir / row["module"]),
                           "--child-cases", json.dumps([row.get("case")
                                                        or row["row"]]),
                           "--out-json", str(record_path), "--probe", str(probe_path),
                           "--progress-log", str(progress_log),
                           "--battery", Path(__file__).stem]
            log(f"lift {index}/{len(rows)} {row['label']} start "
                f"(cells {row.get('grid_cells')})")
            stderr_text = ""
            note = "child died"
            try:
                completed = subprocess.run(command, cwd=str(work), env=environment,
                                           timeout=timeout,
                                           stdout=subprocess.DEVNULL,
                                           stderr=subprocess.PIPE, check=False)
                stderr_text = (completed.stderr or b"").decode("utf-8", "replace")
            except subprocess.TimeoutExpired as expired:
                note = "child timeout"
                stderr_text = ((expired.stderr or b"").decode("utf-8", "replace")
                               if expired.stderr else "")
            if not record_path.exists():
                died = {"row": row["row"], "measured": False, "note": note,
                        "stderr_tail": stderr_text[-1500:]}
                gate_provenance.stamp(died)
                record_path.write_text(json.dumps(died, default=str),
                                       encoding="utf-8")
        payload = json.loads(record_path.read_text(encoding="utf-8"))
        candidates = payload if isinstance(payload, list) else [payload]
        wanted = {row["row"], row.get("case") or row["row"]}
        record = next((r for r in candidates
                       if r.get("row") in wanted or r.get("case") in wanted),
                      candidates[0] if candidates else {})
        record.update({key: value for key, value in row.items() if key != "label"})
        record["label"] = row["label"]
        measured.append(record)
        block = record.get(BLOCK) or {}
        log(f"lift {index}/{len(rows)} {row['label']}: "
            f"measured={record.get('measured')} "
            f"admits={block.get('predicate_admits')} driven={block.get('driven')} "
            f"passed={block.get('passed')} steps={block.get('steps_compared')} "
            f"words={block.get('words_compared')} ({time.time() - started:.1f} s)")
    with (lift_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for record in measured:
            handle.write(json.dumps(record, default=str) + "\n")

    blocks = {r["label"]: (r.get(BLOCK) or {}) for r in measured}
    admitted = sorted(label for label, b in blocks.items()
                      if b.get("predicate_admits"))
    refused = sorted(label for label, b in blocks.items()
                     if b.get("predicate_admits") is False)
    unmeasured = sorted(label for label, b in blocks.items()
                        if "predicate_admits" not in b)
    driven = sorted(label for label in admitted if blocks[label].get("driven"))
    passed_rows = sorted(label for label in driven if blocks[label].get("passed"))
    expected_refused = sorted(facts["rows_with_a_standing_withdraw"])
    if only:
        expected_refused = sorted(set(expected_refused) & set(blocks))
    return {
        "passed": bool(blocks and not unmeasured and refused == expected_refused
                       and driven == admitted and passed_rows == driven
                       and all(blocks[label].get("steps_compared", 0)
                               >= LIFT_CLEAN_STEP_FLOOR for label in driven)),
        "facts": facts,
        "admitted": len(admitted),
        "refused": len(refused),
        "refused_labels": refused,
        "unmeasured": unmeasured,
        "driven": len(driven),
        "driven_and_identical": len(passed_rows),
        "clean_step_floor": LIFT_CLEAN_STEP_FLOOR,
        "steps_compared_per_row": {label: blocks[label].get("steps_compared")
                                   for label in driven},
        "complete_driver_steps_compared": sum(
            int(blocks[label].get("steps_compared") or 0) for label in driven),
        "words_compared": sum(int(blocks[label].get("words_compared") or 0)
                              for label in driven),
        "rows": measured,
    }


#: The coverage function `leg_policy_refusal` asks. Named once so the leg and the
#: rest of the gate cannot ask different predicates.
COVERAGE = family.metal_fused_hd_pair_coverage


def leg_policy_refusal() -> Dict[str, Any]:
    """UNDER A POLICY THE EXECUTOR CANNOT HONOUR, every fixture is refused BY NAME.

    This is the ``keep`` leg's whole measurement, and it is a measurement rather than a
    skip. MPS flushes float32 denormals natively and exposes no lever -- both denormal
    pragma spellings are compile errors -- so a resolved ``keep`` is a policy this
    executor cannot deliver. The shipped predicate does not shrug at that: it refuses,
    by name, on BOTH halves, and a run that quietly stepped anyway would be claiming
    ``keep`` arithmetic while the device flushed.

    So the campaign runs this gate under both policies from policy-separated artifact
    directories, and under ``keep`` the artifact records THIS: the refusal, its reason,
    and the fact that no byte claim was made. ``release.released`` is False for the
    keep leg and that is correct -- there is nothing to release.
    """
    report = subnormal.mps_policy_report()
    rows: Dict[str, Any] = {}
    for name, keywords in CASES:
        driver = build_driver(keywords, 21)
        verdict = COVERAGE(driver.fields, driver.pml, (), Residency())
        joined = " | ".join(verdict.reasons)
        rows[name] = {
            "covered": bool(verdict.covered),
            "refused_naming_the_policy": ("subnormal policy" in joined
                                          and "cannot honour it" in joined),
            "both_halves_refused": ("constitutive half:" in joined
                                    and "curl half:" in joined),
            "reasons": list(verdict.reasons)[:3],
        }
    return {
        "passed": bool(rows and not report.get("admitted")
                       and all(not row["covered"]
                               and row["refused_naming_the_policy"]
                               and row["both_halves_refused"]
                               for row in rows.values())),
        "mps_policy_report": report,
        "cases": rows,
        "what_this_records": (
            "under a resolved policy the MPS executor cannot honour, every fixture is "
            "refused BY NAME on both halves and no byte claim is made. This leg is "
            "the keep campaign's whole content; the flush campaign carries the "
            "identity"),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--legs", default="all")
    parser.add_argument("--lift-steps", type=int, default=12)
    parser.add_argument("--lift-max-cells", type=int, default=None)
    parser.add_argument("--lift-timeout", type=float, default=1800.0)
    parser.add_argument("--lift-resume", action="store_true")
    parser.add_argument("--lift-only", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.legs == "all":
        legs = ALL_LEGS
    elif args.legs in LEG_GROUPS:
        legs = LEG_GROUPS[args.legs]
    else:
        legs = tuple(name.strip() for name in args.legs.split(",") if name.strip())
    unknown = sorted(set(legs) - set(ALL_LEGS))
    if unknown:
        raise SystemExit(f"unknown legs {unknown}; known legs are {list(ALL_LEGS)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    jsonl = args.out.with_suffix(".jsonl")
    jsonl.write_text("", encoding="utf-8")
    started = time.perf_counter()
    rows: List[Dict[str, Any]] = []

    def record(leg: str, payload: Dict[str, Any]) -> None:
        row = {"leg": leg, **payload}
        rows.append(row)
        with jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        log(f"LEG {leg}: {'PASS' if row.get('passed') else 'FAIL'} "
            f"({time.perf_counter() - started:.1f}s)")

    # A POLICY THIS EXECUTOR CANNOT HONOUR IS MEASURED, NOT SKIPPED. See
    # `leg_policy_refusal`: under a resolved `keep` the shipped predicate refuses by
    # name on both halves, so the byte legs have nothing to drive and running them
    # would either crash on a refused plan or -- worse -- report a green harness that
    # measured nothing. The gate records the refusal and exits INCOMPLETE, so
    # `release.released` stays False and nothing can mint this artifact.
    if not subnormal.mps_policy_report().get("admitted"):
        record("policy", leg_policy())
        record("policy_refusal", leg_policy_refusal())
        # EMPTIED, so the dispatch below records nothing twice. `complete` still reads
        # ALL_LEGS, so the artifact reports every leg this run did NOT take.
        legs = ()

    codes = MUTATION_CODES
    log(f"mutation specialisation {MUTATION_CASE} -> {codes}; "
        f"periodic {PERIODIC_MUTATION_CASE} -> {PERIODIC_CODES}")

    if "text_identity" in legs:
        record("text_identity", leg_text_identity())
    if "no_second_product" in legs:
        record("no_second_product", leg_no_second_product())
    if "refusal" in legs:
        record("refusal", leg_refusal())
    if "arbitration" in legs:
        record("arbitration", leg_arbitration())
    if "fixture_shape" in legs:
        record("fixture_shape", leg_fixture_shape())
    if "policy" in legs:
        record("policy", leg_policy())
    if "product" in legs:
        record("product", leg_product(args.steps))
    if "race" in legs:
        launcher, why = HOST_MUTATIONS["H_and_f_w_H_written_in_place"]
        record("race", leg_host_defect(args.steps, "H_and_f_w_H_written_in_place",
                                       launcher, why, [name for name, _ in CASES]))
    if "rotation" in legs:
        launcher, why = HOST_MUTATIONS["rotation_skipped"]
        record("rotation", leg_host_defect(args.steps, "rotation_skipped", launcher,
                                           why, [MUTATION_CASE,
                                                 PERIODIC_MUTATION_CASE,
                                                 "nonlinear_1d_like_the_corpus"]))
    if "launch_structure" in legs:
        record("launch_structure", leg_launch_structure(args.steps))
    if "mutation" in legs:
        record("mutation", leg_mutation(args.steps, codes, PERIODIC_CODES))
    if "lift" in legs:
        record("lift", leg_lift(args.out.parent, args.lift_steps,
                                args.lift_max_cells, args.lift_timeout,
                                args.lift_resume,
                                (args.lift_only.split(",") if args.lift_only
                                 else None)))
    if "byte_neutral" in legs:
        record("byte_neutral", leg_byte_neutral(args.steps, codes))
    if "disarm" in legs:
        record("disarm", leg_disarm(args.steps))

    ran = tuple(dict.fromkeys(row["leg"] for row in rows))
    complete = all(leg in ran for leg in ALL_LEGS)
    all_passed = bool(rows) and all(row.get("passed") for row in rows)
    verdict = ("PASS" if all_passed else "FAIL") if complete else (
        "INCOMPLETE: not every leg ran -- "
        f"{'all requested legs passed' if all_passed else 'a requested leg FAILED'}; "
        f"missing {sorted(set(ALL_LEGS) - set(ran))}")
    import torch  # noqa: PLC0415

    import gate_provenance  # noqa: PLC0415

    mutation_row = next((row for row in rows if row["leg"] == "mutation"), {})
    lift_row = next((row for row in rows if row["leg"] == "lift"), {})
    product_row = next((row for row in rows if row["leg"] == "product"), {})
    result = {
        "verdict": verdict,
        "complete": complete,
        "legs_requested": list(legs),
        "legs_run": list(ran),
        "legs_missing": sorted(set(ALL_LEGS) - set(ran)),
        "elapsed_seconds": time.perf_counter() - started,
        "steps": args.steps,
        "rows": rows,
        "counts": {
            "product_cases": product_row.get("cases"),
            "product_words_compared": product_row.get("words_compared"),
            "product_complete_driver_steps": product_row.get(
                "complete_driver_steps"),
            "shader_mutations_armed": mutation_row.get("armed"),
            "shader_mutations_caught": mutation_row.get("caught"),
            "host_mutations": len(HOST_MUTATIONS) if "mutation" in ran else 0,
            "predicted_null": len(mutation_row.get("predicted_null") or {}),
            "lift_rows_in_cell": (lift_row.get("facts") or {}).get("rows_in_cell"),
            "lift_rows_driven_and_identical": lift_row.get("driven_and_identical"),
            "lift_words_compared": lift_row.get("words_compared"),
        },
        "subject": {
            "family": family.FAMILY,
            "module": "meep_gpu/metal_kernels/fused_hd_pair.py",
            "this_gate_certifies_no_new_kernel": True,
            "what_it_adds": ("the nonlinear widening's device measurement -- the leg "
                             "the fused_hd_pair release owed and the board records as "
                             "not credited"),
            "cell_arms": list(CELL_ARMS),
            "installable": family.INSTALLABLE,
            "absorb_row": metal_launch.FUSED_PAIR_ARMS.get(family.FAMILY),
            "extra_arms_row": getattr(metal_launch, "FUSED_PAIR_EXTRA_ARMS",
                                      {}).get(family.FAMILY),
        },
        "signature": {"packed_bindings": family.PACKED_BINDINGS,
                      "ceiling": MAX_BUFFER_BINDINGS},
        "corpus": {"census": CENSUS, "seam_record": SEAM_RECORD,
                   "cell_arms": list(CELL_ARMS)},
        "subnormal_policy": subnormal.mps_policy_report(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "metal_frontend": metal_frontend_version(),
        "jsonl": str(jsonl),
        "what_this_does_not_license": (
            "flipping fused_hd_pair.INSTALLABLE (a gain is a reason to consider "
            "installing; installing is a composition decision with its own release), "
            "any throughput claim (launch counts are not time, no timing exists for "
            "this shape, and another Metal lane shared this GPU during the run), a "
            "NEW product for this cell (leg no_second_product measures that a second "
            "admitter would be a defect), the (ordinary -> PML) cell's own credit "
            "which its own gate carries, or any step past a row's own first banded "
            "step"),
        "source_sha256": {
            "family": hashlib.sha256(Path(family.__file__).read_bytes()).hexdigest(),
            "gate": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    gate_provenance.stamp(result)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8")
    log(f"VERDICT {verdict} in {result['elapsed_seconds']:.2f}s; artifact {args.out}")
    if not all_passed:
        return 1
    return 0 if complete else EXIT_INCOMPLETE


if __name__ == "__main__":
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))
