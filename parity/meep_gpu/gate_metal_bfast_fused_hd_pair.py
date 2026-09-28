#!/usr/bin/env python3
"""Native MPS byte gate for the BFAST Metal H->D weld: the BFAST-run ``update_H``
welded into the BFAST ``step_D``.

THE THIRD ARM PAIR ON THE FOURTH SEAM, and the first one whose CURL half is not a
member of the certified PML curl family at all. The driver runs

    ``update_H`` (driver.py:3311) -> the electric integrated-source withdraw
    (:3313-3314) -> ``step_D`` (:3315)

and ``meep_gpu/metal_kernels/bfast_fused_hd_pair.py`` performs both consults in ONE
dispatch: the certified ``update_H`` computed into launch-local SCRATCH, the curl's own
cell taken from registers, every one of the BFAST curl's six foreign taps RECOMPUTED
from pre-launch state, ``D``/``fu_D``/``f_bfast_D`` stepped in place, and the
``H``/``f_w_H`` bindings rotated afterwards.

THE CLAIM IS PER COMPLETE DRIVER STEP -- ``FdtdDriver.step``'s own consult order -- over
a stated budget, as uint32 WORDS over every stored volume the engine allocates (never
``allclose``: ``-0.0 == 0.0`` lies), against FOUR reference engines from one seed:

  1. the ARRAY PATH -- ``stepping``'s passes under the driver's own loop;
  2. the CERTIFIED SINGLES -- ``bfast_curl.plan_bfast_run_constitutive(..., 'H')`` at
     ``update_H`` and ``bfast_curl.plan_bfast_pml_curl(..., 'step_D')`` at ``step_D``,
     two dispatches on the seam;
  3. the COMPOSITION THE COMPOSER INSTALLS TODAY -- ``plan_step(fuse=True)``, which on
     every row this product reaches puts the released ``bfast_fused_magnetic_pair``
     across ``step_B``/``update_H`` and ``bfast_fused_electric_pair`` across
     ``step_D``/``update_E``. This is the reference the arbitration finding rests on;
  4. the same slots DISPATCHED UNFUSED -- ``plan_step(fuse=False)``.

=============================================================================
WHAT IS BFAST-SPECIFIC HERE, AND WHY EACH LEG EXISTS
=============================================================================

**The pack is what makes this product exist, so it is MEASURED.** Unpacked the
signature is 33 pointers plus one ``Params`` = 34 against a ceiling of 31. Leg
``binding_ceiling`` compiles that shape and the 44-binding separate-scalar one and
requires both to FAIL, bisects the ceiling on this host, compiles the shipped 26 on
every fixture's own codes, and LAUNCHES a packed probe that reads back every ``Params``
field -- the six Tustin scalars included, because a ``float`` written at a ``uint``'s
offset is a plausible wrong number rather than an error.

**The six k scalars are the one thing this family can get wrong silently.** They are
computed on the HOST, rounded to float32 once, and baked into the packed struct; a sign,
an order or a rounding that moved would give a converged, smooth, wrong Tustin filter.
Leg ``tustin_scalars`` re-derives them from ``stepping``'s own term tables through
``bfast_curl_coefficients``, compares the words the plan wrote into ``Params`` against
them as uint32, and requires the D side's negation (``k1, k2 = -k1, -k2``, MEEP's
``if (ft == D_stuff)``) to be present -- with the magnetic side's values as the control
that must DIFFER.

**A k component of zero makes its target's state invisible.** ``bfast_scaled_k =
(0.2, 0, 0)`` leaves ``f_bfast_Dx`` identically zero for the whole run -- measured -- so
a fixture built on it would compare a volume that never moved and report the tail as
certified. Leg ``state_census`` counts, on the array path, how many words of each
``f_bfast_D`` component move, and the fixture set carries a THREE-component k for
exactly this reason.

**The BFAST tail reads no magnetic cell the plain curl does not.** Its operands are the
curl's own registers, so redirecting the nine plain loads redirects the tail with them.
Leg ``transcription`` requires every difference between the emitted source and the
certified BFAST curl to be a redirected magnetic read and requires no ``gN[`` spelling
to survive at all.

=============================================================================
THE FIVE THINGS THIS GATE REFUSES TO LET PASS SILENTLY
=============================================================================

1. **A silent fallback.** Bytes alone cannot prove the fused path ran: an unlaunched
   plan is byte-identical to the oracle BY CONSTRUCTION. Every case asserts the exact
   launch count from TWO independent witnesses -- the plans' own counters and a wrapper
   around every compiled function -- plus ``dispatched['update_H'] == steps`` and
   ``absorbed['step_D'] == steps``, and requires the seam's outputs to have MOVED.
2. **A hollow pass.** Leg ``disarm`` reruns the identical harness with the shipped bytes
   and requires zero divergence; leg ``byte_neutral`` is the complement.
3. **A vacuous claim.** Byte identity is claimed subject to a CHECKED subnormal-free
   precondition, taken on the oracle's own state AND its own intermediates BEFORE each
   comparison; a banded step is stepped, named and NOT compared.
4. **An unmeasured platform assumption.** Leg ``binding_ceiling``, above.
5. **A refusal that is really an omission.** Leg ``refusal`` names each one and drives
   the inverted BFAST clause in BOTH directions: a run with no BFAST shear must be
   refused naming ``fused_hd_pair``, and ``fused_hd_pair`` must refuse the BFAST fixture
   naming this cell.

LEGS
  host
  driver_order      REPLACES is the driver's two adjacent consults, read off the tree
  arm_pairing       the shipped composer's own two labels on this fixture, and both
                    halves' predicates re-asked one at a time
  transcription     the emitted source differs from the certified BFAST curl and the
                    certified constitutive in EXACTLY the declared edits
  refusal           each refusal by name, in both directions on the BFAST clause
  arbitration       the composer, asked, with this product REGISTERED: zero installed
  seed_scale        the 2^80 seed changes no mantissa and clears the budget
  purity            the foreign-tap race ledger
  ghost_observability which ghost defects a specialisation can observe at all
  tustin_scalars    the six k words the plan bakes are stepping's own, D-side negated
  state_census      every f_bfast_D component moves on the fixture set
  policy            the resolved subnormal policy, recorded
  compile
  binding_ceiling   two refuted signatures FAIL, the ceiling is bisected, the shipped
                    26 COMPILES on every fixture's codes, the packed probe LAUNCHES
  pack_bytes        the packed buffer is the separate vectors' words end to end
  mutants_compile   every armed defect and the byte-neutral control COMPILE
  device
  product           the four-reference identity on every fixture, complete driver steps
  race              H and f_w_H bound IN PLACE must diverge, on every fixture
  rotation          the post-launch rotation dropped must diverge
  sync              the product force-installed with flux_in_box driven mid-run
  launch_structure  launches per step at the seam and over the whole step
  withdraw          an integrated electric source in the seam: hoisted and its null
  lift              every corpus row the standing census puts in this cell, driven
  byte_neutral      the one armed edit required NOT to diverge
  mutation          every armed shader and host defect, each of which MUST diverge
  disarm            the identical harness, shipped bytes, must not diverge

THE GENERIC MACHINERY IS IMPORTED FROM ``gate_metal_fused_hd_pair`` RATHER THAN COPIED:
the driver walk, the capture/restore/compare mechanism, the shim that answers the
engine's own dispatch seam, the two launch counters, the rotation settling, the
orphaned-mirror guard and the two censuses are the SAME mechanism this product needs,
and they carry defects already measured and repaired.

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

# THE POLICY, SET BEFORE ANY meep_gpu MODULE IS REACHED. The campaign runs this gate
# under both policies from separate artifact directories; a resolved `keep` is expected
# to refuse every predicate by name, which is what leg `policy` records.
os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")
HERE = Path(__file__).resolve().parent
# BY NAME, never by parents[N]: a moved harness resolving a wrong root measures nothing.
API_ROOT = next(parent for parent in HERE.parents
                if (parent / "meep_gpu" / "metal_kernels").is_dir())
for _path in (str(API_ROOT), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402

# THE SIBLING GATE AS A LIBRARY. See the module docstring.
import gate_metal_fused_hd_pair as _shared  # noqa: E402

from meep_gpu import stepping, withdraw_hoist  # noqa: E402
from meep_gpu.metal_kernels import (  # noqa: E402
    bfast_curl, bfast_fused_hd_pair as family, coefficient_pack,
    fused_hd_pair as plain, launch as metal_launch, shaders, subnormal,
)
from meep_gpu.metal_kernels.device import (  # noqa: E402
    MAX_BUFFER_BINDINGS, Residency, compile_source, metal_frontend_version,
)
from meep_gpu.triton_kernels.bfast_curl import (  # noqa: E402
    BFAST_STATE_NAMES, bfast_curl_coefficients,
)

# The generic machinery, bound to short names so the BFAST legs read as their own.
ABSORBED = _shared.ABSORBED
AliasedLaunch = _shared.AliasedLaunch
Arrangement = _shared.Arrangement
RotationSkipped = _shared.RotationSkipped
Shim = _shared.Shim
SyncedPlan = _shared.SyncedPlan
capture = _shared.capture
compare_snapshots = _shared.compare_snapshots
declaring = _shared.declaring
differing = _shared.differing
drive = _shared.drive
install = _shared.install
in_place_words = _shared.in_place_words
needle = _shared.needle
pin_array_path = _shared.pin_array_path
rebind_static = _shared.rebind_static
restore = _shared.restore
stored_volumes = _shared.stored_volumes
synced = _shared.synced
words = _shared.words

#: ``measure_predicate_coverage`` digests this package into every lifted row's
#: ``subject_manifest_sha256``. Declared, as every battery declares it.
SUBJECT_PACKAGE = "metal_kernels"

#: The budget every synthetic case runs. SIXTY, the sibling H->D gate's, and for its
#: reason: the state this weld carries between steps -- the rotated ``H``/``f_w_H``
#: pair, the in-place ``fu_D`` AND the Tustin state ``f_bfast_D``, which is a
#: MARGINALLY STABLE IIR the array path never resets -- compounds across steps.
STEPS = 60

#: The steps at which the sync leg calls the flux accessor.
SYNC_STEPS: Tuple[int, ...] = (3, 7)

#: The floor a lifted corpus row must reach to count. See the sibling gate's constant.
LIFT_CLEAN_STEP_FLOOR = 8

#: THE SEED IS SCALED BY 2^80 -- the sibling gate's measurement, DRIVEN here rather than
#: inherited, because this fixture carries a volume the sibling's does not: the Tustin
#: state, whose filter is the one a scale could in principle disturb.
SEED_SCALE_BITS = 80

#: THE SHEAR, AND ALL THREE COMPONENTS ARE NONZERO ON PURPOSE. Measured 2026-09-07 with
#: ``bfast_scaled_k = (0.2, 0.0, 0.0)``: ``f_bfast_Dx`` never moved over the whole run,
#: because target 0's two scalars are read off the OTHER two axes and both were zero. A
#: fixture built on that k would compare a state volume that is identically zero and
#: report the Tustin tail as certified. Leg ``state_census`` is the measurement that
#: keeps this constant honest.
BFAST_K: Tuple[float, float, float] = (0.2, -0.13, 0.07)

#: name -> the fixture's keywords. ALL EIGHT BOUNDARY SPECIALISATIONS, because the ghost
#: rule is exactly what specialisation changes and the recompute's guard is what a
#: redirect is most likely to eat, plus one row on a SINGLE-AXIS shear -- the shape the
#: cell's own corpus row carries, and the one on which two of the three targets' scalars
#: are zero.
CASES: Tuple[Tuple[str, Dict[str, Any]], ...] = tuple(
    (f"bc_{'m' if x else 'p'}{'m' if y else 'p'}{'m' if z else 'p'}",
     {"boundaries": {axis: "metallic" for axis, walled in zip("xyz", (x, y, z))
                     if walled},
      "bfast_scaled_k": BFAST_K})
    for x in (0, 1) for y in (0, 1) for z in (0, 1)
) + (
    ("single_axis_shear_mmm",
     {"boundaries": {axis: "metallic" for axis in "xyz"},
      "bfast_scaled_k": (0.2, 0.0, 0.0)}),
)

#: The case the mutations are armed on: every axis walled, every k component live.
MUTATION_CASE = "bc_mmm"

#: THE SECOND MUTATION CASE. On a metallic axis ``stepping._mask_non_owned_cells``
#: zeroes the very curl component the shifted tap feeds, at the very plane where the
#: ghost is served, so no metallic BOUNDARY VALUE is observable through this kernel's
#: output. On an all-periodic grid that mask does not fire and cell 0's curl is a live
#: stepped value assembled from the WRAPPED row.
PERIODIC_MUTATION_CASE = "bc_ppp"

#: The fixture geometry -- ``metal_composition_matrix.cart``'s own constants.
CELL: Tuple[float, float, float] = (2.0, 2.1, 1.2)
RESOLUTION = 10.0
COURANT = 0.35
PML_CELLS = 2
EPSILON: Dict[str, float] = {"Ex": 2.0, "Ey": 2.5, "Ez": 3.0}

#: The reference engines, in the order the record reports them.
MODES: Tuple[str, ...] = ("array", "singles", "composition_today", "unfused",
                          "weld", "weld_seam_only")

LEG_GROUPS: Dict[str, Tuple[str, ...]] = {
    "host": ("driver_order", "arm_pairing", "transcription", "refusal",
             "arbitration", "seed_scale", "purity", "ghost_observability",
             "tustin_scalars", "state_census", "policy"),
    "compile": ("binding_ceiling", "pack_bytes", "mutants_compile"),
    "device": ("product", "race", "rotation", "sync", "launch_structure",
               "withdraw", "lift", "byte_neutral", "mutation", "disarm"),
}
ALL_LEGS: Tuple[str, ...] = tuple(leg for group in LEG_GROUPS.values() for leg in group)

#: The census this gate lifts its corpus rows from, the seam record whose
#: ``withdraw_in_seam`` flag names the rows the predicate must refuse, and the cell.
CENSUS = "metal_coverage_2026-09-04_m0complex"
SEAM_RECORD = "h_to_d_seam_2026-09-04"
CELL_ARMS: Tuple[str, str] = ("BFAST", "BFAST")

#: HOW MANY TIMES ``h_cell(`` APPEARS IN THE SHIPPED SOURCE: one declaration, the
#: thread's own cell, and one per shifted magnetic tap.
H_CELL_CALL_SITES = 1 + 1 + len(plain.HALO_TAPS)

#: The runner's "cannot certify on this host" code.
EXIT_INCOMPLETE = 75


def log(message: str) -> None:
    print(message, flush=True)


# ---------------------------------------------------------------------------
# The fixture
# ---------------------------------------------------------------------------

def codes_of(name: str) -> Tuple[int, ...]:
    """The per-axis PERIODIC/METALLIC triple one case compiles to, off its keywords."""
    walls = dict(CASES)[name].get("boundaries") or {}
    return tuple(1 if walls.get(axis) == "metallic" else 0 for axis in "xyz")


MUTATION_CODES: Tuple[int, ...] = codes_of(MUTATION_CASE)
PERIODIC_CODES: Tuple[int, ...] = codes_of(PERIODIC_MUTATION_CASE)


def build_driver(keywords: Mapping[str, Any], seed: int,
                 sources: Sequence[Mapping[str, Any]] = (),
                 scale_bits: int = SEED_SCALE_BITS) -> Any:
    """One seeded ``FdtdDriver`` on the ``cart`` row's geometry, with a BFAST shear.

    A DRIVER, not a bare ``Fields``: every leg here is a claim about a COMPLETE driver
    step, and a hand-written walk over the live pass list would be a second model of
    what a step is.
    """
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415

    driver = FdtdDriver(cell_size=CELL, resolution=RESOLUTION, courant=COURANT,
                        force_complex_fields=False,
                        boundaries=dict(keywords.get("boundaries") or {}) or None,
                        bfast_scaled_k=tuple(keywords.get("bfast_scaled_k", BFAST_K)),
                        dimensions=3)
    driver.setup_pml(int(keywords.get("pml", PML_CELLS)))
    shape = tuple(driver.grid.shape)
    driver.fields.set_epsilon_volumes(
        {name: np.full(shape, np.float32(value), np.float32)
         for name, value in EPSILON.items()},
        {name: np.full(shape, np.float32(1.0 / value), np.float32)
         for name, value in EPSILON.items()})
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
    """Reference 2: the two certified BFAST singles at the seam's two slots.

    THE CONSTITUTIVE HALF IS THE CERTIFIED ORDINARY BODY UNDER A BFAST PREDICATE --
    ``bfast_curl.plan_bfast_run_constitutive`` builds through ``launch``'s own
    ``ConstitutivePlan``, so "same kernel" is a fact rather than a claim.
    """
    residency = Residency()
    fields, pml = driver.fields, driver.pml
    constitutive = bfast_curl.plan_bfast_run_constitutive(fields, pml, "H", residency)
    curl = bfast_curl.plan_bfast_pml_curl(fields, pml, "step_D", residency)
    if constitutive is None or curl is None:
        raise RuntimeError(
            f"a certified single was refused on a fixture this gate expects it to "
            f"admit (constitutive={constitutive!r}, curl={curl!r})")
    shim = Shim(fields, {"update_H": synced(constitutive, residency),
                         "step_D": synced(curl, residency)})
    return Arrangement("singles", shim, residency,
                       selected={"update_H": "BFAST", "step_D": "BFAST"})


def build_weld(driver: Any, residency: Residency,
               functions: Optional[Mapping[str, Any]] = None,
               sources: Any = ()) -> Any:
    plan = family.plan_metal_bfast_fused_hd_pair(
        driver.fields, driver.pml, sources=sources, residency=residency,
        functions=functions)
    if plan is None:
        reasons = family.metal_bfast_fused_hd_pair_coverage(
            driver.fields, driver.pml, sources, residency).reasons
        raise RuntimeError("the BFAST fused H/D pair was refused: "
                           + "; ".join(reasons))
    return plan


def arrangement_weld(driver: Any, functions: Optional[Mapping[str, Any]] = None,
                     launcher: Optional[Callable[[Any, Residency, Any], Any]] = None,
                     sync_hazard: bool = False, hoist: bool = False,
                     rest_unfused: bool = True, name: str = "weld") -> Arrangement:
    """The SUBJECT: the weld force-installed at ``update_H``, ``step_D`` absorbed.

    FORCE-INSTALLED, because the composer refuses this product by name (no absorb row,
    ``INSTALLABLE = False``) and this gate must measure it anyway.
    """
    residency = Residency()
    sources = tuple(getattr(driver, "_sources", ()))
    plan = build_weld(driver, residency, functions=functions, sources=())
    occupant: Any = (plan if launcher is None
                     else launcher(plan, residency, driver.pml))
    occupant = SyncedPlan(occupant, residency)
    if hoist:
        occupant = withdraw_hoist.LeadingWithdrawPlan(
            occupant, driver.fields, sources, span=family.REPLACES)
    plans: Dict[str, Any] = {}
    if rest_unfused:
        # ON THIS ARRANGEMENT'S OWN RESIDENCY: the weld writes D and fu_D IN PLACE and
        # ``update_E`` reads them; a second registry would give that read a different
        # device buffer.
        base, _residency = _shared._composed(driver, fuse=False,  # noqa: SLF001
                                             residency=residency)
        for slot in metal_launch.STEP_ORDER:
            if slot in base.plans and slot not in family.REPLACES:
                plans[slot] = synced(base.plans[slot], residency)
    plans["update_H"] = occupant
    plans["step_D"] = ABSORBED
    shim = Shim(driver.fields, plans, sync_hazard=sync_hazard)
    return Arrangement(name, shim, residency,
                       selected={"update_H": "BFAST fused H/D pair (forced)",
                                 "step_D": "BFAST fused H/D pair (forced)"})


def all_arrangements(driver: Any, functions: Optional[Mapping[str, Any]] = None,
                     launcher: Optional[Callable[[Any, Residency, Any], Any]] = None,
                     ) -> Dict[str, Arrangement]:
    """The array path, the three references and the subject TWICE, on one driver."""
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
    """The four-reference identity on one driver, per complete step."""
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
    # THE SEAM'S OWN OUTPUTS INCLUDE THE TUSTIN STATE, which the sibling's do not:
    # this weld steps `f_bfast_D` in place, so a run where it never moved would be a
    # run where the tail's own state was never compared.
    seam_outputs = tuple(
        name for name in
        ([f"{stem}{axis}" for stem in ("D", "fu_D", "H", "f_w_H") for axis in "xyz"]
         + list(BFAST_STATE_NAMES["step_D"]))
        if name in result["volumes_compared"])
    seam_moved = {name: int(result["moved_from_seed"].get(name, 0))
                  for name in seam_outputs}
    # A TUSTIN COMPONENT WHOSE TWO SCALARS ARE BOTH EXACTLY ZERO CANNOT MOVE, and
    # that is a fact about the SHEAR rather than about this kernel. Target 0's k1 and
    # k2 are read off the other two axes (stepping.py:911-912), so
    # `bfast_scaled_k = (0.2, 0, 0)` leaves `f_bfast_Dx` identically zero for the
    # whole run -- measured 2026-09-07, and the reason the fixture set carries a
    # three-component shear. The exclusion is DERIVED from the scalars the plan bakes,
    # named in the record, and never widened to a component whose scalars are live.
    invariant = tuple(driver.grid.is_invariant(axis) for axis in range(3))
    scalars = bfast_curl_coefficients(driver.grid.bfast_scaled_k, invariant,
                                      magnetic=False)
    dead_states = tuple(
        name for index, name in enumerate(BFAST_STATE_NAMES["step_D"])
        if scalars[2 * index] == 0.0 and scalars[2 * index + 1] == 0.0)
    if movement_floor == "all":
        # THE FIXTURE'S OWN FLOOR IS THE SEAM'S OUTPUTS, and every one of them must
        # have moved unless its two scalars make movement impossible.
        floor = not [name for name in result["arrays_that_never_moved"]
                     if name in seam_outputs and name not in dead_states]
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
        "require_full_budget": require_full_budget,
        "clean_floor": clean_floor,
        "every_arrangement_launched_and_the_two_counters_agree": launched,
        "weld_launched_once_per_step_and_absorbed_step_D": weld_ok,
        "movement_floor": movement_floor,
        "movement_floor_met": floor,
        "seam_output_volumes": list(seam_outputs),
        "seam_output_words_moved": seam_moved,
        "seam_output_volumes_that_never_moved": sorted(
            name for name, n in seam_moved.items() if not n),
        "tustin_state_moved": {name: seam_moved.get(name, 0)
                               for name in BFAST_STATE_NAMES["step_D"]},
        "tustin_scalars_this_fixture_bakes": list(scalars),
        "tustin_components_whose_two_scalars_are_both_zero": list(dead_states),
        "why_a_dead_tustin_component_is_excluded_from_the_floor": (
            "its k1 and k2 are read off the OTHER two axes (stepping.py:911-912) and "
            "are both exactly 0.0 under this fixture's shear, so its advance is "
            "`0 - 2*0` at every cell of every step: the component cannot move, and a "
            "floor that demanded it would fail for the shear rather than for the "
            "kernel. Derived from the scalars, never from a name"),
        "weld_agrees_with_the_certified_singles":
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
# The flush precondition's second half, EXTENDED to the Tustin tail
# ---------------------------------------------------------------------------

class BfastIntermediateCensus(_shared.IntermediateCensus):
    """The sibling census, plus the BFAST tail -- which it does not see AT ALL.

    WHY THIS SUBCLASS EXISTS, and it is a SILENT gap rather than a declared one.
    ``gate_metal_fused_hd_pair.IntermediateCensus`` models three helpers and COUNTS two
    more, and a step that reaches a counted one reports ``covered`` False. But
    ``stepping._bfast_term`` is in neither list, so on a BFAST run its intermediates
    are neither censused nor counted and the census reports ``covered`` TRUE while
    half of the arithmetic that produced the step went unlooked at -- measured
    2026-09-07 on this gate's own fixtures, where nine cases passed with
    ``calls_this_census_does_not_model: 0`` and the whole Tustin ladder unmodelled.
    A declared omission is a narrower claim; an undeclared one is a wrong one.

    So the tail is modelled here, from ``stepping._bfast_term``'s own sequence
    (S:894-905), and the model is CHECKED against the engine rather than trusted: the
    value the helper RETURNS and the state it leaves behind must both equal this
    replay's, bit for bit, or :func:`_shared._agrees` raises.

    THE MASK IS THE ENGINE'S OWN. ``_mask_non_owned_cells`` is called rather than
    re-implemented, because the advance's mask and the curl's are the same predicate
    and this census must not become a second reading of it.
    """

    MODELLED: Tuple[str, ...] = (_shared.IntermediateCensus.MODELLED
                                 + ("_bfast_term",))

    def __enter__(self) -> "BfastIntermediateCensus":
        super().__enter__()
        from meep_gpu import stepping as _stepping  # noqa: PLC0415

        original = _stepping._bfast_term  # noqa: SLF001
        self._originals["_bfast_term"] = original
        f32 = np.float32
        note = self._note

        def bfast(fields: Any, term: Any, operands: Any, magnetic: bool) -> Any:
            self.calls += 1
            state = getattr(fields, "f_bfast_" + term.target)
            state_in = np.array(state, copy=True)
            first = np.array(operands.first, copy=True)
            shifted_first = np.array(operands.shifted_first, copy=True)
            second = np.array(operands.second, copy=True)
            shifted_second = np.array(operands.shifted_second, copy=True)
            out = original(fields, term, operands, magnetic)
            if state_in.dtype != np.float32:
                return out
            grid = fields.grid
            bfast_k = grid.bfast_scaled_k
            have_p = not grid.is_invariant(term.first_axis)
            have_m = not grid.is_invariant(term.second_axis)
            k1 = bfast_k[_stepping._bfast_axis(term.second)] if have_m else 0.0  # noqa: SLF001
            k2 = bfast_k[_stepping._bfast_axis(term.first)] if have_p else 0.0  # noqa: SLF001
            if not magnetic:
                k1, k2 = -k1, -k2
            sum1 = note("bfast:shifted_first+first",
                        (shifted_first + first).astype(f32))
            sum2 = note("bfast:shifted_second+second",
                        (shifted_second + second).astype(f32))
            term1 = note("bfast:k1*sum1", (f32(k1) * sum1).astype(f32))
            term2 = note("bfast:k2*sum2", (f32(k2) * sum2).astype(f32))
            total = note("bfast:k1*sum1-k2*sum2", (term1 - term2).astype(f32))
            doubled = note("bfast:2*state", (f32(2.0) * state_in).astype(f32))
            advance = note("bfast:total-2*state", (total - doubled).astype(f32))
            # THE ENGINE'S OWN MASK, called rather than re-implemented.
            _stepping._mask_non_owned_cells(advance, grid, term.iyee)  # noqa: SLF001
            stepped = note("bfast:state+advance", (state_in + advance).astype(f32))
            _shared._agrees("_bfast_term:state", stepped, state)  # noqa: SLF001
            _shared._agrees("_bfast_term:return", -advance, out)  # noqa: SLF001
            return out

        _stepping._bfast_term = bfast  # noqa: SLF001
        return self


# THE CENSUS THE SHARED DRIVER USES IS SUBSTITUTED FOR THIS ONE, at module scope and
# once. ``_shared.drive`` constructs ``IntermediateCensus()`` by name inside its own
# loop, so the ONLY seam a gate has for extending the instrument is the attribute the
# name resolves to. This rebinds a HARNESS attribute in this process, never a product
# one.
_shared.IntermediateCensus = BfastIntermediateCensus


# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------

def _tap_call(coords: str, target: int) -> str:
    return f"h_cell({coords}, {plain.H_CELL_TAIL_ARGS}).a{target}"


def _a_y_line() -> str:
    return f"    float a_y = vy ? {_tap_call('i, sj, k', 0)} : 0.0f;\n"


_ACCUMULATION_0 = ("        a0 = a0 + kp_0 * src0;\n"
                   "        a0 = a0 - km_0 * prev0;\n")


def shader_mutations(codes: Sequence[int]) -> Dict[str, Tuple[str, str]]:
    """Every armed source defect as ``name -> (source, why it must fire)``.

    Every needle is anchored on text only the mutated line carries and is verified to
    resolve EXACTLY ONCE before it is applied.
    """
    base = family.bfast_fused_hd_pair_source(codes)
    out: Dict[str, Tuple[str, str]] = {}
    tap = _a_y_line()

    def arm(name: str, source: str, why: str) -> None:
        out[name] = (source, why)

    # THE LEGS THIS FAMILY EXISTS FOR: the foreign tap.
    arm("foreign_tap_reads_stale_H",
        needle(base, tap, "    float a_y = vy ? hi0[oy] : 0.0f;\n"),
        "the tap reads the PRE-LAUNCH H at the neighbour instead of recomputing "
        "update_H there; fires wherever update_H moves that cell (the purity ledger)")
    arm("foreign_tap_reads_B",
        needle(base, tap, "    float a_y = vy ? b0[oy] : 0.0f;\n"),
        "the tap reads the flux density where the curl needs the stepped H")
    arm("foreign_tap_reads_the_scratch_output",
        needle(base, tap, "    float a_y = vy ? ho0[oy] : 0.0f;\n"),
        "THE PLANTED RACE: the tap reads another thread's store, which holds either "
        "the neighbour's new H (if that thread ran first) or two-launches-old scratch")
    arm("halo_recompute_dropped",
        needle(base, tap, "    float a_y = 0.0f;\n"),
        "one shifted magnetic load replaced by the ghost value everywhere")
    arm("halo_moved_to_the_forward_neighbour",
        needle(base, "    int si = i __SHIFT_PLACEHOLDER__" if False else
               "    int si = i - 1, sj = j - 1, sk = k - 1;\n",
               "    int si = i - 1, sj = j + 1, sk = k - 1;\n"),
        "step_D is the BACKWARD curl; one axis differenced forward is step_B's "
        "stencil")
    arm("own_cell_reads_stale_H_not_the_register",
        needle(base, "    float a   = own.a0;\n", "    float a   = hi0[ii];\n"),
        "the curl takes the pre-launch H at its own cell: the seam undone")
    # THE CONSTITUTIVE HALF, inside h_cell (8-space indent: the lifted body).
    arm("constitutive_accumulations_regrouped",
        needle(base, _ACCUMULATION_0,
               "        a0 = a0 + (kp_0 * src0 - km_0 * prev0);\n"),
        "((f + kps*src) - kms*prev) regrouped is a different float32 number")
    arm("constitutive_accumulations_reversed",
        needle(base, _ACCUMULATION_0,
               "        a0 = a0 - km_0 * prev0;\n        a0 = a0 + kp_0 * src0;\n"),
        "the two accumulations in the other order")
    arm("prev_reads_the_value_the_store_writes",
        needle(base, "        float prev0 = wi0[ii];\n",
               "        float prev0 = b0[ii];\n"),
        "the split-field history read AFTER the store that overwrote it with B")
    arm("kps_kms_swapped",
        needle(base, "        float kp_0 = kp0[i], km_0 = kmx[i];\n",
               "        float kp_0 = kmx[i], km_0 = kp0[i];\n"),
        "the two absorber coefficients exchanged on one component")
    arm("h_store_dropped",
        needle(base, "    ho0[ii] = own.a0; ho1[ii] = own.a1; ho2[ii] = own.a2;\n",
               "    ho1[ii] = own.a1; ho2[ii] = own.a2;\n"),
        "one magnetic component never reaches the scratch; the next step reads stale")
    arm("fw_store_dropped",
        needle(base, "    wo0[ii] = own.src0; wo1[ii] = own.src1; wo2[ii] = own.src2;\n",
               "    wo1[ii] = own.src1; wo2[ii] = own.src2;\n"),
        "one split-field history never advances")
    # THE CURL HALF.
    arm("curl_parens_flattened",
        needle(base, "dtdx * ((c_y - c) + (b - b_z))", "dtdx * (c_y - c + b - b_z)"),
        "shaders.py rule 2: reassociation is a different float32 number")
    arm("ownership_mask_dropped",
        needle(base, "    curl0 = at_y ? 0.0f : curl0;\n", ""),
        "the non-owned plane keeps a curl the array path masks")
    arm("recurrence_axis_pair_swapped",
        needle(base, "    float n0 = ((p0 * km_y) - curl0) * si_y;\n",
               "    float n0 = ((p0 * km_z) - curl0) * si_z;\n"),
        "vec.hpp's cycle_direction: target 0 takes (y, z)")
    # THE PACK: an offset that is one member out is a smooth wrong absorber.
    arm("vector_pack_offset_swapped",
        needle(base, "    device const float* kmx = cpack + prm.off_kmx;\n",
               "    device const float* kmx = cpack + prm.off_kmy;\n"),
        "the x absorber vector read at the y vector's offset: a converged, smooth, "
        "wrong absorber profile rather than a failure")
    arm("constitutive_kp_read_from_the_curl_group",
        needle(base, "    device const float* kp0 = cpack + prm.off_kp0;\n",
               "    device const float* kp0 = cpack + prm.off_sinvx;\n"),
        "the constitutive's kps read at the curl's sinv offset")
    return out


def bfast_mutations(codes: Sequence[int]) -> Dict[str, Tuple[str, str]]:
    """The defects only the BFAST tail carries."""
    base = family.bfast_fused_hd_pair_source(codes)
    out: Dict[str, Tuple[str, str]] = {}

    def arm(name: str, source: str, why: str) -> None:
        out[name] = (source, why)

    arm("tustin_state_not_stored",
        needle(base, "    s0[ii] = st0 + adv0;\n", ""),
        "the Tustin IIR never advances on one component; marginally stable, so it "
        "never decays back into agreement")
    arm("advance_not_subtracted_from_the_curl",
        needle(base, "    curl0 = curl0 - adv0;\n", ""),
        "stepping.py:904 returns -advance and :364-368 ADDS it; dropping the "  # stepping.py live lines for the frozen device-text citation(s) in this string: 904->931, 364-368->392-396
        "subtraction removes the whole second additive term on one target")
    arm("advance_added_rather_than_subtracted",
        needle(base, "    curl0 = curl0 - adv0;\n", "    curl0 = curl0 + adv0;\n"),
        "the caller-subtracts sign convention inverted on one target")
    arm("tustin_sums_use_the_curls_sign_pattern",
        needle(base, "    float total0 = (k1_0 * (c_y + c)) - (k2_0 * (b_z + b));\n",
               "    float total0 = (k1_0 * (c_y - c)) - (k2_0 * (b_z - b));\n"),
        "the tail's two sums are in the SAME orientation (`shifted + centre`), unlike "
        "the curl's differences; this is the sign pattern bfast_curl.py's own comment "
        "warns is NOT the curl's")
    arm("tustin_k1_k2_swapped",
        needle(base, "    float total0 = (k1_0 * (c_y + c)) - (k2_0 * (b_z + b));\n",
               "    float total0 = (k2_0 * (c_y + c)) - (k1_0 * (b_z + b));\n"),
        "k1 multiplies the FIRST term's sum and k2 the SECOND (stepping.py:884-885); "  # stepping.py live lines for the frozen device-text citation(s) in this string: 884-885->911-912
        "exchanged, the filter is a different one on every cell")
    arm("advance_factor_changed",
        needle(base, "    float adv0 = total0 - (2.0f * st0);\n",
               "    float adv0 = total0 - (1.0f * st0);\n"),
        "the Tustin advance is `total - 2*state` (stepping.py:896-904)")  # stepping.py live lines for the frozen device-text citation(s) in this string: 896-904->923-931
    arm("advance_mask_dropped",
        needle(base, "    adv0 = at_y ? 0.0f : adv0;\n", ""),
        "the advance carries the SAME ownership predicate the curl does, applied "
        "BEFORE the state store (:902 precedes :903); dropping it makes a diagnostic "
        "read of f_bfast disagree with MEEP's")
    arm("dtdx_applied_to_the_tail",
        needle(base, "    float adv0 = total0 - (2.0f * st0);\n",
               "    float adv0 = (dtdx * total0) - (2.0f * st0);\n"),
        "MEEP's step_bfast never reads the dtdx it is handed (stepping.py:836-837); "  # stepping.py live lines for the frozen device-text citation(s) in this string: 836-837->863-864
        "this is the Tustin filter, not a curl")
    return out


def predicted_null_mutations(codes: Sequence[int]
                             ) -> Dict[str, Tuple[str, str, str]]:
    """Armed defects required NOT to fire, each with its reason and its control."""
    base = family.bfast_fused_hd_pair_source(codes)
    out: Dict[str, Tuple[str, str, str]] = {}
    if all(codes):
        # THE EDIT MOVES THE GHOST BRANCH, NEVER THE TAKEN ONE. Measured 2026-09-07:
        # an edit written as `vy ? 1.0f : 0.0f` replaces the RECOMPUTE on every cell
        # where `vy` holds, which is the whole interior -- it fired, correctly, and
        # said nothing about the ghost. Only the `: 0.0f` side is the ghost.
        out["metallic_ghost_replaced_by_the_own_cells_constitutive"] = (
            needle(base, _a_y_line(),
                   f"    float a_y = vy ? {_tap_call('i, sj, k', 0)} : own.a0;\n"),
            "INERT BY CONSTRUCTION, and it is a property of the ARRAY PATH rather "
            "than of this weld. `a_y` feeds curl target 2 and nothing else, and "
            "stepping._mask_non_owned_cells (stepping.py:1940-1948) zeroes that "
            "target wherever j == 0 on a metallic y axis -- which is EXACTLY the "
            "plane where `vy` is false and the exact 0.0f past the wall is served. "
            "So no edit to a metallic ghost VALUE can reach this kernel's output. "
            "The same boundary rule IS observable on a PERIODIC axis, where that "
            "mask does not fire, and that is the control. ON THIS FAMILY THE CLAIM "
            "IS WIDER THAN ON THE PLAIN WELD and is still true: `a_y` also feeds the "
            "BFAST tail's `total2`, and `advance_mask` applies the SAME ownership "
            "predicate to `adv2` before the state store (bfast_curl.py:902), so the "
            "ghost is masked out of the Tustin state as well as out of the curl",
            "periodic_wrap_reads_the_own_row")
    return out


def periodic_wrap_mutations(codes: Sequence[int]) -> Dict[str, Tuple[str, str]]:
    """The boundary defects only an ALL-PERIODIC specialisation can carry."""
    base = family.bfast_fused_hd_pair_source(codes)
    return {
        "periodic_wrap_reads_the_own_row": (
            needle(base, "    sj = (sj < 0) ? (nyi - 1) : sj;\n",
                   "    sj = (sj < 0) ? 0 : sj;\n"),
            "the backward y tap at j = 0 wraps to the LAST stored row; reading row 0 "
            "instead is the periodic rule dropped, and on a periodic axis the "
            "ownership mask does not cover that plane"),
        "periodic_wrap_reads_one_row_short": (
            needle(base, "    sj = (sj < 0) ? (nyi - 1) : sj;\n",
                   "    sj = (sj < 0) ? (nyi - 2) : sj;\n"),
            "the wrap lands one row short of the last: a plausible off-by-one that "
            "keeps the branch and moves the cell"),
    }


def byte_neutral_source(codes: Sequence[int]) -> str:
    """One armed edit required NOT to diverge: a comment, and nothing else."""
    base = family.bfast_fused_hd_pair_source(codes)
    return needle(base, "    float curl0 = dtdx * ",
                  "    // byte-neutral control: a comment, no arithmetic\n"
                  "    float curl0 = dtdx * ")


# ---------------------------------------------------------------------------
# BFAST-specific host legs
# ---------------------------------------------------------------------------

def leg_arm_pairing() -> Dict[str, Any]:
    """THE CELL IS THE SHIPPED COMPOSER'S OWN ANSWER, not this gate's inference.

    Unlike the conductive H->D cell, both halves here come from ONE family --
    ``bfast_curl`` registers all four slots -- but the two halves are still different
    predicates over different kernels, and the constitutive one builds through
    ``launch``'s certified ``ConstitutivePlan``. Both are re-asked one at a time, and
    the ORDINARY arms are required to refuse the same fixture, which is what makes the
    pairing a partition rather than a preference.
    """
    from meep_gpu.metal_kernels.coverage import (  # noqa: PLC0415
        constitutive_coverage, pml_curl_coverage,
    )

    driver = build_driver(dict(CASES)[MUTATION_CASE], 4242)
    residency = Residency()
    composed = metal_launch.plan_step(driver.fields, driver.pml, residency=residency,
                                      sources=(), fuse=False)
    selected = dict(composed.selected)
    pair = (selected.get("update_H"), selected.get("step_D"))
    halves = {
        "update_H_bfast_run_constitutive": bfast_curl.bfast_run_constitutive_coverage(
            driver.fields, driver.pml, "H", residency).covered,
        "step_D_bfast_curl": bfast_curl.bfast_pml_curl_coverage(
            driver.fields, driver.pml, "step_D", residency).covered,
        # AND THE ORDINARY ARMS REFUSE, which is the other half of the same
        # measurement: two products claiming one slot is how a composer builds both.
        "update_H_ordinary": constitutive_coverage(driver.fields, driver.pml, "H",
                                                   residency).covered,
        "step_D_ordinary_pml": pml_curl_coverage(driver.fields, driver.pml, "step_D",
                                                 residency).covered,
    }
    return {
        "passed": bool(pair == CELL_ARMS
                       and halves["update_H_bfast_run_constitutive"]
                       and halves["step_D_bfast_curl"]
                       and not halves["update_H_ordinary"]
                       and not halves["step_D_ordinary_pml"]),
        "composer_selected": selected,
        "cell_arms_expected": list(CELL_ARMS),
        "cell_arms_measured": list(pair),
        "half_predicates": halves,
        "bfast_scaled_k": list(driver.grid.bfast_scaled_k),
        "what_this_settles": (
            "on a BFAST run the shipped composer selects the BFAST arm on BOTH slots, "
            "and both ordinary arms refuse the same fixture BY NAME. The constitutive "
            "half is the CERTIFIED ordinary body under a restated predicate "
            "(bfast_curl.plan_bfast_run_constitutive builds through launch's own "
            "ConstitutivePlan), so what this weld lifts on that side is the same text "
            "fused_hd_pair lifts"),
    }


def leg_refusal() -> Dict[str, Any]:
    """Each refusal by NAME, and the inverted BFAST clause driven in BOTH directions."""
    rows: Dict[str, Any] = {}

    def record(name: str, covered: bool, reasons: Sequence[str],
               expect: bool, needs: Sequence[str] = ()) -> None:
        joined = " | ".join(reasons)
        rows[name] = {
            "covered": covered, "expected": expect,
            "reasons": list(reasons)[:6],
            "named": [phrase for phrase in needs if phrase in joined],
            "ok": covered == expect and all(phrase in joined for phrase in needs),
        }

    driver = build_driver(dict(CASES)[MUTATION_CASE], 7)
    verdict = family.metal_bfast_fused_hd_pair_coverage(
        driver.fields, driver.pml, (), Residency())
    record("the_bfast_fixture", verdict.covered, verdict.reasons, True)

    # THE INVERSION, BOTH WAYS. A run with no shear belongs to `fused_hd_pair`.
    plainly = build_driver({"boundaries": {"x": "metallic", "y": "metallic",
                                           "z": "metallic"},
                            "bfast_scaled_k": (0.0, 0.0, 0.0)}, 8)
    verdict = family.metal_bfast_fused_hd_pair_coverage(
        plainly.fields, plainly.pml, (), Residency())
    record("a_run_with_no_shear_is_refused", verdict.covered, verdict.reasons, False)
    verdict = plain.metal_fused_hd_pair_coverage(driver.fields, driver.pml, (),
                                                 Residency())
    record("the_plain_product_refuses_the_bfast_fixture", verdict.covered,
           verdict.reasons, False, ["BFAST is active"])
    verdict = plain.metal_fused_hd_pair_coverage(plainly.fields, plainly.pml, (),
                                                 Residency())
    record("the_plain_product_admits_the_unsheared_grid", verdict.covered,
           verdict.reasons, True)

    # THE SEAM'S ONE PASS. An undeclared source set is not an empty one.
    verdict = family.metal_bfast_fused_hd_pair_coverage(
        driver.fields, driver.pml, None, Residency())
    record("an_undeclared_source_set_is_refused", verdict.covered, verdict.reasons,
           False, ["the source set was not declared"])
    withdrawing = _volume_source(driver.fields, "Ez", integrated=True)
    verdict = family.metal_bfast_fused_hd_pair_coverage(
        driver.fields, driver.pml, (withdrawing,), Residency())
    record("a_standing_integrated_electric_withdraw_is_refused", verdict.covered,
           verdict.reasons, False, ["HOISTS_THE_WITHDRAW = False"])
    unintegrated = _volume_source(driver.fields, "Ez", integrated=False)
    verdict = family.metal_bfast_fused_hd_pair_coverage(
        driver.fields, driver.pml, (unintegrated,), Residency())
    record("a_non_integrated_electric_source_is_admitted", verdict.covered,
           verdict.reasons, True)
    return {"passed": all(row["ok"] for row in rows.values()), "cases": rows}


def leg_tustin_scalars() -> Dict[str, Any]:
    """The six k words the plan bakes are ``stepping``'s own, with the D-side negation.

    THE SCALARS ARE THE ONE THING THIS FAMILY CAN GET WRONG SILENTLY. They are computed
    on the host, rounded to float32 ONCE, and written into a packed struct; a sign, an
    order or a rounding that moved is a converged, smooth, wrong Tustin filter rather
    than a failure. So they are compared as uint32 WORDS against
    ``bfast_curl_coefficients``' own output, and the MAGNETIC side's values are carried
    as the control that must DIFFER -- because a plan that handed the B-side scalars to
    a D-side kernel would otherwise look exactly like this leg passing.
    """
    rows: Dict[str, Any] = {}
    for name, keywords in CASES:
        driver = build_driver(keywords, 1717)
        residency = Residency()
        plan = build_weld(driver, residency)
        params = plan.static_args[-1].cpu().numpy().tobytes()
        struct = ([("nx", "<u4"), ("ny", "<u4"), ("nz", "<u4"), ("n_elem", "<u4"),
                   ("dtdx", "<f4")]
                  + [(scalar, "<f4") for scalar in family.K_SCALARS]
                  + coefficient_pack.record_dtype_fields(family.PACKED_VECTORS))
        decoded = np.frombuffer(params, dtype=np.dtype(struct))[0]
        invariant = tuple(driver.grid.is_invariant(axis) for axis in range(3))
        electric = bfast_curl_coefficients(driver.grid.bfast_scaled_k, invariant,
                                           magnetic=False)
        magnetic = bfast_curl_coefficients(driver.grid.bfast_scaled_k, invariant,
                                           magnetic=True)
        baked = np.array([float(decoded[scalar]) for scalar in family.K_SCALARS],
                         dtype=np.float32)
        rows[name] = {
            "bfast_scaled_k": list(driver.grid.bfast_scaled_k),
            "invariant_axes": list(invariant),
            "baked": baked.tolist(),
            "expected_electric": list(electric),
            "expected_magnetic": list(magnetic),
            "differing_words_vs_the_electric_side": int(
                differing(np.array(electric, dtype=np.float32), baked)),
            "differing_words_vs_the_magnetic_side": int(
                differing(np.array(magnetic, dtype=np.float32), baked)),
            # THE D SIDE NEGATES BOTH (stepping.py:913-914, MEEP's `if (ft ==
            # D_stuff)`), so the two sides are the same magnitudes with opposite signs
            # wherever they are nonzero.
            "the_D_side_is_the_negated_B_side": bool(np.array_equal(
                np.array(electric, dtype=np.float32),
                -np.array(magnetic, dtype=np.float32))),
        }
    nonzero = [name for name, row in rows.items()
               if any(value != 0.0 for value in row["baked"])]
    return {
        "passed": bool(rows and nonzero
                       and all(row["differing_words_vs_the_electric_side"] == 0
                               and row["the_D_side_is_the_negated_B_side"]
                               for row in rows.values())
                       and all(rows[name]["differing_words_vs_the_magnetic_side"] > 0
                               for name in nonzero)),
        "cases": rows,
        "cases_with_a_live_scalar": nonzero,
        "comparator": "uint32 words, never allclose",
        "what_this_licenses": (
            "the six Tustin scalars the plan bakes into Params are bit-for-bit the "
            "ones stepping's own term tables produce for the ELECTRIC side, and they "
            "are NOT the magnetic side's -- which is the control that makes the first "
            "claim worth having"),
    }


def leg_state_census() -> Dict[str, Any]:
    """Every ``f_bfast_D`` component moves somewhere on the fixture set.

    MEASURED ON THE ARRAY PATH, and it is what keeps :data:`BFAST_K` honest. With
    ``bfast_scaled_k = (0.2, 0, 0)`` -- the shape the cell's corpus row carries --
    target 0's two scalars are read off the OTHER two axes and are both zero, so
    ``f_bfast_Dx`` is identically zero for the whole run and a fixture built on it would
    compare a state volume that never moved. The three-component shear is what makes
    every Tustin mutation reachable, and the single-axis row is kept so the corpus
    row's own shape is still exercised.
    """
    rows: Dict[str, Any] = {}
    for name, keywords in CASES:
        driver = build_driver(keywords, 808)
        seed = {state: words(getattr(driver.fields, state)).copy()
                for state in BFAST_STATE_NAMES["step_D"]}
        for _ in range(6):
            driver.step()
        rows[name] = {
            "bfast_scaled_k": list(driver.grid.bfast_scaled_k),
            "words_moved": {state: int(np.count_nonzero(
                seed[state] != words(getattr(driver.fields, state))))
                for state in BFAST_STATE_NAMES["step_D"]},
        }
        rows[name]["components_that_never_moved"] = sorted(
            state for state, moved in rows[name]["words_moved"].items() if not moved)
        log(f"    state_census {name}: {rows[name]['words_moved']}")
    live = {name: row for name, row in rows.items()
            if not row["components_that_never_moved"]}
    return {
        "passed": bool(MUTATION_CASE in live and PERIODIC_MUTATION_CASE in live),
        "cases": rows,
        "fixtures_where_every_tustin_component_moves": sorted(live),
        "what_this_licenses": (
            "on the two fixtures the mutations are armed on, all three f_bfast_D "
            "components carry live state, so every Tustin mutation has words to move. "
            "The single-axis row is kept because it is the shear the cell's corpus row "
            "carries, and its two dead components are a fact about that shear rather "
            "than about this kernel"),
    }



def leg_driver_order() -> Dict[str, Any]:
    """``REPLACES`` is the driver's two ADJACENT consults, read off the tree."""
    source = Path(_shared.stepping.__file__).with_name("driver.py").read_text(
        encoding="utf-8")
    lines = source.splitlines()
    consults = [(index + 1, line.strip()) for index, line in enumerate(lines)
                if 'dispatch("update_H"' in line or 'dispatch("step_D"' in line]
    update_h = [row for row in consults if "update_H" in row[1]]
    step_d = [row for row in consults if "step_D" in row[1]]
    between: List[str] = []
    fallback: List[str] = []
    if update_h and step_d:
        lo = min(row[0] for row in update_h)
        hi = min(row[0] for row in step_d if row[0] > lo)
        # THE CONSULT'S OWN ARRAY-PATH BODY IS NOT "BETWEEN" THE TWO CONSULTS. The
        # driver's call site is `if fast is None or not fast.dispatch(...): <array
        # call>`, so the statement immediately under the update_H consult is that
        # slot's fallback -- the pass this product REPLACES, not a pass it would have
        # to carry across its launch. Counting it as in-seam would report the seam as
        # carrying a constitutive update that is in fact the seam's first half.
        body = [line.strip() for line in lines[lo:hi - 1]
                if line.strip() and not line.strip().startswith("#")]
        fallback = [line for line in body if line.startswith("update_H(")]
        between = [line for line in body if line not in fallback]
    withdraw_only = bool(between) and all(
        line == "for source in electric:" or line.startswith("getattr(source")
        for line in between)
    return {
        "passed": bool(update_h and step_d and family.REPLACES == ("update_H", "step_D")
                       and withdraw_only),
        "replaces": list(family.REPLACES),
        "update_H_consults": update_h,
        "step_D_consults": step_d,
        "the_update_H_consults_own_array_path_fallback": fallback,
        "statements_between_the_two_consults": between,
        "only_the_electric_withdraw_sits_between_them": withdraw_only,
        "carries_deposit_repair": family.CARRIES_DEPOSIT_REPAIR,
        "hoists_the_withdraw": family.HOISTS_THE_WITHDRAW,
        "seam": family.SEAM,
    }


def _tail_after_decode(source: str) -> str:
    """The certified BFAST curl's body below the package's shared decode anchor."""
    return source.split(plain.DECODE_END, 1)[1]


def leg_transcription(codes: Sequence[int]) -> Dict[str, Any]:
    """The emitted source differs from the two certified emitters in EXACTLY the
    declared edits, and every mutation needle resolves once."""
    emitted = family.bfast_fused_hd_pair_source(codes)
    certified_curl = bfast_curl.bfast_curl_source(
        codes, bool(metal_launch.SUB_STEPS["step_D"]["backward"]), has_bfast=True)
    certified_constitutive = shaders.constitutive_source("H")
    welded = family.bfast_welded_curl_tail(codes)
    certified_tail = _tail_after_decode(certified_curl)[: -len("}\n")]

    # THE CURL HALF: every line differs only where a magnetic read was redirected.
    certified_lines = certified_tail.splitlines()
    welded_lines = welded.splitlines()
    changed = [(a, b) for a, b in zip(certified_lines, welded_lines) if a != b]
    only_magnetic = all(("g0[" in a or "g1[" in a or "g2[" in a)
                        and ("h_cell(" in b or "own.a" in b)
                        for a, b in changed)
    survives = [f"g{target}[" for target in range(3)
                if f"g{target}[" in welded]

    # THE CONSTITUTIVE HALF: the lift edits, verified by re-deriving them.
    h_cell = plain.h_cell_function()
    lifted_ok = all(spelling in h_cell for spelling in
                    ("float prev0 = wi0[ii];", "float src0 = b0[ii];",
                     "float a0 = hi0[ii];", "kp_0 = kp0[i], km_0 = kmx[i];"))
    stores_removed = ("w0[ii] = src0;" not in h_cell
                      and "f0[ii] = a0;" not in h_cell)

    needles = {}
    for name, (source, _why) in sorted(shader_mutations(codes).items()):
        needles[name] = source != emitted
    for name, (source, _why) in sorted(bfast_mutations(codes).items()):
        needles[name] = source != emitted
    return {
        "passed": bool(changed and only_magnetic and not survives and lifted_ok
                       and stores_removed and all(needles.values())
                       and emitted.count("h_cell(") == H_CELL_CALL_SITES),
        "lines_changed_in_the_curl_half": len(changed),
        "every_change_is_a_redirected_magnetic_read": only_magnetic,
        "magnetic_pointer_spellings_that_survive": survives,
        "constitutive_lift_edits_present": lifted_ok,
        "constitutive_stores_removed": stores_removed,
        "h_cell_call_sites": emitted.count("h_cell("),
        "h_cell_call_sites_expected": H_CELL_CALL_SITES,
        "every_mutation_needle_resolves": needles,
        "decode_anchor": plain.DECODE_END.strip(),
        "certified_curl_sha256": hashlib.sha256(
            certified_curl.encode()).hexdigest(),
        "certified_constitutive_sha256": hashlib.sha256(
            certified_constitutive.encode()).hexdigest(),
        "emitted_sha256": hashlib.sha256(emitted.encode()).hexdigest(),
    }


def _volume_source(fields: Any, component: str, integrated: bool) -> Any:
    """THE ENGINE'S OWN SOURCE, not a stand-in.

    ``withdraw_hoist._withdraw_does_work`` reads ``_n_source_points``, which only a
    real ``VolumeSource`` publishes -- a duck-typed stub answers 0 and the clause
    that must refuse quietly admits instead. Measured on the first run of this leg.
    """
    from meep_gpu.sources import GaussianEnvelope, VolumeSource  # noqa: PLC0415

    return VolumeSource(grid=fields.grid, component=component,
                        center=(0.15, -0.1, 0.05), size=(0.0, 0.0, 0.0),
                        envelope=GaussianEnvelope(frequency=1.0, fwidth=0.5,
                                                  is_integrated=integrated))


def leg_arbitration() -> Dict[str, Any]:
    """The composer, asked, with this product REGISTERED: zero installed."""
    from meep_gpu.metal_kernels import arms  # noqa: PLC0415

    rows: Dict[str, Any] = {}
    registered = [spec for spec in arms.registered("update_H")
                  if spec.family == family.FAMILY]
    for name, keywords in CASES:
        driver = build_driver(keywords, 909)
        residency = Residency()
        plan = metal_launch.plan_step(driver.fields, driver.pml, residency=residency,
                                      sources=(), fuse=True)
        unfused = metal_launch.plan_step(driver.fields, driver.pml,
                                         residency=Residency(), sources=(),
                                         fuse=False)
        # AND THE SAME QUESTION WITH THE ABSORB ROW APPLIED IN-PROCESS. Without a
        # `launch.FUSED_PAIR_ARMS` row the seam loop refuses this product before
        # asking anything about the run, so a refusal measured only that way says
        # nothing about the arbitration. With the row applied the refusal has to move
        # to the flag, and the released neighbours have to keep their slots anyway --
        # which is the claim INSTALLABLE_REASON makes. The table is restored in a
        # `finally`, because leaving a row behind would wire the product for every
        # later leg in this process.
        saved = dict(metal_launch.FUSED_PAIR_ARMS)
        metal_launch.FUSED_PAIR_ARMS[family.FAMILY] = CELL_ARMS
        try:
            wired = metal_launch.plan_step(driver.fields, driver.pml,
                                           residency=Residency(), sources=(),
                                           fuse=True)
            wired_labels = {slot: str(value)
                            for slot, value in wired.selected.items()}
            wired_reasons = [
                reason for key, value in wired.reasons.items()
                if family.FAMILY in key for reason in
                (value if isinstance(value, (list, tuple)) else [value])][:2]
        finally:
            metal_launch.FUSED_PAIR_ARMS.clear()
            metal_launch.FUSED_PAIR_ARMS.update(saved)
        labels = {slot: str(value) for slot, value in plan.selected.items()}
        rows[name] = {
            "selected_fused": labels,
            "selected_unfused": dict(unfused.selected),
            "this_product_installed": any(family.FAMILY in value
                                          for value in labels.values())
            or any("BFAST fused H/D" in value for value in labels.values()),
            "reasons_naming_this_product": [
                reason for key, value in plan.reasons.items()
                if family.FAMILY in key for reason in
                (value if isinstance(value, (list, tuple)) else [value])][:4],
            # THE SUBSTANTIVE CLAIM, not a positional one. The two RELEASED
            # neighbours keep the four slots they hold today; which LABEL each wears
            # is the composer's business and is RECORDED rather than asserted, because
            # wiring this product changes labels and not ownership.
            "both_neighbouring_pairs_still_hold_their_slots": bool(
                labels.get("step_B") and labels.get("step_B") == labels.get("update_H")
                and labels.get("step_D")
                and labels.get("step_D") == labels.get("update_E")
                and labels.get("step_B") != labels.get("step_D")),
            "the_unfused_cell_is_still_this_products":
                (unfused.selected.get("update_H"),
                 unfused.selected.get("step_D")) == CELL_ARMS,
            "every_slot_is_still_filled": sorted(labels) == sorted(
                ["step_B", "update_H", "step_D", "update_E"]),
            "selected_with_the_absorb_row_applied_in_process": wired_labels,
            "refusal_with_the_absorb_row_applied_in_process": wired_reasons,
            "the_absorb_row_does_not_change_the_selection":
                wired_labels == labels,
            "the_refusal_moves_to_the_flag": any(
                "INSTALLABLE = False" in reason for reason in wired_reasons),
        }
        log(f"    arbitration {name}: installed="
            f"{rows[name]['this_product_installed']} selected={labels}")
    return {
        "passed": bool(rows and len(registered) == 1
                       and registered[0].wired is False
                       and registered[0].replaces == family.REPLACES
                       and family.INSTALLABLE is False
                       and all(not row["this_product_installed"]
                               and row["both_neighbouring_pairs_still_hold_their_slots"]
                               and row["the_unfused_cell_is_still_this_products"]
                               and row["every_slot_is_still_filled"]
                               and row["the_absorb_row_does_not_change_the_selection"]
                               and row["the_refusal_moves_to_the_flag"]
                               for row in rows.values())),
        "registered_rows_on_update_H": len(registered),
        "registered_wired": [spec.wired for spec in registered],
        "installable": family.INSTALLABLE,
        "installable_reason": family.INSTALLABLE_REASON,
        "absorb_row": metal_launch.FUSED_PAIR_ARMS.get(family.FAMILY),
        "cases": rows,
        "what_this_says": (
            "the product is registered so the composition sweep and "
            "_neighbouring_seam_claimant can SEE it, and installed on nothing: "
            "arms.arms_for skips an unwired row, INSTALLABLE is False, and the two "
            "RELEASED neighbours keep all four slots on every fixture. Both "
            "neighbours installing is what makes this product a LOSS: 2 pairs -> 1. "
            "The labels are RECORDED and not asserted -- wiring this product into "
            "FUSED_PAIR_ARMS changes which label a slot wears without changing who "
            "owns it, and a positional assertion would invert on exactly that edit"),
    }


def leg_seed_scale() -> Dict[str, Any]:
    """The 2^80 seed changes no mantissa, and the fixture then clears the budget.

    DRIVEN HERE rather than inherited from the sibling gate: this fixture carries a
    volume the sibling's does not -- the Tustin state, whose filter is marginally
    stable and whose equality under the scale is the claim that has to hold for the
    whole leg to mean anything.
    """
    scale = np.float32(2.0) ** SEED_SCALE_BITS
    rows: Dict[str, Any] = {}
    for name in (MUTATION_CASE, PERIODIC_MUTATION_CASE,
                 "single_axis_shear_mmm"):
        keywords = dict(CASES)[name]
        plain_run = build_driver(keywords, 90100, scale_bits=0)
        scaled = build_driver(keywords, 90100)
        plain_band: Optional[int] = None
        scaled_band: Optional[int] = None
        mismatch: Optional[Dict[str, Any]] = None
        peak = 0.0
        for step in range(1, STEPS + 1):
            plain_run.step()
            scaled.step()
            a = stored_volumes(plain_run.fields)
            b = stored_volumes(scaled.fields)
            if plain_band is None and sum(subnormal.census(v) for v in a.values()):
                plain_band = step
            if scaled_band is None and sum(subnormal.census(v) for v in b.values()):
                scaled_band = step
            peak = max(peak, max(float(np.max(np.abs(v))) for v in b.values()))
            if plain_band is None and mismatch is None:
                bad = {key: differing(np.asarray(a[key]) * scale, b[key])
                       for key in sorted(a)}
                bad = {key: count for key, count in bad.items() if count}
                if bad:
                    mismatch = {"step": step, "volumes": bad}
        rows[name] = {
            "unscaled_first_banded_step": plain_band,
            "scaled_first_banded_step": scaled_band,
            "scaled_peak_magnitude": peak,
            "float32_max": float(np.finfo(np.float32).max),
            "steps_the_equality_was_checked_over": (plain_band - 1 if plain_band
                                                    else STEPS),
            "first_word_that_is_not_the_scaled_twin": mismatch,
        }
        log(f"    seed_scale {name} unscaled_banded_at={plain_band} "
            f"scaled_banded_at={scaled_band} peak={peak:.3e} mismatch={mismatch}")
    needed = sorted(name for name, row in rows.items()
                    if row["unscaled_first_banded_step"] is not None)
    return {
        "passed": bool(rows and all(
            row["scaled_first_banded_step"] is None
            and row["first_word_that_is_not_the_scaled_twin"] is None
            and row["scaled_peak_magnitude"] < row["float32_max"]
            and row["steps_the_equality_was_checked_over"] >= 2
            for row in rows.values())),
        "seed_scale_bits": SEED_SCALE_BITS,
        "steps": STEPS,
        "cases_where_the_unscaled_fixture_enters_the_band": needed,
        "cases": rows,
        "what_this_licenses": (
            "that this gate's synthetic fixture is the composition matrix's own state "
            "with every exponent shifted by a constant, INCLUDING the Tustin "
            "filter's own state, and that the shifted fixture stays out of the "
            "float32 denormal band for the whole budget. It licenses nothing about a "
            "device and nothing about any other amplitude"),
    }


def purity_ledger(driver: Any) -> Dict[str, Any]:
    """Of the curl's valid foreign taps, how many land on a cell ``update_H`` MOVED.

    Measured on the ARRAY PATH: a property of the physics, not of the kernel. Near
    zero would mean the race controls and the stale-H mutation have nothing to catch.
    """
    from meep_gpu.stepping import _boundary_kinds  # noqa: PLC0415

    fields, pml = driver.fields, driver.pml
    codes = tuple(1 if kind == "metallic" else 0
                  for kind in _boundary_kinds(fields.grid, pml))
    before = {name: np.array(getattr(fields, name), copy=True)
              for name in ("Hx", "Hy", "Hz")}
    stepping.update_H(fields, pml)
    moved = {name: words(before[name]) != words(getattr(fields, name))
             for name in before}
    shape = tuple(fields.grid.shape)
    per_tap: Dict[str, Dict[str, int]] = {}
    total_valid = 0
    total_raced = 0
    for var, target in plain.HALO_TAPS:
        axis = "xyz".index(var[-1])
        component = ("Hx", "Hy", "Hz")[target]
        flags = moved[component].reshape(shape)
        shifted = np.roll(flags, 1, axis=axis)
        valid = np.ones(shape, dtype=bool)
        if codes[axis]:
            index: List[Any] = [slice(None)] * 3
            index[axis] = 0
            valid[tuple(index)] = False
        raced = int(np.count_nonzero(shifted & valid))
        count = int(np.count_nonzero(valid))
        per_tap[var] = {"valid_taps": count, "taps_on_a_moved_cell": raced}
        total_valid += count
        total_raced += raced
    return {
        "codes": list(codes),
        "per_tap": per_tap,
        "valid_foreign_taps": total_valid,
        "foreign_taps_whose_recompute_differs_from_the_in_place_read": total_raced,
        "fraction": round(total_raced / total_valid, 6) if total_valid else 0.0,
    }


def leg_purity() -> Dict[str, Any]:
    rows = {name: purity_ledger(build_driver(keywords, 5150))
            for name, keywords in CASES}
    fractions = [row["fraction"] for row in rows.values()]
    floor = min(fractions) if fractions else 0.0
    return {
        "passed": bool(rows and floor >= 0.99),
        "floor_fraction": floor,
        "what_this_licenses": (
            f"on every fixture at least {floor:.4f} of the curl's valid foreign taps "
            f"land on a cell update_H moved, so the race null control, the planted "
            f"race and the stale-H mutation each have that many taps to catch; a "
            f"near-zero number here would have made a green identity worthless"),
        "cases": rows,
    }


def leg_ghost_observability() -> Dict[str, Any]:
    """WHICH ghost defects a specialisation can observe at all, measured on the array
    path rather than asserted.

    On a METALLIC axis ``stepping._mask_non_owned_cells`` zeroes the very curl
    component the shifted tap along that axis feeds, at index 0 of that axis. So the
    ghost VALUE past a metallic wall is unobservable and a mutation that replaces it is
    a PREDICTED NULL; on a PERIODIC axis the mask does not fire and the wrap IS
    observable, which is why the periodic wrap mutations exist and are armed on
    ``bc_ppp``. This leg counts the masked cells per axis so the prediction rests on a
    number.
    """
    rows: Dict[str, Any] = {}
    for name, keywords in CASES:
        driver = build_driver(keywords, 606)
        codes = codes_of(name)
        shape = tuple(driver.grid.shape)
        masked = {}
        for axis, code in enumerate(codes):
            plane = int(np.prod([n for index, n in enumerate(shape)
                                 if index != axis]))
            masked["xyz"[axis]] = {"metallic": bool(code),
                                   "cells_at_index_0": plane,
                                   "curl_components_masked_there": 2 if code else 0}
        rows[name] = {
            "codes": list(codes),
            "per_axis": masked,
            "the_ghost_value_is_observable": not all(codes),
        }
    return {
        "passed": bool(rows
                       and rows[MUTATION_CASE]["the_ghost_value_is_observable"] is False
                       and rows[PERIODIC_MUTATION_CASE]["the_ghost_value_is_observable"]
                       is True),
        "cases": rows,
        "what_this_says": (
            "the metallic ghost's VALUE is unobservable through step_D's output on a "
            "walled axis and observable on a periodic one; the mutation leg therefore "
            "arms the ghost-value defect as a PREDICTED NULL on bc_mmm with "
            "halo_recompute_dropped as its earned control, and arms the periodic wrap "
            "on bc_ppp where it must fire"),
    }


def leg_policy() -> Dict[str, Any]:
    """What the resolved subnormal policy is, and what it does to the predicate.

    RECORDED, NOT ASSERTED. The campaign runs this gate under both policies from
    separate artifact directories; under a resolved ``keep`` the MPS executor cannot
    honour the policy and every Metal predicate refuses BY NAME, which is a measured
    refusal rather than a failure of this product.
    """
    report = subnormal.mps_policy_report()
    driver = build_driver(dict(CASES)[MUTATION_CASE], 11)
    verdict = family.metal_bfast_fused_hd_pair_coverage(
        driver.fields, driver.pml, (), Residency())
    return {
        "passed": True,
        "requested_policy": os.environ.get("MEEP_GPU_SUBNORMAL_POLICY"),
        "mps_policy_report": report,
        "predicate_admits_the_fixture_under_this_policy": bool(verdict.covered),
        "predicate_reasons": list(verdict.reasons)[:6],
        "what_this_records": (
            "the policy this run resolved and whether the shipped predicate admits "
            "the fixture under it. A resolved `keep` refuses by name because the MPS "
            "executor has no lever for the float32 flush; that refusal is the "
            "measurement, and the byte legs below are the flush leg's"),
    }


# ---------------------------------------------------------------------------
# Compile legs
# ---------------------------------------------------------------------------

def _compiles(source: str) -> Tuple[bool, str]:
    try:
        compile_source(source)
    except Exception as exc:  # noqa: BLE001 - the failure IS the measurement
        return False, str(exc).splitlines()[0][:200]
    return True, ""


def leg_binding_ceiling() -> Dict[str, Any]:
    """Three refuted signatures FAIL, the ceiling is bisected, the shipped 27
    COMPILES on every fixture's codes, and the packed probe LAUNCHES."""
    refuted: Dict[str, Any] = {}
    for name, builder, count in (
            ("unpacked_pointer", family.refuted_unpacked_pointer_source,
             family.UNPACKED_POINTER_BINDINGS),
            ("separate_scalar", family.refuted_separate_scalar_source,
             family.SEPARATE_SCALAR_BINDINGS)):
        ok, why = _compiles(builder())
        refuted[name] = {"bindings": count, "compiled": ok, "error": why,
                         "must_fail": True}
        log(f"    binding_ceiling refuted {name} ({count}) compiled={ok}")

    # THE BISECTION: 30 pointers + one packed struct compiles, 31 + struct does not.
    def probe(pointers: int) -> Tuple[bool, str]:
        lines = [f"    device float*       p{index:<6}[[buffer({index})]],"
                 for index in range(pointers)]
        lines.append(f"    constant Params&    prm     [[buffer({pointers})]],")
        source = "\n".join((
            "#include <metal_stdlib>", "using namespace metal;", "",
            "struct Params { uint n_elem; };", "",
            "kernel void ceiling_probe(", *lines,
            "    uint idx [[thread_position_in_grid]])", "{",
            "    if (idx >= prm.n_elem) { return; }",
            "    p0[idx] = p0[idx] + 1.0f;", "}", ""))
        return _compiles(source)

    at_ceiling, _ = probe(MAX_BUFFER_BINDINGS - 1)
    past_ceiling, past_why = probe(MAX_BUFFER_BINDINGS)

    shipped: Dict[str, Any] = {}
    for name, _keywords in CASES:
        ok, why = _compiles(family.bfast_fused_hd_pair_source(codes_of(name)))
        shipped[name] = {"compiled": ok, "error": why}

    # AND THE PACKED PROBE LAUNCHES, reading back every Params field.
    launched = _probe_params_readback()
    return {
        "passed": bool(all(not row["compiled"] for row in refuted.values())
                       and at_ceiling and not past_ceiling
                       and all(row["compiled"] for row in shipped.values())
                       and family.shipped_signature_bindings()
                       == family.PACKED_BINDINGS
                       and launched["passed"]),
        "ceiling": MAX_BUFFER_BINDINGS,
        "shipped_bindings": family.PACKED_BINDINGS,
        "shipped_bindings_counted_off_the_source":
            family.shipped_signature_bindings(),
        "headroom": MAX_BUFFER_BINDINGS - family.PACKED_BINDINGS,
        "refuted_signatures": refuted,
        "bisection": {f"{MAX_BUFFER_BINDINGS - 1}_pointers_plus_struct": at_ceiling,
                      f"{MAX_BUFFER_BINDINGS}_pointers_plus_struct": past_ceiling,
                      "error": past_why},
        "shipped_compiles_per_case": shipped,
        "params_readback": launched,
        "what_this_settles": (
            f"the unpacked shape is {family.UNPACKED_POINTER_BINDINGS} bindings "
            f"against a ceiling of {MAX_BUFFER_BINDINGS} -- over by three -- so the "
            f"coefficient pack, holding the nine read-only per-axis vectors, is what "
            f"makes this product exist at all rather than a tidiness"),
    }


def _probe_params_readback() -> Dict[str, Any]:
    """Launch a kernel with this family's ``Params`` layout and read every field back.

    A packed struct is only as good as its host record: a field written at the wrong
    offset is a plausible wrong number rather than an error, so the record is written,
    bound, and read back on the device.
    """
    import torch  # noqa: PLC0415

    names = family.PACKED_VECTORS
    fields = ([("nx", "<u4"), ("ny", "<u4"), ("nz", "<u4"), ("n_elem", "<u4"),
               ("dtdx", "<f4")]
              + [(scalar, "<f4") for scalar in family.K_SCALARS]
              + coefficient_pack.record_dtype_fields(family.PACKED_VECTORS))
    record = np.zeros(1, dtype=np.dtype(fields))
    # THE SIX TUSTIN SCALARS RIDE THE SAME STRUCT AS THE OFFSETS, so a `float` written
    # at a `uint`'s offset is a plausible wrong number rather than an error; the probe
    # reads every field back and the values are chosen distinct.
    expected = ([3, 5, 7, 105, np.float32(0.5)]
                + [np.float32(0.25 * (index + 1)) for index in
                   range(len(family.K_SCALARS))]
                + list(range(11, 11 + len(names))))
    record[0] = tuple(expected)
    payload = np.frombuffer(record.tobytes(), dtype=np.int32).copy()
    source = "\n".join((
        "#include <metal_stdlib>", "using namespace metal;", "",
        "struct Params {", "    uint nx; uint ny; uint nz; uint n_elem; float dtdx;",
        *[f"    float {scalar};" for scalar in family.K_SCALARS],
        coefficient_pack.params_fields(family.PACKED_VECTORS), "};", "",
        "kernel void params_readback(",
        "    device float*    out [[buffer(0)]],",
        "    constant Params& prm [[buffer(1)]],",
        "    uint idx [[thread_position_in_grid]])", "{",
        "    if (idx != 0) { return; }",
        "    out[0] = float(prm.nx); out[1] = float(prm.ny);",
        "    out[2] = float(prm.nz); out[3] = float(prm.n_elem);",
        "    out[4] = prm.dtdx;",
        *[f"    out[{5 + index}] = prm.{scalar};"
          for index, scalar in enumerate(family.K_SCALARS)],
        *[f"    out[{5 + len(family.K_SCALARS) + index}] = float(prm.off_{name});"
          for index, name in enumerate(names)],
        "}", ""))
    library = compile_source(source)
    out = torch.zeros(5 + len(family.K_SCALARS) + len(names),
                      dtype=torch.float32, device="mps:0")
    params = torch.from_numpy(payload).to(torch.device("mps:0"))
    library.params_readback(out, params)
    got = out.cpu().numpy().tolist()
    want = [float(v) for v in expected]
    return {"passed": got == want, "expected": want, "read_back": got,
            "fields": ["nx", "ny", "nz", "n_elem", "dtdx", *family.K_SCALARS,
                       *names]}


def leg_pack_bytes() -> Dict[str, Any]:
    """The packed buffer is the separate vectors' words end to end.

    A pack is a SECOND COPY of a coefficient, so the copy is compared as uint32 words
    -- never allclose -- and every ``Params`` offset is checked against the layout the
    plan wrote. A pack whose offsets are one member out is a converged, smooth, wrong
    absorber rather than a failure, which is why this is a leg and not a comment.
    """
    rows: Dict[str, Any] = {}
    for name, keywords in CASES:
        driver = build_driver(keywords, 2024)
        residency = Residency()
        plan = build_weld(driver, residency)
        suffix = metal_launch.SUB_STEPS["step_D"]["suffix"]
        vectors = ([np.asarray(getattr(driver.pml, f"{stem}_{axis}{suffix}"),
                               dtype=np.float32)
                    for axis in "xyz" for stem in ("kms", "sinv")]
                   + [np.asarray(getattr(driver.pml, f"kps_{axis}"), dtype=np.float32)
                      for axis in "xyz"])
        expected = np.concatenate([v.reshape(-1) for v in vectors])
        held = residency.host(f"{family.FAMILY}:cpack")
        cpack_words = differing(expected, np.asarray(held).reshape(-1))

        params = plan.static_args[-1].cpu().numpy().tobytes()
        struct = ([("nx", "<u4"), ("ny", "<u4"), ("nz", "<u4"), ("n_elem", "<u4"),
                   ("dtdx", "<f4")]
                  + [(scalar, "<f4") for scalar in family.K_SCALARS]
                  + coefficient_pack.record_dtype_fields(family.PACKED_VECTORS))
        decoded = np.frombuffer(params, dtype=np.dtype(struct))[0]
        running = 0
        offsets_ok = True
        for member, vector in zip(family.PACKED_VECTORS, vectors):
            offsets_ok = offsets_ok and int(decoded[f"off_{member}"]) == running
            running += int(vector.size)
        rows[name] = {
            "cpack_differing_words": int(cpack_words),
            "cpack_elements": int(expected.size),
            "offsets_are_the_layouts": bool(offsets_ok),
            "pack_holds_the_sum_of_the_extents": int(expected.size) == running,
            "grid_shape": list(driver.grid.shape),
            "n_elem": int(decoded["n_elem"]),
            "n_elem_is_the_grid": int(decoded["n_elem"])
            == int(np.prod(driver.grid.shape)),
        }
        log(f"    pack_bytes {name}: cpack={cpack_words} "
            f"elements={rows[name]['cpack_elements']}")
    return {
        "passed": all(row["cpack_differing_words"] == 0
                      and row["offsets_are_the_layouts"]
                      and row["pack_holds_the_sum_of_the_extents"]
                      and row["n_elem_is_the_grid"]
                      for row in rows.values()),
        "cases": rows,
        "comparator": "uint32 words, never allclose",
    }


def leg_mutants_compile(codes: Sequence[int],
                        periodic_codes: Sequence[int]) -> Dict[str, Any]:
    """Every armed defect and the byte-neutral control COMPILE."""
    rows: Dict[str, Any] = {}
    armed = dict(shader_mutations(codes))
    armed.update(bfast_mutations(codes))
    for name, (source, _why) in armed.items():
        ok, why = _compiles(source)
        rows[name] = {"compiled": ok, "error": why}
    for name, (source, _why) in periodic_wrap_mutations(periodic_codes).items():
        ok, why = _compiles(source)
        rows[name] = {"compiled": ok, "error": why}
    for name, (source, _why, _control) in predicted_null_mutations(codes).items():
        ok, why = _compiles(source)
        rows[name] = {"compiled": ok, "error": why}
    ok, why = _compiles(byte_neutral_source(codes))
    rows["byte_neutral_control"] = {"compiled": ok, "error": why}
    return {"passed": all(row["compiled"] for row in rows.values()),
            "armed": len(rows), "mutants": rows}


# ---------------------------------------------------------------------------
# Device legs
# ---------------------------------------------------------------------------

def leg_product(steps: int) -> Dict[str, Any]:
    """The four-reference identity on every fixture, complete driver steps."""
    rows: Dict[str, Any] = {}
    for index, (name, keywords) in enumerate(CASES, start=1):
        started = time.perf_counter()
        driver = build_driver(keywords, 1000 + index)
        rows[name] = run_product(driver, steps)
        log(f"    product {index}/{len(CASES)} {name}: "
            f"passed={rows[name]['passed']} "
            f"words={rows[name]['words_compared']} "
            f"steps={rows[name]['steps_compared']} "
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
    """One HOST defect, launched on every named fixture; each MUST diverge."""
    rows: Dict[str, Any] = {}
    for index, case in enumerate(cases, start=1):
        driver = build_driver(dict(CASES)[case], 2000 + index)
        result = run_product(driver, steps, launcher=launcher,
                             stop_on_divergence=True)
        legs = result["arrangements"]
        rows[case] = {
            "diverged": not legs["weld"]["identical"],
            "first_divergence": legs["weld"]["first_divergence"],
            "differing_words_final": legs["weld"]["differing_words_final"],
            "differing_volumes": legs["weld"]["differing_volumes_final"],
            "launches": legs["weld"]["launches"],
        }
        log(f"    {name} {case}: diverged={rows[case]['diverged']} "
            f"at step {rows[case]['first_divergence']}")
    return {"passed": all(row["diverged"] for row in rows.values()),
            "defect": name, "why_it_must_fire": why, "cases": rows}


def leg_sync(steps: int) -> Dict[str, Any]:
    """FORCE-INSTALLED, with the flux accessor fired mid-run: the hazard, measured."""
    driver = build_driver(dict(CASES)[MUTATION_CASE], 3131)
    hazard = arrangement_weld(driver, sync_hazard=True, name="weld_hazard")
    declining = arrangement_weld(driver, sync_hazard=False, name="weld_declining")
    arrangements = {"array": Arrangement("array", None, None),
                    "weld_declining": declining, "weld_hazard": hazard}

    def hook(step: int, name: str, engine: Any) -> Any:
        if step in SYNC_STEPS:
            return {"step": step, "flux_z": float(engine.flux_in_box(2))}
        return None

    result = drive(driver, arrangements, steps, per_step_hook=hook,
                   stop_on_divergence=False)
    legs = result["arrangements"]
    hazard_rows = legs["weld_hazard"]["per_step"]
    first = legs["weld_hazard"]["first_divergence"]
    diverged_in_D = bool(first is not None and any(
        in_place_words(row) for row in hazard_rows if row["differing_words"]))
    return {
        "passed": bool(legs["weld_declining"]["identical"]
                       and legs["weld_declining"]["sync_refusals"] == len(SYNC_STEPS)
                       and legs["weld_hazard"]["sync_answered"] == len(SYNC_STEPS)
                       and first is not None and diverged_in_D
                       and result["step_error"] is None),
        "sync_steps": list(SYNC_STEPS),
        "declining_identical": legs["weld_declining"]["identical"],
        "declining_sync_refusals": legs["weld_declining"]["sync_refusals"],
        "hazard_sync_answered": legs["weld_hazard"]["sync_answered"],
        "hazard_first_divergence": first,
        "hazard_diverges_in_D_or_fu_D": diverged_in_D,
        "citation": (
            "a product installed at update_H that also advances D corrupts D on every "
            "flux_in_box and field_energy_in_box call unless it declines the "
            "update_H_synchronize consult; this row is the measurement"),
        "steps_compared": result["steps_compared"],
    }


def leg_launch_structure(steps: int) -> Dict[str, Any]:
    """Launches per step, at the seam and over the whole step, two counters each."""
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
            "dispatched": {} if shim is None else dict(shim.dispatched),
            "absorbed": {} if shim is None else dict(shim.absorbed),
            "selected": arrangement.selected,
        }
    weld = rows["weld"]
    singles = rows["singles"]
    today = rows["composition_today"]
    step_saving = (today["launches_per_step_whole_step_plans"]
                   - weld["launches_per_step_whole_step_plans"])
    return {
        "passed": bool(weld["launches_per_step_at_the_seam"] == 1
                       and singles["launches_per_step_at_the_seam"] == 2
                       and all(row["launches_per_step_whole_step_plans"]
                               == row["launches_per_step_whole_step_functions"]
                               for row in rows.values())
                       and result["step_error"] is None),
        "arrangements": rows,
        "seam_reduction_against_the_singles":
            singles["launches_per_step_at_the_seam"]
            - weld["launches_per_step_at_the_seam"],
        "whole_step_reduction_against_the_composition_installed_today": step_saving,
        "what_this_says": (
            f"the weld saves exactly one launch against the two singles at the seam "
            f"and {step_saving:+g} against the composition the composer installs today "
            f"over the whole step. LAUNCH COUNTS ARE NOT TIME and no timing exists "
            f"for this shape"),
    }


#: THE SOURCE IS SCALED WITH THE SEED, and leaving it unscaled DISARMS the withdraw
#: leg: an O(1) dipole added to a 2^80 field rounds away entirely, so both null
#: controls come back not diverging and the leg proves nothing while reporting green.
INTEGRATED_SOURCE: Dict[str, Any] = {
    "component": "Ez", "center": (0.15, -0.1, 0.05), "size": (0.0, 0.0, 0.0),
    "frequency": 1.0, "source_type": "gaussian", "fwidth": 0.5,
    "is_integrated": True, "amplitude": 2.0 ** SEED_SCALE_BITS,
}


def leg_withdraw(steps: int) -> Dict[str, Any]:
    """An integrated electric source in the seam: hoisted, un-hoisted, after_step_D.

    THE PREDICATE REFUSES THIS CONFIGURATION TODAY, by name, because the product
    declares ``HOISTS_THE_WITHDRAW = False``; that refusal is recorded here. What this
    leg then measures is the WIRING the installer would give a product that declared
    True -- ``LeadingWithdrawPlan`` around the launch -- with the two null controls
    that must diverge.
    """
    keywords = dict(CASES)[MUTATION_CASE]
    driver = build_driver(keywords, 5151, sources=(INTEGRATED_SOURCE,))
    sources = tuple(driver._sources)  # noqa: SLF001
    refusal = family.metal_bfast_fused_hd_pair_coverage(
        driver.fields, driver.pml, sources, Residency())
    hoisted = arrangement_weld(driver, hoist=True, name="weld_hoisted")
    not_hoisted = arrangement_weld(driver, hoist=False, name="weld_not_hoisted")
    arrangements = {"array": Arrangement("array", None, None),
                    "weld_hoisted": hoisted, "weld_not_hoisted": not_hoisted}
    result = drive(driver, arrangements, steps, stop_on_divergence=False)
    legs = result["arrangements"]
    return {
        "passed": bool(not refusal.covered
                       and any("HOISTS_THE_WITHDRAW = False" in reason
                               for reason in refusal.reasons)
                       and legs["weld_hoisted"]["identical"]
                       and not legs["weld_not_hoisted"]["identical"]
                       and result["step_error"] is None),
        "predicate_refuses_this_row": not refusal.covered,
        "refusal_reasons": list(refusal.reasons)[:4],
        "hoisted_identical": legs["weld_hoisted"]["identical"],
        "not_hoisted_identical": legs["weld_not_hoisted"]["identical"],
        "not_hoisted_first_divergence": legs["weld_not_hoisted"]["first_divergence"],
        "steps_compared": result["steps_compared"],
        "what_this_licenses": (
            "the WIRING, not the flag. A product declaring HOISTS_THE_WITHDRAW True "
            "would be installed inside LeadingWithdrawPlan and would be byte-identical "
            "on this row; without that wrapper it diverges. This product declares "
            "False and its predicate refuses the row by name, which is what the "
            "board's withdraw_seam bucket records"),
    }


def leg_mutation(steps: int, codes: Sequence[int],
                 periodic_codes: Sequence[int]) -> Dict[str, Any]:
    """Every armed shader and host defect, each of which MUST diverge."""
    caught: Dict[str, Any] = {}
    predicted: Dict[str, Any] = {}

    def run_one(case: str, seed: int, source: str) -> Dict[str, Any]:
        driver = build_driver(dict(CASES)[case], seed)
        function = compile_source(source).bfast_fused_hd_pair_step
        result = run_product(driver, steps,
                             functions={shaders.CONTRACT_OFF: function},
                             stop_on_divergence=True)
        legs = result["arrangements"]
        return {
            "diverged": not legs["weld"]["identical"],
            "first_divergence": legs["weld"]["first_divergence"],
            "differing_words_final": legs["weld"]["differing_words_final"],
            "differing_volumes": legs["weld"]["differing_volumes_final"],
            "launches": legs["weld"]["launches"],
        }

    armed: List[Tuple[str, str, str, str]] = []
    for name, (source, why) in sorted(shader_mutations(codes).items()):
        armed.append((name, source, why, MUTATION_CASE))
    for name, (source, why) in sorted(bfast_mutations(codes).items()):
        armed.append((name, source, why, MUTATION_CASE))
    for name, (source, why) in sorted(periodic_wrap_mutations(periodic_codes).items()):
        armed.append((name, source, why, PERIODIC_MUTATION_CASE))

    for index, (name, source, why, case) in enumerate(armed, start=1):
        row = run_one(case, 6000 + index, source)
        row["why_it_must_fire"] = why
        row["case"] = case
        caught[name] = row
        log(f"    mutation {index}/{len(armed)} {name} on {case}: "
            f"diverged={row['diverged']} at {row['first_divergence']}")

    for index, (name, (source, why, control)) in enumerate(
            sorted(predicted_null_mutations(codes).items()), start=1):
        row = run_one(MUTATION_CASE, 7000 + index, source)
        row["why_it_cannot_fire"] = why
        row["earned_by_control"] = control
        row["control_caught"] = bool(caught.get(control, {}).get("diverged"))
        predicted[name] = row
        log(f"    predicted-null {name}: diverged={row['diverged']} "
            f"(control {control} caught={row['control_caught']})")

    host: Dict[str, Any] = {}
    for name, (launcher, why) in HOST_MUTATIONS.items():
        driver = build_driver(dict(CASES)[MUTATION_CASE], 8000 + len(host))
        result = run_product(driver, steps, launcher=launcher,
                             stop_on_divergence=True)
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
    }


def leg_byte_neutral(steps: int, codes: Sequence[int]) -> Dict[str, Any]:
    """The one armed edit required NOT to diverge."""
    driver = build_driver(dict(CASES)[MUTATION_CASE], 9191)
    function = compile_source(
        byte_neutral_source(codes)).bfast_fused_hd_pair_step
    result = run_product(driver, steps,
                         functions={shaders.CONTRACT_OFF: function})
    legs = result["arrangements"]
    return {"passed": bool(legs["weld"]["identical"] and result["passed"]),
            "identical": legs["weld"]["identical"],
            "words_compared": result["words_compared"],
            "what_this_shows": (
                "the harness does not diverge on everything: an edit that changes no "
                "arithmetic changes no word")}


def leg_disarm(steps: int) -> Dict[str, Any]:
    """The identical harness, SHIPPED bytes, must not diverge."""
    driver = build_driver(dict(CASES)[MUTATION_CASE], 9292)
    result = run_product(driver, steps)
    legs = result["arrangements"]
    return {"passed": bool(legs["weld"]["identical"] and result["passed"]),
            "identical": legs["weld"]["identical"],
            "words_compared": result["words_compared"],
            "what_this_shows": (
                "every mutation reported caught above was caught because of the "
                "mutation: the same harness with the shipped kernel is byte-identical")}


HOST_MUTATIONS: Dict[str, Tuple[Callable[[Any, Residency, Any], Any], str]] = {
    "H_and_f_w_H_written_in_place": (
        lambda plan, _residency, _pml: AliasedLaunch(plan, ("Hx", "Hy", "Hz",
                                                            "f_w_Hx", "f_w_Hy",
                                                            "f_w_Hz")),
        "the scratch bound to the live buffer: every foreign recompute then reads a "
        "cell another thread may already have stored, which is the hazard the whole "
        "shape exists to remove"),
    "f_w_H_written_in_place": (
        lambda plan, _residency, _pml: AliasedLaunch(plan, ("f_w_Hx", "f_w_Hy",
                                                            "f_w_Hz")),
        "on this side f_w_H_new[ii] == B[ii] exactly, so an in-place write hands a "
        "racing neighbour B where it needs B_prev"),
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
    """The census driver's battery hook: the identity on ONE lifted row."""
    steps = int(os.environ.get("MEEP_GPU_BFAST_HD_GATE_STEPS", 12))
    max_cells = os.environ.get("MEEP_GPU_BFAST_HD_GATE_MAX_CELLS")
    sources = tuple(getattr(driver, "_sources", ()) or ())
    block: Dict[str, Any] = {
        "family": family.FAMILY, "steps_requested": steps,
        "n_sources": len(sources),
        "sources": [{"type": type(s).__name__,
                     "field_type": str(getattr(s, "field_type", "")),
                     "is_integrated": bool(getattr(s, "is_integrated", False)),
                     "withdraw_does_work": bool(
                         withdraw_hoist._withdraw_does_work(s))}  # noqa: SLF001
                    for s in sources]}
    pin_array_path(driver)
    verdict = family.metal_bfast_fused_hd_pair_coverage(
        driver.fields, driver.pml, sources, Residency())
    block["predicate_admits"] = bool(verdict.covered)
    block["predicate_reasons"] = list(verdict.reasons)
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
        block["why_not_driven"] = (
            f"{cells} cells exceeds MEEP_GPU_BFAST_HD_GATE_MAX_CELLS={max_cells}; "
            f"refused rather than run partially")
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
    """The rows the standing census puts in this product's cell, DERIVED from the
    census's own ``plan_step.selected``."""
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
                # THE CASE THE CHILD MUST BE ASKED FOR IS NOT ALWAYS THE ROW'S NAME.
                # A `parameterized`-expanded module is lifted through
                # `parity/meep_gpu/shim/parameterized.py`, which DELIBERATELY does not
                # reproduce the upstream method name (re-deriving it would be a guess
                # deciding which recorded row a measurement belongs to) and generates
                # `{func}__idx{N}` instead; the census then matches shim case to
                # upstream row BY FACTS and records both. Asking the child for the
                # upstream name returns an EMPTY record and the parent reads it as an
                # unmeasured row -- measured 2026-09-07 on
                # tests:TestReflectanceAngular.test_reflectance_angular_2_35_7, the
                # BFAST cell's only corpus row. So the child is asked for `case` when
                # the census recorded one, and the LABEL stays the upstream row's.
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


#: The block key this gate's battery hook writes, and the key the lift leg reads back.
#: ONE spelling, so a rename cannot leave the parent reading a block the child stopped
#: writing -- which would report every row as unmeasured while every row in fact ran.
BLOCK = "metal_bfast_fused_hd_pair_gate"


def leg_lift(out_dir: Path, steps: int, max_cells: Optional[int], timeout: float,
             resume: bool, only: Optional[Sequence[str]]) -> Dict[str, Any]:
    """Every corpus row in this cell, re-lifted in its own interpreter and driven.

    The census driver's OWN mechanism, not a second one: the row is a MEEP script or
    test case, so ``measure_predicate_coverage`` lifts it in the interpreter the
    census recorded and calls :func:`evaluate` inside that process.
    """
    import contextlib  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    import gate_provenance  # noqa: PLC0415
    import measure_predicate_coverage as census  # noqa: PLC0415

    # ABSOLUTE, BEFORE ANYTHING IS BUILT FROM IT: the child runs with ``cwd`` set to
    # the lift workdir, so a relative artifact path handed to it resolves somewhere
    # that does not exist and the child dies AFTER the lift with the parent recording
    # a right-shaped row that measured nothing.
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
                "reason": (f"the MEEP corpus is not at {examples_dir} / {tests_dir}; "
                           f"set MEEP_GPU_CORPUS_ROOT to the checkout the census was "
                           f"cut over. Refused rather than measured on nothing")}
    probe_path = API_ROOT / census.PROBE
    environment = dict(os.environ)
    environment.update({"KMP_DUPLICATE_LIB_OK": "TRUE", "MPLBACKEND": "Agg",
                        "PYTHONPATH": str(API_ROOT),
                        "MEEP_GPU_BFAST_HD_GATE_STEPS": str(steps),
                        "MEEP_GPU_BFAST_HD_GATE_PROGRESS": str(progress_log)})
    if max_cells is not None:
        environment["MEEP_GPU_BFAST_HD_GATE_MAX_CELLS"] = str(max_cells)
    work = lift_dir / "workdir"
    work.mkdir(exist_ok=True)
    for entry in examples_dir.iterdir():
        if entry.suffix in (".py", ".ipynb"):
            continue
        link = work / entry.name
        if not link.exists():
            with contextlib.suppress(OSError):
                link.symlink_to(entry)
    # THE CENSUS'S OWN SHIM PATH, for the six MEEP test modules that decorate with
    # ``parameterized`` (not installed here). Added ONLY for the modules whose text
    # asks for it, on the census's own test, so no other row's imports change.
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
        environment["MEEP_GPU_BFAST_HD_GATE_LABEL"] = row["label"]
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
                log(f"lift {index}/{len(rows)} {row['label']}: REFUSED, no module "
                    f"recorded for this tests row")
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
                # STAMPED LIKE ANY OTHER PAYLOAD: this is the record a reader gets
                # when a child produced nothing, which is exactly the case where
                # "which tree was this?" is the first question.
                died = {"row": row["row"], "measured": False, "note": note,
                        "stderr_tail": stderr_text[-1500:]}
                gate_provenance.stamp(died)
                record_path.write_text(json.dumps(died, default=str),
                                       encoding="utf-8")
        payload = json.loads(record_path.read_text(encoding="utf-8"))
        candidates = payload if isinstance(payload, list) else [payload]
        # THE CHILD REPORTS THE CASE IT RAN, which on a shim-expanded module is the
        # `__idx{N}` name and not the upstream row's; match on either.
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
        # THE DENOMINATOR IS THE CELL'S: every row the standing census puts in this
        # cell must be admitted unless the seam record says it carries a standing
        # withdraw, every admitted row must drive, and every driven row must be
        # byte-identical over every step its own precondition held for.
        "passed": bool(blocks and not unmeasured
                       and refused == expected_refused
                       and driven == admitted
                       and passed_rows == driven
                       and all(blocks[label].get("steps_compared", 0)
                               >= LIFT_CLEAN_STEP_FLOOR for label in driven)),
        "facts": facts,
        "admitted": len(admitted),
        "refused": len(refused),
        "refused_labels": refused,
        "expected_refused": expected_refused,
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
COVERAGE = family.metal_bfast_fused_hd_pair_coverage


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
    log(f"mutation specialisation {MUTATION_CASE} -> codes {codes}; "
        f"periodic {PERIODIC_MUTATION_CASE} -> {PERIODIC_CODES}; "
        f"shear {BFAST_K}")

    if "driver_order" in legs:
        record("driver_order", leg_driver_order())
    if "arm_pairing" in legs:
        record("arm_pairing", leg_arm_pairing())
    if "transcription" in legs:
        record("transcription", leg_transcription(codes))
    if "refusal" in legs:
        record("refusal", leg_refusal())
    if "arbitration" in legs:
        record("arbitration", leg_arbitration())
    if "seed_scale" in legs:
        record("seed_scale", leg_seed_scale())
    if "purity" in legs:
        record("purity", leg_purity())
    if "ghost_observability" in legs:
        record("ghost_observability", leg_ghost_observability())
    if "tustin_scalars" in legs:
        record("tustin_scalars", leg_tustin_scalars())
    if "state_census" in legs:
        record("state_census", leg_state_census())
    if "policy" in legs:
        record("policy", leg_policy())
    if "binding_ceiling" in legs:
        record("binding_ceiling", leg_binding_ceiling())
    if "pack_bytes" in legs:
        record("pack_bytes", leg_pack_bytes())
    if "mutants_compile" in legs:
        record("mutants_compile", leg_mutants_compile(codes, PERIODIC_CODES))
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
                                                 PERIODIC_MUTATION_CASE]))
    if "sync" in legs:
        record("sync", leg_sync(args.steps))
    if "launch_structure" in legs:
        record("launch_structure", leg_launch_structure(args.steps))
    if "withdraw" in legs:
        record("withdraw", leg_withdraw(args.steps))
    if "lift" in legs:
        record("lift", leg_lift(args.out.parent, args.lift_steps,
                                args.lift_max_cells, args.lift_timeout,
                                args.lift_resume,
                                (args.lift_only.split(",") if args.lift_only
                                 else None)))
    if "byte_neutral" in legs:
        record("byte_neutral", leg_byte_neutral(args.steps, codes))
    if "mutation" in legs:
        record("mutation", leg_mutation(args.steps, codes, PERIODIC_CODES))
    if "disarm" in legs:
        record("disarm", leg_disarm(args.steps))

    ran = tuple(dict.fromkeys(row["leg"] for row in rows))
    complete = all(leg in ran for leg in ALL_LEGS)
    all_passed = bool(rows) and all(row.get("passed") for row in rows)
    if complete:
        verdict = "PASS" if all_passed else "FAIL"
    else:
        verdict = ("INCOMPLETE: not every leg ran -- "
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
            "byte_neutral_controls": 1 if "byte_neutral" in ran else 0,
            "lift_rows_in_cell": (lift_row.get("facts") or {}).get("rows_in_cell"),
            "lift_rows_admitted": lift_row.get("admitted"),
            "lift_rows_driven": lift_row.get("driven"),
            "lift_rows_driven_and_identical": lift_row.get("driven_and_identical"),
            "lift_words_compared": lift_row.get("words_compared"),
        },
        "mutation_case": MUTATION_CASE,
        "periodic_mutation_case": PERIODIC_MUTATION_CASE,
        "bfast_scaled_k": list(BFAST_K),
        "references": {
            "1": "the array path under the driver's own loop",
            "2": "the certified singles: bfast_curl.plan_bfast_run_constitutive('H') "
                 "then bfast_curl.plan_bfast_pml_curl('step_D')",
            "3": "plan_step(fuse=True): the composition the composer installs today",
            "4": "plan_step(fuse=False): the same slots dispatched unfused",
        },
        "signature": {"pointers": family.PACKED_POINTERS,
                      "packed_bindings": family.PACKED_BINDINGS,
                      "unpacked_bindings": family.UNPACKED_POINTER_BINDINGS,
                      "separate_scalar_bindings": family.SEPARATE_SCALAR_BINDINGS,
                      "ceiling": MAX_BUFFER_BINDINGS,
                      "headroom": MAX_BUFFER_BINDINGS - family.PACKED_BINDINGS},
        "product": {"family": family.FAMILY, "slot": family.SLOT,
                    "replaces": list(family.REPLACES), "seam": family.SEAM,
                    "installable": family.INSTALLABLE,
                    "hoists_the_withdraw": family.HOISTS_THE_WITHDRAW,
                    "carries_deposit_repair": family.CARRIES_DEPOSIT_REPAIR,
                    "weld_owed": family.WELD_OWED,
                    "absorb_row": metal_launch.FUSED_PAIR_ARMS.get(family.FAMILY),
                    "registered_in_FAMILY_MODULES": _in_family_modules()},
        "corpus": {"census": CENSUS, "seam_record": SEAM_RECORD,
                   "cell_arms": list(CELL_ARMS)},
        "subnormal_policy": subnormal.mps_policy_report(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "metal_frontend": metal_frontend_version(),
        "jsonl": str(jsonl),
        "what_this_does_not_license": (
            "installing the product (the composer refuses it by name and the "
            "arbitration leg pins that), flipping HOISTS_THE_WITHDRAW (the withdraw "
            "leg licenses the WIRING, not the flag), any throughput claim (launch "
            "counts are not time, no timing exists for this shape, and another Metal "
            "lane shared this GPU during the run), complex storage under BFAST, a "
            "FOLDED BFAST run (a different cell and a different product), an inactive "
            "absorber, a conductive or beta or cylindrical run, or any step past a "
            "row's own first banded step"),
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


def _in_family_modules() -> bool:
    from meep_gpu.metal_kernels import registry  # noqa: PLC0415

    return (family.FAMILY in registry.FAMILY_MODULES
            or Path(family.__file__).stem in registry.FAMILY_MODULES)


if __name__ == "__main__":
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))
