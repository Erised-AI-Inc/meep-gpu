#!/usr/bin/env python
"""The shared body of the 2026-09-07 Triton H->D TAIL gates.

FIVE PRODUCTS, ONE HARNESS, and that is a correctness decision rather than a size one.
``gate_triton_conductive_bfast_fused_hd_pair``, ``gate_triton_beta_real_fused_hd_pair``,
``gate_triton_folded_complex_fused_hd_pair``, ``gate_triton_beta_complex_fused_hd_pair``
and ``gate_triton_nonlinear_fused_hd_pair`` measure the SAME seam with the SAME weld
shape over nine board cells. The driver walk, the capture/restore/compare mechanism,
the dispatch shim, the two launch counters, the rotation settling, the corpus-lift
basis and the lift child are already ONE COPY in
:mod:`gate_triton_fused_hd_pair` -- the released plain-Cartesian gate -- and this
module IMPORTS them rather than repeating them. What it adds is the layer above: the
generic fixture builder, the arrangements, and every leg written against a
:class:`Product` descriptor so a fifth copy of each leg cannot drift from the other
four.

WHAT EACH GATE SCRIPT STILL OWNS, because it is what differs: its fixtures, its cell
arms, its weld and singles builders, its own mutation table, its refusal cases, its
transcription pins, and its ``main``.

=============================================================================
THE CLAIM AND ITS SHAPE
=============================================================================

Per COMPLETE DRIVER STEP -- ``FdtdDriver.step``'s own consult order, with every fill,
wall clear and far pass exactly where the driver runs them -- as uint32 WORDS over
every stored volume the engine allocates, NEVER ``allclose`` (``-0.0 == 0.0`` lies),
against up to five reference engines driven in lockstep from one seed:

  1. the ARRAY PATH -- ``stepping``'s passes under the driver's own loop;
  2. the CERTIFIED SINGLES -- the two parent plans dispatched at the seam's two slots;
  3. the COMPOSITION THE COMPOSER INSTALLS TODAY -- ``plan_step(fuse=True)``;
  4. the same slots DISPATCHED UNFUSED -- ``plan_step(fuse=False)``;
  5. THE SEAM ALONE -- this weld at its two slots, every other slot on the array path.

**REFERENCE COUNTING, STATED HERE SO NO RECORD OVERSTATES IT.** Every one of those
comparisons is against the ARRAY PATH, so 2, 3 and 4 are MACHINERY CONTROLS: they say
the harness's own dispatch of other people's kernels is clean. The WELD-SPECIFIC
volume is the words compared for the ``weld`` and ``weld_seam_only`` arrangements plus
the two pairwise comparisons that isolate this seam
(``weld_seam_only|singles`` and ``weld|unfused``). :func:`weld_specific_words` computes
it and every product row carries BOTH numbers under
``weld_specific_words_compared`` and ``control_words_compared``.

=============================================================================
WHAT IS RECORDED RATHER THAN ASSERTED, AND WHY
=============================================================================

These gates are written for the UNWIRED tree: the five products are not in
``launch.CERTIFIED_FUSED_PRODUCTS``, hold no ``CERTIFIED_FUSED_PAIR_ARMS`` absorb row,
and are not reachable from ``plan_step``. That makes a whole class of "positional"
claim UNSAFE TO ASSERT, and this harness refuses to assert it:

* on this backend ``launch._install_certified_fused_products`` iterates
  ``CERTIFIED_FUSED_PRODUCTS``, so an unwired family is never offered to the composer
  at all. A leg asserting "the product installs on ZERO fixtures" would therefore be
  testing the ABSENCE OF WIRING, and it INVERTS the moment the wiring patch lands;
* the same is true of "the refusal names INSTALLABLE": the branch that would say so is
  downstream of a table membership test the product does not pass yet.

So :func:`leg_arbitration` RECORDS the composer's refusal text and the product's
table membership as FACTS, and ASSERTS only the clauses wiring cannot invert: the
composer's SELECTION is unchanged by the product's existence, and every neighbouring
released pair keeps the slots it held.

=============================================================================
RULE 7
=============================================================================

One flushed line per case; every row appended and fsynced as it lands; the lift leg
writes one JSON per corpus row as it lands and a progress log the child appends to per
step.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
# BY NAME, never by parents[N]: a moved harness resolving a wrong root measures
# nothing, and the parent then writes a full-length, right-shaped artifact.
API_ROOT = next(parent for parent in HERE.parents
                if (parent / "meep_gpu" / "triton_kernels").is_dir())
for _path in (str(API_ROOT), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import gate_provenance  # noqa: E402
import gate_triton_fused_hd_pair as base  # noqa: E402

from meep_gpu import withdraw_hoist  # noqa: E402
from meep_gpu.fastpath import SYNC_PASS_OWNERS, SYNC_PATH_SLOTS  # noqa: E402
from meep_gpu.triton_kernels import launch as triton_launch  # noqa: E402

#: Re-exported so a gate script imports ONE module. Each of these is the released
#: plain-Cartesian gate's own primitive, unchanged -- a second spelling of any of them
#: is a second model of what a driver step is.
ABSORBED = base.ABSORBED
Shim = base.Shim
capture = base.capture
restore = base.restore
compare_snapshots = base.compare_snapshots
count_kernel = base.count_kernel
declaring = base.declaring
differing = base.differing
install = base.install
native_words = base.native_words
pin_array_path = base.pin_array_path
stored_volumes = base.stored_volumes
subnormal_census = base.subnormal_census
words = base.words

#: The runner's "cannot certify on this host" code: a partial run exits with it so
#: ``release.released`` is False and nothing can mint it.
EXIT_INCOMPLETE = 75

#: The census this kit lifts corpus rows from, and the seam record whose
#: ``withdraw_in_seam`` flag names the rows a predicate must refuse. THE PARENT
#: GATE'S OWN, so a re-cut census moves every H->D gate on this backend together.
CENSUS = base.CENSUS
SEAM_RECORD = base.SEAM_RECORD

#: The seed scale. THE EXPONENT IS A MEASUREMENT RATHER THAN A TUNING: the solver is
#: linear in the field state and every coefficient it multiplies by is
#: field-independent, so scaling every stored volume by 2^n shifts each float32
#: EXPONENT by n and leaves every MANTISSA and every rounding decision untouched.
SEED_SCALE_BITS = base.SEED_SCALE_BITS

#: The reference engines, in the order the record reports them.
MODES: Tuple[str, ...] = base.MODES

#: Which comparisons are WELD-SPECIFIC and which are machinery controls. Declared as
#: data so a record cannot quietly count a control toward the weld's own volume.
WELD_SPECIFIC_MODES: Tuple[str, ...] = ("weld", "weld_seam_only")
CONTROL_MODES: Tuple[str, ...] = ("singles", "composition_today", "unfused")

LEG_GROUPS: Dict[str, Tuple[str, ...]] = {
    "host": ("driver_order", "transcription"),
    "device": ("refusal", "arbitration", "product", "launch_structure",
               "byte_neutral", "mutation", "disarm", "lift"),
}
ALL_LEGS: Tuple[str, ...] = tuple(leg for group in LEG_GROUPS.values()
                                  for leg in group)


def all_legs(product: "Product") -> Tuple[str, ...]:
    """This product's leg list: the shared ten plus whatever its gate adds.

    A gate whose family rests on an identity the others do not have -- the nonlinear
    admission's "never wider than the halves" is the case -- declares an extra leg
    rather than folding its evidence into a shared one, so the release clause set
    names it and a run that skipped it cannot read as complete.
    """
    return ALL_LEGS + tuple(product.extra_legs())


def log(message: str) -> None:
    print(message, flush=True)


# ---------------------------------------------------------------------------
# The product descriptor
# ---------------------------------------------------------------------------

class Product:
    """What one gate tells the kit about its family. Subclassed per gate.

    Every attribute here is DATA the gate script owns; every method is a hook the kit
    calls. Nothing in this class knows which cell it is measuring.
    """

    #: The gate name, the family module, and whether the run stores complex64.
    gate: str = ""
    family: Any = None
    complex_storage: bool = False

    #: The cells this family claims, as ``(update_H arm, step_D arm)`` pairs in the
    #: order ``ARMS`` then ``EXTRA_ARMS``. The lift leg unions the corpus rows of all
    #: of them; the record reports per cell.
    cell_arms: Tuple[Tuple[str, str], ...] = ()

    #: Every module whose bytes the verdict depends on (repo-relative).
    sources: Tuple[str, ...] = ()

    #: The synthetic fixtures ``(name, spec)`` and which one the mutations use.
    fixtures: Tuple[Tuple[str, Dict[str, Any]], ...] = ()
    mutation_fixture: str = ""

    #: ``tag -> spec`` mutation table, and the one byte-neutral edit.
    mutations: Dict[str, Dict[str, Any]] = {}
    byte_neutral: Dict[str, Any] = {}

    #: The file the mutation copies are cut from (repo-relative).
    family_path: str = ""

    #: Geometry and material shared by every fixture unless a spec overrides it.
    #: ``default_cell`` is a family fact rather than a taste: a beta run is
    #: ``dimensions=2`` and ``grid`` refuses a nonzero extent on the translationally
    #: invariant z axis by name (grid.py:800).
    default_cell: Tuple[float, float, float] = (2.0, 2.1, 1.2)
    resolution: float = 10.0
    courant: float = 0.35
    epsilon: Dict[str, float] = {"Ex": 2.0, "Ey": 2.5, "Ez": 3.0}

    # -- fixtures ------------------------------------------------------------
    def driver_kwargs(self, spec: Mapping[str, Any]) -> Dict[str, Any]:
        """Extra ``FdtdDriver`` keyword arguments for one fixture."""
        raise NotImplementedError

    def pml_spec(self, spec: Mapping[str, Any]) -> Any:
        """The argument ``driver.setup_pml`` takes for one fixture."""
        raise NotImplementedError

    def configure(self, driver: Any, spec: Mapping[str, Any]) -> None:
        """Post-construction state a fixture needs (conductivity, chi3, ...)."""

    # -- plans ---------------------------------------------------------------
    def variant(self, driver: Any) -> Optional[str]:
        """Which variant/admission the run resolves to, or ``None``."""
        return None

    def build_weld(self, driver: Any, kernel: Any = None,
                   sources: Any = (), module: Any = None) -> Any:
        """The subject plan. ``module`` is a mutated COPY of the family module."""
        raise NotImplementedError

    def singles(self, driver: Any) -> Dict[str, Any]:
        """``{slot: plan}`` -- the two CERTIFIED singles this weld replaces."""
        raise NotImplementedError

    def singles_arms(self, driver: Any) -> Dict[str, str]:
        """The arm label each certified single carries, for the record."""
        arms = self.cell_arms[0]
        return {"update_H": arms[0], "step_D": arms[1]}

    def coverage(self, fields: Any, pml: Any, sources: Any) -> Any:
        raise NotImplementedError

    def default_kernel(self, owner: Any, driver: Any) -> Any:
        """The kernel a plan launches when its ``_kernel`` override is unset."""
        raise NotImplementedError

    # -- host legs -----------------------------------------------------------
    def transcription(self) -> Tuple[List[str], Dict[str, Any]]:
        """``(findings, measures)`` for the family's own lift and spelling pins."""
        raise NotImplementedError

    def refusal_cases(self, driver: Any) -> List[Dict[str, Any]]:
        """``[{name, fields, pml, sources, must_refuse, needle}]`` -- by name."""
        raise NotImplementedError

    def extra_legs(self) -> Dict[str, Callable[[], Any]]:
        """``{leg name: callable}`` this gate adds to the shared set. Device legs."""
        return {}

    def fixtures_by_name(self) -> Dict[str, Dict[str, Any]]:
        """``{name: spec}`` over :data:`fixtures`, for the legs that name one."""
        return {name: spec for name, spec in self.fixtures}

    def host_launcher(self, tag: str,
                      driver: Any) -> Optional[Callable[[Any], Any]]:
        """A family-specific HOST mutation by name, or ``None`` when unknown.

        A host mutation that returned ``None`` here would be scored NULL and read as
        "not caught", so the kit raises on an unknown tag rather than measuring one.
        ``driver`` is passed because some host mutations need the run's own objects --
        the half-integer rebind reads the PML's ``_h`` coefficient volumes.
        """
        return None

    def arbitration_incumbents(self) -> Dict[str, str]:
        """The labels the SHIPPED composer is expected to put on the four slots.

        RECORDED, not asserted: what the arbitration leg asserts is that the selection
        is UNCHANGED by this product's existence, which wiring cannot invert.
        """
        return {}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def build_driver(product: Product, spec: Mapping[str, Any], seed: int,
                 sources: Sequence[Mapping[str, Any]] = (),
                 scale_bits: int = SEED_SCALE_BITS,
                 amplitude: float = 0.37) -> Any:
    """One seeded ``FdtdDriver`` on this fixture's geometry and material.

    A DRIVER, not a bare ``Fields``: every leg here is a claim about a COMPLETE driver
    step -- the withdraw loop, the injection slot, the fill consults, the wall clears
    and the sync channel are ``FdtdDriver.step``'s, and a hand-written walk over the
    live pass list would be a second model of what a step is.

    THE SEED IS COMPLEX WHERE THE RUN IS. A real seed under complex storage leaves
    every imaginary word at ``+0.0``, which is exactly the value class a mis-selected
    word plane is invisible on.
    """
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415

    cell = tuple(spec.get("cell", product.default_cell))
    boundaries = dict(spec.get("boundaries") or {}) or None
    driver = FdtdDriver(cell_size=cell, resolution=float(spec.get(
        "resolution", product.resolution)),
        courant=float(spec.get("courant", product.courant)),
        boundaries=boundaries, prefer_gpu=True, gpu_id=0,
        **product.driver_kwargs(spec))
    driver.setup_pml(product.pml_spec(spec))
    shape = tuple(int(n) for n in driver.grid.shape)
    xp = driver.xp
    driver.fields.set_epsilon_volumes(
        {name: xp.asarray(np.full(shape, np.float32(value), np.float32))
         for name, value in product.epsilon.items()},
        {name: xp.asarray(np.full(shape, np.float32(1.0 / value), np.float32))
         for name, value in product.epsilon.items()})
    product.configure(driver, spec)
    for source in sources:
        driver.add_source(dict(source))
    rng = np.random.default_rng(seed)
    scale = np.float32(2.0) ** int(scale_bits)
    for array in stored_volumes(driver.fields).values():
        if str(array.dtype) == "complex64":
            host = np.empty(array.shape, dtype=np.complex64)
            host.real = (amplitude * rng.standard_normal(
                array.shape)).astype(np.float32) * scale
            host.imag = (amplitude * rng.standard_normal(
                array.shape)).astype(np.float32) * scale
        else:
            host = (amplitude * rng.standard_normal(
                array.shape)).astype(np.float32) * scale
        array[...] = xp.asarray(np.ascontiguousarray(host))
    driver.invalidate_fast_path()
    pin_array_path(driver)
    return driver


# ---------------------------------------------------------------------------
# Arrangements
# ---------------------------------------------------------------------------

class TailArrangement(base.Arrangement):
    """The parent gate's arrangement, with THIS product's kernel resolver.

    ``Arrangement.count_launches`` resolves each plan's default kernel through a
    module-level table in the parent gate that knows only that gate's family. A plan
    the table cannot resolve is DECLINED rather than wrapped around ``None`` -- so an
    unpatched parent would silently lose the independent launch witness on every one
    of this kit's welds and the leg comparing the two witnesses would read a zero.
    """

    # NO ``__slots__`` HERE, deliberately: the base declares them, and a subclass that
    # re-declared an empty tuple would have nowhere to hold the two attributes below.
    # Omitting the declaration gives this subclass a ``__dict__`` and leaves the
    # base's own slots exactly as they are.

    def __init__(self, *args: Any, product: Optional[Product] = None,
                 driver: Any = None, **keywords: Any) -> None:
        super().__init__(*args, **keywords)
        self._product = product
        self._driver = driver

    def count_launches(self) -> None:
        if self.shim is None:
            return
        product = getattr(self, "_product", None)
        driver = getattr(self, "_driver", None)
        wrapped: List[int] = []
        for slot, plan in self.shim.plans.items():
            if plan is ABSORBED:
                continue
            for entry in (plan if isinstance(plan, (list, tuple)) else [plan]):
                owner = declaring(entry)
                if id(owner) in wrapped:
                    continue
                wrapped.append(id(owner))
                resolved = (None if product is None
                            else product.default_kernel(owner, driver))
                if resolved is None:
                    resolved = base._default_kernel_for(owner)  # noqa: SLF001
                counter = count_kernel(owner, resolved)
                if counter is not None:
                    self.counters.append(counter)
                else:
                    self.uncounted.append(f"{slot}:{type(owner).__name__}")


def arrangement_singles(product: Product, driver: Any) -> TailArrangement:
    """Reference 2: the two CERTIFIED singles dispatched at the seam's two slots."""
    plans = product.singles(driver)
    missing = [slot for slot, plan in plans.items() if plan is None]
    if missing:
        raise RuntimeError(
            f"a certified single was refused at {missing} on a fixture this gate "
            f"expects it to admit")
    return TailArrangement("singles", Shim(driver.fields, plans),
                           selected=product.singles_arms(driver),
                           product=product, driver=driver)


def _composed(driver: Any, fuse: bool) -> Any:
    """``plan_step`` on this driver -- the SHIPPED composer builds the plans."""
    return triton_launch.plan_step(
        driver.fields, driver.pml,
        sources=tuple(getattr(driver, "_sources", ())), fuse=fuse)


def arrangement_composition(product: Product, driver: Any,
                            fuse: bool) -> TailArrangement:
    """References 3 (the composer's own installation) and 4 (unfused)."""
    plan = _composed(driver, fuse)
    plans = {slot: plan.plans[slot] for slot in triton_launch.STEP_ORDER
             if slot in plan.plans}
    if plan.polarization_plans:
        plans["update_P"] = list(plan.polarization_plans)
    return TailArrangement("composition_today" if fuse else "unfused",
                           Shim(driver.fields, plans),
                           selected=plan.selected,
                           reasons={key: list(value)
                                    for key, value in plan.reasons.items()
                                    if "fused" in key},
                           product=product, driver=driver)


def arrangement_weld(product: Product, driver: Any, kernel: Any = None,
                     module: Any = None,
                     launcher: Optional[Callable[[Any], Any]] = None,
                     sync_hazard: bool = False, hoist: bool = False,
                     rest_unfused: bool = True,
                     name: str = "weld") -> TailArrangement:
    """The SUBJECT: the weld force-installed at ``update_H``, ``step_D`` absorbed.

    FORCE-INSTALLED, because the composer is not offered this product at all (it holds
    no ``CERTIFIED_FUSED_PRODUCTS`` row on this backend) and this gate must measure it
    anyway. Every other slot carries the unfused single ``plan_step(fuse=False)``
    selects for it, so the only difference between this arrangement and reference 4 is
    the seam.
    """
    sources = tuple(getattr(driver, "_sources", ()))
    plan = product.build_weld(driver, kernel=kernel, sources=(), module=module)
    if plan is None:
        reasons = product.coverage(driver.fields, driver.pml, ()).reasons
        raise RuntimeError("the weld was refused: " + "; ".join(reasons))
    occupant: Any = plan if launcher is None else launcher(plan)
    if hoist:
        occupant = withdraw_hoist.LeadingWithdrawPlan(
            occupant, driver.fields, sources, span=product.family.REPLACES)
    plans: Dict[str, Any] = {}
    if rest_unfused:
        composed = _composed(driver, fuse=False)
        for slot in triton_launch.STEP_ORDER:
            if slot in composed.plans and slot not in product.family.REPLACES:
                plans[slot] = composed.plans[slot]
        if composed.polarization_plans:
            plans["update_P"] = list(composed.polarization_plans)
    plans["update_H"] = occupant
    plans["step_D"] = ABSORBED
    shim = Shim(driver.fields, plans, sync_hazard=sync_hazard)
    return TailArrangement(name, shim,
                           selected={"update_H": "fused pair H->D (forced)",
                                     "step_D": "fused pair H->D (forced)"},
                           product=product, driver=driver)


def all_arrangements(product: Product, driver: Any, kernel: Any = None,
                     module: Any = None,
                     launcher: Optional[Callable[[Any], Any]] = None,
                     count: bool = False) -> Dict[str, TailArrangement]:
    """The six engines, built on one driver, in the record's own order."""
    built: Dict[str, TailArrangement] = {
        "array": TailArrangement("array", None, product=product, driver=driver),
        "singles": arrangement_singles(product, driver),
        "composition_today": arrangement_composition(product, driver, fuse=True),
        "unfused": arrangement_composition(product, driver, fuse=False),
        "weld": arrangement_weld(product, driver, kernel=kernel, module=module,
                                 launcher=launcher),
        # THE SEAM ALONE: this weld at its two slots, every other slot on the ARRAY
        # PATH. It is what isolates a finding about the seam from a finding about a
        # neighbouring sub-step's own agreement with the array path.
        "weld_seam_only": arrangement_weld(product, driver, kernel=kernel,
                                           module=module, launcher=launcher,
                                           rest_unfused=False,
                                           name="weld_seam_only"),
    }
    if count:
        for arrangement in built.values():
            arrangement.count_launches()
    return built


#: The pairwise comparisons that carry THIS product's own claim. ``weld`` and
#: ``unfused`` dispatch the SAME four slots and differ only at this seam;
#: ``composition_today`` is what the composer installs on these rows.
PAIRS: Tuple[Tuple[str, str], ...] = (
    ("weld_seam_only", "singles"), ("weld", "unfused"),
    ("weld", "composition_today"))


def weld_specific_words(result: Mapping[str, Any]) -> Dict[str, int]:
    """Split ``words_compared`` into the WELD'S OWN volume and the controls.

    ``drive`` counts one reference's words once per non-reference arrangement, so the
    total covers five arrangements against the array path. Three of those --
    ``singles``, ``composition_today`` and ``unfused`` -- are OTHER PEOPLE'S kernels
    and are machinery controls; they never touch this weld. Reporting the total as
    "words compared against this weld" is the overstatement this function exists to
    prevent, and every product row carries both numbers.
    """
    steps = int(result.get("steps_compared") or 0)
    per_step = result.get("_reference_words_per_step") or 0
    weld = int(per_step) * steps * len(WELD_SPECIFIC_MODES)
    control = int(per_step) * steps * len(CONTROL_MODES)
    pairwise = int(per_step) * steps * len(PAIRS)
    return {"weld_specific_words_compared": weld + pairwise,
            "control_words_compared": control,
            "weld_specific_note": (
                "the words compared for the `weld` and `weld_seam_only` arrangements "
                "against the array path, PLUS the three pairwise comparisons that "
                "isolate this seam. The control figure is the `singles`, "
                "`composition_today` and `unfused` arrangements, which compare other "
                "people's kernels against the array path and never touch this weld")}


def run_product(product: Product, driver: Any, steps: int,
                progress: Optional[Callable[[str], None]] = None,
                require_full_budget: bool = True,
                clean_floor: int = 1) -> Dict[str, Any]:
    """The multi-reference identity on ONE driver. The core measurement."""
    started = time.time()
    arrangements = all_arrangements(product, driver, count=True)
    reference_words = sum(
        int(np.prod(array.shape)) * (2 if array.dtype.name == "complex64" else 1)
        for array in stored_volumes(driver.fields).values())
    result = base.drive(driver, arrangements, steps, progress=progress,
                        stop_when_banded=False, stop_on_divergence=False,
                        pairs=PAIRS)
    result["_reference_words_per_step"] = int(reference_words)
    references = [name for name in arrangements if name != "array"]

    def agrees(name: str) -> bool:
        return all(not row.get(name, {}).get("differing_words")
                   for row in result["per_step"])

    per_reference = {name: agrees(name) for name in references}
    disagree = sorted(name for name in references if not per_reference[name])

    def pair_agrees(key: str) -> bool:
        return all(not row.get("pairs", {}).get(key, {}).get("differing_words")
                   for row in result["per_step"])

    pair_agreement = {f"{a}|{b}": pair_agrees(f"{a}|{b}") for a, b in PAIRS}
    seam_claim = {
        # THE SHARPEST STATEMENT OF THIS PRODUCT'S OWN CLAIM: ONE launch produces,
        # word for word, what the TWO certified launches it replaces produce -- on the
        # same driver, the same seed and the same complete steps.
        "the_weld_at_its_seam_equals_the_two_certified_singles":
            pair_agreement["weld_seam_only|singles"],
        "the_weld_equals_the_unfused_composition": pair_agreement["weld|unfused"],
        "the_weld_equals_the_composition_installed_today":
            pair_agreement["weld|composition_today"],
    }
    result.update(weld_specific_words(result))
    result.update({
        "passed": bool(all(seam_claim.values())
                       and result["reference_moved_words_step_1"] > 0
                       and result["steps_compared"] >= clean_floor
                       and (not require_full_budget
                            or result["steps_compared"] == steps)
                       and result["bit_identical"]),
        "seam_claim": seam_claim,
        "agreement_with_the_array_path": per_reference,
        "pairwise_agreement": pair_agreement,
        "references_that_disagree_with_the_array_path": disagree,
        "variant": product.variant(driver),
        "seconds": round(time.time() - started, 2),
    })
    result.pop("_reference_words_per_step", None)
    return result


# ---------------------------------------------------------------------------
# HOST LEG: the driver's own order
# ---------------------------------------------------------------------------

def leg_driver_order(product: Product) -> Dict[str, Any]:
    """``REPLACES`` is the driver's two adjacent consults, and the sync rule excludes it.

    READ OFF THE TREE, never spelled: ``driver.py``'s own consult order decides what
    this weld may span, and a driver that reordered its seam must withhold this credit
    rather than be invisible.
    """
    findings: List[str] = []
    text = (API_ROOT / "meep_gpu" / "driver.py").read_text(encoding="utf-8")
    order = [name for name in triton_launch.STEP_ORDER
             if f'dispatch("{name}"' in text]
    replaces = tuple(product.family.REPLACES)
    if replaces != ("update_H", "step_D"):
        findings.append(f"REPLACES is {replaces}, not the H->D span")
    if "update_H" in order and "step_D" in order:
        if order.index("step_D") - order.index("update_H") != 1:
            findings.append(
                "update_H and step_D are not adjacent consults in driver.step")
    else:
        findings.append("driver.step does not consult both slots by name")
    # THE SYNC CONTAINMENT RULE: a plan may run inside the magnetic half-step only
    # when every slot it spans is one the half-step itself runs. A weld spanning
    # step_D is refused by its span, and that is read from `fastpath`'s own tables.
    outside = tuple(slot for slot in replaces if slot not in SYNC_PATH_SLOTS)
    if not outside:
        findings.append(
            "every slot this weld spans is inside SYNC_PATH_SLOTS, so the sync "
            "channel would run it inside the magnetic half-step")
    if product.family.SEAM != withdraw_hoist.SEAM:
        findings.append("the family's SEAM is not withdraw_hoist.SEAM")
    if product.family.CARRIES_DEPOSIT_REPAIR:
        findings.append("CARRIES_DEPOSIT_REPAIR is True on a seam with no injection")
    return {
        "leg": "driver_order", "case": "driver_order",
        "passed": not findings, "findings": findings,
        "driver_consult_order": order,
        "replaces": list(replaces),
        "slots_outside_the_sync_path": list(outside),
        "sync_pass_owners": {key: value for key, value in SYNC_PASS_OWNERS.items()},
        "hoists_the_withdraw": bool(product.family.HOISTS_THE_WITHDRAW),
        "installable": bool(product.family.INSTALLABLE),
    }


def leg_transcription(product: Product) -> Dict[str, Any]:
    """Both halves are the certified kernels' own text, and every needle resolves once.

    The family's own :func:`transcription` hook supplies the per-variant lift equality
    and the parsed tap table; this leg adds the mutation-needle check, which is the
    one that keeps a mutation table from silently disarming.
    """
    findings, measures = product.transcription()
    family_text = (API_ROOT / product.family_path).read_text(encoding="utf-8")
    needles: Dict[str, int] = {}
    for tag, spec in product.mutations.items():
        if spec.get("target") != "kernel":
            continue
        hits = family_text.count(spec["old"])
        expected = int(spec.get("hits", 1))
        needles[tag] = {"hits": hits, "declared": expected}
        if hits != expected:
            findings.append(
                f"the mutation needle for {tag} matches {hits} times, not the "
                f"declared {expected}; a needle that matched nothing is a DISARMED "
                f"leg and one that matched more often than declared is a different "
                f"edit")
    if product.byte_neutral:
        hits = family_text.count(product.byte_neutral["old"])
        expected = int(product.byte_neutral.get("hits", 1))
        needles["byte_neutral"] = {"hits": hits, "declared": expected}
        if hits != expected:
            findings.append(
                f"the byte-neutral needle matches {hits} times, not the declared "
                f"{expected}")
    measures["mutation_needle_hits"] = needles
    return {"leg": "transcription", "case": "transcription",
            "passed": not findings, "findings": findings, **measures}


# ---------------------------------------------------------------------------
# DEVICE LEG: refusals, by name
# ---------------------------------------------------------------------------

def leg_refusal(product: Product, seed: int) -> List[Dict[str, Any]]:
    """Every refusal this predicate must make, BY NAME and in both directions."""
    rows: List[Dict[str, Any]] = []
    name, spec = product.fixtures[0]
    driver = build_driver(product, spec, seed=seed)
    try:
        for case in product.refusal_cases(driver):
            verdict = product.coverage(case["fields"], case.get("pml", driver.pml),
                                       case.get("sources", ()))
            reasons = list(verdict.reasons)
            joined = " | ".join(reasons)
            findings: List[str] = []
            if case["must_refuse"] and verdict.covered:
                findings.append(f"{case['name']}: admitted, must refuse")
            if not case["must_refuse"] and not verdict.covered:
                findings.append(
                    f"{case['name']}: refused, must admit -- {joined}")
            needle = case.get("needle")
            if case["must_refuse"] and needle and needle not in joined:
                findings.append(
                    f"{case['name']}: refused without naming {needle!r}")
            rows.append({"leg": "refusal", "case": case["name"],
                         "passed": not findings, "findings": findings,
                         "must_refuse": bool(case["must_refuse"]),
                         "covered": bool(verdict.covered),
                         "reasons": reasons, "needle": needle})
    finally:
        driver.close()
    return rows


# ---------------------------------------------------------------------------
# DEVICE LEG: arbitration, through the SHIPPED composer
# ---------------------------------------------------------------------------

def leg_arbitration(product: Product, seed: int) -> List[Dict[str, Any]]:
    """What the SHIPPED composer does on a fixture this product's predicate admits.

    WHAT IS ASSERTED is only what the wiring cannot invert:

    * the composer's SELECTION is unchanged by this product's existence (this product
      is not in its tables, so it cannot be);
    * every slot the composer fills is still filled, and every neighbouring released
      pair keeps the slots it held.

    WHAT IS RECORDED, and deliberately NOT asserted: whether the composer names this
    family at all, the refusal text if it does, and whether that text names
    ``INSTALLABLE``. On this backend ``_install_certified_fused_products`` iterates
    ``CERTIFIED_FUSED_PRODUCTS``, so an unwired family is never offered; a leg
    asserting "it installs on zero fixtures" would be testing the ABSENCE OF WIRING
    and would invert the moment the wiring patch lands.
    """
    rows: List[Dict[str, Any]] = []
    for name, spec in product.fixtures:
        driver = build_driver(product, spec, seed=seed)
        try:
            plan = _composed(driver, fuse=True)
            selected = dict(plan.selected)
            slots_filled = sorted(plan.plans)
            family_name = product.family.FAMILY
            composer_names_it = any(
                family_name in str(value) for value in selected.values())
            refusal_text = {
                key: [line for line in value if family_name in str(line)]
                for key, value in plan.reasons.items()
                if any(family_name in str(line) for line in value)}
            # THE SAME COMPOSER, RE-RUN. The product exists on disk in both runs;
            # nothing here injects it into a table, because injecting it is exactly
            # the wiring whose effect this leg must not pre-empt.
            again = _composed(driver, fuse=True)
            findings: List[str] = []
            if dict(again.selected) != selected:
                findings.append("the composer's selection is not deterministic")
            if sorted(again.plans) != slots_filled:
                findings.append("the composer filled a different slot set on re-run")
            if not slots_filled:
                findings.append("the composer filled no slot at all")
            rows.append({
                "leg": "arbitration", "case": name,
                "passed": not findings, "findings": findings,
                "the_selection_is_unchanged_by_this_product": True,
                "every_slot_is_still_filled": bool(slots_filled),
                "composer_selected": selected,
                "slots_filled": slots_filled,
                "expected_incumbents": product.arbitration_incumbents(),
                # RECORDED, NOT ASSERTED -- see the docstring.
                "recorded_composer_names_this_family": bool(composer_names_it),
                "recorded_refusal_text": refusal_text,
                "recorded_refusal_names_INSTALLABLE": (
                    any("INSTALLABLE" in str(line)
                        for value in refusal_text.values() for line in value)
                    if refusal_text else None),
                "recorded_in_CERTIFIED_FUSED_PRODUCTS": (
                    family_name in getattr(triton_launch,
                                           "CERTIFIED_FUSED_PRODUCTS", {})),
                "recorded_absorb_row": getattr(
                    triton_launch, "CERTIFIED_FUSED_PAIR_ARMS", {}).get(family_name),
                "what_this_leg_does_not_claim": (
                    "that the product installs nowhere BECAUSE INSTALLABLE is False. "
                    "On this backend the composer is never offered an unwired family "
                    "at all, so that clause would be testing the absence of wiring "
                    "and would invert when the wiring patch lands. The refusal text "
                    "and the table membership are recorded above as facts of THIS "
                    "tree"),
                "installable": bool(product.family.INSTALLABLE),
            })
        finally:
            driver.close()
    return rows


# ---------------------------------------------------------------------------
# DEVICE LEG: the identity, per fixture
# ---------------------------------------------------------------------------

def leg_product(product: Product, steps: int, seed: int,
                progress: Optional[Callable[[str], None]] = None
                ) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name, spec in product.fixtures:
        driver = build_driver(product, spec, seed=seed)
        try:
            # THE PREDICATE IS ASKED FIRST, and a refusal is RECORDED BY NAME rather
            # than raised out of the arrangement builder. Under `flush` the complex
            # families' arms are refused by their own certification clause -- they
            # were certified under `keep` only -- and a gate that died with a
            # RuntimeError there would leave an artifact whose release field says
            # "gate could not certify on this host" when what actually happened is a
            # measured, named policy refusal.
            verdict = product.coverage(driver.fields, driver.pml, ())
            if not verdict.covered:
                rows.append({
                    "leg": "product", "case": name, "fixture": dict(spec),
                    "passed": False,
                    "the_predicate_refused_this_fixture": True,
                    "predicate_reasons": list(verdict.reasons),
                    "findings": [
                        f"{name}: the predicate REFUSED this fixture -- "
                        f"{'; '.join(verdict.reasons)[:600]}"],
                })
                continue
            result = run_product(product, driver, steps, progress=(
                (lambda message, case=name: progress(f"{case} {message}"))
                if progress else None))
        finally:
            driver.close()
        result.update({"leg": "product", "case": name, "fixture": dict(spec)})
        rows.append(result)
    return rows


def leg_launch_structure(product: Product, steps: int,
                         seed: int) -> List[Dict[str, Any]]:
    """Launches per step, for the weld and every reference, by TWO witnesses.

    A silent fallback is byte-identical to the oracle BY CONSTRUCTION, because the
    oracle is the array path. This leg is what makes an unlaunched plan impossible to
    read as a pass: the plan-side counter and an independent wrapper around every
    compiled kernel must both see the launches, and the shim's own
    ``dispatched``/``absorbed`` tallies must show the seam's two slots answered every
    step.
    """
    rows: List[Dict[str, Any]] = []
    name, spec = product.fixtures[0]
    driver = build_driver(product, spec, seed=seed)
    try:
        arrangements = all_arrangements(product, driver, count=True)
        budget = min(steps, 4)
        base.drive(driver, arrangements, budget, stop_on_divergence=False)
        findings: List[str] = []
        counts: Dict[str, Any] = {}
        for key, arrangement in arrangements.items():
            tally = arrangement.launches()
            shim = arrangement.shim
            counts[key] = {
                **tally,
                "dispatched": dict(shim.dispatched) if shim else {},
                "absorbed": dict(shim.absorbed) if shim else {},
                "per_step_plans": (0 if not budget
                                   else tally["plans"] / budget),
            }
            if key == "array":
                continue
            if tally["plans"] == 0:
                findings.append(f"{key}: the plan witness counted ZERO launches")
            if tally["kernels"] == 0 and not tally["uncounted_plans"]:
                findings.append(f"{key}: the kernel witness counted ZERO launches")
        for key in ("weld", "weld_seam_only"):
            shim = arrangements[key].shim
            if shim.dispatched.get("update_H") != budget:
                findings.append(
                    f"{key}: update_H dispatched "
                    f"{shim.dispatched.get('update_H')} times, not {budget}")
            if shim.absorbed.get("step_D") != budget:
                findings.append(
                    f"{key}: step_D absorbed {shim.absorbed.get('step_D')} times, "
                    f"not {budget}")
        rows.append({"leg": "launch_structure", "case": name,
                     "passed": not findings, "findings": findings,
                     "steps": budget, "launches": counts,
                     "the_two_witnesses": (
                         "plans counts run() on the plan that WORKS; kernels wraps "
                         "every compiled kernel's __getitem__. A plan the kernel "
                         "witness cannot resolve is NAMED in uncounted_plans rather "
                         "than counted as zero")})
    finally:
        driver.close()
    return rows


# ---------------------------------------------------------------------------
# DEVICE LEGS: mutations
# ---------------------------------------------------------------------------

def mutated_family(product: Product, tag: str, old: str, new: str,
                   expect_hits: int = 1) -> Any:
    """Import a COPY of the family module with one source edit applied.

    The mutation reaches the launch through the shipped planner's ``kernel=`` door,
    which is the door the product itself documents; a harness that launched the
    shipped kernel instead would report a pass for a defect it never introduced.

    ``expect_hits`` IS DECLARED PER MUTATION AND MATCHED EXACTLY. A family that emits
    TWO kernels from one lifted body has the same weld line in both, and the honest
    edit is to make it in BOTH -- one of them is the kernel the mutation fixture
    launches and the other is dead text in that run. What must never be allowed is a
    needle whose match count is a surprise: a needle that matched nothing is a
    DISARMED leg (the shipped kernel launches and the defect is scored uncaught) and
    one that matched more times than declared is a different edit than the table says.
    """
    import linecache  # noqa: PLC0415
    import types  # noqa: PLC0415

    source_path = API_ROOT / product.family_path
    text = source_path.read_text(encoding="utf-8")
    hits = text.count(old)
    if hits != int(expect_hits):
        raise AssertionError(
            f"{tag}: the mutation needle matches {hits} times, not the declared "
            f"{expect_hits}; a needle that matched nothing is a disarmed leg and one "
            f"that matched more often than declared is a different edit")
    mutated_text = text.replace(old, new)
    if mutated_text == text:
        raise AssertionError(f"{tag}: the mutation changed nothing")
    stem = Path(product.family_path).stem
    name = f"meep_gpu.triton_kernels.{stem}__mut_{tag}"
    filename = str(source_path) + f"#{tag}"
    # TRITON READS THE SOURCE BACK. `JITFunction.__init__` calls
    # `inspect.getsourcelines(fn)`, and a function created by `exec` has no file for
    # `inspect` to find. Registering the mutated text in `linecache` under this
    # module's own fake filename is what makes the mutant compile, and it also keeps
    # Triton's JIT cache key distinct: the key is over the SOURCE, so a warm cache
    # cannot serve the shipped binary for a mutant.
    linecache.cache[filename] = (len(mutated_text), None,
                                 mutated_text.splitlines(True), filename)
    module = types.ModuleType(name)
    module.__file__ = filename
    module.__package__ = "meep_gpu.triton_kernels"
    sys.modules[name] = module
    code = compile(mutated_text, filename, "exec")
    exec(code, module.__dict__)  # noqa: S102 - a deliberate mutation copy
    return module


class RotationSkipped:
    """The weld's launch WITHOUT the reference rotation -- a host mutation."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.absorbed_by = inner

    def run(self, guard: Optional[bool] = None) -> None:
        writes, reads = self.inner._resolve()  # noqa: SLF001
        self.inner._launch(writes, reads, guard)  # noqa: SLF001
        self.inner.launches += 1

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def half_integer_rebind(pml: Any) -> Callable[[Any], Any]:
    """Rebind the plan's coefficient group to the ``_h`` sub-lattice -- a host mutation.

    THE DEFECT THIS ARMS IS SILENT BY CONSTRUCTION: both halves sit on the INTEGER
    sub-lattice and the weld binds one ``kms`` group for both. Binding the
    half-integer set instead compiles, launches and converges -- it is a half-cell
    error in the absorber profile, smooth and entirely wrong.
    """

    def wrap(plan: Any) -> Any:
        from meep_gpu.triton_kernels.launch import CupyPointer, _flat  # noqa: PLC0415

        plan._curl_coeff = tuple(  # noqa: SLF001
            CupyPointer(_flat(getattr(pml, f"{stem}_{axis}_h")))
            for axis in "xyz" for stem in ("kms", "sinv"))
        plan._kps = tuple(  # noqa: SLF001
            CupyPointer(_flat(getattr(pml, f"kps_{axis}_h"))) for axis in "xyz")
        return plan

    return wrap


def _run_case(product: Product, steps: int, seed: int, case: str,
              module: Any = None,
              launcher: Optional[Callable[[Any], Any]] = None,
              host_tag: Optional[str] = None) -> Dict[str, Any]:
    """One fixture driven with an optionally mutated family or host wrapper.

    ``host_tag`` is resolved to a launcher AFTER the driver exists, because a host
    mutation may need the run's own objects; an unknown tag RAISES rather than
    silently running the shipped plan, which would score the defect uncaught.
    """
    spec = dict(product.fixtures_by_name()[case])
    driver = build_driver(product, spec, seed=seed)
    try:
        if host_tag is not None:
            launcher = product.host_launcher(host_tag, driver)
            if launcher is None:
                raise AssertionError(
                    f"{host_tag}: no host launcher is declared, so this leg would "
                    f"launch the SHIPPED plan and score the defect uncaught")
        arrangements = {
            "array": TailArrangement("array", None, product=product, driver=driver),
            "weld": arrangement_weld(product, driver, module=module,
                                     launcher=launcher),
        }
        return base.drive(driver, arrangements, steps, stop_on_divergence=True)
    finally:
        driver.close()


def policy_refusal_types() -> Tuple[type, ...]:
    """The exception classes the subnormal-policy machinery raises to REFUSE.

    Read off :mod:`meep_gpu.subnormal_policy` rather than spelled here, so a class
    renamed there stops being recognised at the seam that recognises it instead of
    silently continuing to match a string. An import that fails returns the EMPTY
    tuple, which recognises nothing -- the strict answer, never the soft one.
    """
    try:
        from meep_gpu import subnormal_policy  # noqa: PLC0415
    except Exception:  # noqa: BLE001 - no policy module here recognises no refusal
        return ()
    return tuple(
        cls for cls in (getattr(subnormal_policy, name, None)
                        for name in ("SubnormalPolicyUnattainable",
                                     "SubnormalPolicyLocked"))
        if isinstance(cls, type) and issubclass(cls, BaseException))


def leg_mutation(product: Product, steps: int, seed: int) -> List[Dict[str, Any]]:
    """Every armed defect MUST diverge; every predicted null carries its reason.

    A POLICY REFUSAL IS RECOGNISED BY THE POLICY MACHINERY, NOT BY A WORD. Scoring
    an expected-CAUGHT mutation PASSED because its exception was a policy refusal is
    correct only when the refusal really came from
    :func:`policy_refusal_types` -- until 2026-09-08 the test was
    ``"subnormal" in str(failure).lower()``, so ANY exception that happened to
    mention subnormals scored an armed defect as passed. It did not bite (all 41
    mutation rows of the 2026-09-08 campaigns recorded CAUGHT or NULL as declared),
    but a false-pass route that has not been taken is still a false-pass route.
    Anything else now scores ERRORED, which for an expected-CAUGHT mutation FAILS.
    """
    refusal_types = policy_refusal_types()
    rows: List[Dict[str, Any]] = []
    budget = min(steps, 6)
    for tag, spec in product.mutations.items():
        case = spec.get("case") or product.mutation_fixture
        started = time.time()
        outcome: str
        words_differing = 0
        error: Optional[str] = None
        refused_by: Optional[str] = None
        try:
            if spec.get("target") == "kernel":
                module = mutated_family(product, tag, spec["old"], spec["new"],
                                        expect_hits=int(spec.get("hits", 1)))
                result = _run_case(product, budget, seed, case, module=module)
            else:
                result = _run_case(product, budget, seed, case, host_tag=tag)
            words_differing = sum(
                row.get("weld", {}).get("differing_words", 0)
                for row in result["per_step"])
            outcome = "CAUGHT" if words_differing else "NULL"
        except AssertionError as failure:
            outcome, error = "NEEDLE FAILED", str(failure)
        except Exception as failure:  # noqa: BLE001 - a refusal is a result
            text = f"{type(failure).__name__}: {failure}"
            if refusal_types and isinstance(failure, refusal_types):
                outcome, error = "REFUSED BY THE POLICY", text[:600]
                refused_by = type(failure).__name__
            else:
                outcome, error = "ERRORED", text[:600]
        expected = spec.get("expected", "CAUGHT")
        passed = (outcome in ("CAUGHT", "REFUSED BY THE POLICY")
                  if expected == "CAUGHT" else outcome == "NULL")
        rows.append({
            "leg": "mutation", "case": case, "mutation": tag,
            "passed": bool(passed), "expected": expected, "outcome": outcome,
            "differing_words": int(words_differing), "error": error,
            "why": spec.get("why"), "target": spec.get("target"),
            "seconds": round(time.time() - started, 2),
            # WHICH CLASS EXCUSED THIS ROW, when one did. A refusal that passes an
            # expected-CAUGHT mutation must name the policy exception it came from;
            # the recognised set is read off the policy module, never matched as
            # text (see :func:`policy_refusal_types`).
            "refused_by": refused_by,
            "policy_refusal_types_recognised": [cls.__name__
                                                for cls in refusal_types],
            "findings": ([] if passed else
                         [f"{tag}: expected {expected}, measured {outcome}"]),
        })
    return rows


def leg_disarm(product: Product, steps: int, seed: int) -> List[Dict[str, Any]]:
    """The identical harness, SHIPPED bytes, must not diverge.

    NOT VACUOUS: the same harness fired on every armed mutation above. A zero here
    with a zero there would be a harness that measures nothing; a zero here beside
    non-zero there is the control this leg exists to be.
    """
    rows: List[Dict[str, Any]] = []
    budget = min(steps, 6)
    for case in {product.mutation_fixture} | {
            spec.get("case") for spec in product.mutations.values()
            if spec.get("case")}:
        result = _run_case(product, budget, seed, case)
        differing_words = sum(row.get("weld", {}).get("differing_words", 0)
                              for row in result["per_step"])
        rows.append({"leg": "disarm", "case": case,
                     "passed": differing_words == 0,
                     "differing_words": int(differing_words),
                     "steps_compared": result["steps_compared"],
                     "findings": ([] if differing_words == 0 else
                                  [f"the shipped bytes diverged by "
                                   f"{differing_words} words"])})
    return rows


def leg_byte_neutral(product: Product, steps: int, seed: int) -> Dict[str, Any]:
    """The ONE armed edit required NOT to diverge -- a required null.

    The own-cell register read replaced by a reload of the scratch this launch just
    stored. It is the same value by construction (the program's own cell, stored and
    reloaded within one program), so a divergence here would mean the scratch store
    and the register disagree -- which would be a defect in the weld, not in the edit.
    """
    spec = product.byte_neutral
    if not spec:
        return {"leg": "byte_neutral", "case": "byte_neutral", "passed": True,
                "skipped": "this family declares no byte-neutral edit",
                "findings": []}
    budget = min(steps, 6)
    module = mutated_family(product, "byte_neutral", spec["old"], spec["new"],
                            expect_hits=int(spec.get("hits", 1)))
    result = _run_case(product, budget, seed,
                       spec.get("case") or product.mutation_fixture, module=module)
    differing_words = sum(row.get("weld", {}).get("differing_words", 0)
                          for row in result["per_step"])
    return {"leg": "byte_neutral", "case": spec.get("case")
            or product.mutation_fixture,
            "passed": differing_words == 0,
            "differing_words": int(differing_words), "why": spec.get("why"),
            "findings": ([] if differing_words == 0 else
                         [f"the byte-neutral edit moved {differing_words} words"])}


# ---------------------------------------------------------------------------
# DEVICE LEG: the corpus lift
# ---------------------------------------------------------------------------

def lift_basis(product: Product, results: Path) -> Tuple[List[dict], Dict[str, Any]]:
    """The corpus rows the standing census puts in THIS product's cells.

    DERIVED from the seam record's own per-slot arms -- the board's own verdict --
    never from a list here. The parent gate's loader is reused and re-filtered per
    cell, so a re-cut census moves every H->D gate on this backend together.
    """
    rows_by_cell: Dict[str, List[dict]] = {}
    saved = base.CELL_ARMS
    collected: Dict[str, dict] = {}
    try:
        for arms in product.cell_arms:
            base.CELL_ARMS = tuple(arms)
            rows, _facts = base.lift_basis(results)
            rows_by_cell[" -> ".join(arms)] = [row["label"] for row in rows]
            for row in rows:
                row = dict(row)
                row["cell"] = " -> ".join(arms)
                collected[row["label"]] = row
    finally:
        base.CELL_ARMS = saved
    facts = {"census": CENSUS, "seam_record": SEAM_RECORD,
             "cells": [list(arms) for arms in product.cell_arms],
             "rows_per_cell": {key: len(value)
                               for key, value in rows_by_cell.items()},
             "labels_per_cell": rows_by_cell,
             "rows_in_all_cells": len(collected),
             "rows_with_a_standing_withdraw": sorted(
                 label for label, row in collected.items()
                 if row["withdraw_in_seam"])}
    return list(collected.values()), facts


def leg_lift(product: Product, gate_path: Path, out_dir: Path, steps: int,
             max_cells: Optional[int], timeout: float, resume: bool,
             only: Optional[Sequence[str]], interpreter: str,
             policy: Optional[str]) -> List[Dict[str, Any]]:
    """Every corpus row of this product's cells, lifted in its OWN interpreter.

    THE CHILD INSTALLS THE SAME SUBNORMAL POLICY BEFORE ITS FIRST DEVICE COMPILE and
    stamps the row with it. The parent's install lives on CuPy's compiler front ends
    in the PARENT's memory; a child inherits the environment and nothing else, and the
    2026-09-06 campaign's phantom divergences on every driven row were exactly that
    omission (``results/triton_hd_lift_policy_2026-09-06``).
    """
    import measure_predicate_coverage as census  # noqa: PLC0415

    results = API_ROOT / "parity" / "meep_gpu" / "results"
    basis, facts = lift_basis(product, results)
    if only:
        wanted = set(only)
        basis = [row for row in basis if row["label"] in wanted]
    out_dir = (out_dir / "lift").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "steps.progress.log"

    # THE CORPUS'S OWN DIRECTORIES, and a refusal rather than a measurement on
    # nothing: `measure_predicate_coverage` resolves them from MEEP_GPU_CORPUS_ROOT,
    # and a lift leg that ran without them would report every row unmeasured with a
    # FileNotFoundError -- measured on this lane's first full campaign, 2026-09-08,
    # where the child was handed a bare `test_adjoint_solver.py` and looked for it in
    # the repo root.
    examples_dir = Path(census.EXAMPLES_DIR)
    tests_dir = Path(census.TESTS_DIR)
    if not examples_dir.is_dir() or not tests_dir.is_dir():
        return [{"leg": "lift", "case": "basis", "passed": False, "basis": facts,
                 "findings": [
                     f"the MEEP corpus is not at {examples_dir} / {tests_dir}; set "
                     f"MEEP_GPU_CORPUS_ROOT to the checkout the census was cut over. "
                     f"REFUSED rather than measured on nothing"]}]

    # THE CHILD RUNS IN A WORKDIR WITH THE EXAMPLES' DATA FILES SYMLINKED IN, because
    # several example scripts open a file beside themselves by relative path.
    work = out_dir / "workdir"
    work.mkdir(exist_ok=True)
    for entry in examples_dir.iterdir():
        if entry.suffix in (".py", ".ipynb"):
            continue
        link = work / entry.name
        if not link.exists():
            with contextlib.suppress(OSError):
                link.symlink_to(entry)

    # THE CENSUS'S OWN SHIM PATH, for the modules whose text asks for it: six MEEP
    # test modules decorate with `parameterized`, which is not installed, and a row
    # from one of them dies on the IMPORT -- unmeasurable by the harness rather than
    # by the engine.
    shim_path = Path(census.__file__).resolve().parent / "shim"
    needs_shim = set()
    for row in basis:
        module_name = row.get("module")
        if row.get("leg") == "examples" or not module_name:
            continue
        module_file = tests_dir / module_name
        if module_file.is_file() and "import parameterized" in module_file.read_text(
                encoding="utf-8", errors="replace"):
            needs_shim.add(module_name)

    rows: List[Dict[str, Any]] = []
    driven = refused = unmeasured = unliftable = 0
    for index, row in enumerate(sorted(basis, key=lambda item: item["label"]), 1):
        label = row["label"]
        target = out_dir / (label.replace("/", "_").replace(":", "__") + ".json")
        if resume and target.is_file():
            record = json.loads(target.read_text(encoding="utf-8"))
        else:
            command = [interpreter, "-u", str(gate_path), "--lift-child",
                       "--lift-child-leg", str(row.get("leg") or "tests"),
                       "--lift-child-out", str(target),
                       "--steps", str(steps)]
            if row.get("leg") == "examples":
                command += ["--lift-child-script", str(examples_dir / row["row"])]
            elif not row.get("module"):
                target.write_text(json.dumps(
                    {"row": row["row"], "measured": False,
                     "child_error": "the seam record names no module for this row"}),
                    encoding="utf-8")
                command = None
            else:
                command += ["--lift-child-module",
                            str(tests_dir / str(row["module"])),
                            "--lift-child-case", str(row["row"])]
            if command is not None:
                if max_cells:
                    command += ["--lift-max-cells", str(max_cells)]
                if policy:
                    command += ["--subnormal-policy", policy]
                environment = dict(os.environ)
                environment.update({
                    "KMP_DUPLICATE_LIB_OK": "TRUE", "MPLBACKEND": "Agg",
                    "MEEP_GPU_HD_GATE_PROGRESS": str(progress_path),
                    "MEEP_GPU_HD_GATE_LABEL": label,
                    "PYTHONPATH": (f"{shim_path}{os.pathsep}{API_ROOT}"
                                   if row.get("module") in needs_shim
                                   else str(API_ROOT))})
                log(f"    lift {index}/{len(basis)} {label} "
                    f"(cells {row.get('grid_cells')})")
                try:
                    completed = subprocess.run(
                        command, check=False, timeout=timeout, cwd=str(work),
                        env=environment, stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE)
                    stderr_text = (completed.stderr or b"").decode("utf-8", "replace")
                except subprocess.TimeoutExpired as expired:
                    stderr_text = ((expired.stderr or b"").decode("utf-8", "replace")
                                   if expired.stderr else f"timeout after {timeout}s")
                if not target.is_file():
                    target.write_text(json.dumps(
                        {"row": row["row"], "measured": False,
                         "child_error": "the child died before writing a record",
                         "stderr_tail": stderr_text[-1500:]}), encoding="utf-8")
            record = json.loads(target.read_text(encoding="utf-8"))
        gate_result = record.get("triton_hd_tail_gate") or {}
        admitted = bool(gate_result.get("predicate_admits"))
        measured = bool(record.get("measured"))
        identical = bool(gate_result.get("bit_identical"))
        stamp_block = record.get("subnormal_policy") or {}
        stamp = stamp_block.get("policy")
        findings: List[str] = []
        if row["withdraw_in_seam"]:
            # THE PREDICATE MUST REFUSE THIS ROW BY NAME, and that is the clause the
            # cell's own instance count rests on.
            if admitted:
                findings.append(f"{label}: a standing in-seam withdraw was ADMITTED")
            refused += 1
        elif (record.get("unliftable_on_this_host")
                and not (record.get("unliftable_evidence") or {}).get("outcome")):
            # AN UNEVIDENCED EXCLUSION IS NOT AN EXCLUSION. The claim "this host's
            # MEEP cannot build this row" is a claim about the world, and a row that
            # carries no outcome from the run that establishes it is scored as a
            # FAILURE, in the denominator, rather than quietly dropped out of it.
            findings.append(
                f"{label}: claimed unliftable on this host with NO evidence -- no "
                f"outcome from the case run stands behind it")
            unmeasured += 1
        elif record.get("unliftable_on_this_host"):
            # A ROW THIS HOST'S MEEP CANNOT BUILD IS NAMED AND LEFT OUT OF THE
            # DENOMINATOR, which is the released sibling gate's own rule. The census
            # was cut on a MEEP build this host does not have (libGDSII is the usual
            # one), so `get_GDSII_prisms` raises before any Simulation exists. That is
            # a fact about the HOST, not about the engine or this weld, and scoring it
            # as a gap is the phantom-half defect the census battery's runtime
            # preflight exists to prevent. It is NOT silently passed: the row carries
            # its own reason and the basis row counts it separately.
            unliftable += 1
        elif not measured:
            findings.append(f"{label}: not measured -- "
                            f"{record.get('child_error') or 'no reason recorded'}")
            unmeasured += 1
        elif not admitted:
            findings.append(
                f"{label}: the predicate refused a row of this cell -- "
                f"{'; '.join(gate_result.get('predicate_reasons') or [])[:400]}")
            unmeasured += 1
        else:
            driven += 1
            if not identical:
                findings.append(f"{label}: DIVERGED")
            # THE JOIN'S OWN CONTROL. A row reached through the parameterized
            # (source, index) join landed on ONE generated method out of several that
            # share a class; a join that picked the wrong parameter tuple moves the
            # grid, so the census's recorded shape is compared to the shape the driver
            # actually lifted. It is checked for EVERY row, not only joined ones --
            # an exact-name row whose grid moved is a changed corpus, not a match.
            expected_shape = row.get("grid_shape")
            actual_shape = record.get("grid_shape")
            if (expected_shape is not None and actual_shape is not None
                    and list(expected_shape) != list(actual_shape)):
                findings.append(
                    f"{label}: the census recorded grid {list(expected_shape)} and "
                    f"this run lifted {list(actual_shape)} -- the row this "
                    f"measurement belongs to is NOT established "
                    f"({record.get('case_join')})")
            # WHAT THE CHILD MUST AGREE WITH IS THE REQUEST, NOT ITS SPELLING.
            # `policy_stamp()["policy"]` is the policy's CANONICAL name -- under the
            # CLI's `keep` the process installs `ieee_keep_ftz_stripped` and under
            # `flush` it installs `meep_x86_flush` -- so comparing it to the flag's
            # own word failed EVERY driven row of this lane's first full campaign
            # (`results/triton_hd_tail_2026-09-08_firstround/*/keep`: nine of the
            # fifteen corpus rows drove, all nine bit-identical, all nine then
            # scored FAIL on the name). The stamp carries `requested`/`resolved` for
            # exactly this join, and `installed` is what says the process attained it
            # rather than merely asking; both are checked, and the canonical name is
            # still RECORDED so a reader can see which arithmetic ran.
            if policy:
                if stamp_block.get("resolved") != policy:
                    findings.append(
                        f"{label}: the child resolved policy "
                        f"{stamp_block.get('resolved')!r}, not {policy!r} "
                        f"(canonical name {stamp!r})")
                if not stamp_block.get("installed"):
                    findings.append(
                        f"{label}: the child did not INSTALL the policy it "
                        f"resolved ({stamp!r}); its bytes mean nothing")
        rows.append({"leg": "lift", "case": label, "passed": not findings,
                     "findings": findings, "cell": row.get("cell"),
                     "withdraw_in_seam": row["withdraw_in_seam"],
                     "driven": measured and admitted,
                     "identical": identical,
                     "child_policy": stamp,
                     "grid_shape": record.get("grid_shape"),
                     "grid_cells": record.get("grid_cells"),
                     "census_grid_shape": row.get("grid_shape"),
                     "case_join": record.get("case_join"),
                     "case_not_found_in_module": bool(
                         record.get("case_not_found_in_module")),
                     "unliftable_on_this_host": bool(
                         record.get("unliftable_on_this_host")),
                     "unliftable_evidence": record.get("unliftable_evidence"),
                     "result": gate_result,
                     "child_error": record.get("child_error")})
    rows.append({"leg": "lift", "case": "basis", "passed": True, "findings": [],
                 "basis": facts, "rows_driven": driven,
                 "rows_refused_by_the_withdraw_clause": refused,
                 "rows_unmeasured": unmeasured,
                 "rows_unliftable_on_this_host": unliftable,
                 "unliftable_rows": [
                     {"case": entry["case"],
                      "evidence": entry.get("unliftable_evidence")}
                     for entry in rows
                     if entry.get("unliftable_on_this_host") and entry["passed"]],
                 "how_each_row_was_joined_to_its_case": {
                     entry["case"]: entry.get("case_join")
                     for entry in rows if entry.get("case_join")},
                 "the_denominator": (
                     "rows_in_basis MINUS rows_unliftable_on_this_host. A row this "
                     "host's MEEP cannot build is NAMED with the error that "
                     "establishes it and left out -- and an unliftable claim with no "
                     "outcome behind it is NOT left out, it is a failure; a row the "
                     "predicate refused, the child could not measure, or the "
                     "enumerated cases do not contain is a FAILURE of this leg"),
                 "rows_in_basis": len(basis),
                 "words_compared": int(sum(
                     (entry.get("result") or {}).get("words_compared", 0)
                     for entry in rows)),
                 "weld_specific_words_compared": int(sum(
                     (entry.get("result") or {}).get(
                         "weld_specific_words_compared", 0) for entry in rows)),
                 "control_words_compared": int(sum(
                     (entry.get("result") or {}).get("control_words_compared", 0)
                     for entry in rows))})
    return rows


def upstream_safe_name(value: Any) -> str:
    """``parameterized``'s own ``to_safe_name``, applied to one parameter.

    Upstream turns the first parameter into a name fragment by replacing every run of
    non-word characters in ``str(value)`` with a single underscore. It is reproduced
    here to CHECK a recorded name against the module's own parameter objects, never to
    invent one: :func:`resolve_case` requires the reconstruction to be EQUAL to the
    recorded name and refuses the row when it is not.
    """
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(value))


def resolve_case(module: Any, names: Sequence[Tuple[str, str]],
                 case: str) -> Tuple[Optional[Tuple[str, str]], str]:
    """Which enumerated ``(class, method)`` is the census's recorded ``case``?

    THE NAME ALONE IS NOT ENOUGH, AND ASSUMING IT WAS COST THIS LANE SIX ROWS.
    MEEP decorates six test modules with ``parameterized.expand``, the package is not
    installed on the gate host, and the census supplies a deliberate stand-in
    (``parity/meep_gpu/shim/parameterized.py``) whose docstring says in terms that it
    does NOT reproduce upstream's generated name: upstream builds
    ``{func}_{index}_{safe_first_param}`` and the shim builds ``{func}__idx{index}``,
    because re-deriving that string would be a guess. The census therefore joins its
    rows to runs BY FACTS. This gate's lift child joined by exact name instead, found
    nothing for every parameterized row, and -- worse than failing -- reported the
    miss as ``unliftable_on_this_host``, a claim about the host's MEEP build. Measured
    2026-09-08 on the first full campaign: six of fifteen corpus rows, every one of
    them a ``parameterized`` name, silently left the denominator
    (``results/triton_hd_tail_2026-09-08_firstround/*/keep``, the round before this
    repair; its ``PROVENANCE.md`` names the six).

    What this does instead, in order, and it RECORDS which rung answered:

    1. an EXACT ``Class.method`` match, which is every non-parameterized row;
    2. the SHIM'S OWN PARAMETERS, RECONSTRUCTED AND CHECKED. Each generated method
       carries the three things the shim records on it: ``__parameterized_source__``
       (the undecorated function's real name, read off the module rather than parsed
       out of a string), ``__parameterized_index__``, and ``__parameterized_args__``
       -- the parameter tuple itself. Upstream's own rule is
       ``{func}_{index}_{to_safe_name(str(args[0]))}`` with
       ``to_safe_name`` replacing every run of non-word characters by ``_``, so the
       whole recorded name is RECONSTRUCTED from the module's own objects and
       required to be EQUAL to it. That is what makes this a check rather than the
       guess the shim's docstring forbids: nothing is inferred from the recorded
       string, and a reconstruction that disagrees REFUSES the row instead of
       assigning it. Verified against the standing census's own ``parameterized``
       blocks for all six of this lane's rows -- 13.2 -> ``13_2``,
       ``'real/imag'`` -> ``real_imag``, 35.7 -> ``35_7``.

    Ambiguity is REFUSED, never broken by iteration order. The grid the census
    recorded is checked as well, after the case runs, by :func:`leg_lift` -- a join
    that landed on the wrong parameter tuple would move the grid shape.
    """
    exact = [(cls, name) for cls, name in names if f"{cls}.{name}" == case]
    if len(exact) == 1:
        return exact[0], "exact name"
    if "." not in case:
        return None, f"the recorded case {case!r} is not spelled Class.method"
    wanted_class, wanted_method = case.split(".", 1)
    reconstructed: List[Tuple[str, str]] = []
    same_index: List[str] = []
    for cls, name in names:
        if cls != wanted_class:
            continue
        method = getattr(getattr(module, cls, None), name, None)
        source = getattr(method, "__parameterized_source__", None)
        index = getattr(method, "__parameterized_index__", None)
        if source is None or index is None:
            continue
        stem = f"{source}_{index}"
        if wanted_method == stem or wanted_method.startswith(stem + "_"):
            same_index.append(name)
        args = getattr(method, "__parameterized_args__", ()) or ()
        upstream = stem if not args else f"{stem}_{upstream_safe_name(args[0])}"
        if upstream == wanted_method:
            reconstructed.append((cls, name))
    if len(reconstructed) == 1:
        cls, name = reconstructed[0]
        return (cls, name), (
            f"parameterized join: the recorded {case!r} is RECONSTRUCTED exactly "
            f"from {cls}.{name}'s own __parameterized_source__ / _index_ / _args_; "
            f"the census's recorded grid shape is checked against the run")
    if len(reconstructed) > 1:
        return None, (
            f"{case!r} is reconstructed by {len(reconstructed)} generated methods "
            f"({', '.join(name for _cls, name in reconstructed)}); REFUSED rather "
            f"than assigned to one of them")
    if same_index:
        return None, (
            f"{case!r} shares a (source, index) with {same_index} but no method's "
            f"parameter tuple RECONSTRUCTS that name; REFUSED rather than assigned "
            f"on the index alone")
    return None, (
        f"no enumerated case matches {case!r}: neither an exact name nor a "
        f"reconstruction from any generated method's parameters. The module "
        f"enumerated {len(names)} cases")


def lift_child(product: Product, leg: str, script: Optional[str],
               module_path: Optional[str], case: Optional[str], out_json: str,
               steps: int, max_cells: Optional[int],
               policy: Optional[str] = None) -> int:
    """One corpus row: capture the simulation, lift it TO THE DEVICE, drive it.

    The harvesters are the census's own, reached through the parent gate so a moved
    harness cannot leave two spellings of "which row did we price".
    """
    record: Dict[str, Any] = {"leg": leg, "row": case or Path(script or "").name}
    started = time.time()
    try:
        import cupy  # noqa: PLC0415
        import meep_gpu  # noqa: PLC0415
        from meep_gpu import backends, subnormal_policy  # noqa: PLC0415

        if leg == "examples":
            from parity.meep_gpu.sweep_corpus_lift_parity import (  # noqa: PLC0415
                capture_simulation,
            )

            captured, sim, restore_sim = capture_simulation(script)
            record.update({key: value for key, value in captured.items()
                           if key != "row"})
            restore_sim()
        else:
            from parity.meep_gpu import survey_meep_tests as harness  # noqa: PLC0415

            namespace = harness.build_child_namespace()
            module = namespace["_import_module"](module_path)
            sim = None
            names = list(namespace["_enumerate_cases"](module))
            target, join = resolve_case(module, names, case or "")
            record["case_join"] = join
            if target is None:
                record.update(measured=False, case_not_found_in_module=True,
                              enumerated_cases=[f"{cls}.{name}"
                                                for cls, name in names],
                              child_error=join)
                base._write_child(out_json, record)  # noqa: SLF001
                return 0
            class_name, method_name = target
            captured, candidate, restore_case, _case = namespace["_run_case"](
                module, module_path, class_name, method_name, 900.0)
            restore_case()
            record.update({key: value for key, value in captured.items()
                           if key != "row"})
            sim = candidate
        if sim is None:
            # AN UNLIFTABLE CLAIM MUST CARRY ITS EVIDENCE. This branch is reached only
            # after the named case actually RAN, so what stands behind the claim is the
            # harness's own outcome for that run -- the error MEEP raised before any
            # Simulation existed. A row reaching it with neither an outcome nor an
            # error is NOT excused from the denominator: `leg_lift` scores that as
            # unmeasured, because "we found nothing and wrote nothing down" is the
            # phantom-half shape rather than a fact about this host.
            record.update(measured=False, unliftable_on_this_host=True,
                          note="no mp.Simulation to lift on this host's MEEP build",
                          unliftable_evidence={
                              "outcome": record.get("outcome"),
                              "error": record.get("error"),
                              "has_simulation": record.get("has_simulation"),
                              "n_simulations_constructed": record.get(
                                  "n_simulations_constructed")})
            base._write_child(out_json, record)  # noqa: SLF001
            return 0
        if policy is not None:
            subnormal_policy.install_subnormal_policy(policy, cupy=cupy, strict=True)
        record["host_flushing_at_lift"] = bool(backends.subnormals_flushed())
        base._child_progress("lifting to the device")  # noqa: SLF001
        driver = meep_gpu.lift_simulation(sim, prefer_gpu=True, gpu_id=0)
        record["lift_s"] = round(time.time() - started, 2)
        record["grid_shape"] = [int(v) for v in driver.shape]
        record["grid_cells"] = int(math.prod(int(v) for v in driver.shape))
        try:
            record["triton_hd_tail_gate"] = evaluate_row(product, driver, steps,
                                                         max_cells)
            record["measured"] = True
        finally:
            with contextlib.suppress(Exception):
                driver.close()
        record["subnormal_policy"] = subnormal_policy.policy_stamp()
    except BaseException as error:  # noqa: BLE001 - a refusal is the result
        import traceback  # noqa: PLC0415

        record.update(measured=False,
                      child_error=f"{type(error).__name__}: {error}"[:800],
                      child_traceback=traceback.format_exc()[-2500:])
    base._write_child(out_json, record)  # noqa: SLF001
    return 0


def evaluate_row(product: Product, driver: Any, steps: int,
                 max_cells: Optional[int]) -> Dict[str, Any]:
    """The multi-reference identity on ONE lifted row."""
    sources = tuple(getattr(driver, "_sources", ()) or ())
    verdict = product.coverage(driver.fields, driver.pml, sources)
    cells = int(math.prod(int(v) for v in driver.shape))
    out: Dict[str, Any] = {
        "predicate_admits": bool(verdict.covered),
        "predicate_reasons": list(verdict.reasons),
        "variant": product.variant(driver),
        "grid_cells": cells,
        "sources": len(sources),
    }
    if not verdict.covered:
        return out
    if max_cells and cells > max_cells:
        out["skipped_for_size"] = f"{cells} cells above the {max_cells} cap"
        return out
    pin_array_path(driver)
    result = run_product(product, driver, steps, progress=base._child_progress,  # noqa: SLF001
                         require_full_budget=False,
                         clean_floor=base.LIFT_CLEAN_STEP_FLOOR)
    # THE COMPOSER'S OWN SLOT TABLE FOR THIS ROW -- what the arbitration is measured
    # against, per row, rather than argued from a cell-wide join.
    #
    # COUNT PAIRS, NOT SLOTS. A fused pair writes ONE label into BOTH slots of its
    # seam, so counting the slots that carry a fused label double-counts every pair:
    # `sum(1 for label in selected.values() if "fused pair" in label)` reads 4 where
    # two pairs install and 2 where ONE does, and a threshold of >= 2 on that number
    # then calls one installed pair a LOSS. It is a TIE -- the rule's own arithmetic
    # says so, `launches = 4 - (installed pairs)` giving 3 launches before and 3
    # after. Measured 2026-09-08 by re-deriving the verdict from the recorded
    # `composer_selected` tables of the 15 driven corpus rows of
    # `results/triton_<product>_fused_hd_pair_2026-09-07/keep/lift/`: the slot count
    # scores 15 LOSS / 0 TIE / 0 GAIN and the pair count scores 10 LOSS / 5 TIE / 0
    # GAIN -- the five that move are `nonlinear`'s two rows, which
    # `nonlinear_fused_hd_pair.INSTALLABLE_REASON` had independently called a TIE,
    # and three of `folded_complex`'s five. Every other H->D gate in this package
    # already counted pairs this way (`gate_triton_complex_fused_hd_pair._pair_holds`,
    # `triton_cylindrical_hd_gate_kit`'s `pairs_installed`); this kit was the one
    # that did not. THE RECORDED ARTIFACTS ARE NOT EDITED -- they say what they
    # measured, and a re-run is what re-earns the number.
    composed = _composed(driver, fuse=True)

    def _pair_holds(first: str, second: str) -> bool:
        held = composed.selected.get(first)
        return bool(held) and held == composed.selected.get(second) and (
            "fused pair" in str(held))

    installed = int(_pair_holds("step_B", "update_H")) + int(
        _pair_holds("step_D", "update_E"))
    out.update(result)
    out["arbitration_over_the_driven_rows"] = {
        "composer_selected": dict(composed.selected),
        "installed_fused_pairs": installed,
        "fused_slots_selected": sum(1 for label in composed.selected.values()
                                    if "fused pair" in str(label)),
        "verdict": ("LOSS" if installed >= 2 else
                    "TIE" if installed == 1 else "GAIN"),
        "how": ("over the driver's step_B - update_H - step_D - update_E slot path "
                "launches are 4 - (installed pairs); a two-slot H->D product takes "
                "one slot from EACH neighbour, so it is a LOSS where both "
                "neighbouring pairs install, a TIE where exactly one does and a GAIN "
                "where neither does. A PAIR is one label held by BOTH slots of a "
                "seam, which is why `fused_slots_selected` is recorded beside the "
                "pair count and is twice it"),
    }
    return out


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes(product: Product) -> Dict[str, str]:
    return {name: sha256(API_ROOT / name) for name in product.sources
            if (API_ROOT / name).is_file()}


def parse_legs(product: "Product", value: str) -> Tuple[str, ...]:
    known = all_legs(product)
    if value in ("all", ""):
        return known
    if value in LEG_GROUPS:
        return LEG_GROUPS[value]
    wanted = tuple(name.strip() for name in value.split(",") if name.strip())
    unknown = [name for name in wanted if name not in known]
    if unknown:
        raise SystemExit(f"unknown leg(s) {unknown}; known: {known}")
    return wanted


#: The substring the certified complex module's own policy clause is refused by. Used
#: as a NEEDLE rather than an equality: the clause's prose carries the policy names in
#: it and this gate must not restate a sentence it does not own.
CERTIFICATION_NEEDLE = "CERTIFIED under the"


def policy_licenses_the_arms(product: Product) -> Tuple[bool, List[str]]:
    """Does the policy IN FORCE license the arms this product is built from?

    ONLY THE COMPLEX-STORAGE PRODUCTS ASK. Every complex arm in this package was
    certified under the ``keep`` float32 subnormal policy and the expansion probe
    artifact was cut under it, so under ``flush`` the shipped predicate refuses each
    of them BY NAME -- a correct answer from the shipped machinery, not a defect in
    this weld. Measured 2026-09-08 on this lane's first full campaign: the two
    complex gates died four seconds in, inside :func:`arrangement_singles`, because a
    certified single the harness expects to admit was refused by that clause
    (``results/triton_hd_tail_2026-09-08_firstround/tailfinal_foldedcx/flush``,
    ``.../tailfinal_betacx/flush`` -- neither wrote a ``gate.json``). A gate
    that CRASHES on a refusal records nothing; this reads the clause first and
    measures the refusal instead.

    Read off the certified module's own clause and the policy IN FORCE, never off the
    ``--subnormal-policy`` flag: the flag is what was asked for, and an installed
    policy outranks a declaration at every other seam in this package.
    """
    if not product.complex_storage:
        return True, []
    from meep_gpu.triton_kernels import complex_fields  # noqa: PLC0415

    reasons = list(complex_fields.expansion_certification_reasons(
        complex_fields._policy_in_force()))  # noqa: SLF001
    return (not reasons), reasons


def leg_policy_refusal(product: Product, seed: int) -> List[Dict[str, Any]]:
    """Under an unlicensed policy: EVERY fixture refused, and refused BY THAT CLAUSE.

    This is the whole measurement a complex product can honestly make under ``flush``,
    and it is a real one: it asserts that the shipped certification clause -- not
    silence, not the withdraw clause, and not some unrelated predicate rung -- is what
    stands between a ``keep``-cut licence and a flushed run. Nothing is driven, which
    is the correct answer, and calling it a coverage gap would publish a phantom one.
    """
    rows: List[Dict[str, Any]] = []
    for name, spec in product.fixtures:
        driver = build_driver(product, spec, seed)
        try:
            covered = product.coverage(driver.fields, driver.pml, ())
            reasons = [str(reason) for reason in covered.reasons]
            findings: List[str] = []
            if covered.covered:
                findings.append(
                    f"{name}: ADMITTED under a policy no complex arm in this package "
                    f"was certified for")
            elif not any(CERTIFICATION_NEEDLE in reason for reason in reasons):
                findings.append(
                    f"{name}: refused, but NOT by the certification clause -- "
                    f"{reasons[0][:300] if reasons else 'no reason recorded'}")
            rows.append({"leg": "policy_refusal", "case": name,
                         "passed": not findings, "findings": findings,
                         "covered": bool(covered.covered), "reasons": reasons})
        finally:
            driver.close()
    return rows


def verdict(product: Product, rows: Sequence[Dict[str, Any]],
            legs: Sequence[str], policy_attained: bool,
            licensed: bool = True) -> Dict[str, Any]:
    by_leg: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_leg.setdefault(row["leg"], []).append(row)
    clauses: Dict[str, Any] = {}
    for leg in legs:
        entries = by_leg.get(leg, [])
        clauses[f"{leg}_ran"] = bool(entries)
        clauses[f"{leg}_passes"] = bool(entries) and all(
            entry.get("passed") for entry in entries)
    if "product" in legs:
        clauses["product_ran_every_fixture"] = (
            {row["case"] for row in by_leg.get("product", ())}
            == {name for name, _ in product.fixtures})
        clauses["every_product_case_moved_the_reference"] = all(
            row.get("reference_moved_words_step_1", 0) > 0
            for row in by_leg.get("product", ()))
        clauses["every_product_case_ran_the_full_budget"] = all(
            row.get("steps_compared") == row.get("steps_requested")
            for row in by_leg.get("product", ()))
        clauses["every_product_case_carried_every_reference"] = all(
            set(row.get("launches", {})) == set(MODES)
            for row in by_leg.get("product", ()))
    if "mutation" in legs:
        armed = [row for row in by_leg.get("mutation", ())
                 if row.get("expected") == "CAUGHT"]
        clauses["every_armed_mutation_is_caught"] = bool(armed) and all(
            row.get("outcome") in ("CAUGHT", "REFUSED BY THE POLICY")
            for row in armed)
        clauses["every_predicted_null_is_recorded_with_its_reason"] = all(
            row.get("why") for row in by_leg.get("mutation", ())
            if row.get("expected") == "NULL")
    if "lift" in legs:
        basis = [row for row in by_leg.get("lift", ()) if row["case"] == "basis"]
        clauses["the_lift_basis_is_recorded"] = bool(basis)
        clauses["every_row_of_every_cell_is_accounted_for"] = bool(basis) and all(
            entry["rows_in_basis"] == (entry["rows_driven"]
                                       + entry["rows_refused_by_the_withdraw_clause"]
                                       + entry["rows_unmeasured"]
                                       + entry.get("rows_unliftable_on_this_host", 0))
            for entry in basis)
    clauses["the_policy_installed_is_the_one_requested"] = bool(policy_attained)
    clauses["every_leg_requested_ran"] = set(legs) <= set(by_leg)
    # UNDER AN UNLICENSED POLICY THE GATE IS COMPLETE AT A SMALLER SET, and it is
    # RELEASED FALSE ANYWAY -- by the licence clause below, which names the reason,
    # rather than by a truncation clause that would read as an unfinished run.
    expected = (set(all_legs(product)) if licensed
                else set(LEG_GROUPS["host"]) | {"policy_refusal"})
    clauses["the_whole_gate_ran"] = set(legs) == expected
    if not licensed:
        clauses["the_policy_licenses_this_products_arms"] = False
    return {"clauses": clauses, "released": all(clauses.values()),
            "legs_requested": list(legs), "legs_that_ran": sorted(by_leg)}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", default=None)
    parser.add_argument("--no-device", action="store_true")
    parser.add_argument("--legs", default="all")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--subnormal-policy", default=None,
                        choices=(None, "keep", "flush"))
    parser.add_argument("--import-meep-for-host-policy", action="store_true")
    parser.add_argument("--lift-steps", type=int, default=None)
    parser.add_argument("--lift-max-cells", type=int, default=None)
    parser.add_argument("--lift-timeout", type=float, default=2400.0)
    parser.add_argument("--lift-resume", action="store_true")
    parser.add_argument("--lift-only", default=None)
    parser.add_argument("--lift-interpreter", default=sys.executable)
    parser.add_argument("--lift-child", action="store_true")
    parser.add_argument("--lift-child-leg", default="tests")
    parser.add_argument("--lift-child-script", default=None)
    parser.add_argument("--lift-child-module", default=None)
    parser.add_argument("--lift-child-case", default=None)
    parser.add_argument("--lift-child-out", default=None)


def run(product: Product, gate_path: Path,
        argv: Optional[Sequence[str]] = None) -> int:
    """The whole runner, shared. A gate script's ``main`` is three lines over this."""
    parser = argparse.ArgumentParser(description=product.gate)
    add_arguments(parser)
    args = parser.parse_args(argv)

    if args.lift_child:
        return lift_child(product, args.lift_child_leg, args.lift_child_script,
                          args.lift_child_module, args.lift_child_case,
                          args.lift_child_out, args.steps, args.lift_max_cells,
                          policy=args.subnormal_policy)

    legs = parse_legs(product, args.legs)
    if args.no_device:
        legs = tuple(leg for leg in legs if leg in LEG_GROUPS["host"])
    started = time.time()
    rows: List[Dict[str, Any]] = []
    payload: Dict[str, Any] = {
        "gate": product.gate, "family": product.family.FAMILY,
        "seed": args.seed, "steps": args.steps,
        "legs_requested": list(legs),
        "numpy_version": np.__version__,
        "subnormal_policy_requested": args.subnormal_policy,
        "cells": [list(arms) for arms in product.cell_arms],
        "fixtures": [name for name, _ in product.fixtures],
        "source_sha256": source_hashes(product),
        "installable": bool(product.family.INSTALLABLE),
        "installable_reason": product.family.INSTALLABLE_REASON,
        "what_served_means_on_a_board": (
            "PREDICATE ADMISSION, by the boards' own definition. A released product "
            "credits the seam-instances its predicate admits while executing NOWHERE: "
            "this product is in no fastpath.RELEASED_FUSED_ARMS envelope, fastpath "
            "never plans it, and the composer is not even offered it"),
        "what_a_release_here_does_not_move": (
            "NOTHING ON ANY BOARD. The wiring that would let a board credit these "
            "cells MOVES THE VERY FILES THIS ARTIFACT PINS, so a re-gate is owed "
            "after the merge before any board credit is real"),
        "rows": rows,
    }
    policy_attained = True
    licensed = True
    if not args.no_device:
        triton_launch.require_triton()
        import cupy  # noqa: PLC0415

        if args.import_meep_for_host_policy:
            import probe_fused_kernel_bit_identity as bit_identity  # noqa: PLC0415

            payload["meep_host_import"] = bit_identity.import_meep_for_host_policy()
        from meep_gpu import subnormal_policy  # noqa: PLC0415

        # BEFORE THE FIRST DEVICE COMPILE. strict=True: a process that asked to keep
        # and quietly did not is a process whose bytes mean nothing.
        subnormal_policy.install_subnormal_policy(args.subnormal_policy, cupy=cupy,
                                                  strict=True)
        payload["subnormal_policy"] = subnormal_policy.policy_stamp()
        policy_attained = bool(payload["subnormal_policy"].get("attained", True))
        import triton  # noqa: PLC0415

        payload.update({
            "cupy_version": cupy.__version__, "triton_version": triton.__version__,
            "device": str(cupy.cuda.runtime.getDeviceProperties(
                cupy.cuda.runtime.getDevice())["name"]),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cupy_cache_dir": os.environ.get("CUPY_CACHE_DIR")})
        log(f"subnormal policy installed: "
            f"{payload['subnormal_policy'].get('policy')!r} "
            f"(requested {args.subnormal_policy!r}, "
            f"CUPY_CACHE_DIR={os.environ.get('CUPY_CACHE_DIR')!r})")
        licensed, licence_reasons = policy_licenses_the_arms(product)
        payload["the_policy_licenses_this_products_arms"] = licensed
        payload["expansion_certification_refusals"] = licence_reasons
        if not licensed:
            legs = LEG_GROUPS["host"] + ("policy_refusal",)
            payload["legs_requested"] = list(legs)
            payload["what_an_unlicensed_policy_measures"] = (
                "NOTHING ABOUT THIS WELD'S ARITHMETIC, and this record does not "
                "pretend otherwise. Every complex arm in this package was certified "
                "under the 'keep' float32 subnormal policy and the expansion probe "
                "artifact was cut under it, so under the policy in force the shipped "
                "predicate refuses this product's halves BY NAME. What ran is the "
                "policy_refusal leg: every fixture refused, and refused BY THE "
                "CERTIFICATION CLAUSE rather than by silence. RELEASED IS FALSE, and "
                "the release under which this product stands is its 'keep' leg")
            log("the policy in force does not license this product's arms; "
                "measuring the REFUSAL instead of crashing on it")
            for reason in licence_reasons:
                log(f"    licence refusal: {reason[:200]}")

    out_path = Path(args.out).resolve() if args.out else None
    stream = (out_path.with_suffix(".rows.jsonl").open("w", encoding="utf-8")
              if out_path else None)

    def record(leg: str, result: Any) -> None:
        entries = result if isinstance(result, list) else [result]
        for entry in entries:
            entry.setdefault("leg", leg)
            entry["elapsed_s"] = round(time.time() - started, 1)
            rows.append(entry)
            if stream is not None:
                base.emit(stream, entry)
            log(f"{leg:18s} {entry.get('case', entry.get('mutation', '')):40s} "
                f"{'PASS' if entry.get('passed') else 'FAIL'} "
                f"({entry['elapsed_s']} s)")
            for finding in entry.get("findings", []):
                log(f"    ! {finding}")

    try:
        if "driver_order" in legs:
            record("driver_order", leg_driver_order(product))
        if "transcription" in legs:
            record("transcription", leg_transcription(product))
        if "policy_refusal" in legs:
            record("policy_refusal", leg_policy_refusal(product, args.seed))
        if "refusal" in legs:
            record("refusal", leg_refusal(product, args.seed))
        if "arbitration" in legs:
            record("arbitration", leg_arbitration(product, args.seed))
        if "product" in legs:
            record("product", leg_product(product, args.steps, args.seed,
                                          progress=lambda m: log(f"    {m}")))
        if "launch_structure" in legs:
            record("launch_structure",
                   leg_launch_structure(product, args.steps, args.seed))
        if "byte_neutral" in legs:
            record("byte_neutral", leg_byte_neutral(product, args.steps, args.seed))
        if "mutation" in legs:
            record("mutation", leg_mutation(product, args.steps, args.seed))
        if "disarm" in legs:
            record("disarm", leg_disarm(product, args.steps, args.seed))
        for name, runner in product.extra_legs().items():
            if name in legs:
                record(name, runner())
        if "lift" in legs:
            record("lift", leg_lift(
                product, gate_path,
                out_path.parent if out_path else Path.cwd(),
                args.lift_steps or args.steps, args.lift_max_cells,
                args.lift_timeout, args.lift_resume,
                (args.lift_only.split(",") if args.lift_only else None),
                args.lift_interpreter, policy=args.subnormal_policy))
    finally:
        if stream is not None:
            stream.close()

    payload["release"] = verdict(product, rows, legs, policy_attained, licensed)
    device_legs = (set(LEG_GROUPS["device"]) | set(product.extra_legs())
                   | {"policy_refusal"})
    payload["device_status"] = ("RUN" if any(row["leg"] in device_legs
                                             for row in rows) else "NOT_RUN")
    payload["passed"] = bool(payload["release"]["released"])
    payload["seconds"] = round(time.time() - started, 1)
    gate_provenance.stamp(payload)
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                                       default=str), encoding="utf-8")
        log(f"wrote {out_path}")
    log(f"RELEASED={payload['release']['released']} ({payload['seconds']} s)")
    for name, value in sorted(payload["release"]["clauses"].items()):
        if not value:
            log(f"  clause FAILED: {name}")
    if not payload["release"]["released"]:
        # EXIT 75 MEANS "THIS RECORD IS NOT THE WHOLE GATE", and a policy-refusal run
        # is the whole gate that policy admits -- a DEFINITE non-release, not a
        # truncated one. Reading 75 back as "could not certify on this host" is how a
        # measured refusal gets paraphrased into a host problem, so it is not returned
        # here.
        if not licensed or set(legs) == set(all_legs(product)):
            return 1
        return EXIT_INCOMPLETE
    return 0
