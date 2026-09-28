#!/usr/bin/env python3
"""The shared body of the three 2026-09-07 H->D tail gates.

THREE PRODUCTS, ONE HARNESS, and that is a correctness decision rather than a size
one. ``gate_metal_folded_complex_fused_hd_pair``,
``gate_metal_beta_complex_fused_hd_pair`` and ``gate_metal_beta_real_fused_hd_pair``
measure the SAME seam with the SAME weld shape over five board cells; the driver walk,
the capture/restore/compare mechanism, the dispatch shim, the two launch counters and
the rotation settling are already one copy in ``gate_metal_fused_hd_pair``, which this
module imports rather than repeats. What this module adds is the layer above it: the
generic fixture builder, the five arrangements, and every leg written against a
:class:`Product` descriptor so a third copy of each leg cannot drift from the other
two.

WHAT EACH GATE SCRIPT STILL OWNS, because it is what differs: its fixtures, its cell
arms, its parent emitters and single-plan builders, its own mutation table, and its
``evaluate`` hook (the census driver names the GATE module as its ``--battery``, so
that hook cannot live here).

=============================================================================
THE CLAIM AND ITS SHAPE
=============================================================================

Per COMPLETE DRIVER STEP -- ``FdtdDriver.step``'s own consult order, with every fill,
wall clear and far pass exactly where the driver runs them -- as uint32 WORDS over
every stored volume the engine allocates, never ``allclose`` (``-0.0 == 0.0`` lies),
against up to four reference engines from one seed:

  1. the ARRAY PATH -- ``stepping``'s passes under the driver's own loop;
  2. the CERTIFIED SINGLES -- the two parent plans dispatched at the seam's two slots;
  3. the COMPOSITION THE COMPOSER INSTALLS TODAY -- ``plan_step(fuse=True)``;
  4. the same slots DISPATCHED UNFUSED -- ``plan_step(fuse=False)``.

Five things this harness refuses to let pass silently, inherited from the sibling and
restated because each is a leg here:

1. **A silent fallback.** An unlaunched plan is byte-identical to the oracle BY
   CONSTRUCTION, because the oracle is the array path. Every case asserts the exact
   launch count from TWO independent witnesses, plus
   ``dispatched["update_H"] == steps`` and ``absorbed["step_D"] == steps``, and
   requires every compared array to have MOVED from its seed.
2. **A hollow pass.** Every armed defect must be CAUGHT, and ``disarm`` reruns the
   identical harness with the shipped bytes and requires zero divergence.
3. **A vacuous claim.** On MPS the float32 subnormal flush is native and has no lever,
   so byte identity is claimed subject to a CHECKED subnormal-free precondition taken
   on the oracle's own state BEFORE each comparison; a banded step is stepped, named
   and NOT compared.
4. **An unmeasured platform assumption.** ``binding_ceiling`` compiles the refuted
   signatures and requires each to FAIL, bisects the ceiling on this host, compiles the
   shipped 31 on every fixture's own codes, and LAUNCHES a packed probe that reads
   back every ``Params`` field.
5. **A refusal that is really an omission.** ``refusal`` names each one and drives the
   INVERTED clause in both directions.

=============================================================================
WHAT IS RECORDED RATHER THAN ASSERTED, AND WHY
=============================================================================

These gates are written for the UNWIRED tree: the three products are not in
``registry.FAMILY_MODULES``, hold no ``launch.FUSED_PAIR_ARMS`` absorb row and are not
named by any board. Every clause that depends on THAT state -- "the product is absent
from the tables", "it holds no absorb row", "the composer names it in its refusals" --
is RECORDED as an observation and never asserted, because the wiring diff this round
writes inverts exactly those, and a gate that asserted them would fail the moment the
diff lands.

What IS asserted is the substantive half, which the wiring does not move: the product
is refused BY NAME on the configurations it must refuse; the refusal names
``INSTALLABLE`` False; the composer's SELECTION on every fixture is unchanged with the
product registered; and the released neighbours keep their slots.

Rule 7: one flushed timestamped line per unit of work; every row appended and fsynced
as it lands.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

# THE POLICY IS NOT SET HERE. Each gate sets it before importing anything, because a
# `meep_gpu` module reached first freezes the resolved policy for the process; a
# harness that set a default would silently decide which policy leg a campaign ran.
HERE = Path(__file__).resolve().parent
# BY NAME, never by parents[N]: a moved harness resolving a wrong root measures nothing.
API_ROOT = next(parent for parent in HERE.parents
                if (parent / "meep_gpu" / "metal_kernels").is_dir())
for _path in (str(API_ROOT), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402

import gate_metal_fused_hd_pair as _shared  # noqa: E402

from meep_gpu.metal_kernels import launch as metal_launch  # noqa: E402
from meep_gpu.metal_kernels import subnormal, symmetry  # noqa: E402
from meep_gpu.metal_kernels.device import (  # noqa: E402
    MAX_BUFFER_BINDINGS, Residency, compile_source,
)

# The generic machinery, bound to short names so the legs read as their own.
ABSORBED = _shared.ABSORBED
Arrangement = _shared.Arrangement
CountingFunction = _shared.CountingFunction
Shim = _shared.Shim
capture = _shared.capture
compare_snapshots = _shared.compare_snapshots
declaring = _shared.declaring
differing = _shared.differing
drive = _shared.drive
install = _shared.install
needle = _shared.needle
pin_array_path = _shared.pin_array_path
restore = _shared.restore
stored_volumes = _shared.stored_volumes
synced = _shared.synced
words = _shared.words

#: The runner's "cannot certify on this host" code: a partial run exits with it so
#: ``release.released`` is False and nothing can mint it.
EXIT_INCOMPLETE = 75

#: Per-component permittivity -- ``metal_composition_matrix._epsilon``'s own values,
#: so the material is the certified families' too.
EPSILON: Dict[str, float] = {"Ex": 2.0, "Ey": 2.5, "Ez": 3.0}

#: The fixture geometry constants, ``metal_composition_matrix``'s own.
RESOLUTION = 10.0
COURANT = 0.35

#: THE SEED IS SCALED BY 2^80, and the exponent is the sibling H->D gates' MEASUREMENT
#: reused rather than re-derived: the solver is linear in the field state and every
#: coefficient it multiplies by is field-independent, so scaling every stored volume by
#: 2^n shifts each float32 EXPONENT by n and leaves every MANTISSA and every rounding
#: decision untouched. 2^80 clears a sixty-step budget with twelve decimal orders of
#: headroom under float32's finite range at the top. Leg ``seed_scale`` MEASURES the
#: claim on this product's own fixtures rather than inheriting it.
SEED_SCALE_BITS = 80

#: HOW MANY CLEAN COMPLETE STEPS A LIFTED CORPUS ROW MUST REACH TO COUNT. A FLOOR, not
#: a target: the synthetic fixture's amplitude is the gate's to choose and is placed
#: clear of the denormal band for the whole budget, while a lifted row's state is the
#: ROW's and enters the band on its own schedule. Every row carries its own step count
#: and the words it compared.
LIFT_CLEAN_STEP_FLOOR = 8

#: The marker that separates a certified shader's signature from its body. Every
#: Metal curl template ends its parameter list with this exact line, so ONE anchor
#: cuts either body and a template that stopped carrying it fails the leg rather
#: than being compared against a truncated parent.
BODY_ANCHOR = "uint idx [[thread_position_in_grid]])\n{\n"

CODE_NAMES = {symmetry.CODE_PERIODIC: "P", symmetry.CODE_METALLIC: "M",
              symmetry.CODE_MIRROR_METALLIC: "MM",
              symmetry.CODE_MIRROR_PERIODIC: "MP"}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def codes_label(codes: Sequence[int]) -> str:
    return "_".join(CODE_NAMES.get(int(code), f"?{code}") for code in codes)


def sha256_of(path: Any) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# The product descriptor — everything a leg needs that differs between the three
# ---------------------------------------------------------------------------

class Product:
    """One gate's subject, as data the legs read rather than as three copies of code.

    Every callable here is the GATE's, so this module never decides which emitter or
    which parent predicate a product uses -- it only decides what is measured.
    """

    def __init__(self, *, module: Any, coverage: Callable[..., Any],
                 planner: Callable[..., Any],
                 source: Callable[..., str],
                 parent_curl_source: Callable[..., str],
                 singles: Callable[[Any, Residency], Tuple[Any, Any]],
                 expansion: Optional[Callable[[], Optional[str]]],
                 complex_storage: bool,
                 cases: Sequence[Tuple[str, Dict[str, Any]]],
                 cell_arms: Mapping[str, Tuple[str, str]],
                 mutation_cases: Sequence[str],
                 params_probe: Callable[[], Dict[str, Any]],
                 refuted_signatures: Mapping[str, Tuple[Callable[[], str], int]],
                 shader_mutations: Mapping[str, Tuple[Any, ...]],
                 host_mutations: Mapping[str, Tuple[Any, ...]],
                 census: str, seam_record: str,
                 lift_environment_prefix: str,
                 gate_stem: str) -> None:
        self.module = module
        self.coverage = coverage
        self.planner = planner
        self.source = source
        self.parent_curl_source = parent_curl_source
        self.singles = singles
        self.expansion = expansion
        self.complex_storage = bool(complex_storage)
        self.cases = tuple(cases)
        self.cell_arms = dict(cell_arms)
        self.mutation_cases = tuple(mutation_cases)
        self.params_probe = params_probe
        self.refuted_signatures = dict(refuted_signatures)
        self.shader_mutations = dict(shader_mutations)
        self.host_mutations = dict(host_mutations)
        self.census = census
        self.seam_record = seam_record
        self.lift_environment_prefix = lift_environment_prefix
        self.gate_stem = gate_stem

    @property
    def family(self) -> str:
        return self.module.FAMILY

    def case(self, name: str) -> Dict[str, Any]:
        for label, spec in self.cases:
            if label == name:
                return spec
        raise KeyError(f"{name} is not one of {[label for label, _ in self.cases]}")


# ---------------------------------------------------------------------------
# The fixture — a REAL driver, on the composition matrix's own geometry
# ---------------------------------------------------------------------------

def pml_thickness(shape: Sequence[int], folded: Sequence[int]
                  ) -> Tuple[Tuple[int, int], ...]:
    """``metal_composition_matrix.folded``'s own rule, restated once.

    A folded axis takes its absorber on the FAR face only -- ``(0, 2)`` -- because the
    near face is the mirror plane; an axis too short to hold a two-cell layer takes
    none, because a thickness that does not fit is a ``ValueError`` from the engine
    rather than an inactive layer.
    """
    out: List[Tuple[int, int]] = []
    for index in range(3):
        if int(shape[index]) < 6:
            out.append((0, 0))
        elif index in set(folded):
            out.append((0, 2))
        else:
            out.append((2, 2))
    return tuple(out)


def build_driver(spec: Mapping[str, Any], seed: int,
                 sources: Sequence[Mapping[str, Any]] = (),
                 scale_bits: int = SEED_SCALE_BITS) -> Any:
    """One seeded ``FdtdDriver`` from a fixture spec.

    ``spec`` keys: ``cell`` (three extents), ``folds`` (``(axis, phase)`` pairs),
    ``beta``, ``complex_storage``, ``boundaries``, ``dimensions``, ``k_point``.

    THE SEED IS SCALED BY ``2 ** scale_bits`` AND THE SCALE IS A PARAMETER rather than
    a constant, because leg ``seed_scale`` drives the same fixture at two scales and
    requires the MANTISSAS to be identical -- which is the measurement that licenses
    scaling at all.
    """
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415
    from meep_gpu.grid import Mirror  # noqa: PLC0415

    folds = tuple(spec.get("folds", ()))
    driver = FdtdDriver(
        cell_size=tuple(spec["cell"]), resolution=RESOLUTION, courant=COURANT,
        force_complex_fields=bool(spec.get("complex_storage", False)),
        symmetry=tuple(Mirror(axis, int(phase)) for axis, phase in folds),
        boundaries=spec.get("boundaries"),
        beta=float(spec.get("beta", 0.0)),
        k_point=tuple(spec.get("k_point", (0.0, 0.0, 0.0))),
        dimensions=int(spec.get("dimensions", 2)))
    folded_indices = {"XYZ".index(axis) for axis, _phase in folds}
    # ``pml`` OVERRIDES THE THICKNESS RULE, and it exists for exactly one leg: the
    # inactive-absorber refusal. Every curl and constitutive predicate in the folded
    # and beta families requires an ACTIVE absorber, and a fixture that cannot build
    # one would record that refusal as absence rather than as a measurement.
    override = spec.get("pml")
    driver.setup_pml(tuple(tuple(pair) for pair in override) if override is not None
                     else pml_thickness(driver.grid.shape, sorted(folded_indices)))
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


def resolved_codes(product: Product, driver: Any) -> Tuple[int, ...]:
    """The codes THIS product's own resolver produces for this driver."""
    variant = variant_of(product, driver)
    codes = _resolve_codes(product, variant, driver)
    if codes is None:
        raise RuntimeError(f"{product.family} could not resolve codes on this fixture")
    return codes


def variant_of(product: Product, driver: Any) -> str:
    resolver = getattr(product.module, "resolve_variant", None)
    if resolver is None:
        return "folded"  # the single-variant product: its predicate requires a fold
    variant = resolver(driver.grid)
    if variant is None:
        raise RuntimeError(f"{product.family} could not resolve its variant")
    return str(variant)


def _resolve_codes(product: Product, variant: str,
                   driver: Any) -> Optional[Tuple[int, ...]]:
    resolver = getattr(product.module, "resolve_codes", None)
    if resolver is not None:
        return resolver(variant, driver.grid, driver.pml)
    codes, _reasons = symmetry.folded_axis_kinds(driver.grid, driver.pml)
    return None if codes is None else tuple(int(code) for code in codes)


def boundary_kinds_codes(driver: Any) -> Tuple[int, ...]:
    """THE ARMED STRUCTURAL DEFECT: the unfolded products' own codes expression.

    ``1 if kind == "metallic" else 0`` over ``stepping._boundary_kinds``. On a folded
    grid that resolver reports ``"mirror"``, which this expression maps to 0 = PERIODIC:
    the backward ghost becomes a WRAP to the far plane AND the cell-0 mask is not
    widened. A smooth wrong answer on every folded axis rather than a crash, and the
    emitter cannot catch it either since both codes are valid there.
    """
    from meep_gpu.stepping import _boundary_kinds  # noqa: PLC0415

    return tuple(1 if kind == "metallic" else 0
                 for kind in _boundary_kinds(driver.grid, driver.pml))


# ---------------------------------------------------------------------------
# The arrangements
# ---------------------------------------------------------------------------

def count_functions(plan: Any) -> List[CountingFunction]:
    """Wrap every compiled function a plan holds, whatever shape its table is.

    THE SIBLING'S HELPER IS NOT ENOUGH ON A FOLDED FIXTURE. It assumes
    ``plan._functions`` is ``{mode: function}``, which is true of every plan an
    UNFOLDED step carries; a folded step also carries a mirror-fill plan whose table
    is ``{mode: {axis entry: function}}``, so wrapping the outer values replaces a
    DICT with a counter and the next launch dies on ``'CountingFunction' object is not
    subscriptable``. Both shapes are handled, and an unrecognised one is left ALONE
    rather than wrapped -- a plan this cannot count contributes zero to the independent
    counter, which the product leg scores as a counter disagreement, a visible failure
    rather than a silent one.
    """
    owner = declaring(plan)
    table = getattr(owner, "_functions", None)
    if not isinstance(table, dict):
        return []
    counters: List[CountingFunction] = []
    wrapped: Dict[str, Any] = {}
    for mode, entry in table.items():
        if isinstance(entry, dict):
            inner: Dict[str, Any] = {}
            for key, function in entry.items():
                counter = CountingFunction(function)
                counters.append(counter)
                inner[key] = counter
            wrapped[mode] = inner
        elif callable(entry):
            counter = CountingFunction(entry)
            counters.append(counter)
            wrapped[mode] = counter
        else:
            wrapped[mode] = entry
    owner._functions = wrapped  # noqa: SLF001 - the sibling gates' mutation seam
    return counters


class _Arrangement(Arrangement):
    """The sibling's arrangement with THIS module's function counter.

    Subclassed rather than copied so the rotation settling, the orphan ledger and the
    launch bookkeeping stay one implementation.
    """

    __slots__ = ()

    def __init__(self, name: str, shim: Optional[Shim],
                 residency: Optional[Residency],
                 selected: Optional[Mapping[str, str]] = None,
                 reasons: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__(name, None, residency, selected, reasons)
        self.shim = shim
        self.counters = []
        if shim is not None:
            wrapped: List[int] = []
            for plan in shim.plans.values():
                if plan is ABSORBED:
                    continue
                owner = declaring(plan)
                if id(owner) in wrapped:
                    continue
                wrapped.append(id(owner))
                self.counters.extend(count_functions(plan))
        self.rotating = _shared.rotating_owners(shim)
        self.originals = ({} if shim is None
                          else _shared.rotating_originals(shim.fields, self.rotating))


def arrangement_array() -> Arrangement:
    return _Arrangement("array", None, None)


def arrangement_weld(product: Product, driver: Any, *,
                     functions: Optional[Mapping[str, Any]] = None,
                     codes: Optional[Sequence[int]] = None,
                     beta_words: Optional[Sequence[Any]] = None,
                     name: str = "weld",
                     sync_hazard: bool = False,
                     rotation: bool = True,
                     in_place: bool = False) -> Tuple[Arrangement, Any]:
    """THE SUBJECT: this product's plan at ``update_H``, ``step_D`` ABSORBED.

    ``functions``/``codes``/``beta_words`` are the three mutation seams, passed
    straight through to the family's own planner: a leg that dropped one would launch
    the shipped kernel and report its defect as uncaught.

    ``rotation=False`` and ``in_place=True`` are the two HOST defects -- the rotation
    skipped, and the scratch bound to the pre-launch buffers so the weld writes ``H``
    and ``f_w_H`` in place. Both must diverge; they are what makes the scratch-output
    shape a measurement rather than a design note.
    """
    residency = Residency()
    keywords: Dict[str, Any] = {}
    if functions is not None:
        keywords["functions"] = dict(functions)
    if codes is not None:
        keywords["codes"] = tuple(codes)
    if beta_words is not None:
        keywords["beta_words"] = tuple(beta_words)
    if product.expansion is not None and "probe" in _planner_parameters(product):
        pass  # the planner loads its own probe; the gate never supplies a default
    plan = product.planner(driver.fields, driver.pml,
                           tuple(getattr(driver, "_sources", ())), residency,
                           **keywords)
    if plan is None:
        raise RuntimeError(f"{product.family} refused a fixture this gate expects it "
                           f"to admit")
    if in_place:
        # THE ALIAS IS BOTH ROTATING GROUPS. With H and f_w_H bound in place the
        # weld writes exactly the volumes it also reads, which is the premise the
        # scratch-output shape removes; a foreign recompute at a neighbour whose
        # thread already stored then reads B where it needs B_prev.
        plan = _shared.AliasedLaunch(plan, product.module.ROTATED_NAMES)
    elif not rotation:
        plan = _shared.RotationSkipped(plan)
    shim = Shim(driver.fields,
                {"update_H": synced(plan, residency), "step_D": ABSORBED},
                sync_hazard=sync_hazard)
    return _Arrangement(name, shim, residency), plan


def _planner_parameters(product: Product) -> Tuple[str, ...]:
    import inspect  # noqa: PLC0415

    return tuple(inspect.signature(product.planner).parameters)


def arrangement_singles(product: Product, driver: Any) -> Arrangement:
    """Reference 2: the two CERTIFIED parent plans dispatched at the seam's two slots."""
    residency = Residency()
    constitutive, curl = product.singles(driver, residency)
    if constitutive is None or curl is None:
        raise RuntimeError(
            f"a certified parent plan was refused on a fixture this gate expects it "
            f"to admit (constitutive={constitutive is None}, curl={curl is None})")
    shim = Shim(driver.fields, {"update_H": synced(constitutive, residency),
                                "step_D": synced(curl, residency)})
    variant = variant_of(product, driver)
    arms = product.cell_arms[variant]
    return _Arrangement("singles", shim, residency,
                        selected={"update_H": arms[0], "step_D": arms[1]})


def arrangement_composed(driver: Any, fuse: bool, name: str) -> Arrangement:
    """References 3 and 4: the shipped composer, fused and unfused.

    ONE ARRANGEMENT IS ONE RESIDENCY. The registry's whole purpose is that every plan
    touching a volume binds the SAME device tensor, so an arrangement whose slots were
    built on two registries would launch against two different device buffers for one
    host array and the copies back would overwrite each other -- a silent wrong answer
    the sibling gate measured on 2026-09-05.
    """
    residency = Residency()
    plan = metal_launch.plan_step(driver.fields, driver.pml, residency=residency,
                                  sources=tuple(getattr(driver, "_sources", ())),
                                  fuse=fuse)
    plans = {slot: synced(entry, residency)
             for slot, entry in dict(plan.plans).items() if entry is not None}
    return _Arrangement(name, Shim(driver.fields, plans), residency,
                        selected=dict(plan.selected), reasons=dict(plan.reasons))


# ---------------------------------------------------------------------------
# The host legs
# ---------------------------------------------------------------------------

def leg_driver_order(product: Product) -> Dict[str, Any]:
    """``REPLACES`` is the driver's two ADJACENT consults, read off the tree.

    NEVER SPELLED. The span, the statement between the two consults and the position
    of every fill relative to them are parsed out of ``driver.py``'s own text, so a
    driver edit that moved a fill inside the span fails here rather than leaving a
    weld that silently skips a pass.
    """
    text = (API_ROOT / "meep_gpu" / "driver.py").read_text(encoding="utf-8")
    lines = text.splitlines()
    consults: Dict[str, int] = {}
    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        for slot in ("update_H", "step_D", "fill_B", "fill_D",
                     "fill_folded_far_ghosts_B", "fill_folded_far_ghosts_D"):
            if f'dispatch("{slot}"' in stripped and slot not in consults:
                consults[slot] = number
    missing = [slot for slot in ("update_H", "step_D") if slot not in consults]
    if missing:
        return {"passed": False, "reason": f"no consult found for {missing}"}
    opens, closes = consults["update_H"], consults["step_D"]
    between = [{"line": number, "statement": lines[number - 1].strip()}
               for number in range(opens + 1, closes)
               if lines[number - 1].strip()]
    # THE ARRAY CALL UNDER THE FIRST CONSULT IS PART OF THE CONSULT, NOT OF THE SPAN.
    # `driver.step` writes `if fast is None or not fast.dispatch("update_H", fields):`
    # and puts `update_H(self.fields, self.pml)` in its body, so a naive "every line
    # between the two consult lines" walk counts the fallback the consult exists to
    # skip. The seam is what runs when the consult answered TRUE, so the leading
    # slot's own array call is classified with it and everything else must be the
    # withdraw.
    consulted_fallback = "update_H(self.fields"
    in_the_span = [row for row in between
                   if not row["statement"].startswith(consulted_fallback)]
    only_the_withdraw = all("withdraw" in row["statement"] or
                            row["statement"].startswith("for source in electric")
                            for row in in_the_span)
    fills_outside = {slot: number for slot, number in consults.items()
                     if slot not in ("update_H", "step_D")}
    fills_inside = {slot: number for slot, number in fills_outside.items()
                    if opens < number < closes}
    passed = (tuple(product.module.REPLACES) == ("update_H", "step_D")
              and closes > opens and only_the_withdraw and not fills_inside
              and product.module.CARRIES_DEPOSIT_REPAIR is False)
    return {
        "passed": passed,
        "replaces": list(product.module.REPLACES),
        "update_H_consult_line": opens, "step_D_consult_line": closes,
        "statements_between_the_consults": between,
        "statements_in_the_span": in_the_span,
        "the_leading_slots_own_array_call": [row for row in between
                                             if row not in in_the_span],
        "only_the_withdraw_sits_between_them": only_the_withdraw,
        "fill_consult_lines": fills_outside,
        "fill_consults_inside_the_span": fills_inside,
        "carries_deposit_repair": product.module.CARRIES_DEPOSIT_REPAIR,
        "why_no_deposit_repair": (
            "nothing is INJECTED between the two consults: the magnetic injection is "
            "one seam earlier and the electric one is one seam later, so "
            "deposit_repair has nothing to say about this seam"),
    }


def leg_policy(product: Product, admitting_case: Mapping[str, Any]) -> Dict[str, Any]:
    """THE SUBNORMAL POLICY, MEASURED IN BOTH DIRECTIONS ON THIS PRODUCT.

    A two-policy campaign on this backend is not two attempts at the same claim. On
    MPS the float32 subnormal flush is NATIVE and exposes no lever, so
    ``subnormal.mps_policy_report()`` admits ``flush`` and refuses ``keep`` BY NAME --
    "Metal flushes denormals natively and exposes no lever; both denormal pragma
    spellings are compile errors". The two legs therefore measure two different things
    and this leg says which one it is running:

    * under ``flush`` -- the policy is attainable and admitted, and the product's
      predicate ADMITS the fixture it was built for;
    * under ``keep`` -- the policy is refused by name, and the product's predicate must
      refuse the SAME fixture, carrying that reason. A product that admitted under a
      policy the executor cannot honour would be certifying arithmetic nobody can
      reproduce.

    This is the only leg whose PASS condition depends on which policy the campaign is
    running, and it is written so that BOTH answers are a pass -- the refusal is a
    result, not a failure.
    """
    report = subnormal.mps_policy_report()
    resolved = str(report.get("resolved"))
    admitted = bool(report.get("admitted"))
    driver = build_driver(admitting_case, 3, scale_bits=0)
    verdict = product.coverage(driver.fields, driver.pml, (), Residency())
    joined = " | ".join(verdict.reasons)
    if resolved == "flush":
        passed = admitted and verdict.covered
        expectation = "flush is attainable on MPS and the predicate must ADMIT"
    else:
        passed = (not admitted and not verdict.covered
                  and ("subnormal" in joined or "denormal" in joined
                       or "policy" in joined))
        expectation = ("keep is unattainable on MPS and the predicate must REFUSE, "
                       "carrying the policy's own reason")
    return {
        "passed": passed,
        "resolved_policy": resolved,
        "policy_admitted": admitted,
        "policy_report": report,
        "expectation": expectation,
        "predicate_admits_the_admitting_fixture": bool(verdict.covered),
        "predicate_reasons": list(verdict.reasons)[:6],
        "what_a_keep_run_can_and_cannot_be": (
            "a keep campaign on this backend CANNOT release, and that is structural "
            "rather than a defect: every predicate refuses by name, so the device legs "
            "have no admitted fixture to drive and the run is INCOMPLETE, which makes "
            "release.released False. The keep artifact is the RECORD of that refusal; "
            "the release rests on the flush artifact"),
    }


def leg_transcription(product: Product, driver: Any) -> Dict[str, Any]:
    """The emitted source is the two certified bodies plus the declared edits.

    THREE THINGS ARE MEASURED, none of them asserted from a table alone:

    * the welded curl's tail differs from the PARENT EMITTER's own body in exactly the
      redirected magnetic loads -- every other line is character-identical, counted;
    * no ``gN[`` survives in the curl half, so every magnetic read is either the
      register or a recompute;
    * the ``h_cell`` body is the imported certified constitutive, and on a beta
      product it mentions none of the beta identifiers -- which is what makes "beta is
      a curl-only term" a property of the emission rather than of a docstring.
    """
    variant = variant_of(product, driver)
    codes = resolved_codes(product, driver)
    phased = _phased_of(product, driver)
    expansion = product.expansion() if product.expansion is not None else None
    source = _emit(product, variant, codes, phased, expansion)
    parent = product.parent_curl_source(variant, codes, phased, expansion)

    parent_body = parent.split(BODY_ANCHOR, 1)[1][: -len("}\n")]
    fused_curl = source.split("__CURL_MARK__", 1)[-1]
    del fused_curl  # the marker is not emitted; the comparison below is line-wise

    parent_lines = [line for line in parent_body.splitlines()]
    fused_lines = source.splitlines()
    redirected = [line for line in parent_lines
                  if " ? g0[" in line or " ? g1[" in line or " ? g2[" in line]
    own_loads = [line for line in parent_lines
                 if line.strip().startswith(("float a   = g0[ii]",
                                             "float b   = g1[ii]",
                                             "float c   = g2[ii]",
                                             "float2 a   = g0[ii]",
                                             "float2 b   = g1[ii]",
                                             "float2 c   = g2[ii]"))]
    survived = [line for line in parent_lines
                if line not in redirected and line not in own_loads
                and line.strip() and line in fused_lines]
    curl_half = source.split("h_cell_result own = h_cell(", 1)[-1]
    reads_gn = sorted({stem for stem in ("g0[", "g1[", "g2[") if stem in curl_half})

    # THE CONSTITUTIVE LIFT IS THE `h_cell` FUNCTION, NOT EVERYTHING ABOVE THE KERNEL.
    # Slicing at `kernel void` sweeps in the `struct Params` declaration, which on a
    # beta product NAMES the beta words -- so the beta-free check reported the struct
    # and failed on a kernel whose h_cell is beta-free. Measured 2026-09-07 on the real
    # beta gate. The function's own text is what the claim is about.
    marker = "h_cell_result h_cell("
    if marker not in source or "kernel void" not in source:
        raise AssertionError(
            "the emitted source carries no `h_cell_result h_cell(` definition above "
            "its kernel; the constitutive lift this leg reads is not where it was")
    h_cell = source.split(marker, 1)[1].split("kernel void", 1)[0]
    beta_identifiers = tuple(getattr(product.module, "BETA_IDENTIFIERS", ()))
    beta_in_h_cell = sorted(name for name in beta_identifiers if name in h_cell)
    recomputes = curl_half.count("h_cell(")

    passed = (len(redirected) == 6 and len(own_loads) == 3 and not reads_gn
              and not beta_in_h_cell and recomputes == 6
              and len(survived) >= 40)
    return {
        "passed": passed,
        "variant": variant, "codes": list(codes), "codes_label": codes_label(codes),
        "phased": list(phased), "expansion": expansion,
        "parent_lines": len(parent_lines),
        "shifted_loads_redirected": len(redirected),
        "own_cell_loads_taken_from_the_register": len(own_loads),
        "parent_lines_surviving_character_identical": len(survived),
        "recompute_call_sites_in_the_curl_half": recomputes,
        "magnetic_pointers_still_read_in_the_curl_half": reads_gn,
        "beta_identifiers_in_the_constitutive_lift": beta_in_h_cell,
        "constitutive_lift_characters": len(h_cell),
        "beta_identifiers_declared": list(beta_identifiers),
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
    }


def _phased_of(product: Product, driver: Any) -> Tuple[int, int, int]:
    """The per-axis Bloch flags this product's own planner would bake.

    Read through the parent's own resolver, never retyped: the conjugation rule and
    the "a mirror axis cannot carry a phase" rule each have exactly one home.
    """
    if not product.complex_storage:
        return (0, 0, 0)
    from meep_gpu.stepping import _boundary_kinds  # noqa: PLC0415

    kinds = _boundary_kinds(driver.grid, driver.pml)
    resolver = getattr(product.module, "_PHASE_RESOLVER", None)
    if resolver is not None:
        return tuple(int(flag) for flag in resolver(driver.grid, kinds))
    from meep_gpu.metal_kernels import complex_fields  # noqa: PLC0415

    flags, _values = complex_fields.phase_arguments(
        complex_fields.bloch_phase_table(driver.grid, kinds), backward=True)
    return tuple(int(flag) for flag in flags)


def _emit(product: Product, variant: str, codes: Sequence[int],
          phased: Sequence[int], expansion: Optional[str],
          contract: Optional[str] = None) -> str:
    """This product's shipped source, with the arguments it actually takes."""
    from meep_gpu.metal_kernels import shaders  # noqa: PLC0415

    mode = shaders.CONTRACT_OFF if contract is None else contract
    if expansion is None:
        return product.source(variant, codes, mode)
    return product.source(variant, codes, phased, expansion, mode)


def leg_refusal(product: Product, refusals: Sequence[Dict[str, Any]]
                ) -> Dict[str, Any]:
    """Each refusal BY NAME, and the INVERTED clause driven in both directions.

    ``refusals`` is the gate's own table: each entry names a fixture, the substring
    the refusal must contain, and -- where the clause inverts -- the sibling predicate
    that must ADMIT what this one refuses. RECORDED rather than asserted: whether the
    product appears in a wiring table. ASSERTED: that it is refused by name.
    """
    rows: List[Dict[str, Any]] = []
    for entry in refusals:
        # TWO WAYS TO BUILD A REFUSAL FIXTURE, and the second is not a convenience.
        # ``spec`` goes through this module's driver builder; ``pair`` is a callable
        # returning ``(fields, pml)`` and exists for the configurations a DRIVER
        # cannot be built in -- ``setup_pml`` refuses a layer that absorbs nowhere by
        # name, so the inactive-absorber refusal has to come from
        # ``metal_composition_matrix``'s own no-PML builder or it would be recorded as
        # absence rather than as a measurement.
        if entry.get("pair") is not None:
            fields, pml = entry["pair"]()
            driver = None
        else:
            driver = build_driver(entry["spec"], entry.get("seed", 3), scale_bits=0)
            fields, pml = driver.fields, driver.pml
        verdict = product.coverage(fields, pml, (), Residency())
        joined = " | ".join(verdict.reasons)
        named = entry["must_name"] in joined
        sibling_admits: Optional[bool] = None
        if entry.get("sibling") is not None:
            sibling = entry["sibling"](fields, pml)
            sibling_admits = bool(sibling.covered)
        rows.append({
            "case": entry["name"], "admits": bool(verdict.covered),
            "refused_by_name": named, "must_name": entry["must_name"],
            "sibling_admits_what_this_refuses": sibling_admits,
            "sibling_expected": entry.get("sibling_expected"),
            "reasons": list(verdict.reasons)[:8],
            "passed": (not verdict.covered and named
                       and (entry.get("sibling_expected") is None
                            or sibling_admits == entry["sibling_expected"])),
        })
    return {"passed": bool(rows) and all(row["passed"] for row in rows),
            "refusals": rows, "count": len(rows)}


def leg_withdraw(product: Product, spec: Mapping[str, Any]) -> Dict[str, Any]:
    """The seam's ONE pass: a standing integrated electric withdraw is refused by name.

    THREE STATES, and all three are driven rather than argued: sources UNDECLARED
    (``None``) must refuse, because ``Fields`` does not hold the source list and a
    predicate that inferred "no withdraw stands" from not being told would be the
    over-covering refusal the clause exists to prevent; an EMPTY tuple must admit; and
    a live integrated electric source must refuse, naming ``HOISTS_THE_WITHDRAW`` and
    ``INSTALLABLE``.
    """
    driver = build_driver(spec, 5, scale_bits=0)
    residency = Residency()
    undeclared = product.coverage(driver.fields, driver.pml, None, residency)
    empty = product.coverage(driver.fields, driver.pml, (), residency)

    class _IntegratedElectric:
        """A source that satisfies ``withdraw_hoist._withdraw_does_work``' OWN three
        conditions -- a callable ``withdraw``, ``is_integrated``, and a positive
        ``_n_source_points`` -- and is electric by the driver's own split (any
        ``field_type`` that is not the magnetic one). Built to the module's contract
        rather than to a guess, because a stub the predicate does not recognise would
        make this leg pass vacuously.
        """

        field_type = "electric"
        is_integrated = True
        _n_source_points = 3

        def withdraw(self, fields: Any) -> None:  # noqa: ARG002
            return None

    standing = product.coverage(driver.fields, driver.pml,
                                (_IntegratedElectric(),), residency)
    joined = " | ".join(standing.reasons)
    return {
        "passed": (not undeclared.covered and empty.covered
                   and not standing.covered
                   and "HOISTS_THE_WITHDRAW = False" in joined
                   and "INSTALLABLE = False" in joined
                   and "3313-3314" in joined),
        "undeclared_sources_refused": not undeclared.covered,
        "undeclared_reason": next((r for r in undeclared.reasons
                                   if "not declared" in r), None),
        "empty_source_set_admits": bool(empty.covered),
        "standing_withdraw_refused": not standing.covered,
        "standing_withdraw_reason": next((r for r in standing.reasons
                                          if "withdraw" in r), None),
        "hoists_the_withdraw": product.module.HOISTS_THE_WITHDRAW,
        "rows_in_this_cell_with_a_standing_withdraw": 0,
        "what_this_does_not_license": (
            "flipping HOISTS_THE_WITHDRAW: no leg here drives a hoisted launch, so "
            "every row with a standing in-seam withdraw stays refused BY NAME"),
    }


def leg_binding_ceiling(product: Product, drivers: Sequence[Any]) -> Dict[str, Any]:
    """The ceiling as an EQUALITY, bisected on this host, and the packed probe LAUNCHED.

    Nothing here is counted from a signature: each refuted shape is COMPILED and the
    failure is required; the shipped shape is compiled on EVERY fixture's own resolved
    codes; and the ``Params`` record is bound to a probe kernel that reads every field
    back, because a struct whose host record is a byte short is a plausible number
    rather than a crash.
    """
    refuted: Dict[str, Any] = {}
    for name, (builder, expected) in product.refuted_signatures.items():
        source = builder()
        declared = source.count("[[buffer(")
        try:
            compile_source(source)
            failed = False
            error = None
        except Exception as exc:  # noqa: BLE001 - the failure IS the measurement
            failed = True
            error = f"{type(exc).__name__}: {str(exc)[:160]}"
        refuted[name] = {"declared_bindings": declared, "expected": expected,
                         "compile_failed_as_required": failed, "error": error,
                         "passed": failed and declared == expected}

    bisect: Dict[str, Any] = {}
    for pointers in (MAX_BUFFER_BINDINGS - 1, MAX_BUFFER_BINDINGS):
        source = _bisect_source(pointers)
        try:
            compile_source(source)
            bisect[f"{pointers}_pointers_plus_struct"] = "compiled"
        except Exception as exc:  # noqa: BLE001
            bisect[f"{pointers}_pointers_plus_struct"] = f"failed: {type(exc).__name__}"
    ceiling_is_an_equality = (
        bisect[f"{MAX_BUFFER_BINDINGS - 1}_pointers_plus_struct"] == "compiled"
        and bisect[f"{MAX_BUFFER_BINDINGS}_pointers_plus_struct"].startswith("failed"))

    shipped: List[Dict[str, Any]] = []
    for driver in drivers:
        variant = variant_of(product, driver)
        codes = resolved_codes(product, driver)
        phased = _phased_of(product, driver)
        expansion = product.expansion() if product.expansion is not None else None
        source = _emit(product, variant, codes, phased, expansion)
        declared = (source.split(f"kernel void {product.module.KERNEL}(", 1)[1]
                    .split("uint idx [[thread_position_in_grid]])", 1)[0]
                    .count("[[buffer("))
        try:
            compile_source(source)
            compiled = True
            error = None
        except Exception as exc:  # noqa: BLE001
            compiled = False
            error = f"{type(exc).__name__}: {str(exc)[:160]}"
        shipped.append({"variant": variant, "codes_label": codes_label(codes),
                        "declared_bindings": declared, "compiled": compiled,
                        "error": error,
                        "passed": compiled
                        and declared == product.module.PACKED_BINDINGS})

    params = product.params_probe()
    return {
        "passed": (all(row["passed"] for row in refuted.values())
                   and ceiling_is_an_equality
                   and bool(shipped) and all(row["passed"] for row in shipped)
                   and params.get("passed", False)
                   and product.module.PACKED_BINDINGS == MAX_BUFFER_BINDINGS),
        "packed_bindings": product.module.PACKED_BINDINGS,
        "platform_ceiling": MAX_BUFFER_BINDINGS,
        "headroom": MAX_BUFFER_BINDINGS - product.module.PACKED_BINDINGS,
        "refuted_signatures": refuted,
        "ceiling_bisection": bisect,
        "the_ceiling_is_an_equality_on_this_host": ceiling_is_an_equality,
        "shipped_signature_per_fixture": shipped,
        "params_record": params,
    }


def _bisect_source(pointers: int) -> str:
    """``pointers`` device pointers plus one packed struct, with a body that touches all."""
    lines = [f"    device float* v{index} [[buffer({index})]],"
             for index in range(pointers)]
    lines.append(f"    constant Params& prm [[buffer({pointers})]],")
    body = " + ".join(f"v{index}[0]" for index in range(pointers))
    return "\n".join((
        "#include <metal_stdlib>", "using namespace metal;", "",
        "struct Params { uint n_elem; float dtdx; };", "",
        "kernel void bisect(", *lines,
        "    uint idx [[thread_position_in_grid]])", "{",
        "    if (idx >= prm.n_elem) { return; }",
        f"    v0[idx] = ({body}) + prm.dtdx;", "}", ""))


# ---------------------------------------------------------------------------
# The device legs
# ---------------------------------------------------------------------------

def run_case(product: Product, name: str, spec: Mapping[str, Any], steps: int, *,
             with_composer: bool = True,
             functions: Optional[Mapping[str, Any]] = None,
             codes: Optional[Sequence[int]] = None,
             beta_words: Optional[Sequence[Any]] = None,
             rotation: bool = True, in_place: bool = False,
             sync_hazard: bool = False,
             per_step_hook: Optional[Callable[..., Any]] = None,
             seed: int = 7, scale_bits: int = SEED_SCALE_BITS) -> Dict[str, Any]:
    """One fixture, every arrangement, in lockstep from one seed."""
    driver = build_driver(spec, seed, scale_bits=scale_bits)
    variant = variant_of(product, driver)
    weld, plan = arrangement_weld(product, driver, functions=functions, codes=codes,
                                  beta_words=beta_words, rotation=rotation,
                                  in_place=in_place, sync_hazard=sync_hazard)
    arrangements: Dict[str, Arrangement] = {"array": arrangement_array()}
    arrangements["singles"] = arrangement_singles(product, driver)
    if with_composer:
        arrangements["composition_today"] = arrangement_composed(driver, True,
                                                                 "composition_today")
        arrangements["unfused"] = arrangement_composed(driver, False, "unfused")
    arrangements["weld"] = weld
    result = drive(driver, arrangements, steps, reference="array",
                   per_step_hook=per_step_hook, progress=log)
    legs = result["arrangements"]
    moved = result["moved_from_seed"]
    weld_leg = legs["weld"]
    launches = weld_leg["launches"]
    counters_agree = (launches["plans"] > 0
                      and launches["plans"] == launches["functions"])
    dispatch_shape = (weld_leg["dispatched"].get("update_H", 0)
                      == result["steps_stepped"]
                      and weld_leg["absorbed"].get("step_D", 0)
                      == result["steps_stepped"])
    return {
        "case": name, "variant": variant,
        "codes": list(getattr(plan, "codes", ())),
        "codes_label": codes_label(getattr(plan, "codes", ())),
        "grid_shape": list(driver.grid.shape),
        "steps_compared": result["steps_compared"],
        "steps_stepped": result["steps_stepped"],
        "first_banded_step": result["first_banded_step"],
        "precondition_clean": result["precondition_clean"],
        "words_compared": result["words_compared"],
        "identical_vs_array": weld_leg["identical"],
        "singles_identical_vs_array": legs["singles"]["identical"],
        "first_divergence": weld_leg["first_divergence"],
        "differing_words_final": weld_leg["differing_words_final"],
        "differing_volumes_final": weld_leg["differing_volumes_final"],
        "autopsy": weld_leg["first_divergence_autopsy"],
        "launches": launches,
        "launch_counters_agree": counters_agree,
        "dispatched": weld_leg["dispatched"], "absorbed": weld_leg["absorbed"],
        "dispatch_shape_is_one_per_step": dispatch_shape,
        "composer_selected": (legs["composition_today"]["selected"]
                              if with_composer else None),
        "composition_identical": (legs["composition_today"]["identical"]
                                  if with_composer else None),
        "unfused_identical": (legs["unfused"]["identical"] if with_composer else None),
        "arrays_that_never_moved": result["arrays_that_never_moved"],
        "moved_from_seed": len(moved),
        "every_arrangement_gave_the_engine_its_volumes_back":
            result["every_arrangement_gave_the_engine_its_volumes_back"],
        "step_error": result["step_error"],
        "seconds": result["seconds"],
    }


def leg_product(product: Product, steps: int) -> Dict[str, Any]:
    """The identity claim, on every fixture, against every available reference."""
    rows: List[Dict[str, Any]] = []
    for name, spec in product.cases:
        row = run_case(product, name, spec, steps)
        row["passed"] = bool(
            row["identical_vs_array"] and row["singles_identical_vs_array"]
            and row["launch_counters_agree"] and row["dispatch_shape_is_one_per_step"]
            and row["steps_compared"] == steps and not row["arrays_that_never_moved"]
            and row["every_arrangement_gave_the_engine_its_volumes_back"]
            and row["step_error"] is None)
        rows.append(row)
        log(f"  product {name}: identical={row['identical_vs_array']} "
            f"steps={row['steps_compared']} words={row['words_compared']} "
            f"launches={row['launches']}")
    return {
        "passed": bool(rows) and all(row["passed"] for row in rows),
        "cases": rows, "steps_requested": steps,
        "words_compared_total": int(sum(row["words_compared"] for row in rows)),
        "references": {
            "1": "the array path under the driver's own loop",
            "2": "the two CERTIFIED parent plans, dispatched at the seam's two slots",
            "3": "plan_step(fuse=True): the composition the composer installs today",
            "4": "plan_step(fuse=False): the same slots dispatched unfused",
        },
    }


def leg_seed_scale(product: Product, steps: int) -> Dict[str, Any]:
    """THE SCALING IS EXPONENT-ONLY, MEASURED ON THIS PRODUCT'S OWN FIXTURES.

    The whole synthetic claim rests on a seed scaled by ``2**80`` to clear the denormal
    band, and the licence for that is that the solver is linear in the field state with
    field-independent coefficients -- so a scale shifts every float32 EXPONENT and
    leaves every MANTISSA and every rounding decision untouched. INHERITING that from a
    sibling gate would be inheriting a claim about a different kernel: this product's
    curl carries a beta term or a fold block the sibling's does not.

    Driven: the same fixture at ``2**80`` and at ``2**0``, both fused, with the
    MANTISSA and SIGN words of every stored volume required to be identical and the
    EXPONENT words required to differ by exactly 80 wherever the word is a normal.
    """
    rows: List[Dict[str, Any]] = []
    for name, spec in product.cases[:2]:
        high = _final_state(product, spec, steps)
        low = _final_state(product, spec, steps, scale_bits=0)
        mantissa_diffs = 0
        exponent_shifts: Dict[int, int] = {}
        compared = 0
        for volume in sorted(set(high) & set(low)):
            a, b = words(high[volume]), words(low[volume])
            if a.shape != b.shape:
                mantissa_diffs += max(a.size, b.size)
                continue
            normal = ((a & 0x7F800000) != 0) & ((a & 0x7F800000) != 0x7F800000) \
                & ((b & 0x7F800000) != 0) & ((b & 0x7F800000) != 0x7F800000)
            compared += int(np.count_nonzero(normal))
            mantissa_diffs += int(np.count_nonzero(
                (a[normal] & 0x807FFFFF) != (b[normal] & 0x807FFFFF)))
            shifts = (((a[normal] >> 23) & 0xFF).astype(np.int64)
                      - ((b[normal] >> 23) & 0xFF).astype(np.int64))
            for shift, count in zip(*np.unique(shifts, return_counts=True)):
                exponent_shifts[int(shift)] = (exponent_shifts.get(int(shift), 0)
                                               + int(count))
        rows.append({
            "case": name, "normal_words_compared": compared,
            "mantissa_and_sign_words_that_differ": mantissa_diffs,
            "exponent_shift_histogram": {str(k): v for k, v
                                         in sorted(exponent_shifts.items())},
            "passed": (compared > 0 and mantissa_diffs == 0
                       and set(exponent_shifts) == {SEED_SCALE_BITS}),
        })
        log(f"  seed_scale {name}: normals={compared} mantissa_diffs={mantissa_diffs} "
            f"shifts={sorted(exponent_shifts)}")
    return {
        "passed": bool(rows) and all(row["passed"] for row in rows),
        "scale_bits": SEED_SCALE_BITS, "cases": rows,
        "why": ("the synthetic byte claim is taken on a seed scaled by 2**80 to clear "
                "the float32 denormal band; this leg is what licenses the scale on "
                "THIS kernel rather than on a sibling's"),
    }


def _final_state(product: Product, spec: Mapping[str, Any], steps: int,
                 scale_bits: int = SEED_SCALE_BITS) -> Dict[str, Any]:
    driver = build_driver(spec, 7, scale_bits=scale_bits)
    weld, _plan = arrangement_weld(product, driver)
    install(driver, weld.shim)
    for _ in range(steps):
        driver.step()
        weld.settle(driver.fields)
    return {name: np.array(array, copy=True)
            for name, array in stored_volumes(driver.fields).items()}


def leg_purity(product: Product, steps: int) -> Dict[str, Any]:
    """THE RACE LEDGER: what the foreign recompute reads, and that it is UNWRITTEN.

    Three questions, each measured on every fixture rather than argued once:

    * HOW MANY foreign taps a launch performs and how many of them land on a cell some
      OTHER thread also writes. Under the shipped shape that number is the whole point:
      every one of them reads ``H``/``f_w_H``/``B``, which this launch does not write;
    * whether the taps are OBSERVABLE at all -- a fixture on which every tap fell past
      a wall would license nothing, so the count of taps that land on a REAL stored
      cell is recorded and required to be positive;
    * whether writing ``H``/``f_w_H`` IN PLACE actually diverges. That is the armed
      null control for the whole scratch-output design: if the in-place launch were
      byte-identical, the scratch would be buying nothing and every "no thread observes
      another thread's store" sentence in the family would be unfalsifiable.
    """
    rows: List[Dict[str, Any]] = []
    for name, spec in product.cases:
        driver = build_driver(spec, 7, scale_bits=SEED_SCALE_BITS)
        shape = tuple(int(n) for n in driver.grid.shape)
        codes = resolved_codes(product, driver)
        cells = int(np.prod(shape))
        # THE TAP LEDGER, from the emitted ghost rule rather than from a formula: a
        # thread taps three backward neighbours per component pair, and a tap is REAL
        # when the shifted index is in range (a metallic or mirror face serves the
        # literal ghost instead, and no recompute is evaluated there at all).
        real = 0
        wall = 0
        for index in range(cells):
            k = index % shape[2]
            plane = index // shape[2]
            j = plane % shape[1]
            i = plane // shape[1]
            for axis, position in enumerate((i, j, k)):
                shifted = position - 1
                periodic = int(codes[axis]) == symmetry.CODE_PERIODIC
                if shifted >= 0:
                    real += 2
                elif periodic:
                    real += 2
                else:
                    wall += 2
        row = run_case(product, name, spec, steps, with_composer=False,
                       in_place=True)
        rows.append({
            "case": name, "grid_shape": list(shape),
            "codes_label": codes_label(codes),
            "foreign_taps_per_launch_on_a_real_cell": real,
            "foreign_taps_per_launch_served_the_literal_ghost": wall,
            "in_place_H_and_f_w_H_diverged": not row["identical_vs_array"],
            "in_place_first_divergence": row["first_divergence"],
            "in_place_differing_words": row["differing_words_final"],
            "passed": real > 0 and not row["identical_vs_array"],
        })
        log(f"  purity {name}: real_taps={real} wall_taps={wall} "
            f"in_place_diverged={not row['identical_vs_array']}")
    return {
        "passed": bool(rows) and all(row["passed"] for row in rows),
        "cases": rows,
        "what_is_const_for_the_whole_dispatch": ["H", "f_w_H", "B"],
        "what_steps_in_place": ["D", "fu_D"],
        "why_in_place_D_is_safe": (
            "the curl reads and writes D and fu_D at the THREAD'S OWN CELL only "
            "(f0[ii], u0[ii]), so no thread reads a displacement another thread wrote"),
    }


def drop_ownership_masks(source: str) -> Tuple[str, int]:
    """Every cell-0 ownership-mask line removed, and how many were.

    Generic over the emitted text rather than a per-fixture needle: the emitter writes
    one ``curlN = at_a ? <zero> : curlN;`` line per masked target per non-periodic
    axis, so which lines exist depends on the fixture. A count of zero is reported to
    the caller instead of silently arming nothing.
    """
    kept: List[str] = []
    removed = 0
    for line in source.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("curl") and " = at_" in stripped:
            removed += 1
            continue
        kept.append(line)
    return "".join(kept), removed


def leg_ghost_observability(product: Product, steps: int) -> Dict[str, Any]:
    """IS THE GHOST OBSERVABLE? An EARNED NULL, not an assumed one.

    The lift redirects six magnetic loads and leaves the ternary's GUARD and the
    literal ghost untouched, character for character. Whether that preservation is
    load-bearing for the byte result is a question with an answer, and this leg
    measures it in two arms rather than assuming either:

    * ARM A -- the ghost VALUE replaced by a sentinel of the fixture's own magnitude,
      with the guard and the offset left exactly where they are so nothing is
      evaluated out of range. MEASURED NULL on every fixture: the ghost is DEAD. That
      is ``symmetry.py:30-43``'s claim reproduced on this seam and under this
      storage -- the ghost's only consumers are the curl targets whose Yee shift is 0
      on that axis, and those are exactly the targets the cell-0 ownership mask zeroes;
    * ARM B -- the same sentinel PLUS every cell-0 mask line removed. MUST DIVERGE. It
      is what earns arm A: a null whose paired control does not bite is a claim about
      a branch nothing executes, and this pair shows the branch IS executed and its
      value IS reaching the arithmetic, and that the mask is what kills it.

    So what this leg establishes is stronger than "the ghost is preserved": it names
    the MASK, not the ghost literal, as the thing that carries the result, and it
    prices what a future product that dropped the mask would lose.
    """
    from meep_gpu.metal_kernels import shaders  # noqa: PLC0415

    rows: List[Dict[str, Any]] = []
    for name, spec in product.cases:
        driver = build_driver(spec, 7, scale_bits=0)
        codes = resolved_codes(product, driver)
        variant = variant_of(product, driver)
        phased = _phased_of(product, driver)
        expansion = product.expansion() if product.expansion is not None else None
        source = _emit(product, variant, codes, phased, expansion)
        ghost = getattr(product.module, "_GHOST", "0.0f")
        sentinel = ("float2(0.5f, 0.25f)" if product.complex_storage else "0.5f")
        marker = f" : {ghost};"
        edited, armed = source, 0
        for line in source.splitlines():
            if " ? h_cell(" in line and line.endswith(marker):
                edited = needle(edited, line + "\n",
                                line[: -len(marker)] + f" : {sentinel};\n")
                armed += 1
        if armed != 6:
            # A GATE THAT CANNOT ARM ITS OWN CONTROL MUST SAY SO rather than pass.
            rows.append({"case": name, "passed": False,
                         "reason": f"armed {armed} of the 6 ghost sites"})
            continue
        unmasked, masks_removed = drop_ownership_masks(edited)
        # AN ALL-PERIODIC FIXTURE HAS NO GHOST BRANCH AND NO MASK, and that is a
        # STRUCTURAL ABSENCE rather than a failure. `templates.ghost` leaves the
        # validity flag permanently true on a periodic axis (the shift WRAPS), so no
        # thread ever takes the ternary's ghost arm and `templates.ownership_mask`
        # emits no line to remove. Arm A is still meaningful there -- the sentinel must
        # not move a single word, because nothing reads it -- and arm B does not exist.
        # Scoring the absence as a failure would report a fixture the product serves
        # correctly as a defect; scoring it as a pass without arm A would let a
        # sentinel that DID leak through go unnoticed.
        all_periodic = all(int(code) == symmetry.CODE_PERIODIC for code in codes)
        if not masks_removed and not all_periodic:
            rows.append({"case": name, "passed": False,
                         "reason": "no cell-0 mask line is emitted here and the "
                                   "fixture is not all-periodic, so the null can be "
                                   "neither earned nor explained"})
            continue
        try:
            arm_a = getattr(compile_source(edited), product.module.KERNEL)
            arm_b = (None if all_periodic
                     else getattr(compile_source(unmasked), product.module.KERNEL))
        except Exception as exc:  # noqa: BLE001
            rows.append({"case": name, "passed": False,
                         "reason": f"an armed ghost source did not compile: {exc}"})
            continue
        sentinel_row = run_case(product, name, spec, steps, with_composer=False,
                                functions={shaders.CONTRACT_OFF: arm_a}, scale_bits=0)
        sentinel_null = sentinel_row["identical_vs_array"]
        if arm_b is None:
            rows.append({
                "case": name, "codes_label": codes_label(codes),
                "ghost_sites_armed": armed, "sentinel": sentinel,
                "every_axis_is_periodic": True,
                "mask_lines_removed_for_the_control": 0,
                "sentinel_alone_is_null": sentinel_null,
                "sentinel_words": sentinel_row["differing_words_final"],
                "sentinel_plus_dropped_mask_diverged": None,
                "passed": bool(sentinel_null),
                "verdict": ("every axis is PERIODIC: the shift WRAPS, no thread takes "
                            "the ghost branch and no ownership-mask line is emitted. "
                            "Arm A is still required to be null -- a sentinel that "
                            "moved a word here would mean a thread read past a face "
                            "the emitter says it cannot -- and arm B is structurally "
                            "absent rather than skipped"),
            })
            log(f"  ghost_observability {name}: all-periodic, "
                f"sentinel_null={sentinel_null}")
            continue
        earned_row = run_case(product, name, spec, steps, with_composer=False,
                              functions={shaders.CONTRACT_OFF: arm_b}, scale_bits=0)
        control_bites = not earned_row["identical_vs_array"]
        rows.append({
            "case": name, "codes_label": codes_label(codes),
            "ghost_sites_armed": armed, "sentinel": sentinel,
            "every_axis_is_periodic": False,
            "mask_lines_removed_for_the_control": masks_removed,
            "sentinel_alone_is_null": sentinel_null,
            "sentinel_words": sentinel_row["differing_words_final"],
            "sentinel_plus_dropped_mask_diverged": control_bites,
            "control_words": earned_row["differing_words_final"],
            "control_volumes": earned_row["differing_volumes_final"],
            "passed": bool(sentinel_null and control_bites),
            "verdict": ("the ghost value is DEAD: its only consumers are the curl "
                        "targets the cell-0 ownership mask zeroes. The paired control "
                        "shows the branch IS executed and the MASK is what kills it"),
        })
        log(f"  ghost_observability {name}: sentinel_null={sentinel_null} "
            f"control_bites={control_bites}")
    return {"passed": bool(rows) and all(row["passed"] for row in rows),
            "cases": rows,
            "arm_a": "the literal ghost past a wall replaced by a sentinel -- NULL",
            "arm_b": "the same sentinel with every cell-0 mask line removed -- MUST FIRE",
            "what_this_establishes": (
                "the ghost literal the weld preserves character for character is not "
                "what carries the byte result on this seam; the cell-0 ownership mask "
                "is. The preservation is still correct and still required -- a product "
                "that changed the ghost AND the mask together would move the answer -- "
                "but the null is now measured and earned rather than assumed")}


def leg_masks(product: Product, steps: int,
              builders: Mapping[str, Tuple[Callable[..., str], str]],
              case_name: str) -> Dict[str, Any]:
    """The two ownership masks, SCORED with their attribution asserted.

    ``symmetry.py:55-60`` records the masks as INVISIBLE after a complete step,
    because the driver's fill passes overwrite exactly the planes they protect. On
    THIS seam they are visible, and both statements are right: the masks zero
    ``curlN``, which feeds the split-field auxiliary ``nN`` as well as the
    displacement ``vN``, and ``stepping._fill_symmetry_ghost_cells`` / ``._zero_metal``
    walk ``D_CURL_TERMS`` / ``D_COMPONENTS`` and never ``fu_*`` -- so the fills restore
    the displacement at those planes and nothing restores the auxiliary.

    The leg therefore asserts the ATTRIBUTION rather than the bare verdict: each mask
    defect must move ``fu_D`` and must NOT move ``D``. That is strictly stronger than
    either "it fires" or "it is null", and it is what makes the sibling H->D gate's
    same-day finding a reproduction on these cells rather than a citation.
    """
    from meep_gpu.metal_kernels import shaders  # noqa: PLC0415

    spec = product.case(case_name)
    driver = build_driver(spec, 7, scale_bits=0)
    variant = variant_of(product, driver)
    codes = resolved_codes(product, driver)
    phased = _phased_of(product, driver)
    expansion = product.expansion() if product.expansion is not None else None
    shipped = _emit(product, variant, codes, phased, expansion)
    rows: Dict[str, Any] = {}
    for label, (builder, why) in sorted(builders.items()):
        edited = builder(shipped)
        if edited == shipped:
            rows[label] = {"passed": False,
                           "reason": "the needle matched nothing; the mask this leg "
                                     "scores is not emitted on this fixture"}
            continue
        compiled = getattr(compile_source(edited), product.module.KERNEL)
        row = run_case(product, case_name, spec, steps, with_composer=False,
                       functions={shaders.CONTRACT_OFF: compiled}, scale_bits=0)
        volumes = row["differing_volumes_final"] or {}
        in_d = sum(int(n) for volume, n in volumes.items()
                   if volume in ("Dx", "Dy", "Dz"))
        in_fu = sum(int(n) for volume, n in volumes.items()
                    if volume in ("fu_Dx", "fu_Dy", "fu_Dz"))
        elsewhere = sum(int(n) for volume, n in volumes.items()
                        if volume not in ("Dx", "Dy", "Dz",
                                          "fu_Dx", "fu_Dy", "fu_Dz"))
        rows[label] = {
            "passed": bool(in_fu > 0 and in_d == 0 and elsewhere == 0),
            "why": why, "case": case_name,
            "fired": not row["identical_vs_array"],
            "first_divergence": row["first_divergence"],
            "words_in_fu_D": in_fu, "words_in_D": in_d,
            "words_elsewhere": elsewhere,
            "differing_volumes": volumes,
        }
        log(f"  mask {label}: fu_D={in_fu} D={in_d} elsewhere={elsewhere}")
    return {
        "passed": bool(rows) and all(row["passed"] for row in rows.values()),
        "masks": rows, "case": case_name, "codes_label": codes_label(codes),
        "attribution": (
            "the fills restore D at the masked planes and never touch fu_*, so a mask "
            "defect is invisible in D after a complete step and visible in fu_D"),
        "citation_this_reproduces": (
            "symmetry.py:55-60 and gate_metal_symmetry -- the driver's fill passes "
            "overwrite exactly the planes the two masks protect. Both statements are "
            "right; the earlier comparison did not carry fu_*"),
    }


def leg_launch_structure(product: Product, steps: int) -> Dict[str, Any]:
    """Launches per step at the seam and over the whole step, from two witnesses.

    THE SAVING IS COUNTED, NOT TIMED. One dispatch replaces two, so the seam's launch
    count per step drops from 2 to 1 -- measured against the certified singles on the
    same fixture, twice over (the plans' own counters and an independent wrapper around
    every compiled function). NOTHING HERE IS A THROUGHPUT CLAIM: a launch count is not
    a time, and a second Metal lane was running on this GPU throughout.
    """
    rows: List[Dict[str, Any]] = []
    for name, spec in product.cases[:3]:
        driver = build_driver(spec, 7)
        weld, _plan = arrangement_weld(product, driver)
        singles = arrangement_singles(product, driver)
        arrangements = {"array": arrangement_array(), "singles": singles,
                        "weld": weld}
        result = drive(driver, arrangements, steps, reference="array")
        legs = result["arrangements"]
        stepped = result["steps_stepped"]
        weld_launches = legs["weld"]["launches"]
        single_launches = legs["singles"]["launches"]
        rows.append({
            "case": name, "steps_stepped": stepped,
            "weld_launches": weld_launches, "singles_launches": single_launches,
            "weld_launches_per_step": weld_launches["plans"] / max(stepped, 1),
            "singles_launches_per_step": single_launches["plans"] / max(stepped, 1),
            "seam_launches_saved_per_step": (single_launches["plans"]
                                             - weld_launches["plans"]) / max(stepped, 1),
            "passed": (weld_launches["plans"] == stepped
                       and weld_launches["functions"] == stepped
                       and single_launches["plans"] == 2 * stepped
                       and single_launches["functions"] == 2 * stepped
                       and legs["weld"]["identical"] and legs["singles"]["identical"]),
        })
        log(f"  launch_structure {name}: weld={weld_launches} singles={single_launches}")
    return {"passed": bool(rows) and all(row["passed"] for row in rows),
            "cases": rows,
            "what_this_does_not_license": (
                "any throughput claim. A launch count is not a time, the fused route "
                "does MORE memory traffic than the two singles (three extra pointwise "
                "constitutive evaluations per component per thread), and a second "
                "Metal lane was running on this GPU throughout this campaign")}


def leg_sync(product: Product, steps: int, sync_steps: Sequence[int]) -> Dict[str, Any]:
    """The containment rule, and the armed hazard that must diverge.

    ``synchronize_magnetic_fields`` runs ``step_B``/``fill_B``/``update_H`` and consults
    the fast path under the by-name channel ``update_H_synchronize``. A plan spanning
    ``step_D`` is OUTSIDE that half-step, so it must DECLINE -- and the shim counts the
    refusals rather than trusting the plan. The armed arm answers the consult anyway,
    which is what a product that did not decline would do, and the leg requires ``D``
    to diverge.
    """
    from meep_gpu.fastpath import SYNC_PATH_SLOTS  # noqa: PLC0415

    name, spec = product.cases[0]

    def hook(step: int, arrangement: str, driver: Any) -> Any:  # noqa: ARG001
        if step in tuple(sync_steps):
            with contextlib.suppress(Exception):
                driver.synchronize_magnetic_fields()
                driver.restore_magnetic_fields()
        return None

    clean = run_case(product, name, spec, steps, with_composer=False,
                     per_step_hook=hook)
    armed = run_case(product, name, spec, steps, with_composer=False,
                     per_step_hook=hook, sync_hazard=True)
    return {
        "passed": (clean["identical_vs_array"] and not armed["identical_vs_array"]),
        "case": name, "sync_steps": list(sync_steps),
        "sync_path_slots": sorted(SYNC_PATH_SLOTS),
        "span": list(product.module.REPLACES),
        "outside_the_half_step": [slot for slot in product.module.REPLACES
                                  if slot not in SYNC_PATH_SLOTS],
        "declining_run_identical": clean["identical_vs_array"],
        "armed_run_diverged": not armed["identical_vs_array"],
        "armed_first_divergence": armed["first_divergence"],
        "armed_differing_words": armed["differing_words_final"],
    }


def leg_arbitration(product: Product) -> Dict[str, Any]:
    """THE INSTALLABLE CLAIM, MEASURED THROUGH THE SHIPPED COMPOSER.

    Not argued from a board: ``launch.plan_step(fuse=True)`` is asked on every fixture,
    with this product registered, and what it INSTALLED is read back. Recorded per
    fixture: whether either neighbouring seam is served by a released pair, how many
    pairs the composer installed, and how many launches the four-slot path costs with
    and without this product.

    RECORDED RATHER THAN ASSERTED, because the wiring diff this round writes inverts
    them: whether the composer names this product in its refusals at all (it can only
    do that once the product holds a ``FUSED_PAIR_ARMS`` row), and whether the family
    appears in ``registry.FAMILY_MODULES``.

    ASSERTED: this product installs on ZERO fixtures, and every released neighbour
    keeps the slot it holds today.
    """
    rows: List[Dict[str, Any]] = []
    for name, spec in product.cases:
        driver = build_driver(spec, 4, scale_bits=0)
        residency = Residency()
        plan = metal_launch.plan_step(
            driver.fields, driver.pml, residency=residency,
            sources=tuple(getattr(driver, "_sources", ())), fuse=True)
        selected = dict(plan.selected)
        # A SEAM IS SERVED BY A PAIR WHEN ONE LABEL SITS ON BOTH OF ITS SLOTS, and
        # that test is EXACT where a substring match is not. The composer's `selected`
        # carries each arm's human LABEL ("folded complex fused B/H pair"), not a
        # module name, so matching on `family` or on "magnetic_pair" reports every
        # served seam as unserved -- measured here 2026-09-07 before it was fixed:
        # eight fixtures came back "neighbours=0, verdict=GAIN" while the composer had
        # in fact installed both pairs on all eight. A fused product writes ONE label
        # into BOTH slots of its seam, which is what makes the equality the right test.
        def _spans(first: str, second: str) -> bool:
            left, right = selected.get(first), selected.get(second)
            return bool(left) and left == right

        b_to_h_served = _spans("step_B", "update_H")
        d_to_e_served = _spans("step_D", "update_E")
        this_product_installed = _spans("update_H", "step_D")
        installed_pairs = sorted({selected[slot] for pair in
                                  (("step_B", "update_H"), ("step_D", "update_E"),
                                   ("update_H", "step_D"))
                                  for slot in pair if _spans(*pair)})
        neighbours = int(b_to_h_served) + int(d_to_e_served)
        rows.append({
            "case": name,
            "composer_selected": selected,
            "pairs_installed": installed_pairs,
            "this_product_installed": this_product_installed,
            "b_to_h_seam_served_by_a_pair": b_to_h_served,
            "d_to_e_seam_served_by_a_pair": d_to_e_served,
            "seam_slot_labels": {slot: selected.get(slot)
                                 for slot in ("step_B", "update_H", "step_D",
                                              "update_E")},
            "neighbouring_seams_served": neighbours,
            "four_slot_launches_today": 4 - neighbours,
            "four_slot_launches_with_this_product": 4 - 1,
            "verdict": ("LOSS" if neighbours == 2 else
                        "TIE" if neighbours == 1 else "GAIN"),
            # RECORDED, NOT ASSERTED -- the wiring diff inverts both.
            "composer_named_this_product_in_its_refusals": sorted(
                str(reason) for key, reason in dict(plan.reasons).items()
                if product.family in str(key) or product.family in str(reason))[:4],
            "passed": not this_product_installed,
        })
        log(f"  arbitration {name}: neighbours={neighbours} "
            f"verdict={rows[-1]['verdict']} installed={this_product_installed}")
    verdicts = [row["verdict"] for row in rows]
    return {
        "passed": bool(rows) and all(row["passed"] for row in rows),
        "installable": product.module.INSTALLABLE,
        "cases": rows,
        "loss": verdicts.count("LOSS"), "tie": verdicts.count("TIE"),
        "gain": verdicts.count("GAIN"),
        "denominator": len(rows),
        "recorded_not_asserted": [
            "whether the composer names this product in its refusals (it can only do "
            "that once the product holds a launch.FUSED_PAIR_ARMS row, which the "
            "wiring diff adds)",
            "whether the family is in registry.FAMILY_MODULES (same diff)",
            "whether the product holds an absorb row",
        ],
        "asserted": ["the product installs on ZERO fixtures",
                     "the composer's selection is what it is with the arm registered"],
    }


def leg_mutation(product: Product, steps: int) -> Dict[str, Any]:
    """Every armed defect, each of which MUST diverge, each with its null pair.

    A mutation is scored CAUGHT only when the armed run diverges AND the same harness
    with the shipped bytes does not -- the second half is leg ``disarm``, run
    separately over the identical code path, so a harness that diverged anyway could
    not report a defect as caught.

    A mutation declared ``must_catch=False`` is a MEASURED EQUIVALENCE, not an
    exemption: it is armed, driven, and required NOT to diverge, and its citation is
    recorded beside it.
    """
    from meep_gpu.metal_kernels import shaders  # noqa: PLC0415

    rows: List[Dict[str, Any]] = []
    for label, entry in sorted(product.shader_mutations.items()):
        builder, why, must_catch = entry[0], entry[1], entry[2]
        # A MUTATION MAY DECLARE ITS OWN FIXTURES. A defect whose line is not
        # EMITTED on a fixture cannot be armed there, and arming it anyway would
        # report a no-op needle as an uncaught defect -- so the restriction is
        # declared beside the mutation rather than discovered at run time.
        for case_name in (entry[3] if len(entry) > 3 and entry[3]
                          else product.mutation_cases):
            spec = product.case(case_name)
            driver = build_driver(spec, 7, scale_bits=0)
            variant = variant_of(product, driver)
            codes = resolved_codes(product, driver)
            phased = _phased_of(product, driver)
            expansion = product.expansion() if product.expansion is not None else None
            try:
                mutated = builder(variant, codes, phased, expansion)
            except Exception as exc:  # noqa: BLE001 - a needle that stopped matching
                rows.append({"mutation": label, "case": case_name, "passed": False,
                             "reason": f"could not arm: {type(exc).__name__}: {exc}"})
                continue
            try:
                compiled = getattr(compile_source(mutated), product.module.KERNEL)
            except Exception as exc:  # noqa: BLE001
                rows.append({"mutation": label, "case": case_name, "passed": False,
                             "reason": f"armed source did not compile: {exc}"})
                continue
            row = run_case(product, case_name, spec, steps, with_composer=False,
                           functions={shaders.CONTRACT_OFF: compiled}, scale_bits=0)
            diverged = not row["identical_vs_array"]
            rows.append({
                "mutation": label, "case": case_name, "why": why,
                "must_catch": must_catch, "diverged": diverged,
                "first_divergence": row["first_divergence"],
                "differing_words": row["differing_words_final"],
                "passed": diverged == must_catch,
            })
            log(f"  mutation {label} on {case_name}: diverged={diverged} "
                f"(must_catch={must_catch})")

    for label, entry in sorted(product.host_mutations.items()):
        launcher, why = entry[0], entry[1]
        # A HOST MUTATION MAY DECLARE ITS OWN FIXTURES TOO, and one of them must: the
        # `_boundary_kinds` codes defect is a NO-OP on an unfolded fixture, because
        # there the plain expression IS the expression the plan uses. Arming it there
        # would report an unarmable defect as uncaught -- measured 2026-09-07 on the
        # beta gate's plain variant before the restriction was declared.
        for case_name in (entry[2] if len(entry) > 2 and entry[2]
                          else product.mutation_cases):
            spec = product.case(case_name)
            row = launcher(product, case_name, spec, steps)
            diverged = not row["identical_vs_array"]
            rows.append({
                "mutation": label, "case": case_name, "why": why,
                "must_catch": True, "diverged": diverged, "host_defect": True,
                "first_divergence": row["first_divergence"],
                "differing_words": row["differing_words_final"],
                "passed": diverged,
            })
            log(f"  mutation {label} on {case_name}: diverged={diverged}")

    armed = len(rows)
    caught = sum(1 for row in rows if row.get("diverged"))
    must = sum(1 for row in rows if row.get("must_catch"))
    return {
        "passed": bool(rows) and all(row["passed"] for row in rows),
        "mutations": rows, "armed": armed, "diverged": caught,
        "must_catch_armed": must,
        "must_catch_caught": sum(1 for row in rows
                                 if row.get("must_catch") and row.get("diverged")),
        "measured_equivalences": [row["mutation"] for row in rows
                                  if row.get("must_catch") is False],
    }


def leg_byte_neutral(product: Product, steps: int, edits: Sequence[Dict[str, Any]]
                     ) -> Dict[str, Any]:
    """The armed edits required NOT to diverge -- the complement of ``mutation``.

    A gate whose every armed edit diverges has not shown its comparator is
    discriminating; it has shown the kernel is fragile. Each entry here is a real edit
    to the emitted source that the arithmetic is INVARIANT under, driven and required
    to be byte-identical.
    """
    from meep_gpu.metal_kernels import shaders  # noqa: PLC0415

    rows: List[Dict[str, Any]] = []
    for entry in edits:
        case_name = entry.get("case", product.mutation_cases[0])
        spec = product.case(case_name)
        driver = build_driver(spec, 7, scale_bits=0)
        variant = variant_of(product, driver)
        codes = resolved_codes(product, driver)
        phased = _phased_of(product, driver)
        expansion = product.expansion() if product.expansion is not None else None
        try:
            edited = entry["build"](variant, codes, phased, expansion)
            compiled = getattr(compile_source(edited), product.module.KERNEL)
        except Exception as exc:  # noqa: BLE001
            rows.append({"edit": entry["name"], "passed": False,
                         "reason": f"could not arm: {exc}"})
            continue
        row = run_case(product, case_name, spec, steps, with_composer=False,
                       functions={shaders.CONTRACT_OFF: compiled}, scale_bits=0)
        rows.append({"edit": entry["name"], "case": case_name, "why": entry["why"],
                     "diverged": not row["identical_vs_array"],
                     "differing_words": row["differing_words_final"],
                     "passed": row["identical_vs_array"]})
        log(f"  byte_neutral {entry['name']}: identical={row['identical_vs_array']}")
    return {"passed": bool(rows) and all(row["passed"] for row in rows),
            "edits": rows}


def leg_disarm(product: Product, steps: int) -> Dict[str, Any]:
    """The identical harness, the SHIPPED bytes, required not to diverge.

    Every mutation leg above runs ``run_case`` with a compiled ``functions`` override.
    This runs the same call with the shipped source compiled the ordinary way, on the
    same fixtures at the same seed and the same (unscaled) amplitude the mutation legs
    use -- so a "caught" verdict cannot be a harness that diverges anyway.
    """
    from meep_gpu.metal_kernels import shaders  # noqa: PLC0415

    rows: List[Dict[str, Any]] = []
    for case_name in product.mutation_cases:
        spec = product.case(case_name)
        driver = build_driver(spec, 7, scale_bits=0)
        variant = variant_of(product, driver)
        codes = resolved_codes(product, driver)
        phased = _phased_of(product, driver)
        expansion = product.expansion() if product.expansion is not None else None
        shipped = _emit(product, variant, codes, phased, expansion)
        compiled = getattr(compile_source(shipped), product.module.KERNEL)
        row = run_case(product, case_name, spec, steps, with_composer=False,
                       functions={shaders.CONTRACT_OFF: compiled}, scale_bits=0)
        rows.append({"case": case_name, "identical": row["identical_vs_array"],
                     "steps_compared": row["steps_compared"],
                     "differing_words": row["differing_words_final"],
                     "passed": row["identical_vs_array"]})
        log(f"  disarm {case_name}: identical={row['identical_vs_array']}")
    return {"passed": bool(rows) and all(row["passed"] for row in rows),
            "cases": rows,
            "why": ("the mutation legs run through this exact call with a compiled "
                    "override; if the harness diverged on the shipped bytes too, every "
                    "'caught' verdict above would be worthless")}


# ---------------------------------------------------------------------------
# The lift
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            with contextlib.suppress(json.JSONDecodeError):
                rows.append(json.loads(line))
    return rows


def lift_basis(product: Product, results: Path) -> Tuple[List[dict], Dict[str, Any]]:
    """The rows the standing census puts in THIS product's cells, and the seam facts.

    DERIVED from the census's own ``plan_step.selected`` -- the arms the composer chose
    for each row -- never from a list here, so a row that moved cells moves with it.
    """
    census = results / product.census
    seam = results / product.seam_record
    if not census.is_dir() or not seam.is_dir():
        raise SystemExit(f"the lift leg needs {census} and {seam}")
    seam_rows = {row["label"]: row for row in load_jsonl(seam / "h_to_d_seam.jsonl")}
    wanted = {tuple(arms): variant for variant, arms in product.cell_arms.items()}
    rows: List[dict] = []
    for leg in ("examples", "tests", "tests_param_matched"):
        path = census / f"{leg}.jsonl"
        if not path.is_file():
            continue
        for record in load_jsonl(path):
            selected = ((record.get("plan_step") or {}).get("selected") or {})
            key = (selected.get("update_H"), selected.get("step_D"))
            if key not in wanted:
                continue
            label = f"{record.get('leg', leg)}:{record['row']}"
            seam_row = seam_rows.get(label, {})
            rows.append({
                "label": label, "leg": record.get("leg", leg), "row": record["row"],
                "variant": wanted[key],
                # THE CASE NAME IS NOT ALWAYS THE ROW NAME. A `parameterized` test is
                # lifted by the census under its DECORATED name while the row it files
                # is the expanded one; handing `--child-cases` the row name asks for a
                # case the module does not define, and the child exits 0 having
                # selected nothing -- a row scored unmeasurable by the harness rather
                # than by the engine.
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
    facts = {
        "census": product.census, "seam_record": product.seam_record,
        "cell_arms": {variant: list(arms)
                      for variant, arms in product.cell_arms.items()},
        "rows_in_the_cells": len(unique),
        "rows_per_variant": {variant: sum(1 for row in unique.values()
                                          if row["variant"] == variant)
                             for variant in product.cell_arms},
        "rows_with_a_standing_withdraw": sorted(
            label for label, row in unique.items() if row["withdraw_in_seam"]),
    }
    return list(unique.values()), facts


def carries_a_measurement(path: Path, key: str) -> bool:
    """Did a child's record actually measure this row? Not "does the file exist"."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - an unreadable record is not a measurement
        return False
    candidates = payload if isinstance(payload, list) else [payload]
    return any(key in (entry or {}) for entry in candidates)


def leg_lift(product: Product, out_dir: Path, steps: int,
             max_cells: Optional[int], timeout: float, resume: bool,
             only: Optional[Sequence[str]]) -> Dict[str, Any]:
    """Every corpus row in the product's cells, re-lifted in its own interpreter.

    Rule 7: one JSON per row as it lands, plus a progress log the child appends to per
    step, so an interrupted lift keeps everything up to the failure.
    """
    import measure_predicate_coverage as census_module  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    out_dir = Path(out_dir).resolve()
    rows, facts = lift_basis(product, HERE / "results")
    if only:
        wanted = set(only)
        rows = [row for row in rows if row["label"] in wanted]
    lift_dir = out_dir / "lift"
    per_row = lift_dir / "per_row"
    per_row.mkdir(parents=True, exist_ok=True)
    progress_log = lift_dir / "steps.progress.log"
    examples_dir = Path(census_module.EXAMPLES_DIR)
    tests_dir = Path(census_module.TESTS_DIR)
    if not examples_dir.is_dir() or not tests_dir.is_dir():
        return {"passed": False, "facts": facts,
                "reason": (f"the MEEP corpus is not at {examples_dir} / {tests_dir}; "
                           f"refused rather than measured on nothing")}
    probe_path = API_ROOT / census_module.PROBE
    prefix = product.lift_environment_prefix
    environment = dict(os.environ)
    environment.update({
        "KMP_DUPLICATE_LIB_OK": "TRUE", "MPLBACKEND": "Agg",
        "PYTHONPATH": str(API_ROOT),
        f"{prefix}_STEPS": str(steps),
        f"{prefix}_PROGRESS": str(progress_log)})
    if max_cells is not None:
        environment[f"{prefix}_MAX_CELLS"] = str(max_cells)
    work = lift_dir / "workdir"
    work.mkdir(exist_ok=True)
    for entry in examples_dir.iterdir():
        if entry.suffix in (".py", ".ipynb"):
            continue
        link = work / entry.name
        if not link.exists():
            with contextlib.suppress(OSError):
                link.symlink_to(entry)
    # THE CENSUS'S OWN SHIM, for the MEEP test modules that decorate with
    # ``parameterized`` (not installed here). Without it every row from one of those
    # modules dies on the IMPORT and comes back ``measured: false`` -- a row scored as
    # unmeasurable by the harness rather than by the engine.
    shim_path = Path(census_module.__file__).resolve().parent / "shim"
    needs_shim: set = set()
    for row in rows:
        module_name = row.get("module")
        if row["leg"] == "examples" or not module_name:
            continue
        module_file = tests_dir / module_name
        if module_file.is_file() and "import parameterized" in module_file.read_text(
                encoding="utf-8", errors="replace"):
            needs_shim.add(module_name)
    key = f"{product.family}_gate"
    measured: List[dict] = []
    for index, row in enumerate(rows, start=1):
        record_path = per_row / (row["label"].replace(":", "__").replace(".", "_")
                                 + ".json")
        environment[f"{prefix}_LABEL"] = row["label"]
        environment["PYTHONPATH"] = (f"{shim_path}{os.pathsep}{API_ROOT}"
                                     if row.get("module") in needs_shim
                                     else str(API_ROOT))
        started = time.time()
        reusable = resume and record_path.exists() and carries_a_measurement(
            record_path, key)
        if not reusable:
            if row["leg"] == "examples":
                command = [row["interpreter"], "-u",
                           str(Path(census_module.__file__).resolve()),
                           "--leg", "examples", "--child-script",
                           str(examples_dir / row["row"]),
                           "--out-json", str(record_path), "--probe", str(probe_path),
                           "--progress-log", str(progress_log),
                           "--battery", product.gate_stem]
            elif not row["module"]:
                measured.append({**row, "measured": False,
                                 "note": "no module recorded"})
                log(f"lift {index}/{len(rows)} {row['label']}: REFUSED, no module")
                continue
            else:
                command = [row["interpreter"], "-u",
                           str(Path(census_module.__file__).resolve()),
                           "--leg", "tests", "--child-module",
                           str(tests_dir / row["module"]),
                           "--child-cases", json.dumps([row.get("case")
                                                        or row["row"]]),
                           "--out-json", str(record_path), "--probe", str(probe_path),
                           "--progress-log", str(progress_log),
                           "--battery", product.gate_stem]
            log(f"lift {index}/{len(rows)} {row['label']} start "
                f"(cells {row.get('grid_cells')})")
            stderr_text = ""
            try:
                completed = subprocess.run(command, cwd=str(work), env=environment,
                                           timeout=timeout, stdout=subprocess.DEVNULL,
                                           stderr=subprocess.PIPE, check=False)
                stderr_text = (completed.stderr or b"").decode("utf-8", "replace")
            except subprocess.TimeoutExpired as expired:
                stderr_text = ((expired.stderr or b"").decode("utf-8", "replace")
                               if expired.stderr else "child timeout")
        payload: List[dict] = []
        with contextlib.suppress(Exception):
            loaded = json.loads(record_path.read_text(encoding="utf-8"))
            payload = loaded if isinstance(loaded, list) else [loaded]
        block = next((entry.get(key) for entry in payload
                      if isinstance(entry, dict) and entry.get(key)), None)
        entry = {**row, "seconds": round(time.time() - started, 1),
                 "measured": block is not None, "gate": block}
        if block is None:
            entry["stderr_tail"] = (stderr_text or "")[-600:]
        measured.append(entry)
        with (lift_dir / "rows.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        log(f"lift {index}/{len(rows)} {row['label']}: "
            f"measured={entry['measured']} "
            f"identical={(block or {}).get('identical')} "
            f"steps={(block or {}).get('steps_compared')} "
            f"({entry['seconds']}s)")

    driven = [entry for entry in measured if (entry.get("gate") or {}).get("driven")]
    identical = [entry for entry in driven if entry["gate"].get("identical")]
    admitted = [entry for entry in measured
                if (entry.get("gate") or {}).get("predicate_admits")]
    floor_met = [entry for entry in identical
                 if int(entry["gate"].get("steps_compared") or 0)
                 >= LIFT_CLEAN_STEP_FLOOR]
    withdraw_rows = [entry for entry in measured if entry["withdraw_in_seam"]]
    refused_withdraw = [entry for entry in withdraw_rows
                        if not (entry.get("gate") or {}).get("predicate_admits", True)]
    return {
        "passed": bool(measured)
        and len(identical) == len(driven)
        and len(driven) == len(admitted)
        and len(admitted) == len(measured) - len(withdraw_rows)
        and len(floor_met) == len(identical)
        and len(refused_withdraw) == len(withdraw_rows),
        "facts": facts,
        "rows_in_the_cells": len(measured),
        "rows_measured": sum(1 for entry in measured if entry["measured"]),
        "rows_the_predicate_admitted": len(admitted),
        "rows_driven": len(driven),
        "rows_driven_identical": len(identical),
        "rows_reaching_the_clean_step_floor": len(floor_met),
        "clean_step_floor": LIFT_CLEAN_STEP_FLOOR,
        "rows_with_a_standing_withdraw": len(withdraw_rows),
        "withdraw_rows_refused_by_name": len(refused_withdraw),
        "words_compared": int(sum(int((entry.get("gate") or {}).get("words_compared")
                                      or 0) for entry in driven)),
        "rows": measured,
    }


def evaluate_row(product: Product, driver: Any, steps: int) -> Dict[str, Any]:
    """The census driver's battery hook, shared: the identity claim on ONE lifted row.

    Called from each gate's own ``evaluate`` (the census names the GATE module as its
    ``--battery``, so the hook itself cannot live here).
    """
    from meep_gpu import withdraw_hoist  # noqa: PLC0415

    sources = tuple(getattr(driver, "_sources", ()) or ())
    block: Dict[str, Any] = {
        "family": product.family, "steps_requested": steps,
        "n_sources": len(sources),
        "sources": [{"type": type(s).__name__,
                     "field_type": str(getattr(s, "field_type", "")),
                     "is_integrated": bool(getattr(s, "is_integrated", False)),
                     "withdraw_does_work": bool(
                         withdraw_hoist._withdraw_does_work(s))}  # noqa: SLF001
                    for s in sources]}
    pin_array_path(driver)
    verdict = product.coverage(driver.fields, driver.pml, sources, Residency())
    block["predicate_admits"] = bool(verdict.covered)
    block["predicate_reasons"] = list(verdict.reasons)[:8]
    with contextlib.suppress(Exception):
        block["variant"] = variant_of(product, driver)
        codes = resolved_codes(product, driver)
        block["codes"] = [int(code) for code in codes]
        block["codes_label"] = codes_label(codes)
    block["grid_cells"] = int(np.prod(driver.grid.shape))
    block["grid_shape"] = [int(n) for n in driver.grid.shape]
    if not verdict.covered:
        block["driven"] = False
        return block
    try:
        weld, plan = arrangement_weld(product, driver)
    except Exception as exc:  # noqa: BLE001 - a refusal at plan time is a result
        block["driven"] = False
        block["plan_error"] = f"{type(exc).__name__}: {exc}"
        return block
    arrangements = {"array": arrangement_array(), "weld": weld}
    result = drive(driver, arrangements, steps, reference="array")
    leg = result["arrangements"]["weld"]
    block.update({
        "driven": True,
        "identical": bool(leg["identical"]),
        "steps_compared": result["steps_compared"],
        "steps_stepped": result["steps_stepped"],
        "first_banded_step": result["first_banded_step"],
        "words_compared": result["words_compared"],
        "first_divergence": leg["first_divergence"],
        "differing_words_final": leg["differing_words_final"],
        "differing_volumes_final": leg["differing_volumes_final"],
        "launches": leg["launches"],
        "dispatched": leg["dispatched"], "absorbed": leg["absorbed"],
        "expansion": getattr(plan, "expansion", None),
    })
    return block


def runtime_reasons() -> List[str]:
    """Why THIS process cannot evaluate these batteries at all, or ``[]``."""
    return list(_shared.runtime_reasons())


def host_block(product: Product, gate_file: str) -> Dict[str, Any]:
    """The environment every artifact records, and the two digests it is welded to."""
    import torch  # noqa: PLC0415

    from meep_gpu.metal_kernels.device import metal_frontend_version  # noqa: PLC0415

    report = subnormal.mps_policy_report()
    return {
        "passed": True,
        "policy": os.environ.get("MEEP_GPU_SUBNORMAL_POLICY"),
        "subnormal_policy_report": report,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "metal_frontend": metal_frontend_version(),
        "cache_dir": os.environ.get("CUPY_CACHE_DIR"),
        "source_sha256": {
            "family": sha256_of(product.module.__file__),
            "gate": sha256_of(gate_file),
            "harness": sha256_of(__file__),
            "sibling_harness": sha256_of(_shared.__file__),
        },
    }


# ---------------------------------------------------------------------------
# The runner — one main() for the three gates
# ---------------------------------------------------------------------------

#: The order legs run in: cheap host checks first, so a structural failure is reported
#: before an hour of device work. A partial run exits :data:`EXIT_INCOMPLETE`, which
#: makes ``release.released`` False -- nothing can be minted from it.
LEG_GROUPS: Dict[str, Tuple[str, ...]] = {
    "host": ("host", "policy", "driver_order", "transcription", "refusal",
             "withdraw"),
    "compile": ("binding_ceiling",),
    "device": ("product", "seed_scale", "purity", "ghost_observability", "masks",
               "launch_structure", "sync", "arbitration", "lift", "byte_neutral",
               "mutation", "disarm"),
}
ALL_LEGS: Tuple[str, ...] = (LEG_GROUPS["host"] + LEG_GROUPS["compile"]
                             + LEG_GROUPS["device"])


def run_gate(product: Product, gate_file: str, builders: Mapping[str, Any],
             argv: Optional[Sequence[str]] = None) -> int:
    """Parse, run the requested legs, write the artifact, return the exit code.

    ``builders`` maps a leg name to a zero-argument callable the GATE supplies, so
    every product-specific decision stays in the gate script while the bookkeeping --
    the flushed per-leg log, the fsynced jsonl, the completeness rule and the
    provenance stamp -- is one implementation.
    """
    import argparse  # noqa: PLC0415

    import gate_provenance  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=f"device gate for {product.family}")
    parser.add_argument("--out", type=Path, required=True, help="artifact JSON path")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--legs", default="all")
    parser.add_argument("--lift-steps", type=int, default=12)
    parser.add_argument("--lift-max-cells", type=int, default=None)
    parser.add_argument("--lift-timeout", type=float, default=900.0)
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

    for leg in ALL_LEGS:
        if leg not in legs:
            continue
        builder = builders.get(leg)
        if builder is None:
            record(leg, {"passed": False,
                         "reason": f"this gate declares no builder for leg {leg}"})
            continue
        log(f"LEG {leg}: start")
        try:
            record(leg, builder(args))
        except Exception as exc:  # noqa: BLE001 - a leg that raised is a FAILED leg
            import traceback  # noqa: PLC0415

            traceback.print_exc()
            record(leg, {"passed": False,
                         "reason": f"{type(exc).__name__}: {exc}",
                         "traceback": traceback.format_exc()[-2000:]})

    ran = tuple(dict.fromkeys(row["leg"] for row in rows))
    complete = all(leg in ran for leg in ALL_LEGS)
    all_passed = bool(rows) and all(row.get("passed") for row in rows)
    if complete:
        verdict = "PASS" if all_passed else "FAIL"
    else:
        verdict = ("INCOMPLETE: not every leg ran -- "
                   f"{'all requested legs passed' if all_passed else 'a requested leg FAILED'}; "
                   f"missing {sorted(set(ALL_LEGS) - set(ran))}")

    result: Dict[str, Any] = {
        "gate": Path(gate_file).name,
        "verdict": verdict,
        "all_requested_legs_passed": all_passed,
        "legs_ran": list(ran),
        "legs_declared": list(ALL_LEGS),
        "complete": complete,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "rows": rows,
        "product": {
            "family": product.family, "slot": product.module.SLOT,
            "replaces": list(product.module.REPLACES), "seam": product.module.SEAM,
            "installable": product.module.INSTALLABLE,
            "installable_reason": product.module.INSTALLABLE_REASON,
            "hoists_the_withdraw": product.module.HOISTS_THE_WITHDRAW,
            "carries_deposit_repair": product.module.CARRIES_DEPOSIT_REPAIR,
            "weld_owed": product.module.WELD_OWED,
            "variants": list(getattr(product.module, "VARIANTS", ("folded",))),
            "cells": {variant: list(arms)
                      for variant, arms in product.cell_arms.items()},
            # RECORDED, NEVER ASSERTED: the wiring diff this round writes inverts both.
            "absorb_row_today": metal_launch.FUSED_PAIR_ARMS.get(product.family),
            "in_registry_FAMILY_MODULES_today": _in_family_modules(product),
        },
        "corpus": {"census": product.census, "seam_record": product.seam_record},
        **host_block(product, gate_file),
        "what_this_does_not_license": (
            "installing the product (INSTALLABLE is False and the arbitration leg "
            "measures why), flipping HOISTS_THE_WITHDRAW (no leg here drives a hoisted "
            "launch, so a row with a standing in-seam withdraw stays refused by name), "
            "ANY throughput claim (launch counts are not time, the fused route does "
            "more memory traffic than the two singles, and a second Metal lane was "
            "running on this GPU throughout this campaign), any cell outside this "
            "product's own two arm pairs, any step past a row's own first banded step, "
            "and any dispatch claim -- the Metal backend is not planned against"),
    }
    result.pop("passed", None)
    gate_provenance.stamp(result)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str)
                        + "\n", encoding="utf-8")
    log(f"VERDICT {verdict} in {result['elapsed_seconds']:.2f}s; artifact {args.out}")
    if not all_passed:
        return 1
    return 0 if complete else EXIT_INCOMPLETE


def _in_family_modules(product: Product) -> bool:
    from meep_gpu.metal_kernels import registry  # noqa: PLC0415

    return (product.family in registry.FAMILY_MODULES
            or Path(product.module.__file__).stem in registry.FAMILY_MODULES)
