"""The Metal complex/Bloch kernels inside a REAL ``FdtdDriver.step()``, for a budget.

WHAT THIS MEASURES THAT THE GATE AND THE COMPOSITION PROBE DO NOT.

``gate_metal_complex.py`` certifies four sub-steps in ISOLATION, from a seeded
state, against ``stepping.py``. ``probe_metal_complex_composition.py`` runs the four
back to back on one residency and measures the mirror, the seam refusal and the
zero-init signed-zero class. NEITHER OF THEM HAS EVER RUN ``driver.step()``, and
neither compares PER STEP: both take one comparison at the end of four cycles. That
leaves five things unmeasured, and each is a place a certified kernel can still
produce a wrong RUN:

1. **The complete driver step**, in the driver's own order (driver.py:3277-3306):
   ``step_B`` -> magnetic sources -> symmetry fill -> ``zero_metal_B`` -> folded
   ghosts -> ``update_H`` -> ``step_D`` -> electric sources -> symmetry fill ->
   ``zero_metal_D`` -> folded ghosts -> ``update_E`` -> ``update_P``. The array
   passes are INTERLEAVED with the device sub-steps and every one of them writes a
   volume a mirror shadows.
2. **Per-step comparison.** A comparison taken only at the end of N cycles reports
   the CONSEQUENCES of a divergence and not its cause. Here the full stored
   inventory is compared as uint32 words after EVERY step and the leg stops at the
   first one, so a failure names the case, the step and the array.
3. **A source actually injecting.** The composition probe's seam leg asks the
   coverage question — with a declared source, may a mirror set be held? — and
   answers no. It never injects anything. Both seams are exercised here: a magnetic
   current between ``step_B`` and ``update_H``, an electric one between ``step_D``
   and ``update_E``.
4. **The budget.** One step cannot show a defect that accumulates. The auxiliaries
   (``fu_*``, ``f_w_*``) are STATE, so a kernel that is right on launch 1 and wrong
   forever after only diverges over a budget.
5. **That the device path RAN AT ALL.** Every number below would be produced,
   unchanged and green, by a probe whose shim answered ``False`` to every slot and
   let the array path do all the work — because THE ORACLE IS THE ARRAY PATH. The
   dispatch counters, the plans' own launch counters and the driver-level armed
   mutations are what separate "the Metal kernels reproduce the array path" from
   "the array path reproduces itself".

THE INTERMEDIATE CENSUS IS TAKEN AT THE SEAM, which is this probe's one improvement
on the pattern it was ported from. The shim is called with the live ``Fields`` object
IMMEDIATELY BEFORE each plan launches, so the operands reconstructed there are the
exact arrays the kernel is about to read — the source deposits included. A census
that reconstructed from a post-step snapshot would be reconstructing from operands
the seam had already moved.

THE COMPOSITION IS THE DECLARED-SYNC ONE and no mirror is held across an array pass:
the composition probe's residency leg measured which slots must be synced (the wall
passes on a metallic axis, and both source fills), so a probe that held mirrors
across a real driver step would be measuring a composition its own family refused.
Each dispatched slot syncs in, launches and syncs out. That costs the round trip the
residency layer exists to avoid; this probe measures CORRECTNESS and the throughput
question belongs to a bench.

MEASURED WHILE WRITING THIS, and recorded because it decided the case matrix rather
than being an aside: ``driver.setup_pml``'s docstring (driver.py:2160-2168) states
that "a nonzero ``k_point`` component is refused on an axis that carries an
absorber" and that ``k_point=(0.3, 0, 0)`` with a scalar PML "still raises". NO SUCH
REFUSAL EXISTS IN THE CODE. Measured on this host: that exact pairing builds with
``pml.is_active = True``, in the scalar, per-axis and z-only spellings alike. The
case matrix therefore reuses the gate's OWN configurations unchanged — PML on a
phased axis included, which is the pairing ``stepping._bloch_phases`` admits — so
the driver leg covers exactly the surface the gate certified rather than a subset of
it. The docstring is wrong about its own module and that is reported upward; nothing
here edits ``driver.py``.

STATED WEAKNESSES, because the artifact inherits them:

* there is NO PTX-EQUIVALENT AUDIT on this executor (``compile_shader`` exposes no
  disassembly), so the mutation leg and the launch counters are the only arbiters
  that the device did what the source says;
* the subnormal census reads STORED state plus the operands and intermediates
  reconstructed on the host at the seam, NOT the kernel's own registers, which are
  not readable from here.

    python -u probe_metal_complex_driver_step.py \\
        --out results/metal_complex_2026-08-15/driver_step.json \\
        --probe results/metal_complex_2026-08-15/complex_expansion_probe.json

Progress is one flushed line per case and the artifact is rewritten atomically after
every case (the progress-reporting rule).
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
API_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
if API_ROOT not in sys.path:
    sys.path.insert(0, API_ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# The MPS executor delivers `flush` natively and cannot deliver `keep`, and this
# arm64 host's DEFAULT resolves to keep. Requested EXPLICITLY, before anything
# resolves a policy; `setdefault` leaves an explicit caller override in place.
os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")

import gate_metal_complex as gate  # noqa: E402
import metal_gate_kit as kit  # noqa: E402

from meep_gpu import fastpath  # noqa: E402
from meep_gpu.driver import FdtdDriver  # noqa: E402
from meep_gpu.metal_kernels import complex_fields as cx  # noqa: E402
from meep_gpu.metal_kernels import device, preconditions, shaders, subnormal  # noqa: E402

log, save, differing = kit.log, kit.save, kit.differing

#: THE STEP BUDGET, STATED. 24 steps at courant 0.35 on a resolution-10 grid is
#: dt = 0.035, so the f = 1.0 gaussian's carrier turns over roughly every 28 steps:
#: the budget spans the pulse leaving its launch cell, crossing the interior and
#: entering the absorber, which is the span over which a stale-auxiliary defect
#: compounds. Every row records the budget it ran and the leg ASSERTS that the
#: budget it planned is the number of steps the driver actually took.
STEP_BUDGET = 24

#: The census runs on EVERY step of the budget, not a sample: a window taken at
#: step 0 of a 24-step run certifies step 0 (preconditions.SubnormalWindow's own
#: docstring). Cheap here — 26 volumes on a 1,080-cell grid.
CENSUS_EVERY = 1

#: A SHORT budget for the mutation leg, stated: these rows measure whether the
#: device reaches the compared bytes inside a real step, not how a defect
#: accumulates. The gate's own mutation leg is where the arithmetic is armed
#: exhaustively, at 23 rows.
MUTATION_BUDGET = 4

#: The seed for the physical-band initial state of the SEEDED variant. The sourced
#: variant starts from the driver's own zeros, which is the physical case.
SEED = 20260815

TALLY = {"compared": 0, "disagreed": 0}

#: Bound from the probe artifact before any leg runs. Deliberately NOT defaulted:
#: the arm is a MEASURED platform fact and a leg that ran before it was bound would
#: be binding a guess.
EXPANSION: Optional[str] = None
PROBE: Any = None


# ---------------------------------------------------------------------------
# The case matrix — the GATE'S OWN, so the driver leg covers what was certified
# ---------------------------------------------------------------------------

#: Non-power-of-two FIRST, and it is the courant every case runs: the PML
#: coefficients are the same family of non-representable multiplicands at any
#: courant, and the two configurations that also run at 0.5 are named below.
COURANT = 0.35

#: The Brillouin-edge case and one metallic case also run at courant 0.5, so the
#: budget is exercised at both a representable and a non-representable dt without
#: doubling the whole matrix.
SECOND_COURANT_CASES = ("periodic_edge_x", "metallic_xy_kz")
SECOND_COURANT = 0.5


def cases() -> Tuple[Dict[str, Any], ...]:
    """(name, cell, boundaries, k, courant) rows built from :data:`gate.CONFIGS`.

    READ FROM THE GATE rather than transcribed. A case added to the gate's matrix
    joins this probe automatically; a transcribed copy is how a driver leg quietly
    stops covering what the gate certifies.
    """
    rows: List[Dict[str, Any]] = []
    for name, cell, boundaries, k_point in gate.CONFIGS:
        rows.append({"name": name, "cell": cell, "boundaries": boundaries,
                     "k": k_point, "courant": COURANT})
        if name in SECOND_COURANT_CASES:
            rows.append({"name": f"{name}@C{SECOND_COURANT}", "cell": cell,
                         "boundaries": boundaries, "k": k_point,
                         "courant": SECOND_COURANT})
    return tuple(rows)


CASES = cases()

#: A gaussian pulse on each source family. BOTH SEAMS ARE EXERCISED: the magnetic
#: current lands between `step_B` and `update_H` and the electric one between
#: `step_D` and `update_E` (driver.py:3282-3305, MEEP step.cpp:64-100), so a probe
#: carrying only one would leave half the interleave unmeasured. Both centres sit
#: in the interior: the cell spans |x| < 0.6, |y| < 0.5, |z| < 0.45 and the 2-cell
#: absorber is 0.2 deep, so these are clear of it and the pulse has to REACH the
#: layer over the budget rather than starting inside it.
SOURCES = (
    {"component": "Ez", "center": (0.10, 0.05, 0.00), "size": (0.0, 0.0, 0.0),
     "frequency": 1.0, "source_type": "gaussian", "fwidth": 0.5},
    {"component": "Hy", "center": (-0.10, -0.05, 0.05), "size": (0.0, 0.0, 0.0),
     "frequency": 1.0, "source_type": "gaussian", "fwidth": 0.5},
)


# ---------------------------------------------------------------------------
# The stored inventory, ENUMERATED from the object
# ---------------------------------------------------------------------------

def inventory(fields: Any, shape: Sequence[int]) -> Tuple[str, ...]:
    """Every ndarray the Fields object holds at grid shape, by name.

    ENUMERATED, NOT TRANSCRIBED. ``gate.STATE`` is a hand-written list of the 24
    volumes a complex step reads or writes; this reads the object. The material
    volumes are compared too, and they are the reason to enumerate: nothing in a
    curl should write epsilon, so a kernel that scribbles past its target lands
    there, and a transcribed list would be exactly the list that cannot see it.

    BOTH STORAGE SHAPES ARE READ. ``Fields`` carries an instance ``__dict__`` today
    and the slotted spelling is what a memory-conscious refactor would reach for;
    reading only one and finding nothing is how a "full inventory" silently becomes
    an empty one, which is what the caller's floor below refuses.
    """
    names: List[str] = []
    holder = getattr(fields, "__dict__", None)
    candidates = list(holder) if holder else []
    for klass in type(fields).__mro__:
        candidates.extend(getattr(klass, "__slots__", ()))
    for name in candidates:
        value = getattr(fields, name, None)
        if isinstance(value, np.ndarray) and tuple(value.shape) == tuple(shape):
            names.append(name)
    return tuple(sorted(set(names)))


def snapshot(fields: Any, names: Sequence[str]) -> Dict[str, Any]:
    return {name: np.array(getattr(fields, name), copy=True) for name in names}


def compare(got: Dict[str, Any], oracle: Dict[str, Any]) -> Dict[str, int]:
    """Per-array differing WORDS over the whole inventory, counted into the tally."""
    out = {name: differing(got[name], oracle[name]) for name in oracle}
    TALLY["compared"] += len(oracle)
    bad = {name: count for name, count in out.items() if count}
    TALLY["disagreed"] += len(bad)
    return bad


# ---------------------------------------------------------------------------
# Case construction, through the ENGINE'S OWN driver
# ---------------------------------------------------------------------------

def build_driver(case: Dict[str, Any], sourced: bool = True,
                 seed: Optional[int] = None, scale: float = 1.0) -> FdtdDriver:
    """One driver for a complex/Bloch case, built through the public engine API.

    ``sourced`` is the control axis: the source-free driver is what makes the
    sourced run's movement attributable to the sources rather than to the initial
    state. ``seed`` seeds a physical-band initial state with a +-0 lattice on top,
    so a run whose first step would otherwise act on all zeros still moves state at
    step 1 — zero init is a FIXED POINT of the constitutive sub-step.
    """
    driver = FdtdDriver(cell_size=case["cell"], resolution=10.0,
                        courant=case["courant"], force_complex_fields=True,
                        boundaries=case["boundaries"], dimensions=3,
                        k_point=case["k"])
    # The gate's own absorber: 2 cells on every axis wide enough to hold one. The
    # pairing with a phased axis is what `stepping._bloch_phases` admits and what
    # the gate certifies; see the module docstring for the measured note on
    # setup_pml's docstring claiming a refusal its code does not implement.
    thickness = {axis: 2 for index, axis in enumerate("xyz")
                 if driver.grid.shape[index] >= 6}
    driver.setup_pml(thickness)
    shape = driver.grid.shape
    index = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    epsilon = (1.45 + 0.30 * np.sin(index * np.float32(0.037))).astype(np.float32)
    driver.fields.set_isotropic_epsilon_volume(
        epsilon, (np.float32(1.0) / epsilon).astype(np.float32))
    if sourced:
        for source in SOURCES:
            driver.add_source(dict(source))
    if seed is not None:
        rng = np.random.default_rng(seed)
        for name in gate.STATE:
            array = getattr(driver.fields, name, None)
            if array is None:
                continue
            real = (rng.standard_normal(shape) * (0.37 * scale)).astype(np.float32)
            imag = (rng.standard_normal(shape) * (0.29 * scale)).astype(np.float32)
            # The +-0 lattice rides the seeded variant. The driver's own zeros carry
            # only +0, and the sign of a zero is exactly the class a magnitude
            # comparison cannot see and a uint32 one can.
            real.reshape(-1)[::17] = np.float32(-0.0)
            real.reshape(-1)[7::23] = np.float32(0.0)
            imag.reshape(-1)[3::19] = np.float32(-0.0)
            imag.reshape(-1)[11::29] = np.float32(0.0)
            array[...] = gate.complex_from_planes(real, imag)
    # The epsilon volumes and the seeded state are both plan inputs; a cached plan
    # pointer from before them would be stale. The driver's own invalidation is the
    # supported way to say so.
    driver.invalidate_fast_path()
    return driver


# ---------------------------------------------------------------------------
# The dispatch shim — the ONLY thing this probe installs
# ---------------------------------------------------------------------------

class MetalShim:
    """A ``FastPathPlan``-shaped object that runs this family's Metal plans.

    THE SEAM IS THE ENGINE'S OWN and nothing about dispatch is changed to use it:
    ``driver.step`` reads ``self._fast_path`` once per step and consults
    ``fast.dispatch(slot, fields)`` at four identical sites (driver.py:3281-3305).
    This object answers that call. The shipped planner is untouched, this family
    stays unwired, and the probe is a MEASUREMENT of what wiring it would produce —
    the only honest way to run it before the wiring exists.

    EVERY DISPATCH IS COUNTED and the counts are asserted against the budget. Both
    halves are recorded: ``dispatched`` (True — the device ran the sub-step) and
    ``declined`` (False — the array path did). A shim that quietly declined
    everything would still produce a green byte comparison, because the oracle IS
    the array path, so the counters are the only thing standing between this probe
    and a pass that measures nothing.

    ``census`` is called with the LIVE fields immediately before the launch, which
    is the one moment at which the kernel's operands are exactly the arrays on the
    object — after the seam deposits, before the sub-step overwrites them.
    """

    __slots__ = ("fields", "plans", "residency", "dispatched", "declined",
                 "syncs_in", "syncs_out", "census")

    def __init__(self, fields: Any, plans: Dict[str, Any], residency: Any,
                 census: Any = None) -> None:
        self.fields = fields
        self.plans = {name: plan for name, plan in plans.items() if plan is not None}
        self.residency = residency
        self.dispatched: Dict[str, int] = {}
        self.declined: Dict[str, int] = {}
        self.syncs_in = 0
        self.syncs_out = 0
        self.census = census

    def dispatch(self, slot: str, fields: Any) -> bool:
        plan = self.plans.get(slot)
        if plan is None or fields is not self.fields:
            self.declined[slot] = self.declined.get(slot, 0) + 1
            return False
        self.residency.sync_in()
        self.syncs_in += 1
        if self.census is not None:
            self.census(slot, fields)
        plan.run()
        self.residency.sync_out()
        self.syncs_out += 1
        self.dispatched[slot] = self.dispatched.get(slot, 0) + 1
        return True

    @property
    def launches(self) -> int:
        """Launches the PLANS counted, which is a different witness from the shim's.

        The shim counts calls it answered True to; each plan counts its own
        ``run()``. Asserting the two agree catches a shim that reported a dispatch
        it did not perform.
        """
        return sum(plan.launches for plan in self.plans.values())


def build_plans(driver: FdtdDriver,
                functions: Optional[Dict[str, Dict[str, Any]]] = None
                ) -> Tuple[Dict[str, Any], Any]:
    """The four plans a wired version of this family would build for one driver.

    ``functions`` is the MUTATION SEAM: a per-slot function map that replaces the
    shipped one after the plan is built, so a mutant runs through the same plan
    object, the same bindings and the same launch path as the shipped kernel.
    Dropping it would silently disarm the mutation leg.
    """
    fields, pml = driver.fields, driver.pml
    residency = device.Residency()
    plans: Dict[str, Any] = {}
    for sub_step in ("step_B", "step_D"):
        plan = cx.plan_complex_pml_curl(fields, pml, sub_step, residency,
                                        (shaders.CONTRACT_OFF,), probe=PROBE)
        assert plan is not None, cx.complex_pml_curl_coverage(
            fields, pml, sub_step, residency, PROBE).reasons
        plans[sub_step] = plan
    for slot, side in (("update_H", "H"), ("update_E", "E")):
        plan = cx.plan_complex_constitutive(fields, pml, side, residency,
                                            (shaders.CONTRACT_OFF,), probe=PROBE)
        assert plan is not None, cx.complex_constitutive_coverage(
            fields, pml, side, residency, PROBE).reasons
        plans[slot] = plan
    for slot, replacement in (functions or {}).items():
        plan = plans.get(slot)
        if plan is not None:
            plan._functions = dict(replacement)  # noqa: SLF001 - the mutation seam
    return plans, residency


def install(driver: FdtdDriver, shim: Optional[MetalShim]) -> None:
    """Freeze this driver's fast path as ``shim`` (or as the pure array path).

    ``_fast_path_stale = False`` is what stops ``step()`` from calling the shipped
    planner on the next step and overwriting the shim. Setting it on a driver whose
    fast path is ``None`` is the ORACLE's configuration and is equally deliberate:
    it pins the oracle to NumPy no matter what the ambient dispatch switch says.
    """
    driver._fast_path = shim  # noqa: SLF001 - the engine's own freeze slot
    driver._fast_path_stale = False  # noqa: SLF001


# ---------------------------------------------------------------------------
# LEG wiring — the oracle is the array path, and this family is genuinely unwired
# ---------------------------------------------------------------------------

def leg_wiring(payload: Dict[str, Any], out: str) -> None:
    """Measured, in BOTH dispatch states, before a single byte is compared.

    The oracle driver's credibility rests on it taking the array path, and "the fast
    path is off by default on this tree" is a fact about an environment variable,
    not about this family. So the shipped planner is asked directly, with dispatch
    FORCED ON, whether it carries any slot for a complex/Bloch configuration. If it
    ever answers yes, this probe's oracle would be comparing a kernel against a
    kernel and every row below would be meaningless.

    THE REFUSAL REASON IS RECORDED, NOT INFERRED. On THIS host the shipped planner
    refuses first because the backend is not CuPy — the Triton dispatch route has no
    MPS arm — which is a fact about the HOST and not about this family. Reading
    "plan=None" as "the family is unwired" would be reading a strictly weaker
    measurement than the artifact claims, so the reason text is recorded beside the
    null and the two are not conflated.
    """
    rows: List[Dict[str, Any]] = []
    previous = os.environ.get(fastpath.DISPATCH_ENABLE)
    try:
        for case in CASES:
            driver = build_driver(case, sourced=True)
            record: Dict[str, Any] = {"case": case["name"],
                                      "courant": case["courant"]}
            for label, value in (("ambient", previous), ("forced_on", "1")):
                if value is None:
                    os.environ.pop(fastpath.DISPATCH_ENABLE, None)
                else:
                    os.environ[fastpath.DISPATCH_ENABLE] = str(value)
                plan = fastpath.plan_fast_path(driver.fields, driver.pml,
                                               driver.grid)
                report = fastpath.last_dispatch_report() or {}
                record[label] = {
                    "dispatch_enabled": fastpath.dispatch_enabled(),
                    "plan": None if plan is None else sorted(plan.slots),
                    "decision": report.get("decision"),
                    "refused_because": report.get("refused_because"),
                }
            rows.append(record)
            payload["legs"]["wiring"] = rows
            save(payload, out)
            log(f"[wiring] {case['name']:<22} ambient={record['ambient']['plan']} "
                f"forced_on={record['forced_on']['plan']} "
                f"because={record['forced_on']['refused_because']!r}")
            assert not record["forced_on"]["plan"], (
                f"{case['name']}: the SHIPPED planner claims "
                f"{record['forced_on']['plan']} on a complex/Bloch configuration. "
                f"This family is unwired, so the oracle driver below would be "
                f"running kernels and comparing them against themselves")
            assert record["forced_on"]["dispatch_enabled"], (
                "the forced-on arm did not force dispatch on, so it measured the "
                "same thing the ambient arm did")
            driver.close()
    finally:
        if previous is None:
            os.environ.pop(fastpath.DISPATCH_ENABLE, None)
        else:
            os.environ[fastpath.DISPATCH_ENABLE] = previous


# ---------------------------------------------------------------------------
# LEG driver_steps — the measurement
# ---------------------------------------------------------------------------

#: The two initial states, and why both. `sourced_from_zero` is the PHYSICAL case:
#: the driver's own zeros, moved only by the sources, which is what a run looks
#: like. `seeded_and_sourced` starts from a physical-band random state with a +-0
#: lattice, because a zero initial state makes step 1's constitutive sub-step a
#: fixed point and hides a defect that only shows on arbitrary data.
VARIANTS = (("sourced_from_zero", None), ("seeded_and_sourced", SEED))


def assert_dispatch_accounting(row: Dict[str, Any], shim: MetalShim, budget: int,
                               context: str) -> None:
    """Every device slot was taken by the DEVICE on every step of the budget.

    THIS IS THE GUARD THE WHOLE PROBE RESTS ON, and it is factored out so leg
    ``dispatch_accounting`` can prove it FIRES by running the same code against a
    shim that declines a slot. The oracle IS the array path, so a slot that quietly
    fell back would still compare byte-IDENTICAL: without this check a green row and
    a probe that measured nothing are the same artifact.
    """
    expected = {slot: budget for slot in shim.plans}
    assert row["dispatched"] == expected, (
        f"{context}: the device took {row['dispatched']} of an expected "
        f"{expected}. A slot that silently fell back to the array path would "
        f"still compare IDENTICAL, because the oracle IS the array path")
    assert row["plan_launches"] == budget * len(shim.plans), (
        f"{context}: the plans counted {row['plan_launches']} launches against "
        f"{budget * len(shim.plans)} dispatches")
    assert not row["array_path_slots"], (
        f"{context}: {row['array_path_slots']} were left to the array path, so "
        f"this row certifies less than the four sub-steps it claims")


def leg_driver_steps(payload: Dict[str, Any], out: str) -> None:
    """``FdtdDriver.step()`` for the budget, device against array path, PER STEP.

    Compared after EVERY step over the full stored inventory as uint32 words. The
    first divergence is reported with its case, step and array and the leg STOPS
    there — a family that diverges at step 7 and is compared for 17 more steps
    produces a wall of consequences and one cause.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    for case in CASES:
        for variant, seed in VARIANTS:
            device_driver = build_driver(case, sourced=True, seed=seed)
            oracle_driver = build_driver(case, sourced=True, seed=seed)
            names = inventory(device_driver.fields, device_driver.grid.shape)
            missing = [name for name in gate.STATE if name not in names]
            assert not missing, (
                f"the enumerated inventory is missing {missing}, which the gate "
                f"steps: the enumeration found fewer volumes than the transcribed "
                f"list and would compare less than the gate does")
            plans, residency = build_plans(device_driver)
            shim = MetalShim(device_driver.fields, plans, residency)
            install(device_driver, shim)
            install(oracle_driver, None)

            initial = snapshot(oracle_driver.fields, names)
            row: Dict[str, Any] = {
                "case": case["name"], "courant": case["courant"],
                "k_point": list(case["k"]), "variant": variant,
                "budget": STEP_BUDGET, "device_slots": sorted(shim.plans),
                "array_path_slots": sorted(
                    {"step_B", "update_H", "step_D", "update_E"} - set(shim.plans)),
                "inventory": len(names), "mirrors": len(residency.names),
                "steps": [],
            }
            first_divergence = None
            moved_total = 0
            for step in range(1, STEP_BUDGET + 1):
                device_driver.step()
                oracle_driver.step()
                got = snapshot(device_driver.fields, names)
                oracle = snapshot(oracle_driver.fields, names)
                bad = compare(got, oracle)
                moved_total = sum(differing(oracle[n], initial[n]) for n in names)
                if bad and first_divergence is None:
                    first_divergence = {"step": step,
                                        "arrays": dict(sorted(bad.items()))}
                if step in (1, STEP_BUDGET) or bad:
                    row["steps"].append({"step": step,
                                         "moved_from_initial": moved_total,
                                         "differing": dict(sorted(bad.items()))})
                if bad:
                    break
            row["first_divergence"] = first_divergence
            row["moved_from_initial"] = moved_total
            row["steps_taken"] = device_driver.step_count
            row["dispatched"] = dict(sorted(shim.dispatched.items()))
            row["declined"] = dict(sorted(shim.declined.items()))
            row["plan_launches"] = shim.launches
            row["syncs"] = {"in": shim.syncs_in, "out": shim.syncs_out}
            rows.append(row)
            payload["legs"]["driver_steps"] = rows
            save(payload, out)
            log(f"[driver] {case['name']:<22} {variant:<18} "
                f"steps={row['steps_taken']}/{STEP_BUDGET} "
                f"device={len(row['device_slots'])} moved={moved_total} "
                f"launches={row['plan_launches']} "
                f"{'IDENTICAL' if not first_divergence else first_divergence} "
                f"({time.time() - started:.1f}s)")

            context = f"{case['name']}/{variant}"
            assert first_divergence is None, (
                f"{context}: BYTE DIVERGENCE at step {first_divergence['step']} in "
                f"{first_divergence['arrays']}")
            assert row["steps_taken"] == STEP_BUDGET, (
                f"{context}: took {row['steps_taken']} steps against a stated "
                f"budget of {STEP_BUDGET}")
            kit.assert_moved(moved_total, f"{context} over {STEP_BUDGET} steps")
            # THE DISPATCH ACCOUNTING, which is what makes the comparison mean
            # anything: every device slot must have been taken by the device on
            # every step of the budget, and the plans' own launch counters must
            # agree with the shim's.
            expected = {slot: STEP_BUDGET for slot in shim.plans}
            assert row["dispatched"] == expected, (
                f"{context}: the device took {row['dispatched']} of an expected "
                f"{expected}. A slot that silently fell back to the array path "
                f"would still compare IDENTICAL, because the oracle IS the array "
                f"path")
            assert row["plan_launches"] == STEP_BUDGET * len(shim.plans), (
                f"{context}: the plans counted {row['plan_launches']} launches "
                f"against {STEP_BUDGET * len(shim.plans)} dispatches")
            assert not row["array_path_slots"], (
                f"{context}: {row['array_path_slots']} were left to the array "
                f"path, so this row certifies less than the four sub-steps it "
                f"claims")
            assert residency.names, (
                "VACUOUS: an empty mirror registry syncs nothing and proves nothing")
            device_driver.close()
            oracle_driver.close()


# ---------------------------------------------------------------------------
# LEG source_control — the sources are non-vacuous, against a source-free control
# ---------------------------------------------------------------------------

def leg_source_control(payload: Dict[str, Any], out: str) -> None:
    """The sourced run is compared to a SOURCE-FREE one, on both paths.

    THE CONTROL IS THE POINT. From the driver's own zeros a source-free run is
    identically zero forever — the curl of zero is zero and the constitutive read is
    a fixed point — so if the sourced run moved, the SOURCES moved it, and the
    interleave the device sub-steps ran between is real work rather than a fill of
    zeros. The control is ALSO stepped on the device path, so its null is a measured
    null and not an assumption: a kernel that wrote garbage into a quiescent grid
    would break it.

    Recorded as a PREDICTED NULL with its reason, because that is what it is: the
    control's own device-vs-array comparison is trivially identical and certifies
    nothing on its own.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    for case in CASES:
        driver = build_driver(case, sourced=True, seed=None)
        control = build_driver(case, sourced=False, seed=None)
        control_array = build_driver(case, sourced=False, seed=None)
        names = inventory(driver.fields, driver.grid.shape)
        for target in (driver, control):
            plans, residency = build_plans(target)
            install(target, MetalShim(target.fields, plans, residency))
        install(control_array, None)

        for _ in range(STEP_BUDGET):
            driver.step()
            control.step()
            control_array.step()
        sourced = snapshot(driver.fields, names)
        quiet = snapshot(control.fields, names)
        quiet_array = snapshot(control_array.fields, names)

        injected = sum(differing(sourced[n], quiet[n]) for n in names)
        # THE MATERIAL VOLUMES ARE NONZERO BY CONSTRUCTION and are excluded from the
        # quiescence count BY NAME rather than by a threshold: eps and inv_eps are
        # what `set_isotropic_epsilon_volume` wrote, nothing steps them, and
        # counting them would fail this leg for a reason with nothing to do with the
        # sources. They stay in the BYTE comparison, where a kernel that scribbled
        # past its target would land.
        stepped = tuple(name for name in names if name in gate.STATE)
        constants = tuple(name for name in names if name not in gate.STATE)
        control_nonzero = sum(int(np.count_nonzero(quiet[n])) for n in stepped)
        constant_nonzero = {n: int(np.count_nonzero(quiet[n])) for n in constants}
        control_vs_array = sum(differing(quiet[n], quiet_array[n]) for n in names)
        row = kit.predicted_null(
            {"case": case["name"], "courant": case["courant"],
             "budget": STEP_BUDGET,
             "sources": [source["component"] for source in SOURCES],
             "words_the_sources_moved": injected,
             "stepped_volumes": len(stepped),
             "nonzero_words_in_the_source_free_control": control_nonzero,
             "excluded_material_volumes": constant_nonzero,
             "control_device_vs_array_words": control_vs_array},
            "the source-free control starts at zero and stays there, so its own "
            "device-vs-array comparison is a no-op agreeing with a no-op; it is "
            "here to attribute the sourced run's movement to the sources, not to "
            "certify bytes")
        rows.append(row)
        payload["legs"]["source_control"] = rows
        save(payload, out)
        log(f"[source] {case['name']:<22} injected={injected} "
            f"control_nonzero={control_nonzero} "
            f"control_device_vs_array={control_vs_array} "
            f"({time.time() - started:.1f}s)")
        kit.assert_moved(injected, f"{case['name']} sources vs the quiet control",
                         floor=1000)
        assert control_nonzero == 0, (
            f"{case['name']}: the SOURCE-FREE control holds {control_nonzero} "
            f"nonzero words across its STEPPED volumes. Either the sources leaked "
            f"into it or the device path wrote into a quiescent grid; both make "
            f"the sourced run's movement unattributable")
        assert set(constants) == {"eps", "inv_eps"}, (
            f"{case['name']}: the quiescence count excluded {sorted(constants)}, "
            f"which is not the material pair it was written to exclude. A volume "
            f"that started being stepped would be exempted from the control here "
            f"without anyone noticing")
        assert all(constant_nonzero.values()), (
            f"{case['name']}: an excluded material volume is all zeros "
            f"({constant_nonzero}), so the epsilon this run stepped through is not "
            f"the one the case built")
        assert control_vs_array == 0, (
            f"{case['name']}: the quiet control's device and array paths differ in "
            f"{control_vs_array} words while holding nothing but zeros")
        for target in (driver, control, control_array):
            target.close()


# ---------------------------------------------------------------------------
# LEG precondition — the census, over the whole budget, plus its firing control
# ---------------------------------------------------------------------------

#: Two cases: a phased periodic one (both planes of the rotation live) and a
#: metallic one (the wall passes live). A claim naming both ghost rules that
#: censused only one would have checked half its scope.
CENSUS_CASES = ("periodic_kx", "metallic_xy_kz")

#: (label, scale, expect_clean). The band control is what makes the precondition a
#: measurement rather than a decoration: on this executor the flush is native and
#: has no lever, so the firing case is a genuine refusal.
CENSUS_SCALES = (("physical", 1.0, True), ("subnormal_band", 1e-38, False))


class SeamCensus:
    """Reconstructs the kernel's operands and intermediates AT THE DISPATCH SEAM.

    THE OPERANDS ARE EXACT. The shim calls this with the live ``Fields`` object
    immediately before the plan launches, so the arrays read here are the arrays the
    kernel is about to read — the source deposits and the wall clears already
    applied. That is the whole reason the census lives on the shim rather than on a
    post-step snapshot, which would reconstruct from operands the seam had moved.

    THE INTERMEDIATES ARE RECONSTRUCTED ON THE HOST, which is WEAKER THAN A REGISTER
    READ and is stated as such. ``compile_shader`` exposes no way to read a kernel's
    registers, so the same expressions are formed in the same operand order on the
    host and censused: the curl's ``dtdx * ((sf - f) + (s - ss))`` and the
    constitutive's ``kps * fw`` / ``kms * fw_previous``. Neither is ever stored
    anywhere, so censusing only the stored volumes would leave the claim's own
    arithmetic unchecked.
    """

    __slots__ = ("window", "grid", "pml", "codes", "phases", "inverse", "step",
                 "observations")

    def __init__(self, window: Any, driver: FdtdDriver) -> None:
        self.window = window
        self.grid = driver.grid
        self.pml = driver.pml
        self.codes = gate.boundary_codes(driver.grid, driver.pml)
        self.phases = gate.phase_table(driver.grid, driver.pml)
        self.inverse = {n: driver.fields.inverse_epsilon_for(n)
                        for n in gate.SIDES["E"]["targets"]}
        self.step = 0
        self.observations = 0

    def __call__(self, slot: str, fields: Any) -> None:
        step = self.step
        if step < self.window.first_step or step > self.window.last_step:
            return
        dtdx = float(self.grid.dt / self.grid.dx)
        if slot in gate.SUB_STEPS:
            spec = gate.SUB_STEPS[slot]
            operands = tuple(spec["sources"]) + tuple(spec["targets"]) + tuple(
                "fu_" + n for n in spec["targets"])
            curls = gate.reference_curls(
                [getattr(fields, n) for n in spec["sources"]], self.codes,
                bool(spec["backward"]), dtdx, self.phases)
            intermediates = {f"{slot}:curl:{target}": volume
                             for target, volume in zip(spec["targets"], curls)}
        else:
            side = "H" if slot == "update_H" else "E"
            spec = gate.SIDES[side]
            operands = (tuple(spec["sources"]) + tuple(spec["targets"])
                        + tuple(spec["aux"]))
            coefficients = gate.constitutive_coefficients(self.pml, spec["suffix"])
            intermediates = {}
            for index, axis in enumerate("xyz"):
                target = spec["targets"][index]
                source = getattr(fields, spec["sources"][index])
                product = (source if side == "H"
                           else source * self.inverse[target])
                intermediates[f"{slot}:kps*fw:{target}"] = (
                    coefficients["kps_" + axis] * product)
                intermediates[f"{slot}:kms*fwprev:{target}"] = (
                    coefficients["kms_" + axis] * getattr(fields,
                                                          spec["aux"][index]))
        for name in operands:
            self.window.observe(f"step{step}:{slot}:in:{name}",
                                getattr(fields, name), step=step)
            self.observations += 1
        for tag, volume in intermediates.items():
            self.window.observe_intermediate(f"step{step}:{tag}", volume, step=step)
            self.observations += 1


def leg_precondition(payload: Dict[str, Any], out: str) -> None:
    """The claim is CONDITIONAL and the condition is checked, on EVERY step.

    Two halves, and they are different assertions. The physical run must be CLEAN
    across the whole budget or the gate refuses it. And a scaled control must
    DEMONSTRATE the window firing — a precondition never shown to fire is
    decorative, and on this executor the flush is native and has no lever, so the
    firing case is a genuine refusal rather than a warning.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    for name in CENSUS_CASES:
        case = next(entry for entry in CASES if entry["name"] == name)
        for label, scale, expect_clean in CENSUS_SCALES:
            driver = build_driver(case, sourced=True, seed=SEED, scale=scale)
            oracle_driver = build_driver(case, sourced=True, seed=SEED, scale=scale)
            install(oracle_driver, None)
            names = inventory(driver.fields, driver.grid.shape)
            window = preconditions.SubnormalWindow(
                1, STEP_BUDGET, per_array_words=1, per_intermediate_words=1)
            census = SeamCensus(window, driver)
            plans, residency = build_plans(driver)
            shim = MetalShim(driver.fields, plans, residency, census=census)
            install(driver, shim)

            differing_words = 0
            for step in range(1, STEP_BUDGET + 1):
                census.step = step if step % CENSUS_EVERY == 0 else -1
                driver.step()
                oracle_driver.step()
                if step % CENSUS_EVERY == 0:
                    for volume in names:
                        window.observe(f"step{step}:out:{volume}",
                                       getattr(driver.fields, volume), step=step)
                differing_words = sum(
                    differing(getattr(driver.fields, n),
                              getattr(oracle_driver.fields, n)) for n in names)
                if differing_words:
                    break
            report = window.report()
            row = {"case": case["name"], "scale": label, "factor": scale,
                   "budget": STEP_BUDGET, "census_every": CENSUS_EVERY,
                   "seam_observations": census.observations,
                   "subnormal_words": report["subnormal_words"],
                   "observed_words": report["observed_words"],
                   "intermediate_subnormal_words": sum(
                       v["subnormal_words"]
                       for v in report["intermediates"].values()),
                   "intermediate_words": sum(
                       v["words"] for v in report["intermediates"].values()),
                   "intermediates_censused": len(report["intermediates"]),
                   "clean": report["clean"], "vacuous": report["vacuous"],
                   "differing_words": differing_words,
                   "gate_verdict": ("REFUSED (precondition)" if not report["clean"]
                                    else ("identical" if not differing_words
                                          else "DIFFERS"))}
            rows.append(row)
            payload["legs"]["precondition"] = rows
            save(payload, out)
            log(f"[precondition] {case['name']:<22} {label:<14} "
                f"subnormal={row['subnormal_words']}/{row['observed_words']} "
                f"(intermediates {row['intermediate_subnormal_words']}/"
                f"{row['intermediate_words']}) differing={differing_words} "
                f"verdict={row['gate_verdict']} ({time.time() - started:.1f}s)")
            context = f"{case['name']}/{label}/driver_step"
            # The seam census must have RUN. A window that observed only the stored
            # output volumes would report `clean` while the kernel's own operands
            # and intermediates went unlooked-at, which is the exact hole the
            # intermediate hook exists to close.
            kit.assert_census_floor(census.observations, f"{context} seam census",
                                    floor=4 * len(gate.SUB_STEPS))
            if expect_clean:
                preconditions.assert_clean_or_refuse(window, context)
                assert differing_words == 0, row
            else:
                preconditions.demonstrate_firing(window, context)
                assert differing_words > 0, (
                    f"{context}: the scaled control was expected to DIVERGE inside "
                    f"a driver step and did not, so the cliff this claim rests on "
                    f"was not reproduced at driver level")
            driver.close()
            oracle_driver.close()


# ---------------------------------------------------------------------------
# LEG mutations — armed INSIDE a complete driver step
# ---------------------------------------------------------------------------

#: Driver-level mutations. The gate arms the arithmetic exhaustively at sub-step
#: level (23 rows); these exist to answer a question the gate cannot — is the number
#: this probe compared produced by the METAL KERNEL, inside a real step, with the
#: array passes interleaved? A mutant that runs and is NOT caught here would mean
#: the device path is not reaching the compared bytes.
#:
#: ``curl_sub_steps`` / ``sides`` scope each entry to where the needle GENUINELY
#: EXISTS in the shipped source. Measured on the mutation case's codes (metallic x,
#: metallic y, periodic z with a z phase): the wrapped-lane select and the x
#: ownership mask are emitted by ``step_B`` and NOT by ``step_D``, whose backward
#: stencil wraps the other face. Scoping is not tidiness — the kit counts ANY
#: partial needle miss as NEEDLE-MISSED and fails the row, precisely so a mutation
#: cannot be half-planted and still report a pass.
DRIVER_MUTATIONS: Dict[str, Dict[str, Any]] = {
    "d1_flatten_curl_parens": {
        "must_catch": True,
        "why": "the ONE grouping the whole claim rests on, re-armed inside a "
               "complete driver step: this is the row that proves the compared "
               "inventory carries the Metal kernel's output rather than the array "
               "path's",
        "curl_sub_steps": ("step_B", "step_D"),
        "apply": lambda s: kit.needle(s, "float2 t0 = ((c_y - c) + (b - b_z));",
                                      "float2 t0 = (c_y - c + b - b_z);"),
    },
    "d2_phase_on_every_lane": {
        "must_catch": True,
        "why": "the wrapped-lane SELECT dropped on the phased z axis, so every lane "
               "is rotated. Caught at sub-step level; re-armed here because the "
               "per-dispatch sync could in principle overwrite the device's answer "
               "with the host's before the comparison sees it",
        "curl_sub_steps": ("step_B",),
        "apply": lambda s: kit.needle(s, "bool wz = (k == nzi - 1);",
                                      "bool wz = true;"),
    },
    "d3_drop_ownership_mask_x": {
        "must_catch": True,
        "why": "the metallic x axis stops dropping its unowned cell — a plane of "
               "values MEEP's loop never touches. It is armed HERE and not only in "
               "the gate because the driver's own `zero_metal_B` pass runs over the "
               "same wall between the sub-steps, and a probe that never ran the two "
               "together could not see one masking the other",
        "curl_sub_steps": ("step_B",),
        "apply": lambda s: kit.needle(
            s, "curl0 = at_x ? float2(0.0f, 0.0f) : curl0;",
            "// ownership mask dropped on x"),
    },
    "d4_flatten_constitutive_accumulation": {
        "must_catch": True,
        "why": "`((f + kps*src) - kms*prev)` flattened, on BOTH sides. The "
               "constitutive pair is half this family's claim and the gate's own "
               "audit found it certified with no census; a driver-level arm is what "
               "shows the device reaches those bytes too",
        "sides": ("H", "E"),
        "apply": lambda s: kit.needle(
            s,
            "    a0 = a0 + c_mul_coefficient_left(kp_0, src0);\n"
            "    a0 = a0 - c_mul_coefficient_left(km_0, prev0);",
            "    a0 = a0 + (c_mul_coefficient_left(kp_0, src0) "
            "- c_mul_coefficient_left(km_0, prev0));"),
    },
    "d5_comment_only_is_not_a_defect": {
        # THE MUST-NOT-CATCH ROW. Without it, "every mutation was caught" is equally
        # consistent with a comparison that reports divergence for any recompile at
        # all — a leg detecting recompilation rather than defects.
        "must_catch": False,
        "why": "a comment appended to the curl body — recompiled, relaunched, "
               "arithmetically identical. It must NOT be caught, or this leg is "
               "measuring recompilation and not arithmetic",
        "curl_sub_steps": ("step_B", "step_D"),
        "apply": lambda s: kit.needle(
            s, "float2 t0 = ((c_y - c) + (b - b_z));",
            "float2 t0 = ((c_y - c) + (b - b_z));  // same arithmetic"),
    },
}

#: The mutation case: metallic x and y with a phased periodic z. Chosen because it
#: is the ONE configuration in the matrix where the wall passes, the ownership mask
#: and a live Bloch rotation are all in the same driver step.
MUTATION_CASE = "metallic_xy_kz"


def leg_mutations(payload: Dict[str, Any], out: str) -> None:
    """Armed, launch-counted, run through ``driver.step()`` for a short budget."""
    harness = kit.MutationHarness(payload, out)
    case = next(entry for entry in CASES if entry["name"] == MUTATION_CASE)
    for label, entry in DRIVER_MUTATIONS.items():
        counters: List[kit.Counter] = []
        functions: Dict[str, Dict[str, Any]] = {}
        missed = False
        driver = build_driver(case, sourced=True, seed=SEED)
        oracle_driver = build_driver(case, sourced=True, seed=SEED)
        install(oracle_driver, None)
        codes = list(gate.boundary_codes(driver.grid, driver.pml))
        phases = gate.phase_table(driver.grid, driver.pml)

        for sub_step in entry.get("curl_sub_steps", ()):
            spec = gate.SUB_STEPS[sub_step]
            flags, _ = cx.phase_arguments(tuple(phases),
                                          backward=bool(spec["backward"]))
            shipped = cx.bloch_curl_source(codes, spec["backward"], flags, EXPANSION)
            try:
                mutated = entry["apply"](shipped)
            except LookupError:
                missed = True
                continue
            counter = kit.Counter(
                device.compile_source(mutated).bloch_pml_curl_step)
            counters.append(counter)
            functions[sub_step] = {shaders.CONTRACT_OFF: counter}
        for side in entry.get("sides", ()):
            shipped = cx.bloch_constitutive_source(side, EXPANSION)
            try:
                mutated = entry["apply"](shipped)
            except LookupError:
                missed = True
                continue
            counter = kit.Counter(
                device.compile_source(mutated).bloch_constitutive_step)
            counters.append(counter)
            functions["update_" + side] = {shaders.CONTRACT_OFF: counter}

        plans, residency = build_plans(driver, functions=functions)
        shim = MetalShim(driver.fields, plans, residency)
        install(driver, shim)
        names = inventory(driver.fields, driver.grid.shape)
        for _ in range(MUTATION_BUDGET):
            driver.step()
            oracle_driver.step()
        caught = int(any(differing(getattr(driver.fields, n),
                                   getattr(oracle_driver.fields, n))
                         for n in names))
        launches = sum(counter.launches for counter in counters)
        # ONE DRIVER RUN IS ONE CASE, and the count says so. The gate's rows are per
        # sub-step because it launches them in isolation; here every mutated slot
        # runs inside the SAME driver run and the comparison is one verdict over one
        # final inventory, so `ran` is 1 and the number of slots carrying the needle
        # is recorded beside it. Counting `ran` as the slot count instead would
        # report `CAUGHT 1/2` for a mutation that WAS caught — a false failure — and
        # its mirror image would be a false pass.
        ran = 1 if functions else 0
        harness.record(label, harness.verdict(missed, ran, launches, caught),
                       launches, caught, ran, entry["must_catch"], entry["why"],
                       extra={"budget": MUTATION_BUDGET, "case": case["name"],
                              "slots_carrying_the_needle": sorted(functions),
                              "dispatched": dict(sorted(shim.dispatched.items()))})
        assert launches == MUTATION_BUDGET * len(functions), (
            f"{label}: the mutated kernels launched {launches} times against "
            f"{MUTATION_BUDGET * len(functions)} expected dispatches, so the row's "
            f"verdict is about a different number of launches than it claims")
        driver.close()
        oracle_driver.close()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

LEGS = (
    ("wiring", leg_wiring),
    ("driver_steps", leg_driver_steps),
    ("source_control", leg_source_control),
    ("precondition", leg_precondition),
    ("mutations", leg_mutations),
)

REQUIRED = tuple(name for name, _ in LEGS)


def main() -> int:
    parser = kit.argument_parser(__doc__)
    parser.add_argument("--probe", default="",
                        help="expansion probe artifact (defaults to the environment)")
    arguments = parser.parse_args()
    started = time.time()
    out_dir = os.path.dirname(os.path.abspath(arguments.out)) or "."
    os.makedirs(out_dir, exist_ok=True)

    payload: Dict[str, Any] = {
        "environment": kit.environment_stamp(),
        "subnormal_policy": subnormal.mps_policy_report(),
        "legs": {},
        "budget": {"driver_steps": STEP_BUDGET, "source_control": STEP_BUDGET,
                   "precondition": STEP_BUDGET, "mutations": MUTATION_BUDGET,
                   "census_every": CENSUS_EVERY},
        "question": ("does the certified complex/Bloch curl and constitutive pair "
                     "still reproduce the array path INSIDE a complete "
                     "FdtdDriver.step(), compared after EVERY step for a stated "
                     "budget, with both source seams injecting and the wall passes "
                     "live — and did the device path run at all"),
    }
    save(payload, arguments.out)

    reasons: List[str] = []
    if not payload["environment"].get("mps_available"):
        reasons.append("no MPS device on this host: every leg here launches a Metal "
                       "kernel, and a run that skipped them all must not mint a "
                       "green artifact")
    if not payload["subnormal_policy"]["admitted"]:
        reasons.extend(payload["subnormal_policy"]["reasons"])
    probe = cx.load_expansion_probe(arguments.probe or None)
    if probe is None or cx.expansion_from_probe(probe) is None:
        reasons.append(
            f"no usable complex-expansion probe artifact (pass --probe or set "
            f"{cx.PROBE_PATH_ENVIRONMENT}); the arm is a MEASURED platform fact "
            f"and this probe will not guess it. Run gate_metal_complex.py --legs "
            f"expansion to write one")
    if reasons:
        return kit.cannot_certify(payload, arguments.out, reasons)

    global EXPANSION, PROBE
    PROBE = probe
    EXPANSION = cx.expansion_from_probe(probe)
    # The gate's helpers this probe reuses read the module-level arm; binding it
    # here keeps one arm across both files rather than two that could drift.
    gate.EXPANSION = EXPANSION
    payload["expansion_arm"] = EXPANSION

    kernel_dir = os.path.join(API_ROOT, "meep_gpu", "metal_kernels")
    payload["provenance"] = kit.provenance(out_dir, {
        "stepping.py": os.path.join(API_ROOT, "meep_gpu", "stepping.py"),
        "driver.py": os.path.join(API_ROOT, "meep_gpu", "driver.py"),
        "fastpath.py": os.path.join(API_ROOT, "meep_gpu", "fastpath.py"),
        "metal_kernels/complex_fields.py": os.path.join(kernel_dir,
                                                        "complex_fields.py"),
        "metal_kernels/plans.py": os.path.join(kernel_dir, "plans.py"),
        "metal_kernels/device.py": os.path.join(kernel_dir, "device.py"),
        "metal_kernels/shaders.py": os.path.join(kernel_dir, "shaders.py"),
        "metal_kernels/preconditions.py": os.path.join(kernel_dir,
                                                       "preconditions.py"),
        "gate_metal_complex.py": os.path.join(HERE, "gate_metal_complex.py"),
        "metal_gate_kit.py": os.path.join(HERE, "metal_gate_kit.py"),
        "probe_metal_complex_driver_step.py": os.path.abspath(__file__),
        # ITS OWN FILE NAME. This probe shares a results directory with the gate and
        # the composition probe, and `kit.provenance`'s default name is what one
        # caller used to overwrite another's record wholesale — the evidence-
        # destroying collision the kit's `name` parameter exists for.
    }, kernel_sources=cx.enumerate_complex_sources(EXPANSION),
        name="provenance_driver_step.json")
    save(payload, arguments.out)

    log(f"[env] arm={EXPANSION} torch={payload['environment'].get('torch')} "
        f"numpy={payload['environment']['numpy']} "
        f"frontend={payload['environment'].get('metal_frontend')} "
        f"policy={payload['subnormal_policy']['resolved']} "
        f"budget={STEP_BUDGET} cases={len(CASES)}")

    wanted = kit.wanted_legs(arguments.legs)
    ran = kit.run_legs(LEGS, payload, arguments.out, wanted)
    payload["counters"] = dict(TALLY)
    missing = [name for name in REQUIRED if name not in ran]
    if missing:
        return kit.cannot_certify(payload, arguments.out, [
            f"required legs did not run: {missing}. A --legs subset is for "
            f"debugging and must not be able to mint a certified artifact"])
    return kit.summarize(
        payload, arguments.out,
        claim=("byte-identity to the NumPy array path across a COMPLETE "
               f"FdtdDriver.step(), compared after EVERY step for a stated budget "
               f"of {STEP_BUDGET} steps, under a CHECKED subnormal-free "
               "precondition censused at the dispatch seam on every step"),
        scope=("the complex/Bloch split-field PML curl (step_B/step_D) AND the "
               "dsigw constitutive pair (update_H/update_E) inside the driver's own "
               "step order, complex64 storage, periodic and metallic ghost rules, "
               "per-axis Bloch phase including the Brillouin edge and the "
               "three-axis case, courants 0.35 and 0.5, gaussian magnetic AND "
               "electric sources injecting at both seams; all four sub-steps are "
               "taken by the device on every step. The composition is the "
               "DECLARED-SYNC one — no mirror is held across an array pass — and "
               "the throughput question is not asked here. NOT WIRED: the shipped "
               "planner carries no slot for this family and leg `wiring` measures "
               "that in both dispatch states"),
        stated_weakness=(
            "three. (1) NO PTX-EQUIVALENT AUDIT exists on this executor: "
            "compile_shader exposes no disassembly, so the mutation leg and the "
            "launch counters are the only arbiters that the device did what the "
            "source says. (2) THE SUBNORMAL CENSUS IS RECONSTRUCTED, NOT READ: the "
            "operands are exact (read at the dispatch seam, after the source fills "
            "and wall clears, before the launch) but the kernel's unstored "
            "intermediates are re-formed on the HOST in the same operand order, "
            "because a kernel's registers are not readable from here. (3) THE "
            "DISPATCH IS A SHIM: this family is unwired, so the probe installs a "
            "FastPathPlan-shaped object at the driver's own seam rather than "
            "measuring the shipped planner's composition, and leg `wiring` records "
            "the shipped planner's refusal reason so a host-level refusal is not "
            "read as a family-level one"),
        started=started, legs_run=ran, compared=TALLY["compared"],
        certified=TALLY["disagreed"] == 0,
        extra={"expansion_arm": EXPANSION,
               "expansion_probe": os.path.abspath(arguments.probe)
               if arguments.probe else None,
               "arrays_that_disagreed": TALLY["disagreed"],
               "setup_pml_docstring_finding": (
                   "driver.py:2160-2168 documents a refusal of a nonzero k_point "
                   "on an absorbing axis that the code does not implement; "
                   "measured on this host, k_point=(0.3,0,0) with a scalar, "
                   "per-axis or z-only PML all build with pml.is_active=True. The "
                   "case matrix reuses the gate's configurations because of it")})


if __name__ == "__main__":
    raise SystemExit(main())
