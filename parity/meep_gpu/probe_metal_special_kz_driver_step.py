"""The Metal special_kz curls inside a REAL ``FdtdDriver.step()``, for a budget.

WHAT THIS MEASURES THAT THE GATE AND THE COMPOSITION PROBE DO NOT.

``gate_metal_special_kz`` certifies two sub-steps in ISOLATION, from a seeded
state, against ``stepping.py``. ``probe_metal_special_kz_composition`` runs the four
``STEP_ORDER`` sub-steps back to back and says so in its own row — it deliberately
does NOT run ``zero_metal_B``/``zero_metal_D`` or any source fill, because a
like-for-like four-sub-step comparison is what attributes a divergence to a kernel.

Neither of them has ever run ``driver.step()``. That leaves four things unmeasured,
and each is a place a certified kernel can still produce a wrong RUN:

1. **The complete driver step**, in the driver's own order (driver.py:3277-3306):
   ``step_B`` -> magnetic sources -> symmetry fill -> ``zero_metal_B`` -> folded
   ghosts -> ``update_H`` -> ``step_D`` -> electric sources -> symmetry fill ->
   ``zero_metal_D`` -> folded ghosts -> ``update_E`` -> ``update_P``. Seven of those
   passes are array-path work INTERLEAVED with the device sub-steps, and every one
   of them writes a volume a mirror shadows.
2. **A source actually injecting.** The composition probe's seam leg asks the
   COVERAGE question — with a declared source, may a mirror set be held? — and
   answers no. It never injects anything. A gate whose fields are seeded by a random
   number generator has never seen the seam it refuses.
3. **The budget.** One step cannot show a defect that accumulates. Twenty-four can,
   and the auxiliaries (``fu_*``) are STATE, so a kernel that is right on launch 1
   and wrong forever after only diverges in a multi-step leg.
4. **That the device path RAN AT ALL.** Every number below would be produced,
   unchanged and green, by a probe whose shim answered ``False`` to every slot and
   let the array path do all the work. The launch counters and the driver-level
   armed mutations are what separate "the Metal kernels reproduce the array path"
   from "the array path reproduces itself".

THE COMPOSITION IS THE DECLARED-SYNC ONE, and the probe holds no mirrors across the
step. The residency leg of the composition probe measured which slots MUST be synced
— the wall passes on a metallic axis, the constitutive pair under complex storage,
and the source fills for both families — so a probe that ran a real driver step while
holding mirrors would be measuring a composition its own family already refused.
Each dispatched slot therefore syncs in, launches, and syncs out. That costs the
round trip the residency layer exists to avoid; this probe measures CORRECTNESS, and
the throughput question belongs to a bench, not here.

THE ORACLE IS A SECOND DRIVER, in the same process, same seed, same configuration,
stepping entirely on NumPy. Comparison is uint32 word equality over the FULL STORED
INVENTORY — every ndarray the ``Fields`` object holds at grid shape, the material
volumes included, enumerated from the object rather than transcribed, so a volume
added to the engine joins the comparison instead of quietly escaping it.

STATED, because the artifact inherits it: there is NO PTX-EQUIVALENT AUDIT on this
executor (``compile_shader`` exposes no disassembly), so the mutation leg and the
launch counters are the only arbiters that the device did what the source says.

    python -u probe_metal_special_kz_driver_step.py \\
        --out results/metal_special_kz_2026-08-15_certify/driver_step.json
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
# arm64 host's DEFAULT resolves to keep. Request flush EXPLICITLY, before anything
# resolves a policy — the claim is only as good as the precondition it was measured
# under, and `setdefault` leaves an explicit caller override in place.
os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")

import gate_metal_special_kz as gate  # noqa: E402
import metal_gate_kit as kit  # noqa: E402
import probe_metal_beta_expansion as expansion_probe  # noqa: E402

from meep_gpu import fastpath  # noqa: E402
from meep_gpu.driver import FdtdDriver  # noqa: E402
from meep_gpu.metal_kernels import device, preconditions, special_kz  # noqa: E402
from meep_gpu.metal_kernels import subnormal, templates  # noqa: E402

log, save, differing = kit.log, kit.save, kit.differing

#: THE STEP BUDGET, STATED. Not a round number for its own sake: 24 steps is six
#: full periods of the dt=0.034 grid's source at f=1.0 (period 29.4 dt), enough for
#: the pulse to leave the launch cell, reach the PML and be absorbed, which is the
#: span over which a stale-auxiliary defect compounds. Every row records the budget
#: it ran, and the leg asserts the budget it planned is the budget it took.
STEP_BUDGET = 24

#: The census runs on EVERY step of the budget, not a sample: a window taken at
#: step 0 of a 24-step run certifies step 0 (preconditions.SubnormalWindow's own
#: docstring). Cheap here — 26 volumes on a 320-cell grid.
CENSUS_EVERY = 1

#: The seed for the physical-band initial state of the SEEDED variant. The sourced
#: variant starts from the driver's own zeros, which is the physical case.
SEED = 20260815

TALLY = {"compared": 0, "disagreed": 0}


# ---------------------------------------------------------------------------
# The stored inventory, enumerated from the object
# ---------------------------------------------------------------------------

def inventory(fields: Any, shape: Sequence[int]) -> Tuple[str, ...]:
    """Every ndarray the Fields object holds at grid shape, by name.

    ENUMERATED, NOT TRANSCRIBED. ``gate.STATE`` is a hand-written list of the 24
    volumes a beta step reads or writes; this reads the object and finds 26 — those
    24 plus ``eps``/``inv_eps``. The material volumes are compared too, and they are
    the reason to enumerate rather than transcribe: nothing in a curl should write
    epsilon, so a kernel that scribbles past its target lands there, and a
    transcribed list would be exactly the list that cannot see it.

    BOTH STORAGE SHAPES ARE READ, and the ``__slots__`` branch is not decoration:
    ``Fields`` carries an instance ``__dict__`` today and the slotted spelling is
    what a memory-conscious refactor would reach for. Reading only one and finding
    nothing is how a "full inventory" silently becomes an empty one — which is
    exactly what happened here first (measured: the slots-only spelling enumerated
    zero volumes and the leg's own floor caught it), so the floor below stays.
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


#: The type ``torch.mps.compile_shader`` hands back for a kernel entry point. A
#: PLAIN PYTHON CALLABLE bound in its place is the silent-fallback class in its
#: strongest form — the offdiag family's arming probe measured it: forward to numpy
#: (or even to the real kernel) and every byte the gate compares is correct, the
#: launch counter still increments, the mirrors are still on the device, and only
#: the entry point's TYPE distinguishes the two. Checked by name because there is
#: no PTX-equivalent audit on this executor to check anything deeper.
METAL_KERNEL_TYPE = "_mps_MetalKernel"


def assert_executed_on_metal(residency: Any, plans: Dict[str, Any],
                             context: str) -> Dict[str, Any]:
    """Positive proof the Metal path RAN: mirror device AND entry-point type.

    Two independent claims, and a launch count is neither of them. The mirrors are
    what the shader binds, so their device is the executor's own address; the entry
    points are what the plan calls, so their type is what says a GPU kernel and not
    a Python stand-in was invoked. A ``kit.Counter`` wrapper is unwrapped before the
    type is read — the mutation legs bind exactly that — so this check does not
    reject the harness's own instrumentation while still reading the real callee.
    """
    assert residency.names, (
        f"{context}: the residency registry is EMPTY, so nothing was mirrored to "
        f"any device and there is nothing to check")
    devices = sorted({str(residency.tensor(name).device.type)
                      for name in residency.names})
    assert devices == ["mps"], (
        f"{context}: mirrors live on {devices}, not on the Metal device. A byte "
        f"agreement reached on the CPU is the array path agreeing with itself")
    types: Dict[str, str] = {}
    for slot, plan in plans.items():
        for variant, function in plan._functions.items():  # noqa: SLF001
            while isinstance(function, kit.Counter):
                function = function.function
            types[f"{slot}:{variant}"] = type(function).__name__
    wrong = {key: name for key, name in types.items()
             if name != METAL_KERNEL_TYPE}
    assert not wrong, (
        f"{context}: {wrong} are not {METAL_KERNEL_TYPE} entry points. A Python "
        f"callable in that slot forwards wherever it likes and passes every byte "
        f"comparison, every launch count and every mirror check")
    return {"mirror_count": len(residency.names), "mirror_devices": devices,
            "entry_point_types": sorted(set(types.values()))}


def exact_words(array: Any) -> Any:
    """A uint32 word view that NEVER casts, for arrays outside the field storage.

    ``kit.words`` casts anything that is not complex64 to float32 before viewing —
    correct for the field volumes, which are float32 or complex64 by construction,
    and WRONG here: a DFT accumulator is complex64 but a flux spectrum is float64,
    and casting it to float32 would throw away 29 bits of every word and turn a
    comparison into a tolerance nobody declared. This views the array's own bytes.
    """
    contiguous = np.ascontiguousarray(array)
    raw = contiguous.reshape(-1).view(np.uint8)
    if raw.size % 4 == 0:
        return raw.view(np.uint32)
    return raw


def exact_differing(left: Any, right: Any) -> int:
    return int(np.count_nonzero(exact_words(left) != exact_words(right)))


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

#: A gaussian pulse on each source family. BOTH SEAMS ARE EXERCISED: the magnetic
#: current lands between `step_B` and `update_H` and the electric one between
#: `step_D` and `update_E` (driver.py:3282-3300, MEEP step.cpp:64-100), so a probe
#: carrying only one would leave half the interleave unmeasured.
SOURCES = (
    {"component": "Ez", "center": (0.2, 0.1, 0.0), "size": (0.0, 0.0, 0.0),
     "frequency": 1.0, "source_type": "gaussian", "fwidth": 0.5},
    {"component": "Hy", "center": (-0.3, -0.2, 0.0), "size": (0.0, 0.0, 0.0),
     "frequency": 1.0, "source_type": "gaussian", "fwidth": 0.5},
)


def build_driver(case: Dict[str, Any], sourced: bool = True,
                 seed: Optional[int] = None, scale: float = 1.0) -> FdtdDriver:
    """One driver for a beta case, built through the public engine API.

    ``sourced`` is the control axis: the source-free driver is what makes the
    sourced run's movement attributable to the source rather than to the initial
    state. ``seed`` seeds a physical-band initial state on top, so a run whose first
    step would otherwise act on all zeros still moves state at step 1.
    """
    driver = FdtdDriver(cell_size=(2.0, 1.6, 0.0), resolution=10.0,
                        courant=case["courant"],
                        force_complex_fields=bool(case["complex"]),
                        boundaries=case["boundaries"], dimensions=2,
                        k_point=case["k"], beta=case["beta"])
    driver.setup_pml({"x": 2, "y": 2})
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
            host = (rng.standard_normal(shape) * (0.37 * scale)).astype(np.float32)
            # The ±0 lattice rides the seeded variant: the beta term is a pure
            # scalar-by-field product, so a ±0 partner flows straight through it
            # into the recurrence, and the driver's own zeros carry only +0.
            host.reshape(-1)[::17] = np.float32(-0.0)
            if case["complex"]:
                imag = (rng.standard_normal(shape) * (0.29 * scale)).astype(np.float32)
                imag.reshape(-1)[3::19] = np.float32(-0.0)
                array[...] = gate._complex_of(host, imag)
            else:
                array[...] = host
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
    ``fast.dispatch(slot, fields)`` at seven identical sites (driver.py:3281-3305).
    This object answers that call. The shipped planner is untouched, this family
    stays unwired, and the probe is a MEASUREMENT of what wiring it would produce —
    which is the only honest way to run it before the certification exists.

    EVERY DISPATCH IS COUNTED, and the counts are asserted against the budget. Both
    halves of the answer are recorded: ``dispatched`` (True — the device ran the
    sub-step) and ``declined`` (False — the array path did). A shim that quietly
    declined everything would still produce a green byte comparison, because the
    oracle IS the array path, so the counters are the only thing standing between
    this probe and a pass that measures nothing.

    THE SYNC IS PER DISPATCH, not per step. See the module docstring: the array
    passes between the device sub-steps write the volumes the mirrors shadow, so the
    mirrors cannot be held, and the composition probe's residency leg is where that
    is established rather than assumed here.
    """

    __slots__ = ("fields", "plans", "residency", "dispatched", "declined",
                 "syncs_in", "syncs_out")

    def __init__(self, fields: Any, plans: Dict[str, Any], residency: Any) -> None:
        self.fields = fields
        self.plans = {name: plan for name, plan in plans.items() if plan is not None}
        self.residency = residency
        self.dispatched: Dict[str, int] = {}
        self.declined: Dict[str, int] = {}
        self.syncs_in = 0
        self.syncs_out = 0

    def dispatch(self, slot: str, fields: Any) -> bool:
        plan = self.plans.get(slot)
        if plan is None or fields is not self.fields:
            self.declined[slot] = self.declined.get(slot, 0) + 1
            return False
        self.residency.sync_in()
        self.syncs_in += 1
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


def build_plans(driver: FdtdDriver, probe: Any,
                functions: Optional[Dict[str, Dict[str, Any]]] = None
                ) -> Tuple[Dict[str, Any], Any]:
    """The plans a wired version of this family would build for one driver.

    ``functions`` is the MUTATION SEAM: a per-slot function map that replaces the
    shipped one after the plan is built, so a mutant runs through the same plan
    object, the same bindings and the same launch path as the shipped kernel.
    Dropping it would silently disarm the mutation leg.
    """
    fields, pml = driver.fields, driver.pml
    residency = device.Residency()
    plans: Dict[str, Any] = {}
    complex_storage = bool(np.iscomplexobj(fields.Ex))
    for sub_step in ("step_B", "step_D"):
        plans[sub_step] = (
            special_kz.plan_beta_bloch_pml_curl(fields, pml, sub_step, residency,
                                                probe=probe)
            if complex_storage else
            special_kz.plan_beta_pml_curl(fields, pml, sub_step, residency))
    if not complex_storage:
        for slot, side in (("update_H", "H"), ("update_E", "E")):
            plans[slot] = special_kz.plan_beta_run_constitutive(
                fields, pml, side, residency)
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

    The oracle driver's credibility rests on it taking the array path, and "the
    fast path is off by default on this tree" is a fact about an environment
    variable, not about this family. So the shipped planner is asked directly, with
    dispatch FORCED ON, whether it carries any slot for a beta configuration. If it
    ever answers yes, this probe's oracle would be comparing a kernel against a
    kernel and every row below would be meaningless.

    THE REFUSAL REASON IS RECORDED, NOT INFERRED, and it matters that it is: on
    THIS host the shipped planner refuses first because *the backend is not CuPy*
    — the Triton dispatch route has no MPS arm — which is a fact about the host,
    not about this family. Reading "plan=None" as "the family is unwired" would be
    reading a strictly weaker measurement than the artifact claims. The unwired
    claim's real evidence is the composition probe's disjointness leg, which asks
    every registered arm for a verdict on every case; this leg records the reason
    text beside the null so the two are not confused.
    """
    rows: List[Dict[str, Any]] = []
    previous = os.environ.get(fastpath.DISPATCH_ENABLE)
    try:
        for case in gate.CASES:
            driver = build_driver(case)
            record: Dict[str, Any] = {"case": case["name"],
                                      "complex": case["complex"]}
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
            log(f"[wiring] {case['name']:<28} ambient={record['ambient']['plan']} "
                f"forced_on={record['forced_on']['plan']} "
                f"because={record['forced_on']['refused_because']!r}")
            assert not record["forced_on"]["plan"], (
                f"{case['name']}: the SHIPPED planner claims "
                f"{record['forced_on']['plan']} on a beta configuration. This "
                f"family is unwired, so the oracle driver below would be running "
                f"kernels and comparing them against themselves")
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
#: like. `seeded_and_sourced` starts from a physical-band random state with a ±0
#: lattice, because a zero initial state makes step 1's constitutive sub-step a
#: fixed point and hides a defect that only shows on arbitrary data.
VARIANTS = (("sourced_from_zero", None), ("seeded_and_sourced", SEED))


def leg_driver_steps(payload: Dict[str, Any], out: str) -> None:
    """``FdtdDriver.step()`` for the budget, device against array path, per step.

    Compared after EVERY step over the full stored inventory as uint32 words. The
    first divergence is reported with its case, step and array and the leg STOPS
    there — a family that diverges at step 7 and is compared for 17 more steps
    produces a wall of consequences and one cause.
    """
    rows: List[Dict[str, Any]] = []
    probe = expansion_probe.measure()
    started = time.time()
    for case in gate.CASES:
        for variant, seed in VARIANTS:
            device_driver = build_driver(case, sourced=True, seed=seed)
            oracle_driver = build_driver(case, sourced=True, seed=seed)
            names = inventory(device_driver.fields, device_driver.grid.shape)
            missing = [name for name in gate.STATE if name not in names]
            assert not missing, (
                f"the enumerated inventory is missing {missing}, which the gate "
                f"steps: the enumeration found fewer volumes than the transcribed "
                f"list and would compare less than the gate does")
            plans, residency = build_plans(device_driver, probe)
            shim = MetalShim(device_driver.fields, plans, residency)
            install(device_driver, shim)
            install(oracle_driver, None)
            execution = assert_executed_on_metal(
                residency, shim.plans, f"{case['name']}/{variant}")

            initial = snapshot(oracle_driver.fields, names)
            row: Dict[str, Any] = {
                "case": case["name"], "complex": case["complex"],
                "variant": variant, "budget": STEP_BUDGET,
                "device_slots": sorted(shim.plans),
                "array_path_slots": sorted(
                    set(("step_B", "update_H", "step_D", "update_E"))
                    - set(shim.plans)),
                "inventory": len(names), "mirrors": len(residency.names),
                "execution": execution,
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
                moved = sum(differing(oracle[n], initial[n]) for n in names)
                moved_total = moved
                if bad and first_divergence is None:
                    first_divergence = {"step": step,
                                        "arrays": dict(sorted(bad.items()))}
                if step in (1, STEP_BUDGET) or bad:
                    row["steps"].append({"step": step, "moved_from_initial": moved,
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
            log(f"[driver] {case['name']:<28} {variant:<18} "
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
    a fixed point — so if the sourced run moved, the sources moved it, and the
    interleave the device sub-steps ran between is real work rather than a fill of
    zeros. The control is ALSO stepped on the device path, so its null is a measured
    null and not an assumption: a kernel that wrote garbage into a quiescent grid
    would break it.

    Recorded as a PREDICTED NULL with its reason, because that is what it is: the
    control's own device-vs-array comparison is trivially identical and certifies
    nothing on its own.
    """
    rows: List[Dict[str, Any]] = []
    probe = expansion_probe.measure()
    for case in gate.CASES:
        driver = build_driver(case, sourced=True, seed=None)
        control = build_driver(case, sourced=False, seed=None)
        control_array = build_driver(case, sourced=False, seed=None)
        names = inventory(driver.fields, driver.grid.shape)
        for target in (driver, control):
            plans, residency = build_plans(target, probe)
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
        # THE MATERIAL VOLUMES ARE NONZERO BY CONSTRUCTION and are excluded from
        # the quiescence count BY NAME rather than by a threshold: eps and inv_eps
        # are what `set_isotropic_epsilon_volume` wrote, nothing steps them, and
        # counting them made the first run of this leg fail for a reason that had
        # nothing to do with the sources (measured: 640 words = 2 arrays x 320
        # cells). They stay in the BYTE comparison, where a kernel that scribbled
        # past its target would land.
        stepped = tuple(name for name in names if name in gate.STATE)
        constants = tuple(name for name in names if name not in gate.STATE)
        control_nonzero = sum(int(np.count_nonzero(quiet[n])) for n in stepped)
        constant_nonzero = {n: int(np.count_nonzero(quiet[n])) for n in constants}
        control_vs_array = sum(differing(quiet[n], quiet_array[n]) for n in names)
        row = kit.predicted_null(
            {"case": case["name"], "complex": case["complex"],
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
        log(f"[source] {case['name']:<28} injected={injected} "
            f"control_nonzero={control_nonzero} "
            f"control_device_vs_array={control_vs_array}")
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

#: One case per STORAGE FAMILY, as the gate's own census does: a claim naming
#: complex64 storage that censused only float32 would have checked half its scope.
#: BY NAME, not by index. A positional pick silently re-points the moment a case
#: is inserted anywhere but the end, and the artifact still reports a green leg
#: against whichever case slid into the slot.
CENSUS_CASES = (gate.case_named("r2_metallic_x_negative_beta"),
                gate.case_named("c2_marquee_inplane_bloch"))


def _census_step(window: Any, fields: Any, names: Sequence[str], grid: Any,
                 step: int, complex_storage: bool) -> None:
    """Census one step's stored volumes AND the beta term's own intermediate.

    The intermediate is reconstructed on the host with the kernel's operand order
    (``gate._beta_intermediates``). That is weaker than reading the device's
    registers and this probe says so; it is strictly stronger than censusing only
    what was stored, which is what a term whose coefficient is order 1e-2 needs —
    the product is SMALLER than its operand, so a field near the bottom of the
    normal range can be pushed into the band by the term alone with every stored
    array clean.
    """
    state = {name: getattr(fields, name) for name in names}
    for name in names:
        window.observe(f"step{step}:{name}", state[name], step=step)
    for sub_step in gate.SUB_STEPS:
        for label, value in gate._beta_intermediates(  # noqa: SLF001
                grid, sub_step, state, complex_storage).items():
            window.observe_intermediate(f"step{step}:{sub_step}:{label}", value,
                                        step=step)


def leg_precondition(payload: Dict[str, Any], out: str) -> None:
    """The claim is CONDITIONAL and the condition is checked, every step.

    Two halves, and they are different assertions. The physical run must be CLEAN
    across the whole budget or the gate refuses it. And a scaled control must
    DEMONSTRATE the window firing — a precondition never shown to fire is
    decorative, and on this executor the flush is native and has no lever, so the
    firing case is a genuine refusal rather than a warning.
    """
    rows: List[Dict[str, Any]] = []
    probe = expansion_probe.measure()
    started = time.time()
    for case in CENSUS_CASES:
        for label, scale, expect_clean in (("physical", 1.0, True),
                                           ("subnormal_band", 1e-38, False)):
            driver = build_driver(case, sourced=True, seed=SEED, scale=scale)
            names = inventory(driver.fields, driver.grid.shape)
            plans, residency = build_plans(driver, probe)
            shim = MetalShim(driver.fields, plans, residency)
            install(driver, shim)
            oracle_driver = build_driver(case, sourced=True, seed=SEED, scale=scale)
            install(oracle_driver, None)

            window = preconditions.SubnormalWindow(
                1, STEP_BUDGET, per_array_words=1, per_intermediate_words=1)
            complex_storage = bool(case["complex"])
            differing_words = 0
            for step in range(1, STEP_BUDGET + 1):
                driver.step()
                oracle_driver.step()
                if step % CENSUS_EVERY == 0:
                    _census_step(window, driver.fields, names, driver.grid, step,
                                 complex_storage)
                    _census_step(window, oracle_driver.fields, names,
                                 oracle_driver.grid, step, complex_storage)
                differing_words = sum(
                    differing(getattr(driver.fields, n),
                              getattr(oracle_driver.fields, n)) for n in names)
                if differing_words:
                    break
            report = window.report()
            row = {"case": case["name"], "complex": case["complex"],
                   "scale": label, "factor": scale, "budget": STEP_BUDGET,
                   "census_every": CENSUS_EVERY,
                   "window": report, "differing_words": differing_words,
                   "gate_verdict": ("REFUSED (precondition)" if not report["clean"]
                                    else ("identical" if not differing_words
                                          else "DIFFERS"))}
            rows.append(row)
            payload["legs"]["precondition"] = rows
            save(payload, out)
            log(f"[precondition] {case['name']:<28} {label:<14} "
                f"subnormal_words={report['subnormal_words']} "
                f"observed={report['observed_words']} "
                f"differing={differing_words} verdict={row['gate_verdict']} "
                f"({time.time() - started:.1f}s)")
            context = f"{case['name']}/{label}/driver_step"
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
#: level; these exist to answer a question the gate cannot — is the number this
#: probe compared produced by the Metal kernel, inside a real step, with the array
#: passes interleaved? A mutant that runs and is NOT caught here would mean the
#: device path is not reaching the compared bytes.
DRIVER_MUTATIONS: Dict[str, Dict[str, Any]] = {
    "d1_beta_term_dropped": {
        "must_catch": True,
        "why": "the whole feature deleted inside a complete driver step: this is "
               "the row that proves the compared inventory carries the Metal "
               "kernel's output rather than the array path's",
        "apply": lambda source: kit.needle(
            source, "    curl0 = curl0 - (beta_plus * b);", "    // term dropped"),
    },
    "d2_beta_sign_swap": {
        "must_catch": True,
        "why": "the two host-rounded signs exchanged; caught at sub-step level, "
               "re-armed here because a per-step sync could in principle overwrite "
               "the device's answer with the host's before it is compared",
        "apply": lambda source: kit.needle(
            kit.needle(source, "(beta_plus * b)", "(beta_minus * b)"),
            "(beta_minus * a)", "(beta_plus * a)"),
    },
    "d3_whitespace_only": {
        # THE MUST-NOT-CATCH ROW. Without it, "every mutation was caught" is
        # equally consistent with a comparison that reports divergence for any
        # recompile at all.
        "must_catch": False,
        "why": "a comment appended to the kernel body — recompiled, relaunched, "
               "arithmetically identical. It must NOT be caught, or the leg is "
               "detecting recompilation rather than defects",
        "apply": lambda source: kit.needle(
            source, "    curl0 = curl0 - (beta_plus * b);",
            "    curl0 = curl0 - (beta_plus * b);  // same arithmetic"),
    },
}


def leg_mutations(payload: Dict[str, Any], out: str) -> None:
    """Armed, launch-counted, run through ``driver.step()`` for a short budget.

    A SHORT BUDGET ON PURPOSE, and stated: 4 steps. These rows measure whether the
    device reaches the compared bytes, not how a defect accumulates, and the gate's
    own mutation leg is where the arithmetic is armed exhaustively.
    """
    budget = 4
    harness = kit.MutationHarness(payload, out)
    probe = expansion_probe.measure()
    case = gate.case_named("r2_metallic_x_negative_beta")  # wall passes live, so the interleave
    for label, entry in DRIVER_MUTATIONS.items():
        counters: List[kit.Counter] = []
        functions: Dict[str, Dict[str, Any]] = {}
        missed = False
        codes = None
        driver = build_driver(case, sourced=True, seed=SEED)
        oracle_driver = build_driver(case, sourced=True, seed=SEED)
        install(oracle_driver, None)
        from meep_gpu import stepping  # noqa: PLC0415
        codes = [1 if kind == "metallic" else 0
                 for kind in stepping._boundary_kinds(driver.grid, driver.pml)]
        for sub_step in ("step_B", "step_D"):
            spec = gate.SUB_STEPS[sub_step]
            shipped = special_kz.beta_curl_source(codes, spec["backward"], True)
            try:
                mutated = entry["apply"](shipped)
            except LookupError:
                missed = True
                continue
            function = getattr(device.compile_source(mutated),
                               "beta_pml_curl_step")
            counter = kit.Counter(function)
            counters.append(counter)
            functions[sub_step] = {templates.CONTRACT_OFF: counter}
        plans, residency = build_plans(driver, probe, functions=functions)
        shim = MetalShim(driver.fields, plans, residency)
        install(driver, shim)
        names = inventory(driver.fields, driver.grid.shape)
        for _ in range(budget):
            driver.step()
            oracle_driver.step()
        caught = int(any(differing(getattr(driver.fields, n),
                                   getattr(oracle_driver.fields, n))
                         for n in names))
        launches = sum(counter.launches for counter in counters)
        # ONE DRIVER RUN IS ONE CASE, and the count says so. The gate's rows are
        # per sub-step because it launches them in isolation; here both mutated
        # sub-steps run inside the SAME driver run and the comparison is one
        # verdict over one final inventory, so `ran` is 1 and the number of
        # sub-steps carrying the needle is recorded beside it. Counting `ran` as
        # the sub-step count instead reported `CAUGHT 1/2` for a mutation that was
        # caught, which is a false FAILURE — and the mirror image of it, on a row
        # whose two sub-steps disagreed, would be a false pass.
        ran = 1 if functions else 0
        harness.record(label, harness.verdict(missed, ran, launches, caught),
                       launches, caught, ran, entry["must_catch"], entry["why"],
                       extra={"budget": budget,
                              "sub_steps_carrying_the_needle": sorted(functions),
                              "dispatched": dict(sorted(shim.dispatched.items()))})
        assert launches == budget * len(functions), (
            f"{label}: the mutated kernels launched {launches} times against "
            f"{budget * len(functions)} expected dispatches, so the row's verdict "
            f"is about a different number of launches than it claims")
        driver.close()
        oracle_driver.close()


# ---------------------------------------------------------------------------
# LEG monitors — what a run actually reads out, with its SAMPLE COUNT asserted
# ---------------------------------------------------------------------------

def leg_monitors(payload: Dict[str, Any], out: str) -> None:
    """A DFT monitor and a flux monitor over the budget, on both paths.

    THE FIELDS ARE NOT THE DELIVERABLE. A run's answer is what its monitors
    accumulated, and the monitors are the last thing ``driver.step()`` touches
    (driver.py:3307-3311) — they read the volumes the device wrote, at every step,
    through a completely different code path from the one the byte comparison walks.
    A family that was byte-identical in the arrays and wrong in the readout would
    pass every other leg here.

    THE SAMPLE COUNT IS ASSERTED, not assumed: a monitor that accumulated fewer
    steps than the budget agrees with an oracle that also accumulated fewer, and the
    pair would be identical and meaningless. Both counts are recorded and both are
    checked against the budget, and the accumulator is required to be non-vacuous.

    The comparison uses :func:`exact_differing`, not ``kit.differing``: the flux
    spectrum is float64.
    """
    rows: List[Dict[str, Any]] = []
    probe = expansion_probe.measure()
    started = time.time()
    for case in gate.CASES:
        device_driver = build_driver(case, sourced=True, seed=None)
        oracle_driver = build_driver(case, sourced=True, seed=None)
        monitors = {}
        for label, driver in (("device", device_driver), ("oracle", oracle_driver)):
            monitors[label] = {
                "dft": driver.add_dft_monitor(
                    frequencies=1.0, components=("Ez",),
                    center=(0.0, 0.0, 0.0), size=(1.0, 0.8, 0.0)),
                "flux": driver.add_flux_monitor(
                    frequencies=1.0, center=(0.3, 0.0, 0.0), size=(0.0, 0.8, 0.0)),
            }
        plans, residency = build_plans(device_driver, probe)
        shim = MetalShim(device_driver.fields, plans, residency)
        install(device_driver, shim)
        install(oracle_driver, None)

        for _ in range(STEP_BUDGET):
            device_driver.step()
            oracle_driver.step()

        got_dft = monitors["device"]["dft"]._dft  # noqa: SLF001 - the accumulator
        want_dft = monitors["oracle"]["dft"]._dft  # noqa: SLF001
        dft_differing = sum(exact_differing(got_dft[k], want_dft[k])
                            for k in want_dft)
        dft_nonzero = sum(int(np.count_nonzero(want_dft[k])) for k in want_dft)
        got_flux = monitors["device"]["flux"].get_flux_spectrum()
        want_flux = monitors["oracle"]["flux"].get_flux_spectrum()
        flux_differing = exact_differing(got_flux, want_flux)
        samples = {f"{label}_{kind}": int(monitor._update_calls)  # noqa: SLF001
                   for label, group in monitors.items()
                   for kind, monitor in group.items()}
        # THE READOUT COMPARISONS COUNT TOO. The summary's `comparisons` is what a
        # reader checks the claim's weight against, and a leg whose comparisons are
        # invisible to the tally understates it — one per DFT component plus one
        # for the spectrum, with their outcome carried into `disagreed`.
        TALLY["compared"] += len(want_dft) + 1
        TALLY["disagreed"] += int(bool(dft_differing)) + int(bool(flux_differing))
        row = {"case": case["name"], "complex": case["complex"],
               "budget": STEP_BUDGET, "samples": samples,
               "dft_components": sorted(want_dft),
               "dft_differing_words": dft_differing,
               "dft_nonzero_cells": dft_nonzero,
               "flux_differing_words": flux_differing,
               "flux_spectrum": [float(v) for v in np.atleast_1d(want_flux)],
               "device_dispatched": dict(sorted(shim.dispatched.items()))}
        rows.append(row)
        payload["legs"]["monitors"] = rows
        save(payload, out)
        log(f"[monitors] {case['name']:<28} samples={sorted(set(samples.values()))} "
            f"dft_differing={dft_differing} flux_differing={flux_differing} "
            f"flux={row['flux_spectrum']} ({time.time() - started:.1f}s)")

        context = case["name"]
        for name, count in samples.items():
            assert count == STEP_BUDGET, (
                f"{context}: monitor {name} accumulated {count} samples against a "
                f"budget of {STEP_BUDGET}. A monitor that sampled fewer steps than "
                f"the run agrees with an oracle that did the same, and the pair is "
                f"identical and meaningless")
        kit.assert_census_floor(dft_nonzero, f"{context} DFT accumulator")
        assert dft_differing == 0, (
            f"{context}: the DFT readout differs in {dft_differing} words while the "
            f"field arrays agree — the divergence is in what the run REPORTS")
        assert flux_differing == 0, (
            f"{context}: the flux spectrum differs in {flux_differing} words")
        assert shim.dispatched, (
            f"{context}: the device took no sub-step, so this readout was produced "
            f"entirely by the array path")
        device_driver.close()
        oracle_driver.close()


LEGS = (
    ("wiring", leg_wiring),
    ("driver_steps", leg_driver_steps),
    ("source_control", leg_source_control),
    ("monitors", leg_monitors),
    ("precondition", leg_precondition),
    ("mutations", leg_mutations),
)

REQUIRED = tuple(name for name, _ in LEGS)


def main() -> int:
    parser = kit.argument_parser(__doc__)
    arguments = parser.parse_args()

    started = time.time()
    out_dir = os.path.dirname(os.path.abspath(arguments.out)) or "."
    os.makedirs(out_dir, exist_ok=True)

    environment = kit.environment_stamp()
    kernel_dir = os.path.join(API_ROOT, "meep_gpu", "metal_kernels")
    payload: Dict[str, Any] = {
        "environment": environment,
        "subnormal_policy": subnormal.mps_policy_report(),
        "provenance": kit.provenance(out_dir, {
            "stepping.py": os.path.join(API_ROOT, "meep_gpu", "stepping.py"),
            "driver.py": os.path.join(API_ROOT, "meep_gpu", "driver.py"),
            "fastpath.py": os.path.join(API_ROOT, "meep_gpu", "fastpath.py"),
            "metal_kernels/special_kz.py": os.path.join(kernel_dir,
                                                        "special_kz.py"),
            "metal_kernels/plans.py": os.path.join(kernel_dir, "plans.py"),
            "metal_kernels/device.py": os.path.join(kernel_dir, "device.py"),
            "metal_kernels/preconditions.py": os.path.join(kernel_dir,
                                                           "preconditions.py"),
            "gate_metal_special_kz.py": os.path.join(
                HERE, "gate_metal_special_kz.py"),
            "probe_metal_special_kz_driver_step.py": os.path.abspath(__file__),
        }, kernel_sources=special_kz.enumerate_sources(),
            # ITS OWN FILE NAME, and not a tidiness preference. This probe shares a
            # results directory with the gate and both call `kit.provenance`, whose
            # default name is `provenance.json`; running second with the default
            # OVERWRITES the gate's, which is the evidence-destroying collision
            # metal_gate_kit.provenance documents from the offdiag family. Measured
            # here on the first cut of this directory: the surviving file carried
            # this probe's nine sources and had silently dropped the gate's
            # coverage.py, shaders.py, templates.py and expansion-probe hashes,
            # while both artifacts still stamped `passed`.
            name="provenance_driver_step.json"),
        "legs": {},
        "budget": {"driver_steps": STEP_BUDGET, "mutations": 4,
                   "census_every": CENSUS_EVERY},
        "question": ("does the certified special_kz curl still reproduce the array "
                     "path INSIDE a complete FdtdDriver.step(), for a stated "
                     "budget, with sources injecting and the wall passes live — "
                     "and did the device path run at all"),
    }
    save(payload, arguments.out)

    log(f"[env] torch={environment.get('torch')} numpy={environment['numpy']} "
        f"frontend={environment.get('metal_frontend')} "
        f"policy={payload['subnormal_policy']['resolved']} "
        f"budget={STEP_BUDGET}")

    if not environment.get("mps_available"):
        return kit.cannot_certify(payload, arguments.out, [
            "no MPS device on this host: every leg here launches a Metal kernel, "
            "and a run that skipped them all must not mint a green artifact"])
    if not payload["subnormal_policy"]["admitted"]:
        return kit.cannot_certify(payload, arguments.out, [
            f"the MPS executor refused the resolved subnormal policy: "
            f"{payload['subnormal_policy']['reasons']}"])

    wanted = kit.wanted_legs(arguments.legs)
    ran = kit.run_legs(LEGS, payload, arguments.out, wanted)
    missing = [name for name in REQUIRED if name not in ran]
    payload["counters"] = dict(TALLY)
    if missing:
        return kit.cannot_certify(payload, arguments.out, [
            f"required legs did not run: {missing}. A --legs subset is for "
            f"debugging and must not be able to mint a certified artifact"])
    return kit.summarize(
        payload, arguments.out,
        claim=("byte-identity to the NumPy array path across a COMPLETE "
               "FdtdDriver.step(), for a stated budget of "
               f"{STEP_BUDGET} steps, under a CHECKED subnormal-free "
               "precondition censused on every step"),
        scope=("the special_kz (grid.beta) split-field PML curl inside the "
               "driver's own step order, real float32 and complex64 storage, "
               "periodic and metallic ghost rules, in-plane Bloch phase, both "
               "corpus beta signs, Courants 0.34/0.4375/0.35/0.5, gaussian "
               "magnetic AND electric sources injecting at both seams; the "
               "composition is the DECLARED-SYNC one — no mirror is held across "
               "an array pass — and the throughput question is not asked here"),
        stated_weakness=("no PTX-equivalent audit exists on this executor: "
                         "compile_shader exposes no disassembly, so the mutation "
                         "leg and the launch counters are the only arbiters that "
                         "the device did what the source says"),
        started=started, legs_run=ran,
        compared=TALLY["compared"], certified=(TALLY["disagreed"] == 0),
        extra={"not_covered": (
            "the COMPLEX family's constitutive pair, which runs on the array path "
            "in every row here because no complex constitutive product is "
            "certified on this backend; and throughput, which this probe's "
            "per-dispatch sync makes meaningless by construction")})


if __name__ == "__main__":
    raise SystemExit(main())
