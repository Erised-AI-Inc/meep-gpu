"""ENGINE-ROUTE byte probe for the Metal BFAST curl — the binding the gate does not compare.

WHY THIS EXISTS, stated as the gap it closes rather than as extra assurance.
``gate_metal_bfast`` certifies the family through
``plan_bfast_pml_curl_from_arrays``: the GATE supplies the coefficient dictionary
(picking the Yee suffix itself), the GATE supplies the six scalars, and the plan
mirrors bare host arrays the gate allocated. That is the right substrate for
mutation — a deliberately wrong configuration has to be constructible — but it means
the SHIPPED ENGINE ROUTE, ``bfast.plan_bfast_pml_curl(fields, pml, sub_step,
residency)``, which reaches into a live ``Fields``/``PML`` for its mirrors and
computes its own scalars, is exercised by the gate only in leg 4 (pointer identity)
and leg 9 (a plan was built). NEITHER COMPARES ITS BYTES AGAINST ``stepping.py``.
A wrong suffix, a swapped stem order, a mirror bound to the wrong volume or a k
vector read off the wrong axis would pass both of those legs and this probe is where
it lands.

THE SECOND GAP: HORIZON. The gate's multi-step leg runs FOUR complete driver steps.
For a recurrence whose homogeneous mode is ``(-1)^n``, undamped forever
(stepping.py:868-874), four steps is a short look — the whole hazard of this family
is a state error that never decays and therefore never has to appear early. This
probe runs the same comparison per COMPLETE DRIVER STEP over a long horizon, on a
seeded lattice (where ``-2*state`` is live from step 1) and on a DRIVEN run from
rest (where the values are the ones a real simulation carries).

WHAT IS COMPARED, and it is the same discipline as the gate: uint32 word equality on
float32 storage, never ``allclose``, over the FULL STORED INVENTORY (all 30 volumes:
both curl targets, both PML auxiliary sets, both constitutive sides, and the six IIR
states) after every complete driver step. The oracle is ``stepping.py`` in this
process, on a SECOND ``Fields`` object advanced in lockstep — never a rewind of the
same object, because ``f_bfast`` is exactly the array an incomplete restore misses.

THE RESIDENCY INVARIANT IS PART OF THE ROUTE. A Metal plan owns persistent device
mirrors; ``KernelPlan.run`` does NOT sync them in. Between the two curl launches the
array path (``update_H``/``update_E``) writes the host arrays, so a real engine loop
must ``sync_in`` before each launch. Leg 5 measures that the staleness is REAL — it
counts the differing words a skipped ``sync_in`` leaves — and that skipping it
DIVERGES from ``stepping.py``. A sync whose absence changes nothing would mean this
probe's agreement was never testing the mirrors at all.

CASE DISCIPLINE, inherited from the gate and non-negotiable here: a non-power-of-two
Courant first (0.105679 — the marquee corpus row's own ``(1-kx)/sqrt(3)``); every leg
proves the step MOVED STATE; the subnormal-free precondition is CHECKED per step over
operands, results AND the tail's reconstructed intermediates, and a breach ENDS THE
CLAIM for that case at that step (recorded as REFUSED, characterised, and followed
for :data:`POST_BREACH_STEPS` further steps in observation mode) while a divergence
with a CLEAN census raises, because that would be a defect; the driven leg carries a
SOURCE-FREE control, because a run that was never excited agrees with itself
trivially; every launch is counted, and an arming leg proves each comparison can
fail.

WHAT THIS PROBE BORROWS from ``gate_metal_bfast``: ``build`` (the Grid/Fields/PML
triple and its seeding), ``STATE`` (the stored inventory), ``snapshot``, and
``reference_bfast_curl`` (the host transcription used ONLY to reconstruct the tail's
un-storable ``total``/``advance`` intermediates for the census — the gate's leg 0
pins it against ``stepping.py`` over full cycles). The COMPARISON itself — word
extraction, per-array counts, first divergence — is spelled here rather than taken
from the gate or the kit, so a defect in the gate's comparator cannot make this probe
agree with it.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
API_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
for path in (API_ROOT, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

# Requested BEFORE anything resolves a policy, exactly as the gate does: the MPS
# executor delivers `flush` natively and cannot deliver `keep` (fact (e)), and the
# claim is only as good as the precondition it was certified under.
os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")

from meep_gpu import stepping  # noqa: E402
from meep_gpu.metal_kernels import bfast_curl as bfast  # noqa: E402
from meep_gpu.metal_kernels import device, preconditions, subnormal  # noqa: E402

import gate_metal_bfast as gate  # noqa: E402

SUB_STEPS = bfast.SUB_STEPS
STATES = bfast.BFAST_STATE_NAMES

#: A non-power-of-two Courant FIRST. 0.105679 is the marquee corpus row's own
#: (1 - kx)/sqrt(3); 0.35 is a second, also not a power of two.
COURANTS: Tuple[float, ...] = (0.105679, 0.35)

#: Three boundary triples: all-periodic (no wall, no mask), mixed (both live on the
#: same grid) and all-metallic (every wall and every mask).
CONFIGS: Tuple[Tuple[str, Tuple[float, float, float], Any], ...] = (
    ("periodic_all", (1.2, 1.0, 0.9), "periodic"),
    ("metallic_xy", (1.2, 1.0, 0.9), ("metallic", "metallic", "periodic")),
    ("metallic_all", (1.1, 1.0, 0.9), "metallic"),
)

#: The marquee (k along x alone — four of the six scalars are zero) and a k with all
#: three components nonzero, so every scalar is live somewhere in the sweep.
BFAST_KS: Tuple[Tuple[str, Tuple[float, float, float]], ...] = (
    ("marquee_x", (0.816958, 0.0, 0.0)),
    ("full_3d", (0.31, 0.17, 0.23)),
)

#: THE STEP BUDGET, declared once and consumed as the legs' defaults.
SEEDED_STEPS = 120
DRIVEN_STEPS = 300
SEED = 20260816

#: How far a run is followed PAST a precondition breach, in observation mode. The
#: point is to report what the flush costs rather than only that it happened.
POST_BREACH_STEPS = 20

#: The floor under leg 3/4's "the IIR carried something" statistic. A driven run
#: from rest cannot move f_bfast on the very first steps (the drive has to reach the
#: partner components first), so the non-vacuity test is a FRACTION of the budget
#: rather than every step — stated here rather than chosen when the number came in.
STATE_MOVE_FRACTION = 0.5


def log(message: str) -> None:
    print(message, flush=True)


def save(payload: Dict[str, Any], path: str) -> None:
    import json  # noqa: PLC0415

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True, default=str)


def words(array: Any) -> Any:
    """The float32 storage as uint32 words. Spelled here, not borrowed."""
    flat = np.ascontiguousarray(array, dtype=np.float32).reshape(-1)
    return flat.view(np.uint32)


def differing(left: Any, right: Any) -> int:
    return int(np.count_nonzero(words(left) != words(right)))


def per_name_differing(got: Dict[str, Any], expected: Dict[str, Any],
                       names: Sequence[str]) -> Dict[str, int]:
    counts = {name: differing(got[name], expected[name]) for name in names}
    return {name: count for name, count in counts.items() if count}


def first_divergence(got: Dict[str, Any], expected: Dict[str, Any],
                     names: Sequence[str]) -> Optional[Dict[str, Any]]:
    """The first array and the first WORD at which the device left the array path."""
    for name in names:
        left, right = words(got[name]), words(expected[name])
        if left.shape != right.shape:
            return {"array": name, "reason": "shape mismatch"}
        bad = np.flatnonzero(left != right)
        if bad.size:
            index = int(bad[0])
            return {"array": name, "flat_index": index,
                    "differing_words": int(bad.size),
                    "device_word": int(left[index]),
                    "oracle_word": int(right[index]),
                    "device_value": float(
                        np.asarray(got[name]).reshape(-1)[index]),
                    "oracle_value": float(
                        np.asarray(expected[name]).reshape(-1)[index])}
    return None


def independent_ks(grid, sub_step: str) -> Tuple[float, ...]:
    """The six scalars recomputed from ``stepping.py``'s OWN tables.

    NOT ``bfast_curl_coefficients`` — that helper and the plan builder are the thing
    under test here, and re-running it would compare a value with itself. This walks
    ``stepping.B_CURL_TERMS`` / ``D_CURL_TERMS``, applies ``stepping._bfast_axis``
    and the have_p/have_m gating, negates the D side in host float64 and rounds ONCE
    to float32, which is the sequence ``_bfast_term`` performs at :882-897.
    """
    terms = (stepping.B_CURL_TERMS if sub_step == "step_B"
             else stepping.D_CURL_TERMS)
    magnetic = sub_step == "step_B"
    scaled = tuple(float(v) for v in grid.bfast_scaled_k)
    out: List[float] = []
    for term in terms:
        have_p = not grid.is_invariant(term.first_axis)
        have_m = not grid.is_invariant(term.second_axis)
        k1 = scaled[stepping._bfast_axis(term.second)] if have_m else 0.0
        k2 = scaled[stepping._bfast_axis(term.first)] if have_p else 0.0
        if not magnetic:
            k1, k2 = -k1, -k2
        out.append(float(np.float32(k1)))
        out.append(float(np.float32(k2)))
    return tuple(out)


def band_cells(array: Any) -> Any:
    """The cells in the float32 subnormal band: nonzero and below the normal floor."""
    magnitude = np.abs(np.asarray(array, dtype=np.float32))
    return (magnitude > 0) & (magnitude <= np.float32(preconditions.MAX_SUBNORMAL))


def census_step(state: Dict[str, Any], pml, codes, dtdx: float,
                ks: Dict[str, Sequence[float]],
                window: preconditions.SubnormalWindow,
                step: int, bfast: bool = True) -> Tuple[int, float,
                                                        List[Dict[str, Any]]]:
    """Census one step: every stored volume plus the tail's own intermediates.

    Returns (subnormal words found, smallest nonzero magnitude seen, offenders).
    ``offenders`` NAMES the volumes that carried band words and the smallest
    magnitude each carried — a breach reported as a bare count cannot be attributed
    to an array, and attribution is the whole point of a per-step census. The
    intermediates are reconstructed on a COPY — ``reference_bfast_curl`` updates in
    place — so the live run is untouched.
    """
    found = 0
    smallest = float("inf")
    offenders: List[Dict[str, Any]] = []

    def note(label: str, value: Any, count: int) -> None:
        if not count:
            return
        cells = band_cells(value)
        magnitudes = np.abs(np.asarray(value, dtype=np.float32))[cells]
        offenders.append({"volume": label, "band_words": int(count),
                          "smallest_band_abs": float(magnitudes.min()),
                          "largest_band_abs": float(magnitudes.max()),
                          "flat_indices": [int(i) for i in
                                           np.flatnonzero(cells.reshape(-1))[:8]]})

    for name, value in state.items():
        count = window.observe(f"stored@{step}:{name}", value, step)
        found += count
        note(f"stored:{name}", value, count)
        nonzero = np.abs(np.asarray(value)[np.asarray(value) != 0])
        if nonzero.size:
            smallest = min(smallest, float(nonzero.min()))
    for sub_step, spec in SUB_STEPS.items():
        work = {name: np.array(value, copy=True) for name, value in state.items()}
        # A BFAST-INACTIVE grid allocates no IIR storage at all; the transcription
        # takes zeros it will not read (``has_bfast=False`` skips the tail), which
        # keeps the attribution control on this one code path.
        states = [work[n] if n in work
                  else np.zeros_like(work[spec["targets"][0]])
                  for n in STATES[sub_step]]
        produced = gate.reference_bfast_curl(
            [work[n] for n in spec["targets"]],
            [work["fu_" + n] for n in spec["targets"]],
            [work[n] for n in spec["sources"]],
            states,
            gate.curl_coefficients(pml, spec["suffix"]), codes,
            spec["backward"], dtdx, ks[sub_step], has_bfast=bfast)
        for label, value in produced.items():
            count = window.observe_intermediate(
                f"{sub_step}:{label}@{step}", value, step)
            found += count
            note(f"{sub_step}:{label}", value, count)
            nonzero = np.abs(value[value != 0])
            if nonzero.size:
                smallest = min(smallest, float(nonzero.min()))
    return found, smallest, offenders


def confinement(got: Dict[str, Any], expected: Dict[str, Any],
                names: Sequence[str]) -> Dict[str, Any]:
    """Is every differing word EXACTLY the flush, and nothing else?

    THE QUESTION A BARE BREACH LEAVES OPEN. "The census fired and the bytes
    diverged" is consistent with two very different worlds: the device flushed the
    band cells the census named and agreed everywhere else (fact (e), the cliff, and
    a divergence bounded by 1.18e-38 per cell), or the device is wrong for some
    other reason and the band cell is a coincidence. This separates them by
    classifying EVERY differing word: the oracle's value must be in the band and the
    device's must be an exact zero. Anything else is reported under
    ``unexplained_words`` and is a different, much more serious finding.
    """
    explained = 0
    unexplained: List[Dict[str, Any]] = []
    largest = 0.0
    for name in names:
        left, right = words(got[name]), words(expected[name])
        bad = np.flatnonzero(left != right)
        if not bad.size:
            continue
        device = np.ascontiguousarray(got[name], dtype=np.float32).reshape(-1)
        oracle = np.ascontiguousarray(expected[name], dtype=np.float32).reshape(-1)
        for index in bad:
            index = int(index)
            oracle_value = float(oracle[index])
            device_value = float(device[index])
            in_band = (0.0 < abs(oracle_value)
                       <= float(preconditions.MAX_SUBNORMAL))
            if in_band and device_value == 0.0:
                explained += 1
                largest = max(largest, abs(oracle_value - device_value))
            else:
                unexplained.append({"array": name, "flat_index": index,
                                    "oracle_value": oracle_value,
                                    "device_value": device_value})
    return {"flushed_band_words": explained,
            "unexplained_words": len(unexplained),
            "unexplained": unexplained[:8],
            "largest_absolute_difference": largest,
            "reading": ("every differing word is a band cell the device flushed to "
                        "zero" if not unexplained else
                        "AT LEAST ONE differing word is NOT explained by the "
                        "subnormal flush")}


class EngineRoute:
    """The SHIPPED engine route, driven the way a real step loop would drive it.

    One plan per curl sub-step, built ONCE from the engine's own objects and sharing
    ONE residency — which is the arrangement ``launch.plan_step`` composes and the
    arrangement whose staleness rule leg 5 measures. ``sync`` may be switched off to
    arm that measurement.
    """

    def __init__(self, fields, pml, sync: bool = True,
                 plain: bool = False) -> None:
        from meep_gpu.metal_kernels import launch as metal_launch  # noqa: PLC0415

        self.fields = fields
        self.pml = pml
        self.sync = bool(sync)
        self.plain = bool(plain)
        self.residency = device.Residency()
        self.plans = {}
        for sub_step in SUB_STEPS:
            # ``plain`` takes the CERTIFIED tranche-1 curl instead, which is what the
            # attribution leg needs: the same lattice and the same horizon with the
            # BFAST tail absent entirely.
            plan = (metal_launch.plan_pml_curl(fields, pml, sub_step, self.residency)
                    if plain else
                    bfast.plan_bfast_pml_curl(fields, pml, sub_step, self.residency))
            if plan is None:
                reasons = (metal_launch.pml_curl_coverage(fields, pml, sub_step,
                                                          self.residency).reasons
                           if plain else
                           bfast.bfast_pml_curl_coverage(fields, pml, sub_step,
                                                         self.residency).reasons)
                raise AssertionError(
                    f"the ENGINE route refused {sub_step} on a configuration it "
                    f"covers: {reasons}")
            self.plans[sub_step] = plan
        self.stale_words = 0

    @property
    def launches(self) -> int:
        return sum(plan.launches for plan in self.plans.values())

    def curl(self, sub_step: str) -> None:
        if self.sync:
            # Count what the sync was covering BEFORE it runs: a sync whose absence
            # changes nothing would mean the mirrors were never carrying state.
            self.stale_words += sum(self.residency.verify().values())
            self.residency.sync_in()
        self.plans[sub_step].run()
        self.residency.sync_out()

    def driver_step(self) -> None:
        """One COMPLETE driver step: B -> update_H -> D -> update_E."""
        self.curl("step_B")
        stepping.update_H(self.fields, self.pml)
        self.curl("step_D")
        stepping.update_E(self.fields, self.pml)


class ComposedRoute:
    """The SHIPPED COMPOSER's route: every sub-step ``plan_step`` fills, on the device.

    THE CLAIM THIS EXISTS TO MEASURE IS NEW AS OF TRANCHE 2, and nothing else in this
    family's trio touches it. ``bfast_curl.register_arms`` registers FOUR arms — the
    two curls this family's kernel owns, and ``update_H``/``update_E``, which it fills
    with the CERTIFIED constitutive kernel under a restated predicate. So
    ``launch.plan_step`` now returns a plan for every sub-step of a BFAST run, and the
    composed route runs a COMPLETE DRIVER STEP on the device with no array-path wall
    inside it. :class:`EngineRoute` cannot measure that: it hard-codes
    ``stepping.update_H``/``update_E`` between the curls, which is what the tree did
    before the wiring and is no longer what ships.

    WHY IT IS NOT THE SAME MEASUREMENT AS :class:`EngineRoute` RUN TWICE. Two things
    change and each can be wrong on its own. First, the constitutive halves are now
    computed by a kernel rather than by ``stepping.py``, on a configuration the
    CERTIFIED predicate refuses (clause 11 names BFAST) — the arithmetic is certified
    but this ADMISSION is not, and an admission is exactly what a byte comparison
    tests. Second, the IIR states and the PML auxiliaries now cross a sub-step
    boundary IN DEVICE MEMORY instead of being round-tripped through the host between
    every launch; a mirror bound to the wrong volume, or an aliasing between the two
    families' mirrors of the same array, is invisible while a host round trip papers
    over it every half-step.

    THE SYNC DISCIPLINE IS THE ENGINE'S, not a convenience. One ``sync_in`` at the top
    of the step — the seam where the driver's source injection wrote host arrays — and
    one ``sync_out`` at the bottom, with nothing in between, which is precisely the
    arrangement the residency verdict licenses when every live sub-step is planned.
    ``stale_words`` counts what that sync covered, so a run in which the mirrors never
    went stale reports itself as vacuous rather than passing.
    """

    def __init__(self, fields, pml, sync: bool = True) -> None:
        from meep_gpu.metal_kernels import launch as metal_launch  # noqa: PLC0415

        self.fields = fields
        self.pml = pml
        self.sync = bool(sync)
        self.residency = device.Residency()
        self.step_plan = metal_launch.plan_step(fields, pml,
                                                residency=self.residency,
                                                sources=())
        self.order = tuple(self.step_plan.replaces)
        expected = ("step_B", "update_H", "step_D", "update_E")
        if self.order != expected:
            raise AssertionError(
                f"the SHIPPED composer filled {self.order} on a BFAST run, not "
                f"{expected}; the composed route cannot be measured because it is "
                f"not composed. Refusals: {self.step_plan.reasons}")
        self.selected = dict(self.step_plan.selected)
        self.residency_verdict = self.step_plan.residency
        # A NULL PLAN LAUNCHES NOTHING AND STILL COUNTS. `plan_step` splits the
        # filled slots on a DECLARED `performs_device_work` attribute, and the
        # no-PML family's plan declares False — so "four launches per step" would be
        # satisfied by a composition in which two of them did no device work at all.
        # The byte comparison would catch that (the array path is not run either, so
        # the constitutive sub-steps would simply not happen), but the launch-count
        # assertion is what the reader trusts, and it must not be able to pass
        # hollow.
        hollow = [name for name in self.order
                  if not getattr(self.step_plan.plans[name],
                                 "performs_device_work", True)]
        if hollow:
            raise AssertionError(
                f"the composed route filled {hollow} with plans that declare no "
                f"device work; a launch count over those slots measures nothing")
        # THE RESIDENCY VERDICT IS COMPUTED FOR A SOURCE-FREE STEP (`sources=()`)
        # and is RECORDED, never acted on. The driven case writes a host array at
        # the top of each step, which is a seam the verdict was not asked about —
        # so this route syncs in every step regardless of what the verdict says.
        # Reading the verdict as permission to skip that sync is exactly the defect
        # arm B plants.
        self.stale_words = 0

    @property
    def plans(self):
        return self.step_plan.plans

    @property
    def launches(self) -> int:
        return sum(plan.launches for plan in self.step_plan.plans.values())

    def driver_step(self) -> None:
        """One COMPLETE driver step, every sub-step on the device."""
        if self.sync:
            # Count what the sync was covering BEFORE it runs: a sync whose absence
            # changes nothing would mean the mirrors were never carrying state.
            self.stale_words += sum(self.residency.verify().values())
            self.residency.sync_in()
        for name in self.order:
            self.step_plan.plans[name].run()
        self.residency.sync_out()


def oracle_step(fields, pml) -> None:
    stepping.step_B(fields, pml)
    stepping.update_H(fields, pml)
    stepping.step_D(fields, pml)
    stepping.update_E(fields, pml)


def build_pair(name: str, cell, boundaries, courant: float, bfast_k,
               seeded: bool,
               require_bfast: bool = True) -> Tuple[Any, Any, Any, Any, Any, Any]:
    """Two independent, byte-identical (Grid, Fields, PML) triples."""
    grid, fields, pml = gate.build(cell, boundaries, courant, SEED,
                                   bfast_k=bfast_k)
    grid_o, oracle, pml_o = gate.build(cell, boundaries, courant, SEED,
                                       bfast_k=bfast_k)
    if not seeded:  # A driven run starts from rest, not from seeded noise.
        for volume in gate.STATE:
            for target in (fields, oracle):
                array = getattr(target, volume, None)
                if array is not None:
                    array[...] = np.float32(0.0)
    assert grid.bfast_active == require_bfast, (
        f"{name}: grid.bfast_active is {grid.bfast_active}, and this case needs "
        f"{require_bfast}")
    assert tuple(float(v) for v in grid.bfast_scaled_k) == tuple(
        float(v) for v in bfast_k), "the grid carries a different k than requested"
    # The inventory a BFAST-inactive grid carries is the same minus the six IIR
    # states, which are never allocated there; comparing over the snapshot's own
    # keys keeps the attribution control on the same code path as everything else.
    present = tuple(gate.snapshot(fields))
    drift = per_name_differing(gate.snapshot(oracle), gate.snapshot(fields),
                              present)
    assert not drift, f"{name}: the two builds did not start from the same bytes: {drift}"
    # The two PML objects must carry identical coefficients, or "the routes agree"
    # would be a statement about two different absorbers.
    for suffix in ("", "_h"):
        left = gate.curl_coefficients(pml, suffix)
        right = gate.curl_coefficients(pml_o, suffix)
        bad = {f"{label}{suffix}": differing(left[label], right[label])
               for label in left if differing(left[label], right[label])}
        assert not bad, f"{name}: the two PML objects differ: {bad}"
    return grid, fields, pml, grid_o, oracle, pml_o


# ---------------------------------------------------------------------------
# LEG 1 — what the engine route actually bound
# ---------------------------------------------------------------------------

def leg_binding(payload: Dict[str, Any], out: str) -> None:
    """Every mirror the engine route binds, checked against the engine's own object.

    THE DISCRIMINATION IS RECORDED, not assumed: for each coefficient mirror the
    WRONG-suffix array is compared too, and the row carries how many words separate
    them. If the two suffixes happened to hold identical bytes, "the right one is
    bound" would be an unfalsifiable statement on this configuration, and the row
    would say so instead of passing.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    for name, cell, boundaries in CONFIGS:
        grid, fields, pml = gate.build(cell, boundaries, COURANTS[0], SEED,
                                       bfast_k=BFAST_KS[1][1])
        for sub_step, spec in SUB_STEPS.items():
            residency = device.Residency()
            plan = bfast.plan_bfast_pml_curl(fields, pml, sub_step, residency)
            assert plan is not None, "the engine route refused a covered case"
            mirrors = residency._mirrors  # noqa: SLF001 - the audit's whole subject

            volume_binding = {}
            for volume in (tuple(spec["targets"])
                           + tuple("fu_" + n for n in spec["targets"])
                           + tuple(spec["sources"]) + STATES[sub_step]):
                mirror = mirrors[volume]
                volume_binding[volume] = mirror.host is getattr(fields, volume)

            wrong = "" if spec["suffix"] else "_h"
            coefficient_binding = {}
            separation = {}
            for axis in "xyz":
                for stem in ("kms", "sinv"):
                    label = f"pml:{stem}_{axis}{spec['suffix']}"
                    mirror = mirrors[label]
                    right = getattr(pml, f"{stem}_{axis}{spec['suffix']}")
                    other = getattr(pml, f"{stem}_{axis}{wrong}")
                    coefficient_binding[label] = mirror.host is right
                    separation[f"{stem}_{axis}"] = differing(right, other)

            derived = independent_ks(grid, sub_step)
            row = {
                "case": name, "sub_step": sub_step,
                "arguments": len(plan._args),  # noqa: SLF001
                "volumes_bound_to_engine_arrays": all(volume_binding.values()),
                "volume_binding": volume_binding,
                "suffix_expected": spec["suffix"] or "(integer)",
                "coefficients_bound_to_the_right_suffix":
                    all(coefficient_binding.values()),
                "wrong_suffix_separation_words": separation,
                "plan_ks": list(plan.ks),
                "independently_derived_ks": list(derived),
                "ks_agree": [float(a) for a in plan.ks] == [float(b) for b in derived],
                "mirror_count": len(mirrors),
            }
            rows.append(row)
            log(f"[leg1 binding] {name:<13} {sub_step} args={row['arguments']} "
                f"volumes_ok={row['volumes_bound_to_engine_arrays']} "
                f"suffix={row['suffix_expected']} coeff_ok="
                f"{row['coefficients_bound_to_the_right_suffix']} "
                f"ks_agree={row['ks_agree']} ({time.time() - started:.1f}s)")
            payload["legs"]["binding"] = rows
            save(payload, out)
            assert row["arguments"] == 29, row
            assert row["volumes_bound_to_engine_arrays"], row
            assert row["coefficients_bound_to_the_right_suffix"], row
            assert all(count > 0 for count in separation.values()), (
                f"{name}/{sub_step}: a wrong-suffix coefficient vector is "
                f"byte-identical to the right one on this configuration, so "
                f"'the right suffix is bound' is not falsifiable here: {separation}")
            assert row["ks_agree"], (
                f"{name}/{sub_step}: the plan's six scalars disagree with the "
                f"values derived independently from stepping.py's own tables: "
                f"{row['plan_ks']} vs {row['independently_derived_ks']}")
            assert len(mirrors) == 18, row


# ---------------------------------------------------------------------------
# LEG 2 — one engine-route launch against stepping.py
# ---------------------------------------------------------------------------

def leg_single(payload: Dict[str, Any], out: str) -> None:
    rows: List[Dict[str, Any]] = []
    started = time.time()
    total = 0
    expected = len(CONFIGS) * len(COURANTS) * len(BFAST_KS) * len(SUB_STEPS)
    for name, cell, boundaries in CONFIGS:
        for courant in COURANTS:
            for k_label, bfast_k in BFAST_KS:
                for sub_step in SUB_STEPS:
                    grid, fields, pml, _, oracle, pml_o = build_pair(
                        name, cell, boundaries, courant, bfast_k, seeded=True)
                    before = gate.snapshot(fields)
                    getattr(stepping, sub_step)(oracle, pml_o)
                    after = gate.snapshot(oracle)

                    residency = device.Residency()
                    plan = bfast.plan_bfast_pml_curl(fields, pml, sub_step,
                                                     residency)
                    assert plan is not None
                    plan.run()
                    residency.sync_out()
                    got = gate.snapshot(fields)

                    compared = (tuple(SUB_STEPS[sub_step]["targets"])
                                + tuple("fu_" + n for n
                                        in SUB_STEPS[sub_step]["targets"])
                                + STATES[sub_step])
                    bad = per_name_differing(got, after, compared)
                    moved = sum(differing(after[n], before[n]) for n in compared)
                    state_moved = sum(differing(after[n], before[n])
                                      for n in STATES[sub_step])
                    total += 1
                    row = {"case": name, "courant": courant, "k": k_label,
                           "sub_step": sub_step, "moved": moved,
                           "bfast_state_moved": state_moved,
                           "launches": plan.launches, "differing": bad,
                           "first_divergence": first_divergence(got, after,
                                                                compared)}
                    rows.append(row)
                    log(f"[leg2 single] case {total}/{expected} {name:<13} "
                        f"C={courant} k={k_label:<9} {sub_step} moved={moved:<6} "
                        f"f_bfast={state_moved:<5} launches={plan.launches} "
                        f"{'IDENTICAL' if not bad else 'DIFFERS ' + str(bad)} "
                        f"({time.time() - started:.1f}s)")
                    payload["legs"]["single"] = rows
                    save(payload, out)
                    assert plan.launches == 1, row
                    assert moved > 0, f"VACUOUS: {row}"
                    assert state_moved > 0, f"VACUOUS (no IIR motion): {row}"
                    assert not bad, row


# ---------------------------------------------------------------------------
# LEG 3 / LEG 4 — the long horizon, per COMPLETE driver step
# ---------------------------------------------------------------------------

def run_horizon(payload: Dict[str, Any], out: str, leg: str, case: str,
                cell, boundaries, bfast_k, steps: int, seeded: bool,
                driven: bool, rows: List[Dict[str, Any]], started: float,
                use_bfast: bool = True, composed: bool = False) -> Dict[str, Any]:
    """One long run of both routes in lockstep, compared after EVERY complete step.

    TWO OUTCOMES, AND THEY ARE NOT THE SAME OUTCOME.

    * A BYTE DIVERGENCE WITH A CLEAN CENSUS is a defect and raises. There is no
      configuration of this family in which the device may leave the array path
      while every operand, result and intermediate is a normal number.
    * A PRECONDITION BREACH — a band word anywhere in the census — ENDS THE CLAIM
      for that case at that step, and the case is recorded as REFUSED rather than
      failed. The claim was always "byte-identity subject to a CHECKED
      subnormal-free precondition"; a breach is the precondition being FALSE, which
      is the condition doing its job, not the kernel being wrong.

    A breach is not merely noted. It is CHARACTERISED: which volume carried the band
    word, what the divergence at that step was, whether EVERY differing word is
    explained by the flush (:func:`confinement`), and — over
    :data:`POST_BREACH_STEPS` further steps run WITHOUT assertions — whether the
    error stays confined or grows. "Not byte-identical past step N" and "wrong by
    5e-39 in one cell, and here is what it does next" are different reports, and the
    second is the one worth having.
    """
    grid, fields, pml, _, oracle, pml_o = build_pair(
        case, cell, boundaries, COURANTS[0], bfast_k, seeded=seeded,
        require_bfast=use_bfast)
    # The COMPOSED route puts all four sub-steps on the device, so it launches four
    # times per complete driver step; the per-sub-step route launches twice and runs
    # the constitutive halves on the array path. The rate is carried alongside the
    # route rather than hard-coded, because the launch assertion below is the only
    # thing standing between "the device stepped it" and a silent host fallback.
    route = (ComposedRoute(fields, pml) if composed
             else EngineRoute(fields, pml, plain=not use_bfast))
    launches_per_step = len(route.order) if composed else 2
    state_names = tuple(n for n in STATES["step_B"] + STATES["step_D"]
                        if getattr(fields, n, None) is not None)
    codes = gate.boundary_codes(grid, pml)
    dtdx = float(grid.dt / grid.dx)
    ks = {s: gate.plan_ks(grid, s) for s in SUB_STEPS}
    # The intermediate floor applies where intermediates EXIST. The attribution
    # control has no tail, so there is nothing to reconstruct and demanding one
    # would make its window report vacuous for a structural reason rather than a
    # measured one; its census covers the stored inventory, which is where the
    # breach it is chasing was found in the first place.
    window = preconditions.SubnormalWindow(
        0, steps, per_intermediate_words=1 if use_bfast else 0)
    centre = tuple(n // 2 for n in grid.shape)
    previous = gate.snapshot(oracle)
    steps_with_state_move = 0
    smallest = float("inf")
    peak_subnormal = 0
    breach: Optional[Dict[str, Any]] = None
    clean_steps = 0

    def excite(step: int) -> None:
        if not driven:
            return
        # The SAME float32 word into both routes: a drive that differed between them
        # would make every later comparison meaningless.
        t = step * float(grid.dt)
        envelope = float(np.exp(-((t - 1.2) ** 2) / (2 * 0.25 ** 2)))
        excitation = np.float32(envelope * np.sin(2 * np.pi * 3.0 * t))
        fields.Ez[centre] += excitation
        oracle.Ez[centre] += excitation

    for step in range(1, steps + 1):
        excite(step)
        oracle_step(oracle, pml_o)
        route.driver_step()

        got = gate.snapshot(fields)
        expected = gate.snapshot(oracle)
        compared = tuple(expected)
        bad = per_name_differing(got, expected, compared)
        moved = sum(differing(expected[n], previous[n]) for n in compared)
        state_moved = sum(differing(expected[n], previous[n])
                          for n in state_names)
        found, floor, offenders = census_step(expected, pml_o, codes, dtdx, ks,
                                              window, step, bfast=use_bfast)
        peak_subnormal = max(peak_subnormal, found)
        smallest = min(smallest, floor)
        steps_with_state_move += 1 if state_moved else 0

        row = {"leg": leg, "case": case, "step": step, "of_steps": steps,
               "arrays_compared": len(compared), "moved": moved,
               "bfast_state_moved": state_moved, "subnormal_words": found,
               "launches": route.launches, "differing": bad,
               "first_divergence": first_divergence(got, expected, compared)}
        if offenders:
            row["census_offenders"] = offenders
        rows.append(row)
        if step <= 5 or step % 10 == 0 or step == steps or bad or found:
            log(f"[{leg}] {case:<24} step {step}/{steps} arrays={len(compared)} "
                f"moved={moved:<7} f_bfast={state_moved:<5} subnormal={found} "
                f"launches={route.launches} smallest|f|="
                f"{'n/a' if floor == float('inf') else f'{floor:.3e}'} "
                f"{'IDENTICAL' if not bad else 'DIFFERS ' + str(bad)} "
                f"({time.time() - started:.1f}s)")
        payload["legs"][leg] = rows
        save(payload, out)

        assert moved > 0, f"VACUOUS: {case} step {step} moved no state at all"
        assert route.launches == launches_per_step * step, (
            f"the Metal path did not launch on every sub-step: "
            f"{route.launches} launches after {step} complete steps "
            f"(expected {launches_per_step} per step)")
        if found:
            breach = {"step": step, "offenders": offenders,
                      "differing": bad,
                      "first_divergence": row["first_divergence"],
                      "confinement": confinement(got, expected, compared),
                      "clean_steps_before_breach": clean_steps}
            row["verdict"] = "REFUSED (precondition breach)"
            log(f"[{leg}] {case:<24} PRECONDITION BREACH at step {step}: "
                f"{found} band words in "
                f"{[o['volume'] for o in offenders]}; bytes "
                f"{'DIVERGED ' + str(bad) if bad else 'still agree'}")
            break
        # A divergence with a CLEAN census is a defect, not a refusal.
        assert not bad, (
            f"FIRST DIVERGENCE (census clean) leg={leg} case={case} step={step}: "
            f"{row['first_divergence']} (per-array counts {bad})")
        clean_steps = step
        previous = expected

    post_breach: List[Dict[str, Any]] = []
    if breach is not None:
        # OBSERVATION ONLY, no assertions: how does the flushed cell propagate?
        for extra in range(1, POST_BREACH_STEPS + 1):
            step = breach["step"] + extra
            excite(step)
            oracle_step(oracle, pml_o)
            route.driver_step()
            got = gate.snapshot(fields)
            expected = gate.snapshot(oracle)
            names = tuple(expected)
            bad = per_name_differing(got, expected, names)
            spread = confinement(got, expected, names)
            post_breach.append({
                "step": step, "differing_words": sum(bad.values()),
                "arrays_touched": sorted(bad),
                "flushed_band_words": spread["flushed_band_words"],
                "unexplained_words": spread["unexplained_words"],
                "largest_absolute_difference":
                    spread["largest_absolute_difference"]})
            log(f"[{leg}] {case:<24} post-breach step {step} "
                f"differing={sum(bad.values())} arrays={len(bad)} "
                f"max|delta|={spread['largest_absolute_difference']:.3e} "
                f"unexplained={spread['unexplained_words']}")
        breach["post_breach_observation"] = post_breach

    summary = {"leg": leg, "case": case, "steps_requested": steps,
               "steps_completed_clean": clean_steps,
               "arrays_per_step": len(gate.STATE),
               "steps_with_f_bfast_motion": steps_with_state_move,
               "route": "composed (plan_step, every sub-step on the device)"
                        if composed else
                        "per-sub-step (curls on the device, constitutive on the "
                        "array path)",
               "launches_per_complete_driver_step": launches_per_step,
               "sub_steps_on_the_device": list(getattr(route, "order", SUB_STEPS)),
               "selected_arm_by_slot": dict(getattr(route, "selected", {})),
               "residency_verdict_held": (
                   None if not composed else
                   bool(route.residency_verdict
                        and route.residency_verdict.covered)),
               "launches": route.launches,
               "stale_words_covered_by_sync_in": route.stale_words,
               "peak_subnormal_words": peak_subnormal,
               "censused_words": int(window.observed_words),
               "smallest_nonzero_abs": None if smallest == float("inf") else smallest,
               "precondition_breach": breach,
               "verdict": ("byte-identical for every complete driver step"
                           if breach is None else
                           f"REFUSED from step {breach['step']}: the "
                           f"subnormal-free precondition is FALSE there"),
               "final_f_bfast_nonzero_words": int(sum(
                   np.count_nonzero(getattr(fields, n)) for n in state_names))}
    assert not window.vacuity_reasons(), window.vacuity_reasons()
    ran = clean_steps if breach is None else breach["step"]
    if use_bfast:
        assert steps_with_state_move >= STATE_MOVE_FRACTION * ran, (
            f"VACUOUS IIR: {case} moved f_bfast on only {steps_with_state_move} of "
            f"{ran} steps, below the declared floor")
    else:
        # The attribution control has no tail and therefore no IIR state at all;
        # its non-vacuity is that the absorber's own volumes moved every step,
        # which the per-step `moved > 0` assertion above already carries.
        assert not state_names, (
            "the attribution control allocated f_bfast storage, so its grid did "
            "activate BFAST after all")
    # THE STALENESS FLOOR IS ROUTE-DEPENDENT, and discovering that was a measurement
    # rather than a guess: the composed route's first undriven case tripped a floor
    # inherited from the per-sub-step route.
    #
    #   * PER-SUB-STEP ROUTE — the array path writes the host arrays between the two
    #     curl launches, so the mirrors MUST go stale. A run where they did not is a
    #     run that never tested them, and the floor stands.
    #   * COMPOSED ROUTE, DRIVEN — nothing inside the step touches the host, but the
    #     source seam at the top of each step does, so the Ez mirror is stale on every
    #     sync_in. The floor stands for the same reason, on a different seam.
    #   * COMPOSED ROUTE, UNDRIVEN — nothing whatsoever writes a host array, so
    #     ZERO stale words is the CORRECT answer and the one that licenses holding
    #     the mirror set across a complete step. Asserting equality is the stronger
    #     statement than waiving the check: a nonzero count here would mean something
    #     unaccounted for is writing host memory mid-step.
    if composed and not driven:
        assert route.stale_words == 0, (
            f"{case}: the COMPOSED route found {route.stale_words} stale words with "
            f"no host writer in the loop — something outside the four planned "
            f"sub-steps is writing host memory, and the residency verdict that "
            f"licenses holding the mirrors would be wrong")
        assert route.residency.syncs_in >= ran and \
            route.residency.syncs_out >= ran, (
                f"{case}: {route.residency.syncs_in} sync_in / "
                f"{route.residency.syncs_out} sync_out over {ran} compared steps — "
                f"the mirrors were not round-tripped, so a zero stale count is "
                f"vacuous")
    else:
        assert route.stale_words > 0, (
            f"{case}: sync_in covered NOTHING — the device mirrors never went stale "
            f"between launches, so this run never tested them")
    log(f"[{leg}] {case:<24} SUMMARY clean_steps={clean_steps}/{steps} "
        f"launches={route.launches} f_bfast_moved_on={steps_with_state_move} "
        f"stale_words_covered={route.stale_words} "
        f"peak_subnormal={peak_subnormal} censused={window.observed_words} "
        f"-> {summary['verdict']}")
    return summary


def leg_seeded_long(payload: Dict[str, Any], out: str) -> None:
    """A seeded lattice: ``-2*state`` is live from step 1 and never decays."""
    rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    started = time.time()
    for name, cell, boundaries in CONFIGS:
        for k_label, bfast_k in BFAST_KS:
            summaries.append(run_horizon(
                payload, out, "seeded_long", f"{name}/{k_label}", cell,
                boundaries, bfast_k, SEEDED_STEPS, seeded=True, driven=False,
                rows=rows, started=started))
            payload["seeded_long_summaries"] = summaries
            save(payload, out)


def leg_driven_long(payload: Dict[str, Any], out: str) -> None:
    """A real pulsed run from rest, and the source-free control that makes it a run."""
    rows: List[Dict[str, Any]] = []
    started = time.time()
    name, cell, boundaries = CONFIGS[1]
    summary = run_horizon(payload, out, "driven_long", f"{name}/marquee_x",
                          cell, boundaries, BFAST_KS[0][1], DRIVEN_STEPS,
                          seeded=False, driven=True, rows=rows, started=started)

    # THE SOURCE-FREE CONTROL, on the DEVICE route. From rest with no drive the
    # stepper is linear and homogeneous, so every volume must stay identically zero
    # — including f_bfast, whose marginally stable pole would keep any injected
    # noise alive forever. Without this the driven leg's agreement is consistent
    # with a run that was never excited, which agrees with itself trivially.
    _, control_fields, control_pml, _, _, _ = build_pair(
        "control", cell, boundaries, COURANTS[0], BFAST_KS[0][1], seeded=False)
    control = EngineRoute(control_fields, control_pml)
    for _ in range(DRIVEN_STEPS):
        control.driver_step()
    control_nonzero = {n: int(np.count_nonzero(getattr(control_fields, n)))
                       for n in gate.STATE
                       if getattr(control_fields, n, None) is not None}
    control_total = sum(control_nonzero.values())
    summary["source_free_control"] = {
        "steps": DRIVEN_STEPS, "launches": control.launches,
        "nonzero_words": control_total,
        "nonzero_by_array": {n: c for n, c in control_nonzero.items() if c},
        "driven_f_bfast_nonzero_words": summary["final_f_bfast_nonzero_words"]}
    log(f"[driven_long] source-free control steps={DRIVEN_STEPS} "
        f"launches={control.launches} nonzero_words={control_total} "
        f"driven_f_bfast_nonzero={summary['final_f_bfast_nonzero_words']}")
    payload["driven_long_summary"] = summary
    save(payload, out)
    assert control_total == 0, (
        f"the SOURCE-FREE control did not stay at rest ({control_total} nonzero "
        f"words): the driven run's agreement cannot be attributed to the source")
    assert summary["final_f_bfast_nonzero_words"] > 0, (
        "VACUOUS drive: the driven run left f_bfast identically zero")


def leg_composed_long(payload: Dict[str, Any], out: str) -> None:
    """The SHIPPED COMPOSER's whole-step route: all four sub-steps on the device.

    THE GAP THIS CLOSES was opened by the tranche-2 wiring and by nothing else. Until
    the family registered arms, ``plan_step`` left both curl slots unselected on a
    BFAST run and the array path stepped everything; the honest measurement then was
    the per-sub-step route, which is what :func:`leg_seeded_long` and
    :func:`leg_driven_long` carry. The family now fills all four slots, so what a
    caller of ``plan_step`` actually runs is a complete driver step with no array-path
    wall inside it — a different composition, with the IIR states and the PML
    auxiliaries crossing sub-step boundaries in DEVICE memory. That composition had no
    byte comparison anywhere in this family's trio.

    SEEDED AND DRIVEN, for the reason the other legs are: a seeded lattice makes
    ``-2*state`` live from step 1, and a driven run from rest carries the values a
    real simulation carries. The driven case takes the SOURCE-FREE control, because a
    run that was never excited agrees with itself trivially.

    THE CENSUS IS THE SAME CENSUS and the refusal rule is the same rule: a band word
    anywhere ends the claim for that case at that step and is recorded as REFUSED; a
    divergence with a CLEAN census raises, because on this composition as on the other
    there is no configuration in which the device may leave ``stepping.py``.
    """
    rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    started = time.time()

    name, cell, boundaries = CONFIGS[1]
    summaries.append(run_horizon(
        payload, out, "composed_long", f"{name}/marquee_x/seeded", cell,
        boundaries, BFAST_KS[0][1], SEEDED_STEPS, seeded=True, driven=False,
        rows=rows, started=started, composed=True))
    payload["composed_long_summaries"] = summaries
    save(payload, out)

    driven = run_horizon(payload, out, "composed_long",
                         f"{name}/marquee_x/driven", cell, boundaries,
                         BFAST_KS[0][1], DRIVEN_STEPS, seeded=False, driven=True,
                         rows=rows, started=started, composed=True)

    # THE SOURCE-FREE CONTROL, on the COMPOSED route. Same argument as the driven
    # leg's: from rest with no drive the stepper is linear and homogeneous, so every
    # volume must stay identically zero — and the constitutive kernel is now in the
    # loop, so this control covers it too.
    _, control_fields, control_pml, _, _, _ = build_pair(
        "composed_control", cell, boundaries, COURANTS[0], BFAST_KS[0][1],
        seeded=False)
    control = ComposedRoute(control_fields, control_pml)
    for _ in range(DRIVEN_STEPS):
        control.driver_step()
    control_nonzero = {n: int(np.count_nonzero(getattr(control_fields, n)))
                       for n in gate.STATE
                       if getattr(control_fields, n, None) is not None}
    control_total = sum(control_nonzero.values())
    driven["source_free_control"] = {
        "steps": DRIVEN_STEPS, "launches": control.launches,
        "nonzero_words": control_total,
        "nonzero_by_array": {n: c for n, c in control_nonzero.items() if c},
        "driven_f_bfast_nonzero_words": driven["final_f_bfast_nonzero_words"]}
    summaries.append(driven)
    payload["composed_long_summaries"] = summaries
    save(payload, out)
    log(f"[composed_long] source-free control steps={DRIVEN_STEPS} "
        f"launches={control.launches} nonzero_words={control_total} "
        f"driven_f_bfast_nonzero={driven['final_f_bfast_nonzero_words']}")
    assert control_total == 0, (
        f"the SOURCE-FREE control did not stay at rest ({control_total} nonzero "
        f"words): the driven run's agreement cannot be attributed to the source")
    assert driven["final_f_bfast_nonzero_words"] > 0, (
        "VACUOUS drive: the driven composed run left f_bfast identically zero")
    # A composed route that selected a foreign arm would still be a route, and it
    # would certify somebody else's product under this family's name.
    for summary in summaries:
        assert set(summary["selected_arm_by_slot"]) == {
            "step_B", "update_H", "step_D", "update_E"}, summary
        assert set(summary["selected_arm_by_slot"].values()) == {"BFAST"}, (
            f"the composer selected a foreign arm on the composed route: "
            f"{summary['selected_arm_by_slot']}")


# ---------------------------------------------------------------------------
# LEG 5 — arming: each comparison above must be able to fail
# ---------------------------------------------------------------------------

def leg_arming(payload: Dict[str, Any], out: str) -> None:
    """Three planted defects, each of which MUST make the comparison diverge.

    A probe that cannot fail certifies nothing. Each arm is launch-counted, and a
    defect that produced no divergence is reported as UNARMED rather than passed.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    name, cell, boundaries = CONFIGS[1]

    # ARM A — the engine route's own scalar computation, wrong side. This is the
    # cross-assignment stepping.py calls the easiest mistake in the pass, planted
    # where the ENGINE route computes it rather than where the gate hands it in.
    original = bfast.bfast_curl_coefficients
    try:
        bfast.bfast_curl_coefficients = (  # type: ignore[assignment]
            lambda k, invariant, magnetic: original(k, invariant, not magnetic))
        grid, fields, pml, _, oracle, pml_o = build_pair(
            name, cell, boundaries, COURANTS[0], BFAST_KS[1][1], seeded=True)
        stepping.step_B(oracle, pml_o)
        residency = device.Residency()
        plan = bfast.plan_bfast_pml_curl(fields, pml, "step_B", residency)
        plan.run()
        residency.sync_out()
        compared = ("Bx", "By", "Bz", "fu_Bx", "fu_By", "fu_Bz") + STATES["step_B"]
        bad = per_name_differing(gate.snapshot(fields), gate.snapshot(oracle),
                                 compared)
        rows.append({"arm": "a_engine_scalars_from_the_other_side",
                     "launches": plan.launches, "diverged": bool(bad),
                     "differing": bad,
                     "why": "the six scalars computed for the wrong sub-step side; "
                            "the D side negates both, so this is a sign flip on "
                            "every live term"})
    finally:
        bfast.bfast_curl_coefficients = original  # type: ignore[assignment]
    log(f"[leg5 arming] a_engine_scalars_from_the_other_side "
        f"launches={rows[-1]['launches']} diverged={rows[-1]['diverged']} "
        f"({time.time() - started:.1f}s)")
    payload["legs"]["arming"] = rows
    save(payload, out)
    assert rows[-1]["launches"] == 1, rows[-1]
    assert rows[-1]["diverged"], (
        "UNARMED: the engine route computed its scalars for the WRONG SIDE and the "
        "bytes still agreed, so leg 2 is not testing the scalar path")

    # ARM B — the stale mirror. The plan is launched a second time WITHOUT syncing
    # the host writes the array path made in between. This is the residency
    # invariant, and if skipping the sync changed nothing the mirrors would not be
    # carrying state at all.
    grid, fields, pml, _, oracle, pml_o = build_pair(
        name, cell, boundaries, COURANTS[0], BFAST_KS[1][1], seeded=True)
    stale = EngineRoute(fields, pml, sync=False)
    oracle_step(oracle, pml_o)
    stale.driver_step()
    bad = per_name_differing(gate.snapshot(fields), gate.snapshot(oracle),
                             gate.STATE)
    rows.append({"arm": "b_sync_in_skipped", "launches": stale.launches,
                 "diverged": bool(bad), "differing": bad,
                 "why": "KernelPlan.run does not sync; between the two curl "
                        "launches the array path writes the host arrays, so a "
                        "loop that skips sync_in reads the previous step's bytes"})
    log(f"[leg5 arming] b_sync_in_skipped launches={stale.launches} "
        f"diverged={bool(bad)} arrays={len(bad)} ({time.time() - started:.1f}s)")
    payload["legs"]["arming"] = rows
    save(payload, out)
    assert stale.launches == 2, rows[-1]
    assert bad, (
        "UNARMED: skipping sync_in changed nothing, so the long legs' agreement "
        "says nothing about the device mirrors")

    # ARM C — the unlaunched control. The comparison must fail when no kernel runs,
    # or "identical" was a statement about untouched inputs.
    grid, fields, pml, _, oracle, pml_o = build_pair(
        name, cell, boundaries, COURANTS[0], BFAST_KS[1][1], seeded=True)
    stepping.step_B(oracle, pml_o)
    residency = device.Residency()
    plan = bfast.plan_bfast_pml_curl(fields, pml, "step_B", residency)
    compared = ("Bx", "By", "Bz", "fu_Bx", "fu_By", "fu_Bz") + STATES["step_B"]
    bad = per_name_differing(gate.snapshot(fields), gate.snapshot(oracle),
                             compared)
    rows.append({"arm": "c_never_launched", "launches": plan.launches,
                 "diverged": bool(bad), "differing": bad,
                 "why": "a plan built and never run leaves the inputs untouched; "
                        "the comparison must see that"})
    log(f"[leg5 arming] c_never_launched launches={plan.launches} "
        f"diverged={bool(bad)} ({time.time() - started:.1f}s)")
    payload["legs"]["arming"] = rows
    save(payload, out)
    assert plan.launches == 0, rows[-1]
    assert bad, "UNARMED: an unlaunched plan compared identical to a stepped oracle"

    # ARM D — the COMPOSED route's sub-step ORDER. The four plans are launched in a
    # permuted order (the constitutive pair before its own curl), which is the one
    # defect a whole-step composition can carry that a per-sub-step route cannot:
    # each individual plan is correct and the composition is not. If this did not
    # diverge, the composed leg's agreement would be a statement about four plans
    # that happen to commute, which they do not.
    grid, fields, pml, _, oracle, pml_o = build_pair(
        name, cell, boundaries, COURANTS[0], BFAST_KS[1][1], seeded=True)
    permuted = ComposedRoute(fields, pml)
    permuted.order = ("update_H", "step_B", "update_E", "step_D")
    oracle_step(oracle, pml_o)
    permuted.driver_step()
    bad = per_name_differing(gate.snapshot(fields), gate.snapshot(oracle),
                             gate.STATE)
    rows.append({"arm": "d_composed_sub_step_order_permuted",
                 "launches": permuted.launches, "diverged": bool(bad),
                 "differing": bad,
                 "why": "each plan is individually correct; the constitutive pair "
                        "run BEFORE its own curl reads the previous step's targets, "
                        "which is the composition-only defect"})
    log(f"[leg5 arming] d_composed_sub_step_order_permuted "
        f"launches={permuted.launches} diverged={bool(bad)} arrays={len(bad)} "
        f"({time.time() - started:.1f}s)")
    payload["legs"]["arming"] = rows
    save(payload, out)
    assert permuted.launches == 4, rows[-1]
    assert bad, (
        "UNARMED: permuting the composed route's sub-step order changed nothing, "
        "so the composed leg is not testing the composition")

    # ARM E — the COMPOSED route, never launched. Same argument as arm C, on the
    # route where it is easier to get a hollow pass: a composition that silently
    # planned nothing would leave the host arrays untouched and, on a seeded
    # lattice compared against a stepped oracle, must be visible.
    grid, fields, pml, _, oracle, pml_o = build_pair(
        name, cell, boundaries, COURANTS[0], BFAST_KS[1][1], seeded=True)
    idle = ComposedRoute(fields, pml)
    oracle_step(oracle, pml_o)
    bad = per_name_differing(gate.snapshot(fields), gate.snapshot(oracle),
                             gate.STATE)
    rows.append({"arm": "e_composed_never_launched", "launches": idle.launches,
                 "diverged": bool(bad), "differing": bad,
                 "slots_planned": list(idle.order),
                 "why": "a composed route built and never run leaves every host "
                        "array untouched; the comparison must see that"})
    log(f"[leg5 arming] e_composed_never_launched launches={idle.launches} "
        f"diverged={bool(bad)} ({time.time() - started:.1f}s)")
    payload["legs"]["arming"] = rows
    save(payload, out)
    assert idle.launches == 0, rows[-1]
    assert bad, (
        "UNARMED: an unlaunched composed route compared identical to a stepped "
        "oracle")


def leg_attribution(payload: Dict[str, Any], out: str) -> None:
    """Is a precondition breach a BFAST property, or the absorber recurrence's?

    THE QUESTION THE SEEDED LEG RAISES AND CANNOT ANSWER. If a long seeded run
    reaches the band, the next question is whether the BFAST tail put it there — the
    tail is a CANCELLATION (``advance = total - 2*state``) and cancellations are how
    small numbers are manufactured — or whether the split-field PML recurrence gets
    there on its own, in which case the finding belongs to the CERTIFIED tranche-1
    curl and this family merely happened to be the one looking.

    The control is the same lattice, the same seed, the same boundary triple, the
    same Courant and the same horizon, with ``bfast_scaled_k = (0, 0, 0)`` — so the
    grid does not activate BFAST at all — stepped through ``launch.plan_pml_curl``,
    the certified curl with no tail. Whatever it does is the absorber's own answer.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    name, cell, boundaries = CONFIGS[1]
    summary = run_horizon(payload, out, "attribution", f"{name}/no_bfast",
                          cell, boundaries, (0.0, 0.0, 0.0), SEEDED_STEPS,
                          seeded=True, driven=False, rows=rows, started=started,
                          use_bfast=False)
    payload["attribution_summary"] = summary
    save(payload, out)
    log(f"[attribution] the CERTIFIED tranche-1 curl on the same seeded lattice: "
        f"{summary['verdict']}")


LEGS = (
    ("binding", leg_binding),
    ("single", leg_single),
    ("seeded_long", leg_seeded_long),
    ("driven_long", leg_driven_long),
    ("composed_long", leg_composed_long),
    ("attribution", leg_attribution),
    ("arming", leg_arming),
)


def environment_stamp() -> Dict[str, Any]:
    import torch  # noqa: PLC0415

    return {"platform": platform.platform(), "machine": platform.machine(),
            "torch": torch.__version__, "numpy": np.__version__,
            "mps_available": bool(torch.backends.mps.is_available()),
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def provenance() -> Dict[str, str]:
    paths = {
        "stepping.py": os.path.join(API_ROOT, "meep_gpu", "stepping.py"),
        "metal_kernels/bfast_curl.py": os.path.join(
            API_ROOT, "meep_gpu", "metal_kernels", "bfast_curl.py"),
        "metal_kernels/device.py": os.path.join(
            API_ROOT, "meep_gpu", "metal_kernels", "device.py"),
        "metal_kernels/plans.py": os.path.join(
            API_ROOT, "meep_gpu", "metal_kernels", "plans.py"),
        "gate_metal_bfast.py": os.path.join(HERE, "gate_metal_bfast.py"),
        "probe_metal_bfast_engine_route.py": os.path.abspath(__file__),
    }
    return {label: hashlib.sha256(open(path, "rb").read()).hexdigest()
            for label, path in paths.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--legs", default="")
    arguments = parser.parse_args()

    import torch  # noqa: PLC0415

    if not torch.backends.mps.is_available():
        log("CANNOT CERTIFY: no MPS device on this host")
        return 75

    os.makedirs(os.path.dirname(os.path.abspath(arguments.out)) or ".",
                exist_ok=True)
    started = time.time()
    payload: Dict[str, Any] = {
        "environment": environment_stamp(),
        "subnormal_policy": subnormal.mps_policy_report(),
        "provenance": provenance(),
        "step_budget": {"seeded_long_steps_per_case": SEEDED_STEPS,
                        "seeded_long_cases": len(CONFIGS) * len(BFAST_KS),
                        "driven_long_steps": DRIVEN_STEPS,
                        "composed_long_seeded_steps": SEEDED_STEPS,
                        "composed_long_driven_steps": DRIVEN_STEPS,
                        "composed_source_free_control_steps": DRIVEN_STEPS,
                        "source_free_control_steps": DRIVEN_STEPS,
                        "single_launch_cases": len(CONFIGS) * len(COURANTS)
                                               * len(BFAST_KS) * len(SUB_STEPS),
                        "courants": list(COURANTS),
                        "note": "every step here is a COMPLETE driver step "
                                "(step_B, update_H, step_D, update_E)"},
        "legs": {},
    }
    save(payload, arguments.out)
    log(f"[env] torch={payload['environment']['torch']} "
        f"policy={payload['subnormal_policy']['resolved']} "
        f"admitted={payload['subnormal_policy']['admitted']}")
    if not payload["subnormal_policy"]["admitted"]:
        log(f"CANNOT CERTIFY: {payload['subnormal_policy']['reasons']}")
        return 75

    wanted = tuple(n.strip() for n in arguments.legs.split(",") if n.strip())
    executed: List[str] = []
    for leg_name, leg in LEGS:
        if wanted and leg_name not in wanted:
            continue
        log(f"=== LEG {leg_name} ===")
        leg(payload, arguments.out)
        executed.append(leg_name)

    # THE SECOND, INDEPENDENT READ OF THE OUTCOME, from the rows that were WRITTEN
    # rather than from having reached this line. A REFUSED row (the census fired) is
    # counted separately from a DISAGREEING one (the bytes parted with a clean
    # census): the first bounds the claim, the second would be a defect.
    compared = sum(len(rows) for rows in payload["legs"].values()
                   if isinstance(rows, list))
    problems: List[str] = []
    refusals: List[str] = []
    for leg_name, rows in payload["legs"].items():
        if not isinstance(rows, list):
            continue
        for index, row in enumerate(rows):
            if leg_name == "arming":
                if not row.get("diverged"):
                    problems.append(f"arming[{index}]: {row.get('arm')} UNARMED")
                continue
            refused = str(row.get("verdict", "")).startswith("REFUSED")
            if refused:
                refusals.append(
                    f"{leg_name}[{index}]: {row.get('case')} step {row.get('step')} "
                    f"census fired ({row.get('subnormal_words')} band words) — "
                    f"differing={row.get('differing')}")
            elif row.get("differing"):
                problems.append(f"{leg_name}[{index}]: {row['differing']}")
    passed = compared > 0 and not problems and not refusals
    payload["summary"] = {
        "status": ("passed" if passed else
                   "VACUOUS (nothing compared)" if compared == 0 else
                   "FAILED (a divergence with a clean census)" if problems else
                   "REFUSED (the subnormal-free precondition is FALSE on at least "
                   "one case)"),
        "passed": passed, "rows_compared": compared,
        "disagreeing_rows": problems, "refused_rows": refusals,
        "legs_run": executed,
        "claim": ("BOTH SHIPPED ROUTES are byte-identical to stepping.py per "
                  "COMPLETE driver step over the full stored inventory, under a "
                  "CHECKED subnormal-free precondition: the per-sub-step route "
                  "(plan_bfast_pml_curl on live Fields/PML objects, constitutive "
                  "halves on the array path) and the COMPOSED route "
                  "(launch.plan_step, every sub-step on the device)"),
        "scope": ("real-field split-field PML curl with the BFAST second additive "
                  "pass and its three IIR states; three boundary triples; two "
                  "non-power-of-two Courants; seeded and driven-from-rest runs. "
                  "TWO ROUTES: the per-sub-step route runs the constitutive halves "
                  "on the array path (2 launches per complete driver step); the "
                  "composed route runs all four sub-steps on the device (4 "
                  "launches per step), the constitutive pair being the CERTIFIED "
                  "kernel admitted by this family's restated predicate — no new "
                  "arithmetic, a new admission"),
        "relationship_to_the_gate": (
            "COMPLEMENTARY, not a re-run: gate_metal_bfast certifies the "
            "from_arrays route over 4 complete driver steps; this probe certifies "
            "the ENGINE route — the one that reads mirrors off Fields/PML and "
            "computes its own scalars — over a long horizon"),
        "elapsed_s": round(time.time() - started, 1),
    }
    save(payload, arguments.out)
    for problem in problems:
        log(f"[summary] DISAGREEING {problem}")
    for refusal in refusals:
        log(f"[summary] REFUSED {refusal}")
    log(f"METAL BFAST ENGINE-ROUTE PROBE {payload['summary']['status']} "
        f"(rows={compared}, refusals={len(refusals)}) in "
        f"{payload['summary']['elapsed_s']}s -> {arguments.out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
