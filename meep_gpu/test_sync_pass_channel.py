"""The magnetic synchronization channel — ``fastpath.SYNC_PASS_OWNERS``.

``driver.synchronize_magnetic_fields`` runs ``step``'s magnetic half a second time,
for a flux or a field energy, and then UNDOES it. What it can undo is
``_SYNC_FIELDS`` + ``_SYNC_AUXILIARY``, which are magnetic names only. So a plan
dispatched inside that window may advance ``B``/``H`` and their auxiliaries — the
half-step puts those back — and may advance NOTHING else.

Every fused product shipped today stays inside one field type, so every one of them
is safe there. A product spanning ``update_H`` and ``step_D`` is not: it is correct
in ``step`` and, consulted unchanged in the half-step, it advances ``D`` and ``fu_D``
permanently on every ``flux_in_box`` and ``field_energy_in_box`` call.

This file measures four things rather than arguing them:

* THE PREMISE — ``D``, ``fu_D`` and ``f_cond_D`` really are in neither backup list,
  read off the driver rather than transcribed;
* THE DEFECT — an ``update_H``/``step_D`` product consulted the way the site
  consulted before this channel existed corrupts ``D`` through a flux call, byte
  measured against the array path, with the step it first diverges on reported;
* THE REMEDY — the same product through :data:`fastpath.SYNC_PASS_OWNERS` is refused
  on that path and the run stays byte-identical to the array path;
* THE INCUMBENT — a ``step_B``/``update_H`` product is answered EXACTLY as it was
  before, on both paths, over the same byte comparison. That is the control that
  makes the remedy a narrowing rather than a change.

The comparisons are complete driver steps, every stored volume, as uint32 words —
never ``allclose``, because ``-0.0 == 0.0`` lies.
"""

from __future__ import annotations

import ast
import pathlib
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import pytest

from . import driver as driver_module
from . import fastpath
from .driver import FdtdDriver
from .test_dispatch_contract import step_plan

PACKAGE = pathlib.Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# The premise: what the half-step can and cannot put back
# ---------------------------------------------------------------------------

def test_the_half_step_can_put_back_nothing_electric():
    """Read off the driver, because the whole channel rests on it.

    ``_SYNC_FIELDS`` is averaged and ``_SYNC_AUXILIARY`` is restored; anything in
    neither is left wherever the half-step put it. The three electric arrays a
    ``step_D`` half writes are ``D*``, ``fu_D*`` and ``f_cond_D*``, and none of them
    is in either list — which is the entire reason a product may have to decline the
    site.
    """
    backed_up = set(FdtdDriver._SYNC_FIELDS) | set(FdtdDriver._SYNC_AUXILIARY)
    for axis in "xyz":
        for name in (f"D{axis}", f"fu_D{axis}", f"f_cond_D{axis}"):
            assert name not in backed_up, (
                f"{name} is in the magnetic half-step's backup, which would make this "
                f"whole channel unnecessary — re-derive it before deleting anything")
    # And the converse, so the premise is not "we happened not to list D": every name
    # the half-step DOES carry is magnetic.
    for name in sorted(backed_up):
        assert ("B" in name) or name.startswith("f_w_H") or name.startswith("H"), name


def test_the_channel_names_one_site_and_it_is_the_last_magnetic_slot():
    """``update_H`` is where a product can first reach forward into the electric half."""
    assert fastpath.SYNC_PASS_OWNERS == {
        fastpath.SYNC_UPDATE_H_PASS: "update_H"}
    assert fastpath.SYNC_UPDATE_H_PASS not in fastpath.DRIVER_SLOTS, (
        "the sync pass must NOT be a slot name, or an adapter carrying the slot "
        "vocabulary would answer it by accident — which is the whole hazard")
    assert set(fastpath.SYNC_PATH_SLOTS) < set(fastpath.DRIVER_SLOTS)
    assert fastpath.SYNC_PATH_SLOTS == ("step_B", "fill_B", "update_H"), (
        "these are the driver-slot consults synchronize_magnetic_fields makes; a "
        "change here is a change to what a product may do inside the half-step")


def test_the_driver_consults_the_sync_name_only_on_the_sync_path():
    """``step`` asks for ``update_H``; the half-step asks for the pass name.

    Read out of the source rather than by driving, so a site that stopped consulting
    at all would fail here as loudly as one that consulted the wrong name.
    """
    tree = ast.parse((PACKAGE / "driver.py").read_text(encoding="utf-8"))
    asked: Dict[str, list] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in ("step", "synchronize_magnetic_fields"):
            continue
        names = []
        for inner in ast.walk(node):
            if (isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Attribute)
                    and inner.func.attr == "dispatch"):
                first = inner.args[0]
                if isinstance(first, ast.Constant):
                    names.append(first.value)
                elif isinstance(first, ast.Name):
                    names.append(f"<{first.id}>")
        asked[node.name] = names
    assert asked["step"] == [
        "step_B", "fill_B", "fill_folded_far_ghosts_B", "update_H",
        "step_D", "fill_D", "fill_folded_far_ghosts_D", "update_E", "update_P"]
    assert asked["synchronize_magnetic_fields"] == [
        "step_B", "fill_B", "fill_folded_far_ghosts_B", "<SYNC_UPDATE_H_PASS>"], (
        "the half-step's update_H consult must go through the by-name channel; the "
        "other three are its own magnetic passes and are asked by their own names")


# ---------------------------------------------------------------------------
# The seam tables, and why one name is enough TODAY
# ---------------------------------------------------------------------------

SEAM_TABLES = {
    "cuda": ("cuda_kernels/fused_pairs.py", "FUSED_PAIR_SEAMS"),
    "metal": ("metal_kernels/launch.py", "FUSED_PAIR_SEAMS"),
    "triton": ("triton_kernels/launch.py", "CERTIFIED_FUSED_PAIR_SEAMS"),
}


def _seams(relative: str, table: str) -> Dict[str, str]:
    """``curl slot -> constitutive slot`` read from SOURCE, never by import.

    By source because this file must run on a host with no Triton, no Metal and no
    CuPy — the same reason ``test_fused_pair_deposit_wiring`` reads its allowlist
    that way.
    """
    tree = ast.parse((PACKAGE / relative).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        target = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target = node.target.id
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(
                node.targets[0], ast.Name):
            target = node.targets[0].id
        if target != table or not isinstance(node.value, ast.Dict):
            continue
        out = {}
        for key, value in zip(node.value.keys, node.value.values):
            assert isinstance(key, ast.Constant), ast.dump(key)
            assert isinstance(value, ast.Tuple), ast.dump(value)
            first = value.elts[0]
            assert isinstance(first, ast.Constant), ast.dump(first)
            out[key.value] = first.value
        return out
    raise AssertionError(f"{relative} has no {table}")


@pytest.mark.parametrize("backend", sorted(SEAM_TABLES))
def test_every_seam_touching_the_half_step_is_contained_or_carries_a_name(backend):
    """THE TRIPWIRE — one sync name is enough only while this holds.

    A seam whose slots are all inside :data:`fastpath.SYNC_PATH_SLOTS` is safe: the
    half-step runs every one of them. A seam that reaches OUT of them is safe only if
    the half-step consults its leading slot through the by-name channel, because that
    is the consult that can decline. Everything else — a four-slot ``step_B``..
    ``update_E`` weld is the live example — would be dispatched inside the half-step
    under a plain slot name and would advance electric state nothing can restore.

    So this test fails the day such a product's seam is declared, and the fix is a
    name for the site it leads, in the same change. It is deliberately about the
    SEAM TABLE and not about what is installed: a seam is declared before a product
    exists, which is the moment the name is cheap.
    """
    relative, table = SEAM_TABLES[backend]
    inside = set(fastpath.SYNC_PATH_SLOTS)
    named = set(fastpath.SYNC_PASS_OWNERS.values())
    offenders = []
    for curl, update in _seams(relative, table).items():
        span = {curl, update}
        if span <= inside or not (span & inside):
            continue
        if curl in named:
            continue
        offenders.append((curl, update))
    assert not offenders, (
        f"{relative}: {offenders} span the magnetic half-step's consults and out of "
        f"them, at a slot the half-step asks for by its plain name. Give that slot a "
        f"row in fastpath.SYNC_PASS_OWNERS and a consult site in "
        f"driver.synchronize_magnetic_fields, or the half-step dispatches it")


# ---------------------------------------------------------------------------
# The measurement
# ---------------------------------------------------------------------------

def _walls(dimensions: int) -> Mapping[str, str]:
    live = {1: ("z",), 2: ("x", "y"), 3: ("x", "y", "z")}[dimensions]
    return {axis: "metallic" for axis in live}


def _volumes(driver: FdtdDriver, node: Any = None, prefix: str = "",
             out: Dict[str, np.ndarray] = None, depth: int = 0) -> Dict[str, np.ndarray]:
    """Every stored array on the fields and the PML, as uint32 words.

    Enumerated rather than listed, so an array this file has never heard of is still
    compared. uint32 because that is the spelling every byte-identity claim in this
    tree is made in; uint8 for anything whose itemsize does not divide four.
    """
    if out is None:
        out = {}
        for owner, name in ((driver.fields, "fields"), (getattr(driver, "pml", None), "pml")):
            if owner is not None:
                _volumes(driver, owner, name, out, 0)
        return out
    if depth > 3:
        return out
    if isinstance(node, np.ndarray):
        # A COPY, not a view. A snapshot taken as a view aliases the live buffer, so
        # every step of the run reads back as the last one and a per-step comparison
        # silently becomes an end-of-run comparison.
        flat = np.array(node, copy=True).reshape(-1).view(np.uint8)
        out[prefix] = flat.view(np.uint32) if flat.size % 4 == 0 else flat
        return out
    if isinstance(node, Mapping):
        for key, value in node.items():
            _volumes(driver, value, f"{prefix}[{key}]", out, depth + 1)
        return out
    if isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            _volumes(driver, value, f"{prefix}[{index}]", out, depth + 1)
        return out
    fields = getattr(node, "__dict__", None)
    if fields:
        for key, value in sorted(fields.items()):
            if key.startswith("__"):
                continue
            _volumes(driver, value, f"{prefix}.{key}", out, depth + 1)
    return out


def _differing(left: Mapping[str, np.ndarray],
               right: Mapping[str, np.ndarray]) -> Dict[str, int]:
    assert set(left) == set(right), set(left) ^ set(right)
    out = {}
    for name in sorted(left):
        a, b = left[name], right[name]
        assert a.shape == b.shape and a.dtype == b.dtype, name
        count = int(np.count_nonzero(a != b))
        if count:
            out[name] = count
    return out


class SpanningPlan:
    """A product that performs BOTH halves of its seam at the leading consult.

    The shipped shape: one launch does the curl and the constitutive, the absorbed
    slot holds a sentinel. Here the two halves are the driver's own array functions,
    so the composition is arithmetically the array path and any divergence measured
    against it is a divergence of ORDER — which is exactly what this file is about.
    """

    def __init__(self, driver: FdtdDriver, halves: Tuple[str, ...]):
        self._driver = driver
        self._halves = halves
        self.runs = 0

    def run(self) -> None:
        self.runs += 1
        for half in self._halves:
            pass_ = getattr(driver_module, half)
            # The wall clear takes no PML — it is a boundary pass, not a sub-step.
            if half.startswith("zero_metal"):
                pass_(self._driver.fields)
            else:
                pass_(self._driver.fields, self._driver.pml)


class Sentinel:
    """The absorbed slot. ``absorbed_by`` is what groups it with its leading plan."""

    def __init__(self, leader: Any):
        self.absorbed_by = leader
        self.runs = 0

    def run(self) -> None:
        self.runs += 1


class TranslatingAdapter:
    """The consult site as it stood BEFORE this channel: the sync name means the slot.

    Not a stub of the plan — it wraps the REAL :class:`fastpath.FastPathPlan` and
    changes one thing, the name the half-step's fourth consult resolves to. That
    makes the two legs differ in the site and in nothing else.
    """

    def __init__(self, plan: Any):
        self.plan = plan

    def dispatch(self, slot: str, fields: Any) -> bool:
        if slot == fastpath.SYNC_UPDATE_H_PASS:
            slot = fastpath.SYNC_PASS_OWNERS[slot]
        return self.plan.dispatch(slot, fields)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.plan, name)


class SlotOnlyAdapter:
    """A shim that carries the slot vocabulary and nothing else — the third implementer.

    ``probe_triton_folded_deposit_closure``, ``probe_metal_complex_driver_step``,
    ``probe_metal_special_kz_driver_step`` and a FROZEN copy inside a gate's result
    directory all implement this consult as one method over the names they know.
    This is that shape: an unknown name is False, which is the array path.
    """

    def __init__(self, plan: Any):
        self.plan = plan
        self.refused: list = []

    def dispatch(self, slot: str, fields: Any) -> bool:
        if slot not in fastpath.DRIVER_SLOTS:
            self.refused.append(slot)
            return False
        return self.plan.dispatch(slot, fields)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.plan, name)


def _driver() -> FdtdDriver:
    built = FdtdDriver(cell_size=(1.2, 1.2, 0.0), resolution=20, dimensions=2,
                       boundaries=_walls(2), force_complex_fields=False)
    built.setup_pml({"x": 3, "y": 3})
    generator = np.random.default_rng(20260904)
    for component in ("Dx", "Dy", "Dz", "Bx", "By", "Bz"):
        built.set_field(component,
                        generator.standard_normal(built.grid.shape).astype(np.float32))
    return built


def _install(monkeypatch, built: FdtdDriver, plans: Mapping[str, Any],
             wrapper=None) -> Any:
    plan = fastpath.FastPathPlan(
        fields=built.fields,
        step_plan=step_plan(plans, {slot: "PML" for slot in plans}),
        slots=tuple(name for name in fastpath.DRIVER_SLOTS if name in plans),
        arms={slot: "PML" for slot in plans},
        dropped_null={}, unwarmed={},
        record={"step_path": "fused", "decision": "dispatched", "slots": {}},
    )
    installed = plan if wrapper is None else wrapper(plan)
    built.gpu = "metal"  # a GPU driver's freeze consults the planner
    monkeypatch.setattr(driver_module, "plan_fast_path", lambda *args: installed)
    return installed


#: How many complete steps each leg runs, and which one takes the flux call. Short
#: on purpose: the divergence this file is about appears on the FIRST step after the
#: flux call or not at all, and a long run would only report it later.
STEPS = 6
FLUX_AT = 3


def _run_leg(build_plans, wrapper, monkeypatch) -> Tuple[list, float]:
    """``STEPS`` complete steps with one ``flux_in_box`` at :data:`FLUX_AT`.

    Returns the per-step volume snapshots and the flux, so the comparison is per
    complete driver step rather than only at the end.
    """
    built = _driver()
    if build_plans is not None:
        _install(monkeypatch, built, build_plans(built), wrapper)
    else:
        monkeypatch.setattr(driver_module, "plan_fast_path", lambda *args: None)
    flux = float("nan")
    snapshots = []
    try:
        for index in range(STEPS):
            built.step()
            if index == FLUX_AT:
                centre, extent = built.total_volume()
                flux = built.flux_in_box(0, centre, extent)
            snapshots.append(_volumes(built))
    finally:
        built.close()
    return snapshots, flux


def _first_divergent_step(left: list, right: list) -> Tuple[int, Dict[str, int]]:
    for index, (a, b) in enumerate(zip(left, right)):
        differing = _differing(a, b)
        if differing:
            return index, differing
    return -1, {}


#: The incumbent's span, INCLUDING the wall clear the real product performs
#: in-launch. The driver runs ``zero_metal_B`` between the curl and the constitutive
#: and behind no consult, so a stub that skipped it would compute ``H`` from an
#: uncleared ``B`` and would diverge from the array path for a reason that has
#: nothing to do with this file. The driver's own later clear is idempotent and its
#: two fills are inert on this fixture (no symmetry, no folded axis), which is why
#: the composition lands byte-identical — measured, not assumed, by the leg below.
MAGNETIC_SPAN = ("step_B", "zero_metal_B", "update_H")

#: The hazard's span. Nothing is needed between the two halves: ``update_H`` is the
#: last magnetic pass and ``zero_metal_D`` runs AFTER ``step_D`` on both paths.
H_TO_D_SPAN = ("update_H", "step_D")


def _magnetic_pair(built: FdtdDriver) -> Dict[str, Any]:
    leader = SpanningPlan(built, MAGNETIC_SPAN)
    return {"step_B": leader, "update_H": Sentinel(leader)}


def _h_to_d_pair(built: FdtdDriver) -> Dict[str, Any]:
    leader = SpanningPlan(built, H_TO_D_SPAN)
    return {"update_H": leader, "step_D": Sentinel(leader)}


def test_the_incumbent_magnetic_pair_is_answered_exactly_as_before(monkeypatch):
    """THE CONTROL. A ``step_B``/``update_H`` product must not notice this change.

    Three legs from one seed: the array path, the product through the shipped consult,
    and the product through the consult as it stood before. All three byte-identical
    at every one of :data:`STEPS` complete steps, with a ``flux_in_box`` — which is a
    ``synchronize_magnetic_fields`` and a ``restore`` — inside the run, and the same
    flux number out of all three.
    """
    array, array_flux = _run_leg(None, None, monkeypatch)
    shipped, shipped_flux = _run_leg(_magnetic_pair, None, monkeypatch)
    before, before_flux = _run_leg(_magnetic_pair, TranslatingAdapter, monkeypatch)
    assert _first_divergent_step(array, shipped) == (-1, {}), (
        "the incumbent pair diverged from the array path through the new consult")
    assert _first_divergent_step(shipped, before) == (-1, {}), (
        "the new consult and the old one answered the incumbent differently")
    assert shipped_flux == array_flux == before_flux
    # AND THE COMPARISON IS NOT VACUOUS. An identity claim over nothing is worth
    # nothing, so the denominator is asserted and reported: every stored volume on
    # the fields and the PML, at each of STEPS complete steps.
    arrays = len(array[0])
    words = sum(sum(v.size for v in snapshot.values()) for snapshot in array)
    assert arrays >= 30 and words >= 100_000, (
        f"only {arrays} arrays / {words} words were compared per leg")
    print(f"incumbent control: {arrays} arrays, {words} uint32 words per leg, "
          f"{STEPS} complete steps, 3 legs", flush=True)


def test_the_incumbent_pair_still_dispatches_inside_the_half_step(monkeypatch):
    """The control above would also pass if the pair stopped dispatching entirely.

    So count the launches. ``step`` runs the leading consult once per step and the
    half-step runs it again, which is 7 for six steps and one flux call — and the
    sentinel answers exactly as often, off the same commitment.
    """
    built = _driver()
    leader = SpanningPlan(built, MAGNETIC_SPAN)
    sentinel = Sentinel(leader)
    plan = _install(monkeypatch, built, {"step_B": leader, "update_H": sentinel})
    try:
        for index in range(STEPS):
            built.step()
            if index == FLUX_AT:
                centre, extent = built.total_volume()
                built.flux_in_box(0, centre, extent)
    finally:
        built.close()
    assert leader.runs == STEPS + 1, "the half-step's own launch is missing"
    assert sentinel.runs == STEPS + 1
    assert plan.sync_refusals == {}, (
        "a pair inside the magnetic half-step must never be refused there")


def test_a_product_spanning_into_the_electric_half_corrupts_D_without_the_channel(
        monkeypatch):
    """THE DEFECT, EXECUTED. This is the citation, not the argument.

    The same ``update_H``/``step_D`` product on two legs that differ only in the name
    the half-step's fourth consult spells. Through the OLD name it advances ``D`` and
    ``fu_D`` inside a window whose restore reaches neither, and the run is wrong from
    the flux call onwards. Through the channel it is refused there, the array
    ``update_H`` runs, and the run is byte-identical to the array path.
    """
    array, array_flux = _run_leg(None, None, monkeypatch)
    shipped, shipped_flux = _run_leg(_h_to_d_pair, None, monkeypatch)
    before, before_flux = _run_leg(_h_to_d_pair, TranslatingAdapter, monkeypatch)

    assert _first_divergent_step(array, shipped) == (-1, {}), (
        "the channel did not hold the product out of the magnetic half-step")
    assert shipped_flux == array_flux

    step, differing = _first_divergent_step(array, before)
    assert step == FLUX_AT, (
        f"the old consult was expected to diverge on the step carrying the flux call "
        f"({FLUX_AT}); it diverged at {step} with {differing}")
    electric = {name: count for name, count in differing.items()
                if ".D" in name or ".fu_D" in name or ".f_cond_D" in name}
    print(f"H_to_D through the old consult: first divergent complete step {step}, "
          f"{len(differing)} arrays differing, electric={electric}", flush=True)
    assert electric, (
        f"the divergence must be in the electric state the half-step cannot restore; "
        f"it was {sorted(differing)}")
    assert before_flux != array_flux or step == FLUX_AT
    # The step BEFORE the flux call must be clean, or the legs differ for some other
    # reason and this measurement is about something else.
    assert not _differing(array[FLUX_AT - 1], before[FLUX_AT - 1])


def test_the_refusal_is_counted_and_named(monkeypatch):
    """A refusal nobody can read is a silent fallback. ``sync_refusals`` is the record."""
    built = _driver()
    leader = SpanningPlan(built, H_TO_D_SPAN)
    plan = _install(monkeypatch, built,
                    {"update_H": leader, "step_D": Sentinel(leader)})
    try:
        built.step()
        centre, extent = built.total_volume()
        built.flux_in_box(0, centre, extent)
    finally:
        built.close()
    assert plan.sync_refusals == {"update_H": 1}
    assert plan.report()["sync_refusals"] == {"update_H": 1}
    assert leader.runs == 1, "the product ran in step() and NOT in the half-step"


def test_a_shim_that_does_not_carry_the_name_answers_false(monkeypatch):
    """Why this is a NAME and not a second method.

    An adapter carrying only the slot vocabulary — the three parity probes and the
    frozen copy inside a gate's result directory — answers False here and the driver
    runs the array ``update_H``. A second required method would have raised
    ``AttributeError`` in every one of them, and one of those copies is inside a
    record nothing may edit.
    """
    built = _driver()
    leader = SpanningPlan(built, MAGNETIC_SPAN)
    adapter = _install(monkeypatch, built,
                       {"step_B": leader, "update_H": Sentinel(leader)},
                       SlotOnlyAdapter)
    try:
        built.step()
        centre, extent = built.total_volume()
        built.flux_in_box(0, centre, extent)
    finally:
        built.close()
    assert adapter.refused.count(fastpath.SYNC_UPDATE_H_PASS) == 1, (
        f"the shim must be asked the sync name exactly once per half-step; it was "
        f"asked {adapter.refused}")
    # The far-fill pass names are refused by the same shim for the same reason, and
    # that is the PRECEDENT this channel follows rather than a second finding.
    assert set(adapter.refused) == set(fastpath.FAR_FILL_OWNERS) | {
        fastpath.SYNC_UPDATE_H_PASS}
    assert leader.runs == 2, (
        "step's leading consult plus the half-step's step_B consult, which the shim "
        "DOES carry — the fallback is the fourth consult only")


def test_the_containment_rule_reads_the_span_and_not_the_label(monkeypatch):
    """A single unpaired plan at ``update_H`` is admitted; the same plan paired is not.

    Same object, same slot, same arm label — only the span differs. That is what makes
    :meth:`fastpath.FastPathPlan._sync_path_holds` a rule about the product rather
    than a list of names.
    """
    built = _driver()
    alone = SpanningPlan(built, ("update_H",))
    plan = _install(monkeypatch, built, {"update_H": alone})
    try:
        built.step()
        centre, extent = built.total_volume()
        built.flux_in_box(0, centre, extent)
    finally:
        built.close()
    assert alone.runs == 2, "an unpaired update_H plan runs in the half-step too"
    assert plan.sync_refusals == {}
