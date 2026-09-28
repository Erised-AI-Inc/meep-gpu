"""The deposit repair's LINEAR route, pinned against a frozen copy of the arithmetic it replaced.

WHAT CHANGED, 2026-09-18. ``LeadingRepairPlan`` now owns a ``_PreparedCells``: the cell
set ``repair_cells`` returns is built on the first step and reused, spelled as one C-order
index per component, and ``save``/``apply`` gather and scatter through it. The redundant
``xp.array(copy=True)`` after each gather is gone, the constitutive product is gathered
before it is multiplied rather than formed over the whole volume, and the coefficients at
the cells are gathered once per coefficient object. Measured on NumPy lifts of pml_2d:
save 12 -> 6 array ops, apply 30 -> 27 with no whole-volume product, host time per step
~90 -> ~58 us; a folded grid additionally stops re-running the six boolean-mask gathers of
the image closure every step.

WHAT THIS FILE HOLDS THAT ``test_deposit_repair.py`` DOES NOT. That file proves the repair
against the driver's own order. This one proves the NEW ROUTE against the OLD ARITHMETIC,
word for word, on the configurations the new route could get wrong in ways the old one
could not: a C-order index that aliases a neighbour, a cache that outlives the cell set it
spells, a gather that returns a view, a fallback that silently becomes the only path.

* :func:`_frozen_save` / :func:`_frozen_apply` are the shipped arithmetic as of
  ``deposit_repair.py`` sha256 ``c1d557040c6d`` (:717-757, :760-797, :798-866), frozen
  here so a later edit to the module cannot move both sides of the comparison at once.
* Every comparison asserts WHICH ROUTE RAN, from the plan's counters. A bit-identity
  check against a route that silently fell back to the 3-tuple would pass for the wrong
  reason; the fallback case asserts the opposite.
* Every comparison is paired with a NULL CONTROL -- the same protocol with the repair
  withheld -- which must diverge, or the comparison cannot see the repair at all.

WHAT CHANGED, 2026-09-19 (reviewer finding F3). Module-level ``save``/``apply`` -- what
the gates and probes that bracket a launch by hand call -- took the 3-tuple route when
handed no plan, so those legs certified a route no production installer runs. ``save``
now builds a ONE-SHOT ``_PreparedCells`` for such a caller and stamps it under
``_PREPARED_KEY``. So every comparison here runs THREE legs from the same states -- the
frozen arithmetic, the plan route, and the module-level route -- and asserts the route
counters of BOTH live legs; the module-level counters are summed over the one-shot caches
the calls built.

WHAT CHANGED, 2026-09-20. ``save``/``apply`` repair, per source, only the target that
source's injection WRITES -- one ``(component, source)`` entry per single-component source
where there were three -- and keep all three, counted in ``restriction_fallbacks``, when
the written array cannot be established. Every count pinned here therefore fell from
three per source to one, and the cases that are ABOUT one entry falling back while the
others do not now drive a second source on another component, so they still hold both
kinds of entry. THE FROZEN ARITHMETIC IS HANDED THE SAME TARGETS, from a table spelled in
this file (:data:`WRITTEN_FIELD`), because this protocol's "after" state is NOISE rather
than a launch: the full bracket recomputes Ex at an Ez deposit from the saved state and
lands on the recurrence's value, which the noise is not, so the two would differ at every
unwritten component by construction (measured: 1 word each in Ex, Ey, f_w_Ex, f_w_Ey).
That they AGREE after a real launch -- the claim the restriction rests on -- is
``test_deposit_repair_component_restriction.py``'s subject, against this file's frozen
arithmetic with every target.
"""

from __future__ import annotations

import pathlib
import sys

import numpy
import pytest

from . import deposit_repair, stepping
from .test_deposit_repair import (FAR_ROW_Y, MIRROR_ROW_Y, _build, _folded_build,
                                  _folded_source, _make_source, _plain_build)

ARRAYS = ("Ex", "Ey", "Ez", "Dx", "Dy", "Dz", "Hx", "Hy", "Hz", "Bx", "By", "Bz",
          "f_w_Ex", "f_w_Ey", "f_w_Ez", "f_w_Hx", "f_w_Hy", "f_w_Hz")

TRIALS = 4


# ---------------------------------------------------------------------------
# The frozen reference: the shipped arithmetic, verbatim
# ---------------------------------------------------------------------------


#: WHICH FIELD A SOURCE COMPONENT'S INJECTION FEEDS, spelled HERE and never read from the
#: module under test: ``Ez`` and ``Dz`` both deposit into Dz, whose constitutive result is
#: Ez (``sources.py:120-133``). A restriction bug in ``deposit_repair`` cannot move this
#: side of a comparison.
WRITTEN_FIELD = {"Ex": "Ex", "Ey": "Ey", "Ez": "Ez", "Dx": "Ex", "Dy": "Ey", "Dz": "Ez",
                 "Hx": "Hx", "Hy": "Hy", "Hz": "Hz", "Bx": "Hx", "By": "Hy", "Bz": "Hz"}


def _written(pair, source):
    """The seam targets the frozen arithmetic visits for ONE source under the restriction:
    the one its injection writes, or every target for a source that names no component
    (:class:`_StubSource`), which is the module's own fail-closed answer."""
    targets = deposit_repair.SEAMS[pair]["targets"]
    field = WRITTEN_FIELD.get(getattr(source, "component", None))
    chosen = tuple(target for target in targets if target[0] == field)
    return chosen or targets


def _frozen_save(fields, sources, pair, pml, paths, targets_for=None):
    """``deposit_repair.save`` at sha256 c1d557040c6d (:738-757), verbatim.

    ``targets_for`` is the one thing that is not: ``None`` visits EVERY target of the
    seam for every source, which is that arithmetic exactly and what
    ``test_deposit_repair_component_restriction.py`` compares the restricted module to;
    :func:`_written` visits the targets the restricted module visits."""
    ok, reasons = deposit_repair.repairable(fields, pair, pml, paths=paths)
    if not ok:
        raise deposit_repair.DepositNotRepairable("; ".join(reasons))
    path = deposit_repair._selected_path(pml, paths)
    xp = fields.grid.xp
    saved = {deposit_repair._PATH_KEY: path}
    for index, source in enumerate(sources):
        idx = deposit_repair._deposit_index(source)
        if idx is None:
            continue
        targets = (deposit_repair.SEAMS[pair]["targets"] if targets_for is None
                   else targets_for(pair, source))
        for component, target, _axis in targets:
            cells = deposit_repair.repair_cells(fields, target, idx)
            saved[(component, index, "cells")] = cells
            if path is deposit_repair.PLAIN_PATH:
                continue
            field = getattr(fields, component)
            fw = getattr(fields, "f_w_" + component)
            saved[(component, index, "field")] = xp.array(field[cells], copy=True)
            saved[(component, index, "fw")] = xp.array(fw[cells], copy=True)
    return saved


def _frozen_apply(fields, pml, sources, pair, saved):
    """``deposit_repair._apply_plain`` / ``apply`` at sha256 c1d557040c6d, verbatim, over
    whatever :func:`_frozen_save` captured: a ``(component, source)`` it did not capture
    is skipped, and with every target captured nothing is."""
    repaired = 0
    if saved.get(deposit_repair._PATH_KEY) is deposit_repair.PLAIN_PATH:
        for component, _source_name, _axis in deposit_repair.SEAMS[pair]["targets"]:
            volume = fields.displacement_minus_polarization(component)
            constitutive = volume * fields.inverse_epsilon_for(component)
            field = getattr(fields, component)
            for index, source in enumerate(sources):
                if deposit_repair._deposit_index(source) is None:
                    continue
                if (component, index, "cells") not in saved:
                    continue
                cells = saved[(component, index, "cells")]
                field[cells] = constitutive[cells]
                repaired += int(getattr(cells[0], "size", len(cells[0])))
        return repaired
    spec = deposit_repair.SEAMS[pair]
    xp = fields.grid.xp
    for component, source_name, axis in spec["targets"]:
        field = getattr(fields, component)
        fw = getattr(fields, "f_w_" + component)
        kps, kms = stepping._constitutive_coefficients(
            pml, axis, half_integer=spec["half_integer"])
        if pair == "D":
            volume = fields.displacement_minus_polarization(component)
            constitutive = volume * fields.inverse_epsilon_for(component)
        else:
            constitutive = getattr(fields, source_name)
        kp_full = xp.broadcast_to(xp.asarray(kps), field.shape)
        km_full = xp.broadcast_to(xp.asarray(kms), field.shape)
        for index, source in enumerate(sources):
            if deposit_repair._deposit_index(source) is None:
                continue
            if (component, index, "cells") not in saved:
                continue
            cells = saved[(component, index, "cells")]
            fresh = constitutive[cells]
            restored = saved[(component, index, "field")] + kp_full[cells] * fresh
            restored = restored - km_full[cells] * saved[(component, index, "fw")]
            field[cells] = restored
            fw[cells] = fresh
            repaired += int(getattr(cells[0], "size", len(cells[0])))
    return repaired


# ---------------------------------------------------------------------------
# The protocol: capture, perturb (the launch wrote, the driver injected), repair
# ---------------------------------------------------------------------------


def _live(fields):
    """Every array the repair reads or writes, INCLUDING the inverse permittivity.

    The material is part of the perturbed state on purpose. Most of these builds are
    vacuum, where ``inv_eps`` is exactly 1.0 and ``volume * inv_eps == volume`` -- so a
    route that dropped the multiply, or read ``inv_eps`` at the wrong cell, would pass.
    Perturbing it (once per distinct array; the three components often alias one) makes
    it vary cell to cell and differ from 1.
    """
    out = {name: getattr(fields, name) for name in ARRAYS
           if isinstance(getattr(fields, name, None), numpy.ndarray)}
    seen = set()
    for component in ("Ex", "Ey", "Ez"):
        inverse = fields.inverse_epsilon_for(component)
        if isinstance(inverse, numpy.ndarray) and id(inverse) not in seen:
            seen.add(id(inverse))
            out[f"inv_eps.{component}"] = inverse
    for index, state in enumerate(fields.polarizations):
        for component, array in (getattr(state, "P", None) or {}).items():
            if array is not None:
                out[f"P{index}.{component}"] = array
    return out


def _snapshot(fields):
    return {name: array.copy() for name, array in _live(fields).items()}


def _restore(fields, snapshot):
    live = _live(fields)
    for name, array in snapshot.items():
        live[name][...] = array


def _perturbed(snapshot, generator):
    out = {}
    for name, array in snapshot.items():
        noise = generator.standard_normal(array.shape) * 1e-3
        if numpy.iscomplexobj(array):
            noise = noise + 1j * generator.standard_normal(array.shape) * 1e-3
        out[name] = (array + noise).astype(array.dtype)
    return out


def _words(array):
    return numpy.ascontiguousarray(array).reshape(-1).view(numpy.uint32)


def _differing(left, right):
    return {name: int(numpy.count_nonzero(_words(left[name]) != _words(right[name])))
            for name in left
            if int(numpy.count_nonzero(_words(left[name]) != _words(right[name])))}


class _Inner:
    runs = 0

    def run(self, *_args, **_kwargs):
        type(self).runs += 1


class _Routes:
    """The four route counters, summed over every one-shot cache the module-level
    ``save`` built -- one per call, so no single object holds a run's total."""

    NAMES = ("linear_saves", "cells_saves", "linear_repairs", "cells_repairs")
    #: Per source per save: narrowed to the target it writes, or kept at all three.
    RESTRICTION = ("restricted_sources", "restriction_fallbacks")

    def __init__(self):
        for name in self.NAMES + self.RESTRICTION:
            setattr(self, name, 0)
        self.entries = 0
        self.caches = []

    def add(self, prepared):
        assert isinstance(prepared, deposit_repair._PreparedCells), type(prepared)
        assert all(prepared is not seen for seen in self.caches), (
            "two module-level saves shared one cache; it must be built per call")
        self.caches.append(prepared)
        for name in self.NAMES + self.RESTRICTION:
            setattr(self, name, getattr(self, name) + getattr(prepared, name))
        self.entries = max(self.entries, len(prepared.entries))


class _Comparison:
    """What :func:`_compare` measured. ``plan``/``module`` are the words each live
    route differs from the frozen arithmetic by, ``routes`` the words the two live
    routes differ from EACH OTHER by, ``null`` the words withholding the repair moves."""

    def __init__(self, leading):
        self.plan, self.module, self.routes, self.null = {}, {}, {}, {}
        self.leading = leading
        self.module_routes = _Routes()

    def assert_exact(self):
        assert not self.plan, ("plan route", self.plan)
        assert not self.module, ("module-level route", self.module)
        assert not self.routes, ("plan route vs module-level route", self.routes)
        assert self.null, ("withholding the repair changed nothing, so this comparison "
                           "cannot see the repair")


def _accumulate(total, counts):
    for name, count in counts.items():
        total[name] = total.get(name, 0) + count


def _compare(fields, pml, sources, pair, paths, *, trials=TRIALS, seed=20260918,
             between=None):
    """Run the frozen arithmetic, the plan route and the module-level route from the
    same states; return a :class:`_Comparison`.

    ``between(fields)`` runs after each live leg's save and before its repair -- the
    slot where the driver's inject/fill/clear run and where an array can be swapped out
    from under a cached index -- and returns a callable that undoes it once that leg's
    result is read, so the next leg saves against the arrays the first one did.

    THE FROZEN LEG VISITS THE TARGETS THE SOURCES WRITE (:func:`_written`), as the live
    legs do. It has to: ``after`` is noise, not a launch, so a target the source did not
    write holds a value no recurrence produced and the full bracket would "repair" it to
    something else. See the module docstring, 2026-09-20.
    """
    generator = numpy.random.default_rng(seed)
    leading = deposit_repair.LeadingRepairPlan(_Inner(), fields, pml, sources, pair,
                                               paths)
    trailing = deposit_repair.TrailingRepairPlan("slot", leading, fields, pml)
    result = _Comparison(leading)
    for _trial in range(trials):
        base = _perturbed(_snapshot(fields), generator)
        after = _perturbed(base, generator)

        _restore(fields, base)
        saved = _frozen_save(fields, sources, pair, pml, paths, _written)
        _restore(fields, after)
        frozen_count = _frozen_apply(fields, pml, sources, pair, saved)
        reference = _snapshot(fields)

        _restore(fields, base)                          # the PLAN route
        leading.run()
        undo = between(fields) if between is not None else None
        _restore(fields, after)
        trailing.run()
        assert leading.repairs == frozen_count, (leading.repairs, frozen_count)
        planned = _snapshot(fields)
        if undo is not None:
            undo()
        _accumulate(result.plan, _differing(reference, planned))

        _restore(fields, base)                          # the MODULE-LEVEL route
        saved = deposit_repair.save(fields, sources, pair, pml, paths=paths)
        undo = between(fields) if between is not None else None
        _restore(fields, after)
        count = deposit_repair.apply(fields, pml, sources, pair, saved)
        assert count == frozen_count, (count, frozen_count)
        module = _snapshot(fields)
        if undo is not None:
            undo()
        result.module_routes.add(saved[deposit_repair._PREPARED_KEY])
        _accumulate(result.module, _differing(reference, module))
        _accumulate(result.routes, _differing(planned, module))

        _restore(fields, after)                         # the NULL CONTROL: no repair
        _accumulate(result.null, _differing(reference, _snapshot(fields)))
        _restore(fields, base)
    return result


def _entries(plan):
    return len(plan.prepared.entries)


def _assert_routes(result, *, linear_saves, cells_saves, linear_repairs, cells_repairs,
                   restricted, fallbacks=0):
    """The same counts on BOTH live routes: the module-level caller must take the route
    the plan takes, entry for entry, fallbacks included. ``restricted`` / ``fallbacks``
    are the sources each save narrowed to the target they write, and the ones it could
    not -- so a comparison that passed because the module quietly kept all three targets
    fails here rather than reading as the restricted route."""
    for label, counters in (("plan", result.leading.prepared),
                            ("module-level", result.module_routes)):
        got = tuple(getattr(counters, name) for name in _Routes.NAMES + _Routes.RESTRICTION)
        want = (linear_saves, cells_saves, linear_repairs, cells_repairs, restricted,
                fallbacks)
        assert got == want, (
            f"{label} route counters (linear_saves, cells_saves, linear_repairs, "
            f"cells_repairs, restricted_sources, restriction_fallbacks) = {got}, "
            f"expected {want}")


# ---------------------------------------------------------------------------
# Lifted corpus cases: the configurations the fused route actually brackets
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def route():
    directory = str(pathlib.Path(__file__).resolve().parents[1] / "parity" / "meep_gpu")
    if directory not in sys.path:
        sys.path.insert(0, directory)
    import gate_dispatch_fused_route as module  # noqa: PLC0415
    return module


#: Split-field D, split-field B (the magnetic seam), complex storage, and the PLAIN path
#: on a five-pole dispersive row -- four recurrences, each lifted from the corpus.
LIFTED = ["pml_1d", "magnetic_seam_2d", "complex_1d", "absorber_1d"]


@pytest.mark.parametrize("case", LIFTED)
def test_the_linear_route_is_bit_identical_to_the_frozen_arithmetic(route, case):
    mp = pytest.importorskip("meep")
    import meep_gpu  # noqa: PLC0415

    mp.verbosity(0)
    simulation, _monitors, _units = route.CASES[case](mp)
    driver = meep_gpu.lift_simulation(simulation, prefer_gpu=False)
    driver.run(num_steps=8)
    fields, pml = driver.fields, driver.pml
    compared = 0
    for pair in ("B", "D"):
        sources = deposit_repair.in_seam_sources(list(driver._sources), pair)
        if not sources:
            continue
        paths = (deposit_repair.repair_path_for(pml),)
        result = _compare(fields, pml, sources, pair, paths)
        result.assert_exact()
        entries = _entries(result.leading)
        # ONE cell set per source -- each lifted source drives one component -- where
        # the unrestricted bracket held three.
        assert entries == len(sources), (case, pair, entries, len(sources))
        assert result.module_routes.entries == entries
        # Every entry on the linear route, on BOTH live legs: a comparison against a
        # route that silently fell back would pass for the wrong reason. The plain path
        # saves no state, so it has no saves to count.
        split = paths[0] is deposit_repair.SPLIT_FIELD_PATH
        _assert_routes(result, linear_saves=TRIALS * entries if split else 0,
                       cells_saves=0, linear_repairs=TRIALS * entries, cells_repairs=0,
                       restricted=TRIALS * len(sources))
        compared += 1
    assert compared, f"{case} carries no in-seam source; it measures nothing here"


# ---------------------------------------------------------------------------
# The folded seam: the image closure, built once, spelled linearly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("y,component", [(MIRROR_ROW_Y, "Ez"), (FAR_ROW_Y, "Ey")],
                         ids=["near_fill_row", "far_ghost_reflect_row"])
def test_the_folded_closure_rides_the_linear_route_and_is_built_once(
        monkeypatch, y, component):
    """A deposit ON a row a fill reads from, so the cell set carries its images.

    The closure is where the old per-step cost was worst -- six boolean-mask gathers and
    their concatenates, a device-to-host sync each on CuPy -- and where a cached set is
    most exposed: the images are the cells a stale or mis-spelled index would miss.
    """
    _grid, fields, pml = _folded_build()
    source = _folded_source(fields.grid, y, component)
    points = int(numpy.atleast_1d(source._point_ix).size)
    calls = []
    real = deposit_repair.repair_cells
    monkeypatch.setattr(deposit_repair, "repair_cells",
                        lambda *args: calls.append(args[1]) or real(*args))
    result = _compare(fields, pml, [source], "D", (deposit_repair.SPLIT_FIELD_PATH,))
    result.assert_exact()
    plan = result.leading
    entry = plan.prepared.entries[(component, 0)]
    assert int(entry.linear.size) > points, (
        f"the {component} cell set carries no image ({int(entry.linear.size)} cells for "
        f"{points} deposit points), so this case does not exercise the closure")
    for cache in result.module_routes.caches:
        assert int(cache.entries[(component, 0)].linear.size) == int(entry.linear.size)
    assert set(plan.prepared.entries) == {(component, 0)}, sorted(plan.prepared.entries)
    _assert_routes(result, linear_saves=TRIALS, cells_saves=0,
                   linear_repairs=TRIALS, cells_repairs=0, restricted=TRIALS)
    calls.clear()
    for _step in range(3):
        plan.run()
    assert not calls, (
        f"the plan re-ran repair_cells on {calls} after its cell sets were built; the "
        f"closure must be built once per component and reused")


def test_a_repair_cells_swapped_mid_run_is_honoured_on_the_next_step(monkeypatch):
    """The gates' point-only control leg swaps ``repair_cells`` on the module. A cache
    that kept the closure it built first would repair the images the control removed,
    and the control would stop diverging -- reading as a closure nothing needs."""
    _grid, fields, pml = _folded_build()
    source = _folded_source(fields.grid, MIRROR_ROW_Y)
    plan = deposit_repair.LeadingRepairPlan(_Inner(), fields, pml, [source], "D")
    plan.run()
    closure = int(plan.prepared.entries[("Ez", 0)].linear.size)
    monkeypatch.setattr(deposit_repair, "repair_cells",
                        lambda fields, target, index: index)
    plan.run()
    point_only = int(plan.prepared.entries[("Ez", 0)].linear.size)
    assert point_only < closure, (point_only, closure)


def _point_only(fields, target, index):
    """The gates' point-only ``repair_cells`` (``parity/meep_gpu/
    probe_cuda_electric_fold_closure.py:197-208``): the deposit points, no images."""
    xp = fields.grid.xp
    columns = [xp.asarray(part).reshape(-1) for part in index]
    return columns[0], columns[1], columns[2]


def test_a_point_only_swap_takes_effect_on_the_module_level_route(monkeypatch):
    """The same control, on the route the gates call. The module-level ``save`` builds a
    FRESH cache per call from the ``repair_cells`` the module holds at that call, so a
    swap between calls is honoured with no key to go stale -- and the swapped set still
    rides the linear route, so the control measures the product's route, not a fallback.

    Pinned as the CONTROL'S SIGNAL, not only as a size: the point-only leg must match the
    frozen arithmetic run under the same swap AND diverge from the closure leg, or a
    gate's point-only control would stop firing and read as a closure nothing needs."""
    _grid, fields, pml = _folded_build()
    source = _folded_source(fields.grid, MIRROR_ROW_Y)
    points = int(numpy.atleast_1d(source._point_ix).size)
    key = deposit_repair._PREPARED_KEY
    generator = numpy.random.default_rng(20260919)
    base = _perturbed(_snapshot(fields), generator)
    after = _perturbed(base, generator)

    def leg(saver, applier):
        _restore(fields, base)
        saved = saver(fields, [source], "D", pml, (deposit_repair.SPLIT_FIELD_PATH,))
        _restore(fields, after)
        applier(fields, pml, [source], "D", saved)
        return saved, _snapshot(fields)

    def module_save(fields, sources, pair, pml, paths):
        return deposit_repair.save(fields, sources, pair, pml, paths=paths)

    def frozen_save(fields, sources, pair, pml, paths):
        return _frozen_save(fields, sources, pair, pml, paths, _written)

    closure, closure_state = leg(module_save, deposit_repair.apply)
    monkeypatch.setattr(deposit_repair, "repair_cells", _point_only)
    swapped, swapped_state = leg(module_save, deposit_repair.apply)
    _frozen, frozen_state = leg(frozen_save, _frozen_apply)

    assert swapped[key] is not closure[key], "the module-level cache outlived its call"
    closure_entry = closure[key].entries[("Ez", 0)]
    swapped_entry = swapped[key].entries[("Ez", 0)]
    assert swapped_entry.linear is not None, "the point-only set fell off the linear route"
    assert int(swapped_entry.linear.size) == points < int(closure_entry.linear.size), (
        points, int(swapped_entry.linear.size), int(closure_entry.linear.size))
    assert set(swapped[key].entries) == {("Ez", 0)}, sorted(swapped[key].entries)
    assert swapped[key].linear_repairs == 1 and swapped[key].cells_repairs == 0
    assert (swapped[key].restricted_sources, swapped[key].restriction_fallbacks) == (1, 0)
    assert not _differing(frozen_state, swapped_state), "point-only leg != frozen arithmetic"
    assert _differing(closure_state, swapped_state), (
        "removing the image closure changed nothing on the module-level route, so a "
        "gate's point-only control could no longer fire")


def test_a_rebound_deposit_index_rebuilds_the_entry():
    """The sources REBIND their index arrays when they rebuild them
    (sources.py:1667/:1829/:1943/:2069); the entry is keyed on those objects."""
    _grid, fields, pml = _build()
    source = _make_source(fields.grid, "D", "Ez", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    plan = deposit_repair.LeadingRepairPlan(_Inner(), fields, pml, [source], "D")
    plan.run()
    first = plan.prepared.entries[("Ez", 0)]
    plan.run()
    assert plan.prepared.entries[("Ez", 0)] is first, "the entry was rebuilt every step"
    source._point_ix = source._point_ix.copy()
    plan.run()
    assert plan.prepared.entries[("Ez", 0)] is not first


# ---------------------------------------------------------------------------
# The fallback: exact, never a refusal, and observable
# ---------------------------------------------------------------------------


def _fortran(name):
    """Swap ``name`` for a Fortran-order copy; return the undo."""
    def swap(fields):
        original = getattr(fields, name)
        setattr(fields, name, numpy.asfortranarray(original))
        return lambda: setattr(fields, name, original)
    return swap


def _strided(array):
    """``array``'s values in a SAME-SHAPE view that is not C-contiguous: every other row
    of a buffer twice as long on axis 1, where the grid's extent exceeds 1."""
    shape = array.shape
    buffer = numpy.zeros((shape[0], 2 * shape[1]) + tuple(shape[2:]), dtype=array.dtype)
    view = buffer[:, ::2]
    view[...] = array
    return view


def _padded(array):
    """``array``'s values in a C-contiguous array ONE CELL LONGER on the last axis: every
    3-tuple index stays in range and addresses the same cell, while a C-order index
    built for the original shape addresses a different one wherever ix or iy > 0."""
    shape = array.shape
    padded = numpy.zeros(tuple(shape[:-1]) + (shape[-1] + 1,), dtype=array.dtype)
    padded[..., :shape[-1]] = array
    return padded


@pytest.mark.parametrize("name,layout,build,paths", [
    ("Ez", numpy.asfortranarray, _build, (deposit_repair.SPLIT_FIELD_PATH,)),
    ("Ez", _strided, _build, (deposit_repair.SPLIT_FIELD_PATH,)),
    ("f_w_Ez", _padded, _build, (deposit_repair.SPLIT_FIELD_PATH,)),
    ("Ez", numpy.asfortranarray, _plain_build, (deposit_repair.PLAIN_PATH,)),
], ids=["fortran_target", "strided_target", "reshaped_state", "plain_fortran_target"])
def test_an_array_the_index_cannot_use_falls_back_to_the_3_tuple_route_exactly(
        name, layout, build, paths):
    """An array the linear index cannot use, on BOTH live routes. Non-C-contiguous:
    ``reshape(-1)`` is then a COPY, and a scatter into a copy lands nowhere. A different
    shape: the index was built and range-checked for the target's shape, so it would
    gather and scatter the wrong cells. The guard (``_flat``) must see either and send
    THAT ``(component, source)`` down the 3-tuple route -- still exact, never a refusal,
    and only that entry: a SECOND source, on Ex, keeps the linear route in the same
    bracket. (Until 2026-09-20 the one Ez source carried Ex and Ey entries of its own,
    which is what this read; restricted, it carries Ez alone, so the entry that must NOT
    fall back is another source's.) The plain case is also the one that measures
    ``_constitutive_at``'s gather-then-multiply on the plain path against the frozen
    whole-volume product."""
    _grid, fields, pml = build()
    setattr(fields, name, layout(getattr(fields, name)))
    array = getattr(fields, name)
    if name == "Ez":
        assert not array.flags.c_contiguous, f"the swap left {name} C-contiguous"
    else:
        assert array.shape != fields.Ez.shape, f"the swap left {name} Ez-shaped"
    sources = [_make_source(fields.grid, "D", "Ez", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
               _make_source(fields.grid, "D", "Ex", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))]
    result = _compare(fields, pml, sources, "D", paths)
    result.assert_exact()
    assert set(result.leading.prepared.entries) == {("Ez", 0), ("Ex", 1)}
    split = paths[0] is deposit_repair.SPLIT_FIELD_PATH
    _assert_routes(result, linear_saves=TRIALS if split else 0,
                   cells_saves=TRIALS if split else 0,
                   linear_repairs=TRIALS, cells_repairs=TRIALS, restricted=2 * TRIALS)


def test_an_array_swapped_between_save_and_apply_falls_back_at_apply():
    """The save gathered through the index; by the second consult the target is an
    array the index cannot write through. Only the repair falls back, and only for the
    source whose array moved -- on the plan and on the module-level route alike."""
    _grid, fields, pml = _build()
    sources = [_make_source(fields.grid, "D", "Ez", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
               _make_source(fields.grid, "D", "Ex", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))]
    result = _compare(fields, pml, sources, "D", (deposit_repair.SPLIT_FIELD_PATH,),
                      trials=1, between=_fortran("f_w_Ez"))
    result.assert_exact()
    _assert_routes(result, linear_saves=2, cells_saves=0, linear_repairs=1,
                   cells_repairs=1, restricted=2)


class _StubSource:
    """A deposit whose index the linear spelling must refuse: a NEGATIVE x, which the
    3-tuple index wraps to the last row and a C-order index would read as the cell
    before the array. The per-axis range check is what keeps the two routes equal.

    It names no ``component`` and is no engine source class, so which array it writes
    cannot be established: the repair FAILS CLOSED to all three targets and counts it."""

    field_type = "E"

    def __init__(self, xp, ix, iy, iz):
        self._point_ix = xp.asarray([ix])
        self._point_iy = xp.asarray([iy])
        self._point_iz = xp.asarray([iz])


def test_an_index_outside_its_axis_is_never_spelled_linearly():
    _grid, fields, pml = _build()
    source = _StubSource(numpy, -1, 3, 0)
    result = _compare(fields, pml, [source], "D", (deposit_repair.SPLIT_FIELD_PATH,),
                      trials=2)
    result.assert_exact()
    for prepared in [result.leading.prepared] + result.module_routes.caches:
        assert set(prepared.entries) == {("Ex", 0), ("Ey", 0), ("Ez", 0)}
        assert all(entry.linear is None for entry in prepared.entries.values())
    _assert_routes(result, linear_saves=0, cells_saves=6, linear_repairs=0,
                   cells_repairs=6, restricted=0, fallbacks=2)


# ---------------------------------------------------------------------------
# The plain path and the module-level entry points
# ---------------------------------------------------------------------------


def test_the_plain_path_rides_the_linear_route_on_a_dispersive_row():
    _grid, fields, pml = _plain_build()
    source = _make_source(fields.grid, "D", "Ez", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    result = _compare(fields, pml, [source], "D", (deposit_repair.PLAIN_PATH,))
    result.assert_exact()
    _assert_routes(result, linear_saves=0, cells_saves=0, linear_repairs=TRIALS,
                   cells_repairs=0, restricted=TRIALS)


@pytest.mark.parametrize("pair,component", [("D", "Ez"), ("B", "Hz")])
def test_module_level_save_and_apply_keep_the_shipped_keys_and_values(pair, component):
    """A caller holding no plan -- the gates and probes that call ``save``/``apply``
    directly -- gets HEAD's keys and values FOR THE COMPONENT THE SOURCE WRITES, plus the
    one-shot cache under ``_PREPARED_KEY`` and the per-source targets under
    ``_TARGETS_KEY``, and state that is a COPY. The two unwritten components' keys --
    which HEAD's full bracket carried -- are absent, by name."""
    generator = numpy.random.default_rng(7)
    _grid, fields, pml = _build()
    source = _make_source(fields.grid, pair, component, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    base = _snapshot(fields)
    after = _perturbed(base, generator)

    saved = deposit_repair.save(fields, [source], pair, pml)
    frozen = _frozen_save(fields, [source], pair, pml, (deposit_repair.SPLIT_FIELD_PATH,),
                          _written)
    shipped_keys = set(saved) - {deposit_repair._PREPARED_KEY, deposit_repair._TARGETS_KEY}
    assert shipped_keys == set(frozen), (shipped_keys ^ set(frozen))
    every = set(_frozen_save(fields, [source], pair, pml,
                             (deposit_repair.SPLIT_FIELD_PATH,)))
    others = {target for target, _source, _axis in deposit_repair.SEAMS[pair]["targets"]
              if target != component}
    assert every - shipped_keys == {(other, 0, part) for other in others
                                    for part in ("cells", "field", "fw")}
    assert saved[deposit_repair._TARGETS_KEY] == {0: tuple(
        target for target in deposit_repair.SEAMS[pair]["targets"]
        if target[0] == component)}
    assert isinstance(saved[deposit_repair._PREPARED_KEY], deposit_repair._PreparedCells)
    for key, value in frozen.items():
        if key[-1] in ("field", "fw"):
            assert not numpy.shares_memory(saved[key], getattr(
                fields, key[0] if key[-1] == "field" else "f_w_" + key[0])), key
            assert not _differing({"v": value}, {"v": saved[key]}), key

    _restore(fields, after)
    count = deposit_repair.apply(fields, pml, [source], pair, saved)
    got = _snapshot(fields)
    _restore(fields, after)
    assert count == _frozen_apply(fields, pml, [source], pair, frozen)
    assert not _differing(_snapshot(fields), got)
    _restore(fields, base)


@pytest.mark.parametrize("pair,component,build,paths", [
    ("D", "Ez", _build, (deposit_repair.SPLIT_FIELD_PATH,)),
    ("B", "Hz", _build, (deposit_repair.SPLIT_FIELD_PATH,)),
    ("D", "Ez", _plain_build, (deposit_repair.PLAIN_PATH,)),
], ids=["split_field_D", "split_field_B", "plain_D"])
def test_module_level_save_and_apply_report_the_linear_route(pair, component, build,
                                                              paths):
    """F3, pinned where it lives: the counters of the cache the module-level ``save``
    stamps say the LINEAR route ran -- ONE save (none on the plain path, which saves no
    state) and ONE repair, for the one component the source writes, no fallback of
    either kind -- so a gate calling these two functions certifies the route the
    installers' plans run."""
    _grid, fields, pml = build()
    source = _make_source(fields.grid, pair, component, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    saved = deposit_repair.save(fields, [source], pair, pml, paths=paths)
    prepared = saved[deposit_repair._PREPARED_KEY]
    assert set(prepared.entries) == {(component, 0)}, sorted(prepared.entries)
    assert all(entry.linear is not None for entry in prepared.entries.values())
    split = paths[0] is deposit_repair.SPLIT_FIELD_PATH
    assert (prepared.linear_saves, prepared.cells_saves) == ((1 if split else 0), 0)
    assert (prepared.restricted_sources, prepared.restriction_fallbacks) == (1, 0)
    points = int(numpy.atleast_1d(source._point_ix).size)
    assert deposit_repair.apply(fields, pml, [source], pair, saved) == points
    assert (prepared.linear_repairs, prepared.cells_repairs) == (1, 0)


def test_the_plan_saves_state_that_is_a_copy_not_a_view():
    """The copy the plan's gather makes is what survives the fused launch's overwrite."""
    _grid, fields, pml = _build()
    source = _make_source(fields.grid, "D", "Ez", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    plan = deposit_repair.LeadingRepairPlan(_Inner(), fields, pml, [source], "D")
    plan.run()
    for key, array in (("field", fields.Ez), ("fw", fields.f_w_Ez)):
        assert not numpy.shares_memory(plan.saved[("Ez", 0, key)], array)
    # The source writes Dz, so Ez is the whole of what is saved: no state is captured
    # for a component the repair will not touch.
    for component in ("Ex", "Ey"):
        for key in ("cells", "field", "fw"):
            assert (component, 0, key) not in plan.saved, (component, key)
    assert plan.prepared.linear_saves == 1


# ---------------------------------------------------------------------------
# The cost, pinned as a behaviour: the steady step is proportional to the deposit
# ---------------------------------------------------------------------------


class _Sizing(numpy.ndarray):
    """Records the size of every ufunc result and every fancy index it serves."""

    results: list = []
    fancy: list = []

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        inputs = tuple(i.view(numpy.ndarray) if isinstance(i, _Sizing) else i
                       for i in inputs)
        result = getattr(ufunc, method)(*inputs, **kwargs)
        _Sizing.results.append(int(numpy.size(result)))
        return result

    def __getitem__(self, index):
        parts = index if isinstance(index, tuple) else (index,)
        arrays = sum(isinstance(part, numpy.ndarray) for part in parts)
        if arrays:
            _Sizing.fancy.append(arrays)
        return super().__getitem__(index)


def test_the_steady_step_forms_nothing_the_size_of_the_volume():
    """The whole-volume ``volume * inv_eps`` is gone and every index is ONE array: on a
    device each 3-array fancy index is several kernels and allocations, not one."""
    _grid, fields, pml = _build()
    source = _make_source(fields.grid, "D", "Ez", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    leading = deposit_repair.LeadingRepairPlan(_Inner(), fields, pml, [source], "D")
    trailing = deposit_repair.TrailingRepairPlan("update_E", leading, fields, pml)
    for name in ARRAYS + ("Dx", "Dy", "Dz"):
        array = getattr(fields, name, None)
        if isinstance(array, numpy.ndarray):
            setattr(fields, name, array.view(_Sizing))
    for component in ("Ex", "Ey", "Ez"):
        fields._inv_eps_components[component] = \
            fields._inv_eps_components[component].view(_Sizing)
    leading.run()
    trailing.run()                                   # the first step builds the cache
    _Sizing.results.clear()
    _Sizing.fancy.clear()
    leading.run()
    trailing.run()
    cells = max(int(entry.linear.size) for entry in leading.prepared.entries.values())
    volume = int(fields.Ez.size)
    assert _Sizing.results and max(_Sizing.results) <= cells < volume, (
        f"a steady step formed a {max(_Sizing.results)}-element result on a "
        f"{volume}-cell volume with {cells} deposit cells")
    assert _Sizing.fancy and set(_Sizing.fancy) == {1}, (
        f"a steady step indexed with {sorted(set(_Sizing.fancy))}-array fancy indices")
