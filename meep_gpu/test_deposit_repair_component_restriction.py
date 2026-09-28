"""The deposit repair RESTRICTED to the component a source writes, against the full bracket.

WHAT CHANGED, 2026-09-20. ``save`` and ``apply`` used to repair all three constitutive
targets of a seam for every source. They now repair, per source, only the target whose
constitutive input that source's injection writes -- one entry per single-component
source per step instead of three -- and FAIL CLOSED to all three, counting it, whenever
the written array cannot be established. The argument is that a diagonal constitutive
target reads its own component at the cell and nothing else, so recomputing it where its
input did not change reproduces the launch's value bit for bit; the full bracket was
over-covering. ``the design notes (deposit-repair-kernel-scope)`` section 1 finding 1 and
section 9 carry the measurements this rests on.

WHAT THIS FILE HOLDS. Over-covering is safe and UNDER-covering is the silent wrong
answer, so nothing here is argued:

* every comparison runs the FROZEN FULL-BRACKET arithmetic
  (``test_deposit_repair_linear_route._frozen_save`` / ``_frozen_apply``, all three
  targets, untouched by the restriction) beside the live restricted module, from the same
  state, through complete sub-steps in the driver's own order, and compares EVERY stored
  array as uint32 words -- primaries, PML auxiliaries, conductivity auxiliaries and every
  pole's ``P`` / ``P_prev``;
* every comparison asserts from the counters that the restriction was TAKEN
  (``restricted_sources``) and that nothing fell back (``restriction_fallbacks``): a
  bit-identity check against a module that quietly kept the full bracket would pass for
  the wrong reason;
* every comparison is paired with THE NULL CONTROL -- the same protocol with the repair
  aimed at the WRONG component -- which must diverge, and diverge in the source's own
  component. A case where that leg agrees cannot tell a correct restriction from one
  that repairs nothing.

Two levels. The EMULATED level drives ``stepping``'s array functions in the driver's
order with the fused launch emulated by those same functions, as ``test_deposit_repair``
does. The DRIVER level installs the two repair plans on a real ``FdtdDriver`` and steps
it, which is what reaches the conductive replay
(``driver._inject_electric_through_conductivity``), the magnetic synchronization path
(``driver.synchronize_magnetic_fields``) and the lifted corpus cases.

ALL FOUR LISTED SOURCE CLASSES ARE DRIVEN. ``_make_source`` builds two of them
(``GaussianPulsedSource``, ``VolumeSource``); the ``CW`` cases and the ``continuous``
driver configurations build the other two (``ContinuousSource``, ``ExtendedSource``),
which are what the driver's default source kind constructs. Every emulated source is
bound to its layer as the driver binds it, so a deposit under a layer also writes its
``f_u`` mirror, and the cases that claim that are held to it.

AND THE LIST IS OF INJECTS, NOT OF TYPES: an exact-class source whose ``inject`` has
been rebound -- on the instance or on the class, delegating or writing a second array
-- must fail closed like any other source whose write path was not read.

Word counts are printed per comparison and in total (run with ``-s``).
"""

from __future__ import annotations

import functools
import inspect
import pathlib
import sys
import time
import types
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy
import pytest

from . import deposit_repair, stepping
from . import driver as driver_module
from . import sources as engine_sources
from .dispersion import PolarizationState, Susceptibility
from .driver import FdtdDriver
from .fields import Fields
from .grid import Grid
from .pml import PML
from .sources import (ContinuousSource, ExtendedSource, GaussianEnvelope,
                      GaussianPulsedSource, VolumeSource)
from .test_deposit_repair import (FAR_ROW_Y, MIRROR_ROW_Y, SEEDED, _build, _folded_build,
                                  _folded_source, _make_source, _plain_build, _seed)
from .test_deposit_repair_linear_route import _frozen_apply, _frozen_save
from .test_sync_pass_channel import _install

SPLIT = (deposit_repair.SPLIT_FIELD_PATH,)
PLAIN = (deposit_repair.PLAIN_PATH,)

#: The legacy route counters and the two this change adds.
LEGACY_COUNTERS = ("linear_saves", "cells_saves", "linear_repairs", "cells_repairs")
RESTRICTION_COUNTERS = ("restricted_sources", "restriction_fallbacks")


# ---------------------------------------------------------------------------
# The comparison: every stored array, as uint32 words, with its denominator
# ---------------------------------------------------------------------------

#: Words compared and words differing, per KIND of comparison. ``restricted`` is the
#: claim: the frozen full bracket against the live restricted module. ``reference`` is
#: what makes that claim mean something: the driver's own order (the array path)
#: against the frozen full bracket, on the same case.
TALLY = {kind: {"comparisons": 0, "words": 0, "differing": 0}
         for kind in ("restricted", "reference")}


@pytest.fixture(scope="module", autouse=True)
def _report_the_tally():
    yield
    for kind, title in (("restricted", "frozen FULL bracket vs the RESTRICTED module"),
                        ("reference", "driver order / array path vs frozen full bracket")):
        counts = TALLY[kind]
        print(f"\nTOTAL {title}: {counts['differing']} of {counts['words']} words "
              f"differ over {counts['comparisons']} comparisons", flush=True)


def _state_arrays(fields: Any) -> Dict[str, numpy.ndarray]:
    """EVERY array a step can write, enumerated off the object rather than listed.

    Every ndarray the ``Fields`` object holds DIRECTLY -- the twelve primaries and
    whichever auxiliaries this configuration allocated (``f_w_*``, ``fu_*``,
    ``f_cond_*``, ``f_bfast_*``) -- and every pole's per-component state (``P``,
    ``P_prev``). Enumerated so an auxiliary this file has never heard of is still
    compared. Material and coefficient tables live in dicts (``_inv_eps_components``,
    ``_condinv``, a pole's ``sigma``) that no step writes, and are left out so the
    word count is a count of words a step CAN write.

    SCRATCH IS EXCLUDED BY NAME, and that is the one judgement here:
    ``_fmp_scratch`` is the shared buffer ``displacement_minus_polarization``
    overwrites whole on every call (``fields.py:1096-1105``). It carries nothing between
    calls and legitimately ends a step holding the LAST component formed -- Ez after the
    full bracket, the source's own component after the restricted one.
    """
    out: Dict[str, numpy.ndarray] = {}
    for key, value in sorted(vars(fields).items()):
        if isinstance(value, numpy.ndarray) and "scratch" not in key:
            out[f"fields.{key}"] = value
    for index, state in enumerate(fields.polarizations):
        for attribute, value in sorted(vars(state).items()):
            if attribute == "sigma" or not isinstance(value, Mapping):
                continue
            for component, array in value.items():
                if isinstance(array, numpy.ndarray):
                    out[f"fields.polarizations[{index}].{attribute}[{component}]"] = array
    return out


def _state_words(fields: Any) -> Dict[str, numpy.ndarray]:
    """:func:`_state_arrays` as COPIED uint32 words (a view would alias the live run)."""
    out = {}
    for name, array in _state_arrays(fields).items():
        flat = numpy.array(array, copy=True).reshape(-1).view(numpy.uint8)
        out[name] = flat.view(numpy.uint32) if flat.size % 4 == 0 else flat
    return out


def _compare(label: str, left: Dict[str, numpy.ndarray],
             right: Dict[str, numpy.ndarray], tally: Any = "restricted"
             ) -> Dict[str, int]:
    assert set(left) == set(right), set(left) ^ set(right)
    differing = {}
    words = 0
    for name in sorted(left):
        assert left[name].shape == right[name].shape, name
        words += int(left[name].size)
        count = int(numpy.count_nonzero(left[name] != right[name]))
        if count:
            differing[name] = count
    if tally:
        TALLY[tally]["comparisons"] += 1
        TALLY[tally]["words"] += words
        TALLY[tally]["differing"] += sum(differing.values())
        print(f"\nbit-identity {label}: {sum(differing.values())} of {words} words differ",
              flush=True)
    return differing


# ---------------------------------------------------------------------------
# The emulated level: the driver's order, with the launch emulated by stepping
# ---------------------------------------------------------------------------

#: The fused product's launch: the sub-step in the DRIVER'S OWN ORDER
#: (``driver.py:3291-3311`` magnetic, ``:3315-3331`` electric) with the one thing a
#: launch cannot contain -- the injection -- left out. On an unfolded grid both fills
#: are inert; on a folded one they are what a folded product's REPLACES carries.
LAUNCH = {
    "B": ("step_B", "fill_symmetry_bc_B", "zero_metal_B", "fill_folded_far_ghosts_B",
          "update_H"),
    "D": ("step_D", "fill_symmetry_bc_D", "zero_metal_D", "fill_folded_far_ghosts_D",
          "update_E"),
}


def _call(name: str, fields: Any, pml: Any) -> None:
    function = getattr(stepping, name)
    if name.startswith(("step_", "update_")):
        function(fields, pml)
    else:
        function(fields)


def _launch(fields: Any, pml: Any, pair: str) -> None:
    for name in LAUNCH[pair]:
        _call(name, fields, pml)


def _inject_directly(fields: Any, sources: Sequence[Any], when: float) -> None:
    """``driver.py:3293-3294`` / ``:3321-3322``."""
    for source in sources:
        source.inject(fields, when)


def _inject_through_conductivity(fields: Any, sources: Sequence[Any],
                                 when: float) -> None:
    """THE SHIPPED CONDUCTIVE REPLAY, not a copy of it (``driver.py:3341-3449``). The
    method reads ``self.fields`` and nothing else, so a namespace carries it."""
    FdtdDriver._inject_electric_through_conductivity(
        SimpleNamespace(fields=fields), list(sources), when)


def _withdraw(fields: Any, sources: Sequence[Any]) -> None:
    for source in sources:
        getattr(source, "withdraw", lambda _fields: None)(fields)


def _reference_step(fields, pml, sources, when, dt, pair, inject) -> None:
    """The driver's order, verbatim, on the array path."""
    curl, fill, clear, far, update = LAUNCH[pair]
    _withdraw(fields, sources)
    _call(curl, fields, pml)
    inject(fields, sources, when if pair == "B" else when + 0.5 * dt)
    _call(fill, fields, pml)
    _call(clear, fields, pml)
    _call(far, fields, pml)
    _call(update, fields, pml)
    if pair == "D":
        stepping.update_P(fields, pml)


def _bracketed_step(fields, pml, sources, when, dt, pair, inject, bracket) -> None:
    """The fused pair's two consults with the driver's unconsulted passes between."""
    _, fill, clear, far, _update = LAUNCH[pair]
    _withdraw(fields, sources)
    bracket.leading()
    inject(fields, sources, when if pair == "B" else when + 0.5 * dt)
    _call(fill, fields, pml)
    _call(clear, fields, pml)
    _call(far, fields, pml)
    bracket.trailing()
    if pair == "D":
        stepping.update_P(fields, pml)


class _LaunchInner:
    def __init__(self, fields, pml, pair):
        self._fields, self._pml, self._pair = fields, pml, pair

    def run(self, *_args, **_kwargs) -> None:
        _launch(self._fields, self._pml, self._pair)


class _Counters:
    """Route and restriction counters, summed over the caches a leg used."""

    def __init__(self) -> None:
        self.caches: List[Any] = []

    def add(self, prepared: Any) -> None:
        if all(prepared is not seen for seen in self.caches):
            self.caches.append(prepared)

    def __getattr__(self, name: str) -> int:
        if name in LEGACY_COUNTERS + RESTRICTION_COUNTERS:
            return sum(int(getattr(cache, name)) for cache in self.caches)
        raise AttributeError(name)

    def entries(self) -> set:
        keys = set()
        for cache in self.caches:
            keys |= set(cache.entries)
        return keys


class _FrozenFullBracket:
    """THE UNRESTRICTED ARITHMETIC: every target of the seam, for every source."""

    def __init__(self, fields, pml, sources, pair, paths):
        self._args = (fields, pml, tuple(sources), pair, tuple(paths))
        self.repaired = 0

    def leading(self) -> None:
        fields, pml, sources, pair, paths = self._args
        self._saved = _frozen_save(fields, sources, pair, pml, paths)
        _launch(fields, pml, pair)

    def trailing(self) -> None:
        fields, pml, sources, pair, _paths = self._args
        self.repaired += _frozen_apply(fields, pml, sources, pair, self._saved)


class _PlanBracket:
    """The two shipped plan objects, as every installer puts them in a pair's slots."""

    def __init__(self, fields, pml, sources, pair, paths):
        self.plan = deposit_repair.LeadingRepairPlan(
            _LaunchInner(fields, pml, pair), fields, pml, sources, pair, paths)
        self._trailing = deposit_repair.TrailingRepairPlan("slot", self.plan, fields, pml)
        self.counters = _Counters()
        self.counters.add(self.plan.prepared)
        self.repaired = 0

    def leading(self) -> None:
        self.plan.run()

    def trailing(self) -> None:
        self._trailing.run()
        self.repaired += self.plan.repairs


class _ModuleBracket:
    """The module-level ``save`` / ``apply``, which is what the gates that bracket a
    launch by hand call; one one-shot cache per call."""

    def __init__(self, fields, pml, sources, pair, paths):
        self._args = (fields, pml, tuple(sources), pair, tuple(paths))
        self.counters = _Counters()
        self.repaired = 0

    def leading(self) -> None:
        fields, pml, sources, pair, paths = self._args
        self._saved = deposit_repair.save(fields, sources, pair, pml, paths=paths)
        self.counters.add(self._saved[deposit_repair._PREPARED_KEY])
        _launch(fields, pml, pair)

    def trailing(self) -> None:
        fields, pml, sources, pair, _paths = self._args
        self.repaired += deposit_repair.apply(fields, pml, sources, pair, self._saved)


BRACKETS = {"frozen_full": _FrozenFullBracket, "plan": _PlanBracket,
            "module": _ModuleBracket}


class Case:
    def __init__(self, name: str, pair: str, build: Callable[[], Tuple[Any, Any, Any]],
                 sources: Callable[[Any], List[Any]], *, steps: int = 6,
                 paths: Tuple[str, ...] = SPLIT, inject=_inject_directly,
                 entries: Optional[int] = None, mirror: bool = False,
                 bind: bool = True):
        self.name, self.pair, self.build, self.sources = name, pair, build, sources
        self.steps, self.paths, self.inject = steps, paths, inject
        #: ``(component, source)`` cell sets the restricted bracket holds. One per
        #: single-component source unless the case says otherwise.
        self.entries = entries
        #: The case CLAIMS every source's ``f_u`` mirror is live, and ``_run`` holds it
        #: to that: the injection then writes ``fu_<array>`` beside the primary array.
        self.mirror = mirror
        #: False ONLY where the case has broken something binding itself reads (the
        #: trimmed component map), and the source sits where nothing mirrors anyway.
        self.bind = bind


def _run(case: Case, leg: str):
    grid, fields, pml = case.build()
    sources = case.sources(grid)
    # The module's own predicate, not a class's private count: ``ContinuousSource``
    # keeps no ``_n_source_points``.
    assert sources and all(deposit_repair._deposit_index(source) is not None
                           for source in sources), (
        f"{case.name}: a source deposits nothing, so the case cannot discriminate")
    for source in sources if case.bind else ():
        # AS THE DRIVER DOES (``driver.py:2472-2473``, ``:2509-2510``, and again from
        # ``setup_pml``): a source is bound to the layer it will be stepped under, which
        # is what selects the cells its deposit mirrors into ``fu_<array>``. Unbound,
        # a source "in the PML shell" never writes the mirror at all.
        source.bind_pml(pml)
    if case.mirror:
        assert all(getattr(source, "_fu_ix", None) is not None for source in sources), (
            f"{case.name}: the f_u mirror is not live, so this case does not exercise "
            f"the second array an injection writes")
    bracket = (None if leg == "reference"
               else BRACKETS[leg](fields, pml, sources, case.pair, case.paths))
    dt = grid.dt
    words: Dict[str, numpy.ndarray] = {}
    for step in range(case.steps):
        if bracket is None:
            _reference_step(fields, pml, sources, step * dt, dt, case.pair, case.inject)
        else:
            _bracketed_step(fields, pml, sources, step * dt, dt, case.pair, case.inject,
                            bracket)
        if step < case.steps - 1:
            # THE STATE AFTER EVERY STEP, not only the last. A D-seam case here never
            # steps B, so on an ungraded axis a wrong constitutive result is overwritten
            # by the next step's and the final state would vouch for the last step alone
            # (the comment above ``CW_FREQUENCY`` is where that was measured).
            words.update({f"after step {step + 1} of {case.steps}: {name}": array
                          for name, array in _state_words(fields).items()})
    # The FINAL state keeps its bare names: it is what the null controls read.
    words.update(_state_words(fields))
    return words, bracket, sources


def _aim_at_the_wrong_component(monkeypatch) -> None:
    """THE NULL CONTROL'S PLANT: every source's repair lands on the NEXT target of its
    seam -- a real target, so every guard admits it -- and never on the one it wrote."""
    real = deposit_repair._written_targets

    def wrong(source: Any, pair: str):
        targets = deposit_repair.SEAMS[pair]["targets"]
        right = real(source, pair)
        assert right is not None and len(right) == 1, right
        return (targets[(targets.index(right[0]) + 1) % len(targets)],)

    monkeypatch.setattr(deposit_repair, "_written_targets", wrong)


def _own_arrays(sources: Sequence[Any], pair: str) -> set:
    """The stored arrays a missed repair of these sources must show up in."""
    names = set()
    for source in sources:
        target = engine_sources._ARRAY_FOR_COMPONENT[source.component]
        component = ("H" if pair == "B" else "E") + target[1]
        names |= {f"fields.{component}", f"fields.f_w_{component}"}
    return names


# --- builders ---------------------------------------------------------------


def _pole_build(complex_storage: bool = False, seed: int = 20260920):
    """SPLIT FIELD, DISPERSIVE, EVERY COMPONENT DRIVEN. ``test_deposit_repair``'s
    dispersive build puts its pole on Ez alone, so ``D - sum P`` is formed for one
    component and the shared scratch is never reused inside one repair. Here two poles
    drive Ex, Ey and Ez with different sigmas, so every component's constitutive input
    is formed in the ONE scratch (``fields.py:1096-1105``) and a repair that read it
    after another component's formation would be wrong."""
    grid = Grid(resolution=12.0, cell_size=(1.2, 1.2, 0.0), dimensions=2)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    dtype = numpy.complex64 if complex_storage else numpy.float32
    for index in range(2):
        sigma = {"Ex": 0.30 + 0.05 * index, "Ey": 0.20 if index == 0 else 0.0,
                 "Ez": 0.25 + 0.05 * index}
        fields.polarizations.append(PolarizationState(
            Susceptibility(1.0 + 0.3 * index, 0.1, "lorentzian"), sigma, grid, dtype))
    pml = PML(grid=grid, thickness=(2, 2, 0))
    fields.enable_pml_storage()
    generator = numpy.random.default_rng(seed)
    for name in SEEDED:
        array = getattr(fields, name, None)
        if array is not None:
            _seed(array, generator)
    for state in fields.polarizations:
        for attribute in ("P", "P_prev"):
            for array in (getattr(state, attribute, None) or {}).values():
                if array is not None:
                    _seed(array, generator)
    return grid, fields, pml


def _conductive_build(seed: int = 20260921):
    """SPLIT FIELD WITH A D CONDUCTIVITY: the configuration whose electric injection
    the driver routes through ``_inject_electric_through_conductivity``."""
    grid = Grid(resolution=12.0, cell_size=(1.2, 1.2, 0.0), dimensions=2)
    fields = Fields(grid=grid)
    volume = numpy.full(grid.shape, 0.2, dtype=numpy.float32)
    volume *= numpy.linspace(0.5, 1.5, grid.shape[0], dtype=numpy.float32)[:, None, None]
    fields.set_d_conductivity(volume)
    pml = PML(grid=grid, thickness=(2, 2, 0))
    fields.enable_pml_storage()
    generator = numpy.random.default_rng(seed)
    for name in SEEDED + ("f_cond_Dx", "f_cond_Dy", "f_cond_Dz"):
        array = getattr(fields, name, None)
        if array is not None:
            _seed(array, generator)
    return grid, fields, pml


def _volume_build():
    return _build(cell=(1.0, 1.0, 1.0), resolution=8.0)


def _point(pair, component, center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0)):
    return lambda grid: [_make_source(grid, pair, component, center, size)]


def _several(pair, *specs):
    return lambda grid: [_make_source(grid, pair, component, center, size)
                         for component, center, size in specs]


def _integrated(grid, component, center):
    return VolumeSource(grid=grid, component=component, center=center,
                        size=(0.0, 0.0, 0.0), amplitude=1.0,
                        envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2,
                                                  is_integrated=True))


#: NOT 1.0, and measured rather than chosen for taste. At frequency 1.0 a CW source's
#: current is purely imaginary -- so real storage deposits EXACTLY zero -- on steps 5 and
#: 17 of the 2-D builds (dt = 1/24) and on step 3 of the 3-D one (dt = 1/16), and step 5
#: is the LAST step of a six-step case. That matters because of two properties of the
#: emulated level taken together: a D-seam case never steps B, so ``D`` before each
#: injection does not depend on the ``E`` a missed repair left wrong; and Ez's
#: constitutive telescopes on a 2-D grid (z is ungraded, kps = kms = 1, so the field is
#: ``field_0 + fw_k - fw_0``). A missed Ez repair therefore shows in the FINAL state only
#: through the last step's deposit, and the null control on the conductive Ez case read
#: 0 words at six steps where it read 1 in Ez and 1 in f_w_Ez at one, two and three. At
#: 0.7 no step of the first 24 has a zero real current on any build used here (smallest
#: magnitude 7.8e-2 at dt = 1/24 and 3.9e-2 at dt = 1/16), which
#: ``test_no_cw_case_has_a_step_that_deposits_nothing`` holds every CW case to. The
#: driver level steps both halves, so a missed repair there propagates and persists.
CW_FREQUENCY = 0.7


def _continuous(component, center):
    """``ContinuousSource``: what the driver's default ``continuous`` kind builds for a
    POINT (``driver.py:2494-2500``). Its ``inject`` is its OWN write path
    (``sources.py:1742``), not the ``_inject_points`` the other three classes share."""
    return lambda grid: ContinuousSource(grid=grid, frequency=CW_FREQUENCY,
                                         component=component, center=center)


def _extended(component, center, size):
    """``ExtendedSource``: the same kind with an extent (``driver.py:2501-2506``)."""
    return lambda grid: ExtendedSource(grid=grid, frequency=CW_FREQUENCY,
                                       component=component, center=center, size=size)


def _pulse(component, center, size=(0.0, 0.0, 0.0)):
    return lambda grid: _make_source(grid, "D", component, center, size)


def _built(*makers):
    return lambda grid: [make(grid) for make in makers]


O = (0.0, 0.0, 0.0)
SHELL = (0.5, 0.5, 0.0)
#: Under ONE layer and outside the other's chunk, which is where a deposit mirrors into
#: ``f_u`` (``sources._unsplit_deposit_flags``): measured on ``_build``'s 14x14 grid, an
#: Ez deposit mirrors under the x layer and an Ex deposit under the y layer; Ey's
#: ladder axis is z, which carries no layer on a 2-D grid, so Ey never mirrors here.
UNDER_X = (0.5, 0.0, 0.0)
UNDER_Y = (0.0, 0.5, 0.0)

CASES = [
    # One per component of each seam: the three coefficient layouts ((n,1,1), (1,n,1)
    # and the ungraded (1,1,1) of a 2-D grid's z) each become the ONLY one a bracket
    # reads, so a restriction that picked the wrong axis's coefficients shows here.
    Case("D point Ex", "D", _build, _point("D", "Ex")),
    Case("D point Ey", "D", _build, _point("D", "Ey")),
    Case("D point Ez", "D", _build, _point("D", "Ez")),
    Case("D point Ez, the D spelling of the component", "D", _build, _point("D", "Dz")),
    Case("D point Ez, 24 steps", "D", _build, _point("D", "Ez"), steps=24),
    Case("D point Ex in the PML shell", "D", _build, _point("D", "Ex", SHELL)),
    Case("B point Hx", "B", _build, _point("B", "Hx")),
    Case("B point Hy", "B", _build, _point("B", "Hy")),
    Case("B point Hz", "B", _build, _point("B", "Hz")),
    Case("B point Hz, the B spelling of the component", "B", _build, _point("B", "Bz")),
    Case("B point Hz, 24 steps", "B", _build, _point("B", "Hz"), steps=24),
    Case("B point Hx in the PML shell", "B", _build, _point("B", "Hx", SHELL)),
    # A LINE and a VOLUME source: many cells, one component.
    Case("D line Ez", "D", _build, _point("D", "Ez", O, (0.6, 0.0, 0.0))),
    Case("D area Ex", "D", _build, _point("D", "Ex", O, (0.6, 0.6, 0.0))),
    Case("B line Hx", "B", _build, _point("B", "Hx", O, (0.0, 0.6, 0.0))),
    Case("D volume Ez on a 3-D grid (the (1,1,n) layout)", "D", _volume_build,
         _point("D", "Ez", O, (0.4, 0.4, 0.4))),
    Case("B volume Hy on a 3-D grid", "B", _volume_build,
         _point("B", "Hy", O, (0.4, 0.4, 0.4))),
    # complex64 storage, both seams.
    Case("complex D point Ex in the PML shell", "D",
         lambda: _build(complex_storage=True), _point("D", "Ex", SHELL)),
    Case("complex D line Ez", "D", lambda: _build(complex_storage=True),
         _point("D", "Ez", O, (0.6, 0.0, 0.0))),
    Case("complex B point Hz", "B", lambda: _build(complex_storage=True),
         _point("B", "Hz")),
    Case("complex B line Hx in the PML shell", "B", lambda: _build(complex_storage=True),
         _point("B", "Hx", SHELL, (0.0, 0.3, 0.0))),
    # SEVERAL SOURCES ON ONE SEAM.
    Case("two sources, one component, ONE CELL (a duplicate cell)", "D", _build,
         _several("D", ("Ez", O, O), ("Ez", O, O))),
    Case("two sources, one component, overlapping line and point", "D", _build,
         _several("D", ("Ez", O, (0.6, 0.0, 0.0)), ("Ez", O, O))),
    Case("two sources on DIFFERENT components", "D", _build,
         _several("D", ("Ex", O, O), ("Ez", (0.2, 0.1, 0.0), O))),
    Case("two sources on different components of ONE index triple", "D", _build,
         _several("D", ("Ex", O, O), ("Ez", O, O))),
    Case("three sources, all three components", "D", _build,
         _several("D", ("Ex", O, O), ("Ey", (0.1, 0.0, 0.0), O), ("Ez", SHELL, O))),
    Case("two magnetic sources on different components", "B", _build,
         _several("B", ("Hx", O, O), ("Hz", (0.1, 0.2, 0.0), O))),
    # THE FOLDED SEAM: the image closure is per (component, source), so restricting the
    # component must keep every image of the component that IS written.
    Case("folded, Ez on the near fill's source row", "D", _folded_build,
         lambda grid: [_folded_source(grid, MIRROR_ROW_Y, "Ez")]),
    Case("folded, Ey on the far ghost's reflect row", "D", _folded_build,
         lambda grid: [_folded_source(grid, FAR_ROW_Y, "Ey")]),
    Case("folded, Ez and Ey together", "D", _folded_build,
         lambda grid: [_folded_source(grid, MIRROR_ROW_Y, "Ez"),
                       _folded_source(grid, FAR_ROW_Y, "Ey")]),
    # DISPERSIVE. One driven component, then the shared-scratch cases.
    Case("dispersive (pole on Ez), Ez", "D", lambda: _build(dispersive=True),
         _point("D", "Ez")),
    Case("dispersive (pole on Ez), Ex: D - sum P aliases D", "D",
         lambda: _build(dispersive=True), _point("D", "Ex")),
    Case("SHARED SCRATCH, real: Ex and Ez both driven", "D", _pole_build,
         _several("D", ("Ex", O, O), ("Ez", (0.2, 0.1, 0.0), (0.3, 0.0, 0.0))), steps=12),
    Case("SHARED SCRATCH, real: all three driven, reverse source order", "D", _pole_build,
         _several("D", ("Ez", O, O), ("Ey", (0.1, 0.0, 0.0), O), ("Ex", SHELL, O)),
         steps=12),
    Case("SHARED SCRATCH, complex: Ex and Ez both driven", "D",
         lambda: _pole_build(complex_storage=True),
         _several("D", ("Ex", O, O), ("Ez", (0.2, 0.1, 0.0), (0.3, 0.0, 0.0))), steps=12),
    Case("SHARED SCRATCH, complex: one source on Ex", "D",
         lambda: _pole_build(complex_storage=True), _point("D", "Ex", SHELL), steps=12),
    # THE PLAIN OVERWRITE PATH.
    Case("plain, two poles, Ez", "D", _plain_build, _point("D", "Ez"), paths=PLAIN),
    Case("plain, the one-pole component Ey, line", "D", _plain_build,
         _point("D", "Ey", O, (0.0, 0.6, 0.0)), paths=PLAIN),
    Case("plain, no poles: D - sum P aliases D", "D", lambda: _plain_build(poles=0),
         _point("D", "Ez"), paths=PLAIN),
    Case("plain, Ex and Ez together", "D", _plain_build,
         _several("D", ("Ex", O, (0.6, 0.0, 0.0)), ("Ez", O, O)), paths=PLAIN),
    # THE CONDUCTIVE REPLAY, through the driver's own method.
    Case("conductive replay, split field, Ez", "D", _conductive_build,
         _point("D", "Ez"), inject=_inject_through_conductivity),
    Case("conductive replay, split field, Ex line and Ez point", "D", _conductive_build,
         _several("D", ("Ex", O, (0.6, 0.0, 0.0)), ("Ez", (0.2, 0.1, 0.0), O)),
         inject=_inject_through_conductivity),
    Case("conductive replay, split field, two sources on one component", "D",
         _conductive_build, _several("D", ("Ez", O, O), ("Ez", O, (0.3, 0.0, 0.0))),
         inject=_inject_through_conductivity),
    Case("conductive replay, split field, a scaled Ex beside an INTEGRATED Ez", "D",
         _conductive_build,
         lambda grid: [_make_source(grid, "D", "Ex", O, O),
                       _integrated(grid, "Ez", (0.2, 0.1, 0.0))],
         inject=_inject_through_conductivity),
    Case("conductive replay, plain path, Ex line", "D",
         lambda: _plain_build(conductivity=0.20), _point("D", "Ex", O, (0.6, 0.0, 0.0)),
         paths=PLAIN, inject=_inject_through_conductivity),
    Case("an INTEGRATED magnetic source (withdraw, then the whole dipole)", "B", _build,
         lambda grid: [_integrated(grid, "Hz", O)]),
    # THE TWO CLASSES THE DEFAULT ``continuous`` KIND BUILDS. Both are on
    # ``deposit_repair._READ_INJECTS``, and every case above builds the other two
    # (``_make_source``: GaussianPulsedSource and VolumeSource), so until 2026-09-20 an
    # edit to either ``inject`` that added a second write would have passed this file.
    # Each rides the wrong-component null control like every other case.
    Case("CW point (ContinuousSource) Ex under the y layer, f_u mirror live", "D", _build,
         _built(_continuous("Ex", UNDER_Y)), mirror=True),
    Case("CW point (ContinuousSource) Ey", "D", _build, _built(_continuous("Ey", O))),
    Case("CW point (ContinuousSource) Ez under the x layer, f_u mirror live", "D", _build,
         _built(_continuous("Ez", UNDER_X)), mirror=True),
    Case("CW point (ContinuousSource) Ez, the D spelling, 24 steps", "D", _build,
         _built(_continuous("Dz", UNDER_X)), steps=24, mirror=True),
    Case("CW line (ExtendedSource) Ex under the y layer, f_u mirror live", "D", _build,
         _built(_extended("Ex", UNDER_Y, (0.6, 0.0, 0.0))), mirror=True),
    Case("CW line (ExtendedSource) Ez under the x layer, f_u mirror live", "D", _build,
         _built(_extended("Ez", UNDER_X, (0.0, 0.6, 0.0))), mirror=True),
    Case("CW area (ExtendedSource) Ey", "D", _build,
         _built(_extended("Ey", O, (0.6, 0.6, 0.0)))),
    Case("CW volume (ExtendedSource) Ez on a 3-D grid", "D", _volume_build,
         _built(_extended("Ez", O, (0.4, 0.4, 0.4)))),
    Case("complex CW point Ez under the x layer, f_u mirror live", "D",
         lambda: _build(complex_storage=True), _built(_continuous("Ez", UNDER_X)),
         mirror=True),
    Case("complex CW line Ex under the y layer, f_u mirror live", "D",
         lambda: _build(complex_storage=True),
         _built(_extended("Ex", UNDER_Y, (0.6, 0.0, 0.0))), mirror=True),
    Case("a CW point and a CW line on different components", "D", _build,
         _built(_continuous("Ez", UNDER_X), _extended("Ex", UNDER_Y, (0.6, 0.0, 0.0))),
         steps=12, mirror=True),
    Case("SHARED SCRATCH, real: a CW point on Ex beside a CW line on Ez", "D", _pole_build,
         _built(_continuous("Ex", UNDER_Y), _extended("Ez", UNDER_X, (0.0, 0.6, 0.0))),
         steps=12, mirror=True),
    Case("plain, CW point Ez", "D", _plain_build, _built(_continuous("Ez", O)),
         paths=PLAIN),
    Case("plain, CW line Ex", "D", _plain_build,
         _built(_extended("Ex", O, (0.6, 0.0, 0.0))), paths=PLAIN),
    Case("conductive replay, split field, CW point Ez under the x layer", "D",
         _conductive_build, _built(_continuous("Ez", UNDER_X)),
         inject=_inject_through_conductivity, mirror=True),
    Case("conductive replay, split field, CW line Ex under the y layer", "D",
         _conductive_build, _built(_extended("Ex", UNDER_Y, (0.6, 0.0, 0.0))),
         inject=_inject_through_conductivity, mirror=True),
    Case("conductive replay, split field, a CW point, a CW line and a pulse on ONE "
         "component", "D", _conductive_build,
         _built(_continuous("Ez", UNDER_X), _extended("Ez", O, (0.0, 0.6, 0.0)),
                _pulse("Ez", O)),
         inject=_inject_through_conductivity),
    Case("conductive replay, plain path, CW point Ez", "D",
         lambda: _plain_build(conductivity=0.20), _built(_continuous("Ez", O)),
         paths=PLAIN, inject=_inject_through_conductivity),
]
IDS = [case.name for case in CASES]


def _expected_counters(case: Case, sources: Sequence[Any]) -> Dict[str, int]:
    entries = len(sources) if case.entries is None else case.entries
    split = case.paths == SPLIT
    return {"linear_saves": case.steps * entries if split else 0, "cells_saves": 0,
            "linear_repairs": case.steps * entries, "cells_repairs": 0,
            "restricted_sources": case.steps * len(sources), "restriction_fallbacks": 0}


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_the_restricted_repair_is_word_identical_to_the_full_bracket(case):
    """Three answers from one state: the driver's own order, the frozen FULL bracket,
    and the live restricted module on both of its routes."""
    reference, _none, _sources = _run(case, "reference")
    full, frozen, _sources = _run(case, "frozen_full")
    assert frozen.repaired, f"{case.name}: the full bracket repaired no cell"
    assert not _compare(f"[{case.name}] driver order vs frozen full bracket",
                        reference, full, "reference"), case.name
    for leg in ("plan", "module"):
        live, bracket, sources = _run(case, leg)
        assert not _compare(f"[{case.name}] frozen full bracket vs restricted {leg}",
                            full, live), (case.name, leg)
        want = _expected_counters(case, sources)
        got = {name: getattr(bracket.counters, name) for name in want}
        assert got == want, (case.name, leg, got, want)
        # EXACTLY the written targets, and no others.
        written = {(("H" if case.pair == "B" else "E")
                    + engine_sources._ARRAY_FOR_COMPONENT[source.component][1], index)
                   for index, source in enumerate(sources)}
        assert bracket.counters.entries() == written, (case.name, leg)
        assert 0 < bracket.repaired < frozen.repaired, (
            f"{case.name}: the restricted {leg} route repaired {bracket.repaired} cells "
            f"against the full bracket's {frozen.repaired}; it is not restricted")


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_repairing_the_wrong_component_diverges(case, monkeypatch):
    """THE NULL CONTROL. Without it every case above proves only that two legs agree."""
    full, _frozen, _sources = _run(case, "frozen_full")
    _aim_at_the_wrong_component(monkeypatch)
    wrong, bracket, sources = _run(case, "plan")
    assert bracket.counters.restricted_sources == case.steps * len(sources), (
        "the plant was not reached, so this leg measured the shipped restriction")
    differing = _compare(f"[{case.name}] NULL CONTROL", full, wrong, tally=False)
    assert differing, (
        f"{case.name}: a repair aimed at the WRONG component matched the full bracket, "
        f"so this case cannot tell a correct restriction from one that repairs nothing")
    own = _own_arrays(sources, case.pair)
    stored = {name for name in own if name in full}
    assert stored & set(differing), (case.name, sorted(differing), sorted(own))


CW_CASES = [case for case in CASES if "CW" in case.name]


@pytest.mark.parametrize("case", CW_CASES, ids=[case.name for case in CW_CASES])
def test_no_cw_case_has_a_step_that_deposits_nothing(case):
    """A step whose deposit is exactly zero cannot show a missed repair, and at frequency
    1.0 the sixth step of a six-step case is one (the comment above ``CW_FREQUENCY``)."""
    grid, _fields, _pml = case.build()
    continuous = [source for source in case.sources(grid)
                  if type(source) in (ContinuousSource, ExtendedSource)]
    assert continuous, case.name
    for source in continuous:
        for step in range(case.steps):
            current = source.current(step * grid.dt + 0.5 * grid.dt, grid.dt)
            assert current.real != 0.0, (case.name, step, current)


@pytest.mark.parametrize("pair,component,build,paths", [
    ("D", "Ez", _build, SPLIT), ("D", "Ex", _build, SPLIT), ("B", "Hz", _build, SPLIT),
    ("D", "Ex", _pole_build, SPLIT), ("D", "Ez", _plain_build, PLAIN),
], ids=["D_Ez", "D_Ex", "B_Hz", "D_Ex_dispersive", "D_Ez_plain"])
def test_the_null_control_diverges_in_the_sources_own_component_and_nowhere_else(
        monkeypatch, pair, component, build, paths):
    """ONE STEP, so nothing has propagated: the wrong-component repair must leave the
    written component's arrays wrong -- and must leave the component it DID repair
    bit-identical, because that is the over-coverage argument the restriction rests on
    (``apply`` at a cell whose input did not change reproduces the launch's value)."""
    case = Case("localised null control", pair, build, _point(pair, component), steps=1,
                paths=paths)
    full, _frozen, _sources = _run(case, "frozen_full")
    _aim_at_the_wrong_component(monkeypatch)
    wrong, _bracket, sources = _run(case, "plan")
    differing = _compare("localised null control", full, wrong, tally=False)
    own = _own_arrays(sources, pair)
    field = f"fields.{component}"
    assert field in differing, differing
    allowed = own | {name for name in full if ".polarizations" in name}
    assert set(differing) <= allowed, (
        f"the wrong-component repair moved {sorted(set(differing) - allowed)}: repairing "
        f"a component whose input did not change is NOT bit-identical here, which is "
        f"the premise of the whole restriction")


# ---------------------------------------------------------------------------
# KNOWN VALUE: one hand-built deposit cell per seam, real and complex64
# ---------------------------------------------------------------------------


def _discriminating(kp, generator):
    """``(f_old, fresh)`` float32 for which ``f_old + kp*fresh`` rounded in TWO steps
    differs from the same expression rounded once -- the value a fused multiply-add
    would give. A repair that contracted the recurrence is then a different word."""
    for _attempt in range(20000):
        f_old = numpy.float32(generator.standard_normal())
        fresh = numpy.float32(generator.standard_normal())
        product = numpy.float32(kp * fresh)
        separate = numpy.float32(f_old + product)
        fused = numpy.float32(numpy.float64(f_old)
                              + numpy.float64(kp) * numpy.float64(fresh))
        if separate != fused:
            return f_old, fresh
    raise AssertionError("no discriminating operand pair found")


@pytest.mark.parametrize("pair,component,complex_storage", [
    ("D", "Ex", False), ("D", "Ex", True), ("B", "Hx", False), ("B", "Hx", True)],
    ids=["D_real", "D_complex64", "B_real", "B_complex64"])
def test_known_value_at_one_hand_built_deposit_cell(pair, component, complex_storage):
    """One cell in the x absorber, where ``kps`` / ``kms`` are graded; every operand
    written by hand; the repaired words equal the recurrence evaluated on scalars, equal
    the frozen full bracket's, and differ from the fused-multiply-add value."""
    cell = (1, 5, 0)
    injected = ("B" if pair == "B" else "D") + component[1]
    generator = numpy.random.default_rng(20260920)

    def prepare():
        _grid, fields, pml = _build(complex_storage=complex_storage)
        source = _make_source(fields.grid, pair, component, O, O)
        source._point_ix, source._point_iy, source._point_iz = (
            numpy.array([cell[0]]), numpy.array([cell[1]]), numpy.array([cell[2]]))
        return fields, pml, source

    fields, pml, source = prepare()
    kps, kms = stepping._constitutive_coefficients(pml, "x", half_integer=(pair == "D"))
    kp = numpy.float32(numpy.asarray(kps)[cell[0], 0, 0])
    km = numpy.float32(numpy.asarray(kms)[cell[0], 0, 0])
    assert kp != 1.0 and km != 1.0 and kp != km, (kp, km)
    if pair == "D":
        assert numpy.asarray(fields.inverse_epsilon_for(component))[cell] == 1.0
    f_old, fresh = _discriminating(kp, generator)
    fw_old = numpy.float32(generator.standard_normal())
    if complex_storage:
        f_imag, fresh_imag = _discriminating(kp, generator)
        f_old = numpy.complex64(complex(f_old, f_imag))
        fresh = numpy.complex64(complex(fresh, fresh_imag))
        fw_old = numpy.complex64(complex(fw_old, generator.standard_normal()))

    def drive(fields, pml, source, save, apply):
        getattr(fields, component)[cell] = f_old
        getattr(fields, "f_w_" + component)[cell] = fw_old
        saved = save(fields, [source], pair, pml)
        # The launch wrote its pre-injection answer, whatever it was; the driver then
        # deposited. Both by hand: the repair must overwrite the first from the second.
        getattr(fields, component)[cell] = 7.0
        getattr(fields, "f_w_" + component)[cell] = -3.0
        getattr(fields, injected)[cell] = fresh
        before = _state_words(fields)
        count = apply(fields, pml, [source], pair, saved)
        return before, count

    before, count = drive(fields, pml, source, deposit_repair.save, deposit_repair.apply)
    assert count == 1, count
    # THE RECURRENCE ON SCALARS, each operation rounded on its own, in
    # ``_apply_constitutive_pml``'s operand order (``stepping.py:2083-2095``).
    expected = (f_old + kp * fresh) - km * fw_old
    dtype = numpy.complex64 if complex_storage else numpy.float32
    assert type(expected) is dtype, type(expected)

    def words(value):
        return numpy.asarray(value, dtype=dtype).reshape(-1).view(numpy.uint32).tolist()

    assert words(getattr(fields, component)[cell]) == words(expected)
    assert words(getattr(fields, "f_w_" + component)[cell]) == words(fresh)
    fused = numpy.float32(numpy.float64(numpy.real(f_old))
                          + numpy.float64(kp) * numpy.float64(numpy.real(fresh)))
    assert numpy.float32(numpy.real(f_old) + numpy.float32(kp * numpy.real(fresh))) != fused

    # NOTHING ELSE MOVED. The other two components at this cell were not written by the
    # source, so the restricted repair leaves them exactly as the launch left them.
    after = _state_words(fields)
    moved = {name for name in after if (after[name] != before[name]).any()}
    assert moved == {f"fields.{component}", f"fields.f_w_{component}"}, moved
    for name in moved:
        changed = int(numpy.count_nonzero(after[name] != before[name]))
        assert 1 <= changed <= (2 if complex_storage else 1), (name, changed)

    # ...and the frozen FULL bracket, from the same hand-built state, writes the same
    # words into the written component.
    twin_fields, twin_pml, twin_source = prepare()
    drive(twin_fields, twin_pml, twin_source,
          lambda f, s, p, layer: _frozen_save(f, s, p, layer, SPLIT), _frozen_apply)
    twin = _state_words(twin_fields)
    for name in (f"fields.{component}", f"fields.f_w_{component}"):
        assert not numpy.count_nonzero(after[name] != twin[name]), name


def test_known_value_on_the_plain_overwrite_path():
    """``E = (D - sum P) * inv_eps`` at one hand-built cell, and nothing else moves."""
    cell = (4, 6, 0)
    _grid, fields, pml = _plain_build()
    source = _make_source(fields.grid, "D", "Ey", O, O)
    source._point_ix, source._point_iy, source._point_iz = (
        numpy.array([cell[0]]), numpy.array([cell[1]]), numpy.array([cell[2]]))
    saved = deposit_repair.save(fields, [source], "D", pml, paths=PLAIN)
    fields.Ey[cell] = 7.0
    fields.Dy[cell] = numpy.float32(0.625)
    drive = numpy.float32(fields.displacement_minus_polarization("Ey")[cell])
    inverse = numpy.float32(numpy.asarray(fields.inverse_epsilon_for("Ey"))[cell])
    assert inverse != 1.0, "the background permittivity makes the product non-trivial"
    before = _state_words(fields)
    assert deposit_repair.apply(fields, pml, [source], "D", saved) == 1
    assert fields.Ey[cell] == numpy.float32(drive * inverse)
    after = _state_words(fields)
    assert {name for name in after if (after[name] != before[name]).any()} == {"fields.Ey"}


# ---------------------------------------------------------------------------
# DEGENERATE: nothing to repair, and every way the written array can be unknowable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pair", ["B", "D"])
def test_an_empty_seam_saves_the_stamps_and_repairs_nothing(pair):
    _grid, fields, pml = _build()
    before = _state_words(fields)
    saved = deposit_repair.save(fields, [], pair, pml)
    assert set(saved) == {deposit_repair._PATH_KEY, deposit_repair._PREPARED_KEY,
                          deposit_repair._TARGETS_KEY}
    assert saved[deposit_repair._TARGETS_KEY] == {}
    assert deposit_repair.apply(fields, pml, [], pair, saved) == 0
    assert not _compare("empty seam", before, _state_words(fields), tally=False)
    prepared = saved[deposit_repair._PREPARED_KEY]
    assert all(getattr(prepared, name) == 0
               for name in LEGACY_COUNTERS + RESTRICTION_COUNTERS)


def test_a_source_that_deposits_into_no_cell_is_skipped_not_counted():
    _grid, fields, pml = _build()
    source = _make_source(fields.grid, "D", "Ez", O, O)
    source._point_ix = source._point_iy = source._point_iz = None
    saved = deposit_repair.save(fields, [source], "D", pml)
    assert saved[deposit_repair._TARGETS_KEY] == {}
    prepared = saved[deposit_repair._PREPARED_KEY]
    assert (prepared.restricted_sources, prepared.restriction_fallbacks) == (0, 0)
    assert deposit_repair.apply(fields, pml, [source], "D", saved) == 0


class _Subclassed(GaussianPulsedSource):
    """A class this module has never read: its ``inject`` could write anything."""


#: The component map as the engine ships it, held so a case that trims it can still
#: BUILD its source first (construction validates the component against this map).
SHIPPED_MAP = dict(engine_sources._ARRAY_FOR_COMPONENT)


def _unmapped(monkeypatch):
    trimmed = {name: array for name, array in SHIPPED_MAP.items() if name != "Ez"}
    monkeypatch.setattr(engine_sources, "_ARRAY_FOR_COMPONENT", trimmed)


def _retype(source):
    source.__class__ = _Subclassed
    return source


def _non_string_component(source):
    object.__setattr__(source, "component", ("Ez",))
    return source


#: The engine's own ``inject`` functions, held from import, so a case that REBINDS one
#: always wraps the function that was read and never a wrapper an earlier leg left.
READ_INJECT = {kind: kind.__dict__["inject"] for kind in (
    ContinuousSource, ExtendedSource, GaussianPulsedSource, VolumeSource)}


def _write_Dx_too(source, fields) -> None:
    """THE WRITE THE ALLOW-LIST EXISTS TO CATCH: a second array, at the deposit cells."""
    fields.Dx[source._point_ix, source._point_iy, source._point_iz] -= numpy.float32(0.125)


def _rebind_on_the_instance(source, extra=None):
    """``parity/meep_gpu/probe_dispatch_seam_sweep.py:176-191``'s shape: a plain closure
    over the BOUND real method, stored in the instance's own dict."""
    real = source.inject

    def wrapped(fields, when):
        out = real(fields, when)
        if extra is not None:
            extra(source, fields)
        return out

    source.inject = wrapped
    return source


def _rebind_on_the_class(monkeypatch, source, extra=None, wraps=False):
    """``parity/meep_gpu/gate_metal_complex_conductive_fused_pair.py:958-970``'s shape:
    ``cls.inject = wrapper`` around the class's real function. ``wraps`` dresses the
    wrapper in the real function's ``__name__`` / ``__qualname__`` / ``__module__``."""
    real = READ_INJECT[type(source)]

    def inner(self, fields, when, *args, **kwargs):
        out = real(self, fields, when, *args, **kwargs)
        if extra is not None:
            extra(self, fields)
        return out

    monkeypatch.setattr(type(source), "inject",
                        functools.wraps(real)(inner) if wraps else inner)
    return source


#: why -> (what is done to the built source, whether ``_run`` may bind it afterwards).
FAIL_CLOSED = {
    "the component is absent from the component map":
        (lambda monkeypatch, source: (_unmapped(monkeypatch), source)[1], False),
    "a source class whose inject this module has not read":
        (lambda monkeypatch, source: _retype(source), True),
    # THE INJECT THAT WILL RUN IS NOT THE FUNCTION THAT WAS READ. The type is still the
    # engine's own class, which is all the allow-list looked at until 2026-09-20.
    "inject rebound on the instance, delegating":
        (lambda monkeypatch, source: _rebind_on_the_instance(source), True),
    "inject rebound on the instance, and writing Dx too":
        (lambda monkeypatch, source: _rebind_on_the_instance(source, _write_Dx_too), True),
    "inject rebound on the class, delegating":
        (lambda monkeypatch, source: _rebind_on_the_class(monkeypatch, source), True),
    "inject rebound on the class, and writing Dx too":
        (lambda monkeypatch, source: _rebind_on_the_class(monkeypatch, source,
                                                          _write_Dx_too), True),
    "inject rebound on the class under functools.wraps":
        (lambda monkeypatch, source: _rebind_on_the_class(monkeypatch, source,
                                                          _write_Dx_too, wraps=True), True),
}


@pytest.mark.parametrize("why", sorted(FAIL_CLOSED))
def test_an_unestablished_written_array_fails_closed_to_all_three_and_is_counted(
        monkeypatch, why):
    """FAIL CLOSED, COUNTED, AND STILL EXACT. The full bracket is what the module
    shipped before the restriction, so falling back to it is the known-good answer."""
    steps = 6

    mutate, bind = FAIL_CLOSED[why]

    def sources(grid):
        monkeypatch.setattr(engine_sources, "_ARRAY_FOR_COMPONENT", SHIPPED_MAP)
        return [mutate(monkeypatch, _make_source(grid, "D", "Ez", O, O))]

    case = Case(why, "D", _build, sources, steps=steps, bind=bind)
    reference, _none, _sources = _run(case, "reference")
    full, _frozen, _sources = _run(case, "frozen_full")
    # The driver's own order first: "identical to the full bracket" is only the
    # known-good answer if the full bracket IS the driver's order on this source.
    assert not _compare(f"[fail closed: {why}] driver order vs frozen full bracket",
                        reference, full, "reference")
    for leg in ("plan", "module"):
        live, bracket, _sources = _run(case, leg)
        assert not _compare(f"[fail closed: {why}] frozen full bracket vs {leg}", full,
                            live)
        counters = bracket.counters
        assert counters.restriction_fallbacks == steps, counters.restriction_fallbacks
        assert counters.restricted_sources == 0
        assert counters.entries() == {("Ex", 0), ("Ey", 0), ("Ez", 0)}
        assert counters.linear_saves == counters.linear_repairs == 3 * steps


class _Stub:
    """Publishes an index and a component, and is not an engine source class."""

    field_type = "D"
    component = "Ez"

    def __init__(self):
        self._point_ix, self._point_iy, self._point_iz = (
            numpy.array([3]), numpy.array([4]), numpy.array([0]))


@pytest.mark.parametrize("source,pair,why", [
    (lambda grid: _Stub(), "D", "not an engine source class"),
    (lambda grid: _non_string_component(_make_source(grid, "D", "Ez", O, O)), "D",
     "a component that is not a string"),
    (lambda grid: _make_source(grid, "D", "Ez", O, O), "B",
     "an electric source handed to the magnetic seam"),
    (lambda grid: _make_source(grid, "B", "Hz", O, O), "D",
     "a magnetic source handed to the electric seam"),
], ids=["stub_class", "non_string_component", "electric_on_B", "magnetic_on_D"])
def test_an_ambiguous_source_keeps_the_full_bracket(source, pair, why):
    _grid, fields, pml = _build()
    built = source(fields.grid)
    assert deposit_repair._written_targets(built, pair) is None, why
    saved = deposit_repair.save(fields, [built], pair, pml)
    prepared = saved[deposit_repair._PREPARED_KEY]
    assert (prepared.restricted_sources, prepared.restriction_fallbacks) == (0, 1), why
    assert {key[0] for key in prepared.entries} == {
        target[0] for target in deposit_repair.SEAMS[pair]["targets"]}, why
    assert saved[deposit_repair._TARGETS_KEY][0] == deposit_repair.SEAMS[pair]["targets"]


@pytest.mark.parametrize("answer", [
    (), (("Hz", "Bz", "z"),), (("Ez", "Dz", "z"), ("Ez", "Dz", "z")), "Ez"],
    ids=["empty", "another_seams_target", "a_repeated_target", "not_a_tuple_of_targets"])
def test_a_lookup_that_answers_nonsense_keeps_the_full_bracket(monkeypatch, answer):
    """The lookup is a seam a control patches, so what comes back is CHECKED: anything
    that is not a non-empty, repeat-free selection of this seam's own targets is a
    fallback, never a narrower-than-justified bracket."""
    monkeypatch.setattr(deposit_repair, "_written_targets", lambda source, pair: answer)
    _grid, fields, pml = _build()
    source = _make_source(fields.grid, "D", "Ez", O, O)
    saved = deposit_repair.save(fields, [source], "D", pml)
    prepared = saved[deposit_repair._PREPARED_KEY]
    assert (prepared.restricted_sources, prepared.restriction_fallbacks) == (0, 1)
    assert len(prepared.entries) == 3


def _answer_by_type_alone(source: Any, pair: str):
    """WHAT THE ALLOW-LIST ANSWERED BEFORE IT LOOKED AT ``inject`` (this module at sha256
    ``53ba484c1f96``): the exact engine class and its component, nothing else."""
    if type(source).__name__ not in deposit_repair._READ_INJECTS:
        return None
    written = engine_sources._ARRAY_FOR_COMPONENT[source.component]
    return tuple(target for target in deposit_repair.SEAMS[pair]["targets"]
                 if target[1] == written)


@pytest.mark.parametrize("where", ["instance", "class"])
def test_a_rebound_inject_that_writes_a_second_array_is_repaired_in_full(monkeypatch,
                                                                         where):
    """THE SILENT CASE, WITH ITS NULL CONTROL. An exact-class source whose ``inject`` was
    rebound to write Dx beside Dz: restricted to Ez it leaves Ex and f_w_Ex computed from
    a Dx the launch never saw. The full bracket this module shipped until 2026-09-20 got
    it right only by over-covering; the guard in ``_written_targets`` gets it right by
    refusing to narrow an ``inject`` it has not read."""
    steps = 6

    def sources(grid):
        source = _make_source(grid, "D", "Ez", O, O)
        if where == "instance":
            return [_rebind_on_the_instance(source, _write_Dx_too)]
        return [_rebind_on_the_class(monkeypatch, source, _write_Dx_too)]

    case = Case(f"rebound on the {where}", "D", _build, sources, steps=steps)
    reference, _none, _sources = _run(case, "reference")
    full, _frozen, _sources = _run(case, "frozen_full")
    assert not _compare(f"[inject rebound on the {where}] driver order vs frozen full "
                        f"bracket", reference, full, "reference")
    for leg in ("plan", "module"):
        live, bracket, _sources = _run(case, leg)
        assert not _compare(f"[inject rebound on the {where}] frozen full bracket vs "
                            f"{leg}", full, live), (where, leg)
        assert bracket.counters.restriction_fallbacks == steps
        assert bracket.counters.restricted_sources == 0
    # THE NULL CONTROL: the same source narrowed by its type alone IS wrong, and wrong in
    # the array the rebound inject wrote -- so the legs above can see the difference.
    with monkeypatch.context() as patch:
        patch.setattr(deposit_repair, "_written_targets", _answer_by_type_alone)
        narrowed, bracket, _sources = _run(case, "plan")
    assert bracket.counters.restricted_sources == steps, "the plant was not reached"
    differing = _compare(f"[inject rebound on the {where}] NULL CONTROL", reference,
                         narrowed, tally=False)
    assert {"fields.Ex", "fields.f_w_Ex"} <= set(differing), differing


def test_what_the_lookup_answers_for_each_spelling_of_a_rebound_inject(monkeypatch):
    _grid, fields, _pml = _build()
    ez = deposit_repair.SEAMS["D"]["targets"][2]

    def fresh():
        return _make_source(fields.grid, "D", "Ez", O, O)

    assert deposit_repair._written_targets(fresh(), "D") == (ez,)
    assert deposit_repair._written_targets(_rebind_on_the_instance(fresh()), "D") is None
    # PUT BACK THE WAY THE SEAM PROBE PUTS IT BACK (``probe_dispatch_seam_sweep.py:195``):
    # ``setattr(source, "inject", <the bound real method>)``, which leaves an instance
    # attribute behind -- but one that runs the function that was read, on this source.
    source = fresh()
    real = source.inject
    _rebind_on_the_instance(source)
    source.inject = real
    assert "inject" in vars(source)
    assert deposit_repair._written_targets(source, "D") == (ez,)
    # Another source's bound method is NOT this source's inject: it writes where, and
    # what, the other one says.
    source = fresh()
    source.inject = fresh().inject
    assert deposit_repair._written_targets(source, "D") is None
    # Deleted outright, the class's function is what runs again.
    source = _rebind_on_the_instance(fresh())
    del source.inject
    assert deposit_repair._written_targets(source, "D") == (ez,)
    for wraps in (False, True):
        with monkeypatch.context() as patch:
            source = _rebind_on_the_class(patch, fresh(), wraps=wraps)
            assert deposit_repair._written_targets(source, "D") is None, wraps
            if wraps:  # dressed as the real one in every attribute ``wraps`` copies
                rebound = type(source).__dict__["inject"]
                real_inject = READ_INJECT[type(source)]
                assert rebound.__qualname__ == real_inject.__qualname__
                assert rebound.__module__ == real_inject.__module__
        assert deposit_repair._written_targets(source, "D") == (ez,), "the patch leaked"
    # One engine class's function on another engine class: both were read, neither as
    # the inject of the class now holding it.
    with monkeypatch.context() as patch:
        patch.setattr(GaussianPulsedSource, "inject", READ_INJECT[ExtendedSource])
        assert deposit_repair._written_targets(fresh(), "D") is None


@pytest.mark.parametrize("kind", sorted(READ_INJECT, key=lambda kind: kind.__name__),
                         ids=lambda kind: kind.__name__)
def test_every_listed_class_holds_the_inject_the_guard_looks_for(kind):
    """The guard reads the function out of the class's OWN dict and asks that it was
    defined in the sources module under that class's name. A class that came to INHERIT
    its ``inject``, or a rename, would turn every source of that class into a counted
    fallback -- sound, and three entries where one was possible -- so it is pinned."""
    assert kind.__name__ in deposit_repair._READ_INJECTS
    inject = kind.__dict__.get("inject")
    assert isinstance(inject, types.FunctionType)
    assert inject.__globals__ is vars(engine_sources)
    assert inject.__qualname__ == f"{kind.__name__}.inject"
    assert set(deposit_repair._READ_INJECTS) == {kind.__name__ for kind in READ_INJECT}


def test_a_scalar_part_index_is_restricted_and_exact():
    """A deposit index whose parts are SCALARS rather than arrays. ``repair_cells``
    reshapes every part to one dimension, so the set is still spelled linearly; what is
    pinned is that the restriction and the arithmetic survive the odd spelling."""
    cell = (5, 6, 0)

    def sources(grid):
        source = _make_source(grid, "D", "Ez", O, O)
        source._point_ix, source._point_iy, source._point_iz = cell
        source._point_amps = numpy.asarray(source._point_amps).reshape(-1)[0]
        return [source]

    case = Case("scalar index", "D", _build, sources)
    full, _frozen, _sources = _run(case, "frozen_full")
    live, bracket, _sources = _run(case, "plan")
    assert not _compare("[scalar-part index] frozen full bracket vs restricted plan",
                        full, live)
    counters = bracket.counters
    assert counters.entries() == {("Ez", 0)}
    assert (counters.linear_saves, counters.cells_saves) == (case.steps, 0)
    assert (counters.linear_repairs, counters.cells_repairs) == (case.steps, 0)
    assert bracket.repaired == case.steps, bracket.repaired


# ---------------------------------------------------------------------------
# SCALING: entries per step follow the SOURCES, never the grid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("specs,entries", [
    ((("Ez", O, O),), 1),
    ((("Ex", O, O), ("Ez", O, O)), 2),
    ((("Ex", O, O), ("Ey", O, O), ("Ez", O, O)), 3),
    ((("Ez", O, O), ("Ez", (0.1, 0.1, 0.0), O)), 2),
], ids=["one_source", "two_components", "three_components", "two_sources_one_component"])
def test_entries_per_step_equal_the_written_components_at_every_grid_size(specs, entries):
    per_step = {}
    for resolution in (8.0, 16.0, 32.0):
        _grid, fields, pml = _build(cell=(2.0, 2.0, 0.0), resolution=resolution)
        bracket = _PlanBracket(fields, pml, _several("D", *specs)(fields.grid), "D", SPLIT)
        for _step in range(4):
            bracket.leading()
            bracket.trailing()
        counters = bracket.counters
        per_step[int(fields.Ez.size)] = (
            len(counters.entries()), counters.linear_saves / 4,
            counters.linear_repairs / 4, counters.restriction_fallbacks)
    assert len(per_step) == 3, "the three grids are not three sizes"
    assert set(per_step.values()) == {(entries, entries, entries, 0)}, per_step


@pytest.mark.parametrize("specs,entries", [
    ((("Ex", O, O),), 1), ((("Ex", O, O), ("Ez", O, O)), 2),
    ((("Ez", O, O), ("Ez", (0.1, 0.1, 0.0), O)), 2)],
    ids=["one_component", "two_components", "two_sources_one_component"])
def test_the_linear_route_forms_D_minus_P_at_the_deposit_cells_only(
        monkeypatch, specs, entries):
    """The one cost of the bracket that GREW with the grid: on a dispersive row every
    ``displacement_minus_polarization`` call is ``1 + poles`` whole-volume passes
    (``fields.py:1096-1105``). The full bracket made three, the component-restricted
    one made one per DISTINCT written component, and the sparse linear route makes
    NONE: ``_fresh_at`` gathers D and each contributing P at the deposit cells and
    forms the differences there, once per (component, source) entry, so nothing the
    trailing bracket does on a C-contiguous target scales with the grid."""
    _grid, fields, pml = _pole_build()
    bracket = _PlanBracket(fields, pml, _several("D", *specs)(fields.grid), "D", SPLIT)
    bracket.leading()
    whole, sparse = [], []
    real = type(fields).displacement_minus_polarization
    monkeypatch.setattr(type(fields), "displacement_minus_polarization",
                        lambda self, component: whole.append(component)
                        or real(self, component))
    fresh_at = deposit_repair._fresh_at
    monkeypatch.setattr(deposit_repair, "_fresh_at",
                        lambda fields_, component, linear:
                        sparse.append((component, int(linear.size)))
                        or fresh_at(fields_, component, linear))
    bracket.trailing()
    assert whole == [], f"the linear route formed D - sum P over the whole grid: {whole}"
    assert len(sparse) == entries, sparse
    assert all(count < fields.Ez.size for _c, count in sparse), (
        f"a gather covered the whole grid ({fields.Ez.size} cells): {sparse}")


@pytest.mark.parametrize("specs,formations", [
    ((("Ex", O, O),), 1), ((("Ex", O, O), ("Ez", O, O)), 2),
    ((("Ez", O, O), ("Ez", (0.1, 0.1, 0.0), O)), 1)],
    ids=["one_component", "two_components", "two_sources_one_component"])
def test_the_cells_route_forms_the_whole_volume_once_per_written_component(
        monkeypatch, specs, formations):
    """The fallback for a target that cannot take a flat index keeps the restricted
    bracket's contract: one whole-volume ``D - sum P`` per DISTINCT written component,
    whatever the number of sources, and no sparse formation at all."""
    _grid, fields, pml = _pole_build()
    bracket = _PlanBracket(fields, pml, _several("D", *specs)(fields.grid), "D", SPLIT)
    bracket.leading()
    # Refuse the flat view: every entry takes the cells route.
    monkeypatch.setattr(deposit_repair, "_flat", lambda array, shape: None)
    whole, sparse = [], []
    real = type(fields).displacement_minus_polarization
    monkeypatch.setattr(type(fields), "displacement_minus_polarization",
                        lambda self, component: whole.append(component)
                        or real(self, component))
    monkeypatch.setattr(deposit_repair, "_fresh_at",
                        lambda *a: sparse.append(a) or pytest.fail("sparse form on the cells route"))
    bracket.trailing()
    assert len(whole) == formations == len(set(whole)), whole
    assert sparse == []


# ---------------------------------------------------------------------------
# SHAPE: what the saved state, the cache and the two signatures look like
# ---------------------------------------------------------------------------


def test_the_reserved_keys_and_the_saved_state_of_one_source():
    assert deposit_repair._PATH_KEY == ("__repair_path__",)
    assert deposit_repair._PREPARED_KEY == ("__prepared_cells__",)
    assert deposit_repair._TARGETS_KEY == ("__source_targets__",)
    _grid, fields, pml = _build()
    source = _make_source(fields.grid, "D", "Ez", O, O)
    saved = deposit_repair.save(fields, [source], "D", pml)
    assert set(saved) == {
        deposit_repair._PATH_KEY, deposit_repair._PREPARED_KEY,
        deposit_repair._TARGETS_KEY, ("Ez", 0, "cells"), ("Ez", 0, "field"),
        ("Ez", 0, "fw")}
    assert saved[deposit_repair._TARGETS_KEY] == {0: (("Ez", "Dz", "z"),)}
    assert set(saved[deposit_repair._PREPARED_KEY].entries) == {("Ez", 0)}
    plain_fields = _plain_build()[1]
    plain_source = _make_source(plain_fields.grid, "D", "Ex", O, O)
    plain = deposit_repair.save(plain_fields, [plain_source], "D", None, paths=PLAIN)
    assert set(plain) == {
        deposit_repair._PATH_KEY, deposit_repair._PREPARED_KEY,
        deposit_repair._TARGETS_KEY, ("Ex", 0, "cells")}


def test_the_cache_keeps_the_four_route_counters_and_gains_two():
    slots = deposit_repair._PreparedCells.__slots__
    assert set(LEGACY_COUNTERS) <= set(slots), (
        "parity/meep_gpu/bench_fused_products.REPAIR_COUNTERS reads these four by name")
    assert set(RESTRICTION_COUNTERS) <= set(slots)
    assert "entries" in slots
    fresh = deposit_repair._PreparedCells()
    assert fresh.entries == {}
    assert all(getattr(fresh, name) == 0
               for name in LEGACY_COUNTERS + RESTRICTION_COUNTERS)


def test_save_and_apply_keep_their_signatures():
    """``apply`` is monkeypatched at a FIXED five-argument arity by
    ``parity/meep_gpu/probe_dispatch_corpus_byteparity.py``'s armed control, and
    ``save`` is wrapped by keyword in the timing harness."""
    applied = inspect.signature(deposit_repair.apply).parameters
    assert list(applied) == ["fields", "pml", "sources", "pair", "saved"]
    assert all(parameter.default is inspect.Parameter.empty
               and parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
               for parameter in applied.values())
    saved = inspect.signature(deposit_repair.save).parameters
    assert list(saved) == ["fields", "sources", "pair", "pml", "paths", "prepared"]
    assert [saved[name].kind for name in ("paths", "prepared")] == [
        inspect.Parameter.KEYWORD_ONLY] * 2
    assert saved["paths"].default == SPLIT and saved["prepared"].default is None


def test_the_seam_table_is_not_what_the_restriction_narrows():
    """The restriction selects FROM ``SEAMS``; it never edits it. The table stays the
    mirror of ``stepping``'s constitutive tables that ``test_deposit_repair`` pins."""
    before = {pair: dict(spec) for pair, spec in deposit_repair.SEAMS.items()}
    _grid, fields, pml = _build()
    source = _make_source(fields.grid, "D", "Ez", O, O)
    saved = deposit_repair.save(fields, [source], "D", pml)
    deposit_repair.apply(fields, pml, [source], "D", saved)
    assert deposit_repair.SEAMS == before
    assert deposit_repair.SEAMS["D"]["targets"] == stepping.E_CONSTITUTIVE_TERMS
    assert deposit_repair.SEAMS["B"]["targets"] == stepping.H_CONSTITUTIVE_TERMS


@pytest.mark.parametrize("pair,component,expected", [
    ("D", "Ex", ("Ex", "Dx", "x")), ("D", "Dx", ("Ex", "Dx", "x")),
    ("D", "Ey", ("Ey", "Dy", "y")), ("D", "Ez", ("Ez", "Dz", "z")),
    ("B", "Hx", ("Hx", "Bx", "x")), ("B", "By", ("Hy", "By", "y")),
    ("B", "Hz", ("Hz", "Bz", "z"))])
def test_the_written_target_is_the_one_whose_input_the_injection_writes(pair, component,
                                                                        expected):
    """Read off the INJECTION, not restated: inject one step into zeroed primaries and
    the array that moved is the constitutive input of the target the lookup names."""
    _grid, fields, _pml = _build()
    source = _make_source(fields.grid, pair, component, O, O)
    assert deposit_repair._written_targets(source, pair) == (expected,)
    for name in ("Dx", "Dy", "Dz", "Bx", "By", "Bz"):
        getattr(fields, name)[...] = 0.0
    source.inject(fields, 0.37)
    moved = {name for name in ("Dx", "Dy", "Dz", "Bx", "By", "Bz")
             if numpy.count_nonzero(getattr(fields, name))}
    assert moved == {expected[1]}, moved


@pytest.mark.parametrize("component,layout", [
    ("Ex", lambda n: (n, 1, 1)), ("Ey", lambda n: (1, n, 1)), ("Ez", lambda n: (1, 1, 1))])
def test_each_coefficient_layout_is_the_one_the_restricted_entry_gathers(component,
                                                                         layout):
    _grid, fields, pml = _build()
    axis = component[1]
    kps, kms = stepping._constitutive_coefficients(pml, axis, half_integer=True)
    extent = fields.Ez.shape["xyz".index(axis)]
    assert kps.shape == kms.shape == layout(extent)
    bracket = _PlanBracket(fields, pml, _point("D", component, SHELL)(fields.grid), "D",
                           SPLIT)
    bracket.leading()
    bracket.trailing()
    entry = bracket.plan.prepared.entries[(component, 0)]
    held_kps, held_kms, kp_cells, km_cells = entry.coefficients
    assert held_kps is kps and held_kms is kms
    shape = getattr(fields, component).shape
    assert numpy.array_equal(kp_cells, numpy.broadcast_to(kps, shape)[entry.cells])
    assert numpy.array_equal(km_cells, numpy.broadcast_to(kms, shape)[entry.cells])


def test_a_saved_state_with_no_target_stamp_is_applied_as_the_full_bracket():
    """A dict built by hand -- or by a ``save`` older than the restriction -- names no
    targets, and what it meant was all three. ``apply`` must not narrow what it was
    never told was narrowed."""
    _grid, fields, pml = _build()
    _twin_grid, twin, twin_pml = _build()
    source = _make_source(fields.grid, "D", "Ez", O, O)
    twin_source = _make_source(twin.grid, "D", "Ez", O, O)
    saved = _frozen_save(fields, [source], "D", pml, SPLIT)
    twin_saved = _frozen_save(twin, [twin_source], "D", twin_pml, SPLIT)
    assert deposit_repair._TARGETS_KEY not in saved
    for state in (fields, twin):
        _launch(state, pml if state is fields else twin_pml, "D")
        state.Dz[5, 6, 0] -= numpy.float32(0.5)
    count = deposit_repair.apply(fields, pml, [source], "D", saved)
    assert count == _frozen_apply(twin, twin_pml, [twin_source], "D", twin_saved)
    assert not _compare("an unstamped saved state", _state_words(twin),
                        _state_words(fields), tally=False)
    points = int(numpy.atleast_1d(source._point_ix).size)
    assert count == 3 * points


def test_apply_refuses_a_source_list_save_never_saw():
    _grid, fields, pml = _build()
    first = _make_source(fields.grid, "D", "Ez", O, O)
    second = _make_source(fields.grid, "D", "Ex", O, O)
    saved = deposit_repair.save(fields, [first], "D", pml)
    before = _state_words(fields)
    with pytest.raises(deposit_repair.DepositNotRepairable,
                       match="different source lists"):
        deposit_repair.apply(fields, pml, [first, second], "D", saved)
    assert not _compare("refused apply", before, _state_words(fields), tally=False), (
        "apply wrote before it refused; a partial repair is worse than none")


def test_the_checks_a_restricted_bracket_runs_still_refuse_its_own_component():
    """The per-target absorber clauses are asked of the WRITTEN targets. A layer whose
    coefficient for the written component cannot be recognised is still refused, before
    the launch, by name."""
    _grid, fields, pml = _build()
    source = _make_source(fields.grid, "D", "Ex", O, O)
    broken = numpy.ones((1, fields.Ex.shape[1], 1), dtype=numpy.float32)
    object.__setattr__(pml, "kps_x_h", broken)
    with pytest.raises(deposit_repair.DepositNotRepairable, match="kps_x_h"):
        deposit_repair.save(fields, [source], "D", pml)
    # ...and the predicates, which name no source, still ask about every target.
    ok, reasons = deposit_repair.repairable(fields, "D", pml)
    assert not ok and any("kps_x_h" in reason for reason in reasons), reasons
    assert deposit_repair.seam_source_reasons(
        fields, [source], "D", undeclared="-", refusal=lambda index, source: "-",
        carries_repair=True, pml=pml), "the predicate admitted what it used to refuse"
    # THE OTHER DIRECTION, stated rather than left implicit: a source on Ez neither
    # reads nor writes anything graded on x, so ITS bracket is not refused for it --
    # and what it writes is still the full bracket's answer, because the unrecognised
    # coefficient belongs to a recurrence this repair never evaluates.
    other = _make_source(fields.grid, "D", "Ez", O, O)
    saved = deposit_repair.save(fields, [other], "D", pml)
    assert set(saved[deposit_repair._PREPARED_KEY].entries) == {("Ez", 0)}
    assert deposit_repair.apply(fields, pml, [other], "D", saved) > 0


# ---------------------------------------------------------------------------
# THE HOST CHECKS ARE LIVE EVERY STEP, which is why their verdict is not cached
# ---------------------------------------------------------------------------
#
# ``repairable`` inside ``save`` and ``_absorber_reasons`` inside ``apply`` are the
# restricted bracket's largest remaining host cost, and a per-plan cache of their
# verdict was examined with this change and NOT built. A cache is sound only if every
# attribute the checks read is covered by an event that drops the plan, and that fails
# on named attributes: ``FdtdDriver.fields`` and ``.pml`` are public, their own mutators
# (``Fields.set_nonlinear_volumes``, ``Fields.set_epsilon_volumes``, a rebound ``f_w_*``
# or PML coefficient) never reach ``FdtdDriver.invalidate_fast_path``, and the gates
# that bracket a launch by hand hold no plan to invalidate at all. Each case below
# mutates one such attribute UNDER A LIVE PLAN, through the public surface where one
# exists, and requires the next step to refuse BEFORE the fused launch. They pass
# today because nothing is cached; a verdict cache that did not also key on every one
# of these would turn each refusal into a silent wrong answer and fail here.


def _go_nonlinear(fields, pml, monkeypatch):
    fields.set_nonlinear_volumes({"Ez": 0.0}, {"Ez": 0.1})
    assert fields.has_nonlinearity
    return "chi2/chi3"


def _go_offdiagonal(fields, pml, monkeypatch):
    monkeypatch.setattr(type(fields), "has_offdiagonal_epsilon",
                        property(lambda self: True))
    return "off-diagonal"


def _drop_the_state_array(fields, pml, monkeypatch):
    fields.f_w_Ez = None
    return "f_w_Ez is not allocated"


def _deactivate_the_layer(fields, pml, monkeypatch):
    monkeypatch.setattr(type(pml), "is_active", property(lambda self: False))
    return "is_active False"


def _rebind_a_coefficient(fields, pml, monkeypatch):
    object.__setattr__(pml, "kps_z_h",
                       numpy.ones((1, fields.Ez.shape[1], 1), dtype=numpy.float32))
    return "kps_z_h"


def _fold_onto_a_cylindrical_axis(fields, pml, monkeypatch):
    monkeypatch.setattr(type(fields.grid), "has_symmetry", lambda self: True)
    monkeypatch.setattr(type(fields.grid), "is_axis", lambda self, axis: axis == 0)
    return "cylindrical r = 0 axis"


MUTATIONS = [_go_nonlinear, _go_offdiagonal, _drop_the_state_array,
             _deactivate_the_layer, _rebind_a_coefficient, _fold_onto_a_cylindrical_axis]


@pytest.mark.parametrize("mutate", MUTATIONS, ids=[m.__name__.strip("_") for m in MUTATIONS])
def test_a_check_input_mutated_under_a_live_plan_is_refused_on_the_next_step(
        monkeypatch, mutate):
    _grid, fields, pml = _build()
    source = _make_source(fields.grid, "D", "Ez", O, O)

    class _Inner:
        runs = 0

        def run(self, *_args, **_kwargs):
            type(self).runs += 1

    plan = deposit_repair.LeadingRepairPlan(_Inner(), fields, pml, [source], "D")
    trailing = deposit_repair.TrailingRepairPlan("update_E", plan, fields, pml)
    for _step in range(3):
        plan.run()
        trailing.run()
    assert _Inner.runs == 3
    reason = mutate(fields, pml, monkeypatch)
    with pytest.raises(deposit_repair.DepositNotRepairable, match=reason):
        plan.run()
    assert _Inner.runs == 3, "the fused launch ran on a configuration the checks refuse"


@pytest.mark.parametrize("mutate", [_deactivate_the_layer, _rebind_a_coefficient],
                         ids=["deactivate_the_layer", "rebind_a_coefficient"])
def test_a_layer_mutated_between_the_two_consults_is_refused_before_the_first_write(
        monkeypatch, mutate):
    """``apply``'s own backstop, under the restriction: the layer changes AFTER the
    save admitted it, and the repair refuses without writing a word."""
    _grid, fields, pml = _build()
    source = _make_source(fields.grid, "D", "Ez", O, O)
    saved = deposit_repair.save(fields, [source], "D", pml)
    reason = mutate(fields, pml, monkeypatch)
    before = _state_words(fields)
    with pytest.raises(deposit_repair.DepositNotRepairable, match=reason):
        deposit_repair.apply(fields, pml, [source], "D", saved)
    assert not _compare("refused apply", before, _state_words(fields), tally=False)


# ---------------------------------------------------------------------------
# THE DRIVER LEVEL: complete steps of a real FdtdDriver
# ---------------------------------------------------------------------------

SLOTS = {"B": ("step_B", "update_H"), "D": ("step_D", "update_E")}


class _DriverLaunch:
    """The fused launch on a real driver: the array sub-step, in the driver's order,
    without the injection. ``absorbed_by`` grouping is carried by the repair plans."""

    def __init__(self, built: FdtdDriver, pair: str):
        self._built, self._pair = built, pair

    def run(self, *_args, **_kwargs) -> None:
        _launch(self._built.fields, self._built.pml, self._pair)


class _FrozenLeading:
    """``LeadingRepairPlan``'s shape around the frozen FULL-bracket ``save``."""

    def __init__(self, inner, fields, pml, sources, pair, paths):
        self.inner = self.absorbed_by = inner
        self._args = (fields, pml, tuple(sources), pair, tuple(paths))
        self.saved = None

    def run(self, *args, **kwargs) -> None:
        fields, pml, sources, pair, paths = self._args
        self.saved = _frozen_save(fields, sources, pair, pml, paths)
        self.inner.run(*args, **kwargs)


class _FrozenTrailing:
    def __init__(self, leading):
        self.absorbed_by = leading.inner
        self._leading = leading

    def run(self, *_args, **_kwargs) -> None:
        fields, pml, sources, pair, _paths = self._leading._args
        assert self._leading.saved is not None, "the paired launch did not run"
        _frozen_apply(fields, pml, sources, pair, self._leading.saved)
        self._leading.saved = None


def _bracket_plans(built: FdtdDriver, leg: str) -> Dict[str, Any]:
    plans: Dict[str, Any] = {}
    paths = (deposit_repair.repair_path_for(built.pml),)
    for pair, (curl, update) in SLOTS.items():
        seam = deposit_repair.in_seam_sources(built._sources, pair)
        if not seam or (pair == "B" and paths == PLAIN):
            continue
        inner = _DriverLaunch(built, pair)
        if leg == "frozen_full":
            leading = _FrozenLeading(inner, built.fields, built.pml, seam, pair, paths)
            trailing = _FrozenTrailing(leading)
        else:
            leading = deposit_repair.LeadingRepairPlan(inner, built.fields, built.pml,
                                                       seam, pair, paths)
            trailing = deposit_repair.TrailingRepairPlan(update, leading, built.fields,
                                                         built.pml)
        plans[curl], plans[update] = leading, trailing
    return plans


def _drive(build: Callable[[], FdtdDriver], leg: str, monkeypatch, steps: int,
           flux_at: Optional[int] = None):
    """``steps`` complete driver steps; returns the mid-run and final words, the flux
    taken at ``flux_at`` (through ``synchronize_magnetic_fields``), and the plans."""
    built = build()
    plans: Dict[str, Any] = {}
    if leg == "array":
        monkeypatch.setattr(driver_module, "plan_fast_path", lambda *args: None)
    else:
        plans = _bracket_plans(built, leg)
        assert plans, "no seam of this case carries a deposit; it measures nothing here"
        installed = _install(monkeypatch, built, plans)
    flux = None
    snapshots = []
    try:
        for index in range(steps):
            built.step()
            if index == flux_at:
                centre, extent = built.total_volume()
                flux = built.flux_in_box(0, centre, extent)
            if index in (steps // 2, steps - 1):
                snapshots.append(_state_words(built.fields))
        refusals = {} if leg == "array" else dict(installed.sync_refusals)
    finally:
        built.close()
    return snapshots, flux, plans, refusals


def _synthetic_driver(conductive: bool = False, dispersive: bool = False,
                      complex_fields: bool = False,
                      continuous: bool = False) -> FdtdDriver:
    built = FdtdDriver(cell_size=(1.2, 1.2, 0.0), resolution=20, dimensions=2,
                       boundaries={"x": "metallic", "y": "metallic"},
                       force_complex_fields=complex_fields)
    if conductive:
        volume = numpy.full(built.grid.shape, 0.2, dtype=numpy.float32)
        volume *= numpy.linspace(0.5, 1.5, built.grid.shape[0],
                                 dtype=numpy.float32)[:, None, None]
        built.set_conductivity(volume)
    if dispersive:
        built.add_susceptibility(Susceptibility(1.0, 0.1, "lorentzian"),
                                 {"Ex": 0.3, "Ey": 0.2, "Ez": 0.25})
    built.setup_pml({"x": 3, "y": 3})
    generator = numpy.random.default_rng(20260920)
    dtype = numpy.complex64 if complex_fields else numpy.float32
    for component in ("Dx", "Dy", "Dz", "Bx", "By", "Bz"):
        values = generator.standard_normal(built.grid.shape)
        if complex_fields:
            values = values + 1j * generator.standard_normal(built.grid.shape)
        built.set_field(component, values.astype(dtype))
    for component, center in (("Hz", (0.05, 0.0, 0.0)), ("Hx", (0.0, 0.1, 0.0))):
        built.add_source({"source_type": "gaussian", "component": component,
                          "frequency": 1.0, "fwidth": 0.4, "center": center,
                          "size": (0.0, 0.0, 0.0)})
    if continuous:
        # THE DRIVER'S DEFAULT KIND, through the driver's own constructor and binding
        # (``driver.py:2494-2510``): a point becomes a ``ContinuousSource`` and an extent
        # an ``ExtendedSource``. The line lies under the y layer, so its ``f_u`` mirror
        # is live; the conductive configuration keeps both clear of the layers, as
        # ``_require_source_clear_of_conductive_pml`` demands.
        line_y = 0.05 if conductive else 0.5
        built.add_source({"component": "Ez", "frequency": CW_FREQUENCY,
                          "center": (0.0, 0.0, 0.0), "size": (0.0, 0.0, 0.0)})
        built.add_source({"component": "Ex", "frequency": CW_FREQUENCY,
                          "center": (0.1, line_y, 0.0), "size": (0.3, 0.0, 0.0)})
        kinds = [type(source) for source in built._sources[-2:]]
        assert kinds == [ContinuousSource, ExtendedSource], kinds
        assert conductive or built._sources[-1]._fu_ix is not None, (
            "the line's f_u mirror is not live")
        return built
    for component, center in (("Ez", (0.0, 0.0, 0.0)), ("Ex", (0.1, 0.05, 0.0))):
        built.add_source({"source_type": "gaussian", "component": component,
                          "frequency": 1.0, "fwidth": 0.4, "center": center,
                          "size": (0.0, 0.0, 0.0)})
    return built


SYNTHETIC = {
    "ordinary": dict(),
    "conductive": dict(conductive=True),
    "dispersive_all_components": dict(dispersive=True),
    "complex_dispersive_all_components": dict(dispersive=True, complex_fields=True),
    "continuous": dict(continuous=True),
    "continuous_conductive": dict(continuous=True, conductive=True),
    "continuous_complex_dispersive": dict(continuous=True, dispersive=True,
                                          complex_fields=True),
}


def _live_counters(plans: Mapping[str, Any]) -> Dict[str, Any]:
    return {pair: plans[curl].prepared for pair, (curl, _update) in SLOTS.items()
            if curl in plans}


@pytest.mark.parametrize("name", sorted(SYNTHETIC))
def test_a_real_driver_steps_identically_restricted_through_the_sync_path(
        monkeypatch, name):
    """COMPLETE DRIVER STEPS with a flux call in the middle, so the magnetic bracket
    runs a second way -- ``synchronize_magnetic_fields`` consults ``step_B`` and then
    ``update_H_synchronize`` (``driver.py:4354``, ``:4386``), and ``restore`` then
    writes H back under the plan. Two magnetic and two electric sources, each pair on
    two different components; the conductive configuration routes the electric pair
    through the driver's own replay. The ``continuous`` configurations build the
    electric pair as the driver's DEFAULT source kind does -- a ``ContinuousSource``
    point and an ``ExtendedSource`` line, bound by the driver itself -- which no other
    driver-level case here constructs."""
    steps, flux_at = 8, 3

    def build():
        return _synthetic_driver(**SYNTHETIC[name])

    with monkeypatch.context() as patch:
        array, array_flux, _plans, _refusals = _drive(build, "array", patch, steps,
                                                      flux_at)
    with monkeypatch.context() as patch:
        full, full_flux, _plans, _refusals = _drive(build, "frozen_full", patch, steps,
                                                    flux_at)
    with monkeypatch.context() as patch:
        live, live_flux, plans, refusals = _drive(build, "live", patch, steps, flux_at)
    for index, moment in enumerate(("mid-run", "final")):
        assert not _compare(f"[driver {name}, {moment}] array path vs frozen full bracket",
                            array[index], full[index], "reference")
        assert not _compare(f"[driver {name}, {moment}] frozen full vs restricted plans",
                            full[index], live[index])
    assert array_flux == full_flux == live_flux, (array_flux, full_flux, live_flux)
    assert not refusals, f"the half-step declined the magnetic bracket: {refusals}"
    counters = _live_counters(plans)
    assert set(counters) == {"B", "D"}
    # Two single-component sources per seam; the magnetic bracket ran once more, inside
    # the flux call's half-step.
    assert counters["D"].linear_repairs == 2 * steps, counters["D"].linear_repairs
    assert counters["B"].linear_repairs == 2 * (steps + 1), counters["B"].linear_repairs
    for prepared in counters.values():
        assert prepared.restriction_fallbacks == 0 and prepared.cells_repairs == 0
    assert set(counters["B"].entries) == {("Hz", 0), ("Hx", 1)}
    assert set(counters["D"].entries) == {("Ez", 0), ("Ex", 1)}

    with monkeypatch.context() as patch:
        _aim_at_the_wrong_component(patch)
        wrong, wrong_flux, _plans, _refusals = _drive(build, "live", patch, steps, flux_at)
    assert _compare(f"[driver {name}] NULL CONTROL", full[-1], wrong[-1], tally=False)
    assert wrong_flux != full_flux, (
        "the flux read through the half-step is blind to a magnetic repair aimed at the "
        "wrong component, so the sync-path leg above measures nothing")


# --- lifted corpus cases ------------------------------------------------------


@pytest.fixture(scope="module")
def route():
    directory = str(pathlib.Path(__file__).resolve().parents[1] / "parity" / "meep_gpu")
    if directory not in sys.path:
        sys.path.insert(0, directory)
    import gate_dispatch_fused_route as module  # noqa: PLC0415
    return module


#: ``test_deposit_repair_linear_route.LIFTED`` -- split-field D, the magnetic seam,
#: complex storage, the plain path on a five-pole row -- plus the two 2-D lifts the
#: timing campaigns bracket, the folded seam, a conductive row (the driver's replay),
#: the dispersive rows whose whole-volume work the restriction cuts, and one lift per
#: composition a fused pair carrying the repair is certified on that nothing above
#: reaches: cylindrical real and complex (an Ez and an Er source), a Bloch-periodic
#: row, the two special-kz spellings, a folded complex row and a folded 3-D one.
REALISTIC = ["pml_1d", "magnetic_seam_2d", "complex_1d", "absorber_1d", "pml_2d",
             "folded_2d", "conductive_2d", "dispersive_2d", "folded_dispersive_2d",
             "no_pml_dispersive_2d", "cylindrical_m0", "cylindrical_m1", "bloch_2d",
             "complex_beta_2d", "bfast_1d", "folded_complex_2d", "folded_3d"]
REALISTIC_STEPS = 48


@pytest.mark.parametrize("case", REALISTIC)
def test_a_lifted_corpus_case_steps_identically_restricted(route, monkeypatch, case):
    mp = pytest.importorskip("meep")
    import meep_gpu  # noqa: PLC0415

    mp.verbosity(0)
    started = time.perf_counter()

    def build():
        simulation, _monitors, _units = route.CASES[case](mp)
        return meep_gpu.lift_simulation(simulation, prefer_gpu=False)

    with monkeypatch.context() as patch:
        array, _flux, _plans, _refusals = _drive(build, "array", patch, REALISTIC_STEPS)
    with monkeypatch.context() as patch:
        full, _flux, _plans, _refusals = _drive(build, "frozen_full", patch,
                                                REALISTIC_STEPS)
    with monkeypatch.context() as patch:
        live, _flux, plans, _refusals = _drive(build, "live", patch, REALISTIC_STEPS)
    for index, moment in enumerate(("mid-run", "final")):
        assert not _compare(f"[lift {case}, {moment}] array path vs frozen full bracket",
                            array[index], full[index], "reference")
        assert not _compare(f"[lift {case}, {moment}] frozen full vs restricted plans",
                            full[index], live[index])
    counters = _live_counters(plans)
    assert counters, f"{case} installed no bracket"
    for pair, prepared in counters.items():
        assert len(prepared.entries) == 1, (case, pair, sorted(prepared.entries))
        assert prepared.linear_repairs == REALISTIC_STEPS, (case, pair,
                                                            prepared.linear_repairs)
        assert prepared.restricted_sources == REALISTIC_STEPS
        assert prepared.restriction_fallbacks == 0 and prepared.cells_repairs == 0

    with monkeypatch.context() as patch:
        _aim_at_the_wrong_component(patch)
        wrong, _flux, _plans, _refusals = _drive(build, "live", patch, REALISTIC_STEPS)
    assert _compare(f"[lift {case}] NULL CONTROL", full[-1], wrong[-1], tally=False), (
        f"{case}: a repair aimed at the wrong component matched over "
        f"{REALISTIC_STEPS} steps, so this case cannot see the restriction")
    print(f"lift {case}: {REALISTIC_STEPS} steps x 4 legs "
          f"({time.perf_counter() - started:.1f} s)", flush=True)
