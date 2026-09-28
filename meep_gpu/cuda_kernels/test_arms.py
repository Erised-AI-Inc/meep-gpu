"""The hand-CUDA arm table and the composer that reads it.

WHAT THIS FILE SETTLES, AND WHY IT IS NOT A SMOKE TEST. Until this table existed
nothing selected a CUDA kernel: 22 predicate families admitted 759 of 759 census
slots and every one of them was reachable only from a harness that asked a
hand-picked subset in a hand-written order. So the questions here are not "does it
import" but the four that decide whether a selection can be trusted:

1. **Does the table actually SELECT, and the right arm?** Five real configurations,
   each with a different family as its only admitter, checked by NAME rather than by
   plan class -- several families build the same class, so a check on the class would
   report an ambiguity as a decision.
2. **Does it FAIL CLOSED?** Two admitters leave the slot unselected naming both; a
   raising predicate is a refusal; a raising or refusing builder is a refusal; a
   configuration nothing covers replaces nothing and never raises.
3. **Is a refusal READABLE?** The shared normaliser iterates ``.reasons``, so a bare
   string arrives as one reason PER CHARACTER. That defect is silent -- the verdict
   is still "refused" -- and it is pinned here in both directions.
4. **Is the table COMPLETE before it is read?** Registration is an import side
   effect, and a missing arm does not produce a refusal; it produces a different
   selection with no ambiguity detected.

NOTHING HERE COMPILES OR LAUNCHES ANYTHING, and nothing here needs CuPy: the grid is
real NumPy wearing CuPy's ``__name__``, this directory's standing stand-in, which is
also how the coverage census itself is cut (it lifts with ``prefer_gpu=False``).
"""

from __future__ import annotations

import types

import numpy
import pytest

from ..fields import Fields
from ..grid import Grid
from ..pml import PML
from ..triton_kernels.coverage import Coverage
from . import arms
from . import registry
from .test_single_arm_launches import LAUNCHABLE


class _NumpyWearingCupysName(types.ModuleType):
    """NumPy behind CuPy's ``__name__`` -- this directory's standing stand-in.

    Every CUDA predicate's first question is whether the backend is CuPy at all, and
    that is the one thing about the device library a laptop cannot supply.
    Everything else -- the absorber, the storage width, the fold, the dtypes, the
    off-diagonal rows -- is a real object either way.
    """

    def __init__(self):
        super().__init__("cupy")

    def __getattr__(self, item):
        return getattr(numpy, item)


@pytest.fixture
def xp():
    return _NumpyWearingCupysName()


def build(xp, *, boundaries=("periodic", "periodic", "periodic"), symmetry=(),
          force_complex_fields=False, cell=(8.0, 8.0, 8.0), absorber=True,
          offdiagonal=False, **grid_kwargs):
    """A frozen ``(fields, pml, grid)`` triple, absorber optional.

    ``absorber=False`` builds a real :class:`PML` of zero thickness -- an INERT
    layer, which is what a no-absorber corpus row lifts to -- rather than ``None``,
    so the no-absorber families are asked the question they were written for.
    """
    grid = Grid(resolution=1.0, cell_size=cell, boundaries=boundaries,
                symmetry=symmetry, xp=xp, **grid_kwargs)
    thickness = tuple((0, 0) if grid.shape[axis] < 6
                      else (0, 2) if grid.is_mirrored(axis)
                      else (2, 2)
                      for axis in range(3))
    layer = PML(grid=grid, thickness=thickness if absorber else 0)
    fields = Fields(grid=grid, force_complex_fields=force_complex_fields)
    fields.enable_field_storage()
    if absorber:
        fields.enable_pml_storage()
    if offdiagonal:
        rng = numpy.random.default_rng(5)
        diagonal, inverse = {}, {}
        for component in ("Ex", "Ey", "Ez"):
            values = rng.uniform(0.2, 0.9, size=grid.shape).astype(numpy.float32)
            diagonal[component] = (1.0 / values).astype(numpy.float32)
            inverse[component] = values
        row = {"Ex": {"Ey": rng.uniform(-0.1, 0.1, size=grid.shape)
                      .astype(numpy.float32)}}
        fields.set_epsilon_volumes(diagonal, inverse, chi1inv_offdiagonal=row)
    return fields, layer, grid


@pytest.fixture
def probe_arms():
    """Register arms for one test and take them out again.

    The table is module-global and populated once; a test that left an arm behind
    would change what every LATER test selects, which is the exact failure mode this
    file exists to catch.
    """
    added = []

    def add(**kwargs):
        spec = arms.register(**kwargs)
        added.append(spec)
        return spec

    yield add
    for spec in added:
        arms._REGISTRY[spec.slot].remove(spec)


def covered(*_args, **_kwargs) -> Coverage:
    return Coverage(True, ("covered",))


# --------------------------------------------------------------------------
# 1. THE TABLE
# --------------------------------------------------------------------------

def test_the_table_is_complete_before_it_is_read():
    """Reading the table populates it, and says so."""
    specs = arms.registered()
    assert arms.registration_state() == "done"
    # FLOORS, not budgets: the measured table on 2026-08-28 is 49 arms over 22
    # families, and a family landing later must not have to edit this number. What
    # must never happen is the table SHRINKING, because an arm missing from it does
    # not produce a refusal -- it produces a different selection with no ambiguity
    # detected.
    assert len(specs) >= 49, f"the table holds {len(specs)} arms; 49 were measured"
    assert len(registry.FAMILIES) >= 22
    assert all(spec.wired for spec in specs), (
        "every CUDA family is wired; an unwired one is skipped by arms_for and must "
        "say in registry._TABLE why its composition is deferred")


def test_every_registered_arm_binds_a_real_shipped_predicate():
    """No row of the table names a predicate that is not there to be asked."""
    from importlib import import_module

    for spec in arms.registered():
        assert spec.shape in arms.SHAPES, f"{spec} carries shape {spec.shape!r}"
        package, module, name = spec.predicate_name.split(".")
        assert package == "cuda_kernels"
        assert callable(getattr(import_module(f".{module}", __package__), name)), (
            f"{spec.family} binds {spec.predicate_name}, which is not callable")


def test_all_six_positional_shapes_are_in_use():
    """Six shapes were measured off the shipped signatures; all six carry arms.

    A shape with no arm is a mapping nothing exercises, which is where a wrong
    argument order survives review.
    """
    in_use = {spec.shape for spec in arms.registered()}
    assert in_use == set(arms.SHAPES), (
        f"shapes with no arm: {sorted(set(arms.SHAPES) - in_use)}; "
        f"arms with no shape entry: {sorted(in_use - set(arms.SHAPES))}")


def test_residency_is_not_carried_across_from_the_sibling_track():
    """CUDA has no residency concept and the context must not invent one."""
    context = arms.StepContext(fields=None, pml=None, grid=None)
    assert "residency" not in arms.StepContext.__slots__
    with pytest.raises(AttributeError):
        context.residency  # noqa: B018 - the read IS the assertion


def test_an_arm_is_launchable_exactly_where_its_row_names_a_resolver_and_a_launcher():
    """A resolver is a MEASURED argument binding, not a guess.

    An arm launches only where its registry row names a ``resolve`` token that BOTH
    ``registry._RESOLVERS`` and ``registry._LAUNCHERS`` carry -- the two ends of one
    signature. Every other family ships a certified kernel whose argument resolution
    lives in its own gate; a plan that claimed to know it would be a half-cell error
    away from converged, smooth and wrong. The set that results is the nine arms
    ``test_single_arm_launches.LAUNCHABLE`` names, and this test reads it from there
    rather than keeping a second copy.
    """
    context = arms.StepContext(fields=None, pml=None, grid=None)
    established = {row["family"] for row in registry._TABLE  # noqa: SLF001
                   if row.get("resolve") in registry._RESOLVERS  # noqa: SLF001
                   and row.get("resolve") in registry._LAUNCHERS}  # noqa: SLF001
    launchable = {(spec.family, spec.slot) for spec in arms.registered()
                  if spec.plan(context, spec.slot).launchable}
    assert launchable == {(spec.family, spec.slot) for spec in arms.registered()
                          if spec.family in established}
    assert launchable == LAUNCHABLE


def test_the_nonlinear_family_names_a_different_kernel_on_each_side():
    """The one row whose label is not a format string, and why it may not be.

    The engine reads chi2/chi3 in ``update_E`` only: that side has a dedicated Pade
    body while ``update_H`` reuses the ordinary PML constitutive under a
    nonlinear-only spine arm. One format string over ``{slot}`` produced
    ``fused_update_H_pml_real_nonlinear``, naming a body that side does not run —
    a wrong answer that reads as a right one because the shape of the string is
    plausible.
    """
    context = arms.StepContext(fields=None, pml=None, grid=None)
    labels = {spec.slot: spec.plan(context, spec.slot).kernel_label
              for spec in arms.registered() if spec.family == "cuda_nonlinear"}
    assert labels == {"update_H": "fused_update_H_pml_real",
                      "update_E": "fused_update_E_pml_real_nonlinear"}


def test_the_deliberately_excluded_predicates_have_no_row():
    """Three exclusions, each a decision the registry docstring spells out.

    The per-component ADE helpers would bid twice for a slot their whole-sub-step
    sibling already serves, and the STANDALONE folded off-diagonal verdict admits an
    unfolded grid the certified off-diagonal family claims -- registering it would
    make ``update_E`` permanently ambiguous on every unfolded off-diagonal row.
    """
    bound = {spec.predicate_name for spec in arms.registered()}
    for excluded in ("cuda_kernels.coverage.covers_real_pml_ade_component",
                     "cuda_kernels.dispersive_kernels.covers_no_pml_ade_component",
                     "cuda_kernels.folded_offdiag_kernels."
                     "covers_folded_offdiag_constitutive"):
        assert excluded not in bound, (
            f"{excluded} has a row; registry.py's docstring says why it must not")


def test_every_shipped_predicate_is_registered_or_documented():
    """The converse of the deny-list, and the one that catches a NEW family.

    The deny-list above pins three names that must stay out. Nothing in it notices a
    predicate that lands in the package and gets no row -- that family would simply
    never be selected, no slot would go ambiguous, and no test would go red. That is
    the silent-divergence class this whole track exists to make impossible, so the
    demand here is total: every ``covers_*`` in a shipped module is either bound to
    an arm or named in :data:`registry.NOT_REGISTERED` with its reason.

    Read from the SOURCE with ``ast`` rather than by importing: most family modules
    import CuPy at module scope, so an import-based walk would be unrunnable on the
    backend-free host this table is checked on.
    """
    import ast  # noqa: PLC0415 - a source walk is the point, not the package API
    import pathlib  # noqa: PLC0415

    package = pathlib.Path(registry.__file__).parent
    shipped = set()
    for path in sorted(package.glob("*.py")):
        if path.name.startswith("test_"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if (isinstance(node, ast.FunctionDef)
                    and node.name.startswith("covers_")):
                shipped.add(f"{path.stem}.{node.name}")

    # A floor, so a walk that silently stopped finding anything cannot pass. 38 on
    # 2026-08-28; it may only grow, and growing is what this test is for.
    assert len(shipped) >= 38, f"the source walk found only {len(shipped)}"

    bound = {name.split("cuda_kernels.", 1)[-1]
             for name in (spec.predicate_name for spec in arms.registered())}
    undecided = shipped - bound - set(registry.NOT_REGISTERED)
    assert not undecided, (
        f"{sorted(undecided)} ship a coverage predicate with no arm and no entry in "
        "registry.NOT_REGISTERED: give it a row, or record why it must not have one")

    # And the map may not name a predicate that does not exist, or one that DOES have
    # a row -- either way the reason recorded there would be describing nothing.
    stale = set(registry.NOT_REGISTERED) - shipped
    assert not stale, f"registry.NOT_REGISTERED names {sorted(stale)}, not in the tree"
    contradicted = set(registry.NOT_REGISTERED) & bound
    assert not contradicted, (
        f"{sorted(contradicted)} are both registered and declared unregistered")


# --------------------------------------------------------------------------
# 2. SELECTION
# --------------------------------------------------------------------------

def test_real_pml_run_selects_the_certified_curl_and_constitutive_pair(xp):
    fields, layer, grid = build(xp)
    plan = arms.plan_step(fields, layer, grid)

    assert plan.selected["step_B"] == "PML"
    assert plan.selected["step_D"] == "PML"
    assert plan.selected["update_H"] == "ordinary"
    assert plan.selected["update_E"] == "ordinary"
    assert plan.replaces == ("step_B", "update_H", "step_D", "update_E")
    # These four are the two families with a resolver, so this run is also the only
    # one in this file whose whole covered subset is launchable.
    assert plan.launchable == plan.replaces
    assert plan.plans["step_B"].kernel_label == "fused_step_B_pml_real"


def test_an_inert_layer_selects_the_no_absorber_families(xp):
    """The absorber is the whole inversion: the certified pair refuses BY NAME."""
    fields, layer, grid = build(xp, absorber=False)
    plan = arms.plan_step(fields, layer, grid)

    assert plan.selected["step_B"] == "no-PML curl"
    assert plan.selected["step_D"] == "no-PML curl"
    assert plan.selected["update_H"] == "no-PML null"
    assert any("no active PML layer" in reason
               for reason in plan.reasons["update_E"]), plan.reasons["update_E"]
    # Nothing here is launchable: neither no-absorber family has an established
    # argument resolution on this track.
    assert plan.launchable == ()


def test_a_live_off_diagonal_row_moves_update_E_to_its_own_family(xp):
    """One slot, a different winner, and the curl pair unchanged.

    This is the case that shows the table selects PER SLOT rather than per run: the
    off-diagonal row changes ``update_E`` only, and the ordinary constitutive refuses
    it by name rather than being outranked.
    """
    fields, layer, grid = build(xp, offdiagonal=True)
    plan = arms.plan_step(fields, layer, grid)

    assert plan.selected["step_B"] == "PML"
    assert plan.selected["update_H"] == "ordinary"
    assert plan.selected["update_E"] == "off-diagonal"
    assert plan.plans["update_E"].kernel_label == "fused_update_E_pml_real_offdiag"
    # `update_E` is covered AND launchable: since 2026-09-27 the off-diagonal family
    # carries a launch-argument resolution and a launcher
    # (`arms.resolve_offdiag_launch_args` / `arms.launch_offdiag`), the derivations
    # its own wrapper performs when handed the layer.
    assert "update_E" in plan.replaces
    assert "update_E" in plan.launchable


def test_complex_storage_refuses_by_name_when_no_expansion_licence_is_bound(xp):
    """``None`` IS NOT A DEFAULT ARM.

    Which arm this host's reference complex multiply takes is a measured platform
    fact. A composer that guessed it would licence a kernel performing an expansion
    nothing classified, so an absent licence is a NAMED refusal and the slot falls to
    the array path.
    """
    fields, layer, grid = build(xp, force_complex_fields=True)
    plan = arms.plan_step(fields, layer, grid)

    for slot in ("step_B", "step_D", "update_H", "update_E"):
        assert slot not in plan.selected, f"{slot} selected {plan.selected.get(slot)}"
        assert any("licence" in reason or "license" in reason
                   for reason in plan.reasons[slot]), plan.reasons[slot]
    # And the certified real-field arms refused for the storage width, which is the
    # clause that makes the two families disjoint rather than ordered.
    assert any("complex64 storage" in reason for reason in plan.reasons["step_B"])


def test_the_fill_slots_refuse_an_unfolded_grid_by_name_rather_than_in_silence(xp):
    """A fill slot on an unfolded grid is a REFUSAL with a sentence, not a silence.

    Each fill slot carries exactly one arm since 2026-09-27, the mirror fill over the
    certified in-seam kernels. Both of its pass predicates admit an unfolded grid
    vacuously, so the arm refuses one BY NAME after them; a caller testing
    ``if plan.reasons.get(slot)`` must still read why the slot stayed empty.
    """
    fields, layer, grid = build(xp)
    plan = arms.plan_step(fields, layer, grid)

    for slot in ("fill_B", "fill_D"):
        spec, = arms.registered(slot)
        assert (spec.family, spec.label, spec.shape) == (
            "cuda_mirror_fill", "mirror fill", "fill_by_family")
        assert slot not in plan.selected
        assert plan.reasons[slot], f"{slot} was left with a falsy reason tuple"
        assert "the grid is not mirror-folded" in plan.reasons[slot][0], (
            plan.reasons[slot])


# --------------------------------------------------------------------------
# 3. FAIL-CLOSED
# --------------------------------------------------------------------------

def test_two_admitters_leave_the_slot_unselected_naming_both(xp, probe_arms):
    """An overlap is a predicate defect and must never become a table-order choice."""
    for label in ("deliberate twin A", "deliberate twin B"):
        probe_arms(family="probe_only", slot="update_P", label=label,
                   coverage=covered,
                   plan=lambda context, slot: arms.CudaSlotPlan(
                       "probe_only", slot, "twin", "none"),
                   noun=label, shape="whole_slot")

    fields, layer, grid = build(xp)
    plan = arms.plan_step(fields, layer, grid)

    assert "update_P" not in plan.selected
    assert "update_P" not in plan.plans
    message, = plan.reasons["update_P"]
    assert "admit this configuration" in message
    assert "deliberate twin A" in message and "deliberate twin B" in message


def test_a_predicate_that_raises_is_a_refusal_and_not_an_exception(xp, probe_arms):
    """``plan_step`` never raises into a caller that would have stepped correctly."""
    def explode(context, slot):
        raise RuntimeError("the predicate could not read the grid")

    probe_arms(family="probe_only", slot="update_P", label="raiser",
               coverage=explode,
               plan=lambda context, slot: None, noun="raiser",
               shape="whole_slot")

    fields, layer, grid = build(xp)
    plan = arms.plan_step(fields, layer, grid)

    assert "update_P" not in plan.selected
    assert any("raiser predicate raised" in reason
               for reason in plan.reasons["update_P"]), plan.reasons["update_P"]


def test_a_builder_that_raises_refuses_its_slot(xp, probe_arms):
    """Coverage admitting is not a licence to launch a plan that could not be built."""
    def explode(context, slot):
        raise RuntimeError("no CuPy on this host")

    probe_arms(family="probe_only", slot="update_P", label="bad builder",
               coverage=covered, plan=explode, noun="bad builder",
               shape="whole_slot")

    fields, layer, grid = build(xp)
    plan = arms.plan_step(fields, layer, grid)

    assert "update_P" not in plan.plans
    assert any("builder raised" in reason
               for reason in plan.reasons["update_P"]), plan.reasons["update_P"]


def test_a_configuration_nothing_covers_replaces_nothing(xp):
    """The array path is always correct, and reaching it is not an error.

    A REAL NumPy grid -- not the stand-in -- is refused by every arm on the backend
    clause, which is the one refusal every family shares.
    """
    grid = Grid(resolution=1.0, cell_size=(8.0, 8.0, 8.0),
                boundaries=("periodic",) * 3, xp=numpy)
    layer = PML(grid=grid, thickness=(2, 2, 2))
    fields = Fields(grid=grid)
    fields.enable_field_storage()
    fields.enable_pml_storage()

    plan = arms.plan_step(fields, layer, grid)

    assert plan.replaces == ()
    assert plan.selected == {}
    assert any("backend is not CuPy" in reason
               for reason in plan.reasons["step_B"]), plan.reasons["step_B"]


def test_a_duplicate_registration_is_refused_rather_than_replaced(probe_arms):
    """Keeping one of two would make WHICH KERNEL RUNS depend on import order."""
    probe_arms(family="probe_only", slot="update_P", label="twice",
               coverage=covered,
               plan=lambda context, slot: None, noun="twice", shape="whole_slot")
    with pytest.raises(ValueError, match="already registered"):
        arms.register(family="probe_only", slot="update_P", label="twice",
                      coverage=covered, plan=lambda context, slot: None,
                      noun="twice", shape="whole_slot")


# --------------------------------------------------------------------------
# 4. THE READABILITY OF A REFUSAL
# --------------------------------------------------------------------------

def test_a_refusal_is_one_reason_per_arm_and_never_one_per_character(xp):
    """THE TUPLE TRAP, pinned on the shipped table.

    ``triton_kernels/launch.py`` normalises a verdict with
    ``tuple(str(reason) for reason in reasons)``. Every shipped CUDA predicate
    returns ``(bool, str)``, and a bare string is iterable -- so handing one through
    produces one prefixed "reason" PER CHARACTER while the verdict stays a perfectly
    ordinary refusal. Nothing else in the pipeline notices.
    """
    grid = Grid(resolution=1.0, cell_size=(8.0, 8.0, 8.0),
                boundaries=("periodic",) * 3, xp=numpy)
    layer = PML(grid=grid, thickness=(2, 2, 2))
    fields = Fields(grid=grid)
    fields.enable_field_storage()
    fields.enable_pml_storage()

    plan = arms.plan_step(fields, layer, grid)

    for slot in ("step_B", "step_D", "update_H", "update_E", "update_P"):
        reasons = plan.reasons[slot]
        assert len(reasons) == len(arms.registered(slot)), (
            f"{slot} reported {len(reasons)} reasons for "
            f"{len(arms.registered(slot))} consulted arms; a count that is a "
            f"multiple of the arm count is the exploded-string defect")
        for reason in reasons:
            # Each reason is "<prefix>: <sentence>". The exploded form is the
            # prefix plus exactly one character.
            assert len(reason.split(": ", 1)[-1]) > 1, f"{slot}: {reason!r}"


def test_the_adapter_wraps_a_bare_string_and_the_unwrapped_form_would_explode():
    """Both directions, because only the pair shows the wrap is load-bearing."""
    def predicate(fields, pml, grid):
        return False, "no PML"

    context = arms.StepContext(fields=None, pml=None, grid=None)
    verdict = arms.coverage_adapter(predicate, "whole_slot")(context, "update_E")
    assert verdict.covered is False
    assert verdict.reasons == ("no PML",)

    # The same verdict built WITHOUT the wrap, through the same normaliser the
    # composer uses. Six one-character reasons is the defect this adapter prevents.
    plans, reasons, selected = {}, {}, {}
    arms._select_slot(
        "update_E",
        (arms._Arm("bare", True, lambda: Coverage(False, "no PML"),
                   lambda: None, "bare: ", "bare"),),
        arms.ambiguity("update_E"), plans, reasons, selected)
    assert reasons["update_E"] == ("bare: n", "bare: o", "bare:  ",
                                   "bare: P", "bare: M", "bare: L")


def test_an_unselected_slot_never_carries_a_falsy_reason_tuple(xp):
    """A caller testing ``if plan.reasons.get(slot)`` must not read a refusal as
    "nothing to report"."""
    fields, layer, grid = build(xp, absorber=False)
    plan = arms.plan_step(fields, layer, grid)

    for slot in arms.STEP_ORDER:
        if slot in plan.selected:
            continue
        assert plan.reasons.get(slot), f"{slot} is unselected with nothing to say"


# --------------------------------------------------------------------------
# The single arm's launch: resolve for THIS slot, then the family launcher
# --------------------------------------------------------------------------

def test_a_launchable_plan_runs_its_family_launcher_with_the_resolved_arguments():
    """``run()`` calls the resolver as ``(context, slot)`` -- both resolvers here take
    the slot, unlike a product's -- hands the result to the launcher with the
    context's fields, and reads the block count back for ``launch_grid``."""
    calls = []

    def resolve(context, slot):
        calls.append(("resolve", context.fields, slot))
        return {"tables": "T", "boundary_codes": (0, 0, 0), "dtdx": 0.5}

    class Launcher:
        def __call__(self, fields, slot, arguments):
            calls.append(("launch", fields, slot, arguments))
            return {"launched": True, "blocks": 7}

        def warm(self, slot):
            calls.append(("warm", slot))
            return None

    context = arms.StepContext(fields="F", pml=None, grid=None)
    plan = arms.plan_factory("probe", "PML", "fused_step_B_pml_real", resolve,
                             Launcher())(context, "step_B")
    assert plan.launchable
    assert plan.launch_grid is None, "UNKNOWN before the first launch"
    assert plan.run() == {"launched": True, "blocks": 7}
    assert calls == [("resolve", "F", "step_B"),
                     ("launch", "F", "step_B",
                      {"tables": "T", "boundary_codes": (0, 0, 0), "dtdx": 0.5})]
    assert plan.launches == 1
    assert plan.launch_grid == (7,)
    # RESOLVED ONCE PER PLAN: the second consult launches again and resolves nothing.
    assert plan.run() == {"launched": True, "blocks": 7}
    assert plan.launches == 2
    assert [c[0] for c in calls] == ["resolve", "launch", "launch"]
    assert plan.warm() is None
    assert calls[-1] == ("warm", "step_B")


def test_a_plan_without_an_established_launch_raises_by_name_and_is_not_launchable():
    """A resolver alone is half a launch: not launchable, ``run()`` refuses by name
    rather than guessing a launcher, and ``warm()`` answers with a reason."""
    context = arms.StepContext(fields=None, pml=None, grid=None)
    plan = arms.CudaSlotPlan("probe", "step_D", "PML", "x", lambda _c, _s: {},
                             context=context)
    assert not plan.launchable
    with pytest.raises(RuntimeError, match="probe/step_D has no established launch"):
        plan.run()
    assert "declares no launcher warm" in plan.warm()
    bare = arms.CudaSlotPlan("probe", "step_D", "PML", "x")
    assert not bare.launchable and bare.launch_grid is None


def test_every_launchable_family_carries_a_launcher_that_can_warm_by_entry_point():
    """Every established family carries a launcher beside its resolver, and the
    launcher compiles by ENTRY POINT (never by ``kernel_label``, which is a label).

    The mirror fill names a PAIR of entry points per slot -- its slot is two driver
    passes -- so the entry is normalised to a tuple and every name is checked.
    """
    context = arms.StepContext(fields=None, pml=None, grid=None)
    seen = set()
    for spec in arms.registered():
        plan = spec.plan(context, spec.slot)
        if not plan.launchable:
            continue
        seen.add(spec.family)
        assert plan.launch_kernel is not None
        assert callable(getattr(plan.launch_kernel, "warm", None))
        entry = plan.launch_kernel._ENTRY_POINTS[spec.slot]  # noqa: SLF001
        names = entry if isinstance(entry, tuple) else (entry,)
        assert names, f"{spec.family}/{spec.slot} names no entry point"
        for name in names:
            assert isinstance(name, str) and name
            assert name != plan.kernel_label
            assert not name.startswith("fused_")
    assert seen == {"cuda_curl", "cuda_constitutive", "cuda_offdiag",
                    "cuda_folded_offdiag", "cuda_dispersive_offdiag",
                    "cuda_mirror_fill"}
