"""Laptop contracts for the SCRATCH-OUTPUT off-diagonal D/E weld (unfolded).

WHAT THIS SUITE IS FOR. The device gate
(``parity/meep_gpu/gate_triton_offdiag_stencil_welds.py``) is what certifies the
bytes; this suite is the merge bar, and it holds the things that would otherwise
only be true on a machine nobody runs on the way in: the predicate's refusals BY
NAME, the declarations that describe the wiring rather than request it, the
transcription of the certified bodies, the rotation's own contract, and the two
facts about this family that a later edit could quietly invert — that the deposit
repair is refused for an ARITHMETIC reason and not an unwired one, and that the
weld's in-seam pass is the one the driver actually runs in that slot.

It also runs the host probe's whole verdict, so a transcription drift fails here
rather than at the far end of a queued device run.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pathlib
import sys

import numpy
import pytest

from .fields import IYEE_SHIFTS, Fields
from .grid import Grid
from .pml import PML

HERE = pathlib.Path(__file__).parent
PACKAGE_DIR = HERE / "triton_kernels"
PARITY = HERE.parent / "parity" / "meep_gpu"
MODULE = "meep_gpu.triton_kernels.offdiag_fused_electric_pair"
FAMILY = "offdiag_fused_electric_pair"


@pytest.fixture(scope="module")
def product():
    return importlib.import_module(MODULE)


@pytest.fixture(scope="module")
def probe():
    """The host probe's module, imported for its legs and its tables."""
    for path in (str(PARITY), str(HERE.parent)):
        if path not in sys.path:
            sys.path.insert(0, path)
    return importlib.import_module("probe_triton_offdiag_scratch_weld")


@pytest.fixture(scope="module")
def gate():
    for path in (str(PARITY), str(HERE.parent)):
        if path not in sys.path:
            sys.path.insert(0, path)
    return importlib.import_module("gate_triton_offdiag_stencil_welds")


# ---------------------------------------------------------------------------
# Fixtures — real Grid/Fields/PML on NumPy
# ---------------------------------------------------------------------------

def build(offdiag=True, poles=0, thickness=0.2, boundaries=None, symmetry=(),
          complex_storage=False, sigma=None, rows=None):
    grid = Grid(resolution=10.0, cell_size=(1.2, 1.2, 0.0), dimensions=2,
                boundaries=boundaries or {"x": "metallic", "y": "metallic",
                                          "z": "periodic"},
                symmetry=tuple(symmetry))
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    shape = grid.shape
    if offdiag:
        diagonal = {name: numpy.full(shape, 2.25, dtype=numpy.float32)
                    for name in ("Ex", "Ey", "Ez")}
        inverse = {name: numpy.full(shape, 1.0 / 2.25, dtype=numpy.float32)
                   for name in ("Ex", "Ey", "Ez")}
        if rows is None:
            rows = {"Ex": {"Ey": 0.05, "Ez": 0.03},
                    "Ey": {"Ez": 0.02, "Ex": 0.04},
                    "Ez": {"Ex": 0.01, "Ey": 0.06}}
        volumes = {row: {partner: numpy.full(shape, value, dtype=numpy.float32)
                         for partner, value in entries.items()}
                   for row, entries in rows.items()}
        fields.set_epsilon_volumes(diagonal, inverse, volumes)
    else:
        fields.set_background_eps(2.25)
    if poles:
        from .dispersion import PolarizationState, Susceptibility

        for index in range(poles):
            fields.polarizations.append(PolarizationState(
                Susceptibility(1.0 + 0.3 * index, 0.1, "lorentzian"),
                {"Ex": 0.3, "Ey": 0.2, "Ez": 0.25}, grid,
                numpy.complex64 if complex_storage else numpy.float32))
    if sigma is not None:
        fields.set_d_conductivity(
            numpy.full(shape, float(sigma), dtype=numpy.float32))
    if thickness:
        fields.enable_pml_storage()
    else:
        fields.enable_field_storage()
    # A MIRRORED axis takes the HIGH face only: its cell 0 lies ON the mirror
    # plane, which is a boundary condition and not an absorber, and `PML` refuses
    # the low face there BY NAME (pml.py:408).
    faces = {}
    if thickness:
        for axis, letter in enumerate("xyz"[:2]):
            faces[letter] = ({"high": thickness} if grid.is_mirrored(axis)
                             else thickness)
    pml = PML(grid=grid, thickness=faces or 0)
    return fields, pml


class _Electric:
    field_type = "D"
    component = "Ez"
    is_integrated = True
    _point_ix = numpy.array([1])
    _point_iy = numpy.array([1])
    _point_iz = numpy.array([0])


class _Magnetic:
    field_type = "B"
    component = "Hz"
    is_integrated = True
    _point_ix = numpy.array([1])
    _point_iy = numpy.array([1])
    _point_iz = numpy.array([0])


def _reasons(verdict):
    """The refusal reasons MINUS the array-module clause.

    Every clause here is asked on a NumPy host, where the backend clause always
    fires; subtracting it is what lets the other clauses be asserted at all.
    """
    return [reason for reason in verdict.reasons if "not cupy" not in reason]


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------

def test_the_product_declares_the_seam_it_replaces(product):
    """``REPLACES`` names the driver call sites ONE launch performs, in order.

    ``step_D`` and ``update_E`` are the two consults; ``zero_metal_D`` is the
    pass the closed form carries INLINE. The two symmetry fills are NOT here —
    they are refused rather than carried, and a launch may only claim to replace
    what it actually performs.
    """
    assert product.REPLACES == ("step_D", "zero_metal_D", "update_E")
    assert product.CURL_SUB_STEP == "step_D"
    assert product.CONSTITUTIVE_SIDE == "E"
    assert product.PAIR == "D"
    assert product.BACKWARD == 1
    assert set(product.REFUSED_IN_SEAM_PASSES) == {
        "fill_symmetry_bc_D", "fill_folded_far_ghosts_D"}


def test_replaces_is_the_drivers_own_call_order(product):
    """Each name in ``REPLACES`` is a call the driver makes in that slot, in that
    order. Read out of ``driver.py``'s own source, so a driver that reordered its
    seam fails here rather than silently making the claim false."""
    from . import driver as driver_module

    source = inspect.getsource(driver_module.FdtdDriver.step)
    seam = source.split("fast.dispatch(\"step_D\"")[1].split(
        "fast.dispatch(\"update_P\"")[0]
    positions = []
    for name in product.REPLACES:
        assert name in seam or name == "step_D", name
        positions.append(0 if name == "step_D" else seam.index(name))
    assert positions == sorted(positions), (product.REPLACES, positions)


def test_the_curl_table_is_the_absorber_paths_and_not_the_no_pml_one(product):
    """``step_D`` differences H under an absorber and B without one.

    The two ``SUB_STEPS`` tables in this package disagree on purpose
    (``no_pml.py:178-181``: with no absorber ``Fields`` never allocates H and
    ``get_H`` returns B). This family is the ABSORBER path, so ``launch``'s table
    is the right one — a product that followed the other would bind three
    pointers to the wrong volumes and step a plausible wrong field.
    """
    from .triton_kernels import launch as launch_module
    from .triton_kernels import no_pml

    assert product.CURL_SOURCES == tuple(
        launch_module.SUB_STEPS["step_D"]["sources"]) == ("Hx", "Hy", "Hz")
    assert product.CURL_TARGETS == tuple(
        launch_module.SUB_STEPS["step_D"]["targets"])
    assert product.BACKWARD == int(launch_module.SUB_STEPS["step_D"]["backward"])
    assert product.CURL_SOURCES != tuple(no_pml.SUB_STEPS["step_D"]["sources"])
    fields, _pml = build()
    assert fields.Hx is not None, "an active absorber stores H"


def test_the_rotating_set_is_exactly_what_the_curl_half_writes(product):
    """Six volumes rotate: the three displacements and their three auxiliaries.

    Not five and not seven. ``pml_curl_step`` stores exactly ``f0..f2`` and
    ``u0..u2``; a rotation set that missed one would leave that volume holding
    pre-launch state under the engine's own name, which is invisible on step 1.
    """
    assert product.ROTATED == ("Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz")
    assert set(product.ROTATED) == (
        set(product.CURL_TARGETS)
        | {"fu_" + name for name in product.CURL_TARGETS})


def test_the_deposit_repair_is_refused_for_an_ARITHMETIC_reason(product):
    """``CARRIES_DEPOSIT_REPAIR`` is False, and the reason is the stencil.

    This is the clause a later edit is most likely to invert by accident, because
    every sibling electric pair in this package declares True. Here the refusal
    is not about wiring: ``deposit_repair`` inverts the constitutive recurrence
    AT the deposit cell, and this constitutive half is a STENCIL, so a deposit at
    one cell moves the result of up to four. The predicate's own refusal string
    has to say so, because that string is what a reader of a refused run sees.
    """
    assert product.CARRIES_DEPOSIT_REPAIR is False
    assert product.REPAIR_PATHS == ()
    fields, pml = build()
    verdict = product.offdiag_fused_electric_pair_coverage(
        fields, pml, (_Electric(),))
    reasons = _reasons(verdict)
    assert any("STENCIL" in reason for reason in reasons), reasons
    assert any("up to four" in reason for reason in reasons), reasons


def test_the_stencil_really_does_reach_four_cells(product):
    """The refusal's arithmetic claim, checked against the certified body.

    ``_offdiag_term`` loads the partner displacement at FOUR indices — its own,
    one down the partner's axis, one up the component's own axis, and the corner
    — so a deposit at any one of them moves this cell's coupling. Counted off the
    certified source rather than asserted, so a body that grew a fifth tap makes
    the refusal string wrong and this test red.
    """
    source = _function_source(
        PACKAGE_DIR / "offdiag_update_e.py", "_offdiag_term")
    loads = [line for line in source.splitlines() if "tl.load(g +" in line]
    assert len(loads) == 4, loads


def _function_source(path, name):
    text = pathlib.Path(path).read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(text, node)
    raise AssertionError(f"{path} declares no function {name}")


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------

def test_the_predicate_admits_the_configuration_this_cell_is(product):
    fields, pml = build()
    assert _reasons(product.offdiag_fused_electric_pair_coverage(
        fields, pml, ())) == []


@pytest.mark.parametrize("label,needle,kwargs", [
    ("no absorber", "no active PML layer", {"thickness": 0}),
    ("a fold", "mirror plane", {"symmetry": ("y",)}),
    ("complex storage", "force_complex_fields", {"complex_storage": True}),
    ("a conductivity", "conductivity is installed", {"sigma": 0.4}),
    ("a pole", "susceptibility is registered", {"poles": 1}),
    ("no off-diagonal row", "off-diagonal chi1inv row", {"offdiag": False}),
])
def test_the_predicate_refuses_by_name(product, label, needle, kwargs):
    """Every refusal names its own condition. A predicate that refused without
    saying why sends a reader to look for a kernel gap that is not there."""
    fields, pml = build(**kwargs)
    reasons = _reasons(product.offdiag_fused_electric_pair_coverage(
        fields, pml, ()))
    assert any(needle in reason for reason in reasons), (label, reasons)


def test_an_undeclared_source_list_is_refused(product):
    """``None`` is a REFUSAL, not an empty set. ``Fields`` does not hold the
    source list — the driver does — so inferring "no electric source" from not
    knowing is how a predicate over-covers."""
    fields, pml = build()
    reasons = _reasons(product.offdiag_fused_electric_pair_coverage(
        fields, pml, None))
    assert any("not declared" in reason for reason in reasons), reasons


def test_a_magnetic_source_never_reaches_this_seam(product):
    """The driver injects B-family sources between ``step_B`` and ``update_H``,
    which is the other half of the step. Admitting one here is correct and the
    test is what stops a later widening from being read as a bug."""
    fields, pml = build()
    assert _reasons(product.offdiag_fused_electric_pair_coverage(
        fields, pml, (_Magnetic(),))) == []


def test_the_predicate_and_the_two_certified_halves_agree(product):
    """The product may not admit what either certified half refuses.

    Checked as a conjunction over a small configuration sweep rather than
    asserted once: a predicate that dropped one half's clauses would still pass a
    single positive case.
    """
    from .triton_kernels import coverage as coverage_module
    from .triton_kernels.offdiag_update_e import offdiag_constitutive_coverage

    for kwargs in ({}, {"thickness": 0}, {"sigma": 0.4}, {"poles": 1},
                   {"offdiag": False}, {"symmetry": ("y",)}):
        fields, pml = build(**kwargs)
        whole = not _reasons(product.offdiag_fused_electric_pair_coverage(
            fields, pml, ()))
        halves = (not _reasons(coverage_module.pml_curl_coverage(
                      fields, pml, "step_D"))
                  and not _reasons(offdiag_constitutive_coverage(fields, pml)))
        assert whole <= halves, (kwargs, whole, halves)


def test_the_plan_refuses_rather_than_raises_on_a_numpy_host(product):
    """``None`` is the only refusal: a configuration this kernel does not carry
    must fall back to the array path, never raise into a caller that would
    otherwise have stepped correctly."""
    fields, pml = build()
    assert product.plan_offdiag_fused_electric_pair(fields, pml, ()) is None


# ---------------------------------------------------------------------------
# The closed form and the rotation
# ---------------------------------------------------------------------------

def test_the_wall_clear_axes_are_the_arrays_own_question(product):
    """``zero_metal_axes`` asks ``is_metallic and not is_mirrored`` — the grid's
    DECLARATION, which is the question ``_zero_metal`` asks, and deliberately not
    the resolved ghost rule."""
    from .triton_kernels import coverage as coverage_module

    fields, _pml = build()
    assert coverage_module.zero_metal_axes(fields.grid) == (True, True, False)
    folded, _pml = build(symmetry=("y",))
    assert coverage_module.zero_metal_axes(folded.grid)[1] is False


def test_the_clear_targets_the_components_with_a_zero_yee_shift():
    """``_zero_metal`` writes stored cell 0 of every component whose Yee shift on
    that axis is 0. The closed form's per-axis grouping has to be that table, and
    the table is read from ``fields.IYEE_SHIFTS`` rather than spelled here."""
    clear_by_axis = {axis: tuple(index for index, name
                                 in enumerate(("Dx", "Dy", "Dz"))
                                 if IYEE_SHIFTS[name][axis] == 0)
                     for axis in range(3)}
    assert clear_by_axis == {0: (1, 2), 1: (0, 2), 2: (0, 1)}
    source = _function_source(
        PACKAGE_DIR / "offdiag_fused_electric_pair.py", "_seam_resolve")
    for axis, letter in enumerate("XYZ"):
        block = source.split(f"if ZM_{letter}:")[1].split("if ZM_")[0]
        touched = tuple(sorted({int(line.strip()[1])
                                for line in block.splitlines()
                                if line.strip().startswith("v")}))
        assert touched == clear_by_axis[axis], (letter, touched)


def test_the_rotation_swaps_the_engines_own_references(product):
    """The plan owns twins and moves the ENGINE's attributes, so every later
    pass, read-back and monitor sees the freshly written buffer under the
    engine's own name. Exercised on a stand-in whose attributes are the six the
    plan rotates — the plan never reads anything else off it."""
    from .triton_kernels.offdiag_scratch_weld import ScratchWeldPairPlan

    class _Stand:
        pass

    stand = _Stand()
    live = {}
    twins = {}
    for name in product.ROTATED:
        live[name] = numpy.zeros((2, 2, 1), dtype=numpy.float32)
        twins[name] = numpy.ones((2, 2, 1), dtype=numpy.float32)
        setattr(stand, name, live[name])
    plan = ScratchWeldPairPlan("t", stand, twins, product.ROTATED, (2, 2, 1),
                               8, product.REPLACES)
    writes, reads = plan._resolve()
    assert [id(array) for array in reads] == [id(live[n]) for n in product.ROTATED]
    assert [id(array) for array in writes] == [id(twins[n]) for n in product.ROTATED]
    plan._rotate()
    for name in product.ROTATED:
        assert getattr(stand, name) is twins[name]
        assert plan.rotated[name] is live[name]


def test_an_aliased_twin_is_REFUSED_and_not_launched(product):
    """The whole design is that nothing written is read. A plan whose scratch and
    live buffer were the same allocation would be the 2026-08-20 in-place weld
    wearing this class's name, and would race exactly as that one did."""
    from .triton_kernels.offdiag_scratch_weld import ScratchWeldPairPlan

    class _Stand:
        pass

    stand = _Stand()
    twins = {}
    for name in product.ROTATED:
        array = numpy.zeros((2, 2, 1), dtype=numpy.float32)
        setattr(stand, name, array)
        twins[name] = array          # the alias
    plan = ScratchWeldPairPlan("t", stand, twins, product.ROTATED, (2, 2, 1),
                               8, product.REPLACES)
    with pytest.raises(RuntimeError, match="aliased"):
        plan._resolve()


def test_the_twin_table_refuses_an_unallocated_volume(product):
    """Binding a ``None`` twin gives a ``TypeError`` at Triton's argument
    marshalling, which reports the launcher and not the cause."""
    from .triton_kernels.offdiag_scratch_weld import twin_table

    class _Stand:
        Dx = numpy.zeros((2, 2, 1), dtype=numpy.float32)

    with pytest.raises(ValueError, match="not allocated"):
        twin_table(_Stand(), product.ROTATED)


# ---------------------------------------------------------------------------
# Transcription — the probe's own legs, run at the merge bar
# ---------------------------------------------------------------------------

def test_the_curl_half_is_the_certified_body_byte_for_byte(probe):
    row = probe.transcription_leg(FAMILY)
    assert row["passed"], row["findings"]
    assert row["measures"]["curl_lift_statements"] > 40, row["measures"]
    assert row["measures"]["term_needles_applied"] == 4
    assert row["measures"]["stray_displacement_loads"] == 0


def test_every_redirected_load_reads_the_certified_cell(probe):
    row = probe.tap_binding_leg(FAMILY)
    assert row["passed"], row["findings"]
    assert row["measures"]["taps_declared"] == 15
    assert row["measures"]["distinct_tap_cells"] == 12
    assert row["measures"]["term_call_sites"] == 9


def test_the_binding_leg_is_armed(probe):
    """A source-comparison leg that matched nothing would pass on any kernel at
    all. Six rewrites, one per class of substitution defect, each refused."""
    row = probe.arming_leg(FAMILY)
    assert row["passed"], row["findings"]
    assert row["measures"]["caught"] == row["measures"]["mutations"] == 6


def test_the_choreography_is_byte_identical_to_the_driver_pass_order(probe):
    """The scratch-output choreography against the driver's real pass order, two
    complete steps, every stored volume, uint32 — on every declared fixture and
    value class."""
    for fixture in probe.FIXTURES[FAMILY]:
        for value_class in probe.VALUE_CLASSES:
            row = probe.choreography_leg(FAMILY, fixture, value_class, 2,
                                         20260902)
            assert row["passed"], row


def test_every_armable_null_diverges(probe):
    walled = next(fixture for fixture in probe.FIXTURES[FAMILY]
                  if fixture["label"] == "walled_xy")
    for name, arms, reason in probe.NULLS:
        row = probe.null_leg(FAMILY, walled, "normal", name, arms, reason,
                             20260902)
        assert row["armable"], row
        assert row["differing_words"] > 0, row


# ---------------------------------------------------------------------------
# Wiring — this product is not in dispatch, and its gate knows it
# ---------------------------------------------------------------------------

def test_launch_and_fastpath_route_and_certify_this_module():
    """ROUTED SINCE 2026-09-15, and asserted where the routing lives.

    This test pinned the opposite — no mention in ``launch.py`` or ``fastpath.py``,
    because a product that appeared in ``plan_step`` without its own dispatch round
    would be served by a path nothing measured — until the shared plan base gained
    a ``warm`` that never rotates. The names are now asserted PRESENT in the three
    places that route, import and certify the product, under the label the release
    is keyed by; the dispatch round is the route campaign that drives that label.
    """
    from . import fastpath
    from .triton_kernels import launch

    label = "fused pair D (off-diagonal)"
    assert launch.CERTIFIED_FUSED_PRODUCTS[FAMILY]["label"] == label
    assert launch.CERTIFIED_FUSED_PAIR_ARMS[FAMILY] == ("PML", "off-diagonal")
    assert FAMILY in launch.SUPPORT_MODULES
    assert FAMILY in launch._certified_fused_product_modules()
    assert fastpath.ARM_CERTIFICATION[label][0] == FAMILY
    assert fastpath.FUSED_ARM_CONSTITUENTS[label] == ("PML", "off-diagonal")


class _RecordingKernel:
    """The ``kernel[grid](...)`` boundary, recorded and inert.

    Records every grid it is subscripted with and counts launches, and writes
    nothing, so the only state a warm can move is the plan's own and the engine's
    references — exactly what the two warm tests read.
    """

    def __init__(self):
        self.grids = []
        self.launches = 0

    def __getitem__(self, grid):
        self.grids.append(tuple(grid))

        def launch(*args, **kwargs):
            self.launches += 1

        return launch


def _shipped_plan_on_this_host(product, monkeypatch, kernel):
    """The SHIPPED builder on a NumPy host, past the one clause this host fails.

    The predicate's only refusal on :func:`build`'s grid is the backend clause —
    asserted here rather than assumed — so replacing that predicate for the build
    changes nothing about which configuration the plan is built for.
    """
    import types  # noqa: PLC0415

    from .triton_kernels.coverage import Coverage

    # ``_launch`` imports ``ENABLE_FP_FUSION`` from ``kernels``, which imports Triton
    # at its top. On a host without Triton that ONE name is supplied by a stand-in
    # carrying the shipped value — the technique the complex electric pair's
    # launch-signature test uses — so a launch body that started reading a second
    # name from ``kernels`` fails here rather than being accommodated.
    try:
        importlib.import_module("meep_gpu.triton_kernels.kernels")
    except ImportError:
        stand_in = types.ModuleType("meep_gpu.triton_kernels.kernels")
        stand_in.ENABLE_FP_FUSION = False
        monkeypatch.setitem(sys.modules, "meep_gpu.triton_kernels.kernels", stand_in)

    fields, pml = build()
    verdict = product.offdiag_fused_electric_pair_coverage(fields, pml, ())
    assert _reasons(verdict) == [], verdict.reasons
    monkeypatch.setattr(product, "offdiag_fused_electric_pair_coverage",
                        lambda *a, **k: Coverage(True, ()))
    plan = product.plan_offdiag_fused_electric_pair(fields, pml, sources=(),
                                                    block=64, kernel=kernel)
    assert plan is not None
    return fields, plan


def test_warm_launches_once_on_an_empty_grid_and_rotates_nothing(product, monkeypatch):
    """THE MECHANISM THE ROUTING RESTS ON, through ``fastpath.warm_plan``.

    The warm must launch once at grid ``(0,)``, count nothing, restore the grid,
    and leave every rotating reference — the engine's and the plan's twin table —
    exactly where it was. The step that follows must then launch at the real
    grid, count one, and rotate, so the warm is shown to have left a plan that
    still works rather than one that was merely left alone.
    """
    from . import fastpath

    kernel = _RecordingKernel()
    fields, plan = _shipped_plan_on_this_host(product, monkeypatch, kernel)
    live = {name: getattr(fields, name) for name in product.ROTATED}
    twins = dict(plan.rotated)
    grid = plan._grid
    assert grid != (0,)

    assert fastpath.warm_plan(plan) is None
    assert kernel.grids == [(0,)] and kernel.launches == 1
    assert plan.launches == 0
    assert plan._grid == grid
    for name in product.ROTATED:
        assert getattr(fields, name) is live[name], name
        assert plan.rotated[name] is twins[name], name

    plan.run()
    assert kernel.grids == [(0,), grid] and kernel.launches == 2
    assert plan.launches == 1
    for name in product.ROTATED:
        assert getattr(fields, name) is twins[name], name
        assert plan.rotated[name] is live[name], name


def test_the_empty_grid_fallback_is_the_path_warm_replaces(product, monkeypatch):
    """THE ARMED NULL: the path ``warm_plan`` took before ``warm`` existed.

    ``_warm_with_empty_grid`` empties ``_grid`` and calls ``run``, and on this plan
    that launches nothing, COUNTS a launch and ROTATES — the engine's D volumes
    end on the zero twins before any step. The warm test above passing while this
    one shows the defect is what makes that test a discrimination rather than a
    tautology.
    """
    from . import fastpath

    kernel = _RecordingKernel()
    fields, plan = _shipped_plan_on_this_host(product, monkeypatch, kernel)
    live = {name: getattr(fields, name) for name in product.ROTATED}
    twins = dict(plan.rotated)

    assert fastpath._warm_with_empty_grid(plan, plan.run) is None
    assert kernel.grids == [(0,)]
    assert plan.launches == 1
    for name in product.ROTATED:
        assert getattr(fields, name) is twins[name], name
        assert getattr(fields, name) is not live[name], name


def test_the_package_does_not_import_the_module_eagerly():
    text = (PACKAGE_DIR / "__init__.py").read_text(encoding="utf-8")
    assert "offdiag_fused_electric_pair" not in text


def test_the_kernel_accessor_explains_itself_without_triton(product):
    if product.offdiag_fused_curl_constitutive_D is not None:  # pragma: no cover
        pytest.skip("triton is installed on this host")
    with pytest.raises(ImportError, match="triton"):
        product.offdiag_fused_curl_constitutive_D_kernel()


def test_the_gate_exists_and_names_this_product(gate):
    assert FAMILY in gate.FAMILIES
    assert gate.FAMILIES[FAMILY]["carried_passes"] == ("zero_metal_D",)
    assert gate.SUPERSEDES["artifact"] == (
        "results/triton_fused_offdiag_electric_2026-08-20")


def test_every_mutation_the_gate_declares_is_ARMED_against_the_shipped_source(gate):
    """A needle that stopped matching is a DISARMED leg, and a disarmed leg
    reports a pass. Each must match exactly once."""
    text = (PACKAGE_DIR / "offdiag_fused_electric_pair.py").read_text(
        encoding="utf-8")
    for tag, _target, old, _new, _expected, _scored in gate.MUTATIONS[FAMILY]:
        assert text.count(old) == 1, tag


def test_every_mutation_carries_its_reason(gate):
    for tag, _target, _old, _new, _expected, _scored in gate.MUTATIONS[FAMILY]:
        assert gate.MUTATION_REASONS.get(tag), tag


def test_the_gate_declares_the_two_numbers_it_has_to_beat(gate):
    """The artifact this gate supersedes is named, and so are its numbers. A
    supersession that did not restate what it is superseding cannot be checked
    against it."""
    assert "42 of 60" in gate.SUPERSEDES["its_S1"]
    assert "4384" in gate.SUPERSEDES["its_S2"]
    assert set(gate.BLOCKS) == {64, 128, 256, 512, 1024}
    assert gate.S2_REPEATS >= 6


def test_the_gates_no_device_legs_all_pass_here(gate):
    for row in gate.no_device_legs([FAMILY]):
        assert row["passed"], row
