"""Laptop contracts for the SCRATCH-OUTPUT off-diagonal D/E weld (FOLDED).

The unfolded family's suite (``test_triton_offdiag_fused_electric_pair.py``)
holds the contracts the two share; this one holds what the fold adds, and the
fold is where this family can go wrong in ways that compile and run:

* TWO in-seam passes are carried instead of one, and their ORDER is load-bearing
  at the one cell that is on both a fold plane and a wall;
* the near fill is a READ-SIDE redirect whose parity is a compile-time constant
  taken from the plane, while the constitutive half's down-shift ghost weight is
  a RUNTIME scalar taken from the same plane — two different facts about the same
  fold, and crossing them is a plane of wrong values;
* ``fill_folded_far_ghosts_D`` is REFUSED, by the pass's own liveness condition
  rather than by boundary code, and the refusal is MEASURED rather than argued.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pathlib
import sys

import numpy
import pytest

from .fields import IYEE_SHIFTS, Fields, mirror_parity
from .grid import Grid, Mirror
from .pml import PML

HERE = pathlib.Path(__file__).parent
PACKAGE_DIR = HERE / "triton_kernels"
PARITY = HERE.parent / "parity" / "meep_gpu"
MODULE = "meep_gpu.triton_kernels.folded_offdiag_fused_electric_pair"
FAMILY = "folded_offdiag_fused_electric_pair"


@pytest.fixture(scope="module")
def product():
    return importlib.import_module(MODULE)


@pytest.fixture(scope="module")
def probe():
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


def build(offdiag=True, poles=0, thickness=0.2, boundaries=None,
          symmetry=(("Y", 1),), complex_storage=False, sigma=None,
          dimensions=2, cell=(1.2, 1.2, 0.0)):
    grid = Grid(resolution=10.0, cell_size=cell, dimensions=dimensions,
                boundaries=boundaries or {"x": "metallic", "y": "metallic",
                                          "z": "periodic"},
                symmetry=tuple(Mirror(axis, phase) for axis, phase in symmetry))
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    shape = grid.shape
    if offdiag:
        diagonal = {name: numpy.full(shape, 2.25, dtype=numpy.float32)
                    for name in ("Ex", "Ey", "Ez")}
        inverse = {name: numpy.full(shape, 1.0 / 2.25, dtype=numpy.float32)
                   for name in ("Ex", "Ey", "Ez")}
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
    faces = {}
    if thickness:
        for axis, letter in enumerate("xyz"):
            if dimensions == 2 and letter == "z":
                continue
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


def _reasons(verdict):
    return [reason for reason in verdict.reasons if "not cupy" not in reason]


def _function_source(path, name):
    text = pathlib.Path(path).read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(text, node)
    raise AssertionError(f"{path} declares no function {name}")


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------

def test_the_product_declares_FOUR_call_sites_not_three(product):
    """One more than the unfolded family: the mirror fill is CARRIED here.

    ``REPLACES`` is a claim about what one launch performs, and this family's
    closed form really does resolve ``fill_symmetry_bc_D`` per cell.
    """
    assert product.REPLACES == ("step_D", "fill_symmetry_bc_D", "zero_metal_D",
                                "update_E")
    assert set(product.REFUSED_IN_SEAM_PASSES) == {"fill_folded_far_ghosts_D"}
    assert product.CARRIES_DEPOSIT_REPAIR is False
    assert product.REPAIR_PATHS == ()


def test_replaces_is_the_drivers_own_call_order(product):
    from . import driver as driver_module

    source = inspect.getsource(driver_module.FdtdDriver.step)
    seam = source.split("fast.dispatch(\"step_D\"")[1].split(
        "fast.dispatch(\"update_P\"")[0]
    positions = [0 if name == "step_D" else seam.index(name)
                 for name in product.REPLACES]
    assert positions == sorted(positions), (product.REPLACES, positions)


# ---------------------------------------------------------------------------
# The fold's two facts
# ---------------------------------------------------------------------------

def test_the_fill_phase_and_the_ghost_weight_are_DIFFERENT_facts(product):
    """One plane, two derived quantities, and they are not the same number.

    The near fill images ``cell 0 = phase * cell 2`` for a Yee-shift-0 component,
    so its parity is the plane's phase unchanged. The constitutive half's DOWN
    shift weights the ghost lane by ``mirror_parity(D_partner, axis, phase)``,
    and every D component has Yee shift 1 on its OWN axis, so that one is
    ``-phase``. A product that used one where the other belongs would apply the
    wrong sign on the fold plane — and on an EVEN plane the two differ only in
    sign, which is exactly the class a float comparison is blind to.
    """
    for phase in (1, -1):
        fields, _pml = build(symmetry=(("Y", phase),))
        grid = fields.grid
        assert product.mirror_fill_phases(grid) == (1, phase, 1)
        weights = product.mirror_ghost_weights(grid)
        assert weights[1] == float(mirror_parity("Dy", 1, phase))
        assert weights[1] == -float(phase)
        assert product.mirror_fill_phases(grid)[1] != weights[1]


def test_the_fill_phase_matches_the_certified_fills_own_constexpr(product):
    """``symmetry.ghost_fill_axis_entries`` is what the certified mirror-fill
    kernel is launched with. The closed form's ``PH_*`` must be that number, per
    axis, or the weld images the ghost differently from the pass it replaces."""
    from .triton_kernels import symmetry as symmetry_module

    for phase in (1, -1):
        fields, _pml = build(symmetry=(("Y", phase),))
        entries = symmetry_module.ghost_fill_axis_entries(fields.grid, "D")
        by_axis = {entry["axis"]: entry["phase"] for entry in entries}
        derived = product.mirror_fill_phases(fields.grid)
        for axis, expected in by_axis.items():
            assert derived[axis] == expected, (axis, derived, by_axis)


def test_the_near_fill_touches_only_the_shift_zero_components():
    """MEEP's ``little_owned_corner0(c) = little_corner + 2 - iyee_shift(c)``:
    cell 0 is unowned exactly where the Yee shift on that axis is 0. The closed
    form's per-component structure has to be that table, read from
    ``fields.IYEE_SHIFTS`` rather than spelled in the test."""
    filled = {axis: tuple(index for index, name in enumerate(("Dx", "Dy", "Dz"))
                          if IYEE_SHIFTS[name][axis] == 0)
              for axis in range(3)}
    assert filled == {0: (1, 2), 1: (0, 2), 2: (0, 1)}
    source = _function_source(
        PACKAGE_DIR / "folded_offdiag_fused_electric_pair.py", "_d_final")
    # Each axis's redirect is guarded by `COMP != <own axis>`, which is the same
    # table said the other way round.
    lines = source.splitlines()
    for axis, letter in enumerate("XYZ"):
        position = next(index for index, line in enumerate(lines)
                        if line.strip() == f"if MG_{letter}:")
        assert f"tl.where(near_{letter.lower()}, MIRROR_ROW" in lines[position + 1], (
            axis, lines[position + 1])
        assert f"COMP != {axis}" in lines[position - 1], (axis, lines[position - 1])


def test_the_mirror_row_is_the_engines_own_constant(product):
    from . import stepping
    from .triton_kernels.offdiag_scratch_weld import MIRROR_ROW

    assert MIRROR_ROW == stepping.MIRROR_SOURCE_INDEX == 2


def test_the_wall_clear_and_the_fold_never_share_an_axis(product):
    """``_zero_metal`` skips a folded axis for a reason worth 1.28e+00 relative
    L2 (stepping.py:2257-2277): its stored cell 0 holds the fold's ghost, not a
    wall. The plan REFUSES a binding that declared both on one axis rather than
    resolving it silently."""
    from .triton_kernels import coverage as coverage_module

    fields, _pml = build()
    zero_metal = coverage_module.zero_metal_axes(fields.grid)
    mirrored = tuple(fields.grid.is_mirrored(axis) for axis in range(3))
    assert not any(a and b for a, b in zip(zero_metal, mirrored))
    with pytest.raises(ValueError, match="both folded and declared"):
        product.FoldedOffdiagFusedElectricPairPlan(
            (2, 2, 1), 0.5, (1, 2, 0), (False, True, False), (0, 0, 0),
            (1, 1, 1), (1.0, -1.0, 1.0), 8, object(),
            {name: numpy.zeros((2, 2, 1), dtype=numpy.float32)
             for name in product.ROTATED},
            [], [], [], [], [], [None] * 6, [])


def test_the_plan_refuses_a_fold_with_no_readable_phase(product):
    """The fill's parity is a compile-time constant. A zero is neither ``+1`` nor
    ``-1`` and would install neither; the plan refuses rather than launching."""
    with pytest.raises(ValueError, match="fill phase"):
        product.FoldedOffdiagFusedElectricPairPlan(
            (2, 2, 1), 0.5, (1, 2, 0), (False, False, False), (0, 0, 0),
            (1, 0, 1), (1.0, -1.0, 1.0), 8, object(),
            {name: numpy.zeros((2, 2, 1), dtype=numpy.float32)
             for name in product.ROTATED},
            [], [], [], [], [], [None] * 6, [])


def test_the_plan_refuses_a_ghost_weight_that_is_not_a_parity(product):
    with pytest.raises(ValueError, match="ghost weight"):
        product.FoldedOffdiagFusedElectricPairPlan(
            (2, 2, 1), 0.5, (1, 2, 0), (False, False, False), (0, 0, 0),
            (1, 1, 1), (1.0, 0.5, 1.0), 8, object(),
            {name: numpy.zeros((2, 2, 1), dtype=numpy.float32)
             for name in product.ROTATED},
            [], [], [], [], [], [None] * 6, [])


# ---------------------------------------------------------------------------
# The far-ghost refusal, and its evidence
# ---------------------------------------------------------------------------

def test_the_far_ghost_refusal_asks_the_passs_own_liveness_condition(product):
    """Not "is this axis folded periodic" but "does it store MEEP's far slot".

    The difference is a served row: a folded periodic axis with no extra stored
    slot is one where ``fill_folded_far_ghosts_D`` writes nothing, and refusing
    it for carrying a boundary code would be a predicate narrower than its
    product for no reason. ``stepping._stored_past_owned`` is the condition, read
    rather than restated.
    """
    from . import stepping

    for boundaries, expected in (
            ({"x": "metallic", "y": "metallic", "z": "periodic"},
             (False, False, False)),
            ({"x": "metallic", "y": "periodic", "z": "periodic"},
             (False, True, False))):
        fields, _pml = build(boundaries=boundaries)
        assert product.far_ghost_axes(fields.grid) == expected, boundaries
        assert product.far_ghost_axes(fields.grid) == tuple(
            stepping._stored_past_owned(fields.grid, axis) for axis in range(3))


def test_the_predicate_refuses_a_stored_far_slot_by_name(product):
    fields, pml = build(boundaries={"x": "metallic", "y": "periodic",
                                    "z": "periodic"})
    reasons = _reasons(product.folded_offdiag_fused_electric_pair_coverage(
        fields, pml, ()))
    assert any("far ghost slot" in reason for reason in reasons), reasons


def test_the_far_ghost_refusal_is_MEASURED_and_not_argued(probe):
    """The refusal fixture, walked. On a grid that stores the far slot the weld
    model — which does not carry that pass — must DISAGREE with the driver's pass
    order. A refusal no fixture separates is a guess."""
    fixture = next(entry for entry in probe.FIXTURES[FAMILY]
                   if entry["label"] == "fold_y_periodic_far")
    assert fixture["admitted"] is False
    row = probe.choreography_leg(FAMILY, fixture, "normal", 2, 20260902)
    assert row["passed"], row
    assert all(count > 0 for count in row["differing_words_per_step"]), row


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------

def test_the_predicate_admits_the_configuration_this_cell_is(product):
    fields, pml = build()
    assert _reasons(product.folded_offdiag_fused_electric_pair_coverage(
        fields, pml, ())) == []


@pytest.mark.parametrize("label,needle,kwargs", [
    ("no absorber", "no active PML", {"thickness": 0}),
    ("no fold", "not a composition candidate", {"symmetry": ()}),
    ("complex storage", "complex", {"complex_storage": True}),
    ("a conductivity", "conductivity", {"sigma": 0.4}),
    ("a pole", "susceptibility", {"poles": 1}),
    ("no off-diagonal row", "off-diagonal chi1inv row", {"offdiag": False}),
    ("an in-seam electric source", "STENCIL", {}),
])
def test_the_predicate_refuses_by_name(product, label, needle, kwargs):
    fields, pml = build(**kwargs)
    sources = (_Electric(),) if label == "an in-seam electric source" else ()
    reasons = _reasons(product.folded_offdiag_fused_electric_pair_coverage(
        fields, pml, sources))
    assert any(needle in reason for reason in reasons), (label, reasons)


def test_an_undeclared_source_list_is_refused(product):
    fields, pml = build()
    reasons = _reasons(product.folded_offdiag_fused_electric_pair_coverage(
        fields, pml, None))
    assert any("not declared" in reason for reason in reasons), reasons


def test_the_predicate_and_the_two_certified_halves_agree(product):
    from .triton_kernels import symmetry as symmetry_module
    from .triton_kernels.folded_offdiag_update_e import (
        folded_offdiag_composition_coverage,
    )

    for kwargs in ({}, {"thickness": 0}, {"sigma": 0.4}, {"poles": 1},
                   {"offdiag": False}, {"symmetry": ()}):
        fields, pml = build(**kwargs)
        whole = not _reasons(
            product.folded_offdiag_fused_electric_pair_coverage(fields, pml, ()))
        halves = (not _reasons(symmetry_module.folded_composition_curl_coverage(
                      fields, pml, "step_D"))
                  and not _reasons(folded_offdiag_composition_coverage(
                      fields, pml)))
        assert whole <= halves, (kwargs, whole, halves)


def test_the_two_welds_PARTITION_the_space_rather_than_overlap(product):
    """Exactly one of the two off-diagonal welds may admit any configuration.

    ``launch._select_slot`` empties a slot with more than one admitter, so an
    overlap does not pick a winner — it costs BOTH products the cell. The
    unfolded family refuses a fold through ``coverage._grid_reasons`` clause 5;
    this one refuses an unfolded grid through the COMPOSITION split, and the
    standalone folded predicates (which deliberately admit both, so their own
    gates can prove reduction) are what would have broken it.
    """
    from .triton_kernels import offdiag_fused_electric_pair as unfolded

    for symmetry in ((("Y", 1),), (("X", -1), ("Y", 1)), ()):
        fields, pml = build(symmetry=symmetry)
        admits = {
            "folded": not _reasons(
                product.folded_offdiag_fused_electric_pair_coverage(
                    fields, pml, ())),
            "unfolded": not _reasons(
                unfolded.offdiag_fused_electric_pair_coverage(fields, pml, ())),
        }
        assert sum(admits.values()) == 1, (symmetry, admits)
        assert admits["folded"] is bool(symmetry), (symmetry, admits)


def test_the_plan_refuses_rather_than_raises_on_a_numpy_host(product):
    fields, pml = build()
    assert product.plan_folded_offdiag_fused_electric_pair(fields, pml, ()) is None


# ---------------------------------------------------------------------------
# Transcription — the probe's own legs, at the merge bar
# ---------------------------------------------------------------------------

def test_the_curl_half_is_the_certified_FOLDED_body_byte_for_byte(probe):
    row = probe.transcription_leg(FAMILY)
    assert row["passed"], row["findings"]
    assert row["measures"]["curl_lift_statements"] > 60, row["measures"]
    assert row["measures"]["term_needles_applied"] == 4
    assert row["measures"]["stray_displacement_loads"] == 0


def test_every_redirected_load_reads_the_certified_cell(probe):
    row = probe.tap_binding_leg(FAMILY)
    assert row["passed"], row["findings"]
    assert row["measures"]["taps_declared"] == 15
    assert row["measures"]["distinct_tap_cells"] == 12
    # The fold's two extra carried arguments are what make this family's binding
    # check stronger than the unfolded one's.
    assert "w_d" in row["measures"]["carried_arguments"]
    assert "MG" in row["measures"]["carried_arguments"]


def test_the_binding_leg_is_armed_including_the_two_fold_defects(probe):
    row = probe.arming_leg(FAMILY)
    assert row["passed"], row["findings"]
    assert row["measures"]["caught"] == row["measures"]["mutations"] == 8
    caught = {entry["mutation"] for entry in row["mutations"] if entry["caught"]}
    assert "ghost_weight_taken_from_the_wrong_axis" in caught
    assert "ghost_arm_taken_from_the_wrong_axis" in caught


def test_the_choreography_is_byte_identical_on_every_admitted_fixture(probe):
    for fixture in probe.FIXTURES[FAMILY]:
        if not fixture.get("admitted", True):
            continue
        for value_class in probe.VALUE_CLASSES:
            row = probe.choreography_leg(FAMILY, fixture, value_class, 2,
                                         20260902)
            assert row["passed"], row


def test_every_armable_null_diverges(probe):
    fixture = probe.FIXTURES[FAMILY][0]
    for name, arms, reason in probe.NULLS:
        row = probe.null_leg(FAMILY, fixture, "normal", name, arms, reason,
                             20260902)
        assert row["armable"], row
        assert row["differing_words"] > 0, row


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

def test_launch_and_fastpath_route_and_certify_this_module():
    """ROUTED SINCE 2026-09-15, and asserted where the routing lives.

    This test pinned the opposite — no mention in ``launch.py`` or ``fastpath.py``
    — until the shared plan base gained a ``warm`` that never rotates. The names
    are now asserted PRESENT in the three places that route, import and certify
    the product, under the label the release is keyed by.
    """
    from . import fastpath
    from .triton_kernels import launch

    label = "fused pair D (folded off-diagonal)"
    assert launch.CERTIFIED_FUSED_PRODUCTS[FAMILY]["label"] == label
    assert launch.CERTIFIED_FUSED_PAIR_ARMS[FAMILY] == ("folded PML",
                                                       "folded off-diagonal")
    assert FAMILY in launch.SUPPORT_MODULES
    assert FAMILY in launch._certified_fused_product_modules()
    assert fastpath.ARM_CERTIFICATION[label][0] == FAMILY
    assert fastpath.FUSED_ARM_CONSTITUENTS[label] == ("folded PML",
                                                      "folded off-diagonal")


class _RecordingKernel:
    """The ``kernel[grid](...)`` boundary, recorded and inert.

    Records every grid it is subscripted with and counts launches, and writes
    nothing, so the only state a warm can move is the plan's own and the engine's
    references — exactly what the warm test reads.
    """

    def __init__(self):
        self.grids = []
        self.launches = 0

    def __getitem__(self, grid):
        self.grids.append(tuple(grid))

        def launch(*args, **kwargs):
            self.launches += 1

        return launch


def test_warm_launches_once_on_an_empty_grid_and_rotates_nothing(product, monkeypatch):
    """THE MECHANISM THE ROUTING RESTS ON, through ``fastpath.warm_plan``.

    Built by the SHIPPED builder on a NumPy host: the predicate's only refusal
    here is its backend clause (asserted), so that clause alone is replaced for
    the build. The warm must launch once at grid ``(0,)``, count nothing, restore
    the grid, and leave every rotating reference where it was — and the step
    that follows must launch at the real grid, count one, and rotate.
    """
    import types  # noqa: PLC0415

    from . import fastpath
    from .triton_kernels.coverage import Coverage

    # ``_launch`` imports ``ENABLE_FP_FUSION`` from ``kernels``, which imports Triton
    # at its top; on a host without Triton that ONE name comes from a stand-in, as in
    # the unfolded family's suite.
    try:
        importlib.import_module("meep_gpu.triton_kernels.kernels")
    except ImportError:
        stand_in = types.ModuleType("meep_gpu.triton_kernels.kernels")
        stand_in.ENABLE_FP_FUSION = False
        monkeypatch.setitem(sys.modules, "meep_gpu.triton_kernels.kernels", stand_in)

    fields, pml = build()
    verdict = product.folded_offdiag_fused_electric_pair_coverage(fields, pml, ())
    assert _reasons(verdict) == [], verdict.reasons
    monkeypatch.setattr(product, "folded_offdiag_fused_electric_pair_coverage",
                        lambda *a, **k: Coverage(True, ()))
    kernel = _RecordingKernel()
    plan = product.plan_folded_offdiag_fused_electric_pair(
        fields, pml, sources=(), block=64, kernel=kernel)
    assert plan is not None
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


def test_the_package_does_not_import_the_module_eagerly():
    text = (PACKAGE_DIR / "__init__.py").read_text(encoding="utf-8")
    assert "folded_offdiag_fused_electric_pair" not in text


def test_the_kernel_accessor_explains_itself_without_triton(product):
    if product.folded_offdiag_fused_curl_constitutive_D is not None:
        pytest.skip("triton is installed on this host")  # pragma: no cover
    with pytest.raises(ImportError, match="triton"):
        product.folded_offdiag_fused_curl_constitutive_D_kernel()


def test_the_gate_exists_and_names_this_product(gate):
    assert FAMILY in gate.FAMILIES
    assert gate.FAMILIES[FAMILY]["carried_passes"] == (
        "fill_symmetry_bc_D", "zero_metal_D")


def test_every_mutation_the_gate_declares_is_ARMED_against_the_shipped_source(gate):
    text = (PACKAGE_DIR / "folded_offdiag_fused_electric_pair.py").read_text(
        encoding="utf-8")
    for tag, _target, old, _new, _expected, _scored in gate.MUTATIONS[FAMILY]:
        assert text.count(old) == 1, tag


def test_every_mutation_carries_its_reason(gate):
    for tag, _target, _old, _new, _expected, _scored in gate.MUTATIONS[FAMILY]:
        assert gate.MUTATION_REASONS.get(tag), tag


def test_the_two_parity_mutations_are_SCORED_ON_AN_ODD_PLANE(gate):
    """A mutation scored where it cannot fire is a disarmed leg wearing a pass.

    Both of this family's parity mutations are invisible on an EVEN plane —
    ``PH * value`` IS ``value`` at phase +1, and the pass-order swap leaves
    ``+0.0`` either way — and the first device run of this gate declared both
    CAUGHT and measured both NULL for exactly that reason. Each now names the odd
    fixture it is scored on, and this test is what stops that field being dropped.
    """
    scored = {tag: fixture
              for tag, _t, _o, _n, _e, fixture in gate.MUTATIONS[FAMILY]}
    assert scored["m_fill_phase_dropped"] == "fold_y_odd_walled"
    assert scored["m_clear_before_fill"] == "fold_y_odd_walled"
    labels = {entry["label"] for entry in gate.FAMILIES[FAMILY]["fixtures"]}
    assert "fold_y_odd_walled" in labels
    # ...and the odd fixture really is odd, read off the fixture rather than
    # trusted from its name.
    odd = next(entry for entry in gate.FAMILIES[FAMILY]["fixtures"]
               if entry["label"] == "fold_y_odd_walled")
    assert odd["symmetry"] == (("Y", -1),)


def test_the_gates_no_device_legs_all_pass_here(gate):
    for row in gate.no_device_legs([FAMILY]):
        assert row["passed"], row
