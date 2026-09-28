"""Laptop contracts for the ADE ``update_P`` admission with NO split-field absorber.

Closes residual group (B): 3 slots on ``absorber-1d.py``,
``TestAbsorber.test_absorber`` and ``material-dispersion.py``. Everything here
runs on the NumPy merge-bar machine — the optional-import contract, the
predicate's admission and every per-clause refusal on real Grid/Fields/PML
objects, the drive-field distinction pinned against ``fields.py`` itself, and an
in-test transcription of the ADE recurrence pinned BYTE-for-byte against
``PolarizationState.update``. The kernel's device bytes are the existing ADE
gate's; nothing here launches.

THE LOAD-BEARING HALF OF THIS FILE IS THE CONTROL. The inverted clause admits a
STORED-E drive, and binding the stored E under an ACTIVE layer is "the single
most likely silent wrong answer in dispersion" (fields.py:1149-1156) — the device
leg measured it at 3072 of 15360 words. So there is a test here that the two
drives really do differ under a live absorber, and one that the predicate refuses
an active layer by name.
"""

from __future__ import annotations

import builtins
import importlib
import pathlib
import sys
from types import SimpleNamespace

import numpy
import pytest

from meep_gpu.dispersion import PolarizationState, Susceptibility
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid, Mirror
from meep_gpu.pml import PML
from meep_gpu.triton_kernels import coverage as coverage_module

MODULE_NAME = "meep_gpu.triton_kernels.no_pml_ade"
PACKAGE_DIR = pathlib.Path(coverage_module.__file__).parent
MODULE_PATH = PACKAGE_DIR / "no_pml_ade.py"


@pytest.fixture(scope="module")
def npa():
    """The shipped module itself must be importable without a test-only stub."""
    return importlib.import_module(MODULE_NAME)


def build(absorber=False, storage="field", counts=(1, 1, 1), kind="lorentzian",
          cell_size=(0.8, 0.8, 0.8), dimensions=3, seed=19, complex_storage=False,
          **grid_kwargs):
    """A dispersive Grid/Fields/PML triple with NO active split-field layer.

    ``storage='field'`` is the ``mp.Absorber`` class this admission covers:
    ``enable_field_storage`` (stored E, no ``f_w``), an inert PML object, and
    optionally a conductivity — which is what ``mp.Absorber`` actually installs
    and which the device leg measured as making no difference to this sub-step
    (0 of 7680 words over four cycles).
    """
    grid = Grid(resolution=10.0, cell_size=cell_size, dimensions=dimensions,
                courant=0.35, xp=numpy, **grid_kwargs)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    if storage == "pml":
        fields.enable_pml_storage()
    elif storage == "field":
        fields.enable_field_storage()
    components = ("Ex", "Ey", "Ez")
    eps = numpy.full(grid.shape, 2.25, dtype=numpy.float32)
    fields.set_epsilon_volumes({c: eps for c in components},
                               {c: (1.0 / eps).astype(numpy.float32)
                                for c in components})
    if absorber:
        fields.set_d_conductivity(numpy.full(grid.shape, 0.2, dtype=numpy.float32))
        fields.set_b_conductivity(numpy.full(grid.shape, 0.2, dtype=numpy.float32))
    states = []
    for index in range(max(counts) if counts else 0):
        term = Susceptibility(frequency=1.0, gamma=0.1, kind=kind)
        sigmas = {name: (0.3 + 0.05 * index) if counts[axis] > index else 0.0
                  for axis, name in enumerate(components)}
        state = PolarizationState(term, sigmas, grid, numpy.float32)
        fields.polarizations.append(state)
        states.append(state)
    rng = numpy.random.default_rng(seed)
    for name in ("Dx", "Dy", "Dz", "Ex", "Ey", "Ez"):
        array = getattr(fields, name, None)
        if array is not None:
            array[...] = rng.uniform(-0.4, 0.4, size=grid.shape).astype(array.dtype)
    for state in states:
        for component in state.driven():
            for slot in ("P", "P_prev"):
                getattr(state, slot)[component][...] = rng.uniform(
                    -0.2, 0.2, size=grid.shape).astype(numpy.float32)
    thickness = 2 if storage == "pml" else 0
    return fields, PML(grid=grid, thickness=thickness), states


def reasons(npa, fields, pml, state, component="Ez"):
    return [r for r in npa.no_pml_ade_update_p_coverage(
        fields, pml, state, component).reasons if "not cupy" not in r]


# ---------------------------------------------------------------------------
# The optional dependency must stay optional, and the package must not eagerly load
# ---------------------------------------------------------------------------

def test_the_package_does_not_import_the_module_eagerly():
    for name in list(sys.modules):
        if name == MODULE_NAME:
            del sys.modules[name]
    importlib.import_module("meep_gpu.triton_kernels")
    assert MODULE_NAME not in sys.modules, (
        "the package imported this module at startup; every optional-package "
        "host must be able to import the engine without it")


def test_the_module_answers_coverage_with_triton_absent(monkeypatch):
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "triton" or name.startswith("triton."):
            raise ImportError("triton is blocked for this test")
        return real_import(name, *args, **kwargs)

    for name in [n for n in sys.modules if n.startswith("triton")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.delitem(sys.modules, MODULE_NAME, raising=False)
    monkeypatch.setattr(builtins, "__import__", blocked)
    module = importlib.import_module(MODULE_NAME)
    fields, pml, states = build()
    assert isinstance(
        module.no_pml_ade_update_p_coverage(fields, pml, states[0], "Ez").covered,
        bool)


def test_the_shared_clause_builders_it_composes_from_still_exist(npa):
    for name in npa.SHARED_CLAUSES:
        assert hasattr(coverage_module, name), (
            f"coverage.{name} was renamed; this module's restatement silently "
            f"drops a clause without this assertion")


def test_the_module_reaches_into_no_other_sessions_file_and_names_its_arm(npa):
    """It still imports nothing it does not own — and the claim it carries flipped.

    The module shipped with an "AWAITING WIRING" header, which was true: the
    closure round did not own ``launch.py``. ``plan_step``'s ``update_P`` block
    now chooses between this family and the incumbent, so that header would be a
    false statement about a file this module does not own. What a reader needs
    instead is where the choice is made and what still holds DISPATCH shut.
    """
    text = MODULE_PATH.read_text(encoding="utf-8")
    for forbidden in ("plan_step", "def plan_step", "fastpath", "cuda_kernels"):
        assert f"import {forbidden}" not in text
    assert "AWAITING WIRING" not in text
    assert "DISPATCH_BY_DEFAULT" in text, (
        "a wired family must name what holds dispatch shut, not claim it is "
        "unreachable")

    from meep_gpu.triton_kernels import launch
    assert "no_pml_ade" in launch.FAMILY_MODULES
    assert launch.no_pml_ade_update_p_coverage is not None


# ---------------------------------------------------------------------------
# The drive field — the whole point of the inverted clause
# ---------------------------------------------------------------------------

def test_drive_field_is_the_stored_E_without_a_layer_and_f_w_with_one():
    """Pinned against ``Fields.drive_field`` itself (fields.py:1158-1162).

    This is the fact the inverted clause rests on. If ``drive_field`` ever stops
    branching this way, the restatement below becomes a wrong admission rather
    than a narrow one, and this assertion is where that is caught.
    """
    fields, _pml, _states = build(storage="field")
    for component in ("Ex", "Ey", "Ez"):
        assert fields.drive_field(component) is getattr(fields, component)

    fields, _pml, _states = build(storage="pml")
    for component in ("Ex", "Ey", "Ez"):
        assert fields.drive_field(component) is getattr(fields, "f_w_" + component)


def _reference_update_p(state, drive, sigma, coefficients):
    """``PolarizationState.update`` (dispersion.py:679-691), transcribed.

    Including the three-buffer rotation, which is the part a cached plan gets
    wrong: this step's result lands in ``_scratch``, ``_scratch`` takes over the
    array that held ``P_prev``, and ``P``/``P_prev`` shift down.
    """
    c_now, c_prev, c_drive = coefficients
    for component in state["driven"]:
        w = drive(component)
        p = state["P"][component]
        p_prev = state["P_prev"][component]
        scratch = state["_scratch"]
        numpy.multiply(p, c_now, out=scratch)
        scratch += c_prev * p_prev
        scratch += c_drive * (sigma[component] * w)
        state["P"][component] = scratch
        state["P_prev"][component] = p
        state["_scratch"] = p_prev


def test_the_reference_matches_PolarizationState_update_on_the_admitted_class(npa):
    """The reference test for group (B), byte-for-byte, four complete cycles.

    Driven through ``fields.drive_field`` on BOTH sides — which is exactly what
    ``launch.AdeUpdatePPlan.run`` does — so the leg measures the admitted
    configuration and not a convenient re-derivation of it.
    """
    fields, pml, states = build(absorber=True)
    state = states[0]
    mirror = {
        "driven": tuple(state.driven()),
        "P": {c: state.P[c].copy() for c in state.driven()},
        "P_prev": {c: state.P_prev[c].copy() for c in state.driven()},
        "_scratch": state._scratch.copy(),
    }
    before = {c: state.P[c].copy() for c in state.driven()}
    for cycle in range(4):
        state.update(fields.drive_field, fields.grid.dt)
        _reference_update_p(mirror, fields.drive_field, state.sigma,
                            state._coefficients)
        for component in mirror["driven"]:
            assert mirror["P"][component].tobytes() == state.P[component].tobytes(), (
                f"P[{component}] diverged at cycle {cycle + 1}: max abs delta "
                f"{numpy.max(numpy.abs(mirror['P'][component] - state.P[component])):.3e}")
            assert (mirror["P_prev"][component].tobytes()
                    == state.P_prev[component].tobytes())
    moved = sum(int(numpy.count_nonzero(
        numpy.frombuffer(before[c].tobytes(), dtype=numpy.uint32)
        != numpy.frombuffer(state.P[c].tobytes(), dtype=numpy.uint32)))
        for c in mirror["driven"])
    assert moved > 0, "update_P moved no words: the identity is vacuous"


def test_each_term_of_that_reference_is_load_bearing(npa):
    """Drop one term at a time; each mutation must make the reference FAIL.

    A transcription that agrees with its source no matter what it says is not a
    transcription check. Three armed mutations, each launched and each caught.
    """
    fields, pml, states = build(absorber=True)
    caught = 0
    for dropped in ("c_now", "c_prev", "c_drive"):
        local_fields, _pml, local_states = build(absorber=True)
        state = local_states[0]
        mirror = {
            "driven": tuple(state.driven()),
            "P": {c: state.P[c].copy() for c in state.driven()},
            "P_prev": {c: state.P_prev[c].copy() for c in state.driven()},
            "_scratch": state._scratch.copy(),
        }
        c_now, c_prev, c_drive = state._coefficients
        mutated = {"c_now": (0.0, c_prev, c_drive),
                   "c_prev": (c_now, 0.0, c_drive),
                   "c_drive": (c_now, c_prev, 0.0)}[dropped]
        state.update(local_fields.drive_field, local_fields.grid.dt)
        _reference_update_p(mirror, local_fields.drive_field, state.sigma, mutated)
        if any(mirror["P"][c].tobytes() != state.P[c].tobytes()
               for c in mirror["driven"]):
            caught += 1
    assert caught == 3, (
        f"only {caught} of 3 coefficient mutations changed the answer: the "
        f"reference is insensitive to its own terms")
    del fields, pml, states


def test_the_CONTROL_the_two_drives_really_do_differ_under_a_live_absorber(npa):
    """3072 of 15360 words on the device; here it need only be nonzero.

    Without this the whole distinction could be cosmetic, and an admission that
    let the stored E be bound under a PML would look harmless.
    """
    from meep_gpu import stepping

    fields, pml, states = build(storage="pml")
    assert pml.is_active
    stepping.update_E(fields, pml)
    differing = 0
    for component in ("Ex", "Ey", "Ez"):
        f_w = fields.drive_field(component)
        stored = getattr(fields, component)
        differing += int(numpy.count_nonzero(
            numpy.frombuffer(f_w.tobytes(), dtype=numpy.uint32)
            != numpy.frombuffer(stored.tobytes(), dtype=numpy.uint32)))
    assert differing > 0, (
        "f_w and the stored E agreed everywhere under an ACTIVE absorber: the "
        "drive distinction this module inverts would then be untestable and the "
        "reference leg above would prove nothing")


# ---------------------------------------------------------------------------
# Admission, disjointness, and per-clause refusals
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("absorber", (False, True))
def test_the_predicate_admits_what_the_incumbent_refuses(npa, absorber):
    """Both with and without a conductivity — mp.Absorber installs one.

    The conductive case is a clause this predicate deliberately does NOT carry,
    and the device leg measured it at 0 of 7680 words over four cycles rather
    than assuming it.
    """
    fields, pml, states = build(absorber=absorber)
    for component in ("Ex", "Ey", "Ez"):
        assert reasons(npa, fields, pml, states[0], component) == [], component
        incumbent = coverage_module.ade_update_p_coverage(
            fields, states[0], component)
        assert any("f_w_" in r and "not allocated" in r for r in incumbent.reasons), (
            "the incumbent must still refuse — otherwise the two predicates "
            "overlap and a dispatcher would pick a drive pointer by ordering")


def test_the_two_families_are_disjoint_under_an_active_layer(npa):
    fields, pml, states = build(storage="pml")
    verdict = reasons(npa, fields, pml, states[0])
    assert any("an active PML layer is installed" in r for r in verdict), verdict
    assert any("3072/15360" in r for r in verdict), (
        "the refusal must carry the measurement that makes it load-bearing")
    # And the incumbent takes it, on this laptop's backend clause alone.
    assert [r for r in coverage_module.ade_update_p_coverage(
        fields, states[0], "Ez").reasons if "not cupy" not in r] == []


def test_pml_storage_with_an_inert_layer_is_refused_by_name(npa):
    """The mirror image of ``no_pml``'s clause 3b hazard, and just as silent.

    ``Fields`` in PML storage mode hands back ``f_w``, which ``update_E`` never
    writes on the no-absorber path — a frozen drive field, smooth and wrong.
    """
    fields, pml, states = build(storage="pml")
    inert = PML(grid=fields.grid, thickness=0)
    assert not inert.is_active
    verdict = reasons(npa, fields, inert, states[0])
    assert any("Fields is in PML storage mode" in r for r in verdict), verdict
    assert any("f_w_Ez is allocated" in r for r in verdict), verdict


def test_complex_storage_is_refused_as_an_UNBUILT_kernel_not_a_divergence(npa):
    """Group (I)'s leg RAISED ``KeyError: 'complex64'``; it did not diverge."""
    fields, pml, states = build(complex_storage=True)
    joined = " ".join(npa.no_pml_ade_update_p_coverage(
        fields, pml, states[0], "Ez").reasons)
    assert "KeyError: 'complex64'" in joined
    assert "unbuilt kernel" in joined


def test_a_recomputed_E_is_refused_with_update_E_s_early_return(npa):
    fields, pml, states = build(storage=None)
    assert not fields.stores_E
    verdict = reasons(npa, fields, pml, states[0])
    assert any("stepping.py:984" in r for r in verdict), verdict


def test_a_non_electric_component_is_refused_and_returns_early(npa):
    fields, pml, states = build()
    verdict = npa.no_pml_ade_update_p_coverage(fields, pml, states[0], "Hx")
    assert not verdict.covered
    assert any("outside" in r for r in verdict.reasons)


def test_a_state_that_does_not_drive_the_component_is_refused(npa):
    fields, pml, states = build(counts=(1, 1, 0))
    assert "Ez" not in states[0].driven()
    verdict = reasons(npa, fields, pml, states[0], "Ez")
    assert any("does not drive Ez" in r for r in verdict), verdict


def test_an_uncovered_susceptibility_kind_is_refused(npa):
    fields, pml, states = build()
    states[0].susceptibility = SimpleNamespace(kind="noisy_lorentzian")
    verdict = reasons(npa, fields, pml, states[0])
    assert any("noisy_lorentzian" in r for r in verdict), verdict


def test_a_non_finite_coefficient_is_refused_by_name(npa):
    fields, pml, states = build()
    c_now, c_prev, c_drive = states[0]._coefficients
    states[0]._coefficients = (float("nan"), c_prev, c_drive)
    verdict = reasons(npa, fields, pml, states[0])
    assert any("c_now" in r and "not finite" in r for r in verdict), verdict


def test_coefficients_built_for_another_dt_are_refused(npa):
    fields, pml, states = build()
    other = Grid(resolution=10.0, cell_size=(0.8, 0.8, 0.8), dimensions=3,
                 courant=0.2, xp=numpy)
    states[0].grid = other
    verdict = reasons(npa, fields, pml, states[0])
    assert any("different dt" in r for r in verdict), verdict


@pytest.mark.parametrize("slot", ("P", "P_prev"))
def test_a_malformed_pole_buffer_is_refused_by_name(npa, slot):
    fields, pml, states = build()
    buffers = getattr(states[0], slot)
    buffers["Ez"] = buffers["Ez"][::-1]       # right shape, right dtype, reversed
    verdict = reasons(npa, fields, pml, states[0])
    assert any(f"{slot}[Ez]" in r and "C-contiguous" in r for r in verdict), verdict


def test_a_missing_scratch_is_refused(npa):
    fields, pml, states = build()
    states[0]._scratch = None
    verdict = reasons(npa, fields, pml, states[0])
    assert any("_scratch[Ez] is not allocated" in r for r in verdict), verdict


def test_a_sigma_that_is_neither_scalar_nor_volume_is_refused(npa):
    fields, pml, states = build()
    states[0].sigma["Ez"] = object()
    verdict = reasons(npa, fields, pml, states[0])
    assert any("neither a scalar nor a volume" in r for r in verdict), verdict


def test_a_volume_sigma_is_admitted_and_a_scalar_one_too(npa):
    fields, pml, states = build()
    assert reasons(npa, fields, pml, states[0]) == []      # scalar sigma
    states[0].sigma["Ez"] = numpy.full(fields.grid.shape, 0.3, dtype=numpy.float32)
    assert reasons(npa, fields, pml, states[0]) == []      # volume sigma
    assert coverage_module.sigma_is_volume(states[0], "Ez") is True


def test_a_drive_field_that_is_not_the_stored_array_is_refused(npa):
    """The identity clause, not merely a layout clause."""
    fields, pml, states = build()
    original = fields.drive_field
    impostor = fields.Ez.copy()
    fields.drive_field = (lambda component:
                          impostor if component == "Ez" else original(component))
    verdict = reasons(npa, fields, pml, states[0])
    assert any("is not the stored Ez array" in r for r in verdict), verdict


def test_an_unreadable_drive_field_is_refused_not_assumed(npa):
    fields, pml, states = build()

    def raising(component):
        raise RuntimeError("no drive for you")

    fields.drive_field = raising
    verdict = reasons(npa, fields, pml, states[0])
    assert any("raised" in r for r in verdict), verdict


def test_a_fold_is_NOT_a_clause_but_the_folded_extent_is_checked(npa):
    """This sub-step reads no neighbour, so a fold is admitted — deliberately.

    What a fold changes is the stored extent, and every buffer is required to be
    exactly ``grid.shape``, which on a folded grid IS the folded extent. A buffer
    built for the UNFOLDED extent is refused by that clause, which is the test.
    """
    grid = Grid(resolution=10.0, cell_size=(1.6, 2.0, 0.0), dimensions=2,
                courant=0.35, boundaries="periodic",
                symmetry=(Mirror("Y", 1),), xp=numpy)
    fields = Fields(grid=grid)
    fields.enable_field_storage()
    eps = numpy.full(grid.shape, 2.25, dtype=numpy.float32)
    fields.set_epsilon_volumes({c: eps for c in ("Ex", "Ey", "Ez")},
                               {c: (1.0 / eps).astype(numpy.float32)
                                for c in ("Ex", "Ey", "Ez")})
    term = Susceptibility(frequency=1.0, gamma=0.1, kind="lorentzian")
    state = PolarizationState(term, {"Ex": 0.3, "Ey": 0.3, "Ez": 0.3},
                              grid, numpy.float32)
    fields.polarizations.append(state)
    pml = PML(grid=grid, thickness=0)
    assert reasons(npa, fields, pml, state) == [], "a fold must be admitted here"

    wrong = numpy.zeros((grid.shape[0], grid.shape[1] + 1, grid.shape[2]),
                        dtype=numpy.float32)
    state.P["Ez"] = wrong
    assert any("P[Ez] shape" in r for r in reasons(npa, fields, pml, state))


# ---------------------------------------------------------------------------
# Mutations — each inverted clause must be load-bearing
# ---------------------------------------------------------------------------

def test_mutation_dropping_the_active_layer_clause_breaks_disjointness(npa,
                                                                       monkeypatch):
    original = npa._inert_layer_reasons

    def mutated(fields, pml, component):
        return [r for r in original(fields, pml, component)
                if "an active PML layer is installed" not in r
                and "Fields is in PML storage mode" not in r
                and "is allocated" not in r
                and "is not the stored" not in r]

    monkeypatch.setattr(npa, "_inert_layer_reasons", mutated)
    fields, pml, states = build(storage="pml")
    assert reasons(npa, fields, pml, states[0]) == [], (
        "the mutation is DISARMED: the inverted clause was not what kept the "
        "PML run out")
    assert [r for r in coverage_module.ade_update_p_coverage(
        fields, states[0], "Ez").reasons if "not cupy" not in r] == [], (
        "and the incumbent admits it too — that overlap is what the clause "
        "exists to prevent, and the wrong pick is 3072/15360 words")


def test_the_builder_is_all_or_nothing_per_state(npa):
    """One refused component refuses the whole state — the shared scratch is why."""
    fields, pml, states = build()
    states[0].P["Ey"] = states[0].P["Ey"][::-1]
    assert npa.plan_no_pml_ade_update_p(fields, pml, states[0]) is None
    assert reasons(npa, fields, pml, states[0], "Ez") == [], (
        "Ez alone is still covered, which is why an all-or-nothing builder has "
        "to be the explicit rule rather than a consequence")


def test_the_builder_answers_none_on_this_laptop(npa):
    fields, pml, states = build()
    assert npa.plan_no_pml_ade_update_p(fields, pml, states[0]) is None
    empty = SimpleNamespace(driven=lambda: ())
    assert npa.plan_no_pml_ade_update_p(fields, pml, empty) is None


def test_the_module_records_the_measurement_that_licensed_the_admission(npa):
    text = MODULE_PATH.read_text(encoding="utf-8")
    for case, record in npa.MEASURED["cases"].items():
        assert case in text, f"{case} is not recorded in the module"
        assert f"{record['differing']} / {record['compared']}" in text
    assert npa.MEASURED["cycles"] == 8
    assert npa.MEASURED["cases"]["B_ade_wrong_drive_CONTROL"]["verdict"] == "DIVERGENT"


def test_the_admission_itself_was_measured_not_only_the_kernel_body(npa):
    """The ``cases`` block licenses the KERNEL; ``own_builder`` licenses THIS module.

    The distinction is the whole reason this test exists. The triage legs reach
    the certified body by patching the INCUMBENT predicate in front of the
    INCUMBENT builder, which is honest about the body and says nothing about
    :func:`no_pml_ade_update_p_coverage` or :func:`plan_no_pml_ade_update_p` —
    the two objects that will actually do the admitting, and between which sit a
    clause set and an array resolution where this campaign's live defects have
    been. A separate leg builds through THIS module with THIS module's predicate
    unpatched, and the module must still say so.
    """
    own = npa.MEASURED["own_builder"]
    text = MODULE_PATH.read_text(encoding="utf-8")
    assert own["artifact"] in text
    assert own["cases"], "no leg through this module's own builder is recorded"
    for case, record in own["cases"].items():
        assert case in text, f"{case} is not recorded in the module"
        assert record["differing"] == 0, (
            f"{case} is recorded as diverging; the admission is not licensed")
        assert f"{record['differing']} / {record['compared']}" in text
        # Non-vacuity is a claim about the sub-step alone, per cycle, and the
        # number must be small enough that it CANNOT be the perturbation's
        # footprint — the defect that made the superseded artifact's figure
        # unable to distinguish a live update from a dead one.
        assert 0 < record["moved"] < record["compared"]
