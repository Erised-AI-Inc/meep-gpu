"""Laptop contracts for ``update_E`` with a mirror fold AND a susceptibility live.

Closes residual group (C): 4 slots on four ``TestLoadDump`` 2-D rows, each of
which reaches FULL whole-step coverage once the arm is wired. Everything here
runs on the NumPy merge-bar machine — the optional-import contract, the
predicate's admission and every per-clause refusal on real Grid/Fields/PML
objects, disjointness from BOTH incumbents in both directions, and an in-test
transcription of the dispersive ``update_E`` sub-step pinned BYTE-for-byte
against ``stepping.update_E`` on the newly admitted configuration.

The device leg measured the certified body against ``stepping.update_E`` here at
0 of 69120 words differing over EIGHT complete cycles, with at least 13824 words
moved by the sub-step itself on each one and a +-0 lattice held live in D so a
no-op could not pass as identity; and again at 0 of 9360 on a ROW-SHAPED 2-D grid
(results/residual_closure_2026-08-15/device/bodies/bodies.json). Nothing launches.
"""

from __future__ import annotations

import builtins
import importlib
import pathlib
import sys

import numpy

from meep_gpu import stepping
from meep_gpu.dispersion import PolarizationState, Susceptibility
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid, Mirror
from meep_gpu.pml import PML
from meep_gpu.triton_kernels import folded_dispersive_update_e as module
from meep_gpu.triton_kernels import symmetry

MODULE_NAME = "meep_gpu.triton_kernels.folded_dispersive_update_e"
PACKAGE_DIR = pathlib.Path(symmetry.__file__).parent
MODULE_PATH = PACKAGE_DIR / "folded_dispersive_update_e.py"


def reasons_without_backend(verdict):
    return [r for r in verdict.reasons if "cupy" not in r]


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
    reloaded = importlib.import_module(MODULE_NAME)
    fields, pml = folded_dispersive()
    assert isinstance(
        reloaded.folded_dispersive_constitutive_coverage(fields, pml).covered, bool)


def test_the_shared_clause_builders_it_composes_from_still_exist():
    from meep_gpu.triton_kernels import coverage as coverage_module
    for name in module.SHARED_CLAUSES:
        assert hasattr(coverage_module, name), (
            f"coverage.{name} was renamed; this module's restatement silently "
            f"drops a clause without this assertion")
    for name in ("_folded_constitutive_grid_reasons", "_has_real_fold"):
        assert hasattr(symmetry, name), name


def test_the_module_claims_no_fingerprint_entry_and_says_it_is_an_arm():
    """The module shipped unwired and now is an arm, so the claim it carries flips.

    It used to have to say "AWAITING WIRING" and to mention no ``fastpath``,
    because both were true: the closure round did not own ``launch.py`` and could
    add no arm. Now ``launch.plan_step`` carries a ``folded dispersive`` arm and
    ``launch.FAMILY_MODULES`` names this module, so the sentence a reader needs is
    the one that says WHERE the arm is and what still holds DISPATCH shut — and
    the old header would be a false statement about a file this module does not
    own. The planner-side half of this is
    ``test_a_wired_family_no_longer_claims_the_planner_ignores_it``.
    """
    import json
    record = json.loads((PACKAGE_DIR / "fingerprints.json").read_text(encoding="utf-8"))
    assert "folded_dispersive_update_e.py" not in record["host_sha256"]
    text = MODULE_PATH.read_text(encoding="utf-8")
    assert "AWAITING WIRING" not in text
    assert "plan_step" in text, "the module must say where its arm lives"
    assert "cuda_kernels" not in text
    assert "DISPATCH_BY_DEFAULT" in text, (
        "a wired family must name what holds dispatch shut, not claim it is "
        "unreachable")
    from meep_gpu.triton_kernels import launch
    assert "folded_dispersive_update_e" in launch.FAMILY_MODULES


# ---------------------------------------------------------------------------
# The fixture, and the reference against stepping.update_E
# ---------------------------------------------------------------------------
#
# Closes residual group (C): 4 slots on four TestLoadDump 2-D rows, each of which
# reaches FULL whole-step coverage once the arm is wired. The device leg measured
# the certified dispersive body against stepping.update_E on a synthetic 3-D
# fixture at 0 of 69120 words differing over EIGHT complete cycles, and on a
# ROW-SHAPED 2-D grid at 0 of 9360. `moved` is the per-sub-step minimum over
# cycles, and the +-0 lattice is held live so a no-op cannot pass as identity.

#: results/residual_closure_2026-08-15/device/bodies/bodies.json
_GROUP_C_MEASURED = {"C_folded_dispersive_update_E": (0, 69120, 13824),
                     "C_folded_dispersive_update_E_ROWSHAPE_2D": (0, 9360, 1872)}


def folded_dispersive(counts=(1, 1, 1), kind="lorentzian", rows=None,
                      complex_storage=False, seed=17, boundaries="periodic",
                      **grid_kwargs):
    """A FOLDED 2-D Grid/Fields/PML triple carrying ``counts[c]`` poles on component c.

    Built the way the engine builds one: a pole with a sigma of exactly zero on a
    component does not drive it (``sigma_is_trivial``, dispersion.py:600), so a
    ``(1, 0, 2)`` configuration is an ordinary anisotropic material rather than a
    harness contrivance.
    """
    grid = Grid(resolution=10.0, cell_size=(1.6, 2.0, 0.0), dimensions=2,
                courant=0.35, boundaries=boundaries,
                symmetry=(Mirror("Y", 1),), xp=numpy, **grid_kwargs)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.enable_pml_storage()
    fields.enable_field_storage()
    shape = grid.shape
    eps = numpy.full(shape, 2.25, dtype=numpy.float32)
    inverse = (1.0 / eps).astype(numpy.float32)
    components = ("Ex", "Ey", "Ez")
    fields.set_epsilon_volumes({c: eps for c in components},
                               {c: inverse for c in components}, rows)
    for index in range(max(counts) if counts else 0):
        term = Susceptibility(frequency=1.0, gamma=0.1, kind=kind)
        sigmas = {name: (0.3 + 0.05 * index) if counts[axis] > index else 0.0
                  for axis, name in enumerate(components)}
        fields.polarizations.append(
            PolarizationState(term, sigmas, grid, numpy.float32))
    thickness = []
    for axis in range(3):
        if shape[axis] < 6:
            thickness.append((0, 0))
        elif axis == 1:
            thickness.append((0, 2))   # no LOW-face layer on the folded axis
        else:
            thickness.append((2, 2))
    rng = numpy.random.default_rng(seed)
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
                 "Hx", "Hy", "Hz", "f_w_Ex", "f_w_Ey", "f_w_Ez",
                 "f_w_Hx", "f_w_Hy", "f_w_Hz"):
        array = getattr(fields, name, None)
        if array is not None:
            array[...] = rng.uniform(-0.4, 0.4, size=shape).astype(array.dtype)
    for state in fields.polarizations:
        for component in components:
            for slot in ("P", "P_prev"):
                buffer = getattr(state, slot, {}).get(component)
                if buffer is not None:
                    buffer[...] = rng.uniform(
                        -0.2, 0.2, size=shape).astype(buffer.dtype)
    return fields, PML(grid=grid, thickness=tuple(thickness))


def _folded_dispersive_reasons(fields, pml):
    return [r for r in module.folded_dispersive_constitutive_coverage(
        fields, pml).reasons if "cupy" not in r]


def _reference_dispersive_update_E(fields, pml, state):
    """The certified body, transcribed: (D - sum P) * inv_eps, then the dsigw tail.

    ``stepping.update_E`` at S:983-989 with a pole registered — the source is
    ``Fields.displacement_minus_polarization`` (fields.py:1096-1105) and the tail
    is ``_apply_constitutive_pml`` (S:2065-2096), MEEP's ``step_update_EDHB``::

        fwprev = fw[i]; fw[i] = g[i] * u[i];
        f[i] += (kap+sig)*fw[i] - (kap-sig)*fwprev

    Nothing in it reads a neighbour, which is the whole reason a FOLD cannot
    change it: the fold's only contribution to an element-wise sub-step is the
    stored extent, and every array here already has it.
    """
    for component, source_name, axis in stepping.E_CONSTITUTIVE_TERMS:
        source = state[source_name].copy()
        for pole in state["poles"]:
            if component in pole:
                source -= pole[component]
        constitutive = source * state["inv_eps"][component]
        kps = getattr(pml, f"kps_{axis}_h")
        kms = getattr(pml, f"kms_{axis}_h")
        fw = state["f_w_" + component]
        fw_previous = fw.copy()
        fw[...] = constitutive
        state[component] += kps * fw
        state[component] -= kms * fw_previous


def test_the_reference_matches_stepping_on_the_folded_dispersive_configuration():
    """The reference test for group (C), byte-for-byte against ``stepping.update_E``.

    Three complete calls. The assertion is on the raw bytes of every array the
    sub-step writes, never ``allclose``, and the non-vacuity check is that the
    state actually moved: a no-op agreeing with a no-op is trivially identical.
    """
    fields, pml = folded_dispersive()
    state = {name: getattr(fields, name).copy()
             for name in ("Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
                          "f_w_Ex", "f_w_Ey", "f_w_Ez")}
    state["inv_eps"] = {name: fields.inverse_epsilon_for(name)
                        for name in ("Ex", "Ey", "Ez")}
    before = {name: state[name].copy() for name in ("Ex", "Ey", "Ez")}
    for call in range(3):
        state["poles"] = [{c: pole.P[c] for c in ("Ex", "Ey", "Ez")
                           if pole.drives(c)} for pole in fields.polarizations]
        state["poles"] = [{c: array.copy() for c, array in pole.items()}
                          for pole in state["poles"]]
        stepping.update_E(fields, pml)
        _reference_dispersive_update_E(fields, pml, state)
        for name in ("Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez"):
            assert state[name].tobytes() == getattr(fields, name).tobytes(), (
                f"{name} diverged from stepping.update_E at call {call + 1}: "
                f"max abs delta "
                f"{numpy.max(numpy.abs(state[name] - getattr(fields, name))):.3e}")
    moved = sum(int(numpy.count_nonzero(
        numpy.frombuffer(before[name].tobytes(), dtype=numpy.uint32)
        != numpy.frombuffer(getattr(fields, name).tobytes(), dtype=numpy.uint32)))
        for name in ("Ex", "Ey", "Ez"))
    assert moved > 0, "the sub-step moved no words: the identity is vacuous"


def test_the_pole_sum_is_load_bearing_in_that_reference():
    """Drop the ``- sum P`` and the reference must FAIL — otherwise it proves nothing."""
    fields, pml = folded_dispersive()
    state = {name: getattr(fields, name).copy()
             for name in ("Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
                          "f_w_Ex", "f_w_Ey", "f_w_Ez")}
    state["inv_eps"] = {name: fields.inverse_epsilon_for(name)
                        for name in ("Ex", "Ey", "Ez")}
    state["poles"] = []          # the mutation: no polarization subtracted
    stepping.update_E(fields, pml)
    _reference_dispersive_update_E(fields, pml, state)
    assert state["Ex"].tobytes() != fields.Ex.tobytes(), (
        "removing the pole sum changed nothing: the fixture registers no live "
        "polarization and the reference test above is vacuous")


def test_the_new_predicate_admits_what_both_incumbents_refuse():
    from meep_gpu.triton_kernels import dispersive_update_e as dispersive
    fields, pml = folded_dispersive()
    assert _folded_dispersive_reasons(fields, pml) == []
    assert any("a susceptibility is registered" in r for r in
               symmetry.folded_constitutive_coverage(fields, pml, "E").reasons)
    assert any("folded by a mirror plane" in r for r in
               dispersive.dispersive_constitutive_coverage(fields, pml).reasons)


def test_no_pole_is_refused_toward_the_folded_constitutive_predicate():
    fields, pml = folded_dispersive(counts=(0, 0, 0))
    assert not fields.polarizations
    reasons = _folded_dispersive_reasons(fields, pml)
    assert any("no susceptibility is registered" in r for r in reasons), reasons
    assert reasons_without_backend(
        symmetry.folded_constitutive_coverage(fields, pml, "E")) == []


def test_an_unfolded_grid_is_refused_toward_the_dispersive_predicate():
    from meep_gpu.triton_kernels import dispersive_update_e as dispersive
    fields, pml, _ = _unfolded_dispersive()
    reasons = _folded_dispersive_reasons(fields, pml)
    assert any("no mirror plane is active" in r for r in reasons), reasons
    assert [r for r in dispersive.dispersive_constitutive_coverage(
        fields, pml).reasons if "cupy" not in r] == []


def _unfolded_dispersive():
    grid = Grid(resolution=10.0, cell_size=(0.8, 0.8, 0.8), dimensions=3,
                courant=0.35, xp=numpy)
    fields = Fields(grid=grid)
    fields.enable_pml_storage()
    fields.enable_field_storage()
    term = Susceptibility(frequency=1.0, gamma=0.1, kind="lorentzian")
    state = PolarizationState(term, {"Ex": 0.3, "Ey": 0.3, "Ez": 0.3},
                              grid, numpy.float32)
    fields.polarizations.append(state)
    return fields, PML(grid=grid, thickness=2), state


def test_an_offdiagonal_row_is_refused_with_the_divergence_it_measured():
    probe, _ = folded_dispersive()          # only for its grid shape
    rows = {"Ex": {"Ey": numpy.full(probe.grid.shape, 0.03, dtype=numpy.float32)}}
    fields, pml = folded_dispersive(rows=rows)
    assert fields.has_offdiagonal_epsilon
    reasons = _folded_dispersive_reasons(fields, pml)
    joined = " ".join(reasons)
    assert "off-diagonal chi1inv row is installed" in joined, reasons
    assert "13824/69120" in joined, "the refusal must carry its measurement"


def test_chi2_chi3_complex_storage_beta_and_bloch_stay_refused():
    fields, pml = folded_dispersive()
    fields.set_nonlinear_volumes({"Ez": 0.0}, {"Ez": 0.05})
    assert any("chi2/chi3" in r for r in _folded_dispersive_reasons(fields, pml))

    fields, pml = folded_dispersive(complex_storage=True)
    assert any("force_complex_fields" in r
               for r in _folded_dispersive_reasons(fields, pml))

    fields, pml = folded_dispersive(k_point=(0.1, 0.0, 0.0))
    assert any("k_point" in r for r in _folded_dispersive_reasons(fields, pml))

    fields, pml = folded_dispersive(beta=0.25)
    assert any("beta" in r for r in _folded_dispersive_reasons(fields, pml))


def test_an_inactive_layer_is_refused():
    fields, pml = folded_dispersive()
    inert = PML(grid=fields.grid, thickness=0)
    assert any("no active PML" in r for r in _folded_dispersive_reasons(fields, inert))


def test_too_many_poles_on_one_component_is_refused_by_name():
    from meep_gpu.triton_kernels.dispersive_update_e import MAX_POLES
    fields, pml = folded_dispersive(counts=(1, 1, MAX_POLES + 1))
    reasons = _folded_dispersive_reasons(fields, pml)
    assert any(f"MAX_POLES={MAX_POLES}" in r for r in reasons), reasons


def test_a_malformed_pole_volume_is_refused_by_name():
    fields, pml = folded_dispersive()
    state = fields.polarizations[0]
    state.P["Ez"] = state.P["Ez"][::-1]          # right shape, right dtype, reversed
    reasons = _folded_dispersive_reasons(fields, pml)
    assert any("P[Ez]" in r and "C-contiguous" in r for r in reasons), reasons


def test_an_unreadable_susceptibility_is_refused():
    from types import SimpleNamespace as _NS
    fields, pml = folded_dispersive()
    fields.polarizations.append(_NS(susceptibility=_NS(kind="lorentzian"),
                                    drives=lambda name: False))
    reasons = _folded_dispersive_reasons(fields, pml)
    assert any("does not report driven()" in r for r in reasons), reasons


def test_a_magnetic_susceptibility_is_refused_by_the_shared_clause():
    from types import SimpleNamespace as _NS
    fields, pml = folded_dispersive()
    fields.polarizations.append(_NS(susceptibility=_NS(kind="lorentzian"),
                                    driven=lambda: ("Hx",),
                                    drives=lambda name: name == "Hx"))
    reasons = _folded_dispersive_reasons(fields, pml)
    assert any("Hx" in r for r in reasons), reasons


def test_a_missing_volume_is_refused_by_name():
    fields, pml = folded_dispersive()
    fields.f_w_Ey = None
    assert any("f_w_Ey is not allocated" in r
               for r in _folded_dispersive_reasons(fields, pml))


def test_mutation_dropping_the_pole_clause_overlaps_the_folded_predicate(monkeypatch):
    real = module.folded_dispersive_constitutive_coverage

    def mutated(fields, pml):
        verdict = real(fields, pml)
        kept = tuple(r for r in verdict.reasons
                     if "no susceptibility is registered" not in r)
        return type(verdict)(not kept, kept)

    monkeypatch.setattr(module, "folded_dispersive_constitutive_coverage", mutated)
    fields, pml = folded_dispersive(counts=(0, 0, 0))
    assert _folded_dispersive_reasons(fields, pml) == [], (
        "the mutation is DISARMED: some other clause was keeping the pole-free "
        "run out")
    assert reasons_without_backend(
        symmetry.folded_constitutive_coverage(fields, pml, "E")) == [], (
        "and the incumbent admits it too — the overlap the clause prevents")


def test_mutation_dropping_the_fold_clause_overlaps_the_dispersive_predicate(monkeypatch):
    from meep_gpu.triton_kernels import dispersive_update_e as dispersive
    real = module.folded_dispersive_constitutive_coverage

    def mutated(fields, pml):
        verdict = real(fields, pml)
        kept = tuple(r for r in verdict.reasons
                     if "no mirror plane is active" not in r)
        return type(verdict)(not kept, kept)

    monkeypatch.setattr(module, "folded_dispersive_constitutive_coverage", mutated)
    fields, pml, _ = _unfolded_dispersive()
    assert _folded_dispersive_reasons(fields, pml) == [], (
        "the mutation is DISARMED: the fold clause was not what kept the "
        "unfolded run out")
    assert [r for r in dispersive.dispersive_constitutive_coverage(
        fields, pml).reasons if "cupy" not in r] == []


def test_the_builder_answers_none_on_this_laptop():
    fields, pml = folded_dispersive()
    assert module.plan_folded_dispersive_constitutive(fields, pml) is None


def test_the_module_records_the_measurement_and_names_its_arm():
    text = MODULE_PATH.read_text(encoding="utf-8")
    for case, (differing, compared, moved) in _GROUP_C_MEASURED.items():
        assert case in text, f"{case} is not recorded in the module"
        assert f"{differing} / {compared}" in text
        assert str(moved) in text
    assert "AWAITING WIRING" not in text
    assert "folded dispersive" in text, (
        "the module must name the arm label the planner selects it under")


def test_the_admission_itself_was_measured_not_only_the_kernel_body():
    """``MEASURED['own_builder']`` is the leg through THIS module, unpatched.

    The block above it reaches ``DispersiveConstitutivePlan`` by patching
    ``dispersive_constitutive_coverage``, which licenses the kernel body and says
    nothing about :func:`folded_dispersive_constitutive_coverage` or
    :func:`plan_folded_dispersive_constitutive`.
    """
    text = MODULE_PATH.read_text(encoding="utf-8")
    own = module.MEASURED["own_builder"]
    assert own["artifact"] in text
    assert own["cases"], "no leg through this module's own builder is recorded"
    for case, record in own["cases"].items():
        assert record["differing"] == 0
        assert f"{record['differing']} / {record['compared']}" in text
    assert "UNPATCHED" in text.upper()


def test_the_fixture_is_not_claimed_to_be_a_corpus_row_and_a_rowshape_was_run():
    """The 3-D fixture is synthetic; group (C)'s four rows are 2-D.

    The module used to say the body was measured "on the corpus configuration".
    It was not — and the fix is the row-shaped leg, not a hedge.
    """
    text = MODULE_PATH.read_text(encoding="utf-8")
    rowshape = module.MEASURED["rowshape"]
    assert rowshape["case"] in text
    assert f"{rowshape['differing']} / {rowshape['compared']}" in text
    assert module.MEASURED["fixture_shape"] == [9, 16, 16]
    assert rowshape["fixture_shape"][2] == 1, "the row-shaped leg must be 2-D"
    assert rowshape["boundary_kinds"] == ["metallic", "mirror", "periodic"]
    assert "NOT \"the corpus configuration\"" in text or "NOT the corpus" in text
