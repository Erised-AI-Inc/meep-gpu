"""Laptop contracts for the chi2/chi3 nonlinear update_E Triton tranche.

Everything here runs on the NumPy merge-bar machine: the optional-import
contract, the predicate's admission AND per-clause refusals on real
Grid/Fields/PML objects, the plan builders' shapes, the host scalar arms pinned
BYTE-for-byte against NumPy's own weak-scalar semantics, and an in-test
transcription of the nonlinear sub-step pinned against ``stepping.update_E``
itself. The kernel's device bytes are the gate's
(``parity/meep_gpu/gate_triton_nonlinear.py``); nothing here launches.
"""

from __future__ import annotations

import importlib
import pathlib
import sys
from types import SimpleNamespace

import numpy
import pytest

from meep_gpu import stepping
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid
from meep_gpu.pml import PML
from meep_gpu.test_triton_kernels import GUARD_SPELLING, PACKAGE_DIR, code_of

MODULE_PATH = PACKAGE_DIR / "nonlinear_update_e.py"
MODULE_NAME = "meep_gpu.triton_kernels.nonlinear_update_e"
PARITY_DIR = pathlib.Path(__file__).resolve().parents[1] / "parity" / "meep_gpu"
GATE_PATH = PARITY_DIR / "gate_triton_nonlinear.py"
COMPOSITION_PATH = PARITY_DIR / "probe_triton_nonlinear_composition.py"
SLURM_PATH = PARITY_DIR / "run_triton_nonlinear.slurm"

#: 3rd-harm-1d.py sweeps its chi3 (the script's ``k``) logarithmically around
#: this decade; the tests exercise a value of the corpus's own family.
CHI3_CORPUS = 0.06
CHI2_TEST = 0.045
#: A scalar inverse epsilon whose double-rounded square AND cube both differ
#: from the once-rounded ones (verified by the arm test below).
US_DISCRIMINATOR = 0.37


@pytest.fixture(scope="module")
def nl():
    """The shipped module itself must be importable without a test-only stub."""
    return importlib.import_module(MODULE_NAME)


def _build(cell_size=(0.8, 0.8, 0.8), dimensions=3, boundaries=None,
           pml_thickness=2, complex_storage=False, storage=True,
           k_point=(0.0, 0.0, 0.0), beta=0.0, courant=0.35,
           chi2=None, chi3=None, eps=2.25, seed=11, thickness=None,
           **grid_kwargs):
    """A real nonlinear Grid/Fields/PML triple on NumPy.

    Default: the family's target class — real storage, an active PML, a scalar
    chi3 on every component, k = 0, no fold. Every refusal test perturbs
    exactly one clause off this.
    """
    grid = Grid(resolution=10.0, cell_size=cell_size, dimensions=dimensions,
                boundaries=boundaries, courant=courant, k_point=k_point,
                beta=beta, **grid_kwargs)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.set_background_eps(eps)
    if chi2 is None and chi3 is None:
        chi2 = {"Ex": 0.0, "Ey": 0.0, "Ez": 0.0}
        chi3 = {"Ex": CHI3_CORPUS, "Ey": CHI3_CORPUS, "Ez": CHI3_CORPUS}
    fields.set_nonlinear_volumes(chi2 or {}, chi3 or {})
    if storage:
        fields.enable_pml_storage()
        rng = numpy.random.default_rng(seed)
        for name in ("Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
                     "f_w_Ex", "f_w_Ey", "f_w_Ez"):
            getattr(fields, name)[...] = rng.uniform(
                -0.4, 0.4, size=grid.shape).astype(numpy.float32)
    if thickness is None:
        thickness = tuple(
            (pml_thickness, pml_thickness) if grid.shape[axis] >= 6 else (0, 0)
            for axis in range(3))
    return fields, PML(grid=grid, thickness=thickness)


def _reasons(nl, fields, pml):
    verdict = nl.nonlinear_constitutive_coverage(fields, pml)
    return [r for r in verdict.reasons if "array module" not in r]


# ---------------------------------------------------------------------------
# The optional dependency must stay optional; the restatements must not drift
# ---------------------------------------------------------------------------

def test_the_package_does_not_import_the_module_eagerly():
    for name in list(sys.modules):
        if name == MODULE_NAME:
            del sys.modules[name]
    importlib.import_module("meep_gpu.triton_kernels")
    eagerly_imported = MODULE_NAME in sys.modules
    importlib.import_module(MODULE_NAME)  # restore for the other tests
    assert not eagerly_imported, (
        "the package __init__ must not pull this unwired module in")


def test_the_module_answers_coverage_but_the_kernel_fails_clearly_without_triton(nl):
    if nl.nonlinear_constitutive_step is not None:
        pytest.skip("triton is installed here; the absence arm is exercised on "
                    "the merge-bar box without it")
    fields, pml = _build()
    assert nl.nonlinear_constitutive_coverage(fields, pml).reasons
    with pytest.raises(ImportError, match="triton"):
        nl.nonlinear_constitutive_step_kernel()


def test_the_restated_constants_pin_their_originators(nl):
    def value(x):
        return getattr(x, "value", x)

    assert value(nl.PERIODIC) == 0 and value(nl.METALLIC) == 1
    assert nl.DEFAULT_BLOCK == 256
    kernels_source = (PACKAGE_DIR / "kernels.py").read_text(encoding="utf-8")
    assert "\nPERIODIC = tl.constexpr(0)\n" in kernels_source
    assert "\nMETALLIC = tl.constexpr(1)\n" in kernels_source
    assert "\nDEFAULT_BLOCK = 256\n" in kernels_source
    assert nl.BOUNDARY_CODES == {"periodic": 0, "metallic": 1}


def test_the_term_table_is_steppings_own(nl):
    """E_TERMS restates stepping.E_CONSTITUTIVE_TERMS with the axis as an index."""
    axis_of = {"x": 0, "y": 1, "z": 2}
    assert nl.E_TERMS == tuple(
        (target, source, axis_of[axis_name])
        for target, source, axis_name in stepping.E_CONSTITUTIVE_TERMS)
    assert nl.HALF_INTEGER is True  # stepping.py:1015, half_integer=True


def test_the_transverse_partner_table_is_the_cycle_direction_rule(nl):
    """cycle_direction X -> Y -> Z: own + 1 first, own + 2 second
    (stepping.py:1183-1185) — Ez takes Dx then Dy."""
    assert nl.TRANSVERSE_PARTNERS == tuple(
        ((own + 1) % 3, (own + 2) % 3) for own in range(3))


def test_the_shared_clause_builders_this_predicate_composes_from_still_exist(nl):
    from meep_gpu.triton_kernels import coverage

    for name in nl.SHARED_CLAUSES:
        assert callable(getattr(coverage, name)), (
            f"coverage.{name} was renamed or removed; the predicate composes "
            f"from it and would silently drop a clause")


#: sha256 of ``inspect.getsource(coverage._grid_reasons)`` at the last audit
#: of the restatement. Re-pin ONLY after carrying any clause change into
#: ``_nonlinear_grid_reasons``.
_GRID_REASONS_SOURCE_SHA256 = (
    "14387b8c1c6d6f6538c1c4018872e7cdaa877dee7bc5bade3f88fc3e67687341")


def test_the_restated_grid_clause_set_tracks_the_shared_source(nl):
    """``_nonlinear_grid_reasons`` RESTATES ``coverage._grid_reasons``' clause
    set (chi clause inverted, the special_kz precedent) rather than calling
    it. Nothing behavioral can pin the restated SET on this laptop — a clause
    the shared file gains would only surface when a probe case happens to
    carry the new feature — so this deliberate change-detector is the drift
    tripwire the audit asked for: when it fires, diff the shared builder
    against the restatement, carry the clause change across, and re-pin."""
    import hashlib
    import inspect

    from meep_gpu.triton_kernels import coverage

    digest = hashlib.sha256(
        inspect.getsource(coverage._grid_reasons).encode("utf-8")).hexdigest()
    assert digest == _GRID_REASONS_SOURCE_SHA256, (
        "coverage._grid_reasons changed underfoot. Re-audit "
        "_nonlinear_grid_reasons (nonlinear_update_e.py) clause by clause "
        "against the shared builder, carry any added/changed clause into the "
        "restatement, then re-pin _GRID_REASONS_SOURCE_SHA256 to "
        f"{digest!r}")


def test_the_module_docstring_states_dispatch_is_untouched(nl):
    assert "NOT WIRED" in nl.__doc__
    assert "plan_fast_path" in nl.__doc__


def test_the_module_does_not_reach_into_another_tracks_file():
    code = code_of(MODULE_PATH)
    assert "fastpath" not in code
    assert "cuda_kernels" not in code


def test_the_guard_is_the_packages_one_spelling_and_appears_once():
    code = MODULE_PATH.read_text(encoding="utf-8")
    assert code.count("enable_fp_fusion=") == 1  # the one plan run() method
    assert code.count(GUARD_SPELLING) == 1


def test_no_fingerprint_entry_is_claimed():
    """No GATE entry for this tranche in the table's ledger — the boundary.

    It used to forbid the STRING anywhere in the file, which is a stricter
    spelling than the sibling tranches use (``test_triton_bfast`` and
    ``test_triton_special_kz`` both scan the top-level KEYS), and stricter than the
    boundary itself: what must not happen is this tranche claiming a gate record it
    did not cut. The shared ``family_recert_2026-08-14`` transcription is the one
    place the module may be named, and it exists because the alternative was
    measured to be worse — the dispatcher pointed at this family's provenance by a
    path into a gitignored results directory, so every run artifact that dispatched
    it named a family, a dead path and nothing else. The digest under that key is
    additionally what makes a drift in this module VISIBLE: no ``host_sha256``
    entry covers it, and this one has already drifted from its certified bytes.
    """
    import json

    recorded = json.loads((PACKAGE_DIR / "fingerprints.json")
                          .read_text(encoding="utf-8"))
    assert not any("nonlinear_update_e" in name for name in recorded), (
        "this tranche's provenance lives in the gate's results directory, "
        "not in a gate entry of the table's fingerprints.json")
    shared = json.dumps({key: value for key, value in recorded.items()
                         if key != "family_recert_2026-08-14"})
    assert "nonlinear_update_e" not in shared, (
        "outside the shared family-recert transcription, nothing in this file may "
        "name this tranche's module")


def test_the_gate_and_its_composition_probe_exist_beside_the_other_tranches():
    assert GATE_PATH.exists()
    assert COMPOSITION_PATH.exists()


# ---------------------------------------------------------------------------
# The positive verdict — refused only for the backend on this laptop
# ---------------------------------------------------------------------------

def test_the_corpus_configuration_is_refused_only_for_the_backend(nl):
    """3rd-harm-1d's class: scalar chi3, PML, metallic walls, k = 0."""
    fields, pml = _build(boundaries="metallic")
    assert _reasons(nl, fields, pml) == []


def test_the_periodic_and_mixed_boundary_classes_are_admitted(nl):
    fields, pml = _build()  # all-periodic
    assert _reasons(nl, fields, pml) == []
    fields, pml = _build(boundaries=("periodic", "metallic", "periodic"))
    assert _reasons(nl, fields, pml) == []


def test_a_partial_nonlinearity_is_admitted(nl):
    """chi on Ez only: Ex/Ey take the compiled plain arm (docstring point 5)."""
    fields, pml = _build(chi2={"Ez": CHI2_TEST}, chi3={"Ez": CHI3_CORPUS})
    assert fields.nonlinear_components == ("Ez",)
    assert _reasons(nl, fields, pml) == []


def test_volume_chi_forms_are_admitted(nl):
    grid_shape = Grid(resolution=10.0, cell_size=(0.8, 0.8, 0.8),
                      dimensions=3, courant=0.35).shape
    volume = numpy.full(grid_shape, CHI3_CORPUS, dtype=numpy.float32)
    volume[0] = 0.1
    fields, pml = _build(chi2={"Ez": 0.0}, chi3={"Ez": volume})
    assert _reasons(nl, fields, pml) == []


# ---------------------------------------------------------------------------
# The refusals, one per silent-wrong-answer surface
# ---------------------------------------------------------------------------

def test_zero_chi_is_refused_and_the_two_predicates_are_disjoint(nl):
    """The INVERTED clause: a trivial pair never installs (fields.py:853-857),
    the run is the linear engine's bit for bit, and it belongs to
    constitutive_coverage(side='E') — never to both predicates at once."""
    from meep_gpu.triton_kernels import coverage

    fields, pml = _build(chi2={"Ez": 0.0}, chi3={"Ez": 0.0})
    assert not fields.has_nonlinearity
    reasons = _reasons(nl, fields, pml)
    assert any("no chi2/chi3" in r for r in reasons), reasons
    plain = [r for r in coverage.constitutive_coverage(fields, pml, "E").reasons
             if "array module" not in r]
    assert plain == [], plain  # the plain kernel's side of the split

    nonlinear_fields, nonlinear_pml = _build()
    assert _reasons(nl, nonlinear_fields, nonlinear_pml) == []
    plain_on_nonlinear = coverage.constitutive_coverage(
        nonlinear_fields, nonlinear_pml, "E")
    assert any("chi2/chi3" in r for r in plain_on_nonlinear.reasons), (
        "the shipped predicate stopped refusing chi2/chi3 — the two update_E "
        "predicates now overlap and plan_step would pick by ordering")


def test_no_active_layer_is_refused_with_the_different_sub_step_reason(nl):
    fields, pml = _build(pml_thickness=0)
    reasons = _reasons(nl, fields, pml)
    assert any("990-993" in r or "different sub-step" in r for r in reasons), reasons


def test_complex_storage_is_refused_as_the_phase_b_leg(nl):
    fields, pml = _build(complex_storage=True)
    reasons = _reasons(nl, fields, pml)
    assert any("DOCMP" in r or "force_complex_fields" in r for r in reasons), reasons


def test_a_mirror_plane_is_refused(nl):
    # chi3-only: chi2 respects any plane (fields.py:887-927), so the build
    # succeeds and the PREDICATE is what refuses. The folded axis's low face
    # is the plane itself, so the layer sits on its high face only.
    fields, pml = _build(symmetry=("X",),
                         thickness=((0, 2), (2, 2), (2, 2)))
    reasons = _reasons(nl, fields, pml)
    assert any("mirror" in r or "folded" in r for r in reasons), reasons


def test_a_nonzero_k_point_is_refused(nl):
    fields, pml = _build(k_point=(0.3, 0.0, 0.0))
    reasons = _reasons(nl, fields, pml)
    assert any("k_point" in r for r in reasons), reasons


def test_beta_is_refused(nl):
    fields, pml = _build(cell_size=(0.8, 0.8, 0.0), dimensions=2, beta=0.25)
    reasons = _reasons(nl, fields, pml)
    assert any("beta" in r for r in reasons), reasons


def test_bfast_is_refused(nl):
    fields, pml = _build(bfast_scaled_k=(0.5, 0.0, 0.0))
    reasons = _reasons(nl, fields, pml)
    assert any("BFAST" in r for r in reasons), reasons


def test_a_registered_polarization_is_refused_by_name(nl):
    fields, pml = _build()
    fields.polarizations.append(SimpleNamespace(
        driven=lambda: ("Ez",), drives=lambda name: name == "Ez"))
    reasons = _reasons(nl, fields, pml)
    assert any("susceptibility is registered" in r for r in reasons), reasons


def test_offdiagonal_epsilon_is_refused_toward_the_offdiag_family(nl):
    fields, pml = _build()
    coupling = numpy.full(fields.grid.shape, 0.05, dtype=numpy.float32)
    fields.set_epsilon_volumes(
        {c: fields.epsilon_for(c) for c in ("Ex", "Ey", "Ez")},
        {c: fields.inverse_epsilon_for(c) for c in ("Ex", "Ey", "Ez")},
        chi1inv_offdiagonal={"Ex": {"Ey": coupling}})
    reasons = _reasons(nl, fields, pml)
    assert any("off-diagonal" in r for r in reasons), reasons


def test_missing_storage_is_refused(nl):
    fields, pml = _build(storage=False, pml_thickness=0)
    reasons = _reasons(nl, fields, pml)
    assert any("is not allocated" in r for r in reasons), reasons
    assert any("recomputed from D" in r for r in reasons), reasons


def test_a_malformed_chi_volume_is_refused_by_name(nl):
    fields, pml = _build()
    fields._chi3_components["Ez"] = numpy.full(
        fields.grid.shape, 0.1, dtype=numpy.float64)  # wrong dtype, planted
    reasons = _reasons(nl, fields, pml)
    assert any("chi3[Ez]" in r and "float32" in r for r in reasons), reasons


def test_a_scalar_inverse_epsilon_is_refused_on_the_engine_route(nl):
    """The kernel CARRIES the scalar arm (the harness route exercises it); the
    engine route restates the certified predicate's volume-only clause."""
    fields, pml = _build()
    fields._inv_eps_components["Ez"] = 0.5
    reasons = _reasons(nl, fields, pml)
    assert any("scalar, not a volume" in r for r in reasons), reasons


# ---------------------------------------------------------------------------
# Predicate mutations: each added clause is load-bearing
# ---------------------------------------------------------------------------

def _mutated_predicate(nl, needle: str, replacement: str):
    import inspect
    import textwrap

    source = textwrap.dedent(inspect.getsource(nl.nonlinear_constitutive_coverage))
    assert needle in source, f"needle not found: {needle!r}"
    namespace = dict(vars(nl))
    exec(compile(source.replace(needle, replacement), "<mutated>", "exec"),
         namespace)
    return namespace["nonlinear_constitutive_coverage"]


def test_mutation_dropping_the_polarization_clause_admits_the_ade_configuration(nl):
    fields, pml = _build()
    fields.polarizations.append(SimpleNamespace(
        driven=lambda: ("Ez",), drives=lambda name: name == "Ez"))
    mutated = _mutated_predicate(
        nl, "if states or getattr(fields, \"has_polarizations\", False):",
        "if False:")
    kept = [r for r in mutated(fields, pml).reasons if "array module" not in r]
    assert kept == [], (
        "with the polarization clause dropped nothing else refuses this run — "
        "so the clause is load-bearing and this mutation would silently cover "
        "the dispersive+nonlinear configuration")


def test_mutation_dropping_the_offdiag_clause_admits_the_row_product(nl):
    fields, pml = _build()
    coupling = numpy.full(fields.grid.shape, 0.05, dtype=numpy.float32)
    fields.set_epsilon_volumes(
        {c: fields.epsilon_for(c) for c in ("Ex", "Ey", "Ez")},
        {c: fields.inverse_epsilon_for(c) for c in ("Ex", "Ey", "Ez")},
        chi1inv_offdiagonal={"Ex": {"Ey": coupling}})
    mutated = _mutated_predicate(
        nl, "if getattr(fields, \"has_offdiagonal_epsilon\", False):", "if False:")
    kept = [r for r in mutated(fields, pml).reasons if "array module" not in r]
    assert kept == []


def test_mutation_dropping_the_grid_inverted_chi_clause_breaks_disjointness(nl):
    """The GRID-level clause. Its run-level twin is mutated further down.

    Both were spelled ``test_mutation_dropping_the_inverted_chi_clause_breaks_
    disjointness``; pytest binds a module's tests by name, so this one was
    collected under the other's body and never ran. They pin different clauses —
    this one mutates ``_nonlinear_grid_reasons``, the other
    ``_nonlinear_run_grid_reasons`` — so the collision dropped a real assertion.
    """
    fields, pml = _build(chi2={"Ez": 0.0}, chi3={"Ez": 0.0})
    source_needle = "if not getattr(fields, \"has_nonlinearity\", False):"
    mutated_grid = None
    import inspect
    import textwrap
    source = textwrap.dedent(inspect.getsource(nl._nonlinear_grid_reasons))
    assert source_needle in source
    namespace = dict(vars(nl))
    exec(compile(source.replace(source_needle, "if False:"), "<mutated>", "exec"),
         namespace)
    mutated_grid = namespace["_nonlinear_grid_reasons"]
    kept = [r for r in mutated_grid(fields, pml, fields.grid)
            if "array module" not in r]
    assert kept == [], (
        "with the inverted chi clause dropped the zero-chi run is admitted by "
        "the grid clauses — both update_E predicates would then cover it")


# ---------------------------------------------------------------------------
# Host scalar arms: NumPy's weak-scalar semantics, pinned to the byte
# ---------------------------------------------------------------------------

def test_the_scalar_epsilon_arm_is_the_double_power_rounded_once(nl):
    us = US_DISCRIMINATOR
    arm = nl.scalar_inverse_epsilon_arm(us)
    assert numpy.float32(arm[0]).tobytes() == numpy.float32(us).tobytes()
    assert numpy.float32(arm[1]).tobytes() == numpy.float32(us * us).tobytes()
    assert numpy.float32(arm[2]).tobytes() == numpy.float32((us * us) * us).tobytes()
    # The double-rounded forms are DIFFERENT words for this value — the arm
    # exists precisely because the kernel must not rebuild the powers in f32.
    doubly_sq = numpy.float32(numpy.float32(us) * numpy.float32(us))
    doubly_cu = numpy.float32(doubly_sq * numpy.float32(us))
    assert numpy.float32(arm[1]).tobytes() != doubly_sq.tobytes()
    assert numpy.float32(arm[2]).tobytes() != doubly_cu.tobytes()


def test_the_scalar_arm_reproduces_steppings_scalar_bytes(nl):
    """gs * chi3 * (us*us*us) with PYTHON floats, against the arm's scalars."""
    rng = numpy.random.default_rng(3)
    gs = rng.uniform(-1.0, 1.0, size=(5, 4, 3)).astype(numpy.float32)
    dsqr = (gs * gs).astype(numpy.float32)
    us, chi3 = US_DISCRIMINATOR, 0.7
    reference = stepping.calc_nonlinear_u(dsqr, gs, us, 0.0, chi3)
    arm = nl.scalar_inverse_epsilon_arm(us)
    c2 = (gs * numpy.float32(nl.scalar_chi_arm(0.0))) * numpy.float32(arm[1])
    c3 = (dsqr * numpy.float32(nl.scalar_chi_arm(chi3))) * numpy.float32(arm[2])
    ours = ((numpy.float32(1.0) + c2) + numpy.float32(2.0) * c3) / (
        (numpy.float32(1.0) + numpy.float32(2.0) * c2) + numpy.float32(3.0) * c3)
    assert reference.dtype == numpy.float32
    assert ours.tobytes() == reference.tobytes()


def test_calc_nonlinear_u_volume_groupings_are_the_transcribed_ones(nl):
    rng = numpy.random.default_rng(5)
    shape = (4, 5, 6)
    gs = rng.uniform(-1.0, 1.0, size=shape).astype(numpy.float32)
    dsqr = rng.uniform(0.0, 1.0, size=shape).astype(numpy.float32)
    us = rng.uniform(0.2, 0.9, size=shape).astype(numpy.float32)
    chi2 = rng.uniform(-0.1, 0.1, size=shape).astype(numpy.float32)
    chi3 = rng.uniform(0.0, 0.2, size=shape).astype(numpy.float32)
    reference = stepping.calc_nonlinear_u(dsqr, gs, us, chi2, chi3)
    c2 = (gs * chi2) * (us * us)
    c3 = (dsqr * chi3) * ((us * us) * us)
    ours = ((numpy.float32(1.0) + c2) + numpy.float32(2.0) * c3) / (
        (numpy.float32(1.0) + numpy.float32(2.0) * c2) + numpy.float32(3.0) * c3)
    assert ours.tobytes() == reference.tobytes()


# ---------------------------------------------------------------------------
# Transcription: the in-test reference against stepping.update_E itself
# ---------------------------------------------------------------------------

def _face(axis, index):
    face = [slice(None)] * 3
    face[axis] = index
    return tuple(face)


def _shift_down(field, axis, boundary):
    """f[i-1]: stepping._shift_down's plain PERIODIC/METALLIC branches."""
    rolled = numpy.roll(field, 1, axis=axis)
    if boundary == "metallic":
        rolled[_face(axis, 0)] = 0
    return rolled


def _shift_up(field, axis, boundary):
    """f[i+1]: stepping._shift_up's plain PERIODIC/METALLIC branches."""
    rolled = numpy.roll(field, -1, axis=axis)
    if boundary == "metallic":
        rolled[_face(axis, -1)] = 0
    return rolled


def _transverse_sums(volumes, own_axis, boundaries):
    """stepping._nonlinear_transverse_sums (:1121-1164): the pair association."""
    names = ("Ex", "Ey", "Ez")
    sums = []
    for offset in (1, 2):
        partner_axis = (own_axis + offset) % 3
        values = volumes[names[partner_axis]]
        pair = values + _shift_down(values, partner_axis, boundaries[partner_axis])
        sums.append(pair + _shift_up(pair, own_axis, boundaries[own_axis]))
    return sums


def _reference_update_e(state, coefficients, chi_pairs, inv_eps, boundaries):
    """The gate reference's twin: the whole nonlinear sub-step on plain dicts."""
    volumes = {"Ex": state["Dx"], "Ey": state["Dy"], "Ez": state["Dz"]}
    for own_axis, (target, source, _axis) in enumerate(
            (("Ex", "Dx", 0), ("Ey", "Dy", 1), ("Ez", "Dz", 2))):
        gs = volumes[target]
        us = inv_eps[target]
        pair = chi_pairs[target]
        if pair is None:
            constitutive = gs * us
        else:
            chi2, chi3 = pair
            first, second = _transverse_sums(volumes, own_axis, boundaries)
            dsqr = gs * gs + 0.0625 * (first * first + second * second)
            row = gs * us
            constitutive = row * stepping.calc_nonlinear_u(dsqr, gs, us, chi2, chi3)
        kps, kms = coefficients[target]
        fw = state["f_w_" + target]
        field = state[target]
        fw_previous = fw.copy()
        fw[...] = constitutive
        field += kps * fw
        field -= kms * fw_previous


def _state_of(fields):
    return {name: getattr(fields, name).copy()
            for name in ("Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
                         "f_w_Ex", "f_w_Ey", "f_w_Ez")}


def _pin_against_stepping(nl, fields, pml, boundaries, steps=3):
    state = _state_of(fields)
    coefficients = {
        target: (getattr(pml, f"kps_{axis}_h"), getattr(pml, f"kms_{axis}_h"))
        for target, _source, axis in stepping.E_CONSTITUTIVE_TERMS}
    chi_pairs = {name: nl.chi_pair_for(fields, name)
                 for name in ("Ex", "Ey", "Ez")}
    inv_eps = {name: fields.inverse_epsilon_for(name)
               for name in ("Ex", "Ey", "Ez")}
    for step in range(steps):
        stepping.update_E(fields, pml)
        _reference_update_e(state, coefficients, chi_pairs, inv_eps, boundaries)
        for name in ("Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez"):
            ours = state[name]
            theirs = getattr(fields, name)
            assert ours.dtype == theirs.dtype == numpy.float32
            assert ours.tobytes() == theirs.tobytes(), (
                f"{name} diverged from stepping.update_E at sub-step call "
                f"{step + 1}: max abs delta "
                f"{numpy.max(numpy.abs(ours - theirs)):.3e}")


def test_the_reference_matches_stepping_on_the_corpus_class(nl):
    """Scalar chi3 everywhere, metallic walls — 3rd-harm-1d's family."""
    fields, pml = _build(boundaries="metallic")
    _pin_against_stepping(nl, fields, pml, ("metallic",) * 3)


def test_the_reference_matches_stepping_on_mixed_boundaries_with_chi2(nl):
    fields, pml = _build(
        boundaries=("periodic", "metallic", "periodic"),
        chi2={"Ex": CHI2_TEST, "Ey": CHI2_TEST, "Ez": CHI2_TEST},
        chi3={"Ex": CHI3_CORPUS, "Ey": CHI3_CORPUS, "Ez": CHI3_CORPUS})
    _pin_against_stepping(nl, fields, pml, ("periodic", "metallic", "periodic"))


def test_the_reference_matches_stepping_on_partial_volume_nonlinearity(nl):
    shape = Grid(resolution=10.0, cell_size=(0.8, 0.8, 0.8),
                 dimensions=3, courant=0.35).shape
    rng = numpy.random.default_rng(23)
    chi2 = rng.uniform(-0.05, 0.05, size=shape).astype(numpy.float32)
    chi3 = rng.uniform(0.0, 0.1, size=shape).astype(numpy.float32)
    fields, pml = _build(chi2={"Ez": chi2}, chi3={"Ez": chi3})
    _pin_against_stepping(nl, fields, pml, ("periodic",) * 3)


def test_the_reference_matches_stepping_on_a_1d_grid(nl):
    """The (1, 1, n) marquee shape: collapsed axes wrap onto themselves."""
    fields, pml = _build(cell_size=(0.0, 0.0, 8.0), dimensions=1,
                         boundaries=("periodic", "periodic", "metallic"))
    boundaries = tuple(stepping._boundary_kinds(fields.grid, pml))
    _pin_against_stepping(nl, fields, pml, boundaries)


def test_the_four_corner_association_is_the_pair_one_not_meep_cs(nl):
    """(c + d) + (u + ud) — not MEEP C's ((c + u) + d) + ud left-to-right sum
    (step_generic.cpp:646-648). The two differ in bytes, which is what makes
    the gate's association mutation a real leg."""
    fields, pml = _build()
    displacement = stepping._nonlinear_displacement(fields, pml)
    theirs = stepping._nonlinear_transverse_sums(fields, "Ez", displacement)
    volumes = {name: getattr(fields, "D" + name[1]) for name in ("Ex", "Ey", "Ez")}
    ours = _transverse_sums(volumes, 2, ("periodic",) * 3)
    for a, b in zip(ours, theirs):
        assert a.tobytes() == b.tobytes()
    g = volumes["Ex"]  # Ez's first partner is Dx (down x, up z)
    down = _shift_down(g, 0, "periodic")
    up = _shift_up(g, 2, "periodic")
    corner = _shift_up(down, 2, "periodic")
    meep_c_order = ((g + up) + down) + corner
    assert meep_c_order.tobytes() != theirs[0].tobytes(), (
        "the two associations coincide on this seed — the association is not "
        "byte-visible here and the gate mutation would be vacuous")


def test_the_opposite_shift_directions_are_load_bearing(nl):
    """Down the partner's axis, UP the own axis (stepping.py:1160-1171); both
    taken the same way is the half-cell registration error."""
    fields, pml = _build()
    displacement = stepping._nonlinear_displacement(fields, pml)
    theirs = stepping._nonlinear_transverse_sums(fields, "Ez", displacement)
    g = getattr(fields, "Dx")
    pair = g + _shift_down(g, 0, "periodic")
    same_way = pair + _shift_down(pair, 2, "periodic")  # DOWN own axis: wrong
    assert same_way.tobytes() != theirs[0].tobytes()


def test_the_zero_chi_seam_is_bit_identical_to_the_linear_engine(nl):
    """An installed-then-trivial pair falls to the linear path bit for bit
    (fields.py:853-857) — and the control is non-vacuous: a live chi differs."""
    trivial, pml = _build(chi2={"Ez": 0.0}, chi3={"Ez": 0.0}, seed=77)
    control, control_pml = _build(chi2={}, chi3={}, seed=77)
    live, live_pml = _build(seed=77)
    stepping.update_E(trivial, pml)
    stepping.update_E(control, control_pml)
    stepping.update_E(live, live_pml)
    for name in ("Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez"):
        assert getattr(trivial, name).tobytes() == getattr(control, name).tobytes()
    assert any(
        getattr(live, name).tobytes() != getattr(control, name).tobytes()
        for name in ("Ex", "Ey", "Ez")), (
        "the live-chi control did not separate from the linear run — the seam "
        "check is vacuous on this seed")


# ---------------------------------------------------------------------------
# Plan shapes — buildable on this laptop, launchable only on the device
# ---------------------------------------------------------------------------

def _plan_arrays(shape=(4, 3, 5), scalar_epsilon=False, seed=9):
    rng = numpy.random.default_rng(seed)
    arrays = {}
    for name in ("Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez",
                 "Dx", "Dy", "Dz"):
        arrays[name] = rng.uniform(-1, 1, size=shape).astype(numpy.float32)
    for name in ("Ex", "Ey", "Ez"):
        arrays["inv_eps_" + name] = (
            US_DISCRIMINATOR if scalar_epsilon
            else rng.uniform(0.2, 0.9, size=shape).astype(numpy.float32))
    flat = {}
    for axis, n in zip("xyz", shape):
        flat["kps_" + axis] = rng.uniform(0.5, 1.0, size=n).astype(numpy.float32)
        flat["kms_" + axis] = rng.uniform(0.5, 1.0, size=n).astype(numpy.float32)
    return arrays, flat


def test_the_from_arrays_plan_reports_its_shape(nl):
    arrays, flat = _plan_arrays()
    plan = nl.plan_nonlinear_constitutive_from_arrays(
        arrays, flat,
        {"Ex": None, "Ey": None, "Ez": CHI2_TEST},
        {"Ex": None, "Ey": None, "Ez": CHI3_CORPUS},
        (0, 1, 0))
    assert plan.nonlinear == (0, 0, 1)
    assert plan.boundary_codes == (0, 1, 0)
    assert plan.epsilon_is_volume == (1, 1, 1)
    assert plan.chi2_is_volume == (0, 0, 0)
    assert plan.chi2_scalars[2] == nl.scalar_chi_arm(CHI2_TEST)
    assert plan.chi3_scalars[2] == nl.scalar_chi_arm(CHI3_CORPUS)


def test_the_scalar_epsilon_plan_ships_the_double_power_arm(nl):
    arrays, flat = _plan_arrays(scalar_epsilon=True)
    plan = nl.plan_nonlinear_constitutive_from_arrays(
        arrays, flat, {"Ez": CHI2_TEST}, {"Ez": CHI3_CORPUS}, (0, 0, 0))
    assert plan.epsilon_is_volume == (0, 0, 0)
    assert plan.epsilon_scalars[0] == nl.scalar_inverse_epsilon_arm(US_DISCRIMINATOR)


def test_a_chi_pair_split_between_the_two_maps_is_refused(nl):
    arrays, flat = _plan_arrays()
    with pytest.raises(ValueError, match="both or neither"):
        nl.plan_nonlinear_constitutive_from_arrays(
            arrays, flat, {"Ez": CHI2_TEST}, {"Ez": None}, (0, 0, 0))


def test_an_all_linear_plan_is_refused_toward_the_certified_kernel(nl):
    arrays, flat = _plan_arrays()
    with pytest.raises(ValueError, match="certified plain constitutive"):
        nl.plan_nonlinear_constitutive_from_arrays(
            arrays, flat, {}, {}, (0, 0, 0))


def test_an_output_aliasing_an_input_is_refused(nl):
    arrays, flat = _plan_arrays()
    arrays["Ex"] = arrays["Dx"]  # the schedule-dependent wrong answer
    with pytest.raises(ValueError, match="alias"):
        nl.plan_nonlinear_constitutive_from_arrays(
            arrays, flat, {"Ez": CHI2_TEST}, {"Ez": CHI3_CORPUS}, (0, 0, 0))


def test_a_coefficient_vector_aliasing_an_output_is_refused(nl):
    """The six kps/kms vectors are in the alias inventory too: a coefficient
    re-read per element while its output is written is the same
    schedule-dependent wrong answer (the audit found them missing)."""
    arrays, flat = _plan_arrays()
    shape = arrays["Ex"].shape
    flat["kps_x"] = arrays["Ex"].ravel()[:shape[0]]  # view, same base address
    with pytest.raises(ValueError, match="alias"):
        nl.plan_nonlinear_constitutive_from_arrays(
            arrays, flat, {"Ez": CHI2_TEST}, {"Ez": CHI3_CORPUS}, (0, 0, 0))


def test_bad_boundary_codes_are_refused(nl):
    arrays, flat = _plan_arrays()
    with pytest.raises(ValueError, match="boundary codes"):
        nl.plan_nonlinear_constitutive_from_arrays(
            arrays, flat, {"Ez": CHI2_TEST}, {"Ez": CHI3_CORPUS}, (0, 2, 0))


def test_planning_a_refused_configuration_returns_none_without_triton(nl):
    fields, pml = _build()
    assert nl.plan_nonlinear_constitutive(fields, pml) is None, (
        "this laptop's NumPy backend must refuse, and the builder must answer "
        "None without needing the optional dependency")


# ---------------------------------------------------------------------------
# The OTHER sub-steps of a nonlinear run — admission only, no new kernel
# ---------------------------------------------------------------------------
#
# Closes residual group (A): 6 slots on 3rd-harm-1d.py and
# Test3rdHarm1d.test_3rd_harm_1d, the only residual group whose rows are all
# real physics (zero TestLoadDump) and that takes them to FULL whole-step
# coverage. The device leg's numbers are quoted in the module and re-stated in
# `_GROUP_A_MEASURED` below so a drifting comment fails here.

#: The device leg's verdicts (results/residual_closure_2026-08-15/device/
#: bodies/bodies.json), 8 complete cycles each, uint32 over the whole stored
#: inventory, with the +-0 lattice held live through every cycle.
_GROUP_A_MEASURED = {
    "A_chi_step_B": (0, 12288),
    "A_chi_step_D": (0, 12288),
    "A_chi_update_H": (0, 12288),
    "A_chi_update_E_CONTROL": (1536, 12288),
}

_ALL_VOLUMES = ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
                "Hx", "Hy", "Hz",
                "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
                "f_w_Ex", "f_w_Ey", "f_w_Ez", "f_w_Hx", "f_w_Hy", "f_w_Hz")


def _seeded(fields, seed=7):
    """Fill every stored volume with the same pseudo-random bytes for a given seed."""
    rng = numpy.random.default_rng(seed)
    for name in _ALL_VOLUMES:
        array = getattr(fields, name, None)
        if array is None:
            continue
        array[...] = rng.uniform(-0.4, 0.4, size=array.shape).astype(numpy.float32)
    return fields


def _twins(**kwargs):
    """A nonlinear run and its LINEAR twin, seeded to identical bytes.

    The linear twin passes explicit zeros, which the installer DROPS
    (fields.py:853-857), so ``has_nonlinearity`` is False there and the two
    Fields differ in exactly one thing: whether chi2/chi3 is installed.
    """
    nonlinear, pml_n = _build(seed=7, **kwargs)
    linear, pml_l = _build(seed=7, chi2={"Ex": 0.0, "Ey": 0.0, "Ez": 0.0},
                           chi3={"Ex": 0.0, "Ey": 0.0, "Ez": 0.0}, **kwargs)
    assert nonlinear.has_nonlinearity and not linear.has_nonlinearity
    return _seeded(nonlinear), pml_n, _seeded(linear), pml_l


def _bytes_of(fields, names):
    return {name: getattr(fields, name).tobytes() for name in names}


def _differing_words(left, right):
    a = numpy.frombuffer(left, dtype=numpy.uint32)
    b = numpy.frombuffer(right, dtype=numpy.uint32)
    return int(numpy.count_nonzero(a != b))


def _run_and_compare(sub_step, written, cycles=4, **kwargs):
    """Run ``sub_step`` on a nonlinear run and its linear twin; return (differing, moved).

    uint32 over the WHOLE written inventory, never ``allclose``. ``moved`` is the
    nonlinear leg's own change against its pre-state, which is what stops a
    no-op agreeing with a no-op from scoring as identity.
    """
    nonlinear, pml_n, linear, pml_l = _twins(**kwargs)
    before = _bytes_of(nonlinear, written)
    differing = 0
    moved = 0
    for _ in range(cycles):
        sub_step(nonlinear, pml_n)
        sub_step(linear, pml_l)
        for name in written:
            differing += _differing_words(getattr(nonlinear, name).tobytes(),
                                          getattr(linear, name).tobytes())
    for name in written:
        moved += _differing_words(before[name], getattr(nonlinear, name).tobytes())
    return differing, moved


_CURL_WRITTEN = {
    "step_B": ("Bx", "By", "Bz", "fu_Bx", "fu_By", "fu_Bz"),
    "step_D": ("Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz"),
}
_H_WRITTEN = ("Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz")
_E_WRITTEN = ("Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez")


@pytest.mark.parametrize("sub_step", ("step_B", "step_D"))
def test_chi_does_not_enter_either_curl_and_the_leg_is_not_vacuous(nl, sub_step):
    """The reference test for the curl half of group (A).

    A nonlinear run and its linear twin, from identical bytes, four complete
    cycles: the array path's own curl must write the same words. If it does,
    then the CERTIFIED curl kernel — which binds no chi2, chi3 or inverse
    epsilon pointer at all — cannot see the difference either, which is what the
    device leg measured at 0/12288.
    """
    differing, moved = _run_and_compare(
        getattr(stepping, sub_step), _CURL_WRITTEN[sub_step])
    assert moved > 0, f"{sub_step} moved no words: a no-op agreeing with a no-op"
    assert differing == 0, (
        f"{sub_step} differs between a nonlinear run and its linear twin in "
        f"{differing} words; the device leg measured "
        f"{_GROUP_A_MEASURED['A_chi_' + sub_step]}")


def test_chi_does_not_enter_update_H_and_the_leg_is_not_vacuous(nl):
    differing, moved = _run_and_compare(stepping.update_H, _H_WRITTEN)
    assert moved > 0, "update_H moved no words"
    assert differing == 0, f"update_H differs in {differing} words"


def test_the_CONTROL_update_E_does_differ(nl):
    """The measurement is only sensitive because this one diverges.

    ``update_E`` is where the Pade factor lives (stepping.py:975-979, :999-1000),
    and the device leg's CONTROL measured 1536 of 12288 words differing. A leg
    that could not see THAT would not be evidence about the other three.
    """
    differing, moved = _run_and_compare(stepping.update_E, _E_WRITTEN)
    assert moved > 0
    assert differing > 0, (
        "update_E agreed between a nonlinear run and its linear twin: the "
        "comparison is blind and the three identity legs above prove nothing")


def test_the_predicates_admit_the_nonlinear_run_the_incumbents_refuse(nl):
    fields, pml = _build()
    for sub_step in ("step_B", "step_D"):
        assert _reasons_of(nl.nonlinear_run_pml_curl_coverage(
            fields, pml, sub_step)) == [], sub_step
        assert "chi2/chi3" in " ".join(
            _coverage_module().pml_curl_coverage(fields, pml, sub_step).reasons)
    assert _reasons_of(nl.nonlinear_run_constitutive_coverage(fields, pml, "H")) == []
    assert "chi2/chi3" in " ".join(
        _coverage_module().constitutive_coverage(fields, pml, "H").reasons)


def test_the_two_families_are_disjoint_on_a_linear_run(nl):
    """A linear run must be refused HERE and admitted THERE — no overlap either way."""
    fields, pml = _build(chi2={"Ex": 0.0, "Ey": 0.0, "Ez": 0.0},
                         chi3={"Ex": 0.0, "Ey": 0.0, "Ez": 0.0})
    assert not fields.has_nonlinearity
    for sub_step in ("step_B", "step_D"):
        reasons = _reasons_of(nl.nonlinear_run_pml_curl_coverage(fields, pml, sub_step))
        assert any("no chi2/chi3 is installed" in reason for reason in reasons), reasons
    reasons = _reasons_of(nl.nonlinear_run_constitutive_coverage(fields, pml, "H"))
    assert any("no chi2/chi3 is installed" in reason for reason in reasons), reasons
    # And the incumbents take it, on this laptop's backend clause alone.
    assert _reasons_of(_coverage_module().pml_curl_coverage(fields, pml, "step_B")) == []
    assert _reasons_of(_coverage_module().constitutive_coverage(fields, pml, "H")) == []


def test_side_E_is_refused_by_name_and_points_at_the_built_kernel(nl):
    fields, pml = _build()
    reasons = nl.nonlinear_run_constitutive_coverage(fields, pml, "E").reasons
    joined = " ".join(reasons)
    assert "side='E'" in joined and "nonlinear_constitutive_coverage" in joined
    assert "1536/12288" in joined, "the refusal must carry its measurement"
    assert nl.plan_nonlinear_run_constitutive(fields, pml, "E") is None


@pytest.mark.parametrize("side", ("H", "E"))
def test_an_unknown_side_raises_but_a_real_side_refuses(nl, side):
    fields, pml = _build()
    with pytest.raises(ValueError):
        nl.nonlinear_run_constitutive_coverage(fields, pml, "B")
    assert isinstance(nl.nonlinear_run_constitutive_coverage(fields, pml, side).covered,
                      bool)


def test_an_unknown_sub_step_raises(nl):
    fields, pml = _build()
    with pytest.raises(ValueError):
        nl.nonlinear_run_pml_curl_coverage(fields, pml, "update_E")


# --- per-clause refusals, one perturbation off the admitted class each --------

def test_conductivity_is_refused_per_sub_step_not_per_run(nl):
    """The conductive curl is a DIFFERENT recurrence; the opposite curl is not."""
    fields, pml = _build()
    fields.set_d_conductivity(numpy.full(fields.grid.shape, 0.2, dtype=numpy.float32))
    reasons = _reasons_of(nl.nonlinear_run_pml_curl_coverage(fields, pml, "step_D"))
    assert any("conductivity is installed on D" in reason for reason in reasons), reasons
    assert _reasons_of(nl.nonlinear_run_pml_curl_coverage(fields, pml, "step_B")) == []
    # And it does not touch the constitutive side at all — the same split
    # coverage.constitutive_coverage makes (it carries no conductivity clause).
    assert _reasons_of(nl.nonlinear_run_constitutive_coverage(fields, pml, "H")) == []


def test_an_inactive_layer_is_refused_with_update_H_s_own_reason(nl):
    fields, pml = _build(pml_thickness=0)
    for verdict in (nl.nonlinear_run_pml_curl_coverage(fields, pml, "step_B"),
                    nl.nonlinear_run_constitutive_coverage(fields, pml, "H")):
        joined = " ".join(verdict.reasons)
        assert "no active PML layer" in joined
        assert "stepping.py:944-945" in joined, (
            "the reason must be true where it fires: update_H RETURNS there")


def test_complex_storage_is_refused_toward_the_complex_tranche(nl):
    fields, pml = _build(complex_storage=True)
    joined = " ".join(nl.nonlinear_run_pml_curl_coverage(fields, pml, "step_B").reasons)
    assert "complex_fields.py's certified pair" in joined


def test_a_fold_is_refused_on_both_new_predicates(nl):
    from meep_gpu.grid import Mirror
    fields, pml = _build(cell_size=(1.6, 2.0, 0.0), dimensions=2,
                         boundaries="periodic", symmetry=(Mirror("Y", 1),),
                         thickness=((2, 2), (0, 2), (0, 0)))
    for verdict in (nl.nonlinear_run_pml_curl_coverage(fields, pml, "step_B"),
                    nl.nonlinear_run_constitutive_coverage(fields, pml, "H")):
        assert any("folded by a mirror plane" in r for r in verdict.reasons)


def test_beta_and_bloch_are_refused(nl):
    fields, pml = _build(cell_size=(1.6, 1.6, 0.0), dimensions=2, beta=0.3,
                         thickness=((2, 2), (2, 2), (0, 0)))
    assert any("beta" in r for r in nl.nonlinear_run_pml_curl_coverage(
        fields, pml, "step_B").reasons)
    fields, pml = _build(k_point=(0.1, 0.0, 0.0))
    assert any("k_point" in r for r in nl.nonlinear_run_constitutive_coverage(
        fields, pml, "H").reasons)


def test_a_magnetic_susceptibility_is_refused_by_the_shared_clause(nl):
    fields, pml = _build()
    fields.polarizations.append(SimpleNamespace(
        susceptibility=SimpleNamespace(kind="lorentzian"),
        driven=lambda: ("Hx",), drives=lambda name: name == "Hx"))
    for verdict in (nl.nonlinear_run_pml_curl_coverage(fields, pml, "step_B"),
                    nl.nonlinear_run_constitutive_coverage(fields, pml, "H")):
        assert any("Hx" in r for r in verdict.reasons), verdict.reasons


def test_a_missing_auxiliary_is_refused_by_name(nl):
    fields, pml = _build()
    fields.fu_Dz = None
    assert any("fu_Dz is not allocated" in r
               for r in nl.nonlinear_run_pml_curl_coverage(fields, pml, "step_B").reasons)


@pytest.mark.parametrize("sub_step,target",
                         (("step_B", "Bx"), ("step_B", "By"), ("step_B", "Bz"),
                          ("step_D", "Dx"), ("step_D", "Dy"), ("step_D", "Dz")))
def test_an_unallocated_curl_target_is_refused_and_the_builder_answers_none(
        nl, sub_step, target):
    """The ONE clause this restatement states in force that the incumbent does not.

    ``coverage.pml_curl_coverage``'s allocation clause (:307-315) lists
    ``fu_* + B_SOURCES + D_SOURCES`` and stops; ``_layout_reasons`` then SKIPS a
    ``None`` on the stated grounds that it is "Already reported by the allocation
    clause" (coverage.py:630) — which, for a CURL TARGET, nothing is. So a
    ``Fields`` missing ``Bx`` is ADMITTED there and ``plan_pml_curl`` builds a
    plan binding the ``None``, against the builder contract that an uncarried
    configuration falls back to the array path and never raises into the caller.

    Narrowing a restatement is always safe — a refusal IS the array path — so the
    clause is stated here. The incumbent's own hole is asserted, not assumed, so
    that this test starts failing the day the owning round closes it and the two
    predicates can be brought back into step deliberately.
    """
    fields, pml = _build()
    setattr(fields, target, None)
    reasons = _reasons_of(nl.nonlinear_run_pml_curl_coverage(fields, pml, sub_step))
    assert any(f"{target} is not allocated" in reason for reason in reasons), reasons
    assert nl.plan_nonlinear_run_pml_curl(fields, pml, sub_step) is None

    # The incumbent, on the same object: still admitting. Its own chi clause is
    # dropped first — that is the clause this family inverts, and it is not the
    # question here. When this flips, the shared clause has been fixed and this
    # test's first half is the redundancy.
    incumbent = [r for r in _reasons_of(
        _coverage_module().pml_curl_coverage(fields, pml, sub_step))
        if "chi2/chi3 is installed" not in r]
    assert incumbent == [], (
        "coverage.pml_curl_coverage now refuses an unallocated curl target; the "
        "hole this clause exists to cover is closed and the restatement can be "
        "re-derived from the shared clause set")


def test_the_target_narrowing_is_the_only_clause_that_differs(nl):
    """Disjointness is unaffected: with everything allocated the two agree.

    Guards against the narrowing being written as a broader clause than intended
    — e.g. one that also fires on a well-formed run and silently pushes group
    (A) back onto the array path, which would look like a passing test suite and
    a coverage regression.
    """
    fields, pml = _build()
    for sub_step in ("step_B", "step_D"):
        assert _reasons_of(
            nl.nonlinear_run_pml_curl_coverage(fields, pml, sub_step)) == []


# --- mutations: each new clause must be load-bearing --------------------------

def test_mutation_dropping_the_inverted_chi_clause_breaks_disjointness(nl,
                                                                       monkeypatch):
    """Without clause 10 inverted, a LINEAR run is admitted by both families."""
    original = nl._nonlinear_run_grid_reasons

    def mutated(fields, pml, grid):
        return [r for r in original(fields, pml, grid)
                if "no chi2/chi3 is installed" not in r]

    monkeypatch.setattr(nl, "_nonlinear_run_grid_reasons", mutated)
    fields, pml = _build(chi2={"Ex": 0.0, "Ey": 0.0, "Ez": 0.0},
                         chi3={"Ex": 0.0, "Ey": 0.0, "Ez": 0.0})
    assert _reasons_of(nl.nonlinear_run_pml_curl_coverage(fields, pml, "step_B")) == [], (
        "the mutation is DISARMED: the clause it removes was not the one keeping "
        "the linear run out")
    assert _reasons_of(_coverage_module().pml_curl_coverage(fields, pml, "step_B")) == []


def test_mutation_dropping_the_side_E_refusal_admits_the_pade_configuration(nl,
                                                                            monkeypatch):
    real = nl.nonlinear_run_constitutive_coverage

    def mutated(fields, pml, side):
        verdict = real(fields, pml, side)
        kept = tuple(r for r in verdict.reasons if "side='E'" not in r)
        return type(verdict)(not kept, kept)

    monkeypatch.setattr(nl, "nonlinear_run_constitutive_coverage", mutated)
    fields, pml = _build()
    assert _reasons_of(nl.nonlinear_run_constitutive_coverage(fields, pml, "E")) == [], (
        "the mutation is DISARMED: side='E' was refused by some other clause")


def test_the_restated_clause_set_tracks_coverages_own(nl):
    """Every clause `coverage._grid_reasons` fires must still fire here — except 10.

    Not a text comparison: the two are driven over the same perturbed
    configurations and the VERDICTS compared, so a clause silently dropped from
    the restatement fails here.
    """
    coverage = _coverage_module()
    from meep_gpu.grid import Mirror
    cases = [
        ({}, True),
        ({"complex_storage": True}, False),
        ({"pml_thickness": 0}, False),
        ({"k_point": (0.1, 0.0, 0.0)}, False),
        ({"cell_size": (1.6, 1.6, 0.0), "dimensions": 2, "beta": 0.3,
          "thickness": ((2, 2), (2, 2), (0, 0))}, False),
        ({"cell_size": (1.6, 2.0, 0.0), "dimensions": 2,
          "boundaries": "periodic", "symmetry": (Mirror("Y", 1),),
          "thickness": ((2, 2), (0, 2), (0, 0))}, False),
    ]
    for kwargs, expected in cases:
        fields, pml = _build(**kwargs)
        ours = _reasons_of(nl.nonlinear_run_pml_curl_coverage(fields, pml, "step_B"))
        # The incumbent always carries the chi clause; drop it and compare the rest.
        theirs = [r for r in _reasons_of(
            coverage.pml_curl_coverage(fields, pml, "step_B"))
            if "chi2/chi3 is installed" not in r]
        assert (not ours) == expected, (kwargs, ours)
        # Compared on the clause's OPENING WORDS, not its full text: the
        # restatement deliberately re-words several reasons so they are true
        # where they fire (the `no active PML` one names update_H's early
        # return), and a text-equality assertion would forbid exactly the
        # improvement the restatement exists to make.
        assert sorted(r[:28] for r in ours) == sorted(r[:28] for r in theirs), (
            kwargs, ours, theirs)


def test_the_module_records_the_measurement_that_licensed_the_admission(nl):
    text = MODULE_PATH.read_text(encoding="utf-8")
    for case, (differing, compared) in _GROUP_A_MEASURED.items():
        assert case in text, f"{case} is not recorded in the module"
        assert f"{differing} / {compared}" in text or f"{differing}/{compared}" in text


#: The legs that ran through THIS MODULE's builders with THIS MODULE's predicates
#: unpatched (results/residual_closure_2026-08-15/device/newpred/new_predicates.json).
#: A leg here is evidence about the ADMISSION; `_GROUP_A_MEASURED` above is
#: evidence about the KERNEL BODY, and the two are not interchangeable.
_GROUP_A_OWN_BUILDER = {
    "NEW_nonlinear_run_step_B": (0, 12288),
    "NEW_nonlinear_run_step_D": (0, 12288),
    "NEW_nonlinear_run_update_H": (0, 12288),
}


def test_the_admission_itself_was_measured_not_only_the_kernel_body(nl):
    """The module must record a leg through its OWN builders, not only the triage.

    The triage legs put the certified body on the configuration by patching the
    INCUMBENT predicate in front of the INCUMBENT builder. That licenses a claim
    about ``kernels.pml_curl_step`` and ``launch.ConstitutivePlan``; it says
    nothing about ``nonlinear_run_pml_curl_coverage`` or
    ``plan_nonlinear_run_pml_curl``, which are the objects that will do the
    admitting and where a clause set and an array resolution sit in between.
    """
    text = MODULE_PATH.read_text(encoding="utf-8")
    assert "new_predicates.json" in text, (
        "the module cites no leg through its own builders")
    for case, (differing, compared) in _GROUP_A_OWN_BUILDER.items():
        assert case in text, f"{case} is not recorded in the module"
        assert f"{differing} / {compared}" in text or f"{differing}/{compared}" in text
    assert "unpatched" in text, (
        "the module must say the leg ran with its own predicate unpatched; a "
        "patched one would be measuring the incumbent again")


def test_the_builders_answer_none_on_this_laptop(nl):
    fields, pml = _build()
    assert nl.plan_nonlinear_run_pml_curl(fields, pml, "step_B") is None
    assert nl.plan_nonlinear_run_constitutive(fields, pml, "H") is None
    with pytest.raises(ValueError):
        nl.plan_nonlinear_run_pml_curl(fields, pml, "nope")


def _reasons_of(verdict):
    return [r for r in verdict.reasons if "array module" not in r]


def _coverage_module():
    return importlib.import_module("meep_gpu.triton_kernels.coverage")
