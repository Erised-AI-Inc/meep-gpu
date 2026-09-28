"""Laptop contracts for the BFAST Triton tranche.

Everything here runs on the NumPy merge-bar machine: the optional-import
contract, the predicates' admission AND per-clause refusals on real
Grid/Fields/PML objects, the plan builders' shapes, the host coefficient
transcription pinned against ``stepping._bfast_term``'s own arithmetic, and a
third, in-test transcription of the BFAST curl sub-steps pinned BYTE-for-byte
against ``stepping.step_B``/``step_D`` themselves — six IIR states included.
The kernel's device bytes are the gate's
(``parity/meep_gpu/gate_triton_bfast.py``); nothing here launches.
"""

from __future__ import annotations

import importlib
import math
import pathlib
import re
import sys
from types import SimpleNamespace

import numpy
import pytest

from meep_gpu import stepping
from meep_gpu.fields import BFAST_COMPONENTS, IYEE_SHIFTS, Fields
from meep_gpu.grid import Grid
from meep_gpu.pml import PML
from meep_gpu.test_triton_kernels import GUARD_SPELLING, PACKAGE_DIR, code_of

MODULE_PATH = PACKAGE_DIR / "bfast_curl.py"
MODULE_NAME = "meep_gpu.triton_kernels.bfast_curl"
PARITY_DIR = pathlib.Path(__file__).resolve().parents[1] / "parity" / "meep_gpu"
GATE_PATH = PARITY_DIR / "gate_triton_bfast.py"
COMPOSITION_PATH = PARITY_DIR / "probe_triton_bfast_composition.py"
SLURM_PATH = PARITY_DIR / "run_triton_bfast_composition.slurm"

#: The marquee's own numbers, computed the way MEEP's test computes them
#: (test_refl_angular.py:50-52 does both in USER code): the demand case is
#: n1 * sin(35.7 deg) along x with Courant (1 - kx)/sqrt(3).
KX_MARQUEE = 1.4 * math.sin(math.radians(35.7))       # ~0.816958
COURANT_MARQUEE = (1.0 - KX_MARQUEE) / math.sqrt(3.0)  # ~0.105679, natively NP2
FULL_K = (0.31, 0.17, 0.23)  # all distinct: the k-indexing needle's values

BFAST_STATE_ALL = tuple("f_bfast_" + name for name in BFAST_COMPONENTS)


@pytest.fixture(scope="module")
def bf():
    """The shipped module itself must be importable without a test-only stub."""
    return importlib.import_module(MODULE_NAME)


def _build(cell_size=(0.1, 0.1, 4.8), resolution=10.0, dimensions=3,
           bfast=(KX_MARQUEE, 0.0, 0.0), courant=COURANT_MARQUEE,
           complex_storage=False, k_point=(0.0, 0.0, 0.0),
           pml_override=None, pml_thickness=2, **grid_kwargs):
    """A real BFAST Grid/Fields/PML triple on NumPy.

    Default: the marquee's own class scaled down — dimensions=3 with one-cell
    x/y axes (so every invariance flag is FALSE, grid.py:1165-1183), a z-only
    cell, PML both z faces, kx = 1.4*sin(35.7 deg), the NP2 marquee Courant,
    k_point exactly zero, real f32 storage. Every refusal test perturbs
    exactly one clause off this.
    """
    grid = Grid(resolution=resolution, cell_size=cell_size,
                dimensions=dimensions, courant=courant, k_point=k_point,
                bfast_scaled_k=bfast, **grid_kwargs)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.enable_pml_storage()
    if pml_override is not None:
        return fields, PML(grid=grid, thickness=pml_override)
    thickness = tuple(
        (pml_thickness, pml_thickness) if grid.shape[axis] >= 6 else (0, 0)
        for axis in range(3))
    return fields, PML(grid=grid, thickness=thickness)


def _full_k_build(**kwargs):
    """The k-indexing needle's class: 12x10x14, PML z only, periodic x/y."""
    defaults = dict(cell_size=(1.2, 1.0, 1.4), bfast=FULL_K, courant=0.35,
                    pml_override={"z": 3})
    defaults.update(kwargs)
    return _build(**defaults)


def _guard_2d_build(**kwargs):
    """m6's needle class: DECLARED dimensions=2, kx only, PML x+y."""
    defaults = dict(cell_size=(2.0, 1.6, 0.0), dimensions=2,
                    bfast=(0.4, 0.0, 0.0), courant=0.4375)
    defaults.update(kwargs)
    return _build(**defaults)


def _dims1_build(**kwargs):
    """The D1 arm of the invariance guard: DECLARED dimensions=1, so x AND y
    are invariant and four of the six flags are false at once.

    k must ride z — clause 11a refuses a k component on x or y here, which is
    every OTHER D1 k, so this is the whole admitted D1 class."""
    defaults = dict(cell_size=(0.0, 0.0, 4.8), dimensions=1,
                    bfast=(0.0, 0.0, 0.31), courant=0.35)
    defaults.update(kwargs)
    return _build(**defaults)


def _curl_reasons(bf, fields, pml, sub_step="step_B"):
    verdict = bf.bfast_pml_curl_coverage(fields, pml, sub_step)
    return [r for r in verdict.reasons if "array module" not in r]


def _constitutive_reasons(bf, fields, pml, side="E"):
    verdict = bf.bfast_run_constitutive_coverage(fields, pml, side)
    return [r for r in verdict.reasons if "array module" not in r]


# ---------------------------------------------------------------------------
# The optional dependency must stay optional; the restatements must not drift
# ---------------------------------------------------------------------------

def test_the_package_does_not_import_bfast_curl_eagerly():
    for name in list(sys.modules):
        if name == MODULE_NAME:
            del sys.modules[name]
    importlib.import_module("meep_gpu.triton_kernels")
    eagerly_imported = MODULE_NAME in sys.modules
    importlib.import_module(MODULE_NAME)  # restore for the other tests
    assert not eagerly_imported, (
        "the package __init__ must not pull this unwired module in")


def test_the_module_answers_coverage_but_kernel_fails_clearly_without_triton(bf):
    if not isinstance(bf.bfast_pml_curl_step, bf._UnavailableKernel):
        pytest.skip("triton is installed here; the absence arm is exercised on "
                    "the CI/laptop box without it")
    fields, pml = _build()
    assert bf.bfast_pml_curl_coverage(fields, pml, "step_B").reasons
    with pytest.raises(ImportError, match="triton"):
        bf.bfast_pml_curl_step[(1,)]


def test_the_restated_constants_pin_their_originators(bf):
    def value(x):
        return getattr(x, "value", x)

    assert value(bf.PERIODIC) == 0 and value(bf.METALLIC) == 1
    assert bf.DEFAULT_BLOCK == 256
    # Read kernels.py off the source (importing it needs Triton).
    kernels_source = (PACKAGE_DIR / "kernels.py").read_text(encoding="utf-8")
    assert "\nPERIODIC = tl.constexpr(0)\n" in kernels_source
    assert "\nMETALLIC = tl.constexpr(1)\n" in kernels_source
    assert "\nDEFAULT_BLOCK = 256\n" in kernels_source


def test_the_restated_term_table_pins_stepping(bf):
    """BFAST_TERMS must be stepping.B_CURL_TERMS/D_CURL_TERMS' (first,
    first_axis, second, second_axis) — the k assignment and the invariance
    guard both hang off it, so drift is a silent wrong answer."""
    for sub_step, engine_terms in (("step_B", stepping.B_CURL_TERMS),
                                   ("step_D", stepping.D_CURL_TERMS)):
        restated = bf.BFAST_TERMS[sub_step]
        assert len(restated) == len(engine_terms) == 3
        for (first, first_axis, second, second_axis), term in zip(
                restated, engine_terms):
            assert first == term.first
            assert first_axis == term.first_axis
            assert second == term.second
            assert second_axis == term.second_axis
        assert bf.BFAST_STATE_NAMES[sub_step] == tuple(
            "f_bfast_" + term.target for term in engine_terms)


def test_the_own_axis_helper_pins_stepping_bfast_axis(bf):
    for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
                      "Bx", "By", "Bz", "Dx", "Dy", "Dz"):
        assert bf._own_axis(component) == stepping._bfast_axis(component)


def test_the_module_docstring_states_dispatch_is_untouched(bf):
    assert "NOT WIRED" in bf.__doc__
    assert "plan_fast_path" in bf.__doc__


def test_the_module_does_not_reach_into_another_tracks_file():
    code = code_of(MODULE_PATH)
    assert "fastpath" not in code
    assert "cuda_kernels" not in code


def test_the_guard_is_the_packages_one_spelling_and_appears_once():
    code = MODULE_PATH.read_text(encoding="utf-8")
    assert code.count("enable_fp_fusion=") == 1  # the one plan run() method
    assert code.count(GUARD_SPELLING) == 1


def test_the_fingerprint_welds_are_claimed(bf):
    import json

    recorded = json.loads((PACKAGE_DIR / "fingerprints.json")
                          .read_text(encoding="utf-8"))
    welds = {name for name in recorded if "bfast" in name}
    assert welds == {
        "triton_bfast_fused_electric_pair_device_gate",
        "triton_bfast_fused_magnetic_pair_device_gate",
    }, (
        "Phase B released the BFAST fused pair; each arm seeded its weld in "
        "the fingerprints.json the dispatch route reads")


# ---------------------------------------------------------------------------
# The positive verdicts — refused only for the backend on this laptop
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sub_step", ["step_B", "step_D"])
def test_the_marquee_configuration_is_refused_only_for_the_backend(bf, sub_step):
    """The demand case's own class: dims=3, one-cell x/y, z-PML, kx bfast,
    NP2 Courant, real storage, k_point zero."""
    fields, pml = _build()
    assert fields.grid.bfast_active is True
    assert _curl_reasons(bf, fields, pml, sub_step) == []


@pytest.mark.parametrize("sub_step", ["step_B", "step_D"])
def test_the_full_k_configuration_is_admitted(bf, sub_step):
    fields, pml = _full_k_build()
    assert _curl_reasons(bf, fields, pml, sub_step) == []


def test_the_guard_2d_configuration_is_admitted(bf):
    fields, pml = _guard_2d_build()
    assert fields.grid.is_invariant(2) is True
    assert _curl_reasons(bf, fields, pml, "step_B") == []


@pytest.mark.parametrize("side", ["H", "E"])
def test_constitutive_sides_are_admitted_on_a_bfast_run(bf, side):
    """The restated shipped predicate with the BFAST clause inverted."""
    fields, pml = _build()
    assert _constitutive_reasons(bf, fields, pml, side) == []


def test_dispersion_is_admitted_for_the_curl_and_split_off_update_e(bf):
    """The shipped curl predicate's doctrine carried over: dispersion is a
    constitutive-only feature, so the BFAST curl admits it and the E-side
    delegation refuses it (that configuration belongs to the ADE kernel)."""
    pole = SimpleNamespace(
        driven=lambda: ("Ex",),
        drives=lambda name: name == "Ex",
        susceptibility=SimpleNamespace(kind="lorentzian"),
    )
    fields, pml = _build()
    fields.polarizations = [pole]
    assert _curl_reasons(bf, fields, pml) == []
    assert any("ADE" in r for r in _constitutive_reasons(bf, fields, pml, "E"))


def test_offdiagonal_epsilon_is_admitted_for_the_curl_and_refused_on_e(bf):
    """Same doctrine for the tensor rows: constitutive-only, curl admits,
    E-side delegation refuses (the offdiag family owns that sub-step — and
    ITS predicate refuses bfast through the shared grid clauses, pinned in
    the fence test below)."""
    fields, pml = _build()
    fields._chi1inv_offdiagonal = {"Ex": {"Ey": numpy.full(
        fields.grid.shape, 0.1, dtype=numpy.float32)}}
    assert fields.has_offdiagonal_epsilon is True
    assert _curl_reasons(bf, fields, pml) == []
    assert any("off-diagonal" in r
               for r in _constitutive_reasons(bf, fields, pml, "E"))


# ---------------------------------------------------------------------------
# The refusals, one per silent-wrong-answer surface
# ---------------------------------------------------------------------------

def test_k_zero_is_refused_by_the_inverted_clause(bf):
    """A k = 0 run belongs to the certified plain kernels (the array path
    never enters the fold, stepping.py:364/:451)."""
    fields, pml = _build(bfast=(0.0, 0.0, 0.0))
    assert fields.grid.bfast_active is False
    reasons = _curl_reasons(bf, fields, pml)
    assert any("bfast_active is False" in r for r in reasons), reasons
    assert any("bfast_active is False" in r
               for r in _constitutive_reasons(bf, fields, pml, "E"))


def test_a_nonzero_k_on_an_invariant_axis_is_refused_by_clause_11a(bf):
    """Clause 11a: the ONE reachable class where the array path this kernel
    transcribes is not MEEP's answer.

    MEEP nulls the partner OPERAND as well as zeroing the coefficient
    (step_db.cpp:62-63 vs :130-133) and step_bfast swaps a null g1 into the g2
    slot with k1 := k2 (step_generic.cpp:342-346), so either false flag gives
    F_new = -F_prev with the other term DROPPED. stepping.py:911-912 zeroes
    only the coefficient. The two agree only while the surviving k is zero."""
    fields, pml = _guard_2d_build(bfast=(0.4, 0.0, 0.3))
    assert fields.grid.is_invariant(2) is True
    reasons = _curl_reasons(bf, fields, pml)
    assert any("DECLARED-invariant axis 2" in r for r in reasons), reasons
    assert any("DECLARED-invariant axis 2" in r
               for r in _constitutive_reasons(bf, fields, pml, "E"))
    # The in-plane k on the SAME grid stays admitted: only the invariant axis
    # carries the gap, and the marquee/guard needles must not be fenced off.
    assert _curl_reasons(bf, *_guard_2d_build()) == []


def test_the_refused_class_is_exactly_where_stepping_leaves_meep(bf):
    """The measurement behind clause 11a, on real objects, not a citation.

    MEEP's whole increment on a false flag is F_new = -F_prev. At kz = 0 the
    array path lands on that EXACTLY (max|diff| = 0 in f32) for the two guarded
    targets; at kz != 0 it does not — which is what clause 11a refuses."""
    def measure(bfast):
        fields, pml = _guard_2d_build(bfast=bfast)
        rng = numpy.random.default_rng(20260812)
        for name in ("Ex", "Ey", "Ez", "Bx", "By", "Bz",
                     "fu_Bx", "fu_By", "fu_Bz") + BFAST_STATE_ALL:
            getattr(fields, name)[...] = rng.uniform(
                -1, 1, fields.grid.shape).astype(numpy.float32)
        before = {name: numpy.array(getattr(fields, name))
                  for name in ("f_bfast_Bx", "f_bfast_By")}
        stepping.step_B(fields, pml)
        return {name: float(numpy.max(numpy.abs(
            numpy.asarray(getattr(fields, name)) - (-value))))
            for name, value in before.items()}

    agrees = measure((0.4, 0.0, 0.0))
    assert agrees == {"f_bfast_Bx": 0.0, "f_bfast_By": 0.0}, agrees
    diverges = measure((0.4, 0.0, 0.3))
    assert min(diverges.values()) > 1e-3, diverges


def test_complex_storage_is_refused_by_name(bf):
    fields, pml = _build(complex_storage=True)
    reasons = _curl_reasons(bf, fields, pml)
    assert any("complex family" in r for r in reasons), reasons


def test_bloch_k_is_refused_although_the_grid_allows_the_pairing(bf):
    """grid.py:719-724 DELIBERATELY accepts bfast + k_point (independent MEEP
    constructor slots); Phase A refuses the pairing by name — the composition
    is the complex family's queue."""
    grid = Grid(resolution=10.0, cell_size=(1.2, 1.0, 1.4), dimensions=3,
                courant=0.35, bfast_scaled_k=FULL_K, k_point=(0.25, 0.0, 0.0))
    assert grid.bfast_active is True  # the Grid took both slots
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_pml_storage()
    pml = PML(grid=grid, thickness={"z": 3})
    reasons = _curl_reasons(bf, fields, pml)
    assert any("complex family" in r or "k_point" in r for r in reasons), reasons
    assert any("not exactly zero" in r or "bfast+Bloch" in r or "queue" in r
               for r in reasons), reasons


def test_an_inactive_layer_is_refused(bf):
    fields, pml = _build(pml_thickness=0)
    reasons = _curl_reasons(bf, fields, pml)
    assert any("PML" in r for r in reasons), reasons


MIRROR_PML = {"x": {"high": 2}, "y": 2}  # a mirrored axis absorbs high-face only


def test_a_mirror_plane_is_refused(bf):
    fields, pml = _guard_2d_build(symmetry=("X",), pml_override=MIRROR_PML)
    reasons = _curl_reasons(bf, fields, pml)
    assert any("mirror" in r or "folded" in r for r in reasons), reasons


def test_beta_is_refused_with_the_fold_order_reason_in_both_directions(bf):
    """Clause 12 KEPT: the beta-then-bfast fold order (stepping.py:384-391
    before :392-396) is byte-significant. The special_kz family refuses bfast
    in return — no silent overlap in either direction."""
    fields, pml = _guard_2d_build(beta=0.25)
    assert float(fields.grid.beta) == 0.25
    reasons = _curl_reasons(bf, fields, pml)
    assert any("fold order" in r or "byte-significant" in r
               for r in reasons), reasons

    from meep_gpu.triton_kernels import special_kz

    theirs = special_kz.beta_pml_curl_coverage(fields, pml, "step_B").reasons
    assert any("BFAST" in r for r in theirs), theirs


def test_a_conductivity_is_refused_in_both_directions(bf):
    """The BFAST curl refuses conductive targets by name on BOTH sub-steps
    (strict, all six targets), and the conductive family's own predicate
    refuses bfast (conductivity.py:515-518) — no silent overlap."""
    fields, pml = _build()
    fields.set_d_conductivity(numpy.full(fields.grid.shape, 0.5,
                                         dtype=numpy.float32))
    for sub_step in ("step_B", "step_D"):
        reasons = _curl_reasons(bf, fields, pml, sub_step)
        assert any("conductivity" in r for r in reasons), (sub_step, reasons)

    from meep_gpu.triton_kernels import conductivity as conductive_module

    verdict = conductive_module.conductive_pml_curl_coverage(fields, pml,
                                                             "step_D")
    assert any("BFAST" in r for r in verdict.reasons), verdict.reasons


def test_a_nonlinearity_is_refused(bf, monkeypatch):
    fields, pml = _build()
    monkeypatch.setattr(Fields, "has_nonlinearity", property(lambda self: True))
    reasons = _curl_reasons(bf, fields, pml)
    assert any("chi2/chi3" in r for r in reasons), reasons


def test_cylindrical_bfast_is_unreachable_the_grid_itself_raises(bf):
    """grid.py:706-742: MEEP's own step_generic.cpp:376 drops the '- F[i]' its
    seven single-operand siblings carry, so Grid refuses the geometry at
    construction. The predicate KEEPS its own clause anyway (coverage
    doctrine: never infer a refusal from another module's guard)."""
    with pytest.raises(ValueError, match="cylindrical"):
        Grid(resolution=10.0, cell_size=(0.8, 0.0, 0.8), dimensions=2,
             cylindrical=True, bfast_scaled_k=(0.3, 0.0, 0.0))
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "cylindrical (Dcyl) coordinates are not carried" in source


def test_a_missing_state_is_refused_by_name(bf):
    """The array path RAISES on a missing state (stepping.py:916-921); the
    predicate refuses before any launch."""
    fields, pml = _build()
    fields.f_bfast_Bx = None
    reasons = _curl_reasons(bf, fields, pml, "step_B")
    assert any("f_bfast_Bx is not allocated" in r for r in reasons), reasons
    # The OTHER sub-step's states are out of this sub-step's scope (grouping
    # choice 10): step_D remains admitted.
    assert _curl_reasons(bf, fields, pml, "step_D") == []


def test_a_missing_condfac_reader_is_refused_not_read_as_no_conductivity(bf):
    fields, pml = _build()
    shim = SimpleNamespace(
        grid=fields.grid, force_complex_fields=False,
        has_nonlinearity=False, has_offdiagonal_epsilon=False,
        polarizations=(), stores_E=True,
        **{name: getattr(fields, name)
           for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz",
                        "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
                        "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz")
           },
        **{name: getattr(fields, name) for name in BFAST_STATE_ALL})
    reasons = [r for r in bf.bfast_pml_curl_coverage(shim, pml, "step_B").reasons
               if "array module" not in r]
    assert any("condfac_for" in r for r in reasons), reasons


def test_an_unknown_sub_step_or_side_raises_rather_than_refusing(bf):
    fields, pml = _build()
    with pytest.raises(ValueError, match="sub_step"):
        bf.bfast_pml_curl_coverage(fields, pml, "step_H")
    with pytest.raises(ValueError, match="side"):
        bf.bfast_run_constitutive_coverage(fields, pml, "B")


def test_the_shipped_families_keep_their_bfast_refusals(bf):
    """The no-silent-overlap fence: every shipped predicate that could be
    handed a BFAST run refuses it — CALLED, never grepped.

    A source grep for the token "BFAST" is escapable and, for one module,
    vacuous: ``dispersive_update_e.py`` mentions BFAST only in prose (it has no
    executable clause of its own — it inherits ``coverage._grid_reasons``), and
    deleting ``nonlinear_update_e``'s sole clause leaves the token in its
    docstring, so a grep passes while the predicate ADMITS the run. Every
    module below imports and answers on this Triton-less laptop, so the call is
    available and it is the contract."""
    from meep_gpu.triton_kernels import (complex_fields, coverage as shipped,
                                         dispersive_update_e, no_pml,
                                         nonlinear_update_e, offdiag_update_e)

    fields, pml = _build()
    verdict = shipped.pml_curl_coverage(fields, pml, "step_B")
    assert any("BFAST" in r for r in verdict.reasons), verdict.reasons
    assert any("BFAST" in r
               for r in shipped.constitutive_coverage(fields, pml, "H").reasons)

    ofields, opml = _build()
    ofields._chi1inv_offdiagonal = {"Ex": {"Ey": numpy.full(
        ofields.grid.shape, 0.1, dtype=numpy.float32)}}
    overdict = offdiag_update_e.offdiag_constitutive_coverage(ofields, opml)
    assert any("BFAST" in r for r in overdict.reasons), overdict.reasons

    complex_carrier, complex_pml = _build(complex_storage=True)
    called = {
        "no_pml.plain_curl_coverage":
            no_pml.plain_curl_coverage(fields, None, "step_B"),
        "complex_fields.complex_pml_curl_coverage":
            complex_fields.complex_pml_curl_coverage(
                complex_carrier, complex_pml, "step_B"),
        "complex_fields.complex_constitutive_coverage":
            complex_fields.complex_constitutive_coverage(
                complex_carrier, complex_pml, "H"),
        "nonlinear_update_e.nonlinear_constitutive_coverage":
            nonlinear_update_e.nonlinear_constitutive_coverage(fields, pml),
        # No clause of its own — it composes coverage._grid_reasons, which is
        # exactly why the grep this replaced pinned only a docstring word here.
        "dispersive_update_e.dispersive_constitutive_coverage":
            dispersive_update_e.dispersive_constitutive_coverage(fields, pml),
    }
    for label, answer in called.items():
        assert not answer.covered, f"{label} ADMITTED a BFAST run"
        assert any("BFAST" in r for r in answer.reasons), (label, answer.reasons)


def test_the_two_shipped_predicates_that_do_admit_a_bfast_run_are_pinned(bf):
    """The wiring note's stated EXCEPTIONS, measured rather than asserted away.

    ``coverage.ade_update_p_coverage`` and
    ``fused_ade_state.fused_ade_state_coverage`` never consult
    ``coverage._grid_reasons`` and carry no BFAST clause, so they admit a BFAST
    run. Harmless in substance — BFAST touches only the curl sub-step
    (fields.py:227-230) and this module builds no update_P plan — but the
    module docstring states the exception list, and this pins it so the list
    cannot silently grow."""
    from meep_gpu.dispersion import PolarizationState, Susceptibility
    from meep_gpu.triton_kernels import coverage as shipped
    from meep_gpu.triton_kernels import fused_ade_state

    fields, pml = _full_k_build()
    fields.enable_field_storage()
    shape = fields.grid.shape
    epsilon = numpy.full(shape, numpy.float32(2.25))
    fields.set_isotropic_epsilon_volume(epsilon, numpy.float32(1.0) / epsilon)
    volume = numpy.zeros(shape, dtype=numpy.float32)
    volume[1:-1, 1:-1, 1:-1] = numpy.float32(0.3)
    state = PolarizationState(
        Susceptibility(frequency=1.0, gamma=0.1, kind="lorentzian"),
        {name: (volume if name == "Ez" else 0.0)
         for name in ("Ex", "Ey", "Ez")},
        fields.grid, numpy.float32)
    fields.polarizations.append(state)
    assert fields.grid.bfast_active is True

    admitting = {
        "coverage.ade_update_p_coverage":
            shipped.ade_update_p_coverage(fields, state, "Ez"),
        "fused_ade_state.fused_ade_state_coverage":
            fused_ade_state.fused_ade_state_coverage(fields, state),
    }
    for label, answer in admitting.items():
        remaining = [r for r in answer.reasons if "not cupy" not in r]
        assert remaining == [], (
            f"{label} now refuses a BFAST run for {remaining}; the wiring "
            f"note's exception list in bfast_curl.py is stale — update it")
        assert not any("BFAST" in r for r in answer.reasons), label


# --- predicate mutations: each clause is load-bearing ------------------------

def _mutate(module, function, needle: str, replacement: str):
    import inspect  # noqa: PLC0415
    import textwrap  # noqa: PLC0415

    source = textwrap.dedent(inspect.getsource(function))
    mutated = source.replace(needle, replacement)
    assert mutated != source, f"the clause moved; update this mutation: {needle!r}"
    namespace = dict(module.__dict__)
    exec(compile(mutated, "<mutated bfast_curl>", "exec"), namespace)  # noqa: S102
    return namespace[function.__name__]


def test_m_dropping_the_beta_clause_is_caught(bf, monkeypatch):
    """Nothing else in this family refuses beta, so deleting clause 12 must
    flip a bfast+beta run to admitted — the clause is load-bearing."""
    fields, pml = _guard_2d_build(beta=0.25)
    assert _curl_reasons(bf, fields, pml) != []
    mutated = _mutate(bf, bf._bfast_grid_reasons,
                      'if float(getattr(grid, "beta", 0.0)) != 0.0:',
                      "if False:")
    monkeypatch.setattr(bf, "_bfast_grid_reasons", mutated)
    assert _curl_reasons(bf, fields, pml) == []


def test_m_dropping_the_conductivity_clause_is_caught(bf, monkeypatch):
    fields, pml = _build()
    fields.set_d_conductivity(numpy.full(fields.grid.shape, 0.5,
                                         dtype=numpy.float32))
    assert _curl_reasons(bf, fields, pml, "step_D") != []
    mutated = _mutate(bf, bf._curl_conductivity_reasons,
                      "if conductive:",
                      "if False:")
    monkeypatch.setattr(bf, "_curl_conductivity_reasons", mutated)
    assert _curl_reasons(bf, fields, pml, "step_D") == []


def test_m_widening_the_boundary_set_is_caught(bf, monkeypatch):
    """A folded axis resolves to 'mirror'; with the boundary clause widened
    the only remaining refusals are the explicit fold clauses — delete all
    three and the folded run is admitted, so each is load-bearing."""
    fields, pml = _guard_2d_build(symmetry=("X",), pml_override=MIRROR_PML)
    assert _curl_reasons(bf, fields, pml) != []
    import inspect  # noqa: PLC0415
    import textwrap  # noqa: PLC0415

    source = textwrap.dedent(inspect.getsource(bf._bfast_grid_reasons))
    for needle, replacement in (
            ("if kind not in COVERED_BOUNDARIES:", "if False:"),
            ('if _call(grid, "has_symmetry", default=False):', "if False:"),
            ('if _call(grid, "is_mirrored", axis, default=False):', "if False:")):
        assert needle in source, needle
        source = source.replace(needle, replacement)
    namespace = dict(bf.__dict__)
    exec(compile(source, "<mutated bfast_curl>", "exec"), namespace)  # noqa: S102
    monkeypatch.setattr(bf, "_bfast_grid_reasons",
                        namespace["_bfast_grid_reasons"])
    assert _curl_reasons(bf, fields, pml) == []


def test_m_dropping_the_invariant_axis_clause_is_caught(bf, monkeypatch):
    """Clause 11a is the only thing standing between this product and a
    configuration where stepping.py is not MEEP — delete it and the kz-on-an-
    invariant-axis run is admitted."""
    fields, pml = _guard_2d_build(bfast=(0.4, 0.0, 0.3))
    assert _curl_reasons(bf, fields, pml) != []
    mutated = _mutate(bf, bf._bfast_grid_reasons,
                      "if invariant and component != 0.0:",
                      "if False:")
    monkeypatch.setattr(bf, "_bfast_grid_reasons", mutated)
    assert _curl_reasons(bf, fields, pml) == []


def test_m_dropping_the_bfast_active_clause_is_caught(bf, monkeypatch):
    """Clause 11 INVERTED is what makes this a separate product: without it a
    k = 0 run would be admitted here AND by the shipped curl predicate, which
    is the admitted-overlap ambiguity the wiring note rules out."""
    from meep_gpu.triton_kernels import coverage as shipped

    fields, pml = _build(bfast=(0.0, 0.0, 0.0))
    assert fields.grid.bfast_active is False
    # The shipped curl predicate ADMITS this configuration (no BFAST clause
    # fires on it), so clause 11 is the whole of the separation.
    assert [r for r in shipped.pml_curl_coverage(fields, pml, "step_B").reasons
            if "array module" not in r] == []
    assert any("bfast_active is False" in r
               for r in _curl_reasons(bf, fields, pml))
    mutated = _mutate(bf, bf._bfast_grid_reasons,
                      'if not getattr(grid, "bfast_active", False):',
                      "if False:")
    monkeypatch.setattr(bf, "_bfast_grid_reasons", mutated)
    remaining = _curl_reasons(bf, fields, pml)
    assert not any("bfast_active" in r for r in remaining), remaining
    # What is left is only clause 13: Fields allocates no f_bfast_* on a k = 0
    # grid (fields.py:632-651). That is a second, incidental fence — it would
    # NOT stand on a grid whose states happened to exist, which is why clause
    # 11 is the load-bearing one.
    assert all("is not allocated" in r for r in remaining), remaining


def test_m_dropping_the_k_point_clauses_is_caught(bf, monkeypatch):
    """The k_point clause is load-bearing in the strongest sense: on the
    configuration it refuses, the ARRAY PATH itself raises rather than
    producing an answer, so dropping it would let this kernel return a
    plausible number where the engine refuses to produce one at all."""
    grid = Grid(resolution=10.0, cell_size=(1.2, 1.0, 1.4), dimensions=3,
                courant=0.35, bfast_scaled_k=FULL_K, k_point=(0.25, 0.0, 0.0))
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    pml = PML(grid=grid, thickness={"z": 3})
    reasons = _curl_reasons(bf, fields, pml)
    assert sum(1 for r in reasons if "k_point" in r) == 2, reasons
    with pytest.raises(ValueError, match="complex"):
        stepping.step_B(fields, pml)

    import inspect  # noqa: PLC0415
    import textwrap  # noqa: PLC0415

    source = textwrap.dedent(inspect.getsource(bf._bfast_grid_reasons))
    for needle in ('if getattr(grid, "has_bloch", False):',
                   "if any(float(component) != 0.0 for component in k_point):"):
        assert needle in source, needle
        source = source.replace(needle, "if False:")
    namespace = dict(bf.__dict__)
    exec(compile(source, "<mutated bfast_curl>", "exec"), namespace)  # noqa: S102
    monkeypatch.setattr(bf, "_bfast_grid_reasons",
                        namespace["_bfast_grid_reasons"])
    assert _curl_reasons(bf, fields, pml) == []


# ---------------------------------------------------------------------------
# The host coefficients — transcription pins
# ---------------------------------------------------------------------------

def test_the_cross_assignment_is_the_cross_product(bf):
    """The k-indexing trap, pinned on distinct components: k1 rides the
    FIRST's sum but is indexed by the SECOND's own axis (stepping.py:814-823,
    :911-912). For step_B with k = (kx, ky, kz), no invariance:
    t0 (Bx): (k1, k2) = (ky, kz); t1 (By): (kz, kx); t2 (Bz): (kx, ky)."""
    kx, ky, kz = FULL_K
    live = (False, False, False)
    ks = bf.bfast_curl_coefficients((kx, ky, kz), live, magnetic=True)
    f32 = numpy.float32
    assert ks == (float(f32(ky)), float(f32(kz)),
                  float(f32(kz)), float(f32(kx)),
                  float(f32(kx)), float(f32(ky)))


def test_the_d_side_negates_both_in_host_f64_then_rounds_once(bf):
    kx, ky, kz = FULL_K
    live = (False, False, False)
    b_side = bf.bfast_curl_coefficients((kx, ky, kz), live, magnetic=True)
    d_side = bf.bfast_curl_coefficients((kx, ky, kz), live, magnetic=False)
    f32 = numpy.float32
    for b_value, d_value in zip(b_side, d_side):
        # f64 negation then one f32 rounding == f32 rounding then negation:
        # the commute is exact, and both spellings must produce these bytes.
        assert f32(d_value) == f32(-b_value)
        assert f32(d_value).tobytes() == (-f32(b_value)).tobytes()


def test_the_invariance_guard_is_cross_gated(bf):
    """The m6 surface: at declared dims=2 (z invariant), By's k2 = kx on the
    Ez sum dies through have_p = not invariant(first_axis=z) — the CURL's own
    difference would NOT zero that term (the BFAST sum is 2g, stepping.py
    :889-898). Bz keeps its k1 = kx (both its axes live)."""
    kx = 0.4
    dims2 = (False, False, True)  # z invariant, the declared-2-D table
    ks = bf.bfast_curl_coefficients((kx, 0.0, 0.0), dims2, magnetic=True)
    f32 = numpy.float32
    # t0 (Bx): k1 = ky = 0 (gated by z anyway), k2 = kz = 0.
    assert ks[0] == 0.0 and ks[1] == 0.0
    # t1 (By): k1 = kz = 0; k2 = kx KILLED by have_p (first_axis = z).
    assert ks[2] == 0.0 and ks[3] == 0.0
    # t2 (Bz): k1 = kx survives (x and y both live), k2 = ky = 0.
    assert ks[4] == float(f32(kx)) and ks[5] == 0.0
    # Guard removed (all-live table): By's k2 = kx REAPPEARS — the byte
    # surface m6's needle case measures on device.
    unguarded = bf.bfast_curl_coefficients((kx, 0.0, 0.0),
                                           (False, False, False),
                                           magnetic=True)
    assert unguarded[3] == float(f32(kx))


def test_the_marquee_grid_gates_nothing(bf):
    """Grid.is_invariant is declared-dimensionality, never extent == 1
    (grid.py:1165-1183): the marquee's one-cell x/y axes stay LIVE at dims=3,
    so the physical By/Dz cross terms survive. A shape-based reimplementation
    kills them (the measured 48% refl error, from_meep.py:1936-1942)."""
    fields, _pml = _build()
    grid = fields.grid
    assert tuple(grid.shape[:2]) == (1, 1)
    invariant = tuple(grid.is_invariant(axis) for axis in range(3))
    assert invariant == (False, False, False)
    ks = bf.bfast_curl_coefficients(grid.bfast_scaled_k, invariant,
                                    magnetic=True)
    f32 = numpy.float32
    # t1 (By): k2 = kx on the Ez sum — ALIVE; t2 (Bz): k1 = kx — ALIVE.
    assert ks[3] == float(f32(KX_MARQUEE))
    assert ks[4] == float(f32(KX_MARQUEE))


def test_malformed_coefficient_inputs_raise(bf):
    with pytest.raises(ValueError, match="length-3"):
        bf.bfast_curl_coefficients((0.1, 0.2), (False, False, False), True)
    with pytest.raises(ValueError, match="length-3"):
        bf.bfast_curl_coefficients((0.1, 0.2, 0.3), (False,), True)


# ---------------------------------------------------------------------------
# The plans — shape contracts, no launches
# ---------------------------------------------------------------------------

def _curl_arrays(sub_step="step_B", shape=(4, 3, 5)):
    from meep_gpu.triton_kernels.launch import SUB_STEPS

    spec = SUB_STEPS[sub_step]
    rng = numpy.random.default_rng(20260812)
    names = (tuple(spec["targets"])
             + tuple("fu_" + name for name in spec["targets"])
             + tuple(spec["sources"])
             + tuple("f_bfast_" + name for name in spec["targets"]))
    arrays = {name: rng.uniform(-1, 1, shape).astype(numpy.float32)
              for name in names}
    flat = {f"{stem}_{axis}": numpy.linspace(0.5, 1.0, shape[i]
                                             ).astype(numpy.float32)
            for i, axis in enumerate("xyz") for stem in ("kms", "sinv")}
    return arrays, flat


def test_engine_route_plans_return_none_on_a_numpy_host(bf):
    fields, pml = _build()
    assert bf.plan_bfast_pml_curl(fields, pml, "step_B") is None
    assert bf.plan_bfast_run_constitutive(fields, pml, "H") is None


def test_the_plan_shape_from_arrays(bf):
    arrays, flat = _curl_arrays()
    ks = bf.bfast_curl_coefficients(FULL_K, (False, False, False), True)
    plan = bf.plan_bfast_pml_curl_from_arrays(
        "step_B", arrays, flat, (0, 1, 0), 0.35, ks)
    assert plan.sub_step == "step_B"
    assert plan.shape == (4, 3, 5)
    assert plan.n_elem == 60
    assert plan.backward == 0
    assert plan.bc == (0, 1, 0)
    assert plan.has_bfast == 1
    assert plan.ks == ks and len(plan.ks) == 6
    assert plan.block == bf.DEFAULT_BLOCK
    assert len(plan._targets) == len(plan._aux) == len(plan._sources) == 3
    assert len(plan._states) == 3
    assert len(plan._coefficients) == 6


def test_the_plan_binds_the_state_arrays_pointer_identically(bf):
    """The m9 surface: the driver's flux backup/restore (driver.py:4126-4135)
    only works if the kernel mutates fields.f_bfast_* IN PLACE — so the plan
    must bind the very arrays, never copies or views."""
    arrays, flat = _curl_arrays("step_D")
    ks = bf.bfast_curl_coefficients(FULL_K, (False, False, False), False)
    plan = bf.plan_bfast_pml_curl_from_arrays(
        "step_D", arrays, flat, (0, 0, 0), 0.35, ks)
    for pointer, name in zip(plan._states,
                             ("f_bfast_Dx", "f_bfast_Dy", "f_bfast_Dz")):
        assert pointer.array is arrays[name]
    for pointer, name in zip(plan._targets, ("Dx", "Dy", "Dz")):
        assert pointer.array is arrays[name]


def test_the_identity_arm_is_constructible(bf):
    arrays, flat = _curl_arrays()
    plan = bf.plan_bfast_pml_curl_from_arrays(
        "step_B", arrays, flat, (0, 1, 0), 0.35,
        (0.1, 0.2, 0.3, 0.4, 0.5, 0.6), has_bfast=0)
    assert plan.has_bfast == 0


def test_a_wrong_scalar_count_raises(bf):
    arrays, flat = _curl_arrays()
    with pytest.raises(ValueError, match="six"):
        bf.plan_bfast_pml_curl_from_arrays(
            "step_B", arrays, flat, (0, 1, 0), 0.35, (0.1, 0.2))


# ---------------------------------------------------------------------------
# The third transcription — pinned against stepping bytes, states included
# ---------------------------------------------------------------------------

B_TERMS = (("Bx", "Ez", 1, "Ey", 2, "y", "z"),
           ("By", "Ex", 2, "Ez", 0, "z", "x"),
           ("Bz", "Ey", 0, "Ex", 1, "x", "y"))
D_TERMS = (("Dx", "Hz", 1, "Hy", 2, "y", "z"),
           ("Dy", "Hx", 2, "Hz", 0, "z", "x"),
           ("Dz", "Hy", 0, "Hx", 1, "x", "y"))


def _face(axis, index):
    return tuple(index if a == axis else slice(None) for a in range(3))


def _shift(field, axis, kind, backward):
    shifted = numpy.roll(field, 1 if backward else -1, axis=axis)
    if kind == "metallic":
        shifted[_face(axis, 0 if backward else -1)] = 0
    return shifted


def _transcribed_sub_step(fields, pml, sub_step):
    """One BFAST curl sub-step on COPIES of the state, returned by name —
    stepping.py:923-931 inline: shared operands, cross-indexed k, host-f64 D
    negation, single f32 rounding at use, advance = total - 2*state, advance
    masked BEFORE the state write, curl folded as + (-advance)."""
    grid = fields.grid
    kinds = stepping._boundary_kinds(grid, pml)
    backward = sub_step == "step_D"
    magnetic = not backward
    terms = D_TERMS if backward else B_TERMS
    sources = {name: numpy.array(getattr(fields, name))
               for name in (("Hx", "Hy", "Hz") if backward
                            else ("Ex", "Ey", "Ez"))}
    state_out = {}
    dtdx = grid.dt / grid.dx
    suffix = "" if backward else "_h"
    bfast = grid.bfast_scaled_k
    invariant = tuple(grid.is_invariant(axis) for axis in range(3))
    for target, g1, a1, g2, a2, dsig, dsigu in terms:
        field = numpy.array(getattr(fields, target))
        fu = numpy.array(getattr(fields, "fu_" + target))
        shifted_first = _shift(sources[g1], a1, kinds[a1], backward)
        shifted_second = _shift(sources[g2], a2, kinds[a2], backward)
        curl = dtdx * ((shifted_first - sources[g1])
                       + (sources[g2] - shifted_second))
        if grid.bfast_active:
            have_p = not invariant[a1]
            have_m = not invariant[a2]
            k1 = bfast["xyz".index(g2[-1].lower())] if have_m else 0.0
            k2 = bfast["xyz".index(g1[-1].lower())] if have_p else 0.0
            if not magnetic:
                k1, k2 = -k1, -k2
            state = numpy.array(getattr(fields, "f_bfast_" + target))
            dtype = state.dtype
            total = (dtype.type(k1) * (shifted_first + sources[g1])
                     - dtype.type(k2) * (shifted_second + sources[g2]))
            advance = total - dtype.type(2.0) * state
            iyee = IYEE_SHIFTS[target]
            for axis in range(3):
                if iyee[axis] == 0 and kinds[axis] == "metallic":
                    advance[_face(axis, 0)] = 0
            state = state + advance
            curl = curl + (-advance)
            state_out["f_bfast_" + target] = state
        iyee = IYEE_SHIFTS[target]
        for axis in range(3):
            if iyee[axis] == 0 and kinds[axis] == "metallic":
                curl[_face(axis, 0)] = 0
        kms = getattr(pml, f"kms_{dsig}{suffix}")
        sinv = getattr(pml, f"sinv_{dsig}{suffix}")
        kms_u = getattr(pml, f"kms_{dsigu}{suffix}")
        sinv_u = getattr(pml, f"sinv_{dsigu}{suffix}")
        fu_previous = fu.copy()
        fu *= kms
        fu -= curl
        fu *= sinv
        field *= kms_u
        field += fu
        field -= fu_previous
        field *= sinv_u
        state_out[target] = field
        state_out["fu_" + target] = fu
    return state_out


def _seed(fields, seed=20260812):
    """Random fields, and states carrying signed zeros AND subnormals — the
    one auxiliary where subnormals persist indefinitely (the (-1)^n mode,
    stepping.py:868-874).

    The signed-zero plane goes on the first axis with extent > 1 — the gate's
    own rule (``_seed_host_real``). Pinning it to axis 2 unconditionally wipes
    the WHOLE array on a declared-2-D grid (nz == 1), which would leave every
    operand at -0.0 and make the guard case's k1/k2 multiply zeros: the
    invariance needle would then pin nothing but the -2*state layer."""
    rng = numpy.random.default_rng(seed)
    names = ("Bx", "By", "Bz", "Dx", "Dy", "Dz",
             "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
             "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz")
    shape = tuple(fields.grid.shape)
    live = [axis for axis in range(3) if shape[axis] > 1]
    for name in names:
        target = getattr(fields, name)
        host = rng.uniform(-1, 1, target.shape).astype(numpy.float32)
        if live:
            host[_face(live[0], 0)] = -0.0
        target[...] = host
    for name in BFAST_STATE_ALL:
        target = getattr(fields, name)
        host = rng.uniform(-1, 1, target.shape).astype(numpy.float32)
        flat = host.reshape(-1)
        flat[0::7] = numpy.float32(1e-45)   # subnormals: this family's hot FTZ
        flat[3::11] = numpy.float32(-3e-44)  # surface (marginally stable IIR)
        flat[5::13] = numpy.float32(-0.0)
        target[...] = host


REFERENCE_CASES = (
    ("marquee_z_only", _build, {}),
    ("full_k_3d", _full_k_build, {}),
    ("guard_2d", _guard_2d_build, {}),
    ("metallic_x", _full_k_build,
     dict(bfast=(0.4, 0.25, 0.0),
          boundaries=("metallic", "periodic", "periodic"))),
    # TWO masked axes. The D targets mask two axes each, and the synthetic,
    # subnormal and mutation legs all sweep metallic_xy while m8's needle IS
    # it — without this case that configuration would be validated only
    # against the kernel's own transcription, never against stepping.
    ("metallic_xy", _full_k_build,
     dict(bfast=(0.4, 0.25, 0.0),
          boundaries=("metallic", "metallic", "periodic"))),
)

#: DEGENERATE by construction, kept separate from REFERENCE_CASES: at
#: dimensions=1 the guard zeroes ALL SIX coefficients (see the dedicated tests
#: below), so the case pins the -2*state layer and the D1 arm of the guard but
#: cannot pin a live k. It still gets the byte transcription and non-vacuity
#: pins; it is excluded from the live-coefficient one.
DEGENERATE_CASES = (("dims1_z_only", _dims1_build, {}),)
TRANSCRIPTION_CASES = REFERENCE_CASES + DEGENERATE_CASES


@pytest.mark.parametrize("sub_step", ["step_B", "step_D"])
@pytest.mark.parametrize("name,builder,kwargs", TRANSCRIPTION_CASES,
                         ids=[c[0] for c in TRANSCRIPTION_CASES])
def test_the_bfast_sub_step_transcription_matches_stepping_bytes(
        bf, name, builder, kwargs, sub_step):
    """stepping.step_B/step_D on a real BFAST Fields against the in-test
    transcription, byte for byte, IIR STATES INCLUDED — the k cross-index,
    the invariance guard, the D negation, the single rounding, the 2*state
    advance, the pre-store mask and the fold sign all pinned at once."""
    fields, pml = builder(**kwargs)
    assert fields.grid.bfast_active
    _seed(fields)
    expected = _transcribed_sub_step(fields, pml, sub_step)
    getattr(stepping, sub_step)(fields, pml)
    for key, value in expected.items():
        got = numpy.asarray(getattr(fields, key))
        assert got.tobytes() == value.tobytes(), (
            f"{name}/{sub_step}: {key} diverged from the transcription")


@pytest.mark.parametrize("name,builder,kwargs", TRANSCRIPTION_CASES,
                         ids=[c[0] for c in TRANSCRIPTION_CASES])
def test_the_transcription_is_not_vacuous_against_a_bfast_free_run(
        bf, name, builder, kwargs):
    """EVERY reference case, not one: the same transcription run WITHOUT the
    tail must NOT match a BFAST stepping run, so each case's fold provably
    moves bytes on its own seeds. Run per case because a seeding rule that
    happens to zero one grid's operands (the nz == 1 trap _seed documents)
    makes exactly that case's pin vacuous while the others stay honest."""
    fields, pml = builder(**kwargs)
    _seed(fields)
    twin, twin_pml = builder(**kwargs)
    _seed(twin)
    expected = _transcribed_sub_step(fields, pml, "step_B")
    twin.grid.bfast_scaled_k = (0.0, 0.0, 0.0)  # fold skipped at :336
    stepping.step_B(twin, twin_pml)
    differing = sum(
        int(numpy.asarray(getattr(twin, key)).tobytes() != value.tobytes())
        for key, value in expected.items() if not key.startswith("f_bfast"))
    assert differing >= 4, (
        f"{name}: the BFAST fold moved no bytes on the seeded state; the "
        f"reference pins above would be vacuous")


@pytest.mark.parametrize("name,builder,kwargs", REFERENCE_CASES,
                         ids=[c[0] for c in REFERENCE_CASES])
def test_every_reference_case_carries_a_live_k_coefficient(bf, name, builder,
                                                           kwargs):
    """A case whose six (k1, k2) are all zero pins only the -2*state layer.
    At least one coefficient must survive the invariance guard, and the
    operands it multiplies must not be identically zero."""
    fields, pml = builder(**kwargs)
    _seed(fields)
    invariant = tuple(fields.grid.is_invariant(axis) for axis in range(3))
    ks = bf.bfast_curl_coefficients(fields.grid.bfast_scaled_k, invariant,
                                    magnetic=True)
    assert any(value != 0.0 for value in ks), (name, ks)
    assert any(float(numpy.max(numpy.abs(numpy.asarray(getattr(fields, s)))))
               > 0.0 for s in ("Ex", "Ey", "Ez")), (
        f"{name}: every curl source is identically zero, so the live "
        f"coefficient multiplies nothing")


def test_a_zero_k_target_still_advances_its_state(bf):
    """The always-run-all-six binding (grid.py:744-753), byte-visible only
    with seeded state: on the marquee (kx only), Bx's (k1, k2) are both zero,
    yet its advance is -2*state — after one step_B the state is EXACTLY the
    seeded state negated (0 - 2s added to s), and Bx's bytes moved."""
    fields, pml = _build()
    _seed(fields)
    seeded = numpy.array(fields.f_bfast_Bx)
    before_bx = numpy.array(fields.Bx)
    stepping.step_B(fields, pml)
    assert numpy.asarray(fields.f_bfast_Bx).tobytes() == (-seeded).tobytes(), (
        "the zero-k target's state did not take the -2*state advance; a "
        "kernel that skips zero-k targets would be byte-wrong exactly here")
    assert numpy.asarray(fields.Bx).tobytes() != before_bx.tobytes()


def test_the_d1_arm_of_the_guard_gates_every_coefficient_to_zero(bf):
    """dimensions=1 makes x AND y invariant, so four of the six flags are
    false at once and — for the only k clause 11a admits there, k along z —
    every one of the six scalars dies:

    Bx: k1 = k_y (gated live) = 0; k2 = k_z KILLED by have_p (first_axis = y).
    By: k1 = k_z KILLED by have_m (second_axis = x); k2 = k_x = 0.
    Bz: both axes invariant, both flags false.

    That is what makes the D1 arm degenerate rather than merely reduced."""
    fields, _pml = _dims1_build()
    grid = fields.grid
    invariant = tuple(grid.is_invariant(axis) for axis in range(3))
    assert invariant == (True, True, False)
    for magnetic in (True, False):
        ks = bf.bfast_curl_coefficients(grid.bfast_scaled_k, invariant,
                                        magnetic=magnetic)
        assert ks == (0.0,) * 6, (magnetic, ks)
    # Without the guard the k_z terms come back on By and Dy — the guard is
    # doing real work here, it is not vacuously zero.
    unguarded = bf.bfast_curl_coefficients(grid.bfast_scaled_k,
                                           (False, False, False), magnetic=True)
    assert unguarded[2] == float(numpy.float32(grid.bfast_scaled_k[2]))


def test_the_d1_arm_from_a_zero_state_is_exactly_a_non_bfast_run(bf):
    """The D1 MEEP-fidelity statement, measured.

    MEEP's OTHER reason for a false flag is ``gv.has_field``: in D1 it leaves
    Bx/Bz/Dy/Dz unallocated, so ``step_db``'s ``if (f[cc][cmp])`` never steps
    them, where this engine steps all six. That asymmetry is invisible in a
    RUN, because with every coefficient zero (the test above) the advance is
    ``-2*state`` and the physical state starts at zero: the states stay
    exactly zero and every component is byte-identical to the same run with
    BFAST off. Only an artificially seeded state moves them, which is a
    harness configuration and not a simulation."""
    fields, pml = _dims1_build()
    twin, twin_pml = _dims1_build(bfast=(0.0, 0.0, 0.0))
    assert fields.grid.bfast_active is True
    assert twin.grid.bfast_active is False
    assert float(fields.grid.dt) == float(twin.grid.dt), (
        "BFAST must not move the timestep (grid.py:719-724)")
    rng = numpy.random.default_rng(20260812)
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz",
                 "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
                 "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz"):
        host = rng.uniform(-1, 1, fields.grid.shape).astype(numpy.float32)
        getattr(fields, name)[...] = host
        getattr(twin, name)[...] = host
    for name in BFAST_STATE_ALL:  # the PHYSICAL initial condition: exactly zero
        getattr(fields, name)[...] = numpy.float32(0.0)

    for _ in range(2):
        for sub_step in ("step_B", "update_H", "step_D", "update_E"):
            getattr(stepping, sub_step)(fields, pml)
            getattr(stepping, sub_step)(twin, twin_pml)
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz",
                 "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
                 "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz"):
        assert (numpy.asarray(getattr(fields, name)).tobytes()
                == numpy.asarray(getattr(twin, name)).tobytes()), name
    for name in BFAST_STATE_ALL:
        state = numpy.asarray(getattr(fields, name))
        assert state.tobytes() == numpy.zeros_like(state).tobytes(), name


def test_the_state_write_is_in_place_in_the_array_path(bf):
    """stepping.py:930 is ``state += advance`` on the very array — the
    invariant the kernel must reproduce (driver sync depends on it)."""
    fields, pml = _build()
    _seed(fields)
    held = fields.f_bfast_By
    before = held.tobytes()
    stepping.step_B(fields, pml)
    assert fields.f_bfast_By is held
    assert held.tobytes() != before


# ---------------------------------------------------------------------------
# Kernel source facts the gate relies on
# ---------------------------------------------------------------------------

def _segment(name: str, path=None) -> str:
    source = (path or MODULE_PATH).read_text(encoding="utf-8")
    start = source.index(f"def {name}(")
    ends = [position for position in (source.find("\n@triton.jit", start + 1),
                                      source.find("\n# ----", start + 1),
                                      source.find("\ndef ", start + 1))
            if position != -1]
    return source[start:min(ends) if ends else len(source)]


def _kernel_code_lines(segment: str):
    """A kernel body as stripped code lines: signature, docstring, comments and
    blank lines removed, EVERY parenthesis kept.

    Deliberately textual. An AST comparison would be worse than useless here —
    ``ast.unparse`` drops redundant parentheses, so it cannot tell
    ``dtdx * ((c_y - c) + (b - b_z))`` from a reassociated spelling, and
    reassociation is precisely the byte-moving drift this pin exists for."""
    lines = segment.splitlines()
    start = next(index for index, line in enumerate(lines)
                 if line.rstrip() == "):") + 1
    body = lines[start:]
    if body and body[0].lstrip().startswith('"""'):
        head = body[0].lstrip()
        if head.count('"""') >= 2:
            body = body[1:]
        else:
            close = next(index for index in range(1, len(body))
                         if '"""' in body[index])
            body = body[close + 1:]
    return [line.strip() for line in body
            if line.strip() and not line.strip().startswith("#")]


def _tail_block() -> str:
    segment = _segment("bfast_pml_curl_step")
    block = segment[segment.index("if HAS_BFAST:"):]
    return block[:block.index("# --- ownership mask (stepping._mask_non_owned_cells)")]


def test_the_tail_sits_after_the_curl_and_before_the_mask():
    segment = _segment("bfast_pml_curl_step")
    curl = segment.index("curl0 = dtdx * ((c_y - c)")
    tail = segment.index("if HAS_BFAST:")
    mask = segment.index("# --- ownership mask (stepping._mask_non_owned_cells)")
    assert curl < tail < mask, (
        "the BFAST fold must sit between the curl and the ownership mask, "
        "where the array path adds it (stepping.py:392-396/:475-478)")


def test_the_tail_pairs_shifted_with_center_in_operand_order():
    block = _tail_block()
    assert "total0 = (k1_0 * (c_y + c)) - (k2_0 * (b_z + b))" in block
    assert "total1 = (k1_1 * (a_z + a)) - (k2_1 * (c_x + c))" in block
    assert "total2 = (k1_2 * (b_x + b)) - (k2_2 * (a_y + a))" in block


def test_the_advance_is_masked_before_the_state_store():
    block = _tail_block()
    first_store = block.index("tl.store(s0 + idx")
    mask_marker = block.index("advance ownership mask")
    assert mask_marker < first_store, (
        "stepping.py:929 masks the advance BEFORE the state write; the store "
        "must come after the mask block")
    assert "tl.store(s0 + idx, st0 + adv0, mask=live)" in block
    assert "tl.store(s1 + idx, st1 + adv1, mask=live)" in block
    assert "tl.store(s2 + idx, st2 + adv2, mask=live)" in block


def test_the_fold_and_advance_avoid_unary_minus_and_dtdx():
    """Platform fact: Triton lowers -x as 0.0 - x, canonicalizing signed-zero
    addends; the tail must spell every negation as IEEE subtraction. And the
    Tustin filter carries NO dtdx (stepping.py:863-864)."""
    block = _tail_block()
    assert "curl0 = curl0 - adv0" in block
    assert "curl1 = curl1 - adv1" in block
    assert "curl2 = curl2 - adv2" in block
    assert "adv0 = total0 - (2.0 * st0)" in block
    assert "dtdx" not in block
    for spelling in (" -adv", "(-adv", " -st", "(-st", " -total", "(-total"):
        assert spelling not in block, (
            f"unary minus spelled {spelling!r} in the BFAST tail; Triton's "
            f"0.0 - x lowering canonicalizes signed zeros")


def test_all_three_targets_always_run_the_tail():
    """grid.py:744-753: ANY nonzero component runs the pass for EVERY
    component. One constexpr gates the whole tail; no per-target arm."""
    segment = _segment("bfast_pml_curl_step")
    assert segment.count("if HAS_BFAST:") == 1
    block = _tail_block()
    for index in range(3):
        assert f"total{index} =" in block
        assert f"adv{index} = total{index}" in block
        assert f"tl.store(s{index} + idx" in block


def test_the_certified_body_is_carried_verbatim():
    """EVERY certified statement, in order — not a six-line spot check.

    The identity leg (HAS_BFAST=0) pins the bytes on device for ONE boundary
    triple, one shape and one dtdx; the synthetic sweep covers the rest but
    only on the GPU host. This pins the whole shared text at merge: every code line
    of ``kernels.pml_curl_step``'s body must appear in
    ``bfast_pml_curl_step``'s body, in the same order (the tail is an
    insertion, and ``at_x, at_y, at_z`` is hoisted above it — a subsequence,
    never a reordering)."""
    certified = _kernel_code_lines(
        _segment("pml_curl_step", PACKAGE_DIR / "kernels.py"))
    mine = _kernel_code_lines(_segment("bfast_pml_curl_step"))
    assert len(certified) >= 60, (
        f"only {len(certified)} certified body lines were extracted; the "
        f"segment/docstring parsing drifted and this pin went hollow")

    missing = [line for line in certified if line not in mine]
    assert missing == [], (
        f"{len(missing)} certified statement(s) are no longer carried "
        f"verbatim: {missing[:5]}")

    cursor = 0
    for line in certified:
        cursor = mine.index(line, cursor) + 1  # raises if the order drifted

    # The BFAST tail is the ONLY addition: the surplus, counted as a multiset
    # so the tail's replicated ownership predicate is not silently forgiven.
    import collections  # noqa: PLC0415

    surplus = collections.Counter(mine) - collections.Counter(certified)
    tail_lines = [line.strip() for line in _tail_block().splitlines()
                  if line.strip() and not line.strip().startswith("#")]
    assert sum(surplus.values()) == len(tail_lines) == 36, (
        f"the body carries {sum(surplus.values())} lines the certified kernel "
        f"does not, but the BFAST tail is {len(tail_lines)} lines — something "
        f"outside the tail was added or removed")
    allowed = ("HAS_BFAST", "adv", "total", "st0", "st1", "st2",
               "s0 + idx", "s1 + idx", "s2 + idx")
    # The tail replicates the body's ownership predicate, so its `if BACKWARD:`
    # / `else:` / `if BC? == METALLIC:` lines legitimately recur.
    replicated = ("if BACKWARD:", "else:", "if BCX == METALLIC:",
                  "if BCY == METALLIC:", "if BCZ == METALLIC:")
    for line in surplus:
        assert any(token in line for token in allowed) or line in replicated, (
            f"a non-BFAST line appeared that the certified kernel does not "
            f"carry: {line!r}")


# ---------------------------------------------------------------------------
# The gate trio exists and holds the run discipline
# ---------------------------------------------------------------------------

def test_the_gate_and_its_composition_probe_exist():
    assert GATE_PATH.exists()
    assert COMPOSITION_PATH.exists()


def test_both_gate_files_install_the_strip_before_any_compile():
    for path in (GATE_PATH, COMPOSITION_PATH):
        text = path.read_text(encoding="utf-8")
        assert "install_ftz_strip" in text, path


def test_the_composition_probes_stated_budget_matches_its_case_table():
    """The stated step budget IS the certification's scope statement, so a
    prose number that drifts off the table is a scope overclaim."""
    text = COMPOSITION_PATH.read_text(encoding="utf-8")
    steps = [int(value) for value in re.findall(r'"steps": (\d+)', text)]
    assert len(steps) == 7, steps
    assert f"{sum(steps)} complete steps" in text, (
        f"the docstring's step count is not the case table's {sum(steps)}")


def test_the_gate_does_not_claim_to_hold_the_byte_invisible_choices():
    """f32 cannot see the tail's POSITION (the advance is masked with the same
    predicate, so mask(curl) - mask(adv) == mask(curl - adv)) or the sum's
    operand ORDER (IEEE addition commutes). Those are held by the source
    assertions above; the gate must record them as predicted nulls rather than
    list them among what it holds empirically."""
    text = GATE_PATH.read_text(encoding="utf-8")
    held = text[text.index("WHAT THIS GATE HOLDS"):text.index("Legs, in order")]
    invisible = held[held.index("NOT held here"):]
    for phrase in ("POSITION between curl and mask", "sum ORDER"):
        assert phrase in invisible, phrase
    for key in ("tail_before_the_mask_vs_tail_after_the_mask",
                "shifted_plus_center_vs_center_plus_shifted",
                "curl_minus_adv_vs_curl_plus_negated_adv"):
        assert f'"{key}"' in text, (
            f"{key} is no longer recorded in the gate's predicted_nulls; the "
            f"artifact would imply it was measured")


def test_the_metallic_restricted_mutation_excludes_the_all_periodic_combos():
    """m8's restriction must reach GUARD_CONFIGS, whose single entry declares
    all-periodic boundaries: with no masked row the mutant is byte-identical
    BY CONSTRUCTION there, which is a false miss, not a predicted null."""
    text = GATE_PATH.read_text(encoding="utf-8")
    assert ('guard_configs = () if restriction == "metallic" '
            "else GUARD_CONFIGS") in text
    assert "guard_configs=guard_configs" in text
