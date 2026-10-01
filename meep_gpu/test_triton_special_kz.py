"""Laptop contracts for the special_kz (grid.beta) Triton tranche.

Everything here runs on the NumPy merge-bar machine: the optional-import
contract, the predicates' admission AND per-clause refusals on real
Grid/Fields/PML objects, the plan builders' shapes, the host coefficient
transcription pinned BYTE-for-byte against ``stepping._special_kz_beta_term``,
and a third, in-test transcription of the beta curl sub-steps pinned against
``stepping.step_B``/``step_D`` themselves. The kernels' device bytes are the
gate's (``parity/meep_gpu/gate_triton_special_kz.py``); nothing here launches.
"""

from __future__ import annotations

import importlib
import math
import pathlib
import sys
from types import SimpleNamespace

import numpy
import pytest

from meep_gpu import stepping
from meep_gpu.fields import IYEE_SHIFTS, Fields
from meep_gpu.grid import Grid
from meep_gpu.pml import PML
from meep_gpu.test_triton_kernels import GUARD_SPELLING, PACKAGE_DIR, code_of
from meep_gpu.test_triton_complex_fields import (

    AMBIGUOUS,
    set_contradictory_ambiguity,
    set_pattern,
    stamp_probe_record,
)

# Every probe record this module builds is stamped 'keep'; the complex
# families' policy clause fails closed when the run policy can be neither
# read nor declared, so the premise is declared rather than left implicit.
# See ``run_policy_declared_keep`` in conftest.py.
pytestmark = pytest.mark.usefixtures("run_policy_declared_keep")

MODULE_PATH = PACKAGE_DIR / "special_kz.py"
MODULE_NAME = "meep_gpu.triton_kernels.special_kz"
PARITY_DIR = pathlib.Path(__file__).resolve().parents[1] / "parity" / "meep_gpu"
GATE_PATH = PARITY_DIR / "gate_triton_special_kz.py"
COMPOSITION_PATH = PARITY_DIR / "probe_triton_special_kz_composition.py"
SLURM_PATH = PARITY_DIR / "run_triton_special_kz.slurm"

#: The corpus anchors' own numbers, used throughout so the tests exercise the
#: values the unlock claim is about.
BETA_KZ2D = 0.3321611318837033          # refl-angular-kz2d.py
BETA_GRATING = -0.6850526103319672      # test_binary_grating_special_kz_0_13_2
BETA_SPECIAL_KZ = -0.39073112848927377  # test_special_kz.test_special_kz
KX_SPECIAL_KZ = 0.9205048534524404


@pytest.fixture(scope="module")
def sk():
    """The shipped module itself must be importable without a test-only stub."""
    return importlib.import_module(MODULE_NAME)


def _probe_record(value: str = "FMA_V1", backend: str = "cupy",
                  include_beta_pattern: bool = True):
    module = importlib.import_module(MODULE_NAME)
    names = (module.BETA_PROBE_PATTERNS if include_beta_pattern
             else module.PROBE_PATTERNS)
    return stamp_probe_record(
        {"backend": backend, "patterns": {name: value for name in names}})


def _build(cell_size=(2.0, 1.6, 0.0), pml_thickness=2, complex_storage=False,
           storage=True, k_point=(0.0, 0.0, 0.0), beta=BETA_KZ2D,
           courant=0.35, pml_override=None, **grid_kwargs):
    """A real 2-D beta Grid/Fields/PML triple on NumPy.

    Default: the real family's own target class — real storage, an active PML,
    beta = refl-angular-kz2d's 0.332, k = 0. Every refusal test perturbs
    exactly one clause off this (or off the complex variant of it).
    """
    grid = Grid(resolution=10.0, cell_size=cell_size, dimensions=2,
                courant=courant, k_point=k_point, beta=beta, **grid_kwargs)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    if storage:
        fields.enable_pml_storage()
    # Per-axis thickness: the invariant z axis (n = 1) carries no layer at all,
    # exactly as the complex tranche's reference builder builds it.
    if pml_override is not None:
        return fields, PML(grid=grid, thickness=pml_override)
    thickness = tuple(
        (pml_thickness, pml_thickness) if grid.shape[axis] >= 6 else (0, 0)
        for axis in range(3))
    return fields, PML(grid=grid, thickness=thickness)


def _real_reasons(sk, fields, pml, sub_step="step_B"):
    verdict = sk.beta_pml_curl_coverage(fields, pml, sub_step)
    return [r for r in verdict.reasons if "array module" not in r]


def _complex_reasons(sk, fields, pml, sub_step="step_B", probe="default"):
    record = _probe_record() if probe == "default" else probe
    verdict = sk.beta_bloch_pml_curl_coverage(fields, pml, sub_step, probe=record)
    return [r for r in verdict.reasons if "array module" not in r]


def _constitutive_reasons(sk, fields, pml, side="E"):
    verdict = sk.beta_run_constitutive_coverage(fields, pml, side)
    return [r for r in verdict.reasons if "array module" not in r]


# ---------------------------------------------------------------------------
# The optional dependency must stay optional; the restatements must not drift
# ---------------------------------------------------------------------------

def test_the_package_does_not_import_special_kz_eagerly():
    for name in list(sys.modules):
        if name == MODULE_NAME:
            del sys.modules[name]
    importlib.import_module("meep_gpu.triton_kernels")
    eagerly_imported = MODULE_NAME in sys.modules
    importlib.import_module(MODULE_NAME)  # restore for the other tests
    assert not eagerly_imported, (
        "the package __init__ must not pull this unwired module in")


def test_the_module_answers_coverage_but_kernels_fail_clearly_without_triton(
        sk, monkeypatch):
    if not isinstance(sk.beta_pml_curl_step, sk._UnavailableKernel):
        pytest.skip("triton is installed here; the absence arm is exercised on "
                    "the CI/laptop box without it")
    fields, pml = _build()
    assert sk.beta_pml_curl_coverage(fields, pml, "step_B").reasons
    with pytest.raises(ImportError, match="triton"):
        sk.beta_pml_curl_step[(1,)]
    with pytest.raises(ImportError, match="triton"):
        sk.beta_bloch_pml_curl_step[(1,)]


def test_the_restated_constants_pin_their_originators(sk):
    from meep_gpu.triton_kernels import complex_fields

    def value(x):
        return getattr(x, "value", x)

    assert value(sk.PERIODIC) == 0 and value(sk.METALLIC) == 1
    assert value(sk.NAIVE) == value(complex_fields.NAIVE)
    assert value(sk.FMA_V1) == value(complex_fields.FMA_V1)
    assert sk.DEFAULT_BLOCK == complex_fields.DEFAULT_BLOCK == 256
    # Read kernels.py off the source (importing it needs Triton).
    kernels_source = (PACKAGE_DIR / "kernels.py").read_text(encoding="utf-8")
    assert "\nPERIODIC = tl.constexpr(0)\n" in kernels_source
    assert "\nMETALLIC = tl.constexpr(1)\n" in kernels_source
    assert "\nDEFAULT_BLOCK = 256\n" in kernels_source


def test_the_extended_probe_pattern_set_is_a_strict_superset(sk):
    from meep_gpu.triton_kernels import complex_fields

    assert set(complex_fields.PROBE_PATTERNS) < set(sk.BETA_PROBE_PATTERNS)
    assert sk.BETA_PROBE_PATTERN in sk.BETA_PROBE_PATTERNS
    assert sk.BETA_PROBE_PATTERN not in complex_fields.PROBE_PATTERNS


def test_the_module_docstring_states_dispatch_is_untouched(sk):
    assert "NOT WIRED" in sk.__doc__
    assert "plan_fast_path" in sk.__doc__


def test_the_module_does_not_reach_into_another_tracks_file():
    code = code_of(MODULE_PATH)
    assert "fastpath" not in code
    assert "cuda_kernels" not in code


def test_the_guard_is_the_packages_one_spelling_and_appears_twice():
    code = MODULE_PATH.read_text(encoding="utf-8")
    assert code.count("enable_fp_fusion=") == 2  # the two plan run() methods
    assert code.count(GUARD_SPELLING) == 2


def test_no_fingerprint_entry_is_claimed(sk):
    import json

    recorded = json.loads((PACKAGE_DIR / "fingerprints.json")
                          .read_text(encoding="utf-8"))
    assert not any("special_kz" in name for name in recorded), (
        "this tranche's provenance lives in the gate's results directory, "
        "not in the other track's fingerprints.json")


# ---------------------------------------------------------------------------
# The positive verdicts — refused only for the backend on this laptop
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sub_step", ["step_B", "step_D"])
def test_the_real_beta_configuration_is_refused_only_for_the_backend(sk, sub_step):
    """The real family's target class: refl-angular-kz2d's real/imag leg."""
    fields, pml = _build()
    assert _real_reasons(sk, fields, pml, sub_step) == []


@pytest.mark.parametrize("sub_step", ["step_B", "step_D"])
def test_the_complex_beta_configuration_is_refused_only_for_the_backend(sk, sub_step):
    """C1's class: beta != 0, k = 0, complex storage, PML both axes."""
    fields, pml = _build(complex_storage=True, beta=0.2)
    assert _complex_reasons(sk, fields, pml, sub_step) == []


def test_the_marquee_inplane_bloch_beta_configuration_is_admitted(sk):
    """test_special_kz's own numbers: kx = 0.9205 in-plane + beta = -0.3907."""
    fields, pml = _build(complex_storage=True, beta=BETA_SPECIAL_KZ,
                         k_point=(KX_SPECIAL_KZ, 0.0, 0.0))
    assert fields.grid.has_bloch is True
    assert _complex_reasons(sk, fields, pml) == []


@pytest.mark.parametrize("side", ["H", "E"])
def test_constitutive_sides_are_admitted_on_a_beta_run(sk, side):
    """The restated shipped predicate minus the beta clause, both families."""
    fields, pml = _build()
    assert _constitutive_reasons(sk, fields, pml, side) == []
    cfields, cpml = _build(complex_storage=True, beta=0.2)
    verdict = sk.beta_run_complex_constitutive_coverage(
        cfields, cpml, side, probe=_probe_record())
    assert [r for r in verdict.reasons if "array module" not in r] == []


def test_the_complex_constitutive_binds_the_base_probe_contract(sk):
    """Grouping choice 5: the certified constitutive kernel's contract is the
    BASE pattern set, so a record cut before this tranche admits it — while the
    beta CURL refuses the same record by name (§4(k))."""
    base_only = _probe_record(include_beta_pattern=False)
    fields, pml = _build(complex_storage=True, beta=0.2)
    verdict = sk.beta_run_complex_constitutive_coverage(
        fields, pml, "E", probe=base_only)
    assert [r for r in verdict.reasons if "array module" not in r] == []
    curl = _complex_reasons(sk, fields, pml, probe=base_only)
    assert any(sk.BETA_PROBE_PATTERN in r for r in curl), curl


# ---------------------------------------------------------------------------
# The refusals, one per silent-wrong-answer surface
# ---------------------------------------------------------------------------

def test_beta_zero_is_refused_by_both_families(sk):
    """The INVERTED clause 12: a beta = 0 run belongs to the certified kernels
    (and the array path never enters the term, stepping.py:384/:467)."""
    fields, pml = _build(beta=0.0)
    reasons = _real_reasons(sk, fields, pml)
    assert any("beta is zero" in r for r in reasons), reasons
    cfields, cpml = _build(complex_storage=True, beta=0.0)
    creasons = _complex_reasons(sk, cfields, cpml)
    assert any("beta is zero" in r for r in creasons), creasons
    assert any("beta is zero" in r
               for r in _constitutive_reasons(sk, fields, pml, "E"))


def test_the_two_storage_families_refuse_each_others_runs(sk):
    fields, pml = _build(complex_storage=True, beta=0.2)
    reasons = _real_reasons(sk, fields, pml)
    assert any("complex beta variant" in r for r in reasons), reasons
    rfields, rpml = _build()
    creasons = _complex_reasons(sk, rfields, rpml)
    assert any("real beta kernel" in r for r in creasons), creasons


def test_an_inactive_layer_is_refused(sk):
    fields, pml = _build(pml_thickness=0)
    reasons = _real_reasons(sk, fields, pml)
    assert any("PML" in r for r in reasons), reasons


def test_real_offdiagonal_epsilon_with_beta_is_refused_by_name(sk):
    """§4(f): the implicit-i trick stops cancelling — stepping raises
    (stepping.py:800-810), MEEP aborts (fields.cpp:548-549). Refused for the
    whole real family, and the raise itself is pinned below."""
    fields, pml = _build()
    fields._chi1inv_offdiagonal = {"Ex": {"Ey": numpy.full(
        fields.grid.shape, 0.1, dtype=numpy.float32)}}
    assert fields.has_offdiagonal_epsilon is True
    for sub_step in ("step_B", "step_D"):
        reasons = _real_reasons(sk, fields, pml, sub_step)
        assert any("implicit-i" in r for r in reasons), reasons


def test_stepping_itself_raises_on_real_offdiag_beta(sk):
    """The clause's referent, exercised: the array path's own refusal."""
    grid = SimpleNamespace(beta=0.25, dt=0.05, xp=numpy)
    shim = SimpleNamespace(grid=grid, has_offdiagonal_epsilon=True)
    partner = numpy.ones((3, 3, 1), dtype=numpy.float32)
    with pytest.raises(ValueError, match="off-diagonal"):
        stepping._special_kz_beta_term(shim, partner, +1.0, magnetic=True)


MIRROR_PML = {"x": {"high": 2}, "y": 2}  # a mirrored axis absorbs high-face only


def test_a_mirror_plane_is_refused_with_the_phase_b_pointer(sk):
    """Phase A refuses the fold by name; the reason must say where it goes."""
    fields, pml = _build(symmetry=("X",), beta=0.2, pml_override=MIRROR_PML)
    reasons = _real_reasons(sk, fields, pml)
    assert any("Phase B" in r for r in reasons), reasons
    cfields, cpml = _build(symmetry=("X",), beta=0.2, complex_storage=True,
                           pml_override=MIRROR_PML)
    creasons = _complex_reasons(sk, cfields, cpml)
    assert any("complex tranche" in r and "fold" in r.lower()
               for r in creasons), creasons


def test_beta_on_a_3d_grid_is_refused_by_the_grid_itself(sk):
    """§4(d)/(e): grid.py:668-697, MEEP fields.cpp:546-547 — restated in the
    predicate AND exercised at the constructor."""
    with pytest.raises(ValueError, match="2-D"):
        Grid(resolution=10.0, cell_size=(0.8, 0.8, 0.8), dimensions=3, beta=0.2)


def test_beta_on_a_cylindrical_grid_is_refused_by_the_grid_itself(sk):
    with pytest.raises(ValueError, match="cylindrical"):
        Grid(resolution=10.0, cell_size=(0.8, 0.0, 0.8), dimensions=2,
             cylindrical=True, beta=0.2)


def test_a_non_2d_grid_is_refused_by_the_predicate_not_inferred(sk):
    """The restated clause fires even when the constructor guard was bypassed."""
    fields, pml = _build()
    fields.grid.dimensions = 3  # planted past validation
    reasons = _real_reasons(sk, fields, pml)
    assert any("effective-2-D" in r for r in reasons), reasons


def test_bfast_is_refused(sk):
    fields, pml = _build()
    fields.grid.bfast_scaled_k = (0.0, 0.0, 0.5)
    if fields.grid.bfast_active:
        reasons = _real_reasons(sk, fields, pml)
        assert any("BFAST" in r for r in reasons), reasons


def test_a_nonlinearity_is_refused(sk, monkeypatch):
    fields, pml = _build()
    monkeypatch.setattr(Fields, "has_nonlinearity", property(lambda self: True))
    reasons = _real_reasons(sk, fields, pml)
    assert any("chi2/chi3" in r for r in reasons), reasons


def test_a_conductivity_is_refused_in_both_directions(sk):
    """§4 / grouping choice 7: the beta curls refuse conductive targets by
    name, and the conductive family's own predicate refuses beta — no silent
    overlap in either direction."""
    fields, pml = _build()
    fields.set_d_conductivity(numpy.full(fields.grid.shape, 0.5,
                                         dtype=numpy.float32))
    reasons = _real_reasons(sk, fields, pml, "step_D")
    assert any("conductivity" in r for r in reasons), reasons

    from meep_gpu.triton_kernels import conductivity as conductive_module

    source = (PACKAGE_DIR / "conductivity.py").read_text(encoding="utf-8")
    assert "special_kz beta" in source, (
        "conductivity.py no longer refuses beta by name; the cross-refusal "
        "pairing this module relies on has moved")
    assert conductive_module is not None


def test_a_missing_condfac_reader_is_refused_not_read_as_no_conductivity(sk):
    fields, pml = _build()
    shim = SimpleNamespace(
        grid=fields.grid, force_complex_fields=False,
        has_nonlinearity=False, has_offdiagonal_epsilon=False,
        polarizations=(), stores_E=True,
        **{name: getattr(fields, name)
           for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz",
                        "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
                        "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz")})
    reasons = [r for r in sk.beta_pml_curl_coverage(shim, pml, "step_B").reasons
               if "array module" not in r]
    assert any("condfac_for" in r for r in reasons), reasons


def test_a_registered_susceptibility_splits_real_from_complex(sk):
    """Dispersion is ADMITTED by the REAL beta curl (constitutive-only feature,
    the same split the certified real curl makes) and refused OUTRIGHT by the
    complex variant (complex ADE is a future tranche)."""
    pole = SimpleNamespace(
        driven=lambda: ("Ex",),
        drives=lambda name: name == "Ex",
        susceptibility=SimpleNamespace(kind="lorentzian"),
    )
    fields, pml = _build()
    fields.polarizations = [pole]
    assert _real_reasons(sk, fields, pml) == []
    assert any("ADE" in r for r in _constitutive_reasons(sk, fields, pml, "E"))
    cfields, cpml = _build(complex_storage=True, beta=0.2)
    cfields.polarizations = [pole]
    creasons = _complex_reasons(sk, cfields, cpml)
    assert any("future tranche" in r for r in creasons), creasons


def test_an_unknown_susceptibility_kind_is_refused(sk):
    fields, pml = _build()
    fields.polarizations = [SimpleNamespace(
        driven=lambda: ("Ex",), drives=lambda name: name == "Ex",
        susceptibility=SimpleNamespace(kind="gyrotropic"))]
    reasons = _real_reasons(sk, fields, pml)
    assert any("gyrotropic" in r for r in reasons), reasons


def test_a_missing_or_base_only_probe_refuses_the_complex_curl_by_name(sk):
    fields, pml = _build(complex_storage=True, beta=0.2)
    reasons = _complex_reasons(sk, fields, pml, probe=None)
    assert any("probe" in r for r in reasons), reasons
    base_only = _probe_record(include_beta_pattern=False)
    reasons = _complex_reasons(sk, fields, pml, probe=base_only)
    assert any(sk.BETA_PROBE_PATTERN in r for r in reasons), reasons


def test_a_disagreeing_beta_pattern_refuses(sk):
    record = _probe_record()
    record["patterns"][sk.BETA_PROBE_PATTERN] = "NAIVE"
    assert sk.beta_expansion_from_probe(record) is None
    fields, pml = _build(complex_storage=True, beta=0.2)
    reasons = _complex_reasons(sk, fields, pml, probe=record)
    assert any(sk.BETA_PROBE_PATTERN in r for r in reasons), reasons


# ---------------------------------------------------------------------------
# The non-discriminating clause, over the EXTENDED pattern set
# ---------------------------------------------------------------------------
#
# THE CLAUSE, and the two situations it must never conflate. A pattern that
# CANNOT tell the licensable arms apart constrains nothing and is EXCLUDED from
# the agreement test; a pattern that CAN tell them apart and names a different
# arm is evidence against a licence and must VETO it. Until 2026-08-15 this
# tranche implemented its own loop, which accepted AMBIGUOUS_BOTH for
# BETA_PROBE_PATTERN alone and let any BASE pattern gone blind veto — the
# pre-clause behaviour the base family had already retired. It now calls
# ``complex_fields.expansion_license`` over BETA_PROBE_PATTERNS, so there is one
# implementation of the rule rather than two that can drift apart.

def test_a_pattern_that_cannot_discriminate_is_excluded_from_the_agreement_test(sk):
    """The beta pattern is the expected blind one: with a signed-zero ``c_re``
    the fused arm's extra product is exact, so both arms produce identical bytes
    (measured AMBIGUOUS_BOTH on the reference NumPy, all 16 corpus coefficients,
    2026-08-15). It is excluded and the base four name the arm — and the SAME
    treatment now reaches a base pattern gone blind, which used to veto."""
    record = set_pattern(_probe_record(), sk.BETA_PROBE_PATTERN, AMBIGUOUS,
                         apart=0)
    verdict = sk.beta_expansion_license(record)
    assert verdict["expansion"] == 1
    assert verdict["basis"] == "measured"
    assert verdict["non_discriminating"] == [sk.BETA_PROBE_PATTERN]
    assert set(verdict["discriminating"]) == set(sk.PROBE_PATTERNS)
    assert verdict["refusals"] == []

    # A BASE pattern gone blind is excluded too, and the remaining three decide.
    both_blind = set_pattern(record, "c8_mul_c8", AMBIGUOUS, apart=0)
    verdict = sk.beta_expansion_license(both_blind)
    assert verdict["expansion"] == 1, verdict["refusals"]
    assert verdict["basis"] == "measured"
    assert verdict["non_discriminating"] == ["c8_mul_c8", sk.BETA_PROBE_PATTERN]


def test_a_pattern_that_discriminates_but_disagrees_still_refuses(sk):
    """THE OPPOSITE SITUATION, and the one a loosened clause would swallow.

    Three shapes, all of which must refuse:

    * the beta pattern names the OTHER arm — it discriminated, and it disagrees;
    * the beta pattern names an arm no transcription reproduces (NEITHER);
    * the beta pattern is LABELLED AMBIGUOUS_BOTH while its own detail block
      measures the licensable arms 128 words apart. That is a pattern that DID
      discriminate wearing the label of one that could not, and believing the
      label over the measurement is exactly how a wrong arm gets licensed.
    """
    disagreeing = set_pattern(_probe_record(), sk.BETA_PROBE_PATTERN, "NAIVE")
    verdict = sk.beta_expansion_license(disagreeing)
    assert verdict["expansion"] is None
    assert any("disagree" in reason for reason in verdict["refusals"]), verdict
    assert sk.BETA_PROBE_PATTERN not in verdict["non_discriminating"]

    neither = dict(_probe_record())
    neither["patterns"] = dict(neither["patterns"])
    neither["patterns"][sk.BETA_PROBE_PATTERN] = "NEITHER"
    assert sk.beta_expansion_from_probe(neither) is None

    contradictory = set_contradictory_ambiguity(_probe_record(),
                                                sk.BETA_PROBE_PATTERN)
    verdict = sk.beta_expansion_license(contradictory)
    assert verdict["expansion"] is None
    assert any("words APART" in reason for reason in verdict["refusals"]), verdict
    assert verdict["non_discriminating"] == [], (
        "a pattern whose ambiguity claim its own detail contradicts must never "
        "be recorded as excluded: excluded means MEASURABLY blind")
    assert verdict["ambiguity_refused"] == [sk.BETA_PROBE_PATTERN]

    # ...and the exclusion clause must not rescue it. With every OTHER pattern
    # genuinely blind, the one pattern that discriminated is all the evidence
    # there is, and it says the record is inconsistent.
    only_bad_one_left = set_contradictory_ambiguity(_probe_record(),
                                                    sk.BETA_PROBE_PATTERN)
    for name in sk.PROBE_PATTERNS:
        set_pattern(only_bad_one_left, name, AMBIGUOUS, apart=0)
    assert sk.beta_expansion_from_probe(only_bad_one_left) is None


def test_an_ambiguity_claim_a_diagnostic_arm_also_passes_refuses(sk):
    """A comparison the KNOWN-WRONG transcription passes has no power to license
    anything, whichever word the record puts on it — so it is not an exclusion,
    it is a refusal."""
    record = set_pattern(_probe_record(), sk.BETA_PROBE_PATTERN, AMBIGUOUS,
                         apart=0, diagnostic_matched=True)
    verdict = sk.beta_expansion_license(record)
    assert verdict["expansion"] is None
    assert any("known-wrong transcription" in reason
               for reason in verdict["refusals"]), verdict["refusals"]


def test_an_ambiguity_claim_with_no_measurement_behind_it_refuses(sk):
    """AMBIGUOUS_BOTH is a claim about a MEASUREMENT. Over zero vectors every arm
    "matches", so the word alone is indistinguishable from a comparison of
    nothing and may not be excluded on trust."""
    record = set_pattern(_probe_record(), sk.BETA_PROBE_PATTERN, AMBIGUOUS,
                         apart=0, vectors=0)
    verdict = sk.beta_expansion_license(record)
    assert verdict["expansion"] is None
    assert any("comparison of nothing" in reason
               for reason in verdict["refusals"]), verdict["refusals"]


def test_a_record_on_which_nothing_discriminates_is_not_a_measurement(sk):
    """When EVERY pattern is blind the arm was not measured this run, so the
    licence may only come from the environment table — and this fixture's record
    carries no environment, so it refuses by name rather than picking an arm."""
    record = _probe_record()
    for name in sk.BETA_PROBE_PATTERNS:
        set_pattern(record, name, AMBIGUOUS, apart=0)
    verdict = sk.beta_expansion_license(record)
    assert verdict["expansion"] is None
    assert verdict["arms_coincide_on_every_pattern"] is True
    assert any("matches no measured row" in reason
               for reason in verdict["refusals"]), verdict["refusals"]


def test_the_licence_names_the_pattern_set_and_the_patterns_it_excluded(sk):
    """The artifact a reader gets must say what did NOT contribute, not only the
    arm: an arm licensed from one live pattern with four excluded is a weaker
    claim than one licensed from five, and the verdict has to let them be told
    apart."""
    record = set_pattern(_probe_record(), sk.BETA_PROBE_PATTERN, AMBIGUOUS,
                         apart=0)
    verdict = sk.beta_expansion_license(record)
    assert verdict["probe_patterns"] == list(sk.BETA_PROBE_PATTERNS)
    assert verdict["basis"] == "measured"
    assert verdict["non_discriminating"] == [sk.BETA_PROBE_PATTERN]
    assert sk.beta_expansion_from_probe(record) == verdict["expansion"]


def test_a_metallic_axis_with_a_nonzero_k_component_is_refused(sk):
    fields, pml = _build(complex_storage=True, beta=0.2,
                         boundaries=("metallic", "metallic", "periodic"))
    fields.grid.k_point = (0.3, 0.0, 0.0)
    reasons = _complex_reasons(sk, fields, pml)
    assert any("metallic" in r and "k component" in r for r in reasons), reasons


def test_a_planted_k_on_the_invariant_axis_is_refused_by_name(sk):
    """Clause 5b: the COMPLEX family must refuse a k component on the
    invariant axis by its own restated clause, not by leaning on the Grid
    constructor guard (which refuses the pairing at construction — beta IS
    the analytic z dependence). The real family's k = 0 clause already
    catches the same plant; both are pinned here."""
    with pytest.raises(ValueError, match="invariant"):
        Grid(resolution=10.0, cell_size=(2.0, 1.6, 0.0), dimensions=2,
             beta=0.2, k_point=(0.0, 0.0, 0.25))
    fields, pml = _build(complex_storage=True, beta=0.2)
    fields.grid.k_point = (0.0, 0.0, 0.25)  # planted past validation
    reasons = _complex_reasons(sk, fields, pml)
    assert any("invariant axis" in r for r in reasons), reasons
    rfields, rpml = _build()
    rfields.grid.k_point = (0.0, 0.0, 0.25)
    rreasons = _real_reasons(sk, rfields, rpml)
    assert any("not exactly zero" in r for r in rreasons), rreasons


def test_real_storage_refuses_an_inplane_k(sk):
    """Real storage carries no Bloch phase; in-plane k + beta is the complex
    variant's domain (binary_grating's real/imag case is ALSO folded, and both
    clauses must name their refusal)."""
    fields, pml = _build()
    fields.grid.k_point = (0.3, 0.0, 0.0)
    reasons = _real_reasons(sk, fields, pml)
    assert any("not exactly zero" in r or "complex variant" in r
               for r in reasons), reasons


def test_an_unknown_sub_step_or_side_raises_rather_than_refusing(sk):
    fields, pml = _build()
    with pytest.raises(ValueError, match="sub_step"):
        sk.beta_pml_curl_coverage(fields, pml, "step_H")
    with pytest.raises(ValueError, match="sub_step"):
        sk.beta_bloch_pml_curl_coverage(fields, pml, "step_H")
    with pytest.raises(ValueError, match="side"):
        sk.beta_run_constitutive_coverage(fields, pml, "B")
    with pytest.raises(ValueError, match="side"):
        sk.beta_run_complex_constitutive_coverage(fields, pml, "B")


# --- predicate mutations (m11): each clause is load-bearing ------------------

def _mutate(module, function, needle: str, replacement: str):
    import inspect  # noqa: PLC0415
    import textwrap  # noqa: PLC0415

    source = textwrap.dedent(inspect.getsource(function))
    mutated = source.replace(needle, replacement)
    assert mutated != source, f"the clause moved; update this mutation: {needle!r}"
    namespace = dict(module.__dict__)
    exec(compile(mutated, "<mutated special_kz>", "exec"), namespace)  # noqa: S102
    return namespace[function.__name__]


def test_m11_dropping_the_offdiag_real_clause_is_caught(sk, monkeypatch):
    """Nothing else in the REAL family refuses offdiag (the certified curl
    ADMITS it), so deleting 12b must flip the verdict to admitted — which is
    what makes the clause load-bearing rather than decorative."""
    fields, pml = _build()
    fields._chi1inv_offdiagonal = {"Ex": {"Ey": numpy.full(
        fields.grid.shape, 0.1, dtype=numpy.float32)}}
    assert any("implicit-i" in r for r in _real_reasons(sk, fields, pml))
    mutated = _mutate(sk, sk._beta_real_grid_reasons,
                      'if getattr(fields, "has_offdiagonal_epsilon", False):',
                      "if False:")
    monkeypatch.setattr(sk, "_beta_real_grid_reasons", mutated)
    assert _real_reasons(sk, fields, pml) == []


def test_m11_dropping_the_bfast_clause_is_caught(sk, monkeypatch):
    fields, pml = _build()
    fields.grid.bfast_scaled_k = (0.0, 0.0, 0.5)
    if not fields.grid.bfast_active:
        pytest.skip("this grid build refuses bfast_scaled_k at construction")
    assert any("BFAST" in r for r in _real_reasons(sk, fields, pml))
    mutated = _mutate(sk, sk._beta_real_grid_reasons,
                      'if getattr(grid, "bfast_active", False):',
                      "if False:")
    monkeypatch.setattr(sk, "_beta_real_grid_reasons", mutated)
    assert _real_reasons(sk, fields, pml) == []


def test_m11_widening_the_boundary_set_is_caught(sk, monkeypatch):
    """A folded axis resolves to 'mirror'; with the boundary clause widened the
    only remaining refusals are the explicit fold clauses — delete both and the
    folded run is admitted, so each is measured as load-bearing."""
    fields, pml = _build(symmetry=("X",), beta=0.2, pml_override=MIRROR_PML)
    assert _real_reasons(sk, fields, pml) != []
    source_mutations = (
        ("if kind not in COVERED_BOUNDARIES:", "if False:"),
        ('if _call(grid, "has_symmetry", default=False):', "if False:"),
        ('if _call(grid, "is_mirrored", axis, default=False):', "if False:"),
    )
    import inspect  # noqa: PLC0415
    import textwrap  # noqa: PLC0415

    source = textwrap.dedent(inspect.getsource(sk._beta_real_grid_reasons))
    for needle, replacement in source_mutations:
        assert needle in source, needle
        source = source.replace(needle, replacement)
    namespace = dict(sk.__dict__)
    exec(compile(source, "<mutated special_kz>", "exec"), namespace)  # noqa: S102
    monkeypatch.setattr(sk, "_beta_real_grid_reasons",
                        namespace["_beta_real_grid_reasons"])
    assert _real_reasons(sk, fields, pml) == []


# ---------------------------------------------------------------------------
# The host coefficient — bytes against stepping._special_kz_beta_term
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("beta,dt", [(BETA_KZ2D, 1.0 / 24.0),
                                     (BETA_GRATING, 0.034),
                                     (0.2, 0.05)])
@pytest.mark.parametrize("magnetic,sign", [(True, +1.0), (True, -1.0),
                                           (False, +1.0), (False, -1.0)])
def test_the_real_coefficient_matches_stepping_bytes(sk, beta, dt, magnetic, sign):
    grid = SimpleNamespace(beta=beta, dt=dt, xp=numpy)
    shim = SimpleNamespace(grid=grid, has_offdiagonal_epsilon=False)
    partner = numpy.full((2, 2, 1), 1.0, dtype=numpy.float32)
    expected = stepping._special_kz_beta_term(shim, partner, sign, magnetic)
    plus, minus = sk.beta_curl_coefficients(beta, dt, magnetic, False)
    value = plus if sign > 0 else minus
    got = -(numpy.float32(value) * partner)
    assert got.tobytes() == expected.tobytes()


@pytest.mark.parametrize("beta,dt", [(BETA_SPECIAL_KZ, 0.035), (0.2, 0.05)])
@pytest.mark.parametrize("magnetic,sign", [(True, +1.0), (True, -1.0),
                                           (False, +1.0), (False, -1.0)])
def test_the_complex_coefficient_matches_stepping_bytes(sk, beta, dt, magnetic,
                                                        sign):
    """The signed-zero real word included: the words the plan binds, negated
    and multiplied, must reproduce stepping's complex64 term exactly."""
    grid = SimpleNamespace(beta=beta, dt=dt, xp=numpy)
    shim = SimpleNamespace(grid=grid, has_offdiagonal_epsilon=False)
    partner = numpy.full((2, 2, 1), 1.0 + 0.0j, dtype=numpy.complex64)
    expected = stepping._special_kz_beta_term(shim, partner, sign, magnetic)
    words = sk.beta_curl_coefficients(beta, dt, magnetic, True)
    re, im = words[0] if sign > 0 else words[1]
    # Rebuild the coefficient from the plan's words by PLANE ASSIGNMENT (an
    # addition would destroy the signed zero) and compare the full product.
    coefficient = numpy.empty((), dtype=numpy.complex64)
    coefficient.real = numpy.float32(re)
    coefficient.imag = numpy.float32(im)
    got = -(coefficient * partner)
    assert got.tobytes() == expected.tobytes()


def test_the_complex_coefficient_real_word_is_a_signed_zero_passed_through(sk):
    """B side, minus sign: Python's (negative float) * 1j puts -0.0 in the real
    part. The plan must carry it; synthesizing +0.0 is the m9 hazard."""
    words = sk.beta_curl_coefficients(0.332, 0.05, magnetic=True,
                                      complex_storage=True)
    (_pre, _pim), (mre, _mim) = words
    assert mre == 0.0 and math.copysign(1.0, mre) < 0


# ---------------------------------------------------------------------------
# Plan shapes
# ---------------------------------------------------------------------------

def _curl_arrays(sub_step="step_B", shape=(4, 3, 1), dtype=numpy.float32):
    from meep_gpu.triton_kernels.launch import SUB_STEPS

    spec = SUB_STEPS[sub_step]
    rng = numpy.random.default_rng(7)
    names = (tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
             + tuple(spec["sources"]))
    arrays = {}
    for name in names:
        if dtype == numpy.complex64:
            host = (rng.uniform(-1, 1, shape) + 1j * rng.uniform(-1, 1, shape)
                    ).astype(numpy.complex64)
        else:
            host = rng.uniform(-1, 1, shape).astype(numpy.float32)
        arrays[name] = numpy.ascontiguousarray(host)
    flat = {f"{stem}_{axis}": numpy.ascontiguousarray(
        rng.uniform(0.5, 1.0, shape["xyz".index(axis)]).astype(numpy.float32))
        for axis in "xyz" for stem in ("kms", "sinv")}
    return arrays, flat


def test_engine_route_plans_return_none_on_a_numpy_host(sk):
    fields, pml = _build()
    assert sk.plan_beta_pml_curl(fields, pml, "step_B") is None
    cfields, cpml = _build(complex_storage=True, beta=0.2)
    assert sk.plan_beta_bloch_pml_curl(cfields, cpml, "step_B",
                                       probe=_probe_record()) is None
    assert sk.plan_beta_run_constitutive(fields, pml, "H") is None
    assert sk.plan_beta_run_complex_constitutive(
        cfields, cpml, "E", probe=_probe_record()) is None


def test_the_real_plan_shape_from_arrays(sk):
    arrays, flat = _curl_arrays("step_D")
    plus, minus = sk.beta_curl_coefficients(BETA_KZ2D, 0.05, magnetic=False,
                                            complex_storage=False)
    plan = sk.plan_beta_pml_curl_from_arrays(
        "step_D", arrays, flat, (1, 0, 0), 0.34, plus, minus)
    assert plan.sub_step == "step_D" and plan.backward == 1
    assert plan.bc == (1, 0, 0)
    assert plan.has_beta == 1
    assert plan.beta_plus == plus and plan.beta_minus == minus
    assert len(plan._coefficients) == 6
    assert plan.n_elem == 12


def test_the_complex_plan_carries_the_signed_zero_words(sk):
    arrays, flat = _curl_arrays("step_B", dtype=numpy.complex64)
    words = sk.beta_curl_coefficients(BETA_KZ2D, 0.05, magnetic=True,
                                      complex_storage=True)
    plan = sk.plan_beta_bloch_pml_curl_from_arrays(
        "step_B", arrays, flat, (0, 0, 0), (None, None, None), 0.35, 1, words)
    (_pre, _pim), (mre, _mim) = plan.beta_words
    assert mre == 0.0 and math.copysign(1.0, mre) < 0
    assert plan.has_beta == 1 and plan.expansion == 1
    assert plan.phased == (0, 0, 0)


def test_the_step_D_plan_conjugates_the_phase_itself(sk):
    arrays, flat = _curl_arrays("step_D", dtype=numpy.complex64)
    words = sk.beta_curl_coefficients(0.2, 0.05, magnetic=False,
                                      complex_storage=True)
    phase = complex(0.8, 0.6)
    plan = sk.plan_beta_bloch_pml_curl_from_arrays(
        "step_D", arrays, flat, (0, 0, 0), (phase, None, None), 0.35, 1, words)
    assert plan.phased == (1, 0, 0)
    assert plan.phase_values[0] == pytest.approx(0.8)
    assert plan.phase_values[1] == pytest.approx(-0.6)  # the conjugate


def test_the_identity_arm_is_constructible(sk):
    """has_beta=0 exists for the gate's R5 identity leg only."""
    arrays, flat = _curl_arrays("step_B")
    plan = sk.plan_beta_pml_curl_from_arrays(
        "step_B", arrays, flat, (0, 0, 0), 0.5, 0.1, -0.1, has_beta=0)
    assert plan.has_beta == 0


# ---------------------------------------------------------------------------
# The in-test transcription — bytes against stepping.step_B / step_D
# ---------------------------------------------------------------------------
#
# A THIRD transcription (the gate's is the second), deliberately written from
# the term tables rather than imported, so agreement means the contract and
# not this file agreeing with itself.

B_TERMS = (("Bx", "Ez", 1, "Ey", 2, "y", "z"),
           ("By", "Ex", 2, "Ez", 0, "z", "x"),
           ("Bz", "Ey", 0, "Ex", 1, "x", "y"))
D_TERMS = (("Dx", "Hz", 1, "Hy", 2, "y", "z"),
           ("Dy", "Hx", 2, "Hz", 0, "z", "x"),
           ("Dz", "Hy", 0, "Hx", 1, "x", "y"))
#: target index -> (partner source key index, sign): 0 takes the SECOND source
#: at +1, 1 the FIRST at -1, 2 nothing (stepping.py:384-391/:467-474).
BETA_PARTNERS = {0: ("second", +1.0), 1: ("first", -1.0)}


def _face(axis, index):
    return tuple(index if a == axis else slice(None) for a in range(3))


def _shift(field, axis, kind, backward, phase):
    shifted = numpy.roll(field, 1 if backward else -1, axis=axis)
    index = 0 if backward else -1
    if kind == "periodic":
        if phase is not None:
            factor = phase.conjugate() if backward else phase
            shifted[_face(axis, index)] *= shifted.dtype.type(factor)
        return shifted
    shifted[_face(axis, index)] = 0
    return shifted


def _transcribed_sub_step(fields, pml, sub_step):
    """One beta curl sub-step on COPIES of the state, returned by name."""
    grid = fields.grid
    kinds = stepping._boundary_kinds(grid, pml)
    phases = tuple(grid.bloch_phase(axis) for axis in range(3))
    backward = sub_step == "step_D"
    terms = D_TERMS if backward else B_TERMS
    sources = {name: numpy.array(getattr(fields, name))
               for name in (("Hx", "Hy", "Hz") if backward
                            else ("Ex", "Ey", "Ez"))}
    state = {}
    dtdx = grid.dt / grid.dx
    suffix = "" if backward else "_h"
    magnetic = not backward
    complex_storage = sources[next(iter(sources))].dtype.kind == "c"
    for index, (target, g1, a1, g2, a2, dsig, dsigu) in enumerate(terms):
        field = numpy.array(getattr(fields, target))
        fu = numpy.array(getattr(fields, "fu_" + target))
        shifted_first = _shift(sources[g1], a1, kinds[a1], backward, phases[a1])
        shifted_second = _shift(sources[g2], a2, kinds[a2], backward, phases[a2])
        curl = dtdx * ((shifted_first - sources[g1])
                       + (sources[g2] - shifted_second))
        if index in BETA_PARTNERS:
            which, sign = BETA_PARTNERS[index]
            partner = sources[g2] if which == "second" else sources[g1]
            coefficient = sign * 2.0 * math.pi * grid.beta * grid.dt
            if complex_storage:
                coefficient = coefficient * (1j if magnetic else -1j)
            curl = curl + (-(partner.dtype.type(coefficient) * partner))
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
        state[target] = field
        state["fu_" + target] = fu
    return state


def _seed(fields, complex_storage, seed=20260811):
    rng = numpy.random.default_rng(seed)
    names = ("Bx", "By", "Bz", "Dx", "Dy", "Dz",
             "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
             "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz")
    for name in names:
        target = getattr(fields, name)
        shape = target.shape
        host = rng.uniform(-1, 1, shape).astype(numpy.float32)
        if complex_storage:
            value = numpy.empty(shape, dtype=numpy.complex64)
            value.real = host
            value.imag = rng.uniform(-1, 1, shape).astype(numpy.float32)
            value.imag[_face(0, 0)] = -0.0  # signed-zero rows: the term is a
            value.real[_face(1, 0)] = -0.0  # pure scalar-by-field product
        else:
            value = host
            value[_face(0, 0)] = -0.0
        target[...] = value


REFERENCE_CASES = (
    ("real_kz2d_periodic", dict(beta=BETA_KZ2D, courant=0.5)),
    ("real_grating_sign_metallic_x",
     dict(beta=BETA_GRATING, courant=0.34,
          boundaries=("metallic", "periodic", "periodic"))),
    ("complex_k0", dict(beta=0.2, courant=0.35, complex_storage=True)),
    ("complex_inplane_bloch",
     dict(beta=BETA_SPECIAL_KZ, courant=0.35, complex_storage=True,
          k_point=(KX_SPECIAL_KZ, 0.0, 0.0))),
)


@pytest.mark.parametrize("sub_step", ["step_B", "step_D"])
@pytest.mark.parametrize("name,kwargs", REFERENCE_CASES,
                         ids=[c[0] for c in REFERENCE_CASES])
def test_the_beta_sub_step_transcription_matches_stepping_bytes(
        sk, name, kwargs, sub_step):
    """stepping.step_B/step_D on a real beta Fields against the in-test
    transcription, byte for byte — sign table, partner table, coefficient
    rounding, dtdx-exclusion and term position all pinned at once (any one
    wrong moves f32 bytes on these seeds; measured during construction: a
    dtdx-scaled term differs in thousands of words)."""
    complex_storage = bool(kwargs.get("complex_storage"))
    fields, pml = _build(**kwargs)
    assert fields.grid.beta != 0.0
    _seed(fields, complex_storage)
    expected = _transcribed_sub_step(fields, pml, sub_step)
    getattr(stepping, sub_step)(fields, pml)
    for key, value in expected.items():
        got = numpy.asarray(getattr(fields, key))
        assert got.tobytes() == value.tobytes(), (
            f"{name}/{sub_step}: {key} diverged from the transcription")


def test_the_transcription_is_not_vacuous_against_beta_zero(sk):
    """The same transcription WITHOUT the beta term must NOT match a beta run:
    proves the term moves bytes on these seeds, so the pins above measure it."""
    fields, pml = _build(beta=BETA_KZ2D, courant=0.5)
    _seed(fields, complex_storage=False)
    zero_grid_fields, zero_pml = _build(beta=BETA_KZ2D, courant=0.5)
    _seed(zero_grid_fields, complex_storage=False)
    zero_grid_fields.grid.beta = 0.0  # term skipped at stepping.py:384
    expected = _transcribed_sub_step(fields, pml, "step_B")
    stepping.step_B(zero_grid_fields, zero_pml)
    differing = sum(
        int(numpy.asarray(getattr(zero_grid_fields, key)).tobytes()
            != value.tobytes())
        for key, value in expected.items())
    assert differing >= 4, (
        "the beta term moved no bytes on the seeded state; the reference pins "
        "above would be vacuous")


# ---------------------------------------------------------------------------
# Kernel source facts the gate relies on
# ---------------------------------------------------------------------------

def _segment(name: str) -> str:
    source = MODULE_PATH.read_text(encoding="utf-8")
    start = source.index(f"def {name}(")
    ends = [position for position in (source.find("\n@triton.jit", start + 1),
                                      source.find("\n# ----", start + 1))
            if position != -1]
    return source[start:min(ends) if ends else len(source)]


def test_the_beta_term_sits_after_the_curl_and_before_the_mask():
    for kernel, curl_marker in (
            ("beta_pml_curl_step", "curl0 = dtdx * ((c_y - c)"),
            ("beta_bloch_pml_curl_step", "_mul_coefficient_left(dtdx")):
        segment = _segment(kernel)
        term = segment.index("if HAS_BETA:")
        curl = segment.index(curl_marker)
        mask = segment.index("at_x, at_y, at_z = i == 0, j == 0, k == 0")
        assert curl < term < mask, (
            f"{kernel}: the beta term must sit between the curl and the "
            f"ownership mask, where the array path adds it")


def test_the_real_term_is_a_subtraction_of_the_scaled_center(sk):
    segment = _segment("beta_pml_curl_step")
    assert "curl0 = curl0 - (beta_plus * b)" in segment
    assert "curl1 = curl1 - (beta_minus * a)" in segment
    assert "beta_plus * b_z" not in segment  # never a shifted operand
    assert "curl2 = curl2 -" not in segment  # target 2 gets nothing


def test_the_complex_term_uses_the_module_local_helper_on_center_words(sk):
    segment = _segment("beta_bloch_pml_curl_step")
    assert "_mul_imag_coefficient_left(bp_re, bp_im, b_re, b_im" in segment
    assert "_mul_imag_coefficient_left(bm_re, bm_im, a_re, a_im" in segment
    helper = _segment("_mul_imag_coefficient_left")
    assert "tl.math.fma(c_re, z_re, (c_im * z_im) * -1.0)" in helper
    assert "(c_re * z_re) - (c_im * z_im)" in helper
    # Triton lowers unary minus as ``0.0 - x`` (semantic.py:386-391, 3.1.0),
    # which turns ``-(+0.0)`` into ``+0.0``; the addend must be negated
    # sign-exactly (``* -1.0`` folds to fneg). A device run measured the unary
    # spelling byte-identical to the m9 FOLD on the engineered state.
    assert "-(c_im * z_im)" not in helper


def test_the_dtdx_exclusion_is_source_visible(sk):
    """The term multiplies the coefficient by the CENTER, never by dtdx."""
    for kernel in ("beta_pml_curl_step", "beta_bloch_pml_curl_step"):
        segment = _segment(kernel)
        block = segment[segment.index("if HAS_BETA:"):]
        block = block[:block.index("at_x")]
        assert "dtdx" not in block, (
            f"{kernel}: the beta term is an analytic derivative and must not "
            f"be scaled by dtdx (stepping.py:760-762)")


# ---------------------------------------------------------------------------
# The gate trio exists and holds the run discipline
# ---------------------------------------------------------------------------

def test_the_gate_and_its_composition_probe_exist():
    assert GATE_PATH.exists()
    assert COMPOSITION_PATH.exists()


def test_the_record_the_gate_emits_can_satisfy_the_licence_rule_on_every_pattern(sk):
    """THE TWO HALVES MUST FIT, and the exclusion clause is what makes the second
    half load-bearing here.

    Since the licence became ``complex_fields.expansion_license`` over
    :data:`BETA_PROBE_PATTERNS`, an ``AMBIGUOUS_BOTH`` claim is believed only
    when the record's OWN ``vectors[name]`` and ``detail[name]`` back it. This
    tranche's earlier loop took the word on trust, so nothing forced the gate to
    WRITE that evidence for the pattern it adds — and a record the gate honestly
    emits must not refuse on a bookkeeping gap: a false refusal is as much a
    defect as a false licence. The base gate carried exactly that gap on
    ``python_float_left`` (detail a LIST, no ``vectors`` entry at all), and this
    tranche's beta pattern has the same list shape, one entry per corpus
    coefficient.

    Nothing here launches: ``measure_beta_expansion_record`` on NumPy is host
    arithmetic. The verdict assertions read what the record SAYS rather than
    pinning an arm, because which patterns discriminate is a platform fact and
    this suite runs wherever the merge bar does.
    """
    import importlib as _importlib  # noqa: PLC0415
    import json  # noqa: PLC0415

    sys.path.insert(0, str(PARITY_DIR))
    # Imported as itself, with NO ``sys.modules['cupy']`` stub — such a stub is
    # process-global and never undone, and ``meep_gpu/conftest.py`` fails any
    # test that reinstates one.
    gate = _importlib.import_module(GATE_PATH.stem)
    complex_fields = _importlib.import_module(
        "meep_gpu.triton_kernels.complex_fields")
    record = gate.measure_beta_expansion_record(numpy, "numpy")
    for name in sk.BETA_PROBE_PATTERNS:
        assert record["vectors"].get(name), f"no vector count for {name!r}"
        assert record["detail"].get(name), f"no detail block for {name!r}"

    # Each pattern forced to AMBIGUOUS_BOTH with its own detail made CONSISTENT:
    # the evidence the gate writes must be able to carry the claim.
    for name in sk.BETA_PROBE_PATTERNS:
        probe = json.loads(json.dumps(record, default=str))
        probe["backend"] = "cupy"
        probe["patterns"][name] = complex_fields.AMBIGUOUS_BOTH
        entries = (probe["detail"][name] if isinstance(probe["detail"][name], list)
                   else [probe["detail"][name]])
        for entry in entries:
            entry["licensable_arms_disagreement_words"] = 0
            entry["discriminates"] = False
            entry["matches"] = {"FMA_V1": True, "NAIVE": True,
                                "PLANEWISE_diagnostic": False}
            entry["mismatch_words"] = {"FMA_V1": 0, "NAIVE": 0,
                                       "PLANEWISE_diagnostic": 17}
        reasons = complex_fields._ambiguity_evidence_reasons(probe, name)
        assert reasons == [], (name, reasons)

    # ...and the record AS MEASURED must reach a verdict whose refusals, if any,
    # name a PATTERN rather than a bookkeeping gap. Both branches are asserted,
    # because the resolved subnormal policy is PROCESS-GLOBAL: a session driven
    # to a policy this host's FPU does not implement cuts the candidate arms for
    # one convention while the platform's bytes come off the other, and the
    # honest verdict there is a NEITHER-class pattern refused BY NAME — which is
    # a different fact from an ambiguity the rule had to reject, and must not be
    # mistaken for one. Whichever patterns went blind are recorded as EXCLUDED.
    measured = json.loads(json.dumps(record, default=str))
    measured["backend"] = "cupy"  # the rule binds CuPy's bytes; this is a fixture
    verdict = sk.beta_expansion_license(measured)
    assert verdict["ambiguity_refused"] == [], verdict["refusals"]
    blind = [name for name in verdict["probe_patterns"]
             if measured["patterns"][name] == complex_fields.AMBIGUOUS_BOTH]
    assert verdict["non_discriminating"] == blind, verdict
    unlicensable = sorted(
        name for name in verdict["probe_patterns"]
        if measured["patterns"][name] not in complex_fields.EXPANSIONS
        and measured["patterns"][name] != complex_fields.AMBIGUOUS_BOTH)
    if unlicensable:
        assert verdict["expansion"] is None, (unlicensable, verdict)
        for name in unlicensable:
            assert any(repr(name) in reason for reason in verdict["refusals"]), (
                name, verdict["refusals"])
    else:
        assert verdict["basis"] == "measured", verdict
        assert verdict["expansion"] is not None, verdict["refusals"]


def test_both_artifact_cutters_record_the_basis_and_the_exclusions_by_name():
    """An arm and a word is not a readable licence.

    'measured' against 'environment_default' is the difference between this run
    having DISCRIMINATED the arm and having inherited it from a prior one, and
    the patterns a verdict EXCLUDED are what a reader needs to weigh either — an
    arm carried by one live pattern with four excluded is a weaker claim than
    one carried by five, and an artifact that prints only ``licensed: true``
    cannot be told apart from the weaker case. Pinned by source text because the
    write happens on a CuPy host and this suite has none; the verdict's own
    shape is pinned by behaviour above.
    """
    for path in (GATE_PATH, COMPOSITION_PATH):
        text = path.read_text(encoding="utf-8")
        assert "beta_expansion_license(" in text, path
        assert "expansion_license" in text and "verdict" in text, path
        assert "non_discriminating" in text, path
        assert "basis" in text, path


def test_both_gate_files_install_the_strip_before_any_compile():
    for path in (GATE_PATH, COMPOSITION_PATH):
        text = path.read_text(encoding="utf-8")
        assert "install_ftz_strip" in text, path
        assert "policy_stamp" in text, path
