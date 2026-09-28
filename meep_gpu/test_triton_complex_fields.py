"""Merge-bar tests for :mod:`meep_gpu.triton_kernels.complex_fields`.

Everything here runs on a laptop with NO GPU and NO Triton, which is the point:
the kernels' bytes are the CUDA gate's to certify
(``parity/meep_gpu/gate_triton_complex.py``, run on a CUDA host);
what a laptop can hold is (a) the optional dependency staying optional, (b) the
predicates — pure Python, and the half that decides whether a silent wrong
answer is possible, (c) the host-side phase and probe arithmetic, which is
NumPy-checkable, and (d) the structural pins on the kernel source that the byte
gate's verdict will be cut against, and (e) the SUBNORMAL-POLICY dependence:
the gate certifies the FMA_V1 expansion under the policy it was cut under and
that licence does not transfer (bytes differ in the subnormal range), so the
policy plumbing — the strip's licensing checks, the artifact stamps, the
install-before-compile ordering, and the rule for a pattern that cannot
discriminate under the policy in force — is pinned here where it is pure host
logic.

This module is NOT wired: the package ``__init__`` must not import it, dispatch
stays disabled, and both facts are asserted here as behaviour.
"""

from __future__ import annotations

import ast
import builtins
import cmath
import fractions
import importlib
import json
import pathlib
import sys
from types import SimpleNamespace

import numpy
import pytest

from meep_gpu.fields import Fields
from meep_gpu.grid import Grid
from meep_gpu.pml import PML
from meep_gpu.test_triton_kernels import GUARD_SPELLING, PACKAGE_DIR, code_of

# Every probe record this module builds is stamped 'keep'; the complex
# families' policy clause fails closed when the run policy can be neither
# read nor declared, so the premise is declared rather than left implicit.
# See ``run_policy_declared_keep`` in conftest.py.
pytestmark = pytest.mark.usefixtures("run_policy_declared_keep")


MODULE_PATH = PACKAGE_DIR / "complex_fields.py"
MODULE_NAME = "meep_gpu.triton_kernels.complex_fields"
PARITY_DIR = pathlib.Path(__file__).resolve().parents[1] / "parity" / "meep_gpu"
GATE_PATH = PARITY_DIR / "gate_triton_complex.py"
COMPOSITION_PATH = PARITY_DIR / "probe_triton_complex_composition.py"
SLURM_PATH = PARITY_DIR / "run_triton_complex_composition.slurm"

#: A well-formed probe artifact: the reference laptop's measured classification
#: (FMA_V1 on every orientation), stamped for the backend the kernel binds to.
#:
#: WELL-FORMED IS MORE THAN THE PATTERN MAP, since 2026-08-15. A record must also
#: state the policy its bytes were cut under and the policy its candidates were
#: built under — on EVERY path, not only when something classified
#: AMBIGUOUS_BOTH — and any AMBIGUOUS_BOTH must carry the vector count and the
#: measured arms-apart word count that back it. The audit drove records missing
#: exactly those fields through ``expansion_license`` and they were licensed
#: 'measured' with zero refusals, which is why the helper now supplies them and
#: the dedicated tests below take them away one at a time.
def _probe_record(value: str = "FMA_V1", backend: str = "cupy",
                  policy: str = "ieee_keep_ftz_stripped", resolved: str = "keep",
                  candidate_policy: str = None):
    patterns = {name: value for name in (
        "c8_mul_c8", "c8_mul_f4_field_left",
        "f4_mul_c8_coefficient_left", "python_float_left")}
    record = {
        "backend": backend,
        "patterns": patterns,
        "vectors": {name: 2792 for name in patterns},
        "detail": {name: _pattern_detail(value) for name in patterns},
    }
    if policy is not None or resolved is not None:
        record["subnormal_policy"] = {"policy": policy, "resolved": resolved}
    record["candidates"] = {
        "policy": resolved if candidate_policy is None else candidate_policy}
    return record


def stamp_probe_record(record, *, policy="ieee_keep_ftz_stripped", resolved="keep",
                       candidate_policy=None, vectors=2792):
    """Make a bare ``{backend, patterns}`` fixture a WELL-FORMED probe artifact.

    Shared with the sibling complex families' suites (special_kz, folded_complex,
    cylindrical_complex, and the planner/residual batteries), which extend
    ``PROBE_PATTERNS`` with their own patterns and delegate the licence itself to
    ``complex_fields``. They all built the pre-2026-08-15 shape: a pattern map and
    nothing else, i.e. exactly the artifact gate_triton_complex refuses by name as
    "from before artifacts stated their policy". Once the licence rule stopped
    skipping that check on the all-discriminating path, every one of those
    fixtures became a refusal — correctly, which is why they are stamped here
    rather than the rule being loosened back.

    Mutates and returns ``record`` so a caller can keep composing it.
    """
    patterns = record.get("patterns") or {}
    record["subnormal_policy"] = {"policy": policy, "resolved": resolved}
    record["candidates"] = {
        "policy": resolved if candidate_policy is None else candidate_policy}
    record.setdefault("vectors", {})
    record.setdefault("detail", {})
    for name, value in patterns.items():
        record["vectors"].setdefault(name, vectors)
        if isinstance(value, str):
            record["detail"].setdefault(name, _pattern_detail(value))
    return record


#: Spelled out rather than imported from the module whose rule these fixtures
#: test — the same reason ``_pattern_detail`` builds the evidence block by hand.
AMBIGUOUS = "AMBIGUOUS_BOTH"


def _pattern_detail(value: str, apart: int = 128):
    """The evidence block ``_classify`` writes, shaped for one pattern verdict.

    ``apart`` is the measured candidate-against-candidate disagreement. An
    AMBIGUOUS_BOTH claim needs it to be ZERO (that IS the claim); a named arm
    needs it non-zero, or the pattern could not have named anything.
    """
    ambiguous = value == "AMBIGUOUS_BOTH"  # module.AMBIGUOUS_BOTH, spelled out
    return {
        "matches": {"FMA_V1": ambiguous or value == "FMA_V1",
                    "NAIVE": ambiguous or value == "NAIVE",
                    "PLANEWISE_diagnostic": False},
        "mismatch_words": {"FMA_V1": 0 if (ambiguous or value == "FMA_V1") else 7,
                           "NAIVE": 0 if (ambiguous or value == "NAIVE") else 7,
                           "PLANEWISE_diagnostic": 31},
        "licensable_arms_disagreement_words": 0 if ambiguous else apart,
        "discriminates": not ambiguous,
        "classified": value,
    }


def set_pattern(record, name, value, *, apart=128, vectors=2792,
                diagnostic_matched=False):
    """Set one pattern's verdict AND the evidence block that backs it.

    A probe record is a document, and the licence rule reads BOTH halves: the
    word in ``patterns`` and the measurement in ``detail``. Mutating the word
    alone leaves the detail describing the previous verdict, and the rule
    (correctly) refuses that contradiction — so a test that wants to drive a
    VERDICT must move both halves, and only a test that wants to drive a
    CONTRADICTION should move one. That is what this helper and
    :func:`set_contradictory_ambiguity` are for; shared with the sibling
    tranches' suites, which drive the same rule over a longer pattern list.

    ``diagnostic_matched`` makes a DIAGNOSTIC arm reproduce the bytes too — the
    shape a comparison has when the known-wrong transcription passes it, which
    licenses nothing however the pattern is labelled.
    """
    detail = _pattern_detail(value, apart=apart)
    if diagnostic_matched:
        detail["matches"]["PLANEWISE_diagnostic"] = True
        detail["mismatch_words"]["PLANEWISE_diagnostic"] = 0
    record.setdefault("patterns", {})[name] = value
    record.setdefault("detail", {})[name] = detail
    record.setdefault("vectors", {})[name] = vectors
    return record


def set_contradictory_ambiguity(record, name, *, apart=128, vectors=2792):
    """Label a pattern :data:`AMBIGUOUS_BOTH` while its own detail measures the
    licensable arms ``apart`` words APART.

    THE ADVERSARIAL INPUT the exclusion clause exists to be right about. A
    pattern that CANNOT discriminate and a pattern that DISCRIMINATES AND
    DISAGREES are opposite situations: the first constrains nothing and is
    excluded from the agreement test, the second is evidence against an arm and
    must veto. A record asserting the first while measuring the second is asking
    to be believed on the word rather than the measurement, and a rule that
    obliged would license an arm the platform demonstrably does not implement.
    """
    record.setdefault("patterns", {})[name] = AMBIGUOUS
    detail = _pattern_detail("FMA_V1", apart=apart)
    detail["classified"] = AMBIGUOUS
    detail["matches"]["NAIVE"] = True
    detail["mismatch_words"]["NAIVE"] = 0
    record.setdefault("detail", {})[name] = detail
    record.setdefault("vectors", {})[name] = vectors
    return record


# ---------------------------------------------------------------------------
# Importing the module: the optional dependency stays optional
# ---------------------------------------------------------------------------

def _block_triton(monkeypatch):
    """Make ``import triton`` raise, the way a machine without it would."""
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "triton" or name.startswith("triton."):
            raise ImportError("triton is not installed (simulated)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    for name in list(sys.modules):
        if name == "triton" or name.startswith("triton."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    for name in (MODULE_NAME, "meep_gpu.triton_kernels.kernels",
                 "meep_gpu.triton_kernels"):
        monkeypatch.delitem(sys.modules, name, raising=False)


def test_the_package_imports_without_triton_and_does_not_pull_in_complex_fields(
        monkeypatch):
    """Dispatch is deferred, so the package must not eagerly import this module.

    ``meep_gpu.triton_kernels`` is imported by the engine's capability checks and
    has to answer on a machine that has never heard of Triton; this module is
    reached by tests and the gate through a DIRECT import only.
    """
    _block_triton(monkeypatch)
    package = importlib.import_module("meep_gpu.triton_kernels")
    assert package.triton_available() is False
    assert MODULE_NAME not in sys.modules, (
        "importing the package pulled in complex_fields; wiring is deferred and "
        "must stay a deliberate, gate-re-running change")


def test_the_module_answers_coverage_but_its_kernels_fail_clearly_without_triton(
        monkeypatch):
    """The predicates are optional-dependency-safe; launching a kernel is not."""
    _block_triton(monkeypatch)
    module = importlib.import_module(MODULE_NAME)
    fields, pml = _build()
    verdict = module.complex_pml_curl_coverage(fields, pml, "step_B",
                                               probe=_probe_record())
    assert verdict.covered is False
    assert len(verdict.reasons) == 1 and "cupy" in verdict.reasons[0], verdict.reasons
    with pytest.raises(ImportError, match="optional.*triton|triton.*optional"):
        module.bloch_pml_curl_step[(1,)]
    with pytest.raises(ImportError, match="optional.*triton|triton.*optional"):
        module.bloch_constitutive_step[(1,)]


def test_the_shipped_source_declares_module_level_jit_kernels():
    """The real decorators remain, with the explicit unavailable-launch sentinel."""
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "    import triton\n" in source
    assert "    import triton.language as tl\n" in source
    assert "except ImportError" in source
    assert "class _UnavailableKernel" in source
    assert "\n@triton.jit\ndef bloch_pml_curl_step(" in source
    assert "\n@triton.jit\ndef bloch_constitutive_step(" in source


def test_the_module_docstring_states_the_wiring_it_actually_has():
    """The docstring must not tell a reader the licensing hole is unreachable.

    It said "NOT WIRED ... fastpath.plan_fast_path still returns None on every
    branch" and a test asserted that sentence was present, which made the stale
    claim load-bearing on a green suite. Measured 2026-08-15: ``launch.plan_step``
    on a complex64 Grid/Fields/PML triple emits per-arm reasons prefixed
    'complex PML: ', i.e. it calls this module's predicates, and ``fastpath``
    forwards the probe artifact into that composer. So the assertion is inverted:
    the retired sentence must be GONE, and the live seams must be named.
    """
    module = importlib.import_module(MODULE_NAME)
    doc = module.__doc__
    # The retired ASSERTIONS, not the string "NOT WIRED" — the corrected text
    # quotes the old claim to say it was wrong, which is the point of keeping it.
    assert "Production dispatch is untouched" not in doc
    assert "plan_fast_path`` still" not in doc
    assert "composition is DEFERRED" not in doc
    # ...and the live seams, named.
    assert doc.lstrip().startswith("Complex-field")
    assert "WIRED, as of the plan_step integration." in doc
    assert "launch.plan_step" in doc and "fastpath._decide" in doc


# ---------------------------------------------------------------------------
# The predicates — real Grid/Fields/PML on NumPy
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def complex_module():
    """The shipped module itself must be importable without a test-only stub."""
    return importlib.import_module(MODULE_NAME)


def _build(cell_size=(0.8, 0.8, 0.8), pml_thickness=2, complex_storage=True,
           storage=True, k_point=(0.3, 0.0, 0.0), **grid_kwargs):
    """A real Grid/Fields/PML triple on NumPy.

    Default: the tranche's own target class — complex64 storage, an active PML,
    a Bloch phase on x. Every refusal test perturbs exactly one clause off this.
    """
    grid = Grid(resolution=10.0, cell_size=cell_size, k_point=k_point,
                **grid_kwargs)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    if storage:
        fields.enable_pml_storage()
    return fields, PML(grid=grid, thickness=pml_thickness)


def _curl_reasons(module, fields, pml, sub_step="step_B", probe="default"):
    """Curl refusal reasons with the backend clause dropped — a laptop has no CuPy.

    The filter matches the backend clause's own phrasing, not the bare word
    "cupy", because the probe refusals legitimately NAME the cupy backend.
    """
    record = _probe_record() if probe == "default" else probe
    verdict = module.complex_pml_curl_coverage(fields, pml, sub_step, probe=record)
    return [r for r in verdict.reasons if "array module" not in r]


def _constitutive_reasons(module, fields, pml, side="E", probe="default"):
    record = _probe_record() if probe == "default" else probe
    verdict = module.complex_constitutive_coverage(fields, pml, side, probe=record)
    return [r for r in verdict.reasons if "array module" not in r]


# --- the positive verdict ---------------------------------------------------

@pytest.mark.parametrize("sub_step", ["step_B", "step_D"])
def test_the_bloch_configuration_is_refused_only_for_the_backend(complex_module,
                                                                 sub_step):
    """The configuration this tranche is FOR: complex storage, active PML, k on x.

    Every clause but CuPy must pass on both curl sub-steps — the corpus class of
    binary_grating_oblique / mode_coeff_phase / oblique-planewave.
    """
    fields, pml = _build()
    assert _curl_reasons(complex_module, fields, pml, sub_step) == []


@pytest.mark.parametrize("side", ["H", "E"])
def test_both_constitutive_sides_are_refused_only_for_the_backend(complex_module,
                                                                  side):
    fields, pml = _build()
    assert _constitutive_reasons(complex_module, fields, pml, side) == []


def test_unphased_complex_storage_is_admitted(complex_module):
    """has_bloch False is ADMITTED: the wvg-src/solve-cw storage class.

    With every phase None the kernel compiles PH* = 0 and must reduce to the
    plain complex path; the gate's unphased rows are that reduction measured.
    """
    fields, pml = _build(k_point=(0.0, 0.0, 0.0))
    assert fields.grid.has_bloch is False
    assert _curl_reasons(complex_module, fields, pml) == []
    assert _constitutive_reasons(complex_module, fields, pml, "E") == []


def test_metallic_termination_at_k_zero_is_admitted(complex_module):
    """Metallic walls with k = 0 everywhere: composition case 1's boundary set."""
    fields, pml = _build(k_point=(0.0, 0.0, 0.0), boundaries="metallic")
    assert _curl_reasons(complex_module, fields, pml) == []


# --- the refusals, one per silent-wrong-answer surface ----------------------

def test_real_storage_at_k_zero_is_refused(complex_module):
    """The complement of the shipped kernels' clause 2: a real run is theirs."""
    fields, pml = _build(complex_storage=False, k_point=(0.0, 0.0, 0.0))
    reasons = _curl_reasons(complex_module, fields, pml)
    assert any("real float32" in r for r in reasons), reasons


def test_real_storage_with_nonzero_k_is_refused_by_dtype(complex_module):
    """The array path raises on this configuration (stepping.py:1893-1908);
    here the dtype inspection refuses it before a kernel could mis-stride it."""
    fields, pml = _build(complex_storage=False, k_point=(0.3, 0.0, 0.0))
    reasons = _curl_reasons(complex_module, fields, pml)
    assert any("complex64" in r for r in reasons), reasons


def test_an_inactive_layer_is_refused(complex_module):
    fields, pml = _build(pml_thickness=0)
    reasons = _curl_reasons(complex_module, fields, pml)
    assert any("PML" in r for r in reasons), reasons


def test_a_mirror_plane_is_refused(complex_module):
    """solve-cw's Mirror stays with symmetry.py; only its storage class is here."""
    fields, pml = _build(k_point=(0.0, 0.0, 0.0), symmetry=("X",))
    reasons = _curl_reasons(complex_module, fields, pml)
    assert any("mirror" in r for r in reasons), reasons


def test_a_metallic_axis_with_a_nonzero_k_component_is_refused(complex_module):
    """The extra clause: matches stepping._bloch_phases's raise (grid refuses the
    pairing at construction, so the k_point is planted past validation — a
    mis-bound PH constexpr is a plane of wrong values, not a crash)."""
    fields, pml = _build(k_point=(0.0, 0.0, 0.0), boundaries="metallic")
    fields.grid.k_point = (0.3, 0.0, 0.0)
    reasons = _curl_reasons(complex_module, fields, pml)
    assert any("metallic" in r and "k component" in r for r in reasons), reasons


def test_a_conductivity_is_refused_by_name(complex_module):
    """Conductive complex stepping is a separate, unbuilt product."""
    fields, pml = _build()
    fields.set_d_conductivity(numpy.full(fields.grid.shape, 0.5,
                                         dtype=numpy.float32))
    reasons = _curl_reasons(complex_module, fields, pml)
    assert any("conductivity" in r for r in reasons), reasons


def test_an_unreadable_bloch_phase_is_refused_not_admitted_as_unphased(complex_module):
    """An unreadable phase is not an unphased one (2026-08-11 coverage audit):
    admitting a raising ``grid.bloch_phase`` through ``_call``'s default let
    ``bloch_phase_table`` raise into the plan builder, violating its None-only
    refusal contract. The refusal must fire in every predicate variant."""
    fields, pml = _build()

    def corrupt(axis):
        raise RuntimeError("phase table corrupt")

    fields.grid.bloch_phase = corrupt
    reasons = _curl_reasons(complex_module, fields, pml)
    assert any("bloch_phase" in r and "raised" in r for r in reasons), reasons
    assert any("bloch_phase" in r and "raised" in r
               for r in _constitutive_reasons(complex_module, fields, pml, "E"))
    # The refusing predicate is what keeps the plan builder's contract: None,
    # never a raise out of bloch_phase_table.
    assert complex_module.plan_complex_pml_curl(
        fields, pml, "step_B", probe=_probe_record()) is None


def test_a_non_callable_bloch_phase_is_refused(complex_module):
    """A shadowed / non-callable phase accessor is refused by name, never read
    as "no phase" (the second admission probe of the 2026-08-11 audit)."""
    fields, pml = _build()
    fields.grid.bloch_phase = None
    reasons = _curl_reasons(complex_module, fields, pml)
    assert any("bloch_phase is missing or not callable" in r
               for r in reasons), reasons


def test_a_missing_condfac_reader_is_refused_not_read_as_no_conductivity(
        complex_module):
    """Admission by attribute absence is the reasoning coverage exists to refuse:
    with the reader gone no flag distinguishes an ELECTRIC conductivity at all
    (2026-08-11 audit planted one behind a shadowed reader and was admitted).
    Deliberately stricter than coverage.py's magnetic-flag fallback."""
    fields, pml = _build()
    fields.condfac_for = None
    reasons = _curl_reasons(complex_module, fields, pml)
    assert any("condfac_for" in r for r in reasons), reasons
    assert any("condfac_for" in r
               for r in _constitutive_reasons(complex_module, fields, pml, "H"))


def test_a_registered_susceptibility_is_refused_outright(complex_module):
    """Dispersion x complex has NO corpus demand; complex ADE is a future tranche."""
    fields, pml = _build()
    pole = SimpleNamespace(
        driven=lambda: ("Ex",),
        drives=lambda name: name == "Ex",
        susceptibility=SimpleNamespace(kind="lorentzian"),
    )
    fields.polarizations = [pole]
    reasons = _curl_reasons(complex_module, fields, pml)
    assert any("susceptibility is registered" in r for r in reasons), reasons
    assert any("susceptibility is registered" in r
               for r in _constitutive_reasons(complex_module, fields, pml, "E"))


def test_special_kz_beta_is_refused_and_named_as_not_bloch(complex_module):
    """refl-angular-kz2d / parallel-wvgs-force are grid.beta, NOT bloch_phase."""
    fields, pml = _build(k_point=(0.0, 0.0, 0.0))
    fields.grid.beta = 0.25
    reasons = _curl_reasons(complex_module, fields, pml)
    assert any("beta" in r for r in reasons), reasons


def test_bfast_is_refused(complex_module):
    fields, pml = _build(k_point=(0.0, 0.0, 0.0))
    fields.grid.bfast_scaled_k = (0.0, 0.0, 0.5)
    if fields.grid.bfast_active:
        reasons = _curl_reasons(complex_module, fields, pml)
        assert any("BFAST" in r for r in reasons), reasons


def test_a_nonlinearity_is_refused(complex_module, monkeypatch):
    fields, pml = _build()
    monkeypatch.setattr(Fields, "has_nonlinearity", property(lambda self: True))
    reasons = _curl_reasons(complex_module, fields, pml)
    assert any("chi2/chi3" in r for r in reasons), reasons


def test_cylindrical_coordinates_are_refused(complex_module):
    """disc_extraction_efficiency / point_dipole_cyl are cylindrical_triton's scope."""
    fields, pml = _build(k_point=(0.0, 0.0, 0.0))
    fields.grid.cylindrical = True
    reasons = _curl_reasons(complex_module, fields, pml)
    assert any("cylindrical" in r for r in reasons), reasons


def test_off_diagonal_epsilon_splits_curl_from_constitutive(complex_module):
    """Admitted by the curl (constitutive-only feature, stepping.py:1001-1008),
    refused by the E side (the row product reads neighbours) — the same
    per-sub-step split coverage.py makes, pinned as an asymmetry."""
    fields, pml = _build()
    fields._chi1inv_offdiagonal = {"Ex": {"Ey": numpy.full(
        fields.grid.shape, 0.1, dtype=numpy.float32)}}
    assert fields.has_offdiagonal_epsilon is True
    assert _curl_reasons(complex_module, fields, pml) == []
    reasons = _constitutive_reasons(complex_module, fields, pml, "E")
    assert any("off-diagonal" in r for r in reasons), reasons
    assert _constitutive_reasons(complex_module, fields, pml, "H") == []


def test_a_missing_probe_artifact_is_a_refusal_by_name(complex_module, monkeypatch):
    """Clause 13: the EXPANSION constexpr may not be guessed."""
    monkeypatch.delenv(complex_module.PROBE_PATH_ENVIRONMENT, raising=False)
    fields, pml = _build()
    reasons = _curl_reasons(complex_module, fields, pml, probe=None)
    assert any("probe" in r for r in reasons), reasons


def test_an_ambiguous_probe_artifact_is_a_refusal_by_name(complex_module):
    record = _probe_record()
    record["patterns"]["c8_mul_c8"] = "NAIVE"
    fields, pml = _build()
    reasons = _curl_reasons(complex_module, fields, pml, probe=record)
    assert any("ambiguous" in r for r in reasons), reasons


def test_a_non_contiguous_complex_volume_is_refused(complex_module):
    fields, pml = _build()
    fields.Bx = fields.Bx[::-1]
    reasons = _curl_reasons(complex_module, fields, pml)
    assert any("C-contiguous" in r for r in reasons), reasons


def test_the_word_offset_bound_is_half_the_real_paths(complex_module):
    """Word addressing is ``2*idx`` in int32, so ``2*ncells`` must stay below
    ``2**31`` — the real path's bound halved (coverage.py:622-626)."""
    fake = SimpleNamespace()
    reasons = complex_module._complex_layout_reasons(fake, (1024, 1024, 1024), ())
    assert any("int32" in r and "word" in r for r in reasons), reasons
    assert complex_module._complex_layout_reasons(fake, (512, 1024, 1024), ()) == []


def test_an_unknown_sub_step_or_side_raises_rather_than_refusing(complex_module):
    """A typo must not read as "not covered" — that is how a kernel goes unused."""
    fields, pml = _build()
    with pytest.raises(ValueError, match="sub_step"):
        complex_module.complex_pml_curl_coverage(fields, pml, "step_H")
    with pytest.raises(ValueError, match="side"):
        complex_module.complex_constitutive_coverage(fields, pml, "B")


# --- predicate mutation: the clause with no other guard ---------------------

def _mutate(module, function, needle: str, replacement: str):
    """Recompile one predicate with one clause rewritten, in the module's namespace."""
    import inspect  # noqa: PLC0415
    import textwrap  # noqa: PLC0415

    source = textwrap.dedent(inspect.getsource(function))
    mutated = source.replace(needle, replacement)
    assert mutated != source, f"the clause moved; update this mutation: {needle!r}"
    namespace = dict(module.__dict__)
    exec(compile(mutated, "<mutated complex_fields>", "exec"), namespace)  # noqa: S102
    return namespace[function.__name__]


def test_mutation_dropping_the_metallic_k_clause_is_caught(complex_module,
                                                           monkeypatch):
    """Nothing else refuses a planted k on a metallic axis (the grid's own guard
    was bypassed and has_bloch is stale-False, so the phase-consistency read
    returns None) — so with the clause deleted the configuration must be
    ADMITTED, which is what makes the clause load-bearing rather than decorative."""
    fields, pml = _build(k_point=(0.0, 0.0, 0.0), boundaries="metallic")
    fields.grid.k_point = (0.3, 0.0, 0.0)
    assert any("k component" in r for r in _curl_reasons(complex_module, fields, pml))

    mutated = _mutate(
        complex_module, complex_module._complex_grid_reasons,
        'if kind == "metallic" and float(k_point[axis]) != 0.0:',
        "if False:")
    monkeypatch.setattr(complex_module, "_complex_grid_reasons", mutated)
    assert _curl_reasons(complex_module, fields, pml) == [], (
        "the metallic-k clause is the only guard for a planted k on a walled "
        "axis; if this fails, the mutation is masked and the clause unmeasured")


# ---------------------------------------------------------------------------
# The probe artifact
# ---------------------------------------------------------------------------

def test_expansion_from_probe_reads_an_agreeing_record(complex_module):
    assert complex_module.expansion_from_probe(_probe_record("FMA_V1")) == 1
    assert complex_module.expansion_from_probe(_probe_record("NAIVE")) == 0


@pytest.mark.parametrize("record", [
    None,
    {},
    _probe_record(backend="numpy"),                      # wrong backend
    _probe_record("SOMETHING_ELSE"),                     # unknown arm
    {"backend": "cupy", "patterns": {"c8_mul_c8": "FMA_V1"}},  # incomplete
], ids=["none", "empty", "wrong-backend", "unknown-arm", "incomplete"])
def test_expansion_from_probe_refuses_unusable_records(complex_module, record):
    assert complex_module.expansion_from_probe(record) is None


def test_expansion_from_probe_refuses_disagreement(complex_module):
    record = _probe_record()
    record["patterns"]["f4_mul_c8_coefficient_left"] = "NAIVE"
    assert complex_module.expansion_from_probe(record) is None


# ---------------------------------------------------------------------------
# The licensing rule when the probe cannot discriminate
# ---------------------------------------------------------------------------
#
# MEASURED, results/complex_expansion_flush_coincidence_2026-08-15/: under the
# shipped flush policy the two licensable arms were BIT-IDENTICAL on the three
# mixed-dtype patterns (0 of 4560 words each) because flush destroys exactly the
# subnormal lanes that separated them — on the device AND in the candidates — and
# they still differ on c8_mul_c8 (1088 words), where the platform matches FMA_V1
# uniquely. The refusal those patterns used to produce ("no single constexpr is
# licensed") was an artifact of scoring a flushed device against kept candidates.
#
# THAT IS A FACT ABOUT THAT ROUND'S VECTORS, and the probe's vectors have since
# changed: _NORMAL_UNDERFLOWING_ZR separates the arms without handing the device
# a subnormal operand, so the mixed patterns now measure 128 words apart under
# flush and the gate REFUSES to emit a blind record at all. The non-discriminating
# path these tests drive is therefore no longer reachable from the shipped probe
# — it is reachable from a hand-written or foreign artifact, which is exactly the
# input the licence rule has to be right about, so the tests stay.

_MEASURED_ENVIRONMENT = {"backend": "cupy", "machine": "x86_64",
                         "cupy_version": "13.5.1"}


def _policy_record(patterns, *, resolved="flush", candidate_policy="flush",
                   environment=None, evidence=True):
    """A probe record in the shape the reworked gate writes.

    ``evidence=False`` strips the per-pattern ``vectors``/``detail`` blocks — the
    shape a hand-assembled or truncated artifact has, and the one that used to
    license a comparison with no discriminating power at all.
    """
    record = {
        "backend": "cupy",
        "patterns": dict(patterns),
        "subnormal_policy": {"policy": "meep_x86_flush", "resolved": resolved},
        "candidates": {"policy": candidate_policy},
    }
    if evidence:
        record["vectors"] = {name: 2792 for name in patterns}
        record["detail"] = {name: _pattern_detail(value)
                            for name, value in patterns.items()}
    if environment is not None:
        record["environment"] = dict(environment)
    return record


def _all_ambiguous(complex_module):
    return {name: AMBIGUOUS for name in complex_module.PROBE_PATTERNS}


@pytest.mark.parametrize("verdict_value", ["NEITHER", "PLANEWISE_diagnostic",
                                           "DISAGREES_ACROSS_SCALARS"])
def test_a_platform_matching_no_arm_still_refuses(complex_module, verdict_value):
    """NON-NEGOTIABLE. The probe exists because binding a wrong constexpr yields
    a kernel that differs from the array path; softening AMBIGUOUS_BOTH must not
    soften a genuine "cannot tell". One unlicensable pattern refuses the record
    even when every other pattern is licensable and the environment is known."""
    patterns = _all_ambiguous(complex_module)
    patterns["c8_mul_f4_field_left"] = verdict_value
    verdict = complex_module.expansion_license(
        _policy_record(patterns, environment=_MEASURED_ENVIRONMENT))
    assert verdict["expansion"] is None
    assert any("c8_mul_f4_field_left" in reason and verdict_value in reason
               for reason in verdict["refusals"]), verdict["refusals"]
    # ...and a platform that matches no arm anywhere refuses under EITHER policy.
    for resolved in ("flush", "keep"):
        record = _policy_record({name: "NEITHER"
                                 for name in complex_module.PROBE_PATTERNS},
                                resolved=resolved, candidate_policy=resolved,
                                environment=_MEASURED_ENVIRONMENT)
        assert complex_module.expansion_from_probe(record) is None


def test_a_pattern_that_cannot_discriminate_does_not_veto_the_ones_that_can(
        complex_module):
    """The measured shipped-flush platform. c8_mul_c8 names FMA_V1 uniquely; the
    three mixed patterns cannot tell the arms apart. The arm is MEASURED here —
    the environment table is not consulted at all, which is why an unknown
    environment changes nothing."""
    patterns = _all_ambiguous(complex_module)
    patterns["c8_mul_c8"] = "FMA_V1"
    for environment in (None, _MEASURED_ENVIRONMENT,
                        {"backend": "cupy", "machine": "riscv64",
                         "cupy_version": "99.0"}):
        verdict = complex_module.expansion_license(
            _policy_record(patterns, environment=environment))
        assert verdict["expansion"] == complex_module.EXPANSIONS["FMA_V1"]
        assert verdict["arm"] == "FMA_V1"
        assert verdict["basis"] == "measured"
        assert verdict["environment_default"] is None
        assert verdict["discriminating"] == {"c8_mul_c8": "FMA_V1"}
        assert verdict["non_discriminating"] == [
            "c8_mul_f4_field_left", "f4_mul_c8_coefficient_left",
            "python_float_left"]
        assert verdict["arms_coincide_on_every_pattern"] is False
        assert verdict["refusals"] == []


def test_arms_that_coincide_everywhere_license_with_the_ambiguity_recorded(
        complex_module):
    """Nothing discriminated, so the arm is DEFAULTED from the environment it was
    already measured in — and the record says so, with the artifact that measured
    it and the reason the choice cannot change the emitted bits here."""
    verdict = complex_module.expansion_license(
        _policy_record(_all_ambiguous(complex_module),
                       environment=_MEASURED_ENVIRONMENT))
    assert verdict["arm"] == "FMA_V1"
    assert verdict["expansion"] == complex_module.EXPANSIONS["FMA_V1"]
    assert verdict["basis"] == "environment_default"
    assert verdict["discriminating"] == {}
    assert verdict["arms_coincide_on_every_pattern"] is True
    row = verdict["environment_default"]
    assert row["measured_under_policy"] == "ieee_keep_ftz_stripped"
    assert row["artifact"].endswith("complex_expansion_diagnosis_2026-08-11/"
                                    "gate_stripped.json")
    assert "not a measurement of this run" in verdict["why_arbitrary"]
    assert row["artifact"] in verdict["why_arbitrary"]


@pytest.mark.parametrize("environment", [
    None,
    {},
    dict(_MEASURED_ENVIRONMENT, cupy_version="14.0.0"),
    dict(_MEASURED_ENVIRONMENT, machine="aarch64"),
    {"machine": "x86_64", "cupy_version": "13.5.1"},          # no backend key
], ids=["absent", "empty", "other-cupy", "other-arch", "partial"])
def test_an_unmeasured_environment_refuses_instead_of_guessing(complex_module,
                                                               environment):
    """The table is a record of what was measured where, never a blanket
    default: an unknown platform is exactly where a default would be a guess."""
    verdict = complex_module.expansion_license(
        _policy_record(_all_ambiguous(complex_module), environment=environment))
    assert verdict["expansion"] is None
    assert any("matches no measured row" in reason
               for reason in verdict["refusals"]), verdict["refusals"]


def test_two_discriminating_patterns_that_disagree_still_refuse(complex_module):
    patterns = _all_ambiguous(complex_module)
    patterns["c8_mul_c8"] = "FMA_V1"
    patterns["python_float_left"] = "NAIVE"
    verdict = complex_module.expansion_license(
        _policy_record(patterns, environment=_MEASURED_ENVIRONMENT))
    assert verdict["expansion"] is None
    assert any("disagree" in reason for reason in verdict["refusals"])


def test_ambiguity_measured_across_a_policy_boundary_is_not_ambiguity(
        complex_module):
    """AMBIGUOUS_BOTH is only readable when the candidates were cut under the
    policy the platform's bytes were cut under. Scoring a flushed device against
    kept candidates is the 54/48/125-word NEITHER of 2026-08-11 — a broken
    comparison, and a record that cannot rule it out licenses nothing."""
    patterns = _all_ambiguous(complex_module)
    patterns["c8_mul_c8"] = "FMA_V1"
    mismatched = _policy_record(patterns, resolved="flush",
                                candidate_policy="keep")
    verdict = complex_module.expansion_license(mismatched)
    assert verdict["expansion"] is None
    assert any("broken comparison" in reason for reason in verdict["refusals"])

    undeclared = _policy_record(patterns)
    undeclared.pop("candidates")
    assert complex_module.expansion_from_probe(undeclared) is None

    unstamped = _policy_record(patterns)
    unstamped.pop("subnormal_policy")
    verdict = complex_module.expansion_license(unstamped)
    assert verdict["expansion"] is None
    assert any("no resolved subnormal policy" in reason
               for reason in verdict["refusals"])

    # AND ON THE ALL-DISCRIMINATING PATH TOO. This used to read "a record on
    # which EVERY pattern discriminates needs no such declaration: a policy
    # mismatch cannot produce one (it produces NEITHER), so the artifacts cut
    # before the policy was stated stay licensable" — and asserted a record with
    # no policy block at all licensed FMA_V1. That argument is true of an
    # honestly measured record and false of a hand-assembled, truncated or edited
    # one, and the artifact is a path named by an environment variable. Driven
    # 2026-08-15: an all-FMA_V1 record declaring resolved='keep' against
    # candidates.policy='flush' was licensed 'measured' with zero refusals, as was
    # one carrying no policy block at all — the exact artifact
    # gate_triton_complex refuses BY NAME as "from before artifacts stated their
    # policy". The guard is now unconditional.
    every_pattern_names_an_arm = {name: "FMA_V1"
                                  for name in complex_module.PROBE_PATTERNS}
    contradictory = _policy_record(every_pattern_names_an_arm,
                                   resolved="keep", candidate_policy="flush")
    verdict = complex_module.expansion_license(contradictory)
    assert verdict["expansion"] is None
    assert any("broken comparison" in reason for reason in verdict["refusals"])

    unstamped_all_arms = _policy_record(every_pattern_names_an_arm)
    unstamped_all_arms.pop("subnormal_policy")
    verdict = complex_module.expansion_license(unstamped_all_arms)
    assert verdict["expansion"] is None
    assert any("no resolved subnormal policy" in reason
               for reason in verdict["refusals"])

    # ...and the honest record on the same path still licenses.
    assert complex_module.expansion_from_probe(_probe_record("FMA_V1")) == 1


def test_the_licence_states_the_policy_it_was_cut_under_and_does_not_transfer(
        complex_module):
    patterns = _all_ambiguous(complex_module)
    patterns["c8_mul_c8"] = "FMA_V1"
    verdict = complex_module.expansion_license(_policy_record(patterns))
    assert verdict["policy"] == "meep_x86_flush"
    assert verdict["policy_resolved"] == "flush"
    assert verdict["candidate_policy"] == "flush"
    assert "does not transfer" in verdict["policy_conditional"]
    assert verdict["policy_conditional"] == complex_module.POLICY_CONDITIONAL_LICENCE


# ---------------------------------------------------------------------------
# "Does not transfer" as BEHAVIOUR, not as a string in the verdict
# ---------------------------------------------------------------------------
#
# The five assertions above are all about strings the verdict carries. The audit
# of 2026-08-15 grepped this file for a test that installs or simulates a policy
# and expects a REFUSAL and found none — which is how a licence cut under one
# policy stayed consumable under the other through a fully green suite. These are
# that test.

@pytest.mark.parametrize("cut_under,run_under,refuses", [
    ("keep", "keep", False),
    ("flush", "flush", False),
    ("flush", "keep", True),      # <- the shipped x86 pairing, by construction
    ("keep", "flush", True),
])
def test_a_licence_cut_under_one_policy_is_refused_by_a_run_under_the_other(
        complex_module, cut_under, run_under, refuses):
    """The whole point of POLICY_CONDITIONAL_LICENCE, enforced.

    'flush' cut / 'keep' run is not a hypothetical: on shipped x86 CuPy appends
    -ftz=true to every NVRTC compile, so the artifact the platform naturally cuts
    is stamped 'flush', while dispatch requires and installs 'keep'. Before this
    check the flush-cut record was consumed with zero refusals by that run — under
    a policy where the arms are measured ~24.7% of words apart."""
    record = _probe_record(policy=f"stamp_{cut_under}", resolved=cut_under)
    reasons = complex_module.expansion_policy_reasons(record, run_under)
    assert bool(reasons) is refuses, reasons
    if refuses:
        assert cut_under in reasons[0] and run_under in reasons[0]
        assert "does not transfer" in reasons[0]


def test_a_record_with_no_policy_stamp_transfers_nowhere(complex_module):
    record = _probe_record()
    record.pop("subnormal_policy")
    for policy in ("keep", "flush"):
        reasons = complex_module.expansion_policy_reasons(record, policy)
        assert reasons and "states no resolved subnormal policy" in reasons[0]


def test_the_policy_check_asserts_nothing_when_the_policy_cannot_be_read(
        complex_module):
    """Three-valued, like the device check: an unread policy is not a refusal.

    ``fastpath`` consumes the probe four rungs BEFORE it installs a policy, so at
    the library seam there is often nothing installed to compare against.
    Refusing there would refuse every honest run; asserting a policy there would
    be inventing one. ``None`` means 'not checked here' — and fastpath passes its
    own required policy explicitly rather than relying on this."""
    assert complex_module.expansion_policy_reasons(_probe_record(), None) == []
    assert complex_module.expansion_policy_reasons(None, "keep") == []


def test_the_coverage_seam_applies_the_policy_check_when_a_policy_is_installed(
        complex_module, monkeypatch):
    """``_expansion_reasons`` is what the predicates call, so the check has to
    land THERE to keep an unlicensed arm out of a plan — and ``_resolve_expansion``
    must refuse in lockstep, or the reasons and the bound constexpr disagree."""
    keep_cut = _probe_record(policy="ieee_keep_ftz_stripped", resolved="keep")
    monkeypatch.setattr(complex_module, "_policy_in_force", lambda: "keep")
    assert complex_module._expansion_reasons(keep_cut) == []
    assert complex_module._resolve_expansion(keep_cut) == 1

    monkeypatch.setattr(complex_module, "_policy_in_force", lambda: "flush")
    reasons = complex_module._expansion_reasons(keep_cut)
    assert reasons and "does not transfer" in reasons[0]
    assert complex_module._resolve_expansion(keep_cut) is None, (
        "the plan seam bound a constexpr the coverage seam refused")


# ---------------------------------------------------------------------------
# A comparison with no discriminating power licenses nothing
# ---------------------------------------------------------------------------

def test_an_empty_comparison_is_not_a_coincidence(complex_module):
    """``count_nonzero`` over an empty array is 0, so EVERY arm 'matches' an
    empty probe and every pattern classifies AMBIGUOUS_BOTH. Driven 2026-08-15:
    such a record was licensed FMA_V1, basis 'environment_default', zero
    refusals — a licence from a comparison of nothing."""
    record = _policy_record(_all_ambiguous(complex_module),
                            environment=_MEASURED_ENVIRONMENT)
    record["vectors"] = {name: 0 for name in complex_module.PROBE_PATTERNS}
    verdict = complex_module.expansion_license(record)
    assert verdict["expansion"] is None
    assert any("comparison of nothing" in reason for reason in verdict["refusals"])


def test_an_ambiguity_with_no_recorded_evidence_refuses(complex_module):
    """AMBIGUOUS_BOTH is a claim about a measurement. Asserted as a bare string,
    with no vector count and no measured arms-apart number, it is
    indistinguishable from a blind comparison — so it must be backed."""
    patterns = _all_ambiguous(complex_module)
    patterns["c8_mul_c8"] = "FMA_V1"
    verdict = complex_module.expansion_license(
        _policy_record(patterns, environment=_MEASURED_ENVIRONMENT,
                       evidence=False))
    assert verdict["expansion"] is None
    assert verdict["refusals"]


def test_an_ambiguity_that_contradicts_its_own_detail_block_refuses(
        complex_module):
    """The record says the arms coincide; its own detail block says they are 31
    words apart. One of the two is wrong and neither licenses."""
    patterns = _all_ambiguous(complex_module)
    patterns["c8_mul_c8"] = "FMA_V1"
    record = _policy_record(patterns, environment=_MEASURED_ENVIRONMENT)
    record["detail"]["python_float_left"]["licensable_arms_disagreement_words"] = 31
    verdict = complex_module.expansion_license(record)
    assert verdict["expansion"] is None
    assert any("words APART" in reason for reason in verdict["refusals"])


def test_a_comparison_the_planewise_arm_also_passes_licenses_nothing(
        complex_module):
    """Plane-wise is the transcription the zero cross terms exist to rule out.
    If it reproduces the platform's bytes too, the comparison separated nothing
    at all — driven 2026-08-15, such a record was licensed 'measured' with no
    flag, because the verdict never read the detail block."""
    patterns = _all_ambiguous(complex_module)
    patterns["c8_mul_c8"] = "FMA_V1"
    record = _policy_record(patterns, environment=_MEASURED_ENVIRONMENT)
    record["detail"]["c8_mul_f4_field_left"]["matches"]["PLANEWISE_diagnostic"] = True
    record["detail"]["c8_mul_f4_field_left"]["mismatch_words"]["PLANEWISE_diagnostic"] = 0
    verdict = complex_module.expansion_license(record)
    assert verdict["expansion"] is None
    assert any("known-wrong transcription" in reason
               for reason in verdict["refusals"])


# ---------------------------------------------------------------------------
# A malformed artifact refuses by name and never raises
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [["FMA_V1"], {"arm": "FMA_V1"}, True, 1, 0.5],
                         ids=["list", "dict", "bool", "int", "float"])
def test_a_non_string_pattern_value_refuses_instead_of_raising(complex_module,
                                                               value):
    """``value in EXPANSIONS`` HASHES the JSON value, so a list or an object
    raised TypeError out of two functions whose written contracts promise they
    never raise into a caller ('unreadable is treated exactly like missing';
    'None is the only refusal'). Measured 2026-08-15: it took the gate's engine
    leg down with an unhandled TypeError instead of refusing by name."""
    record = _probe_record()
    record["patterns"]["c8_mul_c8"] = value
    verdict = complex_module.expansion_license(record)   # must not raise
    assert verdict["expansion"] is None
    assert any("c8_mul_c8" in reason for reason in verdict["refusals"])
    assert complex_module.expansion_from_probe(record) is None
    assert complex_module._expansion_reasons(record)
    assert complex_module._resolve_expansion(record) is None


def test_the_environment_table_rows_carry_their_own_evidence(complex_module):
    assert complex_module.ENVIRONMENT_DEFAULT_KEYS == (
        "backend", "machine", "cupy_version")
    assert complex_module.ENVIRONMENT_DEFAULTS, "an empty table refuses everything"
    for row in complex_module.ENVIRONMENT_DEFAULTS:
        assert row["expansion"] in complex_module.EXPANSIONS
        assert row["measured_under_policy"] and row["artifact"] and row["evidence"]
        keyed = {key: row[key] for key in complex_module.ENVIRONMENT_DEFAULT_KEYS}
        assert complex_module.environment_default(keyed) == row
        for key in complex_module.ENVIRONMENT_DEFAULT_KEYS:
            partial = dict(keyed)
            partial.pop(key)
            assert complex_module.environment_default(partial) is None, key
    assert complex_module.environment_default(None) is None
    assert complex_module.environment_default({}) is None


def test_load_expansion_probe_reads_the_environment_artifact(complex_module,
                                                             monkeypatch,
                                                             tmp_path):
    path = tmp_path / "probe.json"
    path.write_text(json.dumps(_probe_record()), encoding="utf-8")
    monkeypatch.setenv(complex_module.PROBE_PATH_ENVIRONMENT, str(path))
    record = complex_module.load_expansion_probe()
    assert complex_module.expansion_from_probe(record) == 1
    # Unreadable is missing, never a default expansion.
    path.write_text("not json", encoding="utf-8")
    assert complex_module.load_expansion_probe() is None


# ---------------------------------------------------------------------------
# The phase table and its argument encoding — NumPy-checkable host arithmetic
# ---------------------------------------------------------------------------

def test_bloch_phase_table_is_none_at_k_zero_never_one(complex_module):
    """None means SKIP the multiply; 1+0j would be a multiply, and k = 0
    bit-identity lives on the skip (stepping.py:1815-1817 / :1866-1868)."""
    fields, _ = _build(k_point=(0.0, 0.0, 0.0))
    table = complex_module.bloch_phase_table(fields.grid,
                                             ("periodic", "periodic", "periodic"))
    assert table == (None, None, None)


def test_bloch_phase_table_matches_the_grids_own_resolution(complex_module):
    fields, _ = _build()
    table = complex_module.bloch_phase_table(fields.grid,
                                             ("periodic", "periodic", "periodic"))
    assert table[0] == fields.grid.bloch_phase(0)
    assert table[1] is None and table[2] is None
    expected = cmath.exp(2j * cmath.pi * 0.3 * 0.8)
    assert abs(table[0] - expected) < 1e-12


def test_the_brillouin_edge_is_exactly_minus_one(complex_module):
    """k*L = 1/2 exactly: the grid's edge test returns -1+0j EXACTLY
    (grid.py:1011-1017); going through exp instead leaves a nonzero imaginary
    word after the complex64 rounding (the gate's mutation h)."""
    fields, _ = _build(k_point=(0.625, 0.0, 0.0))
    table = complex_module.bloch_phase_table(fields.grid,
                                             ("periodic", "periodic", "periodic"))
    assert table[0] == complex(-1.0, 0.0)
    assert table[0].imag == 0.0


def test_bloch_phase_table_raises_on_a_non_periodic_phased_axis(complex_module):
    """Mirrors stepping._bloch_phases's raise (stepping.py:2401-2415); behind an
    admitting predicate it is unreachable, and a caller who skips the predicate
    is refused loudly rather than handed a mis-bound PH constexpr."""
    fields, _ = _build()
    with pytest.raises(ValueError, match="periodic"):
        complex_module.bloch_phase_table(fields.grid,
                                         ("metallic", "periodic", "periodic"))


def test_phase_arguments_round_to_complex64_before_splitting(complex_module):
    """The array path multiplies by ``shifted.dtype.type(phase)`` (stepping.py:1909):
    the python-complex phase is rounded to complex64 FIRST, then split."""
    phase = cmath.exp(2j * cmath.pi * 0.3)
    flags, values = complex_module._phase_arguments((phase, None, None),
                                                    backward=False)
    assert flags == (1, 0, 0)
    rounded = numpy.complex64(phase)
    assert values[0] == float(numpy.float32(rounded.real))
    assert values[1] == float(numpy.float32(rounded.imag))
    assert values[2:] == (1.0, 0.0, 1.0, 0.0)


def test_phase_arguments_conjugate_for_the_backward_sub_step(complex_module):
    """_shift_down multiplies by conj(phase) (stepping.py:1865-1869); taking the
    same factor both directions is the classic band-structure sign error, so the
    negated imaginary word is pinned here and the gate carries mutation m1."""
    phase = cmath.exp(2j * cmath.pi * 0.3)
    _, forward = complex_module._phase_arguments((phase, None, None), backward=False)
    _, backward = complex_module._phase_arguments((phase, None, None), backward=True)
    assert backward[0] == forward[0]
    assert backward[1] == -forward[1] and backward[1] != forward[1]


def test_an_unphased_axis_is_a_flag_not_a_multiply_by_one(complex_module):
    flags, values = complex_module._phase_arguments((None, None, None),
                                                    backward=True)
    assert flags == (0, 0, 0)
    assert values == (1.0, 0.0, 1.0, 0.0, 1.0, 0.0)


# ---------------------------------------------------------------------------
# The word view and the plans
# ---------------------------------------------------------------------------

def test_the_word_view_is_the_same_allocation_interleaved(complex_module):
    array = numpy.array([[[1.0 + 2.0j, -3.0 + 0.5j]]], dtype=numpy.complex64)
    view = complex_module._word_view(array)
    assert view.dtype == numpy.float32
    assert view.base is array
    assert view.ravel().tolist() == [1.0, 2.0, -3.0, 0.5]


def test_the_word_view_refuses_the_wrong_dtype_and_the_wrong_strides(complex_module):
    with pytest.raises(ValueError, match="complex64"):
        complex_module._word_view(numpy.zeros((2, 2, 2), dtype=numpy.float32))
    reversed_view = numpy.zeros((4, 2, 2), dtype=numpy.complex64)[::-1]
    with pytest.raises(ValueError, match="contiguous"):
        complex_module._word_view(reversed_view)


def _curl_arrays(sub_step="step_B", shape=(4, 3, 5)):
    """Bare NumPy stand-ins for the gate's device arrays."""
    rng = numpy.random.default_rng(7)

    def cvol():
        return (rng.standard_normal(shape)
                + 1j * rng.standard_normal(shape)).astype(numpy.complex64)

    targets = ("Bx", "By", "Bz") if sub_step == "step_B" else ("Dx", "Dy", "Dz")
    sources = ("Ex", "Ey", "Ez") if sub_step == "step_B" else ("Hx", "Hy", "Hz")
    arrays = {}
    for name in targets + sources:
        arrays[name] = cvol()
    for name in targets:
        arrays["fu_" + name] = cvol()
    flat = {}
    for axis, count in zip("xyz", shape):
        flat["kms_" + axis] = numpy.linspace(0.5, 1.0, count).astype(numpy.float32)
        flat["sinv_" + axis] = numpy.linspace(0.9, 1.0, count).astype(numpy.float32)
    return arrays, flat


def test_plan_builders_return_none_on_a_numpy_host(complex_module):
    """None is the only refusal — the array path, never an exception."""
    fields, pml = _build()
    assert complex_module.plan_complex_pml_curl(
        fields, pml, "step_B", probe=_probe_record()) is None
    assert complex_module.plan_complex_constitutive(
        fields, pml, "E", probe=_probe_record()) is None


def test_the_curl_plan_shape_from_arrays(complex_module):
    phase = cmath.exp(2j * cmath.pi * 0.17)
    arrays, flat = _curl_arrays("step_B")
    plan = complex_module.plan_complex_pml_curl_from_arrays(
        "step_B", arrays, flat, (0, 0, 0), (phase, None, None), 0.35, expansion=1)
    assert plan.n_elem == 4 * 3 * 5, "n_elem is COMPLEX cells, not words"
    assert plan.backward == 0
    assert plan.phased == (1, 0, 0)
    rounded = numpy.complex64(phase)
    assert plan.phase_values[0] == float(numpy.float32(rounded.real))
    assert plan.phase_values[1] == float(numpy.float32(rounded.imag))
    assert plan.expansion == 1
    assert plan.dtdx == float(0.35)
    sentinel = object()
    mutated = complex_module.plan_complex_pml_curl_from_arrays(
        "step_B", arrays, flat, (0, 0, 0), (None, None, None), 0.35,
        expansion=0, kernel=sentinel)
    assert mutated._kernel is sentinel, "the mutation override must be retained"


def test_the_step_D_plan_conjugates_the_phase_itself(complex_module):
    """The gate hands ONE phase table; the per-sub-step conjugation is the
    plan's, exactly as it is the engine route's."""
    phase = cmath.exp(2j * cmath.pi * 0.17)
    arrays, flat = _curl_arrays("step_D")
    plan = complex_module.plan_complex_pml_curl_from_arrays(
        "step_D", arrays, flat, (0, 0, 0), (phase, None, None), 0.35, expansion=1)
    assert plan.backward == 1
    rounded = numpy.complex64(phase)
    assert plan.phase_values[1] == -float(numpy.float32(rounded.imag))


def test_the_constitutive_plan_shape_from_arrays(complex_module):
    shape = (4, 3, 5)
    rng = numpy.random.default_rng(11)

    def cvol():
        return (rng.standard_normal(shape)
                + 1j * rng.standard_normal(shape)).astype(numpy.complex64)

    arrays = {}
    for name in ("Ex", "Ey", "Ez", "Dx", "Dy", "Dz"):
        arrays[name] = cvol()
    for name in ("Ex", "Ey", "Ez"):
        arrays["f_w_" + name] = cvol()
        arrays["inv_eps_" + name] = numpy.full(shape, 0.5, dtype=numpy.float32)
    flat = {}
    for axis, count in zip("xyz", shape):
        flat["kps_" + axis] = numpy.linspace(1.0, 1.5, count).astype(numpy.float32)
        flat["kms_" + axis] = numpy.linspace(0.5, 1.0, count).astype(numpy.float32)
    plan = complex_module.plan_complex_constitutive_from_arrays(
        "E", arrays, flat, expansion=1)
    assert plan.scale == 1
    assert plan.n_elem == 4 * 3 * 5

    h_arrays = {}
    for name in ("Hx", "Hy", "Hz", "Bx", "By", "Bz"):
        h_arrays[name] = cvol()
    for name in ("Hx", "Hy", "Hz"):
        h_arrays["f_w_" + name] = cvol()
    h_plan = complex_module.plan_complex_constitutive_from_arrays(
        "H", h_arrays, flat, expansion=1)
    assert h_plan.scale == 0
    # The H side binds the source word views as inv_eps placeholders — the loads
    # sit behind the SCALE constexpr, but a pointer argument still has to type.
    assert h_plan._inv_eps is h_plan._sources


def test_plans_refuse_real_storage_at_construction(complex_module):
    arrays, flat = _curl_arrays("step_B")
    arrays["Bx"] = numpy.zeros((4, 3, 5), dtype=numpy.float32)
    with pytest.raises(ValueError, match="complex64"):
        complex_module.plan_complex_pml_curl_from_arrays(
            "step_B", arrays, flat, (0, 0, 0), (None, None, None), 0.35,
            expansion=1)


# ---------------------------------------------------------------------------
# Structure — what the kernels must keep saying to stay byte-faithful
# ---------------------------------------------------------------------------

def _kernel_source_segment(name: str) -> str:
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"complex_fields.py no longer defines {name}")


def test_the_curl_grouping_is_the_one_stepping_uses_on_both_planes():
    """``((sf - f) + (s - ss))`` per plane, parenthesised — complex add/sub is
    component-wise, so the real kernel's grouping note (kernels.py:48-54)
    applies to each plane separately."""
    segment = _kernel_source_segment("bloch_pml_curl_step")
    assert "t0_re = ((c_y_re - c_re) + (b_re - b_z_re))" in segment
    assert "t0_im = ((c_y_im - c_im) + (b_im - b_z_im))" in segment
    assert "t1_re = ((a_z_re - a_re) + (c_re - c_x_re))" in segment
    assert "t1_im = ((a_z_im - a_im) + (c_im - c_x_im))" in segment
    assert "t2_re = ((b_x_re - b_re) + (a_re - a_y_re))" in segment
    assert "t2_im = ((b_x_im - b_im) + (a_im - a_y_im))" in segment


def test_the_zero_cross_terms_are_written_out_not_folded():
    """The ``0.0`` cross terms carry the array path's signed-zero semantics
    (measured 4/8 targeted patterns wrong without them) and must appear as
    products, never as literal zeros."""
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "(z_im * 0.0) * -1.0" in source    # field-left, FMA arm
    assert "(z_re * 0.0)" in source           # field-left, naive arm
    assert "(0.0 * z_im) * -1.0" in source    # coefficient-left, FMA arm
    assert "(0.0 * z_re)" in source           # coefficient-left arms
    assert "tl.math.fma" in source            # the explicit-FMA expansion arm


def test_the_negated_addends_are_spelled_times_minus_one_not_unary_minus():
    """Triton lowers unary ``-x`` as ``0.0 - x`` (3.1.0,
    language/semantic.py:386-391), which canonicalizes every ±0 addend to +0
    under round-to-nearest, while the array path negates the rounded cross
    product sign-exactly. ``* -1.0`` is the IEEE-exact negation (folds to
    neg.f32; device-measured by the special_kz m9 product-layer pin). The
    unary spelling carried until 2026-08-12 could not reproduce the array
    path's -0.0 stored words on the zero-init composition class."""
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "tl.math.fma(g_re, p_re, (g_im * p_im) * -1.0)" in source
    assert "tl.math.fma(z_re, c, (z_im * 0.0) * -1.0)" in source
    assert "tl.math.fma(c, z_re, (0.0 * z_im) * -1.0)" in source
    for stale in ("-(g_im * p_im)", "-(z_im * 0.0)", "-(0.0 * z_im)"):
        assert stale not in source, (
            f"the unary-minus addend spelling {stale!r} is back; Triton's "
            f"0.0 - x lowering loses the addend's zero sign")


def test_the_zero_coincidence_class_needs_the_addends_sign(gate_module):
    """WHY the spelling is load-bearing, in exact-fma32 arithmetic: on the
    zero-init class (a quiet +0.0 word under a negative deep-absorber
    coefficient, non-negative imag companion) the fused product is -0.0 and
    the ADDEND's zero sign decides the stored word — the sign-exact negation
    keeps -0.0, the canonicalized +0.0 addend loses it."""
    km = numpy.float32(-0.5111)  # the measured deepest 3-cell kms (res 12)
    fused = gate_module.fma32
    true_word = fused(0.0, float(km), -0.0)   # addend = -(+0.0*0.0) = -0.0
    lowered_word = fused(0.0, float(km), 0.0)  # addend = 0.0 - (+0.0) = +0.0
    assert true_word.tobytes() == numpy.float32(-0.0).tobytes()
    assert lowered_word.tobytes() == numpy.float32(0.0).tobytes()
    # The rotate-helper (line 251) form of the same class: quiet wrapped read
    # under a second-quadrant phase (p_re < 0 < p_im).
    p_re, p_im = numpy.float32(-0.4818), numpy.float32(0.8763)
    addend_true = -float(gate_module.mul32(0.0, p_im))       # -(+0.0) = -0.0
    addend_lowered = numpy.float32(0.0 - gate_module.mul32(0.0, p_im))
    rotated_true = gate_module.fma32(0.0, float(p_re), addend_true)
    rotated_lowered = gate_module.fma32(0.0, float(p_re), float(addend_lowered))
    assert rotated_true.tobytes() == numpy.float32(-0.0).tobytes()
    assert rotated_lowered.tobytes() == numpy.float32(0.0).tobytes()


def test_the_gate_mutation_needles_match_the_new_spelling(gate_module):
    """The gate's source mutations must still find their needles in the file
    text after the 2026-08-12 respelling — a silent needle miss disarms the
    legs (the gate errors on 0 hits, but the merge bar should catch the drift
    before a device run does). File text, not inspect: laptop has no Triton."""
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert gate_module.mutate_zero_cross_terms_folded(source)[1] == 8
    assert gate_module.mutate_expansion_arm_regrouped(source)[1] == 4
    assert gate_module.mutate_whole_array_phase(source)[1] > 0
    assert gate_module.mutate_wrap_plane_swapped(source)[1] > 0


def test_the_rotation_sits_before_the_difference():
    """The phase multiply occupies _apply_bloch_phase's slot — after the roll,
    before the subtract (stepping.py:1814-1818 / :1865-1869). Source order is a
    proxy the gate's mutation g (phase after the difference) backs numerically."""
    segment = _kernel_source_segment("bloch_pml_curl_step")
    rotation = segment.index("_rotate_field_left(b_x_re")
    difference = segment.index("t0_re = ")
    assert rotation < difference


def test_the_prev_read_precedes_the_w_store():
    segment = _kernel_source_segment("bloch_constitutive_step")
    first_prev = segment.index("prev_re = tl.load(w0")
    first_store = segment.index("tl.store(w0")
    assert first_prev < first_store


def test_the_guard_is_the_packages_one_spelling_and_appears_twice():
    """Two launch sites (curl + constitutive), each passing the shared constant."""
    code = code_of(MODULE_PATH)
    assert code.count("enable_fp_fusion=") == 2
    assert code.count(GUARD_SPELLING) == 2
    assert "options=" not in code


def test_the_module_does_not_reach_into_another_tracks_file():
    """Ownership on the import graph: read-only files are imported, never edited,
    and the other specialized tracks are not touched at all."""
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add("." * node.level + (node.module or ""))
    for forbidden in ("fastpath", "cuda_kernels", "symmetry", "no_pml",
                      "conductivity", "cylindrical_triton", "dispersive"):
        assert not any(forbidden in name for name in imported), (
            f"complex_fields.py imports {forbidden}: {sorted(imported)}")
    # ``..`` is ``from .. import subnormal_policy`` — the policy module, imported
    # lazily inside ``_policy_in_force`` so a licence can be checked against the
    # policy actually in force. Read-only, like ``..stepping``: this module asks
    # it what is installed and never installs anything.
    # ``..expansion_refusal`` is the package-level value module that carries the
    # refusal of an expansion probe artifact and the run policy a dispatch has
    # declared. It is not another track's file and could not be: it depends on
    # nothing — no CuPy, no Triton, no Metal, no kernel module — precisely so
    # every backend can honour a refusal without importing another backend.
    assert imported <= {"__future__", "typing", "triton", "triton.language",
                        ".coverage", ".launch", ".kernels", "..stepping", "..",
                        "..expansion_refusal",
                        "numpy", "json", "os", "inspect"}, imported
    code = code_of(MODULE_PATH)
    for forbidden in ("fastpath", "cuda_kernels"):
        assert forbidden not in code, f"complex_fields.py references {forbidden}"


def test_the_destination_is_where_the_gate_expects_it():
    assert MODULE_PATH.exists()
    assert MODULE_PATH == pathlib.Path(__file__).parent / "triton_kernels" / "complex_fields.py"


# ---------------------------------------------------------------------------
# The zero-init composition class — reachability measured, not assumed
# ---------------------------------------------------------------------------
#
# Job 2330 certified the tranche on random-seeded states, and the seeded
# signed-zero planes carry strictly NEGATIVE imag companions — both provably
# blind to the unary-minus addend lowering (the discriminating class is
# re = -0.0 with a NON-negative imag word). The composition probe's zero-init
# rows exist to reach that class from all-+0.0 state: a thin 3-cell absorber's
# deepest kms = kappa - sigma is negative, and negative-coefficient * quiet
# +0.0 words store re = -0.0 on the array path (224/226 words on every odd
# step, measured 2026-08-12), which the pre-fix kernels canonicalized to +0.0.
# These tests re-measure that reachability at every merge on NumPy.

@pytest.fixture(scope="module")
def composition_module():
    """The composition probe, imported the way the Slurm leg runs it."""
    sys.path.insert(0, str(PARITY_DIR))
    try:
        return importlib.import_module("probe_triton_complex_composition")
    finally:
        sys.path.remove(str(PARITY_DIR))


def _zero_init_rows(module):
    return [case for case in module.CASES if case[9] and case[9].get("zero_init")]


def test_the_zero_init_rows_are_first_class_cases(composition_module):
    """The zero-init class rides the CASES table itself — named rows with odd
    step budgets and in-run non-vacuity requirements — so every future recut
    carries it, not a bolt-on flag."""
    rows = {case[0]: case for case in _zero_init_rows(composition_module)}
    assert set(rows) == {"zero_init_quiet_k0", "zero_init_quiet_bloch"}
    for name, case in rows.items():
        pml_spec, sources, steps, checks = case[5], case[6], case[7], case[9]
        assert steps % 2 == 1, (
            f"{name}: the census is empty on even steps (the -0 feeds back "
            f"as -0*negative = +0), so the final compared step must be odd")
        assert checks["min_final_neg_zero_words"] >= 1
        assert checks["min_final_quiet_fraction"] >= 0.25
        assert pml_spec == {"x": 3, "y": 3}, (
            f"{name}: the thin absorber IS the producer — 3 cells at res 12 "
            f"gives kms_min = -0.5111; 5 cells stays positive and was "
            f"measured census-silent at every amplitude down to 1e-42")
        assert sources, f"{name}: a source-free case cannot separate at level 1"
    assert rows["zero_init_quiet_k0"][8] == (0, 0, 0)
    assert rows["zero_init_quiet_bloch"][8] == (1, 0, 0)
    assert rows["zero_init_quiet_bloch"][9]["min_rotate_events_per_step"] >= 1
    # Second-quadrant phase ON PURPOSE: exp(2*pi*i*k*L) with p_re < 0 < p_im
    # makes every quiet wrap-lane read a line-251 rotate-canonicalization
    # event; any other quadrant zeroes the event count on quiet reads.
    kx, lx = rows["zero_init_quiet_bloch"][4][0], rows["zero_init_quiet_bloch"][2][0]
    phase = cmath.exp(2j * cmath.pi * kx * lx)
    assert phase.real < 0 < phase.imag


def test_the_negation_mutation_sites_map_the_new_spelling_to_the_old(
        composition_module):
    """The armed negation-mutation leg's needles: each spelled form must be in
    the shipped file exactly, and no unary form may be — a drifted needle
    would disarm the leg that proves the zero-init rows discriminate."""
    source = MODULE_PATH.read_text(encoding="utf-8")
    sites = composition_module.NEGATION_MUTATION_SITES
    assert len(sites) == 3
    for spelled, unary in sites:
        assert spelled in source, f"needle drifted: {spelled!r}"
        assert unary not in source, f"old spelling present: {unary!r}"


def test_zero_init_reference_census_reaches_negative_zero_words(
        composition_module):
    """THE REACHABILITY RESULT, re-measured at every merge: from all-+0.0
    state the NumPy array path stores hundreds of -0.0 words in the compared
    volumes on every odd step (and exactly none on even steps), all in the
    discriminating class the unary-minus lowering flips, while a quiet region
    persists and the source separates the run from its control. The device
    probe re-computes the same census on the CuPy reference and fails the
    case as vacuous if it disagrees."""
    for case in _zero_init_rows(composition_module):
        row = composition_module.zero_init_reference_leg(case)
        assert row["separation_step"] == 1, (case[0], row["separation_step"])
        odd = row["census_per_step"][0::2]
        even = row["census_per_step"][1::2]
        least_odd = min(r["neg_zero_words"] for r in odd)
        assert least_odd >= 200, (
            f"{case[0]}: odd-step -0.0 census collapsed to {least_odd} "
            f"(measured 224/226 on 2026-08-12); the producer moved")
        assert all(r["neg_zero_words"] == 0 for r in even), (
            f"{case[0]}: the even-step census is no longer empty — the "
            f"feedback analysis (-0*negative = +0) no longer holds and the "
            f"odd-budget rule must be re-derived")
        assert min(r["discriminating_cells"] for r in odd) >= 200, (
            f"{case[0]}: the -0.0 words left the discriminating class "
            f"(re = -0.0 with non-negative imag)")
        assert row["census_final"]["quiet_fraction"] >= 0.25
        assert set(row["census_final"]["per_array"]) & {
            "fu_Bx", "fu_Bz", "fu_Dx", "fu_Dz"}, (
            f"{case[0]}: the words left the compared fu auxiliaries: "
            f"{row['census_final']['per_array']}")
        if case[0] == "zero_init_quiet_bloch":
            least_rot = min(r["rotate_events"] for r in row["census_per_step"])
            assert least_rot >= 1, (
                "the bloch zero-init case stopped exercising the line-251 "
                "rotate class (measured 72 events/step on 2026-08-12)")


# ---------------------------------------------------------------------------
# The subnormal policy — the license is a statement about IEEE-keep bytes
# ---------------------------------------------------------------------------
#
# Measured 2026-08-11 (jobs 2327/2328 + the ftz_lanes diagnosis): CuPy 13.5.1
# appends '-ftz=true' to every NVRTC compile; under the strip (IEEE
# subnormal-keep) the platform is exact FMA_V1 on every orientation, and the
# gate certifies UNDER THE POLICY IT WAS CUT UNDER and must say so in every
# artifact — bytes differ between the two policies in the subnormal range.
#
# The 54/48/125-word "matches NO licensed arm" of the default leg was NOT a
# property of the flushing platform: re-measured 2026-08-15 with the candidates
# cut under flush too, that same platform matches FMA_V1 uniquely on c8_mul_c8
# and BOTH arms on the three mixed orientations, where flush leaves the arms
# bit-identical. The counts were a comparison across a policy boundary. These
# tests hold the pure-host half of that contract on the laptop.

@pytest.fixture(scope="module")
def gate_module():
    """The sub-step gate, imported the way the composition probe imports it.

    Importable on this Triton-less, CUDA-less laptop by construction — the
    gate's own laptop legs (expansion, reference) depend on it.
    """
    sys.path.insert(0, str(PARITY_DIR))
    try:
        return importlib.import_module("gate_triton_complex")
    finally:
        sys.path.remove(str(PARITY_DIR))


def test_the_gate_docstring_does_not_repeat_the_retired_not_wired_claim(gate_module):
    """The same stale claim the module docstring carried, in its second home.

    ``complex_fields``' docstring was corrected on 2026-08-15; the gate's was
    not, and the gate prints its docstring as ``--help``, so the first thing a
    reader of the gate saw was "NOT wired into production dispatch
    (``plan_fast_path`` keeps returning None; nothing here changes that)".
    Measured false in the same way: ``plan_step`` on a complex64 triple emits
    'complex PML: ' reasons, ``fastpath._decide`` forwards the probe into it, and
    ``ARM_CERTIFICATION`` carries both complex arms over four driver slots.
    """
    doc = gate_module.__doc__ or ""
    # The retired ASSERTION, not the string — the corrected text quotes the old
    # claim in order to say it was measured false, which is the point of keeping
    # it. What may not survive is the claim being MADE.
    assert "complex_fields.py`` — NOT wired" not in doc
    assert "None; nothing here" not in doc
    assert "IS\non the dispatch path" in doc
    assert "launch.plan_step" in doc and "fastpath._decide" in doc

    # The claim the corrected text makes, driven rather than asserted.
    from meep_gpu import fastpath

    assert {"complex", "complex PML"} <= set(fastpath.ARM_CERTIFICATION)
    assert {"step_B", "step_D", "update_H", "update_E"} <= set(fastpath.DRIVER_SLOTS)


def test_the_gate_does_not_claim_the_legacy_control_certified_the_convention(
        gate_module):
    """The 2026-08-11 lane-exact reproduction is evidence for FTZ, NOT for what
    the tininess test is applied to — and the docstring once read as if it were.

    That reproduction ("all 21 mismatch counts lane-exactly from ftz semantics")
    is real and stays in the text. What may not be claimed is that it settled the
    rule: it was cut on the 2280-vector set, and on that set ALL THREE candidate
    rules are BIT-IDENTICAL, so the control is blind to the question by
    construction. A claim that it certified the rule would be a certification the
    evidence cannot deliver — the exact failure mode this whole correction is
    about.

    The blindness is DRIVEN here, not asserted: the legacy slice of the probe's
    own vectors is rebuilt and the rival rules are run over it.
    """
    doc = gate_module.__doc__ or ""
    assert "does NOT certify WHAT THE TININESS TEST IS APPLIED TO" \
        in doc.replace("\n", " "), (
        "the docstring must say what the legacy control does not establish")

    rng = numpy.random.default_rng(gate_module.SEED + 71)
    full = gate_module._zero_imag_operands(rng)
    added = gate_module._normal_underflow_rows()[0].size
    legacy = tuple(column[:-added] for column in full)
    assert legacy[0].size == 2280, "the legacy vector set changed shape"

    original = gate_module.candidate_arithmetic
    try:
        for field_left in (True, False):
            new = gate_module._candidates_zero_imag(
                *legacy, field_left=field_left, flush=True)
            gate_module.candidate_arithmetic = (
                lambda flush=None: _delivered_word_ops(gate_module))
            old = gate_module._candidates_zero_imag(
                *legacy, field_left=field_left, flush=True)
            gate_module.candidate_arithmetic = original
            for arm in ("FMA_V1", "NAIVE"):
                assert gate_module._disagreement_words(new[arm], old[arm]) == 0, (
                    f"the legacy vector set DOES separate the measured rule "
                    f"from the delivered-word rule on {arm} "
                    f"(field_left={field_left}) — if that is now true the "
                    f"docstring's blindness claim is stale")
    finally:
        gate_module.candidate_arithmetic = original


def test_the_gate_names_the_stripped_policy_and_states_the_dependence(gate_module):
    """The policy is named once and the dependence is spelled out: FMA_V1 holds
    under the stripped IEEE-keep policy ONLY. Every artifact stamp reuses these
    strings.

    THE SENTENCE HAS BEEN WRONG TWICE, in opposite directions, and this test
    pinned it both times. It first ended "under CuPy's default -ftz=true the
    mixed orientations match NO licensed arm" — false, and the whole refusal
    rested on it: the NEITHER came from scoring a flushed device against kept
    candidates. Its replacement said the mixed orientations "stop discriminating
    entirely" under flush, which was true only of the vector set of the day:
    every row that separated the arms there fed the device a subnormal OPERAND.
    ``_NORMAL_UNDERFLOWING_ZR`` (all-normal operands, underflowing product) is
    the class that has no operand to destroy, and with it the mixed orientations
    are measured 128 words apart under flush and 74 under keep.

    So the stamp may claim NEITHER of the retired framings, and the number it
    does claim is checked against the gate's OWN candidate builders here rather
    than trusted as prose — a dependence string is what a reader of a fresh
    artifact is told about the evidence behind its licence."""
    assert gate_module.STRIPPED_POLICY == "ieee_keep_ftz_stripped"
    assert gate_module.POLICY_CACHE_TOKEN in gate_module.STRIPPED_POLICY
    assert "-ftz=true" in gate_module.FTZ_STRIP_MECHANISM
    assert "compile_using_nvrtc" in gate_module.FTZ_STRIP_MECHANISM
    dependence = gate_module.POLICY_DEPENDENCE
    assert "STRIPPED" in dependence and "IEEE-KEEP" in dependence
    assert "ship configuration" in dependence
    assert "NO licensed arm" not in dependence
    assert "stop discriminating" not in dependence, (
        "the stamp claims the mixed orientations cannot discriminate under "
        "flush; measured, they are 128 words apart on the probe's own vectors")
    assert "DISCRIMINATES" in dependence
    assert "broken comparison" in dependence
    assert "complex_expansion_flush_coincidence_2026-08-15" in dependence

    # The numbers the stamp states, MEASURED — candidate against candidate on
    # the probe's own vectors, so the artifact's claim about its own evidence
    # cannot drift away from the vectors it was cut from.
    import numpy as np

    rng = np.random.default_rng(gate_module.SEED + 71)
    c8 = gate_module._c8_operands(rng)
    zero_imag = gate_module._zero_imag_operands(rng)
    scalar_range = {}
    for policy, flush, mixed in (("flush", True, 128), ("keep", False, 74)):
        assert gate_module._arms_apart(
            gate_module._candidates_c8(*c8, flush=flush)) == 1088, policy
        for field_left in (True, False):
            apart = gate_module._arms_apart(gate_module._candidates_zero_imag(
                *zero_imag, field_left=field_left, flush=flush))
            assert apart == mixed, (
                f"{policy}: mixed orientation field_left={field_left} measures "
                f"{apart} words apart, the stamp states {mixed}")
        assert f"{mixed}" in dependence
        # python_float_left is the third zero-imaginary orientation and it does
        # NOT track the other two: its coefficient is a broadcast scalar, so the
        # added class only underflows for some scalars. Pinned separately, or
        # the stamp's one number would be read as covering all three.
        zr, zi, _ = zero_imag
        per_scalar = [gate_module._arms_apart(gate_module._candidates_zero_imag(
            zr, zi, np.full(zr.shape, np.float32(s), dtype=np.float32),
            field_left=False, flush=flush)) for s in (0.35, 0.5, -0.35, 0.1)]
        assert all(count > 0 for count in per_scalar), (policy, per_scalar)
        scalar_range[policy] = (min(per_scalar), max(per_scalar))
    # The flush floor read 64 while the FTZ candidates detected tininess AFTER
    # rounding: at scalar 0.5 the tie rows (zr at the top of the min-normal
    # binade) were rounded UP onto the smallest normal by both arms instead of
    # underflowing, and the arms' signed-zero split never happened there. With
    # the convention corrected to the hardware's the floor is 96 — the
    # correction RAISED the probe's power on that scalar by 32 words, all of
    # them zr=0x80FFFFFF. See test_the_correction_moves_only_the_tie_class.
    assert scalar_range == {"flush": (96, 128), "keep": (6, 6)}, scalar_range
    assert "128-96" in dependence and "6 (keep)" in dependence

    # The FLUSH stamp is the one a shipped-x86 artifact carries, and flush is
    # where the evidence was thinnest, so it states the same measured numbers.
    flush_dependence = gate_module.FLUSH_DEPENDENCE
    assert "1088" in flush_dependence and "128" in flush_dependence
    assert "128-96" in flush_dependence
    assert "UNBOUNDED exponent" in flush_dependence, (
        "the flush stamp must say what tininess test its candidates were cut "
        "under, and the unbounded exponent is the load-bearing half of it — it "
        "is the difference between licensing this platform and refusing it")
    assert "DELIVERED float32 word" in flush_dependence, (
        "the flush stamp must name the rule the correction retired, or a "
        "reader cannot tell which cut an artifact carries")
    assert "BEFORE rounding" not in flush_dependence, (
        "the flush stamp claims tininess is tested on the exact value before "
        "rounding; that rule disagrees with both measured hardware executors "
        "on the separating class")
    assert "_NORMAL_UNDERFLOWING_ZR" in flush_dependence
    assert "0 of 4560" in flush_dependence, (
        "the flush stamp must say what its evidence used to be, or a reader "
        "cannot tell a re-cut artifact from one carrying the blind vector set")


def test_probe_records_are_stamped_with_the_policy_they_were_cut_under(
        gate_module, monkeypatch):
    """The stamp is the artifact's statement of WHICH bytes were classified."""
    # Structural: the measurement itself writes the stamp (after the products
    # ran, so the strip counters cover this record's own compiles).
    source = GATE_PATH.read_text(encoding="utf-8")
    assert 'record["subnormal_policy"] = policy_stamp(backend_name)' in source

    assert gate_module.policy_stamp("numpy")["policy"] == "host_ieee_keep"
    if gate_module.cp is None:
        assert gate_module.policy_stamp("cupy")["policy"] == "no_device"

    # Simulated CuPy host, strip installed: the stamp carries the policy name,
    # the dependence, and the counters a downstream consumer re-checks.
    monkeypatch.setattr(gate_module, "cp", object())
    monkeypatch.setattr(gate_module, "_FTZ_STRIP", {
        "installed": True, "policy": gate_module.STRIPPED_POLICY,
        "calls": 5, "removed": 5, "cache_dir": "/x/cupy_cache_ftz_stripped_1",
        "cache_preexisting": False, "example_options": []})
    stamp = gate_module.policy_stamp("cupy")
    assert stamp["policy"] == gate_module.STRIPPED_POLICY
    assert stamp["dependence"] == gate_module.POLICY_DEPENDENCE
    assert stamp["nvrtc_calls"] == 5 and stamp["ftz_removed"] == 5

    # Simulated CuPy host WITHOUT the strip: the stamp is a warning, not a
    # certification, and names the default flush.
    monkeypatch.setattr(gate_module, "_FTZ_STRIP", None)
    stamp = gate_module.policy_stamp("cupy")
    assert stamp["policy"] == "cupy_default_ftz_flush"
    assert "license nothing" in stamp["warning"] or "NOT the ship" in stamp["warning"]


def test_the_license_requires_the_strip_confirmed_exercised(gate_module,
                                                            monkeypatch):
    """ftz_strip_license_reasons is the in-run assertion: strip installed AND
    every observed NVRTC option tuple lost '-ftz=true', with zero compiles
    failing loudly unless a policy-suffixed pre-populated cache explains it."""
    monkeypatch.setattr(gate_module, "_FTZ_STRIP", None)
    reasons = gate_module.ftz_strip_license_reasons()
    assert reasons and "not installed" in reasons[0]
    assert "ship" in reasons[0]

    def state(**overrides):
        base = {"installed": True, "policy": gate_module.STRIPPED_POLICY,
                "calls": 5, "removed": 5, "cache_dir": "/x/ftz_stripped_c",
                "cache_preexisting": False, "example_options": []}
        base.update(overrides)
        return base

    monkeypatch.setattr(gate_module, "_FTZ_STRIP", state())
    assert gate_module.ftz_strip_license_reasons() == []

    monkeypatch.setattr(gate_module, "_FTZ_STRIP", state(removed=3))
    reasons = gate_module.ftz_strip_license_reasons()
    assert reasons and "2/5" in reasons[0] and "seam" in reasons[0]

    monkeypatch.setattr(gate_module, "_FTZ_STRIP",
                        state(calls=0, removed=0))
    reasons = gate_module.ftz_strip_license_reasons()
    assert reasons and "zero NVRTC compiles" in reasons[0]

    explained = state(calls=0, removed=0, cache_preexisting=True)
    monkeypatch.setattr(gate_module, "_FTZ_STRIP", explained)
    assert gate_module.ftz_strip_license_reasons() == []
    assert "zero_compiles_explained" in explained, (
        "a compile-free run must record WHY it still counts (the cache "
        "explains it), not silently pass")


def test_a_probe_artifact_licenses_only_bytes_cut_under_the_stripped_policy(
        gate_module):
    """probe_record_policy_reasons is the fail-closed check a REUSED artifact
    goes through (composition --probe-artifact, the gate's load path): the
    stamp's policy name AND its counters must hold up. Job 2327's artifact —
    default policy, no stamp — must refuse here even though its patterns would
    never license anyway."""
    good_stamp = {"policy": gate_module.STRIPPED_POLICY,
                  "nvrtc_calls": 5, "ftz_removed": 5,
                  "cache_preexisting": False}

    record = _probe_record()
    assert gate_module.probe_record_policy_reasons(record), (
        "an artifact with NO policy stamp licensed — the pre-stamp era is "
        "indistinguishable from the default flush and must refuse")

    record["subnormal_policy"] = {"policy": "cupy_default_ftz_flush"}
    assert gate_module.probe_record_policy_reasons(record)

    record["subnormal_policy"] = dict(good_stamp)
    assert gate_module.probe_record_policy_reasons(record) == []

    record["subnormal_policy"] = dict(good_stamp, nvrtc_calls=0)
    assert gate_module.probe_record_policy_reasons(record), (
        "a stamp recording zero exercised compiles under a fresh cache is "
        "not a certification")

    record["subnormal_policy"] = dict(good_stamp, ftz_removed=3)
    assert gate_module.probe_record_policy_reasons(record)

    record["subnormal_policy"] = {"policy": gate_module.STRIPPED_POLICY}
    assert gate_module.probe_record_policy_reasons(record), (
        "a stamp without counters is unverifiable and must refuse")

    # Absence passes through: a MISSING artifact is already a refusal by name
    # downstream (clause 13); this check is for a present, wrong one.
    assert gate_module.probe_record_policy_reasons(None) == []


def test_the_cache_dir_must_be_private_and_policy_suffixed(gate_module):
    """The cache key is computed above the strip seam, so a shared cache mixes
    policies silently — refused by name before anything compiles."""
    assert gate_module.ftz_cache_reasons(None)
    assert gate_module.ftz_cache_reasons("")
    reasons = gate_module.ftz_cache_reasons("/x/cupy_cache_12345")
    assert reasons and gate_module.POLICY_CACHE_TOKEN in reasons[0]
    assert gate_module.ftz_cache_reasons(
        "/x/cupy_cache_ftz_stripped_12345") == []


def test_the_probe_vectors_survive_a_conversion_that_flushes(gate_module,
                                                             monkeypatch):
    """BLOCKER 1, measured on the device leg of 2026-08-15: in the ship
    configuration ``np.asarray([1e-45, ...], np.float32)`` returns zero — the
    probe's 72 subnormal ``zr`` words, half its discriminating power, destroyed
    at the conversion before the device is asked anything. The words are built
    in integer arithmetic instead, so the same vectors are measured under both
    policies. Here the flushing conversion is imposed to prove the repair does
    the work rather than the host's arm64 FPU."""
    small = list(gate_module._SMALL)
    real_asarray = numpy.asarray

    def flushing_asarray(values, dtype=None, **kwargs):
        out = real_asarray(values, dtype=dtype, **kwargs)
        if getattr(out, "dtype", None) == numpy.float32:
            words = numpy.ascontiguousarray(out).view(numpy.uint32).copy()
            subnormal = (((words >> 23) & 0xFF) == 0) & ((words & 0x7FFFFF) != 0)
            words[subnormal] &= numpy.uint32(0x80000000)
            return words.view(numpy.float32)
        return out

    monkeypatch.setattr(numpy, "asarray", flushing_asarray)
    destroyed = flushing_asarray(small, dtype=numpy.float32)
    rebuilt = gate_module._asarray_float32(small)
    monkeypatch.undo()  # census with the real numpy, or it flushes the answer too
    assert gate_module.subnormal_words(destroyed) == 0, (
        "the imposed conversion must actually destroy them, or this proves nothing")
    assert gate_module.subnormal_words(rebuilt) == len(small)
    assert rebuilt.tobytes() == numpy.asarray(small, dtype=numpy.float32).tobytes()


def test_the_exact_emulation_is_integer_based_so_it_holds_under_both_policies(
        gate_module):
    """BLOCKER 2: ``_self_check_fma32``'s "-0 on underflow" identity raised
    AssertionError in the flushing ship configuration, because its subnormal was
    built by a float32 CONVERSION. fma32/mul32 decode their operands from words
    and round in integer arithmetic, so the identity is a fact about the
    emulation rather than about the host's FPU — and the self-check runs."""
    tiny = gate_module._float32_from_word(0x00000001)
    assert gate_module._float32_word(tiny) == 0x00000001
    assert gate_module._float32_word(
        gate_module.fma32(gate_module._negate32(tiny), 0.25, 0.0)) == 0x80000000
    assert gate_module._float32_word(gate_module.mul32(tiny, 2.0)) == 0x00000002
    assert gate_module._float32_word(gate_module.mul32(tiny, 0.25)) == 0x00000000
    assert "np.float32(1e-45)" not in code_of(GATE_PATH), (
        "the subnormal a self-check underflows must not come from a conversion")
    gate_module._self_check_fma32()


# ---------------------------------------------------------------------------
# THE TININESS RULE — what the FTZ candidate emulation tests, and on what value
# ---------------------------------------------------------------------------
#
# THREE RULES, NOT TWO. IEEE 754-2019 §7.5 lets an implementation detect
# tininess either BEFORE rounding (on the result computed with unbounded
# precision AND unbounded exponent) or AFTER (on that result rounded to the
# destination's precision, still with unbounded exponent), requiring only that
# one be used throughout. A third rule is not an IEEE option at all but is the
# one this file implemented until 2026-08-15: test whether the DELIVERED float32
# word is still subnormal, i.e. round on the 2**-149 subnormal grid instead of
# with an unbounded exponent.
#
# THE MEASURED RULE IS THE MIDDLE ONE. 37,439 vectors whose classes were derived
# by factoring exact rationals — built independently of the gate and of these
# tests — run on two executors: x86 mulss / vfmadd213ss with MXCSR set once per
# pass over all four FTZ|DAZ combinations, and PTX mul.rn.ftz.f32 /
# fma.rn.ftz.f32 through inline asm in a CuPy RawKernel on an RTX A6000. The
# executors agree with each other 37439/37439. Against the three rules:
# before-rounding differs on 10, after-rounding (24 significant bits, UNBOUNDED
# exponent) on 0, delivered-word on 18. The Intel SDM says the same thing —
# Vol. 1 §10.2.3.3 makes FTZ the masked response to the underflow condition and
# §4.9.1.5 defines that condition on "the result of rounding with unbounded
# exponent ... non-zero and tiny". The PTX ISA is SILENT on what the test is
# applied to and settles nothing either way.
#
# TWO SEPARATING CLASSES, and a test that reaches only one of them cannot pin
# the rule:
#   * :data:`TIE_ROWS` — exact value 2**-126 - 2**-150 — separates the delivered
#     word from the other two. It does NOT separate before- from after-rounding:
#     that value has exactly 24 significant bits, so the unbounded-exponent
#     rounding is the identity on it and both rules call it tiny. This is the
#     class the 2026-08-15 round mistook for a discriminator.
#   * :data:`SEPARATING_ROWS` — exact value in [2**-126 - 2**-151, 2**-126) —
#     separates before-rounding from the other two, and nothing else does. The
#     window is a quarter of a subnormal ulp wide, relative measure ~2**-25, so
#     it is unreachable by sampling and its rows must be CONSTRUCTED.
#
# WHAT THESE TESTS DO AND DO NOT PIN, stated because the block here used to
# claim more. The six tests that predate 2026-08-16 pass UNCHANGED under both
# the before-rounding and the after-rounding models — measured, by installing
# each in turn — so they pin FTZ and they pin the delivered-word rule out, but
# they do not pin the rule. ``test_the_rule_is_not_the_exact_value_before_
# rounding`` is the one that does, and it fails under before-rounding.
# ``test_all_three_rules_agree_on_the_ordinary_band`` is the control that passes
# under all three, so the suite cannot be read as merely tracking today's
# output. Every expected word below is derived from the exact arithmetic in the
# test's own docstring, not copied from a run.

#: Half an ulp below 2**-126 reached from four different binades, in both signs.
#: ``0x00FFFFFF`` is 2**-125 - 2**-149, the top of the min-normal binade; its
#: half is 2**-126 - 2**-150, exactly halfway between the largest subnormal and
#: the smallest normal on the SUBNORMAL grid, so the delivered float32 word is
#: the normal 0x00800000 — and the flushing hardware returns zero anyway,
#: because 2**-126 - 2**-150 has exactly 24 significant bits and therefore
#: survives the unbounded-exponent rounding unchanged, below 2**-126.
#:
#: THIS CLASS DOES NOT SEPARATE BEFORE- FROM AFTER-ROUNDING. See
#: :data:`SEPARATING_ROWS` for the class that does.
TIE_ROWS = (
    (0x00FFFFFF, 0.5), (0x80FFFFFF, 0.5),
    (0x00FFFFFF, -0.5), (0x80FFFFFF, -0.5),
    (0x017FFFFF, 0.25), (0x817FFFFF, 0.25),
    (0x01FFFFFF, 0.125), (0x81FFFFFF, 0.125),
    (0x027FFFFF, 0.0625), (0x827FFFFF, 0.0625),
)


#: THE SEPARATING CLASS — the only class that tells a test on the EXACT value
#: apart from a test on the value rounded to 24 significant bits with unbounded
#: exponent. ``(word_a, word_b)`` pairs whose exact product lies in
#: ``[2**-126 - 2**-151, 2**-126)``: below the smallest normal, so a
#: before-rounding test flushes them, but ties UP onto 2**-126 under 24-bit
#: unbounded-exponent rounding (the grid there is 2**-150 and 2**24 is the even
#: neighbour), so the measured rule delivers ``0x00800000``.
#:
#: HOW THESE WERE CONSTRUCTED, because they cannot be sampled. Both operands
#: must be NORMAL or DAZ reaches the row first, so with significands
#: ``ma, mb in [2**23, 2**24)`` the product is ``ma * mb * 2**(Ea + Eb - 46)``.
#: The window forces ``ma * mb in [2**47 - 2**22, 2**47)`` — the ``2**48`` case
#: is empty, since ``(2**24 - 1)**2 < 2**48 - 2**23`` — and forces
#: ``Ea + Eb = -127``. So the class is exactly the integers just below ``2**47``
#: that factor into two 24-bit halves, and these rows are picked from that
#: factoring: at the top of the window, a quarter in, half in, and on the tie
#: endpoint itself. The window is a QUARTER OF A SUBNORMAL ULP wide (relative
#: measure ~2**-25), which is why 400,000 band-aimed random products reached it
#: zero times.
#:
#: ``0x24042108 * 0x1BF80000`` is the row the two executors were measured on:
#: ``8659208 * 16252928 = 2**22 * (2**25 - 1)``, so the exact product is
#: ``2**-126 - 2**-151``, and both x86 and PTX return ``0x00800000``.
SEPARATING_ROWS = (
    # exact product = 2**-126 - 2**-172, at the very top of the window
    (0x24000001, 0x1BFFFFFE), (0xA4000001, 0x1BFFFFFE),
    (0x24000001, 0x9BFFFFFE), (0xA4000001, 0x9BFFFFFE),
    # exact product = 2**-126 - 261547 * 2**-171, about a quarter of the way in
    (0x00857E37, 0x3F75774C), (0x80857E37, 0x3F75774C),
    (0x00857E37, 0xBF75774C), (0x80857E37, 0xBF75774C),
    # exact product = 2**-126 - 2093753 * 2**-173, about half way in
    (0x3F0CAC03, 0x00E8F06D), (0xBF0CAC03, 0x00E8F06D),
    (0x3F0CAC03, 0x80E8F06D), (0xBF0CAC03, 0x80E8F06D),
    # exact product = 2**-126 - 2**-151, the tie endpoint (ma*mb = 2**47 -
    # 2**22), at a second exponent split: Ea = -64, Eb = -63
    (0x1F842108, 0x20780000), (0x9F842108, 0x20780000),
    (0x1F842108, 0xA0780000), (0x9F842108, 0xA0780000),
    # ... and the row the hardware was measured on, Ea = -55, Eb = -72
    (0x24042108, 0x1BF80000), (0xA4042108, 0x1BF80000),
    (0x24042108, 0x9BF80000), (0xA4042108, 0x9BF80000),
)


#: EVERYTHING BELOW GOES THROUGH ``candidate_arithmetic(flush=True)``, the seam
#: the candidate arms are actually built from, and never through a private
#: helper the corrected file happens to spell a particular way. That is what
#: makes these tests runnable against an EARLIER copy of the gate: pointed at
#: the pre-fix file they fail on the WORDS, which is the failure that means
#: something, rather than on a missing attribute, which would only mean a rename.
def _ftz_mul_word(gate_module, a, b) -> int:
    return gate_module._word_of_float32(
        gate_module.candidate_arithmetic(flush=True)["mul_scalar"](a, b))


def _ftz_fma_word(gate_module, a, b, c) -> int:
    return gate_module._word_of_float32(
        gate_module.candidate_arithmetic(flush=True)["fma_scalar"](a, b, c))


def _flushed_operand_word(gate_module, value) -> int:
    """The operand (DAZ) half of .ftz, which ALL THREE rules share."""
    word = gate_module._float32_word(value)
    if ((word >> 23) & 0xFF) == 0 and (word & 0x7FFFFF) != 0:
        return word & 0x80000000
    return word


def _flush_if_delivered_word_subnormal(word: int) -> int:
    """The delivered-word rule's tail: zero a result only if the float32 word
    actually handed back is still subnormal."""
    if ((word >> 23) & 0xFF) == 0 and (word & 0x7FFFFF) != 0:
        return word & 0x80000000
    return word


def _before_rounding_word(gate_module, exact, negative_zero: bool) -> int:
    """WRONG MODEL #2: tininess tested on the EXACT value, before any rounding.

    Installed on 2026-08-15 in place of the delivered-word rule and refuted by
    measurement on 2026-08-16: it returns a sign-preserving zero on
    :data:`SEPARATING_ROWS`, where both hardware executors return 0x00800000.
    Kept here so the difference stays measurable rather than asserted.
    """
    if exact == 0:
        return 0x80000000 if negative_zero else 0
    negative = exact < 0
    if (-exact if negative else exact) < fractions.Fraction(1, 1 << 126):
        return 0x80000000 if negative else 0
    return gate_module._round_float32_word(exact, -1.0 if negative else 1.0)


def _before_rounding_mul_word(gate_module, a, b) -> int:
    wa = _flushed_operand_word(gate_module, a)
    wb = _flushed_operand_word(gate_module, b)
    return _before_rounding_word(
        gate_module,
        gate_module._fraction_of_word(wa) * gate_module._fraction_of_word(wb),
        bool((wa ^ wb) & 0x80000000))


def _delivered_word_mul_word(gate_module, a, b) -> int:
    """WRONG MODEL #1, restated here so the correction stays measurable.

    Operands flushed, the product rounded once to float32 — and only THEN
    zeroed, if the DELIVERED word is still subnormal. This is what
    ``flush_subnormals(mul32(flush_subnormals(a), flush_subnormals(b))))`` did
    before 2026-08-15. It is not an IEEE 754 option: it rounds on the 2**-149
    subnormal grid where the underflow condition calls for an unbounded
    exponent, and the two grids disagree on :data:`TIE_ROWS`.
    """
    return _flush_if_delivered_word_subnormal(gate_module._word_of_float32(gate_module.mul32(
        gate_module._float32_from_word(_flushed_operand_word(gate_module, a)),
        gate_module._float32_from_word(_flushed_operand_word(gate_module, b)))))


def _delivered_word_fma_word(gate_module, a, b, c) -> int:
    """The same, fused."""
    return _flush_if_delivered_word_subnormal(gate_module._word_of_float32(gate_module.fma32(
        gate_module._float32_from_word(_flushed_operand_word(gate_module, a)),
        gate_module._float32_from_word(_flushed_operand_word(gate_module, b)),
        gate_module._float32_from_word(_flushed_operand_word(gate_module, c)))))


def _delivered_word_ops(gate_module):
    """A candidate-arithmetic set with the DELIVERED-WORD rule put back.

    Only the two operations that can tell the rules apart are rebuilt; add and
    sub come from the module itself, because for float32 addition all three
    rules provably coincide (every finite float32 is a multiple of 2**-149, so a
    tiny exact sum is exactly representable and no rounding occurs at all) and a
    difference there would be a different defect.
    """
    def mul_scalar(a, b):
        return gate_module._float32_from_word(
            _delivered_word_mul_word(gate_module, a, b))

    def fma_scalar(a, b, c):
        return gate_module._float32_from_word(
            _delivered_word_fma_word(gate_module, a, b, c))

    def mul(a, b):
        left, right = numpy.broadcast_arrays(
            numpy.asarray(a, dtype=numpy.float32),
            numpy.asarray(b, dtype=numpy.float32))
        words = numpy.array(
            [_delivered_word_mul_word(gate_module, x, y)
             for x, y in zip(numpy.ascontiguousarray(left).ravel(),
                             numpy.ascontiguousarray(right).ravel())],
            dtype=numpy.uint32)
        return words.view(numpy.float32).reshape(left.shape)

    return {"flush": True, "policy": "flush", "semantics": "delivered-word",
            "mul": mul, "add": gate_module._ftz_add, "sub": gate_module._ftz_sub,
            "fma_scalar": fma_scalar, "mul_scalar": mul_scalar}


def test_the_ftz_candidate_flushes_the_tie_in_both_signs(gate_module):
    """THE FIRST DEFECT, in one line of arithmetic, in both signs and under both
    policies. This is the test that rules the DELIVERED-WORD rule out.

    ``0x00FFFFFF`` is 2**-125 - 2**-149. Halved, the EXACT product is
    2**-126 - 2**-150, which has exactly 24 significant bits: rounding it to 24
    significant bits WITH UNBOUNDED EXPONENT leaves it alone, still below the
    smallest normal, so the measured rule flushes it to a sign-preserving zero.
    A test on the EXACT value flushes it too — this class does not separate
    those two, and reading it as if it did is what produced the 2026-08-15
    overshoot (see :data:`SEPARATING_ROWS` for the class that separates them).

    What it DOES separate is the delivered word. The float32 rounding is floored
    at the 2**-149 subnormal grid, where 2**-126 - 2**-150 is a tie between the
    largest subnormal and the smallest normal; half-to-even takes the normal
    0x00800000. A rule that inspects that word keeps a value the device zeroed.

    The IEEE-keep arm must still answer 0x00800000: the two policies have to
    differ here, or the flush candidate has stopped modelling a flush.
    """
    tiny_top = gate_module._float32_from_word(0x00FFFFFF)
    negative_top = gate_module._float32_from_word(0x80FFFFFF)

    assert _ftz_mul_word(gate_module, tiny_top, 0.5) == 0x00000000
    assert _ftz_mul_word(gate_module, negative_top, 0.5) == 0x80000000
    assert _ftz_mul_word(gate_module, tiny_top, -0.5) == 0x80000000
    assert _ftz_mul_word(gate_module, negative_top, -0.5) == 0x00000000
    # The fused arm takes the same route: fma(zr, c, +0.0) is the FMA_V1 real
    # part on exactly these rows.
    assert _ftz_fma_word(gate_module, tiny_top, 0.5, 0.0) == 0x00000000
    assert _ftz_fma_word(gate_module, negative_top, 0.5, 0.0) == 0x80000000

    # KEEP is the control: same operands, other policy, the tie rounds UP.
    assert gate_module._word_of_float32(
        gate_module.mul32(tiny_top, 0.5)) == 0x00800000
    assert gate_module._word_of_float32(
        gate_module.mul32(negative_top, 0.5)) == 0x80800000
    assert gate_module._word_of_float32(
        gate_module.fma32(tiny_top, 0.5, 0.0)) == 0x00800000

    # Reached from three further binades, so this is a property of the
    # convention and not of one operand word.
    for word, scalar in TIE_ROWS:
        operand = gate_module._float32_from_word(word)
        sign = 0x80000000 if (word & 0x80000000) ^ (
            0x80000000 if scalar < 0 else 0) else 0
        assert _ftz_mul_word(gate_module, operand, scalar) == sign, (
            f"0x{word:08x} * {scalar} must flush to a sign-preserving zero")
        assert gate_module._word_of_float32(
            gate_module.mul32(operand, scalar)) == (0x00800000 | sign), (
            f"0x{word:08x} * {scalar} must round onto the smallest normal "
            f"under keep, or it is not the discriminating tie at all")


def test_the_tininess_boundary_belongs_to_the_normal_side(gate_module):
    """Just outside, exactly on, and just inside 2**-126.

    * ``0x01000000 * 0.5`` is EXACTLY 2**-126. Tininess is "strictly below the
      smallest normal", so this is a normal result and must survive: an
      off-by-one that flushed the boundary would destroy the smallest normal
      itself on every flushing run.
    * ``0x01000001 * 0.5`` is 2**-126 + 2**-149, above the boundary — survives,
      and rounds to the smallest normal's neighbour.
    * ``0x00FFFFFE * 0.5`` is 2**-126 - 2**-149, one ulp below and NOT a tie —
      it is tiny under all three rules and flushes under all three.

    THIS ONE IS A CONTROL, and it passes against every version of the gate. That
    is the point: every value here is one all three rules agree on, so it
    catches a correction that moved the boundary (or lost the smallest normal)
    without being able to pass merely because the model flushes more.
    """
    on_boundary = gate_module._float32_from_word(0x01000000)
    assert _ftz_mul_word(gate_module, on_boundary, 0.5) == 0x00800000
    assert _ftz_mul_word(
        gate_module, gate_module._float32_from_word(0x81000000),
        0.5) == 0x80800000
    assert _ftz_fma_word(gate_module, on_boundary, 0.5, 0.0) == 0x00800000

    just_above = gate_module._float32_from_word(0x01000001)
    assert _ftz_mul_word(gate_module, just_above, 0.5) == 0x00800001

    just_below = gate_module._float32_from_word(0x00FFFFFE)
    assert _ftz_mul_word(gate_module, just_below, 0.5) == 0x00000000
    # ... and the delivered-word rule agrees with the measured one there, which
    # is the point: only the TIE separates those two.
    assert _delivered_word_mul_word(gate_module, just_below, 0.5) == 0x00000000


def test_the_measured_rule_and_the_delivered_word_rule_differ_only_at_the_tie(
        gate_module):
    """THE CHECK THAT THE FIRST CORRECTION DID NOT SIMPLY FLUSH MORE THINGS.

    Sweeps operand words along the boundary of every binade whose halving or
    quartering lands on or beside 2**-126, in both signs, against nine
    multipliers, and classifies each row by ARITHMETIC: a row separates the
    measured rule from the delivered-word rule exactly when the exact product is
    nonzero, strictly below 2**-126, and its float32 rounding is NOT subnormal.
    The installed model must differ from the delivered-word one on precisely
    those rows and agree with it on every other — including the rows that flush
    under both, which is where a model "tuned to make the gate pass" would have
    shown up as a broad loosening.

    WHAT THIS SWEEP CANNOT SEE, stated because a previous round read it as a
    bound on the whole change: none of its 414 rows lies in
    :data:`SEPARATING_ROWS`. Its scalars are powers of two plus 0.75 and 1.5, so
    every exact product here is a multiple of 2**-151 or coarser, and the
    separating window is an open interval of width 2**-151 containing no such
    multiple. So this sweep is blind to the before-rounding overshoot by
    construction, and ``test_the_rule_is_not_the_exact_value_before_rounding``
    is what covers it.
    """
    words = (0x00FFFFFD, 0x00FFFFFE, 0x00FFFFFF, 0x01000000, 0x01000001,
             0x01000002, 0x017FFFFE, 0x017FFFFF, 0x01800000, 0x01FFFFFE,
             0x01FFFFFF, 0x02000000, 0x027FFFFF, 0x02800000, 0x00800000,
             0x00800001, 0x007FFFFE, 0x007FFFFF, 0x00000001, 0x00000002,
             0x00400000, 0x3F800000, 0x00000000)
    scalars = (0.5, -0.5, 0.25, 0.125, 1.0, 2.0, 0.75, 1.5, 0.0625)
    minimum_normal = fractions.Fraction(1, 1 << 126)

    agree = 0
    discriminating_rows = []
    for word in words:
        for signed in (word, word ^ 0x80000000):
            operand = gate_module._float32_from_word(signed)
            for scalar in scalars:
                flushed_word = _flushed_operand_word(gate_module, operand)
                exact = (gate_module._fraction_of_word(flushed_word)
                         * gate_module._fraction_of_word(
                             gate_module._float32_word(scalar)))
                rounded = gate_module._round_float32_word(
                    exact, 1.0 if exact >= 0 else -1.0) if exact != 0 else 0
                discriminating = (
                    exact != 0 and abs(exact) < minimum_normal
                    and ((rounded >> 23) & 0xFF) != 0)
                old = _delivered_word_mul_word(gate_module, operand, scalar)
                new = _ftz_mul_word(gate_module, operand, scalar)
                if discriminating:
                    discriminating_rows.append((signed, scalar))
                    assert new != old, (
                        f"0x{signed:08x} * {scalar}: exact {float(exact):.6e} "
                        f"is tiny but its float32 rounding is a normal "
                        f"0x{rounded:08x}; the measured rule and the "
                        f"delivered-word rule MUST differ here and both "
                        f"answered 0x{new:08x}")
                    assert new == (signed ^ gate_module._float32_word(scalar)
                                   ) & 0x80000000, (
                        f"0x{signed:08x} * {scalar}: the flush returns a zero "
                        f"of the exact product's sign")
                else:
                    agree += 1
                    assert new == old, (
                        f"0x{signed:08x} * {scalar}: the measured rule and the "
                        f"delivered-word rule coincide here (exact "
                        f"{float(exact):.6e}) and the installed model answered "
                        f"0x{new:08x} against 0x{old:08x} — the correction "
                        f"changed a row it must not touch")
    # The discriminating rows are named, not counted: they are exactly the
    # binade tops whose scaling lands half an ulp below 2**-126, in both signs,
    # and nothing else in a 414-row sweep reaches the tie.
    assert sorted(discriminating_rows) == sorted(TIE_ROWS), (
        sorted((hex(word), scalar) for word, scalar in discriminating_rows))
    assert len(discriminating_rows) == 10
    assert agree == 404, agree


def test_the_fused_arm_tests_the_exact_sum_not_the_intermediate_product(
        gate_module):
    """The tininess test is applied to the exact a*b + c, with no separate
    flush of the intermediate product — measured on the device, and the only
    way both of these can hold at once.

    ``fma(2**-75, -2**-75, 2**-126)``: all three operands are NORMAL and the
    exact product 2**-150 is far below the band. The exact SUM is
    2**-126 - 2**-150, the same tie as above, so the result is zero. Flush the
    intermediate product instead and the answer would be 2**-126 exactly, i.e.
    0x00800000 — the opposite verdict.

    Its mirror ``fma(2**-75, 2**-75, 2**-126)`` sums to 2**-126 + 2**-150,
    which is NOT tiny under any of the three rules, and must survive as
    0x00800000.
    """
    positive = gate_module._float32_from_word(0x1A000000)   # 2**-75
    negative = gate_module._float32_from_word(0x9A000000)   # -2**-75
    smallest_normal = gate_module._float32_from_word(0x00800000)

    assert _ftz_fma_word(gate_module, positive, negative,
                         smallest_normal) == 0x00000000
    assert _ftz_fma_word(gate_module, positive, positive,
                         smallest_normal) == 0x00800000
    # ... and the DELIVERED-WORD rule answered the opposite on the first of
    # them, reached without a subnormal operand.
    assert _delivered_word_fma_word(gate_module, positive, negative,
                                    smallest_normal) == 0x00800000
    # The keep arm keeps the tie on the normal side of the boundary either way.
    assert gate_module._word_of_float32(gate_module.fma32(
        positive, negative, smallest_normal)) == 0x00800000


def test_the_flush_candidate_set_contains_no_host_float_operation(gate_module):
    """The second defect at the same site: the vectorized flush arms were
    ``flush_subnormals(host_float32_op(...))``, so the rule they implemented was
    whatever FPU the process happened to be running on. On the flushing x86 host
    that was accidentally the hardware's own rule (the hardware did the flush)
    while the scalar arms tested the delivered word; on this arm64 laptop every
    arm tested the delivered word. One 'flush' request meant three things.

    Both halves are checked: the vector arms reproduce the scalar rule word for
    word on the tie class, and the flush branch of ``candidate_arithmetic``
    binds nothing that multiplies float32s on the host.
    """
    rows = [(gate_module._float32_from_word(word), numpy.float32(scalar))
            for word, scalar in TIE_ROWS]
    rows += [(gate_module._float32_from_word(word), numpy.float32(0.5))
             for word in (0x01000000, 0x00800000, 0x3F800000, 0x00000000)]
    left = numpy.array([row[0] for row in rows], dtype=numpy.float32)
    right = numpy.array([row[1] for row in rows], dtype=numpy.float32)

    ops = gate_module.candidate_arithmetic(flush=True)
    vector = ops["mul"](left, right)
    scalar = numpy.array(
        [gate_module._float32_from_word(_ftz_mul_word(gate_module, a, b))
         for a, b in zip(left, right)], dtype=numpy.float32)
    assert vector.tobytes() == scalar.tobytes()
    # ... and both of them flush the tie, which is what says the VECTOR arm
    # is on the measured rule and not merely consistent with a wrong scalar
    # one.
    tie_words = numpy.ascontiguousarray(
        vector[:len(TIE_ROWS)]).view(numpy.uint32)
    assert set(int(word) for word in tie_words) == {0x00000000, 0x80000000}, (
        [hex(int(word)) for word in tie_words])

    # add and sub go through the same rule even though they provably cannot
    # tell the three rules apart (every finite float32 is a multiple of
    # 2**-149, so a tiny exact sum is exactly representable and no rounding
    # occurs at all): the point is that no candidate operation borrows the host
    # FPU.
    assert gate_module._ftz_add(left, -left).tobytes() == numpy.zeros(
        left.shape, dtype=numpy.float32).tobytes()
    for name in ("mul", "add", "sub"):
        assert ops[name] not in (gate_module._plain_mul, gate_module._plain_add,
                                 gate_module._plain_sub)
    # The semantics string must NAME the rule, and naming it means naming the
    # unbounded exponent: "after rounding" alone is what the delivered-word rule
    # called itself.
    semantics = ops["semantics"]
    assert "UNBOUNDED EXPONENT" in semantics.upper(), semantics
    assert "24 SIGNIFICANT BITS" in semantics.upper(), semantics
    assert "BEFORE ROUNDING" not in semantics.upper(), (
        "the stamp claims tininess is tested on the exact value before "
        "rounding; measured, that rule disagrees with both hardware executors "
        "on the separating class")


def test_the_rule_is_not_the_exact_value_before_rounding(gate_module):
    """THE TEST A BEFORE-ROUNDING MODEL FAILS — the one the 2026-08-15 round did
    not have, and the reason it installed the wrong rule and shipped it.

    Every row of :data:`SEPARATING_ROWS` has an exact product in
    ``[2**-126 - 2**-151, 2**-126)``: both operands NORMAL (so DAZ cannot reach
    the row and this is not a statement about operand flushing), magnitude below
    the smallest normal, and a 24-bit unbounded-exponent rounding that ties UP
    onto 2**-126 because the grid there is 2**-150 and 2**24 is the even
    neighbour. A rule that tests the EXACT value returns a sign-preserving zero
    on every one of them. The measured rule delivers 0x00800000, and so does the
    hardware: 37,439 independently constructed vectors on x86 mulss /
    vfmadd213ss under MXCSR and on PTX mul.rn.ftz.f32 / fma.rn.ftz.f32, agreeing
    with each other 37439/37439, disagree with the before-rounding rule on 10
    and with this one on 0.

    The wrong model is RUN here, not described, so the claim that this test can
    fail is itself checked: it must answer the zero, and the installed model
    must not.
    """
    assert len(SEPARATING_ROWS) == 20
    minimum_normal = fractions.Fraction(1, 1 << 126)
    tie_endpoint = minimum_normal - fractions.Fraction(1, 1 << 151)

    # The construction recipe in SEPARATING_ROWS' own docstring, driven: the
    # 2**48 case is empty because the largest product of two 24-bit significands
    # falls short of the window, which is why every row below sits at 2**47.
    assert ((1 << 24) - 1) ** 2 < (1 << 48) - (1 << 23)

    for word_a, word_b in SEPARATING_ROWS:
        # The row is in the class by ARITHMETIC, not by assertion.
        assert ((word_a >> 23) & 0xFF) not in (0, 0xFF), hex(word_a)
        assert ((word_b >> 23) & 0xFF) not in (0, 0xFF), hex(word_b)
        significands = (((word_a & 0x7FFFFF) | (1 << 23))
                        * ((word_b & 0x7FFFFF) | (1 << 23)))
        assert (1 << 47) - (1 << 22) <= significands < (1 << 47), (
            f"0x{word_a:08x} * 0x{word_b:08x}: ma*mb = {significands} is "
            f"outside [2**47 - 2**22, 2**47), so the recipe is not what the "
            f"constant claims")
        assert ((((word_a >> 23) & 0xFF) - 127)
                + (((word_b >> 23) & 0xFF) - 127)) == -127, (
            f"0x{word_a:08x} * 0x{word_b:08x}: the unbiased exponents must sum "
            f"to -127 to put the product at the 2**-126 scale")
        exact = (gate_module._fraction_of_word(word_a)
                 * gate_module._fraction_of_word(word_b))
        assert tie_endpoint <= abs(exact) < minimum_normal, (
            f"0x{word_a:08x} * 0x{word_b:08x} is not in the separating window")
        assert gate_module._round_24_significant_bits(
            abs(exact)) == minimum_normal, (
            f"0x{word_a:08x} * 0x{word_b:08x}: the 24-bit unbounded-exponent "
            f"rounding must tie UP onto 2**-126 or this is not a separating row")

        sign = (word_a ^ word_b) & 0x80000000
        operands = (gate_module._float32_from_word(word_a),
                    gate_module._float32_from_word(word_b))
        got = _ftz_mul_word(gate_module, *operands)
        assert got == (0x00800000 | sign), (
            f"0x{word_a:08x} * 0x{word_b:08x}: the exact product is below "
            f"2**-126 but rounds ONTO it with an unbounded exponent, so the "
            f"result is the smallest normal 0x{0x00800000 | sign:08x}; the "
            f"model answered 0x{got:08x}. Answering 0x{sign:08x} means "
            f"tininess is being tested on the exact value, which both hardware "
            f"executors refute.")
        # The fused arm reaches the same product with c = +0.0.
        assert _ftz_fma_word(gate_module, *operands, 0.0) == (0x00800000 | sign)
        # ... and the before-rounding rule, RUN, answers the zero. If this ever
        # stops holding, the wrong model has drifted and the test above has
        # stopped being able to fail.
        assert _before_rounding_mul_word(gate_module, *operands) == sign
        # The delivered-word rule happens to agree with the measured one here —
        # this class does not separate those two, which is exactly why
        # TIE_ROWS is needed as well.
        assert _delivered_word_mul_word(
            gate_module, *operands) == (0x00800000 | sign)

    # KEEP is the control: with subnormals kept the same products are ordinary
    # normal results and no policy question arises.
    for word_a, word_b in SEPARATING_ROWS:
        sign = (word_a ^ word_b) & 0x80000000
        assert gate_module._word_of_float32(gate_module.mul32(
            gate_module._float32_from_word(word_a),
            gate_module._float32_from_word(word_b))) == (0x00800000 | sign)


def test_all_three_rules_agree_on_the_ordinary_band(gate_module):
    """THE CONTROL THAT PASSES UNDER ALL THREE RULES.

    Without it the pair above could be satisfied by a model that simply flushes
    on one class and keeps on the other while getting the ordinary band wrong;
    with it, the suite states positively where the rules coincide. Nothing here
    is in either separating class:

    * an exact product OF 2**-126 (0x01000000 * 0.5) — the boundary belongs to
      the normal side under all three;
    * 2**-126 + 2**-149, just above;
    * 2**-126 - 2**-149 (0x00FFFFFE * 0.5), a full subnormal ulp below, which is
      exactly representable and tiny under all three;
    * deep-subnormal and ordinary-normal products, and both signed zeros;
    * every sum and difference of float32s, which provably cannot separate the
      rules at all: each is an integer multiple of 2**-149, so a sum whose
      magnitude is BELOW 2**-126 has |k| < 2**23 and is exactly representable in
      24 significant bits — no rounding occurs and all three rules read the same
      value. (Above 2**-126 a sum can need more than 24 bits, but then it is not
      tiny under any of them; both halves are driven below.)
    """
    rows = [(0x01000000, 0.5), (0x81000000, 0.5), (0x01000001, 0.5),
            (0x00FFFFFE, 0.5), (0x80FFFFFE, 0.5), (0x00FFFFFD, 0.5),
            (0x00800000, 0.5), (0x00800000, 0.25), (0x3F800000, 0.5),
            (0x40490FDB, 0.75), (0x00000001, 0.5), (0x00000001, 2.0),
            (0x00000000, 0.5), (0x80000000, 0.5), (0x00FFFFFF, 1.0),
            (0x00FFFFFF, 2.0), (0x24042108, 1.0), (0x1BF80000, 1.0)]
    for word, scalar in rows:
        operand = gate_module._float32_from_word(word)
        measured = _ftz_mul_word(gate_module, operand, scalar)
        assert _before_rounding_mul_word(
            gate_module, operand, scalar) == measured, (
            f"0x{word:08x} * {scalar}: the before-rounding rule differs here, "
            f"so this row is not the control it claims to be")
        assert _delivered_word_mul_word(
            gate_module, operand, scalar) == measured, (
            f"0x{word:08x} * {scalar}: the delivered-word rule differs here, "
            f"so this row is not the control it claims to be")

    # Addition and subtraction: the three rules coincide by construction, and
    # the coincidence is driven rather than asserted, over the whole band.
    for left_word in (0x00800000, 0x00FFFFFF, 0x007FFFFF, 0x00000001,
                      0x01000000, 0x3F800000, 0x00000000, 0x80000000):
        for right_word in (0x00800000, 0x80800000, 0x00000001, 0x80000001,
                           0x00FFFFFF, 0x80FFFFFF, 0x01000000, 0x81000000):
            left = gate_module._float32_from_word(left_word)
            right = gate_module._float32_from_word(right_word)
            exact = (gate_module._fraction_of_word(
                _flushed_operand_word(gate_module, left))
                + gate_module._fraction_of_word(
                    _flushed_operand_word(gate_module, right)))
            if exact == 0:
                continue
            minimum_normal = fractions.Fraction(1, 1 << 126)
            rounded = gate_module._round_24_significant_bits(abs(exact))
            if abs(exact) < minimum_normal:
                assert rounded == abs(exact), (
                    f"0x{left_word:08x} + 0x{right_word:08x}: a TINY exact "
                    f"float32 sum needed rounding to 24 significant bits, "
                    f"which contradicts the derivation the add/sub arms rest "
                    f"on — that every such sum is an integer multiple of "
                    f"2**-149 with |k| < 2**23")
            else:
                assert rounded >= minimum_normal, (
                    f"0x{left_word:08x} + 0x{right_word:08x}: a sum at or "
                    f"above 2**-126 rounded below it, which rounding cannot do")
            # ... and therefore the tininess verdict is the same under all
            # three rules, which is the claim that matters.
            delivered = gate_module._round_float32_word(
                exact, 1.0 if exact > 0 else -1.0)
            delivered_tiny = (((delivered >> 23) & 0xFF) == 0
                              and (delivered & 0x7FFFFF) != 0)
            assert delivered_tiny == (abs(exact) < minimum_normal) \
                == (rounded < minimum_normal), (
                f"0x{left_word:08x} + 0x{right_word:08x}: the three rules "
                f"disagree on an ADDITION, which the derivation says is "
                f"impossible")


def test_the_correction_moves_only_the_tie_class(gate_module):
    """WHAT THE CORRECTION ACTUALLY MOVED, on the probe's own vectors at the
    probe's own seed — the number a reviewer asks for.

    Under flush, the corrected model changes 128 candidate words per arm on ONE
    of the six probe patterns (``python_float_left`` at scalar 0.5) and nothing
    anywhere else. Every changed row carries ``zr`` in {0x00FFFFFF, 0x80FFFFFF}
    and sits inside the ``_NORMAL_UNDERFLOWING_ZR`` block appended on
    2026-08-15 (index >= 2280) — which is also why the defect hid until that
    block existed: no earlier vector reached the tie.

    Under KEEP nothing changes at all, which is the statement that this is a
    correction to the flush model and not a change of arithmetic.
    """
    rng = numpy.random.default_rng(gate_module.SEED + 71)
    c8 = gate_module._c8_operands(rng)
    zero_imag = gate_module._zero_imag_operands(rng)
    legacy = zero_imag[0].size - gate_module._normal_underflow_rows()[0].size

    def old_arms(zr, zi, c, field_left, flush):
        """The same candidate arms with the OLD tininess convention put back.

        Substituted at ``candidate_arithmetic`` — the seam the arm builders
        read — so nothing here depends on how the corrected file spells its
        internals, and the substitution can be pointed at an earlier copy of the
        gate unchanged.
        """
        if not flush:
            return gate_module._candidates_zero_imag(
                zr, zi, c, field_left=field_left, flush=False)
        original = gate_module.candidate_arithmetic
        gate_module.candidate_arithmetic = (
            lambda flush=None: _delivered_word_ops(gate_module))
        try:
            return gate_module._candidates_zero_imag(
                zr, zi, c, field_left=field_left, flush=True)
        finally:
            gate_module.candidate_arithmetic = original

    changed = {}
    for flush in (True, False):
        for scalar in (0.35, 0.5, -0.35, 0.1):
            column = numpy.full(zero_imag[0].shape, numpy.float32(scalar),
                                dtype=numpy.float32)
            new = gate_module._candidates_zero_imag(
                zero_imag[0], zero_imag[1], column, field_left=False,
                flush=flush)
            old = old_arms(zero_imag[0], zero_imag[1], column, False, flush)
            for arm in ("FMA_V1", "NAIVE"):
                indices = []
                for plane in (0, 1):
                    left = numpy.ascontiguousarray(new[arm][plane]).view(
                        numpy.uint32)
                    right = numpy.ascontiguousarray(old[arm][plane]).view(
                        numpy.uint32)
                    indices += list(numpy.nonzero(left != right)[0])
                changed[(flush, scalar, arm)] = indices

    for key, indices in changed.items():
        flush, scalar, _ = key
        if flush and scalar == 0.5:
            assert len(indices) == 128, (key, len(indices))
            assert min(indices) >= legacy, (
                "a row outside the class added on 2026-08-15 changed: the "
                "correction is supposed to reach only the underflowing-product "
                "block")
            zr_words = {gate_module._float32_word(zero_imag[0][index])
                        for index in indices}
            assert zr_words == {0x00FFFFFF, 0x80FFFFFF}, (
                f"the corrected model moved rows outside the predicted tie "
                f"class: {sorted(hex(word) for word in zr_words)}")
        else:
            assert indices == [], (key, indices[:8])

    # The other two orientations and the full-complex pattern do not move at
    # all: their coefficient column never produces the tie.
    for flush in (True, False):
        for field_left in (True, False):
            new = gate_module._candidates_zero_imag(
                *zero_imag, field_left=field_left, flush=flush)
            old = old_arms(*zero_imag, field_left=field_left, flush=flush)
            for arm in ("FMA_V1", "NAIVE"):
                assert gate_module._disagreement_words(
                    new[arm], old[arm]) == 0, (flush, field_left, arm)
    assert gate_module._arms_apart(
        gate_module._candidates_c8(*c8, flush=True)) == 1088


def _arms_apart_map(gate_module, c8, zero_imag):
    """{(pattern, flush): words the two LICENSABLE arms differ in} on one set."""
    apart = {}
    for flush in (True, False):
        arms = gate_module._candidates_c8(*c8, flush=flush)
        apart[("c8_mul_c8", flush)] = gate_module._disagreement_words(
            arms["FMA_V1"], arms["NAIVE"])
        arms = gate_module._candidates_zero_imag(*zero_imag, field_left=True,
                                                 flush=flush)
        apart[("mixed", flush)] = gate_module._disagreement_words(
            arms["FMA_V1"], arms["NAIVE"])
        apart[("planewise", flush)] = gate_module._disagreement_words(
            arms["FMA_V1"], arms["PLANEWISE_diagnostic"])
    return apart


def test_the_legacy_vector_set_was_blind_to_the_arms_under_flush(gate_module):
    """THE DEFECT, kept measurable so the repair below cannot silently regress.

    Until 2026-08-15 the probe's zero-imaginary operands were exactly these, and
    under the flush policy the two licensable arms were bit-identical on both
    mixed-dtype patterns — 0 words apart of 4560. The gate reported
    AMBIGUOUS_BOTH there and ``expansion_license`` licensed the arm from
    ``c8_mul_c8`` instead, a DIFFERENT compiled orientation, with no evidence at
    all for ``_mul_field_left`` / ``_mul_coefficient_left`` under the policy in
    force. The cause was not the arithmetic: every row here that could separate
    the arms feeds the device a SUBNORMAL OPERAND (``_SMALL``, 1e-45…1e-40),
    which the ship policy destroys on first use.
    """
    rng = numpy.random.default_rng(gate_module.SEED + 71)
    c8 = gate_module._c8_operands(rng)
    full = gate_module._zero_imag_operands(rng)
    added = gate_module._normal_underflow_rows()[0].size
    legacy = tuple(column[:-added] for column in full)
    assert legacy[0].size == 2280, "the legacy vector set changed shape"
    assert gate_module.subnormal_words(legacy[0]) == 72

    apart = _arms_apart_map(gate_module, c8, legacy)
    assert apart[("mixed", True)] == 0, apart      # <- blind under the ship policy
    assert apart[("mixed", False)] == 6, apart     # <- and barely sighted under keep
    assert apart[("c8_mul_c8", True)] == 1088, apart
    assert apart[("c8_mul_c8", False)] == 1088, apart
    # It never lost its power against a PLANE-WISE dispatch; only the
    # FMA_V1-vs-NAIVE separation went, which is the separation that licenses.
    assert apart[("planewise", True)] == 89, apart
    assert apart[("planewise", False)] == 71, apart


def test_every_pattern_discriminates_the_arms_under_BOTH_policies(gate_module):
    """THE REPAIR. The probe now carries all-normal operands whose PRODUCT
    underflows (``_NORMAL_UNDERFLOWING_ZR``), so no row it depends on for
    discriminating power can be destroyed by a flushing host — the operands are
    normal, only the result underflows, and the arms split on the sign of the
    resulting zero.

    This is the assertion that makes a licence for the mixed orientations
    MEASURED rather than inherited: a zero on any row here means that pattern
    cannot classify anything, whatever the device returns."""
    rng = numpy.random.default_rng(gate_module.SEED + 71)
    c8 = gate_module._c8_operands(rng)
    zero_imag = gate_module._zero_imag_operands(rng)

    added = gate_module._normal_underflow_rows()
    assert added[0].size == 512
    for column in added:  # the whole point: nothing here is destroyed by a flush
        assert gate_module.subnormal_words(column) == 0

    apart = _arms_apart_map(gate_module, c8, zero_imag)
    for flush in (True, False):
        for pattern in ("c8_mul_c8", "mixed", "planewise"):
            assert apart[(pattern, flush)] > 0, (pattern, flush, apart)
    assert apart[("mixed", True)] == 128, apart    # was 0
    assert apart[("mixed", False)] == 74, apart    # was 6
    assert apart[("c8_mul_c8", True)] == 1088, apart
    assert apart[("c8_mul_c8", False)] == 1088, apart


def test_a_probe_whose_operands_cannot_tell_the_arms_apart_refuses(
        gate_module, monkeypatch):
    """The gate must not EMIT a record from a comparison with no power.

    The pre-existing census counts subnormal operand ENCODING, and under flush it
    read 72 while the mixed patterns had exactly zero power to classify — the
    words survived being written and were flushed the instant they were used. So
    the census could not have caught this. The new guard measures the power
    itself, candidate against candidate, under the policy in force."""
    real = gate_module._zero_imag_operands

    def legacy_only(rng):
        full = real(rng)
        added = gate_module._normal_underflow_rows()[0].size
        return tuple(column[:-added] for column in full)

    monkeypatch.setattr(gate_module, "_zero_imag_operands", legacy_only)
    # Resolve FLUSH so the candidate arms are built the way the ship
    # configuration builds them; this laptop's arm64 FPU keeps.
    monkeypatch.setattr(gate_module, "resolved_policy",
                        lambda: gate_module._policy.FLUSH)
    with pytest.raises(RuntimeError, match="cannot tell FMA_V1 from NAIVE"):
        gate_module.measure_expansion_record(numpy, "numpy")


def test_a_process_that_destroyed_the_probe_vectors_refuses_to_classify(
        gate_module, monkeypatch):
    """The census is the operand-destruction blocker made visible: without it a
    flushing process would classify a strictly weaker question in silence."""
    real = gate_module._zero_imag_operands

    def flushed(rng):
        return tuple(gate_module.flush_subnormals(a) for a in real(rng))

    monkeypatch.setattr(gate_module, "_zero_imag_operands", flushed)
    with pytest.raises(RuntimeError, match="destroyed the operands"):
        gate_module.measure_expansion_record(numpy, "numpy")


def test_the_record_states_the_candidate_policy_and_the_environment(gate_module):
    """Everything the licence depends on is IN the artifact, so a reader on
    another machine can audit it: which policy the candidates were cut under,
    where the record was measured, and how many subnormal words its own operands
    still carried."""
    record = gate_module.measure_expansion_record(numpy, "numpy")
    assert record["candidates"]["policy"] in ("flush", "keep")
    assert record["candidates"]["semantics"]
    assert record["environment"]["backend"] == "numpy"
    assert record["environment"]["machine"]
    assert record["environment"]["keyed_on"] == [
        "backend", "machine", "cupy_version"]
    assert record["vector_census"]["subnormal_words"]["zr"] == 72
    assert record["vector_census"]["interleave_differing_words"] == 0
    for name, detail in record["detail"].items():
        entries = detail if isinstance(detail, list) else [detail]
        for entry in entries:
            assert "licensable_arms_disagreement_words" in entry, name
            assert entry["discriminates"] == (
                entry["licensable_arms_disagreement_words"] > 0)


def test_the_record_the_gate_emits_can_satisfy_the_licence_rule_on_every_pattern(
        complex_module, gate_module):
    """THE TWO HALVES MUST FIT. ``expansion_license`` now requires a positive
    vector count and a detail block behind any AMBIGUOUS_BOTH; if the gate does
    not write them for some pattern, an HONEST record refuses on a bookkeeping
    gap rather than on a measurement — a false refusal is as much a defect as a
    false licence.

    Caught exactly that: the ``python_float_left`` leg records its detail as a
    LIST (one entry per scalar) and wrote no ``vectors`` entry at all, the only
    one of the four missing one."""
    record = gate_module.measure_expansion_record(numpy, "numpy")
    for name in complex_module.PROBE_PATTERNS:
        assert record["vectors"].get(name), f"no vector count for {name!r}"
        assert record["detail"].get(name), f"no detail block for {name!r}"

    # Force each pattern to AMBIGUOUS_BOTH in turn, with its own detail made
    # consistent, and check the record's evidence carries the claim.
    for name in complex_module.PROBE_PATTERNS:
        probe = json.loads(json.dumps(record))
        probe["backend"] = "cupy"
        probe["subnormal_policy"] = {"policy": "p", "resolved":
                                     probe["candidates"]["policy"]}
        probe["patterns"][name] = complex_module.AMBIGUOUS_BOTH
        entries = (probe["detail"][name] if isinstance(probe["detail"][name], list)
                   else [probe["detail"][name]])
        for entry in entries:
            entry["licensable_arms_disagreement_words"] = 0
            entry["discriminates"] = False
            entry["matches"] = {"FMA_V1": True, "NAIVE": True,
                                "PLANEWISE_diagnostic": False}
            entry["mismatch_words"] = {"FMA_V1": 0, "NAIVE": 0,
                                       "PLANEWISE_diagnostic": 17}
        reasons = complex_module._ambiguity_evidence_reasons(probe, name)
        assert reasons == [], (name, reasons)


def test_the_strip_installs_before_any_cupy_compile_in_both_artifact_cutters():
    """Ordering is the mechanism: a strip installed after the first compile
    certifies a mix. In the gate, install_ftz_strip precedes every leg; in the
    composition probe it precedes guard_kernel_compilation (the guard wraps
    whatever is at the seam, so strip-then-guard applies both)."""
    gate_source = GATE_PATH.read_text(encoding="utf-8")
    main_body = gate_source[gate_source.index("\ndef main("):]
    assert main_body.index("install_ftz_strip()") < main_body.index(
        "write_provenance("), "the strip must be the first thing main() does"
    assert main_body.index("install_ftz_strip()") < main_body.index(
        "run_expansion(")
    # The license check runs before the constexpr is bound, inside the leg.
    expansion_leg = gate_source[gate_source.index("\ndef run_expansion("):
                                gate_source.index("\ndef complex_shift(")]
    assert expansion_leg.index("ftz_strip_license_reasons()") < \
        expansion_leg.index("expansion_license(")

    composition = COMPOSITION_PATH.read_text(encoding="utf-8")
    assert composition.index("gate.install_ftz_strip()") < composition.index(
        "backends.guard_kernel_compilation(cp)")
    assert "gate.probe_record_policy_reasons(record)" in composition, (
        "a handed-in probe artifact must pass the policy check")


def test_both_artifacts_state_the_policy_they_certified_under():
    """The instruction the diagnosis closed on: every artifact citing the
    FMA_V1 license states the stripped-policy dependence explicitly."""
    gate_source = GATE_PATH.read_text(encoding="utf-8")
    assert 'results["subnormal_policy"] = policy_stamp("cupy")' in gate_source
    assert '"certified_under_subnormal_policy"' in gate_source
    composition = COMPOSITION_PATH.read_text(encoding="utf-8")
    assert 'gate.policy_stamp("cupy")' in composition
    assert '"certified_under_subnormal_policy"' in composition


# ---------------------------------------------------------------------------
# The engine phase-table mutation — asserted where the defect is byte-visible
# ---------------------------------------------------------------------------
#
# Job 2329 measured the constraint the gate's own m1 note documents: at the
# exact Brillouin edge the naive (cmath-from-k) phase differs from
# grid.bloch_phase's -1+0j by 1.2246e-16j — sub-half-ulp of every nonzero
# float32 word — so a field-level catch on a generically seeded state is
# impossible BY the gate's design (0/15360 device words moved; the run's one
# red expectation). The expectation therefore lives at the layers where the
# seam IS byte-visible — the table itself as complex128 bytes, and the fields
# on a purely imaginary state — and these tests pin that shape AND measure
# every layer's verdict on the NumPy transcription (pinned bit-identical to
# stepping.py by the gate's reference leg), so the device is promised nothing
# the laptop has not already seen move.

def test_the_phase_table_seam_comparator_is_complex128_bytes(gate_module):
    edge = complex(-1.0, 0.0)
    naive_edge = cmath.exp(2j * cmath.pi * 0.25 * 2.0)
    assert naive_edge != edge  # differs only in the 1.2246e-16j imaginary word
    assert gate_module.phase_tables_bytes_differ((naive_edge, None, None),
                                                 (edge, None, None))
    # The field product against a word with NONZERO components erases the
    # delta (sub-half-ulp of every such float32 word) — which is why the seam
    # comparator, not a generic-state field comparison, carries this catch.
    word = numpy.complex64(0.7 - 0.3j)
    assert ((word * numpy.complex64(edge)).tobytes()
            == (word * numpy.complex64(naive_edge)).tobytes())
    # An exactly-zero real component is the one operand class that survives:
    # the imag_only field-level case is built from it.
    zero_real = numpy.complex64(0.0 - 0.3j)
    assert ((zero_real * numpy.complex64(edge)).tobytes()
            != (zero_real * numpy.complex64(naive_edge)).tobytes())
    generic = cmath.exp(2j * cmath.pi * 0.74)
    assert not gate_module.phase_tables_bytes_differ((generic, None, None),
                                                     (generic, None, None))
    # A None mismatch is a table defect too (a phased axis gone silent).
    assert gate_module.phase_tables_bytes_differ((None, None, None),
                                                 (generic, None, None))
    assert not gate_module.phase_tables_bytes_differ((None,) * 3, (None,) * 3)


def test_the_engine_phase_mutation_cases_pin_seam_catch_and_recorded_nulls(
        gate_module):
    """The expectation shape: the defect MUST be caught at the table seam on
    the edge grid, MUST be caught at field level on the imag_only
    construction, and every field-level null carries its recorded prediction
    (the class m5 and null_phase_on_metallic_axis belong to)."""
    cases = {(c["grid"], c["state"]): c
             for c in gate_module.ENGINE_PHASE_MUTATION_CASES}
    assert set(cases) == {("brillouin_edge_x", "seeded"),
                          ("brillouin_edge_x", "imag_only"),
                          ("bloch_one_axis", "seeded")}
    assert cases[("brillouin_edge_x", "seeded")]["table_must_differ"] is True
    assert cases[("brillouin_edge_x", "imag_only")]["table_must_differ"] is True
    assert cases[("brillouin_edge_x", "imag_only")]["must_catch"] is True
    # Generic k IS the cmath route (grid.py:1017): inert at every layer.
    assert cases[("bloch_one_axis", "seeded")]["table_must_differ"] is False
    for key, case in cases.items():
        if case["must_catch"] is False:
            assert case.get("null_reason"), (
                f"{key}: a field-level null must be recorded with its "
                f"prediction, never silently expected")
    assert "sub-half-ulp" in cases[("brillouin_edge_x", "seeded")]["null_reason"]


def _run_phase_mutation_on_the_transcription(gate_module, case):
    """The device leg's comparison, on NumPy: the gate's transcription driven
    with the MUTATED table against stepping with the grid's own. The
    transcription is pinned bit-identical to stepping by the gate's reference
    leg, so any word that moves here is the phase table's doing alone."""
    grid, fields, reference, pml = gate_module.build_phase_mutation_case(
        numpy, case)
    kinds = gate_module.stepping._boundary_kinds(grid, pml)
    naive = gate_module.naive_phase_table(grid)
    parts = {}
    for sub_step in gate_module.CURL_SUB_STEPS:
        arrays = {name: getattr(fields, name)
                  for name in gate_module.FIELD_12 + gate_module.FU_NAMES}
        coefficients = gate_module.probe.layer_coefficients(
            pml, sub_step == "step_B")
        gate_module.complex_pml_step(numpy, arrays, coefficients,
                                     float(grid.dt / grid.dx), sub_step,
                                     kinds, naive)
        getattr(gate_module.stepping, sub_step)(reference, pml)
        targets = (("Bx", "By", "Bz") if sub_step == "step_B"
                   else ("Dx", "Dy", "Dz"))
        for name in targets + tuple("fu_" + t for t in targets):
            parts[sub_step + ":" + name] = gate_module.bit_compare(
                getattr(fields, name), getattr(reference, name))
    return grid, gate_module.combine(parts)


def test_every_engine_phase_mutation_expectation_is_measured_on_numpy(
        gate_module):
    """Both layers of every case, measured, not asserted from theory: the
    table-byte verdicts AND the field-level verdicts (the seeded null, the
    imag_only catch, the generic-k null) must come out exactly as the gate
    will assert them on the device."""
    for case in gate_module.ENGINE_PHASE_MUTATION_CASES:
        grid, verdict = _run_phase_mutation_on_the_transcription(
            gate_module, case)
        naive = gate_module.naive_phase_table(grid)
        table = tuple(grid.bloch_phase(axis) for axis in range(3))
        assert (gate_module.phase_tables_bytes_differ(naive, table)
                == case["table_must_differ"]), (case["grid"], case["state"])
        caught = not verdict["bit_identical"]
        assert caught == case["must_catch"], (
            case["grid"], case["state"],
            f"differing={verdict['differing_floats']}"
            f"/{verdict['total_floats']}")
        if case["must_catch"]:
            assert verdict["differing_floats"] > 0, (
                "the imag_only construction moved no word — the field-level "
                "leg would be hollow on the device")
