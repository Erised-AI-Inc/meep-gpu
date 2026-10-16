"""Merge-bar tests for :mod:`meep_gpu.triton_kernels.complex_fields`.

Everything here runs on a laptop with NO GPU and NO Triton, which is the point:
the kernels' bytes are the CUDA gate's to certify
(``parity/meep_gpu/gate_triton_complex.py``, run on a CUDA host);
what a laptop can hold is (a) the optional dependency staying optional, (b) the
predicates — pure Python, and the half that decides whether a silent wrong
answer is possible, (c) the host-side phase and probe arithmetic, which is
NumPy-checkable, and (d) the structural pins on the kernel source that the byte
gate's verdict will be cut against, and (e) the SUBNORMAL-POLICY dependence:
the gate certifies under the policy it was cut under and that verdict does not
transfer to the other policy (bytes differ in the subnormal range), so the
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
import inspect
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


MODULE_PATH = PACKAGE_DIR / "complex_fields.py"
MODULE_NAME = "meep_gpu.triton_kernels.complex_fields"
PARITY_DIR = pathlib.Path(__file__).resolve().parents[1] / "parity" / "meep_gpu"
GATE_PATH = PARITY_DIR / "gate_triton_complex.py"
COMPOSITION_PATH = PARITY_DIR / "probe_triton_complex_composition.py"
SLURM_PATH = PARITY_DIR / "run_triton_complex_composition.slurm"


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
    verdict = module.complex_pml_curl_coverage(fields, pml, "step_B")
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
    # MEEP's start of H is a runtime argument, so the one launch that starts H
    # compiles the same specialization as every launch after it.
    assert ('\n@triton.jit(do_not_specialize=["start_h"])\n'
            'def bloch_constitutive_step(') in source


def test_the_module_docstring_states_the_wiring_it_actually_has():
    """The docstring must not tell a reader the licensing hole is unreachable.

    It said "NOT WIRED ... fastpath.plan_fast_path still returns None on every
    branch" and a test asserted that sentence was present, which made the stale
    claim load-bearing on a green suite. Measured 2026-08-15: ``launch.plan_step``
    on a complex64 Grid/Fields/PML triple emits per-arm reasons prefixed
    'complex PML: ', i.e. it calls this module's predicates, and ``fastpath``
    composes through it. So the assertion is inverted:
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


def _curl_reasons(module, fields, pml, sub_step="step_B"):
    """Curl refusal reasons with the backend clause dropped — a laptop has no CuPy.

    The filter matches the backend clause's own phrasing, not the bare word
    "cupy", because the probe refusals legitimately NAME the cupy backend.
    """
    verdict = module.complex_pml_curl_coverage(fields, pml, sub_step)
    return [r for r in verdict.reasons if "array module" not in r]


def _constitutive_reasons(module, fields, pml, side="E"):
    verdict = module.complex_constitutive_coverage(fields, pml, side)
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
        fields, pml, "step_B") is None


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


# ---------------------------------------------------------------------------
# "Does not transfer" as BEHAVIOUR, not as a string in the verdict
# ---------------------------------------------------------------------------
#
# The five assertions above are all about strings the verdict carries. The audit
# of 2026-08-15 grepped this file for a test that installs or simulates a policy
# and expects a REFUSAL and found none — which is how a licence cut under one
# policy stayed consumable under the other through a fully green suite. These are
# that test.


# ---------------------------------------------------------------------------
# A comparison with no discriminating power licenses nothing
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# A malformed artifact refuses by name and never raises
# ---------------------------------------------------------------------------


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


def test_the_connection_reads_below_the_window_under_B_and_above_it_under_D(
        complex_module):
    """``stepping._bloch_ghost_operand``: the forward (B) differences read the
    not-owned word BELOW the stored window, ``conj(eikna)``; the backward (D)
    ones the word ABOVE it, ``eikna``. Taking one factor in both directions is
    the band-structure sign error (the gate's host mutation m1)."""
    phase = cmath.exp(2j * cmath.pi * 0.3)
    kinds_b, below = complex_module.connection_arguments((phase, None, None),
                                                         backward=False)
    kinds_d, above = complex_module.connection_arguments((phase, None, None),
                                                         backward=True)
    assert kinds_b == kinds_d == (complex_module.CONNECTION_KINDS["phase"], 0, 0)
    assert below[0] == above[0] == float(numpy.float32(phase.real))
    assert above[1] == float(numpy.float32(phase.imag))
    assert below[1] == float(numpy.float32(-phase.imag))
    assert below[2:] == above[2:] == (1.0, 0.0, 1.0, 0.0)


def test_the_connection_factor_is_rounded_from_double_not_through_complex64(
        complex_module):
    """MEEP forms ``thephase = 1.0 * factor`` in ``complex<double>`` and
    connects with ``complex<realnum>(thephase)``: each part rounded once from
    double (``readback.cmul``, ``part_products.connect_parts``)."""
    from meep_gpu.readback import cmul

    phase = cmath.exp(2j * cmath.pi * 0.17)
    _, values = complex_module.connection_arguments((None, phase, None),
                                                    backward=True)
    pr, pi = cmul((1.0, 0.0), (phase.real, phase.imag))
    assert values[2:4] == (float(numpy.float32(pr)), float(numpy.float32(pi)))


@pytest.mark.parametrize("phase,kind", [
    (complex(-1.0, 0.0), "negate"),
    (cmath.exp(2j * cmath.pi * 0.5), "negate"),       # -1 + 1.2246e-16j
    (complex(1.0, 0.0), "none"),
    (complex(1.0, 5e-14), "none"),                    # within 1e-13 of +1: a copy
    (complex(1.0, 2e-13), "phase"),
])
def test_the_connection_kind_is_meeps_classification(complex_module, phase, kind):
    """``part_products.connect_parts``: within ``CONNECT_TOLERANCE`` of +1 a copy,
    of -1 a negation (sign-bit per part, no product), otherwise the product."""
    for backward in (False, True):
        kinds, values = complex_module.connection_arguments((phase, None, None),
                                                            backward)
        assert kinds[0] == complex_module.CONNECTION_KINDS[kind]
        if kind == "negate":
            assert values[:2] == (-1.0, 0.0)
        if kind == "none":
            assert values[:2] == (1.0, 0.0)


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


def _every_cell(shape):
    """Every cell of every resolved axis inside a layer chunk, both lattices."""
    from meep_gpu.chunk_map import RECORD_PRESENCE_WORD, RECORD_WORDS, record_index

    record = numpy.zeros(RECORD_WORDS, dtype=numpy.int32)
    for axis, n in enumerate(shape):
        if n > 1:
            record[RECORD_PRESENCE_WORD] |= 1 << axis
            for half in (False, True):
                record[record_index(half, axis, 1)] = n
    return record


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
        fields, pml, "step_B") is None
    assert complex_module.plan_complex_constitutive(
        fields, pml, "E") is None


def test_the_curl_plan_shape_from_arrays(complex_module):
    phase = cmath.exp(2j * cmath.pi * 0.17)
    arrays, flat = _curl_arrays("step_B")
    plan = complex_module.plan_complex_pml_curl_from_arrays(
        "step_B", arrays, flat, (0, 0, 0), (phase, None, None), 0.35,
        chunk_bounds=_every_cell((4, 3, 5)))
    assert plan.n_elem == 4 * 3 * 5, "n_elem is COMPLEX cells, not words"
    assert plan.backward == 0
    assert plan.connections == (complex_module.CONNECTION_KINDS["phase"], 0, 0)
    # B reads below the window: conj(eikna), each part rounded once from double.
    assert plan.connection_values[0] == float(numpy.float32(phase.real))
    assert plan.connection_values[1] == float(numpy.float32(-phase.imag))
    assert plan.dtdx == float(0.35)
    assert plan.chunk_bounds.tolist() == _every_cell((4, 3, 5)).tolist()
    sentinel = object()
    mutated = complex_module.plan_complex_pml_curl_from_arrays(
        "step_B", arrays, flat, (0, 0, 0), (None, None, None), 0.35, kernel=sentinel,
        chunk_bounds=_every_cell((4, 3, 5)))
    assert mutated._kernel is sentinel, "the mutation override must be retained"


def test_the_step_D_plan_takes_the_word_above_the_window_itself(complex_module):
    """The gate hands ONE phase table; the per-sub-step direction is the plan's,
    exactly as it is the engine route's: D reads above the window, ``eikna``."""
    phase = cmath.exp(2j * cmath.pi * 0.17)
    arrays, flat = _curl_arrays("step_D")
    plan = complex_module.plan_complex_pml_curl_from_arrays(
        "step_D", arrays, flat, (0, 0, 0), (phase, None, None), 0.35,
        chunk_bounds=_every_cell((4, 3, 5)))
    assert plan.backward == 1
    assert plan.connection_values[1] == float(numpy.float32(phase.imag))


def test_the_curl_plan_refuses_a_record_by_its_absence(complex_module):
    """The harness route has no default division: a plan without a chunk record
    is a refusal by name, never a whole-grid recurrence."""
    arrays, flat = _curl_arrays("step_B")
    with pytest.raises(TypeError, match="chunk_bounds"):
        complex_module.plan_complex_pml_curl_from_arrays(
            "step_B", arrays, flat, (0, 0, 0), (None, None, None), 0.35)


def test_the_curl_plan_refuses_a_connection_on_a_wall(complex_module):
    """``stepping._bloch_ghost_operand`` connects only a wrapping axis; a plan
    asked to connect a metallic axis is refused, not launched."""
    arrays, flat = _curl_arrays("step_B")
    plan = complex_module.plan_complex_pml_curl_from_arrays(
        "step_B", arrays, flat, (0, 1, 0), (None, None, None), 0.35,
        chunk_bounds=_every_cell((4, 3, 5)))
    with pytest.raises(ValueError, match="does not wrap"):
        complex_module.ComplexPmlCurlPlan(
            "step_B", plan.shape, 0.35, (0, 1, 0),
            (0, complex_module.CONNECTION_KINDS["phase"], 0),
            (1.0, 0.0, 0.8, 0.6, 1.0, 0.0), plan.block,
            [arrays[n] for n in ("Bx", "By", "Bz")],
            [arrays["fu_" + n] for n in ("Bx", "By", "Bz")],
            [arrays[n] for n in ("Ex", "Ey", "Ez")],
            [flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
            chunk_bounds=_every_cell((4, 3, 5)))


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
        "E", arrays, flat, chunk_bounds=_every_cell(shape))
    assert plan.scale == 1
    assert plan.n_elem == 4 * 3 * 5
    assert plan.start_h() == 0, "the E side never starts H"

    h_arrays = {}
    for name in ("Hx", "Hy", "Hz", "Bx", "By", "Bz"):
        h_arrays[name] = cvol()
    for name in ("Hx", "Hy", "Hz"):
        h_arrays["f_w_" + name] = cvol()
    h_plan = complex_module.plan_complex_constitutive_from_arrays(
        "H", h_arrays, flat, chunk_bounds=_every_cell(shape))
    assert h_plan.scale == 0
    assert h_plan.start_h() == 0, "a harness plan carries no run state to start"
    # The H side binds the source word views as inv_eps placeholders — the loads
    # sit behind the SCALE constexpr, but a pointer argument still has to type.
    assert h_plan._inv_eps is h_plan._sources


def test_plans_refuse_real_storage_at_construction(complex_module):
    arrays, flat = _curl_arrays("step_B")
    arrays["Bx"] = numpy.zeros((4, 3, 5), dtype=numpy.float32)
    with pytest.raises(ValueError, match="complex64"):
        complex_module.plan_complex_pml_curl_from_arrays(
            "step_B", arrays, flat, (0, 0, 0), (None, None, None), 0.35, chunk_bounds=_every_cell((4, 3, 5)))


def test_the_h_plan_starts_h_once_and_a_warm_pass_starts_nothing(complex_module,
                                                                monkeypatch):
    """MEEP's start of H (``stepping.separate_h_from_b``): the engine route's H
    plan carries the run's ``h_separated`` and passes 1 on the one launch that
    starts H; an empty launch grid (the dispatcher's warm pass) starts nothing."""
    shape = (4, 3, 5)
    rng = numpy.random.default_rng(3)
    arrays = {name: (rng.standard_normal(shape)
                     + 1j * rng.standard_normal(shape)).astype(numpy.complex64)
              for name in ("Hx", "Hy", "Hz", "Bx", "By", "Bz",
                           "f_w_Hx", "f_w_Hy", "f_w_Hz")}
    flat = {f"{stem}_{axis}": numpy.ones(n, dtype=numpy.float32)
            for axis, n in zip("xyz", shape) for stem in ("kps", "kms")}
    state = SimpleNamespace(h_separated=False)
    launches = []
    # run() reads the package's fusion guard from kernels.py, which imports
    # Triton; the guard's value is not this test's subject.
    monkeypatch.setitem(sys.modules, "meep_gpu.triton_kernels.kernels",
                        SimpleNamespace(ENABLE_FP_FUSION=False))

    class _Recorder:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append((grid, args[-1]))
            return launch

    plan = complex_module.ComplexConstitutivePlan(
        "H", shape, 128,
        [arrays[n] for n in ("Hx", "Hy", "Hz")],
        [arrays["f_w_" + n] for n in ("Hx", "Hy", "Hz")],
        [arrays[n] for n in ("Bx", "By", "Bz")], None,
        [flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kps", "kms")],
        kernel=_Recorder(), chunk_bounds=_every_cell(shape), h_state=state)
    assert plan.start_h() == 1
    saved = plan._grid
    plan._grid = (0,)
    plan.run()
    assert state.h_separated is False, "the warm pass must not start H"
    plan._grid = saved
    plan.run()
    assert state.h_separated is True
    plan.run()
    assert [start for _grid, start in launches] == [1, 1, 0]


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


def test_the_curl_is_meeps_stencil_on_both_parts():
    """``forms.curl_stencil`` per stored part: MEEP's left-to-right
    ``((g1s - g1) + g2) - g2s`` with the record's partner presence, on the
    connected unshifted operands (stepping._curl_from_operands)."""
    segment = _kernel_source_segment("bloch_pml_curl_step")
    for line in (
            "curl0_re = curl_stencil(c_y_re, c0_re, b0_re, b_z_re, dtdx, py, pz)",
            "curl0_im = curl_stencil(c_y_im, c0_im, b0_im, b_z_im, dtdx, py, pz)",
            "curl1_re = curl_stencil(a_z_re, a1_re, c1_re, c_x_re, dtdx, pz, px)",
            "curl1_im = curl_stencil(a_z_im, a1_im, c1_im, c_x_im, dtdx, pz, px)",
            "curl2_re = curl_stencil(b_x_re, b2_re, a2_re, a_y_re, dtdx, px, py)",
            "curl2_im = curl_stencil(b_x_im, b2_im, a2_im, a_y_im, dtdx, px, py)"):
        assert line in segment, line
    # The previous pairwise grouping is gone.
    assert "((c_y_re - c_re) + (b_re - b_z_re))" not in segment


def test_every_product_has_a_real_operand_and_nothing_is_fused():
    """MEEP never forms ``(c + 0i) * z`` and the build never contracts: each
    product is one real coefficient times one stored part
    (``part_products.scale``), and no ``tl.math.fma`` and no zero cross term
    remains to carry a signed zero the array path does not form."""
    source = code_of(MODULE_PATH)
    assert "tl.math.fma" not in source
    assert "* 0.0)" not in source and "(0.0 *" not in source
    assert "return (z_re * c, z_im * c)" in source
    assert "return (c * z_re, c * z_im)" in source
    segment = _kernel_source_segment("bloch_constitutive_step")
    assert "src_re = src_re * ie" in segment and "src_im = src_im * ie" in segment


def test_the_negation_is_spelled_times_minus_one_not_unary_minus():
    """Triton lowers unary ``-x`` as ``0.0 - x`` (3.1.0,
    language/semantic.py:386-391), which turns the negation of +0 into +0,
    while MEEP's CONNECT_NEGATE flips the sign bit of each part. ``* -1.0`` is
    the IEEE-exact negation (folds to neg.f32). The connection's negation is
    the kernels' only negation."""
    segment = _kernel_source_segment("_connect")
    assert "out_re = v_re * -1.0" in segment
    assert "out_im = v_im * -1.0" in segment
    for stale in ("= -v_re", "= -v_im", "0.0 - v_re", "0.0 - v_im"):
        assert stale not in segment, stale


def test_the_gate_mutation_needles_match_the_shipped_spelling(gate_module):
    """Every source mutation of the gate must still find its needles in the
    file text — a silent needle miss disarms the leg (the gate errors on 0
    hits, but the merge bar should catch the drift before a device run does).
    File text, not inspect: laptop has no Triton."""
    source = MODULE_PATH.read_text(encoding="utf-8")
    for name, (transform, _family, _must, _state) in gate_module.SOURCE_MUTATIONS.items():
        mutated, hits = transform(source)
        assert hits > 0 and mutated != source, name
    assert gate_module.mutate_connection_on_every_lane(source)[1] == 12
    assert gate_module.mutate_connection_on_the_old_wrapped_plane(source)[1] == 12


def test_the_gate_names_its_mutants_off_tritons_package_prefix(gate_module):
    """A mutant module whose name starts with ``triton`` drops a helper-only
    mutation out of the kernel's cache key (the off-diagonal gate's measured
    defect, 2026-10-08); the gate's mutants must not be named so."""
    code = inspect.getsource(gate_module.compile_mutated)
    assert '"complex_mutant_"' in code
    assert '"triton_mutated' not in code


def test_the_connection_sits_before_the_stencil():
    """The connected copy replaces the unshifted operand BEFORE the curl reads
    it (stepping._curl_operands then _curl_from_operands); source order is a
    proxy the gate's m3 (the connection dropped) backs numerically."""
    segment = _kernel_source_segment("bloch_pml_curl_step")
    connection = segment.index("r_re, r_im = _connect(c_re, c_im, cyr, cyi, CNY)")
    stencil = segment.index("curl0_re = curl_stencil(")
    assert connection < stencil


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
    # ``.forms`` is the Triton table's shared form library; ``..part_products``
    # (the connection tolerance) and ``..readback`` (MEEP's double complex product
    # forming ``thephase``) are read-only host modules, imported lazily.
    assert imported <= {"__future__", "typing", "triton", "triton.language",
                        ".coverage", ".launch", ".kernels", ".forms", "..stepping",
                        "..part_products", "..readback",
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
# The tranche was certified on random-seeded states, and the seeded
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
    assert len(sites) == 2
    for spelled, unary in sites:
        assert spelled in source, f"needle drifted: {spelled!r}"
        assert unary not in source, f"old spelling present: {unary!r}"


def test_zero_init_reference_census_reaches_negative_zero_words(
        composition_module):
    """THE REACHABILITY RESULT, re-measured at every merge: from all-+0.0
    state the NumPy array path stores hundreds of -0.0 words in the compared
    volumes on every odd step (and exactly none on even steps), in cells whose
    BOTH words are -0.0 (the per-part products' class, which a whole-complex
    product would store as ``-0 + 0i``), while a quiet region persists and the
    source separates the run from its control. The device probe re-computes
    the same census on the CuPy reference and fails the case as vacuous if it
    disagrees."""
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
            f"(re = -0.0 and imag = -0.0; measured 283 and 418 cells on "
            f"2026-10-08)")
        assert row["census_final"]["quiet_fraction"] >= 0.25
        assert set(row["census_final"]["per_array"]) & {
            "fu_Bx", "fu_Bz", "fu_Dx", "fu_Dz"}, (
            f"{case[0]}: the words left the compared fu auxiliaries: "
            f"{row['census_final']['per_array']}")
        if case[0] == "zero_init_quiet_bloch":
            least_rot = min(r["rotate_events"] for r in row["census_per_step"])
            assert least_rot >= 1, (
                "the bloch zero-init case stopped exercising the connection "
                "product's signed-zero class")


# ---------------------------------------------------------------------------
# The subnormal policy — the license is a statement about IEEE-keep bytes
# ---------------------------------------------------------------------------
#
# Measured 2026-08-11 (the expansion runs + the ftz_lanes diagnosis): CuPy 13.5.1
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
    'complex PML: ' reasons, ``fastpath._decide`` composes through it, and
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


def test_the_cache_dir_must_be_private_and_policy_suffixed(gate_module):
    """The cache key is computed above the strip seam, so a shared cache mixes
    policies silently — refused by name before anything compiles."""
    assert gate_module.ftz_cache_reasons(None)
    assert gate_module.ftz_cache_reasons("")
    reasons = gate_module.ftz_cache_reasons("/x/cupy_cache_12345")
    assert reasons and gate_module.POLICY_CACHE_TOKEN in reasons[0]
    assert gate_module.ftz_cache_reasons(
        "/x/cupy_cache_ftz_stripped_12345") == []


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


def test_the_strip_installs_before_any_cupy_compile_in_the_gate_and_the_composition():
    """Ordering is the mechanism: a strip installed after the first compile
    certifies a mix. In the gate, install_ftz_strip precedes every leg; in the
    composition probe it precedes guard_kernel_compilation (the guard wraps
    whatever is at the seam, so strip-then-guard applies both)."""
    gate_source = GATE_PATH.read_text(encoding="utf-8")
    main_body = gate_source[gate_source.index("\ndef main("):]
    assert main_body.index("install_ftz_strip()") < main_body.index(
        "write_provenance("), "the strip must be the first thing main() does"

    composition = COMPOSITION_PATH.read_text(encoding="utf-8")
    assert composition.index("gate.install_ftz_strip()") < composition.index(
        "backends.guard_kernel_compilation(cp)")


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
# A device run measured the constraint the gate's own m1 note documents: at the
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
    # MEEP's connection tolerance classifies the naive edge phase as the same
    # negation, so no state can see it at field level: a recorded null.
    assert cases[("brillouin_edge_x", "imag_only")]["must_catch"] is False
    assert "CONNECT_TOLERANCE" in cases[("brillouin_edge_x", "imag_only")]["null_reason"]
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
                                     kinds, naive, gate_module.layer_record(grid, pml))
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
                "a must-catch construction moved no word — the field-level "
                "leg would be hollow on the device")
