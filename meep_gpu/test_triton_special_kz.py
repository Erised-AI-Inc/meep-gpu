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


def kernel_body_and_insert(path, name, flag="HAS_BETA"):
    """``(body lines, insert lines)`` of one shipped kernel, read from the FILE.

    The insert is the kernel's top-level ``if <flag>:`` statement (the fold pass the
    lifters splice in); the body is every other top-level statement after the
    docstring. Each is the exact source text split into stripped, comment-free code
    lines -- never ``ast.unparse``d, so the parenthesisation is what is compared. The
    fused and folded lifters' transcription tests use this to check that removing the
    insert leaves the root body and that the insert is its curl's own.
    """
    import ast  # noqa: PLC0415

    text = pathlib.Path(path).read_text(encoding="utf-8")
    node = next(found for found in ast.walk(ast.parse(text))
                if isinstance(found, ast.FunctionDef) and found.name == name)
    statements = node.body
    if (isinstance(statements[0], ast.Expr)
            and isinstance(statements[0].value, ast.Constant)
            and isinstance(statements[0].value.value, str)):
        statements = statements[1:]

    def lines(nodes):
        out = []
        for statement in nodes:
            for raw in ast.get_source_segment(text, statement).splitlines():
                code = raw.split("#", 1)[0].strip()
                if code:
                    out.append(code)
        return out

    insert = [s for s in statements
              if isinstance(s, ast.If) and getattr(s.test, "id", "") == flag]
    body = [s for s in statements if s not in insert]
    return lines(body), lines(insert)


@pytest.fixture(scope="module")
def sk():
    """The shipped module itself must be importable without a test-only stub."""
    return importlib.import_module(MODULE_NAME)


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


def _complex_reasons(sk, fields, pml, sub_step="step_B"):
    verdict = sk.beta_bloch_pml_curl_coverage(fields, pml, sub_step)
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
    assert sk.DEFAULT_BLOCK == complex_fields.DEFAULT_BLOCK == 256
    # Read kernels.py off the source (importing it needs Triton).
    kernels_source = (PACKAGE_DIR / "kernels.py").read_text(encoding="utf-8")
    assert "\nPERIODIC = tl.constexpr(0)\n" in kernels_source
    assert "\nMETALLIC = tl.constexpr(1)\n" in kernels_source
    assert "\nDEFAULT_BLOCK = 256\n" in kernels_source


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
        cfields, cpml, side)
    assert [r for r in verdict.reasons if "array module" not in r] == []


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


def _quarter_turn_increment(word, partner):
    """The kernel's per-part spelling of the negated quarter turn: ``(c*y, -(c*x))``."""
    c = numpy.float32(word)
    out = numpy.empty_like(partner)
    out.real[...] = c * partner.imag
    out.imag[...] = -(c * partner.real)
    return out


@pytest.mark.parametrize("beta,dt", [(BETA_SPECIAL_KZ, 0.035), (0.2, 0.05)])
@pytest.mark.parametrize("magnetic,sign", [(True, +1.0), (True, -1.0),
                                           (False, +1.0), (False, -1.0)])
def test_the_complex_coefficient_matches_stepping_bytes(sk, beta, dt, magnetic,
                                                        sign):
    """The plan's one real word per sign, formed into the kernel's per-part quarter
    turn, reproduces stepping's complex64 increment word for word -- on a partner
    carrying both zero signs in both parts, the cases a cross term would flip."""
    grid = SimpleNamespace(beta=beta, dt=dt, xp=numpy)
    shim = SimpleNamespace(grid=grid, has_offdiagonal_epsilon=False)
    partner = numpy.empty((3, 3, 1), dtype=numpy.complex64)
    partner.real[...] = numpy.array([1.0, -0.0, 0.0, -0.375, 0.0, -0.0, 2.5, -0.0, 0.0],
                                    dtype=numpy.float32).reshape(3, 3, 1)
    partner.imag[...] = numpy.array([-0.0, 1.0, 0.0, 0.0, -0.0, -0.75, -0.0, 0.0, -1.5],
                                    dtype=numpy.float32).reshape(3, 3, 1)
    expected = stepping._special_kz_beta_term(shim, partner, sign, magnetic)
    plus, minus = sk.beta_curl_coefficients(beta, dt, magnetic, True)
    got = _quarter_turn_increment(plus if sign > 0 else minus, partner)
    assert got.view(numpy.uint32).tobytes() == expected.view(numpy.uint32).tobytes()


def test_the_complex_words_carry_the_sides_quarter_turn(sk):
    """No complex coefficient exists: the B side's word is the real coefficient
    itself, the D side's its negation (the quarter turn by -i), each rounded once."""
    plus_b, minus_b = sk.beta_curl_coefficients(0.332, 0.05, magnetic=True,
                                                complex_storage=True)
    plus_d, minus_d = sk.beta_curl_coefficients(0.332, 0.05, magnetic=False,
                                                complex_storage=True)
    real_plus, real_minus = sk.beta_curl_coefficients(0.332, 0.05, magnetic=True,
                                                      complex_storage=False)
    assert all(isinstance(word, float) for word in (plus_b, minus_b, plus_d, minus_d))
    assert (plus_b, minus_b) == (real_plus, real_minus)
    assert (plus_d, minus_d) == (-real_plus, -real_minus)
    expected = numpy.float32(2.0 * math.pi * 0.332 * 0.05)
    assert numpy.float32(plus_b).tobytes() == expected.tobytes()


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


def _record(shape=(4, 3, 1)):
    from meep_gpu.chunk_map import RECORD_PRESENCE_WORD, RECORD_WORDS, record_index

    record = numpy.zeros(RECORD_WORDS, dtype=numpy.int32)
    presence = 0
    for axis, n in enumerate(shape):
        if n <= 1:
            continue
        presence |= 1 << axis
        for half in (False, True):
            record[record_index(half, axis, 1)] = n
    record[RECORD_PRESENCE_WORD] = presence
    return record


def test_engine_route_plans_return_none_on_a_numpy_host(sk):
    fields, pml = _build()
    assert sk.plan_beta_pml_curl(fields, pml, "step_B") is None
    cfields, cpml = _build(complex_storage=True, beta=0.2)
    assert sk.plan_beta_bloch_pml_curl(cfields, cpml, "step_B") is None
    assert sk.plan_beta_run_constitutive(fields, pml, "H") is None
    assert sk.plan_beta_run_complex_constitutive(
        cfields, cpml, "E") is None


def test_the_real_plan_shape_from_arrays(sk):
    arrays, flat = _curl_arrays("step_D")
    plus, minus = sk.beta_curl_coefficients(BETA_KZ2D, 0.05, magnetic=False,
                                            complex_storage=False)
    plan = sk.plan_beta_pml_curl_from_arrays(
        "step_D", arrays, flat, (1, 0, 0), 0.34, plus, minus,
        chunk_bounds=_record())
    assert plan.sub_step == "step_D" and plan.backward == 1
    assert plan.bc == (1, 0, 0)
    assert plan.has_beta == 1
    assert plan.beta_plus == plus and plan.beta_minus == minus
    assert len(plan._coefficients) == 6
    assert plan.n_elem == 12
    assert plan.chunk_bounds.tolist() == _record().tolist()


def test_a_harness_plan_without_a_record_is_refused_by_name(sk):
    """The from-arrays route has no default division: a mutation leg that fell back
    to one would stop measuring the forms it exists to measure."""
    from meep_gpu.chunk_map import ChunkLayoutRefusal

    arrays, flat = _curl_arrays("step_B")
    with pytest.raises(TypeError):
        sk.plan_beta_pml_curl_from_arrays("step_B", arrays, flat, (0, 0, 0), 0.5,
                                          0.1, -0.1)
    with pytest.raises(ChunkLayoutRefusal, match="no chunk record"):
        sk.plan_beta_pml_curl_from_arrays("step_B", arrays, flat, (0, 0, 0), 0.5,
                                          0.1, -0.1, chunk_bounds=None)
    carrays, cflat = _curl_arrays("step_B", dtype=numpy.complex64)
    with pytest.raises(ChunkLayoutRefusal, match="no chunk record"):
        sk.plan_beta_bloch_pml_curl_from_arrays(
            "step_B", carrays, cflat, (0, 0, 0), (None, None, None), 0.35,
            (0.1, -0.1), chunk_bounds=None)


def test_the_complex_plan_carries_one_real_word_per_sign(sk):
    arrays, flat = _curl_arrays("step_B", dtype=numpy.complex64)
    words = sk.beta_curl_coefficients(BETA_KZ2D, 0.05, magnetic=True,
                                      complex_storage=True)
    plan = sk.plan_beta_bloch_pml_curl_from_arrays(
        "step_B", arrays, flat, (0, 0, 0), (None, None, None), 0.35, words,
        chunk_bounds=_record())
    assert plan.beta_words == tuple(words)
    assert plan.has_beta == 1
    assert plan.connections == (0, 0, 0)
    with pytest.raises(ValueError, match="quarter-turn words"):
        sk.plan_beta_bloch_pml_curl_from_arrays(
            "step_B", arrays, flat, (0, 0, 0), (None, None, None), 0.35,
            ((0.0, 0.1), (-0.0, -0.1)), chunk_bounds=_record())


def test_the_step_D_plan_connects_above_the_window(sk):
    """D's backward curl reads the not-owned word ABOVE the window: ``eikna`` itself,
    in float32 parts (complex_fields.connection_arguments)."""
    arrays, flat = _curl_arrays("step_D", dtype=numpy.complex64)
    words = sk.beta_curl_coefficients(0.2, 0.05, magnetic=False,
                                      complex_storage=True)
    phase = complex(0.8, 0.6)
    plan = sk.plan_beta_bloch_pml_curl_from_arrays(
        "step_D", arrays, flat, (0, 0, 0), (phase, None, None), 0.35, words,
        chunk_bounds=_record())
    from meep_gpu.triton_kernels.complex_fields import CONNECTION_KINDS

    assert plan.connections == (CONNECTION_KINDS["phase"], 0, 0)
    assert plan.connection_values[0] == pytest.approx(0.8)
    assert plan.connection_values[1] == pytest.approx(0.6)
    plan_b = sk.plan_beta_bloch_pml_curl_from_arrays(
        "step_B", *_curl_arrays("step_B", dtype=numpy.complex64), (0, 0, 0),
        (phase, None, None), 0.35, words, chunk_bounds=_record())
    assert plan_b.connection_values[1] == pytest.approx(-0.6)  # conj(eikna) below


def test_the_identity_arm_is_constructible(sk):
    """has_beta=0 exists for the gate's identity leg only."""
    arrays, flat = _curl_arrays("step_B")
    plan = sk.plan_beta_pml_curl_from_arrays(
        "step_B", arrays, flat, (0, 0, 0), 0.5, 0.1, -0.1, has_beta=0,
        chunk_bounds=_record())
    assert plan.has_beta == 0


# ---------------------------------------------------------------------------
# The in-test transcription — bytes against stepping.step_B / step_D
# ---------------------------------------------------------------------------
#
# A THIRD transcription (the gate's is the second), deliberately written from the
# term tables and MEEP's forms rather than imported, so agreement means the contract
# and not this file agreeing with itself. Per component: the curl by partner presence,
# the ownership mask, the form its chunk takes (the recurrence on the dsigu chunk, the
# dsig-only form elsewhere, fu touched on the recurrence's cells only); then, after
# every component, the beta pass through its ladder on targets 0 and 1.

B_TERMS = (("Bx", "Ez", 1, "Ey", 2, "y", "z"),
           ("By", "Ex", 2, "Ez", 0, "z", "x"),
           ("Bz", "Ey", 0, "Ex", 1, "x", "y"))
D_TERMS = (("Dx", "Hz", 1, "Hy", 2, "y", "z"),
           ("Dy", "Hx", 2, "Hz", 0, "z", "x"),
           ("Dz", "Hy", 0, "Hx", 1, "x", "y"))
#: target index -> (partner source key index, sign): 0 takes the SECOND source at
#: +1, 1 the FIRST at -1, 2 nothing (step_db.cpp:161-175).
BETA_PARTNERS = {0: ("second", +1.0), 1: ("first", -1.0)}


def _face(axis, index):
    return tuple(index if a == axis else slice(None) for a in range(3))


def _parts_scale(c, values):
    """A real float32 coefficient times each stored part (MEEP's DOCMP loop)."""
    if values.dtype.kind != "c":
        return numpy.float32(c) * values
    out = numpy.empty_like(values)
    out.real[...] = c * values.real
    out.imag[...] = c * values.imag
    return out


def _shift(field, axis, kind, backward):
    shifted = numpy.roll(field, 1 if backward else -1, axis=axis)
    if kind != "periodic":
        shifted[_face(axis, 0 if backward else -1)] = 0
    return shifted


def _connected(field, axis, kind, backward, phase):
    """Plane 0 of the UNSHIFTED operand as MEEP's not-owned word on a phased axis."""
    if phase is None or kind != "periodic":
        return field
    eikna = complex(phase)
    fr, fi = (eikna.real, eikna.imag) if backward else (eikna.real, -eikna.imag)
    if abs(fi) < 1e-13 and abs(fr - 1.0) < 1e-13:
        return field
    out = field.copy()
    re = field[_face(axis, 0)].real.copy()
    im = field[_face(axis, 0)].imag.copy()
    plane = numpy.empty_like(field[_face(axis, 0)])
    if abs(fi) < 1e-13 and abs(fr + 1.0) < 1e-13:
        plane.real[...] = -re
        plane.imag[...] = -im
    else:
        a, b = numpy.float32(fr), numpy.float32(fi)
        plane.real[...] = (a * re) - (b * im)
        plane.imag[...] = (a * im) + (b * re)
    out[_face(axis, 0)] = plane
    return out


def _membership(record, half, axis, shape):
    from meep_gpu.chunk_map import membership_from_record

    view = [1, 1, 1]
    view[axis] = shape[axis]
    return numpy.broadcast_to(
        membership_from_record(record, half, axis, shape[axis]).reshape(view), shape)


def _transcribed_sub_step(fields, pml, sub_step):
    """One beta curl sub-step on COPIES of the state, returned by name."""
    from meep_gpu.chunk_map import ChunkLayout, chunk_bounds_record

    grid = fields.grid
    kinds = stepping._boundary_kinds(grid, pml)
    phases = tuple(grid.bloch_phase(axis) for axis in range(3))
    record = chunk_bounds_record(ChunkLayout.one_rank(grid, pml))
    present = [not grid.is_invariant(axis) for axis in range(3)]
    backward = sub_step == "step_D"
    terms = D_TERMS if backward else B_TERMS
    sources = {name: numpy.array(getattr(fields, name))
               for name in (("Hx", "Hy", "Hz") if backward
                            else ("Ex", "Ey", "Ez"))}
    dtdx = numpy.float32(grid.dt / grid.dx)
    suffix = "" if backward else "_h"
    complex_storage = sources[next(iter(sources))].dtype.kind == "c"
    shape = tuple(grid.shape)
    state, inside = {}, {}
    for target, g1, a1, g2, a2, dsig, dsigu in terms:
        field = numpy.array(getattr(fields, target))
        fu = numpy.array(getattr(fields, "fu_" + target))
        g1s = _shift(sources[g1], a1, kinds[a1], backward)
        g2s = _shift(sources[g2], a2, kinds[a2], backward)
        g1c = _connected(sources[g1], a1, kinds[a1], backward, phases[a1])
        g2c = _connected(sources[g2], a2, kinds[a2], backward, phases[a2])
        if present[a1] and present[a2]:
            curl = _parts_scale(dtdx, ((g1s - g1c) + g2c) - g2s)
        elif present[a1]:
            curl = _parts_scale(dtdx, g1s - g1c)
        elif present[a2]:
            curl = _parts_scale(-dtdx, g2s - g2c)
        else:
            curl = numpy.zeros_like(field)
        iyee = IYEE_SHIFTS[target]
        for axis in range(3):
            if iyee[axis] == 0 and kinds[axis] == "metallic":
                curl[_face(axis, 0)] = 0
        kms = getattr(pml, f"kms_{dsig}{suffix}")
        sinv = getattr(pml, f"sinv_{dsig}{suffix}")
        kms_u = getattr(pml, f"kms_{dsigu}{suffix}")
        sinv_u = getattr(pml, f"sinv_{dsigu}{suffix}")
        du = _membership(record, not backward, "xyz".index(dsigu), shape)
        n = _parts_scale(sinv, _parts_scale(kms, fu) - curl)
        recurrence = _parts_scale(sinv_u, (_parts_scale(kms_u, field) + n) - fu)
        dsig_only = _parts_scale(sinv, _parts_scale(kms, field) - curl)
        state[target] = numpy.where(du, recurrence, dsig_only)
        state["fu_" + target] = numpy.where(du, n, fu)
        inside[target] = (du, sinv, sinv_u)
    for index, (target, g1, _a1, g2, _a2, _dsig, _dsigu) in enumerate(terms):
        if index not in BETA_PARTNERS:
            continue
        which, sign = BETA_PARTNERS[index]
        partner = sources[g2] if which == "second" else sources[g1]
        coefficient = sign * 2.0 * math.pi * grid.beta * grid.dt
        if complex_storage:
            word = numpy.float32(coefficient if not backward else -coefficient)
            increment = numpy.empty_like(partner)
            increment.real[...] = word * partner.imag
            increment.imag[...] = -(word * partner.real)
        else:
            increment = -(numpy.float32(coefficient) * partner)
        iyee = IYEE_SHIFTS[target]
        for axis in range(3):
            if iyee[axis] == 0 and kinds[axis] == "metallic":
                increment[_face(axis, 0)] = 0
        du, sinv, sinv_u = inside[target]
        df = _parts_scale(sinv, increment)
        state["fu_" + target] = numpy.where(du, state["fu_" + target] - df,
                                            state["fu_" + target])
        state[target] = numpy.where(du, state[target] - _parts_scale(sinv_u, df),
                                    state[target] - df)
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
    """stepping.step_B/step_D on a beta Fields against the in-test transcription,
    byte for byte: the chunk forms, the sign and partner tables, the coefficient
    rounding, the dtdx exclusion and the beta pass's place after every curl and its
    ladder, all pinned at once."""
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
    """The same transcription WITHOUT the beta pass must NOT match a beta run:
    proves the pass moves bytes on these seeds, so the pins above measure it."""
    fields, pml = _build(beta=BETA_KZ2D, courant=0.5)
    _seed(fields, complex_storage=False)
    zero_grid_fields, zero_pml = _build(beta=BETA_KZ2D, courant=0.5)
    _seed(zero_grid_fields, complex_storage=False)
    zero_grid_fields.grid.beta = 0.0  # stepping skips the pass
    expected = _transcribed_sub_step(fields, pml, "step_B")
    stepping.step_B(zero_grid_fields, zero_pml)
    differing = sum(
        int(numpy.asarray(getattr(zero_grid_fields, key)).tobytes()
            != value.tobytes())
        for key, value in expected.items())
    # Bx and By, and the auxiliary of the one of them whose dsigu axis carries a
    # layer (By's x; Bx's dsigu axis z is invariant and stores no layer).
    assert differing >= 3, (
        "the beta pass moved no bytes on the seeded state; the reference pins "
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


def _insert(segment: str) -> str:
    """The HAS_BETA block: from its guard to the blank line that closes it."""
    start = segment.index("    if HAS_BETA:")
    end = segment.index("\n\n", start)
    return segment[start:end]


def test_the_beta_pass_sits_after_the_forms_and_before_the_stores():
    for kernel, last_form, first_store in (
            ("beta_pml_curl_step", "v2, n2 = curl_form(",
             "tl.store(u0 + idx, n0, mask=live & du0)"),
            ("beta_bloch_pml_curl_step", "v2_im, n2_im = curl_form(",
             "tl.store(u0 + 2 * idx, n0_re, mask=live & du0)")):
        segment = _segment(kernel)
        term = segment.index("if HAS_BETA:")
        assert segment.index(last_form) < term < segment.index(first_store), (
            f"{kernel}: the beta pass must follow every curl form and precede the "
            f"stores, as MEEP's pass of its own after every component does")


def test_the_real_pass_negates_the_scaled_center_and_takes_the_ladder(sk):
    block = _insert(_segment("beta_pml_curl_step"))
    assert "q0 = (beta_plus * b) * -1.0" in block
    assert "q1 = (beta_minus * a) * -1.0" in block
    assert "v0, n0 = fold_ladder(v0, n0, q0, si_y, si_z, du0)" in block
    assert "v1, n1 = fold_ladder(v1, n1, q1, si_z, si_x, du1)" in block
    assert "b_z" not in block and "a_y" not in block  # never a shifted operand
    assert "q2" not in block and "v2" not in block   # target 2 gets nothing
    assert "-(" not in block  # no unary minus: Triton lowers it as 0.0 - x


def test_the_complex_pass_is_a_quarter_turn_per_part_on_center_words(sk):
    block = _insert(_segment("beta_bloch_pml_curl_step"))
    assert "q0_re = beta_plus * b_im" in block
    assert "q0_im = (beta_plus * b_re) * -1.0" in block
    assert "q1_re = beta_minus * a_im" in block
    assert "q1_im = (beta_minus * a_re) * -1.0" in block
    for part in ("re", "im"):
        assert f"v0_{part}, n0_{part} = fold_ladder(v0_{part}, n0_{part}, q0_{part}" in block
    assert "b0_re" not in block and "a1_re" not in block  # never a connected copy
    assert "tl.math.fma" not in MODULE_PATH.read_text(encoding="utf-8")
    assert "_mul_imag_coefficient_left(" not in _segment("beta_bloch_pml_curl_step")


def test_the_increment_mask_is_the_curls_own(sk):
    """Each masked plane of the curl's ownership block for targets 0 and 1 masks the
    increment too, and no other plane does."""
    for kernel, suffix in (("beta_pml_curl_step", ""),
                           ("beta_bloch_pml_curl_step", "_re")):
        segment = _segment(kernel)
        block = _insert(segment)
        curl_mask = segment[segment.index("# --- ownership mask"):
                            segment.index("# --- split-field") if "# --- split-field" in segment
                            else segment.index("# --- the PML forms")]
        for target in ("0", "1"):
            curl_planes = sorted(set(
                line.split("(")[1].split(",")[0]
                for line in curl_mask.splitlines()
                if f"curl{target}{suffix} = tl.where(" in line))
            pass_planes = sorted(set(
                line.split("(")[1].split(",")[0]
                for line in block.splitlines()
                if f"q{target}{suffix} = tl.where(" in line))
            assert curl_planes and curl_planes == pass_planes, (kernel, target,
                                                                curl_planes, pass_planes)


def test_the_dtdx_exclusion_is_source_visible(sk):
    """The pass multiplies the coefficient by the CENTER, never by dtdx."""
    for kernel in ("beta_pml_curl_step", "beta_bloch_pml_curl_step"):
        assert "dtdx" not in _insert(_segment(kernel)), (
            f"{kernel}: the beta increment is an analytic derivative and must not be "
            f"scaled by dtdx")


def test_removing_the_pass_reproduces_the_root_bodies(sk):
    """Each kernel is its root body, character for character, plus the one insert."""
    import ast

    def body(path, name):
        text = path.read_text(encoding="utf-8")
        node = next(n for n in ast.walk(ast.parse(text))
                    if isinstance(n, ast.FunctionDef) and n.name == name)
        statements = node.body[1:]
        return "\n".join(ast.get_source_segment(text, s) for s in statements)

    for kernel, root_path, root in (
            ("beta_pml_curl_step", PACKAGE_DIR / "kernels.py", "pml_curl_step"),
            ("beta_bloch_pml_curl_step", PACKAGE_DIR / "complex_fields.py",
             "bloch_pml_curl_step")):
        text = MODULE_PATH.read_text(encoding="utf-8")
        node = next(n for n in ast.walk(ast.parse(text))
                    if isinstance(n, ast.FunctionDef) and n.name == kernel)
        kept = [ast.get_source_segment(text, s) for s in node.body[1:]
                if not (isinstance(s, ast.If) and getattr(s.test, "id", "") == "HAS_BETA")]
        assert "\n".join(kept) == body(root_path, root), kernel


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
        assert "policy_stamp" in text, path
